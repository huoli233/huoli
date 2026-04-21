# ruff: noqa: F401
import asyncio
import math
import random
import re
import time
from dataclasses import dataclass, field, is_dataclass, replace as dataclass_replace
import datetime
from collections import Counter, defaultdict, deque
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from src.chat.utils.timer_calculator import Timer
from src.chat.replyer.context_block_builder import build_reply_context_block
from src.chat.heart_flow.heartFC_chat import HeartFChatting
from src.chat.heart_flow.energy_manager import EnergyChainDimension
from src.chat.heart_flow.heartfc_thresholds import get_heartfc_thresholds
from src.core.watch_state_machine import WatchLevel
from src.core.group_pattern_detector import GroupPattern
from src.core.group_scene_state import AtmosphereType
from src.core.unified_planner import PlanningDecision, ActionType
from src.config.config import global_config
from src.common.logger import get_logger
from src.chat.heart_flow.frequency_control import frequency_control_manager
from src.plugin_system.apis import database_api, message_api, send_api
from src.chat.utils.utils import is_bot_self
from src.common.data_models.heartflow_models import FlowPhase, UnifiedFlowSnapshot
from src.chat.heart_flow.enhanced_modules.shared_runtime import (
    logger,
    _rng,
    _rt_float,
    _parallel_stage_timeout,
    _llm_upgrade_timeout,
    _TICK_FLOOR_SEC,
    _DORMANT_POLL_SEC,
    _PERCEPTION_COOLDOWN_SEC,
    _VOICE_COOLDOWN_SEC,
    _ENERGY_DRAIN_FLOOR,
    _POST_MESSAGE_RETRY_SEC,
    _WATCH_LEVEL_BY_RANK,
    BehaviorGovernorVerdict,
    RestGovernorVerdict,
    ModelGovernorVerdict,
)

if TYPE_CHECKING:
    from src.chat.heart_flow.llm_autonomous_planner import (
        AutonomousDecision,
        EnvironmentSnapshot,
    )
    from src.chat.proactive.proactive_decider import ProactiveDecision

class LoopPhaseTransitionMixin:
    def _query_flow_phase(self) -> str:
        """查询当前频道的心流阶段，从世界快照读取"""
        snap = getattr(self, "_tick_world_snapshot", None)
        if snap is not None and snap.subject.flow_phase:
            return snap.subject.flow_phase
        # 快照尚未构建（首轮tick），默认待命而非活跃，避免刚启动就进入强参与
        return "standby"

    def _evaluate_phase_gate(self, phase_str: str, now: float) -> str:
        """根据心流阶段决定本轮循环的行为

        返回值：
        - "proceed"  继续完整流水线
        - "skip"     跳过本轮（休息期不响应）
        - "glance"   休息期窥屏
        """
        if phase_str == "dormant":
            return self._handle_dormant_gate()
        if phase_str == "pending":
            return self._handle_pending_gate(now)
        if phase_str == "standby":
            # 待命阶段正常进入流水线，但迁移至 ENGAGED
            _snap = self._build_unified_flow_snapshot("phase_gate")
            _snap.watch_state = WatchLevel.ACTIVE_WATCH.value
            _snap.phase = FlowPhase.ENGAGED.value
            self._apply_unified_flow_snapshot(
                _snap,
                watch_reason="phase_gate:message_arrival",
                phase_reason="待命→活跃：消息到达",
            )
            return "proceed"
        return "proceed"

    def _handle_dormant_gate(self) -> str:
        """休息期门控：委托 DormancySupervisor 决策"""
        try:
            from src.chat.heart_flow.rest_handler import (
                get_dormancy_supervisor,
            )
            from src.common.data_models.heartflow_models import DormantReaction

            supervisor = get_dormancy_supervisor()
            rest_snapshot = supervisor.snapshot(self.stream_id)
            reaction = supervisor.on_incoming_msg(self.stream_id)
            if reaction == DormantReaction.AWAKEN:
                try:
                    from src.core.night_cycle_system import get_night_cycle

                    _ncs_reset = get_night_cycle(self.stream_id)
                    _ncs_reset.reset_night_counters()
                    _d6 = EnergyChainDimension.get_instance()
                    _d6s = _d6._ensure_channel(self.stream_id)
                    _d6s.night_reply_count = 0
                    _grace = 120.0 if _d6._get_night_mode(self.stream_id).phase == "DEEP_VALLEY" else 60.0
                    self._wake_grace_until = time.time() + _grace
                    logger.info(
                        f"{self.log_prefix} 😴 动态唤醒宽限期={_grace:.0f}s 阶段={_d6._get_night_mode(self.stream_id).phase}"
                    )
                except Exception:
                    self._wake_grace_until = time.time() + 120.0
                self._apply_attention_transition(
                    target_watch=WatchLevel.ACTIVE_WATCH,
                    watch_reason="休息结束被唤醒",
                    target_phase=FlowPhase.ENGAGED,
                    phase_reason="休息结束被唤醒",
                )
                self._last_flow_blocker = "休息期结束/反悔唤醒"
                return "proceed"
            if reaction == DormantReaction.GLANCE:
                self._last_flow_blocker = f"休息中仅窥屏 remaining={rest_snapshot.get('remaining_sec', 0)}s"
                return "glance"
            self._last_flow_blocker = f"休息中跳过 remaining={rest_snapshot.get('remaining_sec', 0)}s"
            return "skip"
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 休息门控异常，放行: {exc}")
            return "proceed"

    def _estimate_dynamic_rest_profile(self, voice_conclusion) -> Dict[str, float]:
        """根据内心独白、情感状态和资源状态估算休息画像。"""
        desire = int(getattr(voice_conclusion, "reply_desire_level", 5) or 5) if voice_conclusion else 5
        mood_text = str(getattr(voice_conclusion, "current_mood", "") or "").lower() if voice_conclusion else ""
        thought_text = str(getattr(voice_conclusion, "thinking", "") or "") if voice_conclusion else ""
        emotion = getattr(self, "_cached_emotion_state", {}) or {}
        boredom = float(emotion.get("boredom", 0.0) or 0.0)
        fatigue = float(emotion.get("environmental_fatigue", 0.0) or 0.0)
        loneliness = float(emotion.get("loneliness", 0.0) or 0.0)
        snap = getattr(self, "_tick_world_snapshot", None)
        energy_ratio = 0.6
        thinking_ratio = 0.6
        if snap is not None:
            try:
                res = snap.self_resources
                energy_ratio = float((res.chat_ratio() + res.thinking_ratio()) / 2.0)
                thinking_ratio = float(res.thinking_ratio())
            except Exception as exc:
                logger.debug(f"{self.log_prefix} 读取自我资源比例异常: {exc}")
                energy_ratio = 0.5
                thinking_ratio = 0.5
        rest_intensity = 0.35 + max(0.0, (6 - desire)) * 0.09 + max(0.0, 0.55 - energy_ratio) * 0.4
        rest_intensity += max(0.0, 0.5 - thinking_ratio) * 0.35 + fatigue * 0.35
        if any(tag in mood_text for tag in ("tired", "疲", "累", "烦", "倦")):
            rest_intensity += 0.12
        wake_drive = boredom * 0.45 + loneliness * 0.35 + max(0.0, desire - 3) * 0.04
        if "等会" in thought_text or "一会" in thought_text or "稍后" in thought_text:
            wake_drive += 0.08
        regret_chance = boredom * 0.28 + loneliness * 0.22 + max(0.0, desire - 4) * 0.03 - fatigue * 0.18
        return {
            "rest_intensity": max(0.15, min(1.0, rest_intensity)),
            "wake_drive": max(0.0, min(1.0, wake_drive)),
            "regret_chance": max(0.0, min(0.8, regret_chance)),
            "min_rest_sec": 45 if desire >= 4 else 90,
        }

    def _enter_dynamic_rest(self, *, cause: str, trigger: str, blackout: bool = False) -> None:
        """统一进入动态休息，并同步观看状态。"""
        try:
            from src.chat.heart_flow.rest_handler import DormancySupervisor

            rest_profile = self._estimate_dynamic_rest_profile(getattr(self, "_cached_voice", None))
            DormancySupervisor.instance().begin_dynamic_dormancy(
                self.stream_id,
                cause=cause,
                rest_intensity=rest_profile["rest_intensity"],
                wake_drive=rest_profile["wake_drive"],
                regret_chance=rest_profile["regret_chance"],
                min_rest_sec=int(rest_profile["min_rest_sec"]),
            )
            self._last_flow_blocker = (
                f"{trigger}: 动态休息 rest={rest_profile['rest_intensity']:.2f} "
                f"wake={rest_profile['wake_drive']:.2f} regret={rest_profile['regret_chance']:.2f}"
            )
        except Exception as _rest_err:
            logger.debug(f"{self.log_prefix} 动态休息触发异常: {_rest_err}")
        self._apply_attention_transition(
            target_watch=WatchLevel.BLACKOUT if blackout else WatchLevel.PEEK,
            watch_reason=trigger,
            target_phase=FlowPhase.DORMANT,
            phase_reason=cause,
        )


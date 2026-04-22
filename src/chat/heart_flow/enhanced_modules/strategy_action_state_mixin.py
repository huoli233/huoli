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

class StrategyActionStateMixin:
    @staticmethod
    def _metric_ratio(value: Any, default: float = 0.0) -> float:
        try:
            numeric = float(value if value is not None else default)
        except Exception:
            numeric = float(default or 0.0)
        if numeric > 1.0:
            numeric /= 100.0
        return max(0.0, min(1.0, numeric))

    def _decide_action(
        self,
        pinged_msg,
        eagerness: float,
        awareness,
        voice,
        messages: List,
    ) -> bool:
        """融合多维信号决定本轮是否执行行动规划

        决策优先级：
        1. 模型 should_reply → 直接执行/跳过（模型自主决策最高优先）
        2. 觉察引擎参与度 > 阈值 → 高概率行动
        3. 能量意愿 × 频率因子 → 概率行动
        """
        # 模型自主决策优先
        if voice is not None and hasattr(voice, "should_reply"):
            _model_reply = voice.should_reply
            if _model_reply is True:
                return True
            if _model_reply is False:
                return False
        # 被@但模型没明确说时，不强制，交给后续概率决策
        # 内心独白强驱动（仅作参考，不再强制）
        if pinged_msg is not None:
            pass
        # 觉察参与度驱动
        engagement_boost = 0.0
        if awareness is not None and hasattr(awareness, "engagement_pull"):
            engagement_boost = awareness.engagement_pull
            if engagement_boost >= 0.7:
                return True
        dashboard_verdict = {}
        if hasattr(self, "_get_dashboard_verdict"):
            try:
                dashboard_verdict = self._get_dashboard_verdict() or {}
            except Exception as exc:
                logger.debug(f"{self.log_prefix} 仪表盘裁定读取失败: {exc}")
        dashboard_reply = bool(dashboard_verdict.get("reply", False))
        dashboard_urgency = str(dashboard_verdict.get("reply_urgency", "") or "")
        dashboard_confidence = float(dashboard_verdict.get("confidence", 0.0) or 0.0)
        relation_view = {}
        if hasattr(self, "_resolve_relation_view"):
            try:
                relation_view = self._resolve_relation_view() or {}
            except Exception as exc:
                logger.debug(f"{self.log_prefix} 关系快照读取失败: {exc}")
        annoyance_value = float(relation_view.get("annoyance_value", 0.0) or 0.0)
        pressure_value = float(relation_view.get("psychological_pressure", 0.0) or 0.0)
        affection_value = float(relation_view.get("affection", 0.0) or 0.0)
        trust_value = float(relation_view.get("trust_value", relation_view.get("trust_score", 0.0)) or 0.0)
        emotion_state = getattr(self, "_cached_emotion_state", {}) or {}
        boredom = self._metric_ratio(emotion_state.get("boredom", 0.0), 0.0)
        loneliness = self._metric_ratio(emotion_state.get("loneliness", 0.0), 0.0)
        social_desire = self._metric_ratio(emotion_state.get("social_desire", 0.0), 0.0)
        night_phase = str(getattr(getattr(self, "_cached_night_phase", None), "value", "") or "").strip().lower()
        if not night_phase:
            night_phase = str(getattr(getattr(self, "_cached_night_phase", None), "name", "") or "").strip().lower()
        if pinged_msg is None:
            if night_phase in {"deep_sleep", "deep_valley"}:
                return False
            if night_phase in {"burned_out", "burnthrough"}:
                return False
            if not dashboard_reply and dashboard_confidence >= 0.72 and dashboard_urgency in {"跳过", "不回复"}:
                return False
            if annoyance_value >= 70.0 or pressure_value >= 60.0:
                if affection_value < 45.0 and trust_value < 25.0:
                    return False
        if dashboard_reply and dashboard_confidence >= 0.82 and dashboard_urgency in {"立即回复", "尽快回复"}:
            return True
        # 概率决策：基础意愿 × 频率调节 × 觉察加成
        freq_adjust = frequency_control_manager.get_or_create_frequency_control(
            self.stream_id
        ).get_talk_frequency_adjust()
        base_talk = global_config.chat.get_talk_value(self.stream_id)
        dynamic_prob = float(getattr(self, "_reply_probability", 0.5) or 0.5)
        combined_prob = base_talk * freq_adjust * eagerness * dynamic_prob
        _night_sup = getattr(self, "_night_reply_suppression", 0.0)
        if _night_sup > 0:
            combined_prob *= 1.0 - _night_sup
        # 觉察加成：参与度在 0.3~0.7 区间提供线性加成
        if 0.3 <= engagement_boost < 0.7:
            combined_prob *= 1.0 + engagement_boost
        # 内心独白加成：欲望等级 4~6 提供小幅加成
        if voice is not None and hasattr(voice, "reply_desire_level"):
            desire = voice.reply_desire_level
            if 4 <= desire < 7:
                combined_prob *= 1.0 + desire * 0.05
        if dashboard_confidence > 0:
            dashboard_prob = {
                "立即回复": 0.92,
                "尽快回复": 0.76,
                "可稍后回": 0.55,
                "跳过": 0.18,
                "不回复": 0.05,
            }.get(dashboard_urgency, 0.5 if dashboard_reply else 0.25)
            combined_prob = combined_prob * 0.55 + dashboard_prob * 0.45
        emotional_pull = max(boredom, loneliness, social_desire)
        if emotional_pull >= 0.65 and annoyance_value < 55.0 and pressure_value < 45.0:
            combined_prob += 0.12
        elif emotional_pull >= 0.40 and annoyance_value < 65.0:
            combined_prob += 0.06
        if annoyance_value >= 50.0:
            combined_prob *= max(0.18, 1.0 - min(0.55, (annoyance_value - 50.0) / 100.0))
        if pressure_value >= 40.0:
            combined_prob *= max(0.22, 1.0 - min(0.45, (pressure_value - 40.0) / 100.0))
        if affection_value >= 35.0 or trust_value >= 35.0:
            combined_prob += 0.06
        chatterbox_penalty = float(getattr(self, "_chatterbox_penalty", 0.0) or 0.0)
        if chatterbox_penalty > 0:
            combined_prob *= max(0.12, 1.0 - min(0.75, chatterbox_penalty * 0.18))
        if pinged_msg is None and getattr(self, "_pattern_forced_observe", False):
            combined_prob *= 0.35
        if pinged_msg is None and getattr(self, "_defense_mode_active", False):
            combined_prob *= 0.65
        combined_prob = max(0.01, min(0.99, combined_prob))
        return random.random() < combined_prob

    def _shift_to_dormant(self, cause: str = "") -> None:
        """迁移至休息阶段"""
        try:
            snapshot = self._build_unified_flow_snapshot("shift_dormant")
            if snapshot.phase == FlowPhase.DORMANT.value and snapshot.is_resting:
                return
            self._apply_unified_flow_snapshot(
                self._copy_unified_flow_snapshot(
                    snapshot,
                    update={
                        "watch_state": WatchLevel.BLACKOUT.value,
                        "phase": FlowPhase.DORMANT.value,
                        "blocker": cause or "进入休息",
                    }
                ),
                watch_reason="shift_to_dormant",
                phase_reason=cause or "进入休息",
            )
            self._run_flow_side_effects(action_name="begin_rest", reason=cause)
            logger.info(f"{self.log_prefix} 迁移至休息: {cause}")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 迁移休息失败: {exc}")

    def _shift_to_engaged(self, cause: str = "") -> None:
        """迁移至活跃阶段"""
        try:
            snapshot = self._build_unified_flow_snapshot("shift_engaged")
            current = snapshot.phase
            if current == FlowPhase.ENGAGED.value:
                return
            if current in (FlowPhase.DORMANT.value, FlowPhase.LIGHT_REST.value):
                try:
                    _d6 = EnergyChainDimension.get_instance()
                    _nm = _d6._get_night_mode(self.stream_id)
                    _grace = 180.0
                    if _nm and _nm.phase in ("DAWN_RECOVERY", "ACTIVE_TWILIGHT"):
                        _grace = 120.0
                    elif _nm and _nm.phase == "DEEP_VALLEY":
                        _grace = 300.0
                    self._wake_grace_until = time.time() + _grace
                except Exception:
                    self._wake_grace_until = time.time() + 120.0
            self._apply_unified_flow_snapshot(
                self._copy_unified_flow_snapshot(
                    snapshot,
                    update={
                        "watch_state": WatchLevel.ENGAGED.value,
                        "phase": FlowPhase.ENGAGED.value,
                        "blocker": "",
                    }
                ),
                watch_reason="shift_to_engaged",
                phase_reason=cause or "进入参与",
            )
            if current in (FlowPhase.DORMANT.value, FlowPhase.LIGHT_REST.value):
                self._run_flow_side_effects(action_name="engage", conclude_rest=True)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 迁移活跃失败: {exc}")

    def _shift_to_pending(self, cause: str = "", anticipated: str = "") -> None:
        """迁移至等待阶段"""
        try:
            snapshot = self._build_unified_flow_snapshot("shift_pending")
            self._apply_unified_flow_snapshot(
                self._copy_unified_flow_snapshot(
                    snapshot,
                    update={
                        "watch_state": WatchLevel.ACTIVE_WATCH.value,
                        "phase": FlowPhase.PENDING.value,
                        "blocker": cause or "进入等待",
                    }
                ),
                watch_reason="shift_to_pending",
                phase_reason=cause or "进入等待",
            )
            if anticipated:
                self._run_flow_side_effects(action_name="begin_wait", anticipated=anticipated)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 迁移等待失败: {exc}")

    def _update_dynamic_ratio(self, now_ts: float):
        """动态更新回复比例和概率

        核心算法：
        1. 计算当前累计比例（用户消息 vs 机器人回复）
        2. 根据比例偏差调整目标回复概率
        3. 刷屏时自动降低回复概率
        4. 沉默时保持适度回复概率

        比例控制原则：
        - 用户发5条 → 机器人回复1-2条（目标比例40%）
        - 用户发20条 → 机器人回复8-10条（目标比例40-50%）
        - 刷屏时 → 机器人回复更少（20-30%）
        - 沉默时 → 机器人可多说（50-60%）
        """
        actual_user_msgs = len(self._user_msg_timeline)
        actual_bot_replies = len(self._bot_reply_timeline)

        if actual_bot_replies == 0:
            self._cumulative_ratio = 0.5
        else:
            self._cumulative_ratio = actual_bot_replies / max(1, actual_user_msgs)

        recent_user_msgs = sum(1 for t in self._user_msg_timeline if now_ts - t < 300.0)
        recent_bot_replies = sum(1 for t in self._bot_reply_timeline if now_ts - t < 300.0)

        if recent_bot_replies == 0:
            recent_ratio = 0.5
        else:
            recent_ratio = recent_bot_replies / max(1, recent_user_msgs)

        self._target_ratio = 0.25

        if self._is_in_burst:
            burst_intensity = min(1.0, self._burst_user_count / 10.0)
            self._target_ratio = 0.25 - (burst_intensity * 0.15)
            self._target_ratio = max(0.15, self._target_ratio)
        elif recent_user_msgs < 3 and recent_bot_replies < 2:
            self._target_ratio = 0.25
        elif recent_user_msgs > 20 and recent_bot_replies > 10:
            self._target_ratio = 0.2

        ratio_diff = self._cumulative_ratio - self._target_ratio
        if ratio_diff > 0.15:
            self._reply_probability = max(0.05, 0.35 - ratio_diff * 2.2)
        elif ratio_diff < -0.15:
            self._reply_probability = min(0.55, 0.35 - ratio_diff * 1.1)
        else:
            self._reply_probability = 0.35

        if self._is_in_burst:
            burst_penalty = min(0.3, (self._burst_user_count - 3) * 0.05)
            self._reply_probability = max(0.05, self._reply_probability - burst_penalty)

        if now_ts - self._last_bot_reply_ts < 10.0:
            self._reply_probability = max(0.05, self._reply_probability - 0.20)

        self._phase_confidence = 0.5 + (recent_ratio * 0.3)

        logger.debug(
            f"{self.log_prefix} 📊 动态比例: 用户{actual_user_msgs}条/机器人{actual_bot_replies}条 "
            f"实际比例={self._cumulative_ratio:.2f} 目标比例={self._target_ratio:.2f} "
            f"回复概率={self._reply_probability:.2f} "
            f"刷屏={self._is_in_burst}({self._burst_user_count}条)"
        )

    def _record_message_behavior(self, messages: List) -> None:
        try:
            from src.chat.heart_flow.behavior_analyzer import (
                get_behavior_analyzer,
            )

            analyzer = get_behavior_analyzer()
            for msg in messages:
                uid = getattr(msg, "user_id", "") or ""
                if not uid:
                    continue
                text = getattr(msg, "processed_plain_text", "") or ""
                polarity = self._estimate_text_polarity(text)
                analyzer.record(uid, polarity)
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _estimate_text_polarity(self, text: str) -> float:
        """极简文本极性估计：仅感叹号轻微正面，问号不参与负面判定"""
        content = str(text or "").strip()
        if not content:
            return 0.0
        # 问号属于正常提问行为，不应作为负面信号
        punct_positive = content.count("!") + content.count("！")
        length_bias = min(len(content) / 120.0, 1.0) * 0.05
        score = (punct_positive * 0.05) + length_bias
        return max(-0.3, min(0.3, score))

    def _record_interaction_mood(self, messages: List) -> None:
        """将交互记录到氛围追踪器"""
        try:
            from src.chat.heart_flow.emotion_stream import (
                get_channel_mood_tracker,
            )

            tracker = get_channel_mood_tracker()
            for msg in messages:
                text = getattr(msg, "processed_plain_text", "") or ""
                uid = getattr(msg, "user_id", "") or ""
                if text and uid:
                    tracker.record_inquiry(
                        channel_id=self.stream_id,
                        content=text,
                        author_id=uid,
                    )
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _log_energy_status(self, eagerness: float, reason: str) -> None:
        """输出能量状态日志（带间隔控制，避免刷屏）"""
        now = time.time()
        if now - self._last_energy_log_ts < self._energy_log_interval:
            return
        self._last_energy_log_ts = now
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            chat_energy = float(_ch.chat_pool) if _ch else 100.0
            chat_ceiling = float(_ch.chat_ceiling) if _ch else 100.0
            thinking_energy = float(_ch.thinking_value) if _ch else 100.0
            thinking_ceiling = float(_ch.thinking_ceiling) if _ch else 100.0
            freq_ctrl = frequency_control_manager.get_or_create_frequency_control(self.stream_id)
            freq_adjust = freq_ctrl.get_talk_frequency_adjust()
            logger.info(
                f"{self.log_prefix} 能量状态 | "
                f"聊天池={chat_energy:.1f}/{chat_ceiling:.0f} "
                f"思考池={thinking_energy:.1f}/{thinking_ceiling:.0f} "
                f"意愿={eagerness:.2f} "
                f"频率={freq_adjust:.2f}"
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 能量状态输出失败: {exc}")

    def _apply_plan_drain(self) -> None:
        """规划消耗思考值（动态计算）"""
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            base_cost = 3.0
            brain_cost = max(1.0, min(15.0, base_cost))
            if _ch:
                _d6.apply_thinking_cost(
                    self.stream_id,
                    think_cost=brain_cost,
                    source="plan_drain",
                )
                self._sync_runtime_resource_cache_from_d6()
                _state = _d6._ensure_channel(self.stream_id)
                logger.info(f"{self.log_prefix} 消耗思考值 {brain_cost:.1f}，剩余思考值 {_state.thinking_value:.1f}")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 规划消耗失败: {exc}")

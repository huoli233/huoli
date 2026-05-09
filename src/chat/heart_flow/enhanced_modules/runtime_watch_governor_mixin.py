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

class RuntimeWatchGovernorMixin:
    def _apply_governor_watch_transition(
        self,
        *,
        current_watch: str,
        incoming_batch: List,
        has_ping: bool,
    ) -> Optional[str]:
        """在旧 watch 表之前先执行一层统一姿态跳转。

        这层只处理高置信度的“该休息 / 该退回窥屏 / 该从窥屏升到关注”，
        避免后面的手搓条件继续和 BehaviorGovernor / RestGovernor 打架。
        """
        direct_target = self._has_targeted_bot_message(incoming_batch)
        rest_verdict = self._evaluate_rest_governor(
            incoming_batch=incoming_batch,
            source="reactive",
            targeted_to_bot=direct_target,
            admin_force=bool(getattr(self, "_is_admin_forced", False)),
            pinged_msg=(incoming_batch[-1] if has_ping and incoming_batch else None),
        )
        behavior_verdict = self._evaluate_behavior_governor(
            incoming_batch=incoming_batch,
            silence_sec=float(getattr(self, "_last_silence_sec", 0.0) or 0.0),
            requested_mode="reactive",
            target_message=self._get_latest_human_message(incoming_batch),
            is_background=False,
        )

        target_watch: Optional[WatchLevel] = None
        target_phase: Optional[FlowPhase] = None
        watch_reason = ""
        phase_reason = ""

        if rest_verdict.should_rest and not direct_target and not has_ping:
            if current_watch in {"engaged", "active_watch"}:
                target_watch = WatchLevel.BLACKOUT
                target_phase = FlowPhase.DORMANT
            elif current_watch in {"skim_window", "peek"}:
                target_watch = WatchLevel.PEEK
                target_phase = FlowPhase.STANDBY
            if target_watch is not None:
                watch_reason = f"watch:rest_governor:{rest_verdict.posture}"
                phase_reason = f"RestGovernor收口:{self._summarize_rest_governor(rest_verdict)}"
        elif rest_verdict.interruption_policy == "peek_only" and not direct_target and not has_ping:
            if current_watch in {"engaged", "active_watch", "skim_window"}:
                target_watch = WatchLevel.PEEK
                target_phase = FlowPhase.STANDBY
                watch_reason = f"watch:rest_governor:peek_only:{rest_verdict.posture}"
                phase_reason = f"RestGovernor退回窥屏:{self._summarize_rest_governor(rest_verdict)}"
        elif behavior_verdict.interrupt_level == "engage":
            if current_watch == "blackout":
                target_watch = WatchLevel.PEEK
                target_phase = FlowPhase.STANDBY
                watch_reason = f"watch:behavior_governor:blackout_to_peek:{behavior_verdict.reply_mode}"
                phase_reason = f"BehaviorGovernor拉起窥屏:{self._summarize_behavior_governor(behavior_verdict)}"
            elif current_watch in {"peek", "skim_window"}:
                target_watch = WatchLevel.ACTIVE_WATCH
                target_phase = FlowPhase.STANDBY
                watch_reason = f"watch:behavior_governor:to_active:{behavior_verdict.reply_mode}"
                phase_reason = f"BehaviorGovernor拉起关注:{self._summarize_behavior_governor(behavior_verdict)}"
            elif current_watch == "active_watch" and behavior_verdict.reply_mode == "reply":
                target_watch = WatchLevel.ENGAGED
                target_phase = FlowPhase.ENGAGED
                watch_reason = f"watch:behavior_governor:to_engaged:{behavior_verdict.reply_mode}"
                phase_reason = f"BehaviorGovernor拉起参与:{self._summarize_behavior_governor(behavior_verdict)}"

        if target_watch is None:
            return None

        try:
            snapshot = self._build_unified_flow_snapshot("watch_governor")
            snapshot.watch_state = target_watch.value
            snapshot.phase = target_phase.value if target_phase is not None else snapshot.phase
            if target_watch == WatchLevel.BLACKOUT:
                snapshot.blocker = phase_reason
            self._apply_unified_flow_snapshot(
                snapshot,
                watch_reason=watch_reason,
                phase_reason=phase_reason,
            )
            return f"{current_watch}->{target_watch.value}({watch_reason})"
        except Exception as exc:
            logger.debug(f"{self.log_prefix} governor观看态跳转异常: {exc}")
            return None

    def _compute_governor_watch_cap(
        self,
        *,
        behavior_verdict: BehaviorGovernorVerdict,
        rest_verdict: RestGovernorVerdict,
        direct_target: bool,
        has_ping: bool,
        recent_human_activity: bool,
    ) -> Tuple[int, str]:
        """给旧 watch 手工表加一层 Governor 限幅，避免后半段继续越级升级。"""
        _behavior_cap = int(getattr(behavior_verdict, "max_watch_rank", 1) or 1)
        if rest_verdict.should_rest and not direct_target and not has_ping:
            if rest_verdict.interruption_policy == "block":
                return (0, f"RestGovernor:block:{self._summarize_rest_governor(rest_verdict)}")
            return (1, f"RestGovernor:rest:{self._summarize_rest_governor(rest_verdict)}")
        if rest_verdict.interruption_policy == "peek_only" and not direct_target and not has_ping:
            return (1, f"RestGovernor:peek_only:{self._summarize_rest_governor(rest_verdict)}")
        if not recent_human_activity and behavior_verdict.silence_policy == "silent":
            _behavior_cap = min(_behavior_cap, 1)
        if behavior_verdict.interrupt_level == "ignore":
            if behavior_verdict.silence_policy == "silent" and not recent_human_activity:
                return (0, f"BehaviorGovernor:ignore_silent:{self._summarize_behavior_governor(behavior_verdict)}")
        _behavior_cap = max(0, min(4, _behavior_cap))
        return (_behavior_cap, f"BehaviorGovernor:cap={_behavior_cap}:{self._summarize_behavior_governor(behavior_verdict)}")

    def _apply_governor_watch_cap(
        self,
        *,
        current_watch: str,
        cap_rank: int,
        reason: str,
    ) -> Optional[str]:
        current_enum = next(
            (level for level in WatchLevel if level.value == current_watch),
            WatchLevel.PEEK,
        )
        if current_enum.rank() <= cap_rank:
            return None

        target_watch = _WATCH_LEVEL_BY_RANK.get(int(cap_rank), WatchLevel.PEEK)
        target_phase = FlowPhase.ENGAGED if target_watch == WatchLevel.ENGAGED else (
            FlowPhase.DORMANT if target_watch == WatchLevel.BLACKOUT else FlowPhase.STANDBY
        )
        try:
            snapshot = self._build_unified_flow_snapshot("watch_governor_cap")
            snapshot.watch_state = target_watch.value
            snapshot.phase = target_phase.value
            if target_watch == WatchLevel.BLACKOUT:
                snapshot.blocker = reason
            self._apply_unified_flow_snapshot(
                snapshot,
                watch_reason=f"watch:governor_cap:{current_watch}_to_{target_watch.value}",
                phase_reason=f"Governor限幅:{reason}",
            )
            return f"{current_watch}->{target_watch.value}(governor_cap:{cap_rank})"
        except Exception as exc:
            logger.debug(f"{self.log_prefix} governor观看态限幅异常: {exc}")
            return None

    def _should_promote_peek_reflection(
        self,
        *,
        incoming_batch: List,
        peek_verdict: Any,
        pinged_msg: Any = None,
    ) -> Tuple[bool, str]:
        """统一裁决“窥屏反思后要不要真的升级参与”。"""
        if peek_verdict is None or not getattr(peek_verdict, "is_valid", False):
            return (False, "peek_invalid")

        direct_target = bool(pinged_msg is not None or self._has_targeted_bot_message(incoming_batch))
        rest_verdict = self._evaluate_rest_governor(
            incoming_batch=incoming_batch,
            source="peek",
            targeted_to_bot=direct_target,
            admin_force=bool(getattr(self, "_is_admin_forced", False)),
            pinged_msg=pinged_msg,
        )
        behavior_verdict = self._evaluate_behavior_governor(
            incoming_batch=incoming_batch,
            silence_sec=float(getattr(self, "_last_silence_sec", 0.0) or 0.0),
            requested_mode="reactive",
            target_message=self._get_latest_human_message(incoming_batch),
            is_background=False,
        )

        desire = int(float(getattr(peek_verdict, "reply_desire_level", 0) or 0))
        action = str(getattr(peek_verdict, "next_action", "") or "").strip().lower()
        explicit_upgrade = action in {"reply", "followup"}
        explicit_hold = action in {"observe", "wait", "skip", "rest", "defer"}
        behavior_summary = self._summarize_behavior_governor(behavior_verdict)
        rest_summary = self._summarize_rest_governor(rest_verdict)

        if rest_verdict.should_rest and rest_verdict.interruption_policy == "block" and not direct_target:
            return (False, f"RestGovernor阻断: {rest_summary}")
        if not behavior_verdict.allow_generation and not direct_target:
            return (False, f"BehaviorGovernor阻断生成: {behavior_summary}")
        if explicit_hold:
            return (False, f"模型选择{action}: {behavior_summary}")
        if direct_target and (explicit_upgrade or desire >= 4):
            return (True, f"direct_target: {behavior_summary}")
        if behavior_verdict.interrupt_level == "engage" and (explicit_upgrade or desire >= 6):
            return (True, f"engage: {behavior_summary}")
        if (
            explicit_upgrade
            and desire >= 8
            and behavior_verdict.interrupt_level == "skim"
            and rest_verdict.interruption_policy != "block"
        ):
            return (True, f"high_interest_override: {behavior_summary}")
        if (
            not action
            and desire >= 8
            and behavior_verdict.interrupt_level in {"skim", "engage"}
            and rest_verdict.interruption_policy == "allow"
        ):
            return (True, f"gateway_high_desire: {behavior_summary}")
        return (False, f"peek_governor_hold: {behavior_summary} | {rest_summary}")

    def _integrate_watch_state(self, now: float, has_messages: bool = False) -> None:
        """更新关注层级状态机"""
        try:
            _wm = self._orch.get("watch_machine")
            if has_messages:
                _wm.signal_escalate(trigger="新消息到达")
            else:
                _boredom = 0.0
                _silence = 0.0
                _snap = self._tick_world_snapshot
                if _snap:
                    _res = getattr(_snap, "self_resources", None)
                    if _res:
                        _boredom = float(getattr(_res, "boredom", 0.0) or 0.0)
                        _silence = float(getattr(_res, "silence_seconds", 0.0) or 0.0)
                _wm.evaluate_natural_decay(silence_sec=_silence, boredom=_boredom)
            self._cached_watch_level = _wm.current_level
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 关注状态机异常: {exc}")

    def _evaluate_watch_full_transitions(self, now: float, has_ping: bool, incoming_batch: List) -> Optional[str]:
        """GAP-F：观看状态机完整转移表 —— 8条转移路径的统一评估

        5态: blackout / peek / skim_window / active_watch / engaged
        8条转移路径:
          ① blackout → peek      (@提及/无聊↑/熟悉用户/异常热闹/高相关事件)
          ② peek → skim_window    (发现有趣话题/连续多条相关/看到线索/看到熟悉对象)
          ③ skim → active_watch   (显著度↑/线程强相关/多人持续互动/小模型判断值得)
          ④ active → engaged       (决定参与/线程成主线/事件影响强)
          ⑤ engaged → active      (参与后退回观察/线程无推进/先看别人接)
          ⑥ [任意] → blackout     (活跃值↓/持续低价值/过于嘈杂/疲劳↑/明确不想看)
          ⑦ engaged → blackout    (直接从卷入退到完全不看)
          ⑧ skim → peek           (看了没意思退回窥屏)

        返回转移描述字符串，或 None 表示无转移
        """
        try:
            thresholds = get_heartfc_thresholds()
            from src.core.watch_state_machine import (
                get_watch_machine,
                WatchLevel as WatchState,
            )

            _wm = get_watch_machine(self.stream_id)
            _current = _wm.current_level
            if _current is None:
                return None
            _cur_val = _current.value if hasattr(_current, "value") else str(_current)
            _human_batch_count = sum(1 for _m in (incoming_batch or []) if self._is_human_message_obj(_m))
            _governor_transition = self._apply_governor_watch_transition(
                current_watch=_cur_val,
                incoming_batch=incoming_batch,
                has_ping=has_ping,
            )
            if _governor_transition:
                return _governor_transition
            _direct_target = self._has_targeted_bot_message(incoming_batch)
            _recent_human_activity = _human_batch_count > 0 or self._has_recent_human_activity(300.0)
            _rest_governor = self._evaluate_rest_governor(
                incoming_batch=incoming_batch,
                source="reactive",
                targeted_to_bot=_direct_target,
                admin_force=bool(getattr(self, "_is_admin_forced", False)),
                pinged_msg=(incoming_batch[-1] if has_ping and incoming_batch else None),
            )
            _behavior_governor = self._evaluate_behavior_governor(
                incoming_batch=incoming_batch,
                silence_sec=float(getattr(self, "_last_silence_sec", 0.0) or 0.0),
                requested_mode="reactive",
                target_message=self._get_latest_human_message(incoming_batch),
                is_background=False,
            )
            _governor_cap_rank, _governor_cap_reason = self._compute_governor_watch_cap(
                behavior_verdict=_behavior_governor,
                rest_verdict=_rest_governor,
                direct_target=_direct_target,
                has_ping=has_ping,
                recent_human_activity=_recent_human_activity,
            )
            _governor_cap_transition = self._apply_governor_watch_cap(
                current_watch=_cur_val,
                cap_rank=_governor_cap_rank,
                reason=_governor_cap_reason,
            )
            if _governor_cap_transition:
                return _governor_cap_transition
            # ── 收集转移决策所需的全部上下文 ──
            _boredom = 0.0
            _env_fatigue = 0.0
            _fatigue = 0.0
            _activity = 50.0
            _silence = 0.0
            _snap = self._tick_world_snapshot
            if _snap:
                _res = getattr(_snap, "self_resources", None)
                if _res:
                    _boredom = float(getattr(_res, "boredom", 0.0) or 0.0)
                    _activity = float(getattr(_res, "activity_level", 50.0) or 50.0)
                    _silence = float(getattr(_res, "silence_seconds", 0.0) or 0.0)
            _emo = getattr(self, "_cached_emotion_state", None) or {}
            if isinstance(_emo, dict):
                _env_fatigue = float(_emo.get("environmental_fatigue", 0.0) or 0.0) * 100.0
            try:
                from src.core.night_cycle_system import get_night_cycle

                _ncs_state = get_night_cycle(self.stream_id).state_snapshot
                _fatigue = float(getattr(_ncs_state, "daily_fatigue", 0.0) or 0.0)
            except Exception as _exc:
                logger.debug(f"{self.log_prefix} 观看态疲劳采样失败: {_exc}")
            # 关系与好感
            _rel = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
            _aff = float(_rel.get("affection", 0.0) or 0.0)
            # 存在态
            _sw = 0.5
            _aw = 0.5
            _avoid = 0.0
            if self._cached_presence_state:
                _sw = float(getattr(self._cached_presence_state, "social_willingness", 0.5) or 0.5)
                _aw = float(getattr(self._cached_presence_state, "watch_willingness", 0.5) or 0.5)
                _avoid = float(getattr(self._cached_presence_state, "avoidance_tendency", 0.0) or 0.0)
            # 打断权分级（GAP-E）
            _tier3_count = 0
            _tier2_count = 0
            _interruption = getattr(self, "_cached_interruption_tiers", None)
            if isinstance(_interruption, dict):
                _tier3_count = sum(1 for s in _interruption.values() if s >= 0.65)
                _tier2_count = sum(1 for s in _interruption.values() if 0.30 <= s < 0.65)
            # 自我引用强度
            _ref_max = 0.0
            _discuss_n = 0
            if self._cached_self_references:
                for _r in self._cached_self_references:
                    _rs = float(getattr(_r, "strength", 0.0) or 0.0)
                    if _rs > _ref_max:
                        _ref_max = _rs
                    _rt = getattr(_r, "ref_type", None)
                    if _rt:
                        _rv = _rt.value if hasattr(_rt, "value") else str(_rt)
                        if _rv == "discussed_as_topic":
                            _discuss_n += 1
            # 群体模式
            _is_heated = False
            _is_conflict = False
            if self._cached_pattern_evidence:
                for _pe in self._cached_pattern_evidence:
                    _pc = float(getattr(_pe, "confidence", 0.0) or 0.0)
                    if _pc < 0.35:
                        continue
                    _pp = getattr(_pe, "pattern", None)
                    if _pp:
                        _pv = _pp.value if hasattr(_pp, "value") else str(_pp)
                        if _pv == "heated_discussion":
                            _is_heated = True
                        elif _pv in ("conflict_escalation", "argument"):
                            _is_conflict = True
            # 活跃用户数
            _hot_users = 0
            if self._cached_participant_summary:
                _hot_users = int(self._cached_participant_summary.get("hot_count", 0) or 0)
                _hot_users += int(self._cached_participant_summary.get("warm_count", 0) or 0)
            # ══ 转移路径评估 ══
            # 路径⑥：[任意] → blackout（通用退出条件，优先级最高）
            _should_blackout = False
            _blackout_reason = ""
            if _activity < 12 and _env_fatigue > 55 and _silence > 1800:
                _should_blackout = True
                _blackout_reason = "低活跃+高环境疲劳+长静默"
            elif _fatigue > 75 and _silence > 600:
                _should_blackout = True
                _blackout_reason = "高疲劳+静默"
            elif _avoid > 0.7 and _sw < 0.15:
                _should_blackout = True
                _blackout_reason = "高回避+低社交欲"
            elif _hot_users >= 8 and self._metric_has_signal(_aff) and _aff < 20:
                _should_blackout = True
                _blackout_reason = "过于嘈杂+低好感"
            if _should_blackout and _cur_val not in ("blackout",):
                try:
                    _snap = self._build_unified_flow_snapshot("watch_transition")
                    _snap.watch_state = WatchState.BLACKOUT.value
                    _snap.phase = FlowPhase.DORMANT.value
                    _snap.blocker = _blackout_reason
                    self._apply_unified_flow_snapshot(
                        _snap,
                        watch_reason=f"watch:auto_blackout:{_blackout_reason}",
                        phase_reason=f"观看退场: {_blackout_reason}",
                    )
                    return f"{_cur_val}→blackout({_blackout_reason})"
                except Exception as _e:
                    logger.warning(f"{self.log_prefix} 强制进入blackout失败: {_e}")
            # 路径①：blackout → peek
            if _cur_val == "blackout":
                _peek_trigger_strength = 0.0
                if has_ping:
                    _peek_trigger_strength += 0.55
                if _boredom > thresholds.boredom_peek_trigger:
                    _peek_trigger_strength += min(0.25, (_boredom - thresholds.boredom_peek_trigger) / 100.0)
                if _aff > 55:
                    _peek_trigger_strength += 0.10
                if _tier3_count > 0:
                    _peek_trigger_strength += 0.18 * min(1.0, _tier3_count / 3.0)
                if _is_heated:
                    _peek_trigger_strength += 0.08
                if _discuss_n >= 2:
                    _peek_trigger_strength += 0.15
                if _hot_users >= 4 and _hot_users <= 8:
                    _peek_trigger_strength += 0.06
                if _peek_trigger_strength > 0.42:
                    try:
                        _snap = self._build_unified_flow_snapshot("watch_transition")
                        _snap.watch_state = WatchState.PEEK.value
                        _snap.phase = FlowPhase.STANDBY.value
                        self._apply_unified_flow_snapshot(
                            _snap,
                            watch_reason=f"watch:blackout_to_peek:{_peek_trigger_strength:.2f}",
                            phase_reason="黑屏后恢复窥屏",
                        )
                        return f"blackout→peek(trigger={_peek_trigger_strength:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} blackout→peek异常: {_e}")
            # 路径②：peek → skim_window
            if _cur_val == "peek" and _governor_cap_rank >= WatchState.SKIM_WINDOW.rank():
                _skim_trigger = 0.0
                if _tier3_count > 0:
                    _skim_trigger += 0.22 * min(1.0, _tier3_count)
                if _tier2_count >= 2:
                    _skim_trigger += 0.12
                if _ref_max > 0.35:
                    _skim_trigger += 0.15
                if _is_heated:
                    _skim_trigger += 0.10
                if _human_batch_count >= 3:
                    _skim_trigger += 0.08
                if _boredom > thresholds.boredom_scan_trigger:
                    _skim_trigger += 0.06
                if _skim_trigger > 0.38:
                    try:
                        _snap = self._build_unified_flow_snapshot("watch_transition")
                        _snap.watch_state = WatchState.SKIM_WINDOW.value
                        _snap.phase = FlowPhase.STANDBY.value
                        self._apply_unified_flow_snapshot(
                            _snap,
                            watch_reason=f"watch:peek_to_skim:{_skim_trigger:.2f}",
                            phase_reason="窥屏升级为扫视",
                        )
                        return f"peek→skim_window(trigger={_skim_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} peek→skim_window异常: {_e}")
            # 路径③：skim → active_watch
            if _cur_val == "skim_window" and _governor_cap_rank >= WatchState.ACTIVE_WATCH.rank():
                _active_trigger = 0.0
                if _tier3_count >= 1:
                    _active_trigger += 0.28
                if _discuss_n >= 1 and _ref_max > 0.45:
                    _active_trigger += 0.18
                if has_ping:
                    _active_trigger += 0.22
                if _aw > 0.55:
                    _active_trigger += 0.12
                if _is_conflict and _aff > 35:
                    _active_trigger += 0.14
                if _active_trigger > 0.40:
                    try:
                        _snap = self._build_unified_flow_snapshot("watch_transition")
                        _snap.watch_state = WatchState.ACTIVE_WATCH.value
                        _snap.phase = FlowPhase.STANDBY.value
                        self._apply_unified_flow_snapshot(
                            _snap,
                            watch_reason=f"watch:skim_to_active:{_active_trigger:.2f}",
                            phase_reason="扫视升级为积极关注",
                        )
                        return f"skim→active_watch(trigger={_active_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} skim→active_watch异常: {_e}")
            # 路径④：active → engaged
            if _cur_val == "active_watch" and _governor_cap_rank >= WatchState.ENGAGED.rank():
                _engage_trigger = 0.0
                if has_ping:
                    _engage_trigger += 0.32
                if _tier3_count >= 1:
                    _engage_trigger += 0.24
                if _discuss_n >= 2:
                    _engage_trigger += 0.16
                if _sw > 0.6 and _aw > 0.55:
                    _engage_trigger += 0.14
                if _is_heated or _is_conflict:
                    _engage_trigger += 0.10
                if _engage_trigger > 0.48:
                    try:
                        _snap = self._build_unified_flow_snapshot("watch_transition")
                        _snap.watch_state = WatchState.ENGAGED.value
                        _snap.phase = FlowPhase.ENGAGED.value
                        self._apply_unified_flow_snapshot(
                            _snap,
                            watch_reason=f"watch:active_to_engaged:{_engage_trigger:.2f}",
                            phase_reason="积极关注升级为参与",
                        )
                        return f"active→engaged(trigger={_engage_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} active→engaged异常: {_e}")
            # 路径⑤：engaged → active（退回观察）
            if _cur_val == "engaged":
                _disengage_trigger = 0.0
                if _silence > 120 and _tier3_count == 0:
                    _disengage_trigger += 0.25
                if _avoid > 0.45:
                    _disengage_trigger += 0.20
                if _fatigue > 55:
                    _disengage_trigger += 0.15
                if _human_batch_count <= 1 and not has_ping:
                    _disengage_trigger += 0.12
                if _disengage_trigger > 0.48:
                    try:
                        _snap = self._build_unified_flow_snapshot("watch_transition")
                        _snap.watch_state = WatchState.ACTIVE_WATCH.value
                        _snap.phase = FlowPhase.STANDBY.value
                        self._apply_unified_flow_snapshot(
                            _snap,
                            watch_reason=f"watch:engaged_to_active:{_disengage_trigger:.2f}",
                            phase_reason="参与后退回积极关注",
                        )
                        return f"engaged→active(disengage={_disengage_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} engaged→active异常: {_e}")
            # 路径⑦：engaged → blackout（直接退到不看）
            if _cur_val == "engaged":
                _crash_trigger = 0.0
                if _fatigue > 80:
                    _crash_trigger += 0.35
                if _avoid > 0.75:
                    _crash_trigger += 0.30
                if _silence > 600 and _tier3_count == 0:
                    _crash_trigger += 0.20
                if _crash_trigger > 0.60:
                    try:
                        _snap = self._build_unified_flow_snapshot("watch_transition")
                        _snap.watch_state = WatchState.BLACKOUT.value
                        _snap.phase = FlowPhase.DORMANT.value
                        _snap.blocker = f"engaged_crash={_crash_trigger:.2f}"
                        self._apply_unified_flow_snapshot(
                            _snap,
                            watch_reason=f"watch:engaged_to_blackout:{_crash_trigger:.2f}",
                            phase_reason="参与态崩落到休息",
                        )
                        return f"engaged→blackout(crash={_crash_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} engaged→blackout异常: {_e}")
            # 路径⑧：skim → peek（退回窥屏）
            if _cur_val == "skim_window":
                _fallback_trigger = 0.0
                if _tier3_count == 0 and _tier2_count <= 1:
                    _fallback_trigger += 0.22
                if _silence > 60:
                    _fallback_trigger += 0.15
                if _boredom < thresholds.boredom_downgrade_trigger:
                    _fallback_trigger += 0.12
                if _ref_max < 0.15 and not _is_heated:
                    _fallback_trigger += 0.18
                if _fallback_trigger > 0.45:
                    try:
                        _snap = self._build_unified_flow_snapshot("watch_transition")
                        _snap.watch_state = WatchState.PEEK.value
                        _snap.phase = FlowPhase.STANDBY.value
                        self._apply_unified_flow_snapshot(
                            _snap,
                            watch_reason=f"watch:skim_to_peek:{_fallback_trigger:.2f}",
                            phase_reason="扫视退回窥屏",
                        )
                        return f"skim→peek(fallback={_fallback_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} skim→peek异常: {_e}")
            return None
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 观看状态转移评估异常: {exc}")
            return None


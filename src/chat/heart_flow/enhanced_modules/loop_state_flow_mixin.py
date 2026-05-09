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

class LoopStateFlowMixin:
    def _apply_attention_transition(
        self,
        *,
        target_watch=None,
        watch_reason: str = "",
        target_phase: Optional[FlowPhase] = None,
        phase_reason: str = "",
    ) -> None:
        """统一推进观看态与阶段态，避免多处各自改状态。"""
        if target_watch is not None:
            try:
                _wm = self._orch.get("watch_machine")
                _wm.signal_force_level(target_watch, trigger=watch_reason)
                self._cached_watch_level = _wm.current_level
            except Exception as exc:
                logger.debug(f"{self.log_prefix} 关注状态推进失败({watch_reason or target_watch}): {exc}")
        if target_phase is not None:
            try:
                from src.chat.heart_flow.state_machine import get_phase_coordinator

                coordinator = get_phase_coordinator()
                coordinator.register_channel(self.stream_id)
                shifted = coordinator.attempt_shift(
                    self.stream_id,
                    target_phase,
                    reason=phase_reason or watch_reason or "统一状态推进",
                )
                if not shifted and coordinator.current_phase(self.stream_id) != target_phase:
                    logger.debug(
                        f"{self.log_prefix} 阶段推进未生效 {coordinator.current_phase(self.stream_id).value}->{target_phase.value}"
                    )
            except Exception as exc:
                logger.debug(f"{self.log_prefix} 阶段推进失败({phase_reason or target_phase}): {exc}")

    def _align_states_with_inner_voice(self, voice_conclusion, *, source: str = "") -> None:
        """根据内心独白统一同步观看态与阶段态。"""
        if voice_conclusion is None or not getattr(voice_conclusion, "is_valid", False):
            return
        action = str(getattr(voice_conclusion, "next_action", "") or "").strip().lower()
        desire = int(getattr(voice_conclusion, "reply_desire_level", 5) or 5)
        if action in {"rest", "disengage"}:
            return
        if action == "lurk":
            self._apply_attention_transition(
                target_watch=WatchLevel.PEEK,
                watch_reason=f"{source or 'inner_voice'}:潜水",
                target_phase=FlowPhase.STANDBY,
                phase_reason="内心选择潜水，退回待命",
            )
            return
        if getattr(voice_conclusion, "should_reply", None) is True and desire >= 6:
            self._apply_attention_transition(
                target_watch=WatchLevel.ENGAGED,
                watch_reason=f"{source or 'inner_voice'}:想参与",
                target_phase=FlowPhase.ENGAGED,
                phase_reason="内心明确想参与",
            )
            return
        if getattr(voice_conclusion, "should_reply", None) is False and desire <= 4:
            self._apply_attention_transition(
                target_watch=WatchLevel.PEEK,
                watch_reason=f"{source or 'inner_voice'}:先观察",
                target_phase=FlowPhase.STANDBY,
                phase_reason="内心倾向观察",
            )

    def _build_unified_flow_snapshot(self, final_decision: str = "") -> UnifiedFlowSnapshot:
        """聚合当前行为层核心状态，作为统一快照。"""
        voice = getattr(self, "_cached_voice", None)
        emotion = getattr(self, "_cached_emotion_state", {}) or {}
        watch_level = getattr(self, "_cached_watch_level", None)
        watch_text = watch_level.value if hasattr(watch_level, "value") else str(watch_level or "unknown")
        phase_text = FlowPhase.STANDBY.value
        try:
            from src.chat.heart_flow.state_machine import get_phase_coordinator

            coordinator = get_phase_coordinator()
            coordinator.register_channel(self.stream_id)
            phase_text = coordinator.current_phase(self.stream_id).value
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 阶段协调器注册异常: {exc}")
        rest_snapshot = {"resting": False}
        try:
            from src.chat.heart_flow.rest_handler import get_dormancy_supervisor

            rest_snapshot = get_dormancy_supervisor().snapshot(self.stream_id)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 休眠监督器快照异常: {exc}")
        snapshot = UnifiedFlowSnapshot(
            channel_id=self.stream_id,
            phase=phase_text,
            watch_state=watch_text,
            is_resting=bool(rest_snapshot.get("resting", False)),
            rest_remaining_sec=float(rest_snapshot.get("remaining_sec", 0.0) or 0.0),
            rest_intensity=float(rest_snapshot.get("rest_intensity", 0.0) or 0.0),
            wake_drive=float(rest_snapshot.get("wake_drive", 0.0) or 0.0),
            reply_desire=int(getattr(voice, "reply_desire_level", 5) or 5) if voice else 5,
            voice_action=str(getattr(voice, "next_action", "") or "") if voice else "",
            voice_should_reply=getattr(voice, "should_reply", None) if voice else None,
            voice_mood=str(getattr(voice, "current_mood", "") or "") if voice else "",
            voice_thinking=str(getattr(voice, "thinking", "") or "")[:120] if voice else "",
            boredom=float(emotion.get("boredom", 0.0) or 0.0),
            environmental_fatigue=float(emotion.get("environmental_fatigue", 0.0) or 0.0),
            loneliness=float(emotion.get("loneliness", 0.0) or 0.0),
            social_desire=float(emotion.get("social_desire", 0.0) or 0.0),
            blocker=self._last_flow_blocker,
            final_decision=final_decision,
        )
        self._last_unified_flow_snapshot = snapshot
        return snapshot

    def _copy_unified_flow_snapshot(
        self,
        snapshot: UnifiedFlowSnapshot,
        *,
        update: Optional[Dict[str, Any]] = None,
    ) -> UnifiedFlowSnapshot:
        """兼容 dataclass / Pydantic 风格的统一快照复制更新。"""
        update = dict(update or {})
        try:
            model_copy = getattr(snapshot, "model_copy", None)
            if callable(model_copy):
                return model_copy(update=update)
            copy_method = getattr(snapshot, "copy", None)
            if callable(copy_method):
                try:
                    return copy_method(update=update)
                except TypeError:
                    return copy_method(**update)
            if is_dataclass(snapshot):
                return dataclass_replace(snapshot, **update)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 统一快照复制失败，回退重建: {exc}")
        payload = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(getattr(snapshot, "__dict__", {}))
        payload.update(update)
        return UnifiedFlowSnapshot.from_dict(payload)

    def _apply_unified_flow_snapshot(
        self,
        snapshot: UnifiedFlowSnapshot,
        *,
        watch_reason: str = "",
        phase_reason: str = "",
    ) -> None:
        """通过统一快照单入口回写观看态与阶段态。"""
        watch_target = None
        watch_map = {
            "blackout": WatchLevel.BLACKOUT,
            "peek": WatchLevel.PEEK,
            "skim_window": WatchLevel.SKIM_WINDOW,
            "active_watch": WatchLevel.ACTIVE_WATCH,
            "engaged": WatchLevel.ENGAGED,
        }
        phase_target = None
        try:
            if snapshot.watch_state in watch_map:
                watch_target = watch_map[snapshot.watch_state]
            phase_target = FlowPhase(snapshot.phase)
        except Exception:
            phase_target = None
        self._apply_attention_transition(
            target_watch=watch_target,
            watch_reason=watch_reason or snapshot.final_decision or "统一快照回写",
            target_phase=phase_target,
            phase_reason=phase_reason or snapshot.blocker or snapshot.final_decision or "统一快照回写",
        )

    def _run_flow_side_effects(
        self,
        *,
        action_name: str,
        reason: str = "",
        anticipated: str = "",
        conclude_rest: bool = False,
    ) -> None:
        """统一执行状态回写后的副作用管理。"""
        try:
            if action_name == "begin_rest":
                from src.chat.heart_flow.rest_handler import get_dormancy_supervisor

                get_dormancy_supervisor().begin_dormancy(self.stream_id, cause=reason)
                return
            if action_name == "end_rest":
                from src.chat.heart_flow.rest_handler import get_dormancy_supervisor

                get_dormancy_supervisor().conclude_dormancy(self.stream_id)
                return
            if action_name == "begin_wait" and anticipated:
                from src.chat.heart_flow.waiting_handler import get_pending_orchestrator

                get_pending_orchestrator().initiate_watch(self.stream_id, anticipated=anticipated)
                return
            if action_name == "pending_timeout":
                from src.chat.heart_flow.waiting_handler import get_pending_orchestrator

                get_pending_orchestrator().conclude_with_timeout_consequence(self.stream_id)
                return
            if action_name == "engage" and conclude_rest:
                from src.chat.heart_flow.rest_handler import get_dormancy_supervisor

                get_dormancy_supervisor().conclude_dormancy(self.stream_id)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 状态副作用执行失败({action_name}): {exc}")

    def _run_post_action_side_effects(
        self,
        *,
        did_reply: bool,
        incoming_batch: List,
        planner_decision: Optional[Any] = None,
        reply_text: str = "",
        was_proactive: bool = False,
        inaction_reason: str = "",
    ) -> None:
        """统一执行回复后或跳过后的外围副作用。"""
        try:
            from src.chat.proactive.intention_pool import get_intention_pool

            _pool = get_intention_pool()
            _matched_intent_id = (
                str(getattr(planner_decision, "matched_intent_id", "") or "").strip() if planner_decision else ""
            )
            if did_reply:
                _completed_precise = False
                if _matched_intent_id:
                    _completed_precise = bool(_pool.complete_intent(_matched_intent_id))
                    if _completed_precise:
                        logger.info(f"{self.log_prefix} 意图精确结案 id={_matched_intent_id[:10]}")
                if not _completed_precise:
                    _target = (
                        str(getattr(planner_decision, "target_user_id", "") or "").strip()
                        or str(getattr(self, "_last_user_id", "") or "").strip()
                    )
                    if _target:
                        _count = _pool.complete_by_target(self.stream_id, _target, limit=1)
                        if _count > 0:
                            logger.info(f"{self.log_prefix} 意图回退结案 target={_target[:8]} count={_count}")
            elif _matched_intent_id:
                _reason = (
                    str(getattr(planner_decision, "reason", "") or "").strip() or inaction_reason or "planner_no_reply"
                )
                if _pool.fail_intent(_matched_intent_id, reason=_reason[:120]):
                    logger.info(
                        f"{self.log_prefix} 意图精确失败回写 id={_matched_intent_id[:10]} reason={_reason[:30]}"
                    )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 意图副作用执行失败: {exc}")

        try:
            if not did_reply:
                self._integrate_adaptive_collection(incoming_batch, False, "")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 自适应收集副作用失败: {exc}")

        if not did_reply:
            try:
                _was_at_bot = any(getattr(_m, "is_at_bot", False) for _m in incoming_batch if hasattr(_m, "is_at_bot"))
                if _was_at_bot:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker

                    _emo_t = get_emotion_tracker(self.stream_id)
                    for _m in reversed(incoming_batch):
                        _uid = str(getattr(_m, "user_id", "") or "").strip()
                        if self._is_human_message_obj(_m):
                            _emo_t.update_psychological_pressure(_uid, 2.0, "被@未回复")
                            break
            except Exception as exc:
                logger.debug(f"{self.log_prefix} 未回复压力副作用失败: {exc}")

    def _apply_planner_state_update(self, action_name: str, *, reason: str = "", anticipated: str = "") -> None:
        """将规划器的非回复动作统一映射到状态快照。"""
        snapshot = self._build_unified_flow_snapshot(action_name)
        if action_name == "wait":
            self._apply_unified_flow_snapshot(
                self._copy_unified_flow_snapshot(
                    snapshot,
                    update={
                        "watch_state": WatchLevel.ACTIVE_WATCH.value,
                        "phase": FlowPhase.PENDING.value,
                        "blocker": reason or "规划器决定等待",
                    }
                ),
                watch_reason="planner:wait",
                phase_reason=reason or "规划器决定等待",
            )
            self._run_flow_side_effects(action_name="begin_wait", anticipated=anticipated)
            return
        if action_name in {"rest", "check_later", "observe", "no_reply"}:
            self._apply_unified_flow_snapshot(
                self._copy_unified_flow_snapshot(
                    snapshot,
                    update={
                        "watch_state": WatchLevel.PEEK.value,
                        "phase": FlowPhase.STANDBY.value,
                        "blocker": reason or f"规划器决定{action_name}",
                    }
                ),
                watch_reason=f"planner:{action_name}",
                phase_reason=reason or f"规划器决定{action_name}",
            )

    def _apply_post_reply_state(self, *, did_reply: bool, reason: str = "") -> None:
        """回复或跳过后统一回写行为状态。"""
        snapshot = self._build_unified_flow_snapshot("reply_completed" if did_reply else "reply_skipped")
        if did_reply:
            snapshot = self._copy_unified_flow_snapshot(
                snapshot,
                update={
                    "watch_state": WatchLevel.ENGAGED.value,
                    "phase": FlowPhase.ENGAGED.value,
                    "blocker": "",
                    "final_decision": "reply",
                }
            )
            self._apply_unified_flow_snapshot(
                snapshot,
                watch_reason="post_reply:engaged",
                phase_reason=reason or "完成回复后维持参与",
            )
            return
        snapshot = self._copy_unified_flow_snapshot(
            snapshot,
            update={
                "watch_state": WatchLevel.PEEK.value if not snapshot.is_resting else snapshot.watch_state,
                "phase": FlowPhase.STANDBY.value if not snapshot.is_resting else snapshot.phase,
                "blocker": reason or snapshot.blocker,
                "final_decision": "skip",
            }
        )
        self._apply_unified_flow_snapshot(
            snapshot,
            watch_reason="post_reply:skip",
            phase_reason=reason or "本轮未回复",
        )

    def _emit_flow_decision_summary(self, stage: str, final_decision: str) -> None:
        """统一输出心流决策摘要，便于排查谁在拦截。"""
        try:
            self._update_decision_trace(final_decision=final_decision)
            if stage == "final_decision" and not (self._last_decision_trace.get("winner_layer") or ""):
                if str(getattr(self, "_last_gateway_gate", "") or "") == "block" and final_decision != "reply":
                    self._mark_decision_winner("gateway_block")
                elif str(getattr(self, "_last_gateway_gate", "") or "") == "force_reply" and final_decision == "reply":
                    self._mark_decision_winner("gateway_force")
                elif final_decision == "reply":
                    self._mark_decision_winner("algo_reply")
                else:
                    self._mark_decision_winner("final_skip")
            snapshot = self._build_unified_flow_snapshot(final_decision)
            summary = {
                "stage": stage,
                "decision": final_decision,
                "voice_action": snapshot.voice_action,
                "voice_desire": snapshot.reply_desire,
                "voice_should_reply": snapshot.voice_should_reply,
                "voice_mood": snapshot.voice_mood,
                "voice_thinking": snapshot.voice_thinking,
                "boredom": snapshot.boredom,
                "env_fatigue": snapshot.environmental_fatigue,
                "loneliness": snapshot.loneliness,
                "social_desire": snapshot.social_desire,
                "watch": snapshot.watch_state,
                "phase": snapshot.phase,
                "resting": snapshot.is_resting,
                "rest_remaining": snapshot.rest_remaining_sec,
                "blocker": snapshot.blocker,
                "decision_trace": dict(getattr(self, "_last_decision_trace", None) or {}),
            }
            self._last_flow_decision_summary = summary
            try:
                from src.webui.services.runtime_state_hub import emit_runtime_delta

                emit_runtime_delta(
                    self.stream_id,
                    module="flow_runtime",
                    path="domains.flow_runtime.last_flow_decision_summary",
                    value=summary,
                    reason=f"flow_summary:{stage}",
                )
            except Exception as hub_exc:
                logger.debug(f"{self.log_prefix} 运行态心流摘要推送失败: {hub_exc}")
            _trace_brief = self._decision_trace_brief()
            logger.info(
                f"{self.log_prefix} [心流摘要/{stage}] 决策={final_decision} | "
                f"独白(action={summary['voice_action'] or '-'} desire={summary['voice_desire']} "
                f"reply={summary['voice_should_reply']} mood={summary['voice_mood'] or '-'}) | "
                f"情感(bored={summary['boredom']:.2f} fatigue={summary['env_fatigue']:.2f} "
                f"lonely={summary['loneliness']:.2f} social={summary['social_desire']:.2f}) | "
                f"观看={summary['watch']} 阶段={summary['phase']} 休息={summary['resting']}({summary['rest_remaining']:.0f}s) | "
                f"阻断={summary['blocker'] or '-'} | 想法={summary['voice_thinking'] or '-'}"
                + (f" | trace={_trace_brief}" if _trace_brief else "")
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 心流摘要输出失败: {exc}")

    def _apply_inner_voice_priority(
        self,
        *,
        should_act: bool,
        voice_conclusion,
        pinged_msg,
        gateway_gate: str,
        planner_decision,
        is_admin_forced: bool,
    ) -> bool:
        """让内心独白成为默认优先信号，外层只在强约束下覆盖。"""
        if voice_conclusion is None or not getattr(voice_conclusion, "is_valid", False):
            return should_act
        voice_desire = int(getattr(voice_conclusion, "reply_desire_level", 5) or 5)
        voice_should_reply = getattr(voice_conclusion, "should_reply", None)
        voice_action = str(getattr(voice_conclusion, "next_action", "") or "").strip().lower()
        strong_force = bool(is_admin_forced or pinged_msg is not None or gateway_gate == "force_reply")
        if voice_action in {"rest", "disengage", "lurk", "observe", "wait"}:
            self._last_flow_blocker = f"主体意愿={voice_action}"
            return False if not strong_force else should_act
        if voice_should_reply is True and voice_desire >= 6:
            self._last_flow_blocker = ""
            return True
        if voice_should_reply is False and voice_desire <= 4 and not strong_force:
            self._last_flow_blocker = f"主体意愿拒绝 desire={voice_desire}"
            return False
        if planner_decision is None and voice_desire >= 7 and not should_act:
            self._last_flow_blocker = ""
            return True
        return should_act

    def _handle_pending_gate(self, now: float) -> str:
        """等待期门控：委托 PendingOrchestrator 决策"""
        try:
            from src.chat.heart_flow.waiting_handler import (
                get_pending_orchestrator,
            )
            from src.common.data_models.heartflow_models import IdleDecision

            orchestrator = get_pending_orchestrator()
            decision = orchestrator.evaluate_next(self.stream_id)
            if decision == IdleDecision.EXPIRED or decision == IdleDecision.WITHDRAW:
                # 超时退出时触发后果链回写
                self._run_flow_side_effects(action_name="pending_timeout")
                _snap = self._build_unified_flow_snapshot(decision.value)
                _snap.watch_state = WatchLevel.ACTIVE_WATCH.value
                _snap.phase = FlowPhase.ENGAGED.value
                _snap.blocker = f"等待结束:{decision.value}"
                self._apply_unified_flow_snapshot(
                    _snap,
                    watch_reason=f"pending_end:{decision.value}",
                    phase_reason=f"等待结束: {decision.value}",
                )
                return "proceed"
            if decision == IdleDecision.CONTEMPLATE:
                return "proceed"
            if decision == IdleDecision.CHASE:
                return "proceed"
            return "skip"
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 等待门控异常，放行: {exc}")
            return "proceed"

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

class EnhancedSceneAnalysisMixin:
    def _check_proactive_from_emotion(self) -> Tuple[bool, str]:
        """
        从情感状态判断是否应该主动行为

        这不是规则判断，而是"感受"判断：
        - 我感到无聊吗？
        - 我感到孤独吗？
        - 我想找人聊天吗？
        """
        try:
            from src.chat.heart_flow.emotion_driven_core import (
                get_emotion_driven_core,
            )

            emotion_core = get_emotion_driven_core()
            return emotion_core.should_proactive(self.stream_id)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 情感判断失败: {exc}")
            return False, "情感系统不可用"

    def _on_bot_sent(self, was_proactive: bool = False) -> None:
        """通知各系统：机器人发言了"""
        _now_sent = time.time()
        self._bot_reply_counter += 1
        self._bot_reply_timeline.append(_now_sent)
        if len(self._bot_reply_timeline) > 50:
            self._bot_reply_timeline = self._bot_reply_timeline[-50:]
        # 只有主动发言才累加未回应计数，被动回复不应污染主动退避
        if was_proactive:
            self._unanswered_bot_turns += 1
            self._proactive_reply_timeline.append(_now_sent)
            if len(self._proactive_reply_timeline) > 50:
                self._proactive_reply_timeline = self._proactive_reply_timeline[-50:]
            # 向能量系统报告被忽略状态，累积烦躁
            if self._unanswered_bot_turns >= 2:
                try:
                    _d6_ig = EnergyChainDimension.get_instance()

                    class _IgnoreEvt:
                        channel_id = self.stream_id
                        event_type = "bot_unanswered"
                        raw_extras = {"unanswered_turns": self._unanswered_bot_turns}

                    _d6_ig.on_event(_IgnoreEvt())
                except Exception:
                    pass
        else:
            self._reactive_reply_timeline.append(_now_sent)
            if len(self._reactive_reply_timeline) > 50:
                self._reactive_reply_timeline = self._reactive_reply_timeline[-50:]
        self._last_bot_reply_ts = _now_sent
        self._update_dynamic_ratio(_now_sent)
        self._legacy_constraint_hits = 0
        # F27：递增每小时回复计数
        self._roll_hourly_reply_window(_now_sent)
        self._hourly_reply_count += 1
        if was_proactive:
            self._hourly_proactive_reply_count += 1
        # F28：消耗睡眠回复配额（迁移至D6夜模式计数）
        try:
            from src.core.night_cycle_system import get_night_cycle

            _ncs_f28 = get_night_cycle(self.stream_id)
            if _ncs_f28.is_night_hours():
                _ncs_f28.record_night_reply()
                _ncs_f28.consume_sleep_reply()
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            if _ch:
                _ch.night_reply_count = _ncs_f28.state_snapshot.night_reply_count
                if hasattr(_d6, "reset_disturbance"):
                    _d6.reset_disturbance(self.stream_id)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        self._night_soft_wake_info = None
        try:
            from src.chat.heart_flow.emotion_driven_core import (
                get_emotion_driven_core,
            )

            emotion_core = get_emotion_driven_core()
            emotion_core.on_bot_message(self.stream_id)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.trauma_fabric import get_wound_network

            wound = get_wound_network()
            wound.absorb_stimulus(self.stream_id, "bot_reply", 0.5)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.modules.modcore.psychological_core import (
                get_psychological_core,
            )

            psycho = get_psychological_core()
            psycho.update_stamina(self.stream_id, "", 2.0, "reply")
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.modules.social_value.social_affect_fuser import (
                get_social_affect_fuser,
            )

            if self._last_user_id:
                get_social_affect_fuser().record_peer_interaction(self._last_user_id, self.stream_id, "bot", 0.5)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.heartflow import heartflow

            heartflow.note_bot_activity(self.stream_id, was_proactive=was_proactive)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # 通知整合中心：bot发言了
        try:
            from src.chat.proactive.proactive_integration_hub import get_proactive_integration_hub

            _uid_target = str(self._last_user_id or "")
            _bot_content = ""
            _latest_bot = self._latest_recent_bot_utterance()
            if _latest_bot:
                _bot_content = str(_latest_bot.get("text", "") or "").strip()[:160]
            get_proactive_integration_hub().on_bot_message_sent(
                self.stream_id,
                user_id=_uid_target,
                was_proactive=was_proactive,
                content=_bot_content,
            )
        except Exception as _e:
            logger.debug(f"{self.log_prefix} 整合中心发言通知异常: {_e}")
        # 扫描主动决策账本中超期未回复的主动事件（标记为 ignored）
        try:
            from src.chat.proactive.proactive_decider import get_proactive_decider

            _swept = get_proactive_decider().ledger.sweep_expired()
            if _swept > 0:
                logger.debug(f"{self.log_prefix} [闭环] 扫描到 {_swept} 个超期主动事件")
        except Exception as _e:
            logger.debug(f"{self.log_prefix} 过期事件扫描异常: {_e}")
        # 主动行为反馈登记：开启观察窗口等待用户回复
        if was_proactive:
            try:
                from src.chat.proactive.proactive_decider import (
                    get_proactive_decider,
                )

                _proactive_intent_id = str(getattr(self, "_last_proactive_intent_id", "") or "")
                get_proactive_decider().ledger.register_proactive_fire(
                    self.stream_id,
                    source_intent_id=_proactive_intent_id,
                )
                if _proactive_intent_id:
                    logger.info(f"{self.log_prefix} [闭环追踪] 主动行为已登记 intent_id={_proactive_intent_id[:10]}")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        # GAP-N：新鲜度引擎——发言后标记相关消息为"看见并关注"
        try:
            from src.core.freshness_decay_engine import (
                get_freshness_decay_engine,
                VisibilityHistoryEffect,
            )

            if not self._freshness_engine_initialized:
                self._freshness_engine_initialized = True
            _fde = get_freshness_decay_engine(self.stream_id)
            if self._cached_deep_visibility_results:
                for _dvr in self._cached_deep_visibility_results[-5:]:
                    _mid = getattr(_dvr, "message_id", "")
                    if _mid and hasattr(_fde, "update_visibility_history"):
                        _fde.update_visibility_history(_mid, VisibilityHistoryEffect.SEEN_AND_NOTICED)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # GAP-P：自适应更新管道——收集发言事件
        try:
            from src.core.adaptive_update_pipeline import get_adaptive_pipeline

            _ap = get_adaptive_pipeline()
            _ap.collect_event(
                event_type="bot_reply",
                raw_content=f"channel={self.stream_id[:16]}, proactive={was_proactive}",
                source_channel=self.stream_id,
                metadata={"hourly_count": self._hourly_reply_count},
                priority=2,
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # 主动发言后启动"等待-反思-追问"后台任务
        if was_proactive:
            try:
                _proac_text = ""
                _latest_bot = self._latest_recent_bot_utterance()
                if _latest_bot:
                    _proac_text = str(_latest_bot.get("text", "") or "")[:120]
                self._spawn(
                    self._post_proactive_wait_reflect(
                        proactive_text=_proac_text,
                        intent_id=str(getattr(self, "_last_proactive_intent_id", "") or ""),
                    )
                )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} 等待-反思任务启动异常: {_e}")

    async def _post_proactive_wait_reflect(
        self,
        proactive_text: str = "",
        intent_id: str = "",
    ) -> None:
        """主动发言后的等待-反思-追问循环"""
        _reflect_depth = getattr(self, "_proactive_reflect_depth", 0) + 1
        self._proactive_reflect_depth = _reflect_depth
        try:
            if _reflect_depth > 2:
                logger.debug(f"{self.log_prefix} 反思递归深度={_reflect_depth}，终止追问")
                return
            _channel = self.stream_id
            _unanswered_at_start = self._unanswered_bot_turns
            _target_user = str(
                getattr(self, "_last_proactive_target_user_id", "")
                or getattr(self, "_last_user_id", "")
                or ""
            ).strip()
            # 步骤1：创建 WAIT_FOR_REPLY 意图
            try:
                from src.chat.proactive.intention_pool import (
                    get_intention_pool,
                    IntentKind,
                )

                _pool = get_intention_pool()
                _wait_intent = _pool.submit(
                    kind=IntentKind.WAIT_FOR_REPLY,
                    target_user=_target_user,
                    channel_id=_channel,
                    source="post_proactive_wait",
                    description=f"等待对主动消息的回复: {proactive_text[:60]}",
                    urgency=0.5,
                    context_snippet=proactive_text[:200],
                    expected_outcome="对方回复或互动",
                )
                logger.debug(f"{self.log_prefix} [等待-反思] 等待意图已创建 {_wait_intent.intent_id[:8]}")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} [等待-反思] 意图创建异常: {_e}")
            # 步骤2：提升注意力到 ACTIVE_WATCH
            try:
                _wm = self._orch.get("watch_machine")
                _wm.signal_escalate(trigger="主动发言后进入关注态")
                logger.debug(f"{self.log_prefix} [等待-反思] 注意力提升至 {_wm.current_level}")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} [等待-反思] 注意力提升异常: {_e}")
            # 步骤3：等待90秒（分段检查，避免用户已回复后仍在无意义等待）
            _wait_total = 90.0
            _check_interval = 15.0
            _elapsed = 0.0
            while _elapsed < _wait_total:
                await asyncio.sleep(_check_interval)
                _elapsed += _check_interval
                # 用户已回复：未回应计数被重置
                if self._unanswered_bot_turns < _unanswered_at_start:
                    logger.debug(f"{self.log_prefix} [等待-反思] 用户已回复，提前退出等待")
                    return
            # 步骤4：仍然无回复，启动自主反思
            if self._unanswered_bot_turns < _unanswered_at_start:
                return
            logger.info(f"{self.log_prefix} [等待-反思] 90秒无回复，启动自主反思")
            _reflect_verdict = None
            try:
                from src.chat.heart_flow.inner_voice import get_self_dialogue_engine

                _engine = get_self_dialogue_engine(_channel)
                _trigger = f"主动说了「{proactive_text[:40]}」后90秒无人回应，反思是否该追问"
                _reflect_verdict = await _engine.autonomous_reflection(
                    channel_id=_channel,
                    trigger_reason=_trigger,
                )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} [等待-反思] 反思引擎异常: {_e}")
                return
            if _reflect_verdict is None or not getattr(_reflect_verdict, "is_valid", False):
                logger.debug(f"{self.log_prefix} [等待-反思] 反思结果无效，放弃")
                self._deescalate_watch_after_proactive()
                return
            _reflect_desire = int(getattr(_reflect_verdict, "reply_desire_level", 3) or 3)
            _reflect_action = str(getattr(_reflect_verdict, "next_action", "observe") or "observe")
            _reflect_thought = str(getattr(_reflect_verdict, "thinking", "") or "")
            logger.info(
                f"{self.log_prefix} [等待-反思] 反思结论: desire={_reflect_desire} "
                f"action={_reflect_action} thought={_reflect_thought[:50]}"
            )
            # 步骤5：根据反思结论决策
            if _reflect_action in ("reply", "nudge") or _reflect_desire >= 7:
                _reflect_silence = max(
                    0.0,
                    time.time() - float(getattr(self, "_last_user_msg_time", 0.0) or 0.0),
                )
                _followup_governor = self._evaluate_behavior_governor(
                    incoming_batch=[],
                    silence_sec=_reflect_silence,
                    requested_mode="proactive",
                    is_background=True,
                )
                if not _followup_governor.allow_generation:
                    logger.info(
                        f"{self.log_prefix} [等待-反思] 行为Governor阻断追问: "
                        f"{self._summarize_behavior_governor(_followup_governor)}"
                    )
                    self._deescalate_watch_after_proactive()
                    return
                # 追问路径：创建 FOLLOWUP_QUESTION 意图，交给主动决策器
                try:
                    from src.chat.proactive.intention_pool import (
                        get_intention_pool,
                        IntentKind,
                    )

                    _pool = get_intention_pool()
                    _followup_intent = _pool.submit(
                        kind=IntentKind.FOLLOWUP_QUESTION,
                        target_user=_target_user,
                        channel_id=_channel,
                        source="post_proactive_reflect",
                        description=f"追问: {_reflect_thought[:60]}",
                        urgency=min(0.8, _reflect_desire / 10.0),
                        context_snippet=f"原消息: {proactive_text[:100]} | 反思: {_reflect_thought[:100]}",
                        expected_outcome="引起对方注意或回复",
                    )
                    self._last_proactive_intent_id = _followup_intent.intent_id
                    logger.info(
                        f"{self.log_prefix} [等待-反思] 追问意图已创建 "
                        f"{_followup_intent.intent_id[:8]} urgency={_followup_intent.urgency:.2f}"
                    )
                except Exception as _e:
                    logger.debug(f"{self.log_prefix} [等待-反思] 追问意图创建异常: {_e}")
            elif _reflect_action == "wait":
                # 继续等待：不做额外动作，让下一轮背景循环自然评估
                logger.info(f"{self.log_prefix} [等待-反思] 反思决定继续等待")
            else:
                # 放弃：降低注意力
                logger.info(f"{self.log_prefix} [等待-反思] 反思决定放弃主动")
                self._deescalate_watch_after_proactive()
        finally:
            self._proactive_reflect_depth = max(
                0,
                int(getattr(self, "_proactive_reflect_depth", 0) or 0) - 1,
            )

    def _deescalate_watch_after_proactive(self) -> None:
        """主动追问放弃后降低注意力"""
        try:
            _wm = self._orch.get("watch_machine")
            _wm.signal_deescalate(trigger="主动无回应，兴趣回落")
        except Exception as _e:
            logger.debug(f"{self.log_prefix} 注意力降低异常: {_e}")

    def _register_bot_message_record(
        self,
        reply_text: str,
        loop_info: Any,
        target_message: Any,
        was_proactive: bool,
        reply_trace_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(loop_info, dict):
            action_meta = loop_info.get("loop_action_info") or {}
            message_id = str(
                loop_info.get("message_id", "")
                or loop_info.get("reply_message_id", "")
                or loop_info.get("sent_message_id", "")
                or action_meta.get("message_id", "")
                or ""
            ).strip()
        else:
            message_id = str(
                getattr(loop_info, "message_id", "")
                or getattr(loop_info, "reply_message_id", "")
                or getattr(loop_info, "sent_message_id", "")
                or ""
            ).strip()
        if not message_id:
            message_id = f"bot_local_{self.stream_id}_{int(time.time() * 1000)}"

        if self._recent_bot_utterances:
            for item in reversed(self._recent_bot_utterances):
                if str(item.get("text", "") or "").strip() == str(reply_text or "").strip():
                    item["message_id"] = message_id
                    break
        try:
            from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

            acquire_reply_coordinator().bind_utterance_message_id(self.stream_id, reply_text, message_id)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 回复协调消息ID回填失败: {exc}")

        response_to = str(getattr(target_message, "message_id", "") or "").strip()
        try:
            from src.core.self_reply_recognizer import (
                MessageSource,
                get_self_reply_recognizer,
            )

            recognizer = get_self_reply_recognizer()
            recognizer.register_bot_user("bot")
            configured_bot_id = str(getattr(global_config.bot, "qq_account", "") or "").strip()
            if configured_bot_id:
                recognizer.register_bot_user(configured_bot_id)
            recognizer.mark_bot_message(
                message_id=message_id,
                content=reply_text,
                source=(MessageSource.BOT_PROACTIVE if was_proactive else MessageSource.BOT_REPLY),
                response_to=response_to,
                context={"stream_id": self.stream_id},
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 注册bot消息记录失败: {exc}")
        try:
            from src.modules.recall.self_awareness import get_self_awareness

            get_self_awareness().record_message(
                msg_id=message_id,
                stream_id=self.stream_id,
                channel_id=self.stream_id,
                content=reply_text,
                content_type="text",
                context={
                    "response_to": response_to,
                    "target_user_id": str(getattr(target_message, "user_id", "") or ""),
                    "source": "proactive_message" if was_proactive else "reply",
                    "raw_reply": str((reply_trace_meta or {}).get("raw_reply", "") or "").strip(),
                    "final_sent_reply": str((reply_trace_meta or {}).get("final_sent_reply", reply_text) or "").strip(),
                    "pre_send_rewritten": bool((reply_trace_meta or {}).get("pre_send_rewritten", False)),
                    "pre_send_reason": str((reply_trace_meta or {}).get("pre_send_reason", "") or "").strip(),
                    "audit_label": str((reply_trace_meta or {}).get("audit_label", "") or "").strip(),
                },
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 自我觉察消息回填失败: {exc}")

    async def _llm_autonomous_decide(
        self,
        now: float,
        relation_result: Dict,
        messages: List,
        voice_conclusion=None,
    ) -> "AutonomousDecision":
        """
        LLM 自主决策 - 让 LLM 自己感知环境并决定是否要主动发言
        这是真正的主动：不是被规则触发，而是 LLM 自己决定。
        LLM 会感知：
        - 群聊氛围
        - 当前话题
        - 沉默时长
        - 自己的感受（内心独白结果）
        然后自己决定是否要说话、说什么。
        """

        try:
            from src.chat.heart_flow.llm_autonomous_planner import (
                get_llm_autonomous_planner,
                AutonomousDecision,
            )

            planner = get_llm_autonomous_planner()
            # 构建环境快照，传递内心独白结果
            env = self._build_environment_snapshot(now, relation_result, messages, voice_conclusion)
            # 让 LLM 自己决定
            decision = await planner.perceive_and_decide(env)
            # 输出 LLM 的思考过程
            if decision.should_act:
                logger.info(
                    f"{self.log_prefix} 自主决策 "
                    f"感受: {decision.emotional_state}, "
                    f"意图: {decision.social_intention}, "
                    f"决定: {decision.action_type}, "
                    f"内容规划: {decision.content_plan[:60] if decision.content_plan else '无'}"
                )
            else:
                logger.debug(
                    f"{self.log_prefix} 自主决策决定观察: {decision.reasoning[:80] if decision.reasoning else '无原因'}"
                )
            return decision
        except Exception as exc:
            logger.error(f"{self.log_prefix} LLM 自主决策失败: {exc}")
            from src.chat.heart_flow.llm_autonomous_planner import (
                AutonomousDecision,
            )

            return AutonomousDecision(
                should_act=False,
                action_type="observe",
                reasoning=f"LLM 调用失败: {exc}",
            )

    def _ensure_night_cycle(self):
        """[已废弃 2026-04-06] 夜间周期实例获取已迁移至 D6 EnergyChainDimension"""
        return None

    @staticmethod
    def _normalize_night_action(action: Any, allow_internal: bool = False) -> str:
        normalized = str(action or "").strip().lower()
        public_actions = {"sleep_resist", "grumpy_glance", "soft_wake", "full_wake"}
        if normalized in public_actions:
            return normalized
        if allow_internal and normalized in {"deep_sleep", "hard_block", "llm_decide_l2", "force_wake"}:
            return normalized
        return "sleep_resist"

    def _apply_night_cycle_modulation(self, now: float) -> Optional[dict]:
        """夜间节律调制：区分熬夜清醒态 vs 睡眠态
        反馈回路: apply_wake_feedback() 在每次决策后调整动态阈值
        """
        _SLEEP_PHASES = frozenset({"DROWSY", "DEEP_VALLEY"})
        try:
            # 每轮先清空临时夜醒态，避免同一夜里沿用上一条消息残留的困倦/烦躁 prompt。
            self._night_soft_wake_info = None
            _d6 = EnergyChainDimension.get_instance()
            _nm = _d6._get_night_mode(self.stream_id)
            if _nm is None or _nm.phase == "DAYTIME":
                return None
            if self._last_msg_was_admin:
                return {"action": "force_wake", "reason": "管理员强制唤醒"}
            _ncs = None
            _ns = None
            _precise_phase = ""
            try:
                from src.core.night_cycle_system import get_night_cycle

                _ncs = get_night_cycle(self.stream_id)
                _ns = _ncs.state_snapshot
                _phase_obj = getattr(_ns, "current_phase", None)
                _precise_phase = str(
                    getattr(_phase_obj, "value", _phase_obj) or ""
                ).strip().lower()
            except Exception as _phase_exc:
                logger.debug(f"{self.log_prefix} 夜间精确相位读取异常: {_phase_exc}")
            _is_sleep_phase = _precise_phase in {"light_sleep", "deep_sleep"}
            if not _precise_phase:
                _is_sleep_phase = _nm.phase in _SLEEP_PHASES
            if not _is_sleep_phase:
                try:
                    if _ns is None:
                        from src.core.night_cycle_system import get_night_cycle

                        _ncs = get_night_cycle(self.stream_id)
                        _ns = _ncs.state_snapshot
                    _half = _ns.half_asleep_level
                    _drowsy_val = _ns.drowsiness_value
                    _pressure_val = _ns.overnight_pressure
                    _reserve_val = _ns.sleep_reserve
                    _daily_fatigue_val = _ns.daily_fatigue
                    _burned = _ns.is_burnthrough_active
                    _hour = datetime.datetime.now().hour
                    _is_late_night = _hour >= 1 and _hour < 7
                    _should_trigger = (
                        (_half > 0.15)
                        or (_daily_fatigue_val > 20)
                        or (_drowsy_val > 12)
                        or (_pressure_val > 25)
                        or (_reserve_val < 35)
                        or (_is_late_night and _daily_fatigue_val > 12)
                        or (_is_late_night and _ns.interruption_count >= 2)
                        or (_is_late_night and _pressure_val > 15)
                    )
                    if _should_trigger:
                        _d6.record_disturbance(self.stream_id)
                        _ctx_fallback = _d6._build_full_context(
                            _d6._ensure_channel(self.stream_id),
                            _nm,
                            self.stream_id,
                            time.time(),
                        )
                        if _half > 0.6 or _daily_fatigue_val > 65 or (_is_late_night and _daily_fatigue_val > 35):
                            _sw_action = "soft_wake"
                            _sw_reason = (
                                f"透支严重: 日疲劳={_daily_fatigue_val:.0f}% "
                                f"困意={_drowsy_val:.0f} 熬压={_pressure_val:.0f} "
                                f"储备={_reserve_val:.0f}"
                            )
                        elif _drowsy_val > 35 or _pressure_val > 40 or (_is_late_night and _daily_fatigue_val > 22):
                            _sw_action = "grumpy_glance"
                            _sw_reason = (
                                f"熬夜困倦: 困意={_drowsy_val:.0f} "
                                f"日疲劳={_daily_fatigue_val:.0f}% 熬压={_pressure_val:.0f}"
                            )
                        elif _burned:
                            _sw_action = "grumpy_glance"
                            _sw_reason = f"熬穿状态: 日疲劳={_daily_fatigue_val:.0f}% 压力={_pressure_val:.0f}"
                        else:
                            _sw_action = "soft_wake"
                            _sw_reason = (
                                f"凌晨未恢复: 日疲劳={_daily_fatigue_val:.0f}% 困意={_drowsy_val:.0f} hour={_hour}"
                            )
                        self._last_night_context_snapshot = _ctx_fallback
                        logger.info(
                            f"{self.log_prefix} 🌙 [{_nm.phase}] 深夜状态触发类唤醒: {_sw_action} | {_sw_reason}"
                        )
                        return {
                            "action": _sw_action,
                            "reason": _sw_reason,
                            "context": _ctx_fallback,
                            "stimulus": _ctx_fallback.get("adaptive", {}).get("stimulus_strength", 30.0),
                            "desire": max(3, min(6, int(8 - _daily_fatigue_val / 20))),
                        }
                except Exception as _cde_exc:
                    logger.debug(f"{self.log_prefix} CDE疲劳检测异常: {_cde_exc}")
                return None
            _ch = _d6._ensure_channel(self.stream_id)
            try:
                from src.core.night_cycle_system import get_night_cycle

                _ncs_cap = get_night_cycle(self.stream_id)
                if not _ncs_cap.evaluate_sleep_reply_budget():
                    _cap_info = _ncs_cap.state_snapshot
                    return {
                        "action": "hard_block",
                        "reason": (
                            f"睡眠回复限次已达({getattr(_cap_info, 'sleep_reply_used', 0)}/1)"
                        ),
                    }
                if not _ncs_cap.can_reply_tonight():
                    _cap_info = _ncs_cap.state_snapshot
                    return {
                        "action": "hard_block",
                        "reason": f"夜间回复已达上限({_cap_info.night_reply_count}/{_cap_info.night_reply_cap})",
                    }
            except Exception:
                if hasattr(_ch, "night_reply_cap") and hasattr(_ch, "night_reply_count"):
                    if _ch.night_reply_count >= _ch.night_reply_cap:
                        return {
                            "action": "hard_block",
                            "reason": f"夜间回复已达上限({_ch.night_reply_count}/{_ch.night_reply_cap})",
                        }
            _night_ctx = _d6.record_disturbance(self.stream_id)
            _adaptive = _night_ctx.get("adaptive", {})
            _stimulus = _adaptive.get("stimulus_strength", 0.0)
            _threshold = _adaptive.get("threshold", 50.0)
            _exceeds = _adaptive.get("exceeds", False)
            _gap = _adaptive.get("gap", 0.0)
            _cde = _night_ctx.get("cde_3d", {})
            _composite = _cde.get("composite_load", 0.0)
            _collapse = _cde.get("collapse_threshold", 50.0)
            _dist_count = _night_ctx.get("disturbance", {}).get("count", 0)
            _consec_sleeps = _adaptive.get("consecutive_sleeps", 0)
            if _composite >= _collapse * 1.1 or _cde.get("collapse_imminent", False):
                _d6.apply_wake_feedback(self.stream_id, "deep_sleep", 0, _stimulus)
                return {
                    "action": "deep_sleep",
                    "reason": f"CDE溢出(Composite {_composite:.1f}≥{_collapse:.1f}), 强制深睡",
                    "context": _night_ctx,
                    "stimulus": _stimulus,
                    "threshold": _threshold,
                }
            if not _exceeds:
                logger.debug(
                    f"{self.log_prefix} 🌙 🔒 [{_nm.phase}] L1拦截: "
                    f"刺激={_stimulus:.1f} < 阈值={_threshold:.1f} 打扰={_dist_count}"
                )
                _d6.apply_wake_feedback(self.stream_id, "sleep_resist", 0, _stimulus)
                return {
                    "action": "sleep_resist",
                    "reason": f"[{_nm.phase}] L1未达(刺激{_stimulus:.1f}<阈值{_threshold:.1f})",
                    "context": _night_ctx,
                    "stimulus": _stimulus,
                    "threshold": _threshold,
                }
            _need_l2 = (
                (_gap >= 20.0)
                or (_consec_sleeps >= 5 and _stimulus >= _threshold * 0.9)
                or (_nm.phase == "DEEP_VALLEY" and _gap >= 12.0 and _dist_count >= 8)
            )
            if _need_l2:
                self._last_night_context_snapshot = _night_ctx
                logger.info(
                    f"{self.log_prefix} 🌙 🔓 [{_nm.phase}] L1通过+触发L2: "
                    f"刺激={_stimulus:.1f}>={_threshold:.1f}(超{_gap:+.1f}) "
                    f"打扰={_dist_count} 连续睡={_consec_sleeps}"
                )
                return {
                    "action": "llm_decide_l2",
                    "reason": f"[{_nm.phase}] L1通过+极端条件,需L2决策",
                    "context": _night_ctx,
                    "stimulus": _stimulus,
                    "threshold": _threshold,
                }
            _algo_action = self._algorithm_wake_decision(
                _stimulus, _threshold, _gap, _dist_count, _consec_sleeps, _nm.phase, _cde, _night_ctx, _d6
            )
            return _algo_action
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 夜间节律评估异常: {exc}")
        return None

    def _algorithm_wake_decision(
        self, stimulus, threshold, gap, dist_count, consec_sleeps, phase, cde, ctx, d6
    ) -> dict:
        """L1通过后的纯算法决策（不调用LLM）

        仅在非极端情况下使用。根据刺激强度、差距、打扰次数、CDE状态
        直接判定行为，避免每次都调用Inner Voice。
        """
        _composite = cde.get("composite_load", 0.0)
        _collapse = cde.get("collapse_threshold", 50.0)
        _irritation = cde.get("wake_irritation", 0.0)
        _daily_fatigue_val = cde.get("daily_fatigue", 0)
        _half_asleep = cde.get("half_asleep", 0.0)
        _is_lazy = cde.get("is_lazy", False)
        if phase == "DEEP_VALLEY":
            if gap <= 5 or dist_count <= 2:
                _action = "sleep_resist"
                _desire = max(0, min(2, int(gap)))
            elif gap <= 12:
                _action = "grumpy_glance"
                _desire = 3
            elif gap <= 22:
                _action = "soft_wake"
                _desire = 6
            else:
                _action = "full_wake"
                _desire = 9
        elif phase == "DROWSY":
            if gap <= 3 or dist_count <= 1:
                _action = "sleep_resist"
                _desire = 1
            elif gap <= 10:
                _action = "grumpy_glance"
                _desire = 4
            elif gap <= 18:
                _action = "soft_wake"
                _desire = 7
            else:
                _action = "full_wake"
                _desire = 9
        else:
            _action = "soft_wake"
            _desire = 5
        if _is_lazy and _action in ("full_wake",):
            _action = "soft_wake"
            _desire = max(4, _desire - 3)
        if _half_asleep > 0.6 and _action == "full_wake":
            _action = "soft_wake"
            _desire = 7
        if _irritation > 0.4 and _action == "full_wake":
            _action = "grumpy_glance"
            _desire = 5
        d6.apply_wake_feedback(self.stream_id, _action, _desire, stimulus)
        logger.info(
            f"{self.log_prefix} 🌙 ⚙️ [{phase}] 算法决策: {_action} | "
            f"欲望={_desire}/10 | 刺激={stimulus:.1f}/阈值={threshold:.1f}(超{gap:+.1f})"
        )
        return {
            "action": _action,
            "reason": f"[{phase}] 算法决策: desire={_desire}/10 stimulus={stimulus:.1f}",
            "context": ctx,
            "stimulus": stimulus,
            "threshold": threshold,
            "desire": _desire,
        }

    async def _night_llm_decision(self, incoming_batch=None) -> dict:
        """L2: 调用Inner Voice引擎做完整的夜间行为决策

        仅在L1自适应阈值通过后触发。
        将多维状态快照注入BoundaryContext，由LLM自主判断：
        - full_wake: 彻底清醒，正常回复
        - soft_wake: 醒了但慵懒，简短回复
        - grumpy_glance: 烦躁瞥一眼，极简回复或不回
        - sleep_resist: 决定继续睡

        决策完成后自动调用 apply_wake_feedback() 调整动态阈值。
        """
        _ctx = getattr(self, "_last_night_context_snapshot", None)
        if not _ctx:
            return {"action": "sleep_resist", "reason": "无夜间上下文", "desire": 0}
        _adaptive = _ctx.get("adaptive", {})
        _stimulus = _adaptive.get("stimulus_strength", 0.0)
        try:
            from src.chat.heart_flow.inner_voice import SelfDialogueEngine

            _engine = SelfDialogueEngine(self.stream_id)
            _raw_text = ""
            if incoming_batch:
                for msg in incoming_batch if isinstance(incoming_batch, list) else [incoming_batch]:
                    t = str(
                        getattr(msg, "processed_plain_text", "")
                        or getattr(msg, "plain_text", "")
                        or getattr(msg, "content", "")
                        or ""
                    )
                    if t:
                        _raw_text = t[:200]
                        break
            _dist = _ctx.get("disturbance", {})
            _energy = _ctx.get("energy", {})
            _cde = _ctx.get("cde_3d", {})
            _emotion = _ctx.get("emotion", {})
            _verdict = await _engine.generate_reflection(
                heart_state_label=f"夜间-{_ctx.get('d6_label', '未知')}",
                raw_text=_raw_text or "(夜间消息)",
                speaker_name="夜间访客",
                dialogue_history=[],
                mental_drain=_cde.get("composite_load", 0.0) * 0.5,
                endurance=max(1.0, _energy.get("combined_ratio", 0.5) * 100),
                irritation=min(100.0, _cde.get("wake_irritation", 0.0) * 50 + _energy.get("annoyance", 0.0)),
                wound_score=0.0,
                readiness=max(0.05, 1.0 - _cde.get("response_suppression", 0.0)),
                extra_context={
                    "profile_summary": f"夜间状态: 打扰{_dist.get('count', 0)}次 | "
                    f"D={_cde.get('drowsiness', 0):.0f} P={_cde.get('pressure', 0):.0f} S={_cde.get('reserve', 0):.0f} | "
                    f"疲劳{_cde.get('daily_fatigue', 0):.0f}% | "
                    f"情绪V{_emotion.get('vitality', 50):.0f}/W{_emotion.get('weariness', 20):.0f}/X{_emotion.get('vexation', 10):.0f}",
                    "memory_summary": f"睡眠债{_cde.get('sleep_debt', 0):.1f} | "
                    f"打断{_cde.get('interruption_count', 0)}次 | "
                    f"强制唤醒{_cde.get('force_wake_today', 0)}次今天",
                    "execution_hint": "现在是深夜/凌晨。你正在睡觉或半梦半醒间被消息打扰。"
                    f"当前刺激强度={_stimulus:.1f}/100, 已经通过了L1阈值门控。",
                    "rt_annoyance": str(_cde.get("wake_irritation", 0.0) * 30 + _energy.get("annoyance", 0.0)),
                    "rt_pressure": str(_cde.get("composite_load", 0.0)),
                },
                is_admin=bool(incoming_batch and self._is_force_wake_admin(incoming_batch)),
            )
            _desire = getattr(_verdict, "reply_desire_level", 0)
            _should = getattr(_verdict, "should_reply", False)
            _thinking = getattr(_verdict, "thinking", "")
            _mood = getattr(_verdict, "current_mood", "")
            _intent = _verdict.primary_intent() if hasattr(_verdict, "primary_intent") else None
            _intent_type = getattr(_intent, "intent_type", "") if _intent else ""
            _d6 = EnergyChainDimension.get_instance()
            if _desire >= 8 or _intent_type in ("reply", "miao_reply"):
                _action = "full_wake"
            elif _should or _desire >= 6:
                _action = "soft_wake"
            elif _desire >= 3:
                _action = "grumpy_glance"
            else:
                _action = ""
            if _action:
                _d6.apply_wake_feedback(self.stream_id, _action, _desire, _stimulus)
                logger.info(
                    f"{self.log_prefix} 🌙 🧠 L2决策: {_action} | 欲望={_desire}/10 回复={'是' if _should else '否'} "
                    f"意图={_intent_type} 心情={_mood} | {_thinking[:60] if _thinking else ''}"
                )
                return {
                    "action": _action,
                    "reason": f"L2-LLM: desire={_desire}/10 mood={_mood}",
                    "llm_verdict": {
                        "desire": _desire,
                        "should_reply": _should,
                        "thinking": _thinking[:200],
                        "mood": _mood,
                        "intent_type": _intent_type,
                    },
                    "context": _ctx,
                    "stimulus": _stimulus,
                    "desire": _desire,
                }
            _d6.apply_wake_feedback(self.stream_id, "sleep_resist", _desire, _stimulus)
            logger.info(
                f"{self.log_prefix} 🌙 💤 L2决定继续睡: 欲望={_desire}/10 | {_thinking[:60] if _thinking else ''}"
            )
            return {
                "action": "sleep_resist",
                "reason": f"L2-LLM不回应: desire={_desire}/10 mood={_mood}",
                "llm_verdict": {
                    "desire": _desire,
                    "should_reply": False,
                    "thinking": _thinking[:200],
                    "mood": _mood,
                },
                "context": _ctx,
                "stimulus": _stimulus,
                "desire": _desire,
            }
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 夜间L2决策异常，降级为算法: {exc}")
            _ctx_local = getattr(self, "_last_night_context_snapshot", {})
            _cde_local = _ctx_local.get("cde_3d", {})
            _composite_local = _cde_local.get("composite_load", 0.0)
            _collapse_local = _cde_local.get("collapse_threshold", 50.0)
            _dist_local = _ctx_local.get("disturbance", {}).get("count", 0)
            try:
                _d6_fallback = EnergyChainDimension.get_instance()
                _d6_fallback.apply_wake_feedback(self.stream_id, "sleep_resist", 0, _stimulus)
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
            if _composite_local > (_collapse_local * 0.7):
                return {"action": "deep_sleep", "reason": f"算法降级: CDE过高({_composite_local:.1f})", "desire": 0}
            if _dist_local >= 8:
                return {"action": "soft_wake", "reason": f"算法降级: 高频打扰({_dist_local})", "desire": 5}
            if _dist_local >= 4:
                return {"action": "grumpy_glance", "reason": f"算法降级: 中频打扰({_dist_local})", "desire": 3}
            return {"action": "deep_sleep", "reason": "算法降级: 低频打扰", "desire": 0}

    def _is_force_wake_from_batch(self, incoming_batch) -> bool:
        """从消息批次判断是否有管理员强制唤醒"""
        if incoming_batch is None:
            return False
        _admin_ids = global_config.chat.admin_force_wake_qq_ids
        for msg in incoming_batch if isinstance(incoming_batch, list) else [incoming_batch]:
            uid = str(getattr(msg, "user_id", "") or "")
            if uid and uid in _admin_ids:
                return True
        return False

    def _apply_pattern_based_routing(self, incoming_batch, pinged_msg=None) -> Optional[str]:
        """群体模式硬路由：根据检测到的群行为模式直接决定处理策略"""
        if not self._cached_pattern_evidence:
            return None
        _top = self._cached_pattern_evidence[0]
        _conf = float(getattr(_top, "confidence", 0.0) or 0.0)
        if _conf < 0.4:
            return None
        _pat_enum = getattr(_top, "pattern", None)
        if _pat_enum is None:
            return None

        _has_ping = pinged_msg is not None
        if _pat_enum == GroupPattern.COPYCAT_CHAIN and not _has_ping:
            self._pattern_forced_observe = True
            logger.info(f"{self.log_prefix} 🔄 复读跟风模式(置信{_conf:.2f})→进入围观观察")
            return "group_pattern:copycat_observe"
        if _pat_enum in (
            GroupPattern.PILE_ON,
            GroupPattern.CONFLICT_ESCALATION,
        ):
            self._defense_mode_active = True
            if not _has_ping and _conf > 0.55:
                logger.info(f"{self.log_prefix} ⚔️ 围攻/冲突升级(置信{_conf:.2f})→防御沉默")
                return "group_pattern:defense_silence"
            logger.info(f"{self.log_prefix} ⚔️ 围攻/冲突检测(置信{_conf:.2f})→防御模式激活")
            return None
        if _pat_enum == GroupPattern.SPECTATOR_MODE and not _has_ping:
            self._pattern_forced_observe = True
            logger.info(f"{self.log_prefix} 👥 围观模式(置信{_conf:.2f})→仅观察不参与")
            return "group_pattern:spectator"
        if _pat_enum == GroupPattern.MEME_STORM and not _has_ping:
            _msg_count = len(incoming_batch) if incoming_batch else 0
            if _msg_count >= 8:
                self._pattern_forced_observe = True
                logger.info(f"{self.log_prefix} 🌪️ 梗风暴( {_msg_count}条/置信{_conf:.2f})→围观吃瓜")
                return "group_pattern:meme_observe"
        if _pat_enum == GroupPattern.SUDDEN_SILENCE or _pat_enum == GroupPattern.QUIET_REFLECTION:
            if not _has_ping:
                logger.info(f"{self.log_prefix} 🔇 安静/反思模式(置信{_conf:.2f})→降低参与欲")
                self._peek_mode_active = True
                return "group_pattern:quiet_reflect"
        return None

    def _merge_group_context_signal(
        self,
        group_sense_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """统一群场景硬门与群态势软信号，收敛为一份可消费协议。"""
        signal: Dict[str, Any] = dict(group_sense_result or {})
        scene_snapshot = getattr(self, "_cached_scene_snapshot", None)

        scene_joinable = True
        scene_reason = ""
        scene_dominant_speaker = ""
        scene_atmosphere = ""
        scene_atmosphere_label = ""
        scene_topic_hints: List[str] = []
        if scene_snapshot is not None:
            scene_joinable = bool(getattr(scene_snapshot, "suitable_to_join", True))
            scene_reason = str(getattr(scene_snapshot, "join_unsuitable_reason", "") or "")
            scene_dominant_speaker = str(getattr(scene_snapshot, "dominant_speaker", "") or "")
            scene_atmo_raw = getattr(scene_snapshot, "atmosphere", None)
            if scene_atmo_raw is not None:
                scene_atmosphere = getattr(scene_atmo_raw, "value", str(scene_atmo_raw))
                if hasattr(scene_atmo_raw, "label"):
                    try:
                        scene_atmosphere_label = str(scene_atmo_raw.label() or "")
                    except Exception:
                        scene_atmosphere_label = ""
            try:
                scene_topic_hints = list(self._orch.get("scene_state").active_topics(limit=3) or [])
            except Exception as exc:
                logger.debug(f"{self.log_prefix} 群场景话题读取失败: {exc}")

        merged_topic_hints: List[str] = []
        seen_topic_keys = set()
        for hint in [*scene_topic_hints, *(signal.get("topic_hints", []) or [])]:
            normalized = self._normalize_topic_text(str(hint))
            if len(normalized) < 2 or normalized in seen_topic_keys:
                continue
            seen_topic_keys.add(normalized)
            merged_topic_hints.append(normalized[:12])
            if len(merged_topic_hints) >= 5:
                break

        hard_block_reason = ""
        if not scene_joinable:
            hard_block_reason = "scene_constraint:unsuitable_to_join"
        elif scene_atmosphere == AtmosphereType.SPAM_FLOOD.value:
            hard_block_reason = "scene_constraint:spam_flood"

        soft_guard_reason = ""
        if signal.get("burst_detected"):
            soft_guard_reason = "group_sense:burst_detected"
        elif signal.get("controversy_detected"):
            soft_guard_reason = "group_sense:controversy_detected"

        signal.update(
            {
                "suitable_to_join": scene_joinable,
                "join_unsuitable_reason": scene_reason,
                "scene_suitable_to_join": scene_joinable,
                "scene_join_unsuitable_reason": scene_reason,
                "scene_dominant_speaker": scene_dominant_speaker,
                "scene_atmosphere": scene_atmosphere,
                "scene_atmosphere_label": scene_atmosphere_label,
                "topic_hints": merged_topic_hints,
                "hard_block_reason": hard_block_reason,
                "hard_blocked": bool(hard_block_reason),
                "soft_guard_reason": soft_guard_reason,
                "joinability_source": "group_scene_state",
                "pressure_source": "group_sense",
            }
        )
        self._last_group_context_signal = dict(signal)
        return signal

    def _apply_scene_hard_constraints(
        self,
        pinged_msg=None,
        group_context_signal: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """群场景硬约束：suitable_to_join和atmosphere直接影响决策"""
        signal = (
            group_context_signal
            if isinstance(group_context_signal, dict)
            else (getattr(self, "_last_group_context_signal", None) or None)
        )
        if signal is None and self._cached_scene_snapshot:
            signal = self._merge_group_context_signal({})
        if not self._cached_scene_snapshot and not signal:
            return None
        _joinable = bool(
            (signal or {}).get(
                "scene_suitable_to_join",
                getattr(self._cached_scene_snapshot, "suitable_to_join", True),
            )
        )
        _has_ping = pinged_msg is not None
        if not _joinable and not _has_ping:
            _reason = str(
                (signal or {}).get(
                    "scene_join_unsuitable_reason",
                    getattr(self._cached_scene_snapshot, "join_unsuitable_reason", ""),
                )
                or ""
            )
            logger.info(f"{self.log_prefix} 🚫 场景不适合插话: {_reason}")
            return "scene_constraint:unsuitable_to_join"
        _atmo = (signal or {}).get("scene_atmosphere", "")
        if not _atmo and self._cached_scene_snapshot is not None:
            _atmo_raw = getattr(self._cached_scene_snapshot, "atmosphere", None)
            _atmo = getattr(_atmo_raw, "value", _atmo_raw) if _atmo_raw is not None else ""
        if _atmo and not _has_ping:
            if _atmo == AtmosphereType.ARGUMENT.value:
                logger.info(f"{self.log_prefix} ⚖️ 争论氛围→降低主动参与")
                self._defense_mode_active = True
                return None
            if _atmo == AtmosphereType.SPAM_FLOOD.value:
                self._pattern_forced_observe = True
                logger.info(f"{self.log_prefix} 🌀 刷屏氛围→仅观察")
                return "scene_constraint:spam_flood"
        return None

    def _build_environment_snapshot(
        self,
        now: float,
        relation_result: Dict,
        messages: List,
        voice_conclusion=None,
    ) -> "EnvironmentSnapshot":
        """构建环境快照，供 LLM 感知 - 集成共享资源和内心独白结果"""
        from src.chat.heart_flow.llm_autonomous_planner import (
            EnvironmentSnapshot,
        )
        import datetime

        silence_sec = 0.0
        try:
            from src.chat.proactive.silence_watcher import get_quiet_monitor

            silence_sec = get_quiet_monitor().measure_silence_sec(self.stream_id)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        energy = 1.0
        mood = "平静"
        social_value = 0.0
        chat_value = 100.0
        activity_level = 50.0
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            _tv = float(_ch.thinking_value) if _ch else 50.0
            _tc = float(_ch.thinking_ceiling) if _ch else 100.0
            thinking_factor = max(0.0, min(1.0, _tv / max(_tc, 1.0)))
            _cv = float(_ch.chat_pool) if _ch else 50.0
            _cc = float(_ch.chat_ceiling) if _ch else 100.0
            _al = float(_ch.activity_level) if _ch else 50.0
            _sv = float(_ch.social_value) if _ch else 0.0
            chat_value = float(relation_result.get("chat_value", _cv) or 0.0)
            activity_level = float(relation_result.get("activity_level", _al) or 0.0)
            social_value = float(
                relation_result.get(
                    "social_value",
                    relation_result.get("shared_social_value", _sv),
                )
                or 0.0
            )
            chat_factor = max(0.0, min(1.0, float(chat_value) / max(_cc, 1.0)))
            activity_factor = max(0.0, min(1.0, float(activity_level) / 100.0))
            social_factor = max(0.0, min(1.0, (float(social_value) + 100.0) / 200.0))
            energy = max(
                0.05,
                min(
                    1.0,
                    thinking_factor * 0.4 + chat_factor * 0.25 + activity_factor * 0.2 + social_factor * 0.15,
                ),
            )
            pressure = float(relation_result.get("psychological_pressure", 0.0) or 0.0)
            if pressure > 70:
                mood = "紧绷"
            elif social_value > 30:
                mood = "愉快"
            elif social_value < -20:
                mood = "有点烦"
            elif chat_value < 30:
                mood = "有点累"
            elif activity_level > 70:
                mood = "活跃"
            explicit_mood = str(relation_result.get("mood", "") or "").strip()
            if explicit_mood and explicit_mood not in ("平静", "未知"):
                mood = explicit_mood
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        atmosphere = "平静"
        try:
            ambient = self._sample_channel_ambient()
            if ambient:
                atmosphere = ambient.get("category", "平静")
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        recent_messages = []
        current_topics = self._extract_current_topics(messages)
        for msg in messages[-10:]:
            if not self._is_human_message_obj(msg):
                continue
            content = (
                getattr(msg, "processed_plain_text", "")
                or getattr(msg, "raw_plain_text", "")
                or getattr(msg, "content", "")
                or ""
            )
            speaker = getattr(msg, "user_nickname", "") or getattr(msg, "user_name", "") or "未知"
            if content:
                recent_messages.append(
                    {
                        "speaker": speaker,
                        "content": content[:200],
                    }
                )
        if not recent_messages:
            logger.warning(f"{self.log_prefix} 📷 没有检测到最近消息，消息数量={len(messages)}")
        now_dt = datetime.datetime.now()
        time_of_day = now_dt.strftime("%H:%M")
        day_of_week = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now_dt.weekday()]
        active_users = list(
            {
                str(getattr(msg, "user_nickname", "") or getattr(msg, "user_name", "") or "").strip()
                for msg in messages[-20:]
                if self._is_human_message_obj(msg)
                and str(getattr(msg, "user_nickname", "") or getattr(msg, "user_name", "") or "").strip()
            }
        )[:5]
        my_last_message = ""
        my_last_message_time = 0.0
        for msg in reversed(messages[-20:]):
            if self._is_bot_message_obj(msg):
                my_last_message = getattr(msg, "processed_plain_text", "")[:100] or ""
                my_last_message_time = getattr(msg, "timestamp", 0.0)
                break
        # 提取内心独白结果
        inner_voice_desire = 5
        inner_voice_thinking = ""
        inner_voice_mood = ""
        inner_voice_primary_intent = ""
        inner_voice_needs_upgrade = False
        if voice_conclusion is not None:
            if hasattr(voice_conclusion, "reply_desire_level"):
                inner_voice_desire = voice_conclusion.reply_desire_level
            if hasattr(voice_conclusion, "thinking"):
                inner_voice_thinking = voice_conclusion.thinking[:200] if voice_conclusion.thinking else ""
            if hasattr(voice_conclusion, "current_mood"):
                inner_voice_mood = voice_conclusion.current_mood or ""
            if hasattr(voice_conclusion, "needs_upgrade"):
                inner_voice_needs_upgrade = bool(voice_conclusion.needs_upgrade)
            _pi = voice_conclusion.primary_intent()
            if _pi is not None:
                _ptype = getattr(_pi, "intent_type", "") or ""
                _pdesc = getattr(_pi, "description", "") or ""
                inner_voice_primary_intent = f"{_ptype}" + (f"({_pdesc})" if _pdesc else "")
        # 补充对话阶段与创伤压力
        _snap_dialogue_phase = ""
        _snap_trauma_pressure = 0.0
        try:
            from src.chat.proactive.session_tracker import get_memoir_cabinet

            _snap_cabinet = get_memoir_cabinet()
            _snap_user_id = self._resolve_latest_human_user_id(
                messages,
                allow_cached_fallback=False,
            )
            _snap_memoir = _snap_cabinet.retrieve(_snap_user_id) if hasattr(_snap_cabinet, "retrieve") else None
            if _snap_memoir and hasattr(_snap_memoir, "phase"):
                _snap_dialogue_phase = (
                    _snap_memoir.phase.value if hasattr(_snap_memoir.phase, "value") else str(_snap_memoir.phase)
                )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.modules.trauma.trauma_system import get_trauma_system

            _snap_trauma = get_trauma_system().get_state()
            _snap_trauma_pressure = _snap_trauma.stress_accumulation
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        _pending_intentions_hint = ""
        try:
            from src.chat.proactive.intention_pool import get_intention_pool

            _ipool = get_intention_pool()
            _pending = _ipool.get_active_intentions(self.stream_id, limit=3)
            if _pending:
                _pending_intentions_hint = "【跨轮意图】你有以下未完成的意图：\n" + "\n".join(
                    f"- {i.kind.value}: {i.description[:50]} (紧迫度={i.effective_urgency():.2f})" for i in _pending
                )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        _snap = EnvironmentSnapshot(
            channel_id=self.stream_id,
            channel_name=getattr(self.chat_stream, "stream_name", "群聊") or "群聊",
            silence_seconds=silence_sec,
            energy_level=energy,
            mood=mood,
            atmosphere=atmosphere,
            recent_messages=recent_messages,
            active_users=active_users,
            current_topics=current_topics,
            social_value=social_value,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            my_last_message=my_last_message,
            my_last_message_time=my_last_message_time,
            inner_voice_desire=inner_voice_desire,
            inner_voice_thinking=inner_voice_thinking,
            inner_voice_mood=inner_voice_mood,
            inner_voice_primary_intent=inner_voice_primary_intent,
            inner_voice_needs_upgrade=inner_voice_needs_upgrade,
            dialogue_phase=_snap_dialogue_phase,
            trauma_pressure=_snap_trauma_pressure,
            narration_hint=(self._cached_narration_plan.to_prompt_block() if self._cached_narration_plan else ""),
            impression_hint="",
            pending_intentions_hint=_pending_intentions_hint,
        )
        try:
            from src.core.impression_evolution_hub import (
                get_impression_hub,
            )

            _target_uid = ""
            if messages:
                for _dm in reversed(messages):
                    _duid = str(getattr(_dm, "user_id", "") or "")
                    if self._is_human_message_obj(_dm):
                        _target_uid = _duid
                        break
            if _target_uid:
                _ihub = get_impression_hub(self.stream_id)
                _isum = _ihub.get_impression_summary(_target_uid)
                if _isum.get("exists"):
                    _imp_lines = []
                    if _isum.get("nickname"):
                        _imp_lines.append(f"内心称呼: {_isum['nickname']}")
                    if _isum.get("narrative_type") and _isum["narrative_type"] != "stranger":
                        _imp_lines.append(f"关系定位: {_isum['narrative_type']}")
                    if _isum.get("primary_rule") and _isum["primary_rule"] != "normal":
                        _rule_map = {
                            "warm": "热情亲近",
                            "playful": "调皮打闹",
                            "respectful": "尊重客气",
                            "cautious": "谨慎戒备",
                            "dry": "冷淡简短",
                            "avoidant": "回避疏远",
                            "teasing": "调侃戏谑",
                            "protective": "保护性",
                        }
                        _imp_lines.append(f"行为策略: {_rule_map.get(_isum['primary_rule'], _isum['primary_rule'])}")
                    if _isum.get("tags"):
                        _valid_tags = [t for t in _isum["tags"] if t not in ("new",)]
                        if _valid_tags:
                            _imp_lines.append(f"印象标签: {', '.join(_valid_tags[:5])}")
                    if _imp_lines:
                        _snap.impression_hint = "【主观印象】" + "；".join(_imp_lines)
        except Exception as _ierr:
            logger.debug(f"{self.log_prefix} 印象引擎异常: {_ierr}")
        return _snap

    async def _check_identity_context(self, messages: List) -> Dict[str, Any]:
        """身份锚点集成 - 确认当前身份状态"""
        try:
            from src.core.identity_anchor import get_identity_anchor

            anchor = get_identity_anchor()
            user_id = ""
            if messages:
                for msg in reversed(messages):
                    uid = getattr(msg, "user_id", "")
                    if self._is_human_message_obj(msg):
                        user_id = uid
                        break
            context = anchor.get_identity(self.stream_id, user_id)
            conflict = anchor.check_conflict(self.stream_id, user_id)
            if conflict:
                logger.info(f"{self.log_prefix} 🔒 检测到身份冲突: {conflict.conflict_type}")
            return {
                "identity": (context.current_mode.value if context else "default"),
                "response_mode": (context.response_mode.value if context else "normal"),
                "has_conflict": conflict is not None,
            }
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 身份锚点检查失败: {exc}")
            return {
                "identity": "default",
                "response_mode": "normal",
                "has_conflict": False,
            }

    async def _update_dynamic_context(self, messages: List) -> None:
        """兼容占位：消息入口已统一写入 ContextManager，这里不再双写旧动态上下文。"""
        return None

    async def _check_self_reply_risk(self, messages: List) -> Dict[str, Any]:
        """自回复识别集成 - 检查是否在重复回复"""
        try:
            from src.core.self_reply_recognizer import (
                get_self_reply_recognizer,
            )

            recognizer = get_self_reply_recognizer()
            last_bot_msg = None
            for msg in reversed(messages):
                if self._is_bot_message_obj(msg):
                    last_bot_msg = msg
                    break
            if last_bot_msg is None:
                return {"is_self_reply": False, "similarity": 0.0}
            content = getattr(last_bot_msg, "processed_plain_text", "") or getattr(last_bot_msg, "content", "")
            result = recognizer.check_self_reply(
                channel_id=self.stream_id,
                content=content,
                author_id="bot",
            )
            if result.is_self_reply:
                logger.warning(f"{self.log_prefix} 🔄 检测到重复回复风险: 相似度={result.similarity:.2f}")
            return {
                "is_self_reply": result.is_self_reply,
                "similarity": result.similarity,
                "quality": (result.quality.value if hasattr(result, "quality") else "unknown"),
            }
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 自回复识别失败: {exc}")
            return {"is_self_reply": False, "similarity": 0.0}

    async def _track_content_state(self, messages: List) -> None:
        """内容状态追踪集成 - 记录内容处理状态"""
        try:
            from src.core.content_state_tracker import (
                get_content_state_tracker,
            )

            tracker = get_content_state_tracker()
            for msg in messages:
                content = self._extract_message_content(msg)
                if not content:
                    continue
                user_id = getattr(msg, "user_id", "")
                tracker.track_content(
                    content=content,
                    channel_id=self.stream_id,
                    user_id=user_id,
                )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 内容状态追踪失败: {exc}")

    def _evaluate_content_state_signal(self, messages: List) -> Dict[str, Any]:
        """用现有内容状态追踪器判断最新目标内容是否值得继续处理。"""
        result = {
            "should_skip": False,
            "reason": "",
            "decision_reason": "",
            "suggested_action": "",
            "confidence": 0.0,
            "target_content": "",
        }
        try:
            from src.core.content_state_tracker import (
                get_content_state_tracker,
            )

            target_message = self._get_latest_human_message(messages)
            if target_message is None:
                return result
            content = self._extract_message_content(target_message).strip()
            if not content:
                return result
            tracker = get_content_state_tracker()
            decision = tracker.should_process(
                content=content,
                channel_id=self.stream_id,
                user_id=getattr(target_message, "user_id", ""),
            )
            result.update(
                {
                    "decision_reason": decision.reason,
                    "suggested_action": decision.suggested_action,
                    "confidence": float(decision.confidence or 0.0),
                    "target_content": content,
                }
            )
            if decision.should_process:
                return result
            reason_map = {
                "low_interest": "当前内容信息量偏低，先不重复接话",
                "currently_processing": "当前内容正在处理中，避免并行重复回复",
                "over_processed": "这类内容处理次数过多，保持克制",
                "already_well_replied": "这类内容刚处理过且回应质量足够",
            }
            result["should_skip"] = True
            result["reason"] = reason_map.get(
                decision.reason,
                f"内容状态追踪建议跳过: {decision.reason}",
            )
            return result
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 内容状态判定失败: {exc}")
            return result

    def _apply_content_state_skip(self, content_state_signal: Optional[Dict[str, Any]]) -> None:
        """把内容状态拒绝结果回写给追踪器，避免同一类低价值内容重复进入主链。"""
        if not content_state_signal:
            return
        content = str(content_state_signal.get("target_content", "") or "").strip()
        if not content:
            return
        reason = str(content_state_signal.get("decision_reason", "") or "")
        try:
            from src.core.content_state_tracker import (
                IgnoreReason,
                get_content_state_tracker,
            )

            tracker = get_content_state_tracker()
            if reason == "low_interest":
                tracker.mark_ignored(content, IgnoreReason.LOW_INTEREST)
            elif reason == "over_processed":
                tracker.mark_ignored(content, IgnoreReason.OVER_PROCESSED)
            elif reason == "already_well_replied":
                tracker.mark_ignored(content, IgnoreReason.ALREADY_REPLIED)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 回写内容状态失败: {exc}")

    def _extract_message_content(self, msg) -> str:
        if msg is None:
            return ""
        return (
            getattr(msg, "processed_plain_text", "")
            or getattr(msg, "raw_plain_text", "")
            or getattr(msg, "content", "")
            or ""
        )

    def _get_latest_human_message(
        self,
        messages: List,
        preferred_sources: Tuple[str, ...] = ("incoming",),
        allow_fallback: bool = True,
    ):
        human_messages = self._get_human_message_candidates(
            list(messages[-10:]),
            preferred_sources=preferred_sources,
            allow_fallback=allow_fallback,
        )
        return human_messages[-1] if human_messages else None

    def _mark_message_content_processing(self, msg) -> None:
        content = self._extract_message_content(msg).strip()
        if not content:
            return
        try:
            from src.core.content_state_tracker import (
                get_content_state_tracker,
            )

            get_content_state_tracker().mark_processing(content)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 标记内容处理中失败: {exc}")

    def _mark_message_content_processed(
        self,
        msg,
        action_name: str,
        quality: float = 0.5,
    ) -> None:
        content = self._extract_message_content(msg).strip()
        if not content:
            return
        try:
            from src.core.content_state_tracker import (
                get_content_state_tracker,
            )

            get_content_state_tracker().mark_processed(
                content,
                action=action_name,
                quality=quality,
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 标记内容已处理失败: {exc}")

    def _mark_message_content_deferred(self, msg, reason: str) -> None:
        content = self._extract_message_content(msg).strip()
        if not content:
            return
        try:
            from src.core.content_state_tracker import (
                get_content_state_tracker,
            )

            get_content_state_tracker().mark_deferred(content, reason)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 标记内容延后失败: {exc}")

    async def _analyze_group_sense(self, messages: List) -> Dict[str, Any]:
        """
        群聊感知分析 - 分析群聊态势（爆发、争议、沉默等）

        返回：
        - activity_level: 活跃度等级（死寂/冷清/平稳/活跃/爆发）
        - burst_detected: 是否检测到消息爆发
        - controversy_detected: 是否检测到争议
        - dominant_users: 主导用户列表
        - needs_topic: 是否需要新话题
        - description: 态势描述
        """
        try:
            from src.modules.perception.group_sense import get_group_sense

            group_sense = get_group_sense()
            recent_messages = []
            for msg in messages[-50:]:
                if not self._is_human_message_obj(msg):
                    continue
                recent_messages.append(
                    {
                        "sender_id": getattr(msg, "user_id", ""),
                        "timestamp": getattr(msg, "timestamp", 0.0),
                        "text": getattr(msg, "processed_plain_text", "") or getattr(msg, "content", ""),
                    }
                )
            result = group_sense.analyze(
                stream_id=self.stream_id,
                recent_messages=recent_messages,
                last_reply_time=self.last_active_time,
            )
            if result.burst_detected or result.controversy_detected:
                logger.info(f"{self.log_prefix} 👁️ {result.description}")
            self._last_group_sense = result
            return self._merge_group_context_signal(
                {
                    "activity_level": result.activity_level,
                    "message_count_5min": result.message_count_5min,
                    "active_user_count": result.active_user_count,
                    "silence_duration_seconds": result.silence_duration_seconds,
                    "burst_detected": result.burst_detected,
                    "burst_density": result.burst_density,
                    "controversy_detected": result.controversy_detected,
                    "dominant_users": result.dominant_users,
                    "needs_topic": result.needs_topic,
                    "topic_hints": result.topic_hints,
                    "description": result.description,
                }
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 群聊感知失败: {exc}")
            return self._merge_group_context_signal(
                {
                    "activity_level": "未知",
                    "burst_detected": False,
                    "controversy_detected": False,
                    "needs_topic": False,
                    "topic_hints": [],
                }
            )

    async def _analyze_message_preprocessor(self, messages: List) -> Dict[str, Any]:
        """把消息预处理器输出转成主链可直接消费的刷屏与活跃信号。"""
        result = {
            "is_spam": False,
            "spam_type": "正常",
            "activity_level": "平稳",
            "unique_users": 0,
            "has_at_me": False,
        }
        try:
            from src.modules.perception.message_preprocessor import (
                StandardMessage,
                get_message_preprocessor,
            )

            standardized = []
            for msg in messages[-20:]:
                standardized.append(
                    StandardMessage(
                        message_id=str(
                            getattr(msg, "message_id", "") or getattr(msg, "msg_id", "") or getattr(msg, "id", "")
                        ),
                        text=self._extract_message_content(msg),
                        sender_id=getattr(msg, "user_id", "") or "",
                        sender_name=(
                            getattr(msg, "user_nickname", "")
                            or getattr(msg, "user_name", "")
                            or getattr(msg, "nickname", "")
                            or "未知用户"
                        ),
                        timestamp=float(getattr(msg, "timestamp", 0.0) or 0.0),
                        is_at_me=bool(getattr(msg, "is_at_me", False) or getattr(msg, "is_at", False)),
                        has_image=bool(getattr(msg, "has_image", False) or getattr(msg, "image_urls", None)),
                        image_urls=list(getattr(msg, "image_urls", []) or []),
                        is_bot_self=self._is_bot_message_obj(msg),
                        content_type=str(getattr(msg, "content_type", "text") or "text"),
                        is_quote_reply=bool(getattr(msg, "is_quote_reply", False)),
                        quoted_content=str(getattr(msg, "quoted_content", "") or ""),
                        quoted_sender=str(getattr(msg, "quoted_sender", "") or ""),
                        is_forward=bool(getattr(msg, "is_forward", False)),
                        forward_title=str(getattr(msg, "forward_title", "") or ""),
                        forward_items=list(getattr(msg, "forward_items", []) or []),
                    )
                )
            batch = get_message_preprocessor().process(standardized)
            meta = batch.meta
            result.update(
                {
                    "is_spam": bool(meta.is_spam),
                    "spam_type": meta.spam_type,
                    "activity_level": meta.activity_level,
                    "unique_users": int(meta.unique_users),
                    "has_at_me": bool(meta.has_at_me),
                }
            )
            if meta.is_spam:
                logger.info(f"{self.log_prefix} 🚫 检测到{meta.spam_type}")
            return result
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 消息预处理分析失败: {exc}")
            return result

    def _filter_messages_by_user_preference(self, messages: List) -> List:
        """
        用户消息过滤 - 识别刷屏用户和不感兴趣的用户

        过滤规则：
        1. 刷屏用户：短时间内发送大量消息的用户
        2. 不感兴趣的用户：被打上"不感兴趣"标签的用户
        3. 骚扰用户：厌烦值过高的用户
        """
        try:
            from src.modules.social_value.social_affect_fuser import (
                get_social_affect_fuser,
            )

            filtered = []
            user_msg_count = {}
            spam_threshold = 5

            for msg in messages:
                user_id = getattr(msg, "user_id", "")
                if self._is_bot_message_obj(msg):
                    continue

                user_msg_count[user_id] = user_msg_count.get(user_id, 0) + 1

            fuser = get_social_affect_fuser()

            for msg in messages:
                user_id = getattr(msg, "user_id", "")
                if self._is_bot_message_obj(msg):
                    filtered.append(msg)
                    continue

                msg_count = user_msg_count.get(user_id, 0)
                if msg_count > spam_threshold:
                    if not hasattr(self, "_spam_warned_users"):
                        self._spam_warned_users: dict = {}
                    now_ts = time.time()
                    # 清理超过 1 小时的记录
                    if len(self._spam_warned_users) > 200:
                        self._spam_warned_users = {
                            k: v for k, v in self._spam_warned_users.items() if now_ts - v < 3600
                        }
                    if user_id not in self._spam_warned_users:
                        logger.info(f"{self.log_prefix} 🚫 用户 {user_id[:8]} 刷屏 ({msg_count}条)，降低响应优先级")
                        self._spam_warned_users[user_id] = now_ts
                    try:
                        from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

                        self._spawn(
                            acquire_reply_coordinator().cancel_followup_tasks(
                                self.stream_id,
                                reason=f"spam_user={user_id[:8]} count={msg_count}",
                            ),
                            name="cancel_followups_on_spam",
                        )
                    except Exception as _cancel_exc:
                        logger.debug(f"{self.log_prefix} 刷屏取消补充任务异常: {_cancel_exc}")
                    if self._proactive_task and not self._proactive_task.done():
                        self._proactive_task.cancel()
                        logger.info(f"{self.log_prefix} 🚫 检测到刷屏，取消当前主动后台任务")
                    msg._spam_user = True

                dossier = fuser.get_dossier(user_id, self.stream_id)
                if dossier and dossier.custom_nick:
                    negative_labels = [
                        "不感兴趣",
                        "无聊",
                        "讨厌",
                        "骚扰",
                        "烦人",
                        "屏蔽",
                    ]
                    if any(nl in dossier.custom_nick for nl in negative_labels):
                        logger.debug(f"{self.log_prefix} 🚫 跳过不感兴趣用户: {user_id[:8]} ({dossier.custom_nick})")
                        continue

                filtered.append(msg)

            return filtered
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 用户过滤失败: {exc}")
            return messages

    def _get_user_attention_distribution(self, messages: List) -> Dict[str, float]:
        """
        计算用户注意力分配 - 群聊中平均分配注意力

        返回每个用户应该获得的注意力权重
        """
        try:
            user_activity = {}
            now = time.time()
            window = 300.0

            for msg in messages:
                user_id = getattr(msg, "user_id", "")
                if self._is_bot_message_obj(msg):
                    continue
                ts = getattr(msg, "timestamp", 0.0)
                if now - ts <= window:
                    user_activity[user_id] = user_activity.get(user_id, 0) + 1

            if not user_activity:
                return {}

            total_msgs = sum(user_activity.values())
            user_count = len(user_activity)

            distribution = {}
            for user_id, count in user_activity.items():
                activity_ratio = count / total_msgs
                fairness_bonus = 1.0 / user_count
                weight = 0.7 * fairness_bonus + 0.3 * activity_ratio
                distribution[user_id] = weight

            total_weight = sum(distribution.values())
            if total_weight > 0:
                for user_id in distribution:
                    distribution[user_id] /= total_weight

            return distribution
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 注意力分配计算失败: {exc}")
            return {}

    async def _update_user_interaction_styles(self, messages: List) -> None:
        """
        更新用户交互风格 - 使用 LLM 分析用户风格并更新到 UserImpression

        直接使用现有的 UserImpression.interaction_style 字段，避免孤岛代码
        """
        try:
            from src.modules.social_value.social_affect_fuser import (
                get_social_affect_fuser,
            )

            fuser = get_social_affect_fuser()
            user_messages = {}
            for msg in messages:
                user_id = getattr(msg, "user_id", "")
                content = getattr(msg, "processed_plain_text", "") or getattr(msg, "content", "")
                if user_id and not self._is_bot_message_obj(msg) and content:
                    if user_id not in user_messages:
                        user_messages[user_id] = []
                    user_messages[user_id].append(content)

            for user_id, msg_list in user_messages.items():
                recent_msgs = msg_list[-5:]
                try:
                    style_desc = await asyncio.wait_for(
                        self._analyze_user_style_by_llm(user_id, recent_msgs),
                        timeout=20.0,
                    )
                except asyncio.TimeoutError:
                    logger.debug(f"{self.log_prefix} 用户风格分析超时(20s)，跳过用户{user_id[:8]}")
                    continue
                if style_desc:
                    await fuser.refresh_impression(
                        user_id=user_id,
                        channel_id=self.stream_id,
                        interaction_digest="；".join(recent_msgs[-3:]),
                        style=style_desc,
                    )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 用户风格更新失败: {exc}")

    async def _analyze_user_style_by_llm(self, user_id: str, messages: List[str]) -> str:
        """使用 LLM 分析用户的回复风格"""
        if not messages:
            return ""

        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            llm = LLMRequest(
                model_set=model_config.model_task_config.utils,
                request_type="user_style_analysis",
            )

            messages_text = "\n".join([f"- {msg}" for msg in messages[-5:]])

            prompt = f"""请分析以下用户的回复风格特征。

用户最近的消息：
{messages_text}

请从以下维度分析用户的风格，并用 JSON 格式输出：

1. length_preference: 长度偏好（极简/简洁/适中/详细/啰嗦）
2. tone: 语气风格（正式/随意/幽默/严肃/撒娇/高冷/温和/活泼）
3. personality: 性格特点（如：话少、话痨、爱玩梗、可爱、冷淡、阳光、负能量等，列出2-3个）
4. reply_suggestion: 对回复这个用户的建议（一句话）

请直接输出 JSON，不要有其他内容：
{{"length_preference": "...", "tone": "...", "personality": ["...", "..."], "reply_suggestion": "..."}}"""

            response, _ = await llm.generate_response_async(prompt)

            import json
            from json_repair import repair_json

            try:
                result = repair_json(response)
                if isinstance(result, str):
                    result = json.loads(result)
            except Exception:
                result = {}

            if not isinstance(result, dict):
                result = {}

            length = result.get("length_preference", "适中")
            tone = result.get("tone", "随意")
            personality = result.get("personality", [])
            suggestion = result.get("reply_suggestion", "")

            style_desc = f"{length}、{tone}"
            if personality:
                style_desc += f"，特点：{'、'.join(personality[:3])}"
            if suggestion:
                style_desc += f" | 建议：{suggestion}"

            logger.debug(f"{self.log_prefix} 🎨 用户 {user_id[:8]}: {style_desc}")
            return style_desc

        except Exception as e:
            logger.debug(f"{self.log_prefix} LLM 风格分析失败: {e}，使用规则分析")
            return self._analyze_user_style_by_rules(messages[-1] if messages else "")

    def _analyze_user_style_by_rules(self, message: str) -> str:
        """规则分析用户风格（LLM 失败时的保守降级）。"""
        if not message:
            return ""

        msg_length = len(message)

        length_pref = "适中"
        if msg_length <= 10:
            length_pref = "极简"
        elif msg_length <= 25:
            length_pref = "简洁"
        elif msg_length <= 50:
            length_pref = "适中"
        elif msg_length <= 100:
            length_pref = "详细"
        else:
            length_pref = "啰嗦"

        tone = "中性"
        exclaim_count = message.count("!") + message.count("！")
        question_count = message.count("?") + message.count("？")
        if exclaim_count >= 2:
            tone = "外放"
        elif question_count >= 2:
            tone = "探询"
        elif msg_length <= 5:
            tone = "克制"
        elif msg_length >= 60:
            tone = "展开"

        tags = []
        if length_pref == "极简":
            tags.append("话少")
        elif length_pref == "啰嗦":
            tags.append("话痨")
        if tone == "外放":
            tags.append("表达强")
        elif tone == "探询":
            tags.append("追问多")
        elif tone == "克制":
            tags.append("回应短")

        style_desc = f"{length_pref}、{tone}"
        if tags:
            style_desc += f"，特点：{'、'.join(tags)}"

        return style_desc

    def _get_user_style_guide(self, user_id: str) -> str:
        """获取当前这位的表达风格指导，用于回复生成"""
        try:
            from src.modules.social_value.social_affect_fuser import (
                get_social_affect_fuser,
            )

            fuser = get_social_affect_fuser()
            dossier = fuser.get_dossier(user_id, self.stream_id)
            if dossier and dossier.impression and dossier.impression.dialogue_style:
                style = dossier.impression.dialogue_style
                guide_parts = []

                if "极简" in style:
                    guide_parts.append("对方偏好极简回复，回复要非常简短，1-2句话即可")
                elif "简洁" in style:
                    guide_parts.append("对方偏好简洁回复，控制在2-3句话")
                elif "啰嗦" in style:
                    guide_parts.append("对方偏好详细回复，可以展开说")

                if "正式" in style:
                    guide_parts.append("语气要正式、礼貌")
                elif "幽默" in style:
                    guide_parts.append("可以适当幽默、玩梗")
                elif "撒娇" in style:
                    guide_parts.append("可以适当卖萌、撒娇")
                elif "高冷" in style:
                    guide_parts.append("回复简洁，不要太热情")

                if guide_parts:
                    return f"[对方表达风格] {style}。回复建议：{'；'.join(guide_parts)}"
            return ""
        except Exception as _exc:
            logger.warning(f"{self.log_prefix} 风格指南构建异常: {_exc}")
            return ""

    def _build_dynamic_length_hint(self, target_message: Any, user_style_guide: str = "") -> str:
        """根据用户输入长度和互动风格构建动态句长约束。"""
        user_text = ""
        if target_message is not None:
            user_text = (
                getattr(target_message, "processed_plain_text", "") or getattr(target_message, "content", "") or ""
            )
        text_len = len((user_text or "").strip())
        if text_len <= 8:
            low, high = 1, 3
        elif text_len <= 24:
            low, high = 2, 5
        elif text_len <= 60:
            low, high = 3, 8
        else:
            low, high = 4, 12

        style = user_style_guide or ""
        if "极简" in style or "高冷" in style:
            low, high = 1, min(3, high)
        elif "简洁" in style:
            low, high = min(low, 2), min(4, high)
        elif "啰嗦" in style or "详细" in style:
            low, high = max(4, low), max(8, high)

        high = min(15, high + random.randint(0, 2))
        low = max(1, min(low, high))
        return (
            f"[回复长度策略] 根据用户输入动态回复，本次控制在{low}~{high}句；短输入偏短，长输入可展开，不要固定句长。"
        )

    def _build_persona_hint(self) -> str:
        """构建主人格提示片段。"""
        parts: List[str] = ["[人格约束] 无论内心独白还是最终回复，都优先服从主人格，不要临时长篇分析，不要偏成旁白。"]
        try:
            from src.person_info.bot_identity import get_bot_identity_manager

            identity = get_bot_identity_manager()
            persona_brief = identity.build_persona_brief(
                self.stream_id,
                include_name_prefix=True,
                include_style=True,
                include_interests=True,
                include_lore=True,
                fallback_text="自然说话，别端着，也别把自己说成系统或工具。",
            )
            if persona_brief:
                parts.append(f"[主人格摘要] {persona_brief}")
            parts.append("[身份边界] 不要把自己说成AI助手、系统、客服或工具；你就是当前聊天现场里的这个人。")
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.persona_engine import (
                get_character_foundry,
            )

            shard = (get_character_foundry().active_prompt_shard() or "").strip()
            if shard:
                parts.append(f"[当前人格片段] {shard[:240]}")
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return "\n".join(parts)

    def _resolve_latest_human_user_id(
        self,
        messages: Optional[List] = None,
        *,
        allow_cached_fallback: bool = True,
    ) -> str:
        """稳定解析当前轮对应的发言对象 ID，避免误用 stream_id 作为 memoir key。"""
        candidates = list(messages or [])
        for msg in reversed(candidates[-20:]):
            uid = str(getattr(msg, "user_id", "") or "").strip()
            if self._is_human_message_obj(msg):
                return uid
        if allow_cached_fallback:
            return str(getattr(self, "_last_user_id", "") or "").strip()
        return ""

    async def _invoke_unified_planner(
        self,
        messages: List,
        repetition_signal: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """统一规划器集成 - 让规划器感知当前状态并返回决策"""
        try:
            from src.core.unified_planner import get_unified_planner

            planner = get_unified_planner()
            user_id = ""
            last_user_text = ""
            if messages:
                for msg in reversed(messages):
                    uid = getattr(msg, "user_id", "")
                    if self._is_human_message_obj(msg):
                        user_id = uid
                        last_user_text = getattr(msg, "processed_plain_text", "") or getattr(msg, "content", "") or ""
                        break

            repeated_topic_pressure = 0.0
            if repetition_signal:
                repeat_count = float(repetition_signal.get("exact_repeat_count", 0.0) or 0.0)
                repeated_topic_pressure = max(0.0, min(1.0, repeat_count / 8.0))
            harassment_signal = self._analyze_harassment_pressure(messages)
            behavior_signal = self._classify_behavior_signal(
                messages,
                repetition_signal=repetition_signal,
                harassment_signal=harassment_signal,
            )
            relation_snapshot = self._resolve_relation_view()

            target_message = self._get_latest_human_message(messages)
            context_execution_block = self._build_context_execution_block(
                target_message=target_message,
                voice_conclusion=getattr(self, "_cached_voice", None),
                repetition_signal=repetition_signal,
                relation_snapshot=relation_snapshot,
            )
            try:
                planner_mode_summary = "unknown/unknown"
                if user_id:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker,
                    )

                    response_mode = get_emotion_tracker(self.stream_id).get_layered_response_mode(user_id)
                    planner_mode_summary = (
                        f"{str(response_mode.get('tone', 'neutral') or 'neutral')}"
                        f"/{str(response_mode.get('response_length', 'normal') or 'normal')}"
                    )
                logger.info(
                    f"{self.log_prefix} 规划输入摘要 "
                    f"对象={user_id[:8] if user_id else 'none'} "
                    f"重复={int(repetition_signal.get('exact_repeat_count', 0) or 0) if repetition_signal else 0} "
                    f"前情块={'有' if context_execution_block else '无'} "
                    f"压力={repeated_topic_pressure:.2f} "
                    f"模式={planner_mode_summary}"
                )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            # 复用 inner_voice 阶段构建的统一快照，避免同一 tick 重复采集
            from src.core.world_snapshot import build_world_snapshot

            _cached = getattr(self, "_tick_world_snapshot", None)
            if _cached is not None:
                pre_built_snapshot = _cached
                self._tick_world_snapshot = None
            else:
                pre_built_snapshot = await build_world_snapshot(self.stream_id, user_id)
            # 把增强链独有的行为信号和重复压力注入 snapshot
            pre_built_snapshot.behavior_category = str((behavior_signal or {}).get("category", "neutral") or "neutral")
            pre_built_snapshot.behavior_severity = float((behavior_signal or {}).get("severity", 0.0) or 0.0)
            pre_built_snapshot.behavior_reason = str((behavior_signal or {}).get("reason", "") or "")
            pre_built_snapshot.repeated_topic_pressure = repeated_topic_pressure
            pre_built_snapshot.is_new_user = bool((behavior_signal or {}).get("is_new_user", False))
            # 从意图池获取活跃意图摘要供规划器参考
            _intention_hint = ""
            _active_intentions: List[Dict[str, Any]] = []
            try:
                from src.chat.proactive.intention_pool import (
                    get_intention_pool,
                )

                _pool = get_intention_pool()
                _intention_hint = _pool.build_planner_hint(self.stream_id)
                _active_intentions = _pool.build_planner_intent_payload(self.stream_id, limit=3)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            _is_admin_call = bool(messages and self._is_force_wake_admin(messages))
            if _is_admin_call:
                decision = PlanningDecision(
                    action=ActionType.REPLY,
                    reason="管理员消息，极速通道跳过规划器",
                    confidence=1.0,
                    source="admin_fastlane",
                )
                logger.info(f"{self.log_prefix} 👑 管理员极速通道: 跳过规划器LLM调用")
                return decision
            decision = await planner.plan(
                channel_id=self.stream_id,
                user_id=user_id,
                trigger="message_arrival",
                is_admin=bool(messages and self._is_force_wake_admin(messages)),
                hints={
                    "repeated_topic_pressure": repeated_topic_pressure,
                    "last_user_intent": self._build_planner_user_hint(last_user_text, repetition_signal),
                    "memory_hint": str(getattr(self, "_latest_memory_hint", "") or ""),
                    "context_execution_block": context_execution_block,
                    "intention_hint": _intention_hint,
                    "active_intentions": _active_intentions,
                    "active_users": [
                        str(getattr(msg, "user_id", "") or "").strip()
                        for msg in messages[-20:]
                        if str(getattr(msg, "user_id", "") or "").strip() and not self._is_bot_message_obj(msg)
                    ],
                    "current_topics": self._extract_current_topics(messages),
                    "relationship_candidates": relation_snapshot.get("relationship_candidates", []),
                    "group_pattern_hint": self._build_group_pattern_hint(),
                },
                world_snapshot=pre_built_snapshot,
            )
            if decision and decision.action.value != "observe":
                logger.info(
                    f"{self.log_prefix} 规划器决策={decision.action.label()}, "
                    f"原因={decision.reason[:30] if decision.reason else '无'}"
                )
            return decision
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 统一规划器调用失败: {exc}")
            return None

    def _build_group_pattern_hint(self) -> str:
        """F12：将群体行为模式转化为规划器可理解的提示文本"""
        if not self._cached_pattern_evidence:
            return ""
        _top = None
        _top_conf = 0.0
        for _pe in self._cached_pattern_evidence:
            _pc = float(getattr(_pe, "confidence", 0.0) or 0.0)
            if _pc > _top_conf:
                _top_conf = _pc
                _top = _pe
        if _top is None or _top_conf < 0.35:
            return ""
        _pt_val = getattr(getattr(_top, "pattern", None), "value", "") or ""
        _label = getattr(_top.pattern, "label", "") if hasattr(_top, "pattern") else _pt_val
        _hint_map = {
            "newcomer_welcome": "当前群聊正在迎新，适合主动打招呼或表示欢迎。",
            "ritual_greeting": "群里有礼仪性问候在进行，可以自然接一句。",
            "spectator_mode": "群里正在围观某事，不适合主动插话，更适合旁观。",
            "celebration_wave": "群里在庆祝什么，氛围好，可以参与但不抢风头。",
            "support_circle": "群里有人在寻求支持，如果关系够近可以考虑回应。",
            "heated_discussion": "群里讨论很激烈，除非被@否则不要乱插嘴。",
            "argument": "群里在吵架，不要卷进去。",
            "conflict_escalation": "冲突升级中，绝对不要介入。",
            "meme_storm": "群里在玩梗/刷表情包风暴，可以凑热闹但别太认真。",
            "emotional_contagion": "群里情绪在传染，注意不要被带偏。",
            "sudden_silence": "群里突然安静了，可能是好事也可能是尴尬时机。",
            "copycat_chain": "群里在复读，可以跟但不一定要每条都跟。",
            "collective_storytelling": "群里在讲故事，适合安静听或适时反应。",
        }
        return _hint_map.get(_pt_val, f"检测到群体模式: {_label}")

    def _build_planner_user_hint(
        self,
        last_user_text: str,
        repetition_signal: Optional[Dict[str, Any]] = None,
    ) -> str:
        text = str(last_user_text or "").strip()
        if not text:
            return ""
        prior_memory_hint = ""
        try:
            from src.chat.proactive.session_tracker import get_memoir_cabinet

            latest_user_id = str(getattr(self, "_last_user_id", "") or "").strip()
            if latest_user_id:
                memoir = get_memoir_cabinet().retrieve(latest_user_id)
                if memoir is not None:
                    last_topic = str(getattr(memoir, "last_topic", "") or "").strip()
                    if last_topic:
                        prior_memory_hint = f"你们之前已经聊到过{last_topic[:40]}。"
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        if repetition_signal and repetition_signal.get("latest_matches_repeat"):
            reason = str(repetition_signal.get("reason", "重复输入") or "重复输入")
            return (
                f"最近对方原话：{text[:80]}；这是重复短句，不要当成全新话题。"
                f"优先结合你之前已经问过的上下文继续接。{prior_memory_hint}{reason[:60]}"
            )
        if len(self._normalize_repeat_text(text)) <= 6:
            return f"最近对方原话：{text[:80]} ；这是很短的线索，先结合前情判断，不要硬当新话题。{prior_memory_hint} "
        return f"最近对方原话：{text[:120]}"

    async def _store_interaction_memory(self, messages: List) -> None:
        """存储交互记忆 - 将交互存入记忆系统"""
        try:
            from src.memory_system.memory_core import acquire_recollection_hub

            hub = acquire_recollection_hub()
            for msg in messages[-3:]:
                content = getattr(msg, "processed_plain_text", "") or getattr(msg, "content", "")
                if not content:
                    continue
                user_id = getattr(msg, "user_id", "unknown")
                is_bot = self._is_bot_message_obj(msg)
                hub.deposit_memory(
                    stream_id=self.stream_id,
                    content=content,
                    entry_category=("bot_response" if is_bot else "conversation"),
                    user_id=user_id,
                    significance=0.5,
                )
            logger.debug(f"{self.log_prefix} 已存储 {len(messages[-3:])} 条交互记忆")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 存储交互记忆失败: {exc}")

    async def _run_dimension_gateway(
        self,
        decision_messages: list,
        pinged_msg,
        voice_conclusion,
        relation_result: dict,
        repetition_signal: dict | None = None,
        harassment_signal: dict | None = None,
    ) -> dict | None:
        """
        调用多维状态调度器收集投票，然后通过决策网关裁定。
        返回兼容旧格式的 dict，或在调度器不可用时返回 None。
        裁定结果同时缓存在 self._last_gateway_verdict 供后续使用。
        """
        self._last_gateway_verdict = None
        try:
            from src.chat.heart_flow.dimension_dispatcher import DimensionDispatcher
            from src.chat.heart_flow.dimension_protocol import EventContext
            from src.chat.heart_flow.decision_gateway import (
                DecisionGateway,
                GatewayContext,
                gateway_verdict_to_compat,
            )
        except ImportError as ie:
            logger.debug(f"{self.log_prefix} 维度网关模块不可用: {ie}")
            return None
        try:
            dispatcher = await DimensionDispatcher.get_instance()
            if dispatcher.registered_count == 0:
                return None
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 获取调度器失败: {exc}")
            return None
        # 提取最近一条非bot消息的文本和用户ID
        latest_text = ""
        latest_uid = ""
        latest_msg = self._get_latest_human_message(decision_messages)
        if latest_msg is not None:
            latest_uid = str(getattr(latest_msg, "user_id", "") or "")
            latest_text = str(
                getattr(latest_msg, "processed_plain_text", "")
                or getattr(latest_msg, "plain_text", "")
                or getattr(latest_msg, "content", "")
                or ""
            ).strip()
        # 计算情感倾向
        sentiment = 0.0
        if relation_result:
            affection = float(relation_result.get("affection", 0.0) or 0.0)
            annoyance = float(relation_result.get("annoyance_value", 0.0) or 0.0)
            sentiment = max(-1.0, min(1.0, (affection - annoyance) / 100.0))
        _recent_human_msgs = self._get_human_message_candidates(list(decision_messages[-10:]))
        _has_targeted_bot = self._has_targeted_bot_message(_recent_human_msgs)
        _is_reply_to = any(
            bool(getattr(m, "is_reply_to_bot", False) or getattr(m, "is_quote_to_bot", False))
            or self._message_explicitly_replies_to_bot(m)
            for m in _recent_human_msgs
        )
        _has_indirect_mention = any(
            self._message_targets_bot(m) and not self._is_message_pinged(m) and not _is_reply_to
            for m in _recent_human_msgs
        )
        # 构建事件上下文并广播
        evt = EventContext(
            event_type="decision_phase",
            user_id=latest_uid,
            channel_id=self.stream_id,
            message_text=latest_text,
            message_length=len(latest_text),
            is_at_bot=bool(pinged_msg is not None or _has_targeted_bot),
            is_reply_to_bot=_is_reply_to,
            sentiment_score=sentiment,
        )
        try:
            await dispatcher.broadcast_event(evt)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 维度广播异常: {exc}")
        # 收集所有维度投票
        try:
            votes = dispatcher.collect_votes(evt)
        except Exception as exc:
            logger.warning(f"{self.log_prefix} 维度投票收集失败: {exc}")
            return None
        # 构建网关上下文
        desire_level = 5
        voice_should_reply = None
        if voice_conclusion is not None:
            desire_level = int(getattr(voice_conclusion, "reply_desire_level", 5) or 5)
            _voice_reply_raw = getattr(voice_conclusion, "should_reply", None)
            if _voice_reply_raw is not None:
                voice_should_reply = bool(_voice_reply_raw)
        gw_ctx = GatewayContext(
            is_direct_ping=(pinged_msg is not None),
            has_indirect_mention=_has_indirect_mention,
            is_reply_to_bot=_is_reply_to,
            desire_level=desire_level,
            voice_should_reply=voice_should_reply,
            admin_force=self._is_force_wake_admin(decision_messages, pinged_msg),
            message_text=latest_text,
            user_id=latest_uid,
            channel_id=self.stream_id,
        )
        # 执行决策网关裁定
        gateway = DecisionGateway.get_instance()
        verdict = gateway.decide(votes, gw_ctx)
        self._last_gateway_verdict = verdict
        if not verdict.should_skip:
            try:
                from src.chat.heart_flow.flow_planner import acquire_flow_planner

                _recent_text = latest_text[:200] if latest_text else ""
                self._last_reactive_plan = await acquire_flow_planner().generate_reactive_plan(
                    channel_id=self.stream_id,
                    recent_messages=_recent_text,
                )
            except Exception as _rp_exc:
                logger.debug(f"{self.log_prefix} 被动策略规划异常: {_rp_exc}")
                self._last_reactive_plan = None
        else:
            self._last_reactive_plan = None
        logger.info(
            f"{self.log_prefix} [维度网关] gate={verdict.gate} "
            f"prob={verdict.final_probability:.3f} "
            f"skip={verdict.should_skip} src={verdict.decision_source} "
            f"tags={verdict.attitude_tags[:3]} styles={verdict.style_hints[:3]}"
        )
        # 转换为兼容旧格式
        compat = gateway_verdict_to_compat(verdict)
        result = compat.to_dict()
        result["_gateway_verdict"] = verdict
        return result

    def _inject_dimension_state_prompt(self, extra_parts: list) -> None:
        """
        将维度系统的LLM状态提示词注入回复生成的extra_parts。
        需要在 extra_parts 组装阶段调用。
        当维度网关被跳过时（如管理员极速通道），从 emotion_tracker 构建简版灵魂数据。
        """
        verdict = getattr(self, "_last_gateway_verdict", None)
        if verdict is None:
            # 维度网关未执行，从 emotion_tracker 直接构建简版灵魂数据
            self._inject_fallback_soul_state(extra_parts)
            return
        _has_primary_soul_state = bool(
            str(getattr(verdict, "llm_state_prompt", "") or "").strip()
            or list(getattr(verdict, "perception_labels", []) or [])
            or str(getattr(verdict, "inner_conflict", "") or "").strip()
        )
        # 注入状态描述
        if verdict.llm_state_prompt:
            extra_parts.append(verdict.llm_state_prompt)
        # 注入风格修饰
        if verdict.style_hints:
            hints_text = "；".join(verdict.style_hints[:5])
            extra_parts.append(f"[维度风格指引] {hints_text}")
        # 注入态度标签
        if verdict.attitude_tags:
            tags_text = "、".join(verdict.attitude_tags[:5])
            extra_parts.append(f"[维度态度标签] {tags_text}")
        # 面具质量上限警告
        if verdict.quality_cap < 0.8:
            extra_parts.append(
                f"[面具约束] 当前回复质量上限={verdict.quality_cap:.0%}，措辞可能不够自然，偶尔流露真实想法"
            )
        # 创伤潜意识干扰
        if verdict.perception_labels:
            labels = "；".join(verdict.perception_labels[:3])
            extra_parts.append(f"[潜意识干扰] {labels}")
        # 内心冲突注入
        if verdict.inner_conflict:
            extra_parts.append(f"[内心冲突] {verdict.inner_conflict}")
        if not _has_primary_soul_state:
            # verdict 仅有风格/态度标签时，也补一层基础情绪画像，避免 extra_info 空心
            self._inject_fallback_soul_state(extra_parts)
        _reactive_plan = getattr(self, "_last_reactive_plan", None)
        if _reactive_plan is not None:
            _rp_parts = []
            if getattr(_reactive_plan, "reply_strategy", ""):
                _rp_parts.append(f"策略={_reactive_plan.reply_strategy}")
            if getattr(_reactive_plan, "emotion_hint", ""):
                _rp_parts.append(f"情绪={_reactive_plan.emotion_hint}")
            if getattr(_reactive_plan, "suggested_length", 0) > 0:
                _rp_parts.append(f"建议长度≤{_reactive_plan.suggested_length}字")
            if _rp_parts:
                extra_parts.append(f"[被动回复规划] {'; '.join(_rp_parts)}")
        # 印象行为映射注入：从印象演化中枢提取对该用户的定制回复策略
        _target_uid = str(getattr(self, "_last_user_id", "") or "").strip()
        if _target_uid:
            try:
                from src.core.impression_evolution_hub import get_impression_hub

                _bh = get_impression_hub(self.stream_id).get_behavior_hints(_target_uid)
                _bh_parts = []
                _tone = _bh.get("preferred_tone", "")
                if _tone and _tone != "自然":
                    _bh_parts.append(f"语气={_tone}")
                _humor = _bh.get("humor_level", 0.5)
                if _humor > 0.7:
                    _bh_parts.append("适当幽默")
                elif _humor < 0.3:
                    _bh_parts.append("保持正经")
                _formality = _bh.get("formality_level", 0.5)
                if _formality > 0.7:
                    _bh_parts.append("措辞正式")
                elif _formality < 0.3:
                    _bh_parts.append("措辞随意")
                _proactive = _bh.get("proactive_degree", 0.3)
                if _proactive > 0.6:
                    _bh_parts.append("可以主动延伸话题")
                if _bh_parts:
                    extra_parts.append(f"[印象行为映射] {'; '.join(_bh_parts)}")
            except Exception:
                pass

    def _inject_fallback_soul_state(self, extra_parts: list) -> None:
        """维度网关未执行时的回退灵魂数据注入（管理员极速通道等场景）"""
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            target_uid = str(getattr(self, "_last_user_id", "") or "").strip()
            if not target_uid:
                extra_parts.append("[当前心理状态] 烦躁度0，回复平静。")
                extra_parts.append("[当前情感状态] 感到平静")
                return
            tracker = get_emotion_tracker(self.stream_id)
            state = tracker.get_user_state(target_uid, create_if_missing=False)
            relation_view = self._resolve_relation_view()
            if not state:
                annoyance = float(relation_view.get("annoyance_value", 0.0) or 0.0)
                pressure = float(relation_view.get("psychological_pressure", 0.0) or 0.0)
                mood_desc = "平静" if annoyance < 10 else ("有些烦躁" if annoyance < 50 else "很不爽")
                extra_parts.append(f"[当前心理状态] 烦躁度{annoyance:.0f}，压力{pressure:.0f}，回复{mood_desc}。")
                extra_parts.append(f"[当前情感状态] {'感到平静' if annoyance < 10 else '有些不耐烦'}")
                return
            annoyance = max(
                float(getattr(state, "annoyance", 0.0) or 0.0),
                float(relation_view.get("annoyance_value", 0.0) or 0.0),
            )
            trauma = max(
                float(getattr(state, "trauma_score", 0.0) or 0.0),
                float(relation_view.get("trauma_score", 0.0) or 0.0),
            )
            affection = float(getattr(state, "affection", 0.0) or 0.0)
            pressure = max(
                float(getattr(state, "psychological_pressure", 0.0) or 0.0),
                float(relation_view.get("psychological_pressure", 0.0) or 0.0),
            )
            if annoyance > 50:
                mood_desc = "很不爽"
            elif annoyance > 20:
                mood_desc = "有些烦躁"
            elif annoyance > 10:
                mood_desc = "略微不耐烦"
            else:
                mood_desc = "平静"
            extra_parts.append(f"[当前心理状态] 烦躁度{annoyance:.0f}，压力{pressure:.0f}，回复{mood_desc}。")
            feeling_parts = []
            if trauma > 5:
                feeling_parts.append("内心痛苦不安")
            if affection > 50:
                feeling_parts.append("心情不错")
            elif affection < -30:
                feeling_parts.append("不太想理对方")
            if annoyance > 50:
                feeling_parts.append("非常烦躁和不耐烦")
            elif annoyance > 20:
                feeling_parts.append("有些不耐烦")
            if not feeling_parts:
                feeling_parts.append("感到平静")
            extra_parts.append(f"[当前情感状态] {'；'.join(feeling_parts)}")
        except Exception:
            extra_parts.append("[当前心理状态] 烦躁度0，回复平静。")
            extra_parts.append("[当前情感状态] 感到平静")

    @staticmethod
    def _soul_prompt_keywords() -> Tuple[str, ...]:
        return (
            "内心独白",
            "情感状态",
            "当前情感状态",
            "当前心理状态",
            "灵魂指令",
            "冷拒模式",
            "烦躁",
            "厌烦",
            "防御",
            "情绪保护",
            "内心冲突",
            "潜意识",
            "你当前的状态",
            "情绪:",
        )

    def _contains_soul_data(self, text: str) -> bool:
        payload = str(text or "")
        if not payload:
            return False
        return any(keyword in payload for keyword in self._soul_prompt_keywords())

    def _ensure_soul_data_in_extra_info(self, extra_info: str) -> str:
        payload = str(extra_info or "").strip()
        if self._contains_soul_data(payload):
            return payload
        fallback_parts: List[str] = []
        self._inject_fallback_soul_state(fallback_parts)
        fallback_lines = [str(line).strip() for line in fallback_parts if str(line).strip()]
        if not fallback_lines:
            fallback_lines = [
                "[当前心理状态] 烦躁度0，回复平静。",
                "[当前情感状态] 感到平静",
            ]
        fallback_block = "\n".join(fallback_lines[:2]).strip()
        if not fallback_block:
            return payload
        return f"{payload}\n{fallback_block}" if payload else fallback_block

    async def _update_emotion_tracker_state(self, messages: List) -> None:
        """更新情绪追踪器状态 - 好感、烦躁、心理压力、创伤值等"""
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            tracker = get_emotion_tracker(self.stream_id)
            for msg in messages:
                user_id = getattr(msg, "user_id", "")
                if not user_id or self._is_bot_message_obj(msg):
                    continue
                content = getattr(msg, "processed_plain_text", "") or getattr(msg, "content", "")
                if not content:
                    continue
                polarity = self._estimate_text_polarity(content)
                if polarity > 0.1:
                    interaction_type = "positive"
                elif polarity < -0.1:
                    interaction_type = "negative"
                else:
                    interaction_type = "neutral"
                tracker.process_interaction(
                    user_id,
                    interaction_type,
                    content=content[:100],
                    magnitude=(min(2.0, abs(polarity) * 3.0) if interaction_type != "neutral" else 1.0),
                )
                # 额外处理：调教进度和创伤累积（process_interaction 不覆盖这些字段）
                state = tracker.get_user_state(user_id, create_if_missing=False)
                if state:
                    if interaction_type == "positive" and state.training_stage > 0:
                        state.training_progress = min(100.0, state.training_progress + 0.5)
                        state.training_resistance = max(0.0, state.training_resistance - 0.3)
                    elif interaction_type == "negative":
                        state.training_resistance = min(100.0, state.training_resistance + 0.5)
                        state.trauma_accumulation = min(10.0, state.trauma_accumulation + 0.1)
                    tracker._save_user_state(user_id)
            logger.debug(f"{self.log_prefix} 情绪追踪器状态已更新")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 情绪追踪器更新失败: {exc}")

    async def _apply_repetition_emotion_feedback(
        self,
        messages: List,
        repetition_signal: Optional[Dict[str, Any]],
    ) -> None:
        """把重复输入压力回写到当前情绪追踪器，形成稳定的烦躁/压力累积。"""
        _use_fallback = False
        if not repetition_signal or not repetition_signal.get("detected"):
            _use_fallback = True
        if not _use_fallback:
            latest_matches_repeat = bool(repetition_signal.get("latest_matches_repeat", False))
            if not latest_matches_repeat:
                _use_fallback = True
        try:
            target_message = self._get_latest_human_message(messages)
            if target_message is None:
                return
            _msg_uid = str(getattr(target_message, "user_id", "") or "").strip()
            _persisted_uid = str(getattr(self, "_last_user_id", "") or "").strip()
            user_id = _persisted_uid or _msg_uid
            if not user_id:
                return
            _now_ts = time.time()
            _last_repeat_write_ts = float(getattr(self, "_last_repeat_write_timestamps", {}).get(user_id, 0.0))
            if (_now_ts - _last_repeat_write_ts) < 90.0:
                return
            exact_repeat_count = 0
            repeat_user_count = 0
            low_info_cluster = False
            _repeat_reason = ""
            if not _use_fallback:
                exact_repeat_count = int(repetition_signal.get("exact_repeat_count", 0) or 0)
                repeat_user_count = int(repetition_signal.get("repeat_user_count", 0) or 0)
                low_info_cluster = bool(repetition_signal.get("low_info_cluster", False))
                _repeat_reason = repetition_signal.get("reason", "repeat")
            else:
                _latest_text = (
                    getattr(target_message, "processed_plain_text", "") or getattr(target_message, "content", "") or ""
                ).strip()
                _normalized = self._normalize_repeat_text(_latest_text)
                exact_repeat_count = self._count_recent_user_repeats(_normalized, user_id)
                if exact_repeat_count >= 2:
                    _repeat_reason = f"连续重复({exact_repeat_count}次)"
                else:
                    return

            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            tracker = get_emotion_tracker(self.stream_id)
            state = tracker.get_user_state(user_id, create_if_missing=True)
            if state is None:
                return

            # 重复消息烦躁增量（温和版：避免少量重复直接拉满）
            annoyance_delta = 0.0
            pressure_delta = 0.0

            # 完全重复消息：每次 +5 烦躁（少量重复可能是网络卡顿或玩梗）
            if exact_repeat_count >= 1:
                annoyance_delta = exact_repeat_count * 5.0
                pressure_delta = min(5.0, exact_repeat_count * 1.5)

            # 同一用户连续多条消息（非完全重复）：每次 +3
            if repeat_user_count >= 2:
                annoyance_delta += repeat_user_count * 3.0
                pressure_delta += min(3.0, repeat_user_count * 1.0)

            # 低信息量消息群
            if low_info_cluster:
                annoyance_delta += 3.0
                pressure_delta += 1.0

            # 单次最大变化不超过 25（防止一次性拉满）
            annoyance_delta = min(25.0, annoyance_delta)

            tracker.update_annoyance(
                user_id,
                annoyance_delta,
                reason=f"重复输入压力: {_repeat_reason}",
            )
            _ts_dict = getattr(self, "_last_repeat_write_timestamps", {})
            if not isinstance(_ts_dict, dict):
                _ts_dict = {}
                self._last_repeat_write_timestamps = _ts_dict
            _ts_dict[user_id] = time.time()
            state.psychological_pressure = max(
                0.0,
                min(
                    100.0,
                    float(getattr(state, "psychological_pressure", 0.0) or 0.0) + pressure_delta,
                ),
            )
            state.last_interaction = time.time()
            tracker._save_user_state(user_id)

            logger.info(
                f"{self.log_prefix} 重复情绪回写 对象={user_id[:8]} 烦躁+{annoyance_delta:.1f} "
                f"压力+{pressure_delta:.1f}"
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 重复情绪回写失败: {exc}")

    async def _update_trauma_system_state(self, messages: List) -> None:
        """更新创伤系统状态 - 内心混乱、伪装强度、压力累积，同时同步 per-user trauma_score"""
        try:
            from src.modules.trauma.trauma_system import get_trauma_system

            trauma_sys = get_trauma_system()
            has_negative = False
            total_intensity = 0.0
            # 收集每个用户的负面强度，用于同步 per-user trauma_score
            per_user_intensity: dict[str, float] = {}
            for msg in messages:
                user_id = getattr(msg, "user_id", "")
                if self._is_bot_message_obj(msg):
                    continue
                content = getattr(msg, "processed_plain_text", "") or getattr(msg, "content", "")
                if not content:
                    continue
                polarity = self._estimate_text_polarity(content)
                if polarity < -0.1:
                    has_negative = True
                    _abs_pol = abs(polarity)
                    total_intensity += _abs_pol
                    if user_id:
                        per_user_intensity[user_id] = per_user_intensity.get(user_id, 0.0) + _abs_pol
            trauma_state = trauma_sys.get_state()
            if has_negative:
                chaos_increase = min(2.0, total_intensity * 0.5)
                trauma_sys.add_stress(chaos_increase)
                context = {
                    "trauma_score": float(getattr(trauma_state, "stress_accumulation", 0.0) or 0.0),
                    "inner_chaos": float(getattr(trauma_state, "inner_chaos_level", 0.0) or 0.0),
                    "sentiment": "negative",
                    "intensity": total_intensity,
                    "social_pressure": trauma_state.stress_accumulation,
                }
                trauma_sys.update_from_context(context)
                # 同步更新 per-user emotion_tracker.trauma_score
                self._sync_per_user_trauma(per_user_intensity)
                logger.debug(
                    f"{self.log_prefix} 内心混乱={trauma_state.inner_chaos_level:.1f}, "
                    f"伪装强度={trauma_state.surface_mask_strength:.1f}, "
                    f"压力累积={trauma_state.stress_accumulation:.1f}"
                )
            else:
                trauma_sys.reduce_stress(0.5)
            logger.debug(f"{self.log_prefix} 创伤系统状态已更新")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 创伤系统更新失败: {exc}")

    def _sync_per_user_trauma(self, per_user_intensity: dict[str, float]) -> None:
        """将负面消息产生的创伤同步写入对应用户的 emotion_tracker.trauma_score"""
        if not per_user_intensity:
            return
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            tracker = get_emotion_tracker(self.stream_id)
            for uid, intensity in per_user_intensity.items():
                # 负面极性 0~1 映射为创伤增量 0~0.6，避免单次消息冲击过猛
                delta = min(0.6, intensity * 0.3)
                if delta > 0.01:
                    tracker.update_trauma(uid, delta, trigger="negative_message")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} per-user创伤同步失败: {exc}")

    async def _sync_memoir_on_message(self, messages: List) -> None:
        """会话追踪同步 — 消息到达时更新 MemoirCabinet 的对话阶段"""
        try:
            from src.chat.proactive.session_tracker import get_memoir_cabinet
            from src.common.data_models.proactive_models import DialogueStage

            cabinet = get_memoir_cabinet()
            memoir_user_id = self._resolve_latest_human_user_id(
                messages,
                allow_cached_fallback=False,
            )
            if not memoir_user_id:
                return
            memoir = cabinet.retrieve(memoir_user_id) if hasattr(cabinet, "retrieve") else None
            if memoir is None:
                if hasattr(cabinet, "create"):
                    memoir = cabinet.create(memoir_user_id, channel_id=self.stream_id)
                else:
                    return
            has_user_msg = any(getattr(m, "user_id", "") and not self._is_bot_message_obj(m) for m in messages)
            if has_user_msg:
                if hasattr(memoir, "phase"):
                    if memoir.phase in (
                        DialogueStage.OPEN,
                        DialogueStage.IDLE,
                        DialogueStage.COOLING,
                    ):
                        memoir.phase = DialogueStage.ACTIVE
                if hasattr(memoir, "last_human_ts"):
                    memoir.last_human_ts = time.time()
            if hasattr(cabinet, "commit"):
                cabinet.commit(self.stream_id, memoir)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 会话追踪同步失败: {exc}")

    def _integrate_deep_visibility_and_freshness(self, incoming_batch: List) -> None:
        """GAP-M+N：深度可见性评分 + 新鲜度注册 + 回看触发评估
        在消息进入感官门控前执行，为后续决策提供三维偏置评分和衰减数据"""
        if not incoming_batch:
            return
        # GAP-M：深度可见性评分已在 _apply_sensory_gates 中完成
        # 这里补充新鲜度注册和回看触发
        try:
            from src.core.freshness_decay_engine import (
                get_freshness_decay_engine,
                DecaySpeedTier,
                VisibilityHistoryEffect,
            )

            _fde = get_freshness_decay_engine(self.stream_id)
            for _msg in incoming_batch:
                _uid_n = str(getattr(_msg, "user_id", "") or "")
                if not _uid_n or self._is_bot_message_obj(_msg):
                    continue
                _mid_n = str(getattr(_msg, "message_id", "") or getattr(_msg, "msg_id", "") or id(_msg))
                if _mid_n in self._cached_freshness_records:
                    continue
                _plain_n = str(
                    getattr(_msg, "processed_plain_text", "")
                    or getattr(_msg, "plain_text", "")
                    or getattr(_msg, "content", "")
                    or ""
                )
                _is_at_n = getattr(_msg, "is_at", False)
                _is_quote_n = bool(getattr(_msg, "reply_to_message_id", None))
                if _is_at_n or _is_quote_n:
                    _tier_n = DecaySpeedTier.PRESERVED
                    _vis_n = VisibilityHistoryEffect.UNDERSTOOD_DEFERRED
                elif len(_plain_n) > 100:
                    _tier_n = DecaySpeedTier.SLOW_DECAY
                    _vis_n = VisibilityHistoryEffect.SEEN_AND_NOTICED
                else:
                    _tier_n = DecaySpeedTier.MEDIUM_DECAY
                    _vis_n = VisibilityHistoryEffect.GLANCED_BUT_IGNORED
                _rec = _fde.register_message(
                    message_id=_mid_n,
                    user_id=_uid_n,
                    content_fingerprint=_plain_n[:60],
                    initial_tier=_tier_n,
                    visibility=_vis_n,
                    topic_tags=[],
                )
                self._cached_freshness_records[_mid_n] = {
                    "record": _rec,
                    "registered_at": time.time(),
                }
            # 评估回看触发
            _boredom_val = 0.0
            _meta_ns = getattr(self, "_cached_metabolism_constraints", None) or {}
            if _meta_ns:
                _boredom_val = float(_meta_ns.get("boredom_level", 0.0) or 0.0)
            _fde.set_boredom_level(_boredom_val)
            if _boredom_val > get_heartfc_thresholds().boredom_drift_trigger:
                _recall_results = _fde.evaluate_boredom_drift()
                if _recall_results:
                    logger.debug(f"{self.log_prefix} [GAP-N] 无聊回看: 触发了{len(_recall_results)}条消息重新激活")
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _integrate_gossip_ritual_strategy(self, now: float) -> Optional[Dict[str, Any]]:
        """GAP-O：吃瓜/仪式行为策略引擎集成
        基于群体模式检测结果，评估当前事件型话题的参与策略"""
        if not self._cached_pattern_evidence:
            return None
        try:
            from src.core.gossip_ritual_strategy import (
                get_gossip_ritual_engine,
                EventType,
                EventContext,
                SubjectiveReadiness,
            )

            if not self._gossip_engine_initialized:
                self._gossip_engine_initialized = True
            _gre = get_gossip_ritual_engine(self.stream_id)
            _top_pe_o = None
            _top_pc_o = 0.0
            for _pe in self._cached_pattern_evidence:
                _pc_o = float(getattr(_pe, "confidence", 0.0) or 0.0)
                if _pc_o > _top_pc_o:
                    _top_pc_o = _pc_o
                    _top_pe_o = _pe
            if _top_pe_o is None or _top_pc_o < 0.35:
                return None
            _pt_enum_o = getattr(_top_pe_o, "pattern", None)
            _pt_val_o = (_pt_enum_o.value if hasattr(_pt_enum_o, "value") else str(_pt_enum_o)) if _pt_enum_o else ""
            _evt_type_map = {
                "newcomer_welcome": EventType.NEWCOMER_WELCOME,
                "birthday_wish": EventType.BIRTHDAY_CELEBRATION,
                "holiday_greeting": EventType.HOLIDAY_GREETING,
                "congratulation": EventType.CONGRATULATIONS,
                "heated_discussion": EventType.GROUP_CONFLICT,
                "argument": EventType.GROUP_CONFLICT,
                "conflict_escalation": EventType.GROUP_CONFLICT,
                "support_circle": EventType.SUPPORT_CIRCLE,
                "celebration_wave": EventType.CONGRATULATIONS,
                "spectator_mode": EventType.GOSSIP_MELON,
                "chain_reply": EventType.RITUAL_REPLY_CHAIN,
                "meme_chain": EventType.MEME_CHAIN,
                "bot_discussed": EventType.BOT_DISCUSSION,
            }
            _evt_type_o = _evt_type_map.get(_pt_val_o, EventType.NONE)
            if _evt_type_o == EventType.NONE:
                return None
            _involved_users_o = set()
            _msg_count_o = 0
            if hasattr(_top_pe_o, "involved_users") and _top_pe_o.involved_users:
                _involved_users_o = set(_top_pe_o.involved_users)
            if hasattr(_top_pe_o, "message_count"):
                _msg_count_o = int(_top_pe_o.message_count or 0)
            _is_bot_relevant_o = False
            if self._cached_self_references:
                for _ref in self._cached_self_references:
                    _rt_o = getattr(_ref, "ref_type", None)
                    if _rt_o:
                        _rv_o = _rt_o.value if hasattr(_rt_o, "value") else str(_rt_o)
                        if _rv_o in (
                            "discussed_as_topic",
                            "quoted_reply",
                            "direct_at",
                            "nickname_called",
                        ):
                            _is_bot_relevant_o = True
                            break
            _event_ctx = EventContext(
                event_type=_evt_type_o,
                confidence=_top_pc_o,
                involved_users=_involved_users_o,
                is_bot_relevant=_is_bot_relevant_o,
                bot_mentioned_directly=_is_bot_relevant_o
                and any(
                    str(getattr(r, "ref_type", "")) in ("direct_at", "nickname_called")
                    for r in (self._cached_self_references or [])
                ),
                event_intensity=min(1.0, _top_pc_o * 1.2),
                message_count_in_event=_msg_count_o,
            )
            _meta_o = getattr(self, "_cached_metabolism_constraints", None) or {}
            _rel_o = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
            _presence_o = getattr(self, "_cached_presence_state", None)
            _social_will_o = float(getattr(_presence_o, "social_willingness", 0.5) or 0.5)
            _watch_rank_o = int(getattr(_presence_o, "watch_state_rank", 2) or 2)
            _readiness = SubjectiveReadiness(
                energy_ratio=max(
                    0.05,
                    1.0 - float(_meta_o.get("energy_suppression", 0.0) or 0.0),
                ),
                social_willingness=_social_will_o,
                boredom_level=float(_meta_o.get("boredom_level", 0.0) or 0.0),
                loafing_level=float(_meta_o.get("loafing_suppression", 0.0) or 0.0),
                watch_state_rank=_watch_rank_o,
                relation_to_involved_avg=float(
                    _rel_o.get("affection", 0.0) or 0.0
                )
                / 100.0,
                annoyance_to_group=float(_rel_o.get("annoyance_value", 0.0) or 0.0) / 100.0,
                interest_in_topic=0.6 if _is_bot_relevant_o else 0.3,
            )
            _verdict = _gre.evaluate(_event_ctx, _readiness)
            self._cached_gossip_ritual_verdict = _verdict.to_dict()
            if _verdict.should_act:
                logger.info(
                    f"{self.log_prefix} [GAP-O] 事件策略: {_evt_type_o.label()} → "
                    f"{_verdict.posture.label()} (优先级={_verdict.priority:.2f})"
                )
            else:
                logger.debug(
                    f"{self.log_prefix} [GAP-O] 事件策略: {_evt_type_o.label()} → "
                    f"{_verdict.posture.label()}, 原因: {_verdict.defer_reason[:40]}"
                )
            return self._cached_gossip_ritual_verdict
        except Exception as exc:
            logger.debug(f"{self.log_prefix} [GAP-O] 吃瓜策略引擎异常: {exc}")
            return None

    def _run_adaptive_pipeline_periodic(self, now: float) -> Optional[Dict[str, Any]]:
        """GAP-P：自适应更新管道周期性运行
        每隔一段时间收集事件并尝试运行管道"""
        _interval = 1800.0
        if now - self._adaptive_pipeline_last_run < _interval:
            return None
        self._adaptive_pipeline_last_run = now
        try:
            from src.core.adaptive_update_pipeline import get_adaptive_pipeline

            _ap = get_adaptive_pipeline()
            _stats = _ap.get_collection_stats()
            if int(_stats.get("recent_window_count", 0) or 0) < getattr(
                type(self), "_SUMMARY_MIN_EVENTS_FOR_PIPELINE", 3
            ):
                return None
            _run = _ap.run_pipeline()
            self._cached_pipeline_summary = _run.to_dict()
            if _run.final_status == "published":
                logger.info(
                    f"{self.log_prefix} [GAP-P] 自适应管道发布成功: "
                    f"events={_run.events_collected}, candidates={len(_run.candidates)}, "
                    f"published={sum(1 for p in _run.publishes if p.published)}"
                )
            elif _run.final_status not in (
                "skipped_insufficient_events",
                "summarized_no_generation_needed",
            ):
                logger.debug(f"{self.log_prefix} [GAP-P] 管道状态: {_run.final_status}")
            return self._cached_pipeline_summary
        except Exception as exc:
            logger.debug(f"{self.log_prefix} [GAP-P] 自适应管道异常: {exc}")
            return None

    def _integrate_memory_governance(self, now: float) -> Optional[Dict[str, Any]]:
        """GAP-Q：记忆治理引擎周期运行——注册消息记忆+淘汰/合并+动态窗口"""
        _gov_interval = 600.0
        if now - self._last_memory_governance_ts < _gov_interval:
            return self._cached_memory_governance_snap
        self._last_memory_governance_ts = now
        try:
            from src.core.memory_governance_engine import (
                get_memory_governance_engine,
            )

            if not self._memory_governance_initialized:
                self._memory_governance_initialized = True
            _mge = get_memory_governance_engine(self.stream_id)
            _snap = _mge.run_governance_cycle()
            self._cached_memory_governance_snap = _snap.to_dict()
            if _snap.last_operation and _snap.last_operation != "无需操作":
                logger.info(
                    f"{self.log_prefix} [GAP-Q] 记忆治理: {_snap.last_operation} | "
                    f"总量={_snap.total_entries}, 利用率={_snap.utilization_ratio:.1%}"
                )
            return self._cached_memory_governance_snap
        except Exception as exc:
            logger.debug(f"{self.log_prefix} [GAP-Q] 记忆治理异常: {exc}")
            return None

    def register_message_to_memory(
        self,
        content: str,
        user_id: str = "",
        *,
        topic_tags: Optional[List[str]] = None,
    ) -> None:
        """GAP-Q快捷接口：将单条消息内容注册到记忆治理引擎"""
        try:
            from src.core.memory_governance_engine import (
                get_memory_governance_engine,
            )

            if not self._memory_governance_initialized:
                return
            _mge = get_memory_governance_engine(self.stream_id)
            _rel_q = getattr(self, "_last_relation_snapshot", None) or {}
            _emotion_q = float(_rel_q.get("annoyance_value", 0.0) or 0.0)
            _emotion_q = -min(1.0, max(-1.0, _emotion_q / 100.0))
            _mge.register_memory(
                content=content[:500],
                source_user=user_id,
                source_type="message",
                topic_tags=topic_tags,
                emotional_valence=_emotion_q,
                channel_context=self.stream_id[:20],
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def build_planner_injection_prompt(self, messages: List[Any]) -> Dict[str, Any]:
        """GAP-S：构建完整注入提示词——聚合所有GAP机制输出并生成结构化prompt"""
        try:
            from src.core.planner_prompt_injection import (
                get_planner_injection_layer,
            )

            if not self._planner_injection_initialized:
                self._planner_injection_initialized = True
            _pil = get_planner_injection_layer()
            _sit_input = None
            _sit_cached = getattr(self, "_cached_situation_interpretation", None)
            if isinstance(_sit_cached, dict):
                _sit_input = _sit_cached
            _gossip_input = self._cached_gossip_ritual_verdict
            _vis_input = None
            if self._cached_deep_visibility_results:
                from src.core.deep_visibility_scorer import (
                    get_deep_visibility_scorer,
                )

                _dvs = get_deep_visibility_scorer(self.stream_id)
                _vis_stats = _dvs.summary_stats(self._cached_deep_visibility_results)
                _vis_input = _vis_stats
            _safety_input = self._cached_safety_assessment
            _mem_input = None
            _mem_snap = self._cached_memory_governance_snap
            if _mem_snap:
                _mem_input = {
                    "window_size": _mem_snap.get("total", 0),
                    "total_chars": 0,
                    "tier_dist": _mem_snap.get("tiers", {}),
                    "reason": "",
                }
            _fresh_input = None
            try:
                from src.core.freshness_decay_engine import (
                    get_freshness_decay_engine,
                )

                _fde = get_freshness_decay_engine(self.stream_id)
                _fresh_snap = _fde.all_records_snapshot()
                _fresh_input = {
                    "total_tracked": _fresh_snap.get("total_tracked", 0),
                    "fresh_count": _fresh_snap.get("fresh_count", 0),
                    "stale_count": _fresh_snap.get("stale_count", 0),
                    "recalls": sum(
                        1 for v in (_fresh_snap.get("recent_distribution") or {}).values() if "recall" in str(v).lower()
                    ),
                    "tier_dist": _fresh_snap.get("tier_distribution", {}),
                }
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            _meta_raw = self._cached_metabolism_constraints or {}
            _night_phase_raw = None
            _np = getattr(self, "_cached_night_phase", None)
            if _np and hasattr(_np, "value"):
                _night_phase_raw = self._normalized_night_phase_value() or _np.value
            _packet = _pil.build_packet(
                channel_id=self.stream_id,
                situation=_sit_input,
                gossip_verdict=_gossip_input,
                visibility_stats=_vis_input,
                safety_result=_safety_input,
                memory_window=_mem_input,
                freshness_snapshot=_fresh_input,
                metabolism_raw=_meta_raw,
                night_phase=_night_phase_raw,
                relation_snap=getattr(self, "_last_relation_snapshot", None),
                watch_state=self._watch_level_value(),
            )
            _result = _pil.generate_full_prompt(_packet)
            self._cached_planner_injection = _result.to_dict()
            return self._cached_planner_injection
        except Exception as exc:
            logger.debug(f"{self.log_prefix} [GAP-S] 提示词注入异常: {exc}")
            return {"injection_count": 0, "error": str(exc)[:60]}

    def _run_attention_flow_tick(self, now: float) -> Optional[Dict[str, Any]]:
        """GAP-W：主观注意力流控制器tick——每15秒更新注意力状态"""
        try:
            from src.core.subjective_attention_flow import (
                get_attention_flow_controller,
            )

            if not self._attention_flow_initialized:
                self._attention_flow_initialized = True
            _afc = get_attention_flow_controller(self.stream_id)
            _meta_aw = self._cached_metabolism_constraints or {}
            _emotion_snap = getattr(self, "_cached_emotion_state", None) or {}
            _np_aw = getattr(self, "_cached_night_phase", None)
            _safety_aw = getattr(self, "_cached_safety_assessment", None) or {}
            _recent_proactive_count = sum(
                1
                for _ts in (self._proactive_reply_timeline or [])
                if now - float(_ts or 0.0) < 300.0
            )
            _snap = _afc.tick(
                now,
                boredom=float(_meta_aw.get("boredom_level", 0.0) or 0.0),
                social_desire=float(_emotion_snap.get("social_desire", 0.5) or 0.5),
                energy=max(
                    0.05,
                    1.0 - float(_meta_aw.get("energy_suppression", 0.0) or 0.0),
                ),
                mood=float(_emotion_snap.get("mood", 0.5) or 0.5),
                loneliness=float(_emotion_snap.get("loneliness", 0.0) or 0.0),
                curiosity=float(_emotion_snap.get("curiosity", 0.3) or 0.3),
                night_phase=self._normalized_night_phase_value(),
                safety_level=str(_safety_aw.get("level", "") or ""),
                has_pending_user=bool(self._pending_user_ids),
                group_activity_level=self._group_activity_level,
                recent_proactive_count=_recent_proactive_count,
                emotion_overload_flag=self._cached_user_negative_emotion > 70,
            )
            self._cached_attention_snapshot = _snap.to_dict()
            if _snap.state != _snap.previous_state:
                logger.info(
                    f"{self.log_prefix} [GAP-W] 注意力状态: {_snap.previous_state.label()} → {_snap.state.label()} "
                    f"({_snap.transition_reason.label() if _snap.transition_reason else ''})"
                )
            return self._cached_attention_snapshot
        except Exception as exc:
            logger.debug(f"{self.log_prefix} [GAP-W] 注意力流异常: {exc}")
            return None

    def run_emotion_feedback_cycle(self, now: float) -> Optional[Dict[str, Any]]:
        """GAP-V：情感反馈环周期执行——聚合所有引擎输出回写emotion_state"""
        try:
            from src.core.emotion_feedback_loop import (
                get_emotion_feedback_loop,
            )

            if not self._emotion_feedback_initialized:
                self._emotion_feedback_initialized = True
            _efl = get_emotion_feedback_loop()
            _meta_v = self._cached_metabolism_constraints or {}
            _efl.submit_from_metabolism(
                self.stream_id,
                energy_ratio=max(
                    0.05,
                    1.0 - float(_meta_v.get("energy_suppression", 0.0) or 0.0),
                ),
                boredom_level=float(_meta_v.get("boredom_level", 0.0) or 0.0),
                loafing_level=float(_meta_v.get("loafing_suppression", 0.0) or 0.0),
            )
            _sit_v = getattr(self, "_cached_situation_interpretation", None)
            if isinstance(_sit_v, dict):
                _efl.submit_from_situation(
                    self.stream_id,
                    risk_score=float(_sit_v.get("risk_score", 0.0) or 0.0),
                    atmosphere=str(_sit_v.get("atmosphere", "") or ""),
                    engagement_intensity=float(_sit_v.get("intensity", 0.5) or 0.5),
                    internal_energy=float(_sit_v.get("energy", 0.5) or 0.5),
                )
            _safe_v = self._cached_safety_assessment
            if isinstance(_safe_v, dict):
                _efl.submit_from_safety(
                    self.stream_id,
                    safety_level=str(_safe_v.get("level", "") or ""),
                    safety_score=float(_safe_v.get("score", 0.0) or 0.0),
                    blocked=bool(_safe_v.get("block_reply", False)),
                )
            _np_v = getattr(self, "_cached_night_phase", None)
            if _np_v and hasattr(_np_v, "value"):
                _efl.submit_from_night(
                    self.stream_id,
                    night_phase=self._normalized_night_phase_value() or _np_v.value,
                    pressure_total=float(getattr(self, "_overnight_pressure_total", 0.0) or 0.0),
                    is_burnthrough=bool((getattr(self, "_cached_night_summary", {}) or {}).get("is_burnthrough")),
                )
            _latest_msg = None
            _incoming = getattr(self, "_cached_incoming_messages", []) or []
            if _incoming:
                _latest_msg = _incoming[-1]
            elif hasattr(self, "_last_user_message"):
                _latest_msg = self._last_user_message
            if _latest_msg:
                _msg_text = str(getattr(_latest_msg, "processed_plain_text", "") or "")
                _uid = str(getattr(_latest_msg, "user_id", "") or "")
                _is_mentioned = bool(
                    getattr(_latest_msg, "is_at", False) or getattr(_latest_msg, "mentioned_me", False)
                )
                _is_question = "?" in _msg_text or "？" in _msg_text
                _is_first = False
                try:
                    from src.core.impression_evolution_hub import (
                        get_impression_hub,
                    )

                    _ihub = get_impression_hub(self.stream_id)
                    _imp = _ihub.get_impression(_uid) if _uid else None
                    if _imp and hasattr(_imp, "truth"):
                        _is_first = (getattr(_imp.truth, "total_interactions", 0) or 0) <= 2
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
                _repeat_cnt = self._count_recent_user_repeats(_msg_text, _uid) if _msg_text and _uid else 0
                _efl.submit_from_user_interaction(
                    self.stream_id,
                    is_mentioned=_is_mentioned,
                    is_question=_is_question,
                    is_first_contact=_is_first,
                    message_length=len(_msg_text),
                    repeat_count=_repeat_cnt,
                )
            _report = _efl.run_feedback_cycle(self.stream_id)
            self._cached_emotion_feedback_report = _report.to_dict()
            return self._cached_emotion_feedback_report
        except Exception as exc:
            logger.debug(f"{self.log_prefix} [GAP-V] 情感反馈异常: {exc}")
            return None

    def register_cross_engine_outputs(self) -> Dict[str, Any]:
        """GAP-T：跨引擎联动验证——注册所有引擎产出并验证消费状态"""
        try:
            from src.core.cross_engine_validator import (
                get_cross_engine_validator,
                EngineId,
                ConsumptionStatus,
            )

            if not self._cross_validator_initialized:
                self._cross_validator_initialized = True
            _cev = get_cross_engine_validator()
            for _eid in (
                EngineId.ACFN,
                EngineId.SCME,
                EngineId.DEEP_VISIBILITY,
                EngineId.FRESHNESS_DECAY,
                EngineId.GOSSIP_RITUAL,
                EngineId.SAFETY_FUSION,
                EngineId.MEMORY_GOVERNANCE,
                EngineId.PLANNER_INJECTION,
                EngineId.NIGHT_CYCLE,
                EngineId.METABOLISM,
                EngineId.EMOTION_DRIVEN,
                EngineId.GROUP_PATTERN,
                EngineId.INNER_NARRATION,
                EngineId.PRESENCE_CORE,
                EngineId.SELF_REFERENCE,
                EngineId.MULTIMODAL_BUDGETER,
                EngineId.IMPRESSION,
                EngineId.SKILL_LIFECYCLE,
                EngineId.SOCIAL_AFFECT,
            ):
                _cev.register_engine(_eid)
            # 深度可见性：按预定义边字段名记录
            if self._cached_deep_visibility_results:
                _cev.record_output(EngineId.DEEP_VISIBILITY, "modulated_score")
                _cev.record_output(EngineId.DEEP_VISIBILITY, "enter_understanding")
            # 安全边界融合：按预定义边字段名记录
            if self._cached_safety_assessment:
                _cev.record_output(EngineId.SAFETY_FUSION, "activation_bar_delta", is_critical=True)
                _cev.record_output(EngineId.SAFETY_FUSION, "fused_score_delta", is_critical=True)
            # 吃瓜策略：按预定义边字段名记录
            if self._cached_gossip_ritual_verdict:
                _cev.record_output(EngineId.GOSSIP_RITUAL, "verdict.posture")
                _cev.record_output(EngineId.GOSSIP_RITUAL, "verdict.reply_hint")
            # 记忆治理：按预定义边字段名记录
            if self._cached_memory_governance_snap:
                _cev.record_output(EngineId.MEMORY_GOVERNANCE, "window_text")
            # Planner注入：按预定义边字段名记录
            if self._cached_planner_injection:
                _cev.record_output(EngineId.PLANNER_INJECTION, "full_prompt")
            # 代谢引擎：能量比/无聊度/摸鱼度
            if self._cached_metabolism_state:
                _cev.record_output(EngineId.METABOLISM, "energy_ratio")
                _cev.record_output(EngineId.METABOLISM, "boredom_level")
                _cev.record_output(EngineId.METABOLISM, "loafing_level")
            # 昼夜周期：阶段/压力分解
            if self._cached_night_phase:
                _cev.record_output(EngineId.NIGHT_CYCLE, "phase")
                _cev.record_output(EngineId.NIGHT_CYCLE, "pressure_breakdown")
            # 新鲜度衰减：当前新鲜度/回忆决策
            if self._cached_freshness_records:
                _cev.record_output(EngineId.FRESHNESS_DECAY, "current_freshness")
                _cev.record_output(EngineId.FRESHNESS_DECAY, "recall_decisions")
            # 情感驱动核心：情绪/主动意愿
            if self._cached_emotion_state:
                _cev.record_output(EngineId.EMOTION_DRIVEN, "mood")
                _cev.record_output(EngineId.EMOTION_DRIVEN, "proactive_willingness")
            # 群体模式检测：模式证据
            if getattr(self, "_cached_scene_snapshot", None):
                _cev.record_output(EngineId.GROUP_PATTERN, "pattern_evidence")
            # 内心独白规划：独白文本
            if self._cached_narration_plan:
                _cev.record_output(EngineId.INNER_NARRATION, "narration_text")
            # 存在感核心：关注状态排名/关注状态
            if self._cached_presence_state:
                _cev.record_output(EngineId.PRESENCE_CORE, "watch_state_rank")
                _cev.record_output(EngineId.PRESENCE_CORE, "watch_state")
            # 自我指称检测：引用列表
            if self._cached_self_references:
                _cev.record_output(EngineId.SELF_REFERENCE, "ref_list")
            # 多模态预算：预算决策
            if getattr(self, "_cached_media_decisions", None):
                _cev.record_output(EngineId.MULTIMODAL_BUDGETER, "budget_decision")
            # 技能生命周期：成本修正因子
            if getattr(self, "_cached_skill_review", None):
                _cev.record_output(EngineId.SKILL_LIFECYCLE, "cost_modifier")
            # 个体印象：行为参数（依赖好感度缓存）
            if getattr(self, "_cached_affection_value", None) is not None:
                _cev.record_output(EngineId.IMPRESSION, "behavior_params")
            # 社交情感融合：社交分数（依赖好感度缓存）
            if getattr(self, "_cached_affection_value", None) is not None:
                _cev.record_output(EngineId.SOCIAL_AFFECT, "social_score")
            # 标记各引擎产出的下游交付关系
            _cev.mark_consumed(
                EngineId.DEEP_VISIBILITY,
                "modulated_score",
                consumer="决策链",
                status=ConsumptionStatus.BY_DECISION,
            )
            _cev.mark_consumed(
                EngineId.SAFETY_FUSION,
                "activation_bar_delta",
                consumer="安全层",
                status=ConsumptionStatus.BY_SAFETY,
            )
            _cev.mark_consumed(
                EngineId.SAFETY_FUSION,
                "fused_score_delta",
                consumer="安全层",
                status=ConsumptionStatus.BY_SAFETY,
            )
            _cev.mark_consumed(
                EngineId.MEMORY_GOVERNANCE,
                "window_text",
                consumer="规划器",
                status=ConsumptionStatus.BY_PLANNER,
            )
            _cev.mark_consumed(
                EngineId.PLANNER_INJECTION,
                "full_prompt",
                consumer="LLM调用",
                status=ConsumptionStatus.BY_PLANNER,
            )
            _cev.mark_consumed(
                EngineId.GOSSIP_RITUAL,
                "verdict.posture",
                consumer="独白层",
                status=ConsumptionStatus.BY_NARRATION,
            )
            _cev.mark_consumed(
                EngineId.INNER_NARRATION,
                "narration_text",
                consumer="规划器",
                status=ConsumptionStatus.BY_PLANNER,
            )
            # 代谢引擎 → ACFN/状态耦合矩阵
            _cev.mark_consumed(
                EngineId.METABOLISM,
                "energy_ratio",
                consumer="ACFN引擎",
                status=ConsumptionStatus.BY_DECISION,
            )
            _cev.mark_consumed(
                EngineId.METABOLISM,
                "boredom_level",
                consumer="状态耦合矩阵",
                status=ConsumptionStatus.BY_DECISION,
            )
            _cev.mark_consumed(
                EngineId.METABOLISM,
                "loafing_level",
                consumer="状态耦合矩阵",
                status=ConsumptionStatus.BY_DECISION,
            )
            # 昼夜周期 → 规划器/ACFN
            _cev.mark_consumed(
                EngineId.NIGHT_CYCLE,
                "phase",
                consumer="规划器",
                status=ConsumptionStatus.BY_PLANNER,
            )
            _cev.mark_consumed(
                EngineId.NIGHT_CYCLE,
                "pressure_breakdown",
                consumer="ACFN引擎",
                status=ConsumptionStatus.BY_DECISION,
            )
            # 新鲜度衰减 → ACFN/主动层
            _cev.mark_consumed(
                EngineId.FRESHNESS_DECAY,
                "current_freshness",
                consumer="ACFN引擎",
                status=ConsumptionStatus.BY_DECISION,
            )
            _cev.mark_consumed(
                EngineId.FRESHNESS_DECAY,
                "recall_decisions",
                consumer="主动层",
                status=ConsumptionStatus.BY_PROACTIVE,
            )
            # 情感驱动核心 → ACFN/主动层
            _cev.mark_consumed(
                EngineId.EMOTION_DRIVEN,
                "mood",
                consumer="ACFN引擎",
                status=ConsumptionStatus.BY_DECISION,
            )
            _cev.mark_consumed(
                EngineId.EMOTION_DRIVEN,
                "proactive_willingness",
                consumer="主动层",
                status=ConsumptionStatus.BY_PROACTIVE,
            )
            # 群体模式检测 → 吃瓜策略
            _cev.mark_consumed(
                EngineId.GROUP_PATTERN,
                "pattern_evidence",
                consumer="吃瓜策略",
                status=ConsumptionStatus.BY_DECISION,
            )
            # 存在感核心 → 深度可见性/状态转换
            _cev.mark_consumed(
                EngineId.PRESENCE_CORE,
                "watch_state_rank",
                consumer="深度可见性",
                status=ConsumptionStatus.BY_DECISION,
            )
            _cev.mark_consumed(
                EngineId.PRESENCE_CORE,
                "watch_state",
                consumer="状态转换",
                status=ConsumptionStatus.BY_DECISION,
            )
            # 社交情感 → 深度可见性
            _cev.mark_consumed(
                EngineId.SOCIAL_AFFECT,
                "social_score",
                consumer="深度可见性",
                status=ConsumptionStatus.BY_DECISION,
            )
            # 自我指称检测 → 深度可见性
            _cev.mark_consumed(
                EngineId.SELF_REFERENCE,
                "ref_list",
                consumer="深度可见性",
                status=ConsumptionStatus.BY_DECISION,
            )
            # 吃瓜策略回复提示 → 规划器
            _cev.mark_consumed(
                EngineId.GOSSIP_RITUAL,
                "verdict.reply_hint",
                consumer="规划器",
                status=ConsumptionStatus.BY_PLANNER,
            )
            # 深度可见性理解信号 → 记忆层
            _cev.mark_consumed(
                EngineId.DEEP_VISIBILITY,
                "enter_understanding",
                consumer="记忆治理",
                status=ConsumptionStatus.BY_MEMORY,
            )
            # 印象引擎 → 风格层
            _cev.mark_consumed(
                EngineId.IMPRESSION,
                "behavior_params",
                consumer="风格层",
                status=ConsumptionStatus.BY_STYLE,
            )
            # 技能生命周期 → ACFN
            _cev.mark_consumed(
                EngineId.SKILL_LIFECYCLE,
                "cost_modifier",
                consumer="ACFN引擎",
                status=ConsumptionStatus.BY_DECISION,
            )
            # 多模态预算 → 语义桥
            _cev.mark_consumed(
                EngineId.MULTIMODAL_BUDGETER,
                "budget_decision",
                consumer="语义桥",
                status=ConsumptionStatus.BY_DECISION,
            )
            _report = _cev.validate()
            return _report.to_dict()
        except Exception as exc:
            logger.debug(f"{self.log_prefix} [GAP-T] 跨引擎验证异常: {exc}")
            return {"error": str(exc)[:60]}


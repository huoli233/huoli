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

class SceneBotLifecycleMixin:
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
                _ncs_f28.record_overnight_activity("chat", 1.0)
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

    def _apply_night_cycle_modulation(
        self,
        now: float,
        incoming_batch: Optional[List] = None,
        pinged_msg: Optional[Any] = None,
    ) -> Optional[dict]:
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
            _admin_force_wake = bool(
                getattr(self, "_last_msg_was_admin", False)
                or getattr(self, "_is_admin_forced", False)
                or self._is_force_wake_admin(incoming_batch or [], pinged_msg)
            )
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
            if _admin_force_wake:
                _night_ctx = _d6.record_disturbance(self.stream_id)
                _adaptive = _night_ctx.get("adaptive", {})
                return {
                    "action": "force_wake",
                    "reason": "管理员强制唤醒",
                    "context": _night_ctx,
                    "stimulus": _adaptive.get("stimulus_strength", 0.0),
                    "threshold": _adaptive.get("threshold", 50.0),
                }
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
                    _phase_obj = getattr(_cap_info, "current_phase", None)
                    _phase = str(getattr(_phase_obj, "value", _phase_obj) or "").strip().lower()
                    if _phase == "deep_sleep":
                        _reason = (
                            "深睡阶段禁止普通夜间回复"
                            f"({_cap_info.night_reply_count}/{_cap_info.night_reply_cap})"
                        )
                    elif _phase == "burned_out":
                        _reason = (
                            "熬穿阶段禁止普通夜间回复"
                            f"({_cap_info.night_reply_count}/{_cap_info.night_reply_cap})"
                        )
                    else:
                        _reason = f"夜间回复已达上限({_cap_info.night_reply_count}/{_cap_info.night_reply_cap})"
                    return {
                        "action": "hard_block",
                        "reason": _reason,
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
            from src.core.world_snapshot import get_relation_number

            pressure = get_relation_number(
                relation_result,
                "psychological_pressure",
            )
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

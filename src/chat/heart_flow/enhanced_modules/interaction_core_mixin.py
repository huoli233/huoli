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

class EnhancedInteractionCoreMixin:
    def _spawn(self, coro, name: str = ""):
        """安全创建后台任务，防止GC回收未引用的Task，并显式记录异常。"""
        task = asyncio.create_task(coro, name=name or None)

        def _finalize_background_task(done_task: asyncio.Task) -> None:
            self._background_tasks.discard(done_task)
            if done_task.cancelled():
                return
            try:
                exc = done_task.exception()
            except Exception as callback_exc:
                logger.warning(f"{self.log_prefix} 后台任务状态读取失败: {callback_exc}")
                return
            if exc is not None:
                task_name = done_task.get_name()
                logger.error(
                    f"{self.log_prefix} 后台任务异常退出[{task_name if task_name else 'unnamed'}]: {exc}",
                    exc_info=exc,
                )

        self._background_tasks.add(task)
        task.add_done_callback(_finalize_background_task)
        return task

    async def _observe(
        self,
        recent_messages_list: Optional[List[Any]] = None,
        force_reply_message: Optional[Any] = None,
    ) -> bool:
        """增强链路的观察降级出口。

        旧群聊链路并没有 BrainChatting._observe()，但多个失败回退分支会调用它。
        这里统一把“降级观察”显式收敛为未回复，避免回退分支抛 AttributeError。
        """
        if not getattr(self, "_last_flow_blocker", ""):
            target_message = force_reply_message or self._get_latest_human_message(recent_messages_list or [])
            target_desc = ""
            if target_message is not None:
                target_desc = self._extract_message_content(target_message).strip()[:24]
            self._last_flow_blocker = f"降级为观察{f': {target_desc}' if target_desc else ''}"
        return False

    def _get_incoming_batch_from_context(self) -> List[Any]:
        """从增强链缓存中取当前消息批次，供旁路上下文安全读取。"""
        batch = getattr(self, "_cached_incoming_messages", None)
        if isinstance(batch, list):
            return batch
        return []

    def _normalize_self_reference_type(self, ref_type: Any) -> str:
        """兼容不同自指检测器的枚举/字符串输出，统一到增强链的四类语义。"""
        raw = str(getattr(ref_type, "value", ref_type) or "").strip().lower()
        mapping = {
            "direct_at": "direct_at",
            "name_mention": "nickname_called",
            "nickname_called": "nickname_called",
            "quote_bot_msg": "quoted_reply",
            "quoted_reply": "quoted_reply",
            "pronoun_refer": "discussed_as_topic",
            "indirect_topic": "discussed_as_topic",
            "discussed_as_topic": "discussed_as_topic",
        }
        return mapping.get(raw, raw or "none")

    async def _run_self_restraint_check(
        self,
        decision_messages: List,
        source: str,
        repetition_signal: Optional[Dict[str, Any]] = None,
        harassment_signal: Optional[Dict[str, Any]] = None,
        force_bypass: bool = False,
        targeted_to_bot: bool = False,
        admin_force: bool = False,
    ) -> Dict[str, Any]:
        """薄协调层：主要把主链状态桥接到兄弟模块，不再承载大段判断规则。"""
        from src.chat.heart_flow.heartflow_decision import acquire_decision_maker

        rest_verdict = self._evaluate_rest_governor(
            incoming_batch=decision_messages,
            source=source,
            targeted_to_bot=targeted_to_bot,
            admin_force=admin_force,
        )
        if not admin_force and not force_bypass:
            if rest_verdict.interruption_policy == "block" and not targeted_to_bot:
                return {
                    "allow": False,
                    "mode": "skip",
                    "reason": f"休息Governor: {self._summarize_rest_governor(rest_verdict)}",
                }
            if rest_verdict.interruption_policy == "peek_only" and source == "proactive" and not targeted_to_bot:
                return {
                    "allow": False,
                    "mode": "skip",
                    "reason": f"休息Governor: {self._summarize_rest_governor(rest_verdict)}",
                }
            if rest_verdict.interruption_policy == "peek_only" and targeted_to_bot:
                return {
                    "allow": True,
                    "mode": "short_only",
                    "reason": f"休息Governor: {self._summarize_rest_governor(rest_verdict)}",
                }

        _reply_timeline = self._proactive_reply_timeline if source == "proactive" else self._bot_reply_timeline
        recent_reply_burst = sum(1 for t in _reply_timeline if time.time() - t < 180.0)
        chatterbox_penalty = float(getattr(self, "_chatterbox_penalty", 0.0) or 0.0)
        chat_stamina = 100.0
        brain_reserve = 100.0
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            if _ch is not None:
                chat_stamina = float(getattr(_ch, "chat_pool", 100.0) or 100.0)
                brain_reserve = float(getattr(_ch, "thinking_value", 100.0) or 100.0)
            else:
                snap = getattr(self, "_tick_world_snapshot", None)
                res = getattr(snap, "self_resources", None) if snap is not None else None
                if res is not None:
                    chat_stamina = float(getattr(res, "chat_energy", 100.0) or 100.0)
                    brain_reserve = float(getattr(res, "thinking_energy", 100.0) or 100.0)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 自省闸门资源快照读取失败: {exc}")
            chat_stamina = float(getattr(self, "_chat_energy", 100.0) or 100.0)
            brain_reserve = float(getattr(self, "_thinking_energy", 100.0) or 100.0)
        return await acquire_decision_maker().evaluate_restraint_from_state(
            self.stream_id,
            decision_messages=decision_messages,
            relation_view=self._resolve_relation_view(),
            repetition_signal=repetition_signal,
            harassment_signal=harassment_signal,
            chat_stamina=chat_stamina,
            brain_reserve=brain_reserve,
            streak_count=int(getattr(self, "_consecutive_speaks", 0.0) or 0.0),
            recent_reply_burst=recent_reply_burst,
            chatterbox_penalty=chatterbox_penalty,
            source=source,
            force_bypass=force_bypass,
            targeted_to_bot=targeted_to_bot,
            admin_force=admin_force,
        )

    async def _finalize_sent_reply(
        self,
        reply_text: str,
        loop_info: Dict[str, Any],
        target_message: Any,
        reply_reason: str,
        relation_view: Dict[str, Any],
        llm_response: Any,
        was_proactive: bool,
        action_name: str,
        quality: float,
        audit_label: str,
        reply_trace_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.last_active_time = time.time()
        if was_proactive:
            self._last_proactive_target_user_id = str(
                getattr(target_message, "user_id", "") or ""
            ).strip()
        from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

        acquire_reply_coordinator().remember_bot_utterance(self.stream_id, reply_text)
        self._remember_bot_utterance(reply_text)
        self._record_reply_for_diversity(reply_text)
        # 增强链路在发送阶段已合并 reply_set 文本段，这里不再异步补发后续段。
        self._register_bot_message_record(
            reply_text=reply_text,
            loop_info=loop_info,
            target_message=target_message,
            was_proactive=was_proactive,
            reply_trace_meta=reply_trace_meta,
        )
        self._spawn(
            self._capture_reply_behavior_learning(
                reply_text=reply_text,
                reply_reason=reply_reason,
                relation_snapshot=relation_view,
                was_proactive=was_proactive,
            )
        )
        self._mark_message_content_processed(
            target_message,
            action_name=action_name,
            quality=quality,
        )
        try:
            from src.chat.heart_flow.skills.focus_patrol import acquire_focus_patrol

            _risk_note = (
                f"{audit_label} annoyance={relation_view.get('annoyance_value', 0)} "
                f"pressure={relation_view.get('psychological_pressure', 0)}"
            )
            _need_audit, _audit_reason = await acquire_focus_patrol().audit_high_risk_reply(
                reply_text,
                risk_note=_risk_note,
            )
            if _need_audit:
                logger.info(f"{self.log_prefix} 🔍 发送后审查提示({audit_label}): {_audit_reason}")
        except Exception as _audit_exc:
            logger.debug(f"{self.log_prefix} {audit_label}回复审查异常: {_audit_exc}")
        try:
            from src.memory_system.memory_retrieval import invalidate_memory_cache

            invalidate_memory_cache(self.stream_id)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 清除记忆缓存异常: {exc}")

    async def _send_and_store_reply(
        self,
        response_set,
        action_message,
        cycle_timers: Dict[str, float],
        thinking_id,
        actions,
        selected_expressions: Optional[List[int]] = None,
        quote_message: bool = False,
        target_message=None,
        pre_send_risk_note: str = "",
        pre_send_audit_label: str = "reply",
    ) -> Tuple[Dict[str, Any], str, Dict[str, float], Dict[str, Any]]:
        """兼容增强链路的回复发送入口。

        历史上不稳定地变更过这个方法的可用性和签名。
        增强版统一在这里兜底，继续复用现有发送与动作落库流程。
        """
        reply_target = action_message or target_message
        _raw_reply_text = str(self._extract_plain_text_reply(response_set) or "").strip()
        _guarded_reply_text, _guard_changed, _guard_reason, _guard_blocked = await self._apply_pre_send_reply_guard(
            response_set,
            audit_label=pre_send_audit_label,
            risk_note=pre_send_risk_note,
        )
        if _guard_blocked:
            logger.info(
                f"{self.log_prefix} 🧯 发送前审查阻断({pre_send_audit_label}): {str(_guard_reason or '').strip()[:80]}"
            )
            return {}, "", cycle_timers, {}

        with Timer("回复发送", cycle_timers):
            reply_text = await self._send_response(
                reply_set=response_set,
                message_data=reply_target,
                selected_expressions=selected_expressions,
                quote_message=quote_message,
            )

        reply_text = str(reply_text or _guarded_reply_text or "").strip()
        if not reply_text:
            logger.warning(f"{self.log_prefix} 回复发送未返回有效文本，视为发送失败")
            return {}, "", cycle_timers, {}

        reply_trace_meta = {
            "raw_reply": _raw_reply_text,
            "final_sent_reply": reply_text,
            "pre_send_rewritten": bool(_guard_changed and _guarded_reply_text and _guarded_reply_text != _raw_reply_text),
            "pre_send_reason": str(_guard_reason or "").strip(),
            "audit_label": str(pre_send_audit_label or "").strip(),
        }

        from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

        acquire_reply_coordinator().remember_bot_utterance(self.stream_id, reply_text)
        self._remember_bot_utterance(reply_text)
        try:
            from src.memory_system.memory_core import acquire_recollection_hub

            acquire_recollection_hub().deposit_memory(
                stream_id=self.stream_id,
                content=reply_text,
                entry_category="bot_response",
                user_id="bot",
                significance=0.6,
                extra_data={
                    "source": "reply_send",
                    "chat_id": self.stream_id,
                },
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 写入bot回复记忆失败: {exc}")

        platform = None
        if reply_target is not None:
            platform = getattr(getattr(reply_target, "chat_info", None), "platform", None)
            if platform is None:
                platform = getattr(getattr(reply_target, "platform", None), "value", None) or getattr(
                    reply_target, "platform", None
                )
        if platform is None:
            platform = getattr(self.chat_stream, "platform", "unknown")

        user_id = ""
        if reply_target is not None:
            user_id = str(
                getattr(getattr(reply_target, "user_info", None), "user_id", "")
                or getattr(reply_target, "user_id", "")
                or ""
            )

        person_name = "对方"
        if user_id:
            try:
                from src.person_info.person_info import Person

                person = Person(platform=platform, user_id=user_id)
                person_name = person.person_name or person_name
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

        action_prompt_display = f"你对{person_name}进行了回复：{reply_text}"
        await database_api.store_action_info(
            chat_stream=self.chat_stream,
            action_build_into_prompt=False,
            action_prompt_display=action_prompt_display,
            action_done=True,
            thinking_id=thinking_id,
            action_data={
                "reply_text": reply_text,
                "quote_message": bool(quote_message),
            },
            action_name="reply",
        )

        loop_info: Dict[str, Any] = {
            "loop_plan_info": {
                "action_result": actions,
            },
            "loop_action_info": {
                "action_taken": True,
                "reply_text": reply_text,
                "command": "",
                "taken_time": time.time(),
                "quote_message": bool(quote_message),
            },
        }

        return loop_info, reply_text, cycle_timers, reply_trace_meta

    def _mark_user_replied(self, latest_msg: Any = None) -> None:
        _had_pending_proactive = self._unanswered_bot_turns > 0
        self._unanswered_bot_turns = 0
        self._last_proactive_target_user_id = ""
        # 简易情感估算（在 try 块外定义，确保后续可用）
        _sentiment = 0.0
        _relevance = 0.5
        _is_negative = False
        if latest_msg is not None:
            _text = str(
                getattr(latest_msg, "processed_plain_text", "") or getattr(latest_msg, "plain_text", "") or ""
            ).strip()
            if _text:
                _neg_kw = ("闭嘴", "滚", "别说了", "烦死", "吵", "安静", "shut")
                if any(kw in _text for kw in _neg_kw):
                    _is_negative = True
                    _sentiment = -0.6
                elif len(_text) <= 3:
                    _sentiment = 0.1
                    _relevance = 0.3
                else:
                    _sentiment = 0.3
                    _relevance = 0.6
        _uid = str(getattr(latest_msg, "user_id", "") or "") if latest_msg else ""
        _content = ""
        if latest_msg is not None:
            _content = str(
                getattr(latest_msg, "processed_plain_text", "")
                or getattr(latest_msg, "plain_text", "")
                or getattr(latest_msg, "content", "")
                or ""
            ).strip()[:160]
        _ledger_acknowledged = False
        # 通知主动决策账本：用户回复了（结算待处理的主动事件）
        try:
            from src.chat.proactive.proactive_decider import get_proactive_decider

            _ledger_acknowledged = bool(
                get_proactive_decider().ledger.acknowledge_user_reply(
                    self.stream_id,
                    is_negative=_is_negative,
                    reply_sentiment=_sentiment,
                    reply_relevance=_relevance,
                    target_user=_uid,
                )
            )
        except Exception as _e:
            logger.debug(f"{self.log_prefix} 主动决策账本回复确认异常: {_e}")
        # 通知整合中心：收到用户消息 + 主动行为后果反馈
        try:
            from src.chat.proactive.proactive_integration_hub import get_proactive_integration_hub

            _hub = get_proactive_integration_hub()
            _hub.on_user_message_received(self.stream_id, user_id=_uid, content=_content)
            # 如果之前有待结算的主动发言，通知hub记录行为后果
            if _had_pending_proactive and latest_msg is not None:
                _consequence = _hub.on_proactive_outcome(
                    channel_id=self.stream_id,
                    has_reply=True,
                    reply_sentiment=_sentiment,
                    reply_relevance=_relevance,
                    resolve_ledger=not _ledger_acknowledged,
                    target_user=_uid,
                )
                if _consequence:
                    logger.debug(
                        f"{self.log_prefix} [闭环反馈] 主动行为后果: "
                        f"tag={_consequence.outcome_tag} "
                        f"reward={_consequence.reward_delta:.2f}"
                    )
        except Exception as _e:
            logger.debug(f"{self.log_prefix} 整合中心用户消息通知异常: {_e}")

    def _message_explicitly_replies_to_bot(self, msg: Any) -> bool:
        if not msg:
            return False
        user_id = str(getattr(msg, "user_id", "") or "")
        if not user_id or self._is_bot_message_obj(msg):
            return False

        text = str(getattr(msg, "processed_plain_text", "") or getattr(msg, "plain_text", "") or "").strip()
        if not text:
            return False

        if bool(getattr(msg, "is_reply_to_bot", False) or getattr(msg, "is_quote_to_bot", False)):
            return True

        reply_to = str(getattr(msg, "reply_to", "") or "").strip()
        if reply_to:
            try:
                from src.core.self_reply_recognizer import (
                    get_self_reply_recognizer,
                )

                recognizer = get_self_reply_recognizer()
                if recognizer.get_bot_message(reply_to) is not None:
                    return True
                chain = recognizer.get_reply_chain(reply_to)
                if chain and chain.bot_message_ids:
                    return True
            except Exception as _e:
                logger.debug(f"{self.log_prefix} 回复链检测异常: {_e}")

        referenced_text = str(getattr(msg, "quoted_content", "") or "").strip()
        recent_bot_texts = self._recent_bot_texts()
        if referenced_text and recent_bot_texts:
            if any(referenced_text in said or said in referenced_text for said in recent_bot_texts[-3:]):
                return True

        return False

    def _message_looks_like_user_reply(self, msg: Any) -> bool:
        if self._message_explicitly_replies_to_bot(msg):
            return True
        if not msg:
            return False
        user_id = str(getattr(msg, "user_id", "") or "")
        if not user_id or self._is_bot_message_obj(msg):
            return False

        text = str(getattr(msg, "processed_plain_text", "") or getattr(msg, "plain_text", "") or "").strip()
        if not text:
            return False

        recent_bot_texts = self._recent_bot_texts()
        # bot说完后用户很快跟一句，视为接话
        _msg_ts = float(getattr(msg, "time", 0.0) or getattr(msg, "timestamp", 0.0) or time.time())
        _pending_proactive = int(getattr(self, "_unanswered_bot_turns", 0) or 0) > 0
        _expected_uid = str(
            getattr(
                self,
                "_last_proactive_target_user_id",
                getattr(self, "_last_user_id", ""),
            )
            or ""
        )
        if (
            self._last_bot_reply_ts > 0
            and _msg_ts > self._last_bot_reply_ts
            and (_msg_ts - self._last_bot_reply_ts) <= 180.0
        ):
            if self._message_targets_bot(msg):
                return True
            if _expected_uid and user_id == _expected_uid:
                return True

        recent_self = recent_bot_texts[-1] if recent_bot_texts else ""
        if recent_self:
            bot_terms = [part for part in re.split(r"[，。！？、；：,.!?\s]+", recent_self) if len(part) >= 2]
            if bot_terms and any(term in text for term in bot_terms[:4]):
                if not _pending_proactive:
                    return True
                if self._message_targets_bot(msg):
                    return True
                if _expected_uid and user_id == _expected_uid:
                    return True

        return False

    def _message_targets_bot(self, msg: Any) -> bool:
        """统一识别是否属于“在找 bot”的输入。"""
        if not msg or self._is_bot_message_obj(msg):
            return False
        if self._is_message_pinged(msg):
            return True
        if bool(getattr(msg, "mentioned_me", False)):
            return True
        if bool(getattr(msg, "is_at_bot", False) or getattr(msg, "is_quote_to_bot", False)):
            return True
        if bool(getattr(msg, "is_reply_to_bot", False)):
            return True
        return self._message_explicitly_replies_to_bot(msg)

    def _has_targeted_bot_message(self, messages: List) -> bool:
        candidates = self._get_human_message_candidates(
            messages or [],
            preferred_sources=("incoming",),
            allow_fallback=True,
        )
        for _msg in candidates:
            if self._message_targets_bot(_msg):
                return True
        return False

    def _batch_contains_real_user_reply(self, incoming_batch: List[Any]) -> bool:
        for msg in reversed(incoming_batch or []):
            if self._message_looks_like_user_reply(msg):
                return True
        return False

    def _build_self_reply_memory(self) -> str:
        recent_bot_texts = self._recent_bot_texts()
        if not recent_bot_texts:
            return ""
        lines = []
        for idx, text in enumerate(recent_bot_texts[-3:], start=1):
            brief = text if len(text) <= 120 else f"{text[:120]}..."
            lines.append(f"{idx}. {brief}")
        return "[你刚刚自己说过的话]\n" + "\n".join(lines)

    def _build_self_continuity_context(self) -> str:
        recent_bot_texts = self._recent_bot_texts()
        if not recent_bot_texts:
            return ""
        lines = []
        for idx, text in enumerate(recent_bot_texts[-3:], start=1):
            brief = text if len(text) <= 160 else f"{text[:160]}..."
            lines.append(f"- 第{idx}句：{brief}")
        return (
            "[连续对话参考]\n"
            "下面这些是你刚刚在这个群聊里亲口说过的话。\n"
            "如果现在继续说话，优先顺着它们往下接，不要像失忆一样重开话题。\n" + "\n".join(lines)
        )

    def _should_include_self_reference_context(self, target_message: Optional[Any]) -> bool:
        if target_message is None:
            return False
        if bool(getattr(target_message, "is_repeated_short_input", False)):
            return True
        if str(getattr(target_message, "context_continuation_hint", "") or "").strip():
            return True
        if bool(getattr(target_message, "is_quote_reply", False)):
            return True
        if bool(getattr(target_message, "is_reply_to_bot", False)):
            return True
        if bool(getattr(target_message, "is_quote_to_bot", False)):
            return True
        if self._message_explicitly_replies_to_bot(target_message):
            return True
        return False

    def _build_self_reference_parts(self, target_message: Optional[Any]) -> Dict[str, str]:
        if not self._should_include_self_reference_context(target_message):
            return {
                "self_memory": "",
                "continuity_context": "",
            }
        return {
            "self_memory": self._build_self_reply_memory(),
            "continuity_context": self._build_self_continuity_context(),
        }

    def _select_preferred_reply_message(
        self, anchor_message: Optional[Any], candidate_messages: List[Any]
    ) -> Optional[Any]:
        if anchor_message is None:
            return anchor_message
        try:
            from src.chat.behavior.target_selector import (
                choose_preferred_message_target,
            )

            relation_snapshot = self._resolve_relation_view()
            relationship_candidates = relation_snapshot.get("relationship_candidates", []) or []
            candidate_pool = self._get_human_message_candidates(candidate_messages)
            if not candidate_pool:
                candidate_pool = [
                    message for message in (candidate_messages or []) if not self._is_bot_message_obj(message)
                ]
            if candidate_pool and self._get_decision_message_source(anchor_message) != "incoming":
                anchor_message = candidate_pool[-1]
            active_user_ids = [
                str(getattr(message, "user_id", "") or "").strip()
                for message in candidate_pool
                if str(getattr(message, "user_id", "") or "").strip() and not self._is_bot_message_obj(message)
            ]
            preferred = choose_preferred_message_target(
                anchor_message=anchor_message,
                candidate_messages=candidate_pool or candidate_messages,
                relationship_candidates=relationship_candidates,
                active_user_ids=active_user_ids,
            )
            latest = candidate_pool[-1] if candidate_pool else None
            if self._should_keep_latest_reply_target(latest, preferred):
                return latest
            return preferred
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 构建锚定消息异常: {exc}")
            return anchor_message

    def _message_ts(self, msg: Any) -> float:
        try:
            return float(getattr(msg, "timestamp", 0.0) or getattr(msg, "time", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _message_user_id(self, msg: Any) -> str:
        return str(getattr(msg, "user_id", "") or "").strip()

    def _message_text(self, msg: Any) -> str:
        try:
            return str(self._extract_message_content(msg) or "").strip()
        except Exception:
            return str(
                getattr(msg, "processed_plain_text", "")
                or getattr(msg, "plain_text", "")
                or getattr(msg, "content", "")
                or ""
            ).strip()

    def _should_keep_latest_reply_target(self, latest: Any, preferred: Any) -> bool:
        if latest is None or preferred is None or latest is preferred:
            return False
        latest_text = self._message_text(latest)
        if not latest_text:
            return False
        latest_ts = self._message_ts(latest)
        preferred_ts = self._message_ts(preferred)
        if latest_ts and preferred_ts and latest_ts + 0.01 < preferred_ts:
            return False
        if latest_ts and preferred_ts and latest_ts - preferred_ts > 180.0:
            return True
        if self._message_user_id(latest) == self._message_user_id(preferred):
            latest_norm = latest_text.strip()
            preferred_norm = self._message_text(preferred).strip()
            if latest_norm and latest_norm != preferred_norm:
                return True
        return False

    def _build_current_target_message_block(self, target_message: Optional[Any]) -> str:
        if target_message is None:
            return ""
        text = self._message_text(target_message)
        if not text:
            return ""
        name = str(
            getattr(target_message, "user_nickname", "")
            or getattr(target_message, "nickname", "")
            or getattr(target_message, "user_cardname", "")
            or getattr(target_message, "user_id", "")
            or "对方"
        ).strip()
        return (
            "[当前必须回应的最新消息]\n"
            f"发送者：{name}\n"
            f"内容：{text[:240]}\n"
            "请优先回应这条消息。除非这条消息明确承接旧话题，否则不要把回复焦点转回更早的聊天内容。"
        )

    @staticmethod
    def _tag_decision_message(msg: Any, source: str) -> Any:
        if msg is None:
            return None
        normalized_source = str(source or "incoming").strip() or "incoming"
        try:
            msg._decision_message_source = normalized_source
        except Exception:
            return msg
        try:
            msg._is_context_backfill = normalized_source == "historical_context"
        except Exception:
            pass
        try:
            msg._is_synthetic_self_context = normalized_source == "synthetic_self" or bool(getattr(msg, "_is_synthetic_self_context", False))
        except Exception:
            pass
        return msg

    @staticmethod
    def _get_decision_message_source(msg: Any) -> str:
        source = str(getattr(msg, "_decision_message_source", "") or "").strip()
        if source:
            return source
        if bool(getattr(msg, "_is_context_backfill", False)):
            return "historical_context"
        if bool(getattr(msg, "_is_synthetic_self_context", False)):
            return "synthetic_self"
        return "incoming"

    def _get_human_message_candidates(
        self,
        messages: List,
        preferred_sources: Tuple[str, ...] = ("incoming",),
        allow_fallback: bool = True,
    ) -> List[Any]:
        preferred: List[Any] = []
        fallback: List[Any] = []
        for msg in messages or []:
            if not self._is_human_message_obj(msg):
                continue
            source = self._get_decision_message_source(msg)
            if source in preferred_sources:
                preferred.append(msg)
            elif allow_fallback and source != "synthetic_self":
                fallback.append(msg)
        return preferred or fallback

    def _get_latest_human_user_id(
        self,
        messages: List,
        preferred_sources: Tuple[str, ...] = ("incoming",),
        allow_fallback: bool = True,
    ) -> str:
        msg = self._get_latest_human_message(
            messages,
            preferred_sources=preferred_sources,
            allow_fallback=allow_fallback,
        )
        return str(getattr(msg, "user_id", "") or "").strip() if msg is not None else ""

    def _get_latest_bot_context_message(
        self,
        messages: List,
        preferred_sources: Tuple[str, ...] = ("incoming",),
        allow_synthetic_fallback: bool = True,
    ) -> Optional[Any]:
        synthetic_fallback = None
        for msg in reversed(messages or []):
            if not self._is_bot_message_obj(msg):
                continue
            source = self._get_decision_message_source(msg)
            if source in preferred_sources:
                return msg
            if allow_synthetic_fallback and synthetic_fallback is None and source == "synthetic_self":
                synthetic_fallback = msg
        return synthetic_fallback

    def _collect_recent_bot_utterance_entries(self, limit: int = 3) -> List[Dict[str, Any]]:
        try:
            from src.core.self_reply_recognizer import get_self_reply_recognizer

            recognizer_entries = get_self_reply_recognizer().build_recent_message_snapshots(self.stream_id, limit=limit)
            if recognizer_entries:
                return recognizer_entries
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 自我识别统一快照回填异常: {exc}")

        recent_items: List[Dict[str, Any]] = []
        seen_texts = set()
        self._prune_recent_bot_utterances()
        fallback_items = self._recent_bot_utterances[-limit:]
        if not fallback_items:
            try:
                from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

                fallback_items = acquire_reply_coordinator().recent_bot_utterance_entries(self.stream_id, limit=limit)
            except Exception as exc:
                logger.debug(f"{self.log_prefix} 回复协调正文回填异常: {exc}")
        if not fallback_items:
            fallback_items = [
                {"text": text, "ts": ts}
                for text, ts in self._load_recent_persisted_bot_texts(with_timestamps=True)[-limit:]
            ]
        for item in fallback_items:
            brief = str(item.get("text", "") or "").strip()
            if not brief or brief in seen_texts:
                continue
            seen_texts.add(brief)
            recent_items.append(
                {
                    "text": brief,
                    "ts": float(item.get("ts", 0.0) or 0.0) or time.time(),
                    "message_id": str(item.get("message_id", "") or ""),
                }
            )
            if len(recent_items) >= limit:
                break

        recent_items.sort(key=lambda item: float(item.get("ts", 0.0) or 0.0))
        return recent_items[-limit:]

    def _build_synthetic_self_messages(self) -> List[Any]:
        synthetic_messages: List[Any] = []
        recent_items = self._collect_recent_bot_utterance_entries(limit=3)
        if not recent_items:
            return synthetic_messages

        total = len(recent_items)
        for idx, item in enumerate(recent_items, start=1):
            brief = str(item.get("text", "") or "").strip()
            item_ts = float(item.get("ts", 0.0) or 0.0) or time.time()
            if not brief:
                continue
            synthetic_messages.append(
                SimpleNamespace(
                    user_id="bot",
                    processed_plain_text=brief,
                    plain_text=brief,
                    message_id=str(item.get("message_id", "") or f"self_ctx_{int(item_ts)}_{idx}"),
                    timestamp=item_ts - (total - idx) * 0.01,
                    is_mentioned=False,
                    is_at=False,
                    _is_synthetic_self_context=True,
                    _decision_message_source="synthetic_self",
                )
            )
        return synthetic_messages

    def _remember_bot_utterance(self, content: str) -> None:
        text = str(content or "").strip()
        if not text:
            return
        now = time.time()
        self._recent_bot_utterances.append({"text": text, "ts": now})
        self._prune_recent_bot_utterances(now)

    def _prune_recent_bot_utterances(self, now: Optional[float] = None) -> None:
        current = now if now is not None else time.time()
        ttl_sec = 180.0
        recent_items = []
        for item in self._recent_bot_utterances[-5:]:
            text = str(item.get("text", "") or "").strip()
            ts = float(item.get("ts", 0.0) or 0.0)
            if not text:
                continue
            if ts > 0 and (current - ts) <= ttl_sec:
                recent_items.append(
                    {
                        "text": text,
                        "ts": ts,
                        "message_id": str(item.get("message_id", "") or ""),
                    }
                )
        self._recent_bot_utterances = recent_items[-3:]

    def _recent_bot_texts(self) -> List[str]:
        return [
            str(item.get("text", "") or "").strip()
            for item in self._collect_recent_bot_utterance_entries(limit=3)
            if str(item.get("text", "") or "").strip()
        ]

    def _latest_recent_bot_utterance(self) -> Dict[str, Any]:
        recent_items = self._collect_recent_bot_utterance_entries(limit=1)
        return recent_items[-1] if recent_items else {}

    def _is_bot_message_obj(self, msg: Any) -> bool:
        if msg is None:
            return False
        add_cfg = (
            getattr(getattr(msg, "message_info", None), "additional_config", None)
            or getattr(msg, "additional_config", None)
            or {}
        )
        if isinstance(add_cfg, dict) and (add_cfg.get("is_bot_self") or add_cfg.get("is_bot")):
            return True
        user_id = str(getattr(msg, "user_id", "") or "").strip()
        if not user_id:
            return False
        if user_id == "bot":
            return True
        platform = str(
            getattr(msg, "user_platform", "")
            or getattr(getattr(msg, "chat_info", None), "platform", "")
            or getattr(getattr(msg, "message_info", None), "platform", "")
            or getattr(getattr(self, "chat_stream", None), "platform", "")
            or ""
        ).strip()
        if platform and is_bot_self(platform, user_id):
            return True
        try:
            from src.core.self_reply_recognizer import get_self_reply_recognizer

            return get_self_reply_recognizer().is_bot_message(msg)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 自我回复识别异常: {exc}")
            return False

    def _is_human_message_obj(self, msg: Any) -> bool:
        if msg is None:
            return False
        uid = str(getattr(msg, "user_id", "") or "").strip()
        if not uid:
            return False
        return not self._is_bot_message_obj(msg)

    def _has_recent_human_activity(self, window_sec: float = 180.0) -> bool:
        now = time.time()
        last_user_ts = float(getattr(self, "_last_user_msg_time", 0.0) or 0.0)
        if last_user_ts > 0 and (now - last_user_ts) <= max(1.0, float(window_sec or 0.0)):
            return True
        timeline = list(getattr(self, "_user_msg_timeline", []) or [])
        cutoff = now - max(1.0, float(window_sec or 0.0))
        return any(float(ts or 0.0) >= cutoff for ts in timeline)

    def _batch_has_human_messages(self, incoming_batch: Optional[List[Any]]) -> bool:
        for msg in incoming_batch or []:
            if self._is_human_message_obj(msg):
                return True
        return False

    def _should_short_circuit_self_echo_batch(self, incoming_batch: Optional[List[Any]]) -> bool:
        batch = list(incoming_batch or [])
        return bool(batch) and not self._batch_has_human_messages(batch)

    def _behavior_ratio(self, value: Any, default: float = 0.0) -> float:
        try:
            numeric = float(value if value is not None else default)
        except Exception:
            numeric = float(default or 0.0)
        if numeric > 1.0:
            numeric /= 100.0
        return max(0.0, min(1.0, numeric))

    def _summarize_behavior_governor(self, verdict: BehaviorGovernorVerdict) -> str:
        reasons = [str(code).strip() for code in list(getattr(verdict, "reason_codes", []) or []) if str(code).strip()]
        summary = "/".join(reasons[:3]) if reasons else verdict.reply_mode
        return (
            f"模式={self._BG_REPLY_MODE_CN.get(verdict.reply_mode, verdict.reply_mode)} "
            f"打断={self._BG_INTERRUPT_LEVEL_CN.get(verdict.interrupt_level, verdict.interrupt_level)} "
            f"引用={self._BG_QUOTE_POLICY_CN.get(verdict.quote_policy, verdict.quote_policy)} "
            f"静默={self._BG_SILENCE_POLICY_CN.get(verdict.silence_policy, verdict.silence_policy)} "
            f"允许={'是' if verdict.allow_generation else '否'} "
            f"观察上限={int(getattr(verdict, 'max_watch_rank', 1) or 1)} "
            f"模型={self._BG_MODEL_TIER_CN.get(verdict.model_tier, verdict.model_tier)} "
            f"原因={summary}"
        )

    def _summarize_rest_governor(self, verdict: RestGovernorVerdict) -> str:
        reasons = [str(code).strip() for code in list(getattr(verdict, "reason_codes", []) or []) if str(code).strip()]
        summary = "/".join(reasons[:3]) if reasons else verdict.posture
        return (
            f"姿态={self._RG_POSTURE_CN.get(verdict.posture, verdict.posture)} "
            f"打断策略={self._RG_INTERRUPTION_POLICY_CN.get(verdict.interruption_policy, verdict.interruption_policy)} "
            f"休息={'是' if verdict.should_rest else '否'} "
            f"摸鱼={'是' if verdict.should_loaf else '否'} "
            f"原因={summary}"
        )

    def _summarize_model_governor(self, verdict: ModelGovernorVerdict) -> str:
        reasons = [
            str(code).strip()
            for code in list(getattr(verdict, "upgrade_reason_codes", []) or [])
            if str(code).strip()
        ]
        summary = "/".join(reasons[:3]) if reasons else verdict.tier
        return (
            f"等级={self._MG_TIER_CN.get(verdict.tier, verdict.tier)} "
            f"限流={'是' if verdict.rate_limited else '否'} "
            f"降级={'是' if verdict.fallback_to_small else '否'} "
            f"冷却={float(getattr(verdict, 'dynamic_cooldown_sec', 0.0) or 0.0):.0f}秒 "
            f"小时上限={int(getattr(verdict, 'dynamic_hourly_cap', 0) or 0)} "
            f"原因={summary}"
        )

    def _compute_dynamic_large_model_constraints(
        self,
        *,
        now: float,
        is_proactive: bool,
        targeted: bool,
        high_risk: bool,
        behavior_verdict: Optional[BehaviorGovernorVerdict],
        source: str,
    ) -> Tuple[float, int]:
        """按上下文动态计算大模型冷却和小时预算，避免硬编码固定秒数/固定次数。"""
        unanswered = int(getattr(self, "_unanswered_bot_turns", 0) or 0)
        chatter = float(getattr(self, "_chatterbox_penalty", 0.0) or 0.0)
        consecutive = float(getattr(self, "_consecutive_speaks", 0.0) or 0.0)
        proactive_hourly = int(self._current_hourly_proactive_reply_count(now))
        recent_human = bool(self._has_recent_human_activity(900.0))
        last_user_ts = float(getattr(self, "_last_user_msg_time", 0.0) or 0.0)
        silence_sec = max(0.0, now - last_user_ts) if last_user_ts > 0.0 else 3600.0
        interrupt_level = str(getattr(behavior_verdict, "interrupt_level", "") or "").strip().lower()
        reply_mode = str(getattr(behavior_verdict, "reply_mode", "") or "").strip().lower()
        source_label = str(source or "").strip().lower()
        source_proactive = bool(is_proactive or "proactive" in source_label)

        cooldown = 25.0
        cooldown += float(min(6, unanswered)) * 16.0
        cooldown += max(0.0, chatter - 0.6) * 30.0
        cooldown += max(0.0, consecutive - 2.0) * 7.5
        if source_proactive:
            cooldown += float(max(0, proactive_hourly - 1)) * 12.0
        if silence_sec < 180.0:
            cooldown += 30.0
        elif silence_sec < 600.0:
            cooldown += 12.0
        if not recent_human:
            cooldown += 18.0
        if targeted:
            cooldown -= 28.0
        if high_risk:
            cooldown -= 20.0
        if interrupt_level == "engage":
            cooldown -= 10.0
        elif interrupt_level == "ignore":
            cooldown += 8.0
        if reply_mode in {"rest", "defer", "observe"}:
            cooldown += 14.0
        if reply_mode == "reply":
            cooldown -= 6.0
        if source_proactive and reply_mode != "proactive":
            cooldown += 10.0
        cooldown = max(15.0, min(420.0, cooldown))

        hourly_cap = 2
        if source_proactive:
            hourly_cap += 1
        if recent_human:
            hourly_cap += 1
        if targeted:
            hourly_cap += 1
        if high_risk:
            hourly_cap += 1
        if interrupt_level == "engage":
            hourly_cap += 1
        if silence_sec < 300.0:
            hourly_cap += 1
        if unanswered >= 3:
            hourly_cap -= 1
        if chatter >= 2.0:
            hourly_cap -= 1
        if consecutive >= 5.0:
            hourly_cap -= 1
        if reply_mode in {"rest", "defer", "observe"}:
            hourly_cap -= 1
        if not recent_human and source_proactive:
            hourly_cap -= 1
        hourly_cap = int(max(1, min(8, hourly_cap)))
        return cooldown, hourly_cap

    def _evaluate_rest_governor(
        self,
        *,
        incoming_batch: Optional[List[Any]] = None,
        source: str = "",
        targeted_to_bot: bool = False,
        admin_force: bool = False,
        pinged_msg: Any = None,
    ) -> RestGovernorVerdict:
        """统一收口“什么时候该歇/该摸鱼/不该被弱消息打断”。

        行为 Governor 决定“要不要说”，Rest Governor 决定“当前姿态允许被什么级别的刺激拉起”。
        """
        verdict = RestGovernorVerdict()
        if admin_force:
            verdict.reason_codes.append("admin_force")
            self._last_rest_governor_verdict = verdict
            return verdict

        batch = list(incoming_batch or [])
        human_messages = self._get_human_message_candidates(
            batch,
            preferred_sources=("incoming",),
            allow_fallback=True,
        )
        direct_target = bool(targeted_to_bot or pinged_msg is not None or any(self._message_targets_bot(msg) for msg in human_messages))
        recent_human_activity = bool(human_messages) or self._has_recent_human_activity(240.0)
        presence_state = getattr(self, "_cached_presence_state", None)
        emotion_state = getattr(self, "_cached_emotion_state", None)
        metabolism = getattr(self, "_cached_metabolism_constraints", None) or {}

        quiet_preference = self._behavior_ratio(getattr(presence_state, "quiet_preference", 0.2), 0.2)
        avoidance = self._behavior_ratio(getattr(presence_state, "avoidance_tendency", 0.0), 0.0)
        social_willingness = self._behavior_ratio(getattr(presence_state, "social_willingness", 0.5), 0.5)
        loafing = self._behavior_ratio(
            metabolism.get("loafing_level", metabolism.get("loafing", metabolism.get("loafing_suppression", 0.0))),
            0.0,
        )
        boredom = self._behavior_ratio(
            getattr(emotion_state, "boredom", metabolism.get("boredom_level", metabolism.get("boredom", 0.0))),
            0.0,
        )
        loneliness = self._behavior_ratio(getattr(emotion_state, "loneliness", 0.0), 0.0)
        energy_ratio = self._behavior_ratio(metabolism.get("chat_value", getattr(self, "_chat_energy", 50.0)), 0.5)
        thinking_ratio = self._behavior_ratio(
            metabolism.get("thinking_value", getattr(self, "_thinking_energy", 50.0)),
            0.5,
        )
        unanswered = int(getattr(self, "_unanswered_bot_turns", 0) or 0)
        watch_rank = int(getattr(presence_state, "watch_state_rank", 2) or 2)
        thresholds = get_heartfc_thresholds()

        night_phase = getattr(self, "_cached_night_phase", None)
        night_name = getattr(night_phase, "name", str(night_phase)) if night_phase is not None else ""
        if "DEEP" in night_name or "VALLEY" in night_name or "BURNED" in night_name:
            verdict.posture = "rest"
            verdict.interruption_policy = "block"
            verdict.should_rest = True
            verdict.reason_codes.append("night_rest")
        elif energy_ratio <= thresholds.rest_energy_ratio or thinking_ratio <= thresholds.rest_thinking_ratio:
            verdict.posture = "rest"
            verdict.interruption_policy = "block" if not direct_target else "peek_only"
            verdict.should_rest = True
            verdict.reason_codes.append("resource_exhausted")
        elif (
            loafing >= thresholds.loafing_high
            and quiet_preference >= thresholds.quiet_high
            and avoidance >= thresholds.avoidance_high
            and not direct_target
        ):
            verdict.posture = "loaf"
            verdict.interruption_policy = "block" if not recent_human_activity else "peek_only"
            verdict.should_loaf = True
            verdict.reason_codes.append("high_loafing_quiet")
        elif (
            loafing >= thresholds.loafing_medium
            and (quiet_preference >= thresholds.quiet_medium or avoidance >= thresholds.avoidance_medium)
            and not direct_target
        ):
            verdict.posture = "peek_only"
            verdict.interruption_policy = "peek_only"
            verdict.should_loaf = True
            verdict.reason_codes.append("soft_loafing_guard")
        elif unanswered >= 3 and not recent_human_activity and source == "proactive":
            verdict.posture = "rest"
            verdict.interruption_policy = "block"
            verdict.should_rest = True
            verdict.reason_codes.append("ignored_chain_rest")
        elif watch_rank <= 1 and quiet_preference >= thresholds.quiet_watch_low and not direct_target:
            verdict.posture = "peek_only"
            verdict.interruption_policy = "peek_only"
            verdict.reason_codes.append("watch_low_quiet")

        if direct_target:
            verdict.interruption_policy = "allow"
            if verdict.posture == "peek_only":
                verdict.reason_codes.append("direct_target_override")
            elif verdict.posture == "rest" and not verdict.should_rest:
                verdict.reason_codes.append("direct_target_override")
        elif (
            verdict.interruption_policy == "allow"
            and not recent_human_activity
            and loafing >= thresholds.loafing_idle
            and max(boredom, loneliness, social_willingness) < thresholds.social_low_willingness
        ):
            verdict.posture = "loaf"
            verdict.interruption_policy = "peek_only"
            verdict.should_loaf = True
            verdict.reason_codes.append("quiet_idle_loaf")

        deduped_codes: List[str] = []
        for code in verdict.reason_codes:
            normalized = str(code).strip()
            if normalized and normalized not in deduped_codes:
                deduped_codes.append(normalized)
        verdict.reason_codes = deduped_codes
        self._last_rest_governor_verdict = verdict
        return verdict

    def _evaluate_behavior_governor(
        self,
        *,
        incoming_batch: Optional[List[Any]] = None,
        silence_sec: float = 0.0,
        requested_mode: str = "reactive",
        target_message: Any = None,
        delivery_form: str = "",
        reference_user_name: str = "",
        mention_user_name: str = "",
        is_background: bool = False,
    ) -> BehaviorGovernorVerdict:
        """统一评估“该不该说 / 该不该打断 / 该怎么接前情”。

        目标不是替代所有下游判断，而是把高频硬门、主动节流和引用策略收口成一份稳定 verdict。
        """
        batch = list(incoming_batch or [])
        human_messages = self._get_human_message_candidates(
            batch,
            preferred_sources=("incoming",),
            allow_fallback=True,
        )
        latest_human = human_messages[-1] if human_messages else None
        if target_message is not None and self._is_human_message_obj(target_message):
            latest_human = target_message

        presence_state = getattr(self, "_cached_presence_state", None)
        emotion_state = getattr(self, "_cached_emotion_state", None)
        metabolism = getattr(self, "_cached_metabolism_constraints", None) or {}
        scene_snapshot = getattr(self, "_cached_scene_snapshot", None)

        watch_rank = int(getattr(presence_state, "watch_state_rank", 2) or 2)
        quiet_preference = self._behavior_ratio(getattr(presence_state, "quiet_preference", 0.2), 0.2)
        avoidance = self._behavior_ratio(getattr(presence_state, "avoidance_tendency", 0.0), 0.0)
        social_willingness = self._behavior_ratio(getattr(presence_state, "social_willingness", 0.5), 0.5)

        boredom = self._behavior_ratio(
            getattr(emotion_state, "boredom", metabolism.get("boredom_level", metabolism.get("boredom", 0.0))),
            0.0,
        )
        loneliness = self._behavior_ratio(getattr(emotion_state, "loneliness", social_willingness), social_willingness)
        social_desire = self._behavior_ratio(getattr(emotion_state, "social_desire", social_willingness), social_willingness)
        loafing = self._behavior_ratio(
            metabolism.get("loafing_level", metabolism.get("loafing", metabolism.get("loafing_suppression", 0.0))),
            0.0,
        )
        energy_ratio = self._behavior_ratio(metabolism.get("chat_value", getattr(self, "_chat_energy", 50.0)), 0.5)
        thinking_ratio = self._behavior_ratio(
            metabolism.get("thinking_value", getattr(self, "_thinking_energy", 50.0)),
            0.5,
        )
        unanswered = int(getattr(self, "_unanswered_bot_turns", 0) or 0)
        chatterbox_penalty = float(getattr(self, "_chatterbox_penalty", 0.0) or 0.0)
        consecutive_speaks = float(getattr(self, "_consecutive_speaks", 0.0) or 0.0)
        recent_human_activity = bool(human_messages) or self._has_recent_human_activity(300.0)
        recent_bot_only_burst = sum(1 for ts in (self._bot_reply_timeline or []) if (time.time() - float(ts or 0.0)) <= 240.0)

        direct_relevance = any(self._message_targets_bot(msg) for msg in human_messages)
        explicit_quote_anchor = bool(reference_user_name.strip() or "quote" in str(delivery_form or ""))
        if latest_human is not None:
            explicit_quote_anchor = explicit_quote_anchor or bool(
                getattr(latest_human, "is_quote_reply", False)
                or getattr(latest_human, "is_reply_to_bot", False)
                or getattr(latest_human, "reply_to", "")
                or getattr(latest_human, "reply_to_message_id", None)
            )
        scene_suitable = True
        if scene_snapshot is not None:
            scene_suitable = bool(getattr(scene_snapshot, "suitable_to_join", True))

        verdict = BehaviorGovernorVerdict()
        verdict.target_user_id = str(getattr(latest_human, "user_id", "") or "").strip()

        if direct_relevance or explicit_quote_anchor or mention_user_name.strip():
            verdict.interrupt_level = "engage"
            verdict.reason_codes.append("direct_relevance")
        elif human_messages:
            if watch_rank <= 1 or quiet_preference >= 0.70 or avoidance >= 0.70:
                verdict.interrupt_level = "peek"
                verdict.reason_codes.append("weak_human_signal")
            else:
                verdict.interrupt_level = "skim"
                verdict.reason_codes.append("ambient_human_signal")
        elif recent_human_activity:
            verdict.interrupt_level = "peek"
            verdict.reason_codes.append("recent_human_activity")

        emotional_pull = max(boredom, loneliness, social_desire)
        if recent_human_activity:
            verdict.silence_policy = "ambient"
        elif emotional_pull >= 0.58 and loafing < 0.78 and thinking_ratio >= 0.22 and energy_ratio >= 0.15:
            verdict.silence_policy = "reengage"
            verdict.reason_codes.append("interest_rebound")
        else:
            verdict.silence_policy = "silent"
            verdict.reason_codes.append("default_silence")

        target_age_sec = float("inf")
        if latest_human is not None:
            try:
                target_age_sec = max(
                    0.0,
                    time.time()
                    - float(
                        getattr(latest_human, "timestamp", 0.0)
                        or getattr(latest_human, "time", 0.0)
                        or time.time()
                    ),
                )
            except Exception:
                target_age_sec = float("inf")

        _quote_fresh_sec = 300.0
        _quote_soft_sec = 1200.0
        if latest_human is None or not self._is_human_message_obj(latest_human):
            verdict.quote_policy = "none"
            verdict.reason_codes.append("no_anchor_quote_none")
        elif explicit_quote_anchor and target_age_sec <= _quote_fresh_sec:
            verdict.quote_policy = "quote_reply"
            verdict.reason_codes.append("explicit_quote_anchor")
        elif explicit_quote_anchor and target_age_sec <= _quote_soft_sec:
            verdict.quote_policy = "soft_reference"
            verdict.reason_codes.append("anchor_soft_reference")
        elif explicit_quote_anchor:
            verdict.quote_policy = "none"
            verdict.reason_codes.append("anchor_timeout_quote_none")
        elif direct_relevance and target_age_sec <= _quote_fresh_sec and requested_mode != "proactive":
            verdict.quote_policy = "quote_reply"
            verdict.reason_codes.append("recent_targeted_quote")
        elif target_age_sec <= _quote_soft_sec:
            verdict.quote_policy = "soft_reference"
            verdict.reason_codes.append("stale_context_soft_reference")
        else:
            verdict.quote_policy = "none"
            verdict.reason_codes.append("context_timeout_quote_none")

        night_phase = getattr(self, "_cached_night_phase", None)
        night_name = getattr(night_phase, "name", str(night_phase)) if night_phase is not None else ""
        if "DEEP" in night_name or "VALLEY" in night_name or "BURNED" in night_name:
            verdict.reply_mode = "rest"
            verdict.silence_policy = "silent"
            verdict.reason_codes.append("night_freeze")
        elif requested_mode == "proactive":
            bg_block_reason = self._background_proactive_block_reason() if is_background else ""
            idle_backoff = self._should_back_off_idle_proactive()
            if bg_block_reason:
                verdict.reply_mode = "rest" if unanswered >= 5 else "defer"
                verdict.silence_policy = "silent"
                verdict.reason_codes.append("ignored_chain")
            elif unanswered >= 5 or (unanswered >= 3 and chatterbox_penalty >= 2.0):
                verdict.reply_mode = "rest"
                verdict.silence_policy = "silent"
                verdict.reason_codes.append("chatterbox_guard")
            elif unanswered >= 3 and not recent_human_activity:
                verdict.reply_mode = "defer"
                verdict.silence_policy = "silent"
                verdict.reason_codes.append("ignored_chain")
            elif unanswered >= 1 and recent_bot_only_burst >= 1 and not recent_human_activity:
                verdict.reply_mode = "defer"
                verdict.silence_policy = "silent"
                verdict.reason_codes.append("air_chat_guard")
            elif consecutive_speaks >= 5.0 and unanswered >= 1:
                verdict.reply_mode = "defer"
                verdict.silence_policy = "silent"
                verdict.reason_codes.append("consecutive_speaks_guard")
            elif idle_backoff:
                verdict.reply_mode = "observe"
                verdict.silence_policy = "silent"
                verdict.reason_codes.append("idle_backoff")
            elif verdict.silence_policy == "silent":
                verdict.reply_mode = "observe"
            else:
                verdict.reply_mode = "proactive"
                if scene_suitable:
                    verdict.reason_codes.append("scene_allows_reengage")
        else:
            if direct_relevance:
                verdict.reply_mode = "reply"
            elif verdict.interrupt_level in {"peek", "skim"}:
                verdict.reply_mode = "observe"
            else:
                verdict.reply_mode = "defer"

        if scene_snapshot is not None and not scene_suitable and verdict.reply_mode == "proactive":
            verdict.reply_mode = "defer"
            verdict.reason_codes.append("scene_unsuitable")
        if silence_sec >= 600.0 and not recent_human_activity and verdict.reply_mode == "proactive":
            verdict.reason_codes.append("long_silence")

        _watch_cap = 1
        if verdict.interrupt_level == "ignore":
            _watch_cap = 0 if verdict.silence_policy == "silent" and not recent_human_activity else 1
        elif verdict.interrupt_level == "peek":
            _watch_cap = 1
        elif verdict.interrupt_level == "skim":
            _watch_cap = 2
        elif verdict.interrupt_level == "engage":
            _watch_cap = 4 if verdict.reply_mode == "reply" else 3
        if verdict.reply_mode in {"rest", "defer"}:
            _watch_cap = min(_watch_cap, 1)
        if requested_mode == "proactive" and (not recent_human_activity or verdict.silence_policy == "silent"):
            _watch_cap = min(_watch_cap, 1)
        if direct_relevance:
            _watch_cap = max(_watch_cap, 3)
        verdict.max_watch_rank = int(max(0, min(4, _watch_cap)))
        verdict.allow_generation = verdict.reply_mode in {"reply", "proactive"}

        _relation_for_model = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
        _model_risky = (
            float(_relation_for_model.get("annoyance_value", 0.0) or 0.0) >= 65.0
            or float(_relation_for_model.get("psychological_pressure", 0.0) or 0.0) >= 55.0
        )
        if not verdict.allow_generation:
            verdict.model_tier = "skip"
        elif direct_relevance and _model_risky and requested_mode != "proactive":
            verdict.model_tier = "large"
            verdict.reason_codes.append("risk_large_candidate")
        else:
            verdict.model_tier = "small"

        deduped_codes: List[str] = []
        for code in verdict.reason_codes:
            normalized = str(code).strip()
            if normalized and normalized not in deduped_codes:
                deduped_codes.append(normalized)
        verdict.reason_codes = deduped_codes
        self._last_behavior_governor_verdict = verdict
        return verdict

    def _record_large_model_usage(self, *, now: float, is_proactive: bool) -> None:
        if not is_proactive:
            return
        if self._model_large_hour_window_start <= 0.0 or (now - self._model_large_hour_window_start) >= 3600.0:
            self._model_large_hour_window_start = now
            self._model_large_proactive_hour_calls = 0
        self._model_large_last_ts = now
        self._model_large_proactive_hour_calls += 1

    def _evaluate_model_governor(
        self,
        *,
        now: float,
        desired_level: int,
        incoming_batch: Optional[List[Any]] = None,
        pinged_msg: Any = None,
        is_proactive: bool = False,
        source: str = "",
        behavior_verdict: Optional[BehaviorGovernorVerdict] = None,
        high_risk: bool = False,
    ) -> ModelGovernorVerdict:
        verdict = ModelGovernorVerdict()
        reason_codes: List[str] = []
        batch = list(incoming_batch or [])
        admin_force = bool(getattr(self, "_is_admin_forced", False))
        if not admin_force:
            try:
                admin_force = bool(self._is_force_wake_admin(batch, pinged_msg))
            except Exception as exc:
                logger.debug(f"{self.log_prefix} 模型Governor管理员识别失败: {exc}")
                admin_force = False
        targeted = bool(
            pinged_msg is not None
            or self._has_targeted_bot_message(batch)
            or bool(getattr(self, "_cached_targeted_to_bot", False))
            or admin_force
        )
        if admin_force and desired_level <= 0:
            desired_level = 1
            reason_codes.append("admin_force_model_bypass")

        if behavior_verdict is not None and str(getattr(behavior_verdict, "model_tier", "") or "").strip() == "skip" and not targeted:
            verdict.tier = "skip"
            reason_codes.append("behavior_model_skip")
        elif behavior_verdict is not None and not bool(getattr(behavior_verdict, "allow_generation", True)) and not targeted:
            verdict.tier = "skip"
            reason_codes.append("behavior_blocked")
        elif desired_level <= 0:
            verdict.tier = "skip"
            reason_codes.append("low_signal_skip")
        else:
            verdict.tier = "small"
            reason_codes.append("small_default")

        _source = str(source or ("proactive" if is_proactive else "reactive")).strip().lower()
        dynamic_cooldown_sec, dynamic_hourly_cap = self._compute_dynamic_large_model_constraints(
            now=now,
            is_proactive=is_proactive,
            targeted=targeted,
            high_risk=high_risk,
            behavior_verdict=behavior_verdict,
            source=_source,
        )
        verdict.dynamic_cooldown_sec = float(dynamic_cooldown_sec)
        verdict.dynamic_hourly_cap = int(dynamic_hourly_cap)
        candidate_large = desired_level >= 2 and verdict.tier != "skip"
        high_value = bool(
            targeted
            or high_risk
            or (
                behavior_verdict is not None
                and str(getattr(behavior_verdict, "interrupt_level", "") or "") == "engage"
            )
        )
        if candidate_large:
            if not high_value:
                reason_codes.append("large_demoted_noncritical")
            else:
                if is_proactive or "proactive" in _source:
                    if self._model_large_hour_window_start <= 0.0 or (
                        now - self._model_large_hour_window_start
                    ) >= 3600.0:
                        self._model_large_hour_window_start = now
                        self._model_large_proactive_hour_calls = 0
                    _elapsed = max(0.0, now - float(self._model_large_last_ts or 0.0))
                    if self._model_large_last_ts > 0.0 and _elapsed < dynamic_cooldown_sec:
                        verdict.rate_limited = True
                        verdict.fallback_to_small = True
                        reason_codes.append(
                            f"large_dynamic_cooldown(elapsed={_elapsed:.0f}s<need={dynamic_cooldown_sec:.0f}s)"
                        )
                    elif self._model_large_proactive_hour_calls >= dynamic_hourly_cap:
                        verdict.rate_limited = True
                        verdict.fallback_to_small = True
                        reason_codes.append(
                            f"large_dynamic_hour_cap(used={self._model_large_proactive_hour_calls}/cap={dynamic_hourly_cap})"
                        )
                if not verdict.rate_limited:
                    verdict.tier = "large"
                    reason_codes.append("large_high_value")

        if verdict.rate_limited and verdict.tier != "skip":
            verdict.tier = "small"
        if verdict.tier == "small" and behavior_verdict is not None and behavior_verdict.model_tier == "skip":
            reason_codes.append("behavior_prefers_skip")
        deduped: List[str] = []
        for code in reason_codes:
            normalized = str(code).strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        verdict.upgrade_reason_codes = deduped
        self._last_model_governor_verdict = verdict
        return verdict

    def _background_proactive_block_reason(self) -> str:
        unanswered = int(getattr(self, "_unanswered_bot_turns", 0) or 0)
        chatter = float(getattr(self, "_chatterbox_penalty", 0.0) or 0.0)
        consecutive = float(getattr(self, "_consecutive_speaks", 0.0) or 0.0)
        if unanswered >= 3 and not self._has_recent_human_activity(900.0):
            return f"长期无人回应(unanswered={unanswered})"
        if chatter >= 2.0 and unanswered >= 2:
            return f"话痨惩罚过高(chatter={chatter:.1f}, unanswered={unanswered})"
        if consecutive >= 5.0 and unanswered >= 1:
            return f"连续发言过多(consecutive={consecutive:.1f})"
        return ""

    def _build_proactive_no_target_guard(self, incoming_batch: Optional[List[Any]]) -> str:
        human_msgs = self._get_human_message_candidates(
            list(incoming_batch or []),
            preferred_sources=("incoming", "history", "recent"),
            allow_fallback=True,
        )
        if human_msgs and self._has_recent_human_activity(600.0):
            return ""
        return (
            "[主动边界]\n"
            "当前没有明确的对话对象，也没有人刚刚来找你。\n"
            "如果你要主动开口，只能轻量、开放式、留退路地抛一句话题或状态。\n"
            "不要像在催人、逼问人、命令人，禁止使用“说事”“直说”“有事就说”“看见了”这类压迫式开场。"
        )

    def _extract_plain_text_reply(self, reply_set: Any) -> str:
        parts: List[str] = []
        for item in list(getattr(reply_set, "reply_data", []) or []):
            content_type = getattr(item, "content_type", "")
            content_type_val = content_type.value if hasattr(content_type, "value") else str(content_type)
            if content_type_val != "text":
                return ""
            parts.append(str(getattr(item, "content", "") or ""))
        return "".join(parts).strip()

    def _replace_reply_set_plain_text(self, reply_set: Any, text: str) -> bool:
        try:
            from src.common.data_models.message_data_model import ReplyContent, ReplyContentType

            cleaned = str(text or "").strip()
            if not cleaned:
                return False
            reply_set.reply_data = [
                ReplyContent(content_type=ReplyContentType.TEXT, content=cleaned)
            ]
            return True
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 替换回复文本失败: {exc}")
            return False

    @staticmethod
    def _looks_customer_service_reply(text: str) -> bool:
        payload = str(text or "").strip()
        if not payload:
            return False
        strong_markers = (
            "建议您",
            "请您",
            "请问",
            "感谢理解",
            "谢谢配合",
            "为便于",
            "便于我",
            "烦请",
        )
        if any(marker in payload for marker in strong_markers):
            return True
        soft_markers = ("您", "整理完整", "一次性发送", "全面理解", "更便于", "完整后")
        return sum(1 for marker in soft_markers if marker in payload) >= 2

    @classmethod
    def _looks_over_formal_rewrite(cls, original_text: str, rewritten_text: str) -> bool:
        rewritten = str(rewritten_text or "").strip()
        if not rewritten:
            return False
        if cls._looks_customer_service_reply(rewritten):
            return True
        original = str(original_text or "").strip()
        return len(rewritten) > max(len(original) * 2 + 8, len(original) + 18)

    @staticmethod
    def _soften_high_risk_reply_locally(original_text: str) -> str:
        payload = str(original_text or "").strip()
        if not payload:
            return ""
        coarse_markers = (
            "复读机",
            "说全",
            "到底要说啥",
            "说事啊",
            "有事就直说",
            "扯淡",
            "滚",
            "人话",
        )
        if any(marker in payload for marker in coarse_markers):
            return "慢慢说，具体咋了？"
        if ("能不能" in payload and "说" in payload) or ("你这" in payload and "吗" in payload):
            return "你展开说说呗。"
        return ""

    async def _apply_pre_send_reply_guard(
        self,
        reply_set: Any,
        *,
        audit_label: str,
        risk_note: str = "",
    ) -> Tuple[str, bool, str, bool]:
        """发送前快速审查：仅做本地规则检查，不调LLM。LLM审查移至发送后异步执行。"""
        original_text = self._extract_plain_text_reply(reply_set)
        if not original_text:
            return "", False, "", False
        _local_flagged, _local_reason = self._quick_local_risk_check(original_text)
        if _local_flagged:
            fallback_text = self._soften_high_risk_reply_locally(original_text)
            if fallback_text and fallback_text != original_text:
                if self._replace_reply_set_plain_text(reply_set, fallback_text):
                    logger.info(
                        f"{self.log_prefix} 🛡️ 本地风险拦截({audit_label}): {_local_reason[:80]} | "
                        f"{original_text[:40]} -> {fallback_text[:40]}"
                    )
                    return fallback_text, True, _local_reason, False
            logger.info(
                f"{self.log_prefix} 🧯 本地风险拦截取消({audit_label}): {_local_reason[:80]}"
            )
            return "", True, _local_reason, True
        self._spawn(self._post_send_llm_audit(original_text, audit_label, risk_note))
        return original_text, False, "", False

    def _quick_local_risk_check(self, text: str) -> Tuple[bool, str]:
        """纯本地规则快速风险检查，不调LLM。"""
        if not text:
            return False, ""
        _banned_patterns = [
            ("自杀", "涉及自伤内容"),
            ("自残", "涉及自伤内容"),
            ("去死", "攻击性内容"),
            ("杀了", "暴力内容"),
        ]
        for pattern, reason in _banned_patterns:
            if pattern in text:
                return True, reason
        return False, ""

    async def _post_send_llm_audit(self, original_text: str, audit_label: str, risk_note: str = ""):
        """发送后异步LLM审查，发现高风险则记录日志。"""
        try:
            from src.chat.heart_flow.skills.focus_patrol import acquire_focus_patrol
            _patrol = acquire_focus_patrol()
            try:
                _style_hint = self._build_reply_style_context(getattr(self, "_last_relation_snapshot", None) or None)
            except Exception:
                _style_hint = "自然口语，别客服腔，别写成说明书。"
            try:
                _recent_bot_samples = self._load_recent_persisted_bot_texts()
            except Exception:
                _recent_bot_samples = []
            if hasattr(_patrol, "push_review_context"):
                _patrol.push_review_context(
                    style_hint=_style_hint,
                    recent_bot_samples=_recent_bot_samples,
                    audit_label=audit_label,
                )
            try:
                _, flagged, reason = await _patrol.review_before_send(
                    original_text,
                    risk_note=risk_note,
                )
            finally:
                if hasattr(_patrol, "clear_review_context"):
                    _patrol.clear_review_context()
            if flagged:
                logger.warning(
                    f"{self.log_prefix} ⚠️ 发送后LLM审查发现风险({audit_label}): "
                    f"{reason[:80]} | 内容: {original_text[:60]}"
                )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 发送后LLM审查异常({audit_label}): {exc}")

    def _load_recent_persisted_bot_texts(self, with_timestamps: bool = False) -> List[Any]:
        try:
            recent_messages = message_api.get_messages_by_time_in_chat(
                chat_id=self.stream_id,
                start_time=time.time() - 1800,
                end_time=time.time(),
                limit=12,
                limit_mode="latest",
                filter_mai=False,
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 加载最近bot历史失败: {exc}")
            return []

        results: List[Any] = []
        seen: set[str] = set()
        for msg in reversed(recent_messages or []):
            if not self._is_bot_message_obj(msg):
                continue
            text = str(
                getattr(msg, "processed_plain_text", "")
                or getattr(msg, "plain_text", "")
                or getattr(msg, "content", "")
                or ""
            ).strip()
            try:
                from src.modules.recall.self_awareness import get_self_awareness

                item = get_self_awareness().get_message(str(getattr(msg, "message_id", "") or "").strip())
                context = getattr(item, "context", None) or {}
                if isinstance(context, dict):
                    raw_reply = str(context.get("raw_reply", "") or "").strip()
                    if raw_reply:
                        text = raw_reply
            except Exception:
                pass
            if not text or text in seen:
                continue
            seen.add(text)
            ts = float(getattr(msg, "timestamp", 0.0) or getattr(msg, "time", 0.0) or time.time())
            results.append((text, ts) if with_timestamps else text)
        return results[-3:]

    def _should_back_off_idle_proactive(self) -> bool:
        """自适应退避：由内心独白欲望、无聊/孤独情绪综合调节冷却
        不用硬编码阈值，让情感系统自然累积感受来调节频率。
        仅在极端情况（多次无回复+低欲望）才硬性阻断。
        """
        if self._unanswered_bot_turns < 1:
            return False
        # 采集内心独白欲望值（小模型已决定的行动意愿）
        _voice_desire = 5
        _cached_voice = getattr(self, "_cached_voice", None)
        if _cached_voice is not None and hasattr(_cached_voice, "reply_desire_level"):
            _voice_desire = int(getattr(_cached_voice, "reply_desire_level", 5) or 5)
        # 从情感核心或缓存代谢态获取无聊度/孤独感
        _boredom = 0.0
        _loneliness = 0.0
        try:
            from src.chat.heart_flow.emotion_driven_core import get_emotion_driven_core

            _emo_snap = get_emotion_driven_core().get_state_snapshot(self.stream_id)
            _boredom = float(_emo_snap.get("boredom", 0.0) or 0.0)
            _loneliness = float(_emo_snap.get("loneliness", 0.0) or 0.0)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 情绪状态快照读取异常: {exc}")
            _mc = getattr(self, "_cached_metabolism_constraints", {}) or {}
            _boredom_raw = _mc.get("boredom_level", _mc.get("boredom", 0.0))
            _boredom = max(0.0, min(1.0, float(_boredom_raw or 0.0)))
            if _boredom > 1.0:
                _boredom = max(0.0, min(1.0, _boredom / 100.0))
            _ps = getattr(self, "_cached_presence_state", None)
            if _ps:
                _loneliness = float(getattr(_ps, "social_willingness", 0.3) or 0.3)
        _emotional_drive = max(_boredom, _loneliness)
        try:
            from src.chat.heart_flow.skills.rest_timing import acquire_rest_timing_engine

            _recent_reply_burst = sum(
                1
                for t in self._proactive_reply_timeline
                if time.time() - t < 180.0
            )
            self._IDLE_PROACTIVE_COOLDOWN_SEC = acquire_rest_timing_engine().estimate_reply_cooldown(
                unanswered_turns=self._unanswered_bot_turns,
                voice_desire=_voice_desire,
                emotional_drive=_emotional_drive,
                recent_reply_burst=_recent_reply_burst,
            )
        except Exception as _exc:
            logger.debug(f"{self.log_prefix} 主动冷却评估异常: {_exc}")
            self._IDLE_PROACTIVE_COOLDOWN_SEC = 120.0
        # 只剩自己在说时，不要继续把沉默误当作“还能接着聊”
        if (
            self._unanswered_bot_turns >= 1
            and _recent_reply_burst >= 1
            and not self._has_recent_human_activity(180.0)
            and _voice_desire < 8
        ):
            return True
        # 极端情况：8次以上未回复且欲望低，彻底冷却
        if self._unanswered_bot_turns >= 8 and _voice_desire < 4:
            return True
        # 不强制阻断，让冷却计时器和频率控制共同调节节奏
        return False

    async def start(self):
        self._ensure_phase_registration()
        logger.info(f"{self.log_prefix} 独立自主代理已停用，统一由增强主循环负责主动判断")
        self._pipeline_ready = True
        self._proactive_start_ts = time.time()
        self._proactive_startup_grace_until = self._proactive_start_ts + 60.0
        self._proactive_task: Optional[asyncio.Task] = None
        self._proactive_running: bool = True
        await super().start()
        # 父类先置 running=True，避免主动后台循环首轮条件检查直接退出
        try:
            self._proactive_task = self._spawn(
                self._proactive_background_loop(),
                name=f"heartfc_proactive_{self.stream_id}",
            )
            logger.info(f"{self.log_prefix} 🔔 主动感兴趣通道已启动（后台常驻）")
        except Exception as exc:
            logger.warning(f"{self.log_prefix} 主动通道启动失败，回退到主循环触发模式: {exc}")
            self._proactive_task = None
        logger.info(f"{self.log_prefix} 增强版控制器已接管，管线就绪")

    async def stop(self) -> None:
        self._pipeline_ready = False
        self._proactive_running = False
        if self._proactive_task and not self._proactive_task.done():
            self._proactive_task.cancel()
            try:
                await self._proactive_task
            except asyncio.CancelledError:
                pass
            logger.info(f"{self.log_prefix} 🔔 主动感兴趣通道已停止")
        self._teardown_voice_engine()
        await super().stop()

    def _ensure_phase_registration(self) -> None:
        if self._phase_registered:
            return
        try:
            from src.chat.heart_flow.state_machine import get_phase_coordinator
            from src.common.data_models.heartflow_models import FlowPhase

            coordinator = get_phase_coordinator()
            coordinator.register_channel(self.stream_id, initial=FlowPhase.ENGAGED)
            self._phase_registered = True
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 注册阶段协调器失败: {exc}")

    def _teardown_voice_engine(self) -> None:
        try:
            from src.chat.heart_flow.inner_voice import teardown_engine

            teardown_engine(self.stream_id)
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    async def _update_user_impression_after_reply(self, reply_text: str) -> None:
        """回复成功后更新用户印象

        让模型根据互动内容动态生成用户印象标签
        """
        try:
            from src.modules.social_value.social_affect_fuser import (
                get_social_affect_fuser,
            )

            fuser = get_social_affect_fuser()
            recent_interactions = reply_text[:200] if reply_text else ""
            await fuser.refresh_impression(
                user_id=self._last_user_id,
                channel_id=self.stream_id,
                interaction_digest=recent_interactions,
                topics=[],
                style="",
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 更新用户印象失败: {exc}")

    @property
    def _orch(self):
        """懒加载统一引擎编排器"""
        if self._engine_orch is None:
            from src.core.engine_orchestrator import get_engine_orchestrator

            self._engine_orch = get_engine_orchestrator(self.stream_id)
        return self._engine_orch

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

class ScenePlannerBridgeMixin:
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
        if gw_ctx.admin_force:
            self._last_reactive_plan = None
            logger.debug(f"{self.log_prefix} 👑 管理员极速通道: 跳过被动策略规划")
        elif not verdict.should_skip:
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

            self._inject_night_soul_state(extra_parts)
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

    def _inject_night_soul_state(self, extra_parts: list) -> None:
        """管理员快回也必须带上夜间/清晨身体状态，避免跳过独白后变成默认平静。"""
        try:
            from src.core.night_cycle_system import get_night_cycle

            summary = get_night_cycle(self.stream_id).night_behavior_summary()
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 夜间灵魂状态注入失败: {exc}")
            summary = {}
        if not isinstance(summary, dict) or not summary:
            return
        hour = int(summary.get("current_hour", 0) or 0) % 24
        time_label = str(summary.get("time_band_label", "") or "")
        phase_label = str(summary.get("phase_label", "") or "")
        body_tag = str(summary.get("body_state_tag", "") or "").strip()
        mood_hint = str(summary.get("monologue_mood_hint", "") or "").strip()
        drowsiness = float(summary.get("drowsiness_value", 0.0) or 0.0)
        pressure = float(summary.get("overnight_pressure", 0.0) or 0.0)
        half_asleep = float(summary.get("half_asleep_level", 0.0) or 0.0)
        sleep_debt = float(summary.get("sleep_debt", 0.0) or 0.0)
        sleep_window = bool(summary.get("is_sleep_window", False))
        pressure_window = bool(summary.get("is_pressure_window", False))
        dawn_like = str(summary.get("time_band", "") or "") == "dawn" or 5 <= hour < 8
        if not (sleep_window or pressure_window or dawn_like or half_asleep >= 0.2 or pressure >= 35.0 or drowsiness >= 35.0):
            return
        tone_bits = []
        if sleep_window:
            tone_bits.append("睡眠窗口")
        if pressure_window:
            tone_bits.append("熬夜压力窗口")
        if dawn_like:
            tone_bits.append("清晨刚醒")
        if body_tag:
            tone_bits.append(body_tag)
        extra_parts.append(
            "[夜间身体状态] "
            f"现在约{hour}点({time_label or phase_label})，"
            f"困意={drowsiness:.0f}，熬夜压力={pressure:.0f}，半醒={half_asleep:.2f}，睡眠债={sleep_debt:.2f}；"
            f"{' / '.join(tone_bits) if tone_bits else phase_label}。"
        )
        if dawn_like:
            extra_parts.append("[夜间语气约束] 回复要像清晨刚醒或熬夜后半醒的人：短、轻、带一点困倦，不要精神饱满地说“不困”。")
        elif sleep_window or half_asleep >= 0.35:
            extra_parts.append("[夜间语气约束] 回复必须带困倦、迟钝或被打断的底色；不要热情、不要长篇解释。")
        elif pressure >= 60.0:
            extra_parts.append("[夜间语气约束] 熬夜压力很高，回复要显得累、慢、克制。")
        if mood_hint:
            extra_parts.append(f"[夜间心境提示] {mood_hint}")

    @staticmethod
    def _soul_prompt_keywords() -> Tuple[str, ...]:
        return (
            "内心独白",
            "情感状态",
            "当前情感状态",
            "当前心理状态",
            "夜间身体状态",
            "夜间语气约束",
            "夜间心境提示",
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
        fallback_block = "\n".join(fallback_lines[:4]).strip()
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

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.chat.proactive.proactive_decider import (
    SignalBundle,
    ProactiveDecision,
    get_proactive_decider,
    classify_behavior_outcome,
    BehaviorConsequence,
)

logger = get_logger("主动整合中心")


@dataclass
class IntegratedState:
    """整合后的完整状态快照"""

    channel_id: str = ""
    timestamp: float = 0.0
    emotion_state: Dict[str, Any] = field(default_factory=dict)
    perception_verdict: Dict[str, Any] = field(default_factory=dict)
    perception_context: Dict[str, Any] = field(default_factory=dict)
    self_awareness_data: Dict[str, Any] = field(default_factory=dict)
    session_data: Dict[str, Any] = field(default_factory=dict)
    silence_data: Dict[str, Any] = field(default_factory=dict)
    intention_data: Dict[str, Any] = field(default_factory=dict)
    energy_data: Dict[str, Any] = field(default_factory=dict)
    metabolism_data: Dict[str, Any] = field(default_factory=dict)
    memory_data: Dict[str, Any] = field(default_factory=dict)
    signal_bundle: Optional[SignalBundle] = None
    verdict: Optional[ProactiveDecision] = None


class ProactiveIntegrationHub:
    """
    主动行为整合中心

    核心职责：
    1. 从所有分散模块收集状态数据
    2. 整合为统一的 SignalBundle
    3. 调用 proactive_decider 进行决策
    4. 将决策结果反馈回各模块
    """

    _instance: Optional["ProactiveIntegrationHub"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._emotion_core = None
        self._awareness_engine = None
        self._perception_generator = None
        self._self_awareness = None
        self._memoir_cabinet = None
        self._quiet_monitor = None
        self._intention_pool = None
        self._vitality_pool = None
        self._metabolism_engine = None
        self._decider = None
        self._state_cache: Dict[str, IntegratedState] = {}
        self._last_tick: Dict[str, float] = {}
        logger.info("[整合中心] 主动行为整合中心初始化完成")

    @staticmethod
    def _metric_or_default(value: Any, default: float) -> float:
        """仅在数值缺失时使用默认值，避免把合法 0 洗回中性。"""
        if value is None or value == "":
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _summarize_awareness_content(content: Any, fallback: str = "", limit: int = 160) -> str:
        """规整自我觉察正文，避免只留下空账本。"""
        text = " ".join(str(content or "").split()).strip()
        if not text:
            text = " ".join(str(fallback or "").split()).strip()
        if len(text) > limit:
            text = text[:limit].rstrip() + "..."
        return text

    def _lazy_load_modules(self):
        """延迟加载所有模块，避免循环导入"""
        if self._emotion_core is not None:
            return
        try:
            from src.chat.heart_flow.emotion_driven_core import get_emotion_driven_core

            self._emotion_core = get_emotion_driven_core()
        except Exception as e:
            logger.debug(f"情感驱动核心加载失败: {e}")
        try:
            from src.chat.proactive.perception_engine import get_awareness_engine

            self._awareness_engine = get_awareness_engine()
        except Exception as e:
            logger.debug(f"感知觉察引擎加载失败: {e}")
        try:
            from src.modules.modcore.perception.perception_generator import get_perception_generator

            self._perception_generator = get_perception_generator()
        except Exception as e:
            logger.debug(f"感知生成器加载失败: {e}")
        try:
            from src.modules.recall.self_awareness import get_self_awareness

            self._self_awareness = get_self_awareness()
        except Exception as e:
            logger.debug(f"自我意识模块加载失败: {e}")
        try:
            from src.chat.proactive.session_tracker import get_memoir_cabinet

            self._memoir_cabinet = get_memoir_cabinet()
        except Exception as e:
            logger.debug(f"会话追踪器加载失败: {e}")
        try:
            from src.chat.proactive.silence_watcher import get_quiet_monitor

            self._quiet_monitor = get_quiet_monitor()
        except Exception as e:
            logger.debug(f"静默期巡查器加载失败: {e}")
        try:
            from src.chat.proactive.intention_pool import get_intention_pool

            self._intention_pool = get_intention_pool()
        except Exception as e:
            logger.debug(f"意图池管理器加载失败: {e}")
        try:
            from src.chat.heart_flow.energy_manager import EnergyChainDimension

            self._vitality_pool = EnergyChainDimension.get_instance()
        except Exception as e:
            logger.debug(f"能量管理器加载失败: {e}")
        try:
            from src.core.metabolism_engine import get_metabolism_engine

            self._metabolism_engine = get_metabolism_engine()
        except Exception as e:
            logger.debug(f"代谢引擎加载失败: {e}")
        self._decider = get_proactive_decider()

    async def collect_state(
        self,
        channel_id: str,
        raw_messages: Optional[List[Dict]] = None,
        self_state: Optional[Dict] = None,
        relation_state: Optional[Dict] = None,
    ) -> IntegratedState:
        """
        收集所有模块的状态数据

        参数:
            channel_id: 频道ID
            raw_messages: 最近消息列表
            self_state: 自身状态（mood/trauma等）
            relation_state: 关系数据（favor/annoyance等）

        返回:
            IntegratedState: 整合后的完整状态
        """
        self._lazy_load_modules()
        state = IntegratedState(channel_id=channel_id, timestamp=time.time())
        raw_messages = raw_messages or []
        self_state = self_state or {}
        relation_state = relation_state or {}
        state.emotion_state = self._collect_emotion_state(channel_id)
        state.perception_verdict = await self._collect_perception_verdict(
            channel_id, raw_messages, self_state, relation_state
        )
        state.perception_context = self._collect_perception_context(channel_id, self_state)
        state.self_awareness_data = self._collect_self_awareness(channel_id)
        state.session_data = self._collect_session_data(channel_id)
        state.silence_data = self._collect_silence_data(channel_id)
        state.intention_data = self._collect_intention_data(channel_id)
        state.energy_data = self._collect_energy_data(channel_id)
        state.metabolism_data = self._collect_metabolism_data(channel_id)
        state.memory_data = self._collect_memory_data(channel_id)
        state.signal_bundle = self._build_signal_bundle(state)
        self._state_cache[channel_id] = state
        return state

    def _collect_emotion_state(self, channel_id: str) -> Dict[str, Any]:
        """从情感驱动核心收集状态"""
        if self._emotion_core is None:
            return {}
        try:
            silence_sec = 0.0
            if self._quiet_monitor:
                silence_sec = self._quiet_monitor.measure_silence_sec(channel_id)
            unanswered = 0
            if self._memoir_cabinet:
                memoir = self._memoir_cabinet.lookup_by_channel(channel_id)
                if memoir:
                    unanswered = getattr(memoir, "consecutive_timeouts", 0)
            emotion_state = self._emotion_core.tick(channel_id, silence_sec, unanswered)
            return {
                "boredom": emotion_state.boredom,
                "loneliness": emotion_state.loneliness,
                "environmental_fatigue": getattr(emotion_state, "environmental_fatigue", 0.0),
                "social_desire": emotion_state.social_desire,
                "mood": emotion_state.mood,
                "energy": emotion_state.energy,
                "curiosity": emotion_state.curiosity,
                "proactive_willingness": emotion_state.proactive_willingness,
                "feeling_description": emotion_state.feeling_description,
                "silence_duration": emotion_state.silence_duration,
                "unanswered_count": emotion_state.unanswered_count,
                "monitoring": emotion_state.monitoring,
            }
        except Exception as e:
            logger.debug(f"情感状态收集失败: {e}")
            return {}

    async def _collect_perception_verdict(
        self,
        channel_id: str,
        raw_messages: List[Dict],
        self_state: Dict,
        relation_state: Dict,
    ) -> Dict[str, Any]:
        """从感知觉察引擎收集判定结果"""
        if self._awareness_engine is None:
            return {}
        try:
            mention_bot = any(
                msg.get("mention_bot", False) or msg.get("is_reply_to_bot", False) for msg in raw_messages[-5:]
            )
            is_reply_to_bot = any(msg.get("is_reply_to_bot", False) for msg in raw_messages[-3:])
            verdict = await self._awareness_engine.gather_awareness(
                channel_id=channel_id,
                raw_messages=raw_messages,
                self_state=self_state,
                relation_state=relation_state,
                mention_bot=mention_bot,
                is_reply_to_bot=is_reply_to_bot,
            )
            return verdict.to_dict()
        except Exception as e:
            logger.debug(f"感知判定收集失败: {e}")
            return {}

    def _collect_perception_context(self, channel_id: str, self_state: Dict) -> Dict[str, Any]:
        """从感知生成器收集感知上下文"""
        if self._perception_generator is None:
            return {}
        try:
            context = self._perception_generator.generate_from_state(self_state, context_seed=channel_id)
            return context.to_dict()
        except Exception as e:
            logger.debug(f"感知上下文收集失败: {e}")
            return {}

    def _collect_self_awareness(self, channel_id: str) -> Dict[str, Any]:
        """从自我意识模块收集数据"""
        if self._self_awareness is None:
            return {}
        try:
            recent_messages = self._self_awareness.get_recent_messages(channel_id, limit=5)
            recent_actions = self._self_awareness.get_recent_actions(channel_id, limit=5)
            recent_events = self._self_awareness.get_recent_user_events(channel_id, limit=5)
            stats = self._self_awareness.get_interaction_stats(channel_id)
            return {
                "recent_messages_count": len(recent_messages),
                "recent_actions_count": len(recent_actions),
                "recent_events_count": len(recent_events),
                "recent_messages": [
                    {
                        "msg_id": str(getattr(item, "msg_id", "") or ""),
                        "content": self._summarize_awareness_content(getattr(item, "content", "")),
                        "sent_at": float(getattr(item, "sent_at", 0.0) or 0.0),
                    }
                    for item in recent_messages
                ],
                "recent_actions": [
                    {
                        "action_type": str(getattr(item, "action_type", "") or ""),
                        "target": str(getattr(item, "target", "") or ""),
                        "content": self._summarize_awareness_content(getattr(item, "content", "")),
                        "result": str(getattr(item, "result", "") or ""),
                        "timestamp": float(getattr(item, "timestamp", 0.0) or 0.0),
                    }
                    for item in recent_actions
                ],
                "recent_events": [
                    {
                        "event_type": str(getattr(item, "event_type", "") or ""),
                        "user_id": str(getattr(item, "user_id", "") or ""),
                        "content": self._summarize_awareness_content(getattr(item, "content", "")),
                        "timestamp": float(getattr(item, "timestamp", 0.0) or 0.0),
                    }
                    for item in recent_events
                ],
                "stats": stats,
            }
        except Exception as e:
            logger.debug(f"自我意识数据收集失败: {e}")
            return {}

    def _collect_session_data(self, channel_id: str) -> Dict[str, Any]:
        """从会话追踪器收集数据"""
        if self._memoir_cabinet is None:
            return {}
        try:
            memoir = self._memoir_cabinet.lookup_by_channel(channel_id)
            if memoir is None:
                return {}
            return {
                "phase": str(memoir.phase),
                "exchange_tally": memoir.exchange_tally,
                "consecutive_timeouts": memoir.consecutive_timeouts,
                "last_user_spoke_at": memoir.last_user_spoke_at,
                "is_waiting": memoir.is_waiting(),
                "born_at": memoir.born_at,
            }
        except Exception as e:
            logger.debug(f"会话数据收集失败: {e}")
            return {}

    def _collect_silence_data(self, channel_id: str) -> Dict[str, Any]:
        """从静默期巡查器收集数据"""
        if self._quiet_monitor is None:
            return {}
        try:
            silence_sec = self._quiet_monitor.measure_silence_sec(channel_id)
            return {
                "silence_seconds": silence_sec,
                "silence_minutes": silence_sec / 60.0,
            }
        except Exception as e:
            logger.debug(f"静默数据收集失败: {e}")
            return {}

    def _collect_intention_data(self, channel_id: str) -> Dict[str, Any]:
        """从意图池管理器收集数据"""
        if self._intention_pool is None:
            return {}
        try:
            intentions = self._intention_pool.get_active_intentions(channel_id)
            drive_score = self._intention_pool.build_decision_signal(channel_id)
            top = intentions[0] if intentions else None
            return {
                "active_count": len(intentions),
                "top_intention": top.to_dict() if top else None,
                "drive_score": drive_score,
            }
        except Exception as e:
            logger.debug(f"意图数据收集失败: {e}")
            return {}

    def _collect_energy_data(self, channel_id: str) -> Dict[str, Any]:
        """从能量管理器收集数据"""
        if self._vitality_pool is None:
            return {}
        try:
            state = self._vitality_pool._ensure_channel(channel_id)
            return {
                "chat_pool": state.chat_pool,
                "thinking_value": state.thinking_value,
                "energy_ratio": state.combined_ratio(),
                "is_resting": state.is_exhausted(),
                "social_value": self._metric_or_default(getattr(state, "social_value", None), 0.0),
                "activity_level": self._metric_or_default(getattr(state, "activity_level", None), 50.0),
            }
        except Exception as e:
            logger.debug(f"能量数据收集失败: {e}")
            return {}

    def _collect_metabolism_data(self, channel_id: str) -> Dict[str, Any]:
        """从代谢引擎收集数据"""
        if self._metabolism_engine is None:
            return {}
        try:
            engine_state = self._metabolism_engine.get_state(channel_id)
            return {
                "chat_fuel": engine_state.chat_fuel,
                "thinking_fuel": engine_state.thinking_fuel,
                "activity_gauge": engine_state.activity_gauge,
                "chat_fuel_ratio": engine_state.chat_fuel_ratio(),
                "thinking_fuel_ratio": engine_state.thinking_fuel_ratio(),
            }
        except Exception as e:
            logger.debug(f"代谢数据收集失败: {e}")
            return {}

    def _collect_memory_data(self, channel_id: str) -> Dict[str, Any]:
        """收集短期/长期/回忆录/知识/过载治理状态。"""
        memory_data: Dict[str, Any] = {}
        try:
            from src.core.memory_governance_engine import get_memory_governance_engine

            memory_data["governance"] = get_memory_governance_engine(channel_id).get_snapshot().to_dict()
        except Exception as e:
            logger.debug(f"记忆治理数据收集失败: {e}")
        try:
            from src.memory_system.memory_consolidator import acquire_ephemeral_bank

            bank = acquire_ephemeral_bank()
            channel_buffers = getattr(bank, "_channel_buffers", {}) or {}
            memory_data["short_term_count"] = len(channel_buffers.get(channel_id, []) or [])
        except Exception as e:
            logger.debug(f"短期记忆数据收集失败: {e}")
        try:
            from src.memory_system.memory_core import get_memory_core

            hub = get_memory_core()
            memory_data["long_term_stats"] = hub.compile_statistics(channel_id)
            memory_data["health"] = hub.generate_health_assessment(channel_id)
            memory_data["overload"] = hub.retrieve_overload_status(channel_id)
        except Exception as e:
            logger.debug(f"长期记忆数据收集失败: {e}")
        try:
            from src.core.memory_reactivation_gate import get_reactivation_gate

            memory_data["reactivation"] = get_reactivation_gate(channel_id).recent_activation_summary()
        except Exception as e:
            logger.debug(f"记忆激活数据收集失败: {e}")
        try:
            from src.core.unified_planner import get_unified_planner

            memory_data["planner_context"] = get_unified_planner().get_context_stats(channel_id)
        except Exception as e:
            logger.debug(f"规划器上下文统计收集失败: {e}")
        try:
            from src.modules.modcore.social_cognition.knowledge_graph import get_knowledge_graph_manager

            kg = get_knowledge_graph_manager(channel_id)
            memory_data["knowledge"] = {"entry_count": int(getattr(kg, "entry_count", 0) or 0)}
        except Exception as e:
            logger.debug(f"知识图谱统计收集失败: {e}")
        return memory_data

    def _build_signal_bundle(self, state: IntegratedState) -> SignalBundle:
        """将整合状态转换为 SignalBundle"""
        emotion = state.emotion_state
        perception = state.perception_verdict
        silence = state.silence_data
        intention = state.intention_data
        energy = state.energy_data
        metabolism = state.metabolism_data
        memory_data = state.memory_data
        emotional_readiness = emotion.get("proactive_willingness", 0.0)
        if perception:
            emotional_readiness = max(emotional_readiness, perception.get("engagement_pull", 0.0))
        vitality_ratio = 1.0
        if energy:
            vitality_ratio = energy.get("energy_ratio", 1.0)
        elif metabolism:
            vitality_ratio = metabolism.get("thinking_fuel_ratio", 1.0)
        intention_drive = 0.0
        if intention:
            intention_drive = intention.get("drive_score", 0.0)
        silence_seconds = silence.get("silence_seconds", 0.0)
        bundle = SignalBundle(
            emotional_readiness=emotional_readiness,
            boredom=emotion.get("boredom", 0.0),
            loneliness=emotion.get("loneliness", 0.0),
            vitality_ratio=vitality_ratio,
            content_novelty=self._estimate_content_novelty(state),
            social_standing=self._extract_social_standing(state),
            inner_voice_desire=int(round(emotion.get("social_desire", 0.5) * 10)),
            intention_drive=intention_drive,
            silence_seconds=silence_seconds,
            channel_id=state.channel_id,
        )
        governance = memory_data.get("governance", {}) if isinstance(memory_data.get("governance"), dict) else {}
        overload = memory_data.get("overload", {}) if isinstance(memory_data.get("overload"), dict) else {}
        planner_context = memory_data.get("planner_context", {}) if isinstance(memory_data.get("planner_context"), dict) else {}
        session = state.session_data

        utilization = self._metric_or_default(governance.get("utilization"), 0.0)
        if utilization >= 0.85:
            bundle.content_novelty = max(0.12, bundle.content_novelty - 0.18)
            bundle.intention_drive = max(0.0, bundle.intention_drive - 0.08)
        elif utilization >= 0.65:
            bundle.content_novelty = max(0.18, bundle.content_novelty - 0.08)

        overload_ratio = self._metric_or_default(overload.get("load_ratio"), 0.0)
        amnesia_pressure = self._metric_or_default(overload.get("amnesia_pressure"), 0.0)
        if overload.get("emergency_needed"):
            bundle.vitality_ratio = min(bundle.vitality_ratio, 0.18)
            bundle.inner_voice_desire = min(bundle.inner_voice_desire, 3)
        elif overload_ratio >= 0.8 or amnesia_pressure >= 0.6:
            bundle.vitality_ratio = min(bundle.vitality_ratio, 0.35)
            bundle.inner_voice_desire = min(bundle.inner_voice_desire, 4)

        if session.get("is_waiting") and int(session.get("consecutive_timeouts", 0) or 0) <= 1:
            bundle.intention_drive = min(1.0, bundle.intention_drive + 0.12)
        elif int(session.get("consecutive_timeouts", 0) or 0) >= 3:
            bundle.intention_drive = max(0.0, bundle.intention_drive - 0.10)

        short_term_count = int(planner_context.get("short_term_count", 0) or 0)
        long_term_count = int(planner_context.get("long_term_count", 0) or 0)
        if short_term_count > 0 and long_term_count > 0:
            bundle.content_novelty = min(1.0, bundle.content_novelty + 0.04)
        return bundle

    def _extract_social_standing(self, state: IntegratedState) -> float:
        """从能量链数据中提取社交声望值（范围 -50~100 → 归一化传递给主动决策器）"""
        energy = state.energy_data
        if energy:
            raw = self._metric_or_default(energy.get("social_value"), 0.0)
            return raw
        return 0.0

    def _estimate_content_novelty(self, state: IntegratedState) -> float:
        """基于沉默时长和会话状态估算内容新鲜度"""
        silence = state.silence_data
        session = state.session_data
        novelty = 0.5
        silence_sec = float(silence.get("silence_seconds", 0.0) or 0.0)
        if silence_sec > 300:
            novelty += 0.2
        elif silence_sec > 120:
            novelty += 0.1
        elif silence_sec < 30:
            novelty -= 0.15
        if session:
            exchange = int(session.get("exchange_tally", 0) or 0)
            if exchange < 3:
                novelty += 0.1
            elif exchange > 15:
                novelty -= 0.1
        return max(0.0, min(1.0, novelty))

    async def evaluate_proactive(
        self,
        channel_id: str,
        raw_messages: Optional[List[Dict]] = None,
        self_state: Optional[Dict] = None,
        relation_state: Optional[Dict] = None,
    ) -> Tuple[ProactiveDecision, IntegratedState]:
        """
        评估是否应该执行主动行为

        返回:
            (ProactiveDecision, IntegratedState): 决策结果和完整状态
        """
        self._lazy_load_modules()
        state = await self.collect_state(channel_id, raw_messages, self_state, relation_state)
        if state.signal_bundle is None:
            state.verdict = ProactiveDecision(
                should_proceed=False,
                rationale="信号包构建失败",
            )
            return state.verdict, state
        verdict = self._decider.evaluate(state.signal_bundle)
        state.verdict = verdict
        if verdict.should_proceed:
            logger.info(
                f"[整合中心] {channel_id[:8]} 决定主动行为 | "
                f"融合分={verdict.fused_score:.3f} | "
                f"理由={verdict.rationale}"
            )
        return verdict, state

    def on_bot_message_sent(
        self,
        channel_id: str,
        user_id: str = "",
        was_proactive: bool = False,
        content: str = "",
    ) -> None:
        """机器人发送消息后的回调（主动/被动通用）"""
        self._lazy_load_modules()
        if self._emotion_core:
            self._emotion_core.on_bot_message(channel_id)
        if self._quiet_monitor:
            self._quiet_monitor.note_activity(channel_id, is_bot=True)
        # 仅主动发言才登记到主动决策账本（被动回复不应产生挂起事件）
        # 注意：显式的 register_proactive_fire 已在 heartFC _on_bot_sent 中处理
        if self._self_awareness:
            _action_type = "proactive_message" if was_proactive else "reply"
            _content = self._summarize_awareness_content(
                content,
                fallback=f"{_action_type} target={user_id}" if user_id else _action_type,
            )
            self._self_awareness.record_action(
                stream_id=channel_id,
                action_type=_action_type,
                target=user_id,
                content=_content,
                result="sent",
                extra={"target_user_id": user_id, "was_proactive": bool(was_proactive)},
            )
        logger.debug(f"[整合中心] {channel_id[:8]} 机器人发送消息后状态更新")

    def on_user_message_received(
        self,
        channel_id: str,
        user_id: str = "",
        content: str = "",
    ) -> None:
        """用户消息到达后的回调"""
        self._lazy_load_modules()
        if self._emotion_core:
            self._emotion_core.on_user_message(channel_id, user_id)
        if self._quiet_monitor:
            self._quiet_monitor.note_activity(channel_id, speaker_id=user_id, is_bot=False)
        if self._memoir_cabinet:
            memoir = self._memoir_cabinet.lookup_sync(user_id) or self._memoir_cabinet.lookup_by_channel(channel_id)
            if memoir:
                memoir.consecutive_timeouts = 0
        if self._self_awareness:
            _content = self._summarize_awareness_content(
                content,
                fallback=f"user_message from={user_id}" if user_id else "user_message",
            )
            self._self_awareness.record_user_event(
                stream_id=channel_id,
                user_id=user_id,
                event_type="user_message",
                content=_content,
                extra={"user_id": user_id},
            )
        logger.debug(f"[整合中心] {channel_id[:8]} 用户消息到达后状态更新")

    def on_proactive_outcome(
        self,
        channel_id: str,
        has_reply: bool,
        reply_sentiment: float = 0.0,
        reply_relevance: float = 0.5,
        was_interrupted: bool = False,
        resolve_ledger: bool = True,
        target_user: str = "",
    ) -> BehaviorConsequence:
        """
        主动行为结果反馈

        参数:
            channel_id: 频道ID
            has_reply: 用户是否回复
            reply_sentiment: 回复情感 (-1 到 +1)
            reply_relevance: 回复相关性 (0 到 1)
            was_interrupted: 是否被打断

        返回:
            BehaviorConsequence: 结构化行为后果
        """
        self._lazy_load_modules()
        consequence = classify_behavior_outcome(
            has_reply=has_reply,
            reply_sentiment=reply_sentiment,
            reply_relevance=reply_relevance,
            was_interrupted=was_interrupted,
        )
        if self._decider and resolve_ledger:
            self._decider.ledger.resolve_pending_with_consequence(
                channel_id,
                consequence,
                target_user=target_user,
            )
        if self._emotion_core:
            try:
                from src.core.emotion_feedback_loop import get_emotion_feedback_loop, FeedbackSource

                _efl = get_emotion_feedback_loop()
                _mood_d = 0.0
                if consequence.emotion_feedback == "satisfied":
                    _mood_d = 0.05
                elif consequence.emotion_feedback == "disappointed":
                    _mood_d = -0.05
                elif consequence.emotion_feedback == "hurt":
                    _mood_d = -0.15
                if abs(_mood_d) > 1e-6:
                    _efl.submit_delta(
                        channel_id,
                        FeedbackSource.PROACTIVE_OUTCOME,
                        mood=_mood_d,
                        confidence=0.8,
                    )
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
        logger.info(
            f"[整合中心] {channel_id[:8]} 主动行为结果 | "
            f"结果={consequence.outcome_tag} | "
            f"关系变化={consequence.relationship_delta:.2f} | "
            f"情绪反馈={consequence.emotion_feedback}"
        )
        return consequence

    def get_state_snapshot(self, channel_id: str) -> Dict[str, Any]:
        """获取指定频道的完整状态快照"""
        state = self._state_cache.get(channel_id)
        if state is None:
            return {"error": "状态未缓存，请先调用 collect_state"}
        return {
            "channel_id": state.channel_id,
            "timestamp": state.timestamp,
            "emotion_state": state.emotion_state,
            "perception_verdict": state.perception_verdict,
            "perception_context": state.perception_context,
            "self_awareness_data": state.self_awareness_data,
            "session_data": state.session_data,
            "silence_data": state.silence_data,
            "intention_data": state.intention_data,
            "energy_data": state.energy_data,
            "metabolism_data": state.metabolism_data,
            "memory_data": state.memory_data,
            "signal_bundle": state.signal_bundle.__dict__ if state.signal_bundle else None,
            "verdict": state.verdict.__dict__ if state.verdict else None,
        }

    def get_all_cached_channels(self) -> List[str]:
        """获取所有已缓存的频道ID"""
        return list(self._state_cache.keys())

    def clear_cache(self, channel_id: Optional[str] = None) -> None:
        """清除缓存"""
        if channel_id:
            self._state_cache.pop(channel_id, None)
        else:
            self._state_cache.clear()


_integration_hub: Optional[ProactiveIntegrationHub] = None


def get_proactive_integration_hub() -> ProactiveIntegrationHub:
    """获取主动行为整合中心单例"""
    global _integration_hub
    if _integration_hub is None:
        _integration_hub = ProactiveIntegrationHub()
    return _integration_hub

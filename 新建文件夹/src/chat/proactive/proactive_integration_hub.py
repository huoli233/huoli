import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.chat.proactive.proactive_arbiter import (
    SignalBundle,
    ArbiterVerdict,
    get_proactive_arbiter,
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
    signal_bundle: Optional[SignalBundle] = None
    verdict: Optional[ArbiterVerdict] = None


class ProactiveIntegrationHub:
    """
    主动行为整合中心
    
    核心职责：
    1. 从所有分散模块收集状态数据
    2. 整合为统一的 SignalBundle
    3. 调用 proactive_arbiter 进行决策
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
        self._arbiter = None
        self._state_cache: Dict[str, IntegratedState] = {}
        self._last_tick: Dict[str, float] = {}
        logger.info("[整合中心] 主动行为整合中心初始化完成")
    
    def _lazy_load_modules(self):
        """延迟加载所有模块，避免循环导入"""
        if self._emotion_core is not None:
            return
        try:
            from src.chat.heart_flow.emotion_stream import EmotionAxisDimension
            self._emotion_core = EmotionAxisDimension.get_instance()
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
            from src.chat.heart_flow.energy_manager import get_vitality_pool
            self._vitality_pool = get_vitality_pool()
        except Exception as e:
            logger.debug(f"能量管理器加载失败: {e}")
        try:
            from src.core.metabolism_engine import get_metabolism_engine
            self._metabolism_engine = get_metabolism_engine()
        except Exception as e:
            logger.debug(f"代谢引擎加载失败: {e}")
        self._arbiter = get_proactive_arbiter()
    
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
                memoir = self._memoir_cabinet.lookup_sync(channel_id)
                if memoir:
                    unanswered = getattr(memoir, 'consecutive_timeouts', 0)
            emotion_state = self._emotion_core.tick(channel_id, silence_sec, unanswered)
            return {
                "boredom": emotion_state.boredom,
                "loneliness": emotion_state.loneliness,
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
                msg.get("mention_bot", False) or msg.get("is_reply_to_bot", False)
                for msg in raw_messages[-5:]
            )
            is_reply_to_bot = any(
                msg.get("is_reply_to_bot", False)
                for msg in raw_messages[-3:]
            )
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
    
    def _collect_perception_context(
        self, channel_id: str, self_state: Dict
    ) -> Dict[str, Any]:
        """从感知生成器收集感知上下文"""
        if self._perception_generator is None:
            return {}
        try:
            context = self._perception_generator.generate_from_state(
                self_state, context_seed=channel_id
            )
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
            stats = self._self_awareness.get_interaction_stats(channel_id)
            return {
                "recent_messages_count": len(recent_messages),
                "recent_actions_count": len(recent_actions),
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
            memoir = self._memoir_cabinet.lookup_sync(channel_id)
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
            top_intention = self._intention_pool.get_top_intention(channel_id)
            drive_score = self._intention_pool.calculate_drive_score(channel_id)
            return {
                "active_count": len(intentions),
                "top_intention": top_intention.to_dict() if top_intention else None,
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
            pool = self._vitality_pool.get_pool(channel_id)
            return {
                "chat_pool": pool.chat_pool,
                "thinking_value": pool.thinking_value,
                "energy_ratio": pool.energy_ratio(),
                "is_resting": pool.is_resting,
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
    
    def _build_signal_bundle(self, state: IntegratedState) -> SignalBundle:
        """将整合状态转换为 SignalBundle"""
        emotion = state.emotion_state
        perception = state.perception_verdict
        silence = state.silence_data
        intention = state.intention_data
        energy = state.energy_data
        metabolism = state.metabolism_data
        emotional_readiness = emotion.get("proactive_willingness", 0.0)
        if perception:
            emotional_readiness = max(
                emotional_readiness,
                perception.get("engagement_pull", 0.0)
            )
        vitality_ratio = 1.0
        if energy:
            vitality_ratio = energy.get("energy_ratio", 1.0)
        elif metabolism:
            vitality_ratio = metabolism.get("thinking_fuel_ratio", 1.0)
        intention_drive = 0.0
        if intention:
            intention_drive = intention.get("drive_score", 0.0)
        silence_seconds = silence.get("silence_seconds", 0.0)
        return SignalBundle(
            emotional_readiness=emotional_readiness,
            boredom=emotion.get("boredom", 0.0),
            loneliness=emotion.get("loneliness", 0.0),
            vitality_ratio=vitality_ratio,
            content_novelty=0.5,
            social_standing=0.0,
            inner_voice_desire=int(round(emotion.get("social_desire", 0.5) * 10)),
            intention_drive=intention_drive,
            silence_seconds=silence_seconds,
            channel_id=state.channel_id,
        )
    
    async def evaluate_proactive(
        self,
        channel_id: str,
        raw_messages: Optional[List[Dict]] = None,
        self_state: Optional[Dict] = None,
        relation_state: Optional[Dict] = None,
    ) -> Tuple[ArbiterVerdict, IntegratedState]:
        """
        评估是否应该执行主动行为
        
        返回:
            (ArbiterVerdict, IntegratedState): 决策结果和完整状态
        """
        self._lazy_load_modules()
        state = await self.collect_state(
            channel_id, raw_messages, self_state, relation_state
        )
        if state.signal_bundle is None:
            state.verdict = ArbiterVerdict(
                should_proceed=False,
                rationale="信号包构建失败",
            )
            return state.verdict, state
        verdict = self._arbiter.evaluate(state.signal_bundle)
        state.verdict = verdict
        if verdict.should_proceed:
            logger.info(
                f"[整合中心] {channel_id[:8]} 决定主动行为 | "
                f"融合分={verdict.fused_score:.3f} | "
                f"理由={verdict.rationale}"
            )
        return verdict, state
    
    def on_bot_message_sent(self, channel_id: str, user_id: str = "") -> None:
        """机器人发送消息后的回调"""
        self._lazy_load_modules()
        if self._emotion_core:
            self._emotion_core.on_bot_message(channel_id)
        if self._quiet_monitor:
            self._quiet_monitor.note_activity(channel_id, is_bot=True)
        if self._arbiter:
            self._arbiter.ledger.register_proactive_fire(channel_id)
        if self._self_awareness:
            self._self_awareness.record_action(
                stream_id=channel_id,
                action_type="proactive_message",
                target=user_id,
                content="",
            )
        logger.debug(f"[整合中心] {channel_id[:8]} 机器人发送消息后状态更新")
    
    def on_user_message_received(
        self, channel_id: str, user_id: str = ""
    ) -> None:
        """用户消息到达后的回调"""
        self._lazy_load_modules()
        if self._emotion_core:
            self._emotion_core.on_user_message(channel_id, user_id)
        if self._quiet_monitor:
            self._quiet_monitor.note_activity(channel_id, speaker_id=user_id, is_bot=False)
        if self._memoir_cabinet:
            memoir = self._memoir_cabinet.lookup_sync(channel_id)
            if memoir:
                memoir.consecutive_timeouts = 0
        if self._self_awareness:
            self._self_awareness.record_user_event(
                stream_id=channel_id,
                user_id=user_id,
                event_type="user_message",
                content="",
            )
        logger.debug(f"[整合中心] {channel_id[:8]} 用户消息到达后状态更新")
    
    def on_proactive_outcome(
        self,
        channel_id: str,
        has_reply: bool,
        reply_sentiment: float = 0.0,
        reply_relevance: float = 0.5,
        was_interrupted: bool = False,
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
        if self._arbiter:
            if has_reply:
                self._arbiter.ledger.resolve_pending(
                    channel_id, outcome="replied", reward=consequence.reward_delta
                )
            else:
                self._arbiter.ledger.resolve_pending(
                    channel_id, outcome="ignored", reward=consequence.reward_delta
                )
        if self._emotion_core:
            state = self._emotion_core._get_state(channel_id)
            if consequence.emotion_feedback == "satisfied":
                state.mood = min(1.0, state.mood + 0.05)
            elif consequence.emotion_feedback == "disappointed":
                state.mood = max(0.0, state.mood - 0.05)
            elif consequence.emotion_feedback == "hurt":
                state.mood = max(0.0, state.mood - 0.15)
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

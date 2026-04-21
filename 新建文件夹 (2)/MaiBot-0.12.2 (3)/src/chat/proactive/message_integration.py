import time
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("message_integration")


@dataclass
class ProactiveConfig:
    enabled: bool = True
    think_interval: float = 30.0
    silence_threshold: float = 60.0
    engagement_threshold: float = 0.5
    max_think_per_hour: int = 10


@dataclass
class ChannelThinkState:
    channel_id: str
    last_think_time: float = 0.0
    think_count: int = 0
    is_thinking: bool = False
    last_message_time: float = 0.0
    silence_duration: float = 0.0
    engagement_score: float = 0.0

    def update_last_message(self):
        self.last_message_time = time.time()
        self.silence_duration = 0.0

    def update_silence_duration(self):
        self.silence_duration = time.time() - self.last_message_time

    def start_thinking(self):
        self.is_thinking = True
        self.last_think_time = time.time()
        self.think_count += 1

    def stop_thinking(self):
        self.is_thinking = False


class ProactiveIntegration:
    def __init__(self, context: Any = None):
        self._context = context
        self._channel_states: Dict[str, ChannelThinkState] = {}
        self._initialized = False
        self._enabled = True
        self._unified_scheduler = None

    def initialize(self):
        if self._initialized:
            return
        try:
            self._init_unified_scheduler()
            self._initialized = True
            logger.info("主动思考集成器初始化完成")
        except Exception as e:
            logger.error(f"主动思考集成器初始化失败: {e}")

    def _init_unified_scheduler(self):
        try:
            from src.chat.proactive.schedule_strategy import ScheduleConfig, init_unified_scheduler
            config = ScheduleConfig(
                private_min_interval_minutes=30,
                private_max_interval_minutes=120,
                group_silence_min_hours=1.0,
                group_silence_max_hours=2.0,
                quiet_hours="23-7",
                max_unanswered_count=1,
                daily_max_private=9999,
                daily_max_group=9999,
            )
            self._unified_scheduler = init_unified_scheduler(config, self._on_proactive_trigger)
            logger.info("主动回复调度器已初始化（群聊沉默1-2小时触发）")
        except Exception as e:
            logger.warning(f"主动回复调度器初始化失败: {e}")
            self._unified_scheduler = None

    async def _on_proactive_trigger(self, session_id: str):
        try:
            logger.info(f"[主动回复] 触发: {session_id[:16]}...")
            from src.chat.proactive.think_scheduler import get_think_scheduler, ThinkCategory, ThinkPriority
            scheduler = get_think_scheduler()
            if not scheduler._is_active:
                logger.info(f"[主动回复] 调度器未激活，尝试启动...")
                try:
                    await scheduler.start()
                except Exception as e:
                    logger.warning(f"[主动回复] 调度器启动失败: {e}")
                    return
            task_id = await scheduler.schedule_task(
                category=ThinkCategory.SPONTANEOUS_INIT,
                user_id="", stream_id=session_id,
                priority=ThinkPriority.NORMAL,
                params={
                    "silence_duration": "超过1小时",
                    "trigger_reason": "群聊沉默检测触发",
                },
                delay_seconds=0, allow_duplicate=False,
            )
            if task_id:
                logger.info(f"[主动回复] 任务已调度: {session_id[:16]}... -> task_id={task_id[:8]}...")
            else:
                logger.debug(f"[主动回复] 任务被去重或调度失败: {session_id[:16]}...")
        except Exception as e:
            logger.error(f"主动回复触发失败: {e}")

    def get_channel_state(self, channel_id: str) -> ChannelThinkState:
        if channel_id not in self._channel_states:
            self._channel_states[channel_id] = ChannelThinkState(channel_id=channel_id)
        return self._channel_states[channel_id]

    async def on_message_received(self, channel_id: str, user_id: str, message_text: str,
                                 is_group: bool = False, is_targeted: bool = False):
        try:
            if not self._enabled:
                return
            state = self.get_channel_state(channel_id)
            state.update_last_message()
            if self._unified_scheduler:
                await self._unified_scheduler.on_message_received(
                    channel_id, is_user=True, is_wake_up=is_targeted
                )
            logger.debug(f"[主动思考] 处理消息: channel={channel_id[:8]}, user={user_id[:8]}, is_group={is_group}, is_targeted={is_targeted}")
        except Exception as e:
            logger.debug(f"处理消息失败: {e}")

    async def on_reply_sent(self, channel_id: str, content: str, event: Any = None, is_proactive: bool = False):
        try:
            state = self.get_channel_state(channel_id)
            state.stop_thinking()
            if self._unified_scheduler:
                await self._unified_scheduler.on_reply_sent(channel_id, is_proactive=is_proactive)
            logger.debug(f"[主动思考] 回复已发送: channel={channel_id[:8]}, is_proactive={is_proactive}")
        except Exception as e:
            logger.debug(f"处理回复发送失败: {e}")

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled


class ChatterIntegration:
    def __init__(self, context: Any = None):
        self._context = context
        self._initialized = False
        self._enabled = True

    def initialize(self):
        if self._initialized:
            return
        try:
            self._initialized = True
            logger.info("聊天处理器集成初始化完成")
        except Exception as e:
            logger.error(f"聊天处理器集成初始化失败: {e}")

    async def process_message(self, channel_id: str, message_text: str, user_id: str = None):
        try:
            if not self._enabled:
                return None
            logger.debug(f"[聊天集成] 处理消息: channel={channel_id[:8]}")
            return None
        except Exception as e:
            logger.debug(f"聊天消息处理失败: {e}")
            return None

    async def calculate_engagement(
        self, channel_id: str, user_id: str, message_text: str, is_targeted: bool = False
    ) -> Dict:
        try:
            if not self._enabled:
                return {"should_reply": is_targeted, "score": 0.5, "reason": "集成器未启用"}
            try:
                from src.chat.chatter.interest_calculator import get_engagement_scorer
                scorer = get_engagement_scorer()
                if scorer:
                    result = scorer.calculate_interest(message_text)
                    return {
                        "should_reply": getattr(result, 'should_reply', is_targeted),
                        "score": getattr(result, 'score', 0.5),
                        "reason": getattr(result, 'reason', '兴趣度评分')
                    }
            except Exception as e:
                logger.debug(f"兴趣度评分器调用失败: {e}")
            score = 0.5
            should_reply = is_targeted
            reason = "默认规则"
            if is_targeted:
                score = 0.8
                should_reply = True
                reason = "被@提及"
            elif len(message_text.strip()) <= 2:
                score = 0.3
                should_reply = False
                reason = "消息过短"
            elif '?' in message_text or '？' in message_text:
                score = 0.6
                should_reply = True
                reason = "包含疑问"
            return {"should_reply": should_reply, "score": score, "reason": reason}
        except Exception as e:
            logger.error(f"计算参与度失败: {e}")
            return {"should_reply": is_targeted, "score": 0.5, "reason": f"计算失败: {e}"}

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled


_proactive_integration: Optional[ProactiveIntegration] = None
_chatter_integration: Optional[ChatterIntegration] = None


def get_proactive_integration(context: Any = None) -> ProactiveIntegration:
    global _proactive_integration
    if _proactive_integration is None:
        _proactive_integration = ProactiveIntegration(context)
        _proactive_integration.initialize()
    return _proactive_integration


def get_chatter_integration(context: Any = None) -> ChatterIntegration:
    global _chatter_integration
    if _chatter_integration is None:
        _chatter_integration = ChatterIntegration(context)
        _chatter_integration.initialize()
    return _chatter_integration


def init_proactive_integration(context: Any = None):
    global _proactive_integration, _chatter_integration
    try:
        _proactive_integration = ProactiveIntegration(context)
        _proactive_integration.initialize()
        _chatter_integration = ChatterIntegration(context)
        _chatter_integration.initialize()
        logger.info("主动思考集成初始化完成")
    except Exception as e:
        logger.error(f"主动思考集成初始化失败: {e}")


def init_chatter_integration(context: Any = None):
    global _chatter_integration
    try:
        _chatter_integration = ChatterIntegration(context)
        _chatter_integration.initialize()
        logger.info("聊天处理器集成初始化完成")
    except Exception as e:
        logger.error(f"聊天处理器集成初始化失败: {e}")


def reset_integrations():
    global _proactive_integration, _chatter_integration
    _proactive_integration = None
    _chatter_integration = None

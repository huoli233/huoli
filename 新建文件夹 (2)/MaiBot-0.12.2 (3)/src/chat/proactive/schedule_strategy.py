import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Callable, Dict, List
from src.common.logger import get_logger
from src.chat.proactive.silence_detector import get_silence_detector

logger = get_logger("schedule_strategy")


@dataclass
class ScheduleConfig:
    private_min_interval_minutes: int = 30
    private_max_interval_minutes: int = 120
    group_silence_min_hours: float = 1.0
    group_silence_max_hours: float = 2.0
    quiet_hours: str = "22-7"
    max_unanswered_count: int = 1
    enable_hooks: bool = True
    daily_max_private: int = 9999
    daily_max_group: int = 9999
    interest_threshold: float = 0.6


@dataclass
class ScheduleState:
    session_id: str
    session_type: str
    next_trigger_time: Optional[float] = None
    last_trigger_time: Optional[float] = None
    unanswered_count: int = 0
    daily_count: int = 0
    daily_date: str = ""
    is_paused: bool = False
    created_at: float = field(default_factory=time.time)


class ScheduleStrategy(ABC):
    @abstractmethod
    async def on_message_received(self, session_id: str, is_user: bool = True, is_wake_up: bool = False) -> None:
        pass

    @abstractmethod
    async def on_reply_sent(self, session_id: str, is_proactive: bool = False) -> None:
        pass

    @abstractmethod
    async def schedule_next(self, session_id: str) -> Optional[float]:
        pass

    @abstractmethod
    async def cancel(self, session_id: str) -> bool:
        pass


class PrivateChatStrategy(ScheduleStrategy):
    def __init__(self, config: ScheduleConfig, trigger_callback: Callable[[str], Any]):
        self.config = config
        self.trigger_callback = trigger_callback
        self._states: Dict[str, ScheduleState] = {}
        self._timers: Dict[str, asyncio.TimerHandle] = {}
        logger.debug("[PrivateStrategy] 私聊调度策略已初始化")

    async def on_message_received(self, session_id: str, is_user: bool = True, is_wake_up: bool = False) -> None:
        if not is_user:
            return
        state = self._get_or_create_state(session_id, "private")
        if state.unanswered_count > 0:
            logger.info(f"[PrivateStrategy] {session_id} 用户已回复，重置未回复计数 {state.unanswered_count} -> 0")
        state.unanswered_count = 0
        if is_wake_up:
            logger.debug(f"[PrivateStrategy] {session_id} 唤醒词消息，不重置主动调度计时器")
            return
        await self.schedule_next(session_id)

    async def on_reply_sent(self, session_id: str, is_proactive: bool = False) -> None:
        """
        发送回复后调用

        Args:
            session_id: 会话ID
            is_proactive: 是否是主动发送的消息（True=主动消息，False=被动回复）
        """
        state = self._get_or_create_state(session_id, "private")

        if not is_proactive:
            logger.debug(
                f"[PrivateStrategy] {session_id} 被动回复，不重新调度主动消息"
            )
            return

        state.unanswered_count += 1
        state.last_trigger_time = time.time()

        today = datetime.now().strftime("%Y-%m-%d")
        if state.daily_date != today:
            state.daily_date = today
            state.daily_count = 0
        state.daily_count += 1

        logger.debug(
            f"[PrivateStrategy] {session_id} 已发送主动回复，"
            f"未回复计数: {state.unanswered_count}"
        )

        if state.unanswered_count >= self.config.max_unanswered_count:
            logger.info(
                f"[PrivateStrategy] {session_id} 达到未回复上限 "
                f"{self.config.max_unanswered_count}，暂停主动消息"
            )
            state.is_paused = True
            return

        await self.schedule_next(session_id)

    async def schedule_next(self, session_id: str) -> Optional[float]:
        """
        调度下一次主动消息

        Returns:
            下次触发的时间戳，如果不调度则返回 None
        """
        state = self._get_or_create_state(session_id, "private")

        await self.cancel(session_id)

        if state.is_paused:
            logger.debug(f"[PrivateStrategy] {session_id} 已暂停，跳过调度")
            return None

        today = datetime.now().strftime("%Y-%m-%d")
        if state.daily_date == today and state.daily_count >= self.config.daily_max_private:
            logger.info(f"[PrivateStrategy] {session_id} 今日已达上限")
            return None
        min_seconds = self.config.private_min_interval_minutes * 60
        max_seconds = self.config.private_max_interval_minutes * 60
        interval = random.randint(min_seconds, max_seconds)
        trigger_time = time.time() + interval
        state.next_trigger_time = trigger_time
        try:
            loop = asyncio.get_running_loop()
            def _on_timer(captured_id=session_id):
                asyncio.create_task(self._handle_trigger(captured_id))
            self._timers[session_id] = loop.call_later(interval, _on_timer)
            logger.info(f"[PrivateStrategy] {session_id} 已调度，{interval // 60} 分钟后触发")
            return trigger_time
        except RuntimeError as e:
            logger.warning(f"[PrivateStrategy] 无法获取事件循环: {e}")
            return None

    async def _handle_trigger(self, session_id: str) -> None:
        state = self._states.get(session_id)
        if not state:
            return
        if self._is_quiet_time():
            logger.info(f"[PrivateStrategy] {session_id} 当前为免打扰时段，跳过")
            await self.schedule_next(session_id)
            return
        if state.is_paused:
            return
        self._timers.pop(session_id, None)
        state.next_trigger_time = None
        logger.info(f"[PrivateStrategy] {session_id} 触发主动消息")
        try:
            if asyncio.iscoroutinefunction(self.trigger_callback):
                await self.trigger_callback(session_id)
            else:
                self.trigger_callback(session_id)
        except Exception as e:
            logger.error(f"[PrivateStrategy] 触发回调失败: {e}")

    async def cancel(self, session_id: str) -> bool:
        timer = self._timers.pop(session_id, None)
        if timer:
            timer.cancel()
            state = self._states.get(session_id)
            if state:
                state.next_trigger_time = None
            return True
        return False

    def _get_or_create_state(self, session_id: str, session_type: str) -> ScheduleState:
        if session_id not in self._states:
            self._states[session_id] = ScheduleState(session_id=session_id, session_type=session_type)
        return self._states[session_id]

    def _is_quiet_time(self) -> bool:
        return is_quiet_time(self.config.quiet_hours)

    def get_state(self, session_id: str) -> Optional[ScheduleState]:
        return self._states.get(session_id)

    def resume(self, session_id: str) -> bool:
        state = self._states.get(session_id)
        if state and state.is_paused:
            state.is_paused = False
            state.unanswered_count = 0
            asyncio.create_task(self.schedule_next(session_id))
            return True
        return False


class GroupChatStrategy(ScheduleStrategy):
    def __init__(self, config: ScheduleConfig, trigger_callback: Callable[[str], Any]):
        self.config = config
        self.trigger_callback = trigger_callback
        self._states: Dict[str, ScheduleState] = {}
        self._detector = get_silence_detector()
        self._detector.set_callback(self._on_silence_detected)
        logger.debug("[GroupStrategy] 群聊调度策略已初始化")

    def _get_random_silence_minutes(self) -> int:
        min_minutes = int(self.config.group_silence_min_hours * 60)
        max_minutes = int(self.config.group_silence_max_hours * 60)
        return random.randint(min_minutes, max_minutes)

    async def on_message_received(self, session_id: str, is_user: bool = True, is_wake_up: bool = False) -> None:
        state = self._get_or_create_state(session_id, "group")
        if is_user and state.unanswered_count > 0:
            logger.debug(f"[GroupStrategy] {session_id} 用户已回复，重置未回复计数 {state.unanswered_count} -> 0")
            state.unanswered_count = 0
            state.is_paused = False
        if is_user:
            try:
                from src.chat.proactive.engagement_energy import get_engagement_energy_manager
                energy_mgr = get_engagement_energy_manager()
                energy_mgr.on_user_message(session_id)
            except Exception:
                pass
        if is_wake_up:
            logger.debug(f"[GroupStrategy] {session_id} 唤醒词消息，不重置沉默倒计时")
            return
        if not state.is_paused:
            idle_minutes = self._get_random_silence_minutes()
            self._detector.on_activity(session_id, idle_minutes)
            logger.debug(f"[GroupStrategy] {session_id} 重置沉默倒计时: {idle_minutes} 分钟 ({idle_minutes / 60:.1f} 小时)")

    async def on_reply_sent(self, session_id: str, is_proactive: bool = False) -> None:
        if not is_proactive:
            logger.debug(f"[GroupStrategy] {session_id} 被动回复，不更新主动消息状态")
            return
        state = self._get_or_create_state(session_id, "group")
        state.unanswered_count += 1
        state.last_trigger_time = time.time()
        today = datetime.now().strftime("%Y-%m-%d")
        if state.daily_date != today:
            state.daily_date = today
            state.daily_count = 0
        state.daily_count += 1
        logger.info(f"[GroupStrategy] {session_id} 已发送主动消息，今日第 {state.daily_count} 次（上限 {self.config.daily_max_group}）")

    async def schedule_next(self, session_id: str) -> Optional[float]:
        state = self._get_or_create_state(session_id, "group")
        if state.is_paused:
            return None
        today = datetime.now().strftime("%Y-%m-%d")
        if state.daily_date == today and state.daily_count >= self.config.daily_max_group:
            logger.info(f"[GroupStrategy] {session_id} 今日已达上限 {self.config.daily_max_group} 次")
            return None
        idle_minutes = self._get_random_silence_minutes()
        self._detector.on_activity(session_id, idle_minutes)
        return time.time() + idle_minutes * 60

    async def cancel(self, session_id: str) -> bool:
        return self._detector.cancel(session_id)

    async def _on_silence_detected(self, session_id: str) -> None:
        state = self._states.get(session_id)
        if not state:
            return
        if is_quiet_time(self.config.quiet_hours):
            logger.info(f"[GroupStrategy] {session_id} 当前为免打扰时段（{self.config.quiet_hours}），跳过")
            return
        if state.is_paused:
            return
        today = datetime.now().strftime("%Y-%m-%d")
        if state.daily_date == today and state.daily_count >= self.config.daily_max_group:
            logger.info(f"[GroupStrategy] {session_id} 今日已达上限 {self.config.daily_max_group} 次，不再触发")
            return
        if state.unanswered_count >= self.config.max_unanswered_count:
            logger.info(f"[GroupStrategy] {session_id} 上次主动消息未获回复，不再追问")
            return
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(session_id)
            all_states = tracker.get_all_user_states()
            if all_states:
                for uid, user_state in all_states.items():
                    if user_state.trauma_score > 5:
                        logger.info(f"[GroupStrategy] {session_id} 存在创伤用户(trauma={user_state.trauma_score:.1f})，不主动打扰")
                        return
                    if user_state.affection < -50 and user_state.annoyance > 70:
                        logger.info(f"[GroupStrategy] {session_id} 存在敌对用户(affection={user_state.affection:.0f})，不主动打扰")
                        return
        except Exception as e:
            logger.debug(f"[GroupStrategy] 好感度检查失败: {e}")
        topic_relevance = 0.5
        try:
            from src.chat.proactive.topic_extractor import get_topic_extractor
            from src.chat.proactive.session_manager import get_session_hub
            session_hub = get_session_hub()
            session = await session_hub.get_session(session_id, session_id) if session_hub else None
            if session:
                extractor = get_topic_extractor()
                records = session.fetch_recent_records(limit=20)
                if records:
                    topic_info = extractor.extract_from_records(records, max_age_hours=2.0)
                    if topic_info and topic_info.is_valid():
                        topic_relevance = topic_info.relevance_score
                        logger.debug(f"[GroupStrategy] {session_id} 话题兴趣度: {topic_relevance:.2f}")
        except Exception as e:
            logger.debug(f"[GroupStrategy] 话题兴趣度获取失败: {e}")
        try:
            from src.chat.proactive.engagement_energy import get_engagement_energy_manager
            energy_mgr = get_engagement_energy_manager()
            can_speak, reason, current_energy = energy_mgr.can_speak(session_id, interest_level=topic_relevance)
            if not can_speak:
                logger.info(f"[GroupStrategy] {session_id} 能量检查不通过: {reason} (能量={current_energy:.1f}, 兴趣={topic_relevance:.2f})")
                return
        except Exception as e:
            logger.debug(f"[GroupStrategy] 能量检查失败: {e}")
        logger.info(f"[GroupStrategy] {session_id} 群聊沉默触发，今日第 {state.daily_count + 1} 次 (兴趣度={topic_relevance:.2f})")
        try:
            from src.chat.proactive.engagement_energy import get_engagement_energy_manager
            energy_mgr = get_engagement_energy_manager()
            energy_mgr.on_speak(session_id, interest_level=topic_relevance)
        except Exception:
            pass
        try:
            if asyncio.iscoroutinefunction(self.trigger_callback):
                await self.trigger_callback(session_id)
            else:
                self.trigger_callback(session_id)
        except Exception as e:
            logger.error(f"[GroupStrategy] 触发回调失败: {e}")

    def _get_or_create_state(self, session_id: str, session_type: str) -> ScheduleState:
        if session_id not in self._states:
            self._states[session_id] = ScheduleState(session_id=session_id, session_type=session_type)
        return self._states[session_id]

    def get_state(self, session_id: str) -> Optional[ScheduleState]:
        return self._states.get(session_id)

    def resume(self, session_id: str) -> bool:
        state = self._states.get(session_id)
        if state and state.is_paused:
            state.is_paused = False
            state.unanswered_count = 0
            idle_minutes = self._get_random_silence_minutes()
            self._detector.on_activity(session_id, idle_minutes)
            return True
        return False


def is_quiet_time(quiet_hours: str, timezone=None) -> bool:
    if not quiet_hours or quiet_hours.strip() == "":
        return False
    try:
        parts = quiet_hours.split("-")
        if len(parts) != 2:
            return False
        start_hour = int(parts[0].strip())
        end_hour = int(parts[1].strip())
        if timezone:
            current_hour = datetime.now(timezone).hour
        else:
            current_hour = datetime.now().hour
        if start_hour <= end_hour:
            return start_hour <= current_hour < end_hour
        else:
            return current_hour >= start_hour or current_hour < end_hour
    except (ValueError, AttributeError) as e:
        logger.warning(f"[QuietTime] 解析免打扰时段失败: {quiet_hours}, {e}")
        return False


class UnifiedScheduler:
    def __init__(self, config: ScheduleConfig, trigger_callback: Callable[[str], Any]):
        self.config = config
        self.trigger_callback = trigger_callback
        self._private_strategy = PrivateChatStrategy(config, trigger_callback)
        self._group_strategy = GroupChatStrategy(config, trigger_callback)
        self._is_active = False
        logger.info("[UnifiedScheduler] 统一调度器已初始化")

    async def start(self) -> bool:
        if self._is_active:
            logger.debug("[UnifiedScheduler] 调度器已在运行")
            return True
        try:
            if hasattr(self._private_strategy, 'start'):
                await self._private_strategy.start()
            if hasattr(self._group_strategy, 'start'):
                await self._group_strategy.start()
            self._is_active = True
            logger.info("[UnifiedScheduler] 调度器已启动")
            return True
        except Exception as e:
            logger.error(f"[UnifiedScheduler] 启动失败: {e}")
            return False

    async def stop(self) -> bool:
        if not self._is_active:
            return True
        try:
            if hasattr(self._private_strategy, 'stop'):
                await self._private_strategy.stop()
            if hasattr(self._group_strategy, 'stop'):
                await self._group_strategy.stop()
            self._is_active = False
            logger.info("[UnifiedScheduler] 调度器已停止")
            return True
        except Exception as e:
            logger.error(f"[UnifiedScheduler] 停止失败: {e}")
            return False

    def _is_group_session(self, session_id: str) -> bool:
        session_lower = session_id.lower()
        return "group" in session_lower or "groupmessage" in session_lower

    def _is_private_session(self, session_id: str) -> bool:
        session_lower = session_id.lower()
        return "friend" in session_lower or "private" in session_lower or "friendmessage" in session_lower

    def _get_strategy(self, session_id: str) -> Optional[ScheduleStrategy]:
        if self._is_group_session(session_id):
            return self._group_strategy
        elif self._is_private_session(session_id):
            return self._private_strategy
        logger.debug(f"[UnifiedScheduler] 未知会话类型: {session_id}，使用私聊策略")
        return self._private_strategy

    async def on_message_received(self, session_id: str, is_user: bool = True, is_wake_up: bool = False) -> None:
        strategy = self._get_strategy(session_id)
        if strategy:
            await strategy.on_message_received(session_id, is_user, is_wake_up)

    async def on_reply_sent(self, session_id: str, is_proactive: bool = False) -> None:
        strategy = self._get_strategy(session_id)
        if strategy:
            await strategy.on_reply_sent(session_id, is_proactive)

    async def schedule_next(self, session_id: str) -> Optional[float]:
        strategy = self._get_strategy(session_id)
        if strategy:
            return await strategy.schedule_next(session_id)
        return None

    async def cancel(self, session_id: str) -> bool:
        strategy = self._get_strategy(session_id)
        if strategy:
            return await strategy.cancel(session_id)
        return False

    def get_state(self, session_id: str) -> Optional[ScheduleState]:
        strategy = self._get_strategy(session_id)
        if hasattr(strategy, 'get_state'):
            return strategy.get_state(session_id)
        return None

    def resume(self, session_id: str) -> bool:
        strategy = self._get_strategy(session_id)
        if hasattr(strategy, 'resume'):
            return strategy.resume(session_id)
        return False

    def get_statistics(self) -> dict:
        private_states = len(self._private_strategy._states)
        group_states = len(self._group_strategy._states)
        private_active = sum(1 for s in self._private_strategy._states.values() if not s.is_paused)
        group_active = sum(1 for s in self._group_strategy._states.values() if not s.is_paused)
        return {
            "private_sessions": private_states, "private_active": private_active,
            "group_sessions": group_states, "group_active": group_active,
            "total_sessions": private_states + group_states,
            "total_active": private_active + group_active,
        }

    async def cancel_all(self) -> int:
        count = 0
        for session_id in list(self._private_strategy._timers.keys()):
            if await self._private_strategy.cancel(session_id):
                count += 1
        count += get_silence_detector().cancel_all()
        return count


_unified_scheduler: Optional[UnifiedScheduler] = None


def get_unified_scheduler(
    config: Optional[ScheduleConfig] = None,
    trigger_callback: Optional[Callable] = None,
) -> Optional[UnifiedScheduler]:
    global _unified_scheduler
    if _unified_scheduler is None:
        if config is None or trigger_callback is None:
            logger.warning("[UnifiedScheduler] 首次获取需要提供 config 和 callback")
            return None
        _unified_scheduler = UnifiedScheduler(config, trigger_callback)
    return _unified_scheduler


def init_unified_scheduler(config: ScheduleConfig, trigger_callback: Callable) -> UnifiedScheduler:
    global _unified_scheduler
    _unified_scheduler = UnifiedScheduler(config, trigger_callback)
    logger.info("[UnifiedScheduler] 全局统一调度器已初始化")
    return _unified_scheduler

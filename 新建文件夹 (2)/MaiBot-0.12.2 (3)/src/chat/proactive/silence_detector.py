import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, Dict
from src.common.logger import get_logger

logger = get_logger("silence_detector")


@dataclass
class SilenceTimerState:
    channel_id: str
    timer_handle: Optional[asyncio.TimerHandle] = None
    idle_minutes: int = 10
    created_at: float = field(default_factory=time.time)
    last_reset_at: float = field(default_factory=time.time)
    is_paused: bool = False
    trigger_count: int = 0


class GroupSilenceDetector:
    _instance: Optional["GroupSilenceDetector"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._timer_states: Dict[str, SilenceTimerState] = {}
        self._on_silence_callback: Optional[Callable] = None
        self._stats = {
            "total_resets": 0,
            "total_triggers": 0,
            "total_cancels": 0,
        }
        logger.info("[SilenceDetector] 群聊沉默检测器已初始化")

    def set_callback(self, callback: Callable[[str], Any]) -> None:
        self._on_silence_callback = callback
        logger.debug("[SilenceDetector] 已设置沉默触发回调")

    def on_activity(
        self, channel_id: str, idle_minutes: int = 10,
        callback: Optional[Callable] = None,
    ) -> None:
        self._cancel_timer(channel_id)
        if channel_id not in self._timer_states:
            self._timer_states[channel_id] = SilenceTimerState(
                channel_id=channel_id, idle_minutes=idle_minutes,
            )
        state = self._timer_states[channel_id]
        state.idle_minutes = idle_minutes
        state.last_reset_at = time.time()
        if state.is_paused:
            logger.debug(f"[SilenceDetector] {channel_id} 已暂停，跳过设置计时器")
            return
        try:
            loop = asyncio.get_running_loop()
            def _on_timer_expired(captured_id=channel_id):
                self._handle_silence_trigger(captured_id, callback)
            state.timer_handle = loop.call_later(idle_minutes * 60, _on_timer_expired)
            self._stats["total_resets"] += 1
            logger.debug(f"[SilenceDetector] {channel_id} 沉默倒计时已重置: {idle_minutes}分钟")
        except RuntimeError as e:
            logger.warning(f"[SilenceDetector] 无法获取事件循环: {e}")

    def _cancel_timer(self, channel_id: str) -> bool:
        state = self._timer_states.get(channel_id)
        if state and state.timer_handle:
            try:
                state.timer_handle.cancel()
                state.timer_handle = None
                return True
            except Exception as e:
                logger.warning(f"[SilenceDetector] 取消计时器失败: {e}")
        return False

    def _handle_silence_trigger(
        self, channel_id: str, specific_callback: Optional[Callable] = None,
    ) -> None:
        state = self._timer_states.get(channel_id)
        if not state:
            return
        if state.is_paused:
            logger.debug(f"[SilenceDetector] {channel_id} 已暂停，跳过触发")
            return
        state.timer_handle = None
        state.trigger_count += 1
        self._stats["total_triggers"] += 1
        silence_duration = time.time() - state.last_reset_at
        silence_minutes = silence_duration / 60
        logger.info(f"[SilenceDetector] {channel_id} 已沉默 {silence_minutes:.1f} 分钟，触发主动消息")
        callback = specific_callback or self._on_silence_callback
        if callback:
            try:
                asyncio.create_task(
                    self._execute_callback(channel_id, callback),
                    name=f"silence_trigger_{channel_id[:8]}"
                )
            except Exception as e:
                logger.error(f"[SilenceDetector] 创建回调任务失败: {e}")
        else:
            logger.warning(f"[SilenceDetector] {channel_id} 没有设置回调函数")

    async def _execute_callback(self, channel_id: str, callback: Callable) -> None:
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(channel_id)
            else:
                callback(channel_id)
        except Exception as e:
            logger.error(f"[SilenceDetector] 执行回调失败 {channel_id}: {e}")

    def pause(self, channel_id: str) -> bool:
        state = self._timer_states.get(channel_id)
        if not state:
            return False
        self._cancel_timer(channel_id)
        state.is_paused = True
        logger.debug(f"[SilenceDetector] {channel_id} 已暂停")
        return True

    def resume(self, channel_id: str, idle_minutes: Optional[int] = None) -> bool:
        state = self._timer_states.get(channel_id)
        if not state:
            return False
        state.is_paused = False
        minutes = idle_minutes or state.idle_minutes
        self.on_activity(channel_id, minutes)
        logger.debug(f"[SilenceDetector] {channel_id} 已恢复")
        return True

    def cancel(self, channel_id: str) -> bool:
        if self._cancel_timer(channel_id):
            self._timer_states.pop(channel_id, None)
            self._stats["total_cancels"] += 1
            logger.debug(f"[SilenceDetector] {channel_id} 已取消检测")
            return True
        return False

    def cancel_all(self) -> int:
        count = 0
        for channel_id in list(self._timer_states.keys()):
            if self.cancel(channel_id):
                count += 1
        logger.info(f"[SilenceDetector] 已取消 {count} 个频道的沉默检测")
        return count

    def get_remaining_seconds(self, channel_id: str) -> Optional[float]:
        state = self._timer_states.get(channel_id)
        if not state or not state.timer_handle or state.is_paused:
            return None
        elapsed = time.time() - state.last_reset_at
        remaining = (state.idle_minutes * 60) - elapsed
        return max(0, remaining)

    def get_channel_state(self, channel_id: str) -> Optional[Dict]:
        state = self._timer_states.get(channel_id)
        if not state:
            return None
        remaining = self.get_remaining_seconds(channel_id)
        return {
            "channel_id": channel_id,
            "idle_minutes": state.idle_minutes,
            "is_paused": state.is_paused,
            "is_active": state.timer_handle is not None,
            "remaining_seconds": remaining,
            "trigger_count": state.trigger_count,
            "last_reset_at": state.last_reset_at,
        }

    def get_statistics(self) -> Dict:
        active_count = sum(
            1 for s in self._timer_states.values()
            if s.timer_handle and not s.is_paused
        )
        paused_count = sum(1 for s in self._timer_states.values() if s.is_paused)
        return {
            **self._stats,
            "total_channels": len(self._timer_states),
            "active_channels": active_count,
            "paused_channels": paused_count,
        }


_silence_detector: Optional[GroupSilenceDetector] = None


def get_silence_detector() -> GroupSilenceDetector:
    global _silence_detector
    if _silence_detector is None:
        _silence_detector = GroupSilenceDetector()
    return _silence_detector

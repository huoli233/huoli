import asyncio
import time as _tm
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine, Dict, Optional

from src.common.logger import get_logger

logger = get_logger("reply_coord")


# ---------------------------------------------------------------------------
#  优先级枚举
# ---------------------------------------------------------------------------


class ReplyUrgency(IntEnum):
    """回复紧急程度，数值越大优先级越高"""

    PASSIVE = 1  # 被动闲聊
    MONITOR = 2  # 监控触发
    ACTIVE = 3  # 主动/@ 触发


# ---------------------------------------------------------------------------
#  协调器
# ---------------------------------------------------------------------------


@dataclass
class _ChannelSlot:
    """单频道的回复锁与元数据"""

    mutex: asyncio.Lock = field(default_factory=asyncio.Lock)
    current_urgency: ReplyUrgency = ReplyUrgency.PASSIVE
    executing: bool = False
    last_finish_ts: float = 0.0


class ReplyCoordinator:
    """频道级回复协调器

    确保同一频道同一时刻只有一个回复任务在执行。
    高优先级任务可以跳过等待。
    """

    def __init__(self):
        self._slots: Dict[str, _ChannelSlot] = {}

    def _slot(self, channel_id: str) -> _ChannelSlot:
        if channel_id not in self._slots:
            self._slots[channel_id] = _ChannelSlot()
        return self._slots[channel_id]

    async def acquire_permission(
        self,
        channel_id: str,
        urgency: ReplyUrgency = ReplyUrgency.PASSIVE,
        wait_timeout: float = 15.0,
    ) -> bool:
        """尝试获取回复许可

        低优先级任务在锁被占用时等候 wait_timeout 秒；
        高优先级任务直接获得许可（不会阻塞）。
        """
        slot = self._slot(channel_id)
        if slot.executing and urgency <= slot.current_urgency:
            try:
                acquired = await asyncio.wait_for(
                    slot.mutex.acquire(), timeout=wait_timeout
                )
                if acquired:
                    slot.mutex.release()
            except asyncio.TimeoutError:
                logger.debug(f"[回复协调] {channel_id} 等待超时，放弃本次回复")
                return False
        slot.current_urgency = urgency
        return True

    async def execute_reply(
        self,
        channel_id: str,
        reply_coro: Coroutine,
        urgency: ReplyUrgency = ReplyUrgency.PASSIVE,
    ) -> Optional[Any]:
        """在锁保护下执行回复协程"""
        slot = self._slot(channel_id)
        async with slot.mutex:
            slot.executing = True
            slot.current_urgency = urgency
            try:
                result = await reply_coro
                return result
            except Exception as exc:
                logger.error(f"[回复协调] {channel_id} 回复执行失败: {exc}")
                return None
            finally:
                slot.executing = False
                slot.last_finish_ts = _tm.time()
                # 回复完成后通知网关记录一次机器人回复
                try:
                    from src.chat.heart_flow.reply_decision_gateway import (
                        acquire_reply_gateway,
                    )

                    acquire_reply_gateway().record_reply(channel_id)
                except Exception:
                    pass

    def seconds_since_last(self, channel_id: str) -> float:
        """距离上次回复完成的秒数"""
        slot = self._slot(channel_id)
        if slot.last_finish_ts == 0.0:
            return float("inf")
        return _tm.time() - slot.last_finish_ts


# ---------------------------------------------------------------------------
#  单例
# ---------------------------------------------------------------------------

_coordinator_instance: Optional[ReplyCoordinator] = None


def acquire_reply_coordinator() -> ReplyCoordinator:
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = ReplyCoordinator()
    return _coordinator_instance

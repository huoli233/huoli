import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from src.common.logger import get_logger

logger = get_logger("sleep")


class SleepConsolidator:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self._is_sleeping = False
        self._last_sleep_date: Optional[str] = None
        self._sleep_task: Optional[asyncio.Task] = None
        self._sleep_start_hour = 2
        self._sleep_end_hour = 5
        self._importance_keep_threshold = 0.6
        self._importance_delete_threshold = 0.3
        self._max_memories_per_day = 100
        self._last_consolidation_stats: Dict[str, int] = {}

    def _get_memory_model(self):
        try:
            from src.common.database.database_model import MemoryEntry
            return MemoryEntry
        except Exception:
            return None

    def is_sleep_time(self) -> bool:
        now = datetime.now()
        return self._sleep_start_hour <= now.hour < self._sleep_end_hour

    def should_sleep_today(self) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        return self._last_sleep_date != today

    async def check_and_start_sleep(self):
        if not self.is_sleep_time():
            return False
        if not self.should_sleep_today():
            return False
        if self._is_sleeping:
            return False
        await self.start_sleep_consolidation()
        return True

    async def start_sleep_consolidation(self):
        if self._is_sleeping:
            return
        self._is_sleeping = True
        today = datetime.now().strftime("%Y-%m-%d")
        self._last_sleep_date = today
        logger.info("[睡眠整合] 开始夜间记忆整理...")
        try:
            streams = await self._get_active_streams()
            total_stats = {"kept": 0, "compressed": 0, "deleted": 0, "streams": len(streams)}
            for stream_id in streams:
                stats = await self._consolidate_stream(stream_id)
                total_stats["kept"] += stats.get("kept", 0)
                total_stats["compressed"] += stats.get("compressed", 0)
                total_stats["deleted"] += stats.get("deleted", 0)
            self._last_consolidation_stats = total_stats
            logger.info(
                f"[睡眠整合] 整理完成 - "
                f"处理{total_stats['streams']}个会话, "
                f"保留{total_stats['kept']}条, "
                f"压缩{total_stats['compressed']}条, "
                f"删除{total_stats['deleted']}条"
            )
            await self._reset_emotion_states()
        except Exception as e:
            logger.error(f"[睡眠整合] 整理失败: {e}")
        finally:
            self._is_sleeping = False

    async def _get_active_streams(self) -> List[str]:
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return []
        try:
            rows = MemoryEntry.select(MemoryEntry.stream_id).distinct().limit(50)
            return [r.stream_id for r in rows]
        except Exception as e:
            logger.error(f"[睡眠整合] 获取会话列表失败: {e}")
            return []

    async def _consolidate_stream(self, stream_id: str) -> Dict[str, int]:
        stats = {"kept": 0, "compressed": 0, "deleted": 0}
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return stats
        try:
            now = time.time()
            cutoff = now - 86400
            today_memories = list(
                MemoryEntry.select()
                .where(
                    (MemoryEntry.stream_id == stream_id) &
                    (MemoryEntry.created_at >= cutoff) &
                    (MemoryEntry.memory_type == "conversation")
                )
                .order_by(MemoryEntry.created_at.desc())
                .limit(self._max_memories_per_day)
            )
            if not today_memories:
                return stats
            logger.debug(f"[睡眠整合] 会话 {stream_id[:8]}... 有 {len(today_memories)} 条待整理记忆")
            for memory in today_memories:
                importance = memory.importance
                if importance >= self._importance_keep_threshold:
                    memory.memory_type = "fact"
                    memory.importance = min(1.0, importance + 0.1)
                    memory.save()
                    stats["kept"] += 1
                elif importance >= self._importance_delete_threshold:
                    if len(memory.content) > 150:
                        memory.summary = memory.content[:100]
                        memory.memory_type = "compressed"
                        memory.save()
                        stats["compressed"] += 1
                    else:
                        stats["kept"] += 1
                else:
                    memory.delete_instance()
                    stats["deleted"] += 1
            return stats
        except Exception as e:
            logger.error(f"[睡眠整合] 会话 {stream_id[:8]}... 整理失败: {e}")
            return stats

    async def _reset_emotion_states(self):
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            from src.chat.message_receive.chat_stream import get_chat_manager
            chat_mgr = get_chat_manager()
            for sid in chat_mgr.streams:
                try:
                    tracker = get_emotion_tracker(sid)
                    if tracker and hasattr(tracker, '_user_states'):
                        for uid, state in tracker._user_states.items():
                            if hasattr(state, 'annoyance'):
                                state.annoyance = max(0.0, state.annoyance - 50)
                            if hasattr(state, 'mental_fatigue'):
                                state.mental_fatigue = 0.0
                except Exception:
                    pass
            logger.debug("[睡眠整合] 情绪状态已重置")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[睡眠整合] 重置情绪失败: {e}")

    def get_last_stats(self) -> Dict[str, int]:
        return self._last_consolidation_stats.copy()

    def get_sleep_status(self) -> Dict[str, Any]:
        now = datetime.now()
        return {
            "is_sleeping": self._is_sleeping,
            "is_sleep_time": self.is_sleep_time(),
            "last_sleep_date": self._last_sleep_date,
            "current_hour": now.hour,
            "sleep_window": f"{self._sleep_start_hour}:00 - {self._sleep_end_hour}:00",
            "last_stats": self._last_consolidation_stats,
        }


class SleepScheduler:
    def __init__(self, consolidator: SleepConsolidator):
        self.consolidator = consolidator
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        self._check_interval = 1800

    async def start(self):
        if self._running:
            return
        self._running = True
        self._check_task = asyncio.create_task(self._check_loop())
        logger.info("[睡眠调度] 启动完成，将在凌晨自动执行记忆整理")

    async def stop(self):
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

    async def _check_loop(self):
        while self._running:
            try:
                await self.consolidator.check_and_start_sleep()
            except Exception as e:
                logger.error(f"[睡眠调度] 检查失败: {e}")
            await asyncio.sleep(self._check_interval)


_sleep_consolidator: Optional[SleepConsolidator] = None
_sleep_scheduler: Optional[SleepScheduler] = None


def get_sleep_consolidator(memory_manager=None) -> Optional[SleepConsolidator]:
    global _sleep_consolidator
    if _sleep_consolidator is None and memory_manager is not None:
        _sleep_consolidator = SleepConsolidator(memory_manager)
    return _sleep_consolidator


def get_sleep_scheduler(consolidator: Optional[SleepConsolidator] = None) -> Optional[SleepScheduler]:
    global _sleep_scheduler
    if _sleep_scheduler is None and consolidator is not None:
        _sleep_scheduler = SleepScheduler(consolidator)
    return _sleep_scheduler


async def init_sleep_system(memory_manager) -> Optional[SleepScheduler]:
    consolidator = get_sleep_consolidator(memory_manager)
    scheduler = get_sleep_scheduler(consolidator)
    if scheduler:
        await scheduler.start()
    return scheduler

import time
import asyncio
from typing import Optional, Dict, Any
from src.common.logger import get_logger

logger = get_logger("cleanup_scheduler")


class CleanupScheduler:
    _instance: Optional["CleanupScheduler"] = None
    _background_task: Optional[asyncio.Task] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self._running = False
        self._lock = asyncio.Lock()
        self._cleanup_intervals = {
            "chatter": 300,
            "session": 600,
            "context": 300,
            "memory": 1800,
            "active_user": 300,
            "cache": 600,
            "temp_files": 3600,
        }
        self._last_cleanup_times = {
            "chatter": 0.0,
            "session": 0.0,
            "context": 0.0,
            "memory": 0.0,
            "active_user": 0.0,
            "cache": 0.0,
            "temp_files": 0.0,
        }
        self._cleanup_stats = {
            "total_runs": 0,
            "chatter_cleaned": 0,
            "session_cleaned": 0,
            "context_cleaned": 0,
            "memory_cleaned": 0,
            "active_user_cleaned": 0,
            "cache_cleaned": 0,
            "temp_files_cleaned": 0,
        }
        self._cleanup_handlers: Dict[str, Any] = {}
        logger.info(
            f"CleanupScheduler 初始化: 间隔配置={self._cleanup_intervals}"
        )

    def register_cleanup_handler(self, name: str, handler) -> None:
        self._cleanup_handlers[name] = handler
        logger.debug(f"注册清理处理器: {name}")

    def unregister_cleanup_handler(self, name: str) -> None:
        if name in self._cleanup_handlers:
            del self._cleanup_handlers[name]
            logger.debug(f"注销清理处理器: {name}")

    async def start(self):
        if self._running:
            logger.debug("CleanupScheduler 已在运行")
            return
        self._running = True
        CleanupScheduler._background_task = asyncio.create_task(
            self._cleanup_loop(), name="cleanup_scheduler_loop"
        )
        logger.info("CleanupScheduler 已启动")

    async def stop(self):
        self._running = False
        if CleanupScheduler._background_task:
            CleanupScheduler._background_task.cancel()
            try:
                await CleanupScheduler._background_task
            except asyncio.CancelledError:
                logger.debug("清理调度器后台任务已取消")
            CleanupScheduler._background_task = None
        logger.info("CleanupScheduler 已停止")

    async def _cleanup_loop(self):
        await asyncio.sleep(60)
        while self._running:
            try:
                await self._run_cleanup_tasks()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理循环错误: {e}")
                await asyncio.sleep(30)

    async def _run_cleanup_tasks(self):
        now = time.time()
        self._cleanup_stats["total_runs"] += 1
        cleanup_tasks = [
            ("active_user", self._cleanup_active_users),
            ("chatter", self._cleanup_chatters),
            ("session", self._cleanup_sessions),
            ("context", self._cleanup_contexts),
            ("memory", self._cleanup_memory),
            ("cache", self._cleanup_cache),
            ("temp_files", self._cleanup_temp_files),
        ]
        for task_name, cleanup_func in cleanup_tasks:
            interval = self._cleanup_intervals.get(task_name, 300)
            last_time = self._last_cleanup_times.get(task_name, 0)
            if now - last_time >= interval:
                try:
                    async with self._lock:
                        cleaned = await cleanup_func()
                        self._last_cleanup_times[task_name] = now
                        self._cleanup_stats[f"{task_name}_cleaned"] += cleaned
                        if cleaned > 0:
                            logger.debug(f"清理了 {cleaned} 个 {task_name} 项")
                except Exception as e:
                    logger.error(f"清理 {task_name} 失败: {e}")

    async def _cleanup_active_users(self) -> int:
        try:
            from src.modules.active_user_manager import get_active_user_manager

            manager = get_active_user_manager()
            return manager.force_cleanup()
        except ImportError:
            return 0
        except Exception as e:
            logger.debug(f"活跃用户清理失败: {e}")
            return 0

    async def _cleanup_chatters(self) -> int:
        try:
            if "chatter" in self._cleanup_handlers:
                handler = self._cleanup_handlers["chatter"]
                if callable(handler):
                    return await handler()
            return 0
        except Exception as e:
            logger.debug(f"Chatter清理失败: {e}")
            return 0

    async def _cleanup_sessions(self) -> int:
        try:
            if "session" in self._cleanup_handlers:
                handler = self._cleanup_handlers["session"]
                if callable(handler):
                    return await handler()
            return 0
        except Exception as e:
            logger.debug(f"会话清理失败: {e}")
            return 0

    async def _cleanup_contexts(self) -> int:
        try:
            if "context" in self._cleanup_handlers:
                handler = self._cleanup_handlers["context"]
                if callable(handler):
                    return await handler()
            return 0
        except Exception as e:
            logger.debug(f"上下文清理失败: {e}")
            return 0

    async def _cleanup_memory(self) -> int:
        try:
            if "memory" in self._cleanup_handlers:
                handler = self._cleanup_handlers["memory"]
                if callable(handler):
                    return await handler()
            return 0
        except Exception as e:
            logger.debug(f"记忆清理失败: {e}")
            return 0

    async def _cleanup_cache(self) -> int:
        try:
            if "cache" in self._cleanup_handlers:
                handler = self._cleanup_handlers["cache"]
                if callable(handler):
                    return await handler()
            return 0
        except Exception as e:
            logger.debug(f"缓存清理失败: {e}")
            return 0

    async def _cleanup_temp_files(self) -> int:
        try:
            if "temp_files" in self._cleanup_handlers:
                handler = self._cleanup_handlers["temp_files"]
                if callable(handler):
                    return await handler()
            return 0
        except Exception as e:
            logger.debug(f"临时文件清理失败: {e}")
            return 0

    async def force_cleanup_all(self) -> dict:
        results = {}
        results["active_user"] = await self._cleanup_active_users()
        results["chatter"] = await self._cleanup_chatters()
        results["session"] = await self._cleanup_sessions()
        results["context"] = await self._cleanup_contexts()
        results["memory"] = await self._cleanup_memory()
        results["cache"] = await self._cleanup_cache()
        results["temp_files"] = await self._cleanup_temp_files()
        logger.info(f"强制清理完成: {results}")
        return results

    async def force_cleanup(self, task_name: str) -> int:
        cleanup_funcs = {
            "active_user": self._cleanup_active_users,
            "chatter": self._cleanup_chatters,
            "session": self._cleanup_sessions,
            "context": self._cleanup_contexts,
            "memory": self._cleanup_memory,
            "cache": self._cleanup_cache,
            "temp_files": self._cleanup_temp_files,
        }
        if task_name in cleanup_funcs:
            return await cleanup_funcs[task_name]()
        return 0

    def get_statistics(self) -> dict:
        return {
            **self._cleanup_stats,
            "intervals": self._cleanup_intervals,
            "last_cleanup_times": self._last_cleanup_times,
            "is_running": self._running,
            "registered_handlers": list(self._cleanup_handlers.keys()),
        }

    def set_interval(self, task_name: str, seconds: int) -> bool:
        if task_name in self._cleanup_intervals:
            self._cleanup_intervals[task_name] = max(60, seconds)
            return True
        return False

    def get_interval(self, task_name: str) -> int:
        return self._cleanup_intervals.get(task_name, 300)


_cleanup_scheduler: Optional[CleanupScheduler] = None


def get_cleanup_scheduler() -> CleanupScheduler:
    global _cleanup_scheduler
    if _cleanup_scheduler is None:
        _cleanup_scheduler = CleanupScheduler()
    return _cleanup_scheduler


async def start_cleanup_scheduler():
    scheduler = get_cleanup_scheduler()
    await scheduler.start()


async def stop_cleanup_scheduler():
    scheduler = get_cleanup_scheduler()
    await scheduler.stop()


async def force_cleanup_all() -> dict:
    scheduler = get_cleanup_scheduler()
    return await scheduler.force_cleanup_all()

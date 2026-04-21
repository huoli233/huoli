import time
import asyncio
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, Any, Dict, List
from collections import OrderedDict
from src.common.logger import get_logger
from src.common.task_utils import safe_create_task

logger = get_logger("用户预加载")


@dataclass
class AccessPattern:
    user_id: str
    access_count: int = 0
    last_access: float = 0.0
    score: float = 0.0
    channel_id: str = ""


class UserDataPreloader:
    _instance: Optional["UserDataPreloader"] = None
    _background_task: Optional[asyncio.Task] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self._decay_factor = 0.9
        self._preload_threshold = 0.5
        self._max_patterns = 500
        self._preload_interval = 60
        self._patterns: Dict[str, AccessPattern] = {}
        self._preload_callbacks: OrderedDict[
            str, Callable[[], Awaitable[Any]]
        ] = OrderedDict()
        self._max_callbacks = 100
        self._stats = {
            "total_accesses": 0,
            "preload_count": 0,
            "preload_hits": 0,
        }
        self._running = False
        self._lock: Optional[asyncio.Lock] = None
        self._sync_lock = threading.Lock()
        logger.info(
            f"UserDataPreloader 初始化: decay={
                self._decay_factor}, threshold={
                self._preload_threshold}"
        )

    def _ensure_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def record_access(self, user_id: str, channel_id: str = "") -> None:
        async with self._ensure_lock():
            self._stats["total_accesses"] += 1
            now = time.time()
            if user_id in self._patterns:
                pattern = self._patterns[user_id]
                pattern.access_count += 1
                pattern.last_access = now
                if channel_id:
                    pattern.channel_id = channel_id
            else:
                pattern = AccessPattern(
                    user_id=user_id,
                    access_count=1,
                    last_access=now,
                    channel_id=channel_id,
                )
                self._patterns[user_id] = pattern
            pattern.score = self._calculate_score(pattern)
            if len(self._patterns) > self._max_patterns:
                self._cleanup_low_score_patterns()

    def _calculate_score(self, pattern: AccessPattern) -> float:
        now = time.time()
        hours_passed = (now - pattern.last_access) / 3600
        decay = self._decay_factor**hours_passed
        return pattern.access_count * decay

    def _cleanup_low_score_patterns(self):
        if len(self._patterns) <= self._max_patterns:
            return
        for pattern in self._patterns.values():
            pattern.score = self._calculate_score(pattern)
        sorted_patterns = sorted(
            self._patterns.items(), key=lambda x: x[1].score
        )
        to_remove = len(self._patterns) - self._max_patterns
        for user_id, _ in sorted_patterns[:to_remove]:
            del self._patterns[user_id]

    async def should_preload(self, user_id: str) -> bool:
        async with self._ensure_lock():
            pattern = self._patterns.get(user_id)
            if not pattern:
                return False
            pattern.score = self._calculate_score(pattern)
            return pattern.score >= self._preload_threshold

    async def get_users_to_preload(self, limit: int = 20) -> List[str]:
        async with self._ensure_lock():
            for pattern in self._patterns.values():
                pattern.score = self._calculate_score(pattern)
            sorted_patterns = sorted(
                self._patterns.values(),
                key=lambda p: p.score,
                reverse=True,
            )
            return [
                p.user_id
                for p in sorted_patterns[:limit]
                if p.score >= self._preload_threshold
            ]

    def register_preload_callback(
        self, key: str, callback: Callable[[], Awaitable[Any]]
    ) -> None:
        with self._sync_lock:
            self._preload_callbacks[key] = callback
            self._preload_callbacks.move_to_end(key)
            while len(self._preload_callbacks) > self._max_callbacks:
                self._preload_callbacks.popitem(last=False)

    def unregister_preload_callback(self, key: str) -> None:
        with self._sync_lock:
            if key in self._preload_callbacks:
                del self._preload_callbacks[key]

    async def start(self):
        if self._running:
            return
        self._running = True
        UserDataPreloader._background_task = safe_create_task(
            self._preload_loop(),
            name="user_preloader_loop",
        )
        logger.info("UserDataPreloader 已启动")

    async def stop(self):
        self._running = False
        if UserDataPreloader._background_task:
            UserDataPreloader._background_task.cancel()
            try:
                await UserDataPreloader._background_task
            except asyncio.CancelledError:
                pass
            UserDataPreloader._background_task = None
        logger.info("UserDataPreloader 已停止")

    async def _preload_loop(self):
        await asyncio.sleep(30)
        while self._running:
            try:
                await self._run_preload()
                await asyncio.sleep(self._preload_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"预加载循环错误: {e}")
                await asyncio.sleep(30)

    async def _run_preload(self):
        users_to_preload = await self.get_users_to_preload()
        if not users_to_preload:
            return
        for user_id in users_to_preload:
            if user_id in self._preload_callbacks:
                try:
                    callback = self._preload_callbacks[user_id]
                    await callback()
                    self._stats["preload_count"] += 1
                    logger.debug(f"预加载用户数据: {user_id}")
                except Exception as e:
                    logger.debug(f"预加载失败 {user_id}: {e}")

    async def record_hit(self, user_id: str) -> None:
        async with self._ensure_lock():
            self._stats["preload_hits"] += 1

    def get_statistics(self) -> dict:
        hit_rate = (
            self._stats["preload_hits"] / self._stats["preload_count"]
            if self._stats["preload_count"] > 0
            else 0.0
        )
        return {
            **self._stats,
            "preload_hit_rate": hit_rate,
            "tracked_users": len(self._patterns),
            "registered_callbacks": len(self._preload_callbacks),
            "is_running": self._running,
        }

    def set_threshold(self, value: float) -> None:
        self._preload_threshold = max(0.0, min(1.0, value))

    def set_decay_factor(self, value: float) -> None:
        self._decay_factor = max(0.1, min(0.99, value))

    def get_user_pattern(self, user_id: str) -> Optional[AccessPattern]:
        with self._sync_lock:
            return self._patterns.get(user_id)

    def clear_pattern(self, user_id: str) -> bool:
        with self._sync_lock:
            if user_id in self._patterns:
                del self._patterns[user_id]
                return True
            return False

    def clear_all_patterns(self) -> int:
        with self._sync_lock:
            count = len(self._patterns)
            self._patterns.clear()
            return count


_user_preloader: Optional[UserDataPreloader] = None


def get_user_preloader() -> UserDataPreloader:
    global _user_preloader
    if _user_preloader is None:
        _user_preloader = UserDataPreloader()
    return _user_preloader


async def record_user_access(user_id: str, channel_id: str = "") -> None:
    preloader = get_user_preloader()
    await preloader.record_access(user_id, channel_id)


async def start_user_preloader():
    preloader = get_user_preloader()
    await preloader.start()


async def stop_user_preloader():
    preloader = get_user_preloader()
    await preloader.stop()

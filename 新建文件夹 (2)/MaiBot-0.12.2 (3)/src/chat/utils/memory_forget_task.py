import time
import asyncio
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from src.common.logger import get_logger

logger = get_logger("memory_forget_task")


@dataclass
class MemoryEntry:
    key: str
    content: Any
    importance: float = 1.0
    created_ts: float = field(default_factory=time.time)
    accessed_ts: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: float = 3600.0


class MemoryForgetManager:
    def __init__(self):
        self._memories: Dict[str, MemoryEntry] = {}
        self._max_memories = 1000
        self._decay_rate = 0.05
        self._forget_threshold = 0.1
        self._running = False
        self._task: Optional[asyncio.Task] = None
        try:
            from src.config.config import global_config
            self._max_memories = getattr(global_config, 'memory_cache_size', 1000)
        except Exception:
            pass

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._forget_loop())
        logger.info("记忆遗忘管理器已启动")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                return
        logger.info("记忆遗忘管理器已停止")

    async def _forget_loop(self):
        while self._running:
            try:
                await asyncio.sleep(60)
                await self._decay_memories()
                await self._forget_weak_memories()
                await self._evict_if_needed()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"记忆遗忘循环错误: {e}")

    async def _decay_memories(self):
        current_time = time.time()
        for key, entry in list(self._memories.items()):
            age = current_time - entry.accessed_ts
            decay_factor = 1.0 - (self._decay_rate * (age / 3600))
            entry.importance *= max(0.01, decay_factor)

    async def _forget_weak_memories(self):
        to_forget = [
            key for key, entry in self._memories.items()
            if entry.importance < self._forget_threshold
        ]
        for key in to_forget:
            del self._memories[key]
            logger.debug(f"遗忘记忆: {key}")

    async def _evict_if_needed(self):
        if len(self._memories) <= self._max_memories:
            return
        sorted_entries = sorted(
            self._memories.items(),
            key=lambda x: x[1].importance
        )
        evict_count = len(self._memories) - self._max_memories
        for key, _ in sorted_entries[:evict_count]:
            del self._memories[key]

    def add_memory(self, key: str, content: Any, importance: float = 1.0, ttl: float = 3600.0):
        entry = MemoryEntry(key=key, content=content, importance=importance, ttl=ttl)
        self._memories[key] = entry

    def get_memory(self, key: str) -> Optional[Any]:
        if key not in self._memories:
            return None
        entry = self._memories[key]
        entry.accessed_ts = time.time()
        entry.access_count += 1
        entry.importance = min(1.0, entry.importance * 1.1)
        return entry.content

    def update_importance(self, key: str, delta: float):
        if key in self._memories:
            entry = self._memories[key]
            entry.importance = max(0.01, min(1.0, entry.importance + delta))

    def remove_memory(self, key: str) -> bool:
        if key in self._memories:
            del self._memories[key]
            return True
        return False

    def get_all_keys(self) -> List[str]:
        return list(self._memories.keys())

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_memories": len(self._memories),
            "max_memories": self._max_memories,
            "running": self._running,
        }

    def clear(self):
        self._memories.clear()


class ShortTermMemory:
    def __init__(self, capacity: int = 20):
        self._items: List[Any] = []
        self._capacity = capacity
        self._recall_stack: List[Any] = []

    def add(self, item: Any):
        self._items.append(item)
        if len(self._items) > self._capacity:
            self._items.pop(0)

    def get_recent(self, n: int = 5) -> List[Any]:
        return self._items[-n:]

    def recall(self, count: int = 1) -> List[Any]:
        recalled = []
        for _ in range(min(count, len(self._items))):
            if self._items:
                item = self._items.pop()
                self._recall_stack.append(item)
                recalled.append(item)
        return recalled

    def undo_recall(self) -> Optional[Any]:
        if self._recall_stack:
            item = self._recall_stack.pop()
            self._items.append(item)
            return item
        return None

    def clear(self):
        self._items.clear()
        self._recall_stack.clear()

    def __len__(self) -> int:
        return len(self._items)


class LongTermMemory:
    def __init__(self):
        self._memories: Dict[str, Dict[str, Any]] = {}
        self._consolidation_threshold = 3

    def consolidate(self, key: str, content: Any, access_count: int = 1):
        if key in self._memories:
            self._memories[key]["access_count"] += access_count
            self._memories[key]["last_access"] = time.time()
        else:
            self._memories[key] = {
                "content": content,
                "access_count": access_count,
                "created_ts": time.time(),
                "last_access": time.time(),
            }

    def retrieve(self, key: str) -> Optional[Any]:
        if key not in self._memories:
            return None
        self._memories[key]["access_count"] += 1
        self._memories[key]["last_access"] = time.time()
        return self._memories[key]["content"]

    def get_frequent_memories(self, min_count: int = 3) -> List[str]:
        return [
            key for key, data in self._memories.items()
            if data["access_count"] >= min_count
        ]

    def prune_old_memories(self, max_age_seconds: float = 86400):
        current_time = time.time()
        to_remove = [
            key for key, data in self._memories.items()
            if current_time - data["last_access"] > max_age_seconds
        ]
        for key in to_remove:
            del self._memories[key]
        return len(to_remove)


class MemoryConsolidator:
    def __init__(self, short_term: ShortTermMemory, long_term: LongTermMemory):
        self._short = short_term
        self._long = long_term
        self._consolidation_threshold = 5

    def consolidate(self):
        recent = self._short.get_recent(self._consolidation_threshold)
        count = 0
        for item in recent:
            if isinstance(item, dict) and "key" in item:
                self._long.consolidate(item["key"], item.get("content", item))
                count += 1
        return count

    def set_threshold(self, threshold: int):
        self._consolidation_threshold = threshold


class MemoryPrioritizer:
    def __init__(self):
        self._priorities: Dict[str, float] = {}

    def set_priority(self, key: str, priority: float):
        self._priorities[key] = max(0.0, min(1.0, priority))

    def get_priority(self, key: str) -> float:
        return self._priorities.get(key, 0.5)

    def get_high_priority_keys(self, threshold: float = 0.7) -> List[str]:
        return [k for k, p in self._priorities.items() if p >= threshold]

    def decay_priorities(self, decay_factor: float = 0.95):
        for key in self._priorities:
            self._priorities[key] *= decay_factor

    def clear(self):
        self._priorities.clear()


class MemoryTagManager:
    def __init__(self):
        self._tags: Dict[str, List[str]] = defaultdict(list)
        self._tag_index: Dict[str, List[str]] = defaultdict(list)

    def add_tag(self, key: str, tag: str):
        if tag not in self._tags[key]:
            self._tags[key].append(tag)
            self._tag_index[tag].append(key)

    def remove_tag(self, key: str, tag: str):
        if tag in self._tags[key]:
            self._tags[key].remove(tag)
            if key in self._tag_index[tag]:
                self._tag_index[tag].remove(key)

    def get_tags(self, key: str) -> List[str]:
        return list(self._tags.get(key, []))

    def get_keys_by_tag(self, tag: str) -> List[str]:
        return list(self._tag_index.get(tag, []))

    def get_all_tags(self) -> List[str]:
        return list(self._tag_index.keys())

    def clear(self):
        self._tags.clear()
        self._tag_index.clear()


class MemorySearcher:
    def __init__(self, short_term: ShortTermMemory, long_term: LongTermMemory):
        self._short = short_term
        self._long = long_term

    def search(self, query: str) -> List[Tuple[str, Any]]:
        results = []
        for item in self._short._items:
            if isinstance(item, dict):
                key = item.get("key", "")
                if query.lower() in key.lower():
                    results.append((key, item))
        for key in self._long._memories:
            if query.lower() in key.lower():
                content = self._long.retrieve(key)
                if content:
                    results.append((key, content))
        return results


class MemoryStatistics:
    def __init__(self, short_term: ShortTermMemory, long_term: LongTermMemory):
        self._short = short_term
        self._long = long_term

    def get_stats(self) -> Dict[str, Any]:
        short_count = len(self._short)
        long_count = len(self._long._memories)
        return {
            "short_term_count": short_count,
            "long_term_count": long_count,
            "total_count": short_count + long_count,
        }

    def get_memory_ages(self) -> Dict[str, float]:
        ages = {}
        current_time = time.time()
        for key, data in self._long._memories.items():
            ages[key] = current_time - data.get("created_ts", current_time)
        return ages


_forget_manager: Optional[MemoryForgetManager] = None
_short_term_memory: Optional[ShortTermMemory] = None
_long_term_memory: Optional[LongTermMemory] = None
_consolidator: Optional[MemoryConsolidator] = None
_prioritizer: Optional[MemoryPrioritizer] = None
_tag_manager: Optional[MemoryTagManager] = None
_memory_searcher: Optional[MemorySearcher] = None
_memory_stats: Optional[MemoryStatistics] = None


def get_forget_manager() -> MemoryForgetManager:
    global _forget_manager
    if _forget_manager is None:
        _forget_manager = MemoryForgetManager()
    return _forget_manager


def get_short_term_memory() -> ShortTermMemory:
    global _short_term_memory
    if _short_term_memory is None:
        _short_term_memory = ShortTermMemory()
    return _short_term_memory


def get_long_term_memory() -> LongTermMemory:
    global _long_term_memory
    if _long_term_memory is None:
        _long_term_memory = LongTermMemory()
    return _long_term_memory


def get_memory_consolidator() -> MemoryConsolidator:
    global _consolidator
    stm = get_short_term_memory()
    ltm = get_long_term_memory()
    if _consolidator is None:
        _consolidator = MemoryConsolidator(stm, ltm)
    return _consolidator


def get_memory_prioritizer() -> MemoryPrioritizer:
    global _prioritizer
    if _prioritizer is None:
        _prioritizer = MemoryPrioritizer()
    return _prioritizer


def get_tag_manager() -> MemoryTagManager:
    global _tag_manager
    if _tag_manager is None:
        _tag_manager = MemoryTagManager()
    return _tag_manager


def get_memory_searcher() -> MemorySearcher:
    global _memory_searcher
    stm = get_short_term_memory()
    ltm = get_long_term_memory()
    if _memory_searcher is None:
        _memory_searcher = MemorySearcher(stm, ltm)
    return _memory_searcher


def get_memory_statistics() -> MemoryStatistics:
    global _memory_stats
    stm = get_short_term_memory()
    ltm = get_long_term_memory()
    if _memory_stats is None:
        _memory_stats = MemoryStatistics(stm, ltm)
    return _memory_stats


async def initialize_memory_system():
    manager = get_forget_manager()
    await manager.start()
    return manager

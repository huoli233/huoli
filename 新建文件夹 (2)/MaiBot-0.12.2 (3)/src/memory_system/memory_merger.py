import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from src.common.logger import get_logger

logger = get_logger("mem_merger")


class ConsolidationAction(Enum):
    KEEP = "keep"
    MERGE = "merge"
    PROMOTE = "promote"
    DECAY = "decay"
    DISCARD = "discard"


@dataclass
class MemoryFragment:
    fragment_id: str
    content: str
    source_user: str = ""
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    access_count: int = 0
    last_access: float = 0.0
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()[:12]

    def age_hours(self) -> float:
        return (time.time() - self.timestamp) / 3600.0


@dataclass
class ConsolidationStats:
    processed: int = 0
    promoted: int = 0
    merged: int = 0
    decayed: int = 0
    discarded: int = 0
    elapsed_ms: float = 0.0


@dataclass
class MergerConfig:
    consolidation_interval: float = 1800.0
    batch_size: int = 15
    promote_threshold: float = 0.65
    decay_threshold: float = 0.15
    discard_threshold: float = 0.05
    decay_rate: float = 0.92
    age_decay_hours: float = 24.0
    max_short_term: int = 100
    max_long_term: int = 500


class MemoryMerger:
    _shared: Optional["MemoryMerger"] = None

    @classmethod
    def shared(cls) -> "MemoryMerger":
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def __init__(self, config: Optional[MergerConfig] = None):
        self._config = config or MergerConfig()
        self._short_term: Dict[str, MemoryFragment] = {}
        self._long_term: Dict[str, MemoryFragment] = {}
        self._last_consolidation = time.time()
        self._background_task: Optional[asyncio.Task] = None
        self._running = False
        self._on_promote: Optional[Callable[[MemoryFragment], None]] = None
        self._on_discard: Optional[Callable[[MemoryFragment], None]] = None

    def add_fragment(self, fragment: MemoryFragment) -> str:
        if len(self._short_term) >= self._config.max_short_term:
            self._evict_oldest_short_term()
        self._short_term[fragment.fragment_id] = fragment
        logger.debug(f"添加记忆片段: {fragment.fragment_id}")
        return fragment.fragment_id

    def create_and_add(self, content: str, source_user: str = "",
                       importance: float = 0.5,
                       tags: Optional[List[str]] = None) -> MemoryFragment:
        frag_id = f"frag_{int(time.time() * 1000)}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        fragment = MemoryFragment(
            fragment_id=frag_id, content=content,
            source_user=source_user, importance=importance,
            tags=tags or []
        )
        self.add_fragment(fragment)
        return fragment

    def _evict_oldest_short_term(self) -> Optional[str]:
        if not self._short_term:
            return None
        oldest_id = min(self._short_term, key=lambda k: self._short_term[k].timestamp)
        evicted = self._short_term.pop(oldest_id)
        logger.debug(f"淘汰短期记忆: {oldest_id}")
        if self._on_discard:
            self._on_discard(evicted)
        return oldest_id

    def access_fragment(self, fragment_id: str) -> Optional[MemoryFragment]:
        if fragment_id in self._short_term:
            frag = self._short_term[fragment_id]
            frag.access_count += 1
            frag.last_access = time.time()
            frag.importance = min(1.0, frag.importance + 0.05)
            return frag
        if fragment_id in self._long_term:
            frag = self._long_term[fragment_id]
            frag.access_count += 1
            frag.last_access = time.time()
            return frag
        return None

    def search_by_keywords(self, keywords: List[str], top_k: int = 5) -> List[MemoryFragment]:
        all_frags = list(self._short_term.values()) + list(self._long_term.values())
        scored: List[Tuple[float, MemoryFragment]] = []
        keywords_lower = [k.lower() for k in keywords]
        for frag in all_frags:
            content_lower = frag.content.lower()
            match_count = sum(1 for kw in keywords_lower if kw in content_lower)
            if match_count > 0:
                score = match_count * frag.importance * (1 + frag.access_count * 0.1)
                scored.append((score, frag))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [frag for _, frag in scored[:top_k]]
        for frag in results:
            frag.access_count += 1
            frag.last_access = time.time()
        return results

    async def consolidate(self) -> ConsolidationStats:
        from src.memory_system.memory_helpers import calculate_memory_clarity
        start_ts = time.time()
        stats = ConsolidationStats()
        to_promote: List[str] = []
        to_decay: List[str] = []
        to_discard: List[str] = []
        cfg = self._config
        for frag_id, frag in self._short_term.items():
            stats.processed += 1
            clarity = calculate_memory_clarity(
                frag.timestamp, importance=frag.importance, access_count=frag.access_count
            )
            effective_importance = frag.importance * clarity
            if effective_importance >= cfg.promote_threshold:
                to_promote.append(frag_id)
            elif effective_importance <= cfg.discard_threshold:
                to_discard.append(frag_id)
            elif effective_importance <= cfg.decay_threshold:
                to_decay.append(frag_id)
        for frag_id in to_promote:
            frag = self._short_term.pop(frag_id)
            if len(self._long_term) >= cfg.max_long_term:
                self._evict_lowest_importance_long_term()
            self._long_term[frag_id] = frag
            stats.promoted += 1
            if self._on_promote:
                self._on_promote(frag)
        for frag_id in to_decay:
            if frag_id in self._short_term:
                self._short_term[frag_id].importance *= cfg.decay_rate
                stats.decayed += 1
        for frag_id in to_discard:
            if frag_id in self._short_term:
                discarded = self._short_term.pop(frag_id)
                stats.discarded += 1
                if self._on_discard:
                    self._on_discard(discarded)
        for frag in self._long_term.values():
            frag.importance *= (cfg.decay_rate ** 0.5)
        self._last_consolidation = time.time()
        stats.elapsed_ms = (time.time() - start_ts) * 1000
        logger.info(f"记忆整合完成: 处理={stats.processed}, 提升={stats.promoted}, 衰减={stats.decayed}, 丢弃={stats.discarded}")
        return stats

    def _evict_lowest_importance_long_term(self) -> Optional[str]:
        if not self._long_term:
            return None
        lowest_id = min(self._long_term, key=lambda k: self._long_term[k].importance)
        evicted = self._long_term.pop(lowest_id)
        logger.debug(f"淘汰长期记忆: {lowest_id}")
        if self._on_discard:
            self._on_discard(evicted)
        return lowest_id

    async def start_background_consolidation(self) -> None:
        if self._running:
            return
        self._running = True
        self._background_task = asyncio.create_task(self._consolidation_loop())
        logger.info("启动后台记忆整合任务")

    async def stop_background_consolidation(self) -> None:
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        logger.info("停止后台记忆整合任务")

    async def _consolidation_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._config.consolidation_interval)
            if self._running:
                await self.consolidate()

    def set_promote_callback(self, callback: Callable[[MemoryFragment], None]) -> None:
        self._on_promote = callback

    def set_discard_callback(self, callback: Callable[[MemoryFragment], None]) -> None:
        self._on_discard = callback

    def get_short_term_count(self) -> int:
        return len(self._short_term)

    def get_long_term_count(self) -> int:
        return len(self._long_term)

    def get_all_short_term(self) -> List[MemoryFragment]:
        return list(self._short_term.values())

    def get_all_long_term(self) -> List[MemoryFragment]:
        return list(self._long_term.values())

    def clear_short_term(self) -> int:
        count = len(self._short_term)
        self._short_term.clear()
        logger.info(f"清空短期记忆: {count}条")
        return count

    def clear_all(self) -> Tuple[int, int]:
        short_count = len(self._short_term)
        long_count = len(self._long_term)
        self._short_term.clear()
        self._long_term.clear()
        logger.info(f"清空所有记忆: 短期={short_count}, 长期={long_count}")
        return short_count, long_count


def get_memory_merger() -> MemoryMerger:
    return MemoryMerger.shared()


def add_memory(content: str, source: str = "", importance: float = 0.5) -> MemoryFragment:
    return MemoryMerger.shared().create_and_add(content, source, importance)


def search_memories(keywords: List[str], top_k: int = 5) -> List[MemoryFragment]:
    return MemoryMerger.shared().search_by_keywords(keywords, top_k)


async def run_consolidation() -> ConsolidationStats:
    return await MemoryMerger.shared().consolidate()

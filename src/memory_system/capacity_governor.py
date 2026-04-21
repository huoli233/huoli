import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine
from src.memory_system.memory_models import BufferedMemory, capacity_to_retention_index

logger = get_logger("capacity_governor")


class MemoryTier(Enum):
    IMMEDIATE = "immediate"
    WORKING = "working"
    RECENT = "recent"
    MIDTERM = "midterm"
    LONGTERM = "longterm"


@dataclass
class EvictionDecision:
    memory_id: str
    tier: MemoryTier
    clarity: float


def calculate_memory_clarity(
    created_at: float, importance: float, access_count: int
) -> float:
    now = time.time()
    age_days = (now - created_at) / 86400.0
    age_factor = max(0.1, 1.0 - age_days / 365.0)
    access_factor = min(1.0, 0.3 + access_count * 0.1)
    return importance * age_factor * access_factor


class PromptManagerInterface:
    def get_raw(self, name: str) -> Optional[str]:
        raise NotImplementedError


class DefaultPromptManager(PromptManagerInterface):
    def get_raw(self, name: str) -> Optional[str]:
        return None


class LLMClientInterface:
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError


class DefaultLLMClient(LLMClientInterface):
    async def generate(self, prompt: str) -> str:
        return '{"evict_ids": []}'


class CapacityGovernor:
    def __init__(
        self, config_engine=None, prompt_manager=None, llm_client=None
    ):
        self._config = config_engine or get_default_config_engine()
        self._prompt_manager = prompt_manager or DefaultPromptManager()
        self._llm_client = llm_client or DefaultLLMClient()

        self._review_interval_seconds = 900
        self._tier_limits: Dict[MemoryTier, int] = {
            MemoryTier.IMMEDIATE: 120,
            MemoryTier.WORKING: 180,
            MemoryTier.RECENT: 260,
            MemoryTier.MIDTERM: 380,
            MemoryTier.LONGTERM: 800,
        }
        self._review_task: Optional[asyncio.Task] = None
        self._background_tasks: Set[asyncio.Task] = set()
        self._running = False
        self._last_review_time: float = 0.0
        self._eviction_stats: Dict[str, int] = {}
        self._load_config()

    def _load_config(self) -> None:
        self._review_interval_seconds = max(
            60,
            self._config.get(
                "memory_capacity", "review_interval_seconds", 900
            ),
        )

        raw_limits = self._config.get(
            "memory_capacity", "max_count_by_tier", {}
        )
        if isinstance(raw_limits, dict):
            self._tier_limits = {
                MemoryTier.IMMEDIATE: max(
                    1, int(raw_limits.get("immediate", 120))
                ),
                MemoryTier.WORKING: max(
                    1, int(raw_limits.get("working", 180))
                ),
                MemoryTier.RECENT: max(1, int(raw_limits.get("recent", 260))),
                MemoryTier.MIDTERM: max(
                    1, int(raw_limits.get("midterm", 380))
                ),
                MemoryTier.LONGTERM: max(
                    1, int(raw_limits.get("longterm", 800))
                ),
            }

    def bind_llm(self, prompt_manager, llm_client) -> None:
        self._prompt_manager = prompt_manager
        self._llm_client = llm_client

    def evaluate_overflow(
        self,
        tier: MemoryTier,
        memories: Dict[str, BufferedMemory],
    ) -> List[str]:
        limit = self._tier_limits.get(tier, 0)
        count = len(memories)
        if count <= limit:
            return []

        overflow = count - limit
        ranked = self._rank_eviction_candidates(tier=tier, memories=memories)
        return [decision.memory_id for decision in ranked[:overflow]]

    def _rank_eviction_candidates(
        self,
        tier: MemoryTier,
        memories: Dict[str, BufferedMemory],
    ) -> List[EvictionDecision]:
        candidates: List[EvictionDecision] = []
        for memory_id, memory in memories.items():
            clarity = calculate_memory_clarity(
                created_at=memory.created_at,
                importance=memory.importance,
                access_count=memory.access_count,
            )
            candidates.append(
                EvictionDecision(
                    memory_id=memory_id, tier=tier, clarity=clarity
                )
            )

        candidates.sort(key=lambda item: item.clarity)
        return candidates

    async def filter_with_llm(
        self,
        tier: MemoryTier,
        memories: Dict[str, BufferedMemory],
        memory_ids: List[str],
    ) -> List[str]:
        if not memory_ids:
            return []
        if self._prompt_manager is None or self._llm_client is None:
            return memory_ids

        payload = []
        for memory_id in memory_ids:
            memory = memories.get(memory_id)
            if memory is None:
                continue
            payload.append(
                {
                    "memory_id": memory_id,
                    "summary": memory.summary or memory.content[:120],
                    "importance": memory.importance,
                    "access_count": memory.access_count,
                    "created_at": memory.created_at,
                }
            )

        if not payload:
            return []

        prompt = self._build_prompt(tier=tier, candidates=payload)
        _llm_timeout = 15.0
        try:
            response = await asyncio.wait_for(
                self._llm_client.generate(prompt), timeout=_llm_timeout
            )
            data = json.loads(response)
            selected = data.get("evict_ids", [])
            if isinstance(selected, list):
                selected_ids = [str(item) for item in selected]
                return [
                    memory_id
                    for memory_id in memory_ids
                    if memory_id in selected_ids
                ]
            return memory_ids
        except asyncio.TimeoutError:
            logger.warning(
                f"[容量治理] LLM淘汰判断超时({_llm_timeout}s)，使用基础策略"
            )
            return memory_ids
        except Exception as e:
            logger.debug(f"[容量治理] LLM淘汰判断失败，使用基础策略: {e}")
            return memory_ids

    def _build_prompt(
        self, tier: MemoryTier, candidates: List[Dict[str, object]]
    ) -> str:
        if self._prompt_manager is None:
            return ""

        template = self._prompt_manager.get_raw("memory_eviction_judge")
        if not template:
            return ""

        personality = self._config.get("personality", "name", "")

        return template.format(
            main_personality=personality,
            memory_tier=tier.name.lower(),
            candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
        )

    async def apply_capacity_control(
        self,
        layers: Dict[MemoryTier, Dict[str, BufferedMemory]],
        on_evict=None,
    ) -> Dict[str, int]:
        removed_counts: Dict[str, int] = {}

        for tier, memories in layers.items():
            base_candidates = self.evaluate_overflow(
                tier=tier, memories=memories
            )
            final_candidates = await self.filter_with_llm(
                tier=tier, memories=memories, memory_ids=base_candidates
            )

            removed = 0
            evicted_ids = []
            for memory_id in final_candidates:
                if memory_id in memories:
                    del memories[memory_id]
                    evicted_ids.append(memory_id)
                    removed += 1

            if removed > 0:
                removed_counts[tier.name.lower()] = removed
                self._notify_eviction(tier, evicted_ids, on_evict)

        self._eviction_stats = removed_counts
        self._last_review_time = time.time()
        return removed_counts

    def _notify_eviction(
        self,
        tier: MemoryTier,
        evicted_ids: List[str],
        on_evict=None,
    ) -> None:
        """驱逐后通知海马体和数据库同步删除。"""
        retention_idx = capacity_to_retention_index(tier.value)
        if on_evict and callable(on_evict):
            try:
                on_evict(evicted_ids, tier_name=tier.value, retention_idx=retention_idx)
            except Exception as exc:
                logger.debug(f"[容量治理] 驱逐回调失败: {exc}")
        self._sync_eviction_to_hippocampus(evicted_ids, retention_idx)
        self._sync_eviction_to_database(evicted_ids)

    @staticmethod
    def _sync_eviction_to_hippocampus(
        evicted_ids: List[str], retention_idx: Optional[int]
    ) -> None:
        """将驱逐的碎片从海马体缓冲区中移除。"""
        if retention_idx is None:
            return
        try:
            from src.memory_system.hippocampus_buffer import (
                get_hippocampus_buffer,
                _buffer_pool,
            )
            for buffer in _buffer_pool.values():
                for frag_id in evicted_ids:
                    for layer_bucket in buffer._buckets.values():
                        if frag_id in layer_bucket:
                            del layer_bucket[frag_id]
                            break
        except Exception as exc:
            logger.debug(f"[容量治理] 海马体同步驱逐失败: {exc}")

    @staticmethod
    def _sync_eviction_to_database(evicted_ids: List[str]) -> None:
        """将驱逐的记录从数据库中删除。"""
        if not evicted_ids:
            return
        try:
            from src.common.database.database_model import MemoryRecord
            MemoryRecord.delete().where(
                MemoryRecord.record_id.in_(evicted_ids)
            ).execute()
        except Exception as exc:
            logger.debug(f"[容量治理] 数据库同步驱逐失败: {exc}")

    def start_periodic_review(self, review_callback) -> None:
        if self._review_task is not None:
            return

        self._running = True

        async def _loop():
            while self._running:
                await asyncio.sleep(self._review_interval_seconds)
                try:
                    await review_callback()
                except Exception as e:
                    logger.error(f"[容量治理] 定期检查失败: {e}")

        self._review_task = asyncio.create_task(_loop())
        logger.info(
            f"[容量治理] 定期检查已启动, 间隔 {self._review_interval_seconds} 秒"
        )

    async def stop_periodic_review(self) -> None:
        self._running = False
        if self._review_task:
            self._review_task.cancel()
            try:
                await self._review_task
            except asyncio.CancelledError:
                pass
            self._review_task = None

        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.debug("容量治理后台任务已取消")
        self._background_tasks.clear()

    def get_tier_limit(self, tier: MemoryTier) -> int:
        return self._tier_limits.get(tier, 0)

    def get_tier_usage(
        self, tier: MemoryTier, memories: Dict[str, BufferedMemory]
    ) -> Dict[str, Any]:
        limit = self._tier_limits.get(tier, 0)
        count = len(memories)
        return {
            "tier": tier.name.lower(),
            "count": count,
            "limit": limit,
            "usage_ratio": count / max(1, limit),
            "overflow": max(0, count - limit),
        }

    def get_all_tier_usage(
        self, layers: Dict[MemoryTier, Dict[str, BufferedMemory]]
    ) -> Dict[str, Any]:
        result = {}
        total_count = 0
        total_limit = 0
        for tier in MemoryTier:
            memories = layers.get(tier, {})
            usage = self.get_tier_usage(tier, memories)
            result[tier.name.lower()] = usage
            total_count += usage["count"]
            total_limit += usage["limit"]
        result["total"] = {
            "count": total_count,
            "limit": total_limit,
            "usage_ratio": total_count / max(1, total_limit),
        }
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "review_interval_seconds": self._review_interval_seconds,
            "tier_limits": {
                t.name.lower(): v for t, v in self._tier_limits.items()
            },
            "last_review_time": self._last_review_time,
            "eviction_stats": self._eviction_stats,
            "is_running": self._running,
        }


_capacity_governor: Optional[CapacityGovernor] = None


def get_capacity_governor(
    config_engine=None, prompt_manager=None, llm_client=None
) -> CapacityGovernor:
    global _capacity_governor
    if _capacity_governor is None:
        _capacity_governor = CapacityGovernor(
            config_engine=config_engine,
            prompt_manager=prompt_manager,
            llm_client=llm_client,
        )
    else:
        _capacity_governor.bind_llm(
            prompt_manager=prompt_manager, llm_client=llm_client
        )
    return _capacity_governor

import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine
from src.memory_system.memory_models import BufferedMemory, retention_to_dedup_tier

logger = get_logger("去重融合")


class MemoryTier(Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    ARCHIVE = "archive"


@dataclass
class MergePair:
    memory_a_id: str
    memory_b_id: str
    tier: MemoryTier
    similarity: float


class EmbeddingVaultInterface:
    async def search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError

    async def add_document(
        self, doc_id: str, content: str, metadata: Dict[str, Any]
    ) -> None:
        raise NotImplementedError

    def remove_document(self, doc_id: str) -> None:
        raise NotImplementedError


class DefaultEmbeddingVault(EmbeddingVaultInterface):
    async def search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        return []

    async def add_document(
        self, doc_id: str, content: str, metadata: Dict[str, Any]
    ) -> None:
        raise NotImplementedError("add_document 需要子类实现")

    def remove_document(self, doc_id: str) -> None:
        raise NotImplementedError("remove_document 需要子类实现")


class LLMClientInterface:
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError


class DefaultLLMClient(LLMClientInterface):
    async def generate(self, prompt: str) -> str:
        return '{"should_merge": false}'


class PromptManagerInterface:
    def get_raw(self, name: str) -> Optional[str]:
        raise NotImplementedError


class DefaultPromptManager(PromptManagerInterface):
    def get_raw(self, name: str) -> Optional[str]:
        return None


class DedupFuser:
    def __init__(
        self,
        config_engine=None,
        prompt_manager: Optional[PromptManagerInterface] = None,
        llm_client: Optional[LLMClientInterface] = None,
        embedding_vault: Optional[EmbeddingVaultInterface] = None,
    ):
        self._config = config_engine or get_default_config_engine()
        self._prompt_manager = prompt_manager or DefaultPromptManager()
        self._llm_client = llm_client or DefaultLLMClient()
        self._embedding_vault = embedding_vault or DefaultEmbeddingVault()

        self._similarity_threshold = 0.9
        self._scan_interval_seconds = 1800
        self._max_pairs_per_scan = 60
        self._merge_top_k = 20
        self._scan_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_scan_time: float = 0.0
        self._merge_stats: Dict[str, int] = {}
        self._load_config()

    def _load_config(self) -> None:
        self._similarity_threshold = self._config.get(
            "memory_dedup", "similarity_threshold", 0.9
        )
        self._scan_interval_seconds = max(
            60, self._config.get("memory_dedup", "scan_interval_seconds", 1800)
        )
        self._max_pairs_per_scan = max(
            1, self._config.get("memory_dedup", "max_pairs_per_scan", 60)
        )
        self._merge_top_k = max(
            2, self._config.get("memory_dedup", "merge_top_k", 20)
        )

    def bind_llm(
        self,
        prompt_manager: PromptManagerInterface,
        llm_client: LLMClientInterface,
    ) -> None:
        self._prompt_manager = prompt_manager
        self._llm_client = llm_client

    def bind_embedding_vault(
        self, embedding_vault: Optional[EmbeddingVaultInterface]
    ) -> None:
        self._embedding_vault = embedding_vault or DefaultEmbeddingVault()

    async def scan_duplicates(
        self, layers: Dict[MemoryTier, Dict[str, BufferedMemory]]
    ) -> List[MergePair]:
        pair_candidates: List[MergePair] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for tier, memories in layers.items():
            if len(memories) < 2:
                continue

            ordered = sorted(
                memories.values(),
                key=lambda item: item.created_at,
                reverse=True,
            )
            for source_memory in ordered:
                if len(pair_candidates) >= self._max_pairs_per_scan:
                    return pair_candidates

                try:
                    hits = await self._embedding_vault.search(
                        query=source_memory.content, top_k=self._merge_top_k
                    )
                except Exception as e:
                    logger.debug(f"[记忆去重] 向量检索失败: {e}")
                    continue

                for target_id, similarity in hits:
                    if target_id == source_memory.memory_id:
                        continue
                    if similarity < self._similarity_threshold:
                        continue
                    if target_id not in memories:
                        continue

                    pair_key = tuple(
                        sorted((source_memory.memory_id, target_id))
                    )
                    if pair_key in seen_pairs:
                        continue

                    seen_pairs.add(pair_key)
                    should_merge = await self._judge_should_merge(
                        memory_a=memories[pair_key[0]],
                        memory_b=memories[pair_key[1]],
                        similarity=similarity,
                    )
                    if should_merge:
                        pair_candidates.append(
                            MergePair(
                                memory_a_id=pair_key[0],
                                memory_b_id=pair_key[1],
                                tier=tier,
                                similarity=similarity,
                            )
                        )
                        if len(pair_candidates) >= self._max_pairs_per_scan:
                            return pair_candidates

        return pair_candidates

    async def fuse_pair(
        self, memory_a: BufferedMemory, memory_b: BufferedMemory
    ) -> Dict[str, object]:
        prompt = self._build_fuse_prompt(memory_a=memory_a, memory_b=memory_b)
        if not prompt:
            return {}

        try:
            response = await self._llm_client.generate(prompt)
            data = json.loads(response)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.debug(f"[记忆去重] 融合内容生成失败: {e}")
            return {}

    async def apply_dedup(
        self, layers: Dict[MemoryTier, Dict[str, BufferedMemory]]
    ) -> Dict[str, int]:
        merge_pairs = await self.scan_duplicates(layers)
        merged_counts: Dict[str, int] = {}
        if not merge_pairs:
            return merged_counts

        for pair in merge_pairs:
            tier_memories = layers.get(pair.tier, {})
            memory_a = tier_memories.get(pair.memory_a_id)
            memory_b = tier_memories.get(pair.memory_b_id)
            if memory_a is None or memory_b is None:
                continue

            fused_payload = await self.fuse_pair(memory_a, memory_b)
            merged_content = str(
                fused_payload.get("merged_content", "")
            ).strip()
            if not merged_content:
                continue

            merged_summary = str(
                fused_payload.get("merged_summary", "")
            ).strip()
            if not merged_summary:
                merged_summary = (
                    merged_content[:80]
                    if len(merged_content) > 80
                    else merged_content
                )

            merged_importance = fused_payload.get(
                "importance", max(memory_a.importance, memory_b.importance)
            )
            try:
                merged_importance = max(
                    0.0, min(1.0, float(merged_importance))
                )
            except Exception:
                merged_importance = max(
                    memory_a.importance, memory_b.importance
                )

            merged_keywords = fused_payload.get(
                "keywords", memory_a.keywords or memory_b.keywords
            )
            if not isinstance(merged_keywords, list):
                merged_keywords = memory_a.keywords or memory_b.keywords

            memory_a.content = merged_content
            memory_a.summary = merged_summary
            memory_a.importance = merged_importance
            memory_a.keywords = [str(item) for item in merged_keywords][:20]
            memory_a.access_count = (
                memory_a.access_count + memory_b.access_count
            )
            memory_a.last_accessed = max(
                memory_a.last_accessed, memory_b.last_accessed
            )
            memory_a.created_at = min(memory_a.created_at, memory_b.created_at)

            del tier_memories[memory_b.memory_id]
            merged_counts[pair.tier.name.lower()] = (
                merged_counts.get(pair.tier.name.lower(), 0) + 1
            )

            await self._refresh_embedding_index(memory_a, memory_b.memory_id)

        self._merge_stats = merged_counts
        return merged_counts

    async def _refresh_embedding_index(
        self, merged_memory: BufferedMemory, removed_id: str
    ) -> None:
        self._embedding_vault.remove_document(merged_memory.memory_id)
        self._embedding_vault.remove_document(removed_id)
        try:
            await self._embedding_vault.add_document(
                doc_id=merged_memory.memory_id,
                content=merged_memory.content,
                metadata={
                    "channel_id": merged_memory.channel_id,
                    "user_id": merged_memory.user_id or "",
                    "importance": merged_memory.importance,
                    "source": "hippocampus",
                },
            )
        except Exception as e:
            logger.debug(f"[记忆去重] 刷新向量索引失败: {e}")

    async def _judge_should_merge(
        self,
        memory_a: BufferedMemory,
        memory_b: BufferedMemory,
        similarity: float,
    ) -> bool:
        prompt = self._build_judge_prompt(
            memory_a=memory_a, memory_b=memory_b, similarity=similarity
        )
        if not prompt:
            return False

        try:
            response = await self._llm_client.generate(prompt)
            data = json.loads(response)
            return bool(data.get("should_merge", False))
        except Exception as e:
            logger.debug(f"[记忆去重] 合并判断失败: {e}")
            return False

    def _build_judge_prompt(
        self,
        memory_a: BufferedMemory,
        memory_b: BufferedMemory,
        similarity: float,
    ) -> str:
        if self._prompt_manager is None:
            return ""

        template = self._prompt_manager.get_raw("memory_merge_judge")
        if not template:
            return ""

        personality = self._config.get("personality", "name", "")

        return template.format(
            main_personality=personality,
            memory_a=memory_a.content,
            memory_b=memory_b.content,
            similarity=similarity,
            importance_a=memory_a.importance,
            importance_b=memory_b.importance,
            access_count_a=memory_a.access_count,
            access_count_b=memory_b.access_count,
        )

    def _build_fuse_prompt(
        self, memory_a: BufferedMemory, memory_b: BufferedMemory
    ) -> str:
        if self._prompt_manager is None:
            return ""

        template = self._prompt_manager.get_raw("memory_merge_fuse")
        if not template:
            return ""

        personality = self._config.get("personality", "name", "")

        return template.format(
            main_personality=personality,
            memory_a=memory_a.content,
            memory_b=memory_b.content,
            summary_a=memory_a.summary or "",
            summary_b=memory_b.summary or "",
            importance_a=memory_a.importance,
            importance_b=memory_b.importance,
            keywords_a=", ".join(memory_a.keywords),
            keywords_b=", ".join(memory_b.keywords),
        )

    def start_periodic_scan(self, scan_callback) -> None:
        if self._scan_task is not None:
            return

        self._running = True

        async def _loop():
            while self._running:
                await asyncio.sleep(self._scan_interval_seconds)
                try:
                    await scan_callback()
                    self._last_scan_time = time.time()
                except Exception as e:
                    logger.error(f"[记忆去重] 定期扫描失败: {e}")

        self._scan_task = asyncio.create_task(_loop())
        logger.info(
            f"[记忆去重] 定期扫描已启动, 间隔 {self._scan_interval_seconds} 秒"
        )

    async def stop_periodic_scan(self) -> None:
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
            self._scan_task = None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "similarity_threshold": self._similarity_threshold,
            "scan_interval_seconds": self._scan_interval_seconds,
            "max_pairs_per_scan": self._max_pairs_per_scan,
            "merge_top_k": self._merge_top_k,
            "last_scan_time": self._last_scan_time,
            "merge_stats": self._merge_stats,
            "is_running": self._running,
        }


_dedup_fuser: Optional[DedupFuser] = None


def get_dedup_fuser(
    config_engine=None,
    prompt_manager: Optional[PromptManagerInterface] = None,
    llm_client: Optional[LLMClientInterface] = None,
    embedding_vault: Optional[EmbeddingVaultInterface] = None,
) -> DedupFuser:
    global _dedup_fuser
    if _dedup_fuser is None:
        _dedup_fuser = DedupFuser(
            config_engine=config_engine,
            prompt_manager=prompt_manager,
            llm_client=llm_client,
            embedding_vault=embedding_vault,
        )
    else:
        if prompt_manager is not None and llm_client is not None:
            _dedup_fuser.bind_llm(prompt_manager, llm_client)
        if embedding_vault is not None:
            _dedup_fuser.bind_embedding_vault(embedding_vault)
    return _dedup_fuser

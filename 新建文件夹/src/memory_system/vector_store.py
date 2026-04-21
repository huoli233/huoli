import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine

logger = get_logger("向量存储")


@dataclass
class EmbeddingRecord:
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]
    created_at: float


class AdapterInterface:
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        raise NotImplementedError


class DefaultAdapter(AdapterInterface):
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        return None


class EmbeddingVault:
    def __init__(
        self, config_engine=None, adapter: Optional[AdapterInterface] = None
    ):
        self._config = config_engine or get_default_config_engine()
        self._adapter = adapter or DefaultAdapter()
        self._records: Dict[str, EmbeddingRecord] = {}
        self._norm_cache: Dict[str, float] = {}

        self._max_documents = 5000
        self._embedding_dimension = 1536
        self._load_config()

    def _load_config(self) -> None:
        self._max_documents = max(
            1, int(self._config.get("memory_retrieval", "max_documents", 5000))
        )
        self._embedding_dimension = max(
            1,
            int(
                self._config.get(
                    "memory_retrieval", "embedding_dimension", 1536
                )
            ),
        )

    def bind_adapter(self, adapter: AdapterInterface) -> None:
        self._adapter = adapter

    async def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self._adapter or not content:
            return False
        embedding = await self._adapter.get_embedding(content)
        if not embedding:
            return False
        normalized = self._normalize_dimension(embedding)
        record = EmbeddingRecord(
            doc_id=doc_id,
            content=content,
            metadata=metadata or {},
            embedding=normalized,
            created_at=time.time(),
        )
        self._records[doc_id] = record
        self._norm_cache[doc_id] = self._vector_norm(normalized)
        self._enforce_capacity()
        return True

    def remove_document(self, doc_id: str) -> None:
        self._records.pop(doc_id, None)
        self._norm_cache.pop(doc_id, None)

    async def search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        if not self._adapter or not query or not self._records:
            return []
        query_embedding = await self._adapter.get_embedding(query)
        if not query_embedding:
            return []
        normalized_query = self._normalize_dimension(query_embedding)
        query_norm = self._vector_norm(normalized_query)
        if query_norm <= 0.0:
            return []

        scored: List[Tuple[str, float]] = []
        for doc_id, record in self._records.items():
            score = self._cosine_similarity(
                normalized_query,
                query_norm,
                record.embedding,
                self._norm_cache.get(doc_id, 0.0),
            )
            if score > 0.0:
                scored.append((doc_id, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, top_k)]

    def get_document(self, doc_id: str) -> Optional[EmbeddingRecord]:
        return self._records.get(doc_id)

    def get_size(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()
        self._norm_cache.clear()

    def _enforce_capacity(self) -> None:
        overflow = len(self._records) - self._max_documents
        if overflow <= 0:
            return
        ordered_ids = sorted(
            self._records.keys(), key=lambda key: self._records[key].created_at
        )
        for doc_id in ordered_ids[:overflow]:
            self.remove_document(doc_id)

    def _normalize_dimension(self, vector: List[float]) -> List[float]:
        dim = self._embedding_dimension
        if len(vector) == dim:
            return [float(value) for value in vector]
        if len(vector) > dim:
            return [float(value) for value in vector[:dim]]
        padded = [float(value) for value in vector]
        padded.extend([0.0] * (dim - len(vector)))
        return padded

    @staticmethod
    def _vector_norm(vector: List[float]) -> float:
        return math.sqrt(sum(value * value for value in vector))

    @staticmethod
    def _cosine_similarity(
        query_vec: List[float],
        query_norm: float,
        doc_vec: List[float],
        doc_norm: float,
    ) -> float:
        if query_norm <= 0.0 or doc_norm <= 0.0:
            return 0.0
        dot = sum(left * right for left, right in zip(query_vec, doc_vec))
        return max(0.0, dot / (query_norm * doc_norm))

    def export_state(self) -> Dict[str, Any]:
        return {
            "records": [
                {
                    "doc_id": r.doc_id,
                    "content": r.content,
                    "metadata": r.metadata,
                    "embedding": r.embedding,
                    "created_at": r.created_at,
                }
                for r in self._records.values()
            ],
            "max_documents": self._max_documents,
            "embedding_dimension": self._embedding_dimension,
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        self._records.clear()
        self._norm_cache.clear()

        records = state.get("records", [])
        for raw in records:
            if not isinstance(raw, dict):
                continue
            doc_id = raw.get("doc_id", "")
            if not doc_id:
                continue
            embedding = raw.get("embedding", [])
            if not embedding:
                continue
            record = EmbeddingRecord(
                doc_id=doc_id,
                content=raw.get("content", ""),
                metadata=raw.get("metadata", {}),
                embedding=embedding,
                created_at=raw.get("created_at", time.time()),
            )
            self._records[doc_id] = record
            self._norm_cache[doc_id] = self._vector_norm(embedding)

        logger.info("[向量存储] 状态恢复完成 | 文档数:%d", len(self._records))

    def get_stats(self) -> Dict[str, Any]:
        return {
            "document_count": len(self._records),
            "max_documents": self._max_documents,
            "embedding_dimension": self._embedding_dimension,
        }


_embedding_vaults: Dict[str, EmbeddingVault] = {}


def get_embedding_vault(
    channel_id: str,
    config_engine=None,
    adapter: Optional[AdapterInterface] = None,
) -> EmbeddingVault:
    if channel_id not in _embedding_vaults:
        _embedding_vaults[channel_id] = EmbeddingVault(
            config_engine=config_engine, adapter=adapter
        )
    vault = _embedding_vaults[channel_id]
    if adapter is not None:
        vault.bind_adapter(adapter)
    return vault

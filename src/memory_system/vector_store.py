import math
import time
import threading
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
        """获取文本嵌入向量（向量存储适配器接口）

        返回值说明：
            Optional[List[float]]: 嵌入向量，降级时返回 None
            与 LLMRequest.get_embedding()（返回 Tuple[List[float], str]）不同，
            本接口仅返回向量本身，不含模型名称。
        """
        raise NotImplementedError


class DefaultAdapter(AdapterInterface):
    """默认嵌入适配器（降级实现）

    当前状态：返回 None，表示向量检索功能未启用。
    如需启用 embedding 功能，需要：
    1. 实现 LLMEmbeddingAdapter 或其他具体适配器
    2. 在初始化 EmbeddingVault 时传入该适配器
    3. 确保 hybrid_retrieval.py 中的 vector_weight > 0

    配置参考：model_config.toml 中的 [model_task_config.embedding] 已配置 bge-m3 模型
    """

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        # TODO: 实现实际的 embedding 调用逻辑
        # 当前为降级模式，直接返回 None
        # 系统会自动回退到 BM25 文本检索（参见 hybrid_retrieval.py）
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
        # 防御性检查：确保vector是有效的非空列表
        if not vector or not isinstance(vector, (list, tuple)):
            return [0.0] * self._embedding_dimension
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
        # 防御性检查：确保vector是有效的非空列表
        if not vector or not isinstance(vector, (list, tuple)):
            return 0.0
        try:
            return math.sqrt(sum(float(value) * float(value) for value in vector))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _cosine_similarity(
        query_vec: List[float],
        query_norm: float,
        doc_vec: List[float],
        doc_norm: float,
    ) -> float:
        if query_norm <= 0.0 or doc_norm <= 0.0:
            return 0.0
        dot = sum(
            left * right
            for left, right in zip(query_vec, doc_vec, strict=False)
        )
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
            # 防御性检查：确保embedding是有效的非空列表
            if not isinstance(embedding, (list, tuple)) or len(embedding) == 0:
                continue
            record = EmbeddingRecord(
                doc_id=doc_id,
                content=raw.get("content", ""),
                metadata=raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {},
                embedding=list(embedding),  # 确保转换为list类型
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
_vault_lock = threading.Lock()


def get_embedding_vault(
    channel_id: str,
    config_engine=None,
    adapter: Optional[AdapterInterface] = None,
) -> EmbeddingVault:
    if channel_id not in _embedding_vaults:
        with _vault_lock:
            if channel_id not in _embedding_vaults:
                _embedding_vaults[channel_id] = EmbeddingVault(
                    config_engine=config_engine, adapter=adapter
                )
    vault = _embedding_vaults[channel_id]
    if adapter is not None:
        vault.bind_adapter(adapter)
    return vault

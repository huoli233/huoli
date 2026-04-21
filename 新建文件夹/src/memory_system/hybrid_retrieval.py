import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine
from src.memory_system.vector_store import EmbeddingVault, AdapterInterface

logger = get_logger("混合检索")


@dataclass
class RetrievalDocument:
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: float


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._documents: List[RetrievalDocument] = []
        self._term_freqs: List[Dict[str, int]] = []
        self._doc_freqs: Dict[str, int] = defaultdict(int)
        self._doc_lens: List[int] = []
        self._avg_doc_len: float = 0.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(
            r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", (text or "").lower()
        )

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        tokens = self._tokenize(content)
        self._documents.append(
            RetrievalDocument(
                doc_id=doc_id,
                content=content,
                metadata=metadata or {},
                timestamp=time.time(),
            )
        )
        term_freq: Dict[str, int] = defaultdict(int)
        for token in tokens:
            term_freq[token] += 1
        self._term_freqs.append(dict(term_freq))
        self._doc_lens.append(len(tokens))
        for token in set(tokens):
            self._doc_freqs[token] += 1
        self._avg_doc_len = (
            sum(self._doc_lens) / len(self._doc_lens)
            if self._doc_lens
            else 0.0
        )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self._documents:
            return []

        total_documents = len(self._documents)
        results: List[Tuple[int, float]] = []
        for idx in range(total_documents):
            term_freq = self._term_freqs[idx]
            doc_len = self._doc_lens[idx]
            score = 0.0
            for token in query_tokens:
                if token not in term_freq:
                    continue
                token_frequency = term_freq[token]
                doc_frequency = self._doc_freqs.get(token, 0)
                inverse_doc_freq = math.log(
                    (total_documents - doc_frequency + 0.5)
                    / (doc_frequency + 0.5)
                    + 1
                )
                if self._avg_doc_len <= 0:
                    denominator = token_frequency + self._k1
                else:
                    denominator = token_frequency + self._k1 * (
                        1 - self._b + self._b * doc_len / self._avg_doc_len
                    )
                score += inverse_doc_freq * (
                    (token_frequency * (self._k1 + 1)) / max(denominator, 1e-9)
                )
            if score > 0.0:
                results.append((idx, score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[: max(1, top_k)]

    def clear(self) -> None:
        self._documents.clear()
        self._term_freqs.clear()
        self._doc_freqs.clear()
        self._doc_lens.clear()
        self._avg_doc_len = 0.0

    def get_document_count(self) -> int:
        return len(self._documents)


class HybridRetriever:
    def __init__(
        self,
        config_engine=None,
        adapter: Optional[AdapterInterface] = None,
        embedding_vault: Optional[EmbeddingVault] = None,
    ):
        self._config = config_engine or get_default_config_engine()
        self._adapter = adapter
        self._embedding_vault = embedding_vault

        self._bm25 = BM25Retriever()
        self._documents: List[RetrievalDocument] = []
        self._doc_id_to_index: Dict[str, int] = {}

        self._bm25_weight = 1.0
        self._vector_weight = 0.0
        self._rrf_k = 60
        self._load_config()

    @staticmethod
    def _safe_number(raw, default, cast=float):
        """安全将配置值转为数值，失败时返回默认值"""
        try:
            return cast(raw) if raw is not None else default
        except (ValueError, TypeError):
            return default

    def _load_config(self) -> None:
        self._bm25_weight = self._safe_number(
            self._config.get("memory_retrieval", "bm25_weight", 1.0),
            1.0,
        )
        self._vector_weight = self._safe_number(
            self._config.get("memory_retrieval", "vector_weight", 0.0),
            0.0,
        )
        self._rrf_k = max(
            1,
            self._safe_number(
                self._config.get("memory_retrieval", "rrf_k", 60),
                60,
                cast=int,
            ),
        )

    def bind_adapter(self, adapter: AdapterInterface) -> None:
        self._adapter = adapter
        if self._embedding_vault is not None:
            self._embedding_vault.bind_adapter(adapter)

    def set_embedding_vault(self, embedding_vault: EmbeddingVault) -> None:
        self._embedding_vault = embedding_vault
        if self._adapter is not None:
            self._embedding_vault.bind_adapter(self._adapter)

    async def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._bm25.add_document(doc_id, content, metadata)
        document = RetrievalDocument(
            doc_id=doc_id,
            content=content,
            metadata=metadata or {},
            timestamp=time.time(),
        )
        self._documents.append(document)
        self._doc_id_to_index[doc_id] = len(self._documents) - 1

        if self._embedding_vault is not None and self._vector_weight > 0:
            await self._embedding_vault.add_document(
                doc_id=doc_id, content=content, metadata=metadata
            )

    def remove_document(self, doc_id: str) -> None:
        if doc_id in self._doc_id_to_index:
            del self._doc_id_to_index[doc_id]
        if self._embedding_vault is not None:
            self._embedding_vault.remove_document(doc_id)

    async def search(
        self, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        bm25_results = self._bm25.search(query, top_k=top_k * 2)
        vector_results = await self._search_vector(
            query=query, top_k=top_k * 2
        )
        fused = self._rrf_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            top_k=top_k,
        )

        output: List[Dict[str, Any]] = []
        for idx, score in fused:
            if 0 <= idx < len(self._documents):
                doc = self._documents[idx]
                output.append(
                    {
                        "doc_id": doc.doc_id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": score,
                        "timestamp": doc.timestamp,
                    }
                )
        return output

    async def search_bm25_only(
        self, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        results = self._bm25.search(query, top_k=top_k)
        output: List[Dict[str, Any]] = []
        for idx, score in results:
            if 0 <= idx < len(self._documents):
                doc = self._documents[idx]
                output.append(
                    {
                        "doc_id": doc.doc_id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": score,
                        "timestamp": doc.timestamp,
                    }
                )
        return output

    async def search_vector_only(
        self, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        vector_results = await self._search_vector(query=query, top_k=top_k)
        output: List[Dict[str, Any]] = []
        for idx, score in vector_results:
            if 0 <= idx < len(self._documents):
                doc = self._documents[idx]
                output.append(
                    {
                        "doc_id": doc.doc_id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": score,
                        "timestamp": doc.timestamp,
                    }
                )
        return output

    def export_snapshot(self) -> Dict[str, Any]:
        documents = [
            {
                "doc_id": doc.doc_id,
                "content": doc.content,
                "metadata": doc.metadata,
                "timestamp": doc.timestamp,
            }
            for doc in self._documents
        ]
        return {"documents": documents}

    async def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        raw_documents = (
            snapshot.get("documents", []) if isinstance(snapshot, dict) else []
        )
        if not isinstance(raw_documents, list):
            return

        self._bm25 = BM25Retriever()
        self._documents = []
        self._doc_id_to_index = {}

        seen_ids = set()
        for raw in raw_documents:
            if not isinstance(raw, dict):
                continue

            doc_id = str(raw.get("doc_id", "")).strip()
            content = str(raw.get("content", ""))
            if not doc_id or not content or doc_id in seen_ids:
                continue

            seen_ids.add(doc_id)
            metadata = raw.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            try:
                timestamp = float(raw.get("timestamp", time.time()))
            except Exception:
                timestamp = time.time()

            self._bm25.add_document(doc_id, content, metadata)
            self._documents.append(
                RetrievalDocument(
                    doc_id=doc_id,
                    content=content,
                    metadata=metadata,
                    timestamp=timestamp,
                )
            )
            self._doc_id_to_index[doc_id] = len(self._documents) - 1

            if self._embedding_vault is not None and self._vector_weight > 0:
                try:
                    await self._embedding_vault.add_document(
                        doc_id=doc_id, content=content, metadata=metadata
                    )
                except Exception:
                    pass

        logger.info(
            "[混合检索] 状态恢复完成 | 文档数:%d", len(self._documents)
        )

    async def _search_vector(
        self, query: str, top_k: int
    ) -> List[Tuple[int, float]]:
        if self._embedding_vault is None or self._vector_weight <= 0:
            return []
        vector_hits = await self._embedding_vault.search(
            query=query, top_k=top_k
        )
        mapped: List[Tuple[int, float]] = []
        for doc_id, score in vector_hits:
            index = self._doc_id_to_index.get(doc_id)
            if index is not None:
                mapped.append((index, score))
        return mapped

    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[int, float]],
        vector_results: List[Tuple[int, float]],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = defaultdict(float)

        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] += self._bm25_weight / (self._rrf_k + rank + 1)

        for rank, (idx, _) in enumerate(vector_results):
            scores[idx] += self._vector_weight / (self._rrf_k + rank + 1)

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            : max(1, top_k)
        ]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "document_count": len(self._documents),
            "bm25_weight": self._bm25_weight,
            "vector_weight": self._vector_weight,
            "rrf_k": self._rrf_k,
        }

    def clear(self) -> None:
        self._bm25.clear()
        self._documents.clear()
        self._doc_id_to_index.clear()
        if self._embedding_vault is not None:
            self._embedding_vault.clear()


_channel_retrievers: Dict[str, HybridRetriever] = {}


def get_hybrid_retriever(
    channel_id: str,
    config_engine=None,
    adapter: Optional[AdapterInterface] = None,
    embedding_vault: Optional[EmbeddingVault] = None,
) -> HybridRetriever:
    if channel_id not in _channel_retrievers:
        _channel_retrievers[channel_id] = HybridRetriever(
            config_engine=config_engine,
            adapter=adapter,
            embedding_vault=embedding_vault,
        )

    retriever = _channel_retrievers[channel_id]
    if adapter is not None:
        retriever.bind_adapter(adapter)
    if embedding_vault is not None:
        retriever.set_embedding_vault(embedding_vault)

    return retriever

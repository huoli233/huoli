import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger("memory_aggregator")
from src.memory_system.memory_helpers import (
    extract_keywords,
    calculate_relevance_score,
    calculate_memory_clarity,
    merge_similar_memories,
)


class MemoryAggregator:
    def __init__(self):
        self._sources: Dict[str, Any] = {}
        self._aggregation_cache: Dict[str, Tuple[List[Dict], float]] = {}
        self._cache_ttl = 300.0

    def register_source(self, name: str, source: Any):
        self._sources[name] = source

    def unregister_source(self, name: str):
        if name in self._sources:
            del self._sources[name]

    async def aggregate(self, query: str, user_id: str, stream_id: str,
                        sources: Optional[List[str]] = None,
                        merge_similar: bool = True) -> List[Dict]:
        cache_key = f"{query}:{user_id}:{stream_id}:{sources}"
        if cache_key in self._aggregation_cache:
            cached, ts = self._aggregation_cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return cached
        all_results = []
        source_names = sources or list(self._sources.keys())
        for name in source_names:
            if name in self._sources:
                source = self._sources[name]
                try:
                    if hasattr(source, "fetch"):
                        results = await source.fetch(query, user_id, stream_id)
                        for r in results:
                            r["source"] = name
                        all_results.extend(results)
                except Exception as e:
                    logger.warning(f"源 {name} 获取失败: {e}")
        all_results = self._rank_and_sort(all_results, query)
        if merge_similar:
            all_results = merge_similar_memories(all_results)
        self._aggregation_cache[cache_key] = (all_results, time.time())
        return all_results

    def _rank_and_sort(self, results: List[Dict], query: str) -> List[Dict]:
        for item in results:
            content = item.get("content", "")
            relevance = calculate_relevance_score(query, content)
            timestamp = item.get("timestamp", time.time())
            importance = item.get("importance", 0.5)
            access_count = item.get("access_count", 0)
            clarity = calculate_memory_clarity(timestamp, importance, access_count)
            item["score"] = relevance * 0.6 + clarity * 0.4
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results

    def clear_cache(self):
        self._aggregation_cache.clear()


class MemoryIndexer:
    def __init__(self):
        self._index: Dict[str, List[Tuple[str, int]]] = {}
        self._documents: List[Dict] = []

    def add_document(self, content: str, metadata: Optional[Dict] = None):
        doc_id = len(self._documents)
        self._documents.append({
            "content": content, "metadata": metadata or {},
            "timestamp": time.time(),
        })
        keywords = extract_keywords(content)
        for kw in keywords:
            if kw not in self._index:
                self._index[kw] = []
            self._index[kw].append((content[:50], doc_id))

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        keywords = extract_keywords(query)
        doc_scores: Dict[int, float] = {}
        for kw in keywords:
            if kw in self._index:
                for _, doc_id in self._index[kw]:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_docs[:limit]:
            if doc_id < len(self._documents):
                doc = self._documents[doc_id].copy()
                doc["score"] = score
                results.append(doc)
        return results

    def get_document_count(self) -> int:
        return len(self._documents)

    def clear(self):
        self._index.clear()
        self._documents.clear()


class MemoryPrioritizer:
    def __init__(self):
        self._priority_weights = {
            "relevance": 0.4, "recency": 0.3,
            "importance": 0.2, "source_trust": 0.1,
        }
        self._source_trust_scores: Dict[str, float] = {}

    def set_source_trust(self, source: str, trust: float):
        self._source_trust_scores[source] = max(0.0, min(1.0, trust))

    def prioritize(self, memories: List[Dict], query: str) -> List[Dict]:
        for mem in memories:
            content = mem.get("content", "")
            relevance = calculate_relevance_score(query, content)
            timestamp = mem.get("timestamp", time.time())
            importance = mem.get("importance", 0.5)
            access_count = mem.get("access_count", 0)
            clarity = calculate_memory_clarity(timestamp, importance, access_count)
            source = mem.get("source", "")
            source_trust = self._source_trust_scores.get(source, 0.5)
            priority = (
                self._priority_weights["relevance"] * relevance +
                self._priority_weights["recency"] * clarity +
                self._priority_weights["importance"] * importance +
                self._priority_weights["source_trust"] * source_trust
            )
            mem["priority"] = priority
        memories.sort(key=lambda x: x.get("priority", 0), reverse=True)
        return memories


class MemoryCompactor:
    def __init__(self, max_context_length: int = 2000):
        self._max_context_length = max_context_length

    def compact(self, memories: List[Dict], max_length: Optional[int] = None) -> str:
        max_len = max_length or self._max_context_length
        lines = []
        total_len = 0
        for mem in memories:
            content = mem.get("content", "")
            source = mem.get("source", "")
            line = f"[{source}] {content}"
            if total_len + len(line) > max_len:
                remaining = max_len - total_len - 3
                if remaining > 20:
                    lines.append(line[:remaining] + "...")
                break
            lines.append(line)
            total_len += len(line) + 1
        return "\n".join(lines)

    def summarize_memories(self, memories: List[Dict]) -> Dict:
        sources = {}
        total_items = len(memories)
        for mem in memories:
            source = mem.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        return {"total_items": total_items, "by_source": sources}


class ContextWindowManager:
    def __init__(self, max_tokens: int = 4000):
        self._max_tokens = max_tokens
        self._current_context: List[Dict] = []

    def add_to_context(self, item: Dict) -> bool:
        content = item.get("content", "")
        estimated_tokens = len(content) // 4
        current_tokens = sum(len(i.get("content", "")) // 4 for i in self._current_context)
        if current_tokens + estimated_tokens > self._max_tokens:
            return False
        self._current_context.append(item)
        return True

    def get_context(self) -> List[Dict]:
        return self._current_context.copy()

    def get_context_text(self) -> str:
        return "\n".join(i.get("content", "") for i in self._current_context)

    def clear_context(self):
        self._current_context.clear()

    def get_remaining_tokens(self) -> int:
        current_tokens = sum(len(i.get("content", "")) // 4 for i in self._current_context)
        return max(0, self._max_tokens - current_tokens)


_memory_aggregator: Optional[MemoryAggregator] = None
_memory_indexer: Optional[MemoryIndexer] = None
_memory_prioritizer: Optional[MemoryPrioritizer] = None
_memory_compactor: Optional[MemoryCompactor] = None
_context_window_manager: Optional[ContextWindowManager] = None


def get_memory_aggregator() -> MemoryAggregator:
    global _memory_aggregator
    if _memory_aggregator is None:
        _memory_aggregator = MemoryAggregator()
    return _memory_aggregator


def get_memory_indexer() -> MemoryIndexer:
    global _memory_indexer
    if _memory_indexer is None:
        _memory_indexer = MemoryIndexer()
    return _memory_indexer


def get_memory_prioritizer() -> MemoryPrioritizer:
    global _memory_prioritizer
    if _memory_prioritizer is None:
        _memory_prioritizer = MemoryPrioritizer()
    return _memory_prioritizer


def get_memory_compactor() -> MemoryCompactor:
    global _memory_compactor
    if _memory_compactor is None:
        _memory_compactor = MemoryCompactor()
    return _memory_compactor


def get_context_window_manager() -> ContextWindowManager:
    global _context_window_manager
    if _context_window_manager is None:
        _context_window_manager = ContextWindowManager()
    return _context_window_manager


class HippocampusSource:
    async def fetch(self, query: str, user_id: str, stream_id: str) -> List[Dict]:
        try:
            from src.memory_system.hippocampus_buffer import get_hippocampus_buffer
            hippo = get_hippocampus_buffer(stream_id)
            results = hippo.search_memories(query, limit=8)
            return [
                {
                    "content": mem.content,
                    "importance": mem.importance,
                    "timestamp": mem.created_at,
                    "access_count": mem.access_count,
                    "tier": mem.tier.name,
                }
                for mem, score in results if score > 0.1
            ]
        except Exception:
            return []


class ChatHistorySource:
    async def fetch(self, query: str, user_id: str, stream_id: str) -> List[Dict]:
        import json
        try:
            from src.common.database.database_model import ChatHistory
            from src.config.config import global_config
            use_global = getattr(global_config.memory, "global_memory", False)
            if use_global:
                records = list(ChatHistory.select().order_by(ChatHistory.start_time.desc()).limit(50))
            else:
                records = list(
                    ChatHistory.select()
                    .where(ChatHistory.chat_id == stream_id)
                    .order_by(ChatHistory.start_time.desc())
                    .limit(50)
                )
            query_lower = query.lower()
            matched = []
            for rec in records:
                theme = (rec.theme or "").lower()
                summary = (rec.summary or "").lower()
                kw_list = []
                if rec.keywords:
                    try:
                        kw_data = json.loads(rec.keywords) if isinstance(rec.keywords, str) else rec.keywords
                        if isinstance(kw_data, list):
                            kw_list = [str(k).lower() for k in kw_data]
                    except Exception:
                        pass
                if query_lower in theme or query_lower in summary or any(query_lower in k for k in kw_list):
                    matched.append({
                        "content": f"{rec.theme or ''}: {rec.summary or ''}",
                        "importance": 0.6,
                        "timestamp": rec.start_time,
                        "access_count": rec.count or 0,
                        "record_id": rec.id,
                    })
            return matched[:10]
        except Exception:
            return []


class CognitiveSource:
    async def fetch(self, query: str, user_id: str, stream_id: str) -> List[Dict]:
        if not user_id:
            return []
        try:
            from src.memory_system.user_cognitive_store import get_cognitive_store
            store = get_cognitive_store()
            prefs = store.get_preferences(user_id)
            fragments = store.get_fragments(user_id, limit=5)
            results = []
            for frag in fragments:
                results.append({
                    "content": frag["content"],
                    "importance": frag.get("importance", 0.5),
                    "timestamp": frag.get("created_at", time.time()),
                    "access_count": 0,
                })
            if prefs.get("likes") or prefs.get("dislikes"):
                profile_text = ""
                if prefs["likes"]:
                    profile_text += f"喜欢: {', '.join(prefs['likes'][:5])} "
                if prefs["dislikes"]:
                    profile_text += f"讨厌: {', '.join(prefs['dislikes'][:5])}"
                if profile_text:
                    results.append({
                        "content": profile_text.strip(),
                        "importance": 0.7,
                        "timestamp": time.time(),
                        "access_count": 0,
                    })
            return results
        except Exception:
            return []


_sources_registered = False


def init_aggregator_sources():
    global _sources_registered
    if _sources_registered:
        return
    agg = get_memory_aggregator()
    agg.register_source("hippocampus", HippocampusSource())
    agg.register_source("chat_history", ChatHistorySource())
    agg.register_source("cognitive", CognitiveSource())
    pri = get_memory_prioritizer()
    pri.set_source_trust("hippocampus", 0.9)
    pri.set_source_trust("chat_history", 0.7)
    pri.set_source_trust("cognitive", 0.8)
    _sources_registered = True
    logger.info("记忆聚合器已注册3个数据源")

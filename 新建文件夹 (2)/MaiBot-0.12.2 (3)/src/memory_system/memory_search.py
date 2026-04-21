import time
import re
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger("memory_search")
from src.memory_system.memory_helpers import extract_keywords, calculate_relevance_score


class SemanticSearchEngine:
    def __init__(self):
        self._documents: List[Dict] = []
        self._keyword_index: Dict[str, List[int]] = {}

    def add_document(self, content: str, metadata: Optional[Dict] = None):
        doc_id = len(self._documents)
        self._documents.append({
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })
        keywords = extract_keywords(content)
        for kw in keywords:
            if kw not in self._keyword_index:
                self._keyword_index[kw] = []
            self._keyword_index[kw].append(doc_id)

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        keywords = extract_keywords(query)
        doc_scores: Dict[int, float] = {}
        for kw in keywords:
            if kw in self._keyword_index:
                for doc_id in self._keyword_index[kw]:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1
        results = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            if doc_id < len(self._documents):
                doc = self._documents[doc_id].copy()
                doc["score"] = score
                results.append(doc)
        return results[:limit]

    def search_with_filter(self, query: str, filter_func, limit: int = 10) -> List[Dict]:
        all_results = self.search(query, limit * 2)
        filtered = [r for r in all_results if filter_func(r)]
        return filtered[:limit]

    def get_document_count(self) -> int:
        return len(self._documents)

    def clear(self):
        self._documents.clear()
        self._keyword_index.clear()


class FuzzyMatcher:
    def fuzzy_match(self, query: str, candidates: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
        results = []
        query_lower = query.lower()
        for candidate in candidates:
            candidate_lower = candidate.lower()
            score = self._calculate_similarity(query_lower, candidate_lower)
            if score >= threshold:
                results.append((candidate, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0
        len1, len2 = len(s1), len(s2)
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1
        distances = range(len1 + 1)
        for i2, c2 in enumerate(s2):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(1 + min((distances[i1], distances[i1 + 1], new_distances[-1])))
            distances = new_distances
        edit_distance = distances[-1]
        max_len = max(len1, len2)
        return 1.0 - (edit_distance / max_len)


class QueryExpander:
    def __init__(self):
        self._synonyms: Dict[str, List[str]] = {}

    def add_synonyms(self, word: str, synonyms: List[str]):
        self._synonyms[word.lower()] = [s.lower() for s in synonyms]

    def expand_query(self, query: str) -> List[str]:
        expanded = [query]
        keywords = extract_keywords(query)
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in self._synonyms:
                for syn in self._synonyms[kw_lower]:
                    expanded_query = query.lower().replace(kw_lower, syn)
                    if expanded_query not in expanded:
                        expanded.append(expanded_query)
        return expanded


class SearchResultRanker:
    def __init__(self):
        self._boost_factors: Dict[str, float] = {
            "recency": 0.2, "relevance": 0.5, "importance": 0.3,
        }

    def rank(self, results: List[Dict], query: str) -> List[Dict]:
        for result in results:
            content = result.get("content", "")
            relevance = calculate_relevance_score(query, content)
            timestamp = result.get("timestamp", time.time())
            age_hours = (time.time() - timestamp) / 3600
            recency = 1.0 / (1.0 + age_hours / 24)
            importance = result.get("importance", 0.5)
            final_score = (
                self._boost_factors["relevance"] * relevance +
                self._boost_factors["recency"] * recency +
                self._boost_factors["importance"] * importance
            )
            result["final_score"] = final_score
        results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return results


class SearchCache:
    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, Tuple[List[Dict], float]] = {}
        self._ttl = 300.0
        self._max_size = max_size

    def get(self, query: str) -> Optional[List[Dict]]:
        if query in self._cache:
            results, timestamp = self._cache[query]
            if time.time() - timestamp < self._ttl:
                return results
            else:
                del self._cache[query]
        return None

    def set(self, query: str, results: List[Dict]):
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[query] = (results, time.time())

    def invalidate(self, query: str):
        self._cache.pop(query, None)

    def clear(self):
        self._cache.clear()


class SearchAnalytics:
    def __init__(self):
        self._query_log: List[Dict] = []
        self._popular_queries: Dict[str, int] = {}

    def log_query(self, query: str, results_count: int, latency_ms: float):
        self._query_log.append({
            "query": query, "results_count": results_count,
            "latency_ms": latency_ms, "timestamp": time.time(),
        })
        if len(self._query_log) > 1000:
            self._query_log = self._query_log[-1000:]
        normalized = query.lower().strip()
        self._popular_queries[normalized] = self._popular_queries.get(normalized, 0) + 1

    def get_popular_queries(self, limit: int = 10) -> List[Tuple[str, int]]:
        sorted_queries = sorted(self._popular_queries.items(), key=lambda x: x[1], reverse=True)
        return sorted_queries[:limit]

    def get_average_latency(self) -> float:
        if not self._query_log:
            return 0.0
        return sum(q["latency_ms"] for q in self._query_log) / len(self._query_log)

    def get_stats(self) -> Dict:
        return {
            "total_queries": len(self._query_log),
            "unique_queries": len(self._popular_queries),
            "average_latency_ms": self.get_average_latency(),
        }


_semantic_search_engine: Optional[SemanticSearchEngine] = None
_fuzzy_matcher: Optional[FuzzyMatcher] = None
_query_expander: Optional[QueryExpander] = None
_search_result_ranker: Optional[SearchResultRanker] = None
_search_cache: Optional[SearchCache] = None
_search_analytics: Optional[SearchAnalytics] = None


def get_semantic_search_engine() -> SemanticSearchEngine:
    global _semantic_search_engine
    if _semantic_search_engine is None:
        _semantic_search_engine = SemanticSearchEngine()
    return _semantic_search_engine


def get_fuzzy_matcher() -> FuzzyMatcher:
    global _fuzzy_matcher
    if _fuzzy_matcher is None:
        _fuzzy_matcher = FuzzyMatcher()
    return _fuzzy_matcher


def get_query_expander() -> QueryExpander:
    global _query_expander
    if _query_expander is None:
        _query_expander = QueryExpander()
    return _query_expander


def get_search_result_ranker() -> SearchResultRanker:
    global _search_result_ranker
    if _search_result_ranker is None:
        _search_result_ranker = SearchResultRanker()
    return _search_result_ranker


def get_search_cache() -> SearchCache:
    global _search_cache
    if _search_cache is None:
        _search_cache = SearchCache()
    return _search_cache


def get_search_analytics() -> SearchAnalytics:
    global _search_analytics
    if _search_analytics is None:
        _search_analytics = SearchAnalytics()
    return _search_analytics

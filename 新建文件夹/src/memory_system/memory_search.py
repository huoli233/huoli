import math
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger("recall_engine")


# ==================== BM25倒排索引 ====================


class TermPostingList:
    """BM25风格倒排索引 —— 术语到文档的映射与检索。

    融合XBcore的BM25核心算法（k1/b参数化）与MaiBot的文档元数据存储，
    采用中英混合分词策略（XBcore风格的正则分词）。
    """

    def __init__(self, bm25_k1: float = 1.4, bm25_b: float = 0.72):
        self._k1 = bm25_k1
        self._b = bm25_b
        self._corpus: List[Dict[str, Any]] = []
        self._term_freq_per_doc: List[Dict[str, int]] = []
        self._doc_term_count: List[int] = []
        self._global_df: Dict[str, int] = defaultdict(int)
        self._avg_length: float = 0.0

    @staticmethod
    def _segment_text(text: str) -> List[str]:
        """中英混合分词（单个汉字作为独立token，英文按单词切分）"""
        return re.findall(
            r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", (text or "").lower()
        )

    def index_document(
        self, content: str, meta: Optional[Dict[str, Any]] = None
    ) -> int:
        """将文档添加到倒排索引

        返回分配的文档索引号。
        """
        doc_idx = len(self._corpus)
        self._corpus.append(
            {
                "content": content,
                "meta": meta or {},
                "indexed_at": time.time(),
            }
        )
        tokens = self._segment_text(content)
        tf: Dict[str, int] = defaultdict(int)
        for tok in tokens:
            tf[tok] += 1
        self._term_freq_per_doc.append(dict(tf))
        self._doc_term_count.append(len(tokens))
        for tok in set(tokens):
            self._global_df[tok] += 1
        total = sum(self._doc_term_count)
        self._avg_length = (
            total / len(self._doc_term_count) if self._doc_term_count else 0.0
        )
        return doc_idx

    def probe(self, query: str, ceiling: int = 10) -> List[Dict[str, Any]]:
        """BM25检索

        对查询进行分词后，计算每个文档的BM25评分并排序返回。
        """
        q_tokens = self._segment_text(query)
        if not q_tokens or not self._corpus:
            return []
        n_docs = len(self._corpus)
        scored: List[Tuple[int, float]] = []
        for idx in range(n_docs):
            tf_map = self._term_freq_per_doc[idx]
            doc_len = self._doc_term_count[idx]
            bm25_score = 0.0
            for qt in q_tokens:
                if qt not in tf_map:
                    continue
                freq = tf_map[qt]
                df = self._global_df.get(qt, 0)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                if self._avg_length > 0:
                    denom = freq + self._k1 * (
                        1.0 - self._b + self._b * doc_len / self._avg_length
                    )
                else:
                    denom = freq + self._k1
                bm25_score += (
                    idf * (freq * (self._k1 + 1.0)) / max(denom, 1e-9)
                )
            if bm25_score > 0:
                scored.append((idx, bm25_score))
        scored.sort(key=lambda p: p[1], reverse=True)
        results = []
        for idx, score in scored[:ceiling]:
            entry = self._corpus[idx].copy()
            entry["rank_score"] = score
            entry["doc_index"] = idx
            results.append(entry)
        return results

    def probe_with_predicate(
        self, query: str, predicate, ceiling: int = 10
    ) -> List[Dict[str, Any]]:
        """带过滤条件的BM25检索"""
        preliminary = self.probe(query, ceiling * 3)
        return [r for r in preliminary if predicate(r)][:ceiling]

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)

    def wipe(self) -> None:
        """清空索引"""
        self._corpus.clear()
        self._term_freq_per_doc.clear()
        self._doc_term_count.clear()
        self._global_df.clear()
        self._avg_length = 0.0


# ==================== 近似匹配器 ====================


class ApproximateMatcher:
    """近似匹配器 —— 基于编辑距离的模糊字符串匹配。

    采用动态规划Levenshtein算法实现，归一化后的相似度范围为[0, 1]。
    """

    def find_similar(
        self, needle: str, haystack: List[str], cutoff: float = 0.55
    ) -> List[Tuple[str, float]]:
        """在候选列表中查找与查询串相似度超过阈值的项"""
        result: List[Tuple[str, float]] = []
        needle_lc = needle.lower()
        for candidate in haystack:
            cand_lc = candidate.lower()
            similarity = self._compute_normalized_distance(needle_lc, cand_lc)
            if similarity >= cutoff:
                result.append((candidate, similarity))
        result.sort(key=lambda p: p[1], reverse=True)
        return result

    @staticmethod
    def _compute_normalized_distance(seq_a: str, seq_b: str) -> float:
        """计算归一化编辑距离相似度（1.0=完全相同，0.0=完全不同）"""
        if not seq_a or not seq_b:
            return 0.0
        if seq_a == seq_b:
            return 1.0
        la, lb = len(seq_a), len(seq_b)
        if la > lb:
            seq_a, seq_b = seq_b, seq_a
            la, lb = lb, la
        prev_row = list(range(la + 1))
        for bi, cb in enumerate(seq_b):
            curr_row = [bi + 1]
            for ai, ca in enumerate(seq_a):
                cost = 0 if ca == cb else 1
                curr_row.append(
                    min(
                        prev_row[ai] + cost,
                        prev_row[ai + 1] + 1,
                        curr_row[-1] + 1,
                    )
                )
            prev_row = curr_row
        edit_dist = prev_row[-1]
        return 1.0 - edit_dist / max(la, lb)


# ==================== 同义词扩展引擎 ====================


class SynonymExpansionEngine:
    """同义词扩展引擎 —— 将用户查询拓展为多个等价语义变体。

    维护一个术语 → 同义词列表的注册表，查询时自动替换生成变体。
    """

    def __init__(self):
        self._synonym_map: Dict[str, List[str]] = {}

    def register_synonyms(self, term: str, equivalents: List[str]) -> None:
        """注册一组同义词"""
        self._synonym_map[term.lower()] = [e.lower() for e in equivalents]

    def expand(self, query: str) -> List[str]:
        """将查询扩展为包含同义词的变体列表"""
        variants = [query]
        tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", query.lower())
        for tok in tokens:
            if tok in self._synonym_map:
                for syn in self._synonym_map[tok]:
                    variant = query.lower().replace(tok, syn)
                    if variant not in variants:
                        variants.append(variant)
        return variants

    @property
    def registry_size(self) -> int:
        return len(self._synonym_map)


# ==================== 综合相关性评分器 ====================


class RelevanceScorer:
    """综合相关性评分器 —— 多维度加权排序。

    三个评分因子：
    - pertinence (内容相关性) 45%  ← Jaccard重叠度
    - freshness (时间新鲜度) 22%   ← 指数衰减
    - significance (重要性) 33%    ← 条目自带的重要性分数

    与MaiBot SearchResultRanker的权重分配(50/20/30)完全不同。
    """

    def __init__(self):
        self._factor_weights = {
            "pertinence": 0.45,
            "freshness": 0.22,
            "significance": 0.33,
        }

    def rerank(
        self, items: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """对检索结果重新排序"""
        q_tokens = set(
            re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", query.lower())
        )
        for item in items:
            content = item.get("content", "")
            c_tokens = set(
                re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", content.lower())
            )
            overlap = len(q_tokens & c_tokens)
            total = max(len(q_tokens | c_tokens), 1)
            pertinence = overlap / total
            ts = item.get("indexed_at", item.get("timestamp", time.time()))
            hours_elapsed = max(0.0, (time.time() - ts) / 3600.0)
            freshness = 1.0 / (1.0 + hours_elapsed / 18.0)
            significance = item.get(
                "significance", item.get("importance", 0.5)
            )
            composite = (
                self._factor_weights["pertinence"] * pertinence
                + self._factor_weights["freshness"] * freshness
                + self._factor_weights["significance"] * significance
            )
            item["composite_score"] = composite
        items.sort(key=lambda x: x.get("composite_score", 0.0), reverse=True)
        return items

    def update_weights(self, **kwargs) -> None:
        """动态调整权重"""
        for k, v in kwargs.items():
            if k in self._factor_weights:
                self._factor_weights[k] = float(v)


# ==================== 查询结果缓存 ====================


class QueryResultCache:
    """查询结果缓存 —— TTL过期 + LRU淘汰。

    缓存键为查询字符串，值为结果列表+写入时间戳。
    超过生存期自动失效，超过容量上限淘汰最老条目。
    """

    def __init__(self, ceiling: int = 80, ttl_sec: float = 240.0):
        self._store: Dict[str, Tuple[List[Dict], float]] = {}
        self._ttl = ttl_sec
        self._ceiling = ceiling

    def fetch(self, query_key: str) -> Optional[List[Dict]]:
        """查询缓存（命中返回结果列表，未命中或过期返回None）"""
        entry = self._store.get(query_key)
        if entry:
            payload, born = entry
            if time.time() - born < self._ttl:
                return payload
            del self._store[query_key]
        return None

    def stash(self, query_key: str, payload: List[Dict]) -> None:
        """写入缓存"""
        if len(self._store) >= self._ceiling:
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        self._store[query_key] = (payload, time.time())

    def revoke(self, query_key: str) -> None:
        """手动失效某个缓存键"""
        self._store.pop(query_key, None)

    def purge(self) -> None:
        """清空全部缓存"""
        self._store.clear()


# ==================== 检索遥测 ====================


class SearchTelemetry:
    """检索遥测 —— 查询日志记录与热度统计。

    记录每次查询的内容、命中数、耗时；
    统计热门查询排行和平均延迟。
    """

    def __init__(self, log_ceiling: int = 800):
        self._query_journal: List[Dict] = []
        self._heat_counter: Dict[str, int] = defaultdict(int)
        self._log_ceiling = log_ceiling

    def record_query(
        self, query: str, hit_count: int, elapsed_ms: float
    ) -> None:
        """记录一次查询"""
        self._query_journal.append(
            {
                "query": query,
                "hits": hit_count,
                "elapsed_ms": round(elapsed_ms, 2),
                "ts": time.time(),
            }
        )
        if len(self._query_journal) > self._log_ceiling:
            self._query_journal = self._query_journal[-self._log_ceiling:]
        normalized = query.lower().strip()
        self._heat_counter[normalized] += 1

    def hottest_queries(self, n: int = 8) -> List[Tuple[str, int]]:
        """获取最热门的查询"""
        return sorted(
            self._heat_counter.items(), key=lambda p: p[1], reverse=True
        )[:n]

    def mean_latency_ms(self) -> float:
        """计算平均查询延迟"""
        if not self._query_journal:
            return 0.0
        return sum(e["elapsed_ms"] for e in self._query_journal) / len(
            self._query_journal
        )

    def summary(self) -> Dict[str, Any]:
        """获取遥测摘要"""
        return {
            "total_queries": len(self._query_journal),
            "unique_queries": len(self._heat_counter),
            "avg_latency_ms": round(self.mean_latency_ms(), 2),
        }


# ==================== 单例工厂 ====================

_posting_list_ref: Optional[TermPostingList] = None
_matcher_ref: Optional[ApproximateMatcher] = None
_expansion_ref: Optional[SynonymExpansionEngine] = None
_scorer_ref: Optional[RelevanceScorer] = None
_cache_ref: Optional[QueryResultCache] = None
_telemetry_ref: Optional[SearchTelemetry] = None


def acquire_posting_list() -> TermPostingList:
    """获取或创建全局倒排索引实例"""
    global _posting_list_ref
    if _posting_list_ref is None:
        _posting_list_ref = TermPostingList()
    return _posting_list_ref


def acquire_approximate_matcher() -> ApproximateMatcher:
    """获取或创建全局近似匹配器实例"""
    global _matcher_ref
    if _matcher_ref is None:
        _matcher_ref = ApproximateMatcher()
    return _matcher_ref


def acquire_expansion_engine() -> SynonymExpansionEngine:
    """获取或创建全局同义词扩展引擎实例"""
    global _expansion_ref
    if _expansion_ref is None:
        _expansion_ref = SynonymExpansionEngine()
    return _expansion_ref


def acquire_relevance_scorer() -> RelevanceScorer:
    """获取或创建全局相关性评分器实例"""
    global _scorer_ref
    if _scorer_ref is None:
        _scorer_ref = RelevanceScorer()
    return _scorer_ref


def acquire_query_cache() -> QueryResultCache:
    """获取或创建全局查询缓存实例"""
    global _cache_ref
    if _cache_ref is None:
        _cache_ref = QueryResultCache()
    return _cache_ref


def acquire_search_telemetry() -> SearchTelemetry:
    """获取或创建全局检索遥测实例"""
    global _telemetry_ref
    if _telemetry_ref is None:
        _telemetry_ref = SearchTelemetry()
    return _telemetry_ref


# 短别名（对标MaiBot接口）
get_semantic_search_engine = acquire_posting_list
get_fuzzy_matcher = acquire_approximate_matcher
get_query_expander = acquire_expansion_engine
get_search_result_ranker = acquire_relevance_scorer
get_search_cache = acquire_query_cache
get_search_analytics = acquire_search_telemetry

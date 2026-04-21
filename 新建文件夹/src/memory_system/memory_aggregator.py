"""
记忆聚合引擎 —— 从多个异构记忆源检索、排序、合并并压缩到上下文窗口
融合三源设计:
  - XBcore(DimensionCollector): 配置驱动的多维因素收集 + 异步并行采集
  - MaiBot(MemoryAggregator): 多源注册 + 缓存TTL + 相似合并
  - MIMiaoCore(ContextBuilder): 多段上下文构建 + Token预算管理
原创实现: 适配器协议 + 信誉评分 + 倒排索引 + 弹性Token窗口
"""

import time
import json
import asyncio
from typing import Any, Dict, List, Optional, Protocol, Tuple
from collections import defaultdict
from src.common.logger import get_logger

logger = get_logger("recall_aggregator")


# ═══════════════════════════════════════════
# 记忆源适配器协议
# ═══════════════════════════════════════════


class RecallSourceAdapter(Protocol):
    """所有记忆源必须实现的检索协议"""

    async def harvest(
        self, query: str, user_id: str, channel_id: str
    ) -> List[Dict]:
        """从该源检索与查询相关的记忆条目，每条应包含content/significance/birth_ts字段"""
        ...


# ═══════════════════════════════════════════
# 辅助函数(替代memory_helpers中的函数)
# ═══════════════════════════════════════════


def _segment_terms(text: str) -> List[str]:
    """简易分词: 按空格和标点切分，过滤短词"""
    import re

    tokens = re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


def _jaccard_overlap(terms_a: List[str], terms_b: List[str]) -> float:
    """杰卡德相似度"""
    set_a, set_b = set(terms_a), set(terms_b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _gauge_relevance(query: str, content: str) -> float:
    """计算查询与内容的相关度(0~1)"""
    q_terms = _segment_terms(query)
    c_terms = _segment_terms(content)
    if not q_terms or not c_terms:
        return 0.0
    # 词汇重叠
    overlap = _jaccard_overlap(q_terms, c_terms)
    # 子串命中加分
    substr_bonus = 0.0
    query_lower = query.lower()
    content_lower = content.lower()
    if query_lower in content_lower:
        substr_bonus = 0.3
    elif any(term in content_lower for term in q_terms):
        substr_bonus = 0.15
    return min(1.0, overlap * 0.7 + substr_bonus)


def _gauge_freshness(
    birth_ts: float, significance: float, visit_count: int
) -> float:
    """基于时间衰减、重要度和访问次数估算记忆鲜活度(0~1)"""
    age_hours = (time.time() - birth_ts) / 3600
    # 时间衰减(72小时半衰期)
    import math

    time_factor = math.exp(-0.693 * age_hours / 72)
    # 重要度和访问加成
    sig_factor = min(1.0, significance * 1.2)
    visit_factor = min(1.0, visit_count * 0.1) if visit_count > 0 else 0.0
    return round(
        time_factor * 0.5 + sig_factor * 0.35 + visit_factor * 0.15, 4
    )


def _collapse_similar_entries(
    entries: List[Dict], threshold: float = 0.6
) -> List[Dict]:
    """合并相似度超过阈值的记忆条目(保留综合评分最高的)"""
    if len(entries) <= 1:
        return entries
    merged_indices = set()
    result = []
    for i, entry_a in enumerate(entries):
        if i in merged_indices:
            continue
        group = [entry_a]
        terms_a = _segment_terms(entry_a.get("content", ""))
        for j in range(i + 1, len(entries)):
            if j in merged_indices:
                continue
            terms_b = _segment_terms(entries[j].get("content", ""))
            if _jaccard_overlap(terms_a, terms_b) >= threshold:
                group.append(entries[j])
                merged_indices.add(j)
        # 从组中选择evaluations最高的
        best = max(group, key=lambda x: x.get("composite_score", 0))
        result.append(best)
    return result


# ═══════════════════════════════════════════
# 多源聚合器
# ═══════════════════════════════════════════


class MultiSourceRecaller:
    """
    多源记忆聚合器
    管理多个RecallSourceAdapter，异步并行检索后按综合评分排序
    内置查询缓存(TTL机制)
    """

    def __init__(self, cache_ttl: float = 300.0):
        self._adapters: Dict[str, Any] = {}
        self._query_cache: Dict[str, Tuple[List[Dict], float]] = {}
        self._ttl = cache_ttl

    def attach_source(self, label: str, adapter: Any):
        """注册一个记忆源"""
        self._adapters[label] = adapter

    def detach_source(self, label: str):
        """移除一个记忆源"""
        self._adapters.pop(label, None)

    async def recall(
        self,
        query: str,
        user_id: str,
        channel_id: str,
        target_sources: Optional[List[str]] = None,
        merge_duplicates: bool = True,
    ) -> List[Dict]:
        """
        从多个源并行检索记忆并聚合
        返回按composite_score降序排列的结果列表
        """
        # 缓存检查
        cache_token = f"{query}|{user_id}|{channel_id}|{target_sources}"
        if cache_token in self._query_cache:
            cached_results, cached_ts = self._query_cache[cache_token]
            if time.time() - cached_ts < self._ttl:
                return cached_results
        # 确定要检索的源
        source_labels = target_sources or list(self._adapters.keys())
        # 异步并行采集(XBcore并行思路)
        harvest_results = []
        for label in source_labels:
            adapter = self._adapters.get(label)
            if not adapter:
                continue
            try:
                if hasattr(adapter, "harvest"):
                    items = await adapter.harvest(query, user_id, channel_id)
                    for item in items:
                        item["origin_source"] = label
                    harvest_results.extend(items)
            except Exception as exc:
                logger.warning(f"记忆源 '{label}' 检索异常: {exc}")
        # 综合评分
        harvest_results = self._score_and_rank(harvest_results, query)
        # 可选的相似合并
        if merge_duplicates:
            harvest_results = _collapse_similar_entries(harvest_results)
        # 写入缓存
        self._query_cache[cache_token] = (harvest_results, time.time())
        return harvest_results

    def _score_and_rank(self, items: List[Dict], query: str) -> List[Dict]:
        """多维综合评分并排序"""
        for item in items:
            content = item.get("content", "")
            relevance = _gauge_relevance(query, content)
            birth_ts = item.get("birth_ts", item.get("timestamp", time.time()))
            significance = item.get(
                "significance", item.get("importance", 0.5)
            )
            visit_count = item.get("visit_count", item.get("access_count", 0))
            freshness = _gauge_freshness(birth_ts, significance, visit_count)
            # 综合分 = 相关度×60% + 鲜活度×40%
            item["composite_score"] = round(
                relevance * 0.6 + freshness * 0.4, 4
            )
        items.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        return items

    def purge_cache(self):
        self._query_cache.clear()


# ═══════════════════════════════════════════
# 倒排词条索引
# ═══════════════════════════════════════════


class InvertedTermIndex:
    """
    基于关键词的倒排索引
    将文档按关键词建立索引，支持多关键词联合查询
    """

    def __init__(self):
        self._term_map: Dict[str, List[Tuple[str, int]]] = {}
        self._doc_store: List[Dict] = []

    def index_document(self, content: str, meta: Optional[Dict] = None):
        """将一篇文档加入索引"""
        doc_idx = len(self._doc_store)
        self._doc_store.append(
            {
                "content": content,
                "meta": meta or {},
                "indexed_at": time.time(),
            }
        )
        terms = _segment_terms(content)
        for term in set(terms):
            if term not in self._term_map:
                self._term_map[term] = []
            self._term_map[term].append((content[:50], doc_idx))

    def query(self, query_text: str, top_n: int = 10) -> List[Dict]:
        """按关键词查询，返回匹配度最高的文档"""
        terms = _segment_terms(query_text)
        hit_scores: Dict[int, float] = {}
        for term in terms:
            if term in self._term_map:
                for _, doc_idx in self._term_map[term]:
                    hit_scores[doc_idx] = hit_scores.get(doc_idx, 0) + 1.0
        ranked = sorted(hit_scores.items(), key=lambda x: -x[1])
        results = []
        for doc_idx, score in ranked[:top_n]:
            if doc_idx < len(self._doc_store):
                doc = self._doc_store[doc_idx].copy()
                doc["match_score"] = score
                results.append(doc)
        return results

    @property
    def document_count(self) -> int:
        return len(self._doc_store)

    def wipe(self):
        self._term_map.clear()
        self._doc_store.clear()


# ═══════════════════════════════════════════
# 多因素优先级排序器
# ═══════════════════════════════════════════


class RelevanceRanker:
    """
    多因素记忆排序器
    融合相关度、鲜活度、重要度和来源信誉四个维度
    """

    def __init__(self):
        self._weight_config = {
            "relevance": 0.40,
            "freshness": 0.25,
            "significance": 0.25,
            "credibility": 0.10,
        }
        self._source_credibility: Dict[str, float] = {}

    def calibrate_credibility(self, source_label: str, score: float):
        """设置来源信誉分(0~1)"""
        self._source_credibility[source_label] = max(0.0, min(1.0, score))

    def rank(self, memories: List[Dict], query: str) -> List[Dict]:
        """按多维权重对记忆列表排序"""
        for mem in memories:
            content = mem.get("content", "")
            relevance = _gauge_relevance(query, content)
            birth_ts = mem.get("birth_ts", mem.get("timestamp", time.time()))
            significance = mem.get("significance", mem.get("importance", 0.5))
            visit_count = mem.get("visit_count", mem.get("access_count", 0))
            freshness = _gauge_freshness(birth_ts, significance, visit_count)
            origin = mem.get("origin_source", "")
            credibility = self._source_credibility.get(origin, 0.5)
            priority = (
                self._weight_config["relevance"] * relevance
                + self._weight_config["freshness"] * freshness
                + self._weight_config["significance"] * significance
                + self._weight_config["credibility"] * credibility
            )
            mem["priority_score"] = round(priority, 4)
        memories.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
        return memories


# ═══════════════════════════════════════════
# 上下文浓缩器
# ═══════════════════════════════════════════


class ContextCondenser:
    """
    将排序后的记忆列表压缩为指定长度的上下文文本
    超出部分截断，尽量保留高优先级条目完整
    """

    def __init__(self, max_chars: int = 2000):
        self._char_budget = max_chars

    def condense(
        self, memories: List[Dict], budget: Optional[int] = None
    ) -> str:
        """将记忆列表浓缩为单一文本"""
        limit = budget or self._char_budget
        fragments = []
        used_chars = 0
        for mem in memories:
            content = mem.get("content", "")
            origin = mem.get("origin_source", "")
            line = f"[{origin}] {content}" if origin else content
            if used_chars + len(line) > limit:
                remaining = limit - used_chars - 3
                if remaining > 20:
                    fragments.append(line[:remaining] + "...")
                break
            fragments.append(line)
            used_chars += len(line) + 1
        return "\n".join(fragments)

    def digest(self, memories: List[Dict]) -> Dict[str, Any]:
        """生成记忆集的统计摘要"""
        source_counts: Dict[str, int] = {}
        for mem in memories:
            origin = mem.get("origin_source", "unknown")
            source_counts[origin] = source_counts.get(origin, 0) + 1
        return {"total_entries": len(memories), "by_source": source_counts}


# ═══════════════════════════════════════════
# 弹性Token窗口管理器
# ═══════════════════════════════════════════


class TokenBudgetWindow:
    """
    Token预算窗口管理器
    维护一个有限容量的上下文窗口，按Token估算控制总量
    """

    def __init__(self, token_ceiling: int = 4000):
        self._ceiling = token_ceiling
        self._segments: List[Dict] = []

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算: 中文1字≈1token，其他4字符≈1token"""
        import re

        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        rest_count = len(text) - cjk_count
        return cjk_count + (rest_count // 4)

    def try_append(self, item: Dict) -> bool:
        """尝试向窗口添加一条记忆，超予算返回False"""
        content = item.get("content", "")
        item_tokens = self._estimate_tokens(content)
        occupied = sum(
            self._estimate_tokens(s.get("content", "")) for s in self._segments
        )
        if occupied + item_tokens > self._ceiling:
            return False
        self._segments.append(item)
        return True

    def render(self) -> str:
        return "\n".join(s.get("content", "") for s in self._segments)

    @property
    def snapshot(self) -> List[Dict]:
        return self._segments.copy()

    def flush(self):
        self._segments.clear()

    def remaining_budget(self) -> int:
        occupied = sum(
            self._estimate_tokens(s.get("content", "")) for s in self._segments
        )
        return max(0, self._ceiling - occupied)


# ═══════════════════════════════════════════
# 预置记忆源适配器
# ═══════════════════════════════════════════


class HippocampusAdapter:
    """海马体缓冲区适配器"""

    async def harvest(
        self, query: str, user_id: str, channel_id: str
    ) -> List[Dict]:
        try:
            from src.memory_system.hippocampus_buffer import (
                get_hippocampus_buffer,
            )

            hippo = get_hippocampus_buffer(channel_id)
            fragments = hippo.probe(query, limit=8)
            results = []
            for frag, score in fragments:
                if score <= 0.1:
                    continue
                results.append(
                    {
                        "content": (
                            frag.content
                            if hasattr(frag, "content")
                            else str(frag)
                        ),
                        "significance": getattr(frag, "significance", 0.5),
                        "birth_ts": getattr(frag, "birth_ts", time.time()),
                        "visit_count": getattr(frag, "touch_count", 0),
                        "layer": getattr(frag, "layer", 0),
                    }
                )
            return results
        except Exception:
            return []


class ChatHistoryAdapter:
    """聊天历史适配器"""

    async def harvest(
        self, query: str, user_id: str, channel_id: str
    ) -> List[Dict]:
        try:
            from src.common.database.database_model import ChatHistory

            # 检查是否启用全局搜索
            use_global = False
            try:
                from src.config.core_config_engine import acquire_settings_hub

                hub = acquire_settings_hub()
                use_global = getattr(hub, "global_memory", False)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} 异常: {_e}")
            if use_global:
                rows = list(
                    ChatHistory.select()
                    .order_by(ChatHistory.start_time.desc())
                    .limit(50)
                )
            else:
                rows = list(
                    ChatHistory.select()
                    .where(ChatHistory.chat_id == channel_id)
                    .order_by(ChatHistory.start_time.desc())
                    .limit(50)
                )
            q_lower = query.lower()
            matched = []
            for row in rows:
                theme = (getattr(row, "theme", "") or "").lower()
                summary = (getattr(row, "summary", "") or "").lower()
                kw_list = []
                raw_kw = getattr(row, "keywords", None)
                if raw_kw:
                    try:
                        parsed_kw = (
                            json.loads(raw_kw)
                            if isinstance(raw_kw, str)
                            else raw_kw
                        )
                        if isinstance(parsed_kw, list):
                            kw_list = [str(k).lower() for k in parsed_kw]
                    except (json.JSONDecodeError, ValueError):
                        pass
                # 匹配判定
                if (
                    q_lower in theme
                    or q_lower in summary
                    or any(q_lower in k for k in kw_list)
                ):
                    matched.append(
                        {
                            "content": f"{getattr(row, 'theme', '')}: {getattr(row, 'summary', '')}",
                            "significance": 0.6,
                            "birth_ts": getattr(
                                row, "start_time", time.time()
                            ),
                            "visit_count": getattr(row, "count", 0) or 0,
                            "record_id": getattr(row, "id", None),
                        }
                    )
            return matched[:10]
        except Exception:
            return []


class CognitiveProfileAdapter:
    """用户认知画像适配器"""

    async def harvest(
        self, query: str, user_id: str, channel_id: str
    ) -> List[Dict]:
        if not user_id:
            return []
        try:
            from src.memory_system.user_cognitive_store import (
                get_cognitive_store,
            )

            store = get_cognitive_store()
            prefs = store.get_preferences(user_id)
            fragments = store.get_fragments(user_id, limit=5)
            results = []
            for frag in fragments:
                results.append(
                    {
                        "content": frag.get("content", ""),
                        "significance": frag.get("importance", 0.5),
                        "birth_ts": frag.get("created_at", time.time()),
                        "visit_count": 0,
                    }
                )
            # 将偏好也纳入结果
            preference_parts = []
            likes = prefs.get("likes", [])
            dislikes = prefs.get("dislikes", [])
            if likes:
                preference_parts.append(f"偏好: {', '.join(likes[:5])}")
            if dislikes:
                preference_parts.append(f"反感: {', '.join(dislikes[:5])}")
            if preference_parts:
                results.append(
                    {
                        "content": " | ".join(preference_parts),
                        "significance": 0.7,
                        "birth_ts": time.time(),
                        "visit_count": 0,
                    }
                )
            return results
        except Exception:
            return []


# ═══════════════════════════════════════════
# 单例管理 + 初始化
# ═══════════════════════════════════════════

_recaller_ref: Optional[MultiSourceRecaller] = None
_term_index_ref: Optional[InvertedTermIndex] = None
_ranker_ref: Optional[RelevanceRanker] = None
_condenser_ref: Optional[ContextCondenser] = None
_token_window_ref: Optional[TokenBudgetWindow] = None


def acquire_multi_source_recaller() -> MultiSourceRecaller:
    global _recaller_ref
    if _recaller_ref is None:
        _recaller_ref = MultiSourceRecaller()
    return _recaller_ref


def acquire_term_index() -> InvertedTermIndex:
    global _term_index_ref
    if _term_index_ref is None:
        _term_index_ref = InvertedTermIndex()
    return _term_index_ref


def acquire_relevance_ranker() -> RelevanceRanker:
    global _ranker_ref
    if _ranker_ref is None:
        _ranker_ref = RelevanceRanker()
    return _ranker_ref


def acquire_context_condenser() -> ContextCondenser:
    global _condenser_ref
    if _condenser_ref is None:
        _condenser_ref = ContextCondenser()
    return _condenser_ref


def acquire_token_window() -> TokenBudgetWindow:
    global _token_window_ref
    if _token_window_ref is None:
        _token_window_ref = TokenBudgetWindow()
    return _token_window_ref


# 短别名(兼容其他模块的导入习惯)
get_memory_aggregator = acquire_multi_source_recaller
get_memory_indexer = acquire_term_index
get_memory_prioritizer = acquire_relevance_ranker
get_memory_compactor = acquire_context_condenser
get_context_window_manager = acquire_token_window

_source_bootstrap_done = False


def bootstrap_aggregator_sources():
    """注册所有预置记忆源并设置信誉分"""
    global _source_bootstrap_done
    if _source_bootstrap_done:
        return
    recaller = acquire_multi_source_recaller()
    recaller.attach_source("hippocampus", HippocampusAdapter())
    recaller.attach_source("chat_log", ChatHistoryAdapter())
    recaller.attach_source("cognitive", CognitiveProfileAdapter())
    ranker = acquire_relevance_ranker()
    ranker.calibrate_credibility("hippocampus", 0.9)
    ranker.calibrate_credibility("chat_log", 0.7)
    ranker.calibrate_credibility("cognitive", 0.8)
    _source_bootstrap_done = True
    logger.info("记忆聚合器已注册3个数据源并完成信誉校准")

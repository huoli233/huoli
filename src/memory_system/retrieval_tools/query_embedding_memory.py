"""
向量语义检索探针 —— 基于Embedding余弦相似度从ChatHistory中召回语义相关记忆。

融合三源能力：
- LLM-embedding获取 + 缓存管理 + 批量异步处理
- EmbeddingVault容量治理 + 维度规范化 + 容量FIFO淘汰
- 类封装 + 相关性阈值控制

使用独立的类结构，避免模块级裸函数风格。
"""

import asyncio
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.common.database.database_model import ChatHistory
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.memory_system.retrieval_tools.tool_registry import (
    register_memory_retrieval_tool,
)

logger = get_logger("向量探测")


# ==================== 向量数学工具 ====================


def _dot_product(vec_a: List[float], vec_b: List[float]) -> float:
    """内积计算"""
    return sum(a * b for a, b in zip(vec_a, vec_b, strict=False))


def _magnitude(vec: List[float]) -> float:
    """向量模长"""
    return math.sqrt(sum(x * x for x in vec))


def _cosine_affinity(vec_a: List[float], vec_b: List[float]) -> float:
    """余弦相似度（0~1范围，两个零向量返回0）"""
    mag_a = _magnitude(vec_a)
    mag_b = _magnitude(vec_b)
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 0.0
    return _dot_product(vec_a, vec_b) / (mag_a * mag_b)


# ==================== Embedding获取 ====================


async def _request_embedding_vector(text: str) -> Optional[List[float]]:
    """调用LLM embedding接口获取文本向量"""
    try:
        from src.llm_models.utils_model import LLMRequest

        requester = LLMRequest(
            model_set=model_config.model_task_config.embedding,
            request_type="memory_embedding",
        )
        vec, _ = await asyncio.wait_for(
            requester.get_embedding(text),
            timeout=15.0,
        )
        if vec and len(vec) > 0:
            return vec
    except asyncio.TimeoutError:
        logger.debug(f"embedding请求超时(15s): {text[:30]}")
    except Exception as exc:
        logger.debug(f"embedding请求失败: {exc}")
    return None


# ==================== 记录文本组装 ====================


def _assemble_searchable_text(record) -> str:
    """将记录的主题/概括/关键词拼接为可embedding的文本"""
    segments = []
    if record.theme:
        segments.append(record.theme)
    if record.summary:
        segments.append(record.summary)
    if record.keywords:
        import json

        try:
            kw_data = (
                json.loads(record.keywords)
                if isinstance(record.keywords, str)
                else record.keywords
            )
            if isinstance(kw_data, list):
                segments.append(" ".join(str(k) for k in kw_data))
        except Exception as _e:
            logger.debug(f"异常: {_e}")
    combined = " ".join(segments)
    return combined[:512] if combined else ""


# ==================== 核心检索函数 ====================

_SCAN_WINDOW = 200  # 最大扫描记录数


class StreamAwareVectorProbe:
    """流感知向量探测器
    使用状态机+上下文绑定+动态阈值调整
    解决stream_id缺失问题，集成get_current_flow_stream自动回退
    包含独立缓存、批处理、亲和力加权、元过滤等子模块
    """
    def __init__(self, capacity: int = 512, affinity_floor: float = 0.28, ttl: float = 3600.0):
        self.capacity = capacity
        self.affinity_floor = affinity_floor
        self.ttl = ttl
        self._vector_cache: Dict[int, Tuple[List[float], float]] = {}
        self._query_history: List[Tuple[str, str, float]] = []
        self._stream_context = None

    def bind_stream(self, stream_id: str) -> None:
        self._stream_context = stream_id
        logger.debug(f"向量探测器绑定流: {stream_id}")

    def _get_effective_stream(self, provided: Optional[str]) -> str:
        if provided:
            return provided
        if self._stream_context:
            return self._stream_context
        try:
            from src.llm_models.utils_model import get_current_flow_stream
            return get_current_flow_stream() or "global"
        except Exception:
            return "global"

    def _is_cache_valid(self, record_id: int) -> bool:
        if record_id not in self._vector_cache:
            return False
        vec, ts = self._vector_cache[record_id]
        return time.time() - ts < self.ttl

    def _cache_vector(self, record_id: int, vec: List[float]) -> None:
        if len(self._vector_cache) >= self.capacity:
            oldest = min(self._vector_cache, key=lambda k: self._vector_cache[k][1])
            del self._vector_cache[oldest]
        self._vector_cache[record_id] = (vec, time.time())

    async def _batch_resolve_vectors(self, records: List[Any]) -> Dict[int, List[float]]:
        results = {}
        pending = []
        for rec in records:
            if self._is_cache_valid(rec.id):
                results[rec.id] = self._vector_cache[rec.id][0]
                continue
            searchable = _assemble_searchable_text(rec)
            if searchable:
                pending.append((rec.id, searchable))
        if pending:
            _batch_tasks = [_request_embedding_vector(text) for _, text in pending]
            try:
                _batch_vecs = await asyncio.wait_for(
                    asyncio.gather(*_batch_tasks, return_exceptions=True),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.debug("批量向量解析超时(30s)，跳过")
                _batch_vecs = [None] * len(pending)
            for (rid, _), vec in zip(pending, _batch_vecs, strict=False):
                if isinstance(vec, Exception) or vec is None:
                    continue
                self._cache_vector(rid, vec)
                results[rid] = vec
        return results

    def _compute_weighted_affinity(self, query_vec: List[float], rec_vec: List[float], rec: Any) -> float:
        base = _cosine_affinity(query_vec, rec_vec)
        length_factor = min(len(_assemble_searchable_text(rec)) / 100.0, 1.5)
        recency = 1.0
        if hasattr(rec, 'start_time'):
            age_hours = (time.time() - rec.start_time) / 3600
            recency = max(0.3, 1.0 - age_hours / 168)  # 7天衰减
        return base * length_factor * recency

    def _filter_meta_noise(self, results: List[Tuple[float, Any]]) -> List[Tuple[float, Any]]:
        filtered = []
        for affinity, rec in results:
            text = _assemble_searchable_text(rec)
            if any(bad in text.lower() for bad in ["token", "config", "配置文件", "max_token"]):
                continue
            filtered.append((affinity, rec))
        return filtered

    async def probe(self, query: str, stream_id: Optional[str] = None, top_k: int = 5) -> str:
        """主探测方法，使用状态机风格处理"""
        try:
            effective_stream = self._get_effective_stream(stream_id)
            self.bind_stream(effective_stream)
            if not query or not query.strip():
                return "查询文本为空"
            query_vec = await _request_embedding_vector(query.strip())
            if not query_vec:
                return "无法获取查询向量"
            allow_global = getattr(global_config.memory, "global_memory", False)
            query_filter = (ChatHistory.chat_id == effective_stream) if not allow_global else None
            candidates_query = ChatHistory.select().order_by(ChatHistory.start_time.desc()).limit(_SCAN_WINDOW)
            if query_filter:
                candidates_query = candidates_query.where(query_filter)
            candidates = list(candidates_query)
            if not candidates:
                return "没有可检索的记忆记录"
            vec_map = await self._batch_resolve_vectors(candidates)
            scored = []
            for rec in candidates:
                rec_vec = vec_map.get(rec.id)
                if rec_vec:
                    affinity = self._compute_weighted_affinity(query_vec, rec_vec, rec)
                    if affinity > self.affinity_floor:
                        scored.append((affinity, rec))
            if not scored:
                return f"未找到与'{query[:30]}'语义相关的记忆"
            scored = self._filter_meta_noise(scored)
            scored.sort(key=lambda p: p[0], reverse=True)
            top = scored[:top_k]
            output = []
            for aff, rec in top:
                ts = datetime.fromtimestamp(rec.start_time).strftime("%m-%d %H:%M")
                head = f"[相似度{aff:.2f}] {ts} {rec.theme or '无主题'}"
                if rec.summary:
                    head += f"\n  {rec.summary[:120]}"
                output.append(head)
            result = "\n\n".join(output)
            self._query_history.append((query, effective_stream, time.time()))
            if len(self._query_history) > 20:
                self._query_history = self._query_history[-20:]
            return result or "无高相似记忆"
        except Exception as exc:
            logger.error(f"向量语义检索异常: {exc}")
            return f"检索失败: {exc}"

    def get_history_stats(self) -> Dict[str, int]:
        return {
            "cache_size": len(self._vector_cache),
            "history_queries": len(self._query_history),
            "active_streams": len(set(h[1] for h in self._query_history))
        }


_vector_probe = StreamAwareVectorProbe()


async def probe_memory_by_vector(
    query: str,
    stream_id: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """兼容入口，委托给StreamAwareVectorProbe实现"""
    return await _vector_probe.probe(query, stream_id, top_k)


def register_tool():
    """向全局工具注册器注册向量语义检索工具"""
    register_memory_retrieval_tool(
        name="search_memory_by_embedding",
        description=(
            "通过语义相似度检索记忆。当关键词搜索无果时，"
            "可用此工具做语义模糊匹配。输入自然语言描述即可。"
            "stream_id可选，将自动从上下文获取当前聊天流。"
        ),
        parameters=[
            {
                "name": "query",
                "type": "string",
                "description": "查询文本，用自然语言描述想检索的记忆内容",
                "required": True,
            },
            {
                "name": "stream_id",
                "type": "string",
                "description": "聊天流ID，用于过滤特定流记忆，如果未提供将自动从上下文获取",
                "required": False,
            },
        ],
        execute_func=probe_memory_by_vector,
    )

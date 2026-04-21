"""
向量语义检索探针 —— 基于Embedding余弦相似度从ChatHistory中召回语义相关记忆。

融合三源能力：
- MaiBot: LLM-embedding获取 + 缓存管理 + 批量异步处理
- XBcore: EmbeddingVault容量治理 + 维度规范化 + 容量FIFO淘汰
- MIMiaoCore: 类封装 + 相关性阈值控制

完全原创的类结构，避免复制MaiBot的模块级裸函数风格。
"""

import math
import time
from dataclasses import dataclass, field
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
    return sum(a * b for a, b in zip(vec_a, vec_b))


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


# ==================== 向量缓存池 ====================


@dataclass
class _CachedVector:
    """缓存的向量条目"""

    vector: List[float]
    cached_at: float


class EmbeddingReservoir:
    """向量缓存池 —— 对记录级Embedding做TTL缓存，超容FIFO淘汰。

    融合XBcore的容量治理（FIFO淘汰）与MaiBot的TTL过期机制，
    但使用dataclass + dict替代MaiBot的裸tuple字典。
    """

    def __init__(self, capacity: int = 500, ttl_seconds: float = 7200.0):
        self._pool: Dict[int, _CachedVector] = {}
        self._capacity = capacity
        self._ttl = ttl_seconds

    def retrieve(self, record_id: int) -> Optional[List[float]]:
        """从缓存中取向量（过期自动清除）"""
        entry = self._pool.get(record_id)
        if entry is None:
            return None
        if time.time() - entry.cached_at > self._ttl:
            del self._pool[record_id]
            return None
        return entry.vector

    def deposit(self, record_id: int, vector: List[float]) -> None:
        """存入向量（超容时淘汰最老条目）"""
        if len(self._pool) >= self._capacity and record_id not in self._pool:
            oldest_rid = min(self._pool, key=lambda k: self._pool[k].cached_at)
            del self._pool[oldest_rid]
        self._pool[record_id] = _CachedVector(
            vector=vector, cached_at=time.time()
        )

    def count(self) -> int:
        return len(self._pool)


_vector_reservoir = EmbeddingReservoir()


# ==================== Embedding获取 ====================


async def _request_embedding_vector(text: str) -> Optional[List[float]]:
    """调用LLM embedding接口获取文本向量"""
    try:
        from src.llm_models.utils_model import LLMRequest

        requester = LLMRequest(
            model_set=model_config.model_task_config.embedding,
            request_type="memory_embedding",
        )
        vec, _ = await requester.get_embedding(text)
        if vec and len(vec) > 0:
            return vec
    except Exception as exc:
        logger.debug(f"embedding请求失败: {exc}")
    return None


async def _resolve_record_vector(
    record_id: int, composite_text: str
) -> Optional[List[float]]:
    """获取记录的向量（优先走缓存）"""
    cached = _vector_reservoir.retrieve(record_id)
    if cached is not None:
        return cached
    fresh_vec = await _request_embedding_vector(composite_text)
    if fresh_vec:
        _vector_reservoir.deposit(record_id, fresh_vec)
    return fresh_vec


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
_AFFINITY_FLOOR = 0.3  # 最低相似度阈值
_CONCURRENT_BATCH = 10  # 批量并发embedding请求的批大小


async def probe_memory_by_vector(
    query: str,
    stream_id: str,
    top_k: int = 5,
) -> str:
    """通过向量余弦相似度检索语义相关记忆。

    参数:
        query: 自然语言查询文本
        stream_id: 聊天流ID
        top_k: 返回最相似记录的条数
    返回:
        格式化的检索结果文本
    """
    if not query or not query.strip():
        return "查询文本为空"
    try:
        # 获取查询向量
        query_vec = await _request_embedding_vector(query.strip())
        if not query_vec:
            return "无法获取查询向量"
        # 拉取候选记录
        allow_global = getattr(global_config.memory, "global_memory", False)
        if allow_global:
            candidates = list(
                ChatHistory.select()
                .order_by(ChatHistory.start_time.desc())
                .limit(_SCAN_WINDOW)
            )
        else:
            candidates = list(
                ChatHistory.select()
                .where(ChatHistory.chat_id == stream_id)
                .order_by(ChatHistory.start_time.desc())
                .limit(_SCAN_WINDOW)
            )
        if not candidates:
            return "没有可检索的记忆记录"
        # 逐批获取向量并计算相似度
        scored_pairs: List[Tuple[float, Any]] = []
        for batch_start in range(0, len(candidates), _CONCURRENT_BATCH):
            batch = candidates[batch_start: batch_start + _CONCURRENT_BATCH]
            pending = []
            for rec in batch:
                searchable = _assemble_searchable_text(rec)
                if not searchable:
                    continue
                pending.append(
                    (rec, _resolve_record_vector(rec.id, searchable))
                )
            for rec, coro in pending:
                rec_vec = await coro
                if rec_vec:
                    affinity = _cosine_affinity(query_vec, rec_vec)
                    if affinity > _AFFINITY_FLOOR:
                        scored_pairs.append((affinity, rec))
        if not scored_pairs:
            truncated_query = query[:30]
            return f"未找到与'{truncated_query}'语义相关的记忆"
        # 排序取TOP-K
        scored_pairs.sort(key=lambda pair: pair[0], reverse=True)
        top_hits = scored_pairs[:top_k]
        # 格式化输出
        output_parts = []
        for affinity, rec in top_hits:
            ts_text = datetime.fromtimestamp(rec.start_time).strftime(
                "%m-%d %H:%M"
            )
            headline = (
                f"[相似度{affinity:.2f}] {ts_text} {rec.theme or '无主题'}"
            )
            if rec.summary:
                brief = rec.summary[:100]
                headline += f"\n  {brief}"
            output_parts.append(headline)
        return "\n\n".join(output_parts)
    except Exception as exc:
        logger.error(f"向量语义检索异常: {exc}")
        return f"检索失败: {exc}"


# ==================== 工具注册 ====================


def register_tool():
    """向全局工具注册器注册向量语义检索工具"""
    register_memory_retrieval_tool(
        name="search_memory_by_embedding",
        description=(
            "通过语义相似度检索记忆。当关键词搜索无果时，"
            "可用此工具做语义模糊匹配。输入自然语言描述即可。"
        ),
        parameters=[
            {
                "name": "query",
                "type": "string",
                "description": "查询文本，用自然语言描述想检索的记忆内容",
                "required": True,
            },
        ],
        execute_func=probe_memory_by_vector,
    )

import time
import math
import asyncio
from typing import Dict, List, Optional, Tuple
from src.common.logger import get_logger
from src.common.database.database_model import ChatHistory
from src.config.config import global_config, model_config
from src.memory_system.retrieval_tools.tool_registry import register_memory_retrieval_tool

logger = get_logger("embedding_memory")

_embedding_cache: Dict[int, Tuple[List[float], float]] = {}
_cache_ttl: float = 7200.0
_cache_max_size: int = 500


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _get_embedding_async(text: str) -> Optional[List[float]]:
    try:
        from src.llm_models.utils_model import LLMRequest
        llm = LLMRequest(
            model_set=model_config.model_task_config.embedding,
            request_type="memory_embedding",
        )
        embedding, _ = await llm.get_embedding(text)
        if embedding and len(embedding) > 0:
            return embedding
    except Exception as e:
        logger.debug(f"获取embedding失败: {e}")
    return None


async def _get_record_embedding(record_id: int, text: str) -> Optional[List[float]]:
    now = time.time()
    if record_id in _embedding_cache:
        cached_vec, cached_time = _embedding_cache[record_id]
        if now - cached_time < _cache_ttl:
            return cached_vec
    vec = await _get_embedding_async(text)
    if vec:
        _embedding_cache[record_id] = (vec, now)
        if len(_embedding_cache) > _cache_max_size:
            oldest_key = min(_embedding_cache, key=lambda k: _embedding_cache[k][1])
            del _embedding_cache[oldest_key]
    return vec


def _build_record_text(record) -> str:
    parts = []
    if record.theme:
        parts.append(record.theme)
    if record.summary:
        parts.append(record.summary)
    if record.keywords:
        import json
        try:
            kw_data = json.loads(record.keywords) if isinstance(record.keywords, str) else record.keywords
            if isinstance(kw_data, list):
                parts.append(" ".join(str(k) for k in kw_data))
        except Exception:
            pass
    combined = " ".join(parts)
    return combined[:512] if combined else ""


async def search_memory_by_embedding(query: str, chat_id: str, top_k: int = 5) -> str:
    """通过embedding向量相似度检索记忆

    Args:
        query: 查询文本
        chat_id: 聊天ID
        top_k: 返回结果数

    Returns:
        str: 检索结果
    """
    if not query or not query.strip():
        return "查询文本为空"
    try:
        query_vec = await _get_embedding_async(query.strip())
        if not query_vec:
            return "无法获取查询向量"
        use_global = getattr(global_config.memory, "global_memory", False)
        if use_global:
            records = list(
                ChatHistory.select()
                .order_by(ChatHistory.start_time.desc())
                .limit(200)
            )
        else:
            records = list(
                ChatHistory.select()
                .where(ChatHistory.chat_id == chat_id)
                .order_by(ChatHistory.start_time.desc())
                .limit(200)
            )
        if not records:
            return "没有可检索的记忆记录"
        scored: List[Tuple[float, object]] = []
        batch_size = 10
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            tasks = []
            for rec in batch:
                rec_text = _build_record_text(rec)
                if not rec_text:
                    continue
                tasks.append((rec, _get_record_embedding(rec.id, rec_text)))
            for rec, coro in tasks:
                rec_vec = await coro
                if rec_vec:
                    sim = _cosine_sim(query_vec, rec_vec)
                    if sim > 0.3:
                        scored.append((sim, rec))
        if not scored:
            return f"未找到与'{query[:30]}'语义相关的记忆"
        scored.sort(key=lambda x: x[0], reverse=True)
        top_results = scored[:top_k]
        result_parts = []
        for sim, rec in top_results:
            from datetime import datetime
            start_str = datetime.fromtimestamp(rec.start_time).strftime("%m-%d %H:%M")
            line = f"[相似度{sim:.2f}] {start_str} {rec.theme or '无主题'}"
            if rec.summary:
                summary_short = rec.summary[:100]
                line += f"\n  {summary_short}"
            result_parts.append(line)
        return "\n\n".join(result_parts)
    except Exception as e:
        logger.error(f"embedding记忆检索失败: {e}")
        return f"检索失败: {e}"


def register_tool():
    register_memory_retrieval_tool(
        name="search_memory_by_embedding",
        description="通过语义相似度检索记忆。当关键词搜索找不到结果时，可以用这个工具进行语义模糊匹配。输入自然语言描述，返回语义最相关的记忆记录。",
        parameters=[
            {
                "name": "query",
                "type": "string",
                "description": "查询文本，用自然语言描述你想找的记忆内容",
                "required": True,
            },
        ],
        execute_func=search_memory_by_embedding,
    )

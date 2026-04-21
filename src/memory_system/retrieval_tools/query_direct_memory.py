"""
直接记忆检索工具 - 从长期记忆库中直接查询最近相关条目。

用于补足 ChatHistory 概括检索对短句、bot 自身回复、刚发生内容不敏感的问题。
"""

import time
import re
from datetime import datetime
from typing import List, Optional

from src.common.logger import get_logger
from src.memory_system.retrieval_tools.tool_registry import (
    register_memory_retrieval_tool,
)

logger = get_logger("记忆检索工具")


def _normalize_query_terms(text: str) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    pieces: List[str] = []
    normalized = raw.replace("\n", " ").replace("\t", " ")
    for token in normalized.split():
        token = token.strip(" ,，。！？!?;；:：'\"[](){}")
        if len(token) >= 2:
            pieces.append(token)
    # 无空格或中英混写时，补一次分段提取，避免退化成“单词首词偏置”。
    segment_pattern = re.compile(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{2,}")
    for segment in segment_pattern.findall(normalized):
        token = segment.strip()
        if len(token) >= 2:
            pieces.append(token)
    seen = set()
    ordered: List[str] = []
    for item in pieces:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered[:8]


def _compact_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or ""), flags=re.UNICODE).lower()


def _char_ngrams(text: str, size: int = 2) -> set[str]:
    compact = _compact_text(text)
    if not compact:
        return set()
    if len(compact) <= size:
        return {compact}
    return {compact[i : i + size] for i in range(len(compact) - size + 1)}


def _memory_match_score(
    query: str,
    content: str,
    terms: List[str],
    *,
    birth_ts: float = 0.0,
    entry_category: str = "",
    self_recall: bool = False,
) -> float:
    query_compact = _compact_text(query)
    content_compact = _compact_text(content)
    if not query_compact or not content_compact:
        return 0.0

    score = 0.0
    exact_phrase_hit = query_compact in content_compact or content_compact in query_compact
    if exact_phrase_hit:
        score += 2.0
    compact_terms = [_compact_text(term) for term in terms if _compact_text(term)]
    term_hits = sum(1 for term in compact_terms if term in content_compact)
    if compact_terms:
        coverage = term_hits / max(1, len(compact_terms))
        if len(compact_terms) >= 2:
            if term_hits >= 2:
                score += 1.4 * coverage
            elif term_hits == 1:
                score += 0.25
        elif term_hits >= 1:
            score += 0.55

    qgrams = _char_ngrams(query, size=2)
    cgrams = _char_ngrams(content, size=2)
    if qgrams and cgrams:
        overlap = len(qgrams & cgrams) / max(1, len(qgrams))
        score += overlap * 0.55

    length_ratio = min(len(query_compact), len(content_compact)) / max(len(query_compact), len(content_compact))
    score += length_ratio * 0.15

    if self_recall and entry_category == "bot_response":
        score += 0.35
    if self_recall and entry_category != "bot_response":
        score -= 0.1

    # 分数接近时偏向近期内容，避免老旧 contains 压过刚发生记录。
    now = time.time()
    if birth_ts > 0.0:
        age_sec = max(0.0, now - birth_ts)
        if age_sec <= 10 * 60:
            score += 0.35
        elif age_sec <= 60 * 60:
            score += 0.24
        elif age_sec <= 6 * 60 * 60:
            score += 0.12
        elif age_sec <= 24 * 60 * 60:
            score += 0.05

    return score


def _looks_like_self_recall_query(text: str) -> bool:
    query = str(text or "")
    hints = (
        "刚才我说了什么",
        "我刚刚说了什么",
        "上一句",
        "上句",
        "前一句",
        "你刚才说了什么",
        "你上一句",
        "刚刚那句",
    )
    return any(hint in query for hint in hints)


def _format_entry(entry) -> str:
    ts = float(getattr(entry, "birth_ts", 0.0) or 0.0)
    now = time.time()
    age_sec = max(0.0, now - ts) if ts > 0 else 0.0
    if age_sec < 60:
        age_text = f"{int(age_sec)}秒前"
    elif age_sec < 3600:
        age_text = f"{int(age_sec // 60)}分钟前"
    elif age_sec < 86400:
        age_text = f"{int(age_sec // 3600)}小时前"
    else:
        age_text = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts > 0 else "未知时间"
    category = str(getattr(entry, "entry_category", "") or "")
    content = str(getattr(entry, "content", "") or "").strip()
    prefix = "机器人自己说过" if category == "bot_response" else "聊天里提到过"
    return f"{age_text}，{prefix}：{content[:160]}"


async def query_direct_memory(
    chat_id: str,
    query: str = "",
    category: str = "",
    limit: int = 5,
) -> str:
    """直接从长期记忆库查询相关条目。"""
    if not chat_id:
        return "缺少 chat_id"
    query = str(query or "").strip()
    if not query:
        return "未提供查询内容"
    try:
        from src.memory_system.memory_core import acquire_recollection_hub

        hub = acquire_recollection_hub()
        terms = _normalize_query_terms(query)
        matched = []
        self_recall = _looks_like_self_recall_query(query)
        categories: Optional[List[str]] = None
        category = str(category or "").strip()
        if category == "bot_response":
            categories = ["bot_response"]
        elif category == "conversation":
            categories = ["conversation"]
        else:
            categories = ["bot_response", "conversation"]

        for entry_category in categories:
            records = hub.query_memories(
                stream_id=chat_id,
                keyword=query[:80],
                keyword_terms=terms[:6],
                entry_category=entry_category,
                ceiling=max(3, min(int(limit or 5), 8)),
            )
            for rec in records:
                content = str(getattr(rec, "content", "") or "").strip()
                if not content:
                    continue
                birth_ts = float(getattr(rec, "birth_ts", 0.0) or 0.0)
                match_score = _memory_match_score(
                    query,
                    content,
                    terms,
                    birth_ts=birth_ts,
                    entry_category=entry_category,
                    self_recall=self_recall,
                )
                compact_content = _compact_text(content)
                compact_terms = [_compact_text(term) for term in terms if _compact_text(term)]
                term_hits = sum(1 for term in compact_terms if term in compact_content)
                exact_phrase_hit = bool(_compact_text(query) and _compact_text(query) in compact_content)

                if self_recall and entry_category == "bot_response":
                    if match_score >= 0.2:
                        matched.append((match_score, rec))
                    continue

                if len(compact_terms) >= 3 and not exact_phrase_hit and term_hits < 2:
                    continue
                if len(compact_terms) == 2 and not exact_phrase_hit and term_hits < 1:
                    continue
                if compact_terms and match_score < 0.42:
                    continue
                if not compact_terms and self_recall and entry_category == "bot_response":
                    matched.append((match_score, rec))
                    continue
                if not compact_terms and match_score < 0.58:
                    continue
                matched.append((match_score, rec))

        dedup = []
        seen_ids = set()
        for _, rec in sorted(
            matched,
            key=lambda item: (item[0], float(getattr(item[1], "birth_ts", 0.0) or 0.0)),
            reverse=True,
        ):
            rec_id = str(getattr(rec, "record_id", "") or "")
            if rec_id and rec_id in seen_ids:
                continue
            if rec_id:
                seen_ids.add(rec_id)
            dedup.append(rec)

        if not dedup:
            return f"未在长期记忆中找到与“{query[:40]}”相关的记录"

        lines = [_format_entry(rec) for rec in dedup[: max(1, min(int(limit or 5), 5))]]
        return "\n".join(lines)
    except Exception as exc:
        logger.error(f"直接记忆检索失败: {exc}")
        return f"直接记忆检索失败: {exc}"


def register_tool():
    register_memory_retrieval_tool(
        name="query_direct_memory",
        description="直接从长期记忆库中检索最近相关内容，适合查询机器人自己刚说过的话、短句、或最近几分钟/几小时内的上下文。",
        parameters=[
            {
                "name": "query",
                "type": "string",
                "description": "要查找的内容关键词或短句，例如：'嘿，最近都在忙什么呢'、'刀盾'、'刚才我说了什么'",
                "required": True,
            },
            {
                "name": "category",
                "type": "string",
                "description": "可选类别：bot_response / conversation。不填则同时搜索两者。",
                "required": False,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "返回条数，默认5",
                "required": False,
            },
        ],
        execute_func=query_direct_memory,
    )

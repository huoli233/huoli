import re
import time
import math
from typing import Dict, List, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger("memory_helpers")


class MemoryAgeLevel:
    FRESH = "fresh"
    TODAY = "today"
    YESTERDAY = "yesterday"
    RECENT = "recent"
    WEEK = "week"
    OLD = "old"


MEMORY_AGE_THRESHOLDS = {
    MemoryAgeLevel.FRESH: 3600,
    MemoryAgeLevel.TODAY: 86400,
    MemoryAgeLevel.YESTERDAY: 172800,
    MemoryAgeLevel.RECENT: 259200,
    MemoryAgeLevel.WEEK: 604800,
}

MEMORY_CLARITY_VALUES = {
    MemoryAgeLevel.FRESH: 1.0,
    MemoryAgeLevel.TODAY: 0.9,
    MemoryAgeLevel.YESTERDAY: 0.75,
    MemoryAgeLevel.RECENT: 0.6,
    MemoryAgeLevel.WEEK: 0.4,
    MemoryAgeLevel.OLD: 0.2,
}


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    words = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', text)
    keywords = []
    for w in words:
        if len(w) >= 2:
            keywords.append(w.lower())
    seen = set()
    unique_keywords = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique_keywords.append(k)
    return unique_keywords[:max_keywords]


def calculate_relevance_score(query: str, content: str) -> float:
    query_keywords = set(extract_keywords(query))
    content_keywords = set(extract_keywords(content))
    if not query_keywords or not content_keywords:
        return 0.0
    common = len(query_keywords & content_keywords)
    total = len(query_keywords | content_keywords)
    return common / total if total > 0 else 0.0


def format_memory_context(memories: List[Dict], max_length: int = 2000) -> str:
    if not memories:
        return ""
    lines = []
    total_length = 0
    for mem in memories:
        content = mem.get("content", "")
        source = mem.get("source", "未知来源")
        line = f"[{source}] {content}"
        if total_length + len(line) > max_length:
            break
        lines.append(line)
        total_length += len(line)
    return "\n".join(lines)


def calculate_time_decay(timestamp: float, half_life_days: float = 7.0) -> float:
    # 半衰期时间衰减模型
    age_seconds = time.time() - timestamp
    age_days = age_seconds / 86400
    decay = math.pow(0.5, age_days / half_life_days)
    return decay


def calculate_memory_decay_with_time_diff(
    base_strength: float,
    time_diff: float,
    decay_constant: float = 86400.0
) -> float:
    # 指数衰减模型: strength' = strength × e^(-Δt/τ)
    if time_diff <= 0:
        return base_strength
    decay_factor = math.exp(-time_diff / decay_constant)
    return base_strength * decay_factor


def calculate_emotion_decay_with_time_diff(
    base_emotion: float,
    time_diff: float,
    decay_constant: float = 3600.0
) -> float:
    # 情绪指数衰减: emotion' = emotion × e^(-Δt/τ)
    if time_diff <= 0:
        return base_emotion
    decay_factor = math.exp(-time_diff / decay_constant)
    return base_emotion * decay_factor


def calculate_intimacy_decay_with_time_diff(
    base_intimacy: float,
    time_diff: float,
    decay_rate: float = 2.0
) -> float:
    # 亲密度线性衰减: intimacy' = intimacy - (Δt/60 × decay_rate)
    if time_diff <= 0:
        return base_intimacy
    decay_minutes = time_diff / 60.0
    decay_amount = decay_minutes * decay_rate
    return max(0.0, base_intimacy - decay_amount)


def _get_age_level(age_seconds: float) -> str:
    if age_seconds < MEMORY_AGE_THRESHOLDS[MemoryAgeLevel.FRESH]:
        return MemoryAgeLevel.FRESH
    elif age_seconds < MEMORY_AGE_THRESHOLDS[MemoryAgeLevel.TODAY]:
        return MemoryAgeLevel.TODAY
    elif age_seconds < MEMORY_AGE_THRESHOLDS[MemoryAgeLevel.YESTERDAY]:
        return MemoryAgeLevel.YESTERDAY
    elif age_seconds < MEMORY_AGE_THRESHOLDS[MemoryAgeLevel.RECENT]:
        return MemoryAgeLevel.RECENT
    elif age_seconds < MEMORY_AGE_THRESHOLDS[MemoryAgeLevel.WEEK]:
        return MemoryAgeLevel.WEEK
    else:
        return MemoryAgeLevel.OLD


def calculate_memory_clarity(timestamp: float, importance: float = 0.5, access_count: int = 0) -> float:
    # 记忆清晰度 = 基础清晰度(年龄层级) + 访问奖励(最多0.3) + 重要性奖励(最多0.1)
    now = time.time()
    age_seconds = now - timestamp
    age_level = _get_age_level(age_seconds)
    base_clarity = MEMORY_CLARITY_VALUES.get(age_level, 0.2)
    access_bonus = min(0.3, access_count * 0.03)
    importance_bonus = importance * 0.1
    clarity = min(1.0, base_clarity + access_bonus + importance_bonus)
    return clarity


def merge_similar_memories(memories: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
    if not memories:
        return []
    merged = []
    used = set()
    for i, mem1 in enumerate(memories):
        if i in used:
            continue
        group = [mem1]
        for j, mem2 in enumerate(memories[i + 1:], start=i + 1):
            if j in used:
                continue
            score = calculate_relevance_score(
                mem1.get("content", ""), mem2.get("content", "")
            )
            if score >= similarity_threshold:
                group.append(mem2)
                used.add(j)
        if len(group) == 1:
            merged.append(mem1)
        else:
            combined_content = " | ".join(m.get("content", "") for m in group)
            merged.append({
                "content": combined_content,
                "source": "merged",
                "count": len(group),
            })
        used.add(i)
    return merged


def truncate_memory(content: str, max_length: int = 500) -> str:
    if len(content) <= max_length:
        return content
    return content[:max_length - 3] + "..."


def build_memory_prompt(
    query: str,
    memories: List[Dict],
    system_prompt: str = "",
) -> str:
    memory_context = format_memory_context(memories)
    parts = []
    if system_prompt:
        parts.append(system_prompt)
    if memory_context:
        parts.append(f"相关记忆:\n{memory_context}")
    parts.append(f"当前问题: {query}")
    return "\n\n".join(parts)

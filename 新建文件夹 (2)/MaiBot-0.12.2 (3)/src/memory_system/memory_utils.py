# -*- coding: utf-8 -*-
"""
记忆系统工具函数
包含模糊查找、相似度计算等工具函数
"""

import json
import re
from datetime import datetime
from typing import Tuple, List, Dict, Optional
from json_repair import repair_json

from src.common.logger import get_logger


logger = get_logger("memory_utils")


def parse_questions_json(response: str) -> Tuple[List[str], List[str]]:
    """解析问题JSON，返回概念列表和问题列表

    Args:
        response: LLM返回的响应

    Returns:
        Tuple[List[str], List[str]]: (概念列表, 问题列表)
    """
    try:
        # 尝试提取JSON（可能包含在```json代码块中）
        json_pattern = r"```json\s*(.*?)\s*```"
        matches = re.findall(json_pattern, response, re.DOTALL)

        if matches:
            json_str = matches[0]
        else:
            # 尝试直接解析整个响应
            json_str = response.strip()

        # 修复可能的JSON错误
        repaired_json = repair_json(json_str)

        # 解析JSON
        parsed = json.loads(repaired_json)

        # 只支持新格式：包含concepts和questions的对象
        if not isinstance(parsed, dict):
            logger.warning(f"解析的JSON不是对象格式: {parsed}")
            return [], []

        concepts_raw = parsed.get("concepts", [])
        questions_raw = parsed.get("questions", [])

        # 确保是列表
        if not isinstance(concepts_raw, list):
            concepts_raw = []
        if not isinstance(questions_raw, list):
            questions_raw = []

        # 确保所有元素都是字符串
        concepts = [c for c in concepts_raw if isinstance(c, str) and c.strip()]
        questions = [q for q in questions_raw if isinstance(q, str) and q.strip()]

        return concepts, questions

    except Exception as e:
        logger.error(f"解析问题JSON失败: {e}, 响应内容: {response[:200]}...")
        return [], []


def parse_datetime_to_timestamp(value: str) -> float:
    """
    接受多种常见格式并转换为时间戳（秒）
    支持示例：
    - 2025-09-29
    - 2025-09-29 00:00:00
    - 2025/09/29 00:00
    - 2025-09-29T00:00:00
    """
    value = value.strip()
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]
    last_err = None
    for fmt in fmts:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.timestamp()
        except Exception as e:
            last_err = e
    raise ValueError(f"无法解析时间: {value} ({last_err})")


def format_chat_history(messages: List[dict], max_messages: int = 20) -> str:
    if not messages:
        return ""
    import time as _time
    lines = []
    for msg in messages[-max_messages:]:
        sender = msg.get("sender", "未知")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", 0)
        if timestamp:
            time_str = _time.strftime("%H:%M", _time.localtime(timestamp))
            lines.append(f"[{time_str}] {sender}: {content}")
        else:
            lines.append(f"{sender}: {content}")
    return "\n".join(lines)


def extract_time_references(text: str) -> List[Tuple[str, str]]:
    patterns = [
        (r"(\d+)\s*天前", "days_ago"),
        (r"(\d+)\s*小时前", "hours_ago"),
        (r"(\d+)\s*分钟前", "minutes_ago"),
        (r"昨天", "yesterday"),
        (r"前天", "day_before_yesterday"),
        (r"上周", "last_week"),
        (r"上个月", "last_month"),
        (r"之前", "before"),
        (r"以前", "in_the_past"),
        (r"上次", "last_time"),
    ]
    results = []
    for pattern, ref_type in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            results.append((str(match), ref_type))
    return results


def calculate_time_from_reference(ref_type: str, value: str = "1") -> float:
    import time as _time
    now = _time.time()
    try:
        num = int(value) if value.isdigit() else 1
    except ValueError:
        num = 1
    if ref_type == "days_ago":
        return now - (num * 86400)
    elif ref_type == "hours_ago":
        return now - (num * 3600)
    elif ref_type == "minutes_ago":
        return now - (num * 60)
    elif ref_type == "yesterday":
        return now - 86400
    elif ref_type == "day_before_yesterday":
        return now - (2 * 86400)
    elif ref_type == "last_week":
        return now - (7 * 86400)
    elif ref_type == "last_month":
        return now - (30 * 86400)
    return now - 86400


def normalize_query(query: str) -> str:
    query = query.strip()
    query = re.sub(r'\s+', ' ', query)
    query = query.lower()
    return query


def split_query_into_parts(query: str) -> List[str]:
    parts = re.split(r'[，,。.？?！!;；]', query)
    return [p.strip() for p in parts if p.strip()]


def merge_memory_results(results: List[dict], max_results: int = 10) -> List[dict]:
    if not results:
        return []
    seen_contents = set()
    unique_results = []
    for result in results:
        content = result.get("content", "")
        content_key = content[:100].lower()
        if content_key not in seen_contents:
            seen_contents.add(content_key)
            unique_results.append(result)
    unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return unique_results[:max_results]


def format_memory_for_prompt(memories: List[dict], max_chars: int = 2000) -> str:
    if not memories:
        return "暂无相关记忆"
    lines = []
    total_chars = 0
    for mem in memories:
        content = mem.get("content", "")
        source = mem.get("source", "")
        line = f"- [{source}] {content}"
        if total_chars + len(line) > max_chars:
            break
        lines.append(line)
        total_chars += len(line)
    return "\n".join(lines)


def extract_entities(text: str) -> List[str]:
    patterns = [
        r'@(\w+)',
        r'[\u4e00-\u9fa5]{2,4}(?:君|酱|桑|先生|女士|老师|同学)',
    ]
    entities = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        entities.extend(matches)
    return list(set(entities))


def calculate_query_complexity(query: str) -> float:
    length_score = min(1.0, len(query) / 100)
    time_refs = extract_time_references(query)
    time_score = min(1.0, len(time_refs) * 0.3)
    entities = extract_entities(query)
    entity_score = min(1.0, len(entities) * 0.2)
    question_words = ["什么", "为什么", "怎么", "如何", "哪", "谁", "何时", "多少"]
    question_score = sum(0.1 for w in question_words if w in query)
    question_score = min(1.0, question_score)
    total = (length_score * 0.2 + time_score * 0.3 + entity_score * 0.2 + question_score * 0.3)
    return min(1.0, total)


class QueryAnalyzer:
    def __init__(self):
        self._initialized = True

    def analyze(self, query: str) -> dict:
        return {
            "normalized": normalize_query(query),
            "parts": split_query_into_parts(query),
            "time_refs": extract_time_references(query),
            "entities": extract_entities(query),
            "complexity": calculate_query_complexity(query),
        }


class MemoryFormatter:
    def __init__(self):
        self._max_chars = 2000

    def format(self, memories: List[Dict]) -> str:
        return format_memory_for_prompt(memories, self._max_chars)

    def set_max_chars(self, max_chars: int):
        self._max_chars = max_chars


_query_analyzer = None
_memory_formatter = None


def get_query_analyzer() -> QueryAnalyzer:
    global _query_analyzer
    if _query_analyzer is None:
        _query_analyzer = QueryAnalyzer()
    return _query_analyzer


def get_memory_formatter() -> "MemoryFormatter":
    global _memory_formatter
    if _memory_formatter is None:
        _memory_formatter = MemoryFormatter()
    return _memory_formatter

import json
import re
import time as _stdlib_time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from json_repair import repair_json

from src.common.logger import get_logger

logger = get_logger("记忆工具")


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
        concepts = [
            c for c in concepts_raw if isinstance(c, str) and c.strip()
        ]
        questions = [
            q for q in questions_raw if isinstance(q, str) and q.strip()
        ]

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


# ---------------------------------------------------------------------------
#  对话记录渲染
# ---------------------------------------------------------------------------


def render_dialogue_sequence(
    records: List[dict],
    tail_limit: int = 20,
) -> str:
    """将消息记录列表格式化为可阅读的对话文本

    只保留最近 tail_limit 条，每条按 "[HH:MM] 发言者: 内容" 排列。
    如果记录中缺少时间戳则省略时间前缀。
    """
    if not records:
        return ""
    segment = records[-tail_limit:]
    rendered_lines: list = []
    for rec in segment:
        who = rec.get("sender", "匿名")
        body = rec.get("content", "")
        ts = rec.get("timestamp", 0)
        if ts:
            clock_label = _stdlib_time.strftime(
                "%H:%M", _stdlib_time.localtime(ts)
            )
            rendered_lines.append(f"[{clock_label}] {who}: {body}")
        else:
            rendered_lines.append(f"{who}: {body}")
    return "\n".join(rendered_lines)


# ---------------------------------------------------------------------------
#  自然语言时间提取
# ---------------------------------------------------------------------------

# 预编译的时间标识模式表：(编译后正则, 标识类别)
_TEMPORAL_MARKER_TABLE: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(\d+)\s*天前"), "days_back"),
    (re.compile(r"(\d+)\s*小时前"), "hours_back"),
    (re.compile(r"(\d+)\s*分钟前"), "minutes_back"),
    (re.compile(r"昨天"), "prior_day"),
    (re.compile(r"前天"), "two_days_prior"),
    (re.compile(r"上周"), "prev_week"),
    (re.compile(r"上个月"), "prev_month"),
    (re.compile(r"之前"), "generic_past"),
    (re.compile(r"以前"), "generic_past"),
    (re.compile(r"上次"), "prev_occurrence"),
]


def scan_temporal_markers(text: str) -> List[Tuple[str, str]]:
    """从文本中提取所有时间相关的表述

    返回列表，每个元素是 (匹配到的原始值, 标识类别) 二元组。
    例如 "3天前" 会返回 ("3", "days_back")。
    """
    discovered: list = []
    for compiled_pat, kind in _TEMPORAL_MARKER_TABLE:
        hits = compiled_pat.findall(text)
        for h in hits:
            discovered.append((str(h) if h else "", kind))
    return discovered


# 类别 → 秒数偏移量的映射表（值为负偏移秒数的计算因子）
_OFFSET_SECONDS_MAP: Dict[str, int] = {
    "days_back": 86400,
    "hours_back": 3600,
    "minutes_back": 60,
    "prior_day": 86400,
    "two_days_prior": 2 * 86400,
    "prev_week": 7 * 86400,
    "prev_month": 30 * 86400,
    "generic_past": 86400,
    "prev_occurrence": 86400,
}


def resolve_temporal_offset(marker_kind: str, quantity: str = "1") -> float:
    """把时间标识类别转换为对应的 UNIX 时间戳

    根据 marker_kind 查表得到单位秒数，乘以 quantity 后从当前时刻回溯。
    """
    now_ts = _stdlib_time.time()
    try:
        multiplier = int(quantity) if quantity.isdigit() else 1
    except (ValueError, TypeError):
        multiplier = 1
    unit_seconds = _OFFSET_SECONDS_MAP.get(marker_kind, 86400)
    # 固定偏移类型（如 prior_day）不需要乘 multiplier
    fixed_kinds = {
        "prior_day",
        "two_days_prior",
        "prev_week",
        "prev_month",
        "generic_past",
        "prev_occurrence",
    }
    if marker_kind in fixed_kinds:
        return now_ts - unit_seconds
    return now_ts - (unit_seconds * multiplier)


# ---------------------------------------------------------------------------
#  查询文本预处理
# ---------------------------------------------------------------------------

_WHITESPACE_COLLAPSE = re.compile(r"\s+")
# 中英文标点分隔符集合
_CLAUSE_DELIMITERS = re.compile(r"[，,。.？?！!；;、\n]+")


def sanitize_search_text(raw_input: str) -> str:
    """清洗查询文本：去首尾空白、压缩连续空格、统一小写"""
    trimmed = raw_input.strip()
    collapsed = _WHITESPACE_COLLAPSE.sub(" ", trimmed)
    return collapsed.lower()


def segment_by_punctuation(sentence: str) -> List[str]:
    """按中英文标点把长句切成短语片段，过滤空段"""
    fragments = _CLAUSE_DELIMITERS.split(sentence)
    return [f.strip() for f in fragments if f.strip()]


# ---------------------------------------------------------------------------
#  记忆条目去重与合并
# ---------------------------------------------------------------------------


def deduplicate_recall_entries(
    entries: List[dict],
    cap: int = 10,
) -> List[dict]:
    """对检索结果按内容前缀指纹去重，按得分降序截取前 cap 条

    指纹取内容前100字符小写形式，相同指纹只保留首次出现的条目。
    """
    if not entries:
        return []
    fingerprint_seen: set = set()
    unique_batch: list = []
    for entry in entries:
        raw_content = entry.get("content", "")
        fp = raw_content[:100].lower()
        if fp in fingerprint_seen:
            continue
        fingerprint_seen.add(fp)
        unique_batch.append(entry)
    # 按 score 降序排列
    unique_batch.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return unique_batch[:cap]


# ---------------------------------------------------------------------------
#  记忆格式化为文本
# ---------------------------------------------------------------------------


def compose_recall_block(
    items: List[dict],
    char_budget: int = 2000,
) -> str:
    """将记忆条目列表拼接为带来源标签的文本块

    按顺序追加，超过 char_budget 字符限额时截断。
    空列表返回占位提示。
    """
    if not items:
        return "暂无相关记忆"
    assembled: list = []
    consumed = 0
    for item in items:
        body = item.get("content", "")
        origin = item.get("source", "")
        single_line = f"- [{origin}] {body}" if origin else f"- {body}"
        if consumed + len(single_line) > char_budget:
            break
        assembled.append(single_line)
        consumed += len(single_line)
    return "\n".join(assembled)


# ---------------------------------------------------------------------------
#  实体抽取
# ---------------------------------------------------------------------------

# 预编译的实体匹配正则
_AT_MENTION_PAT = re.compile(r"@(\w+)")
_CN_HONORIFIC_PAT = re.compile(
    r"[\u4e00-\u9fa5]{2,4}(?:君|酱|桑|先生|女士|老师|同学)"
)


def detect_named_mentions(text: str) -> List[str]:
    """从文本中提取 @提及 和带中文敬称后缀的人名

    返回去重后的列表。
    """
    found: list = []
    found.extend(_AT_MENTION_PAT.findall(text))
    found.extend(_CN_HONORIFIC_PAT.findall(text))
    return list(set(found))


# ---------------------------------------------------------------------------
#  查询复杂度评分
# ---------------------------------------------------------------------------

# 疑问关键词库
_INTERROGATIVE_TOKENS = frozenset(
    [
        "什么",
        "为什么",
        "怎么",
        "如何",
        "哪",
        "谁",
        "何时",
        "多少",
    ]
)


def estimate_question_depth(question: str) -> float:
    """综合评估一条查询的"复杂度"，返回 0.0~1.0 的归一化分值

    四个维度加权：
    - 文本长度 (权重 0.2)
    - 时间标识数量 (权重 0.3)
    - 实体提及数量 (权重 0.2)
    - 疑问词密度 (权重 0.3)
    """
    # 维度1: 长度因子
    length_factor = min(1.0, len(question) / 100)
    # 维度2: 时间引用因子
    temporal_hits = scan_temporal_markers(question)
    temporal_factor = min(1.0, len(temporal_hits) * 0.3)
    # 维度3: 实体因子
    entity_hits = detect_named_mentions(question)
    entity_factor = min(1.0, len(entity_hits) * 0.2)
    # 维度4: 疑问词因子
    interrogative_count = sum(
        1 for tok in _INTERROGATIVE_TOKENS if tok in question
    )
    interrogative_factor = min(1.0, interrogative_count * 0.1)
    # 加权汇总
    composite = (
        length_factor * 0.2
        + temporal_factor * 0.3
        + entity_factor * 0.2
        + interrogative_factor * 0.3
    )
    return min(1.0, composite)


# ---------------------------------------------------------------------------
#  查询分析器（组合上述工具）
# ---------------------------------------------------------------------------


@dataclass
class SearchTextInspector:
    """对查询文本做一站式分析，返回结构化报告"""

    _ready: bool = field(default=True, init=False, repr=False)

    def inspect(self, text: str) -> Dict:
        """返回字典包含清洗文本、分段、时间标记、实体、复杂度"""
        cleaned = sanitize_search_text(text)
        return {
            "normalized": cleaned,
            "parts": segment_by_punctuation(text),
            "time_refs": scan_temporal_markers(text),
            "entities": detect_named_mentions(text),
            "complexity": estimate_question_depth(text),
        }


# ---------------------------------------------------------------------------
#  记忆格式化包装器
# ---------------------------------------------------------------------------


@dataclass
class RecallBlockComposer:
    """管理字符预算并将记忆条目渲染为文本块"""

    _budget: int = field(default=2000, init=False)

    def compose(self, items: List[Dict]) -> str:
        """使用当前预算渲染记忆条目"""
        return compose_recall_block(items, self._budget)

    def set_budget(self, new_budget: int):
        """动态调整字符预算"""
        self._budget = max(0, new_budget)


# ---------------------------------------------------------------------------
#  模块级单例访问器
# ---------------------------------------------------------------------------

_text_inspector: Optional[SearchTextInspector] = None
_block_composer: Optional[RecallBlockComposer] = None


def acquire_text_inspector() -> SearchTextInspector:
    """获取全局 SearchTextInspector 单例"""
    global _text_inspector
    if _text_inspector is None:
        _text_inspector = SearchTextInspector()
    return _text_inspector


def acquire_block_composer() -> RecallBlockComposer:
    """获取全局 RecallBlockComposer 单例"""
    global _block_composer
    if _block_composer is None:
        _block_composer = RecallBlockComposer()
    return _block_composer

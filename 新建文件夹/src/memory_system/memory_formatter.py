"""
记忆展示渲染器 —— 将检索到的记忆条目转化为可嵌入Prompt的文本块。

融合三套源码的格式化能力：
- 按时间桶分组展示（MaiBot的时间分组概念）
- 可配置渲染参数与双模式输出（XBcore的config驱动 + structured/summary模式）
- 管道式内容净化（原创设计，替代单一巨型正则方法）
"""

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.memory_system.context_saver import TemporalTag, annotate_temporal

logger = get_logger("mem_renderer")


# ==================== 渲染配置 ====================


@dataclass
class RenderProfile:
    """控制记忆渲染行为的参数集。"""

    ceiling: int = 10  # 单次渲染最大条目数
    token_budget: int = 600  # 文本预算上限（字符数）
    snippet_cap: int = 120  # 单条记忆最大摘要长度
    compact_limit: int = 5  # 紧凑模式最大条目数
    attach_timestamps: bool = True  # 是否附带时间信息
    heading: str = "### 相关记忆"  # 输出块标题


# ==================== 内容净化管道 ====================

# 每条规则是 (pattern_str, flags) 元组，统一替换为空串
_MEDIA_TAG_RULES: List[Tuple[str, int]] = [
    (
        r"\[(?:图片|语音|视频)(?:URLs?)?\]\s*:.*?(?=\n|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
]

_METADATA_EXTRACT_RULES: List[Tuple[str, int]] = [
    (r"(?:图片|文件)大小[：:]\s*\d+[KMG]?B?", re.IGNORECASE),
    (r"\d+\s*[xX×]\s*\d+\s*像素", 0),
    (r"\d+\s*[KMG]?B?\s*\(.*?\)", 0),
]

_TOOL_INVOCATION_RULES: List[Tuple[str, int]] = [
    (r'\{"(?:tool|tool_call|tool_name)"[^}]*\}', 0),
    (r'\{"name"[^}]*"type"[^}]*\}', 0),
]

_RECOGNITION_META_RULES: List[Tuple[str, int]] = [
    (
        r"(?:图片识别|识别结果|图片分析|语音识别|语音转文字)[：:].*?(?=\n|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
]

_PARAM_RESULT_RULES: List[Tuple[str, int]] = [
    (r"(?:参数|结果|params|result)[：:]\s*\{[^}]*\}", re.IGNORECASE),
]

_URL_RULES: List[Tuple[str, int]] = [
    (r"https?://\S+", 0),
    (r"www\.\S+", 0),
]


def _compile_sanitizer_pipeline() -> List[Tuple[re.Pattern, str]]:
    """预编译全部净化正则，组装成有序管道。"""
    all_rules = (
        _MEDIA_TAG_RULES
        + _METADATA_EXTRACT_RULES
        + _TOOL_INVOCATION_RULES
        + _RECOGNITION_META_RULES
        + _PARAM_RESULT_RULES
        + _URL_RULES
    )
    compiled = []
    for pattern_text, flags in all_rules:
        compiled.append((re.compile(pattern_text, flags), ""))
    return compiled


_SANITIZER_CHAIN = _compile_sanitizer_pipeline()
_BLANK_LINE_COLLAPSE = re.compile(r"\n\s*\n\s*\n+")


class ContentSanitizer:
    """通过预编译管道逐步剥离记忆文本中的冗余元数据。"""

    @staticmethod
    def purify(raw_text: str) -> str:
        """对原始文本执行全管道净化，返回干净的对话核心内容。"""
        if not raw_text:
            return ""
        result = raw_text
        for compiled_pattern, replacement in _SANITIZER_CHAIN:
            result = compiled_pattern.sub(replacement, result)
        # 合并多余空行
        result = _BLANK_LINE_COLLAPSE.sub("\n\n", result)
        result = result.strip()
        # 净化后若内容过短，尝试保留原文首句
        if len(result) < 5:
            return ContentSanitizer._salvage_lead_sentence(raw_text)
        return result

    @staticmethod
    def _salvage_lead_sentence(original: str) -> str:
        """从原始内容中抢救首句作为降级展示。"""
        if not original:
            return ""
        segments = re.split(r"[。！？\n]", original)
        lead = segments[0].strip() if segments else ""
        if len(lead) > 5:
            return lead[:100]
        return original[:100]


# ==================== 来源徽章翻译 ====================

_ORIGIN_BADGE_MAP: Dict[str, str] = {
    "short_term": "短时记忆",
    "hippocampus": "海马缓存",
    "database": "持久记忆",
    "embedding": "向量召回",
    "chat_history": "对话历史",
}


def translate_origin_badge(
    source_key: str, extra: Optional[Dict[str, Any]] = None
) -> str:
    """将内部来源标识转换为面向Prompt的中文徽章。"""
    badge = _ORIGIN_BADGE_MAP.get(source_key)
    if badge is None:
        return source_key
    # 海马缓存可附带层级信息
    if source_key == "hippocampus" and extra:
        tier_label = extra.get("tier", "")
        if tier_label:
            return f"{badge}-{tier_label}"
    return badge


# ==================== 时间桶分组器 ====================

# TemporalTag按时间远近的展示排序
_TEMPORAL_DISPLAY_ORDER: List[TemporalTag] = [
    TemporalTag.INSTANT,
    TemporalTag.MINUTES_AGO,
    TemporalTag.WITHIN_HOUR,
    TemporalTag.SAME_DAY,
    TemporalTag.PRIOR_DAY,
    TemporalTag.TWO_DAYS,
    TemporalTag.SAME_WEEK,
    TemporalTag.LONG_AGO,
]


def _resolve_timestamp(record: Dict[str, Any]) -> float:
    """从记忆字典中提取或解析时间戳。"""
    raw = record.get("timestamp") or record.get("created_at") or time.time()
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(
                raw.replace("Z", "+00:00")
            ).timestamp()
        except (ValueError, TypeError):
            return time.time()
    return time.time()


class TemporalBucketGrouper:
    """将记忆列表按TemporalTag分桶，保持时间远近有序。"""

    @staticmethod
    def distribute(records: List[Dict[str, Any]]) -> OrderedDict:
        """
        将记录分配到时间桶中。

        返回 OrderedDict[TemporalTag, List[(record, tag, description, ts)]]
        按从近到远的顺序排列。
        """
        buckets: Dict[
            TemporalTag, List[Tuple[Dict, TemporalTag, str, float]]
        ] = {}
        for rec in records:
            ts = _resolve_timestamp(rec)
            tag, readable_desc = annotate_temporal(ts)
            entry = (rec, tag, readable_desc, ts)
            if tag not in buckets:
                buckets[tag] = []
            buckets[tag].append(entry)
        # 按预定义顺序排列输出
        ordered = OrderedDict()
        for tag in _TEMPORAL_DISPLAY_ORDER:
            if tag in buckets:
                ordered[tag] = buckets[tag]
        return ordered


# ==================== 文本截断工具 ====================

_SENTENCE_TERMINATORS = frozenset("。？！.?!")


def trim_at_natural_boundary(text: str, budget: int = 120) -> str:
    """在自然句尾处截断文本，超长时添加省略标记。"""
    if len(text) <= budget:
        return text
    window = text[:budget]
    # 从后向前搜索最近的句终止符
    best_cut = -1
    threshold = int(budget * 0.6)
    for idx in range(len(window) - 1, threshold - 1, -1):
        if window[idx] in _SENTENCE_TERMINATORS:
            best_cut = idx + 1
            break
    if best_cut > 0:
        return window[:best_cut] + "..."
    return window + "..."


# ==================== 核心渲染器 ====================


class MemoryDisplayRenderer:
    """
    将记忆检索结果渲染为可嵌入Prompt的文本块。

    支持两种输出模式：
    - 完整模式(render_block): 按时间桶分组、附带来源徽章
    - 紧凑模式(render_condensed): 按相关度排序的精简列表
    """

    def __init__(self, profile: Optional[RenderProfile] = None):
        self._cfg = profile or RenderProfile()
        self._sanitizer = ContentSanitizer()
        self._grouper = TemporalBucketGrouper()

    @property
    def profile(self) -> RenderProfile:
        return self._cfg

    def render_block(
        self,
        records: List[Dict[str, Any]],
        channel_label: str = "当前群聊",
        speaker_name: str = "用户",
    ) -> str:
        """
        完整模式：按时间分桶渲染记忆块。

        参数:
            records: 记忆字典列表（需含content字段）
            channel_label: 频道/群聊名称
            speaker_name: 默认发言者名称
        返回:
            渲染后的Markdown文本块，无记忆时返回空串
        """
        if not records:
            return ""
        temporal_buckets = self._grouper.distribute(records)
        rendered_sections: List[str] = []
        items_rendered = 0
        for tag, entries in temporal_buckets.items():
            if items_rendered >= self._cfg.ceiling:
                break
            section_lines = self._render_single_bucket(
                entries, channel_label, speaker_name, items_rendered
            )
            if section_lines is not None:
                rendered_sections.append(section_lines[0])
                items_rendered += section_lines[1]
        if not rendered_sections:
            return ""
        body = "\n\n".join(rendered_sections)
        return f"{self._cfg.heading}\n\n{body}\n"

    def _render_single_bucket(
        self,
        entries: List[Tuple[Dict, TemporalTag, str, float]],
        channel_label: str,
        speaker_name: str,
        already_rendered: int,
    ) -> Optional[Tuple[str, int]]:
        """渲染单个时间桶，返回 (渲染文本, 本桶渲染条数) 或 None。"""
        if not entries:
            return None
        # 取第一条的时间描述作为桶标题
        _, _, bucket_desc, _ = entries[0]
        item_lines: List[str] = []
        count = 0
        for rec, _tag, _desc, _ts in entries:
            if already_rendered + count >= self._cfg.ceiling:
                break
            line = self._render_single_item(rec, speaker_name)
            if line:
                item_lines.append(line)
                count += 1
        if not item_lines:
            return None
        bucket_header = f"- 【{bucket_desc} ({channel_label})】"
        section_text = bucket_header + "\n" + "\n".join(item_lines)
        return section_text, count

    def _render_single_item(
        self, rec: Dict[str, Any], fallback_speaker: str
    ) -> Optional[str]:
        """渲染单条记忆为一行文本。"""
        raw_content = rec.get("content", "")
        if not raw_content:
            return None
        clean_text = self._sanitizer.purify(raw_content)
        if not clean_text:
            return None
        snippet = trim_at_natural_boundary(clean_text, self._cfg.snippet_cap)
        origin_key = rec.get("source", "未知")
        badge = translate_origin_badge(origin_key, rec)
        # 有user_id说明是特定用户的记忆，否则标记为群聊
        who = fallback_speaker if rec.get("user_id") else "群聊"
        return f"  {who}: {snippet} [{badge}]"

    def render_condensed(
        self,
        records: List[Dict[str, Any]],
        cap: Optional[int] = None,
    ) -> str:
        """
        紧凑模式：按相关度评分排序，输出精简列表。

        参数:
            records: 记忆字典列表（需含content字段，可选score字段）
            cap: 覆盖默认的紧凑条目上限
        返回:
            精简的Markdown列表文本
        """
        if not records:
            return ""
        limit = cap if cap is not None else self._cfg.compact_limit
        ranked = sorted(
            records, key=lambda r: r.get("score", 0.0), reverse=True
        )[:limit]
        output_lines: List[str] = []
        for rec in ranked:
            raw_content = rec.get("content", "")
            if not raw_content:
                continue
            clean_text = self._sanitizer.purify(raw_content)
            if not clean_text:
                continue
            brief = trim_at_natural_boundary(
                clean_text, max(60, self._cfg.snippet_cap - 40)
            )
            origin_key = rec.get("source", "未知")
            badge = translate_origin_badge(origin_key, rec)
            output_lines.append(f"- {brief} [{badge}]")
        if not output_lines:
            return ""
        return f"{self._cfg.heading}\n\n" + "\n".join(output_lines) + "\n"

    def render_with_token_budget(
        self,
        records: List[Dict[str, Any]],
        channel_label: str = "当前群聊",
        speaker_name: str = "用户",
    ) -> str:
        """
        带token预算的渲染：在字符预算内尽量多输出。

        逻辑：逐条渲染，超出budget时停止。
        融合自XBcore的max_tokens概念。
        """
        if not records:
            return ""
        temporal_buckets = self._grouper.distribute(records)
        rendered_sections: List[str] = []
        total_chars = 0
        items_rendered = 0
        for tag, entries in temporal_buckets.items():
            if items_rendered >= self._cfg.ceiling:
                break
            if total_chars >= self._cfg.token_budget:
                break
            _, _, bucket_desc, _ = entries[0]
            bucket_header = f"- 【{bucket_desc} ({channel_label})】"
            item_lines: List[str] = []
            for rec, _tag, _desc, _ts in entries:
                if items_rendered >= self._cfg.ceiling:
                    break
                if total_chars >= self._cfg.token_budget:
                    break
                line = self._render_single_item(rec, speaker_name)
                if line:
                    line_cost = len(line) + 1  # 换行符
                    if (
                        total_chars + line_cost > self._cfg.token_budget
                        and item_lines
                    ):
                        break
                    item_lines.append(line)
                    total_chars += line_cost
                    items_rendered += 1
            if item_lines:
                section_text = bucket_header + "\n" + "\n".join(item_lines)
                rendered_sections.append(section_text)
                total_chars += len(bucket_header) + 2  # 桶标题开销
        if not rendered_sections:
            return ""
        body = "\n\n".join(rendered_sections)
        return f"{self._cfg.heading}\n\n{body}\n"


# ==================== 模块级便捷接口 ====================

_shared_renderer: Optional[MemoryDisplayRenderer] = None


def _get_shared_renderer() -> MemoryDisplayRenderer:
    """获取模块级共享渲染器实例（懒初始化）。"""
    global _shared_renderer
    if _shared_renderer is None:
        _shared_renderer = MemoryDisplayRenderer()
    return _shared_renderer


def render_memory_block(
    memories: List[Dict[str, Any]],
    stream_name: str = "当前群聊",
    sender_name: str = "用户",
    max_display: Optional[int] = None,
) -> str:
    """模块级便捷函数：完整模式渲染记忆块。"""
    if max_display is not None:
        custom_profile = RenderProfile(ceiling=max_display)
        renderer = MemoryDisplayRenderer(profile=custom_profile)
    else:
        renderer = _get_shared_renderer()
    return renderer.render_block(memories, stream_name, sender_name)


def render_compact_memories(
    memories: List[Dict[str, Any]],
    cap: Optional[int] = None,
) -> str:
    """模块级便捷函数：紧凑模式渲染记忆列表。"""
    renderer = _get_shared_renderer()
    return renderer.render_condensed(memories, cap=cap)


def render_budgeted_block(
    memories: List[Dict[str, Any]],
    stream_name: str = "当前群聊",
    sender_name: str = "用户",
    token_budget: int = 600,
) -> str:
    """模块级便捷函数：带字符预算的记忆渲染。"""
    profile = RenderProfile(token_budget=token_budget)
    renderer = MemoryDisplayRenderer(profile=profile)
    return renderer.render_with_token_budget(
        memories, stream_name, sender_name
    )

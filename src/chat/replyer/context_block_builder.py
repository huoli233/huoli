import re
from typing import List, Tuple


LIGHTWEIGHT_EXTRA_PREFIXES = (
    "最近对话:",
    "前情:",
    "补充:",
    "避免复读:",
    "回复形式:",
    "当前状态:",
    "关系:",
    "语气:",
    "这轮:",
    "禁区:",
    "边界:",
    "续接提示:",
    "重复短句判定:",
    "当前对象:",
    "最新用户话头:",
    "你上一句:",
)

LIGHTWEIGHT_EXTRA_INLINE_MARKERS = (
    "关系=",
    "续接提示:",
    "重复短句判定:",
    "当前对象:",
    "最新用户话头:",
    "你上一句:",
)

STYLE_GUIDANCE_KEYWORDS = (
    "自然",
    "口语",
    "语气",
    "短句",
    "回复",
    "别",
    "不要",
    "禁止",
    "关系",
    "状态",
    "情绪",
    "防备",
    "烦",
    "接话",
)

LEGACY_EXTRA_HEADER_PREFIX_MAP = {
    "[前情执行块]": "前情:",
    "[决策上下文包]": "这轮:",
    "[避免复读]": "避免复读:",
    "[回复形式]": "回复形式:",
    "[主人格摘要]": "当前状态:",
    "[身份边界]": "边界:",
    "[接管意图]": "这轮:",
    "[内心独白执行约束]": "禁区:",
    "[关系阶段]": "关系:",
    "[关系信号]": "关系:",
    "[心理信号]": "当前状态:",
    "[回复边界]": "边界:",
    "[表达节奏]": "语气:",
    "[情绪保护]": "禁区:",
    "[分层回复模式]": "回复形式:",
    "[表达长度]": "回复形式:",
    "[表达细节]": "补充:",
    "[避免模式]": "禁区:",
    "[自我风格学习]": "语气:",
    "[时段调节]": "当前状态:",
    "[★冷拒模式★]": "语气:",
    "[★烦躁模式★]": "语气:",
    "[灵魂指令]": "补充:",
    "[防御模式]": "边界:",
    "[当前情感状态]": "当前状态:",
    "[当前心理状态]": "当前状态:",
    "[内心思考]": "补充:",
    "[当前情绪]": "当前状态:",
    "[人格约束]": "边界:",
    "[当前人格片段]": "当前状态:",
    "[连续对话参考]": "前情:",
    "[回复长度策略]": "回复形式:",
    "[精力状态]": "当前状态:",
    "[主动行为意图]": "这轮:",
    "[内容规划]": "这轮:",
    "[当前感受]": "当前状态:",
    "[最近观察]": "补充:",
    "[投递指令]": "这轮:",
    "[群聊氛围感知]": "当前状态:",
    "[⏰凌晨情绪放大]": "当前状态:",
    "[⚠️独立人格感知]": "边界:",
    "[🛡️人格底线]": "边界:",
    "[📏建议长度]": "回复形式:",
    "[🎬场景风格]": "语气:",
    "[🚫禁止]": "禁区:",
    "[😑敷衍厌烦]": "语气:",
    "[🫠勉强应付]": "语气:",
    "[😤情绪底色]": "当前状态:",
    "[😒情绪底色]": "当前状态:",
    "[★夜间深睡★]": "当前状态:",
    "[★夜间浅睡★]": "当前状态:",
    "[★熬穿状态★]": "当前状态:",
    "[★夜间活跃★]": "当前状态:",
    "[★社交夜★]": "当前状态:",
    "[★午夜反思★]": "当前状态:",
    "[★安静沉思★]": "当前状态:",
    "[★黎明恢复★]": "当前状态:",
    "[🌙🧠内心独白]": "补充:",
    "[🌙🧠烦躁瞥眼]": "当前状态:",
    "[🌙😤烦躁强制约束]": "禁区:",
    "[🌙🧠情绪基调]": "当前状态:",
    "[🌙📊唤醒门槛]": "当前状态:",
    "[🌙身体状态]": "当前状态:",
    "[💫灵魂情绪覆盖]": "当前状态:",
}


def _context_view() -> dict:
    try:
        from src.config.core_config_engine import get_core_config

        return get_core_config().resolve_module_view("context").values
    except Exception:
        return {}


def _context_int(key: str, fallback: int) -> int:
    try:
        return int(_context_view().get(key, fallback))
    except Exception:
        return fallback


def _context_float(key: str, fallback: float) -> float:
    try:
        return float(_context_view().get(key, fallback))
    except Exception:
        return fallback


def estimate_token_count(text: str) -> int:
    """估算文本 token 数（中文约1.5字/token，英文约4字符/token）"""
    if not text:
        return 0
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + ascii_chars / 4)


def budget_aware_trim(text: str, max_tokens: int, priority_keywords: List[str] = None) -> str:
    """Token 预算驱动的智能裁剪"""
    if not text or not text.strip():
        return ""
    current_tokens = estimate_token_count(text)
    if current_tokens <= max_tokens:
        return text
    lines = text.split("\n")
    scored_lines = []
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        score = 1.0
        if line.startswith("[") and line.endswith("]"):
            score = 2.0
        elif priority_keywords:
            for keyword in priority_keywords:
                if keyword in line:
                    score = 3.0
                    break
        scored_lines.append((line, idx, score))
    scored_lines.sort(key=lambda x: x[2], reverse=True)
    selected_lines = []
    accumulated_tokens = 0
    for line, original_idx, score in scored_lines:
        line_tokens = estimate_token_count(line)
        if accumulated_tokens + line_tokens <= max_tokens:
            selected_lines.append((line, original_idx, score))
            accumulated_tokens += line_tokens
        elif accumulated_tokens >= max_tokens * _context_float("selection_saturation_ratio", 0.8):
            break
    selected_lines.sort(key=lambda x: x[1])
    return "\n".join(line for line, _, _ in selected_lines)


def sanitize_extra_info(extra_info: str, max_tokens: int | None = None) -> str:
    if max_tokens is None:
        max_tokens = _context_int("extra_info_max_tokens", 200)
    text = str(extra_info or "").strip()
    if not text:
        return ""
    kept: List[str] = []
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    for block in blocks:
        lines = [line.strip() for line in str(block).splitlines() if line.strip()]
        if not lines:
            continue
        header = lines[0]
        prefix = LEGACY_EXTRA_HEADER_PREFIX_MAP.get(header, "")
        if prefix:
            payload = _compact_context_text(
                "\n".join(lines[1:]),
                max_tokens=_context_int("compact_line_max_tokens", 42),
                priority_keywords=["重要", "当前", "别", "不要", "禁止"],
            )
            if payload:
                kept.append(f"{prefix} {payload}")
            continue
        for line in lines:
            legacy_inline = re.match(r"^(\[[^\]]+\])\s*(.*)$", line)
            if legacy_inline:
                legacy_prefix = LEGACY_EXTRA_HEADER_PREFIX_MAP.get(legacy_inline.group(1), "")
                if legacy_prefix:
                    inline_payload = _compact_context_text(
                        legacy_inline.group(2),
                        max_tokens=_context_int("compact_line_max_tokens", 42),
                        priority_keywords=["重要", "当前", "别", "不要", "禁止"],
                    )
                    if inline_payload:
                        kept.append(f"{legacy_prefix} {inline_payload}")
                    continue
            if line.startswith(LIGHTWEIGHT_EXTRA_PREFIXES) or any(
                marker in line for marker in LIGHTWEIGHT_EXTRA_INLINE_MARKERS
            ):
                compact_line = _compact_context_text(
                    line,
                    max_tokens=_context_int("compact_line_max_tokens", 42),
                    priority_keywords=["重要", "当前", "别", "不要", "禁止"],
                )
                if compact_line:
                    kept.append(compact_line)
                continue
            if any(keyword in line for keyword in STYLE_GUIDANCE_KEYWORDS):
                compact_line = _compact_context_text(
                    line,
                    max_tokens=_context_int("compact_line_max_tokens", 42),
                    priority_keywords=["重要", "当前", "别", "不要", "禁止"],
                )
                if compact_line:
                    kept.append(f"补充: {compact_line}")
    deduped: List[str] = []
    seen = set()
    for block in kept:
        normalized = re.sub(r"\s+", " ", block).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(block)
    result = "\n".join(deduped[: _context_int("max_extra_blocks", 8)])
    current_tokens = estimate_token_count(result)
    if current_tokens > max_tokens:
        priority_keywords = ["重要", "关键", "注意", "当前", "问题"]
        result = budget_aware_trim(result, max_tokens, priority_keywords)
    return result


def _normalize_context_line(line: str) -> str:
    payload = str(line or "").strip()
    if not payload:
        return ""
    payload = re.sub(r"^\[[^\]]+\]\s*", "", payload)
    payload = re.sub(r"^[\-*•]\s*", "", payload)
    payload = re.sub(r"^\d+[.、:：]\s*", "", payload)
    payload = re.sub(r"\s+", " ", payload).strip("；;，,。 ")
    if not payload:
        return ""
    if any(
        marker in payload
        for marker in (
            "以下是你在回复时需要参考",
            "以上是回复时必须参考",
            "请先消化再回复",
            "请根据此意图调整",
        )
    ):
        return ""
    return payload


def _compact_context_text(text: str, max_tokens: int, priority_keywords: List[str] = None) -> str:
    payload = str(text or "").strip()
    if not payload:
        return ""
    seen = set()
    lines: List[str] = []
    for raw_line in payload.splitlines():
        line = _normalize_context_line(raw_line)
        if not line:
            continue
        normalized = re.sub(r"\s+", " ", line)
        if normalized in seen:
            continue
        seen.add(normalized)
        lines.append(line)
    if not lines:
        return ""
    merged = "；".join(lines)
    if estimate_token_count(merged) <= max_tokens:
        return merged
    trimmed = budget_aware_trim("\n".join(lines), max_tokens, priority_keywords or [])
    compact_lines = [_normalize_context_line(line) for line in trimmed.splitlines()]
    compact_lines = [line for line in compact_lines if line]
    return "；".join(compact_lines)


def _ensure_lightweight_prefix(text: str, default_prefix: str = "补充:") -> str:
    payload = str(text or "").strip()
    if not payload:
        return ""
    if payload.startswith(LIGHTWEIGHT_EXTRA_PREFIXES):
        return payload
    return f"{default_prefix} {payload}".strip()


def build_reply_context_block(
    recent_context: str,
    relevant_context: str,
    extra_info: str,
    recent_reply_guard: str = "",
    max_total_tokens: int | None = None,
) -> str:
    """构建轻量回复上下文块，避免拼成说明书。"""
    if max_total_tokens is None:
        max_total_tokens = _context_int("reply_context_max_tokens", 180)
    parts: List[Tuple[str, int]] = []
    priority_keywords = ["问题", "关键", "重要", "当前", "注意"]
    recent_text = str(recent_context or "").strip()
    if recent_text:
        recent_budget = int(max_total_tokens * 0.45)
        recent_text = _compact_context_text(recent_text, recent_budget, priority_keywords)
        if recent_text:
            parts.append((f"最近对话: {recent_text}", 3))
    relevant_text = str(relevant_context or "").strip()
    if relevant_text:
        relevant_budget = int(max_total_tokens * 0.25)
        relevant_text = _compact_context_text(relevant_text, relevant_budget, priority_keywords)
        if relevant_text:
            parts.append((f"前情: {relevant_text}", 2))
    sanitized_extra = sanitize_extra_info(extra_info, max_tokens=int(max_total_tokens * 0.25))
    if sanitized_extra:
        compact_extra = _compact_context_text(
            sanitized_extra,
            int(max_total_tokens * 0.2),
            priority_keywords,
        )
        if compact_extra:
            parts.append((_ensure_lightweight_prefix(compact_extra, "补充:"), 1))
    guard = str(recent_reply_guard or "").strip()
    if guard:
        guard_budget = max(16, int(max_total_tokens * 0.12))
        guard = _compact_context_text(guard, guard_budget, priority_keywords)
        if guard:
            parts.append((f"避免复读: {guard}", 3))
    if not parts:
        fallback_extra = _compact_context_text(
            str(extra_info or "").strip(),
            max(24, int(max_total_tokens * 0.22)),
            priority_keywords,
        )
        if fallback_extra:
            return _ensure_lightweight_prefix(fallback_extra, "补充:")
        return ""
    parts.sort(key=lambda x: x[1], reverse=True)
    selected_parts = []
    accumulated_tokens = 0
    for part_text, _priority in parts:
        part_tokens = estimate_token_count(part_text)
        if accumulated_tokens + part_tokens <= max_total_tokens:
            selected_parts.append(part_text)
            accumulated_tokens += part_tokens
        elif accumulated_tokens >= max_total_tokens * _context_float("selection_saturation_ratio", 0.8):
            break
    if not selected_parts:
        return ""
    return "\n".join(selected_parts)


def extend_reply_parts(parts: List[str], *candidates: str) -> List[str]:
    """向回复片段列表中追加非空且去重的文本块。"""
    seen = {str(p or "").strip() for p in parts if str(p or "").strip()}
    for item in candidates:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        parts.append(text)
        seen.add(text)
    return parts


def append_reply_style(parts: List[str], style_route: dict) -> List[str]:
    """追加统一的回复形式描述。"""
    if not isinstance(style_route, dict):
        return parts
    reply_style = str(style_route.get("reply_style", "") or "").strip()
    reason = str(style_route.get("reason", "") or "").strip()
    if not reply_style and not reason:
        return parts
    chunks = [chunk for chunk in (reply_style, reason) if chunk]
    return extend_reply_parts(parts, f"回复形式: {' | '.join(chunks)}".strip())


def merge_extra_info(base_text: str, *blocks: str) -> str:
    """将已有 extra_info 与追加块合并，自动去空行。"""
    parts: List[str] = []
    if str(base_text or "").strip():
        parts.append(str(base_text).strip())
    extend_reply_parts(parts, *blocks)
    return "\n".join(parts)


def build_shared_reply_parts(
    *,
    self_memory: str = "",
    continuity_context: str = "",
    user_style_guide: str = "",
    persona_hint: str = "",
    reply_style_context: str = "",
    length_hint: str = "",
    restraint_mode: str = "allow",
    short_only_text: str = "",
) -> List[str]:
    """构建共享回复片段，按价值排序后轻量合并。"""
    prioritized_blocks = [
        (reply_style_context, 70),
        (persona_hint, 60),
        (length_hint, 28),
        (short_only_text if restraint_mode == "short_only" else "", 26),
        (continuity_context, 55),
        (self_memory, 48),
        (user_style_guide, 45),
    ]
    parts: List[str] = []
    seen = set()
    total_budget = 170
    consumed = 0
    for block, budget in prioritized_blocks:
        compact = _compact_context_text(block, budget)
        if not compact:
            continue
        normalized = re.sub(r"\s+", " ", compact).strip()
        if normalized in seen:
            continue
        block_tokens = estimate_token_count(compact)
        if consumed + block_tokens > total_budget and parts:
            continue
        parts.append(compact)
        seen.add(normalized)
        consumed += block_tokens
    return parts

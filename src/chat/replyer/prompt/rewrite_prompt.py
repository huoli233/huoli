from typing import Any, Dict

from src.chat.utils.prompt_builder import Prompt, global_prompt_manager

_COMPACT_REWRITER_TEMPLATE = """
{identity}
{expression_habits_block}
{reply_style}
{chat_target}
{chat_info}

你现在{chat_target_2}，准备把一句太冲或太别扭的话顺手收一收。
{reply_target_block}
原句：{raw_reply}
主要问题：{reason}

只做同语域软化，保持像同一个人顺手接话。
- 贴着当前聊天现场，短句、口语、自然。
- 不要客服腔、公文腔、通知腔、系统提示腔，也不要编辑感太重。
- 禁止出现“您”“请问”“建议您”“感谢理解”“谢谢配合”。
- 可以压短、换说法、轻微重组，但保留基本意思和态度。
{moderation_prompt}

只输出改写后的那一句，不要解释，不要加引号，不要输出多余符号。
""".strip()


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return ""


# ---------------------------------------------------------------------------
#  改写提示词辅助函数
# ---------------------------------------------------------------------------


def get_chat_target_desc(is_group: bool, sender_name: str = "") -> str:
    """获取聊天场景描述（长格式，用于提示词开头）"""
    if is_group:
        template = global_prompt_manager.get_prompt("chat_target_group1")
        return str(template) if template is not None else "你在群里聊天，下面是眼前这点上下文:"
    template = global_prompt_manager.get_prompt("chat_target_private1")
    template_text = str(template) if template is not None else "你在和{sender_name}聊天，下面是眼前这点上下文："
    return template_text.format(sender_name=sender_name)


def get_chat_target_short(is_group: bool, sender_name: str = "") -> str:
    """获取聊天场景描述（短格式，用于提示词中间部分）"""
    if is_group:
        template = global_prompt_manager.get_prompt("chat_target_group2")
        return str(template) if template is not None else "正在群里聊天"
    template = global_prompt_manager.get_prompt("chat_target_private2")
    template_text = str(template) if template is not None else "和{sender_name}聊天"
    return template_text.format(sender_name=sender_name)


def estimate_prompt_token_count(prompt_text: str) -> int:
    """粗略估算提示词 token 数，保证改写链也走预算压缩。"""
    if not prompt_text:
        return 0
    chinese_chars = 0
    special_chars = 0
    ascii_chars = 0
    for ch in prompt_text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
            chinese_chars += 1
        elif cp > 0x7F:
            special_chars += 1
        else:
            ascii_chars += 1
    return int(chinese_chars * 1.5 + special_chars * 1.8 + ascii_chars / 4)


def _is_valid_content(content: Any) -> bool:
    if content is None:
        return False
    return bool(str(content).strip())


def _compress_long_text(text: str, max_tokens: int = 100) -> str:
    if not _is_valid_content(text):
        return ""
    text = str(text).strip()
    current_tokens = estimate_prompt_token_count(text)
    if current_tokens <= max_tokens:
        return text
    ratio = max_tokens / max(current_tokens, 1)
    target_len = max(12, int(len(text) * ratio))
    truncated = text[:target_len]
    for sep in ["\n", "。", "！", "？", ".", "!", "?"]:
        cut = truncated.rfind(sep)
        if cut >= int(target_len * 0.65):
            return truncated[: cut + 1].strip()
    return truncated.strip() + "..."


def _compress_chat_info(text: str, max_tokens: int = 160) -> str:
    if not _is_valid_content(text):
        return ""
    text = str(text).strip()
    if estimate_prompt_token_count(text) <= max_tokens:
        return text
    lines = []
    used_tokens = 0
    for raw_line in text.splitlines():
        line = str(raw_line or "").strip()
        if not line:
            continue
        line_tokens = estimate_prompt_token_count(line)
        if used_tokens + line_tokens > max_tokens:
            break
        lines.append(line)
        used_tokens += line_tokens
    if lines:
        return "\n".join(lines)
    return _compress_long_text(text, max_tokens)


def _filter_rewriter_variables(**kwargs) -> Dict[str, str]:
    compression_config = {
        "expression_habits_block": {"max_tokens": 56, "compressor": _compress_long_text},
        "chat_info": {"max_tokens": 120, "compressor": _compress_chat_info},
        "identity": {"max_tokens": 80, "compressor": _compress_long_text},
        "reply_target_block": {"max_tokens": 60, "compressor": _compress_long_text},
        "raw_reply": {"max_tokens": 60, "compressor": _compress_long_text},
        "reason": {"max_tokens": 36, "compressor": _compress_long_text},
        "reply_style": {"max_tokens": 60, "compressor": _compress_long_text},
        "moderation_prompt": {"max_tokens": 24, "compressor": _compress_long_text},
    }
    filtered: Dict[str, str] = {}
    for key, value in kwargs.items():
        if not _is_valid_content(value):
            continue
        text = str(value).strip()
        config = compression_config.get(key)
        if config:
            text = config["compressor"](text, config["max_tokens"])
        if _is_valid_content(text):
            filtered[key] = text
    return filtered


def _safe_format_template(template_text: str, variables: Dict[str, str]) -> str:
    merged = dict(variables)
    for _ in range(24):
        try:
            return template_text.format(**merged)
        except KeyError as exc:
            merged[str(exc).strip("'")] = ""
    return template_text.format_map(_SafeFormatDict(merged))


def get_rewriter_prompt(
    *,
    is_group: bool = True,
    sender_name: str = "",
    expression_habits_block: str = "",
    chat_info: str = "",
    identity: str = "",
    reply_target_block: str = "",
    raw_reply: str = "",
    reason: str = "",
    reply_style: str = "",
    moderation_prompt: str = "",
) -> str:
    """渲染改写提示词"""
    template = global_prompt_manager.get_prompt("compact_rewriter_prompt")
    if template is None:
        template = global_prompt_manager.get_prompt("default_expressor_prompt")
    template_text = str(template) if template is not None else _COMPACT_REWRITER_TEMPLATE
    chat_target = get_chat_target_desc(is_group, sender_name)
    chat_target_2 = get_chat_target_short(is_group, sender_name)
    filtered_vars = _filter_rewriter_variables(
        expression_habits_block=expression_habits_block,
        chat_info=chat_info,
        identity=identity,
        reply_target_block=reply_target_block,
        raw_reply=raw_reply,
        reason=reason,
        reply_style=reply_style,
        moderation_prompt=moderation_prompt,
    )
    filtered_vars["chat_target"] = chat_target
    filtered_vars["chat_target_2"] = chat_target_2
    return _safe_format_template(template_text, filtered_vars)


def init_rewrite_prompt():
    Prompt("你在群里聊天，下面是眼前这点上下文:", "chat_target_group1")
    Prompt(
        "你在和{sender_name}聊天，下面是眼前这点上下文：",
        "chat_target_private1",
    )
    Prompt("正在群里聊天", "chat_target_group2")
    Prompt("和{sender_name}聊天", "chat_target_private2")

    Prompt(_COMPACT_REWRITER_TEMPLATE, "compact_rewriter_prompt")
    Prompt(_COMPACT_REWRITER_TEMPLATE, "default_expressor_prompt")

from typing import Any, Dict
from src.common.logger import get_logger
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager

logger = get_logger("私聊提示词")

_PRIVATE_REPLYER_PROMPT_TEMPLATE = """{extra_info_block}
{knowledge_prompt}{tool_info_block}
{expression_habits_block}{memory_retrieval}{jargon_explanation}
{planner_reasoning}
{identity}
{chat_prompt}
你正在和{sender_name}私聊，顺着眼前的话头自然回一句。
别用“您”“请问”“好的”“当然”这类客服话，也别写成说明书。
{reply_style}

{time_block}
最近聊天:
{dialogue_prompt}

这轮对方说:
{reply_target_block}
只输出要发的话，不要解释、前缀或括号。
{moderation_prompt}"""

_PRIVATE_REPLYER_SELF_TEMPLATE = """{extra_info_block}
{knowledge_prompt}{tool_info_block}
{expression_habits_block}{memory_retrieval}{jargon_explanation}
{identity}
{chat_prompt}
你在和对方私聊，现在只是顺手补半句。
别用“您”“请问”“好的”“当然”这类客服话，也别写成说明书。
{reply_style}

{time_block}
最近聊天:
{dialogue_prompt}

你刚说过:
{target}
这次只是顺手再补一句:
{reason}
只输出要发的话，不要解释、前缀或括号。
{moderation_prompt}"""


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return ""


def _resolve_private_template(template_name: str):
    template = global_prompt_manager.get_prompt(template_name)
    if template is not None:
        return template
    init_replyer_private_prompt()
    template = global_prompt_manager.get_prompt(template_name)
    if template is not None:
        logger.warning(f"私聊模板 '{template_name}' 运行时缺失，已自动重建。")
        return template
    logger.error(f"私聊模板 '{template_name}' 缺失且自动重建失败，使用内置兜底模板。")
    if template_name == "private_replyer_prompt":
        return _PRIVATE_REPLYER_PROMPT_TEMPLATE
    if template_name == "private_replyer_self_prompt":
        return _PRIVATE_REPLYER_SELF_TEMPLATE
    return ""


def _safe_format_template(template_text: str, variables: Dict[str, str]) -> str:
    """安全格式化模板，缺失字段自动补空，避免 KeyError 导致回复中断。"""
    merged = dict(variables)
    for _ in range(24):
        try:
            return template_text.format(**merged)
        except KeyError as exc:
            merged[str(exc).strip("'")] = ""
    try:
        return template_text.format_map(_SafeFormatDict(merged))
    except Exception:
        logger.exception("私聊模板格式化失败，返回空字符串")
        return ""


def estimate_prompt_token_count(prompt_text: str) -> int:
    """粗略估算提示词 token 数，保证私聊链也能走预算压缩。"""
    if not prompt_text:
        return 0
    chinese_chars = 0
    emoji_special_chars = 0
    ascii_chars = 0
    for ch in prompt_text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
            chinese_chars += 1
        elif cp >= 0x1F000 or (0x2600 <= cp <= 0x27BF) or (0xFE00 <= cp <= 0xFE0F):
            emoji_special_chars += 1
        elif cp > 0x7F:
            emoji_special_chars += 1
        else:
            ascii_chars += 1
    return int(chinese_chars * 1.5 + emoji_special_chars * 2.0 + ascii_chars / 4)


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


def _compress_chat_block(text: str, max_tokens: int = 180) -> str:
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


def _filter_private_variables(**kwargs) -> Dict[str, str]:
    compression_config = {
        "extra_info_block": {"max_tokens": 180, "compressor": _compress_chat_block},
        "memory_retrieval": {"max_tokens": 100, "compressor": _compress_long_text},
        "jargon_explanation": {"max_tokens": 80, "compressor": _compress_long_text},
        "tool_info_block": {"max_tokens": 120, "compressor": _compress_long_text},
        "expression_habits_block": {"max_tokens": 90, "compressor": _compress_long_text},
        "knowledge_prompt": {"max_tokens": 120, "compressor": _compress_long_text},
        "planner_reasoning": {"max_tokens": 80, "compressor": _compress_long_text},
        "reply_style": {"max_tokens": 70, "compressor": _compress_long_text},
        "dialogue_prompt": {"max_tokens": 220, "compressor": _compress_chat_block},
        "reply_target_block": {"max_tokens": 60, "compressor": _compress_long_text},
        "reason": {"max_tokens": 60, "compressor": _compress_long_text},
        "target": {"max_tokens": 60, "compressor": _compress_long_text},
        "identity": {"max_tokens": 80, "compressor": _compress_long_text},
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


def get_private_responder_prompt(
    *,
    sender_name: str = "",
    identity: str = "",
    chat_prompt: str = "",
    reply_style: str = "",
    knowledge_prompt: str = "",
    tool_info_block: str = "",
    extra_info_block: str = "",
    expression_habits_block: str = "",
    memory_retrieval: str = "",
    jargon_explanation: str = "",
    time_block: str = "",
    dialogue_prompt: str = "",
    reply_target_block: str = "",
    planner_reasoning: str = "",
    moderation_prompt: str = "",
) -> str:
    """渲染私聊回复提示词"""
    template = _resolve_private_template("private_replyer_prompt")
    if template is None:
        return ""
    template_text = str(template)
    filtered_vars = _filter_private_variables(
        sender_name=sender_name,
        identity=identity,
        chat_prompt=chat_prompt,
        reply_style=reply_style,
        knowledge_prompt=knowledge_prompt,
        tool_info_block=tool_info_block,
        extra_info_block=extra_info_block,
        expression_habits_block=expression_habits_block,
        memory_retrieval=memory_retrieval,
        jargon_explanation=jargon_explanation,
        time_block=time_block,
        dialogue_prompt=dialogue_prompt,
        reply_target_block=reply_target_block,
        planner_reasoning=planner_reasoning,
        moderation_prompt=moderation_prompt,
    )
    return _safe_format_template(template_text, filtered_vars)


def get_self_continuation_prompt(
    *,
    sender_name: str = "",
    target: str = "",
    reason: str = "",
    identity: str = "",
    chat_prompt: str = "",
    reply_style: str = "",
    knowledge_prompt: str = "",
    tool_info_block: str = "",
    extra_info_block: str = "",
    expression_habits_block: str = "",
    memory_retrieval: str = "",
    jargon_explanation: str = "",
    time_block: str = "",
    dialogue_prompt: str = "",
    moderation_prompt: str = "",
) -> str:
    """渲染私聊自我补充提示词"""
    template = _resolve_private_template("private_replyer_self_prompt")
    if template is None:
        return ""
    template_text = str(template)
    filtered_vars = _filter_private_variables(
        sender_name=sender_name,
        target=target,
        reason=reason,
        identity=identity,
        chat_prompt=chat_prompt,
        reply_style=reply_style,
        knowledge_prompt=knowledge_prompt,
        tool_info_block=tool_info_block,
        extra_info_block=extra_info_block,
        expression_habits_block=expression_habits_block,
        memory_retrieval=memory_retrieval,
        jargon_explanation=jargon_explanation,
        time_block=time_block,
        dialogue_prompt=dialogue_prompt,
        moderation_prompt=moderation_prompt,
    )
    return _safe_format_template(template_text, filtered_vars)


def init_replyer_private_prompt():
    Prompt(_PRIVATE_REPLYER_PROMPT_TEMPLATE, "private_replyer_prompt")
    Prompt(_PRIVATE_REPLYER_SELF_TEMPLATE, "private_replyer_self_prompt")

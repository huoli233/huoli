from typing import Dict, Any
from src.common.logger import get_logger
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager

logger = get_logger("回复提示词")


_REPLYER_PROMPT_0_TEMPLATE = """{behavioral_directive}
{knowledge_prompt}{tool_info_block}{extra_info_block}
{expression_habits_block}{memory_retrieval}{jargon_explanation}
{identity}
{chat_prompt}
你就在这个群里，按眼前的话题自然接一句。
别用"您""请问""好的""当然"这类客服话，也别像说明书一样分析或列点。
{length_guide}
{reply_style}

{time_block}
最近聊天:
{dialogue_prompt}

{reply_target_block}
{planner_reasoning}
只输出这次要发的话，不要前后缀、括号、表情包、at或@。"""

_REPLYER_PROMPT_TEMPLATE = """{behavioral_directive}
{knowledge_prompt}{tool_info_block}{extra_info_block}
{expression_habits_block}{memory_retrieval}{jargon_explanation}
{identity}
{chat_prompt}
你就在这个群里，顺着当前话题自然回一句。
别用"您""请问""好的""当然"这类客服话，也别把回复写成说明书。
{length_guide}
{reply_style}

{time_block}
最近聊天:
{dialogue_prompt}

{reply_target_block}
{planner_reasoning}
只输出要发的话，不要额外前缀、括号、at或@。"""

_REPLYER_SYSTEM_0_TEMPLATE = """{behavioral_directive}
{knowledge_prompt}{tool_info_block}{extra_info_block}
{expression_habits_block}{memory_retrieval}{jargon_explanation}
{identity}
{chat_prompt}
你就在这个群里，像平时一样随手接话。
别用"您""请问""好的""当然"这类客服话，也别写成说明书。
{length_guide}
{reply_style}
只输出发言内容，不要前缀、括号、表情包、at或@。

下面是最近对话，标注 {bot_name}(你) 的是你自己。"""

_REPLYER_SYSTEM_TEMPLATE = """{behavioral_directive}
{knowledge_prompt}{tool_info_block}{extra_info_block}
{expression_habits_block}{memory_retrieval}{jargon_explanation}
{identity}
{chat_prompt}
你就在这个群里，顺着当前话题自然接一句。
别用"您""请问""好的""当然"这类客服话，也别写成说明书。
{length_guide}
{reply_style}
只输出发言内容，不要前缀、括号、at或@。

下面是最近对话，标注 {bot_name}(你) 的是你自己。"""

_FALLBACK_TEMPLATE_BY_NAME = {
    "replyer_prompt_0": _REPLYER_PROMPT_0_TEMPLATE,
    "replyer_prompt": _REPLYER_PROMPT_TEMPLATE,
    "replyer_system_0": _REPLYER_SYSTEM_0_TEMPLATE,
    "replyer_system": _REPLYER_SYSTEM_TEMPLATE,
}


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return ""


def _resolve_reply_template(template_name: str):
    """获取回复模板；若运行时模板缺失，尝试自恢复并兜底。"""
    template = global_prompt_manager.get_prompt(template_name)
    if template is not None:
        return template
    init_replyer_prompt()
    template = global_prompt_manager.get_prompt(template_name)
    if template is not None:
        logger.warning(f"回复模板 '{template_name}' 运行时缺失，已自动重建。")
        return template
    logger.error(f"回复模板 '{template_name}' 缺失且自动重建失败，使用内置兜底模板。")
    return _FALLBACK_TEMPLATE_BY_NAME.get(template_name, "")


def _safe_format_template(template_text: str, variables: Dict[str, str]) -> str:
    """安全格式化模板，缺失字段自动补空，避免 KeyError 终止回复链。"""
    merged = dict(variables)
    for _ in range(24):
        try:
            return template_text.format(**merged)
        except KeyError as exc:
            merged[str(exc).strip("'")] = ""
    try:
        return template_text.format_map(_SafeFormatDict(merged))
    except Exception:
        logger.exception("回复模板格式化失败，返回空字符串")
        return ""


def estimate_prompt_token_count(prompt_text: str) -> int:
    """粗略估算提示词 token 数（中文约1.5字/token，英文约4字符/token，emoji/特殊字符约1-3 token/字符）"""
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
    """检查内容是否有效（非空且非纯空白）"""
    if content is None:
        return False
    text = str(content).strip()
    return bool(text)


def _compress_long_text(text: str, max_tokens: int = 100) -> str:
    """智能压缩过长文本到指定 token 预算，优先保留语义完整性"""
    if not _is_valid_content(text):
        return ""
    text = str(text).strip()
    current_tokens = estimate_prompt_token_count(text)
    if current_tokens <= max_tokens:
        return text
    ratio = max_tokens / current_tokens
    target_len = int(len(text) * ratio)
    truncated = text[:target_len]
    for sep in ["。", "！", "？", "\n", ".", "!", "?"]:
        last_sep = truncated.rfind(sep)
        if last_sep > target_len * 0.7:
            return truncated[: last_sep + 1].strip()
    return truncated.strip() + "..."


def _smart_compress_context_block(text: str, max_tokens: int = 150) -> str:
    """智能压缩上下文块，提取关键信息"""
    if not _is_valid_content(text):
        return ""
    text = str(text).strip()
    current_tokens = estimate_prompt_token_count(text)
    if current_tokens <= max_tokens:
        return text
    lines = text.split("\n")
    compressed_lines = []
    accumulated_tokens = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line_tokens = estimate_prompt_token_count(line)
        if accumulated_tokens + line_tokens > max_tokens:
            break
        if line.startswith("[") and line.endswith("]"):
            compressed_lines.append(line)
            accumulated_tokens += line_tokens
        elif any(
            marker in line
            for marker in ["问题", "回复", "关键", "重要", "注意", "当前", "关系", "好感", "信任", "厌烦", "压力", "禁回"]
        ):
            compressed_lines.append(line)
            accumulated_tokens += line_tokens
        elif accumulated_tokens < max_tokens * 0.6:
            compressed_lines.append(line)
            accumulated_tokens += line_tokens
    if not compressed_lines:
        return _compress_long_text(text, max_tokens)
    return "\n".join(compressed_lines)


def _filter_and_compress_variables(**kwargs) -> Dict[str, str]:
    """过滤空变量并压缩长文本"""
    compression_config = {
        "extra_info_block": {"max_tokens": 180, "compressor": _smart_compress_context_block},
        "memory_retrieval": {"max_tokens": 100, "compressor": _compress_long_text},
        "jargon_explanation": {"max_tokens": 80, "compressor": _compress_long_text},
        "tool_info_block": {"max_tokens": 120, "compressor": _compress_long_text},
        "expression_habits_block": {"max_tokens": 100, "compressor": _compress_long_text},
        "knowledge_prompt": {"max_tokens": 150, "compressor": _compress_long_text},
        "planner_reasoning": {"max_tokens": 100, "compressor": _compress_long_text},
        "reply_style": {"max_tokens": 80, "compressor": _compress_long_text},
        "length_guide": {"max_tokens": 50, "compressor": _compress_long_text},
        "behavioral_directive": {"max_tokens": 120, "compressor": _compress_long_text},
        "emotional_lens": {"max_tokens": 80, "compressor": _compress_long_text},
    }
    valid_parts = {}
    for key, value in kwargs.items():
        if not _is_valid_content(value):
            continue
        text = str(value).strip()
        if key in compression_config:
            config = compression_config[key]
            compressor = config["compressor"]
            max_tokens = config["max_tokens"]
            compressed = compressor(text, max_tokens)
            if _is_valid_content(compressed):
                valid_parts[key] = compressed
        else:
            valid_parts[key] = text
    return valid_parts


def get_group_responder_prompt(
    *,
    think_level: int = 1,
    bot_name: str = "",
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
    length_guide: str = "",
    behavioral_directive: str = "",
    emotional_lens: str = "",
) -> str:
    """按 think_level 渲染群聊回复提示词，自动过滤空变量并压缩长文本"""
    template_name = (
        "replyer_prompt_0" if think_level == 0 else "replyer_prompt"
    )
    template = _resolve_reply_template(template_name)
    if template is None:
        return ""
    template_text = str(template)
    filtered_vars = _filter_and_compress_variables(
        bot_name=bot_name,
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
        length_guide=length_guide,
        behavioral_directive=behavioral_directive,
        emotional_lens=emotional_lens,
    )
    return _safe_format_template(template_text, filtered_vars)


def get_group_system_prompt(
    *,
    think_level: int = 1,
    bot_name: str = "",
    identity: str = "",
    chat_prompt: str = "",
    reply_style: str = "",
    knowledge_prompt: str = "",
    tool_info_block: str = "",
    extra_info_block: str = "",
    expression_habits_block: str = "",
    memory_retrieval: str = "",
    jargon_explanation: str = "",
    length_guide: str = "",
    behavioral_directive: str = "",
    emotional_lens: str = "",
) -> str:
    """按 think_level 渲染群聊多轮对话的 system 提示词，自动过滤空变量并压缩长文本"""
    template_name = (
        "replyer_system_0" if think_level == 0 else "replyer_system"
    )
    template = _resolve_reply_template(template_name)
    if template is None:
        return ""
    template_text = str(template)
    filtered_vars = _filter_and_compress_variables(
        bot_name=bot_name,
        identity=identity,
        chat_prompt=chat_prompt,
        reply_style=reply_style,
        knowledge_prompt=knowledge_prompt,
        tool_info_block=tool_info_block,
        extra_info_block=extra_info_block,
        expression_habits_block=expression_habits_block,
        memory_retrieval=memory_retrieval,
        jargon_explanation=jargon_explanation,
        length_guide=length_guide,
        behavioral_directive=behavioral_directive,
        emotional_lens=emotional_lens,
    )
    return _safe_format_template(template_text, filtered_vars)


def budget_truncate_prompt(prompt_text: str, max_tokens: int = 3000) -> str:
    """若估算 token 超过预算，优先保留头部系统指令和尾部近期对话"""
    if not _is_valid_content(prompt_text):
        return ""
    current_tokens = estimate_prompt_token_count(prompt_text)
    if current_tokens <= max_tokens:
        return prompt_text
    head_reserve_tokens = int(max_tokens * 0.3)
    tail_reserve_tokens = max_tokens - head_reserve_tokens
    head_chars = 0
    head_token_sum = 0
    for ch in prompt_text:
        head_chars += 1
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
            head_token_sum += 1.5
        elif cp > 0x7F:
            head_token_sum += 2.0
        else:
            head_token_sum += 0.25
        if head_token_sum >= head_reserve_tokens:
            break
    head_part = prompt_text[:head_chars]
    newline_pos = head_part.rfind("\n")
    if newline_pos > 0:
        head_part = head_part[: newline_pos + 1]
    tail_ratio = tail_reserve_tokens / max(current_tokens, 1)
    tail_len = int(len(prompt_text) * tail_ratio)
    tail_part = prompt_text[-tail_len:]
    first_nl = tail_part.find("\n")
    if first_nl != -1 and first_nl < len(tail_part) // 4:
        tail_part = tail_part[first_nl + 1 :]
    return head_part + "\n...(内容过长已省略)...\n" + tail_part


def init_replyer_prompt():
    Prompt(_REPLYER_PROMPT_0_TEMPLATE, "replyer_prompt_0")
    Prompt(_REPLYER_PROMPT_TEMPLATE, "replyer_prompt")
    Prompt(_REPLYER_SYSTEM_0_TEMPLATE, "replyer_system_0")
    Prompt(_REPLYER_SYSTEM_TEMPLATE, "replyer_system")

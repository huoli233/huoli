from typing import Dict, Optional
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager


# ---------------------------------------------------------------------------
#  提示词辅助函数
# ---------------------------------------------------------------------------


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
    moderation_prompt: str = "",
    length_guide: str = "",
) -> str:
    """按 think_level 渲染群聊回复提示词"""
    template_name = (
        "replyer_prompt_0" if think_level == 0 else "replyer_prompt"
    )
    template = global_prompt_manager.get_prompt(template_name)
    if not template:
        return ""
    return template.format(
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
    )


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
) -> str:
    """按 think_level 渲染群聊多轮对话的 system 提示词"""
    template_name = (
        "replyer_system_0" if think_level == 0 else "replyer_system"
    )
    template = global_prompt_manager.get_prompt(template_name)
    if not template:
        return ""
    return template.format(
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
    )


def estimate_prompt_token_count(prompt_text: str) -> int:
    """粗略估算提示词 token 数（中文约1.5字/token，英文约4字符/token）"""
    chinese_chars = sum(1 for ch in prompt_text if "\u4e00" <= ch <= "\u9fff")
    ascii_chars = len(prompt_text) - chinese_chars
    return int(chinese_chars * 1.5 + ascii_chars / 4)


def budget_truncate_prompt(prompt_text: str, max_tokens: int = 3000) -> str:
    """若估算 token 超过预算，从开头裁剪多余部分"""
    current_tokens = estimate_prompt_token_count(prompt_text)
    if current_tokens <= max_tokens:
        return prompt_text
    ratio = max_tokens / current_tokens
    target_len = int(len(prompt_text) * ratio)
    truncated = prompt_text[-target_len:]
    first_newline = truncated.find("\n")
    if first_newline != -1 and first_newline < len(truncated) // 4:
        truncated = truncated[first_newline + 1:]
    return "...\n" + truncated


def init_replyer_prompt():
    # 原有模板保留用于兼容
    Prompt(
        """{behavioral_directive}
{knowledge_prompt}{tool_info_block}{extra_info_block}
{expression_habits_block}{memory_retrieval}{jargon_explanation}

你正在qq群里聊天，下面是群里正在聊的内容，其中包含聊天记录和聊天中的图片
其中标注 {bot_name}(你) 的发言是你自己的发言，请注意区分:
{time_block}
{dialogue_prompt}

{reply_target_block}。
{planner_reasoning}
{identity}
{chat_prompt}你是群里的一员，像真人一样随手打字聊天。
不要用"您""请问""好的""当然"等客服用语，不要像程序说明书那样分析问题、列点回答。
像群友水群一样，随意、口语化、简短。
{length_guide}
{reply_style}
不要输出多余内容(包括前后缀、冒号、括号、表情包、at或@)，只输出发言内容。
一次回一个话题就够。
现在，你说：""",
        "replyer_prompt_0",
    )

    Prompt(
        """{behavioral_directive}
{knowledge_prompt}{tool_info_block}{extra_info_block}
{expression_habits_block}{memory_retrieval}{jargon_explanation}

你正在qq群里聊天，下面是群里正在聊的内容，其中包含聊天记录和聊天中的图片
其中标注 {bot_name}(你) 的发言是你自己的发言，请注意区分:
{time_block}
{dialogue_prompt}

{reply_target_block}。
{planner_reasoning}
{identity}
{chat_prompt}你是群里的一员，像真人一样随手打字聊天，把握当前的话题。
不要用"您""请问""好的""当然"等客服用语，不要像程序说明书那样分析问题、列点回答。
像群友聊天一样，自然、口语化。
{length_guide}
{reply_style}
不要输出多余内容(包括前后缀、冒号、括号、at或@)，只输出发言内容。
现在，你说：""",
        "replyer_prompt",
    )

    # 新增：多轮对话模式的 System 模板 (think_level=0)
    Prompt(
        """{behavioral_directive}
{knowledge_prompt}{tool_info_block}{extra_info_block}
{expression_habits_block}{memory_retrieval}{jargon_explanation}

{identity}
{chat_prompt}你是群里的一员，像真人一样随手打字聊天。
不要用"您""请问""好的""当然"等客服用语，不要像程序说明书那样分析问题。
像群友水群一样，随意、口语化、简短。
{length_guide}
{reply_style}
不要输出多余内容(包括前后缀、冒号、括号、表情包、at或@)，只输出发言内容。
一次回一个话题就够。

以下是对话历史，其中标注 {bot_name}(你) 的发言是你自己的发言，请注意区分。""",
        "replyer_system_0",
    )

    # 新增：多轮对话模式的 System 模板 (think_level=1)
    Prompt(
        """{behavioral_directive}
{knowledge_prompt}{tool_info_block}{extra_info_block}
{expression_habits_block}{memory_retrieval}{jargon_explanation}

{identity}
{chat_prompt}你是群里的一员，像真人一样随手打字聊天，把握当前的话题。
不要用"您""请问""好的""当然"等客服用语，不要像程序说明书那样分析问题。
像群友聊天一样，自然、口语化。
{length_guide}
{reply_style}
不要输出多余内容(包括前后缀、冒号、括号、at或@)，只输出发言内容。

以下是对话历史，其中标注 {bot_name}(你) 的发言是你自己的发言，请注意区分。""",
        "replyer_system",
    )

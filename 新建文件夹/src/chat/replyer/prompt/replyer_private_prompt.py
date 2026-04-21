from typing import Optional
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager


# ---------------------------------------------------------------------------
#  私聊提示词辅助函数
# ---------------------------------------------------------------------------


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
    template = global_prompt_manager.get_prompt("private_replyer_prompt")
    if not template:
        return ""
    return template.format(
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
    template = global_prompt_manager.get_prompt("private_replyer_self_prompt")
    if not template:
        return ""
    return template.format(
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


def init_replyer_private_prompt():
    Prompt(
        """{extra_info_block}
{knowledge_prompt}{tool_info_block}
{expression_habits_block}{memory_retrieval}{jargon_explanation}

[对话记录]
{time_block}
{dialogue_prompt}

[当前消息]
{reply_target_block}

{planner_reasoning}
{identity}
{chat_prompt}
=== 回复规则（优先级从高到低） ===
第一优先：上面[★冷拒模式★]/[★烦躁模式★]/[灵魂指令]里的要求，绝对不能违反。
第二优先：如果内心独白写了"哈。""又来""烦""无不无聊"，你的回复必须体现这种态度。
第三优先：真人说话方式——省略、吐槽、反问、只回半个词都可以。不要每句都完整工整。
第四优先：短句为主，通常20字以内。除非对方说了值得展开的事。
{reply_style}
只输出回复内容，不要任何前后缀或解释。
{moderation_prompt}""",
        "private_replyer_prompt",
    )

    Prompt(
        """{extra_info_block}
{knowledge_prompt}{tool_info_block}
{expression_habits_block}{memory_retrieval}{jargon_explanation}

[对话记录]
{time_block}
{dialogue_prompt}

[补充说明]
你现在想补充说明你刚刚自己的发言内容：{target}，原因是{reason}
注意，{target} 是刚刚你自己的发言，你要在这基础上进一步发言。

{identity}
{chat_prompt}
=== 回复规则（优先级从高到低） ===
第一优先：上面[★冷拒模式★]/[★烦躁模式★]/[灵魂指令]里的要求，绝对不能违反。
第二优先：如果心里想的是"又来"或"烦"，就带着这种态度接下去——可以敷衍、可以带刺、可以只回半句。
第三优先：真人说话不会每句都工整完整。省略、吐槽、反问都可以。
{reply_style}
只输出回复内容，不要任何前后缀或解释。
{moderation_prompt}""",
        "private_replyer_self_prompt",
    )

from typing import Optional
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager


# ---------------------------------------------------------------------------
#  改写提示词辅助函数
# ---------------------------------------------------------------------------


def get_chat_target_desc(is_group: bool, sender_name: str = "") -> str:
    """获取聊天场景描述（长格式，用于提示词开头）"""
    if is_group:
        return (
            global_prompt_manager.get_prompt("chat_target_group1")
            or "你正在qq群里聊天，下面是群里正在聊的内容:"
        )
    template = (
        global_prompt_manager.get_prompt("chat_target_private1")
        or "你正在和{sender_name}聊天，这是你们之前聊的内容："
    )
    return template.format(sender_name=sender_name)


def get_chat_target_short(is_group: bool, sender_name: str = "") -> str:
    """获取聊天场景描述（短格式，用于提示词中间部分）"""
    if is_group:
        return (
            global_prompt_manager.get_prompt("chat_target_group2")
            or "正在群里聊天"
        )
    template = (
        global_prompt_manager.get_prompt("chat_target_private2")
        or "和{sender_name}聊天"
    )
    return template.format(sender_name=sender_name)


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
    template = global_prompt_manager.get_prompt("default_expressor_prompt")
    if not template:
        return ""
    chat_target = get_chat_target_desc(is_group, sender_name)
    chat_target_2 = get_chat_target_short(is_group, sender_name)
    return template.format(
        expression_habits_block=expression_habits_block,
        chat_target=chat_target,
        chat_info=chat_info,
        identity=identity,
        chat_target_2=chat_target_2,
        reply_target_block=reply_target_block,
        raw_reply=raw_reply,
        reason=reason,
        reply_style=reply_style,
        moderation_prompt=moderation_prompt,
    )


def init_rewrite_prompt():
    Prompt("你正在qq群里聊天，下面是群里正在聊的内容:", "chat_target_group1")
    Prompt(
        "你正在和{sender_name}聊天，这是你们之前聊的内容：",
        "chat_target_private1",
    )
    Prompt("正在群里聊天", "chat_target_group2")
    Prompt("和{sender_name}聊天", "chat_target_private2")

    Prompt(
        """
{expression_habits_block}
{chat_target}
{chat_info}
{identity}

你正在{chat_target_2},{reply_target_block}
现在请你对这句内容进行改写，请你参考上述内容进行改写，原句是：{raw_reply}：
原因是：{reason}
现在请你将这条具体内容改写成一条适合在群聊中发送的回复消息。
你需要使用合适的语法和句法，参考聊天内容，组织一条日常且口语化的回复。请你修改你想表达的原句，符合你的表达风格和语言习惯
{reply_style}
你可以完全重组回复，保留最基本的表达含义就好，但重组后保持语意通顺。
{moderation_prompt}
不要输出多余内容(包括冒号和引号，表情包，emoji,at或 @等 )，只输出一条回复就好。不要思考的太长。
改写后的回复：
""",
        "default_expressor_prompt",
    )

from src.chat.utils.prompt_builder import Prompt


# ---------------------------------------------------------------------------
#  知识检索提示词模板
# ---------------------------------------------------------------------------

KNOWLEDGE_RETRIEVAL_TEMPLATE = """你是一个知识检索助手。
请根据以下聊天上下文，判断是否需要查询知识库来获取相关信息。

聊天内容：
{conversation_context}

当前话题片段：{topic_keywords}

请分析聊天中是否涉及需要查询的知识点。如果需要查询，请提取最重要的查询线索（最多3个），
用逗号分隔输出。如果不需要查询，输出"无需查询"。

注意：
- 日常闲聊不需要查询
- 专业知识、历史事件、技术概念等需要查询
- 对方明确提问时需要查询
"""


def build_knowledge_query_prompt(
    conversation_context: str,
    topic_keywords: str = "",
) -> str:
    """构建知识检索查询提示词"""
    return KNOWLEDGE_RETRIEVAL_TEMPLATE.format(
        conversation_context=conversation_context,
        topic_keywords=topic_keywords,
    )


KNOWLEDGE_INJECTION_TEMPLATE = """以下是从知识库中检索到的相关信息：
{knowledge_content}

请注意：
- 这些信息仅供参考，不要生硬地复述
- 自然地融入回复中，如果信息不相关可以忽略
- 用自己的话表达，而不是直接引用
"""


def build_knowledge_injection_block(knowledge_content: str) -> str:
    """构建知识注入块（嵌入到回复提示词中）"""
    if not knowledge_content or not knowledge_content.strip():
        return ""
    return KNOWLEDGE_INJECTION_TEMPLATE.format(
        knowledge_content=knowledge_content.strip()
    )


def init_lpmm_prompt():
    """注册知识检索相关提示词"""
    Prompt(KNOWLEDGE_RETRIEVAL_TEMPLATE, "knowledge_retrieval_query")
    Prompt(KNOWLEDGE_INJECTION_TEMPLATE, "knowledge_injection_block")

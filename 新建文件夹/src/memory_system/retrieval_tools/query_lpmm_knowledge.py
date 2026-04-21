"""
通过LPMM知识库查询信息 - 工具实现
"""

from src.common.logger import get_logger  # noqa: E402
from src.config.config import global_config  # noqa: E402

try:
    from src.chat.knowledge import get_qa_manager  # noqa: E402
except ImportError as e:
    raise ImportError(
        f"无法导入 knowledge 模块，请检查模块路径是否正确: {e}"
    ) from e  # noqa: E402

try:
    from .tool_registry import register_memory_retrieval_tool  # noqa: E402
except ImportError as e:
    raise ImportError(
        f"无法导入 tool_registry 模块，请检查模块路径是否正确: {e}"
    ) from e  # noqa: E402

logger = get_logger("记忆检索工具")


async def query_lpmm_knowledge(query: str, limit: int = 5) -> str:
    """在LPMM知识库中查询相关信息

    Args:
        query: 查询内容

    Returns:
        str: 查询结果
    """
    try:
        content = str(query).strip()
        if not content:
            return "查询线索为空"

        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            limit_value = 5
        limit_value = max(1, limit_value)

        if not global_config.memory.enable_lpmm:
            logger.debug("LPMM知识库未启用")
            return "LPMM知识库未启用"

        qa_manager = get_qa_manager()
        if qa_manager is None:
            logger.debug("LPMM知识库未初始化，跳过查询")
            return "LPMM知识库未初始化"

        knowledge_info = await qa_manager.get_knowledge(
            content, limit=limit_value
        )
        logger.debug(f"LPMM知识库查询结果: {knowledge_info}")

        if knowledge_info:
            return f"你从LPMM知识库中找到以下信息：\n{knowledge_info}"

        return f"在LPMM知识库中未找到与“{content}”相关的信息"

    except Exception as e:
        logger.error(f"LPMM知识库查询失败: {e}")
        return f"LPMM知识库查询失败：{str(e)}"


def register_tool():
    """注册知识库查询工具"""
    register_memory_retrieval_tool(
        name="search_knowledge",
        description="从知识库中搜索相关信息，可以查询过往事件、过去提到过的概念、某段时间发生的事件等。使用自然语言问句检索",
        parameters=[
            {
                "name": "query",
                "type": "string",
                "description": "需要查询的问题，使用一句疑问句提问，例如：什么是AI？",
                "required": True,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "希望返回的相关知识条数，默认为5",
                "required": False,
            },
        ],
        execute_func=query_lpmm_knowledge,
    )

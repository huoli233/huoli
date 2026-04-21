from typing import List, Tuple

from src.chat.knowledge import get_qa_manager
from src.chat.message_receive.message import Message
from src.chat.utils.chat_message_builder import build_readable_messages
from src.common.logger import get_logger
from src.config.config import global_config
from src.llm_models.utils_model import LLMRequest

logger = get_logger("知识获取")


class _FallbackHippocampusManager:
    """兜底记忆管理器，缺失外部实现时返回空结果。"""

    @classmethod
    def get_instance(cls, stream_id: str = ""):
        return cls()

    async def get_memory_from_text(self, **kwargs):
        return []


try:
    from src.memory_system.hippocampus_buffer import get_hippocampus_buffer

    class _HippoAdapterForPFC:
        """将 HippoMemoryBuffer 适配为 PFC 知识调取所需的接口。"""

        def __init__(self, stream_id: str = ""):
            self._stream_id = stream_id

        @classmethod
        def get_instance(cls, stream_id: str = ""):
            return cls(stream_id)

        async def get_memory_from_text(
            self, text: str = "", max_memory_num: int = 3, **kwargs
        ):
            try:
                buf = get_hippocampus_buffer(self._stream_id)
                hits = buf.search_memories(text, limit=max_memory_num)
                return [(frag.fragment_id, frag.body) for frag, _score in hits]
            except Exception:
                return []

    HippocampusManager = _HippoAdapterForPFC
except ImportError:
    HippocampusManager = _FallbackHippocampusManager


class KnowledgeFetcher:
    """知识调取器"""

    def __init__(self, private_name: str):
        self.llm = LLMRequest(
            model=global_config.llm_normal,
            temperature=global_config.llm_normal["temp"],
            request_type="knowledge_fetch",
        )
        self.private_name = private_name

    async def _lpmm_get_knowledge(self, query: str) -> str:
        """获取相关知识。"""

        logger.debug(f"[私聊][{self.private_name}]正在从LPMM知识库中获取知识")
        try:
            qa_manager = get_qa_manager()
            if qa_manager is None:
                return "未找到匹配的知识"
            knowledge_info = await qa_manager.get_knowledge(query)
            if not knowledge_info:
                return "未找到匹配的知识"
            logger.debug(
                f"[私聊][{
                    self.private_name}]LPMM知识库查询结果: {
                    str(knowledge_info)[
                        :150]}"
            )
            return str(knowledge_info)
        except Exception as e:
            logger.error(
                f"[私聊][{self.private_name}]LPMM知识库搜索工具执行失败: {str(e)}"
            )
            return "未找到匹配的知识"

    async def fetch(
        self, query: str, chat_history: List[Message]
    ) -> Tuple[str, str]:
        """获取相关知识

        Args:
            query: 查询内容
            chat_history: 聊天历史

        Returns:
            Tuple[str, str]: (获取的知识, 知识来源)
        """
        # 构建查询上下文
        chat_history_text = await build_readable_messages(
            chat_history,
            replace_bot_name=True,
            merge_messages=False,
            timestamp_mode="relative",
            read_mark=0.0,
        )

        # 从记忆中获取相关知识
        related_memory = (
            await HippocampusManager.get_instance().get_memory_from_text(
                text=f"{query}\n{chat_history_text}",
                max_memory_num=3,
                max_memory_length=2,
                max_depth=3,
                fast_retrieval=False,
            )
        )
        knowledge_text = ""
        sources_text = "无记忆匹配"  # 默认值
        if related_memory:
            sources = []
            for memory in related_memory:
                knowledge_text += memory[1] + "\n"
                sources.append(f"记忆片段{memory[0]}")
            knowledge_text = knowledge_text.strip()
            sources_text = "，".join(sources)

        knowledge_text += "\n现在有以下**知识**可供参考：\n "
        knowledge_text += await self._lpmm_get_knowledge(query)
        knowledge_text += "\n请记住这些**知识**，并根据**知识**回答问题。\n"

        return knowledge_text or "未找到相关知识", sources_text or "无记忆匹配"

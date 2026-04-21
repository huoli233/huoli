import time
import traceback
from typing import Dict, Optional

from src.chat.brain_chat.PFC.chat_observer import ChatObserver
from src.chat.brain_chat.PFC.conversation import Conversation
from src.common.logger import get_logger

logger = get_logger("PFC管理")


class PFCManager:
    """PFC对话管理器，负责管理所有对话实例。"""

    _instance: Optional["PFCManager"] = None
    _instances: Dict[str, Conversation] = {}
    _initializing: Dict[str, bool] = {}

    @classmethod
    def get_instance(cls) -> "PFCManager":
        """获取管理器单例。"""
        if cls._instance is None:
            cls._instance = PFCManager()
        return cls._instance

    async def get_or_create_conversation(
        self, stream_id: str, private_name: str
    ) -> Optional[Conversation]:
        """获取或创建对话实例。"""
        if stream_id in self._initializing and self._initializing[stream_id]:
            logger.debug(
                f"[私聊][{private_name}]会话实例正在初始化中: {stream_id}"
            )
            return None
        if (
            stream_id in self._instances
            and self._instances[stream_id].should_continue
        ):
            logger.debug(
                f"[私聊][{private_name}]使用现有会话实例: {stream_id}"
            )
            return self._instances[stream_id]
        if stream_id in self._instances:
            instance = self._instances[stream_id]
            if (
                hasattr(instance, "ignore_until_timestamp")
                and instance.ignore_until_timestamp
                and time.time() < instance.ignore_until_timestamp
            ):
                logger.debug(
                    f"[私聊][{private_name}]会话实例当前处于忽略状态: {stream_id}"
                )
                return None
            if instance.should_continue:
                logger.debug(
                    f"[私聊][{private_name}]使用现有会话实例: {stream_id}"
                )
                return instance
        try:
            logger.info(f"[私聊][{private_name}]创建新的对话实例: {stream_id}")
            self._initializing[stream_id] = True
            conversation_instance = Conversation(stream_id, private_name)
            self._instances[stream_id] = conversation_instance
            await self._initialize_conversation(conversation_instance)
        except Exception as e:
            logger.error(
                f"[私聊][{private_name}]创建会话实例失败: {stream_id}, 错误: {e}"
            )
            self._initializing.pop(stream_id, None)
            self._instances.pop(stream_id, None)
            return None
        return conversation_instance

    async def _initialize_conversation(
        self, conversation: Conversation
    ) -> None:
        """初始化会话实例。"""
        stream_id = conversation.stream_id
        private_name = conversation.private_name
        try:
            logger.info(
                f"[私聊][{private_name}]开始初始化会话实例: {stream_id}"
            )
            await conversation._initialize()
            self._initializing[stream_id] = False
            logger.info(
                f"[私聊][{private_name}]会话实例 {stream_id} 初始化完成"
            )
        except Exception as e:
            logger.error(
                f"[私聊][{private_name}]管理器初始化会话实例失败: {stream_id}, 错误: {e}"
            )
            logger.error(f"[私聊][{private_name}]{traceback.format_exc()}")

    async def get_conversation(self, stream_id: str) -> Optional[Conversation]:
        """获取已存在的会话实例。"""
        return self._instances.get(stream_id)

    def remove_conversation(self, stream_id: str) -> None:
        """移除会话实例。"""
        if stream_id in self._instances:
            del self._instances[stream_id]
        if stream_id in self._initializing:
            del self._initializing[stream_id]
        logger.info(f"已移除会话实例: {stream_id}")

    def has_conversation(self, stream_id: str) -> bool:
        """检查是否存在会话实例。"""
        return stream_id in self._instances

    def get_all_conversations(self) -> Dict[str, Conversation]:
        """获取所有会话实例。"""
        return self._instances.copy()

    async def stop_all_conversations(self) -> None:
        """停止所有会话实例。"""
        for stream_id, conversation in self._instances.items():
            try:
                conversation.should_continue = False
                logger.info(
                    f"[私聊][{conversation.private_name}]已停止会话: {stream_id}"
                )
            except Exception as e:
                logger.error(f"停止会话 {stream_id} 失败: {e}")
        self._instances.clear()
        self._initializing.clear()

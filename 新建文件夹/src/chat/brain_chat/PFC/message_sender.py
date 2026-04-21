import time
from typing import Optional

from src.chat.message_receive.chat_stream import ChatStream
from src.chat.message_receive.message import Message, MessageSending
from src.chat.message_receive.storage import MessageStorage
from src.common.logger import get_logger
from src.config.config import global_config

logger = get_logger("消息发送")


class DirectMessageSender:
    """直接消息发送器，负责发送消息到聊天流。"""

    def __init__(self, private_name: str):
        self.private_name = private_name
        self.storage = MessageStorage()

    async def send_message(
        self,
        chat_stream: ChatStream,
        content: str,
        reply_to_message: Optional[Message] = None,
    ) -> bool:
        """发送消息到聊天流。

        Args:
            chat_stream: 聊天流对象
            content: 消息内容
            reply_to_message: 要回复的消息（可选）

        Returns:
            bool: 发送是否成功
        """
        try:
            segments = self._build_segments(content)
            bot_user_info = self._build_bot_user_info(chat_stream)
            message_id = self._generate_message_id()
            message = self._build_message(
                message_id=message_id,
                chat_stream=chat_stream,
                bot_user_info=bot_user_info,
                content=segments,
                reply_to_message=reply_to_message,
            )
            await message.process()
            sent = await self._send_to_api(message)
            if sent:
                await self.storage.store_message(message, chat_stream)
                logger.info(
                    f"[私聊][{self.private_name}]消息已发送: {content[:50]}..."
                )
                return True
            else:
                logger.error(f"[私聊][{self.private_name}]消息发送失败")
                return False
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]消息发送异常: {e}")
            return False

    def _build_segments(self, content: str) -> dict:
        """构建消息段结构。"""
        return {
            "type": "seglist",
            "data": [{"type": "text", "data": content}],
        }

    def _build_bot_user_info(self, chat_stream: ChatStream) -> dict:
        """构建机器人用户信息。"""
        return {
            "user_id": str(getattr(global_config.bot, "qq_account", "") or ""),
            "user_nickname": global_config.bot.nickname,
            "platform": chat_stream.platform,
        }

    def _generate_message_id(self) -> str:
        """生成消息ID。"""
        return f"dm{round(time.time(), 2)}"

    def _build_message(
        self,
        message_id: str,
        chat_stream: ChatStream,
        bot_user_info: dict,
        content: dict,
        reply_to_message: Optional[Message],
    ) -> MessageSending:
        """构建消息发送对象。"""
        sender_info = None
        if reply_to_message:
            sender_info = getattr(reply_to_message, "user_info", None)
        return MessageSending(
            message_id=message_id,
            chat_stream=chat_stream,
            bot_user_info=bot_user_info,
            sender_info=sender_info,
            message_segment=content,
            reply=reply_to_message,
            is_head=True,
            is_emoji=False,
            thinking_start_time=time.time(),
        )

    async def _send_to_api(self, message: MessageSending) -> bool:
        """发送消息到API。"""
        try:
            from src.chat.message_receive.uni_message_sender import (
                send_message,
            )

            return await send_message(message, show_log=True)
        except ImportError:
            logger.warning(
                "[私聊][{self.private_name}]uni_message_sender模块未找到，使用备用发送方式"
            )
            return await self._fallback_send(message)

    async def _fallback_send(self, message: MessageSending) -> bool:
        """备用发送方式。"""
        try:
            content = message.message_segment
            if isinstance(content, dict) and "data" in content:
                text_data = content["data"]
                if isinstance(text_data, list) and len(text_data) > 0:
                    text_content = text_data[0].get("data", "")
                else:
                    text_content = str(text_data)
            else:
                text_content = str(content)
            logger.info(
                f"[私聊][{self.private_name}]备用发送: {text_content[:50]}..."
            )
            return True
        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]备用发送失败: {e}")
            return False

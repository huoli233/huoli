from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple
from src.common.logger import get_logger
from src.plugin_system.base.component_types import CommandInfo, ComponentType

if TYPE_CHECKING:
    from src.chat.message_receive.message import MessageRecv
    from src.common.database.database_model import DatabaseMessages

logger = get_logger("基础命令")


class BaseCommand(ABC):
    """命令组件基类，用于处理命令请求"""

    command_name: str = ""
    command_description: str = ""
    command_pattern: str = r""

    def __init__(
        self, message: "MessageRecv", plugin_config: Optional[dict] = None
    ):
        self.message = message
        self.matched_groups: Dict[str, str] = {}
        self.plugin_config = plugin_config or {}
        self.log_prefix = "[Command]"
        logger.debug(f"{self.log_prefix} Command组件初始化完成")

    def set_matched_groups(self, groups: Dict[str, str]) -> None:
        """设置正则表达式匹配的命名组"""
        self.matched_groups = groups

    @abstractmethod
    async def execute(self) -> Tuple[bool, Optional[str], int]:
        """执行命令，返回(成功与否, 回复消息, 拦截级别)"""
        pass

    def get_config(self, key: str, default=None):
        """获取插件配置值，支持嵌套键访问"""
        if not self.plugin_config:
            return default
        keys = key.split(".")
        current = self.plugin_config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    async def send_text(
        self,
        content: str,
        set_reply: bool = False,
        reply_message: Optional["DatabaseMessages"] = None,
        storage_message: bool = True,
    ) -> bool:
        """发送文本消息"""
        from src.plugin_system.apis import send_api

        chat_stream = self.message.chat_stream
        if not chat_stream or not hasattr(chat_stream, "stream_id"):
            logger.error(f"{self.log_prefix} 缺少聊天流或stream_id")
            return False
        return await send_api.text_to_stream(
            text=content,
            stream_id=chat_stream.stream_id,
            set_reply=set_reply,
            reply_message=reply_message,
            storage_message=storage_message,
        )

    async def send_image(
        self,
        image_base64: str,
        set_reply: bool = False,
        reply_message: Optional["DatabaseMessages"] = None,
        storage_message: bool = True,
    ) -> bool:
        """发送图片消息"""
        from src.plugin_system.apis import send_api

        chat_stream = self.message.chat_stream
        if not chat_stream or not hasattr(chat_stream, "stream_id"):
            logger.error(f"{self.log_prefix} 缺少聊天流或stream_id")
            return False
        return await send_api.image_to_stream(
            image_base64,
            chat_stream.stream_id,
            set_reply=set_reply,
            reply_message=reply_message,
            storage_message=storage_message,
        )

    async def send_emoji(
        self,
        emoji_base64: str,
        set_reply: bool = False,
        reply_message: Optional["DatabaseMessages"] = None,
        storage_message: bool = True,
    ) -> bool:
        """发送表情包"""
        from src.plugin_system.apis import send_api

        chat_stream = self.message.chat_stream
        if not chat_stream or not hasattr(chat_stream, "stream_id"):
            logger.error(f"{self.log_prefix} 缺少聊天流或stream_id")
            return False
        return await send_api.emoji_to_stream(
            emoji_base64,
            chat_stream.stream_id,
            set_reply=set_reply,
            reply_message=reply_message,
        )

    async def send_command(
        self,
        command_name: str,
        args: Optional[dict] = None,
        display_message: str = "",
        storage_message: bool = True,
    ) -> bool:
        """发送命令消息"""
        from src.plugin_system.apis import send_api

        try:
            chat_stream = self.message.chat_stream
            if not chat_stream or not hasattr(chat_stream, "stream_id"):
                logger.error(f"{self.log_prefix} 缺少聊天流或stream_id")
                return False
            command_data = {"name": command_name, "args": args or {}}
            success = await send_api.command_to_stream(
                command=command_data,
                stream_id=chat_stream.stream_id,
                storage_message=storage_message,
                display_message=display_message,
            )
            if success:
                logger.info(f"{self.log_prefix} 成功发送命令: {command_name}")
            else:
                logger.error(f"{self.log_prefix} 发送命令失败: {command_name}")
            return success
        except Exception as e:
            logger.error(f"{self.log_prefix} 发送命令时出错: {e}")
            return False

    async def send_voice(self, voice_base64: str) -> bool:
        """发送语音消息"""
        from src.plugin_system.apis import send_api

        chat_stream = self.message.chat_stream
        if not chat_stream or not hasattr(chat_stream, "stream_id"):
            logger.error(f"{self.log_prefix} 缺少聊天流或stream_id")
            return False
        return await send_api.custom_to_stream(
            message_type="voice",
            content=voice_base64,
            stream_id=chat_stream.stream_id,
            typing=False,
            set_reply=False,
            reply_message=None,
            storage_message=False,
        )

    async def send_custom(
        self,
        message_type: str,
        content: str | Dict,
        display_message: str = "",
        typing: bool = False,
        set_reply: bool = False,
        reply_message: Optional["DatabaseMessages"] = None,
        storage_message: bool = True,
    ) -> bool:
        """发送自定义类型消息"""
        from src.plugin_system.apis import send_api

        chat_stream = self.message.chat_stream
        if not chat_stream or not hasattr(chat_stream, "stream_id"):
            logger.error(f"{self.log_prefix} 缺少聊天流或stream_id")
            return False
        return await send_api.custom_to_stream(
            message_type=message_type,
            content=content,
            stream_id=chat_stream.stream_id,
            display_message=display_message,
            typing=typing,
            set_reply=set_reply,
            reply_message=reply_message,
            storage_message=storage_message,
        )

    @classmethod
    def get_command_info(cls) -> "CommandInfo":
        """从类属性生成 CommandInfo"""
        if "." in cls.command_name:
            logger.error(f"Command名称 '{cls.command_name}' 包含非法字符 '.'")
            raise ValueError(
                f"Command名称 '{cls.command_name}' 包含非法字符 '.'"
            )
        return CommandInfo(
            name=cls.command_name,
            component_type=ComponentType.COMMAND,
            description=cls.command_description,
            command_pattern=cls.command_pattern,
        )

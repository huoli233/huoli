import time
import asyncio
from abc import ABC, abstractmethod
from typing import Tuple, Optional, TYPE_CHECKING, Dict, List
from src.common.logger import get_logger
from src.common.data_models.message_data_model import (
    ReplyContentType,
    ReplyContent,
    ReplySetModel,
    ForwardNode,
)
from src.chat.message_receive.chat_stream import ChatStream
from src.plugin_system.base.component_types import (
    ActionActivationType,
    ActionInfo,
    ComponentType,
)

if TYPE_CHECKING:
    from src.common.data_models.database_data_model import DatabaseMessages

logger = get_logger("基础动作")


class BaseAction(ABC):
    """Action组件基类
    Action是插件的一种组件类型，用于处理聊天中的动作逻辑
    子类可以通过类属性定义激活条件，这些会在实例化时转换为实例属性：
    - focus_activation_type: 专注模式激活类型
    - normal_activation_type: 普通模式激活类型
    - activation_keywords: 激活关键词列表
    - keyword_case_sensitive: 关键词是否区分大小写
    - parallel_action: 是否允许并行执行
    - random_activation_probability: 随机激活概率
    """

    def __init__(
        self,
        action_data: dict,
        action_reasoning: str,
        cycle_timers: dict,
        thinking_id: str,
        chat_stream: ChatStream,
        plugin_config: Optional[dict] = None,
        action_message: Optional["DatabaseMessages"] = None,
        **kwargs,
    ):
        if plugin_config is None:
            plugin_config = {}
        self.action_data = action_data
        self.reasoning = ""
        self.cycle_timers = cycle_timers
        self.thinking_id = thinking_id
        self.action_reasoning = action_reasoning
        self.plugin_config = plugin_config or {}
        self.action_name: str = getattr(
            self,
            "action_name",
            self.__class__.__name__.lower().replace("action", ""),
        )
        self.action_description: str = getattr(
            self, "action_description", self.__doc__ or "Action组件"
        )
        self.action_parameters: dict = getattr(
            self.__class__, "action_parameters", {}
        ).copy()
        self.action_require: list[str] = getattr(
            self.__class__, "action_require", []
        ).copy()
        self.activation_type = getattr(self.__class__, "activation_type", "focus_activation")
        self.random_activation_probability: float = getattr(
            self.__class__, "random_activation_probability", 0.0
        )
        self.activation_keywords: list[str] = getattr(
            self.__class__, "activation_keywords", []
        ).copy()
        self.keyword_case_sensitive: bool = getattr(
            self.__class__, "keyword_case_sensitive", False
        )
        self.parallel_action: bool = getattr(
            self.__class__, "parallel_action", True
        )
        self.associated_types: list[str] = getattr(
            self.__class__, "associated_types", []
        ).copy()
        self.chat_stream = chat_stream or kwargs.get("chat_stream")
        if self.chat_stream is None:
            raise ValueError("chat_stream 不能为 None")
        self.chat_id = self.chat_stream.stream_id
        self.platform = getattr(self.chat_stream, "platform", None)
        self.action_message = action_message
        self.group_id = None
        self.group_name = None
        self.user_id = None
        self.user_nickname = None
        self.is_group = False
        self.target_id = None
        if self.action_message is not None:
            self.group_id = (
                str(self.action_message.chat_info.group_info.group_id)
                if self.action_message.chat_info.group_info
                else None
            )
            self.group_name = (
                self.action_message.chat_info.group_info.group_name
                if self.action_message.chat_info.group_info
                else None
            )
            self.user_id = str(self.action_message.user_info.user_id)
            self.user_nickname = self.action_message.user_info.user_nickname
        if self.group_id:
            self.is_group = True
            self.target_id = self.group_id
            self.log_prefix = f"[{self.group_name}]"
        else:
            self.is_group = False
            self.target_id = self.user_id
            self.log_prefix = f"[{self.user_nickname} 的 私聊]"
        logger.debug(
            f"{
                self.log_prefix} 聊天信息: 类型={
                '群聊' if self.is_group else '私聊'}, 平台={
                self.platform}, 目标={
                    self.target_id}"
        )

    @abstractmethod
    async def execute(self) -> Tuple[bool, str]:
        """执行Action的抽象方法，子类必须实现
        Returns:
            Tuple[bool, str]: (是否执行成功, 回复文本)
        """
        pass

    async def send_text(
        self,
        content: str,
        set_reply: bool = False,
        reply_message: Optional["DatabaseMessages"] = None,
        typing: bool = False,
        storage_message: bool = True,
    ) -> bool:
        """发送文本消息"""
        if not self.chat_id:
            logger.error(f"{self.log_prefix} 缺少聊天ID")
            return False
        from src.plugin_system.apis import send_api

        return await send_api.text_to_stream(
            text=content,
            stream_id=self.chat_id,
            set_reply=set_reply,
            reply_message=reply_message,
            typing=typing,
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
        if not self.chat_id:
            logger.error(f"{self.log_prefix} 缺少聊天ID")
            return False
        from src.plugin_system.apis import send_api

        return await send_api.emoji_to_stream(
            emoji_base64,
            self.chat_id,
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
        """发送图片"""
        if not self.chat_id:
            logger.error(f"{self.log_prefix} 缺少聊天ID")
            return False
        from src.plugin_system.apis import send_api

        return await send_api.image_to_stream(
            image_base64,
            self.chat_id,
            set_reply=set_reply,
            reply_message=reply_message,
            storage_message=storage_message,
        )

    async def send_command(
        self,
        command_name: str,
        args: Optional[dict] = None,
        display_message: str = "",
        storage_message: bool = True,
    ) -> bool:
        """发送命令消息"""
        if not self.chat_id:
            logger.error(f"{self.log_prefix} 缺少聊天ID")
            return False
        from src.plugin_system.apis import send_api

        command_data = {"name": command_name, "args": args or {}}
        return await send_api.command_to_stream(
            command=command_data,
            stream_id=self.chat_id,
            storage_message=storage_message,
            display_message=display_message,
        )

    async def send_custom(
        self,
        message_type: str,
        content: str | Dict,
        typing: bool = False,
        set_reply: bool = False,
        reply_message: Optional["DatabaseMessages"] = None,
        storage_message: bool = True,
    ) -> bool:
        """发送自定义类型消息"""
        if not self.chat_id:
            logger.error(f"{self.log_prefix} 缺少聊天ID")
            return False
        from src.plugin_system.apis import send_api

        return await send_api.custom_to_stream(
            message_type=message_type,
            content=content,
            stream_id=self.chat_id,
            typing=typing,
            set_reply=set_reply,
            reply_message=reply_message,
            storage_message=storage_message,
        )

    async def send_hybrid(
        self,
        message_tuple_list: List[Tuple[ReplyContentType | str, str]],
        typing: bool = False,
        set_reply: bool = False,
        reply_message: Optional["DatabaseMessages"] = None,
        storage_message: bool = True,
    ) -> bool:
        """发送混合类型消息"""
        if not self.chat_id:
            logger.error(f"{self.log_prefix} 缺少聊天ID")
            return False
        from src.plugin_system.apis import send_api

        reply_set = ReplySetModel()
        reply_set.add_hybrid_content_by_raw(message_tuple_list)
        return await send_api.custom_reply_set_to_stream(
            reply_set=reply_set,
            stream_id=self.chat_id,
            typing=typing,
            set_reply=set_reply,
            reply_message=reply_message,
            storage_message=storage_message,
        )

    async def send_forward(
        self,
        messages_list: List[
            Tuple[str, str, List[Tuple[ReplyContentType | str, str]]] | str
        ],
        storage_message: bool = True,
    ) -> bool:
        """转发消息"""
        if not self.chat_id:
            logger.error(f"{self.log_prefix} 缺少聊天ID")
            return False
        from src.plugin_system.apis import send_api

        reply_set = ReplySetModel()
        forward_message_nodes: List[ForwardNode] = []
        for message in messages_list:
            if isinstance(message, str):
                forward_message_node = ForwardNode.construct_as_id_reference(
                    message
                )
            elif isinstance(message, Tuple) and len(message) == 3:
                sender_id, nickname, content_list = message
                single_node_content_list: List[ReplyContent] = []
                for node_content_type, node_content in content_list:
                    reply_node_content = ReplyContent(
                        content_type=node_content_type, content=node_content
                    )
                    single_node_content_list.append(reply_node_content)
                forward_message_node = ForwardNode.construct_as_created_node(
                    user_id=sender_id,
                    user_nickname=nickname,
                    content=single_node_content_list,
                )
            else:
                logger.warning(
                    f"{self.log_prefix} 转发消息时遇到无效的消息格式: {message}"
                )
                continue
            forward_message_nodes.append(forward_message_node)
        reply_set.add_forward_content(forward_message_nodes)
        return await send_api.custom_reply_set_to_stream(
            reply_set=reply_set,
            stream_id=self.chat_id,
            storage_message=storage_message,
            set_reply=False,
            reply_message=None,
        )

    async def send_voice(self, audio_base64: str) -> bool:
        """发送语音消息"""
        if not audio_base64:
            logger.error(f"{self.log_prefix} 缺少音频内容")
            return False
        from src.plugin_system.apis import send_api

        reply_set = ReplySetModel()
        reply_set.add_voice_content(audio_base64)
        return await send_api.custom_reply_set_to_stream(
            reply_set=reply_set,
            stream_id=self.chat_id,
            storage_message=False,
        )

    async def store_action_info(
        self,
        action_build_into_prompt: bool = False,
        action_prompt_display: str = "",
        action_done: bool = True,
    ) -> None:
        """存储动作信息到数据库"""
        from src.plugin_system.apis import database_api

        await database_api.store_action_info(
            chat_stream=self.chat_stream,
            action_build_into_prompt=action_build_into_prompt,
            action_prompt_display=action_prompt_display,
            action_done=action_done,
            thinking_id=self.thinking_id,
            action_data=self.action_data,
            action_name=self.action_name,
            action_reasoning=self.action_reasoning,
        )

    async def wait_for_new_message(
        self, timeout: int = 1200
    ) -> Tuple[bool, str]:
        """等待新消息或超时"""
        try:
            loop_start_time = self.action_data.get(
                "loop_start_time", time.time()
            )
            logger.info(
                f"{self.log_prefix} 开始等待新消息... (最长等待: {timeout}秒, 从时间点: {loop_start_time})"
            )
            if not self.chat_id:
                logger.error(
                    f"{self.log_prefix} 等待新消息失败: 没有有效的chat_id"
                )
                return False, "没有有效的chat_id"
            from src.plugin_system.apis import message_api

            wait_start_time = asyncio.get_running_loop().time()
            while True:
                current_time = time.time()
                new_message_count = message_api.count_new_messages(
                    chat_id=self.chat_id,
                    start_time=loop_start_time,
                    end_time=current_time,
                )
                if new_message_count > 0:
                    logger.info(
                        f"{self.log_prefix} 检测到{new_message_count}条新消息，聊天ID: {self.chat_id}"
                    )
                    return True, ""
                elapsed_time = (
                    asyncio.get_running_loop().time() - wait_start_time
                )
                if elapsed_time > timeout:
                    logger.warning(
                        f"{self.log_prefix} 等待新消息超时({timeout}秒)，聊天ID: {self.chat_id}"
                    )
                    return False, ""
                if int(elapsed_time) % 15 == 0 and int(elapsed_time) > 0:
                    logger.debug(
                        f"{self.log_prefix} 已等待{int(elapsed_time)}秒，继续等待新消息..."
                    )
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} 等待新消息被中断 (CancelledError)")
            return False, ""
        except Exception as e:
            logger.error(f"{self.log_prefix} 等待新消息时发生错误: {e}")
            return False, f"等待新消息失败: {str(e)}"

    @classmethod
    def get_action_info(cls) -> "ActionInfo":
        """从类属性生成ActionInfo"""
        name = getattr(
            cls, "action_name", cls.__name__.lower().replace("action", "")
        )
        if "." in name:
            logger.error(
                f"Action名称 '{name}' 包含非法字符 '.'，请使用下划线替代"
            )
            raise ValueError(
                f"Action名称 '{name}' 包含非法字符 '.'，请使用下划线替代"
            )
        focus_activation_type = getattr(
            cls, "focus_activation_type", ActionActivationType.ALWAYS
        )
        activation_type = getattr(
            cls, "activation_type", focus_activation_type
        )
        return ActionInfo(
            name=name,
            component_type=ComponentType.ACTION,
            description=getattr(cls, "action_description", "Action动作"),
            activation_type=activation_type,
            activation_keywords=getattr(cls, "activation_keywords", []).copy(),
            keyword_case_sensitive=getattr(
                cls, "keyword_case_sensitive", False
            ),
            parallel_action=getattr(cls, "parallel_action", True),
            random_activation_probability=getattr(
                cls, "random_activation_probability", 0.0
            ),
            action_parameters=getattr(cls, "action_parameters", {}).copy(),
            action_require=getattr(cls, "action_require", []).copy(),
            associated_types=getattr(cls, "associated_types", []).copy(),
        )

    def get_config(self, key: str, default=None):
        """获取插件配置值，使用嵌套键访问"""
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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.plugin_system.base.component_types import (
    ComponentType,
    ToolInfo,
    ToolParamType,
)

if TYPE_CHECKING:
    from src.chat.message_receive.chat_stream import ChatStream

logger = get_logger("基础工具")


class BaseTool(ABC):
    """工具基类，供 LLM 调用"""

    name: str = ""
    description: str = ""
    parameters: List[
        Tuple[str, ToolParamType, str, bool, Optional[List[str]]]
    ] = []
    available_for_llm: bool = False

    def __init__(
        self,
        plugin_config: Optional[dict] = None,
        chat_stream: Optional["ChatStream"] = None,
    ):
        self.plugin_config = plugin_config or {}
        self.chat_stream = chat_stream
        self.chat_id = self.chat_stream.stream_id if self.chat_stream else None
        self.platform = (
            getattr(self.chat_stream, "platform", None)
            if self.chat_stream
            else None
        )

    @classmethod
    def get_tool_definition(cls) -> Dict[str, Any]:
        """获取工具定义，用于 LLM 工具调用"""
        if not cls.name or not cls.description or cls.parameters is None:
            raise NotImplementedError(
                f"工具类 {cls.__name__} 必须定义 name, description 和 parameters 属性"
            )
        return {
            "name": cls.name,
            "description": cls.description,
            "parameters": cls.parameters,
        }

    @classmethod
    def get_tool_info(cls) -> ToolInfo:
        """获取工具信息"""
        if not cls.name or not cls.description or cls.parameters is None:
            raise NotImplementedError(
                f"工具类 {cls.__name__} 必须定义 name, description 和 parameters 属性"
            )
        return ToolInfo(
            name=cls.name,
            tool_description=cls.description,
            enabled=cls.available_for_llm,
            tool_parameters=cls.parameters,
            component_type=ComponentType.TOOL,
        )

    @abstractmethod
    async def execute(self, function_args: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具函数，供 LLM 调用"""
        raise NotImplementedError("子类必须实现 execute 方法")

    async def direct_execute(self, **function_args) -> Dict[str, Any]:
        """直接执行工具函数，供插件调用"""
        required_params = [param[0] for param in self.parameters if param[3]]
        for param_name in required_params:
            if param_name not in function_args:
                raise ValueError(
                    f"工具类 {
                        self.__class__.__name__} 缺少必要参数: {param_name}"
                )
        return await self.execute(function_args)

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

    def get_chat_context(self) -> Optional[Dict[str, Any]]:
        """获取聊天上下文信息"""
        if not self.chat_stream:
            return None
        return {
            "chat_id": self.chat_id,
            "platform": self.platform,
            "stream_id": (
                self.chat_stream.stream_id
                if hasattr(self.chat_stream, "stream_id")
                else None
            ),
            "group_info": getattr(self.chat_stream, "group_info", None),
            "user_info": getattr(self.chat_stream, "user_info", None),
        }

from typing import Optional, Type, TYPE_CHECKING, Any, Dict, List, Tuple

from src.common.logger import get_logger
from src.plugin_system.base.base_tool import BaseTool
from src.plugin_system.base.component_types import ComponentType

if TYPE_CHECKING:
    from src.chat.message_receive.chat_stream import ChatStream

logger = get_logger("工具API")


def get_tool_instance(
    tool_name: str, chat_stream: Optional["ChatStream"] = None
) -> Optional[BaseTool]:
    """获取公开工具实例

    参数:
        tool_name: 工具名称
        chat_stream: 聊天流对象，用于传递聊天上下文信息

    返回:
        Optional[BaseTool]: 工具实例，如果未找到则返回None
    """
    from src.plugin_system.core import component_registry

    tool_info = component_registry.get_component_info(
        tool_name, ComponentType.TOOL
    )
    if tool_info:
        plugin_config = component_registry.get_plugin_config(
            tool_info.plugin_name
        )
    else:
        plugin_config = None
    tool_class: Type[BaseTool] = component_registry.get_component_class(
        tool_name, ComponentType.TOOL
    )
    return tool_class(plugin_config, chat_stream) if tool_class else None


def get_llm_available_tool_definitions() -> List[Tuple[str, Dict[str, Any]]]:
    """获取LLM可用的工具定义列表

    返回:
        List[Tuple[str, Dict[str, Any]]]: 工具定义列表，为[("tool_name", 定义)]
    """
    from src.plugin_system.core import component_registry

    llm_available_tools = component_registry.get_llm_available_tools()
    return [
        (name, tool_class.get_tool_definition())
        for name, tool_class in llm_available_tools.items()
    ]


def get_all_tool_names() -> List[str]:
    """获取所有工具名称列表"""
    from src.plugin_system.core import component_registry

    return component_registry.get_component_names(ComponentType.TOOL)


def get_tool_info(tool_name: str) -> Optional[Any]:
    """获取工具信息"""
    from src.plugin_system.core import component_registry

    return component_registry.get_component_info(tool_name, ComponentType.TOOL)


def is_tool_enabled(tool_name: str) -> bool:
    """检查工具是否启用"""
    from src.plugin_system.core import component_registry

    tool_info = component_registry.get_component_info(
        tool_name, ComponentType.TOOL
    )
    if tool_info is None:
        return False
    return tool_info.enabled


def is_tool_available_for_llm(tool_name: str) -> bool:
    """检查工具是否对LLM可用"""
    from src.plugin_system.core import component_registry

    tool_class = component_registry.get_component_class(
        tool_name, ComponentType.TOOL
    )
    if tool_class is None:
        return False
    return getattr(tool_class, "available_for_llm", False)

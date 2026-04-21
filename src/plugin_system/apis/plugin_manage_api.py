import asyncio
from typing import List, Dict, Any, Optional, Tuple
from src.common.logger import get_logger
from src.plugin_system.core.plugin_manager import plugin_manager

logger = get_logger("插件管理API")


async def get_plugin_list() -> List[Dict[str, Any]]:
    try:
        plugins = []
        for plugin_name, plugin in plugin_manager.loaded_plugins.items():
            plugin_info = {
                "name": plugin_name,
                "version": getattr(plugin, "version", "1.0.0"),
                "description": getattr(plugin, "description", ""),
                "author": getattr(plugin, "author", ""),
                "enabled": getattr(plugin, "enable_plugin", True),
                "loaded": True,
            }
            plugins.append(plugin_info)
        logger.debug(f"[插件管理API] 获取插件列表: {len(plugins)} 个插件")
        return plugins
    except Exception as e:
        logger.error(f"[插件管理API] 获取插件列表失败: {e}")
        return []


async def enable_plugin(plugin_name: str) -> Dict[str, Any]:
    try:
        plugin = plugin_manager.get_plugin_instance(plugin_name)
        if not plugin:
            return {
                "success": False,
                "message": f"插件 '{plugin_name}' 不存在",
            }
        if hasattr(plugin, "on_enable"):
            await plugin.on_enable()
        plugin.enable_plugin = True
        logger.info(f"[插件管理API] 启用插件: {plugin_name}")
        return {"success": True, "message": f"插件 '{plugin_name}' 已启用"}
    except Exception as e:
        error_msg = f"启用插件 '{plugin_name}' 失败: {str(e)}"
        logger.error(f"[插件管理API] {error_msg}")
        return {"success": False, "message": error_msg}


async def disable_plugin(plugin_name: str) -> Dict[str, Any]:
    try:
        plugin = plugin_manager.get_plugin_instance(plugin_name)
        if not plugin:
            return {
                "success": False,
                "message": f"插件 '{plugin_name}' 不存在",
            }
        if hasattr(plugin, "on_disable"):
            await plugin.on_disable()
        plugin.enable_plugin = False
        logger.info(f"[插件管理API] 禁用插件: {plugin_name}")
        return {"success": True, "message": f"插件 '{plugin_name}' 已禁用"}
    except Exception as e:
        error_msg = f"禁用插件 '{plugin_name}' 失败: {str(e)}"
        logger.error(f"[插件管理API] {error_msg}")
        return {"success": False, "message": error_msg}


async def reload_plugin(plugin_name: str) -> bool:
    """重载指定插件

    参数:
        plugin_name: 插件名称

    返回:
        bool: 重载是否成功
    """
    try:
        result = await plugin_manager.reload_registered_plugin(plugin_name)
        if result:
            logger.info(f"[插件管理API] 重载插件: {plugin_name}")
        return result
    except Exception as e:
        logger.error(f"[插件管理API] 重载插件 '{plugin_name}' 失败: {e}")
        return False


async def get_plugin_info(plugin_name: str) -> Optional[Dict[str, Any]]:
    try:
        plugin = plugin_manager.get_plugin_instance(plugin_name)
        if not plugin:
            return None
        info = {
            "name": plugin_name,
            "version": getattr(plugin, "version", "1.0.0"),
            "description": getattr(plugin, "description", ""),
            "author": getattr(plugin, "author", ""),
            "enabled": getattr(plugin, "enable_plugin", True),
            "loaded": True,
            "config": getattr(plugin, "config", {}),
            "dependencies": getattr(plugin, "dependencies", []),
        }
        return info
    except Exception as e:
        logger.error(f"[插件管理API] 获取插件信息失败: {e}")
        return None


async def get_plugin_config(plugin_name: str) -> Optional[Dict[str, Any]]:
    try:
        plugin = plugin_manager.get_plugin_instance(plugin_name)
        if not plugin:
            return None
        return getattr(plugin, "config", {})
    except Exception as e:
        logger.error(f"[插件管理API] 获取插件配置失败: {e}")
        return None


async def set_plugin_config(
    plugin_name: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        plugin = plugin_manager.get_plugin_instance(plugin_name)
        if not plugin:
            return {
                "success": False,
                "message": f"插件 '{plugin_name}' 不存在",
            }
        if hasattr(plugin, "_load_plugin_config"):
            if not isinstance(plugin.config, dict):
                plugin.config = {}
            plugin.config.update(config)
        else:
            plugin.config = config
        logger.info(f"[插件管理API] 更新插件配置: {plugin_name}")
        return {"success": True, "message": f"插件 '{plugin_name}' 配置已更新"}
    except Exception as e:
        error_msg = f"更新插件 '{plugin_name}' 配置失败: {str(e)}"
        logger.error(f"[插件管理API] {error_msg}")
        return {"success": False, "message": error_msg}


async def get_plugin_status(plugin_name: str) -> Dict[str, Any]:
    try:
        plugin = plugin_manager.get_plugin_instance(plugin_name)
        if not plugin:
            return {"exists": False, "enabled": False, "loaded": False}
        return {
            "exists": True,
            "enabled": getattr(plugin, "enable_plugin", True),
            "loaded": True,
        }
    except Exception as e:
        logger.error(f"[插件管理API] 获取插件状态失败: {e}")
        return {"exists": False, "enabled": False, "loaded": False}


async def call_plugin_method(
    plugin_name: str, method_name: str, *args, **kwargs
) -> Dict[str, Any]:
    try:
        plugin = plugin_manager.get_plugin_instance(plugin_name)
        if not plugin:
            return {
                "success": False,
                "message": f"插件 '{plugin_name}' 不存在",
                "result": None,
            }
        method = getattr(plugin, method_name, None)
        if not method or not callable(method):
            return {
                "success": False,
                "message": f"方法 '{method_name}' 不存在",
                "result": None,
            }
        if asyncio.iscoroutinefunction(method):
            result = await method(*args, **kwargs)
        else:
            result = method(*args, **kwargs)
        return {"success": True, "message": "调用成功", "result": result}
    except Exception as e:
        error_msg = f"调用插件方法失败: {str(e)}"
        logger.error(f"[插件管理API] {error_msg}")
        return {"success": False, "message": error_msg, "result": None}


def get_plugin_count() -> int:
    try:
        return len(plugin_manager.loaded_plugins)
    except Exception as e:
        logger.error(f"[插件管理API] 获取插件数量失败: {e}")
        return 0


def get_enabled_plugin_count() -> int:
    try:
        return len(
            [
                p
                for p in plugin_manager.loaded_plugins.values()
                if getattr(p, "enable_plugin", True)
            ]
        )
    except Exception as e:
        logger.error(f"[插件管理API] 获取启用插件数量失败: {e}")
        return 0


def list_loaded_plugins() -> List[str]:
    """列出所有已加载的插件名称

    返回:
        List[str]: 已加载的插件名称列表
    """
    try:
        return plugin_manager.list_loaded_plugins()
    except Exception as e:
        logger.error(f"[插件管理API] 列出已加载插件失败: {e}")
        return []


def list_registered_plugins() -> List[str]:
    """列出所有已注册的插件类名称

    返回:
        List[str]: 已注册的插件类名称列表
    """
    try:
        return plugin_manager.list_registered_plugins()
    except Exception as e:
        logger.error(f"[插件管理API] 列出已注册插件失败: {e}")
        return []


def load_plugin(plugin_name: str) -> Tuple[bool, int]:
    """加载指定插件

    参数:
        plugin_name: 插件名称

    返回:
        Tuple[bool, int]: (是否成功, 组件计数)
    """
    try:
        return plugin_manager.load_registered_plugin_classes(plugin_name)
    except Exception as e:
        logger.error(f"[插件管理API] 加载插件 '{plugin_name}' 失败: {e}")
        return False, 0


async def remove_plugin(plugin_name: str) -> bool:
    """卸载指定插件

    参数:
        plugin_name: 插件名称

    返回:
        bool: 卸载是否成功
    """
    try:
        result = await plugin_manager.remove_registered_plugin(plugin_name)
        if result:
            logger.info(f"[插件管理API] 卸载插件: {plugin_name}")
        return result
    except Exception as e:
        logger.error(f"[插件管理API] 卸载插件 '{plugin_name}' 失败: {e}")
        return False


def rescan_plugin_directory() -> Tuple[int, int]:
    """重新扫描所有插件目录

    返回:
        Tuple[int, int]: (成功扫描数, 失败数)
    """
    try:
        result = plugin_manager.rescan_plugin_directory()
        logger.info(f"[插件管理API] 重新扫描插件目录完成: 成功={result[0]}, 失败={result[1]}")
        return result
    except Exception as e:
        logger.error(f"[插件管理API] 重新扫描插件目录失败: {e}")
        return 0, 0


def add_plugin_directory(directory: str) -> bool:
    """添加插件目录

    参数:
        directory: 插件目录路径

    返回:
        bool: 添加是否成功
    """
    try:
        result = plugin_manager.add_plugin_directory(directory)
        if result:
            logger.info(f"[插件管理API] 添加插件目录: {directory}")
        return result
    except Exception as e:
        logger.error(f"[插件管理API] 添加插件目录 '{directory}' 失败: {e}")
        return False

import asyncio
from typing import List, Dict, Any, Optional
from src.common.logger import get_logger
from src.plugin_system.core.plugin_manager import plugin_manager

logger = get_logger("插件管理API")


async def get_plugin_list() -> List[Dict[str, Any]]:
    try:
        plugins = []
        for plugin in plugin_manager.plugins:
            plugin_info = {
                "name": plugin.plugin_name,
                "version": getattr(plugin, "version", "1.0.0"),
                "description": getattr(plugin, "description", ""),
                "author": getattr(plugin, "author", ""),
                "enabled": getattr(plugin, "is_enabled", True),
                "loaded": getattr(plugin, "is_loaded", False),
            }
            plugins.append(plugin_info)
        logger.debug(f"[PluginManageAPI] 获取插件列表: {len(plugins)} 个插件")
        return plugins
    except Exception as e:
        logger.error(f"[PluginManageAPI] 获取插件列表失败: {e}")
        return []


async def enable_plugin(plugin_name: str) -> Dict[str, Any]:
    try:
        plugin = plugin_manager.get_plugin(plugin_name)
        if not plugin:
            return {
                "success": False,
                "message": f"插件 '{plugin_name}' 不存在",
            }
        if hasattr(plugin, "enable"):
            await plugin.enable()
        else:
            plugin.is_enabled = True
        logger.info(f"[PluginManageAPI] 启用插件: {plugin_name}")
        return {"success": True, "message": f"插件 '{plugin_name}' 已启用"}
    except Exception as e:
        error_msg = f"启用插件 '{plugin_name}' 失败: {str(e)}"
        logger.error(f"[PluginManageAPI] {error_msg}")
        return {"success": False, "message": error_msg}


async def disable_plugin(plugin_name: str) -> Dict[str, Any]:
    try:
        plugin = plugin_manager.get_plugin(plugin_name)
        if not plugin:
            return {
                "success": False,
                "message": f"插件 '{plugin_name}' 不存在",
            }
        if hasattr(plugin, "disable"):
            await plugin.disable()
        else:
            plugin.is_enabled = False
        logger.info(f"[PluginManageAPI] 禁用插件: {plugin_name}")
        return {"success": True, "message": f"插件 '{plugin_name}' 已禁用"}
    except Exception as e:
        error_msg = f"禁用插件 '{plugin_name}' 失败: {str(e)}"
        logger.error(f"[PluginManageAPI] {error_msg}")
        return {"success": False, "message": error_msg}


async def reload_plugin(plugin_name: str) -> Dict[str, Any]:
    try:
        plugin = plugin_manager.get_plugin(plugin_name)
        if not plugin:
            return {
                "success": False,
                "message": f"插件 '{plugin_name}' 不存在",
            }
        if hasattr(plugin, "reload"):
            await plugin.reload()
        else:
            await disable_plugin(plugin_name)
            await enable_plugin(plugin_name)
        logger.info(f"[PluginManageAPI] 重载插件: {plugin_name}")
        return {"success": True, "message": f"插件 '{plugin_name}' 已重载"}
    except Exception as e:
        error_msg = f"重载插件 '{plugin_name}' 失败: {str(e)}"
        logger.error(f"[PluginManageAPI] {error_msg}")
        return {"success": False, "message": error_msg}


async def get_plugin_info(plugin_name: str) -> Optional[Dict[str, Any]]:
    try:
        plugin = plugin_manager.get_plugin(plugin_name)
        if not plugin:
            return None
        info = {
            "name": plugin.plugin_name,
            "version": getattr(plugin, "version", "1.0.0"),
            "description": getattr(plugin, "description", ""),
            "author": getattr(plugin, "author", ""),
            "enabled": getattr(plugin, "is_enabled", True),
            "loaded": getattr(plugin, "is_loaded", False),
            "config": getattr(plugin, "config", {}),
            "dependencies": getattr(plugin, "dependencies", []),
        }
        return info
    except Exception as e:
        logger.error(f"[PluginManageAPI] 获取插件信息失败: {e}")
        return None


async def get_plugin_config(plugin_name: str) -> Optional[Dict[str, Any]]:
    try:
        plugin = plugin_manager.get_plugin(plugin_name)
        if not plugin:
            return None
        return getattr(plugin, "config", {})
    except Exception as e:
        logger.error(f"[PluginManageAPI] 获取插件配置失败: {e}")
        return None


async def set_plugin_config(
    plugin_name: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        plugin = plugin_manager.get_plugin(plugin_name)
        if not plugin:
            return {
                "success": False,
                "message": f"插件 '{plugin_name}' 不存在",
            }
        if hasattr(plugin, "update_config"):
            plugin.update_config(config)
        else:
            plugin.config = config
        logger.info(f"[PluginManageAPI] 更新插件配置: {plugin_name}")
        return {"success": True, "message": f"插件 '{plugin_name}' 配置已更新"}
    except Exception as e:
        error_msg = f"更新插件 '{plugin_name}' 配置失败: {str(e)}"
        logger.error(f"[PluginManageAPI] {error_msg}")
        return {"success": False, "message": error_msg}


async def get_plugin_status(plugin_name: str) -> Dict[str, Any]:
    try:
        plugin = plugin_manager.get_plugin(plugin_name)
        if not plugin:
            return {"exists": False, "enabled": False, "loaded": False}
        return {
            "exists": True,
            "enabled": getattr(plugin, "is_enabled", True),
            "loaded": getattr(plugin, "is_loaded", False),
        }
    except Exception as e:
        logger.error(f"[PluginManageAPI] 获取插件状态失败: {e}")
        return {"exists": False, "enabled": False, "loaded": False}


async def call_plugin_method(
    plugin_name: str, method_name: str, *args, **kwargs
) -> Dict[str, Any]:
    try:
        plugin = plugin_manager.get_plugin(plugin_name)
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
        logger.error(f"[PluginManageAPI] {error_msg}")
        return {"success": False, "message": error_msg, "result": None}


def get_plugin_count() -> int:
    try:
        return len(plugin_manager.plugins)
    except Exception as e:
        logger.error(f"[PluginManageAPI] 获取插件数量失败: {e}")
        return 0


def get_enabled_plugin_count() -> int:
    try:
        return len(
            [
                p
                for p in plugin_manager.plugins
                if getattr(p, "is_enabled", True)
            ]
        )
    except Exception as e:
        logger.error(f"[PluginManageAPI] 获取启用插件数量失败: {e}")
        return 0

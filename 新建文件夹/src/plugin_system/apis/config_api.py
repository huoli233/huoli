from typing import Any
from src.common.logger import get_logger
from src.config.config import global_config

logger = get_logger("config_api")


def get_global_config(key: str, default: Any = None) -> Any:
    keys = key.split(".")
    current = global_config
    try:
        for k in keys:
            if hasattr(current, k):
                current = getattr(current, k)
            elif isinstance(current, dict) and k in current:
                current = current[k]
            else:
                raise KeyError(f"配置中不存在子空间或键 '{k}'")
        return current
    except Exception as e:
        logger.warning(f"[ConfigAPI] 获取全局配置 {key} 失败: {e}")
        return default


def get_plugin_config(
    plugin_config: dict, key: str, default: Any = None
) -> Any:
    keys = key.split(".")
    current = plugin_config
    try:
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            elif hasattr(current, k):
                current = getattr(current, k)
            else:
                raise KeyError(f"配置中不存在子空间或键 '{k}'")
        return current
    except Exception as e:
        logger.warning(f"[ConfigAPI] 获取插件配置 {key} 失败: {e}")
        return default


def get_bot_config() -> dict:
    try:
        bot_config = {}
        if hasattr(global_config, "bot"):
            bot = global_config.bot
            for attr in dir(bot):
                if not attr.startswith("_"):
                    try:
                        value = getattr(bot, attr)
                        if not callable(value):
                            bot_config[attr] = value
                    except Exception:
                        continue
        return bot_config
    except Exception as e:
        logger.error(f"[ConfigAPI] 获取机器人配置失败: {e}")
        return {}


def get_personality_config() -> dict:
    try:
        personality_config = {}
        if hasattr(global_config, "personality"):
            personality = global_config.personality
            for attr in dir(personality):
                if not attr.startswith("_"):
                    try:
                        value = getattr(personality, attr)
                        if not callable(value):
                            personality_config[attr] = value
                    except Exception:
                        continue
        return personality_config
    except Exception as e:
        logger.error(f"[ConfigAPI] 获取人格配置失败: {e}")
        return {}


def get_chat_config() -> dict:
    try:
        chat_config = {}
        if hasattr(global_config, "chat"):
            chat = global_config.chat
            for attr in dir(chat):
                if not attr.startswith("_"):
                    try:
                        value = getattr(chat, attr)
                        if not callable(value):
                            chat_config[attr] = value
                    except Exception:
                        continue
        return chat_config
    except Exception as e:
        logger.error(f"[ConfigAPI] 获取聊天配置失败: {e}")
        return {}


def get_memory_config() -> dict:
    try:
        memory_config = {}
        if hasattr(global_config, "memory"):
            memory = global_config.memory
            for attr in dir(memory):
                if not attr.startswith("_"):
                    try:
                        value = getattr(memory, attr)
                        if not callable(value):
                            memory_config[attr] = value
                    except Exception:
                        continue
        return memory_config
    except Exception as e:
        logger.error(f"[ConfigAPI] 获取记忆配置失败: {e}")
        return {}


def get_expression_config() -> dict:
    try:
        expression_config = {}
        if hasattr(global_config, "expression"):
            expression = global_config.expression
            for attr in dir(expression):
                if not attr.startswith("_"):
                    try:
                        value = getattr(expression, attr)
                        if not callable(value):
                            expression_config[attr] = value
                    except Exception:
                        continue
        return expression_config
    except Exception as e:
        logger.error(f"[ConfigAPI] 获取表达配置失败: {e}")
        return {}


def is_feature_enabled(feature_name: str) -> bool:
    try:
        value = get_global_config(feature_name, False)
        return bool(value)
    except Exception as e:
        logger.warning(
            f"[ConfigAPI] 检查功能开关失败: {feature_name}, error={e}"
        )
        return False

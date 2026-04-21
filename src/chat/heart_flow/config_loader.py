import os
import yaml
from typing import Dict, List, Any, Optional
from src.common.logger import get_logger

logger = get_logger("心流配置加载")

_HEARTFLOW_CONFIG_CACHE: Dict[str, Any] = {}


def _get_heartflow_config_path(relative_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)


def load_heartflow_yaml_config(
    config_name: str, relative_path: str, use_cache: bool = True
) -> Dict[str, Any]:
    if use_cache and config_name in _HEARTFLOW_CONFIG_CACHE:
        return _HEARTFLOW_CONFIG_CACHE[config_name]
    config_path = _get_heartflow_config_path(relative_path)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if use_cache:
            _HEARTFLOW_CONFIG_CACHE[config_name] = config
        return config
    except FileNotFoundError:
        logger.warning(f"配置文件不存在: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"配置文件解析错误: {config_path} | {e}")
        return {}


def get_heartflow_config() -> Dict[str, Any]:
    return load_heartflow_yaml_config(
        "heartflow_config", "configs/heartflow_config.yaml"
    )


def get_waiting_thoughts() -> Dict[str, List[str]]:
    config = get_heartflow_config()
    inner_voice = config.get("inner_voice", {})
    return inner_voice.get(
        "waiting_thoughts",
        {
            "short": ["对方可能在忙吧...", "再等等看", "不知道在做什么呢"],
            "medium": ["等了一会了...", "是不是忘记回复了？", "嗯...还没消息"],
            "long": ["等了挺久了", "要不要主动说点什么...", "快到时间了"],
        },
    )


def get_internal_action_names() -> List[str]:
    config = get_heartflow_config()
    decision = config.get("decision", {})
    return decision.get(
        "internal_action_names",
        ["complete_talk", "reply", "wait_time", "wait", "listening"],
    )


def get_mention_patterns() -> List[str]:
    config = get_heartflow_config()
    return config.get("mention_patterns", ["@", "提到", "说", "觉得"])


def clear_heartflow_config_cache(config_name: Optional[str] = None):
    global _HEARTFLOW_CONFIG_CACHE
    if config_name:
        _HEARTFLOW_CONFIG_CACHE.pop(config_name, None)
    else:
        _HEARTFLOW_CONFIG_CACHE.clear()

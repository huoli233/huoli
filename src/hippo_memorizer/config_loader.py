import os
import time as _time
import yaml
from typing import Dict, List, Any, Optional
from src.common.logger import get_logger

logger = get_logger("海马配置加载")

_HIPPO_CONFIG_CACHE: Dict[str, Any] = {}
_HIPPO_CONFIG_TIMESTAMPS: Dict[str, float] = {}
_CACHE_TTL_SEC = 300.0


def _get_hippo_config_path(relative_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)


def load_hippo_yaml_config(
    config_name: str, relative_path: str, use_cache: bool = True
) -> Dict[str, Any]:
    if use_cache and config_name in _HIPPO_CONFIG_CACHE:
        if (
            _time.time() - _HIPPO_CONFIG_TIMESTAMPS.get(config_name, 0)
            < _CACHE_TTL_SEC
        ):
            return _HIPPO_CONFIG_CACHE[config_name]
        _HIPPO_CONFIG_CACHE.pop(config_name, None)
        _HIPPO_CONFIG_TIMESTAMPS.pop(config_name, None)
    config_path = _get_hippo_config_path(relative_path)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if use_cache:
            _HIPPO_CONFIG_CACHE[config_name] = config
            _HIPPO_CONFIG_TIMESTAMPS[config_name] = _time.time()
        return config
    except FileNotFoundError:
        logger.warning(f"配置文件不存在: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"配置文件解析错误: {config_path} | {e}")
        return {}


def get_hippo_config() -> Dict[str, Any]:
    return load_hippo_yaml_config("hippo_config", "configs/hippo_config.yaml")


def get_hippo_settings() -> Dict[str, Any]:
    config = get_hippo_config()
    return config.get(
        "hippo_config",
        {
            "batch_size": 30,
            "batch_interval": 300.0,
            "min_condense_interval": 600.0,
            "max_buffer_size": 50,
            "max_age_minutes": 30,
            "max_summaries": 100,
        },
    )


def get_action_templates() -> Dict[str, str]:
    config = get_hippo_config()
    return config.get("action_templates", {})


def get_condense_prompts() -> Dict[str, Any]:
    config = get_hippo_config()
    return config.get("condense_prompts", {})


def get_condense_prompt(prompt_name: str) -> str:
    prompts = get_condense_prompts()
    prompt_config = prompts.get(prompt_name, {})
    return prompt_config.get("template", "")


def get_batch_size() -> int:
    return get_hippo_settings().get("batch_size", 30)


def get_batch_interval() -> float:
    return get_hippo_settings().get("batch_interval", 300.0)


def get_min_condense_interval() -> float:
    return get_hippo_settings().get("min_condense_interval", 600.0)


def get_max_buffer_size() -> int:
    return get_hippo_settings().get("max_buffer_size", 50)


def get_max_age_minutes() -> int:
    return get_hippo_settings().get("max_age_minutes", 30)


def get_max_summaries() -> int:
    return get_hippo_settings().get("max_summaries", 100)


def clear_hippo_config_cache(config_name: Optional[str] = None):
    global _HIPPO_CONFIG_CACHE
    if config_name:
        _HIPPO_CONFIG_CACHE.pop(config_name, None)
    else:
        _HIPPO_CONFIG_CACHE.clear()

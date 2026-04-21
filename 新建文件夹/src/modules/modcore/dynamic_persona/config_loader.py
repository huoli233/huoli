import os
import yaml
from typing import Dict, List, Any, Optional
from src.common.logger import get_logger

logger = get_logger("config_loader")

_CONFIG_CACHE: Dict[str, Any] = {}


def _get_config_path(relative_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)


def load_yaml_config(
    config_name: str, relative_path: str, use_cache: bool = True
) -> Dict[str, Any]:
    if use_cache and config_name in _CONFIG_CACHE:
        return _CONFIG_CACHE[config_name]
    config_path = _get_config_path(relative_path)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if use_cache:
            _CONFIG_CACHE[config_name] = config
        return config
    except FileNotFoundError:
        logger.warning(f"配置文件不存在: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"配置文件解析错误: {config_path} | {e}")
        return {}


def get_emotion_config() -> Dict[str, Any]:
    return load_yaml_config("emotion_config", "configs/emotion_config.yaml")


def get_persona_config() -> Dict[str, Any]:
    return load_yaml_config("persona_config", "configs/persona_config.yaml")


def get_relation_thresholds() -> List[tuple]:
    """从配置文件获取关系等级阈值"""
    config = get_emotion_config()
    thresholds = config.get("relation_thresholds", [])
    if thresholds:
        return [(t["threshold"], t["label"]) for t in thresholds]
    return []


def get_relation_stage_by_value(value: float) -> str:
    """根据数值获取关系阶段标签"""
    thresholds = get_relation_thresholds()
    for threshold, label in thresholds:
        if value >= threshold:
            return label
    return "未知"


def get_all_relation_stages() -> List[str]:
    """获取所有关系阶段标签列表"""
    thresholds = get_relation_thresholds()
    return [label for _, label in thresholds]


def get_emotion_config_values() -> Dict[str, float]:
    config = get_emotion_config()
    emotion_cfg = config.get("emotion_config", {})
    return {
        "affection_max": emotion_cfg.get("affection_max", 100.0),
        "affection_min": emotion_cfg.get("affection_min", -100.0),
        "trust_max": emotion_cfg.get("trust_max", 100.0),
        "trust_min": emotion_cfg.get("trust_min", -100.0),
        "annoyance_max": emotion_cfg.get("annoyance_max", 100.0),
        "annoyance_min": emotion_cfg.get("annoyance_min", 0.0),
        "trauma_max": emotion_cfg.get("trauma_max", 10.0),
        "trauma_min": emotion_cfg.get("trauma_min", 0.0),
        "trust_threshold": emotion_cfg.get("trust_threshold", 30.0),
        "block_threshold": emotion_cfg.get("block_threshold", 80.0),
        "block_duration_base": emotion_cfg.get("block_duration_base", 60.0),
        "block_duration_max": emotion_cfg.get("block_duration_max", 3600.0),
        "global_rage_threshold": emotion_cfg.get(
            "global_rage_threshold", 150.0
        ),
    }


def get_response_mode_keywords() -> Dict[str, Dict[str, List[str]]]:
    config = get_emotion_config()
    return config.get("response_mode_keywords", {})


def clear_config_cache(config_name: Optional[str] = None):
    global _CONFIG_CACHE
    if config_name:
        _CONFIG_CACHE.pop(config_name, None)
    else:
        _CONFIG_CACHE.clear()

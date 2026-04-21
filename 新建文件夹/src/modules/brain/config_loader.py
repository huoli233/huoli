import os
import yaml
from typing import Dict, List, Any, Optional
from src.common.logger import get_logger

logger = get_logger("brain_config_loader")

_BRAIN_CONFIG_CACHE: Dict[str, Any] = {}


def _get_brain_config_path(relative_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)


def load_brain_yaml_config(
    config_name: str, relative_path: str, use_cache: bool = True
) -> Dict[str, Any]:
    if use_cache and config_name in _BRAIN_CONFIG_CACHE:
        return _BRAIN_CONFIG_CACHE[config_name]
    config_path = _get_brain_config_path(relative_path)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if use_cache:
            _BRAIN_CONFIG_CACHE[config_name] = config
        return config
    except FileNotFoundError:
        logger.warning(f"配置文件不存在: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"配置文件解析错误: {config_path} | {e}")
        return {}


def get_decision_config() -> Dict[str, Any]:
    return load_brain_yaml_config(
        "decision_config", "configs/decision_config.yaml"
    )


def get_recall_keywords() -> List[str]:
    config = get_decision_config()
    keywords = config.get("recall_keywords", [])
    return keywords


def get_mention_patterns() -> List[str]:
    config = get_decision_config()
    patterns = config.get("mention_patterns", [])
    return patterns


def get_high_emotion_types() -> List[str]:
    config = get_decision_config()
    return config.get("high_emotion_types", [])


def get_relationship_types() -> Dict[str, List[str]]:
    config = get_decision_config()
    return config.get(
        "relationship_types",
        {
            "intimate": [],
            "familiar": [],
            "stranger": [],
            "negative": [],
        },
    )


def is_intimate_relationship(relationship: str) -> bool:
    rel_types = get_relationship_types()
    return relationship in rel_types.get("intimate", [])


def is_familiar_relationship(relationship: str) -> bool:
    rel_types = get_relationship_types()
    return relationship in rel_types.get("familiar", [])


def get_decision_weights() -> Dict[str, float]:
    config = get_decision_config()
    weights = config.get("decision_weights", {})
    if not weights:
        return {
            "confidence": 0.25,
            "risk": 0.20,
            "interest": 0.20,
            "urgency": 0.15,
            "social": 0.10,
            "mood": 0.10,
        }
    return weights


def get_confidence_thresholds() -> Dict[str, float]:
    config = get_decision_config()
    thresholds = config.get("confidence_thresholds", {})
    if not thresholds:
        return {"high": 0.7, "medium": 0.5, "low": 0.3}
    return thresholds


def clear_brain_config_cache(config_name: Optional[str] = None):
    global _BRAIN_CONFIG_CACHE
    if config_name:
        _BRAIN_CONFIG_CACHE.pop(config_name, None)
    else:
        _BRAIN_CONFIG_CACHE.clear()

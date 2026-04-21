import os
import yaml
from typing import Dict, List, Any, Optional
from src.common.logger import get_logger

logger = get_logger("kg_config_loader")

_KG_CONFIG_CACHE: Dict[str, Any] = {}


def _get_kg_config_path(relative_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)


def load_kg_yaml_config(
    config_name: str, relative_path: str, use_cache: bool = True
) -> Dict[str, Any]:
    if use_cache and config_name in _KG_CONFIG_CACHE:
        return _KG_CONFIG_CACHE[config_name]
    config_path = _get_kg_config_path(relative_path)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if use_cache:
            _KG_CONFIG_CACHE[config_name] = config
        return config
    except FileNotFoundError:
        logger.warning(f"配置文件不存在: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"配置文件解析错误: {config_path} | {e}")
        return {}


def get_kg_config() -> Dict[str, Any]:
    return load_kg_yaml_config(
        "knowledge_graph_config", "configs/knowledge_graph_config.yaml"
    )


def get_kg_settings() -> Dict[str, Any]:
    config = get_kg_config()
    return config.get(
        "knowledge_graph",
        {
            "capacity": 500,
            "min_text_length": 8,
            "max_body_length": 500,
            "min_keywords": 1,
            "eviction_ratio": 0.25,
        },
    )


def get_stopwords() -> List[str]:
    config = get_kg_config()
    return config.get(
        "stopwords",
        [
            "的",
            "了",
            "是",
            "在",
            "我",
            "有",
            "和",
            "就",
            "不",
            "人",
            "都",
            "一",
            "这",
            "中",
            "到",
            "那",
            "个",
            "你",
            "他",
            "她",
            "它",
            "们",
            "会",
            "对",
            "也",
            "很",
            "要",
            "能",
            "上",
            "下",
            "啊",
            "呢",
            "吗",
            "吧",
            "哦",
        ],
    )


def get_separators() -> List[str]:
    config = get_kg_config()
    return config.get(
        "separators",
        ["，", "。", "！", "？", "；", "\n", ",", ".", "!", "?", ";"],
    )


def get_keyword_extraction_settings() -> Dict[str, Any]:
    config = get_kg_config()
    return config.get(
        "keyword_extraction",
        {
            "min_segment_length": 2,
            "max_segment_length": 6,
            "bigram_enabled": True,
        },
    )


def get_capacity() -> int:
    return get_kg_settings().get("capacity", 500)


def get_eviction_ratio() -> float:
    return get_kg_settings().get("eviction_ratio", 0.25)


def clear_kg_config_cache(config_name: Optional[str] = None):
    global _KG_CONFIG_CACHE
    if config_name:
        _KG_CONFIG_CACHE.pop(config_name, None)
    else:
        _KG_CONFIG_CACHE.clear()

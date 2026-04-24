from __future__ import annotations

from typing import Any, Dict, List


def social_value_module_view(module_name: str) -> Dict[str, Any]:
    try:
        from src.config.core_config_engine import get_core_config

        return get_core_config().resolve_module_view(module_name).values
    except Exception:
        return {}


def social_value_float(module_name: str, key: str, fallback: float) -> float:
    try:
        return float(social_value_module_view(module_name).get(key, fallback))
    except Exception:
        return fallback


def social_value_int(module_name: str, key: str, fallback: int) -> int:
    try:
        return int(social_value_module_view(module_name).get(key, fallback))
    except Exception:
        return fallback


def social_value_dict(
    module_name: str, key: str, fallback: Dict[str, Any]
) -> Dict[str, Any]:
    value = social_value_module_view(module_name).get(key, fallback)
    return value if isinstance(value, dict) else fallback


def social_value_list(
    module_name: str, key: str, fallback: List[Any]
) -> List[Any]:
    value = social_value_module_view(module_name).get(key, fallback)
    return value if isinstance(value, list) else fallback

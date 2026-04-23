from __future__ import annotations

from typing import Any, Dict, List


def recall_module_view(module_name: str) -> Dict[str, Any]:
    try:
        from src.config.core_config_engine import get_core_config

        return get_core_config().resolve_module_view(module_name).values
    except Exception:
        return {}


def recall_float(module_name: str, key: str, fallback: float) -> float:
    try:
        return float(recall_module_view(module_name).get(key, fallback))
    except Exception:
        return fallback


def recall_int(module_name: str, key: str, fallback: int) -> int:
    try:
        return int(recall_module_view(module_name).get(key, fallback))
    except Exception:
        return fallback


def recall_bool(module_name: str, key: str, fallback: bool) -> bool:
    try:
        value = recall_module_view(module_name).get(key, fallback)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    except Exception:
        return fallback


def recall_list(module_name: str, key: str, fallback: List[Any]) -> List[Any]:
    value = recall_module_view(module_name).get(key, fallback)
    return value if isinstance(value, list) else fallback


def recall_dict(module_name: str, key: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    value = recall_module_view(module_name).get(key, fallback)
    return value if isinstance(value, dict) else fallback

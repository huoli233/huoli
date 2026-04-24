from __future__ import annotations

from typing import Any, Dict, List


def identity_module_view(module_name: str) -> Dict[str, Any]:
    try:
        from src.config.core_config_engine import get_core_config

        return get_core_config().resolve_module_view(module_name).values
    except Exception:
        return {}


def identity_float(module_name: str, key: str, fallback: float) -> float:
    try:
        return float(identity_module_view(module_name).get(key, fallback))
    except Exception:
        return fallback


def identity_int(module_name: str, key: str, fallback: int) -> int:
    try:
        return int(identity_module_view(module_name).get(key, fallback))
    except Exception:
        return fallback


def identity_bool(module_name: str, key: str, fallback: bool) -> bool:
    try:
        value = identity_module_view(module_name).get(key, fallback)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    except Exception:
        return fallback


def identity_list(module_name: str, key: str, fallback: List[Any]) -> List[Any]:
    value = identity_module_view(module_name).get(key, fallback)
    return value if isinstance(value, list) else fallback


def identity_dict(module_name: str, key: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    value = identity_module_view(module_name).get(key, fallback)
    return value if isinstance(value, dict) else fallback

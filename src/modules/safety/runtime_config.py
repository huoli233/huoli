from __future__ import annotations

from typing import Any, Dict


def safety_module_view(module_name: str) -> Dict[str, Any]:
    try:
        from src.config.core_config_engine import get_core_config

        return get_core_config().resolve_module_view(module_name).values
    except Exception:
        return {}


def safety_float(module_name: str, key: str, fallback: float) -> float:
    try:
        return float(safety_module_view(module_name).get(key, fallback))
    except Exception:
        return fallback


def safety_int(module_name: str, key: str, fallback: int) -> int:
    try:
        return int(safety_module_view(module_name).get(key, fallback))
    except Exception:
        return fallback


def safety_bool(module_name: str, key: str, fallback: bool) -> bool:
    try:
        return bool(safety_module_view(module_name).get(key, fallback))
    except Exception:
        return fallback

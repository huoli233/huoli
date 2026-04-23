from __future__ import annotations

from typing import Any, Dict


def memory_module_view(module_name: str) -> Dict[str, Any]:
    try:
        from src.config.core_config_engine import get_core_config

        return get_core_config().resolve_module_view(module_name).values
    except Exception:
        return {}


def memory_float(module_name: str, key: str, fallback: float) -> float:
    try:
        return float(memory_module_view(module_name).get(key, fallback))
    except Exception:
        return fallback


def memory_int(module_name: str, key: str, fallback: int) -> int:
    try:
        return int(memory_module_view(module_name).get(key, fallback))
    except Exception:
        return fallback

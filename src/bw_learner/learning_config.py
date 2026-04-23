from __future__ import annotations

from typing import Any, List


def adaptive_learning_view() -> dict[str, Any]:
    try:
        from src.config.core_config_engine import get_core_config

        return get_core_config().resolve_module_view("adaptive_learning").values
    except Exception:
        return {}


def adaptive_float(key: str, fallback: float) -> float:
    try:
        return float(adaptive_learning_view().get(key, fallback))
    except Exception:
        return fallback


def adaptive_int(key: str, fallback: int) -> int:
    try:
        return int(adaptive_learning_view().get(key, fallback))
    except Exception:
        return fallback


def adaptive_list(key: str, fallback: List[int]) -> List[int]:
    raw = adaptive_learning_view().get(key, fallback)
    if not isinstance(raw, list):
        return list(fallback)
    result: List[int] = []
    for item in raw:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result or list(fallback)

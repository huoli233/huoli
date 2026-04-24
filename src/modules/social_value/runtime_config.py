from __future__ import annotations

from typing import Any, Dict, List
from src.modules.social_value.models import AlgorithmParams, CategoryConfig


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


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def social_intent_multipliers() -> Dict[str, float]:
    raw = social_value_dict("social_calculator", "intents", {})
    return {str(key): _as_float(value, 1.0) for key, value in raw.items()}


def social_category_configs() -> Dict[str, CategoryConfig]:
    raw_categories = social_value_dict("social_calculator", "categories", {})
    built: Dict[str, CategoryConfig] = {}
    default = CategoryConfig()
    for cat_name, raw_value in raw_categories.items():
        cat_data = raw_value if isinstance(raw_value, dict) else {}
        intent_multipliers = (
            cat_data.get("intent_multipliers", {})
            if isinstance(cat_data.get("intent_multipliers", {}), dict)
            else {}
        )
        built[cat_name] = CategoryConfig(
            base_score=_as_float(
                cat_data.get("base_score", default.base_score),
                default.base_score,
            ),
            max_change_per_reply=_as_float(
                cat_data.get("max_change_per_reply", default.max_change_per_reply),
                default.max_change_per_reply,
            ),
            acceleration_threshold=_as_float(
                cat_data.get("acceleration_threshold", default.acceleration_threshold),
                default.acceleration_threshold,
            ),
            acceleration_multiplier=_as_float(
                cat_data.get(
                    "acceleration_multiplier", default.acceleration_multiplier
                ),
                default.acceleration_multiplier,
            ),
            decay_rate=_as_float(
                cat_data.get("decay_rate", default.decay_rate),
                default.decay_rate,
            ),
            trauma_impact=_as_float(
                cat_data.get("trauma_impact", default.trauma_impact),
                default.trauma_impact,
            ),
            intent_multipliers={
                str(key): _as_float(value, 1.0)
                for key, value in intent_multipliers.items()
            },
        )
    if "neutral" not in built:
        built["neutral"] = CategoryConfig()
    _apply_personality_adjustments(built)
    return built


def _apply_personality_adjustments(
    categories: Dict[str, CategoryConfig]
) -> None:
    factors = social_value_module_view("personality_factors")
    sensitivity = _as_float(factors.get("sensitivity", 0.6), 0.6)
    tolerance = _as_float(factors.get("tolerance", 0.4), 0.4)
    reactiveness = _as_float(factors.get("reactiveness", 0.5), 0.5)
    recovery_speed = _as_float(factors.get("recovery_speed", 0.5), 0.5)
    social_warmth = _as_float(factors.get("social_warmth", 0.6), 0.6)
    adjustments = {
        "friendly": {
            "base_score_factor": 0.6 + social_warmth * 0.8,
            "max_change_factor": 0.7 + reactiveness * 0.6,
            "threshold_offset": -sensitivity * 0.1,
            "multiplier_factor": 0.8 + reactiveness * 0.4,
            "decay_factor": 0.7 + recovery_speed * 0.6,
        },
        "neutral": {
            "base_score_factor": 1.0,
            "max_change_factor": 0.8 + reactiveness * 0.4,
            "threshold_offset": 0.0,
            "multiplier_factor": 1.0,
            "decay_factor": 0.8 + recovery_speed * 0.4,
        },
        "unfriendly": {
            "base_score_factor": 0.7 + sensitivity * 0.6,
            "max_change_factor": 0.7 + reactiveness * 0.6,
            "threshold_offset": -sensitivity * 0.1,
            "multiplier_factor": 0.8 + reactiveness * 0.4,
            "decay_factor": 0.7 + recovery_speed * 0.6,
        },
        "hostile": {
            "base_score_factor": 0.7 + sensitivity * 0.6,
            "max_change_factor": 0.7 + reactiveness * 0.6,
            "threshold_offset": -sensitivity * 0.15,
            "multiplier_factor": 0.8 + reactiveness * 0.4,
            "decay_factor": 0.7 + recovery_speed * 0.6,
        },
        "harassing": {
            "base_score_factor": 0.6 + (1 - tolerance) * 0.8,
            "max_change_factor": 0.6 + reactiveness * 0.8,
            "threshold_offset": -sensitivity * 0.15,
            "multiplier_factor": 0.7 + reactiveness * 0.6,
            "decay_factor": 0.6 + recovery_speed * 0.8,
        },
    }
    for cat_name, adj in adjustments.items():
        category = categories.get(cat_name)
        if category is None:
            continue
        category.base_score *= adj["base_score_factor"]
        category.max_change_per_reply *= adj["max_change_factor"]
        category.acceleration_threshold = max(
            0.3,
            min(
                0.99,
                category.acceleration_threshold + adj["threshold_offset"],
            ),
        )
        category.acceleration_multiplier *= adj["multiplier_factor"]
        category.decay_rate *= adj["decay_factor"]


def social_algorithm_params(
    fine_type: str, intent: str = "other"
) -> AlgorithmParams:
    categories = social_category_configs()
    raw_types = social_value_dict("social_calculator", "types", {})
    type_cfg = (
        raw_types.get(fine_type, {})
        if isinstance(raw_types.get(fine_type, {}), dict)
        else {}
    )
    category_name = str(type_cfg.get("category", "neutral"))
    category_cfg = categories.get(category_name) or categories.get(
        "neutral", CategoryConfig()
    )
    override = bool(type_cfg.get("override", False))
    base_score = (
        _as_float(
            type_cfg.get("base_score", category_cfg.base_score),
            category_cfg.base_score,
        )
        if override
        else category_cfg.base_score
    )
    acceleration_multiplier = (
        _as_float(
            type_cfg.get(
                "acceleration_multiplier",
                category_cfg.acceleration_multiplier,
            ),
            category_cfg.acceleration_multiplier,
        )
        if override
        else category_cfg.acceleration_multiplier
    )
    type_bonus = _as_float(type_cfg.get("bonus", 1.0), 1.0)
    intent_multiplier = social_intent_multipliers().get(intent, 1.0)
    category_intent = category_cfg.intent_multipliers.get(intent)
    if category_intent is not None:
        intent_multiplier = category_intent
    return AlgorithmParams(
        category=category_name,
        base_score=base_score,
        intent_multiplier=intent_multiplier,
        max_change_per_reply=category_cfg.max_change_per_reply,
        acceleration_threshold=category_cfg.acceleration_threshold,
        acceleration_multiplier=acceleration_multiplier,
        decay_rate=category_cfg.decay_rate,
        trauma_impact=category_cfg.trauma_impact,
        type_bonus=type_bonus,
    )

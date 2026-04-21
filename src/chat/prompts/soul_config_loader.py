import os
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.common.logger import get_logger

logger = get_logger("灵魂引擎配置")

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs"
)
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "soul_engine.yaml")

_cache: Dict[str, Any] = {}
_cache_ts: float = 0.0
_CACHE_TTL: float = 300.0


def _load_raw() -> Dict[str, Any]:
    if not os.path.exists(_CONFIG_FILE):
        logger.warning(f"灵魂引擎配置文件不存在: {_CONFIG_FILE}")
        return _get_fallback_config()
    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or "soul_engine" not in data:
            logger.warning("灵魂引擎配置格式异常，使用内置默认值")
            return _get_fallback_config()
        return data.get("soul_engine", {})
    except Exception as e:
        logger.error(f"灵魂引擎配置加载失败: {e}")
        return _get_fallback_config()


def _get_fallback_config() -> Dict[str, Any]:
    return {
        "meme_library": {"base_words": [], "abbreviations": []},
        "sarcasm_templates": {"low": [], "mid": [], "high": []},
        "warm_templates": [],
        "scene_matrix": {},
        "response_modes": {},
        "thresholds": {
            "cold_rejection_min": 35,
            "irritated_min": 15,
            "trauma_defense_min": 5,
        },
        "diversity": {
            "max_history": 12,
            "warning_threshold": 0.55,
            "suggestion_threshold": 0.35,
        },
        "injection": {
            "meme_cache_ttl": 180,
            "max_meme_pick": 18,
            "min_meme_pick": 8,
        },
    }


def get_config() -> Dict[str, Any]:
    now = time.time()
    global _cache_ts, _cache
    if _cache and (now - _cache_ts) < _CACHE_TTL:
        return _cache
    _cache = _load_raw()
    _cache_ts = now
    return _cache


def refresh() -> None:
    global _cache_ts, _cache
    _cache = _load_raw()
    _cache_ts = time.time()


def get_all_meme_words(limit: int = 50) -> List[str]:
    cfg = get_config()
    meme_lib = cfg.get("meme_library", {})
    words: List[str] = []
    for entry in meme_lib.get("base_words", []):
        if isinstance(entry, dict):
            w = entry.get("word", "")
        else:
            w = str(entry)
        if w and len(w) <= 15:
            words.append(w)
    for entry in meme_lib.get("abbreviations", []):
        w = str(entry).strip()
        if w and len(w) <= 12 and w not in words:
            words.append(w)
    if limit > 0:
        words = words[:limit]
    return words


def get_sarcasm_templates(level: str = "mid") -> List[str]:
    cfg = get_config()
    templates = cfg.get("sarcasm_templates", {})
    pool = templates.get(level, [])
    if not pool and level == "mid":
        pool = templates.get("low", [])
    if not pool:
        pool = templates.get("high", [])
    return [str(t) for t in pool]


def get_warm_templates(count: int = 10) -> List[str]:
    cfg = get_config()
    raw = cfg.get("warm_templates", [])
    return [str(t) for t in raw[:count]]


def get_scene_matrix() -> Dict[str, Dict]:
    cfg = get_config()
    matrix = cfg.get("scene_matrix", {})
    result: Dict[str, Dict] = {}
    for scene_name, scene_data in matrix.items():
        if not isinstance(scene_data, dict):
            continue
        result[scene_name] = {
            "tone": str(scene_data.get("tone", "neutral")),
            "length_range": tuple(scene_data.get("length_range", [15, 120])),
            "style_hints": list(scene_data.get("style_hints", [])),
            "forbidden_patterns": list(scene_data.get("forbidden_patterns", [])),
            "condition": str(scene_data.get("condition", "")),
        }
    return result


def match_scene(
    annoyance_val: float,
    affection_val: float,
    trauma_score: float,
    hour_now: int,
    relation_label: str,
    is_familiar: bool = False,
    is_stranger: bool = False,
) -> Dict[str, Any]:
    matrix = get_scene_matrix()
    is_midnight = 1 <= hour_now < 5
    is_late_night = (hour_now >= 0 and hour_now < 6) or (hour_now >= 23)
    is_morning = 6 <= hour_now < 11

    priority_order = [
        ("trauma_defense", trauma_score >= 5),
        ("hostile_midnight", annoyance_val >= 45 and is_midnight),
        ("hostile_daytime", annoyance_val >= 45 and not is_midnight),
        ("irritated_midnight", annoyance_val >= 20 and is_midnight),
        ("irritated_daytime", annoyance_val >= 20 and not is_midnight),
        ("drowsy_irritable", is_midnight and annoyance_val >= 8),
        ("late_casual", is_late_night and annoyance_val < 10),
        ("warm_casual", is_familiar and affection_val >= 30),
        ("cold_distant", is_stranger and affection_val < -10),
        ("morning_lazy", is_morning),
        ("normal_casual", True),
    ]
    for scene_name, condition in priority_order:
        if scene_name in matrix and condition:
            matched = dict(matrix[scene_name])
            matched["scene_name"] = scene_name
            return matched
    default = matrix.get("normal_casual", {
        "tone": "normal_casual",
        "length_range": (15, 120),
        "style_hints": ["正常说话", "自然随意"],
        "forbidden_patterns": ["客服腔", "机械式回复"],
        "condition": "default",
    })
    default["scene_name"] = "normal_casual"
    return default


def get_response_mode_template(mode: str, **extra_vars) -> str:
    cfg = get_config()
    modes = cfg.get("response_modes", {})
    mode_data = modes.get(mode, {})
    template_str = mode_data.get("template", "")
    fmt_vars = {}
    if mode == "cold_rejection":
        high_pool = get_sarcasm_templates("high")
        sarcasm_examples = " | ".join(random.sample(high_pool, min(6, len(high_pool))))
        fmt_vars["sarcasm_examples"] = sarcasm_examples
    elif mode == "irritated":
        mid_pool = get_sarcasm_templates("mid")
        mid_sarcasm = " | ".join(random.sample(mid_pool, min(5, len(mid_pool))))
        fmt_vars["mid_sarcasm_examples"] = mid_sarcasm
    elif mode == "dismissive":
        fmt_vars["annoyance_value"] = extra_vars.get("annoyance_value", 0.5)
    fmt_vars.update(extra_vars)
    try:
        template_str = template_str.format(**fmt_vars)
    except (KeyError, IndexError):
        import re
        template_str = re.sub(r'\{[^}]+\}', '', template_str)
    return template_str


def get_thresholds() -> Dict[str, float]:
    cfg = get_config()
    thresholds = cfg.get("thresholds", {})
    return {
        "cold_rejection": float(thresholds.get("cold_rejection_min", 35)),
        "irritated": float(thresholds.get("irritated_min", 15)),
        "trauma_defense": float(thresholds.get("trauma_defense_min", 5)),
        "rhythm_pressure": float(thresholds.get("rhythm_pressure_min", 35)),
        "emotion_protection": float(thresholds.get("emotion_protection_trauma_min", 35)),
        "energy_high": float(thresholds.get("energy_high", 55)),
        "energy_medium": float(thresholds.get("energy_medium", 35)),
        "energy_low": float(thresholds.get("energy_low", 18)),
    }


def get_diversity_config() -> Dict[str, Any]:
    cfg = get_config()
    return cfg.get("diversity", {})


def get_injection_config() -> Dict[str, Any]:
    cfg = get_config()
    return cfg.get("injection", {})


def get_multi_segment_config() -> Dict[str, Any]:
    """获取多段式回复配置"""
    cfg = get_config()
    ms = cfg.get("multi_segment", {})
    return {
        "enabled": bool(ms.get("enabled", False)),
        "separator": str(ms.get("separator", "||")),
        "max_segments": int(ms.get("max_segments", 3)),
        "delay_range": tuple(ms.get("delay_range", [1.5, 4.0])),
        "trigger_scenes": list(ms.get("trigger_scenes", [])),
        "probability_by_scene": dict(ms.get("probability_by_scene", {})),
        "segment_hint": str(ms.get("segment_hint", "")),
    }


def should_trigger_multi_segment(scene_name: str, mood: str = "casual") -> bool:
    """判断当前场景+情绪是否应该触发多段提示"""
    ms_cfg = get_multi_segment_config()
    if not ms_cfg["enabled"]:
        return False
    if scene_name not in ms_cfg["trigger_scenes"]:
        return False
    prob = float(ms_cfg["probability_by_mood"].get(mood, 0.0))
    return prob > 0 and random.random() < prob


def preload() -> None:
    """启动时预加载配置，避免首次调用时的延迟"""
    global _cache_ts, _cache
    _cache = _load_raw()
    _cache_ts = time.time()
    meme_count = len(get_all_meme_words())
    scene_count = len(get_scene_matrix())
    logger.info(
        f"灵魂引擎配置预加载完成 | 梗词={meme_count} | 场景矩阵={scene_count}个"
    )


preload()

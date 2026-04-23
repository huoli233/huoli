from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

from src.common.logger import get_logger

logger = get_logger("heartfc_thresholds")


@dataclass(frozen=True)
class HeartFCThresholds:
    """HeartFC first-stage dynamic thresholds.

    Defaults mirror the previous hard-coded values in heartFC_chat_enhanced.py.
    Invalid configured values fall back per-field instead of failing startup.
    """

    rest_energy_ratio: float = 0.10
    rest_thinking_ratio: float = 0.08
    loafing_high: float = 0.60
    loafing_medium: float = 0.45
    loafing_idle: float = 0.45
    quiet_high: float = 0.55
    quiet_medium: float = 0.45
    quiet_watch_low: float = 0.60
    avoidance_high: float = 0.40
    avoidance_medium: float = 0.45
    social_low_willingness: float = 0.45
    boredom_peek_trigger: float = 55.0
    boredom_scan_trigger: float = 40.0
    boredom_downgrade_trigger: float = 35.0
    loafing_social_inhibit: float = 0.40
    boredom_reply_suppress_high: float = 0.60
    boredom_reply_suppress_medium: float = 0.30
    boredom_activation_reduce: float = 0.75
    boredom_drift_trigger: float = 0.50

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


_DEFAULTS = HeartFCThresholds()
_RATIO_FIELDS = {
    "rest_energy_ratio",
    "rest_thinking_ratio",
    "loafing_high",
    "loafing_medium",
    "loafing_idle",
    "quiet_high",
    "quiet_medium",
    "quiet_watch_low",
    "avoidance_high",
    "avoidance_medium",
    "social_low_willingness",
    "loafing_social_inhibit",
    "boredom_reply_suppress_high",
    "boredom_reply_suppress_medium",
    "boredom_activation_reduce",
    "boredom_drift_trigger",
}
_PERCENT_FIELDS = {
    "boredom_peek_trigger",
    "boredom_scan_trigger",
    "boredom_downgrade_trigger",
}


def _coerce_float(name: str, raw_value: Any, default: float) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        logger.warning(f"heartfc_thresholds.{name} 非法，使用默认值 {default}")
        return default
    if name in _RATIO_FIELDS:
        if 0.0 <= value <= 1.0:
            return value
        logger.warning(f"heartfc_thresholds.{name} 超出 0..1，使用默认值 {default}")
        return default
    if name in _PERCENT_FIELDS:
        if 0.0 <= value <= 100.0:
            return value
        logger.warning(f"heartfc_thresholds.{name} 超出 0..100，使用默认值 {default}")
        return default
    return value


def load_heartfc_thresholds(config_block: Dict[str, Any] | None = None) -> HeartFCThresholds:
    """Load thresholds from CoreSettingsHub with safe per-field fallback."""

    if config_block is None:
        try:
            from src.config.core_config_engine import get_core_config

            config_block = get_core_config().resolve_module_view("heartfc_thresholds").values
        except Exception as exc:
            logger.debug(f"读取 heartfc_thresholds 配置失败，使用默认值: {exc}")
            config_block = {}
    if not isinstance(config_block, dict):
        config_block = {}

    values: Dict[str, float] = {}
    for item in fields(HeartFCThresholds):
        default_value = getattr(_DEFAULTS, item.name)
        values[item.name] = _coerce_float(
            item.name,
            config_block.get(item.name, default_value),
            default_value,
        )
    return HeartFCThresholds(**values)


def get_heartfc_thresholds() -> HeartFCThresholds:
    return load_heartfc_thresholds()

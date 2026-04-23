import time as _tm
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

from src.common.logger import get_logger

logger = get_logger("心流配置")


# ---------------------------------------------------------------------------
#  枚举
# ---------------------------------------------------------------------------


class HeartFlowOperatingMode(Enum):
    """心流运行模式"""

    STANDARD = "standard"
    FOCUSED = "focused"
    PASSIVE = "passive"


# ---------------------------------------------------------------------------
#  子系统配置数据类
# ---------------------------------------------------------------------------


@dataclass
class StandardChatParams:
    """普通聊天参数"""

    base_reply_probability: float = 0.5
    reply_cooldown_sec: float = 3.0
    max_consecutive_replies: int = 6
    context_window_size: int = 15
    ignore_short_msg_len: int = 1


@dataclass
class FocusedChatParams:
    """聚焦聊天参数"""

    focus_cooldown_sec: float = 1.0
    focus_context_window: int = 20
    focus_max_tokens: int = 300
    recall_probability: float = 0.1
    post_send_review_probability: float = 0.05


@dataclass
class InitiativeParams:
    """主动发言参数"""

    initiative_interval_sec: float = 300.0
    initiative_probability: float = 0.3
    quiet_threshold_sec: float = 180.0
    max_initiative_per_hour: int = 5


@dataclass
class StaminaParams:
    """思考值系统参数"""

    stamina_max: float = 100.0
    stamina_cost_per_reply: float = 5.0
    stamina_recovery_per_sec: float = 0.1
    stamina_low_threshold: float = 20.0


@dataclass
class PsychologyParams:
    """心理系统参数"""

    trauma_decay_rate: float = 0.001
    annoyance_decay_rate: float = 0.01
    mood_inertia: float = 0.7
    chaos_threshold: float = 0.6


@dataclass
class MonitorDrivenParams:
    """模型驱动监控参数"""

    monitor_interval_sec: float = 5.0
    message_batch_size: int = 10
    quick_decision_timeout_sec: float = 10.0
    stamina_recovery_per_tick: float = 0.5
    topic_cooldown_sec: float = 120.0


@dataclass
class HeartFlowCoreParams:
    """心流核心配置汇总"""

    standard_chat: StandardChatParams = field(
        default_factory=StandardChatParams
    )
    focused_chat: FocusedChatParams = field(default_factory=FocusedChatParams)
    initiative: InitiativeParams = field(default_factory=InitiativeParams)
    stamina: StaminaParams = field(default_factory=StaminaParams)
    psychology: PsychologyParams = field(default_factory=PsychologyParams)
    monitor: MonitorDrivenParams = field(default_factory=MonitorDrivenParams)


# ---------------------------------------------------------------------------
#  配置加载与缓存
# ---------------------------------------------------------------------------

_cached_cfg: Optional[HeartFlowCoreParams] = None
_cached_at: float = 0.0
_CACHE_TTL: float = 60.0
_reload_observers: List[Callable[[HeartFlowCoreParams], None]] = []


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _validate_and_clamp(cfg: HeartFlowCoreParams) -> None:
    """(原地)校正明显超范围的参数"""
    sc = cfg.standard_chat
    sc.base_reply_probability = _clamp(sc.base_reply_probability, 0.0, 1.0)
    sc.reply_cooldown_sec = max(0.0, sc.reply_cooldown_sec)
    sc.max_consecutive_replies = max(1, sc.max_consecutive_replies)
    sc.context_window_size = max(1, sc.context_window_size)

    fc = cfg.focused_chat
    fc.focus_cooldown_sec = max(0.0, fc.focus_cooldown_sec)
    fc.recall_probability = _clamp(fc.recall_probability, 0.0, 1.0)
    fc.post_send_review_probability = _clamp(
        fc.post_send_review_probability, 0.0, 1.0
    )

    ip = cfg.initiative
    ip.initiative_interval_sec = max(10.0, ip.initiative_interval_sec)
    ip.initiative_probability = _clamp(ip.initiative_probability, 0.0, 1.0)
    ip.quiet_threshold_sec = max(10.0, ip.quiet_threshold_sec)
    ip.max_initiative_per_hour = max(0, ip.max_initiative_per_hour)

    sp = cfg.stamina
    sp.stamina_max = max(1.0, sp.stamina_max)
    sp.stamina_cost_per_reply = max(0.0, sp.stamina_cost_per_reply)
    sp.stamina_recovery_per_sec = max(0.0, sp.stamina_recovery_per_sec)
    sp.stamina_low_threshold = _clamp(
        sp.stamina_low_threshold, 0.0, sp.stamina_max
    )

    mp = cfg.monitor
    mp.monitor_interval_sec = max(1.0, mp.monitor_interval_sec)
    mp.message_batch_size = max(1, mp.message_batch_size)


def _fresh_config_from_sources() -> HeartFlowCoreParams:
    """从全局配置和核心配置引擎读取实际值，并做范围校正"""
    cfg = HeartFlowCoreParams()
    try:
        from src.config.config import global_config

        bot_cfg = global_config.bot
        # 普通聊天参数
        cfg.standard_chat.base_reply_probability = getattr(
            bot_cfg, "base_reply_probability", 0.5
        )
        cfg.standard_chat.max_consecutive_replies = getattr(
            bot_cfg, "max_consecutive_replies", 6
        )
    except Exception as exc:
        logger.debug(f"读取global_config失败，使用默认值: {exc}")
    try:
        from src.config.core_config_engine import get_core_config

        core = get_core_config()
        hf_section = core.resolve_module_view("heartflow_runtime").values
        if hf_section:
            cfg.initiative.initiative_interval_sec = hf_section.get(
                "initiative_interval_sec",
                cfg.initiative.initiative_interval_sec,
            )
            cfg.initiative.initiative_probability = hf_section.get(
                "initiative_probability", cfg.initiative.initiative_probability
            )
            cfg.initiative.quiet_threshold_sec = hf_section.get(
                "quiet_threshold_sec", cfg.initiative.quiet_threshold_sec
            )
            cfg.stamina.stamina_max = hf_section.get(
                "stamina_max", cfg.stamina.stamina_max
            )
            cfg.monitor.monitor_interval_sec = hf_section.get(
                "monitor_interval_sec", cfg.monitor.monitor_interval_sec
            )
    except Exception as exc:
        logger.debug(f"读取core_config失败，使用默认值: {exc}")
    _validate_and_clamp(cfg)
    return cfg


def load_heartflow_params() -> HeartFlowCoreParams:
    """加载心流参数（带缓存）"""
    global _cached_cfg, _cached_at
    now = _tm.time()
    if _cached_cfg is not None and (now - _cached_at) < _CACHE_TTL:
        return _cached_cfg
    _cached_cfg = _fresh_config_from_sources()
    _cached_at = now
    logger.info("[心流配置] 已加载/刷新")
    _notify_observers(_cached_cfg)
    return _cached_cfg


def reload_heartflow_params() -> HeartFlowCoreParams:
    """强制重新加载（清除缓存）"""
    global _cached_cfg, _cached_at
    _cached_cfg = None
    _cached_at = 0.0
    return load_heartflow_params()


def invalidate_config_cache() -> None:
    """外部调用：仅失效缓存，下次 load 时重新读取"""
    global _cached_cfg, _cached_at
    _cached_cfg = None
    _cached_at = 0.0


def register_config_observer(
    callback: Callable[[HeartFlowCoreParams], None],
) -> None:
    """注册配置变更观察者，每次加载/刷新后回调"""
    if callback not in _reload_observers:
        _reload_observers.append(callback)


def _notify_observers(cfg: HeartFlowCoreParams) -> None:
    for cb in _reload_observers:
        try:
            cb(cfg)
        except Exception as exc:
            logger.debug(f"[心流配置] 观察者回调异常: {exc}")


def get_monitor_driven_params() -> MonitorDrivenParams:
    """便捷访问监控参数"""
    return load_heartflow_params().monitor


def is_focused_mode_for_channel(channel_id: str) -> bool:
    """检查指定频道是否处于聚焦模式"""
    try:
        from src.config.core_config_engine import get_core_config

        core = get_core_config()
        focus_channels = core.resolve_module_view("heartflow_runtime").values.get("focus_channels", [])
        return channel_id in focus_channels
    except Exception as _exc:
        logger.warning(f"焦点频道检查异常: {_exc}")
        return False

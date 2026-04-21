import random
from dataclasses import dataclass, field
from typing import List

from src.common.logger import get_logger
from src.config.config import model_config
from src.core.watch_state_machine import WatchLevel

logger = get_logger("心流增强")

# 独立随机数生成器实例，避免 random.seed() 污染全局状态
_rng = random.Random()


@dataclass
class BehaviorGovernorVerdict:
    """统一行为裁决输出，避免主动/窥屏/引用各自拼零散布尔值。"""

    reply_mode: str = "observe"
    interrupt_level: str = "ignore"
    quote_policy: str = "none"
    target_user_id: str = ""
    silence_policy: str = "silent"
    max_watch_rank: int = 4
    allow_generation: bool = True
    model_tier: str = "small"
    reason_codes: List[str] = field(default_factory=list)


@dataclass
class RestGovernorVerdict:
    """统一休息/摸鱼/打断许可裁决，避免不同入口各自解释“该歇一下”。"""

    posture: str = "active"
    interruption_policy: str = "allow"
    should_rest: bool = False
    should_loaf: bool = False
    reason_codes: List[str] = field(default_factory=list)


@dataclass
class ModelGovernorVerdict:
    """模型路由裁决：统一 small/large/skip 与限流回退。"""

    tier: str = "skip"
    upgrade_reason_codes: List[str] = field(default_factory=list)
    rate_limited: bool = False
    fallback_to_small: bool = False
    dynamic_cooldown_sec: float = 0.0
    dynamic_hourly_cap: int = 0


_WATCH_LEVEL_BY_RANK = {
    0: WatchLevel.BLACKOUT,
    1: WatchLevel.PEEK,
    2: WatchLevel.SKIM_WINDOW,
    3: WatchLevel.ACTIVE_WATCH,
    4: WatchLevel.ENGAGED,
}


def _rt_float(key: str, default: float) -> float:
    """从 runtime_tuning 读取浮点参数，失败时返回默认值"""
    try:
        from src.config.core_config_engine import CoreSettingsHub

        val = CoreSettingsHub().runtime_tuning_block().get(key, default)
        return float(val)
    except Exception as exc:
        logger.debug(f"读取运行时配置 {key} 失败: {exc}")
        return default


def _task_provider_timeout(task_name: str, default: float = 10.0) -> float:
    """读取某个任务绑定模型的 provider 超时上限。"""
    try:
        task_cfg = getattr(model_config.model_task_config, task_name, None)
        provider_timeouts: list[float] = []
        for model_name in getattr(task_cfg, "model_list", []) or []:
            model_info = model_config.get_model_info(model_name)
            provider = model_config.get_provider(model_info.api_provider)
            provider_timeouts.append(float(getattr(provider, "timeout", default) or default))
        if provider_timeouts:
            return max(provider_timeouts)
    except Exception as exc:
        logger.debug(f"读取任务 {task_name} provider 超时失败: {exc}")
    return default


def _parallel_stage_timeout() -> float:
    default_timeout = max(
        20.0,
        _task_provider_timeout("focus_chat") + 10.0,
        _task_provider_timeout("utils") + 10.0,
    )
    return _rt_float("heartfc_parallel_stage_timeout_seconds", default_timeout)


def _llm_upgrade_timeout() -> float:
    default_timeout = max(15.0, _task_provider_timeout("lightweight") + 10.0)
    return _rt_float("heartfc_llm_upgrade_timeout_seconds", default_timeout)


_TICK_FLOOR_SEC = _rt_float("heartfc_tick_floor_sec", 0.15)
_DORMANT_POLL_SEC = _rt_float("heartfc_dormant_poll_sec", 1.0)
_PERCEPTION_COOLDOWN_SEC = _rt_float("heartfc_perception_cooldown_sec", 5.0)
_VOICE_COOLDOWN_SEC = _rt_float("heartfc_voice_cooldown_sec", 6.0)
_ENERGY_DRAIN_FLOOR = _rt_float("heartfc_energy_drain_floor", 0.08)
_POST_MESSAGE_RETRY_SEC = _rt_float("heartfc_post_message_retry_sec", 2.0)

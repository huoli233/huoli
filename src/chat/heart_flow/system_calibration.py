import time as _tm
from dataclasses import dataclass
from typing import Callable, List, Optional

from src.common.logger import get_logger

logger = get_logger("sys_calibrate")


# ---------------------------------------------------------------------------
#  校准等级
# ---------------------------------------------------------------------------

_LIGHT_THRESHOLD = 3600.0  # 1小时
_MODERATE_THRESHOLD = 86400.0  # 24小时


@dataclass
class CalibrationSummary:
    """校准结果摘要"""

    offline_seconds: float = 0.0
    level: str = "无"
    intimacy_delta: float = 0.0
    emotion_delta: float = 0.0
    memory_forget_probability: float = 0.0


def _determine_level(offline_sec: float) -> str:
    if offline_sec < _LIGHT_THRESHOLD:
        return "轻度"
    if offline_sec < _MODERATE_THRESHOLD:
        return "中度"
    return "重度"


# ---------------------------------------------------------------------------
#  衰减函数
# ---------------------------------------------------------------------------


def _linear_decay(current: float, rate_per_hour: float, hours: float) -> float:
    """线性衰减，下限为0"""
    return max(0.0, current - rate_per_hour * hours)


def _exponential_decay(
    current: float, half_life_hours: float, hours: float
) -> float:
    """指数衰减"""
    import math

    if half_life_hours <= 0:
        return 0.0
    decay_factor = math.exp(-0.693 * hours / half_life_hours)
    return current * decay_factor


def _memory_forget_prob(hours_offline: float) -> float:
    """根据离线时长计算记忆遗忘概率"""
    import math

    return 1.0 - math.exp(-0.01 * hours_offline)


# ---------------------------------------------------------------------------
#  校准器
# ---------------------------------------------------------------------------


class SystemCalibrator:
    """系统校准管理器"""

    def __init__(self):
        self._last_active_ts: float = _tm.time()
        self._calibrated: bool = False
        self._post_calibrate_hooks: List[
            Callable[[CalibrationSummary], None]
        ] = []
        self._latest_summary: Optional[CalibrationSummary] = None

    def register_hook(
        self, hook: Callable[[CalibrationSummary], None]
    ) -> None:
        """注册校准完成后的回调。签名: hook(summary) -> None"""
        self._post_calibrate_hooks.append(hook)

    @property
    def latest_summary(self) -> Optional[CalibrationSummary]:
        return self._latest_summary

    def mark_active(self) -> None:
        """标记当前活跃时间"""
        self._last_active_ts = _tm.time()
        self._calibrated = False

    def calibrate(
        self,
        *,
        current_intimacy: float = 50.0,
        current_emotion: float = 50.0,
        intimacy_decay_rate: float = 0.5,
        emotion_half_life: float = 12.0,
        reference_ts: Optional[float] = None,
    ) -> CalibrationSummary:
        """执行一次校准

        参数:
            current_intimacy: 当前亲密度(0-100)
            current_emotion: 当前情绪指数(0-100)
            intimacy_decay_rate: 每小时亲密度衰减
            emotion_half_life: 情绪半衰期(小时)
            reference_ts: 参考时间戳(用于测试), 默认取当前时间
        """
        now = reference_ts or _tm.time()
        offline_sec = max(0.0, now - self._last_active_ts)
        hours = offline_sec / 3600.0
        level = _determine_level(offline_sec)
        summary = CalibrationSummary(
            offline_seconds=offline_sec,
            level=level,
        )
        if level == "轻度":
            summary.intimacy_delta = 0.0
            summary.emotion_delta = 0.0
            summary.memory_forget_probability = 0.0
            logger.info(f"[校准] 轻度({offline_sec:.0f}s)，几乎无衰减")
            self._calibrated = True
            self._latest_summary = summary
            self._fire_hooks(summary)
            return summary
        # 中度/重度
        new_intimacy = _linear_decay(
            current_intimacy, intimacy_decay_rate, hours
        )
        summary.intimacy_delta = new_intimacy - current_intimacy
        new_emotion = _exponential_decay(
            current_emotion, emotion_half_life, hours
        )
        summary.emotion_delta = new_emotion - current_emotion
        summary.memory_forget_probability = _memory_forget_prob(hours)
        logger.info(
            f"[校准] {level}({hours:.1f}h) "
            f"亲密度{summary.intimacy_delta:+.1f} "
            f"情绪{summary.emotion_delta:+.1f} "
            f"遗忘率{summary.memory_forget_probability:.3f}"
        )
        self._calibrated = True
        self._last_active_ts = now
        self._latest_summary = summary
        self._fire_hooks(summary)
        return summary

    def _fire_hooks(self, summary: CalibrationSummary) -> None:
        """依次调用已注册的回调"""
        for hook in self._post_calibrate_hooks:
            try:
                hook(summary)
            except Exception as exc:
                logger.warning(f"[校准] 回调异常: {exc}")

    def calibrate_and_propagate(
        self,
        **kwargs,
    ) -> CalibrationSummary:
        """执行校准并自动传播到 flow_planner 和 phase_governor。

        这是推荐的顶层入口，替代单独调用 calibrate() 后手动分发。
        首次调用时自动注册默认钩子（仅注册一次）。
        """
        self._ensure_default_hooks()
        return self.calibrate(**kwargs)

    def _ensure_default_hooks(self) -> None:
        """懒注册内置传播钩子（幂等）"""
        marker = "_default_hooks_registered"
        if getattr(self, marker, False):
            return
        setattr(self, marker, True)

        def _propagate_to_planner(summary: CalibrationSummary) -> None:
            try:
                from src.chat.heart_flow.flow_planner import (
                    acquire_flow_planner,
                )

                acquire_flow_planner().apply_calibration(summary)
            except Exception as _e:
                logger.debug(f"[校准] 异常: {_e}")

        def _propagate_stamina_reset(summary: CalibrationSummary) -> None:
            if summary.level == "轻度":
                return
            try:
                from src.chat.heart_flow.phase_governor import _governor_cache

                for gov in _governor_cache.values():
                    gov.force_standby(
                        f"校准传播({summary.level}，离线{summary.offline_seconds / 3600:.1f}h)"
                    )
            except Exception as _e:
                logger.debug(f"[校准] 异常: {_e}")

        self.register_hook(_propagate_to_planner)
        self.register_hook(_propagate_stamina_reset)

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated


# ---------------------------------------------------------------------------
#  单例
# ---------------------------------------------------------------------------

_calibrator_ref: Optional[SystemCalibrator] = None


def acquire_system_calibrator() -> SystemCalibrator:
    global _calibrator_ref
    if _calibrator_ref is None:
        _calibrator_ref = SystemCalibrator()
    return _calibrator_ref

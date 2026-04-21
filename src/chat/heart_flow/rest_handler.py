import random
import time
from typing import Any, Dict, Optional
from src.common.logger import get_logger
from src.config.core_config_engine import get_core_config
from src.common.data_models.heartflow_models import (
    DormantSetup,
    DormantReaction,
)

logger = get_logger("dormancy_supervisor")


class DormancySupervisor:
    """
    休息阶段管理器
    频道进入 DORMANT 后，决定对消息的反应策略（跳过/瞥一眼/唤醒）。
    概率由配置驱动，用均匀随机数判定，无词表匹配。
    """

    _solo: Optional["DormancySupervisor"] = None

    @classmethod
    def instance(cls) -> "DormancySupervisor":
        if cls._solo is None:
            cls._solo = cls()
        return cls._solo

    @classmethod
    def teardown(cls) -> None:
        if cls._solo is not None:
            cls._solo._channel_dormancy.clear()
        cls._solo = None

    def __init__(self):
        self._channel_dormancy: Dict[str, DormantSetup] = {}
        self._rng = random.Random()

    def _timing_cfg(self) -> Dict[str, Any]:
        hub = get_core_config()
        return hub.heartflow_timing()

    def begin_dormancy(self, channel_id: str, cause: str = "") -> DormantSetup:
        """开始休息期"""
        cfg = self._timing_cfg()
        setup = DormantSetup(
            cause=cause,
            duration_sec=int(cfg.get("rest_default_seconds", 300)),
            entered_at=time.time(),
            skip_streak=0,
            glance_probability=float(cfg.get("peek_chance_ratio", 0.05)),
            allow_awakening=True,
        )
        self._channel_dormancy[channel_id] = setup
        logger.info(f'频道 {channel_id[:8]} 进入休息，时长={setup.duration_sec}秒 原因="{cause}"')
        return setup

    def begin_dynamic_dormancy(
        self,
        channel_id: str,
        *,
        cause: str = "",
        rest_intensity: float = 0.5,
        wake_drive: float = 0.3,
        regret_chance: float = 0.0,
        min_rest_sec: int = 45,
        max_rest_sec: int = 900,
    ) -> DormantSetup:
        """开始可反悔的动态休息期。"""
        cfg = self._timing_cfg()
        default_duration = int(cfg.get("rest_default_seconds", 300))
        intensity = max(0.0, min(1.0, float(rest_intensity)))
        wake_pressure = max(0.0, min(1.0, float(wake_drive)))
        regret_rate = max(0.0, min(0.95, float(regret_chance)))
        base_duration = default_duration * (0.55 + intensity * 1.15)
        softened_duration = base_duration * (1.0 - wake_pressure * 0.35)
        planned_duration = int(max(min_rest_sec, min(max_rest_sec, softened_duration)))
        setup = DormantSetup(
            cause=cause,
            duration_sec=planned_duration,
            entered_at=time.time(),
            skip_streak=0,
            glance_probability=float(cfg.get("peek_chance_ratio", 0.05)),
            allow_awakening=True,
            rest_intensity=intensity,
            wake_drive=wake_pressure,
            regret_chance=regret_rate,
            min_rest_sec=int(max(0, min_rest_sec)),
            last_reconsider_at=0.0,
        )
        self._channel_dormancy[channel_id] = setup
        logger.info(
            f"频道 {channel_id[:8]} 进入动态休息，时长={setup.duration_sec}秒 "
            f"强度={setup.rest_intensity:.2f} 唤醒驱动={setup.wake_drive:.2f} "
            f'反悔率={setup.regret_chance:.2f} 原因="{cause}"'
        )
        return setup

    def fetch_dormancy(self, channel_id: str) -> Optional[DormantSetup]:
        return self._channel_dormancy.get(channel_id)

    def check_remaining(self, channel_id: str) -> float:
        """返回剩余休息秒数"""
        setup = self._channel_dormancy.get(channel_id)
        if setup is None:
            return 0.0
        return setup.remaining_sec()

    def is_dormancy_over(self, channel_id: str) -> bool:
        setup = self._channel_dormancy.get(channel_id)
        if setup is None:
            return True
        return setup.is_finished()

    def on_incoming_msg(self, channel_id: str) -> DormantReaction:
        """
        休息期间收到消息时的决策。
        如果已过期 -> 唤醒；否则用随机数和概率决定跳过或瞥一眼。
        连续跳过次数过多时增加窥屏概率。
        """
        setup = self._channel_dormancy.get(channel_id)
        if setup is None:
            return DormantReaction.AWAKEN
        if setup.is_finished():
            logger.info(f"频道 {channel_id[:8]} 休息已到期，唤醒")
            return DormantReaction.AWAKEN
        if self._should_regret_awaken(setup):
            logger.info(
                f"频道 {channel_id[:8]} 休息中反悔唤醒 rest={setup.rest_intensity:.2f} wake={setup.wake_drive:.2f}"
            )
            return DormantReaction.AWAKEN
        if not setup.allow_awakening:
            setup.skip_streak += 1
            return DormantReaction.SKIP
        adjusted_prob = setup.glance_probability
        if setup.skip_streak >= 5:
            adjusted_prob = min(1.0, adjusted_prob * 2.0)
        elif setup.skip_streak >= 3:
            adjusted_prob = min(1.0, adjusted_prob * 1.5)
        roll = self._rng.random()
        if roll < adjusted_prob:
            setup.skip_streak = 0
            logger.debug(f"频道 {channel_id[:8]} 休息中瞥了一眼 (roll={roll:.3f} prob={adjusted_prob:.3f})")
            return DormantReaction.GLANCE
        cfg = self._timing_cfg()
        awaken_threshold = int(cfg.get("awaken_skip_threshold", 0))
        if awaken_threshold > 0 and setup.skip_streak >= awaken_threshold:
            logger.info(f"频道 {channel_id[:8]} 连续跳过{setup.skip_streak}次，强制唤醒")
            return DormantReaction.AWAKEN
        setup.skip_streak += 1
        return DormantReaction.SKIP

    def _should_regret_awaken(self, setup: DormantSetup) -> bool:
        """休息中重新想说话时提前唤醒。"""
        remaining = setup.remaining_sec()
        elapsed = max(0.0, time.time() - setup.entered_at)
        if elapsed < max(10.0, float(setup.min_rest_sec)):
            return False
        if remaining <= 0:
            return True
        if not setup.can_reconsider(20.0):
            return False
        setup.mark_reconsidered()
        progress = setup.progress_ratio()
        awaken_bias = setup.wake_drive * (0.55 + progress * 0.35) + setup.regret_chance
        awaken_bias -= setup.rest_intensity * max(0.15, 1.0 - progress) * 0.45
        awaken_bias = max(0.0, min(0.85, awaken_bias))
        return self._rng.random() < awaken_bias

    def conclude_dormancy(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """结束休息，应用能量恢复并返回摘要"""
        setup = self._channel_dormancy.pop(channel_id, None)
        if setup is None:
            return None
        duration_actual = time.time() - setup.entered_at if setup.entered_at > 0 else 0
        if duration_actual > 0:
            try:
                from src.chat.heart_flow.energy_manager import EnergyChainDimension

                rest_min = duration_actual / 60.0
                _d6 = EnergyChainDimension.get_instance()
                _ch = _d6._ensure_channel(channel_id)
                if _ch:
                    recover_amount = rest_min * 2.0
                    _ch.chat_pool = min(_ch.chat_ceiling, _ch.chat_pool + recover_amount)
                    _ch.thinking_value = min(_ch.thinking_ceiling, _ch.thinking_value + recover_amount * 0.5)
                    _ch.last_update = time.time()
            except Exception as exc:
                logger.debug(f"频道 {channel_id[:8]} 休息回血失败: {exc}")
        summary = {
            "cause": setup.cause,
            "planned_duration_sec": setup.duration_sec,
            "actual_duration_sec": round(duration_actual, 1),
            "skip_streak_final": setup.skip_streak,
            "rest_intensity": round(setup.rest_intensity, 3),
            "wake_drive": round(setup.wake_drive, 3),
            "regret_chance": round(setup.regret_chance, 3),
        }
        logger.info(f"频道 {channel_id[:8]} 休息结束: {summary}")
        return summary

    def is_resting(self, channel_id: str) -> bool:
        return channel_id in self._channel_dormancy

    def dormancy_progress(self, channel_id: str) -> float:
        setup = self._channel_dormancy.get(channel_id)
        if setup is None:
            return 1.0
        return setup.progress_ratio()

    def snapshot(self, channel_id: str) -> Dict[str, Any]:
        setup = self._channel_dormancy.get(channel_id)
        if setup is None:
            return {"resting": False}
        return {
            "resting": True,
            "cause": setup.cause,
            "remaining_sec": round(setup.remaining_sec(), 1),
            "progress": round(setup.progress_ratio(), 2),
            "skip_streak": setup.skip_streak,
            "rest_intensity": round(setup.rest_intensity, 3),
            "wake_drive": round(setup.wake_drive, 3),
            "regret_chance": round(setup.regret_chance, 3),
        }

    def all_resting_channels(self) -> list:
        return list(self._channel_dormancy.keys())

    def force_awaken(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """强制唤醒：结束休息并返回摘要"""
        if channel_id not in self._channel_dormancy:
            return None
        logger.info(f"频道 {channel_id[:8]} 强制唤醒")
        return self.conclude_dormancy(channel_id)


def get_dormancy_supervisor() -> DormancySupervisor:
    """获取全局休息管理器"""
    return DormancySupervisor.instance()

import time as _tm
from dataclasses import dataclass, field
from typing import Dict

from src.common.logger import get_logger
from src.common.data_models.heartflow_models import FlowPhase

logger = get_logger("阶段治理")


# ---------------------------------------------------------------------------
#  状态信息快照
# ---------------------------------------------------------------------------


@dataclass
class PhaseSnapshot:
    """当前状态的只读快照"""

    current_phase: FlowPhase = FlowPhase.STANDBY
    phase_label: str = ""
    since_entered: float = 0.0
    is_observing: bool = False
    is_peeking: bool = False
    is_slacking: bool = False
    can_reply: bool = True
    stamina_pct: float = 100.0


# ---------------------------------------------------------------------------
#  思考值追踪
# ---------------------------------------------------------------------------

_STAMINA_FLOOR = 0.0
_STAMINA_CEILING = 100.0


@dataclass
class _StaminaTracker:
    """内部思考值追踪器"""

    pool: float = 100.0
    ceiling: float = 100.0
    recovery_per_sec: float = 0.1
    last_tick_at: float = field(default_factory=_tm.time)

    def consume(self, cost: float) -> None:
        self.pool = max(_STAMINA_FLOOR, self.pool - cost)
        self.last_tick_at = _tm.time()

    def tick_recovery(self) -> None:
        now = _tm.time()
        elapsed = now - self.last_tick_at
        if elapsed > 0 and self.pool < self.ceiling:
            gain = elapsed * self.recovery_per_sec
            self.pool = min(self.ceiling, self.pool + gain)
        self.last_tick_at = now

    def ratio(self) -> float:
        return self.pool / self.ceiling if self.ceiling > 0 else 0.0


# ---------------------------------------------------------------------------
#  管理器
# ---------------------------------------------------------------------------


class PhaseGovernor:
    """心流状态生命周期管理器（每频道一个实例）"""

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._current_phase = FlowPhase.STANDBY
        self._entered_at: float = _tm.time()
        self._observing: bool = False
        self._peeking: bool = False
        self._slacking: bool = False
        self._stamina = _StaminaTracker()

    # ---- 属性访问 ----

    @property
    def current_phase(self) -> FlowPhase:
        return self._current_phase

    @property
    def can_reply(self) -> bool:
        return self._current_phase in (FlowPhase.STANDBY, FlowPhase.ENGAGED)

    # ---- 状态切换 ----

    def apply_decision(
        self, target_phase: FlowPhase, reason: str = ""
    ) -> bool:
        """执行模型决策驱动的状态切换

        参照合法迁移表，非法切换被拒绝。返回是否切换成功。
        """
        if self._current_phase == target_phase:
            # 允许同态回写（幂等迁移），避免与协调器短暂不同步时误报“非法迁移”。
            self._entered_at = _tm.time()
            self._observing = target_phase == FlowPhase.STANDBY
            self._peeking = False
            self._slacking = target_phase == FlowPhase.DORMANT
            logger.debug(
                f"[状态管理] 幂等迁移 {self._current_phase.value}→{target_phase.value}，原因: {reason}"
            )
            return True

        from src.common.data_models.heartflow_models import is_transition_legal

        if not is_transition_legal(self._current_phase, target_phase):
            logger.warning(
                f"[状态管理] 非法迁移 {
                    self._current_phase.value}→{
                    target_phase.value}，"
                f"原因: {reason}"
            )
            return False
        old_label = self._current_phase.label()
        self._current_phase = target_phase
        self._entered_at = _tm.time()
        self._observing = target_phase == FlowPhase.STANDBY
        self._peeking = False
        self._slacking = target_phase == FlowPhase.DORMANT
        logger.info(
            f"[状态管理] {old_label}→{target_phase.label()} | {reason}"
        )
        # 状态迁移 → 裁决缓存失效
        self._invalidate_verdict_cache()
        return True

    def check_expiration(self, idle_timeout_sec: float = 300.0) -> bool:
        """检查当前状态是否到期，到期则自动回落到 STANDBY"""
        if self._current_phase == FlowPhase.STANDBY:
            return False
        elapsed = _tm.time() - self._entered_at
        if (
            self._current_phase == FlowPhase.PENDING
            and elapsed >= idle_timeout_sec
        ):
            logger.info(f"[状态管理] 等待超时 {elapsed:.0f}s，回落到待命")
            self._current_phase = FlowPhase.STANDBY
            self._entered_at = _tm.time()
            return True
        return False

    def on_message_arrived(self) -> None:
        """收到消息时更新状态标志"""
        if self._current_phase == FlowPhase.DORMANT:
            self._peeking = True
        self._stamina.tick_recovery()

    def check_rest_trigger(self, stamina_threshold: float = 20.0) -> bool:
        """检查思考值是否低于阈值，暗示需要休息"""
        self._stamina.tick_recovery()
        return self._stamina.pool < stamina_threshold

    def force_standby(self, reason: str = "强制重置") -> None:
        """强制终止当前状态回到待命"""
        self._current_phase = FlowPhase.STANDBY
        self._entered_at = _tm.time()
        self._observing = True
        self._peeking = False
        self._slacking = False
        logger.info(f"[状态管理] 强制待命: {reason}")
        self._invalidate_verdict_cache()

    def consume_stamina(self, cost: float) -> None:
        """消耗思考值"""
        self._stamina.consume(cost)

    def _invalidate_verdict_cache(self) -> None:
        """状态迁移/强制重置后使裁决缓存失效"""
        try:
            from src.chat.heart_flow.heartflow_decision import (
                acquire_decision_maker,
            )

            acquire_decision_maker().signal_activity_change(self._channel_id)
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    # ---- 状态快照 ----

    def snapshot(self) -> PhaseSnapshot:
        """生成当前状态的只读快照"""
        self._stamina.tick_recovery()
        return PhaseSnapshot(
            current_phase=self._current_phase,
            phase_label=self._current_phase.label(),
            since_entered=_tm.time() - self._entered_at,
            is_observing=self._observing,
            is_peeking=self._peeking,
            is_slacking=self._slacking,
            can_reply=self.can_reply,
            stamina_pct=self._stamina.pool,
        )


# ---------------------------------------------------------------------------
#  频道级实例管理
# ---------------------------------------------------------------------------

_governor_cache: Dict[str, PhaseGovernor] = {}


def acquire_phase_governor(channel_id: str) -> PhaseGovernor:
    """获取频道对应的状态管理器"""
    if channel_id not in _governor_cache:
        _governor_cache[channel_id] = PhaseGovernor(channel_id)
    return _governor_cache[channel_id]

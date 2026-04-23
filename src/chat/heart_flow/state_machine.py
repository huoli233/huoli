import time
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger
from src.config.core_config_engine import get_core_config
from src.common.data_models.heartflow_models import (
    FlowPhase,
    PhaseTransitionRecord,
    is_transition_legal,
)

logger = get_logger("阶段协调器")


class PhaseCoordinator:
    """
    心流阶段协调器
    按频道维护 FlowPhase 有限状态机，验证迁移合法性并记录转换历史。
    所有时序阈值来自 CoreSettingsHub，禁止硬编码。
    """

    _solo: Optional["PhaseCoordinator"] = None

    @classmethod
    def instance(cls) -> "PhaseCoordinator":
        if cls._solo is None:
            cls._solo = cls()
        return cls._solo

    @classmethod
    def teardown(cls) -> None:
        cls._solo = None

    def __init__(self):
        self._channel_phase: Dict[str, FlowPhase] = {}
        self._channel_entered_at: Dict[str, float] = {}
        self._transition_log: Dict[str, List[PhaseTransitionRecord]] = {}
        self._max_log_per_channel: int = 50
        # 迁移拒绝日志节流：同一频道+同一迁移方向，N秒内只输出一次
        self._reject_log_cooldown: Dict[str, float] = {}
        self._reject_cooldown_sec: float = 30.0

    def _cfg_block(self) -> Dict[str, Any]:
        """读取心流时序配置段"""
        hub = get_core_config()
        return hub.resolve_module_view("phase_timing").values

    def register_channel(
        self, channel_id: str, initial: FlowPhase = FlowPhase.STANDBY
    ) -> None:
        """首次注册频道，设定初始阶段"""
        if channel_id in self._channel_phase:
            return
        self._channel_phase[channel_id] = initial
        self._channel_entered_at[channel_id] = time.time()
        self._transition_log[channel_id] = []
        logger.debug(f"频道 {channel_id[:8]} 注册，初始阶段={initial.label()}")

    def current_phase(self, channel_id: str) -> FlowPhase:
        """获取频道当前阶段，未注册则返回 STANDBY"""
        return self._channel_phase.get(channel_id, FlowPhase.STANDBY)

    def phase_duration_sec(self, channel_id: str) -> float:
        """当前阶段持续时间（秒）"""
        entered = self._channel_entered_at.get(channel_id)
        if entered is None or entered <= 0:
            return 0.0
        return time.time() - entered

    def phase_duration_min(self, channel_id: str) -> float:
        return self.phase_duration_sec(channel_id) / 60.0

    def attempt_shift(
        self,
        channel_id: str,
        target: FlowPhase,
        reason: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        尝试阶段迁移。
        合法则更新并记录日志，返回 True；非法不修改，返回 False。
        """
        current = self.current_phase(channel_id)
        if current == target:
            return True
        if not is_transition_legal(current, target):
            _reject_key = f"{channel_id}:{current.value}->{target.value}"
            _now = time.time()
            _last = self._reject_log_cooldown.get(_reject_key, 0.0)
            if (_now - _last) >= self._reject_cooldown_sec:
                logger.warning(
                    f"频道 {channel_id[:8]} 迁移被拒: {current.label()} -> {target.label()}"
                )
                self._reject_log_cooldown[_reject_key] = _now
            return False
        now = time.time()
        record = PhaseTransitionRecord(
            from_phase=current.value,
            to_phase=target.value,
            trigger_reason=reason,
            occurred_at=now,
            extra=extra or {},
        )
        self._channel_phase[channel_id] = target
        self._channel_entered_at[channel_id] = now
        log_list = self._transition_log.setdefault(channel_id, [])
        log_list.append(record)
        if len(log_list) > self._max_log_per_channel:
            log_list[:] = log_list[-self._max_log_per_channel:]
        logger.info(
            f"频道 {channel_id[:8]} 迁移: {current.label()} -> {target.label()} 原因={reason}"
        )
        try:
            from src.chat.heart_flow.phase_governor import acquire_phase_governor
            _gov = acquire_phase_governor(channel_id)
            _gov.apply_decision(target, reason=reason)
        except Exception as exc:
            logger.warning(f"状态迁移通知phase_governor失败: {exc}")
        return True

    def force_reset(
        self,
        channel_id: str,
        target: FlowPhase = FlowPhase.STANDBY,
        reason: str = "强制重置",
    ) -> None:
        """跳过合法性校验，直接设置阶段（仅限外部主循环复位）"""
        old = self.current_phase(channel_id)
        now = time.time()
        self._channel_phase[channel_id] = target
        self._channel_entered_at[channel_id] = now
        log_list = self._transition_log.setdefault(channel_id, [])
        log_list.append(
            PhaseTransitionRecord(
                from_phase=old.value,
                to_phase=target.value,
                trigger_reason=reason,
                occurred_at=now,
                extra={"forced": True},
            )
        )
        logger.info(f"频道 {channel_id[:8]} 强制重置 -> {target.label()}")
        try:
            from src.chat.heart_flow.phase_governor import acquire_phase_governor
            _gov = acquire_phase_governor(channel_id)
            _gov.force_standby(reason=reason)
        except Exception as exc:
            logger.warning(f"强制重置通知phase_governor失败: {exc}")

    def discard_channel(self, channel_id: str) -> None:
        """丢弃频道数据"""
        self._channel_phase.pop(channel_id, None)
        self._channel_entered_at.pop(channel_id, None)
        self._transition_log.pop(channel_id, None)

    def recent_transitions(
        self, channel_id: str, count: int = 10
    ) -> List[PhaseTransitionRecord]:
        """获取最近的迁移记录"""
        log_list = self._transition_log.get(channel_id, [])
        if count <= 0:
            return list(log_list)
        return log_list[-count:]

    def all_channels(self) -> Dict[str, str]:
        """返回所有频道及其当前阶段标签"""
        return {
            cid: phase.label() for cid, phase in self._channel_phase.items()
        }

    def channel_snapshot(self, channel_id: str) -> Dict[str, Any]:
        """频道状态快照"""
        phase = self.current_phase(channel_id)
        return {
            "channel_id": channel_id,
            "phase": phase.value,
            "phase_label": phase.label(),
            "duration_sec": round(self.phase_duration_sec(channel_id), 1),
            "transition_count": len(self._transition_log.get(channel_id, [])),
        }

    def is_engaged(self, channel_id: str) -> bool:
        return self.current_phase(channel_id) == FlowPhase.ENGAGED

    def is_dormant(self, channel_id: str) -> bool:
        return self.current_phase(channel_id) == FlowPhase.DORMANT

    def is_pending(self, channel_id: str) -> bool:
        return self.current_phase(channel_id) == FlowPhase.PENDING

    def is_standby(self, channel_id: str) -> bool:
        return self.current_phase(channel_id) == FlowPhase.STANDBY

    def should_auto_dormant(self, channel_id: str) -> bool:
        """判断是否应自动进入休息（待命超时）"""
        if not self.is_standby(channel_id):
            return False
        cfg = self._cfg_block()
        auto_dormant_after_sec = int(cfg.get("auto_dormant_after_sec", 0))
        if auto_dormant_after_sec <= 0:
            return False
        return self.phase_duration_sec(channel_id) >= auto_dormant_after_sec

    def should_auto_standby(self, channel_id: str) -> bool:
        """判断是否应自动回到待命（活跃超时）"""
        if not self.is_engaged(channel_id):
            return False
        cfg = self._cfg_block()
        max_engaged_sec = int(cfg.get("max_engaged_duration_sec", 0))
        if max_engaged_sec <= 0:
            return False
        return self.phase_duration_sec(channel_id) >= max_engaged_sec

    def evaluate_auto_transitions(
        self, channel_id: str
    ) -> Optional[FlowPhase]:
        """
        综合评估自动迁移，返回建议目标阶段。
        不执行迁移本身，由调用方决定是否 attempt_shift。
        优先级：能量阈值 > 时间阈值
        """
        _energy_forced = self._evaluate_energy_forced_transition(channel_id)
        if _energy_forced is not None:
            return _energy_forced
        if self.should_auto_dormant(channel_id):
            return FlowPhase.DORMANT
        if self.should_auto_standby(channel_id):
            return FlowPhase.STANDBY
        if self.is_pending(channel_id):
            cfg = self._cfg_block()
            pending_expire_sec = int(cfg.get("pending_expire_sec", 0))
            if (
                pending_expire_sec > 0
                and self.phase_duration_sec(channel_id) >= pending_expire_sec
            ):
                return FlowPhase.STANDBY
        return None

    def _evaluate_energy_forced_transition(
        self, channel_id: str
    ) -> Optional["FlowPhase"]:
        """基于能量阈值强制迁移（优先于时间阈值）"""
        try:
            from src.chat.heart_flow.energy_manager import EnergyChainDimension

            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(channel_id)
            chat_val = float(_ch.chat_pool) if _ch else 100.0
            _tv = float(_ch.thinking_value) if _ch else 100.0
            _tc = float(_ch.thinking_ceiling) if _ch else 100.0
            think_ratio = max(0.0, min(1.0, _tv / max(_tc, 1.0)))
            activity_val = float(_ch.activity_level) if _ch else 50.0
            current = self.current_phase(channel_id)
            if current is None:
                return None
            from src.common.data_models.heartflow_models import FlowPhase

            if current in (FlowPhase.ENGAGED, FlowPhase.PENDING):
                if chat_val <= 15.0 or think_ratio <= 0.15:
                    return FlowPhase.DORMANT
                if chat_val <= 25.0 or think_ratio <= 0.25:
                    return FlowPhase.LIGHT_REST
                if activity_val <= 10.0:
                    return FlowPhase.LIGHT_REST
            elif current == FlowPhase.LIGHT_REST:
                if (
                    chat_val >= 60.0
                    and think_ratio >= 0.5
                    and activity_val >= 30.0
                ):
                    return FlowPhase.PENDING
            elif current == FlowPhase.DORMANT:
                if (
                    chat_val >= 70.0
                    and think_ratio >= 0.6
                    and activity_val >= 40.0
                ):
                    return FlowPhase.STANDBY
            elif current == FlowPhase.STANDBY:
                if chat_val >= 80.0 and think_ratio >= 0.7:
                    return FlowPhase.PENDING
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return None

    def persist_phases(self) -> None:
        """将所有频道的当前阶段写入 persistent_state_db"""
        if not self._channel_phase:
            return
        try:
            from src.chat.heart_flow.persistent_state_db import (
                acquire_persistent_store,
            )

            store = acquire_persistent_store()
            for cid, phase in self._channel_phase.items():
                store.set_state(
                    "phase",
                    cid,
                    {
                        "phase": phase.value,
                        "entered_at": self._channel_entered_at.get(cid, 0),
                    },
                )
        except Exception as exc:
            logger.debug(f"阶段持久化失败: {exc}")

    def restore_phase(self, channel_id: str) -> bool:
        """从 persistent_state_db 恢复频道阶段状态"""
        try:
            from src.chat.heart_flow.persistent_state_db import (
                acquire_persistent_store,
            )

            data = acquire_persistent_store().get_state("phase", channel_id)
            if not data or not isinstance(data, dict):
                return False
            phase_value = data.get("phase")
            target = None
            for member in FlowPhase:
                if member.value == phase_value:
                    target = member
                    break
            if target is None:
                return False
            self._channel_phase[channel_id] = target
            self._channel_entered_at[channel_id] = float(
                data.get("entered_at", time.time())
            )
            self._transition_log.setdefault(channel_id, [])
            logger.debug(f"频道 {channel_id[:8]} 阶段恢复为 {target.label()}")
            return True
        except Exception as exc:
            logger.debug(f"频道 {channel_id[:8]} 阶段恢复失败: {exc}")
            return False


def get_phase_coordinator() -> PhaseCoordinator:
    """获取全局阶段协调器"""
    return PhaseCoordinator.instance()

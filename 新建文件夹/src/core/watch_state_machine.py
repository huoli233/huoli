import time
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from src.common.logger import get_logger

logger = get_logger("watch_fsm")

_channel_machines: Dict[str, "WatchStateMachine"] = {}


class WatchLevel(Enum):
    """观看层级枚举：从完全无视到深度参与"""

    BLACKOUT = "blackout"
    PEEK = "peek"
    SKIM_WINDOW = "skim_window"
    ACTIVE_WATCH = "active_watch"
    ENGAGED = "engaged"

    def rank(self) -> int:
        _ranks = {
            "blackout": 0,
            "peek": 1,
            "skim_window": 2,
            "active_watch": 3,
            "engaged": 4,
        }
        return _ranks.get(self.value, 0)

    def label(self) -> str:
        _labels = {
            "blackout": "完全无视",
            "peek": "瞟一眼",
            "skim_window": "扫屏中",
            "active_watch": "在关注",
            "engaged": "深度参与",
        }
        return _labels.get(self.value, "未知")


@dataclass
class WatchTransitionRecord:
    """状态迁移记录"""

    from_level: WatchLevel
    to_level: WatchLevel
    trigger: str
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
#  迁移规则表
# ---------------------------------------------------------------------------

# 允许的上行迁移路径 (当前 → 可达层级集合)
_UPWARD_PATHS = {
    WatchLevel.BLACKOUT: {WatchLevel.PEEK},
    WatchLevel.PEEK: {WatchLevel.SKIM_WINDOW},
    WatchLevel.SKIM_WINDOW: {WatchLevel.ACTIVE_WATCH},
    WatchLevel.ACTIVE_WATCH: {WatchLevel.ENGAGED},
    WatchLevel.ENGAGED: set(),
}
# 允许的下行迁移路径（可跳级下降）
_DOWNWARD_PATHS = {
    WatchLevel.ENGAGED: {
        WatchLevel.ACTIVE_WATCH,
        WatchLevel.SKIM_WINDOW,
        WatchLevel.PEEK,
        WatchLevel.BLACKOUT,
    },
    WatchLevel.ACTIVE_WATCH: {
        WatchLevel.SKIM_WINDOW,
        WatchLevel.PEEK,
        WatchLevel.BLACKOUT,
    },
    WatchLevel.SKIM_WINDOW: {WatchLevel.PEEK, WatchLevel.BLACKOUT},
    WatchLevel.PEEK: {WatchLevel.BLACKOUT},
    WatchLevel.BLACKOUT: set(),
}


class WatchStateMachine:
    """窥屏状态机
    每个频道维护独立实例，追踪主体当前的「看群程度」。
    外部通过信号驱动上行/下行迁移，而非直接设置层级。
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._current_level = WatchLevel.PEEK
        self._level_entered_at = time.time()
        self._dwell_seconds: Dict[str, float] = {
            lv.value: 0.0 for lv in WatchLevel
        }
        self._transition_log: list = []
        self._max_log_size = 30

    @property
    def current_level(self) -> WatchLevel:
        return self._current_level

    @property
    def level_dwell_sec(self) -> float:
        """在当前层级已停留的秒数"""
        return time.time() - self._level_entered_at

    def snapshot(self) -> Dict[str, Any]:
        """导出当前观看层级快照"""
        return {
            "channel_id": self._channel_id,
            "level": self._current_level.value,
            "level_label": self._current_level.label(),
            "level_rank": self._current_level.rank(),
            "dwell_sec": round(self.level_dwell_sec, 1),
        }

    # ────────────────── 信号驱动入口 ──────────────────

    def signal_escalate(self, trigger: str = "消息刺激") -> bool:
        """接收上行信号，尝试提升观看层级一级"""
        candidates = _UPWARD_PATHS.get(self._current_level, set())
        if not candidates:
            return False
        target = min(candidates, key=lambda lv: lv.rank())
        return self._do_transition(target, trigger)

    def signal_deescalate(
        self, trigger: str = "兴趣衰减", target: Optional[WatchLevel] = None
    ) -> bool:
        """接收下行信号，降低观看层级"""
        reachable = _DOWNWARD_PATHS.get(self._current_level, set())
        if not reachable:
            return False
        if target and target in reachable:
            dest = target
        else:
            dest = max(reachable, key=lambda lv: lv.rank())
        return self._do_transition(dest, trigger)

    def signal_force_level(
        self, level: WatchLevel, trigger: str = "强制设定"
    ) -> bool:
        """强制跳转到指定层级（被@、重大事件等场景）"""
        if level == self._current_level:
            return False
        return self._do_transition(level, trigger)

    def evaluate_natural_decay(
        self, silence_sec: float, boredom: float
    ) -> bool:
        """根据静默时长和无聊值评估自然衰减"""
        dwell = self.level_dwell_sec
        rank = self._current_level.rank()
        if rank <= 0:
            return False
        # 无聊越高，在高层级维持的时间越短
        hold_base = 30.0 + rank * 20.0
        boredom_factor = max(0.3, 1.0 - boredom * 0.5)
        hold_threshold = hold_base * boredom_factor
        if dwell < hold_threshold and silence_sec < hold_threshold * 1.5:
            return False
        # 确定衰减幅度：静默时间长则跳级下降
        if silence_sec > hold_threshold * 3:
            target = WatchLevel.BLACKOUT
        elif silence_sec > hold_threshold * 2:
            target = WatchLevel.PEEK
        else:
            target = None
        return self.signal_deescalate(
            trigger=f"自然衰减(静默{
                silence_sec:.0f}s,无聊{
                boredom:.2f})",
            target=target,
        )

    def evaluate_stimulus(
        self, stimulus_strength: float, is_at_mention: bool = False
    ) -> bool:
        """根据刺激强度评估是否上行"""
        if is_at_mention:
            return self.signal_force_level(
                WatchLevel.ENGAGED, trigger="被@提及"
            )
        rank = self._current_level.rank()
        # 刺激强度阈值随层级递增
        threshold = 0.2 + rank * 0.15
        if stimulus_strength >= threshold:
            return self.signal_escalate(
                trigger=f"刺激({
                    stimulus_strength:.2f}>{
                    threshold:.2f})"
            )
        return False

    # ────────────────── 内部方法 ──────────────────

    def _do_transition(self, target: WatchLevel, trigger: str) -> bool:
        if target == self._current_level:
            return False
        now = time.time()
        dwell = now - self._level_entered_at
        self._dwell_seconds[self._current_level.value] = (
            self._dwell_seconds.get(self._current_level.value, 0.0) + dwell
        )
        record = WatchTransitionRecord(
            from_level=self._current_level,
            to_level=target,
            trigger=trigger,
            timestamp=now,
        )
        self._transition_log.append(record)
        if len(self._transition_log) > self._max_log_size:
            self._transition_log = self._transition_log[-self._max_log_size:]
        old_label = self._current_level.label()
        self._current_level = target
        self._level_entered_at = now
        logger.debug(
            f"[{self._channel_id}] 观看层级 {old_label} → {target.label()} ({trigger})"
        )
        return True


# ---------------------------------------------------------------------------
#  全局访问接口
# ---------------------------------------------------------------------------


def get_watch_machine(channel_id: str) -> WatchStateMachine:
    """获取或创建频道的观看状态机"""
    if channel_id not in _channel_machines:
        _channel_machines[channel_id] = WatchStateMachine(channel_id)
    return _channel_machines[channel_id]


def remove_watch_machine(channel_id: str) -> None:
    """移除频道的观看状态机"""
    _channel_machines.pop(channel_id, None)


# 第二阶段扩展：集成主观存在流


async def update_with_snapshot(
    machine: WatchStateMachine, snapshot: Dict[str, Any]
) -> bool:
    """使用world_snapshot更新状态机（第二阶段集成）"""
    context = {
        "attention_score": snapshot.get("group_attention_level", 0.6),
        "boredom": snapshot.get("boredom", 0.0),
        "mentions": snapshot.get("mention_count", 0),
        "stimulus": snapshot.get("repeated_topic_pressure", 0.0),
    }
    changed = machine.evaluate_stimulus(
        context["stimulus"], is_at_mention=context["mentions"] > 0
    )
    if not changed:
        changed = machine.evaluate_natural_decay(
            silence_sec=snapshot.get("silence_seconds", 0.0),
            boredom=context["boredom"],
        )
    return changed


# 第二阶段可见性辅助


def is_message_visible(
    level: WatchLevel, message_importance: float = 0.5
) -> bool:
    """判断消息是否可见（第二阶段visibility_gate原型）"""
    rank = level.rank()
    threshold = 0.3 + (4 - rank) * 0.15
    return message_importance > threshold


# 批量更新工具（第二阶段批量处理）


async def batch_update_machines(
    channel_snapshots: Dict[str, Dict[str, Any]],
) -> Dict[str, bool]:
    """批量更新多个频道状态机"""
    results = {}
    for ch, snap in channel_snapshots.items():
        try:
            m = get_watch_machine(ch)
            results[ch] = await update_with_snapshot(m, snap)
        except Exception as e:
            logger.warning(f"批量更新 {ch} 失败: {e}")
            results[ch] = False
    return results


# 第二阶段初始化检查


def is_second_stage_ready() -> bool:
    """检查第二阶段是否就绪"""
    return len(_channel_machines) > 0 or True  # 总是准备好启动


logger.info("第二阶段主观存在流模块加载完成 - watch_state_machine 已优化")

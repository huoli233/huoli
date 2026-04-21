import time as _tm
from typing import Dict, Optional

from src.common.logger import get_logger

logger = get_logger("频率控制")


# ---------------------------------------------------------------------------
#  衰减参数
# ---------------------------------------------------------------------------
_DECAY_HALF_LIFE_SEC = 300.0  # 频率调整值半衰期（秒）
_MIN_FACTOR = 0.1
_MAX_FACTOR = 5.0
_DEFAULT_FACTOR = 1.0


class FrequencyControl:
    """频率控制器，管理单个频道的发言频率调整

    支持:
      - 手动加减频率
      - 基于消息长度自适应微调
      - 随时间自然衰减回归默认值
    """

    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.talk_frequency_adjust: float = _DEFAULT_FACTOR
        self._last_adjust_ts: float = _tm.time()
        self._cumulative_boost: float = 0.0

    def _apply_decay(self) -> None:
        """按半衰期对频率因子做自然衰减"""
        import math

        now = _tm.time()
        elapsed = now - self._last_adjust_ts
        if elapsed <= 0:
            return
        diff = self.talk_frequency_adjust - _DEFAULT_FACTOR
        if abs(diff) < 0.01:
            self.talk_frequency_adjust = _DEFAULT_FACTOR
            self._last_adjust_ts = now
            return
        decay = math.exp(-0.693 * elapsed / _DECAY_HALF_LIFE_SEC)
        self.talk_frequency_adjust = _DEFAULT_FACTOR + diff * decay
        self._last_adjust_ts = now

    def get_talk_frequency_adjust(self) -> float:
        """获取当前频率因子（先执行衰减）"""
        self._apply_decay()
        return self.talk_frequency_adjust

    def set_talk_frequency_adjust(self, value: float) -> None:
        """设置频率因子，限制在合理范围"""
        self.talk_frequency_adjust = max(_MIN_FACTOR, min(_MAX_FACTOR, value))
        self._last_adjust_ts = _tm.time()
        logger.debug(
            f"[{self.chat_id}] 频率因子→{self.talk_frequency_adjust:.2f}"
        )

    def increase_frequency(self, delta: float = 0.1) -> None:
        """提高发言频率"""
        self._apply_decay()
        self.set_talk_frequency_adjust(self.talk_frequency_adjust + delta)

    def decrease_frequency(self, delta: float = 0.1) -> None:
        """降低发言频率"""
        self._apply_decay()
        self.set_talk_frequency_adjust(self.talk_frequency_adjust - delta)

    def reset_frequency(self) -> None:
        """重置为默认值"""
        self.talk_frequency_adjust = _DEFAULT_FACTOR
        self._cumulative_boost = 0.0
        self._last_adjust_ts = _tm.time()

    def adapt_by_message_length(self, msg_length: int) -> None:
        """根据消息长度自适应微调

        长消息 → 稍微提高频率（对方在认真聊）
        极短消息 → 稍微降低频率（对方可能不想聊）
        """
        if msg_length > 80:
            self.increase_frequency(0.05)
        elif msg_length < 5:
            self.decrease_frequency(0.03)

    def adapt_by_interval(self, seconds_since_last_msg: float) -> None:
        """根据消息间隔自适应微调

        消息来得快 → 提高频率
        消息来得慢 → 降低频率
        """
        if seconds_since_last_msg < 5.0:
            self.increase_frequency(0.08)
        elif seconds_since_last_msg > 60.0:
            self.decrease_frequency(0.05)

    def effective_probability(self, base_prob: float) -> float:
        """计算有效回复概率 = 基础概率 × 频率因子"""
        factor = self.get_talk_frequency_adjust()
        return min(1.0, max(0.0, base_prob * factor))


class FrequencyControlManager:
    """频率控制管理器，管理多个频道的频率控制实例"""

    def __init__(self):
        self.frequency_control_dict: Dict[str, FrequencyControl] = {}

    def get_or_create_frequency_control(
        self, chat_id: str
    ) -> FrequencyControl:
        """获取或创建指定频道的频率控制实例"""
        if chat_id not in self.frequency_control_dict:
            self.frequency_control_dict[chat_id] = FrequencyControl(chat_id)
            logger.debug(f"创建频率控制实例: {chat_id}")
        return self.frequency_control_dict[chat_id]

    def remove_frequency_control(self, chat_id: str) -> bool:
        """移除指定频道的频率控制实例"""
        if chat_id in self.frequency_control_dict:
            del self.frequency_control_dict[chat_id]
            logger.debug(f"移除频率控制实例: {chat_id}")
            return True
        return False

    def cleanup_stale(self, max_idle_sec: float = 1800.0) -> int:
        """清理长时间未调整的频率控制实例"""
        now = _tm.time()
        stale = [
            cid
            for cid, ctrl in self.frequency_control_dict.items()
            if now - ctrl._last_adjust_ts > max_idle_sec
        ]
        for cid in stale:
            del self.frequency_control_dict[cid]
        if stale:
            logger.info(f"清理了 {len(stale)} 个过期频率控制实例")
        return len(stale)

    def get_all_chat_ids(self) -> list:
        """获取所有有频率控制的聊天ID"""
        return list(self.frequency_control_dict.keys())

    def adjust_all_frequency(self, factor: float) -> None:
        """对所有频道的频率因子乘以指定系数"""
        for ctrl in self.frequency_control_dict.values():
            ctrl.set_talk_frequency_adjust(ctrl.talk_frequency_adjust * factor)

    def summary(self) -> Dict[str, float]:
        """返回所有频道当前频率因子"""
        return {
            cid: ctrl.get_talk_frequency_adjust()
            for cid, ctrl in self.frequency_control_dict.items()
        }


# ---------------------------------------------------------------------------
#  模块级单例
# ---------------------------------------------------------------------------

frequency_control_manager = FrequencyControlManager()

_manager_ref: Optional[FrequencyControlManager] = None


def acquire_frequency_manager() -> FrequencyControlManager:
    global _manager_ref
    if _manager_ref is None:
        _manager_ref = FrequencyControlManager()
    return _manager_ref


# ============================================================
# D3 维度接口：频率控制
# 包装 FrequencyControlManager 使其符合 DimensionBase 统一协议
# ============================================================
from src.chat.heart_flow.dimension_protocol import (
    DimensionBase,
    DimensionScope,
    EventContext,
    TickResult,
)
from src.chat.heart_flow.vote_types import FrequencyVote


# 连续不回复计数追踪（per-channel）
_consecutive_skips: Dict[str, int] = {}
# 上次回复时间追踪（per-channel）
_last_reply_timestamps: Dict[str, float] = {}
# 连续不回复上限：超过此值应强制回复一次
_SKIP_LIMIT = 5


class FrequencyDimension(DimensionBase):
    """
    D3 频率控制维度。
    包装 FrequencyControlManager，将半衰期衰减+自适应微调
    转化为统一维度接口的 tick/on_event/vote 模式。
    """

    _singleton: Optional["FrequencyDimension"] = None

    @classmethod
    def get_instance(cls) -> "FrequencyDimension":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        self._manager = acquire_frequency_manager()

    @property
    def dimension_name(self) -> str:
        return "frequency_control"

    @property
    def scope(self) -> DimensionScope:
        return DimensionScope.PER_CHANNEL

    @property
    def needs_persistence(self) -> bool:
        return False

    @property
    def tick_interval_sec(self) -> float:
        return 5.0

    def tick(self, elapsed_sec: float) -> TickResult:
        """周期性驱动所有频道的频率衰减"""
        updated = False
        for chat_id, ctrl in self._manager.frequency_control_dict.items():
            old_val = ctrl.talk_frequency_adjust
            ctrl._apply_decay()
            if abs(ctrl.talk_frequency_adjust - old_val) > 0.001:
                updated = True
        # 清理过期实例
        cleaned = self._manager.cleanup_stale(max_idle_sec=1800.0)
        return TickResult(
            dimension_name=self.dimension_name,
            updated=updated or cleaned > 0,
            summary=f"频道数={len(self._manager.frequency_control_dict)}"
            if updated
            else "",
        )

    def on_event(self, ctx: EventContext):
        """
        接收消息事件并调整频率因子。
        message_received: 根据消息长度和间隔自适应调整
        reply_completed: 重置连续跳过计数，记录回复时间
        reply_skipped: 累加连续跳过计数
        """
        if not ctx.channel_id:
            return
        ctrl = self._manager.get_or_create_frequency_control(ctx.channel_id)
        if ctx.event_type == "message_received":
            ctrl.adapt_by_message_length(ctx.message_length)
            last_ts = _last_reply_timestamps.get(ctx.channel_id, 0.0)
            if last_ts > 0:
                interval = _tm.time() - last_ts
                ctrl.adapt_by_interval(interval)
        elif ctx.event_type == "reply_completed":
            _consecutive_skips[ctx.channel_id] = 0
            _last_reply_timestamps[ctx.channel_id] = _tm.time()
            ctrl.increase_frequency(0.02)
        elif ctx.event_type == "reply_skipped":
            _consecutive_skips[ctx.channel_id] = (
                _consecutive_skips.get(ctx.channel_id, 0) + 1
            )

    def vote(self, ctx: EventContext) -> FrequencyVote:
        """
        根据当前频率因子和跳过计数生成投票。
        频率因子 > 1.0 → 倾向回复，< 1.0 → 倾向沉默。
        连续跳过次数超过阈值时建议强制回复。
        """
        if not ctx.channel_id:
            return FrequencyVote()
        ctrl = self._manager.get_or_create_frequency_control(ctx.channel_id)
        factor = ctrl.get_talk_frequency_adjust()
        skips = _consecutive_skips.get(ctx.channel_id, 0)
        last_reply = _last_reply_timestamps.get(ctx.channel_id, 0.0)
        since_reply = _tm.time() - last_reply if last_reply > 0 else 999.0
        # 冷却判定：刚回复完5秒内算冷却
        in_cooldown = since_reply < 5.0
        # 连续跳过太多次 → 建议强制触发
        force_trigger = skips >= _SKIP_LIMIT
        # 概率乘数：直接使用频率因子（已经由半衰期衰减管理）
        prob = max(0.1, min(2.0, factor))
        if in_cooldown:
            prob *= 0.3
        return FrequencyVote(
            probability_factor=prob,
            force_trigger=force_trigger,
            adjust_factor=factor,
            dynamic_threshold=0.5,
            in_cooldown=in_cooldown,
            consecutive_skip_count=skips,
            skip_limit=_SKIP_LIMIT,
            seconds_since_last_reply=round(since_reply, 1),
            debug_reason=f"factor={factor:.2f} skips={skips} cd={in_cooldown}",
        )

    def calibrate(self, offline_seconds: float):
        """离线校准：离线超过10分钟重置所有频率"""
        if offline_seconds > 600:
            for ctrl in self._manager.frequency_control_dict.values():
                ctrl.reset_frequency()
            _consecutive_skips.clear()
            _last_reply_timestamps.clear()

    def get_state_summary(self) -> dict:
        return {
            "dimension": self.dimension_name,
            "scope": self.scope.value,
            "channel_count": len(self._manager.frequency_control_dict),
            "factors": self._manager.summary(),
            "consecutive_skips": dict(_consecutive_skips),
        }

    def reset(self, user_id: str = "", channel_id: str = ""):
        if channel_id:
            ctrl = self._manager.frequency_control_dict.get(channel_id)
            if ctrl:
                ctrl.reset_frequency()
            _consecutive_skips.pop(channel_id, None)
            _last_reply_timestamps.pop(channel_id, None)

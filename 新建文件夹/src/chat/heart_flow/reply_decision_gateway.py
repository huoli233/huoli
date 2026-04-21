import time as _tm
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.common.logger import get_logger

logger = get_logger("reply_gateway")


# ---------------------------------------------------------------------------
#  决策结果
# ---------------------------------------------------------------------------


@dataclass
class GatewayVerdict:
    """回复决策三值结果"""

    should_reply: bool = False
    reason: str = ""
    confidence: float = 0.0


# ---------------------------------------------------------------------------
#  回复历史记录
# ---------------------------------------------------------------------------


@dataclass
class _ReplyRecord:
    """单频道回复记录"""

    last_reply_ts: float = 0.0
    recent_reply_count: int = 0
    window_start: float = 0.0
    _WINDOW_SEC: float = 60.0
    consecutive_bot_replies: int = 0  # 连续机器人回复次数

    def log_reply(self) -> None:
        now = _tm.time()
        if now - self.window_start > self._WINDOW_SEC:
            self.window_start = now
            self.recent_reply_count = 0
        self.recent_reply_count += 1
        self.consecutive_bot_replies += 1
        self.last_reply_ts = now

    def on_user_message(self) -> None:
        """用户发了新消息，重置连续机器人回复计数"""
        self.consecutive_bot_replies = 0

    def replies_in_window(self) -> int:
        if _tm.time() - self.window_start > self._WINDOW_SEC:
            self.recent_reply_count = 0
            self.window_start = _tm.time()
        return self.recent_reply_count


# ---------------------------------------------------------------------------
#  触发/抑制规则
# ---------------------------------------------------------------------------


def _check_force_trigger(
    msg_text: str,
    bot_name: str,
    is_at_bot: bool,
) -> Optional[GatewayVerdict]:
    """检查强制触发条件：@机器人、直呼其名"""
    if is_at_bot:
        return GatewayVerdict(True, "被直接@", 1.0)
    if bot_name and bot_name in msg_text:
        return GatewayVerdict(True, "提到了名字", 0.95)
    return None


def _check_suppression(
    channel_id: str,
    record: _ReplyRecord,
    cooldown_sec: float,
    burst_limit: int,
) -> Optional[GatewayVerdict]:
    """检查强制抑制条件：冷却中、窗口刷屏上限、连续机器人回复"""
    elapsed = _tm.time() - record.last_reply_ts
    if elapsed < cooldown_sec:
        return GatewayVerdict(
            False, f"冷却中({elapsed:.1f}s/{cooldown_sec}s)", 1.0
        )
    if record.replies_in_window() >= burst_limit:
        return GatewayVerdict(False, f"窗口内已回复{burst_limit}次", 0.9)
    if record.consecutive_bot_replies >= burst_limit:
        return GatewayVerdict(
            False,
            f"连续机器人回复{record.consecutive_bot_replies}次，等用户响应",
            0.85,
        )
    return None


def _probability_evaluate(
    base_prob: float,
    msg_length: int,
    seconds_since_last: float,
) -> GatewayVerdict:
    """基于概率的软性评估"""
    import random

    adjusted = base_prob
    if msg_length > 50:
        adjusted = min(1.0, adjusted + 0.15)
    if seconds_since_last > 120:
        adjusted = min(1.0, adjusted + 0.1)
    roll = random.random()
    hit = roll < adjusted
    return GatewayVerdict(
        should_reply=hit,
        reason=f"概率评估 {adjusted:.2f} vs {roll:.2f}",
        confidence=adjusted,
    )


# ---------------------------------------------------------------------------
#  入口类
# ---------------------------------------------------------------------------


class ReplyDecisionGateway:
    """回复决策入口（单例）"""

    _sole_instance = None

    def __new__(cls):
        if cls._sole_instance is None:
            cls._sole_instance = super().__new__(cls)
            cls._sole_instance._channel_tracks = {}
        return cls._sole_instance

    def _track(self, channel_id: str) -> _ReplyRecord:
        if channel_id not in self._channel_tracks:
            self._channel_tracks[channel_id] = _ReplyRecord()
        return self._channel_tracks[channel_id]

    def should_reply(
        self,
        channel_id: str,
        msg_text: str,
        *,
        bot_name: str = "",
        is_at_bot: bool = False,
        cooldown_sec: float = 3.0,
        burst_limit: int = 6,
        base_prob: float = 0.5,
    ) -> GatewayVerdict:
        """三阶段判断入口

        阶段一: 强制触发（@、提名）
        阶段二: 强制抑制（冷却、刷屏）
        阶段三: 概率评估
        """
        # 阶段一
        forced = _check_force_trigger(msg_text, bot_name, is_at_bot)
        if forced is not None:
            logger.debug(f"[回复决策] {channel_id} 强制触发: {forced.reason}")
            return forced
        # 阶段二
        record = self._track(channel_id)
        suppressed = _check_suppression(
            channel_id, record, cooldown_sec, burst_limit
        )
        if suppressed is not None:
            logger.debug(f"[回复决策] {channel_id} 抑制: {suppressed.reason}")
            return suppressed
        # 阶段三
        elapsed = _tm.time() - record.last_reply_ts
        verdict = _probability_evaluate(base_prob, len(msg_text), elapsed)
        logger.debug(f"[回复决策] {channel_id} {verdict.reason}")
        return verdict

    def record_reply(self, channel_id: str) -> None:
        """记录一次机器人回复"""
        self._track(channel_id).log_reply()

    def notify_user_spoke(self, channel_id: str) -> None:
        """用户发了新消息，重置连续机器人回复计数。

        应在消息处理入口调用。
        """
        self._track(channel_id).on_user_message()
        # 同时通知裁决缓存活跃度变化
        try:
            from src.chat.heart_flow.heartflow_decision import (
                acquire_decision_maker,
            )

            acquire_decision_maker().notify_message_arrived(channel_id)
        except Exception as _e:
            logger.debug(f"异常: {_e}")


# ---------------------------------------------------------------------------
#  便捷获取
# ---------------------------------------------------------------------------


def acquire_reply_gateway() -> ReplyDecisionGateway:
    return ReplyDecisionGateway()

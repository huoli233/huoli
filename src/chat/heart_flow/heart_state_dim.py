import time
import math
from dataclasses import dataclass, field
from typing import Dict, Optional
from src.common.logger import get_logger
from src.chat.heart_flow.dimension_protocol import (
    DimensionBase,
    DimensionScope,
    EventContext,
    TickResult,
)
from src.chat.heart_flow.vote_types import HeartStateVote, HeartWaitPhase

logger = get_logger("heart_state_dim")


# ============================================================
# 超时阶梯配置（等待时长 → 主动发言概率）
# ============================================================
_TIMEOUT_ESCALATION = [
    # (最小等待秒, 最大等待秒, 概率乘数, 态度标签)
    (0.0, 120.0, 0.6, ""),
    (120.0, 300.0, 0.85, "mild_curiosity"),
    (300.0, 600.0, 1.2, "growing_concern"),
    (600.0, 1200.0, 1.5, "lonely_waiting"),
    (1200.0, float("inf"), 1.8, "abandoned_feeling"),
]


def _escalation_for_seconds(sec: float) -> tuple:
    """根据等待时长查找对应的概率乘数和态度标签"""
    for lo, hi, factor, tag in _TIMEOUT_ESCALATION:
        if lo <= sec < hi:
            return factor, tag
    return 1.0, ""


# ============================================================
# 频道等待记录
# ============================================================
@dataclass
class ChannelWaitRecord:
    """单频道的等待状态档案"""
    # 当前等待阶段
    phase: HeartWaitPhase = HeartWaitPhase.IDLE
    # 进入等待的时间戳（phase=WAITING 时有效）
    wait_started_at: float = 0.0
    # 最后一次收到用户消息的时间戳
    last_user_msg_ts: float = 0.0
    # 最后一次 bot 回复的时间戳
    last_bot_reply_ts: float = 0.0
    # 连续超时次数（用户连续未回复的轮次）
    consecutive_timeouts: int = 0
    # 本次等待的超时阈值（秒）
    current_timeout_sec: float = 300.0
    # 是否已触发过主动发言（每次超时只触发一次）
    initiated_this_round: bool = False
    # 累计主动搭话次数
    total_initiations: int = 0
    # 上次主动搭话时间戳
    last_initiation_ts: float = 0.0
    # 主动搭话冷却（秒）：避免连续搭话
    initiation_cooldown_sec: float = 180.0


# ============================================================
# D8 维度实现：心流等待状态
# ============================================================
class HeartStateDimension(DimensionBase):
    """
    D8 心流等待状态维度。
    追踪每个频道的对话等待状态：
      - bot 完成回复后进入 WAITING 阶段（期待用户回应）
      - 用户消息到达后重回 IDLE 阶段
      - 超过阈值未收到回复则进入 TIMEOUT 阶段
    TIMEOUT 阶段根据等待时长分级：
      短暂超时 → 微弱好奇
      中等超时 → 想搭话
      长时超时 → 孤独感，可触发主动发言
    作用域：PER_CHANNEL（所有用户共享同一频道等待状态）
    """

    _singleton: Optional["HeartStateDimension"] = None

    @classmethod
    def get_instance(cls) -> "HeartStateDimension":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        # channel_id → ChannelWaitRecord
        self._records: Dict[str, ChannelWaitRecord] = {}
        # 基础超时阈值（秒）
        self._base_timeout_sec: float = 300.0
        # 连续超时时缩短阈值的衰减系数（越超时越快判超时）
        self._timeout_decay_rate: float = 0.85
        # 最短超时阈值
        self._min_timeout_sec: float = 60.0
        # 最长超时阈值
        self._max_timeout_sec: float = 900.0
        # 主动搭话全局冷却（秒）
        self._global_initiation_cd: float = 180.0
        # 单频道最大连续超时次数（超过后不再主动搭话，避免打扰）
        self._max_consecutive_before_silence: int = 3

    @property
    def dimension_name(self) -> str:
        return "heart_state"

    @property
    def scope(self) -> DimensionScope:
        return DimensionScope.PER_CHANNEL

    @property
    def needs_persistence(self) -> bool:
        return True

    @property
    def tick_interval_sec(self) -> float:
        return 5.0

    def _get_record(self, channel_id: str) -> ChannelWaitRecord:
        if channel_id not in self._records:
            self._records[channel_id] = ChannelWaitRecord(
                last_user_msg_ts=time.time(),
                current_timeout_sec=self._base_timeout_sec,
            )
        return self._records[channel_id]

    def tick(self, elapsed_sec: float) -> TickResult:
        """检查所有频道的等待状态，判定超时转换"""
        now = time.time()
        transitioned = 0
        for channel_id, record in self._records.items():
            if record.phase != HeartWaitPhase.WAITING:
                continue
            wait_elapsed = now - record.wait_started_at
            if wait_elapsed >= record.current_timeout_sec:
                record.phase = HeartWaitPhase.TIMEOUT
                record.consecutive_timeouts += 1
                record.initiated_this_round = False
                transitioned += 1
                logger.debug(
                    "频道 %s 等待超时(%.0fs)，连续超时=%d",
                    channel_id[:8],
                    wait_elapsed,
                    record.consecutive_timeouts,
                )
        updated = transitioned > 0
        return TickResult(
            dimension_name=self.dimension_name,
            updated=updated,
            summary=f"转换超时={transitioned}" if updated else "",
        )

    def on_event(self, ctx: EventContext):
        """
        根据事件更新等待状态。
        消息到达 → 结束等待，回到 IDLE
        回复完成 → 进入 WAITING
        """
        if not ctx.channel_id:
            return
        record = self._get_record(ctx.channel_id)
        now = time.time()
        if ctx.event_type == "message_received":
            # 收到用户消息，无论当前什么状态都回到 IDLE
            was_timeout = record.phase == HeartWaitPhase.TIMEOUT
            record.phase = HeartWaitPhase.IDLE
            record.last_user_msg_ts = now
            record.initiated_this_round = False
            if was_timeout:
                # 超时后用户回来了，连续超时计数重置
                record.consecutive_timeouts = 0
                logger.debug(
                    "频道 %s 用户归来，超时计数已重置",
                    ctx.channel_id[:8],
                )
            # 恢复基础超时阈值
            record.current_timeout_sec = self._base_timeout_sec
        elif ctx.event_type == "reply_completed":
            # bot 完成回复，进入等待
            record.phase = HeartWaitPhase.WAITING
            record.wait_started_at = now
            record.last_bot_reply_ts = now
            record.initiated_this_round = False
            # 连续超时时缩短下次等待阈值
            if record.consecutive_timeouts > 0:
                shrink = self._timeout_decay_rate ** record.consecutive_timeouts
                new_threshold = self._base_timeout_sec * shrink
                record.current_timeout_sec = max(
                    self._min_timeout_sec,
                    min(self._max_timeout_sec, new_threshold),
                )
            else:
                record.current_timeout_sec = self._base_timeout_sec
            logger.debug(
                "频道 %s 开始等待(阈值=%.0fs)",
                ctx.channel_id[:8],
                record.current_timeout_sec,
            )

    def vote(self, ctx: EventContext) -> HeartStateVote:
        """
        根据频道等待状态输出投票。
        IDLE → 中性，不干预概率
        WAITING → 降低概率（bot 刚说过话，不要频繁插嘴）
        TIMEOUT → 根据超时时长分级，长时间超时可触发主动搭话
        """
        if not ctx.channel_id:
            return HeartStateVote()
        _hb_will = 0.0
        try:
            from src.core.emotion_feedback_loop import get_emotion_feedback_loop
            _efl = get_emotion_feedback_loop()
            _efl_state = _efl.get_state(ctx.channel_id)
            if _efl_state:
                _hb_will = round(_efl_state.proactive_willingness, 3)
        except Exception as _exc:
            logger.debug(f"非关键异常: {_exc}")
        record = self._get_record(ctx.channel_id)
        now = time.time()
        if record.phase == HeartWaitPhase.IDLE:
            # 空闲阶段：检查距上次互动的时间，长时间无互动时轻微提升主动意愿
            idle_sec = now - record.last_user_msg_ts if record.last_user_msg_ts > 0 else 0.0
            should_initiate = False
            factor = 1.0
            attitude = ""
            if idle_sec > 1800.0:
                # 超过30分钟无任何互动，轻微提升主动概率
                factor = 1.1
                attitude = "quiet_reflection"
                # 但只有连续超时不多时才主动
                if (
                    record.consecutive_timeouts < self._max_consecutive_before_silence
                    and not record.initiated_this_round
                    and (now - record.last_initiation_ts) > self._global_initiation_cd
                ):
                    should_initiate = True
            return HeartStateVote(
                probability_factor=factor,
                attitude_tag=attitude,
                wait_phase=HeartWaitPhase.IDLE,
                should_initiate=should_initiate,
                waiting_seconds=0.0,
                timeout_threshold=record.current_timeout_sec,
                last_interaction_channel=ctx.channel_id,
                heartbeat_proactive_willingness=_hb_will,
                debug_reason=f"idle(空闲{idle_sec:.0f}s)",
            )
        elif record.phase == HeartWaitPhase.WAITING:
            wait_sec = now - record.wait_started_at
            progress = min(1.0, wait_sec / record.current_timeout_sec) if record.current_timeout_sec > 0 else 0.0
            dampening = 0.3 + 0.5 * progress
            return HeartStateVote(
                probability_factor=round(dampening, 3),
                wait_phase=HeartWaitPhase.WAITING,
                waiting_seconds=round(wait_sec, 1),
                timeout_threshold=record.current_timeout_sec,
                last_interaction_channel=ctx.channel_id,
                heartbeat_proactive_willingness=_hb_will,
                style_hint="耐心等待中" if progress < 0.6 else "等待中略显焦虑",
                debug_reason=f"waiting({wait_sec:.0f}s/{record.current_timeout_sec:.0f}s, progress={progress:.2f})",
            )
        else:
            # TIMEOUT 阶段：超时未收到回复
            wait_sec = now - record.wait_started_at
            escalation_factor, escalation_tag = _escalation_for_seconds(wait_sec)
            # 检查是否应该主动搭话
            should_initiate = False
            force_trigger = False
            if (
                not record.initiated_this_round
                and record.consecutive_timeouts < self._max_consecutive_before_silence
                and (now - record.last_initiation_ts) > self._global_initiation_cd
            ):
                should_initiate = True
                # 长时间超时（>10分钟）且连续超时不多时，触发强制回复
                if wait_sec > 600.0 and record.consecutive_timeouts <= 1:
                    force_trigger = True
                # 标记本轮已触发
                record.initiated_this_round = True
                record.total_initiations += 1
                record.last_initiation_ts = now
            # 连续超时过多时主动沉默
            if record.consecutive_timeouts >= self._max_consecutive_before_silence:
                escalation_factor = 0.3
                escalation_tag = "giving_up_waiting"
                should_initiate = False
                force_trigger = False
            return HeartStateVote(
                probability_factor=round(escalation_factor, 3),
                force_trigger=force_trigger,
                attitude_tag=escalation_tag,
                wait_phase=HeartWaitPhase.TIMEOUT,
                should_initiate=should_initiate,
                waiting_seconds=round(wait_sec, 1),
                timeout_threshold=record.current_timeout_sec,
                last_interaction_channel=ctx.channel_id,
                heartbeat_proactive_willingness=_hb_will,
                style_hint=self._timeout_style_hint(wait_sec, record.consecutive_timeouts),
                debug_reason=(
                    f"timeout({wait_sec:.0f}s, consecutive={record.consecutive_timeouts}, "
                    f"initiate={should_initiate}, force={force_trigger})"
                ),
            )

    def _timeout_style_hint(self, wait_sec: float, consecutive: int) -> str:
        """根据超时时长和连续次数生成风格提示"""
        if consecutive >= 3:
            return "已多次等待无回应，保持沉默观察"
        if wait_sec > 1200:
            return "长时间未收到回复，语气中带有一丝失落"
        if wait_sec > 600:
            return "等了很久没有回应，略显不安"
        if wait_sec > 300:
            return "已等待较长时间，可以轻声试探"
        return "刚刚超时，可以表达轻微好奇"

    def serialize(self) -> dict:
        result = {}
        for channel_id, rec in self._records.items():
            result[channel_id] = {
                "phase": rec.phase.value,
                "wait_at": rec.wait_started_at,
                "last_user": rec.last_user_msg_ts,
                "last_bot": rec.last_bot_reply_ts,
                "consec": rec.consecutive_timeouts,
                "timeout_th": rec.current_timeout_sec,
                "total_init": rec.total_initiations,
                "last_init_ts": rec.last_initiation_ts,
            }
        return result

    def deserialize(self, data: dict):
        if not isinstance(data, dict):
            return
        for channel_id, vals in data.items():
            if not isinstance(vals, dict):
                continue
            phase_str = vals.get("phase", "idle")
            try:
                phase = HeartWaitPhase(phase_str)
            except ValueError:
                phase = HeartWaitPhase.IDLE
            self._records[channel_id] = ChannelWaitRecord(
                phase=phase,
                wait_started_at=float(vals.get("wait_at", 0.0)),
                last_user_msg_ts=float(vals.get("last_user", 0.0)),
                last_bot_reply_ts=float(vals.get("last_bot", 0.0)),
                consecutive_timeouts=int(vals.get("consec", 0)),
                current_timeout_sec=max(self._min_timeout_sec, float(vals.get("timeout_th", self._base_timeout_sec))),
                total_initiations=int(vals.get("total_init", 0)),
                last_initiation_ts=float(vals.get("last_init_ts", 0.0)),
            )

    def calibrate(self, offline_seconds: float):
        """
        离线校准：
        - 所有等待中的频道自动超时
        - 连续超时计数递增
        - TIMEOUT 频道保持但不触发主动搭话
        """
        for record in self._records.values():
            if record.phase == HeartWaitPhase.WAITING:
                record.phase = HeartWaitPhase.TIMEOUT
                record.consecutive_timeouts += 1
                record.initiated_this_round = True
            elif record.phase == HeartWaitPhase.TIMEOUT:
                # 离线期间超时状态回归空闲
                if offline_seconds > 3600:
                    record.phase = HeartWaitPhase.IDLE
                    record.initiated_this_round = False

    def get_state_summary(self) -> dict:
        channel_summaries = {}
        for ch, rec in list(self._records.items())[:20]:
            channel_summaries[ch[:12]] = {
                "phase": rec.phase.value,
                "consec_timeout": rec.consecutive_timeouts,
                "total_init": rec.total_initiations,
            }
        phase_counts = {"idle": 0, "waiting": 0, "timeout": 0}
        for rec in self._records.values():
            phase_counts[rec.phase.value] = phase_counts.get(rec.phase.value, 0) + 1
        return {
            "dimension": self.dimension_name,
            "scope": self.scope.value,
            "channel_count": len(self._records),
            "phase_distribution": phase_counts,
            "channels": channel_summaries,
        }

    def reset(self, user_id: str = "", channel_id: str = ""):
        """重置指定频道的等待状态"""
        if channel_id and channel_id in self._records:
            del self._records[channel_id]
        elif not channel_id:
            self._records.clear()

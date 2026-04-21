import time as _tm
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional

from src.common.logger import get_logger

logger = get_logger("user_probe")


# ---------------------------------------------------------------------------
#  用户意图枚举
# ---------------------------------------------------------------------------


class UserEngagement(Enum):
    """用户参与意愿"""

    EAGER = "积极聊天"
    NEUTRAL = "一般互动"
    RELUCTANT = "不太想聊"
    HOSTILE = "带有攻击性"
    ABSENT = "已离线/沉默"


# ---------------------------------------------------------------------------
#  交互记录
# ---------------------------------------------------------------------------


@dataclass
class InteractionEvent:
    """单条交互事件"""

    timestamp: float
    text: str
    length: int
    is_question: bool = False
    is_at_bot: bool = False


@dataclass
class EngagementReport:
    """分析报告"""

    engagement: UserEngagement = UserEngagement.NEUTRAL
    confidence: float = 0.0
    short_summary: str = ""
    msg_frequency: float = 0.0
    avg_msg_length: float = 0.0
    question_ratio: float = 0.0


# ---------------------------------------------------------------------------
#  维度打分
# ---------------------------------------------------------------------------


def _score_frequency(
    events: List[InteractionEvent], window_sec: float = 120.0
) -> float:
    """消息频率得分 0~1，频率越高分越高"""
    if not events:
        return 0.0
    now = _tm.time()
    recent = [e for e in events if now - e.timestamp < window_sec]
    count = len(recent)
    if count == 0:
        return 0.0
    if count >= 8:
        return 1.0
    return min(1.0, count / 8.0)


def _score_content_richness(events: List[InteractionEvent]) -> float:
    """消息内容丰富度得分 0~1"""
    if not events:
        return 0.0
    avg_len = sum(e.length for e in events) / len(events)
    if avg_len > 50:
        return 1.0
    if avg_len > 20:
        return 0.6
    if avg_len > 5:
        return 0.3
    return 0.1


def _score_question_ratio(events: List[InteractionEvent]) -> float:
    """提问比例得分"""
    if not events:
        return 0.0
    q_count = sum(1 for e in events if e.is_question)
    return q_count / len(events)


def _score_mention_ratio(events: List[InteractionEvent]) -> float:
    """@机器人比例得分"""
    if not events:
        return 0.0
    at_count = sum(1 for e in events if e.is_at_bot)
    return at_count / len(events)


def _score_time_of_day() -> float:
    """根据当前时段给出活跃度预期 0~1"""
    import datetime

    hour = datetime.datetime.now().hour
    if 0 <= hour < 6:
        return 0.2
    if 6 <= hour < 9:
        return 0.5
    if 9 <= hour < 12:
        return 0.8
    if 12 <= hour < 14:
        return 0.6
    if 14 <= hour < 18:
        return 0.8
    if 18 <= hour < 22:
        return 0.9
    return 0.4


def _detect_hostility(events: List[InteractionEvent]) -> float:
    """攻击性检测 0~1，依赖结构强度而不是固定词表"""
    if not events:
        return 0.0
    hostility_score = 0.0
    for e in events:
        text = (e.text or "").strip()
        if not text:
            continue
        exclamations = text.count("!") + text.count("！")
        all_caps_ratio = sum(
            1 for ch in text if ch.isalpha() and ch.isupper()
        ) / max(1, sum(1 for ch in text if ch.isalpha()))
        repeated_punct = (
            1.0
            if "!!" in text or "！！" in text or "??" in text or "？？" in text
            else 0.0
        )
        short_burst = 1.0 if len(text) <= 8 and exclamations > 0 else 0.0
        hostility_score += min(
            1.0,
            exclamations * 0.2
            + all_caps_ratio * 0.8
            + repeated_punct * 0.5
            + short_burst * 0.3,
        )
    return min(1.0, hostility_score / len(events))


# ---------------------------------------------------------------------------
#  综合评估
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "frequency": 0.30,
    "richness": 0.20,
    "question": 0.15,
    "mention": 0.15,
    "time": 0.10,
    "hostility": 0.10,
}


def _weighted_engagement(events: List[InteractionEvent]) -> EngagementReport:
    """多维度加权评估"""
    freq = _score_frequency(events)
    rich = _score_content_richness(events)
    qr = _score_question_ratio(events)
    mr = _score_mention_ratio(events)
    tod = _score_time_of_day()
    host = _detect_hostility(events)
    # 攻击性是负面指标，取反
    positive_score = (
        _WEIGHTS["frequency"] * freq
        + _WEIGHTS["richness"] * rich
        + _WEIGHTS["question"] * qr
        + _WEIGHTS["mention"] * mr
        + _WEIGHTS["time"] * tod
    )
    hostility_penalty = _WEIGHTS["hostility"] * host
    final = max(0.0, positive_score - hostility_penalty)
    # 映射到枚举
    if host > 0.5:
        engagement = UserEngagement.HOSTILE
    elif final > 0.65:
        engagement = UserEngagement.EAGER
    elif final > 0.35:
        engagement = UserEngagement.NEUTRAL
    elif final > 0.1:
        engagement = UserEngagement.RELUCTANT
    else:
        engagement = UserEngagement.ABSENT
    avg_len = sum(e.length for e in events) / len(events) if events else 0
    return EngagementReport(
        engagement=engagement,
        confidence=min(1.0, final + 0.2) if events else 0.0,
        short_summary=f"freq={
            freq:.2f} rich={
                rich:.2f} q={
                    qr:.2f} at={
                        mr:.2f} host={
                            host:.2f}",
        msg_frequency=freq,
        avg_msg_length=avg_len,
        question_ratio=qr,
    )


# ---------------------------------------------------------------------------
#  用户状态探测器
# ---------------------------------------------------------------------------

_HISTORY_CAPACITY = 50
_STALE_EVENT_SEC = 7200.0  # 超过2小时的事件视为过期


class UserEngagementProbe:
    """追踪用户消息并分析参与意愿

    支持将分析结果保存到 persistent_state_db，恢复时回注。
    """

    def __init__(self):
        self._histories: Dict[str, Deque[InteractionEvent]] = {}

    def _history(self, user_key: str) -> Deque[InteractionEvent]:
        if user_key not in self._histories:
            self._histories[user_key] = deque(maxlen=_HISTORY_CAPACITY)
        return self._histories[user_key]

    def record_message(
        self,
        user_key: str,
        text: str,
        *,
        is_at_bot: bool = False,
    ) -> None:
        """记录一条用户消息"""
        question_marks = text.count("？") + text.count("?")
        trailing_pause = text.rstrip().endswith(("吗", "呢", "么", "吧"))
        is_q = question_marks > 0 or trailing_pause
        evt = InteractionEvent(
            timestamp=_tm.time(),
            text=text,
            length=len(text),
            is_question=is_q,
            is_at_bot=is_at_bot,
        )
        self._history(user_key).append(evt)

    def analyze(self, user_key: str) -> EngagementReport:
        """分析用户当前参与意愿"""
        self._prune_stale(user_key)
        history = self._history(user_key)
        events = list(history)
        if not events:
            return EngagementReport(
                engagement=UserEngagement.ABSENT,
                short_summary="无历史消息",
            )
        report = _weighted_engagement(events)
        logger.debug(
            f"[用户探测] {user_key} → {report.engagement.value} "
            f"({report.short_summary})"
        )
        return report

    def _prune_stale(self, user_key: str) -> None:
        """清除超过 _STALE_EVENT_SEC 的旧事件"""
        history = self._histories.get(user_key)
        if not history:
            return
        cutoff = _tm.time() - _STALE_EVENT_SEC
        while history and history[0].timestamp < cutoff:
            history.popleft()

    def recent_text_joined(self, user_key: str, limit: int = 10) -> str:
        """获取用户最近消息拼接文本，供LLM参考"""
        history = self._history(user_key)
        recent = list(history)[-limit:]
        return "\n".join(e.text for e in recent)

    def clear_user(self, user_key: str) -> None:
        """清除用户历史"""
        self._histories.pop(user_key, None)

    # ---- 持久化集成 ----

    def persist_report(self, user_key: str) -> None:
        """将当前分析结果写入 persistent_state_db 的 engagement 分区"""
        report = self.analyze(user_key)
        payload = {
            "engagement": report.engagement.value,
            "confidence": report.confidence,
            "summary": report.short_summary,
            "msg_frequency": report.msg_frequency,
            "avg_msg_length": report.avg_msg_length,
            "question_ratio": report.question_ratio,
            "saved_at": _tm.time(),
        }
        try:
            from src.chat.heart_flow.persistent_state_db import (
                acquire_persistent_store,
            )

            acquire_persistent_store().save_engagement(user_key, payload)
        except Exception as exc:
            logger.debug(f"[用户探测] 持久化失败 {user_key}: {exc}")

    def restore_report(self, user_key: str) -> Optional[EngagementReport]:
        """从 persistent_state_db 读取上次保存的参与度快照。

        仅在内存历史为空时有用（例如重启后）。
        """
        try:
            from src.chat.heart_flow.persistent_state_db import (
                acquire_persistent_store,
            )

            data = acquire_persistent_store().load_engagement(user_key)
            if not data:
                return None
            engagement_str = data.get("engagement", "")
            eng = UserEngagement.NEUTRAL
            for member in UserEngagement:
                if member.value == engagement_str:
                    eng = member
                    break
            return EngagementReport(
                engagement=eng,
                confidence=float(data.get("confidence", 0.0)),
                short_summary=str(data.get("summary", "")),
                msg_frequency=float(data.get("msg_frequency", 0.0)),
                avg_msg_length=float(data.get("avg_msg_length", 0.0)),
                question_ratio=float(data.get("question_ratio", 0.0)),
            )
        except Exception as exc:
            logger.debug(f"[用户探测] 恢复失败 {user_key}: {exc}")
            return None

    def persist_all_active(self) -> int:
        """批量持久化所有有历史的用户，返回保存数"""
        saved = 0
        for user_key in list(self._histories.keys()):
            if self._histories[user_key]:
                self.persist_report(user_key)
                saved += 1
        return saved


# ---------------------------------------------------------------------------
#  单例
# ---------------------------------------------------------------------------

_probe_ref: Optional[UserEngagementProbe] = None


def acquire_engagement_probe() -> UserEngagementProbe:
    global _probe_ref
    if _probe_ref is None:
        _probe_ref = UserEngagementProbe()
    return _probe_ref


# ============================================================
# D4 维度接口：用户意愿检测
# 包装 UserEngagementProbe 使其符合 DimensionBase 统一协议
# ============================================================
from src.chat.heart_flow.dimension_protocol import (
    DimensionBase,
    DimensionScope,
    EventContext,
    TickResult,
)
from src.chat.heart_flow.vote_types import (
    UserStateVote,
    UserEngagementState,
)


# 内部映射：UserEngagement枚举 → UserEngagementState枚举
_ENGAGEMENT_MAP = {
    UserEngagement.EAGER: UserEngagementState.EAGER,
    UserEngagement.NEUTRAL: UserEngagementState.NEUTRAL,
    UserEngagement.RELUCTANT: UserEngagementState.RELUCTANT,
    UserEngagement.HOSTILE: UserEngagementState.HOSTILE,
    UserEngagement.ABSENT: UserEngagementState.ABSENT,
}

# 参与度 → 概率乘数映射
_ENGAGEMENT_PROB = {
    UserEngagementState.EAGER: 1.2,
    UserEngagementState.NEUTRAL: 1.0,
    UserEngagementState.RELUCTANT: 0.6,
    UserEngagementState.HOSTILE: 0.3,
    UserEngagementState.ABSENT: 0.1,
}


class UserStateDimension(DimensionBase):
    """
    D4 用户意愿检测维度。
    包装 UserEngagementProbe，将6因子加权分析
    转化为统一维度接口的投票模式。

    检测因子(权重):
      频率30% + 内容丰富度20% + 提问比15% + @比15% + 时段10% + 攻击性10%
    输出五态: EAGER / NEUTRAL / RELUCTANT / HOSTILE / ABSENT
    """

    _singleton: Optional["UserStateDimension"] = None

    @classmethod
    def get_instance(cls) -> "UserStateDimension":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        self._probe = acquire_engagement_probe()
        # 缓存分析结果（避免频繁重复计算）
        self._report_cache: Dict[str, EngagementReport] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl: float = 30.0

    @property
    def dimension_name(self) -> str:
        return "user_state"

    @property
    def scope(self) -> DimensionScope:
        return DimensionScope.PER_USER

    @property
    def needs_persistence(self) -> bool:
        return False

    @property
    def tick_interval_sec(self) -> float:
        return 10.0

    def tick(self, elapsed_sec: float) -> TickResult:
        """周期性清理过期事件和过期缓存"""
        cleaned_users = 0
        now = _tm.time()
        for user_key in list(self._probe._histories.keys()):
            self._probe._prune_stale(user_key)
            history = self._probe._histories.get(user_key)
            if history is not None and len(history) == 0:
                del self._probe._histories[user_key]
                cleaned_users += 1
        # 清理过期缓存
        expired_keys = [
            k for k, ts in self._cache_timestamps.items()
            if now - ts > self._cache_ttl * 2
        ]
        for k in expired_keys:
            self._report_cache.pop(k, None)
            self._cache_timestamps.pop(k, None)
        return TickResult(
            dimension_name=self.dimension_name,
            updated=cleaned_users > 0,
            summary=f"活跃用户={len(self._probe._histories)} 清理={cleaned_users}"
            if cleaned_users > 0
            else "",
        )

    def on_event(self, ctx: EventContext):
        """
        接收消息事件并记录到用户历史。
        message_received: 记录用户消息到滑窗。
        """
        if ctx.event_type == "message_received" and ctx.user_id:
            user_key = ctx.user_id
            self._probe.record_message(
                user_key,
                ctx.message_text,
                is_at_bot=ctx.is_at_bot,
            )
            # 使缓存失效
            self._report_cache.pop(user_key, None)
            self._cache_timestamps.pop(user_key, None)

    def _get_cached_report(self, user_key: str) -> EngagementReport:
        """获取（带缓存的）用户参与度分析报告"""
        now = _tm.time()
        cached_ts = self._cache_timestamps.get(user_key, 0.0)
        if now - cached_ts < self._cache_ttl:
            cached = self._report_cache.get(user_key)
            if cached is not None:
                return cached
        report = self._probe.analyze(user_key)
        self._report_cache[user_key] = report
        self._cache_timestamps[user_key] = now
        return report

    def vote(self, ctx: EventContext) -> UserStateVote:
        """
        根据用户参与度分析生成投票。
        EAGER → 提高概率, HOSTILE → 大幅降低概率, ABSENT → 几乎不回复。
        """
        if not ctx.user_id:
            return UserStateVote()
        report = self._get_cached_report(ctx.user_id)
        engagement_state = _ENGAGEMENT_MAP.get(
            report.engagement, UserEngagementState.NEUTRAL
        )
        prob = _ENGAGEMENT_PROB.get(engagement_state, 1.0)
        # 攻击性检测
        structural_hostility = engagement_state == UserEngagementState.HOSTILE
        # 置信度
        want_confidence = report.confidence
        # 态度标签
        attitude = ""
        if structural_hostility:
            attitude = "defensive"
        elif engagement_state == UserEngagementState.EAGER:
            attitude = "enthusiastic"
        elif engagement_state == UserEngagementState.RELUCTANT:
            attitude = "reserved"
        return UserStateVote(
            probability_factor=prob,
            force_refuse=False,
            attitude_tag=attitude,
            engagement=engagement_state,
            want_to_chat_confidence=round(want_confidence, 2),
            msg_frequency=round(report.msg_frequency, 2),
            avg_msg_length=round(report.avg_msg_length, 1),
            question_ratio=round(report.question_ratio, 2),
            structural_hostility=structural_hostility,
            debug_reason=report.short_summary,
        )

    def calibrate(self, offline_seconds: float):
        """离线校准: 超过2小时清空所有缓存和历史"""
        if offline_seconds > 7200:
            self._probe._histories.clear()
            self._report_cache.clear()
            self._cache_timestamps.clear()

    def get_state_summary(self) -> dict:
        summaries = {}
        for user_key in list(self._probe._histories.keys()):
            report = self._get_cached_report(user_key)
            summaries[user_key[:8]] = {
                "engagement": report.engagement.value,
                "confidence": round(report.confidence, 2),
            }
        return {
            "dimension": self.dimension_name,
            "scope": self.scope.value,
            "tracked_users": len(self._probe._histories),
            "users": summaries,
        }

    def reset(self, user_id: str = "", channel_id: str = ""):
        if user_id:
            self._probe.clear_user(user_id)
            self._report_cache.pop(user_id, None)
            self._cache_timestamps.pop(user_id, None)

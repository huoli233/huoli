import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("可见性门控")

_gate_instances: Dict[str, "VisibilityGate"] = {}


class VisibilityLevel(Enum):
    """消息可见性层级"""

    RECEIVED = "received"
    GLANCED = "glanced"
    NOTICED = "noticed"
    FOCUSED = "focused"
    UNDERSTOOD_BUT_IGNORED = "understood_but_ignored"
    DEFERRED = "deferred"

    def rank(self) -> int:
        _ranks = {
            "received": 0,
            "glanced": 1,
            "noticed": 2,
            "focused": 3,
            "understood_but_ignored": 3,
            "deferred": 2,
        }
        return _ranks.get(self.value, 0)

    def enters_understanding(self) -> bool:
        """该层级是否进入理解链"""
        return self in (
            VisibilityLevel.FOCUSED,
            VisibilityLevel.UNDERSTOOD_BUT_IGNORED,
        )

    def label(self) -> str:
        _labels = {
            "received": "系统收到",
            "glanced": "瞟到了",
            "noticed": "注意到了",
            "focused": "认真在看",
            "understood_but_ignored": "看懂了但不想理",
            "deferred": "先存着回头看",
        }
        return _labels.get(self.value, "")


@dataclass
class VisibilityDecision:
    """对单条消息的可见性判定结果"""

    message_id: str = ""
    user_id: str = ""
    level: VisibilityLevel = VisibilityLevel.RECEIVED
    score: float = 0.0
    enter_understanding: bool = False
    reasons: List[str] = field(default_factory=list)
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "level": self.level.value,
            "level_label": self.level.label(),
            "score": round(self.score, 3),
            "enter_understanding": self.enter_understanding,
            "reasons": self.reasons[:5],
        }


# ---------------------------------------------------------------------------
#  各加分/扣分因子的权重常量
# ---------------------------------------------------------------------------
_WEIGHT_AT_MENTION = 0.45
_WEIGHT_QUOTE_REPLY = 0.35
_WEIGHT_NAME_MENTION = 0.30
_WEIGHT_CLOSE_FRIEND = 0.20
_WEIGHT_INTERESTING_TOPIC = 0.15
_WEIGHT_HIGH_ACTIVITY = 0.10
_WEIGHT_NEW_USER = 0.08
_WEIGHT_IMAGE_MEDIA = 0.05
# 扣分因子
_PENALTY_LOW_WATCH = 0.25
_PENALTY_QUIET_PREFERENCE = 0.15
_PENALTY_HIGH_ANNOYANCE = 0.12
_PENALTY_LOW_ENERGY = 0.10
_PENALTY_HIGH_LOAFING = 0.35
_PENALTY_MODERATE_LOAFING = 0.20
# 层级阈值
_THRESHOLD_GLANCED = 0.15
_THRESHOLD_NOTICED = 0.35
_THRESHOLD_FOCUSED = 0.55


class VisibilityGate:
    """消息可见性门控
    每条到达的消息经过此门控后，获得一个可见性层级。
    只有达到阈值的消息才能进入主链理解层。
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._recent_decisions: List[VisibilityDecision] = []
        self._max_cache = 50

    def evaluate(
        self,
        message: Any,
        *,
        watch_level_rank: int = 2,
        presence_state: Optional[Any] = None,
        user_relationship_level: int = 2,
        user_trust: float = 0.0,
        topic_relevance: float = 0.0,
        is_at_mention: bool = False,
        is_quote_reply: bool = False,
        is_name_mention: bool = False,
        has_media: bool = False,
        is_new_user: bool = False,
        loafing_level: float = 0.0,
    ) -> VisibilityDecision:
        """评估单条消息的可见性层级

        Args:
            loafing_level: 摸鱼值 (0-100)，高摸鱼时普通消息应被降权
        """
        msg_id = str(getattr(message, "message_id", "") or "")
        user_id = str(getattr(message, "user_id", "") or "")
        reasons: List[str] = []
        score = 0.0
        # 加分项
        if is_at_mention:
            score += _WEIGHT_AT_MENTION
            reasons.append("被@提及")
        if is_quote_reply:
            score += _WEIGHT_QUOTE_REPLY
            reasons.append("引用回复")
        if is_name_mention:
            score += _WEIGHT_NAME_MENTION
            reasons.append("提到名字")
        if user_relationship_level >= 4:
            score += _WEIGHT_CLOSE_FRIEND
            reasons.append("亲密用户")
        elif user_relationship_level >= 3:
            score += _WEIGHT_CLOSE_FRIEND * 0.5
            reasons.append("友好用户")
        if topic_relevance > 0.5:
            score += _WEIGHT_INTERESTING_TOPIC * topic_relevance
            reasons.append(f"话题相关({topic_relevance:.2f})")
        if user_trust > 5.0:
            score += min(0.12, user_trust * 0.01)
            reasons.append("高信任用户")
        if is_new_user:
            score += _WEIGHT_NEW_USER
            reasons.append("新用户")
        if has_media:
            score += _WEIGHT_IMAGE_MEDIA
            reasons.append("含媒体")
        # 扣分项
        if watch_level_rank <= 0:
            score -= _PENALTY_LOW_WATCH
            reasons.append("完全无视状态")
        elif watch_level_rank == 1:
            score -= _PENALTY_LOW_WATCH * 0.5
            reasons.append("仅瞟一眼")
        if presence_state is not None:
            quiet_pref = getattr(presence_state, "quiet_preference", 0.0)
            avoidance = getattr(presence_state, "avoidance_tendency", 0.0)
            if quiet_pref > 0.5:
                score -= _PENALTY_QUIET_PREFERENCE * quiet_pref
                reasons.append("想安静")
            if avoidance > 0.4:
                score -= _PENALTY_HIGH_ANNOYANCE * avoidance
                reasons.append("回避倾向")
            outward = getattr(presence_state, "outward_attention", 0.5)
            if outward < 0.3:
                score -= _PENALTY_LOW_ENERGY * (1.0 - outward)
                reasons.append("注意力低")
        # 摸鱼值扣分：高摸鱼时普通消息降权（@和引用不受影响）
        if loafing_level > 60 and not is_at_mention and not is_quote_reply:
            _loaf_factor = min(1.0, (loafing_level - 60) / 40.0)
            score -= _PENALTY_HIGH_LOAFING * _loaf_factor
            reasons.append(f"高摸鱼降权({loafing_level:.0f})")
        elif loafing_level > 35 and not is_at_mention and not is_quote_reply:
            _loaf_factor = min(1.0, (loafing_level - 35) / 25.0)
            score -= _PENALTY_MODERATE_LOAFING * _loaf_factor
            reasons.append(f"摸鱼中降权({loafing_level:.0f})")
        # 观看层级加成：高层级自然更容易看到
        watch_bonus = watch_level_rank * 0.08
        score += watch_bonus
        score = max(0.0, min(1.0, score))
        # 判定层级
        level = self._score_to_level(score, is_at_mention, is_quote_reply)
        decision = VisibilityDecision(
            message_id=msg_id,
            user_id=user_id,
            level=level,
            score=score,
            enter_understanding=level.enters_understanding(),
            reasons=reasons,
        )
        self._recent_decisions.append(decision)
        if len(self._recent_decisions) > self._max_cache:
            self._recent_decisions = self._recent_decisions[-self._max_cache:]
        return decision

    def evaluate_batch(
        self,
        messages: List[Any],
        *,
        watch_level_rank: int = 2,
        presence_state: Optional[Any] = None,
        user_relationships: Optional[Dict[str, int]] = None,
        at_mention_ids: Optional[set] = None,
        quote_reply_ids: Optional[set] = None,
        name_mention_ids: Optional[set] = None,
        loafing_level: float = 0.0,
    ) -> List[VisibilityDecision]:
        """批量评估消息可见性

        Args:
            loafing_level: 摸鱼值 (0-100)，高摸鱼时普通消息应被降权
        """
        results: List[VisibilityDecision] = []
        at_set = at_mention_ids or set()
        quote_set = quote_reply_ids or set()
        name_set = name_mention_ids or set()
        rel_map = user_relationships or {}
        for msg in messages:
            mid = str(getattr(msg, "message_id", "") or "")
            uid = str(getattr(msg, "user_id", "") or "")
            decision = self.evaluate(
                msg,
                watch_level_rank=watch_level_rank,
                presence_state=presence_state,
                user_relationship_level=rel_map.get(uid, 2),
                is_at_mention=mid in at_set
                or bool(getattr(msg, "is_at", False)),
                is_quote_reply=mid in quote_set,
                is_name_mention=mid in name_set
                or bool(getattr(msg, "is_mentioned", False)),
                has_media=bool(
                    getattr(msg, "has_image", False)
                    or getattr(msg, "image_list", None)
                ),
                loafing_level=loafing_level,
            )
            results.append(decision)
        return results

    def filter_for_understanding(
        self, decisions: List[VisibilityDecision]
    ) -> List[str]:
        """从判定结果中筛选出进入理解链的消息ID"""
        return [d.message_id for d in decisions if d.enter_understanding]

    def recent_visibility_summary(self) -> Dict[str, int]:
        """最近各层级的消息数统计"""
        counts: Dict[str, int] = {}
        for d in self._recent_decisions[-20:]:
            key = d.level.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ────────────────── 内部方法 ──────────────────

    def _score_to_level(
        self, score: float, is_at: bool, is_quote: bool
    ) -> VisibilityLevel:
        """分数映射到可见性层级"""
        # 被@或引用回复直接进入 FOCUSED
        if is_at or is_quote:
            return VisibilityLevel.FOCUSED
        if score >= _THRESHOLD_FOCUSED:
            return VisibilityLevel.FOCUSED
        if score >= _THRESHOLD_NOTICED:
            return VisibilityLevel.NOTICED
        if score >= _THRESHOLD_GLANCED:
            return VisibilityLevel.GLANCED
        return VisibilityLevel.RECEIVED


# ---------------------------------------------------------------------------
#  全局访问接口
# ---------------------------------------------------------------------------


def get_visibility_gate(channel_id: str) -> VisibilityGate:
    if channel_id not in _gate_instances:
        _gate_instances[channel_id] = VisibilityGate(channel_id)
    return _gate_instances[channel_id]


def remove_visibility_gate(channel_id: str) -> None:
    _gate_instances.pop(channel_id, None)

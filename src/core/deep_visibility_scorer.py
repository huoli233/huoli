import time
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from src.common.logger import get_logger

logger = get_logger("深度可见性")

_scorer_instances: Dict[str, "DeepVisibilityScorer"] = {}

# ═══════════════════════════════════════════════════
#  偏置维度枚举
# ═══════════════════════════════════════════════════


class BiasDimension(Enum):
    """偏置维度"""

    USER = "user"
    EVENT = "event"
    TOPIC = "topic"
    GROUP_ENV = "group_env"
    TIME_DECAY = "time_decay"


class VisibilityDepthLevel(Enum):
    """深层可见性层级（扩展自基础VisibilityLevel）"""

    SYSTEM_RECEIVED = "system_received"
    GLANCED_OVER = "glanced_over"
    NOTICED_SURFACE = "noticed_surface"
    ATTENDED = "attended"
    FOCUSED_DEEP = "focused_deep"
    UNDERSTOOD_DEFERRED = "understood_deferred"
    UNDERSTOOD_IGNORED = "understood_ignored"

    def rank(self) -> int:
        return {
            "system_received": 0,
            "glanced_over": 1,
            "noticed_surface": 2,
            "attended": 3,
            "focused_deep": 4,
            "understood_deferred": 4,
            "understood_ignored": 4,
        }.get(self.value, 0)

    def enters_understanding(self) -> bool:
        return self in (
            VisibilityDepthLevel.FOCUSED_DEEP,
            VisibilityDepthLevel.UNDERSTOOD_DEFERRED,
            VisibilityDepthLevel.UNDERSTOOD_IGNORED,
        )

    def label(self) -> str:
        return {
            "system_received": "系统收到",
            "glanced_over": "扫到",
            "noticed_surface": "注意到",
            "attended": "纳入注意",
            "focused_deep": "深度关注",
            "understood_deferred": "理解但暂存",
            "understood_ignored": "理解但不理",
        }.get(self.value, "")


@dataclass
class UserBiasProfile:
    """用户维度的偏置画像"""

    user_id: str = ""
    familiarity: float = 0.0
    affection: float = 0.5
    trust: float = 0.5
    annoyance: float = 0.0
    recent_interaction_count: int = 0
    last_interaction_sec: float = 0.0
    is_high_risk: bool = False
    is_avoided: bool = False
    impression_tags: List[str] = field(default_factory=list)

    def compute_user_bias_score(self) -> float:
        score = 0.0
        score += self.familiarity * 0.20
        score += max(0.0, (self.affection - 0.5)) * 0.25
        score += max(0.0, (self.trust - 0.3)) * 0.10
        if self.recent_interaction_count >= 5:
            score += 0.08
        elif self.recent_interaction_count >= 3:
            score += 0.04
        time_fresh = max(0.0, 1.0 - self.last_interaction_sec / 3600.0)
        score += time_fresh * 0.07
        if self.is_high_risk:
            score += 0.12
        if self.is_avoided:
            score -= 0.25
        _conflict = {"容易起冲突", "冲突型", "攻击性强"}
        if any(t in self.impression_tags for t in _conflict):
            score += 0.06
        _friendly = {"积极友善", "热情", "有趣"}
        if any(t in self.impression_tags for t in _friendly):
            score += 0.05
        return max(-0.3, min(0.5, score))


@dataclass
class EventBiasInput:
    """事件维度的输入"""

    is_at_mention: bool = False
    is_quote_reply: bool = False
    is_name_mention: bool = False
    emotion_intensity: float = 0.0
    group_activity_spike: bool = False
    is_self_reference: bool = False
    self_ref_strength: float = 0.0
    has_media: bool = False
    media_discussed_by_others: bool = False
    message_length: int = 0

    def compute_event_bias_score(self) -> float:
        score = 0.0
        if self.is_at_mention:
            score += 0.40
        if self.is_quote_reply:
            score += 0.30
        if self.is_name_mention:
            score += 0.22
        if self.emotion_intensity > 0.7:
            score += 0.15
        elif self.emotion_intensity > 0.4:
            score += 0.07
        if self.group_activity_spike:
            score += 0.10
        if self.is_self_reference and self.self_ref_strength > 0.5:
            score += 0.18 * self.self_ref_strength
        if self.has_media and self.media_discussed_by_others:
            score += 0.10
        if self.message_length > 200:
            score += 0.05
        elif self.message_length < 5:
            score -= 0.05
        return max(0.0, min(0.8, score))


@dataclass
class TopicBiasInput:
    """话题维度的输入"""

    current_lingering_topics: List[str] = field(default_factory=list)
    topic_relevance: float = 0.0
    pending_user_ids: Set[str] = field(default_factory=set)
    subject_interest_keywords: List[str] = field(default_factory=list)
    message_text: str = ""

    def compute_topic_bias_score(self) -> float:
        score = 0.0
        score += self.topic_relevance * 0.18
        if self.current_lingering_topics:
            text_lower = self.message_text.lower()
            matched = sum(
                1
                for t in self.current_lingering_topics
                if t.lower() in text_lower
            )
            if matched > 0:
                score += min(0.20, matched * 0.08)
        if self.pending_user_ids:
            score += 0.05
        if self.subject_interest_keywords:
            text_lower = self.message_text.lower()
            interest_hit = sum(
                1 for kw in self.subject_interest_keywords if kw in text_lower
            )
            if interest_hit > 0:
                score += min(0.12, interest_hit * 0.04)
        return max(0.0, min(0.35, score))


@dataclass
class GroupEnvModulation:
    """群环境调制因子"""

    group_atmosphere: str = ""  # cold/calm/active/explosive
    active_member_count: int = 0
    is_multi_thread: bool = False
    bot_being_discussed: bool = False
    suitable_to_interrupt: bool = True
    watch_state_rank: int = 2
    boredom_level: float = 0.0
    loafing_level: float = 0.0
    energy_ratio: float = 1.0
    social_willingness: float = 0.5

    def compute_modulation_multiplier(self) -> float:
        mult = 1.0
        _atmo_map = {
            "cold": 0.75,
            "calm": 0.90,
            "active": 1.15,
            "explosive": 1.05,
        }
        mult *= _atmo_map.get(self.group_atmosphere, 1.0)
        if self.active_member_count > 15:
            mult *= 0.88
        elif self.active_member_count > 8:
            mult *= 0.95
        if self.is_multi_thread:
            mult *= 0.92
        if self.bot_being_discussed:
            mult *= 1.20
        if not self.suitable_to_interrupt:
            mult *= 0.70
        if self.watch_state_rank <= 1:
            mult *= 0.60
        elif self.watch_state_rank == 2:
            mult *= 0.85
        elif self.watch_state_rank >= 4:
            mult *= 1.15
        if self.loafing_level > 0.6:
            mult *= 0.70
        elif self.boredom_level > 0.6:
            mult *= 1.10
        if self.energy_ratio < 0.2:
            mult *= 0.55
        elif self.energy_ratio < 0.4:
            mult *= 0.80
        mult *= 0.85 + self.social_willingness * 0.30
        return max(0.30, min(1.50, mult))


@dataclass
class TimeDecayFactor:
    """时间衰减因子"""

    message_age_sec: float = 0.0
    since_last_glance_sec: float = 0.0
    visibility_history_level: str = ""
    original_freshness: float = 1.0

    def compute_decay_multiplier(self) -> float:
        if self.message_age_sec <= 5:
            age_mult = 1.0
        elif self.message_age_sec <= 30:
            age_mult = 1.0 - (self.message_age_sec - 5) / 200.0
        elif self.message_age_sec <= 120:
            age_mult = 0.875 - (self.message_age_sec - 30) / 400.0
        elif self.message_age_sec <= 600:
            age_mult = 0.66 - (self.message_age_sec - 120) / 1500.0
        else:
            age_mult = max(0.20, 0.46 - (self.message_age_sec - 600) / 3000.0)
        glance_factor = max(0.40, 1.0 - self.since_last_glance_sec / 1800.0)
        history_bonus = 1.0
        if self.visibility_history_level in (
            "received",
            "glanced_over",
            "system_received",
        ):
            history_bonus = 1.10
        elif self.visibility_history_level in (
            "understood_ignored",
            "focused_deep",
        ):
            history_bonus = 0.92
        base = (
            self.original_freshness * age_mult * glance_factor * history_bonus
        )
        return max(0.10, min(1.0, base))


@dataclass
class DeepVisibilityResult:
    """深度可见性评分结果"""

    message_id: str = ""
    user_id: str = ""
    level: VisibilityDepthLevel = VisibilityDepthLevel.SYSTEM_RECEIVED
    raw_score: float = 0.0
    modulated_score: float = 0.0
    user_bias: float = 0.0
    event_bias: float = 0.0
    topic_bias: float = 0.0
    group_modulation: float = 1.0
    time_decay: float = 1.0
    enter_understanding: bool = False
    reasons: List[str] = field(default_factory=list)
    bias_breakdown: Dict[str, float] = field(default_factory=dict)
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "level": self.level.value,
            "level_label": self.level.label(),
            "raw_score": round(self.raw_score, 3),
            "modulated_score": round(self.modulated_score, 3),
            "enter_understanding": self.enter_understanding,
            "bias_breakdown": {
                k: round(v, 3) for k, v in self.bias_breakdown.items()
            },
            "reasons": self.reasons[:6],
        }


# 层级阈值常量
_THRESH_ATTENDED = 0.28
_THRESH_FOCUSED_DEEP = 0.48
_THRESH_UNDERSTOOD_DEFERRED = 0.38


class DeepVisibilityScorer:
    """消息可见性深度评分引擎
    在基础 VisibilityGate 的加权评分之上，引入：
    1. 用户维度偏置：熟悉度/好感/信任/风险/回避/印象标签
    2. 事件维度偏置：@/引用/名字/情绪强度/群活跃峰值/自我指称/媒体讨论
    3. 话题维度偏置：余味话题匹配/挂念用户出现/兴趣关键词命中
    4. 群环境调制：氛围/人数/多线程/是否被讨论/插话适合度/观看状态/摸鱼值/精力
    5. 时间衰减：消息年龄/距上次窥屏/历史可见性层级
    输出 7 级深层可见性，支持"没看见≠看见但不理"的差异化处理。
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._recent_results: List[DeepVisibilityResult] = []
        self._max_cache = 60
        self._user_profile_cache: Dict[str, UserBiasProfile] = {}
        self._profile_cache_ttl = 300.0

    def evaluate(
        self,
        message: Any,
        *,
        user_profile: Optional[UserBiasProfile] = None,
        event_input: Optional[EventBiasInput] = None,
        topic_input: Optional[TopicBiasInput] = None,
        group_mod: Optional[GroupEnvModulation] = None,
        time_decay: Optional[TimeDecayFactor] = None,
        force_level_override: Optional[str] = None,
    ) -> DeepVisibilityResult:
        """执行深度可见性评分"""
        msg_id = str(getattr(message, "message_id", "") or "")
        user_id = str(getattr(message, "user_id", "") or "")
        reasons: List[str] = []
        breakdown: Dict[str, float] = {}
        up = user_profile or UserBiasProfile(user_id=user_id)
        ei = event_input or EventBiasInput()
        ti = topic_input or TopicBiasInput(
            message_text=str(
                getattr(message, "processed_plain_text", "")
                or getattr(message, "plain_text", "")
                or ""
            ),
        )
        gm = group_mod or GroupEnvModulation()
        td = time_decay or TimeDecayFactor()
        user_score = up.compute_user_bias_score()
        event_score = ei.compute_event_bias_score()
        topic_score = ti.compute_topic_bias_score()
        group_mult = gm.compute_modulation_multiplier()
        decay_mult = td.compute_decay_multiplier()
        breakdown["user_bias"] = user_score
        breakdown["event_bias"] = event_score
        breakdown["topic_bias"] = topic_score
        breakdown["group_mod"] = group_mult
        breakdown["time_decay"] = decay_mult
        raw = 0.15 + user_score + event_score + topic_score
        raw = max(0.0, min(1.0, raw))
        modulated = raw * group_mult * decay_mult
        modulated = max(0.0, min(1.0, modulated))
        if force_level_override:
            level = self._force_level(force_level_override)
            reasons.append(f"强制层级: {force_level_override}")
        else:
            level = self._score_to_depth_level(modulated, ei, up)
            self._collect_reasons(
                reasons,
                level,
                user_score,
                event_score,
                topic_score,
                group_mult,
                decay_mult,
            )
        result = DeepVisibilityResult(
            message_id=msg_id,
            user_id=user_id,
            level=level,
            raw_score=raw,
            modulated_score=modulated,
            user_bias=user_score,
            event_bias=event_score,
            topic_bias=topic_score,
            group_modulation=group_mult,
            time_decay=decay_mult,
            enter_understanding=level.enters_understanding(),
            reasons=reasons,
            bias_breakdown=breakdown,
        )
        self._recent_results.append(result)
        if len(self._recent_results) > self._max_cache:
            self._recent_results = self._recent_results[-self._max_cache:]
        return result

    def evaluate_batch(
        self,
        messages: List[Any],
        *,
        user_profiles: Optional[Dict[str, UserBiasProfile]] = None,
        default_event: Optional[EventBiasInput] = None,
        default_topic: Optional[TopicBiasInput] = None,
        default_group_mod: Optional[GroupEnvModulation] = None,
        default_time_decay: Optional[TimeDecayFactor] = None,
    ) -> List[DeepVisibilityResult]:
        """批量深度评分"""
        results: List[DeepVisibilityResult] = []
        profiles = user_profiles or {}
        for msg in messages:
            uid = str(getattr(msg, "user_id", "") or "")
            result = self.evaluate(
                msg,
                user_profile=profiles.get(uid),
                event_input=default_event,
                topic_input=default_topic,
                group_mod=default_group_mod,
                time_decay=default_time_decay,
            )
            results.append(result)
        return results

    def filter_for_understanding(
        self, results: List[DeepVisibilityResult]
    ) -> List[str]:
        return [r.message_id for r in results if r.enter_understanding]

    def filter_deferred(
        self, results: List[DeepVisibilityResult]
    ) -> List[str]:
        return [
            r.message_id
            for r in results
            if r.level == VisibilityDepthLevel.UNDERSTOOD_DEFERRED
        ]

    def filter_ignored_but_seen(
        self, results: List[DeepVisibilityResult]
    ) -> List[str]:
        return [
            r.message_id
            for r in results
            if r.level == VisibilityDepthLevel.UNDERSTOOD_IGNORED
        ]

    def never_seen_messages(
        self, results: List[DeepVisibilityResult]
    ) -> List[DeepVisibilityResult]:
        return [r for r in results if r.level.rank() <= 1]

    def high_priority_messages(
        self, results: List[DeepVisibilityResult], threshold: float = 0.55
    ) -> List[DeepVisibilityResult]:
        return [r for r in results if r.modulated_score >= threshold]

    def summary_stats(
        self, results: Optional[List[DeepVisibilityResult]] = None
    ) -> Dict[str, Any]:
        pool = results or self._recent_results[-30:]
        level_counts: Dict[str, int] = {}
        avg_score = 0.0
        understanding_rate = 0.0
        for r in pool:
            lv = r.level.value
            level_counts[lv] = level_counts.get(lv, 0) + 1
            avg_score += r.modulated_score
            if r.enter_understanding:
                understanding_rate += 1
        n = len(pool) or 1
        return {
            "total_evaluated": len(pool),
            "level_distribution": level_counts,
            "avg_modulated_score": round(avg_score / n, 3),
            "understanding_rate": round(understanding_rate / n, 3),
        }

    # ────────────────── 内部方法 ──────────────────

    @staticmethod
    def _score_to_depth_level(
        score: float,
        event: EventBiasInput,
        user: UserBiasProfile,
    ) -> VisibilityDepthLevel:
        if event.is_at_mention or event.is_quote_reply:
            if user.is_avoided and score < 0.70:
                return VisibilityDepthLevel.NOTICED_SURFACE
            return VisibilityDepthLevel.FOCUSED_DEEP
        if event.is_name_mention and score > 0.35:
            return VisibilityDepthLevel.ATTENDED
        if score >= _THRESH_FOCUSED_DEEP:
            return VisibilityDepthLevel.FOCUSED_DEEP
        if score >= _THRESH_UNDERSTOOD_DEFERRED:
            if user.annoyance > 0.5 or user.is_avoided:
                return VisibilityDepthLevel.UNDERSTOOD_IGNORED
            return VisibilityDepthLevel.UNDERSTOOD_DEFERRED
        if score >= _THRESH_ATTENDED:
            return VisibilityDepthLevel.ATTENDED
        if score >= 0.12:
            return VisibilityDepthLevel.NOTICED_SURFACE
        if score >= 0.04:
            return VisibilityDepthLevel.GLANCED_OVER
        return VisibilityDepthLevel.SYSTEM_RECEIVED

    @staticmethod
    def _force_level(override: str) -> VisibilityDepthLevel:
        mapping = {
            "system_received": VisibilityDepthLevel.SYSTEM_RECEIVED,
            "glanced_over": VisibilityDepthLevel.GLANCED_OVER,
            "noticed_surface": VisibilityDepthLevel.NOTICED_SURFACE,
            "attended": VisibilityDepthLevel.ATTENDED,
            "focused_deep": VisibilityDepthLevel.FOCUSED_DEEP,
            "understood_deferred": VisibilityDepthLevel.UNDERSTOOD_DEFERRED,
            "understood_ignored": VisibilityDepthLevel.UNDERSTOOD_IGNORED,
        }
        return mapping.get(override, VisibilityDepthLevel.SYSTEM_RECEIVED)

    @staticmethod
    def _collect_reasons(
        reasons: List[str],
        level: VisibilityDepthLevel,
        user_score: float,
        event_score: float,
        topic_score: float,
        group_mult: float,
        decay_mult: float,
    ) -> None:
        reasons.append(f"层级={level.label()}")
        if user_score > 0.15:
            reasons.append("用户偏置高")
        if user_score < -0.10:
            reasons.append("用户回避")
        if event_score > 0.30:
            reasons.append("强事件信号")
        if topic_score > 0.15:
            reasons.append("话题相关")
        if group_mult > 1.15:
            reasons.append("群环境放大")
        if group_mult < 0.75:
            reasons.append("群环境压制")
        if decay_mult < 0.50:
            reasons.append("时间衰减大")


def get_deep_visibility_scorer(channel_id: str) -> DeepVisibilityScorer:
    if channel_id not in _scorer_instances:
        _scorer_instances[channel_id] = DeepVisibilityScorer(channel_id)
    return _scorer_instances[channel_id]


def remove_deep_visibility_scorer(channel_id: str) -> None:
    _scorer_instances.pop(channel_id, None)

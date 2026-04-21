import time
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from src.common.logger import get_logger
from src.chat.heart_flow.dimension_protocol import (
    DimensionBase,
    DimensionScope,
    EventContext,
    TickResult,
)
from src.chat.heart_flow.vote_types import SocialValueVote

logger = get_logger("social_value_dim")


# ============================================================
# 社交值档位与标签映射
# ============================================================
_SOCIAL_TIERS: list[Tuple[float, float, str, str, float]] = [
    # (lo, hi, tag, 描述, prob_modifier)
    (-100.0, -60.0, "social_pariah", "群内极度厌弃、被排斥", 0.1),
    (-60.0, -30.0, "social_outcast", "明显不受欢迎", 0.4),
    (-30.0, -5.0, "social_cold", "存在感低、被忽视", 0.7),
    (-5.0, 5.0, "social_neutral", "普通存在", 1.0),
    (5.0, 30.0, "social_accepted", "被接纳、有存在感", 1.05),
    (30.0, 60.0, "social_liked", "受欢迎、被认可", 1.15),
    (60.0, 101.0, "social_star", "人气极高、群宠", 1.25),
]


def _lookup_social_tier(score: float) -> Tuple[str, str, float]:
    """查找社交值对应的标签、描述和概率调整"""
    for lo, hi, tag, desc, prob in _SOCIAL_TIERS:
        if lo <= score < hi:
            return tag, desc, prob
    return "social_neutral", "普通存在", 1.0


# ============================================================
# 单用户社交值状态
# ============================================================
@dataclass
class UserSocialProfile:
    """单个用户在特定频道的社交值档案"""
    score: float = 0.0
    positive_count: int = 0
    negative_count: int = 0
    # 最近互动时间戳
    last_activity_ts: float = 0.0
    # 连续正面互动计数（用于加成叠加）
    streak_positive: int = 0
    # 连续负面互动计数
    streak_negative: int = 0
    # 累计被其他用户提及的次数
    mention_received: int = 0
    # 累计主动提及其他用户的次数
    mention_given: int = 0


# ============================================================
# 社交值变动原因 → 基底值映射
# ============================================================
_REASON_DELTA_MAP = {
    "helpful_response": 2.0,
    "funny_message": 1.5,
    "insightful_input": 1.5,
    "active_participation": 0.8,
    "emoji_reaction": 0.3,
    "mentioned_by_others": 1.0,
    "ignored_by_all": -0.5,
    "spam_message": -2.0,
    "offensive_content": -4.0,
    "repetitive_content": -1.5,
    "off_topic": -0.8,
    "conversation_killer": -1.0,
    "neutral": 0.0,
}


# ============================================================
# D7 维度实现：社交值
# ============================================================
class SocialValueDimension(DimensionBase):
    """
    D7 社交值维度。
    衡量用户（或bot自身）在群聊中的社交地位和受欢迎程度。
    社交值是独立于好感度的群体层面指标：
      - 好感度(D2)是 bot对用户的单向评价
      - 社交值(D7)是 用户在群体中的综合地位
    per-user-per-channel 粒度。
    """

    _singleton: Optional["SocialValueDimension"] = None

    @classmethod
    def get_instance(cls) -> "SocialValueDimension":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        # "{user_id}:{channel_id}" → UserSocialProfile
        self._profiles: Dict[str, UserSocialProfile] = {}
        # 自然衰减：社交值向0回归的速率（每小时）
        self._natural_decay_per_hour: float = 0.2
        # 连续正面互动加成上限
        self._streak_bonus_cap: int = 5
        # 连续负面互动惩罚上限
        self._streak_penalty_cap: int = 3

    @property
    def dimension_name(self) -> str:
        return "social_value"

    @property
    def scope(self) -> DimensionScope:
        return DimensionScope.PER_USER_CHANNEL

    @property
    def needs_persistence(self) -> bool:
        return True

    @property
    def tick_interval_sec(self) -> float:
        return 45.0

    def _composite_key(self, user_id: str, channel_id: str) -> str:
        return f"{user_id}:{channel_id}"

    def _get_profile(self, user_id: str, channel_id: str) -> UserSocialProfile:
        key = self._composite_key(user_id, channel_id)
        if key not in self._profiles:
            self._profiles[key] = UserSocialProfile(last_activity_ts=time.time())
        return self._profiles[key]

    def tick(self, elapsed_sec: float) -> TickResult:
        """定期自然衰减：社交值随时间向0回归"""
        hours = elapsed_sec / 3600.0
        if hours < 0.01:
            return TickResult(dimension_name=self.dimension_name, updated=False)
        updated = False
        decay_amount = self._natural_decay_per_hour * hours
        for profile in self._profiles.values():
            if abs(profile.score) < 0.1:
                continue
            if profile.score > 0:
                profile.score = max(0.0, profile.score - decay_amount)
                updated = True
            elif profile.score < 0:
                # 负面值恢复更慢
                recovery = decay_amount * 0.5
                profile.score = min(0.0, profile.score + recovery)
                updated = True
        return TickResult(
            dimension_name=self.dimension_name,
            updated=updated,
            summary=f"档案数={len(self._profiles)}" if updated else "",
        )

    def on_event(self, ctx: EventContext):
        """
        根据消息事件更新社交值。
        支持多种触发来源：
          - 通过 raw_extras["social_reason"] 传入具体原因
          - 通过 sentiment_score 自动推导
          - 通过 raw_extras["mentioned_users"] 增加被提及者的社交值
        """
        if not ctx.user_id or not ctx.channel_id:
            return
        profile = self._get_profile(ctx.user_id, ctx.channel_id)
        if ctx.event_type == "message_received":
            profile.last_activity_ts = time.time()
            # 显式社交原因（最优先）
            reason = ctx.raw_extras.get("social_reason", "")
            if reason and reason in _REASON_DELTA_MAP:
                delta = _REASON_DELTA_MAP[reason]
                self._apply_delta(profile, delta)
            else:
                # 没有显式原因时，按 sentiment 自动推导
                sent = ctx.sentiment_score
                if sent > 0.3:
                    self._apply_delta(profile, 0.5)
                elif sent < -0.3:
                    self._apply_delta(profile, -0.8)
                else:
                    # 中性消息也有微量正面贡献（活跃就是贡献）
                    self._apply_delta(profile, 0.1)
            # 被提及的用户获得社交值加成
            mentioned = ctx.raw_extras.get("mentioned_users", [])
            if isinstance(mentioned, list):
                for target_uid in mentioned:
                    if str(target_uid) == ctx.user_id:
                        continue
                    target_profile = self._get_profile(str(target_uid), ctx.channel_id)
                    target_profile.mention_received += 1
                    self._apply_delta(target_profile, 0.5)
                    profile.mention_given += 1
        elif ctx.event_type == "reply_completed":
            profile.last_activity_ts = time.time()
        # 外部直接调整
        explicit_delta = ctx.raw_extras.get("social_delta")
        if explicit_delta is not None:
            self._apply_delta(profile, float(explicit_delta))

    def _apply_delta(self, profile: UserSocialProfile, delta: float):
        """带连续叠加效应的社交值变动"""
        if delta > 0:
            profile.positive_count += 1
            profile.streak_positive += 1
            profile.streak_negative = 0
            # 连续正面加成：streak越高，加成越大（上限+100%）
            streak_multiplier = 1.0 + min(
                profile.streak_positive, self._streak_bonus_cap
            ) * 0.1
            effective_delta = delta * streak_multiplier
        elif delta < 0:
            profile.negative_count += 1
            profile.streak_negative += 1
            profile.streak_positive = 0
            # 连续负面惩罚：streak越高，惩罚越大（上限+60%）
            streak_multiplier = 1.0 + min(
                profile.streak_negative, self._streak_penalty_cap
            ) * 0.2
            effective_delta = delta * streak_multiplier
        else:
            return
        # 极化阻力：越接近极端值，变化越困难
        polarization = abs(profile.score) / 100.0
        resistance = 0.7 + 0.3 * (1.0 - polarization)
        effective_delta *= resistance
        profile.score = max(-100.0, min(100.0, profile.score + effective_delta))

    def vote(self, ctx: EventContext) -> SocialValueVote:
        """
        根据用户社交值输出投票。
        社交值高的用户，bot倾向更积极地互动。
        社交值低的用户，bot降低互动意愿。
        """
        if not ctx.user_id or not ctx.channel_id:
            return SocialValueVote()
        profile = self._get_profile(ctx.user_id, ctx.channel_id)
        score = profile.score
        tag, desc, prob_mod = _lookup_social_tier(score)
        # 是否警戒
        in_alert = score < -30.0
        # 态度标签
        attitude = ""
        if score < -60:
            attitude = "social_avoid"
        elif score < -30:
            attitude = "social_cautious"
        elif score > 60:
            attitude = "social_enthusiastic"
        elif score > 30:
            attitude = "social_warm"
        return SocialValueVote(
            probability_factor=round(prob_mod, 3),
            attitude_tag=attitude,
            social_score=round(score, 1),
            social_tag=tag,
            in_alert=in_alert,
            positive_interactions=profile.positive_count,
            negative_interactions=profile.negative_count,
            proactive_boost=round(max(0.0, score / 100.0), 3),
            debug_reason=f"social={score:.1f}({tag}) +{profile.positive_count}/-{profile.negative_count}",
        )

    def serialize(self) -> dict:
        result = {}
        for key, profile in self._profiles.items():
            result[key] = {
                "score": profile.score,
                "pos": profile.positive_count,
                "neg": profile.negative_count,
                "last_ts": profile.last_activity_ts,
                "streak_p": profile.streak_positive,
                "streak_n": profile.streak_negative,
                "mention_r": profile.mention_received,
                "mention_g": profile.mention_given,
            }
        return result

    def deserialize(self, data: dict):
        if not isinstance(data, dict):
            return
        for key, vals in data.items():
            if not isinstance(vals, dict):
                continue
            self._profiles[key] = UserSocialProfile(
                score=float(vals.get("score", 0.0)),
                positive_count=int(vals.get("pos", 0)),
                negative_count=int(vals.get("neg", 0)),
                last_activity_ts=float(vals.get("last_ts", time.time())),
                streak_positive=int(vals.get("streak_p", 0)),
                streak_negative=int(vals.get("streak_n", 0)),
                mention_received=int(vals.get("mention_r", 0)),
                mention_given=int(vals.get("mention_g", 0)),
            )

    def calibrate(self, offline_seconds: float):
        """离线校准：社交值按离线时长向0衰减"""
        hours = offline_seconds / 3600.0
        decay = self._natural_decay_per_hour * hours
        for profile in self._profiles.values():
            if profile.score > 0:
                profile.score = max(0.0, profile.score - decay)
            elif profile.score < 0:
                profile.score = min(0.0, profile.score + decay * 0.5)
            # 连续计数离线后重置
            profile.streak_positive = 0
            profile.streak_negative = 0

    def get_state_summary(self) -> dict:
        summaries = {}
        for key, profile in list(self._profiles.items())[:20]:
            tag, _, _ = _lookup_social_tier(profile.score)
            summaries[key[:16]] = {
                "score": round(profile.score, 1),
                "tag": tag,
                "positive": profile.positive_count,
                "negative": profile.negative_count,
            }
        return {
            "dimension": self.dimension_name,
            "scope": self.scope.value,
            "profile_count": len(self._profiles),
            "profiles": summaries,
        }

    def reset(self, user_id: str = "", channel_id: str = ""):
        if user_id and channel_id:
            key = self._composite_key(user_id, channel_id)
            self._profiles.pop(key, None)
        elif user_id:
            to_remove = [k for k in self._profiles if k.startswith(f"{user_id}:")]
            for k in to_remove:
                del self._profiles[k]

    # ---- 外部调用接口 ----

    def adjust_social_score(
        self, user_id: str, channel_id: str, delta: float, reason: str = ""
    ):
        """直接调整社交值（供外部模块调用）"""
        profile = self._get_profile(user_id, channel_id)
        old = profile.score
        self._apply_delta(profile, delta)
        logger.debug(
            f"社交值调整: {user_id[:8]}@{channel_id[:8]} "
            f"{old:.1f}→{profile.score:.1f} ({reason})"
        )

    def get_social_score(self, user_id: str, channel_id: str) -> float:
        """获取指定用户的当前社交值"""
        profile = self._profiles.get(self._composite_key(user_id, channel_id))
        return profile.score if profile else 0.0

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from src.common.logger import get_logger
from src.modules.modcore.dynamic_persona.emotion_tracker import (
    get_emotion_tracker,
)

logger = get_logger("印象枢纽")

_impression_instances: Dict[str, "ImpressionEvolutionHub"] = {}


@dataclass
class TruthLayer:
    """量化真值层：系统记录的客观交互数据，第三阶段升级：增加更多量化指标"""

    total_interactions: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    last_interaction_at: float = 0.0
    average_reply_quality: float = 0.5
    topic_overlap_ratio: float = 0.0
    conflict_count: int = 0
    cooperation_count: int = 0
    emotional_exchange_count: int = 0
    shared_experience_count: int = 0
    mutual_understanding_score: float = 0.0
    communication_efficiency: float = 0.5
    response_timeliness: float = 0.5

    def positivity_ratio(self) -> float:
        if self.total_interactions == 0:
            return 0.5
        return self.positive_interactions / max(1, self.total_interactions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_interactions": self.total_interactions,
            "positive_interactions": self.positive_interactions,
            "negative_interactions": self.negative_interactions,
            "positivity_ratio": round(self.positivity_ratio(), 3),
            "average_reply_quality": round(self.average_reply_quality, 3),
            "conflict_count": self.conflict_count,
            "cooperation_count": self.cooperation_count,
        }


@dataclass
class SubjectiveTagLayer:
    """主观标签层：bot 对用户的感性印象，第三阶段升级：增加更多感性维度"""

    personality_tags: List[str] = field(default_factory=list)
    speaking_style: str = ""
    emotional_tendency: str = ""
    interest_topics: List[str] = field(default_factory=list)
    memorable_moments: List[str] = field(default_factory=list)
    custom_nickname: str = ""
    overall_feeling: str = "一般"
    last_updated: float = field(default_factory=time.time)
    humor_sense: float = 0.5
    empathy_level: float = 0.5
    creativity_index: float = 0.5
    social_harmony: float = 0.5
    intellectual_depth: float = 0.5
    emotional_stability: float = 0.5

    def add_tag(self, tag: str) -> None:
        if tag and tag not in self.personality_tags:
            self.personality_tags.append(tag)
            if len(self.personality_tags) > 15:
                self.personality_tags = self.personality_tags[-15:]

    def add_moment(self, moment: str) -> None:
        if moment:
            self.memorable_moments.append(moment[:100])
            if len(self.memorable_moments) > 10:
                self.memorable_moments = self.memorable_moments[-10:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "personality_tags": self.personality_tags[:10],
            "speaking_style": self.speaking_style,
            "emotional_tendency": self.emotional_tendency,
            "interest_topics": self.interest_topics[:8],
            "memorable_moments": self.memorable_moments[-5:],
            "custom_nickname": self.custom_nickname,
            "overall_feeling": self.overall_feeling,
        }


@dataclass
class RelationshipNarrative:
    """关系叙事层：bot 与用户的关系故事，第三阶段升级：增加叙事深度"""

    first_met_summary: str = ""
    relationship_stage: str = "陌生"
    turning_points: List[str] = field(default_factory=list)
    current_dynamic: str = ""
    ongoing_topics: List[str] = field(default_factory=list)
    unresolved_tensions: List[str] = field(default_factory=list)
    shared_memories: List[str] = field(default_factory=list)
    future_expectations: List[str] = field(default_factory=list)
    communication_patterns: Dict[str, int] = field(default_factory=dict)
    emotional_trajectory: List[Tuple[float, str]] = field(default_factory=list)

    def add_turning_point(self, event: str) -> None:
        if event:
            self.turning_points.append(event[:120])
            if len(self.turning_points) > 8:
                self.turning_points = self.turning_points[-8:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "first_met_summary": self.first_met_summary[:100],
            "relationship_stage": self.relationship_stage,
            "turning_points": self.turning_points[-5:],
            "current_dynamic": self.current_dynamic,
            "ongoing_topics": self.ongoing_topics[:5],
            "unresolved_tensions": self.unresolved_tensions[:3],
        }


@dataclass
class BehaviorMappingLayer:
    """行为映射层：bot 对该用户应采取什么行为模式，第三阶段升级：增加行为策略深度"""

    preferred_tone: str = "自然"
    reply_length_hint: str = "中等"
    humor_level: float = 0.5
    formality_level: float = 0.5
    proactive_degree: float = 0.3
    avoidance_topics: List[str] = field(default_factory=list)
    encouraged_topics: List[str] = field(default_factory=list)
    special_rules: List[str] = field(default_factory=list)
    social_openness_preference: float = 0.5
    emotional_resonance_target: float = 0.5
    intellectual_engagement_level: float = 0.5
    conflict_resolution_style: str = "温和"
    support_providing_mode: str = "适度"
    feedback_reception_mode: str = "开放"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preferred_tone": self.preferred_tone,
            "reply_length_hint": self.reply_length_hint,
            "humor_level": round(self.humor_level, 2),
            "formality_level": round(self.formality_level, 2),
            "proactive_degree": round(self.proactive_degree, 2),
            "avoidance_topics": self.avoidance_topics[:5],
            "encouraged_topics": self.encouraged_topics[:5],
            "special_rules": self.special_rules[:3],
        }


@dataclass
class InteractionStyleProfile:
    """USR-06: 用户互动风格画像 — 统计每个用户的发言特征"""

    avg_message_length: float = 0.0
    emoji_frequency: float = 0.0
    question_ratio: float = 0.0
    media_share_ratio: float = 0.0
    response_speed_tier: str = "normal"
    total_style_samples: int = 0
    _cumulative_msg_len: float = 0.0
    _cumulative_emoji_count: int = 0
    _cumulative_question_count: int = 0
    _cumulative_media_count: int = 0

    def update(
        self,
        *,
        msg_length: int = 0,
        has_emoji: bool = False,
        is_question: bool = False,
        has_media: bool = False,
        response_seconds: float = -1.0,
    ) -> None:
        """每次交互时调用，增量更新互动风格统计"""
        self.total_style_samples += 1
        n = self.total_style_samples
        self._cumulative_msg_len += msg_length
        self.avg_message_length = self._cumulative_msg_len / n
        if has_emoji:
            self._cumulative_emoji_count += 1
        self.emoji_frequency = self._cumulative_emoji_count / n
        if is_question:
            self._cumulative_question_count += 1
        self.question_ratio = self._cumulative_question_count / n
        if has_media:
            self._cumulative_media_count += 1
        self.media_share_ratio = self._cumulative_media_count / n
        if response_seconds >= 0:
            if response_seconds < 5:
                self.response_speed_tier = "instant"
            elif response_seconds < 30:
                self.response_speed_tier = "fast"
            elif response_seconds < 120:
                self.response_speed_tier = "normal"
            else:
                self.response_speed_tier = "slow"

    def style_label(self) -> str:
        """推导一个简短的互动风格标签"""
        if self.total_style_samples < 3:
            return "未知"
        labels = []
        if self.avg_message_length > 80:
            labels.append("长文型")
        elif self.avg_message_length < 15:
            labels.append("简短型")
        if self.emoji_frequency > 0.4:
            labels.append("表情达人")
        if self.question_ratio > 0.3:
            labels.append("好奇宝宝")
        if self.media_share_ratio > 0.2:
            labels.append("爱分享")
        return "/".join(labels) if labels else "普通"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_message_length": round(self.avg_message_length, 1),
            "emoji_frequency": round(self.emoji_frequency, 3),
            "question_ratio": round(self.question_ratio, 3),
            "media_share_ratio": round(self.media_share_ratio, 3),
            "response_speed_tier": self.response_speed_tier,
            "style_label": self.style_label(),
            "sample_count": self.total_style_samples,
        }


@dataclass
class TopicPreferenceMap:
    """USR-07: 用户话题偏好图谱 — 追踪用户参与不同话题的权重"""

    topic_weights: Dict[str, float] = field(default_factory=dict)
    topic_counts: Dict[str, int] = field(default_factory=dict)
    total_topic_mentions: int = 0
    _max_topics: int = 30

    def record_topic(self, topic: str, weight: float = 1.0) -> None:
        """记录一次话题参与"""
        if not topic:
            return
        topic = topic.strip()[:30]
        self.total_topic_mentions += 1
        self.topic_counts[topic] = self.topic_counts.get(topic, 0) + 1
        # 指数移动平均更新权重
        old_w = self.topic_weights.get(topic, 0.0)
        self.topic_weights[topic] = old_w * 0.8 + weight * 0.2
        # 容量控制：淘汰最低权重
        if len(self.topic_weights) > self._max_topics:
            min_topic = min(self.topic_weights, key=self.topic_weights.get)
            del self.topic_weights[min_topic]
            self.topic_counts.pop(min_topic, None)

    def top_topics(self, limit: int = 5) -> List[Tuple[str, float]]:
        """返回权重最高的话题列表"""
        sorted_items = sorted(
            self.topic_weights.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_items[:limit]

    def preference_for(self, topic: str) -> float:
        """查询对特定话题的偏好强度"""
        return self.topic_weights.get(topic, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_topics": [
                {"topic": t, "weight": round(w, 3)}
                for t, w in self.top_topics(8)
            ],
            "total_mentions": self.total_topic_mentions,
            "unique_topics": len(self.topic_weights),
        }


@dataclass
class BehaviorHabitProfile:
    """USR-08: 用户行为习惯画像 — 追踪活跃时段和频率模式"""

    hourly_activity: List[int] = field(default_factory=lambda: [0] * 24)
    weekday_activity: List[int] = field(default_factory=lambda: [0] * 7)
    total_sessions: int = 0
    last_active_hour: int = -1
    last_active_weekday: int = -1
    consecutive_days_active: int = 0
    _last_active_date: str = ""

    def record_activity(self, timestamp: float = 0.0) -> None:
        """记录一次活跃行为"""
        import datetime

        ts = timestamp or time.time()
        dt = datetime.datetime.fromtimestamp(ts)
        hour = dt.hour
        weekday = dt.weekday()
        date_str = dt.strftime("%Y-%m-%d")
        self.hourly_activity[hour] += 1
        self.weekday_activity[weekday] += 1
        self.last_active_hour = hour
        self.last_active_weekday = weekday
        self.total_sessions += 1
        # 连续活跃天数
        if date_str != self._last_active_date:
            if self._last_active_date:
                prev = datetime.datetime.strptime(
                    self._last_active_date, "%Y-%m-%d"
                )
                diff = (dt.date() - prev.date()).days
                if diff == 1:
                    self.consecutive_days_active += 1
                elif diff > 1:
                    self.consecutive_days_active = 1
            else:
                self.consecutive_days_active = 1
            self._last_active_date = date_str

    def peak_hours(self, top_n: int = 3) -> List[int]:
        """返回活跃峰值小时"""
        indexed = [
            (count, hour) for hour, count in enumerate(self.hourly_activity)
        ]
        indexed.sort(reverse=True)
        return [hour for count, hour in indexed[:top_n] if count > 0]

    def activity_pattern_label(self) -> str:
        """推导活跃模式标签"""
        if self.total_sessions < 5:
            return "未知"
        peaks = self.peak_hours(3)
        if not peaks:
            return "低频"
        morning = any(6 <= h < 12 for h in peaks)
        afternoon = any(12 <= h < 18 for h in peaks)
        evening = any(18 <= h < 24 for h in peaks)
        night = any(0 <= h < 6 for h in peaks)
        if night and not morning:
            return "夜猫子"
        if morning and not evening:
            return "早起鸟"
        if evening:
            return "晚间活跃"
        if afternoon:
            return "午后活跃"
        return "全天候"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_hours": self.peak_hours(3),
            "pattern_label": self.activity_pattern_label(),
            "total_sessions": self.total_sessions,
            "consecutive_days": self.consecutive_days_active,
            "weekday_preference": [
                ("周一", self.weekday_activity[0]),
                ("周二", self.weekday_activity[1]),
                ("周三", self.weekday_activity[2]),
                ("周四", self.weekday_activity[3]),
                ("周五", self.weekday_activity[4]),
                ("周六", self.weekday_activity[5]),
                ("周日", self.weekday_activity[6]),
            ],
        }


@dataclass
class UserImpression:
    """用户完整印象档案"""

    user_id: str = ""
    user_name: str = ""
    truth: TruthLayer = field(default_factory=TruthLayer)
    tags: SubjectiveTagLayer = field(default_factory=SubjectiveTagLayer)
    narrative: RelationshipNarrative = field(
        default_factory=RelationshipNarrative
    )
    behavior_map: BehaviorMappingLayer = field(
        default_factory=BehaviorMappingLayer
    )
    interaction_style: InteractionStyleProfile = field(
        default_factory=InteractionStyleProfile
    )
    topic_preference: TopicPreferenceMap = field(
        default_factory=TopicPreferenceMap
    )
    behavior_habit: BehaviorHabitProfile = field(
        default_factory=BehaviorHabitProfile
    )
    created_at: float = field(default_factory=time.time)
    last_evolved_at: float = field(default_factory=time.time)
    evolution_count: int = 0

    def compact_summary(self) -> str:
        """生成紧凑的一行摘要"""
        parts = []
        if self.tags.custom_nickname:
            parts.append(f"昵称={self.tags.custom_nickname}")
        if self.tags.overall_feeling != "一般":
            parts.append(f"感觉={self.tags.overall_feeling}")
        if self.narrative.relationship_stage != "陌生":
            parts.append(f"阶段={self.narrative.relationship_stage}")
        parts.append(f"互动{self.truth.total_interactions}次")
        if self.tags.personality_tags:
            parts.append(f"标签=[{','.join(self.tags.personality_tags[:3])}]")
        _style = self.interaction_style.style_label()
        if _style and _style != "未知":
            parts.append(f"风格={_style}")
        _habit = self.behavior_habit.activity_pattern_label()
        if _habit and _habit != "未知":
            parts.append(f"习惯={_habit}")
        return "，".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "truth": self.truth.to_dict(),
            "tags": self.tags.to_dict(),
            "narrative": self.narrative.to_dict(),
            "behavior_map": self.behavior_map.to_dict(),
            "interaction_style": self.interaction_style.to_dict(),
            "topic_preference": self.topic_preference.to_dict(),
            "behavior_habit": self.behavior_habit.to_dict(),
            "evolution_count": self.evolution_count,
            "compact_summary": self.compact_summary(),
        }


class ImpressionEvolutionHub:
    """印象演化中心
    维护四层印象：量化真值 → 主观标签 → 关系叙事 → 行为映射。
    每层都可独立更新，也可联动推演。
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._impressions: Dict[str, UserImpression] = {}
        # 周期性重写参数
        self._rewrite_interval_sec: float = 3600.0
        self._min_interactions_for_rewrite: int = 5

    def get_or_create(
        self, user_id: str, user_name: str = ""
    ) -> UserImpression:
        if user_id not in self._impressions:
            self._impressions[user_id] = UserImpression(
                user_id=user_id,
                user_name=user_name or user_id,
            )
        imp = self._impressions[user_id]
        if user_name and not imp.user_name:
            imp.user_name = user_name
        return imp

    def record_interaction(
        self,
        user_id: str,
        *,
        user_name: str = "",
        is_positive: bool = True,
        quality: float = 0.5,
        topic: str = "",
        event_summary: str = "",
        emotional_intensity: float = 0.0,
        shared_experience: bool = False,
        mutual_understanding: float = 0.0,
        msg_length: int = 0,
        has_emoji: bool = False,
        is_question: bool = False,
        has_media: bool = False,
        response_seconds: float = -1.0,
    ) -> UserImpression:
        """记录一次交互并推进印象演化，第三阶段升级：加入情感和共享体验维度"""
        imp = self.get_or_create(user_id, user_name)
        now = time.time()
        # 更新真值层
        imp.truth.total_interactions += 1
        if is_positive:
            imp.truth.positive_interactions += 1
        else:
            imp.truth.negative_interactions += 1
        imp.truth.last_interaction_at = now
        # 平滑更新平均回复质量
        alpha = 0.1
        imp.truth.average_reply_quality = (
            imp.truth.average_reply_quality * (1 - alpha) + quality * alpha
        )
        # 第三阶段新增：情感交流和共享体验计数
        if emotional_intensity > 0.3:
            imp.truth.emotional_exchange_count += 1
        if shared_experience:
            imp.truth.shared_experience_count += 1
        # 更新相互理解分数
        imp.truth.mutual_understanding_score = (
            imp.truth.mutual_understanding_score * 0.9
            + mutual_understanding * 0.1
        )
        # 更新标签层
        if topic:
            if topic not in imp.tags.interest_topics:
                imp.tags.interest_topics.append(topic)
                if len(imp.tags.interest_topics) > 10:
                    imp.tags.interest_topics = imp.tags.interest_topics[-10:]
            if topic not in imp.narrative.ongoing_topics:
                imp.narrative.ongoing_topics.append(topic)
                if len(imp.narrative.ongoing_topics) > 8:
                    imp.narrative.ongoing_topics = (
                        imp.narrative.ongoing_topics[-8:]
                    )
        # 第三阶段新增：更新感性维度
        self._update_subjective_dimensions(
            imp, quality, emotional_intensity, topic
        )
        # 事件摘要加入叙事层
        if event_summary:
            imp.tags.add_moment(event_summary)
        if shared_experience and event_summary:
            imp.narrative.shared_memories.append(event_summary[:100])
            if len(imp.narrative.shared_memories) > 5:
                imp.narrative.shared_memories = imp.narrative.shared_memories[
                    -5:
                ]
        imp.tags.last_updated = now
        imp.last_evolved_at = now
        imp.evolution_count += 1
        # 每10次互动触发一次阶段评估
        if imp.evolution_count % 10 == 0:
            self._evolve_relationship_stage(imp)
            self._evolve_behavior_map(imp)
        # USR-06: 更新互动风格画像
        imp.interaction_style.update(
            msg_length=msg_length,
            has_emoji=has_emoji,
            is_question=is_question,
            has_media=has_media,
            response_seconds=response_seconds,
        )
        # USR-07: 更新话题偏好
        if topic:
            imp.topic_preference.record_topic(topic, weight=quality)
        # USR-08: 更新行为习惯
        imp.behavior_habit.record_activity(now)
        # 将主观标签同步写入 PhaseTracker 的 PersonaImpression
        self._sync_tags_to_phase_tracker(user_id, imp.tags.personality_tags)
        return imp

    def record_conflict(
        self, user_id: str, *, user_name: str = "", reason: str = ""
    ) -> None:
        """记录冲突事件"""
        imp = self.get_or_create(user_id, user_name)
        imp.truth.conflict_count += 1
        imp.truth.negative_interactions += 1
        imp.truth.total_interactions += 1
        if reason:
            imp.narrative.unresolved_tensions.append(reason[:80])
            if len(imp.narrative.unresolved_tensions) > 5:
                imp.narrative.unresolved_tensions = (
                    imp.narrative.unresolved_tensions[-5:]
                )
        imp.last_evolved_at = time.time()

    def record_cooperation(
        self, user_id: str, *, user_name: str = "", event: str = ""
    ) -> None:
        """记录合作事件"""
        imp = self.get_or_create(user_id, user_name)
        imp.truth.cooperation_count += 1
        imp.truth.positive_interactions += 1
        imp.truth.total_interactions += 1
        if event:
            imp.narrative.add_turning_point(f"合作：{event}")
        imp.last_evolved_at = time.time()

    def add_personality_tag(self, user_id: str, tag: str) -> None:
        imp = self.get_or_create(user_id)
        imp.tags.add_tag(tag)

    def set_custom_nickname(self, user_id: str, nickname: str) -> None:
        imp = self.get_or_create(user_id)
        imp.tags.custom_nickname = nickname

    def get_impression(self, user_id: str) -> Optional[UserImpression]:
        return self._impressions.get(user_id)

    def get_snapshot(self, user_id: str) -> Optional[Dict[str, Any]]:
        """公开导出指定用户的印象快照。"""
        imp = self._impressions.get(user_id)
        if imp is None:
            return None
        return imp.to_dict()

    def get_behavior_hints(self, user_id: str) -> Dict[str, Any]:
        """获取针对该用户的行为提示"""
        imp = self._impressions.get(user_id)
        if imp is None:
            return {"tone": "自然", "length": "中等", "proactive": 0.3}
        return imp.behavior_map.to_dict()

    def get_all_layers(self, user_id: str) -> Optional[Dict[str, Any]]:
        """返回指定用户的全部层引用，供主链快速访问各层对象"""
        imp = self._impressions.get(user_id)
        if imp is None:
            return None
        return {
            "truth": imp.truth,
            "subjective_tag": imp.tags,
            "narrative": imp.narrative,
            "behavior_map": imp.behavior_map,
            "interaction_style": imp.interaction_style,
            "topic_preference": imp.topic_preference,
            "behavior_habit": imp.behavior_habit,
        }

    def get_user_profile_summary(self, user_id: str) -> Dict[str, Any]:
        """返回用户三维画像摘要（互动风格+话题偏好+行为习惯）"""
        imp = self._impressions.get(user_id)
        if imp is None:
            return {}
        return {
            "interaction_style": imp.interaction_style.to_dict(),
            "topic_preference": imp.topic_preference.to_dict(),
            "behavior_habit": imp.behavior_habit.to_dict(),
            "style_label": imp.interaction_style.style_label(),
            "pattern_label": imp.behavior_habit.activity_pattern_label(),
            "top_topics": [t for t, _ in imp.topic_preference.top_topics(5)],
        }

    def classify_bot_attitude(
        self,
        user_id: str,
        *,
        bot_interaction_count: int = 0,
        recent_sentiment: float = 0.0,
    ) -> Dict[str, Any]:
        """USR-11: 判定指定用户对bot的态度"""
        imp = self._impressions.get(user_id)
        if imp is None:
            return {
                "attitude": "indifferent",
                "label": "无感",
                "confidence": 0.2,
                "signals": ["无印象记录"],
            }
        return _bot_attitude_classifier.classify(
            imp,
            bot_interaction_count=bot_interaction_count,
            recent_sentiment=recent_sentiment,
        )

    def all_user_ids(self) -> List[str]:
        return list(self._impressions.keys())

    def summary_for_prompt(self, user_id: str) -> str:
        """生成可注入 prompt 的印象摘要"""
        imp = self._impressions.get(user_id)
        if imp is None:
            return ""
        return imp.compact_summary()

    # ────────────────── 周期性重写 ──────────────────

    def needs_periodic_rewrite(self, user_id: str) -> bool:
        """检查该用户是否到了周期性重写的时机"""
        imp = self._impressions.get(user_id)
        if imp is None:
            return False
        now = time.time()
        since_last = now - imp.last_evolved_at
        if since_last < self._rewrite_interval_sec:
            return False
        if imp.truth.total_interactions < self._min_interactions_for_rewrite:
            return False
        return True

    def periodic_rewrite(self, user_id: str) -> bool:
        """对指定用户执行全量印象重写
        重新从真值层推导标签层、叙事层、行为映射层，
        不依赖增量更新，消除累积偏差。
        返回 True 表示执行了重写。
        """
        imp = self._impressions.get(user_id)
        if imp is None:
            return False
        # 重写关系阶段和整体感觉
        self._evolve_relationship_stage(imp)
        # 重写行为映射
        self._evolve_behavior_map(imp)
        # 重写情绪倾向标签
        imp.tags.emotional_tendency = self._infer_emotional_tendency(imp)
        # 自动更新主观命名
        auto_nick = self._generate_auto_nickname(imp)
        if auto_nick and not imp.tags.custom_nickname:
            imp.tags.custom_nickname = auto_nick
        # 清理过期的兴趣话题（保留最近的8个）
        if len(imp.tags.interest_topics) > 8:
            imp.tags.interest_topics = imp.tags.interest_topics[-8:]
        # 清理过期的叙事张力
        if len(imp.narrative.unresolved_tensions) > 3:
            imp.narrative.unresolved_tensions = (
                imp.narrative.unresolved_tensions[-3:]
            )
        imp.last_evolved_at = time.time()
        imp.evolution_count += 1
        logger.debug(
            f"[{self._channel_id}] 用户 {user_id} 印象周期重写完成 "
            f"阶段={imp.narrative.relationship_stage} "
            f"感觉={imp.tags.overall_feeling}"
        )
        return True

    def periodic_rewrite_all(self) -> int:
        """对所有达到重写条件的用户执行周期性重写"""
        rewritten = 0
        for user_id in list(self._impressions.keys()):
            if self.needs_periodic_rewrite(user_id):
                if self.periodic_rewrite(user_id):
                    rewritten += 1
        return rewritten

    # ────────────────── 自动主观命名 ──────────────────

    def auto_generate_nickname(self, user_id: str) -> str:
        """根据印象自动生成主观昵称"""
        imp = self._impressions.get(user_id)
        if imp is None:
            return ""
        nickname = self._generate_auto_nickname(imp)
        if nickname:
            imp.tags.custom_nickname = nickname
        return nickname

    def _generate_auto_nickname(self, imp: UserImpression) -> str:
        """内部方法：基于四层印象数据推导昵称
        命名规则:
          - 通过性格标签 + 关系阶段 + 显著行为特征生成
          - 好朋友阶段用亲昵称呼方式
          - 陌生/有隔阂阶段保持距离感
        """
        tags = imp.tags
        narrative = imp.narrative
        truth = imp.truth
        # 收集命名素材
        stage = narrative.relationship_stage
        feeling = tags.overall_feeling
        style = tags.speaking_style
        personality = tags.personality_tags[:3]
        # 亲密阶段直接取显著特征
        if stage in ("好朋友", "老朋友"):
            if personality:
                return f"{personality[0]}的{imp.user_name[:4]}"
            if style:
                return f"{style[:4]}系朋友"
            return f"老朋友{imp.user_name[:3]}"
        # 熟识/朋友阶段用特征命名
        if stage in ("朋友", "熟识", "熟人"):
            if personality:
                return f"{personality[0]}人"
            if tags.interest_topics:
                return f"聊{tags.interest_topics[0][:4]}的"
            return ""
        # 疏远/复杂关系用谨慎命名
        if stage in ("疏远", "有隔阂", "复杂"):
            if truth.conflict_count > 3:
                return "需要注意的人"
            return ""
        # 初识/陌生阶段不自动命名
        return ""

    def _infer_emotional_tendency(self, imp: UserImpression) -> str:
        """从真值层推导情绪倾向"""
        pos_ratio = imp.truth.positivity_ratio()
        conflicts = imp.truth.conflict_count
        total = imp.truth.total_interactions
        if total < 3:
            return "未知"
        if pos_ratio > 0.8:
            return "积极友善"
        if pos_ratio > 0.6:
            return "比较友好"
        if conflicts > 5 and pos_ratio < 0.4:
            return "容易起冲突"
        if pos_ratio < 0.3:
            return "偏消极"
        return "中性"

    # ────────────────── 标签同步 ──────────────────

    def _sync_tags_to_phase_tracker(
        self, user_id: str, tags: List[str]
    ) -> None:
        """将 SubjectiveTagLayer 标签同步到 PhaseTracker 的 PersonaImpression"""
        if not tags:
            return
        try:
            from src.chat.heart_flow.fondness_trust import FondnessTrustDimension

            _tracker = PhaseTracker()
            _dossier = _tracker.fetch_dossier(user_id, self.channel_id)
            if _dossier and _dossier.impression:
                _existing = set(_dossier.impression.character_tags or [])
                _new_tags = [t for t in tags if t not in _existing]
                if _new_tags:
                    _dossier.impression.character_tags = list(
                        _existing | set(_new_tags)
                    )
                    if len(_dossier.impression.character_tags) > 15:
                        _dossier.impression.character_tags = (
                            _dossier.impression.character_tags[-15:]
                        )
        except Exception as exc:
            logger.debug(f"标签同步到PhaseTracker失败({user_id}): {exc}")

    # ────────────────── 演化逻辑 ──────────────────

    def _evolve_relationship_stage(self, imp: UserImpression) -> None:
        """根据真值层数据推演关系阶段"""
        total = imp.truth.total_interactions
        pos_ratio = imp.truth.positivity_ratio()
        conflicts = imp.truth.conflict_count
        if total < 5:
            imp.narrative.relationship_stage = "初识"
            imp.tags.overall_feeling = "一般"
        elif total < 20:
            if pos_ratio > 0.7:
                imp.narrative.relationship_stage = "熟识"
                imp.tags.overall_feeling = "不错"
            elif pos_ratio < 0.3:
                imp.narrative.relationship_stage = "有隔阂"
                imp.tags.overall_feeling = "一般"
            else:
                imp.narrative.relationship_stage = "普通"
                imp.tags.overall_feeling = "一般"
        elif total < 50:
            if pos_ratio > 0.75 and conflicts < 3:
                imp.narrative.relationship_stage = "朋友"
                imp.tags.overall_feeling = "喜欢"
            elif pos_ratio < 0.3 or conflicts > 5:
                imp.narrative.relationship_stage = "疏远"
                imp.tags.overall_feeling = "不太舒服"
            else:
                imp.narrative.relationship_stage = "熟人"
                imp.tags.overall_feeling = "还行"
        else:
            if pos_ratio > 0.8 and conflicts < 5:
                imp.narrative.relationship_stage = "好朋友"
                imp.tags.overall_feeling = "很喜欢"
            elif pos_ratio > 0.6:
                imp.narrative.relationship_stage = "老朋友"
                imp.tags.overall_feeling = "喜欢"
            elif conflicts > 10:
                imp.narrative.relationship_stage = "复杂"
                imp.tags.overall_feeling = "复杂"
            else:
                imp.narrative.relationship_stage = "熟人"
                imp.tags.overall_feeling = "还行"

    def _evolve_behavior_map(self, imp: UserImpression) -> None:
        """根据交互数据推演行为映射"""
        stage = imp.narrative.relationship_stage
        conflicts = imp.truth.conflict_count
        bm = imp.behavior_map
        if stage in ("好朋友", "老朋友"):
            bm.preferred_tone = "亲近"
            bm.humor_level = 0.7
            bm.formality_level = 0.2
            bm.proactive_degree = 0.6
            bm.reply_length_hint = "可以长一点"
        elif stage in ("朋友", "熟识"):
            bm.preferred_tone = "友好"
            bm.humor_level = 0.5
            bm.formality_level = 0.3
            bm.proactive_degree = 0.4
            bm.reply_length_hint = "中等"
        elif stage in ("疏远", "有隔阂"):
            bm.preferred_tone = "礼貌但疏离"
            bm.humor_level = 0.2
            bm.formality_level = 0.7
            bm.proactive_degree = 0.1
            bm.reply_length_hint = "简短"
        elif stage == "复杂":
            bm.preferred_tone = "谨慎"
            bm.humor_level = 0.3
            bm.formality_level = 0.5
            bm.proactive_degree = 0.2
            bm.reply_length_hint = "适中"
        else:
            bm.preferred_tone = "自然"
            bm.humor_level = 0.5
            bm.formality_level = 0.5
            bm.proactive_degree = 0.3
            bm.reply_length_hint = "中等"
        # 冲突多 → 回避敏感话题
        if conflicts > 3:
            if "敏感话题" not in bm.avoidance_topics:
                bm.avoidance_topics.append("敏感话题")

    async def assemble_relation_snapshot(self, user_id: str) -> Dict[str, Any]:
        """从多源聚合完整的关系/情绪/社交快照，作为 WorldSnapshot 的降级替代"""
        result = {
            "chat_value": 100.0,
            "activity_level": 50.0,
            "shared_social_value": 0.0,
            "social_value": 0.0,
            "trust_value": 0.0,
            "annoyance_value": 0.0,
            "relationship_level": 2,
            "custom_label": "",
            "interaction_count": 0,
            "affection": 0.0,
            "trust_score": 0.0,
            "psychological_pressure": 0.0,
            "trauma_score": 0.0,
            "mood": "平静",
            "attribute_influences": {},
            "relationship_candidates": [],
            "compact_summary": "",
            "custom_nickname": "",
            "overall_feeling": "一般",
            "personality_tags": [],
            "ongoing_topics": [],
        }
        imp = self._impressions.get(user_id) if user_id else None
        if imp is not None and self.needs_periodic_rewrite(user_id):
            self.periodic_rewrite(user_id)
        # 共享资源池
        try:
            from src.config.core_config_engine import acquire_settings_hub

            acquire_settings_hub()
            from src.chat.heart_flow.energy_manager import (
                get_vitality_pool,
                get_shared_resource_manager,
            )

            pool = get_vitality_pool()
            shared_vals = get_shared_resource_manager().get_shared_values(
                self._channel_id
            )
            result["chat_value"] = float(shared_vals.get("chat_value") or 0.0)
            result["activity_level"] = float(
                shared_vals.get("activity_level") or 0.0
            )
            result["shared_social_value"] = float(
                shared_vals.get("social_value") or 0.0
            )
            if user_id:
                snap = await pool.capture_full_snapshot(
                    self._channel_id, user_id
                )
            else:
                snap = pool.capture_snapshot(self._channel_id)
            result["social_value"] = snap.social_value
            result["trust_value"] = snap.trust_value
            result["annoyance_value"] = snap.annoyance_value
            result["relationship_level"] = snap.relationship_level
            result["custom_label"] = snap.custom_label
            result["interaction_count"] = snap.interaction_count
        except Exception as exc:
            logger.warning(f"关系聚合-资源池读取失败: {exc}")
        # 情绪追踪
        if user_id:
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import (
                    get_emotion_tracker, )

                state = get_emotion_tracker(self._channel_id).get_user_state(
                    user_id, create_if_missing=False
                )
                if state is not None:
                    result["affection"] = float(
                        getattr(state, "affection", 0.0) or 0.0
                    )
                    result["trust_score"] = float(
                        getattr(state, "trust_score", 0.0) or 0.0
                    )
                    result["psychological_pressure"] = float(
                        getattr(state, "psychological_pressure", 0.0) or 0.0
                    )
                    result["trauma_score"] = float(
                        getattr(state, "trauma_score", 0.0) or 0.0
                    )
                    result["mood"] = str(
                        getattr(state, "mood", "平静") or "平静"
                    )
            except Exception:
                logger.debug(f"关系聚合-情绪追踪读取失败: {user_id}")
        # 关系候选人
        if user_id:
            try:
                from src.person_info.person_info import get_unified_profile_hub

                hub = get_unified_profile_hub()
                top_rels = hub.get_top_relationships(user_id, limit=5)
                result["relationship_candidates"] = [
                    (str(tid or "").strip(), float(w or 0.0))
                    for tid, w in top_rels
                    if str(tid or "").strip()
                ]
            except Exception:
                logger.debug(f"关系聚合-候选人读取失败: {user_id}")
        self._merge_local_impression_snapshot(result, imp)
        return result

    def _merge_local_impression_snapshot(
        self,
        result: Dict[str, Any],
        imp: Optional[UserImpression],
    ) -> None:
        """用本地印象档案补足外部资源层缺失时的关系快照。"""
        if imp is None:
            return
        positivity = imp.truth.positivity_ratio()
        inferred_relationship = self._relationship_stage_to_level(
            imp.narrative.relationship_stage
        )
        if int(result.get("interaction_count", 0) or 0) <= 0:
            result["interaction_count"] = imp.truth.total_interactions
        if int(result.get("relationship_level", 0) or 0) <= 2:
            result["relationship_level"] = inferred_relationship
        if not result.get("custom_label"):
            result["custom_label"] = (
                imp.tags.custom_nickname
                or imp.narrative.relationship_stage
                or ""
            )
        if not float(result.get("social_value", 0.0) or 0.0):
            result["social_value"] = round(
                max(
                    0.0,
                    min(
                        100.0,
                        positivity * 50.0 - imp.truth.conflict_count * 3.0,
                    ),
                ),
                1,
            )
        if not float(result.get("trust_value", 0.0) or 0.0):
            result["trust_value"] = round(
                max(
                    0.0,
                    min(100.0, imp.truth.mutual_understanding_score * 50.0),
                ),
                1,
            )
        if not float(result.get("annoyance_value", 0.0) or 0.0):
            try:
                _emo_annoy = get_emotion_tracker(
                    self._channel_id
                ).get_user_state(
                    str(self.user_id or ""), create_if_missing=False
                )
                if _emo_annoy is not None:
                    _emo_val = float(
                        getattr(_emo_annoy, "annoyance", -1) or -1
                    )
                    if _emo_val >= 0:
                        result["annoyance_value"] = round(_emo_val, 1)
                    else:
                        result["annoyance_value"] = round(
                            max(
                                0.0,
                                min(100.0, imp.truth.conflict_count * 12.0),
                            ),
                            1,
                        )
                else:
                    result["annoyance_value"] = round(
                        max(0.0, min(100.0, imp.truth.conflict_count * 12.0)),
                        1,
                    )
            except Exception:
                result["annoyance_value"] = round(
                    max(0.0, min(100.0, imp.truth.conflict_count * 12.0)), 1
                )
        if not float(result.get("affection", 0.0) or 0.0):
            result["affection"] = round(
                max(
                    0.0,
                    min(
                        100.0,
                        imp.truth.positive_interactions * 2.0
                        + imp.truth.shared_experience_count * 4.0,
                    ),
                ),
                1,
            )
        if not float(result.get("trust_score", 0.0) or 0.0):
            result["trust_score"] = round(
                max(
                    0.0,
                    min(100.0, imp.truth.mutual_understanding_score * 50.0),
                ),
                1,
            )
        if not float(result.get("psychological_pressure", 0.0) or 0.0):
            result["psychological_pressure"] = round(
                max(
                    0.0,
                    min(
                        100.0,
                        len(imp.narrative.unresolved_tensions) * 15.0
                        + imp.truth.conflict_count * 5.0,
                    ),
                ),
                1,
            )
        if not float(result.get("trauma_score", 0.0) or 0.0):
            result["trauma_score"] = round(
                max(0.0, min(100.0, imp.truth.conflict_count * 4.0)),
                1,
            )
        if result.get("mood", "平静") == "平静":
            result["mood"] = (
                imp.tags.overall_feeling
                if imp.tags.overall_feeling
                and imp.tags.overall_feeling != "一般"
                else (imp.tags.emotional_tendency or "平静")
            )
        result["compact_summary"] = imp.compact_summary()
        result["custom_nickname"] = imp.tags.custom_nickname
        result["overall_feeling"] = imp.tags.overall_feeling
        result["personality_tags"] = imp.tags.personality_tags[:8]
        result["ongoing_topics"] = imp.narrative.ongoing_topics[:5]

    def _relationship_stage_to_level(self, stage: str) -> int:
        """把叙事层阶段映射成兼容旧快照的数值级别。"""
        mapping = {
            "陌生": 1,
            "初识": 2,
            "普通": 3,
            "熟识": 4,
            "熟人": 5,
            "朋友": 6,
            "老朋友": 7,
            "好朋友": 8,
            "有隔阂": 2,
            "疏远": 1,
            "复杂": 3,
        }
        return mapping.get(stage, 2)

    # 第三阶段新增辅助方法
    def _update_subjective_dimensions(
        self,
        imp: UserImpression,
        quality: float,
        emotional_intensity: float,
        topic: str,
    ) -> None:
        """更新主观感性维度，第三阶段新增"""
        # 幽默感：基于互动质量和话题类型
        humor_boost = 0.1 if "玩笑" in topic or "梗" in topic else 0.0
        imp.tags.humor_sense = min(
            1.0, imp.tags.humor_sense + quality * 0.05 + humor_boost
        )
        # 同理心：基于情感强度
        imp.tags.empathy_level = min(
            1.0, imp.tags.empathy_level + emotional_intensity * 0.08
        )
        # 创造力：基于话题多样性和质量
        creativity_boost = 0.05 if len(imp.tags.interest_topics) > 3 else 0.0
        imp.tags.creativity_index = min(
            1.0, imp.tags.creativity_index + quality * 0.03 + creativity_boost
        )
        # 社交和谐：基于正面互动比例
        positivity = imp.truth.positivity_ratio()
        imp.tags.social_harmony = (
            imp.tags.social_harmony * 0.95 + positivity * 0.05
        )
        # 智力深度：基于话题复杂度和互动质量
        depth_boost = 0.02 if len(topic) > 20 else 0.0
        imp.tags.intellectual_depth = min(
            1.0, imp.tags.intellectual_depth + quality * 0.04 + depth_boost
        )
        # 情绪稳定性：基于冲突计数和正面互动
        stability_penalty = imp.truth.conflict_count * 0.01
        stability_boost = positivity * 0.02
        imp.tags.emotional_stability = max(
            0.0,
            min(
                1.0,
                imp.tags.emotional_stability
                - stability_penalty
                + stability_boost,
            ),
        )

    def evolve_with_group_context(
        self,
        user_id: str,
        group_atmosphere: str,
        social_density: float,
        pattern_type: str,
    ) -> None:
        """基于群语境演化印象，第三阶段新增"""
        imp = self._impressions.get(user_id)
        if not imp:
            return
        # 群氛围影响
        if group_atmosphere == "celebration":
            imp.tags.social_harmony = min(1.0, imp.tags.social_harmony + 0.05)
        elif group_atmosphere == "heated_discussion":
            imp.tags.intellectual_depth = min(
                1.0, imp.tags.intellectual_depth + 0.03
            )
        # 社交密度影响
        if social_density > 0.7:
            imp.behavior_map.social_openness_preference = min(
                1.0, imp.behavior_map.social_openness_preference + 0.02
            )
        # 群体模式影响
        if pattern_type == "support_circle":
            imp.tags.empathy_level = min(1.0, imp.tags.empathy_level + 0.04)
            imp.behavior_map.support_providing_mode = "积极"
        elif pattern_type == "debate_circle":
            imp.tags.intellectual_depth = min(
                1.0, imp.tags.intellectual_depth + 0.03
            )
            imp.behavior_map.intellectual_engagement_level = min(
                1.0, imp.behavior_map.intellectual_engagement_level + 0.03
            )
        elif pattern_type == "emotional_contagion":
            imp.behavior_map.emotional_resonance_target = min(
                1.0, imp.behavior_map.emotional_resonance_target + 0.02
            )
        imp.last_evolved_at = time.time()


class BotAttitudeClassifier:
    """USR-11: 判定用户对bot的态度
    五种态度：友好(friendly)、调侃(teasing)、攻击(hostile)、围观(observing)、无感(indifferent)
    基于交互历史、情感维度、冲突记录等多源信号综合判定
    """

    ATTITUDE_FRIENDLY = "friendly"
    ATTITUDE_TEASING = "teasing"
    ATTITUDE_HOSTILE = "hostile"
    ATTITUDE_OBSERVING = "observing"
    ATTITUDE_INDIFFERENT = "indifferent"
    _attitude_labels = {
        "friendly": "友好",
        "teasing": "调侃",
        "hostile": "攻击",
        "observing": "围观",
        "indifferent": "无感",
    }

    def classify(
        self,
        imp: UserImpression,
        *,
        bot_interaction_count: int = 0,
        recent_sentiment: float = 0.0,
    ) -> Dict[str, Any]:
        """对指定用户进行态度分类
        参数:
            imp: 用户印象档案
            bot_interaction_count: 与bot直接交互次数
            recent_sentiment: 最近交互的情感极性 (-1~+1)
        返回:
            {"attitude": str, "label": str, "confidence": float, "signals": list}
        """
        truth = imp.truth
        tags = imp.tags
        total = truth.total_interactions
        if total < 2:
            return {
                "attitude": self.ATTITUDE_INDIFFERENT,
                "label": "无感",
                "confidence": 0.3,
                "signals": ["交互不足"],
            }
        signals = []
        scores = {
            self.ATTITUDE_FRIENDLY: 0.0,
            self.ATTITUDE_TEASING: 0.0,
            self.ATTITUDE_HOSTILE: 0.0,
            self.ATTITUDE_OBSERVING: 0.0,
            self.ATTITUDE_INDIFFERENT: 0.0,
        }
        pos_ratio = truth.positivity_ratio()
        # 信号1: 正面互动比例
        if pos_ratio > 0.75:
            scores[self.ATTITUDE_FRIENDLY] += 0.35
            signals.append("高正面比")
        elif pos_ratio > 0.5:
            scores[self.ATTITUDE_FRIENDLY] += 0.15
            scores[self.ATTITUDE_TEASING] += 0.1
        elif pos_ratio < 0.3:
            scores[self.ATTITUDE_HOSTILE] += 0.3
            signals.append("低正面比")
        # 信号2: 冲突频率
        conflict_rate = truth.conflict_count / max(total, 1)
        if conflict_rate > 0.2:
            scores[self.ATTITUDE_HOSTILE] += 0.3
            signals.append("高冲突率")
        elif conflict_rate > 0.1:
            scores[self.ATTITUDE_TEASING] += 0.2
            signals.append("中等冲突")
        # 信号3: 与bot的直接交互频率
        bot_interact_ratio = bot_interaction_count / max(total, 1)
        if bot_interact_ratio > 0.5:
            scores[self.ATTITUDE_FRIENDLY] += 0.2
            signals.append("频繁与bot交互")
        elif bot_interact_ratio < 0.1 and total > 10:
            scores[self.ATTITUDE_OBSERVING] += 0.25
            scores[self.ATTITUDE_INDIFFERENT] += 0.15
            signals.append("极少与bot交互")
        # 信号4: 合作与情感交流
        if truth.cooperation_count > 2:
            scores[self.ATTITUDE_FRIENDLY] += 0.15
            signals.append("有合作记录")
        if truth.emotional_exchange_count > 3:
            scores[self.ATTITUDE_FRIENDLY] += 0.1
            signals.append("有情感交流")
        # 信号5: 社交和谐与幽默感
        if tags.social_harmony > 0.7:
            scores[self.ATTITUDE_FRIENDLY] += 0.1
        if tags.humor_sense > 0.7:
            scores[self.ATTITUDE_TEASING] += 0.15
            signals.append("幽默感高")
        # 信号6: 最近情感极性
        if recent_sentiment > 0.3:
            scores[self.ATTITUDE_FRIENDLY] += 0.15
        elif recent_sentiment < -0.3:
            scores[self.ATTITUDE_HOSTILE] += 0.15
            signals.append("近期负面")
        # 信号7: 互动风格（问句多可能是好奇/友好）
        if imp.interaction_style.question_ratio > 0.3:
            scores[self.ATTITUDE_FRIENDLY] += 0.05
        # 信号8: 活跃度低且无特殊表现 → 围观/无感
        if (
            total < 5
            and truth.conflict_count == 0
            and bot_interaction_count < 2
        ):
            scores[self.ATTITUDE_OBSERVING] += 0.2
            scores[self.ATTITUDE_INDIFFERENT] += 0.2
        # 取最高分作为判定结果
        best_attitude = max(scores, key=scores.get)
        best_score = scores[best_attitude]
        total_score = sum(scores.values()) or 1.0
        confidence = (
            min(1.0, best_score / total_score * 1.5)
            if total_score > 0
            else 0.3
        )
        return {
            "attitude": best_attitude,
            "label": self._attitude_labels.get(best_attitude, "未知"),
            "confidence": round(confidence, 3),
            "signals": signals,
            "scores": {k: round(v, 3) for k, v in scores.items()},
        }


_bot_attitude_classifier = BotAttitudeClassifier()


def get_impression_hub(channel_id: str) -> ImpressionEvolutionHub:
    if channel_id not in _impression_instances:
        _impression_instances[channel_id] = ImpressionEvolutionHub(channel_id)
    return _impression_instances[channel_id]


def remove_impression_hub(channel_id: str) -> None:
    _impression_instances.pop(channel_id, None)

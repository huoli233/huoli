import math
import time
import hashlib
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.logger import get_logger
from src.common.singleton import _get_class_lock

logger = get_logger("印象引擎")

# ──────────────────────────────────────────────
#  枚举与常量
# ──────────────────────────────────────────────


class RelationNarrativeType(Enum):
    """关系叙事类型bot对该用户的内部定位"""

    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    FRIEND = "friend"
    CLOSE_FRIEND = "close_friend"
    CONFIDANT = "confidant"
    MENTOR = "mentor"
    MENTEE = "mentee"
    RIVAL = "rival"
    ANNOYING = "annoying"
    IGNORED = "ignored"


class BehaviorMappingRule(Enum):
    """行为映射规则——对该用户的默认回应策略"""

    NORMAL = "normal"
    WARM = "warm"
    PLAYFUL = "playful"
    RESPECTFUL = "respectful"
    CAUTIOUS = "cautious"
    DRY = "dry"
    AVOIDANT = "avoidant"
    TEASING = "teasing"
    PROTECTIVE = "protective"


@dataclass
class SubjectiveName:
    """主观命名——机器人在内心如何称呼这个用户"""

    display_name: str = ""
    internal_nickname: str = ""
    nickname_origin: str = ""
    tags: List[str] = field(default_factory=list)
    tag_confidence: Dict[str, float] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


@dataclass
class RelationNarrative:
    """关系叙事——"我们之间是什么关系"的内部故事"""

    narrative_type: RelationNarrativeType = RelationNarrativeType.STRANGER
    story_summary: str = ""
    key_moments: List[Dict[str, Any]] = field(default_factory=list)
    trust_trajectory: deque = field(default_factory=lambda: deque(maxlen=50))
    affection_trajectory: deque = field(
        default_factory=lambda: deque(maxlen=50)
    )
    conflict_count: int = 0
    support_count: int = 0
    last_interaction_type: str = ""
    relationship_age_days: float = 0.0


@dataclass
class BehaviorMapping:
    """行为映射——对该用户的回应策略规则集"""

    primary_rule: BehaviorMappingRule = BehaviorMappingRule.NORMAL
    secondary_rules: List[BehaviorMappingRule] = field(default_factory=list)
    reply_length_bias: float = 1.0
    emoji_usage_bias: float = 1.0
    formality_level: float = 0.5
    teasing_allowed: bool = False
    sarcasm_allowed: bool = False
    personal_topic_depth: float = 0.5
    response_delay_modifier: float = 1.0
    interrupt_tolerance: float = 0.5
    custom_rules: Dict[str, float] = field(default_factory=dict)


@dataclass
class UserImpression:
    """单个用户的完整印象快照"""

    user_id: str = ""
    subjective_name: SubjectiveName = field(default_factory=SubjectiveName)
    relation_narrative: RelationNarrative = field(
        default_factory=RelationNarrative
    )
    behavior_mapping: BehaviorMapping = field(default_factory=BehaviorMapping)
    raw_affection: float = 50.0
    raw_trust: float = 50.0
    raw_familiarity: float = 0.0
    interaction_count: int = 0
    last_direct_interaction: float = 0.0
    last_observed_interaction: float = 0.0
    total_words_exchanged: int = 0
    positive_events: int = 0
    negative_events: int = 0
    impression_hash: str = ""


# ──────────────────────────────────────────────
#  核心：个体化印象演化引擎
# ──────────────────────────────────────────────


class IndividualImpressionEngine:
    """个体化印象演化引擎

    管理所有用户的印象数据，提供：
      - 三层模型的读写接口
      - 三路径更新（直接/旁观/时间衰减）
      - 印象→行为参数的自动推导
      - 印象一致性校验（防止矛盾）
    """

    MAX_USERS_TRACKED = 200
    IMPRESSION_DECAY_HALF_LIFE_DAYS = 30.0
    NARRATIVE_TRANSITION_THRESHOLD = 0.15

    def __init__(self):
        self._impressions: Dict[str, UserImpression] = {}
        self._global_stats = {
            "total_impressions": 0,
            "avg_affection": 50.0,
            "narrative_distribution": {},
        }
        self._last_decay_pass: float = 0.0
        self._decay_interval_sec: float = 3600.0

    def get_or_create(
        self, user_id: str, display_name: str = ""
    ) -> UserImpression:
        """获取或创建用户印象（主入口）"""
        uid_key = self._hash_user_id(user_id)
        if uid_key in self._impressions:
            return self._impressions[uid_key]
        imp = UserImpression(
            user_id=user_id,
            impression_hash=uid_key,
        )
        if display_name:
            imp.subjective_name.display_name = display_name
            imp.subjective_name.internal_nickname = display_name
        self._impressions[uid_key] = imp
        self._global_stats["total_impressions"] += 1
        logger.debug(
            f"[印象引擎] 创建新用户印象: {display_name or user_id[:8]}"
        )
        return imp

    def update_from_direct_interaction(
        self, user_id: str, display_name: str, interaction_data: Dict[str, Any]
    ) -> UserImpression:
        """Path A: 直接交互更新 —— 每次对话后调用

        interaction_data 包含:
          - message_length: 消息长度
          - sentiment: 情感极性 (-1~1)
          - topics: 提及的话题列表
          - was_helpful: 是否有帮助
          - was_hostile: 是否有敌意
          - was_teasing: 是否在开玩笑
          - mentioned_bot: 是否提及机器人
          - emoji_used: 使用表情数量
        """
        imp = self.get_or_create(user_id, display_name)
        now = time.time()
        imp.interaction_count += 1
        imp.last_direct_interaction = now
        # 更新原始指标
        sent = float(interaction_data.get("sentiment", 0.0) or 0.0)
        msg_len = int(interaction_data.get("message_length", 0) or 0)
        imp.total_words_exchanged += max(1, msg_len // 3)
        if sent > 0.2:
            imp.positive_events += 1
            delta_aff = min(2.0, sent * (1.0 + msg_len / 500.0))
            imp.raw_affection = min(100.0, imp.raw_affection + delta_aff)
            delta_trust = min(1.5, sent * 0.7)
            imp.raw_trust = min(100.0, imp.raw_trust + delta_trust)
            imp.relation_narrative.support_count += 1
        elif sent < -0.2:
            imp.negative_events += 1
            delta_neg = min(-3.0, sent * 1.5)
            imp.raw_affection = max(0.0, imp.raw_affection + delta_neg)
            if sent < -0.5:
                delta_trust_neg = min(-2.0, sent * 1.2)
                imp.raw_trust = max(0.0, imp.raw_trust + delta_trust_neg)
                imp.relation_narrative.conflict_count += 1
        # 熟悉度增长
        fam_gain = min(1.0, 1.0 / max(1, imp.interaction_count) * 0.5)
        imp.raw_familiarity = min(100.0, imp.raw_familiarity + fam_gain)
        # 更新轨迹
        imp.relation_narrative.affection_trajectory.append(
            (now, imp.raw_affection)
        )
        imp.relation_narrative.trust_trajectory.append((now, imp.raw_trust))
        imp.relation_narrative.last_interaction_type = (
            "hostile"
            if sent < -0.3
            else "supportive" if sent > 0.3 else "neutral"
        )
        # 标签演化
        self._evolve_tags_from_interaction(imp, interaction_data)
        # 自动推导叙事类型
        self._update_narrative_type(imp)
        # 自动推导行为映射
        self._derive_behavior_mapping(imp)
        # 记录关键时刻
        if abs(sent) > 0.5 or interaction_data.get("mentioned_bot"):
            imp.relation_narrative.key_moments.append(
                {
                    "time": now,
                    "type": "direct",
                    "sentiment": sent,
                    "summary": f"{'积极' if sent > 0 else '消极'}直接交互",
                }
            )
            if len(imp.relation_narrative.key_moments) > 20:
                imp.relation_narrative.key_moments.pop(0)
        return imp

    def update_from_observation(
        self,
        observer_user_id: str,
        target_user_id: str,
        target_display: str,
        observation_data: Dict[str, Any],
    ) -> None:
        """Path B: 旁观数据更新 —— 看到目标用户与其他人互动时调用

        这是间接学习：机器人通过观察用户与他人的相处模式来推断其性格
        """
        target_imp = self.get_or_create(target_user_id, target_display)
        now = time.time()
        target_imp.last_observed_interaction = now
        obs_type = str(observation_data.get("type", "") or "")
        obs_sentiment = float(observation_data.get("sentiment", 0.0) or 0.0)
        was_leader = bool(observation_data.get("was_leader", False))
        helped_someone = bool(observation_data.get("helped_someone", False))
        caused_conflict = bool(observation_data.get("caused_conflict", False))
        # 旁观权重低于直接交互（约 40%）
        obs_weight = 0.40
        if obs_sentiment > 0.3:
            target_imp.positive_events += 1
            target_imp.raw_affection = min(
                100.0,
                target_imp.raw_affection + obs_sentiment * 0.12 * obs_weight,
            )
            if helped_someone:
                target_imp.raw_affection = min(
                    100.0, target_imp.raw_affection + 0.08 * obs_weight
                )
                target_imp.raw_trust = min(
                    100.0, target_imp.raw_trust + 0.05 * obs_weight
                )
        elif obs_sentiment < -0.3:
            target_imp.negative_events += 1
            target_imp.raw_affection = max(
                0.0,
                target_imp.raw_affection + obs_sentiment * 0.10 * obs_weight,
            )
            if caused_conflict:
                target_imp.raw_affection = max(
                    0.0, target_imp.raw_affection - 0.12 * obs_weight
                )
                target_imp.raw_trust = max(
                    0.0, target_imp.raw_trust - 0.08 * obs_weight
                )
        if was_leader:
            target_imp.subjective_name.tag_confidence["leader"] = (
                target_imp.subjective_name.tag_confidence.get("leader", 0.0)
                + 0.08 * obs_weight
            )
            if "leader" not in target_imp.subjective_name.tags:
                target_imp.subjective_name.tags.append("leader")
        # 记录观察到的关键时刻
        if abs(obs_sentiment) > 0.4:
            target_imp.relation_narrative.key_moments.append(
                {
                    "time": now,
                    "type": "observation",
                    "obs_type": obs_type,
                    "sentiment": obs_sentiment,
                }
            )
        self._derive_behavior_mapping(target_imp)

    def apply_time_decay(self, now: Optional[float] = None) -> Dict[str, int]:
        """Path C: 时间衰减 + 自然漂移 —— 定期调用（建议每小时一次）

        长期不互动的用户：
          - 好感缓慢向中性回归（半衰期 ~30天）
          - 信任更慢地回归（信任建立难、丢失也慢）
          - 熟悉度几乎不衰减（一旦认识就不会忘记）
          - 标签置信度逐渐降低
          - 叙事可能降级（密友→好友→熟人）
        """
        now = now or time.time()
        if (now - self._last_decay_pass) < self._decay_interval_sec:
            return {"skipped": True, "reason": "too_soon"}
        self._last_decay_pass = now
        decayed_count = 0
        narrative_downgrades = 0
        for _uid_key, imp in list(self._impressions.items()):
            days_since_direct = 0.0
            if imp.last_direct_interaction > 0:
                days_since_direct = (
                    now - imp.last_direct_interaction
                ) / 86400.0
            else:
                days_since_direct = 999.0
            days_since_any = 0.0
            last_any = max(
                imp.last_direct_interaction, imp.last_observed_interaction
            )
            if last_any > 0:
                days_since_any = (now - last_any) / 86400.0
            else:
                days_since_any = 999.0
            if days_since_direct < 1.0:
                continue
            decayed_count += 1
            half_life = self.IMPRESSION_DECAY_HALF_LIFE_DAYS
            decay_factor = math.exp(-0.693 * days_since_direct / half_life)
            neutral_target = 50.0
            aff_range = imp.raw_affection - neutral_target
            imp.raw_affection = neutral_target + aff_range * decay_factor
            trust_half_life = half_life * 1.5
            trust_decay = math.exp(
                -0.693 * days_since_direct / trust_half_life
            )
            trust_range = imp.raw_trust - neutral_target
            imp.raw_trust = neutral_target + trust_range * trust_decay
            fam_half_life = half_life * 4.0
            fam_decay = math.exp(-0.693 * days_since_direct / fam_half_life)
            imp.raw_familiarity = max(5.0, imp.raw_familiarity * fam_decay)
            for tag in list(imp.subjective_name.tag_confidence.keys()):
                conf = imp.subjective_name.tag_confidence[tag]
                imp.subjective_name.tag_confidence[tag] = max(
                    0.02, conf * 0.98
                )
                if conf < 0.05 and tag in imp.subjective_name.tags:
                    imp.subjective_name.tags.remove(tag)
            # 叙事降级检查
            old_narr = imp.relation_narrative.narrative_type
            self._update_narrative_type(imp)
            new_narr = imp.relation_narrative.narrative_type
            if old_narr != new_narr:
                _narr_order = {
                    RelationNarrativeType.CONFIDANT: 6,
                    RelationNarrativeType.CLOSE_FRIEND: 5,
                    RelationNarrativeType.FRIEND: 4,
                    RelationNarrativeType.MENTOR: 4,
                    RelationNarrativeType.MENTEE: 4,
                    RelationNarrativeType.ACQUAINTANCE: 3,
                    RelationNarrativeType.RIVAL: 2,
                    RelationNarrativeType.ANNOYING: 1,
                    RelationNarrativeType.STRANGER: 0,
                    RelationNarrativeType.IGNORED: -1,
                }
                if _narr_order.get(new_narr, 0) < _narr_order.get(old_narr, 0):
                    narrative_downgrades += 1
            self._derive_behavior_mapping(imp)
        self._recalc_global_stats()
        return {
            "decayed": decayed_count,
            "narrative_downgrades": narrative_downgrades,
            "total_tracked": len(self._impressions),
        }

    def get_behavior_params(self, user_id: str) -> BehaviorMapping:
        """获取用户的行为映射参数（供回复生成器使用）"""
        uid_key = self._hash_user_id(user_id)
        imp = self._impressions.get(uid_key)
        if not imp:
            return BehaviorMapping()
        return imp.behavior_mapping

    def get_impression_summary(self, user_id: str) -> Dict[str, Any]:
        """获取用户印象摘要（供调试面板使用）"""
        uid_key = self._hash_user_id(user_id)
        imp = self._impressions.get(uid_key)
        if not imp:
            return {"exists": False}
        return {
            "exists": True,
            "display_name": imp.subjective_name.display_name,
            "nickname": imp.subjective_name.internal_nickname,
            "tags": imp.subjective_name.tags,
            "narrative_type": imp.relation_narrative.narrative_type.value,
            "affection": round(imp.raw_affection, 1),
            "trust": round(imp.raw_trust, 1),
            "trust_value": round(imp.raw_trust, 1),
            "familiarity": round(imp.raw_familiarity, 1),
            "interactions": imp.interaction_count,
            "primary_rule": imp.behavior_mapping.primary_rule.value,
            "positive_events": imp.positive_events,
            "negative_events": imp.negative_events,
            "days_since_contact": (
                round((time.time() - imp.last_direct_interaction) / 86400.0, 1)
                if imp.last_direct_interaction > 0
                else -1.0
            ),
        }

    # ──── 内部方法 ────

    def _hash_user_id(self, user_id: str) -> str:
        return hashlib.md5(str(user_id).encode()).hexdigest()[:12]

    def _evolve_tags_from_interaction(
        self, imp: UserImpression, data: Dict[str, Any]
    ) -> None:
        """从交互数据中演化标签"""
        sent = float(data.get("sentiment", 0.0) or 0.0)
        msg_len = int(data.get("message_length", 0) or 0)
        mentioned = bool(data.get("mentioned_bot", False))
        emoji_n = int(data.get("emoji_used", 0) or 0)
        teasing = bool(data.get("was_teasing", False))
        helpful = bool(data.get("was_helpful", False))
        hostile = bool(data.get("was_hostile", False))
        tc = imp.subjective_name.tag_confidence
        tags = imp.subjective_name.tags
        boost = 0.06
        if sent > 0.5 and helpful:
            tc["helpful"] = tc.get("helpful", 0.0) + boost
            if "helpful" not in tags:
                tags.append("helpful")
        if sent > 0.4 and emoji_n >= 2:
            tc["expressive"] = tc.get("expressive", 0.0) + boost
            if "expressive" not in tags:
                tags.append("expressive")
        if teasing and sent > -0.1:
            tc["playful"] = tc.get("playful", 0.0) + boost
            if "playful" not in tags:
                tags.append("playful")
            imp.behavior_mapping.teasing_allowed = True
        if hostile or sent < -0.4:
            tc["hostile"] = tc.get("hostile", 0.0) + boost * 1.5
            if "hostile" not in tags:
                tags.append("hostile")
        if mentioned and sent > 0.0:
            tc["attentive_to_bot"] = tc.get("attentive_to_bot", 0.0) + boost
            if "attentive_to_bot" not in tags:
                tags.append("attentive_to_bot")
        if msg_len > 80:
            tc["talkative"] = tc.get("talkative", 0.0) + boost * 0.5
            if "talkative" not in tags:
                tags.append("talkative")
        for t in list(tc.keys()):
            tc[t] = min(1.0, tc[t])

    def _update_narrative_type(self, imp: UserImpression) -> None:
        """根据当前指标自动推导叙事类型"""
        aff = imp.raw_affection
        tr = imp.raw_trust
        fam = imp.raw_familiarity
        pos_ratio = imp.positive_events / max(
            1, imp.positive_events + imp.negative_events
        )
        neg_ratio = imp.negative_events / max(
            1, imp.positive_events + imp.negative_events
        )
        conflict_rate = imp.relation_narrative.conflict_count / max(
            1, imp.interaction_count
        )
        current = imp.relation_narrative.narrative_type
        candidates: List[Tuple[RelationNarrativeType, float]] = []
        if aff > 78 and tr > 70 and fam > 60 and pos_ratio > 0.82:
            candidates.append(
                (
                    RelationNarrativeType.CONFIDANT,
                    aff * 0.35 + tr * 0.30 + fam * 0.20 + pos_ratio * 15.0,
                )
            )
        if aff > 65 and tr > 55 and fam > 45 and pos_ratio > 0.70:
            candidates.append(
                (
                    RelationNarrativeType.CLOSE_FRIEND,
                    aff * 0.32 + tr * 0.28 + fam * 0.22 + pos_ratio * 12.0,
                )
            )
        if aff > 48 and tr > 40 and fam > 25 and pos_ratio > 0.55:
            candidates.append(
                (
                    RelationNarrativeType.FRIEND,
                    aff * 0.28 + tr * 0.25 + fam * 0.25 + pos_ratio * 10.0,
                )
            )
        if aff > 55 and tr > 60 and imp.interaction_count >= 10:
            candidates.append(
                (
                    RelationNarrativeType.MENTOR,
                    tr * 0.38 + aff * 0.20 + fam * 0.15 + pos_ratio * 8.0,
                )
            )
        if aff < 20 and neg_ratio > 0.55 and conflict_rate > 0.25:
            candidates.append(
                (
                    RelationNarrativeType.ANNOYING,
                    neg_ratio * 30.0 + conflict_rate * 20.0 - aff * 0.1,
                )
            )
        if aff < 15 and tr < 18 and neg_ratio > 0.65:
            candidates.append(
                (
                    RelationNarrativeType.IGNORED,
                    neg_ratio * 35.0 - tr * 0.1 - aff * 0.05,
                )
            )
        if aff > 42 and tr < 35 and conflict_rate > 0.15:
            candidates.append(
                (
                    RelationNarrativeType.RIVAL,
                    aff * 0.15 + conflict_rate * 25.0 - tr * 0.2,
                )
            )
        if not candidates:
            if fam > 15:
                candidates.append(
                    (
                        RelationNarrativeType.ACQUAINTANCE,
                        fam * 0.4 + aff * 0.15 + tr * 0.1,
                    )
                )
            else:
                candidates.append(
                    (
                        RelationNarrativeType.STRANGER,
                        max(0.01, 10.0 - fam * 0.3 - aff * 0.05),
                    )
                )
        best_type, _best_score = max(candidates, key=lambda x: x[1])
        if best_type != current:
            imp.relation_narrative.narrative_type = best_type
            imp.relation_narrative.relationship_age_days = (
                time.time() - (imp.last_direct_interaction or time.time())
            ) / 86400.0

    def _derive_behavior_mapping(self, imp: UserImpression) -> None:
        """从印象数据自动推导行为映射参数"""
        aff = imp.raw_affection
        tr = imp.raw_trust
        fam = imp.raw_familiarity
        pos_r = imp.positive_events / max(
            1, imp.positive_events + imp.negative_events
        )
        neg_r = imp.negative_events / max(
            1, imp.positive_events + imp.negative_events
        )
        bm = imp.behavior_mapping
        # 主规则推导
        if aff > 72 and tr > 65 and pos_r > 0.78:
            bm.primary_rule = BehaviorMappingRule.WARM
            bm.secondary_rules = [
                BehaviorMappingRule.PLAYFUL,
                BehaviorMappingRule.PROTECTIVE,
            ]
            bm.reply_length_bias = 1.15
            bm.emoji_usage_bias = 1.30
            bm.formality_level = 0.25
            bm.teasing_allowed = True
            bm.sarcasm_allowed = True
            bm.personal_topic_depth = 0.75
            bm.response_delay_modifier = 0.85
            bm.interrupt_tolerance = 0.80
        elif aff > 52 and tr > 45 and pos_r > 0.60:
            bm.primary_rule = BehaviorMappingRule.NORMAL
            bm.secondary_rules = [BehaviorMappingRule.RESPECTFUL]
            bm.reply_length_bias = 1.0
            bm.emoji_usage_bias = 1.1
            bm.formality_level = 0.40
            bm.teasing_allowed = "playful" in imp.subjective_name.tags
            bm.sarcasm_allowed = False
            bm.personal_topic_depth = 0.55
            bm.response_delay_modifier = 1.0
            bm.interrupt_tolerance = 0.55
        elif aff > 30 and tr > 25:
            bm.primary_rule = BehaviorMappingRule.CAUTIOUS
            bm.secondary_rules = []
            bm.reply_length_bias = 0.80
            bm.emoji_usage_bias = 0.75
            bm.formality_level = 0.58
            bm.teasing_allowed = False
            bm.sarcasm_allowed = False
            bm.personal_topic_depth = 0.30
            bm.response_delay_modifier = 1.2
            bm.interrupt_tolerance = 0.30
        elif aff < 18 or (neg_r > 0.55 and tr < 22):
            bm.primary_rule = (
                BehaviorMappingRule.DRY
                if aff >= 10
                else BehaviorMappingRule.AVOIDANT
            )
            bm.secondary_rules = []
            bm.reply_length_bias = 0.45 if aff >= 10 else 0.20
            bm.emoji_usage_bias = 0.35
            bm.formality_level = 0.72
            bm.teasing_allowed = False
            bm.sarcasm_allowed = False
            bm.personal_topic_depth = 0.10
            bm.response_delay_modifier = 1.5
            bm.interrupt_tolerance = 0.12
        else:
            narr = imp.relation_narrative.narrative_type
            if narr == RelationNarrativeType.MENTOR:
                bm.primary_rule = BehaviorMappingRule.RESPECTFUL
                bm.formality_level = 0.55
            elif narr == RelationNarrativeType.RIVAL:
                bm.primary_rule = BehaviorMappingRule.TEASING
                bm.teasing_allowed = True
                bm.sarcasm_allowed = True
            elif narr == RelationNarrativeType.ANNOYING:
                bm.primary_rule = BehaviorMappingRule.DRY
                bm.reply_length_bias = 0.60
            else:
                bm.primary_rule = BehaviorMappingRule.NORMAL
        # 熟悉度修正
        if fam > 60:
            bm.personal_topic_depth = min(0.90, bm.personal_topic_depth + 0.10)

    def _recalc_global_stats(self) -> None:
        """重新计算全局统计"""
        if not self._impressions:
            return
        total_aff = sum(i.raw_affection for i in self._impressions.values())
        self._global_stats["avg_affection"] = total_aff / len(
            self._impressions
        )
        narr_dist: Dict[str, int] = {}
        for i in self._impressions.values():
            nv = i.relation_narrative.narrative_type.value
            narr_dist[nv] = narr_dist.get(nv, 0) + 1
        self._global_stats["narrative_distribution"] = narr_dist


# ──────────────────────────────────────────────
#  全局单例（线程安全，基于统一单例工具）
# ──────────────────────────────────────────────

_iie_instance: Optional[IndividualImpressionEngine] = None
_iie_lock = _get_class_lock("IndividualImpressionEngine")


def get_impression_engine() -> IndividualImpressionEngine:
    global _iie_instance
    if _iie_instance is None:
        with _iie_lock:
            if _iie_instance is None:
                _iie_instance = IndividualImpressionEngine()
                logger.info(
                    "[印象引擎] 个体化印象演化初始化完成 — 三层模型+三路径更新"
                )
    return _iie_instance

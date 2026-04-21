import time
import math
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from src.common.logger import get_logger

logger = get_logger("freshness_decay")

_decay_instances: Dict[str, "FreshnessDecayEngine"] = {}

# ═══════════════════════════════════════════════════
#  衰减层级枚举
# ═══════════════════════════════════════════════════


class DecaySpeedTier(Enum):
    """四层衰减速度"""

    FAST_DECAY = "fast_decay"
    MEDIUM_DECAY = "medium_decay"
    SLOW_DECAY = "slow_decay"
    PRESERVED = "preserved"

    def half_life_sec(self) -> float:
        return {
            "fast_decay": 60.0,
            "medium_decay": 180.0,
            "slow_decay": 600.0,
            "preserved": 3600.0,
        }.get(self.value, 180.0)

    def label(self) -> str:
        return {
            "fast_decay": "快速衰减",
            "medium_decay": "中速衰减",
            "slow_decay": "慢速衰减",
            "preserved": "保鲜锁定",
        }.get(self.value, "")


class RecallTriggerType(Enum):
    """四种回看触发类型"""

    TOPIC_ACTIVATION = "topic_activation"
    USER_RETURN = "user_return"
    EMOTION_SPIKE = "emotion_spike"
    BOREDOM_DRIFT = "boredom_drift"

    def label(self) -> str:
        return {
            "topic_activation": "话题激活",
            "user_return": "用户回归",
            "emotion_spike": "情绪激活",
            "boredom_drift": "无聊漫游",
        }.get(self.value, "")


class VisibilityHistoryEffect(Enum):
    """可见性历史对后续行为的影响差异"""

    NEVER_SEEN = "never_seen"
    GLANCED_BUT_IGNORED = "glanced_but_ignored"
    SEEN_AND_NOTICED = "seen_and_noticed"
    UNDERSTOOD_DEFERRED = "understood_deferred"
    UNDERSTOOD_IGNORED = "understood_ignored"

    def memory_impact_weight(self) -> float:
        return {
            "never_seen": 0.05,
            "glanced_but_ignored": 0.15,
            "seen_and_noticed": 0.40,
            "understood_deferred": 0.65,
            "understood_ignored": 0.80,
        }.get(self.value, 0.30)

    def relation_impact_weight(self) -> float:
        return {
            "never_seen": 0.02,
            "glanced_but_ignored": 0.08,
            "seen_and_noticed": 0.25,
            "understood_deferred": 0.45,
            "understood_ignored": 0.60,
        }.get(self.value, 0.20)

    def future_recall_probability(self) -> float:
        return {
            "never_seen": 0.03,
            "glanced_but_ignored": 0.10,
            "seen_and_noticed": 0.30,
            "understood_deferred": 0.55,
            "understood_ignored": 0.70,
        }.get(self.value, 0.20)


@dataclass
class FreshnessRecord:
    """单条消息的新鲜度记录（扩展版）"""

    message_id: str = ""
    user_id: str = ""
    content_fingerprint: str = ""
    original_freshness: float = 1.0
    current_freshness: float = 1.0
    decay_tier: DecaySpeedTier = DecaySpeedTier.MEDIUM_DECAY
    visibility_history: VisibilityHistoryEffect = (
        VisibilityHistoryEffect.SEEN_AND_NOTICED
    )
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    reactivation_count: int = 0
    last_reactivation_trigger: Optional[RecallTriggerType] = None
    reactivation_reasons: List[str] = field(default_factory=list)
    topic_tags: List[str] = field(default_factory=list)
    emotion_tag: str = ""
    is_stale: bool = False

    def age_sec(self) -> float:
        return time.time() - self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id[:12],
            "freshness": round(self.current_freshness, 3),
            "decay_tier": self.decay_tier.label(),
            "visibility_history": self.visibility_history.value,
            "age_sec": round(self.age_sec(), 0),
            "reactivation_count": self.reactivation_count,
            "is_stale": self.is_stale,
        }


@dataclass
class RecallDecision:
    """回看决策结果"""

    message_id: str = ""
    should_recall: bool = False
    recall_priority: float = 0.0
    trigger_type: Optional[RecallTriggerType] = None
    reason: str = ""
    new_freshness: float = 0.0
    old_freshness: float = 0.0
    action_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id[:12],
            "should_recall": self.should_recall,
            "priority": round(self.recall_priority, 3),
            "trigger": (
                self.trigger_type.label() if self.trigger_type else ""
            ),
            "reason": self.reason[:60],
            "freshness_change": f"{self.old_freshness:.2f}→{self.new_freshness:.2f}",
            "action_hint": self.action_hint,
        }


@dataclass
class BatchRecallResult:
    """批量回看结果"""

    decisions: List[RecallDecision] = field(default_factory=list)
    total_evaluated: int = 0
    recall_count: int = 0
    deferred_count: int = 0
    ignored_count: int = 0
    triggered_by: Dict[str, int] = field(default_factory=dict)
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evaluated": self.total_evaluated,
            "recall_count": self.recall_count,
            "deferred_count": self.deferred_count,
            "ignored_count": self.ignored_count,
            "trigger_distribution": dict(self.triggered_by),
            "top_decisions": [
                d.to_dict()
                for d in sorted(
                    self.decisions,
                    key=lambda x: x.recall_priority,
                    reverse=True,
                )[:5]
            ],
        }


# 全局常量
_STALE_THRESHOLD = 0.12
_REACTIVATION_COOLDOWN_SEC = 90.0
_MAX_RECORDS = 300
_PRUNE_INTERVAL_SEC = 150.0
_MIN_RECALL_PRIORITY = 0.18
_BOOST_CAP = 0.55


class FreshnessDecayEngine:
    """心理新鲜度衰减与回看决策引擎
    实现：
    1. 四层衰减速度：快速(扫到不重要)/中速(看见未处理)/慢速(纳入关注)/保鲜(高相关或情绪关联)
    2. 四种回看触发：话题匹配/用户回归/情绪波动/无聊漫游
    3. 可见性历史差异化："没看见"≠"看见但不理" 对记忆和关系的影响完全不同
    4. 回看后行为建议：直接回忆/再次观看/进入当前关注范围
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._records: Dict[str, FreshnessRecord] = {}
        self._cooldown_map: Dict[str, float] = {}
        self._last_prune_ts = time.time()
        self._global_topic_buffer: List[str] = []
        self._pending_user_set: Set[str] = set()
        self._current_emotion_intensity = 0.0
        self._current_boredom = 0.0
        self._stats = {
            "evaluations": 0,
            "recalls_triggered": 0,
            "decays_applied": 0,
        }

    # ────────────────── 消息注册 ──────────────────

    def register_message(
        self,
        message_id: str,
        user_id: str = "",
        content_fingerprint: str = "",
        *,
        initial_tier: DecaySpeedTier = DecaySpeedTier.MEDIUM_DECAY,
        visibility: VisibilityHistoryEffect = VisibilityHistoryEffect.SEEN_AND_NOTICED,
        topic_tags: Optional[List[str]] = None,
        emotion_tag: str = "",
    ) -> FreshnessRecord:
        now = time.time()
        record = FreshnessRecord(
            message_id=message_id,
            user_id=user_id,
            content_fingerprint=content_fingerprint,
            original_freshness=1.0,
            current_freshness=1.0,
            decay_tier=initial_tier,
            visibility_history=visibility,
            created_at=now,
            last_accessed_at=now,
            topic_tags=topic_tags or [],
            emotion_tag=emotion_tag,
        )
        self._records[message_id] = record
        self._maybe_prune(now)
        return record

    def update_visibility_history(
        self, message_id: str, effect: VisibilityHistoryEffect
    ) -> bool:
        record = self._records.get(message_id)
        if record is None:
            return False
        old_effect = record.visibility_history
        record.visibility_history = effect
        if effect.value != old_effect.value:
            logger.debug(
                f"[{self._channel_id}] 消息 {message_id[:12]} "
                f"可见性历史更新: {old_effect.value} → {effect.value}"
            )
        return True

    def upgrade_decay_tier(
        self, message_id: str, new_tier: DecaySpeedTier
    ) -> bool:
        record = self._records.get(message_id)
        if record is None:
            return False
        tier_order = [
            DecaySpeedTier.FAST_DECAY,
            DecaySpeedTier.MEDIUM_DECAY,
            DecaySpeedTier.SLOW_DECAY,
            DecaySpeedTier.PRESERVED,
        ]
        old_idx = (
            tier_order.index(record.decay_tier)
            if record.decay_tier in tier_order
            else 1
        )
        new_idx = tier_order.index(new_tier) if new_tier in tier_order else 1
        if new_idx <= old_idx:
            return False
        record.decay_tier = new_tier
        logger.debug(
            f"[{self._channel_id}] 消息 {message_id[:12]} 衰减层级升级 → {new_tier.label()}"
        )
        return True

    # ────────────────── 新鲜度查询 ──────────────────

    def get_freshness(self, message_id: str) -> float:
        record = self._records.get(message_id)
        if record is None:
            return 0.0
        self._apply_decay(record)
        return record.current_freshness

    def get_freshness_batch(self, message_ids: List[str]) -> Dict[str, float]:
        return {mid: self.get_freshness(mid) for mid in message_ids}

    def fresh_messages(self, threshold: float = 0.25) -> List[FreshnessRecord]:
        now = time.time()
        result = []
        for record in self._records.values():
            self._apply_decay(record, now)
            if record.current_freshness >= threshold and not record.is_stale:
                result.append(record)
        result.sort(key=lambda r: r.current_freshness, reverse=True)
        return result

    def stale_messages(self) -> List[FreshnessRecord]:
        now = time.time()
        stale = []
        for record in self._records.values():
            self._apply_decay(record, now)
            if record.is_stale or record.current_freshness < _STALE_THRESHOLD:
                stale.append(record)
        return stale

    def all_records_snapshot(self) -> Dict[str, Any]:
        now = time.time()
        total = len(self._records)
        fresh_count = 0
        stale_count = 0
        tier_dist: Dict[str, int] = {}
        vis_dist: Dict[str, int] = {}
        for record in self._records.values():
            self._apply_decay(record, now)
            if record.current_freshness >= _STALE_THRESHOLD:
                fresh_count += 1
            else:
                stale_count += 1
            t = record.decay_tier.value
            tier_dist[t] = tier_dist.get(t, 0) + 1
            v = record.visibility_history.value
            vis_dist[v] = vis_dist.get(v, 0) + 1
        return {
            "total_tracked": total,
            "fresh_count": fresh_count,
            "stale_count": stale_count,
            "tier_distribution": tier_dist,
            "visibility_distribution": vis_dist,
            "stats": dict(self._stats),
        }

    # ────────────────── 全局状态设置 ──────────────────

    def set_current_topics(self, topics: List[str]) -> None:
        self._global_topic_buffer = list(topics)[:10]

    def add_pending_user(self, user_id: str) -> None:
        self._pending_user_set.add(user_id)

    def clear_pending_users(self) -> None:
        self._pending_user_set.clear()

    def set_emotion_state(self, intensity: float) -> None:
        self._current_emotion_intensity = max(0.0, min(1.0, intensity))

    def set_boredom_level(self, boredom: float) -> None:
        self._current_boredom = max(0.0, min(1.0, boredom))

    # ────────────────── 回看触发评估 ──────────────────

    def evaluate_topic_activation(self) -> List[RecallDecision]:
        """T1 触发：当前消息触发了对旧消息的话题回想"""
        decisions = []
        if not self._global_topic_buffer:
            return decisions
        stale = self.stale_messages()
        for record in stale[:20]:
            if self._in_cooldown(record.message_id):
                continue
            matched_topics = []
            for topic in self._global_topic_buffer:
                fp = record.content_fingerprint
                topic_hash = str(hash(topic[:40]))
                if fp and topic_hash[:5] in fp:
                    matched_topics.append(topic)
            if not matched_topics:
                continue
            old_fresh = record.current_freshness
            boost = min(_BOOST_CAP, 0.25 + len(matched_topics) * 0.08)
            success = self._reactivate_record(
                record,
                boost,
                RecallTriggerType.TOPIC_ACTIVATION,
                f"话题「{matched_topics[0][:16]}」匹配",
            )
            if success:
                decisions.append(
                    RecallDecision(
                        message_id=record.message_id,
                        should_recall=True,
                        recall_priority=min(1.0, boost * 2.0 + 0.15),
                        trigger_type=RecallTriggerType.TOPIC_ACTIVATION,
                        reason=f"话题匹配({len(matched_topics)}个): {matched_topics[0][:20]}",
                        new_freshness=record.current_freshness,
                        old_freshness=old_fresh,
                        action_hint=self._derive_action_hint(record),
                    )
                )
        self._stats["recalls_triggered"] += len(decisions)
        return decisions

    def evaluate_user_return(
        self, returning_user_id: str
    ) -> List[RecallDecision]:
        """T2 触发：沉默用户重新出现时激活其旧消息"""
        decisions = []
        if not returning_user_id:
            return decisions
        activated = 0
        for record in self._records.values():
            if record.user_id != returning_user_id:
                continue
            if self._in_cooldown(record.message_id):
                continue
            old_fresh = record.current_freshness
            boost = 0.32
            success = self._reactivate_record(
                record,
                boost,
                RecallTriggerType.USER_RETURN,
                f"用户{returning_user_id[:8]}回归",
            )
            if success and activated < 5:
                decisions.append(
                    RecallDecision(
                        message_id=record.message_id,
                        should_recall=True,
                        recall_priority=0.55 + (0.05 * (4 - activated)),
                        trigger_type=RecallTriggerType.USER_RETURN,
                        reason=f"用户 {returning_user_id[:8]} 重新出现，激活旧消息",
                        new_freshness=record.current_freshness,
                        old_freshness=old_fresh,
                        action_hint=self._derive_action_hint(record),
                    )
                )
                activated += 1
        self._stats["recalls_triggered"] += len(decisions)
        return decisions

    def evaluate_emotion_spike(
        self, intensity: Optional[float] = None
    ) -> List[RecallDecision]:
        """T3 触发：情绪波动大时回想相关旧消息"""
        emo = intensity or self._current_emotion_intensity
        decisions = []
        if emo < 0.55:
            return decisions
        stale = self.stale_messages()
        boosted = 0
        for record in stale[:12]:
            if self._in_cooldown(record.message_id):
                continue
            if record.emotion_tag and boosted < 4:
                old_fresh = record.current_freshness
                boost = min(_BOOST_CAP, emo * 0.35)
                success = self._reactivate_record(
                    record,
                    boost,
                    RecallTriggerType.EMOTION_SPIKE,
                    f"情绪强度{emo:.2f}+标签{record.emotion_tag}",
                )
                if success:
                    decisions.append(
                        RecallDecision(
                            message_id=record.message_id,
                            should_recall=True,
                            recall_priority=emo * 0.65 + 0.10 * (3 - boosted),
                            trigger_type=RecallTriggerType.EMOTION_SPIKE,
                            reason=f"情绪波动(强度={emo:.2f})+情感标签匹配",
                            new_freshness=record.current_freshness,
                            old_freshness=old_fresh,
                            action_hint=self._derive_action_hint(record),
                        )
                    )
                    boosted += 1
        self._stats["recalls_triggered"] += len(decisions)
        return decisions

    def evaluate_boredom_drift(
        self, boredom: Optional[float] = None
    ) -> List[RecallDecision]:
        """T4 触发：无聊时随机回想旧消息"""
        b = boredom or self._current_boredom
        decisions = []
        if b < 0.50:
            return decisions
        stale = self.stale_messages()
        if not stale:
            return decisions
        pick_n = 1 if b < 0.7 else (2 if b < 0.85 else 3)
        candidates = random.sample(stale, min(pick_n, len(stale)))
        for record in candidates:
            if self._in_cooldown(record.message_id):
                continue
            old_fresh = record.current_freshness
            boost = 0.22 + b * 0.18
            success = self._reactivate_record(
                record,
                boost,
                RecallTriggerType.BOREDOM_DRIFT,
                f"无聊值{
                    b:.2f}漫游",
            )
            if success:
                decisions.append(
                    RecallDecision(
                        message_id=record.message_id,
                        should_recall=b > 0.65,
                        recall_priority=b * 0.42,
                        trigger_type=RecallTriggerType.BOREDOM_DRIFT,
                        reason=f"无聊漫游(值={b:.2f})，随机翻找旧消息",
                        new_freshness=record.current_freshness,
                        old_freshness=old_fresh,
                        action_hint=(
                            "随意翻看" if b < 0.75 else "找点有趣的内容看看"
                        ),
                    )
                )
        self._stats["recalls_triggered"] += len(decisions)
        return decisions

    def evaluate_all_triggers(self) -> BatchRecallResult:
        """一次性评估所有触发类型"""
        self._stats["evaluations"] += 1
        all_decisions: List[RecallDecision] = []
        trigger_counts: Dict[str, int] = {}
        t1 = self.evaluate_topic_activation()
        all_decisions.extend(t1)
        trigger_counts["topic_activation"] = len(t1)
        if self._pending_user_set:
            for uid in list(self._pending_user_set)[:3]:
                t2 = self.evaluate_user_return(uid)
                all_decisions.extend(t2)
            trigger_counts["user_return"] = sum(
                1
                for d in all_decisions
                if d.trigger_type == RecallTriggerType.USER_RETURN
            )
        t3 = self.evaluate_emotion_spike()
        all_decisions.extend(t3)
        trigger_counts["emotion_spike"] = len(t3)
        t4 = self.evaluate_boredom_drift()
        all_decisions.extend(t4)
        trigger_counts["boredom_drift"] = len(t4)
        recall_c = sum(1 for d in all_decisions if d.should_recall)
        defer_c = sum(
            1
            for d in all_decisions
            if not d.should_recall and d.recall_priority > _MIN_RECALL_PRIORITY
        )
        ignore_c = len(all_decisions) - recall_c - defer_c
        return BatchRecallResult(
            decisions=all_decisions,
            total_evaluated=len(all_decisions),
            recall_count=recall_c,
            deferred_count=defer_c,
            ignored_count=ignore_c,
            triggered_by=trigger_counts,
        )

    # ────────────────── 可见性差异化处理 ──────────────────

    def compute_visibility_differential_impact(
        self, message_id: str
    ) -> Dict[str, float]:
        """计算某条消息的可见性历史对记忆和关系的差异化影响"""
        record = self._records.get(message_id)
        if record is None:
            return {"memory": 0.0, "relation": 0.0, "future_recall": 0.0}
        vis = record.visibility_history
        freshness = max(0.0, record.current_freshness)
        base_memory = vis.memory_impact_weight() * freshness
        base_relation = vis.relation_impact_weight() * freshness
        base_future = vis.future_recall_probability() * freshness
        if vis == VisibilityHistoryEffect.NEVER_SEEN:
            return {
                "memory": round(base_memory, 3),
                "relation": round(base_relation, 3),
                "future_recall": round(base_future, 3),
                "note": "未进入主观注意力，几乎不留痕迹",
            }
        elif vis == VisibilityHistoryEffect.UNDERSTOOD_IGNORED:
            return {
                "memory": round(base_memory, 3),
                "relation": round(base_relation, 3),
                "future_recall": round(base_future, 3),
                "note": "看过且故意不理，形成连续感，未来容易想起",
            }
        elif vis == VisibilityHistoryEffect.UNDERSTOOD_DEFERRED:
            return {
                "memory": round(base_memory, 3),
                "relation": round(base_relation, 3),
                "future_recall": round(base_future, 3),
                "note": "理解但暂存，保留较高回想概率",
            }
        elif vis == VisibilityHistoryEffect.GLANCED_BUT_IGNORED:
            return {
                "memory": round(base_memory, 3),
                "relation": round(base_relation, 3),
                "future_recall": round(base_future, 3),
                "note": "仅扫到，留浅层印象",
            }
        return {
            "memory": round(base_memory, 3),
            "relation": round(base_relation, 3),
            "future_recall": round(base_future, 3),
            "note": "正常关注级别",
        }

    # ────────────────── 内部方法 ──────────────────

    def _apply_decay(
        self, record: FreshnessRecord, now: Optional[float] = None
    ) -> None:
        current = now or time.time()
        elapsed = current - record.last_accessed_at
        if elapsed <= 0:
            return
        self._stats["decays_applied"] += 1
        half_life = record.decay_tier.half_life_sec()
        decay_factor = math.pow(0.5, elapsed / half_life)
        record.current_freshness *= decay_factor
        record.current_freshness = max(0.0, min(1.0, record.current_freshness))
        record.last_accessed_at = current
        record.is_stale = record.current_freshness < _STALE_THRESHOLD

    def _reactivate_record(
        self,
        record: FreshnessRecord,
        boost: float,
        trigger: RecallTriggerType,
        reason: str,
    ) -> bool:
        self._apply_decay(record)
        old = record.current_freshness
        record.current_freshness = min(1.0, old + boost)
        record.reactivation_count += 1
        record.last_reactivation_trigger = trigger
        record.reactivation_reasons.append(reason[:80])
        if len(record.reactivation_reasons) > 8:
            record.reactivation_reasons = record.reactivation_reasons[-8:]
        self._mark_cooldown(record.message_id)
        logger.debug(
            f"[{self._channel_id}] 回看激活: {record.message_id[:12]} "
            f"{old:.3f}→{record.current_freshness:.3f} "
            f"[{trigger.label()}] {reason[:40]}"
        )
        return True

    def _in_cooldown(self, message_id: str) -> bool:
        ts = self._cooldown_map.get(message_id, 0.0)
        return (time.time() - ts) < _REACTIVATION_COOLDOWN_SEC

    def _mark_cooldown(self, message_id: str) -> None:
        self._cooldown_map[message_id] = time.time()
        if len(self._cooldown_map) > 200:
            cutoff = time.time() - _REACTIVATION_COOLDOWN_SEC
            self._cooldown_map = {
                k: v for k, v in self._cooldown_map.items() if v > cutoff
            }

    @staticmethod
    def _derive_action_hint(record: FreshnessRecord) -> str:
        vis = record.visibility_history
        fresh = record.current_freshness
        if vis == VisibilityHistoryEffect.NEVER_SEEN:
            return "首次注意，快速浏览内容"
        if vis == VisibilityHistoryEffect.UNDERSTOOD_IGNORED:
            return "之前故意没理，现在再想想要不要接"
        if vis == VisibilityHistoryEffect.UNDERSTOOD_DEFERRED:
            return "之前存着的，现在可以拿出来聊了"
        if vis == VisibilityHistoryEffect.GLANCED_BUT_IGNORED:
            return "之前只扫了一眼，现在认真看看"
        if fresh > 0.6:
            return "还比较新鲜，可以直接回忆"
        if fresh > 0.35:
            return "有点旧了，但还能想起来"
        return "已经很模糊了，大概有个印象"

    def _maybe_prune(self, now: float) -> None:
        if now - self._last_prune_ts < _PRUNE_INTERVAL_SEC:
            return
        self._last_prune_ts = now
        if len(self._records) <= _MAX_RECORDS:
            return
        sorted_items = sorted(
            self._records.items(),
            key=lambda kv: kv[1].current_freshness,
            reverse=True,
        )
        keep = sorted_items[:_MAX_RECORDS]
        removed = len(self._records) - len(keep)
        self._records = dict(keep)
        if removed > 0:
            logger.debug(
                f"[{self._channel_id}] 新鲜度记录清理: 移除 {removed} 条过期记录"
            )


def get_freshness_decay_engine(channel_id: str) -> FreshnessDecayEngine:
    if channel_id not in _decay_instances:
        _decay_instances[channel_id] = FreshnessDecayEngine(channel_id)
    return _decay_instances[channel_id]


def remove_freshness_decay_engine(channel_id: str) -> None:
    _decay_instances.pop(channel_id, None)

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("mem_reactivation")

_gate_instances: Dict[str, "MemoryReactivationGate"] = {}


class ReactivationTrigger(Enum):
    """重新激活触发类型"""

    TOPIC_MATCH = "topic_match"
    USER_RETURN = "user_return"
    EMOTION_SPIKE = "emotion_spike"
    BOREDOM_DRIFT = "boredom_drift"
    TIME_ECHO = "time_echo"
    KEYWORD_HIT = "keyword_hit"
    NONE = "none"

    def label(self) -> str:
        _labels = {
            "topic_match": "话题匹配",
            "user_return": "用户回归",
            "emotion_spike": "情绪激活",
            "boredom_drift": "无聊漫游",
            "time_echo": "时间回响",
            "keyword_hit": "关键词命中",
            "none": "无触发",
        }
        return _labels.get(self.value, "无触发")


@dataclass
class ReactivationCandidate:
    """待激活消息候选"""

    message_id: str = ""
    user_id: str = ""
    trigger: ReactivationTrigger = ReactivationTrigger.NONE
    relevance_score: float = 0.0
    freshness_before: float = 0.0
    freshness_after: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "trigger": self.trigger.value,
            "trigger_label": self.trigger.label(),
            "relevance_score": round(self.relevance_score, 3),
            "freshness_before": round(self.freshness_before, 3),
            "freshness_after": round(self.freshness_after, 3),
            "reason": self.reason,
        }


class MemoryReactivationGate:
    """记忆重新激活门控
    根据当前上下文评估是否需要激活旧消息记忆，
    让主体具备"回想"能力而非只处理最新消息。
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._recent_activations: List[ReactivationCandidate] = []
        self._max_cache = 30
        # 激活冷却：同一条消息在冷却期内不重复激活
        self._cooldown_map: Dict[str, float] = {}
        self._cooldown_sec = 120.0

    def evaluate_topic_trigger(
        self,
        current_topics: List[str],
        time_stream: Any,
    ) -> List[ReactivationCandidate]:
        """话题匹配触发：当前话题与旧消息内容关联时激活"""
        candidates: List[ReactivationCandidate] = []
        if not current_topics or time_stream is None:
            return candidates
        stale_ids = time_stream.stale_message_ids()
        for mid in stale_ids[:20]:
            if self._in_cooldown(mid):
                continue
            record = time_stream._records.get(mid)
            if record is None:
                continue
            old_freshness = time_stream.get_freshness(mid)
            # 简化的话题匹配：检查内容哈希与话题是否有交集
            matched = False
            for topic in current_topics:
                topic_hash = str(hash(topic[:50]))
                if (
                    record.content_hash
                    and topic_hash[:4] in record.content_hash
                ):
                    matched = True
                    break
            if matched:
                time_stream.reactivate(mid, boost=0.35, reason="话题匹配")
                new_freshness = time_stream.get_freshness(mid)
                candidate = ReactivationCandidate(
                    message_id=mid,
                    user_id=record.user_id,
                    trigger=ReactivationTrigger.TOPIC_MATCH,
                    relevance_score=0.6,
                    freshness_before=old_freshness,
                    freshness_after=new_freshness,
                    reason=f"话题「{current_topics[0][:20]}」触发回想",
                )
                candidates.append(candidate)
                self._mark_cooldown(mid)
        self._append_candidates(candidates)
        return candidates

    def evaluate_user_return_trigger(
        self,
        returning_user_id: str,
        time_stream: Any,
    ) -> List[ReactivationCandidate]:
        """用户回归触发：沉默用户重新出现时激活其旧消息"""
        candidates: List[ReactivationCandidate] = []
        if not returning_user_id or time_stream is None:
            return candidates
        count = time_stream.reactivate_by_user(returning_user_id, boost=0.3)
        if count > 0:
            candidate = ReactivationCandidate(
                message_id="",
                user_id=returning_user_id,
                trigger=ReactivationTrigger.USER_RETURN,
                relevance_score=0.5,
                reason=f"用户回归，激活了 {count} 条旧消息",
            )
            candidates.append(candidate)
        self._append_candidates(candidates)
        return candidates

    def evaluate_emotion_trigger(
        self,
        emotion_intensity: float,
        time_stream: Any,
    ) -> List[ReactivationCandidate]:
        """情绪激活触发：情绪波动大时回想相关旧消息"""
        candidates: List[ReactivationCandidate] = []
        if emotion_intensity < 0.6 or time_stream is None:
            return candidates
        # 情绪强度越高，激活幅度越大
        boost = min(0.5, emotion_intensity * 0.4)
        stale_ids = time_stream.stale_message_ids()
        activated = 0
        for mid in stale_ids[:10]:
            if self._in_cooldown(mid):
                continue
            old_fresh = time_stream.get_freshness(mid)
            time_stream.reactivate(mid, boost=boost, reason="情绪激活")
            new_fresh = time_stream.get_freshness(mid)
            candidates.append(
                ReactivationCandidate(
                    message_id=mid,
                    trigger=ReactivationTrigger.EMOTION_SPIKE,
                    relevance_score=emotion_intensity * 0.7,
                    freshness_before=old_fresh,
                    freshness_after=new_fresh,
                    reason=f"情绪强度{emotion_intensity:.2f}触发回想",
                )
            )
            self._mark_cooldown(mid)
            activated += 1
            if activated >= 3:
                break
        self._append_candidates(candidates)
        return candidates

    def evaluate_boredom_trigger(
        self,
        boredom: float,
        time_stream: Any,
    ) -> List[ReactivationCandidate]:
        """无聊漫游触发：无聊时随机回想旧消息"""
        candidates: List[ReactivationCandidate] = []
        if boredom < 0.5 or time_stream is None:
            return candidates
        import random

        stale_ids = time_stream.stale_message_ids()
        if not stale_ids:
            return candidates
        # 无聊度越高，回想概率越高
        pick_count = 1 if boredom < 0.7 else 2
        pick_ids = random.sample(stale_ids, min(pick_count, len(stale_ids)))
        for mid in pick_ids:
            if self._in_cooldown(mid):
                continue
            old_fresh = time_stream.get_freshness(mid)
            time_stream.reactivate(mid, boost=0.25, reason="无聊漫游")
            new_fresh = time_stream.get_freshness(mid)
            candidates.append(
                ReactivationCandidate(
                    message_id=mid,
                    trigger=ReactivationTrigger.BOREDOM_DRIFT,
                    relevance_score=boredom * 0.4,
                    freshness_before=old_fresh,
                    freshness_after=new_fresh,
                    reason=f"无聊值{boredom:.2f}触发随机回想",
                )
            )
            self._mark_cooldown(mid)
        self._append_candidates(candidates)
        return candidates

    def recent_activation_summary(self) -> Dict[str, int]:
        """最近各触发类型的计数"""
        counts: Dict[str, int] = {}
        for c in self._recent_activations[-20:]:
            key = c.trigger.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def evaluate_visibility_aware(
        self,
        time_stream: Any,
        visibility_history: Dict[str, str],
    ) -> List[ReactivationCandidate]:
        """可见性感知的差异化激活
        visibility_history: 消息ID → 可见性层级值 的映射
        "没看见"(received/glanced) 的消息用更高激活幅度；
        "看见但不理"(understood_but_ignored) 的消息只做低幅度标记式激活。
        """
        candidates: List[ReactivationCandidate] = []
        if time_stream is None:
            return candidates
        stale_ids = time_stream.stale_message_ids()
        for mid in stale_ids[:25]:
            if self._in_cooldown(mid):
                continue
            record = time_stream._records.get(mid)
            if record is None:
                continue
            vis_level = visibility_history.get(mid, "received")
            old_freshness = time_stream.get_freshness(mid)
            # 根据可见性历史决定激活策略
            if vis_level in ("received", "glanced"):
                # 没看见的消息：高幅度激活，因为主体从未处理过
                boost = 0.45
                trigger = ReactivationTrigger.TIME_ECHO
                reason = f"未曾注意的消息重新浮现(原层级={vis_level})"
            elif vis_level == "noticed":
                # 注意到但没深入：中等激活
                boost = 0.3
                trigger = ReactivationTrigger.TOPIC_MATCH
                reason = f"曾注意到但未深入的消息(原层级={vis_level})"
            elif vis_level in ("understood_but_ignored", "focused"):
                # 已经看懂但选择不理：低幅度标记式激活
                boost = 0.12
                trigger = ReactivationTrigger.KEYWORD_HIT
                reason = f"曾看懂但未回应的消息(原层级={vis_level})"
            elif vis_level == "deferred":
                # 之前特意延后的消息：中高幅度激活
                boost = 0.38
                trigger = ReactivationTrigger.TIME_ECHO
                reason = "之前延后处理的消息现在触发回看"
            else:
                boost = 0.25
                trigger = ReactivationTrigger.KEYWORD_HIT
                reason = f"重新激活(层级={vis_level})"
            time_stream.reactivate(mid, boost=boost, reason=reason)
            new_freshness = time_stream.get_freshness(mid)
            candidate = ReactivationCandidate(
                message_id=mid,
                user_id=record.user_id,
                trigger=trigger,
                relevance_score=min(1.0, boost * 2),
                freshness_before=old_freshness,
                freshness_after=new_freshness,
                reason=reason,
            )
            candidates.append(candidate)
            self._mark_cooldown(mid)
            if len(candidates) >= 5:
                break
        self._append_candidates(candidates)
        return candidates

    # ────────────────── 内部方法 ──────────────────

    def _in_cooldown(self, message_id: str) -> bool:
        ts = self._cooldown_map.get(message_id, 0.0)
        return (time.time() - ts) < self._cooldown_sec

    def _mark_cooldown(self, message_id: str) -> None:
        self._cooldown_map[message_id] = time.time()
        # 清理过期冷却
        if len(self._cooldown_map) > 100:
            cutoff = time.time() - self._cooldown_sec
            self._cooldown_map = {
                k: v for k, v in self._cooldown_map.items() if v > cutoff
            }

    def _append_candidates(
        self, candidates: List[ReactivationCandidate]
    ) -> None:
        self._recent_activations.extend(candidates)
        if len(self._recent_activations) > self._max_cache:
            self._recent_activations = self._recent_activations[
                -self._max_cache:
            ]


def get_reactivation_gate(channel_id: str) -> MemoryReactivationGate:
    if channel_id not in _gate_instances:
        _gate_instances[channel_id] = MemoryReactivationGate(channel_id)
    return _gate_instances[channel_id]


def remove_reactivation_gate(channel_id: str) -> None:
    _gate_instances.pop(channel_id, None)

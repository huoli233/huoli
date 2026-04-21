import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger("engagement_scorer")


@dataclass
class EngagementResult:
    success: bool
    message_id: str
    engagement_score: float = 0.0
    final_score: float = 0.0
    should_reply: bool = False
    should_act: bool = False
    calculation_time: float = 0.0
    error_message: Optional[str] = None
    decision_reason: str = ""
    semantic_score: float = 0.0
    relationship_score: float = 0.0
    mention_score: float = 0.0
    keyword_score: float = 0.0
    sentiment_score: float = 0.0
    topic_relevance_score: float = 0.0


@dataclass
class ScoringWeights:
    semantic: float = 0.3
    keyword: float = 0.2
    sentiment: float = 0.15
    topic_relevance: float = 0.15
    relationship: float = 0.1
    mention: float = 0.1


@dataclass
class InterestResult:
    score: float = 0.0
    final_decision: str = "neutral"
    reason: str = ""


class EngagementScorer:
    def __init__(self):
        from src.config.config import global_config
        self.enable_model_driven = True
        try:
            from src.config.behavior_tuner import get_behavior_tuner
            _bt = get_behavior_tuner()
        except Exception:
            _bt = None
        _bg = lambda k, d: _bt.get(k, d) if _bt else d
        self.weights = ScoringWeights(
            semantic=_bg("scoring_semantic_weight", 0.3),
            keyword=_bg("scoring_keyword_weight", 0.2),
            sentiment=_bg("scoring_sentiment_weight", 0.15),
            topic_relevance=_bg("scoring_topic_weight", 0.15),
            relationship=_bg("scoring_relationship_weight", 0.1),
            mention=_bg("scoring_mention_weight", 0.1),
        )
        self.reply_threshold = _bg("reply_threshold", global_config.get("interest_threshold", 0.5))
        self.action_threshold = _bg("action_threshold", 0.4)
        self.mention_boost = _bg("mention_boost", 0.5)
        self.keyword_high_threshold = 3
        self.keyword_medium_threshold = 1
        self.sentiment_positive_threshold = _bg("sentiment_positive_threshold", 0.1)
        self.sentiment_negative_threshold = _bg("sentiment_negative_threshold", -0.1)
        self._load_keywords_from_config()
        self.no_reply_count = 0
        self.max_no_reply_count = 5
        self.threshold_decay_per_skip = _bg("threshold_decay_per_skip", 0.05)
        self.post_reply_boost_enabled = True
        self.post_reply_boost_remaining = 0
        self.post_reply_boost_max = _bg("post_reply_boost_max", 3)
        self.post_reply_threshold_reduction = _bg("post_reply_threshold_reduction", 0.1)
        self.post_reply_decay_rate = _bg("post_reply_decay_rate", 0.7)
        self._relationship_cache: Dict[str, float] = {}
        self._cache_ttl = _bg("relationship_cache_ttl", 300)
        self._cache_timestamps: Dict[str, float] = {}
        self._metrics = {"total_calculations": 0, "replies_triggered": 0, "actions_triggered": 0}
        logger.info(
            f"[EngagementScorer] 初始化完成 | 模型驱动:{self.enable_model_driven} | "
            f"回复阈值:{self.reply_threshold:.2f}"
        )

    def calculate_interest(self, content: str) -> InterestResult:
        if not content or not content.strip():
            return InterestResult(score=0.0, final_decision="bored", reason="空消息")
        base_score = min(len(content) / 100, 0.5)
        interesting_keywords = ["有趣", "好玩", "想", "喜欢", "帮", "问", "看看", "图片", "照片"]
        keyword_boost = sum(0.1 for kw in interesting_keywords if kw in content)
        keyword_boost = min(keyword_boost, 0.3)
        score = min(base_score + keyword_boost, 1.0)
        if score <= 0.3:
            final_decision = "bored"
        elif score >= 0.7:
            final_decision = "interested"
        else:
            final_decision = "neutral"
        return InterestResult(score=score, final_decision=final_decision, reason="基于内容长度和关键词分析")

    async def calculate_score(self, message_text: str, is_mentioned: bool = False,
                               relationship_score: float = 0.5, channel_id: str = "",
                               user_id: str = "", **kwargs) -> EngagementResult:
        msg_id = kwargs.get("message_id", str(uuid.uuid4()))
        return await self.calculate(
            msg_id, message_text, user_id,
            is_mentioned=is_mentioned,
            extra_data={"rel_score": relationship_score, "stream_id": channel_id}
        )

    async def calculate(self, message_id: str, content: str, user_id: str,
                        is_mentioned: bool = False, is_private: bool = False,
                        extra_data: Optional[Dict] = None) -> EngagementResult:
        start_time = time.time()
        self._metrics["total_calculations"] += 1
        try:
            semantic_score = await self._calculate_semantic_score(content)
            keyword_score = 0.0
            sentiment_score = 0.0
            topic_relevance_score = 0.0
            rel_from_extra = extra_data.get("rel_score") if extra_data else None
            relationship_score = rel_from_extra if rel_from_extra is not None else await self._calculate_relationship_score(user_id)
            mention_score = self._calculate_mention_score(is_mentioned, is_private)
            raw_score = (
                semantic_score * self.weights.semantic +
                keyword_score * self.weights.keyword +
                sentiment_score * self.weights.sentiment +
                topic_relevance_score * self.weights.topic_relevance +
                relationship_score * self.weights.relationship +
                mention_score * self.weights.mention
            )
            engagement_score = min(raw_score, 1.0)
            adjusted_reply_threshold, adjusted_action_threshold = self._apply_threshold_adjustment()
            should_reply = True
            should_act = engagement_score >= adjusted_action_threshold
            if should_reply:
                self._metrics["replies_triggered"] += 1
            if should_act:
                self._metrics["actions_triggered"] += 1
            calculation_time = time.time() - start_time
            reason_parts = [
                f"综合:{engagement_score:.2f}",
                f"语义:{semantic_score:.2f}",
                f"关系:{relationship_score:.2f}",
                f"提及:{mention_score:.2f}",
                "→LLM决策"
            ]
            return EngagementResult(
                success=True, message_id=message_id,
                engagement_score=engagement_score, final_score=engagement_score,
                should_reply=should_reply, should_act=should_act,
                calculation_time=calculation_time,
                semantic_score=semantic_score,
                relationship_score=relationship_score,
                mention_score=mention_score,
                keyword_score=keyword_score,
                sentiment_score=sentiment_score,
                topic_relevance_score=topic_relevance_score,
                decision_reason=" | ".join(reason_parts)
            )
        except Exception as e:
            logger.error(f"计算失败: {e}")
            return EngagementResult(
                success=False, message_id=message_id,
                error_message=str(e), calculation_time=time.time() - start_time,
            )

    async def _calculate_semantic_score(self, content: str) -> float:
        if not content or not content.strip():
            return 0.0
        base_score = min(len(content) / 100, 0.5)
        interesting_keywords = ["有趣", "好玩", "想", "喜欢", "帮", "问"]
        keyword_boost = sum(0.1 for kw in interesting_keywords if kw in content)
        keyword_boost = min(keyword_boost, 0.3)
        return min(base_score + keyword_boost, 1.0)

    async def _calculate_relationship_score(self, user_id: str) -> float:
        if not user_id:
            return 0.3
        current_time = time.time()
        if user_id in self._relationship_cache:
            cache_time = self._cache_timestamps.get(user_id, 0)
            if current_time - cache_time < self._cache_ttl:
                return self._relationship_cache[user_id]
        relationship_score = 0.5
        self._relationship_cache[user_id] = relationship_score
        self._cache_timestamps[user_id] = current_time
        return relationship_score

    def _calculate_mention_score(self, is_mentioned: bool, is_private: bool) -> float:
        if is_private:
            return 1.0
        if is_mentioned:
            return 0.8
        return 0.0

    def _apply_threshold_adjustment(self) -> Tuple[float, float]:
        base_reply_threshold = self.reply_threshold
        base_action_threshold = self.action_threshold
        total_reduction = 0.0
        if 0 < self.no_reply_count < self.max_no_reply_count:
            no_reply_reduction = self.no_reply_count * self.threshold_decay_per_skip
            total_reduction += no_reply_reduction
        if self.post_reply_boost_enabled and self.post_reply_boost_remaining > 0:
            decay_factor = self.post_reply_decay_rate ** (
                self.post_reply_boost_max - self.post_reply_boost_remaining
            )
            post_reply_reduction = self.post_reply_threshold_reduction * decay_factor
            self.post_reply_boost_remaining -= 1
            total_reduction += post_reply_reduction
        adjusted_reply = max(0.0, base_reply_threshold - total_reduction)
        adjusted_action = max(0.0, base_action_threshold - total_reduction)
        return adjusted_reply, adjusted_action

    def update_relationship(self, user_id: str, score: float) -> None:
        self._relationship_cache[user_id] = score
        self._cache_timestamps[user_id] = time.time()

    def on_reply_sent(self) -> None:
        self.no_reply_count = 0
        if self.post_reply_boost_enabled:
            self.post_reply_boost_remaining = self.post_reply_boost_max

    def on_message_skipped(self) -> None:
        self.no_reply_count = min(self.no_reply_count + 1, self.max_no_reply_count)
        if self.post_reply_boost_remaining > 0:
            self.post_reply_boost_remaining -= 1

    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self._metrics,
            "no_reply_count": self.no_reply_count,
            "post_reply_boost_remaining": self.post_reply_boost_remaining,
            "cached_relationships": len(self._relationship_cache),
        }

    def reset_state(self) -> None:
        self.no_reply_count = 0
        self.post_reply_boost_remaining = 0

    def _load_keywords_from_config(self):
        pass

    def _calculate_keyword_score(self, content: str) -> Tuple[float, str]:
        if not content or not content.strip():
            return 0.0, "无"
        return 0.0, "模型驱动"

    def _calculate_sentiment_score(self, content: str) -> float:
        if not content or not content.strip():
            return 0.0
        return 0.0

    def _calculate_topic_relevance_score(self, content: str) -> float:
        if not content or not content.strip():
            return 0.0
        return 0.0


_engagement_scorer: Optional[EngagementScorer] = None


def get_engagement_scorer() -> EngagementScorer:
    global _engagement_scorer
    if _engagement_scorer is None:
        _engagement_scorer = EngagementScorer()
    return _engagement_scorer

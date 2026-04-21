import time
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque
from src.common.logger import get_logger

logger = get_logger("user_state_detector")


class UserState(Enum):
    WANT_TO_CHAT = "want_to_chat"
    NOT_WANT_TO_CHAT = "not_want_to_chat"
    UNCERTAIN = "uncertain"

    def __repr__(self) -> str:
        return self.value


@dataclass
class InteractionRecord:
    timestamp: float = field(default_factory=time.time)
    message_content: str = ""
    user_id: str = ""
    is_initiated_by_user: bool = False
    message_length: int = 0
    contains_question: bool = False
    contains_bot_name: bool = False
    sentiment_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "message_content": self.message_content,
            "user_id": self.user_id,
            "is_initiated_by_user": self.is_initiated_by_user,
            "message_length": self.message_length,
            "contains_question": self.contains_question,
            "contains_bot_name": self.contains_bot_name,
            "sentiment_score": self.sentiment_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionRecord":
        return cls(
            timestamp=data.get("timestamp", time.time()),
            message_content=data.get("message_content", ""),
            user_id=data.get("user_id", ""),
            is_initiated_by_user=data.get("is_initiated_by_user", False),
            message_length=data.get("message_length", 0),
            contains_question=data.get("contains_question", False),
            contains_bot_name=data.get("contains_bot_name", False),
            sentiment_score=data.get("sentiment_score", 0.0),
        )


@dataclass
class UserStateAnalysis:
    state: UserState = UserState.UNCERTAIN
    confidence: float = 0.5
    reasoning: str = ""
    factors: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "factors": self.factors,
            "timestamp": self.timestamp,
        }


class UserStateDetector:
    QUESTION_PATTERNS = [
        r"[？?]$", r"^(谁|什么|哪|怎么|为什么|如何|几|多少)",
        r"(吗|呢|嘛|呗|不|没)$", r"^(是不是|能不能|有没有|要不要)",
    ]
    NEGATIVE_INDICATORS = [
        "滚", "闭嘴", "别说了", "烦", "走开", "不想聊", "拉黑",
        "讨厌", "恶心", "无聊", "别烦", "安静",
    ]
    POSITIVE_INDICATORS = [
        "谢谢", "喜欢", "好棒", "有趣", "继续", "然后呢",
        "再说说", "厉害", "哈哈", "笑死",
    ]

    def __init__(self, max_history_size: int = 50):
        self._max_history_size = max_history_size
        self._interaction_history: Dict[str, deque] = {}
        self._user_state_cache: Dict[str, UserStateAnalysis] = {}
        self._cache_ttl = 60.0

    def _get_user_history(self, user_id: str) -> deque:
        if user_id not in self._interaction_history:
            self._interaction_history[user_id] = deque(maxlen=self._max_history_size)
        return self._interaction_history[user_id]

    def _contains_bot_name(self, message: str) -> bool:
        return False

    def _contains_question(self, message: str) -> bool:
        if not message:
            return False
        for pattern in self.QUESTION_PATTERNS:
            if re.search(pattern, message):
                return True
        return False

    def _calculate_sentiment_score(self, message: str) -> float:
        if not message:
            return 0.0
        score = 0.0
        for indicator in self.POSITIVE_INDICATORS:
            if indicator in message:
                score += 0.3
        for indicator in self.NEGATIVE_INDICATORS:
            if indicator in message:
                score -= 0.5
        return max(-1.0, min(1.0, score))

    def _analyze_time_factor(self, user_id: str) -> float:
        history = self._get_user_history(user_id)
        if not history:
            return 0.5
        now = time.time()
        recent_records = [r for r in history if now - r.timestamp < 3600]
        if not recent_records:
            return 0.3
        time_gaps = []
        for i in range(1, len(recent_records)):
            gap = recent_records[i].timestamp - recent_records[i - 1].timestamp
            time_gaps.append(gap)
        if not time_gaps:
            return 0.5
        avg_gap = sum(time_gaps) / len(time_gaps)
        if avg_gap < 60:
            return 0.9
        elif avg_gap < 300:
            return 0.7
        elif avg_gap < 600:
            return 0.5
        elif avg_gap < 1800:
            return 0.3
        return 0.1

    def _analyze_interaction_frequency(self, user_id: str) -> float:
        history = self._get_user_history(user_id)
        if not history:
            return 0.0
        now = time.time()
        recent_hour = sum(1 for r in history if now - r.timestamp < 3600)
        recent_day = sum(1 for r in history if now - r.timestamp < 86400)
        if recent_hour >= 10:
            return 1.0
        elif recent_hour >= 5:
            return 0.8
        elif recent_hour >= 3:
            return 0.6
        elif recent_hour >= 1:
            return 0.4
        elif recent_day >= 5:
            return 0.3
        elif recent_day >= 1:
            return 0.2
        return 0.1

    def _analyze_message_content(self, message: str) -> float:
        """分析消息内容

        基于长度、关键词、正则的规则计算。

        Returns:
            float: 内容得分，范围0-1，越高表示越想聊天
        """
        if not message:
            return 0.3

        min_length = 3
        enable_filter = True
        penalty = 0.4

        message_len = len(message.strip())

        # 短消息基础分降低
        # 长度1-2："在"、"嗯" → 基础分0.25（明显降低）
        # 长度3-4："哦哦"、"好的" → 基础分0.35（适度降低）
        # 长度5+："你好呀" → 基础分0.5（正常）
        if enable_filter and message_len < min_length:
            score = 0.25
        elif message_len < 5:
            score = 0.35
        else:
            score = 0.5

        message_lower = message.lower()

        if self._contains_question(message):
            score += 0.3

        for indicator in self.POSITIVE_INDICATORS:
            if indicator in message:
                score += 0.25
                break

        for indicator in self.NEGATIVE_INDICATORS:
            if indicator in message:
                score -= 0.5
                break

        if len(message) > 20:
            score += 0.1
        elif len(message) > 5:
            score += 0.05

        # 应用短消息惩罚系数
        # 先累加各种加分（问题+0.3、提及机器人+0.2等）
        # 最后再整体乘以惩罚系数，确保短消息得分大幅降低
        # 示例：
        # - "在" (1字) → 基础分0.25 → 无加分 → 惩罚后0.1
        # - "在?" (2字) → 基础分0.25 → 问题+0.3=0.55 → 惩罚后0.22
        if enable_filter and message_len < min_length:
            score *= penalty
        return max(0.0, min(1.0, score))

    def _analyze_sentiment_trend(self, user_id: str) -> float:
        history = self._get_user_history(user_id)
        if len(history) < 3:
            return 0.5
        recent_records = list(history)[-10:]
        sentiment_scores = [r.sentiment_score for r in recent_records]
        if not sentiment_scores:
            return 0.5
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        if avg_sentiment > 0.3:
            return 0.8
        elif avg_sentiment > 0.0:
            return 0.6
        elif avg_sentiment > -0.3:
            return 0.4
        return 0.2

    def _calculate_initiation_rate(self, user_id: str) -> float:
        history = self._get_user_history(user_id)
        if not history:
            return 0.5
        initiated_count = sum(1 for r in history if r.is_initiated_by_user)
        total_count = len(history)
        if total_count == 0:
            return 0.5
        return initiated_count / total_count

    def record_interaction(self, user_id: str, message_content: str, is_initiated_by_user: bool = False) -> None:
        history = self._get_user_history(user_id)
        record = InteractionRecord(
            message_content=message_content,
            user_id=user_id,
            is_initiated_by_user=is_initiated_by_user,
            message_length=len(message_content),
            contains_question=self._contains_question(message_content),
            sentiment_score=self._calculate_sentiment_score(message_content),
        )
        history.append(record)
        logger.debug(
            f"[用户状态检测] 记录交互: user={user_id[:8]}, "
            f"len={record.message_length}, question={record.contains_question}, "
            f"sentiment={record.sentiment_score:.2f}"
        )

    def detect_user_state(self, user_id: str, current_message: str = "", use_cache: bool = True) -> UserStateAnalysis:
        if use_cache:
            cached = self._get_cached_state(user_id)
            if cached:
                return cached
        factors = {}
        factors["time_factor"] = self._analyze_time_factor(user_id)
        factors["frequency_factor"] = self._analyze_interaction_frequency(user_id)
        factors["content_factor"] = self._analyze_message_content(current_message)
        factors["sentiment_factor"] = self._analyze_sentiment_trend(user_id)
        factors["initiation_factor"] = self._calculate_initiation_rate(user_id)
        weights = {
            "content_factor": 0.35,
            "time_factor": 0.20,
            "frequency_factor": 0.15,
            "sentiment_factor": 0.15,
            "initiation_factor": 0.15,
        }
        weighted_score = sum(factors.get(key, 0.0) * weight for key, weight in weights.items())
        state, confidence, reasoning = self._determine_state_from_score(weighted_score, factors)
        analysis = UserStateAnalysis(state=state, confidence=confidence, reasoning=reasoning, factors=factors)
        self._cache_state(user_id, analysis)
        logger.info(
            f"[用户状态检测] user={user_id[:8]}, state={state.value}, "
            f"score={weighted_score:.2f}, confidence={confidence:.2f}"
        )
        return analysis

    def _determine_state_from_score(self, score: float, factors: Dict[str, float]) -> Tuple[UserState, float, str]:
        if score >= 0.7:
            state = UserState.WANT_TO_CHAT
            confidence = min(0.9, 0.6 + (score - 0.7) * 1.0)
            reasoning = self._generate_reasoning("want_to_chat", factors)
        elif score <= 0.3:
            state = UserState.NOT_WANT_TO_CHAT
            confidence = min(0.9, 0.6 + (0.3 - score) * 1.0)
            reasoning = self._generate_reasoning("not_want_to_chat", factors)
        else:
            state = UserState.UNCERTAIN
            confidence = 0.5
            reasoning = self._generate_reasoning("uncertain", factors)
        return state, confidence, reasoning

    def _generate_reasoning(self, state: str, factors: Dict[str, float]) -> str:
        if state == "want_to_chat":
            primary_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:2]
            reasons = []
            factor_labels = {
                "content_factor": "消息内容积极",
                "time_factor": "近期活跃",
                "frequency_factor": "交互频繁",
                "sentiment_factor": "情感积极",
                "initiation_factor": "主动发起",
            }
            for name, value in primary_factors:
                if value > 0.6 and name in factor_labels:
                    reasons.append(factor_labels[name])
            return "、".join(reasons) if reasons else "综合指标较高"
        elif state == "not_want_to_chat":
            reasons = []
            if factors.get("content_factor", 0.0) < 0.3:
                reasons.append("消息内容消极")
            if factors.get("time_factor", 0.0) < 0.3:
                reasons.append("近期不活跃")
            if factors.get("frequency_factor", 0.0) < 0.2:
                reasons.append("交互稀少")
            if factors.get("sentiment_factor", 0.0) < 0.3:
                reasons.append("情感消极")
            return "、".join(reasons) if reasons else "综合指标较低"
        return "状态不确定，需进一步观察"

    def _get_cached_state(self, user_id: str) -> Optional[UserStateAnalysis]:
        if user_id not in self._user_state_cache:
            return None
        cached = self._user_state_cache[user_id]
        if time.time() - cached.timestamp < self._cache_ttl:
            return cached
        del self._user_state_cache[user_id]
        return None

    def _cache_state(self, user_id: str, analysis: UserStateAnalysis) -> None:
        self._user_state_cache[user_id] = analysis
        if len(self._user_state_cache) > 100:
            oldest_key = min(self._user_state_cache.keys(), key=lambda k: self._user_state_cache[k].timestamp)
            del self._user_state_cache[oldest_key]

    def get_user_history(self, user_id: str) -> List[InteractionRecord]:
        return list(self._get_user_history(user_id))

    def clear_user_history(self, user_id: str) -> None:
        if user_id in self._interaction_history:
            del self._interaction_history[user_id]
        if user_id in self._user_state_cache:
            del self._user_state_cache[user_id]

    def get_statistics(self, user_id: str) -> Dict[str, Any]:
        history = self._get_user_history(user_id)
        if not history:
            return {
                "total_interactions": 0, "initiated_count": 0, "initiation_rate": 0.0,
                "avg_message_length": 0.0, "question_count": 0, "avg_sentiment": 0.0,
            }
        initiated_count = sum(1 for r in history if r.is_initiated_by_user)
        question_count = sum(1 for r in history if r.contains_question)
        return {
            "total_interactions": len(history),
            "initiated_count": initiated_count,
            "initiation_rate": initiated_count / len(history),
            "avg_message_length": sum(r.message_length for r in history) / len(history),
            "question_count": question_count,
            "avg_sentiment": sum(r.sentiment_score for r in history) / len(history),
        }

    def cleanup_old_records(self, max_age_seconds: int = 86400) -> int:
        now = time.time()
        cleaned_count = 0
        for user_id in list(self._interaction_history.keys()):
            history = self._interaction_history[user_id]
            original_size = len(history)
            valid_records = deque(
                [r for r in history if now - r.timestamp < max_age_seconds],
                maxlen=self._max_history_size,
            )
            cleaned_count += original_size - len(valid_records)
            if valid_records:
                self._interaction_history[user_id] = valid_records
            else:
                del self._interaction_history[user_id]
                if user_id in self._user_state_cache:
                    del self._user_state_cache[user_id]
        if cleaned_count > 0:
            logger.info(f"[用户状态检测] 清理过期记录: {cleaned_count}条")
        return cleaned_count


_global_user_state_detector: Optional[UserStateDetector] = None


def get_user_state_detector() -> UserStateDetector:
    global _global_user_state_detector
    if _global_user_state_detector is None:
        _global_user_state_detector = UserStateDetector()
    return _global_user_state_detector


def reset_user_state_detector() -> None:
    global _global_user_state_detector
    _global_user_state_detector = None

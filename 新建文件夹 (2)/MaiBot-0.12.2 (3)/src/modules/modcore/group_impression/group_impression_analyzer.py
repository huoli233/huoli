import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque
from src.common.logger import get_logger

logger = get_logger("group_impression")


class GroupAtmosphere(str, Enum):
    HARMONIOUS = "和谐"
    CHEERFUL = "欢快"
    NEUTRAL = "中立"
    TENSE = "紧张"
    HOSTILE = "恶意"


class AIPosition(str, Enum):
    GROUP_MEMBER = "群友"
    OBSERVER = "观察者"
    DEFENDER = "防御者"


@dataclass
class MessageSentiment:
    content: str
    sentiment: str
    intensity: float
    timestamp: float
    sender_id: str


@dataclass
class GroupImpressionState:
    atmosphere: GroupAtmosphere
    ai_position: AIPosition
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    recent_message_count: int
    analysis_window_size: int = 50


class GroupImpressionAnalyzer:
    def __init__(self, window_size: int = 50):
        self._window_size = window_size
        self._message_history: Dict[str, deque] = {}
        self._impression_cache: Dict[str, GroupImpressionState] = {}
        self._cache_ttl = 60
        self._last_analysis: Dict[str, float] = {}
        logger.info(f"群体印象分析器初始化完成 (窗口大小:{window_size})")

    def analyze_sentiment_simple(self, content: str) -> Tuple[str, float]:
        positive_keywords = [
            "好", "棒", "赞", "爱", "喜欢", "开心", "快乐", "哈哈", "笑", "优秀",
            "厉害", "牛", "强", "谢谢", "感谢", "温柔", "可爱", "有趣", "舒服"
        ]
        negative_keywords = [
            "傻", "笨", "蠢", "垃圾", "废物", "滚", "死", "恨", "讨厌", "烦",
            "操", "fuck", "shit", "cnm", "nmsl", "sb", "傻逼", "草", "艹"
        ]
        content_lower = content.lower()
        positive_count = sum(1 for kw in positive_keywords if kw in content_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in content_lower)
        total = positive_count + negative_count
        if total == 0:
            return "neutral", 0.3
        if negative_count > positive_count:
            intensity = min(0.3 + (negative_count / max(total, 1)) * 0.7, 1.0)
            return "negative", intensity
        elif positive_count > negative_count:
            intensity = min(0.3 + (positive_count / max(total, 1)) * 0.7, 1.0)
            return "positive", intensity
        return "neutral", 0.5

    def record_message(self, stream_id: str, sender_id: str, content: str):
        if stream_id not in self._message_history:
            self._message_history[stream_id] = deque(maxlen=self._window_size)
        sentiment, intensity = self.analyze_sentiment_simple(content)
        msg_sentiment = MessageSentiment(
            content=content, sentiment=sentiment,
            intensity=intensity, timestamp=time.time(), sender_id=sender_id,
        )
        self._message_history[stream_id].append(msg_sentiment)
        self._impression_cache.pop(stream_id, None)

    def analyze_group_impression(self, stream_id: str) -> GroupImpressionState:
        current_time = time.time()
        last_analysis_time = self._last_analysis.get(stream_id, 0)
        if stream_id in self._impression_cache and (current_time - last_analysis_time) < self._cache_ttl:
            return self._impression_cache[stream_id]
        if stream_id not in self._message_history or len(self._message_history[stream_id]) == 0:
            default_state = GroupImpressionState(
                atmosphere=GroupAtmosphere.NEUTRAL, ai_position=AIPosition.GROUP_MEMBER,
                positive_ratio=0.0, negative_ratio=0.0, neutral_ratio=1.0, recent_message_count=0,
            )
            self._impression_cache[stream_id] = default_state
            self._last_analysis[stream_id] = current_time
            return default_state
        messages = list(self._message_history[stream_id])
        positive_count = sum(1 for m in messages if m.sentiment == "positive")
        negative_count = sum(1 for m in messages if m.sentiment == "negative")
        neutral_count = sum(1 for m in messages if m.sentiment == "neutral")
        total = len(messages)
        positive_ratio = positive_count / total if total > 0 else 0
        negative_ratio = negative_count / total if total > 0 else 0
        neutral_ratio = neutral_count / total if total > 0 else 1
        atmosphere = self._determine_atmosphere(positive_ratio, negative_ratio, neutral_ratio)
        ai_position = self._determine_ai_position(atmosphere, negative_ratio)
        state = GroupImpressionState(
            atmosphere=atmosphere, ai_position=ai_position,
            positive_ratio=positive_ratio, negative_ratio=negative_ratio,
            neutral_ratio=neutral_ratio, recent_message_count=total,
        )
        self._impression_cache[stream_id] = state
        self._last_analysis[stream_id] = current_time
        logger.debug(f"群体印象 {stream_id[:8]}: 氛围={atmosphere.value}, 定位={ai_position.value}")
        return state

    def _determine_atmosphere(self, positive_ratio: float, negative_ratio: float,
                               neutral_ratio: float) -> GroupAtmosphere:
        if negative_ratio > 0.4:
            return GroupAtmosphere.HOSTILE
        elif negative_ratio > 0.2:
            return GroupAtmosphere.TENSE
        elif positive_ratio > 0.5:
            return GroupAtmosphere.CHEERFUL
        elif positive_ratio > 0.3:
            return GroupAtmosphere.HARMONIOUS
        return GroupAtmosphere.NEUTRAL

    def _determine_ai_position(self, atmosphere: GroupAtmosphere,
                                negative_ratio: float) -> AIPosition:
        if atmosphere == GroupAtmosphere.HOSTILE:
            return AIPosition.DEFENDER
        elif atmosphere == GroupAtmosphere.TENSE:
            return AIPosition.OBSERVER if negative_ratio > 0.25 else AIPosition.GROUP_MEMBER
        return AIPosition.GROUP_MEMBER


_group_impression_analyzer_instance: Optional[GroupImpressionAnalyzer] = None


def get_group_impression_analyzer() -> GroupImpressionAnalyzer:
    global _group_impression_analyzer_instance
    if _group_impression_analyzer_instance is None:
        _group_impression_analyzer_instance = GroupImpressionAnalyzer()
    return _group_impression_analyzer_instance

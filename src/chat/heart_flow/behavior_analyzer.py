import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine

logger = get_logger("behavior_analyzer")


class BehaviorType(Enum):
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    INQUISITIVE = "inquisitive"
    PLAYFUL = "playful"
    HOSTILE = "hostile"
    WITHDRAWING = "withdrawing"


@dataclass
class SentimentSample:
    polarity: float = 0.0
    sampled_at: float = 0.0


@dataclass
class InterestEntry:
    topic: str = ""
    weight: float = 1.0
    recorded_at: float = 0.0


@dataclass
class MemoryNote:
    content: str = ""
    importance: float = 1.0
    created_at: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class BehaviorDigest:
    sentiment_trend: str = "stable"
    average_polarity: float = 0.0
    top_interests: List[Tuple[str, float]] = field(default_factory=list)
    message_frequency: float = 0.0
    dominant_behavior: str = "neutral"
    recent_memory_count: int = 0


class SentimentTracker:
    def __init__(self, config_engine=None):
        self._config = config_engine or get_default_config_engine()
        self._user_samples: Dict[str, Deque[SentimentSample]] = {}
        self._window_size: int = 200
        self._trend_threshold: float = 0.15
        self._load_config()

    def _load_config(self) -> None:
        self._window_size = int(self._config.get("behavior_analyzer", "sentiment_window_size", 200))
        self._trend_threshold = float(self._config.get("behavior_analyzer", "sentiment_trend_threshold", 0.15))

    def record(self, uid: str, polarity: float) -> None:
        if uid not in self._user_samples:
            self._user_samples[uid] = deque(maxlen=self._window_size)
        self._user_samples[uid].append(SentimentSample(polarity=polarity, sampled_at=time.time()))

    def compute_average(self, uid: str) -> float:
        samples = self._user_samples.get(uid)
        if not samples:
            return 0.0
        return sum(s.polarity for s in samples) / len(samples)

    def derive_trend(self, uid: str, window_sec: float = 1800.0) -> str:
        samples = self._user_samples.get(uid)
        if not samples or len(samples) < 3:
            return "stable"
        cutoff = time.time() - window_sec
        recent = [s for s in samples if s.sampled_at >= cutoff]
        if len(recent) < 2:
            return "stable"
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]
        avg_first = sum(s.polarity for s in first_half) / len(first_half)
        avg_second = sum(s.polarity for s in second_half) / len(second_half)
        delta = avg_second - avg_first
        if delta > self._trend_threshold:
            return "improving"
        if delta < -self._trend_threshold:
            return "declining"
        return "stable"


class InterestProfiler:
    def __init__(self, config_engine=None):
        self._config = config_engine or get_default_config_engine()
        self._user_interests: Dict[str, List[InterestEntry]] = {}
        self._decay_factor: float = 0.95
        self._min_weight_threshold: float = 0.1
        self._load_config()

    def _load_config(self) -> None:
        self._decay_factor = float(self._config.get("behavior_analyzer", "interest_decay_factor", 0.95))
        self._min_weight_threshold = float(self._config.get("behavior_analyzer", "interest_min_weight", 0.1))

    def record_topic(self, uid: str, topic: str, weight: float = 1.0) -> None:
        if uid not in self._user_interests:
            self._user_interests[uid] = []
        self.apply_decay(uid)
        entries = self._user_interests[uid]
        for entry in entries:
            if entry.topic == topic:
                entry.weight = min(5.0, entry.weight + weight)
                entry.recorded_at = time.time()
                return
        entries.append(InterestEntry(topic=topic, weight=weight, recorded_at=time.time()))

    def apply_decay(self, uid: str) -> None:
        entries = self._user_interests.get(uid, [])
        survived = []
        for entry in entries:
            entry.weight *= self._decay_factor
            if entry.weight >= self._min_weight_threshold:
                survived.append(entry)
        self._user_interests[uid] = survived

    def top_interests(self, uid: str, cap: int = 10) -> List[Tuple[str, float]]:
        entries = self._user_interests.get(uid, [])
        ranked = sorted(entries, key=lambda e: e.weight, reverse=True)
        return [(e.topic, round(e.weight, 2)) for e in ranked[:cap]]


class InteractionMemoryBank:
    def __init__(self, config_engine=None, cap: int = 60):
        self._config = config_engine or get_default_config_engine()
        self._cap = cap
        self._user_memories: Dict[str, List[MemoryNote]] = {}

    def add(
        self,
        uid: str,
        content: str,
        importance: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> None:
        if uid not in self._user_memories:
            self._user_memories[uid] = []
        note = MemoryNote(
            content=content,
            importance=importance,
            created_at=time.time(),
            tags=tags or [],
        )
        self._user_memories[uid].append(note)
        if len(self._user_memories[uid]) > self._cap:
            self._user_memories[uid].sort(key=lambda n: n.importance, reverse=True)
            self._user_memories[uid] = self._user_memories[uid][: self._cap]

    def search(self, uid: str, query: str, cap: int = 5) -> List[MemoryNote]:
        notes = self._user_memories.get(uid, [])
        query_tokens = set(query.lower().split())
        scored = []
        for note in notes:
            note_tokens = set(note.content.lower().split())
            overlap = len(query_tokens & note_tokens)
            if overlap > 0:
                scored.append((note, overlap))
        scored.sort(key=lambda p: (p[1], p[0].importance), reverse=True)
        return [p[0] for p in scored[:cap]]

    def count(self, uid: str) -> int:
        return len(self._user_memories.get(uid, []))


class UnifiedBehaviorAnalyzer:
    def __init__(self, config_engine=None):
        self._config = config_engine or get_default_config_engine()
        self._memory_cap: int = 60
        self._load_config()
        self.sentiment = SentimentTracker(self._config)
        self.interests = InterestProfiler(self._config)
        self.memory_bank = InteractionMemoryBank(self._config, cap=self._memory_cap)
        self._message_timestamps: Dict[str, Deque[float]] = {}
        self._positive_threshold: float = 0.3
        self._negative_threshold: float = -0.3

    def _load_config(self) -> None:
        self._memory_cap = int(self._config.get("behavior_analyzer", "memory_cap", 60))
        self._positive_threshold = float(self._config.get("behavior_analyzer", "positive_polarity_threshold", 0.3))
        self._negative_threshold = float(self._config.get("behavior_analyzer", "negative_polarity_threshold", -0.3))

    def ingest(
        self,
        uid: str,
        content: str,
        polarity: float = 0.0,
        behavior_type: str = "neutral",
        topic: str = "",
        topic_weight: float = 1.0,
        memory_importance: float = 0.0,
    ) -> None:
        self.sentiment.record(uid, polarity)
        if topic:
            self.interests.record_topic(uid, topic, topic_weight)
        if memory_importance > 0:
            self.memory_bank.add(uid, content, memory_importance)
        if uid not in self._message_timestamps:
            self._message_timestamps[uid] = deque(maxlen=500)
        self._message_timestamps[uid].append(time.time())

    def get_message_frequency(self, uid: str, window_hours: float = 1.0) -> float:
        timestamps = self._message_timestamps.get(uid)
        if not timestamps:
            return 0.0
        cutoff = time.time() - window_hours * 3600
        recent = [t for t in timestamps if t >= cutoff]
        return len(recent) / max(window_hours, 0.01)

    def compile_digest(self, uid: str) -> BehaviorDigest:
        trend = self.sentiment.derive_trend(uid)
        avg_pol = self.sentiment.compute_average(uid)
        top_int = self.interests.top_interests(uid, 5)
        freq = self.get_message_frequency(uid)
        mem_count = self.memory_bank.count(uid)
        dominant = "neutral"
        if avg_pol > self._positive_threshold:
            dominant = "positive"
        elif avg_pol < self._negative_threshold:
            dominant = "negative"
        return BehaviorDigest(
            sentiment_trend=trend,
            average_polarity=round(avg_pol, 3),
            top_interests=top_int,
            message_frequency=round(freq, 1),
            dominant_behavior=dominant,
            recent_memory_count=mem_count,
        )

    def classify_behavior(self, polarity: float, has_question: bool, is_playful: bool) -> BehaviorType:
        if polarity < -0.5:
            return BehaviorType.HOSTILE
        if polarity < -0.2:
            return BehaviorType.NEGATIVE
        if is_playful:
            return BehaviorType.PLAYFUL
        if has_question:
            return BehaviorType.INQUISITIVE
        if polarity > 0.3:
            return BehaviorType.POSITIVE
        if polarity < -0.1:
            return BehaviorType.WITHDRAWING
        return BehaviorType.NEUTRAL

    def get_stats(self) -> Dict[str, Any]:
        return {
            "memory_cap": self._memory_cap,
            "positive_threshold": self._positive_threshold,
            "negative_threshold": self._negative_threshold,
            "user_count": len(self._message_timestamps),
        }

    @staticmethod
    def detect_low_info_repeat_mode(repetition_signal: Optional[Dict[str, Any]]) -> bool:
        """判断是否落入低信息复读/玩梗模式。"""
        if not repetition_signal or not repetition_signal.get("detected"):
            return False
        exact_repeat_count = int(repetition_signal.get("exact_repeat_count", 0) or 0)
        low_info_cluster = bool(repetition_signal.get("low_info_cluster", False))
        latest_matches_repeat = bool(repetition_signal.get("latest_matches_repeat", False))
        return latest_matches_repeat and low_info_cluster and exact_repeat_count >= 3


_behavior_analyzer: Optional[UnifiedBehaviorAnalyzer] = None


def get_behavior_analyzer(config_engine=None) -> UnifiedBehaviorAnalyzer:
    global _behavior_analyzer
    if _behavior_analyzer is None:
        _behavior_analyzer = UnifiedBehaviorAnalyzer(config_engine)
    return _behavior_analyzer

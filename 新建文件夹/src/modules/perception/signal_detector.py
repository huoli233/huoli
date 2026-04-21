import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine

logger = get_logger("信号检测")


class SignalLevel(Enum):
    NONE = "none"
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"


@dataclass
class StandardMessage:
    message_id: str = ""
    content: str = ""
    sender_id: str = ""
    channel_id: str = ""
    timestamp: float = 0.0
    is_at_me: bool = False
    is_reply: bool = False
    reply_to_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalResult:
    level: SignalLevel = SignalLevel.NONE
    score: float = 0.0
    factors: List[str] = field(default_factory=list)
    message_count: int = 0
    at_count: int = 0
    name_mention_count: int = 0
    detected_at: float = field(default_factory=time.time)


class SignalDetector:
    def __init__(self, config_engine=None):
        self._config = config_engine or get_default_config_engine()
        self._medium_threshold: int = 10
        self._strong_at_threshold: int = 1
        self._name_match_enabled: bool = True
        self._history: List[SignalResult] = []
        self._max_history: int = 50
        self._load_config()

    def _load_config(self) -> None:
        self._medium_threshold = int(
            self._config.get("signal_detector", "medium_threshold", 10)
        )
        self._strong_at_threshold = int(
            self._config.get("signal_detector", "strong_at_threshold", 1)
        )
        self._name_match_enabled = bool(
            self._config.get("signal_detector", "name_match_enabled", True)
        )
        self._max_history = int(
            self._config.get("signal_detector", "max_history", 50)
        )

    def detect(
        self, buffer: List[StandardMessage], bot_name: str = ""
    ) -> SignalResult:
        if not buffer:
            return SignalResult(level=SignalLevel.NONE)
        at_count = sum(1 for m in buffer if m.is_at_me)
        if at_count >= self._strong_at_threshold:
            result = SignalResult(
                level=SignalLevel.STRONG,
                score=1.0,
                factors=[f"被@了{at_count}次"],
                message_count=len(buffer),
                at_count=at_count,
            )
            self._update_history(result)
            return result
        name_mention_count = 0
        if bot_name and self._name_match_enabled:
            name_mention_count = sum(
                1
                for m in buffer
                if m.content and bot_name.lower() in m.content.lower()
            )
        message_count = len(buffer)
        score = self._calculate_score(
            message_count, at_count, name_mention_count
        )
        factors = []
        if message_count >= self._medium_threshold:
            factors.append(f"消息量{message_count}条")
        if name_mention_count > 0:
            factors.append(f"名字被提及{name_mention_count}次")
        if at_count > 0:
            factors.append(f"被@{at_count}次")
        if score >= 0.7:
            level = SignalLevel.STRONG
        elif score >= 0.4:
            level = SignalLevel.MEDIUM
        elif score > 0:
            level = SignalLevel.WEAK
        else:
            level = SignalLevel.NONE
        result = SignalResult(
            level=level,
            score=score,
            factors=factors,
            message_count=message_count,
            at_count=at_count,
            name_mention_count=name_mention_count,
        )
        self._update_history(result)
        return result

    def _calculate_score(
        self, message_count: int, at_count: int, name_mention_count: int
    ) -> float:
        score = 0.0
        if message_count > 0:
            score += min(0.3, message_count / self._medium_threshold * 0.3)
        score += at_count * 0.5
        score += name_mention_count * 0.2
        return min(1.0, score)

    def _update_history(self, result: SignalResult) -> None:
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_recent_signals(self, limit: int = 10) -> List[SignalResult]:
        return self._history[-limit:]

    def get_signal_frequency(
        self, window_minutes: float = 5.0
    ) -> Dict[str, int]:
        if not self._history:
            return {"none": 0, "weak": 0, "medium": 0, "strong": 0}
        cutoff = time.time() - window_minutes * 60
        recent = [s for s in self._history if s.detected_at >= cutoff]
        counts = {"none": 0, "weak": 0, "medium": 0, "strong": 0}
        for s in recent:
            counts[s.level.value] = counts.get(s.level.value, 0) + 1
        return counts

    def get_stats(self) -> Dict[str, Any]:
        return {
            "medium_threshold": self._medium_threshold,
            "strong_at_threshold": self._strong_at_threshold,
            "name_match_enabled": self._name_match_enabled,
            "history_size": len(self._history),
        }


_signal_detector: Optional[SignalDetector] = None


def get_signal_detector(config_engine=None) -> SignalDetector:
    global _signal_detector
    if _signal_detector is None:
        _signal_detector = SignalDetector(config_engine)
    return _signal_detector

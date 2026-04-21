import re
from dataclasses import dataclass
from typing import Optional
from src.common.logger import get_logger

logger = get_logger("interest_calc")

_QUESTION_MARKERS = re.compile(
    r"[？?]|谁|什么|怎么|为什么|哪|如何|吗$|呢$|嘛$"
)
_EXCLAIM_MARKERS = re.compile(r"[！!]{1,3}")
_URL_PATTERN = re.compile(r"https?://\S+")
_DIGIT_RUN = re.compile(r"\d{2,}")
_HIGH_SIGNAL_WORDS = frozenset(
    [
        "建议",
        "推荐",
        "分析",
        "总结",
        "原因",
        "方案",
        "对比",
        "区别",
        "步骤",
        "教程",
        "原理",
        "经验",
        "技巧",
        "注意",
    ]
)


@dataclass(frozen=True)
class EngagementResult:
    score: float
    curiosity: float
    intensity: float
    density: float


class TextEngagementScorer:
    """文本参与度评分器"""

    def __init__(
        self,
        base_weight: float = 0.35,
        curiosity_weight: float = 0.25,
        intensity_weight: float = 0.15,
        density_weight: float = 0.25,
    ):
        self._w_base = base_weight
        self._w_curiosity = curiosity_weight
        self._w_intensity = intensity_weight
        self._w_density = density_weight

    def calculate_interest(self, text: str) -> EngagementResult:
        if not text or not text.strip():
            return EngagementResult(
                score=0.0, curiosity=0.0, intensity=0.0, density=0.0
            )
        cleaned = text.strip()
        base = self._length_score(cleaned)
        curiosity = self._curiosity_score(cleaned)
        intensity = self._intensity_score(cleaned)
        density = self._density_score(cleaned)
        combined = (
            base * self._w_base
            + curiosity * self._w_curiosity
            + intensity * self._w_intensity
            + density * self._w_density
        )
        clamped = max(0.0, min(1.0, combined))
        return EngagementResult(
            score=round(clamped, 4),
            curiosity=round(curiosity, 4),
            intensity=round(intensity, 4),
            density=round(density, 4),
        )

    @staticmethod
    def _length_score(text: str) -> float:
        n = len(text)
        if n < 3:
            return 0.1
        if n < 10:
            return 0.3
        if n < 30:
            return 0.5
        if n < 80:
            return 0.7
        if n < 200:
            return 0.85
        return 1.0

    @staticmethod
    def _curiosity_score(text: str) -> float:
        hits = len(_QUESTION_MARKERS.findall(text))
        if hits >= 3:
            return 1.0
        if hits >= 1:
            return 0.6 + hits * 0.15
        return 0.0

    @staticmethod
    def _intensity_score(text: str) -> float:
        excl = len(_EXCLAIM_MARKERS.findall(text))
        val = min(1.0, excl * 0.35)
        if any(c.isupper() and c.isascii() for c in text):
            val = min(1.0, val + 0.15)
        return val

    @staticmethod
    def _density_score(text: str) -> float:
        n = len(text)
        if n == 0:
            return 0.0
        signals = 0
        signals += len(_URL_PATTERN.findall(text)) * 8
        signals += len(_DIGIT_RUN.findall(text)) * 3
        for kw in _HIGH_SIGNAL_WORDS:
            if kw in text:
                signals += 4
        ratio = signals / max(n, 1)
        return min(1.0, ratio * 5.0)


_scorer_ref: Optional[TextEngagementScorer] = None


def get_engagement_scorer() -> TextEngagementScorer:
    global _scorer_ref
    if _scorer_ref is None:
        _scorer_ref = TextEngagementScorer()
    return _scorer_ref

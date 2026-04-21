from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import time
from src.common.logger import get_logger

logger = get_logger("scoring_system")


class DecisionType(Enum):
    SEND_STICKER = "send_sticker"
    RECALL_MESSAGE = "recall_message"
    SEND_EMOJI = "send_emoji"
    JOIN_CHAT = "join_chat"
    REST = "rest"


class ScoreLevel(Enum):
    FORBIDDEN = "forbidden"
    NOT_RECOMMENDED = "not_recommended"
    NEUTRAL = "neutral"
    RECOMMENDED = "recommended"
    HIGHLY_RECOMMENDED = "highly_recommended"


@dataclass
class ScoringFactors:
    favor: float = 0.0
    trust: float = 0.0
    annoyance: float = 0.0
    scene_type: str = "group"
    context_relevance: float = 0.5
    timing_score: float = 0.5
    risk_level: float = 0.0
    dialogue_length: int = 0
    user_activity: float = 0.5
    chat_value: float = 100.0
    brain_power: float = 100.0
    consecutive_replies: int = 0
    mood: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionScore:
    total_score: float
    level: ScoreLevel
    dimension_scores: Dict[str, float]
    reasoning: str
    recommendation: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


class UnifiedScoringSystem:
    def __init__(self):
        self._weights = {
            "emotion": 0.30,
            "scene": 0.20,
            "context": 0.20,
            "timing": 0.15,
            "risk": 0.15,
        }
        self._thresholds = {
            ScoreLevel.HIGHLY_RECOMMENDED: 90,
            ScoreLevel.RECOMMENDED: 70,
            ScoreLevel.NEUTRAL: 50,
            ScoreLevel.NOT_RECOMMENDED: 30,
            ScoreLevel.FORBIDDEN: 0,
        }

    def calculate_score(self, decision_type: DecisionType, factors: ScoringFactors) -> DecisionScore:
        dims = {}
        dims["emotion"] = self._score_emotion(factors)
        dims["scene"] = self._score_scene(factors, decision_type)
        dims["context"] = self._score_context(factors)
        dims["timing"] = self._score_timing(factors)
        dims["risk"] = self._score_risk(factors)
        total = sum(dims[d] * self._weights[d] for d in dims)
        level = self._classify(total)
        reasoning = self._build_reasoning(dims, factors, decision_type)
        recommendation = self._build_recommendation(level, decision_type)
        confidence = self._calc_confidence(dims)
        logger.debug(f"[评分] {decision_type.value} 总分={total:.1f} 等级={level.value}")
        return DecisionScore(
            total_score=total, level=level, dimension_scores=dims,
            reasoning=reasoning, recommendation=recommendation,
            confidence=confidence,
        )

    def _score_emotion(self, f: ScoringFactors) -> float:
        aff_norm = (f.favor + 100) / 200
        trust_norm = (f.trust + 100) / 200
        ann_norm = (f.annoyance + 100) / 200
        score = (aff_norm * 0.4 + trust_norm * 0.4 - ann_norm * 0.2) * 100
        return max(0, min(100, score))

    def _score_scene(self, f: ScoringFactors, dt: DecisionType) -> float:
        table = {
            "group": {
                "send_sticker": 90, "recall_message": 50,
                "send_emoji": 85, "join_chat": 80, "rest": 60,
            },
            "private": {
                "send_sticker": 85, "recall_message": 70,
                "send_emoji": 80, "join_chat": 90, "rest": 50,
            },
        }
        return table.get(f.scene_type, {}).get(dt.value, 50)

    def _score_context(self, f: ScoringFactors) -> float:
        score = f.context_relevance * 100
        if f.dialogue_length > 10:
            score += 10
        elif f.dialogue_length < 3:
            score -= 10
        return max(0, min(100, score))

    def _score_timing(self, f: ScoringFactors) -> float:
        score = f.timing_score * 100
        activity_bonus = (f.user_activity - 0.5) * 20
        score += activity_bonus
        return max(0, min(100, score))

    def _score_risk(self, f: ScoringFactors) -> float:
        return max(0, min(100, 100 - f.risk_level * 100))

    def _classify(self, total: float) -> ScoreLevel:
        for level in [ScoreLevel.HIGHLY_RECOMMENDED, ScoreLevel.RECOMMENDED,
                      ScoreLevel.NEUTRAL, ScoreLevel.NOT_RECOMMENDED]:
            if total >= self._thresholds[level]:
                return level
        return ScoreLevel.FORBIDDEN

    def _build_reasoning(self, dims: Dict[str, float], f: ScoringFactors, dt: DecisionType) -> str:
        parts = []
        if dims["emotion"] >= 70:
            parts.append(f"情感良好(好感{f.favor:.0f}/信任{f.trust:.0f})")
        elif dims["emotion"] < 40:
            parts.append(f"情感不佳(烦躁{f.annoyance:.0f})")
        if dims["scene"] >= 80:
            parts.append(f"场景适配({f.scene_type})")
        elif dims["scene"] < 50:
            parts.append(f"场景不适({f.scene_type})")
        if dims["context"] >= 70:
            parts.append(f"上下文相关(对话{f.dialogue_length}轮)")
        elif dims["context"] < 40:
            parts.append("上下文弱")
        if dims["timing"] >= 70:
            parts.append(f"时机好(活跃{f.user_activity:.1f})")
        elif dims["timing"] < 40:
            parts.append("时机差")
        if dims["risk"] < 60:
            parts.append(f"风险高({f.risk_level:.2f})")
        return "；".join(parts) if parts else "综合评估"

    def _build_recommendation(self, level: ScoreLevel, dt: DecisionType) -> str:
        names = {
            DecisionType.SEND_STICKER: "发送表情包",
            DecisionType.RECALL_MESSAGE: "撤回消息",
            DecisionType.SEND_EMOJI: "发送表情",
            DecisionType.JOIN_CHAT: "融入聊天",
            DecisionType.REST: "休息",
        }
        name = names.get(dt, "执行操作")
        labels = {
            ScoreLevel.HIGHLY_RECOMMENDED: f"强烈建议{name}",
            ScoreLevel.RECOMMENDED: f"建议{name}",
            ScoreLevel.NEUTRAL: f"可以考虑{name}",
            ScoreLevel.NOT_RECOMMENDED: f"不建议{name}",
            ScoreLevel.FORBIDDEN: f"禁止{name}",
        }
        return labels.get(level, name)

    def _calc_confidence(self, dims: Dict[str, float]) -> float:
        scores = list(dims.values())
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        return max(0, min(1, 1 - (std_dev / 50)))


_scoring_instance: Optional[UnifiedScoringSystem] = None


def get_scoring_system() -> UnifiedScoringSystem:
    global _scoring_instance
    if _scoring_instance is None:
        _scoring_instance = UnifiedScoringSystem()
    return _scoring_instance

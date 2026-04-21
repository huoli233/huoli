from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
import time
from src.common.logger import get_logger

logger = get_logger("scoring_system")


class DecisionType(Enum):
    SEND_STICKER = "send_sticker"
    SEND_VOICE = "send_voice"
    RECALL_MESSAGE = "recall_message"
    SEND_IMAGE = "send_image"
    SEND_QUOTE = "send_quote"


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
    scene_type: str = "private"
    context_relevance: float = 0.5
    timing_score: float = 0.5
    risk_level: float = 0.0
    dialogue_length: int = 0
    user_activity: float = 0.5
    extra_factors: Dict[str, Any] = field(default_factory=dict)


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
        self._score_weights = {
            "emotion": 0.30,
            "scene": 0.20,
            "context": 0.20,
            "timing": 0.15,
            "risk": 0.15,
        }
        self._score_thresholds = {
            ScoreLevel.HIGHLY_RECOMMENDED: 90,
            ScoreLevel.RECOMMENDED: 70,
            ScoreLevel.NEUTRAL: 50,
            ScoreLevel.NOT_RECOMMENDED: 30,
            ScoreLevel.FORBIDDEN: 0,
        }

    def calculate_score(self, decision_type: DecisionType, factors: ScoringFactors) -> DecisionScore:
        dimension_scores = {}
        dimension_scores["emotion"] = self._calculate_emotion_score(factors)
        dimension_scores["scene"] = self._calculate_scene_score(factors, decision_type)
        dimension_scores["context"] = self._calculate_context_score(factors)
        dimension_scores["timing"] = self._calculate_timing_score(factors)
        dimension_scores["risk"] = self._calculate_risk_score(factors)
        total_score = sum(dimension_scores[dim] * self._score_weights[dim] for dim in dimension_scores)
        level = self._determine_level(total_score)
        reasoning = self._generate_reasoning(dimension_scores, factors, decision_type)
        recommendation = self._generate_recommendation(level, decision_type)
        confidence = self._calculate_confidence(dimension_scores)
        logger.debug(f"[评分系统] {decision_type.value} 总分: {total_score:.1f} 等级: {level.value}")
        return DecisionScore(
            total_score=total_score, level=level, dimension_scores=dimension_scores,
            reasoning=reasoning, recommendation=recommendation, confidence=confidence,
        )

    def _calculate_emotion_score(self, factors: ScoringFactors) -> float:
        affection_norm = (factors.favor + 100) / 200
        trust_norm = (factors.trust + 100) / 200
        annoyance_norm = (factors.annoyance + 100) / 200
        emotion_score = (affection_norm * 0.4 + trust_norm * 0.4 - annoyance_norm * 0.2) * 100
        return max(0, min(100, emotion_score))

    def _calculate_scene_score(self, factors: ScoringFactors, decision_type: DecisionType) -> float:
        scene_scores = {
            "private": {"send_sticker": 85, "send_voice": 80, "recall_message": 70, "send_image": 75, "send_quote": 70},
            "group": {"send_sticker": 90, "send_voice": 60, "recall_message": 50, "send_image": 80, "send_quote": 85},
            "forward": {"send_sticker": 50, "send_voice": 30, "recall_message": 40, "send_image": 60, "send_quote": 70},
        }
        return scene_scores.get(factors.scene_type, {}).get(decision_type.value, 50)

    def _calculate_context_score(self, factors: ScoringFactors) -> float:
        context_score = factors.context_relevance * 100
        if factors.dialogue_length > 10:
            context_score += 10
        elif factors.dialogue_length < 3:
            context_score -= 10
        return max(0, min(100, context_score))

    def _calculate_timing_score(self, factors: ScoringFactors) -> float:
        timing_score = factors.timing_score * 100
        activity_bonus = (factors.user_activity - 0.5) * 20
        timing_score += activity_bonus
        return max(0, min(100, timing_score))

    def _calculate_risk_score(self, factors: ScoringFactors) -> float:
        risk_penalty = factors.risk_level * 100
        safety_score = 100 - risk_penalty
        return max(0, min(100, safety_score))

    def _determine_level(self, total_score: float) -> ScoreLevel:
        if total_score >= self._score_thresholds[ScoreLevel.HIGHLY_RECOMMENDED]:
            return ScoreLevel.HIGHLY_RECOMMENDED
        elif total_score >= self._score_thresholds[ScoreLevel.RECOMMENDED]:
            return ScoreLevel.RECOMMENDED
        elif total_score >= self._score_thresholds[ScoreLevel.NEUTRAL]:
            return ScoreLevel.NEUTRAL
        elif total_score >= self._score_thresholds[ScoreLevel.NOT_RECOMMENDED]:
            return ScoreLevel.NOT_RECOMMENDED
        return ScoreLevel.FORBIDDEN

    def _generate_reasoning(self, dimension_scores: Dict[str, float], factors: ScoringFactors, decision_type: DecisionType) -> str:
        reasons = []
        if dimension_scores["emotion"] >= 70:
            reasons.append(f"情感状态良好(好感{factors.favor:.0f}/信任{factors.trust:.0f})")
        elif dimension_scores["emotion"] < 40:
            reasons.append(f"情感状态不佳(厌恶{factors.annoyance:.0f})")
        if dimension_scores["scene"] >= 80:
            reasons.append(f"场景适配度高({factors.scene_type})")
        elif dimension_scores["scene"] < 50:
            reasons.append(f"场景不太适合({factors.scene_type})")
        if dimension_scores["context"] >= 70:
            reasons.append(f"上下文相关性强(对话{factors.dialogue_length}轮)")
        elif dimension_scores["context"] < 40:
            reasons.append("上下文相关性弱")
        if dimension_scores["timing"] >= 70:
            reasons.append(f"时机合适(活跃度{factors.user_activity:.1f})")
        elif dimension_scores["timing"] < 40:
            reasons.append("时机不佳")
        if dimension_scores["risk"] < 60:
            reasons.append(f"风险较高(风险值{factors.risk_level:.2f})")
        return "；".join(reasons) if reasons else "综合评估"

    def _generate_recommendation(self, level: ScoreLevel, decision_type: DecisionType) -> str:
        action_names = {
            DecisionType.SEND_STICKER: "发送表情包",
            DecisionType.SEND_VOICE: "发送语音",
            DecisionType.RECALL_MESSAGE: "撤回消息",
            DecisionType.SEND_IMAGE: "发送图片",
            DecisionType.SEND_QUOTE: "引用回复",
        }
        action_name = action_names.get(decision_type, "执行操作")
        level_prefixes = {
            ScoreLevel.HIGHLY_RECOMMENDED: "强烈建议",
            ScoreLevel.RECOMMENDED: "建议",
            ScoreLevel.NEUTRAL: "可以考虑",
            ScoreLevel.NOT_RECOMMENDED: "不建议",
            ScoreLevel.FORBIDDEN: "禁止",
        }
        return f"{level_prefixes.get(level, '评估')}{action_name}"

    def _calculate_confidence(self, dimension_scores: Dict[str, float]) -> float:
        scores = list(dimension_scores.values())
        avg_score = sum(scores) / len(scores)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        return max(0, min(1, 1 - (std_dev / 50)))


_scoring_system_instance: Optional[UnifiedScoringSystem] = None


def get_scoring_system() -> UnifiedScoringSystem:
    global _scoring_system_instance
    if _scoring_system_instance is None:
        _scoring_system_instance = UnifiedScoringSystem()
    return _scoring_system_instance

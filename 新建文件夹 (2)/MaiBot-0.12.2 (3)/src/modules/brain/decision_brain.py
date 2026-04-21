import time
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from src.common.logger import get_logger

logger = get_logger("decision_brain")


class DecisionType(Enum):
    REPLY = "reply"
    CONTENT = "content"
    STRATEGY = "strategy"
    TIMING = "timing"
    INTENSITY = "intensity"


class DecisionConfidence(Enum):
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class DecisionContext:
    stream_id: str = ""
    user_id: str = ""
    message_content: str = ""
    conversation_history: List[Dict] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    current_mood: str = "neutral"
    time_context: Dict[str, Any] = field(default_factory=dict)
    social_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionFactors:
    confidence_score: float = 0.5
    risk_score: float = 0.5
    interest_score: float = 0.5
    urgency_score: float = 0.5
    social_score: float = 0.5
    mood_score: float = 0.5


@dataclass
class DecisionResult:
    decision_type: DecisionType
    decision: Any
    confidence: DecisionConfidence
    reasoning: str = ""
    factors: DecisionFactors = field(default_factory=DecisionFactors)
    alternatives: List[Dict] = field(default_factory=list)
    execution_plan: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class DecisionBrain:
    def __init__(self):
        from src.config.config import global_config
        brain_cfg = global_config.brain
        self._decision_history: List[DecisionResult] = []
        self._max_history = brain_cfg.max_decision_history
        self._learning_enabled = True
        self._weights = {
            'confidence': 0.25,
            'risk': 0.20,
            'interest': 0.20,
            'urgency': 0.15,
            'social': 0.10,
            'mood': 0.10
        }
        self._thresholds = {
            DecisionConfidence.VERY_LOW: 0.3,
            DecisionConfidence.LOW: 0.5,
            DecisionConfidence.MEDIUM: 0.7,
            DecisionConfidence.HIGH: 0.85,
            DecisionConfidence.VERY_HIGH: 0.95
        }
        logger.info("决策大脑初始化完成")

    async def make_decision(
        self,
        decision_type: DecisionType,
        context: DecisionContext,
        factors: Optional[DecisionFactors] = None
    ) -> DecisionResult:
        try:
            if factors is None:
                factors = await self._calculate_factors(context)
            if decision_type == DecisionType.REPLY:
                decision = self._decide_whether_to_reply(context, factors)
            elif decision_type == DecisionType.CONTENT:
                decision = self._decide_content_type(context, factors)
            elif decision_type == DecisionType.STRATEGY:
                decision = self._decide_strategy(context, factors)
            elif decision_type == DecisionType.TIMING:
                decision = self._decide_timing(context, factors)
            elif decision_type == DecisionType.INTENSITY:
                decision = self._decide_intensity(context, factors)
            else:
                decision = self._make_general_decision(context, factors)
            confidence_level = self._calculate_overall_confidence(factors)
            reasoning = self._generate_reasoning(decision_type, decision, factors)
            result = DecisionResult(
                decision_type=decision_type,
                decision=decision,
                confidence=confidence_level,
                reasoning=reasoning,
                factors=factors,
                alternatives=self._generate_alternatives(decision_type, decision),
                execution_plan=self._create_execution_plan(decision_type, decision, context)
            )
            self._record_decision(result)
            if decision_type == DecisionType.REPLY:
                self._update_engagement_state(result.decision)
            logger.debug(f"决策完成: {decision_type.value} -> {decision} (置信度: {confidence_level.name})")
            return result
        except Exception as e:
            logger.error(f"决策过程出错: {e}")
            return DecisionResult(
                decision_type=decision_type,
                decision=self._get_safe_decision(decision_type),
                confidence=DecisionConfidence.VERY_LOW,
                reasoning=f"决策过程出错: {e}",
                factors=factors or DecisionFactors()
            )

    async def _calculate_factors(self, context: DecisionContext) -> DecisionFactors:
        factors = DecisionFactors()
        try:
            factors.confidence_score = self._get_confidence_score(context)
            factors.risk_score = self._get_risk_score(context)
            factors.interest_score = await self._get_interest_score(context)
            factors.urgency_score = self._calculate_urgency(context)
            factors.social_score = self._calculate_social_factor(context)
            factors.mood_score = self._calculate_mood_factor(context)
        except Exception as e:
            logger.warning(f"计算决策因素出错: {e}")
        return factors

    def _get_confidence_score(self, context: DecisionContext) -> float:
        try:
            from src.modules.brain.confidence_calculator import get_confidence_calculator
            calculator = get_confidence_calculator()
            return calculator.calculate_confidence(context)
        except (ImportError, Exception):
            return self._calculate_default_confidence(context)

    def _calculate_default_confidence(self, context: DecisionContext) -> float:
        base_confidence = 0.6
        content_length = len(context.message_content)
        if content_length < 10:
            base_confidence -= 0.1
        elif content_length > 100:
            base_confidence += 0.1
        if isinstance(context.user_profile, dict):
            user_relationship = context.user_profile.get('relationship_score', 0.5)
        else:
            user_relationship = 0.5
        base_confidence += (user_relationship - 0.5) * 0.2
        return max(0.1, min(1.0, base_confidence))

    def _get_risk_score(self, context: DecisionContext) -> float:
        try:
            from src.modules.brain.risk_prediction_system import get_risk_predictor
            predictor = get_risk_predictor()
            return predictor.predict_risk(context)
        except (ImportError, Exception):
            return 0.3

    async def _get_interest_score(self, context: DecisionContext) -> float:
        try:
            from src.chat.focus_chat.expressors.interest_calculator import InterestCalculator
            calc = InterestCalculator()
            return 0.5
        except (ImportError, Exception):
            return 0.5

    def _calculate_urgency(self, context: DecisionContext) -> float:
        urgency = 0.5
        urgent_keywords = ['紧急', '急', '快', '现在', '马上', 'urgent', 'asap']
        content_lower = context.message_content.lower()
        for keyword in urgent_keywords:
            if keyword in content_lower:
                urgency += 0.3
                break
        if '?' in context.message_content or '？' in context.message_content:
            urgency += 0.1
        current_hour = time.localtime().tm_hour
        if 9 <= current_hour <= 17:
            urgency += 0.1
        elif 22 <= current_hour or current_hour <= 6:
            urgency -= 0.1
        return max(0.0, min(1.0, urgency))

    def _calculate_social_factor(self, context: DecisionContext) -> float:
        social = 0.5
        if isinstance(context.user_profile, dict):
            relationship = context.user_profile.get('relationship_score', 0.5)
        else:
            relationship = 0.5
        social += (relationship - 0.5) * 0.4
        if context.social_context:
            group_activity = context.social_context.get('activity_level', 0.5)
            social += (group_activity - 0.5) * 0.3
        return max(0.0, min(1.0, social))

    def _calculate_mood_factor(self, context: DecisionContext) -> float:
        mood_map = {
            'happy': 0.8, 'excited': 0.9, 'neutral': 0.5,
            'sad': 0.3, 'angry': 0.2
        }
        mood = context.current_mood.lower()
        return mood_map.get(mood, 0.5)

    def _decide_whether_to_reply(self, context: DecisionContext, factors: DecisionFactors) -> bool:
        is_targeted = False
        is_flagged = False
        if context.social_context:
            is_targeted = context.social_context.get("is_targeted", False)
            is_flagged = context.social_context.get("is_flagged", False)
        if is_targeted or is_flagged:
            return True
        if factors.urgency_score > 0.8:
            return True
        overall_score = self._calculate_weighted_score(factors)
        try:
            from src.config.config import global_config
            reply_threshold = getattr(global_config.chat, 'normal_desire_probability', 0.15)
        except Exception:
            reply_threshold = 0.15
        return overall_score > reply_threshold

    def _decide_content_type(self, context: DecisionContext, factors: DecisionFactors) -> str:
        if factors.interest_score > 0.8:
            return "enthusiastic"
        elif factors.risk_score > 0.7:
            return "cautious"
        elif factors.social_score > 0.8:
            return "social"
        return "normal"

    def _decide_strategy(self, context: DecisionContext, factors: DecisionFactors) -> str:
        strategies = ["direct", "humorous", "empathetic", "informative", "questioning"]
        if factors.interest_score > 0.8:
            return "enthusiastic"
        elif factors.risk_score > 0.7:
            return "safe"
        elif factors.social_score > 0.8:
            return "engaging"
        return random.choice(strategies)

    def _decide_timing(self, context: DecisionContext, factors: DecisionFactors) -> str:
        if factors.urgency_score > 0.8:
            return "immediate"
        elif factors.confidence_score < 0.4:
            return "delayed"
        return "normal"

    def _decide_intensity(self, context: DecisionContext, factors: DecisionFactors) -> str:
        overall_score = self._calculate_weighted_score(factors)
        if overall_score > 0.8:
            return "high"
        elif overall_score > 0.6:
            return "medium"
        elif overall_score > 0.3:
            return "low"
        return "minimal"

    def _make_general_decision(self, context: DecisionContext, factors: DecisionFactors) -> Any:
        overall_score = self._calculate_weighted_score(factors)
        if overall_score > 0.7:
            return "positive"
        elif overall_score > 0.4:
            return "neutral"
        return "negative"

    def _calculate_weighted_score(self, factors: DecisionFactors) -> float:
        score = (
            factors.confidence_score * self._weights['confidence'] +
            (1 - factors.risk_score) * self._weights['risk'] +
            factors.interest_score * self._weights['interest'] +
            factors.urgency_score * self._weights['urgency'] +
            factors.social_score * self._weights['social'] +
            factors.mood_score * self._weights['mood']
        )
        return max(0.0, min(1.0, score))

    def _calculate_overall_confidence(self, factors: DecisionFactors) -> DecisionConfidence:
        overall_score = self._calculate_weighted_score(factors)
        for confidence, threshold in sorted(self._thresholds.items(), key=lambda x: x[1], reverse=True):
            if overall_score >= threshold:
                return confidence
        return DecisionConfidence.VERY_LOW

    def _generate_reasoning(self, decision_type: DecisionType, decision: Any, factors: DecisionFactors) -> str:
        parts = []
        if factors.confidence_score > 0.8:
            parts.append("置信度很高")
        elif factors.confidence_score < 0.4:
            parts.append("置信度较低")
        if factors.interest_score > 0.7:
            parts.append("对内容很感兴趣")
        elif factors.interest_score < 0.3:
            parts.append("对内容兴趣不高")
        if factors.risk_score > 0.7:
            parts.append("存在较高风险")
        elif factors.risk_score < 0.3:
            parts.append("风险较低")
        return f"基于{', '.join(parts)}，决定{decision_type.value}为{decision}" if parts else f"决定{decision_type.value}为{decision}"

    def _generate_alternatives(self, decision_type: DecisionType, current_decision: Any) -> List[Dict]:
        if decision_type == DecisionType.REPLY:
            return [{"decision": True, "reason": "积极回复"}, {"decision": False, "reason": "保持沉默"}]
        elif decision_type == DecisionType.CONTENT:
            return [{"decision": "enthusiastic", "reason": "热情回复"}, {"decision": "cautious", "reason": "谨慎回复"}, {"decision": "normal", "reason": "正常回复"}]
        return []

    def _create_execution_plan(self, decision_type: DecisionType, decision: Any, context: DecisionContext) -> Dict[str, Any]:
        plan = {
            "decision_type": decision_type.value,
            "decision": decision,
            "stream_id": context.stream_id,
            "user_id": context.user_id,
            "timestamp": time.time()
        }
        if decision_type == DecisionType.REPLY and decision:
            plan["steps"] = ["analyze_context", "generate_content", "validate_response", "send_reply"]
        elif decision_type == DecisionType.TIMING:
            delay_map = {"immediate": 0, "delayed": 30, "normal": 5}
            plan["delay_seconds"] = delay_map.get(decision, 5)
        return plan

    def _get_safe_decision(self, decision_type: DecisionType) -> Any:
        safe_decisions = {
            DecisionType.REPLY: False,
            DecisionType.CONTENT: "normal",
            DecisionType.STRATEGY: "safe",
            DecisionType.TIMING: "normal",
            DecisionType.INTENSITY: "low"
        }
        return safe_decisions.get(decision_type, "neutral")

    def _record_decision(self, result: DecisionResult):
        self._decision_history.append(result)
        if len(self._decision_history) > self._max_history:
            self._decision_history = self._decision_history[-self._max_history:]

    def _update_engagement_state(self, should_reply: bool):
        try:
            from src.chat.focus_chat.expressors.interest_calculator import InterestCalculator
        except Exception:
            pass

    def get_decision_stats(self) -> Dict[str, Any]:
        if not self._decision_history:
            return {"total_decisions": 0}
        total_decisions = len(self._decision_history)
        confidence_counts = {}
        recent = self._decision_history[-100:]
        for decision in recent:
            confidence = decision.confidence.name
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        return {
            "total_decisions": total_decisions,
            "recent_confidence_distribution": confidence_counts,
            "average_confidence": sum(d.confidence.value for d in recent) / len(recent)
        }


_decision_brain: Optional[DecisionBrain] = None


def get_decision_brain() -> DecisionBrain:
    global _decision_brain
    if _decision_brain is None:
        _decision_brain = DecisionBrain()
    return _decision_brain

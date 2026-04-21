from enum import Enum
from dataclasses import dataclass, field
from collections import Counter
from typing import Dict, List, Optional, Any, Tuple, Iterable
import time
from src.common.logger import get_logger

logger = get_logger("决策大脑")


class DecisionType(Enum):
    REPLY = "reply"
    CONTENT = "content"
    STRATEGY = "strategy"
    TIMING = "timing"
    INTENSITY = "intensity"
    EMOTION = "emotion"
    ACTION = "action"
    TOPIC = "topic"


class ConfidenceLevel(Enum):
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 1.0


class RiskLevel(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DecisionFactor:
    name: str
    weight: float = 1.0
    value: float = 0.0
    confidence: float = 0.5
    source: str = "unknown"

    def compute_score(self) -> float:
        return self.value * self.weight * self.confidence


@dataclass
class DecisionContext:
    user_id: str = ""
    stream_id: str = ""
    message_content: str = ""
    conversation_history: List[Dict] = field(default_factory=list)
    emotion_state: Dict = field(default_factory=dict)
    relationship: str = "陌生人"
    recent_interactions: int = 0
    time_context: Dict = field(default_factory=dict)
    topic_context: Dict = field(default_factory=dict)
    social_context: Dict = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class DecisionResult:
    decision_type: DecisionType
    action: str
    confidence: float
    factors: Dict[str, float]
    risk_level: RiskLevel
    reasoning: str
    alternatives: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_type": self.decision_type.value,
            "action": self.action,
            "confidence": self.confidence,
            "factors": self.factors,
            "risk_level": self.risk_level.value,
            "reasoning": self.reasoning,
            "alternatives": self.alternatives,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class DecisionPatternSnapshot:
    top_patterns: List[str] = field(default_factory=list)
    matched_clues: int = 0
    novelty: float = 0.0


class AdaptiveThresholdSystem:
    """根据上下文动态调整阈值"""

    def __init__(self, config_engine=None):
        self._config = config_engine

    def resolve(
        self, key: str, baseline: float, context: Dict[str, Any] | None = None
    ) -> float:
        context = context or {}
        drift = float(context.get("drift", 0.0) or 0.0)
        floor = float(context.get("floor", baseline - 0.25))
        ceil = float(context.get("ceil", baseline + 0.25))
        intensity = float(context.get("intensity", 1.0) or 1.0)
        value = baseline + drift * intensity
        return max(floor, min(ceil, value))


class ConfidenceCalculator:
    """将多信号融合为统一置信度"""

    def calculate(self, scores: Iterable[float]) -> float:
        values = [max(0.0, min(1.0, float(s))) for s in scores]
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        spread = max(values) - min(values) if len(values) > 1 else 0.0
        agreement_bonus = max(0.0, 0.15 - spread * 0.15)
        return max(0.0, min(1.0, mean + agreement_bonus))


class PatternMiningSystem:
    """提取最近窗口中的高频模式"""

    def top_patterns(
        self, items: Iterable[str], limit: int = 5
    ) -> DecisionPatternSnapshot:
        normalized = [
            x.strip() for x in items if isinstance(x, str) and x.strip()
        ]
        if not normalized:
            return DecisionPatternSnapshot()
        counter = Counter(normalized)
        top = [name for name, _ in counter.most_common(max(1, int(limit)))]
        matched = sum(counter.values())
        novelty = 1.0 / max(1.0, float(counter.most_common(1)[0][1]))
        return DecisionPatternSnapshot(
            top_patterns=top, matched_clues=matched, novelty=novelty
        )


class RiskPredictionSystem:
    """把输入信号映射为风险值"""

    def predict(self, signal: Dict[str, float] | None = None) -> float:
        signal = signal or {}
        toxicity = max(
            0.0,
            min(
                1.0,
                float(
                    signal.get("toxicity", signal.get("risk_hint", 0.0)) or 0.0
                ),
            ),
        )
        volatility = max(
            0.0, min(1.0, float(signal.get("volatility", 0.0) or 0.0))
        )
        instability = max(
            0.0, min(1.0, float(signal.get("instability", 0.0) or 0.0))
        )
        social_penalty = max(
            0.0,
            1.0
            - max(
                0.0, min(1.0, float(signal.get("social_value", 0.5) or 0.5))
            ),
        )
        value = (
            toxicity * 0.45
            + volatility * 0.2
            + instability * 0.15
            + social_penalty * 0.2
        )
        return max(0.0, min(1.0, value))


class StrategyGenerationSystem:
    """基于风险与置信度给出候选策略"""

    def generate(
        self,
        *,
        risk: float,
        confidence: float,
        pattern_snapshot: DecisionPatternSnapshot | None = None,
    ) -> List[Dict[str, float | str]]:
        snapshot = pattern_snapshot or DecisionPatternSnapshot()
        options: List[Dict[str, float | str]] = []
        if risk >= 0.78:
            options.append({"name": "observe", "score": 0.9})
        else:
            reply_score = confidence * (1.0 - risk * 0.45)
            if snapshot.matched_clues:
                reply_score += min(0.1, snapshot.matched_clues * 0.02)
            options.append(
                {"name": "reply", "score": max(0.0, min(1.0, reply_score))}
            )
            topic_score = (
                confidence * 0.72 + snapshot.novelty * 0.1 - risk * 0.25
            )
            options.append(
                {
                    "name": "start_topic",
                    "score": max(0.0, min(1.0, topic_score)),
                }
            )
            observe_score = 0.45 + risk * 0.35 - confidence * 0.1
            options.append(
                {"name": "observe", "score": max(0.0, min(1.0, observe_score))}
            )
        return sorted(
            options,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )


class ScoringSystem:
    def __init__(self):
        self._weights = {
            "confidence": 0.25,
            "risk": 0.20,
            "interest": 0.20,
            "urgency": 0.15,
            "social": 0.10,
            "mood": 0.10,
        }
        self._thresholds = {
            "high_confidence": 0.7,
            "medium_confidence": 0.5,
            "low_confidence": 0.3,
        }

    def compute_score(self, factors: Dict[str, DecisionFactor]) -> float:
        if not factors:
            return 0.5
        total_weight = sum(f.weight for f in factors.values())
        if total_weight == 0:
            return 0.5
        weighted_sum = sum(f.compute_score() for f in factors.values())
        return min(1.0, max(0.0, weighted_sum / total_weight))

    def compute_confidence_level(self, score: float) -> ConfidenceLevel:
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.6:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.VERY_LOW

    def assess_risk(
        self, factors: Dict[str, DecisionFactor], context: DecisionContext
    ) -> RiskLevel:
        risk_score = 0.0
        if "risk" in factors:
            risk_score += factors["risk"].value * 0.4
        emotion_state = context.emotion_state
        if emotion_state:
            trauma = emotion_state.get("trauma_score", 0)
            if trauma > 5:
                risk_score += 0.3
            elif trauma > 3:
                risk_score += 0.15
            annoyance = emotion_state.get("annoyance", 0)
            if annoyance > 60:
                risk_score += 0.2
            elif annoyance > 40:
                risk_score += 0.1
        if context.constraints:
            risk_score += min(0.2, len(context.constraints) * 0.05)
        if risk_score >= 0.7:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.5:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        elif risk_score >= 0.1:
            return RiskLevel.LOW
        return RiskLevel.MINIMAL

    def rank_alternatives(
        self, alternatives: List[Dict], factors: Dict[str, DecisionFactor]
    ) -> List[Dict]:
        if not alternatives:
            return []
        scored = []
        for alt in alternatives:
            score = self.compute_score(factors)
            alt_score = alt.get("base_score", 0.5)
            combined = score * 0.6 + alt_score * 0.4
            scored.append({**alt, "final_score": combined})
        scored.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return scored


class GroupDecision:
    def __init__(self):
        self._group_context: Dict[str, Dict] = {}
        self._member_states: Dict[str, Dict] = {}
        self._conversation_flow: List[Dict] = []

    def update_group_context(self, stream_id: str, context: Dict):
        self._group_context[stream_id] = context

    def update_member_state(self, user_id: str, state: Dict):
        self._member_states[user_id] = state

    def add_conversation_turn(self, turn: Dict):
        self._conversation_flow.append(turn)
        if len(self._conversation_flow) > 50:
            self._conversation_flow = self._conversation_flow[-30:]

    def should_reply(self, context: DecisionContext) -> Tuple[bool, float]:
        if not context.message_content:
            return False, 0.0
        mentioned = self._check_mentioned(context.message_content)
        if mentioned:
            return True, 0.9
        interest_score = self._compute_interest_score(context)
        if interest_score > 0.7:
            return True, interest_score
        silence_duration = context.time_context.get("silence_duration", 0)
        if silence_duration > 300:
            return True, 0.6
        recent_activity = len(
            [
                t
                for t in self._conversation_flow
                if __import__("time").time() - t.get("time", 0) < 300
            ]
        )
        if recent_activity < 3 and interest_score > 0.4:
            return True, interest_score * 0.8
        return interest_score > 0.5, interest_score

    def _check_mentioned(self, content: str) -> bool:
        try:
            from src.chat.heart_flow.config_loader import get_mention_patterns

            mention_patterns = get_mention_patterns()
        except Exception:
            mention_patterns = []
        return any(p in content for p in mention_patterns)

    def _compute_interest_score(self, context: DecisionContext) -> float:
        score = 0.5
        emotion = context.emotion_state
        if emotion:
            affection = emotion.get("affection", 0)
            score += affection / 200.0
            trust = emotion.get("trust", 0)
            score += trust / 200.0
        relationship = context.relationship
        try:
            from src.modules.brain.config_loader import (
                is_intimate_relationship,
                is_familiar_relationship,
            )

            if is_intimate_relationship(relationship):
                score += 0.2
            elif is_familiar_relationship(relationship):
                score += 0.1
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return min(1.0, max(0.0, score))

    def select_target_user(
        self, stream_id: str, candidates: List[str]
    ) -> Optional[str]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        scores = {}
        for user_id in candidates:
            state = self._member_states.get(user_id, {})
            try:
                score = float(state.get("affection", 0)) / 100.0
                score += float(state.get("interaction_count", 0)) / 1000.0
            except (TypeError, ValueError):
                score = 0.0
            scores[user_id] = score
        return max(scores.items(), key=lambda x: x[1])[0]


class StickerDecision:
    def __init__(self):
        self._sticker_history: List[str] = []
        self._max_history = 20

    def should_use_sticker(
        self, context: DecisionContext
    ) -> Tuple[bool, float]:
        emotion = context.emotion_state
        if not emotion:
            return False, 0.0
        dominant, intensity = emotion.get("dominant_emotion", ("neutral", 0.0))
        if isinstance(dominant, tuple):
            dominant, intensity = dominant
        try:
            from src.modules.brain.config_loader import get_high_emotion_types

            high_emotion_types = get_high_emotion_types()
        except Exception:
            high_emotion_types = []
        if dominant in high_emotion_types and abs(float(intensity)) > 30:
            return True, 0.8
        relationship = context.relationship
        try:
            from src.modules.brain.config_loader import (
                is_intimate_relationship,
            )

            if is_intimate_relationship(relationship):
                return True, 0.6
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        recent_stickers = len(
            [
                s
                for s in self._sticker_history
                if __import__("time").time()
                - float(s.split("_")[-1] if "_" in s else 0)
                < 300
            ]
        )
        if recent_stickers < 3:
            return True, 0.5
        return False, 0.3

    def select_sticker_type(self, emotion_state: Dict) -> str:
        dominant, intensity = emotion_state.get(
            "dominant_emotion", ("neutral", 0.0)
        )
        if isinstance(dominant, tuple):
            dominant, intensity = dominant
        dominant = str(dominant).lower()
        intensity = float(intensity)
        if dominant == "joy" and intensity > 50:
            return "excited"
        elif dominant == "joy":
            return "happy"
        elif dominant == "sadness":
            return "sad"
        elif dominant == "anger":
            return "angry"
        elif dominant == "surprise":
            return "surprised"
        elif dominant == "fear":
            return "scared"
        return "neutral"

    def record_sticker_use(self, sticker_id: str):
        timestamp = __import__("time").time()
        self._sticker_history.append(f"{sticker_id}_{timestamp}")
        if len(self._sticker_history) > self._max_history:
            self._sticker_history = self._sticker_history[-self._max_history:]


class RecallMonitor:
    def __init__(self):
        self._recall_history: List[Dict] = []
        self._max_history = 100

    def should_recall(self, context: DecisionContext) -> Tuple[bool, str]:
        content = context.message_content.lower()
        try:
            from src.modules.brain.config_loader import get_recall_keywords

            recall_keywords = get_recall_keywords()
        except Exception:
            recall_keywords = []
        for kw in recall_keywords:
            if kw in content:
                return True, f"检测到回忆关键词: {kw}"
        emotion = context.emotion_state
        if emotion:
            trauma = emotion.get("trauma_score", 0)
            if trauma > 3:
                return True, "创伤相关记忆触发"
        relationship = context.relationship
        try:
            from src.modules.brain.config_loader import (
                is_intimate_relationship,
            )

            if is_intimate_relationship(relationship):
                interaction_count = (
                    emotion.get("interaction_count", 0) if emotion else 0
                )
                if interaction_count > 50:
                    return True, "深度关系记忆检索"
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return False, ""

    def record_recall(self, recall_data: Dict):
        self._recall_history.append(recall_data)
        if len(self._recall_history) > self._max_history:
            self._recall_history = self._recall_history[-self._max_history:]


class DecisionBrain:
    def __init__(self):
        self.scoring = ScoringSystem()
        self.group = GroupDecision()
        self.sticker = StickerDecision()
        self.recall = RecallMonitor()
        self._thresholds = AdaptiveThresholdSystem()
        self._confidence_calc = ConfidenceCalculator()
        self._patterns = PatternMiningSystem()
        self._risk = RiskPredictionSystem()
        self._strategy = StrategyGenerationSystem()
        self._decision_history: List[DecisionResult] = []
        self._max_history = 200
        self._llm_bridge = None
        self._emotion_tracker = None
        self._last_snapshot: Dict[str, Any] = {
            "top_patterns": [],
            "matched_clues": 0,
            "novelty": 0.0,
            "updated_at": 0.0,
        }

    def set_llm_bridge(self, bridge):
        self._llm_bridge = bridge

    def set_emotion_tracker(self, tracker):
        self._emotion_tracker = tracker

    async def make_decision(
        self,
        decision_type: DecisionType,
        context: DecisionContext,
        factors: Optional[Dict[str, DecisionFactor]] = None,
    ) -> DecisionResult:
        if factors is None:
            factors = self._extract_factors(context)
        score = self.scoring.compute_score(factors)
        confidence = self.scoring.compute_confidence_level(score)
        risk = self.scoring.assess_risk(factors, context)
        action = await self._determine_action(
            decision_type, context, factors, score
        )
        reasoning = self._generate_reasoning(
            decision_type, factors, score, risk
        )
        alternatives = self._generate_alternatives(
            decision_type, context, factors
        )
        result = DecisionResult(
            decision_type=decision_type,
            action=action,
            confidence=score,
            factors={k: v.value for k, v in factors.items()},
            risk_level=risk,
            reasoning=reasoning,
            alternatives=alternatives[:3],
            metadata={"confidence_level": confidence.value},
        )
        self._record_decision(result)
        return result

    def _extract_factors(
        self, context: DecisionContext
    ) -> Dict[str, DecisionFactor]:
        factors = {}
        emotion = context.emotion_state
        if emotion:
            try:
                trust_val = float(emotion.get("trust", 0))
            except (TypeError, ValueError):
                trust_val = 0.0
            try:
                affection_val = float(emotion.get("affection", 0))
            except (TypeError, ValueError):
                affection_val = 0.0
            factors["confidence"] = DecisionFactor(
                name="confidence",
                weight=0.25,
                value=min(1.0, (trust_val + 50) / 100),
                confidence=0.8,
                source="emotion_state",
            )
            factors["mood"] = DecisionFactor(
                name="mood",
                weight=0.10,
                value=(affection_val + 100) / 200,
                confidence=0.7,
                source="emotion_state",
            )
        factors["interest"] = DecisionFactor(
            name="interest",
            weight=0.20,
            value=self._compute_interest(context),
            confidence=0.6,
            source="content_analysis",
        )
        factors["urgency"] = DecisionFactor(
            name="urgency",
            weight=0.15,
            value=self._compute_urgency(context),
            confidence=0.7,
            source="time_context",
        )
        factors["social"] = DecisionFactor(
            name="social",
            weight=0.10,
            value=self._compute_social_factor(context),
            confidence=0.6,
            source="social_context",
        )
        return factors

    def _compute_interest(self, context: DecisionContext) -> float:
        score = 0.5
        content = context.message_content
        if len(content) > 100:
            score += 0.1
        question_marks = content.count("?") + content.count("？")
        if question_marks > 0:
            score += 0.15
        emotion = context.emotion_state
        if emotion:
            affection = emotion.get("affection", 0)
            score += max(0, affection / 200)
        return min(1.0, score)

    def _compute_urgency(self, context: DecisionContext) -> float:
        time_ctx = context.time_context
        if not time_ctx:
            return 0.5
        silence = time_ctx.get("silence_duration", 0)
        if silence > 600:
            return 0.8
        elif silence > 300:
            return 0.6
        elif silence > 60:
            return 0.4
        return 0.3

    def _compute_social_factor(self, context: DecisionContext) -> float:
        relationship = context.relationship
        try:
            from src.modules.brain.config_loader import get_relationship_types

            rel_types = get_relationship_types()
            if relationship in rel_types.get("intimate", []):
                return 0.9
            elif relationship in rel_types.get("familiar", []):
                return 0.7
            elif relationship in rel_types.get("stranger", []):
                return 0.3
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return 0.4

    async def _determine_action(
        self,
        decision_type: DecisionType,
        context: DecisionContext,
        factors: Dict[str, DecisionFactor],
        score: float,
    ) -> str:
        if decision_type == DecisionType.REPLY:
            return await self._determine_reply_action(context, score)
        elif decision_type == DecisionType.CONTENT:
            return await self._determine_content_action(context, score)
        elif decision_type == DecisionType.STRATEGY:
            return self._determine_strategy_action(context, score)
        elif decision_type == DecisionType.TIMING:
            return self._determine_timing_action(context, score)
        elif decision_type == DecisionType.INTENSITY:
            return self._determine_intensity_action(context, score)
        elif decision_type == DecisionType.EMOTION:
            return self._determine_emotion_action(context, score)
        return "default"

    async def _determine_reply_action(
        self, context: DecisionContext, score: float
    ) -> str:
        if score >= 0.7:
            return "immediate_reply"
        elif score >= 0.5:
            return "delayed_reply"
        elif score >= 0.3:
            return "consider_reply"
        return "skip_reply"

    async def _determine_content_action(
        self, context: DecisionContext, score: float
    ) -> str:
        emotion = context.emotion_state
        if emotion:
            dominant, _ = emotion.get("dominant_emotion", ("neutral", 0.0))
            if isinstance(dominant, tuple):
                dominant = dominant[0]
            if str(dominant) == "joy":
                return "enthusiastic_response"
            elif str(dominant) == "sadness":
                return "comforting_response"
            elif str(dominant) == "anger":
                return "calming_response"
        if score >= 0.6:
            return "detailed_response"
        return "brief_response"

    def _determine_strategy_action(
        self, context: DecisionContext, score: float
    ) -> str:
        relationship = context.relationship
        try:
            from src.modules.brain.config_loader import (
                is_intimate_relationship,
                is_familiar_relationship,
            )

            if is_intimate_relationship(relationship):
                return "intimate_strategy"
            elif is_familiar_relationship(relationship):
                return "friendly_strategy"
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return "polite_strategy"

    def _determine_timing_action(
        self, context: DecisionContext, score: float
    ) -> str:
        time_ctx = context.time_context
        if time_ctx:
            silence = time_ctx.get("silence_duration", 0)
            if silence > 300:
                return "proactive_timing"
        if score >= 0.7:
            return "immediate_timing"
        elif score >= 0.4:
            return "normal_timing"
        return "delayed_timing"

    def _determine_intensity_action(
        self, context: DecisionContext, score: float
    ) -> str:
        emotion = context.emotion_state
        if emotion:
            _, intensity = emotion.get("dominant_emotion", ("neutral", 0.0))
            if isinstance(intensity, tuple):
                intensity = intensity[1] if len(intensity) > 1 else 0
            if float(intensity) > 50:
                return "high_intensity"
            elif float(intensity) > 20:
                return "medium_intensity"
        return "low_intensity"

    def _determine_emotion_action(
        self, context: DecisionContext, score: float
    ) -> str:
        emotion = context.emotion_state
        if not emotion:
            return "neutral_emotion"
        dominant, _ = emotion.get("dominant_emotion", ("neutral", 0.0))
        if isinstance(dominant, tuple):
            dominant = dominant[0]
        return f"express_{str(dominant)}"

    def _generate_reasoning(
        self,
        decision_type: DecisionType,
        factors: Dict[str, DecisionFactor],
        score: float,
        risk: RiskLevel,
    ) -> str:
        top_factors = sorted(
            factors.items(), key=lambda x: x[1].compute_score(), reverse=True
        )[:3]
        factor_desc = ", ".join([f"{k}:{v.value:.2f}" for k, v in top_factors])
        return f"决策类型:{
            decision_type.value} | 置信度:{
            score:.2f} | 风险:{
            risk.value} | 关键因素:{factor_desc}"

    def _generate_alternatives(
        self,
        decision_type: DecisionType,
        context: DecisionContext,
        factors: Dict[str, DecisionFactor],
    ) -> List[Dict]:
        alternatives = []
        base_score = self.scoring.compute_score(factors)
        if decision_type == DecisionType.REPLY:
            alternatives = [
                {"action": "immediate_reply", "base_score": base_score},
                {"action": "delayed_reply", "base_score": base_score * 0.8},
                {"action": "skip_reply", "base_score": base_score * 0.3},
            ]
        elif decision_type == DecisionType.CONTENT:
            alternatives = [
                {"action": "detailed_response", "base_score": base_score},
                {"action": "brief_response", "base_score": base_score * 0.7},
                {"action": "emoji_only", "base_score": base_score * 0.5},
            ]
        return self.scoring.rank_alternatives(alternatives, factors)

    def _record_decision(self, result: DecisionResult):
        self._decision_history.append(result)
        if len(self._decision_history) > self._max_history:
            self._decision_history = self._decision_history[
                -self._max_history:
            ]

    def get_decision_history(self, limit: int = 20) -> List[Dict]:
        return [d.to_dict() for d in self._decision_history[-limit:]]

    def get_statistics(self) -> Dict[str, Any]:
        if not self._decision_history:
            return {"total": 0}
        type_counts = {}
        risk_counts = {}
        total_confidence = 0
        for d in self._decision_history:
            type_counts[d.decision_type.value] = (
                type_counts.get(d.decision_type.value, 0) + 1
            )
            risk_counts[d.risk_level.value] = (
                risk_counts.get(d.risk_level.value, 0) + 1
            )
            total_confidence += d.confidence
        return {
            "total": len(self._decision_history),
            "average_confidence": round(
                total_confidence / len(self._decision_history), 3
            ),
            "decision_types": type_counts,
            "risk_distribution": risk_counts,
        }

    def decide(
        self, signal: Dict[str, float], clues: List[str] | None = None
    ) -> Dict[str, float | str]:
        """综合决策方法 - 使用 5 个子系统"""
        clues = clues or []
        confidence = self._confidence_calc.calculate(signal.values())
        risk = self._risk.predict(signal)
        snapshot = self._patterns.top_patterns(clues)
        self._last_snapshot = {
            "top_patterns": snapshot.top_patterns,
            "matched_clues": snapshot.matched_clues,
            "novelty": snapshot.novelty,
            "updated_at": time.time(),
        }
        policy = self._strategy.generate(
            risk=risk, confidence=confidence, pattern_snapshot=snapshot
        )
        pick = policy[0] if policy else {"name": "observe", "score": 0.0}
        return {
            "action": str(pick.get("name", "observe")),
            "confidence": confidence,
            "risk": risk,
            "score": float(pick.get("score", 0.0)),
            "reply_threshold": self._thresholds.resolve(
                "reply",
                0.55,
                {
                    "drift": confidence - 0.5 - risk * 0.1,
                    "intensity": 1.0 + snapshot.matched_clues * 0.05,
                },
            ),
            "pattern_hits": float(snapshot.matched_clues),
            "novelty": snapshot.novelty,
        }

    def get_last_snapshot(self) -> Dict[str, Any]:
        return dict(self._last_snapshot)


_brain_instance: Optional[DecisionBrain] = None


def get_decision_brain() -> DecisionBrain:
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = DecisionBrain()
    return _brain_instance


def reset_decision_brain():
    global _brain_instance
    _brain_instance = None

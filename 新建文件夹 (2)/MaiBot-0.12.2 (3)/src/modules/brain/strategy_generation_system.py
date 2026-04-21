import time
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from src.common.logger import get_logger

logger = get_logger("strategy_gen")


class StrategyType(Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    QUESTIONING = "questioning"
    HUMOROUS = "humorous"
    EMPATHETIC = "empathetic"
    INFORMATIVE = "informative"
    DEFENSIVE = "defensive"
    ENGAGING = "engaging"
    SAFE = "safe"


@dataclass
class StrategyOption:
    strategy_type: StrategyType
    suitability_score: float = 0.0
    expected_outcome: str = ""
    risk_level: str = "medium"
    description: str = ""
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class StrategyRecommendation:
    recommended_strategy: StrategyType
    confidence_score: float = 0.0
    reasoning: str = ""
    alternatives: List[StrategyOption] = field(default_factory=list)
    execution_notes: List[str] = field(default_factory=list)
    success_probability: float = 0.0


class StrategyGenerationSystem:
    def __init__(self):
        self._strategy_history: List[StrategyRecommendation] = []
        self._max_history = 200
        self._evaluation_weights = {
            'confidence': 0.25, 'risk': 0.20, 'interest': 0.20,
            'social': 0.15, 'context': 0.10, 'personality': 0.10
        }
        self._strategy_templates = self._initialize_strategy_templates()

    def _initialize_strategy_templates(self) -> Dict[StrategyType, Dict[str, Any]]:
        return {
            StrategyType.DIRECT: {
                "description": "直接坦率的回复",
                "suitable_scenarios": ["fact_questions", "clear_requests", "professional_topics"],
                "risk_level": "low",
                "expected_outcome": "高效沟通",
                "prerequisites": ["clear_context", "high_confidence"]
            },
            StrategyType.INDIRECT: {
                "description": "委婉含蓄的回复",
                "suitable_scenarios": ["sensitive_topics", "unclear_intent", "emotional_context"],
                "risk_level": "medium",
                "expected_outcome": "避免冲突",
                "prerequisites": ["emotional_awareness"]
            },
            StrategyType.QUESTIONING: {
                "description": "通过提问引导对话",
                "suitable_scenarios": ["unclear_intent", "deep_topics", "exploratory_conversation"],
                "risk_level": "low",
                "expected_outcome": "深入了解",
                "prerequisites": ["curiosity"]
            },
            StrategyType.HUMOROUS: {
                "description": "使用幽默化解气氛",
                "suitable_scenarios": ["tense_situations", "casual_topics", "friendly_context"],
                "risk_level": "medium",
                "expected_outcome": "活跃气氛",
                "prerequisites": ["humor_sense"]
            },
            StrategyType.EMPATHETIC: {
                "description": "展现共情和理解",
                "suitable_scenarios": ["emotional_topics", "personal_issues", "support_needed"],
                "risk_level": "low",
                "expected_outcome": "建立信任",
                "prerequisites": ["emotional_intelligence"]
            },
            StrategyType.INFORMATIVE: {
                "description": "提供有用信息",
                "suitable_scenarios": ["knowledge_requests", "educational_topics", "factual_questions"],
                "risk_level": "low",
                "expected_outcome": "提供价值",
                "prerequisites": ["knowledge_base"]
            },
            StrategyType.DEFENSIVE: {
                "description": "保护性回复",
                "suitable_scenarios": ["controversial_topics", "high_risk_situations"],
                "risk_level": "low",
                "expected_outcome": "避免风险",
                "prerequisites": ["caution"]
            },
            StrategyType.ENGAGING: {
                "description": "积极互动引导深入对话",
                "suitable_scenarios": ["interesting_topics", "engaged_users", "positive_context"],
                "risk_level": "medium",
                "expected_outcome": "促进对话延续",
                "prerequisites": ["engagement_interest"]
            },
            StrategyType.SAFE: {
                "description": "保守安全的回复",
                "suitable_scenarios": ["unknown_users", "unclear_intent", "high_uncertainty"],
                "risk_level": "very_low",
                "expected_outcome": "避免风险",
                "prerequisites": ["caution"]
            }
        }

    def generate_strategy(self, context: Any) -> StrategyType:
        try:
            recommendation = self.evaluate_strategies(context)
            return recommendation.recommended_strategy
        except Exception:
            return StrategyType.SAFE

    def evaluate_strategies(self, context: Any) -> StrategyRecommendation:
        try:
            options = self._evaluate_all_strategies(context)
            if not options:
                return StrategyRecommendation(
                    recommended_strategy=StrategyType.SAFE,
                    confidence_score=0.8, reasoning="无合适策略",
                    success_probability=0.7
                )
            best = max(options, key=lambda s: s.suitability_score)
            confidence = self._calculate_confidence(best, context)
            reasoning = self._generate_reasoning(best, context)
            notes = self._generate_notes(best, context)
            success_prob = self._estimate_success(best, context)
            rec = StrategyRecommendation(
                recommended_strategy=best.strategy_type,
                confidence_score=confidence, reasoning=reasoning,
                alternatives=options[:3], execution_notes=notes,
                success_probability=success_prob
            )
            self._record_strategy(rec)
            return rec
        except Exception as e:
            return StrategyRecommendation(
                recommended_strategy=StrategyType.SAFE,
                confidence_score=0.5, reasoning=f"策略评估失败: {e}",
                success_probability=0.6
            )

    def _evaluate_all_strategies(self, context) -> List[StrategyOption]:
        options = []
        for stype, template in self._strategy_templates.items():
            score = self._evaluate_suitability(stype, template, context)
            if score > 0:
                options.append(StrategyOption(
                    strategy_type=stype, suitability_score=score,
                    expected_outcome=template.get("expected_outcome", ""),
                    risk_level=template.get("risk_level", "medium"),
                    description=template.get("description", ""),
                    prerequisites=template.get("prerequisites", [])
                ))
        options.sort(key=lambda s: s.suitability_score, reverse=True)
        return options

    def _evaluate_suitability(self, stype: StrategyType, template: Dict, context) -> float:
        suitability = 0.5
        try:
            scenario_match = self._evaluate_scenario_match(stype, template, context)
            suitability += scenario_match * 0.3
            risk_match = self._evaluate_risk_match(template, context)
            suitability += risk_match * 0.2
            relationship_match = self._evaluate_relationship_match(stype, context)
            suitability += relationship_match * 0.2
            context_match = self._evaluate_context_match(stype, context)
            suitability += context_match * 0.2
            timing_match = self._evaluate_timing_match(stype, context)
            suitability += timing_match * 0.1
        except Exception:
            pass
        return max(0.0, min(1.0, suitability))

    def _evaluate_scenario_match(self, stype: StrategyType, template: Dict, context) -> float:
        suitable = template.get("suitable_scenarios", [])
        if not suitable:
            return 0.5
        current = self._analyze_scenario(context)
        match_count = sum(1 for s in suitable if s in current)
        return min(1.0, match_count / len(suitable)) if suitable else 0.5

    def _analyze_scenario(self, context) -> List[str]:
        scenarios = []
        if hasattr(context, 'message_content'):
            content = context.message_content.lower()
            if '?' in content or '？' in content:
                scenarios.append("fact_questions")
                if any(w in content for w in ['怎么', '如何', '为什么']):
                    scenarios.append("deep_topics")
            if any(w in content for w in ['难过', '开心', '生气']):
                scenarios.append("emotional_topics")
            if any(w in content for w in ['游戏', '电影', '音乐', '动漫']):
                scenarios.append("casual_topics")
        if hasattr(context, 'user_profile') and isinstance(context.user_profile, dict):
            rel = context.user_profile.get('relationship_score', 0.5)
            if rel > 0.8:
                scenarios.append("engaged_users")
            elif rel < 0.3:
                scenarios.append("unknown_users")
        if hasattr(context, 'conversation_history'):
            if len(context.conversation_history) < 3:
                scenarios.append("unclear_intent")
            elif len(context.conversation_history) > 10:
                scenarios.append("exploratory_conversation")
        return scenarios

    def _evaluate_risk_match(self, template: Dict, context) -> float:
        risk_levels = {"very_low": 0.1, "low": 0.3, "medium": 0.5, "high": 0.7}
        strategy_risk = risk_levels.get(template.get("risk_level", "medium"), 0.5)
        current_risk = 0.3
        try:
            from src.modules.brain.risk_prediction_system import get_risk_predictor
            current_risk = get_risk_predictor().predict_risk(context)
        except Exception:
            pass
        return max(0.0, 1.0 - abs(strategy_risk - current_risk))

    def _evaluate_relationship_match(self, stype: StrategyType, context) -> float:
        if not hasattr(context, 'user_profile'):
            return 0.5
        profile = context.user_profile if isinstance(context.user_profile, dict) else {}
        rel = profile.get('relationship_score', 0.5)
        weights = {
            StrategyType.DIRECT: 0.7, StrategyType.INDIRECT: 0.5,
            StrategyType.QUESTIONING: 0.6, StrategyType.HUMOROUS: 0.8,
            StrategyType.EMPATHETIC: 0.6, StrategyType.INFORMATIVE: 0.4,
            StrategyType.DEFENSIVE: 0.3, StrategyType.ENGAGING: 0.8,
            StrategyType.SAFE: 0.2
        }
        required = weights.get(stype, 0.5)
        if rel >= required:
            return 1.0
        return max(0.0, 1.0 - (required - rel) * 2)

    def _evaluate_context_match(self, stype: StrategyType, context) -> float:
        complexity = self._analyze_context_complexity(context)
        pref = {
            StrategyType.DIRECT: 0.3, StrategyType.INDIRECT: 0.6,
            StrategyType.QUESTIONING: 0.7, StrategyType.HUMOROUS: 0.5,
            StrategyType.EMPATHETIC: 0.6, StrategyType.INFORMATIVE: 0.4,
            StrategyType.DEFENSIVE: 0.8, StrategyType.ENGAGING: 0.6,
            StrategyType.SAFE: 0.9
        }
        return 1.0 - abs(complexity - pref.get(stype, 0.5))

    def _analyze_context_complexity(self, context) -> float:
        """分析上下文复杂度"""
        complexity = 0.5
        if hasattr(context, 'message_content'):
            length = len(context.message_content)
            if length < 20:
                complexity -= 0.2
            elif length > 200:
                complexity += 0.2
        if hasattr(context, 'conversation_history'):
            history_length = len(context.conversation_history)
            if history_length < 3:
                complexity += 0.2
            elif history_length > 15:
                complexity -= 0.1
        if hasattr(context, 'current_mood'):
            mood = context.current_mood.lower()
            if mood in ['confused', 'mixed', 'complex']:
                complexity += 0.2
        return max(0.0, min(1.0, complexity))

    def _analyze_context_certainty(self, context) -> float:
        """分析上下文确定性"""
        certainty = 0.8
        if not hasattr(context, 'message_content') or not context.message_content.strip():
            certainty -= 0.3
        if hasattr(context, 'conversation_history') and len(context.conversation_history) < 2:
            certainty -= 0.2
        if hasattr(context, 'user_profile') and context.user_profile:
            if isinstance(context.user_profile, dict):
                profile_completeness = len([v for v in context.user_profile.values() if v])
            elif hasattr(context.user_profile, 'to_dict'):
                profile_dict = context.user_profile.to_dict()
                profile_completeness = len([v for v in profile_dict.values() if v])
            else:
                profile_completeness = 0
            certainty += min(0.1, profile_completeness * 0.02)
        return max(0.2, min(1.0, certainty))

    def _evaluate_timing_match(self, stype: StrategyType, context) -> float:
        return 0.5

    def _calculate_confidence(self, best: StrategyOption, context) -> float:
        base = best.suitability_score
        hist_success = self._get_historical_success(best.strategy_type)
        base = base * 0.7 + hist_success * 0.3
        return max(0.1, min(0.95, base))

    def _get_historical_success(self, stype: StrategyType) -> float:
        if not self._strategy_history:
            return 0.6
        recent = [r for r in reversed(self._strategy_history) if r.recommended_strategy == stype][:10]
        if not recent:
            return 0.6
        return sum(r.success_probability for r in recent) / len(recent)

    def _generate_reasoning(self, best: StrategyOption, context) -> str:
        parts = []
        if best.suitability_score > 0.8:
            parts.append("适合度很高")
        elif best.suitability_score > 0.6:
            parts.append("适合度良好")
        scenarios = self._analyze_scenario(context)
        if scenarios:
            parts.append(f"适用场景: {', '.join(scenarios[:2])}")
        return f"选择{best.strategy_type.value}策略：{best.description}。{'；'.join(parts)}" if parts else f"选择{best.strategy_type.value}策略"

    def _generate_notes(self, best: StrategyOption, context) -> List[str]:
        notes = []
        if best.strategy_type == StrategyType.HUMOROUS:
            notes.append("确保幽默合适")
        elif best.strategy_type == StrategyType.EMPATHETIC:
            notes.append("展现真诚关心")
        elif best.strategy_type == StrategyType.DEFENSIVE:
            notes.append("保持中立立场")
        if hasattr(context, 'user_profile') and isinstance(context.user_profile, dict):
            if context.user_profile.get('relationship_score', 0.5) < 0.3:
                notes.append("用户关系较陌生，建议保守回复")
        return notes

    def _estimate_success(self, best: StrategyOption, context) -> float:
        base = best.suitability_score * 0.8 + 0.2
        hist = self._get_historical_success(best.strategy_type)
        base = base * 0.7 + hist * 0.3
        if hasattr(context, 'user_profile') and isinstance(context.user_profile, dict):
            rel = context.user_profile.get('relationship_score', 0.5)
            base += (rel - 0.5) * 0.2
        return max(0.1, min(0.95, base))

    def _record_strategy(self, rec: StrategyRecommendation):
        self._strategy_history.append(rec)
        if len(self._strategy_history) > self._max_history:
            self._strategy_history = self._strategy_history[-self._max_history:]

    def get_strategy_stats(self) -> Dict[str, Any]:
        if not self._strategy_history:
            return {"total_recommendations": 0}
        total = len(self._strategy_history)
        recent = self._strategy_history[-100:]
        counts: Dict[str, int] = {}
        for r in recent:
            s = r.recommended_strategy.value
            counts[s] = counts.get(s, 0) + 1
        return {
            "total_recommendations": total,
            "recent_strategy_distribution": counts,
            "average_confidence": sum(r.confidence_score for r in recent) / len(recent),
            "average_success_probability": sum(r.success_probability for r in recent) / len(recent)
        }


_strategy_generator: Optional[StrategyGenerationSystem] = None


def get_strategy_generator() -> StrategyGenerationSystem:
    global _strategy_generator
    if _strategy_generator is None:
        _strategy_generator = StrategyGenerationSystem()
    return _strategy_generator

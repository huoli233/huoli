import time
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("confidence_calc")


@dataclass
class ConfidenceFactors:
    data_quality: float = 0.5
    context_clarity: float = 0.5
    pattern_recognition: float = 0.5
    historical_success: float = 0.5
    uncertainty_level: float = 0.5
    time_pressure: float = 0.5


@dataclass
class ConfidenceResult:
    score: float = 0.5
    confidence_level: str = "medium"
    factors: ConfidenceFactors = field(default_factory=ConfidenceFactors)
    reasoning: str = ""
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class ConfidenceCalculator:
    def __init__(self):
        self._confidence_history: List[ConfidenceResult] = []
        self._max_history = 1000
        self._factor_weights = {
            'data_quality': 0.20,
            'context_clarity': 0.25,
            'pattern_recognition': 0.20,
            'historical_success': 0.15,
            'uncertainty_level': 0.10,
            'time_pressure': 0.10
        }
        self._confidence_levels = {
            'very_low': 0.2, 'low': 0.4, 'medium': 0.6,
            'high': 0.8, 'very_high': 0.9
        }
        try:
            from src.config.behavior_tuner import get_behavior_tuner
            self._learning_rate = get_behavior_tuner().get("confidence_learning_rate", 0.01)
        except Exception:
            self._learning_rate = 0.01
        self._confidence_adjustments: Dict[str, float] = {}
        logger.info("置信度计算器初始化完成")

    def calculate_confidence(self, context: Any) -> float:
        try:
            factors = self._analyze_factors(context)
            confidence_score = self._calculate_weighted_score(factors)
            adjusted_score = self._apply_learning_adjustments(confidence_score, context)
            final_score = max(0.0, min(1.0, adjusted_score))
            result = ConfidenceResult(
                score=final_score,
                confidence_level=self._get_confidence_level(final_score),
                factors=factors,
                reasoning=self._generate_reasoning(factors),
                recommendations=self._generate_recommendations(factors)
            )
            self._record_result(result)
            return final_score
        except Exception as e:
            logger.error(f"置信度计算出错: {e}")
            return 0.5

    def _analyze_factors(self, context) -> ConfidenceFactors:
        factors = ConfidenceFactors()
        try:
            factors.data_quality = self._assess_data_quality(context)
            factors.context_clarity = self._assess_context_clarity(context)
            factors.pattern_recognition = self._assess_pattern_recognition(context)
            factors.historical_success = self._assess_historical_success(context)
            factors.uncertainty_level = self._assess_uncertainty(context)
            factors.time_pressure = self._assess_time_pressure(context)
        except Exception as e:
            logger.warning(f"分析置信度因素出错: {e}")
        return factors

    def _assess_data_quality(self, context) -> float:
        quality = 0.5
        if hasattr(context, 'message_content'):
            content = context.message_content
            content_length = len(content.strip())
            if content_length < 5:
                quality -= 0.2
            elif content_length > 200:
                quality += 0.1
            if '?' in content or '？' in content:
                quality += 0.1
            special_chars = sum(1 for c in content if not c.isalnum() and not c.isspace() and c not in '，。！？、：；""''')
            if special_chars / max(1, content_length) > 0.3:
                quality -= 0.1
        if hasattr(context, 'user_profile') and context.user_profile:
            if isinstance(context.user_profile, dict):
                profile_completeness = len([v for v in context.user_profile.values() if v])
            else:
                profile_completeness = 0
            quality += min(0.2, profile_completeness * 0.05)
        return max(0.0, min(1.0, quality))

    def _assess_context_clarity(self, context) -> float:
        clarity = 0.5
        if hasattr(context, 'conversation_history'):
            history = context.conversation_history
            if len(history) == 0:
                clarity -= 0.2
            elif len(history) >= 5:
                clarity += 0.1
            if len(history) >= 3:
                topics = []
                for msg in history[-3:]:
                    if isinstance(msg, dict) and 'content' in msg:
                        topics.append(msg['content'][:50])
                if len(set(topics)) == 1:
                    clarity += 0.1
                elif len(set(topics)) == len(topics):
                    clarity -= 0.1
        if hasattr(context, 'time_context') and context.time_context:
            if context.time_context.get('is_business_hours', False):
                clarity += 0.05
        return max(0.0, min(1.0, clarity))

    def _assess_pattern_recognition(self, context) -> float:
        try:
            from src.modules.brain.pattern_mining_system import get_pattern_miner
            miner = get_pattern_miner()
            return miner.analyze_patterns(context)
        except (ImportError, Exception):
            return self._default_pattern_recognition(context)

    def _default_pattern_recognition(self, context) -> float:
        recognition = 0.5
        if hasattr(context, 'message_content'):
            content = context.message_content.lower()
            greetings = ['你好', 'hello', 'hi', '早上好', '晚上好']
            if any(g in content for g in greetings):
                recognition += 0.1
            if '?' in content or '？' in content:
                recognition += 0.1
            commands = ['帮我', '请', '能不能', '可以吗']
            if any(cmd in content for cmd in commands):
                recognition += 0.1
        return max(0.0, min(1.0, recognition))

    def _assess_historical_success(self, context) -> float:
        success_rate = 0.5
        if hasattr(context, 'stream_id') and hasattr(context, 'user_id'):
            similar = self._find_similar_decisions(context)
            if similar:
                total = len(similar)
                successful = sum(1 for d in similar if d.get('success', False))
                if total > 0:
                    success_rate = 0.4 + (successful / total) * 0.6
        return max(0.0, min(1.0, success_rate))

    def _assess_uncertainty(self, context) -> float:
        uncertainty = 0.5
        if hasattr(context, 'message_content'):
            content = context.message_content
            uncertain_words = ['可能', '大概', '也许', '或许', '不清楚', '不知道']
            uncertain_count = sum(1 for w in uncertain_words if w in content)
            uncertainty += min(0.3, uncertain_count * 0.1)
            if '还是' in content or '或者' in content:
                uncertainty += 0.1
        if hasattr(context, 'conversation_history'):
            if len(context.conversation_history) < 2:
                uncertainty += 0.1
        return max(0.0, min(1.0, uncertainty))

    def _assess_time_pressure(self, context) -> float:
        pressure = 0.5
        if hasattr(context, 'message_content'):
            content = context.message_content.lower()
            urgent_keywords = ['紧急', '急', '快', '现在', '马上', 'urgent', 'asap']
            if any(kw in content for kw in urgent_keywords):
                pressure += 0.3
        if hasattr(context, 'time_context') and context.time_context:
            if context.time_context.get('is_business_hours', False):
                pressure += 0.1
            deadline = context.time_context.get('response_deadline')
            if deadline:
                time_left = deadline - time.time()
                if time_left < 60:
                    pressure += 0.2
                elif time_left < 300:
                    pressure += 0.1
        return max(0.0, min(1.0, pressure))

    def _calculate_weighted_score(self, factors: ConfidenceFactors) -> float:
        score = (
            factors.data_quality * self._factor_weights['data_quality'] +
            factors.context_clarity * self._factor_weights['context_clarity'] +
            factors.pattern_recognition * self._factor_weights['pattern_recognition'] +
            factors.historical_success * self._factor_weights['historical_success'] +
            (1 - factors.uncertainty_level) * self._factor_weights['uncertainty_level'] +
            factors.time_pressure * self._factor_weights['time_pressure']
        )
        return max(0.0, min(1.0, score))

    def _apply_learning_adjustments(self, base_score: float, context) -> float:
        sig = self._generate_context_signature(context)
        adj = self._confidence_adjustments.get(sig, 0.0)
        return max(0.0, min(1.0, base_score + adj * self._learning_rate))

    def _generate_context_signature(self, context) -> str:
        elements = []
        if hasattr(context, 'stream_id'):
            elements.append(f"s:{context.stream_id}")
        if hasattr(context, 'user_id'):
            elements.append(f"u:{context.user_id}")
        if hasattr(context, 'message_content'):
            elements.append(f"c:{hash(context.message_content[:100])}")
        return "|".join(elements)

    def _get_confidence_level(self, score: float) -> str:
        for level, threshold in sorted(self._confidence_levels.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return level
        return 'very_low'

    def _generate_reasoning(self, factors: ConfidenceFactors) -> str:
        reasons = []
        if factors.data_quality > 0.7:
            reasons.append("数据质量良好")
        elif factors.data_quality < 0.3:
            reasons.append("数据质量较差")
        if factors.context_clarity > 0.7:
            reasons.append("上下文清晰")
        elif factors.context_clarity < 0.3:
            reasons.append("上下文模糊")
        if factors.pattern_recognition > 0.7:
            reasons.append("识别到明确模式")
        if factors.historical_success > 0.7:
            reasons.append("历史成功率高")
        return "；".join(reasons) if reasons else "置信度计算完成"

    def _generate_recommendations(self, factors: ConfidenceFactors) -> List[str]:
        recs = []
        if factors.data_quality < 0.4:
            recs.append("建议收集更多数据")
        if factors.context_clarity < 0.4:
            recs.append("建议提供更多上下文")
        if factors.uncertainty_level > 0.7:
            recs.append("不确定性较高，建议谨慎处理")
        if factors.time_pressure > 0.8:
            recs.append("时间压力较大")
        return recs

    def _find_similar_decisions(self, context) -> List[Dict]:
        return []

    def _record_result(self, result: ConfidenceResult):
        self._confidence_history.append(result)
        if len(self._confidence_history) > self._max_history:
            self._confidence_history = self._confidence_history[-self._max_history:]

    def update_learning(self, context_signature: str, actual: float, predicted: float):
        error = actual - predicted
        adjustment = self._learning_rate * error
        self._confidence_adjustments[context_signature] = max(-0.2, min(0.2,
            self._confidence_adjustments.get(context_signature, 0.0) + adjustment
        ))

    def get_confidence_stats(self) -> Dict[str, Any]:
        if not self._confidence_history:
            return {"total_calculations": 0}
        total = len(self._confidence_history)
        recent = self._confidence_history[-100:]
        level_counts: Dict[str, int] = {}
        for r in recent:
            level_counts[r.confidence_level] = level_counts.get(r.confidence_level, 0) + 1
        return {
            "total_calculations": total,
            "recent_level_distribution": level_counts,
            "average_score": sum(r.score for r in recent) / len(recent),
            "learning_adjustments": len(self._confidence_adjustments)
        }


_confidence_calculator: Optional[ConfidenceCalculator] = None


def get_confidence_calculator() -> ConfidenceCalculator:
    global _confidence_calculator
    if _confidence_calculator is None:
        _confidence_calculator = ConfidenceCalculator()
    return _confidence_calculator

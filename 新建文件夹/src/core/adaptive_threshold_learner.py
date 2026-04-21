import time
import math
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from src.common.logger import get_logger

logger = get_logger("adaptive_threshold")


@dataclass
class FactorVector:
    """多维度因子向量"""

    # 消息维度
    msg_density: float = 0.0
    msg_complexity: float = 0.0
    msg_urgency: float = 0.0
    msg_repetition: float = 0.0
    msg_topic_controversy: float = 0.0
    msg_emotional_intensity: float = 0.0
    msg_has_image: float = 0.0
    msg_has_forward: float = 0.0
    msg_length_score: float = 0.0
    msg_question_marks: float = 0.0

    # 社交维度
    social_affection: float = 0.0
    social_trust: float = 0.0
    social_annoyance: float = 0.0
    social_pressure: float = 0.0
    social_relationship_level: float = 0.0
    social_group_support: float = 0.0
    social_group_attack: float = 0.0
    social_familiarity: float = 0.0
    social_dominance: float = 0.0

    # 能量维度
    energy_ratio: float = 0.8
    thinking_ratio: float = 0.8
    mood_valence: float = 0.5
    mood_arousal: float = 0.5
    stress_accumulation: float = 0.0
    trauma_score: float = 0.0

    # 时间维度
    time_hour_sin: float = 0.0
    time_hour_cos: float = 0.0
    time_recency: float = 1.0
    time_conversation_length: float = 0.0
    time_since_last_reply: float = 0.0

    # 群组维度
    group_pattern_confidence: float = 0.0
    group_pattern_type: float = 0.0
    group_active_users: float = 0.0
    group_message_velocity: float = 0.0
    group_topic_stability: float = 0.0
    group_atmosphere_tension: float = 0.0
    group_me_mentioned_ratio: float = 0.0
    group热闹程度: float = 0.0

    # 历史维度
    history_skip_count: int = 0
    history_llm_call_count: int = 0
    history_success_count: int = 0
    history_fail_count: int = 0
    history_streak_skip: int = 0
    history_streak_llm: int = 0

    def to_vector(self) -> List[float]:
        return [
            self.msg_density,
            self.msg_complexity,
            self.msg_urgency,
            self.msg_repetition,
            self.msg_topic_controversy,
            self.msg_emotional_intensity,
            self.msg_has_image,
            self.msg_has_forward,
            self.msg_length_score,
            self.msg_question_marks,
            self.social_affection,
            self.social_trust,
            self.social_annoyance,
            self.social_pressure,
            self.social_relationship_level,
            self.social_group_support,
            self.social_group_attack,
            self.social_familiarity,
            self.social_dominance,
            self.energy_ratio,
            self.thinking_ratio,
            self.mood_valence,
            self.mood_arousal,
            self.stress_accumulation,
            self.trauma_score,
            self.time_hour_sin,
            self.time_hour_cos,
            self.time_recency,
            self.time_conversation_length,
            self.time_since_last_reply,
            self.group_pattern_confidence,
            self.group_pattern_type,
            self.group_active_users,
            self.group_message_velocity,
            self.group_topic_stability,
            self.group_atmosphere_tension,
            self.group_me_mentioned_ratio,
            self.group热闹程度,
            self.history_skip_count,
            self.history_llm_call_count,
            self.history_success_count,
            self.history_fail_count,
            self.history_streak_skip,
            self.history_streak_llm,
        ]

    @staticmethod
    def vector_dimensions() -> int:
        return 46


@dataclass
class LearnedWeights:
    """可学习的权重向量"""

    base_weights: List[float] = field(default_factory=list)
    bias: float = 0.0
    context_biases: Dict[str, float] = field(default_factory=dict)
    version: int = 0
    last_update_time: float = 0.0

    def __post_init__(self):
        if not self.base_weights:
            dim = FactorVector.vector_dimensions()
            self.base_weights = [0.0] * dim


@dataclass
class DecisionRecord:
    """决策记录（用于反馈学习）"""

    timestamp: float
    factors: FactorVector
    raw_score: float
    final_level: int
    actual_llm_called: bool
    actual_response_quality: float
    user_feedback: float
    stream_id: str = ""
    user_id: str = ""


class AdaptiveThresholdLearner:
    """自适应动态阈值学习器

    核心机制：
    1. 多维度因子提取：从各系统提取46维因子向量
    2. 复杂评分公式：多因子加权 + 非线性变换 + 动态阈值
    3. 自适应学习：根据决策结果反向传播更新权重
    4. 多维度联动：与好感/信任/能量/社交值等深度耦合
    """

    # 基础配置
    BASE_SKIP_THRESHOLD = 0.65
    BASE_ESCALATE_THRESHOLD = 0.35
    BASE_LLM_THRESHOLD = 0.50

    # 学习参数
    LEARNING_RATE = 0.05
    MOMENTUM = 0.3
    DECAY_FACTOR = 0.95
    MIN_SAMPLES_FOR_UPDATE = 10

    # 历史窗口
    HISTORY_WINDOW = 200
    SUCCESS_WINDOW = 50

    # 自适应参数
    ADAPTIVE_DECAY = 0.98
    RECENT_WEIGHT = 2.0
    STREAK_BONUS = 0.1
    STREAK_PENALTY = 0.15

    def __init__(self, stream_id: str = "default"):
        self.stream_id = stream_id
        self._weights = LearnedWeights()
        self._history: deque = deque(maxlen=self.HISTORY_WINDOW)
        self._success_history: deque = deque(maxlen=self.SUCCESS_WINDOW)
        self._streak_skip = 0
        self._streak_llm = 0
        self._last_context = ""
        self._decision_count = 0
        self._context_stats: Dict[str, Dict[str, float]] = {}

    def compute_factors(
        self,
        messages: List[Any],
        social_metrics: Dict[str, float],
        energy_metrics: Dict[str, float],
        group_metrics: Dict[str, float],
        time_metrics: Dict[str, float],
        history_metrics: Dict[str, Any],
    ) -> FactorVector:
        """从多维度提取因子向量"""
        f = FactorVector()

        _msg_count = len(messages) if messages else 0

        if _msg_count > 0:
            f.msg_density = self._sigmoid(_msg_count / 20.0)
            texts = [
                getattr(m, "plain_text", "") or "" for m in messages[-10:]
            ]
            total_len = sum(len(t) for t in texts)
            f.msg_length_score = self._sigmoid(total_len / 500.0)
            f.msg_complexity = self._calc_complexity(texts)
            f.msg_emotional_intensity = self._calc_emotional_intensity(texts)
            f.msg_urgency = self._calc_urgency(texts)
            f.msg_repetition = self._calc_repetition(texts)
            f.msg_topic_controversy = self._calc_controversy(texts)
            f.msg_has_image = (
                1.0
                if any(getattr(m, "has_image", False) for m in messages)
                else 0.0
            )
            f.msg_has_forward = (
                1.0
                if any(getattr(m, "has_forward", False) for m in messages)
                else 0.0
            )
            f.msg_question_marks = sum(1 for t in texts if "?" in t) / max(
                1, _msg_count
            )

        f.social_affection = social_metrics.get("affection", 0.0)
        f.social_trust = social_metrics.get("trust", 0.0)
        f.social_annoyance = social_metrics.get("annoyance", 0.0)
        f.social_pressure = social_metrics.get("pressure", 0.0)
        f.social_relationship_level = social_metrics.get(
            "relationship_level", 0.0
        )
        f.social_group_support = social_metrics.get("group_support", 0.0)
        f.social_group_attack = social_metrics.get("group_attack", 0.0)
        f.social_familiarity = social_metrics.get("familiarity", 0.0)
        f.social_dominance = social_metrics.get("dominance", 0.0)

        f.energy_ratio = energy_metrics.get("energy_ratio", 0.8)
        f.thinking_ratio = energy_metrics.get("thinking_ratio", 0.8)
        f.mood_valence = energy_metrics.get("mood_valence", 0.5)
        f.mood_arousal = energy_metrics.get("mood_arousal", 0.5)
        f.stress_accumulation = energy_metrics.get("stress_accumulation", 0.0)
        f.trauma_score = energy_metrics.get("trauma_score", 0.0)

        _hour = time.localtime().tm_hour
        f.time_hour_sin = math.sin(2 * math.pi * _hour / 24)
        f.time_hour_cos = math.cos(2 * math.pi * _hour / 24)
        f.time_recency = time_metrics.get("recency", 1.0)
        f.time_conversation_length = time_metrics.get(
            "conversation_length", 0.0
        )
        f.time_since_last_reply = time_metrics.get("since_last_reply", 0.0)

        f.group_pattern_confidence = group_metrics.get(
            "pattern_confidence", 0.0
        )
        f.group_pattern_type = group_metrics.get("pattern_type", 0.0)
        f.group_active_users = group_metrics.get("active_users", 0.0)
        f.group_message_velocity = group_metrics.get("message_velocity", 0.0)
        f.group_topic_stability = group_metrics.get("topic_stability", 1.0)
        f.group_atmosphere_tension = group_metrics.get(
            "atmosphere_tension", 0.0
        )
        f.group_me_mentioned_ratio = group_metrics.get(
            "me_mentioned_ratio", 0.0
        )
        f.group热闹程度 = group_metrics.get("excitement_level", 0.0)

        f.history_skip_count = history_metrics.get("skip_count", 0)
        f.history_llm_call_count = history_metrics.get("llm_call_count", 0)
        f.history_success_count = history_metrics.get("success_count", 0)
        f.history_fail_count = history_metrics.get("fail_count", 0)
        f.history_streak_skip = history_metrics.get("streak_skip", 0)
        f.history_streak_llm = history_metrics.get("streak_llm", 0)

        return f

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))

    def _tanh(self, x: float) -> float:
        return math.tanh(max(-10, min(10, x)))

    def _calc_complexity(self, texts: List[str]) -> float:
        if not texts:
            return 0.0
        total_score = 0.0
        for text in texts:
            words = text.split()
            total_score += min(1.0, len(words) / 30.0)
            special_chars = sum(
                1 for c in text if c in "，。！？；：" "''（）"
            )
            total_score += min(0.5, special_chars / 10.0)
            try:
                total_score += min(0.3, len(text) / 200.0)
            except Exception as e:
                logger.debug(f"计算文本长度分数失败: {e}")
        return self._sigmoid(total_score / max(1, len(texts)))

    def _calc_emotional_intensity(self, texts: List[str]) -> float:
        _emotion_words = [
            "好",
            "喜欢",
            "开心",
            "棒",
            "哈哈",
            "呵",
            "滚",
            "烦",
            "讨厌",
            "气",
            "怒",
            "哭",
            "笑",
            "爱",
            "恨",
        ]
        _high_intensity = ["太", "超", "巨", "爆", "真的", "简直"]
        score = 0.0
        for text in texts:
            lower = text.lower()
            for w in _emotion_words:
                if w in lower:
                    score += 0.15
            for w in _high_intensity:
                if w in lower:
                    score += 0.1
        return min(1.0, score / max(1, len(texts)))

    def _calc_urgency(self, texts: List[str]) -> float:
        _urgency_keywords = [
            "快",
            "快说",
            "赶紧",
            "马上",
            "立刻",
            "急",
            "现在",
            "立刻",
            "赶紧",
            "催",
        ]
        score = 0.0
        for text in texts:
            for kw in _urgency_keywords:
                if kw in text:
                    score += 0.2
        return min(1.0, score)

    def _calc_repetition(self, texts: List[str]) -> float:
        if len(texts) < 2:
            return 0.0
        score = 0.0
        for i in range(len(texts) - 1):
            if texts[i] == texts[i + 1]:
                score += 0.3
            elif texts[i] in texts[i + 1] or texts[i + 1] in texts[i]:
                score += 0.15
        return min(1.0, score)

    def _calc_controversy(self, texts: List[str]) -> float:
        _controversy_words = [
            "不对",
            "不是",
            "但是",
            "可是",
            "虽然",
            "其实",
            "应该",
            "不一定",
            "可能",
            "未必",
            "反对",
            "同意",
            "傻",
            "蠢",
        ]
        score = 0.0
        for text in texts:
            for w in _controversy_words:
                if w in text:
                    score += 0.2
        return min(1.0, score)

    def compute_skip_score(self, f: FactorVector, context: str) -> float:
        """计算跳过LLM的评分（越高越应该跳过）"""
        v = f.to_vector()
        w = self._weights

        _w_base = (
            w.base_weights if len(w.base_weights) == len(v) else [0.0] * len(v)
        )
        weighted_sum = sum(vi * wi for vi, wi in zip(v, _w_base))

        raw = self._sigmoid(weighted_sum + w.bias)

        context_adj = (
            w.context_biases.get(context, 0.0) if w.context_biases else 0.0
        )

        _hour = time.localtime().tm_hour
        _is_night = 1.0 if (_hour >= 23 or _hour < 6) else 0.0

        _energy_penalty = (
            max(0.0, 0.4 - f.energy_ratio * 0.4)
            if f.energy_ratio < 0.3
            else 0.0
        )
        _thinking_penalty = (
            max(0.0, 0.3 - f.thinking_ratio * 0.3)
            if f.thinking_ratio < 0.3
            else 0.0
        )
        _night_bonus = _is_night * 0.15
        _quiet_pref_bonus = (
            f.social_annoyance * 0.1 if f.social_annoyance > 30 else 0.0
        )
        _repeat_bonus = min(0.3, f.msg_repetition * 0.5)
        _streak_skip_bonus = (
            min(0.2, f.history_streak_skip * self.STREAK_PENALTY)
            if f.history_streak_skip > 3
            else 0.0
        )
        _pattern_skip_bonus = (
            f.group_pattern_confidence * 0.3
            if f.group_pattern_type in [1, 2, 6]
            else 0.0
        )
        _msg_density_bonus = f.msg_density * 0.15

        final = (
            raw * 0.4
            + context_adj * 0.15
            + _energy_penalty * 0.15
            + _thinking_penalty * 0.10
            + _night_bonus * 0.08
            + _quiet_pref_bonus * 0.05
            + _repeat_bonus * 0.05
            + _streak_skip_bonus * 0.05
            + _pattern_skip_bonus * 0.10
            + _msg_density_bonus * 0.05
        )

        return max(0.0, min(1.0, final))

    def compute_escalate_score(self, f: FactorVector, context: str) -> float:
        """计算升级大模型的评分（越高越应该升级）"""
        v = f.to_vector()

        _complexity_weight = 0.25
        _controversy_weight = 0.20
        _emotion_weight = 0.15
        _pattern_weight = 0.20
        _social_weight = 0.20

        _complexity_component = f.msg_complexity * _complexity_weight
        _controversy_component = f.msg_topic_controversy * _controversy_weight
        _emotion_component = f.msg_emotional_intensity * _emotion_weight
        _pattern_component = (
            f.group_pattern_confidence * 0.3
            if f.group_pattern_type in [3, 4, 5, 7, 8, 9]
            else f.group_pattern_confidence * 0.1
        ) * _pattern_weight
        _social_component = (
            f.social_group_attack * 0.15
            + f.social_pressure * 0.10
            + (1.0 - f.social_affection / 100.0) * 0.10
            + (1.0 - f.social_trust / 100.0) * 0.10
        ) * _social_weight

        _hour = time.localtime().tm_hour
        _is_workhour = 1.0 if (9 <= _hour <= 22) else 0.0

        _streak_llm_penalty = (
            min(0.25, f.history_streak_llm * self.STREAK_PENALTY)
            if f.history_streak_llm > 2
            else 0.0
        )
        _question_bonus = f.msg_question_marks * 0.1
        _forward_bonus = f.msg_has_forward * 0.15
        _high_energy_bonus = (
            max(0.0, (f.energy_ratio - 0.5) * 0.2)
            if f.energy_ratio > 0.7
            else 0.0
        )
        _low_stress_bonus = max(0.0, (0.5 - f.stress_accumulation) * 0.1)

        final = (
            _complexity_component
            + _controversy_component
            + _emotion_component
            + _pattern_component
            + _social_component
            + _streak_llm_penalty
            + _question_bonus
            + _forward_bonus
            + _high_energy_bonus
            + _low_stress_bonus
        )

        return max(0.0, min(1.0, final))

    def decide_level(
        self,
        messages: List[Any],
        social_metrics: Dict[str, float],
        energy_metrics: Dict[str, float],
        group_metrics: Dict[str, float],
        time_metrics: Dict[str, float],
        history_metrics: Dict[str, Any],
        pinged: bool = False,
    ) -> Tuple[int, float, float]:
        """决定调用等级

        Returns:
            Tuple[int, float, float]: (level, skip_score, escalate_score)
            level: 0=跳过，1=小模型，2=大模型
        """
        self._decision_count += 1

        f = self.compute_factors(
            messages,
            social_metrics,
            energy_metrics,
            group_metrics,
            time_metrics,
            history_metrics,
        )

        _context = self._get_context_signature(
            messages, social_metrics, group_metrics
        )
        self._last_context = _context

        if pinged:
            return 1, 0.0, 0.0

        skip_score = self.compute_skip_score(f, _context)
        escalate_score = self.compute_escalate_score(f, _context)

        net_score = skip_score - escalate_score

        _threshold_skip = (
            self.BASE_SKIP_THRESHOLD
            + self._get_contextual_adjustment(_context, "skip")
        )
        _threshold_escalate = (
            self.BASE_ESCALATE_THRESHOLD
            + self._get_contextual_adjustment(_context, "escalate")
        )

        if net_score >= _threshold_skip:
            level = 0
        elif net_score <= _threshold_escalate:
            level = 2
        else:
            level = 1

        logger.debug(
            f"[自适应阈值] skip={skip_score:.3f} esc={escalate_score:.3f} "
            f"net={net_score:.3f} → level={level} "
            f"(阈值: skip>{_threshold_skip:.2f} esc<{_threshold_escalate:.2f})"
        )

        return level, skip_score, escalate_score

    def record_outcome(
        self,
        level: int,
        skip_score: float,
        escalate_score: float,
        actual_llm_called: bool,
        response_quality: float,
        user_feedback: float,
        stream_id: str = "",
        user_id: str = "",
    ):
        """记录决策结果（用于反馈学习）"""
        if not self._history:
            return

        try:
            last = self._history[-1]
            last.actual_llm_called = actual_llm_called
            last.actual_response_quality = response_quality
            last.user_feedback = user_feedback

            self._success_history.append(
                {
                    "level": level,
                    "quality": response_quality,
                    "feedback": user_feedback,
                    "skip_score": skip_score,
                    "escalate_score": escalate_score,
                }
            )

            if response_quality > 0.7 or user_feedback > 0.5:
                self._streak_llm += 1
                self._streak_skip = 0
            else:
                self._streak_skip += 1
                self._streak_llm = 0

            if len(self._success_history) >= self.MIN_SAMPLES_FOR_UPDATE:
                self._update_weights()
        except Exception as e:
            logger.debug(f"[自适应阈值] 记录结果异常: {e}")

    def add_decision_record(self, factors: FactorVector, level: int):
        """添加决策记录"""
        record = DecisionRecord(
            timestamp=time.time(),
            factors=factors,
            raw_score=0.0,
            final_level=level,
            actual_llm_called=False,
            actual_response_quality=0.5,
            user_feedback=0.0,
        )
        self._history.append(record)

    def _get_context_signature(
        self, messages: List, social: Dict, group: Dict
    ) -> str:
        try:
            msg_hash = hashlib.md5(
                "|".join(
                    [
                        str(getattr(m, "plain_text", "")[:20])
                        for m in (messages or [])[-3:]
                    ]
                ).encode()
            ).hexdigest()[:8]
            aff = social.get("affection", 0)
            group_type = group.get("pattern_type", 0)
            return f"{msg_hash}_{aff:.0f}_{group_type}"
        except Exception:
            return "default"

    def _get_contextual_adjustment(
        self, context: str, direction: str
    ) -> float:
        if not self._context_stats:
            return 0.0
        stats = self._context_stats.get(context, {})
        if direction == "skip":
            return stats.get("skip_adjust", 0.0)
        return stats.get("escalate_adjust", 0.0)

    def _update_weights(self):
        """基于历史反馈更新权重"""
        if len(self._success_history) < self.MIN_SAMPLES_FOR_UPDATE:
            return

        try:
            _successes = [
                h for h in self._success_history if h["quality"] > 0.6
            ]
            _fails = [h for h in self._success_history if h["quality"] <= 0.4]

            if not _successes or not _fails:
                return

            _success_avg_skip = sum(h["skip_score"] for h in _successes) / max(1, len(
                _successes
            ))
            _fail_avg_skip = sum(h["skip_score"] for h in _fails) / max(1, len(_fails))

            _success_avg_esc = sum(
                h["escalate_score"] for h in _successes
            ) / max(1, len(_successes))
            _fail_avg_esc = sum(h["escalate_score"] for h in _fails) / max(1, len(
                _fails
            ))

            _skip_delta = (
                _fail_avg_skip - _success_avg_skip
            ) * self.LEARNING_RATE
            _esc_delta = (
                _success_avg_esc - _fail_avg_esc
            ) * self.LEARNING_RATE

            if not self._weights.base_weights:
                dim = FactorVector.vector_dimensions()
                self._weights.base_weights = [0.0] * dim

            for i in range(len(self._weights.base_weights)):
                self._weights.base_weights[i] = (
                    self._weights.base_weights[i] * self.MOMENTUM
                    + _skip_delta * (0.5 - _fail_avg_skip)
                    + _esc_delta * (_success_avg_esc - 0.5)
                ) * self.DECAY_FACTOR

            self._weights.bias += _skip_delta * 0.1
            self._weights.version += 1
            self._weights.last_update_time = time.time()

            logger.debug(
                f"[自适应阈值] 权重已更新 v={self._weights.version} "
                f"skip_delta={_skip_delta:.4f} esc_delta={_esc_delta:.4f}"
            )
        except Exception as e:
            logger.debug(f"[自适应阈值] 权重更新异常: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "decision_count": self._decision_count,
            "history_size": len(self._history),
            "weights_version": self._weights.version,
            "last_update": self._weights.last_update_time,
            "streak_skip": self._streak_skip,
            "streak_llm": self._streak_llm,
        }


_global_learner: Optional[AdaptiveThresholdLearner] = None


def get_adaptive_threshold_learner(
    stream_id: str = "default",
) -> AdaptiveThresholdLearner:
    global _global_learner
    if _global_learner is None:
        _global_learner = AdaptiveThresholdLearner(stream_id)
    return _global_learner

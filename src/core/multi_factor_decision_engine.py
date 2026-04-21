import math
import time
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.common.singleton import _get_class_lock

logger = get_logger("acfn_engine")

# ──────────────────────────────────────────────
#  数学工具库
# ──────────────────────────────────────────────


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _sigmoid(x: float, midpoint: float = 0.5, steepness: float = 8.0) -> float:
    z = steepness * (x - midpoint)
    z = max(-15.0, min(15.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _tanh_scaled(x: float, scale: float = 1.0) -> float:
    z = max(-10.0, min(10.0, x / scale))
    return math.tanh(z)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gaussian_noise(mean: float = 0.0, stddev: float = 0.01) -> float:
    return random.gauss(mean, stddev)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp01(t)


def _fatigue_curve(minutes: float) -> float:
    if minutes < 10:
        return 1.0
    elif minutes < 30:
        return 1.0 + (minutes - 10) * 0.012
    elif minutes < 60:
        s1 = 1.24 + (30 - 10) * 0.012
        return s1 + (minutes - 30) * 0.02
    else:
        s2 = 1.24 + 20 * 0.012 + 30 * 0.02
        return min(2.5, s2 + (minutes - 60) * 0.008)


# ──────────────────────────────────────────────
#  因子索引常量（用于耦合矩阵寻址）
# ──────────────────────────────────────────────

IDX_ENERGY = 0
IDX_SLEEP_DEBT = 1
IDX_SLEEP_PHASE = 2
IDX_SOCIAL_WILL = 3
IDX_WATCH_WILL = 4
IDX_AVOIDANCE = 5
IDX_QUIET_PREF = 6
IDX_CURIOSITY = 7
IDX_OUTWARD_ATT = 8
IDX_AFFECTION = 9
IDX_TRUST = 10
IDX_ANNOYANCE = 11
IDX_TRAUMA = 12
IDX_AVERSION = 13
IDX_TOPIC_OWN = 14
IDX_TOPIC_CONF = 15
IDX_PATTERN = 16
IDX_PATTERN_CONF = 17
IDX_SPECTATOR = 18
IDX_EVENT_REL = 19
IDX_SELF_REF_STR = 20
IDX_DISCUSS_N = 21
IDX_QUOTE_N = 22
IDX_DIRECT_AT = 23
IDX_NARR_STRAT = 24
IDX_NARR_MOOD = 25
IDX_MEDIA_STORM = 26
IDX_INTENTION = 27
FACTOR_DIM = 28

_FACTOR_NAMES = [
    "energy",
    "sleep_debt",
    "sleep_phase",
    "social_will",
    "watch_will",
    "avoidance",
    "quiet_pref",
    "curiosity",
    "outward_att",
    "affection",
    "trust",
    "annoyance",
    "trauma",
    "aversion",
    "topic_own",
    "topic_conf",
    "pattern",
    "pattern_conf",
    "spectator",
    "event_rel",
    "self_ref_str",
    "discuss_n",
    "quote_n",
    "direct_at",
    "narr_strat",
    "narr_mood",
    "media_storm",
    "intention",
]

# ──────────────────────────────────────────────
#  输入数据载体
# ──────────────────────────────────────────────


@dataclass
class FactorInput:
    """多因子融合器的完整输入快照（28维原始数据）"""

    energy_chat_ratio: float = 1.0
    energy_think_ratio: float = 1.0
    energy_activity_ratio: float = 0.5
    energy_social_ratio: float = 0.5
    energy_composite: float = 1.0
    night_phase: str = "awake"
    sleep_debt: float = 0.0
    sleep_peek_budget: bool = True
    sleep_reply_budget: bool = True
    sleep_hours_into_night: float = 0.0
    # GAP-A：熬夜压力与亢奋链
    overnight_pressure: float = 0.0
    arousal_chain_state: str = "normal"
    is_burnthrough: bool = False
    burnthrough_impulse: float = 0.0
    social_willingness: float = 0.5
    watch_willingness: float = 0.5
    avoidance_tendency: float = 0.0
    quiet_preference: float = 0.2
    curiosity_level: float = 0.3
    outward_attention: float = 0.5
    affection: float = 50.0
    trust_value: float = 50.0
    annoyance_value: float = 0.0
    trauma_score: float = 0.0
    aversion_value: float = 0.0
    topic_ownership: str = "ambiguous_topic"
    topic_confidence: float = 0.5
    dominant_pattern: str = ""
    pattern_confidence: float = 0.0
    is_spectator_mode: bool = False
    event_relevance: float = 0.0
    self_ref_types: List[str] = field(default_factory=list)
    self_ref_max_strength: float = 0.0
    discuss_as_topic_count: int = 0
    quoted_reply_count: int = 0
    direct_at: bool = False
    narration_strategy: str = ""
    narration_mood: str = ""
    media_total_items: int = 0
    media_skipped_count: int = 0
    media_storm_active: bool = False
    consecutive_active_minutes: float = 0.0
    hourly_reply_used: int = 0
    silence_seconds: float = 0.0
    intention_drive: float = 0.0
    intent_override_candidate: bool = False
    coupling_boredom_mod: float = 0.0
    coupling_activity_mod: float = 0.0
    coupling_style_tier: str = ""
    impression_narrative_type: str = ""
    impression_affection_proxy: float = 50.0
    impression_trust_proxy: float = 50.0
    skill_cost_modifier: float = 1.0
    mm_avg_valence: float = 0.0
    mm_engagement: float = 0.0


@dataclass
class FactorVerdict:
    """多因子融合器的结构化输出"""

    bar_modifier: float = 0.0
    score_modifier: float = 1.0
    should_hard_block: bool = False
    block_reason: str = ""
    breakdown: Dict[str, float] = field(default_factory=dict)
    dominant_factors: List[str] = field(default_factory=list)
    convergence_passes: int = 0
    final_confidence: float = 0.0
    coupling_strength: float = 0.0
    computed_at: float = field(default_factory=time.time)

    def apply_to_arbiter_verdict(self, arbiter_verdict: Any) -> None:
        if self.should_hard_block:
            arbiter_verdict.should_proceed = False
            arbiter_verdict.rationale = self.block_reason
            arbiter_verdict.breakdown["acfn_hard_block"] = -1.0
            return
        if abs(self.bar_modifier) > 0.0001:
            new_bar = arbiter_verdict.activation_bar + self.bar_modifier
            arbiter_verdict.activation_bar = max(0.05, min(0.95, new_bar))
            arbiter_verdict.breakdown["acfn_bar"] = round(self.bar_modifier, 5)
        if abs(self.score_modifier - 1.0) > 0.0001:
            arbiter_verdict.fused_score *= self.score_modifier
            arbiter_verdict.breakdown["acfn_score"] = round(
                self.score_modifier, 5
            )
        for k, val in self.breakdown.items():
            if k not in arbiter_verdict.breakdown:
                arbiter_verdict.breakdown[k] = val
        arbiter_verdict.breakdown["acfn_confidence"] = round(
            self.final_confidence, 4
        )
        arbiter_verdict.breakdown["acfn_passes"] = self.convergence_passes
        arbiter_verdict.breakdown["acfn_coupling"] = round(
            self.coupling_strength, 4
        )


# ──────────────────────────────────────────────
#  内部状态：滞后记忆与历史追踪
# ──────────────────────────────────────────────


@dataclass
class _EngineState:
    """引擎内部可变状态 —— 跨轮次持久化"""

    # 滞后记忆（最近N次输出的指数加权移动平均）
    bar_history: deque = field(default_factory=lambda: deque(maxlen=20))
    score_history: deque = field(default_factory=lambda: deque(maxlen=20))
    block_history: deque = field(default_factory=lambda: deque(maxlen=20))
    # 上一次归一化特征向量（用于计算时序导数）
    last_feature_vector: Optional[List[float]] = None
    last_feature_time: float = 0.0
    # 自适应阈值（根据近期决策分布自动漂移）
    adaptive_energy_threshold: float = 0.35
    adaptive_sleep_threshold: float = 0.50
    adaptive_relation_threshold: float = 0.40
    # 反馈回路：最近M次的决策方向统计（防止单向偏移）
    positive_decision_count: int = 0
    negative_decision_count: int = 0
    total_decisions: int = 0
    # 耦合矩阵缓存（避免每轮重建）
    coupling_matrix: Optional[List[List[float]]] = None
    coupling_dirty: bool = True
    # 迭代计数
    total_evaluations: int = 0
    cache_hit_count: int = 0


# ──────────────────────────────────────────────
#  核心：自适应耦合因子网络
# ──────────────────────────────────────────────


class MultiFactorDecisionEngine:
    """自适应耦合因子网络（Adaptive Coupled Factor Network）

    核心创新点：
      ① 因子间非独立：28维因子通过 N×N 耦合矩阵相互调制
      ② 时序感知：不仅看当前值，还看变化率（导数层）
      ③ 自适应权重：根据历史反馈和方差动态调整每个通道权重
      ④ 滞后效应：连续相似输入产生有差异的输出（防抖动/防震荡）
      ⑤ 多轮迭代：3~5轮收敛精炼，每轮修正上一轮偏差
      ⑥ 随机微扰：低置信度区域引入受控噪声（防死循环）
      ⑦ 反馈自校正：检测到持续单向偏移时自动反向补偿
    """

    MAX_ITERATION_PASSES = 5
    CONVERGENCE_EPSILON = 0.003
    HISTORY_WINDOW = 20
    DERIVATIVE_DECAY_SEC = 60.0

    def __init__(self):
        self._state = _EngineState()
        self._base_weights = self._init_base_weights()
        self._coupling_template = self._build_coupling_template()

    def evaluate(self, inp: FactorInput) -> FactorVerdict:
        """执行完整的多轮迭代评估（主入口）"""
        self._state.total_evaluations += 1
        now = time.time()
        # ══ L1：原始输入 → 归一化特征向量（28维）══
        features = self._normalize_input(inp)
        # ══ L3：附加时序导数（变化率维度）══
        derivatives = self._compute_derivatives(features, now)
        # ══ L2：构建/更新交叉耦合矩阵 ═══
        coupling = self._get_coupling_matrix(features, derivatives)
        # ══ L4：计算自适应权重（历史反馈+方差+滞后）══
        adaptive_weights = self._compute_adaptive_weights(
            features, derivatives
        )
        # ══ L5：多轮迭代精炼 ═══
        prev_bar = 0.0
        prev_score = 1.0
        verdict = None
        for pass_num in range(1, self.MAX_ITERATION_PASSES + 1):
            verdict = self._single_pass(
                inp,
                features,
                derivatives,
                coupling,
                adaptive_weights,
                pass_num,
                prev_bar,
                prev_score,
            )
            delta_bar = abs(verdict.bar_modifier - prev_bar)
            delta_score = abs(verdict.score_modifier - prev_score)
            prev_bar = verdict.bar_modifier
            prev_score = verdict.score_modifier
            verdict.convergence_passes = pass_num
            if (
                delta_bar < self.CONVERGENCE_EPSILON
                and delta_score < self.CONVERGENCE_EPSILON * 0.5
            ):
                break
        # ══ L6：随机微扰层 ═══
        self._apply_stochastic_perturbation(verdict, features)
        # ══ L7：输出聚合 + 反馈更新 ═══
        verdict.final_confidence = self._compute_confidence(verdict, features)
        verdict.coupling_strength = self._measure_coupling_activation(
            coupling, features
        )
        self._update_feedback_state(verdict)
        self._state.last_feature_vector = features[:]
        self._state.last_feature_time = now
        verdict.computed_at = now
        return verdict

    # ──────────────────────────────────────────
    #  L1：输入归一化
    # ─────────────────────────────────────────-

    def _normalize_input(self, inp: FactorInput) -> List[float]:
        """将 28 个原始字段映射到 [0,1] 归一化特征向量"""
        f = [0.0] * FACTOR_DIM
        f[IDX_ENERGY] = _clamp01(inp.energy_composite)
        f[IDX_SLEEP_DEBT] = _clamp01(inp.sleep_debt / 120.0)
        phase_map = {
            "awake": 0.0,
            "drowsy": 0.12,
            "night_active": 0.08,
            "excited": 0.05,
            "quiet_contemplate": 0.30,
            "light_sleep": 0.55,
            "burned_out": 0.75,
            "exhausted": 0.82,
            "deep_sleep": 1.0,
        }
        f[IDX_SLEEP_PHASE] = phase_map.get(inp.night_phase.lower(), 0.0)
        f[IDX_SOCIAL_WILL] = _clamp01(inp.social_willingness)
        f[IDX_WATCH_WILL] = _clamp01(inp.watch_willingness)
        f[IDX_AVOIDANCE] = _clamp01(inp.avoidance_tendency)
        f[IDX_QUIET_PREF] = _clamp01(inp.quiet_preference)
        f[IDX_CURIOSITY] = _clamp01(inp.curiosity_level)
        f[IDX_OUTWARD_ATT] = _clamp01(inp.outward_attention)
        f[IDX_AFFECTION] = _clamp01(inp.affection / 100.0)
        f[IDX_TRUST] = _clamp01(inp.trust_value / 100.0)
        f[IDX_ANNOYANCE] = _clamp01(inp.annoyance_value / 100.0)
        f[IDX_TRAUMA] = _clamp01(inp.trauma_score / 100.0)
        f[IDX_AVERSION] = _clamp01(inp.aversion_value / 100.0)
        own_map = {
            "self_topic": 0.0,
            "shared_group_topic": 0.25,
            "target_user_topic": 0.5,
            "ambiguous_topic": 0.72,
            "external_topic": 1.0,
        }
        f[IDX_TOPIC_OWN] = own_map.get(inp.topic_ownership, 0.72)
        f[IDX_TOPIC_CONF] = _clamp01(inp.topic_confidence)
        pat_map = {
            "newcomer_welcome": 0.05,
            "ritual_greeting": 0.10,
            "birthday_wish": 0.08,
            "holiday_greeting": 0.08,
            "congratulation": 0.08,
            "chain_reply": 0.18,
            "group_photo": 0.14,
            "celebration_wave": 0.17,
            "support_circle": 0.06,
            "spectator_mode": 0.70,
            "heated_discussion": 0.85,
            "argument": 0.82,
            "conflict_escalation": 0.95,
        }
        f[IDX_PATTERN] = pat_map.get(inp.dominant_pattern, 0.50)
        f[IDX_PATTERN_CONF] = _clamp01(inp.pattern_confidence)
        f[IDX_SPECTATOR] = 1.0 if inp.is_spectator_mode else 0.0
        f[IDX_EVENT_REL] = _clamp01(inp.event_relevance)
        f[IDX_SELF_REF_STR] = _clamp01(inp.self_ref_max_strength)
        f[IDX_DISCUSS_N] = _clamp01(inp.discuss_as_topic_count / 5.0)
        f[IDX_QUOTE_N] = _clamp01(inp.quoted_reply_count / 4.0)
        f[IDX_DIRECT_AT] = 1.0 if inp.direct_at else 0.0
        strat_map = {
            "observe_only": 0.80,
            "defensive": 0.95,
            "spectator_eat_melon": 0.70,
            "engage_active": 0.10,
        }
        f[IDX_NARR_STRAT] = strat_map.get(inp.narration_strategy.lower(), 0.50)
        mood_map = {
            "annoyed": 0.85,
            "tired": 0.70,
            "overwhelmed": 0.92,
            "curious": 0.15,
            "playful": 0.05,
            "calm": 0.25,
        }
        f[IDX_NARR_MOOD] = mood_map.get(inp.narration_mood.lower(), 0.40)
        f[IDX_MEDIA_STORM] = 1.0 if inp.media_storm_active else 0.0
        f[IDX_INTENTION] = _clamp01(inp.intention_drive)
        return f

    # ──────────────────────────────────────────
    #  L3：时序导数层
    # ─────────────────────────────────────────-

    def _compute_derivatives(
        self, current: List[float], now: float
    ) -> List[float]:
        """计算每个因子的时间导数（变化率）

        核心思想：
          - 急剧下降的能量比稳定在低位更危险（导数为负且大）
          - 快速上升的好感比长期高好感更有激励效果（导数为正且大）
          - 导数的绝对值越大，该因子在本轮应获得更高关注度
        """
        deriv = [0.0] * FACTOR_DIM
        if self._state.last_feature_vector is None:
            return deriv
        elapsed = max(0.001, now - self._state.last_feature_time)
        decay = math.exp(-elapsed / self.DERIVATIVE_DECAY_SEC)
        for i in range(FACTOR_DIM):
            prev_val = self._state.last_feature_vector[i]
            if abs(prev_val) < 0.0001:
                continue
            raw_deriv = (current[i] - prev_val) / elapsed
            deriv[i] = raw_deriv * 10.0
            deriv[i] = max(-2.0, min(2.0, deriv[i]))
        return deriv

    # ──────────────────────────────────────────
    #  L2：因子交叉耦合矩阵
    # ─────────────────────────────────────────-

    def _build_coupling_template(self) -> List[List[float]]:
        """构建 28×28 因子耦合模板矩阵

        C[i][j] 表示因子 i 对因子 j 权重的调制系数：
          正值 → i 高时 j 的权重增强
          负值 → i 高时 j 的权重削弱
          零   → i 与 j 独立

        设计原则：
          - 创伤高 → 能量约束更严苛（创伤放大疲劳感知）
          - 厌烦高 → 社交意愿的影响被削弱（厌烦让人不在乎社交信号）
          - 睡眠债务高 → 好感的正向作用衰减（困倦时关系不再重要）
          - 意图驱动高 → 大部分负面因子被抑制（紧急意图优先）
          - 围观模式 → 自我引用增益放大（围观时更在意是否被提及）
        """
        c = [[0.0] * FACTOR_DIM for _ in range(FACTOR_DIM)]
        c[IDX_TRAUMA][IDX_ENERGY] = 0.35
        c[IDX_TRAUMA][IDX_AVOIDANCE] = 0.28
        c[IDX_ANNOYANCE][IDX_SOCIAL_WILL] = -0.32
        c[IDX_ANNOYANCE][IDX_AFFECTION] = -0.20
        c[IDX_SLEEP_DEBT][IDX_AFFECTION] = -0.25
        c[IDX_SLEEP_DEBT][IDX_TRUST] = -0.15
        c[IDX_SLEEP_DEBT][IDX_CURIOSITY] = -0.30
        c[IDX_INTENTION][IDX_AVOIDANCE] = -0.55
        c[IDX_INTENTION][IDX_QUIET_PREF] = -0.45
        c[IDX_INTENTION][IDX_ENERGY] = -0.30
        c[IDX_INTENTION][IDX_SLEEP_PHASE] = -0.40
        c[IDX_SPECTATOR][IDX_SELF_REF_STR] = 0.42
        c[IDX_SPECTATOR][IDX_EVENT_REL] = 0.38
        c[IDX_EVENT_REL][IDX_SPECTATOR] = -0.50
        c[IDX_AFFECTION][IDX_TOPIC_OWN] = -0.22
        c[IDX_AVOIDANCE][IDX_OUTWARD_ATT] = -0.38
        c[IDX_AVOIDANCE][IDX_SOCIAL_WILL] = -0.30
        c[IDX_CURIOSITY][IDX_WATCH_WILL] = 0.33
        c[IDX_SELF_REF_STR][IDX_PATTERN] = -0.18
        c[IDX_DISCUSS_N][IDX_SELF_REF_STR] = 0.25
        c[IDX_NARR_STRAT][IDX_SOCIAL_WILL] = -0.28
        c[IDX_NARR_MOOD][IDX_AVOIDANCE] = 0.22
        c[IDX_MEDIA_STORM][IDX_CURIOSITY] = -0.20
        for i in range(FACTOR_DIM):
            c[i][i] = 0.0
        return c

    def _get_coupling_matrix(
        self, features: List[float], derivatives: List[float]
    ) -> List[List[float]]:
        """获取当前耦合矩阵（带动态调制）

        耦合强度不是固定的，而是随因子值动态缩放：
          - 只有当源因子超过激活阈值时才施加耦合
          - 导数大的因子其耦合效应被放大（快速变化的因子影响更强）
        """
        if (
            not self._state.coupling_dirty
            and self._state.coupling_matrix is not None
        ):
            return self._state.coupling_matrix
        template = self._coupling_template
        result = [[0.0] * FACTOR_DIM for _ in range(FACTOR_DIM)]
        for i in range(FACTOR_DIM):
            for j in range(FACTOR_DIM):
                base_c = template[i][j]
                if abs(base_c) < 0.01:
                    continue
                src_act = _sigmoid(features[i], midpoint=0.35, steepness=6.0)
                tgt_sensitivity = 1.0 - _sigmoid(
                    features[j], midpoint=0.65, steepness=4.0
                )
                deriv_boost = 1.0 + abs(derivatives[i]) * 0.25
                result[i][j] = base_c * src_act * tgt_sensitivity * deriv_boost
        self._state.coupling_matrix = result
        self._state.coupling_dirty = False
        return result

    # ──────────────────────────────────────────
    #  L4：自适应权重引擎
    # ─────────────────────────────────────────-

    def _init_base_weights(self) -> Dict[str, float]:
        """初始化各因子通道的基础权重"""
        return {
            "energy_constraint": 0.18,
            "sleep_rhythm": 0.14,
            "presence_modulation": 0.13,
            "relation_trauma": 0.12,
            "topic_ownership": 0.09,
            "group_pattern": 0.10,
            "self_reference": 0.07,
            "narration_strategy": 0.05,
            "multimodal_throttle": 0.04,
            "spectator_convert": 0.03,
            "hourly_cap": 0.03,
            "intent_shield": 0.02,
        }

    def _compute_adaptive_weights(
        self, features: List[float], derivatives: List[float]
    ) -> Dict[str, float]:
        """计算自适应权重（基础权重 × 调制系数）

        调制来源：
          ① 方差注意力：近期变化大的因子获得更高权重
          ② 滞后偏差：如果最近一直做同一类决策，反向微调
          ③ 因子饱和度：接近边界的因子权重降低（边际递减）
          ④ 耦合聚合度：被多个其他因子耦合的因子权重提升
        """
        base = dict(self._base_weights)
        variance_attn = {}
        for i in range(FACTOR_DIM):
            d = abs(derivatives[i])
            variance_attn[_FACTOR_NAMES[i]] = 1.0 + d * 0.15
        saturation = {}
        for i in range(FACTOR_DIM):
            v = features[i]
            sat = 1.0 - abs(v - 0.5) * 0.6
            saturation[_FACTOR_NAMES[i]] = max(0.35, sat)
        hysteresis_bias = self._compute_hysteresis_bias()
        coupling_agg = self._compute_coupling_aggregation()
        adaptive = {}
        for ch_name, w_base in base.items():
            ch_factors = self._channel_to_factor_indices(ch_name)
            var_mod = 1.0
            sat_mod = 1.0
            coup_mod = 1.0
            for fi in ch_factors:
                fn = _FACTOR_NAMES[fi]
                var_mod *= variance_attn.get(fn, 1.0)
                sat_mod *= saturation.get(fn, 1.0)
                coup_mod += coupling_agg.get(fn, 0.0) * 0.08
            hyst_mod = hysteresis_bias.get(ch_name, 1.0)
            adaptive[ch_name] = (
                w_base * var_mod * sat_mod * coup_mod * hyst_mod
            )
        total = sum(adaptive.values())
        if total > 0.001:
            for k in adaptive:
                adaptive[k] /= total
        return adaptive

    def _channel_to_factor_indices(self, channel: str) -> List[int]:
        mapping = {
            "energy_constraint": [IDX_ENERGY],
            "sleep_rhythm": [IDX_SLEEP_DEBT, IDX_SLEEP_PHASE],
            "presence_modulation": [
                IDX_SOCIAL_WILL,
                IDX_WATCH_WILL,
                IDX_AVOIDANCE,
                IDX_QUIET_PREF,
                IDX_CURIOSITY,
                IDX_OUTWARD_ATT,
            ],
            "relation_trauma": [
                IDX_AFFECTION,
                IDX_TRUST,
                IDX_ANNOYANCE,
                IDX_TRAUMA,
                IDX_AVERSION,
            ],
            "topic_ownership": [IDX_TOPIC_OWN, IDX_TOPIC_CONF],
            "group_pattern": [IDX_PATTERN, IDX_PATTERN_CONF, IDX_SPECTATOR],
            "self_reference": [
                IDX_SELF_REF_STR,
                IDX_DISCUSS_N,
                IDX_QUOTE_N,
                IDX_DIRECT_AT,
            ],
            "narration_strategy": [IDX_NARR_STRAT, IDX_NARR_MOOD],
            "multimodal_throttle": [IDX_MEDIA_STORM],
            "spectator_convert": [IDX_SPECTATOR, IDX_EVENT_REL],
            "hourly_cap": [IDX_ENERGY, IDX_SLEEP_PHASE],
            "intent_shield": [IDX_INTENTION],
        }
        return mapping.get(channel, [])

    def _compute_hysteresis_bias(self) -> Dict[str, float]:
        """滞后偏差 —— 防止决策震荡

        如果最近 N 次都是正向决策（放行），则轻微增加负面通道权重（保守倾向）
        如果最近 N 次都是负向决策（拦截），则轻微降低负面通道权重（释放倾向）
        """
        pos = self._state.positive_decision_count
        neg = self._state.negative_decision_count
        total = self._state.total_decisions
        if total < 5:
            return {k: 1.0 for k in self._base_weights}
        pos_ratio = pos / max(total, 1)
        neg_ratio = neg / max(total, 1)
        bias = 1.0
        if pos_ratio > 0.78:
            bias = 1.0 + (pos_ratio - 0.78) * 0.25
        elif neg_ratio > 0.78:
            bias = max(0.75, 1.0 - (neg_ratio - 0.78) * 0.30)
        negative_channels = {
            "energy_constraint",
            "sleep_rhythm",
            "relation_trauma",
            "multimodal_throttle",
            "hourly_cap",
        }
        result = {}
        for ch in self._base_weights:
            if ch in negative_channels:
                result[ch] = bias
            else:
                result[ch] = 2.0 - bias
        return result

    def _compute_coupling_aggregation(self) -> Dict[str, float]:
        """计算每个因子被多少其他因子耦合（聚合度）"""
        agg = {}
        if self._state.coupling_matrix is None:
            return agg
        cm = self._state.coupling_matrix
        for j in range(FACTOR_DIM):
            incoming = sum(abs(cm[i][j]) for i in range(FACTOR_DIM))
            agg[_FACTOR_NAMES[j]] = incoming
        return agg

    # ──────────────────────────────────────────
    #  L5：单轮评估（被迭代调用）
    # ─────────────────────────────────────────-

    def _single_pass(
        self,
        inp: FactorInput,
        features: List[float],
        derivatives: List[float],
        coupling: List[List[float]],
        adaptive_weights: Dict[str, float],
        pass_num: int,
        prev_bar: float,
        prev_score: float,
    ) -> FactorVerdict:
        """执行一轮完整的因子融合计算

        每个通道的计算都受到：
          ① 自身基础权重（来自 adaptive_weights）
          ② 其他因子的耦合调制（来自 coupling 矩阵）
          ③ 时序导数的额外贡献
          ④ 迭代轮次的收敛压力（后续轮次逐渐缩小修正幅度）
        """
        v = FactorVerdict(breakdown={})
        convergence_damp = 1.0 / (1.0 + pass_num * 0.35)
        cw = adaptive_weights
        self._ch_energy(
            inp, features, derivatives, coupling, v, cw, convergence_damp
        )
        self._ch_sleep(
            inp, features, derivatives, coupling, v, cw, convergence_damp
        )
        self._ch_presence(
            features, derivatives, coupling, v, cw, convergence_damp
        )
        self._ch_relation(
            features, derivatives, coupling, v, cw, convergence_damp
        )
        self._ch_topic(
            inp, features, derivatives, coupling, v, cw, convergence_damp
        )
        self._ch_pattern(
            inp, features, derivatives, coupling, v, cw, convergence_damp
        )
        self._ch_selfref(
            inp, features, derivatives, coupling, v, cw, convergence_damp
        )
        self._ch_narration(
            inp, features, derivatives, coupling, v, cw, convergence_damp
        )
        self._ch_media(
            inp, features, derivatives, coupling, v, cw, convergence_damp
        )
        self._ch_spectator(
            inp, features, derivatives, coupling, v, cw, convergence_damp
        )
        self._ch_hourly(
            inp, features, derivatives, coupling, v, cw, convergence_damp
        )
        self._ch_intent(
            inp,
            features,
            derivatives,
            coupling,
            v,
            cw,
            convergence_damp,
            pass_num,
        )
        if pass_num > 1:
            blend_t = min(1.0, 0.55 + 0.12 * pass_num)
            v.bar_modifier = _lerp(prev_bar, v.bar_modifier, blend_t)
            v.score_modifier = _lerp(prev_score, v.score_modifier, blend_t)
        return v

    def _coupled_weight(
        self,
        factor_idx: int,
        channel_name: str,
        coupling: List[List[float]],
        base_w: float,
        damp: float,
    ) -> float:
        """计算单个因子的耦合后有效权重

        effective_w = base_w × (1 + Σ(C[j][factor_idx] × f[j]))
        即：基础权重 × (1 + 所有其他因子对它的耦合加权和)
        """
        coupled_sum = 0.0
        cm = coupling
        for j in range(FACTOR_DIM):
            c_val = cm[j][factor_idx]
            if abs(c_val) < 0.005:
                continue
            coupled_sum += c_val
        modulation = 1.0 + _tanh_scaled(coupled_sum, scale=1.5)
        return base_w * modulation * damp

    # ─── 各通道具体实现 ───

    def _ch_energy(
        self,
        inp: FactorInput,
        feat: List[float],
        deriv: List[float],
        coup: List[List[float]],
        v: FactorVerdict,
        cw: Dict[str, float],
        damp: float,
    ) -> None:
        e = feat[IDX_ENERGY]
        de = deriv[IDX_ENERGY]
        ew = self._coupled_weight(
            IDX_ENERGY,
            "energy_constraint",
            coup,
            cw.get("energy_constraint", 0.15),
            damp,
        )
        base_suppress = _sigmoid(
            -e, midpoint=self._state.adaptive_energy_threshold, steepness=7.0
        )
        derivative_penalty = 0.0
        if de < -0.15:
            derivative_penalty = (
                _sigmoid(-de, midpoint=0.2, steepness=4.0) * 0.12
            )
        combined_pressure = base_suppress + derivative_penalty
        score_mod = 0.62 + (1.0 - combined_pressure) * 0.38
        score_mod = _clamp(score_mod, 0.55, 1.02)
        bar_raise = combined_pressure * 0.24 * ew * 6.0
        v.score_modifier *= score_mod
        v.bar_modifier += bar_raise
        v.breakdown["energy_combined"] = round(combined_pressure, 5)
        v.breakdown["energy_deriv"] = round(de, 4)
        if e < 0.12 and not inp.intent_override_candidate:
            v.should_hard_block = True
            v.block_reason = f"深度耗尽(e={e:.3f}, dE/dt={de:.3f})"

    def _ch_sleep(
        self,
        inp: FactorInput,
        feat: List[float],
        deriv: List[float],
        coup: List[List[float]],
        v: FactorVerdict,
        cw: Dict[str, float],
        damp: float,
    ) -> None:
        sp = feat[IDX_SLEEP_PHASE]
        if sp < 0.08:
            return
        sd = feat[IDX_SLEEP_DEBT]
        dsd = deriv[IDX_SLEEP_DEBT]
        sw = self._coupled_weight(
            IDX_SLEEP_DEBT,
            "sleep_rhythm",
            coup,
            cw.get("sleep_rhythm", 0.12),
            damp,
        )
        aw = self._coupled_weight(
            IDX_SLEEP_PHASE, "sleep_rhythm", coup, 0.0, damp
        )
        # 基础压力 = 相位严重度 × (债务权重 + 时间衰减)
        phase_pressure = sp * (0.38 + sd * 0.42)
        # GAP-A：叠加熬夜分项压力（聊天/窥屏/思考/打断）
        op = _clamp01(inp.overnight_pressure)
        pressure_from_activity = op * 0.20
        debt_acceleration = 0.0
        if dsd > 0.08:
            debt_acceleration = (
                _sigmoid(dsd, midpoint=0.1, steepness=5.0) * 0.10
            )
        total_pressure = (
            phase_pressure + debt_acceleration + pressure_from_activity
        )
        # GAP-A：亢奋链状态调制——arousal 时压制衰减，burnthrough 时行为异常化
        arousal_state = getattr(inp, "arousal_chain_state", "normal")
        is_bt = getattr(inp, "is_burnthrough", False)
        bt_impulse = _clamp01(getattr(inp, "burnthrough_impulse", 0.0))
        arousal_mod = 1.0
        if arousal_state == "arousal":
            arousal_mod = 0.55
        elif arousal_state == "stubborn":
            arousal_mod = 0.78
        elif arousal_state == "drowsy":
            arousal_mod = 1.18
        elif is_bt:
            arousal_mod = 0.35 + bt_impulse * 0.50
        suppress = (
            _sigmoid(
                total_pressure,
                midpoint=self._state.adaptive_sleep_threshold,
                steepness=5.5,
            )
            * arousal_mod
        )
        score_mod = 1.0 - suppress * 0.42 * sw * 8.0
        bar_raise = suppress * 0.22 * (sw + aw) * 8.0
        # GAP-A：熬穿冲动——bar降低但score波动增大（想回但质量不稳定）
        if is_bt and bt_impulse > 0.4:
            bar_raise *= 1.0 - bt_impulse * 0.35
            score_mod = score_mod * (0.85 + bt_impulse * 0.30)
            v.breakdown["bt_impulse_effect"] = round(bt_impulse, 4)
        v.score_modifier *= max(0.40, score_mod)
        v.bar_modifier += bar_raise
        v.breakdown["sleep_pressure"] = round(total_pressure, 5)
        v.breakdown["sleep_debt_acc"] = round(debt_acceleration, 5)
        v.breakdown["overnight_press"] = round(pressure_from_activity, 5)
        v.breakdown["arousal_state"] = arousal_state
        if not inp.sleep_reply_budget:
            block_prob = _clamp01(total_pressure * 0.92)
            if block_prob > 0.68 and not inp.intent_override_candidate:
                v.should_hard_block = True
                v.block_reason = f"睡眠限次(pressure={
                    total_pressure:.3f}, arousal={arousal_state})"
            v.breakdown["sleep_block_prob"] = round(block_prob, 4)

    def _ch_presence(
        self,
        feat: List[float],
        deriv: List[float],
        coup: List[List[float]],
        v: FactorVerdict,
        cw: Dict[str, float],
        damp: float,
    ) -> None:
        indices = [
            IDX_SOCIAL_WILL,
            IDX_WATCH_WILL,
            IDX_AVOIDANCE,
            IDX_QUIET_PREF,
            IDX_CURIOSITY,
            IDX_OUTWARD_ATT,
        ]
        pos_coefs = [0.36, 0.24, 0.0, 0.0, 0.22, 0.18]
        neg_coefs = [0.0, 0.0, 0.42, 0.46, 0.0, 0.0]
        pw = cw.get("presence_modulation", 0.12)
        positive = 0.0
        negative = 0.0
        for idx, pc, nc in zip(indices, pos_coefs, neg_coefs, strict=True):
            val = feat[idx]
            dval = deriv[idx]
            cw_local = self._coupled_weight(
                idx, "presence_modulation", coup, pw, damp
            )
            effective_pc = pc * cw_local * 6.0
            effective_nc = nc * cw_local * 6.0
            dynamic_val = val
            if dval > 0.08:
                dynamic_val = val + dval * 0.15
            elif dval < -0.08:
                dynamic_val = val + dval * 0.10
            positive += max(0.0, dynamic_val) * effective_pc
            negative += max(0.0, dynamic_val) * effective_nc
        net_drive = positive - negative * 0.62
        bar_mod_raw = (net_drive - 0.22) * -0.14
        bar_direction = 1.0 if net_drive >= 0.0 else -1.0
        v.bar_modifier += _clamp01(abs(bar_mod_raw)) * bar_direction * pw * 8.0
        v.breakdown["presence_net"] = round(net_drive, 5)
        v.breakdown["presence_pos"] = round(positive, 5)
        v.breakdown["presence_neg"] = round(negative, 5)

    def _ch_relation(
        self,
        feat: List[float],
        deriv: List[float],
        coup: List[List[float]],
        v: FactorVerdict,
        cw: Dict[str, float],
        damp: float,
    ) -> None:
        trauma = feat[IDX_TRAUMA]
        annoy = feat[IDX_ANNOYANCE]
        aversion = feat[IDX_AVERSION]
        aff = feat[IDX_AFFECTION]
        trust = feat[IDX_TRUST]
        rw = cw.get("relation_trauma", 0.10)
        tw = self._coupled_weight(
            IDX_TRAUMA, "relation_trauma", coup, rw, damp
        )
        aw = self._coupled_weight(
            IDX_ANNOYANCE, "relation_trauma", coup, rw * 0.7, damp
        )
        trauma_amp = (
            _sigmoid(trauma, midpoint=0.22, steepness=5.5) * 0.17 * tw * 10.0
        )
        v.bar_modifier += trauma_amp
        annoy_suppress = (
            _sigmoid(annoy, midpoint=0.32, steepness=6.5) * 0.13 * aw * 10.0
        )
        v.score_modifier *= max(0.78, 1.0 - annoy_suppress)
        aversion_extra = (
            _sigmoid(aversion, midpoint=0.30, steepness=5.0) * 0.09
        )
        v.bar_modifier += aversion_extra
        rel_product = aff * trust
        if rel_product < 0.04:
            avoid_pen = (0.12 - aff * 0.08) * (0.12 - trust * 0.08) * rw * 10.0
            v.bar_modifier += avoid_pen
            v.breakdown["rel_avoid_pen"] = round(avoid_pen, 5)
        elif rel_product > 0.42:
            aff_boost = -0.09 * rel_product * rw * 8.0
            v.bar_modifier += aff_boost
            v.breakdown["rel_affinity"] = round(aff_boost, 5)
        v.breakdown["trauma_amp"] = round(trauma_amp, 5)
        v.breakdown["annoy_suppress"] = round(annoy_suppress, 5)

    def _ch_topic(
        self,
        inp: FactorInput,
        feat: List[float],
        deriv: List[float],
        coup: List[List[float]],
        v: FactorVerdict,
        cw: Dict[str, float],
        damp: float,
    ) -> None:
        own = feat[IDX_TOPIC_OWN]
        conf = feat[IDX_TOPIC_CONF]
        aff = feat[IDX_AFFECTION]
        tw = cw.get("topic_ownership", 0.08)
        ow = self._coupled_weight(
            IDX_TOPIC_OWN, "topic_ownership", coup, tw, damp
        )
        cw_conf = self._coupled_weight(
            IDX_TOPIC_CONF, "topic_ownership", coup, tw * 0.6, damp
        )
        base_mods_arr = [-0.14, -0.06, 0.02, 0.16, 0.08]
        seg = int(_clamp(own * 5.0, 0, 4))
        base = base_mods_arr[seg]
        aff_mod = (aff - 0.48) * -0.12 * own
        conf_boost = _smoothstep(0.25, 0.82, conf)
        final = base * conf_boost * (1.0 + aff_mod) * ow * cw_conf * 12.0
        v.bar_modifier += final
        own_labels = [
            "self",
            "shared_group",
            "target_user",
            "ambiguous",
            "external",
        ]
        label = own_labels[seg] if seg < len(own_labels) else "unknown"
        v.breakdown[f"topic_{label}"] = round(final, 5)

    def _ch_pattern(
        self,
        inp: FactorInput,
        feat: List[float],
        deriv: List[float],
        coup: List[List[float]],
        v: FactorVerdict,
        cw: Dict[str, float],
        damp: float,
    ) -> None:
        pat = feat[IDX_PATTERN]
        pconf = feat[IDX_PATTERN_CONF]
        spec = feat[IDX_SPECTATOR]
        erel = feat[IDX_EVENT_REL]
        pw = cw.get("group_pattern", 0.09)
        pat_w = self._coupled_weight(
            IDX_PATTERN, "group_pattern", coup, pw, damp
        )
        spec_w = self._coupled_weight(
            IDX_SPECTATOR, "group_pattern", coup, pw * 0.8, damp
        )
        base_pat = (pat - 0.50) * 0.34
        if spec > 0.5:
            flip_force = (erel - 0.38) * 0.18
            flip_curve = _tanh_scaled(flip_force, scale=0.15)
            base_pat = 0.04 + flip_curve * spec_w * 6.0
            v.breakdown["spec_flip"] = round(flip_curve, 5)
        conf_envelope = _smoothstep(0.20, 0.78, pconf)
        final_pat = base_pat * conf_envelope * pat_w * 10.0
        v.bar_modifier += final_pat
        if abs(final_pat) > 0.003:
            v.breakdown["pattern_effect"] = round(final_pat, 5)

    def _ch_selfref(
        self,
        inp: FactorInput,
        feat: List[float],
        deriv: List[float],
        coup: List[List[float]],
        v: FactorVerdict,
        cw: Dict[str, float],
        damp: float,
    ) -> None:
        strength = feat[IDX_SELF_REF_STR]
        disc_n = feat[IDX_DISCUSS_N]
        quote_n = feat[IDX_QUOTE_N]
        direct = feat[IDX_DIRECT_AT]
        srw = cw.get("self_reference", 0.06)
        str_w = self._coupled_weight(
            IDX_SELF_REF_STR, "self_reference", coup, srw, damp
        )
        disc_w = self._coupled_weight(
            IDX_DISCUSS_N, "self_reference", coup, srw * 0.85, damp
        )
        base_boost = strength * 0.07 * str_w * 10.0
        discuss_bonus = (
            _sigmoid(disc_n, midpoint=0.35, steepness=4.5)
            * 0.14
            * disc_w
            * 8.0
        )
        quote_bonus = _clamp01(quote_n) * 0.04 * str_w * 8.0
        direct_bonus = direct * 0.03 * str_w * 8.0
        total_boost = -(
            base_boost + discuss_bonus + quote_bonus + direct_bonus
        )
        v.bar_modifier += total_boost
        if disc_n > 0.35 or (strength > 0.52 and len(inp.self_ref_types) >= 2):
            score_add = (discuss_bonus + base_boost) * 0.42
            v.score_modifier += score_add
        v.breakdown["selfref_total"] = round(total_boost, 5)

    def _ch_narration(
        self,
        inp: FactorInput,
        feat: List[float],
        deriv: List[float],
        coup: List[List[float]],
        v: FactorVerdict,
        cw: Dict[str, float],
        damp: float,
    ) -> None:
        strat = feat[IDX_NARR_STRAT]
        mood = feat[IDX_NARR_MOOD]
        nrw = cw.get("narration_strategy", 0.04)
        strat_w = self._coupled_weight(
            IDX_NARR_STRAT, "narration_strategy", coup, nrw, damp
        )
        mood_w = self._coupled_weight(
            IDX_NARR_MOOD, "narration_strategy", coup, nrw * 0.7, damp
        )
        strat_mod_raw = (strat - 0.45) * 0.18 * strat_w * 12.0
        v.bar_modifier += strat_mod_raw
        if strat > 0.55:
            v.score_modifier *= max(0.82, 1.0 - strat_mod_raw * 0.38)
        if mood > 0.55:
            mood_p = (mood - 0.50) * 0.07 * mood_w * 10.0
            v.bar_modifier += mood_p
            v.breakdown["narr_mood_eff"] = round(mood_p, 5)
        if abs(strat_mod_raw) > 0.002:
            v.breakdown["narr_strat_eff"] = round(strat_mod_raw, 5)

    def _ch_media(
        self,
        inp: FactorInput,
        feat: List[float],
        deriv: List[float],
        coup: List[List[float]],
        v: FactorVerdict,
        cw: Dict[str, float],
        damp: float,
    ) -> None:
        storm = feat[IDX_MEDIA_STORM]
        mw = cw.get("multimodal_throttle", 0.03)
        storm_w = self._coupled_weight(
            IDX_MEDIA_STORM, "multimodal_throttle", coup, mw, damp
        )
        if storm > 0.5:
            storm_p = 0.14 * storm_w * 10.0
            v.bar_modifier += storm_p
            v.breakdown["media_storm_eff"] = round(storm_p, 5)
        total_items = max(0, inp.media_total_items)
        skipped = max(0, inp.media_skipped_count)
        if total_items >= 5 and skipped > 0:
            skip_rate = skipped / total_items
            throttle = (
                _sigmoid(skip_rate, midpoint=0.48, steepness=5.5)
                * 0.10
                * mw
                * 12.0
            )
            v.bar_modifier += throttle
            v.breakdown["media_throttle_eff"] = round(throttle, 5)
        if total_items > 0 and skipped >= total_items:
            v.breakdown["media_all_skip"] = 1.0

    def _ch_spectator(
        self,
        inp: FactorInput,
        feat: List[float],
        deriv: List[float],
        coup: List[List[float]],
        v: FactorVerdict,
        cw: Dict[str, float],
        damp: float,
    ) -> None:
        erel = feat[IDX_EVENT_REL]
        spec = feat[IDX_SPECTATOR]
        if spec < 0.5 or erel <= 0.40:
            return
        scw = cw.get("spectator_convert", 0.025)
        engage_bonus = (erel - 0.38) * 0.10 * scw * 15.0
        v.score_modifier += engage_bonus
        v.breakdown["spec_engage"] = round(engage_bonus, 5)

    def _ch_hourly(
        self,
        inp: FactorInput,
        feat: List[float],
        deriv: List[float],
        coup: List[List[float]],
        v: FactorVerdict,
        cw: Dict[str, float],
        damp: float,
    ) -> None:
        used = max(0, inp.hourly_reply_used)
        e = feat[IDX_ENERGY]
        sp = feat[IDX_SLEEP_PHASE]
        hcw = cw.get("hourly_cap", 0.025)
        e_disc = 0.32 + e * 0.68
        if sp > 0.80:
            s_disc = 0.0
        elif sp > 0.45:
            s_disc = 0.10
        else:
            s_disc = 1.0
        dyn_cap = 8.0 * e_disc * s_disc
        dyn_cap = max(0.0, dyn_cap)
        usage_ratio = used / max(dyn_cap, 1.0)
        if usage_ratio >= 1.0:
            if not inp.intent_override_candidate:
                v.should_hard_block = True
                v.block_reason = f"每小时上限(cap={dyn_cap:.1f}, used={used})"
            v.breakdown["hr_cap_hit"] = -1.0
        elif usage_ratio > 0.72:
            near_cap = (
                _sigmoid(usage_ratio - 0.72, midpoint=0.14, steepness=8.5)
                * 0.10
                * hcw
                * 15.0
            )
            v.bar_modifier += near_cap
            v.breakdown["hr_near_cap"] = round(near_cap, 5)
        v.breakdown["hr_dyn_cap"] = round(dyn_cap, 2)
        v.breakdown["hr_usage"] = round(usage_ratio, 4)

    def _ch_intent(
        self,
        inp: FactorInput,
        feat: List[float],
        deriv: List[float],
        coup: List[List[float]],
        v: FactorVerdict,
        cw: Dict[str, float],
        damp: float,
        pass_num: int,
    ) -> None:
        if not inp.intent_override_candidate:
            return
        drive = feat[IDX_INTENTION]
        if drive < 0.74:
            return
        ishw = cw.get("intent_shield", 0.018)
        shield = _sigmoid(drive - 0.74, midpoint=0.10, steepness=11.0)
        pass_boost = 1.0 + (pass_num - 1) * 0.08
        shield *= min(1.5, pass_boost)
        if v.bar_modifier > 0.001:
            v.bar_modifier *= 1.0 - shield * 0.72 * ishw * 30.0
        if v.score_modifier < 0.995:
            v.score_modifier = 1.0 - (1.0 - v.score_modifier) * (
                1.0 - shield * 0.82 * ishw * 30.0
            )
        v.breakdown["intent_shield"] = round(shield, 5)

    # ──────────────────────────────────────────
    #  L6：随机微扰层
    # ─────────────────────────────────────────-

    def _apply_stochastic_perturbation(
        self, v: FactorVerdict, features: List[float]
    ) -> None:
        """受控随机微扰 —— 低置信度时引入探索噪声

        设计原则：
          - 高置信度（因子值远离边界）：几乎不加噪声
          - 中等置信度（因子值在中间区）：微量噪声
          - 低置信度（因子值接近边界/冲突）：较大噪声
          - 噪声幅度永远不超过 ±0.015（防止失控）
        """
        entropy = 0.0
        for f in features:
            if 0.15 < f < 0.85:
                entropy += 1.0
            elif 0.05 < f < 0.95:
                entropy += 0.5
        entropy /= FACTOR_DIM
        uncertainty = 1.0 - entropy
        noise_scale = uncertainty * uncertainty * 0.012
        if noise_scale < 0.0005:
            return
        bar_noise = _gaussian_noise(0.0, noise_scale)
        score_noise = _gaussian_noise(1.0, noise_scale * 0.5)
        v.bar_modifier += bar_noise
        v.score_modifier *= score_noise
        v.breakdown["stochastic_bar"] = round(bar_noise, 6)
        v.breakdown["stochastic_score"] = round(score_noise, 6)

    # ──────────────────────────────────────────
    #  辅助方法
    # ─────────────────────────────────────────-

    def _compute_confidence(
        self, v: FactorVerdict, features: List[float]
    ) -> float:
        """计算最终决策置信度

        置信度高意味着：
          - 大多数因子指向同一方向（一致性强）
          - 没有因子处于临界模糊区
          - 耦合激活度适中（不过度依赖耦合）
        """
        bar_sign = 1.0 if v.bar_modifier >= 0 else -1.0
        agreement = 0.0
        total_abs = 0.0
        for key, val in v.breakdown.items():
            if key.startswith("stochastic"):
                continue
            if abs(val) < 0.001:
                continue
            total_abs += abs(val)
            if val * bar_sign >= 0:
                agreement += abs(val)
        if total_abs < 0.001:
            return 0.5
        directional_agreement = agreement / total_abs
        boundary_proximity = 0.0
        for f in features:
            bp = min(f, 1.0 - f) * 2.0
            boundary_proximity += bp
        boundary_proximity /= FACTOR_DIM
        confidence = (
            directional_agreement * 0.65 + (1.0 - boundary_proximity) * 0.35
        )
        return _clamp01(confidence)

    def _measure_coupling_activation(
        self, coupling: List[List[float]], features: List[float]
    ) -> float:
        """测量耦合矩阵的整体激活强度"""
        total = 0.0
        count = 0
        for i in range(FACTOR_DIM):
            for j in range(FACTOR_DIM):
                c = coupling[i][j]
                if abs(c) > 0.005:
                    total += abs(c)
                    count += 1
        if count == 0:
            return 0.0
        return total / count

    def _update_feedback_state(self, v: FactorVerdict) -> None:
        """更新内部反馈状态（用于下次评估的自适应权重计算）"""
        self._state.bar_history.append(v.bar_modifier)
        self._state.score_history.append(v.score_modifier)
        self._state.total_decisions += 1
        if v.should_hard_block or v.bar_modifier > 0.05:
            self._state.negative_decision_count += 1
        else:
            self._state.positive_decision_count += 1
        if self._state.total_decisions % 15 == 0:
            self._drift_adaptive_thresholds()
        self._state.coupling_dirty = True

    def _drift_adaptive_thresholds(self) -> None:
        """自适应阈值缓慢漂移 —— 让系统跟随环境变化

        如果最近 bar_modifier 平均值为正（整体偏向拦截），则放宽能量阈值
        如果最近 bar_modifier 平均值为负（整体偏向放行），则收紧能量阈值
        """
        if len(self._state.bar_history) < 8:
            return
        recent_bars = list(self._state.bar_history)[-10:]
        avg_bar = sum(recent_bars) / len(recent_bars)
        drift_speed = 0.008
        if avg_bar > 0.04:
            self._state.adaptive_energy_threshold = _clamp(
                self._state.adaptive_energy_threshold - drift_speed,
                0.20,
                0.50,
            )
            self._state.adaptive_sleep_threshold = _clamp(
                self._state.adaptive_sleep_threshold - drift_speed * 0.7,
                0.35,
                0.65,
            )
        elif avg_bar < -0.04:
            self._state.adaptive_energy_threshold = _clamp(
                self._state.adaptive_energy_threshold + drift_speed,
                0.20,
                0.50,
            )
            self._state.adaptive_sleep_threshold = _clamp(
                self._state.adaptive_sleep_threshold + drift_speed * 0.7,
                0.35,
                0.65,
            )


# ──────────────────────────────────────────────
#  全局单例（线程安全，基于统一单例工具）
# ──────────────────────────────────────────────

_engine_instance: Optional[MultiFactorDecisionEngine] = None
_engine_lock = _get_class_lock("MultiFactorDecisionEngine")


def get_multi_factor_engine() -> MultiFactorDecisionEngine:
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = MultiFactorDecisionEngine()
                logger.info(
                    "[ACFN引擎] 自适应耦合因子网络初始化完成 — 28维/12通道/5轮迭代"
                )
    return _engine_instance

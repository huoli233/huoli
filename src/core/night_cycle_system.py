import time
import math
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger("夜间状态")

_night_system_instances: Dict[str, "NightCycleSystem"] = {}

# ──────────────────────────────────────────────
#  ACFN级数学工具库（数值稳定/有界输出）
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
    if edge1 == edge0:
        return 0.0 if x < edge0 else 1.0
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


# 三维引擎维度索引
IDX_DROWSINESS = 0
IDX_PRESSURE = 1
IDX_RESERVE = 2
DIM_3D = 3

_DIM_NAMES = ["drowsiness", "pressure", "reserve"]


class NightPhase(Enum):
    """夜间节律阶段，第三阶段升级：增加更多精细化阶段"""

    AWAKE = "awake"
    DROWSY = "drowsy"
    LIGHT_SLEEP = "light_sleep"
    DEEP_SLEEP = "deep_sleep"
    EXCITED = "excited"
    BURNED_OUT = "burned_out"
    NIGHT_ACTIVE = "night_active"
    MIDNIGHT_REFLECT = "midnight_reflect"
    DAWN_RECOVER = "dawn_recover"
    SOCIAL_NIGHT = "social_night"
    QUIET_CONTEMPLATE = "quiet_contemplate"

    def label(self) -> str:
        labels = {
            NightPhase.AWAKE: "清醒",
            NightPhase.DROWSY: "犯困",
            NightPhase.LIGHT_SLEEP: "浅睡",
            NightPhase.DEEP_SLEEP: "深睡",
            NightPhase.EXCITED: "亢奋",
            NightPhase.BURNED_OUT: "熬穿",
            NightPhase.NIGHT_ACTIVE: "晚上阶段",
            NightPhase.MIDNIGHT_REFLECT: "凌晨阶段",
            NightPhase.DAWN_RECOVER: "清晨恢复",
            NightPhase.SOCIAL_NIGHT: "夜间社交",
            NightPhase.QUIET_CONTEMPLATE: "凌晨沉思",
        }
        return labels.get(self, self.value)

    @property
    def is_sleeping(self) -> bool:
        return self in (NightPhase.LIGHT_SLEEP, NightPhase.DEEP_SLEEP)

    @property
    def can_interact(self) -> bool:
        return self not in (NightPhase.DEEP_SLEEP, NightPhase.BURNED_OUT)

    @property
    def is_night_social(self) -> bool:
        return self in (
            NightPhase.NIGHT_ACTIVE,
            NightPhase.SOCIAL_NIGHT,
            NightPhase.MIDNIGHT_REFLECT,
        )

    @property
    def is_reflective(self) -> bool:
        return self in (
            NightPhase.MIDNIGHT_REFLECT,
            NightPhase.QUIET_CONTEMPLATE,
            NightPhase.DAWN_RECOVER,
        )


class EnergyMode(Enum):
    """能量条双形态：白天/夜间使用不同的恢复和扣除参数"""

    DAY = "day"
    NIGHT = "night"
    DAWN_TRANSITION = "dawn_transition"
    EVENING_TRANSITION = "evening_transition"


@dataclass
class NightCycleState:
    """夜间节律内部状态：三维复合困倦引擎(CDE)

    三维模型:
      困意值(drowsiness_value): 0~100, 正弦波昼夜调制+随机扰动+间歇性更新
      熬夜压力(overnight_pressure): 0~100, 从指定时间起指数加速累积
      睡眠储备(sleep_reserve): 0~100, 睡眠时恢复/清醒消耗/回复扣减

    核心公式: composite_load = D × P / √(S + ε)
    当 composite_load > collapse_threshold → 熬不住了(必须睡)
    S越高→分母越大→复合负载越低→能多撑一会 (相乘相消)
    """

    current_phase: NightPhase = NightPhase.AWAKE
    sleepiness: float = 0.0
    excitement_counter: float = 0.0
    burn_accumulated: float = 0.0
    phase_entered_at: float = field(default_factory=time.time)
    last_stimulus_at: float = 0.0
    forced_awake: bool = False
    night_reply_count: int = 0
    night_reply_cap: int = 5
    night_cap_reset_at: float = 0.0
    peek_window_open: bool = False
    peek_window_opened_at: float = 0.0
    peek_window_duration: float = 30.0
    last_peek_window_at: float = 0.0
    peek_window_interval: float = 1200.0
    sleep_reply_used: int = 0
    sleep_reply_window_start: float = 0.0
    sleep_reply_window_sec: float = 3600.0
    night_social_energy: float = 50.0
    night_social_threshold: float = 30.0
    reflection_depth: float = 0.0
    last_reflection_at: float = 0.0
    dawn_recovery_progress: float = 0.0
    night_observation_mode: bool = False
    observation_start_at: float = 0.0
    sleep_debt: float = 0.0
    last_sleep_quality: float = 1.0
    daily_fatigue: float = 0.0
    daily_fatigue_limit: float = 100.0
    fatigue_per_reply: float = 2.0
    fatigue_recovery_rate: float = 5.0
    last_fatigue_reset_date: str = ""
    in_night_mode: bool = False
    # ── 三阶段睡眠时间线 ──
    sleep_stage: str = "awake"
    sleep_started_at: float = 0.0
    light_sleep_duration_min: float = 30.0
    deep_sleep_duration_min: float = 180.0
    rem_sleep_duration_min: float = 90.0
    # ── 唤醒/打断 ──
    interruption_count: int = 0
    interruption_energy_penalty: float = 1.0
    stay_in_bed_probability: float = 0.7
    full_energy_threshold: float = 9.0
    # ── 夜间疲劳加速（19:00起） ──
    night_fatigue_start_hour: int = 19
    debt_accumulated_at: float = field(default_factory=time.time)
    # ── GAP-A 第四阶段新增：熬夜压力分项追踪 ──
    pressure_chat_minutes: float = 0.0
    pressure_peek_minutes: float = 0.0
    pressure_think_intensity: float = 0.0
    pressure_interrupt_count: int = 0
    pressure_last_interrupt_at: float = 0.0
    total_overnight_pressure: float = 0.0
    # ── GAP-A：亢奋控制器状态 ──
    stubborn_duration: float = 0.0
    stubborn_started_at: float = 0.0
    arousal_trigger_type: str = ""
    arousal_stimulus_strength: float = 0.0
    arousal_consecutive_pulls: int = 0
    is_burnthrough_active: bool = False
    burnthrough_started_at: float = 0.0
    burnthrough_max_duration: float = 1500.0
    # ── GAP-A：熬穿表达变形参数 ──
    expression_punctuation_density: float = 1.0
    expression_sentence_fragmentation: float = 0.0
    expression_emotion_leakage: float = 0.0
    expression_impulse_level: float = 0.0

    # ════════════════════════════════════════════
    #  ★★★ 三维困倦引擎(CDE)核心字段 ★★★
    # ════════════════════════════════════════════

    # ── 维度一：困意值 (Drowsiness Value) ──
    drowsiness_value: float = 0.0
    drowsiness_base_rate: float = 0.35
    drowsiness_noise_amplitude: float = 0.12
    # 正弦波昼夜调制参数
    circadian_phase_offset: float = 0.0
    circadian_peak_hour: float = 3.0
    circadian_trough_hour: float = 14.0
    # 间歇性更新控制
    last_drowsiness_tick: float = field(default_factory=time.time)
    drowsiness_tick_interval: float = 6.0
    drowsiness_pending_delta: float = 0.0

    # ── 维度二：熬夜压力 (Overnight Pressure) ──
    overnight_pressure: float = 0.0
    pressure_start_hour: int = 22
    pressure_acceleration_base: float = 0.08
    pressure_exponential_factor: float = 0.15
    pressure_activity_multiplier: float = 1.0
    pressure_last_tick: float = field(default_factory=time.time)
    pressure_critical_threshold: float = 85.0
    # 压力累积历史（用于计算加速度导数）
    pressure_history: Dict[float, float] = field(default_factory=dict)
    pressure_history_max_len: int = 20

    # ── 维度三：睡眠储备 (Sleep Reserve) ──
    sleep_reserve: float = 55.0
    sleep_reserve_max: float = 100.0
    sleep_reserve_min: float = 0.0
    # 恢复速率参数
    light_sleep_recovery_rate: float = 0.4
    deep_sleep_recovery_rate: float = 1.2
    awake_drain_rate: float = 0.03
    reply_cost: float = 0.8
    # 睡眠储备质量因子（影响相乘相消的分母权重）
    reserve_quality_factor: float = 1.0

    # ── 复合判断输出 ──
    composite_load: float = 0.0
    collapse_threshold: float = 50.0
    collapse_imminent: bool = False
    collapse_time_estimate: float = 0.0

    # ── 行为链输出字段 ──
    laziness_probability: float = 0.0
    wake_irritation_level: float = 0.0
    half_asleep_level: float = 0.0
    yawn_frequency: float = 0.0
    response_suppression_coef: float = 0.0
    body_state_tag: str = ""
    monologue_mood_hint: str = ""

    # ── 强制唤醒痕迹追踪 ──
    force_wake_count_today: int = 0
    last_force_wake_time: float = 0.0
    force_wake_irritation_decay: float = 0.0
    # 赖床状态追踪
    in_lazy_state: bool = False
    lazy_state_started: float = 0.0
    lazy_resistance_cycles: int = 0

    # ════════════════════════════════════════════
    #  ★★★ 能量条双形态系统 (Energy Dual-Mode) ★★★
    # ════════════════════════════════════════════

    energy_value: float = 10.0
    energy_max: float = 10.0
    energy_mode: str = "day"
    # ── 白天模式参数 ──
    day_recovery_rate_per_min: float = 0.05
    day_reply_cost_base: float = 0.3
    day_think_cost_base: float = 0.15
    day_activity_cost_base: float = 0.08
    noon_event_recovery_bonus: float = 2.0
    morning_boost_hours: Tuple[int, int] = (7, 11)
    afternoon_slump_hours: Tuple[int, int] = (13, 15)
    # ── 夜间模式参数 ──
    night_drain_multiplier: float = 1.8
    night_reply_cost_multiplier: float = 2.5
    night_think_cost_multiplier: float = 2.0
    night_protection_threshold: float = 85.0
    night_overwork_penalty_rate: float = 0.008
    # ── 过渡时段(黄昏/黎明) ──
    transition_blend_factor: float = 0.0

    # ════════════════════════════════════════════
    #  ★★★ 睡眠值→能量条映射 (Sleep→Energy Map) ★★★
    # ════════════════════════════════════════════

    sleep_to_energy_ratio: float = 0.12
    sleep_to_energy_quality_weight: float = 0.6
    sleep_to_energy_time_discount: float = 0.0
    energy_from_last_sleep: float = 0.0
    last_sleep_mapping_time: float = 0.0
    mapping_efficiency_curve: str = "logistic"

    # ════════════════════════════════════════════
    #  ★★★ 动态预知唤醒 (Predictive Wake) ★★★
    # ════════════════════════════════════════════

    predictive_wake_enabled: bool = True
    predictive_wake_confidence: float = 0.0
    predicted_wake_time: float = 0.0
    wake_readiness_score: float = 0.0
    reserve_sufficiency_threshold: float = 65.0
    drowsiness_falloff_threshold: float = 25.0
    pressure_dissipation_target: float = 20.0
    natural_wake_anticipation_sec: float = 600.0
    wake_decision_made: bool = False
    early_wake_reason: str = ""

    # ════════════════════════════════════════════
    #  ★★★ 特殊事件/敏感话题系统 ★★★
    # ════════════════════════════════════════════

    sensitive_topic_counter: int = 0
    sensitive_topic_threshold: int = 5
    sensitive_topic_tags: Dict[str, int] = field(default_factory=dict)
    special_event_triggered: bool = False
    special_event_time: float = 0.0
    weakness_active: bool = False
    weakness_end_time: float = 0.0
    weakness_duration_base: float = 1800.0
    weakness_fatigue_cost: float = 90.0
    # ── 临时Buff系统 ──
    active_buffs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    buff_coffee_fatigue_limit_bonus: float = 40.0
    buff_coffee_duration: float = 7200.0
    buff_water_energy_bonus: float = 3.0
    # ── 随机事件状态 ──
    daily_event_count: int = 0
    daily_event_limit: int = 5
    last_event_check: float = field(default_factory=time.time)
    event_check_interval: float = 600.0
    pending_event_message: str = ""

    # ════════════════════════════════════════════
    #  ★★★ 多维度复杂因素成本/恢复引擎 ★★★
    # ════════════════════════════════════════════

    think_active_value: float = 100.0
    think_cost_per_unit: float = 0.02
    conversation_complexity: float = 1.0
    conversation_length_factor: float = 1.0
    emotional_intensity_factor: float = 1.0
    event_recovery_pool: float = 0.0
    event_recovery_cap: float = 15.0
    time_segment_modifier: float = 1.0
    cumulative_today_cost: float = 0.0
    cumulative_today_recovery: float = 0.0
    cost_history: Dict[float, float] = field(default_factory=dict)
    recovery_history: Dict[float, float] = field(default_factory=dict)

    # ════════════════════════════════════════════
    #  ★★★ ACFN级算法引擎扩展字段 ★★★
    # ════════════════════════════════════════════

    # ── 导数感知层：三维度时序变化率追踪 ──
    last_d_value: float = 0.0
    last_p_value: float = 0.0
    last_s_value: float = 0.0
    derivative_timestamp: float = 0.0
    derivative_decay_sec: float = 60.0
    drowsiness_derivative: float = 0.0
    pressure_derivative: float = 0.0
    reserve_derivative: float = 0.0

    # ── 三维耦合矩阵缓存 ──
    coupling_matrix_3d: Optional[List[List[float]]] = None
    coupling_dirty_flag: bool = True

    # ── 滞后反馈回路：复合负载输出历史 ──
    composite_output_history: List[float] = field(default_factory=list)
    composite_history_max_len: int = 15
    positive_composite_count: int = 0
    negative_composite_count: int = 0
    total_composite_evaluations: int = 0
    hysteresis_correction_active: bool = False
    last_hysteresis_direction: int = 0

    # ── 自适应阈值漂移 ──
    adaptive_collapse_base: float = 50.0
    threshold_drift_accumulator: float = 0.0
    last_drift_time: float = 0.0
    drift_evaluation_interval: int = 12
    drift_speed: float = 0.006
    drift_max_amplitude: float = 12.0

    # ── 迭代收敛状态 ──
    iteration_passes_last: int = 0
    convergence_achieved: bool = False
    last_convergence_delta_bar: float = 0.0
    last_convergence_delta_score: float = 0.0

    # ── 置信度与随机微扰 ──
    composite_confidence: float = 0.5
    perturbation_entropy: float = 0.0
    perturbation_applied: float = 0.0

    # ── 耦合激活快照（供调试/日志使用） ──
    coupling_activation_snapshot: Dict[str, float] = field(
        default_factory=dict
    )
    derivative_snapshot: Dict[str, float] = field(default_factory=dict)
    overwork_protection_active: bool = False

    def phase_duration_sec(self) -> float:
        return time.time() - self.phase_entered_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.current_phase.value,
            "phase_label": self.current_phase.label(),
            "sleepiness": round(self.sleepiness, 2),
            "excitement_counter": round(self.excitement_counter, 2),
            "burn_accumulated": round(self.burn_accumulated, 2),
            "phase_duration_sec": round(self.phase_duration_sec(), 1),
            "forced_awake": self.forced_awake,
            "night_reply_count": self.night_reply_count,
            "night_reply_cap": self.night_reply_cap,
            "peek_window_open": self.peek_window_open,
            "night_social_energy": round(self.night_social_energy, 2),
            "reflection_depth": round(self.reflection_depth, 2),
            "dawn_recovery_progress": round(self.dawn_recovery_progress, 2),
            "night_observation_mode": self.night_observation_mode,
            "sleep_debt": round(self.sleep_debt, 2),
            # ── 三维困倦引擎(CDE)快照 ──
            "drowsiness_value": round(self.drowsiness_value, 2),
            "overnight_pressure": round(self.overnight_pressure, 2),
            "sleep_reserve": round(self.sleep_reserve, 2),
            "composite_load": round(self.composite_load, 3),
            "collapse_threshold": round(self.collapse_threshold, 2),
            "collapse_imminent": self.collapse_imminent,
            "laziness_prob": round(self.laziness_probability, 3),
            "wake_irritation": round(self.wake_irritation_level, 3),
            "half_asleep": round(self.half_asleep_level, 3),
            "yawn_freq": round(self.yawn_frequency, 3),
            "response_suppression": round(self.response_suppression_coef, 3),
            "body_state": self.body_state_tag,
        }


class NightCycleSystem:
    """夜间节律系统，第三阶段升级：精细化夜间行为模式，支持社交夜、反思夜、观察夜等
    根据当前时间、活跃信号、精力状态、社交语境等多因素判定夜间阶段。
    状态机约束扩展：
        awake → drowsy → light_sleep → deep_sleep
        awake → excited → burned_out
        awake → night_active → social_night（社交活跃）
        awake → midnight_reflect → quiet_contemplate（反思模式）
        drowsy → awake（被刺激唤醒）
        light_sleep → awake（强刺激唤醒）
        deep_sleep 不可被直接唤醒（需要等待自然醒来）
        burned_out → deep_sleep（强制休息）
        night_active → dawn_recover（黎明恢复）
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._state = NightCycleState()
        # 可配置参数
        self._sleep_start_hour = 23
        self._sleep_end_hour = 7
        self._drowsy_threshold = 40.0
        self._light_sleep_threshold = 65.0
        self._deep_sleep_threshold = 85.0
        self._excitement_ceiling = 80.0
        self._burnout_threshold = 100.0
        # 第三阶段新增参数
        self._night_social_start = 20
        self._midnight_reflect_start = 0
        self._dawn_recover_start = 4
        self._reflection_build_rate = 0.1
        self._social_energy_decay = 0.05

    @property
    def phase(self) -> NightPhase:
        return self._state.current_phase

    @property
    def state_snapshot(self) -> NightCycleState:
        return self._state

    def resolve_time_band(self, hour: Optional[int] = None) -> Dict[str, Any]:
        """返回前端和心流共用的作息分区。"""
        h = time.localtime().tm_hour if hour is None else int(hour) % 24
        if 13 <= h < 17:
            band = ("afternoon", "下午", "午后低谷")
        elif 17 <= h < 20:
            band = ("early_evening", "傍晚", "傍晚过渡")
        elif 20 <= h < 23:
            band = ("evening", "晚上", "晚上阶段")
        elif 23 <= h or h < 2:
            band = ("late_night", "半夜", "半夜熬夜")
        elif 2 <= h < 5:
            band = ("midnight", "凌晨", "凌晨深夜")
        elif 5 <= h < 7:
            band = ("dawn", "清晨", "清晨恢复")
        elif 7 <= h < 11:
            band = ("morning", "上午", "上午清醒")
        elif 11 <= h < 13:
            band = ("noon", "中午", "中午平稳")
        else:
            band = ("daytime", "白天", "白天平稳")
        return {"key": band[0], "label": band[1], "description": band[2], "hour": h}

    def update_night_social_energy(
        self, stimulus: float, group_activity: float
    ) -> None:
        """更新夜间社交能量，第三阶段新增"""
        now = time.time()
        last_stimulus_at = self._state.last_stimulus_at or now
        decay = (
            self._social_energy_decay
            * (now - last_stimulus_at)
            / 60.0
        )
        self._state.night_social_energy = max(
            0.0,
            min(
                100.0,
                self._state.night_social_energy
                - decay
                + stimulus
                + group_activity * 0.5,
            ),
        )
        self._state.last_stimulus_at = now

    def update_reflection_depth(
        self, quiet_time: float, emotional_intensity: float
    ) -> None:
        """更新反思深度，第三阶段新增"""
        if quiet_time > 10:  # 安静时间超过10分钟
            build_rate = self._reflection_build_rate * (
                1.0 + emotional_intensity * 0.5
            )
            self._state.reflection_depth = min(
                1.0, self._state.reflection_depth + build_rate
            )
            self._state.last_reflection_at = time.time()
        else:
            # 逐渐衰减反思深度
            decay = (
                0.01 * (time.time() - self._state.last_reflection_at) / 60.0
            )
            self._state.reflection_depth = max(
                0.0, self._state.reflection_depth - decay
            )

    def enter_observation_mode(self) -> None:
        """进入夜间观察模式，第三阶段新增"""
        if not self._state.night_observation_mode:
            self._state.night_observation_mode = True
            self._state.observation_start_at = time.time()
            logger.debug(f"[夜间节律] {self._channel_id} 进入夜间观察模式")

    def exit_observation_mode(self) -> None:
        """退出夜间观察模式，第三阶段新增"""
        if self._state.night_observation_mode:
            self._state.night_observation_mode = False
            duration = time.time() - self._state.observation_start_at
            logger.debug(
                f"[夜间节律] {
                    self._channel_id} 退出夜间观察模式，持续 {
                    duration:.1f} 秒"
            )

    def get_night_behavior_modifier(self) -> Dict[str, float]:
        """获取夜间行为调节因子，第三阶段新增"""
        phase = self._state.current_phase
        base_modifiers = {
            "response_probability": 1.0,
            "social_openness": 1.0,
            "reflection_bonus": 0.0,
            "observation_focus": 1.0,
        }
        if phase.is_night_social:
            base_modifiers.update(
                {
                    "response_probability": 1.2,
                    "social_openness": 1.3,
                    "reflection_bonus": 0.1,
                }
            )
        elif phase.is_reflective:
            base_modifiers.update(
                {
                    "response_probability": 0.7,
                    "social_openness": 0.8,
                    "reflection_bonus": 0.5,
                    "observation_focus": 1.4,
                }
            )
        elif phase == NightPhase.DAWN_RECOVER:
            base_modifiers.update(
                {
                    "response_probability": 0.9,
                    "social_openness": 1.1,
                    "reflection_bonus": 0.3,
                }
            )
        return base_modifiers

    def is_night_hours(self, hour: Optional[int] = None) -> bool:
        """查看是否处于夜间时段"""
        if hour is None:
            import datetime

            hour = datetime.datetime.now().hour
        if self._sleep_start_hour > self._sleep_end_hour:
            return (
                hour >= self._sleep_start_hour or hour < self._sleep_end_hour
            )
        return self._sleep_start_hour <= hour < self._sleep_end_hour

    def evaluate(
        self,
        *,
        energy_ratio: float = 1.0,
        activity_level: float = 50.0,
        boredom: float = 0.0,
        social_stimulus: float = 0.0,
        consecutive_active_minutes: float = 0.0,
        current_hour: Optional[int] = None,
    ) -> NightPhase:
        """综合评估夜间状态，推进状态机（集成三维困倦引擎CDE）

        流程变更:
          旧: _update_sleepiness(线性) → _update_excitement → _resolve_transition
          新: tick_3d_engine(三维复合) → _update_excitement → _resolve_transition(增强)

        三维引擎输出的 composite_load 和 collapse_imminent 会影响阶段转换决策
        """
        hour = time.localtime().tm_hour if current_hour is None else int(current_hour) % 24
        clock_is_night = self.is_night_hours(hour)
        evening_social_window = self._night_social_start <= hour < self._sleep_start_hour
        is_night = clock_is_night or evening_social_window
        now = time.time()
        self._state.in_night_mode = is_night
        if is_night:
            self.update_night_social_energy(
                stimulus=max(0.0, min(1.0, social_stimulus)) * 8.0,
                group_activity=max(0.0, min(1.0, activity_level / 100.0)),
            )
            quiet_minutes = max(0.0, 22.0 - social_stimulus * 18.0 - max(0.0, activity_level - 25.0) / 3.5)
            self.update_reflection_depth(
                quiet_time=quiet_minutes,
                emotional_intensity=max(0.0, min(1.0, boredom / 100.0 + social_stimulus * 0.4)),
            )
        else:
            self.update_reflection_depth(quiet_time=0.0, emotional_intensity=0.0)
        _3d_result = self.tick_3d_engine(
            is_night=is_night,
            energy_ratio=energy_ratio,
            boredom=boredom,
            consecutive_active_minutes=consecutive_active_minutes,
            social_stimulus=social_stimulus,
            activity_level=activity_level,
            current_hour=current_hour,
        )
        if clock_is_night and not self.should_stay_awake_late():
            self._state.sleepiness = max(self._state.sleepiness, self._drowsy_threshold + 2.0)
            if _3d_result["collapse_imminent"] or self._state.overnight_pressure > self._state.pressure_critical_threshold:
                self._state.sleepiness = max(self._state.sleepiness, self._light_sleep_threshold + 1.0)
        self._update_sleepiness(
            is_night=is_night,
            energy_ratio=energy_ratio,
            boredom=boredom,
            consecutive_active_minutes=consecutive_active_minutes,
        )
        self._update_excitement(
            social_stimulus=social_stimulus,
            activity_level=activity_level,
            is_night=is_night,
        )
        old_phase = self._state.current_phase
        new_phase = self._resolve_transition()
        if new_phase != old_phase:
            logger.debug(
                f"[夜间节律] {
                    self._channel_id} "
                f"{
                    old_phase.label()} → {
                    new_phase.label()} "
                f"困倦={
                    self._state.sleepiness:.1f} 疲劳={
                        self._state.daily_fatigue:.1f} "
                f"3D:[D={
                    _3d_result['drowsiness_value']:.1f} P={
                    _3d_result['overnight_pressure']:.1f} "
                f"S={
                    _3d_result['sleep_reserve']:.1f} 复合={
                    _3d_result['composite_load']:.2f}"
                f"/阈值={
                    _3d_result['collapse_threshold']:.1f} 崩溃={
                    '是' if _3d_result['collapse_imminent'] else '否'}]"
            )
            self._state.current_phase = new_phase
            self._state.phase_entered_at = now
            if new_phase in (NightPhase.LIGHT_SLEEP, NightPhase.DEEP_SLEEP):
                if self._state.sleep_started_at <= 0:
                    self._state.sleep_started_at = now
                    self._state.sleep_stage = "light"
                self.add_reply_fatigue()
            else:
                if old_phase in (
                    NightPhase.LIGHT_SLEEP,
                    NightPhase.DEEP_SLEEP,
                ):
                    _sleep_dur = now - (self._state.sleep_started_at or now)
                    _quality = min(1.0, max(0.1, _sleep_dur / 21600.0))
                    self._state.last_sleep_quality = _quality
                    reserve_bonus = _quality * 35.0
                    self._state.sleep_reserve = min(
                        self._state.sleep_reserve_max,
                        self._state.sleep_reserve + reserve_bonus,
                    )
                self._state.sleep_started_at = 0.0
                self._state.sleep_stage = "awake"
                self.recover_fatigue(hours=0.5)
        if is_night and new_phase in (
            NightPhase.AWAKE,
            NightPhase.NIGHT_ACTIVE,
            NightPhase.SOCIAL_NIGHT,
            NightPhase.MIDNIGHT_REFLECT,
            NightPhase.QUIET_CONTEMPLATE,
        ):
            _debt_rate = 0.008
            if new_phase == NightPhase.NIGHT_ACTIVE:
                _debt_rate = 0.015
            elif new_phase == NightPhase.EXCITED:
                _debt_rate = 0.025
            elif new_phase == NightPhase.BURNED_OUT:
                _debt_rate = 0.003
            self._state.sleep_debt = min(
                100.0, self._state.sleep_debt + _debt_rate
            )
            self._state.debt_accumulated_at = now
        return self._state.current_phase

    def stimulus_wake(self, strength: float = 1.0) -> bool:
        """外部刺激尝试唤醒（增强版：集成三维引擎惩罚）

        唤醒流程:
        1. 判断当前阶段是否可唤醒
        2. 执行阶段转换
        3. 调用 apply_force_wake_penalty() 记录三维惩罚:
           - 睡眠储备扣减
           - 困意值反弹
           - 唤醒烦躁度上升
           - 半梦半醒程度上升
        """
        phase = self._state.current_phase
        if phase == NightPhase.DEEP_SLEEP:
            if strength >= 0.9:
                self.apply_force_wake_penalty(strength=strength)
                self._state.sleepiness = 0.0
                self._state.current_phase = NightPhase.AWAKE
                self._state.phase_entered_at = time.time()
                self._state.forced_awake = True
                self._state.last_stimulus_at = time.time()
                return True
            return False
        if phase == NightPhase.BURNED_OUT:
            return False
        if phase == NightPhase.LIGHT_SLEEP:
            if strength > 0.6:
                self.apply_force_wake_penalty(strength=strength * 0.7)
                _reduction = 30.0 * strength
                if strength >= 0.9:
                    _reduction = self._state.sleepiness + 1.0
                self._state.sleepiness = max(
                    0.0, self._state.sleepiness - _reduction
                )
                self._state.current_phase = NightPhase.AWAKE
                self._state.phase_entered_at = time.time()
                self._state.forced_awake = True
                self._state.last_stimulus_at = time.time()
                return True
            return False
        if phase == NightPhase.DROWSY:
            self.apply_force_wake_penalty(strength=strength * 0.3)
            _reduction = 20.0 * strength
            if strength >= 0.9:
                _reduction = self._state.sleepiness + 1.0
            self._state.sleepiness = max(
                0.0, self._state.sleepiness - _reduction
            )
            if self._state.sleepiness < self._drowsy_threshold:
                self._state.current_phase = NightPhase.AWAKE
                self._state.phase_entered_at = time.time()
                self._state.forced_awake = True
                return True
        self._state.last_stimulus_at = time.time()
        return False

    def force_sleep(self) -> None:
        """强制进入深睡"""
        self._state.current_phase = NightPhase.DEEP_SLEEP
        self._state.sleepiness = self._deep_sleep_threshold
        self._state.phase_entered_at = time.time()

    def natural_wake(self) -> None:
        """自然醒来（增强版：重置三维引擎状态）

        自然醒来时:
        1. 计算睡眠质量并清偿部分睡眠债务
        2. 按质量补充睡眠储备（高质量睡眠=高储备恢复）
        3. 重置困意值（自然醒=困意消散）
        4. 清除熬夜压力（新的一天开始）
        5. 清除强制唤醒痕迹
        6. 重置崩溃标志
        """
        _sleep_duration = time.time() - self._state.phase_entered_at
        if self._state.current_phase in (
            NightPhase.DEEP_SLEEP,
            NightPhase.LIGHT_SLEEP,
        ):
            _quality = min(1.0, max(0.1, _sleep_duration / 21600.0))
            self._state.last_sleep_quality = _quality
            _debt_clear = _quality * 0.7 * self._state.sleep_debt
            self._state.sleep_debt = max(
                0.0, self._state.sleep_debt - _debt_clear
            )
            reserve_restore = _quality * 50.0
            if _sleep_duration > 21600:
                bonus = min(20.0, (_sleep_duration - 21600) / 1800.0)
                reserve_restore += bonus
            self._state.sleep_reserve = min(
                self._state.sleep_reserve_max,
                self._state.sleep_reserve + reserve_restore,
            )
            self._state.reserve_quality_factor = min(1.5, 1.0 + _quality * 0.4)
        else:
            self._state.last_sleep_quality = 0.0
        self._state.current_phase = NightPhase.AWAKE
        self._state.sleepiness = 0.0
        self._state.drowsiness_value = max(
            0.0, self._state.drowsiness_value * 0.15
        )
        self._state.excitement_counter = 0.0
        self._state.burn_accumulated = 0.0
        self._state.forced_awake = False
        self._state.phase_entered_at = time.time()
        self._state.night_reply_count = 0
        self._state.peek_window_open = False
        self._reset_expression_deformation()
        self._state.pressure_chat_minutes = 0.0
        self._state.pressure_peek_minutes = 0.0
        self._state.pressure_think_intensity = 0.0
        self._state.pressure_interrupt_count = 0
        self._state.total_overnight_pressure = 0.0
        # ── 三维引擎重置 ──
        self._state.overnight_pressure = max(
            0.0, self._state.overnight_pressure * 0.08
        )
        self._state.pressure_activity_multiplier = 1.0
        self._state.pressure_history.clear()
        self._state.composite_load = 0.0
        self._state.collapse_imminent = False
        self._state.collapse_time_estimate = 0.0
        self._state.force_wake_count_today = 0
        self._state.last_force_wake_time = 0.0
        self._state.force_wake_irritation_decay = 0.0
        self._state.wake_irritation_level = 0.0
        self._state.half_asleep_level = 0.0
        self._state.in_lazy_state = False
        self._state.lazy_state_started = 0.0
        self._state.lazy_resistance_cycles = 0

    def get_recovery_rate_modifier(self) -> float:
        """返回睡眠债务导致的恢复速率修正因子（1.0=正常，<1=受损）"""
        return max(0.3, 1.0 - self._state.sleep_debt / 120.0)

    # ────────────────── 夜间回复上限 ──────────────────

    def can_reply_tonight(self) -> bool:
        """检查夜间回复是否仍在上限以内"""
        if not self.is_night_hours():
            return True
        phase = self._state.current_phase
        if phase == NightPhase.DEEP_SLEEP:
            return False
        if phase == NightPhase.BURNED_OUT:
            return False
        return self._state.night_reply_count < self._state.night_reply_cap

    def record_night_reply(self) -> None:
        """记录一次夜间回复"""
        self._state.night_reply_count += 1
        logger.debug(
            f"[{self._channel_id}] 夜间回复计数 "
            f"{self._state.night_reply_count}/{self._state.night_reply_cap}"
        )

    def remaining_night_replies(self) -> int:
        """剩余可回复次数（白天返回 -1 表示无限制）"""
        if not self.is_night_hours():
            return -1
        return max(
            0, self._state.night_reply_cap - self._state.night_reply_count
        )

    def reset_night_counters(self) -> None:
        """重置夜间计数器（切换到白天时调用）"""
        self._state.night_reply_count = 0
        self._state.night_cap_reset_at = time.time()

    # ────────────────── 睡眠窥屏窗口 ──────────────────

    def evaluate_sleep_peek(self) -> bool:
        """评估浅睡阶段是否开放窥屏窗口
        浅睡状态下，每隔一段间隔打开一个短暂窥屏窗口，
        窗口期间允许"瞟一眼群"但不允许回复。
        返回 True 表示当前窥屏窗口已打开。
        """
        now = time.time()
        s = self._state
        # 只在浅睡阶段生效
        if s.current_phase != NightPhase.LIGHT_SLEEP:
            s.peek_window_open = False
            return False
        # 窗口已打开 → 检查是否过期
        if s.peek_window_open:
            elapsed = now - s.peek_window_opened_at
            if elapsed >= s.peek_window_duration:
                s.peek_window_open = False
                s.last_peek_window_at = now
                logger.debug(
                    f"[{self._channel_id}] 睡眠窥屏窗口关闭(持续{elapsed:.0f}s)"
                )
                return False
            return True
        # 窗口未打开 → 检查间隔是否足够
        since_last = now - s.last_peek_window_at
        if since_last >= s.peek_window_interval:
            s.peek_window_open = True
            s.peek_window_opened_at = now
            logger.debug(
                f"[{self._channel_id}] 睡眠窥屏窗口打开(间隔{since_last:.0f}s)"
            )
            return True
        return False

    def is_peek_window_active(self) -> bool:
        """查询当前窥屏窗口是否处于开放状态"""
        return (
            self._state.peek_window_open
            and self._state.current_phase == NightPhase.LIGHT_SLEEP
        )

    # F28：睡眠期间回复限次——1小时内最多回复1次，防止轰炸穿透

    def evaluate_sleep_reply_budget(self) -> bool:
        now = time.time()
        s = self._state
        _is_sleep = s.current_phase in (
            NightPhase.LIGHT_SLEEP,
            NightPhase.DEEP_SLEEP,
        )
        if not _is_sleep:
            s.sleep_reply_used = 0
            s.sleep_reply_window_start = 0.0
            return True
        if (
            s.sleep_reply_window_start == 0.0
            or (now - s.sleep_reply_window_start) >= s.sleep_reply_window_sec
        ):
            s.sleep_reply_window_start = now
            s.sleep_reply_used = 0
        if s.sleep_reply_used >= 1:
            logger.debug(
                f"[{self._channel_id}] 睡眠回复限次已达(本窗口已用{s.sleep_reply_used}次)"
            )
            return False
        return True

    def consume_sleep_reply(self) -> None:
        s = self._state
        if s.current_phase in (NightPhase.LIGHT_SLEEP, NightPhase.DEEP_SLEEP):
            s.sleep_reply_used += 1

    # ────────────────── 内部方法 ──────────────────

    def _update_sleepiness(
        self,
        *,
        is_night: bool,
        energy_ratio: float,
        boredom: float,
        consecutive_active_minutes: float,
    ) -> None:
        s = self._state
        now = time.localtime()
        hour = now.tm_hour
        _is_dawn = self._dawn_recover_start <= hour < self._sleep_end_hour
        if _is_dawn:
            s.sleepiness = max(0.0, s.sleepiness - 0.8)
            return
        if is_night:
            rise_rate = 0.5
            if energy_ratio < 0.3:
                rise_rate += 0.4
            if boredom > 50:
                rise_rate += 0.2
            s.sleepiness = min(100.0, s.sleepiness + rise_rate)
        else:
            decay_rate = 0.3
            if energy_ratio > 0.7:
                decay_rate += 0.2
            s.sleepiness = max(0.0, s.sleepiness - decay_rate)
        if consecutive_active_minutes > 90:
            fatigue_push = (consecutive_active_minutes - 90) * 0.02
            s.sleepiness = min(100.0, s.sleepiness + fatigue_push)

    def _update_excitement(
        self,
        *,
        social_stimulus: float,
        activity_level: float,
        is_night: bool,
    ) -> None:
        s = self._state
        # 社交刺激和活跃提升亢奋
        if social_stimulus > 0.5 and is_night:
            s.excitement_counter = min(
                self._burnout_threshold,
                s.excitement_counter + social_stimulus * 2.0,
            )
            s.burn_accumulated += social_stimulus * 0.5
        if activity_level > 70 and is_night:
            s.excitement_counter = min(
                self._burnout_threshold,
                s.excitement_counter + 0.3,
            )
        # 亢奋自然衰减
        if s.excitement_counter > 0:
            s.excitement_counter = max(0.0, s.excitement_counter - 0.15)

    def _resolve_transition(self) -> NightPhase:
        s = self._state
        current = s.current_phase
        now = time.localtime()
        hour = now.tm_hour
        # 第三阶段新增：夜间精细阶段转换
        if self.is_night_hours(hour) or hour >= self._night_social_start:
            # 午夜反思阶段 (0-4点)
            if self._midnight_reflect_start <= hour < self._dawn_recover_start:
                if current == NightPhase.AWAKE and s.reflection_depth > 0.3:
                    return NightPhase.MIDNIGHT_REFLECT
                elif (
                    current == NightPhase.MIDNIGHT_REFLECT
                    and s.reflection_depth > 0.7
                ):
                    return NightPhase.QUIET_CONTEMPLATE
                elif (
                    current == NightPhase.QUIET_CONTEMPLATE
                    and s.reflection_depth < 0.2
                ):
                    return NightPhase.MIDNIGHT_REFLECT
            # 黎明恢复阶段 (4-7点)
            elif self._dawn_recover_start <= hour < self._sleep_end_hour:
                if current in (
                    NightPhase.LIGHT_SLEEP,
                    NightPhase.QUIET_CONTEMPLATE,
                ):
                    return NightPhase.DAWN_RECOVER
                elif (
                    current == NightPhase.DAWN_RECOVER
                    and s.dawn_recovery_progress > 0.8
                ):
                    return NightPhase.AWAKE
            # 夜间活跃阶段 (20-24点)
            elif hour >= self._night_social_start:
                if (
                    current == NightPhase.AWAKE
                    and s.night_social_energy > s.night_social_threshold
                ):
                    return NightPhase.NIGHT_ACTIVE
                elif (
                    current == NightPhase.NIGHT_ACTIVE
                    and s.night_social_energy > 70
                ):
                    return NightPhase.SOCIAL_NIGHT
                elif (
                    current == NightPhase.SOCIAL_NIGHT
                    and s.night_social_energy < 30
                ):
                    return NightPhase.NIGHT_ACTIVE
        # 原有逻辑保持不变
        # 亢奋检测优先
        if (
            s.excitement_counter >= self._excitement_ceiling
            and current != NightPhase.BURNED_OUT
        ):
            if current in (
                NightPhase.AWAKE,
                NightPhase.DROWSY,
                NightPhase.NIGHT_ACTIVE,
            ):
                return NightPhase.EXCITED
        # 熬穿检测
        if current == NightPhase.EXCITED:
            if s.burn_accumulated >= self._burnout_threshold:
                return NightPhase.BURNED_OUT
            if s.excitement_counter < self._excitement_ceiling * 0.3:
                return NightPhase.DROWSY
            return NightPhase.EXCITED
        # 熬穿后强制深睡
        if current == NightPhase.BURNED_OUT:
            if s.phase_duration_sec() > 600:
                return NightPhase.DEEP_SLEEP
            return NightPhase.BURNED_OUT
        # 睡眠递进
        if current == NightPhase.DEEP_SLEEP:
            if s.sleepiness < 20 and s.phase_duration_sec() > 3600:
                return NightPhase.AWAKE
            return NightPhase.DEEP_SLEEP
        if current == NightPhase.LIGHT_SLEEP:
            if s.sleepiness >= self._deep_sleep_threshold:
                return NightPhase.DEEP_SLEEP
            if (
                s.sleepiness < self._drowsy_threshold
                and not self.is_night_hours()
            ):
                return NightPhase.AWAKE
            return NightPhase.LIGHT_SLEEP
        if current == NightPhase.DROWSY:
            if s.sleepiness >= self._light_sleep_threshold:
                return NightPhase.LIGHT_SLEEP
            if s.sleepiness < self._drowsy_threshold * 0.6:
                return NightPhase.AWAKE
            return NightPhase.DROWSY
        # 清醒状态
        if s.sleepiness >= self._drowsy_threshold:
            return NightPhase.DROWSY
        return NightPhase.AWAKE

    def sleep_debt(self) -> float:
        """睡眠债务 0.0~1.0
        基于累积熬夜压力(burn_accumulated)和困倦值推算。
        债务越高，恢复需要越长的深度休息。
        """
        s = self._state
        burn_ratio = min(
            1.0, s.burn_accumulated / max(1.0, self._burnout_threshold)
        )
        sleepy_ratio = min(1.0, s.sleepiness / 100.0)
        raw = burn_ratio * 0.6 + sleepy_ratio * 0.4
        return round(max(0.0, min(1.0, raw)), 3)

    def compute_wake_grace_duration(self) -> float:
        """动态计算唤醒后的宽限期（秒）

        基于多维状态自适应计算：
        - 睡眠债务越高 → 需要更长的清醒缓冲期
        - 凌晨/深夜时段 → 身体恢复中，需要更长缓冲
        - 熬夜压力越大 → 越不容易清醒，需要更长保护期
        - 上次睡眠质量差 → 需要更多时间稳定

        返回值范围：45 ~ 420秒（约0.75~7分钟）
        """
        s = self._state
        now = time.localtime()
        hour = now.tm_hour
        base = 60.0
        _debt = min(100.0, s.sleep_debt * 100)
        base += _debt * 2.0
        if 4 <= hour < 7:
            base += 90.0
            base += (6 - hour) * 15.0
        elif 23 <= hour or hour < 3:
            base += 40.0
        elif 3 <= hour < 4:
            base += 70.0
        _burn_ratio = min(
            1.0, s.burn_accumulated / max(1.0, self._burnout_threshold)
        )
        base += _burn_ratio * 80.0
        if s.last_sleep_quality > 0:
            base -= (1.0 - s.last_sleep_quality) * 50.0
        else:
            base -= 20.0
        _sleepiness_after = max(0.0, s.sleepiness)
        if _sleepiness_after > 60:
            base += (_sleepiness_after - 60) * 1.5
        grace = max(45.0, min(420.0, base))
        return round(grace, 1)

    # ═══════════════════════════════════════
    #  从MaiohaBot移植：完整疲劳/睡眠/熬夜系统
    # ═══════════════════════════════════════

    def get_night_fatigue_multiplier(self) -> float:
        """夜间疲劳加速倍率（19:00起阶梯式增长）
        19:00=1.01  20:00=1.02  21:00=1.05  22:00=2.0  23:00=2.5  24:00+=3.0"""
        hour = time.localtime().tm_hour
        if hour < self._state.night_fatigue_start_hour:
            return 1.0
        offset = hour - self._state.night_fatigue_start_hour
        _table = {0: 1.01, 1: 1.02, 2: 1.05, 3: 2.0, 4: 2.5}
        return _table.get(min(offset, 5), 3.0)

    def add_reply_fatigue(self) -> float:
        """每次回复增加疲劳值（带夜间加速 + 三维储备消耗）"""
        s = self._state
        if not s.daily_fatigue_limit or s.daily_fatigue_limit <= 0:
            return 0.0
        base = s.fatigue_per_reply
        mult = self.get_night_fatigue_multiplier()
        amount = base * mult
        old_val = s.daily_fatigue
        s.daily_fatigue = min(s.daily_fatigue_limit, s.daily_fatigue + amount)
        self.consume_reserve_for_reply()
        return s.daily_fatigue - old_val

    def recover_fatigue(self, hours: float) -> float:
        """基于时间恢复疲劳值（白天正常/晚上75%/深夜50%/睡眠归零）"""
        s = self._state
        if not s.daily_fatigue_limit or s.daily_fatigue_limit <= 0:
            return 0.0
        now_hour = time.localtime().tm_hour
        phase = s.current_phase
        if phase in (NightPhase.LIGHT_SLEEP, NightPhase.DEEP_SLEEP):
            recovered = s.daily_fatigue
            s.daily_fatigue = 0.0
            return recovered
        time_factor = 1.0
        if 6 <= now_hour < 19:
            time_factor = 1.0
        elif 19 <= now_hour < 23:
            time_factor = 0.75
        else:
            time_factor = 0.5
        recovery = hours * s.fatigue_recovery_rate * time_factor
        old_val = s.daily_fatigue
        s.daily_fatigue = max(0.0, s.daily_fatigue - recovery)
        today_str = time.strftime("%Y-%m-%d")
        if s.last_fatigue_reset_date != today_str:
            s.daily_fatigue = 0.0
            s.last_fatigue_reset_date = today_str
        return old_val - s.daily_fatigue

    def get_sleep_stage(self) -> str:
        """三阶段睡眠推进（light→deep→rem→light），基于入睡后经过的分钟数"""
        s = self._state
        if s.sleep_stage == "awake" or s.current_phase == NightPhase.AWAKE:
            return "awake"
        if s.sleep_started_at <= 0:
            return "awake"
        elapsed_min = (time.time() - s.sleep_started_at) / 60.0
        light_dur = s.light_sleep_duration_min
        deep_dur = s.deep_sleep_duration_min
        rem_dur = s.rem_sleep_duration_min
        total_cycle = light_dur + deep_dur + rem_dur
        pos_in_cycle = elapsed_min % total_cycle
        if pos_in_cycle < light_dur:
            stage = "light"
        elif pos_in_cycle < light_dur + deep_dur:
            stage = "deep"
        elif pos_in_cycle < total_cycle:
            stage = "rem"
        else:
            stage = "light"
        if stage != s.sleep_stage:
            s.sleep_stage = stage
        return stage

    def get_wakeup_probability_by_stage(self, is_mentioned: bool) -> float:
        """按睡眠阶段返回唤醒概率（深睡极低/浅睡较高/REM中等）"""
        if not is_mentioned:
            return 0.0
        stage = self.get_sleep_stage()
        probs = {"light": 0.20, "deep": 0.01, "rem": 0.15}
        return probs.get(stage, 0.05)

    def apply_interruption_penalty(self) -> float:
        """强制唤醒打断惩罚：扣能量+打断计数"""
        s = self._state
        s.interruption_count += 1
        penalty = s.interruption_energy_penalty * s.interruption_count
        s.sleepiness = max(0.0, s.sleepiness - penalty * 10)
        return penalty

    def check_stay_in_bed(self, energy_value: float) -> bool:
        """赖床检查（增强版：融合三维引擎赖床概率）

        赖床决策融合多个维度:
        1. 基础能量检查：能量满了就不赖
        2. 三维复合赖床概率：困意高+储备低 → 赖床概率大
        3. 强制唤醒痕迹：被强制唤醒次数多 → 更想赖床报复性补睡
        4. 半梦半醒程度：意识模糊 → 不想起
        """
        s = self._state
        if energy_value >= s.full_energy_threshold:
            s.interruption_count = 0
            return False
        ratio = energy_value / 10.0
        base_prob = s.stay_in_bed_probability * (1.0 - ratio)
        _3d_laziness = s.laziness_probability
        combined_prob = base_prob * (1.0 - _3d_laziness) + _3d_laziness * 0.85
        if s.force_wake_count_today >= 2:
            revenge_factor = min(0.3, s.force_wake_count_today * 0.08)
            combined_prob += revenge_factor
        if s.half_asleep_level > 0.35:
            combined_prob += s.half_asleep_level * 0.25
        combined_prob = max(0.0, min(1.0, combined_prob))
        result = random.random() < combined_prob
        if result:
            if not s.in_lazy_state:
                s.in_lazy_state = True
                s.lazy_state_started = time.time()
            s.lazy_resistance_cycles += 1
        else:
            s.in_lazy_state = False
            s.lazy_resistance_cycles = 0
        return result

    def should_stay_awake_late(self) -> bool:
        """三维复合判断：是否还能撑住不睡

        旧逻辑: daily_fatigue < 70% → 可以熬夜
        新逻辑: 三维复合判断
          - composite_load < collapse_threshold * 0.65 → 还能撑（睡眠储备充足）
          - composite_load >= collapse_threshold * 0.9 → 熬不住了必须睡
          - 中间地带 → 结合睡眠储备和熬夜压力综合判断

        相乘相消: 即使困意高(D大)，如果睡眠储备也高(S大)，
                分母增大→composite降低→还能多撑一会
        """
        s = self._state
        if (
            s.daily_fatigue_limit > 0
            and s.daily_fatigue >= s.daily_fatigue_limit * 0.7
        ):
            return False
        if s.composite_load <= 0 or s.collapse_threshold <= 0:
            return True
        ratio = s.composite_load / s.collapse_threshold
        if ratio >= 0.92:
            return False
        if ratio < 0.55:
            return True
        reserve_ratio = s.sleep_reserve / max(1.0, s.sleep_reserve_max)
        if reserve_ratio > 0.6 and s.overnight_pressure < 55:
            return True
        if reserve_ratio < 0.2 or s.overnight_pressure > 80:
            return False
        return ratio < 0.75

    def night_behavior_summary(self) -> Dict[str, Any]:
        """返回完整夜间行为约束字典，供主链和面板消费（增强版：含三维引擎数据）"""
        s = self._state
        phase = s.current_phase
        debt = self.sleep_debt()
        stamina_pressure = min(
            1.0, s.burn_accumulated / max(1.0, self._burnout_threshold)
        )
        excitement = min(
            1.0, s.excitement_counter / max(1.0, self._excitement_ceiling)
        )
        _3d_status = self.get_3d_engine_status()
        pressure_breakdown = self.get_overnight_pressure_breakdown()
        time_band = self.resolve_time_band()
        return {
            "phase": phase.value,
            "phase_label": phase.label(),
            "time_band": time_band["key"],
            "time_band_label": time_band["label"],
            "time_band_description": time_band["description"],
            "current_hour": time_band["hour"],
            "is_night": bool(s.in_night_mode or self.is_night_hours(time_band["hour"])),
            "can_interact": phase.can_interact,
            "can_reply": self.can_reply_tonight(),
            "remaining_replies": self.remaining_night_replies(),
            "is_peek_window": self.is_peek_window_active(),
            "sleep_debt": debt,
            "stamina_pressure": round(stamina_pressure, 3),
            "excitement_level": round(excitement, 3),
            "burned_out": phase == NightPhase.BURNED_OUT,
            "is_sleeping": phase.is_sleeping,
            "is_night_social": phase.is_night_social,
            "is_reflective": phase.is_reflective,
            "night_social_energy": round(s.night_social_energy, 1),
            "reflection_depth": round(s.reflection_depth, 2),
            # ── 三维引擎输出 ──
            "drowsiness_value": _3d_status["dimensions"]["D_drowsiness"][
                "value"
            ],
            "overnight_pressure": _3d_status["dimensions"]["P_pressure"][
                "value"
            ],
            "sleep_reserve": _3d_status["dimensions"]["S_reserve"]["value"],
            "composite_load": _3d_status["composite"]["load"],
            "collapse_threshold": _3d_status["composite"]["threshold"],
            "collapse_imminent": _3d_status["composite"]["imminent"],
            "laziness_probability": _3d_status["behavior"]["laziness_prob"],
            "wake_irritation": _3d_status["behavior"]["wake_irritation"],
            "half_asleep_level": _3d_status["behavior"]["half_asleep"],
            "yawn_frequency": _3d_status["behavior"]["yawn_freq"],
            "response_suppression": _3d_status["behavior"][
                "response_suppression"
            ],
            "body_state_tag": _3d_status["behavior"]["body_tag"],
            "monologue_mood_hint": _3d_status["behavior"]["mood_hint"],
            "pressure_breakdown": pressure_breakdown,
        }

    # ═════════════════════════════════════════
    #  GAP-A：熬夜压力分项追踪器
    #  按活动类型分别累积压力，供 ACFN 精确读取
    # ═════════════════════════════════════════

    def record_overnight_activity(
        self, activity_type: str, intensity: float = 1.0
    ) -> None:
        """记录一次夜间活动，按类型分项累积压力

        activity_type: "chat" / "peek" / "think" / "interrupt"
        intensity: 0.0~1.0，该次活动的强度
        """
        hour = time.localtime().tm_hour
        if not (self.is_night_hours(hour) or hour >= self._state.pressure_start_hour):
            return
        s = self._state
        now = time.time()
        if activity_type == "chat":
            s.pressure_chat_minutes += 1.0 * intensity
        elif activity_type == "peek":
            s.pressure_peek_minutes += 1.0 * intensity
        elif activity_type == "think":
            s.pressure_think_intensity += 0.5 * intensity
        elif activity_type == "interrupt":
            s.pressure_interrupt_count += 1
            s.pressure_last_interrupt_at = now
            recent_count = sum(
                1 for _ in range(min(3, s.pressure_interrupt_count))
            )
            s.total_overnight_pressure += recent_count * 0.15 * intensity
        self._recalc_total_pressure()

    def _recalc_total_pressure(self) -> None:
        """重新计算综合熬夜压力（加权求和 + 时间衰减）"""
        s = self._state
        chat_p = min(1.0, s.pressure_chat_minutes / 120.0) * 0.30
        peek_p = min(1.0, s.pressure_peek_minutes / 90.0) * 0.20
        think_p = min(1.0, s.pressure_think_intensity / 50.0) * 0.25
        interrupt_p = min(1.0, s.pressure_interrupt_count / 15.0) * 0.25
        raw_total = chat_p + peek_p + think_p + interrupt_p
        decay_factor = math.exp(
            -max(0, time.time() - s.debt_accumulated_at) / 7200.0
        )
        s.total_overnight_pressure = raw_total * (1.0 - decay_factor * 0.3)

    def get_overnight_pressure_breakdown(self) -> Dict[str, Any]:
        """返回熬夜压力完整分解（供 ACFN 和调试面板使用）"""
        s = self._state
        return {
            "total": round(s.total_overnight_pressure, 4),
            "chat_minutes": round(s.pressure_chat_minutes, 1),
            "peek_minutes": round(s.pressure_peek_minutes, 1),
            "think_intensity": round(s.pressure_think_intensity, 2),
            "interrupt_count": s.pressure_interrupt_count,
            "chat_component": round(
                min(1.0, s.pressure_chat_minutes / 120.0) * 0.30, 4
            ),
            "peek_component": round(
                min(1.0, s.pressure_peek_minutes / 90.0) * 0.20, 4
            ),
            "think_component": round(
                min(1.0, s.pressure_think_intensity / 50.0) * 0.25, 4
            ),
            "interrupt_component": round(
                min(1.0, s.pressure_interrupt_count / 15.0) * 0.25, 4
            ),
        }

    # ═════════════════════════════════════════
    #  GAP-A：夜间亢奋控制器
    #  困倦→硬撑→短时亢奋→熬穿 的完整状态链
    # ═════════════════════════════════════════

    def evaluate_arousal_chain(
        self, stimulus_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """评估夜间亢奋链当前所处阶段

        返回值: "normal" / "drowsy" / "stubborn" / "arousal" / "burnthrough"

        stimulus_info 可包含:
          - topic_stimulus: float (话题刺激度 0~1)
          - mentioned: bool (是否被@或引用)
          - continuous_interaction: int (连续互动轮数)
          - high_interest_topic: bool (是否有高兴趣话题)
        """
        s = self._state
        phase = s.current_phase
        if not self.is_night_hours():
            s.is_burnthrough_active = False
            self._reset_expression_deformation()
            return "normal"
        if phase in (NightPhase.LIGHT_SLEEP, NightPhase.DEEP_SLEEP):
            return "asleep"
        info = stimulus_info or {}
        topic_stim = float(info.get("topic_stimulus", 0.0))
        is_mentioned = bool(info.get("mentioned", False))
        cont_interact = int(info.get("continuous_interaction", 0))
        is_high_interest = bool(info.get("high_interest_topic", False))
        sleepiness_ratio = max(0.0, min(1.0, s.sleepiness / 100.0))
        pressure_ratio = max(0.0, min(1.0, s.total_overnight_pressure))
        combined_load = sleepiness_ratio * 0.55 + pressure_ratio * 0.45
        if combined_load < 0.22:
            return "normal"
        if combined_load < 0.40:
            if s.stubborn_duration > 0:
                s.stubborn_duration += 1.0
                return "stubborn"
            return "drowsy"
        if combined_load >= 0.40 and combined_load < 0.65:
            if s.stubborn_started_at == 0:
                s.stubborn_started_at = time.time()
            s.stubborn_duration = time.time() - s.stubborn_started_at
            arousal_potential = 0.0
            if is_mentioned:
                arousal_potential += 0.28
            if topic_stim > 0.45:
                arousal_potential += topic_stim * 0.22
            if cont_interact >= 3:
                arousal_potential += min(0.18, cont_interact * 0.04)
            if is_high_interest:
                arousal_potential += 0.15
            if arousal_potential > 0.35 and s.excitement_counter > 25:
                s.arousal_trigger_type = "stimulated"
                s.arousal_stimulus_strength = arousal_potential
                s.arousal_consecutive_pulls = cont_interact
                self._enter_arousal_state()
                return "arousal"
            return "stubborn"
        if combined_load >= 0.65 or phase in (
            NightPhase.EXCITED,
            NightPhase.BURNED_OUT,
        ):
            if phase == NightPhase.BURNED_OUT or s.is_burnthrough_active:
                self._update_burnthrough_deformation()
                return "burnthrough"
            if phase == NightPhase.EXCITED:
                if s.burn_accumulated >= self._burnout_threshold * 0.85:
                    self._enter_burnthrough_state()
                    return "burnthrough"
                return "arousal"
            if combined_load >= 0.80:
                self._enter_burnthrough_state()
                return "burnthrough"
            s.arousal_trigger_type = "overload"
            self._enter_arousal_state()
            return "arousal"
        return "drowsy"

    def _enter_arousal_state(self) -> None:
        """进入亢奋态——更新表达变形参数"""
        s = self._state
        s.expression_punctuation_density = 1.15
        s.expression_sentence_fragmentation = 0.12
        s.expression_emotion_leakage = 0.10
        s.expression_impulse_level = 0.35

    def _enter_burnthrough_state(self) -> None:
        """进入熬穿态——表达变形加剧"""
        s = self._state
        s.is_burnthrough_active = True
        s.burnthrough_started_at = time.time()
        s.expression_punctuation_density = 1.45
        s.expression_sentence_fragmentation = 0.38
        s.expression_emotion_leakage = 0.42
        s.expression_impulse_level = 0.72

    def _update_burnthrough_deformation(self) -> None:
        """熬穿态随时间推移变形加剧，接近上限时突然崩溃"""
        s = self._state
        if not s.is_burnthrough_active:
            return
        elapsed = time.time() - s.burnthrough_started_at
        progress = min(1.0, elapsed / s.burnthrough_max_duration)
        if progress > 0.85:
            s.expression_impulse_level = min(
                1.0, s.expression_impulse_level + 0.08
            )
            s.expression_sentence_fragmentation = min(
                0.75, s.expression_sentence_fragmentation + 0.05
            )
            s.expression_emotion_leakage = min(
                0.88, s.expression_emotion_leakage + 0.06
            )
            if progress >= 0.98:
                s.current_phase = NightPhase.DEEP_SLEEP
                s.is_burnthrough_active = False
                logger.debug(f"[{self._channel_id}] 熬穿崩溃→强制深睡")
        else:
            ramp = progress * progress * (3.0 - 2.0 * progress)
            s.expression_punctuation_density = 1.15 + ramp * 0.40
            s.expression_sentence_fragmentation = 0.12 + ramp * 0.32
            s.expression_emotion_leakage = 0.10 + ramp * 0.36
            s.expression_impulse_level = 0.35 + ramp * 0.42

    def _reset_expression_deformation(self) -> None:
        """退出夜间/醒来时重置所有表达变形"""
        s = self._state
        s.expression_punctuation_density = 1.0
        s.expression_sentence_fragmentation = 0.0
        s.expression_emotion_leakage = 0.0
        s.expression_impulse_level = 0.0
        s.is_burnthrough_active = False
        s.stubborn_duration = 0.0
        s.stubborn_started_at = 0.0
        s.arousal_trigger_type = ""
        s.arousal_consecutive_pulls = 0

    def get_night_expression_profile(self) -> Dict[str, Any]:
        """获取夜间表达风格完整画像（供 planner 使用）

        返回三种模板的量化参数：
          - drowsy_template: 困倦型参数
          - woken_template: 被吵醒型参数
          - burnthrough_template: 熬穿型参数
          - current: 当前活跃模板及具体数值
        """
        s = self._state
        phase = s.current_phase
        arousal_state = self.evaluate_arousal_chain()
        base = {
            "sentence_length_target": 1.0,
            "punctuation_looseness": 0.0,
            "fragment_probability": 0.0,
            "emotion_surface": 0.0,
            "impulse_reply_tendency": 0.0,
            "complaint_tendency": 0.0,
            "ramble_tendency": 0.0,
        }
        if arousal_state == "drowsy":
            profile = dict(base)
            profile["sentence_length_target"] = 0.55
            profile["punctuation_looseness"] = 0.25
            profile["fragment_probability"] = 0.15
            profile["emotion_surface"] = 0.05
            active_label = "drowsy"
        elif arousal_state == "stubborn":
            profile = dict(base)
            profile["sentence_length_target"] = 0.70
            profile["punctuation_looseness"] = 0.18
            profile["fragment_probability"] = 0.10
            profile["impulse_reply_tendency"] = 0.20
            active_label = "stubborn"
        elif arousal_state == "arousal":
            profile = dict(base)
            pd = s.expression_punctuation_density
            profile["punctuation_looseness"] = (pd - 1.0) * 2.0
            profile["sentence_length_target"] = 1.15
            profile["impulse_reply_tendency"] = s.expression_impulse_level
            profile["emotion_surface"] = s.expression_emotion_leakage
            active_label = "arousal"
        elif arousal_state == "burnthrough":
            profile = dict(base)
            profile["sentence_length_target"] = 1.35
            profile["punctuation_looseness"] = (
                s.expression_punctuation_density - 1.0
            ) * 2.5
            profile["fragment_probability"] = (
                s.expression_sentence_fragmentation
            )
            profile["emotion_surface"] = s.expression_emotion_leakage
            profile["impulse_reply_tendency"] = s.expression_impulse_level
            profile["ramble_tendency"] = s.expression_impulse_level * 0.6
            active_label = "burnthrough"
        else:
            profile = dict(base)
            active_label = "normal"
        return {
            "current_state": arousal_state,
            "active_template": active_label,
            "phase": phase.value,
            "params": {k: round(v, 4) for k, v in profile.items()},
            "raw_expression": {
                "punct_density": round(s.expression_punctuation_density, 3),
                "fragmentation": round(s.expression_sentence_fragmentation, 3),
                "emotion_leak": round(s.expression_emotion_leakage, 3),
                "impulse": round(s.expression_impulse_level, 3),
            },
        }

    # ════════════════════════════════════════════════
    #  ★★★ ACFN级三维困倦引擎 (3D-ACFN) ★★★
    #  七层管线: 维度更新→导数层→耦合矩阵→迭代精炼→滞后反馈→阈值漂移→随机微扰
    # ════════════════════════════════════════════════

    def tick_3d_engine(
        self,
        *,
        is_night: bool,
        energy_ratio: float = 1.0,
        boredom: float = 0.0,
        consecutive_active_minutes: float = 0.0,
        social_stimulus: float = 0.0,
        activity_level: float = 50.0,
        current_hour: Optional[int] = None,
    ) -> Dict[str, Any]:
        """ACFN级三维困倦引擎主入口 —— 七层完整评估管线

        管线流程:
          Layer 0 (维度更新): _tick_drowsiness / _tick_pressure / _tick_reserve
          Layer 1 (导数感知):   _compute_3d_derivatives → [dD/dt, dP/dt, dS/dt]
          Layer 2 (耦合矩阵):   _get_3d_coupling_matrix → 3×3动态调制
          Layer 3 (归一化):      原始值 → [0,1] 特征向量
          Layer 4 (迭代精炼):   _iterative_composite_refinement → 3~5轮收敛
          Layer 5 (滞后修正):   _apply_hysteresis_correction → 反馈补偿
          Layer 6 (阈值漂移):   _drift_adaptive_threshold → 自适应崩溃线
          Layer 7 (随机微扰):   _apply_confidence_perturbation → 受控噪声
          Output (置信度):     _compute_acfn_confidence → 最终置信度

        返回完整引擎状态字典（含ACFN扩展字段）
        """
        now = time.time()
        hour = (
            current_hour
            if current_hour is not None
            else time.localtime().tm_hour
        )
        s = self._state
        s.coupling_dirty_flag = True
        self._tick_drowsiness_dimension(
            is_night=is_night,
            energy_ratio=energy_ratio,
            boredom=boredom,
            consecutive_active_minutes=consecutive_active_minutes,
            hour=hour,
            now=now,
        )
        self._tick_pressure_dimension(
            is_night=is_night,
            social_stimulus=social_stimulus,
            activity_level=activity_level,
            hour=hour,
            now=now,
        )
        self._tick_reserve_dimension(
            hour=hour,
            now=now,
            energy_ratio=energy_ratio,
        )
        raw_d = max(0.0, s.drowsiness_value)
        raw_p = max(0.0, s.overnight_pressure)
        raw_s = max(1e-6, s.sleep_reserve)
        quality = max(0.1, s.reserve_quality_factor)
        derivatives = self._compute_3d_derivatives(
            raw_d=raw_d, raw_p=raw_p, raw_s=raw_s, now=now
        )
        coupling = self._get_3d_coupling_matrix(
            features=[raw_d / 100.0, raw_p / 100.0, raw_s / 100.0],
            derivatives=derivatives,
        )
        (iter_composite, iter_passes, delta_bar, delta_score) = (
            self._iterative_composite_refinement(
                raw_d=raw_d,
                raw_p=raw_p,
                raw_s=raw_s,
                quality=quality,
                features=[raw_d / 100.0, raw_p / 100.0, raw_s / 100.0],
                derivatives=derivatives,
                coupling=coupling,
                hour=hour,
            )
        )
        hysteresis_composite = self._apply_hysteresis_correction(
            iter_composite
        )
        base_threshold = self._compute_collapse_threshold(hour=hour)
        drifted_threshold = self._drift_adaptive_threshold(
            base_threshold=base_threshold, hour=hour
        )
        perturbed_composite = self._apply_confidence_perturbation(
            hysteresis_composite, confidence=0.5
        )
        composite_ratio = perturbed_composite / max(0.1, drifted_threshold)
        final_confidence = self._compute_acfn_confidence(
            passes=iter_passes,
            delta_bar=delta_bar,
            delta_score=delta_score,
            composite_ratio=composite_ratio,
        )
        final_composite = self._apply_confidence_perturbation(
            hysteresis_composite, confidence=final_confidence
        )
        collapsing = final_composite > drifted_threshold
        s.composite_load = final_composite
        s.collapse_threshold = drifted_threshold
        s.collapse_imminent = collapsing
        if collapsing and s.collapse_time_estimate <= 0:
            remaining_headroom = drifted_threshold - final_composite
            if remaining_headroom < 0:
                deriv_contribution = (
                    abs(derivatives[IDX_DROWSINESS]) * 15.0
                    + abs(derivatives[IDX_PRESSURE]) * 10.0
                )
                rate_of_change = max(
                    0.01,
                    deriv_contribution
                    + abs(s.drowsiness_value) * 0.001
                    + abs(s.overnight_pressure) * 0.0008,
                )
                s.collapse_time_estimate = (
                    abs(remaining_headroom) / rate_of_change
                )
            else:
                s.collapse_time_estimate = 0.0
        behavior = self._compute_behavior_chain(
            composite=final_composite, threshold=drifted_threshold, hour=hour
        )
        s.laziness_probability = behavior["laziness_prob"]
        s.wake_irritation_level = behavior["wake_irritation"]
        s.half_asleep_level = behavior["half_asleep"]
        s.yawn_frequency = behavior["yawn_freq"]
        s.response_suppression_coef = behavior["suppression"]
        s.body_state_tag = behavior["body_tag"]
        s.monologue_mood_hint = behavior["mood_hint"]
        return {
            "drowsiness_value": round(s.drowsiness_value, 3),
            "overnight_pressure": round(s.overnight_pressure, 3),
            "sleep_reserve": round(s.sleep_reserve, 3),
            "composite_load": round(final_composite, 4),
            "collapse_threshold": round(drifted_threshold, 3),
            "collapse_imminent": collapsing,
            "collapse_estimate_sec": round(s.collapse_time_estimate, 1),
            "acfn_passes": iter_passes,
            "acfn_converged": s.convergence_achieved,
            "acfn_delta_bar": round(delta_bar, 6),
            "acfn_confidence": round(final_confidence, 4),
            "acfn_hysteresis_active": s.hysteresis_correction_active,
            "acfn_drift_amount": round(s.threshold_drift_accumulator, 4),
            "derivatives": s.derivative_snapshot,
            "coupling_activation": s.coupling_activation_snapshot,
            "behavior": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in behavior.items()
            },
        }

    def _tick_drowsiness_dimension(
        self,
        *,
        is_night: bool,
        energy_ratio: float,
        boredom: float,
        consecutive_active_minutes: float,
        hour: int,
        now: float,
    ) -> None:
        """维度一：困意值动态更新

        算法特征：
        1. 正弦波昼夜调制 — 凌晨3-5点困意波峰，下午2-4点波谷
        2. 高斯随机扰动 — 模拟生物节律的不规则性
        3. 间歇性更新 — 非每tick线性累加，而是基于时间差分批量结算
        4. 多因子叠加 — 能量比/无聊度/连续活跃时间共同影响基础速率
        5. 黎明恢复段 — 4-7点困意自然回落（皮质醇觉醒反应）
        """
        s = self._state
        dt = now - s.last_drowsiness_tick
        if dt < s.drowsiness_tick_interval:
            pending = s.drowsiness_pending_delta or 0.0
            s.drowsiness_pending_delta = pending
            return
        tick_count = max(1.0, int(dt / s.drowsiness_tick_interval))
        effective_dt = tick_count * s.drowsiness_tick_interval
        s.last_drowsiness_tick = now - (dt - effective_dt)
        _is_dawn = self._dawn_recover_start <= hour < self._sleep_end_hour
        if _is_dawn:
            dawn_recovery_strength = 0.6
            hour_progress = (hour - self._dawn_recover_start) / max(
                1, self._sleep_end_hour - self._dawn_recover_start
            )
            dawn_recovery_strength *= 0.5 + hour_progress * 0.8
            circadian_mod = self._circadian_sine(hour)
            noise = random.gauss(0, s.drowsiness_noise_amplitude * 0.5)
            delta = (
                -(dawn_recovery_strength + abs(circadian_mod) * 0.3 + noise)
                * tick_count
            )
            s.drowsiness_value = max(0.0, s.drowsiness_value + delta)
            s.drowsiness_pending_delta = 0.0
            return
        base_rate = s.drowsiness_base_rate
        if energy_ratio < 0.3:
            base_rate += 0.35
        elif energy_ratio < 0.5:
            base_rate += 0.15
        if boredom > 50:
            base_rate += 0.18 * (boredom / 100.0)
        if consecutive_active_minutes > 90:
            overwork_factor = min(
                1.5, (consecutive_active_minutes - 90) / 180.0
            )
            base_rate += overwork_factor * 0.4
        circadian_mod = self._circadian_sine(hour)
        if is_night:
            night_boost = 1.0 + abs(circadian_mod) * 0.8
            base_rate *= night_boost
        else:
            day_suppress = max(0.3, 1.0 - abs(circadian_mod) * 0.4)
            base_rate *= day_suppress
            base_rate = max(0.02, base_rate - 0.15)
        noise = random.gauss(0, s.drowsiness_noise_amplitude)
        effective_rate = base_rate + circadian_mod * 0.15 + noise
        effective_rate = max(0.01, effective_rate)
        delta = effective_rate * tick_count
        s.drowsiness_value = max(0.0, min(100.0, s.drowsiness_value + delta))
        s.drowsiness_pending_delta = 0.0

    def _circadian_sine(self, hour: int) -> float:
        """正弦波昼夜调制函数

        返回值范围 [-1.0, 1.0]:
          正值 → 困意增强时段（深夜/凌晨）
          负值 → 困意减弱时段（午后）
          零点交叉 → 过渡时段

        波峰在 circadian_peak_hour(默认3点，凌晨最困)
        波谷在 circadian_trough_hour(默认14点，下午最清醒)
        """
        s = self._state
        peak_rad = math.radians((s.circadian_peak_hour / 24.0) * 360.0)
        trough_rad = math.radians((s.circadian_trough_hour / 24.0) * 360.0)
        hour_norm = (hour + s.circadian_phase_offset) / 24.0
        hour_rad = math.radians(hour_norm * 360.0)
        phase_align = (peak_rad + trough_rad) / 2.0
        amplitude = 1.0
        value = amplitude * math.sin(2.0 * (hour_rad - phase_align))
        return max(-1.0, min(1.0, value))

    def _tick_pressure_dimension(
        self,
        *,
        is_night: bool,
        social_stimulus: float,
        activity_level: float,
        hour: int,
        now: float,
    ) -> None:
        """维度二：熬夜压力动态累积

        算法特征：
        1. 从 pressure_start_hour(默认22点) 开始激活
        2. 指数加速：越晚→加速度越大→压力暴涨
        3. 活动加成：聊天/思考/被@ 都会额外增加压力
        4. 压力历史记录：用于计算二阶导数（加速度变化率）
        5. 白天缓慢衰减：不是归零而是逐渐消散
        """
        s = self._state
        dt = now - s.pressure_last_tick
        if dt < 3.0:
            return
        s.pressure_last_tick = now
        hours_past_start = 0.0
        if hour >= s.pressure_start_hour:
            hours_past_start = hour - s.pressure_start_hour
        elif hour < self._sleep_end_hour:
            hours_past_start = (24 - s.pressure_start_hour) + hour
        if not is_night and hours_past_start <= 0:
            decay = 0.008 * (dt / 60.0)
            s.overnight_pressure = max(0.0, s.overnight_pressure - decay)
            s.pressure_activity_multiplier = max(
                1.0, s.pressure_activity_multiplier * 0.995
            )
            self._record_pressure_history(now, s.overnight_pressure)
            return
        if hours_past_start <= 0 and is_night:
            hours_past_start = max(0.0, hour - 19.0)
        base_acc = s.pressure_acceleration_base
        exp_component = (
            math.exp(hours_past_start * s.pressure_exponential_factor) - 1.0
        )
        exp_component = min(exp_component, 15.0)
        time_accelerated_rate = base_acc * (1.0 + exp_component)
        activity_bonus = 0.0
        if social_stimulus > 0.3:
            activity_bonus += social_stimulus * 0.06
        if activity_level > 60:
            activity_bonus += (activity_level - 60) / 200.0
        if s.pressure_chat_minutes > 30:
            chat_overload = min(0.15, s.pressure_chat_minutes / 600.0)
            activity_bonus += chat_overload
        if s.pressure_interrupt_count > 5:
            interrupt_penalty = min(0.20, s.pressure_interrupt_count * 0.025)
            activity_bonus += interrupt_penalty
        effective_rate = (
            time_accelerated_rate * s.pressure_activity_multiplier
            + activity_bonus
        )
        pressure_delta = effective_rate * (dt / 60.0)
        s.overnight_pressure = min(
            100.0, s.overnight_pressure + pressure_delta
        )
        if s.overnight_pressure > s.pressure_critical_threshold * 0.7:
            overload_mult = (
                1.0
                + (s.overnight_pressure - s.pressure_critical_threshold * 0.7)
                / 50.0
            )
            s.pressure_activity_multiplier = min(
                3.0, s.pressure_activity_multiplier * overload_mult
            )
        self._record_pressure_history(now, s.overnight_pressure)

    def _record_pressure_history(self, timestamp: float, value: float) -> None:
        """记录压力历史用于二阶导数计算"""
        s = self._state
        s.pressure_history[timestamp] = value
        if len(s.pressure_history) > s.pressure_history_max_len:
            oldest_key = min(s.pressure_history.keys())
            del s.pressure_history[oldest_key]

    def _get_pressure_acceleration(self) -> float:
        """计算压力变化的二阶导数（加速度）

        正值 → 压力增长在加速（越来越快接近崩溃）
        负值 → 压力增长在减速（可能在恢复）
        接近零 → 压力线性变化
        """
        s = self._state
        history = s.pressure_history
        if len(history) < 4:
            return 0.0
        sorted_ts = sorted(history.keys())
        values = [history[t] for t in sorted_ts]
        n = len(values)
        if n < 4:
            return 0.0
        first_derivatives = []
        for i in range(1, n):
            dt = sorted_ts[i] - sorted_ts[i - 1]
            if dt > 0:
                first_derivatives.append((values[i] - values[i - 1]) / dt)
        if len(first_derivatives) < 2:
            return 0.0
        second_derivatives = []
        for i in range(1, len(first_derivatives)):
            second_derivatives.append(
                first_derivatives[i] - first_derivatives[i - 1]
            )
        if not second_derivatives:
            return 0.0
        return sum(second_derivatives) / len(second_derivatives)

    def _tick_reserve_dimension(
        self, *, hour: int, now: float, energy_ratio: float
    ) -> None:
        """维度三：睡眠储备更新

        睡眠储备(S)是相乘相消公式中的分母:
          composite = D × P / √(S + ε)

        S越高 → 分母越大 → 复合负载越低 → 能多撑一会
        S越低 → 分母越小 → 复合负载越高 → 快要崩

        恢复来源:
          - 浅睡阶段: 缓慢恢复(rate=0.4/tick)
          - 深睡阶段: 快速恢复(rate=1.2/tick)
          - 自然醒来时: 按睡眠质量批量补充

        消耗来源:
          - 清醒状态: 极慢自然消耗(0.03/tick)
          - 每次回复: 扣减固定成本(0.8)
          - 强制唤醒: 大额惩罚扣减
        """
        s = self._state
        phase = s.current_phase
        if phase == NightPhase.DEEP_SLEEP:
            recovery = s.deep_sleep_recovery_rate
            quality_boost = 1.0 + s.last_sleep_quality * 0.5
            recovery *= quality_boost
            s.sleep_reserve = min(
                s.sleep_reserve_max, s.sleep_reserve + recovery
            )
            s.reserve_quality_factor = min(
                1.5, s.reserve_quality_factor + 0.02
            )
        elif phase == NightPhase.LIGHT_SLEEP:
            recovery = s.light_sleep_recovery_rate
            stage_duration = (
                now - s.phase_entered_at if s.phase_entered_at > 0 else 0
            )
            if stage_duration > 300:
                deepening_bonus = min(0.3, (stage_duration - 300) / 1200.0)
                recovery += deepening_bonus
            s.sleep_reserve = min(
                s.sleep_reserve_max, s.sleep_reserve + recovery
            )
            s.reserve_quality_factor = min(
                1.3, s.reserve_quality_factor + 0.01
            )
        elif phase in (
            NightPhase.AWAKE,
            NightPhase.DROWSY,
            NightPhase.EXCITED,
            NightPhase.NIGHT_ACTIVE,
            NightPhase.SOCIAL_NIGHT,
            NightPhase.MIDNIGHT_REFLECT,
            NightPhase.QUIET_CONTEMPLATE,
            NightPhase.DAWN_RECOVER,
            NightPhase.BURNED_OUT,
        ):
            drain = s.awake_drain_rate
            if energy_ratio < 0.3:
                drain *= 2.5
            elif energy_ratio < 0.5:
                drain *= 1.5
            if s.overnight_pressure > 50:
                pressure_drain = (s.overnight_pressure / 100.0) * 0.04
                drain += pressure_drain
            s.sleep_reserve = max(s.sleep_reserve_min, s.sleep_reserve - drain)
            s.reserve_quality_factor = max(
                0.6, s.reserve_quality_factor - 0.002
            )
        s.sleep_reserve = max(
            s.sleep_reserve_min, min(s.sleep_reserve_max, s.sleep_reserve)
        )

    def consume_reserve_for_reply(self) -> float:
        """每次回复时消耗睡眠储备"""
        s = self._state
        cost = s.reply_cost
        if s.overnight_pressure > 60:
            overtime_cost = cost * (s.overnight_pressure / 100.0) * 0.6
            cost += overtime_cost
        if s.drowsiness_value > 70:
            drowsy_cost = cost * (s.drowsiness_value / 100.0) * 0.4
            cost += drowsy_cost
        old_reserve = s.sleep_reserve
        s.sleep_reserve = max(s.sleep_reserve_min, s.sleep_reserve - cost)
        return old_reserve - s.sleep_reserve

    def apply_force_wake_penalty(self, strength: float = 1.0) -> None:
        """强制唤醒惩罚：大额扣减睡眠储备 + 记录唤醒痕迹

        强制唤醒会导致:
        1. 睡眠储备大幅下降（强度越大扣越多）
        2. 唤醒烦躁度上升（后续影响回复语气）
        3. 困意值反弹（被打断的睡眠需求不会消失）
        4. 半梦半醒程度上升（意识模糊）
        """
        s = self._state
        now = time.time()
        base_penalty = 8.0 * strength
        if s.current_phase == NightPhase.DEEP_SLEEP:
            base_penalty *= 2.5
        elif s.current_phase == NightPhase.LIGHT_SLEEP:
            base_penalty *= 1.5
        s.sleep_reserve = max(
            s.sleep_reserve_min, s.sleep_reserve - base_penalty
        )
        s.force_wake_count_today += 1
        s.last_force_wake_time = now
        irritation_spike = 0.35 * strength
        if s.force_wake_count_today >= 3:
            irritation_spike *= 1.0 + s.force_wake_count_today * 0.3
        s.wake_irritation_level = min(
            1.0, s.wake_irritation_level + irritation_spike
        )
        s.half_asleep_level = min(1.0, s.half_asleep_level + 0.25 * strength)
        rebound = 12.0 * strength
        s.drowsiness_value = min(100.0, s.drowsiness_value + rebound)
        s.force_wake_irritation_decay = irritation_spike

    def decay_force_wake_effects(self, elapsed_seconds: float) -> None:
        """随时间衰减强制唤醒的负面效果"""
        s = self._state
        if elapsed_seconds <= 0:
            return
        decay_rate = 0.0008
        irritation_decay = s.force_wake_irritation_decay * math.exp(
            -decay_rate * elapsed_seconds
        )
        s.wake_irritation_level = max(
            0.0,
            s.wake_irritation_level
            - (s.force_wake_irritation_decay - irritation_decay),
        )
        s.force_wake_irritation_decay = max(0.0, irritation_decay)
        half_asleep_decay = 0.001 * elapsed_seconds
        s.half_asleep_level = max(0.0, s.half_asleep_level - half_asleep_decay)

    def _compute_composite_load(self) -> float:
        """三维复合负载计算（核心公式）

        composite_load = D × P / √(S × Q + ε)

        其中:
          D = drowsiness_value (困意值, 0~100)
          P = overnight_pressure (熬夜压力, 0~100)
          S = sleep_reserve (睡眠储备, 0~100)
          Q = reserve_quality_factor (储备质量因子, 0.6~1.5)
          ε = 1e-6 (防除零)

        相乘相消逻辑:
          - D高P高S低 → 分子大分母小 → composite飙升 → 快崩
          - D高P高但S也高 → 分母大增 → composite被压制 → 还能撑
          - D低或P低 → 分子小 → composite低 → 没事

        返回值范围: 0 ~ 约10000（原始值），通常在 0~200 之间有意义
        """
        s = self._state
        D = max(0.0, s.drowsiness_value)
        P = max(0.0, s.overnight_pressure)
        S = max(1e-6, s.sleep_reserve)
        Q = max(0.1, s.reserve_quality_factor)
        epsilon = 1e-6
        denominator = math.sqrt(S * Q + epsilon)
        raw_composite = (D * P) / denominator
        normalized = raw_composite / 100.0
        return max(0.0, normalized)

    def _compute_collapse_threshold(
        self, *, hour: Optional[int] = None
    ) -> float:
        """动态崩溃阈值计算

        阈值不是固定的，而是随以下因素变化:
        1. 基础阈值: 个人基准线(默认50)
        2. 昼夜调制: 深夜阈值降低（更容易崩）白天阈值升高
        3. 历史适应: 经常熬夜的人阈值略微提高（耐受性）
        4. 当前阶段: 不同NightPhase有不同的阈值修正

        返回值: 崩溃触发阈值，composite_load超过此值则判定为"熬不住了"
        """
        s = self._state
        base_threshold = 50.0
        h = hour if hour is not None else time.localtime().tm_hour
        if 2 <= h < 6:
            time_modifier = 0.65
        elif 22 <= h or h < 2:
            time_modifier = 0.78
        elif 6 <= h < 9:
            time_modifier = 1.15
        elif 13 <= h < 16:
            time_modifier = 1.25
        else:
            time_modifier = 1.0
        phase = s.current_phase
        if phase == NightPhase.BURNED_OUT:
            phase_modifier = 0.4
        elif phase == NightPhase.EXCITED:
            phase_modifier = 1.35
        elif phase == NightPhase.DEEP_SLEEP:
            phase_modifier = 2.0
        elif phase == NightPhase.LIGHT_SLEEP:
            phase_modifier = 1.6
        elif phase == NightPhase.DROWSY:
            phase_modifier = 0.88
        else:
            phase_modifier = 1.0
        adaptivity_boost = 0.0
        if s.force_wake_count_today >= 5:
            adaptivity_boost = 3.0
        elif s.force_wake_count_today >= 3:
            adaptivity_boost = 1.5
        reserve_buffer = (
            s.sleep_reserve / max(1.0, s.sleep_reserve_max)
        ) * 8.0
        dynamic_threshold = (
            base_threshold * time_modifier * phase_modifier
            + adaptivity_boost
            + reserve_buffer
        )
        return max(15.0, min(120.0, dynamic_threshold))

    def _compute_behavior_chain(
        self,
        composite: float,
        threshold: float,
        hour: int,
    ) -> Dict[str, Any]:
        """行为链计算：从三维复合状态推导出所有行为输出

        输入: composite_load, collapse_threshold, 当前小时
        输出: 包含赖床概率/唤醒烦躁/半梦半醒等的行为字典

        行为之间的关联逻辑:
          - composite接近threshold → 赖床概率上升
          - 刚被强制唤醒 + composite高 → 烦躁度高
          - 睡眠储备低 + 困意高 → 半梦半醒程度深
          - 困意中等 + 压力中等 → 打哈欠频率增加
          - composite超过80%阈值 → 回复抑制系数上升
        """
        s = self._state
        ratio = composite / max(0.1, threshold)
        ratio_clamped = max(0.0, min(2.5, ratio))
        laziness_base = 0.0
        if s.sleep_reserve < 30:
            laziness_base += (30 - s.sleep_reserve) / 30.0 * 0.45
        if s.drowsiness_value > 55:
            laziness_base += (s.drowsiness_value - 55) / 45.0 * 0.35
        if ratio_clamped > 0.6:
            laziness_base += (ratio_clamped - 0.6) * 0.6
        if s.in_lazy_state:
            laziness_base = min(1.0, laziness_base + 0.25)
        laziness_prob = max(0.0, min(1.0, laziness_base))
        wake_irr = s.wake_irritation_level
        time_since_wake = time.time() - s.last_force_wake_time
        if time_since_wake < 300 and s.force_wake_count_today > 0:
            freshness = 1.0 - (time_since_wake / 300.0)
            wake_irr += freshness * s.force_wake_irritation_decay * 0.5
        if composite > threshold * 0.7:
            wake_irr += (ratio_clamped - 0.7) * 0.3
        if s.overnight_pressure > 70:
            wake_irr += (s.overnight_pressure - 70) / 30.0 * 0.2
        wake_irritation = max(0.0, min(1.0, wake_irr))
        half_asleep = s.half_asleep_level
        if s.drowsiness_value > 65 and s.sleep_reserve < 25:
            deficit_depth = ((s.drowsiness_value - 65) / 35.0) * 0.4
            reserve_depth = ((25 - s.sleep_reserve) / 25.0) * 0.35
            half_asleep += deficit_depth + reserve_depth
        if s.current_phase in (NightPhase.LIGHT_SLEEP, NightPhase.DEEP_SLEEP):
            half_asleep = max(
                half_asleep,
                0.6 if s.current_phase == NightPhase.DEEP_SLEEP else 0.35,
            )
        half_asleep = max(0.0, min(1.0, half_asleep))
        yawn = 0.0
        if 40 < s.drowsiness_value < 75:
            yawn_zone = (s.drowsiness_value - 40) / 35.0
            yawn = yawn_zone * 0.5 * (math.sin(time.time() / 30.0) * 0.3 + 0.7)
        if 23 <= hour or hour < 5:
            yawn *= 1.4
        if s.sleep_reserve < 40:
            yawn += (40 - s.sleep_reserve) / 40.0 * 0.25
        yawn_frequency = max(0.0, min(1.0, yawn))
        suppression = 0.0
        if ratio_clamped > 0.4:
            suppression = (ratio_clamped - 0.4) ** 1.5 * 0.7
        if s.current_phase == NightPhase.DEEP_SLEEP:
            suppression = max(suppression, 0.95)
        elif s.current_phase == NightPhase.LIGHT_SLEEP:
            suppression = max(suppression, 0.75)
        elif s.current_phase == NightPhase.BURNED_OUT:
            suppression = max(suppression, 0.85)
        if s.half_asleep_level > 0.5:
            suppression = max(suppression, s.half_asleep_level * 0.5)
        suppression = max(0.0, min(0.98, suppression))
        body_tag = self._resolve_body_state_tag(
            ratio_clamped=ratio_clamped,
            laziness_prob=laziness_prob,
            wake_irritation=wake_irritation,
            half_asleep=half_asleep,
            yawn=yawn_frequency,
        )
        mood_hint = self._resolve_monologue_mood(
            composite_ratio=ratio_clamped,
            drowsiness=s.drowsiness_value,
            pressure=s.overnight_pressure,
            reserve=s.sleep_reserve,
            irritation=wake_irritation,
        )
        return {
            "laziness_prob": round(laziness_prob, 4),
            "wake_irritation": round(wake_irritation, 4),
            "half_asleep": round(half_asleep, 4),
            "yawn_freq": round(yawn_frequency, 4),
            "suppression": round(suppression, 4),
            "body_tag": body_tag,
            "mood_hint": mood_hint,
        }

    def _resolve_body_state_tag(
        self,
        ratio_clamped: float,
        laziness_prob: float,
        wake_irritation: float,
        half_asleep: float,
        yawn: float,
    ) -> str:
        """根据三维状态解析为身体状态标签（供prompt和日志使用）"""
        if half_asleep > 0.7:
            return "deep_half_asleep"
        if half_asleep > 0.4:
            return "light_half_asleep"
        if wake_irritation > 0.6:
            return "irritated_woken"
        if wake_irritation > 0.3:
            return "slightly_grumpy"
        if laziness_prob > 0.65:
            return "stubborn_lazy"
        if laziness_prob > 0.35:
            return "reluctant_wake"
        if yawn > 0.5:
            return "frequently_yawning"
        if yawn > 0.25:
            return "occasionally_yawning"
        if ratio_clamped > 0.85:
            return "on_verge_collapse"
        if ratio_clamped > 0.65:
            return "heavily_strained"
        if ratio_clamped > 0.4:
            return "noticeably_tired"
        return "normal"

    def _resolve_monologue_mood(
        self,
        composite_ratio: float,
        drowsiness: float,
        pressure: float,
        reserve: float,
        irritation: float,
    ) -> str:
        """根据三维状态解析为内心独白情绪提示词"""
        if irritation > 0.6:
            return "烦躁不想说话"
        if irritation > 0.3:
            return "有点不爽被打扰"
        if composite_ratio > 0.9:
            return "熬不住了好想睡"
        if composite_ratio > 0.7:
            return "眼皮打架撑不住了"
        if drowsiness > 75 and pressure > 60:
            return "困得要死还在硬撑"
        if drowsiness > 60:
            return "昏昏沉沉想趴下"
        if pressure > 55:
            return "身体在抗议该休息了"
        if reserve < 25:
            return "电量见底需要充电"
        if drowsiness > 40:
            return "有点犯困"
        if composite_ratio > 0.3:
            return "有点疲惫"
        return ""

    def get_3d_engine_status(self) -> Dict[str, Any]:
        """获取三维引擎完整状态快照（供外部查询）"""
        s = self._state
        return {
            "dimensions": {
                "D_drowsiness": {
                    "value": round(s.drowsiness_value, 3),
                    "base_rate": s.drowsiness_base_rate,
                    "noise_amp": s.drowsiness_noise_amplitude,
                    "peak_hour": s.circadian_peak_hour,
                    "trough_hour": s.circadian_trough_hour,
                },
                "P_pressure": {
                    "value": round(s.overnight_pressure, 3),
                    "start_hour": s.pressure_start_hour,
                    "critical_threshold": s.pressure_critical_threshold,
                    "activity_mult": round(s.pressure_activity_multiplier, 3),
                    "acceleration": round(
                        self._get_pressure_acceleration(), 6
                    ),
                },
                "S_reserve": {
                    "value": round(s.sleep_reserve, 3),
                    "max": s.sleep_reserve_max,
                    "quality_factor": round(s.reserve_quality_factor, 3),
                    "reply_cost": s.reply_cost,
                },
            },
            "composite": {
                "load": round(s.composite_load, 4),
                "threshold": round(s.collapse_threshold, 3),
                "ratio": (
                    round(
                        s.composite_load / max(0.01, s.collapse_threshold), 3
                    )
                    if s.collapse_threshold > 0
                    else 0
                ),
                "imminent": s.collapse_imminent,
                "estimate_sec": round(s.collapse_time_estimate, 1),
            },
            "behavior": {
                "laziness_prob": round(s.laziness_probability, 4),
                "wake_irritation": round(s.wake_irritation_level, 4),
                "half_asleep": round(s.half_asleep_level, 4),
                "yawn_freq": round(s.yawn_frequency, 4),
                "response_suppression": round(s.response_suppression_coef, 4),
                "body_tag": s.body_state_tag,
                "mood_hint": s.monologue_mood_hint,
            },
            "force_wake_tracking": {
                "count_today": s.force_wake_count_today,
                "last_time": s.last_force_wake_time,
                "irritation_decay": round(s.force_wake_irritation_decay, 4),
            },
            "lazy_state": {
                "active": s.in_lazy_state,
                "started": s.lazy_state_started,
                "resistance_cycles": s.lazy_resistance_cycles,
            },
        }

    # ════════════════════════════════════════════════
    #  ★★★ 能量条双形态系统 (Energy Dual-Mode) ★★★
    # ════════════════════════════════════════════════

    def resolve_energy_mode(self, hour: Optional[int] = None) -> str:
        h = hour if hour is not None else time.localtime().tm_hour
        s = self._state
        if 7 <= h < 19:
            s.energy_mode = "day"
            s.transition_blend_factor = 0.0
        elif 22 <= h or h < 5:
            s.energy_mode = "night"
            s.transition_blend_factor = 0.0
        elif 19 <= h < 22:
            s.energy_mode = "evening_transition"
            s.transition_blend_factor = (h - 19) / 3.0
        else:
            s.energy_mode = "dawn_transition"
            s.transition_blend_factor = (h - 5) / 2.0
        self._apply_time_segment_modifier(h)
        return s.energy_mode

    def _apply_time_segment_modifier(self, hour: int) -> None:
        s = self._state
        if 7 <= hour < 11:
            s.time_segment_modifier = 0.9
        elif 11 <= hour < 13:
            s.time_segment_modifier = 1.0
        elif 13 <= hour < 15:
            s.time_segment_modifier = 1.2
        elif 15 <= hour < 17:
            s.time_segment_modifier = 1.05
        elif 17 <= hour < 19:
            s.time_segment_modifier = 1.15
        elif 19 <= hour < 22:
            s.time_segment_modifier = 1.4
        elif 22 <= hour or hour < 2:
            s.time_segment_modifier = 2.0
        elif 2 <= hour < 5:
            s.time_segment_modifier = 1.8
        else:
            s.time_segment_modifier = 0.92

    def compute_reply_energy_cost(
        self,
        *,
        message_length: int = 0,
        complexity: float = 1.0,
        emotional_intensity: float = 1.0,
        current_hour: Optional[int] = None,
    ) -> float:
        s = self._state
        mode = self.resolve_energy_mode(current_hour)
        if mode == "day":
            base = s.day_reply_cost_base
            mode_mult = 1.0
        elif mode == "night":
            base = s.day_reply_cost_base * s.night_reply_cost_multiplier
            mode_mult = s.night_reply_cost_multiplier
        else:
            day_base = s.day_reply_cost_base
            night_base = day_base * s.night_reply_cost_multiplier
            base = (
                day_base + (night_base - day_base) * s.transition_blend_factor
            )
            mode_mult = (
                1.0
                + (s.night_reply_cost_multiplier - 1.0)
                * s.transition_blend_factor
            )
        seg_mod = s.time_segment_modifier
        complexity_clamped = max(0.3, min(3.0, complexity))
        emotion_clamped = max(0.5, min(2.5, emotional_intensity))
        length_factor = 1.0 + min(0.5, message_length / 500.0)
        buff_mod = self._get_buff_cost_modifier()
        raw_cost = (
            base
            * mode_mult
            * seg_mod
            * complexity_clamped
            * emotion_clamped
            * length_factor
        )
        final_cost = raw_cost * buff_mod
        if s.overnight_pressure > 60:
            final_cost += (s.overnight_pressure / 100.0) * 0.25 * final_cost
        if s.sleep_reserve < 25:
            final_cost += (25 - s.sleep_reserve) / 25.0 * 0.2 * final_cost
        now = time.time()
        s.cost_history[now] = final_cost
        s.cumulative_today_cost += final_cost
        if len(s.cost_history) > 200:
            oldest = min(s.cost_history.keys())
            del s.cost_history[oldest]
        s.energy_value = max(0.0, s.energy_value - final_cost)
        return final_cost

    def compute_think_energy_cost(
        self,
        *,
        think_units: float = 1.0,
        depth_level: float = 1.0,
        current_hour: Optional[int] = None,
    ) -> float:
        s = self._state
        mode = self.resolve_energy_mode(current_hour)
        think_base = s.think_cost_per_unit
        if mode == "night":
            think_base *= s.night_think_cost_multiplier
        elif mode in ("evening_transition", "dawn_transition"):
            night_think = s.think_cost_per_unit * s.night_think_cost_multiplier
            think_base = (
                s.think_cost_per_unit
                + (night_think - s.think_cost_per_unit)
                * s.transition_blend_factor
            )
        depth_clamped = max(0.5, min(3.0, depth_level))
        raw_cost = (
            think_units * think_base * depth_clamped * s.time_segment_modifier
        )
        s.think_active_value = max(
            0.0, s.think_active_value - think_units * 10.0
        )
        s.energy_value = max(0.0, s.energy_value - raw_cost)
        s.cumulative_today_cost += raw_cost
        return raw_cost

    def apply_event_recovery(
        self,
        *,
        event_type: str,
        base_amount: float,
        current_hour: Optional[int] = None,
    ) -> float:
        s = self._state
        type_map = {
            "noon_rest": s.noon_event_recovery_bonus,
            "snack": 5.0,
            "drink_water": s.buff_water_energy_bonus,
            "stretch": 2.0,
            "social_fun": 4.0,
            "interesting": 3.0,
        }
        amount = type_map.get(event_type, base_amount)
        mode = self.resolve_energy_mode(current_hour)
        if mode == "night":
            amount *= 0.4
        elif mode in ("evening_transition", "dawn_transition"):
            amount *= 0.6 + 0.4 * (1.0 - s.transition_blend_factor)
        amount *= self._get_buff_recovery_modifier()
        pool_space = s.event_recovery_cap - s.event_recovery_pool
        if pool_space <= 0:
            return 0.0
        actual = min(amount, pool_space)
        s.event_recovery_pool += actual
        s.cumulative_today_recovery += actual
        s.energy_value = min(s.energy_max, s.energy_value + actual)
        now = time.time()
        s.recovery_history[now] = actual
        if len(s.recovery_history) > 100:
            oldest = min(s.recovery_history.keys())
            del s.recovery_history[oldest]
        return actual

    def tick_natural_energy_recovery(
        self, elapsed_minutes: float = 1.0
    ) -> float:
        s = self._state
        mode = s.energy_mode
        if mode not in (
            "day",
            "night",
            "evening_transition",
            "dawn_transition",
        ):
            mode = self.resolve_energy_mode()
        if mode == "day":
            rate = s.day_recovery_rate_per_min
        elif mode == "night":
            rate = s.day_recovery_rate_per_min * 0.08
        elif mode == "evening_transition":
            dr = s.day_recovery_rate_per_min
            nr = dr * 0.08
            rate = dr + (nr - dr) * s.transition_blend_factor
        else:
            dr = s.day_recovery_rate_per_min
            rate = dr + (dr * 1.4 - dr) * s.transition_blend_factor
        rate *= 2.0 - s.time_segment_modifier
        if 7 <= time.localtime().tm_hour < 11:
            rate *= 1.2
        recovery = rate * random.uniform(0.85, 1.15) * elapsed_minutes
        recovery *= self._get_buff_recovery_modifier()
        old_val = s.energy_value
        s.energy_value = min(s.energy_max, s.energy_value + recovery)
        self._check_overwork_protection()
        return s.energy_value - old_val

    def _check_overwork_protection(self) -> None:
        s = self._state
        if (
            s.energy_mode == "night"
            and s.energy_value
            < s.night_protection_threshold * 0.01 * s.energy_max
        ):
            if not s.overwork_protection_active:
                s.overwork_protection_active = True
                logger.debug(
                    f"[过度保护] {self._channel_id} 能量过低({s.energy_value:.1f}/{s.energy_max})"
                )
        elif (
            s.energy_value > s.night_protection_threshold * 0.25 * s.energy_max
        ):
            s.overwork_protection_active = False

    # ── 睡眠→能量映射 ──

    def map_sleep_to_energy(self) -> Dict[str, Any]:
        s = self._state
        S = s.sleep_reserve
        Q = max(0.1, s.reserve_quality_factor)
        E_max = s.energy_max
        alpha = 6.0 * s.sleep_to_energy_ratio
        weighted_S = (S / max(1.0, s.sleep_reserve_max)) * 100.0
        quality_adjusted = weighted_S * (
            0.5 + s.sleep_to_energy_quality_weight * Q
        )
        if s.mapping_efficiency_curve == "logistic":
            z = alpha * (quality_adjusted / 100.0) - 3.0
            extraction_ratio = 1.0 / (1.0 + math.exp(-z))
        elif s.mapping_efficiency_curve == "quadratic":
            extraction_ratio = (quality_adjusted / 100.0) ** 1.8
        else:
            extraction_ratio = min(1.0, quality_adjusted / 100.0)
        if s.sleep_to_energy_time_discount > 0:
            extraction_ratio *= math.exp(
                -s.sleep_to_energy_time_discount / 3600.0
            )
        raw_gain = E_max * extraction_ratio
        capped_gain = min(raw_gain, E_max - s.energy_value)
        s.energy_from_last_sleep = capped_gain
        s.last_sleep_mapping_time = time.time()
        s.energy_value = min(s.energy_max, s.energy_value + capped_gain)
        return {
            "reserve": round(S, 2),
            "quality": round(Q, 3),
            "extraction": round(extraction_ratio, 4),
            "gain": round(capped_gain, 3),
            "energy_after": round(s.energy_value, 3),
        }

    # ── 动态预知唤醒 ──

    def evaluate_predictive_wake(self) -> Dict[str, Any]:
        s = self._state
        if not s.predictive_wake_enabled:
            return {
                "should_wake": False,
                "confidence": 0.0,
                "reason": "未启用",
            }
        phase = s.current_phase
        if phase not in (NightPhase.LIGHT_SLEEP, NightPhase.DEEP_SLEEP):
            return {
                "should_wake": False,
                "confidence": 0.0,
                "reason": f"非睡眠阶段({
                    phase.value})",
            }
        reserve_score = min(
            1.0, s.sleep_reserve / s.reserve_sufficiency_threshold
        )
        drowsiness_peak = max(s.drowsiness_value, 70.0)
        drowsiness_falloff = max(
            0.0,
            (drowsiness_peak - s.drowsiness_value) / max(1.0, drowsiness_peak),
        )
        drowsiness_score = (
            1.0
            if s.drowsiness_value < s.drowsiness_falloff_threshold
            else drowsiness_falloff
        )
        pressure_score = max(
            0.0,
            1.0
            - (s.overnight_pressure / max(1.0, s.pressure_dissipation_target)),
        )
        total_score = (
            0.40 * reserve_score
            + 0.35 * drowsiness_score
            + 0.25 * pressure_score
        )
        s.wake_readiness_score = total_score
        if total_score >= 0.75:
            confidence = min(1.0, (total_score - 0.75) / 0.25)
            should_wake = True
        elif total_score >= 0.55:
            confidence = (total_score - 0.55) / 0.20
            should_wake = confidence > random.random()
        else:
            confidence = 0.0
            should_wake = False
        s.predictive_wake_confidence = confidence
        if should_wake:
            s.wake_decision_made = True
            s.early_wake_reason = (
                "储备充足"
                if reserve_score > 0.75
                else ("困意消退" if drowsiness_score > 0.7 else "压力消散")
            )
            s.predicted_wake_time = (
                time.time()
                + s.natural_wake_anticipation_sec * (1.0 - confidence)
            )
        else:
            s.wake_decision_made = False
            s.early_wake_reason = ""
            s.predicted_wake_time = 0.0
        return {
            "should_wake": should_wake,
            "confidence": round(confidence, 4),
            "score": round(total_score, 4),
            "reason": s.early_wake_reason or "尚未满足",
        }

    def should_predictively_wake(self) -> bool:
        return self.evaluate_predictive_wake()["should_wake"]

    # ── 特殊事件/敏感话题 ──

    def record_sensitive_topic(
        self, topic_tag: str = "general"
    ) -> Dict[str, Any]:
        s = self._state
        s.sensitive_topic_counter += 1
        s.sensitive_topic_tags[topic_tag] = (
            s.sensitive_topic_tags.get(topic_tag, 0) + 1
        )
        result = {
            "counter": s.sensitive_topic_counter,
            "threshold": s.sensitive_topic_threshold,
            "can_trigger": s.sensitive_topic_counter
            >= s.sensitive_topic_threshold
            and not s.special_event_triggered
            and not s.weakness_active,
        }
        if result["can_trigger"] and random.random() < 0.08:
            self._trigger_special_weakness_event()
            result["triggered"] = True
        return result

    def _trigger_special_weakness_event(self) -> None:
        s = self._state
        now = time.time()
        s.special_event_triggered = True
        s.special_event_time = now
        s.weakness_active = True
        duration = s.weakness_duration_base * (0.8 + random.random() * 0.4)
        s.weakness_end_time = now + duration
        fatigue_deduct = s.weakness_fatigue_cost * (
            0.85 + random.random() * 0.3
        )
        s.daily_fatigue = min(
            s.daily_fatigue_limit, s.daily_fatigue + fatigue_deduct
        )
        s.sleep_reserve = max(s.sleep_reserve_min, s.sleep_reserve - 35.0)
        s.energy_value = max(0.0, s.energy_value - 6.0)
        s.wake_irritation_level = min(1.0, s.wake_irritation_level + 0.45)
        logger.info(
            f"[特殊事件] {
                self._channel_id} 虚弱 持续{
                duration /
                60:.0f}min 疲劳+{
                fatigue_deduct:.0f}"
        )

    def check_weakness_state(self) -> Dict[str, Any]:
        s = self._state
        if not s.weakness_active:
            return {"active": False}
        now = time.time()
        if now >= s.weakness_end_time:
            s.weakness_active = False
            s.energy_value = min(s.energy_max, s.energy_value + 1.5)
            s.wake_irritation_level = max(0.0, s.wake_irritation_level - 0.2)
            return {"active": False, "just_ended": True}
        remaining_min = (s.weakness_end_time - now) / 60.0
        severity = (
            "high"
            if remaining_min > 20
            else ("medium" if remaining_min > 10 else "low")
        )
        hint = (
            "身体很虚很累，四肢无力"
            if remaining_min > 20
            else ("还是有点累" if remaining_min > 10 else "稍微恢复了一点")
        )
        return {
            "active": True,
            "severity": severity,
            "remaining_min": round(remaining_min, 1),
            "hint": hint,
        }

    def get_weakness_reply_modifier(self) -> str:
        state = self.check_weakness_state()
        if state.get("active") and state.get("hint"):
            return f"[当前身体状态: {state['hint']}]"
        return ""

    # ── Buff系统 ──

    def activate_coffee_buff(self) -> None:
        s = self._state
        now = time.time()
        boost = s.daily_fatigue_limit * (
            s.buff_coffee_fatigue_limit_bonus / 100.0
        )
        s.active_buffs["coffee"] = {
            "name": "咖啡因",
            "type": "fatigue_limit",
            "value": s.buff_coffee_fatigue_limit_bonus,
            "boost_amount": boost,
            "start": now,
            "end": now + s.buff_coffee_duration,
        }
        s.daily_fatigue_limit += boost

    def activate_water_buff(self) -> float:
        s = self._state
        s.energy_value = min(
            s.energy_max, s.energy_value + s.buff_water_energy_bonus
        )
        return s.buff_water_energy_bonus

    def cleanup_expired_buffs(self) -> int:
        s = self._state
        now = time.time()
        expired = [
            k for k, v in s.active_buffs.items() if v.get("end", 0) <= now
        ]
        count = 0
        for k in expired:
            buff = s.active_buffs.pop(k)
            if buff.get("type") == "fatigue_limit":
                boost_amount = buff.get("boost_amount", 0)
                s.daily_fatigue_limit = max(
                    100.0,
                    s.daily_fatigue_limit - boost_amount,
                )
            count += 1
        return count

    def _get_buff_cost_modifier(self) -> float:
        self.cleanup_expired_buffs()
        mod = 1.0
        for b in self._state.active_buffs.values():
            if b.get("type") == "cost_reduction":
                mod *= 1.0 - b["value"] / 100.0
        return max(0.3, mod)

    def _get_buff_recovery_modifier(self) -> float:
        self.cleanup_expired_buffs()
        mult = 1.0
        for b in self._state.active_buffs.values():
            if b.get("type") == "recovery_boost":
                mult *= 1.0 + b["value"] / 100.0
        return mult

    # ── 随机事件 ──

    def check_random_event(self) -> Optional[str]:
        s = self._state
        now = time.time()
        if now - s.last_event_check < s.event_check_interval:
            return None
        s.last_event_check = now
        today = time.localtime().tm_yday
        if (
            not hasattr(self, "_event_reset_date")
            or self._event_reset_date != today
        ):
            s.daily_event_count = 0
            self._event_reset_date = today
        if s.daily_event_count >= s.daily_event_limit:
            return None
        self.cleanup_expired_buffs()
        events = [
            (
                "drink_water",
                0.15,
                lambda: self.apply_event_recovery(
                    event_type="drink_water", base_amount=3.0
                ),
            ),
            (
                "rest",
                0.10,
                lambda: self.apply_event_recovery(
                    event_type="noon_rest",
                    base_amount=s.noon_event_recovery_bonus,
                ),
            ),
            (
                "stretch",
                0.12,
                lambda: self.apply_event_recovery(
                    event_type="stretch", base_amount=2.0
                ),
            ),
            (
                "snack",
                0.08,
                lambda: self.apply_event_recovery(
                    event_type="snack", base_amount=5.0
                ),
            ),
            (
                "overtime_work",
                0.05,
                lambda: self._apply_negative_event(cost=5.0, label="加班工作"),
            ),
            (
                "distracted",
                0.08,
                lambda: self._apply_negative_event(cost=3.0, label="分心走神"),
            ),
            (
                "phone_scroll",
                0.10,
                lambda: self._apply_negative_event(cost=4.0, label="刷手机"),
            ),
        ]
        for name, prob, handler in events:
            if random.random() < prob:
                try:
                    handler()
                    s.daily_event_count += 1
                    labels = {
                        "drink_water": "去喝了杯水",
                        "rest": "休息了一会儿",
                        "stretch": "伸了个懒腰",
                        "snack": "吃了点零食",
                        "overtime_work": "被拉去加班...",
                        "distracted": "走神了",
                        "phone_scroll": "刷手机刷太久了",
                    }
                    if random.random() < 0.3:
                        return labels.get(name, name)
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
                break
        return None

    def _apply_negative_event(self, *, cost: float, label: str) -> float:
        s = self._state
        s.energy_value = max(0.0, s.energy_value - cost * 0.3)
        s.daily_fatigue = min(s.daily_fatigue_limit, s.daily_fatigue + cost)
        s.cumulative_today_cost += cost
        return cost

    # ── 完整状态输出 ──

    def get_full_system_status(self) -> Dict[str, Any]:
        s = self._state
        mode = self.resolve_energy_mode()
        predictive = self.evaluate_predictive_wake()
        weakness = self.check_weakness_state()
        _energy_mode_dict = {
            "mode": mode,
            "blend": round(s.transition_blend_factor, 3),
            "segment_mod": round(s.time_segment_modifier, 3),
            "energy": round(s.energy_value, 3),
            "max": s.energy_max,
        }
        _sleep_map_dict = {
            "reserve": round(s.sleep_reserve, 3),
            "mapped": round(s.energy_from_last_sleep, 3),
        }
        return {
            "energy_mode": _energy_mode_dict,
            "sleep_map": _sleep_map_dict,
            "predictive_wake": predictive,
            "special_events": {
                "sensitive": f"{s.sensitive_topic_counter}/{s.sensitive_topic_threshold}",
                "weakness": weakness,
                "buffs": list(s.active_buffs.keys()),
                "events_today": f"{s.daily_event_count}/{s.daily_event_limit}",
            },
            "multifactor": {
                "think": round(s.think_active_value, 2),
                "cost_today": round(s.cumulative_today_cost, 2),
                "recovery_today": round(s.cumulative_today_recovery, 2),
                "event_pool": f"{round(s.event_recovery_pool, 1)}/{s.event_recovery_cap}",
            },
        }

    # ════════════════════════════════════════════════════════
    #  ★★★ ACFN级三维困倦引擎升级 ★★★
    #  耦合矩阵 / 导数感知 / 迭代收敛 / 滞后反馈 / 阈值漂移 / 随机微扰
    # ════════════════════════════════════════════════════════

    MAX_ITERATION_PASSES = 5
    CONVERGENCE_EPSILON = 0.003

    def _build_3d_coupling_template(self) -> List[List[float]]:
        """构建三维耦合矩阵模板(3×3)

        耦合语义:
          C[i][j] > 0 : 维度i高时，维度j在复合公式中的权重增强
          C[i][j] < 0 : 维度i高时，维度j在复合公式中的权重削弱
          C[i][i] = 0  : 自耦合为零（避免正反馈爆炸）

        设计原则：
          - 困意高(D) → 压力(P)的破坏力放大（困+压=致命组合）
          - 压力高(P) → 困意(D)的感受放大（压力让人更敏感地感知疲劳）
          - 压力高(P) → 储备(S)的保护力削弱（压力大时睡再多也没用）
          - 储备低(S) → 困意的威胁放大（没存货时一点困就崩溃）
          - 储备低(S) → 压力的威胁放大（没缓冲时压力直接打在脸上）
          - 困意高(D) → 储备消耗加速（困的时候储备掉得更快）
        """
        c = [[0.0] * DIM_3D for _ in range(DIM_3D)]
        c[IDX_DROWSINESS][IDX_PRESSURE] = 0.32
        c[IDX_DROWSINESS][IDX_RESERVE] = -0.18
        c[IDX_PRESSURE][IDX_DROWSINESS] = 0.28
        c[IDX_PRESSURE][IDX_RESERVE] = -0.35
        c[IDX_RESERVE][IDX_DROWSINESS] = 0.22
        c[IDX_RESERVE][IDX_PRESSURE] = 0.26
        for i in range(DIM_3D):
            c[i][i] = 0.0
        return c

    def _get_3d_coupling_matrix(
        self, features: List[float], derivatives: List[float]
    ) -> List[List[float]]:
        """获取动态调制的三维耦合矩阵

        与ACFN一致的动态调制策略：
          ① sigmoid门控：源因子超过激活阈值才施加耦合效应
          ② 目标饱和度衰减：目标因子已饱和时接收能力下降
          ③ 导数增幅：正在快速变化的因子其耦合效应被放大
        """
        s = self._state
        if not s.coupling_dirty_flag and s.coupling_matrix_3d is not None:
            return s.coupling_matrix_3d
        template = self._build_3d_coupling_template()
        result = [[0.0] * DIM_3D for _ in range(DIM_3D)]
        activation_snapshot = {}
        for i in range(DIM_3D):
            for j in range(DIM_3D):
                base_c = template[i][j]
                if abs(base_c) < 0.01:
                    continue
                src_act = _sigmoid(features[i], midpoint=0.35, steepness=6.0)
                tgt_sensitivity = 1.0 - _sigmoid(
                    features[j], midpoint=0.65, steepness=4.0
                )
                deriv_boost = 1.0 + abs(derivatives[i]) * 0.25
                dynamic_c = base_c * src_act * tgt_sensitivity * deriv_boost
                result[i][j] = dynamic_c
                pair_key = f"{_DIM_NAMES[i]}→{_DIM_NAMES[j]}"
                activation_snapshot[pair_key] = round(dynamic_c, 4)
        s.coupling_matrix_3d = result
        s.coupling_dirty_flag = False
        s.coupling_activation_snapshot = activation_snapshot
        return result

    def _compute_3d_derivatives(
        self,
        raw_d: float,
        raw_p: float,
        raw_s: float,
        now: float,
    ) -> List[float]:
        """计算三维度时序导数(dF/dt) —— 变化率感知层

        返回归一化的变化率向量 [dD/dt, dP/dt, dS/dt]:
          正值 → 维度在增长（困意在上升/压力在增加/储备在恢复）
          负值 → 维度在下降
          绝对值大 → 变化剧烈，应在复合公式中获得更高权重

        导数用途：
          1. 耦合矩阵调制：快速变化的因子耦合效应更强
          2. 复合负载修正："正在加速恶化"比"已经很差但稳定"更危险
          3. 置信度评估：变化剧烈时置信度降低（预测不确定性增大）
        """
        s = self._state
        dt = now - s.derivative_timestamp
        if dt < 1.0 or s.derivative_timestamp <= 0:
            s.last_d_value = raw_d
            s.last_p_value = raw_p
            s.last_s_value = raw_s
            s.derivative_timestamp = now
            s.drowsiness_derivative = 0.0
            s.pressure_derivative = 0.0
            s.reserve_derivative = 0.0
            s.derivative_snapshot = {"dDdt": 0.0, "dPdt": 0.0, "dSdt": 0.0}
            return [0.0, 0.0, 0.0]
        decay_factor = math.exp(-dt / s.derivative_decay_sec)
        d_d = (raw_d - s.last_d_value) / max(dt, 0.1)
        d_p = (raw_p - s.last_p_value) / max(dt, 0.1)
        d_s = (raw_s - s.last_s_value) / max(dt, 0.1)
        smooth_d = s.drowsiness_derivative * decay_factor + d_d * (
            1.0 - decay_factor
        )
        smooth_p = s.pressure_derivative * decay_factor + d_p * (
            1.0 - decay_factor
        )
        smooth_s = s.reserve_derivative * decay_factor + d_s * (
            1.0 - decay_factor
        )
        d_norm = 50.0
        p_norm = 30.0
        s_norm = 20.0
        norm_d = _clamp(smooth_d / d_norm, -1.0, 1.0)
        norm_p = _clamp(smooth_p / p_norm, -1.0, 1.0)
        norm_s = _clamp(smooth_s / s_norm, -1.0, 1.0)
        s.last_d_value = raw_d
        s.last_p_value = raw_p
        s.last_s_value = raw_s
        s.derivative_timestamp = now
        s.drowsiness_derivative = smooth_d
        s.pressure_derivative = smooth_p
        s.reserve_derivative = smooth_s
        s.derivative_snapshot = {
            "dDdt": round(norm_d, 4),
            "dPdt": round(norm_p, 4),
            "dSdt": round(norm_s, 4),
        }
        return [norm_d, norm_p, norm_s]

    def _iterative_composite_refinement(
        self,
        raw_d: float,
        raw_p: float,
        raw_s: float,
        quality: float,
        features: List[float],
        derivatives: List[float],
        coupling: List[List[float]],
        hour: int,
    ) -> Tuple[float, int, float, float]:
        """多轮迭代精炼引擎 —— ACFN核心

        每轮执行：
          1. 用耦合矩阵修正各维度的有效权重
          2. 计算该轮的composite_load
          3. 导数参与修正（加速恶化额外加权）
          4. 收敛阻尼：后续轮次修正幅度递减

        收敛条件（任一满足即停止）：
          - delta_bar < CONVERGENCE_EPSILON (0.003)
          - delta_score < CONVERGENCE_EPSILON * 0.5
          - 达到最大轮次 MAX_ITERATION_PASSES (5)

        返回: (final_composite, passes_used, final_delta_bar, final_delta_score)
        """
        s = self._state
        prev_composite = 0.0
        verdict_composite = 0.0
        for pass_num in range(1, self.MAX_ITERATION_PASSES + 1):
            convergence_damp = 1.0 / (1.0 + pass_num * 0.35)
            d_eff = self._coupled_dimension_value(
                IDX_DROWSINESS,
                raw_d,
                features,
                derivatives,
                coupling,
                convergence_damp,
            )
            p_eff = self._coupled_dimension_value(
                IDX_PRESSURE,
                raw_p,
                features,
                derivatives,
                coupling,
                convergence_damp,
            )
            s_eff = self._coupled_dimension_value(
                IDX_RESERVE,
                raw_s,
                features,
                derivatives,
                coupling,
                convergence_damp,
            )
            derivative_penalty = 0.0
            if derivatives[IDX_DROWSINESS] > 0.15:
                derivative_penalty += (
                    derivatives[IDX_DROWSINESS] * 0.12 * convergence_damp
                )
            if derivatives[IDX_PRESSURE] > 0.12:
                derivative_penalty += (
                    derivatives[IDX_PRESSURE] * 0.10 * convergence_damp
                )
            if derivatives[IDX_RESERVE] < -0.10:
                derivative_penalty += (
                    abs(derivatives[IDX_RESERVE]) * 0.08 * convergence_damp
                )
            d_active = _sigmoid(d_eff / 100.0, midpoint=0.30, steepness=7.0)
            p_active = _sigmoid(p_eff / 100.0, midpoint=0.25, steepness=6.5)
            s_protect = _sigmoid(s_eff / 100.0, midpoint=0.20, steepness=5.0)
            epsilon = 1e-6
            numerator = (d_active * 100.0) * (p_active * 100.0)
            denominator = math.sqrt(
                max(epsilon, (s_protect * 100.0) * quality)
            )
            raw_composite = numerator / denominator
            derivative_boost = 1.0 + derivative_penalty
            pass_composite = raw_composite * derivative_boost / 100.0
            pass_composite = max(0.0, pass_composite)
            if pass_num > 1:
                blend_t = 0.55 + 0.12 * pass_num
                pass_composite = _lerp(prev_composite, pass_composite, blend_t)
            delta_bar = abs(pass_composite - prev_composite)
            delta_score = (
                abs((pass_composite / max(0.001, prev_composite)) - 1.0)
                if prev_composite > 0.001
                else 0.0
            )
            prev_composite = pass_composite
            verdict_composite = pass_composite
            s.iteration_passes_last = pass_num
            s.convergence_achieved = False
            s.last_convergence_delta_bar = delta_bar
            s.last_convergence_delta_score = delta_score
            if (
                delta_bar < self.CONVERGENCE_EPSILON
                and delta_score < self.CONVERGENCE_EPSILON * 0.5
            ):
                s.convergence_achieved = True
                break
        return (
            verdict_composite,
            s.iteration_passes_last,
            s.last_convergence_delta_bar,
            s.last_convergence_delta_score,
        )

    def _coupled_dimension_value(
        self,
        dim_idx: int,
        raw_value: float,
        features: List[float],
        derivatives: List[float],
        coupling: List[List[float]],
        damp: float,
    ) -> float:
        """计算单个维度的耦合后有效值

        effective_value = raw_value × (1 + tanh(Σ(C[j][dim_idx])))

        即原始值乘以一个由所有其他维度通过耦合矩阵施加的调制因子。
        tanh保证调制因子在合理范围内（约[-1,+1]映射到[0,2]倍区间）。
        """
        coupled_sum = 0.0
        cm = coupling
        for j in range(DIM_3D):
            c_val = cm[j][dim_idx]
            if abs(c_val) < 0.005:
                continue
            coupled_sum += c_val
        modulation = 1.0 + _tanh_scaled(coupled_sum, scale=1.5)
        return raw_value * modulation * damp

    def _apply_hysteresis_correction(self, composite: float) -> float:
        """滞后反馈回路修正 —— 防止复合负载输出震荡

        机制：
          追踪最近N次复合负载输出，如果连续多次同向偏高/偏低，
          则施加反向微调补偿。

        触发条件：
          - 总评估次数 >= 6
          - 单向比例 > 72%（正向或负向占绝对多数）

        补偿强度：
          - 与超出比例成正比（越偏激补偿越大）
          - 最大补偿幅度 ±8%（防止过度校正）
        """
        s = self._state
        s.total_composite_evaluations += 1
        ref_threshold = self._compute_collapse_threshold()
        is_high = composite > ref_threshold * 0.6
        if is_high:
            s.positive_composite_count += 1
        else:
            s.negative_composite_count += 1
        s.composite_output_history.append(composite)
        if len(s.composite_output_history) > s.composite_history_max_len:
            s.composite_output_history.pop(0)
        total = s.total_composite_evaluations
        if total < 6:
            s.hysteresis_correction_active = False
            return composite
        pos_ratio = s.positive_composite_count / max(total, 1)
        neg_ratio = s.negative_composite_count / max(total, 1)
        correction = 0.0
        if pos_ratio > 0.72:
            excess = pos_ratio - 0.72
            correction = -excess * 0.06 * composite
            s.last_hysteresis_direction = -1
            s.hysteresis_correction_active = True
        elif neg_ratio > 0.72:
            excess = neg_ratio - 0.72
            correction = excess * 0.04 * max(ref_threshold * 0.3, composite)
            s.last_hysteresis_direction = 1
            s.hysteresis_correction_active = True
        else:
            s.hysteresis_correction_active = False
            s.last_hysteresis_direction = 0
        correction = _clamp(correction, -composite * 0.08, composite * 0.08)
        return max(0.0, composite + correction)

    def _drift_adaptive_threshold(
        self, base_threshold: float, hour: int
    ) -> float:
        """自适应阈值漂移 —— 崩溃阈值根据历史决策缓慢移动

        机制：
          每隔 drift_evaluation_interval(12) 次评估后，
          根据近期复合负载的平均方向缓慢漂移基础阈值。

        漂移规则：
          - 近期复合负载持续偏高 → 提高阈值（身体适应了高压）
          - 近期复合负载持续偏低 → 降低阈值（变得更敏感）
          - 漂移速度 drift_speed = 0.006（非常慢，避免震荡）
          - 最大漂移幅度 drift_max_amplitude = ±12

        这模拟了生物体的耐受性适应：
          经常熬夜的人对疲劳的耐受阈值会缓慢提高，
          但同时也意味着他们更容易突然崩溃（阈值虚高）。
        """
        s = self._state
        s.total_composite_evaluations += 1
        eval_interval = s.drift_evaluation_interval
        if s.total_composite_evaluations % eval_interval != 0:
            drifted = base_threshold + s.threshold_drift_accumulator
            return max(15.0, min(120.0, drifted))
        history = s.composite_output_history
        if len(history) < 4:
            return base_threshold + s.threshold_drift_accumulator
        recent_avg = sum(history[-min(len(history), 8):]) / min(
            len(history), 8
        )
        normalized_recent = recent_avg / max(1.0, base_threshold)
        if normalized_recent > 0.75:
            drift_direction = 1.0
            drift_magnitude = (
                (normalized_recent - 0.75) * s.drift_speed * base_threshold
            )
        elif normalized_recent < 0.35:
            drift_direction = -1.0
            drift_magnitude = (
                (0.35 - normalized_recent)
                * s.drift_speed
                * base_threshold
                * 0.7
            )
        else:
            drift_direction = 0.0
            drift_magnitude = 0.0
        drift_magnitude = _clamp(
            drift_magnitude,
            -s.drift_max_amplitude * 0.15,
            s.drift_max_amplitude * 0.15,
        )
        new_accumulator = (
            s.threshold_drift_accumulator + drift_direction * drift_magnitude
        )
        new_accumulator = _clamp(
            new_accumulator, -s.drift_max_amplitude, s.drift_max_amplitude
        )
        s.threshold_drift_accumulator = new_accumulator
        s.last_drift_time = time.time()
        drifted = base_threshold + new_accumulator
        return max(15.0, min(120.0, drifted))

    def _apply_confidence_perturbation(
        self, composite: float, confidence: float
    ) -> float:
        """置信度校准的随机微扰层

        与ACFN一致的微扰策略：
          - 噪声规模与不确定性的平方成正比（低置信度=高噪声）
          - 噪声上限固定为 ±0.015（防止数值爆炸）
          - 高置信度时噪声趋近于零（结果可信，不需要扰动）

        公式：
          perturbation = gaussian_noise(0, uncertainty² × 0.012)
          uncertainty = 1 - confidence

        目的：
          ① 低置信度区域引入受控随机性（打破死循环/局部最优）
          ② 高置信度区域保持稳定输出（结果已经可靠）
          ③ 整体系统具有非确定性特征（更像生物体而非纯数学模型）
        """
        s = self._state
        uncertainty = 1.0 - _clamp01(confidence)
        s.perturbation_entropy = uncertainty
        noise_scale = uncertainty * uncertainty * 0.012
        noise_scale = _clamp(noise_scale, 0.0, 0.015)
        perturbation = _gaussian_noise(mean=0.0, stddev=noise_scale)
        s.perturbation_applied = perturbation
        result = composite + perturbation * max(0.5, composite)
        return max(0.0, result)

    def _compute_acfn_confidence(
        self,
        passes: int,
        delta_bar: float,
        delta_score: float,
        composite_ratio: float,
    ) -> float:
        """ACFN级置信度评估

        置信度由以下因素综合决定：
          ① 方向一致性收敛：迭代轮次少且delta小 → 高置信度
          ② 边界距离：composite远离0和极端值 → 高置信度（不处于边界模糊区）
          ③ 耦合强度：耦合激活度高 → 因子间相互印证 → 高置信度
          ④ 导数稳定性：导数接近零 → 系统平稳 → 高置信度

        返回值范围 [0.05, 1.0]
        """
        s = self._state
        convergence_score = 1.0 - _clamp(delta_bar / 0.05, 0.0, 1.0)
        passes_bonus = 0.0
        if passes <= 2:
            passes_bonus = 0.12
        elif passes == 3:
            passes_bonus = 0.06
        boundary_penalty = 0.0
        if composite_ratio < 0.15:
            boundary_penalty = 0.15
        elif composite_ratio > 1.8:
            boundary_penalty = 0.10
        coupling_sum = 0.0
        snap = s.coupling_activation_snapshot
        if snap:
            coupling_sum = sum(abs(v) for v in snap.values())
        coupling_score = _clamp(coupling_sum / 1.5, 0.0, 1.0)
        deriv_mag = (
            abs(s.drowsiness_derivative)
            + abs(s.pressure_derivative)
            + abs(s.reserve_derivative)
        )
        stability_score = 1.0 - _clamp(deriv_mag / 2.0, 0.0, 0.35)
        raw_confidence = (
            convergence_score * 0.40
            + (convergence_score + passes_bonus) * 0.15
            + (1.0 - boundary_penalty) * 0.15
            + coupling_score * 0.15
            + stability_score * 0.15
        )
        s.composite_confidence = _clamp(raw_confidence, 0.05, 1.0)
        return s.composite_confidence


def get_night_cycle(channel_id: str) -> NightCycleSystem:
    if channel_id not in _night_system_instances:
        _night_system_instances[channel_id] = NightCycleSystem(channel_id)
    return _night_system_instances[channel_id]


def remove_night_cycle(channel_id: str) -> None:
    _night_system_instances.pop(channel_id, None)

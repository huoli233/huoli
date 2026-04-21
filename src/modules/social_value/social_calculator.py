import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional
from src.common.config.config_engine import (
    ConfigEngine,
    get_default_config_engine,
)
from src.common.config.config_engine import AlgorithmParams


@dataclass
class _StreakState:
    """行为连续性追踪状态"""

    consecutive_positive: int = 0
    consecutive_negative: int = 0
    threshold_offset: float = 0.0
    last_update_ts: float = field(default_factory=time.time)
    recent_intensities: Deque[float] = field(
        default_factory=lambda: deque(maxlen=30)
    )


class SocialCalculator:
    """社交值计算器 - 通过四层映射获取参数，使用统一算法公式

    完整流程:
    1. 四层映射: behavior_type + intent → AlgorithmParams
    2. 基础分 × 意图倍率 × 严重程度 × 类型加成
    3. 渐进限制: 每次最多变化 max_change
    4. 阈值加速（含连续性偏移）: 达到动态阈值后变化加速
    5. 累积平衡: 心理状态调节变化效率
    """

    def __init__(self, config_engine: ConfigEngine | None = None):
        self._config = config_engine or get_default_config_engine()
        self._streak_states: Dict[str, _StreakState] = {}
        self._offset_decay_per_hour: float = 0.15
        self._positive_streak_coeff: float = 1.5
        self._negative_streak_coeff: float = 2.0
        self._min_change_threshold: float = 0.001
        self._min_elapsed_hours: float = 0.01
        self._load_config()

    def _load_config(self):
        """从配置加载所有阈值"""
        if not self._config:
            raise RuntimeError("配置引擎未设置，无法加载社交计算器阈值")
        social_cfg = self._config.get_social_config()
        self._offset_decay_per_hour = float(
            social_cfg.get("offset_decay_per_hour", 0.15)
        )
        self._positive_streak_coeff = float(
            social_cfg.get("positive_streak_coeff", 1.5)
        )
        self._negative_streak_coeff = float(
            social_cfg.get("negative_streak_coeff", 2.0)
        )
        self._min_change_threshold = float(
            social_cfg.get("min_change_threshold", 0.001)
        )
        self._min_elapsed_hours = float(
            social_cfg.get("min_elapsed_hours", 0.01)
        )

    def get_category_params(self, behavior_type: str, intent: str):
        """获取行为类型和意图对应的分类参数（公开接口）"""
        if not self._config:
            return AlgorithmParams()
        return self._config.get_params(behavior_type, intent)

    def record_outcome(self, uid: str, delta: float, intensity: float) -> None:
        """记录一次计算结果，更新行为连续性追踪

        Args:
            uid: 用户标识
            delta: 本次社交值变化量（正值=积极，负值=消极）
            intensity: 本次行为强度 (0~1)
        """
        st = self._streak_states.get(uid)
        if st is None:
            st = _StreakState()
            self._streak_states[uid] = st
        self._decay_offset(st)
        st.recent_intensities.append(abs(intensity))
        if delta > 0:
            st.consecutive_positive += 1
            st.consecutive_negative = 0
            streak_bonus = self._streak_bonus(st.consecutive_positive)
            st.threshold_offset -= (
                self._positive_streak_coeff * intensity * (1.0 + streak_bonus)
            )
        elif delta < 0:
            st.consecutive_negative += 1
            st.consecutive_positive = 0
            streak_bonus = self._streak_bonus(st.consecutive_negative)
            st.threshold_offset += (
                self._negative_streak_coeff * intensity * (1.0 + streak_bonus)
            )
        else:
            st.consecutive_positive = 0
            st.consecutive_negative = 0
        st.threshold_offset = max(-30.0, min(30.0, st.threshold_offset))
        st.last_update_ts = time.time()

    def get_streak_info(self, uid: str) -> Dict:
        """获取连续性追踪信息（调试用）"""
        st = self._streak_states.get(uid)
        if st is None:
            return {"positive_streak": 0, "negative_streak": 0, "offset": 0.0}
        self._decay_offset(st)
        return {
            "positive_streak": st.consecutive_positive,
            "negative_streak": st.consecutive_negative,
            "offset": round(st.threshold_offset, 2),
        }

    def calculate(
        self,
        behavior_type: str,
        severity: float,
        intent: str,
        current_value: float,
        psychological_pressure: float = 0.0,
        training_resistance: float = 0.0,
        uid: str = "",
    ) -> float:
        """计算社交值变化量

        完整流程:
        1. 四层映射: behavior_type + intent → AlgorithmParams
        2. 基础分 × 意图倍率 × 严重程度 × 类型加成
        3. 渐进限制: 每次最多变化 max_change
        4. 阈值加速（含连续性偏移）: 达到动态阈值后变化加速
        5. 累积平衡: 心理状态调节变化效率

        Args:
            behavior_type: 模型识别的行为类型 (如 "genuine_praise", "direct_insult")
            severity: 模型识别的严重程度 (0.0-1.0)
            intent: 模型识别的意图 (如 "compliment", "insult")
            current_value: 当前社交值 (-100 ~ +100)
            psychological_pressure: 心理压力 (0~100)，来自 TraumaSystem.inner_chaos × 10
            training_resistance: 交互抵抗 (0~120)，交互次数越多变化越小
            uid: 用户标识（用于连续性追踪，空字符串则跳过）

        Returns:
            float: 社交值变化量
        """
        params = self._config.get_params(behavior_type, intent)
        adjusted_score = self._compute_adjusted_score(params, severity)
        progressive_score = self._apply_progressive_limit(
            adjusted_score, params, current_value
        )
        accelerated_score = self._apply_threshold_acceleration(
            progressive_score, params, current_value, uid
        )
        final_score = self._apply_accumulation_efficiency(
            accelerated_score,
            psychological_pressure,
            training_resistance,
            current_value,
        )
        if uid:
            self.record_outcome(uid, final_score, severity)
        return final_score

    def calculate_with_details(
        self,
        behavior_type: str,
        severity: float,
        intent: str,
        current_value: float,
        psychological_pressure: float = 0.0,
        training_resistance: float = 0.0,
        uid: str = "",
    ) -> Dict:
        """计算并返回详细的中间过程（调试用）"""
        params = self._config.get_params(behavior_type, intent)
        adjusted_score = self._compute_adjusted_score(params, severity)
        max_change = params.max_change_per_reply * abs(current_value + 100)
        progressive_score = max(-max_change, min(max_change, adjusted_score))
        dynamic_threshold = self._dynamic_threshold(params, uid)
        threshold_triggered = abs(current_value) > dynamic_threshold
        accelerated_score = progressive_score
        if threshold_triggered:
            accelerated_score = (
                progressive_score * params.acceleration_multiplier
            )
        efficiency_score = self._apply_accumulation_efficiency(
            accelerated_score,
            psychological_pressure,
            training_resistance,
            current_value,
        )
        if uid:
            self.record_outcome(uid, efficiency_score, severity)
        streak_info = self.get_streak_info(uid) if uid else {}
        return {
            "behavior_type": behavior_type,
            "intent": intent,
            "severity": severity,
            "current_value": current_value,
            "category": params.category,
            "base_score": params.base_score,
            "intent_multiplier": params.intent_multiplier,
            "type_bonus": params.type_bonus,
            "adjusted_score": adjusted_score,
            "max_change": max_change,
            "progressive_score": progressive_score,
            "dynamic_threshold": dynamic_threshold,
            "threshold_triggered": threshold_triggered,
            "acceleration_multiplier": params.acceleration_multiplier,
            "psychological_pressure": psychological_pressure,
            "training_resistance": training_resistance,
            "streak_info": streak_info,
            "accumulation_efficiency_applied": efficiency_score
            != accelerated_score,
            "final_score": efficiency_score,
            "new_value": max(
                -100.0, min(100.0, current_value + efficiency_score)
            ),
        }

    def _compute_adjusted_score(
        self, params: AlgorithmParams, severity: float
    ) -> float:
        """Step 1-2: 计算调整后的基础分

        公式: base_score × intent_multiplier × severity × type_bonus
        """
        return (
            params.base_score
            * params.intent_multiplier
            * severity
            * params.type_bonus
        )

    def _apply_progressive_limit(
        self,
        adjusted_score: float,
        params: AlgorithmParams,
        current_value: float,
    ) -> float:
        """Step 3: 渐进限制

        公式: max_change = max_change_per_reply × |current_value + 100|
        限制: clamp(adjusted_score, -max_change, max_change)

        效果: 社交值越极端，每次允许的变化量越大（加速恶化/加速好转）
              社交值接近 -100 时 max_change 接近 0（减速接近极限）
        """
        max_change = params.max_change_per_reply * abs(current_value + 100)
        if max_change < self._min_change_threshold:
            max_change = self._min_change_threshold
        return max(-max_change, min(max_change, adjusted_score))

    def _apply_threshold_acceleration(
        self,
        progressive_score: float,
        params: AlgorithmParams,
        current_value: float,
        uid: str = "",
    ) -> float:
        """Step 4: 阈值加速（含连续性偏移）

        基础阈值 = acceleration_threshold × 100
        动态偏移 = 由行为连续性追踪计算的 threshold_offset
        有效阈值 = clamp(基础阈值 + 偏移, 20, 95)

        当 |current_value| 超过有效阈值时，变化量乘以 acceleration_multiplier

        效果:
            - 连续积极互动 → 偏移为负 → 阈值降低 → 更容易触发加速巩固
            - 连续消极互动 → 偏移为正 → 阈值升高 → 恶化需要更极端才加速
            - 无连续性数据 → 退化为静态阈值
        """
        dynamic_threshold = self._dynamic_threshold(params, uid)
        if abs(current_value) > dynamic_threshold:
            return progressive_score * params.acceleration_multiplier
        return progressive_score

    def _dynamic_threshold(self, params: AlgorithmParams, uid: str) -> float:
        """计算考虑连续性偏移后的动态阈值"""
        base_threshold = params.acceleration_threshold * 100.0
        st = self._streak_states.get(uid) if uid else None
        if st is None:
            return base_threshold
        self._decay_offset(st)
        return max(20.0, min(95.0, base_threshold + st.threshold_offset))

    @staticmethod
    def _streak_bonus(consecutive_count: int) -> float:
        """连续次数转换为加成系数（对数增长，避免线性爆炸）

        bonus = min(0.6, ln(1 + count) × 0.25)
        """
        return min(0.6, math.log(1.0 + consecutive_count) * 0.25)

    def _decay_offset(self, st: _StreakState) -> None:
        """时间衰减连续性偏移（每小时衰减配置的百分比）"""
        now = time.time()
        elapsed_hours = (now - st.last_update_ts) / 3600.0
        if elapsed_hours > self._min_elapsed_hours:
            decay_factor = max(
                0.0, 1.0 - self._offset_decay_per_hour * elapsed_hours
            )
            st.threshold_offset *= decay_factor
            st.last_update_ts = now

    def _apply_accumulation_efficiency(
        self,
        score: float,
        psychological_pressure: float,
        training_resistance: float,
        current_value: float,
    ) -> float:
        """Step 5: 累积平衡 — 心理状态调节社交值变化效率

        公式: efficiency = base × balance_coeff × resist_factor × stress_factor

        base = 1.0
        balance_coeff: 当前社交值越接近中性(0)，变化效率越高；
                       越极端则变化逐渐困难（极化阻力）
                       = 0.7 + 0.3 × (1 - abs(current_value) / 100)
        resist_factor: 交互次数越多，用户"脱敏"越严重
                       = 1.0 - min(training_resistance, 120) / 120
        stress_factor: 心理压力越高，社交感知能力下降
                       = 1.0 - (psychological_pressure / 100) × 0.3

        所有因子 clamp 到 [0.1, 1.0] 避免归零或反转。
        """
        polarization_ratio = abs(current_value) / 100.0
        balance_coeff = 0.7 + 0.3 * (1.0 - polarization_ratio)
        capped_resistance = min(max(training_resistance, 0.0), 120.0)
        resist_factor = 1.0 - capped_resistance / 120.0
        capped_pressure = min(max(psychological_pressure, 0.0), 100.0)
        stress_factor = 1.0 - (capped_pressure / 100.0) * 0.3
        efficiency = (
            balance_coeff * max(0.1, resist_factor) * max(0.1, stress_factor)
        )
        return score * max(0.1, min(1.0, efficiency))

    def get_decay_amount(self, behavior_type: str, time_hours: float) -> float:
        """计算衰减量 - 长时间无互动时社交值向 0 衰减

        公式: decay = decay_rate × time_hours × |current_offset|
        """
        params = self._config.get_params(behavior_type)
        return params.decay_rate * time_hours

    def get_trauma_impact(self, behavior_type: str, intent: str) -> float:
        """获取行为对创伤系统的影响系数"""
        params = self._config.get_params(behavior_type, intent)
        return params.trauma_impact

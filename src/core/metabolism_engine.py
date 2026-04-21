import time
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger("代谢引擎")

_metabolism_instances: Dict[str, "MetabolismEngine"] = {}


class BehaviorCostType(Enum):
    """行为消耗类型"""

    REPLY = "reply"
    THINK = "think"
    PEEK = "peek"
    IDLE = "idle"
    SOCIAL_INTERACT = "social_interact"
    EMOTIONAL_RESPOND = "emotional_respond"
    LONG_CONVERSATION = "long_conversation"
    MEDIA_ANALYZE = "media_analyze"
    GROUP_OBSERVE = "group_observe"
    PATTERN_ANALYZE = "pattern_analyze"
    IMPRESSION_UPDATE = "impression_update"
    SCENE_ADAPT = "scene_adapt"


@dataclass
class RecoveryProfile:
    """恢复曲线配置"""

    base_rate: float = 0.5
    quiet_multiplier: float = 1.8
    social_boost: float = 0.3
    boredom_drain: float = -0.1
    night_multiplier: float = 0.4
    ceiling: float = 100.0
    floor: float = 0.0
    group_activity_boost: float = 0.2
    scene_complexity_penalty: float = -0.15
    impression_evolution_cost: float = 0.8


@dataclass
class MetabolismDelta:
    """单次代谢变化量"""

    chat_cost: float = 0.0
    thinking_cost: float = 0.0
    activity_delta: float = 0.0
    social_delta: float = 0.0
    boredom_delta: float = 0.0
    loafing_delta: float = 0.0
    group_observe_cost: float = 0.0
    pattern_analyze_cost: float = 0.0
    impression_update_cost: float = 0.0
    scene_adapt_cost: float = 0.0
    source_behavior: str = ""
    timestamp: float = field(default_factory=time.time)

    def total_cost(self) -> float:
        return (
            self.chat_cost
            + self.thinking_cost
            + self.group_observe_cost
            + self.pattern_analyze_cost
            + self.impression_update_cost
            + self.scene_adapt_cost
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chat_cost": round(self.chat_cost, 3),
            "thinking_cost": round(self.thinking_cost, 3),
            "activity_delta": round(self.activity_delta, 3),
            "social_delta": round(self.social_delta, 3),
            "boredom_delta": round(self.boredom_delta, 3),
            "loafing_delta": round(self.loafing_delta, 3),
            "group_observe_cost": round(self.group_observe_cost, 3),
            "pattern_analyze_cost": round(self.pattern_analyze_cost, 3),
            "impression_update_cost": round(self.impression_update_cost, 3),
            "scene_adapt_cost": round(self.scene_adapt_cost, 3),
            "source_behavior": self.source_behavior,
        }


@dataclass
class MetabolismState:
    """代谢层的完整资源状态"""

    chat_fuel: float = 100.0
    thinking_fuel: float = 100.0
    activity_gauge: float = 50.0
    social_gauge: float = 0.0
    boredom_level: float = 0.0
    loafing_level: float = 0.0
    fatigue_accumulator: float = 0.0
    consecutive_active_minutes: float = 0.0
    last_action_time: float = field(default_factory=time.time)
    last_recovery_time: float = field(default_factory=time.time)
    total_consumed_today: float = 0.0
    group_observation_fuel: float = 80.0
    pattern_analysis_fuel: float = 70.0
    impression_evolution_fuel: float = 90.0
    scene_adaptation_energy: float = 60.0
    circadian_rhythm_phase: float = 0.0
    social_context_sensitivity: float = 0.5
    emotional_resonance_level: float = 0.3

    def chat_ratio(self) -> float:
        return max(0.0, min(1.0, self.chat_fuel / 100.0))

    def thinking_ratio(self) -> float:
        return max(0.0, min(1.0, self.thinking_fuel / 100.0))

    def is_exhausted(self) -> bool:
        return self.chat_fuel < 5.0 and self.thinking_fuel < 5.0

    def fatigue_factor(self) -> float:
        """疲劳放大因子：连续活跃越久消耗越快"""
        minutes = self.consecutive_active_minutes
        if minutes < 10:
            return 1.0
        elif minutes < 30:
            return 1.0 + (minutes - 10) * 0.015
        elif minutes < 60:
            return 1.3 + (minutes - 30) * 0.025
        return min(2.5, 2.05 + (minutes - 60) * 0.01)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chat_fuel": round(self.chat_fuel, 2),
            "thinking_fuel": round(self.thinking_fuel, 2),
            "activity_gauge": round(self.activity_gauge, 2),
            "social_gauge": round(self.social_gauge, 2),
            "boredom_level": round(self.boredom_level, 2),
            "loafing_level": round(self.loafing_level, 2),
            "fatigue_accumulator": round(self.fatigue_accumulator, 2),
            "consecutive_active_minutes": round(
                self.consecutive_active_minutes, 1
            ),
            "total_consumed_today": round(self.total_consumed_today, 2),
            "group_observation_fuel": round(self.group_observation_fuel, 2),
            "pattern_analysis_fuel": round(self.pattern_analysis_fuel, 2),
            "impression_evolution_fuel": round(
                self.impression_evolution_fuel, 2
            ),
            "scene_adaptation_energy": round(self.scene_adaptation_energy, 2),
            "circadian_rhythm_phase": round(self.circadian_rhythm_phase, 2),
            "social_context_sensitivity": round(
                self.social_context_sensitivity, 2
            ),
            "emotional_resonance_level": round(
                self.emotional_resonance_level, 2
            ),
        }


# 行为消耗基表
_BEHAVIOR_COST_TABLE: Dict[str, Dict[str, float]] = {
    BehaviorCostType.REPLY.value: {
        "chat": 2.5,
        "thinking": 1.5,
        "activity": 1.0,
        "boredom": -0.8,
        "loafing": -0.5,
        "group_observe": 0.0,
        "pattern_analyze": 0.0,
        "impression_update": 0.0,
        "scene_adapt": 0.0,
    },
    BehaviorCostType.THINK.value: {
        "chat": 0.5,
        "thinking": 3.0,
        "activity": 0.3,
        "boredom": -0.3,
        "loafing": -0.2,
        "group_observe": 0.0,
        "pattern_analyze": 0.0,
        "impression_update": 0.0,
        "scene_adapt": 0.0,
    },
    BehaviorCostType.PEEK.value: {
        "chat": 0.3,
        "thinking": 0.2,
        "activity": 0.1,
        "boredom": -0.1,
        "loafing": 0.15,
        "group_observe": 0.0,
        "pattern_analyze": 0.0,
        "impression_update": 0.0,
        "scene_adapt": 0.0,
    },
    BehaviorCostType.IDLE.value: {
        "chat": 0.0,
        "thinking": 0.0,
        "activity": -0.5,
        "boredom": 0.4,
        "loafing": 0.3,
        "group_observe": 0.0,
        "pattern_analyze": 0.0,
        "impression_update": 0.0,
        "scene_adapt": 0.0,
    },
    BehaviorCostType.SOCIAL_INTERACT.value: {
        "chat": 1.8,
        "thinking": 0.8,
        "activity": 1.5,
        "boredom": -1.0,
        "loafing": -0.8,
        "group_observe": 0.0,
        "pattern_analyze": 0.0,
        "impression_update": 0.0,
        "scene_adapt": 0.0,
    },
    BehaviorCostType.EMOTIONAL_RESPOND.value: {
        "chat": 2.0,
        "thinking": 2.5,
        "activity": 0.5,
        "boredom": -0.5,
        "loafing": -0.3,
        "group_observe": 0.0,
        "pattern_analyze": 0.0,
        "impression_update": 0.0,
        "scene_adapt": 0.0,
    },
    BehaviorCostType.LONG_CONVERSATION.value: {
        "chat": 3.5,
        "thinking": 2.0,
        "activity": 1.5,
        "boredom": -1.2,
        "loafing": -1.0,
        "group_observe": 0.0,
        "pattern_analyze": 0.0,
        "impression_update": 0.0,
        "scene_adapt": 0.0,
    },
    BehaviorCostType.MEDIA_ANALYZE.value: {
        "chat": 0.8,
        "thinking": 4.0,
        "activity": 0.2,
        "boredom": -0.2,
        "loafing": -0.1,
        "group_observe": 0.0,
        "pattern_analyze": 0.0,
        "impression_update": 0.0,
        "scene_adapt": 0.0,
    },
    BehaviorCostType.GROUP_OBSERVE.value: {
        "chat": 0.2,
        "thinking": 1.2,
        "activity": 0.8,
        "boredom": -0.4,
        "loafing": -0.3,
        "group_observe": 2.0,
        "pattern_analyze": 0.0,
        "impression_update": 0.0,
        "scene_adapt": 0.0,
    },
    BehaviorCostType.PATTERN_ANALYZE.value: {
        "chat": 0.1,
        "thinking": 2.5,
        "activity": 0.3,
        "boredom": -0.2,
        "loafing": -0.1,
        "group_observe": 0.0,
        "pattern_analyze": 3.0,
        "impression_update": 0.0,
        "scene_adapt": 0.0,
    },
    BehaviorCostType.IMPRESSION_UPDATE.value: {
        "chat": 0.3,
        "thinking": 1.8,
        "activity": 0.5,
        "boredom": -0.3,
        "loafing": -0.2,
        "group_observe": 0.0,
        "pattern_analyze": 0.0,
        "impression_update": 2.5,
        "scene_adapt": 0.0,
    },
    BehaviorCostType.SCENE_ADAPT.value: {
        "chat": 0.4,
        "thinking": 1.5,
        "activity": 1.0,
        "boredom": -0.5,
        "loafing": -0.4,
        "group_observe": 0.0,
        "pattern_analyze": 0.0,
        "impression_update": 0.0,
        "scene_adapt": 2.2,
    },
}


class MetabolismEngine:
    """统一资源代谢引擎
    将聊天值、思考值、活跃度、社交值、无聊值、摸鱼值
    纳入统一代谢框架，通过行为类型和疲劳因子驱动消耗与恢复。
    第三阶段升级：新增群观察、模式分析、印象演化、场景适应等多维度资源管理，
    引入昼夜节律、社会语境敏感度、情感共鸣水平等动态调节因子。
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._state = MetabolismState()
        self._recovery_profile = RecoveryProfile()
        self._delta_history: List[MetabolismDelta] = []
        self._max_delta_history = 50
        self._circadian_cycle = self._init_circadian_cycle()
        self._social_context_modifiers = {}
        self._emotional_resonance_tracker = []

    @property
    def state(self) -> MetabolismState:
        return self._state

    def consume(
        self,
        behavior: str,
        *,
        intensity: float = 1.0,
        content_length: int = 0,
        is_repeat: bool = False,
        social_context_factor: float = 1.0,
        emotional_resonance: float = 0.0,
    ) -> MetabolismDelta:
        """根据行为类型消耗资源，第三阶段升级：加入社会语境因子和情感共鸣调节"""
        cost_row = _BEHAVIOR_COST_TABLE.get(
            behavior, _BEHAVIOR_COST_TABLE[BehaviorCostType.IDLE.value]
        )
        fatigue = self._state.fatigue_factor()
        circadian_mod = self._calculate_circadian_modifier()
        social_mod = self._calculate_social_context_modifier(
            social_context_factor
        )
        # 内容长度修正因子
        length_scale = 1.0
        if content_length > 0:
            length_scale = max(0.6, min(2.0, content_length / 40.0))
        # 重复惩罚
        repeat_mod = 1.2 if is_repeat else 1.0
        # 情感共鸣调节
        resonance_mod = 1.0 + emotional_resonance * 0.3
        delta = MetabolismDelta(
            source_behavior=behavior, timestamp=time.time()
        )
        delta.chat_cost = (
            cost_row["chat"]
            * intensity
            * fatigue
            * length_scale
            * repeat_mod
            * circadian_mod
            * social_mod
            * resonance_mod
        )
        delta.thinking_cost = (
            cost_row["thinking"]
            * intensity
            * fatigue
            * length_scale
            * circadian_mod
            * social_mod
        )
        delta.activity_delta = cost_row["activity"] * intensity * circadian_mod
        delta.boredom_delta = (
            cost_row.get("boredom", 0.0) * intensity * social_mod
        )
        delta.loafing_delta = (
            cost_row.get("loafing", 0.0) * intensity * circadian_mod
        )
        delta.group_observe_cost = (
            cost_row.get("group_observe", 0.0) * intensity * social_mod
        )
        delta.pattern_analyze_cost = (
            cost_row.get("pattern_analyze", 0.0) * intensity * fatigue
        )
        delta.impression_update_cost = (
            cost_row.get("impression_update", 0.0)
            * intensity
            * emotional_resonance
        )
        delta.scene_adapt_cost = (
            cost_row.get("scene_adapt", 0.0)
            * intensity
            * social_context_factor
        )
        # 应用消耗
        self._state.chat_fuel = max(
            0.0, self._state.chat_fuel - delta.chat_cost
        )
        self._state.thinking_fuel = max(
            0.0, self._state.thinking_fuel - delta.thinking_cost
        )
        self._state.activity_gauge = max(
            0.0, min(100.0, self._state.activity_gauge + delta.activity_delta)
        )
        self._state.boredom_level = max(
            0.0, min(100.0, self._state.boredom_level + delta.boredom_delta)
        )
        self._state.loafing_level = max(
            0.0, min(100.0, self._state.loafing_level + delta.loafing_delta)
        )
        self._state.group_observation_fuel = max(
            0.0, self._state.group_observation_fuel - delta.group_observe_cost
        )
        self._state.pattern_analysis_fuel = max(
            0.0, self._state.pattern_analysis_fuel - delta.pattern_analyze_cost
        )
        self._state.impression_evolution_fuel = max(
            0.0,
            self._state.impression_evolution_fuel
            - delta.impression_update_cost,
        )
        self._state.scene_adaptation_energy = max(
            0.0, self._state.scene_adaptation_energy - delta.scene_adapt_cost
        )
        self._state.fatigue_accumulator += delta.total_cost() * 0.1
        self._state.total_consumed_today += delta.total_cost()
        # 更新活跃计时
        now = time.time()
        gap = now - self._state.last_action_time
        if gap < 180:
            self._state.consecutive_active_minutes += gap / 60.0
        else:
            self._state.consecutive_active_minutes = 0.0
        self._state.last_action_time = now
        # 更新情感共鸣追踪
        self._update_emotional_resonance(emotional_resonance)
        self._delta_history.append(delta)
        if len(self._delta_history) > self._max_delta_history:
            self._delta_history = self._delta_history[
                -self._max_delta_history:
            ]
        return delta

    def _init_circadian_cycle(self) -> Dict[str, float]:
        """初始化昼夜节律周期配置"""
        return {
            "dawn_recovery": 1.2,
            "morning_active": 0.9,
            "afternoon_stable": 1.0,
            "evening_social": 1.1,
            "night_rest": 0.6,
            "midnight_low": 0.4,
        }

    def _calculate_circadian_modifier(self) -> float:
        """计算当前时间的昼夜节律调节因子"""
        now = time.localtime()
        hour = now.tm_hour
        if 5 <= hour < 8:
            return self._circadian_cycle["dawn_recovery"]
        elif 8 <= hour < 12:
            return self._circadian_cycle["morning_active"]
        elif 12 <= hour < 18:
            return self._circadian_cycle["afternoon_stable"]
        elif 18 <= hour < 22:
            return self._circadian_cycle["evening_social"]
        elif 22 <= hour < 24:
            return self._circadian_cycle["night_rest"]
        else:
            return self._circadian_cycle["midnight_low"]

    def _calculate_social_context_modifier(
        self, social_factor: float
    ) -> float:
        """计算社会语境调节因子"""
        base_sensitivity = self._state.social_context_sensitivity
        return 1.0 + (social_factor - 1.0) * base_sensitivity

    def _update_emotional_resonance(self, resonance: float) -> None:
        """更新情感共鸣水平追踪"""
        self._emotional_resonance_tracker.append((time.time(), resonance))
        if len(self._emotional_resonance_tracker) > 20:
            self._emotional_resonance_tracker = (
                self._emotional_resonance_tracker[-20:]
            )
        # 计算平均情感共鸣水平
        if self._emotional_resonance_tracker:
            recent_resonances = [
                r for _, r in self._emotional_resonance_tracker[-10:]
            ]
            self._state.emotional_resonance_level = sum(
                recent_resonances
            ) / len(recent_resonances)

    def recover(
        self,
        elapsed_seconds: float,
        *,
        is_night: bool = False,
        group_activity_level: float = 0.0,
        scene_complexity: float = 0.0,
    ) -> MetabolismDelta:
        """按时间推进恢复，第三阶段升级：加入群活跃度和场景复杂度调节"""
        if elapsed_seconds <= 0:
            return MetabolismDelta(source_behavior="noop")
        minutes = elapsed_seconds / 60.0
        profile = self._recovery_profile
        base = profile.base_rate * minutes
        circadian_mod = self._calculate_circadian_modifier()
        # 安静场景恢复更快
        quiet_bonus = 1.0
        if self._state.activity_gauge < 20:
            quiet_bonus = profile.quiet_multiplier
        # 群活跃度提升恢复
        group_boost = 1.0 + group_activity_level * profile.group_activity_boost
        # 场景复杂度惩罚
        complexity_penalty = (
            1.0 + scene_complexity * profile.scene_complexity_penalty
        )
        night_mod = profile.night_multiplier if is_night else 1.0
        # F10：睡眠债务修正——高债务时恢复速率打折
        _debt_mod = 1.0
        try:
            from src.core.night_cycle_system import get_night_cycle

            _nc = get_night_cycle(self._channel_id)
            _debt_mod = _nc.get_recovery_rate_modifier()
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        chat_recovery = (
            base
            * quiet_bonus
            * night_mod
            * circadian_mod
            * group_boost
            * complexity_penalty
            * 0.8
            * _debt_mod
        )
        thinking_recovery = (
            base
            * quiet_bonus
            * night_mod
            * circadian_mod
            * group_boost
            * complexity_penalty
            * 1.0
            * _debt_mod
        )
        group_observe_recovery = (
            base
            * quiet_bonus
            * night_mod
            * circadian_mod
            * group_boost
            * 0.6
            * _debt_mod
        )
        pattern_analyze_recovery = (
            base * quiet_bonus * night_mod * circadian_mod * 0.7 * _debt_mod
        )
        impression_update_recovery = (
            base
            * quiet_bonus
            * night_mod
            * circadian_mod
            * (1.0 + self._state.emotional_resonance_level * 0.5)
            * 0.5
            * _debt_mod
        )
        scene_adapt_recovery = (
            base
            * quiet_bonus
            * night_mod
            * circadian_mod
            * (1.0 - scene_complexity * 0.3)
            * 0.8
            * _debt_mod
        )
        delta = MetabolismDelta(source_behavior="recovery")
        delta.chat_cost = -chat_recovery
        delta.thinking_cost = -thinking_recovery
        delta.group_observe_cost = -group_observe_recovery
        delta.pattern_analyze_cost = -pattern_analyze_recovery
        delta.impression_update_cost = -impression_update_recovery
        delta.scene_adapt_cost = -scene_adapt_recovery
        self._state.chat_fuel = min(
            100.0, self._state.chat_fuel + chat_recovery
        )
        self._state.thinking_fuel = min(
            100.0, self._state.thinking_fuel + thinking_recovery
        )
        self._state.group_observation_fuel = min(
            80.0, self._state.group_observation_fuel + group_observe_recovery
        )
        self._state.pattern_analysis_fuel = min(
            70.0, self._state.pattern_analysis_fuel + pattern_analyze_recovery
        )
        self._state.impression_evolution_fuel = min(
            90.0,
            self._state.impression_evolution_fuel + impression_update_recovery,
        )
        self._state.scene_adaptation_energy = min(
            60.0, self._state.scene_adaptation_energy + scene_adapt_recovery
        )
        # 活跃度自然衰减
        activity_decay = minutes * 0.3 * (1.0 - group_activity_level * 0.5)
        self._state.activity_gauge = max(
            0.0, self._state.activity_gauge - activity_decay
        )
        # 无聊随时间上升（受场景复杂度影响）
        _boredom_base = minutes * 0.15 * (1.0 + scene_complexity * 0.2)
        # F23：群环境调制——高热闹低价值群加速推高无聊和摸鱼
        _group_mod = 1.0
        try:
            from src.core.group_scene_state import get_group_scene

            _gs = get_group_scene(self._channel_id)
            _atmos = str(getattr(_gs, "current_atmosphere", "") or "").strip().lower()
            _hot_count = int(getattr(_gs, "active_user_count", 0) or 0)
            _hot_atmospheres = {
                "heated_discussion",
                "chaotic_spam",
                "meme_storm",
                "memeing",
                "social_gaming",
                "spam_flood",
            }
            _quiet_atmospheres = {
                "quiet_dormant",
                "cold_awkward",
                "quiet",
                "bored_drift",
                "mourning",
            }
            if (
                _atmos in _hot_atmospheres
                and _hot_count >= 5
            ):
                _group_mod = 1.6
                _loafing_group_boost = 1.4
            elif _atmos in _quiet_atmospheres:
                _group_mod = 0.7
                _loafing_group_boost = 0.8
            else:
                _loafing_group_boost = 1.0
        except Exception:
            _loafing_group_boost = 1.0
        boredom_rise = _boredom_base * _group_mod
        self._state.boredom_level = min(
            100.0, self._state.boredom_level + boredom_rise
        )
        delta.boredom_delta = boredom_rise
        # 摸鱼值随时间上升（受群活跃度影响）
        loafing_rise = (
            minutes
            * 0.1
            * (1.0 - group_activity_level * 0.3)
            * _loafing_group_boost
        )
        self._state.loafing_level = min(
            100.0, self._state.loafing_level + loafing_rise
        )
        delta.loafing_delta = loafing_rise
        # 疲劳累积器缓慢衰减
        fatigue_decay = (
            minutes
            * 0.05
            * (1.0 + self._state.emotional_resonance_level * 0.2)
        )
        self._state.fatigue_accumulator = max(
            0.0, self._state.fatigue_accumulator - fatigue_decay
        )
        # 连续活跃计时器在长时间空闲后归零
        if elapsed_seconds > 300:
            self._state.consecutive_active_minutes = 0.0
        self._state.last_recovery_time = time.time()
        return delta

    def tick(self) -> MetabolismDelta:
        """主循环中每轮调用一次，自动计算距上次的恢复量"""
        now = time.time()
        elapsed = now - self._state.last_recovery_time
        if elapsed < 5:
            return MetabolismDelta(source_behavior="tick_skip")
        return self.recover(elapsed)

    def should_perfunctory(self) -> bool:
        """是否应该进入敷衍状态，第三阶段升级：加入新资源维度判断"""
        return (
            self._state.chat_ratio() < 0.15
            or self._state.fatigue_accumulator > 8.0
            or (
                self._state.boredom_level > 70
                and self._state.chat_ratio() < 0.3
            )
            or self._state.group_observation_fuel < 20.0
            or self._state.pattern_analysis_fuel < 15.0
            or self._state.impression_evolution_fuel < 25.0
            or self._state.scene_adaptation_energy < 10.0
        )

    def should_withdraw(self) -> bool:
        """是否应该退出对话，第三阶段升级：加入新资源维度判断"""
        return (
            self._state.is_exhausted()
            or self._state.fatigue_accumulator > 15.0
            or (
                self._state.group_observation_fuel < 5.0
                and self._state.pattern_analysis_fuel < 5.0
            )
            or self._state.scene_adaptation_energy < 3.0
        )

    def recent_deltas(self, count: int = 5) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in self._delta_history[-count:]]

    def reset_daily(self) -> None:
        """每日重置累计消耗"""
        self._state.total_consumed_today = 0.0
        self._state.fatigue_accumulator = 0.0
        self._state.consecutive_active_minutes = 0.0

    def behavior_constraint_summary(self) -> Dict[str, Any]:
        """返回各资源维度对行为的约束汇总
        供主链决策点使用，将代谢值转化为可执行的行为约束。
        """
        s = self._state
        # META-01: 思考值预算约束
        thinking_ok = s.thinking_ratio() > 0.15
        # META-02: 聊天值决定回复长度提示
        chat_r = s.chat_ratio()
        if chat_r > 0.5:
            reply_hint = "normal"
        elif chat_r > 0.25:
            reply_hint = "brief"
        elif chat_r > 0.1:
            reply_hint = "perfunctory"
        else:
            reply_hint = "silent"
        # META-03: 活跃值影响行为启动能力
        activity_can_initiate = s.activity_gauge > 15.0
        # META-04: 社交值影响互动开放度
        social_norm = min(1.0, max(0.0, s.social_gauge / 50.0))
        if social_norm > 0.5:
            social_openness = "open"
        elif social_norm > 0.2:
            social_openness = "neutral"
        else:
            social_openness = "closed"
        # META-05: 无聊值推动窥屏倾向（0~1）
        boredom_drive = min(1.0, s.boredom_level / 80.0)
        # META-06: 摸鱼值压制参与（0~1）
        loafing_suppress = min(1.0, s.loafing_level / 60.0)
        # META-07: 能量总闸门
        gate_open = not self.should_withdraw()
        # 辅助维度
        is_perfunctory = self.should_perfunctory()
        fatigue = round(s.fatigue_factor(), 2)
        return {
            "thinking_budget_ok": thinking_ok,
            "reply_length_hint": reply_hint,
            "activity_can_initiate": activity_can_initiate,
            "social_openness": social_openness,
            "boredom_watch_drive": round(boredom_drive, 3),
            "loafing_suppression": round(loafing_suppress, 3),
            "energy_gate_open": gate_open,
            "is_perfunctory": is_perfunctory,
            "fatigue_factor": fatigue,
            "chat_fuel": round(s.chat_fuel, 1),
            "thinking_fuel": round(s.thinking_fuel, 1),
        }


def get_metabolism_engine(channel_id: str) -> MetabolismEngine:
    if channel_id not in _metabolism_instances:
        _metabolism_instances[channel_id] = MetabolismEngine(channel_id)
    return _metabolism_instances[channel_id]


def remove_metabolism_engine(channel_id: str) -> None:
    _metabolism_instances.pop(channel_id, None)

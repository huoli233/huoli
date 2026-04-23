import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.modules.trauma.runtime_config import trauma_module_view

logger = get_logger("创伤系统")


class TraumaState(Enum):
    """创伤状态"""

    SURFACE_NORMAL = "表面正常"
    SLIGHT_ABNORMAL = "轻微异常"
    MASK_SLIP = "伪装滑落"
    FORCED_NORMAL = "强撑正常"
    MASK_WAVERING = "伪装动摇"
    HALF_BREAKDOWN = "半崩溃"
    BARELY_HOLDING = "勉强支撑"
    IMMINENT_COLLAPSE = "即将崩溃"
    TOTAL_BREAKDOWN = "完全崩溃"


@dataclass
class TraumaFragment:
    """创伤碎片"""

    fragment_id: str = ""
    trigger_context: str = ""
    trigger_words: List[str] = field(default_factory=list)
    emotional_charge: float = 0.0
    distortion_level: float = 0.0
    memory_clarity: float = 5.0
    created_time: float = field(default_factory=time.time)
    last_activated: float = 0.0
    activation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "trigger_context": self.trigger_context,
            "trigger_words": self.trigger_words,
            "emotional_charge": self.emotional_charge,
            "distortion_level": self.distortion_level,
            "memory_clarity": self.memory_clarity,
            "created_time": self.created_time,
            "last_activated": self.last_activated,
            "activation_count": self.activation_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraumaFragment":
        return cls(
            fragment_id=data.get("fragment_id", ""),
            trigger_context=data.get("trigger_context", ""),
            trigger_words=data.get("trigger_words", []),
            emotional_charge=data.get("emotional_charge", 0.0),
            distortion_level=data.get("distortion_level", 0.0),
            memory_clarity=data.get("memory_clarity", 5.0),
            created_time=data.get("created_time", time.time()),
            last_activated=data.get("last_activated", 0.0),
            activation_count=data.get("activation_count", 0),
        )


@dataclass
class WorldviewCollapse:
    """世界观崩塌状态"""

    core_beliefs_damaged: List[str] = field(default_factory=list)
    trust_level: float = 5.0
    reality_distortion: float = 0.0
    cognitive_fragmentation: float = 0.0
    meaning_collapse: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_beliefs_damaged": self.core_beliefs_damaged,
            "trust_level": self.trust_level,
            "reality_distortion": self.reality_distortion,
            "cognitive_fragmentation": self.cognitive_fragmentation,
            "meaning_collapse": self.meaning_collapse,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldviewCollapse":
        return cls(
            core_beliefs_damaged=data.get("core_beliefs_damaged", []),
            trust_level=data.get("trust_level", 5.0),
            reality_distortion=data.get("reality_distortion", 0.0),
            cognitive_fragmentation=data.get("cognitive_fragmentation", 0.0),
            meaning_collapse=data.get("meaning_collapse", 0.0),
        )


@dataclass
class TraumaSystemState:
    """创伤系统完整状态"""

    inner_chaos_level: float = 0.0
    surface_mask_strength: float = 0.0
    current_state: TraumaState = TraumaState.SURFACE_NORMAL
    fragments: List[TraumaFragment] = field(default_factory=list)
    worldview: WorldviewCollapse = field(default_factory=WorldviewCollapse)
    stress_accumulation: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inner_chaos_level": self.inner_chaos_level,
            "surface_mask_strength": self.surface_mask_strength,
            "current_state": self.current_state.value,
            "fragments": [f.to_dict() for f in self.fragments],
            "worldview": self.worldview.to_dict(),
            "stress_accumulation": self.stress_accumulation,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraumaSystemState":
        state = cls(
            inner_chaos_level=data.get("inner_chaos_level", 0.0),
            surface_mask_strength=data.get("surface_mask_strength", 0.0),
            stress_accumulation=data.get("stress_accumulation", 0.0),
            last_updated=data.get("last_updated", time.time()),
        )
        state_str = data.get("current_state", "表面正常")
        for s in TraumaState:
            if s.value == state_str:
                state.current_state = s
                break
        for f_data in data.get("fragments", []):
            state.fragments.append(TraumaFragment.from_dict(f_data))
        wv_data = data.get("worldview", {})
        state.worldview = WorldviewCollapse.from_dict(wv_data)
        return state


class TraumaSystem:
    """创伤系统核心

    管理创伤碎片、触发评估、内心混乱和世界观冲击
    所有阈值从配置读取，禁止硬编码
    """

    def __init__(self):
        self._state = TraumaSystemState()
        self._model_client: Any = None
        self._prompt_manager: Any = None

        self._chaos_base_level = 0.0
        self._chaos_fragment_bonus = 0.0
        self._chaos_worldview_bonus = 0.0
        self._mask_base_strength = 8.0
        self._mask_strain = 0.0
        self._mask_social_pressure = 0.0
        self._mask_energy_level = 10.0
        self._last_mask_update = time.time()

        self._load_config()
        logger.info("创伤系统初始化完成")

    def _load_config(self):
        """加载配置"""
        trauma_cfg = trauma_module_view("trauma_system")

        self._max_fragments = trauma_cfg.get("max_fragments", 50)
        self._fragment_decay_rate = trauma_cfg.get("fragment_decay_rate", 0.01)
        self._chaos_decay_rate = trauma_cfg.get("chaos_decay_rate", 0.05)
        self._mask_recovery_rate = trauma_cfg.get("mask_recovery_rate", 0.02)
        self._stress_threshold = trauma_cfg.get("stress_threshold", 8.0)
        self._max_chaos_level = trauma_cfg.get("max_chaos_level", 10.0)
        self._max_mask_strength = trauma_cfg.get("max_mask_strength", 10.0)
        self._fragment_id_chars = int(trauma_cfg.get("fragment_id_chars", 16))
        self._fragment_bonus_scale = trauma_cfg.get("fragment_bonus_scale", 0.1)
        self._negative_trauma_scale = trauma_cfg.get("negative_trauma_scale", 1.2)
        self._negative_intensity_scale = trauma_cfg.get("negative_intensity_scale", 2.0)
        self._neutral_trauma_scale = trauma_cfg.get("neutral_trauma_scale", 0.8)
        self._worldview_meaning_bonus_scale = trauma_cfg.get("worldview_meaning_bonus_scale", 0.3)
        self._mask_base_strength = trauma_cfg.get("mask_base_strength", self._mask_base_strength)
        self._mask_energy_level = trauma_cfg.get("mask_energy_level", self._mask_energy_level)
        self._social_pressure_cap = trauma_cfg.get("social_pressure_cap", 10.0)
        self._fallback_social_pressure_scale = trauma_cfg.get("fallback_social_pressure_scale", 0.5)
        self._mask_strain_rate = trauma_cfg.get("mask_strain_rate", 0.3)
        self._negative_intensity_strain_scale = trauma_cfg.get("negative_intensity_strain_scale", 0.5)
        self._negative_social_strain_scale = trauma_cfg.get("negative_social_strain_scale", 0.2)
        self._social_strain_scale = trauma_cfg.get("social_strain_scale", 0.1)
        self._strain_recovery_per_minute = trauma_cfg.get("strain_recovery_per_minute", 0.02)
        self._strain_cap = trauma_cfg.get("strain_cap", 10.0)
        self._energy_cap = trauma_cfg.get("energy_cap", 10.0)
        self._energy_drain_scale = trauma_cfg.get("energy_drain_scale", 0.1)
        self._strain_penalty_scale = trauma_cfg.get("strain_penalty_scale", 0.5)
        self._detection_chaos_scale = trauma_cfg.get("detection_chaos_scale", 0.3)
        self._detection_strain_scale = trauma_cfg.get("detection_strain_scale", 0.2)
        self._detection_penalty_scale = trauma_cfg.get("detection_penalty_scale", 0.3)
        self._energy_baseline = trauma_cfg.get("energy_baseline", 5.0)
        self._energy_bonus_scale = trauma_cfg.get("energy_bonus_scale", 0.2)
        self._chaos_low_threshold = trauma_cfg.get("chaos_low_threshold", 4.0)
        self._chaos_high_threshold = trauma_cfg.get("chaos_high_threshold", 7.0)
        self._mask_high_threshold = trauma_cfg.get("mask_high_threshold", 8.0)
        self._mask_mid_threshold = trauma_cfg.get("mask_mid_threshold", 4.0)
        self._worldview_cap = trauma_cfg.get("worldview_cap", 10.0)
        self._stress_cap = trauma_cfg.get("stress_cap", 10.0)
        self._worldview_heal_scale = trauma_cfg.get("worldview_heal_scale", 0.5)

    def set_model_client(self, client: Any):
        """设置模型客户端"""
        self._model_client = client

    def set_prompt_manager(self, manager: Any):
        """设置提示词管理器"""
        self._prompt_manager = manager

    def add_fragment(self, fragment: TraumaFragment) -> None:
        """添加创伤碎片"""
        if not fragment.fragment_id:
            fragment.fragment_id = uuid.uuid4().hex[: self._fragment_id_chars]

        self._state.fragments.append(fragment)

        if len(self._state.fragments) > self._max_fragments:
            self._state.fragments = sorted(
                self._state.fragments,
                key=lambda f: f.emotional_charge,
                reverse=True,
            )[: self._max_fragments]

        self._chaos_fragment_bonus = sum(
            f.emotional_charge * self._fragment_bonus_scale for f in self._state.fragments
        )
        logger.debug(f"添加创伤碎片: {fragment.fragment_id}")

    def remove_fragment(self, fragment_id: str) -> bool:
        """移除创伤碎片"""
        for i, f in enumerate(self._state.fragments):
            if f.fragment_id == fragment_id:
                self._state.fragments.pop(i)
                self._recalculate_fragment_bonus()
                return True
        return False

    def _recalculate_fragment_bonus(self):
        """重新计算碎片加成"""
        self._chaos_fragment_bonus = sum(
            f.emotional_charge * self._fragment_bonus_scale for f in self._state.fragments
        )

    def update_from_context(self, context: Dict[str, Any]) -> None:
        """从上下文更新状态"""
        trauma_score = context.get("trauma_score", 0.0)
        sentiment = context.get("sentiment", "neutral")
        intensity = float(context.get("intensity", 0.0))
        social_pressure = context.get("social_pressure", 0.0)

        self._update_inner_chaos(trauma_score, sentiment, intensity)
        self._update_surface_mask(sentiment, intensity, social_pressure)
        self._evaluate_state()

        self._state.last_updated = time.time()

    def _update_inner_chaos(
        self, trauma_score: float, sentiment: str, intensity: float
    ):
        """更新内心混乱"""
        if sentiment == "negative":
            self._chaos_base_level = min(
                self._max_chaos_level,
                trauma_score * self._negative_trauma_scale + intensity * self._negative_intensity_scale,
            )
        elif trauma_score > 0:
            self._chaos_base_level = min(
                self._max_chaos_level, trauma_score * self._neutral_trauma_scale
            )
        else:
            self._chaos_base_level = max(
                0.0, self._chaos_base_level - self._chaos_decay_rate
            )

        self._chaos_worldview_bonus = (
            self._state.worldview.meaning_collapse * self._worldview_meaning_bonus_scale
        )

        self._state.inner_chaos_level = min(
            self._max_chaos_level,
            self._chaos_base_level
            + self._chaos_fragment_bonus
            + self._chaos_worldview_bonus,
        )

    def _update_surface_mask(
        self, sentiment: str, intensity: float, social_pressure: float
    ):
        """更新表面伪装"""
        current_time = time.time()
        time_elapsed = (current_time - self._last_mask_update) / 60.0

        self._mask_social_pressure = min(
            self._social_pressure_cap,
            social_pressure
            or self._state.inner_chaos_level * self._fallback_social_pressure_scale,
        )

        if sentiment == "negative":
            strain_increase = self._mask_strain_rate * (
                1 + intensity * self._negative_intensity_strain_scale + self._mask_social_pressure * self._negative_social_strain_scale
            )
        else:
            strain_increase = self._mask_strain_rate * (
                1 + self._mask_social_pressure * self._social_strain_scale
            )

        strain_recovery = time_elapsed * self._strain_recovery_per_minute
        self._mask_strain = max(
            0.0,
            min(self._strain_cap, self._mask_strain + strain_increase - strain_recovery),
        )

        energy_drain = self._mask_strain * self._energy_drain_scale
        energy_recovery = time_elapsed * self._mask_recovery_rate
        self._mask_energy_level = max(
            0.0,
            min(
                self._energy_cap, self._mask_energy_level - energy_drain + energy_recovery
            ),
        )

        strain_penalty = self._mask_strain * self._strain_penalty_scale
        detection_penalty = (
            min(
                self._energy_cap,
                self._state.inner_chaos_level * self._detection_chaos_scale
                + self._mask_strain * self._detection_strain_scale,
            )
            * self._detection_penalty_scale
        )
        energy_bonus = (self._mask_energy_level - self._energy_baseline) * self._energy_bonus_scale

        self._state.surface_mask_strength = max(
            0.0,
            min(
                self._max_mask_strength,
                self._mask_base_strength
                - strain_penalty
                - detection_penalty
                + energy_bonus,
            ),
        )

        self._last_mask_update = current_time

    def _evaluate_state(self):
        """评估当前状态"""
        chaos = self._state.inner_chaos_level
        mask = self._state.surface_mask_strength

        if chaos < self._chaos_low_threshold:
            if mask >= self._mask_high_threshold:
                self._state.current_state = TraumaState.SURFACE_NORMAL
            elif mask >= self._mask_mid_threshold:
                self._state.current_state = TraumaState.SLIGHT_ABNORMAL
            else:
                self._state.current_state = TraumaState.MASK_SLIP
        elif chaos < self._chaos_high_threshold:
            if mask >= self._mask_high_threshold:
                self._state.current_state = TraumaState.FORCED_NORMAL
            elif mask >= self._mask_mid_threshold:
                self._state.current_state = TraumaState.MASK_WAVERING
            else:
                self._state.current_state = TraumaState.HALF_BREAKDOWN
        else:
            if mask >= self._mask_high_threshold:
                self._state.current_state = TraumaState.BARELY_HOLDING
            elif mask >= self._mask_mid_threshold:
                self._state.current_state = TraumaState.IMMINENT_COLLAPSE
            else:
                self._state.current_state = TraumaState.TOTAL_BREAKDOWN

    def get_state(self) -> TraumaSystemState:
        """获取当前状态"""
        return self._state

    def get_chaos_level(self) -> float:
        """获取混乱等级"""
        return self._state.inner_chaos_level

    def get_mask_strength(self) -> float:
        """获取伪装强度"""
        return self._state.surface_mask_strength

    def get_current_state_name(self) -> str:
        """获取当前状态名称"""
        return self._state.current_state.value

    def get_fragments(self) -> List[TraumaFragment]:
        """获取所有碎片"""
        return self._state.fragments

    def get_worldview(self) -> WorldviewCollapse:
        """获取世界观状态"""
        return self._state.worldview

    def apply_worldview_impact(
        self,
        trust_shift: float = 0.0,
        reality_shift: float = 0.0,
        meaning_shift: float = 0.0,
        cognition_shift: float = 0.0,
        damaged_belief: str = "",
    ) -> None:
        """应用世界观冲击"""
        wv = self._state.worldview

        wv.trust_level = max(0.0, min(self._worldview_cap, wv.trust_level + trust_shift))
        wv.reality_distortion = max(
            0.0, min(self._worldview_cap, wv.reality_distortion + reality_shift)
        )
        wv.meaning_collapse = max(
            0.0, min(self._worldview_cap, wv.meaning_collapse + meaning_shift)
        )
        wv.cognitive_fragmentation = max(
            0.0, min(self._worldview_cap, wv.cognitive_fragmentation + cognition_shift)
        )

        if damaged_belief and damaged_belief not in wv.core_beliefs_damaged:
            wv.core_beliefs_damaged.append(damaged_belief)

    def heal_worldview(self, healing_amount: float = 0.5) -> None:
        """治愈世界观"""
        wv = self._state.worldview
        wv.trust_level = min(self._worldview_cap, wv.trust_level + healing_amount)
        wv.reality_distortion = max(
            0.0, wv.reality_distortion - healing_amount * self._worldview_heal_scale
        )
        wv.meaning_collapse = max(
            0.0, wv.meaning_collapse - healing_amount * self._worldview_heal_scale
        )

    def add_stress(self, amount: float) -> None:
        """添加压力"""
        self._state.stress_accumulation = min(
            self._stress_cap, self._state.stress_accumulation + amount
        )

    def reduce_stress(self, amount: float) -> None:
        """减少压力"""
        self._state.stress_accumulation = max(
            0.0, self._state.stress_accumulation - amount
        )

    def is_stressed(self) -> bool:
        """是否处于高压状态"""
        return self._state.stress_accumulation >= self._stress_threshold

    def get_psychological_profile(self) -> Dict[str, Any]:
        """获取心理档案"""
        return {
            "current_state": self._state.current_state.value,
            "inner_chaos_level": self._state.inner_chaos_level,
            "surface_mask_strength": self._state.surface_mask_strength,
            "stress_accumulation": self._state.stress_accumulation,
            "fragment_count": len(self._state.fragments),
            "worldview": self._state.worldview.to_dict(),
            "mask_strain": self._mask_strain,
            "mask_energy_level": self._mask_energy_level,
        }

    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return self._state.to_dict()

    def from_dict(self, data: Dict[str, Any]) -> None:
        """反序列化"""
        self._state = TraumaSystemState.from_dict(data)

    def reset(self) -> None:
        """重置状态"""
        self._state = TraumaSystemState()
        self._chaos_base_level = 0.0
        self._chaos_fragment_bonus = 0.0
        self._chaos_worldview_bonus = 0.0
        self._mask_strain = 0.0
        self._mask_energy_level = 10.0
        self._last_mask_update = time.time()


_trauma_system: Optional[TraumaSystem] = None


def get_trauma_system() -> TraumaSystem:
    """获取创伤系统单例"""
    global _trauma_system
    if _trauma_system is None:
        _trauma_system = TraumaSystem()
    return _trauma_system

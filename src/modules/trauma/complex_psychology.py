import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.modules.trauma.runtime_config import trauma_module_view

logger = get_logger("complex_psychology")


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
class PsychologicalLayer:
    """心理层级"""

    surface_layer: str = ""
    conscious_layer: str = ""
    subconscious_layer: str = ""
    trauma_echoes: List[str] = field(default_factory=list)
    inner_monologue: str = ""
    physical_symptoms: str = ""
    worldview_distortion: str = ""
    escape_urge: str = ""


@dataclass
class WorldviewState:
    """世界观状态"""

    damaged_beliefs: List[str] = field(default_factory=list)
    trust_level: float = 5.0
    reality_distortion: float = 0.0
    cognitive_fragmentation: float = 0.0
    meaning_collapse: float = 0.0


class ComplexTraumaPsychology:
    """复杂创伤心理系统

    独立双轨架构:
    - 内心混乱系统: 管理内部创伤积累
    - 表面伪装系统: 管理对外表现
    """

    def __init__(self):
        self._prompt_manager: Any = None
        self._llm_client: Any = None

        self.inner_chaos_level = 0.0
        self.chaos_base_level = 0.0
        self.chaos_fragment_bonus = 0.0
        self.chaos_worldview_bonus = 0.0

        self.surface_mask_strength = 8.0
        self.mask_base_strength = 8.0
        self.mask_strain = 0.0
        self.mask_social_pressure = 0.0
        self.mask_energy_level = 10.0
        self.last_mask_update = time.time()

        self.current_state = TraumaState.SURFACE_NORMAL
        self.combined_severity = 0.0

        self.worldview = WorldviewState()
        self.stress_accumulation = 0.0
        self.interaction_count = 0

        self._load_config()

    def _load_config(self):
        """加载配置"""
        trauma_cfg = trauma_module_view("trauma_complex")
        self._max_chaos_level = trauma_cfg.get("max_chaos_level", 10.0)
        self._max_mask_strength = trauma_cfg.get("max_mask_strength", 10.0)
        self._mask_strain_rate = trauma_cfg.get("mask_strain_rate", 0.3)
        self._mask_recovery_rate = trauma_cfg.get("mask_recovery_rate", 0.05)
        self._negative_trauma_scale = trauma_cfg.get("negative_trauma_scale", 1.2)
        self._negative_intensity_scale = trauma_cfg.get("negative_intensity_scale", 2.0)
        self._neutral_trauma_scale = trauma_cfg.get("neutral_trauma_scale", 0.8)
        self._worldview_meaning_bonus_scale = trauma_cfg.get("worldview_meaning_bonus_scale", 0.3)
        self._social_pressure_cap = trauma_cfg.get("social_pressure_cap", 10.0)
        self._fallback_social_pressure_scale = trauma_cfg.get("fallback_social_pressure_scale", 0.5)
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
        self._combined_chaos_weight = trauma_cfg.get("combined_chaos_weight", 0.6)
        self._combined_mask_weight = trauma_cfg.get("combined_mask_weight", 0.4)
        self._chaos_low_threshold = trauma_cfg.get("chaos_low_threshold", 4.0)
        self._chaos_high_threshold = trauma_cfg.get("chaos_high_threshold", 7.0)
        self._mask_high_threshold = trauma_cfg.get("mask_high_threshold", 8.0)
        self._mask_mid_threshold = trauma_cfg.get("mask_mid_threshold", 4.0)
        self._layer_high_chaos_threshold = trauma_cfg.get("layer_high_chaos_threshold", 7.0)
        self._layer_mid_chaos_threshold = trauma_cfg.get("layer_mid_chaos_threshold", 4.0)
        self._negative_intensity_threshold = trauma_cfg.get("negative_intensity_threshold", 0.6)
        self._worldview_damage_threshold = trauma_cfg.get("worldview_damage_threshold", 2.0)
        self._escape_chaos_threshold = trauma_cfg.get("escape_chaos_threshold", 5.0)
        self._trigger_max_chars = int(trauma_cfg.get("trigger_max_chars", 80))

    def set_prompt_manager(self, manager: Any):
        """设置提示词管理器"""
        self._prompt_manager = manager

    def set_llm_client(self, client: Any):
        """设置LLM客户端"""
        self._llm_client = client

    def _check_llm_available(self) -> bool:
        """检查LLM是否可用"""
        return (
            self._prompt_manager is not None and self._llm_client is not None
        )

    async def process_stimulus(
        self, content: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理外部刺激"""
        self._update_inner_chaos_system(context)
        self._update_surface_mask_system(context)
        apparent_state = self._evaluate_combined_state()
        self.current_state = apparent_state

        psychological_layers = self._build_psychological_layers(
            apparent_state, content, context
        )

        self.interaction_count += 1

        return {
            "apparent_state": apparent_state.value,
            "inner_chaos_level": self.inner_chaos_level,
            "surface_mask_strength": self.surface_mask_strength,
            "worldview_status": self._get_worldview_status(),
            "psychological_layers": psychological_layers,
        }

    def _update_inner_chaos_system(self, context: Dict):
        """更新内心混乱系统"""
        trauma_score = context.get("trauma_score", 0.0)
        sentiment = context.get("sentiment", "neutral")
        intensity = float(context.get("intensity", 0.0))

        if sentiment == "negative":
            self.chaos_base_level = min(
                self._max_chaos_level,
                trauma_score * self._negative_trauma_scale + intensity * self._negative_intensity_scale,
            )
        elif trauma_score > 0:
            self.chaos_base_level = min(
                self._max_chaos_level, trauma_score * self._neutral_trauma_scale
            )
        else:
            self.chaos_base_level = max(0.0, self.chaos_base_level - 0.0)

        self.chaos_worldview_bonus = self.worldview.meaning_collapse * self._worldview_meaning_bonus_scale

        self.inner_chaos_level = min(
            self._max_chaos_level,
            self.chaos_base_level
            + self.chaos_fragment_bonus
            + self.chaos_worldview_bonus,
        )

    def _update_surface_mask_system(self, context: Dict):
        """更新表面伪装系统"""
        current_time = time.time()
        time_elapsed_minutes = (current_time - self.last_mask_update) / 60.0

        sentiment = context.get("sentiment", "neutral")
        intensity = float(context.get("intensity", 0.0))

        self.mask_social_pressure = context.get("social_pressure", 0.0)
        if self.mask_social_pressure == 0.0:
            self.mask_social_pressure = min(
                self._social_pressure_cap,
                self.inner_chaos_level * self._fallback_social_pressure_scale,
            )

        if sentiment == "negative":
            strain_increase = self._mask_strain_rate * (
                1 + intensity * self._negative_intensity_strain_scale + self.mask_social_pressure * self._negative_social_strain_scale
            )
        else:
            strain_increase = self._mask_strain_rate * (
                1 + self.mask_social_pressure * self._social_strain_scale
            )

        strain_recovery = time_elapsed_minutes * self._strain_recovery_per_minute
        self.mask_strain = max(
            0.0,
            min(self._strain_cap, self.mask_strain + strain_increase - strain_recovery),
        )

        energy_drain = self.mask_strain * self._energy_drain_scale
        energy_recovery = time_elapsed_minutes * self._mask_recovery_rate
        self.mask_energy_level = max(
            0.0,
            min(self._energy_cap, self.mask_energy_level - energy_drain + energy_recovery),
        )

        strain_penalty = self.mask_strain * self._strain_penalty_scale
        detection_penalty = (
            min(
                self._energy_cap,
                self.inner_chaos_level * self._detection_chaos_scale
                + self.mask_strain * self._detection_strain_scale,
            )
            * self._detection_penalty_scale
        )
        energy_bonus = (self.mask_energy_level - self._energy_baseline) * self._energy_bonus_scale

        self.surface_mask_strength = max(
            0.0,
            min(
                self._max_mask_strength,
                self.mask_base_strength
                - strain_penalty
                - detection_penalty
                + energy_bonus,
            ),
        )

        self.last_mask_update = current_time

    def _evaluate_combined_state(self) -> TraumaState:
        """综合评估状态"""
        chaos = self.inner_chaos_level
        mask = self.surface_mask_strength

        self.combined_severity = chaos * self._combined_chaos_weight + (
            self._max_mask_strength - mask
        ) * self._combined_mask_weight

        if chaos < self._chaos_low_threshold:
            if mask >= self._mask_high_threshold:
                return TraumaState.SURFACE_NORMAL
            elif mask >= self._mask_mid_threshold:
                return TraumaState.SLIGHT_ABNORMAL
            else:
                return TraumaState.MASK_SLIP
        elif chaos < self._chaos_high_threshold:
            if mask >= self._mask_high_threshold:
                return TraumaState.FORCED_NORMAL
            elif mask >= self._mask_mid_threshold:
                return TraumaState.MASK_WAVERING
            else:
                return TraumaState.HALF_BREAKDOWN
        else:
            if mask >= self._mask_high_threshold:
                return TraumaState.BARELY_HOLDING
            elif mask >= self._mask_mid_threshold:
                return TraumaState.IMMINENT_COLLAPSE
            else:
                return TraumaState.TOTAL_BREAKDOWN

    def _build_psychological_layers(
        self, state: TraumaState, content: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """构建心理层级"""
        sentiment = str(context.get("sentiment", "neutral") or "neutral")
        intensity = max(
            0.0, min(1.0, float(context.get("intensity", 0.0) or 0.0))
        )
        trigger = (content or "").strip()[: self._trigger_max_chars]

        if self.inner_chaos_level >= self._layer_high_chaos_threshold:
            inner = "脑海像被拧紧了一样，很多念头同时拉扯着注意力。"
            body = "呼吸发浅，动作会先变僵，再努力维持镇定. "
        elif self.inner_chaos_level >= self._layer_mid_chaos_threshold:
            inner = "表面还维持得住，但心里已经明显开始不适. "
            body = "肩颈和手指会先一步绷紧. "
        else:
            inner = "情绪有波动，但还在可控范围内. "
            body = "身体只是轻微紧张. "

        if sentiment == "negative" and intensity >= self._negative_intensity_threshold:
            conscious = "理智层优先考虑拉开距离、缩短互动或降低暴露. "
        else:
            conscious = "理智层仍试图维持礼貌与表面秩序. "

        worldview = "对外界可信度的判断略有下滑. "
        if (
            self.worldview.reality_distortion >= self._worldview_damage_threshold
            or self.worldview.meaning_collapse >= self._worldview_damage_threshold
        ):
            worldview = (
                "世界观开始出现裂缝，对他人意图和现实稳定性的信任正在下降. "
            )

        return {
            "surface_layer": f"表面呈现为{
                state.value}. ",
            "conscious_layer": conscious,
            "subconscious_layer": inner,
            "trauma_echoes": [trigger] if trigger else [],
            "inner_monologue": inner,
            "physical_symptoms": body,
            "worldview_distortion": worldview,
            "escape_urge": (
                "想先拉开一点距离再继续判断. "
                if self.inner_chaos_level >= self._escape_chaos_threshold
                else "暂时还能继续应对. "
            ),
        }

    def _get_worldview_status(self) -> Dict[str, Any]:
        """获取世界观状态"""
        return {
            "damaged_beliefs": self.worldview.damaged_beliefs,
            "trust_level": self.worldview.trust_level,
            "reality_distortion": self.worldview.reality_distortion,
            "cognitive_fragmentation": self.worldview.cognitive_fragmentation,
        }

    def get_current_psychological_profile(self) -> Dict[str, Any]:
        """获取当前完整心理档案"""
        return {
            "current_state": self.current_state.value,
            "inner_chaos_level": self.inner_chaos_level,
            "surface_mask_strength": self.surface_mask_strength,
            "worldview_collapse": self._get_worldview_status(),
            "stress_accumulation": self.stress_accumulation,
            "combined_severity": self.combined_severity,
            "mask_energy_level": self.mask_energy_level,
            "mask_strain": self.mask_strain,
        }

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "current_state": self.current_state.value,
            "inner_chaos_level": self.inner_chaos_level,
            "surface_mask_strength": self.surface_mask_strength,
            "mask_strain": self.mask_strain,
            "combined_severity": self.combined_severity,
            "stress_accumulation": self.stress_accumulation,
            "worldview_collapse": {
                "core_beliefs_damaged": self.worldview.damaged_beliefs,
                "trust_level": self.worldview.trust_level,
                "reality_distortion": self.worldview.reality_distortion,
            },
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典加载"""
        if not data:
            return

        state_str = data.get("current_state", "表面正常")
        for s in TraumaState:
            if s.value == state_str:
                self.current_state = s
                break

        self.inner_chaos_level = data.get("inner_chaos_level", 0.0)
        self.surface_mask_strength = data.get("surface_mask_strength", 8.0)
        self.mask_strain = data.get("mask_strain", 1.0)
        self.combined_severity = data.get("combined_severity", 1.0)
        self.stress_accumulation = data.get("stress_accumulation", 1.0)

        wv_data = data.get("worldview_collapse", {})
        self.worldview = WorldviewState(
            damaged_beliefs=wv_data.get("core_beliefs_damaged", []),
            trust_level=wv_data.get("trust_level", 5.0),
            reality_distortion=wv_data.get("reality_distortion", 1.0),
        )


_complex_psychology: Optional[ComplexTraumaPsychology] = None


def get_complex_psychology() -> ComplexTraumaPsychology:
    """获取复杂心理系统单例"""
    global _complex_psychology
    if _complex_psychology is None:
        _complex_psychology = ComplexTraumaPsychology()
    return _complex_psychology

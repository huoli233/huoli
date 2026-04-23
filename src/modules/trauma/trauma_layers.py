import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.modules.trauma.runtime_config import trauma_module_view

logger = get_logger("创伤层")


class TraumaLayerType(Enum):
    """创伤层级类型"""

    SURFACE = "表层"
    SHALLOW = "浅层"
    MEDIUM = "中层"
    DEEP = "深层"
    CORE = "核心"


@dataclass
class TraumaLayer:
    """创伤层级"""

    layer_type: TraumaLayerType = TraumaLayerType.SURFACE
    severity: float = 0.0
    description: str = ""
    triggers: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activated: float = 0.0
    activation_count: int = 0
    healing_progress: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_type": self.layer_type.value,
            "severity": self.severity,
            "description": self.description,
            "triggers": self.triggers,
            "symptoms": self.symptoms,
            "created_at": self.created_at,
            "last_activated": self.last_activated,
            "activation_count": self.activation_count,
            "healing_progress": self.healing_progress,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraumaLayer":
        layer_type = TraumaLayerType.SURFACE
        for lt in TraumaLayerType:
            if lt.value == data.get("layer_type", "表层"):
                layer_type = lt
                break
        return cls(
            layer_type=layer_type,
            severity=data.get("severity", 0.0),
            description=data.get("description", ""),
            triggers=data.get("triggers", []),
            symptoms=data.get("symptoms", []),
            created_at=data.get("created_at", time.time()),
            last_activated=data.get("last_activated", 0.0),
            activation_count=data.get("activation_count", 0),
            healing_progress=data.get("healing_progress", 0.0),
        )


class TraumaLayerManager:
    """创伤层级管理器

    管理创伤的层级结构、深度评估和层级间影响
    """

    def __init__(self):
        self._layers: Dict[TraumaLayerType, TraumaLayer] = {}
        self._layer_influence_matrix: Dict[
            TraumaLayerType, Dict[TraumaLayerType, float]
        ] = {}
        self._load_config()
        self._initialize_layers()

    def _load_config(self):
        """加载配置"""
        trauma_cfg = trauma_module_view("trauma_layers")

        self._max_severity = trauma_cfg.get("max_severity", 10.0)
        self._healing_rate = trauma_cfg.get("healing_rate", 0.01)
        self._activation_decay = trauma_cfg.get("activation_decay", 0.1)
        self._cross_layer_influence = trauma_cfg.get(
            "cross_layer_influence", 0.3
        )
        self._max_triggers_per_layer = int(trauma_cfg.get("max_triggers_per_layer", 20))
        self._max_symptoms_per_layer = int(trauma_cfg.get("max_symptoms_per_layer", 15))
        self._decay_window_seconds = float(trauma_cfg.get("decay_window_seconds", 3600.0))

        default_influence_matrix = {
            TraumaLayerType.CORE: {
                TraumaLayerType.DEEP: 0.8,
                TraumaLayerType.MEDIUM: 0.5,
                TraumaLayerType.SHALLOW: 0.3,
                TraumaLayerType.SURFACE: 0.1,
            },
            TraumaLayerType.DEEP: {
                TraumaLayerType.CORE: 0.2,
                TraumaLayerType.MEDIUM: 0.6,
                TraumaLayerType.SHALLOW: 0.4,
                TraumaLayerType.SURFACE: 0.2,
            },
            TraumaLayerType.MEDIUM: {
                TraumaLayerType.CORE: 0.1,
                TraumaLayerType.DEEP: 0.3,
                TraumaLayerType.SHALLOW: 0.5,
                TraumaLayerType.SURFACE: 0.3,
            },
            TraumaLayerType.SHALLOW: {
                TraumaLayerType.CORE: 0.05,
                TraumaLayerType.DEEP: 0.1,
                TraumaLayerType.MEDIUM: 0.3,
                TraumaLayerType.SURFACE: 0.4,
            },
            TraumaLayerType.SURFACE: {
                TraumaLayerType.CORE: 0.02,
                TraumaLayerType.DEEP: 0.05,
                TraumaLayerType.MEDIUM: 0.1,
                TraumaLayerType.SHALLOW: 0.2,
            },
        }
        configured_matrix = trauma_cfg.get("layer_influence_matrix", {})
        self._layer_influence_matrix = self._build_layer_matrix(
            configured_matrix,
            default_influence_matrix,
        )
        self._severity_weights = self._build_layer_weights(
            trauma_cfg.get("severity_weights", {}),
            {
                TraumaLayerType.SURFACE: 0.1,
                TraumaLayerType.SHALLOW: 0.2,
                TraumaLayerType.MEDIUM: 0.25,
                TraumaLayerType.DEEP: 0.25,
                TraumaLayerType.CORE: 0.2,
            },
        )

    def _coerce_layer_type(self, value: Any) -> Optional[TraumaLayerType]:
        text = str(value).strip().lower()
        for layer_type in TraumaLayerType:
            if text in {layer_type.name.lower(), layer_type.value.lower()}:
                return layer_type
        return None

    def _build_layer_weights(
        self,
        raw_weights: Any,
        fallback: Dict[TraumaLayerType, float],
    ) -> Dict[TraumaLayerType, float]:
        if not isinstance(raw_weights, dict):
            return fallback
        weights = dict(fallback)
        for key, value in raw_weights.items():
            layer_type = self._coerce_layer_type(key)
            if layer_type is None:
                continue
            try:
                weights[layer_type] = float(value)
            except (TypeError, ValueError):
                continue
        return weights

    def _build_layer_matrix(
        self,
        raw_matrix: Any,
        fallback: Dict[TraumaLayerType, Dict[TraumaLayerType, float]],
    ) -> Dict[TraumaLayerType, Dict[TraumaLayerType, float]]:
        if not isinstance(raw_matrix, dict):
            return fallback
        matrix = {source: dict(targets) for source, targets in fallback.items()}
        for source_key, raw_targets in raw_matrix.items():
            source_type = self._coerce_layer_type(source_key)
            if source_type is None or not isinstance(raw_targets, dict):
                continue
            target_map = matrix.setdefault(source_type, {})
            for target_key, value in raw_targets.items():
                target_type = self._coerce_layer_type(target_key)
                if target_type is None:
                    continue
                try:
                    target_map[target_type] = float(value)
                except (TypeError, ValueError):
                    continue
        return matrix

    def _initialize_layers(self):
        """初始化各层级"""
        for layer_type in TraumaLayerType:
            self._layers[layer_type] = TraumaLayer(layer_type=layer_type)

    def get_layer(self, layer_type: TraumaLayerType) -> TraumaLayer:
        """获取指定层级"""
        return self._layers.get(layer_type, TraumaLayer(layer_type=layer_type))

    def update_layer(
        self,
        layer_type: TraumaLayerType,
        severity_delta: float = 0.0,
        trigger: str = "",
        symptom: str = "",
    ) -> None:
        """更新层级状态"""
        layer = self._layers.get(layer_type)
        if not layer:
            return

        layer.severity = max(
            0.0, min(self._max_severity, layer.severity + severity_delta)
        )
        layer.last_activated = time.time()
        layer.activation_count += 1

        if trigger and trigger not in layer.triggers:
            layer.triggers.append(trigger)
            if len(layer.triggers) > self._max_triggers_per_layer:
                layer.triggers = layer.triggers[-self._max_triggers_per_layer:]

        if symptom and symptom not in layer.symptoms:
            layer.symptoms.append(symptom)
            if len(layer.symptoms) > self._max_symptoms_per_layer:
                layer.symptoms = layer.symptoms[-self._max_symptoms_per_layer:]

        self._propagate_influence(layer_type, severity_delta)

    def _propagate_influence(self, source_type: TraumaLayerType, delta: float):
        """传播层级间影响"""
        influence_map = self._layer_influence_matrix.get(source_type, {})
        for target_type, influence_factor in influence_map.items():
            if target_type == source_type:
                continue
            target_layer = self._layers.get(target_type)
            if target_layer:
                influence = (
                    delta * influence_factor * self._cross_layer_influence
                )
                target_layer.severity = max(
                    0.0,
                    min(self._max_severity, target_layer.severity + influence),
                )

    def heal_layer(
        self, layer_type: TraumaLayerType, amount: float = 0.1
    ) -> float:
        """治愈层级"""
        layer = self._layers.get(layer_type)
        if not layer:
            return 0.0

        actual_heal = min(amount, layer.severity)
        layer.severity = max(0.0, layer.severity - actual_heal)
        layer.healing_progress = min(
            1.0, layer.healing_progress + actual_heal / self._max_severity
        )

        return actual_heal

    def get_total_severity(self) -> float:
        """获取总严重度"""
        total = 0.0
        for layer_type, layer in self._layers.items():
            weight = self._severity_weights.get(layer_type, 0.1)
            total += layer.severity * weight
        return total

    def get_dominant_layer(self) -> TraumaLayerType:
        """获取主导层级"""
        max_severity = 0.0
        dominant = TraumaLayerType.SURFACE
        for layer_type, layer in self._layers.items():
            if layer.severity > max_severity:
                max_severity = layer.severity
                dominant = layer_type
        return dominant

    def get_layer_summary(self, layer_type: TraumaLayerType) -> str:
        """获取层级摘要"""
        layer = self._layers.get(layer_type)
        if not layer:
            return f"{layer_type.value}: 无数据"

        parts = [f"{layer_type.value}: 严重度={layer.severity:.1f}"]

        if layer.triggers:
            parts.append(f"触发词: {', '.join(layer.triggers[:3])}")
        if layer.symptoms:
            parts.append(f"症状: {', '.join(layer.symptoms[:3])}")
        if layer.healing_progress > 0:
            parts.append(f"治愈进度: {layer.healing_progress * 100:.0f}%")

        return " | ".join(parts)

    def get_all_layers_status(self) -> Dict[str, Any]:
        """获取所有层级状态"""
        return {
            layer_type.value: layer.to_dict()
            for layer_type, layer in self._layers.items()
        }

    def apply_decay(self) -> None:
        """应用衰减"""
        for layer in self._layers.values():
            decay_amount = (
                self._activation_decay
                * (time.time() - layer.last_activated)
                / self._decay_window_seconds
            )
            layer.severity = max(0.0, layer.severity - decay_amount)

    def reset(self) -> None:
        """重置所有层级"""
        self._initialize_layers()

    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return self.get_all_layers_status()

    def from_dict(self, data: Dict[str, Any]) -> None:
        """反序列化"""
        for layer_type in TraumaLayerType:
            layer_data = data.get(layer_type.value, {})
            if layer_data:
                self._layers[layer_type] = TraumaLayer.from_dict(layer_data)


_trauma_layer_manager: Optional[TraumaLayerManager] = None


def get_trauma_layer_manager() -> TraumaLayerManager:
    """获取创伤层级管理器单例"""
    global _trauma_layer_manager
    if _trauma_layer_manager is None:
        _trauma_layer_manager = TraumaLayerManager()
    return _trauma_layer_manager

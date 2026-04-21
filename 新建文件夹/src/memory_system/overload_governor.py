import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine

logger = get_logger("负载治理")


class CognitiveLoadLevel(Enum):
    NORMAL = "normal"
    STRESSED = "stressed"
    OVERLOADED = "overloaded"
    CRITICAL = "critical"


@dataclass
class CognitiveSaturation:
    load_level: CognitiveLoadLevel = CognitiveLoadLevel.NORMAL
    capacity_ratio: float = 0.0
    memory_count: int = 0
    creation_factor: float = 1.0
    recall_factor: float = 1.0
    learning_factor: float = 1.0
    focus_factor: float = 1.0
    confusion_rate: float = 0.0
    cognitive_prompt: str = ""
    behavioral_hint: str = ""
    assessed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "load_level": self.load_level.value,
            "capacity_ratio": self.capacity_ratio,
            "memory_count": self.memory_count,
            "creation_factor": self.creation_factor,
            "recall_factor": self.recall_factor,
            "learning_factor": self.learning_factor,
            "focus_factor": self.focus_factor,
            "confusion_rate": self.confusion_rate,
            "cognitive_prompt": self.cognitive_prompt,
            "behavioral_hint": self.behavioral_hint,
            "assessed_at": self.assessed_at,
        }

    def is_overloaded(self) -> bool:
        return self.load_level.value >= CognitiveLoadLevel.OVERLOADED.value

    def is_stressed(self) -> bool:
        return self.load_level.value >= CognitiveLoadLevel.STRESSED.value


class ModelInterface:
    async def generate_raw(self, task_type: str, **kwargs) -> str:
        raise NotImplementedError


class DefaultModelInterface(ModelInterface):
    async def generate_raw(self, task_type: str, **kwargs) -> str:
        return '{"cognitive_prompt": "", "behavioral_hint": ""}'


class OverloadStateEngine:
    def __init__(
        self,
        config_engine=None,
        model_interface: Optional[ModelInterface] = None,
    ):
        self._config = config_engine or get_default_config_engine()
        self._model = model_interface or DefaultModelInterface()
        self._cached_saturation: Optional[CognitiveSaturation] = None
        self._cache_valid_seconds: float = 60.0
        self._level_thresholds: Dict[str, float] = {}
        self._saturation_curves: Dict[str, Dict[str, float]] = {}
        self._total_capacity: int = 1000
        self._load_config()

    def _load_config(self) -> None:
        self._level_thresholds = {
            "stressed": self._config.get(
                "overload", "stressed_threshold", 0.6
            ),
            "overloaded": self._config.get(
                "overload", "overloaded_threshold", 0.8
            ),
            "critical": self._config.get(
                "overload", "critical_threshold", 0.95
            ),
        }

        self._saturation_curves = {
            CognitiveLoadLevel.NORMAL.value: {
                "creation": 1.0,
                "recall": 1.0,
                "learning": 1.0,
                "focus": 1.0,
                "confusion": 0.0,
            },
            CognitiveLoadLevel.STRESSED.value: self._config.get(
                "overload",
                "stressed_factors",
                {
                    "creation": 0.85,
                    "recall": 0.9,
                    "learning": 0.8,
                    "focus": 0.85,
                    "confusion": 0.1,
                },
            ),
            CognitiveLoadLevel.OVERLOADED.value: self._config.get(
                "overload",
                "overloaded_factors",
                {
                    "creation": 0.6,
                    "recall": 0.7,
                    "learning": 0.5,
                    "focus": 0.6,
                    "confusion": 0.3,
                },
            ),
            CognitiveLoadLevel.CRITICAL.value: self._config.get(
                "overload",
                "critical_factors",
                {
                    "creation": 0.3,
                    "recall": 0.45,
                    "learning": 0.25,
                    "focus": 0.35,
                    "confusion": 0.6,
                },
            ),
        }

        raw_limits = self._config.get(
            "memory_capacity", "max_count_by_tier", {}
        )
        self._total_capacity = sum(
            int(raw_limits.get(k, 200))
            for k in ("immediate", "working", "recent", "midterm", "longterm")
        )
        self._cache_valid_seconds = self._config.get(
            "overload", "cache_seconds", 60.0
        )

    def set_model_interface(self, model: ModelInterface) -> None:
        self._model = model

    def _determine_level(self, ratio: float) -> CognitiveLoadLevel:
        if ratio >= self._level_thresholds["critical"]:
            return CognitiveLoadLevel.CRITICAL
        if ratio >= self._level_thresholds["overloaded"]:
            return CognitiveLoadLevel.OVERLOADED
        if ratio >= self._level_thresholds["stressed"]:
            return CognitiveLoadLevel.STRESSED
        return CognitiveLoadLevel.NORMAL

    def _compute_factors(self, level: CognitiveLoadLevel) -> Dict[str, float]:
        return dict(
            self._saturation_curves.get(
                level.value,
                self._saturation_curves[CognitiveLoadLevel.NORMAL.value],
            )
        )

    def assess(self, memory_count: int) -> CognitiveSaturation:
        if (
            self._cached_saturation
            and (time.time() - self._cached_saturation.assessed_at)
            < self._cache_valid_seconds
            and self._cached_saturation.memory_count == memory_count
        ):
            return self._cached_saturation

        ratio = min(1.0, memory_count / max(1, self._total_capacity))
        level = self._determine_level(ratio)
        factors = self._compute_factors(level)

        saturation = CognitiveSaturation(
            load_level=level,
            capacity_ratio=ratio,
            memory_count=memory_count,
            creation_factor=float(factors.get("creation", 1.0)),
            recall_factor=float(factors.get("recall", 1.0)),
            learning_factor=float(factors.get("learning", 1.0)),
            focus_factor=float(factors.get("focus", 1.0)),
            confusion_rate=float(factors.get("confusion", 0.0)),
        )
        self._cached_saturation = saturation
        return saturation

    async def assess_with_prompt(
        self, memory_count: int
    ) -> CognitiveSaturation:
        saturation = self.assess(memory_count)

        if saturation.load_level == CognitiveLoadLevel.NORMAL:
            return saturation

        if saturation.cognitive_prompt:
            return saturation

        if not self._model:
            return saturation

        try:
            raw_output = await self._model.generate_raw(
                "overload_prompt",
                overload_level=saturation.load_level.value,
                memory_count=str(saturation.memory_count),
                capacity_ratio=f"{saturation.capacity_ratio:.2f}",
                creation_factor=f"{saturation.creation_factor:.2f}",
                recall_factor=f"{saturation.recall_factor:.2f}",
                learning_factor=f"{saturation.learning_factor:.2f}",
                focus_factor=f"{saturation.focus_factor:.2f}",
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.debug("[过载] LLM 提示词生成失败: %s", exc)
            return saturation

        import json

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            return saturation

        if parsed:
            saturation.cognitive_prompt = str(
                parsed.get("cognitive_prompt", "")
            )[:200]
            saturation.behavioral_hint = str(
                parsed.get("behavioral_hint", "")
            )[:100]
            self._cached_saturation = saturation

        return saturation

    def invalidate_cache(self) -> None:
        self._cached_saturation = None

    def get_thresholds(self) -> Dict[str, float]:
        return dict(self._level_thresholds)

    def get_total_capacity(self) -> int:
        return self._total_capacity

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_capacity": self._total_capacity,
            "cache_valid_seconds": self._cache_valid_seconds,
            "level_thresholds": self._level_thresholds,
            "cached_saturation": (
                self._cached_saturation.to_dict()
                if self._cached_saturation
                else None
            ),
        }


_overload_engine: Optional[OverloadStateEngine] = None


def get_overload_engine(
    config_engine=None, model_interface: Optional[ModelInterface] = None
) -> OverloadStateEngine:
    global _overload_engine
    if _overload_engine is None:
        _overload_engine = OverloadStateEngine(config_engine, model_interface)
    elif model_interface is not None:
        _overload_engine.set_model_interface(model_interface)
    return _overload_engine

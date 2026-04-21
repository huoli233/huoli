import time
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine

logger = get_logger("情感重要性")


@dataclass
class ImportanceAdjustment:
    original_importance: float = 0.5
    delta: float = 0.0
    adjusted_importance: float = 0.5
    memory_tag: str = "normal"
    reason: str = ""
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_importance": self.original_importance,
            "delta": self.delta,
            "adjusted_importance": self.adjusted_importance,
            "memory_tag": self.memory_tag,
            "reason": self.reason,
            "evaluated_at": self.evaluated_at,
        }


class ModelInterface:
    async def generate_raw(self, task_type: str, **kwargs) -> str:
        raise NotImplementedError


class DefaultModelInterface(ModelInterface):
    async def generate_raw(self, task_type: str, **kwargs) -> str:
        return '{"importance_delta": 0.0, "adjusted_importance": 0.5, "memory_tag": "normal"}'


class EmotionImportanceLinker:
    def __init__(
        self,
        config_engine=None,
        model_interface: Optional[ModelInterface] = None,
    ):
        self._config = config_engine or get_default_config_engine()
        self._model = model_interface or DefaultModelInterface()
        self._evaluation_cooldown: float = 5.0
        self._last_eval_at: float = 0.0
        self._skip_below_length: int = 10
        self._enabled: bool = True
        self._adjustment_history: List[Dict[str, Any]] = []
        self._max_history: int = 100
        self._load_config()

    def _load_config(self) -> None:
        self._evaluation_cooldown = self._config.get(
            "emotion_importance", "cooldown_seconds", 5.0
        )
        self._skip_below_length = self._config.get(
            "emotion_importance", "min_content_length", 10
        )
        self._enabled = self._config.get("emotion_importance", "enabled", True)

    def set_model_interface(self, model: ModelInterface) -> None:
        self._model = model

    async def evaluate(
        self,
        content: str,
        base_importance: float,
        emotion_context: Dict[str, Any],
    ) -> ImportanceAdjustment:
        passthrough = ImportanceAdjustment(
            original_importance=base_importance,
            adjusted_importance=base_importance,
        )

        if not self._enabled:
            return passthrough

        if len(content) < self._skip_below_length:
            return passthrough

        now = time.time()
        if (now - self._last_eval_at) < self._evaluation_cooldown:
            return passthrough

        self._last_eval_at = now

        try:
            raw_output = await self._model.generate_raw(
                "emotion_importance",
                base_importance=f"{base_importance:.2f}",
                content=content[:1500],
                affection_delta=str(
                    emotion_context.get("affection_delta", 0.0)
                ),
                chaos_level=str(emotion_context.get("chaos_level", 0.0)),
                annoyance_level=str(
                    emotion_context.get("annoyance_level", "low")
                ),
                sentiment=str(emotion_context.get("sentiment", "neutral")),
                trauma_related=str(
                    emotion_context.get("trauma_related", False)
                ),
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.debug("[情绪重要性] LLM 评估失败: %s", exc)
            return passthrough

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            return passthrough

        if not parsed:
            return passthrough

        delta = max(-0.3, min(0.4, float(parsed.get("importance_delta", 0.0))))
        adjusted = max(
            0.0,
            min(
                1.0,
                float(
                    parsed.get("adjusted_importance", base_importance + delta)
                ),
            ),
        )
        tag = str(parsed.get("memory_tag", "normal"))
        if tag not in ("normal", "emotional", "trauma", "joyful", "conflict"):
            tag = "normal"

        result = ImportanceAdjustment(
            original_importance=base_importance,
            delta=delta,
            adjusted_importance=adjusted,
            memory_tag=tag,
            reason=str(parsed.get("reason", ""))[:100],
        )

        if abs(delta) > 0.05:
            logger.info(
                "[情绪重要性] 调整 %.2f -> %.2f (tag=%s): %s",
                base_importance,
                adjusted,
                tag,
                result.reason,
            )

        self._adjustment_history.append(
            {
                "time": now,
                "delta": delta,
                "tag": tag,
                "content_preview": content[:50],
            }
        )
        if len(self._adjustment_history) > self._max_history:
            self._adjustment_history = self._adjustment_history[
                -self._max_history:
            ]

        return result

    def evaluate_sync(
        self,
        content: str,
        base_importance: float,
        emotion_context: Dict[str, Any],
    ) -> ImportanceAdjustment:
        passthrough = ImportanceAdjustment(
            original_importance=base_importance,
            adjusted_importance=base_importance,
        )

        if not self._enabled:
            return passthrough

        if len(content) < self._skip_below_length:
            return passthrough

        affection_delta = emotion_context.get("affection_delta", 0.0)
        chaos_level = emotion_context.get("chaos_level", 0.0)
        trauma_related = emotion_context.get("trauma_related", False)
        sentiment = emotion_context.get("sentiment", "neutral")

        delta = 0.0
        tag = "normal"
        reason = ""

        if trauma_related:
            delta += 0.25
            tag = "trauma"
            reason = "涉及创伤内容"

        if affection_delta > 0.3:
            delta += 0.15
            tag = "joyful" if tag == "normal" else tag
            reason = "高好感度互动"
        elif affection_delta < -0.3:
            delta -= 0.1
            tag = "conflict" if tag == "normal" else tag
            reason = "低好感度互动"

        if chaos_level > 0.5:
            delta += 0.1
            reason = "高混乱状态"

        if sentiment == "positive":
            delta += 0.05
        elif sentiment == "negative":
            delta -= 0.05

        delta = max(-0.3, min(0.4, delta))
        adjusted = max(0.0, min(1.0, base_importance + delta))

        return ImportanceAdjustment(
            original_importance=base_importance,
            delta=delta,
            adjusted_importance=adjusted,
            memory_tag=tag,
            reason=reason,
        )

    def get_adjustment_stats(self) -> Dict[str, Any]:
        if not self._adjustment_history:
            return {
                "total_adjustments": 0,
                "avg_delta": 0.0,
                "tag_distribution": {},
            }

        total = len(self._adjustment_history)
        avg_delta = (
            sum(a.get("delta", 0.0) for a in self._adjustment_history) / total
        )
        tag_counts: Dict[str, int] = {}
        for a in self._adjustment_history:
            tag = a.get("tag", "normal")
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total_adjustments": total,
            "avg_delta": avg_delta,
            "tag_distribution": tag_counts,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "evaluation_cooldown": self._evaluation_cooldown,
            "skip_below_length": self._skip_below_length,
            "last_eval_at": self._last_eval_at,
            "adjustment_stats": self.get_adjustment_stats(),
        }

    def clear_history(self) -> None:
        self._adjustment_history.clear()


_emotion_importance_linker: Optional[EmotionImportanceLinker] = None


def get_emotion_importance_linker(
    config_engine=None,
    model_interface: Optional[ModelInterface] = None,
) -> EmotionImportanceLinker:
    global _emotion_importance_linker
    if _emotion_importance_linker is None:
        _emotion_importance_linker = EmotionImportanceLinker(
            config_engine, model_interface
        )
    elif model_interface is not None:
        _emotion_importance_linker.set_model_interface(model_interface)
    return _emotion_importance_linker

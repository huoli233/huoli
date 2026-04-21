import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine
from src.memory_system.memory_models import BufferedMemory

logger = get_logger("aging_processor")


@dataclass
class AgingAction:
    memory_id: str = ""
    action_type: str = "none"
    age_days: float = 0.0
    clarity_label: str = ""
    original_length: int = 0
    compressed_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "action_type": self.action_type,
            "age_days": self.age_days,
            "clarity_label": self.clarity_label,
            "original_length": self.original_length,
            "compressed_length": self.compressed_length,
        }


class ClarityLabelMapper:
    def __init__(self, config_engine=None):
        self._config = config_engine or get_default_config_engine()
        self._label_thresholds: List[Tuple[float, str]] = []
        self._prompt_templates: Dict[str, str] = {}
        self._load_config()

    def _load_config(self) -> None:
        self._label_thresholds = [
            (
                self._config.get("clarity_labels", "vivid_threshold", 0.85),
                "清晰",
            ),
            (
                self._config.get("clarity_labels", "clear_threshold", 0.65),
                "比较清楚",
            ),
            (
                self._config.get("clarity_labels", "hazy_threshold", 0.45),
                "有些模糊",
            ),
            (
                self._config.get("clarity_labels", "dim_threshold", 0.25),
                "很模糊",
            ),
            (0.0, "几乎遗忘"),
        ]
        self._prompt_templates = self._config.get(
            "clarity_labels",
            "prompt_hints",
            {
                "清晰": "你清晰地记得这件事",
                "比较清楚": "你比较清楚地记得这件事",
                "有些模糊": "你对这件事的记忆有些模糊了",
                "很模糊": "你对这件事已经记忆很模糊，只记得大概",
                "几乎遗忘": "你几乎已经忘记了这件事，只有零星印象",
            },
        )

    def get_label(self, clarity_value: float) -> str:
        for threshold, label in self._label_thresholds:
            if clarity_value >= threshold:
                return label
        return "几乎遗忘"

    def get_prompt_hint(self, clarity_value: float) -> str:
        label = self.get_label(clarity_value)
        return self._prompt_templates.get(label, "")

    def annotate_memories(
        self, memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        for mem in memories:
            clarity = float(mem.get("clarity", mem.get("importance", 0.5)))
            mem["clarity_label"] = self.get_label(clarity)
            mem["clarity_hint"] = self.get_prompt_hint(clarity)
        return memories


class ModelInterface:
    async def generate_raw(self, task_type: str, **kwargs) -> str:
        raise NotImplementedError


class DefaultModelInterface(ModelInterface):
    async def generate_raw(self, task_type: str, **kwargs) -> str:
        return '{"compressed_content": ""}'


class MemoryAgingProcessor:
    def __init__(
        self,
        config_engine=None,
        model_interface: Optional[ModelInterface] = None,
    ):
        self._config = config_engine or get_default_config_engine()
        self._model = model_interface or DefaultModelInterface()
        self._clarity_mapper = ClarityLabelMapper(self._config)
        self._compress_age_days: float = 60.0
        self._compress_min_length: int = 150
        self._cleanup_age_days: float = 180.0
        self._cleanup_importance_threshold: float = 0.1
        self._max_compress_per_cycle: int = 5
        self._load_config()

    def _load_config(self) -> None:
        self._compress_age_days = self._config.get(
            "memory_aging", "compress_age_days", 60.0
        )
        self._compress_min_length = self._config.get(
            "memory_aging", "compress_min_length", 150
        )
        self._cleanup_age_days = self._config.get(
            "memory_aging", "cleanup_age_days", 180.0
        )
        self._cleanup_importance_threshold = self._config.get(
            "memory_aging", "cleanup_importance_threshold", 0.1
        )
        self._max_compress_per_cycle = self._config.get(
            "memory_aging", "max_compress_per_cycle", 5
        )

    @property
    def clarity_mapper(self) -> ClarityLabelMapper:
        return self._clarity_mapper

    def set_model_interface(self, model: ModelInterface) -> None:
        self._model = model

    async def process_aging_cycle(
        self, memories: List[BufferedMemory]
    ) -> List[AgingAction]:
        actions: List[AgingAction] = []
        now = time.time()
        compress_candidates = []

        for mem in memories:
            age_days = (now - mem.created_at) / 86400.0

            if (
                age_days >= self._cleanup_age_days
                and mem.importance < self._cleanup_importance_threshold
            ):
                actions.append(
                    AgingAction(
                        memory_id=mem.memory_id,
                        action_type="cleanup",
                        age_days=age_days,
                        clarity_label=self._clarity_mapper.get_label(
                            mem.importance * 0.5
                        ),
                        original_length=len(mem.content),
                    )
                )
                continue

            if (
                age_days >= self._compress_age_days
                and len(mem.content) >= self._compress_min_length
                and not mem.compressed_content
            ):
                compress_candidates.append((mem, age_days))

        for mem, age_days in compress_candidates[
            : self._max_compress_per_cycle
        ]:
            compressed = await self._compress_memory(mem, age_days)
            if compressed:
                actions.append(
                    AgingAction(
                        memory_id=mem.memory_id,
                        action_type="compress",
                        age_days=age_days,
                        clarity_label=self._clarity_mapper.get_label(
                            mem.importance * 0.7
                        ),
                        original_length=len(mem.content),
                        compressed_length=len(compressed),
                    )
                )

        if actions:
            logger.info(
                "[老化] 周期完成: %d 条压缩, %d 条清理",
                sum(1 for a in actions if a.action_type == "compress"),
                sum(1 for a in actions if a.action_type == "cleanup"),
            )

        return actions

    async def _compress_memory(
        self, mem: BufferedMemory, age_days: float
    ) -> Optional[str]:
        try:
            raw_output = await self._model.generate_raw(
                "compress_old",
                original_content=mem.content[:2000],
                age_days=f"{age_days:.0f}",
                importance=f"{mem.importance:.2f}",
                original_length=str(len(mem.content)),
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.debug("[老化] 压缩失败 %s: %s", mem.memory_id, exc)
            return None

        import json

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            return None

        if not parsed:
            return None

        compressed = str(parsed.get("compressed_content", ""))
        if compressed and len(compressed) < len(mem.content):
            mem.compressed_content = compressed
            return compressed

        return None

    def get_cleanup_ids(self, actions: List[AgingAction]) -> List[str]:
        return [a.memory_id for a in actions if a.action_type == "cleanup"]

    def get_age_level(self, created_at: float) -> str:
        age_days = (time.time() - created_at) / 86400.0
        if age_days < 7:
            return "fresh"
        elif age_days < 30:
            return "recent"
        elif age_days < 90:
            return "aging"
        elif age_days < 180:
            return "old"
        else:
            return "ancient"

    def should_compress(self, mem: BufferedMemory) -> bool:
        age_days = (time.time() - mem.created_at) / 86400.0
        return (
            age_days >= self._compress_age_days
            and len(mem.content) >= self._compress_min_length
            and not mem.compressed_content
        )

    def should_cleanup(self, mem: BufferedMemory) -> bool:
        age_days = (time.time() - mem.created_at) / 86400.0
        return (
            age_days >= self._cleanup_age_days
            and mem.importance < self._cleanup_importance_threshold
        )


_aging_processor: Optional[MemoryAgingProcessor] = None


def get_aging_processor(
    config_engine=None,
    model_interface: Optional[ModelInterface] = None,
) -> MemoryAgingProcessor:
    global _aging_processor
    if _aging_processor is None:
        _aging_processor = MemoryAgingProcessor(config_engine, model_interface)
    return _aging_processor

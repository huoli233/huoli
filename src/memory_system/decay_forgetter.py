import asyncio
import time
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from src.common.logger import get_logger
from src.memory_system.memory_models import BufferedMemory
from src.memory_system.runtime_config import memory_module_view

logger = get_logger("衰减遗忘")


@dataclass
class DecayReport:
    processed_count: int = 0
    decayed_count: int = 0
    forgotten_count: int = 0
    forgotten_ids: List[str] = field(default_factory=list)
    cycle_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processed_count": self.processed_count,
            "decayed_count": self.decayed_count,
            "forgotten_count": self.forgotten_count,
            "forgotten_ids": list(self.forgotten_ids),
            "cycle_at": self.cycle_at,
        }


class ModelInterface:
    async def generate_raw(self, task_type: str, **kwargs) -> str:
        raise NotImplementedError


class DefaultModelInterface(ModelInterface):
    async def generate_raw(self, task_type: str, **kwargs) -> str:
        return '{"decisions": []}'


class ImportanceDecayScheduler:
    def __init__(
        self,
        config_engine=None,
        model_interface: Optional[ModelInterface] = None,
    ):
        self._config = config_engine
        self._model = model_interface or DefaultModelInterface()
        self._decay_rate_per_hour: float = 0.005
        self._forget_threshold: float = 0.15
        self._cycle_interval_seconds: float = 300.0
        self._max_batch_for_llm: int = 15
        self._min_age_for_decay_seconds: float = 3600.0
        self._scheduler_task: Optional[asyncio.Task] = None
        self._background_tasks: Set[asyncio.Task] = set()
        self._recent_reports: List[DecayReport] = []
        self._running = False
        self._load_config()

    def _load_config(self) -> None:
        if self._config is not None:
            self._decay_rate_per_hour = self._config.get(
                "decay_forgetting", "decay_rate_per_hour", 0.005
            )
            self._forget_threshold = self._config.get(
                "decay_forgetting", "forget_threshold", 0.15
            )
            self._cycle_interval_seconds = self._config.get(
                "decay_forgetting", "cycle_interval_seconds", 300.0
            )
            self._max_batch_for_llm = self._config.get(
                "decay_forgetting", "max_batch_for_llm", 15
            )
            self._min_age_for_decay_seconds = self._config.get(
                "decay_forgetting", "min_age_for_decay_seconds", 3600.0
            )
            return
        view = memory_module_view("memory_decay")
        self._decay_rate_per_hour = float(view.get("decay_rate_per_hour", 0.005))
        self._forget_threshold = float(view.get("forget_threshold", 0.15))
        self._cycle_interval_seconds = float(view.get("cycle_interval_seconds", 300.0))
        self._max_batch_for_llm = int(view.get("max_batch_for_llm", 15))
        self._min_age_for_decay_seconds = float(view.get("min_age_for_decay_seconds", 3600.0))

    def set_model_interface(self, model: ModelInterface) -> None:
        self._model = model

    def apply_passive_decay(self, memories: List[BufferedMemory]) -> int:
        now = time.time()
        decayed_count = 0
        elapsed_hours = self._cycle_interval_seconds / 3600.0

        for mem in memories:
            age = now - mem.created_at
            if age < self._min_age_for_decay_seconds:
                continue
            decay_amount = self._decay_rate_per_hour * elapsed_hours
            age_multiplier = min(2.0, age / 86400.0)
            access_protection = min(0.5, mem.access_count * 0.05)
            effective_decay = max(
                0.0,
                decay_amount * age_multiplier
                - access_protection * decay_amount,
            )
            if effective_decay > 0.0001:
                mem.importance = max(0.0, mem.importance - effective_decay)
                decayed_count += 1

        return decayed_count

    def collect_forget_candidates(
        self, memories: List[BufferedMemory]
    ) -> List[BufferedMemory]:
        candidates = []
        now = time.time()
        for mem in memories:
            if mem.importance < self._forget_threshold:
                age = now - mem.created_at
                if age > self._min_age_for_decay_seconds:
                    candidates.append(mem)
        candidates.sort(key=lambda m: m.importance)
        return candidates[: self._max_batch_for_llm]

    async def evaluate_batch_forget(
        self, candidates: List[BufferedMemory]
    ) -> List[str]:
        if not candidates or not self._model:
            return []

        batch_text_parts = []
        for mem in candidates:
            age_days = (time.time() - mem.created_at) / 86400.0
            batch_text_parts.append(
                f"- id={mem.memory_id}, 年龄={age_days:.1f}天, "
                f"重要性={mem.importance:.3f}, 访问={mem.access_count}次, "
                f"内容前80字: {mem.content[:80]}"
            )

        batch_text = "\n".join(batch_text_parts)

        try:
            raw_output = await self._model.generate_raw(
                "batch_forget",
                memory_batch=batch_text,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.debug("[遗忘] LLM 批量评估失败: %s", exc)
            return []

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            return []

        if not parsed:
            return []

        decisions = parsed.get("decisions", [])
        if not isinstance(decisions, list):
            return []

        forget_ids = []
        for decision in decisions:
            if isinstance(decision, dict) and bool(
                decision.get("should_forget", False)
            ):
                mid = str(decision.get("memory_id", ""))
                if mid:
                    forget_ids.append(mid)

        return forget_ids

    async def run_decay_cycle(
        self,
        memory_provider: Callable[[], List[BufferedMemory]],
        memory_remover: Optional[Callable[[List[str]], Coroutine]] = None,
    ) -> DecayReport:
        memories = memory_provider()
        report = DecayReport(processed_count=len(memories))

        report.decayed_count = self.apply_passive_decay(memories)

        candidates = self.collect_forget_candidates(memories)
        if candidates:
            forget_ids = await self.evaluate_batch_forget(candidates)
            if forget_ids and memory_remover:
                await memory_remover(forget_ids)
            report.forgotten_count = len(forget_ids)
            report.forgotten_ids = forget_ids

        self._recent_reports.append(report)
        if len(self._recent_reports) > 50:
            self._recent_reports = self._recent_reports[-30:]

        if report.forgotten_count > 0:
            logger.info(
                "[遗忘] 周期完成: 处理%d条, 衰减%d条, 遗忘%d条",
                report.processed_count,
                report.decayed_count,
                report.forgotten_count,
            )

        return report

    def start_scheduler(
        self,
        memory_provider: Callable[[], List[BufferedMemory]],
        memory_remover: Optional[Callable[[List[str]], Coroutine]] = None,
    ) -> None:
        if self._scheduler_task and not self._scheduler_task.done():
            return

        self._running = True

        async def _loop():
            while self._running:
                await asyncio.sleep(self._cycle_interval_seconds)
                try:
                    await self.run_decay_cycle(memory_provider, memory_remover)
                except Exception as exc:
                    logger.warning("[遗忘] 衰减周期异常: %s", exc)

        self._scheduler_task = asyncio.create_task(_loop())
        logger.info(
            "[遗忘] 定时衰减已启动, 间隔 %.0f 秒", self._cycle_interval_seconds
        )

    async def stop_scheduler(self) -> None:
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._background_tasks.clear()

    def get_recent_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._recent_reports[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "decay_rate_per_hour": self._decay_rate_per_hour,
            "forget_threshold": self._forget_threshold,
            "cycle_interval_seconds": self._cycle_interval_seconds,
            "max_batch_for_llm": self._max_batch_for_llm,
            "min_age_for_decay_seconds": self._min_age_for_decay_seconds,
            "is_running": self._running,
            "recent_reports_count": len(self._recent_reports),
        }


_decay_scheduler: Optional[ImportanceDecayScheduler] = None


def get_decay_scheduler(
    config_engine=None,
    model_interface: Optional[ModelInterface] = None,
) -> ImportanceDecayScheduler:
    global _decay_scheduler
    if _decay_scheduler is None:
        _decay_scheduler = ImportanceDecayScheduler(
            config_engine, model_interface
        )
    elif model_interface is not None:
        _decay_scheduler.set_model_interface(model_interface)
    return _decay_scheduler

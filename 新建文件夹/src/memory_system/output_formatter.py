import json
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine

logger = get_logger("输出格式化")


class BufferedMemory:
    memory_id: str
    content: str
    created_at: float
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = 0.0
    compressed_content: str = ""
    summary: str = ""
    keywords: List[str] = None
    channel_id: str = ""
    user_id: str = ""
    tier: Any = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.keywords is None:
            self.keywords = []


class PromptManagerInterface:
    def get_raw(self, name: str) -> Optional[str]:
        raise NotImplementedError


class DefaultPromptManager(PromptManagerInterface):
    def get_raw(self, name: str) -> Optional[str]:
        return None


class LLMClientInterface:
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError


class DefaultLLMClient(LLMClientInterface):
    async def generate(self, prompt: str) -> str:
        return ""


class OutputFormatter:
    def __init__(
        self, config_engine=None, prompt_manager=None, llm_client=None
    ):
        self._config = config_engine or get_default_config_engine()
        self._prompt_manager = prompt_manager or DefaultPromptManager()
        self._llm_client = llm_client or DefaultLLMClient()

        self._default_mode = "structured"
        self._max_tokens = 600
        self._include_timestamps = True
        self._max_items = 12
        self._load_config()

    def _load_config(self) -> None:
        self._default_mode = str(
            self._config.get("memory_format", "default_mode", "structured")
        )
        self._max_tokens = max(
            64, int(self._config.get("memory_format", "max_tokens", 600))
        )
        self._include_timestamps = bool(
            self._config.get("memory_format", "include_timestamps", True)
        )
        self._max_items = max(
            1, int(self._config.get("memory_format", "max_items", 12))
        )

    def bind_llm(self, prompt_manager, llm_client) -> None:
        self._prompt_manager = prompt_manager
        self._llm_client = llm_client

    async def format_for_prompt(
        self,
        memories: List[BufferedMemory],
        query: Optional[str] = None,
        max_tokens: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> str:
        if not memories:
            return ""

        selected = memories[: self._max_items]
        template = self._get_template()
        if not template:
            return self._fallback_format(selected)

        payload = self._build_payload(selected)
        budget = max_tokens if max_tokens is not None else self._max_tokens
        chosen_mode = mode or self._default_mode

        main_personality = str(self._config.get("personality", "name", ""))

        prompt = template.format(
            main_personality=main_personality,
            mode=chosen_mode,
            query=query or "",
            max_tokens=max(64, int(budget)),
            include_timestamps=self._include_timestamps,
            memories=json.dumps(payload, ensure_ascii=False, indent=2),
        )

        if self._llm_client is None:
            return self._fallback_format(selected)

        try:
            result = await self._llm_client.generate(prompt)
            return (result or "").strip()
        except Exception as e:
            logger.debug("[记忆格式化] LLM格式化失败，回退基础格式: %s", e)
            return self._fallback_format(selected)

    async def format_summary(self, memories: List[BufferedMemory]) -> str:
        return await self.format_for_prompt(memories=memories, mode="summary")

    async def format_structured(self, memories: List[BufferedMemory]) -> str:
        return await self.format_for_prompt(
            memories=memories, mode="structured"
        )

    async def format_timeline(self, memories: List[BufferedMemory]) -> str:
        return await self.format_for_prompt(memories=memories, mode="timeline")

    def _get_template(self) -> str:
        if self._prompt_manager is None:
            return ""
        return self._prompt_manager.get_raw("memory_format_template")

    def _build_payload(
        self, memories: List[BufferedMemory]
    ) -> List[Dict[str, object]]:
        output: List[Dict[str, object]] = []
        for memory in memories:
            item: Dict[str, object] = {
                "memory_id": memory.memory_id,
                "content": memory.content,
                "summary": memory.summary or "",
                "importance": memory.importance,
                "access_count": memory.access_count,
                "keywords": memory.keywords or [],
            }
            if hasattr(memory, "tier") and memory.tier is not None:
                tier_name = (
                    memory.tier.name.lower()
                    if hasattr(memory.tier, "name")
                    else str(memory.tier)
                )
                item["tier"] = tier_name
            if self._include_timestamps:
                item["created_at"] = memory.created_at
                item["last_accessed"] = memory.last_accessed
            output.append(item)
        return output

    def _fallback_format(self, memories: List[BufferedMemory]) -> str:
        lines = []
        for memory in memories:
            text = memory.summary or memory.content[:100]
            lines.append(f"- {text}")
        return "\n".join(lines)

    def format_sync(
        self, memories: List[BufferedMemory], mode: Optional[str] = None
    ) -> str:
        if not memories:
            return ""

        selected = memories[: self._max_items]
        chosen_mode = mode or self._default_mode

        if chosen_mode == "summary":
            return self._format_summary_sync(selected)
        elif chosen_mode == "timeline":
            return self._format_timeline_sync(selected)
        else:
            return self._fallback_format(selected)

    def _format_summary_sync(self, memories: List[BufferedMemory]) -> str:
        grouped: Dict[str, List[str]] = {}
        for memory in memories:
            date_key = self._get_date_key(memory.created_at)
            if date_key not in grouped:
                grouped[date_key] = []
            text = memory.summary or memory.content[:80]
            grouped[date_key].append(text)

        lines = []
        for date_key in sorted(grouped.keys(), reverse=True):
            lines.append(f"[{date_key}]")
            for text in grouped[date_key][:3]:
                lines.append(f"  - {text}")
        return "\n".join(lines)

    def _format_timeline_sync(self, memories: List[BufferedMemory]) -> str:
        sorted_memories = sorted(
            memories, key=lambda m: m.created_at, reverse=True
        )
        lines = []
        for memory in sorted_memories:
            date_str = self._format_timestamp(memory.created_at)
            text = memory.summary or memory.content[:60]
            lines.append(f"[{date_str}] {text}")
        return "\n".join(lines)

    def _get_date_key(self, timestamp: float) -> str:
        import time

        t = time.localtime(timestamp)
        return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"

    def _format_timestamp(self, timestamp: float) -> str:
        import time

        t = time.localtime(timestamp)
        return f"{t.tm_mon:02d}/{t.tm_mday:02d} {t.tm_hour:02d}:{t.tm_min:02d}"

    def get_stats(self) -> Dict[str, Any]:
        return {
            "default_mode": self._default_mode,
            "max_tokens": self._max_tokens,
            "include_timestamps": self._include_timestamps,
            "max_items": self._max_items,
        }


_output_formatter: Optional[OutputFormatter] = None


def get_output_formatter(
    config_engine=None, prompt_manager=None, llm_client=None
) -> OutputFormatter:
    global _output_formatter
    if _output_formatter is None:
        _output_formatter = OutputFormatter(
            config_engine=config_engine,
            prompt_manager=prompt_manager,
            llm_client=llm_client,
        )
    else:
        _output_formatter.bind_llm(
            prompt_manager=prompt_manager, llm_client=llm_client
        )
    return _output_formatter

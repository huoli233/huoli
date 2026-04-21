import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine

logger = get_logger("content_shuffler")


@dataclass
class ShuffleResult:
    original: str = ""
    shuffled: str = ""
    shuffle_type: str = ""
    confidence: float = 0.0
    reasoning: str = ""


class ModelInterface:
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError


class DefaultModelInterface(ModelInterface):
    async def generate(self, prompt: str) -> str:
        return '{"shuffled_content": ""}'


class PromptManagerInterface:
    def get(self, name: str, **kwargs) -> Optional[str]:
        raise NotImplementedError


class DefaultPromptManager(PromptManagerInterface):
    def get(self, name: str, **kwargs) -> Optional[str]:
        return None


class ContentShuffler:
    def __init__(
        self,
        config_engine=None,
        prompt_manager: Optional[PromptManagerInterface] = None,
        model_client: Optional[ModelInterface] = None,
    ):
        self._config = config_engine or get_default_config_engine()
        self._prompts = prompt_manager or DefaultPromptManager()
        self._model = model_client or DefaultModelInterface()
        self._shuffle_probability: float = 0.15
        self._max_shuffle_length: int = 100
        self._shuffle_types: List[str] = [
            "typo",
            "wrong_word",
            "tone_shift",
            "incomplete",
        ]
        self._load_config()

    def _load_config(self) -> None:
        self._shuffle_probability = float(
            self._config.get("content_shuffle", "probability", 0.15)
        )
        self._max_shuffle_length = int(
            self._config.get("content_shuffle", "max_length", 100)
        )

    def set_model_client(self, client: ModelInterface) -> None:
        self._model = client

    def set_prompt_manager(self, manager: PromptManagerInterface) -> None:
        self._prompts = manager

    async def shuffle(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ShuffleResult:
        if not text or len(text) > self._max_shuffle_length:
            return ShuffleResult(original=text, shuffled=text)
        if random.random() > self._shuffle_probability:
            return ShuffleResult(original=text, shuffled=text)
        prompt = self._build_prompt(text, context)
        try:
            raw_response = await self._model.generate(prompt)
            return self._parse_response(raw_response, text)
        except Exception as e:
            logger.debug("[内容混淆] 模型调用失败: %s", e)
            return ShuffleResult(original=text, shuffled=text)

    def _build_prompt(
        self, text: str, context: Optional[Dict[str, Any]]
    ) -> str:
        context_str = json.dumps(
            context or {}, ensure_ascii=False, default=str
        )[:300]
        return f"生成说错话效果:\n\n原始内容: {text}\n上下文: {context_str}\n\n返回JSON: shuffled_content, shuffle_type, confidence, reasoning"

    def _parse_response(
        self, raw_response: str, original: str
    ) -> ShuffleResult:
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return ShuffleResult(original=original, shuffled=original)
        if not isinstance(parsed, dict):
            return ShuffleResult(original=original, shuffled=original)
        shuffled = str(parsed.get("shuffled_content", "")).strip()
        if not shuffled or shuffled == original:
            return ShuffleResult(original=original, shuffled=original)
        return ShuffleResult(
            original=original,
            shuffled=shuffled,
            shuffle_type=str(parsed.get("shuffle_type", "typo")),
            confidence=self._safe_float(parsed.get("confidence", 0.5)),
            reasoning=str(parsed.get("reasoning", ""))[:100],
        )

    def shuffle_sync(self, text: str) -> ShuffleResult:
        if not text or random.random() > self._shuffle_probability:
            return ShuffleResult(original=text, shuffled=text)
        shuffle_type = random.choice(self._shuffle_types)
        shuffled = self._apply_simple_shuffle(text, shuffle_type)
        return ShuffleResult(
            original=text,
            shuffled=shuffled,
            shuffle_type=shuffle_type,
            confidence=0.6,
            reasoning="简单混淆规则",
        )

    def _apply_simple_shuffle(self, text: str, shuffle_type: str) -> str:
        if shuffle_type == "typo" and len(text) > 3:
            chars = list(text)
            idx = random.randint(1, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            return "".join(chars)
        elif shuffle_type == "wrong_word":
            wrong_words = {
                "好": "坏",
                "对": "错",
                "是": "不是",
                "喜欢": "讨厌",
            }
            for k, v in wrong_words.items():
                if k in text:
                    return text.replace(k, v, 1)
        elif shuffle_type == "incomplete":
            if len(text) > 5:
                return text[: len(text) // 2] + "..."
        return text

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (ValueError, TypeError):
            return 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "shuffle_probability": self._shuffle_probability,
            "max_shuffle_length": self._max_shuffle_length,
            "shuffle_types": self._shuffle_types,
        }


_content_shuffler: Optional[ContentShuffler] = None


def get_content_shuffler(
    config_engine=None,
    prompt_manager: Optional[PromptManagerInterface] = None,
    model_client: Optional[ModelInterface] = None,
) -> ContentShuffler:
    global _content_shuffler
    if _content_shuffler is None:
        _content_shuffler = ContentShuffler(
            config_engine=config_engine,
            prompt_manager=prompt_manager,
            model_client=model_client,
        )
    elif model_client is not None:
        _content_shuffler.set_model_client(model_client)
    return _content_shuffler

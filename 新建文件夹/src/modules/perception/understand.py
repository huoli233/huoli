import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine

logger = get_logger("understand")


@dataclass
class UnderstandResult:
    task: str = ""
    confidence: float = 0.5
    reasoning: str = ""
    raw_response: str = ""

    behavior_type: str = "neutral"
    intent: str = "chat"
    severity: float = 0.0

    sentiment: str = "neutral"
    intensity: float = 0.0
    emotions: Dict[str, float] = field(default_factory=dict)

    is_harassment: bool = False
    harassment_type: str = "none"

    state: str = "uncertain"
    willingness: float = 0.5
    interest_level: float = 0.5
    attention_level: float = 0.5

    atmosphere: str = "neutral"
    tension_level: float = 0.0
    hostility_level: float = 0.0
    activity_level: float = 0.5
    key_factors: List[str] = field(default_factory=list)
    social_role: str = "observer"

    topic: str = ""
    novelty: float = 0.5
    social_relevance: float = 0.5


class ModelInterface:
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError


class DefaultModelInterface(ModelInterface):
    async def generate(self, prompt: str) -> str:
        return '{"confidence": 0.5}'


class PromptManagerInterface:
    def get(self, name: str, **kwargs) -> Optional[str]:
        raise NotImplementedError


class DefaultPromptManager(PromptManagerInterface):
    def get(self, name: str, **kwargs) -> Optional[str]:
        return None


class ModelUnderstand:
    def __init__(
        self,
        config_engine=None,
        prompt_manager: Optional[PromptManagerInterface] = None,
        model_client: Optional[ModelInterface] = None,
    ):
        self._config = config_engine or get_default_config_engine()
        self._prompts = prompt_manager or DefaultPromptManager()
        self._model_client = model_client or DefaultModelInterface()
        self._cache: Dict[str, UnderstandResult] = {}
        self._cache_ttl: float = 60.0
        self._cache_max_size: int = 1000
        self._load_config()

    def _load_config(self) -> None:
        self._cache_ttl = float(
            self._config.get("understand", "cache_ttl", 60.0)
        )
        self._cache_max_size = int(
            self._config.get("understand", "cache_max_size", 1000)
        )

    def set_model_client(self, client: ModelInterface) -> None:
        self._model_client = client

    def set_prompt_manager(self, manager: PromptManagerInterface) -> None:
        self._prompts = manager

    async def generate_raw(self, template_name: str, **kwargs) -> str:
        prompt = self._prompts.get(template_name, **kwargs)
        if not prompt:
            return "{}"
        return await self._model_client.generate(prompt)

    async def analyze(
        self, task: str, content: str, context: Optional[Dict[str, Any]] = None
    ) -> UnderstandResult:
        cache_key = self._generate_cache_key(task, content, context)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        prompt = self._build_prompt(task, content, context or {})
        raw_response = await self._model_client.generate(prompt)
        result = self._parse_response(raw_response, task)
        result.raw_response = raw_response
        self._cache_set(cache_key, result)
        return result

    async def analyze_batch(
        self,
        content: str,
        context: Dict[str, Any],
        tasks: List[str],
    ) -> Dict[str, UnderstandResult]:
        if len(tasks) == 1:
            result = await self.analyze(tasks[0], content, context)
            return {tasks[0]: result}
        combined_prompt = self._build_combined_prompt(content, context, tasks)
        raw_response = await self._model_client.generate(combined_prompt)
        return self._parse_combined_response(raw_response, tasks)

    def _generate_cache_key(
        self, task: str, content: str, context: Optional[Dict[str, Any]]
    ) -> str:
        import hashlib

        combined = f"{task}:{content}:{
            json.dumps(
                context,
                sort_keys=True,
                default=str)}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def _cache_set(self, key: str, result: UnderstandResult) -> None:
        if len(self._cache) >= self._cache_max_size:
            self._cache.popitem()
        self._cache[key] = result

    def _build_prompt(
        self, task: str, content: str, context: Dict[str, Any]
    ) -> str:
        task_templates = {
            "behavior_evaluation": "分析用户行为类型、意图和严重程度",
            "sentiment_analysis": "分析情感倾向、强度和情绪分布",
            "intent_classification": "分析意图类型和置信度",
            "user_state_inference": "分析用户状态和聊天意愿",
            "atmosphere_perception": "分析群聊氛围和紧张程度",
            "harassment_detection": "分析是否骚扰及类型",
            "interest_evaluation": "分析话题兴趣和新颖度",
        }
        task_desc = task_templates.get(task, task)
        context_str = json.dumps(context, ensure_ascii=False, default=str)[
            :1000
        ]
        return f"任务: {task_desc}\n\n用户消息: {content}\n\n上下文: {context_str}\n\n请以JSON格式返回分析结果。"

    def _build_combined_prompt(
        self, content: str, context: Dict[str, Any], tasks: List[str]
    ) -> str:
        task_descriptions = {
            "behavior_evaluation": "行为类型(behavior_type)、意图(intent)、严重程度(severity)",
            "sentiment_analysis": "情感倾向(sentiment)、强度(intensity)、情绪分布(emotions)",
            "harassment_detection": "是否骚扰(is_harassment)、类型(type)、严重度(severity)",
            "intent_classification": "意图类型(intent)、置信度(confidence)",
            "user_state_inference": "用户状态(state)、聊天意愿(willingness)",
        }
        task_list = "\n".join(
            f"- {task}: {task_descriptions.get(task, task)}" for task in tasks
        )
        context_str = json.dumps(context, ensure_ascii=False, default=str)[
            :1500
        ]
        return f"请同时完成以下多个分析任务：\n\n用户消息: {content}\n\n上下文: {context_str}\n\n需要分析的任务:\n{task_list}\n\n请以JSON格式返回，每个任务作为一个键。 "

    def _parse_response(
        self, raw_response: str, task: str
    ) -> UnderstandResult:
        result = UnderstandResult(task=task)
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return result
        if not isinstance(parsed, dict):
            return result
        result.confidence = self._safe_float(
            parsed.get("confidence", 0.5), 0.5
        )
        result.reasoning = str(parsed.get("reasoning", ""))[:200]
        if task == "behavior_evaluation":
            result.behavior_type = str(parsed.get("behavior_type", "neutral"))
            result.intent = str(parsed.get("intent", "chat"))
            result.severity = self._safe_float(
                parsed.get("severity", 0.0), 0.0
            )
        elif task == "sentiment_analysis":
            result.sentiment = str(parsed.get("sentiment", "neutral"))
            result.intensity = self._safe_float(
                parsed.get("intensity", 0.0), 0.0
            )
            result.emotions = parsed.get("emotions", {})
        elif task == "intent_classification":
            result.intent = str(parsed.get("intent", "chat"))
        elif task == "user_state_inference":
            result.state = str(parsed.get("state", "uncertain"))
            result.willingness = self._safe_float(
                parsed.get("willingness", 0.5), 0.5
            )
            result.interest_level = self._safe_float(
                parsed.get("interest_level", 0.5), 0.5
            )
            result.attention_level = self._safe_float(
                parsed.get("attention_level", 0.5), 0.5
            )
        elif task == "atmosphere_perception":
            result.atmosphere = str(parsed.get("atmosphere", "neutral"))
            result.tension_level = self._safe_float(
                parsed.get("tension_level", 0.0), 0.0
            )
            result.hostility_level = self._safe_float(
                parsed.get("hostility_level", 0.0), 0.0
            )
            result.activity_level = self._safe_float(
                parsed.get("activity_level", 0.5), 0.5
            )
            result.key_factors = parsed.get("key_factors", [])
            result.social_role = str(parsed.get("social_role", "observer"))
        elif task == "harassment_detection":
            result.is_harassment = bool(parsed.get("is_harassment", False))
            result.harassment_type = str(parsed.get("type", "none"))
            result.severity = self._safe_float(
                parsed.get("severity", 0.0), 0.0
            )
        elif task == "interest_evaluation":
            result.topic = str(parsed.get("topic", ""))
            result.novelty = self._safe_float(parsed.get("novelty", 0.5), 0.5)
            result.social_relevance = self._safe_float(
                parsed.get("social_relevance", 0.5), 0.5
            )
        return result

    def _parse_combined_response(
        self, raw_response: str, tasks: List[str]
    ) -> Dict[str, UnderstandResult]:
        results = {}
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            for task in tasks:
                results[task] = UnderstandResult(task=task)
            return results
        if not isinstance(parsed, dict):
            for task in tasks:
                results[task] = UnderstandResult(task=task)
            return results
        for task in tasks:
            task_data = parsed.get(task, {})
            if task_data and isinstance(task_data, dict):
                task_json = json.dumps(task_data, ensure_ascii=False)
                results[task] = self._parse_response(task_json, task)
            else:
                results[task] = UnderstandResult(task=task)
        return results

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (ValueError, TypeError):
            return default

    def get_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self._cache),
            "cache_ttl": self._cache_ttl,
            "cache_max_size": self._cache_max_size,
        }


_model_understand: Optional[ModelUnderstand] = None


def get_model_understand(
    config_engine=None,
    prompt_manager: Optional[PromptManagerInterface] = None,
    model_client: Optional[ModelInterface] = None,
) -> ModelUnderstand:
    global _model_understand
    if _model_understand is None:
        _model_understand = ModelUnderstand(
            config_engine=config_engine,
            prompt_manager=prompt_manager,
            model_client=model_client,
        )
    elif model_client is not None:
        _model_understand.set_model_client(model_client)
    return _model_understand

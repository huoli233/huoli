from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.common.logger import get_logger
from src.modules.perception.runtime_config import perception_module_view

logger = get_logger("兴趣评分")


@dataclass
class InterestScore:
    topic: str = ""
    novelty: float = 0.5
    social_relevance: float = 0.5
    confidence: float = 0.5
    reasoning: str = ""


class ModelInterface:
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError


class DefaultModelInterface(ModelInterface):
    async def generate(self, prompt: str) -> str:
        return '{"topic": "", "novelty": 0.5, "social_relevance": 0.5}'


class InterestScorer:
    def __init__(
        self, config_engine=None, model_client: Optional[ModelInterface] = None
    ):
        del config_engine
        self._model = model_client or DefaultModelInterface()
        self._novelty_weight: float = 0.5
        self._relevance_weight: float = 0.5
        self._high_interest_threshold: float = 0.7
        self._load_config()

    def _load_config(self) -> None:
        config = perception_module_view("perception_interest")
        self._novelty_weight = float(
            config.get("novelty_weight", 0.5)
        )
        self._relevance_weight = float(
            config.get("relevance_weight", 0.5)
        )
        self._high_interest_threshold = float(
            config.get("high_threshold", 0.7)
        )

    def set_model_client(self, client: ModelInterface) -> None:
        self._model = client

    async def score(
        self, content: str, context: Optional[Dict[str, Any]] = None
    ) -> InterestScore:
        if not content:
            return InterestScore()
        prompt = self._build_prompt(content, context or {})
        try:
            raw_response = await self._model.generate(prompt)
            return self._parse_response(raw_response)
        except Exception as e:
            logger.debug("[兴趣评分] 模型调用失败: %s", e)
            return InterestScore(reasoning=str(e))

    def _build_prompt(self, content: str, context: Dict[str, Any]) -> str:
        import json

        context_str = json.dumps(context, ensure_ascii=False, default=str)[
            :300
        ]
        return f"评估话题兴趣度:\n\n内容: {content}\n上下文: {context_str}\n\n返回JSON: topic, novelty(0-1), social_relevance(0-1), confidence, reasoning"

    def _parse_response(self, raw_response: str) -> InterestScore:
        import json

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return InterestScore()
        if not isinstance(parsed, dict):
            return InterestScore()
        return InterestScore(
            topic=str(parsed.get("topic", "")),
            novelty=self._safe_float(parsed.get("novelty", 0.5)),
            social_relevance=self._safe_float(
                parsed.get("social_relevance", 0.5)
            ),
            confidence=self._safe_float(parsed.get("confidence", 0.5)),
            reasoning=str(parsed.get("reasoning", ""))[:100],
        )

    def compute_combined_score(self, score: InterestScore) -> float:
        return (
            score.novelty * self._novelty_weight
            + score.social_relevance * self._relevance_weight
        )

    def is_high_interest(self, score: InterestScore) -> bool:
        combined = self.compute_combined_score(score)
        return combined >= self._high_interest_threshold

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (ValueError, TypeError):
            return 0.5

    def get_stats(self) -> Dict[str, Any]:
        return {
            "novelty_weight": self._novelty_weight,
            "relevance_weight": self._relevance_weight,
            "high_interest_threshold": self._high_interest_threshold,
        }


_interest_scorer: Optional[InterestScorer] = None


def get_interest_scorer(
    config_engine=None,
    model_client: Optional[ModelInterface] = None,
) -> InterestScorer:
    global _interest_scorer
    if _interest_scorer is None:
        _interest_scorer = InterestScorer(config_engine, model_client)
    elif model_client is not None:
        _interest_scorer.set_model_client(model_client)
    return _interest_scorer

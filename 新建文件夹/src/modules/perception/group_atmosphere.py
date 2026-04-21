import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine

logger = get_logger("群氛围")


@dataclass
class AtmosphereResult:
    atmosphere: str = "neutral"
    tension_level: float = 0.0
    hostility_level: float = 0.0
    activity_level: float = 0.5
    social_role: str = "observer"
    key_factors: List[str] = field(default_factory=list)
    confidence: float = 0.5
    reasoning: str = ""


class ModelInterface:
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError


class DefaultModelInterface(ModelInterface):
    async def generate(self, prompt: str) -> str:
        return '{"atmosphere": "neutral", "tension_level": 0.0, "hostility_level": 0.0, "activity_level": 0.5}'


class GroupAtmosphereAnalyzer:
    def __init__(
        self, config_engine=None, model_client: Optional[ModelInterface] = None
    ):
        self._config = config_engine or get_default_config_engine()
        self._model = model_client or DefaultModelInterface()
        self._history: List[Dict[str, Any]] = []
        self._max_history: int = 100
        self._decay_rate: float = 0.1
        self._load_config()

    def _load_config(self) -> None:
        self._max_history = int(
            self._config.get("group_atmosphere", "max_history", 100)
        )
        self._decay_rate = float(
            self._config.get("group_atmosphere", "decay_rate", 0.1)
        )

    def set_model_client(self, client: ModelInterface) -> None:
        self._model = client

    async def analyze(
        self, content: str, context: Dict[str, Any]
    ) -> AtmosphereResult:
        prompt = self._build_prompt(content, context)
        try:
            raw_response = await self._model.generate(prompt)
            result = self._parse_response(raw_response)
        except Exception as e:
            logger.debug("[群氛围] 分析失败: %s", e)
            result = AtmosphereResult(reasoning=str(e))
        self._update_history(result)
        return result

    def _build_prompt(self, content: str, context: Dict[str, Any]) -> str:
        import json

        context_str = json.dumps(context, ensure_ascii=False, default=str)[
            :500
        ]
        return f"分析群聊氛围:\n\n消息: {content}\n\n上下文: {context_str}\n\n返回JSON: atmosphere, tension_level, hostility_level, activity_level, key_factors, reasoning"

    def _parse_response(self, raw_response: str) -> AtmosphereResult:
        import json

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return AtmosphereResult()
        if not isinstance(parsed, dict):
            return AtmosphereResult()
        return AtmosphereResult(
            atmosphere=str(parsed.get("atmosphere", "neutral")),
            tension_level=self._safe_float(parsed.get("tension_level", 0.0)),
            hostility_level=self._safe_float(
                parsed.get("hostility_level", 0.0)
            ),
            activity_level=self._safe_float(parsed.get("activity_level", 0.5)),
            social_role=str(parsed.get("social_role", "observer")),
            key_factors=(
                parsed.get("key_factors", [])
                if isinstance(parsed.get("key_factors"), list)
                else []
            ),
            confidence=self._safe_float(parsed.get("confidence", 0.5)),
            reasoning=str(parsed.get("reasoning", ""))[:200],
        )

    def _update_history(self, result: AtmosphereResult) -> None:
        self._history.append(
            {
                "atmosphere": result.atmosphere,
                "tension_level": result.tension_level,
                "hostility_level": result.hostility_level,
                "activity_level": result.activity_level,
                "timestamp": time.time(),
            }
        )
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_trend(self, window_minutes: float = 30.0) -> Dict[str, Any]:
        if not self._history:
            return {
                "trend": "stable",
                "avg_tension": 0.0,
                "avg_hostility": 0.0,
            }
        cutoff = time.time() - window_minutes * 60
        recent = [h for h in self._history if h["timestamp"] >= cutoff]
        if not recent:
            return {
                "trend": "stable",
                "avg_tension": 0.0,
                "avg_hostility": 0.0,
            }
        avg_tension = sum(h["tension_level"] for h in recent) / len(recent)
        avg_hostility = sum(h["hostility_level"] for h in recent) / len(recent)
        if len(recent) >= 2:
            first_half = recent[: len(recent) // 2]
            second_half = recent[len(recent) // 2:]
            first_tension = sum(h["tension_level"] for h in first_half) / len(
                first_half
            )
            second_tension = sum(
                h["tension_level"] for h in second_half
            ) / len(second_half)
            if second_tension > first_tension + 0.1:
                trend = "escalating"
            elif second_tension < first_tension - 0.1:
                trend = "de_escalating"
            else:
                trend = "stable"
        else:
            trend = "stable"
        return {
            "trend": trend,
            "avg_tension": round(avg_tension, 3),
            "avg_hostility": round(avg_hostility, 3),
            "sample_count": len(recent),
        }

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (ValueError, TypeError):
            return 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "history_size": len(self._history),
            "max_history": self._max_history,
            "decay_rate": self._decay_rate,
        }


_group_atmosphere_analyzer: Optional[GroupAtmosphereAnalyzer] = None


def get_group_atmosphere_analyzer(
    config_engine=None,
    model_client: Optional[ModelInterface] = None,
) -> GroupAtmosphereAnalyzer:
    global _group_atmosphere_analyzer
    if _group_atmosphere_analyzer is None:
        _group_atmosphere_analyzer = GroupAtmosphereAnalyzer(
            config_engine, model_client
        )
    elif model_client is not None:
        _group_atmosphere_analyzer.set_model_client(model_client)
    return _group_atmosphere_analyzer

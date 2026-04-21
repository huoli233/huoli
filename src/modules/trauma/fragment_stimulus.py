import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.common.config.config_engine import ConfigEngine
from src.common.logger import get_logger
from src.modules.trauma.trauma_system import (
    TraumaFragment,
    TraumaSystem,
    get_trauma_system,
)

logger = get_logger("碎片刺激")


@dataclass
class StimulusEvaluationResult:
    """刺激评估结果"""

    should_record: bool = False
    semantic_context: str = ""
    emotional_charge: float = 0.0
    distortion_level: float = 0.0
    memory_clarity: float = 5.0
    confidence: float = 0.0
    fragment_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_record": self.should_record,
            "semantic_context": self.semantic_context,
            "emotional_charge": self.emotional_charge,
            "distortion_level": self.distortion_level,
            "memory_clarity": self.memory_clarity,
            "confidence": self.confidence,
            "fragment_id": self.fragment_id,
        }


class FragmentStimulusEvaluator:
    """创伤碎片刺激评估器

    由模型对「是否构成可记录刺激」及语义描述做定性输出，
    再由算法实例化 TraumaFragment。
    """

    def __init__(
        self,
        config_engine: Optional[ConfigEngine] = None,
        trauma_system: Optional[TraumaSystem] = None,
        model_client: Any = None,
    ):
        self._config = config_engine or ConfigEngine.get_instance()
        self._trauma = trauma_system or get_trauma_system()
        self._model_client = model_client
        self._last_eval_at: Dict[str, float] = {}
        self._load_config()

    def _load_config(self):
        """加载配置"""
        trauma_cfg = self._config.get("trauma", {})
        self._enabled = trauma_cfg.get("enabled", True)
        self._min_interval = float(
            trauma_cfg.get("fragment_eval_min_interval", 120.0)
        )
        self._max_content_len = int(trauma_cfg.get("max_content_length", 4000))
        self._max_context_len = int(trauma_cfg.get("max_context_length", 2000))

    def set_model_client(self, client: Any):
        """设置模型客户端"""
        self._model_client = client

    async def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[StimulusEvaluationResult]:
        """评估内容是否构成创伤刺激"""
        if not self._enabled:
            return None

        stream_key = (
            str(context.get("stream_id", "global") or "global")
            if context
            else "global"
        )
        now = time.time()
        last = self._last_eval_at.get(stream_key, 0.0)

        if self._min_interval > 0 and (now - last) < self._min_interval:
            return None

        self._last_eval_at[stream_key] = now

        truncated_content = content[: self._max_content_len]
        truncated_context = (
            str(context)[: self._max_context_len] if context else ""
        )

        prompt = self._build_evaluation_prompt(
            truncated_content, truncated_context
        )

        try:
            if self._model_client is None:
                return self._fallback_evaluation(content, context)
            response = await self._call_model(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.warning(f"模型调用异常: {e}")
            return self._fallback_evaluation(content, context)

        return None

    def _build_evaluation_prompt(self, content: str, context: str) -> str:
        prompt = f"""请评估以下内容是否构成创伤刺激。

内容: {content}
上下文: {context}
请以JSON格式返回结果:
{{
    "should_record": true/false,
    "semantic_context": "触发语义描述",
    "emotional_charge": 0.0-10.0,
    "distortion_level": 0.0-10.0,
    "memory_clarity": 0.0-10.0,
    "confidence": 0.0-1.0
}}"""
        return prompt

    async def _call_model(self, prompt: str) -> str:
        try:
            if hasattr(self._model_client, "generate"):
                return await self._model_client.generate(prompt)
            elif hasattr(self._model_client, "chat"):
                return await self._model_client.chat(prompt)
            else:
                logger.warning("模型客户端没有generate或chat方法")
                return ""
        except Exception as e:
            logger.warning(f"模型调用失败: {e}")
            return ""

    def _parse_response(
        self, response: str
    ) -> Optional[StimulusEvaluationResult]:
        """解析模型响应"""
        try:
            data = json.loads(response)
            should_record = bool(data.get("should_record", False))
            semantic_context = str(data.get("semantic_context", ""))
            emotional_charge = float(data.get("emotional_charge", 0.0))
            distortion_level = float(data.get("distortion_level", 0.0))
            memory_clarity = float(data.get("memory_clarity", 5.0))
            confidence = float(data.get("confidence", 0.0))
            fragment_id = str(data.get("fragment_id", ""))

            emotional_charge = max(0.0, min(10.0, emotional_charge))
            distortion_level = max(0.0, min(10.0, distortion_level))
            memory_clarity = max(0.0, min(10.0, memory_clarity))
            confidence = max(0.0, min(1.0, confidence))

            return StimulusEvaluationResult(
                should_record=should_record,
                semantic_context=semantic_context,
                emotional_charge=emotional_charge,
                distortion_level=distortion_level,
                memory_clarity=memory_clarity,
                confidence=confidence,
                fragment_id=fragment_id,
            )
        except json.JSONDecodeError as e:
            logger.debug(f"JSON解析失败: {e}")
            return None

    def _fallback_evaluation(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[StimulusEvaluationResult]:
        """备用评估逻辑（无模型时使用规则）"""
        negative_keywords = [
            "伤害",
            "痛苦",
            "恐惧",
            "绝望",
            "崩溃",
            "创伤",
            "阴影",
        ]
        intensity = 0.0
        for kw in negative_keywords:
            if kw in content:
                intensity += 1.0
        should_record = intensity >= 2.0

        if not should_record:
            return None
        charge = min(10.0, intensity)
        distortion = min(10.0, intensity * 0.5)
        clarity = 5.0
        return StimulusEvaluationResult(
            should_record=True,
            semantic_context=f"检测到负面关键词: intensity={intensity}",
            emotional_charge=charge,
            distortion_level=distortion,
            memory_clarity=clarity,
            confidence=0.7,
        )

    def _create_fragment(
        self, result: StimulusEvaluationResult
    ) -> Optional[TraumaFragment]:
        """创建创伤碎片"""
        if not result.should_record:
            return None
        fragment = TraumaFragment(
            fragment_id=result.fragment_id or uuid.uuid4().hex[:16],
            trigger_context=result.semantic_context,
            emotional_charge=result.emotional_charge,
            distortion_level=result.distortion_level,
            memory_clarity=result.memory_clarity,
        )
        self._trauma.add_fragment(fragment)
        return fragment

    def get_evaluation_stats(self) -> Dict[str, Any]:
        """获取评估统计"""
        return {
            "enabled": self._enabled,
            "last_evaluations": len(self._last_eval_at),
            "min_interval": self._min_interval,
        }


_fragment_stimulus_evaluator: Optional[FragmentStimulusEvaluator] = None


def get_fragment_stimulus_evaluator(
    config_engine: Optional[ConfigEngine] = None,
    trauma_system: Optional[TraumaSystem] = None,
    model_client: Any = None,
) -> FragmentStimulusEvaluator:
    global _fragment_stimulus_evaluator
    if _fragment_stimulus_evaluator is None:
        _fragment_stimulus_evaluator = FragmentStimulusEvaluator(
            config_engine, trauma_system, model_client
        )
    return _fragment_stimulus_evaluator

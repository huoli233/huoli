import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.chat.utils.prompt_builder import global_prompt_manager
from src.common.logger import get_logger
from src.modules.trauma.runtime_config import trauma_module_view
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
        trauma_system: Optional[TraumaSystem] = None,
        model_client: Any = None,
    ):
        self._trauma = trauma_system or get_trauma_system()
        self._model_client = model_client
        self._last_eval_at: Dict[str, float] = {}
        self._load_config()

    def _load_config(self):
        """加载配置"""
        trauma_cfg = trauma_module_view("trauma_fragment")
        self._enabled = trauma_cfg.get("enabled", True)
        self._min_interval = float(
            trauma_cfg.get("fragment_eval_min_interval", 120.0)
        )
        self._max_content_len = int(trauma_cfg.get("max_content_length", 4000))
        self._max_context_len = int(trauma_cfg.get("max_context_length", 2000))
        self._prompt_key = str(trauma_cfg.get("prompt_key", "trauma_shard_stimulus"))
        self._fallback_negative_keywords = trauma_cfg.get(
            "fallback_negative_keywords",
            ["伤害", "痛苦", "恐惧", "绝望", "崩溃", "创伤", "阴影"],
        )
        self._fallback_keyword_weight = float(trauma_cfg.get("fallback_keyword_weight", 1.0))
        self._fallback_record_threshold = float(trauma_cfg.get("fallback_record_threshold", 2.0))
        self._fallback_distortion_scale = float(trauma_cfg.get("fallback_distortion_scale", 0.5))
        self._fallback_memory_clarity = float(trauma_cfg.get("fallback_memory_clarity", 5.0))
        self._fallback_confidence = float(trauma_cfg.get("fallback_confidence", 0.7))
        self._score_cap = float(trauma_cfg.get("score_cap", 10.0))
        self._confidence_cap = float(trauma_cfg.get("confidence_cap", 1.0))
        self._fragment_id_chars = int(trauma_cfg.get("fragment_id_chars", 16))

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
        template = global_prompt_manager.get_prompt(self._prompt_key)
        if template is None:
            try:
                import src.chat.prompts.catalog  # noqa: F401

                template = global_prompt_manager.get_prompt(self._prompt_key)
            except Exception as exc:
                logger.debug(f"碎片刺激提示词注册表加载失败: {exc}")
        if template is not None:
            return template.render(content=content, context=context)
        return (
            "[碎片刺激评估]\n"
            f"输入内容：{content}\n"
            f"上下文：{context}\n\n"
            "请判断此内容是否构成可记录的创伤刺激。输出 JSON。"
        )

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

            emotional_charge = max(0.0, min(self._score_cap, emotional_charge))
            distortion_level = max(0.0, min(self._score_cap, distortion_level))
            memory_clarity = max(0.0, min(self._score_cap, memory_clarity))
            confidence = max(0.0, min(self._confidence_cap, confidence))

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
        intensity = 0.0
        for kw in self._fallback_negative_keywords:
            if kw in content:
                intensity += self._fallback_keyword_weight
        should_record = intensity >= self._fallback_record_threshold

        if not should_record:
            return None
        charge = min(self._score_cap, intensity)
        distortion = min(self._score_cap, intensity * self._fallback_distortion_scale)
        clarity = self._fallback_memory_clarity
        return StimulusEvaluationResult(
            should_record=True,
            semantic_context=f"检测到负面关键词: intensity={intensity}",
            emotional_charge=charge,
            distortion_level=distortion,
            memory_clarity=clarity,
            confidence=self._fallback_confidence,
        )

    def _create_fragment(
        self, result: StimulusEvaluationResult
    ) -> Optional[TraumaFragment]:
        """创建创伤碎片"""
        if not result.should_record:
            return None
        fragment = TraumaFragment(
            fragment_id=result.fragment_id or uuid.uuid4().hex[: self._fragment_id_chars],
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
    trauma_system: Optional[TraumaSystem] = None,
    model_client: Any = None,
) -> FragmentStimulusEvaluator:
    global _fragment_stimulus_evaluator
    if _fragment_stimulus_evaluator is None:
        _fragment_stimulus_evaluator = FragmentStimulusEvaluator(
            trauma_system, model_client
        )
    return _fragment_stimulus_evaluator

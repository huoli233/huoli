from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from src.common.logger import get_logger
from src.modules.recall.runtime_config import recall_dict, recall_module_view

logger = get_logger("拼写生成")


class TypoType(Enum):
    """打错字类型"""

    TYPO = "typo"
    SWAP = "swap"
    OMIT = "omit"
    DUPLICATE = "duplicate"
    WRONG_CHAR = "wrong_char"
    EXTRA_CHAR = "extra_char"


@dataclass
class TypoDecision:
    """打错字决策"""

    should_typo: bool = False
    typo_type: Optional[TypoType] = None
    error_content: str = ""
    original_content: str = ""
    confidence: float = 0.5
    reasoning: str = ""


class TypoGenerator:
    """打错字生成器

    模型驱动的打错字决策:
    - 收集多维度因素
    - 构建提示词（从模板加载，禁止硬编码）
    - 调用模型决策
    - 返回决策结果

    重要:
    - 所有判断由模型完成，禁止硬编码阈值
    - 所有提示词从模板文件加载，禁止在代码中硬编码
    """

    def __init__(
        self,
        model_client: Any = None,
        prompt_manager: Any = None,
        dimension_collector: Any = None,
    ):
        self._model = model_client
        self._prompts = prompt_manager
        self._dimensions = dimension_collector

        self._load_config()
        logger.info("打错字生成器初始化完成")

    def _load_config(self):
        """从配置加载参数"""
        config = recall_module_view("recall_typo")
        self._prompt_key = str(config.get("prompt_key", "typo_decision"))
        self._max_typo_length = int(config.get("max_typo_length", 50))
        self._typo_probability = float(config.get("probability", 0.05))
        self._generation_task = str(
            config.get("generation_task", "typo_generation")
        )
        self._random_confidence = float(config.get("random_confidence", 0.3))
        self._random_reasoning = str(
            config.get("random_reasoning", "随机打错字")
        )
        self._skip_reasoning = str(
            config.get("skip_reasoning", "随机检查通过")
        )
        self._extra_chars = str(config.get("extra_chars", "的了啊呢吧"))
        self._wrong_char_map = recall_dict(
            "recall_typo",
            "wrong_char_map",
            {
                "的": "地",
                "地": "的",
                "了": "乐",
                "是": "事",
                "在": "再",
                "再": "在",
                "有": "又",
                "我": "卧",
            },
        )

    def set_dimension_collector(self, collector: Any):
        """设置维度收集器"""
        self._dimensions = collector

    async def decide(
        self,
        original_content: str,
        user_id: str,
        channel_id: str,
        stream_id: str,
    ) -> TypoDecision:
        """决定是否打错字

        流程:
        1. 检查是否可以打错字
        2. 收集多维度因素
        3. 构建提示词（从模板加载）
        4. 调用模型决策
        5. 返回决策结果
        """
        if self._dimensions and not self._dimensions.can_typo(stream_id):
            return TypoDecision(
                should_typo=False,
                original_content=original_content,
                reasoning="最近打错字次数过多",
            )

        if len(original_content) > self._max_typo_length:
            return TypoDecision(
                should_typo=False,
                original_content=original_content,
                reasoning="消息过长，不适合打错字",
            )

        dimensions = None
        if self._dimensions:
            dimensions = await self._dimensions.collect(
                user_id=user_id,
                channel_id=channel_id,
                stream_id=stream_id,
                content=original_content,
            )

        if self._model is None:
            return self._random_typo_decision(original_content)

        prompt = self._build_prompt(dimensions, original_content)

        if not prompt:
            logger.warning("无法加载提示词模板，跳过打错字决策")
            return TypoDecision(
                should_typo=False,
                original_content=original_content,
                reasoning="提示词模板加载失败",
            )

        try:
            if hasattr(self._model, "analyze"):
                response = await self._model.analyze(
                    {
                        "task": "typo_decision",
                        "content": original_content,
                        "context": dimensions.to_dict() if dimensions else {},
                        "prompt": prompt,
                    }
                )
            elif hasattr(self._model, "generate"):
                response = await self._model.generate(prompt)
            else:
                return self._random_typo_decision(original_content)

            decision = self._parse_response(response, original_content)

            if decision.should_typo and self._dimensions:
                self._dimensions.record_typo(stream_id)
                logger.info(
                    f"打错字决策: 类型={
                        decision.typo_type}, 理由={
                        decision.reasoning}"
                )

            return decision

        except Exception as e:
            logger.error(f"打错字决策失败: {e}")
            return TypoDecision(
                should_typo=False,
                original_content=original_content,
                reasoning=f"决策异常: {e}",
            )

    def _random_typo_decision(self, original_content: str) -> TypoDecision:
        """随机打错字决策（降级方案）"""
        import random

        if random.random() < self._typo_probability:
            typo_type = random.choice(list(TypoType))
            error_content = self._simple_typo(original_content, typo_type)
            return TypoDecision(
                should_typo=True,
                typo_type=typo_type,
                error_content=error_content,
                original_content=original_content,
                confidence=self._random_confidence,
                reasoning=self._random_reasoning,
            )
        return TypoDecision(
            should_typo=False,
            original_content=original_content,
            reasoning=self._skip_reasoning,
        )

    def _build_prompt(
        self,
        dimensions: Any,
        original_content: str,
    ) -> str:
        """构建提示词

        从模板加载，禁止在此处硬编码提示词
        如果模板加载失败，返回空字符串，由调用方决定如何处理
        """
        if self._prompts is not None:
            try:
                template = self._prompts.get(self._prompt_key)
                if template:
                    main_personality = ""
                    if self._dimensions and hasattr(
                        self._dimensions, "get_main_personality"
                    ):
                        main_personality = (
                            self._dimensions.get_main_personality()
                        )

                    if hasattr(template, "render"):
                        return template.render(
                            original_content=original_content,
                            main_personality=main_personality,
                            **(dimensions.to_dict() if dimensions else {}),
                        )
                    return str(template)
            except Exception as e:
                logger.error(f"加载提示词模板失败: {e}")

        return ""

    def _parse_response(
        self,
        response: Any,
        original_content: str,
    ) -> TypoDecision:
        """解析模型响应"""
        if hasattr(response, "to_dict"):
            data = response.to_dict()
        elif isinstance(response, dict):
            data = response
        else:
            data = {}

        should_typo = data.get("should_typo", False)
        typo_type_str = data.get("typo_type")

        typo_type = None
        if typo_type_str:
            try:
                typo_type = TypoType(typo_type_str)
            except ValueError:
                typo_type = TypoType.TYPO

        error_content = data.get("error_content", "")
        if should_typo and not error_content and typo_type:
            error_content = self._simple_typo(original_content, typo_type)

        return TypoDecision(
            should_typo=should_typo,
            typo_type=typo_type,
            error_content=error_content,
            original_content=original_content,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
        )

    async def generate_error_content(
        self,
        original: str,
        typo_type: TypoType,
        dimensions: Any = None,
    ) -> str:
        """生成错误内容

        由模型生成具体的错误内容
        """
        if self._model is None:
            return self._simple_typo(original, typo_type)

        try:
            prompt = f"""请为以下消息生成一个自然的打字错误。

原始消息: {original}
错误类型: {typo_type.value}

要求:
1. 错误要自然，像真人打字时的失误
2. 不要改变原意太多
3. 返回错误后的内容即可

错误内容:"""

            if hasattr(self._model, "analyze"):
                response = await self._model.analyze(
                    {
                        "task": self._generation_task,
                        "content": original,
                        "context": {"typo_type": typo_type.value},
                        "prompt": prompt,
                    }
                )
            elif hasattr(self._model, "generate"):
                response = await self._model.generate(prompt)
            else:
                return self._simple_typo(original, typo_type)

            if hasattr(response, "content"):
                return response.content
            elif isinstance(response, dict):
                return response.get("error_content", original)
            elif isinstance(response, str):
                return response.strip()
            return original

        except Exception as e:
            logger.debug(f"生成错误内容失败: {e}")
            return self._simple_typo(original, typo_type)

    def _simple_typo(self, content: str, typo_type: TypoType) -> str:
        """简单打错字（降级方案）

        注意：这是降级方案，正常情况应由模型生成
        """
        if not content or len(content) < 2:
            return content

        if typo_type == TypoType.SWAP and len(content) >= 2:
            chars = list(content)
            chars[0], chars[1] = chars[1], chars[0]
            return "".join(chars)

        elif typo_type == TypoType.OMIT:
            return content[:-1] if len(content) > 1 else content

        elif typo_type == TypoType.DUPLICATE:
            return content + content[-1] if content else content

        elif typo_type == TypoType.EXTRA_CHAR:
            import random

            pos = random.randint(0, len(content))
            return (
                content[:pos]
                + random.choice(self._extra_chars)
                + content[pos:]
            )

        elif typo_type == TypoType.WRONG_CHAR:
            for k, v in self._wrong_char_map.items():
                if k in content:
                    return content.replace(k, v, 1)
            return content

        return content

    def get_typo_stats(self) -> Dict[str, Any]:
        """获取打错字统计"""
        return {
            "max_typo_length": self._max_typo_length,
            "typo_probability": self._typo_probability,
            "has_model": self._model is not None,
            "has_dimensions": self._dimensions is not None,
        }


_typo_generator_instance: Optional[TypoGenerator] = None


def get_typo_generator(
    model_client: Any = None,
    prompt_manager: Any = None,
    dimension_collector: Any = None,
) -> TypoGenerator:
    """获取打错字生成器单例"""
    global _typo_generator_instance
    if _typo_generator_instance is None:
        _typo_generator_instance = TypoGenerator(
            model_client, prompt_manager, dimension_collector
        )
    return _typo_generator_instance

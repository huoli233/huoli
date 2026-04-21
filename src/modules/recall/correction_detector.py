import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.common.config.config_engine import ConfigEngine
from src.common.logger import get_logger

logger = get_logger("纠正检测")


@dataclass
class CorrectionEvent:
    """纠正事件"""

    stream_id: str = ""
    bot_msg_id: str = ""
    bot_content: str = ""
    sent_at: float = 0.0
    user_reply: str = ""
    detected_at: float = 0.0
    need_recall: bool = False
    reason: str = ""
    confidence: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class CorrectionDetector:
    """纠正检测器

    核心理念：
    - 判断优先交给模型完成
    - 模型不可用时自动降级为不触发
    - 不使用关键词匹配，让模型自主理解用户意图
    """

    def __init__(
        self,
        config_engine: Optional[ConfigEngine] = None,
        prompt_manager: Any = None,
        model_client: Any = None,
    ):
        self._config = config_engine or ConfigEngine.get_instance()
        self._prompts = prompt_manager
        self._model = model_client

        self._history: Dict[str, List[CorrectionEvent]] = {}
        self._history_limit = 12
        self._timeout_sec = 150.0
        self._load_config()
        logger.info("纠正检测器初始化完成")

    def _load_config(self) -> None:
        """从配置加载参数"""
        try:
            self._history_limit = self._config.get(
                "correction", "history_limit", 12
            )
            self._timeout_sec = self._config.get(
                "correction", "timeout_seconds", 150.0
            )
        except Exception as exc:
            logger.debug(f"加载纠正检测配置失败: {exc}")

    def log_bot_send(self, stream_id: str, msg_id: str, content: str) -> None:
        """记录智能体发送的消息"""
        if stream_id not in self._history:
            self._history[stream_id] = []

        event = CorrectionEvent(
            stream_id=stream_id,
            bot_msg_id=msg_id,
            bot_content=content,
            sent_at=time.time(),
        )
        self._history[stream_id].append(event)

        if len(self._history[stream_id]) > self._history_limit:
            self._history[stream_id] = self._history[stream_id][
                -self._history_limit:
            ]

        logger.debug(f"记录机器人消息: {msg_id}")

    async def check_correction(
        self,
        stream_id: str,
        user_msg: str,
    ) -> Optional[CorrectionEvent]:
        """检查用户消息是否构成纠正

        让模型自主理解用户意图，而不是关键词匹配
        """
        history = self._history.get(stream_id) or []
        if not history:
            return None

        recent = history[-1]
        elapsed = time.time() - recent.sent_at
        if elapsed > self._timeout_sec:
            return None

        is_correction, confidence, reason = await self._evaluate_with_model(
            recent.bot_content, user_msg
        )

        recent.user_reply = user_msg
        recent.detected_at = time.time()
        recent.need_recall = bool(is_correction)
        recent.confidence = float(confidence)
        recent.reason = reason

        if not recent.need_recall:
            return None
        return recent

    async def _evaluate_with_model(
        self,
        bot_content: str,
        user_message: str,
    ) -> Tuple[bool, float, str]:
        """调用模型执行纠正判断

        让模型自主理解用户是否表达了纠正意图
        """
        if self._model is None:
            return False, 0.0, "model_unavailable"

        prompt = await self._build_prompt(bot_content, user_message)

        try:
            if hasattr(self._model, "generate"):
                raw = await self._model.generate(prompt)
            elif hasattr(self._model, "analyze"):
                raw = await self._model.analyze({"prompt": prompt})
            else:
                return False, 0.0, "unsupported_model_interface"

            data = self._parse_json(raw or "")
            if not isinstance(data, dict):
                return False, 0.0, "json_parse_failed"

            is_correction = bool(data.get("is_correction", False))
            confidence = float(data.get("confidence", 0.0) or 0.0)
            reasoning = str(data.get("reasoning", ""))
            return is_correction, confidence, reasoning

        except Exception as exc:
            logger.error(f"纠正检测模型调用失败: {exc}")
            return False, 0.0, "model_error"

    async def _build_prompt(self, bot_content: str, user_message: str) -> str:
        """构建纠正判断提示词

        从模板加载，如果失败则使用内置模板
        """
        if self._prompts is not None:
            try:
                rendered = self._prompts.get(
                    "correction_detection",
                    main_personality="理性、克制、不过度触发",
                    bot_content=bot_content,
                    user_message=user_message,
                )
                if rendered:
                    return rendered
            except Exception as exc:
                logger.debug(f"纠正提示词模板加载失败: {exc}")

        return f"""你是一个智能体，需要判断用户的消息是否表达了纠正意图。

## 任务

分析用户的消息，判断是否包含纠正意图。

纠正意图包括但不限于：
1. 用户说"你理解错了"、"你搞错了"等直接纠正
2. 用户说"不是这个意思"、"我的意思是"等澄清
3. 用户对智能体的回复表示不满或困惑
4. 用户指出智能体说错了或理解偏差

## 智能体最近发送的消息

{bot_content}

## 用户回复

{user_message}

---

请分析用户的消息是否表达了纠正意图。

返回 JSON 格式结果：
{{"is_correction": boolean, "confidence": float, "reasoning": "string - 简要说明判断理由"}}"""

    def _parse_json(self, text: str) -> Optional[dict]:
        """解析JSON"""
        import json

        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return None

    def get_last_bot_message(
        self, stream_id: str
    ) -> Optional[CorrectionEvent]:
        """获取最后一条机器人消息"""
        history = self._history.get(stream_id) or []
        if history:
            return history[-1]
        return None

    def clear_history(self, stream_id: Optional[str] = None):
        """清除历史"""
        if stream_id:
            self._history.pop(stream_id, None)
        else:
            self._history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = sum(len(v) for v in self._history.values())
        corrections = sum(
            1
            for events in self._history.values()
            for e in events
            if e.need_recall
        )
        return {
            "total_messages": total,
            "corrections_detected": corrections,
            "active_streams": len(self._history),
        }


_correction_detector: Optional[CorrectionDetector] = None


def get_correction_detector(
    config_engine: Optional[ConfigEngine] = None,
    prompt_manager: Any = None,
    model_client: Any = None,
) -> CorrectionDetector:
    """获取纠正检测器单例"""
    global _correction_detector
    if _correction_detector is None:
        _correction_detector = CorrectionDetector(
            config_engine, prompt_manager, model_client
        )
    return _correction_detector

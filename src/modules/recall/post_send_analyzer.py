import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Awaitable

from src.common.config.config_engine import ConfigEngine
from src.common.logger import get_logger
from src.common.task_utils import safe_create_task
from src.modules.recall.dimension_collector import (
    DimensionFactors,
    DimensionCollector,
)

logger = get_logger("发送后分析")


class RecallReason(Enum):
    """撤回原因"""

    TYPO = "typo"
    AWKWARD = "awkward"
    INAPPROPRIATE = "inappropriate"
    REGRET = "regret"
    MISUNDERSTOOD = "misunderstood"
    TOO_HASTY = "too_hasty"
    USER_NEGATIVE_REACTION = "user_negative_reaction"
    NULL = "null"


class AfterRecallAction(Enum):
    """撤回后行为"""

    CORRECT = "correct"
    SUPPLEMENT = "supplement"
    CASUAL_CORRECT = "casual_correct"
    SILENT = "silent"
    REGRET = "regret"
    SHY = "shy"
    APOLOGIZE = "apologize"
    REDIRECT = "redirect"
    SELF_DEPRECATE = "self_deprecate"


@dataclass
class RecallDecision:
    """撤回决策"""

    should_recall: bool = False
    recall_reason: Optional[RecallReason] = None
    delay_seconds: float = 5.0
    after_action: AfterRecallAction = AfterRecallAction.CORRECT
    confidence: float = 0.5
    reasoning: str = ""


@dataclass
class PendingRecall:
    """待处理撤回"""

    msg_id: str = ""
    stream_id: str = ""
    channel_id: str = ""
    user_id: str = ""
    sent_content: str = ""
    original_content: str = ""
    scheduled_at: float = 0.0
    recall_at: float = 0.0
    recalled: bool = False
    completed: bool = False
    recall_decision: Optional[RecallDecision] = None
    task: Optional[asyncio.Task] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class PostSendAnalyzer:
    """发送后分析器

    在消息发送后的时间窗口内持续分析是否需要撤回

    时间窗口: 5-110秒
    分析间隔: 由配置决定

    职责边界:
    - 本模块只负责决策
    - 内容生成由其他模块完成
    """

    def __init__(
        self,
        config_engine: Optional[ConfigEngine] = None,
        model_client: Any = None,
        prompt_manager: Any = None,
        dimension_collector: Optional[DimensionCollector] = None,
    ):
        self._config = config_engine or ConfigEngine.get_instance()
        self._model = model_client
        self._prompts = prompt_manager
        self._dimensions = dimension_collector or DimensionCollector(
            self._config
        )

        self._pending: Dict[str, PendingRecall] = {}
        self._background_tasks: set = set()

        self._min_delay = 5.0
        self._max_delay = 110.0
        self._analysis_interval = 3.0
        self._prompt_key = "recall_decision"

        self._load_config()
        logger.info(
            f"发送后分析器初始化完成 | 时间窗口: {self._min_delay}-{self._max_delay}秒"
        )

    def _load_config(self):
        """从配置加载参数"""
        self._min_delay = self._config.get("recall", "min_delay_seconds", 5.0)
        self._max_delay = self._config.get(
            "recall", "max_delay_seconds", 110.0
        )
        self._analysis_interval = self._config.get(
            "recall", "analysis_interval_seconds", 3.0
        )

    async def start_analysis(
        self,
        pending: PendingRecall,
        on_recall: Optional[Callable[[PendingRecall], Awaitable[None]]] = None,
    ) -> PendingRecall:
        """启动发送后分析"""
        self._pending[pending.msg_id] = pending

        pending.task = asyncio.create_task(
            self._analysis_loop(pending, on_recall)
        )
        self._background_tasks.add(pending.task)
        pending.task.add_done_callback(self._background_tasks.discard)

        return pending

    async def _analysis_loop(
        self,
        pending: PendingRecall,
        on_recall: Optional[Callable[[PendingRecall], Awaitable[None]]] = None,
    ):
        """分析循环 - 在时间窗口内持续分析"""
        start_time = pending.scheduled_at

        while not pending.completed:
            elapsed = time.time() - start_time

            if elapsed >= self._max_delay:
                pending.completed = True
                logger.debug(f"撤回分析超时 | msg_id={pending.msg_id}")
                break

            try:
                decision = await self._make_decision(pending, elapsed)

                if decision.should_recall:
                    pending.recall_decision = decision
                    pending.recall_at = time.time() + decision.delay_seconds

                    if self._dimensions:
                        self._dimensions.record_recall(pending.stream_id)

                    logger.info(
                        f"撤回决策: 原因={decision.recall_reason}, "
                        f"延迟={decision.delay_seconds}秒, "
                        f"行为={decision.after_action}"
                    )

                    if on_recall:
                        safe_create_task(on_recall(pending), name="recall_callback")
                    break

            except Exception as e:
                logger.error(f"撤回分析失败: {e}")

            await asyncio.sleep(self._analysis_interval)

    async def _make_decision(
        self,
        pending: PendingRecall,
        elapsed: float,
    ) -> RecallDecision:
        """模型决策是否撤回"""
        if not self._dimensions:
            return RecallDecision(should_recall=False)

        dimensions = await self._dimensions.collect(
            user_id=pending.user_id,
            channel_id=pending.channel_id,
            stream_id=pending.stream_id,
            content=pending.sent_content,
            time_elapsed=elapsed,
        )

        if self._model is None:
            return self._rule_based_decision(dimensions, elapsed)

        prompt = self._build_prompt(
            dimensions, pending.sent_content, pending.original_content
        )

        try:
            if hasattr(self._model, "analyze"):
                response = await self._model.analyze(
                    {
                        "task": "recall_decision",
                        "content": pending.sent_content,
                        "context": dimensions.to_dict(),
                        "prompt": prompt,
                    }
                )
            elif hasattr(self._model, "generate"):
                response = await self._model.generate(prompt)
            else:
                return RecallDecision(should_recall=False)

            return self._parse_response(response)

        except Exception as e:
            logger.error(f"撤回决策失败: {e}")
            return RecallDecision(should_recall=False)

    def _rule_based_decision(
        self, dimensions: DimensionFactors, elapsed: float
    ) -> RecallDecision:
        """基于规则的决策（降级方案）"""
        if dimensions.recent_recall_count >= 3:
            return RecallDecision(
                should_recall=False, reasoning="最近撤回次数过多"
            )

        if elapsed > 60:
            return RecallDecision(
                should_recall=False, reasoning="已过撤回窗口"
            )

        return RecallDecision(should_recall=False, reasoning="规则检查通过")

    def _build_prompt(
        self,
        dimensions: DimensionFactors,
        sent_content: str,
        original_content: str,
    ) -> str:
        """构建提示词"""
        if self._prompts is not None:
            try:
                template = self._prompts.get(self._prompt_key)
                if template:
                    if hasattr(template, "render"):
                        return template.render(
                            **dimensions.to_dict(),
                            sent_content=sent_content,
                            original_content=original_content,
                        )
                    return str(template)
            except Exception as e:
                logger.debug(f"加载撤回决策模板失败: {e}")

        return self._build_default_prompt(
            dimensions, sent_content, original_content
        )

    def _build_default_prompt(
        self,
        dimensions: DimensionFactors,
        sent_content: str,
        original_content: str,
    ) -> str:
        """默认提示词（降级方案）"""
        return f"""你是一个模拟人类消息撤回行为的智能体。请综合多维度因素决定是否撤回消息。

## 多维度因素

### 社交维度
- 社交分: {dimensions.social_value}
- 关系深度: {dimensions.relationship_depth}
- 信任值: {dimensions.trust_value}
- 交互次数: {dimensions.interaction_count}

### 心理维度
- 情绪状态: {dimensions.emotional_state}

### 场景维度
- 是否私聊: {dimensions.is_private}
- 群氛围: {dimensions.group_atmosphere}

### 时间维度
- 时间段: {dimensions.time_of_day}
- 发送后经过: {dimensions.time_elapsed} 秒
- 最近撤回次数: {dimensions.recent_recall_count}

## 已发送内容

{sent_content}

## 任务

请综合以上所有维度因素，决定是否应该撤回这条消息。

返回JSON格式结果:
{{"should_recall": boolean, "recall_reason": "typo|awkward|inappropriate|regret|null", "delay_seconds": float, "after_action": "correct|silent|regret", "confidence": float, "reasoning": "string"}}"""

    def _parse_response(self, response: Any) -> RecallDecision:
        """解析模型响应"""
        if hasattr(response, "to_dict"):
            data = response.to_dict()
        elif isinstance(response, dict):
            data = response
        else:
            data = {}

        should_recall = data.get("should_recall", False)
        recall_reason_str = data.get("recall_reason")

        recall_reason = None
        if recall_reason_str:
            try:
                recall_reason = RecallReason(recall_reason_str)
            except ValueError:
                recall_reason = RecallReason.TYPO

        after_action_str = data.get("after_action", "correct")
        try:
            after_action = AfterRecallAction(after_action_str)
        except ValueError:
            after_action = AfterRecallAction.CORRECT

        delay = float(data.get("delay_seconds", 5.0))
        delay = max(self._min_delay, min(self._max_delay, delay))

        return RecallDecision(
            should_recall=should_recall,
            recall_reason=recall_reason,
            delay_seconds=delay,
            after_action=after_action,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
        )

    async def handle_user_reaction(
        self,
        msg_id: str,
        user_reaction: str,
    ) -> Optional[RecallDecision]:
        """处理用户反应 - 可能触发紧急撤回"""
        pending = self._pending.get(msg_id)
        if not pending or pending.recalled or pending.completed:
            return None

        decision = await self._make_decision(
            pending,
            time.time() - pending.scheduled_at,
        )

        if decision.should_recall:
            pending.recall_decision = decision
            pending.recall_at = time.time()
            if self._dimensions:
                self._dimensions.record_recall(pending.stream_id)

        return decision if decision.should_recall else None

    def get_pending(self, msg_id: str) -> Optional[PendingRecall]:
        """获取待处理撤回"""
        return self._pending.get(msg_id)

    def cancel_pending(self, msg_id: str) -> bool:
        """取消待处理撤回"""
        pending = self._pending.get(msg_id)
        if pending and pending.task:
            pending.task.cancel()
            pending.completed = True
            return True
        return False

    def cleanup_completed(self):
        """清理已完成的待处理"""
        now = time.time()
        expired = [
            msg_id
            for msg_id, pending in self._pending.items()
            if pending.completed
            or (now - pending.scheduled_at) > self._max_delay * 2
        ]
        for msg_id in expired:
            del self._pending[msg_id]

        if expired:
            logger.debug(f"清理过期待处理撤回: {len(expired)}条")

    async def shutdown(self) -> None:
        """关闭分析器并取消所有后台任务"""
        for pending in list(self._pending.values()):
            pending.completed = True

        for task in list(self._background_tasks):
            task.cancel()

        self._background_tasks.clear()


_post_send_analyzer_instance: Optional[PostSendAnalyzer] = None


def get_post_send_analyzer(
    config_engine: Optional[ConfigEngine] = None,
    model_client: Any = None,
    prompt_manager: Any = None,
    dimension_collector: Optional[DimensionCollector] = None,
) -> PostSendAnalyzer:
    """获取发送后分析器单例"""
    global _post_send_analyzer_instance
    if _post_send_analyzer_instance is None:
        _post_send_analyzer_instance = PostSendAnalyzer(
            config_engine, model_client, prompt_manager, dimension_collector
        )
    return _post_send_analyzer_instance

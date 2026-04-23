import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.common.task_utils import safe_create_task
from src.modules.recall.runtime_config import recall_module_view

logger = get_logger("行为学习")


@dataclass
class LearnedPattern:
    """学习到的模式"""

    stream_id: str = ""
    relation_stage: str = ""
    action_type: str = ""
    style_hint: str = ""
    confidence: float = 0.5
    sample_count: int = 0
    learned_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "relation_stage": self.relation_stage,
            "action_type": self.action_type,
            "style_hint": self.style_hint,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "learned_at": self.learned_at,
        }


class SelfBehaviorLearner:
    """自我行为学习器

    从自我行为记录中提炼可复用表达模式
    """

    def __init__(
        self,
        model_client: Any = None,
        prompt_manager: Any = None,
        self_awareness: Any = None,
        memory_core: Any = None,
    ):
        self._model = model_client
        self._prompts = prompt_manager
        self._self_awareness = self_awareness
        self._memory_core = memory_core
        self._quick_learn_buffer: Dict[str, List[Dict[str, Any]]] = {}

        self._last_learn_ts_by_stream: Dict[str, float] = {}
        self._patterns_by_stream: Dict[str, List[LearnedPattern]] = {}

        self._load_config()
        logger.info("自我行为学习器初始化完成")

    def _load_config(self) -> None:
        """从配置加载参数"""
        config = recall_module_view("recall_self_behavior")
        self._enabled = bool(config.get("enabled", True))
        self._min_samples = max(
            2, int(config.get("min_samples", 3))
        )
        self._cooldown_seconds = max(
            30.0,
            float(config.get("cooldown_seconds", 90.0)),
        )
        self._max_patterns_per_stream = max(
            5,
            int(config.get("max_patterns_per_stream", 20)),
        )
        self._model_task = str(
            config.get("model_task", "self_behavior_learn")
        )
        self._default_action_type = str(
            config.get("default_action_type", "send_reply")
        )
        self._default_style_hint = str(
            config.get("default_style_hint", "保持简洁、自然、贴近上下文")
        )
        self._style_hint_chars = int(config.get("style_hint_chars", 60))
        self._confidence_floor = float(config.get("confidence_floor", 0.35))
        self._confidence_ceiling = float(
            config.get("confidence_ceiling", 0.95)
        )
        self._precipitate_confidence_threshold = float(
            config.get("precipitate_confidence_threshold", 0.5)
        )
        self._precipitate_style_min_chars = int(
            config.get("precipitate_style_min_chars", 4)
        )
        self._quick_learn_soft_limit = int(
            config.get("quick_learn_soft_limit", 15)
        )
        self._quick_learn_keep = int(config.get("quick_learn_keep", 8))
        self._quick_learn_trigger_count = int(
            config.get("quick_learn_trigger_count", 3)
        )

    def bind_memory_core(self, memory_core: Any) -> None:
        """绑定记忆核心"""
        self._memory_core = memory_core

    def bind_self_awareness(self, self_awareness: Any) -> None:
        """绑定自我意识模块"""
        self._self_awareness = self_awareness

    async def capture_event(
        self,
        stream_id: str,
        action_type: str,
        content: str,
        result: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录行为并在条件满足时触发学习"""
        if not self._enabled:
            return

        relation_stage = self._resolve_relation_stage(context)
        now = time.time()

        if self._self_awareness:
            self._self_awareness.record_action(
                stream_id=stream_id,
                action_type=action_type,
                target=relation_stage,
                content=content,
                result=result,
                extra={
                    "relation_stage": relation_stage,
                    "capture_source": "self_behavior_learner",
                },
            )

        try:
            if not self._should_trigger_learning(stream_id, now):
                return

            learned = await self._learn_for_stream(stream_id)
            if learned is None:
                return

            self._last_learn_ts_by_stream[stream_id] = now
            self._append_pattern(stream_id, learned)

            if self._memory_core is not None and hasattr(
                self._memory_core, "store_message"
            ):
                await self._memory_core.store_message(
                    channel_id=stream_id,
                    content=f"[自我行为学习]{learned.style_hint}",
                    user_id="self_behavior_learner",
                    importance=min(1.0, max(0.1, learned.confidence)),
                    is_bot=True,
                    extra_metadata={
                        "memory_type": "self_behavior_pattern",
                        "relation_stage": learned.relation_stage,
                        "action_type": learned.action_type,
                        "sample_count": learned.sample_count,
                    },
                )

            logger.info(
                f"自我行为学习: stream={stream_id}, stage={
                    learned.relation_stage}, "
                f"action={
                    learned.action_type}, confidence={
                    learned.confidence:.2f}"
            )
        except Exception as e:
            logger.debug(f"自我行为学习降级: {e}")

    def get_style_hints(
        self,
        stream_id: str,
        relation_stage: Optional[str] = None,
        limit: int = 3,
    ) -> List[str]:
        """获取风格提示"""
        patterns = self._patterns_by_stream.get(stream_id, [])
        if not patterns:
            return []

        selected: List[LearnedPattern] = []
        if relation_stage:
            selected = [
                p for p in patterns if p.relation_stage == relation_stage
            ]
        if not selected:
            selected = patterns

        selected = sorted(
            selected, key=lambda p: (p.confidence, p.learned_at), reverse=True
        )
        return [p.style_hint for p in selected[: max(1, limit)]]

    def get_latest_patterns(
        self, stream_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """获取最新学习模式"""
        patterns = self._patterns_by_stream.get(stream_id, [])
        ordered = sorted(patterns, key=lambda p: p.learned_at, reverse=True)
        return [p.to_dict() for p in ordered[: max(1, limit)]]

    def _resolve_relation_stage(
        self, context: Optional[Dict[str, Any]]
    ) -> str:
        """解析关系阶段"""
        if not context:
            return "unknown"
        stage = str(context.get("relation_stage", "")).strip()
        return stage or "unknown"

    def _should_trigger_learning(self, stream_id: str, now: float) -> bool:
        """判断是否应该触发学习"""
        last_ts = self._last_learn_ts_by_stream.get(stream_id, 0.0)
        if now - last_ts < self._cooldown_seconds:
            return False

        if not self._self_awareness:
            return False

        recent_actions = self._self_awareness.get_recent_actions(
            stream_id=stream_id, limit=self._min_samples
        )
        return len(recent_actions) >= self._min_samples

    async def _learn_for_stream(
        self, stream_id: str
    ) -> Optional[LearnedPattern]:
        """为流学习模式"""
        if not self._self_awareness:
            return None

        actions = self._self_awareness.get_recent_actions(
            stream_id=stream_id, limit=max(self._min_samples, 20)
        )
        if len(actions) < self._min_samples:
            return None

        if self._model is not None:
            try:
                payload = [
                    {
                        "action_type": action.action_type,
                        "target": action.target,
                        "content": (
                            action.content[:120] if action.content else ""
                        ),
                        "result": action.result,
                        "timestamp": action.timestamp,
                        "extra": action.extra,
                    }
                    for action in actions
                ]

                if hasattr(self._model, "generate_raw"):
                    raw = await self._model.generate_raw(
                        self._model_task,
                        stream_id=stream_id,
                        actions=str(payload),
                    )
                    parsed = self._parse_structured_result(raw)
                    if parsed is not None:
                        return parsed
            except Exception as e:
                logger.debug(f"模型学习失败，降级统计学习: {e}")

        return self._fallback_learn(stream_id, actions)

    def _parse_structured_result(self, raw: str) -> Optional[LearnedPattern]:
        """解析结构化结果"""
        import json

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
            else:
                return None
        except json.JSONDecodeError:
            return None

        try:
            return LearnedPattern(
                stream_id=str(data.get("stream_id", "")),
                relation_stage=str(data.get("relation_stage", "unknown")),
                action_type=str(
                    data.get("action_type", self._default_action_type)
                ),
                style_hint=str(data.get("style_hint", "")),
                confidence=float(data.get("confidence", 0.5)),
                sample_count=int(data.get("sample_count", 0)),
                learned_at=time.time(),
            )
        except Exception as exc:
            logger.warning(f"行为学习解析异常: {exc}")
            return None

    def _fallback_learn(
        self, stream_id: str, actions: List[Any]
    ) -> Optional[LearnedPattern]:
        """降级学习"""
        if not actions:
            return None

        succeeded = [
            a
            for a in actions
            if str(getattr(a, "result", "")).lower() == "success"
        ]
        base = succeeded if succeeded else actions

        stage_counter: Dict[str, int] = {}
        action_counter: Dict[str, int] = {}
        for item in base:
            extra = getattr(item, "extra", {}) or {}
            stage = str(
                extra.get("relation_stage", getattr(item, "target", "unknown"))
            )
            stage_counter[stage] = stage_counter.get(stage, 0) + 1
            action_type = getattr(item, "action_type", "unknown")
            action_counter[action_type] = (
                action_counter.get(action_type, 0) + 1
            )

        top_stage = (
            max(stage_counter.items(), key=lambda x: x[1])[0]
            if stage_counter
            else "unknown"
        )
        top_action = (
            max(action_counter.items(), key=lambda x: x[1])[0]
            if action_counter
            else self._default_action_type
        )

        examples = [
            getattr(a, "content", "").strip()
            for a in base
            if getattr(a, "content", "").strip()
        ]
        style_hint = (
            examples[-1][: self._style_hint_chars]
            if examples
            else self._default_style_hint
        )

        confidence = min(
            self._confidence_ceiling,
            max(
                self._confidence_floor,
                len(base) / max(float(self._min_samples), 1.0),
            ),
        )
        return LearnedPattern(
            stream_id=stream_id,
            relation_stage=top_stage,
            action_type=top_action,
            style_hint=style_hint,
            confidence=confidence,
            sample_count=len(base),
            learned_at=time.time(),
        )

    def _append_pattern(self, stream_id: str, pattern: LearnedPattern) -> None:
        """追加模式"""
        if not pattern.stream_id:
            pattern.stream_id = stream_id

        bucket = self._patterns_by_stream.setdefault(stream_id, [])
        bucket.append(pattern)
        if len(bucket) > self._max_patterns_per_stream:
            self._patterns_by_stream[stream_id] = bucket[
                -self._max_patterns_per_stream:
            ]
        try:
            self._precipitate_to_expression_table(stream_id, pattern)
        except Exception as _pe:
            logger.debug(f"风格沉淀到Expression表失败: {_pe}")

    def _precipitate_to_expression_table(
        self, stream_id: str, pattern: LearnedPattern
    ) -> None:
        """将学习到的风格模式沉淀到Expression表

        让 StylePicker 能在后续的风格选择中发现并使用这些模式。
        只沉淀置信度>0.5的模式，避免低质量污染。
        """
        if pattern.confidence < self._precipitate_confidence_threshold:
            return
        if (
            not pattern.style_hint
            or len(pattern.style_hint.strip())
            < self._precipitate_style_min_chars
        ):
            return
        try:
            from src.common.database.database_model import Expression

            _scene_tag = f"{pattern.relation_stage}_{pattern.action_type}"
            _existing = (
                Expression.select()
                .where(Expression.chat_id == stream_id)
                .where(Expression.situation.contains(_scene_tag[:30]))
                .limit(1)
            )
            if _existing.exists():
                _rec = _existing.get()
                _rec.count = (_rec.count or 0) + 1
                _rec.style = pattern.style_hint[:200]
                _rec.modify_ts = time.time()
                _rec.save()
            else:
                Expression.create(
                    chat_id=stream_id,
                    situation=_scene_tag[:100],
                    style=pattern.style_hint[:200],
                    count=1,
                    rejected=False,
                    birth_ts=time.time(),
                    modify_ts=time.time(),
                )
            logger.debug(
                f"风格沉淀: stream={stream_id} scene={_scene_tag[:20]} "
                f"hint={pattern.style_hint[:30]} conf={pattern.confidence:.2f}"
            )
        except Exception as _e:
            logger.debug(f"Expression表写入异常: {_e}")

    def quick_capture(
        self,
        stream_id: str,
        content: str,
        context_summary: str = "",
    ) -> None:
        """快速捕获一条回复样本(不需要等cooldown)

        用于高频场景下的轻量级学习，
        缓存到buffer中达到阈值后批量触发学习。
        """
        if not content or not content.strip():
            return
        buf = self._quick_learn_buffer.setdefault(stream_id, [])
        buf.append({
            "content": content.strip()[:120],
            "context": context_summary[:80],
            "ts": time.time(),
        })
        if len(buf) > self._quick_learn_soft_limit:
            self._quick_learn_buffer[stream_id] = buf[-self._quick_learn_keep:]
        if len(buf) >= self._quick_learn_trigger_count and self._should_trigger_learning(stream_id, time.time()):
            safe_create_task(self.capture_event(
                stream_id=stream_id,
                action_type="reply",
                content=content,
                result="success",
                context={"relation_stage": context_summary[:40] or "unknown"},
            ), name="quick_learn_capture")

    def clear_patterns(self, stream_id: Optional[str] = None):
        """清除学习模式"""
        if stream_id:
            self._patterns_by_stream.pop(stream_id, None)
        else:
            self._patterns_by_stream.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_patterns = sum(len(v) for v in self._patterns_by_stream.values())
        return {
            "enabled": self._enabled,
            "total_patterns": total_patterns,
            "active_streams": len(self._patterns_by_stream),
            "min_samples": self._min_samples,
            "cooldown_seconds": self._cooldown_seconds,
        }


_self_behavior_learner_instance: Optional[SelfBehaviorLearner] = None


def get_self_behavior_learner(
    model_client: Any = None,
    prompt_manager: Any = None,
    self_awareness: Any = None,
    memory_core: Any = None,
) -> SelfBehaviorLearner:
    """获取自我行为学习器单例"""
    global _self_behavior_learner_instance
    if _self_behavior_learner_instance is None:
        _self_behavior_learner_instance = SelfBehaviorLearner(
            model_client,
            prompt_manager,
            self_awareness,
            memory_core,
        )
    return _self_behavior_learner_instance

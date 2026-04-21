import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import Counter
from src.common.logger import get_logger

logger = get_logger("dynamic_context")


@dataclass
class MessageImportance:
    """消息重要性评估"""

    message_id: str
    content: str
    importance_score: float = 0.5
    relevance_score: float = 0.5
    topic_relevance: float = 0.5
    entity_density: float = 0.0
    is_question: bool = False
    is_response_trigger: bool = False
    fragments: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)


@dataclass
class TopicContext:
    """话题上下文"""

    topic_id: str
    topic_fragments: Set[str]
    start_time: float
    last_active: float
    message_count: int = 0
    importance_sum: float = 0.0

    def is_active(self, now: float, timeout: float = 300.0) -> bool:
        return (now - self.last_active) < timeout


@dataclass
class ContextWindowConfig:
    """上下文窗口配置"""

    min_window: int = 5
    max_window: int = 50
    default_window: int = 15
    compression_threshold: int = 30
    importance_threshold: float = 0.3
    topic_timeout: float = 300.0


class DynamicContextWindow:
    """
    动态上下文窗口系统

    根据对话复杂度自动调整窗口大小，
    通过重要性加权保留关键上下文。
    """

    _instance: Optional["DynamicContextWindow"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[ContextWindowConfig] = None):
        if self._initialized:
            return
        self._initialized = True
        self._config = config or ContextWindowConfig()
        self._channel_windows: Dict[str, List[MessageImportance]] = {}
        self._channel_topics: Dict[str, List[TopicContext]] = {}
        self._channel_current_size: Dict[str, int] = {}
        self._channel_summaries: Dict[str, str] = {}
        logger.info(
            f"[动态上下文] 初始化完成，窗口范围: {self._config.min_window}-{self._config.max_window}"
        )

    def add_message(
        self,
        channel_id: str,
        message_id: str,
        content: str,
        speaker: str = "",
        is_bot: bool = False,
    ) -> MessageImportance:
        """添加消息到上下文窗口"""
        if channel_id not in self._channel_windows:
            self._channel_windows[channel_id] = []
            self._channel_current_size[channel_id] = (
                self._config.default_window
            )
        importance = self._evaluate_importance(
            message_id, content, speaker, is_bot
        )
        self._update_topics(channel_id, content, importance)
        self._channel_windows[channel_id].append(importance)
        self._adjust_window_size(channel_id)
        if (
            len(self._channel_windows[channel_id])
            > self._channel_current_size[channel_id]
        ):
            self._compress_window(channel_id)
        return importance

    def _evaluate_importance(
        self,
        message_id: str,
        content: str,
        speaker: str,
        is_bot: bool,
    ) -> MessageImportance:
        """评估消息重要性，依赖结构信号与文本密度"""
        importance = MessageImportance(
            message_id=message_id,
            content=content,
        )
        question_marks = content.count("?") + content.count("？")
        exclamations = content.count("!") + content.count("！")
        line_breaks = content.count("\n")
        importance.is_question = (
            question_marks > 0
            or content.rstrip().endswith(("吗", "呢", "么", "吧"))
        )
        if importance.is_question:
            importance.importance_score += 0.2
        if exclamations >= 2:
            importance.importance_score += 0.1
        if len(content) > 100:
            importance.importance_score += 0.1
        elif len(content) < 10:
            importance.importance_score -= 0.1
        if line_breaks >= 2:
            importance.importance_score += 0.05
        fragments = self._extract_fragments(content)
        importance.fragments = fragments
        importance.entity_density = len(fragments) / max(
            1, len(self._normalize_text(content)) // 2
        )
        if importance.entity_density > 0.3:
            importance.importance_score += 0.1
        importance.importance_score = max(
            0.0, min(1.0, importance.importance_score)
        )
        return importance

    def _extract_fragments(self, content: str) -> List[str]:
        """提取文本片段，用于话题追踪与轻量相关性。"""
        normalized = self._normalize_text(content)
        if not normalized:
            return []
        if len(normalized) <= 2:
            return [normalized]
        grams = {
            normalized[index: index + 2]
            for index in range(len(normalized) - 1)
        }
        return list(grams)

    @staticmethod
    def _normalize_text(content: str) -> str:
        if not content:
            return ""
        normalized = content.lower()
        for char in [
            " ",
            "\n",
            "\t",
            "，",
            "。",
            "！",
            "？",
            "、",
            "；",
            "：",
            '"',
            "'",
            "（",
            "）",
            "(",
            ")",
            "[",
            "]",
            "【",
            "】",
        ]:
            normalized = normalized.replace(char, "")
        return normalized.strip()

    def _update_topics(
        self,
        channel_id: str,
        content: str,
        importance: MessageImportance,
    ) -> None:
        """更新话题追踪"""
        if channel_id not in self._channel_topics:
            self._channel_topics[channel_id] = []
        now = time.time()
        fragments = set(importance.fragments)
        if not fragments:
            return
        matched = False
        for topic in self._channel_topics[channel_id]:
            overlap = len(fragments & topic.topic_fragments)
            if overlap >= 2:
                topic.topic_fragments.update(fragments)
                topic.last_active = now
                topic.message_count += 1
                topic.importance_sum += importance.importance_score
                matched = True
                break
        if not matched:
            topic_id = f"topic_{int(now * 1000)}"
            self._channel_topics[channel_id].append(
                TopicContext(
                    topic_id=topic_id,
                    topic_fragments=fragments,
                    start_time=now,
                    last_active=now,
                    message_count=1,
                    importance_sum=importance.importance_score,
                )
            )
        self._channel_topics[channel_id] = [
            t
            for t in self._channel_topics[channel_id]
            if t.is_active(now, self._config.topic_timeout)
        ]

    def _adjust_window_size(self, channel_id: str) -> None:
        """根据复杂度调整窗口大小"""
        if channel_id not in self._channel_windows:
            return
        messages = self._channel_windows[channel_id]
        if len(messages) < 5:
            return
        complexity = self._calculate_complexity(channel_id)
        current_size = self._channel_current_size.get(
            channel_id, self._config.default_window
        )
        if complexity > 0.7:
            new_size = min(self._config.max_window, int(current_size * 1.5))
        elif complexity < 0.3:
            new_size = max(self._config.min_window, int(current_size * 0.8))
        else:
            new_size = current_size
        self._channel_current_size[channel_id] = new_size
        if new_size != current_size:
            logger.debug(
                f"[动态上下文] {channel_id[:8]} 窗口调整: {current_size} -> {new_size} (复杂度: {complexity:.2f})"
            )

    def _calculate_complexity(self, channel_id: str) -> float:
        """计算对话复杂度"""
        messages = self._channel_windows.get(channel_id, [])
        if len(messages) < 5:
            return 0.5
        recent = messages[-20:]
        topic_count = len(self._channel_topics.get(channel_id, []))
        avg_importance = sum(m.importance_score for m in recent) / len(recent)
        question_ratio = sum(1 for m in recent if m.is_question) / len(recent)
        entity_density = sum(m.entity_density for m in recent) / len(recent)
        complexity = (
            min(1.0, topic_count / 5.0) * 0.3
            + avg_importance * 0.25
            + question_ratio * 0.25
            + min(1.0, entity_density) * 0.2
        )
        return max(0.0, min(1.0, complexity))

    def _compress_window(self, channel_id: str) -> None:
        """压缩上下文窗口"""
        messages = self._channel_windows.get(channel_id, [])
        if len(messages) <= self._config.compression_threshold:
            return
        target_size = self._channel_current_size.get(
            channel_id, self._config.default_window
        )
        scored = [
            (m, m.importance_score + m.relevance_score) for m in messages
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        keep_ids = {m.message_id for m, _ in scored[:target_size]}
        compressed = [m for m in messages if m.message_id in keep_ids]
        if len(compressed) < len(messages):
            summary = self._generate_summary(messages[: -len(compressed)])
            self._channel_summaries[channel_id] = summary
            logger.debug(
                f"[动态上下文] {channel_id[:8]} 压缩窗口: {len(messages)} -> {len(compressed)}"
            )
        self._channel_windows[channel_id] = compressed

    def _generate_summary(self, messages: List[MessageImportance]) -> str:
        """生成压缩摘要"""
        if not messages:
            return ""
        all_fragments = []
        for m in messages:
            all_fragments.extend(m.fragments)
        fragment_counts = Counter(all_fragments)
        top_fragments = [frag for frag, _ in fragment_counts.most_common(10)]
        return f"[历史话题片段: {', '.join(top_fragments[:5])}]"

    def get_context(
        self,
        channel_id: str,
        query: str = "",
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """获取上下文"""
        messages = self._channel_windows.get(channel_id, [])
        if not messages:
            return []
        window_size = max_messages or self._channel_current_size.get(
            channel_id, self._config.default_window
        )
        if query:
            query_fragments = set(self._extract_fragments(query))
            for m in messages:
                overlap = len(query_fragments & set(m.fragments))
                density_bonus = min(
                    0.2,
                    len(m.fragments) / max(5, len(query_fragments) or 1) * 0.1,
                )
                m.relevance_score = min(
                    1.0,
                    overlap / max(1, len(query_fragments)) * 0.6
                    + m.importance_score * 0.3
                    + density_bonus,
                )
            messages.sort(key=lambda x: x.relevance_score, reverse=True)
            messages = messages[:window_size]
            messages.sort(key=lambda x: x.message_id)
        else:
            messages = messages[-window_size:]
        result = []
        for m in messages:
            result.append(
                {
                    "message_id": m.message_id,
                    "content": m.content,
                    "importance": m.importance_score,
                    "relevance": m.relevance_score,
                    "is_question": m.is_question,
                    "fragments": m.fragments[:5],
                }
            )
        return result

    def get_context_with_summary(
        self,
        channel_id: str,
        query: str = "",
        max_messages: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """获取上下文（包含摘要）"""
        summary = self._channel_summaries.get(channel_id, "")
        context = self.get_context(channel_id, query, max_messages)
        return summary, context

    def get_window_stats(self, channel_id: str) -> Dict[str, Any]:
        """获取窗口统计"""
        messages = self._channel_windows.get(channel_id, [])
        topics = self._channel_topics.get(channel_id, [])
        return {
            "current_size": len(messages),
            "max_size": self._channel_current_size.get(
                channel_id, self._config.default_window
            ),
            "complexity": self._calculate_complexity(channel_id),
            "active_topics": len(topics),
            "has_summary": channel_id in self._channel_summaries,
        }

    def clear_channel(self, channel_id: str) -> None:
        """清除频道上下文"""
        self._channel_windows.pop(channel_id, None)
        self._channel_topics.pop(channel_id, None)
        self._channel_current_size.pop(channel_id, None)
        self._channel_summaries.pop(channel_id, None)


_dynamic_context: Optional[DynamicContextWindow] = None


def get_dynamic_context() -> DynamicContextWindow:
    """获取动态上下文窗口单例"""
    global _dynamic_context
    if _dynamic_context is None:
        _dynamic_context = DynamicContextWindow()
    return _dynamic_context

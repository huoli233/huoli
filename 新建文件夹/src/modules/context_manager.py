import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContextMessage:
    """稳定的上下文消息对象，供主链统一裁剪前情。"""

    message_id: str = ""
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    user_id: str = ""
    user_name: str = ""
    relevance_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """频道级上下文容器。"""

    channel_id: str = ""
    messages: List[ContextMessage] = field(default_factory=list)
    max_messages: int = 20
    max_age_seconds: float = 3600.0
    last_updated: float = field(default_factory=time.time)

    def add_message(self, message: ContextMessage) -> None:
        self.messages.append(message)
        self.last_updated = time.time()
        self._cleanup_expired()
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_recent_messages(self, limit: int = 10) -> List[ContextMessage]:
        return self.messages[-limit:] if self.messages else []

    def search_relevant(
        self, query: str, limit: int = 5
    ) -> List[ContextMessage]:
        text = str(query or "").strip().lower()
        if not text:
            return []

        query_terms = {term for term in text.split() if term}
        scored: List[tuple[float, ContextMessage]] = []
        now = time.time()
        for message in self.messages:
            content = str(message.content or "").lower()
            content_terms = {term for term in content.split() if term}
            overlap = len(query_terms & content_terms)
            if overlap <= 0 and text not in content:
                continue
            score = float(overlap or 1)
            age = max(0.0, now - float(message.timestamp or now))
            decay = max(0.15, 1.0 - age / self.max_age_seconds)
            scored.append((score * decay, message))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    def _cleanup_expired(self) -> None:
        cutoff = time.time() - self.max_age_seconds
        self.messages = [
            msg
            for msg in self.messages
            if float(msg.timestamp or 0.0) >= cutoff
        ]


class ContextManager:
    """全局上下文管理器。"""

    def __init__(self):
        self._contexts: Dict[str, ConversationContext] = {}
        self._default_max_messages = 24
        self._default_max_age = 3600.0

    def get_context(self, channel_id: str) -> ConversationContext:
        context = self._contexts.get(channel_id)
        if context is None:
            context = ConversationContext(
                channel_id=channel_id,
                max_messages=self._default_max_messages,
                max_age_seconds=self._default_max_age,
            )
            self._contexts[channel_id] = context
        return context

    def add_message(self, channel_id: str, message: ContextMessage) -> None:
        self.get_context(channel_id).add_message(message)

    def get_recent_context(
        self, channel_id: str, limit: int = 10
    ) -> List[ContextMessage]:
        return self.get_context(channel_id).get_recent_messages(limit)

    def search_context(
        self, channel_id: str, query: str, limit: int = 5
    ) -> List[ContextMessage]:
        return self.get_context(channel_id).search_relevant(query, limit)


_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


def create_context_message(
    content: str,
    user_id: str,
    user_name: str = "",
    message_id: str = "",
    timestamp: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> ContextMessage:
    return ContextMessage(
        message_id=message_id,
        content=content,
        timestamp=timestamp or time.time(),
        user_id=user_id,
        user_name=user_name,
        metadata=metadata or {},
    )

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _compact_context_text(text: str) -> str:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return ""
    normalized = re.sub(r"\s+", "", normalized)
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)


def _extract_context_terms(text: str) -> List[str]:
    raw = str(text or "").strip().lower()
    if not raw:
        return []

    terms: List[str] = []
    seen = set()
    for part in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9_]{2,}", raw):
        term = part.strip("_")
        if len(term) < 2 or term.isdigit() or term in seen:
            continue
        seen.add(term)
        terms.append(term[:16])
        if len(terms) >= 10:
            break
    return terms


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
            self.messages = self.messages[-self.max_messages :]

    def get_recent_messages(self, limit: int = 10) -> List[ContextMessage]:
        return self.messages[-limit:] if self.messages else []

    def search_relevant(self, query: str, limit: int = 5) -> List[ContextMessage]:
        text = str(query or "").strip().lower()
        if not text:
            return []

        query_terms = _extract_context_terms(text)
        compact_query = _compact_context_text(text)
        scored: List[tuple[float, float, ContextMessage]] = []
        now = time.time()
        for message in self.messages:
            content = str(message.content or "")
            compact_content = _compact_context_text(content)
            if not compact_content:
                continue

            matched_terms = [term for term in query_terms if term in compact_content]
            exact_match = bool(compact_query and compact_query in compact_content)
            if not matched_terms and not exact_match:
                continue

            coverage = len(set(matched_terms)) / max(1, len(query_terms))
            match_weight = sum(min(len(term), 8) for term in set(matched_terms))
            char_overlap = 0.0
            if compact_query:
                shared_chars = len(set(compact_query) & set(compact_content))
                char_overlap = shared_chars / max(1, len(set(compact_query)))
            age = max(0.0, now - float(message.timestamp or 0.0))
            decay = max(0.15, 1.0 - age / self.max_age_seconds)
            score = 0.0
            if exact_match:
                score += 1.2
            score += coverage * 0.9
            score += min(0.45, match_weight * 0.05)
            score += char_overlap * 0.2
            score += decay * 0.15
            message.relevance_score = round(score, 4)
            scored.append((score, float(message.timestamp or 0.0), message))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:limit]]

    def _cleanup_expired(self) -> None:
        cutoff = time.time() - self.max_age_seconds
        self.messages = [msg for msg in self.messages if float(msg.timestamp or 0.0) >= cutoff]


class ContextManager:
    """全局上下文管理器。"""

    def __init__(self):
        self._contexts: Dict[str, ConversationContext] = {}
        self._default_max_messages = 10
        self._default_max_age = 1800.0

    def get_context(self, channel_id: str) -> ConversationContext:
        context = self._contexts.get(channel_id)
        if context is None:
            context = ConversationContext(
                channel_id=channel_id,
                max_messages=self._default_max_messages,
                max_age_seconds=self._default_max_age,
            )
            self._contexts[channel_id] = context
        else:
            if context.max_messages < self._default_max_messages:
                context.max_messages = self._default_max_messages
            if context.max_age_seconds < self._default_max_age:
                context.max_age_seconds = self._default_max_age
        return context

    def add_message(self, channel_id: str, message: ContextMessage) -> None:
        self.get_context(channel_id).add_message(message)

    def get_recent_context(self, channel_id: str, limit: int = 10) -> List[ContextMessage]:
        context = self.get_context(channel_id)
        bounded_limit = max(1, min(int(limit or self._default_max_messages), int(context.max_messages or 1)))
        return context.get_recent_messages(bounded_limit)

    def search_context(self, channel_id: str, query: str, limit: int = 5) -> List[ContextMessage]:
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

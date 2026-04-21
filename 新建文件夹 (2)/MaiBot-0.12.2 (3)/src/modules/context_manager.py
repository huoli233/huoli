import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from src.common.logger import get_logger

logger = get_logger("context_mgr")


@dataclass
class ContextMessage:
    message_id: str = ""
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    user_id: str = ""
    user_name: str = ""
    message_type: str = "text"
    relevance_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    stream_id: str = ""
    messages: List[ContextMessage] = field(default_factory=list)
    max_messages: int = 20
    max_age_seconds: float = 3600.0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def add_message(self, message: ContextMessage):
        self.messages.append(message)
        self.last_updated = time.time()
        self._cleanup_expired()
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_recent_messages(self, limit: int = 10) -> List[ContextMessage]:
        return self.messages[-limit:] if self.messages else []

    def get_messages_by_time(self, since_ts: float) -> List[ContextMessage]:
        return [msg for msg in self.messages if msg.timestamp > since_ts]

    def search_relevant(self, query: str, limit: int = 5) -> List[ContextMessage]:
        if not query:
            return []
        scored = []
        query_lower = query.lower()
        for msg in self.messages:
            content_lower = msg.content.lower()
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            common_words = query_words.intersection(content_words)
            score = len(common_words) / len(query_words) if common_words else 0.0
            time_diff = time.time() - msg.timestamp
            time_decay = max(0.1, 1.0 - (time_diff / 3600.0))
            scored.append((score * time_decay, msg))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [msg for _, msg in scored[:limit]]

    def _cleanup_expired(self):
        cutoff = time.time() - self.max_age_seconds
        self.messages = [msg for msg in self.messages if msg.timestamp > cutoff]

    def clear(self):
        self.messages.clear()
        self.last_updated = time.time()

    def get_summary(self) -> Dict[str, Any]:
        if not self.messages:
            return {"total_messages": 0, "time_span": 0, "last_activity": None}
        time_span = self.messages[-1].timestamp - self.messages[0].timestamp
        last_activity = datetime.fromtimestamp(self.messages[-1].timestamp)
        return {
            "total_messages": len(self.messages),
            "time_span_seconds": time_span,
            "last_activity": last_activity.isoformat(),
            "users_involved": len(set(msg.user_id for msg in self.messages)),
        }


class ContextManager:
    def __init__(self):
        self._contexts: Dict[str, ConversationContext] = {}
        self._max_contexts = 50
        self._default_max_messages = 20
        self._default_max_age = 3600
        self._cleanup_interval = 300
        self._last_cleanup = time.time()

    def get_context(self, stream_id: str) -> ConversationContext:
        if stream_id not in self._contexts:
            context = ConversationContext(
                stream_id=stream_id,
                max_messages=self._default_max_messages,
                max_age_seconds=self._default_max_age
            )
            self._contexts[stream_id] = context
            if len(self._contexts) > self._max_contexts:
                self._cleanup_old_contexts()
        return self._contexts[stream_id]

    def add_message(self, stream_id: str, message: ContextMessage):
        context = self.get_context(stream_id)
        context.add_message(message)
        if time.time() - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired_contexts()
            self._last_cleanup = time.time()

    def get_recent_context(self, stream_id: str, limit: int = 10) -> List[ContextMessage]:
        return self.get_context(stream_id).get_recent_messages(limit)

    def search_context(self, stream_id: str, query: str, limit: int = 5) -> List[ContextMessage]:
        return self.get_context(stream_id).search_relevant(query, limit)

    def clear_context(self, stream_id: str):
        if stream_id in self._contexts:
            self._contexts[stream_id].clear()

    def get_context_summary(self, stream_id: str) -> Dict[str, Any]:
        return self.get_context(stream_id).get_summary()

    def merge_contexts(self, target_stream: str, source_streams: List[str]):
        target_context = self.get_context(target_stream)
        for source in source_streams:
            if source in self._contexts and source != target_stream:
                for msg in self._contexts[source].messages:
                    msg.metadata["merged_from"] = source
                    target_context.add_message(msg)

    def _cleanup_old_contexts(self):
        if not self._contexts:
            return
        sorted_contexts = sorted(self._contexts.items(), key=lambda x: x[1].last_updated)
        if len(sorted_contexts) > self._max_contexts:
            to_remove = sorted_contexts[:len(sorted_contexts) - self._max_contexts]
            for sid, _ in to_remove:
                del self._contexts[sid]

    def _cleanup_expired_contexts(self):
        expired = [sid for sid, ctx in self._contexts.items() if time.time() - ctx.last_updated > 86400]
        for sid in expired:
            del self._contexts[sid]

    def get_stats(self) -> Dict[str, Any]:
        total_messages = sum(len(ctx.messages) for ctx in self._contexts.values())
        active = len(self._contexts)
        return {
            "total_contexts": active, "total_messages": total_messages,
            "avg_messages_per_context": total_messages / active if active > 0 else 0,
            "max_contexts_limit": self._max_contexts,
        }


_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


def create_context_message(
    content: str, user_id: str, user_name: str = "",
    message_type: str = "text", message_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ContextMessage:
    if message_id is None:
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        message_id = f"{user_id}_{int(time.time())}_{content_hash}"
    return ContextMessage(
        message_id=message_id, content=content, user_id=user_id,
        user_name=user_name, message_type=message_type, metadata=metadata or {}
    )

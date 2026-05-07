import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from src.common.logger import get_logger

logger = get_logger("content_state_tracker")


class ContentStatus(Enum):
    """内容处理状态"""

    NEW = "new"
    PROCESSING = "processing"
    PROCESSED = "processed"
    IGNORED = "ignored"
    DEFERRED = "deferred"
    FAILED = "failed"


class IgnoreReason(Enum):
    """忽略原因"""

    LOW_INTEREST = "low_interest"
    SPAM = "spam"
    ALREADY_REPLIED = "already_replied"
    OVER_PROCESSED = "over_processed"
    DUPLICATE = "duplicate"
    OFF_TOPIC = "off_topic"
    INAPPROPRIATE = "inappropriate"
    USER_REQUESTED = "user_requested"


@dataclass
class ContentProcessingState:
    """内容处理状态"""

    content_hash: str
    content_preview: str
    first_seen: float
    last_processed: float = 0.0
    processing_count: int = 0
    status: ContentStatus = ContentStatus.NEW
    action_taken: str = ""
    ignore_reason: str = ""
    importance_score: float = 0.5
    interest_score: float = 0.5
    relevance_score: float = 0.5
    response_id: str = ""
    response_quality: float = 0.0
    channel_id: str = ""
    user_id: str = ""
    topic_tags: List[str] = field(default_factory=list)
    processing_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ContentDecision:
    """内容处理决策"""

    should_process: bool
    reason: str
    suggested_action: str = ""
    confidence: float = 0.5
    state: Optional[ContentProcessingState] = None


class ContentStateTracker:
    """
    内容状态追踪器

    追踪内容的处理状态，决定是否应该响应。
    """

    _instance: Optional["ContentStateTracker"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        max_cache_size: int = 2000,
        similarity_threshold: float = 0.85,
        max_processing_count: int = 3,
    ):
        if self._initialized:
            return
        self._initialized = True
        self._max_cache_size = max_cache_size
        self._similarity_threshold = similarity_threshold
        self._max_processing_count = max_processing_count
        self._state_cache: Dict[str, ContentProcessingState] = {}
        self._hash_index: Dict[str, str] = {}
        self._channel_index: Dict[str, List[str]] = {}
        self._user_index: Dict[str, List[str]] = {}
        self._interest_keywords: Dict[str, float] = {}
        self._ignore_patterns: List[str] = []
        logger.info(f"[内容状态追踪] 初始化完成，缓存大小: {max_cache_size}")

    def track_content(
        self,
        content: str,
        channel_id: str = "",
        user_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContentProcessingState:
        """追踪内容"""
        content_hash = self._hash_content(content, channel_id)
        if content_hash in self._state_cache:
            state = self._state_cache[content_hash]
            state.last_processed = time.time()
            self._add_index_entry(content_hash, channel_id, user_id)
            return state
        state = ContentProcessingState(
            content_hash=content_hash,
            content_preview=content[:100],
            first_seen=time.time(),
            channel_id=channel_id,
            user_id=user_id,
        )
        if metadata:
            state.importance_score = metadata.get("importance", 0.5)
            state.interest_score = metadata.get("interest", 0.5)
            state.topic_tags = metadata.get("topics", [])
        self._state_cache[content_hash] = state
        self._hash_index[content_hash] = content_hash
        self._add_index_entry(content_hash, channel_id, user_id)
        self._cleanup_cache()
        return state

    def should_process(
        self,
        content: str,
        channel_id: str = "",
        user_id: str = "",
    ) -> ContentDecision:
        """判断是否应该处理该内容"""
        content_hash = self._hash_content(content, channel_id)
        state = self._state_cache.get(content_hash)
        if state is None:
            state = self.track_content(content, channel_id, user_id)
            return ContentDecision(
                should_process=True,
                reason="new_content",
                suggested_action="process",
                confidence=0.8,
                state=state,
            )
        if state.status == ContentStatus.IGNORED:
            return ContentDecision(
                should_process=False,
                reason=f"already_ignored:{state.ignore_reason}",
                suggested_action="skip",
                confidence=0.9,
                state=state,
            )
        if state.status == ContentStatus.PROCESSED:
            if state.processing_count >= self._max_processing_count:
                return ContentDecision(
                    should_process=False,
                    reason="over_processed",
                    suggested_action="skip",
                    confidence=0.85,
                    state=state,
                )
            if state.response_quality > 0.7:
                return ContentDecision(
                    should_process=False,
                    reason="already_well_replied",
                    suggested_action="skip",
                    confidence=0.8,
                    state=state,
                )
        if state.status == ContentStatus.PROCESSING:
            return ContentDecision(
                should_process=False,
                reason="currently_processing",
                suggested_action="wait",
                confidence=0.7,
                state=state,
            )
        interest = self._evaluate_interest(content)
        if interest < 0.3:
            return ContentDecision(
                should_process=False,
                reason="low_interest",
                suggested_action="ignore",
                confidence=0.6,
                state=state,
            )
        return ContentDecision(
            should_process=True,
            reason="re_process_allowed",
            suggested_action="process",
            confidence=0.7,
            state=state,
        )

    def mark_processing(
        self,
        content: str,
        channel_id: str = "",
        user_id: str = "",
    ) -> bool:
        """标记内容正在处理"""
        content_hash = self._hash_content(content, channel_id)
        state = self._state_cache.get(content_hash)
        if state:
            state.status = ContentStatus.PROCESSING
            state.last_processed = time.time()
            self._add_index_entry(content_hash, channel_id, user_id)
            return True
        return False

    def mark_processed(
        self,
        content: str,
        action: str = "replied",
        response_id: str = "",
        quality: float = 0.5,
        channel_id: str = "",
        user_id: str = "",
    ) -> bool:
        """标记内容已处理"""
        content_hash = self._hash_content(content, channel_id)
        state = self._state_cache.get(content_hash)
        if state:
            state.status = ContentStatus.PROCESSED
            state.processing_count += 1
            state.last_processed = time.time()
            state.action_taken = action
            state.response_id = response_id
            state.response_quality = quality
            self._add_index_entry(content_hash, channel_id, user_id)
            state.processing_history.append(
                {
                    "action": action,
                    "response_id": response_id,
                    "quality": quality,
                    "timestamp": time.time(),
                }
            )
            logger.debug(
                f"[内容状态追踪] 标记已处理: {content_hash[:8]}, 动作: {action}, 质量: {quality:.2f}"
            )
            return True
        return False

    def mark_ignored(
        self,
        content: str,
        reason: IgnoreReason = IgnoreReason.LOW_INTEREST,
        channel_id: str = "",
        user_id: str = "",
    ) -> bool:
        """标记内容已忽略"""
        content_hash = self._hash_content(content, channel_id)
        state = self._state_cache.get(content_hash)
        if state:
            state.status = ContentStatus.IGNORED
            state.ignore_reason = reason.value
            state.last_processed = time.time()
            self._add_index_entry(content_hash, channel_id, user_id)
            state.processing_history.append(
                {
                    "action": "ignored",
                    "reason": reason.value,
                    "timestamp": time.time(),
                }
            )
            logger.debug(
                f"[内容状态追踪] 标记已忽略: {content_hash[:8]}, 原因: {reason.value}"
            )
            return True
        return False

    def mark_deferred(
        self,
        content: str,
        reason: str = "",
        channel_id: str = "",
        user_id: str = "",
    ) -> bool:
        """标记内容已延迟处理"""
        content_hash = self._hash_content(content, channel_id)
        state = self._state_cache.get(content_hash)
        if state:
            state.status = ContentStatus.DEFERRED
            state.last_processed = time.time()
            self._add_index_entry(content_hash, channel_id, user_id)
            state.processing_history.append(
                {
                    "action": "deferred",
                    "reason": reason,
                    "timestamp": time.time(),
                }
            )
            return True
        return False

    def _evaluate_interest(self, content: str) -> float:
        """评估内容兴趣度，尽量依赖结构信号而不是固定词表"""
        score = 0.5
        content_lower = content.lower()
        for keyword, weight in self._interest_keywords.items():
            if keyword in content_lower:
                score += weight * 0.1
        question_marks = content.count("?") + content.count("？")
        exclamations = content.count("!") + content.count("！")
        pauses = content.count(".") + content.count("。") + content.count("…")
        length = len(content.strip())
        if question_marks > 0:
            score += 0.15
        if exclamations >= 2:
            score += 0.1
        if length >= 40:
            score += 0.1
        elif length <= 4:
            score -= 0.1
        if pauses == 0 and length <= 3:
            score -= 0.1
        return max(0.0, min(1.0, score))

    def add_interest_keyword(self, keyword: str, weight: float = 1.0) -> None:
        """添加兴趣关键词"""
        self._interest_keywords[keyword] = max(-1.0, min(1.0, weight))

    def add_ignore_pattern(self, pattern: str) -> None:
        """添加忽略模式"""
        self._ignore_patterns.append(pattern)

    def get_channel_stats(self, channel_id: str) -> Dict[str, Any]:
        """获取频道统计"""
        hashes = self._channel_index.get(channel_id, [])
        if not hashes:
            return {"total": 0}
        states = [
            self._state_cache.get(h) for h in hashes if h in self._state_cache
        ]
        by_status: Dict[str, int] = {}
        for s in states:
            if s:
                st = s.status.value
                by_status[st] = by_status.get(st, 0) + 1
        return {
            "total": len(states),
            "by_status": by_status,
            "avg_interest": (
                sum(s.interest_score for s in states if s) / len(states)
                if states
                else 0
            ),
        }

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计"""
        hashes = self._user_index.get(user_id, [])
        if not hashes:
            return {"total": 0}
        states = [
            self._state_cache.get(h) for h in hashes if h in self._state_cache
        ]
        return {
            "total": len(states),
            "processed": sum(
                1 for s in states if s and s.status == ContentStatus.PROCESSED
            ),
            "ignored": sum(
                1 for s in states if s and s.status == ContentStatus.IGNORED
            ),
        }

    def _hash_content(self, content: str, channel_id: str = "") -> str:
        """计算内容哈希；频道作用域避免同一句话在不同会话间互相污染。"""
        normalized = content.strip().lower()
        scope = str(channel_id or "__global__").strip() or "__global__"
        return hashlib.md5(f"{scope}\0{normalized}".encode()).hexdigest()

    def _add_index_entry(self, content_hash: str, channel_id: str = "", user_id: str = "") -> None:
        channel_key = str(channel_id or "").strip()
        if channel_key:
            entries = self._channel_index.setdefault(channel_key, [])
            if content_hash not in entries:
                entries.append(content_hash)
        user_key = str(user_id or "").strip()
        if user_key:
            entries = self._user_index.setdefault(user_key, [])
            if content_hash not in entries:
                entries.append(content_hash)

    def _cleanup_cache(self) -> None:
        """清理缓存"""
        if len(self._state_cache) > self._max_cache_size:
            sorted_states = sorted(
                self._state_cache.items(),
                key=lambda x: x[1].last_processed,
            )
            remove_count = len(self._state_cache) - self._max_cache_size + 200
            for content_hash, _ in sorted_states[:remove_count]:
                self._state_cache.pop(content_hash, None)
                self._hash_index.pop(content_hash, None)
                for entries in self._channel_index.values():
                    while content_hash in entries:
                        entries.remove(content_hash)
                for entries in self._user_index.values():
                    while content_hash in entries:
                        entries.remove(content_hash)
            self._channel_index = {k: v for k, v in self._channel_index.items() if v}
            self._user_index = {k: v for k, v in self._user_index.items() if v}
            logger.debug(
                f"[内容状态追踪] 清理缓存: 移除 {remove_count} 条记录"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        by_status: Dict[str, int] = {}
        for state in self._state_cache.values():
            st = state.status.value
            by_status[st] = by_status.get(st, 0) + 1
        return {
            "total_tracked": len(self._state_cache),
            "by_status": by_status,
            "channels": len(self._channel_index),
            "users": len(self._user_index),
            "interest_keywords": len(self._interest_keywords),
        }


_content_state_tracker: Optional[ContentStateTracker] = None


def get_content_state_tracker() -> ContentStateTracker:
    """获取内容状态追踪器单例"""
    global _content_state_tracker
    if _content_state_tracker is None:
        _content_state_tracker = ContentStateTracker()
    return _content_state_tracker

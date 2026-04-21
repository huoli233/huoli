import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from src.common.logger import get_logger
from src.common.singleton import SingletonMeta

logger = get_logger("自回复识别")


class MessageSource(Enum):
    """消息来源类型"""

    USER = "user"
    BOT_REPLY = "bot_reply"
    BOT_PROACTIVE = "bot_proactive"
    BOT_REACTION = "bot_reaction"
    SYSTEM = "system"


class ResponseQuality(Enum):
    """回复质量等级"""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class BotMessageRecord:
    """机器人消息记录"""

    message_id: str
    content: str
    content_hash: str
    source: MessageSource
    response_to: str = ""
    created_at: float = field(default_factory=time.time)
    quality: ResponseQuality = ResponseQuality.ACCEPTABLE
    quality_score: float = 0.5
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    generation_model: str = ""
    generation_time: float = 0.0
    token_count: int = 0
    user_reactions: List[str] = field(default_factory=list)
    follow_up_count: int = 0


@dataclass
class ReplyChain:
    """回复链"""

    chain_id: str
    root_message_id: str
    message_ids: List[str] = field(default_factory=list)
    bot_message_ids: List[str] = field(default_factory=list)
    user_message_ids: List[str] = field(default_factory=list)
    depth: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


class SelfReplyRecognizer(metaclass=SingletonMeta):
    """自我回复识别器

    识别和分析机器人自己生成的回复。
    线程安全单例，基于统一单例工具
    """

    _initialized = False

    def __init__(
        self,
        max_cache_size: int = 1000,
        hash_threshold: float = 0.9,
    ):
        if self._initialized:
            return
        self._initialized = True
        self._max_cache_size = max_cache_size
        self._hash_threshold = hash_threshold
        self._bot_messages: Dict[str, BotMessageRecord] = {}
        self._content_hash_map: Dict[str, str] = {}
        self._reply_chains: Dict[str, ReplyChain] = {}
        self._message_to_chain: Dict[str, str] = {}
        self._bot_user_ids: set = set()
        logger.info(f"[自我回复识别] 初始化完成，缓存大小: {max_cache_size}")

    def register_bot_user(self, user_id: str) -> None:
        """注册机器人用户ID"""
        self._bot_user_ids.add(user_id)
        logger.debug(f"[自我回复识别] 注册机器人用户: {user_id}")

    def is_bot_user(self, user_id: str) -> bool:
        """检查是否是机器人用户"""
        return user_id in self._bot_user_ids

    def mark_bot_message(
        self,
        message_id: str,
        content: str,
        source: MessageSource = MessageSource.BOT_REPLY,
        response_to: str = "",
        context: Optional[Dict[str, Any]] = None,
        model: str = "",
        generation_time: float = 0.0,
        token_count: int = 0,
    ) -> BotMessageRecord:
        """标记机器人消息"""
        content_hash = self._hash_content(content)
        record = BotMessageRecord(
            message_id=message_id,
            content=content,
            content_hash=content_hash,
            source=source,
            response_to=response_to,
            context_snapshot=context or {},
            generation_model=model,
            generation_time=generation_time,
            token_count=token_count,
        )
        self._bot_messages[message_id] = record
        self._content_hash_map[content_hash] = message_id
        if response_to:
            self._update_reply_chain(response_to, message_id, is_bot=True)
        self._cleanup_cache()
        logger.debug(f"[自我回复识别] 标记机器人消息: {message_id[:8]}, 来源: {source.value}")
        return record

    def recognize_self_message(self, message: Any) -> Optional[BotMessageRecord]:
        """识别是否是自己的消息"""
        message_id = getattr(message, "message_id", "")
        if message_id and message_id in self._bot_messages:
            return self._bot_messages[message_id]
        user_id = getattr(message, "user_id", "")
        content = getattr(message, "processed_plain_text", "") or getattr(message, "content", "")
        content_hash = self._hash_content(content) if content else ""
        if self.is_bot_user(user_id):
            if content_hash in self._content_hash_map:
                cached_id = self._content_hash_map[content_hash]
                if cached_id in self._bot_messages:
                    return self._bot_messages[cached_id]
        if content_hash and content_hash in self._content_hash_map:
            cached_id = self._content_hash_map[content_hash]
            record = self._bot_messages.get(cached_id)
            if record is not None:
                message_ts = getattr(message, "timestamp", 0.0) or getattr(message, "time", 0.0)
                if not message_ts:
                    message_ts = time.time()
                # 某些平台回流自消息时会携带错误 user_id，短时间内允许按内容哈希兜底识别。
                if abs(float(message_ts) - float(record.created_at)) <= 15.0:
                    return record
        return None

    def is_bot_message(self, message: Any) -> bool:
        """检查是否是机器人消息"""
        return self.recognize_self_message(message) is not None

    def get_bot_message(self, message_id: str) -> Optional[BotMessageRecord]:
        """获取机器人消息记录"""
        return self._bot_messages.get(message_id)

    def remap_message_id(self, old_message_id: str, new_message_id: str) -> bool:
        """将本地临时消息ID映射为平台实际ID。"""
        old_id = str(old_message_id or "").strip()
        new_id = str(new_message_id or "").strip()
        if not old_id or not new_id or old_id == new_id:
            return False
        record = self._bot_messages.pop(old_id, None)
        if record is None:
            return False
        record.message_id = new_id
        self._bot_messages[new_id] = record
        if self._content_hash_map.get(record.content_hash) == old_id:
            self._content_hash_map[record.content_hash] = new_id
        if old_id in self._message_to_chain:
            chain_id = self._message_to_chain.pop(old_id)
            self._message_to_chain[new_id] = chain_id
            chain = self._reply_chains.get(chain_id)
            if chain is not None:
                if chain.root_message_id == old_id:
                    chain.root_message_id = new_id
                chain.message_ids = [new_id if item == old_id else item for item in chain.message_ids]
                chain.bot_message_ids = [new_id if item == old_id else item for item in chain.bot_message_ids]
                chain.user_message_ids = [new_id if item == old_id else item for item in chain.user_message_ids]
                chain.last_updated = time.time()
        logger.debug(f"[自我回复识别] 消息ID映射: {old_id[:8]} -> {new_id[:8]}")
        return True

    def update_quality(
        self,
        message_id: str,
        quality: ResponseQuality,
        score: float = 0.5,
    ) -> bool:
        """更新消息质量"""
        record = self._bot_messages.get(message_id)
        if record:
            record.quality = quality
            record.quality_score = score
            logger.debug(f"[自我回复识别] 更新消息质量: {message_id[:8]} -> {quality.value} ({score:.2f})")
            return True
        return False

    def record_user_reaction(
        self,
        bot_message_id: str,
        reaction: str,
    ) -> bool:
        """记录用户对机器人消息的反应"""
        record = self._bot_messages.get(bot_message_id)
        if record:
            record.user_reactions.append(reaction)
            record.follow_up_count += 1
            positive_reactions = {
                "继续",
                "好的",
                "谢谢",
                "感谢",
                "对",
                "是的",
                "嗯",
                "好",
            }
            if reaction in positive_reactions:
                record.quality_score = min(1.0, record.quality_score + 0.1)
            negative_reactions = {"不对", "错了", "不是", "不好", "不行"}
            if reaction in negative_reactions:
                record.quality_score = max(0.0, record.quality_score - 0.2)
            return True
        return False

    def _update_reply_chain(
        self,
        parent_id: str,
        child_id: str,
        is_bot: bool = True,
    ) -> None:
        """更新回复链"""
        if parent_id in self._message_to_chain:
            chain_id = self._message_to_chain[parent_id]
            chain = self._reply_chains.get(chain_id)
            if chain:
                chain.message_ids.append(child_id)
                if is_bot:
                    chain.bot_message_ids.append(child_id)
                else:
                    chain.user_message_ids.append(child_id)
                chain.depth = max(chain.depth, len(chain.message_ids))
                chain.last_updated = time.time()
                self._message_to_chain[child_id] = chain_id
        else:
            chain_id = f"chain_{int(time.time() * 1000)}"
            chain = ReplyChain(
                chain_id=chain_id,
                root_message_id=parent_id,
                message_ids=[parent_id, child_id],
                bot_message_ids=[child_id] if is_bot else [],
                user_message_ids=[] if is_bot else [child_id],
                depth=2,
            )
            self._reply_chains[chain_id] = chain
            self._message_to_chain[parent_id] = chain_id
            self._message_to_chain[child_id] = chain_id

    def get_reply_chain(self, message_id: str) -> Optional[ReplyChain]:
        """获取消息所在的回复链"""
        chain_id = self._message_to_chain.get(message_id)
        if chain_id:
            return self._reply_chains.get(chain_id)
        return None

    def _recent_records_for_stream(
        self,
        stream_id: str,
        limit: int = 10,
    ) -> List[BotMessageRecord]:
        """按频道提取最近 bot 消息记录。"""
        records = [
            record
            for record in self._bot_messages.values()
            if not stream_id
            or str(record.context_snapshot.get("stream_id", "") or "").strip() == str(stream_id or "").strip()
        ]
        records.sort(key=lambda item: float(item.created_at or 0.0), reverse=True)
        return records[: max(1, int(limit or 1))]

    def build_recent_message_snapshots(
        self,
        stream_id: str,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """构建统一的 bot 自消息快照。

        优先合并 SelfAwareness 的正文记录，再用 recognizer 自身缓存补齐
        source / response_to / reply_chain 等元信息。
        """
        normalized_stream = str(stream_id or "").strip()
        bounded_limit = max(1, int(limit or 1))
        snapshots: List[Dict[str, Any]] = []
        seen_ids = set()
        seen_texts = set()

        try:
            from src.modules.recall.self_awareness import get_self_awareness

            awareness_messages = get_self_awareness().get_recent_messages(normalized_stream, limit=bounded_limit)
        except Exception as exc:
            logger.debug(f"[自我回复识别] 构建统一快照时读取自我觉察失败: {exc}")
            awareness_messages = []

        for item in awareness_messages:
            msg_id = str(getattr(item, "msg_id", "") or "").strip()
            content = str(getattr(item, "content", "") or "").strip()
            if not content:
                continue
            record = self.get_bot_message(msg_id) if msg_id else None
            chain = self.get_reply_chain(msg_id) if msg_id else None
            context = getattr(item, "context", None) or {}
            snapshots.append(
                {
                    "message_id": msg_id,
                    "text": content,
                    "ts": float(getattr(item, "sent_at", 0.0) or 0.0) or time.time(),
                    "source": record.source.value if record is not None else str(context.get("source", "") or ""),
                    "response_to": (
                        str(record.response_to or "").strip()
                        if record is not None
                        else str(context.get("response_to", "") or "").strip()
                    ),
                    "quality_score": float(getattr(record, "quality_score", 0.0) or 0.0) if record is not None else 0.0,
                    "follow_up_count": int(getattr(record, "follow_up_count", 0) or 0) if record is not None else 0,
                    "chain_id": str(getattr(chain, "chain_id", "") or "") if chain is not None else "",
                    "chain_depth": int(getattr(chain, "depth", 0) or 0) if chain is not None else 0,
                    "target_user_id": str(context.get("target_user_id", "") or ""),
                    "has_recognizer_record": record is not None,
                }
            )
            if msg_id:
                seen_ids.add(msg_id)
            seen_texts.add(content)

        if len(snapshots) < bounded_limit:
            for record in self._recent_records_for_stream(normalized_stream, limit=max(bounded_limit * 3, 6)):
                if record.message_id in seen_ids or record.content in seen_texts:
                    continue
                chain = self.get_reply_chain(record.message_id)
                snapshots.append(
                    {
                        "message_id": str(record.message_id or "").strip(),
                        "text": str(record.content or "").strip(),
                        "ts": float(record.created_at or 0.0) or time.time(),
                        "source": record.source.value,
                        "response_to": str(record.response_to or "").strip(),
                        "quality_score": float(record.quality_score or 0.0),
                        "follow_up_count": int(record.follow_up_count or 0),
                        "chain_id": str(getattr(chain, "chain_id", "") or "") if chain is not None else "",
                        "chain_depth": int(getattr(chain, "depth", 0) or 0) if chain is not None else 0,
                        "target_user_id": str(record.context_snapshot.get("target_user_id", "") or ""),
                        "has_recognizer_record": True,
                    }
                )
                if record.message_id:
                    seen_ids.add(record.message_id)
                if record.content:
                    seen_texts.add(record.content)
                if len(snapshots) >= bounded_limit:
                    break

        snapshots.sort(key=lambda item: float(item.get("ts", 0.0) or 0.0))
        return snapshots[-bounded_limit:]

    def analyze_response_patterns(self, limit: int = 100) -> Dict[str, Any]:
        """分析回复模式"""
        messages = list(self._bot_messages.values())[-limit:]
        if not messages:
            return {"total": 0}
        by_source: Dict[str, int] = {}
        by_quality: Dict[str, int] = {}
        total_tokens = 0
        total_time = 0.0
        for m in messages:
            src = m.source.value
            by_source[src] = by_source.get(src, 0) + 1
            ql = m.quality.value
            by_quality[ql] = by_quality.get(ql, 0) + 1
            total_tokens += m.token_count
            total_time += m.generation_time
        return {
            "total": len(messages),
            "by_source": by_source,
            "by_quality": by_quality,
            "avg_tokens": total_tokens / len(messages) if messages else 0,
            "avg_generation_time": (total_time / len(messages) if messages else 0),
            "positive_reaction_rate": sum(1 for m in messages if m.quality_score > 0.6) / len(messages),
        }

    def _hash_content(self, content: str) -> str:
        """计算内容哈希"""
        normalized = content.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _cleanup_cache(self) -> None:
        """清理缓存"""
        if len(self._bot_messages) > self._max_cache_size:
            sorted_messages = sorted(
                self._bot_messages.items(),
                key=lambda x: x[1].created_at,
            )
            remove_count = len(self._bot_messages) - self._max_cache_size + 100
            for msg_id, _ in sorted_messages[:remove_count]:
                record = self._bot_messages.pop(msg_id, None)
                if record:
                    self._content_hash_map.pop(record.content_hash, None)
            logger.debug(f"[自我回复识别] 清理缓存: 移除 {remove_count} 条记录")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "cached_messages": len(self._bot_messages),
            "cached_hashes": len(self._content_hash_map),
            "reply_chains": len(self._reply_chains),
            "bot_users": len(self._bot_user_ids),
        }


_self_reply_recognizer: Optional[SelfReplyRecognizer] = None


def get_self_reply_recognizer() -> SelfReplyRecognizer:
    """获取自我回复识别器单例"""
    global _self_reply_recognizer
    if _self_reply_recognizer is None:
        _self_reply_recognizer = SelfReplyRecognizer()
    return _self_reply_recognizer

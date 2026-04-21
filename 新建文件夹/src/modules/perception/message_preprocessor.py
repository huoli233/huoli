import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine

logger = get_logger("消息预处理")


@dataclass
class ProcessedMessage:
    msg_id: str = ""
    sender_name: str = ""
    sender_id: str = ""
    text: str = ""
    timestamp: float = 0.0
    content_type: str = "text"
    has_image: bool = False
    image_count: int = 0
    is_at_me: bool = False
    is_quote_reply: bool = False
    quoted_content: str = ""
    forward_summary: str = ""
    relative_time: str = ""
    relation_tag: str = ""


@dataclass
class BatchMeta:
    total_count: int = 0
    unique_users: int = 0
    time_span_seconds: float = 0.0
    has_images: bool = False
    image_count: int = 0
    has_quote: bool = False
    has_forward: bool = False
    has_at_me: bool = False
    is_spam: bool = False
    spam_type: str = "正常"
    activity_level: str = "平稳"


@dataclass
class ProcessedBatch:
    messages: List[ProcessedMessage] = field(default_factory=list)
    meta: BatchMeta = field(default_factory=BatchMeta)


@dataclass
class StandardMessage:
    message_id: str = ""
    text: str = ""
    sender_id: str = ""
    sender_name: str = ""
    timestamp: float = 0.0
    is_at_me: bool = False
    has_image: bool = False
    image_urls: List[str] = field(default_factory=list)
    is_bot_self: bool = False
    content_type: str = "text"
    is_quote_reply: bool = False
    quoted_content: str = ""
    quoted_sender: str = ""
    is_forward: bool = False
    forward_title: str = ""
    forward_items: List[str] = field(default_factory=list)


class MessagePreprocessor:
    def __init__(self, config_engine=None):
        self._config = config_engine or get_default_config_engine()
        self._spam_interval_threshold: float = 2.0
        self._activity_burst_count: int = 15
        self._activity_burst_span: float = 120.0
        self._activity_burst_users: int = 4
        self._activity_active_count: int = 6
        self._activity_active_users: int = 3
        self._activity_stable_count: int = 2
        self._text_max_length: int = 100
        self._quote_max_length: int = 50
        self._forward_max_items: int = 3
        self._load_config()

    def _load_config(self) -> None:
        self._spam_interval_threshold = float(
            self._config.get("preprocessor", "spam_interval_threshold", 2.0)
        )
        self._activity_burst_count = int(
            self._config.get("preprocessor", "activity_burst_count", 15)
        )
        self._activity_burst_span = float(
            self._config.get("preprocessor", "activity_burst_span", 120.0)
        )
        self._activity_burst_users = int(
            self._config.get("preprocessor", "activity_burst_users", 4)
        )
        self._activity_active_count = int(
            self._config.get("preprocessor", "activity_active_count", 6)
        )
        self._activity_active_users = int(
            self._config.get("preprocessor", "activity_active_users", 3)
        )
        self._activity_stable_count = int(
            self._config.get("preprocessor", "activity_stable_count", 2)
        )
        self._text_max_length = int(
            self._config.get("preprocessor", "message_text_max_length", 100)
        )
        self._quote_max_length = int(
            self._config.get("preprocessor", "quote_max_length", 50)
        )
        self._forward_max_items = int(
            self._config.get("preprocessor", "forward_summary_max_items", 3)
        )

    def process(
        self,
        messages: List[StandardMessage],
        bot_name: str = "",
        user_relations: Optional[Dict[str, str]] = None,
    ) -> ProcessedBatch:
        if not messages:
            return ProcessedBatch()
        if user_relations is None:
            user_relations = {}
        now = time.time()
        filtered = [m for m in messages if not m.is_bot_self]
        if not filtered:
            return ProcessedBatch()
        processed_list: List[ProcessedMessage] = []
        for msg in filtered:
            pm = self._process_single_message(
                msg, bot_name, user_relations, now
            )
            processed_list.append(pm)
        meta = self._build_meta(processed_list, filtered, now, bot_name)
        batch = ProcessedBatch(messages=processed_list, meta=meta)
        logger.debug(
            f"预处理完成: {meta.total_count}条消息, "
            f"{meta.unique_users}人, "
            f"活跃度={meta.activity_level}, "
            f"刷屏={meta.spam_type}"
        )
        return batch

    def _process_single_message(
        self,
        msg: StandardMessage,
        bot_name: str,
        user_relations: Dict[str, str],
        now: float,
    ) -> ProcessedMessage:
        pm = ProcessedMessage()
        pm.msg_id = msg.message_id or self._generate_message_id(msg)
        pm.sender_name = msg.sender_name or "未知用户"
        pm.sender_id = msg.sender_id or ""
        pm.timestamp = msg.timestamp
        raw_text = self._clean_text(msg.text or "")
        if len(raw_text) > self._text_max_length:
            pm.text = raw_text[: self._text_max_length] + "..."
        else:
            pm.text = raw_text
        pm.content_type = msg.content_type or "text"
        pm.has_image = msg.has_image or bool(msg.image_urls)
        pm.image_count = (
            len(msg.image_urls)
            if msg.image_urls
            else (1 if msg.has_image else 0)
        )
        if not pm.text and pm.has_image:
            pm.text = "[图片]"
            pm.content_type = "image"
        elif pm.has_image and "[图片]" not in pm.text:
            pm.text = f"[图片] {pm.text}"
        if pm.content_type == "emoji" and not pm.text:
            pm.text = "[表情]"
        if pm.content_type == "voice" and not pm.text:
            pm.text = "[语音]"
        if pm.content_type == "file" and not pm.text:
            pm.text = "[文件]"
        pm.is_at_me = msg.is_at_me
        if not pm.is_at_me and bot_name and raw_text:
            if bot_name in raw_text:
                pm.is_at_me = True
            aliases = self._config.get("preprocessor", "bot_aliases", [])
            if aliases:
                for alias in aliases:
                    if alias and alias in raw_text:
                        pm.is_at_me = True
                        break
        pm.is_quote_reply = msg.is_quote_reply
        if msg.quoted_content:
            quoted = msg.quoted_content
            if len(quoted) > self._quote_max_length:
                quoted = quoted[: self._quote_max_length] + "..."
            if msg.quoted_sender:
                pm.quoted_content = f"{msg.quoted_sender}: {quoted}"
            else:
                pm.quoted_content = quoted
        if msg.is_forward:
            items = (
                msg.forward_items[: self._forward_max_items]
                if msg.forward_items
                else []
            )
            title = msg.forward_title or "聊天记录"
            if items:
                summary_parts = []
                for item in items:
                    if len(item) > 60:
                        summary_parts.append(item[:57] + "...")
                    else:
                        summary_parts.append(item)
                pm.forward_summary = f"[合并转发: {title}] " + " | ".join(
                    summary_parts
                )
            else:
                pm.forward_summary = f"[合并转发: {title}]"
            if not pm.text or pm.text == "[图片]":
                pm.text = pm.forward_summary
                pm.content_type = "forward"
        pm.relative_time = self._calc_relative_time(msg.timestamp, now)
        pm.relation_tag = user_relations.get(msg.sender_id, "")
        return pm

    def _build_meta(
        self,
        processed: List[ProcessedMessage],
        raw: List[StandardMessage],
        now: float,
        bot_name: str,
    ) -> BatchMeta:
        meta = BatchMeta()
        if not processed:
            return meta
        meta.total_count = len(processed)
        user_ids = set(m.sender_id for m in processed if m.sender_id)
        meta.unique_users = len(user_ids)
        timestamps = [m.timestamp for m in processed if m.timestamp > 0]
        if len(timestamps) >= 2:
            meta.time_span_seconds = max(timestamps) - min(timestamps)
        meta.has_images = any(m.has_image for m in processed)
        meta.image_count = sum(m.image_count for m in processed)
        meta.has_quote = any(m.is_quote_reply for m in processed)
        meta.has_forward = any(m.forward_summary for m in processed)
        meta.has_at_me = any(m.is_at_me for m in processed)
        meta.is_spam, meta.spam_type = self._detect_spam(raw, now)
        meta.activity_level = self._calc_activity(
            meta.total_count,
            meta.unique_users,
            meta.time_span_seconds,
        )
        return meta

    def _detect_spam(
        self, messages: List[StandardMessage], now: float
    ) -> Tuple[bool, str]:
        if len(messages) < 3:
            return False, "正常"
        same_user_threshold = int(
            self._config.get("preprocessor", "spam_same_user_threshold", 5)
        )
        window_seconds = float(
            self._config.get("preprocessor", "spam_same_user_window", 30)
        )
        repeat_threshold = int(
            self._config.get("preprocessor", "spam_repeat_threshold", 3)
        )
        user_recent_counts: Counter = Counter()
        for msg in messages:
            if msg.timestamp > 0 and (now - msg.timestamp) <= window_seconds:
                user_recent_counts[msg.sender_id] += 1
            elif msg.timestamp <= 0:
                user_recent_counts[msg.sender_id] += 1
        for user_id, count in user_recent_counts.items():
            if count >= same_user_threshold:
                sender_name = ""
                for m in messages:
                    if m.sender_id == user_id:
                        sender_name = m.sender_name
                        break
                logger.debug(
                    f"检测到单人刷屏: {sender_name}({user_id}) 在{window_seconds}秒内发了{count}条"
                )
                return True, "单人刷屏"
        consecutive = 1
        for i in range(1, len(messages)):
            curr_text = (messages[i].text or "").strip()
            prev_text = (messages[i - 1].text or "").strip()
            if curr_text and curr_text == prev_text:
                consecutive += 1
                if consecutive >= repeat_threshold:
                    logger.debug(
                        f"检测到复读刷屏: '{curr_text[:30]}...' 连续{consecutive}条"
                    )
                    return True, "复读"
            else:
                consecutive = 1
        if len(user_recent_counts) >= 3:
            active_users = [
                uid for uid, count in user_recent_counts.items() if count >= 3
            ]
            if len(active_users) >= 3:
                recent_timestamps = sorted(
                    m.timestamp
                    for m in messages
                    if m.timestamp > 0
                    and (now - m.timestamp) <= window_seconds
                )
                if len(recent_timestamps) >= 6:
                    intervals = [
                        recent_timestamps[i + 1] - recent_timestamps[i]
                        for i in range(len(recent_timestamps) - 1)
                    ]
                    avg_interval = (
                        sum(intervals) / len(intervals) if intervals else 999
                    )
                    if avg_interval < self._spam_interval_threshold:
                        logger.debug(
                            f"检测到多人灌水: {
                                len(active_users)}人高频发言, 平均间隔{
                                avg_interval:.1f}秒"
                        )
                        return True, "多人灌水"
        return False, "正常"

    def _calc_activity(self, count: int, users: int, span: float) -> str:
        if (
            count >= self._activity_burst_count
            and span < self._activity_burst_span
        ):
            return "爆发"
        if count >= 10 and users >= self._activity_burst_users:
            return "爆发"
        if (
            count >= self._activity_active_count
            or users >= self._activity_active_users
        ):
            return "活跃"
        if count >= self._activity_stable_count:
            return "平稳"
        return "冷清"

    def _calc_relative_time(self, timestamp: float, now: float) -> str:
        if timestamp <= 0:
            return "未知时间"
        diff = int(now - timestamp)
        if diff < 0:
            diff = 0
        if diff <= 10:
            return "刚刚"
        if diff < 60:
            return f"{diff}秒前"
        if diff < 3600:
            return f"{diff // 60}分钟前"
        if diff < 86400:
            return f"{diff // 3600}小时前"
        return f"{diff // 86400}天前"

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = (
            text.replace("\u200b", "")
            .replace("\u200c", "")
            .replace("\u200d", "")
            .replace("\ufeff", "")
        )
        text = re.sub(r"\[CQ:[^\]]+\]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _generate_message_id(self, msg: StandardMessage) -> str:
        text_hash = hash(msg.text or "") & 0xFFFFFFFF
        return f"{int(msg.timestamp)}_{msg.sender_id}_{text_hash:08x}"

    def get_stats(self) -> Dict[str, Any]:
        return {
            "spam_interval_threshold": self._spam_interval_threshold,
            "activity_burst_count": self._activity_burst_count,
            "text_max_length": self._text_max_length,
        }


_message_preprocessor: Optional[MessagePreprocessor] = None


def get_message_preprocessor(config_engine=None) -> MessagePreprocessor:
    global _message_preprocessor
    if _message_preprocessor is None:
        _message_preprocessor = MessagePreprocessor(config_engine)
    return _message_preprocessor

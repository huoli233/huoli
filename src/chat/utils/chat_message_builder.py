import ast
import json
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from src.chat.utils.utils import translate_timestamp_to_human_readable
from src.common.database.database_model import (
    ActionRecords as ActionRecord,
    Messages as MessageRecord,
)
from src.common.logger import get_logger
from src.config.config import global_config
from src.person_info.person_info import Person

logger = get_logger("消息构建器")

_PICTURE_DESCRIPTION_CACHE: dict[str, str] = {}
_PICTURE_CACHE_MAX_SIZE: int = 2000
_CONTENT_BUILDERS: dict[str, "PacketContentBuilder"] = {}
_HISTORY_MANAGERS: dict[str, "MessageHistoryManager"] = {}
_MESSAGE_FORMATTER: Optional["MessageFormatter"] = None
_MESSAGE_DEDUPLICATOR: Optional["MessageDeduplicator"] = None
_MESSAGE_SPLITTER: Optional["MessageSplitter"] = None
_TOPIC_EXTRACTOR: Optional["TopicExtractor"] = None


@dataclass
class TimelineEntry:
    """时间线条目，用于统一表示消息和动作记录。"""

    timestamp: float
    speaker: str
    content: str
    kind: str
    source: Any
    marker: str = ""
    is_action: bool = False


@dataclass
class MessageActionModel:
    """消息与动作的统一数据模型。"""

    time: float
    user_id: str
    user_platform: str
    user_nickname: str
    user_cardname: str
    processed_plain_text: str
    display_message: str
    chat_info_platform: str
    is_action_record: bool = False
    action_name: str = ""
    is_command: bool = False

    @classmethod
    def from_message_record(
        cls, record: MessageRecord
    ) -> "MessageActionModel":
        """从消息记录构建模型实例。"""
        return cls(
            time=float(record.time or 0),
            user_id=str(record.user_id or ""),
            user_platform=str(
                record.user_platform or record.chat_info_user_platform or "qq"
            ),
            user_nickname=str(
                record.user_nickname or record.chat_info_user_nickname or ""
            ),
            user_cardname=str(
                record.user_cardname or record.chat_info_user_cardname or ""
            ),
            processed_plain_text=str(record.processed_plain_text or ""),
            display_message=str(record.display_message or ""),
            chat_info_platform=str(record.chat_info_platform or "qq"),
            is_action_record=False,
            is_command=bool(getattr(record, "is_command", False)),
        )

    @classmethod
    def from_action_record(
        cls, record: ActionRecord, bot_nickname: str
    ) -> "MessageActionModel":
        """从动作记录构建模型实例。"""
        return cls(
            time=float(record.time or 0),
            user_id=str(getattr(global_config.bot, "qq_account", "") or ""),
            user_platform=str(
                getattr(global_config.bot, "platform", "qq") or "qq"
            ),
            user_nickname=bot_nickname,
            user_cardname="",
            processed_plain_text=str(record.action_reasoning or ""),
            display_message=str(record.action_prompt_display or ""),
            chat_info_platform=str(record.chat_info_platform or "qq"),
            is_action_record=True,
            action_name=str(record.action_name or ""),
        )


@dataclass
class CompactMessage:
    """轻量消息模型，用于历史缓存与二次构建。"""

    timestamp: float = 0.0
    speaker: str = ""
    content: str = ""
    topic: str = ""


class MessageHistoryManager:
    """消息历史管理器，按频道保存最近结构化消息。"""

    def __init__(self, channel_id: str, max_items: int = 300):
        self.channel_id = channel_id
        self._history: deque[CompactMessage] = deque(maxlen=max_items)

    def append(self, entry: CompactMessage) -> None:
        if not entry.content:
            return
        self._history.append(entry)

    def extend(self, entries: list[CompactMessage]) -> None:
        for entry in entries:
            self.append(entry)

    def recent(self, limit: int = 20) -> list[CompactMessage]:
        if limit <= 0:
            return []
        return list(self._history)[-limit:]

    def clear(self) -> None:
        self._history.clear()


class MessageFormatter:
    """消息格式器，负责将缓存历史转换为简洁文本。"""

    def format_history(
        self, entries: list[CompactMessage], include_topic: bool = False
    ) -> str:
        lines: list[str] = []
        for item in entries:
            speaker = item.speaker or "某人"
            content = item.content.strip()
            if not content:
                continue
            if include_topic and item.topic:
                lines.append(f"[{item.topic}] {speaker}: {content}")
            else:
                lines.append(f"{speaker}: {content}")
        return "\n".join(lines)


class MessageDeduplicator:
    """短窗口去重器，避免相同消息重复进入构建链路。"""

    def __init__(self, ttl_seconds: float = 45.0):
        self._ttl_seconds = ttl_seconds
        self._seen: dict[str, float] = {}

    @staticmethod
    def _digest(text: str) -> str:
        source = (text or "").strip().lower()
        if not source:
            return ""
        compact = re.sub(r"\s+", "", source)
        return compact[:200]

    def _purge(self) -> None:
        now = time.time()
        expired = [
            key
            for key, ts in self._seen.items()
            if (now - ts) > self._ttl_seconds
        ]
        for key in expired:
            self._seen.pop(key, None)

    def is_duplicate(self, text: str) -> bool:
        digest = self._digest(text)
        if not digest:
            return False
        self._purge()
        if digest in self._seen:
            return True
        self._seen[digest] = time.time()
        return False


class MessageSplitter:
    """长文本分段器，优先在自然断点切分。"""

    def __init__(self, max_length: int = 220):
        self._max_length = max(80, max_length)
        self._split_points = ["。", "！", "？", "\n", ",", "，", " "]

    def split(self, content: str) -> list[str]:
        if len(content) <= self._max_length:
            return [content]
        parts: list[str] = []
        current = ""
        for char in content:
            current += char
            if len(current) < self._max_length:
                continue
            cut_at = -1
            for point in self._split_points:
                pos = current.rfind(point)
                if pos > len(current) // 2:
                    cut_at = pos + 1
                    break
            if cut_at > 0:
                parts.append(current[:cut_at].strip())
                current = current[cut_at:]
            else:
                parts.append(current.strip())
                current = ""
        if current.strip():
            parts.append(current.strip())
        return [item for item in parts if item]


class TopicExtractor:
    """轻量话题提取器，提供构建期主题片段。"""

    def __init__(self):
        self._fragment_width = 2

    def extract_topics(self, content: str) -> list[str]:
        text = self._normalize_text(content)
        if not text:
            return []
        if len(text) <= self._fragment_width:
            return [text]
        fragments = {
            text[index: index + self._fragment_width]
            for index in range(len(text) - self._fragment_width + 1)
        }
        ranked = sorted(fragments)
        return ranked[:5]

    def get_main_topic(self, content: str) -> str:
        topics = self.extract_topics(content)
        if not topics:
            return ""
        return topics[0]

    @staticmethod
    def _normalize_text(content: str) -> str:
        text = (content or "").strip().lower()
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
            text = text.replace(char, "")
        return text


class PacketContentBuilder:
    """频道内容构建器，融合分段、去重、历史缓存与话题标记。"""

    def __init__(self, channel_id: str):
        self.channel_id = channel_id
        self._splitter = get_message_splitter()
        self._deduplicator = get_message_deduplicator()
        self._history = get_history_manager(channel_id)
        self._topic_extractor = get_topic_extractor()

    def build(
        self, speaker: str, content: str, timestamp: Optional[float] = None
    ) -> list[CompactMessage]:
        raw = (content or "").strip()
        if not raw:
            return []
        if self._deduplicator.is_duplicate(raw):
            return []
        built: list[CompactMessage] = []
        for chunk in self._splitter.split(raw):
            topic = self._topic_extractor.get_main_topic(chunk)
            item = CompactMessage(
                timestamp=float(timestamp or time.time()),
                speaker=speaker or "某人",
                content=chunk,
                topic=topic,
            )
            built.append(item)
        self._history.extend(built)
        return built


def get_content_builder(channel_id: str) -> PacketContentBuilder:
    """获取频道级内容构建器。"""
    if channel_id not in _CONTENT_BUILDERS:
        _CONTENT_BUILDERS[channel_id] = PacketContentBuilder(channel_id)
    return _CONTENT_BUILDERS[channel_id]


def get_history_manager(channel_id: str) -> MessageHistoryManager:
    """获取频道消息历史管理器。"""
    if channel_id not in _HISTORY_MANAGERS:
        _HISTORY_MANAGERS[channel_id] = MessageHistoryManager(channel_id)
    return _HISTORY_MANAGERS[channel_id]


def get_message_formatter() -> MessageFormatter:
    """获取全局消息格式器。"""
    global _MESSAGE_FORMATTER
    if _MESSAGE_FORMATTER is None:
        _MESSAGE_FORMATTER = MessageFormatter()
    return _MESSAGE_FORMATTER


def get_message_deduplicator() -> MessageDeduplicator:
    """获取全局去重器。"""
    global _MESSAGE_DEDUPLICATOR
    if _MESSAGE_DEDUPLICATOR is None:
        _MESSAGE_DEDUPLICATOR = MessageDeduplicator()
    return _MESSAGE_DEDUPLICATOR


def get_message_splitter() -> MessageSplitter:
    """获取全局文本分段器。"""
    global _MESSAGE_SPLITTER
    if _MESSAGE_SPLITTER is None:
        _MESSAGE_SPLITTER = MessageSplitter()
    return _MESSAGE_SPLITTER


def get_topic_extractor() -> TopicExtractor:
    """获取全局话题提取器。"""
    global _TOPIC_EXTRACTOR
    if _TOPIC_EXTRACTOR is None:
        _TOPIC_EXTRACTOR = TopicExtractor()
    return _TOPIC_EXTRACTOR


def _is_bot_user(user_id: str) -> bool:
    """判断用户ID是否为机器人自身。"""
    bot_account = str(getattr(global_config.bot, "qq_account", "") or "")
    telegram_account = str(
        getattr(global_config.bot, "telegram_account", "") or ""
    )
    clean_id = str(user_id or "").strip()
    return clean_id in {"bot", bot_account, telegram_account}


def _resolve_person_name(
    platform: str, user_id: str, fallback_name: str, replace_bot_name: bool
) -> str:
    """解析用户名称，优先使用人物信息。"""
    clean_user_id = str(user_id or "").strip()
    if replace_bot_name and _is_bot_user(clean_user_id):
        return f"{global_config.bot.nickname}(你)"
    try:
        person = Person(platform=platform or "qq", user_id=clean_user_id)
        if person.is_known and person.person_name:
            return person.person_name
    except Exception as _e:
        logger.debug(f"异常: {_e}")
    return fallback_name or clean_user_id


def translate_pid_to_description(pic_id: str) -> str:
    """将图片ID转换为描述文本。"""
    picture_id = str(pic_id or "").strip()
    if not picture_id:
        return "图片资源缺失"
    cached_description = _PICTURE_DESCRIPTION_CACHE.get(picture_id)
    if cached_description:
        return cached_description
    description = f"资源标识 {picture_id}"
    try:
        from src.common.database.database_model import Images

        if Images is not None:
            record = Images.get_or_none(
                Images.image_id == picture_id
            )
            if record and getattr(record, "description", ""):
                description = str(record.description).strip()
    except Exception as _e:
        logger.debug(f"异常: {_e}")
    if len(_PICTURE_DESCRIPTION_CACHE) >= _PICTURE_CACHE_MAX_SIZE:
        _evict_keys = list(_PICTURE_DESCRIPTION_CACHE.keys())[
            : _PICTURE_CACHE_MAX_SIZE // 4
        ]
        for _k in _evict_keys:
            _PICTURE_DESCRIPTION_CACHE.pop(_k, None)
    _PICTURE_DESCRIPTION_CACHE[picture_id] = description
    return description


def replace_user_references(
    content: Optional[str],
    platform: str = "",
    name_resolver: Optional[Callable[[str, str], str]] = None,
    replace_bot_name: bool = True,
) -> str:
    """替换内容中的用户引用格式，包括回复<aaa:bbb>和@<aaa:bbb>格式。"""
    if not content:
        return ""
    if name_resolver is None:

        def default_name_resolver(inner_platform: str, user_id: str) -> str:
            return _resolve_person_name(
                inner_platform, user_id, user_id, replace_bot_name
            )

        name_resolver = default_name_resolver

    def replace_reply(match: re.Match[str]) -> str:
        raw_name = match.group(1)
        raw_user_id = match.group(2)
        try:
            if replace_bot_name and _is_bot_user(raw_user_id):
                return f"回复 {global_config.bot.nickname}(你)"
            return f"回复 {name_resolver(platform, raw_user_id) or raw_name}"
        except Exception:
            return f"回复 {raw_name}"

    def replace_at(match: re.Match[str]) -> str:
        raw_name = match.group(1)
        raw_user_id = match.group(2)
        try:
            if replace_bot_name and _is_bot_user(raw_user_id):
                return f"@{global_config.bot.nickname}(你)"
            return f"@{name_resolver(platform, raw_user_id) or raw_name}"
        except Exception:
            return f"@{raw_name}"

    updated = re.sub(r"回复<([^:<>]+):([^:<>]+)>", replace_reply, content)
    updated = re.sub(r"@<([^:<>]+):([^:<>]+)>", replace_at, updated)
    return updated


def assign_message_ids(
    messages: list[MessageRecord],
) -> list[tuple[str, MessageRecord]]:
    """为消息列表分配唯一标识符。"""
    result: list[tuple[str, MessageRecord]] = []
    for index, message in enumerate(messages, start=1):
        message_id = f"msg_{index}_{int(message.time or 0)}"
        result.append((message_id, message))
    return result


def _apply_limit(query, field, limit: int, limit_mode: str):
    """应用分页限制到查询。"""
    if limit <= 0:
        return query.order_by(field.asc())
    if limit_mode == "earliest":
        return query.order_by(field.asc()).limit(limit)
    limited_query = query.order_by(field.desc()).limit(limit)
    rows = list(limited_query)
    rows.reverse()
    return rows


def _materialize_query(query_result) -> list:
    """将查询结果物化为列表。"""
    if isinstance(query_result, list):
        return query_result
    return list(query_result)


def get_raw_msg_by_timestamp(
    timestamp_start: float,
    timestamp_end: float,
    limit: int = 0,
    limit_mode: str = "latest",
) -> list[MessageRecord]:
    """获取从指定时间戳到指定时间戳的消息，按时间升序排序。"""
    query = MessageRecord.select().where(
        (MessageRecord.time > timestamp_start)
        & (MessageRecord.time < timestamp_end)
    )
    return _materialize_query(
        _apply_limit(query, MessageRecord.time, limit, limit_mode)
    )


def get_raw_msg_by_timestamp_with_chat(
    chat_id: str,
    timestamp_start: float,
    timestamp_end: float,
    limit: int = 0,
    limit_mode: str = "latest",
    filter_bot: bool = False,
    filter_command: bool = False,
    filter_intercept_message_level: Optional[int] = None,
) -> list[MessageRecord]:
    """获取在特定聊天从指定时间戳到指定时间戳的消息。"""
    del filter_intercept_message_level
    query = MessageRecord.select().where(
        (MessageRecord.chat_id == chat_id)
        & (MessageRecord.time > timestamp_start)
        & (MessageRecord.time < timestamp_end)
    )
    if filter_bot:
        query = query.where(
            MessageRecord.user_id
            != getattr(global_config.bot, "qq_account", "")
        )
    if filter_command:
        query = query.where(MessageRecord.is_command.is_(False))
    return _materialize_query(
        _apply_limit(query, MessageRecord.time, limit, limit_mode)
    )


def get_raw_msg_by_timestamp_with_chat_inclusive(
    chat_id: str,
    timestamp_start: float,
    timestamp_end: float,
    limit: int = 0,
    limit_mode: str = "latest",
    filter_bot: bool = False,
    filter_command: bool = False,
    filter_intercept_message_level: Optional[int] = None,
) -> list[MessageRecord]:
    """获取在特定聊天从指定时间戳到指定时间戳的消息（包含边界）。"""
    del filter_intercept_message_level
    query = MessageRecord.select().where(
        (MessageRecord.chat_id == chat_id)
        & (MessageRecord.time >= timestamp_start)
        & (MessageRecord.time <= timestamp_end)
    )
    if filter_bot:
        query = query.where(
            MessageRecord.user_id
            != getattr(global_config.bot, "qq_account", "")
        )
    if filter_command:
        query = query.where(MessageRecord.is_command.is_(False))
    return _materialize_query(
        _apply_limit(query, MessageRecord.time, limit, limit_mode)
    )


def get_raw_msg_by_timestamp_with_users(
    timestamp_start: float,
    timestamp_end: float,
    person_ids: list[str],
    limit: int = 0,
    limit_mode: str = "latest",
) -> list[MessageRecord]:
    """获取某些特定用户在所有聊天中从指定时间戳到指定时间戳的消息。"""
    if not person_ids:
        return []
    query = MessageRecord.select().where(
        (MessageRecord.time > timestamp_start)
        & (MessageRecord.time < timestamp_end)
        & (MessageRecord.user_id.in_(person_ids))
    )
    return _materialize_query(
        _apply_limit(query, MessageRecord.time, limit, limit_mode)
    )


def get_raw_msg_by_timestamp_with_chat_users(
    chat_id: str,
    timestamp_start: float,
    timestamp_end: float,
    person_ids: list[str],
    limit: int = 0,
    limit_mode: str = "latest",
) -> list[MessageRecord]:
    """获取某些特定用户在特定聊天从指定时间戳到指定时间戳的消息。"""
    if not person_ids:
        return []
    query = MessageRecord.select().where(
        (MessageRecord.chat_id == chat_id)
        & (MessageRecord.time > timestamp_start)
        & (MessageRecord.time < timestamp_end)
        & (MessageRecord.user_id.in_(person_ids))
    )
    return _materialize_query(
        _apply_limit(query, MessageRecord.time, limit, limit_mode)
    )


def get_raw_msg_before_timestamp(
    timestamp: float, limit: int = 0
) -> list[MessageRecord]:
    """获取指定时间戳之前的消息。"""
    query = MessageRecord.select().where(MessageRecord.time < timestamp)
    return _materialize_query(
        _apply_limit(query, MessageRecord.time, limit, "latest")
    )


def get_raw_msg_before_timestamp_with_chat(
    chat_id: str,
    timestamp: float,
    limit: int = 0,
    filter_intercept_message_level: Optional[int] = None,
) -> list[MessageRecord]:
    """获取指定聊天在指定时间戳之前的消息。"""
    del filter_intercept_message_level
    query = MessageRecord.select().where(
        (MessageRecord.chat_id == chat_id) & (MessageRecord.time < timestamp)
    )
    return _materialize_query(
        _apply_limit(query, MessageRecord.time, limit, "latest")
    )


def get_raw_msg_before_timestamp_with_users(
    timestamp: float,
    person_ids: list[str],
    limit: int = 0,
) -> list[MessageRecord]:
    """获取指定时间戳之前的消息（按用户过滤）。"""
    if not person_ids:
        return []
    query = MessageRecord.select().where(
        (MessageRecord.time < timestamp)
        & (MessageRecord.user_id.in_(person_ids))
    )
    return _materialize_query(
        _apply_limit(query, MessageRecord.time, limit, "latest")
    )


def get_actions_by_timestamp_with_chat(
    chat_id: str,
    timestamp_start: float = 0,
    timestamp_end: float = time.time(),
    limit: int = 0,
    limit_mode: str = "latest",
) -> list[ActionRecord]:
    """获取在特定聊天从指定时间戳到指定时间戳的动作记录。"""
    query = ActionRecord.select().where(
        (ActionRecord.chat_id == chat_id)
        & (ActionRecord.time > timestamp_start)
        & (ActionRecord.time < timestamp_end)
    )
    return _materialize_query(
        _apply_limit(query, ActionRecord.time, limit, limit_mode)
    )


def get_actions_by_timestamp_with_chat_inclusive(
    chat_id: str,
    timestamp_start: float,
    timestamp_end: float,
    limit: int = 0,
    limit_mode: str = "latest",
) -> list[ActionRecord]:
    """获取在特定聊天从指定时间戳到指定时间戳的动作记录（包含边界）。"""
    query = ActionRecord.select().where(
        (ActionRecord.chat_id == chat_id)
        & (ActionRecord.time >= timestamp_start)
        & (ActionRecord.time <= timestamp_end)
    )
    return _materialize_query(
        _apply_limit(query, ActionRecord.time, limit, limit_mode)
    )


def get_raw_msg_by_timestamp_random(
    timestamp_start: float,
    timestamp_end: float,
    limit: int = 0,
    limit_mode: str = "latest",
) -> list[MessageRecord]:
    """先在范围时间戳内随机选择一条消息，取得消息的chat_id，然后根据chat_id获取该聊天在指定时间戳范围内的消息。"""
    candidates = get_raw_msg_by_timestamp(timestamp_start, timestamp_end)
    if not candidates:
        return []
    chosen_message = random.choice(candidates)
    return get_raw_msg_by_timestamp_with_chat(
        chat_id=chosen_message.chat_id,
        timestamp_start=chosen_message.time,
        timestamp_end=timestamp_end,
        limit=limit,
        limit_mode=limit_mode,
    )


def num_new_messages_since(
    chat_id: str,
    timestamp_start: float = 0.0,
    timestamp_end: Optional[float] = None,
) -> int:
    """检查特定聊天从timestamp_start到timestamp_end之间有多少新消息。"""
    end_time = timestamp_end if timestamp_end is not None else time.time()
    if timestamp_start >= end_time:
        return 0
    return (
        MessageRecord.select()
        .where(
            (MessageRecord.chat_id == chat_id)
            & (MessageRecord.time > timestamp_start)
            & (MessageRecord.time < end_time)
        )
        .count()
    )


def num_new_messages_since_with_users(
    chat_id: str,
    timestamp_start: float,
    timestamp_end: float,
    person_ids: list[str],
) -> int:
    """检查某些特定用户在特定聊天在指定时间戳之间有多少新消息。"""
    if not person_ids or timestamp_start >= timestamp_end:
        return 0
    return (
        MessageRecord.select()
        .where(
            (MessageRecord.chat_id == chat_id)
            & (MessageRecord.time > timestamp_start)
            & (MessageRecord.time < timestamp_end)
            & (MessageRecord.user_id.in_(person_ids))
        )
        .count()
    )


def _normalize_text(content: str, remove_emoji_stickers: bool) -> str:
    """规范化文本内容，处理用户引用和表情包。"""
    updated = replace_user_references(content)
    if remove_emoji_stickers:
        updated = re.sub(r"\[表情包：[^\]]+\]", "", updated)
    return updated.strip()


def _replace_picture_tokens(
    content: str,
    picture_mapping: dict[str, str],
    picture_descriptions: dict[str, str],
    single_picture: bool,
    pic_counter: dict[str, int],
) -> str:
    """处理内容中的图片ID，将其替换为[图片x]格式。"""
    pattern = re.compile(r"\[picid:([^\]]+)\]")

    def replace_match(match: re.Match[str]) -> str:
        picture_id = match.group(1)
        if picture_id not in picture_mapping:
            picture_mapping[picture_id] = f"图片{pic_counter['count']}"
            pic_counter["count"] += 1
        picture_name = picture_mapping[picture_id]
        if picture_id not in picture_descriptions:
            picture_descriptions[picture_id] = translate_pid_to_description(
                picture_id
            )
        if single_picture:
            return f"[图片：{picture_descriptions[picture_id]}]"
        return f"[{picture_name}]"

    return pattern.sub(replace_match, content)


def _parse_action_data(raw_value: str) -> dict[str, Any]:
    """解析动作数据字段。"""
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, dict):
            return parsed
    except Exception as _e:
        logger.debug(f"异常: {_e}")
    try:
        parsed = ast.literal_eval(raw_value)
        if isinstance(parsed, dict):
            return parsed
    except Exception as _e:
        logger.debug(f"异常: {_e}")
    return {}


def _describe_action(action: ActionRecord) -> str:
    """生成动作记录的可读描述。"""
    payload = _parse_action_data(action.action_data or "")
    if action.action_name == "no_reply":
        return ""
    target_message_id = str(payload.get("target_message_id", "")).strip()
    reply_text = str(
        payload.get("reply_text")
        or payload.get("force_reply_text")
        or payload.get("content")
        or ""
    ).strip()
    if action.action_name == "reply" and reply_text:
        if target_message_id:
            return f"你回复了 {target_message_id}：{reply_text}"
        return f"你进行了回复：{reply_text}"
    if reply_text:
        return f"执行动作 {action.action_name}，内容：{reply_text}"
    unknown_words = payload.get("unknown_words", [])
    if isinstance(unknown_words, list):
        cleaned_words = [
            str(item).strip() for item in unknown_words if str(item).strip()
        ]
        if cleaned_words:
            return f"执行动作 {
                action.action_name} ，关注的陌生词：{
                '、'.join(cleaned_words[: 6])} "
    reason = (action.action_reason or "").strip()
    if reason:
        return f"执行动作 {action.action_name}，理由：{reason}"
    if payload:
        return f"执行动作 {action.action_name}，参数：{payload}"
    return f"执行动作 {action.action_name}"


def build_pic_mapping_info(pic_id_mapping: dict[str, str]) -> str:
    """构建图片映射信息字符串，显示图片的具体描述内容。"""
    if not pic_id_mapping:
        return ""
    lines = []
    ordered_items = sorted(
        pic_id_mapping.items(),
        key=lambda item: int(item[1].replace("图片", "")),
    )
    for picture_id, picture_name in ordered_items:
        lines.append(
            f"[{picture_name}] 的内容：{translate_pid_to_description(picture_id)}"
        )
    return "\n".join(lines)


def build_readable_actions(
    actions: list[ActionRecord], mode: str = "relative"
) -> str:
    """将动作列表转换为可读的文本格式。"""
    lines = []
    for action in actions:
        action_text = _describe_action(action)
        if not action_text:
            continue
        readable_time = translate_timestamp_to_human_readable(
            action.time, mode=mode
        )
        lines.append(f"{readable_time}, {action_text}")
    return "\n".join(lines)


def _truncate_message_by_position(
    content: str, position_percent: float
) -> str:
    """根据消息在列表中的位置百分比进行截断。"""
    original_len = len(content)
    limit = -1
    replace_content = ""
    if position_percent < 0.2:
        limit = 50
        replace_content = "......（记不清了）"
    elif position_percent < 0.5:
        limit = 100
        replace_content = "......（有点记不清了）"
    elif position_percent < 0.7:
        limit = 200
        replace_content = "......（内容太长了）"
    elif position_percent <= 1.0:
        limit = 400
        replace_content = "......（内容太长了）"
    if 0 < limit < original_len:
        return f"{content[:limit]}{replace_content}"
    return content


def _build_long_time_notice(
    current_timestamp: float, previous_timestamp: float
) -> str:
    """构建长时间间隔提示文本。"""
    time_diff = current_timestamp - previous_timestamp
    time_diff_hours = time_diff / 3600
    prev_date = time.strftime("%Y-%m-%d", time.localtime(previous_timestamp))
    current_date = time.strftime("%Y-%m-%d", time.localtime(current_timestamp))
    is_cross_day = prev_date != current_date
    if time_diff_hours > 8 or is_cross_day:
        current_time_struct = time.localtime(current_timestamp)
        year = current_time_struct.tm_year
        month = current_time_struct.tm_mon
        day = current_time_struct.tm_mday
        date_str = f"{year}年{month}月{day}日"
        hours_str = f"{int(time_diff_hours)}h"
        return (
            f"以下聊天开始时间：{date_str}。距离上一条消息过去了{hours_str}\n"
        )
    return ""


def _collect_visible_actions(
    messages: list[MessageRecord],
) -> list[ActionRecord]:
    """收集消息时间范围内的可见动作记录。"""
    if not messages:
        return []
    stream_id = messages[0].chat_id
    min_time = min(item.time for item in messages)
    max_time = max(item.time for item in messages)
    visible_actions = get_actions_by_timestamp_with_chat_inclusive(
        chat_id=stream_id,
        timestamp_start=min_time,
        timestamp_end=max_time,
    )
    trailing_actions = get_actions_by_timestamp_with_chat(
        chat_id=stream_id,
        timestamp_start=max_time,
        timestamp_end=max_time + 3600,
        limit=1,
        limit_mode="earliest",
    )
    merged_actions: list[ActionRecord] = []
    seen_ids: set[int] = set()
    for action in list(visible_actions) + list(trailing_actions):
        action_id = int(getattr(action, "id", 0) or 0)
        if action_id in seen_ids:
            continue
        seen_ids.add(action_id)
        if not getattr(action, "done", True):
            continue
        if not _describe_action(action):
            continue
        merged_actions.append(action)
    return merged_actions


def _build_readable_messages_internal(
    messages: list[MessageActionModel],
    replace_bot_name: bool = True,
    timestamp_mode: str = "relative",
    truncate: bool = False,
    pic_id_mapping: Optional[dict[str, str]] = None,
    pic_counter: int = 1,
    show_pic: bool = True,
    message_id_list: Optional[list[tuple[str, MessageRecord]]] = None,
    pic_single: bool = False,
    long_time_notice: bool = False,
) -> tuple[str, list[tuple[float, str, str]], dict[str, str], int]:
    """内部辅助函数，构建可读消息字符串和原始消息详情列表。"""
    if not messages:
        return "", [], pic_id_mapping or {}, pic_counter
    detailed_messages_raw: list[tuple[float, str, str, bool]] = []
    if pic_id_mapping is None:
        pic_id_mapping = {}
    current_pic_counter = pic_counter
    pic_description_cache: dict[str, str] = {}
    timestamp_to_id_mapping: dict[float, str] = {}
    if message_id_list:
        for msg_id, msg in message_id_list:
            timestamp = msg.time
            if timestamp is not None:
                timestamp_to_id_mapping[timestamp] = msg_id

    def process_pic_ids(content: str) -> str:
        """处理内容中的图片ID，将其替换为[图片x]格式。"""
        pic_pattern = r"\[picid:([^\]]+)\]"

        def replace_pic_id(match: re.Match) -> str:
            nonlocal current_pic_counter
            pic_id = match.group(1)
            if pic_single:
                if pic_id not in pic_description_cache:
                    description = "内容正在阅读，请稍等"
                    try:
                        from src.common.database.database_model import Images

                        if Images is not None:
                            record = Images.get_or_none(
                                Images.image_id == pic_id
                            )
                            if record and getattr(record, "description", ""):
                                description = str(record.description).strip()
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
                    pic_description_cache[pic_id] = description
                return f"[图片：{pic_description_cache[pic_id]}]"
            if pic_id not in pic_id_mapping:
                pic_id_mapping[pic_id] = f"图片{current_pic_counter}"
                current_pic_counter += 1
            return f"[{pic_id_mapping[pic_id]}]"

        return re.sub(pic_pattern, replace_pic_id, content)

    for message in messages:
        if message.is_action_record:
            content = process_pic_ids(message.display_message)
            detailed_messages_raw.append(
                (message.time, message.user_nickname, content, True)
            )
            continue
        platform = message.user_platform
        user_id = message.user_id
        user_nickname = message.user_nickname
        user_cardname = message.user_cardname
        timestamp = message.time
        content = message.display_message or message.processed_plain_text or ""
        if show_pic:
            content = process_pic_ids(content)
        if not all([platform, user_id, timestamp is not None]):
            continue
        person = Person(platform=platform, user_id=user_id)
        person_name = (
            person.person_name
            or f"{user_nickname}"
            or (f"昵称：{user_cardname}" if user_cardname else "某人")
        )
        if replace_bot_name and _is_bot_user(user_id):
            person_name = f"{global_config.bot.nickname}(你)"
        if content := replace_user_references(
            content, platform, replace_bot_name=replace_bot_name
        ):
            if message.is_command:
                content = f"[is_command=True] {content}"
            detailed_messages_raw.append(
                (timestamp, person_name, content, False)
            )

    if not detailed_messages_raw:
        return "", [], pic_id_mapping, current_pic_counter
    detailed_messages_raw.sort(key=lambda x: x[0])
    detailed_message: list[tuple[float, str, str, bool]] = []
    messages_count = len(detailed_messages_raw)
    if truncate and messages_count > 0:
        for i, (timestamp, name, content, is_action) in enumerate(
            detailed_messages_raw
        ):
            if is_action:
                detailed_message.append((timestamp, name, content, is_action))
                continue
            percentile = i / messages_count
            truncated_content = _truncate_message_by_position(
                content, percentile
            )
            detailed_message.append(
                (timestamp, name, truncated_content, is_action)
            )
    else:
        detailed_message = detailed_messages_raw

    output_lines: list[str] = []
    prev_timestamp: Optional[float] = None
    for timestamp, name, content, is_action in detailed_message:
        if long_time_notice and prev_timestamp is not None:
            notice = _build_long_time_notice(timestamp, prev_timestamp)
            if notice:
                output_lines.append(notice)
        readable_time = translate_timestamp_to_human_readable(
            timestamp, mode=timestamp_mode
        )
        message_id = timestamp_to_id_mapping.get(timestamp, "")
        id_prefix = f"[{message_id}]" if message_id else ""
        if is_action:
            output_lines.append(f"{id_prefix}{readable_time}, {content}")
        else:
            output_lines.append(
                f"{id_prefix}{readable_time}, {name}: {content}"
            )
        output_lines.append("\n")
        prev_timestamp = timestamp
    formatted_string = "".join(output_lines).strip()
    return (
        formatted_string,
        [
            (t, n, c)
            for t, n, c, is_action in detailed_message
            if not is_action
        ],
        pic_id_mapping,
        current_pic_counter,
    )


def build_readable_messages(
    messages: list[MessageRecord],
    replace_bot_name: bool = True,
    timestamp_mode: str = "relative",
    read_mark: float = 0.0,
    truncate: bool = False,
    show_actions: bool = False,
    show_pic: bool = True,
    message_id_list: Optional[list[tuple[str, MessageRecord]]] = None,
    remove_emoji_stickers: bool = False,
    pic_single: bool = False,
    long_time_notice: bool = False,
) -> str:
    """将消息列表转换为可读的文本格式。如果提供了read_mark，则在相应位置插入已读标记。"""
    if not messages:
        return ""
    channel_id = str(messages[0].chat_id or "")
    if channel_id:
        builder = get_content_builder(channel_id)
        for msg in messages[-20:]:
            speaker = _resolve_person_name(
                platform=str(
                    msg.chat_info_platform or msg.user_platform or "qq"
                ),
                user_id=str(msg.user_id or ""),
                fallback_name=str(msg.user_nickname or msg.user_id or "某人"),
                replace_bot_name=replace_bot_name,
            )
            builder.build(
                speaker=speaker,
                content=str(
                    msg.processed_plain_text or msg.display_message or ""
                ),
                timestamp=float(msg.time or 0.0),
            )
    if remove_emoji_stickers:
        filtered_messages = []
        for msg in messages:
            content = msg.processed_plain_text or msg.display_message or ""
            emoji_pattern = r"\[表情包：[^\]]+\]"
            content = re.sub(emoji_pattern, "", content)
            if content.strip():
                filtered_messages.append(msg)
        messages = filtered_messages
    copy_messages: list[MessageActionModel] = []
    for msg in messages:
        model = MessageActionModel.from_message_record(msg)
        if remove_emoji_stickers and model.processed_plain_text:
            model.processed_plain_text = re.sub(
                r"\[表情包：[^\]]+\]", "", model.processed_plain_text
            )
        copy_messages.append(model)
    if show_actions and copy_messages:
        min_time = min(msg.time or 0 for msg in copy_messages)
        max_time = max(msg.time or 0 for msg in copy_messages)
        chat_id = messages[0].chat_id if messages else None
        actions_in_range = get_actions_by_timestamp_with_chat_inclusive(
            chat_id=chat_id,
            timestamp_start=min_time,
            timestamp_end=max_time,
        )
        trailing_action = get_actions_by_timestamp_with_chat(
            chat_id=chat_id,
            timestamp_start=max_time,
            timestamp_end=max_time + 3600,
            limit=1,
            limit_mode="earliest",
        )
        all_actions = list(actions_in_range) + list(trailing_action)
        for action in all_actions:
            if getattr(action, "done", True):
                action_msg = MessageActionModel.from_action_record(
                    action, global_config.bot.nickname
                )
                copy_messages.append(action_msg)
        copy_messages.sort(key=lambda x: x.time or 0)

    if read_mark <= 0:
        formatted_string, _, pic_id_mapping, _ = (
            _build_readable_messages_internal(
                copy_messages,
                replace_bot_name,
                timestamp_mode,
                truncate,
                show_pic=show_pic,
                message_id_list=message_id_list,
                pic_single=pic_single,
                long_time_notice=long_time_notice,
            )
        )
        if not pic_single:
            pic_mapping_info = build_pic_mapping_info(pic_id_mapping)
            if pic_mapping_info:
                return f"{pic_mapping_info}\n\n{formatted_string}"
        return formatted_string
    else:
        messages_before_mark = [
            msg for msg in copy_messages if (msg.time or 0) <= read_mark
        ]
        messages_after_mark = [
            msg for msg in copy_messages if (msg.time or 0) > read_mark
        ]
        pic_id_mapping = {}
        pic_counter = 1
        formatted_before, _, pic_id_mapping, pic_counter = (
            _build_readable_messages_internal(
                messages_before_mark,
                replace_bot_name,
                timestamp_mode,
                truncate,
                pic_id_mapping,
                pic_counter,
                show_pic=show_pic,
                message_id_list=message_id_list,
                pic_single=pic_single,
                long_time_notice=long_time_notice,
            )
        )
        formatted_after, _, pic_id_mapping, _ = (
            _build_readable_messages_internal(
                messages_after_mark,
                replace_bot_name,
                timestamp_mode,
                False,
                pic_id_mapping,
                pic_counter,
                show_pic=show_pic,
                message_id_list=message_id_list,
                pic_single=pic_single,
                long_time_notice=long_time_notice,
            )
        )
        read_mark_line = (
            "\n--- 以上消息是你已经看过，请关注以下未读的新消息---\n"
        )
        if not pic_single:
            if pic_id_mapping:
                pic_mapping_info = f"图片信息：\n{
                    build_pic_mapping_info(pic_id_mapping)}\n聊天记录信息：\n"
            else:
                pic_mapping_info = "聊天记录信息：\n"
        else:
            pic_mapping_info = ""
        result_parts = []
        if pic_mapping_info:
            result_parts.extend((pic_mapping_info, "\n"))
        if formatted_before and formatted_after:
            result_parts.extend(
                [formatted_before, read_mark_line, formatted_after]
            )
        elif formatted_before:
            result_parts.extend([formatted_before, read_mark_line])
        elif formatted_after:
            result_parts.extend([read_mark_line, formatted_after])
        else:
            result_parts.append(read_mark_line.strip())
        return "".join(result_parts)


async def build_readable_messages_with_list(
    messages: list[MessageRecord],
    replace_bot_name: bool = True,
    timestamp_mode: str = "relative",
    truncate: bool = False,
    pic_single: bool = False,
) -> tuple[str, list[tuple[float, str, str]]]:
    """将消息列表转换为可读的文本格式，并返回原始(时间戳, 昵称, 内容)列表。"""
    formatted_string, details_list, pic_id_mapping, _ = (
        _build_readable_messages_internal(
            [MessageActionModel.from_message_record(msg) for msg in messages],
            replace_bot_name,
            timestamp_mode,
            truncate,
            pic_id_mapping=None,
            pic_counter=1,
            show_pic=True,
            message_id_list=None,
            pic_single=pic_single,
            long_time_notice=False,
        )
    )
    if not pic_single:
        if pic_mapping_info := build_pic_mapping_info(pic_id_mapping):
            formatted_string = f"{pic_mapping_info}\n\n{formatted_string}"
    return formatted_string, details_list


def build_readable_messages_with_id(
    messages: list[MessageRecord],
    replace_bot_name: bool = True,
    timestamp_mode: str = "relative",
    read_mark: float = 0.0,
    truncate: bool = False,
    show_actions: bool = False,
    show_pic: bool = True,
    remove_emoji_stickers: bool = False,
    pic_single: bool = False,
) -> tuple[str, list[tuple[str, MessageRecord]]]:
    """将消息列表转换为可读的文本格式，并返回原始(消息ID, 消息记录)列表。"""
    message_id_list = assign_message_ids(messages)
    formatted_string = build_readable_messages(
        messages=messages,
        replace_bot_name=replace_bot_name,
        timestamp_mode=timestamp_mode,
        truncate=truncate,
        show_actions=show_actions,
        show_pic=show_pic,
        read_mark=read_mark,
        message_id_list=message_id_list,
        remove_emoji_stickers=remove_emoji_stickers,
        pic_single=pic_single,
    )
    return formatted_string, message_id_list


async def build_anonymous_messages(
    messages: list[MessageRecord], show_ids: bool = False
) -> str:
    """构建匿名可读消息，将不同人的名称转为唯一占位符（A、B、C...），bot自己用SELF。"""
    if not messages:
        logger.warning("没有消息，无法构建匿名消息")
        return ""
    person_map: dict[str, str] = {}
    current_char = ord("A")
    output_lines: list[str] = []
    pic_id_mapping: dict[str, str] = {}
    pic_counter = 1
    pic_description_cache: dict[str, str] = {}

    def process_pic_ids(content: str) -> str:
        """处理内容中的图片ID。"""
        nonlocal pic_counter
        pic_pattern = r"\[picid:([^\]]+)\]"

        def replace_pic_id(match: re.Match) -> str:
            nonlocal pic_counter
            pic_id = match.group(1)
            if pic_id not in pic_id_mapping:
                pic_id_mapping[pic_id] = f"图片{pic_counter}"
                pic_counter += 1
            return f"[{pic_id_mapping[pic_id]}]"

        return re.sub(pic_pattern, replace_pic_id, content)

    def get_anon_name(platform: str, user_id: str) -> str:
        """获取匿名占位符名称。"""
        if _is_bot_user(user_id):
            return "SELF"
        try:
            person = Person(platform=platform, user_id=user_id)
            person_id = getattr(person, "person_id", "") or user_id
        except Exception:
            person_id = user_id
        if not person_id:
            return "?"
        if person_id not in person_map:
            nonlocal current_char
            person_map[person_id] = chr(current_char)
            current_char += 1
        return person_map[person_id]

    for i, msg in enumerate(messages):
        try:
            platform = str(msg.chat_info_platform or msg.user_platform or "qq")
            user_id = str(msg.user_id or "")
            content = msg.processed_plain_text or msg.display_message or ""
            content = process_pic_ids(content)
            anon_name = get_anon_name(platform, user_id)

            def anon_name_resolver(
                inner_platform: str, inner_user_id: str
            ) -> str:
                try:
                    return get_anon_name(inner_platform, inner_user_id)
                except Exception:
                    return "?"

            content = replace_user_references(
                content, platform, anon_name_resolver, replace_bot_name=False
            )
            if show_ids:
                header = f"[{i + 1}] {anon_name}说 "
            else:
                header = f"{anon_name}说 "
            output_lines.append(header)
            stripped_line = content.strip()
            if stripped_line:
                if stripped_line.endswith("。"):
                    stripped_line = stripped_line[:-1]
                output_lines.append(stripped_line)
            output_lines.append("\n")
        except Exception:
            continue
    final_output_lines: list[str] = []
    pic_mapping_info = build_pic_mapping_info(pic_id_mapping)
    if pic_mapping_info:
        final_output_lines.append(pic_mapping_info)
        final_output_lines.append("\n\n")
    final_output_lines.extend(output_lines)
    return "".join(final_output_lines).strip()


async def get_person_id_list(messages: list[dict[str, Any]]) -> list[str]:
    """从消息列表中提取不重复的person_id列表（忽略机器人自身）。"""
    person_ids_set: set[str] = set()
    for msg in messages:
        platform = str(msg.get("platform") or msg.get("user_platform") or "qq")
        user_id = str(msg.get("user_id") or "")
        if not all([platform, user_id]) or _is_bot_user(user_id):
            continue
        try:
            person = Person(platform=platform, user_id=user_id)
            person_id = getattr(person, "person_id", "")
            if person_id:
                person_ids_set.add(person_id)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
    return list(person_ids_set)


async def build_bare_messages(messages: list[MessageRecord]) -> str:
    """构建简化版消息字符串，只包含processed_plain_text内容，不考虑用户名和时间戳。"""
    if not messages:
        return ""
    output_lines: list[str] = []
    for msg in messages:
        content = msg.processed_plain_text or msg.display_message or ""
        pic_pattern = r"\[picid:[^\]]+\]"
        content = re.sub(pic_pattern, "[图片]", content)
        reply_pattern = r"回复<[^:<>]+:[^:<>]+>"
        content = re.sub(reply_pattern, "回复[某人]", content)
        at_pattern = r"@<[^:<>]+:[^:<>]+>"
        content = re.sub(at_pattern, "@[某人]", content)
        content = content.strip()
        if content:
            output_lines.append(content)
    return "\n".join(output_lines)

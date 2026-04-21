import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("semantic_router")

_router_singleton = None


class MessageCategory(Enum):
    """消息语义类别"""

    PLAIN_TEXT = "plain_text"
    AT_MENTION = "at_mention"
    QUOTE_REPLY = "quote_reply"
    FORWARD = "forward"
    MERGED_FORWARD = "merged_forward"
    IMAGE = "image"
    STICKER = "sticker"
    MULTI_IMAGE = "multi_image"
    VOICE = "voice"
    VIDEO = "video"
    MUSIC_SHARE = "music_share"
    FILE = "file"
    SYSTEM_NOTICE = "system_notice"
    RED_PACKET = "red_packet"
    RECALL = "recall"
    UNKNOWN = "unknown"

    def is_media(self) -> bool:
        return self in (
            MessageCategory.IMAGE,
            MessageCategory.STICKER,
            MessageCategory.MULTI_IMAGE,
            MessageCategory.VOICE,
            MessageCategory.VIDEO,
            MessageCategory.FILE,
        )

    def is_interactive(self) -> bool:
        return self in (
            MessageCategory.AT_MENTION,
            MessageCategory.QUOTE_REPLY,
        )

    def label(self) -> str:
        _labels = {
            "plain_text": "纯文本",
            "at_mention": "@提及",
            "quote_reply": "引用回复",
            "forward": "转发消息",
            "merged_forward": "合并转发",
            "image": "单图",
            "sticker": "表情包",
            "multi_image": "多图",
            "voice": "语音",
            "video": "视频",
            "music_share": "音乐分享",
            "file": "文件",
            "system_notice": "系统通知",
            "red_packet": "红包",
            "recall": "撤回",
            "unknown": "未知",
        }
        return _labels.get(self.value, "未知")


@dataclass
class SemanticRouteResult:
    """语义路由结果"""

    category: MessageCategory = MessageCategory.PLAIN_TEXT
    text_content: str = ""
    has_text: bool = False
    media_count: int = 0
    is_at_bot: bool = False
    is_quote_to_bot: bool = False
    mentioned_user_ids: List[str] = field(default_factory=list)
    reply_anchor: str = ""
    quoted_preview: str = ""
    has_reply_chain: bool = False
    is_name_mention: bool = False
    is_merged_forward: bool = False
    has_music_share: bool = False
    forwarded_source: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "category_label": self.category.label(),
            "has_text": self.has_text,
            "media_count": self.media_count,
            "is_at_bot": self.is_at_bot,
            "is_quote_to_bot": self.is_quote_to_bot,
            "mentioned_user_ids": self.mentioned_user_ids[:8],
            "reply_anchor": self.reply_anchor,
            "quoted_preview": self.quoted_preview,
            "has_reply_chain": self.has_reply_chain,
            "is_name_mention": self.is_name_mention,
            "is_merged_forward": self.is_merged_forward,
            "has_music_share": self.has_music_share,
            "confidence": round(self.confidence, 2),
        }


class MessageSemanticRouter:
    """消息语义路由器
    将原始消息对象归类为统一的语义类别，
    供下游门控和决策链使用。
    """

    def __init__(self):
        self._bot_id: str = ""
        self._bot_names: List[str] = []
        self._load_bot_identity()

    def classify(self, message: Any) -> SemanticRouteResult:
        """对单条消息进行语义分类"""
        result = SemanticRouteResult()
        text = str(
            getattr(message, "processed_plain_text", "")
            or getattr(message, "plain_text", "")
            or ""
        ).strip()
        result.text_content = text
        result.has_text = bool(text)
        # 检测@提及
        is_at = bool(getattr(message, "is_at", False))
        is_mentioned = bool(getattr(message, "is_mentioned", False))
        at_list = getattr(message, "at_list", None) or []
        result.mentioned_user_ids = [
            str(a).strip() for a in at_list if str(a).strip()
        ]
        if is_at or is_mentioned:
            result.is_at_bot = True
        if self._bot_id and self._bot_id in [str(a) for a in at_list]:
            result.is_at_bot = True
        if text and self._bot_names:
            result.is_name_mention = any(
                name and name in text for name in self._bot_names
            )
        # 检测引用回复
        reply_to = str(getattr(message, "reply_to", "") or "").strip()
        quoted_content = str(
            getattr(message, "quoted_content", "") or ""
        ).strip()
        has_quote = bool(reply_to or quoted_content)
        result.reply_anchor = reply_to
        result.quoted_preview = quoted_content[:60]
        result.has_reply_chain = bool(reply_to)
        # 检测媒体内容
        image_list = getattr(message, "image_list", None) or []
        has_image = bool(getattr(message, "has_image", False)) or bool(
            image_list
        )
        sticker = getattr(message, "sticker", None) or getattr(
            message, "face_id", None
        )
        voice = getattr(message, "voice", None) or getattr(
            message, "record", None
        )
        video = getattr(message, "video", None)
        music_share = (
            getattr(message, "music", None)
            or getattr(message, "music_share", None)
            or getattr(message, "share_type", None) == "music"
            or getattr(message, "sub_type", None) == "music"
        )
        file_info = getattr(message, "file", None)
        forward = getattr(message, "forward", None) or getattr(
            message, "forwarded_msg", None
        )
        recall = getattr(message, "recall", None) or getattr(
            message, "revoke", None
        )
        system_msg = getattr(message, "system_msg", None) or getattr(
            message, "notice_type", None
        )
        # 分类优先级：系统 > 撤回 > 转发 > @> 引用 > 多图 > 单图 > 表情 > 语音 > 视频 > 文件 > 文本
        if system_msg:
            result.category = MessageCategory.SYSTEM_NOTICE
            return result
        if recall:
            result.category = MessageCategory.RECALL
            return result
        if forward:
            forward_nodes = (
                getattr(forward, "nodes", None)
                or getattr(forward, "message_list", None)
                or getattr(forward, "messages", None)
                or []
            )
            result.is_merged_forward = (
                isinstance(forward_nodes, (list, tuple))
                and len(forward_nodes) > 1
            )
            result.category = (
                MessageCategory.MERGED_FORWARD
                if result.is_merged_forward
                else MessageCategory.FORWARD
            )
            result.forwarded_source = str(getattr(forward, "source", "") or "")
            return result
        if music_share:
            result.category = MessageCategory.MUSIC_SHARE
            result.has_music_share = True
            return result
        if result.is_at_bot:
            result.category = MessageCategory.AT_MENTION
            return result
        if has_quote:
            result.category = MessageCategory.QUOTE_REPLY
            # 判定是否引用的是 bot 的消息
            if self._is_quote_to_bot(reply_to, quoted_content):
                result.is_quote_to_bot = True
            return result
        if has_image:
            image_count = len(image_list) if image_list else 1
            result.media_count = image_count
            if image_count > 1:
                result.category = MessageCategory.MULTI_IMAGE
            elif sticker:
                result.category = MessageCategory.STICKER
            else:
                result.category = MessageCategory.IMAGE
            return result
        if sticker:
            result.category = MessageCategory.STICKER
            return result
        if voice:
            result.category = MessageCategory.VOICE
            result.media_count = 1
            return result
        if video:
            result.category = MessageCategory.VIDEO
            result.media_count = 1
            return result
        if file_info:
            result.category = MessageCategory.FILE
            result.media_count = 1
            return result
        # 默认纯文本
        result.category = MessageCategory.PLAIN_TEXT
        return result

    def classify_batch(self, messages: List[Any]) -> List[SemanticRouteResult]:
        return [self.classify(msg) for msg in messages]

    def summarize_batch(
        self, results: List[SemanticRouteResult]
    ) -> Dict[str, int]:
        """统计批次中各类别的消息数"""
        counts: Dict[str, int] = {}
        for r in results:
            key = r.category.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ────────────────── 内部方法 ──────────────────

    def _load_bot_identity(self) -> None:
        """加载 bot 身份信息用于检测自我相关"""
        try:
            from src.config.config import global_config

            self._bot_id = str(
                getattr(global_config, "bot_qq", "")
                or getattr(global_config, "bot_id", "")
                or ""
            )
            nick = str(
                getattr(global_config, "bot_nickname", "")
                or getattr(global_config, "nickname", "")
                or ""
            )
            names = [nick] if nick else []
            aliases = (
                getattr(global_config, "bot_alias", None)
                or getattr(global_config, "nicknames", None)
                or []
            )
            if isinstance(aliases, (list, tuple)):
                names.extend(str(a) for a in aliases if a)
            self._bot_names = [n for n in names if n]
        except Exception:
            self._bot_id = ""
            self._bot_names = []

    def _is_quote_to_bot(self, reply_to: str, quoted_content: str) -> bool:
        """判断引用的消息是否来自 bot"""
        if not reply_to and not quoted_content:
            return False
        try:
            from src.core.self_reply_recognizer import (
                get_self_reply_recognizer,
            )

            recognizer = get_self_reply_recognizer()
            if recognizer.get_bot_message(reply_to) is not None:
                return True
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # 降级：检查引用内容是否含 bot 名字
        if quoted_content and self._bot_names:
            for name in self._bot_names:
                if name in quoted_content:
                    return True
        return False


def get_semantic_router() -> MessageSemanticRouter:
    global _router_singleton
    if _router_singleton is None:
        _router_singleton = MessageSemanticRouter()
    return _router_singleton

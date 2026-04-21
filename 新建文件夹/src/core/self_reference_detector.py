import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("self_ref_detect")

_detector_singleton = None


class ReferenceType(Enum):
    """自我指称类型"""

    DIRECT_AT = "direct_at"
    NAME_MENTION = "name_mention"
    PRONOUN_REFER = "pronoun_refer"
    INDIRECT_TOPIC = "indirect_topic"
    QUOTE_BOT_MSG = "quote_bot_msg"
    NONE = "none"

    def strength(self) -> float:
        _strengths = {
            "direct_at": 1.0,
            "name_mention": 0.85,
            "quote_bot_msg": 0.9,
            "pronoun_refer": 0.5,
            "indirect_topic": 0.3,
            "none": 0.0,
        }
        return _strengths.get(self.value, 0.0)

    def label(self) -> str:
        _labels = {
            "direct_at": "直接@",
            "name_mention": "叫名字",
            "quote_bot_msg": "引用bot消息",
            "pronoun_refer": "代词指代",
            "indirect_topic": "间接话题相关",
            "none": "无关",
        }
        return _labels.get(self.value, "无关")


@dataclass
class SelfReferenceResult:
    """自我指称检测结果"""

    ref_type: ReferenceType = ReferenceType.NONE
    strength: float = 0.0
    matched_pattern: str = ""
    context_snippet: str = ""

    def is_relevant(self) -> bool:
        return self.ref_type != ReferenceType.NONE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref_type": self.ref_type.value,
            "ref_label": self.ref_type.label(),
            "strength": round(self.strength, 3),
            "matched_pattern": self.matched_pattern,
        }


# 第三人称代词模式
_PRONOUN_PATTERNS = [
    re.compile(r"(?:它|这个?(?:bot|机器人|AI|人工智能))", re.IGNORECASE),
    re.compile(
        r"(?:那个?(?:bot|机器人|AI)|这?(?:机器人|bot)(?:怎么|咋|能|会|是))",
        re.IGNORECASE,
    ),
]
# 间接话题关键词
_INDIRECT_KEYWORDS = [
    "自动回复",
    "智能回复",
    "回复速度",
    "回消息",
    "bot功能",
    "机器人功能",
    "聊天机器",
    "人工智障",
]


class SelfReferenceDetector:
    """自我指称检测器
    判断消息内容是否在谈论/指代 bot 本身。
    """

    def __init__(self):
        self._bot_id: str = ""
        self._bot_names: List[str] = []
        self._name_patterns: List[re.Pattern] = []
        self._initialized = False

    def _ensure_init(self) -> None:
        if self._initialized:
            return
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
            aliases = (
                getattr(global_config, "bot_alias", None)
                or getattr(global_config, "nicknames", None)
                or []
            )
            names = []
            if nick:
                names.append(nick)
            if isinstance(aliases, (list, tuple)):
                names.extend(str(a) for a in aliases if a)
            self._bot_names = [n.strip() for n in names if n.strip()]
            # 为每个名字构建正则
            for name in self._bot_names:
                escaped = re.escape(name)
                self._name_patterns.append(re.compile(escaped, re.IGNORECASE))
        except Exception:
            self._bot_names = []
        self._initialized = True

    def detect(self, message: Any) -> SelfReferenceResult:
        """检测消息是否在指代 bot"""
        self._ensure_init()
        text = str(
            getattr(message, "processed_plain_text", "")
            or getattr(message, "plain_text", "")
            or ""
        ).strip()
        # 优先级 1: 直接@
        is_at = bool(getattr(message, "is_at", False))
        is_mentioned = bool(getattr(message, "is_mentioned", False))
        if is_at or is_mentioned:
            return SelfReferenceResult(
                ref_type=ReferenceType.DIRECT_AT,
                strength=1.0,
                matched_pattern="@提及",
                context_snippet=text[:80],
            )
        # 优先级 2: 引用 bot 消息
        reply_to = str(getattr(message, "reply_to", "") or "").strip()
        if reply_to:
            try:
                from src.core.self_reply_recognizer import (
                    get_self_reply_recognizer,
                )

                recognizer = get_self_reply_recognizer()
                if recognizer.get_bot_message(reply_to) is not None:
                    return SelfReferenceResult(
                        ref_type=ReferenceType.QUOTE_BOT_MSG,
                        strength=0.9,
                        matched_pattern="引用bot消息",
                        context_snippet=text[:80],
                    )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} 异常: {_e}")
        if not text:
            return SelfReferenceResult()
        # 优先级 3: 名字出现
        for idx, pattern in enumerate(self._name_patterns):
            match = pattern.search(text)
            if match:
                name = (
                    self._bot_names[idx] if idx < len(self._bot_names) else ""
                )
                return SelfReferenceResult(
                    ref_type=ReferenceType.NAME_MENTION,
                    strength=0.85,
                    matched_pattern=f"名字「{name}」",
                    context_snippet=text[:80],
                )
        # 优先级 4: 第三人称代词
        for pat in _PRONOUN_PATTERNS:
            match = pat.search(text)
            if match:
                return SelfReferenceResult(
                    ref_type=ReferenceType.PRONOUN_REFER,
                    strength=0.5,
                    matched_pattern=match.group(0),
                    context_snippet=text[:80],
                )
        # 优先级 5: 间接话题关键词
        text_lower = text.lower()
        for kw in _INDIRECT_KEYWORDS:
            if kw in text_lower:
                return SelfReferenceResult(
                    ref_type=ReferenceType.INDIRECT_TOPIC,
                    strength=0.3,
                    matched_pattern=kw,
                    context_snippet=text[:80],
                )
        return SelfReferenceResult()

    def detect_batch(self, messages: List[Any]) -> List[SelfReferenceResult]:
        return [self.detect(msg) for msg in messages]

    def has_any_reference(self, messages: List[Any]) -> bool:
        """批次中是否有任何消息在提及 bot"""
        return any(self.detect(msg).is_relevant() for msg in messages)

    def strongest_reference(self, messages: List[Any]) -> SelfReferenceResult:
        """批次中最强的自我指称"""
        best = SelfReferenceResult()
        for msg in messages:
            result = self.detect(msg)
            if result.strength > best.strength:
                best = result
        return best


def get_self_reference_detector() -> SelfReferenceDetector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = SelfReferenceDetector()
    return _detector_singleton

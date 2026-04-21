import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ActivityKind(Enum):
    INCOMING_MSG = "incoming_msg"
    BOT_DECISION = "bot_decision"
    AWAIT_BEGIN = "await_begin"
    AWAIT_SHIFT = "await_shift"
    REPLY_TIMELY = "reply_timely"
    REPLY_DELAYED = "reply_delayed"
    AWAIT_EXPIRED = "await_expired"
    SPONTANEOUS_TRIGGER = "spontaneous_trigger"

    def __str__(self) -> str:
        return self.value


class ConversationState(Enum):
    READY = "ready"
    AWAITING = "awaiting"

    def __str__(self) -> str:
        return self.value


@dataclass
class PendingConfig:
    anticipated_reply: str = ""
    timeout_seconds: int = 0
    initiated_at: float = 0.0
    last_reflect_at: float = 0.0
    reflect_count: int = 0
    chase_count: int = 0

    def is_active(self) -> bool:
        return self.timeout_seconds > 0 and self.initiated_at > 0

    def elapsed_seconds(self) -> float:
        if not self.is_active():
            return 0.0
        return time.time() - self.initiated_at

    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds() / 60

    def is_expired(self) -> bool:
        if not self.is_active():
            return False
        return self.elapsed_seconds() >= self.timeout_seconds

    def progress_ratio(self) -> float:
        if not self.is_active() or self.timeout_seconds <= 0:
            return 0.0
        return min(self.elapsed_seconds() / self.timeout_seconds, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anticipated_reply": self.anticipated_reply,
            "timeout_seconds": self.timeout_seconds,
            "initiated_at": self.initiated_at,
            "last_reflect_at": self.last_reflect_at,
            "reflect_count": self.reflect_count,
            "chase_count": self.chase_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PendingConfig":
        return cls(
            anticipated_reply=data.get("anticipated_reply", ""),
            timeout_seconds=data.get("timeout_seconds", 0),
            initiated_at=data.get("initiated_at", 0.0),
            last_reflect_at=data.get("last_reflect_at", 0.0),
            reflect_count=data.get("reflect_count", 0),
            chase_count=data.get("chase_count", 0),
        )

    def clear(self) -> None:
        self.anticipated_reply = ""
        self.timeout_seconds = 0
        self.initiated_at = 0.0
        self.last_reflect_at = 0.0
        self.reflect_count = 0
        self.chase_count = 0


@dataclass
class MindLogRecord:
    kind: ActivityKind
    recorded_at: float
    text_content: str = ""
    sender_name: str = ""
    sender_id: str = ""
    inner_thought: str = ""
    behavior_list: List[Dict] = field(default_factory=list)
    anticipated_reply: str = ""
    timeout_seconds: int = 0
    elapsed_time: float = 0.0
    reflect_thought: str = ""
    current_mood: str = ""
    extra_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": str(self.kind),
            "recorded_at": self.recorded_at,
            "text_content": self.text_content,
            "sender_name": self.sender_name,
            "sender_id": self.sender_id,
            "inner_thought": self.inner_thought,
            "behavior_list": self.behavior_list,
            "anticipated_reply": self.anticipated_reply,
            "timeout_seconds": self.timeout_seconds,
            "elapsed_time": self.elapsed_time,
            "reflect_thought": self.reflect_thought,
            "current_mood": self.current_mood,
            "extra_data": self.extra_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MindLogRecord":
        kind_str = data.get("kind", "incoming_msg")
        try:
            kind = ActivityKind(kind_str)
        except ValueError:
            kind = ActivityKind.INCOMING_MSG
        return cls(
            kind=kind,
            recorded_at=data.get("recorded_at", time.time()),
            text_content=data.get("text_content", ""),
            sender_name=data.get("sender_name", ""),
            sender_id=data.get("sender_id", ""),
            inner_thought=data.get("inner_thought", ""),
            behavior_list=data.get("behavior_list", []),
            anticipated_reply=data.get("anticipated_reply", ""),
            timeout_seconds=data.get("timeout_seconds", 0),
            elapsed_time=data.get("elapsed_time", 0.0),
            reflect_thought=data.get("reflect_thought", ""),
            current_mood=data.get("current_mood", ""),
            extra_data=data.get("extra_data", {}),
        )

    def format_time(self, pattern: str = "%H:%M") -> str:
        return time.strftime(pattern, time.localtime(self.recorded_at))


@dataclass
class BehaviorModel:
    action_type: str
    action_params: Dict[str, Any] = field(default_factory=dict)
    action_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.action_type}
        if self.action_reason:
            result["reason"] = self.action_reason
        result.update(self.action_params)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorModel":
        action_type = data.get("type", "idle")
        action_reason = data.get("reason", "")
        action_params = {k: v for k, v in data.items() if k not in ("type", "reason")}
        return cls(action_type=action_type, action_params=action_params, action_reason=action_reason)

    def describe(self) -> str:
        if self.action_type == "miao_reply":
            content = self.action_params.get("content", "")
            preview = content[:50] + "..." if len(content) > 50 else content
            return f'发送消息："{preview}"'
        elif self.action_type == "nudge_user":
            return "轻戳对方"
        elif self.action_type == "idle":
            return "保持沉默"
        elif self.action_type == "send_sticker":
            sticker = self.action_params.get("sticker", "")
            return f"发送表情：{sticker}"
        return f"执行行为：{self.action_type}"


@dataclass
class GeneratorResult:
    inner_thought: str
    behaviors: List[BehaviorModel]
    anticipated_reply: str = ""
    timeout_seconds: int = 0
    current_mood: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.inner_thought,
            "actions": [b.to_dict() for b in self.behaviors],
            "expected_reaction": self.anticipated_reply,
            "max_wait_seconds": self.timeout_seconds,
            "mood": self.current_mood,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratorResult":
        behaviors_data = data.get("actions", [])
        behaviors = [BehaviorModel.from_dict(b) for b in behaviors_data] if behaviors_data else []
        if not behaviors:
            behaviors = [BehaviorModel(action_type="idle")]
        timeout = data.get("max_wait_seconds", 0)
        try:
            timeout = int(timeout)
            timeout = max(0, min(timeout, 1800))
        except (ValueError, TypeError):
            timeout = 0
        return cls(
            inner_thought=data.get("thought", ""),
            behaviors=behaviors,
            anticipated_reply=data.get("expected_reaction", ""),
            timeout_seconds=timeout,
            current_mood=data.get("mood", ""),
        )

    @classmethod
    def create_fallback(cls, error_msg: str) -> "GeneratorResult":
        return cls(
            inner_thought=f"出现异常：{error_msg}",
            behaviors=[BehaviorModel(action_type="idle")],
            anticipated_reply="",
            timeout_seconds=0,
        )

    def contains_reply(self) -> bool:
        return any(b.action_type in ("miao_reply", "respond") for b in self.behaviors)

    def extract_reply_text(self) -> str:
        for behavior in self.behaviors:
            if behavior.action_type in ("miao_reply", "respond"):
                return behavior.action_params.get("content", "")
        return ""

    def summarize_behaviors(self) -> str:
        descriptions = [b.describe() for b in self.behaviors]
        return " + ".join(descriptions)


@dataclass
class ProactiveScores:
    topic_score: float = 0.0
    intimacy_score: float = 0.0
    timeliness_score: float = 0.0
    mood_score: float = 0.0
    activity_score: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "topic": self.topic_score,
            "intimacy": self.intimacy_score,
            "timeliness": self.timeliness_score,
            "mood": self.mood_score,
            "activity": self.activity_score,
            "final": self.final_score,
        }


@dataclass
class SilenceTimerState:
    stream_id: str
    silence_start: float = 0.0
    is_monitoring: bool = False
    trigger_threshold_hours: float = 2.0
    last_check: float = 0.0


@dataclass
class EngagementState:
    energy: float = 100.0
    max_energy: float = 100.0
    last_update: float = field(default_factory=time.time)
    consecutive_speaks: int = 0
    rest_level: int = 0
    rest_until: float = 0.0
    daily_speak_count: int = 0
    hourly_speak_count: int = 0
    last_hour_reset: float = field(default_factory=time.time)
    last_day_reset: float = field(default_factory=time.time)


@dataclass
class ScheduleState:
    last_private_time: float = 0.0
    last_group_time: float = 0.0
    daily_private_count: int = 0
    daily_group_count: int = 0
    last_day_reset: float = field(default_factory=time.time)


@dataclass
class TopicInfo:
    keywords: List[str] = field(default_factory=list)
    summary: str = ""
    relevance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keywords": self.keywords,
            "summary": self.summary,
            "relevance": self.relevance,
        }

    def is_interesting(self, threshold: float = 0.5) -> bool:
        return self.relevance >= threshold

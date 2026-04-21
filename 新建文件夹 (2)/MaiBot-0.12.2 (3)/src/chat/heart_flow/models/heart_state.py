import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(Enum):
    USER_MESSAGE = "user_message"
    BOT_THINKING = "bot_thinking"
    BOT_REPLY = "bot_reply"
    WAITING_START = "waiting_start"
    WAITING_UPDATE = "waiting_update"
    WAIT_TIMEOUT = "wait_timeout"
    PROACTIVE_TRIGGER = "proactive_trigger"
    REST_START = "rest_start"
    REST_END = "rest_end"
    REST_PEEK = "rest_peek"

    def __str__(self) -> str:
        return self.value


class HeartStatus(Enum):
    IDLE = "idle"
    WAITING = "waiting"
    RESTING = "resting"

    def __str__(self) -> str:
        return self.value


@dataclass
class WaitingConfig:
    # 等待对方回复时的参数
    expected_reaction: str = ""
    max_wait_seconds: int = 0
    started_at: float = 0.0
    last_thinking_at: float = 0.0
    thinking_count: int = 0
    followup_count: int = 0

    def is_active(self) -> bool:
        return self.max_wait_seconds > 0 and self.started_at > 0

    def get_elapsed_seconds(self) -> float:
        if not self.is_active():
            return 0.0
        return time.time() - self.started_at

    def get_elapsed_minutes(self) -> float:
        return self.get_elapsed_seconds() / 60.0

    def is_timeout(self) -> bool:
        if not self.is_active():
            return False
        return self.get_elapsed_seconds() >= self.max_wait_seconds

    def get_progress(self) -> float:
        if not self.is_active() or self.max_wait_seconds <= 0:
            return 0.0
        return min(self.get_elapsed_seconds() / self.max_wait_seconds, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_reaction": self.expected_reaction,
            "max_wait_seconds": self.max_wait_seconds,
            "started_at": self.started_at,
            "last_thinking_at": self.last_thinking_at,
            "thinking_count": self.thinking_count,
            "followup_count": self.followup_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WaitingConfig":
        return cls(
            expected_reaction=data.get("expected_reaction", ""),
            max_wait_seconds=data.get("max_wait_seconds", 0),
            started_at=data.get("started_at", 0.0),
            last_thinking_at=data.get("last_thinking_at", 0.0),
            thinking_count=data.get("thinking_count", 0),
            followup_count=data.get("followup_count", 0),
        )

    def reset(self) -> None:
        self.expected_reaction = ""
        self.max_wait_seconds = 0
        self.started_at = 0.0
        self.last_thinking_at = 0.0
        self.thinking_count = 0
        self.followup_count = 0


@dataclass
class RestConfig:
    # 休息时的参数配置
    reason: str = ""
    rest_duration: int = 0
    started_at: float = 0.0
    consecutive_skip_count: int = 0
    peek_probability: float = 0.05
    can_interrupt: bool = True

    def is_active(self) -> bool:
        return self.rest_duration > 0 and self.started_at > 0

    def get_elapsed_seconds(self) -> float:
        if not self.is_active():
            return 0.0
        return time.time() - self.started_at

    def is_finished(self) -> bool:
        if not self.is_active():
            return True
        return self.get_elapsed_seconds() >= self.rest_duration

    def get_remaining_seconds(self) -> float:
        if not self.is_active():
            return 0.0
        return max(0.0, self.rest_duration - self.get_elapsed_seconds())

    def get_progress(self) -> float:
        if not self.is_active() or self.rest_duration <= 0:
            return 0.0
        return min(self.get_elapsed_seconds() / self.rest_duration, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "rest_duration": self.rest_duration,
            "started_at": self.started_at,
            "consecutive_skip_count": self.consecutive_skip_count,
            "peek_probability": self.peek_probability,
            "can_interrupt": self.can_interrupt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RestConfig":
        return cls(
            reason=data.get("reason", ""),
            rest_duration=data.get("rest_duration", 0),
            started_at=data.get("started_at", 0.0),
            consecutive_skip_count=data.get("consecutive_skip_count", 0),
            peek_probability=data.get("peek_probability", 0.05),
            can_interrupt=data.get("can_interrupt", True),
        )

    def reset(self) -> None:
        self.reason = ""
        self.rest_duration = 0
        self.started_at = 0.0
        self.consecutive_skip_count = 0
        self.peek_probability = 0.05
        self.can_interrupt = True


@dataclass
class MentalEvent:
    # 心理活动事件节点
    event_type: EventType
    timestamp: float
    content: str = ""
    user_name: str = ""
    user_id: str = ""
    thought: str = ""
    actions: List[Dict] = field(default_factory=list)
    expected_reaction: str = ""
    max_wait_seconds: int = 0
    elapsed_seconds: float = 0.0
    waiting_thought: str = ""
    mood: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": str(self.event_type),
            "timestamp": self.timestamp,
            "content": self.content,
            "user_name": self.user_name,
            "user_id": self.user_id,
            "thought": self.thought,
            "actions": self.actions,
            "expected_reaction": self.expected_reaction,
            "max_wait_seconds": self.max_wait_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "waiting_thought": self.waiting_thought,
            "mood": self.mood,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MentalEvent":
        event_type_str = data.get("event_type", "user_message")
        try:
            event_type = EventType(event_type_str)
        except ValueError:
            event_type = EventType.USER_MESSAGE
        return cls(
            event_type=event_type,
            timestamp=data.get("timestamp", time.time()),
            content=data.get("content", ""),
            user_name=data.get("user_name", ""),
            user_id=data.get("user_id", ""),
            thought=data.get("thought", ""),
            actions=data.get("actions", []),
            expected_reaction=data.get("expected_reaction", ""),
            max_wait_seconds=data.get("max_wait_seconds", 0),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
            waiting_thought=data.get("waiting_thought", ""),
            mood=data.get("mood", ""),
            metadata=data.get("metadata", {}),
        )

    def get_time_str(self, fmt: str = "%H:%M") -> str:
        return time.strftime(fmt, time.localtime(self.timestamp))


@dataclass
class ActionIntent:
    # 内心决策的单个动作意图
    action_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.action_type}
        if self.reason:
            result["reason"] = self.reason
        result.update(self.params)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionIntent":
        action_type = data.get("type", "do_nothing")
        reason = data.get("reason", "")
        params = {k: v for k, v in data.items() if k not in ("type", "reason")}
        return cls(action_type=action_type, params=params, reason=reason)

    def get_description(self) -> str:
        if self.action_type == "reply":
            content = self.params.get("content", "")
            return f'发送消息："{content[:50]}{"..." if len(content) > 50 else ""}"'
        elif self.action_type == "poke":
            return "戳了戳对方"
        elif self.action_type == "do_nothing":
            return "什么都没做"
        elif self.action_type == "emoji":
            emoji = self.params.get("emoji", "")
            return f"发送表情：{emoji}"
        return f"执行动作：{self.action_type}"


@dataclass
class HeartState:
    # 心流状态容器，管理内心活动状态
    stream_id: str
    status: HeartStatus = HeartStatus.IDLE
    current_thought: str = ""
    waiting_config: WaitingConfig = field(default_factory=WaitingConfig)
    rest_config: RestConfig = field(default_factory=RestConfig)
    mental_log: List[MentalEvent] = field(default_factory=list)
    last_activity_at: float = field(default_factory=time.time)
    last_proactive_at: float = 0.0
    consecutive_timeout_count: int = 0

    def is_waiting(self) -> bool:
        return self.status == HeartStatus.WAITING and self.waiting_config.is_active()

    def is_resting(self) -> bool:
        return self.status == HeartStatus.RESTING and self.rest_config.is_active()

    def start_waiting(self, expected_reaction: str, max_wait_seconds: int) -> None:
        self.status = HeartStatus.WAITING
        self.waiting_config = WaitingConfig(
            expected_reaction=expected_reaction,
            max_wait_seconds=max_wait_seconds,
            started_at=time.time(),
        )
        self.add_event(MentalEvent(
            event_type=EventType.WAITING_START,
            timestamp=time.time(),
            expected_reaction=expected_reaction,
            max_wait_seconds=max_wait_seconds,
        ))

    def end_waiting(self) -> None:
        self.status = HeartStatus.IDLE
        self.waiting_config.reset()
        self.consecutive_timeout_count = 0

    def start_resting(self, reason: str, rest_duration: int, can_interrupt: bool = True) -> None:
        self.status = HeartStatus.RESTING
        self.rest_config = RestConfig(
            reason=reason,
            rest_duration=rest_duration,
            started_at=time.time(),
            can_interrupt=can_interrupt,
        )
        self.add_event(MentalEvent(
            event_type=EventType.REST_START,
            timestamp=time.time(),
            metadata={"reason": reason, "duration": rest_duration},
        ))

    def end_resting(self) -> None:
        self.status = HeartStatus.IDLE
        self.rest_config.reset()
        self.add_event(MentalEvent(
            event_type=EventType.REST_END,
            timestamp=time.time(),
        ))

    def add_event(self, event: MentalEvent) -> None:
        self.mental_log.append(event)
        self.last_activity_at = time.time()
        if len(self.mental_log) > 100:
            self.mental_log = self.mental_log[-80:]

    def add_user_message(self, content: str, user_name: str, user_id: str) -> None:
        self.add_event(MentalEvent(
            event_type=EventType.USER_MESSAGE,
            timestamp=time.time(),
            content=content,
            user_name=user_name,
            user_id=user_id,
        ))
        if self.is_waiting():
            self.end_waiting()
        if self.is_resting() and self.rest_config.can_interrupt:
            self.end_resting()

    def add_bot_thinking(self, thought: str, actions: List[Dict],
                         expected_reaction: str = "", max_wait_seconds: int = 0) -> None:
        self.add_event(MentalEvent(
            event_type=EventType.BOT_THINKING,
            timestamp=time.time(),
            thought=thought,
            actions=actions,
            expected_reaction=expected_reaction,
            max_wait_seconds=max_wait_seconds,
        ))
        if max_wait_seconds > 0:
            self.start_waiting(expected_reaction, max_wait_seconds)

    def add_waiting_update(self, waiting_thought: str, mood: str = "") -> None:
        self.add_event(MentalEvent(
            event_type=EventType.WAITING_UPDATE,
            timestamp=time.time(),
            waiting_thought=waiting_thought,
            mood=mood,
            elapsed_seconds=self.waiting_config.get_elapsed_seconds(),
        ))
        self.waiting_config.thinking_count += 1
        self.waiting_config.last_thinking_at = time.time()

    def get_recent_events(self, limit: int = 20) -> List[MentalEvent]:
        return self.mental_log[-limit:]

    def get_last_bot_reply(self) -> Optional[str]:
        for event in reversed(self.mental_log):
            if event.event_type == EventType.BOT_REPLY:
                return event.content
            if event.event_type == EventType.BOT_THINKING and event.actions:
                for action in event.actions:
                    if action.get("type") == "reply":
                        return action.get("content", "")
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "status": str(self.status),
            "current_thought": self.current_thought,
            "waiting_config": self.waiting_config.to_dict(),
            "rest_config": self.rest_config.to_dict(),
            "mental_log": [e.to_dict() for e in self.mental_log[-50:]],
            "last_activity_at": self.last_activity_at,
            "last_proactive_at": self.last_proactive_at,
            "consecutive_timeout_count": self.consecutive_timeout_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HeartState":
        status_str = data.get("status", "idle")
        try:
            status = HeartStatus(status_str)
        except ValueError:
            status = HeartStatus.IDLE
        mental_log = [MentalEvent.from_dict(e) for e in data.get("mental_log", [])]
        stream_id = data.get("stream_id", "") or data.get("channel_id", "")
        return cls(
            stream_id=stream_id,
            status=status,
            current_thought=data.get("current_thought", ""),
            waiting_config=WaitingConfig.from_dict(data.get("waiting_config", {})),
            rest_config=RestConfig.from_dict(data.get("rest_config", {})),
            mental_log=mental_log,
            last_activity_at=data.get("last_activity_at", time.time()),
            last_proactive_at=data.get("last_proactive_at", 0.0),
            consecutive_timeout_count=data.get("consecutive_timeout_count", 0),
        )

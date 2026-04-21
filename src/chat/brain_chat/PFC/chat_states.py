from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
from src.common.constants import get_local_now


class ChatState(Enum):
    """聊天状态枚举。"""

    NORMAL = auto()
    NEW_MESSAGE = auto()
    COLD_CHAT = auto()
    ACTIVE_CHAT = auto()
    BOT_SPEAKING = auto()
    USER_SPEAKING = auto()
    SILENT = auto()
    ERROR = auto()


class NotificationType(Enum):
    """通知类型枚举。"""

    NEW_MESSAGE = auto()
    COLD_CHAT = auto()
    ACTIVE_CHAT = auto()
    BOT_SPEAKING = auto()
    USER_SPEAKING = auto()
    MESSAGE_DELETED = auto()
    USER_JOINED = auto()
    USER_LEFT = auto()
    ERROR = auto()


@dataclass
class ChatStateInfo:
    """聊天状态信息数据类。"""

    state: ChatState
    last_message_time: Optional[float] = None
    last_message_content: Optional[str] = None
    last_speaker: Optional[str] = None
    message_count: int = 0
    cold_duration: float = 0.0
    active_duration: float = 0.0
    updated_at: Optional[float] = None


@dataclass
class Notification:
    """通知基类数据类。"""

    type: NotificationType
    timestamp: float
    sender: str
    target: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "type": self.type.name,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "target": self.target,
            "data": self.data,
        }


@dataclass
class StateNotification(Notification):
    """持续状态通知数据类。"""

    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict["is_active"] = self.is_active
        return base_dict


class NotificationHandler(ABC):
    """通知处理器抽象基类。"""

    @abstractmethod
    async def handle_notification(self, notification: Notification) -> None:
        """处理通知。"""
        pass


class NotificationManager:
    """通知管理器，负责分发和管理各类通知。"""

    def __init__(self, history_limit: int = 300):
        self._handlers: Dict[
            str, Dict[NotificationType, List[NotificationHandler]]
        ] = {}
        self._active_states: Set[NotificationType] = set()
        self._target_active_states: Dict[str, Set[NotificationType]] = {}
        self._notification_history: List[Notification] = []
        self._history_limit = max(history_limit, 50)

    def register_handler(
        self,
        target: str,
        notification_type: NotificationType,
        handler: NotificationHandler,
    ) -> None:
        """注册通知处理器。"""
        if target not in self._handlers:
            self._handlers[target] = {}
        if notification_type not in self._handlers[target]:
            self._handlers[target][notification_type] = []
        self._handlers[target][notification_type].append(handler)

    def unregister_handler(
        self,
        target: str,
        notification_type: NotificationType,
        handler: NotificationHandler,
    ) -> None:
        """注销通知处理器。"""
        if (
            target in self._handlers
            and notification_type in self._handlers[target]
        ):
            handlers = self._handlers[target][notification_type]
            if handler in handlers:
                handlers.remove(handler)
                if not handlers:
                    del self._handlers[target][notification_type]
                    if not self._handlers[target]:
                        del self._handlers[target]

    async def send_notification(self, notification: Notification) -> None:
        """发送通知到目标处理器。"""
        self._notification_history.append(notification)
        if len(self._notification_history) > self._history_limit:
            self._notification_history = self._notification_history[
                -self._history_limit:
            ]

        if isinstance(notification, StateNotification):
            if notification.is_active:
                self._active_states.add(notification.type)
            else:
                self._active_states.discard(notification.type)

            target_states = self._target_active_states.setdefault(
                notification.target, set()
            )
            if notification.is_active:
                target_states.add(notification.type)
            else:
                target_states.discard(notification.type)

        target = notification.target
        if target in self._handlers:
            handlers = self._handlers[target].get(notification.type, [])
            for handler in handlers:
                await handler.handle_notification(notification)

    def get_active_states(self) -> Set[NotificationType]:
        """获取当前活跃的状态集合。"""
        return self._active_states.copy()

    def is_state_active(self, state_type: NotificationType) -> bool:
        """检查特定状态是否活跃。"""
        return state_type in self._active_states

    def get_target_active_states(self, target: str) -> Set[NotificationType]:
        """获取指定接收方当前活跃状态。"""
        return self._target_active_states.get(target, set()).copy()

    def get_notification_history(
        self,
        sender: Optional[str] = None,
        target: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Notification]:
        """获取通知历史记录。"""
        history = self._notification_history
        if sender:
            history = [n for n in history if n.sender == sender]
        if target:
            history = [n for n in history if n.target == target]
        if limit is not None:
            history = history[-limit:]
        return history

    def clear_history(self) -> None:
        """清空通知历史。"""
        self._notification_history = []

    def describe_runtime(self) -> Dict[str, Any]:
        """返回通知系统运行态摘要。"""
        return {
            "handler_targets": len(self._handlers),
            "active_states": [state.name for state in self._active_states],
            "history_size": len(self._notification_history),
            "target_active_states": {
                target: [state.name for state in states]
                for target, states in self._target_active_states.items()
            },
        }


def create_new_message_notification(
    sender: str, target: str, message: Dict[str, Any]
) -> Notification:
    """创建新消息通知。"""
    return Notification(
        type=NotificationType.NEW_MESSAGE,
        timestamp=get_local_now().timestamp(),
        sender=sender,
        target=target,
        data={
            "message_id": message.get("message_id"),
            "processed_plain_text": message.get("processed_plain_text"),
            "detailed_plain_text": message.get("detailed_plain_text"),
            "user_info": message.get("user_info"),
            "time": message.get("time"),
        },
    )


def create_cold_chat_notification(
    sender: str, target: str, is_cold: bool
) -> StateNotification:
    """创建冷场状态通知。"""
    return StateNotification(
        type=NotificationType.COLD_CHAT,
        timestamp=get_local_now().timestamp(),
        sender=sender,
        target=target,
        data={"is_cold": is_cold},
        is_active=is_cold,
    )


def create_active_chat_notification(
    sender: str, target: str, is_active: bool
) -> StateNotification:
    """创建活跃状态通知。"""
    return StateNotification(
        type=NotificationType.ACTIVE_CHAT,
        timestamp=get_local_now().timestamp(),
        sender=sender,
        target=target,
        data={"is_active": is_active},
        is_active=is_active,
    )


class ChatStateManager:
    """聊天状态管理器，负责跟踪和更新聊天状态。"""

    def __init__(self, history_limit: int = 200):
        self.current_state = ChatState.NORMAL
        self.state_info = ChatStateInfo(
            state=ChatState.NORMAL, updated_at=get_local_now().timestamp()
        )
        self.state_history: List[ChatStateInfo] = []
        self._history_limit = max(history_limit, 50)

    def _push_history_snapshot(self) -> None:
        self.state_history.append(copy.deepcopy(self.state_info))
        if len(self.state_history) > self._history_limit:
            self.state_history = self.state_history[-self._history_limit:]

    def update_state(self, new_state: ChatState, **kwargs) -> None:
        """更新聊天状态。"""
        now_ts = get_local_now().timestamp()
        previous_state = self.state_info.state

        if previous_state == ChatState.COLD_CHAT:
            self.state_info.cold_duration += max(
                now_ts - (self.state_info.updated_at or now_ts), 0.0
            )
        if previous_state == ChatState.ACTIVE_CHAT:
            self.state_info.active_duration += max(
                now_ts - (self.state_info.updated_at or now_ts), 0.0
            )

        self.current_state = new_state
        self.state_info.state = new_state
        for key, value in kwargs.items():
            if hasattr(self.state_info, key):
                setattr(self.state_info, key, value)
        self.state_info.updated_at = now_ts
        self._push_history_snapshot()

    def record_message(
        self, content: str, speaker: str, timestamp: Optional[float] = None
    ) -> None:
        """记录消息并根据活跃度更新状态。"""
        ts = timestamp or get_local_now().timestamp()
        self.state_info.last_message_time = ts
        self.state_info.last_message_content = content
        self.state_info.last_speaker = speaker
        self.state_info.message_count += 1

        if self.is_active_chat():
            self.update_state(ChatState.ACTIVE_CHAT)
        else:
            self.update_state(ChatState.NEW_MESSAGE)

    def get_current_state_info(self) -> ChatStateInfo:
        """获取当前状态信息。"""
        return self.state_info

    def get_state_history(self) -> List[ChatStateInfo]:
        """获取状态历史记录。"""
        return self.state_history

    def get_state_counts(self) -> Dict[str, int]:
        """统计历史状态分布。"""
        counts: Dict[str, int] = {}
        for state_item in self.state_history:
            key = state_item.state.name
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_runtime_snapshot(self) -> Dict[str, Any]:
        """返回当前状态机快照。"""
        return {
            "current_state": self.current_state.name,
            "last_speaker": self.state_info.last_speaker,
            "last_message_time": self.state_info.last_message_time,
            "message_count": self.state_info.message_count,
            "cold_duration": round(self.state_info.cold_duration, 2),
            "active_duration": round(self.state_info.active_duration, 2),
            "history_size": len(self.state_history),
            "state_counts": self.get_state_counts(),
        }

    def is_cold_chat(self, threshold: float = 60.0) -> bool:
        """判断是否处于冷场状态。"""
        if not self.state_info.last_message_time:
            return True
        current_time = datetime.now().timestamp()
        return (current_time - self.state_info.last_message_time) > threshold

    def is_active_chat(self, threshold: float = 5.0) -> bool:
        """判断是否处于活跃状态。"""
        if not self.state_info.last_message_time:
            return False
        current_time = datetime.now().timestamp()
        return (current_time - self.state_info.last_message_time) <= threshold

    def evaluate_chat_temperature(
        self, active_threshold: float = 6.0, cold_threshold: float = 60.0
    ) -> Tuple[ChatState, float]:
        """基于最近消息时间返回推荐状态和间隔秒数。"""
        if not self.state_info.last_message_time:
            return ChatState.COLD_CHAT, float("inf")

        current_time = datetime.now().timestamp()
        interval = current_time - self.state_info.last_message_time
        if interval <= active_threshold:
            return ChatState.ACTIVE_CHAT, interval
        if interval >= cold_threshold:
            return ChatState.COLD_CHAT, interval
        return ChatState.NORMAL, interval

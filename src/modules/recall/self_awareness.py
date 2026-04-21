import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.common.config.config_engine import ConfigEngine
from src.common.logger import get_logger

logger = get_logger("自我觉察")


@dataclass
class SentMessage:
    """已发送消息记录"""

    msg_id: str = ""
    stream_id: str = ""
    channel_id: str = ""
    content: str = ""
    content_type: str = "text"
    sent_at: float = 0.0
    recalled: bool = False
    recall_reason: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionRecord:
    """行为记录"""

    stream_id: str = ""
    action_type: str = ""
    target: str = ""
    content: str = ""
    timestamp: float = 0.0
    result: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserEvent:
    """用户事件"""

    stream_id: str = ""
    user_id: str = ""
    event_type: str = ""
    content: str = ""
    timestamp: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class SelfAwareness:
    """自我意识模块

    追踪智能体的行为历史，为撤回系统提供支持
    所有参数从配置文件读取，禁止硬编码

    职责边界:
    - 只负责记录和查询
    - 不做决策判断
    - 决策由模型完成
    """

    def __init__(self, config_engine: Optional[ConfigEngine] = None):
        self._config = config_engine or ConfigEngine.get_instance()

        self._sent_messages: Dict[str, SentMessage] = {}
        self._stream_history: Dict[str, List[str]] = {}
        self._my_actions: Dict[str, List[ActionRecord]] = {}
        self._user_events: Dict[str, List[UserEvent]] = {}

        self._max_history = 100
        self._decay_seconds = 120.0
        self._half_life_seconds = 300.0

        self._load_config()
        logger.info("自我意识模块初始化完成")

    def _load_config(self):
        """从配置加载参数"""
        self._max_history = self._config.get(
            "self_awareness", "max_history", 100
        )
        self._decay_seconds = self._config.get(
            "self_awareness", "decay_seconds", 120.0
        )
        self._half_life_seconds = self._config.get(
            "self_awareness", "half_life_seconds", 300.0
        )

    def record_message(
        self,
        msg_id: str,
        stream_id: str,
        channel_id: str,
        content: str,
        content_type: str = "text",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录发送的消息"""
        msg = SentMessage(
            msg_id=msg_id,
            stream_id=stream_id,
            channel_id=channel_id,
            content=content,
            content_type=content_type,
            sent_at=time.time(),
            context=context or {},
        )

        self._sent_messages[msg_id] = msg

        if stream_id not in self._stream_history:
            self._stream_history[stream_id] = []
        self._stream_history[stream_id].append(msg_id)

        if len(self._stream_history[stream_id]) > self._max_history:
            old_msg_id = self._stream_history[stream_id].pop(0)
            self._sent_messages.pop(old_msg_id, None)

        logger.debug(f"记录消息: {msg_id} | {content[:30]}...")

    def mark_recalled(
        self, msg_id: str, reason: str = ""
    ) -> Optional[SentMessage]:
        """标记消息已撤回"""
        if msg_id not in self._sent_messages:
            return None

        msg = self._sent_messages[msg_id]
        msg.recalled = True
        msg.recall_reason = reason

        logger.debug(f"标记撤回: {msg_id} | {reason}")
        return msg

    def get_recent_messages(
        self,
        stream_id: str,
        limit: int = 10,
        include_recalled: bool = False,
    ) -> List[SentMessage]:
        """获取最近发送的消息"""
        if stream_id not in self._stream_history:
            return []

        msg_ids = self._stream_history[stream_id][-limit:]
        messages = []

        for msg_id in reversed(msg_ids):
            if msg_id in self._sent_messages:
                msg = self._sent_messages[msg_id]
                if include_recalled or not msg.recalled:
                    messages.append(msg)

        return messages

    def get_message(self, msg_id: str) -> Optional[SentMessage]:
        """获取指定消息"""
        return self._sent_messages.get(msg_id)

    def get_last_message(self, stream_id: str) -> Optional[SentMessage]:
        """获取最后一条消息"""
        if (
            stream_id not in self._stream_history
            or not self._stream_history[stream_id]
        ):
            return None

        for msg_id in reversed(self._stream_history[stream_id]):
            if msg_id in self._sent_messages:
                return self._sent_messages[msg_id]

        return None

    def record_action(
        self,
        stream_id: str,
        action_type: str,
        target: str,
        content: str,
        result: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录行为"""
        if stream_id not in self._my_actions:
            self._my_actions[stream_id] = []

        record = ActionRecord(
            stream_id=stream_id,
            action_type=action_type,
            target=target,
            content=content,
            timestamp=time.time(),
            result=result,
            extra=extra or {},
        )

        self._my_actions[stream_id].append(record)

        if len(self._my_actions[stream_id]) > self._max_history:
            self._my_actions[stream_id] = self._my_actions[stream_id][
                -self._max_history:
            ]

        logger.debug(f"记录行为: {action_type} | {target}")

    def get_recent_actions(
        self,
        stream_id: str,
        action_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[ActionRecord]:
        """获取最近的行为记录"""
        if stream_id not in self._my_actions:
            return []

        actions = self._my_actions[stream_id]

        if action_type:
            actions = [a for a in actions if a.action_type == action_type]

        return actions[-limit:]

    def record_user_event(
        self,
        stream_id: str,
        user_id: str,
        event_type: str,
        content: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录用户事件"""
        if stream_id not in self._user_events:
            self._user_events[stream_id] = []

        event = UserEvent(
            stream_id=stream_id,
            user_id=user_id,
            event_type=event_type,
            content=content,
            timestamp=time.time(),
            extra=extra or {},
        )

        self._user_events[stream_id].append(event)

        if len(self._user_events[stream_id]) > self._max_history:
            self._user_events[stream_id] = self._user_events[stream_id][
                -self._max_history:
            ]

        logger.debug(f"记录用户事件: {event_type} | {user_id}")

    def get_recent_user_events(
        self,
        stream_id: str,
        event_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[UserEvent]:
        """获取最近的用户事件"""
        if stream_id not in self._user_events:
            return []

        events = self._user_events[stream_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    def get_context_for_recall(self, msg_id: str) -> Dict[str, Any]:
        """获取撤回决策所需的上下文"""
        msg = self.get_message(msg_id)
        if not msg:
            return {}

        stream_id = msg.stream_id

        recent_messages = self.get_recent_messages(stream_id, limit=5)
        recent_actions = self.get_recent_actions(stream_id, limit=5)
        recent_events = self.get_recent_user_events(stream_id, limit=5)

        return {
            "target_message": {
                "msg_id": msg.msg_id,
                "content": msg.content,
                "sent_at": msg.sent_at,
                "recalled": msg.recalled,
            },
            "recent_messages": [
                {"content": m.content, "sent_at": m.sent_at}
                for m in recent_messages
            ],
            "recent_actions": [
                {
                    "type": a.action_type,
                    "target": a.target,
                    "timestamp": a.timestamp,
                }
                for a in recent_actions
            ],
            "recent_user_events": [
                {
                    "type": e.event_type,
                    "user_id": e.user_id,
                    "content": e.content[:50],
                }
                for e in recent_events
            ],
        }

    def get_interaction_stats(self, stream_id: str) -> Dict[str, Any]:
        """获取交互统计"""
        actions = self._my_actions.get(stream_id, [])
        events = self._user_events.get(stream_id, [])
        messages = self._stream_history.get(stream_id, [])

        action_types: Dict[str, int] = {}
        for a in actions:
            action_types[a.action_type] = (
                action_types.get(a.action_type, 0) + 1
            )

        event_types: Dict[str, int] = {}
        for e in events:
            event_types[e.event_type] = event_types.get(e.event_type, 0) + 1

        return {
            "stream_id": stream_id,
            "total_messages": len(messages),
            "total_actions": len(actions),
            "total_events": len(events),
            "action_breakdown": action_types,
            "event_breakdown": event_types,
        }

    def cleanup_expired(self, max_age_seconds: float = 3600.0) -> int:
        """清理过期记录"""
        now = time.time()
        cutoff = now - max_age_seconds
        cleaned = 0

        expired_msg_ids = [
            msg_id
            for msg_id, msg in self._sent_messages.items()
            if msg.sent_at < cutoff
        ]

        for msg_id in expired_msg_ids:
            self._sent_messages.pop(msg_id, None)
            cleaned += 1

        for stream_id in self._stream_history:
            self._stream_history[stream_id] = [
                msg_id
                for msg_id in self._stream_history[stream_id]
                if msg_id in self._sent_messages
            ]

        for stream_id in list(self._my_actions.keys()):
            self._my_actions[stream_id] = [
                a for a in self._my_actions[stream_id] if a.timestamp >= cutoff
            ]

        for stream_id in list(self._user_events.keys()):
            self._user_events[stream_id] = [
                e
                for e in self._user_events[stream_id]
                if e.timestamp >= cutoff
            ]

        if cleaned > 0:
            logger.debug(f"清理过期记录: {cleaned} 条")

        return cleaned

    def clear_stream(self, stream_id: str):
        """清除指定流的所有记录"""
        self._stream_history.pop(stream_id, None)
        self._my_actions.pop(stream_id, None)
        self._user_events.pop(stream_id, None)

        msg_ids_to_remove = [
            msg_id
            for msg_id, msg in self._sent_messages.items()
            if msg.stream_id == stream_id
        ]
        for msg_id in msg_ids_to_remove:
            self._sent_messages.pop(msg_id, None)

    def clear_all(self):
        """清除所有记录"""
        self._sent_messages.clear()
        self._stream_history.clear()
        self._my_actions.clear()
        self._user_events.clear()


_self_awareness: Optional[SelfAwareness] = None


def get_self_awareness(
    config_engine: Optional[ConfigEngine] = None,
) -> SelfAwareness:
    """获取自我意识模块单例"""
    global _self_awareness

    if _self_awareness is None:
        _self_awareness = SelfAwareness(config_engine)

    return _self_awareness

import time
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("recall")


@dataclass
class SentMessage:
    msg_id: str
    stream_id: str
    group_id: str
    content: str
    content_type: str
    sent_at: float
    recalled: bool = False
    recall_reason: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionRecord:
    stream_id: str
    action_type: str
    target: str
    content: str
    timestamp: float
    result: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserEvent:
    stream_id: str
    user_id: str
    event_type: str
    content: str
    timestamp: float
    extra: Dict[str, Any] = field(default_factory=dict)


class SelfAwareness:
    DECAY_SECONDS = 120.0
    HALF_LIFE_SECONDS = 300.0

    def __init__(self, max_history: int = 100):
        self._sent_messages: Dict[str, SentMessage] = {}
        self._stream_history: Dict[str, List[str]] = {}
        self._my_actions: Dict[str, List[ActionRecord]] = {}
        self._user_events: Dict[str, List[UserEvent]] = {}
        self._user_data: Dict[str, Dict[str, Any]] = {}
        self._deep_memories: List[Dict[str, Any]] = []
        self._amnesia_state: Dict[str, Dict[str, Any]] = {}
        self._blocked_memories: Dict[str, List[str]] = {}
        self._max_history = max_history
        self._llm_bridge = None

    def _get_llm(self):
        if self._llm_bridge is None:
            try:
                from src.llm_models.utils_model import LLMRequest
                from src.config.config import model_config
                self._llm_bridge = LLMRequest(model_config.lightweight, request_type="self_awareness")
            except Exception:
                pass
        return self._llm_bridge

    def record_message(self, msg_id: str, stream_id: str, group_id: str, content: str, content_type: str = "text"):
        self.record_sent(msg_id, stream_id, group_id, content, content_type)

    def record_sent(self, msg_id: str, stream_id: str, group_id: str, content: str, content_type: str = "text"):
        msg = SentMessage(
            msg_id=msg_id, stream_id=stream_id, group_id=group_id,
            content=content, content_type=content_type, sent_at=time.time(),
        )
        self._sent_messages[msg_id] = msg
        if stream_id not in self._stream_history:
            self._stream_history[stream_id] = []
        self._stream_history[stream_id].append(msg_id)
        if len(self._stream_history[stream_id]) > self._max_history:
            old_id = self._stream_history[stream_id].pop(0)
            self._sent_messages.pop(old_id, None)
        logger.debug(f"记录发送: {msg_id[:12]} 类型={content_type}")

    def get_message(self, msg_id: str) -> Optional[SentMessage]:
        return self._sent_messages.get(msg_id)

    def get_recent_messages(self, stream_id: str, limit: int = 10) -> List[SentMessage]:
        if stream_id not in self._stream_history:
            return []
        msg_ids = self._stream_history[stream_id][-limit:]
        return [self._sent_messages[mid] for mid in msg_ids if mid in self._sent_messages]

    def mark_recalled(self, msg_id: str, reason: str = ""):
        msg = self._sent_messages.get(msg_id)
        if msg:
            msg.recalled = True
            msg.recall_reason = reason
            logger.debug(f"标记撤回: {msg_id[:12]} 原因={reason}")

    async def analyze_if_wrong(
        self, stream_id: str, user_msg: str,
        user_id: Optional[str] = None,
        context_messages: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        recent = self.get_recent_messages(stream_id, limit=5)
        if not recent:
            return {"is_wrong": False, "reason": "无历史消息"}
        last_msg = recent[-1]
        if last_msg.recalled:
            return {"is_wrong": False, "reason": "已撤回"}
        if time.time() - last_msg.sent_at > self.DECAY_SECONDS:
            return {"is_wrong": False, "reason": "超过时限"}
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            my_content = last_msg.content[:200]
            recent_context = self.get_recent_context(stream_id)
            context_str = ""
            if recent_context["actions"]:
                actions_desc = [f"- {a.action_type}: {a.content[:50]}" for a in recent_context["actions"][-3:]]
                context_str += f"\n我刚才做过的事：\n" + "\n".join(actions_desc)
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "self_awareness",
                "self_awareness_wrong.template",
                my_content=my_content,
                context_str=context_str,
                user_msg=user_msg
            )
            request = LLMRequest(model_config.focus_chat, request_type="self_awareness")
            response_text, _ = await request.generate_response_async(prompt)
            if response_text:
                response_text = response_text.strip().upper()
                if response_text.startswith("YES"):
                    parts = response_text.split("|")
                    reason = parts[1].strip() if len(parts) > 1 else "用户指出错误"
                    if user_id:
                        self.record_user_interaction(user_id, "correction", user_msg)
                    return {
                        "is_wrong": True,
                        "msg_id": last_msg.msg_id,
                        "content": last_msg.content,
                        "reason": reason,
                        "confidence": 0.85,
                    }
        except Exception as e:
            logger.debug(f"LLM分析失败: {e}")
        return {"is_wrong": False, "reason": "分析未发现错误"}

    async def should_trigger_amnesia(self, stream_id: str, trigger_content: str) -> Tuple[bool, str]:
        triggers = ["不想说了", "算了", "别提了", "不想回忆", "忘了吧", "不说了"]
        has_trigger = any(t in trigger_content for t in triggers)
        if not has_trigger:
            return False, ""
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            recent_context = self.get_recent_context(stream_id)
            context_summary = ""
            if recent_context["messages"]:
                msgs = [m.content[:50] for m in recent_context["messages"][-3:]]
                context_summary = "刚才的对话：" + "；".join(msgs)
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "self_awareness",
                "amnesia_check.template",
                context_summary=context_summary,
                trigger_content=trigger_content
            )
            request = LLMRequest(model_config.focus_chat, request_type="amnesia_check")
            response_text, _ = await request.generate_response_async(prompt)
            if response_text and response_text.strip().upper().startswith("YES"):
                parts = response_text.split("|")
                reason = parts[1].strip() if len(parts) > 1 else "不想回忆"
                return True, reason
        except Exception as e:
            logger.debug(f"失忆触发分析失败: {e}")
        return False, ""

    def record_my_action(self, stream_id: str, action_type: str, target: str, content: str, result: str = "", extra: Optional[Dict] = None):
        if stream_id not in self._my_actions:
            self._my_actions[stream_id] = []
        record = ActionRecord(
            stream_id=stream_id, action_type=action_type, target=target,
            content=content, timestamp=time.time(), result=result, extra=extra or {},
        )
        self._my_actions[stream_id].append(record)
        if len(self._my_actions[stream_id]) > self._max_history:
            self._my_actions[stream_id] = self._my_actions[stream_id][-self._max_history:]
        logger.debug(f"记录动作: {action_type} -> {target[:20]}")

    def record_user_event(self, stream_id: str, user_id: str, event_type: str, content: str, extra: Optional[Dict] = None):
        if stream_id not in self._user_events:
            self._user_events[stream_id] = []
        event = UserEvent(
            stream_id=stream_id, user_id=user_id, event_type=event_type,
            content=content, timestamp=time.time(), extra=extra or {},
        )
        self._user_events[stream_id].append(event)
        if len(self._user_events[stream_id]) > self._max_history:
            self._user_events[stream_id] = self._user_events[stream_id][-self._max_history:]

    def get_my_recent_actions(self, stream_id: str, limit: int = 10) -> List[ActionRecord]:
        if stream_id not in self._my_actions:
            return []
        return self._my_actions[stream_id][-limit:]

    def get_user_recent_events(self, stream_id: str, limit: int = 10) -> List[UserEvent]:
        if stream_id not in self._user_events:
            return []
        return self._user_events[stream_id][-limit:]

    def clear_stream(self, stream_id: str):
        self._stream_history.pop(stream_id, None)
        self._my_actions.pop(stream_id, None)
        self._user_events.pop(stream_id, None)
        to_remove = [mid for mid, msg in self._sent_messages.items() if msg.stream_id == stream_id]
        for mid in to_remove:
            self._sent_messages.pop(mid, None)

    def get_or_create_user_data(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self._user_data:
            self._user_data[user_id] = {
                "first_seen": time.time(), "last_seen": time.time(),
                "interaction_count": 0, "events": [],
            }
        return self._user_data[user_id]

    def record_user_interaction(self, user_id: str, event_type: str, content: str, extra: Optional[Dict] = None):
        data = self.get_or_create_user_data(user_id)
        data["last_seen"] = time.time()
        data["interaction_count"] += 1
        event = {"type": event_type, "content": content[:200], "time": time.time(), "extra": extra or {}}
        data["events"].append(event)
        if len(data["events"]) > self._max_history:
            data["events"] = data["events"][-self._max_history:]

    def add_deep_memory(self, content: str, user_id: Optional[str] = None):
        memory = {"content": content, "user_id": user_id, "time": time.time()}
        self._deep_memories.append(memory)
        if len(self._deep_memories) > 50:
            self._deep_memories = self._deep_memories[-50:]
        logger.debug(f"内心记忆: {content[:30]}")

    def get_recent_context(self, stream_id: str, seconds: Optional[float] = None) -> Dict[str, Any]:
        if seconds is None:
            seconds = self.DECAY_SECONDS
        cutoff = time.time() - seconds
        recent_msgs = [m for m in self._sent_messages.values() if m.stream_id == stream_id and m.sent_at >= cutoff]
        recent_actions = [a for a in self._my_actions.get(stream_id, []) if a.timestamp >= cutoff]
        recent_events = [e for e in self._user_events.get(stream_id, []) if e.timestamp >= cutoff]
        return {"messages": recent_msgs, "actions": recent_actions, "events": recent_events}

    def cleanup_old_data(self, max_age_seconds: float = 300.0):
        cutoff = time.time() - max_age_seconds
        for stream_id in list(self._my_actions.keys()):
            self._my_actions[stream_id] = [a for a in self._my_actions[stream_id] if a.timestamp >= cutoff]
            if not self._my_actions[stream_id]:
                self._my_actions.pop(stream_id, None)
        for stream_id in list(self._user_events.keys()):
            self._user_events[stream_id] = [e for e in self._user_events[stream_id] if e.timestamp >= cutoff]
            if not self._user_events[stream_id]:
                self._user_events.pop(stream_id, None)
        to_remove = [mid for mid, msg in self._sent_messages.items() if msg.sent_at < cutoff]
        for mid in to_remove:
            self._sent_messages.pop(mid, None)

    def enter_amnesia(self, stream_id: str, duration_seconds: float = 300.0, reason: str = ""):
        now = time.time()
        self._amnesia_state[stream_id] = {
            "start_time": now, "end_time": now + duration_seconds,
            "reason": reason, "blocked_count": 0,
        }
        recent = self.get_recent_messages(stream_id, limit=20)
        self._blocked_memories[stream_id] = [m.msg_id for m in recent]
        self._amnesia_state[stream_id]["blocked_count"] = len(self._blocked_memories[stream_id])
        logger.info(f"进入暂时失忆: {stream_id[:12]} 时长={duration_seconds}s 屏蔽={len(recent)}条 原因={reason}")

    def exit_amnesia(self, stream_id: str):
        self._amnesia_state.pop(stream_id, None)
        self._blocked_memories.pop(stream_id, None)

    def is_in_amnesia(self, stream_id: str) -> bool:
        if stream_id not in self._amnesia_state:
            return False
        state = self._amnesia_state[stream_id]
        if time.time() > state["end_time"]:
            self.exit_amnesia(stream_id)
            return False
        return True

    def get_amnesia_state(self, stream_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_in_amnesia(stream_id):
            return None
        return self._amnesia_state.get(stream_id)

    def is_memory_blocked(self, stream_id: str, msg_id: str) -> bool:
        if not self.is_in_amnesia(stream_id):
            return False
        return msg_id in self._blocked_memories.get(stream_id, [])

    def get_accessible_messages(self, stream_id: str, limit: int = 10) -> List[SentMessage]:
        all_msgs = self.get_recent_messages(stream_id, limit=limit * 2)
        if not self.is_in_amnesia(stream_id):
            return all_msgs[:limit]
        blocked = set(self._blocked_memories.get(stream_id, []))
        accessible = [m for m in all_msgs if m.msg_id not in blocked]
        return accessible[:limit]

    def get_amnesia_remaining(self, stream_id: str) -> float:
        if stream_id not in self._amnesia_state:
            return 0.0
        return max(0.0, self._amnesia_state[stream_id]["end_time"] - time.time())

    def get_stats(self) -> Dict:
        total = len(self._sent_messages)
        recalled = sum(1 for m in self._sent_messages.values() if m.recalled)
        by_type = {}
        for m in self._sent_messages.values():
            by_type[m.content_type] = by_type.get(m.content_type, 0) + 1
        return {
            "total_messages": total, "recalled": recalled, "by_type": by_type,
            "actions_count": sum(len(a) for a in self._my_actions.values()),
            "events_count": sum(len(e) for e in self._user_events.values()),
            "deep_memories": len(self._deep_memories),
            "amnesia_streams": len(self._amnesia_state),
            "blocked_memories": sum(len(b) for b in self._blocked_memories.values()),
        }


_awareness: Optional[SelfAwareness] = None


def get_self_awareness() -> SelfAwareness:
    global _awareness
    if _awareness is None:
        _awareness = SelfAwareness()
    return _awareness

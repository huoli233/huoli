import time
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
from src.common.logger import get_logger

logger = get_logger("self_record")

BEHAVIOR_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "behavior_records"


@dataclass
class SelfAction:
    action_id: str
    action_type: str
    stream_id: str
    content: str
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True

    def to_dict(self) -> Dict:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "stream_id": self.stream_id,
            "content": self.content,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "metadata": self.metadata,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SelfAction":
        return cls(
            action_id=data.get("action_id", ""),
            action_type=data.get("action_type", ""),
            stream_id=data.get("stream_id", ""),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            user_id=data.get("user_id"),
            metadata=data.get("metadata", {}),
            success=data.get("success", True),
        )

    def get_age_seconds(self) -> float:
        return time.time() - self.timestamp

    def get_age_str(self) -> str:
        age = self.get_age_seconds()
        if age < 60:
            return f"{int(age)}秒前"
        elif age < 3600:
            return f"{int(age / 60)}分钟前"
        elif age < 86400:
            return f"{int(age / 3600)}小时前"
        return f"{int(age / 86400)}天前"

    def to_memory_content(self) -> str:
        time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.timestamp))
        prefix = f"[{time_str}] "
        type_labels = {
            "response_sent": "[我说]", "emotion_change": "[情感]",
            "llm_call": "[思考]", "persona_switch": "[心境]",
            "tool_use": "[工具]", "internal_thought": "[内心]",
            "recall": "[撤回]", "search": "[搜索]",
            "kick": "[管理]", "mute": "[禁言]",
            "reply": "[回复]", "send_image": "[发图]",
            "send_sticker": "[表情]",
        }
        label = type_labels.get(self.action_type, "")
        if label:
            return f"{prefix}{label} {self.content}"
        return prefix + self.content


class ActionDescriptionGenerator:
    def __init__(self):
        self._templates: Dict[str, str] = {
            "recall": "撤回了一条消息: {content}",
            "kick": "踢出了成员: {target}",
            "mute": "禁言了成员: {target} ({duration}秒)",
            "unmute": "解除了成员禁言: {target}",
            "ban": "封禁了成员: {target}",
            "unban": "解封了成员: {target}",
            "reply": "回复了消息: {content}",
            "send_image": "发送了图片",
            "send_sticker": "发送了表情包: {sticker_name}",
            "search": "搜索了: {query}",
            "set_card": "设置了群名片: {target} -> {card}",
            "delete_msg": "删除了消息",
            "response_sent": "发送了回复: {content}",
            "tool_use": "使用了工具: {tool_name}",
            "llm_call": "调用了模型: {model}",
            "persona_switch": "切换人格: {persona_name}",
        }

    def generate(self, action_type: str, context: Dict) -> str:
        if action_type not in self._templates:
            return f"执行了{action_type}操作"
        template = self._templates[action_type]
        try:
            return template.format(**context)
        except KeyError:
            return f"执行了{action_type}操作"

    def add_template(self, action_type: str, template: str):
        self._templates[action_type] = template


class StreamActionBuffer:
    def __init__(self, max_size: int = 50):
        self._buffer: deque = deque(maxlen=max_size)
        self._action_counts: Dict[str, int] = {}

    def append(self, record: SelfAction):
        self._buffer.append(record)
        self._action_counts[record.action_type] = self._action_counts.get(record.action_type, 0) + 1

    def get_recent(self, limit: int = 10, max_age_minutes: int = 30) -> List[SelfAction]:
        cutoff = time.time() - (max_age_minutes * 60)
        recent = [r for r in self._buffer if r.timestamp >= cutoff]
        return list(recent)[-limit:]

    def get_by_type(self, action_type: str, limit: int = 5) -> List[SelfAction]:
        matched = [r for r in self._buffer if r.action_type == action_type]
        return matched[-limit:]

    def has_recent(self, action_type: str, max_age_minutes: int = 10) -> Optional[SelfAction]:
        cutoff = time.time() - (max_age_minutes * 60)
        for record in reversed(self._buffer):
            if record.action_type == action_type and record.timestamp >= cutoff:
                return record
        return None

    def get_action_count(self, action_type: str) -> int:
        return self._action_counts.get(action_type, 0)

    def clear_old(self, max_age_hours: int = 24):
        cutoff = time.time() - (max_age_hours * 3600)
        while self._buffer and self._buffer[0].timestamp < cutoff:
            old = self._buffer.popleft()
            self._action_counts[old.action_type] = max(0, self._action_counts.get(old.action_type, 1) - 1)


class SelfBehaviorRecorder:
    ACTION_TYPES = [
        "emotion_change", "tool_use", "llm_call", "response_sent",
        "persona_switch", "memory_operation", "internal_thought",
        "error", "system_event", "user_message",
        "recall", "kick", "mute", "unmute", "reply",
        "search", "send_image", "send_sticker",
    ]

    def __init__(self, max_records_per_stream: int = 50):
        self._streams: Dict[str, StreamActionBuffer] = {}
        self._max_records = max_records_per_stream
        self._desc_generator = ActionDescriptionGenerator()
        self._pending_actions: List[SelfAction] = []
        self._max_pending = 50
        self._last_flush_time = time.time()
        self._flush_interval = 30.0
        self._global_stats: Dict[str, int] = {}
        self._last_cleanup = time.time()
        self._last_save_time = time.time()
        self._auto_save_interval = 300
        BEHAVIOR_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load_all_streams()

    def _ensure_stream_buffer(self, stream_id: str) -> StreamActionBuffer:
        if stream_id not in self._streams:
            self._streams[stream_id] = StreamActionBuffer(self._max_records)
        return self._streams[stream_id]

    def _get_hippocampus(self, stream_id: str):
        try:
            from src.memory_system.hippocampus_buffer import get_hippocampus_buffer
            return get_hippocampus_buffer(stream_id)
        except Exception:
            return None

    def _get_memory_manager(self):
        try:
            from src.memory_system.memory_core import get_memory_manager
            return get_memory_manager()
        except Exception:
            return None

    def _flush_to_hippocampus(self, stream_id: str):
        if not self._pending_actions:
            return
        hippocampus = self._get_hippocampus(stream_id)
        memory_manager = self._get_memory_manager()
        if not hippocampus and not memory_manager:
            self._pending_actions = []
            return
        for action in self._pending_actions:
            memory_content = action.to_memory_content()
            importance = 0.6 if action.action_type in ["response_sent", "emotion_change", "persona_switch"] else 0.4
            if hippocampus:
                hippocampus.add_memory(memory_content, action.user_id or "bot", importance)
            if memory_manager:
                memory_manager.create_memory(
                    stream_id=action.stream_id, content=memory_content,
                    memory_type="self_action", user_id=action.user_id or "bot",
                    importance=importance,
                )
        count = len(self._pending_actions)
        self._pending_actions = []
        self._last_flush_time = time.time()
        logger.debug(f"已刷新 {count} 条行为记录到海马体")

    def record(self, stream_id: str, action_type: str, content: str,
               user_id: Optional[str] = None, metadata: Optional[Dict] = None,
               success: bool = True) -> str:
        action_id = f"{int(time.time() * 1000)}_{action_type[:4]}"
        ctx = metadata or {}
        if user_id:
            ctx["user_id"] = user_id
        if not content:
            content = self._desc_generator.generate(action_type, ctx)
        action = SelfAction(
            action_id=action_id, action_type=action_type,
            stream_id=stream_id, content=content,
            user_id=user_id, metadata=ctx, success=success,
        )
        buffer = self._ensure_stream_buffer(stream_id)
        buffer.append(action)
        self._pending_actions.append(action)
        self._global_stats[action_type] = self._global_stats.get(action_type, 0) + 1
        now = time.time()
        if len(self._pending_actions) >= self._max_pending or now - self._last_flush_time > self._flush_interval:
            self._flush_to_hippocampus(stream_id)
        if now - self._last_save_time > self._auto_save_interval:
            self._auto_save_all()
            self._last_save_time = now
        if now - self._last_cleanup > 3600:
            self._cleanup_old_records()
        logger.debug(f"记录行为: {action_type} - {content[:50]}")
        return action_id

    def record_user_message(self, stream_id: str, user_id: str, user_name: str, content: str, is_activated: bool = True):
        if not is_activated:
            return ""
        return self.record(stream_id, "user_message", content, user_id, {
            "user_name": user_name, "message_length": len(content), "activated": is_activated,
        })

    def record_emotion_change(self, stream_id: str, user_id: str, changes: Dict[str, float], reason: str = ""):
        return self.record(stream_id, "emotion_change", reason or "emotion_update", user_id, {
            "changes": changes, "reason": reason,
            "total_delta": sum(changes.values()),
            "change_count": len([v for v in changes.values() if v != 0]),
        })

    def record_tool_use(self, stream_id: str, tool_name: str, params: Dict, result: str = ""):
        return self.record(stream_id, "tool_use", tool_name, None, {
            "tool_name": tool_name, "params": params, "result": result,
            "param_count": len(params) if params else 0,
            "has_result": bool(result), "result_length": len(result) if result else 0,
        })

    def record_llm_call(self, stream_id: str, model: str, prompt_len: int, response_len: int, duration: float):
        return self.record(stream_id, "llm_call", model, None, {
            "model": model, "prompt_len": prompt_len,
            "response_len": response_len, "duration": duration,
            "tokens_per_second": response_len / duration if duration > 0 else 0,
            "efficiency_ratio": response_len / prompt_len if prompt_len > 0 else 0,
        })

    def record_response(self, stream_id: str, response: str, target_user: str = ""):
        return self.record(stream_id, "response_sent", response, None, {
            "target_user": target_user, "response_length": len(response),
            "has_target": bool(target_user),
            "word_count": len(response.split()) if response else 0,
        })

    def record_persona_switch(self, stream_id: str, persona_name: str, reason: str, duration: float):
        return self.record(stream_id, "persona_switch", persona_name, None, {
            "persona_name": persona_name, "reason": reason,
            "duration": duration, "switch_timestamp": time.time(),
            "is_temporary": duration > 0,
        })

    def record_thought(self, stream_id: str, thought: str):
        return self.record(stream_id, "internal_thought", thought, None, {
            "thought_length": len(thought),
            "word_count": len(thought.split()) if thought else 0,
            "is_complex": len(thought) > 100,
        })

    def record_error(self, stream_id: str, error_type: str, error_msg: str):
        return self.record(stream_id, "error", error_type, None, {
            "error_type": error_type, "error_msg": error_msg,
            "error_length": len(error_msg),
            "severity": "high" if any(word in error_msg.lower() for word in ["critical", "fatal"]) else "medium",
        })

    def record_system_event(self, stream_id: str, event_type: str, details: str):
        return self.record(stream_id, "system_event", event_type, None, {
            "event_type": event_type, "details": details,
            "details_length": len(details), "timestamp": time.time(),
        })

    def record_recall(self, stream_id: str, original_content: str = ""):
        return self.record(
            stream_id, "recall",
            content=self._desc_generator.generate("recall", {"content": original_content[:30]}),
            metadata={"content": original_content[:30] if original_content else ""},
        )

    def record_kick(self, stream_id: str, target_id: str, target_name: str = ""):
        display_target = target_name or target_id
        return self.record(
            stream_id, "kick",
            content=self._desc_generator.generate("kick", {"target": display_target}),
            metadata={"target": display_target, "target_id": target_id},
        )

    def record_mute(self, stream_id: str, target_id: str, duration: int, target_name: str = ""):
        display_target = target_name or target_id
        return self.record(
            stream_id, "mute",
            content=self._desc_generator.generate("mute", {"target": display_target, "duration": duration}),
            metadata={"target": display_target, "target_id": target_id, "duration": duration},
        )

    def record_reply(self, stream_id: str, content: str):
        return self.record(
            stream_id, "reply",
            content=self._desc_generator.generate("reply", {"content": content[:50]}),
            metadata={"content": content[:50] if content else ""},
        )

    def record_search(self, stream_id: str, query: str):
        return self.record(
            stream_id, "search",
            content=self._desc_generator.generate("search", {"query": query[:30]}),
            metadata={"query": query[:30] if query else ""},
        )

    def flush(self, stream_id: str):
        self._flush_to_hippocampus(stream_id)

    def get_recent_actions(self, stream_id: str, limit: int = 5,
                           max_age_minutes: int = 30) -> List[SelfAction]:
        if stream_id not in self._streams:
            return []
        return self._streams[stream_id].get_recent(limit, max_age_minutes)

    def has_recent_action(self, stream_id: str, action_type: str,
                          max_age_minutes: int = 10) -> Optional[SelfAction]:
        if stream_id not in self._streams:
            return None
        return self._streams[stream_id].has_recent(action_type, max_age_minutes)

    def get_action_prompt(self, stream_id: str, max_age_minutes: int = 10,
                          limit: int = 5) -> str:
        recent = self.get_recent_actions(stream_id, limit, max_age_minutes)
        if not recent:
            return ""
        lines = ["【我最近的行为】"]
        for action in recent:
            lines.append(f"- {action.get_age_str()}: {action.content[:80]}")
        return "\n".join(lines)

    def get_recent_actions_prompt(self, stream_id: str, limit: int = 15) -> str:
        recent = self.get_recent_actions(stream_id, limit, max_age_minutes=60)
        if recent:
            lines = ["【我最近1小时的行为】"]
            for action in recent:
                lines.append(action.to_memory_content())
            return "\n".join(lines)
        memory_manager = self._get_memory_manager()
        if not memory_manager:
            return ""
        try:
            memories = memory_manager.search_memories(
                stream_id=stream_id, memory_type="self_action", limit=limit,
            )
            if not memories:
                return ""
            lines = ["【我最近1小时的行为】"]
            for m in memories[-limit:]:
                content = getattr(m, 'content', '') if hasattr(m, 'content') else str(m)
                lines.append(content)
            return "\n".join(lines)
        except Exception:
            return ""

    def get_stream_stats(self, stream_id: str) -> Dict:
        if stream_id not in self._streams:
            return {"total": 0, "types": {}}
        buffer = self._streams[stream_id]
        return {
            "total": len(buffer._buffer),
            "types": dict(buffer._action_counts),
        }

    def get_global_stats(self) -> Dict:
        return {
            "streams": len(self._streams),
            "total_actions": sum(self._global_stats.values()),
            "action_types": dict(self._global_stats),
        }

    def get_tool_usage_count(self, stream_id: str, tool_name: str,
                              max_age_minutes: int = 30) -> int:
        if stream_id not in self._streams:
            return 0
        cutoff = time.time() - (max_age_minutes * 60)
        count = 0
        for record in self._streams[stream_id]._buffer:
            if record.timestamp < cutoff:
                continue
            if record.action_type == tool_name or tool_name in record.action_type:
                count += 1
        return count

    def get_tool_fatigue_level(self, stream_id: str, tool_name: str,
                                max_age_minutes: int = 30) -> float:
        usage_count = self.get_tool_usage_count(stream_id, tool_name, max_age_minutes)
        if usage_count <= 2:
            return 0.0
        elif usage_count <= 5:
            return 0.1 + (usage_count - 2) * 0.1
        elif usage_count <= 8:
            return 0.4 + (usage_count - 5) * 0.15
        return min(1.0, 0.85 + (usage_count - 8) * 0.05)

    def get_fatigue_prompt(self, stream_id: str, max_age_minutes: int = 30) -> str:
        if stream_id not in self._streams:
            return ""
        cutoff = time.time() - (max_age_minutes * 60)
        tool_counts: Dict[str, int] = {}
        for record in self._streams[stream_id]._buffer:
            if record.timestamp < cutoff:
                continue
            tool_counts[record.action_type] = tool_counts.get(record.action_type, 0) + 1
        fatigued_tools = []
        for tool, count in tool_counts.items():
            fatigue = self.get_tool_fatigue_level(stream_id, tool, max_age_minutes)
            if fatigue >= 0.6:
                fatigued_tools.append((tool, count, fatigue))
        if not fatigued_tools:
            return ""
        lines = ["[工具使用情况]"]
        for tool, count, fatigue in sorted(fatigued_tools, key=lambda x: -x[2]):
            if fatigue >= 0.8:
                lines.append(f"- {tool}已使用{count}次，你很累了，可能会拒绝或建议休息")
            elif fatigue >= 0.6:
                lines.append(f"- {tool}已使用{count}次，你有点疲劳，不太想继续用")
        return "\n".join(lines)

    def record_with_fatigue(self, stream_id: str, action_type: str,
                            content: str = "", metadata: Optional[Dict] = None,
                            success: bool = True, user_id: Optional[str] = None,
                            is_admin: bool = False) -> Tuple[str, float]:
        action_id = self.record(
            stream_id=stream_id, action_type=action_type,
            content=content, metadata=metadata,
            success=success, user_id=user_id,
        )
        try:
            from src.modules.emotion_state import get_emotion_manager
            emotion_mgr = get_emotion_manager()
            usage_count, _ = emotion_mgr.record_tool_usage(
                stream_id=stream_id, tool_name=action_type,
            )
            fatigue = min(1.0, usage_count * 0.1) if usage_count > 2 else 0.0
            return action_id, fatigue
        except (ImportError, AttributeError, TypeError):
            fatigue = self.get_tool_fatigue_level(stream_id, action_type)
            return action_id, fatigue

    def export_stream(self, stream_id: str) -> List[Dict]:
        if stream_id not in self._streams:
            return []
        return [r.to_dict() for r in self._streams[stream_id]._buffer]

    def import_stream(self, stream_id: str, records: List[Dict]):
        buffer = self._ensure_stream_buffer(stream_id)
        for data in records:
            record = SelfAction.from_dict(data)
            buffer.append(record)

    def save_to_file(self, stream_id: str):
        try:
            data = self.export_stream(stream_id)
            if not data:
                return
            safe_name = stream_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            file_path = BEHAVIOR_DATA_DIR / f"{safe_name}.json"
            file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"保存行为记录到文件失败: {e}")

    def load_from_file(self, stream_id: str):
        try:
            safe_name = stream_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            file_path = BEHAVIOR_DATA_DIR / f"{safe_name}.json"
            if not file_path.exists():
                return
            raw = file_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            self.import_stream(stream_id, data)
            logger.debug(f"已从文件加载 {stream_id} 的 {len(data)} 条记录")
        except Exception as e:
            logger.debug(f"从文件加载行为记录失败: {e}")

    def _load_all_streams(self):
        try:
            if not BEHAVIOR_DATA_DIR.exists():
                return
            loaded_count = 0
            for file_path in BEHAVIOR_DATA_DIR.glob("*.json"):
                try:
                    raw = file_path.read_text(encoding="utf-8")
                    data = json.loads(raw)
                    stream_id = file_path.stem.replace("_", "/", 1)
                    if data:
                        self.import_stream(stream_id, data)
                        loaded_count += 1
                except Exception as e:
                    logger.debug(f"加载单条记录失败: {e}")
            if loaded_count > 0:
                logger.debug(f"已恢复 {loaded_count} 个会话流的行为记录")
        except Exception as e:
            logger.debug(f"行为记录恢复跳过: {e}")

    def save_all_streams(self):
        saved_count = 0
        for stream_id in list(self._streams.keys()):
            try:
                self.save_to_file(stream_id)
                saved_count += 1
            except Exception as e:
                logger.debug(f"保存会话流 {stream_id} 失败: {e}")
        if saved_count > 0:
            logger.info(f"已保存 {saved_count} 个会话流的行为记录")

    def _auto_save_all(self):
        try:
            saved_count = 0
            for stream_id in list(self._streams.keys()):
                try:
                    self.save_to_file(stream_id)
                    saved_count += 1
                except Exception as e:
                    logger.debug(f"自动保存 {stream_id} 失败: {e}")
            if saved_count > 0:
                logger.debug(f"自动保存了 {saved_count} 个会话流的行为记录")
        except Exception as e:
            logger.debug(f"自动保存失败: {e}")

    def _cleanup_old_records(self):
        for buffer in self._streams.values():
            buffer.clear_old(max_age_hours=24)
        empty_streams = [
            sid for sid, buf in self._streams.items()
            if len(buf._buffer) == 0
        ]
        for sid in empty_streams:
            del self._streams[sid]
            safe_name = sid.replace("/", "_").replace("\\", "_").replace(":", "_")
            file_path = BEHAVIOR_DATA_DIR / f"{safe_name}.json"
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass
        self._last_cleanup = time.time()


_self_recorder: Optional[SelfBehaviorRecorder] = None


def get_self_recorder() -> SelfBehaviorRecorder:
    global _self_recorder
    if _self_recorder is None:
        _self_recorder = SelfBehaviorRecorder()
    return _self_recorder


def init_self_recorder(max_records: int = 50) -> SelfBehaviorRecorder:
    global _self_recorder
    _self_recorder = SelfBehaviorRecorder(max_records_per_stream=max_records)
    logger.info("行为记录器初始化完成")
    return _self_recorder

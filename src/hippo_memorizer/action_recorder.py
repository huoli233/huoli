import time
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Deque, Tuple
from dataclasses import dataclass, field
from collections import deque
from src.common.logger import get_logger
from src.hippo_memorizer.config_loader import (
    get_max_buffer_size,
    get_max_age_minutes,
    get_action_templates,
)

logger = get_logger("action_recorder")

_ACTION_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "actions"


@dataclass
class ActionRecord:
    action_type: str
    description: str
    channel_id: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    success: bool = True

    def to_dict(self) -> Dict:
        return {
            "action_type": self.action_type,
            "description": self.description,
            "channel_id": self.channel_id,
            "timestamp": self.timestamp,
            "context": self.context,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ActionRecord":
        return cls(
            action_type=data.get("action_type", ""),
            description=data.get("description", ""),
            channel_id=data.get("channel_id", ""),
            timestamp=data.get("timestamp", time.time()),
            context=data.get("context", {}),
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


class ActionDescriptionGenerator:
    def __init__(self):
        self._templates: Dict[str, str] = get_action_templates()

    def generate(self, action_type: str, context: Dict) -> str:
        if action_type not in self._templates:
            default_template = self._templates.get(
                "default", "执行了{action_type}操作"
            )
            return default_template.format(action_type=action_type)
        template = self._templates[action_type]
        try:
            return template.format(**context)
        except KeyError:
            return f"执行了{action_type}操作"

    def add_template(self, action_type: str, template: str):
        self._templates[action_type] = template


class ChannelActionBuffer:
    def __init__(self, max_size: Optional[int] = None):
        self._max_size = max_size or get_max_buffer_size()
        self._buffer: Deque[ActionRecord] = deque(maxlen=self._max_size)
        self._action_counts: Dict[str, int] = {}

    def append(self, record: ActionRecord):
        self._buffer.append(record)
        self._action_counts[record.action_type] = (
            self._action_counts.get(record.action_type, 0) + 1
        )

    def get_recent(
        self, limit: int = 10, max_age_minutes: Optional[int] = None
    ) -> List[ActionRecord]:
        max_age = (max_age_minutes or get_max_age_minutes()) * 60
        cutoff = time.time() - max_age
        result = [r for r in reversed(self._buffer) if r.timestamp >= cutoff]
        return result[:limit]

    def get_by_type(
        self, action_type: str, limit: int = 10
    ) -> List[ActionRecord]:
        result = [
            r for r in reversed(self._buffer) if r.action_type == action_type
        ]
        return result[:limit]

    def get_action_counts(self) -> Dict[str, int]:
        return self._action_counts.copy()

    def clear(self):
        self._buffer.clear()
        self._action_counts.clear()

    def __len__(self) -> int:
        return len(self._buffer)


class ActionRecorder:
    def __init__(self):
        self._channel_buffers: Dict[str, ChannelActionBuffer] = {}
        self._desc_generator = ActionDescriptionGenerator()
        self._global_buffer: Deque[ActionRecord] = deque(maxlen=200)

    def _get_channel_buffer(self, channel_id: str) -> ChannelActionBuffer:
        if channel_id not in self._channel_buffers:
            self._channel_buffers[channel_id] = ChannelActionBuffer()
        return self._channel_buffers[channel_id]

    def record(
        self,
        action_type: str,
        channel_id: str,
        context: Optional[Dict] = None,
        success: bool = True,
    ) -> ActionRecord:
        description = self._desc_generator.generate(action_type, context or {})
        record = ActionRecord(
            action_type=action_type,
            description=description,
            channel_id=channel_id,
            context=context or {},
            success=success,
        )
        self._get_channel_buffer(channel_id).append(record)
        self._global_buffer.append(record)
        logger.debug(f"记录动作: {action_type} | {channel_id} | {description}")
        return record

    def get_channel_actions(
        self,
        channel_id: str,
        limit: int = 10,
        max_age_minutes: Optional[int] = None,
    ) -> List[ActionRecord]:
        buffer = self._channel_buffers.get(channel_id)
        if not buffer:
            return []
        return buffer.get_recent(limit, max_age_minutes)

    def get_global_actions(self, limit: int = 20) -> List[ActionRecord]:
        return list(self._global_buffer)[-limit:]

    def get_action_statistics(
        self, channel_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if channel_id:
            buffer = self._channel_buffers.get(channel_id)
            if not buffer:
                return {"total": 0, "by_type": {}}
            return {
                "total": len(buffer),
                "by_type": buffer.get_action_counts(),
            }
        total = len(self._global_buffer)
        by_type: Dict[str, int] = {}
        for record in self._global_buffer:
            by_type[record.action_type] = (
                by_type.get(record.action_type, 0) + 1
            )
        return {
            "total": total,
            "by_type": by_type,
            "channels": len(self._channel_buffers),
        }

    def clear_channel(self, channel_id: str):
        if channel_id in self._channel_buffers:
            self._channel_buffers[channel_id].clear()

    def clear_all(self):
        for buffer in self._channel_buffers.values():
            buffer.clear()
        self._channel_buffers.clear()
        self._global_buffer.clear()


_recorder_instance: Optional[ActionRecorder] = None


def get_action_recorder() -> ActionRecorder:
    global _recorder_instance
    if _recorder_instance is None:
        _recorder_instance = ActionRecorder()
    return _recorder_instance


def record_action(
    action_type: str,
    channel_id: str,
    context: Optional[Dict] = None,
    success: bool = True,
) -> ActionRecord:
    return get_action_recorder().record(
        action_type, channel_id, context, success
    )

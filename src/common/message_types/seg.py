from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum


class SegmentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    EMOJI = "emoji"
    VOICE = "voice"
    AT = "at"
    REPLY = "reply"
    SEGLIST = "seglist"
    FORWARD = "forward"
    MENTION_BOT = "mention_bot"
    PRIORITY_INFO = "priority_info"
    VIDEO_CARD = "video_card"
    MUSIC_CARD = "music_card"
    MINIAPP_CARD = "miniapp_card"


@dataclass
class Seg:
    type: str
    data: Any = None

    def __post_init__(self):
        if isinstance(self.type, SegmentType):
            self.type = self.type.value

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.data is not None:
            if isinstance(self.data, Seg):
                result["data"] = self.data.to_dict()
            elif isinstance(self.data, list):
                result["data"] = [
                    item.to_dict() if isinstance(item, Seg) else item
                    for item in self.data
                ]
            elif isinstance(self.data, dict):
                result["data"] = self.data
            else:
                result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Seg":
        if not data:
            return cls(type="text", data="")
        seg_type = data.get("type", "text")
        seg_data = data.get("data")
        if seg_type == "seglist" and isinstance(seg_data, list):
            seg_data = [cls.from_dict(item) if isinstance(item, dict) else item for item in seg_data]
        elif seg_type == "forward" and isinstance(seg_data, list):
            seg_data = seg_data
        return cls(type=seg_type, data=seg_data)

    @classmethod
    def text(cls, content: str) -> "Seg":
        return cls(type="text", data=content)

    @classmethod
    def image(cls, base64_data: str) -> "Seg":
        return cls(type="image", data=base64_data)

    @classmethod
    def emoji(cls, base64_data: str) -> "Seg":
        return cls(type="emoji", data=base64_data)

    @classmethod
    def voice(cls, base64_data: str) -> "Seg":
        return cls(type="voice", data=base64_data)

    @classmethod
    def at(cls, user_id: str) -> "Seg":
        return cls(type="at", data=user_id)

    @classmethod
    def reply(cls, message_id: str) -> "Seg":
        return cls(type="reply", data=message_id)

    @classmethod
    def seglist(cls, segments: List["Seg"]) -> "Seg":
        return cls(type="seglist", data=segments)

    @classmethod
    def forward(cls, nodes: List[Dict]) -> "Seg":
        return cls(type="forward", data=nodes)

    @classmethod
    def mention_bot(cls, mention_value: float = 1.0) -> "Seg":
        return cls(type="mention_bot", data=mention_value)

    def __repr__(self) -> str:
        if self.type == "text":
            preview = str(self.data)[:30] if self.data else ""
            return f"Seg(text: {preview}...)"
        elif self.type == "seglist":
            count = len(self.data) if isinstance(self.data, list) else 0
            return f"Seg(seglist: {count} items)"
        return f"Seg({self.type})"

from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from .seg import Seg
from .message_info import BaseMessageInfo


@dataclass
class MessageBase:
    message_info: BaseMessageInfo
    message_segment: Optional[Seg] = None
    raw_message: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_info": self.message_info.to_dict() if self.message_info else None,
            "message_segment": self.message_segment.to_dict() if self.message_segment else None,
            "raw_message": self.raw_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageBase":
        if not data:
            return cls(message_info=BaseMessageInfo())
        message_info_data = data.get("message_info")
        message_segment_data = data.get("message_segment")
        return cls(
            message_info=BaseMessageInfo.from_dict(message_info_data) if message_info_data else BaseMessageInfo(),
            message_segment=Seg.from_dict(message_segment_data) if message_segment_data else None,
            raw_message=data.get("raw_message"),
        )

    @classmethod
    def create(
        cls,
        platform: str,
        message_id: str,
        user_info: "UserInfo",
        message_segment: Seg,
        group_info: Optional["GroupInfo"] = None,
        time: float = 0.0,
    ) -> "MessageBase":
        from .message_info import BaseMessageInfo
        message_info = BaseMessageInfo(
            platform=platform,
            message_id=message_id,
            time=time,
            user_info=user_info,
            group_info=group_info,
        )
        return cls(
            message_info=message_info,
            message_segment=message_segment,
        )

    def get_plain_text(self) -> str:
        if not self.message_segment:
            return ""
        return self._extract_text(self.message_segment)

    def _extract_text(self, segment: Seg) -> str:
        if segment.type == "text":
            return str(segment.data) if segment.data else ""
        elif segment.type == "seglist":
            if isinstance(segment.data, list):
                texts = []
                for seg in segment.data:
                    if isinstance(seg, Seg):
                        texts.append(self._extract_text(seg))
                    elif isinstance(seg, dict):
                        texts.append(self._extract_text(Seg.from_dict(seg)))
                return "".join(texts)
        return ""

    def __repr__(self) -> str:
        text_preview = self.get_plain_text()[:30]
        return f"MessageBase({self.message_info}: {text_preview}...)"

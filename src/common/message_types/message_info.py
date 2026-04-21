from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from .user_info import UserInfo
from .group_info import GroupInfo


@dataclass
class TemplateInfo:
    template_name: str = ""
    template_default: bool = True
    template_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template_name,
            "template_default": self.template_default,
            "template_params": self.template_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemplateInfo":
        if not data:
            return cls()
        return cls(
            template_name=str(data.get("template_name", "")),
            template_default=bool(data.get("template_default", True)),
            template_params=data.get("template_params") or {},
        )


@dataclass
class FormatInfo:
    accept_format: list = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accept_format": self.accept_format,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FormatInfo":
        if not data:
            return cls()
        return cls(
            accept_format=data.get("accept_format") or [],
            extra=data.get("extra") or {},
        )


@dataclass
class BaseMessageInfo:
    platform: str = ""
    message_id: str = ""
    time: float = 0.0
    user_info: Optional[UserInfo] = None
    group_info: Optional[GroupInfo] = None
    template_info: Optional[TemplateInfo] = None
    format_info: Optional[FormatInfo] = None
    additional_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "message_id": self.message_id,
            "time": self.time,
            "user_info": self.user_info.to_dict() if self.user_info else None,
            "group_info": self.group_info.to_dict() if self.group_info else None,
            "template_info": self.template_info.to_dict() if self.template_info else None,
            "format_info": self.format_info.to_dict() if self.format_info else None,
            "additional_config": self.additional_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseMessageInfo":
        if not data:
            return cls()
        user_info_data = data.get("user_info")
        group_info_data = data.get("group_info")
        template_info_data = data.get("template_info")
        format_info_data = data.get("format_info")
        return cls(
            platform=str(data.get("platform", "")),
            message_id=str(data.get("message_id", "")),
            time=float(data.get("time", 0.0)),
            user_info=UserInfo.from_dict(user_info_data) if user_info_data else None,
            group_info=GroupInfo.from_dict(group_info_data) if group_info_data else None,
            template_info=TemplateInfo.from_dict(template_info_data) if template_info_data else None,
            format_info=FormatInfo.from_dict(format_info_data) if format_info_data else None,
            additional_config=data.get("additional_config") or {},
        )

    def __repr__(self) -> str:
        user_str = str(self.user_info) if self.user_info else "Unknown"
        group_str = f"@{self.group_info.group_name}" if self.group_info else ""
        return f"MsgInfo({self.platform}:{self.message_id}:{user_str}{group_str})"

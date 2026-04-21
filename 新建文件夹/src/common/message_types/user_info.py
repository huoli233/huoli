from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class UserInfo:
    user_id: str = ""
    user_nickname: str = ""
    platform: str = ""
    user_cardname: str = ""
    user_gender: str = ""
    user_age: Optional[int] = None
    user_level: Optional[int] = None
    user_title: str = ""
    user_avatar_url: str = ""
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "user_nickname": self.user_nickname,
            "platform": self.platform,
            "user_cardname": self.user_cardname,
            "user_gender": self.user_gender,
            "user_age": self.user_age,
            "user_level": self.user_level,
            "user_title": self.user_title,
            "user_avatar_url": self.user_avatar_url,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserInfo":
        if not data:
            return cls()
        return cls(
            user_id=str(data.get("user_id", "")),
            user_nickname=str(data.get("user_nickname", "")),
            platform=str(data.get("platform", "")),
            user_cardname=str(data.get("user_cardname", "")),
            user_gender=str(data.get("user_gender", "")),
            user_age=data.get("user_age"),
            user_level=data.get("user_level"),
            user_title=str(data.get("user_title", "")),
            user_avatar_url=str(data.get("user_avatar_url", "")),
            extra=data.get("extra") or {},
        )

    @classmethod
    def create(
        cls,
        user_id: str,
        user_nickname: str,
        platform: str = "",
        user_cardname: str = "",
    ) -> "UserInfo":
        return cls(
            user_id=user_id,
            user_nickname=user_nickname,
            platform=platform,
            user_cardname=user_cardname,
        )

    def __repr__(self) -> str:
        display_name = self.user_cardname or self.user_nickname or self.user_id
        return f"UserInfo({self.platform}:{self.user_id}:{display_name})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, UserInfo):
            return False
        return (
            self.user_id == other.user_id
            and self.platform == other.platform
        )

    def __hash__(self) -> int:
        return hash((self.user_id, self.platform))

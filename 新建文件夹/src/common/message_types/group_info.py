from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class GroupInfo:
    group_id: str = ""
    group_name: str = ""
    platform: str = ""
    group_owner_id: str = ""
    group_member_count: Optional[int] = None
    group_max_member_count: Optional[int] = None
    group_avatar_url: str = ""
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "group_name": self.group_name,
            "platform": self.platform,
            "group_owner_id": self.group_owner_id,
            "group_member_count": self.group_member_count,
            "group_max_member_count": self.group_max_member_count,
            "group_avatar_url": self.group_avatar_url,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupInfo":
        if not data:
            return cls()
        return cls(
            group_id=str(data.get("group_id", "")),
            group_name=str(data.get("group_name", "")),
            platform=str(data.get("platform", "")),
            group_owner_id=str(data.get("group_owner_id", "")),
            group_member_count=data.get("group_member_count"),
            group_max_member_count=data.get("group_max_member_count"),
            group_avatar_url=str(data.get("group_avatar_url", "")),
            extra=data.get("extra") or {},
        )

    @classmethod
    def create(
        cls,
        group_id: str,
        group_name: str,
        platform: str = "",
    ) -> "GroupInfo":
        return cls(
            group_id=group_id,
            group_name=group_name,
            platform=platform,
        )

    def __repr__(self) -> str:
        display_name = self.group_name or self.group_id
        return f"GroupInfo({self.platform}:{self.group_id}:{display_name})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, GroupInfo):
            return False
        return (
            self.group_id == other.group_id
            and self.platform == other.platform
        )

    def __hash__(self) -> int:
        return hash((self.group_id, self.platform))

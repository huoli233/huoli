import time
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from src.common.logger import get_logger

logger = get_logger("cognitive_store")


@dataclass
class UserPreferenceProfile:
    user_id: str
    likes: List[str] = field(default_factory=list)
    dislikes: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


@dataclass
class CognitiveFragment:
    content: str
    importance: float
    created_at: float
    source_msg_id: str = ""
    category: str = "general"


class UserCognitiveStore:
    """用户认知存储库"""

    def __init__(self, fragment_cap_per_user: int = 100):
        self._preferences: Dict[str, UserPreferenceProfile] = {}
        self._fragments: Dict[str, List[CognitiveFragment]] = defaultdict(list)
        self._fragment_cap = fragment_cap_per_user

    def get_preferences(self, user_id: str) -> Dict[str, Any]:
        profile = self._preferences.get(user_id)
        if profile is None:
            self._try_load_from_db(user_id)
            profile = self._preferences.get(user_id)
        if profile is None:
            return {"likes": [], "dislikes": []}
        return {
            "likes": list(profile.likes),
            "dislikes": list(profile.dislikes),
        }

    def update_preference(
        self, user_id: str, like: str = "", dislike: str = ""
    ) -> None:
        profile = self._preferences.get(user_id)
        if profile is None:
            profile = UserPreferenceProfile(user_id=user_id)
            self._preferences[user_id] = profile
        if like and like not in profile.likes:
            profile.likes.append(like)
        if dislike and dislike not in profile.dislikes:
            profile.dislikes.append(dislike)
        profile.updated_at = time.time()

    def get_fragments(
        self, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        frags = self._fragments.get(user_id, [])
        sorted_frags = sorted(frags, key=lambda f: f.importance, reverse=True)[
            :limit
        ]
        return [
            {
                "content": f.content,
                "importance": f.importance,
                "created_at": f.created_at,
                "category": f.category,
            }
            for f in sorted_frags
        ]

    def add_fragment(
        self,
        user_id: str,
        content: str,
        importance: float = 0.5,
        category: str = "general",
        msg_id: str = "",
    ) -> None:
        frag = CognitiveFragment(
            content=content[:300],
            importance=max(0.0, min(1.0, importance)),
            created_at=time.time(),
            source_msg_id=msg_id,
            category=category,
        )
        bucket = self._fragments[user_id]
        bucket.append(frag)
        if len(bucket) > self._fragment_cap:
            bucket.sort(key=lambda f: f.importance)
            self._fragments[user_id] = bucket[len(bucket) // 4:]

    def _try_load_from_db(self, user_id: str) -> None:
        """尝试从数据库加载用户偏好"""
        try:
            from src.common.database.database_model import PersonInfo

            record = PersonInfo.get_or_none(PersonInfo.person_id == user_id)
            if record:
                likes = []
                dislikes = []
                if record.memory_points:
                    try:
                        points = json.loads(record.memory_points) if isinstance(record.memory_points, str) else record.memory_points
                        if isinstance(points, dict):
                            likes = points.get("likes", [])
                            dislikes = points.get("dislikes", [])
                    except (json.JSONDecodeError, TypeError):
                        pass
                profile = UserPreferenceProfile(
                    user_id=user_id,
                    likes=likes,
                    dislikes=dislikes,
                )
                self._preferences[user_id] = profile
        except Exception as _e:
            logger.debug(f"异常: {_e}")


_store_ref: Optional[UserCognitiveStore] = None


def get_cognitive_store() -> UserCognitiveStore:
    global _store_ref
    if _store_ref is None:
        _store_ref = UserCognitiveStore()
    return _store_ref

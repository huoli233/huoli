import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from src.common.logger import get_logger

logger = get_logger("active_user_mgr")


@dataclass
class UserActivityState:
    user_id: str
    channel_id: str = ""
    last_activity: float = field(default_factory=time.time)
    activity_count: int = 0
    last_activity_type: str = "unknown"
    priority_score: float = 1.0


class ActiveUserManager:
    _instance: Optional["ActiveUserManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self._active_users: Dict[str, UserActivityState] = {}
        self._lock = asyncio.Lock()
        self._activity_threshold_seconds = 1800
        self._max_tracked_users = 500
        self._cleanup_interval = 300
        self._last_cleanup = time.time()
        logger.info(
            f"ActiveUserManager 初始化: 阈值={
                self._activity_threshold_seconds}s, 最大用户数={
                self._max_tracked_users}"
        )

    def is_active(self, user_id: str) -> bool:
        state = self._active_users.get(user_id)
        if not state:
            return False
        elapsed = time.time() - state.last_activity
        return elapsed < self._activity_threshold_seconds

    def record_activity(
        self,
        user_id: str,
        channel_id: str = "",
        activity_type: str = "message",
    ) -> None:
        now = time.time()
        if user_id in self._active_users:
            state = self._active_users[user_id]
            state.last_activity = now
            state.activity_count += 1
            state.last_activity_type = activity_type
            if channel_id:
                state.channel_id = channel_id
            state.priority_score = self._calculate_priority(state)
        else:
            state = UserActivityState(
                user_id=user_id,
                channel_id=channel_id,
                last_activity=now,
                activity_count=1,
                last_activity_type=activity_type,
                priority_score=1.0,
            )
            self._active_users[user_id] = state
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_inactive_users()
            self._last_cleanup = now

    def get_active_user_ids(self) -> List[str]:
        now = time.time()
        active_users = []
        for user_id, state in self._active_users.items():
            elapsed = now - state.last_activity
            if elapsed < self._activity_threshold_seconds:
                active_users.append((state.priority_score, user_id))
        active_users.sort(reverse=True, key=lambda x: x[0])
        return [uid for _, uid in active_users]

    def get_user_state(self, user_id: str) -> Optional[UserActivityState]:
        return self._active_users.get(user_id)

    def get_active_users_for_channel(self, channel_id: str) -> List[str]:
        now = time.time()
        active_users = []
        for user_id, state in self._active_users.items():
            if state.channel_id != channel_id:
                continue
            elapsed = now - state.last_activity
            if elapsed < self._activity_threshold_seconds:
                active_users.append((state.priority_score, user_id))
        active_users.sort(reverse=True, key=lambda x: x[0])
        return [uid for _, uid in active_users]

    def should_load_full_context(self, user_id: str) -> bool:
        state = self._active_users.get(user_id)
        if not state:
            return False
        if not self.is_active(user_id):
            return False
        return state.priority_score > 0.5

    def get_context_limit_for_user(
        self, user_id: str, default_limit: int = 20
    ) -> int:
        state = self._active_users.get(user_id)
        if not state:
            return 5
        if not self.is_active(user_id):
            return 5
        if state.priority_score > 0.7:
            return default_limit
        elif state.priority_score > 0.4:
            return int(default_limit * 0.7)
        else:
            return int(default_limit * 0.5)

    def _calculate_priority(self, state: UserActivityState) -> float:
        now = time.time()
        elapsed = now - state.last_activity
        time_factor = max(
            0.0, 1.0 - (elapsed / self._activity_threshold_seconds)
        )
        activity_factor = min(1.0, state.activity_count / 50)
        priority = time_factor * 0.7 + activity_factor * 0.3
        return round(priority, 3)

    def _cleanup_inactive_users(self) -> int:
        now = time.time()
        extended_threshold = self._activity_threshold_seconds * 2
        users_to_remove = []
        for user_id, state in self._active_users.items():
            elapsed = now - state.last_activity
            if elapsed > extended_threshold:
                users_to_remove.append(user_id)
        if len(self._active_users) > self._max_tracked_users:
            sorted_users = sorted(
                self._active_users.items(), key=lambda x: x[1].priority_score
            )
            excess_count = len(self._active_users) - self._max_tracked_users
            users_to_remove.extend(
                [uid for uid, _ in sorted_users[:excess_count]]
            )
        users_to_remove = list(set(users_to_remove))
        for user_id in users_to_remove:
            del self._active_users[user_id]
        if users_to_remove:
            logger.debug(f"清理了 {len(users_to_remove)} 个不活跃用户")
        return len(users_to_remove)

    def force_cleanup(self) -> int:
        return self._cleanup_inactive_users()

    def get_statistics(self) -> dict:
        now = time.time()
        total = len(self._active_users)
        active = sum(
            1
            for state in self._active_users.values()
            if (now - state.last_activity) < self._activity_threshold_seconds
        )
        return {
            "total_tracked": total,
            "active_users": active,
            "inactive_users": total - active,
            "threshold_seconds": self._activity_threshold_seconds,
            "max_tracked": self._max_tracked_users,
        }

    def set_threshold(self, seconds: int):
        self._activity_threshold_seconds = max(60, seconds)

    def set_max_tracked(self, count: int):
        self._max_tracked_users = max(10, count)


_active_user_manager: Optional[ActiveUserManager] = None


def get_active_user_manager() -> ActiveUserManager:
    global _active_user_manager
    if _active_user_manager is None:
        _active_user_manager = ActiveUserManager()
    return _active_user_manager


def is_user_active(user_id: str) -> bool:
    return get_active_user_manager().is_active(user_id)


def record_user_activity(
    user_id: str, channel_id: str = "", activity_type: str = "message"
) -> None:
    get_active_user_manager().record_activity(
        user_id, channel_id, activity_type
    )

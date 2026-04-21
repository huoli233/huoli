import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("user_interaction_limiter")


@dataclass
class UserInteractionRecord:
    user_id: str
    message_count: int = 0
    last_interaction: float = field(default_factory=time.time)
    cooldown_until: float = 0.0
    warning_count: int = 0
    is_restricted: bool = False
    violation_count: int = 0


class UserInteractionLimiter:
    def __init__(self):
        from src.config.config import global_config
        limit_cfg = global_config.interaction_limit
        self._user_records: Dict[str, UserInteractionRecord] = {}
        try:
            from src.config.behavior_tuner import get_behavior_tuner
            _bt = get_behavior_tuner()
            self._time_window = _bt.get("interaction_time_window", limit_cfg.time_window)
            self._max_messages = _bt.get("interaction_max_per_window", limit_cfg.max_messages_per_window)
            self._cooldown_duration = _bt.get("interaction_cooldown", limit_cfg.cooldown_duration)
            self._max_warnings = _bt.get("interaction_max_warnings", limit_cfg.max_warnings)
        except Exception:
            self._time_window = limit_cfg.time_window
            self._max_messages = limit_cfg.max_messages_per_window
            self._cooldown_duration = limit_cfg.cooldown_duration
            self._max_warnings = limit_cfg.max_warnings

    def should_process_message(self, user_id: str) -> bool:
        current_time = time.time()
        record = self._get_or_create_record(user_id)
        if current_time < record.cooldown_until:
            return False
        if current_time - record.last_interaction > self._time_window:
            record.message_count = 0
            record.warning_count = 0
        record.message_count += 1
        record.last_interaction = current_time
        if record.message_count > self._max_messages:
            record.warning_count += 1
            if record.warning_count > self._max_warnings:
                record.cooldown_until = current_time + self._cooldown_duration
                record.message_count = 0
                logger.info(f"用户 {user_id[:8]} 触发冷却 {int(self._cooldown_duration/60)}分钟")
                return False
            return True
        return True

    def record_message_processed(self, user_id: str) -> Dict:
        record = self._get_or_create_record(user_id)
        return {
            "user_id": user_id, "total_messages": record.message_count,
            "last_interaction": record.last_interaction,
            "restricted": record.is_restricted,
            "violation_count": record.violation_count
        }

    def _get_or_create_record(self, user_id: str) -> UserInteractionRecord:
        if user_id not in self._user_records:
            self._user_records[user_id] = UserInteractionRecord(user_id=user_id)
        return self._user_records[user_id]

    def get_user_status(self, user_id: str) -> Dict:
        record = self._get_or_create_record(user_id)
        current_time = time.time()
        return {
            "user_id": user_id, "message_count": record.message_count,
            "last_interaction": record.last_interaction,
            "is_cooldown": current_time < record.cooldown_until,
            "cooldown_remaining": max(0, int(record.cooldown_until - current_time)),
            "warning_count": record.warning_count,
        }

    def reset_user(self, user_id: str):
        self._user_records.pop(user_id, None)

    def set_cooldown_duration(self, duration: float):
        self._cooldown_duration = max(60.0, duration)

    def set_max_messages(self, max_count: int):
        self._max_messages = max(1, max_count)

    def set_time_window(self, window: float):
        self._time_window = max(10.0, window)

    def get_stats(self) -> Dict:
        current_time = time.time()
        total_users = len(self._user_records)
        cooldown_users = sum(1 for r in self._user_records.values() if current_time < r.cooldown_until)
        return {
            "total_users": total_users, "cooldown_users": cooldown_users,
            "time_window": self._time_window, "max_messages": self._max_messages,
            "cooldown_duration": self._cooldown_duration,
        }

    def cleanup_old_records(self, max_age: float = 86400.0):
        current_time = time.time()
        to_remove = [uid for uid, rec in self._user_records.items() if current_time - rec.last_interaction > max_age]
        for uid in to_remove:
            del self._user_records[uid]
        if to_remove:
            logger.debug(f"清理了 {len(to_remove)} 个旧用户记录")


_user_limiter: Optional[UserInteractionLimiter] = None


def get_user_interaction_limiter() -> UserInteractionLimiter:
    global _user_limiter
    if _user_limiter is None:
        _user_limiter = UserInteractionLimiter()
    return _user_limiter

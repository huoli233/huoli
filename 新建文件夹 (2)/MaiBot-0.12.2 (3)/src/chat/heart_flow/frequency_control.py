from typing import Dict, Optional
from src.common.logger import get_logger

logger = get_logger("frequency_control")


class FrequencyControl:
    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self._frequency_adjust: float = 1.0
        self._consecutive_no_reply: int = 0
        self._min_adjust = 0.1
        self._max_adjust = 5.0
        self._decay_factor = 0.9

    @property
    def talk_frequency_adjust(self) -> float:
        return self._frequency_adjust

    @talk_frequency_adjust.setter
    def talk_frequency_adjust(self, value: float):
        self._frequency_adjust = max(self._min_adjust, min(self._max_adjust, value))

    def get_talk_frequency_adjust(self) -> float:
        return self._frequency_adjust

    def get_frequency_adjust(self) -> float:
        return self._frequency_adjust

    def set_talk_frequency_adjust(self, value: float) -> None:
        self._frequency_adjust = max(self._min_adjust, min(self._max_adjust, value))

    def set_frequency_adjust(self, value: float):
        self._frequency_adjust = max(self._min_adjust, min(self._max_adjust, value))

    def increment_no_reply(self):
        self._consecutive_no_reply += 1
        self._frequency_adjust = max(self._min_adjust, self._frequency_adjust * self._decay_factor)

    def reset_no_reply(self):
        self._consecutive_no_reply = 0
        self._frequency_adjust = 1.0

    def get_dynamic_threshold(self) -> int:
        if self._consecutive_no_reply >= 5:
            return 3
        elif self._consecutive_no_reply >= 3:
            return 2
        return 1


class FrequencyControlManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._controls: Dict[str, FrequencyControl] = {}
        return cls._instance

    def get_or_create(self, chat_id: str) -> FrequencyControl:
        if chat_id not in self._controls:
            self._controls[chat_id] = FrequencyControl(chat_id)
        return self._controls[chat_id]

    def get_or_create_frequency_control(self, chat_id: str) -> FrequencyControl:
        return self.get_or_create(chat_id)

    def remove(self, chat_id: str) -> bool:
        if chat_id in self._controls:
            del self._controls[chat_id]
            return True
        return False

    def remove_frequency_control(self, chat_id: str) -> bool:
        return self.remove(chat_id)

    def get_all_chat_ids(self) -> list:
        return list(self._controls.keys())


frequency_control_manager = FrequencyControlManager()


def get_frequency_manager() -> FrequencyControlManager:
    return frequency_control_manager


import time
import random
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Any


@dataclass
class IntervalRecord:
    channel_id: str
    last_engage_ts: float = 0.0
    engage_count: int = 0
    skip_count: int = 0
    cooldown_until: float = 0.0
    base_interval: float = 2.0
    dynamic_interval: float = 2.0


class IntervalController:
    def __init__(self):
        self._records: Dict[str, IntervalRecord] = {}
        self._global_min_interval = 1.0
        self._global_max_interval = 30.0
        self._cooldown_duration = 60.0

    def get_or_create_record(self, channel_id: str) -> IntervalRecord:
        if channel_id not in self._records:
            self._records[channel_id] = IntervalRecord(channel_id=channel_id)
        return self._records[channel_id]

    def can_engage(self, channel_id: str) -> bool:
        record = self.get_or_create_record(channel_id)
        current_ts = time.time()
        if current_ts < record.cooldown_until:
            return False
        if current_ts - record.last_engage_ts < record.dynamic_interval:
            return False
        return True

    def record_engage(self, channel_id: str):
        record = self.get_or_create_record(channel_id)
        record.last_engage_ts = time.time()
        record.engage_count += 1
        record.skip_count = 0
        self._adjust_interval(record)

    def record_skip(self, channel_id: str):
        record = self.get_or_create_record(channel_id)
        record.skip_count += 1
        self._adjust_interval(record)

    def _adjust_interval(self, record: IntervalRecord):
        if record.engage_count > 5:
            record.dynamic_interval = min(record.base_interval * 1.5, self._global_max_interval)
        elif record.skip_count > 3:
            record.dynamic_interval = max(record.base_interval * 0.8, self._global_min_interval)
        else:
            record.dynamic_interval = record.base_interval

    def set_cooldown(self, channel_id: str, duration: float = None):
        if duration is None:
            duration = self._cooldown_duration
        record = self.get_or_create_record(channel_id)
        record.cooldown_until = time.time() + duration
        logger.debug(f"设置冷却: {channel_id} 冷却{duration}秒")

    def clear_cooldown(self, channel_id: str):
        record = self.get_or_create_record(channel_id)
        record.cooldown_until = 0.0

    def get_remaining_cooldown(self, channel_id: str) -> float:
        record = self.get_or_create_record(channel_id)
        remaining = record.cooldown_until - time.time()
        return max(0.0, remaining)

    def reset_record(self, channel_id: str):
        if channel_id in self._records:
            del self._records[channel_id]

    def get_stats(self, channel_id: str) -> Dict[str, Any]:
        record = self.get_or_create_record(channel_id)
        return {
            "channel_id": channel_id,
            "last_engage_ts": record.last_engage_ts,
            "engage_count": record.engage_count,
            "skip_count": record.skip_count,
            "cooldown_until": record.cooldown_until,
            "dynamic_interval": record.dynamic_interval,
        }


class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._request_times: Dict[str, List[float]] = {}

    def is_allowed(self, key: str) -> bool:
        current_ts = time.time()
        self._cleanup_old_requests(key, current_ts)
        if key not in self._request_times:
            self._request_times[key] = []
        if len(self._request_times[key]) >= self._max_requests:
            return False
        return True

    def record_request(self, key: str):
        current_ts = time.time()
        if key not in self._request_times:
            self._request_times[key] = []
        self._request_times[key].append(current_ts)

    def _cleanup_old_requests(self, key: str, current_ts: float):
        if key not in self._request_times:
            return
        cutoff = current_ts - self._window_seconds
        self._request_times[key] = [ts for ts in self._request_times[key] if ts > cutoff]

    def get_remaining_requests(self, key: str) -> int:
        current_ts = time.time()
        self._cleanup_old_requests(key, current_ts)
        if key not in self._request_times:
            return self._max_requests
        return max(0, self._max_requests - len(self._request_times[key]))


class BurstController:
    def __init__(self, burst_limit: int = 5, cooldown_seconds: float = 10.0):
        self._burst_limit = burst_limit
        self._cooldown = cooldown_seconds
        self._burst_counts: Dict[str, int] = defaultdict(int)
        self._last_burst_ts: Dict[str, float] = {}

    def can_burst(self, key: str) -> bool:
        current_ts = time.time()
        last_ts = self._last_burst_ts.get(key, 0)
        if current_ts - last_ts >= self._cooldown:
            self._burst_counts[key] = 0
        return self._burst_counts[key] < self._burst_limit

    def record_burst(self, key: str):
        self._burst_counts[key] += 1
        self._last_burst_ts[key] = time.time()


@dataclass
class ChatModeState:
    is_focus_mode: bool = False
    high_interest_threshold: float = 0.7
    medium_interest_threshold: float = 0.4
    base_reply_probability: float = 0.15


class DualChatModeController:
    def __init__(self):
        self._channel_modes: Dict[str, ChatModeState] = {}
        logger.info("双聊天模式控制器已初始化")

    def switch_to_focus_mode(self, channel_id: str):
        if channel_id not in self._channel_modes:
            self._channel_modes[channel_id] = ChatModeState()
        self._channel_modes[channel_id].is_focus_mode = True
        logger.info(f"[模式切换] {channel_id} 普通 -> 专注模式")

    def switch_to_normal_mode(self, channel_id: str):
        if channel_id not in self._channel_modes:
            self._channel_modes[channel_id] = ChatModeState()
        self._channel_modes[channel_id].is_focus_mode = False
        logger.info(f"[模式切换] {channel_id} 专注 -> 普通模式")

    def is_focus_mode(self, channel_id: str) -> bool:
        if channel_id not in self._channel_modes:
            return False
        return self._channel_modes[channel_id].is_focus_mode

    async def assess_interest_focus_mode(self, message: str) -> Dict[str, Any]:
        score = self._heuristic_interest_score(message)
        if score >= 0.7:
            return {"score": score, "should_reply": True, "level": "high"}
        elif score >= 0.4:
            return {"score": score, "should_reply": True, "level": "medium"}
        return {"score": score, "should_reply": False, "level": "low"}

    def _heuristic_interest_score(self, message: str) -> float:
        if len(message) < 3:
            return 0.2
        if any(q in message for q in ["？", "?", "吗", "呢"]):
            return 0.8
        if any(kw in message for kw in ["哈哈", "嘿嘿", "哦", "嗯"]):
            return 0.3
        return 0.5


_interval_controller: Optional[IntervalController] = None
_rate_limiter: Optional[RateLimiter] = None
_burst_controller: Optional[BurstController] = None
_dual_chat_mode_controller: Optional[DualChatModeController] = None


def get_interval_controller() -> IntervalController:
    global _interval_controller
    if _interval_controller is None:
        _interval_controller = IntervalController()
    return _interval_controller


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def get_burst_controller() -> BurstController:
    global _burst_controller
    if _burst_controller is None:
        _burst_controller = BurstController()
    return _burst_controller


def get_dual_chat_mode_controller() -> DualChatModeController:
    global _dual_chat_mode_controller
    if _dual_chat_mode_controller is None:
        _dual_chat_mode_controller = DualChatModeController()
    return _dual_chat_mode_controller

import time
import hashlib
import random
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
from src.common.logger import get_logger

logger = get_logger("fatigue_controller")


@dataclass
class SocialPressureContext:
    # 社交压力上下文
    mental_fatigue: float = 0.0
    trauma_score: float = 0.0
    annoyance: float = 0.0
    recent_interaction_count: int = 0
    time_since_last_reply: float = 0.0

    @property
    def overall_pressure(self) -> float:
        # 综合社交压力 (0-100)
        pressure = (
            self.mental_fatigue * 0.4 +
            self.trauma_score * 10 * 0.3 +
            self.annoyance * 0.2 +
            min(self.recent_interaction_count * 2, 20) * 0.1
        )
        return min(pressure, 100.0)

    @property
    def needs_rest(self) -> bool:
        return self.overall_pressure > 75.0


class FatigueBasedResponseController:
    # 基于疲劳度的响应控制器
    # 不强制禁言，而是降低回复意愿
    # 疲劳度越高，回复概率越低，回复长度越短
    def __init__(self):
        self._last_reply_times: Dict[str, float] = {}
        self._interaction_counts: Dict[str, int] = {}
        self._fatigue_levels: Dict[str, float] = {}
        self._count_reset_interval = 300
        logger.info("疲劳驱动响应控制器初始化完成")

    def calculate_response_willingness(
        self,
        stream_id: str,
        user_id: str,
        mental_fatigue: float,
        trauma_score: float,
        annoyance: float,
    ) -> Tuple[float, str]:
        # 计算回复意愿度 (0-1.0)
        user_key = f"{stream_id}_{user_id}"
        current_time = time.time()
        last_reply_time = self._last_reply_times.get(user_key, 0)
        time_since_last = current_time - last_reply_time if last_reply_time > 0 else 9999
        self._clean_old_counts(current_time)
        recent_count = self._interaction_counts.get(user_key, 0)
        context = SocialPressureContext(
            mental_fatigue=mental_fatigue,
            trauma_score=trauma_score,
            annoyance=annoyance,
            recent_interaction_count=recent_count,
            time_since_last_reply=time_since_last,
        )
        base_willingness = 1.0
        fatigue_penalty = min(mental_fatigue / 100.0, 0.8)
        trauma_penalty = min(trauma_score / 10.0 * 0.6, 0.6)
        annoyance_penalty = min(annoyance / 100.0 * 0.5, 0.5)
        frequency_bonus = min(time_since_last / 60.0 * 0.1, 0.3)
        willingness = base_willingness - fatigue_penalty - trauma_penalty - annoyance_penalty + frequency_bonus
        willingness = max(0.05, min(1.0, willingness))
        reason_parts = []
        if mental_fatigue > 60:
            reason_parts.append(f"疲劳度{mental_fatigue:.0f}")
        if trauma_score > 5:
            reason_parts.append(f"创伤{trauma_score:.1f}")
        if annoyance > 50:
            reason_parts.append(f"烦躁{annoyance:.0f}")
        reason = f"意愿{willingness:.2f}" + (f" ({', '.join(reason_parts)})" if reason_parts else " (正常)")
        return willingness, reason

    def should_respond(
        self,
        stream_id: str,
        user_id: str,
        mental_fatigue: float,
        trauma_score: float,
        annoyance: float,
        force: bool = False,
    ) -> Tuple[bool, str]:
        # 判断是否应该回复
        if force:
            return True, "强制回复"
        willingness, reason = self.calculate_response_willingness(
            stream_id, user_id, mental_fatigue, trauma_score, annoyance
        )
        random_seed = hashlib.md5(f"{stream_id}_{user_id}_{int(time.time()/10)}".encode()).hexdigest()
        random.seed(random_seed)
        roll = random.random()
        random.seed()
        should = roll < willingness
        decision_reason = f"{reason}, 随机{roll:.2f}, {'回复' if should else '沉默'}"
        if should:
            self._record_interaction(stream_id, user_id)
        return should, decision_reason

    def suggest_response_length(
        self,
        mental_fatigue: float,
        trauma_score: float,
    ) -> Tuple[str, str]:
        # 根据疲劳建议回复长度
        pressure = mental_fatigue * 0.6 + trauma_score * 10 * 0.4
        if pressure < 20:
            return "normal", "正常长度（8-20字）"
        elif pressure < 40:
            return "moderate", "适中长度（5-15字）"
        elif pressure < 60:
            return "brief", "简短回复（3-10字）"
        elif pressure < 80:
            return "minimal", "极简回复（2-6字）"
        else:
            return "ultra_minimal", "超简回复（1-3字）"

    def get_fatigue_state(self, stream_id: str) -> Dict[str, Any]:
        current_time = time.time()
        self._clean_old_counts(current_time)
        total_interactions = 0
        last_activity = 0.0
        for key, count in self._interaction_counts.items():
            if key.startswith(f"{stream_id}_"):
                total_interactions += count
        for key, t in self._last_reply_times.items():
            if key.startswith(f"{stream_id}_"):
                last_activity = max(last_activity, t)
        idle_seconds = current_time - last_activity if last_activity > 0 else 9999
        fatigue_level = self._fatigue_levels.get(stream_id, 0.0)
        max_consecutive = max(3, 10 - int(fatigue_level / 15))
        return {
            "fatigue_level": fatigue_level,
            "max_consecutive": max_consecutive,
            "total_interactions": total_interactions,
            "idle_seconds": idle_seconds,
        }

    def record_reply(self, stream_id: str):
        current = self._fatigue_levels.get(stream_id, 0.0)
        self._fatigue_levels[stream_id] = min(100.0, current + 3.0)

    def update_fatigue(self, stream_id: str, level: float):
        self._fatigue_levels[stream_id] = max(0.0, min(100.0, level))

    def _record_interaction(self, stream_id: str, user_id: str):
        user_key = f"{stream_id}_{user_id}"
        self._last_reply_times[user_key] = time.time()
        self._interaction_counts[user_key] = self._interaction_counts.get(user_key, 0) + 1

    def _clean_old_counts(self, current_time: float):
        keys_to_remove = []
        for key, last_time in self._last_reply_times.items():
            if current_time - last_time > self._count_reset_interval:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self._last_reply_times.pop(key, None)
            self._interaction_counts.pop(key, None)


_fatigue_controller_instance: Optional[FatigueBasedResponseController] = None


def get_fatigue_controller() -> FatigueBasedResponseController:
    global _fatigue_controller_instance
    if _fatigue_controller_instance is None:
        _fatigue_controller_instance = FatigueBasedResponseController()
    return _fatigue_controller_instance

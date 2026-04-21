import time
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger("engagement_energy")


@dataclass
class EngagementState:
    channel_id: str
    energy: float = 0.0
    last_speak_time: float = field(default_factory=time.time)
    last_recovery_time: float = field(default_factory=time.time)
    last_save_time: float = field(default_factory=time.time)
    speak_count_today: int = 0
    speak_count_hour: int = 0
    hour_marker: int = field(default_factory=lambda: time.localtime().tm_hour)
    day_marker: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))
    is_resting: bool = False
    rest_until: float = 0.0
    consecutive_speaks: int = 0
    interest_accumulator: float = 0.0
    chatterbox_penalty: float = 0.0
    chatterbox_strikes: int = 0
    skip_count: int = 0
    total_skips: int = 0
    last_skip_time: float = 0.0
    rest_level: str = "normal"
    is_peeping: bool = False
    peeping_count: int = 0


class EngagementEnergyManager:
    ENERGY_MAX = 100.0
    ENERGY_MIN = 0.0
    SPEAK_COST_BASE = 15.0
    CONSECUTIVE_PENALTY = 5.0
    RECOVERY_RATE_PER_MINUTE = 2.0
    REST_THRESHOLD = 20.0
    REST_DURATION_MINUTES = 30.0
    INTEREST_THRESHOLD = 0.7
    HOURLY_SPEAK_LIMIT = 5
    DAILY_SPEAK_LIMIT = 15
    CHATTERBOX_THRESHOLD = 4
    CHATTERBOX_PENALTY_DURATION = 60.0
    CHATTERBOX_COOLDOWN_PENALTY = 1.5
    REST_LEVELS = {
        "normal": {"min_minutes": 1, "max_minutes": 5, "peeping_prob": 0.02},
        "medium": {"min_minutes": 5, "max_minutes": 10, "peeping_prob": 0.05},
        "heavy": {"min_minutes": 15, "max_minutes": 30, "peeping_prob": 0.10},
        "extreme": {"min_minutes": 60, "max_minutes": 60, "peeping_prob": 0.15},
    }

    def __init__(self):
        self._states: Dict[str, EngagementState] = {}
        self._state_db = None
        self._initialize_storage()
        logger.info("[EngagementEnergy] 发言能量管理器已初始化(支持离线同步)")

    def _initialize_storage(self) -> None:
        try:
            from src.chat.heart_flow.persistent_state_db import get_persistent_state_db
            self._state_db = get_persistent_state_db()
            self._load_all_states()
        except Exception as e:
            logger.debug(f"[EngagementEnergy] 持久化存储初始化失败: {e}")

    def _load_all_states(self) -> None:
        if not self._state_db:
            return
        try:
            all_data = self._state_db.load_engagement_states()
            if not all_data:
                return
            now = time.time()
            for channel_id, data in all_data.items():
                state = self._create_state_from_data(channel_id, data)
                self._apply_offline_recovery(state, now)
                self._states[channel_id] = state
            if self._states:
                logger.info(f"[EngagementEnergy] 从持久化存储加载了 {len(self._states)} 个频道的能量状态")
        except Exception as e:
            logger.debug(f"[EngagementEnergy] 加载持久化状态失败: {e}")

    def _create_state_from_data(self, channel_id: str, data: dict) -> EngagementState:
        state = EngagementState(channel_id=channel_id)
        state.energy = data.get("energy", 0.0)
        state.last_speak_time = data.get("last_speak_time", time.time())
        state.last_recovery_time = data.get("last_recovery_time", time.time())
        state.last_save_time = data.get("last_save_time", time.time())
        state.speak_count_today = data.get("speak_count_today", 0)
        state.speak_count_hour = data.get("speak_count_hour", 0)
        state.hour_marker = data.get("hour_marker", time.localtime().tm_hour)
        state.day_marker = data.get("day_marker", time.strftime("%Y-%m-%d"))
        state.is_resting = data.get("is_resting", False)
        state.rest_until = data.get("rest_until", 0.0)
        state.consecutive_speaks = data.get("consecutive_speaks", 0)
        state.interest_accumulator = data.get("interest_accumulator", 0.0)
        state.chatterbox_penalty = data.get("chatterbox_penalty", 0.0)
        state.chatterbox_strikes = data.get("chatterbox_strikes", 0)
        state.skip_count = data.get("skip_count", 0)
        state.total_skips = data.get("total_skips", 0)
        state.last_skip_time = data.get("last_skip_time", 0.0)
        state.rest_level = data.get("rest_level", "normal")
        state.is_peeping = data.get("is_peeping", False)
        state.peeping_count = data.get("peeping_count", 0)
        return state

    def _apply_offline_recovery(self, state: EngagementState, now: float) -> None:
        offline_duration = now - state.last_save_time
        if offline_duration <= 0:
            return
        offline_minutes = offline_duration / 60.0
        offline_hours = offline_duration / 3600.0
        base_recovery = offline_minutes * self.RECOVERY_RATE_PER_MINUTE
        if offline_hours >= 1.0:
            base_recovery += min(offline_hours * 10.0, 50.0)
        if offline_hours >= 6.0:
            state.consecutive_speaks = 0
            state.chatterbox_penalty = 0.0
            state.chatterbox_strikes = max(0, state.chatterbox_strikes - 1)
        if offline_hours >= 12.0:
            state.chatterbox_strikes = 0
        old_energy = state.energy
        state.energy = min(self.ENERGY_MAX, state.energy + base_recovery)
        if state.is_resting and now >= state.rest_until:
            state.is_resting = False
            state.energy = max(state.energy, 60.0)
        state.last_recovery_time = now
        if base_recovery > 5:
            logger.info(f"[EngagementEnergy] {_short(state.channel_id)} 离线{offline_hours:.1f}小时，能量 {old_energy:.1f} -> {state.energy:.1f}")

    def _save_state(self, state: EngagementState) -> None:
        if not self._state_db:
            return
        try:
            state.last_save_time = time.time()
            data = {
                "energy": state.energy,
                "last_speak_time": state.last_speak_time,
                "last_recovery_time": state.last_recovery_time,
                "last_save_time": state.last_save_time,
                "speak_count_today": state.speak_count_today,
                "speak_count_hour": state.speak_count_hour,
                "hour_marker": state.hour_marker,
                "day_marker": state.day_marker,
                "is_resting": state.is_resting,
                "rest_until": state.rest_until,
                "consecutive_speaks": state.consecutive_speaks,
                "interest_accumulator": state.interest_accumulator,
                "chatterbox_penalty": state.chatterbox_penalty,
                "chatterbox_strikes": state.chatterbox_strikes,
                "skip_count": state.skip_count,
                "total_skips": state.total_skips,
                "last_skip_time": state.last_skip_time,
                "rest_level": state.rest_level,
                "is_peeping": state.is_peeping,
                "peeping_count": state.peeping_count,
            }
            self._state_db.save_engagement_state(state.channel_id, data)
        except Exception as e:
            logger.debug(f"[EngagementEnergy] 保存状态失败: {e}")

    def _get_or_create_state(self, channel_id: str) -> EngagementState:
        if channel_id not in self._states:
            if self._state_db:
                try:
                    data = self._state_db.load_engagement_state(channel_id)
                    if data:
                        state = self._create_state_from_data(channel_id, data)
                        self._apply_offline_recovery(state, time.time())
                        self._states[channel_id] = state
                        return state
                except Exception:
                    pass
            state = EngagementState(channel_id=channel_id)
            state.energy = self.ENERGY_MAX * 0.7
            self._states[channel_id] = state
        return self._states[channel_id]

    def _update_time_markers(self, state: EngagementState) -> None:
        current_hour = time.localtime().tm_hour
        current_day = time.strftime("%Y-%m-%d")
        if state.hour_marker != current_hour:
            state.hour_marker = current_hour
            state.speak_count_hour = 0
        if state.day_marker != current_day:
            state.day_marker = current_day
            state.speak_count_today = 0
            state.chatterbox_penalty = max(0, state.chatterbox_penalty - 0.5)

    def _apply_recovery(self, state: EngagementState) -> None:
        now = time.time()
        elapsed_minutes = (now - state.last_recovery_time) / 60.0
        if elapsed_minutes > 0:
            base_recovery = elapsed_minutes * self.RECOVERY_RATE_PER_MINUTE
            silence_bonus = 0.0
            silence_hours = (now - state.last_speak_time) / 3600.0
            if silence_hours > 1.0:
                silence_bonus = min(silence_hours * 5.0, 20.0)
            penalty_reduction = state.chatterbox_penalty * 0.5
            total_recovery = max(0, base_recovery + silence_bonus - penalty_reduction)
            state.energy = min(self.ENERGY_MAX, state.energy + total_recovery)
            state.last_recovery_time = now
            if silence_hours > 0.5:
                state.chatterbox_penalty = max(0, state.chatterbox_penalty - elapsed_minutes * 0.1)
            if state.is_resting and now >= state.rest_until:
                state.is_resting = False
                state.energy = max(state.energy, 50.0)
                logger.info(f"[EngagementEnergy] {_short(state.channel_id)} 休息结束，能量恢复到 {state.energy:.1f}")

    def _check_chatterbox(self, state: EngagementState) -> None:
        if state.speak_count_hour >= self.CHATTERBOX_THRESHOLD:
            state.chatterbox_penalty += 1.0
            state.chatterbox_strikes += 1
            extra_rest = state.chatterbox_strikes * 10
            rest_duration = self.CHATTERBOX_PENALTY_DURATION + extra_rest
            state.is_resting = True
            state.rest_until = time.time() + rest_duration * 60
            logger.warning(f"[EngagementEnergy] {_short(state.channel_id)} 话痨惩罚！本小时发言{state.speak_count_hour}次，强制休息{rest_duration:.0f}分钟 (累计{state.chatterbox_strikes}次)")

    def _calculate_rest_duration(self, skip_count: int) -> float:
        if skip_count <= 3:
            level = "normal"
        elif skip_count <= 6:
            level = "medium"
        elif skip_count <= 9:
            level = "heavy"
        else:
            level = "extreme"
        level_config = self.REST_LEVELS[level]
        if level == "extreme":
            return level_config["max_minutes"]
        return random.uniform(level_config["min_minutes"], level_config["max_minutes"])

    def _determine_rest_level(self, skip_count: int) -> str:
        if skip_count <= 3:
            return "normal"
        elif skip_count <= 6:
            return "medium"
        elif skip_count <= 9:
            return "heavy"
        return "extreme"

    def on_skip(self, channel_id: str) -> dict:
        state = self._get_or_create_state(channel_id)
        now = time.time()
        time_since_last_skip = now - state.last_skip_time
        if time_since_last_skip > 300:
            state.skip_count = 1
        else:
            state.skip_count += 1
        state.total_skips += 1
        state.last_skip_time = now
        rest_level = self._determine_rest_level(state.skip_count)
        rest_duration = self._calculate_rest_duration(state.skip_count)
        state.is_resting = True
        state.rest_until = now + rest_duration * 60
        state.rest_level = rest_level
        self._save_state(state)
        logger.info(f"[EngagementEnergy] {_short(channel_id)} 跳过发言({state.skip_count}次)，进入{rest_level}级休息，时长{rest_duration:.1f}分钟")
        return {
            "skip_count": state.skip_count, "total_skips": state.total_skips,
            "is_resting": True, "rest_level": rest_level,
            "rest_duration_minutes": rest_duration, "rest_until": state.rest_until,
        }

    def should_peep(self, channel_id: str) -> bool:
        state = self._get_or_create_state(channel_id)
        if not state.is_resting:
            return False
        if state.is_peeping:
            return False
        level = state.rest_level if state.rest_level else "normal"
        peeping_prob = self.REST_LEVELS.get(level, {}).get("peeping_prob", 0.02)
        return random.random() < peeping_prob

    def peep(self, channel_id: str) -> dict:
        state = self._get_or_create_state(channel_id)
        state.is_peeping = True
        state.peeping_count += 1
        self._save_state(state)
        logger.debug(f"[EngagementEnergy] {_short(channel_id)} 窥屏检查({state.peeping_count}次)，休息等级:{state.rest_level}")
        return {"is_peeping": True, "peeping_count": state.peeping_count, "rest_level": state.rest_level}

    def end_peep(self, channel_id: str) -> None:
        state = self._get_or_create_state(channel_id)
        state.is_peeping = False
        self._save_state(state)

    async def evaluate_interest_during_peep(self, channel_id: str, message: str) -> Tuple[bool, float]:
        state = self._get_or_create_state(channel_id)
        base_interest = 0.5
        if len(message) > 50:
            base_interest += 0.2
        elif len(message) > 20:
            base_interest += 0.1
        interest_keywords = ["?", "？", "为什么", "怎么", "什么", "吗", "呢"]
        for kw in interest_keywords:
            if kw in message:
                base_interest += 0.15
                break
        positive_words = ["哈哈", "有趣", "好玩的", "分享"]
        for pw in positive_words:
            if pw in message:
                base_interest += 0.1
                break
        interest_score = min(base_interest, 1.0)
        level = state.rest_level if state.rest_level else "normal"
        threshold = 0.7 if level == "extreme" else 0.8
        if interest_score >= threshold:
            state.is_resting = False
            state.is_peeping = False
            state.energy = min(state.energy + 20, self.ENERGY_MAX)
            logger.info(f"[EngagementEnergy] {_short(channel_id)} 窥屏时发现高兴趣消息({interest_score:.2f})，提前结束休息")
            return True, interest_score
        state.is_peeping = False
        self._save_state(state)
        return False, interest_score

    def _calculate_speak_cost(self, state: EngagementState, interest_level: float = 0.5) -> float:
        base_cost = self.SPEAK_COST_BASE
        consecutive_cost = state.consecutive_speaks * self.CONSECUTIVE_PENALTY
        frequency_factor = 1.0
        time_since_last = time.time() - state.last_speak_time
        if time_since_last < 60:
            frequency_factor = 2.0
        elif time_since_last < 300:
            frequency_factor = 1.5
        elif time_since_last < 600:
            frequency_factor = 1.2
        chatterbox_factor = 1.0 + state.chatterbox_penalty * self.CHATTERBOX_COOLDOWN_PENALTY
        interest_discount = max(0.5, 1.0 - interest_level * 0.5)
        total_cost = (base_cost + consecutive_cost) * frequency_factor * chatterbox_factor * interest_discount
        return max(5.0, min(80.0, total_cost))

    def can_speak(self, channel_id: str, interest_level: float = 0.5) -> Tuple[bool, str, float]:
        state = self._get_or_create_state(channel_id)
        self._update_time_markers(state)
        self._apply_recovery(state)
        if state.is_resting:
            remaining = (state.rest_until - time.time()) / 60.0
            self._save_state(state)
            return False, f"休息中，还需{remaining:.1f}分钟", state.energy
        if state.speak_count_hour >= self.HOURLY_SPEAK_LIMIT:
            self._save_state(state)
            return False, f"本小时发言达上限({self.HOURLY_SPEAK_LIMIT})", state.energy
        if state.speak_count_today >= self.DAILY_SPEAK_LIMIT:
            self._save_state(state)
            return False, f"今日发言达上限({self.DAILY_SPEAK_LIMIT})", state.energy
        speak_cost = self._calculate_speak_cost(state, interest_level)
        if state.energy < speak_cost:
            self._save_state(state)
            return False, f"能量不足(需要{speak_cost:.1f}，当前{state.energy:.1f})", state.energy
        if state.energy < self.REST_THRESHOLD and interest_level < self.INTEREST_THRESHOLD:
            self._save_state(state)
            return False, "能量低且话题不够感兴趣", state.energy
        if interest_level < 0.3 and state.energy < 60:
            self._save_state(state)
            return False, "话题不感兴趣且能量较低", state.energy
        if state.chatterbox_penalty > 2.0 and interest_level < 0.8:
            self._save_state(state)
            return False, f"话痨冷却中(惩罚={state.chatterbox_penalty:.1f})", state.energy
        self._save_state(state)
        return True, "可以发言", state.energy

    def on_speak(self, channel_id: str, interest_level: float = 0.5) -> dict:
        state = self._get_or_create_state(channel_id)
        self._update_time_markers(state)
        self._apply_recovery(state)
        speak_cost = self._calculate_speak_cost(state, interest_level)
        state.energy = max(self.ENERGY_MIN, state.energy - speak_cost)
        state.last_speak_time = time.time()
        state.speak_count_hour += 1
        state.speak_count_today += 1
        state.consecutive_speaks += 1
        state.interest_accumulator = state.interest_accumulator * 0.8 + interest_level * 0.2
        self._check_chatterbox(state)
        if state.energy <= self.REST_THRESHOLD and not state.is_resting:
            state.is_resting = True
            state.rest_until = time.time() + self.REST_DURATION_MINUTES * 60
            logger.info(f"[EngagementEnergy] {_short(channel_id)} 能量过低({state.energy:.1f})，进入休息状态{self.REST_DURATION_MINUTES}分钟")
        self._save_state(state)
        logger.debug(f"[EngagementEnergy] {_short(channel_id)} 发言消耗{speak_cost:.1f}能量，剩余{state.energy:.1f}")
        return {
            "energy": state.energy, "cost": speak_cost,
            "is_resting": state.is_resting, "speak_count_hour": state.speak_count_hour,
            "speak_count_today": state.speak_count_today, "consecutive": state.consecutive_speaks,
            "chatterbox_penalty": state.chatterbox_penalty,
        }

    def on_user_message(self, channel_id: str) -> None:
        state = self._get_or_create_state(channel_id)
        if state.consecutive_speaks > 0:
            state.consecutive_speaks = 0
            logger.debug(f"[EngagementEnergy] {_short(channel_id)} 用户消息，重置连续发言计数")

    def get_status(self, channel_id: str) -> dict:
        state = self._get_or_create_state(channel_id)
        self._update_time_markers(state)
        self._apply_recovery(state)
        self._save_state(state)
        return {
            "channel_id": channel_id, "energy": state.energy,
            "is_resting": state.is_resting,
            "rest_remaining_minutes": max(0, (state.rest_until - time.time()) / 60.0) if state.is_resting else 0,
            "speak_count_hour": state.speak_count_hour, "speak_count_today": state.speak_count_today,
            "consecutive_speaks": state.consecutive_speaks, "interest_avg": state.interest_accumulator,
            "last_speak_ago_minutes": (time.time() - state.last_speak_time) / 60.0,
            "chatterbox_penalty": state.chatterbox_penalty, "chatterbox_strikes": state.chatterbox_strikes,
        }

    def force_rest(self, channel_id: str, minutes: float = 30.0) -> None:
        state = self._get_or_create_state(channel_id)
        state.is_resting = True
        state.rest_until = time.time() + minutes * 60
        self._save_state(state)
        logger.info(f"[EngagementEnergy] {_short(channel_id)} 强制休息{minutes}分钟")

    def force_recover(self, channel_id: str, amount: float = 50.0) -> None:
        state = self._get_or_create_state(channel_id)
        state.energy = min(self.ENERGY_MAX, state.energy + amount)
        state.is_resting = False
        self._save_state(state)
        logger.info(f"[EngagementEnergy] {_short(channel_id)} 强制恢复{amount}能量，当前{state.energy:.1f}")


def _short(channel_id: str) -> str:
    return channel_id[:8] + "..." if len(channel_id) > 8 else channel_id


_energy_manager: Optional[EngagementEnergyManager] = None


def channel_id_short(channel_id: str) -> str:
    return channel_id[:8] + "..." if len(channel_id) > 8 else channel_id


def get_engagement_energy_manager() -> EngagementEnergyManager:
    global _energy_manager
    if _energy_manager is None:
        _energy_manager = EngagementEnergyManager()
    return _energy_manager

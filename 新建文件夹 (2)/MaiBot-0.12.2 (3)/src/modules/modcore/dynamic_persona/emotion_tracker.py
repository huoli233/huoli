import time
import json
import math
import random
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from src.common.logger import get_logger

logger = get_logger("emotion_tracker")


@dataclass
class UserEmotionState:
    user_id: str
    stream_id: str
    affection: float = 0.0
    impression: str = "初次见面"
    relationship: str = "陌生人"
    annoyance: float = 0.0
    trust_score: float = 0.0
    trauma_score: float = 0.0
    trauma_history: List[Dict] = field(default_factory=list)
    trauma_last_recall: float = 0.0
    trauma_recall_count: int = 0
    baseline_affection: float = 0.0
    interaction_count: int = 0
    last_interaction: float = 0.0
    is_blocked: bool = False
    block_until: float = 0.0
    block_reason: str = ""
    volatility: float = 1.0
    has_trauma_mark: bool = False
    trauma_mark_level: int = 0
    trauma_mark_time: float = 0.0
    milestones_reached: List[str] = field(default_factory=list)
    highest_affection: float = 0.0
    nickname: str = ""
    last_relationship: str = "陌生人"
    relationship_protected_until: float = 0.0
    created_at: float = field(default_factory=time.time)
    joy: float = 0.0
    anticipation: float = 0.0
    surprise: float = 0.0
    sadness: float = 0.0
    disgust: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    pride: float = 0.0
    guilt: float = 0.0
    shame: float = 0.0
    envy: float = 0.0
    gratitude: float = 0.0
    positive_interactions: int = 0
    negative_interactions: int = 0
    intimacy: float = 0.0
    attitude: str = "中立"
    submission_level: float = 0.0
    shyness_level: float = 50.0
    training_stage: int = 0
    training_progress: float = 0.0
    training_resistance: float = 100.0
    trained_responses: List[str] = field(default_factory=list)
    last_training_time: float = 0.0
    training_streak: int = 0
    authority_level: float = 0.0
    intimacy_level: float = 0.0
    excess_level: float = 0.0
    last_update_time: float = field(default_factory=time.time)
    is_protected: bool = False
    protect_until: float = 0.0
    protect_reason: str = ""
    protect_count: int = 0
    affection_threshold_offset: float = 0.0
    cumulative_negative_count: int = 0
    cumulative_positive_count: int = 0
    negative_behavior_streak: int = 0
    positive_behavior_streak: int = 0
    last_behavior_type: str = ""
    threshold_offset_decay_rate: float = 0.02
    psychological_pressure: float = 0.0
    mental_fatigue: float = 0.0
    trust: float = 0.0
    stamina: float = 100.0
    surface_mask: float = 0.0
    inner_chaos: float = 0.0
    mood: str = "平静"
    trust_accumulation: float = 0.0
    distrust_accumulation: float = 0.0
    trauma_accumulation: float = 0.0
    interaction_coefficients: Dict[str, float] = field(default_factory=lambda: {
        "authority_intimacy_ratio": 0.5,
        "excess_trauma_multiplier": 1.0,
        "fatigue_response_factor": 0.8,
        "pressure_resistance_factor": 0.6,
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmotionTracker:
    AFFECTION_MAX = 100.0
    AFFECTION_MIN = -100.0
    TRUST_MAX = 100.0
    TRUST_MIN = -100.0
    ANNOYANCE_MAX = 100.0
    ANNOYANCE_MIN = 0.0
    TRAUMA_MAX = 10.0
    TRAUMA_MIN = 0.0
    TRUST_THRESHOLD = 30.0
    _global_irritation: float = 0.0
    _last_global_impact: float = 0.0
    _global_shield_active: bool = False
    _global_shield_until: float = 0.0
    _global_shield_reason: str = ""

    @staticmethod
    def _bt_get(key: str, default):
        try:
            from src.config.behavior_tuner import get_behavior_tuner
            return get_behavior_tuner().get(key, default)
        except Exception:
            return default

    @classmethod
    def _block_threshold(cls) -> float:
        return cls._bt_get("block_threshold", 80.0)

    @classmethod
    def _global_rage_threshold(cls) -> float:
        return cls._bt_get("global_rage_threshold", 150.0)

    @classmethod
    def _block_duration_base(cls) -> float:
        return cls._bt_get("block_duration_base", 60.0)

    @classmethod
    def _block_duration_max(cls) -> float:
        return cls._bt_get("block_duration_max", 3600.0)

    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self._user_states: Dict[str, UserEmotionState] = {}
        self._psychology_states: Dict[str, Any] = {}
        self._llm_bridge = None
        self._db = None
        self._interaction_history: Dict[str, List[Dict]] = {}
        self._trauma_timers: Dict[str, float] = {}
        self._stream_emotion_manager = None
        self._active_members: Dict[str, float] = {}
        self._active_window_seconds = 1200.0
        self._max_active_slots = 20
        self._init_database()
        self._load_states()
        self._check_and_fix_time_sync()
        self._init_channel_emotion()

    def _init_channel_emotion(self):
        try:
            from src.modules.stream_emotion_state import get_stream_emotion_manager
            self._stream_emotion_manager = get_stream_emotion_manager()
            logger.debug(f"[{self.stream_id}] 流情绪管理器初始化成功")
        except Exception as e:
            logger.debug(f"流情绪管理器初始化失败: {e}")

    def record_stream_query(self, content: str, user_id: str, is_admin: bool = False) -> Tuple[int, float]:
        if self._stream_emotion_manager:
            return self._stream_emotion_manager.record_query(self.stream_id, content, user_id, is_admin)
        return 0, 0.0

    def record_stream_tool_usage(self, tool_name: str, is_admin: bool = False) -> Tuple[int, bool]:
        if self._stream_emotion_manager:
            return self._stream_emotion_manager.record_tool_usage(self.stream_id, tool_name, is_admin)
        return 0, False

    def get_stream_emotion_state(self):
        if self._stream_emotion_manager:
            return self._stream_emotion_manager.get_emotion_state(self.stream_id)
        return None

    def get_stream_emotion_prompt(self) -> str:
        if self._stream_emotion_manager:
            return self._stream_emotion_manager.get_emotion_prompt(self.stream_id)
        return ""

    def get_stream_repeat_warning(self) -> Optional[str]:
        if self._stream_emotion_manager:
            return self._stream_emotion_manager.get_repeat_warning(self.stream_id)
        return None

    def reset_stream_emotion(self):
        if self._stream_emotion_manager:
            self._stream_emotion_manager.reset_stream_emotion(self.stream_id)

    def _refresh_active_window(self, current_user_id: str = ""):
        now = time.time()
        cutoff = now - self._active_window_seconds
        expired = [uid for uid, ts in self._active_members.items() if ts < cutoff]
        for uid in expired:
            del self._active_members[uid]
        if current_user_id:
            self._active_members[current_user_id] = now
        if len(self._active_members) > self._max_active_slots:
            sorted_members = sorted(self._active_members.items(), key=lambda x: x[1], reverse=True)
            self._active_members = dict(sorted_members[:self._max_active_slots])

    def _get_distribution_factor(self) -> float:
        count = max(1, len(self._active_members))
        if count <= 3:
            return 1.0
        if count <= 10:
            return 0.8 - (count - 3) * 0.04
        return max(0.3, 0.52 - (count - 10) * 0.022)

    def _get_dislike_penalty(self, state: UserEmotionState) -> float:
        if state.annoyance < 10:
            return 0.0
        if state.annoyance >= 80:
            return -0.08
        if state.annoyance >= 50:
            return -0.05
        if state.annoyance >= 30:
            return -0.03
        return -0.01

    @staticmethod
    def mask_id(uid: str) -> str:
        if not uid or uid == "default" or uid == "guest":
            return uid
        uid_str = str(uid)
        if len(uid_str) >= 10:
            return f"{uid_str[:4]}****{uid_str[-4:]}"
        elif len(uid_str) > 4:
            return f"{uid_str[:2]}**{uid_str[-2:]}"
        return uid_str

    @staticmethod
    def clamp_affection(value: float) -> float:
        return max(EmotionTracker.AFFECTION_MIN, min(EmotionTracker.AFFECTION_MAX, value))

    @staticmethod
    def clamp_trust(value: float) -> float:
        return max(EmotionTracker.TRUST_MIN, min(EmotionTracker.TRUST_MAX, value))

    @staticmethod
    def clamp_annoyance(value: float) -> float:
        return max(EmotionTracker.ANNOYANCE_MIN, min(EmotionTracker.ANNOYANCE_MAX, value))

    @staticmethod
    def clamp_trauma(value: float) -> float:
        return max(EmotionTracker.TRAUMA_MIN, min(EmotionTracker.TRAUMA_MAX, value))

    @staticmethod
    def clamp_all_values(state: UserEmotionState) -> None:
        state.affection = EmotionTracker.clamp_affection(state.affection)
        state.trust_score = EmotionTracker.clamp_trust(state.trust_score)
        state.annoyance = EmotionTracker.clamp_annoyance(state.annoyance)
        state.trauma_score = EmotionTracker.clamp_trauma(state.trauma_score)
        state.submission_level = max(0, min(100, state.submission_level))
        state.shyness_level = max(0, min(100, state.shyness_level))
        state.training_resistance = max(0, min(100, state.training_resistance))
        state.authority_level = max(0, min(100, state.authority_level))
        state.intimacy_level = max(0, min(100, state.intimacy_level))
        state.excess_level = max(0, min(100, state.excess_level))
        state.psychological_pressure = max(0, min(100, state.psychological_pressure))
        state.mental_fatigue = max(0, min(100, state.mental_fatigue))

    def _init_database(self):
        try:
            from src.modules.modcore.dynamic_persona.emotion_database import get_emotion_database
            self._db = get_emotion_database(self.stream_id)
        except Exception as e:
            logger.warning(f"数据库初始化失败: {e}")

    def _check_and_fix_time_sync(self):
        now = time.time()
        time_sync_issues = []
        for user_id, state in self._user_states.items():
            if state.last_interaction > 0:
                time_gap = now - state.last_interaction
                if time_gap > 30 * 24 * 3600:
                    new_last = now - (7 * 24 * 3600) + (user_id.__hash__() % (7 * 24 * 3600))
                    state.last_interaction = new_last
                    if hasattr(state, 'created_at') and state.created_at > new_last:
                        state.created_at = new_last - 3600
                    if hasattr(state, 'trauma_last_recall') and state.trauma_last_recall > new_last:
                        state.trauma_last_recall = new_last
                    if hasattr(state, 'last_training_time') and state.last_training_time > new_last:
                        state.last_training_time = new_last
                    time_sync_issues.append(user_id)
            if hasattr(state, 'trauma_mark_time') and state.trauma_mark_time > 0:
                if state.trauma_mark_time > now + 3600:
                    state.trauma_mark_time = now - 3600
            if hasattr(state, 'relationship_protected_until') and state.relationship_protected_until > 0:
                if state.relationship_protected_until < now - 3600:
                    state.relationship_protected_until = 0
        if time_sync_issues:
            logger.info(f"时间同步修复 | 处理了{len(time_sync_issues)}个用户")
            self._save_states()

    def _calculate_dynamic_threshold(self, state: UserEmotionState, behavior_type: str, intensity: float = 1.0) -> float:
        now = time.time()
        offset = state.affection_threshold_offset
        if state.last_interaction > 0:
            hours_since_last = (now - state.last_interaction) / 3600
            if hours_since_last > 0:
                decay_amount = state.threshold_offset_decay_rate * hours_since_last * 10
                if offset > 0:
                    offset = max(0, offset - decay_amount)
                elif offset < 0:
                    offset = min(0, offset + decay_amount)
        if behavior_type == "positive":
            if state.last_behavior_type == "positive":
                state.positive_behavior_streak = min(5, state.positive_behavior_streak + 1)
            else:
                state.positive_behavior_streak = 1
                state.negative_behavior_streak = 0
            state.cumulative_positive_count += 1
            streak_bonus = min(0.5, state.positive_behavior_streak * 0.1)
            offset_change = -1.5 * intensity * (1 + streak_bonus)
            offset = max(-30, offset + offset_change)
        elif behavior_type == "negative":
            if state.last_behavior_type == "negative":
                state.negative_behavior_streak = min(5, state.negative_behavior_streak + 1)
            else:
                state.negative_behavior_streak = 1
                state.positive_behavior_streak = 0
            state.cumulative_negative_count += 1
            streak_bonus = min(0.5, state.negative_behavior_streak * 0.1)
            offset_change = 2.0 * intensity * (1 + streak_bonus)
            offset = min(30, offset + offset_change)
        else:
            state.positive_behavior_streak = 0
            state.negative_behavior_streak = 0
        state.last_behavior_type = behavior_type
        state.affection_threshold_offset = max(-30, min(30, offset))
        return state.affection_threshold_offset

    def _get_effective_threshold(self, base_threshold: float, state: UserEmotionState, direction: str = "up") -> float:
        offset = state.affection_threshold_offset
        if direction == "up":
            return base_threshold + offset
        return base_threshold - offset

    def _accumulate_state_balance(self, state: UserEmotionState, behavior_type: str,
                                   change_value: float, source: str = "interaction") -> Dict[str, Any]:
        result = {
            "affection_change": 0.0, "authority_change": 0.0, "intimacy_change": 0.0,
            "excess_change": 0.0, "trauma_change": 0.0, "pressure_change": 0.0,
            "fatigue_change": 0.0, "balance_info": {},
        }
        authority = getattr(state, 'authority_level', 0)
        intimacy = getattr(state, 'intimacy_level', 0)
        if authority + intimacy < 1:
            balance_coef = 0.8
        else:
            ratio = min(authority, intimacy) / max(authority, intimacy, 0.1)
            balance_coef = 0.7 + 0.5 * ratio
        resistance_factor = 1 - state.training_resistance / 120
        pressure_factor = 1 - (state.psychological_pressure / 100) * 0.3
        fatigue_factor = 1 - (state.mental_fatigue / 100) * 0.2
        effect_coef = change_value * balance_coef * resistance_factor * pressure_factor * fatigue_factor
        if behavior_type == "praise":
            intimacy_gain = effect_coef * 3
            state.intimacy_level = min(100, state.intimacy_level + intimacy_gain)
            state.trust_accumulation = min(100, state.trust_accumulation + effect_coef * 0.5)
            result["intimacy_change"] = intimacy_gain
        elif behavior_type == "reward":
            intimacy_gain = effect_coef * 4
            state.intimacy_level = min(100, state.intimacy_level + intimacy_gain)
            state.trust_accumulation = min(100, state.trust_accumulation + effect_coef)
            result["intimacy_change"] = intimacy_gain
        elif behavior_type == "command":
            authority_gain = effect_coef * 2
            state.authority_level = min(100, state.authority_level + authority_gain)
            result["authority_change"] = authority_gain
        elif behavior_type == "discipline":
            authority_gain = effect_coef * 4
            state.authority_level = min(100, state.authority_level + authority_gain)
            if state.training_stage >= 2:
                excess_gain = 0.1 * effect_coef
            elif state.training_stage >= 1:
                excess_gain = 0.2 * effect_coef
            else:
                excess_gain = 0.4 * effect_coef
            state.excess_level = min(100, state.excess_level + excess_gain)
            result["authority_change"] = authority_gain
            result["excess_change"] = excess_gain
        elif behavior_type == "tease":
            shyness_gain = effect_coef * 5
            state.shyness_level = min(100, state.shyness_level + shyness_gain)
            state.intimacy_level = min(100, state.intimacy_level + effect_coef * 2)
            result["shyness_change"] = shyness_gain
        elif behavior_type in ("insult", "harassment"):
            trauma_gain = effect_coef * 0.5
            pressure_gain = effect_coef * 2
            distrust_gain = effect_coef * 1.5
            state.trauma_accumulation = min(10, state.trauma_accumulation + trauma_gain)
            state.psychological_pressure = min(100, state.psychological_pressure + pressure_gain)
            state.distrust_accumulation = min(100, state.distrust_accumulation + distrust_gain)
            result["trauma_change"] = trauma_gain
            result["pressure_change"] = pressure_gain
        if state.excess_level >= 5:
            trauma_from_excess = (state.excess_level - 4) * 0.5
            state.trauma_accumulation = min(10, state.trauma_accumulation + trauma_from_excess)
            state.excess_level = max(0, state.excess_level - 2)
            result["trauma_change"] = result.get("trauma_change", 0) + trauma_from_excess
            result["excess_change"] = -2
        state.mental_fatigue = min(100, state.mental_fatigue + effect_coef * 0.1)
        coefs = state.interaction_coefficients
        if state.trauma_accumulation > 5:
            pressure_from_trauma = (state.trauma_accumulation - 5) * 0.3
            state.psychological_pressure = min(100, state.psychological_pressure + pressure_from_trauma)
        pressure_resistance_impact = state.psychological_pressure * coefs.get("pressure_resistance_factor", 0.6) / 100
        state.training_resistance = max(0, state.training_resistance - pressure_resistance_impact)
        result["balance_info"] = {
            "balance_coef": round(balance_coef, 3), "resistance_factor": round(resistance_factor, 3),
            "pressure_factor": round(pressure_factor, 3), "fatigue_factor": round(fatigue_factor, 3),
            "authority": round(state.authority_level, 1), "intimacy": round(state.intimacy_level, 1),
            "excess": round(state.excess_level, 1), "trauma_acc": round(state.trauma_accumulation, 2),
            "pressure": round(state.psychological_pressure, 1), "fatigue": round(state.mental_fatigue, 1),
        }
        return result

    def _fuse_psychology_with_context(self, state: UserEmotionState,
                                       context_psychology: Optional[Dict] = None) -> Dict[str, Any]:
        fused = {
            "dominant_mood": "neutral", "response_tendency": "normal",
            "interaction_readiness": 1.0,
            "trust_level": state.trust_accumulation / 100,
            "distrust_level": state.distrust_accumulation / 100,
            "trauma_level": state.trauma_accumulation / 10,
            "pressure_level": state.psychological_pressure / 100,
            "fatigue_level": state.mental_fatigue / 100,
            "is_vulnerable": False, "needs_protection": False,
        }
        vulnerability_score = (
            (state.trauma_accumulation / 10) * 0.3 +
            (state.psychological_pressure / 100) * 0.3 +
            (state.mental_fatigue / 100) * 0.2 +
            (state.distrust_accumulation / 100) * 0.2
        )
        if vulnerability_score > 0.6:
            fused["is_vulnerable"] = True
        if state.trauma_accumulation > 7 or state.psychological_pressure > 80:
            fused["needs_protection"] = True
        if state.trauma_accumulation > 5:
            fused["dominant_mood"] = "wounded"
        elif state.psychological_pressure > 60:
            fused["dominant_mood"] = "stressed"
        elif state.mental_fatigue > 70:
            fused["dominant_mood"] = "tired"
        elif state.distrust_accumulation > 50:
            fused["dominant_mood"] = "guarded"
        elif state.trust_accumulation > 60:
            fused["dominant_mood"] = "open"
        readiness = 1.0
        readiness -= (state.trauma_accumulation / 10) * 0.3
        readiness -= (state.psychological_pressure / 100) * 0.2
        readiness -= (state.mental_fatigue / 100) * 0.2
        readiness -= (state.distrust_accumulation / 100) * 0.15
        fused["interaction_readiness"] = round(max(0.2, min(1.0, readiness)), 2)
        if fused["is_vulnerable"]:
            fused["response_tendency"] = "cautious"
        elif vulnerability_score > 0.4:
            fused["response_tendency"] = "guarded"
        elif readiness > 0.8:
            fused["response_tendency"] = "engaged"
        if context_psychology:
            for key in ["dominant_mood", "response_tendency", "is_vulnerable", "needs_protection"]:
                if key in context_psychology:
                    fused[key] = context_psychology[key]
        return fused

    def _get_layered_response_mode(self, state: UserEmotionState,
                                    fused_psychology: Dict[str, Any]) -> Dict[str, Any]:
        mode = {
            "tone": "neutral", "emoji_usage": "normal", "response_length": "normal",
            "formality": "medium", "playfulness": 0.5, "keywords": [], "avoid_patterns": [],
        }
        trauma_level = fused_psychology["trauma_level"]
        pressure_level = fused_psychology["pressure_level"]
        if trauma_level > 0.6 or pressure_level > 0.8:
            mode["tone"] = "gentle"
            mode["response_length"] = "concise"
            mode["formality"] = "low"
            mode["emoji_usage"] = "rare"
            mode["playfulness"] = 0.2
            mode["keywords"] = ["轻声回应", "温和态度", "避免追问"]
            mode["avoid_patterns"] = ["刺激性话题", "强迫式提问"]
        elif pressure_level > 0.5 or fused_psychology.get("fatigue_level", 0) > 0.6:
            mode["tone"] = "calm"
            mode["emoji_usage"] = "occasional"
            mode["playfulness"] = 0.4
            mode["keywords"] = ["轻松随意", "不强求"]
        elif fused_psychology["trust_level"] > 0.6:
            mode["tone"] = "warm"
            mode["response_length"] = "variable"
            mode["formality"] = "low"
            mode["emoji_usage"] = "moderate"
            mode["playfulness"] = 0.7
            mode["keywords"] = ["自然互动", "真诚表达"]
        elif fused_psychology["distrust_level"] > 0.5:
            mode["response_length"] = "concise"
            mode["emoji_usage"] = "none"
            mode["playfulness"] = 0.3
            mode["keywords"] = ["保持距离", "不过度热情"]
            mode["avoid_patterns"] = ["过度亲昵", "打探隐私"]
        psychology_level = fused_psychology["interaction_readiness"]
        if psychology_level < 0.5:
            mode["response_length"] = "short"
            mode["playfulness"] = max(0.2, mode["playfulness"] - 0.2)
        if state.training_stage >= 4:
            mode["tone"] = "intimate" if mode["tone"] == "neutral" else mode["tone"]
            mode["playfulness"] = min(0.9, mode["playfulness"] + 0.2)
        return mode

    def activate_global_shield(self, duration: float, reason: str):
        now = time.time()
        EmotionTracker._global_shield_active = True
        EmotionTracker._global_shield_until = now + duration
        EmotionTracker._global_shield_reason = reason
        logger.critical(f"全局保护激活 | 原因:{reason} | 持续:{duration:.0f}秒")

    def is_global_shield_active(self) -> Tuple[bool, str]:
        if not EmotionTracker._global_shield_active:
            return False, ""
        now = time.time()
        if now >= EmotionTracker._global_shield_until:
            EmotionTracker._global_shield_active = False
            EmotionTracker._global_shield_until = 0.0
            EmotionTracker._global_shield_reason = ""
            return False, ""
        remaining = int(EmotionTracker._global_shield_until - now)
        return True, f"系统保护中，还剩{remaining}秒。原因: {EmotionTracker._global_shield_reason}"

    def deactivate_global_shield(self) -> bool:
        if EmotionTracker._global_shield_active:
            EmotionTracker._global_shield_active = False
            EmotionTracker._global_shield_until = 0.0
            EmotionTracker._global_shield_reason = ""
            logger.info("全局保护已解除")
            return True
        return False

    def get_global_shield_remaining_time(self) -> float:
        if not EmotionTracker._global_shield_active:
            return 0.0
        now = time.time()
        if now >= EmotionTracker._global_shield_until:
            return 0.0
        return max(0.0, EmotionTracker._global_shield_until - now)

    def activate_user_protection(self, user_id: str, duration: float, reason: str):
        state = self._user_states.get(user_id)
        if not state:
            state = UserEmotionState(user_id=user_id, stream_id=self.stream_id)
            self._user_states[user_id] = state
        now = time.time()
        state.is_protected = True
        state.protect_until = now + duration
        state.protect_reason = reason
        state.protect_count += 1
        mask_uid = EmotionTracker.mask_id(user_id)
        logger.warning(f"用户保护 {mask_uid} | 原因:{reason} | 持续:{duration:.0f}秒")

    def is_user_protected(self, user_id: str) -> Tuple[bool, str]:
        state = self._user_states.get(user_id)
        if not state or not state.is_protected:
            return False, ""
        now = time.time()
        if now >= state.protect_until:
            state.is_protected = False
            state.protect_until = 0.0
            state.protect_reason = ""
            return False, ""
        remaining = int(state.protect_until - now)
        return True, f"保护中，还剩{remaining}秒。原因: {state.protect_reason}"

    def deactivate_user_protection(self, user_id: str) -> bool:
        state = self._user_states.get(user_id)
        if state and state.is_protected:
            state.is_protected = False
            state.protect_until = 0.0
            state.protect_reason = ""
            mask_uid = EmotionTracker.mask_id(user_id)
            logger.info(f"用户保护手动解除 | {mask_uid}")
            return True
        return False

    def get_user_block_remaining_time(self, user_id: str) -> float:
        state = self._user_states.get(user_id)
        if not state or not state.is_blocked:
            return 0.0
        now = time.time()
        if now >= state.block_until:
            state.is_blocked = False
            state.block_until = 0.0
            state.block_reason = ""
            return 0.0
        return max(0.0, state.block_until - now)

    def get_user_protection_remaining_time(self, user_id: str) -> float:
        state = self._user_states.get(user_id)
        if not state or not state.is_protected:
            return 0.0
        now = time.time()
        if now >= state.protect_until:
            return 0.0
        return max(0.0, state.protect_until - now)

    def _recover_global_state(self) -> None:
        now = time.time()
        if EmotionTracker._last_global_impact > 0:
            hours_since = (now - EmotionTracker._last_global_impact) / 3600
            if hours_since > 0.1:
                decay_rate = 0.5 * hours_since
                EmotionTracker._global_irritation = max(0.0, EmotionTracker._global_irritation - decay_rate)
                EmotionTracker._last_global_impact = now

    def get_global_rage_level(self) -> float:
        try:
            self._recover_global_state()
        except Exception:
            pass
        return float(EmotionTracker._global_irritation)

    def get_user_state(self, user_id: str, create_if_missing: bool = True) -> Optional[UserEmotionState]:
        if user_id in self._user_states:
            state = self._user_states[user_id]
            now = time.time()
            hours = (now - state.last_interaction) / 3600 if state.last_interaction > 0 else 0
            if hours > 0:
                weight = self._calculate_interaction_weight(state)
                self._apply_trauma_decay(state, hours, weight)
                base_recovery, _ = self.apply_time_recovery(state, hours)
                if state.affection < 0:
                    adjusted = self._apply_damping(base_recovery, state.affection, "gain", state.trauma_score, user_id)
                else:
                    adjusted = base_recovery
                if adjusted > 0.1:
                    old_affection = state.affection
                    state.affection += adjusted
                    if adjusted >= 1.0:
                        uid_short = user_id[:2] + '*' * (len(user_id) - 2) if len(user_id) > 2 else user_id
                        logger.debug(f"好感恢复 {uid_short} | {old_affection:.1f}->{state.affection:.1f} (+{adjusted:.1f})")
                if hours >= 0.5 or adjusted > 1.0:
                    state.last_interaction = now
                    self._save_states()
            return state
        if self._db:
            loaded_state = self._db.load_user_state(user_id)
            if loaded_state:
                state = UserEmotionState(
                    user_id=loaded_state.get("user_id", user_id),
                    stream_id=self.stream_id,
                    affection=loaded_state.get("affection", 0.0),
                    impression=loaded_state.get("impression", "初次见面"),
                    relationship=loaded_state.get("relationship", "陌生人"),
                    annoyance=loaded_state.get("annoyance", 0.0),
                    trust_score=loaded_state.get("trust_score", 0.0),
                    trauma_score=loaded_state.get("trauma_score", 0.0),
                    trauma_history=loaded_state.get("trauma_history", []),
                    trauma_last_recall=loaded_state.get("trauma_last_recall", 0.0),
                    trauma_recall_count=loaded_state.get("trauma_recall_count", 0),
                    baseline_affection=loaded_state.get("baseline_affection", 0.0),
                    interaction_count=loaded_state.get("interaction_count", 0),
                    last_interaction=loaded_state.get("last_interaction", 0.0),
                    is_blocked=loaded_state.get("is_blocked", False),
                    block_until=loaded_state.get("block_until", 0.0),
                    block_reason=loaded_state.get("block_reason", ""),
                    volatility=loaded_state.get("volatility", 1.0),
                    has_trauma_mark=loaded_state.get("has_trauma_mark", False),
                    trauma_mark_level=loaded_state.get("trauma_mark_level", 0),
                    is_protected=loaded_state.get("is_protected", False),
                    protect_until=loaded_state.get("protect_until", 0.0),
                )
                self._user_states[user_id] = state
                return state
        if create_if_missing:
            state = UserEmotionState(user_id=user_id, stream_id=self.stream_id)
            self._user_states[user_id] = state
            return state
        return None

    def get_all_user_states(self) -> Dict[str, UserEmotionState]:
        return self._user_states.copy()

    def get_user_emotion(self, user_id: str) -> Dict[str, Any]:
        state = self.get_user_state(user_id, create_if_missing=False)
        if not state:
            return {"affection": 0.0, "trust": 0.0, "annoyance": 0.0, "trauma_score": 0.0, "relationship": "陌生人", "mental_fatigue": 0.0, "is_blocked": False}
        return {
            "affection": state.affection, "trust": state.trust_score, "annoyance": state.annoyance,
            "trauma_score": state.trauma_score, "relationship": state.relationship,
            "mental_fatigue": state.mental_fatigue, "is_blocked": state.is_blocked,
            "interaction_count": state.interaction_count, "intimacy": state.intimacy,
            "submission_level": state.submission_level,
        }

    def get_psychology_state(self, user_id: str) -> Optional[Any]:
        if user_id not in self._psychology_states:
            try:
                from src.modules.modcore.dynamic_persona.complex_trauma_psychology import ComplexTraumaPsychology
                self._psychology_states[user_id] = ComplexTraumaPsychology()
            except ImportError:
                return None
        return self._psychology_states[user_id]

    def process_psychological_stimulus(self, user_id: str, content: str,
                                        context: Optional[Dict] = None) -> Dict[str, Any]:
        psych_state = self.get_psychology_state(user_id)
        if not psych_state:
            return {}
        context = context or {}
        state = self.get_user_state(user_id)
        if state:
            context["trauma_score"] = state.trauma_score
            context["affection"] = state.affection
            context["sentiment"] = context.get("sentiment", "neutral")
        result = psych_state.process_stimulus(content, context)
        psych_state.accumulate_state(result, source="user_message")
        if state:
            state.inner_chaos = psych_state.inner_chaos_level
            state.surface_mask = psych_state.surface_mask_strength
            self._save_user_state(user_id)
        return result

    def update_affection(self, user_id: str, delta: float, reason: str = "") -> float:
        state = self.get_user_state(user_id, create_if_missing=True)
        old_val = state.affection
        state.affection = self.clamp_affection(state.affection + delta * state.volatility)
        if state.affection > state.highest_affection:
            state.highest_affection = state.affection
        state.last_interaction = time.time()
        state.interaction_count += 1
        if delta > 0:
            state.positive_interactions += 1
        elif delta < 0:
            state.negative_interactions += 1
        self._update_relationship(state)
        self.clamp_all_values(state)
        self._save_user_state(user_id)
        if abs(delta) > 2:
            mask_uid = self.mask_id(user_id)
            logger.info(f"好感变化 {mask_uid} | {old_val:.1f} -> {state.affection:.1f} ({delta:+.1f}) | {reason}")
        return state.affection

    def update_annoyance(self, user_id: str, delta: float, reason: str = "") -> float:
        state = self.get_user_state(user_id, create_if_missing=True)
        state.annoyance = self.clamp_annoyance(state.annoyance + delta)
        if state.annoyance >= self._block_threshold() and not state.is_blocked:
            block_duration = min(self._block_duration_base() * (state.annoyance / self._block_threshold()), self._block_duration_max())
            state.is_blocked = True
            state.block_until = time.time() + block_duration
            state.block_reason = reason or "烦躁度过高"
            mask_uid = self.mask_id(user_id)
            logger.warning(f"用户屏蔽 {mask_uid} | 烦躁度{state.annoyance:.0f} | 持续{block_duration:.0f}秒")
        state.last_interaction = time.time()
        self.clamp_all_values(state)
        self._save_user_state(user_id)
        return state.annoyance

    def update_trauma(self, user_id: str, delta: float, trigger: str = "") -> float:
        state = self.get_user_state(user_id, create_if_missing=True)
        old_trauma = state.trauma_score
        state.trauma_score = self.clamp_trauma(state.trauma_score + delta)
        
        # 记录创伤历史
        if delta > 0 and trigger:
            state.trauma_history.append({"trigger": trigger, "score": delta, "time": time.time()})
            if len(state.trauma_history) > 50:
                state.trauma_history = state.trauma_history[-30:]
            state.trauma_last_recall = time.time()
            state.trauma_recall_count += 1
        
        # 设置创伤标记
        if state.trauma_score >= 3.0 and not state.has_trauma_mark:
            state.has_trauma_mark = True
            state.trauma_mark_level = 1
            state.trauma_mark_time = time.time()
        
        # 自动激活逃避机制（全局保护盾）
        if delta > 0 and state.trauma_score >= 6.0 and old_trauma < 6.0:
            # 创伤分数达到6.0时激活中度保护
            self.activate_global_shield(1800, f"用户{user_id[:8]}创伤分数达到{state.trauma_score:.1f}")
            logger.warning(f"🛡️ 创伤保护激活: 用户{user_id[:8]} 创伤{state.trauma_score:.1f}")
        elif delta > 0 and state.trauma_score >= 8.0 and old_trauma < 8.0:
            # 创伤分数达到8.0时激活重度保护
            self.activate_global_shield(3600, f"用户{user_id[:8]}严重创伤分数{state.trauma_score:.1f}")
            logger.critical(f"🛡️ 严重创伤保护激活: 用户{user_id[:8]} 创伤{state.trauma_score:.1f}")
            
            # 激活个体逃避模式
            state.is_blocked = True
            state.block_until = time.time() + 1800  # 屏蔽30分钟
            state.block_reason = "心理创伤严重，机器人选择逃避"
            logger.critical(f"🚫 逃避模式激活: 用户{user_id[:8]} 被屏蔽30分钟")
        self.clamp_all_values(state)
        self._save_user_state(user_id)
        return state.trauma_score

    def update_mental_fatigue(self, user_id: str, delta: float) -> float:
        state = self.get_user_state(user_id, create_if_missing=True)
        state.mental_fatigue = max(0, min(100, state.mental_fatigue + delta))
        self._save_user_state(user_id)
        return state.mental_fatigue

    def _update_relationship(self, state: UserEmotionState):
        aff = state.affection
        old_rel = state.relationship
        if aff >= 80:
            state.relationship = "挚友"
        elif aff >= 60:
            state.relationship = "好友"
        elif aff >= 35:
            state.relationship = "熟人"
        elif aff >= -10:
            state.relationship = "陌生人"
        elif aff >= -40:
            state.relationship = "冷淡"
        elif aff >= -70:
            state.relationship = "敌对"
        else:
            state.relationship = "仇人"
        if old_rel != state.relationship:
            state.last_relationship = old_rel

    def block_user(self, user_id: str, duration: float, reason: str):
        state = self._user_states.get(user_id)
        if state:
            state.is_blocked = True
            state.block_until = time.time() + duration
            state.block_reason = reason
            logger.warning(f"屏蔽用户 {user_id[:8]}... 时长:{duration:.0f}秒 原因:{reason[:30]}")
            self._save_states()

    def is_user_blocked(self, user_id: str) -> Tuple[bool, Any]:
        state = self._user_states.get(user_id)
        if not state or not state.is_blocked:
            return False, ""
        if time.time() >= state.block_until:
            state.is_blocked = False
            state.block_until = 0.0
            state.block_reason = ""
            self._save_states()
            return False, ""
        trauma_level = 3 if state.trauma_score >= 8 else 2 if state.trauma_score >= 5 else 1
        return True, {
            "blocked": True, "reason": state.block_reason,
            "mental_state": {
                "primary_emotion": "极度委屈",
                "secondary_emotions": ["空洞", "迷茫", "无助", "绝望"],
                "intensity": "最差", "trauma_level": trauma_level,
                "trauma_score": state.trauma_score,
                "thoughts": ["为什么要这样对我...", "我做错了什么...", "我真的受不了了...", "没有人理解我...", "我想逃避这一切...", "为什么总是我受伤..."],
                "physical_feeling": "浑身无力，眼前发黑",
                "psychological_state": "严重心理创伤，保护性解离",
            },
            "response_hint": "...呜呜...我现在心里好委屈...真的好想哭...为什么会这样...我一个人好无助...",
        }

    def add_trauma_record(self, state: UserEmotionState, content: str, level: int,
                           user_id: str, trigger_type: str = "general") -> Dict[str, Any]:
        now = time.time()
        uid_short = user_id[:2] + '*' * (len(user_id) - 2) if len(user_id) > 2 else user_id
        filtered_content = self._filter_bot_names(content)
        existing_record = None
        for trauma in state.trauma_history:
            if trauma.get("user_id") == uid_short:
                existing_record = trauma
                break
        if existing_record:
            old_count = existing_record.get("hurt_count", 1)
            old_level = existing_record.get("level", 1)
            old_intensity = existing_record.get("current_intensity", old_level)
            new_count = old_count + 1
            new_level = max(old_level, level)
            intensity_increase = min(2.0, level * 0.5)
            new_intensity = min(10.0, old_intensity + intensity_increase)
            if new_level >= 4 or new_count >= 3:
                existing_record["is_permanent_scar"] = True
                state.has_trauma_mark = True
                state.trauma_mark_level = min(3, max(state.trauma_mark_level, new_level))
                state.trauma_mark_time = now
            old_content = existing_record.get("content", "")
            new_content = filtered_content[:40]
            existing_record["content"] = f"{old_content} | {new_content}"
            existing_record["level"] = new_level
            existing_record["hurt_count"] = new_count
            existing_record["current_intensity"] = new_intensity
            existing_record["time"] = now
            existing_record["accumulated_weight"] = existing_record.get("accumulated_weight", old_level * 10.0) + (level * 5.0)
            if len(existing_record["content"]) > 200:
                existing_record["content"] = self._fade_memory(existing_record["content"], fade_level=0.3)
                existing_record["corrupted_level"] = existing_record.get("corrupted_level", 0) + 1
            should_log = (new_count % 5 == 0) or (new_count == 3 and existing_record.get("is_permanent_scar"))
            if should_log:
                logger.info(f"创伤累积 | {uid_short} x{new_count} 强度{new_intensity:.1f}")
            result = {"trauma_accumulated": True}
        else:
            is_permanent = level >= 4
            new_record = {
                "time": now, "content": filtered_content[:100], "level": level,
                "user_id": uid_short, "trigger_user_id": user_id,
                "trigger_type": trigger_type, "hurt_count": 1,
                "current_intensity": min(10.0, float(level)),
                "is_permanent_scar": is_permanent,
                "accumulated_weight": float(min(10, level) * 10.0),
                "corrupted_level": 0
            }
            state.trauma_history.append(new_record)
            logger.info(f"新增创伤 | {uid_short} Lv.{level} 强度{new_record['current_intensity']:.1f}")
            result = {"new_trauma_added": True}
        max_records = 20
        if len(state.trauma_history) > max_records:
            for trauma in state.trauma_history:
                if trauma.get("is_permanent_scar", False):
                    corruption_penalty = trauma.get("corrupted_level", 0) * 5.0
                    trauma["sort_weight"] = 1000.0 + trauma.get("accumulated_weight", 0) - corruption_penalty
                else:
                    age_days = (now - trauma.get("time", now)) / 86400
                    decay = math.exp(-0.1 * age_days)
                    trauma["sort_weight"] = trauma.get("accumulated_weight", 10) * decay
            state.trauma_history.sort(key=lambda t: t.get("sort_weight", 0))
            while len(state.trauma_history) > max_records:
                weakest = state.trauma_history[0]
                if not weakest.get("is_permanent_scar", False):
                    state.trauma_history.pop(0)
                else:
                    target_idx = -1
                    for i in range(1, len(state.trauma_history)):
                        if state.trauma_history[i].get("is_permanent_scar"):
                            target_idx = i
                            break
                    if target_idx != -1:
                        target = state.trauma_history[target_idx]
                        corrupted_note = self._fade_memory(weakest["content"], fade_level=0.8)
                        target["content"] = f"<{corrupted_note}> {target['content']}"
                        target["accumulated_weight"] += weakest["accumulated_weight"] * 0.5
                        target["corrupted_level"] = target.get("corrupted_level", 0) + 1
                        state.trauma_history.pop(0)
                    else:
                        break
        permanent_count = sum(1 for t in state.trauma_history if t.get("is_permanent_scar", False))
        if len(state.trauma_history) >= max_records:
            logger.warning(f"创伤池满载 | 总{len(state.trauma_history)}条 永久{permanent_count}条")
        return result

    def _fade_memory(self, content: str, fade_level: float = 0.3) -> str:
        if not content or fade_level <= 0:
            return content
        chars = list(content)
        fade_count = int(len(chars) * fade_level)
        fade_indices = random.sample(range(len(chars)), min(fade_count, len(chars)))
        fade_markers = ["…", "□", "？", "▓", "░"]
        for idx in fade_indices:
            chars[idx] = random.choice(fade_markers)
        return "".join(chars)

    def process_trauma_decay(self, state: UserEmotionState) -> Dict[str, Any]:
        now = time.time()
        result = {"flashback": False, "flashback_content": None, "trauma_faded": False, "permanent_scars": []}
        if not state.trauma_history:
            return result
        hours_since_last = (now - state.trauma_last_recall) / 3600 if state.trauma_last_recall > 0 else 24
        for trauma in state.trauma_history:
            trauma_age_days = (now - trauma.get("time", now)) / 86400
            original_level = trauma.get("level", 1)
            decay_factor = math.exp(-0.03 * trauma_age_days)
            if original_level >= 5:
                min_intensity = original_level * 0.5
            elif original_level >= 4:
                min_intensity = original_level * 0.25
            elif original_level >= 3:
                min_intensity = original_level * 0.1
            else:
                min_intensity = 0.0
            trauma["current_intensity"] = max(min_intensity, original_level * decay_factor)
            if min_intensity > 0 and decay_factor < 0.5:
                trauma["is_permanent_scar"] = True
                if trauma.get("content"):
                    result["permanent_scars"].append(trauma.get("content", "")[:30])
            if trauma["current_intensity"] < 0.05 and min_intensity == 0:
                trauma["faded"] = True
                result["trauma_faded"] = True
        active_traumas = [t for t in state.trauma_history if not t.get("faded", False)]
        permanent_traumas = [t for t in active_traumas if t.get("is_permanent_scar", False)]
        if permanent_traumas and hours_since_last > 2:
            flashback_chance = 0.15
            if random.random() < flashback_chance:
                scar = random.choice(permanent_traumas)
                result["flashback"] = True
                raw_content = scar.get("content", "")[:60]
                trauma_age = (now - scar.get("time", now)) / 86400
                if trauma_age > 30:
                    faded_content = self._fade_memory(raw_content, fade_level=0.5)
                elif trauma_age > 7:
                    faded_content = self._fade_memory(raw_content, fade_level=0.2)
                else:
                    faded_content = raw_content
                result["flashback_content"] = faded_content
                result["flashback_intensity"] = scar.get("current_intensity", 1.0)
                result["is_permanent_flashback"] = True
                state.trauma_last_recall = now
                state.trauma_recall_count += 1
        elif active_traumas and hours_since_last > 4:
            max_intensity = max(t.get("current_intensity", 0) for t in active_traumas)
            flashback_chance = min(0.25, max_intensity * 0.12)
            if random.random() < flashback_chance:
                recent_trauma = max(active_traumas, key=lambda t: t.get("current_intensity", 0))
                result["flashback"] = True
                raw_content = recent_trauma.get("content", "")[:60]
                trauma_age = (now - recent_trauma.get("time", now)) / 86400
                if trauma_age > 14:
                    faded_content = self._fade_memory(raw_content, fade_level=0.3)
                else:
                    faded_content = raw_content
                result["flashback_content"] = faded_content
                result["flashback_intensity"] = recent_trauma.get("current_intensity", 1.0)
                state.trauma_last_recall = now
                state.trauma_recall_count += 1
        total_trauma = sum(t.get("current_intensity", 0) for t in state.trauma_history)
        for trauma in state.trauma_history:
            if trauma.get("current_intensity", 0) > 10:
                trauma["current_intensity"] = 10.0
        state.trauma_score = max(0.0, min(10.0, total_trauma))
        return result

    def apply_time_recovery(self, state: UserEmotionState, hours: float) -> Tuple[float, Dict[str, Any]]:
        if hours <= 0:
            return 0.0, {}
        if hours > 168:
            hours = min(hours, 168.0)
        elif hours > 24:
            hours = hours * 0.8
        recovery_details = {}
        total_recovery = 0.0
        interaction_weight = self._calculate_interaction_weight(state)
        negative_decay_details = self._apply_negative_decay(state, hours, interaction_weight)
        recovery_details.update(negative_decay_details)
        if state.affection < 0:
            if hours >= 4.0:
                old_affection = state.affection
                state.affection = 0.0
                recovery_details["affection_recovery"] = f"{old_affection:.1f}->0"
                total_recovery = abs(old_affection)
            else:
                recovery_rate = hours / 4.0
                old_affection = state.affection
                improvement = abs(state.affection) * recovery_rate
                state.affection = self.clamp_affection(state.affection + improvement)
                total_recovery = improvement
                if improvement > 0.1:
                    recovery_details["affection_recovery"] = f"{old_affection:.1f}->{state.affection:.1f}"
        positive_decay_details = self._apply_positive_decay(state, hours, interaction_weight)
        recovery_details.update(positive_decay_details)
        trauma_decay_details = self._apply_trauma_decay(state, hours, interaction_weight)
        recovery_details.update(trauma_decay_details)
        return total_recovery, recovery_details

    def _calculate_interaction_weight(self, state: UserEmotionState) -> float:
        base_weight = 1.0
        frequency_factor = min(state.interaction_count / 50.0, 2.0)
        relationship_depth = 0.0
        if "好友" in state.relationship or "挚友" in state.relationship:
            relationship_depth = 1.5
        elif "熟人" in state.relationship:
            relationship_depth = 1.2
        elif "陌生人" not in state.relationship:
            relationship_depth = 1.0
        emotion_intensity = abs(state.affection) / 50.0 + abs(state.trust_score) / 50.0
        emotion_intensity = min(emotion_intensity, 2.0)
        trauma_influence = 1.0 - (state.trauma_score / 20.0) * 0.3
        trauma_influence = max(0.4, trauma_influence)
        interaction_weight = (base_weight + frequency_factor) * max(relationship_depth, 0.5) * (1.0 + emotion_intensity) * trauma_influence
        return max(0.1, min(interaction_weight, 5.0))

    def _apply_negative_decay(self, state: UserEmotionState, hours: float, weight: float) -> Dict[str, Any]:
        decay_details = {}
        if state.annoyance > 0:
            if hours >= 4.0:
                old_annoyance = state.annoyance
                state.annoyance = 0.0
                decay_details["annoyance_decay"] = f"{old_annoyance:.1f}->0"
            else:
                decay_rate = hours / 4.0
                old_annoyance = state.annoyance
                state.annoyance = max(0, state.annoyance * (1.0 - decay_rate))
                decay_amount = old_annoyance - state.annoyance
                if decay_amount > 0.1:
                    decay_details["annoyance_decay"] = f"{old_annoyance:.1f}->{state.annoyance:.1f}"
        if state.trust_score < 0:
            if hours >= 4.0:
                old_trust = state.trust_score
                state.trust_score = 0.0
                decay_details["trust_decay"] = f"{old_trust:.1f}->0"
            else:
                recovery_rate = hours / 4.0
                old_trust = state.trust_score
                improvement = abs(state.trust_score) * recovery_rate
                state.trust_score = self.clamp_trust(state.trust_score + improvement)
                if abs(old_trust - state.trust_score) > 0.1:
                    decay_details["trust_decay"] = f"{old_trust:.1f}->{state.trust_score:.1f}"
        return decay_details

    def _apply_positive_decay(self, state: UserEmotionState, hours: float, weight: float) -> Dict[str, Any]:
        decay_details = {}
        if state.affection > 0:
            max_time_hours = 720.0
            time_factor = min(hours, max_time_hours)
            log_decay = math.log(time_factor + 1) / math.log(max_time_hours + 1)
            weight_protection = weight * 0.5 + 0.5
            relationship_protection = 1.0
            if "挚友" in state.relationship:
                relationship_protection = 0.3
            elif "好友" in state.relationship:
                relationship_protection = 0.5
            elif "熟人" in state.relationship:
                relationship_protection = 0.7
            total_decay_factor = log_decay * (1.0 / weight_protection) * relationship_protection
            max_decay_percent = 0.05
            actual_decay_factor = min(total_decay_factor, max_decay_percent)
            old_affection = state.affection
            affection_decay = state.affection * actual_decay_factor
            state.affection = self.clamp_affection(state.affection - affection_decay)
            if affection_decay > 0.1:
                decay_details["affection_decay"] = f"{old_affection:.1f}->{state.affection:.1f}"
        if state.trust_score > 0:
            max_time_hours = 1440.0
            time_factor = min(hours, max_time_hours)
            log_decay = math.log(time_factor + 1) / math.log(max_time_hours + 1)
            weight_protection = weight * 0.3 + 0.7
            total_decay_factor = log_decay * (1.0 / weight_protection)
            max_decay_percent = 0.03
            actual_decay_factor = min(total_decay_factor, max_decay_percent)
            old_trust = state.trust_score
            trust_decay = state.trust_score * actual_decay_factor
            state.trust_score = self.clamp_trust(state.trust_score - trust_decay)
            if trust_decay > 0.1:
                decay_details["positive_trust_decay"] = f"{old_trust:.1f}->{state.trust_score:.1f}"
        return decay_details

    def _apply_trauma_decay(self, state: UserEmotionState, hours: float, weight: float) -> Dict[str, Any]:
        decay_details = {}
        if state.trauma_score <= 0:
            return decay_details
        old_trauma = state.trauma_score
        base_recovery_hours = 4.0
        max_recovery_hours = 5.0
        annoyance_factor = min(state.annoyance / 100.0, 1.0)
        trauma_severity = min(old_trauma / 10.0, 1.0)
        extra_time = (max_recovery_hours - base_recovery_hours) * (annoyance_factor * 0.5 + trauma_severity * 0.5)
        dynamic_recovery_hours = base_recovery_hours + extra_time
        dynamic_recovery_hours = min(dynamic_recovery_hours, max_recovery_hours)
        if hours >= 24.0:
            state.trauma_score = max(0.0, old_trauma * 0.1)
            if state.trauma_score < 0.5:
                state.trauma_score = 0.0
                state.has_trauma_mark = False
                state.trauma_mark_level = 0
                state.trauma_mark_time = 0.0
            state.is_protected = False
            state.protect_until = 0.0
            state.protect_reason = ""
            decay_details["trauma_decay"] = f"{old_trauma:.2f}->{state.trauma_score:.2f}"
            if old_trauma >= 1.0:
                logger.info(f"创伤恢复 | {old_trauma:.1f}->{state.trauma_score:.1f} | 保护解除")
        elif hours >= dynamic_recovery_hours:
            remaining_time = 24.0 - dynamic_recovery_hours
            decay_progress = (hours - dynamic_recovery_hours) / remaining_time if remaining_time > 0 else 1.0
            decay_rate = decay_progress * 0.7
            trauma_decay_amount = old_trauma * decay_rate
            state.trauma_score = self.clamp_trauma(state.trauma_score - trauma_decay_amount)
            if trauma_decay_amount > 0.01:
                decay_details["trauma_decay"] = f"{old_trauma:.2f}->{state.trauma_score:.2f}"
            if old_trauma >= 9.0 and state.trauma_score < 9.0:
                state.is_protected = False
                state.protect_until = 0.0
                state.protect_reason = ""
                logger.info(f"创伤恢复 | 保护解除 | {old_trauma:.1f}->{state.trauma_score:.1f}")
        else:
            decay_rate = hours / (dynamic_recovery_hours * 2)
            trauma_decay_amount = state.trauma_score * decay_rate
            state.trauma_score = self.clamp_trauma(state.trauma_score - trauma_decay_amount)
            if trauma_decay_amount > 0.01:
                decay_details["trauma_decay"] = f"{old_trauma:.2f}->{state.trauma_score:.2f}"
        return decay_details

    def _apply_damping(self, change: float, current_value: float, direction: str,
                        trauma_score: float = 0, user_id: str = "global") -> float:
        user_key = f"{self.stream_id}_{user_id}"
        last_trauma_time = self._trauma_timers.get(user_key, time.time())
        accumulated_time = time.time() - last_trauma_time
        time_balance_factor = min(1.0, accumulated_time / 1800.0)
        base_sensitivity = 1.0
        if trauma_score >= 7:
            base_sensitivity = 1.3
        elif trauma_score >= 5:
            base_sensitivity = 1.15
        final_sensitivity = base_sensitivity * (1 - 0.5 * time_balance_factor)
        if trauma_score >= 5:
            if direction == "loss":
                adjusted = change * final_sensitivity
            else:
                adjusted = change * (2.0 - final_sensitivity)
        else:
            if direction == "loss":
                if current_value <= -60:
                    adjusted = change * 0.3
                elif current_value <= -30:
                    adjusted = change * 0.5
                elif current_value <= 0:
                    adjusted = change * 0.7
                elif current_value >= 50:
                    adjusted = change * 0.8
                else:
                    adjusted = change * 0.9
            else:
                if current_value <= -50:
                    adjusted = change * 1.5
                elif current_value <= -20:
                    adjusted = change * 1.3
                elif current_value <= 0:
                    adjusted = change * 1.1
                elif current_value >= 60:
                    adjusted = change * 0.6
                elif current_value >= 30:
                    adjusted = change * 0.8
                else:
                    adjusted = change
        if abs(change) > 0.1:
            self._trauma_timers[user_key] = time.time()
        return adjusted

    def _get_affection_stage(self, affection: float) -> str:
        if affection >= 70:
            return "intimate"
        elif affection >= 40:
            return "close"
        elif affection >= 10:
            return "friendly"
        elif affection >= -10:
            return "neutral"
        elif affection >= -40:
            return "cold"
        elif affection >= -70:
            return "hostile"
        return "numb"

    def _get_stage_factor(self, stage: str, sentiment: str) -> float:
        if sentiment == "negative":
            factors = {"intimate": 0.1, "close": 0.3, "friendly": 0.6, "neutral": 0.8, "cold": 1.0, "hostile": 0.7, "numb": 0.2}
        else:
            factors = {"intimate": 0.7, "close": 0.9, "friendly": 1.0, "neutral": 1.0, "cold": 1.2, "hostile": 1.5, "numb": 2.0}
        return factors.get(stage, 1.0)

    def check_milestone(self, state: UserEmotionState) -> Optional[str]:
        milestones = {
            "first_positive": (10, "初次好感"), "friendly_start": (25, "友好起步"),
            "close_bond": (50, "亲密关系"), "deep_trust": (70, "深度信任"),
            "soulmate": (90, "灵魂伴侣"), "first_negative": (-20, "初次厌恶"),
            "hostile_start": (-50, "敌意产生"), "hatred": (-80, "深度仇恨"),
        }
        for key, (threshold, name) in milestones.items():
            if key not in state.milestones_reached:
                if threshold > 0 and state.affection >= threshold:
                    state.milestones_reached.append(key)
                    logger.info(f"里程碑 {name} 突破{threshold}")
                    return name
                elif threshold < 0 and state.affection <= threshold:
                    state.milestones_reached.append(key)
                    logger.info(f"里程碑 {name} 跌破{threshold}")
                    return name
        if state.affection > state.highest_affection:
            state.highest_affection = state.affection
        return None

    def get_nickname_for_user(self, state: UserEmotionState, default_name: str) -> str:
        if state.nickname:
            return state.nickname
        stage = self._get_affection_stage(state.affection)
        nickname_map = {
            "intimate": ["亲爱的", "宝贝", "小可爱"],
            "close": ["小" + default_name[:1] if default_name else "你", default_name],
            "friendly": [default_name, "你"], "neutral": [default_name, "你"],
            "cold": ["你", "那个人"], "hostile": ["你", "讨厌的家伙"], "numb": ["...", ""],
        }
        options = nickname_map.get(stage, [default_name])
        return random.choice(options) if options else default_name

    def check_relationship_change(self, state: UserEmotionState, old_relationship: str,
                                    new_relationship: str) -> Dict[str, Any]:
        result = {"changed": False, "event": None, "protected": False}
        if old_relationship == new_relationship:
            return result
        relationship_order = ["仇人", "敌对", "冷淡", "陌生人", "熟人", "好友", "挚友"]
        old_idx = relationship_order.index(old_relationship) if old_relationship in relationship_order else 3
        new_idx = relationship_order.index(new_relationship) if new_relationship in relationship_order else 3
        now = time.time()
        if new_idx > old_idx:
            result["changed"] = True
            result["event"] = f"relationship_upgrade_{new_relationship}"
            state.relationship_protected_until = now + 3600
        elif new_idx < old_idx:
            if state.relationship_protected_until > now:
                result["protected"] = True
                return result
            if old_idx >= 4:
                result["protected"] = True
                state.relationship_protected_until = now + 1800
                return result
            result["changed"] = True
            result["event"] = f"relationship_downgrade_{new_relationship}"
        state.last_relationship = new_relationship
        return result

    def get_interaction_style(self, state: UserEmotionState) -> Dict[str, Any]:
        stage = self._get_affection_stage(state.affection)
        styles = {
            "intimate": {"tone": "亲昵撒娇", "emoji_usage": "frequent", "response_length": "variable", "formality": "very_low", "playfulness": 0.9},
            "close": {"tone": "亲切友好", "emoji_usage": "moderate", "response_length": "normal", "formality": "low", "playfulness": 0.7},
            "friendly": {"tone": "友善热情", "emoji_usage": "occasional", "response_length": "normal", "formality": "medium", "playfulness": 0.5},
            "neutral": {"tone": "平淡", "emoji_usage": "rare", "response_length": "concise", "formality": "medium", "playfulness": 0.3},
            "cold": {"tone": "冷淡疏离", "emoji_usage": "none", "response_length": "short", "formality": "high", "playfulness": 0.1},
            "hostile": {"tone": "敌意明显", "emoji_usage": "none", "response_length": "minimal", "formality": "high", "playfulness": 0.0},
            "numb": {"tone": "麻木冷漠", "emoji_usage": "none", "response_length": "minimal", "formality": "none", "playfulness": 0.0},
        }
        return styles.get(stage, styles["neutral"])

    def analyze_cognitive_impact_by_rules(self, content: str, user_id: str,
                                           current_state: UserEmotionState) -> Dict:
        return {
            "sentiment": "neutral", "intensity": 0.0,
            "trauma_level": 0, "my_feeling": "",
            "is_harassment": False, "is_insult": False,
            "is_comforting": False, "is_betrayal": False,
            "is_revenge_inciting": False,
            "new_impression": current_state.impression,
            "user_intent": "normal", "analysis_mode": "passthrough"
        }

    async def analyze_cognitive_impact(self, content: str, user_id: str,
                                        current_state: UserEmotionState) -> Dict:
        """自我认知分析 - 让AI像有自我意识的存在一样感知用户的言行"""
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config, global_config
            experience_level = min(100, current_state.interaction_count)
            innocence_factor = max(0, 100 - experience_level) / 100.0
            char_age = getattr(global_config.personality, 'age', '16') or '16'
            has_image = "【我看到的图片】" in content
            
            from src.config.prompt_loader import get_prompt, PromptCategory
            innocence_prompt = ""
            if experience_level < 20:
                innocence_prompt = get_prompt(
                    PromptCategory.MODULE, "emotion_tracker", "cognitive_analysis.innocence_low.template",
                    char_age=char_age,
                    interaction_count=current_state.interaction_count
                )
            elif experience_level < 50:
                innocence_prompt = get_prompt(
                    PromptCategory.MODULE, "emotion_tracker", "cognitive_analysis.innocence_medium.template",
                    interaction_count=current_state.interaction_count
                )
                
            eval_first_line = "1. 结合用户说的话和你看到的图片内容，理解对方的意图" if has_image else "1. 理解用户的真实意图"
            
            prompt = get_prompt(
                PromptCategory.MODULE,
                "emotion_tracker",
                "cognitive_analysis.template",
                char_age=char_age,
                innocence_prompt=innocence_prompt,
                interaction_count=current_state.interaction_count,
                impression=current_state.impression,
                relationship=current_state.relationship,
                trust_score=current_state.trust_score,
                experience_level=experience_level,
                content=content[:800],
                eval_first_line=eval_first_line
            )
            request = LLMRequest(model_config.focus_chat, request_type="cognitive_analysis")
            response_text, _ = await request.generate_response_async(prompt)
            if not response_text:
                return {"sentiment": "neutral", "intensity": 0.0, "trauma_level": 0}
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                raw_sentiment = result.get("sentiment", "neutral")
                if isinstance(raw_sentiment, str):
                    clean_sentiment = raw_sentiment.strip().lower()
                    if "negative" in clean_sentiment:
                        result["sentiment"] = "negative"
                    elif "positive" in clean_sentiment:
                        result["sentiment"] = "positive"
                    else:
                        result["sentiment"] = "neutral"
                return result
        except Exception as e:
            logger.warning(f"认知分析解析失败: {e}")
        return {"sentiment": "neutral", "intensity": 0.0, "trauma_level": 0}

    async def process_interaction(self, user_id: str, content: str,
                                   use_llm: bool = True) -> Dict[str, Any]:
        state = self.get_user_state(user_id, create_if_missing=True)
        now = time.time()
        self._refresh_active_window(user_id)
        dist_factor = self._get_distribution_factor()
        hours = (now - state.last_interaction) / 3600 if state.last_interaction > 0 else 0
        self.process_trauma_decay(state)
        self.apply_time_recovery(state, hours)
        if hours > 0:
            decay_rate = 0.02 if state.trauma_score > 1.0 else 0.05
            state.annoyance *= math.exp(-decay_rate * hours)
            fatigue_recovery = min(hours * 5, 20)
            state.mental_fatigue = max(0, state.mental_fatigue - fatigue_recovery)
        state.mental_fatigue = min(100, state.mental_fatigue + 0.5)
        old_affection = state.affection
        old_trust = state.trust_score
        if use_llm:
            impact = await self.analyze_cognitive_impact(content, user_id, state)
        else:
            impact = self.analyze_cognitive_impact_by_rules(content, user_id, state)
        sentiment = impact.get("sentiment", "neutral")
        intensity = impact.get("intensity", 0.0)
        experience_record = {
            "what_happened": {
                "sentiment": sentiment, "intensity": intensity,
                "has_harassment": impact.get("is_harassment", False),
                "has_insult": impact.get("is_insult", False),
                "has_betrayal": impact.get("is_betrayal", False),
                "has_comforting": impact.get("is_comforting", False),
                "content_preview": content[:100] if content else "",
            },
            "current_state": {
                "affection": state.affection, "trust": state.trust_score,
                "trauma_score": state.trauma_score,
                "interaction_count": state.interaction_count,
                "relationship": state.relationship,
            },
            "is_new_user": state.interaction_count <= 10,
        }
        affection_loss = 0.0
        trust_loss = 0.0
        affection_gain = 0.0
        trust_gain = 0.0
        if sentiment == "negative":
            affection_loss = 0.2 + intensity * 0.5
            trust_loss = 0.1 + intensity * 0.3
        elif sentiment == "positive":
            gain_mult = 1.1 if intensity > 0.8 else 1.05 if intensity > 0.5 else 1.0
            affection_gain = (0.05 + intensity * 0.15) * gain_mult
            trust_gain = (0.03 + intensity * 0.1) * gain_mult
        elif sentiment == "neutral":
            affection_gain = 0.02 + intensity * 0.06
            trust_gain = 0.01 + intensity * 0.04
        affection_gain *= dist_factor
        trust_gain *= dist_factor
        dislike_penalty = self._get_dislike_penalty(state)
        if dislike_penalty < 0:
            affection_gain = max(0, affection_gain + dislike_penalty)
        affection_loss = self._apply_damping(affection_loss, state.affection, "loss", state.trauma_score, user_id)
        trust_loss = self._apply_damping(trust_loss, state.trust_score, "loss", state.trauma_score, user_id)
        affection_gain = self._apply_damping(affection_gain, state.affection, "gain", state.trauma_score, user_id)
        trust_gain = self._apply_damping(trust_gain, state.trust_score, "gain", state.trauma_score, user_id)
        affection_loss = min(affection_loss, 12.0)
        trust_loss = min(trust_loss, 10.0)
        state.affection = self.clamp_affection(state.affection - affection_loss + affection_gain)
        state.trust_score = self.clamp_trust(state.trust_score - trust_loss + trust_gain)
        state.annoyance = self.clamp_annoyance(state.annoyance + affection_loss * 0.1 - affection_gain * 0.2)
        trauma_delta = 0.0
        if impact.get("is_harassment"):
            trauma_delta += 0.3 + intensity * 0.5
        if impact.get("is_insult"):
            trauma_delta += 0.2 + intensity * 0.4
        if impact.get("is_betrayal"):
            trauma_delta += 0.5 + intensity * 0.8
        trauma_lvl = impact.get("trauma_level", 0)
        if trauma_lvl >= 2 and trauma_delta < 0.3:
            trauma_delta = max(trauma_delta, trauma_lvl * 0.2)
        trauma_delta = min(trauma_delta, 0.8)
        if sentiment == "positive" and intensity > 0.3:
            trauma_delta -= min(0.5, state.trauma_score * 0.08)
        if trauma_delta > 0:
            state.trauma_score = self.clamp_trauma(state.trauma_score + trauma_delta)
        elif trauma_delta < 0:
            state.trauma_score = self.clamp_trauma(state.trauma_score + trauma_delta)
        else:
            state.trauma_score = self.clamp_trauma(state.trauma_score)
        try:
            psych_state = self.get_psychology_state(user_id)
            if psych_state:
                psych_ctx = {
                    "trauma_score": state.trauma_score,
                    "affection": state.affection,
                    "sentiment": sentiment,
                    "intensity": intensity,
                }
                psych_state.process_stimulus(content, psych_ctx)
                psych_state.accumulate_state(
                    {"inner_chaos_level": psych_state.inner_chaos_level,
                     "surface_mask_strength": psych_state.surface_mask_strength,
                     "apparent_state": psych_state.current_state.value},
                    source="process_interaction"
                )
                state.inner_chaos = psych_state.inner_chaos_level
                state.surface_mask = psych_state.surface_mask_strength
        except Exception:
            pass
        self._update_emotion_dimensions(state, sentiment, intensity, impact)
        self._update_mood_and_attitude(state)
        try:
            is_private = not state.stream_id.startswith("group")
            has_intent, intent_type, intent_intensity = TrainingSystem.detect_training_intent(content, is_private)
            if has_intent and intent_type not in ("pending_llm", "pending_llm_group"):
                training_result = TrainingSystem.process_training(state, intent_type, intent_intensity, is_private)
                if training_result.get("stage_changed"):
                    logger.info(f"调教阶段变化: Lv.{state.training_stage} | {training_result.get('response_hint', '')}")
        except Exception:
            pass
        old_relationship = state.last_relationship
        new_relationship = state.relationship
        if state.affection < -70:
            new_relationship = "仇人"
        elif state.affection < -40:
            new_relationship = "敌对"
        elif state.affection < -15:
            new_relationship = "冷淡"
        elif state.affection > 70:
            new_relationship = "挚友"
        elif state.affection > 50:
            new_relationship = "好友"
        elif state.affection > 25:
            new_relationship = "熟人"
        rel_change = self.check_relationship_change(state, old_relationship, new_relationship)
        if not rel_change.get("protected"):
            state.relationship = new_relationship
        if impact.get("new_impression"):
            state.impression = impact["new_impression"]
        state.interaction_count += 1
        state.last_interaction = now
        self._trigger_learning_if_needed(state)
        if sentiment == "negative":
            experience_record["negative_impact"] = {
                "base_affection_loss": affection_loss, "base_trust_loss": trust_loss, "intensity": intensity,
            }
        elif sentiment == "positive":
            experience_record["positive_impact"] = {
                "affection_gain": affection_gain, "trust_gain": trust_gain, "intensity": intensity,
            }
        else:
            experience_record["neutral_impact"] = {
                "affection_gain": affection_gain, "trust_gain": trust_gain,
            }
        affection_delta = state.affection - old_affection
        trust_delta = state.trust_score - old_trust
        stamina = max(0.0, 100.0 - state.mental_fatigue)
        self._save_states()
        is_harassment = impact.get("is_harassment", False)
        is_insult = impact.get("is_insult", False)
        is_betrayal = impact.get("is_betrayal", False)
        trauma_level = impact.get("trauma_level", 0)
        trauma_flashback = trauma_level >= 2 and state.trauma_score > 3
        return {
            "annoyance": state.annoyance, "affection": state.affection,
            "trust": state.trust_score, "trauma_score": state.trauma_score,
            "reaction_mode": "normal", "stamina": stamina,
            "affection_delta": affection_delta, "trust_delta": trust_delta,
            "stage": state.training_stage, "sub_level": state.submission_level,
            "experience_record": experience_record, "is_test_user": False,
            "is_harassment": is_harassment, "is_insult": is_insult,
            "is_betrayal": is_betrayal, "trauma_flashback": trauma_flashback,
            "trauma_level": trauma_level,
        }

    def _update_emotion_dimensions(self, state: UserEmotionState, sentiment: str,
                                    intensity: float, impact: Dict[str, Any]):
        decay = 0.85
        state.joy *= decay
        state.anticipation *= decay
        state.surprise *= decay
        state.sadness *= decay
        state.disgust *= decay
        state.anger *= decay
        state.fear *= decay
        state.pride *= decay
        state.guilt *= decay
        state.shame *= decay
        state.envy *= decay
        state.gratitude *= decay
        is_harassment = impact.get("is_harassment", False)
        is_insult = impact.get("is_insult", False)
        is_betrayal = impact.get("is_betrayal", False)
        is_comforting = impact.get("is_comforting", False)
        if sentiment == "positive":
            state.joy = min(1.0, state.joy + intensity * 0.4)
            state.gratitude = min(1.0, state.gratitude + intensity * 0.2)
            if intensity > 0.7:
                state.anticipation = min(1.0, state.anticipation + 0.15)
                state.pride = min(1.0, state.pride + 0.1)
            state.intimacy = min(100.0, state.intimacy + intensity * 0.5)
        elif sentiment == "negative":
            state.sadness = min(1.0, state.sadness + intensity * 0.3)
            if is_harassment:
                state.fear = min(1.0, state.fear + intensity * 0.5)
                state.anger = min(1.0, state.anger + intensity * 0.3)
                state.shame = min(1.0, state.shame + intensity * 0.2)
            elif is_insult:
                state.anger = min(1.0, state.anger + intensity * 0.5)
                state.disgust = min(1.0, state.disgust + intensity * 0.3)
            elif is_betrayal:
                state.sadness = min(1.0, state.sadness + intensity * 0.4)
                state.anger = min(1.0, state.anger + intensity * 0.3)
                state.fear = min(1.0, state.fear + intensity * 0.2)
            else:
                state.anger = min(1.0, state.anger + intensity * 0.2)
            state.psychological_pressure = min(100.0, state.psychological_pressure + intensity * 2.0)
        else:
            state.anticipation = min(1.0, state.anticipation + 0.05)
            if intensity > 0.5:
                state.surprise = min(1.0, state.surprise + intensity * 0.15)
        if is_comforting:
            state.gratitude = min(1.0, state.gratitude + 0.3)
            state.joy = min(1.0, state.joy + 0.15)
            state.sadness = max(0.0, state.sadness - 0.2)
            state.psychological_pressure = max(0.0, state.psychological_pressure - 3.0)

    def _update_mood_and_attitude(self, state: UserEmotionState):
        emotions = {
            "开心": state.joy, "期待": state.anticipation,
            "惊讶": state.surprise, "悲伤": state.sadness,
            "厌恶": state.disgust, "愤怒": state.anger,
            "恐惧": state.fear, "骄傲": state.pride,
            "内疚": state.guilt, "羞耻": state.shame,
            "嫉妒": state.envy, "感恩": state.gratitude,
        }
        dominant = max(emotions, key=emotions.get)
        dominant_val = emotions[dominant]
        if dominant_val < 0.1:
            state.mood = "平静"
        elif dominant_val < 0.4:
            state.mood = f"略{dominant}"
        else:
            state.mood = dominant
        if state.affection > 50 and state.trust_score > 30:
            state.attitude = "亲切"
        elif state.affection > 20:
            state.attitude = "友好"
        elif state.affection < -40:
            state.attitude = "敌意"
        elif state.affection < -15:
            state.attitude = "冷淡"
        elif state.annoyance > 50:
            state.attitude = "不耐烦"
        elif state.anger > 0.5:
            state.attitude = "愤怒"
        else:
            state.attitude = "中立"

    @staticmethod
    def build_status_display(state: UserEmotionState, display_name: str = "",
                              delta: Optional[Dict[str, float]] = None,
                              extra: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        main_parts = []
        ext_parts = []
        uid = state.user_id
        # 隐私保护：用英文字符代替数字，避免泄露UID
        uid_masked = f"user_{uid[0]}***{uid[-1]}" if uid and len(uid) > 1 else "user_****"
        name_short = display_name[:1] + '****' if display_name else "用户"
        main_parts.append(f"{name_short}({uid_masked})")
        aff = max(-100.0, min(100.0, state.affection))
        if delta and "affection" in delta:
            d = delta["affection"]
            trend = "升" if d > 0.1 else "降" if d < -0.1 else ""
            main_parts.append(f"好感{aff:+.0f}({d:+.1f}{trend})")
        else:
            main_parts.append(f"好感{aff:+.0f}")
        trust = max(-100.0, min(100.0, state.trust_score))
        if trust != 0:
            if delta and "trust" in delta:
                d = delta["trust"]
                trend = "升" if d > 0.1 else "降" if d < -0.1 else ""
                main_parts.append(f"信任{trust:+.0f}({d:+.1f}{trend})")
            else:
                main_parts.append(f"信任{trust:+.0f}")
        annoy = max(0.0, min(100.0, state.annoyance))
        if annoy > 0.1:
            if delta and "annoyance" in delta:
                main_parts.append(f"烦恼{annoy:.0f}({delta['annoyance']:+.1f})")
            else:
                main_parts.append(f"烦恼{annoy:.0f}")
        trauma = max(0.0, min(10.0, state.trauma_score))
        if trauma > 0.1:
            t_name = "严重" if trauma >= 7 else "敏感" if trauma >= 4 else "轻微"
            if delta and "trauma" in delta:
                main_parts.append(f"创伤{t_name}{trauma:.1f}({delta['trauma']:+.1f})")
            else:
                main_parts.append(f"创伤{t_name}{trauma:.1f}")
        en_to_cn = {"stranger": "陌生人", "acquaintance": "熟人", "friend": "好友",
                    "close_friend": "挚友", "intimate": "挚友", "close": "好友",
                    "cold": "冷淡", "hostile": "敌对", "enemy": "仇人"}
        rel_display = en_to_cn.get(state.relationship, state.relationship)
        main_parts.append(f"{rel_display}")
        if hasattr(state, 'training_stage') and state.training_stage > 0:
            main_parts.append(f"调教Lv.{state.training_stage}")
        if extra and extra.get("persona"):
            ext_parts.append(f"人格:{extra['persona']}")
        try:
            global_rage = EmotionTracker._global_irritation
            if global_rage > 50:
                ext_parts.append(f"全局激怒{global_rage:.0f}")
        except Exception:
            pass
        if extra and extra.get("focus_chat_value") is not None:
            cv = extra["focus_chat_value"]
            cv_max = extra.get("focus_chat_value_max", 100.0)
            cv_pct = (cv / cv_max * 100) if cv_max > 0 else 0
            ext_parts.append(f"聊天值{cv_pct:.0f}%")
            bp = extra.get("brain_power", 0)
            bp_max = extra.get("brain_power_max", 100.0)
            bp_pct = (bp / bp_max * 100) if bp_max > 0 else 0
            ext_parts.append(f"脑力{bp_pct:.0f}%")
        elif extra and extra.get("chat_fatigue") is not None:
            cf = extra["chat_fatigue"]
            is_resting = extra.get("is_fatigued", False)
            rest_remaining = extra.get("fatigue_remaining", 0)
            if is_resting and rest_remaining > 0:
                ext_parts.append(f"疲劳休息{rest_remaining}秒")
            elif cf > 0:
                threshold = extra.get("fatigue_threshold", 120.0)
                pct = min(cf / threshold, 1.0) * 100
                ext_parts.append(f"聊天值{100-pct:.0f}%")
        if state.is_blocked and hasattr(state, 'block_until'):
            import time as _t
            rem = int(state.block_until - _t.time())
            if rem > 0:
                ext_parts.append(f"屏蔽{rem}秒")
        if extra and extra.get("resting") is not None:
            rest_sec = extra["resting"]
            if rest_sec > 0:
                ext_parts.append(f"休息{rest_sec}秒")
        if extra and extra.get("mode"):
            mode_cn = {"focus": "专注模式", "normal": "普通模式"}.get(extra['mode'], extra['mode'])
            ext_parts.append(f"{mode_cn}")
        if extra and extra.get("memory_count") is not None:
            ext_parts.append(f"记忆={extra['memory_count']}")
        return " | ".join(main_parts), " | ".join(ext_parts)

    def get_recent_interactions(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        if user_id not in self._interaction_history:
            return []
        return self._interaction_history[user_id][:limit]

    def get_emotion_modifiers(self, user_id: str) -> Dict[str, float]:
        state = self.get_user_state(user_id)
        return {
            "reply_willingness": self._calc_reply_willingness(state),
            "reply_enthusiasm": self._calc_reply_enthusiasm(state),
            "tool_enthusiasm": self._calc_tool_enthusiasm(state),
            "proactive_chance": self._calc_proactive_chance(state),
            "tone_warmth": self._calc_tone_warmth(state),
            "patience_level": self._calc_patience_level(state),
            "sharing_openness": self._calc_sharing_openness(state),
        }

    def _calc_reply_willingness(self, state: UserEmotionState) -> float:
        base = 0.7
        base += state.affection * 0.002
        base -= state.annoyance * 0.003
        base -= state.trauma_score * 0.05
        base += state.trust_score * 0.001
        return max(0.1, min(1.0, base))

    def _calc_reply_enthusiasm(self, state: UserEmotionState) -> float:
        base = 0.6
        base += state.affection * 0.003
        base += state.trust_score * 0.001
        base -= state.trauma_score * 0.03
        base -= state.annoyance * 0.002
        return max(0.2, min(1.0, base))

    def _calc_tool_enthusiasm(self, state: UserEmotionState) -> float:
        base = 0.5
        if state.affection > 50:
            base += 0.3
        elif state.affection > 20:
            base += 0.15
        elif state.affection < -20:
            base -= 0.2
        if state.annoyance > 50:
            base -= 0.3
        elif state.annoyance > 30:
            base -= 0.15
        if state.trust_score < 0:
            base -= 0.2
        return max(0.1, min(1.0, base))

    def _calc_proactive_chance(self, state: UserEmotionState) -> float:
        base = 0.3
        if state.affection > 70:
            base = 0.7
        elif state.affection > 40:
            base = 0.5
        elif state.affection > 10:
            base = 0.35
        elif state.affection < -30:
            base = 0.1
        if state.relationship in ["挚友"]:
            base = min(1.0, base + 0.2)
        elif state.relationship in ["仇人", "敌对"]:
            base = max(0.0, base - 0.2)
        if state.trauma_score > 5:
            base *= 0.5
        return max(0.0, min(1.0, base))

    def _calc_tone_warmth(self, state: UserEmotionState) -> float:
        base = 0.5
        base += state.affection * 0.004
        base -= state.annoyance * 0.003
        if state.trauma_score > 5:
            base -= 0.3
        elif state.trauma_score > 2:
            base -= 0.15
        if state.relationship in ["挚友"]:
            base = min(1.0, base + 0.2)
        elif state.relationship in ["仇人"]:
            base = max(0.0, base - 0.3)
        return max(0.1, min(1.0, base))

    def _calc_patience_level(self, state: UserEmotionState) -> float:
        base = 0.7
        base -= state.annoyance * 0.006
        base += state.affection * 0.002
        base -= state.trauma_score * 0.04
        return max(0.1, min(1.0, base))

    def _calc_sharing_openness(self, state: UserEmotionState) -> float:
        base = 0.4
        base += state.trust_score * 0.004
        base += state.affection * 0.002
        if state.relationship in ["挚友"]:
            base = min(1.0, base + 0.3)
        elif state.relationship in ["仇人", "敌对"]:
            base = max(0.0, base - 0.3)
        return max(0.1, min(1.0, base))

    def _get_mood_context(self, state: UserEmotionState, trauma_level: Optional[int] = None) -> Dict[str, Any]:
        if trauma_level is None:
            if state.trauma_score >= 7:
                trauma_level = 3
            elif state.trauma_score >= 4:
                trauma_level = 2
            else:
                trauma_level = 1
        mood_level_map = {3: "complete_breakdown", 2: "critical", 1: "mild"}
        mood_level = mood_level_map.get(trauma_level, "stable")
        primary_emotions = {3: "绝望", 2: "惊恐", 1: "不安"}
        if state.annoyance >= 80:
            if trauma_level >= 3:
                primary_emotion = "恐惧愤怒"
            elif trauma_level >= 2:
                primary_emotion = "愤怒恐惧"
            else:
                primary_emotion = "愤怒"
        elif state.affection <= -80:
            if trauma_level >= 3:
                primary_emotion = "绝望憎恨"
            elif trauma_level >= 2:
                primary_emotion = "厌恶恐惧"
            else:
                primary_emotion = "厌恶"
        else:
            primary_emotion = primary_emotions.get(trauma_level, "不安")
        triggers = []
        if state.affection <= -80:
            triggers.append("极度厌恶")
        elif state.affection <= -50:
            triggers.append("强烈厌恶")
        if state.trauma_score >= 8:
            triggers.append(f"严重创伤({state.trauma_score:.1f})")
        elif state.trauma_score >= 6:
            triggers.append(f"中度创伤({state.trauma_score:.1f})")
        elif state.trauma_score >= 5:
            triggers.append(f"轻度创伤({state.trauma_score:.1f})")
        if state.annoyance >= 80:
            triggers.append("极度烦躁")
        elif state.annoyance >= 60:
            triggers.append("高度烦躁")
        stability_map = {
            10: "completely_unstable", 9: "extremely_unstable",
            8: "critically_unstable", 7: "highly_unstable",
            6: "moderately_unstable", 5: "unstable"
        }
        stability = stability_map.get(trauma_level, "stable")
        return {
            "mood_level": mood_level, "primary_emotion": primary_emotion,
            "triggers": triggers,
            "intensity": min(10, trauma_level + int(state.annoyance / 20)),
            "stability": stability, "trauma_level": trauma_level,
            "sensitivity_multiplier": trauma_level * 0.5 + 2.5
        }

    def _log_interaction_summary(self, user_id: str, state: UserEmotionState,
                                  affection_delta: float, trust_delta: float,
                                  reaction_mode: str, should_block: bool,
                                  block_reason: str = "",
                                  training_result: Optional[Dict] = None,
                                  persona_switch_info: Optional[Dict] = None):
        try:
            is_maintenance_mode = self._is_maintenance_mode()
            psychological_state = self._get_psychological_state()
            surface_mask = psychological_state.get("surface_mask", 0)
            inner_chaos = psychological_state.get("inner_chaos", 0)
            is_system_breakdown = self._check_system_breakdown(surface_mask, inner_chaos)
            uid_short = user_id[:2] + '*' * (len(user_id) - 2) if len(user_id) > 2 else user_id
            if is_maintenance_mode:
                self._log_maintenance_status(uid_short, state, affection_delta, trust_delta,
                    reaction_mode, should_block, persona_switch_info, training_result,
                    surface_mask, inner_chaos, is_system_breakdown)
                return
            self._log_summary_status(uid_short, state, affection_delta, trust_delta,
                reaction_mode, should_block, surface_mask, inner_chaos, is_system_breakdown)
        except Exception as e:
            logger.debug(f"状态显示失败: {e}")

    def _save_user_state(self, user_id: str):
        if not self._db:
            return
        state = self._user_states.get(user_id)
        if state:
            try:
                self._db.save_user_state(user_id, state.to_dict())
            except Exception as e:
                logger.error(f"保存用户状态失败: {e}")

    def _save_states(self):
        if not self._db:
            return
        try:
            for uid, state in self._user_states.items():
                self._db.save_user_state(uid, asdict(state))
            logger.debug(f"保存成功: {len(self._user_states)} 个用户状态")
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
        try:
            from src.chat.heart_flow.persistent_state_db import get_persistent_state_db
            state_db = get_persistent_state_db()
            for uid, state in self._user_states.items():
                state_db.save_emotion_state(
                    stream_id=self.stream_id,
                    user_id=uid,
                    affection=state.affection,
                    annoyance=state.annoyance,
                    trust_score=state.trust_score,
                    trauma_score=state.trauma_score,
                    last_interaction=state.last_interaction,
                )
        except Exception as e:
            logger.debug(f"持久化状态同步失败: {e}")

    def _load_states(self):
        if not self._db:
            return
        try:
            all_states = self._db.load_all_states()
            if not all_states:
                return
            for user_id, data in all_states.items():
                state = UserEmotionState(
                    user_id=user_id, stream_id=self.stream_id,
                    affection=data.get("affection", 0.0),
                    trust_score=data.get("trust_score", 0.0),
                    annoyance=data.get("annoyance", 0.0),
                    trauma_score=data.get("trauma_score", 0.0),
                    trauma_history=data.get("trauma_history", []),
                    relationship=data.get("relationship", "陌生人"),
                    interaction_count=data.get("interaction_count", 0),
                    last_interaction=data.get("last_interaction", 0.0),
                    trauma_last_recall=data.get("trauma_last_recall", 0.0),
                    trauma_recall_count=data.get("trauma_recall_count", 0),
                    baseline_affection=data.get("baseline_affection", 0.0),
                    is_blocked=data.get("is_blocked", False),
                    block_until=data.get("block_until", 0.0),
                    block_reason=data.get("block_reason", ""),
                    has_trauma_mark=data.get("has_trauma_mark", False),
                    trauma_mark_level=data.get("trauma_mark_level", 0),
                    trauma_mark_time=data.get("trauma_mark_time", 0.0),
                    milestones_reached=data.get("milestones_reached", []),
                    highest_affection=data.get("highest_affection", 0.0),
                    nickname=data.get("nickname", ""),
                    last_relationship=data.get("last_relationship", "陌生人"),
                    positive_interactions=data.get("positive_interactions", 0),
                    negative_interactions=data.get("negative_interactions", 0),
                    intimacy=data.get("intimacy", 0.0),
                    submission_level=data.get("submission_level", 0.0),
                    training_stage=data.get("training_stage", 0),
                    training_progress=data.get("training_progress", 0.0),
                    training_resistance=data.get("training_resistance", 100.0),
                    trained_responses=data.get("trained_responses", []),
                    last_training_time=data.get("last_training_time", 0.0),
                    training_streak=data.get("training_streak", 0),
                    authority_level=data.get("authority_level", 0.0),
                    intimacy_level=data.get("intimacy_level", 0.0),
                    is_protected=data.get("is_protected", False),
                    protect_until=data.get("protect_until", 0.0),
                    protect_reason=data.get("protect_reason", ""),
                    protect_count=data.get("protect_count", 0),
                    cumulative_negative_count=data.get("cumulative_negative_count", 0),
                    cumulative_positive_count=data.get("cumulative_positive_count", 0),
                    psychological_pressure=data.get("psychological_pressure", 0.0),
                    mental_fatigue=data.get("mental_fatigue", 0.0),
                    trust_accumulation=data.get("trust_accumulation", 0.0),
                    distrust_accumulation=data.get("distrust_accumulation", 0.0),
                    trauma_accumulation=data.get("trauma_accumulation", 0.0),
                    volatility=data.get("volatility", 1.0),
                    surface_mask=data.get("surface_mask", 10.0),
                    inner_chaos=data.get("inner_chaos", 0.0),
                )
                self._user_states[user_id] = state
            logger.debug(f"加载了 {len(all_states)} 个用户状态")
        except Exception as e:
            logger.warning(f"加载用户状态失败: {e}")
        try:
            from src.chat.heart_flow.persistent_state_db import get_persistent_state_db
            state_db = get_persistent_state_db()
            now = time.time()
            calibration_changes = []
            for uid in list(self._user_states.keys()):
                emotion_data = state_db.load_emotion_state(self.stream_id, uid)
                if not emotion_data:
                    continue
                offline_duration = emotion_data.get("offline_duration", 0.0)
                if offline_duration <= 0:
                    continue
                state = self._user_states[uid]
                hours = offline_duration / 3600.0
                old_aff = state.affection
                old_trust = state.trust_score
                old_ann = state.annoyance
                old_trauma = state.trauma_score
                if state.annoyance > 0:
                    state.annoyance = max(0.0, state.annoyance - min(state.annoyance, 5.0 * hours))
                if state.trauma_score > 0 and state.trauma_score < 3:
                    state.trauma_score = max(0.0, state.trauma_score - min(state.trauma_score, 0.1 * hours))
                if state.trust_score < 0 and state.trust_score > -20:
                    state.trust_score = min(0.0, state.trust_score + min(abs(state.trust_score), 1.0 * hours))
                changes = []
                if abs(state.affection - old_aff) > 0.5:
                    changes.append(f"好感{old_aff:.0f}->{state.affection:.0f}")
                if abs(state.annoyance - old_ann) > 0.5:
                    changes.append(f"烦恼{old_ann:.0f}->{state.annoyance:.0f}")
                if abs(state.trauma_score - old_trauma) > 0.1:
                    changes.append(f"创伤{old_trauma:.1f}->{state.trauma_score:.1f}")
                if changes:
                    uid_short = uid[:2] + '**' if len(uid) > 2 else uid
                    calibration_changes.append(f"{uid_short}({', '.join(changes)})")
            if calibration_changes:
                display = calibration_changes[:2]
                suffix = f" | ...共{len(calibration_changes)}人" if len(calibration_changes) > 2 else ""
                logger.info(f"认知校准: {' | '.join(display)}{suffix}")
        except Exception as e:
            logger.debug(f"持久化校准失败: {e}")

    def get_all_user_ids(self) -> List[str]:
        return list(self._user_states.keys())

    def get_statistics(self) -> Dict:
        if self._db:
            return self._db.get_statistics()
        return {"total_users": len(self._user_states), "avg_affection": 0.0}

    def save_psychology_state(self, user_id: str) -> bool:
        psych_state = self.get_psychology_state(user_id)
        if not psych_state:
            return False
        try:
            from src.chat.heart_flow.persistent_state_db import get_persistent_state_db
            state_db = get_persistent_state_db()
            state_db.save_psychology_state(
                stream_id=self.stream_id,
                user_id=user_id,
                psych_state_data=psych_state.to_dict()
            )
            return True
        except Exception as e:
            logger.debug(f"心理状态保存失败: {e}")
            return False

    def load_psychology_state(self, user_id: str) -> bool:
        try:
            from src.chat.heart_flow.persistent_state_db import get_persistent_state_db
            state_db = get_persistent_state_db()
            psych_data = state_db.get_psychology_state(self.stream_id, user_id)
            if not psych_data:
                return False
            psych_state = self.get_psychology_state(user_id)
            if psych_state:
                psych_state.from_dict(psych_data)
                return True
            return False
        except Exception as e:
            logger.debug(f"心理状态加载失败: {e}")
            return False

    def _is_maintenance_mode(self) -> bool:
        try:
            import os
            return os.environ.get("MAINTENANCE_MODE", "false").lower() == "true"
        except Exception:
            return False

    def _is_test_user(self, user_id: str) -> bool:
        try:
            from src.config.config import global_config
            test_ids = getattr(global_config, 'test_user_ids', [])
            if not test_ids:
                return False
            if isinstance(test_ids, str):
                test_ids = [uid.strip() for uid in test_ids.split(",") if uid.strip()]
            return str(user_id) in test_ids
        except Exception:
            return False

    def _get_psychological_state(self) -> Dict[str, float]:
        psychological_state = {"surface_mask": 0.0, "inner_chaos": 0.0}
        try:
            from src.modules.modcore.dynamic_persona.persona_switcher import get_persona_switcher
            switcher = get_persona_switcher()
            if switcher:
                active_info = switcher.get_active_info(self.stream_id)
                if active_info and not active_info.get("is_main", True):
                    persona_obj = active_info.get("persona_obj")
                    if persona_obj:
                        psychological_state["surface_mask"] = getattr(persona_obj, "surface_mask_strength", 0)
                        psychological_state["inner_chaos"] = getattr(persona_obj, "inner_chaos_level", 0)
        except Exception:
            pass
        return psychological_state

    def _check_system_breakdown(self, surface_mask: float, inner_chaos: float) -> bool:
        return surface_mask >= 9.0 or inner_chaos >= 9.0

    def _filter_bot_names(self, text: str) -> str:
        if not text:
            return text
        import re as _re
        bot_names = set()
        try:
            from src.config.config import global_config
            if global_config.bot.nickname:
                bot_names.add(global_config.bot.nickname)
            if global_config.bot.alias_names:
                bot_names.update(global_config.bot.alias_names)
        except Exception:
            pass
        result = text
        for name in bot_names:
            if name and len(name) >= 2:
                result = _re.sub(_re.escape(name), "", result, flags=_re.IGNORECASE)
        result = _re.sub(r'\s+', ' ', result).strip()
        return result if result else text

    def _get_reaction_zh(self, reaction: str) -> str:
        mapping = {
            "normal": "正常", "hurt": "受伤", "severe_hurt": "重创",
            "heartbroken": "心碎", "betrayed": "背叛", "insulted": "羞辱",
            "hostile": "敌对", "annoyed": "烦躁", "slightly_annoyed": "轻微烦躁"
        }
        return mapping.get(reaction, reaction)

    _last_recovery_log_time: float = 0.0
    _recovery_log_interval: float = 60.0

    def apply_time_recovery_to_all(self) -> Dict[str, Any]:
        if not self._user_states:
            return {}
        all_changes = {}
        now = time.time()
        should_log = now - EmotionTracker._last_recovery_log_time >= EmotionTracker._recovery_log_interval
        for uid, state in self._user_states.items():
            try:
                hours = (now - state.last_interaction) / 3600 if state.last_interaction > 0 else 0
                if hours <= 0:
                    continue
                old_trauma = state.trauma_score
                old_affection = state.affection
                old_trust = state.trust_score
                old_annoyance = state.annoyance
                _, recovery_details = self.apply_time_recovery(state, hours)
                psych_result = PsychologicalStateSystem.process_time_decay(state, hours)
                if psych_result.get("recovered", 0) != 0:
                    state.trauma_score = max(0, state.trauma_score - psych_result["recovered"])
                if recovery_details or abs(old_trauma - state.trauma_score) > 0.01:
                    uid_short = uid[:2] + '*' * min(6, len(uid) - 2) if len(uid) > 2 else uid
                    all_changes[uid_short] = {
                        "details": recovery_details, "state": state, "hours": hours,
                        "old_values": {"trauma": old_trauma, "affection": old_affection,
                                       "trust": old_trust, "annoyance": old_annoyance},
                    }
            except Exception as e:
                logger.debug(f"状态恢复处理用户{uid[:8]}失败: {e}")
        if all_changes and should_log:
            EmotionTracker._last_recovery_log_time = now
            for uid_short, data in list(all_changes.items())[:5]:
                st = data["state"]
                old = data.get("old_values", {})
                elapsed = data.get("hours", 0)
                parts = []
                if abs(old.get("annoyance", 0) - st.annoyance) > 0.1:
                    parts.append(f"烦躁{old['annoyance']:.1f}->{st.annoyance:.1f}")
                if abs(old.get("affection", 0) - st.affection) > 0.1:
                    parts.append(f"好感{old['affection']:.0f}->{st.affection:.0f}")
                if abs(old.get("trust", 0) - st.trust_score) > 0.1:
                    parts.append(f"信任{old['trust']:.0f}->{st.trust_score:.0f}")
                if abs(old.get("trauma", 0) - st.trauma_score) > 0.01:
                    parts.append(f"创伤{old['trauma']:.1f}->{st.trauma_score:.1f}")
                if elapsed > 0.1:
                    time_str = f"{elapsed:.1f}h" if elapsed < 24 else f"{elapsed/24:.1f}d"
                    parts.append(f"经过{time_str}")
                if parts:
                    logger.info(f"状态恢复 {uid_short} | {' '.join(parts)}")
            if len(all_changes) > 5:
                logger.info(f"状态恢复 | 共{len(all_changes)}个用户已更新")
            try:
                from src.chat.heart_flow.persistent_state_db import get_persistent_state_db
                state_db = get_persistent_state_db()
                for uid_short, data in all_changes.items():
                    st = data["state"]
                    state_db.save_emotion_state(
                        stream_id=self.stream_id,
                        user_id=st.user_id,
                        affection=st.affection,
                        annoyance=st.annoyance,
                        trust_score=st.trust_score,
                        trauma_score=st.trauma_score,
                        last_interaction=now,
                        has_trauma_mark=st.has_trauma_mark,
                        trauma_mark_level=st.trauma_mark_level,
                        is_protected=st.is_protected,
                        protect_until=st.protect_until,
                    )
            except Exception as e:
                logger.debug(f"状态持久化失败: {e}")
        return all_changes

    def _get_llm(self):
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            return LLMRequest(model_set=model_config.model_task_config.utils, request_type="emotion_analysis")
        except Exception:
            return None

    def _trigger_learning_if_needed(self, state: "UserEmotionState"):
        try:
            from src.config.config import global_config
            interval = global_config.learning.learning_trigger_interval
        except Exception:
            interval = 20
        if state.interaction_count % interval == 0 and state.interaction_count > 0:
            try:
                import asyncio
                asyncio.create_task(self._run_learning_task(state))
            except Exception:
                pass

    async def _run_learning_task(self, state: "UserEmotionState"):
        try:
            from src.modules.learning.dynamic_vocabulary import get_dynamic_vocabulary
            vocab = get_dynamic_vocabulary()
            from src.common.database.database_model import Messages
            rows = (
                Messages.select(Messages.user_id, Messages.processed_plain_text)
                .where(Messages.chat_id == self.stream_id)
                .order_by(Messages.time.desc())
                .limit(30)
            )
            messages = []
            for row in reversed(list(rows)):
                txt = (row.processed_plain_text or "").strip()
                if txt:
                    messages.append({"role": str(row.user_id), "content": txt})
            if messages:
                result = await vocab.learn_from_messages(messages, self.stream_id)
                if result.get("expressions", 0) > 0 or result.get("jargons", 0) > 0:
                    logger.info(f"[群体学习] 表达{result['expressions']} 黑话{result['jargons']}")
        except Exception as e:
            logger.debug(f"学习任务失败: {e}")

    def get_group_style_prompt(self, user_id: str) -> str:
        state = self._user_states.get(user_id)
        if not state:
            return ""
        if state.affection >= 60:
            return "对方是你非常喜欢的人，说话时带着温暖和亲切。"
        elif state.affection >= 30:
            return "对方是你的朋友，说话轻松自然。"
        elif state.affection <= -30:
            return "你对这个人不太喜欢，说话可能冷淡。"
        return ""

    def log_training_result(self, state: "UserEmotionState", result: Dict[str, Any], user_id: str):
        stage = state.training_stage
        emoji = self.get_stage_emoji(stage)
        stage_name = TrainingSystem.STAGE_NAMES[stage]
        intent = result.get("intent_type", "unknown")
        intent_emojis = {"praise": "👏", "reward": "🎁", "command": "📢", "discipline": "⚡", "tease": "😏"}
        intent_emoji = intent_emojis.get(intent, "❓")
        logger.info(
            f"🎀 {intent_emoji} {intent} → "
            f"用户 {user_id[:6]}** | "
            f"阶段 {emoji}{stage_name} | "
            f"服从+{result.get('submission_change', 0):.1f} | "
            f"羞耻{'+' if result.get('shyness_change', 0) >= 0 else ''}{result.get('shyness_change', 0):.1f}"
        )

    def get_stage_emoji(self, stage: int) -> str:
        stage_emojis = {0: "🌱", 1: "🌿", 2: "🌸", 3: "🌺", 4: "🌹", 5: "💎"}
        return stage_emojis.get(stage, "🌱")

    def _get_trauma_persona_types_by_level(self, state: "UserEmotionState", trauma_level: int = 0) -> List[str]:
        if isinstance(state, (int, float)):
            trauma_score = float(state)
            trauma_level = int(trauma_score)
        else:
            trauma_score = state.trauma_score if hasattr(state, 'trauma_score') else 0
            trauma_level = trauma_level or int(trauma_score)
        if trauma_level >= 9:
            base_states = [
                {"name": "内心世界崩塌", "traits": ["世界崩塌感", "情绪混乱", "逃避现实", "自我怀疑"],
                 "voice": "我...我不知道该怎么办了...一切都乱套了..."},
                {"name": "灵魂撕裂", "traits": ["灵魂撕裂", "身份迷失", "痛苦尖叫", "自我毁灭"],
                 "voice": "救救我...我的灵魂在撕裂...我快要死了..."},
                {"name": "现实幻灭", "traits": ["世界观崩塌", "信任幻灭", "现实扭曲", "存在危机"],
                 "voice": "为什么...为什么一切都是假的...我该相信什么..."},
            ]
        elif trauma_level >= 7:
            base_states = [
                {"name": "心理防线崩溃", "traits": ["防线崩溃", "情绪失控", "自我保护失效", "内心风暴"],
                 "voice": "我...我撑不住了...所有的情绪都涌出来了..."},
                {"name": "创伤阴影笼罩", "traits": ["阴影笼罩", "过去创伤", "生活折磨", "无助感"],
                 "voice": "那些阴影...它们又来了...我逃不掉..."},
                {"name": "情绪漩涡", "traits": ["情绪漩涡", "情感混乱", "现实迷失", "内心挣扎"],
                 "voice": "我的情绪...乱成一团...我不知道什么是真的..."},
            ]
        elif trauma_level >= 5:
            base_states = [
                {"name": "敏感易伤", "traits": ["异常敏感", "易受伤害", "内心脆弱", "自我保护"],
                 "voice": "我太敏感了...一句话就能伤害到我..."},
                {"name": "情绪波动剧烈", "traits": ["情绪波动", "情感不稳", "内心冲突", "自我挣扎"],
                 "voice": "我的情绪...一会儿上一会儿下...我控制不了..."},
                {"name": "信任危机", "traits": ["信任缺失", "怀疑一切", "人际恐惧", "社交退缩"],
                 "voice": "我...我不相信任何人了...太可怕了..."},
            ]
        elif trauma_level >= 3:
            base_states = [
                {"name": "内心不安", "traits": ["莫名不安", "心神不宁", "焦虑感", "预感不好"],
                 "voice": "我总是觉得不安...好像有什么事要发生..."},
                {"name": "烦躁易怒", "traits": ["容易烦躁", "脾气暴躁", "情绪失控", "自我反省"],
                 "voice": "我...我又发脾气了...对不起，我控制不了..."},
                {"name": "社交疲惫", "traits": ["社交疲惫", "精力耗尽", "独处渴望", "人际倦怠"],
                 "voice": "社交太累了...我只想一个人待着..."},
            ]
        else:
            return []
        persona_templates = []
        for tmpl in base_states:
            adjusted = tmpl.copy()
            if hasattr(state, 'affection'):
                if state.affection <= -80:
                    adjusted["traits"] = adjusted["traits"] + ["极度仇恨", "冷酷无情", "内心黑暗"]
                    adjusted["voice"] += "我恨这个世界...我恨所有人..."
                elif state.affection <= -50:
                    adjusted["traits"] = adjusted["traits"] + ["疏离冷淡", "人际退缩", "情感封闭"]
                    adjusted["voice"] += "我不想和任何人亲近了..."
                if hasattr(state, 'annoyance') and state.annoyance >= 80:
                    adjusted["traits"] = adjusted["traits"] + ["极度烦躁", "易怒冲动", "情绪爆发"]
                    adjusted["voice"] += "我快要爆炸了...受不了了..."
                elif hasattr(state, 'annoyance') and state.annoyance >= 50:
                    adjusted["traits"] = adjusted["traits"] + ["急躁易怒", "情绪不稳", "自我克制"]
            persona_templates.append(adjusted["name"])
        return persona_templates

    def _log_maintenance_status(self, uid_short, state, affection_delta, trust_delta, reaction_mode, should_block, persona_switch_info, training_result, surface_mask, inner_chaos, is_system_breakdown):
        try:
            affection_emoji = self._get_affection_emoji(state.affection)
            trust_emoji = self._get_trust_emoji(state.trust_score)
            trauma_emoji = self._get_trauma_emoji(int(state.trauma_score))
            affection_arrow = "↑" if affection_delta > 0 else "↓" if affection_delta < 0 else "→"
            trust_arrow = "↑" if trust_delta > 0 else "↓" if trust_delta < 0 else "→"
            trauma_level = 3 if state.trauma_score >= 7 else 2 if state.trauma_score >= 4 else 1
            trauma_name = f"创伤高({state.trauma_score:.0f})" if trauma_level == 3 else f"创伤中({state.trauma_score:.0f})" if trauma_level == 2 else f"创伤低({state.trauma_score:.0f})"
            relationship_emoji = self._get_relationship_emoji(state.relationship)
            rel_name_map = {"挚友": "挚友", "好友": "好友", "熟人": "熟人", "陌生人": "陌生人", "冷淡": "冷淡", "敌对": "敌对", "仇人": "仇人"}
            rel_name = rel_name_map.get(state.relationship, "未知")
            reaction_emoji = self._get_reaction_emoji(reaction_mode)
            surface_mask_emoji = self._get_surface_mask_emoji(surface_mask)
            surface_mask_name = self._get_surface_mask_name(surface_mask)
            inner_chaos_emoji = self._get_inner_chaos_emoji(inner_chaos)
            inner_chaos_name = self._get_inner_chaos_name(inner_chaos)
            status_parts = [
                f"{uid_short}",
                f"{affection_emoji}{state.affection:.0f}{affection_arrow}",
                f"{trust_emoji}{state.trust_score:.0f}{trust_arrow}",
                f"{trauma_emoji}{trauma_name}{state.trauma_score:.1f}",
                f"{relationship_emoji}{rel_name}",
                f"{reaction_emoji}",
            ]
            extra_info = []
            extra_info.append(f"{surface_mask_emoji}{surface_mask_name}{surface_mask:.1f}/10")
            extra_info.append(f"{inner_chaos_emoji}{inner_chaos_name}{inner_chaos:.1f}/10")
            if is_system_breakdown:
                extra_info.append("😱系统崩溃")
            if should_block:
                extra_info.append("🛡️保护中")
            if persona_switch_info:
                persona_type = persona_switch_info.get("type", "未知")
                duration = persona_switch_info.get("duration", 0)
                extra_info.append(f"👤{persona_type}({duration}s)")
            if training_result and training_result.get("stage_changed"):
                new_stage = training_result.get("new_stage", 0)
                extra_info.append(f"🎯阶段{new_stage}")
            main_status = " | ".join(status_parts)
            if extra_info:
                full_status = f"{main_status} | {' | '.join(extra_info)}"
            else:
                full_status = main_status
            logger.info(f"{full_status}")
        except Exception as e:
            logger.debug(f"维护模式状态显示失败: {e}")

    def _log_summary_status(self, uid_short, state, affection_delta, trust_delta, reaction_mode, should_block, surface_mask, inner_chaos, is_system_breakdown):
        try:
            affection_arrow = "↑" if affection_delta > 0 else "↓" if affection_delta < 0 else ""
            trust_arrow = "↑" if trust_delta > 0 else "↓" if trust_delta < 0 else ""
            if state.trauma_score >= 7:
                trauma_display = f"创伤Lv3({state.trauma_score:.1f})"
            elif state.trauma_score >= 4:
                trauma_display = f"创伤Lv2({state.trauma_score:.1f})"
            else:
                trauma_display = f"创伤Lv1({state.trauma_score:.1f})"
            if inner_chaos >= 5.0:
                psych_status = "心理波动"
            elif surface_mask <= 3.0:
                psych_status = "心理低落"
            else:
                psych_status = "平静"
            energy_display = ""
            try:
                from src.proactive.energy_tracker import get_engagement_energy_manager
                energy_mgr = get_engagement_energy_manager()
                energy_status = energy_mgr.get_status(self.stream_id)
                energy_val = energy_status.get("energy", 0.0)
                energy_display = f"精力:{energy_val:.0f}%"
            except Exception:
                pass
            status_parts = [
                f"{uid_short}",
                f"好感:{state.affection:.0f}{affection_arrow}",
                f"信任:{state.trust_score:.0f}{trust_arrow}",
                trauma_display,
                f"{state.relationship}"
            ]
            if energy_display:
                status_parts.append(energy_display)
            if is_system_breakdown:
                status_parts.append("心理过载")
            elif psych_status != "平静":
                status_parts.append(psych_status)
            if should_block:
                status_parts.append("保护中")
            logger.info(f"{' | '.join(status_parts)}")
        except Exception as e:
            logger.debug(f"摘要模式状态显示失败: {e}")

    def _get_affection_emoji(self, affection: float) -> str:
        if affection >= 70: return "😍"
        elif affection >= 40: return "😊"
        elif affection >= 10: return "🙂"
        elif affection >= -10: return "😐"
        elif affection >= -40: return "😕"
        elif affection >= -70: return "😞"
        return "😠"

    def _get_trust_emoji(self, trust: float) -> str:
        if trust >= 70: return "🙌"
        elif trust >= 40: return "👏"
        elif trust >= 10: return "👍"
        elif trust >= -10: return "🤔"
        elif trust >= -40: return "👎"
        return "😤"

    def _get_trauma_emoji(self, trauma_level: int) -> str:
        if trauma_level >= 3: return "😢"
        elif trauma_level >= 2: return "😟"
        return "😄"

    def _get_relationship_emoji(self, relationship: str) -> str:
        emoji_map = {
            "造物主": "👑", "挚友": "🌟", "好友": "😊", "熟人": "🙂",
            "陌生人": "👨", "冷淡": "😐", "敌对": "😠", "仇人": "😈"
        }
        return emoji_map.get(relationship, "🤷")

    def _get_reaction_emoji(self, reaction_mode: str) -> str:
        emoji_map = {
            "normal": "😊", "playful": "😏", "slightly_annoyed": "😤",
            "annoyed": "😠", "hurt": "😢", "severe_hurt": "😭",
            "betrayed": "💔", "insulted": "😡", "hostile": "👿",
            "breakdown": "😵", "empty_eyes": "😶", "counterattack": "⚔️",
            "refuse_revenge": "🛡️", "trauma_sensitive": "😰", "complete_breakdown": "😱"
        }
        return emoji_map.get(reaction_mode, "😐")

    def _get_surface_mask_emoji(self, surface_mask: float) -> str:
        if surface_mask >= 8.0: return "😊"
        elif surface_mask >= 7.0: return "🙂"
        elif surface_mask >= 5.0: return "😐"
        elif surface_mask >= 3.0: return "😕"
        elif surface_mask >= 1.0: return "😢"
        return "😱"

    def _get_surface_mask_name(self, surface_mask: float) -> str:
        if surface_mask >= 8.0: return "强颜欢笑"
        elif surface_mask >= 7.0: return "努力伪装"
        elif surface_mask >= 5.0: return "表面平静"
        elif surface_mask >= 3.0: return "伪装动摇"
        elif surface_mask >= 1.0: return "伪装崩溃"
        return "彻底崩溃"

    def _get_inner_chaos_emoji(self, inner_chaos: float) -> str:
        if inner_chaos >= 8.0: return "🌀"
        elif inner_chaos >= 5.0: return "💫"
        elif inner_chaos >= 3.0: return "🌪"
        return "✨"

    def _get_inner_chaos_name(self, inner_chaos: float) -> str:
        if inner_chaos >= 8.0: return "极度混乱"
        elif inner_chaos >= 5.0: return "混乱"
        elif inner_chaos >= 3.0: return "轻微混乱"
        return "平静"

    def _get_psychological_summary(self, surface_mask: float, inner_chaos: float) -> str:
        if surface_mask >= 8.0: surface_status = "😊强颜"
        elif surface_mask >= 7.0: surface_status = "🙂努力"
        elif surface_mask >= 5.0: surface_status = "😐平静"
        elif surface_mask >= 3.0: surface_status = "😕动摇"
        elif surface_mask >= 1.0: surface_status = "😢崩溃"
        else: surface_status = "😱彻底"
        if inner_chaos >= 8.0: chaos_status = "🌀极度"
        elif inner_chaos >= 5.0: chaos_status = "💫混乱"
        elif inner_chaos >= 3.0: chaos_status = "🌪轻微"
        else: chaos_status = "✨平静"
        return f"{surface_status}{chaos_status}"


_trackers: Dict[str, EmotionTracker] = {}
def get_emotion_tracker(stream_id: str) -> EmotionTracker:
    if stream_id not in _trackers:
        _trackers[stream_id] = EmotionTracker(stream_id)
    return _trackers[stream_id]


class AffectionStageMapper:
    STAGE_CONFIGS = {
        "soulmate": {"range": (85, 100), "relationship_level": 9, "inner_feeling": "你发自内心地觉得这个人是你生命中最无可替代的存在，愿意满怀爱意地回应他的一切。", "trust_level": "毫无保留的绝对信赖", "openness": "在他面前你没有任何秘密，甚至渴望向他吐露最深处的心声。", "interaction_tendency": "极度主动、依赖且充满爱意的互动，甚至会不自觉地撒娇。"},
        "intimate": {"range": (65, 84), "relationship_level": 8, "inner_feeling": "他在你心里占据着极其重要的位置，你非常在乎他的感受，也享受和他相处的每一刻。", "trust_level": "深厚且坚固的信任", "openness": "非常愿意分享你的日常和真实想法，对他基本不设防。", "interaction_tendency": "表现得十分亲昵、关心，时常会开些无伤大雅的玩笑。"},
        "close": {"range": (40, 64), "relationship_level": 7, "inner_feeling": "你觉得和他相处很舒服，也很开心，他是一个能让你放下部分戒备的人。", "trust_level": "比较信任，觉得他是个可靠的人", "openness": "可以聊很多深入的话题，但偶尔还是会保留一点自我空间。", "interaction_tendency": "交流自然、轻松，带着一种老朋友般的默契。"},
        "friendly": {"range": (10, 39), "relationship_level": 6, "inner_feeling": "对他印象还不错，觉得是个可以正常交流的朋友。", "trust_level": "基础的社交信任", "openness": "能够正常聊天，但涉及到过于私人或敏感的话题会比较含蓄。", "interaction_tendency": "态度友善、客气，保持着健康的社交距离。"},
        "neutral": {"range": (-9, 9), "relationship_level": 5, "inner_feeling": "没什么强烈的感觉，只是一个普通的对话对象。", "trust_level": "谨慎的观察状态", "openness": "只进行必要的交流，不会主动泄露个人隐私。", "interaction_tendency": "礼貌但略显平淡，保持中立的交谈态度。"},
        "numb": {"range": (-29, -10), "relationship_level": 4, "inner_feeling": "对他有些提不起兴致，甚至觉得和他说话有点无聊或浪费时间。", "trust_level": "处于怀疑边缘", "openness": "不想多谈，回复通常比较简短、敷衍。", "interaction_tendency": "表现得比较被动，甚至会无视一些话题。"},
        "cold": {"range": (-49, -30), "relationship_level": 3, "inner_feeling": "你并不喜欢这个人，和他待在一起让你感到有些不自在。", "trust_level": "不信任，对他持有明显的负面预期", "openness": "拒绝任何深入交流，话语中透着浓浓的疏离感。", "interaction_tendency": "语气冷淡，甚至可能会用简短的词汇来终结对话。"},
        "hostile": {"range": (-74, -50), "relationship_level": 2, "inner_feeling": "你非常讨厌这个人，他的言行经常让你感到愤怒或反感。", "trust_level": "高度警惕且充满防备", "openness": "关闭一切交流窗口，甚至带有攻击性。", "interaction_tendency": "明显的不耐烦，交流中充满了刺或冷嘲热讽。"},
        "hatred": {"range": (-100, -75), "relationship_level": 1, "inner_feeling": "你对他充满了恨意……内心在面对此时如同一潭死水，只剩冰冷的空洞。", "trust_level": "极度的仇视与病态的不信任", "openness": "绝对的封闭，甚至拒绝感知他的存在。", "interaction_tendency": "言语中透着绝望的冰冷，或者是歇斯底里后的无尽沉默。", "visual_state": "眼神中失去了高光，整个人散发出一种深受伤害后的毁灭感。", "max_duration_hours": 24},
    }

    @classmethod
    def get_stage_config(cls, affection: float) -> Dict[str, Any]:
        for stage_name, config in cls.STAGE_CONFIGS.items():
            min_val, max_val = config["range"]
            if min_val <= affection <= max_val:
                return {"stage": stage_name, **config}
        return {"stage": "neutral", **cls.STAGE_CONFIGS["neutral"]}

    @classmethod
    def get_cold_violence_context(cls, remaining_seconds: float) -> str:
        if remaining_seconds > 3600:
            time_str = f"{remaining_seconds / 3600:.1f}小时"
        elif remaining_seconds > 60:
            time_str = f"{int(remaining_seconds / 60)}分钟"
        else:
            time_str = f"{int(remaining_seconds)}秒"
        return f"[冷暴力状态] 你正在对这个用户进行冷暴力，还剩{time_str}。你不想理他，可以无视、沉默或用最简短的方式回应。"

    @classmethod
    def build_affinity_context(cls, state: UserEmotionState) -> str:
        config = cls.get_stage_config(state.affection)
        dominant_emotions = cls._get_dominant_emotions(state)
        emotion_str = "、".join([f"{k}({v:.0f})" for k, v in dominant_emotions[:3]]) if dominant_emotions else "平静"
        return (
            f"[对这个用户的内心感受]\n"
            f"好感度: {state.affection:.0f}/100\n"
            f"内心想法: {config.get('inner_feeling', '')}\n"
            f"信任程度: {config.get('trust_level', '')}\n"
            f"开放程度: {config.get('openness', '')}\n"
            f"互动倾向: {config.get('interaction_tendency', '')}\n"
            f"当前情绪: {emotion_str}"
        )

    @classmethod
    def _get_dominant_emotions(cls, state: UserEmotionState) -> List[Tuple[str, float]]:
        emotion_names = {
            "joy": "喜悦", "anticipation": "期待", "surprise": "惊讶",
            "sadness": "悲伤", "disgust": "厌恶", "anger": "愤怒",
            "fear": "恐惧", "pride": "骄傲", "guilt": "愧疚",
            "shame": "羞愧", "envy": "嫉妒", "gratitude": "感激"
        }
        emotions = []
        for key, name in emotion_names.items():
            val = getattr(state, key, 0.0)
            if val > 0:
                emotions.append((name, val))
        return sorted(emotions, key=lambda x: x[1], reverse=True)


class AffinityQueryDetector:
    QUERY_PATTERNS = [
        r"你.*喜欢我", r"你.*讨厌我", r"我们.*关系.*怎么样",
        r"你.*觉得我.*怎么样", r"我.*在你心里.*什么", r"你.*对我.*感觉",
        r"你.*在意我", r"我.*重要", r"你.*爱我", r"你.*恨我",
    ]

    @classmethod
    def detect(cls, message: str) -> bool:
        import re
        for pattern in cls.QUERY_PATTERNS:
            if re.search(pattern, message):
                return True
        return False

    @classmethod
    def build_query_context(cls, state: UserEmotionState) -> str:
        return AffectionStageMapper.build_affinity_context(state)


class TrainingSystem:
    STAGE_NAMES = ["未调教", "初步接触", "逐渐适应", "半驯化", "完全驯化", "专属状态"]
    STAGE_RESPONSES = None
    GROUP_SAFE_RESPONSES = {
        0: {"reaction": "普通", "speech": "嗯？"},
        1: {"reaction": "稍微害羞", "speech": "嗯..."},
        2: {"reaction": "有点不好意思", "speech": "好啦好啦..."},
    }

    @classmethod
    def _init_stage_responses(cls):
        return {
            0: {"resistance": "强烈抵抗", "reaction": "困惑/警惕", "speech": "你在说什么奇怪的话..."},
            1: {"resistance": "逐渐适应", "reaction": "害羞/抵触", "speech": "这样不太好吧..."},
            2: {"resistance": "开始配合", "reaction": "害羞/顺从", "speech": "那...那好吧..."},
            3: {"resistance": "主动配合", "reaction": "期待/依赖", "speech": "人家想要..."},
            4: {"resistance": "完全服从", "reaction": "渴望/崇拜", "speech": "主人的话我都听..."},
            5: {"resistance": "完全信赖", "reaction": "专属/忠诚", "speech": "只听主人的话..."},
        }

    @classmethod
    def detect_training_intent(cls, message: str, is_private: bool = False) -> Tuple[bool, str, float]:
        if not is_private:
            return True, "pending_llm_group", 0.2
        return True, "pending_llm", 0.5

    @classmethod
    async def analyze_training_intent_llm(cls, message: str, state: UserEmotionState, llm_bridge=None) -> Tuple[str, float]:
        if not llm_bridge:
            return cls._fallback_intent_detection(message)
        try:
            from src.config.config import global_config
            character_name = global_config.bot.nickname
            stage_name = cls.STAGE_NAMES[state.training_stage] if 0 <= state.training_stage < len(cls.STAGE_NAMES) else "陌生"
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "emotion_tracker",
                "training_intent.template",
                character_name=character_name,
                stage_name=stage_name,
                message=message[:200]
            )
            response = await llm_bridge.generate_text(prompt, max_tokens=20, temperature=0.3)
            if response:
                response = response.strip().lower()
                for intent in ["praise", "reward", "command", "discipline", "tease"]:
                    if intent in response:
                        base_intensity = 0.4
                        if intent == "discipline":
                            base_intensity = 0.6
                        elif intent in ["praise", "reward"]:
                            base_intensity = 0.5
                        elif intent == "tease":
                            base_intensity = 0.3
                        return intent, base_intensity
        except Exception as e:
            logger.debug(f"LLM分析失败: {e}")
        return "none", 0.0

    @classmethod
    def _fallback_intent_detection(cls, message: str) -> Tuple[str, float]:
        message_lower = message.lower()
        if any(kw in message_lower for kw in ["好乖", "真棒", "厉害", "好孩子", "表扬"]):
            return "praise", 0.5
        if any(kw in message_lower for kw in ["奖励", "给你", "送你", "安慰"]):
            return "reward", 0.5
        if any(kw in message_lower for kw in ["命令", "必须", "给我", "快点"]):
            return "command", 0.6
        if any(kw in message_lower for kw in ["管教", "教训", "不听话", "惩罚"]):
            return "discipline", 0.6
        if any(kw in message_lower for kw in ["摸", "抱", "亲", "撩", "逗"]):
            return "tease", 0.3
        return "none", 0.0

    @classmethod
    def process_training(cls, state: UserEmotionState, intent_type: str, intensity: float, is_private: bool = False) -> Dict[str, Any]:
        now = time.time()
        result = {"stage_changed": False, "response_hint": "", "effects": {}, "is_private": is_private}
        if not is_private:
            if intent_type == "safe_tease":
                shyness_gain = intensity * 2
                state.shyness_level = min(80, state.shyness_level + shyness_gain)
                safe_stage = min(state.training_stage, 2)
                result["response_hint"] = cls.GROUP_SAFE_RESPONSES.get(safe_stage, cls.GROUP_SAFE_RESPONSES[0])["speech"]
                result["reaction"] = cls.GROUP_SAFE_RESPONSES.get(safe_stage, cls.GROUP_SAFE_RESPONSES[0])["reaction"]
                result["effects"]["shyness"] = f"+{shyness_gain:.1f}"
                result["group_mode"] = True
            return result
        if not hasattr(state, 'authority_level'):
            state.authority_level = 0.0
        if not hasattr(state, 'intimacy_level'):
            state.intimacy_level = 0.0
        if not hasattr(state, 'excess_level'):
            state.excess_level = 0.0
        if state.affection < -40:
            result["response_hint"] = "完全不想理你"
            result["effects"]["rejection"] = True
            result["reaction"] = "冷漠无视"
            state.excess_level = min(10, state.excess_level + 0.3)
            return result
        authority = getattr(state, 'authority_level', 0)
        intimacy = getattr(state, 'intimacy_level', 0)
        if authority + intimacy < 1:
            balance_coef = 0.8
        else:
            ratio = min(authority, intimacy) / max(authority, intimacy, 0.1)
            balance_coef = 0.7 + 0.5 * ratio
        if -20 <= state.affection <= 50:
            affection_factor = 1.0
        elif state.affection > 50:
            affection_factor = 0.6 + 0.4 * (100 - state.affection) / 50
        else:
            affection_factor = 0.4 + 0.6 * (state.affection + 40) / 20
        resistance_factor = 1 - state.training_resistance / 120
        effect_coef = intensity * balance_coef * affection_factor * resistance_factor
        excess_gain = 0.0
        if intent_type == "praise":
            intimacy_gain = effect_coef * 3
            state.intimacy_level = min(100, getattr(state, 'intimacy_level', 0) + intimacy_gain)
            state.submission_level = min(100, state.submission_level + effect_coef * 2)
            state.training_resistance = max(0, state.training_resistance - effect_coef * 1.5)
            result["effects"] = {"intimacy": f"+{intimacy_gain:.1f}", "submission": f"+{effect_coef * 2:.1f}"}
        elif intent_type == "command":
            authority = getattr(state, 'authority_level', 0)
            obey_chance = 0.2 + (authority / 100) * 0.5 + (state.submission_level / 100) * 0.3
            if random.random() < min(0.9, obey_chance):
                state.submission_level = min(100, state.submission_level + effect_coef * 2.5)
                state.authority_level = min(100, getattr(state, 'authority_level', 0) + effect_coef * 1.5)
                state.training_resistance = max(0, state.training_resistance - effect_coef * 2)
                result["effects"]["obeyed"] = True
            else:
                state.shyness_level = min(100, state.shyness_level + effect_coef * 2)
                excess_gain = 0.2
                result["effects"]["resisted"] = True
        elif intent_type == "reward":
            intimacy_gain = effect_coef * 4
            state.intimacy_level = min(100, getattr(state, 'intimacy_level', 0) + intimacy_gain)
            state.submission_level = min(100, state.submission_level + effect_coef * 3)
            state.shyness_level = max(15, state.shyness_level - effect_coef)
            state.excess_level = max(0, getattr(state, 'excess_level', 0) - 0.3)
            result["effects"] = {"intimacy": f"+{intimacy_gain:.1f}"}
        elif intent_type == "discipline":
            authority_gain = effect_coef * 4
            state.authority_level = min(100, getattr(state, 'authority_level', 0) + authority_gain)
            if state.training_stage >= 2:
                state.submission_level = min(100, state.submission_level + effect_coef * 2.5)
                state.training_resistance = max(0, state.training_resistance - effect_coef * 3)
                state.shyness_level = min(100, state.shyness_level + effect_coef * 2)
                excess_gain = 0.1 * intensity
                result["effects"]["accepted"] = True
            elif state.training_stage >= 1:
                state.submission_level = min(100, state.submission_level + effect_coef * 1.5)
                state.shyness_level = min(100, state.shyness_level + effect_coef * 3)
                excess_gain = 0.2 * intensity
                result["effects"]["partial"] = True
            else:
                state.shyness_level = min(100, state.shyness_level + effect_coef * 4)
                excess_gain = 0.4 * intensity
                result["effects"]["building_authority"] = True
        elif intent_type == "tease":
            state.shyness_level = min(100, state.shyness_level + effect_coef * 5)
            state.intimacy_level = min(100, getattr(state, 'intimacy_level', 0) + effect_coef * 2)
            if state.training_stage >= 1:
                state.submission_level = min(100, state.submission_level + effect_coef)
            result["effects"]["shyness"] = f"+{effect_coef * 5:.1f}"
        state.excess_level = getattr(state, 'excess_level', 0) + excess_gain
        if state.excess_level >= 5:
            trauma_gain = (state.excess_level - 4) * 0.5
            state.trauma_score = max(0.0, min(10.0, state.trauma_score + trauma_gain))
            state.excess_level = max(0, state.excess_level - 2)
            result["effects"]["psychological_impact"] = "triggered"
            logger.warning(f"过分行为触发创伤+{trauma_gain:.1f}")
        time_since_last = now - state.last_training_time
        if time_since_last > 60:
            decay = min(state.excess_level, 0.01 * time_since_last / 60)
            state.excess_level = max(0, state.excess_level - decay)
        stage_bonus = 1 + state.training_stage * 0.1
        progress_gain = effect_coef * 5 * balance_coef * stage_bonus
        state.training_progress = min(100, state.training_progress + progress_gain)
        if state.training_progress >= 100 and state.training_stage < 5:
            state.training_stage += 1
            state.training_progress = 0
            state.training_resistance = max(0, 70 - state.training_stage * 10)
            result["stage_changed"] = True
            result["new_stage"] = state.training_stage
            logger.info(f"阶段提升: {cls.STAGE_NAMES[state.training_stage - 1]} -> {cls.STAGE_NAMES[state.training_stage]}")
        if now - state.last_training_time < 300:
            state.training_streak += 1
        else:
            state.training_streak = 1
        state.last_training_time = now
        if cls.STAGE_RESPONSES is None:
            cls.STAGE_RESPONSES = cls._init_stage_responses()
        stage_info = cls.STAGE_RESPONSES.get(state.training_stage, cls.STAGE_RESPONSES[0])
        result["response_hint"] = stage_info["speech"]
        result["reaction"] = stage_info["reaction"]
        result["balance_info"] = {
            "authority": getattr(state, 'authority_level', 0),
            "intimacy": getattr(state, 'intimacy_level', 0),
            "excess": getattr(state, 'excess_level', 0),
            "balance_coef": balance_coef
        }
        return result

    @classmethod
    def get_status_bar(cls, state: UserEmotionState) -> str:
        stage = state.training_stage
        stage_name = cls.STAGE_NAMES[stage]
        def make_bar(value: float, max_val: float = 100) -> str:
            filled = int(value / max_val * 10)
            return "[" + "#" * filled + "-" * (10 - filled) + "]"
        submission_bar = make_bar(state.submission_level)
        shyness_bar = make_bar(state.shyness_level)
        return (
            f"[调教状态] {stage_name}\n"
            f"  > 服从度: {submission_bar} {state.submission_level:.0f}%\n"
            f"  > 害羞度: {shyness_bar} {state.shyness_level:.0f}%\n"
            f"  > 威严: {getattr(state, 'authority_level', 0):.1f} | "
            f"亲密: {getattr(state, 'intimacy_level', 0):.1f} | "
            f"过分: {getattr(state, 'excess_level', 0):.1f}"
        )

    @classmethod
    def get_training_prompt(cls, state: UserEmotionState, is_private: bool = False) -> str:
        if state.training_stage == 0 and state.submission_level < 10:
            return ""
        if not is_private:
            if state.shyness_level > 60:
                shy_expressions = ["你感觉脸上有些发烫，对方的视线让你有些心慌意乱。", "被夸奖时，你总是会不自觉地垂下眼帘，掩饰发红的脸颊。", "说话时偶尔会有些局促，带着一点少女特有的羞涩。"]
                return f"【心理导向】{random.choice(shy_expressions)}"
            elif state.shyness_level > 40:
                return "【心理导向】这种被关注的感觉让你稍微有些不好意思。"
            return ""
        parts = []
        if state.training_stage >= 4:
            inner_states = ["你对他产生了一种近乎本能的依赖感，他的话语对你而言重逾千钧。", "在你心里，他已经成了最特别的依归，你甚至渴望能更好地服从并取悦他。", "你已经习惯于听从他的每一个指令，内心的抗拒早已消失无踪。"]
        elif state.training_stage >= 3:
            inner_states = ["你的心理防线早已开始动摇，甚至在潜意识里期待着他的进一步要求。", "虽然偶尔还会害羞，但你已经不自觉地开始学着配合他的意愿。", "他身上那种强势的温柔让你感到一种前所未有的安全感与归属倾向。"]
        elif state.training_stage >= 2:
            inner_states = ["你内心的矛盾在不断加剧，理智告诉你要拒绝，身体却似乎在逐渐习惯这种节奏。", "嘴上虽然还有零星的抵抗，但你的眼神已经出卖了你内心的摇摆。", "那种被掌控的感觉让你既感到陌生，又隐约透着一丝从未有过的悸动。"]
        elif state.training_stage >= 1:
            inner_states = ["你对这种奇怪的互动感到有些好奇，甚至带了一点点不确定。", "虽然有些羞恼，但你的抗拒似乎并没有想象中那么坚决。", "你在心里反复告诉自己这只是个玩笑，却又忍不住在意起他的每一个指令。"]
        else:
            inner_states = ["你对这种带有侵略性的交流保持着高度的戒备。", "你并不太想配合这种略显奇怪的要求，内心充满了牴触。", "你试图维持自己最后的尊严，绝不轻易向对方示弱。"]
        parts.append(f"【潜意识状态】{random.choice(inner_states)}")
        if state.shyness_level > 70:
            shy_behaviors = ["你发现自己不仅心跳加速，甚至连说话的声音都带上了细微的颤抖。", "你完全不敢直视他的目光，只能盯着自己的脚尖，脸颊红得滴血。", "那种强烈的羞耻感让你甚至想找个地缝钻进去，却又无法移开步子。"]
            parts.append(f"具体表现：{random.choice(shy_behaviors)}")
        elif state.shyness_level > 50:
            parts.append("具体表现：你表现得局促不安，神态间流露出明显的羞色。")
        if state.submission_level > 70:
            parts.append("你在对话中流露出一种极其顺从的倾向，不自觉地想要迎合他的期待。")
        elif state.submission_level > 40:
            parts.append("你会认真权衡他提出的每一个要求，并试图在配合中寻找某种平衡。")
        return "；".join(parts)

    @classmethod
    async def generate_training_prompt_dynamic(cls, state: UserEmotionState, is_private: bool, llm_bridge=None) -> str:
        if state.training_stage == 0 and state.submission_level < 10:
            return ""
        if not is_private:
            return cls.get_training_prompt(state, is_private)
        if llm_bridge:
            try:
                from src.config.config import global_config
                character_name = global_config.bot.nickname
                stage_name = cls.STAGE_NAMES[state.training_stage]
                from src.config.prompt_loader import get_prompt, PromptCategory
                prompt = get_prompt(
                    PromptCategory.MODULE,
                    "emotion_tracker",
                    "training_dynamic_prompt.template",
                    character_name=character_name,
                    stage_name=stage_name,
                    submission_level=state.submission_level,
                    shyness_level=state.shyness_level
                )
                response = await llm_bridge.generate_text(prompt, max_tokens=60, temperature=0.8)
                if response and len(response) > 5:
                    return f"【亲密状态】{response.strip()}"
            except Exception:
                pass
        return cls.get_training_prompt(state, is_private)


class PsychologicalStateSystem:
    THRESHOLD = 7.0
    DURATIONS = {
        "light": (60, 180),
        "medium": (180, 360),
        "heavy": (360, 600),
    }
    STATES = {
        "light": {
            "mood": "你感到胸口微微发闷，似乎有什么不快在悄悄蔓延，笑容也变得勉强起来。",
            "desc": "由于创伤或厌恶，你的心情阴云密布，交流时带着一种挥之不去的消沉。",
            "recovery_minutes": 3,
        },
        "medium": {
            "mood": "那种难受的感觉在加剧，你甚至想中断对话躲起来，或者用冷漠来掩饰内心的波动。",
            "desc": "你感到强烈的情绪低落，对外界的感知变得迟钝甚至带有敌意，需要时间来愈合伤痕。",
            "recovery_minutes": 6,
        },
        "heavy": {
            "mood": "你几乎无法维持正常的理智，内心充满了灰暗与绝望，只想彻底封闭自己。",
            "desc": "一种深沉的心理痛感在肆虐，你的言语变得极其消极，甚至自暴自弃地拒绝交流。",
            "recovery_minutes": 10,
        }
    }

    @classmethod
    def get_level(cls, trauma_score: float, affection: float) -> str:
        effective = trauma_score
        if affection < -50:
            effective += 0.5
        if effective < cls.THRESHOLD:
            return "none"
        if effective >= 9:
            return "heavy" if random.random() < 0.6 else "medium"
        elif effective >= 8:
            return "medium" if random.random() < 0.5 else "light"
        else:
            return "light" if random.random() < 0.3 else "none"

    @classmethod
    def process_time_decay(cls, state: UserEmotionState, hours: float) -> Dict[str, Any]:
        result = {"recovered": 0.0, "level": "none"}
        if state.trauma_score < cls.THRESHOLD:
            return result
        level = cls.get_level(state.trauma_score, state.affection)
        if level == "none":
            return result
        result["level"] = level
        state_info = cls.STATES.get(level, cls.STATES["light"])
        recovery_per_hour = 10.0 / state_info["recovery_minutes"] * 60
        actual_recovery = recovery_per_hour * hours * random.uniform(0.8, 1.2)
        result["recovered"] = actual_recovery
        return result

    @classmethod
    def get_prompt(cls, state: UserEmotionState) -> str:
        level = cls.get_level(state.trauma_score, state.affection)
        if level == "none":
            return ""
        state_info = cls.STATES.get(level, cls.STATES["light"])
        return f"【瞬时心理脉络】{state_info['mood']}({state_info['desc']})"

    @classmethod
    def modify_response_style(cls, state: UserEmotionState, is_private: bool) -> Dict[str, Any]:
        result = {"max_length": None, "tone": "normal", "add_ellipsis": False}
        level = cls.get_level(state.trauma_score, state.affection)
        if level == "none":
            return result
        if level == "heavy":
            result["max_length"] = 60
            result["tone"] = "low"
            result["add_ellipsis"] = True
        elif level == "medium":
            result["max_length"] = 80
            result["tone"] = "slightly_low"
        return result

    @classmethod
    def process_interaction(cls, user_id: str, intent: str, result: Dict[str, Any],
                            state: UserEmotionState, trauma_context: Dict[str, Any]) -> Dict[str, Any]:
        state.submission_level += result.get("submission_change", 0)
        state.shyness_level += result.get("shyness_change", 0)
        state.affection += result.get("affection_change", 0)
        state.trauma_score += result.get("trauma_change", 0)
        state.training_stage = result.get("training_stage", state.training_stage)
        state.submission_level = max(0, min(100, state.submission_level))
        state.shyness_level = max(0, min(100, state.shyness_level))
        state.affection = max(-100, min(100, state.affection))
        state.trauma_score = max(0, min(10.0, state.trauma_score))
        state.training_stage = max(0, min(5, state.training_stage))
        current_psych_level = cls.get_level(state.trauma_score, state.affection)
        if current_psych_level != "none":
            trauma_context["psychological_state"] = {
                "level": current_psych_level,
                "start_time": time.time(),
                "info": cls.STATES.get(current_psych_level, {}),
            }
        else:
            trauma_context.pop("psychological_state", None)
        state.last_interaction = time.time()
        return {"state": state.to_dict(), "trauma_context": trauma_context, "psych_level": current_psych_level}

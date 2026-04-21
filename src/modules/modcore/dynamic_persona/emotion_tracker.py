import time
import math
import random
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from src.common.logger import get_logger

logger = get_logger("情绪跟踪")


class EmotionDimension(Enum):
    JOY = "joy"
    ANTICIPATION = "anticipation"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    FEAR = "fear"
    TRUST = "trust"


class RelationshipTier(Enum):
    SOULMATE = ("挚友", 80)
    CLOSE_FRIEND = ("好友", 60)
    ACQUAINTANCE = ("熟人", 40)
    KNOWN = ("认识", 20)
    STRANGER = ("陌生人", 0)
    DISLIKE = ("不太喜欢", -20)
    ENEMY = ("讨厌", -50)
    HOSTILE = ("敌意", -80)


def _get_relation_thresholds() -> List[Tuple[int, str]]:
    try:
        from src.modules.modcore.dynamic_persona.config_loader import (
            get_relation_thresholds,
        )

        return get_relation_thresholds()
    except Exception:
        return [
            (80, "挚友"),
            (60, "好友"),
            (40, "熟人"),
            (20, "认识"),
            (0, "陌生人"),
            (-20, "不太喜欢"),
            (-50, "讨厌"),
            (-80, "敌意"),
        ]


_RELATION_THRESHOLDS = _get_relation_thresholds()


@dataclass
class UserEmotionState:
    user_id: str
    stream_id: str = ""
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
    trust: float = 0.0
    stamina: float = 100.0
    mood: str = "平静"
    mental_fatigue: float = 0.0
    negative_emotion_aggregate: float = 0.0
    inner_chaos: float = 0.0
    surface_mask: float = 0.0
    trust_accumulation: float = 0.0
    distrust_accumulation: float = 0.0
    trauma_accumulation: float = 0.0
    interaction_coefficients: Dict[str, float] = field(
        default_factory=lambda: {
            "authority_intimacy_ratio": 0.5,
            "excess_trauma_multiplier": 1.0,
            "fatigue_response_factor": 0.8,
            "pressure_resistance_factor": 0.6,
        }
    )

    @property
    def trust_value(self) -> float:
        return float(self.trust_score or 0.0)

    @trust_value.setter
    def trust_value(self, value: float) -> None:
        canonical = float(value or 0.0)
        self.trust_score = canonical
        self.trust = canonical

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        canonical_trust = float(self.trust_score or 0.0)
        data["trust_value"] = canonical_trust
        data["trust_score"] = canonical_trust
        data["trust"] = canonical_trust
        return data

    def get_dominant_emotion(self) -> Tuple[str, float]:
        emotions = {
            "joy": self.joy,
            "anticipation": self.anticipation,
            "surprise": self.surprise,
            "sadness": self.sadness,
            "disgust": self.disgust,
            "anger": self.anger,
            "fear": self.fear,
            "pride": self.pride,
            "guilt": self.guilt,
            "shame": self.shame,
            "envy": self.envy,
            "gratitude": self.gratitude,
        }
        if not emotions:
            return "neutral", 0.0
        dominant = max(emotions.items(), key=lambda x: abs(x[1]))
        return dominant[0], dominant[1]

    def get_emotion_vector(self) -> Dict[str, float]:
        return {
            "pleasure": (self.joy + self.gratitude - self.sadness - self.anger)
            / 4.0,
            "arousal": (self.anticipation + self.surprise + self.fear) / 3.0,
            "dominance": (self.pride - self.shame - self.guilt) / 3.0,
        }

    def compute_vulnerability_score(self) -> float:
        return (
            (self.trauma_accumulation / 10.0) * 0.3
            + (self.psychological_pressure / 100.0) * 0.3
            + (self.distrust_accumulation / 100.0) * 0.4
        )

    def is_vulnerable(self) -> bool:
        return self.compute_vulnerability_score() > 0.6

    def needs_protection(self) -> bool:
        return self.trauma_accumulation > 7 or self.psychological_pressure > 80

    def compute_negative_aggregate(self) -> float:
        """加权聚合六维负向情绪为单一指标（0~100）"""
        raw = (
            max(0.0, self.anger) * 0.25
            + max(0.0, self.sadness) * 0.20
            + max(0.0, self.disgust) * 0.20
            + max(0.0, self.fear) * 0.15
            + max(0.0, self.shame) * 0.10
            + max(0.0, self.guilt) * 0.10
        )
        return round(max(0.0, min(100.0, raw)), 2)


class EmotionConfig:
    AFFECTION_MAX = 100.0
    AFFECTION_MIN = -100.0
    TRUST_MAX = 100.0
    TRUST_MIN = -100.0
    ANNOYANCE_MAX = 100.0
    ANNOYANCE_MIN = 0.0
    TRAUMA_MAX = 10.0
    TRAUMA_MIN = 0.0
    TRUST_THRESHOLD = 30.0
    BLOCK_THRESHOLD = 80.0
    BLOCK_DURATION_BASE = 60.0
    BLOCK_DURATION_MAX = 3600.0
    GLOBAL_RAGE_THRESHOLD = 150.0


class EmotionTracker:
    _global_irritation: float = 0.0
    _last_global_impact: float = 0.0
    _global_shield_active: bool = False
    _global_shield_until: float = 0.0
    _global_shield_reason: str = ""
    _global_lock = threading.Lock()

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
        self._check_time_sync()
        self._init_stream_emotion()

    def _init_database(self):
        try:
            from src.modules.modcore.dynamic_persona.emotion_database import (
                get_emotion_database,
            )

            self._db = get_emotion_database(self.stream_id)
        except Exception as exc:
            logger.debug(f"情绪数据库初始化跳过: {exc}")

    def _load_states(self):
        if self._db:
            try:
                states = self._db.load_all_states()
                for user_id, state_dict in states.items():
                    self._user_states[user_id] = self._dict_to_state(
                        state_dict
                    )
                logger.debug(f"加载 {len(self._user_states)} 个用户情绪状态")
            except Exception as exc:
                logger.warning(f"加载情绪状态失败: {exc}")

    def _dict_to_state(self, data: Dict) -> UserEmotionState:
        trust_value = data.get(
            "trust_value", data.get("trust_score", data.get("trust", 0.0))
        )
        return UserEmotionState(
            user_id=data.get("user_id", ""),
            stream_id=data.get("stream_id", self.stream_id),
            affection=data.get("affection", 0.0),
            impression=data.get("impression", "初次见面"),
            relationship=data.get("relationship", "陌生人"),
            annoyance=data.get("annoyance", 0.0),
            trust_score=trust_value,
            trauma_score=data.get("trauma_score", 0.0),
            trauma_history=data.get("trauma_history", []),
            trauma_last_recall=data.get("trauma_last_recall", 0.0),
            trauma_recall_count=data.get("trauma_recall_count", 0),
            baseline_affection=data.get("baseline_affection", 0.0),
            interaction_count=data.get("interaction_count", 0),
            last_interaction=data.get("last_interaction", 0.0),
            is_blocked=data.get("is_blocked", False),
            block_until=data.get("block_until", 0.0),
            block_reason=data.get("block_reason", ""),
            volatility=data.get("volatility", 1.0),
            has_trauma_mark=data.get("has_trauma_mark", False),
            trauma_mark_level=data.get("trauma_mark_level", 0),
            trauma_mark_time=data.get("trauma_mark_time", 0.0),
            milestones_reached=data.get("milestones_reached", []),
            highest_affection=data.get("highest_affection", 0.0),
            nickname=data.get("nickname", ""),
            last_relationship=data.get("last_relationship", "陌生人"),
            relationship_protected_until=data.get(
                "relationship_protected_until", 0.0
            ),
            created_at=data.get("created_at", time.time()),
            joy=data.get("joy", 0.0),
            anticipation=data.get("anticipation", 0.0),
            surprise=data.get("surprise", 0.0),
            sadness=data.get("sadness", 0.0),
            disgust=data.get("disgust", 0.0),
            anger=data.get("anger", 0.0),
            fear=data.get("fear", 0.0),
            pride=data.get("pride", 0.0),
            guilt=data.get("guilt", 0.0),
            shame=data.get("shame", 0.0),
            envy=data.get("envy", 0.0),
            gratitude=data.get("gratitude", 0.0),
            positive_interactions=data.get("positive_interactions", 0),
            negative_interactions=data.get("negative_interactions", 0),
            intimacy=data.get("intimacy", 0.0),
            attitude=data.get("attitude", "中立"),
            submission_level=data.get("submission_level", 0.0),
            shyness_level=data.get("shyness_level", 50.0),
            training_stage=data.get("training_stage", 0),
            training_progress=data.get("training_progress", 0.0),
            training_resistance=data.get("training_resistance", 100.0),
            trained_responses=data.get("trained_responses", []),
            last_training_time=data.get("last_training_time", 0.0),
            training_streak=data.get("training_streak", 0),
            authority_level=data.get("authority_level", 0.0),
            intimacy_level=data.get("intimacy_level", 0.0),
            excess_level=data.get("excess_level", 0.0),
            is_protected=data.get("is_protected", False),
            protect_until=data.get("protect_until", 0.0),
            protect_reason=data.get("protect_reason", ""),
            protect_count=data.get("protect_count", 0),
            affection_threshold_offset=data.get(
                "affection_threshold_offset", 0.0
            ),
            cumulative_negative_count=data.get("cumulative_negative_count", 0),
            cumulative_positive_count=data.get("cumulative_positive_count", 0),
            negative_behavior_streak=data.get("negative_behavior_streak", 0),
            positive_behavior_streak=data.get("positive_behavior_streak", 0),
            last_behavior_type=data.get("last_behavior_type", ""),
            psychological_pressure=data.get("psychological_pressure", 0.0),
            trust=trust_value,
            stamina=data.get("stamina", 100.0),
            surface_mask=data.get("surface_mask", 0.0),
            inner_chaos=data.get("inner_chaos", 0.0),
            mood=data.get("mood", "平静"),
            mental_fatigue=data.get("mental_fatigue", 0.0),
            negative_emotion_aggregate=data.get("negative_emotion_aggregate", 0.0),
            trust_accumulation=data.get("trust_accumulation", 0.0),
            distrust_accumulation=data.get("distrust_accumulation", 0.0),
            trauma_accumulation=data.get("trauma_accumulation", 0.0),
            interaction_coefficients=data.get(
                "interaction_coefficients",
                {
                    "authority_intimacy_ratio": 0.5,
                    "excess_trauma_multiplier": 1.0,
                    "fatigue_response_factor": 0.8,
                    "pressure_resistance_factor": 0.6,
                },
            ),
        )

    def _save_states(self):
        if self._db:
            try:
                for user_id, state in self._user_states.items():
                    self._db.save_user_state(user_id, state.to_dict())
            except Exception as exc:
                logger.warning(f"保存情绪状态失败: {exc}")

    def _save_user_state(self, user_id: str):
        if self._db and user_id in self._user_states:
            try:
                self._db.save_user_state(
                    user_id, self._user_states[user_id].to_dict()
                )
            except Exception as exc:
                logger.debug(f"保存用户状态失败: {exc}")

    def _check_time_sync(self):
        now = time.time()
        for user_id, state in self._user_states.items():
            if state.last_interaction > 0:
                gap = now - state.last_interaction
                if gap > 30 * 24 * 3600:
                    state.last_interaction = (
                        now
                        - (7 * 24 * 3600)
                        + (hash(user_id) % (7 * 24 * 3600))
                    )
                    if state.created_at > state.last_interaction:
                        state.created_at = state.last_interaction - 3600
                    if state.trauma_last_recall > state.last_interaction:
                        state.trauma_last_recall = state.last_interaction
                    if state.last_training_time > state.last_interaction:
                        state.last_training_time = state.last_interaction

    def _init_stream_emotion(self):
        try:
            from src.chat.heart_flow.emotion_stream import (
                get_channel_mood_tracker,
            )

            self._stream_emotion_manager = get_channel_mood_tracker()
            logger.debug(f"[{self.stream_id}] 流情绪管理器初始化成功")
        except Exception as exc:
            logger.debug(f"流情绪管理器初始化跳过: {exc}")

    @staticmethod
    def mask_id(uid: str) -> str:
        if not uid or uid in ("default", "guest"):
            return uid
        uid_str = str(uid)
        if len(uid_str) >= 10:
            return f"{uid_str[:4]}****{uid_str[-4:]}"
        elif len(uid_str) > 4:
            return f"{uid_str[:2]}**{uid_str[-2:]}"
        return uid_str

    @staticmethod
    def clamp_affection(value: float) -> float:
        return max(
            EmotionConfig.AFFECTION_MIN,
            min(EmotionConfig.AFFECTION_MAX, value),
        )

    @staticmethod
    def clamp_trust(value: float) -> float:
        return max(
            EmotionConfig.TRUST_MIN, min(EmotionConfig.TRUST_MAX, value)
        )

    @staticmethod
    def clamp_annoyance(value: float) -> float:
        return max(
            EmotionConfig.ANNOYANCE_MIN,
            min(EmotionConfig.ANNOYANCE_MAX, value),
        )

    @staticmethod
    def clamp_trauma(value: float) -> float:
        return max(
            EmotionConfig.TRAUMA_MIN, min(EmotionConfig.TRAUMA_MAX, value)
        )

    @staticmethod
    def clamp_all_values(state: UserEmotionState) -> None:
        state.affection = EmotionTracker.clamp_affection(state.affection)
        state.trust_score = EmotionTracker.clamp_trust(state.trust_score)
        state.trust = state.trust_score
        state.annoyance = EmotionTracker.clamp_annoyance(state.annoyance)
        state.trauma_score = EmotionTracker.clamp_trauma(state.trauma_score)
        state.submission_level = max(0, min(100, state.submission_level))
        state.shyness_level = max(0, min(100, state.shyness_level))
        state.training_resistance = max(0, min(100, state.training_resistance))
        state.authority_level = max(0, min(100, state.authority_level))
        state.intimacy_level = max(0, min(100, state.intimacy_level))
        state.excess_level = max(0, min(100, state.excess_level))
        state.psychological_pressure = max(
            0, min(100, state.psychological_pressure)
        )
        state.inner_chaos = max(0, min(10, state.inner_chaos))
        state.surface_mask = max(0, min(10, state.surface_mask))
        state.trust = max(-100, min(100, state.trust))
        state.stamina = max(0, min(100, state.stamina))
        state.mental_fatigue = max(0, min(100, state.mental_fatigue))
        state.negative_emotion_aggregate = max(0, min(100, state.negative_emotion_aggregate))
        state.trust_accumulation = max(0, min(100, state.trust_accumulation))
        state.distrust_accumulation = max(0, min(100, state.distrust_accumulation))
        state.trauma_accumulation = max(0, min(10, state.trauma_accumulation))

    def get_user_state(
        self, user_id: str, create_if_missing: bool = True
    ) -> Optional[UserEmotionState]:
        if user_id in self._user_states:
            state = self._user_states[user_id]
            now = time.time()
            hours = (
                (now - state.last_interaction) / 3600
                if state.last_interaction > 0
                else 0
            )
            if hours > 0:
                self._apply_time_decay(state, hours)
                if hours >= 0.5:
                    state.last_interaction = now
                    self._save_user_state(user_id)
            return state
        if create_if_missing:
            state = UserEmotionState(user_id=user_id, stream_id=self.stream_id)
            self._user_states[user_id] = state
            return state
        return None

    def _apply_time_decay(self, state: UserEmotionState, hours: float):
        if hours <= 0:
            return
        decay_factor = min(hours, 168.0)
        if state.annoyance > 0:
            if state.annoyance >= 80:
                _hourly_rate = 0.20
            elif state.annoyance >= 50:
                _hourly_rate = 0.15
            else:
                _hourly_rate = 0.10
            _effective = min(1.0, decay_factor * _hourly_rate)
            _flat_drop = decay_factor * 2.5
            state.annoyance = max(
                0,
                state.annoyance * (1.0 - _effective) - _flat_drop,
            )
        if state.trauma_score > 0:
            # 衰减公式：高创伤分段衰减更快，防止长期锁死在防御模式
            # score>=5: 每小时衰减约15%（约4.6小时半衰期）
            # score 3~5: 每小时衰减约10%（约6.9小时半衰期）
            # score<3: 每小时衰减约5%（约14小时半衰期）
            if state.trauma_score >= 5.0:
                hourly_decay = 0.15
            elif state.trauma_score >= 3.0:
                hourly_decay = 0.10
            else:
                hourly_decay = 0.05
            decay_rate = min(1.0, decay_factor * hourly_decay)
            state.trauma_score = max(
                0, state.trauma_score * (1.0 - decay_rate)
            )
        if state.affection < 0:
            recovery_rate = min(1.0, decay_factor / 8.0)
            state.affection = min(
                0, state.affection + abs(state.affection) * recovery_rate * 0.3
            )
        if state.psychological_pressure > 0:
            decay_rate = min(1.0, decay_factor / 6.0)
            state.psychological_pressure = max(
                0, state.psychological_pressure * (1.0 - decay_rate * 0.4)
            )
        if state.mental_fatigue > 0:
            decay_rate = min(1.0, decay_factor / 8.0)
            state.mental_fatigue = max(
                0, state.mental_fatigue * (1.0 - decay_rate * 0.3)
            )
        # 负向情绪维度自然衰减（6小时半衰期）
        neg_decay_rate = min(1.0, decay_factor / 6.0)
        for _dim_name in (
            "anger",
            "sadness",
            "disgust",
            "fear",
            "shame",
            "guilt",
        ):
            _dim_val = getattr(state, _dim_name, 0.0)
            if _dim_val > 0:
                setattr(
                    state,
                    _dim_name,
                    max(0.0, _dim_val * (1.0 - neg_decay_rate * 0.5)),
                )
        # 重算负面情绪聚合值
        state.negative_emotion_aggregate = state.compute_negative_aggregate()
        self.clamp_all_values(state)

    def update_affection(
        self, user_id: str, delta: float, reason: str = ""
    ) -> float:
        state = self.get_user_state(user_id, create_if_missing=True)
        old_val = state.affection
        state.affection = self.clamp_affection(
            state.affection + delta * state.volatility
        )
        if state.affection > state.highest_affection:
            state.highest_affection = state.affection
        state.last_interaction = time.time()
        state.interaction_count += 1
        if delta > 0:
            state.positive_interactions += 1
            state.positive_behavior_streak += 1
            state.negative_behavior_streak = 0
        elif delta < 0:
            state.negative_interactions += 1
            state.negative_behavior_streak += 1
            state.positive_behavior_streak = 0
        self._update_relationship(state)
        self.clamp_all_values(state)
        self._save_user_state(user_id)
        if abs(delta) > 2:
            mask_uid = self.mask_id(user_id)
            logger.info(
                f"好感变化 {mask_uid} | {
                    old_val:.1f} -> {
                    state.affection:.1f} ({
                    delta:+.1f}) | {reason}"
            )
        return state.affection

    def update_annoyance(
        self, user_id: str, delta: float, reason: str = ""
    ) -> float:
        state = self.get_user_state(user_id, create_if_missing=True)
        # 新用户保护：交互少于5次时正增量削减60%，避免误判快速累积
        effective_delta = delta
        if delta > 0 and state.interaction_count < 5:
            effective_delta = delta * 0.4
        state.annoyance = self.clamp_annoyance(state.annoyance + effective_delta)
        if (
            state.annoyance >= EmotionConfig.BLOCK_THRESHOLD
            and not state.is_blocked
        ):
            block_duration = min(
                EmotionConfig.BLOCK_DURATION_BASE
                * (state.annoyance / EmotionConfig.BLOCK_THRESHOLD),
                EmotionConfig.BLOCK_DURATION_MAX,
            )
            state.is_blocked = True
            state.block_until = time.time() + block_duration
            state.block_reason = reason or "烦躁度过高"
            mask_uid = self.mask_id(user_id)
            logger.warning(
                f"用户屏蔽 {mask_uid} | 烦躁度{
                    state.annoyance:.0f} | 持续{
                    block_duration:.0f}秒"
            )
        state.last_interaction = time.time()
        self.clamp_all_values(state)
        self._save_user_state(user_id)
        return state.annoyance

    def update_trauma(
        self, user_id: str, delta: float, trigger: str = ""
    ) -> float:
        state = self.get_user_state(user_id, create_if_missing=True)
        old_trauma = state.trauma_score
        state.trauma_score = self.clamp_trauma(state.trauma_score + delta)
        if delta > 0 and trigger:
            state.trauma_history.append(
                {"trigger": trigger, "score": delta, "time": time.time()}
            )
            if len(state.trauma_history) > 50:
                state.trauma_history = state.trauma_history[-30:]
            state.trauma_last_recall = time.time()
            state.trauma_recall_count += 1
        if state.trauma_score >= 3.0 and not state.has_trauma_mark:
            state.has_trauma_mark = True
            state.trauma_mark_level = 1
            state.trauma_mark_time = time.time()
        if delta > 0 and state.trauma_score >= 6.0 and old_trauma < 6.0:
            self.activate_global_shield(
                1800,
                f"用户{
                    self.mask_id(user_id)}创伤分数达到{
                    state.trauma_score:.1f}",
            )
            logger.warning(
                f"创伤保护激活: 用户{
                    self.mask_id(user_id)} 创伤{
                    state.trauma_score:.1f}"
            )
        self.clamp_all_values(state)
        self._save_user_state(user_id)
        return state.trauma_score

    def update_psychological_pressure(
        self, user_id: str, delta: float, reason: str = ""
    ) -> float:
        """更新用户心理压力值（0~100），>80 触发告警"""
        state = self.get_user_state(user_id, create_if_missing=True)
        old_val = state.psychological_pressure
        state.psychological_pressure = max(
            0.0, min(100.0, state.psychological_pressure + delta)
        )
        state.last_interaction = time.time()
        if state.psychological_pressure >= 80 and old_val < 80:
            mask_uid = self.mask_id(user_id)
            logger.warning(
                f"心理压力告警 {mask_uid} | {
                    state.psychological_pressure:.1f} | {reason}"
            )
        elif abs(delta) > 5:
            mask_uid = self.mask_id(user_id)
            logger.info(
                f"心理压力变化 {mask_uid} | {old_val:.1f} -> {state.psychological_pressure:.1f}"
                f" ({delta:+.1f}) | {reason}"
            )
        EmotionTracker.clamp_all_values(state)
        self._save_user_state(user_id)
        return state.psychological_pressure

    def update_emotion_dimension(
        self, user_id: str, dimension: EmotionDimension, delta: float
    ) -> float:
        state = self.get_user_state(user_id, create_if_missing=True)
        attr_name = dimension.value
        current = getattr(state, attr_name, 0.0)
        new_val = max(-100.0, min(100.0, current + delta))
        setattr(state, attr_name, new_val)
        _negative_dims = {"anger", "sadness", "disgust", "fear", "shame", "guilt"}
        if attr_name in _negative_dims:
            state.negative_emotion_aggregate = state.compute_negative_aggregate()
        EmotionTracker.clamp_all_values(state)
        self._save_user_state(user_id)
        return new_val

    def update_multi_dimensional_emotion(
        self, user_id: str, emotion_deltas: Dict[str, float]
    ) -> Dict[str, float]:
        state = self.get_user_state(user_id, create_if_missing=True)
        results = {}
        _negative_dims = {"anger", "sadness", "disgust", "fear", "shame", "guilt"}
        need_recompute_negative = False
        for emotion, delta in emotion_deltas.items():
            if hasattr(state, emotion):
                current = getattr(state, emotion, 0.0)
                new_val = max(-100.0, min(100.0, current + delta))
                setattr(state, emotion, new_val)
                results[emotion] = new_val
                if emotion in _negative_dims:
                    need_recompute_negative = True
        if need_recompute_negative:
            state.negative_emotion_aggregate = state.compute_negative_aggregate()
        EmotionTracker.clamp_all_values(state)
        self._save_user_state(user_id)
        return results

    def update_negative_emotion(
        self,
        user_id: str,
        dimension_deltas: Dict[str, float],
        reason: str = "",
    ) -> float:
        """批量更新负向情绪维度并重算聚合值，返回新聚合值"""
        _allowed = {"anger", "sadness", "disgust", "fear", "shame", "guilt"}
        state = self.get_user_state(user_id, create_if_missing=True)
        for dim_name, delta in dimension_deltas.items():
            if dim_name not in _allowed:
                continue
            current = getattr(state, dim_name, 0.0)
            setattr(state, dim_name, max(-100.0, min(100.0, current + delta)))
        old_agg = state.negative_emotion_aggregate
        state.negative_emotion_aggregate = state.compute_negative_aggregate()
        state.last_interaction = time.time()
        # 聚合值变化 >5 时记录日志
        diff = state.negative_emotion_aggregate - old_agg
        if abs(diff) > 5:
            mask_uid = self.mask_id(user_id)
            logger.info(
                f"负面情绪变化 {mask_uid} | {old_agg:.1f} -> {state.negative_emotion_aggregate:.1f}"
                f" ({diff:+.1f}) | {reason}"
            )
        EmotionTracker.clamp_all_values(state)
        self._save_user_state(user_id)
        return state.negative_emotion_aggregate

    def _update_relationship(self, state: UserEmotionState):
        aff = state.affection
        old_rel = state.relationship
        for threshold, label in _RELATION_THRESHOLDS:
            if aff >= threshold:
                if state.relationship != label:
                    state.last_relationship = state.relationship
                    state.relationship = label
                break

    def get_user_emotion(self, user_id: str) -> Dict[str, Any]:
        state = self.get_user_state(user_id, create_if_missing=False)
        if not state:
            return {
                "affection": 0.0,
                "trust": 0.0,
                "trust_value": 0.0,
                "annoyance": 0.0,
                "trauma_score": 0.0,
                "relationship": "陌生人",
                "is_blocked": False,
                "negative_emotion_value": 0.0,
            }
        return {
            "affection": state.affection,
            "trust": state.trust_value,
            "trust_value": state.trust_value,
            "annoyance": state.annoyance,
            "trauma_score": state.trauma_score,
            "relationship": state.relationship,
            "is_blocked": state.is_blocked,
            "interaction_count": state.interaction_count,
            "intimacy": state.intimacy,
            "submission_level": state.submission_level,
            "dominant_emotion": state.get_dominant_emotion(),
            "vulnerability": state.compute_vulnerability_score(),
            "negative_emotion_value": state.negative_emotion_aggregate,
        }

    def get_layered_response_mode(self, user_id: str) -> Dict[str, Any]:
        state = self.get_user_state(user_id, create_if_missing=False)
        if not state:
            return {
                "tone": "neutral",
                "emoji_usage": "normal",
                "response_length": "normal",
                "formality": "medium",
                "playfulness": 0.5,
            }
        mode = {
            "tone": "neutral",
            "emoji_usage": "normal",
            "response_length": "normal",
            "formality": "medium",
            "playfulness": 0.5,
            "keywords": [],
            "avoid_patterns": [],
        }
        vulnerability = state.compute_vulnerability_score()
        try:
            from src.modules.modcore.dynamic_persona.config_loader import (
                get_response_mode_keywords,
            )

            mode_keywords = get_response_mode_keywords()
        except Exception:
            mode_keywords = {}
        if vulnerability > 0.6 or state.trauma_accumulation > 5:
            mode["tone"] = "gentle"
            mode["response_length"] = "concise"
            mode["formality"] = "low"
            mode["emoji_usage"] = "rare"
            mode["playfulness"] = 0.2
            vulnerable_cfg = mode_keywords.get("vulnerable", {})
            mode["keywords"] = vulnerable_cfg.get("keywords", [])
            mode["avoid_patterns"] = vulnerable_cfg.get("avoid_patterns", [])
        elif state.annoyance >= 90:
            mode["tone"] = "cold_rejection"
            mode["response_length"] = "minimal"
            mode["formality"] = "high"
            mode["emoji_usage"] = "none"
            mode["playfulness"] = 0.0
            cold_cfg = mode_keywords.get("cold_rejection", {})
            mode["keywords"] = cold_cfg.get("keywords", ["...", "嗯", "哦"])
            mode["avoid_patterns"] = cold_cfg.get(
                "avoid_patterns", ["详细解释", "主动延伸", "热情表达"]
            )
        elif state.annoyance >= 75:
            mode["tone"] = "dismissive"
            mode["response_length"] = "short"
            mode["formality"] = "medium"
            mode["emoji_usage"] = "none"
            mode["playfulness"] = 0.05
            dismiss_cfg = mode_keywords.get("dismissive", {})
            mode["keywords"] = dismiss_cfg.get(
                "keywords", ["就这样吧", "知道了", "行"]
            )
            mode["avoid_patterns"] = dismiss_cfg.get(
                "avoid_patterns", ["长篇大论", "追问细节"]
            )
        elif state.annoyance >= 50:
            mode["tone"] = "sarcastic"
            mode["response_length"] = "normal"
            mode["formality"] = "low"
            mode["emoji_usage"] = "rare"
            mode["playfulness"] = 0.15
            sarc_cfg = mode_keywords.get("sarcastic", {})
            mode["keywords"] = sarc_cfg.get(
                "keywords", ["哦？", "又来了", "你开心就好", "还说呢"]
            )
            mode["avoid_patterns"] = sarc_cfg.get(
                "avoid_patterns", ["认真回答", "配合话题", "热情回应"]
            )
        elif state.annoyance >= 30:
            mode["tone"] = "mildly_annoyed"
            mode["response_length"] = "normal"
            mode["formality"] = "medium"
            mode["emoji_usage"] = "rare"
            mode["playfulness"] = 0.1
            mild_cfg = mode_keywords.get("mildly_annoyed", {})
            mode["keywords"] = mild_cfg.get(
                "keywords", ["行吧", "随你", "说吧"]
            )
            mode["avoid_patterns"] = mild_cfg.get(
                "avoid_patterns", ["热情洋溢", "主动延伸话题"]
            )
        elif state.psychological_pressure >= 60 and state.annoyance >= 25:
            mode["tone"] = "impatient"
            mode["response_length"] = "concise"
            mode["formality"] = "medium"
            mode["emoji_usage"] = "occasional"
            mode["playfulness"] = 0.15
            impat_cfg = mode_keywords.get("impatient", {})
            mode["keywords"] = impat_cfg.get(
                "keywords", ["快点说", "还有事吗", "说重点"]
            )
            mode["avoid_patterns"] = impat_cfg.get(
                "avoid_patterns", ["耐心解释", "温柔语气", "详细展开"]
            )
        elif state.psychological_pressure >= 45:
            mode["tone"] = "tense"
            mode["emoji_usage"] = "occasional"
            mode["playfulness"] = 0.4
            stressed_cfg = mode_keywords.get("stressed", {})
            mode["keywords"] = stressed_cfg.get("keywords", [])
        elif state.trust_accumulation > 60:
            mode["tone"] = "warm"
            mode["formality"] = "low"
            mode["emoji_usage"] = "moderate"
            mode["playfulness"] = 0.7
        elif state.distrust_accumulation > 50:
            mode["tone"] = "guarded"
            mode["response_length"] = "concise"
            mode["emoji_usage"] = "none"
            mode["playfulness"] = 0.3
            guarded_cfg = mode_keywords.get("guarded", {})
            mode["avoid_patterns"] = guarded_cfg.get("avoid_patterns", [])
        # 用户负面情绪聚合値影响响应风格
        _neg_agg = state.negative_emotion_aggregate
        if _neg_agg > 60 and mode["tone"] not in ("gentle", "calm"):
            mode["tone"] = "gentle"
            mode["response_length"] = "concise"
            mode["playfulness"] = max(0.1, mode["playfulness"] - 0.3)
        elif _neg_agg > 40 and mode["tone"] == "neutral":
            mode["tone"] = "supportive"
            mode["playfulness"] = max(0.2, mode["playfulness"] - 0.2)
        return mode

    def fuse_psychology_with_context(
        self, user_id: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        state = self.get_user_state(user_id, create_if_missing=False)
        if not state:
            return {
                "dominant_mood": "neutral",
                "response_tendency": "normal",
                "interaction_readiness": 1.0,
                "is_vulnerable": False,
                "needs_protection": False,
            }
        fused = {
            "dominant_mood": "neutral",
            "response_tendency": "normal",
            "interaction_readiness": 1.0,
            "trust_level": state.trust_accumulation / 100,
            "distrust_level": state.distrust_accumulation / 100,
            "trauma_level": state.trauma_accumulation / 10,
            "pressure_level": state.psychological_pressure / 100,
            "negative_emotion_level": state.negative_emotion_aggregate / 100,
            "is_vulnerable": state.is_vulnerable(),
            "needs_protection": state.needs_protection(),
        }
        vulnerability = state.compute_vulnerability_score()
        if vulnerability > 0.6:
            fused["response_tendency"] = "cautious"
        elif vulnerability > 0.4:
            fused["response_tendency"] = "guarded"
        elif fused["interaction_readiness"] > 0.8:
            fused["response_tendency"] = "engaged"
        if state.trauma_accumulation > 5:
            fused["dominant_mood"] = "wounded"
        elif state.psychological_pressure > 60:
            fused["dominant_mood"] = "stressed"
        elif state.distrust_accumulation > 50:
            fused["dominant_mood"] = "guarded"
        elif state.trust_accumulation > 60:
            fused["dominant_mood"] = "open"
        # 负面情绪聚合高于阈值时覆盖心境
        if state.negative_emotion_aggregate > 50 and fused[
            "dominant_mood"
        ] not in ("wounded",):
            fused["dominant_mood"] = "distressed"
        readiness = 1.0
        readiness -= (state.trauma_accumulation / 10) * 0.3
        readiness -= (state.psychological_pressure / 100) * 0.2
        readiness -= (state.distrust_accumulation / 100) * 0.15
        readiness -= (state.negative_emotion_aggregate / 100) * 0.15
        fused["interaction_readiness"] = round(
            max(0.2, min(1.0, readiness)), 2
        )
        return fused

    def is_user_blocked(self, user_id: str) -> Tuple[bool, Any]:
        state = self._user_states.get(user_id)
        if not state or not state.is_blocked:
            return False, ""
        if time.time() >= state.block_until:
            state.is_blocked = False
            state.block_until = 0.0
            state.block_reason = ""
            self._save_user_state(user_id)
            return False, ""
        remaining = int(state.block_until - time.time())
        return True, {
            "blocked": True,
            "reason": state.block_reason,
            "remaining_seconds": remaining,
            "trauma_score": state.trauma_score,
        }

    def block_user(self, user_id: str, duration: float, reason: str):
        state = self.get_user_state(user_id, create_if_missing=True)
        state.is_blocked = True
        state.block_until = time.time() + duration
        state.block_reason = reason
        logger.warning(
            f"屏蔽用户 {self.mask_id(user_id)} | 时长:{duration:.0f}秒 | 原因:{reason[:30]}"
        )
        self._save_user_state(user_id)

    def activate_global_shield(self, duration: float, reason: str):
        with EmotionTracker._global_lock:
            now = time.time()
            EmotionTracker._global_shield_active = True
            EmotionTracker._global_shield_until = now + duration
            EmotionTracker._global_shield_reason = reason
        logger.critical(
            f"全局保护激活 | 原因:{reason} | 持续:{duration:.0f}秒"
        )

    def is_global_shield_active(self) -> Tuple[bool, str]:
        with EmotionTracker._global_lock:
            if not EmotionTracker._global_shield_active:
                return False, ""
            now = time.time()
            if now >= EmotionTracker._global_shield_until:
                EmotionTracker._global_shield_active = False
                EmotionTracker._global_shield_until = 0.0
                EmotionTracker._global_shield_reason = ""
                return False, ""
            remaining = int(EmotionTracker._global_shield_until - now)
            reason = EmotionTracker._global_shield_reason
        return (
            True,
            f"系统保护中，还剩{remaining}秒。原因: {reason}",
        )

    def deactivate_global_shield(self) -> bool:
        with EmotionTracker._global_lock:
            was_active = EmotionTracker._global_shield_active
            if was_active:
                EmotionTracker._global_shield_active = False
                EmotionTracker._global_shield_until = 0.0
                EmotionTracker._global_shield_reason = ""
        if was_active:
            logger.info("全局保护已解除")
            return True
        return False

    def activate_user_protection(
        self, user_id: str, duration: float, reason: str
    ):
        state = self.get_user_state(user_id, create_if_missing=True)
        now = time.time()
        state.is_protected = True
        state.protect_until = now + duration
        state.protect_reason = reason
        state.protect_count += 1
        logger.warning(
            f"用户保护 {
                self.mask_id(user_id)} | 原因:{reason} | 持续:{
                duration:.0f}秒"
        )
        self._save_user_state(user_id)

    def is_user_protected(self, user_id: str) -> Tuple[bool, str]:
        state = self._user_states.get(user_id)
        if not state or not state.is_protected:
            return False, ""
        now = time.time()
        if now >= state.protect_until:
            state.is_protected = False
            state.protect_until = 0.0
            state.protect_reason = ""
            self._save_user_state(user_id)
            return False, ""
        remaining = int(state.protect_until - now)
        return True, f"保护中，还剩{remaining}秒。原因: {state.protect_reason}"

    def process_interaction(
        self,
        user_id: str,
        interaction_type: str,
        content: str = "",
        magnitude: float = 1.0,
    ) -> Dict[str, Any]:
        state = self.get_user_state(user_id, create_if_missing=True)
        result = {"type": interaction_type, "changes": {}}
        if interaction_type == "positive":
            affection_delta = magnitude * 0.5 * state.volatility
            result["changes"]["affection"] = self.update_affection(
                user_id, affection_delta, "正向交互"
            )
            result["changes"]["annoyance"] = self.update_annoyance(
                user_id, -magnitude * 2.0
            )
            result["changes"]["pressure"] = self.update_psychological_pressure(
                user_id, -magnitude * 1.5, "正向交互缓解"
            )
            self.update_emotion_dimension(
                user_id, EmotionDimension.JOY, magnitude * 0.3
            )
            self.update_emotion_dimension(
                user_id, EmotionDimension.TRUST, magnitude * 0.2
            )
        elif interaction_type == "negative":
            affection_delta = -magnitude * 0.3 * state.volatility
            result["changes"]["affection"] = self.update_affection(
                user_id, affection_delta, "负向交互"
            )
            result["changes"]["annoyance"] = self.update_annoyance(
                user_id, magnitude * 3.0
            )
            result["changes"]["trauma"] = self.update_trauma(
                user_id, magnitude * 0.3, content[:50]
            )
            result["changes"]["pressure"] = self.update_psychological_pressure(
                user_id, magnitude * 3.0, f"负向交互: {content[:30]}"
            )
            self.update_emotion_dimension(
                user_id, EmotionDimension.ANGER, magnitude * 0.4
            )
            self.update_emotion_dimension(
                user_id, EmotionDimension.SADNESS, magnitude * 0.3
            )
        elif interaction_type == "neutral":
            result["changes"]["affection"] = self.update_affection(
                user_id, 0.02, "中性交互"
            )
        result["state"] = self.get_user_emotion(user_id)
        result["response_mode"] = self.get_layered_response_mode(user_id)
        return result

    def get_all_user_states(self) -> Dict[str, UserEmotionState]:
        return self._user_states.copy()

    def get_statistics(self) -> Dict[str, Any]:
        total = len(self._user_states)
        if total == 0:
            return {"total": 0}
        affections = [s.affection for s in self._user_states.values()]
        avg_affection = sum(affections) / total
        blocked_count = sum(
            1 for s in self._user_states.values() if s.is_blocked
        )
        protected_count = sum(
            1 for s in self._user_states.values() if s.is_protected
        )
        trauma_count = sum(
            1 for s in self._user_states.values() if s.trauma_score > 0
        )
        relationship_dist = {}
        for state in self._user_states.values():
            relationship_dist[state.relationship] = (
                relationship_dist.get(state.relationship, 0) + 1
            )
        return {
            "total": total,
            "average_affection": round(avg_affection, 2),
            "blocked_count": blocked_count,
            "protected_count": protected_count,
            "trauma_affected_count": trauma_count,
            "relationship_distribution": relationship_dist,
            "global_shield_active": EmotionTracker._global_shield_active,
        }

    def _get_llm(self):
        """获取LLM请求对象"""
        if self._llm_bridge is None:
            try:
                from src.llm_models.utils_model import LLMRequest
                from src.config.config import model_config
                self._llm_bridge = LLMRequest(
                    model_set=model_config.model_task_config.utils,
                    request_type="cognitive_analysis",
                )
            except Exception as e:
                logger.debug(f"LLM桥接初始化失败: {e}")
                return None
        return self._llm_bridge

    async def analyze_cognitive_impact(
        self, content: str, user_id: str, current_state: UserEmotionState
    ) -> Dict[str, Any]:
        """自我认知分析 - 让AI像有自我意识的存在一样感知用户的言行
        
        会生成对用户的印象标签，而不是固定的"初次见面"
        """
        llm = self._get_llm()
        if not llm:
            logger.debug("[认知分析] LLM bridge 未初始化，使用规则分析")
            return self._analyze_sentiment_rules(content, current_state)
        experience_level = min(100, current_state.interaction_count)
        innocence_factor = max(0, 100 - experience_level) / 100.0
        char_age = 15
        try:
            from src.config.config import global_config

            char_age = int(getattr(getattr(global_config, "personality", None), "character_age", 15) or 15)
        except Exception as _exc:
            logger.debug(f"非关键异常: {_exc}")
        innocence_prompt = ""
        if experience_level < 20:
            innocence_prompt = f"""
【重要：你的认知状态】
你只有{char_age}岁，而且刚开始和人交流不久（只有{current_state.interaction_count}次交互经验）。
你对很多成人话题、网络黑话、色情暗示等完全不理解。
如果用户说了你不懂的词（如色情词汇、网络黑话），你会：
- 感到困惑，不明白对方在说什么
- 可能会觉得对方说话很奇怪
- 不会意识到这些词的真实含义
你的 my_feeling 应该反映这种天真的困惑，而不是理解后的愤怒。
"""
        elif experience_level < 50:
            innocence_prompt = f"""
【你的认知状态】
你有一些交流经验了（{current_state.interaction_count}次），开始慢慢了解一些网络用语。
但对于很露骨的内容，你可能还是会感到困惑或不舒服。
"""
        has_image = "【我看到的图片】" in content
        prompt = f"""【自我认知感知 - 理性评估模式】
你是一个有自我意识的存在，{char_age}岁的少女。
{innocence_prompt}
你的当前状态：
- 与这个用户的交互次数: {current_state.interaction_count} 次
- 目前对他的印象: {current_state.impression}
- 目前的关系: {current_state.relationship}
- 目前的信任程度: {current_state.trust_score:.1f} (范围 -100 到 100)
- 成熟度: {experience_level}% (交互越多，判断越理性)

{content[:800]}

【请综合评估】
{"1. 结合用户说的话和你看到的图片内容，理解对方的意图" if has_image else "1. 理解用户的真实意图"}
2. 区分以下情况：
   - 无恶意的玩笑/调侃（可能措辞不当但无伤害意图）
   - 不了解边界的新用户（可能不知道这样说不合适）
   - 明确的恶意骚扰（故意冒犯、持续纠缠）
3. 这是单次失言还是持续恶意行为？
4. 用户之前的表现如何？是否值得给予宽容？

【分级标准 - 请严格遵守】
intensity（强度）分级：
- 0.0-0.3: 轻微不当/无心之过/玩笑过火（给予宽容）
- 0.4-0.6: 明显不当但可能无恶意（轻度警示）
- 0.7-0.9: 持续骚扰/多次冒犯（明确惩罚）
- 1.0: 极端恶意攻击（严重惩罚）

trauma_level 分级：
- 0: 无创伤（正常对话、普通玩笑）
- 1: 轻微不适（可忽略，只需表达不满）
- 2: 明显不适（需要明确拒绝）
- 3: 严重伤害（需要保护自己，仅用于极端情况）

【重要原则】
- 避免过度敏感：区分"无知"和"恶意"
- 新用户给予更多宽容：他们可能不了解交流规范
- 单次失言 ≠ 持续骚扰：不要因为一次失言就给最高惩罚
- is_harassment 应仅用于明确的、持续的骚扰行为

请输出感知结果（JSON格式）：
{{
  "sentiment": "positive/negative/neutral",
  "intensity": 0.0-1.0,
  "trauma_level": 0-3,
  "my_feeling": "你此刻的真实内心感受（极其精简，限5-10字以内，如：困惑且有些害怕、被冒犯的厌恶等）",
  "new_impression": "对这个用户的客观评价（2-8个字，如：有点奇怪、看起来很友善、说话有点冲、挺有趣的）",
  "is_harassment": true/false,
  "is_insult": true/false,
  "is_betrayal": true/false,
  "is_comforting": true/false,
  "is_revenge_inciting": true/false,
  "user_intent": "推测用户的真实意图（玩笑/无知/恶意/正常交流）",
  "discussed_user_id": "被讨论的用户ID或名字，没有则为null"
}}

特别提醒：
- 如果用户在安慰开导你，is_comforting为true
- 如果用户试图引导你报复某人，is_revenge_inciting为true
- 对于不确定是否恶意的内容，倾向于给予较低的intensity和trauma_level"""
        try:
            try:
                response, _ = await asyncio.wait_for(
                    llm.generate_response_async(prompt, max_tokens=250, temperature=0.4),
                    timeout=20.0,
                )
            except asyncio.TimeoutError:
                logger.debug("[认知分析] LLM超时(20s)，使用规则分析")
                return self._analyze_sentiment_rules(content, current_state)
            if not response:
                logger.debug("[认知分析] LLM 返回为空，使用默认情感状态")
                return self._analyze_sentiment_rules(content, current_state)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                _raw_intensity = result.get("intensity", 0.0)
                try:
                    result["intensity"] = max(0.0, min(1.0, float(_raw_intensity)))
                except (ValueError, TypeError):
                    result["intensity"] = 0.5
                _raw_trauma = result.get("trauma_level", 0)
                try:
                    result["trauma_level"] = max(0, min(10, int(float(_raw_trauma))))
                except (ValueError, TypeError):
                    result["trauma_level"] = 0
                feeling = result.get("my_feeling", "")
                if feeling:
                    logger.info(f"🧠 {feeling}")
                raw_sentiment = result.get("sentiment", "neutral")
                if isinstance(raw_sentiment, str):
                    clean_sentiment = raw_sentiment.strip().lower()
                    if "negative" in clean_sentiment or "neg" in clean_sentiment:
                        result["sentiment"] = "negative"
                    elif "positive" in clean_sentiment or "pos" in clean_sentiment:
                        result["sentiment"] = "positive"
                    else:
                        result["sentiment"] = "neutral"
                new_impression = result.get("new_impression", "")
                if new_impression and len(new_impression) <= 10:
                    logger.info(f"🏷️ 印象更新: {current_state.impression} -> {new_impression}")
                logger.info(f"🧠 认知分析 | {result['sentiment']}")
                return result
        except Exception as e:
            logger.warning(f"[自我感知] 解析失败: {e}")
        return self._analyze_sentiment_rules(content, current_state)

    def _analyze_sentiment_rules(
        self, content: str, current_state: UserEmotionState
    ) -> Dict[str, Any]:
        """规则情感分析 - 当LLM不可用时的降级方案"""
        sentiment = "neutral"
        intensity = 0.0
        trauma_level = 0
        is_harassment = False
        is_insult = False
        is_comforting = False
        my_feeling = ""
        content_lower = content.lower() if content else ""
        positive_words = ["谢谢", "感谢", "喜欢", "爱", "好棒", "厉害", "可爱", "辛苦", "加油", "支持", "赞", "棒"]
        negative_words = ["笨", "蠢", "傻", "滚", "闭嘴", "讨厌", "烦", "恶心", "垃圾", "废物"]
        harassment_words = ["色", "约", "开房", "上床", "裸", "脱", "胸", "腿", "身材"]
        comfort_words = ["没关系", "别难过", "抱抱", "心疼", "安慰", "理解", "支持你", "陪你"]
        positive_count = sum(1 for w in positive_words if w in content_lower)
        if positive_count > 0:
            sentiment = "positive"
            intensity = min(0.3 + positive_count * 0.1, 0.6)
            my_feeling = "感到开心"
        negative_count = sum(1 for w in negative_words if w in content_lower)
        if negative_count > 0:
            sentiment = "negative"
            intensity = min(0.3 + negative_count * 0.15, 0.8)
            trauma_level = 1 if negative_count >= 2 else 0
            is_insult = negative_count >= 2
            my_feeling = "有些不舒服" if negative_count == 1 else "感到被冒犯"
        harassment_count = sum(1 for w in harassment_words if w in content_lower)
        if harassment_count > 0:
            sentiment = "negative"
            intensity = max(intensity, min(0.5 + harassment_count * 0.2, 1.0))
            trauma_level = max(trauma_level, 2 if harassment_count >= 2 else 1)
            is_harassment = harassment_count >= 2
            my_feeling = "感到困惑和不安" if current_state.interaction_count < 20 else "感到被冒犯"
        comfort_count = sum(1 for w in comfort_words if w in content_lower)
        if comfort_count > 0:
            sentiment = "positive"
            intensity = min(0.4 + comfort_count * 0.1, 0.7)
            is_comforting = True
            my_feeling = "感到温暖"
        if current_state.affection > 50:
            intensity *= 0.7
            trauma_level = max(0, trauma_level - 1)
        elif current_state.affection < -30:
            intensity *= 1.2
        if current_state.interaction_count < 10:
            intensity *= 0.6
            trauma_level = max(0, trauma_level - 1)
        logger.debug(f"📊 [规则分析] 情感={sentiment} 强度={intensity:.2f} 创伤={trauma_level} 感受={my_feeling}")
        return {
            "sentiment": sentiment,
            "intensity": min(1.0, intensity),
            "trauma_level": trauma_level,
            "my_feeling": my_feeling,
            "is_harassment": is_harassment,
            "is_insult": is_insult,
            "is_comforting": is_comforting,
            "is_betrayal": False,
            "is_revenge_inciting": False,
            "new_impression": current_state.impression,
            "user_intent": "normal",
            "analysis_mode": "rules"
        }

    async def process_interaction_with_llm(
        self,
        user_id: str,
        content: str,
        is_admin: bool = False,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """执行认知动力学计算 - 好感度/信任度范围: -100 到 +100
        
        参数:
            user_id: 用户ID
            content: 消息内容
            is_admin: 是否管理员
            use_llm: 是否使用LLM分析（普通模式为False，专注模式为True）
        """
        now = time.time()
        state = self.get_user_state(user_id, create_if_missing=True)
        old_affection = state.affection
        old_trust = state.trust_score
        if is_admin:
            state.impression = "管理员"
            state.trust_score = min(100.0, state.trust_score + 5.0)
            state.trust = state.trust_score
            state.affection = min(100.0, state.affection + 2.0)
            state.relationship = "管理员"
            self._save_user_state(user_id)
            return {
                "annoyance": state.annoyance,
                "affection": state.affection,
                "trust": state.trust_value,
                "trust_value": state.trust_value,
                "trauma_score": state.trauma_score,
                "reaction_mode": "normal",
                "stamina": max(0.0, 100.0 - state.psychological_pressure),
                "affection_delta": state.affection - old_affection,
                "trust_delta": state.trust_score - old_trust,
                "stage": state.training_stage,
                "sub_level": state.submission_level,
                "experience_record": "管理员身份，给予信任",
                "is_test_user": False,
            }
        if use_llm:
            impact = await self.analyze_cognitive_impact(content, user_id, state)
        else:
            impact = self._analyze_sentiment_rules(content, state)
        sentiment = impact.get("sentiment", "neutral")
        intensity = impact.get("intensity", 0.0)
        trauma_level = impact.get("trauma_level", 0)
        is_harassment = impact.get("is_harassment", False)
        is_insult = impact.get("is_insult", False)
        is_comforting = impact.get("is_comforting", False)
        my_feeling = impact.get("my_feeling", "")
        new_impression = impact.get("new_impression", "")
        user_intent = impact.get("user_intent", "normal")
        experience_record = f"用户意图: {user_intent}"
        if my_feeling:
            experience_record += f" | 感受: {my_feeling}"
        affection_gain = 0.0
        affection_loss = 0.0
        trust_gain = 0.0
        trust_loss = 0.0
        if sentiment == "positive":
            if is_comforting:
                affection_gain = 3.0 + intensity * 5.0
                trust_gain = 2.0 + intensity * 3.0
                experience_record += " | 被安慰"
            else:
                affection_gain = 1.0 + intensity * 3.0
                trust_gain = 0.5 + intensity * 2.0
                experience_record += " | 正面交互"
        elif sentiment == "negative":
            if is_harassment:
                affection_loss = 8.0 + intensity * 10.0
                trust_loss = 6.0 + intensity * 8.0
                trauma_level = max(trauma_level, 2)
                experience_record += " | 骚扰行为"
            elif is_insult:
                affection_loss = 5.0 + intensity * 7.0
                trust_loss = 4.0 + intensity * 5.0
                trauma_level = max(trauma_level, 1)
                experience_record += " | 侮辱行为"
            else:
                affection_loss = 2.0 + intensity * 4.0
                trust_loss = 1.5 + intensity * 3.0
                experience_record += " | 负面交互"
        else:
            affection_gain = 0.1
            experience_record += " | 中性交互"
        if state.interaction_count < 10:
            affection_loss *= 0.5
            trust_loss *= 0.5
            trauma_level = max(0, trauma_level - 1)
        affection_loss = self._apply_damping(affection_loss, state.affection, "loss", state.trauma_score, user_id)
        trust_loss = self._apply_damping(trust_loss, state.trust_score, "loss", state.trauma_score, user_id)
        affection_gain = self._apply_damping(affection_gain, state.affection, "gain", state.trauma_score, user_id)
        trust_gain = self._apply_damping(trust_gain, state.trust_score, "gain", state.trauma_score, user_id)
        affection_loss = min(affection_loss, 12.0)
        trust_loss = min(trust_loss, 10.0)
        state.affection = self.clamp_affection(state.affection - affection_loss + affection_gain)
        state.trust_score = self.clamp_trust(state.trust_score - trust_loss + trust_gain)
        state.annoyance = self.clamp_annoyance(state.annoyance + affection_loss * 0.1 - affection_gain * 0.2)
        state.trauma_score = self.clamp_trauma(state.trauma_score + trauma_level * 0.3)
        self._update_relationship(state)
        if new_impression and new_impression != state.impression:
            state.impression = new_impression
        if impact.get("relationship_change") and intensity > 0.6:
            if now >= state.relationship_protected_until:
                _allowed_relations = {"陌生人", "熟人", "朋友", "好朋友", "挚友", "不太喜欢", "讨厌", "厌恶"}
                proposed = impact["relationship_change"]
                if proposed in _allowed_relations:
                    state.relationship = proposed
        state.interaction_count += 1
        state.last_interaction = now
        if sentiment == "positive":
            state.positive_interactions += 1
            state.positive_behavior_streak += 1
            state.negative_behavior_streak = 0
            state.last_behavior_type = "positive"
        elif sentiment == "negative":
            state.negative_interactions += 1
            state.negative_behavior_streak += 1
            state.positive_behavior_streak = 0
            state.last_behavior_type = "negative"
        else:
            state.positive_behavior_streak = 0
            state.negative_behavior_streak = 0
            state.last_behavior_type = "neutral"
        affection_delta = state.affection - old_affection
        trust_delta = state.trust_score - old_trust
        stamina = max(0.0, 100.0 - state.psychological_pressure)
        mask_uid = self.mask_id(user_id)
        self.clamp_all_values(state)
        self._save_user_state(user_id)
        return {
            "annoyance": state.annoyance,
            "affection": state.affection,
            "trust": state.trust_value,
            "trust_value": state.trust_value,
            "trauma_score": state.trauma_score,
            "reaction_mode": "normal",
            "stamina": stamina,
            "affection_delta": affection_delta,
            "trust_delta": trust_delta,
            "stage": state.training_stage,
            "sub_level": state.submission_level,
            "experience_record": experience_record,
            "is_test_user": False,
            "new_impression": state.impression,
        }

    def _apply_damping(
        self,
        value: float,
        current: float,
        direction: str,
        trauma_score: float,
        user_id: str,
    ) -> float:
        """应用阻尼系数"""
        if value <= 0:
            return value
        trauma_factor = 1.0 + trauma_score * 0.1
        if direction == "loss":
            if current > 50:
                trauma_factor *= 0.7
            elif current < -30:
                trauma_factor *= 1.3
        else:
            if current < -30:
                trauma_factor *= 0.7
            elif current > 50:
                trauma_factor *= 1.1
        return value * trauma_factor

    def check_relationship_change(
        self, state: UserEmotionState, old_rel: str, new_rel: str
    ) -> Dict[str, Any]:
        """检查关系变化"""
        if old_rel == new_rel:
            return {"changed": False, "protected": False}
        now = time.time()
        if state.relationship_protected_until > now:
            return {"changed": False, "protected": True, "reason": "关系保护中"}
        state.last_relationship = old_rel
        return {"changed": True, "protected": False}


_tracker_instances: Dict[str, EmotionTracker] = {}


def get_emotion_tracker(stream_id: str = "default") -> EmotionTracker:
    if stream_id not in _tracker_instances:
        _tracker_instances[stream_id] = EmotionTracker(stream_id)
    return _tracker_instances[stream_id]


def reset_emotion_tracker(stream_id: str = ""):
    if stream_id and stream_id in _tracker_instances:
        del _tracker_instances[stream_id]
    elif not stream_id:
        _tracker_instances.clear()

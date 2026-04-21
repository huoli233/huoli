import math
import time
import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

from src.chat.heart_flow.dimension_protocol import (
    DimensionBase,
    DimensionScope,
    EventContext,
    TickResult,
)
from src.chat.heart_flow.vote_types import EnergyChainVote, EnergyStage
from src.common.logger import get_logger
from src.config.core_config_engine import get_core_config
from src.common.data_models.heartflow_models import EnergySnapshot

logger = get_logger("vitality_pool")


@dataclass
class NightMode:
    """夜间模式数据"""

    phase: str = "DAYTIME"
    prob_multiplier: float = 1.0
    ceiling_penalty: float = 0.0
    perception_label: str = ""
    hour_start: int = 0
    hour_end: int = 24


@dataclass
class ChannelEnergyState:
    """频道能量状态"""

    chat_pool: float = 100.0
    chat_ceiling: float = 100.0
    thinking_value: float = 100.0
    thinking_ceiling: float = 100.0
    annoyance_level: float = 0.0
    activity_level: float = 50.0
    social_value: float = 0.0
    chain_count: int = 0
    last_update: float = field(default_factory=time.time)
    last_reply_time: float = 0.0
    night_reply_count: int = 0
    night_reply_cap: int = 3
    total_consumed_today: float = 0.0
    boost_active: bool = False
    boost_source: str = ""
    boost_expire: float = 0.0
    night_disturbance_count: int = 0
    last_disturbance_time: float = 0.0
    prev_disturbance_time: float = 0.0
    last_disturbance_interval_sec: float = 0.0
    wake_adaptive_threshold: float = 50.0
    wake_learning_rate: float = 6.0
    wake_min_threshold: float = 15.0
    wake_max_threshold: float = 90.0
    wake_decision_history: list = field(default_factory=list)
    last_stimulus_strength: float = 0.0
    consecutive_sleep_decisions: int = 0
    consecutive_wake_decisions: int = 0

    def chat_ratio(self) -> float:
        return max(0.0, min(1.0, self.chat_pool / max(self.chat_ceiling, 1.0)))

    def thinking_ratio(self) -> float:
        return max(0.0, min(1.0, self.thinking_value / max(self.thinking_ceiling, 1.0)))

    def combined_ratio(self) -> float:
        return (self.chat_ratio() + self.thinking_ratio()) / 2.0

    def is_exhausted(self) -> bool:
        return self.combined_ratio() < 0.05 or self.chat_pool < 3.0

    def to_snapshot(self):
        return EnergySnapshot(
            chat_pool=self.chat_pool,
            chat_ceiling=self.chat_ceiling,
            thinking_value=self.thinking_value,
            thinking_ceiling=self.thinking_ceiling,
            social_value=self.social_value,
            trust_value=0.0,
            annoyance_level=self.annoyance_level,
            activity_level=self.activity_level,
        )


_NIGHT_PHASES: Dict[str, NightMode] = {
    "ACTIVE_TWILIGHT": NightMode(
        phase="ACTIVE_TWILIGHT",
        prob_multiplier=0.5,
        ceiling_penalty=0.15,
        perception_label="黄昏微倦",
        hour_start=22,
        hour_end=23,
    ),
    "DROWSY": NightMode(
        phase="DROWSY",
        prob_multiplier=0.25,
        ceiling_penalty=0.30,
        perception_label="浅睡朦胧",
        hour_start=23,
        hour_end=1,
    ),
    "DEEP_VALLEY": NightMode(
        phase="DEEP_VALLEY",
        prob_multiplier=0.08,
        ceiling_penalty=0.55,
        perception_label="深睡沉寂",
        hour_start=1,
        hour_end=4,
    ),
    "DAWN_RECOVERY": NightMode(
        phase="DAWN_RECOVERY",
        prob_multiplier=0.4,
        ceiling_penalty=0.20,
        perception_label="黎明苏醒",
        hour_start=4,
        hour_end=6,
    ),
}


class EnergyChainDimension(DimensionBase):
    """
    D6 能量链条维度 — 独立实现的能量管理系统

    不再包装 VitalityPoolManager / SharedResourceManager。
    自身管理全部能量状态，内置:
    - 水库+管道模型 (energy_bar_total → 子数值分配)
    - 链条扣除机制 (子数值耗尽→总额扣%)
    - 动态阻尼恢复 (总额越低恢复越慢)
    - 夜间四阶段感知驱动 (ACTIVE_TWILIGHT/DROWSY/DEEP_VALLEY/DAWN_RECOVERY)
    - 特殊事件充电池 (interesting_topic/positive_interaction/long_silence/topic_resonance)
    """

    _singleton: Optional["EnergyChainDimension"] = None

    @classmethod
    def get_instance(cls) -> "EnergyChainDimension":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        self._channels: Dict[str, ChannelEnergyState] = {}
        self._legacy_pool: Optional[Any] = None
        self._legacy_shared: Optional[Any] = None
        self._migrated_channels: Set[str] = set()

    @property
    def dimension_name(self) -> str:
        return "energy_chain"

    @property
    def scope(self) -> DimensionScope:
        return DimensionScope.PER_CHANNEL

    @property
    def needs_persistence(self) -> bool:
        return True

    @property
    def tick_interval_sec(self) -> float:
        return 3.0

    @property
    def persist_interval_sec(self) -> float:
        return 60.0

    def initialize(self):
        for cid in list(self._channels.keys()):
            self._ensure_channel(cid)

    def _ensure_channel(self, channel_id: str) -> ChannelEnergyState:
        if channel_id not in self._channels:
            state = ChannelEnergyState()
            state.last_update = time.time()
            self._channels[channel_id] = state
        return self._channels[channel_id]

    def tick(self, elapsed_sec: float):
        updated = False
        now = time.time()
        for _cid, state in self._channels.items():
            night_mode = self._get_night_mode(_cid)
            old_chat = state.chat_pool
            old_think = state.thinking_value
            elapsed_min = elapsed_sec / 60.0
            combined = state.combined_ratio()
            damping = max(0.15, combined * 0.6 + 0.25)
            if night_mode.phase != "DAYTIME":
                damping *= 0.5
            base_chat_recovery = 3.0 * damping * elapsed_min
            base_think_recovery = 0.8 * damping * elapsed_min
            if state.boost_active and now < state.boost_expire:
                base_chat_recovery *= 3.0
                base_think_recovery *= 3.0
            elif state.boost_active and now >= state.boost_expire:
                state.boost_active = False
                state.boost_source = ""
            effective_chat_ceiling = state.chat_ceiling * (1.0 - night_mode.ceiling_penalty)
            state.chat_pool = min(effective_chat_ceiling, state.chat_pool + base_chat_recovery)
            state.thinking_value = min(state.thinking_ceiling, state.thinking_value + base_think_recovery)
            decay_rate = 0.5 * elapsed_min
            state.annoyance_level = max(0.0, state.annoyance_level - decay_rate)
            state.activity_level = max(10.0, min(100.0, state.activity_level - elapsed_min * 0.3))
            if night_mode.phase == "DAYTIME":
                state.night_disturbance_count = 0
                state.wake_adaptive_threshold = 50.0
                state.consecutive_sleep_decisions = 0
                state.consecutive_wake_decisions = 0
                state.last_stimulus_strength = 0.0
                state.wake_decision_history.clear()
            _quiet_sec = 0.0
            if state.night_disturbance_count > 0 and state.last_disturbance_time > 0:
                _quiet_sec = now - state.last_disturbance_time
                if _quiet_sec > 90.0:
                    _decay = int(_quiet_sec / 90.0)
                    state.night_disturbance_count = max(0, state.night_disturbance_count - _decay)
            _hist_len = len(state.wake_decision_history)
            if _hist_len > 20:
                state.wake_decision_history = state.wake_decision_history[-15:]
            elif _hist_len > 8 and _quiet_sec > 300:
                _recent_wakes = sum(
                    1
                    for d in state.wake_decision_history[-6:]
                    if d.get("action") in ("soft_wake", "full_wake", "grumpy_glance")
                )
                if _recent_wakes <= 1:
                    state.wake_adaptive_threshold = min(state.wake_max_threshold, state.wake_adaptive_threshold + 2.0)
            state.last_update = now
            if abs(state.chat_pool - old_chat) > 0.01 or abs(state.thinking_value - old_think) > 0.01:
                updated = True
        return TickResult(
            dimension_name=self.dimension_name,
            updated=updated,
            summary=f"频道数={len(self._channels)}" if updated else "",
        )

    def on_event(self, ctx):
        if not ctx.channel_id:
            return
        state = self._ensure_channel(ctx.channel_id)
        now = time.time()
        event_type = getattr(ctx, "event_type", "") or ""
        raw_extras = getattr(ctx, "raw_extras", {}) or {}
        if event_type in ("message_received", "user_message"):
            is_repeat = bool(raw_extras.get("is_repeat", False))
            is_mentioned = bool(raw_extras.get("is_mentioned", False))
            content_len = getattr(ctx, "message_length", 0) or 0
            intensity = min(1.0, content_len / 200.0)
            base_boost = 0.3 + intensity * 0.4
            if is_repeat:
                base_boost *= 0.3
            state.chat_pool = min(state.chat_ceiling, state.chat_pool + base_boost)
            state.activity_level = min(100.0, state.activity_level + 0.15 + intensity * 0.25)
            state.social_value = max(-50.0, min(100.0, state.social_value + 0.3))
            # 触发条件：重复提问同一话题 → 累积烦躁
            if is_repeat:
                state.annoyance_level = min(80.0, state.annoyance_level + 3.0)
                logger.debug(f"重复话题触发烦躁 +3.0 → {state.annoyance_level:.1f}")
            # 触发条件：深夜被打扰 → 累积烦躁
            night_mode = self._get_night_mode(ctx.channel_id)
            if night_mode.phase != "DAYTIME":
                _night_annoy = 2.5 if "DEEP" in night_mode.phase else 1.5
                state.annoyance_level = min(80.0, state.annoyance_level + _night_annoy)
                logger.debug(f"深夜消息触发烦躁 +{_night_annoy:.1f} → {state.annoyance_level:.1f}")
            # 触发条件：短时间内被大量@ → 累积烦躁
            if is_mentioned:
                _mention_gap = now - getattr(state, "_last_mention_time", 0.0)
                _mention_burst = getattr(state, "_mention_burst_count", 0)
                if _mention_gap < 60.0:
                    _mention_burst += 1
                else:
                    _mention_burst = 1
                state._last_mention_time = now
                state._mention_burst_count = _mention_burst
                if _mention_burst >= 3:
                    _spam_annoy = min(5.0, _mention_burst * 1.2)
                    state.annoyance_level = min(80.0, state.annoyance_level + _spam_annoy)
                    logger.debug(
                        f"频繁@触发烦躁 +{_spam_annoy:.1f} ({_mention_burst}次/min) → {state.annoyance_level:.1f}"
                    )
        elif event_type in ("reply_completed", "reply"):
            reply_tokens = getattr(ctx, "reply_tokens", 0) or 80
            complexity = max(0.05, min(1.0, reply_tokens / 500.0))
            is_admin_msg = bool(getattr(ctx, "is_admin", False) or raw_extras.get("is_admin", False))
            night_mode = self._get_night_mode(ctx.channel_id)
            if night_mode.phase != "DAYTIME":
                try:
                    from src.core.night_cycle_system import get_night_cycle

                    _ncs_em = get_night_cycle(ctx.channel_id)
                    _ncs_em.record_night_reply()
                    state.night_reply_count = _ncs_em.state_snapshot.night_reply_count
                    if not _ncs_em.can_reply_tonight():
                        return
                except Exception:
                    state.night_reply_count += 1
                    if state.night_reply_count >= state.night_reply_cap:
                        return
            _think_remaining = state.thinking_value
            _chat_remaining = state.chat_pool
            chat_cost_base = 1.2 + complexity * 3.5
            think_cost_base = 0.8 + complexity * 2.0
            ratio = state.combined_ratio()
            cost_mult = 0.4 if ratio < 0.2 else (0.7 if ratio < 0.5 else 1.0)
            if is_admin_msg:
                cost_mult *= 0.5
            if _think_remaining < 10:
                think_cost_base *= 0.4
            elif _think_remaining < 25:
                think_cost_base *= 0.6
            elif _think_remaining < 50:
                think_cost_base *= 0.8
            if _chat_remaining < 10:
                chat_cost_base *= 0.3
            elif _chat_remaining < 25:
                chat_cost_base *= 0.6
            final_chat_cost = max(0.2, min(chat_cost_base * cost_mult, _chat_remaining * 0.5))
            final_think_cost = max(0.1, min(think_cost_base * cost_mult, _think_remaining * 0.5))
            state.chat_pool = max(0.0, state.chat_pool - final_chat_cost)
            state.thinking_value = max(0.0, state.thinking_value - final_think_cost)
            state.chain_count += 1
            state.total_consumed_today += final_chat_cost + final_think_cost
            state.last_reply_time = now
            state.activity_level = max(5.0, state.activity_level - 0.8 * complexity)
            if state.chat_pool < state.chat_ceiling * 0.15:
                state.annoyance_level = min(80.0, state.annoyance_level + 1.5)
        elif event_type == "thinking_completed":
            think_cost = getattr(ctx, "complexity", 1.0) or 1.0
            state.thinking_value = max(0.0, state.thinking_value - think_cost * 1.5)
        elif event_type == "glance":
            state.thinking_value = max(0.0, state.thinking_value - 0.3)
        elif event_type == "bot_unanswered":
            # 触发条件：被忽略（连续多条bot消息无人回复）→ 累积烦躁
            unanswered_turns = int(raw_extras.get("unanswered_turns", 0) or 0)
            if unanswered_turns >= 2:
                # 2次未回复+1.5，3次+3.0，4次+4.5，最高+6.0
                _ignore_annoy = min(6.0, (unanswered_turns - 1) * 1.5)
                state.annoyance_level = min(80.0, state.annoyance_level + _ignore_annoy)
                # 被忽略也降低社交值
                state.social_value = max(-50.0, state.social_value - 0.5 * unanswered_turns)
                logger.debug(
                    f"被忽略触发烦躁 +{_ignore_annoy:.1f} (连续{unanswered_turns}次未回复) "
                    f"→ annoyance={state.annoyance_level:.1f} social={state.social_value:.1f}"
                )

    _NCS_TO_D6_MAP = {
        "DROWSY": "ACTIVE_TWILIGHT",
        "NIGHT_ACTIVE": "ACTIVE_TWILIGHT",
        "SOCIAL_NIGHT": "ACTIVE_TWILIGHT",
        "LIGHT_SLEEP": "DROWSY",
        "MIDNIGHT_REFLECT": "DROWSY",
        "QUIET_CONTEMPLATE": "DROWSY",
        "DEEP_SLEEP": "DEEP_VALLEY",
        "BURNED_OUT": "DEEP_VALLEY",
        "DAWN_RECOVER": "DAWN_RECOVERY",
    }

    def _get_night_mode(self, channel_id: str = "") -> NightMode:
        try:
            from src.core.night_cycle_system import get_night_cycle

            def _resolve_d6_phase(_night_cycle: Any) -> str:
                _snap = getattr(_night_cycle, "state_snapshot", None)
                _phase_obj = getattr(_snap, "current_phase", None)
                if _phase_obj is None:
                    _phase_obj = getattr(_night_cycle, "phase", None)
                _phase_value = str(
                    _phase_obj.value
                    if hasattr(_phase_obj, "value")
                    else str(_phase_obj or "")
                ).strip().upper()
                return self._NCS_TO_D6_MAP.get(_phase_value, "")

            if channel_id:
                _ncs_mode = get_night_cycle(channel_id)
                _d6_phase = _resolve_d6_phase(_ncs_mode)
                if _d6_phase and _d6_phase in _NIGHT_PHASES:
                    return _NIGHT_PHASES[_d6_phase]

            for _channel_id in list(self._channels.keys()):
                _ncs_mode = get_night_cycle(_channel_id)
                _d6_phase = _resolve_d6_phase(_ncs_mode)
                if _d6_phase and _d6_phase in _NIGHT_PHASES:
                    return _NIGHT_PHASES[_d6_phase]
        except Exception as _exc:
            logger.debug(f"非关键异常: {_exc}")
        now = datetime.datetime.now().hour
        for phase_cfg in _NIGHT_PHASES.values():
            _start = int(phase_cfg.hour_start)
            _end = int(phase_cfg.hour_end)
            _in_slot = (_start <= _end and _start <= now < _end) or (_start > _end and (now >= _start or now < _end))
            if _in_slot:
                return phase_cfg
        return NightMode(phase="DAYTIME", prob_multiplier=1.0, perception_label="")

    def record_disturbance(self, channel_id: str) -> dict:
        """记录一次夜间打扰信号，计算刺激强度，返回多维状态快照

        返回值包含:
          - stimulus_strength: 归一化刺激强度(0-100)，用于与自适应阈值比较
          - threshold: 当前动态阈值
          - exceeds_threshold: 刺激是否超过阈值（决定是否触发L1/L2模型）
          - context: 完整多维状态快照
        """
        state = self._ensure_channel(channel_id)
        now = time.time()
        _prev = float(getattr(state, "last_disturbance_time", 0.0) or 0.0)
        state.prev_disturbance_time = _prev
        state.last_disturbance_interval_sec = max(0.0, now - _prev) if _prev > 0 else 0.0
        state.night_disturbance_count += 1
        state.last_disturbance_time = now
        _nm = self._get_night_mode(channel_id)
        _context = self._build_full_context(state, _nm, channel_id, now)
        _stimulus = self._compute_stimulus_strength(_context, state)
        state.last_stimulus_strength = _stimulus
        _threshold = state.wake_adaptive_threshold
        _context["adaptive"] = {
            "stimulus_strength": round(_stimulus, 1),
            "threshold": round(_threshold, 1),
            "exceeds": _stimulus >= _threshold,
            "gap": round(_stimulus - _threshold, 1),
            "consecutive_sleeps": state.consecutive_sleep_decisions,
            "consecutive_wakes": state.consecutive_wake_decisions,
        }
        return _context

    def _build_full_context(self, state, night_mode, channel_id, now):
        """构建完整的多维状态上下文"""
        _context = {
            "disturbance": {
                "count": state.night_disturbance_count,
                "last_time": state.last_disturbance_time,
                "quiet_seconds": round(float(getattr(state, "last_disturbance_interval_sec", 0.0) or 0.0), 3),
                "interval_density": self._calc_interval_density(state),
            },
            "d6_phase": night_mode.phase,
            "d6_label": night_mode.perception_label or "",
            "d6_prob_mult": night_mode.prob_multiplier,
            "d6_ceiling_penalty": night_mode.ceiling_penalty,
            "energy": {
                "chat_ratio": round(state.chat_ratio(), 3),
                "thinking_ratio": round(state.thinking_ratio(), 3),
                "combined_ratio": round(state.combined_ratio(), 3),
                "annoyance": round(state.annoyance_level, 2),
                "activity": round(state.activity_level, 1),
                "social": round(state.social_value, 2),
                "night_replies_used": state.night_reply_count,
                "night_replies_cap": state.night_reply_cap,
                "is_exhausted": state.is_exhausted(),
            },
        }
        try:
            from src.core.night_cycle_system import get_night_cycle

            _ncs = get_night_cycle(channel_id)
            _ncs_state = _ncs.state_snapshot
            _context["cde_3d"] = {
                "drowsiness": round(_ncs_state.drowsiness_value, 2),
                "pressure": round(_ncs_state.overnight_pressure, 2),
                "reserve": round(_ncs_state.sleep_reserve, 2),
                "composite_load": round(_ncs_state.composite_load, 2),
                "collapse_threshold": round(_ncs_state.collapse_threshold, 2),
                "collapse_imminent": _ncs_state.collapse_imminent,
                "laziness_prob": round(_ncs_state.laziness_probability, 3),
                "wake_irritation": round(_ncs_state.wake_irritation_level, 3),
                "half_asleep": round(_ncs_state.half_asleep_level, 3),
                "response_suppression": round(_ncs_state.response_suppression_coef, 3),
                "body_state": _ncs_state.body_state_tag or "",
                "interruption_count": _ncs_state.interruption_count,
                "force_wake_today": _ncs_state.force_wake_count_today,
                "is_lazy": _ncs_state.in_lazy_state,
                "is_burnthrough": _ncs_state.is_burnthrough_active,
                "phase_11": _ncs_state.current_phase.value,
                "sleep_stage": _ncs_state.sleep_stage,
                "daily_fatigue": round(_ncs_state.daily_fatigue, 1),
                "sleep_debt": round(_ncs_state.sleep_debt, 1),
                "monologue_mood": _ncs_state.monologue_mood_hint or "",
            }
        except Exception as _exc:
            logger.debug(f"非关键异常: {_exc}")
        try:
            from src.chat.heart_flow.rest_handler import get_dormancy_supervisor

            _dsup = get_dormancy_supervisor()
            _dorm = _dsup.fetch_dormancy(channel_id)
            if _dorm:
                _context["dormancy"] = {
                    "is_resting": True,
                    "cause": _dorm.cause or "",
                    "skip_streak": _dorm.skip_streak,
                    "remaining_sec": round(_dorm.remaining_sec(), 1),
                    "progress": round(_dorm.progress_ratio(), 3),
                    "glance_prob": round(_dorm.glance_probability, 3),
                }
            else:
                _context["dormancy"] = {"is_resting": False}
        except Exception as _exc:
            logger.debug(f"非关键异常: {_exc}")
        try:
            from src.chat.heart_flow.emotion_stream import get_channel_mood_tracker

            _mood = get_channel_mood_tracker()
            _mood_snap = _mood.get_current_mood(channel_id) if hasattr(_mood, "get_current_mood") else None
            if _mood_snap and isinstance(_mood_snap, dict):
                _context["emotion"] = {
                    "vitality": round(_mood_snap.get("vitality", 50.0), 1),
                    "weariness": round(_mood_snap.get("weariness", 20.0), 1),
                    "vexation": round(_mood_snap.get("vexation", 10.0), 1),
                }
            elif _mood_snap and hasattr(_mood_snap, "vitality"):
                _context["emotion"] = {
                    "vitality": round(getattr(_mood_snap, "vitality", 50.0), 1),
                    "weariness": round(getattr(_mood_snap, "weariness", 20.0), 1),
                    "vexation": round(getattr(_mood_snap, "vexation", 10.0), 1),
                }
        except Exception as _exc:
            logger.debug(f"非关键异常: {_exc}")
        return _context

    @staticmethod
    def _calc_interval_density(state) -> float:
        """计算消息间隔密度(越高=越密集=越吵)"""
        if state.night_disturbance_count <= 1:
            return 0.0
        elapsed = float(getattr(state, "last_disturbance_interval_sec", 0.0) or 0.0)
        if elapsed <= 0:
            return 35.0
        if elapsed <= 5.0:
            return 100.0
        if elapsed <= 15.0:
            return 80.0
        if elapsed <= 30.0:
            return 55.0
        if elapsed <= 60.0:
            return 35.0
        if elapsed <= 120.0:
            return 18.0
        return 8.0

    @staticmethod
    def _compute_stimulus_strength(context: dict, state) -> float:
        """计算归一化刺激强度(0-100)

        加权融合4个子系统的状态:
          - CDE三维引擎(40%): 困意×压力/储备 → 核心驱动力
          - D6能量系统(25%): chat/thinking池 + annoyance
          - 打扰模式(20%): 次数 + 间隔密度 + 连续睡眠次数
          - 情绪系统(15%): weariness/vexation提升刺激，vitality降低
        """
        _dist = context.get("disturbance", {})
        _energy = context.get("energy", {})
        _cde = context.get("cde_3d", {})
        _emotion = context.get("emotion", {})
        _dorm = context.get("dormancy", {})
        _score_cde = 0.0
        if _cde:
            _d = min(100.0, (_cde.get("drowsiness", 0) / max(_cde.get("collapse_threshold", 50), 1)) * 100)
            _p = min(100.0, _cde.get("pressure", 0))
            _s = max(1.0, _cde.get("reserve", 10))
            _composite = _cde.get("composite_load", 0)
            _collapse = max(_cde.get("collapse_threshold", 50), 1)
            _base = min(100.0, (_composite / _collapse) * 80)
            _irritation = min(50.0, _cde.get("wake_irritation", 0) * 30)
            _suppression = _cde.get("response_suppression", 0)
            _fatigue = min(40.0, _cde.get("daily_fatigue", 0) * 0.4)
            _int_count = min(30.0, _cde.get("interruption_count", 0) * 3)
            _score_cde = _base + _irritation + _fatigue + _int_count - (_suppression * 15)
            if _cde.get("collapse_imminent"):
                _score_cde += 20.0
            if _cde.get("is_burnthrough"):
                _score_cde -= 15.0
        _score_energy = 0.0
        if _energy:
            _chat_r = _energy.get("chat_ratio", 0.5)
            _think_r = _energy.get("thinking_ratio", 0.5)
            _combined = (_chat_r + _think_r) / 2.0
            _annoy = min(30.0, _energy.get("annoyance", 0))
            _activity = _energy.get("activity", 50)
            _exhausted_bonus = 25.0 if _energy.get("is_exhausted") else 0.0
            _night_used_ratio = 0.0
            _cap = _energy.get("night_replies_cap", 3)
            _used = _energy.get("night_replies_used", 0)
            if _cap > 0:
                _night_used_ratio = _used / _cap
            _score_energy = ((1.0 - _combined) * 35) + _annoy + _exhausted_bonus + (_night_used_ratio * 20)
            if _activity > 70:
                _score_energy += (_activity - 70) * 0.3
        _score_disturb = 0.0
        if _dist:
            _count = _dist.get("count", 0)
            _density = _dist.get("interval_density", 0.0)
            _consec_sleep = getattr(state, "consecutive_sleep_decisions", 0)
            _count_score = min(35.0, _count * 3.5 + (_count**1.3) * 0.8)
            _density_score = _density * 0.22
            _consec_boost = min(20.0, _consec_sleep * 4.0)
            _score_disturb = _count_score + _density_score + _consec_boost
        _score_emotion = 0.0
        if _emotion:
            _w = _emotion.get("weariness", 20)
            _v = _emotion.get("vitality", 50)
            _x = _emotion.get("vexation", 10)
            _score_emotion = (_w * 0.45) + (_x * 0.55) - max(0, (_v - 60) * 0.15)
        _score_dorm = 0.0
        if _dorm and _dorm.get("is_resting"):
            _skip = _dorm.get("skip_streak", 0)
            _glance = _dorm.get("glance_prob", 0)
            _progress = _dorm.get("progress", 0)
            _score_dorm = (_skip * 3.0) + (_glance * 15) + ((1.0 - _progress) * 12)
        _total = (
            _score_cde * 0.38
            + _score_energy * 0.24
            + _score_disturb * 0.23
            + _score_emotion * 0.10
            + _score_dorm * 0.05
        )
        return max(0.0, min(100.0, _total))

    def apply_wake_feedback(self, channel_id: str, action: str, desire: int = 0, stimulus: float = 0.0):
        """根据决策结果在线学习调整动态阈值

        反馈规则:
        ┌──────────────┬──────────┬─────────────────────────────────┐
        │ 决策结果      │ 条件     │ 阈值调整                       │
        ├──────────────┼──────────┼─────────────────────────────────┤
        │ sleep_resist │ desire<3 │ ↑ 变难醒 (不该被轻易叫醒)       │
        │ sleep_resist │ desire≥7 │ ↓ 变易醒 (模型想回但没过阈值)   │
        │ grumpy_glance│ desire≤4 │ ↑ 微升   (不情愿的瞥眼)         │
        │ soft_wake    │ desire≤5 │ ↑ 微升   (醒了但不太想回)       │
        │ soft_wake    │ desire≥7 │ ↓ 微降   (正确唤醒且积极)       │
        │ full_wake    │ desire≥8 │ ↓↓ 降明显 (完全正确的唤醒)      │
        │ full_wake    │ desire≤4 │ ↑↑ 升明显 (误唤醒)              │
        │ 管理理强制     │ -        │ ↓↓ 降明显 (确实有事)           │
        └──────────────┴──────────┴─────────────────────────────────┘

        学习率动态调节:
          - 连续多次同方向决策 → 学习率衰减(避免震荡)
          - 阈值接近边界(min/max) → 学习率衰减
          - 刺激强度远超/远低于阈值 → 大幅调整
        """
        state = self._ensure_channel(channel_id)
        _lr = state.wake_learning_rate
        _history_entry = {
            "action": action,
            "desire": desire,
            "stimulus": round(stimulus, 1),
            "threshold_before": round(state.wake_adaptive_threshold, 1),
            "time": time.time(),
        }
        state.wake_decision_history.append(_history_entry)
        if len(state.wake_decision_history) > 30:
            state.wake_decision_history = state.wake_decision_history[-20:]
        if action == "force_wake_admin":
            _delta = -_lr * 1.8
            state.consecutive_sleep_decisions = 0
            state.consecutive_wake_decisions += 1
        elif action in ("deep_sleep", "sleep_resist"):
            state.consecutive_sleep_decisions += 1
            state.consecutive_wake_decisions = 0
            if desire >= 7:
                _delta = -_lr * 0.6
            elif desire >= 4:
                _delta = _lr * 0.3
            else:
                _consec = state.consecutive_sleep_decisions
                if _consec >= 5:
                    _delta = _lr * 1.5
                    _lr *= 0.85
                elif _consec >= 3:
                    _delta = _lr * 0.9
                else:
                    _delta = _lr * 0.5
        elif action == "grumpy_glance":
            state.consecutive_sleep_decisions = 0
            state.consecutive_wake_decisions += 1
            if desire <= 4:
                _delta = _lr * 0.4
            else:
                _delta = -_lr * 0.3
        elif action in ("soft_wake", "full_wake"):
            state.consecutive_sleep_decisions = 0
            state.consecutive_wake_decisions += 1
            if action == "full_wake" and desire >= 8:
                _delta = -_lr * 1.4
            elif action == "full_wake" and desire <= 4:
                _delta = _lr * 1.2
            elif desire >= 7:
                _delta = -_lr * 0.6
            elif desire <= 4:
                _delta = _lr * 0.5
            else:
                _delta = 0.0
        else:
            _delta = 0.0
        if stimulus > 0 and abs(stimulus - state.wake_adaptive_threshold) > 30:
            _delta *= 1.4
        _new_thresh = state.wake_adaptive_threshold + _delta
        _new_thresh = max(state.wake_min_threshold, min(state.wake_max_threshold, _new_thresh))
        state.wake_adaptive_threshold = _new_thresh
        state.wake_learning_rate = max(1.5, min(12.0, _lr))
        _history_entry["threshold_after"] = round(_new_thresh, 1)
        _history_entry["delta"] = round(_delta, 1)

    def reset_disturbance(self, channel_id: str):
        """重置夜间打扰计数和连续决策计数"""
        state = self._ensure_channel(channel_id)
        state.night_disturbance_count = 0
        state.last_disturbance_time = 0.0
        state.prev_disturbance_time = 0.0
        state.last_disturbance_interval_sec = 0.0
        state.last_stimulus_strength = 0.0

    def vote(self, ctx):
        if not ctx.channel_id:
            return EnergyChainVote()
        state = self._ensure_channel(ctx.channel_id)
        chat_ratio = state.chat_ratio()
        think_ratio = state.thinking_ratio()
        combined = min(chat_ratio, think_ratio)
        stage = self._classify_stage(state)
        night_mode = self._get_night_mode(ctx.channel_id)
        prob_table = {
            EnergyStage.DEPLETED: 0.05,
            EnergyStage.CRITICAL: 0.30,
            EnergyStage.LOW: 0.60 + combined * 0.50,
            EnergyStage.ADEQUATE: 0.85 + combined * 0.15,
            EnergyStage.FULL: 1.0,
        }
        prob = prob_table.get(stage, 0.5)
        if night_mode.phase != "DAYTIME":
            prob *= night_mode.prob_multiplier
        force_refuse = state.is_exhausted()
        estimated_cost = max(0.01, min(0.20, chat_ratio * 0.08))
        recovery_damping = max(0.20, chat_ratio * 0.60 + 0.30)
        tokens_cap = 0
        if stage == EnergyStage.CRITICAL:
            tokens_cap = 150
        elif stage == EnergyStage.LOW:
            tokens_cap = 350
        perception_parts = []
        if night_mode.perception_label:
            perception_parts.append(night_mode.perception_label)
        if stage == EnergyStage.DEPLETED:
            perception_parts.append("精力彻底耗尽")
        elif stage == EnergyStage.CRITICAL:
            perception_parts.append(f"聊天值仅剩{chat_ratio * 100:.0f}%")
        elif state.boost_active:
            perception_parts.append(f"受到{state.boost_source}激励")
        if state.annoyance_level > 30:
            perception_parts.append(f"烦躁度{state.annoyance_level:.0f}")
        perception_label = "，".join(perception_parts) if perception_parts else ""
        return EnergyChainVote(
            probability_factor=max(0.05, prob),
            force_refuse=force_refuse,
            attitude_tag="exhausted" if force_refuse else "",
            max_tokens_cap=tokens_cap,
            chat_energy_ratio=round(chat_ratio, 3),
            thinking_energy_ratio=round(think_ratio, 3),
            energy_stage=stage,
            estimated_cost=round(estimated_cost, 4),
            night_mode=(night_mode.phase != "DAYTIME"),
            recovery_damping=round(recovery_damping, 3),
            chat_ceiling=round(state.chat_ceiling * (1.0 - night_mode.ceiling_penalty), 1),
            chat_current=round(state.chat_pool, 1),
            debug_reason=f"night={night_mode.phase} percep={perception_label or '-'}",
        )

    def _classify_stage(self, state):
        combined = state.combined_ratio()
        if combined > 0.80:
            return EnergyStage.FULL
        elif combined > 0.50:
            return EnergyStage.ADEQUATE
        elif combined > 0.20:
            return EnergyStage.LOW
        elif combined > 0.05:
            return EnergyStage.CRITICAL
        else:
            return EnergyStage.DEPLETED

    def capture_snapshot(self, channel_id: str):
        return self._ensure_channel(channel_id).to_snapshot()

    def apply_special_boost(self, channel_id: str, source: str, amount: float = 15.0, duration_sec: float = 300.0):
        """特殊事件充电池"""
        state = self._ensure_channel(channel_id)
        state.chat_pool = min(state.chat_ceiling, state.chat_pool + amount)
        state.thinking_value = min(state.thinking_ceiling, state.thinking_value + amount * 0.6)
        state.boost_active = True
        state.boost_source = source
        state.boost_expire = time.time() + duration_sec
        logger.info(f"🔋 D6 充电: {source} +{amount:.1f} ch={channel_id[:8]}")

    def serialize(self) -> dict:
        result = {}
        for cid, state in self._channels.items():
            result[cid] = {
                "chat_pool": round(state.chat_pool, 2),
                "chat_ceiling": round(state.chat_ceiling, 2),
                "thinking_value": round(state.thinking_value, 2),
                "thinking_ceiling": round(state.thinking_ceiling, 2),
                "annoyance_level": round(state.annoyance_level, 2),
                "activity_level": round(state.activity_level, 2),
                "social_value": round(state.social_value, 2),
                "chain_count": state.chain_count,
                "night_reply_count": state.night_reply_count,
                "total_consumed_today": round(state.total_consumed_today, 2),
                "night_disturbance_count": state.night_disturbance_count,
                "prev_disturbance_time": round(state.prev_disturbance_time, 3),
                "last_disturbance_interval_sec": round(state.last_disturbance_interval_sec, 3),
                "wake_adaptive_threshold": round(state.wake_adaptive_threshold, 1),
                "wake_learning_rate": round(state.wake_learning_rate, 1),
                "consecutive_sleep_decisions": state.consecutive_sleep_decisions,
                "consecutive_wake_decisions": state.consecutive_wake_decisions,
            }
        return result

    def deserialize(self, data: dict):
        if not isinstance(data, dict):
            return
        for cid, vals in data.items():
            if not isinstance(vals, dict):
                continue
            state = self._ensure_channel(cid)
            state.chat_pool = min(float(vals.get("chat_pool", 100.0)), state.chat_ceiling)
            state.thinking_value = min(float(vals.get("thinking_value", 100.0)), state.thinking_ceiling)
            state.annoyance_level = max(0.0, float(vals.get("annoyance_level", 0.0)))
            state.activity_level = max(0.0, min(100.0, float(vals.get("activity_level", 50.0))))
            state.social_value = max(-50.0, min(100.0, float(vals.get("social_value", 0.0))))
            state.chain_count = int(vals.get("chain_count", 0))
            state.night_reply_count = int(vals.get("night_reply_count", 0))
            state.night_disturbance_count = int(vals.get("night_disturbance_count", 0))
            state.prev_disturbance_time = float(vals.get("prev_disturbance_time", 0.0) or 0.0)
            state.last_disturbance_interval_sec = max(
                0.0, float(vals.get("last_disturbance_interval_sec", 0.0) or 0.0)
            )
            _saved_thresh = vals.get("wake_adaptive_threshold")
            if _saved_thresh is not None:
                state.wake_adaptive_threshold = max(
                    state.wake_min_threshold, min(state.wake_max_threshold, float(_saved_thresh))
                )
            _saved_lr = vals.get("wake_learning_rate")
            if _saved_lr is not None:
                state.wake_learning_rate = max(1.5, min(12.0, float(_saved_lr)))
            state.consecutive_sleep_decisions = int(vals.get("consecutive_sleep_decisions", 0))
            state.consecutive_wake_decisions = int(vals.get("consecutive_wake_decisions", 0))

    def calibrate(self, offline_seconds: float):
        offline_min = offline_seconds / 60.0
        recovery_cap = 0.85
        ratio = min(1.0, offline_min / 180.0)
        for ch_id, state in self._channels.items():
            recovery = offline_min * 2.5 * ratio * recovery_cap
            state.chat_pool = min(state.chat_ceiling, state.chat_pool + recovery)
            state.thinking_value = min(state.thinking_ceiling, state.thinking_value + recovery * 0.6)
            state.annoyance_level = max(0.0, state.annoyance_level - offline_min * 3.0)
            state.last_update = time.time()
            state.boost_active = False
            state.night_reply_count = 0
            state.total_consumed_today = 0.0
            try:
                from src.core.night_cycle_system import get_night_cycle

                _ncs_cal = get_night_cycle(ch_id)
                _ncs_cal.reset_night_counters()
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")

    def get_state_summary(self) -> dict:
        summaries = {}
        for cid, state in self._channels.items():
            summaries[cid[:8]] = {
                "chat_ratio": round(state.chat_ratio(), 2),
                "think_ratio": round(state.thinking_ratio(), 2),
                "combined": round(state.combined_ratio(), 2),
                "annoyance": round(state.annoyance_level, 1),
                "exhausted": state.is_exhausted(),
                "chain_count": state.chain_count,
                "activity": round(state.activity_level, 1),
                "social": round(state.social_value, 1),
                "night_replies": state.night_reply_count,
                "boost": state.boost_active,
            }
        return {
            "dimension": self.dimension_name,
            "scope": self.scope.value if hasattr(self.scope, "value") else "per_channel",
            "channel_count": len(self._channels),
            "channels": summaries,
        }

    def reset(self, user_id: str = "", channel_id: str = ""):
        if channel_id and channel_id in self._channels:
            del self._channels[channel_id]

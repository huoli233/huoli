import time
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from src.common.logger import get_logger

logger = get_logger("情感反馈")

_feedback_singleton = None


class FeedbackSource(Enum):
    """反馈来源"""

    ACFN_VERDICT = "acfn_verdict"
    SITUATION_INTERPRETATION = "situation_interpretation"
    SAFETY_ASSESSMENT = "safety_assessment"
    MULTIMODAL_BRIDGE = "multimodal_bridge"
    GOSSIP_RITUAL = "gossip_ritual"
    VISIBILITY_SCORER = "visibility_scorer"
    FRESHNESS_DECAY = "freshness_decay"
    MEMORY_GOVERNANCE = "memory_governance"
    NIGHT_CYCLE = "night_cycle"
    METABOLISM = "metabolism"
    USER_INTERACTION = "user_interaction"
    PROACTIVE_OUTCOME = "proactive_outcome"

    def label(self) -> str:
        return {
            "acfn_verdict": "ACFN裁决",
            "situation_interpretation": "情境解释",
            "safety_assessment": "安全评估",
            "multimodal_bridge": "多模态桥接",
            "gossip_ritual": "吃瓜策略",
            "visibility_scorer": "可见性评分",
            "freshness_decay": "新鲜度衰减",
            "memory_governance": "记忆治理",
            "night_cycle": "昼夜周期",
            "metabolism": "代谢引擎",
            "user_interaction": "用户交互",
            "proactive_outcome": "主动行为结果",
        }.get(self.value, "")


@dataclass
class EmotionDelta:
    """单条情感增量——来自某个引擎的反馈"""

    source: FeedbackSource
    mood_delta: float = 0.0
    energy_delta: float = 0.0
    boredom_delta: float = 0.0
    social_desire_delta: float = 0.0
    curiosity_delta: float = 0.0
    loneliness_delta: float = 0.0
    confidence: float = 0.5
    reason: str = ""
    applied_at: float = field(default_factory=time.time)
    is_strong_signal: bool = False

    def total_magnitude(self) -> float:
        return (
            abs(self.mood_delta)
            + abs(self.energy_delta)
            + abs(self.boredom_delta)
            + abs(self.social_desire_delta)
            + abs(self.curiosity_delta)
            + abs(self.loneliness_delta)
        ) / 6.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.label(),
            "mood": f"{self.mood_delta:+.3f}",
            "energy": f"{self.energy_delta:+.3f}",
            "boredom": f"{self.boredom_delta:+.3f}",
            "social": f"{self.social_desire_delta:+.3f}",
            "curiosity": f"{self.curiosity_delta:+.3f}",
            "reason": self.reason[:50],
            "strong": self.is_strong_signal,
        }


@dataclass
class EmotionStateSnapshot:
    """情感状态快照（回写目标）"""

    channel_id: str = ""
    mood: float = 0.5
    energy: float = 1.0
    boredom: float = 0.0
    social_desire: float = 0.5
    curiosity: float = 0.3
    loneliness: float = 0.0
    proactive_willingness: float = 0.0
    feeling_description: str = ""
    last_update: float = field(default_factory=time.time)
    update_count: int = 0

    def clamp(self) -> None:
        self.mood = max(0.0, min(1.0, self.mood))
        self.energy = max(0.05, min(1.0, self.energy))
        self.boredom = max(0.0, min(1.0, self.boredom))
        self.social_desire = max(0.0, min(1.0, self.social_desire))
        self.curiosity = max(0.0, min(1.0, self.curiosity))
        self.loneliness = max(0.0, min(1.0, self.loneliness))
        self.proactive_willingness = max(
            0.0, min(1.0, self.proactive_willingness)
        )

    def compute_proactive_willingness(self) -> float:
        base = (
            self.social_desire * 0.30
            + self.curiosity * 0.20
            + self.boredom * 0.20
            + (1.0 - self.energy) * -0.10
            + self.loneliness * 0.15
            + (self.mood - 0.5) * 0.05
        )
        return max(0.0, min(1.0, base))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mood": round(self.mood, 3),
            "energy": round(self.energy, 3),
            "boredom": round(self.boredom, 3),
            "social_desire": round(self.social_desire, 3),
            "curiosity": round(self.curiosity, 3),
            "loneliness": round(self.loneliness, 3),
            "proactive": round(self.proactive_willingness, 3),
            "feeling": self.feeling_description[:60],
        }


@dataclass
class FeedbackCycleReport:
    """一轮反馈周期的报告"""

    cycle_id: str = ""
    deltas_applied: int = 0
    total_magnitude: float = 0.0
    state_before: Optional[EmotionStateSnapshot] = None
    state_after: Optional[EmotionStateSnapshot] = None
    dominant_source: Optional[FeedbackSource] = None
    strong_signals: List[EmotionDelta] = field(default_factory=list)
    sources_contributed: List[str] = field(default_factory=list)
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle_id[:12],
            "deltas": self.deltas_applied,
            "magnitude": round(self.total_magnitude, 4),
            "dominant": (
                self.dominant_source.label() if self.dominant_source else ""
            ),
            "strong_signals": len(self.strong_signals),
            "sources": self.sources_contributed,
            "mood_change": (
                f"{self.state_after.mood - self.state_before.mood:+.3f}"
                if self.state_after and self.state_before
                else "?"
            ),
        }


# 反馈强度常量
_STRONG_THRESHOLD = 0.08
_MAX_DELTA_PER_SOURCE = 0.15
_FEEDBACK_DECAY_HALF_LIFE = 180.0
_MAX_DELTAS_PER_CYCLE = 20


class EmotionFeedbackLoop:
    """情感驱动核心反馈环
    将所有GAP引擎的输出回写到 emotion_driven_core 的状态中，
    形成闭环：引擎决策→行为执行→情感演化→影响下一轮决策。

    反馈来源（12种）：
    - ACFN裁决：参与度/抑制/激活值 → mood/energy
    - 情境解释：风险/氛围 → curiosity/social_desire
    - 安全评估：威胁等级 → energy下降/boredom上升
    - 多模态桥接：情感极性 → mood微调
    - 吃瓜策略：参与姿态 → social_desire
    - 可见性评分：关注分布 → boredom/curiosity
    - 新鲜度衰减：过期消息量 → loneliness
    - 记忆治理：淘汰事件 → 轻微mood波动
    - 昼夜周期：夜间阶段 → energy/mood压制
    - 代谢引擎：精力比/摸鱼值 → 直接写入energy/boredom
    - 用户交互：@/引用/对话 → mood提升
    - 主动行为结果：成功/失败 → proactive_willingness调整
    """

    def __init__(self):
        self._states: Dict[str, EmotionStateSnapshot] = {}
        self._delta_buffer: List[EmotionDelta] = []
        self._max_buffer = _MAX_DELTAS_PER_CYCLE
        self._cycle_history: List[FeedbackCycleReport] = []
        self._max_history = 30
        self._last_cycle_ts: Dict[str, float] = {}
        self._min_cycle_interval_sec = 5.0

    # ═════════════════ 反馈注册接口 ═════════════════

    def submit_delta(
        self,
        channel_id: str,
        source: FeedbackSource,
        *,
        mood: float = 0.0,
        energy: float = 0.0,
        boredom: float = 0.0,
        social_desire: float = 0.0,
        curiosity: float = 0.0,
        loneliness: float = 0.0,
        confidence: float = 0.5,
        reason: str = "",
        is_strong: bool = False,
    ) -> EmotionDelta:
        delta = EmotionDelta(
            source=source,
            mood_delta=max(
                -_MAX_DELTA_PER_SOURCE, min(_MAX_DELTA_PER_SOURCE, mood)
            ),
            energy_delta=max(
                -_MAX_DELTA_PER_SOURCE, min(_MAX_DELTA_PER_SOURCE, energy)
            ),
            boredom_delta=max(
                -_MAX_DELTA_PER_SOURCE, min(_MAX_DELTA_PER_SOURCE, boredom)
            ),
            social_desire_delta=max(
                -_MAX_DELTA_PER_SOURCE,
                min(_MAX_DELTA_PER_SOURCE, social_desire),
            ),
            curiosity_delta=max(
                -_MAX_DELTA_PER_SOURCE, min(_MAX_DELTA_PER_SOURCE, curiosity)
            ),
            loneliness_delta=max(
                -_MAX_DELTA_PER_SOURCE, min(_MAX_DELTA_PER_SOURCE, loneliness)
            ),
            confidence=max(0.1, min(1.0, confidence)),
            reason=reason,
            is_strong_signal=is_strong
            or abs(mood) > _STRONG_THRESHOLD
            or abs(energy) > _STRONG_THRESHOLD,
        )
        self._delta_buffer.append(delta)
        if len(self._delta_buffer) > self._max_buffer:
            self._delta_buffer = self._delta_buffer[-self._max_buffer:]
        return delta

    def submit_from_acfn(
        self,
        channel_id: str,
        *,
        fused_score: float,
        activation_bar: float,
        intent_override: float = 0.0,
        should_proceed: bool = True,
    ) -> EmotionDelta:
        if not should_proceed:
            return self.submit_delta(
                channel_id,
                FeedbackSource.ACFN_VERDICT,
                mood=-0.03,
                energy=-0.02,
                boredom=0.04,
                confidence=0.8,
                reason="ACFN判定不回复，轻微挫败感",
            )
        score_mod = (fused_score - 0.5) * 0.08
        bar_mod = (activation_bar - 0.5) * -0.04
        intent_mod = (intent_override - 0.5) * 0.06
        return self.submit_delta(
            channel_id,
            FeedbackSource.ACFN_VERDICT,
            mood=score_mod + bar_mod,
            energy=score_mod * 0.6,
            social_desire=intent_mod,
            curiosity=fused_score * 0.03,
            confidence=0.85,
            reason=f"ACFN: score={fused_score:.2f} bar={activation_bar:.2f}",
            is_strong=fused_score > 0.7 or activation_bar > 0.7,
        )

    def submit_from_situation(
        self,
        channel_id: str,
        *,
        risk_score: float = 0.0,
        atmosphere: str = "",
        engagement_intensity: float = 0.5,
        internal_energy: float = 0.5,
    ) -> EmotionDelta:
        risk_mod = -risk_score * 0.12
        eng_mod = engagement_intensity * 0.06
        energy_mod = (internal_energy - 0.5) * 0.04
        atm_bonus = 0.0
        atm_lower = atmosphere.lower()
        if "explosive" in atm_lower:
            atm_bonus = -0.03
        elif "calm" in atm_lower:
            atm_bonus = 0.02
        elif "active" in atm_lower:
            atm_bonus = 0.03
        return self.submit_delta(
            channel_id,
            FeedbackSource.SITUATION_INTERPRETATION,
            mood=risk_mod + atm_bonus,
            curiosity=eng_mod,
            social_desire=engagement_intensity * 0.04,
            energy=energy_mod,
            confidence=0.75,
            reason=f"情境: risk={risk_score:.2f}, atm={atmosphere}",
            is_strong=risk_score > 0.4,
        )

    def submit_from_safety(
        self,
        channel_id: str,
        *,
        safety_level: str = "",
        safety_score: float = 0.0,
        blocked: bool = False,
    ) -> EmotionDelta:
        level_penalty = {
            "low": -0.01,
            "medium": -0.04,
            "high": -0.10,
            "critical": -0.22,
        }
        mod = level_penalty.get(safety_level.lower(), 0.0)
        if blocked:
            mod -= 0.05
        return self.submit_delta(
            channel_id,
            FeedbackSource.SAFETY_ASSESSMENT,
            mood=mod * 0.6,
            energy=mod * 0.4,
            boredom=max(0, -mod) * 0.15,
            confidence=0.9,
            reason=f"安全: {safety_level}(score={safety_score:.2f})",
            is_strong=safety_level in ("high", "critical"),
        )

    def submit_from_multimodal(
        self,
        channel_id: str,
        *,
        avg_valence: float = 0.0,
        avg_engagement: float = 0.0,
        has_nsfw: bool = False,
        group_sticker_wave: bool = False,
    ) -> EmotionDelta:
        mood_mod = avg_valence * 0.06
        engage_mod = avg_engagement * 0.04
        if has_nsfw:
            mood_mod -= 0.08
        if group_sticker_wave:
            engage_mod += 0.03
        return self.submit_delta(
            channel_id,
            FeedbackSource.MULTIMODAL_BRIDGE,
            mood=mood_mod,
            curiosity=engage_mod,
            social_desire=engage_mod * 0.5,
            confidence=0.55,
            reason=f"多模态: val={avg_valence:.2f} eng={avg_engagement:.2f}",
        )

    def submit_from_metabolism(
        self,
        channel_id: str,
        *,
        energy_ratio: float = 1.0,
        boredom_level: float = 0.0,
        loafing_level: float = 0.0,
    ) -> EmotionDelta:
        e_mod = (energy_ratio - 0.7) * 0.08
        b_mod = (boredom_level - 0.3) * 0.10
        l_mod = loafing_level * 0.06
        return self.submit_delta(
            channel_id,
            FeedbackSource.METABOLISM,
            energy=e_mod,
            boredom=b_mod,
            social_desire=-l_mod,
            confidence=0.95,
            reason=f"代谢: e={
                energy_ratio:.2f} b={
                boredom_level:.2f} l={
                loafing_level:.2f}",
            is_strong=boredom_level > 0.6 or energy_ratio < 0.25,
        )

    def submit_from_night(
        self,
        channel_id: str,
        *,
        night_phase: str = "",
        pressure_total: float = 0.0,
        is_burnthrough: bool = False,
    ) -> EmotionDelta:
        phase_mod_map = {
            "awake": 0.0,
            "active": 0.0,
            "winding_down": -0.03,
            "drowsy": -0.08,
            "light_sleep": -0.18,
            "deep_sleep": -0.35,
        }
        pm = phase_mod_map.get(night_phase.lower(), -0.05)
        if is_burnthrough:
            pm += 0.15
        return self.submit_delta(
            channel_id,
            FeedbackSource.NIGHT_CYCLE,
            mood=pm * 0.6,
            energy=pm * 0.8,
            curiosity=-abs(pm) * 0.3,
            confidence=0.9,
            reason=f"夜间: {night_phase}, 压力={pressure_total:.1f}",
            is_strong=night_phase in ("deep_sleep", "burnthrough"),
        )

    def submit_from_user_interaction(
        self,
        channel_id: str,
        *,
        is_mentioned: bool = False,
        is_question: bool = False,
        is_first_contact: bool = False,
        message_length: int = 0,
        repeat_count: int = 0,
    ) -> EmotionDelta:
        _mood_boost = 0.015
        if is_mentioned:
            _mood_boost += 0.025
        if is_question:
            _mood_boost += 0.01
        if is_first_contact:
            _mood_boost += 0.02
        if message_length > 10:
            _mood_boost += min(0.015, message_length * 0.001)
        if repeat_count > 0:
            _mood_boost -= min(0.04, repeat_count * 0.015)
        return self.submit_delta(
            channel_id,
            FeedbackSource.USER_INTERACTION,
            mood=_mood_boost,
            energy=0.01 + (0.02 if is_mentioned else 0),
            curiosity=0.02 if is_question else 0.005,
            social_desire=0.015 if not repeat_count else -0.02,
            confidence=0.85,
            reason=f"用户交互: mentioned={is_mentioned} question={is_question} repeat={repeat_count}",
            is_strong=is_mentioned or is_first_contact,
        )

    # ═════════════════ 周期执行 ═════════════════

    def run_feedback_cycle(self, channel_id: str) -> FeedbackCycleReport:
        now = time.time()
        last = self._last_cycle_ts.get(channel_id, 0.0)
        if now - last < self._min_cycle_interval_sec:
            snap = self._get_or_create_state(channel_id)
            return FeedbackCycleReport(
                cycle_id="skipped",
                state_after=snap,
                computed_at=now,
            )
        self._last_cycle_ts[channel_id] = now
        state = self._get_or_create_state(channel_id)
        before = EmotionStateSnapshot(
            **{
                k: v
                for k, v in state.__dict__.items()
                if not k.startswith("_")
            },
        )
        before.last_update = state.last_update
        before.update_count = state.update_count
        deltas = list(self._delta_buffer)
        self._delta_buffer.clear()
        if not deltas:
            return FeedbackCycleReport(
                cycle_id="empty",
                state_before=before,
                state_after=state,
                computed_at=now,
            )
        total_mag = 0.0
        strong_list: List[EmotionDelta] = []
        src_set: Set[str] = set()
        best_src: Optional[FeedbackSource] = None
        best_mag = 0.0
        for d in deltas:
            w = d.confidence * (1.5 if d.is_strong_signal else 1.0)
            age_factor = math.exp(
                -(now - d.applied_at) / _FEEDBACK_DECAY_HALF_LIFE
            )
            effective_w = w * age_factor
            state.mood += d.mood_delta * effective_w
            state.energy += d.energy_delta * effective_w
            state.boredom += d.boredom_delta * effective_w
            state.social_desire += d.social_desire_delta * effective_w
            state.curiosity += d.curiosity_delta * effective_w
            state.loneliness += d.loneliness_delta * effective_w
            mag = d.total_magnitude() * effective_w
            total_mag += mag
            if d.is_strong_signal:
                strong_list.append(d)
            src_set.add(d.source.value)
            if mag > best_mag:
                best_mag = mag
                best_src = d.source
        state.clamp()
        state.proactive_willingness = state.compute_proactive_willingness()
        state.feeling_description = self._generate_feeling(state)
        state.last_update = now
        state.update_count += 1
        report = FeedbackCycleReport(
            cycle_id=f"{channel_id[:8]}:{int(now)}",
            deltas_applied=len(deltas),
            total_magnitude=total_mag,
            state_before=before,
            state_after=state,
            dominant_source=best_src,
            strong_signals=strong_list,
            sources_contributed=list(src_set),
            computed_at=now,
        )
        self._cycle_history.append(report)
        if len(self._cycle_history) > self._max_history:
            self._cycle_history = self._cycle_history[-self._max_history:]
        logger.info(
            f"[情感反馈] {channel_id[:12]}: {len(deltas)}条增量 | "
            f"mood {before.mood:.3f}→{state.mood:.3f}, "
            f"energy {before.energy:.3f}→{state.energy:.3f}, "
            f"主动意愿 {state.proactive_willingness:.3f}"
        )
        return report

    def get_state(self, channel_id: str) -> EmotionStateSnapshot:
        return self._get_or_create_state(channel_id)

    def get_recent_cycles(self, limit: int = 5) -> List[FeedbackCycleReport]:
        return list(reversed(self._cycle_history[-limit:]))

    # ────────────────── 内部方法 ──────────────────

    def _get_or_create_state(self, channel_id: str) -> EmotionStateSnapshot:
        if channel_id not in self._states:
            self._states[channel_id] = EmotionStateSnapshot(
                channel_id=channel_id
            )
        return self._states[channel_id]

    @staticmethod
    def _generate_feeling(snap: EmotionStateSnapshot) -> str:
        parts: List[str] = []
        if snap.mood > 0.7:
            parts.append("心情不错")
        elif snap.mood < 0.3:
            parts.append("有点低落")
        if snap.energy < 0.3:
            parts.append("很累")
        elif snap.energy > 0.8:
            parts.append("精力充沛")
        if snap.boredom > 0.6:
            parts.append("有点无聊")
        if snap.loneliness > 0.5:
            parts.append("有点孤独")
        if snap.social_desire > 0.7:
            parts.append("想找人聊聊")
        elif snap.social_desire < 0.2:
            parts.append("不太想社交")
        if snap.curiosity > 0.6:
            parts.append("对周围很好奇")
        if not parts:
            parts.append("一切如常")
        return "，".join(parts)


def get_emotion_feedback_loop() -> EmotionFeedbackLoop:
    global _feedback_singleton
    if _feedback_singleton is None:
        _feedback_singleton = EmotionFeedbackLoop()
    return _feedback_singleton

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger
from src.common.singleton import MultiInstanceManager

logger = get_logger("attention_flow")

_flow_singleton = None

_attention_flow_manager = MultiInstanceManager()


class AttentionState(Enum):
    """主观注意力状态"""

    FULLY_ENGAGED = "fully_engaged"
    SELECTIVE_ATTENTION = "selective_attention"
    CASUAL_GLANCE = "casual_glance"
    PEEKING = "peeking"
    WITHDRAWN = "withdrawn"
    DEEP_WITHDRAWAL = "deep_withdrawal"

    def label(self) -> str:
        return {
            "fully_engaged": "全神贯注",
            "selective_attention": "选择性关注",
            "casual_glance": "随意扫视",
            "peeking": "偶尔窥屏",
            "withdrawn": "暂时退出",
            "deep_withdrawal": "深度退出",
        }.get(self.value, "")

    def visibility_threshold(self) -> float:
        return {
            "fully_engaged": 0.05,
            "selective_attention": 0.18,
            "casual_glance": 0.35,
            "peeking": 0.55,
            "withdrawn": 0.75,
            "deep_withdrawal": 0.92,
        }.get(self.value, 0.50)

    def message_process_ratio(self) -> float:
        return {
            "fully_engaged": 1.0,
            "selective_attention": 0.65,
            "casual_glance": 0.30,
            "peeking": 0.12,
            "withdrawn": 0.03,
            "deep_withdrawal": 0.00,
        }.get(self.value, 0.10)


class AttentionTransitionReason(Enum):
    """状态转换原因"""

    BOREDOM_SPIKE = "boredom_spike"
    CURIOSITY_TRIGGER = "curiosity_trigger"
    SOCIAL_DESIRE_RISE = "social_desire_rise"
    FATIGUE_SETTLING = "fatigue_settling"
    OVERSTIMULATION = "overstimulation"
    NIGHT_FALL = "night_fall"
    USER_RETURN = "user_return"
    SELF_DECIDED_PEEK = "self_decided_peek"
    NOTHING_INTERESTING = "nothing_interesting"
    EMOTIONAL_OVERLOAD = "emotional_overload"
    SAFETY_RETREAT = "safety_retreat"
    PROACTIVE_AFTERMATH = "proactive_aftermath"

    def label(self) -> str:
        return {
            "boredom_spike": "无聊激增，想看看有没有动静",
            "curiosity_trigger": "好奇心被触发",
            "social_desire_rise": "想社交了",
            "fatigue_settling": "累了想安静",
            "overstimulation": "刺激太多需要退一退",
            "night_fall": "夜深了",
            "user_return": "熟悉的人出现了",
            "self_decided_peek": "自己想瞄一眼",
            "nothing_interesting": "看了没意思又关掉",
            "emotional_overload": "情绪波动太大先撤",
            "safety_retreat": "安全机制触发撤退",
            "proactive_aftermath": "主动行为后休息",
        }.get(self.value, "")


@dataclass
class AttentionFlowSnapshot:
    """注意力流快照"""

    channel_id: str = ""
    state: AttentionState = AttentionState.SELECTIVE_ATTENTION
    previous_state: AttentionState = AttentionState.SELECTIVE_ATTENTION
    transition_reason: Optional[AttentionTransitionReason] = None
    openness: float = 0.5
    peek_desire: float = 0.3
    silence_tolerance: float = 0.5
    interrupt_tolerance: float = 0.5
    curiosity_level: float = 0.3
    social_hunger: float = 0.5
    withdrawal_depth: float = 0.0
    look_budget_state: float = 0.0
    process_budget_state: float = 0.0
    time_since_last_look: float = 0.0
    time_since_last_act: float = 0.0
    consecutive_peeks_without_action: int = 0
    state_entered_at: float = field(default_factory=time.time)
    last_transition_at: float = field(default_factory=time.time)
    transition_history: List[
        Tuple[float, AttentionState, AttentionTransitionReason]
    ] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        reason_value = self.transition_reason.value if self.transition_reason else ""
        reason_label = self.transition_reason.label() if self.transition_reason else ""
        return {
            "state": self.state.value,
            "state_label": self.state.label(),
            "attention_mode": self.state.value,
            "previous_state": self.previous_state.value,
            "previous_state_label": self.previous_state.label(),
            "previous": self.previous_state.label(),
            "reason": reason_value,
            "reason_label": reason_label,
            "openness": round(self.openness, 3),
            "peek_desire": round(self.peek_desire, 3),
            "visibility_threshold": round(self.state.visibility_threshold(), 3),
            "process_ratio": round(self.state.message_process_ratio(), 3),
            "silence_tolerance": round(self.silence_tolerance, 3),
            "interrupt_tolerance": round(self.interrupt_tolerance, 3),
            "curiosity_level": round(self.curiosity_level, 3),
            "social_hunger": round(self.social_hunger, 3),
            "withdrawal_depth": round(self.withdrawal_depth, 3),
            "look_budget_state": round(self.look_budget_state, 3),
            "process_budget_state": round(self.process_budget_state, 3),
            "since_last_look_sec": round(self.time_since_last_look, 1),
            "since_last_act_sec": round(self.time_since_last_act, 1),
            "consecutive_empty_peeks": self.consecutive_peeks_without_action,
            "empty_peeks": self.consecutive_peeks_without_action,
        }


@dataclass
class MessageFilterResult:
    """消息过滤结果——注意力流对每条消息的准入判断"""

    message_id: str = ""
    user_id: str = ""
    passed: bool = False
    attention_level: float = 0.0
    filter_reason: str = ""
    perceived_as: str = ""
    will_remember: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.message_id[:12],
            "passed": self.passed,
            "level": round(self.attention_level, 3),
            "reason": self.filter_reason[:40],
            "perceived": self.perceived_as[:20],
        }


# 状态转换规则表：(当前状态, 条件) → (新状态, 原因)
_TRANSITION_RULES: List[
    Tuple[AttentionState, str, AttentionState, AttentionTransitionReason]
] = [
    (
        AttentionState.DEEP_WITHDRAWAL,
        "boredom_high",
        AttentionState.PEEKING,
        AttentionTransitionReason.BOREDOM_SPIKE,
    ),
    (
        AttentionState.DEEP_WITHDRAWAL,
        "user_return",
        AttentionState.PEEKING,
        AttentionTransitionReason.USER_RETURN,
    ),
    (
        AttentionState.WITHDRAWN,
        "boredom_medium",
        AttentionState.PEEKING,
        AttentionTransitionReason.SELF_DECIDED_PEEK,
    ),
    (
        AttentionState.WITHDRAWN,
        "curiosity",
        AttentionState.CASUAL_GLANCE,
        AttentionTransitionReason.CURIOSITY_TRIGGER,
    ),
    (
        AttentionState.WITHDRAWN,
        "social_high",
        AttentionState.SELECTIVE_ATTENTION,
        AttentionTransitionReason.SOCIAL_DESIRE_RISE,
    ),
    (
        AttentionState.PEEKING,
        "found_interesting",
        AttentionState.CASUAL_GLANCE,
        AttentionTransitionReason.CURIOSITY_TRIGGER,
    ),
    (
        AttentionState.PEEKING,
        "nothing_there",
        AttentionState.WITHDRAWN,
        AttentionTransitionReason.NOTHING_INTERESTING,
    ),
    (
        AttentionState.PEEKING,
        "peeked_too_much",
        AttentionState.WITHDRAWN,
        AttentionTransitionReason.FATIGUE_SETTLING,
    ),
    (
        AttentionState.CASUAL_GLANCE,
        "engaging_content",
        AttentionState.SELECTIVE_ATTENTION,
        AttentionTransitionReason.CURIOSITY_TRIGGER,
    ),
    (
        AttentionState.CASUAL_GLANCE,
        "nothing_special",
        AttentionState.PEEKING,
        AttentionTransitionReason.NOTHING_INTERESTING,
    ),
    (
        AttentionState.CASUAL_GLANCE,
        "too_much_noise",
        AttentionState.WITHDRAWN,
        AttentionTransitionReason.OVERSTIMULATION,
    ),
    (
        AttentionState.SELECTIVE_ATTENTION,
        "high_engagement",
        AttentionState.FULLY_ENGAGED,
        AttentionTransitionReason.SOCIAL_DESIRE_RISE,
    ),
    (
        AttentionState.SELECTIVE_ATTENTION,
        "tired_bored",
        AttentionState.CASUAL_GLANCE,
        AttentionTransitionReason.FATIGUE_SETTLING,
    ),
    (
        AttentionState.SELECTIVE_ATTENTION,
        "emotional_heavy",
        AttentionState.WITHDRAWN,
        AttentionTransitionReason.EMOTIONAL_OVERLOAD,
    ),
    (
        AttentionState.FULLY_ENGAGED,
        "overwhelmed",
        AttentionState.SELECTIVE_ATTENTION,
        AttentionTransitionReason.OVERSTIMULATION,
    ),
    (
        AttentionState.FULLY_ENGAGED,
        "safety_alert",
        AttentionState.WITHDRAWN,
        AttentionTransitionReason.SAFETY_RETREAT,
    ),
    (
        AttentionState.FULLY_ENGAGED,
        "post_active_tired",
        AttentionState.SELECTIVE_ATTENTION,
        AttentionTransitionReason.PROACTIVE_AFTERMATH,
    ),
    (
        AttentionState.FULLY_ENGAGED,
        "night_coming",
        AttentionState.CASUAL_GLANCE,
        AttentionTransitionReason.NIGHT_FALL,
    ),
    (
        AttentionState.SELECTIVE_ATTENTION,
        "night_coming",
        AttentionState.PEEKING,
        AttentionTransitionReason.NIGHT_FALL,
    ),
    (
        AttentionState.CASUAL_GLANCE,
        "night_deep",
        AttentionState.WITHDRAWN,
        AttentionTransitionReason.NIGHT_FALL,
    ),
]


class SubjectiveAttentionFlowController:
    """主观注意力流控制器（文档第1930行核心主轴）

    实现文档描述的"围绕注意力流而非消息触发流"的架构：

    核心哲学：
    - 机器人不是在"等待消息"，而是在"过自己的状态流"
    - 外界消息只是会打断/吸引/触发这种内部流
    - 消息不是天然重要，而是需要经过"注意力准入"

    六种注意力状态：
    1. 全神贯注(FULLY_ENGAGED)：深度参与，处理所有消息
    2. 选择性关注(SELECTIVE_ATTENTION)：只看重要消息(65%)
    3. 随意扫视(CASUAL_GLANCE)：快速扫一眼(30%)
    4. 偶尔窥屏(PEEKING)：偷偷看一眼(12%)
    5. 暂时退出(WITHDRAWN)：不想看群(3%)
    6. 深度退出(DEEP_WITHDRAWAL)：完全关闭(0%)

    21条状态转换规则驱动状态流转。
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._snapshot = AttentionFlowSnapshot(channel_id=channel_id)
        self._last_tick_ts = 0.0
        self._tick_interval_sec = 15.0
        self._peek_cooldown_sec = 60.0
        self._last_peek_at = 0.0
        self._max_history = 50
        self._look_budget_accumulator = 0.0
        self._process_budget_accumulator = 0.0

    # ═════════════════ 主入口：tick + 过滤 ═════════════════

    def tick(
        self,
        now: float,
        *,
        boredom: float = 0.0,
        social_desire: float = 0.5,
        energy: float = 1.0,
        mood: float = 0.5,
        loneliness: float = 0.0,
        curiosity: float = 0.3,
        night_phase: str = "",
        safety_level: str = "",
        has_pending_user: bool = False,
        group_activity_level: float = 0.0,
        recent_proactive_count: int = 0,
        emotion_overload_flag: bool = False,
    ) -> AttentionFlowSnapshot:
        if now - self._last_tick_ts < self._tick_interval_sec:
            return self._snapshot
        self._last_tick_ts = now
        old_state = self._snapshot.state
        self._update_internal_metrics(
            now, boredom, social_desire, energy, mood, loneliness, curiosity
        )
        new_state = self._evaluate_transitions(
            now,
            boredom,
            social_desire,
            energy,
            mood,
            loneliness,
            curiosity,
            night_phase,
            safety_level,
            has_pending_user,
            group_activity_level,
            recent_proactive_count,
            emotion_overload_flag,
        )
        if new_state != old_state:
            reason = self._detect_transition_reason(
                old_state,
                new_state,
                boredom,
                social_desire,
                energy,
                mood,
                night_phase,
                safety_level,
                has_pending_user,
                emotion_overload_flag,
            )
            self._apply_transition(new_state, reason, now)
        self._snapshot.time_since_last_look = now - self._last_peek_at
        self._snapshot.time_since_last_act = (
            now - self._snapshot.last_transition_at
        )
        return self._snapshot

    def filter_messages(
        self,
        messages: List[Any],
        *,
        force_state: Optional[AttentionState] = None,
    ) -> Tuple[List[MessageFilterResult], List[Any]]:
        state = force_state or self._snapshot.state
        threshold = state.visibility_threshold()
        process_ratio = state.message_process_ratio()
        results: List[MessageFilterResult] = []
        passed: List[Any] = []
        for msg in messages:
            uid = str(getattr(msg, "user_id", "") or "")
            mid = str(
                getattr(msg, "message_id", "")
                or getattr(msg, "msg_id", "")
                or id(msg)
            )
            is_at = getattr(msg, "is_at", False)
            is_quote = bool(getattr(msg, "reply_to_message_id", None))
            level = self._compute_message_attention(msg, state)
            passed_filter = level >= threshold
            if not passed_filter and (is_at or is_quote):
                level += 0.25
                passed_filter = level >= threshold
            if passed_filter:
                self._process_budget_accumulator = min(
                    2.0, self._process_budget_accumulator + process_ratio
                )
                self._snapshot.process_budget_state = self._process_budget_accumulator
                if self._process_budget_accumulator >= 1.0:
                    self._process_budget_accumulator -= 1.0
                    self._snapshot.process_budget_state = self._process_budget_accumulator
                else:
                    passed_filter = False
            perceived = self._classify_perception(level, passed_filter, msg)
            result = MessageFilterResult(
                message_id=mid,
                user_id=uid,
                passed=passed_filter,
                attention_level=level,
                filter_reason=self._filter_reason_text(
                    level, threshold, passed_filter, is_at, is_quote
                ),
                perceived_as=perceived,
                will_remember=passed_filter and level > 0.6,
            )
            results.append(result)
            if passed_filter:
                passed.append(msg)
        if passed:
            self._last_peek_at = time.time()
            self._snapshot.consecutive_peeks_without_action = 0
        else:
            self._snapshot.consecutive_peeks_without_action += 1
        logger.debug(
            f"[注意力流] {self._channel_id[:12]} {state.label()}: "
            f"{len(messages)}条→{len(passed)}通过 "
            f"(阈值={threshold:.2f}, 比例={process_ratio:.0%})"
        )
        return results, passed

    def should_even_look(self, now: float) -> bool:
        state = self._snapshot.state
        if state == AttentionState.FULLY_ENGAGED:
            self._snapshot.look_budget_state = 1.0
            return True
        if state == AttentionState.DEEP_WITHDRAWAL:
            if now - self._last_peek_at < self._peek_cooldown_sec * 3:
                return False
            self._look_budget_accumulator = min(1.2, self._look_budget_accumulator + 0.08)
            self._snapshot.look_budget_state = self._look_budget_accumulator
            if self._look_budget_accumulator < 1.0:
                return False
            self._look_budget_accumulator -= 1.0
            self._snapshot.look_budget_state = self._look_budget_accumulator
            return True
        if state == AttentionState.WITHDRAWN:
            if now - self._last_peek_at < self._peek_cooldown_sec:
                return False
            peek_prob = 0.15 + self._snapshot.peek_desire * 0.20
            self._look_budget_accumulator = min(1.5, self._look_budget_accumulator + peek_prob)
            self._snapshot.look_budget_state = self._look_budget_accumulator
            if self._look_budget_accumulator < 1.0:
                return False
            self._look_budget_accumulator -= 1.0
            self._snapshot.look_budget_state = self._look_budget_accumulator
            return True
        if state == AttentionState.PEEKING:
            peek_prob = 0.4 + self._snapshot.peek_desire * 0.3
            self._look_budget_accumulator = min(1.5, self._look_budget_accumulator + peek_prob)
            self._snapshot.look_budget_state = self._look_budget_accumulator
            if self._look_budget_accumulator < 1.0:
                return False
            self._look_budget_accumulator -= 1.0
            self._snapshot.look_budget_state = self._look_budget_accumulator
            return True
        self._snapshot.look_budget_state = 1.0
        return True

    def get_snapshot(self) -> AttentionFlowSnapshot:
        return self._snapshot

    # ────────────────── 内部方法 ──────────────────

    def _update_internal_metrics(
        self,
        now: float,
        boredom: float,
        social_desire: float,
        energy: float,
        mood: float,
        loneliness: float,
        curiosity: float,
    ) -> None:
        snap = self._snapshot
        _openness_raw = (
            0.5
            + (social_desire - 0.5) * 0.25
            + (curiosity - 0.3) * 0.20
            + (mood - 0.5) * 0.10
            + (energy - 0.5) * 0.15
            - boredom * 0.20
            - loneliness * 0.10
            - snap.withdrawal_depth * 0.15
        )
        snap.openness = max(0.0, min(1.0, _openness_raw))
        _peek_raw = (
            boredom * 0.45
            + loneliness * 0.25
            + curiosity * 0.20
            + (1.0 - snap.withdrawal_depth) * 0.10
        )
        snap.peek_desire = max(0.0, min(1.0, _peek_raw))
        snap.silence_tolerance = max(0.0, min(1.0, 1.0 - boredom * 0.7))
        _interrupt_raw = (
            snap.openness * 0.6 + (1.0 - loneliness) * 0.25 + energy * 0.15
        )
        snap.interrupt_tolerance = max(0.0, min(1.0, _interrupt_raw))
        snap.curiosity_level = curiosity
        snap.social_hunger = social_desire

    def _evaluate_transitions(
        self,
        now: float,
        boredom: float,
        social_desire: float,
        energy: float,
        mood: float,
        loneliness: float,
        curiosity: float,
        night_phase: str,
        safety_level: str,
        has_pending_user: bool,
        group_activity: float,
        proactive_count: int,
        emotion_overload: bool,
    ) -> AttentionState:
        current = self._snapshot.state
        conditions: Dict[str, bool] = {
            "boredom_high": boredom > 0.65,
            "boredom_medium": boredom > 0.40,
            "curiosity": curiosity > 0.55,
            "social_high": social_desire > 0.70,
            "found_interesting": group_activity > 0.6 and curiosity > 0.4,
            "nothing_there": (
                self._snapshot.consecutive_peeks_without_action >= 3
                or (group_activity < 0.1 and boredom > 0.3)
            ),
            "peeked_too_much": self._snapshot.consecutive_peeks_without_action
            >= 5,
            "engaging_content": group_activity > 0.7 and social_desire > 0.5,
            "nothing_special": group_activity < 0.25,
            "too_much_noise": group_activity > 0.9 and energy < 0.3,
            "high_engagement": social_desire > 0.75
            and mood > 0.6
            and energy > 0.5,
            "tired_bored": energy < 0.3 or boredom > 0.55,
            "emotional_heavy": emotion_overload
            or (abs(mood - 0.5) > 0.35 and loneliness > 0.4),
            "overwhelmed": group_activity > 0.85
            and self._snapshot.openness < 0.3,
            "safety_alert": safety_level in ("high", "critical"),
            "post_active_tired": proactive_count > 0 and energy < 0.4,
            "night_coming": night_phase in ("drowsy", "light_sleep"),
            "night_deep": night_phase in ("deep_sleep",),
            "user_return": has_pending_user,
        }
        for rule_from, cond_key, rule_to, _rule_reason in _TRANSITION_RULES:
            if current == rule_from and conditions.get(cond_key, False):
                return rule_to
        return current

    @staticmethod
    def _detect_transition_reason(
        from_state: AttentionState,
        to_state: AttentionState,
        boredom: float,
        social_desire: float,
        energy: float,
        mood: float,
        night_phase: str,
        safety_level: str,
        has_pending_user: bool,
        emotion_overload: bool,
    ) -> AttentionTransitionReason:
        if to_state == AttentionState.PEEKING:
            if has_pending_user:
                return AttentionTransitionReason.USER_RETURN
            if boredom > 0.5:
                return AttentionTransitionReason.BOREDOM_SPIKE
            return AttentionTransitionReason.SELF_DECIDED_PEEK
        if to_state == AttentionState.WITHDRAWN:
            if safety_level in ("high", "critical"):
                return AttentionTransitionReason.SAFETY_RETREAT
            if emotion_overload:
                return AttentionTransitionReason.EMOTIONAL_OVERLOAD
            return AttentionTransitionReason.FATIGUE_SETTLING
        if to_state == AttentionState.CASUAL_GLANCE:
            if night_phase in ("drowsy", "light_sleep"):
                return AttentionTransitionReason.NIGHT_FALL
            return AttentionTransitionReason.CURIOSITY_TRIGGER
        if to_state == AttentionState.SELECTIVE_ATTENTION:
            if social_desire > 0.65:
                return AttentionTransitionReason.SOCIAL_DESIRE_RISE
            return AttentionTransitionReason.CURIOSITY_TRIGGER
        if to_state == AttentionState.FULLY_ENGAGED:
            return AttentionTransitionReason.SOCIAL_DESIRE_RISE
        if to_state == AttentionState.DEEP_WITHDRAWAL:
            return AttentionTransitionReason.FATIGUE_SETTLING
        return AttentionTransitionReason.SELF_DECIDED_PEEK

    def _apply_transition(
        self,
        new_state: AttentionState,
        reason: AttentionTransitionReason,
        now: float,
    ) -> None:
        snap = self._snapshot
        snap.previous_state = snap.state
        snap.state = new_state
        snap.transition_reason = reason
        snap.last_transition_at = now
        snap.state_entered_at = now
        if new_state in (
            AttentionState.WITHDRAWN,
            AttentionState.DEEP_WITHDRAWAL,
        ):
            snap.withdrawal_depth = min(1.0, snap.withdrawal_depth + 0.15)
        else:
            snap.withdrawal_depth = max(0.0, snap.withdrawal_depth - 0.08)
        snap.transition_history.append((now, new_state, reason))
        if len(snap.transition_history) > self._max_history:
            snap.transition_history = snap.transition_history[
                -self._max_history:
            ]
        logger.info(
            f"[注意力流] {self._channel_id[:12]} 状态转换: "
            f"{snap.previous_state.label()} → {new_state.label()} ({reason.label()})"
        )

    @staticmethod
    def _compute_message_attention(msg: Any, state: AttentionState) -> float:
        base = 0.25
        is_at = getattr(msg, "is_at", False)
        is_quote = bool(getattr(msg, "reply_to_message_id", None))
        plain_len = len(
            str(
                getattr(msg, "processed_plain_text", "")
                or getattr(msg, "plain_text", "")
                or ""
            )
        )
        if is_at:
            base += 0.45
        if is_quote:
            base += 0.28
        if plain_len > 100:
            base += 0.08
        elif plain_len < 8:
            base -= 0.06
        if state == AttentionState.FULLY_ENGAGED:
            base *= 1.3
        elif state == AttentionState.SELECTIVE_ATTENTION:
            base *= 1.0
        elif state == AttentionState.CASUAL_GLANCE:
            base *= 0.65
        elif state == AttentionState.PEEKING:
            base *= 0.35
        else:
            base *= 0.15
        return max(0.0, min(1.0, base))

    @staticmethod
    def _classify_perception(level: float, passed: bool, msg: Any) -> str:
        if not passed:
            if level < 0.10:
                return "完全没注意"
            return "扫到但没在意"
        if level > 0.80:
            return "深度关注"
        if level > 0.55:
            return "认真看了"
        return "纳入注意"

    @staticmethod
    def _filter_reason_text(
        level: float,
        threshold: float,
        passed: bool,
        is_at: bool,
        is_quote: bool,
    ) -> str:
        if passed:
            if is_at:
                return "@提及强制通过"
            if is_quote:
                return "引用回复通过"
            if level > 0.7:
                return "高关注度自然通过"
            return f"超过阈值({threshold:.2f})"
        if is_at and not passed:
            return "@提及但注意力太低"
        return f"低于可见性阈值({level:.2f}<{threshold:.2f})"


_attention_flow_manager = MultiInstanceManager()


def get_attention_flow_controller(
    channel_id: str,
) -> SubjectiveAttentionFlowController:
    return _attention_flow_manager.get_or_create(
        channel_id,
        lambda: SubjectiveAttentionFlowController(channel_id)
    )

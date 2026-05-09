import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from src.common.data_models.base_data_model import BaseDataModel


class FlowPhase(Enum):
    """心流运行阶段，五个互斥状态构成有限状态机"""

    STANDBY = "standby"
    PENDING = "pending"
    DORMANT = "dormant"
    LIGHT_REST = "light_rest"
    ENGAGED = "engaged"

    def label(self) -> str:
        labels = {
            "standby": "待命",
            "pending": "等待",
            "dormant": "休息",
            "light_rest": "轻度休息",
            "engaged": "活跃",
        }
        return labels.get(self.value, self.value)


class IdleDecision(Enum):
    """等待阶段的决策动作"""

    HOLD = "hold"
    CONTEMPLATE = "contemplate"
    CHASE = "chase"
    EXPIRED = "expired"
    WITHDRAW = "withdraw"


class DormantReaction(Enum):
    """休息期间收到消息时的反应"""

    SKIP = "skip"
    GLANCE = "glance"
    AWAKEN = "awaken"


LEGAL_TRANSITIONS: Dict[FlowPhase, set] = {
    FlowPhase.STANDBY: {
        FlowPhase.PENDING,
        FlowPhase.DORMANT,
        FlowPhase.LIGHT_REST,
        FlowPhase.ENGAGED,
    },
    FlowPhase.PENDING: {FlowPhase.STANDBY, FlowPhase.DORMANT, FlowPhase.LIGHT_REST, FlowPhase.ENGAGED},
    FlowPhase.LIGHT_REST: {
        FlowPhase.STANDBY,
        FlowPhase.PENDING,
        FlowPhase.DORMANT,
        FlowPhase.ENGAGED,
    },
    FlowPhase.DORMANT: {
        FlowPhase.STANDBY,
        FlowPhase.PENDING,
        FlowPhase.LIGHT_REST,
        FlowPhase.ENGAGED,
    },
    FlowPhase.ENGAGED: {
        FlowPhase.STANDBY,
        FlowPhase.PENDING,
        FlowPhase.DORMANT,
        FlowPhase.LIGHT_REST,
    },
}


def is_transition_legal(current: FlowPhase, target: FlowPhase) -> bool:
    """检查状态迁移是否合法"""
    allowed = LEGAL_TRANSITIONS.get(current)
    if allowed is None:
        return False
    return target in allowed


@dataclass
class IdleSetup(BaseDataModel):
    """等待阶段运行时参数，字段默认值为零值，实际值从配置引擎注入"""

    anticipated_response: str = ""
    max_wait_sec: int = 0
    entered_at: float = 0.0
    last_contemplate_at: float = 0.0
    contemplate_rounds: int = 0
    chase_rounds: int = 0
    max_chase_allowed: int = 0

    def elapsed_sec(self) -> float:
        if self.entered_at <= 0:
            return 0.0
        return time.time() - self.entered_at

    def elapsed_min(self) -> float:
        return self.elapsed_sec() / 60.0

    def is_timed_out(self) -> bool:
        if self.max_wait_sec <= 0 or self.entered_at <= 0:
            return False
        return self.elapsed_sec() >= self.max_wait_sec

    def wait_progress(self) -> float:
        if self.max_wait_sec <= 0 or self.entered_at <= 0:
            return 0.0
        return min(self.elapsed_sec() / self.max_wait_sec, 1.0)

    def can_contemplate(self, interval_sec: int = 0) -> bool:
        if interval_sec <= 0:
            return False
        if self.last_contemplate_at <= 0:
            return self.elapsed_sec() >= interval_sec
        return (time.time() - self.last_contemplate_at) >= interval_sec

    def can_chase(self) -> bool:
        return self.chase_rounds < self.max_chase_allowed

    def record_contemplate(self) -> None:
        self.last_contemplate_at = time.time()
        self.contemplate_rounds += 1

    def record_chase(self) -> None:
        self.chase_rounds += 1

    def clear(self) -> None:
        self.anticipated_response = ""
        self.max_wait_sec = 0
        self.entered_at = 0.0
        self.last_contemplate_at = 0.0
        self.contemplate_rounds = 0
        self.chase_rounds = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anticipated_response": self.anticipated_response,
            "max_wait_sec": self.max_wait_sec,
            "entered_at": self.entered_at,
            "last_contemplate_at": self.last_contemplate_at,
            "contemplate_rounds": self.contemplate_rounds,
            "chase_rounds": self.chase_rounds,
            "max_chase_allowed": self.max_chase_allowed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdleSetup":
        return cls(
            anticipated_response=data.get("anticipated_response", ""),
            max_wait_sec=data.get("max_wait_sec", 0),
            entered_at=data.get("entered_at", 0.0),
            last_contemplate_at=data.get("last_contemplate_at", 0.0),
            contemplate_rounds=data.get("contemplate_rounds", 0),
            chase_rounds=data.get("chase_rounds", 0),
            max_chase_allowed=data.get("max_chase_allowed", 0),
        )


@dataclass
class DormantSetup(BaseDataModel):
    """休息阶段运行时参数"""

    cause: str = ""
    duration_sec: int = 0
    entered_at: float = 0.0
    skip_streak: int = 0
    glance_probability: float = 0.0
    allow_awakening: bool = False
    rest_intensity: float = 0.5
    wake_drive: float = 0.3
    regret_chance: float = 0.0
    min_rest_sec: int = 0
    last_reconsider_at: float = 0.0

    def remaining_sec(self) -> float:
        if self.entered_at <= 0:
            return float(self.duration_sec)
        passed = time.time() - self.entered_at
        return max(0.0, self.duration_sec - passed)

    def can_reconsider(self, cooldown_sec: float = 20.0) -> bool:
        if cooldown_sec <= 0:
            return True
        if self.last_reconsider_at <= 0:
            return True
        return (time.time() - self.last_reconsider_at) >= cooldown_sec

    def mark_reconsidered(self) -> None:
        self.last_reconsider_at = time.time()

    def is_finished(self) -> bool:
        return self.remaining_sec() <= 0

    def progress_ratio(self) -> float:
        if self.duration_sec <= 0:
            return 1.0
        if self.entered_at <= 0:
            return 0.0
        return min((time.time() - self.entered_at) / self.duration_sec, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause": self.cause,
            "duration_sec": self.duration_sec,
            "entered_at": self.entered_at,
            "skip_streak": self.skip_streak,
            "glance_probability": self.glance_probability,
            "allow_awakening": self.allow_awakening,
            "rest_intensity": self.rest_intensity,
            "wake_drive": self.wake_drive,
            "regret_chance": self.regret_chance,
            "min_rest_sec": self.min_rest_sec,
            "last_reconsider_at": self.last_reconsider_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DormantSetup":
        return cls(
            cause=data.get("cause", ""),
            duration_sec=int(data.get("duration_sec", 0)),
            entered_at=float(data.get("entered_at", 0.0)),
            skip_streak=int(data.get("skip_streak", 0)),
            glance_probability=float(data.get("glance_probability", 0.0)),
            allow_awakening=bool(data.get("allow_awakening", False)),
            rest_intensity=float(data.get("rest_intensity", 0.5)),
            wake_drive=float(data.get("wake_drive", 0.3)),
            regret_chance=float(data.get("regret_chance", 0.0)),
            min_rest_sec=int(data.get("min_rest_sec", 0)),
            last_reconsider_at=float(data.get("last_reconsider_at", 0.0)),
        )


@dataclass
class BehaviorIntent(BaseDataModel):
    """内心独白产出的单个行为指令"""

    intent_type: str = ""
    target_id: str = ""
    payload: str = ""
    defer_sec: float = 0.0
    priority: int = 0

    def is_reply(self) -> bool:
        return self.intent_type in ("reply", "miao_reply")

    def is_passive(self) -> bool:
        return self.intent_type in ("observe", "idle", "ignore", "lurk", "rest", "disengage", "")

    def describe(self) -> str:
        if self.intent_type in ("reply", "miao_reply"):
            preview = self.payload[:40] + "..." if len(self.payload) > 40 else self.payload
            return f'回复: "{preview}"'
        elif self.intent_type == "nudge":
            return "戳一下"
        elif self.intent_type == "sticker":
            return f"发送表情: {self.payload}"
        elif self.intent_type == "wait":
            return f"等待 {self.defer_sec:.0f}秒"
        elif self.intent_type in ("observe", "idle"):
            return "保持沉默"
        elif self.intent_type == "followup":
            return "追问"
        elif self.intent_type == "lurk":
            return "潜水观望"
        elif self.intent_type == "rest":
            return "休息一会"
        elif self.intent_type == "disengage":
            return "放下手机"
        return f"动作: {self.intent_type}"

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"type": self.intent_type}
        if self.target_id:
            result["target"] = self.target_id
        if self.payload:
            result["content"] = self.payload
        if self.defer_sec > 0:
            result["delay"] = self.defer_sec
        if self.priority > 0:
            result["priority"] = self.priority
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorIntent":
        return cls(
            intent_type=data.get("type", ""),
            target_id=data.get("target", ""),
            payload=data.get("content", ""),
            defer_sec=float(data.get("delay", 0.0)),
            priority=int(data.get("priority", 0)),
        )


@dataclass
class VoiceVerdict(BaseDataModel):
    """内心独白的完整输出"""

    thinking: str = ""
    reply_desire_level: int = 0
    should_reply: bool = False
    next_action: str = ""
    intents: List[BehaviorIntent] = field(default_factory=list)
    expected_reaction: str = ""
    max_wait_after: int = 0
    current_mood: str = ""
    is_valid: bool = False
    error_detail: Optional[str] = None
    dominant_unfinished_intent: Optional[Dict[str, Any]] = None
    comprehension_confidence: float = 0.5
    needs_upgrade: bool = False
    thinking_source: str = ""

    def has_active_intent(self) -> bool:
        return any(not i.is_passive() for i in self.intents)

    def primary_intent(self) -> Optional[BehaviorIntent]:
        if not self.intents:
            return None
        return sorted(self.intents, key=lambda x: x.priority, reverse=True)[0]

    def reply_intents(self) -> List[BehaviorIntent]:
        return [i for i in self.intents if i.is_reply()]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thinking,
            "actions": [i.to_dict() for i in self.intents],
            "expected_reaction": self.expected_reaction,
            "max_wait_seconds": self.max_wait_after,
            "mood": self.current_mood,
            "reply_desire": self.reply_desire_level,
            "should_reply": self.should_reply,
            "next_action": self.next_action,
            "dominant_intent": self.dominant_unfinished_intent,
            "comprehension_confidence": self.comprehension_confidence,
            "needs_upgrade": self.needs_upgrade,
            "thinking_source": self.thinking_source,
            "is_valid": self.is_valid,
            "error_detail": self.error_detail,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceVerdict":
        actions_raw = data.get("actions", [])
        intents = [BehaviorIntent.from_dict(a) for a in actions_raw] if actions_raw else []
        return cls(
            thinking=data.get("thought", ""),
            reply_desire_level=int(data.get("reply_desire", 0)),
            should_reply=bool(data.get("should_reply", False)),
            next_action=str(data.get("next_action", "")),
            intents=intents,
            expected_reaction=data.get("expected_reaction", ""),
            max_wait_after=int(data.get("max_wait_seconds", 0)),
            current_mood=data.get("mood", ""),
            is_valid=bool(data.get("is_valid", True)),
            error_detail=data.get("error_detail"),
            dominant_unfinished_intent=data.get("dominant_intent"),
            comprehension_confidence=float(data.get("comprehension_confidence", 0.5)),
            needs_upgrade=bool(data.get("needs_upgrade", False)),
            thinking_source=str(data.get("thinking_source", "") or ""),
        )

    @classmethod
    def make_error(cls, msg: str) -> "VoiceVerdict":
        return cls(is_valid=False, error_detail=msg)


@dataclass
class EnergySnapshot(BaseDataModel):
    """能量系统即时快照 - 重构版

    核心指标：
    - chat_pool: 聊天值（群聊共享）
    - activity_level: 活跃度（群聊共享）
    - social_value: 社交值（群聊共享）
    - favorability: 好感度（用户独立）

    - thinking_value: 思考值
    """

    chat_pool: float = 0.0
    chat_ceiling: float = 100.0
    activity_level: float = 50.0
    activity_ceiling: float = 100.0
    annoyance_level: float = 0.0
    captured_at: float = 0.0
    social_value: float = 0.0
    positive_dim: float = 0.0
    negative_dim: float = 0.0
    trust_value: float = 0.0
    annoyance_value: float = 0.0
    favorability: float = 0.0
    interaction_count: int = 0
    relationship_level: int = 2
    custom_label: str = ""
    trend_direction: str = "稳定"
    user_id: str = ""
    mode: str = "shared"
    user_tags: List[str] = field(default_factory=list)
    thinking_value: float = 0.0
    thinking_ceiling: float = 100.0

    def chat_ratio(self) -> float:
        if self.chat_ceiling <= 0:
            return 0.0
        return max(0.0, min(1.0, self.chat_pool / self.chat_ceiling))

    def activity_ratio(self) -> float:
        if self.activity_ceiling <= 0:
            return 0.5
        return max(0.0, min(1.0, self.activity_level / self.activity_ceiling))

    def thinking_ratio(self) -> float:
        if self.thinking_ceiling <= 0:
            return 0.5
        return max(0.0, min(1.0, self.thinking_value / self.thinking_ceiling))

    def is_exhausted(self) -> bool:
        return self.chat_ratio() < 0.1

    def is_overflowing(self, bar: float = 0.0) -> bool:
        if bar <= 0:
            return False
        return self.chat_ratio() > bar

    def social_ratio(self) -> float:
        return (self.social_value + 100.0) / 200.0

    def trust_ratio(self) -> float:
        return (self.trust_value + 100.0) / 200.0

    def favorability_ratio(self) -> float:
        return (self.favorability + 100.0) / 200.0

    def combined_consumption_factor(self) -> float:
        """综合消耗因子 = 聊天值比例 × 活跃度比例"""
        return self.chat_ratio() * self.activity_ratio()

    def summary_text(self) -> str:
        return (
            f"聊天值={self.chat_pool:.1f}/{self.chat_ceiling:.0f}"
            f" 活跃度={self.activity_level:.1f}"
            f" 社交值={self.social_value:.1f}"
            f" 好感度={self.favorability:.1f}"
            f" 标签={self.user_tags}"
        )


@dataclass
class DecisionContextPacket(BaseDataModel):
    """决策前的轻量上下文包，统一承载续接与重复信号。"""

    target_user_id: str = ""
    target_name: str = ""
    latest_user_text: str = ""
    latest_bot_text: str = ""
    continuation_hint: str = ""
    repeated_short_input: bool = False
    repeat_reason: str = ""

    def compact_context(self) -> str:
        lines: List[str] = ["[决策上下文包]"]
        if self.target_name or self.target_user_id:
            lines.append(f"- 当前对象: {self.target_name or self.target_user_id}")
        if self.latest_user_text:
            lines.append(f"- 最新用户话头: {self.latest_user_text[:120]}")
        if self.latest_bot_text:
            lines.append(f"- 你上一句: {self.latest_bot_text[:120]}")
        if self.continuation_hint:
            lines.append(f"- 续接提示: {self.continuation_hint[:160]}")
        if self.repeated_short_input:
            reason = self.repeat_reason or "这是重复短句/前情续接，不要硬切成新话题"
            lines.append(f"- 重复短句判定: {reason[:160]}")
        return "\n".join(lines)


@dataclass
class PhaseTransitionRecord(BaseDataModel):
    """状态迁移日志条目"""

    from_phase: str = ""
    to_phase: str = ""
    trigger_reason: str = ""
    occurred_at: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def age_sec(self) -> float:
        if self.occurred_at <= 0:
            return 0.0
        return time.time() - self.occurred_at


@dataclass
class ChannelPulse(BaseDataModel):
    """频道实时态势"""

    vitality_tag: str = ""
    quiet_duration_sec: float = 0.0
    live_speaker_count: int = 0
    recent_msg_count_5min: int = 0
    dispute_detected: bool = False
    seeking_topic: bool = False
    narrative: str = ""

    @staticmethod
    def tag_from_count(count_5min: int) -> str:
        if count_5min >= 30:
            return "爆发"
        elif count_5min >= 15:
            return "活跃"
        elif count_5min >= 5:
            return "平稳"
        elif count_5min >= 1:
            return "冷清"
        return "死寂"

    def is_quiet(self, bar_sec: float = 0.0) -> bool:
        if bar_sec <= 0:
            return False
        return self.quiet_duration_sec >= bar_sec

    def should_start_topic(self) -> bool:
        return self.seeking_topic or self.vitality_tag in ("冷清", "死寂")


@dataclass
class SelfSnapshot(BaseDataModel):
    """机器人自身状态"""

    since_last_reply: str = ""
    time_period: str = ""
    clock_display: str = ""
    online_duration_min: float = 0.0
    recent_decisions: str = ""


@dataclass
class MultisensoryData(BaseDataModel):
    """多维感知汇总"""

    channel_pulse: ChannelPulse = field(default_factory=ChannelPulse)
    user_relation_tags: Dict[str, str] = field(default_factory=dict)
    self_snapshot: SelfSnapshot = field(default_factory=SelfSnapshot)

    def relation_label(self, uid: str) -> str:
        return self.user_relation_tags.get(uid, "陌生人")


@dataclass
class ContextDemand(BaseDataModel):
    """感知引擎的上下文申请单"""

    require_memory: bool = False
    require_full_context: bool = False
    history_depth: int = 0
    memory_keywords: str = ""
    focus_user_ids: List[str] = field(default_factory=list)

    def has_memory_need(self) -> bool:
        return self.require_memory and bool(self.memory_keywords)


@dataclass
class PerceptionConclusion(BaseDataModel):
    """感知引擎的决策输出"""

    action_tag: str = ""
    hold_duration_sec: int = 0
    inner_thought: str = ""
    reasoning: str = ""
    context_demand: ContextDemand = field(default_factory=ContextDemand)
    parse_ok: bool = False

    def wants_reply(self) -> bool:
        return self.action_tag in ("reply", "回复")

    def wants_topic(self) -> bool:
        return self.action_tag in ("start_topic", "发起话题")

    def is_passive(self) -> bool:
        return self.action_tag in ("observe", "idle", "观望", "空闲", "")


@dataclass
class BotProfile(BaseDataModel):
    """机器人身份档案"""

    display_name: str = ""
    persona_desc: str = ""
    character_age: int = 0
    age_bracket: str = ""
    hobbies_desc: str = ""
    reply_style_hint: str = ""


@dataclass
class ReplyContextBundle(BaseDataModel):
    """传递给大模型的回复上下文"""

    message_history: List[Dict[str, str]] = field(default_factory=list)
    retrieved_memories: List[str] = field(default_factory=list)
    persona_prompt: str = ""
    user_relations: Dict[str, str] = field(default_factory=dict)
    perception_brief: str = ""
    scene_label: str = ""
    thought_guide: str = ""
    available_styles: List[str] = field(default_factory=list)
    slang_mapping: Dict[str, str] = field(default_factory=dict)
    tool_outputs: List[Dict] = field(default_factory=list)
    image_analyses: List[Dict] = field(default_factory=list)
    psyche_snapshot: Dict[str, Any] = field(default_factory=dict)

    def history_count(self) -> int:
        return len(self.message_history)

    def has_memories(self) -> bool:
        return len(self.retrieved_memories) > 0


@dataclass
class ImageAnalysisRequest(BaseDataModel):
    """图片分析请求"""

    msg_index: int = 0
    image_urls: List[str] = field(default_factory=list)
    analysis_hint: str = ""


@dataclass
class PerceptionOutcome(BaseDataModel):
    """感知系统最终输出"""

    decided_action: str = ""
    inner_thought: str = ""
    reasoning: str = ""
    reply_bundle: Optional[ReplyContextBundle] = None
    generated_text: str = ""
    image_requests: List[ImageAnalysisRequest] = field(default_factory=list)
    behavior_mode: str = ""
    thinking_score: float = 0.0
    suggested_duration_sec: int = 0
    needs_image_callback: bool = False

    def should_reply(self) -> bool:
        return self.decided_action in ("reply", "start_topic")

    def is_observation(self) -> bool:
        return self.decided_action in ("observe", "")

    def summary_line(self) -> str:
        tail = self.reasoning[:60] if self.reasoning else ""
        return f"[{self.decided_action}] {tail}"


# ==========================================================================
#  内心活动事件日志 & 心流状态容器（B10-2 补全）
# ==========================================================================


class ActivityKind(Enum):
    """心流内部事件条目类型"""

    PEER_SPOKE = "peer_spoke"
    SELF_CONSIDERED = "self_considered"
    SELF_REPLIED = "self_replied"
    IDLE_ENTERED = "idle_entered"
    IDLE_REFRESHED = "idle_refreshed"
    IDLE_EXPIRED = "idle_expired"
    INITIATIVE_FIRED = "initiative_fired"
    NAP_STARTED = "nap_started"
    NAP_ENDED = "nap_ended"
    NAP_PEEKED = "nap_peeked"

    def __str__(self) -> str:
        return self.value


@dataclass
class MindLogEntry(BaseDataModel):
    """单条内心活动日志条目"""

    kind: ActivityKind = ActivityKind.PEER_SPOKE
    occurred_at: float = 0.0
    body: str = ""
    speaker_name: str = ""
    speaker_id: str = ""
    reflection: str = ""
    action_list: List[Dict] = field(default_factory=list)
    anticipated: str = ""
    max_hold_sec: int = 0
    hold_elapsed: float = 0.0
    musing_text: str = ""
    mood_tag: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def clock_str(self, fmt: str = "%H:%M") -> str:
        if self.occurred_at <= 0:
            return "--:--"
        return time.strftime(fmt, time.localtime(self.occurred_at))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": str(self.kind),
            "occurred_at": self.occurred_at,
            "body": self.body,
            "speaker_name": self.speaker_name,
            "speaker_id": self.speaker_id,
            "reflection": self.reflection,
            "action_list": self.action_list,
            "anticipated": self.anticipated,
            "max_hold_sec": self.max_hold_sec,
            "hold_elapsed": self.hold_elapsed,
            "musing_text": self.musing_text,
            "mood_tag": self.mood_tag,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MindLogEntry":
        kind_raw = data.get("kind", "peer_spoke")
        try:
            kind = ActivityKind(kind_raw)
        except ValueError:
            kind = ActivityKind.PEER_SPOKE
        return cls(
            kind=kind,
            occurred_at=data.get("occurred_at", time.time()),
            body=data.get("body", ""),
            speaker_name=data.get("speaker_name", ""),
            speaker_id=data.get("speaker_id", ""),
            reflection=data.get("reflection", ""),
            action_list=data.get("action_list", []),
            anticipated=data.get("anticipated", ""),
            max_hold_sec=data.get("max_hold_sec", 0),
            hold_elapsed=data.get("hold_elapsed", 0.0),
            musing_text=data.get("musing_text", ""),
            mood_tag=data.get("mood_tag", ""),
            meta=data.get("meta", {}),
        )


_LOG_CAP = 100
_LOG_TRIM = 80


@dataclass
class FlowStateContainer(BaseDataModel):
    """心流运行状态容器，管理当前阶段、等待/休息参数和内心日志"""

    channel_ref: str = ""
    phase: FlowPhase = FlowPhase.STANDBY
    current_reflection: str = ""
    idle_setup: IdleSetup = field(default_factory=IdleSetup)
    dormant_setup: DormantSetup = field(default_factory=DormantSetup)
    mind_log: List[MindLogEntry] = field(default_factory=list)
    last_active_at: float = field(default_factory=time.time)
    last_initiative_at: float = 0.0
    consecutive_timeouts: int = 0
    # ---- 状态查询 ----

    def is_in_idle(self) -> bool:
        return self.phase == FlowPhase.PENDING and self.idle_setup.entered_at > 0

    def is_in_nap(self) -> bool:
        return self.phase == FlowPhase.DORMANT and self.dormant_setup.entered_at > 0

    # ---- 等待生命周期 ----

    def enter_idle(self, anticipated: str, hold_sec: int) -> None:
        self.phase = FlowPhase.PENDING
        self.idle_setup = IdleSetup(
            anticipated_response=anticipated,
            max_wait_sec=hold_sec,
            entered_at=time.time(),
        )
        self._append_entry(
            MindLogEntry(
                kind=ActivityKind.IDLE_ENTERED,
                occurred_at=time.time(),
                anticipated=anticipated,
                max_hold_sec=hold_sec,
            )
        )

    def leave_idle(self) -> None:
        self.phase = FlowPhase.STANDBY
        self.idle_setup.clear()
        self.consecutive_timeouts = 0

    # ---- 休息生命周期 ----

    def enter_nap(self, cause: str, duration_sec: int, interruptible: bool = True) -> None:
        self.phase = FlowPhase.DORMANT
        self.dormant_setup = DormantSetup(
            cause=cause,
            duration_sec=duration_sec,
            entered_at=time.time(),
            allow_awakening=interruptible,
        )
        self._append_entry(
            MindLogEntry(
                kind=ActivityKind.NAP_STARTED,
                occurred_at=time.time(),
                meta={"cause": cause, "duration": duration_sec},
            )
        )

    def leave_nap(self) -> None:
        self.phase = FlowPhase.STANDBY
        self.dormant_setup = DormantSetup()
        self._append_entry(
            MindLogEntry(
                kind=ActivityKind.NAP_ENDED,
                occurred_at=time.time(),
            )
        )

    # ---- 事件登记 ----

    def _append_entry(self, entry: MindLogEntry) -> None:
        self.mind_log.append(entry)
        self.last_active_at = time.time()
        if len(self.mind_log) > _LOG_CAP:
            self.mind_log = self.mind_log[-_LOG_TRIM:]

    def log_peer_message(self, content: str, name: str, uid: str) -> None:
        self._append_entry(
            MindLogEntry(
                kind=ActivityKind.PEER_SPOKE,
                occurred_at=time.time(),
                body=content,
                speaker_name=name,
                speaker_id=uid,
            )
        )
        if self.is_in_idle():
            self.leave_idle()
        if self.is_in_nap() and self.dormant_setup.allow_awakening:
            self.leave_nap()

    def log_self_thinking(
        self,
        reflection: str,
        actions: List[Dict],
        anticipated: str = "",
        hold_sec: int = 0,
    ) -> None:
        self._append_entry(
            MindLogEntry(
                kind=ActivityKind.SELF_CONSIDERED,
                occurred_at=time.time(),
                reflection=reflection,
                action_list=actions,
                anticipated=anticipated,
                max_hold_sec=hold_sec,
            )
        )
        if hold_sec > 0:
            self.enter_idle(anticipated, hold_sec)

    def log_idle_musing(self, musing: str, mood: str = "") -> None:
        self._append_entry(
            MindLogEntry(
                kind=ActivityKind.IDLE_REFRESHED,
                occurred_at=time.time(),
                musing_text=musing,
                mood_tag=mood,
                hold_elapsed=self.idle_setup.elapsed_sec(),
            )
        )
        self.idle_setup.record_contemplate()

    def recent_entries(self, cap: int = 20) -> List[MindLogEntry]:
        return self.mind_log[-cap:]

    def last_self_reply_body(self) -> Optional[str]:
        for entry in reversed(self.mind_log):
            if entry.kind == ActivityKind.SELF_REPLIED:
                return entry.body
            if entry.kind == ActivityKind.SELF_CONSIDERED and entry.action_list:
                for act in entry.action_list:
                    if act.get("type") == "reply":
                        return act.get("content", "")
        return None

    # ---- 序列化 ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_ref": self.channel_ref,
            "phase": self.phase.value,
            "current_reflection": self.current_reflection,
            "idle_setup": self.idle_setup.to_dict(),
            "dormant_setup": self.dormant_setup.to_dict(),
            "mind_log": [e.to_dict() for e in self.mind_log[-50:]],
            "last_active_at": self.last_active_at,
            "last_initiative_at": self.last_initiative_at,
            "consecutive_timeouts": self.consecutive_timeouts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlowStateContainer":
        phase_raw = data.get("phase", "standby")
        try:
            phase = FlowPhase(phase_raw)
        except ValueError:
            phase = FlowPhase.STANDBY
        log_raw = data.get("mind_log", [])
        entries = [MindLogEntry.from_dict(e) for e in log_raw]
        idle_raw = data.get("idle_setup", {})
        dormant_raw = data.get("dormant_setup", {})
        return cls(
            channel_ref=data.get("channel_ref", "") or data.get("stream_id", ""),
            phase=phase,
            current_reflection=data.get("current_reflection", ""),
            idle_setup=IdleSetup.from_dict(idle_raw),
            dormant_setup=DormantSetup.from_dict(dormant_raw),
            mind_log=entries,
            last_active_at=data.get("last_active_at", time.time()),
            last_initiative_at=data.get("last_initiative_at", 0.0),
            consecutive_timeouts=data.get("consecutive_timeouts", 0),
        )


@dataclass
class UnifiedFlowSnapshot(BaseDataModel):
    """统一行为状态快照，作为单入口状态更新的载体。"""

    channel_id: str = ""
    phase: str = FlowPhase.STANDBY.value
    watch_state: str = "unknown"
    is_resting: bool = False
    rest_remaining_sec: float = 0.0
    rest_intensity: float = 0.0
    wake_drive: float = 0.0
    reply_desire: int = 5
    voice_action: str = ""
    voice_should_reply: Optional[bool] = None
    voice_mood: str = ""
    voice_thinking: str = ""
    boredom: float = 0.0
    environmental_fatigue: float = 0.0
    loneliness: float = 0.0
    social_desire: float = 0.0
    blocker: str = ""
    final_decision: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "phase": self.phase,
            "watch_state": self.watch_state,
            "is_resting": self.is_resting,
            "rest_remaining_sec": round(float(self.rest_remaining_sec), 1),
            "rest_intensity": round(float(self.rest_intensity), 3),
            "wake_drive": round(float(self.wake_drive), 3),
            "reply_desire": int(self.reply_desire),
            "voice_action": self.voice_action,
            "voice_should_reply": self.voice_should_reply,
            "voice_mood": self.voice_mood,
            "voice_thinking": self.voice_thinking,
            "boredom": round(float(self.boredom), 3),
            "environmental_fatigue": round(float(self.environmental_fatigue), 3),
            "loneliness": round(float(self.loneliness), 3),
            "social_desire": round(float(self.social_desire), 3),
            "blocker": self.blocker,
            "final_decision": self.final_decision,
        }

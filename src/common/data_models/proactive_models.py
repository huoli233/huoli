import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from src.common.data_models.base_data_model import BaseDataModel


class JournalKind(Enum):
    """活动流日志类型标记"""

    USER_UTTERANCE = "user_utterance"
    BOT_VERDICT = "bot_verdict"
    IDLE_ENTER = "idle_enter"
    IDLE_SHIFT = "idle_shift"
    TIMELY_DISPATCH = "timely_dispatch"
    LATE_DISPATCH = "late_dispatch"
    IDLE_EXPIRED = "idle_expired"
    SPONTANEOUS_IMPULSE = "spontaneous_impulse"

    def __str__(self) -> str:
        return self.value

    def is_user_event(self) -> bool:
        return self == JournalKind.USER_UTTERANCE

    def is_bot_event(self) -> bool:
        return self in (
            JournalKind.BOT_VERDICT,
            JournalKind.TIMELY_DISPATCH,
            JournalKind.LATE_DISPATCH,
        )

    def is_idle_event(self) -> bool:
        return self in (
            JournalKind.IDLE_ENTER,
            JournalKind.IDLE_SHIFT,
            JournalKind.IDLE_EXPIRED,
        )


class DialogueStage(Enum):
    """会话当前阶段"""

    OPEN = "open"
    ANTICIPATING = "anticipating"
    # 兼容心流侧历史命名
    IDLE = "idle"
    ACTIVE = "active"
    COOLING = "cooling"

    def __str__(self) -> str:
        return self.value

    def is_waiting(self) -> bool:
        return self == DialogueStage.ANTICIPATING


@dataclass
class TriggerSignal(BaseDataModel):
    """主动回复触发信号，由调度器产生"""

    signal_tag: str = ""
    confidence: float = 0.0
    source_module: str = ""
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_strong(self, bar: float = 0.0) -> bool:
        if bar <= 0:
            return False
        return self.confidence >= bar

    def age_sec(self) -> float:
        if self.created_at <= 0:
            return 0.0
        return time.time() - self.created_at

    def describe(self) -> str:
        age = self.age_sec()
        return f"[{self.signal_tag}] conf={self.confidence:.2f} from={self.source_module} age={age:.0f}s"


@dataclass
class ThinkTask(BaseDataModel):
    """思维调度器分派的任务"""

    task_id: str = ""
    category: str = ""
    prompt_text: str = ""
    channel_id: str = ""
    target_user_id: str = ""
    issued_at: float = 0.0
    deadline_sec: int = 0
    priority_rank: int = 0
    auxiliary: Dict[str, Any] = field(default_factory=dict)

    def is_overdue(self) -> bool:
        if self.deadline_sec <= 0 or self.issued_at <= 0:
            return False
        return (time.time() - self.issued_at) >= self.deadline_sec

    def remaining_sec(self) -> float:
        if self.deadline_sec <= 0 or self.issued_at <= 0:
            return 0.0
        return max(0.0, self.deadline_sec - (time.time() - self.issued_at))


@dataclass
class ThinkOutcome(BaseDataModel):
    """思维执行器返回的结果"""

    task_id: str = ""
    success: bool = False
    raw_output: str = ""
    parsed_thought: str = ""
    action_tag: str = ""
    content_body: str = ""
    mood_shift: str = ""
    latency_ms: float = 0.0
    error_msg: Optional[str] = None

    def is_actionable(self) -> bool:
        return self.success and self.action_tag not in ("", "noop", "observe")

    def summary_line(self) -> str:
        if not self.success:
            return f"FAIL: {self.error_msg or 'unknown'}"
        return f"[{self.action_tag}] {self.parsed_thought[:60]}"


@dataclass
class MindJournalEntry(BaseDataModel):
    """心理活动日志行"""

    kind: JournalKind = JournalKind.USER_UTTERANCE
    written_at: float = 0.0
    body_text: str = ""
    speaker_name: str = ""
    speaker_id: str = ""
    inner_monologue: str = ""
    action_list: List[Dict] = field(default_factory=list)
    expected_response: str = ""
    wait_limit_sec: int = 0
    elapsed_wait: float = 0.0
    reflection: str = ""
    mood_label: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    def stamp_display(self, fmt: str = "%H:%M") -> str:
        if self.written_at <= 0:
            return "--:--"
        return time.strftime(fmt, time.localtime(self.written_at))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": str(self.kind),
            "written_at": self.written_at,
            "body_text": self.body_text,
            "speaker_name": self.speaker_name,
            "speaker_id": self.speaker_id,
            "inner_monologue": self.inner_monologue,
            "action_list": self.action_list,
            "expected_response": self.expected_response,
            "wait_limit_sec": self.wait_limit_sec,
            "elapsed_wait": self.elapsed_wait,
            "reflection": self.reflection,
            "mood_label": self.mood_label,
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MindJournalEntry":
        kind_raw = data.get("kind", "user_utterance")
        try:
            kind = JournalKind(kind_raw)
        except ValueError:
            kind = JournalKind.USER_UTTERANCE
        return cls(
            kind=kind,
            written_at=data.get("written_at", 0.0),
            body_text=data.get("body_text", ""),
            speaker_name=data.get("speaker_name", ""),
            speaker_id=data.get("speaker_id", ""),
            inner_monologue=data.get("inner_monologue", ""),
            action_list=data.get("action_list", []),
            expected_response=data.get("expected_response", ""),
            wait_limit_sec=data.get("wait_limit_sec", 0),
            elapsed_wait=data.get("elapsed_wait", 0.0),
            reflection=data.get("reflection", ""),
            mood_label=data.get("mood_label", ""),
            extras=data.get("extras", {}),
        )


@dataclass
class SessionContext(BaseDataModel):
    """会话追踪上下文"""

    channel_id: str = ""
    stage: DialogueStage = DialogueStage.OPEN
    journal: List[MindJournalEntry] = field(default_factory=list)
    active_speakers: Dict[str, float] = field(default_factory=dict)
    last_bot_reply_at: float = 0.0
    consecutive_silence_rounds: int = 0
    topic_anchor: str = ""

    def journal_length(self) -> int:
        return len(self.journal)

    def append_entry(self, entry: MindJournalEntry) -> None:
        self.journal.append(entry)

    def trim_journal(self, keep_last: int = 0) -> None:
        if keep_last <= 0 or len(self.journal) <= keep_last:
            return
        self.journal = self.journal[-keep_last:]

    def since_last_reply_sec(self) -> float:
        if self.last_bot_reply_at <= 0:
            return 0.0
        return time.time() - self.last_bot_reply_at

    def record_bot_reply(self) -> None:
        self.last_bot_reply_at = time.time()
        self.consecutive_silence_rounds = 0

    def record_silence_round(self) -> None:
        self.consecutive_silence_rounds += 1

    def touch_speaker(self, uid: str) -> None:
        self.active_speakers[uid] = time.time()

    def recent_speakers(self, within_sec: float = 0.0) -> List[str]:
        if within_sec <= 0:
            return list(self.active_speakers.keys())
        cutoff = time.time() - within_sec
        return [
            uid for uid, ts in self.active_speakers.items() if ts >= cutoff
        ]

    def latest_entries(self, count: int = 0) -> List[MindJournalEntry]:
        if count <= 0:
            return list(self.journal)
        return self.journal[-count:]

    def latest_user_entries(self, count: int = 0) -> List[MindJournalEntry]:
        user_entries = [
            e for e in self.journal if e.kind == JournalKind.USER_UTTERANCE
        ]
        if count <= 0:
            return user_entries
        return user_entries[-count:]


@dataclass
class DeliveryPayload(BaseDataModel):
    """回复分发载荷"""

    channel_id: str = ""
    content_text: str = ""
    reply_to_msg_id: str = ""
    sticker_id: str = ""
    delay_sec: float = 0.0
    segment_index: int = 0
    total_segments: int = 0
    is_proactive: bool = False

    def is_text_reply(self) -> bool:
        return bool(self.content_text)

    def is_sticker(self) -> bool:
        return bool(self.sticker_id)

    def has_delay(self) -> bool:
        return self.delay_sec > 0

    def describe(self) -> str:
        if self.is_sticker():
            return f"sticker:{self.sticker_id}"
        preview = (
            self.content_text[:30] + "..."
            if len(self.content_text) > 30
            else self.content_text
        )
        return f'text:"{preview}"'


@dataclass
class ExecutionReport(BaseDataModel):
    """一次完整主动思考周期的执行报告"""

    cycle_id: str = ""
    channel_id: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    trigger: Optional[TriggerSignal] = None
    think_result: Optional[ThinkOutcome] = None
    deliveries: List[DeliveryPayload] = field(default_factory=list)
    final_status: str = ""
    error_note: Optional[str] = None

    def duration_ms(self) -> float:
        if self.started_at <= 0 or self.finished_at <= 0:
            return 0.0
        return (self.finished_at - self.started_at) * 1000

    def did_reply(self) -> bool:
        return any(d.is_text_reply() for d in self.deliveries)

    def delivery_count(self) -> int:
        return len(self.deliveries)

    def is_success(self) -> bool:
        return self.final_status == "ok"

    def summary(self) -> str:
        dur = self.duration_ms()
        tag = self.trigger.signal_tag if self.trigger else "none"
        return f"cycle={
            self.cycle_id} trigger={tag} status={
            self.final_status} dur={
            dur:.0f}ms replies={
                self.delivery_count()}"


@dataclass
class SocialValuation(BaseDataModel):
    """用户社交值评估"""

    user_id: str = ""
    channel_id: str = ""
    composite_score: float = 0.0
    affinity_dim: float = 0.0
    aversion_dim: float = 0.0
    credibility: float = 0.0
    irritability: float = 0.0
    interaction_tally: int = 0
    last_seen_at: float = 0.0
    first_seen_at: float = 0.0
    refreshed_at: float = 0.0

    def net_sentiment(self) -> float:
        return self.affinity_dim - self.aversion_dim

    def is_trusted(self, bar: float = 0.0) -> bool:
        if bar <= 0:
            return False
        return self.credibility >= bar

    def is_irritated(self, bar: float = 0.0) -> bool:
        if bar <= 0:
            return False
        return self.irritability >= bar

    def days_since_first(self) -> float:
        if self.first_seen_at <= 0:
            return 0.0
        return (time.time() - self.first_seen_at) / 86400.0


@dataclass
class SocialAdjustment(BaseDataModel):
    """社交值单次调整结果"""

    prev_score: float = 0.0
    curr_score: float = 0.0
    delta: float = 0.0
    behavior_tag: str = ""
    intent_label: str = ""
    severity: float = 0.0
    mapping_category: str = ""
    applied_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MappingWeights(BaseDataModel):
    """四层映射最终权重输出"""

    category: str = ""
    base_score: float = 0.0
    intent_coef: float = 0.0
    ceiling_per_round: float = 0.0
    accel_bar: float = 0.0
    accel_coef: float = 0.0
    decay_speed: float = 0.0
    trauma_coef: float = 0.0
    type_boost: float = 0.0


@dataclass
class CategoryWeightSet(BaseDataModel):
    """单个类别的权重配置"""

    base_value: float = 0.0
    round_ceiling: float = 0.0
    accel_threshold: float = 0.0
    accel_factor: float = 0.0
    natural_decay: float = 0.0
    trauma_effect: float = 0.0
    intent_map: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScheduleTicket(BaseDataModel):
    """调度器周期性计划"""

    ticket_id: str = ""
    interval_sec: int = 0
    last_fire_at: float = 0.0
    jitter_sec: int = 0
    enabled: bool = False

    def next_fire_at(self) -> float:
        if self.last_fire_at <= 0:
            return 0.0
        return self.last_fire_at + self.interval_sec

    def is_due(self) -> bool:
        if not self.enabled or self.interval_sec <= 0:
            return False
        if self.last_fire_at <= 0:
            return True
        return time.time() >= self.next_fire_at()

    def mark_fired(self) -> None:
        self.last_fire_at = time.time()


@dataclass
class SilenceCandidate(BaseDataModel):
    """静默检测候选"""

    channel_id: str = ""
    quiet_sec: float = 0.0
    last_msg_at: float = 0.0
    speaker_count_before: int = 0
    topic_tag: str = ""

    def exceeds_bar(self, bar_sec: float = 0.0) -> bool:
        if bar_sec <= 0:
            return False
        return self.quiet_sec >= bar_sec


@dataclass
class ProactiveDecision(BaseDataModel):
    """主动回复最终决策"""

    should_act: bool = False
    action_type: str = ""
    target_channel: str = ""
    target_user: str = ""
    content_hint: str = ""
    trigger_source: str = ""
    confidence: float = 0.0
    wait_before_sec: float = 0.0

    def describe(self) -> str:
        if not self.should_act:
            return "noop"
        return f"[{self.action_type}] ch={self.target_channel} conf={self.confidence:.2f} hint={self.content_hint[:30]}"

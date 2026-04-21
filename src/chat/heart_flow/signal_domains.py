from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from src.chat.heart_flow.dimension_protocol import DimensionVote


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


class AffinitySignalLevel(Enum):
    EXTREME_DISLIKE = "extreme_dislike"
    DISLIKE = "dislike"
    DISSATISFIED = "dissatisfied"
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    FOND = "fond"
    ADORE = "adore"


class TrustSignalLevel(Enum):
    TOTAL_DISTRUST = "total_distrust"
    DISTRUST = "distrust"
    DOUBTFUL = "doubtful"
    ORDINARY = "ordinary"
    FAVORABLE = "favorable"
    TRUSTED = "trusted"
    FULL_TRUST = "full_trust"


class CounterpartyEngagementState(Enum):
    EAGER = "eager"
    NEUTRAL = "neutral"
    RELUCTANT = "reluctant"
    HOSTILE = "hostile"
    ABSENT = "absent"


class GroupContextTier(Enum):
    HEATED = "heated"
    LIVELY = "lively"
    NORMAL = "normal"
    QUIET = "quiet"
    DEAD = "dead"


class AgentContextRole(Enum):
    CORE = "core"
    PARTICIPANT = "participant"
    OBSERVER = "observer"
    OUTSIDER = "outsider"


class ResourceLedgerStage(Enum):
    FULL = "full"
    ADEQUATE = "adequate"
    LOW = "low"
    CRITICAL = "critical"
    DEPLETED = "depleted"


class PendingEngagementPhase(Enum):
    IDLE = "idle"
    WAITING = "waiting"
    TIMEOUT = "timeout"


class RuntimeCalibrationBand(Enum):
    SKIP = "skip"
    LIGHT = "light"
    MODERATE = "moderate"
    DEEP = "deep"
    FULL_RESET = "full_reset"


class TraumaLoadBand(Enum):
    NONE = "none"
    MILD = "mild"
    SEVERE = "severe"
    EXTREME = "extreme"


class SurfaceMaskMode(Enum):
    NATURAL = "natural"
    MASKING = "masking"
    CRACKING = "cracking"
    BROKEN = "broken"


@dataclass
class EmotionBaselineSignal(DimensionVote):
    dimension_name: str = "emotion_baseline"
    legacy_name: str = field(default="EmotionAxisVote", init=False, repr=False)
    patience_snapshot: float = 100.0
    annoyance_snapshot: float = 0.0
    fatigue_snapshot: float = 0.0
    mood_modifier: str = ""
    should_refuse: bool = False
    emotion_proactive_willingness: float = 0.0

    @property
    def patience_level(self) -> float:
        return self.patience_snapshot

    @property
    def annoyance_level(self) -> float:
        return self.annoyance_snapshot

    @property
    def fatigue_level(self) -> float:
        return self.fatigue_snapshot

    @property
    def proactive_pull(self) -> float:
        return self.emotion_proactive_willingness

    def to_domain_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.dimension_name,
            "legacy_name": self.legacy_name,
            "patience_level": self.patience_level,
            "annoyance_level": self.annoyance_level,
            "fatigue_level": self.fatigue_level,
            "mood_modifier": self.mood_modifier,
            "refusal_requested": self.should_refuse,
            "proactive_pull": self.proactive_pull,
            "probability_factor": self.probability_factor,
        }


@dataclass
class RapportTrustSignal(DimensionVote):
    dimension_name: str = "rapport_trust"
    legacy_name: str = field(default="FondnessTrustVote", init=False, repr=False)
    fondness_value: float = 0.0
    fondness_level: AffinitySignalLevel = AffinitySignalLevel.NEUTRAL
    fondness_factor: float = 0.85
    trust_value: float = 0.0
    trust_level: TrustSignalLevel = TrustSignalLevel.ORDINARY
    trust_factor: float = 1.0
    cross_hint: str = ""
    proactive_willingness: float = 0.3

    @property
    def affinity_score(self) -> float:
        return self.fondness_value

    @property
    def affinity_level(self) -> AffinitySignalLevel:
        return self.fondness_level

    @property
    def affinity_factor(self) -> float:
        return self.fondness_factor

    @property
    def trust_score(self) -> float:
        return self.trust_value

    @property
    def trust_band(self) -> TrustSignalLevel:
        return self.trust_level

    @property
    def rapport_hint(self) -> str:
        return self.cross_hint

    @property
    def proactive_pull(self) -> float:
        return self.proactive_willingness

    def to_domain_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.dimension_name,
            "legacy_name": self.legacy_name,
            "affinity_score": self.affinity_score,
            "affinity_level": self.affinity_level.value,
            "affinity_factor": self.affinity_factor,
            "trust_score": self.trust_score,
            "trust_band": self.trust_band.value,
            "trust_factor": self.trust_factor,
            "rapport_hint": self.rapport_hint,
            "proactive_pull": self.proactive_pull,
            "probability_factor": self.probability_factor,
        }


@dataclass
class BoundaryIncidentRecord:
    category: str = ""
    severity: float = 0.0
    timestamp: float = 0.0
    description: str = ""
    decay_rate: float = 0.01


@dataclass
class BoundaryBlockSignal(DimensionVote):
    dimension_name: str = "boundary_block"
    legacy_name: str = field(default="DislikeVote", init=False, repr=False)
    is_blocked: bool = False
    dislike_penalty: float = 0.0
    total_severity: float = 0.0
    worst_category: str = ""
    block_threshold: float = 30.0
    proactive_suppression: float = 0.0

    def to_domain_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.dimension_name,
            "legacy_name": self.legacy_name,
            "blocked": self.is_blocked,
            "penalty": self.dislike_penalty,
            "severity_total": self.total_severity,
            "dominant_category": self.worst_category,
            "threshold": self.block_threshold,
            "proactive_drag": self.proactive_suppression,
            "probability_factor": self.probability_factor,
        }


@dataclass
class TempoControlSignal(DimensionVote):
    dimension_name: str = "tempo_control"
    legacy_name: str = field(default="FrequencyVote", init=False, repr=False)
    adjust_factor: float = 1.0
    dynamic_threshold: float = 0.5
    in_cooldown: bool = False
    consecutive_skip_count: int = 0
    skip_limit: int = 5
    seconds_since_last_reply: float = 0.0

    def to_domain_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.dimension_name,
            "legacy_name": self.legacy_name,
            "tempo_factor": self.adjust_factor,
            "adaptive_threshold": self.dynamic_threshold,
            "cooling_down": self.in_cooldown,
            "ignore_streak": self.consecutive_skip_count,
            "ignore_limit": self.skip_limit,
            "seconds_since_last_reply": self.seconds_since_last_reply,
            "probability_factor": self.probability_factor,
        }


@dataclass
class CounterpartyReadinessSignal(DimensionVote):
    dimension_name: str = "counterparty_readiness"
    legacy_name: str = field(default="UserStateVote", init=False, repr=False)
    engagement: CounterpartyEngagementState = CounterpartyEngagementState.NEUTRAL
    want_to_chat_confidence: float = 0.5
    msg_frequency: float = 0.0
    avg_msg_length: float = 0.0
    question_ratio: float = 0.0
    structural_hostility: bool = False

    def to_domain_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.dimension_name,
            "legacy_name": self.legacy_name,
            "engagement_state": self.engagement.value,
            "readiness_confidence": self.want_to_chat_confidence,
            "message_frequency": self.msg_frequency,
            "average_message_length": self.avg_msg_length,
            "question_density": self.question_ratio,
            "hostility_detected": self.structural_hostility,
            "probability_factor": self.probability_factor,
        }


@dataclass
class GroupContextSignal(DimensionVote):
    dimension_name: str = "group_context"
    legacy_name: str = field(default="GroupAtmosphereVote", init=False, repr=False)
    atmosphere_tier: GroupContextTier = GroupContextTier.NORMAL
    ai_position: AgentContextRole = AgentContextRole.OBSERVER
    engagement_boost: float = 0.0
    active_user_count: int = 0
    topic_keywords: list[str] = field(default_factory=list)
    style_constraint: str = ""
    topic_relevance: float = 0.5

    def to_domain_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.dimension_name,
            "legacy_name": self.legacy_name,
            "context_tier": self.atmosphere_tier.value,
            "agent_role": self.ai_position.value,
            "engagement_bias": self.engagement_boost,
            "active_participants": self.active_user_count,
            "topic_tokens": list(self.topic_keywords),
            "style_constraint": self.style_constraint,
            "relevance_score": self.topic_relevance,
            "probability_factor": self.probability_factor,
        }


@dataclass
class ResourceLedgerSignal(DimensionVote):
    dimension_name: str = "resource_ledger"
    legacy_name: str = field(default="EnergyChainVote", init=False, repr=False)
    chat_energy_ratio: float = 1.0
    thinking_energy_ratio: float = 1.0
    energy_stage: ResourceLedgerStage = ResourceLedgerStage.FULL
    estimated_cost: float = 0.05
    night_mode: bool = False
    recovery_damping: float = 1.0
    chat_ceiling: float = 100.0
    chat_current: float = 100.0

    def to_domain_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.dimension_name,
            "legacy_name": self.legacy_name,
            "chat_ratio": self.chat_energy_ratio,
            "thinking_ratio": self.thinking_energy_ratio,
            "stage": self.energy_stage.value,
            "expected_cost": self.estimated_cost,
            "low_power_mode": self.night_mode,
            "recovery_factor": self.recovery_damping,
            "chat_capacity": self.chat_ceiling,
            "chat_available": self.chat_current,
            "probability_factor": self.probability_factor,
        }


@dataclass
class SocialBalanceSignal(DimensionVote):
    dimension_name: str = "social_balance"
    legacy_name: str = field(default="SocialValueVote", init=False, repr=False)
    social_score: float = 0.0
    social_tag: str = ""
    in_alert: bool = False
    positive_interactions: int = 0
    negative_interactions: int = 0
    proactive_boost: float = 0.0

    def to_domain_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.dimension_name,
            "legacy_name": self.legacy_name,
            "balance_score": self.social_score,
            "balance_tag": self.social_tag,
            "alerting": self.in_alert,
            "positive_count": self.positive_interactions,
            "negative_count": self.negative_interactions,
            "proactive_bias": self.proactive_boost,
            "probability_factor": self.probability_factor,
        }


@dataclass
class PendingEngagementSignal(DimensionVote):
    dimension_name: str = "pending_engagement"
    legacy_name: str = field(default="HeartStateVote", init=False, repr=False)
    wait_phase: PendingEngagementPhase = PendingEngagementPhase.IDLE
    should_initiate: bool = False
    waiting_seconds: float = 0.0
    timeout_threshold: float = 300.0
    last_interaction_channel: str = ""
    heartbeat_proactive_willingness: float = 0.0

    def to_domain_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.dimension_name,
            "legacy_name": self.legacy_name,
            "pending_phase": self.wait_phase.value,
            "should_reengage": self.should_initiate,
            "pending_seconds": self.waiting_seconds,
            "pending_threshold": self.timeout_threshold,
            "last_channel": self.last_interaction_channel,
            "proactive_pull": self.heartbeat_proactive_willingness,
            "probability_factor": self.probability_factor,
        }


@dataclass
class RuntimeCalibrationProfile:
    legacy_name: str = field(default="CalibrationReport", init=False, repr=False)
    grade: RuntimeCalibrationBand = RuntimeCalibrationBand.SKIP
    offline_seconds: float = 0.0
    emotion_decay_factor: float = 0.0
    fondness_decay_factor: float = 0.0
    energy_recovery_factor: float = 0.0
    trauma_heal_factor: float = 0.0
    mask_fatigue_recovery: float = 0.0
    frequency_should_reset: bool = False

    def to_domain_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": "runtime_calibration",
            "legacy_name": self.legacy_name,
            "band": self.grade.value,
            "offline_seconds": self.offline_seconds,
            "emotion_decay_factor": self.emotion_decay_factor,
            "rapport_decay_factor": self.fondness_decay_factor,
            "energy_recovery_factor": self.energy_recovery_factor,
            "trauma_heal_factor": self.trauma_heal_factor,
            "mask_recovery_factor": self.mask_fatigue_recovery,
            "tempo_should_reset": self.frequency_should_reset,
        }


@dataclass
class TraumaLoadSignal(DimensionVote):
    dimension_name: str = "trauma_load"
    legacy_name: str = field(default="TraumaVote", init=False, repr=False)
    trauma_index: float = 0.0
    trauma_grade: TraumaLoadBand = TraumaLoadBand.NONE
    cognitive_impairment: float = 0.0
    flashback_risk: float = 0.0
    triggered_fragment_labels: list[str] = field(default_factory=list)
    perception_labels: list[str] = field(default_factory=list)
    inner_chaos: float = 0.0

    def to_domain_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.dimension_name,
            "legacy_name": self.legacy_name,
            "load_index": self.trauma_index,
            "load_band": self.trauma_grade.value,
            "cognitive_drag": self.cognitive_impairment,
            "flashback_probability": self.flashback_risk,
            "triggered_fragments": list(self.triggered_fragment_labels),
            "subconscious_hints": list(self.perception_labels),
            "chaos_level": self.inner_chaos,
            "probability_factor": self.probability_factor,
        }


@dataclass
class SurfaceMaskSignal(DimensionVote):
    dimension_name: str = "surface_mask"
    legacy_name: str = field(default="SurfaceMaskVote", init=False, repr=False)
    mask_strength: float = 0.0
    behavior_mode: SurfaceMaskMode = SurfaceMaskMode.NATURAL
    is_breaking: bool = False
    reply_quality_cap: float = 1.0
    mask_fatigue: float = 0.0
    detection_risk: float = 0.0
    inner_conflict_hint: str = ""
    matrix_row: int = 0
    matrix_col: int = 0

    def to_domain_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.dimension_name,
            "legacy_name": self.legacy_name,
            "mask_strength": self.mask_strength,
            "mode": self.behavior_mode.value,
            "cracking": self.is_breaking,
            "reply_quality_cap": self.reply_quality_cap,
            "fatigue": self.mask_fatigue,
            "detection_probability": self.detection_risk,
            "inner_conflict_hint": self.inner_conflict_hint,
            "matrix_row": self.matrix_row,
            "matrix_col": self.matrix_col,
            "probability_factor": self.probability_factor,
        }


@dataclass
class DecisionSignalBundle:
    emotion_baseline: EmotionBaselineSignal = field(default_factory=EmotionBaselineSignal)
    rapport_trust: RapportTrustSignal = field(default_factory=RapportTrustSignal)
    boundary_block: BoundaryBlockSignal = field(default_factory=BoundaryBlockSignal)
    tempo_control: TempoControlSignal = field(default_factory=TempoControlSignal)
    counterparty_readiness: CounterpartyReadinessSignal = field(default_factory=CounterpartyReadinessSignal)
    group_context: GroupContextSignal = field(default_factory=GroupContextSignal)
    resource_ledger: ResourceLedgerSignal = field(default_factory=ResourceLedgerSignal)
    social_balance: SocialBalanceSignal = field(default_factory=SocialBalanceSignal)
    pending_engagement: PendingEngagementSignal = field(default_factory=PendingEngagementSignal)
    trauma_load: TraumaLoadSignal = field(default_factory=TraumaLoadSignal)
    surface_mask: SurfaceMaskSignal = field(default_factory=SurfaceMaskSignal)

    @property
    def emotion(self) -> EmotionBaselineSignal:
        return self.emotion_baseline

    @emotion.setter
    def emotion(self, value: EmotionBaselineSignal) -> None:
        self.emotion_baseline = value

    @property
    def fondness_trust(self) -> RapportTrustSignal:
        return self.rapport_trust

    @fondness_trust.setter
    def fondness_trust(self, value: RapportTrustSignal) -> None:
        self.rapport_trust = value

    @property
    def dislike(self) -> BoundaryBlockSignal:
        return self.boundary_block

    @dislike.setter
    def dislike(self, value: BoundaryBlockSignal) -> None:
        self.boundary_block = value

    @property
    def frequency(self) -> TempoControlSignal:
        return self.tempo_control

    @frequency.setter
    def frequency(self, value: TempoControlSignal) -> None:
        self.tempo_control = value

    @property
    def user_state(self) -> CounterpartyReadinessSignal:
        return self.counterparty_readiness

    @user_state.setter
    def user_state(self, value: CounterpartyReadinessSignal) -> None:
        self.counterparty_readiness = value

    @property
    def atmosphere(self) -> GroupContextSignal:
        return self.group_context

    @atmosphere.setter
    def atmosphere(self, value: GroupContextSignal) -> None:
        self.group_context = value

    @property
    def energy(self) -> ResourceLedgerSignal:
        return self.resource_ledger

    @energy.setter
    def energy(self, value: ResourceLedgerSignal) -> None:
        self.resource_ledger = value

    @property
    def social(self) -> SocialBalanceSignal:
        return self.social_balance

    @social.setter
    def social(self, value: SocialBalanceSignal) -> None:
        self.social_balance = value

    @property
    def heart(self) -> PendingEngagementSignal:
        return self.pending_engagement

    @heart.setter
    def heart(self, value: PendingEngagementSignal) -> None:
        self.pending_engagement = value

    @property
    def trauma(self) -> TraumaLoadSignal:
        return self.trauma_load

    @trauma.setter
    def trauma(self, value: TraumaLoadSignal) -> None:
        self.trauma_load = value

    @property
    def mask(self) -> SurfaceMaskSignal:
        return self.surface_mask

    @mask.setter
    def mask(self, value: SurfaceMaskSignal) -> None:
        self.surface_mask = value

    def all_votes(self) -> list[DimensionVote]:
        return [
            self.emotion_baseline,
            self.rapport_trust,
            self.boundary_block,
            self.tempo_control,
            self.counterparty_readiness,
            self.group_context,
            self.resource_ledger,
            self.social_balance,
            self.pending_engagement,
            self.trauma_load,
            self.surface_mask,
        ]

    def has_any_force_refuse(self) -> bool:
        return any(v.force_refuse for v in self.all_votes())

    def has_any_force_trigger(self) -> bool:
        return any(v.force_trigger for v in self.all_votes())

    def combined_probability_factor(self) -> float:
        product = 1.0
        count = 0
        for vote in self.all_votes():
            if not vote.force_refuse and not vote.force_trigger:
                product *= vote.probability_factor
                count += 1
        if count <= 1 or product >= 1.0:
            return product
        dampening = max(0.25, 1.5 / (1.0 + count * 0.3))
        return product ** dampening

    def collect_attitude_tags(self) -> list[str]:
        return [vote.attitude_tag for vote in self.all_votes() if vote.attitude_tag]

    def collect_style_hints(self) -> list[str]:
        return [vote.style_hint for vote in self.all_votes() if vote.style_hint]

    def min_max_tokens_cap(self) -> int:
        caps = [vote.max_tokens_cap for vote in self.all_votes() if vote.max_tokens_cap > 0]
        return min(caps) if caps else 0

    def debug_summary(self) -> dict[str, Any]:
        result = {}
        for vote in self.all_votes():
            result[vote.dimension_name] = {
                "legacy_name": getattr(vote, "legacy_name", ""),
                "prob_factor": round(vote.probability_factor, 3),
                "force_refuse": vote.force_refuse,
                "force_trigger": vote.force_trigger,
                "attitude": vote.attitude_tag,
                "reason": vote.debug_reason,
            }
        return result

    def to_domains_dict(self) -> dict[str, Any]:
        return {
            "emotion_baseline": self.emotion_baseline.to_domain_dict(),
            "rapport_trust": self.rapport_trust.to_domain_dict(),
            "boundary_block": self.boundary_block.to_domain_dict(),
            "tempo_control": self.tempo_control.to_domain_dict(),
            "counterparty_readiness": self.counterparty_readiness.to_domain_dict(),
            "group_context": self.group_context.to_domain_dict(),
            "resource_ledger": self.resource_ledger.to_domain_dict(),
            "social_balance": self.social_balance.to_domain_dict(),
            "pending_engagement": self.pending_engagement.to_domain_dict(),
            "runtime_calibration": {},
            "trauma_load": self.trauma_load.to_domain_dict(),
            "surface_mask": self.surface_mask.to_domain_dict(),
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "emotion": _plain(asdict(self.emotion_baseline)),
            "fondness_trust": _plain(asdict(self.rapport_trust)),
            "dislike": _plain(asdict(self.boundary_block)),
            "frequency": _plain(asdict(self.tempo_control)),
            "user_state": _plain(asdict(self.counterparty_readiness)),
            "atmosphere": _plain(asdict(self.group_context)),
            "energy": _plain(asdict(self.resource_ledger)),
            "social": _plain(asdict(self.social_balance)),
            "heart": _plain(asdict(self.pending_engagement)),
            "trauma": _plain(asdict(self.trauma_load)),
            "mask": _plain(asdict(self.surface_mask)),
        }


LEGACY_SIGNAL_NAME_MAP = {
    "EmotionAxisVote": "EmotionBaselineSignal",
    "FondnessTrustVote": "RapportTrustSignal",
    "DislikeVote": "BoundaryBlockSignal",
    "FrequencyVote": "TempoControlSignal",
    "UserStateVote": "CounterpartyReadinessSignal",
    "GroupAtmosphereVote": "GroupContextSignal",
    "EnergyChainVote": "ResourceLedgerSignal",
    "SocialValueVote": "SocialBalanceSignal",
    "HeartStateVote": "PendingEngagementSignal",
    "CalibrationReport": "RuntimeCalibrationProfile",
    "TraumaVote": "TraumaLoadSignal",
    "SurfaceMaskVote": "SurfaceMaskSignal",
}


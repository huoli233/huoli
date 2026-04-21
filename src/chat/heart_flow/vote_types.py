"""兼容层：旧版 D1-D11 投票类型名称映射到新的语义信号模型。"""

from src.chat.heart_flow.signal_domains import (
    AgentContextRole,
    AffinitySignalLevel,
    BoundaryBlockSignal,
    BoundaryIncidentRecord,
    CounterpartyEngagementState,
    CounterpartyReadinessSignal,
    DecisionSignalBundle,
    EmotionBaselineSignal,
    GroupContextSignal,
    GroupContextTier,
    PendingEngagementPhase,
    PendingEngagementSignal,
    RapportTrustSignal,
    ResourceLedgerSignal,
    ResourceLedgerStage,
    RuntimeCalibrationBand,
    RuntimeCalibrationProfile,
    SocialBalanceSignal,
    SurfaceMaskMode,
    SurfaceMaskSignal,
    TempoControlSignal,
    TrustSignalLevel,
    TraumaLoadBand,
    TraumaLoadSignal,
)

# 旧类型名 -> 新语义主模型
EmotionAxisVote = EmotionBaselineSignal
FondnessTrustVote = RapportTrustSignal
DislikeEntry = BoundaryIncidentRecord
DislikeVote = BoundaryBlockSignal
FrequencyVote = TempoControlSignal
UserEngagementState = CounterpartyEngagementState
UserStateVote = CounterpartyReadinessSignal
AtmosphereTier = GroupContextTier
AIPositioning = AgentContextRole
GroupAtmosphereVote = GroupContextSignal
EnergyStage = ResourceLedgerStage
EnergyChainVote = ResourceLedgerSignal
SocialValueVote = SocialBalanceSignal
HeartWaitPhase = PendingEngagementPhase
HeartStateVote = PendingEngagementSignal
CalibrationGrade = RuntimeCalibrationBand
CalibrationReport = RuntimeCalibrationProfile
TraumaGrade = TraumaLoadBand
TraumaVote = TraumaLoadSignal
MaskBehaviorMode = SurfaceMaskMode
SurfaceMaskVote = SurfaceMaskSignal
FondnessLevel = AffinitySignalLevel
TrustLevel = TrustSignalLevel
VoteCollection = DecisionSignalBundle

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

__all__ = [
    "EmotionBaselineSignal",
    "RapportTrustSignal",
    "BoundaryIncidentRecord",
    "BoundaryBlockSignal",
    "TempoControlSignal",
    "CounterpartyEngagementState",
    "CounterpartyReadinessSignal",
    "GroupContextTier",
    "AgentContextRole",
    "GroupContextSignal",
    "ResourceLedgerStage",
    "ResourceLedgerSignal",
    "SocialBalanceSignal",
    "PendingEngagementPhase",
    "PendingEngagementSignal",
    "RuntimeCalibrationBand",
    "RuntimeCalibrationProfile",
    "TraumaLoadBand",
    "TraumaLoadSignal",
    "SurfaceMaskMode",
    "SurfaceMaskSignal",
    "AffinitySignalLevel",
    "TrustSignalLevel",
    "DecisionSignalBundle",
    "EmotionAxisVote",
    "FondnessTrustVote",
    "DislikeEntry",
    "DislikeVote",
    "FrequencyVote",
    "UserEngagementState",
    "UserStateVote",
    "AtmosphereTier",
    "AIPositioning",
    "GroupAtmosphereVote",
    "EnergyStage",
    "EnergyChainVote",
    "SocialValueVote",
    "HeartWaitPhase",
    "HeartStateVote",
    "CalibrationGrade",
    "CalibrationReport",
    "TraumaGrade",
    "TraumaVote",
    "MaskBehaviorMode",
    "SurfaceMaskVote",
    "FondnessLevel",
    "TrustLevel",
    "VoteCollection",
    "LEGACY_SIGNAL_NAME_MAP",
]

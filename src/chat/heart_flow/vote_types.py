from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from src.chat.heart_flow.dimension_protocol import DimensionVote


# ============================================================
# D1: 情绪三轴投票
# ============================================================
@dataclass
class EmotionAxisVote(DimensionVote):
    """D1 情绪三轴（patience/annoyance/fatigue）的投票结构"""
    dimension_name: str = "emotion_axis"
    patience_snapshot: float = 100.0
    annoyance_snapshot: float = 0.0
    fatigue_snapshot: float = 0.0
    mood_modifier: str = ""
    should_refuse: bool = False
    emotion_proactive_willingness: float = 0.0


# ============================================================
# D2: 好感/信任双核投票
# ============================================================
class FondnessLevel(Enum):
    """好感度七档枚举"""
    EXTREME_DISLIKE = "extreme_dislike"    # 极端反感 < -70
    DISLIKE = "dislike"                    # 讨厌 -70~-40
    DISSATISFIED = "dissatisfied"          # 不满 -40~-10
    NEUTRAL = "neutral"                    # 中立 -10~10
    FRIENDLY = "friendly"                  # 友好 10~40
    FOND = "fond"                          # 喜爱 40~70
    ADORE = "adore"                        # 极度喜爱 > 70


class TrustLevel(Enum):
    """信任度七档枚举"""
    TOTAL_DISTRUST = "total_distrust"      # 彻底反感 < -70
    DISTRUST = "distrust"                  # 不信任 -70~-40
    DOUBTFUL = "doubtful"                  # 有疑虑 -40~-10
    ORDINARY = "ordinary"                  # 普通 -10~10
    FAVORABLE = "favorable"                # 有好感 10~40
    TRUSTED = "trusted"                    # 基本信任 40~70
    FULL_TRUST = "full_trust"              # 完全信任 > 70


@dataclass
class FondnessTrustVote(DimensionVote):
    """D2 好感/信任双核系统的投票结构"""
    dimension_name: str = "fondness_trust"
    # 好感度数值（-100~100）
    fondness_value: float = 0.0
    # 好感度等级标签
    fondness_level: FondnessLevel = FondnessLevel.NEUTRAL
    # 好感概率乘数（从七档映射表查得）
    fondness_factor: float = 0.85
    # 信任度数值（-100~100）
    trust_value: float = 0.0
    # 信任度等级标签
    trust_level: TrustLevel = TrustLevel.ORDINARY
    # 信任概率乘数
    trust_factor: float = 1.0
    # 交叉组合描述（如"喜欢但留心"、"尊重但冷淡"）
    cross_hint: str = ""
    # 主动发言意愿等级（0.0~1.0，由好感度决定）
    proactive_willingness: float = 0.3


# ============================================================
# D2b: 讨厌度投票
# ============================================================
@dataclass
class DislikeEntry:
    """单条讨厌记录"""
    category: str = ""          # 讨厌类别（insult/spam/harmful/etc）
    severity: float = 0.0       # 严重程度（0~10）
    timestamp: float = 0.0      # 记录时间戳
    description: str = ""       # 行为描述
    decay_rate: float = 0.01    # 衰减速率（每天）


@dataclass
class DislikeVote(DimensionVote):
    """D2b 讨厌度注册表的投票结构"""
    dimension_name: str = "dislike_registry"
    # 是否达到屏蔽阈值（True时决策网关直接拒绝）
    is_blocked: bool = False
    dislike_penalty: float = 0.0
    total_severity: float = 0.0
    worst_category: str = ""
    block_threshold: float = 30.0
    proactive_suppression: float = 0.0


# ============================================================
# D3: 频率控制投票
# ============================================================
@dataclass
class FrequencyVote(DimensionVote):
    """D3 频率控制的投票结构"""
    dimension_name: str = "frequency_control"
    # 当前频率调整系数（>0表示倾向回复，<0表示倾向沉默）
    adjust_factor: float = 1.0
    # 动态阈值（根据对话节奏自适应调整）
    dynamic_threshold: float = 0.5
    # 是否处于冷却期（True时建议不回复）
    in_cooldown: bool = False
    # 连续不回复次数
    consecutive_skip_count: int = 0
    # 连续不回复阈值（超过此值应强制回复一次）
    skip_limit: int = 5
    # 距上次回复的秒数
    seconds_since_last_reply: float = 0.0


# ============================================================
# D4: 用户意愿检测投票
# ============================================================
class UserEngagementState(Enum):
    """用户参与度五态枚举"""
    EAGER = "eager"           # 积极互动
    NEUTRAL = "neutral"       # 正常状态
    RELUCTANT = "reluctant"   # 消极抵触
    HOSTILE = "hostile"       # 敌意明显
    ABSENT = "absent"         # 已离场/不活跃


@dataclass
class UserStateVote(DimensionVote):
    """D4 用户意愿检测的投票结构"""
    dimension_name: str = "user_state"
    # 用户当前参与度状态
    engagement: UserEngagementState = UserEngagementState.NEUTRAL
    # 对方是否想和机器人聊天的置信度（0.0~1.0）
    want_to_chat_confidence: float = 0.5
    # 消息频率指标（条/分钟）
    msg_frequency: float = 0.0
    # 平均消息长度
    avg_msg_length: float = 0.0
    # 提问比率（0.0~1.0）
    question_ratio: float = 0.0
    # 是否检测到结构性敌意
    structural_hostility: bool = False


# ============================================================
# D5: 群体氛围投票
# ============================================================
class AtmosphereTier(Enum):
    """群氛围五档枚举"""
    HEATED = "heated"           # 火热（高活跃、多人参与）
    LIVELY = "lively"           # 活跃
    NORMAL = "normal"           # 正常
    QUIET = "quiet"             # 安静
    DEAD = "dead"               # 沉寂


class AIPositioning(Enum):
    """AI在群聊中的定位"""
    CORE = "core"               # 话题核心（多人@或讨论AI）
    PARTICIPANT = "participant"  # 普通参与者
    OBSERVER = "observer"       # 旁观者
    OUTSIDER = "outsider"       # 局外人


@dataclass
class GroupAtmosphereVote(DimensionVote):
    """D5 群体氛围的投票结构"""
    dimension_name: str = "group_atmosphere"
    # 当前氛围等级
    atmosphere_tier: AtmosphereTier = AtmosphereTier.NORMAL
    # AI在群聊中的定位
    ai_position: AIPositioning = AIPositioning.OBSERVER
    # 参与度加成（正值增加回复概率，负值降低）
    engagement_boost: float = 0.0
    # 频道活跃人数
    active_user_count: int = 0
    # 当前讨论话题关键词
    topic_keywords: list[str] = field(default_factory=list)
    # 风格约束（群氛围决定的回复风格偏好）
    style_constraint: str = ""
    # 话题相关度（AI参与当前话题的适合程度，0.0~1.0）
    topic_relevance: float = 0.5


# ============================================================
# D6: 能量链条系统投票
# ============================================================
class EnergyStage(Enum):
    """能量阶段枚举"""
    FULL = "full"               # 满电（>80%）
    ADEQUATE = "adequate"       # 充足（50%~80%）
    LOW = "low"                 # 偏低（20%~50%）
    CRITICAL = "critical"       # 告急（5%~20%）
    DEPLETED = "depleted"       # 透支（<5%）


@dataclass
class EnergyChainVote(DimensionVote):
    """D6 能量链条系统的投票结构"""
    dimension_name: str = "energy_chain"
    # 当前聊天能量百分比（0.0~1.0）
    chat_energy_ratio: float = 1.0
    # 当前思考能量百分比（0.0~1.0）
    thinking_energy_ratio: float = 1.0
    # 能量阶段
    energy_stage: EnergyStage = EnergyStage.FULL
    # 回复消耗预估（预计这次回复需要消耗多少能量，0.0~1.0）
    estimated_cost: float = 0.05
    # 是否处于夜间低功耗模式
    night_mode: bool = False
    # 能量恢复速率（当前阻尼系数，0.0~1.0）
    recovery_damping: float = 1.0
    # 聊天值天花板（群聊级别的能量池上限）
    chat_ceiling: float = 100.0
    # 当前聊天值
    chat_current: float = 100.0


# ============================================================
# D7: 社交值投票
# ============================================================
@dataclass
class SocialValueVote(DimensionVote):
    """D7 社交值的投票结构"""
    dimension_name: str = "social_value"
    # 社交值数值（-100~100）
    social_score: float = 0.0
    social_tag: str = ""
    in_alert: bool = False
    positive_interactions: int = 0
    negative_interactions: int = 0
    proactive_boost: float = 0.0


# ============================================================
# D8: 心流等待状态投票
# ============================================================
class HeartWaitPhase(Enum):
    """心流等待三段枚举"""
    IDLE = "idle"               # 空闲（无等待）
    WAITING = "waiting"         # 等待中（期望对方回复）
    TIMEOUT = "timeout"         # 等待超时（对方未回复）


@dataclass
class HeartStateVote(DimensionVote):
    """D8 心流等待状态的投票结构"""
    dimension_name: str = "heart_state"
    wait_phase: HeartWaitPhase = HeartWaitPhase.IDLE
    should_initiate: bool = False
    waiting_seconds: float = 0.0
    timeout_threshold: float = 300.0
    last_interaction_channel: str = ""
    heartbeat_proactive_willingness: float = 0.0


# ============================================================
# D9: 系统校准（不产生投票，但需要结构定义用于内部通讯）
# ============================================================
class CalibrationGrade(Enum):
    """校准等级"""
    SKIP = "skip"               # 离线<5分钟，不校准
    LIGHT = "light"             # 离线5分钟~2小时，轻度校准
    MODERATE = "moderate"       # 离线2~12小时，中度校准
    DEEP = "deep"               # 离线12~48小时，深度校准
    FULL_RESET = "full_reset"   # 离线>48小时，完全重置


@dataclass
class CalibrationReport:
    """D9 校准结果报告（不是投票，而是发给其他维度的校准指令）"""
    grade: CalibrationGrade = CalibrationGrade.SKIP
    offline_seconds: float = 0.0
    # 情绪衰减系数（离线越久衰减越多）
    emotion_decay_factor: float = 0.0
    # 好感衰减系数
    fondness_decay_factor: float = 0.0
    # 能量恢复系数（离线期间被动恢复）
    energy_recovery_factor: float = 0.0
    # 创伤衰减系数（创伤随时间缓慢愈合）
    trauma_heal_factor: float = 0.0
    # 伪装疲劳恢复系数
    mask_fatigue_recovery: float = 0.0
    # 频率控制重置标志
    frequency_should_reset: bool = False


# ============================================================
# D10: 创伤系统投票
# ============================================================
class TraumaGrade(Enum):
    """创伤等级四档"""
    NONE = "none"               # 无创伤（0~2）
    MILD = "mild"               # 轻度（2~5）
    SEVERE = "severe"           # 重度（5~8）
    EXTREME = "extreme"         # 极端（8~10）


@dataclass
class TraumaVote(DimensionVote):
    """D10 创伤系统的投票结构"""
    dimension_name: str = "trauma"
    # 当前创伤指数（0~10）
    trauma_index: float = 0.0
    # 创伤等级
    trauma_grade: TraumaGrade = TraumaGrade.NONE
    # 认知损伤度（创伤导致的认知能力下降，0.0~1.0）
    cognitive_impairment: float = 0.0
    # 闪回风险（当前语境是否可能触发创伤闪回，0.0~1.0）
    flashback_risk: float = 0.0
    # 闪回触发的碎片标签列表
    triggered_fragment_labels: list[str] = field(default_factory=list)
    # 潜意识干扰描述（注入LLM的内心活动）
    perception_labels: list[str] = field(default_factory=list)
    # 内心混乱度（0~100，越高越无法正常交流）
    inner_chaos: float = 0.0


# ============================================================
# D11: 伪装/强颜欢笑投票
# ============================================================
class MaskBehaviorMode(Enum):
    """伪装行为模式"""
    NATURAL = "natural"         # 自然状态（不需要伪装）
    MASKING = "masking"         # 正在伪装（内外不一致）
    CRACKING = "cracking"       # 快要装不住了
    BROKEN = "broken"           # 伪装已崩塌


@dataclass
class SurfaceMaskVote(DimensionVote):
    """D11 伪装/强颜欢笑的投票结构"""
    dimension_name: str = "surface_mask"
    # 伪装强度（0~100，越高伪装越成功）
    mask_strength: float = 0.0
    # 当前行为模式
    behavior_mode: MaskBehaviorMode = MaskBehaviorMode.NATURAL
    # 伪装是否正在破裂
    is_breaking: bool = False
    # 回复质量上限（伪装破裂时降低回复质量，0.0~1.0）
    reply_quality_cap: float = 1.0
    # 伪装疲劳度（0~100，越高越容易崩塌）
    mask_fatigue: float = 0.0
    # 被识破风险（0.0~1.0）
    detection_risk: float = 0.0
    # 伪装破裂时注入LLM的内心描述
    inner_conflict_hint: str = ""
    # 九宫格矩阵坐标（混乱度×伪装力度）
    matrix_row: int = 0       # 混乱度等级（0=低 1=中 2=高）
    matrix_col: int = 0       # 伪装力度等级（0=弱 1=中 2=强）


# ============================================================
# 投票集合：决策网关使用的完整投票包
# ============================================================
@dataclass
class VoteCollection:
    """
    所有维度投票的聚合容器。
    决策网关从 DimensionDispatcher.collect_votes() 获取此对象，
    然后按优先级规则组合各维度的建议。
    """
    emotion: EmotionAxisVote = field(default_factory=EmotionAxisVote)
    fondness_trust: FondnessTrustVote = field(default_factory=FondnessTrustVote)
    dislike: DislikeVote = field(default_factory=DislikeVote)
    frequency: FrequencyVote = field(default_factory=FrequencyVote)
    user_state: UserStateVote = field(default_factory=UserStateVote)
    atmosphere: GroupAtmosphereVote = field(default_factory=GroupAtmosphereVote)
    energy: EnergyChainVote = field(default_factory=EnergyChainVote)
    social: SocialValueVote = field(default_factory=SocialValueVote)
    heart: HeartStateVote = field(default_factory=HeartStateVote)
    trauma: TraumaVote = field(default_factory=TraumaVote)
    mask: SurfaceMaskVote = field(default_factory=SurfaceMaskVote)

    def all_votes(self) -> list[DimensionVote]:
        """返回所有投票的列表（用于遍历）"""
        return [
            self.emotion,
            self.fondness_trust,
            self.dislike,
            self.frequency,
            self.user_state,
            self.atmosphere,
            self.energy,
            self.social,
            self.heart,
            self.trauma,
            self.mask,
        ]

    def has_any_force_refuse(self) -> bool:
        """是否有任何维度建议强制拒绝"""
        return any(v.force_refuse for v in self.all_votes())

    def has_any_force_trigger(self) -> bool:
        """是否有任何维度建议强制触发"""
        return any(v.force_trigger for v in self.all_votes())

    def combined_probability_factor(self) -> float:
        """
        计算所有维度概率乘数的组合因子（排除force_refuse/force_trigger的维度）。
        使用维度数量归一化的调和指数，防止多维同时偏低时的指数坍缩。
        """
        product = 1.0
        n = 0
        for v in self.all_votes():
            if not v.force_refuse and not v.force_trigger:
                product *= v.probability_factor
                n += 1
        if n <= 1 or product >= 1.0:
            return product
        dampening = max(0.25, 1.5 / (1.0 + n * 0.3))
        return product ** dampening

    def collect_attitude_tags(self) -> list[str]:
        """收集所有非空的态度标签"""
        return [v.attitude_tag for v in self.all_votes() if v.attitude_tag]

    def collect_style_hints(self) -> list[str]:
        """收集所有非空的风格修饰"""
        return [v.style_hint for v in self.all_votes() if v.style_hint]

    def min_max_tokens_cap(self) -> int:
        """取所有维度中最严格的token上限（0表示无限制）"""
        caps = [v.max_tokens_cap for v in self.all_votes() if v.max_tokens_cap > 0]
        return min(caps) if caps else 0

    def debug_summary(self) -> dict:
        """返回所有投票的调试摘要"""
        result = {}
        for v in self.all_votes():
            result[v.dimension_name] = {
                "prob_factor": round(v.probability_factor, 3),
                "force_refuse": v.force_refuse,
                "force_trigger": v.force_trigger,
                "attitude": v.attitude_tag,
                "reason": v.debug_reason,
            }
        return result

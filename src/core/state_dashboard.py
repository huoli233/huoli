import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("状态面板")

_dashboard_singleton = None


# ═══════════════════════════════════════════════════
#  第一层：核心状态枚举（用户可见的"状态栏"）
# ═══════════════════════════════════════════════════


class VitalityLevel(Enum):
    """精力层级——状态栏主指标"""

    FULL = "full"
    GOOD = "good"
    NORMAL = "normal"
    TIRED = "tired"
    EXHAUSTED = "exhausted"
    CRASHED = "crashed"

    def bar_color(self) -> str:
        return {
            "full": "#22c55e",
            "good": "#82c91e",
            "normal": "#ffd93d",
            "tired": "#ff9f43",
            "exhausted": "#ff6b35",
            "crashed": "#eb3b5a",
        }.get(self.value, "#888")

    def label(self) -> str:
        return {
            "full": "精力充沛",
            "good": "状态良好",
            "normal": "一般",
            "tired": "有点累了",
            "exhausted": "很疲惫",
            "crashed": "濒临崩溃",
        }.get(self.value, "")

    def icon(self) -> str:
        return {
            "full": "🔋",
            "good": "✨",
            "normal": "💡",
            "tired": "😪",
            "exhausted": "😵",
            "crashed": "💀",
        }.get(self.value, "?")


class SocialPosture(Enum):
    """社交姿态——当前对群的态度"""

    WARM_ENGAGED = "warm_engaged"
    CASUAL_OPEN = "casual_open"
    NEUTRAL_OBSERVING = "neutral_observing"
    COOL_DISTANT = "cool_distant"
    WITHDRAWN = "withdrawn"
    AVOIDANT = "avoidant"

    def label(self) -> str:
        return {
            "warm_engaged": "热情参与",
            "casual_open": "轻松开放",
            "neutral_observing": "中立观察",
            "cool_distant": "冷淡疏远",
            "withdrawn": "暂时退出",
            "avoidant": "回避躲藏",
        }.get(self.value, "")

    def icon(self) -> str:
        return {
            "warm_engaged": "🤗",
            "casual_open": "👋",
            "neutral_observing": "👀",
            "cool_distant": "❄",
            "withdrawn": "🚪",
            "avoidant": "🙅",
        }.get(self.value, "?")


class MentalState(Enum):
    """心理状态——内心感受"""

    CHEERFUL = "cheerful"
    CALM = "calm"
    CONTENT = "content"
    THOUGHTFUL = "thoughtful"
    BORED = "bored"
    RESTLESS = "restless"
    ANNOYED = "annoyed"
    DROWSY = "drowsy"
    IRRITATED = "irritated"
    OVERWHELMED = "overwhelmed"

    def label(self) -> str:
        return {
            "cheerful": "开心愉悦",
            "calm": "平静安详",
            "content": "满足自在",
            "thoughtful": "若有所思",
            "bored": "有些无聊",
            "restless": "躁动不安",
            "annoyed": "有点烦躁",
            "drowsy": "困倦昏沉",
            "irritated": "易怒敏感",
            "overwhelmed": "不堪重负",
        }.get(self.value, "")

    def icon(self) -> str:
        return {
            "cheerful": "😊",
            "calm": "😌",
            "content": "😌",
            "thoughtful": "🤔",
            "bored": "😑",
            "restless": "😤",
            "annoyed": "😒",
            "drowsy": "😴",
            "irritated": "😠",
            "overwhelmed": "😵‍♂️",
        }.get(self.value, "?")


class AttentionMode(Enum):
    """注意力模式"""

    FOCUS_LOCKED = "focus_locked"
    SCANNING = "scanning"
    GLANCING = "glancing"
    PEEKING = "peeking"
    ABSENT = "absent"
    SLEEPING = "sleeping"

    def label(self) -> str:
        return {
            "focus_locked": "专注锁定",
            "scanning": "扫描模式",
            "glancing": "随意扫视",
            "peeking": "偶尔窥屏",
            "absent": "不在状态",
            "sleeping": "休眠中",
        }.get(self.value, "")

    def icon(self) -> str:
        return {
            "focus_locked": "🎯",
            "scanning": "📡",
            "glancing": "👁",
            "peeking": "🫣",
            "absent": "⬛",
            "sleeping": "💤",
        }.get(self.value, "?")


class GroupEngagement(Enum):
    """群组参与度"""

    ACTIVE_CORE = "active_core"
    PARTICIPANT = "participant"
    SPECTATOR = "spectator"
    LURKER = "lurker"
    GHOST = "ghost"

    def label(self) -> str:
        return {
            "active_core": "活跃核心",
            "participant": "参与者",
            "spectator": "围观者",
            "lurker": "潜水者",
            "ghost": "隐形人",
        }.get(self.value, "")

    def icon(self) -> str:
        return {
            "active_core": "⭐",
            "participant": "💬",
            "spectator": "👥",
            "lurker": "👻",
            "ghost": "👻‍️",
        }.get(self.value, "?")


# ═══════════════════════════════════════════════════
#  第二层：触发层枚举（内部决策用）
# ═══════════════════════════════════════════════════


class TriggerTier(Enum):
    HARD_BLOCK_GATE = "hard_block_gate"
    CONTEXT_GATE = "context_gate"
    EMERGENCE_DECISION = "emergence_decision"


class ActionUrgency(Enum):
    MUST_REPLY_NOW = "must_reply_now"
    SHOULD_REPLY_SOON = "should_reply_soon"
    MAY_REPLY_LATER = "may_reply_later"
    CAN_SKIP = "can_skip"
    MUST_NOT_REPLY = "must_not_reply"

    def label(self) -> str:
        return {
            "must_reply_now": "立即回复",
            "should_reply_soon": "尽快回复",
            "may_reply_later": "可稍后回",
            "can_skip": "跳过",
            "must_not_reply": "不回复",
        }.get(self.value, "")


# ═══════════════════════════════════════════════════
#  第三层：数据结构
# ═══════════════════════════════════════════════════


@dataclass
class VitalityBar:
    """精力条——状态栏最显眼的指标"""

    level: VitalityLevel = VitalityLevel.NORMAL
    energy_ratio: float = 1.0
    fatigue_accumulated: float = 0.0
    recovery_rate: float = 1.0
    debt_modifier: float = 1.0
    effective_recovery: float = 1.0
    trend: str = ""  # "recovering" | "stable" | "declining"
    time_to_full: float = 0.0

    def display_percent(self) -> int:
        return int(self.energy_ratio * 100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.label(),
            "icon": self.level.icon(),
            "color": self.level.bar_color(),
            "percent": self.display_percent(),
            "trend": self.trend,
            "effective_recovery": round(self.effective_recovery, 2),
            "time_to_full_min": round(self.time_to_full / 60.0, 1),
        }


@dataclass
class SocialBar:
    """社交条"""

    posture: SocialPosture = SocialPosture.NEUTRAL_OBSERVING
    social_willingness: float = 0.5
    group_engagement: GroupEngagement = GroupEngagement.SPECTATOR
    openness: float = 0.5
    interrupt_tolerance: float = 0.5
    avoidance_tendency: float = 0.0
    loneliness: float = 0.0
    social_hunger: float = 0.5
    # 真实社交值（来自social_value_core）
    core_social_value: float = 0.0
    core_trust_value: float = 0.0
    core_affection: float = 0.0
    core_annoyance: float = 0.0
    interaction_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "posture": self.posture.label(),
            "icon": self.posture.icon(),
            "willingness": round(self.social_willingness, 2),
            "engagement": self.group_engagement.label(),
            "openness": round(self.openness, 2),
            "tolerance": round(self.interrupt_tolerance, 2),
            "loneliness": round(self.loneliness, 2),
            "social_hunger": round(self.social_hunger, 2),
            "social_value": round(self.core_social_value, 1),
            "trust": round(self.core_trust_value, 1),
            "affection": round(self.core_affection, 1),
            "annoyance": round(self.core_annoyance, 1),
            "interactions": self.interaction_count,
        }


@dataclass
class MoodBar:
    """心情条"""

    mental: MentalState = MentalState.CONTENT
    mood_value: float = 0.5
    curiosity: float = 0.3
    boredom: float = 0.0
    loafing: float = 0.0
    annoyance: float = 0.0
    proactive_drive: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mental": self.mental.label(),
            "icon": self.mental.icon(),
            "mood": round(self.mood_value, 2),
            "curiosity": round(self.curiosity, 2),
            "boredom": round(self.boredom, 2),
            "loafing": round(self.loafing, 2),
            "annoyance": round(self.annoyance, 2),
            "drive": round(self.proactive_drive, 2),
        }


@dataclass
class AttentionBar:
    """注意力条"""

    mode: AttentionMode = AttentionMode.SCANNING
    visibility_threshold: float = 0.30
    process_ratio: float = 0.50
    peek_desire: float = 0.3
    withdrawal_depth: float = 0.0
    consecutive_empty_peeks: int = 0
    last_look_ago_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.label(),
            "icon": self.mode.icon(),
            "threshold": round(self.visibility_threshold, 2),
            "process_ratio": f"{int(self.process_ratio * 100)}%",
            "peek_desire": round(self.peek_desire, 2),
            "withdrawal": round(self.withdrawal_depth, 2),
            "empty_peeks": self.consecutive_empty_peeks,
            "last_look_sec": round(self.last_look_ago_sec, 0),
        }


@dataclass
class NightBar:
    """昼夜条"""

    phase_name: str = ""
    phase_icon: str = "☀️"
    is_night: bool = False
    pressure_total: float = 0.0
    is_burnthrough: bool = False
    expression_style: str = "normal"
    next_phase_in_min: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase_name,
            "icon": self.phase_icon,
            "is_night": self.is_night,
            "pressure": round(self.pressure_total, 1),
            "burnthrough": self.is_burnthrough,
            "style": self.expression_style,
        }


@dataclass
class SafetyBar:
    """安全条"""

    level_label: str = "安全"
    level_icon: str = "🛡️"
    score: float = 0.0
    dominant_threat: str = ""
    blocked: bool = False
    bar_penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level_label,
            "icon": self.level_icon,
            "score": round(self.score, 3),
            "threat": self.dominant_threat[:20],
            "blocked": self.blocked,
            "penalty": round(self.bar_penalty, 3),
        }


@dataclass
class MemoryBar:
    """记忆条"""

    total_entries: int = 0
    utilization: float = 0.0
    fresh_count: int = 0
    stale_count: int = 0
    window_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total_entries,
            "utilization": f"{int(self.utilization * 100)}%",
            "fresh": self.fresh_count,
            "stale": self.stale_count,
            "window": self.window_size,
        }


@dataclass
class ActionVerdict:
    """最终行动裁定——这是仪表盘最重要的输出"""

    urgency: ActionUrgency = ActionUrgency.MAY_REPLY_LATER
    should_process_message: bool = True
    should_reply: bool = False
    primary_reason: str = ""
    secondary_reasons: List[str] = field(default_factory=list)
    recommended_tone: str = ""
    suggested_action: str = ""
    trigger_path: str = ""
    legacy_trigger_path: str = ""
    confidence: float = 0.5
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "urgency": self.urgency.label(),
            "process": self.should_process_message,
            "reply": self.should_reply,
            "reason": self.primary_reason[:60],
            "details": self.secondary_reasons[:3],
            "tone": self.recommended_tone,
            "action": self.suggested_action[:80],
            "path": self.trigger_path,
            "legacy_path": self.legacy_trigger_path,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class FullDashboardSnapshot:
    """完整仪表盘快照——一次性聚合所有状态"""

    channel_id: str = ""
    timestamp: float = field(default_factory=time.time)
    vitality: VitalityBar = field(default_factory=VitalityBar)
    social: SocialBar = field(default_factory=SocialBar)
    mood: MoodBar = field(default_factory=MoodBar)
    attention: AttentionBar = field(default_factory=AttentionBar)
    night: NightBar = field(default_factory=NightBar)
    safety: SafetyBar = field(default_factory=SafetyBar)
    memory: MemoryBar = field(default_factory=MemoryBar)
    verdict: ActionVerdict = field(default_factory=ActionVerdict)
    raw_engine_outputs: Dict[str, Any] = field(default_factory=dict)
    change_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_display_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel_id[:16],
            "time": time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "vitality": self.vitality.to_dict(),
            "social": self.social.to_dict(),
            "mood": self.mood.to_dict(),
            "attention": self.attention.to_dict(),
            "night": self.night.to_dict(),
            "safety": self.safety.to_dict(),
            "memory": self.memory.to_dict(),
            "verdict": self.verdict.to_dict(),
        }

    def to_status_line(self) -> str:
        v = self.vitality
        m = self.mood.mental
        a = self.attention.mode
        s = self.social.posture
        g = self.social.group_engagement
        n = self.night.phase_icon
        saf = self.safety.level_label
        # 社交值摘要：有真实数据时显示数值，否则仅显示姿态
        _sv_str = ""
        if abs(self.social.core_social_value) > 0.1:
            _sv_val = int(self.social.core_social_value)
            _sv_prefix = "+" if _sv_val > 0 else ""
            _sv_str = f"(社交{_sv_prefix}{_sv_val})"
        return f"{
            v.level.icon()}{
            v.level.label()} | {
            m.icon()}{
                m.label()} | {
                    a.icon()}{
                        a.label()} | {
                            s.icon()}{
                                s.label()}{_sv_str} | {
                                    g.icon()}{
                                        g.label()} | {n}{
                                            self.night.phase_name} | {saf}"


# ═══════════════════════════════════════════════════
#  核心引擎：统一状态仪表盘
# ═══════════════════════════════════════════════════


class StateDashboardEngine:
    """统一状态仪表盘引擎

    解决的问题：
    1. 状态散乱 —— 40+个_cached变量 → 7个标准化Bar
    2. 触发混乱 —— 没有清晰层级 → 三层过滤架构
    3. 展示缺失 —— 不知道该显示什么 → 标准化to_dict/to_status_line
    4. 行为不明 —— 状态→动作映射不清晰 → ActionVerdict统一输出

    三层触发架构：
      L1 快速过滤（<1ms）：硬阻断条件（崩溃/深睡/安全封锁/完全退出）
         → 直接返回 MUST_NOT_REPLY，不做任何复杂计算

      L2 中度评估（<10ms）：状态组合预判
         → 基于Vitality+Mood+Attention+Safety的查表
         → 输出 MAY_REPLY_LATER / CAN_SKIP / SHOULD_REPLY_SOON

      L3 深度决策（<50ms）：ACFN全量计算
         → 仅在L2通过时才执行
         → 输出最终 MUST_REPLY_NOW + 具体策略
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._snapshot_cache: Optional[FullDashboardSnapshot] = None
        self._cache_ttl_sec = 3.0
        self._last_build_ts = 0.0
        self._change_history: List[Dict[str, Any]] = []
        self._max_history = 20
        self._prev_verdict_urgency: Optional[ActionUrgency] = None
        # L1快速阻断缓存
        self._l1_block_reason: str = ""
        self._l1_blocked_until: float = 0.0

    @staticmethod
    def _metric_or_default(value: Any, default: float) -> float:
        """仅把真正缺失值回退为默认值，保留合法的 0 和低值。"""
        if value is None or value == "":
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _normalize_attention_mode(raw_mode: Any) -> str:
        value = str(raw_mode or "").strip().lower()
        return {
            "fully_engaged": "focus_locked",
            "全神贯注": "focus_locked",
            "focus_locked": "focus_locked",
            "selective_attention": "scanning",
            "选择性关注": "scanning",
            "scanning": "scanning",
            "casual_glance": "glancing",
            "随意扫视": "glancing",
            "glancing": "glancing",
            "peeking": "peeking",
            "偶尔窥屏": "peeking",
            "withdrawn": "absent",
            "暂时退出": "absent",
            "absent": "absent",
            "deep_withdrawal": "sleeping",
            "deep_withdrawn": "sleeping",
            "深度退出": "sleeping",
            "sleeping": "sleeping",
        }.get(value, value or "scanning")

    # ═════════════════ 主入口：构建完整仪表盘 ═════════════════

    def build_dashboard(
        self,
        *,
        force: bool = False,
        raw_sources: Optional[Dict[str, Any]] = None,
    ) -> FullDashboardSnapshot:
        now = time.time()
        if (
            not force
            and (now - self._last_build_ts) < self._cache_ttl_sec
            and self._snapshot_cache is not None
        ):
            return self._snapshot_cache
        self._last_build_ts = now
        snap = FullDashboardSnapshot(
            channel_id=self._channel_id, timestamp=now
        )
        sources = raw_sources or {}
        self._build_vitality_bar(snap, sources)
        self._build_social_bar(snap, sources)
        self._build_mood_bar(snap, sources)
        self._build_attention_bar(snap, sources)
        self._build_night_bar(snap, sources)
        self._build_safety_bar(snap, sources)
        self._build_memory_bar(snap, sources)
        self._compute_verdict(snap)
        snap.raw_engine_outputs = dict(sources)
        self._detect_changes(snap)
        self._snapshot_cache = snap
        return snap

    # ═════════════════ L1：快速过滤（<1ms）═════════════════

    def l1_fast_check(self, now: float) -> Optional[str]:
        """第一层：硬阻断条件检查
        返回None表示通过，返回字符串表示阻断原因"""
        if now < self._l1_blocked_until:
            return f"L1-时间冷却中(剩{self._l1_blocked_until - now:.0f}s)"
        vitality = self._get_quick_vitality(now)
        if vitality <= 0.05:
            self._l1_blocked_until = now + 120.0
            return "L1-精力耗尽(≤5%)，强制休息120s"
        night_phase = str(self._get_quick_night_phase())
        if night_phase in ("deep_sleep", "deep_valley"):
            self._l1_blocked_until = now + 300.0
            return "L1-深度睡眠中，不处理消息"
        safety = self._get_quick_safety()
        if safety >= 0.85:
            return "L1-安全等级过高(≥0.85)，阻断回复"
        attention_mode = self._get_quick_attention_mode()
        if attention_mode == "sleeping":
            self._l1_blocked_until = now + 60.0
            return "L1-注意力处于休眠态"
        if attention_mode == "absent":
            peek = self._get_quick_peek_desire()
            if peek < 0.08 and now - self._get_quick_last_look() > 600:
                return "L1-已长时间离线且无窥屏欲望"
        return None

    # ═════════════════ L2：中度评估（查表）═════════════════

    def l2_contextual_eval(
        self,
        dashboard: FullDashboardSnapshot,
        *,
        is_at_bot: bool = False,
        is_quote: bool = False,
    ) -> ActionUrgency:
        """第二层：基于标准化Bar的状态组合查表"""
        v = dashboard.vitality
        m = dashboard.mood
        a = dashboard.attention
        s = dashboard.social
        saf = dashboard.safety
        score = 0.0
        if is_at_bot:
            score += 0.35
        if is_quote:
            score += 0.25
        if v.level in (VitalityLevel.FULL, VitalityLevel.GOOD):
            score += 0.15
        elif v.level == VitalityLevel.NORMAL:
            score += 0.08
        elif v.level == VitalityLevel.TIRED:
            score -= 0.05
        elif v.level in (VitalityLevel.EXHAUSTED, VitalityLevel.CRASHED):
            score -= 0.25
        if m.mental in (
            MentalState.CHEERFUL,
            MentalState.CALM,
            MentalState.CONTENT,
        ):
            score += 0.10
        elif m.mental == MentalState.BORED:
            score += 0.05
        elif m.mental in (
            MentalState.ANNOYED,
            MentalState.IRRITATED,
            MentalState.OVERWHELMED,
        ):
            score -= 0.15
        if a.mode == AttentionMode.FOCUS_LOCKED:
            score += 0.20
        elif a.mode == AttentionMode.SCANNING:
            score += 0.10
        elif a.mode == AttentionMode.GLANCING:
            score += 0.02
        elif a.mode == AttentionMode.PEEKING:
            score -= 0.05
        elif a.mode == AttentionMode.ABSENT:
            score -= 0.20
        if s.posture in (
            SocialPosture.WARM_ENGAGED,
            SocialPosture.CASUAL_OPEN,
        ):
            score += 0.12
        elif s.posture == SocialPosture.NEUTRAL_OBSERVING:
            score += 0.04
        elif s.posture == SocialPosture.WITHDRAWN:
            score -= 0.18
        elif s.posture == SocialPosture.AVOIDANT:
            score -= 0.28
        if saf.score < 0.15:
            score -= 0.02
        elif saf.score < 0.35:
            score -= 0.08
        else:
            score -= 0.15
        if n := dashboard.night:
            if n.is_night:
                if not n.is_burnthrough:
                    score -= 0.12
                else:
                    score += 0.05
        if score >= 0.55:
            return ActionUrgency.SHOULD_REPLY_SOON
        if score >= 0.30:
            return ActionUrgency.MAY_REPLY_LATER
        if score >= 0.10:
            return ActionUrgency.CAN_SKIP
        return ActionUrgency.MUST_NOT_REPLY

    # ═════════════════ 各Bar构建方法 ═════════════════

    def _build_vitality_bar(
        self, snap: FullDashboardSnapshot, src: Dict[str, Any]
    ) -> None:
        e = float(src.get("energy_ratio", 1.0))
        fat = self._metric_or_default(src.get("fatigue"), 0.0)
        rec = self._metric_or_default(src.get("recovery_rate"), 1.0)
        debt = self._metric_or_default(src.get("debt_modifier"), 1.0)
        eff_rec = rec / max(0.3, debt)
        e_clamped = max(0.02, min(1.0, e))
        if e_clamped > 0.85:
            level = VitalityLevel.FULL
        elif e_clamped > 0.65:
            level = VitalityLevel.GOOD
        elif e_clamped > 0.35:
            level = VitalityLevel.NORMAL
        elif e_clamped > 0.18:
            level = VitalityLevel.TIRED
        elif e_clamped > 0.05:
            level = VitalityLevel.EXHAUSTED
        else:
            level = VitalityLevel.CRASHED
        trend = (
            "declining"
            if fat > 0.6
            else ("stable" if abs(e - 0.5) < 0.1 else "recovering")
        )
        deficit = max(0, 1.0 - e_clamped)
        time_full = deficit / max(0.01, eff_rec) * 3600.0
        snap.vitality = VitalityBar(
            level=level,
            energy_ratio=e_clamped,
            fatigue_accumulated=fat,
            recovery_rate=rec,
            debt_modifier=debt,
            effective_recovery=eff_rec,
            trend=trend,
            time_to_full=time_full,
        )

    def _build_social_bar(
        self, snap: FullDashboardSnapshot, src: Dict[str, Any]
    ) -> None:
        sw = self._metric_or_default(src.get("social_willingness"), 0.5)
        open_val = self._metric_or_default(src.get("openness"), 0.5)
        tol = self._metric_or_default(src.get("interrupt_tolerance"), 0.5)
        avoid = self._metric_or_default(src.get("avoidance"), 0.0)
        lonely = self._metric_or_default(src.get("loneliness"), 0.0)
        hunger = self._metric_or_default(src.get("social_hunger"), 0.5)
        eng_raw = self._metric_or_default(src.get("group_engagement_score"), 0.5)
        # 真实社交值（来自social_value_core或relation_result回填）
        core_sv = float(src.get("core_social_value", 0.0) or 0.0)
        core_trust = float(src.get("core_trust_value", 0.0) or 0.0)
        core_affection = float(src.get("core_affection", 0.0) or 0.0)
        core_annoyance = float(src.get("core_annoyance", 0.0) or 0.0)
        core_interactions = int(src.get("core_interaction_count", 0) or 0)
        if eng_raw > 0.75:
            eng = GroupEngagement.ACTIVE_CORE
        elif eng_raw > 0.45:
            eng = GroupEngagement.PARTICIPANT
        elif eng_raw > 0.2:
            eng = GroupEngagement.SPECTATOR
        elif eng_raw > 0.05:
            eng = GroupEngagement.LURKER
        else:
            eng = GroupEngagement.GHOST
        post = SocialPosture.NEUTRAL_OBSERVING
        # 姿态判定：融合存在感引擎 + 真实社交值双重信号
        if sw > 0.7 and avoid < 0.2 and core_sv > 10:
            post = SocialPosture.WARM_ENGAGED
        elif sw > 0.5 and avoid < 0.3 and core_sv > 0:
            post = SocialPosture.CASUAL_OPEN
        elif (sw < 0.25 or avoid > 0.5) and core_sv < -10:
            post = SocialPosture.WITHDRAWN
        if avoid > 0.65 or core_annoyance > 30:
            post = SocialPosture.AVOIDANT
        if core_sv > 40 and core_affection > 20:
            post = SocialPosture.WARM_ENGAGED
        snap.social = SocialBar(
            posture=post,
            social_willingness=sw,
            group_engagement=eng,
            openness=open_val,
            interrupt_tolerance=tol,
            avoidance_tendency=avoid,
            loneliness=lonely,
            social_hunger=hunger,
            core_social_value=core_sv,
            core_trust_value=core_trust,
            core_affection=core_affection,
            core_annoyance=core_annoyance,
            interaction_count=core_interactions,
        )

    def _build_mood_bar(
        self, snap: FullDashboardSnapshot, src: Dict[str, Any]
    ) -> None:
        mood_v = self._metric_or_default(src.get("mood"), 0.5)
        curio = self._metric_or_default(src.get("curiosity"), 0.3)
        bored = self._metric_or_default(src.get("boredom"), 0.0)
        loaf = self._metric_or_default(src.get("loafing"), 0.0)
        annoy = self._metric_or_default(src.get("annoyance_value", src.get("annoyance")), 0.0)
        drive = self._metric_or_default(src.get("proactive_drive"), 0.3)
        mental = MentalState.CONTENT
        if mood_v > 0.72:
            mental = MentalState.CHEERFUL
        elif mood_v > 0.58:
            mental = MentalState.CALM
        elif mood_v > 0.42:
            mental = MentalState.CONTENT
        elif mood_v > 0.28:
            mental = MentalState.THOUGHTFUL
        elif bored > 0.45:
            mental = MentalState.BORED
        elif loaf > 0.4:
            mental = MentalState.RESTLESS
        elif annoy > 0.35:
            mental = MentalState.ANNOYED
        elif mood_v < 0.22:
            mental = MentalState.DROWSY
        elif annoy > 0.55:
            mental = MentalState.IRRITATED
        elif mood_v < 0.12 or (annoy > 0.6 and bored > 0.5):
            mental = MentalState.OVERWHELMED
        snap.mood = MoodBar(
            mental=mental,
            mood_value=mood_v,
            curiosity=curio,
            boredom=bored,
            loafing=loaf,
            annoyance=annoy,
            proactive_drive=drive,
        )

    def _build_attention_bar(
        self, snap: FullDashboardSnapshot, src: Dict[str, Any]
    ) -> None:
        mode_str = self._normalize_attention_mode(
            src.get("attention_mode", src.get("state", ""))
        )
        mode_map = {
            "focus_locked": AttentionMode.FOCUS_LOCKED,
            "scanning": AttentionMode.SCANNING,
            "glancing": AttentionMode.GLANCING,
            "peeking": AttentionMode.PEEKING,
            "absent": AttentionMode.ABSENT,
            "sleeping": AttentionMode.SLEEPING,
        }
        mode = mode_map.get(mode_str, AttentionMode.SCANNING)
        thresh = self._metric_or_default(src.get("visibility_threshold"), 0.30)
        ratio = self._metric_or_default(src.get("process_ratio"), 0.50)
        peek = self._metric_or_default(src.get("peek_desire"), 0.3)
        withd = self._metric_or_default(src.get("withdrawal_depth"), 0.0)
        empty_peeks = int(src.get("consecutive_empty_peeks", src.get("empty_peeks", 0)) or 0)
        last_look = self._metric_or_default(
            src.get("last_look_ago", src.get("since_last_look_sec")),
            999.0,
        )
        snap.attention = AttentionBar(
            mode=mode,
            visibility_threshold=thresh,
            process_ratio=ratio,
            peek_desire=peek,
            withdrawal_depth=withd,
            consecutive_empty_peeks=empty_peeks,
            last_look_ago_sec=last_look,
        )

    def _build_night_bar(
        self, snap: FullDashboardSnapshot, src: Dict[str, Any]
    ) -> None:
        phase = str(src.get("night_phase", "") or "").lower()
        phase = {
            "burned_out": "burnthrough",
            "drowsy": "drowsy",
            "deep_valley": "deep_valley",
            "active_twilight": "active_twilight",
            "dawn_recovery": "dawn_recovery",
        }.get(phase, phase)
        is_night = phase in ("drowsy", "light_sleep", "deep_sleep", "deep_valley", "burnthrough")
        icon_map = {
            "awake": "☀️",
            "active": "☀️",
            "daytime": "☀️",
            "winding_down": "🌆",
            "active_twilight": "🌆",
            "drowsy": "🌙",
            "light_sleep": "😴",
            "deep_sleep": "💤",
            "deep_valley": "💤",
            "arousal": "⚡",
            "dawn_recovery": "⚡",
            "burnthrough": "🔥",
        }
        label_map = {
            "awake": "清醒",
            "active": "活跃",
            "daytime": "清醒",
            "winding_down": "放松",
            "active_twilight": "黄昏微倦",
            "drowsy": "犯困",
            "light_sleep": "浅睡",
            "deep_sleep": "深睡",
            "deep_valley": "深睡沉寂",
            "arousal": "唤醒",
            "dawn_recovery": "黎明苏醒",
            "burnthrough": "熬穿",
            "excited": "亢奋",
            "night_active": "夜间活跃",
            "midnight_reflect": "午夜反思",
            "dawn_recover": "黎明恢复",
            "social_night": "社交夜",
            "quiet_contemplate": "安静沉思",
        }
        snap.night = NightBar(
            phase_name=label_map.get(phase, phase),
            phase_icon=icon_map.get(phase, "☀️"),
            is_night=is_night,
            pressure_total=float(src.get("pressure", 0.0) or 0.0),
            is_burnthrough=str(src.get("burnthrough", "") or "") == "True",
            expression_style=str(
                src.get("expression_style", "normal") or "normal"
            ),
        )

    def _build_safety_bar(
        self, snap: FullDashboardSnapshot, src: Dict[str, Any]
    ) -> None:
        level = str(src.get("safety_level", "安全") or "安全")
        score = float(src.get("safety_score", 0.0) or 0.0)
        threat = str(src.get("dominant_threat", "") or "")
        blocked = str(src.get("blocked", "False") or "False") == "True"
        penalty = float(src.get("bar_delta", 0.0) or 0.0)
        icon_map = {
            "安全": "🛡️",
            "低风险": "⚡",
            "中风险": "⚠️",
            "高风险": "🛑",
            "严重威胁": "🚨",
        }
        snap.safety = SafetyBar(
            level_label=level,
            level_icon=icon_map.get(level, "🛡️"),
            score=score,
            dominant_threat=threat,
            blocked=blocked,
            bar_penalty=penalty,
        )

    def _build_memory_bar(
        self, snap: FullDashboardSnapshot, src: Dict[str, Any]
    ) -> None:
        total = int(src.get("total", 0) or 0)
        util = float(src.get("utilization", 0.0) or 0.0)
        fresh = int(src.get("fresh", 0) or 0)
        stale = int(src.get("stale", 0) or 0)
        window = int(src.get("window", 0) or 0)
        snap.memory = MemoryBar(
            total_entries=total,
            utilization=util,
            fresh_count=fresh,
            stale_count=stale,
            window_size=window,
        )

    # ═════════════════ 裁定计算 ═════════════════

    def _compute_verdict(self, snap: FullDashboardSnapshot) -> None:
        l1 = self.l1_fast_check(snap.timestamp)
        if l1:
            snap.verdict = ActionVerdict(
                urgency=ActionUrgency.MUST_NOT_REPLY,
                should_process_message=False,
                should_reply=False,
                primary_reason=l1,
                trigger_path="HardBlockGate",
                legacy_trigger_path="L1_HARD_BLOCK",
                confidence=0.95,
            )
            return
        l2 = self.l2_contextual_eval(snap)
        if l2 == ActionUrgency.MUST_NOT_REPLY:
            snap.verdict = ActionVerdict(
                urgency=l2,
                should_process_message=True,
                should_reply=False,
                primary_reason="L2综合评估不通过",
                trigger_path="ContextGateReject",
                legacy_trigger_path="L2_CONTEXTUAL_REJECT",
                confidence=0.75,
            )
            return
        if l2 == ActionUrgency.CAN_SKIP:
            snap.verdict = ActionVerdict(
                urgency=l2,
                should_process_message=True,
                should_reply=False,
                primary_reason="L2评估：可跳过此消息",
                secondary_reasons=["进入观察但不回复"],
                trigger_path="ContextGateSkip",
                legacy_trigger_path="L2_CONTEXTUAL_SKIP",
                confidence=0.55,
            )
            return
        tone = self._derive_tone(snap)
        action = self._derive_action(snap, l2)
        urgency_final = (
            ActionUrgency.SHOULD_REPLY_SOON
            if l2 == ActionUrgency.MAY_REPLY_LATER
            else l2
        )
        snap.verdict = ActionVerdict(
            urgency=urgency_final,
            should_process_message=True,
            should_reply=True,
            primary_reason=f"L{
                '2' if l2 == ActionUrgency.MAY_REPLY_LATER else '3'}通过，{
                snap.mood.mental.label()}+{
                snap.attention.mode.label()}",
            recommended_tone=tone,
            suggested_action=action,
            trigger_path="EmergenceDecision",
            legacy_trigger_path="L3_FULL_EVALUATION",
            confidence=0.85 if l2 == ActionUrgency.MUST_REPLY_NOW else 0.70,
        )

    @staticmethod
    def _derive_tone(snap: FullDashboardSnapshot) -> str:
        if snap.safety.score > 0.5:
            return "谨慎克制"
        if snap.night.is_night and not snap.night.is_burnthrough:
            return "安静温和"
        if snap.mood.mental in (MentalState.CHEERFUL, MentalState.CALM):
            return "轻松自然"
        if snap.mood.mental == MentalState.BORED:
            return "好奇试探"
        if snap.social.posture == SocialPosture.WARM_ENGAGED:
            return "热情亲切"
        if snap.mood.mental in (MentalState.ANNOYED, MentalState.IRRITATED):
            return "冷淡简短"
        return "正常回应"

    @staticmethod
    def _derive_action(snap: FullDashboardSnapshot, l2: ActionUrgency) -> str:
        parts = []
        if l2 == ActionUrgency.MUST_REPLY_NOW:
            parts.append("立即处理并生成回复")
        elif l2 == ActionUrgency.SHOULD_REPLY_SOON:
            parts.append("准备回复，可在下一tick执行")
        else:
            parts.append("标记待回复，观察后续消息")
        if snap.vitality.level in (
            VitalityLevel.EXHAUSTED,
            VitalityLevel.CRASHED,
        ):
            parts.append("（注意：精力不足，建议短回复）")
        if snap.night.is_burnthrough:
            parts.append("（夜猫子模式激活，可能话多）")
        return "；".join(parts)

    # ═════════════════ 变更检测 ═════════════════

    def _detect_changes(self, snap: FullDashboardSnapshot) -> None:
        events = []
        if (
            self._prev_verdict_urgency is not None
            and self._prev_verdict_urgency != snap.verdict.urgency
        ):
            events.append(
                {
                    "type": "urgency_change",
                    "from": self._prev_verdict_urgency.label(),
                    "to": snap.verdict.urgency.label(),
                    "at": snap.timestamp,
                }
            )
        if len(events) > 0:
            self._change_history.extend(events)
            if len(self._change_history) > self._max_history:
                self._change_history = self._change_history[
                    -self._max_history:
                ]
        self._prev_verdict_urgency = snap.verdict.urgency

    # ────────────────── 快速查询辅助 ──────────────────

    def _get_quick_vitality(self, now: float) -> float:
        try:
            from src.chat.heart_flow.energy_manager import EnergyChainDimension
            _d6 = EnergyChainDimension.get_instance()
            _state = _d6._ensure_channel(self._channel_id)
            return max(0.02, _state.combined_ratio())
        except Exception as _exc:
            logger.warning(f"快速活力查询异常: {_exc}")
            return 0.5

    def _get_quick_night_phase(self) -> str:
        cached = getattr(self, "_cached_night_phase", None)
        if cached and hasattr(cached, "value"):
            return str(getattr(cached, "value", "") or "").strip().lower()
        return "awake"

    def _get_quick_safety(self) -> float:
        cached = getattr(self, "_cached_safety_assessment", None)
        if isinstance(cached, dict):
            return float(cached.get("score", 0.0) or 0.0)
        return 0.0

    def _get_quick_attention_mode(self) -> str:
        cached = getattr(self, "_cached_attention_snapshot", None)
        if isinstance(cached, dict):
            return self._normalize_attention_mode(
                cached.get("attention_mode", cached.get("state", ""))
            )
        return "scanning"

    def _get_quick_peek_desire(self) -> float:
        cached = getattr(self, "_cached_attention_snapshot", None)
        if isinstance(cached, dict):
            return self._metric_or_default(cached.get("peek_desire"), 0.3)
        return 0.3

    def _get_quick_last_look(self) -> float:
        cached = getattr(self, "_cached_attention_snapshot", None)
        if isinstance(cached, dict):
            return self._metric_or_default(cached.get("since_last_look_sec"), 999.0)
        return 999.0

    # ═════════════════ 公开接口 ═════════════════

    def get_status_line(self) -> str:
        snap = self.build_dashboard()
        return snap.to_status_line()

    def get_display_dict(self) -> Dict[str, Any]:
        snap = self.build_dashboard()
        return snap.to_display_dict()

    def get_verdict(self) -> ActionVerdict:
        snap = self.build_dashboard()
        return snap.verdict

    def get_change_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        return list(reversed(self._change_history[-limit:]))


def get_state_dashboard(channel_id: str) -> StateDashboardEngine:
    global _dashboard_singleton
    if _dashboard_singleton is None:
        _dashboard_singleton = {}
    if channel_id not in _dashboard_singleton:
        _dashboard_singleton[channel_id] = StateDashboardEngine(channel_id)
    return _dashboard_singleton[channel_id]

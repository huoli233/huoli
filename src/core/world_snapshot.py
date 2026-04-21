import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger
from src.common.singleton import SingletonMeta

logger = get_logger("世界快照")


def _metric_has_signal(value: Any, threshold: float = 1e-6) -> bool:
    """统一判断数值信号是否存在，避免 0/负值被真值判断吞掉。"""
    try:
        return abs(float(value or 0.0)) >= threshold
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  自身资源状态
# ---------------------------------------------------------------------------


@dataclass
class SelfResourceState:
    """bot自身状态。"""

    chat_energy: float = 100.0
    chat_ceiling: float = 100.0
    thinking_energy: float = 100.0
    thinking_ceiling: float = 100.0
    activity_level: float = 50.0
    channel_annoyance: float = 0.0
    consecutive_replies: int = 0
    silence_seconds: float = 0.0
    boredom: float = 0.0
    loneliness: float = 0.0
    social_desire: float = 0.0
    proactive_willingness: float = 0.0

    def chat_ratio(self) -> float:
        if self.chat_ceiling <= 0:
            return 0.0
        return max(0.0, min(1.0, self.chat_energy / self.chat_ceiling))

    def thinking_ratio(self) -> float:
        if self.thinking_ceiling <= 0:
            return 0.5
        return max(0.0, min(1.0, self.thinking_energy / self.thinking_ceiling))


# ---------------------------------------------------------------------------
#  主体状态
# ---------------------------------------------------------------------------


@dataclass
class SubjectState:
    """主体状态汇总。

    第一阶段收拢主链阶段、等待态与是否可回复。
    第二阶段扩展到观看等级、夜间周期、代谢摸鱼等主体根状态，
    让消费方不再分别向各子系统取值。
    """

    flow_phase: str = "standby"
    flow_phase_label: str = "待命"
    phase_duration_sec: float = 0.0
    pending_active: bool = False
    pending_elapsed_sec: float = 0.0
    pending_progress: float = 0.0
    pending_expected: str = ""
    can_reply: bool = True
    # 观看等级（watch_state_machine）
    watch_level: str = "peek"
    watch_level_label: str = "瞥一眼"
    # 夜间周期（night_cycle_system）
    night_phase: str = "day"
    is_sleeping: bool = False
    # 代谢/摸鱼（metabolism_engine）
    loafing_level: float = 0.0
    # 综合情绪基调（主体自身，非对目标用户）
    overall_mood: str = "平静"
    # 新增群友主观存在字段（第二阶段优化）
    social_openness: float = 0.5
    quiet_desire: float = 0.3
    group_attention_level: float = 0.6
    self_consciousness_in_group: float = 0.4
    psychological_freshness: float = 1.0


# ---------------------------------------------------------------------------
#  目标用户状态
# ---------------------------------------------------------------------------


@dataclass
class TargetUserState:
    """群友印象状态。

    所有字段均从群友互动角度定义：这个群友如何看待我、我在群里的位置、互动评价。
    禁止使用抽象用户扩展术语，全部以群友语境描述。
    字段归属：
      - relationship_level/custom_label/trend_direction/interaction_count：群友关系档案（权威）
      - social_value/trust_value/annoyance_value：群内社交值核心（权威）
      - favorability：social_value 的别名，仅为兼容保留
      - affection/psychological_pressure/trauma_score/mood：群友情感追踪（权威）
      - trust_score：trust_value 的情感侧视图，仅为兼容保留
      - positive_dim/negative_dim：群友心理核心评价
    """

    group_friend_id: str = ""
    group_friend_name: str = ""
    # 群友关系档案 (relationship_tracker) - 权威
    relationship_level: int = 2
    custom_label: str = ""
    trend_direction: str = "稳定"
    interaction_count: int = 0
    # 群内社交值 (social_value_core) - 权威
    social_value: float = 0.0
    favorability: float = 0.0
    trust_value: float = 0.0
    annoyance_value: float = 0.0
    # 群友情感追踪 (emotion_tracker) - 权威
    affection: float = 0.0
    trust_score: float = 0.0
    trauma_score: float = 0.0
    psychological_pressure: float = 0.0
    # 群友心理核心 (psychological_core)
    positive_dim: float = 0.0
    negative_dim: float = 0.0
    # 群友印象备注
    impression_style: str = ""
    mood: str = "平静"
    attribute_influences: Dict[str, Any] = field(default_factory=dict)

    @property
    def user_id(self) -> str:
        """兼容旧消费方读取 user_id。"""
        return self.group_friend_id

    @user_id.setter
    def user_id(self, value: str) -> None:
        self.group_friend_id = str(value or "")

    @property
    def user_name(self) -> str:
        """兼容旧消费方读取 user_name。"""
        return self.group_friend_name

    @user_name.setter
    def user_name(self, value: str) -> None:
        self.group_friend_name = str(value or "")

    def profile_summary(self) -> str:
        """生成群友画像摘要字符串（从群友视角评价）"""
        bits: List[str] = []
        if self.custom_label:
            bits.append(f"群友标签={self.custom_label}")
        if self.relationship_level is not None:
            bits.append(f"群内关系级别={self.relationship_level}")
        if _metric_has_signal(self.social_value):
            bits.append(f"群社交度={self.social_value:.1f}")
        if _metric_has_signal(self.trust_value):
            bits.append(f"群信任度={self.trust_value:.1f}")
        if _metric_has_signal(self.annoyance_value):
            bits.append(f"群烦躁度={self.annoyance_value:.1f}")
        if self.mood and self.mood != "平静":
            bits.append(f"群印象气氛={self.mood}")
        return " | ".join(bits)


# ---------------------------------------------------------------------------
#  场景状态
# ---------------------------------------------------------------------------


@dataclass
class SceneState:
    """当前频道或会话的场景状态。"""

    channel_id: str = ""
    is_group: bool = True
    active_users: List[str] = field(default_factory=list)
    current_topics: List[str] = field(default_factory=list)
    relationship_candidates: List[Tuple[str, float]] = field(default_factory=list)
    # 会话追踪 (session_tracker)
    session_phase: str = ""
    last_topic: str = ""
    last_mood: str = ""
    # 频道氛围 (emotion_stream / ChannelMoodTracker)
    vexation: float = 0.0
    weariness: float = 0.0


# ---------------------------------------------------------------------------
#  组合快照
# ---------------------------------------------------------------------------


@dataclass
class WorldSnapshot:
    """全局决策快照：一次采集，多处消费。"""

    subject: SubjectState = field(default_factory=SubjectState)
    self_resources: SelfResourceState = field(default_factory=SelfResourceState)
    target_user: TargetUserState = field(default_factory=TargetUserState)
    scene: SceneState = field(default_factory=SceneState)
    # 行为信号（由主链分类后注入）
    behavior_category: str = "neutral"
    behavior_severity: float = 0.0
    behavior_reason: str = ""
    # 外部提示（主链传入的额外上下文）
    repeated_topic_pressure: float = 0.0
    last_user_intent: str = ""
    memory_hint: str = ""
    context_execution_block: str = ""
    is_new_user: bool = False
    # 元数据
    snapshot_time: float = field(default_factory=time.time)

    def to_relation_dict(self) -> Dict[str, Any]:
        """兼容 Enhanced 现有 _last_relation_snapshot 字典格式。
        注意：此方法保持向后兼容，新消费方应使用 to_canonical_state() 代替。
        """
        u = self.target_user
        r = self.self_resources
        return {
            "subject_phase": self.subject.flow_phase,
            "subject_phase_label": self.subject.flow_phase_label,
            "subject_can_reply": self.subject.can_reply,
            "pending_active": self.subject.pending_active,
            "pending_elapsed_sec": self.subject.pending_elapsed_sec,
            "pending_progress": self.subject.pending_progress,
            "chat_value": r.chat_energy,
            "activity_level": r.activity_level,
            "shared_social_value": getattr(r, "shared_social_value", 0.0),
            "social_value": u.social_value,
            "trust_value": u.trust_value,
            "annoyance_value": u.annoyance_value,
            "relationship_level": u.relationship_level,
            "custom_label": u.custom_label,
            "interaction_count": u.interaction_count,
            "trend_direction": u.trend_direction,
            "positive_dim": u.positive_dim,
            "negative_dim": u.negative_dim,
            "affection": u.affection,
            "trust_score": u.trust_score,
            "psychological_pressure": u.psychological_pressure,
            "trauma_score": u.trauma_score,
            "mood": u.mood,
            "attribute_influences": u.attribute_influences,
            "behavior_signal": {
                "category": self.behavior_category,
                "severity": self.behavior_severity,
                "reason": self.behavior_reason,
            },
        }

    def to_canonical_state(self) -> Dict[str, Any]:
        """完整版状态输出。
        包含主体、资源、目标用户、场景、行为信号、元数据及所有扩展字段，
        确保决策消费方可一次性获取全部所需上下文，避免多次跨模块查询。
        """
        s = self.subject
        r = self.self_resources
        u = self.target_user
        sc = self.scene
        # 完整主体状态
        subject_section = {
            "phase": s.flow_phase,
            "phase_label": s.flow_phase_label,
            "phase_duration_sec": round(s.phase_duration_sec, 1),
            "can_reply": s.can_reply,
            "pending_active": s.pending_active,
            "pending_elapsed_sec": round(s.pending_elapsed_sec, 1),
            "pending_progress": round(s.pending_progress, 2),
            "pending_expected": s.pending_expected,
            "watch_level": s.watch_level,
            "watch_level_label": getattr(s, "watch_level_label", "瞥一眼"),
            "night_phase": s.night_phase,
            "is_sleeping": s.is_sleeping,
            "loafing_level": round(s.loafing_level, 2),
            "overall_mood": s.overall_mood,
            "flow_phase": s.flow_phase,
            # 群友主观存在字段（优化后新增）
            "social_openness": round(s.social_openness, 2),
            "quiet_desire": round(s.quiet_desire, 2),
            "group_attention_level": round(s.group_attention_level, 2),
            "self_consciousness_in_group": round(s.self_consciousness_in_group, 2),
            "psychological_freshness": round(s.psychological_freshness, 2),
        }
        # 完整资源状态
        resources_section = {
            "chat_energy": round(r.chat_energy, 1),
            "chat_ceiling": round(r.chat_ceiling, 1),
            "chat_ratio": round(r.chat_ratio(), 3),
            "thinking_energy": round(r.thinking_energy, 1),
            "thinking_ceiling": round(r.thinking_ceiling, 1),
            "thinking_ratio": round(r.thinking_ratio(), 3),
            "activity_level": round(r.activity_level, 1),
            "shared_social_value": round(getattr(r, "shared_social_value", 0.0), 2),
            "channel_annoyance": round(r.channel_annoyance, 2),
            "silence_seconds": round(r.silence_seconds, 1),
            "consecutive_replies": r.consecutive_replies,
            "boredom": round(r.boredom, 2),
            "loneliness": round(r.loneliness, 2),
            "social_desire": round(r.social_desire, 2),
            "proactive_willingness": round(getattr(r, "proactive_willingness", 0.0), 2),
        }
        # 完整目标用户状态
        target_section = {
            "group_friend_id": getattr(
                u,
                "group_friend_id",
                u.user_id if hasattr(u, "user_id") else "",
            ),
            "group_friend_name": getattr(
                u,
                "group_friend_name",
                u.user_name if hasattr(u, "user_name") else "",
            ),
            "relationship_level": u.relationship_level,
            "custom_label": u.custom_label,
            "trend_direction": u.trend_direction,
            "interaction_count": u.interaction_count,
            "social_value": round(u.social_value, 2),
            "favorability": round(u.favorability, 2),
            "trust_value": round(u.trust_value, 2),
            "annoyance_value": round(u.annoyance_value, 2),
            "affection": round(u.affection, 2),
            "trust_score": round(u.trust_score, 2),
            "trauma_score": round(u.trauma_score, 2),
            "psychological_pressure": round(u.psychological_pressure, 2),
            "positive_dim": round(u.positive_dim, 2),
            "negative_dim": round(u.negative_dim, 2),
            "mood": u.mood,
            "impression_style": u.impression_style,
            "profile_summary": u.profile_summary(),
            "attribute_influences": u.attribute_influences,
        }
        # 完整场景状态
        scene_section = {
            "channel_id": sc.channel_id,
            "is_group": sc.is_group,
            "active_users": sc.active_users,
            "current_topics": sc.current_topics,
            "session_phase": sc.session_phase,
            "last_topic": sc.last_topic,
            "last_mood": sc.last_mood,
            "vexation": round(sc.vexation, 2),
            "weariness": round(sc.weariness, 2),
            "relationship_candidates": sc.relationship_candidates,
        }
        # 行为信号
        behavior_section = {
            "category": self.behavior_category,
            "severity": round(self.behavior_severity, 2),
            "reason": self.behavior_reason,
        }
        # 元数据与上下文
        meta_section = {
            "snapshot_time": self.snapshot_time,
            "repeated_topic_pressure": round(self.repeated_topic_pressure, 2),
            "last_user_intent": self.last_user_intent,
            "memory_hint": self.memory_hint,
            "context_execution_block": self.context_execution_block,
            "is_new_user": self.is_new_user,
        }
        return {
            "subject": subject_section,
            "resources": resources_section,
            "target": target_section,
            "scene": scene_section,
            "behavior": behavior_section,
            "meta": meta_section,
            "summary": {
                "overall_readiness": ("high" if s.can_reply and r.chat_ratio() > 0.3 else "low"),
                "social_tone": u.mood,
                "key_constraints": [k for k, v in resources_section.items() if isinstance(v, (int, float)) and v < 0.3],
            },
        }

    def to_rapport_dict(self) -> Dict[str, Any]:
        """兼容导出关系字典，主语义固定为 canonical relation contract。"""
        u = self.target_user
        sc = self.scene
        memory_bits: List[str] = []
        if sc.session_phase:
            memory_bits.append(f"会话阶段={sc.session_phase}")
        if sc.last_topic:
            memory_bits.append(f"上次话题={sc.last_topic}")
        if sc.last_mood:
            memory_bits.append(f"上次气氛={sc.last_mood}")
        return {
            "relation_contract_version": 2,
            "unified_source": "world_snapshot",
            "relationship": u.custom_label or "",
            "affection": u.affection,
            "trust": u.trust_value,
            "social_value": u.social_value,
            "positive_dim": u.positive_dim,
            "negative_dim": u.negative_dim,
            "favorability": u.favorability,
            "trust_value": u.trust_value,
            "trust_score": u.trust_score,
            "annoyance_value": u.annoyance_value,
            "psychological_pressure": u.psychological_pressure,
            "trauma_score": u.trauma_score,
            "mood": u.mood,
            "interaction_count": u.interaction_count,
            "relationship_level": u.relationship_level,
            "custom_label": u.custom_label,
            "trend_direction": u.trend_direction,
            "profile_summary": u.profile_summary(),
            "memory_summary": " | ".join(memory_bits),
            "execution_hint": "先形成一句短的心里话，再决定要不要回；如果要回，后续回复必须顺着这句心里话。",
        }


def build_world_snapshot_sync(
    channel_id: str,
    user_id: str = "",
    user_name: str = "",
    ambient: Optional[Dict[str, Any]] = None,
) -> WorldSnapshot:
    """同步构建最小可用世界快照，供同步消费口复用统一导出 contract。"""
    snap = WorldSnapshot(
        self_resources=SelfResourceState(),
        target_user=TargetUserState(
            group_friend_id=user_id,
            group_friend_name=user_name,
        ),
        scene=SceneState(channel_id=channel_id),
    )
    snap.snapshot_time = time.time()
    lookup = str(user_id or user_name or "").strip()
    if lookup:
        _collect_social_affect_sync(snap, channel_id, lookup)
        _collect_emotion_state_sync(snap, channel_id, lookup)
        _collect_psychological_state_sync(snap, channel_id, lookup)
    _collect_session_memoir(snap, channel_id, lookup)
    _collect_ambient(snap, ambient)
    _reconcile_overlapping_values(snap)
    mark_first_stage_complete(snap)
    return snap


def build_relation_rapport_snapshot(
    channel_id: str,
    user_id: str = "",
    user_name: str = "",
    ambient: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """同步导出统一关系视图，供同步消费口读取 canonical relation contract。"""
    return build_world_snapshot_sync(
        channel_id=channel_id,
        user_id=user_id,
        user_name=user_name,
        ambient=ambient,
    ).to_rapport_dict()


# ---------------------------------------------------------------------------
# 采集器：从各子系统聚合数据
# ---------------------------------------------------------------------------


async def build_world_snapshot(
    channel_id: str,
    user_id: str = "",
    user_name: str = "",
    ambient: Optional[Dict[str, Any]] = None,
) -> WorldSnapshot:
    """
    一次性从所有子系统采集数据，构建决策快照。
    每个子系统的采集互相独立，单个失败不影响整体。
    """
    snap = WorldSnapshot(
        self_resources=SelfResourceState(),
        target_user=TargetUserState(
            group_friend_id=user_id,
            group_friend_name=user_name,
        ),
        scene=SceneState(channel_id=channel_id),
    )
    snap.snapshot_time = time.time()
    # 并行采集互不依赖的子系统
    tasks = [
        _collect_subject_state(snap, channel_id),
        _collect_subject_extensions(snap, channel_id),
        _collect_energy_state(snap, channel_id),
        _collect_silence_and_frequency(snap, channel_id),
        _collect_emotion_driven_state(snap, channel_id),
    ]
    if user_id:
        tasks.append(_collect_social_affect(snap, channel_id, user_id))
        tasks.append(_collect_emotion_state(snap, channel_id, user_id))
        tasks.append(_collect_psychological_state(snap, channel_id, user_id))
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=10.0,
        )
    except asyncio.TimeoutError:
        logger.debug("世界快照采集超时(10s)，部分数据可能缺失")
    # 同步采集（非异步）
    _collect_session_memoir(snap, channel_id, user_id or user_name)
    _collect_ambient(snap, ambient)
    _reconcile_overlapping_values(snap)
    # 第一阶段完成标记（优化后新增）
    mark_first_stage_complete(snap)
    return snap


# ---------------------------------------------------------------------------
#  各子系统采集函数
# ---------------------------------------------------------------------------


async def _collect_subject_state(snap: WorldSnapshot, channel_id: str) -> None:
    """采集主体主链状态。

    统一汇总 phase_coordinator 与 pending_orchestrator 的权威结果，
    作为第一阶段的主体状态对象骨架。
    """
    try:
        from src.chat.heart_flow.state_machine import get_phase_coordinator

        coordinator = get_phase_coordinator()
        phase = coordinator.current_phase(channel_id)
        snap.subject.flow_phase = str(getattr(phase, "value", "standby") or "standby")
        snap.subject.flow_phase_label = str(phase.label())
        snap.subject.phase_duration_sec = round(coordinator.phase_duration_sec(channel_id), 1)
        snap.subject.can_reply = snap.subject.flow_phase in {
            "standby",
            "engaged",
        }
    except Exception as exc:
        logger.debug(f"[快照] 主体阶段采集失败: {exc}")
    try:
        from src.chat.heart_flow.waiting_handler import (
            get_pending_orchestrator,
        )

        pending = get_pending_orchestrator()
        pending_snap = pending.snapshot(channel_id)
        snap.subject.pending_active = bool(pending_snap.get("active", False))
        snap.subject.pending_elapsed_sec = float(pending_snap.get("elapsed_sec", 0.0) or 0.0)
        snap.subject.pending_progress = float(pending_snap.get("progress", 0.0) or 0.0)
        snap.subject.pending_expected = str(pending_snap.get("anticipated", "") or "")
        if snap.subject.pending_active:
            snap.subject.can_reply = False
    except Exception as exc:
        logger.debug(f"[快照] 主体等待态采集失败: {exc}")


async def _collect_subject_extensions(snap: WorldSnapshot, channel_id: str) -> None:
    """采集主体扩展字段：观看等级、夜间周期、代谢摸鱼、综合情绪及群友主观存在指标"""
    # 观看等级
    try:
        from src.core.watch_state_machine import get_watch_machine

        wsm = get_watch_machine(channel_id)
        snap.subject.watch_level = wsm.current_level.value
        snap.subject.watch_level_label = wsm.current_level.label()
    except Exception as _e:
        logger.warning(f"观看等级采集异常: {_e}")
    # 夜间周期：D6 EnergyChainDimension 四阶段系统读取
    try:
        from src.chat.heart_flow.energy_manager import EnergyChainDimension

        _d6 = EnergyChainDimension.get_instance()
        _night = _d6._get_night_mode(channel_id)
        snap.subject.night_phase = _night.phase
        _is_sleeping = _night.phase == "DEEP_VALLEY"
        try:
            from src.core.night_cycle_system import get_night_cycle

            _ncs = get_night_cycle(channel_id)
            _phase_obj = getattr(_ncs.state_snapshot, "current_phase", None)
            _phase_value = str(getattr(_phase_obj, "value", _phase_obj) or "").strip()
            if _phase_value:
                snap.subject.night_phase = _phase_value
            _ncs_phase = getattr(_ncs.state_snapshot, "current_phase", None)
            if _ncs_phase is not None and hasattr(_ncs_phase, "is_sleeping"):
                _is_sleeping = bool(_ncs_phase.is_sleeping)
        except Exception as _night_flag_exc:
            logger.debug(f"[快照] 夜间睡眠标记细化失败: {_night_flag_exc}")
        snap.subject.is_sleeping = _is_sleeping
    except Exception as _e:
        logger.warning(f"夜间周期采集异常: {_e}")
    # 代谢摸鱼：D6 EnergyChainDimension 读取
    try:
        from src.chat.heart_flow.energy_manager import EnergyChainDimension

        _d6b = EnergyChainDimension.get_instance()
        _d6s = _d6b._ensure_channel(channel_id)
        _combined = _d6s.combined_ratio()
        snap.subject.loafing_level = max(0.0, min(100.0, (1.0 - _d6s.activity_level / 100.0) * 80.0))
    except Exception as _e:
        logger.warning(f"代谢状态采集异常: {_e}")
    # 新增群友主观存在指标优化（第二阶段准备）
    try:
        from src.chat.heart_flow.emotion_driven_core import get_emotion_driven_core

        core = get_emotion_driven_core()
        state = core.get_state_snapshot(channel_id) if hasattr(core, "get_state_snapshot") else None
        if state:
            snap.subject.social_openness = float(state.get("social_openness", 0.5))
            snap.subject.quiet_desire = float(state.get("quiet_desire", 0.3))
            snap.subject.group_attention_level = float(state.get("group_attention_level", 0.6))
            snap.subject.self_consciousness_in_group = float(state.get("self_consciousness", 0.4))
            snap.subject.psychological_freshness = float(state.get("freshness", 1.0))
            boredom = float(state.get("boredom", 0.0) or 0.0)
            loneliness = float(state.get("loneliness", 0.0) or 0.0)
            social_desire = float(state.get("social_desire", 0.0) or 0.0)
            if boredom > 0.7 and social_desire < 0.3:
                snap.subject.overall_mood = "厌倦"
            elif loneliness > 0.6 and social_desire > 0.6:
                snap.subject.overall_mood = "想说话"
            elif snap.subject.quiet_desire > 0.7:
                snap.subject.overall_mood = "想安静"
            elif snap.subject.social_openness > 0.7:
                snap.subject.overall_mood = "外向打开"
    except Exception:
        # 默认值已设置，无需额外处理
        pass


async def _collect_energy_state(snap: WorldSnapshot, channel_id: str) -> None:
    """采集能量与精力状态，读取 D6 EnergyChainDimension，替代旧 VitalityPool 和 SharedResource。"""
    try:
        from src.chat.heart_flow.energy_manager import EnergyChainDimension

        _d6 = EnergyChainDimension.get_instance()
        _d6_snap = _d6.capture_snapshot(channel_id)
        _d6_state = _d6._ensure_channel(channel_id)
        snap.self_resources.chat_energy = _d6_snap.chat_pool
        snap.self_resources.chat_ceiling = _d6_snap.chat_ceiling
        snap.self_resources.thinking_energy = _d6_snap.thinking_value
        snap.self_resources.thinking_ceiling = _d6_snap.thinking_ceiling
        snap.self_resources.channel_annoyance = _d6_snap.annoyance_level
        snap.self_resources.activity_level = _d6_state.activity_level
        snap.self_resources.shared_social_value = _d6_state.social_value
    except Exception as exc:
        logger.warning(f"[快照] 能量采集失败(D6): {exc}")


async def _collect_silence_and_frequency(snap: WorldSnapshot, channel_id: str) -> None:
    """采集静默时长和连续回复数"""
    try:
        from src.chat.proactive.silence_watcher import get_quiet_monitor

        snap.self_resources.silence_seconds = get_quiet_monitor().measure_silence_sec(channel_id)
    except Exception as _e:
        logger.warning(f"静默时长采集异常: {_e}")
    try:
        from src.chat.heart_flow.frequency_control import (
            acquire_frequency_manager,
        )

        fm = acquire_frequency_manager()
        if hasattr(fm, "get_consecutive_replies"):
            snap.self_resources.consecutive_replies = fm.get_consecutive_replies(channel_id)
    except Exception as _e:
        logger.warning(f"连续回复计数采集异常: {_e}")


async def _collect_emotion_driven_state(snap: WorldSnapshot, channel_id: str) -> None:
    """采集 EmotionDrivenCore 的主观感受（无聊、孤独、社交欲望、主动意愿）。"""
    try:
        from src.chat.heart_flow.emotion_driven_core import get_emotion_driven_core

        core = get_emotion_driven_core()
        state = core.get_state_snapshot(channel_id)
        if state is None:
            return
        res = snap.self_resources
        res.boredom = float(state.get("boredom", 0.0) or 0.0)
        res.loneliness = float(state.get("loneliness", 0.0) or 0.0)
        res.social_desire = float(state.get("social_desire", 0.0) or 0.0)
        res.proactive_willingness = float(state.get("proactive_willingness", 0.0) or 0.0)
    except Exception as exc:
        logger.debug(f"[快照] 主观感受采集失败: {exc}")


async def _collect_social_affect(snap: WorldSnapshot, channel_id: str, user_id: str) -> None:
    """采集社交情感状态（通过统一融合器）"""
    try:
        from src.modules.social_value.social_affect_fuser import (
            get_social_affect_fuser,
        )

        fuser = get_social_affect_fuser()
        affect = await fuser.get_snapshot(user_id, channel_id)
        u = snap.target_user
        u.social_value = affect.social_score
        u.relationship_level = int(affect.phase)
        u.custom_label = affect.phase_label
        u.trend_direction = affect.trend
        u.trust_value = affect.trust_value
        if affect.impression_nick:
            u.impression_style = affect.impression_nick
        # 互动次数从公开接口补充
        record = await fuser.read_full_record(user_id, channel_id)
        if record is not None:
            u.interaction_count = int(record.interaction_count)
            if _metric_has_signal(record.annoyance_value):
                u.annoyance_value = float(record.annoyance_value)
    except Exception as exc:
        logger.debug(f"[快照] 社交情感采集失败: {exc}")


async def _collect_emotion_state(snap: WorldSnapshot, channel_id: str, user_id: str) -> None:
    """采集情感追踪状态。"""
    try:
        from src.modules.modcore.dynamic_persona.emotion_tracker import (
            get_emotion_tracker,
        )

        state = get_emotion_tracker(channel_id).get_user_state(user_id, create_if_missing=False)
        if state is None:
            return
        u = snap.target_user
        u.affection = float(getattr(state, "affection", 0.0) or 0.0)
        u.trust_score = float(getattr(state, "trust_score", 0.0) or 0.0)
        if _metric_has_signal(u.trust_score) and (
            not _metric_has_signal(u.trust_value) or abs(u.trust_score) > abs(u.trust_value)
        ):
            u.trust_value = u.trust_score
        u.trauma_score = float(getattr(state, "trauma_score", 0.0) or 0.0)
        u.psychological_pressure = float(getattr(state, "psychological_pressure", 0.0) or 0.0)
        mood_val = getattr(state, "mood", None) or getattr(state, "current_mood", None)
        if mood_val:
            u.mood = str(mood_val)
    except Exception as exc:
        logger.debug(f"[快照] 情感状态采集失败: {exc}")


def _collect_social_affect_sync(snap: WorldSnapshot, channel_id: str, user_id: str) -> None:
    """同步采集社交情感状态，供同步消费口复用统一关系真源。"""
    try:
        from src.modules.social_value.social_affect_fuser import (
            get_social_affect_fuser,
        )

        fuser = get_social_affect_fuser()
        dossier = fuser.get_dossier(user_id, channel_id)
        u = snap.target_user
        if dossier is None:
            return
        latest_score = 0.0
        recent_values = list(getattr(dossier, "recent_values", []) or [])
        if recent_values:
            latest_score = float(getattr(recent_values[-1], "value", 0.0) or 0.0)
        u.social_value = latest_score
        u.relationship_level = int(getattr(dossier, "current_phase", u.relationship_level) or u.relationship_level)
        u.custom_label = str(
            fuser.get_phase_label(user_id, channel_id)
            or getattr(dossier, "custom_nick", "")
            or u.custom_label
        )
        u.trend_direction = str(
            getattr(dossier, "trend_label", "")
            or fuser.get_trend(user_id, channel_id)
            or u.trend_direction
        )
        u.trust_value = float(latest_score * fuser.get_phase_weight(user_id, channel_id))
        u.interaction_count = len(recent_values)
        impression = getattr(dossier, "impression", None)
        if impression is not None:
            u.impression_style = str(
                getattr(impression, "current_nick", "")
                or getattr(impression, "dialogue_style", "")
                or u.impression_style
            )
    except Exception as exc:
        logger.debug(f"[快照] 同步社交情感采集失败: {exc}")


def _collect_emotion_state_sync(snap: WorldSnapshot, channel_id: str, user_id: str) -> None:
    """同步采集情感追踪状态。"""
    try:
        from src.modules.modcore.dynamic_persona.emotion_tracker import (
            get_emotion_tracker,
        )

        state = get_emotion_tracker(channel_id).get_user_state(user_id, create_if_missing=False)
        if state is None:
            return
        u = snap.target_user
        u.affection = float(getattr(state, "affection", 0.0) or 0.0)
        u.trust_score = float(getattr(state, "trust_score", 0.0) or 0.0)
        if _metric_has_signal(u.trust_score) and (
            not _metric_has_signal(u.trust_value) or abs(u.trust_score) > abs(u.trust_value)
        ):
            u.trust_value = u.trust_score
        u.trauma_score = float(getattr(state, "trauma_score", 0.0) or 0.0)
        u.psychological_pressure = float(getattr(state, "psychological_pressure", 0.0) or 0.0)
        mood_val = getattr(state, "mood", None) or getattr(state, "current_mood", None)
        if mood_val:
            u.mood = str(mood_val)
    except Exception as exc:
        logger.debug(f"[快照] 同步情感状态采集失败: {exc}")


async def _collect_psychological_state(snap: WorldSnapshot, channel_id: str, user_id: str) -> None:
    """采集心理核心状态。"""
    try:
        from src.modules.modcore.psychological_core import (
            get_psychological_core,
        )

        core = get_psychological_core()
        if core is None:
            return
        state = core.get_user_psychological_state(channel_id, user_id)
        if not state:
            return
        u = snap.target_user
        u.positive_dim = float(state.get("positive_dim", 0.0) or 0.0)
        u.negative_dim = float(state.get("negative_dim", 0.0) or 0.0)
        _trust_val = state.get("trust_value", state.get("trust"))
        if _trust_val is not None and not getattr(u, "trust_value", 0.0):
            u.trust_value = float(_trust_val)
        _annoy_val = state.get("annoyance_value", state.get("annoyance"))
        if _annoy_val is not None and not getattr(u, "annoyance_value", 0.0):
            u.annoyance_value = float(_annoy_val)
    except Exception as exc:
        logger.debug(f"[快照] 心理状态采集失败: {exc}")


def _collect_psychological_state_sync(snap: WorldSnapshot, channel_id: str, user_id: str) -> None:
    """同步采集心理核心状态。"""
    try:
        from src.modules.modcore.psychological_core import (
            get_psychological_core,
        )

        core = get_psychological_core()
        if core is None:
            return
        state = core.get_user_psychological_state(channel_id, user_id)
        if not state:
            return
        u = snap.target_user
        u.positive_dim = float(state.get("positive_dim", 0.0) or 0.0)
        u.negative_dim = float(state.get("negative_dim", 0.0) or 0.0)
        _trust_val = state.get("trust_value", state.get("trust"))
        if _trust_val is not None and not getattr(u, "trust_value", 0.0):
            u.trust_value = float(_trust_val)
        _annoy_val = state.get("annoyance_value", state.get("annoyance"))
        if _annoy_val is not None and not getattr(u, "annoyance_value", 0.0):
            u.annoyance_value = float(_annoy_val)
    except Exception as exc:
        logger.debug(f"[快照] 同步心理状态采集失败: {exc}")


def _collect_session_memoir(snap: WorldSnapshot, channel_id: str, lookup: str) -> None:
    """采集会话追踪记忆（同步方法）"""
    if not lookup:
        return
    try:
        from src.chat.proactive.session_tracker import get_memoir_cabinet

        memoir = get_memoir_cabinet().retrieve(lookup)
        if memoir is None:
            return
        phase = getattr(memoir, "phase", None)
        snap.scene.session_phase = str(getattr(phase, "value", phase) or "") if phase else ""
        snap.scene.last_topic = str(getattr(memoir, "last_topic", "") or "")
        snap.scene.last_mood = str(getattr(memoir, "last_mood", "") or "")
    except Exception as exc:
        logger.debug(f"[快照] 会话记忆采集失败: {exc}")


def _collect_ambient(snap: WorldSnapshot, ambient: Optional[Dict[str, Any]]) -> None:
    """从频道氛围采样中提取频道级情绪数据；若外部未传入则主动从 ChannelMoodTracker 读取"""
    if ambient is not None:
        snap.scene.vexation = float(ambient.get("vexation", 0.0) or 0.0)
        snap.scene.weariness = float(ambient.get("weariness", 0.0) or 0.0)
        return
    try:
        from src.chat.heart_flow.emotion_stream import get_channel_mood_tracker

        ledger = get_channel_mood_tracker().fetch_mood(snap.scene.channel_id)
        if ledger is not None:
            snap.scene.vexation = float(getattr(ledger, "vexation", 0.0) or 0.0)
            snap.scene.weariness = float(getattr(ledger, "weariness", 0.0) or 0.0)
    except Exception as exc:
        logger.debug(f"[快照] 频道氛围主动采集失败: {exc}")


def _reconcile_overlapping_values(snap: WorldSnapshot) -> None:
    """
    多个子系统可能提供同一语义的数值（如 trust）。
    此函数对重叠字段做最终校准，确保一致性。
    """
    u = snap.target_user
    # canonical -> alias 单向镜像：兼容字段不能再反向污染主语义字段。
    if _metric_has_signal(u.social_value) and not _metric_has_signal(u.favorability):
        u.favorability = u.social_value
    if _metric_has_signal(u.trust_value) and not _metric_has_signal(u.trust_score):
        u.trust_score = u.trust_value


# 第一阶段完成优化：添加 PanelIntegrationManager 类，封装 panel 统一读取


class PanelIntegrationManager(metaclass=SingletonMeta):
    """第一阶段收口专用：统一panel从snapshot读取逻辑
    避免消费方分散拼接，全部通过此管理器获取标准化输出
    线程安全单例，基于统一单例工具
    """

    _initialized = False

    def initialize(self):
        if not self._initialized:
            self._initialized = True
            logger.info("PanelIntegrationManager 第一阶段收口初始化完成")

    def get_standard_panel_data(self, snapshot: WorldSnapshot) -> Dict[str, Any]:
        """从完整 snapshot 生成标准化 panel 数据（第一阶段统一入口）。"""
        canon = snapshot.to_canonical_state()
        panel_data = {
            "subject_status": canon["subject"],
            "resource_metrics": canon["resources"],
            "group_friend_impression": canon["target"],  # 使用群友定位
            "scene_context": canon.get("scene", {}),
            "behavior_signal": canon.get("behavior", {}),
            "meta_context": canon.get("meta", {}),
            "readiness_summary": canon.get("summary", {}),
            "panel_version": "first_stage_complete_v1",
            "unified_source": "world_snapshot",
            "last_reconciled": time.time(),
        }
        # 添加第一阶段特定panel字段
        panel_data["can_reply_confidence"] = canon["subject"].get("can_reply", True)
        panel_data["group_attention_score"] = canon["subject"].get("group_attention_level", 0.6)
        panel_data["quiet_mode_active"] = canon["subject"].get("quiet_desire", 0.3) > 0.5
        return panel_data

    def integrate_with_panel_adapter(self, snapshot: WorldSnapshot):
        """与 panel_state_adapter 集成（第一阶段完成标志）。"""
        try:
            from src.core.panel_state_adapter import get_panel_adapter

            get_panel_adapter()
            data = self.get_standard_panel_data(snapshot)
            # 模拟集成调用
            logger.debug(f"Panel集成成功，数据字段数: {len(data)}")
            return data
        except Exception as e:
            logger.debug(f"Panel集成警告: {e}")
            return self.get_standard_panel_data(snapshot)


# 第一阶段完成全局实例
_panel_manager = None


def get_panel_integration_manager():
    global _panel_manager
    if _panel_manager is None:
        _panel_manager = PanelIntegrationManager()
        _panel_manager.initialize()
    return _panel_manager


# 第一阶段完成标记函数
_first_stage_logged = False


def mark_first_stage_complete(snapshot: WorldSnapshot) -> bool:
    """标记第一阶段全部完成，验证统一主体态覆盖。"""
    global _first_stage_logged
    manager = get_panel_integration_manager()
    panel_data = manager.get_standard_panel_data(snapshot)
    required_keys = [
        "subject_status",
        "resource_metrics",
        "group_friend_impression",
        "scene_context",
    ]
    coverage = all(k in panel_data for k in required_keys)
    if coverage and not _first_stage_logged:
        _first_stage_logged = True
        logger.info("第一阶段全部完成：主体态统一、panel集成、群友定位全部就绪")
    return coverage

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("panel_adapter")

_adapter_singleton = None


# ---------------------------------------------------------------------------
#  面板区块数据结构
# ---------------------------------------------------------------------------


@dataclass
class VitalsGauge:
    """全局体征状态栏"""

    chat_energy_ratio: float = 1.0
    thinking_energy_ratio: float = 1.0
    activity_level: float = 50.0
    boredom: float = 0.0
    loneliness: float = 0.0
    social_desire: float = 0.0
    proactive_willingness: float = 0.0
    channel_annoyance: float = 0.0
    consecutive_replies: int = 0
    silence_seconds: float = 0.0
    flow_phase: str = "standby"
    flow_phase_label: str = "待命"
    can_reply: bool = True
    pending_active: bool = False
    pending_progress: float = 0.0
    # 观看状态（来自 watch_state_machine）
    watch_level: str = "peek"
    watch_level_label: str = "愁一眼"
    # 摸鱼值（来自 metabolism_engine）
    loafing_level: float = 0.0
    # 代谢约束（来自 metabolism_engine.behavior_constraint_summary）
    metabolism_reply_hint: str = ""
    metabolism_gate_open: bool = True
    metabolism_is_perfunctory: bool = False
    # 夜间节律（来自 night_cycle_system.night_behavior_summary）
    night_phase_detail: str = ""
    night_can_reply: bool = True
    night_sleep_debt: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chat_energy_ratio": round(self.chat_energy_ratio, 3),
            "thinking_energy_ratio": round(self.thinking_energy_ratio, 3),
            "activity_level": round(self.activity_level, 1),
            "boredom": round(self.boredom, 2),
            "loneliness": round(self.loneliness, 2),
            "social_desire": round(self.social_desire, 2),
            "proactive_willingness": round(self.proactive_willingness, 2),
            "channel_annoyance": round(self.channel_annoyance, 2),
            "consecutive_replies": self.consecutive_replies,
            "silence_seconds": round(self.silence_seconds, 1),
            "flow_phase": self.flow_phase,
            "flow_phase_label": self.flow_phase_label,
            "can_reply": self.can_reply,
            "pending_active": self.pending_active,
            "pending_progress": round(self.pending_progress, 2),
            "watch_level": self.watch_level,
            "watch_level_label": self.watch_level_label,
            "loafing_level": round(self.loafing_level, 2),
            "metabolism_reply_hint": self.metabolism_reply_hint,
            "metabolism_gate_open": self.metabolism_gate_open,
            "metabolism_is_perfunctory": self.metabolism_is_perfunctory,
            "night_phase_detail": self.night_phase_detail,
            "night_can_reply": self.night_can_reply,
            "night_sleep_debt": round(self.night_sleep_debt, 3),
        }


@dataclass
class ImpressionGauge:
    """当前对象印象栏"""

    user_id: str = ""
    user_name: str = ""
    relationship_level: int = 2
    custom_label: str = ""
    trend_direction: str = "稳定"
    social_value: float = 0.0
    trust_value: float = 0.0
    annoyance_value: float = 0.0
    affection: float = 0.0
    trauma_score: float = 0.0
    psychological_pressure: float = 0.0
    impression_style: str = ""
    mood: str = "平静"
    interaction_count: int = 0
    profile_summary: str = ""
    # 最近话题（来自 impression_evolution_hub / target_user）
    recent_topics: List[str] = field(default_factory=list)
    emotional_tendency: str = ""
    # 机器人相关性（来自 self_reference_detector）
    is_discussing_bot: bool = False
    is_high_activity_user: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "relationship_level": self.relationship_level,
            "custom_label": self.custom_label,
            "trend_direction": self.trend_direction,
            "social_value": round(self.social_value, 2),
            "trust_value": round(self.trust_value, 2),
            "annoyance_value": round(self.annoyance_value, 2),
            "affection": round(self.affection, 2),
            "trauma_score": round(self.trauma_score, 2),
            "psychological_pressure": round(self.psychological_pressure, 2),
            "impression_style": self.impression_style,
            "mood": self.mood,
            "interaction_count": self.interaction_count,
            "profile_summary": self.profile_summary,
            "recent_topics": self.recent_topics[:5],
            "emotional_tendency": self.emotional_tendency,
            "is_discussing_bot": self.is_discussing_bot,
            "is_high_activity_user": self.is_high_activity_user,
        }


@dataclass
class DecisionGauge:
    """动作裁定栏"""

    behavior_category: str = "neutral"
    behavior_severity: float = 0.0
    behavior_reason: str = ""
    repeated_topic_pressure: float = 0.0
    last_user_intent: str = ""
    is_new_user: bool = False
    # 场景与话题上下文（来自 group_scene_state / understanding_gate）
    scene_type: str = ""
    scene_type_label: str = ""
    topic_attribution: str = ""
    comprehension_level: str = ""
    deep_analysis_triggered: bool = False
    # 主观存在判定（来自 subjective_presence_core）
    participation_verdict: str = ""
    self_mood_label: str = ""
    topic_interest_score: float = 0.0
    want_learn_first: bool = False
    # 自引用强度（来自 self_reference_detector）
    self_ref_strength: float = 0.0
    self_ref_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "behavior_category": self.behavior_category,
            "behavior_severity": round(self.behavior_severity, 2),
            "behavior_reason": self.behavior_reason,
            "repeated_topic_pressure": round(self.repeated_topic_pressure, 2),
            "last_user_intent": self.last_user_intent,
            "is_new_user": self.is_new_user,
            "scene_type": self.scene_type,
            "scene_type_label": self.scene_type_label,
            "topic_attribution": self.topic_attribution,
            "comprehension_level": self.comprehension_level,
            "deep_analysis_triggered": self.deep_analysis_triggered,
            "participation_verdict": self.participation_verdict,
            "self_mood_label": self.self_mood_label,
            "topic_interest_score": round(self.topic_interest_score, 3),
            "want_learn_first": self.want_learn_first,
            "self_ref_strength": round(self.self_ref_strength, 3),
            "self_ref_type": self.self_ref_type,
        }


@dataclass
class SceneGauge:
    """群聊环境栏"""

    channel_id: str = ""
    is_group: bool = True
    active_user_count: int = 0
    current_topics: List[str] = field(default_factory=list)
    session_phase: str = ""
    last_topic: str = ""
    vexation: float = 0.0
    weariness: float = 0.0
    # 群氛围（来自 group_scene_state）
    atmosphere: str = ""
    atmosphere_label: str = ""
    # 多线程（来自 group_scene_state）
    has_multi_thread: bool = False
    thread_count: int = 0
    # 群体行为模式（来自 group_pattern_detector）
    active_pattern: str = ""
    active_pattern_label: str = ""
    pattern_confidence: float = 0.0
    # 机器人场景姿态
    bot_scene_posture: str = ""
    # 消息类型分布（来自 route_summary）
    msg_type_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "is_group": self.is_group,
            "active_user_count": self.active_user_count,
            "current_topics": self.current_topics[:5],
            "session_phase": self.session_phase,
            "last_topic": self.last_topic,
            "vexation": round(self.vexation, 2),
            "weariness": round(self.weariness, 2),
            "atmosphere": self.atmosphere,
            "atmosphere_label": self.atmosphere_label,
            "has_multi_thread": self.has_multi_thread,
            "thread_count": self.thread_count,
            "active_pattern": self.active_pattern,
            "active_pattern_label": self.active_pattern_label,
            "pattern_confidence": round(self.pattern_confidence, 3),
            "bot_scene_posture": self.bot_scene_posture,
            "msg_type_distribution": dict(self.msg_type_distribution),
        }


@dataclass
class ResultGauge:
    """结果汇总栏"""

    action_taken: bool = False
    reply_text: str = ""
    quote_message: bool = False
    action_type: str = ""
    reasoning: str = ""
    taken_time: float = 0.0
    # 本轮学习/缓存/识别结果
    learned_new_info: bool = False
    learned_content_hint: str = ""
    cached_image_semantic: bool = False
    detected_pattern: bool = False
    detected_pattern_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_taken": self.action_taken,
            "reply_text": self.reply_text[:200] if self.reply_text else "",
            "quote_message": self.quote_message,
            "action_type": self.action_type,
            "reasoning": self.reasoning[:120] if self.reasoning else "",
            "taken_time": round(self.taken_time, 2),
            "learned_new_info": self.learned_new_info,
            "learned_content_hint": self.learned_content_hint[:50],
            "cached_image_semantic": self.cached_image_semantic,
            "detected_pattern": self.detected_pattern,
            "detected_pattern_name": self.detected_pattern_name,
        }


@dataclass
class PanelSnapshot:
    """五层状态栏聚合快照"""

    vitals: VitalsGauge = field(default_factory=VitalsGauge)
    impression: ImpressionGauge = field(default_factory=ImpressionGauge)
    decision: DecisionGauge = field(default_factory=DecisionGauge)
    scene: SceneGauge = field(default_factory=SceneGauge)
    result: ResultGauge = field(default_factory=ResultGauge)
    assembled_at: float = field(default_factory=time.time)

    def to_full_dict(self) -> Dict[str, Any]:
        return {
            "vitals": self.vitals.to_dict(),
            "impression": self.impression.to_dict(),
            "decision": self.decision.to_dict(),
            "scene": self.scene.to_dict(),
            "result": self.result.to_dict(),
            "assembled_at": self.assembled_at,
        }


# ---------------------------------------------------------------------------
#  面板适配器核心
# ---------------------------------------------------------------------------


class PanelStateAdapter:
    """面板状态适配器
    从 WorldSnapshot 一次性装配五层状态栏，
    消费方统一通过本适配器读取，不再散读底层子系统。
    """

    def assemble(
        self,
        snapshot,
        module_context: Optional[Dict[str, Any]] = None,
    ) -> PanelSnapshot:
        """从 WorldSnapshot 装配完整面板快照"""
        panel = PanelSnapshot(assembled_at=time.time())
        self._fill_vitals(panel.vitals, snapshot)
        self._fill_impression(panel.impression, snapshot)
        self._fill_decision(panel.decision, snapshot)
        self._fill_scene(panel.scene, snapshot)
        if module_context:
            self.enrich_from_modules(panel, **module_context)
        return panel

    def assemble_with_result(
        self,
        snapshot,
        action_result: Optional[Dict[str, Any]] = None,
        module_context: Optional[Dict[str, Any]] = None,
    ) -> PanelSnapshot:
        """装配面板并注入行为结果"""
        panel = self.assemble(snapshot, module_context=module_context)
        if action_result:
            self._fill_result(panel.result, action_result)
        return panel

    def extract_vitals_dict(self, snapshot) -> Dict[str, Any]:
        """仅提取体征栏字典"""
        gauge = VitalsGauge()
        self._fill_vitals(gauge, snapshot)
        return gauge.to_dict()

    def extract_impression_dict(self, snapshot) -> Dict[str, Any]:
        """仅提取印象栏字典"""
        gauge = ImpressionGauge()
        self._fill_impression(gauge, snapshot)
        return gauge.to_dict()

    def extract_relation_dict(self, snapshot) -> Dict[str, Any]:
        """兼容旧版 to_relation_dict 格式"""
        return snapshot.to_relation_dict()

    # ────────────────── 内部装配方法 ──────────────────

    def _fill_vitals(self, gauge: VitalsGauge, snapshot) -> None:
        subj = snapshot.subject
        res = snapshot.self_resources
        gauge.chat_energy_ratio = res.chat_ratio()
        gauge.thinking_energy_ratio = res.thinking_ratio()
        gauge.activity_level = res.activity_level
        gauge.boredom = res.boredom
        gauge.loneliness = res.loneliness
        gauge.social_desire = res.social_desire
        gauge.proactive_willingness = res.proactive_willingness
        gauge.channel_annoyance = res.channel_annoyance
        gauge.consecutive_replies = res.consecutive_replies
        gauge.silence_seconds = res.silence_seconds
        gauge.flow_phase = subj.flow_phase
        gauge.flow_phase_label = subj.flow_phase_label
        gauge.can_reply = subj.can_reply
        gauge.pending_active = subj.pending_active
        gauge.pending_progress = subj.pending_progress
        # 从 SubjectState 扩展字段填充（不再需要 enrich 补充）
        gauge.watch_level = subj.watch_level
        gauge.watch_level_label = subj.watch_level_label
        gauge.loafing_level = subj.loafing_level

    def _fill_impression(self, gauge: ImpressionGauge, snapshot) -> None:
        usr = snapshot.target_user
        gauge.user_id = str(
            getattr(usr, "group_friend_id", "")
            or getattr(usr, "user_id", "")
            or ""
        )
        gauge.user_name = str(
            getattr(usr, "group_friend_name", "")
            or getattr(usr, "user_name", "")
            or ""
        )
        gauge.relationship_level = usr.relationship_level
        gauge.custom_label = usr.custom_label
        gauge.trend_direction = usr.trend_direction
        gauge.social_value = usr.social_value
        gauge.trust_value = usr.trust_value
        gauge.annoyance_value = usr.annoyance_value
        gauge.affection = usr.affection
        gauge.trauma_score = usr.trauma_score
        gauge.psychological_pressure = usr.psychological_pressure
        gauge.impression_style = usr.impression_style
        gauge.mood = usr.mood
        gauge.interaction_count = usr.interaction_count
        gauge.profile_summary = usr.profile_summary()

    def _fill_decision(self, gauge: DecisionGauge, snapshot) -> None:
        gauge.behavior_category = snapshot.behavior_category
        gauge.behavior_severity = snapshot.behavior_severity
        gauge.behavior_reason = snapshot.behavior_reason
        gauge.repeated_topic_pressure = snapshot.repeated_topic_pressure
        gauge.last_user_intent = snapshot.last_user_intent
        gauge.is_new_user = snapshot.is_new_user

    def _fill_scene(self, gauge: SceneGauge, snapshot) -> None:
        sc = snapshot.scene
        gauge.channel_id = sc.channel_id
        gauge.is_group = sc.is_group
        gauge.active_user_count = len(sc.active_users)
        gauge.current_topics = list(sc.current_topics)
        gauge.session_phase = sc.session_phase
        gauge.last_topic = sc.last_topic
        gauge.vexation = sc.vexation
        gauge.weariness = sc.weariness

    def _fill_result(
        self, gauge: ResultGauge, action_result: Dict[str, Any]
    ) -> None:
        action_info = action_result.get("loop_action_info", {})
        plan_info = action_result.get("loop_plan_info", {})
        gauge.action_taken = bool(action_info.get("action_taken", False))
        gauge.reply_text = str(action_info.get("reply_text", "") or "")
        gauge.quote_message = bool(action_info.get("quote_message", False))
        gauge.taken_time = float(action_info.get("taken_time", 0.0) or 0.0)
        ar = plan_info.get("action_result", {})
        if isinstance(ar, dict):
            gauge.action_type = str(ar.get("action_type", "") or "")
            gauge.reasoning = str(ar.get("reasoning", "") or "")

    # ────────────────── 模块增强注入 ──────────────────

    def enrich_from_modules(self, panel: PanelSnapshot, **kwargs) -> None:
        """从各核心模块的计算结果补充面板中无法直接从 WorldSnapshot 获取的字段。
        调用方通过关键字参数传入各模块结果字典，本方法按需写入对应 gauge。
        参数命名约定：
            watch_info   — watch_state_machine 输出
            metabolism   — metabolism_engine 输出
            impression   — impression_evolution_hub 输出
            self_ref     — self_reference_detector 输出
            scene_state  — group_scene_state 输出
            pattern      — group_pattern_detector 输出
            understanding— understanding_gate 输出
            learning     — learning_hub 输出
            budgeter     — multimodal_budgeter 输出
        """
        self._enrich_vitals(panel.vitals, kwargs)
        self._enrich_impression(panel.impression, kwargs)
        self._enrich_decision(panel.decision, kwargs)
        self._enrich_scene(panel.scene, kwargs)
        self._enrich_result(panel.result, kwargs)

    def _enrich_vitals(self, gauge: VitalsGauge, ctx: Dict[str, Any]) -> None:
        watch = ctx.get("watch_info")
        if isinstance(watch, dict):
            gauge.watch_level = str(watch.get("level", gauge.watch_level))
            gauge.watch_level_label = str(
                watch.get("label", gauge.watch_level_label)
            )
        metab = ctx.get("metabolism")
        if isinstance(metab, dict):
            gauge.loafing_level = float(
                metab.get("loafing", gauge.loafing_level)
            )
            gauge.metabolism_reply_hint = str(
                metab.get("reply_length_hint", "")
            )
            gauge.metabolism_gate_open = bool(
                metab.get("energy_gate_open", True)
            )
            gauge.metabolism_is_perfunctory = bool(
                metab.get("is_perfunctory", False)
            )
        night = ctx.get("night")
        if isinstance(night, dict):
            gauge.night_phase_detail = str(night.get("phase_label", ""))
            gauge.night_can_reply = bool(night.get("can_reply", True))
            gauge.night_sleep_debt = float(night.get("sleep_debt", 0.0))

    def _enrich_impression(
        self, gauge: ImpressionGauge, ctx: Dict[str, Any]
    ) -> None:
        imp = ctx.get("impression")
        if isinstance(imp, dict):
            topics = imp.get("recent_topics")
            if not isinstance(topics, list):
                topics = imp.get("ongoing_topics")
            if isinstance(topics, list):
                gauge.recent_topics = [str(t) for t in topics[:10]]
            tendency = imp.get("emotional_tendency") or imp.get(
                "overall_feeling"
            )
            if tendency:
                gauge.emotional_tendency = str(tendency)
            if "is_high_activity" in imp:
                gauge.is_high_activity_user = bool(imp["is_high_activity"])
            if "is_high_activity_user" in imp:
                gauge.is_high_activity_user = bool(
                    imp["is_high_activity_user"]
                )
        sref = ctx.get("self_ref")
        if isinstance(sref, dict):
            gauge.is_discussing_bot = bool(
                sref.get("is_discussing_bot", False)
            )
            if "strength" in sref:
                gauge.is_discussing_bot = (
                    gauge.is_discussing_bot or float(sref["strength"]) > 0.3
                )

    def _enrich_decision(
        self, gauge: DecisionGauge, ctx: Dict[str, Any]
    ) -> None:
        ss = ctx.get("scene_state")
        if isinstance(ss, dict):
            gauge.scene_type = str(ss.get("atmosphere", ""))
            gauge.scene_type_label = str(ss.get("atmosphere_label", ""))
            if "topic_attribution" in ss:
                gauge.topic_attribution = str(ss["topic_attribution"])
        ug = ctx.get("understanding")
        if isinstance(ug, dict):
            gauge.comprehension_level = str(ug.get("comprehension_level", ""))
            gauge.deep_analysis_triggered = bool(
                ug.get("deep_analysis", False)
            )
        # 主观存在判定
        pres = ctx.get("presence")
        if isinstance(pres, dict):
            gauge.participation_verdict = str(
                pres.get("participation_verdict", "")
            )
            gauge.self_mood_label = str(pres.get("mood_label", ""))
            gauge.topic_interest_score = float(
                pres.get("topic_interest_score", 0.0)
            )
            gauge.want_learn_first = bool(pres.get("want_learn_first", False))
        # 自引用强度
        sref = ctx.get("self_ref")
        if isinstance(sref, dict):
            gauge.self_ref_strength = float(sref.get("strength", 0.0))
            gauge.self_ref_type = str(sref.get("ref_type", ""))

    def _enrich_scene(self, gauge: SceneGauge, ctx: Dict[str, Any]) -> None:
        ss = ctx.get("scene_state")
        if isinstance(ss, dict):
            gauge.atmosphere = str(ss.get("atmosphere", ""))
            gauge.atmosphere_label = str(ss.get("atmosphere_label", ""))
            gauge.has_multi_thread = bool(ss.get("multi_thread", False))
            gauge.thread_count = int(ss.get("thread_count", 0))
            if "bot_posture" in ss:
                gauge.bot_scene_posture = str(ss["bot_posture"])
        pat = ctx.get("pattern")
        if isinstance(pat, dict):
            gauge.active_pattern = str(pat.get("pattern", ""))
            gauge.active_pattern_label = str(pat.get("pattern_label", ""))
            gauge.pattern_confidence = float(pat.get("confidence", 0.0))
        route_sum = ctx.get("route_summary")
        if isinstance(route_sum, dict):
            gauge.msg_type_distribution = dict(route_sum)

    def _enrich_result(self, gauge: ResultGauge, ctx: Dict[str, Any]) -> None:
        learn = ctx.get("learning")
        if isinstance(learn, dict):
            gauge.learned_new_info = bool(learn.get("learned", False))
            gauge.learned_content_hint = str(learn.get("hint", ""))[:50]
        budget = ctx.get("budgeter")
        if isinstance(budget, dict):
            gauge.cached_image_semantic = bool(
                budget.get("cached_image", False)
            )
        pat = ctx.get("pattern")
        if isinstance(pat, dict):
            gauge.detected_pattern = bool(pat.get("detected", False))
            gauge.detected_pattern_name = str(pat.get("pattern", ""))


def get_panel_adapter() -> PanelStateAdapter:
    """获取面板状态适配器单例"""
    global _adapter_singleton
    if _adapter_singleton is None:
        _adapter_singleton = PanelStateAdapter()
    return _adapter_singleton

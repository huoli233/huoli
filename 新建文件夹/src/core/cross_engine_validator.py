import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from src.common.logger import get_logger

logger = get_logger("engine_validation")

_validation_singleton = None

# ═══════════════════════════════════════════════════
#  引擎注册与数据流枚举
# ═══════════════════════════════════════════════════


class EngineId(Enum):
    """所有已注册引擎ID"""

    ACFN = "multi_factor_decision_engine"
    SCME = "state_coupling_matrix"
    IMPRESSION = "individual_impression_engine"
    SKILL_LIFECYCLE = "skill_lifecycle_manager"
    DEEP_VISIBILITY = "deep_visibility_scorer"
    FRESHNESS_DECAY = "freshness_decay_engine"
    GOSSIP_RITUAL = "gossip_ritual_strategy"
    ADAPTIVE_PIPELINE = "adaptive_update_pipeline"
    MEMORY_GOVERNANCE = "memory_governance_engine"
    SAFETY_FUSION = "safety_boundary_fusion"
    PLANNER_INJECTION = "planner_prompt_injection"
    NIGHT_CYCLE = "night_cycle_system"
    METABOLISM = "metabolism_engine"
    PRESENCE_CORE = "presence_core"
    EMOTION_DRIVEN = "emotion_driven_core"
    SOCIAL_AFFECT = "social_affect_fuser"
    GROUP_PATTERN = "group_pattern_detector"
    INNER_NARRATION = "inner_narration_planner"
    MULTIMODAL_BUDGETER = "multimodal_budgeter"
    SUBJECTIVE_TIME = "subjective_time_stream"
    SELF_REFERENCE = "self_reference_detector"
    VISIBILITY_GATE = "visibility_gate"
    UNDERSTANDING_GATE = "understanding_gate"

    def label(self) -> str:
        return {
            "multi_factor_decision_engine": "ACFN多因子融合",
            "state_coupling_matrix": "状态耦合矩阵",
            "individual_impression_engine": "个体印象引擎",
            "skill_lifecycle_manager": "技能生命周期",
            "deep_visibility_scorer": "深度可见性",
            "freshness_decay_engine": "新鲜度衰减",
            "gossip_ritual_strategy": "吃瓜策略",
            "adaptive_update_pipeline": "自适应管道",
            "memory_governance_engine": "记忆治理",
            "safety_boundary_fusion": "安全边界",
            "planner_prompt_injection": "Planner注入",
            "night_cycle_system": "昼夜周期",
            "metabolism_engine": "代谢引擎",
            "presence_core": "存在感核心",
            "emotion_driven_core": "情感驱动核心",
            "social_affect_fuser": "社交情感融合",
            "group_pattern_detector": "群体模式检测",
            "inner_narration_planner": "内心独白规划",
            "multimodal_budgeter": "多模态预算",
            "subjective_time_stream": "主观时间流",
            "self_reference_detector": "自我指称检测",
            "visibility_gate": "可见性门控",
            "understanding_gate": "理解门控",
        }.get(self.value, self.value)


class ConsumptionStatus(Enum):
    """输出消费状态"""

    PRODUCED = "produced"
    CONSUMED_BY_ARBITER = "consumed_by_arbiter"
    CONSUMED_BY_PLANNER = "consumed_by_planner"
    CONSUMED_BY_STYLE = "consumed_by_style"
    CONSUMED_BY_NARRATION = "consumed_by_narration"
    CONSUMED_BY_MEMORY = "consumed_by_memory"
    CONSUMED_BY_PROACTIVE = "consumed_by_proactive"
    CONSUMED_BY_SAFETY = "consumed_by_safety"
    STALE = "stale"
    ORPHANED = "orphaned"

    def label(self) -> str:
        return {
            "produced": "已产出",
            "consumed_by_arbiter": "被仲裁器消费",
            "consumed_by_planner": "被Planner消费",
            "consumed_by_style": "被风格层消费",
            "consumed_by_narration": "被独白层消费",
            "consumed_by_memory": "被记忆层消费",
            "consumed_by_proactive": "被主动层消费",
            "consumed_by_safety": "被安全层消费",
            "stale": "已过期",
            "orphaned": "孤岛(未消费)",
        }.get(self.value, "")


class DataFlowEdge:
    """数据流边——从引擎A的某个输出字段到引擎B/主链的某个输入"""

    def __init__(
        self,
        source: EngineId,
        source_field: str,
        target: str,
        target_field: str,
    ):
        self.source = source
        self.source_field = source_field
        self.target = target
        self.target_field = target_field

    def key(self) -> str:
        return f"{
            self.source.value}.{
            self.source_field}→{
            self.target}.{
                self.target_field}"


@dataclass
class EngineOutputRecord:
    """单条引擎产出记录"""

    engine_id: EngineId
    output_key: str
    produced_at: float = field(default_factory=time.time)
    consumed_at: float = 0.0
    status: ConsumptionStatus = ConsumptionStatus.PRODUCED
    consumer: str = ""
    consumer_count: int = 0
    data_hash: int = 0
    size_bytes: int = 0
    is_critical: bool = False
    ttl_sec: float = 300.0

    def is_expired(self) -> bool:
        if self.status in (
            ConsumptionStatus.CONSUMED_BY_ARBITER,
            ConsumptionStatus.CONSUMED_BY_PLANNER,
            ConsumptionStatus.CONSUMED_BY_SAFETY,
        ):
            return (time.time() - self.consumed_at) > self.ttl_sec * 3
        return (time.time() - self.produced_at) > self.ttl_sec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine_id.label(),
            "output": self.output_key,
            "status": self.status.label(),
            "consumer": self.consumer,
            "age_sec": round(time.time() - self.produced_at, 1),
            "critical": self.is_critical,
        }


@dataclass
class ValidationReport:
    """验证报告"""

    total_engines_registered: int = 0
    total_outputs_produced: int = 0
    total_consumed: int = 0
    orphaned_outputs: int = 0
    stale_outputs: int = 0
    critical_orphans: List[EngineOutputRecord] = field(default_factory=list)
    engine_health: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    flow_edges_verified: int = 0
    flow_edges_broken: int = 0
    broken_edges: List[str] = field(default_factory=list)
    overall_score: float = 1.0
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engines": self.total_engines_registered,
            "outputs": self.total_outputs_produced,
            "consumed": self.total_consumed,
            "orphaned": self.orphaned_outputs,
            "stale": self.stale_outputs,
            "critical_orphans": len(self.critical_orphans),
            "flow_verified": self.flow_edges_verified,
            "flow_broken": self.flow_edges_broken,
            "score": round(self.overall_score, 3),
            "warnings": self.warnings[:8],
            "recommendations": self.recommendations[:5],
        }


# 预定义的数据流边（从文档架构图推导）
_PREDEFINED_EDGES: List[Tuple[EngineId, str, str, str]] = [
    (EngineId.METABOLISM, "energy_ratio", "ACFN", "energy"),
    (EngineId.METABOLISM, "boredom_level", "SCME", "boredom"),
    (EngineId.METABOLISM, "loafing_level", "SCME", "loafing"),
    (EngineId.NIGHT_CYCLE, "phase", "PLANNER_INJECTION", "night_phase"),
    (EngineId.NIGHT_CYCLE, "pressure_breakdown", "ACFN", "overnight_pressure"),
    (
        EngineId.PRESENCE_CORE,
        "watch_state_rank",
        "DEEP_VISIBILITY",
        "watch_state_rank",
    ),
    (EngineId.PRESENCE_CORE, "watch_state", "GAP-F_TRANSITION", "watch_input"),
    (
        EngineId.DEEP_VISIBILITY,
        "modulated_score",
        "ARBITER_CHAIN",
        "visibility_weight",
    ),
    (
        EngineId.DEEP_VISIBILITY,
        "enter_understanding",
        "MEMORY_GOVERNANCE",
        "should_remember",
    ),
    (
        EngineId.FRESHNESS_DECAY,
        "current_freshness",
        "ACFN",
        "freshness_factor",
    ),
    (
        EngineId.FRESHNESS_DECAY,
        "recall_decisions",
        "PROACTIVE_ENGINE",
        "recall_hints",
    ),
    (
        EngineId.GOSSIP_RITUAL,
        "verdict.posture",
        "INNER_NARRATION",
        "event_posture",
    ),
    (
        EngineId.GOSSIP_RITUAL,
        "verdict.reply_hint",
        "PLANNER_INJECTION",
        "strategy_hint",
    ),
    (
        EngineId.IMPRESSION,
        "behavior_params",
        "STYLE_CONTEXT",
        "per_user_style_mod",
    ),
    (EngineId.SKILL_LIFECYCLE, "cost_modifier", "ACFN", "skill_cost_factor"),
    (
        EngineId.SAFETY_FUSION,
        "activation_bar_delta",
        "ARBITER_CHAIN",
        "safety_bar_addon",
    ),
    (
        EngineId.SAFETY_FUSION,
        "fused_score_delta",
        "ARBITER_CHAIN",
        "safety_score_mod",
    ),
    (
        EngineId.MEMORY_GOVERNANCE,
        "window_text",
        "PLANNER_INJECTION",
        "context_memory",
    ),
    (EngineId.PLANNER_INJECTION, "full_prompt", "LLM_CALL", "system_prompt"),
    (EngineId.EMOTION_DRIVEN, "mood", "ACFN", "emotion_mood"),
    (
        EngineId.EMOTION_DRIVEN,
        "proactive_willingness",
        "PROACTIVE_ENGINE",
        "drive_signal",
    ),
    (
        EngineId.SOCIAL_AFFECT,
        "social_score",
        "DEEP_VISIBILITY",
        "user_affection",
    ),
    (
        EngineId.GROUP_PATTERN,
        "pattern_evidence",
        "GOSSIP_RITUAL",
        "event_context",
    ),
    (
        EngineId.SELF_REFERENCE,
        "ref_list",
        "DEEP_VISIBILITY",
        "self_ref_signals",
    ),
    (
        EngineId.INNER_NARRATION,
        "narration_text",
        "PLANNER_INJECTION",
        "narration_hint",
    ),
    (
        EngineId.MULTIMODAL_BUDGETER,
        "budget_decision",
        "SEMANTIC_BRIDGE",
        "media_analysis_request",
    ),
]


class CrossEngineValidator:
    """跨引擎联动验证器
    追踪所有引擎的产出记录，验证：
    1. 每个引擎是否正常产出（心跳检测）
    2. 每条产出是否被下游消费（孤岛检测）
    3. 预定义数据流边是否存在断裂（流完整性）
    4. 关键路径上的产出是否超时未消费（关键孤岛告警）
    5. 整体系统健康度评分（0-1，<0.7需关注）
    """

    def __init__(self):
        self._outputs: List[EngineOutputRecord] = []
        self._max_outputs = 500
        self._registered_engines: Set[EngineId] = set()
        self._edges: List[DataFlowEdge] = []
        self._edge_status: Dict[str, bool] = {}
        self._consumer_log: List[Tuple[float, str, str, str]] = []
        self._max_consumer_log = 300
        self._last_report: Optional[ValidationReport] = None
        self._build_predefined_edges()

    # ═════════════════ 注册与产出 ═════════════════

    def register_engine(self, engine_id: EngineId) -> None:
        self._registered_engines.add(engine_id)
        logger.debug(f"[联动验证] 引擎注册: {engine_id.label()}")

    def record_output(
        self,
        engine_id: EngineId,
        output_key: str,
        *,
        data_size: int = 0,
        is_critical: bool = False,
        ttl_sec: float = 300.0,
    ) -> EngineOutputRecord:
        record = EngineOutputRecord(
            engine_id=engine_id,
            output_key=output_key,
            data_size=data_size,
            is_critical=is_critical,
            ttl_sec=ttl_sec,
        )
        self._outputs.append(record)
        if len(self._outputs) > self._max_outputs:
            self._outputs = self._outputs[-self._max_outputs:]
        return record

    def mark_consumed(
        self,
        engine_id: EngineId,
        output_key: str,
        consumer: str,
        status: ConsumptionStatus = ConsumptionStatus.CONSUMED_BY_ARBITER,
    ) -> bool:
        for rec in reversed(self._outputs):
            if (
                rec.engine_id == engine_id
                and rec.output_key == output_key
                and rec.status == ConsumptionStatus.PRODUCED
            ):
                rec.status = status
                rec.consumer = consumer
                rec.consumed_at = time.time()
                rec.consumer_count += 1
                self._consumer_log.append(
                    (
                        time.time(),
                        f"{engine_id.value}:{output_key}",
                        consumer,
                        status.value,
                    )
                )
                if len(self._consumer_log) > self._max_consumer_log:
                    self._consumer_log = self._consumer_log[
                        -self._max_consumer_log:
                    ]
                return True
        return False

    # ═════════════════ 验证入口 ═════════════════

    def validate(self, full_check: bool = False) -> ValidationReport:
        now = time.time()
        report = ValidationReport()
        report.total_engines_registered = len(self._registered_engines)
        active = [r for r in self._outputs if not r.is_expired()]
        stale = [r for r in self._outputs if r.is_expired()]
        consumed = [
            r
            for r in active
            if r.status != ConsumptionStatus.PRODUCED
            and r.status != ConsumptionStatus.ORPHANED
        ]
        orphaned = [
            r
            for r in active
            if r.status == ConsumptionStatus.PRODUCED
            or r.status == ConsumptionStatus.ORPHANED
        ]
        critical_orphaned = [r for r in orphaned if r.is_critical]
        report.total_outputs_produced = len(active)
        report.total_consumed = len(consumed)
        report.orphaned_outputs = len(orphaned)
        report.stale_outputs = len(stale)
        report.critical_orphans = critical_orphaned
        engine_stats: Dict[str, Dict[str, Any]] = {}
        for eng in self._registered_engines:
            eng_outputs = [r for r in active if r.engine_id == eng]
            eng_consumed = [
                r
                for r in eng_outputs
                if r.status
                not in (ConsumptionStatus.PRODUCED, ConsumptionStatus.ORPHANED)
            ]
            eng_orphaned = [
                r
                for r in eng_outputs
                if r.status
                in (ConsumptionStatus.PRODUCED, ConsumptionStatus.ORPHANED)
            ]
            health = 1.0
            if eng_outputs:
                health = len(eng_consumed) / len(eng_outputs)
            engine_stats[eng.value] = {
                "label": eng.label(),
                "total_outputs": len(eng_outputs),
                "consumed": len(eng_consumed),
                "orphaned": len(eng_orphaned),
                "health": round(health, 3),
                "last_output_age": min(
                    (now - r.produced_at for r in eng_outputs), default=9999
                ),
            }
        report.engine_health = engine_stats
        verified = 0
        broken = 0
        broken_list: List[str] = []
        for edge in self._edges:
            ek = edge.key()
            found = any(
                r.engine_id == edge.source
                and r.output_key == edge.source_field
                and r.status not in (ConsumptionStatus.ORPHANED,)
                for r in active
            )
            if found:
                verified += 1
                self._edge_status[ek] = True
            else:
                broken += 1
                self._edge_status[ek] = False
                broken_list.append(ek)
        report.flow_edges_verified = verified
        report.flow_edges_broken = broken
        report.broken_edges = broken_list
        score = 1.0
        n_eng = max(1, report.total_engines_registered)
        expected_min_engines = 15
        if report.total_engines_registered < expected_min_engines:
            score *= (
                0.7
                + (report.total_engines_registered / expected_min_engines)
                * 0.3
            )
            report.warnings.append(
                f"仅{report.total_engines_registers}个引擎注册，预期≥{expected_min_engines}"
            )
        if active:
            consume_ratio = report.total_consumed / max(
                1, report.total_outputs_produced
            )
            score *= 0.6 + consume_ratio * 0.4
        if critical_orphaned:
            score *= max(0.3, 1.0 - len(critical_orphaned) * 0.15)
            report.warnings.append(
                f"{len(critical_orphaned)}条关键产出未被消费"
            )
        if broken > 0:
            edge_penalty = broken / max(1, verified + broken)
            score *= 1.0 - edge_penalty * 0.3
        report.overall_score = max(0.0, min(1.0, score))
        recs = []
        if report.orphaned_outputs > 3:
            recs.append(f"有{report.orphaned_outputs}条孤岛产出")
        if report.flow_edges_broken > 2:
            recs.append(f"有{report.flow_edges_broken}条数据流边断裂")
        if report.overall_score < 0.6:
            recs.append("系统健康度低于0.6，建议排查")
        if critical_orphaned:
            for co in critical_orphaned[:3]:
                recs.append(
                    f"关键孤岛: {co.engine_id.label()}/{co.output_key}"
                )
        report.recommendations = recs
        report.validated_at = now
        self._last_report = report
        logger.info(
            f"[联动验证] 完成: score={
                report.overall_score:.3f}, "
            f"引擎={
                report.total_engines_registered}, 产出={
                report.total_outputs_produced}, "
            f"消费={
                    report.total_consumed}, 孤岛={
                        report.orphaned_outputs}, "
            f"流边={verified}/{
                verified + broken}"
        )
        return report

    def get_quick_status(self) -> Dict[str, Any]:
        if self._last_report is None:
            return {"status": "尚未运行验证"}
        r = self._last_report
        return {
            "score": round(r.overall_score, 3),
            "engines": r.total_engines_registered,
            "orphaned": r.orphaned_outputs,
            "broken_flows": r.flow_edges_broken,
            "critical_alerts": len(r.critical_orphans),
            "healthy": r.overall_score >= 0.7,
        }

    # ────────────────── 内部方法 ──────────────────

    def _build_predefined_edges(self) -> None:
        for src_id, src_field, tgt, tgt_field in _PREDEFINED_EDGES:
            edge = DataFlowEdge(src_id, src_field, tgt, tgt_field)
            self._edges.append(edge)
            self._edge_status[edge.key()] = False
        logger.info(f"[联动验证] 预定义{len(self._edges)}条数据流边")


def get_cross_engine_validator() -> CrossEngineValidator:
    global _validation_singleton
    if _validation_singleton is None:
        _validation_singleton = CrossEngineValidator()
    return _validation_singleton

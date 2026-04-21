import time
import threading
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
    """产出流转状态：标记每条引擎输出被哪个下游环节接走"""
    WAITING = "waiting"
    BY_DECISION = "by_decision"
    BY_PLANNER = "by_planner"
    BY_STYLE = "by_style"
    BY_NARRATION = "by_narration"
    BY_MEMORY = "by_memory"
    BY_PROACTIVE = "by_proactive"
    BY_SAFETY = "by_safety"
    EXPIRED = "expired"
    ORPHAN = "orphan"

    def label(self) -> str:
        return {
            "waiting": "等待接收",
            "by_decision": "已交付→决策链",
            "by_planner": "已交付→规划器",
            "by_style": "已交付→风格层",
            "by_narration": "已交付→独白层",
            "by_memory": "已交付→记忆层",
            "by_proactive": "已交付→主动层",
            "by_safety": "已交付→安全层",
            "expired": "已过期",
            "orphan": "未被接收",
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
    status: ConsumptionStatus = ConsumptionStatus.WAITING
    consumer: str = ""
    consumer_count: int = 0
    data_hash: int = 0
    size_bytes: int = 0
    is_critical: bool = False
    ttl_sec: float = 300.0

    def is_expired(self) -> bool:
        if self.status in (
            ConsumptionStatus.BY_DECISION,
            ConsumptionStatus.BY_PLANNER,
            ConsumptionStatus.BY_SAFETY,
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
    """联动验证报告"""
    registered_count: int = 0
    active_output_count: int = 0
    delivered_count: int = 0
    orphan_count: int = 0
    expired_count: int = 0
    critical_orphans: List[EngineOutputRecord] = field(default_factory=list)
    engine_health: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    flow_ok: int = 0
    flow_broken: int = 0
    broken_edges: List[str] = field(default_factory=list)
    overall_score: float = 1.0
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "已注册引擎": self.registered_count,
            "活跃产出": self.active_output_count,
            "已交付": self.delivered_count,
            "未接收": self.orphan_count,
            "已过期": self.expired_count,
            "关键未接收": len(self.critical_orphans),
            "流边通过": self.flow_ok,
            "流边断裂": self.flow_broken,
            "健康度": round(self.overall_score, 3),
            "警告": self.warnings[:8],
            "建议": self.recommendations[:5],
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
            size_bytes=data_size,
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
        status: ConsumptionStatus = ConsumptionStatus.BY_DECISION,
    ) -> bool:
        for rec in reversed(self._outputs):
            if (
                rec.engine_id == engine_id
                and rec.output_key == output_key
                and rec.status == ConsumptionStatus.WAITING
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
        report.registered_count = len(self._registered_engines)
        active = [r for r in self._outputs if not r.is_expired()]
        stale = [r for r in self._outputs if r.is_expired()]
        consumed = [
            r
            for r in active
            if r.status != ConsumptionStatus.WAITING
            and r.status != ConsumptionStatus.ORPHAN
        ]
        orphaned = [
            r
            for r in active
            if r.status == ConsumptionStatus.WAITING
            or r.status == ConsumptionStatus.ORPHAN
        ]
        critical_orphaned = [r for r in orphaned if r.is_critical]
        report.active_output_count = len(active)
        report.delivered_count = len(consumed)
        report.orphan_count = len(orphaned)
        report.expired_count = len(stale)
        report.critical_orphans = critical_orphaned
        engine_stats: Dict[str, Dict[str, Any]] = {}
        for eng in self._registered_engines:
            eng_outputs = [r for r in active if r.engine_id == eng]
            eng_consumed = [
                r
                for r in eng_outputs
                if r.status
                not in (ConsumptionStatus.WAITING, ConsumptionStatus.ORPHAN)
            ]
            eng_orphaned = [
                r
                for r in eng_outputs
                if r.status
                in (ConsumptionStatus.WAITING, ConsumptionStatus.ORPHAN)
            ]
            health = 1.0
            if eng_outputs:
                health = len(eng_consumed) / len(eng_outputs)
            engine_stats[eng.value] = {
                "名称": eng.label(),
                "产出数": len(eng_outputs),
                "已交付": len(eng_consumed),
                "未接收": len(eng_orphaned),
                "健康度": round(health, 3),
                "最近产出秒数": min(
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
                and r.status not in (ConsumptionStatus.ORPHAN,)
                for r in active
            )
            if found:
                verified += 1
                self._edge_status[ek] = True
            else:
                broken += 1
                self._edge_status[ek] = False
                broken_list.append(ek)
        report.flow_ok = verified
        report.flow_broken = broken
        report.broken_edges = broken_list
        score = 1.0
        expected_min_engines = 15
        if report.registered_count < expected_min_engines:
            score *= (
                0.7
                + (report.registered_count / expected_min_engines)
                * 0.3
            )
            report.warnings.append(
                f"仅{report.registered_count}个引擎注册，预期≥{expected_min_engines}"
            )
        if active:
            deliver_ratio = report.delivered_count / max(
                1, report.active_output_count
            )
            score *= 0.6 + deliver_ratio * 0.4
        if critical_orphaned:
            score *= max(0.3, 1.0 - len(critical_orphaned) * 0.15)
            report.warnings.append(
                f"{len(critical_orphaned)}条关键产出未被接收"
            )
        if broken > 0:
            edge_penalty = broken / max(1, verified + broken)
            score *= 1.0 - edge_penalty * 0.3
        report.overall_score = max(0.0, min(1.0, score))
        recs = []
        if report.orphan_count > 3:
            recs.append(f"有{report.orphan_count}条未被接收的产出")
        if report.flow_broken > 2:
            recs.append(f"有{report.flow_broken}条数据流边断裂")
        if report.overall_score < 0.6:
            recs.append("系统健康度低于0.6，建议排查")
        if critical_orphaned:
            for co in critical_orphaned[:3]:
                recs.append(
                    f"关键未接收: {co.engine_id.label()}/{co.output_key}"
                )
        report.recommendations = recs
        report.validated_at = now
        self._last_report = report
        logger.info(
            f"[联动验证] 完成: 健康度={report.overall_score:.3f}, "
            f"引擎={report.registered_count}, "
            f"产出={report.active_output_count}, "
            f"已交付={report.delivered_count}, "
            f"未接收={report.orphan_count}, "
            f"流边={verified}/{verified + broken}"
        )
        return report

    def get_quick_status(self) -> Dict[str, Any]:
        if self._last_report is None:
            return {"status": "尚未运行验证"}
        r = self._last_report
        return {
            "健康度": round(r.overall_score, 3),
            "引擎数": r.registered_count,
            "未接收": r.orphan_count,
            "断裂流边": r.flow_broken,
            "关键告警": len(r.critical_orphans),
            "是否健康": r.overall_score >= 0.7,
        }

    # ────────────────── 内部方法 ──────────────────

    def _build_predefined_edges(self) -> None:
        for src_id, src_field, tgt, tgt_field in _PREDEFINED_EDGES:
            edge = DataFlowEdge(src_id, src_field, tgt, tgt_field)
            self._edges.append(edge)
            self._edge_status[edge.key()] = False
        logger.info(f"[联动验证] 预定义{len(self._edges)}条数据流边")


_validation_lock = threading.Lock()


def get_cross_engine_validator() -> CrossEngineValidator:
    global _validation_singleton
    if _validation_singleton is None:
        with _validation_lock:
            if _validation_singleton is None:
                _validation_singleton = CrossEngineValidator()
    return _validation_singleton

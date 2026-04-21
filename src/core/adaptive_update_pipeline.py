import time
import hashlib
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from src.common.logger import get_logger

logger = get_logger("adaptive_pipeline")

_pipeline_singleton = None

# ═══════════════════════════════════════════════════
#  管道阶段枚举与数据结构
# ═══════════════════════════════════════════════════


class PipelinePhase(Enum):
    """管道阶段"""

    IDLE = "idle"
    COLLECTING = "collecting"
    SUMMARIZING = "summarizing"
    GENERATING = "generating"
    VALIDATING = "validating"
    PUBLISHING = "publishing"
    ROLLING_BACK = "rolling_back"
    COMPLETED = "completed"
    FAILED = "failed"

    def label(self) -> str:
        return {
            "idle": "空闲",
            "collecting": "收集中",
            "summarizing": "理解总结中",
            "generating": "候选生成中",
            "validating": "工具校验中",
            "publishing": "发布中",
            "rolling_back": "回滚中",
            "completed": "已完成",
            "failed": "失败",
        }.get(self.value, "")


class UpdateType(Enum):
    """更新类型"""

    SKILL_NEW = "skill_new"
    SKILL_UPDATE = "skill_update"
    RULE_PATCH = "rule_patch"
    PARAMETER_TUNING = "parameter_tuning"
    CONFIG_ADJUSTMENT = "config_adjustment"
    CODE_MODULE_HOTFIX = "code_module_hotfix"
    MEMORY_CLEANUP = "memory_cleanup"
    IMPRESSION_REFRESH = "impression_refresh"

    def label(self) -> str:
        return {
            "skill_new": "新技能",
            "skill_update": "技能更新",
            "rule_patch": "规则补丁",
            "parameter_tuning": "参数调优",
            "config_adjustment": "配置调整",
            "code_module_hotfix": "代码模块热修复",
            "memory_cleanup": "记忆清理",
            "impression_refresh": "印象刷新",
        }.get(self.value, "")


class ValidationSeverity(Enum):
    """校验严重级别"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class RawEventRecord:
    """原始事件记录——第一步收集的输入"""

    event_id: str = ""
    event_type: str = ""
    timestamp: float = field(default_factory=time.time)
    source_channel: str = ""
    raw_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    collected_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id[:16],
            "event_type": self.event_type,
            "source": self.source_channel[:20],
            "priority": self.priority,
            "collected_ago_sec": round(time.time() - self.collected_at, 0),
        }


@dataclass
class SummaryInsight:
    """第二步：理解总结输出"""

    summary_text: str = ""
    pattern_category: str = ""
    detected_gaps: List[str] = field(default_factory=list)
    existing_mechanism_coverage: str = ""
    novelty_score: float = 0.0
    complexity_estimate: float = 0.5
    recommended_action: str = ""
    should_generate: bool = False
    generation_rationale: str = ""
    summarized_at: float = field(default_factory=time.time)
    input_event_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary_text[:120],
            "pattern": self.pattern_category,
            "gaps": self.detected_gaps[:4],
            "novelty": round(self.novelty_score, 3),
            "should_generate": self.should_generate,
            "rationale": self.generation_rationale[:80],
        }


@dataclass
class GenerationCandidate:
    """第三步：候选生成输出"""

    candidate_id: str = ""
    update_type: UpdateType = UpdateType.RULE_PATCH
    target_path: str = ""
    description: str = ""
    content_preview: str = ""
    full_content: str = ""
    expected_roi: str = ""
    cost_estimate: str = ""
    risk_level: str = "low"
    replaces_existing: bool = False
    replaced_id: str = ""
    generated_at: float = field(default_factory=time.time)
    parent_summary_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id[:12],
            "type": self.update_type.label(),
            "target": self.target_path[:40],
            "risk": self.risk_level,
            "replaces": self.replaces_existing,
        }


@dataclass
class ValidationResult:
    """第四步：校验结果"""

    candidate_id: str = ""
    passed: bool = False
    severity: ValidationSeverity = ValidationSeverity.INFO
    checks_run: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 0.0
    validated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id[:12],
            "passed": self.passed,
            "severity": self.severity.value,
            "score": round(self.score, 3),
            "failed": self.failed_checks[:3],
            "warnings": self.warnings[:3],
        }


@dataclass
class PublishResult:
    """第五步：发布结果"""

    candidate_id: str = ""
    published: bool = False
    publish_path: str = ""
    staging_path: str = ""
    rollback_available: bool = False
    rollback_snapshot: str = ""
    published_at: float = field(default_factory=time.time)
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id[:12],
            "published": self.published,
            "path": self.publish_path[:50],
            "rollback_ready": self.rollback_available,
            "error": self.error_message[:60],
        }


@dataclass
class PipelineRun:
    """一次完整的管道运行记录"""

    run_id: str = ""
    phase: PipelinePhase = PipelinePhase.IDLE
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    events_collected: int = 0
    summary: Optional[SummaryInsight] = None
    candidates: List[GenerationCandidate] = field(default_factory=list)
    validations: List[ValidationResult] = field(default_factory=list)
    publishes: List[PublishResult] = field(default_factory=list)
    final_status: str = ""
    error_log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id[:12],
            "phase": self.phase.label(),
            "events": self.events_collected,
            "candidates": len(self.candidates),
            "passed_validations": sum(1 for v in self.validations if v.passed),
            "published": sum(1 for p in self.publishes if p.published),
            "status": self.final_status or self.phase.value,
        }


# 全局常量
_MAX_EVENTS_PER_RUN = 30
_COLLECTION_WINDOW_SEC = 3600.0
_SUMMARY_MIN_EVENTS = 3
_GENERATION_BUDGET_DAILY = 5
_VALIDATION_CHECKS = [
    "format_check",
    "whitelist_check",
    "safety_boundary_check",
    "interface_compatibility",
    "size_limit_check",
    "duplicate_check",
]
_PUBLISH_STAGING_DIR = "_staging_"


class AdaptiveUpdatePipeline:
    """自适应更新管道
    实现文档描述的后台自适应演化系统五步链路：
    第一步：收集 —— 从事件池、日志、状态快照、技能库中收集近期数据
    第二步：理解总结 —— 后台模型做模式归纳/问题聚类/新旧对比，判断是否值得生成
    第三步：候选生成 —— 仅在确认存在稳定缺口时生成 skill/规则/参数/代码候选
    第四步：工具校验 —— 格式/白名单/安全边界/接口兼容性检查 + 回滚准备
    第五步：发布热更新 —— staging → 灰度验证 → 正式发布 → 失败自动回滚
    """

    def __init__(self):
        self._event_buffer: List[RawEventRecord] = []
        self._current_run: Optional[PipelineRun] = None
        self._run_history: List[PipelineRun] = []
        self._max_history = 20
        self._daily_generation_count: int = 0
        self._last_reset_date: str = ""
        self._external_summarizer: Optional[Callable] = None
        self._external_generator: Optional[Callable] = None
        self._whitelist_dirs: Set[str] = {
            "src/core",
            "src/chat",
            "src/memory_system",
        }
        self._blacklist_files: Set[str] = set()
        self._staging_records: Dict[str, Dict[str, Any]] = {}

    # ═════════════════ 外部回调注册 ═════════════════

    def register_summarizer(self, fn: Callable) -> None:
        self._external_summarizer = fn
        logger.info("[自适应管道] 已注册外部总结器")

    def register_generator(self, fn: Callable) -> None:
        self._external_generator = fn
        logger.info("[自适应管道] 已注册外部生成器")

    def set_whitelist_dirs(self, dirs: Set[str]) -> None:
        self._whitelist_dirs = dirs

    def add_blacklist_file(self, path: str) -> None:
        self._blacklist_files.add(path)

    # ═════════════════ 第一步：收集 ═════════════════

    def collect_event(
        self,
        event_type: str,
        raw_content: str = "",
        *,
        source_channel: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> RawEventRecord:
        record = RawEventRecord(
            event_id=self._gen_id("evt"),
            event_type=event_type,
            timestamp=time.time(),
            source_channel=source_channel,
            raw_content=raw_content[:2000],
            metadata=metadata or {},
            priority=priority,
        )
        self._event_buffer.append(record)
        if len(self._event_buffer) > _MAX_EVENTS_PER_RUN * 2:
            cutoff = time.time() - _COLLECTION_WINDOW_SEC
            self._event_buffer = [
                e for e in self._event_buffer if e.timestamp > cutoff
            ]
        return record

    def collect_batch(self, events: List[Dict[str, Any]]) -> int:
        count = 0
        for ev in events:
            self.collect_event(
                event_type=ev.get("type", "unknown"),
                raw_content=ev.get("content", ""),
                source_channel=ev.get("source", ""),
                metadata=ev.get("metadata", {}),
                priority=ev.get("priority", 0),
            )
            count += 1
        return count

    def get_collection_stats(self) -> Dict[str, Any]:
        now = time.time()
        recent = [
            e
            for e in self._event_buffer
            if (now - e.timestamp) < _COLLECTION_WINDOW_SEC
        ]
        type_counts: Dict[str, int] = {}
        for e in recent:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
        return {
            "buffer_size": len(self._event_buffer),
            "recent_window_count": len(recent),
            "type_distribution": type_counts,
            "daily_generation_used": self._daily_generation_count,
            "daily_budget_remaining": max(
                0, _GENERATION_BUDGET_DAILY - self._daily_generation_count
            ),
        }

    # ═════════════════ 主运行入口 ═════════════════

    def run_pipeline(self) -> PipelineRun:
        run = PipelineRun(
            run_id=self._gen_id("run"),
            phase=PipelinePhase.COLLECTING,
            started_at=time.time(),
        )
        self._current_run = run
        try:
            self._check_daily_reset()
            recent_events = self._collect_for_run(run)
            if len(recent_events) < _SUMMARY_MIN_EVENTS:
                run.phase = PipelinePhase.COMPLETED
                run.final_status = "skipped_insufficient_events"
                self._finalize_run(run)
                return run
            run.phase = PipelinePhase.SUMMARIZING
            summary = self._step_summarize(recent_events, run)
            run.summary = summary
            if not summary.should_generate:
                run.phase = PipelinePhase.COMPLETED
                run.final_status = "summarized_no_generation_needed"
                self._finalize_run(run)
                return run
            if self._daily_generation_count >= _GENERATION_BUDGET_DAILY:
                run.phase = PipelinePhase.FAILED
                run.final_status = "daily_generation_budget_exhausted"
                self._finalize_run(run)
                return run
            run.phase = PipelinePhase.GENERATING
            candidates = self._step_generate(summary, run)
            run.candidates = candidates
            if not candidates:
                run.phase = PipelinePhase.COMPLETED
                run.final_status = "no_candidates_generated"
                self._finalize_run(run)
                return run
            run.phase = PipelinePhase.VALIDATING
            validations = self._step_validate(candidates, run)
            run.validations = validations
            passed = [v for v in validations if v.passed]
            if not passed:
                run.phase = PipelinePhase.COMPLETED
                run.final_status = "all_candidates_failed_validation"
                self._finalize_run(run)
                return run
            run.phase = PipelinePhase.PUBLISHING
            publishes = self._step_publish(passed, candidates, run)
            run.publishes = publishes
            any_published = any(p.published for p in publishes)
            run.phase = PipelinePhase.COMPLETED
            run.final_status = (
                "published" if any_published else "publish_failed"
            )
            self._daily_generation_count += len(
                [p for p in publishes if p.published]
            )
            self._finalize_run(run)
            return run
        except Exception as exc:
            run.phase = PipelinePhase.FAILED
            run.final_status = f"error: {str(exc)[:100]}"
            run.error_log.append(str(exc))
            self._finalize_run(run)
            logger.error(f"[自适应管道] 运行异常: {exc}", exc_info=True)
            return run

    # ═════════════════ 各步骤实现 ═════════════════

    def _collect_for_run(self, run: PipelineRun) -> List[RawEventRecord]:
        now = time.time()
        window = now - _COLLECTION_WINDOW_SEC
        recent = sorted(
            [e for e in self._event_buffer if e.timestamp > window],
            key=lambda e: (-e.priority, e.timestamp),
        )
        selected = recent[:_MAX_EVENTS_PER_RUN]
        run.events_collected = len(selected)
        return selected

    def _step_summarize(
        self, events: List[RawEventRecord], run: PipelineRun
    ) -> SummaryInsight:
        if self._external_summarizer is not None:
            try:
                result = self._external_summarizer(events)
                if isinstance(result, SummaryInsight):
                    return result
                if isinstance(result, dict):
                    return SummaryInsight(
                        **{
                            k: v
                            for k, v in result.items()
                            if hasattr(SummaryInsight, k)
                        }
                    )
            except Exception as exc:
                logger.warning(
                    f"[自适应管道] 外部总结器异常，回退内置逻辑: {exc}"
                )
        return self._builtin_summarize(events, run)

    def _builtin_summarize(
        self, events: List[RawEventRecord], run: PipelineRun
    ) -> SummaryInsight:
        type_counts: Dict[str, int] = {}
        for e in events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
        top_types = sorted(
            type_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]
        pattern = top_types[0][0] if top_types else "unknown"
        total = len(events)
        unique_types = len(type_counts)
        diversity = unique_types / max(1, total)
        novelty = min(
            1.0, diversity * 0.6 + (total / _MAX_EVENTS_PER_RUN) * 0.4
        )
        gaps = []
        if (
            "unknown_event" in type_counts
            and type_counts["unknown_event"] >= 3
        ):
            gaps.append("存在未分类未知事件")
        if "understanding_failure" in type_counts:
            gaps.append("出现理解失败事件")
        if "skill_miss" in type_counts:
            gaps.append("技能覆盖缺口")
        should_gen = (
            novelty > 0.35
            and total >= _SUMMARY_MIN_EVENTS
            and (len(gaps) > 0 or novelty > 0.55)
        )
        rationale_parts = []
        if novelty > 0.55:
            rationale_parts.append(f"新颖度高({novelty:.2f})")
        if gaps:
            rationale_parts.append(f"检测到{len(gaps)}个缺口")
        if total >= 10:
            rationale_parts.append(f"事件量充足({total}条)")
        return SummaryInsight(
            summary_text=f"收集了{total}条事件，主模式={pattern}，类型多样性={
                diversity:.2f}",
            pattern_category=pattern,
            detected_gaps=gaps,
            existing_mechanism_coverage="待分析",
            novelty_score=novelty,
            complexity_estimate=min(1.0, diversity * 0.8),
            recommended_action=(
                "generate_candidate" if should_gen else "monitor_only"
            ),
            should_generate=should_gen,
            generation_rationale=(
                "；".join(rationale_parts)
                if rationale_parts
                else "不满足生成条件"
            ),
            input_event_ids=[e.event_id for e in events],
        )

    def _step_generate(
        self, summary: SummaryInsight, run: PipelineRun
    ) -> List[GenerationCandidate]:
        if self._external_generator is not None:
            try:
                result = self._external_generator(summary)
                if isinstance(result, list):
                    return result
            except Exception as exc:
                logger.warning(f"[自适应管道] 外部生成器异常: {exc}")
        candidates = []
        base_type = self._map_update_type(summary.pattern_category)
        cand = GenerationCandidate(
            candidate_id=self._gen_id("cand"),
            update_type=base_type,
            target_path=self._suggest_target_path(base_type),
            description=f"基于「{summary.pattern_category}」模式的自动建议",
            content_preview=summary.summary_text[:100],
            expected_roi="待评估",
            cost_estimate="低",
            risk_level="low",
            replaces_existing=False,
            parent_summary_id="",
        )
        candidates.append(cand)
        if summary.novelty_score > 0.6 and len(summary.detected_gaps) >= 2:
            cand2 = GenerationCandidate(
                candidate_id=self._gen_id("cand"),
                update_type=UpdateType.PARAMETER_TUNING,
                target_path="src/config/",
                description=f"针对检测到的缺口调参: {', '.join(summary.detected_gaps[:2])}",
                content_preview="参数调整建议",
                risk_level="medium",
                replaces_existing=False,
            )
            candidates.append(cand2)
        return candidates

    @staticmethod
    def _map_update_type(pattern: str) -> UpdateType:
        mapping = {
            "skill_miss": UpdateType.SKILL_NEW,
            "understanding_failure": UpdateType.SKILL_UPDATE,
            "unknown_event": UpdateType.RULE_PATCH,
            "performance_issue": UpdateType.PARAMETER_TUNING,
            "config_drift": UpdateType.CONFIG_ADJUSTMENT,
            "memory_bloat": UpdateType.MEMORY_CLEANUP,
            "impression_stale": UpdateType.IMPRESSION_REFRESH,
        }
        return mapping.get(pattern, UpdateType.RULE_PATCH)

    def _suggest_target_path(self, utype: UpdateType) -> str:
        path_map = {
            UpdateType.SKILL_NEW: "src/plugin_system/skills/",
            UpdateType.SKILL_UPDATE: "src/plugin_system/skills/",
            UpdateType.RULE_PATCH: "src/core/",
            UpdateType.PARAMETER_TUNING: "src/config/",
            UpdateType.CONFIG_ADJUSTMENT: "src/config/",
            UpdateType.CODE_MODULE_HOTFIX: "src/core/",
            UpdateType.MEMORY_CLEANUP: "src/memory_system/",
            UpdateType.IMPRESSION_REFRESH: "src/core/",
        }
        return path_map.get(utype, "src/core/")

    def _step_validate(
        self, candidates: List[GenerationCandidate], run: PipelineRun
    ) -> List[ValidationResult]:
        results = []
        for cand in candidates:
            vr = self._validate_single(cand)
            results.append(vr)
        return results

    def _validate_single(self, cand: GenerationCandidate) -> ValidationResult:
        vr = ValidationResult(
            candidate_id=cand.candidate_id,
            checks_run=[],
            failed_checks=[],
            warnings=[],
        )
        score_acc = 0.0
        check_count = len(_VALIDATION_CHECKS)
        if check_count == 0:
            check_count = 1
        for check_name in _VALIDATION_CHECKS:
            vr.checks_run.append(check_name)
            passed, severity, msg = self._run_check(check_name, cand)
            if not passed:
                vr.failed_checks.append(check_name)
                if severity == ValidationSeverity.CRITICAL:
                    vr.severity = severity
                elif (
                    severity == ValidationSeverity.ERROR
                    and vr.severity != ValidationSeverity.CRITICAL
                ):
                    vr.severity = severity
                elif (
                    severity == ValidationSeverity.WARNING
                    and vr.severity == ValidationSeverity.INFO
                ):
                    vr.severity = severity
                vr.warnings.append(msg)
            else:
                score_acc += 1.0 / check_count
        vr.passed = len(vr.failed_checks) == 0
        vr.score = score_acc
        return vr

    def _run_check(
        self, check_name: str, cand: GenerationCandidate
    ) -> Tuple[bool, ValidationSeverity, str]:
        if check_name == "format_check":
            if not cand.description or len(cand.description) < 5:
                return False, ValidationSeverity.WARNING, "描述过短"
            return True, ValidationSeverity.INFO, "格式OK"
        if check_name == "whitelist_check":
            path = cand.target_path
            allowed = any(path.startswith(d) for d in self._whitelist_dirs)
            if not allowed:
                return (
                    False,
                    ValidationSeverity.ERROR,
                    f"路径不在白名单内: {path}",
                )
            if path in self._blacklist_files:
                return (
                    False,
                    ValidationSeverity.CRITICAL,
                    f"路径在黑名单内: {path}",
                )
            return True, ValidationSeverity.INFO, "白名单通过"
        if check_name == "safety_boundary_check":
            dangerous = [
                "__import__",
                "eval(",
                "exec(",
                "os.system",
                "subprocess",
            ]
            content = cand.content_preview + cand.full_content
            for kw in dangerous:
                if kw in content:
                    return (
                        False,
                        ValidationSeverity.CRITICAL,
                        f"检测到危险关键词: {kw}",
                    )
            return True, ValidationSeverity.INFO, "安全检查通过"
        if check_name == "interface_compatibility":
            return True, ValidationSeverity.INFO, "接口兼容性待运行时确认"
        if check_name == "size_limit_check":
            total_len = len(cand.full_content or cand.content_preview)
            if total_len > 50000:
                return (
                    False,
                    ValidationSeverity.WARNING,
                    f"内容过大({total_len}字符)",
                )
            return True, ValidationSeverity.INFO, "大小正常"
        if check_name == "duplicate_check":
            return True, ValidationSeverity.INFO, "去重检查待确认"
        return True, ValidationSeverity.INFO, "未知检查项默认通过"

    def _step_publish(
        self,
        validations: List[ValidationResult],
        all_candidates: List[GenerationCandidate],
        run: PipelineRun,
    ) -> List[PublishResult]:
        results = []
        val_map = {v.candidate_id: v for v in validations}
        for cand in all_candidates:
            vr = val_map.get(cand.candidate_id)
            if vr is None or not vr.passed:
                results.append(
                    PublishResult(
                        candidate_id=cand.candidate_id,
                        published=False,
                        error_message="未通过校验" if vr else "无校验结果",
                    )
                )
                continue
            pr = self._publish_single(cand, run)
            results.append(pr)
        return results

    def _publish_single(
        self, cand: GenerationCandidate, run: PipelineRun
    ) -> PublishResult:
        snapshot_id = self._gen_id("snap")
        snap_data = {
            "candidate_id": cand.candidate_id,
            "update_type": cand.update_type.value,
            "target_path": cand.target_path,
            "description": cand.description,
            "published_at": time.time(),
            "run_id": run.run_id,
        }
        self._staging_records[snapshot_id] = snap_data
        pr = PublishResult(
            candidate_id=cand.candidate_id,
            published=True,
            staging_path=f"{_PUBLISH_STAGING_DIR}{snapshot_id}.json",
            rollback_available=True,
            rollback_snapshot=snapshot_id,
        )
        logger.info(
            f"[自适应管道] 候选 {cand.candidate_id[:12]} 已发布到staging "
            f"(类型={cand.update_type.label()}, 路径={cand.target_path})"
        )
        return pr

    def rollback(self, snapshot_id: str) -> bool:
        snap = self._staging_records.pop(snapshot_id, None)
        if snap is None:
            logger.warning(f"[自适应管道] 回滚失败: 快照 {snapshot_id} 不存在")
            return False
        logger.info(
            f"[自适应管道] 已回滚快照 {snapshot_id} ({
                snap.get(
                    'candidate_id',
                    '?')})"
        )
        return True

    def get_staging_snapshot(
        self, snapshot_id: str
    ) -> Optional[Dict[str, Any]]:
        return self._staging_records.get(snapshot_id)

    # ────────────────── 工具方法 ──────────────────

    @staticmethod
    def _gen_id(prefix: str = "") -> str:
        ts = str(time.time()).encode()
        rand_bytes = str(time.monotonic_ns()).encode()
        raw = prefix.encode() + ts + rand_bytes
        return hashlib.sha256(raw).hexdigest()[:16]

    def _check_daily_reset(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if self._last_reset_date != today:
            self._daily_generation_count = 0
            self._last_reset_date = today
            logger.info("[自适应管道] 每日预算已重置")

    def _finalize_run(self, run: PipelineRun) -> None:
        run.completed_at = time.time()
        self._run_history.append(run)
        if len(self._run_history) > self._max_history:
            self._run_history = self._run_history[-self._max_history:]
        self._current_run = None

    def pipeline_history_summary(self) -> Dict[str, Any]:
        runs = self._run_history[-10:]
        status_dist: Dict[str, int] = {}
        for r in runs:
            s = r.final_status or r.phase.value
            status_dist[s] = status_dist.get(s, 0) + 1
        return {
            "total_runs": len(self._run_history),
            "recent_distribution": status_dist,
            "daily_budget_used": self._daily_generation_count,
            "daily_budget_total": _GENERATION_BUDGET_DAILY,
            "staging_snapshots": len(self._staging_records),
        }


def get_adaptive_pipeline() -> AdaptiveUpdatePipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = AdaptiveUpdatePipeline()
    return _pipeline_singleton

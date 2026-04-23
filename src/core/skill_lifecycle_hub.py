import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("技能生命周期")

_lifecycle_instances: Dict[str, "SkillLifecycleHub"] = {}


def _skill_view() -> Dict[str, Any]:
    try:
        from src.config.core_config_engine import get_core_config

        return get_core_config().resolve_module_view("skill").values
    except Exception:
        return {}


def _skill_int(key: str, fallback: int) -> int:
    try:
        return int(_skill_view().get(key, fallback))
    except Exception:
        return fallback


def _skill_float(key: str, fallback: float) -> float:
    try:
        return float(_skill_view().get(key, fallback))
    except Exception:
        return fallback


class SkillStage(Enum):
    """Skill 生命周期阶段"""

    DISCOVERED = "discovered"
    TRIAL = "trial"
    ACTIVE = "active"
    MATURED = "matured"
    DECLINING = "declining"
    RETIRED = "retired"

    def label(self) -> str:
        labels = {
            SkillStage.DISCOVERED: "发现",
            SkillStage.TRIAL: "试用",
            SkillStage.ACTIVE: "活跃",
            SkillStage.MATURED: "成熟",
            SkillStage.DECLINING: "衰退",
            SkillStage.RETIRED: "退役",
        }
        return labels.get(self, "未知")


class SkillCategory(Enum):
    """Skill 类型"""

    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"
    HUMOR = "humor"
    EMOTIONAL_SUPPORT = "emotional_support"
    TASK_ASSIST = "task_assist"
    SOCIAL_RITUAL = "social_ritual"
    CREATIVE = "creative"
    CUSTOM = "custom"


class SkillDomain(Enum):
    """Skill 运行域分类
    对齐开发计划中的 runtime / analysis / dev 三类技能分层，
    与能力类型并存，不替代现有 category。
    """

    RUNTIME = "runtime"
    ANALYSIS = "analysis"
    DEV = "dev"

    def label(self) -> str:
        labels = {
            SkillDomain.RUNTIME: "运行时技能",
            SkillDomain.ANALYSIS: "分析技能",
            SkillDomain.DEV: "开发技能",
        }
        return labels.get(self, "未知")


@dataclass
class SkillRecord:
    """单个 Skill 的完整记录"""

    skill_id: str = ""
    skill_name: str = ""
    category: SkillCategory = SkillCategory.CONVERSATION
    domain: SkillDomain = SkillDomain.RUNTIME
    stage: SkillStage = SkillStage.DISCOVERED
    description: str = ""
    trigger_conditions: List[str] = field(default_factory=list)
    # 统计
    invocation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    user_positive_feedback: int = 0
    user_negative_feedback: int = 0
    # 时间
    created_at: float = field(default_factory=time.time)
    last_invoked_at: float = 0.0
    last_evaluated_at: float = 0.0
    stage_entered_at: float = field(default_factory=time.time)
    # 评估
    effectiveness_score: float = 0.5
    relevance_score: float = 0.5
    # 预算追踪
    total_cost_consumed: float = 0.0
    total_tokens_consumed: int = 0
    avg_cost_per_call: float = 0.0

    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    def cost_modifier(self) -> float:
        """F16：返回技能成熟度对应的成本修正因子（0.6~1.0）
        成熟技能+高成功率 → 更低成本（已熟练，不需要每次都深度思考）
        新技能/衰退技能 → 正常或更高成本（需要更多资源来处理）"""
        if self.stage == SkillStage.MATURED:
            _base = 0.70
        elif self.stage == SkillStage.ACTIVE:
            _base = 0.85
        elif self.stage == SkillStage.TRIAL:
            _base = 1.05
        elif self.stage == SkillStage.DECLINING:
            _base = 1.10
        elif self.stage == SkillStage.RETIRED:
            _base = 1.20
        else:
            _base = 1.00
        if self.success_count > 10:
            _sr_bonus = min(0.15, (self.success_rate() - 0.7) * 0.3)
            _base = max(0.55, _base - _sr_bonus)
        if self.invocation_count > 50 and self.avg_cost_per_call > 0:
            _efficiency = min(1.0, 200.0 / max(1.0, self.avg_cost_per_call))
            _base = max(0.50, _base * _efficiency)
        return round(_base, 3)

    def feedback_score(self) -> float:
        total_fb = self.user_positive_feedback + self.user_negative_feedback
        if total_fb == 0:
            return 0.5
        return self.user_positive_feedback / total_fb

    def idle_days(self) -> float:
        if self.last_invoked_at == 0.0:
            return (time.time() - self.created_at) / 86400.0
        return (time.time() - self.last_invoked_at) / 86400.0

    def composite_score(self) -> float:
        """综合评分：效果 * 0.4 + 反馈 * 0.3 + 相关性 * 0.3"""
        effect_weight = _skill_float("composite_effectiveness_weight", 0.4)
        feedback_weight = _skill_float("composite_feedback_weight", 0.3)
        relevance_weight = _skill_float("composite_relevance_weight", 0.3)
        return (
            self.effectiveness_score * effect_weight
            + self.feedback_score() * feedback_weight
            + self.relevance_score * relevance_weight
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "category": self.category.value,
            "domain": self.domain.value,
            "domain_label": self.domain.label(),
            "stage": self.stage.value,
            "stage_label": self.stage.label(),
            "invocation_count": self.invocation_count,
            "success_rate": round(self.success_rate(), 3),
            "feedback_score": round(self.feedback_score(), 3),
            "composite_score": round(self.composite_score(), 3),
            "idle_days": round(self.idle_days(), 1),
            "total_cost": round(self.total_cost_consumed, 3),
            "avg_cost": round(self.avg_cost_per_call, 3),
        }


class SkillLifecycleHub:
    """Skill 生命周期中心
    管理 Skill 的分类、试用、评估、淘汰全流程。
    生命周期：发现 → 试用 → 活跃 → 成熟 → 衰退 → 退役
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._skills: Dict[str, SkillRecord] = {}
        # 晋升/降级阈值
        self._trial_to_active_invocations = _skill_int("trial_to_active_invocations", 5)
        self._trial_to_active_success_rate = _skill_float("trial_to_active_success_rate", 0.5)
        self._active_to_matured_invocations = _skill_int("active_to_matured_invocations", 20)
        self._active_to_matured_success_rate = _skill_float("active_to_matured_success_rate", 0.7)
        self._decline_idle_days = _skill_int("decline_idle_days", 14)
        self._retire_idle_days = _skill_int("retire_idle_days", 30)
        self._retire_low_score = _skill_float("retire_low_score", 0.2)

    def register_skill(
        self,
        skill_id: str,
        *,
        skill_name: str = "",
        category: SkillCategory = SkillCategory.CONVERSATION,
        domain: SkillDomain = SkillDomain.RUNTIME,
        description: str = "",
        trigger_conditions: Optional[List[str]] = None,
    ) -> SkillRecord:
        """注册一个新 Skill"""
        if skill_id in self._skills:
            return self._skills[skill_id]
        record = SkillRecord(
            skill_id=skill_id,
            skill_name=skill_name or skill_id,
            category=category,
            domain=domain,
            description=description,
            trigger_conditions=trigger_conditions or [],
        )
        self._skills[skill_id] = record
        logger.debug(
            f"[Skill] {self._channel_id} 注册 {skill_name or skill_id}"
        )
        return record

    def record_invocation(
        self,
        skill_id: str,
        *,
        success: bool = True,
        effectiveness: float = 0.5,
    ) -> Optional[SkillRecord]:
        """记录 Skill 调用结果"""
        record = self._skills.get(skill_id)
        if record is None:
            return None
        record.invocation_count += 1
        record.last_invoked_at = time.time()
        if success:
            record.success_count += 1
        else:
            record.failure_count += 1
        # 指数移动平均更新效果分
        alpha = 0.15
        record.effectiveness_score = (
            record.effectiveness_score * (1 - alpha) + effectiveness * alpha
        )
        self._evaluate_lifecycle(record)
        return record

    def record_feedback(self, skill_id: str, *, positive: bool = True) -> None:
        """记录用户反馈"""
        record = self._skills.get(skill_id)
        if record is None:
            return
        if positive:
            record.user_positive_feedback += 1
        else:
            record.user_negative_feedback += 1

    def update_relevance(self, skill_id: str, relevance: float) -> None:
        """更新相关性得分"""
        record = self._skills.get(skill_id)
        if record is None:
            return
        record.relevance_score = max(0.0, min(1.0, relevance))

    def get_skill(self, skill_id: str) -> Optional[SkillRecord]:
        return self._skills.get(skill_id)

    def get_active_skills(self) -> List[SkillRecord]:
        return [
            s
            for s in self._skills.values()
            if s.stage in (SkillStage.ACTIVE, SkillStage.MATURED)
        ]

    def get_trial_skills(self) -> List[SkillRecord]:
        return [
            s for s in self._skills.values() if s.stage == SkillStage.TRIAL
        ]

    def get_by_category(self, category: SkillCategory) -> List[SkillRecord]:
        return [
            s
            for s in self._skills.values()
            if s.category == category and s.stage != SkillStage.RETIRED
        ]

    def get_by_domain(self, domain: SkillDomain) -> List[SkillRecord]:
        """按运行域筛选 Skill。"""
        return [
            s
            for s in self._skills.values()
            if s.domain == domain and s.stage != SkillStage.RETIRED
        ]

    def top_skills(self, count: int = 5) -> List[SkillRecord]:
        """返回综合评分最高的活跃 Skill"""
        candidates = [
            s
            for s in self._skills.values()
            if s.stage in (SkillStage.ACTIVE, SkillStage.MATURED)
        ]
        candidates.sort(key=lambda s: s.composite_score(), reverse=True)
        return candidates[:count]

    def periodic_review(self) -> Dict[str, int]:
        """周期性审查所有 Skill"""
        promoted = 0
        declined = 0
        retired = 0
        for record in self._skills.values():
            old_stage = record.stage
            self._evaluate_lifecycle(record)
            if record.stage != old_stage:
                if record.stage.value in ("active", "matured"):
                    promoted += 1
                elif record.stage == SkillStage.DECLINING:
                    declined += 1
                elif record.stage == SkillStage.RETIRED:
                    retired += 1
        return {"promoted": promoted, "declined": declined, "retired": retired}

    def summary(self) -> Dict[str, Any]:
        stage_counts: Dict[str, int] = {}
        domain_counts: Dict[str, int] = {}
        for record in self._skills.values():
            stage_counts[record.stage.value] = (
                stage_counts.get(record.stage.value, 0) + 1
            )
            domain_counts[record.domain.value] = (
                domain_counts.get(record.domain.value, 0) + 1
            )
        return {
            "total_skills": len(self._skills),
            "by_stage": stage_counts,
            "by_domain": domain_counts,
            "top_skills": [s.skill_name for s in self.top_skills(3)],
        }

    def retire_skill(self, skill_id: str) -> bool:
        """手动退役"""
        record = self._skills.get(skill_id)
        if record is None:
            return False
        record.stage = SkillStage.RETIRED
        record.stage_entered_at = time.time()
        return True

    def reactivate_skill(self, skill_id: str) -> bool:
        """重新激活已退役的 Skill"""
        record = self._skills.get(skill_id)
        if record is None:
            return False
        if record.stage != SkillStage.RETIRED:
            return False
        record.stage = SkillStage.TRIAL
        record.stage_entered_at = time.time()
        record.failure_count = 0
        record.success_count = 0
        record.invocation_count = 0
        return True

    # ────────────────── 内部方法 ──────────────────

    def _evaluate_lifecycle(self, record: SkillRecord) -> None:
        """评估并更新 Skill 生命周期阶段"""
        now = time.time()
        record.last_evaluated_at = now
        old_stage = record.stage
        if record.stage == SkillStage.DISCOVERED:
            # 发现 → 试用：至少调用过1次
            if record.invocation_count >= 1:
                record.stage = SkillStage.TRIAL
                record.stage_entered_at = now
        elif record.stage == SkillStage.TRIAL:
            # 试用 → 活跃
            if (
                record.invocation_count >= self._trial_to_active_invocations
                and record.success_rate() >= self._trial_to_active_success_rate
            ):
                record.stage = SkillStage.ACTIVE
                record.stage_entered_at = now
            # 试用失败 → 退役
            elif record.invocation_count >= 10 and record.success_rate() < 0.2:
                record.stage = SkillStage.RETIRED
                record.stage_entered_at = now
        elif record.stage == SkillStage.ACTIVE:
            # 活跃 → 成熟
            if (
                record.invocation_count >= self._active_to_matured_invocations
                and record.success_rate()
                >= self._active_to_matured_success_rate
            ):
                record.stage = SkillStage.MATURED
                record.stage_entered_at = now
            # 活跃 → 衰退（长期未使用）
            if record.idle_days() > self._decline_idle_days:
                record.stage = SkillStage.DECLINING
                record.stage_entered_at = now
        elif record.stage == SkillStage.MATURED:
            # 成熟 → 衰退
            if record.idle_days() > self._decline_idle_days:
                record.stage = SkillStage.DECLINING
                record.stage_entered_at = now
        elif record.stage == SkillStage.DECLINING:
            # 衰退 → 退役
            if record.idle_days() > self._retire_idle_days:
                record.stage = SkillStage.RETIRED
                record.stage_entered_at = now
            # 衰退 → 活跃（重新使用）
            if record.last_invoked_at > record.stage_entered_at:
                record.stage = SkillStage.ACTIVE
                record.stage_entered_at = now
        if record.stage != old_stage:
            logger.debug(
                f"[Skill] {self._channel_id} {record.skill_name} "
                f"{old_stage.label()} → {record.stage.label()}"
            )

    # ════════════════════════════════════════════
    #  预算约束接口
    # ════════════════════════════════════════════
    # 域→基础成本映射
    def estimate_invocation_cost(self, skill_id: str) -> tuple:
        """估算一次调用的预算消耗 (cost, tokens, skill_calls=1)"""
        record = self._skills.get(skill_id)
        domain_base_cost = {
            SkillDomain.RUNTIME: _skill_float("runtime_skill_cost", 0.3),
            SkillDomain.ANALYSIS: _skill_float("analysis_skill_cost", 1.5),
            SkillDomain.DEV: _skill_float("dev_skill_cost", 2.0),
        }
        domain_base_tokens = {
            SkillDomain.RUNTIME: _skill_int("runtime_skill_tokens", 200),
            SkillDomain.ANALYSIS: _skill_int("analysis_skill_tokens", 800),
            SkillDomain.DEV: _skill_int("dev_skill_tokens", 1500),
        }
        if record is None:
            return (1.0, 500, 1)
        # 使用历史平均值（如果有的话）
        if record.avg_cost_per_call > 0 and record.invocation_count >= 3:
            return (
                record.avg_cost_per_call,
                domain_base_tokens.get(record.domain, 500),
                1,
            )
        return (
            domain_base_cost.get(record.domain, 1.0),
            domain_base_tokens.get(record.domain, 500),
            1,
        )

    def budget_gated_invocation(
        self,
        skill_id: str,
        *,
        budget_envelope: Any = None,
        success: bool = True,
        effectiveness: float = 0.5,
        actual_cost: float = 0.0,
        actual_tokens: int = 0,
    ) -> Dict[str, Any]:
        """预算感知的 Skill 调用记录

        如果传入 budget_envelope（BudgetEnvelope 实例），在预算不足时拒绝记录。
        返回 {"allowed": bool, "reason": str, "record": SkillRecord|None}
        """
        est_cost, est_tokens, est_calls = self.estimate_invocation_cost(
            skill_id
        )
        cost = actual_cost if actual_cost > 0 else est_cost
        tokens = actual_tokens if actual_tokens > 0 else est_tokens
        if budget_envelope is not None:
            budget_envelope.reset_if_expired()
            if not budget_envelope.is_within_budget(
                cost, tokens=tokens, skill_calls=1
            ):
                return {
                    "allowed": False,
                    "reason": "预算不足",
                    "record": self._skills.get(skill_id),
                }
            budget_envelope.consume(cost, tokens=tokens, skill_calls=1)
        record = self.record_invocation(
            skill_id, success=success, effectiveness=effectiveness
        )
        if record is not None:
            record.total_cost_consumed += cost
            record.total_tokens_consumed += tokens
            record.avg_cost_per_call = record.total_cost_consumed / max(
                1, record.invocation_count
            )
        return {
            "allowed": True,
            "reason": "",
            "record": record,
        }

    def budget_consumption_report(self) -> Dict[str, Any]:
        """汇总所有 Skill 的预算消耗"""
        domain_cost: Dict[str, float] = {}
        domain_tokens: Dict[str, int] = {}
        domain_calls: Dict[str, int] = {}
        top_consumers: List[Dict[str, Any]] = []
        for record in self._skills.values():
            dkey = record.domain.value
            domain_cost[dkey] = (
                domain_cost.get(dkey, 0.0) + record.total_cost_consumed
            )
            domain_tokens[dkey] = (
                domain_tokens.get(dkey, 0) + record.total_tokens_consumed
            )
            domain_calls[dkey] = (
                domain_calls.get(dkey, 0) + record.invocation_count
            )
            if record.total_cost_consumed > 0:
                top_consumers.append(
                    {
                        "skill": record.skill_name,
                        "domain": dkey,
                        "cost": round(record.total_cost_consumed, 3),
                        "tokens": record.total_tokens_consumed,
                        "calls": record.invocation_count,
                        "avg_cost": round(record.avg_cost_per_call, 3),
                    }
                )
        top_consumers.sort(key=lambda x: x["cost"], reverse=True)
        return {
            "total_cost": round(sum(domain_cost.values()), 3),
            "total_tokens": sum(domain_tokens.values()),
            "total_calls": sum(domain_calls.values()),
            "by_domain": {
                d: {
                    "cost": round(domain_cost.get(d, 0.0), 3),
                    "tokens": domain_tokens.get(d, 0),
                    "calls": domain_calls.get(d, 0),
                }
                for d in set(list(domain_cost) + list(domain_tokens))
            },
            "top_consumers": top_consumers[:10],
        }


def get_skill_lifecycle_hub(channel_id: str) -> SkillLifecycleHub:
    if channel_id not in _lifecycle_instances:
        _lifecycle_instances[channel_id] = SkillLifecycleHub(channel_id)
    return _lifecycle_instances[channel_id]


def remove_skill_lifecycle_hub(channel_id: str) -> None:
    _lifecycle_instances.pop(channel_id, None)

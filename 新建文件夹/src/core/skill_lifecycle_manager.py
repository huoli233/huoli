"""Skill 六段生命周期管理器（Skill Lifecycle Manager）

每个"技能"（可复用的行为模式）经历完整的六段生命周期：
  ① DISCOVERED   —— 发现阶段（首次观察到该行为模式可用）
  ② PROPOSED     —— 提案阶段（内部评估是否值得开发）
  ③ TRIAL        —— 试运行阶段（小范围低频使用）
  ④ EVALUATING  —— 评估阶段（收集使用效果数据）
  ⑤ STABILIZED   —— 稳定阶段（高频可靠使用）
  ⑥ DECLINING/RETIRED —— 衰退/淘汰阶段（逐渐弃用）

每段有独立的：
  - 使用概率和频率限制
  - 成本修正系数
  - 失败回退策略
  - 段间转移条件

参考文档: 复杂群聊智能化与状态机制规划文档.md L3254-3572
"""

import math
import time
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.common.logger import get_logger

logger = get_logger("技能生命周期")


class SkillPhase(Enum):
    DISCOVERED = "discovered"
    PROPOSED = "proposed"
    TRIAL = "trial"
    EVALUATING = "evaluating"
    STABILIZED = "stabilized"
    DECLINING = "declining"
    RETIRED = "retired"


@dataclass
class SkillMetadata:
    """技能元信息"""

    skill_id: str = ""
    name: str = ""
    description: str = ""
    category: str = ""
    discovered_at: float = field(default_factory=time.time)
    discovered_from: str = ""
    tags: List[str] = field(default_factory=list)
    required_contexts: List[str] = field(default_factory=list)


@dataclass
class SkillPhaseState:
    """当前阶段的状态快照"""

    phase: SkillPhase = SkillPhase.DISCOVERED
    entered_at: float = field(default_factory=time.time)
    total_usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    partial_count: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_used_at: float = 0.0
    last_result: str = ""
    avg_confidence: float = 0.5
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=30))
    trial_budget_remaining: int = 10
    stabilization_threshold: float = 0.72
    decline_trigger: str = ""


@dataclass
class SkillRecord:
    """单个技能的完整记录"""

    metadata: SkillMetadata = field(default_factory=SkillMetadata)
    state: SkillPhaseState = field(default_factory=SkillPhaseState)
    is_active: bool = True
    cost_modifier: float = 1.0
    usage_frequency_cap: Dict[str, int] = field(
        default_factory=lambda: {
            "per_hour": 8,
            "per_day": 30,
            "per_week": 100,
        }
    )
    usage_counter: Dict[str, int] = field(
        default_factory=lambda: {
            "current_hour": 0,
            "current_day": 0,
            "current_week": 0,
        }
    )
    window_start: float = field(default_factory=time.time)


class SkillLifecycleManager:
    """Skill 六段生命周期管理器

    管理所有已发现技能的生命周期流转，
    为 ACFN 和 planner 提供成本修正和使用建议。
    """

    MAX_SKILLS_TRACKED = 150
    TRIAL_DEFAULT_BUDGET = 10
    STABILIZATION_MIN_SUCCESS_RATE = 0.70
    DECLINE_INACTIVITY_DAYS = 14.0
    AUTO_RETIRE_SUCCESS_THRESHOLD = 0.35

    def __init__(self):
        self._skills: Dict[str, SkillRecord] = {}
        self._global_stats = {
            "total_skills": 0,
            "phase_distribution": {},
            "avg_success_rate": 0.0,
        }
        self._last_phase_check: float = 0.0
        self._phase_check_interval_sec: float = 600.0

    def discover_skill(
        self,
        skill_id: str,
        name: str,
        category: str,
        source: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> SkillRecord:
        """注册一个新发现的技能（进入 DISCOVERED 阶段）"""
        if skill_id in self._skills:
            return self._skills[skill_id]
        record = SkillRecord(
            metadata=SkillMetadata(
                skill_id=skill_id,
                name=name,
                category=category,
                discovered_from=source,
                description=description,
                tags=tags or [],
            ),
            state=SkillPhaseState(phase=SkillPhase.DISCOVERED),
        )
        self._skills[skill_id] = record
        self._global_stats["total_skills"] += 1
        logger.debug(
            f"[Skill生命周期] 发现新技能: {name}({category}) from {source}"
        )
        return record

    def propose(
        self, skill_id: str, proposer: str = "auto", reason: str = ""
    ) -> bool:
        """将技能从 DISCOVERED 推进到 PROPOSED（提案阶段）

        返回 True 表示提案成功
        """
        rec = self._skills.get(skill_id)
        if not rec or rec.state.phase != SkillPhase.DISCOVERED:
            return False
        if reason:
            rec.metadata.description += f" | 提案理由: {reason}"
        self._transition_to(
            rec, SkillPhase.PROPOSED, f"proposed_by_{proposer}"
        )
        rec.state.trial_budget_remaining = self.TRIAL_DEFAULT_BUDGET
        return True

    def start_trial(self, skill_id: str) -> bool:
        """将技能从 PROPOSED 推进到 TRIAL（试运行阶段）"""
        rec = self._skills.get(skill_id)
        if not rec or rec.state.phase != SkillPhase.PROPOSED:
            return False
        self._transition_to(rec, SkillPhase.TRIAL, "trial_started")
        return True

    def evaluate_usage(
        self,
        skill_id: str,
        success: bool,
        confidence: float = 0.5,
        partial: bool = False,
    ) -> bool:
        """记录一次使用结果，并可能触发阶段转移

        返回 True 表示本次使用被允许
        """
        rec = self._skills.get(skill_id)
        if not rec or not rec.is_active:
            return False
        now = time.time()
        # 使用计数器检查
        self._reset_counters_if_needed(rec, now)
        h_key = "current_hour"
        d_key = "current_day"
        w_key = "current_week"
        cap_h = rec.usage_frequency_cap.get("per_hour", 8)
        cap_d = rec.usage_frequency_cap.get("per_day", 30)
        cap_w = rec.usage_frequency_cap.get("per_week", 100)
        if rec.usage_counter.get(h_key, 0) >= cap_h:
            return False
        if rec.usage_counter.get(d_key, 0) >= cap_d:
            return False
        if rec.usage_counter.get(w_key, 0) >= cap_w:
            return False
        # 记录使用
        rec.state.total_usage_count += 1
        rec.state.last_used_at = now
        rec.usage_counter[h_key] = rec.usage_counter.get(h_key, 0) + 1
        rec.usage_counter[d_key] = rec.usage_counter.get(d_key, 0) + 1
        rec.usage_counter[w_key] = rec.usage_counter.get(w_key, 0) + 1
        # 结果记录
        if partial:
            rec.state.partial_count += 1
            rec.state.last_result = "partial"
            rec.state.consecutive_successes = 0
            rec.state.consecutive_failures = 0
        elif success:
            rec.state.success_count += 1
            rec.state.last_result = "success"
            rec.state.consecutive_successes += 1
            rec.state.consecutive_failures = 0
        else:
            rec.state.failure_count += 1
            rec.state.last_result = "failure"
            rec.state.consecutive_failures += 1
            rec.state.consecutive_successes = 0
        # 置信度更新
        rec.state.confidence_history.append(confidence)
        recent = list(rec.state.confidence_history)[-15:]
        rec.state.avg_confidence = sum(recent) / max(1, len(recent))
        # 试运行预算消耗
        if rec.state.phase == SkillPhase.TRIAL:
            rec.state.trial_budget_remaining -= 1
        # 阶段转移检查
        self._check_phase_transitions(rec)
        return True

    def get_cost_modifier(self, skill_id: str) -> float:
        """获取技能的当前成本修正系数

        不同阶段有不同的成本折扣：
          DISCOVERED: 1.20（不熟悉，额外开销）
          PROPOSED:    1.10（刚提案，保守使用）
          TRIAL:       1.00（正常成本）
          EVALUATING: 0.95（已有数据，微调中）
          STABILIZED:  0.70~0.85（成熟，高效）
          DECLINING:   1.05~1.20（衰退，不稳定）
          RETIRED:     2.00+（基本禁用）
        """
        rec = self._skills.get(skill_id)
        if not rec or not rec.is_active:
            return 2.00
        phase = rec.state.phase
        base_modifiers = {
            SkillPhase.DISCOVERED: 1.20,
            SkillPhase.PROPOSED: 1.10,
            SkillPhase.TRIAL: 1.00,
            SkillPhase.EVALUATING: 0.95,
            SkillPhase.STABILIZED: 0.78,
            SkillPhase.DECLINING: 1.12,
            SkillPhase.RETIRED: 2.50,
        }
        base = base_modifiers.get(phase, 1.0)
        if phase == SkillPhase.STABILIZED:
            success_rate = self._get_success_rate(rec)
            if success_rate > 0.88:
                base = 0.68
            elif success_rate > 0.75:
                base = 0.75
            elif success_rate > 0.60:
                base = 0.82
        elif phase == SkillPhase.DECLINING:
            days_inactive = 0.0
            if rec.state.last_used_at > 0:
                days_inactive = (
                    time.time() - rec.state.last_used_at
                ) / 86400.0
            inactivity_penalty = min(
                0.30, days_inactive / self.DECLINE_INACTIVITY_DAYS * 0.20
            )
            base += inactivity_penalty
        rec.cost_modifier = base
        return round(base, 3)

    def get_active_skills(
        self,
        category: Optional[str] = None,
        min_phase: SkillPhase = SkillPhase.TRIAL,
    ) -> List[Dict[str, Any]]:
        """获取所有可用技能的摘要列表"""
        result = []
        phase_order = {
            SkillPhase.RETIRED: 0,
            SkillPhase.DECLINING: 1,
            SkillPhase.DISCOVERED: 2,
            SkillPhase.PROPOSED: 3,
            SkillPhase.TRIAL: 4,
            SkillPhase.EVALUATING: 5,
            SkillPhase.STABILIZED: 6,
        }
        for sid, rec in self._skills.items():
            if not rec.is_active:
                continue
            if phase_order.get(rec.state.phase, 0) < phase_order.get(
                min_phase, 0
            ):
                continue
            if category and rec.metadata.category != category:
                continue
            sr = self._get_success_rate(rec)
            result.append(
                {
                    "id": rec.metadata.skill_id,
                    "name": rec.metadata.name,
                    "category": rec.metadata.category,
                    "phase": rec.state.phase.value,
                    "usage_total": rec.state.total_usage_count,
                    "success_rate": round(sr, 3),
                    "cost_modifier": rec.cost_modifier,
                    "last_used_ago_hours": (
                        round(
                            (time.time() - rec.state.last_used_at) / 3600.0, 1
                        )
                        if rec.state.last_used_at > 0
                        else -1.0
                    ),
                }
            )
        result.sort(
            key=lambda x: (x["success_rate"], -x["usage_total"]), reverse=True
        )
        return result

    def run_phase_maintenance(self) -> Dict[str, Any]:
        """定期维护——检查所有技能的阶段转移条件

        建议每 10 分钟调用一次
        """
        now = time.time()
        if (now - self._last_phase_check) < self._phase_check_interval_sec:
            return {"skipped": True}
        self._last_phase_check = now
        transitions = 0
        retirements = 0
        for sid, rec in list(self._skills.items()):
            old_phase = rec.state.phase
            self._check_phase_transitions(rec)
            if rec.state.phase != old_phase:
                transitions += 1
            if rec.state.phase == SkillPhase.RETIRED:
                retirements += 1
        self._recalc_global_stats()
        return {
            "transitions": transitions,
            "retirements": retirements,
            "total": len(self._skills),
        }

    # ──── 内部方法 ────

    def _transition_to(
        self, rec: SkillRecord, target: SkillPhase, reason: str
    ) -> None:
        now = time.time()
        rec.state.phase = target
        rec.state.entered_at = now
        rec.state.consecutive_failures = 0
        rec.state.consecutive_successes = 0
        logger.debug(
            f"[Skill生命周期] {
                rec.metadata.name}: {
                rec.state.phase.value} ({reason})"
        )

    def _check_phase_transitions(self, rec: SkillRecord) -> None:
        """检查并执行阶段转移"""
        phase = rec.state.phase
        now = time.time()
        if phase == SkillPhase.TRIAL:
            if rec.state.trial_budget_remaining <= 0:
                sr = self._get_success_rate(rec)
                if sr >= self.STABILIZATION_MIN_SUCCESS_RATE:
                    self._transition_to(
                        rec,
                        SkillPhase.EVALUATING,
                        f"trial_complete(sr={sr:.2f})",
                    )
                else:
                    if rec.state.consecutive_failures >= 3:
                        self._transition_to(
                            rec,
                            SkillPhase.PROPOSED,
                            "trial_failed_retry_propose",
                        )
                    else:
                        self._transition_to(
                            rec,
                            SkillPhase.DECLINING,
                            "trial_exhausted_declining",
                        )
            elif rec.state.total_usage_count >= 5:
                sr = self._get_success_rate(rec)
                if sr >= 0.80:
                    self._transition_to(
                        rec,
                        SkillPhase.EVALUATING,
                        f"early_promote(sr={sr:.2f})",
                    )
        elif phase == SkillPhase.EVALUATING:
            if rec.state.total_usage_count >= 15:
                sr = self._get_success_rate(rec)
                if sr >= self.STABILIZATION_MIN_SUCCESS_RATE:
                    self._transition_to(
                        rec, SkillPhase.STABILIZED, f"stabilized(sr={sr:.2f})"
                    )
                elif sr < 0.40:
                    self._transition_to(
                        rec,
                        SkillPhase.DECLINING,
                        f"evaluated_poorly(sr={sr:.2f})",
                    )
        elif phase == SkillPhase.STABILIZED:
            sr = self._get_success_rate(rec)
            if sr < self.AUTO_RETIRE_SUCCESS_THRESHOLD:
                days_since_use = 0.0
                if rec.state.last_used_at > 0:
                    days_since_use = (now - rec.state.last_used_at) / 86400.0
                if days_since_use > self.DECLINE_INACTIVITY_DAYS:
                    self._transition_to(
                        rec,
                        SkillPhase.DECLINING,
                        f"inactive_{
                            days_since_use:.0f}d",
                    )
            elif rec.state.consecutive_failures >= 5:
                self._transition_to(
                    rec, SkillPhase.DECLINING, "streak_failures"
                )
        elif phase == SkillPhase.DECLINING:
            days_inactive = 0.0
            if rec.state.last_used_at > 0:
                days_inactive = (now - rec.state.last_used_at) / 86400.0
            if days_inactive > self.DECLINE_INACTIVITY_DAYS * 2.0:
                self._transition_to(
                    rec,
                    SkillPhase.RETIRED,
                    f"retired_after_{
                        days_inactive:.0f}d",
                )
                rec.is_active = False
            elif rec.state.consecutive_failures >= 8:
                self._transition_to(
                    rec, SkillPhase.RETIRED, "persistent_failure"
                )
                rec.is_active = False

    def _reset_counters_if_needed(self, rec: SkillRecord, now: float) -> None:
        """重置小时/天/周计数器"""
        ws = rec.window_start
        if (now - ws) >= 3600:
            rec.usage_counter["current_hour"] = 0
        if (now - ws) >= 86400:
            rec.usage_counter["current_day"] = 0
        if (now - ws) >= 604800:
            rec.usage_counter["current_week"] = 0
            rec.window_start = now

    def _get_success_rate(self, rec: SkillRecord) -> float:
        total = (
            rec.state.success_count
            + rec.state.failure_count
            + rec.state.partial_count
        )
        if total == 0:
            return rec.state.avg_confidence
        weighted = (
            rec.state.success_count * 1.0 + rec.state.partial_count * 0.5
        ) / total
        return max(0.0, min(1.0, weighted))

    def _recalc_global_stats(self) -> None:
        total_s = sum(1 for r in self._skills.values() if r.is_active)
        if total_s == 0:
            return
        total_success = 0
        total_attempts = 0
        dist: Dict[str, int] = {}
        for r in self._skills.values():
            if not r.is_active:
                continue
            total_success += r.state.success_count
            total_attempts += r.state.total_usage_count
            pv = r.state.phase.value
            dist[pv] = dist.get(pv, 0) + 1
        self._global_stats["total_skills"] = total_s
        self._global_stats["avg_success_rate"] = total_success / max(
            1, total_attempts
        )
        self._global_stats["phase_distribution"] = dist


_slm_instance: Optional[SkillLifecycleManager] = None


def get_skill_lifecycle_manager() -> SkillLifecycleManager:
    global _slm_instance
    if _slm_instance is None:
        _slm_instance = SkillLifecycleManager()
        logger.info("[Skill生命周期] 初始化完成 — 六段完整生命周期管理")
    return _slm_instance

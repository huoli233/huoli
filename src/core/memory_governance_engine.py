import time
import math
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from src.common.logger import get_logger

logger = get_logger("记忆治理")

_governance_instances: Dict[str, "MemoryGovernanceEngine"] = {}

# ═══════════════════════════════════════════════════
#  记忆重要性分级与操作枚举
# ═══════════════════════════════════════════════════


class MemoryImportanceTier(Enum):
    """记忆重要性层级"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TRIVIAL = "trivial"

    def eviction_priority(self) -> int:
        return {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
            "trivial": 4,
        }.get(self.value, 2)

    def retention_multiplier(self) -> float:
        return {
            "critical": 5.0,
            "high": 3.0,
            "medium": 1.5,
            "low": 0.7,
            "trivial": 0.3,
        }.get(self.value, 1.0)

    def context_window_weight(self) -> float:
        return {
            "critical": 1.0,
            "high": 0.85,
            "medium": 0.55,
            "low": 0.25,
            "trivial": 0.08,
        }.get(self.value, 0.5)

    def label(self) -> str:
        return {
            "critical": "核心记忆",
            "high": "重要记忆",
            "medium": "一般记忆",
            "low": "次要记忆",
            "trivial": "琐碎记忆",
        }.get(self.value, "")


class MemoryOperation(Enum):
    """记忆操作类型"""

    INSERT = "insert"
    UPDATE = "update"
    ACCESS = "access"
    EVICT = "evict"
    MERGE = "merge"
    PROMOTE = "promote"
    DEMOTE = "demote"


@dataclass
class MemoryEntry:
    """单条记忆条目"""

    memory_id: str = ""
    content: str = ""
    source_user: str = ""
    source_type: str = "message"
    topic_tags: List[str] = field(default_factory=list)
    importance: MemoryImportanceTier = MemoryImportanceTier.MEDIUM
    importance_score: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    decay_factor: float = 1.0
    is_merged: bool = False
    merged_from: List[str] = field(default_factory=list)
    size_estimate: int = 0
    emotional_valence: float = 0.0
    channel_context: str = ""

    def age_sec(self) -> float:
        return time.time() - self.created_at

    def since_last_access_sec(self) -> float:
        return time.time() - self.last_accessed_at

    def effective_score(self) -> float:
        base = self.importance_score * self.importance.retention_multiplier()
        time_decay = math.exp(
            -self.since_last_access_sec() / (3600.0 * self.decay_factor)
        )
        access_boost = min(
            1.5, 1.0 + math.log(max(1, self.access_count)) * 0.1
        )
        return max(0.001, base * time_decay * access_boost)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.memory_id[:12],
            "content_preview": self.content[:40],
            "tier": self.importance.label(),
            "score": round(self.effective_score(), 4),
            "age_min": round(self.age_sec() / 60.0, 1),
            "accesses": self.access_count,
            "merged": self.is_merged,
        }


@dataclass
class GovernancePolicy:
    """治理策略配置"""

    max_total_entries: int = 500
    max_critical_entries: int = 20
    max_high_entries: int = 60
    soft_eviction_threshold: float = 0.85
    hard_eviction_threshold: float = 0.95
    merge_similarity_threshold: float = 0.75
    merge_min_fragment_size: int = 10
    auto_promote_access_count: int = 8
    auto_demote_age_hours: float = 72.0
    default_decay_factor: float = 6.0
    critical_topic_keywords: List[str] = field(
        default_factory=lambda: [
            "重要",
            "约定",
            "承诺",
            "秘密",
            "生日",
            "纪念日",
        ]
    )
    trivial_keywords: List[str] = field(
        default_factory=lambda: ["哈哈", "嗯嗯", "好的", "哦", "啊", "表情"]
    )


@dataclass
class WindowSizingResult:
    """动态窗口裁剪结果"""

    original_count: int = 0
    window_size: int = 0
    selected_entries: List[MemoryEntry] = field(default_factory=list)
    evicted_entries: List[MemoryEntry] = field(default_factory=list)
    total_chars: int = 0
    tier_distribution: Dict[str, int] = field(default_factory=dict)
    cutoff_reason: str = ""
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original_count,
            "window_size": self.window_size,
            "selected": len(self.selected_entries),
            "evicted": len(self.evicted_entries),
            "total_chars": self.total_chars,
            "tier_dist": dict(self.tier_distribution),
            "reason": self.cutoff_reason[:60],
        }


@dataclass
class GovernanceSnapshot:
    """治理引擎快照"""

    total_entries: int = 0
    tier_counts: Dict[str, int] = field(default_factory=dict)
    utilization_ratio: float = 0.0
    avg_effective_score: float = 0.0
    oldest_entry_age_hr: float = 0.0
    most_accessed_content: str = ""
    pending_evictions: int = 0
    last_operation: str = ""
    last_operation_time: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total_entries,
            "tiers": dict(self.tier_counts),
            "utilization": round(self.utilization_ratio, 3),
            "avg_score": round(self.avg_effective_score, 4),
            "oldest_hr": round(self.oldest_entry_age_hr, 1),
            "pending_evict": self.pending_evictions,
            "last_op": self.last_operation,
        }


class MemoryGovernanceEngine:
    """记忆动态窗口与过载保护引擎
    实现文档描述的"记忆激活层"能力：
    1. 容量感知：总条目上限+各层级独立上限，达到阈值自动触发淘汰
    2. 重要性分级：五级分层(CRITICAL→TRIVIAL)，每级有不同的保留倍率/窗口权重/淘汰优先级
    3. 自动淘汰：基于有效得分(重要性×时间衰减×访问加成)排序，软阈值预警/硬阈值强制清理
    4. 碎片合并：相似度超过阈值的短记忆合并为一条，减少碎片化
    5. 动态窗口：根据当前能量状态和话题相关性动态调整上下文窗口大小和内容选择
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._entries: Dict[str, MemoryEntry] = {}
        self._policy = GovernancePolicy()
        self._topic_index: Dict[str, Set[str]] = {}
        self._user_index: Dict[str, Set[str]] = {}
        self._operation_log: List[Tuple[float, MemoryOperation, str]] = []
        self._max_op_log = 200
        self._last_governance_ts = 0.0
        self._governance_interval_sec = 300.0

    # ═════════════════ 记忆注册与查询 ═════════════════

    def register_memory(
        self,
        content: str,
        *,
        source_user: str = "",
        source_type: str = "message",
        topic_tags: Optional[List[str]] = None,
        importance_hint: Optional[str] = None,
        emotional_valence: float = 0.0,
        channel_context: str = "",
    ) -> MemoryEntry:
        mid = self._gen_id("mem")
        imp_tier = self._classify_importance(content, importance_hint or "")
        imp_score = self._compute_importance_score(
            content, imp_tier, emotional_valence
        )
        entry = MemoryEntry(
            memory_id=mid,
            content=content[:2000],
            source_user=source_user,
            source_type=source_type,
            topic_tags=topic_tags or [],
            importance=imp_tier,
            importance_score=imp_score,
            size_estimate=len(content.encode("utf-8")),
            emotional_valence=max(-1.0, min(1.0, emotional_valence)),
            channel_context=channel_context[:100],
        )
        self._entries[mid] = entry
        self._index_entry(entry)
        self._log_operation(MemoryOperation.INSERT, mid)
        logger.debug(
            f"[{self._channel_id}] 记忆注册: {mid[:12]} "
            f"tier={imp_tier.label()}, score={imp_score:.3f}"
        )
        if (
            len(self._entries)
            > self._policy.max_total_entries
            * self._policy.soft_eviction_threshold
        ):
            self._run_governance_cycle()
        return entry

    def access_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        entry = self._entries.get(memory_id)
        if entry is None:
            return None
        entry.last_accessed_at = time.time()
        entry.access_count += 1
        self._log_operation(MemoryOperation.ACCESS, memory_id)
        if (
            entry.access_count >= self._policy.auto_promote_access_count
            and entry.importance != MemoryImportanceTier.CRITICAL
        ):
            self._promote_entry(entry)
        return entry

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        return self._entries.get(memory_id)

    def query_by_topic(self, topic: str, limit: int = 20) -> List[MemoryEntry]:
        mids = self._topic_index.get(topic.lower(), set())
        results = []
        for mid in mids:
            e = self._entries.get(mid)
            if e:
                results.append(e)
        results.sort(key=lambda x: x.effective_score(), reverse=True)
        return results[:limit]

    def query_by_user(
        self, user_id: str, limit: int = 20
    ) -> List[MemoryEntry]:
        mids = self._user_index.get(user_id, set())
        results = []
        for mid in mids:
            e = self._entries.get(mid)
            if e:
                results.append(e)
        results.sort(key=lambda x: x.effective_score(), reverse=True)
        return results[:limit]

    def query_by_recency(self, limit: int = 30) -> List[MemoryEntry]:
        all_entries = list(self._entries.values())
        all_entries.sort(key=lambda x: x.created_at, reverse=True)
        return all_entries[:limit]

    def query_by_importance(
        self, tier: Optional[MemoryImportanceTier] = None, limit: int = 50
    ) -> List[MemoryEntry]:
        all_entries = list(self._entries.values())
        if tier:
            all_entries = [e for e in all_entries if e.importance == tier]
        all_entries.sort(key=lambda x: x.effective_score(), reverse=True)
        return all_entries[:limit]

    # ═════════════════ 动态窗口裁剪 ═════════════════

    def build_dynamic_window(
        self,
        *,
        max_chars: int = 4000,
        energy_ratio: float = 1.0,
        current_topics: Optional[List[str]] = None,
        focus_user: Optional[str] = None,
    ) -> WindowSizingResult:
        all_entries = list(self._entries.values())
        if not all_entries:
            return WindowSizingResult(
                original_count=0, window_size=0, cutoff_reason="无记忆"
            )
        scored = [(e, e.effective_score()) for e in all_entries]
        topics_set = set(t.lower() for t in (current_topics or []))
        for e, s in scored:
            _topic_bonus = 0.0
            if topics_set and e.topic_tags:
                hit = sum(1 for t in e.topic_tags if t.lower() in topics_set)
                if hit > 0:
                    _topic_bonus = min(0.30, hit * 0.10)
            if focus_user and e.source_user == focus_user:
                _topic_bonus += 0.15
            _energy_mod = 0.8 + energy_ratio * 0.4
            s_adj = s * (1.0 + _topic_bonus) * _energy_mod
        scored.sort(key=lambda x: x[1], reverse=True)
        selected: List[MemoryEntry] = []
        total_c = 0
        evicted: List[MemoryEntry] = []
        tier_dist: Dict[str, int] = {}
        for e, _s in scored:
            est_size = e.size_estimate or len(e.content.encode("utf-8"))
            if (
                total_c + est_size <= max_chars
                or e.importance == MemoryImportanceTier.CRITICAL
            ):
                if (
                    total_c + est_size > max_chars
                    and e.importance != MemoryImportanceTier.CRITICAL
                ):
                    evicted.append(e)
                    continue
                selected.append(e)
                total_c += est_size
                lv = e.importance.value
                tier_dist[lv] = tier_dist.get(lv, 0) + 1
            else:
                evicted.append(e)
        reason_parts = []
        if energy_ratio < 0.3:
            reason_parts.append(f"低能量({energy_ratio:.2f})压缩窗口")
        if focus_user:
            reason_parts.append(f"聚焦用户={focus_user[:8]}")
        if topics_set:
            reason_parts.append(f"话题加权({len(topics_set)}个)")
        reason = "；".join(reason_parts) if reason_parts else "正常窗口"
        return WindowSizingResult(
            original_count=len(all_entries),
            window_size=len(selected),
            selected_entries=selected,
            evicted_entries=evicted,
            total_chars=total_c,
            tier_distribution=tier_dist,
            cutoff_reason=reason,
        )

    def build_window_text(
        self, window: WindowSizingResult, *, max_output_chars: int = 3500
    ) -> str:
        parts: List[str] = []
        current_len = 0
        for entry in window.selected_entries:
            prefix = f"[{entry.importance.label()}]"
            text = f"{prefix} {entry.content[:150]}"
            if current_len + len(text) > max_output_chars:
                break
            parts.append(text)
            current_len += len(text) + 1
        return "\n".join(parts)

    # ═════════════════ 治理周期（淘汰/合并/升降级） ═════════════════

    def run_governance_cycle(self, force: bool = False) -> GovernanceSnapshot:
        now = time.time()
        if (
            not force
            and (now - self._last_governance_ts)
            < self._governance_interval_sec
        ):
            return self._build_snapshot()
        self._last_governance_ts = now
        evicted = self._run_eviction()
        merged = self._run_merge()
        promoted = self._run_auto_promote()
        demoted = self._run_auto_demote()
        snap = self._build_snapshot()
        snap.pending_evictions = len(evicted)
        op_parts = []
        if evicted:
            op_parts.append(f"淘汰{len(evicted)}条")
        if merged:
            op_parts.append(f"合并{len(merged)}组")
        if promoted:
            op_parts.append(f"升级{len(promoted)}条")
        if demoted:
            op_parts.append(f"降级{len(demoted)}条")
        snap.last_operation = "；".join(op_parts) if op_parts else "无需操作"
        logger.info(
            f"[{self._channel_id}][记忆治理] {snap.last_operation} | "
            f"总量={snap.total_entries}, 利用率={snap.utilization_ratio:.2%}"
        )
        return snap

    def _run_eviction(self) -> List[str]:
        total = len(self._entries)
        if (
            total
            <= self._policy.max_total_entries
            * self._policy.soft_eviction_threshold
        ):
            return []
        tier_limits = {
            MemoryImportanceTier.CRITICAL: self._policy.max_critical_entries,
            MemoryImportanceTier.HIGH: self._policy.max_high_entries,
        }
        tier_counts: Dict[MemoryImportanceTier, int] = {}
        for e in self._entries.values():
            tc = tier_counts.get(e.importance, 0)
            tier_counts[e.importance] = tc + 1
        candidates: List[Tuple[str, float]] = []
        for mid, entry in self._entries.items():
            limit = tier_limits.get(entry.importance)
            if limit and tier_counts.get(entry.importance, 0) > limit:
                if entry.importance not in (MemoryImportanceTier.CRITICAL,):
                    candidates.append((mid, entry.effective_score()))
        if (
            not candidates
            and total
            >= self._policy.max_total_entries
            * self._policy.hard_eviction_threshold
        ):
            for mid, entry in self._entries.items():
                if entry.importance in (
                    MemoryImportanceTier.LOW,
                    MemoryImportanceTier.TRIVIAL,
                ):
                    candidates.append((mid, entry.effective_score()))
        candidates.sort(key=lambda x: x[1])
        need_remove = total - int(
            self._policy.max_total_entries
            * self._policy.soft_eviction_threshold
        )
        evicted_ids = []
        for mid, _score in candidates[: max(need_remove, len(candidates))]:
            entry = self._entries.pop(mid, None)
            if entry:
                self._unindex_entry(entry)
                evicted_ids.append(mid)
                self._log_operation(MemoryOperation.EVICT, mid)
        return evicted_ids

    def _run_merge(self) -> List[List[str]]:
        low_tier_entries = [
            e
            for e in self._entries.values()
            if e.importance
            in (MemoryImportanceTier.LOW, MemoryImportanceTier.TRIVIAL)
            and not e.is_merged
            and (e.size_estimate or 0)
            < self._policy.merge_min_fragment_size * 2
        ]
        if len(low_tier_entries) < 2:
            return []
        merged_groups: List[List[str]] = []
        processed: Set[str] = set()
        for i, e1 in enumerate(low_tier_entries):
            if e1.memory_id in processed:
                continue
            group = [e1.memory_id]
            for e2 in low_tier_entries[i + 1:]:
                if e2.memory_id in processed:
                    continue
                sim = self._compute_similarity(e1, e2)
                if sim >= self._policy.merge_similarity_threshold:
                    group.append(e2.memory_id)
                    processed.add(e2.memory_id)
            if len(group) >= 2:
                self._merge_group(group)
                merged_groups.append(group)
                for gid in group:
                    processed.add(gid)
        return merged_groups

    def _merge_group(self, ids: List[str]) -> None:
        entries = [self._entries[mid] for mid in ids if mid in self._entries]
        if len(entries) < 2:
            return
        primary = entries[0]
        combined_content = "；".join(e.content for e in entries)
        combined_tags = list(set(t for e in entries for t in e.topic_tags))
        primary.content = combined_content[:2000]
        primary.topic_tags = combined_tags
        primary.is_merged = True
        primary.merged_from = [e.memory_id for e in entries[1:]]
        primary.size_estimate = len(combined_content.encode("utf-8"))
        primary.importance_score = max(e.importance_score for e in entries)
        for e in entries[1:]:
            self._unindex_entry(e)
            self._entries.pop(e.memory_id, None)
            self._log_operation(MemoryOperation.MERGE, e.memory_id)
        logger.debug(
            f"[{self._channel_id}] 记忆合并: {len(ids)}条 → {primary.memory_id[:12]}"
        )

    def _run_auto_promote(self) -> List[str]:
        promoted = []
        for entry in self._entries.values():
            if entry.access_count >= self._policy.auto_promote_access_count:
                old_tier = entry.importance
                if old_tier == MemoryImportanceTier.TRIVIAL:
                    entry.importance = MemoryImportanceTier.LOW
                    promoted.append(entry.memory_id)
                    self._log_operation(
                        MemoryOperation.PROMOTE, entry.memory_id
                    )
                elif old_tier == MemoryImportanceTier.LOW:
                    entry.importance = MemoryImportanceTier.MEDIUM
                    promoted.append(entry.memory_id)
                    self._log_operation(
                        MemoryOperation.PROMOTE, entry.memory_id
                    )
                elif (
                    old_tier == MemoryImportanceTier.MEDIUM
                    and entry.access_count
                    >= self._policy.auto_promote_access_count * 2
                ):
                    entry.importance = MemoryImportanceTier.HIGH
                    promoted.append(entry.memory_id)
                    self._log_operation(
                        MemoryOperation.PROMOTE, entry.memory_id
                    )
        return promoted

    def _run_auto_demote(self) -> List[str]:
        demoted = []
        threshold_hr = self._policy.auto_demote_age_hours
        for entry in self._entries.values():
            if entry.importance in (
                MemoryImportanceTier.CRITICAL,
                MemoryImportanceTier.HIGH,
            ):
                continue
            if (
                entry.age_sec() > threshold_hr * 3600
                and entry.access_count <= 1
            ):
                old = entry.importance
                if old == MemoryImportanceTier.MEDIUM:
                    entry.importance = MemoryImportanceTier.LOW
                    demoted.append(entry.memory_id)
                    self._log_operation(
                        MemoryOperation.DEMOTE, entry.memory_id
                    )
                elif old == MemoryImportanceTier.LOW:
                    entry.importance = MemoryImportanceTier.TRIVIAL
                    demoted.append(entry.memory_id)
                    self._log_operation(
                        MemoryOperation.DEMOTE, entry.memory_id
                    )
        return demoted

    def _promote_entry(self, entry: MemoryEntry) -> None:
        promotion_map = {
            MemoryImportanceTier.TRIVIAL: MemoryImportanceTier.LOW,
            MemoryImportanceTier.LOW: MemoryImportanceTier.MEDIUM,
            MemoryImportanceTier.MEDIUM: MemoryImportanceTier.HIGH,
            MemoryImportanceTier.HIGH: MemoryImportanceTier.CRITICAL,
        }
        new_tier = promotion_map.get(entry.importance)
        if new_tier and new_tier != entry.importance:
            entry.importance = new_tier
            self._log_operation(MemoryOperation.PROMOTE, entry.memory_id)
            logger.debug(
                f"[{self._channel_id}] 记忆升级: {entry.memory_id[:12]} → {new_tier.label()}"
            )

    # ═════════════════ 快照与统计 ═════════════════

    def get_snapshot(self) -> GovernanceSnapshot:
        return self._build_snapshot()

    def _build_snapshot(self) -> GovernanceSnapshot:
        entries = list(self._entries.values())
        tier_counts: Dict[str, int] = {}
        total_score = 0.0
        oldest_age = 0.0
        top_entry = None
        top_score = -1.0
        for e in entries:
            tv = e.importance.value
            tier_counts[tv] = tier_counts.get(tv, 0) + 1
            sc = e.effective_score()
            total_score += sc
            age = e.age_sec() / 3600.0
            if age > oldest_age:
                oldest_age = age
            if sc > top_score:
                top_score = sc
                top_entry = e
        n = len(entries) or 1
        last_op = ""
        last_op_time = 0.0
        if self._operation_log:
            last_op, last_op_time = (
                self._operation_log[-1][1],
                self._operation_log[-1][0],
            )
        return GovernanceSnapshot(
            total_entries=len(entries),
            tier_counts=tier_counts,
            utilization_ratio=len(entries)
            / max(1, self._policy.max_total_entries),
            avg_effective_score=total_score / n,
            oldest_entry_age_hr=oldest_age,
            most_accessed_content=(
                top_entry.content[:50] if top_entry else ""
            ),
            last_operation=(
                str(last_op.value)
                if hasattr(last_op, "value")
                else str(last_op)
            ),
            last_operation_time=last_op_time,
        )

    # ────────────────── 内部方法 ──────────────────

    @staticmethod
    def _classify_importance(content: str, hint: str) -> MemoryImportanceTier:
        content_lower = content.lower()
        hint_lower = hint.lower()
        critical_kw = [
            "重要",
            "约定",
            "承诺",
            "秘密",
            "生日",
            "纪念日",
            "决定",
            "计划",
        ]
        high_kw = [
            "喜欢",
            "讨厌",
            "开心",
            "难过",
            "生气",
            "担心",
            "感谢",
            "对不起",
        ]
        trivial_kw = [
            "哈哈",
            "嗯嗯",
            "好的",
            "哦",
            "啊",
            "表情",
            "🤣",
            "😂",
            "👍",
            "❤",
        ]
        if hint_lower in ("critical", "核心"):
            return MemoryImportanceTier.CRITICAL
        if hint_lower in ("high", "重要"):
            return MemoryImportanceTier.HIGH
        if hint_lower in ("low", "次要"):
            return MemoryImportanceTier.LOW
        if hint_lower in ("trivial", "琐碎"):
            return MemoryImportanceTier.TRIVIAL
        for kw in critical_kw:
            if kw in content:
                return MemoryImportanceTier.CRITICAL
        for kw in high_kw:
            if kw in content:
                return MemoryImportanceTier.HIGH
        for kw in trivial_kw:
            if kw in content:
                return MemoryImportanceTier.TRIVIAL
        if len(content) < 8:
            return MemoryImportanceTier.TRIVIAL
        elif len(content) < 25:
            return MemoryImportanceTier.LOW
        return MemoryImportanceTier.MEDIUM

    @staticmethod
    def _compute_importance_score(
        content: str, tier: MemoryImportanceTier, emotion: float
    ) -> float:
        base_scores = {
            MemoryImportanceTier.CRITICAL: 0.90,
            MemoryImportanceTier.HIGH: 0.70,
            MemoryImportanceTier.MEDIUM: 0.45,
            MemoryImportanceTier.LOW: 0.22,
            MemoryImportanceTier.TRIVIAL: 0.08,
        }
        score = base_scores.get(tier, 0.45)
        length_factor = min(1.0, len(content) / 80.0)
        score *= 0.6 + length_factor * 0.4
        emotion_abs = abs(emotion)
        if emotion_abs > 0.6:
            score *= 1.0 + emotion_abs * 0.2
        return max(0.01, min(1.0, score))

    @staticmethod
    def _compute_similarity(a: MemoryEntry, b: MemoryEntry) -> float:
        if a.source_user != b.source_user:
            user_match = 0.0
        else:
            user_match = 0.3 if a.source_user else 0.0
        tags_a = set(a.topic_tags)
        tags_b = set(b.topic_tags)
        if tags_a and tags_b:
            intersection = len(tags_a & tags_b)
            union = len(tags_a | tags_b)
            tag_sim = intersection / max(1, union)
        else:
            tag_sim = 0.0
        content_a = a.content[:60].lower()
        content_b = b.content[:60].lower()
        if content_a and content_b:
            words_a = set(content_a)
            words_b = set(content_b)
            common = len(words_a & words_b)
            content_sim = common / max(1, min(len(words_a), len(words_b)))
        else:
            content_sim = 0.0
        return user_match * 0.15 + tag_sim * 0.35 + content_sim * 0.50

    def _index_entry(self, entry: MemoryEntry) -> None:
        for tag in entry.topic_tags:
            key = tag.lower()
            if key not in self._topic_index:
                self._topic_index[key] = set()
            self._topic_index[key].add(entry.memory_id)
        if entry.source_user:
            if entry.source_user not in self._user_index:
                self._user_index[entry.source_user] = set()
            self._user_index[entry.source_user].add(entry.memory_id)

    def _unindex_entry(self, entry: MemoryEntry) -> None:
        for tag in entry.topic_tags:
            key = tag.lower()
            bucket = self._topic_index.get(key)
            if bucket:
                bucket.discard(entry.memory_id)
                if not bucket:
                    del self._topic_index[key]
        if entry.source_user:
            bucket = self._user_index.get(entry.source_user)
            if bucket:
                bucket.discard(entry.memory_id)
                if not bucket:
                    del self._user_index[entry.source_user]

    def _log_operation(self, op: MemoryOperation, target_id: str) -> None:
        self._operation_log.append((time.time(), op, target_id))
        if len(self._operation_log) > self._max_op_log:
            self._operation_log = self._operation_log[-self._max_op_log:]

    @staticmethod
    def _gen_id(prefix: str = "") -> str:
        raw = f"{prefix}:{time.time()}:{time.monotonic_ns()}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]


def get_memory_governance_engine(channel_id: str) -> MemoryGovernanceEngine:
    if channel_id not in _governance_instances:
        _governance_instances[channel_id] = MemoryGovernanceEngine(channel_id)
    return _governance_instances[channel_id]


def remove_memory_governance_engine(channel_id: str) -> None:
    _governance_instances.pop(channel_id, None)

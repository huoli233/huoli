import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.common.logger import get_logger
from src.chat.heart_flow.dimension_protocol import (
    DimensionBase,
    DimensionScope,
    EventContext,
    TickResult,
)
from src.chat.heart_flow.vote_types import (
    DislikeVote,
    DislikeEntry,
)

logger = get_logger("dislike_registry")


# ============================================================
# 讨厌类别与严重度映射
# ============================================================
_CATEGORY_SEVERITY = {
    "insult": 5.0,
    "harassment": 7.0,
    "spam": 3.0,
    "harmful_request": 8.0,
    "privacy_violation": 6.0,
    "manipulation": 4.0,
    "false_accusation": 5.0,
    "repeated_offense": 3.0,
}

# 屏蔽阈值（讨厌度累计到此值触发拒绝回复）
_BLOCK_THRESHOLD = 30.0

# 每条记录每天的默认衰减百分比（0~1之间）
_DEFAULT_DAILY_DECAY = 0.05

# 衰减后低于此值的记录自动移除
_REMOVAL_THRESHOLD = 0.3

# 单用户记录的最大条数
_MAX_ENTRIES_PER_USER = 50


# ============================================================
# 单用户讨厌记录集合
# ============================================================
@dataclass
class UserDislikeBook:
    """单个用户的讨厌记录集"""
    entries: List[DislikeEntry] = field(default_factory=list)
    blocked_since: float = 0.0
    cumulative_severity: float = 0.0

    def add_record(self, category: str, description: str = "", severity: float = 0.0):
        """添加一条讨厌记录"""
        if severity <= 0.0:
            severity = _CATEGORY_SEVERITY.get(category, 2.0)
        entry = DislikeEntry(
            category=category,
            severity=severity,
            timestamp=time.time(),
            description=description,
            decay_rate=_DEFAULT_DAILY_DECAY,
        )
        self.entries.append(entry)
        # 容量限制：按严重程度排序，保留最严重的N条
        if len(self.entries) > _MAX_ENTRIES_PER_USER:
            self.entries.sort(key=lambda e: e.severity, reverse=True)
            self.entries = self.entries[:_MAX_ENTRIES_PER_USER]
        self._recalc_cumulative()

    def apply_time_decay(self, elapsed_sec: float):
        """按时间衰减每条记录的严重程度"""
        days = elapsed_sec / 86400.0
        if days < 0.001:
            return
        remaining = []
        for entry in self.entries:
            decay_amount = entry.severity * entry.decay_rate * days
            entry.severity = max(0.0, entry.severity - decay_amount)
            if entry.severity >= _REMOVAL_THRESHOLD:
                remaining.append(entry)
        self.entries = remaining
        self._recalc_cumulative()

    def _recalc_cumulative(self):
        """重新计算累计严重度"""
        self.cumulative_severity = sum(e.severity for e in self.entries)

    @property
    def is_blocked(self) -> bool:
        return self.cumulative_severity >= _BLOCK_THRESHOLD

    @property
    def worst_entry(self) -> Optional[DislikeEntry]:
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.severity)


# ============================================================
# D2b 维度实现：讨厌度注册表
# ============================================================
class DislikeRegistryDimension(DimensionBase):
    """
    D2b 讨厌度注册表维度。
    采用事件记录模型（而非单纯数值模型）：
      - 每次讨厌行为记录为独立条目（类别+严重度+时间戳）
      - 各条目随时间自然衰减（每天按衰减率百分比降低严重度）
      - 累计严重度达到阈值时触发屏蔽
    per-user 粒度（用户级别，跨频道生效）。
    """

    _singleton: Optional["DislikeRegistryDimension"] = None

    @classmethod
    def get_instance(cls) -> "DislikeRegistryDimension":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        # user_id → UserDislikeBook
        self._books: Dict[str, UserDislikeBook] = {}
        self._last_decay_ts: float = time.time()

    @property
    def dimension_name(self) -> str:
        return "dislike_registry"

    @property
    def scope(self) -> DimensionScope:
        return DimensionScope.PER_USER

    @property
    def needs_persistence(self) -> bool:
        return True

    @property
    def tick_interval_sec(self) -> float:
        return 60.0

    def _get_book(self, user_id: str) -> UserDislikeBook:
        if user_id not in self._books:
            self._books[user_id] = UserDislikeBook()
        return self._books[user_id]

    def tick(self, elapsed_sec: float) -> TickResult:
        """定期对所有记录进行时间衰减"""
        now = time.time()
        real_elapsed = now - self._last_decay_ts
        self._last_decay_ts = now
        if real_elapsed < 60.0:
            return TickResult(dimension_name=self.dimension_name, updated=False)
        total_removed = 0
        for user_id, book in self._books.items():
            before = len(book.entries)
            book.apply_time_decay(real_elapsed)
            total_removed += before - len(book.entries)
            # 解除屏蔽检查
            if book.blocked_since > 0 and not book.is_blocked:
                book.blocked_since = 0.0
                logger.info(f"用户 {user_id[:8]} 讨厌度降至阈值以下，解除屏蔽")
        return TickResult(
            dimension_name=self.dimension_name,
            updated=total_removed > 0,
            summary=f"衰减移除记录={total_removed}" if total_removed > 0 else "",
        )

    def on_event(self, ctx: EventContext):
        """
        响应讨厌相关事件。
        外部系统通过 raw_extras 传入讨厌信息：
          - dislike_category: 讨厌类别
          - dislike_severity: 严重度覆盖（可选）
          - dislike_description: 行为描述（可选）
        """
        if not ctx.user_id:
            return
        if ctx.event_type == "dislike_reported":
            category = ctx.raw_extras.get("dislike_category", "")
            if not category:
                return
            severity = float(ctx.raw_extras.get("dislike_severity", 0.0))
            description = str(ctx.raw_extras.get("dislike_description", ""))
            book = self._get_book(ctx.user_id)
            book.add_record(category, description, severity)
            logger.info(
                f"讨厌记录: user={ctx.user_id[:8]} cat={category} "
                f"sev={severity:.1f} total={book.cumulative_severity:.1f}"
            )
            # 检查屏蔽
            if book.is_blocked and book.blocked_since == 0.0:
                book.blocked_since = time.time()
                logger.warning(f"用户 {ctx.user_id[:8]} 讨厌度达到屏蔽阈值")
        elif ctx.event_type == "message_received":
            # 检测消息中的敌意内容（轻度惩罚）
            hostility = ctx.raw_extras.get("hostility_score", 0.0)
            if isinstance(hostility, (int, float)) and hostility > 0.7:
                book = self._get_book(ctx.user_id)
                book.add_record(
                    "detected_hostility",
                    description=f"自动检测到敌意(score={hostility:.2f})",
                    severity=hostility * 3.0,
                )

    def vote(self, ctx: EventContext) -> DislikeVote:
        """
        讨厌度投票：
          - 屏蔽时 is_blocked=True, force_refuse=True
          - 未屏蔽但有记录时，按累计严重度计算惩罚乘数
        """
        if not ctx.user_id:
            return DislikeVote()
        book = self._get_book(ctx.user_id)
        if not book.entries:
            return DislikeVote(probability_factor=1.0)
        blocked = book.is_blocked
        total_sev = book.cumulative_severity
        # 惩罚计算：用sigmoid曲线把0~30映射到0~0.5
        penalty = 0.5 / (1.0 + math.exp(-0.2 * (total_sev - 15.0)))
        prob_factor = max(0.1, 1.0 - penalty)
        # 最严重类别
        worst = book.worst_entry
        worst_cat = worst.category if worst else ""
        # 态度标签
        attitude = ""
        if blocked:
            attitude = "blocked_user"
            prob_factor = 0.0
        elif total_sev > 20:
            attitude = "dislike_high"
        elif total_sev > 10:
            attitude = "dislike_moderate"
        return DislikeVote(
            probability_factor=round(prob_factor, 3),
            force_refuse=blocked,
            attitude_tag=attitude,
            is_blocked=blocked,
            dislike_penalty=round(penalty, 3),
            total_severity=round(total_sev, 1),
            worst_category=worst_cat,
            block_threshold=_BLOCK_THRESHOLD,
            proactive_suppression=round(penalty * 0.5, 3),
            debug_reason=f"sev={total_sev:.1f} penalty={penalty:.3f} blocked={blocked}",
        )

    def serialize(self) -> dict:
        result = {}
        for user_id, book in self._books.items():
            entries_data = []
            for entry in book.entries:
                entries_data.append({
                    "cat": entry.category,
                    "sev": entry.severity,
                    "ts": entry.timestamp,
                    "desc": entry.description,
                    "decay": entry.decay_rate,
                })
            result[user_id] = {
                "entries": entries_data,
                "blocked_since": book.blocked_since,
            }
        return result

    def deserialize(self, data: dict):
        if not isinstance(data, dict):
            return
        for user_id, vals in data.items():
            if not isinstance(vals, dict):
                continue
            book = UserDislikeBook(blocked_since=float(vals.get("blocked_since", 0.0)))
            for entry_raw in vals.get("entries", []):
                if not isinstance(entry_raw, dict):
                    continue
                book.entries.append(DislikeEntry(
                    category=str(entry_raw.get("cat", "")),
                    severity=float(entry_raw.get("sev", 0.0)),
                    timestamp=float(entry_raw.get("ts", 0.0)),
                    description=str(entry_raw.get("desc", "")),
                    decay_rate=float(entry_raw.get("decay", _DEFAULT_DAILY_DECAY)),
                ))
            book._recalc_cumulative()
            self._books[user_id] = book

    def calibrate(self, offline_seconds: float):
        """离线校准：批量衰减所有记录"""
        for book in self._books.values():
            book.apply_time_decay(offline_seconds)

    def get_state_summary(self) -> dict:
        blocked_users = [uid for uid, b in self._books.items() if b.is_blocked]
        active_count = sum(1 for b in self._books.values() if b.entries)
        return {
            "dimension": self.dimension_name,
            "scope": self.scope.value,
            "total_users": len(self._books),
            "active_users": active_count,
            "blocked_users_count": len(blocked_users),
            "blocked_user_ids": [uid[:8] for uid in blocked_users[:10]],
        }

    def reset(self, user_id: str = "", channel_id: str = ""):
        if user_id:
            self._books.pop(user_id, None)

    # ---- 外部调用接口 ----

    def report_dislike(self, user_id: str, category: str, severity: float = 0.0, description: str = ""):
        """外部显式报告讨厌行为"""
        book = self._get_book(user_id)
        book.add_record(category, description, severity)
        if book.is_blocked and book.blocked_since == 0.0:
            book.blocked_since = time.time()
            logger.warning(f"用户 {user_id[:8]} 讨厌度累计触发屏蔽")

    def revoke_entry(self, user_id: str, index: int) -> bool:
        """撤销指定序号的讨厌记录"""
        book = self._books.get(user_id)
        if not book or index < 0 or index >= len(book.entries):
            return False
        removed = book.entries.pop(index)
        book._recalc_cumulative()
        logger.info(f"撤销讨厌记录: user={user_id[:8]} cat={removed.category}")
        return True

    def get_user_severity(self, user_id: str) -> float:
        """获取指定用户的当前累计讨厌度"""
        book = self._books.get(user_id)
        return book.cumulative_severity if book else 0.0

    def is_user_blocked(self, user_id: str) -> bool:
        """检查用户是否被屏蔽"""
        book = self._books.get(user_id)
        return book.is_blocked if book else False

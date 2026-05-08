from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class TurnScheduleConfig:
    """单频道 turn 调度参数。"""

    normal_limit: int = 20
    overflow_scan_limit: int = 120
    priority_keep_limit: int = 12
    regular_keep_limit: int = 8


@dataclass(frozen=True)
class TurnScheduleStats:
    """本轮消息调度结果，用于状态页和日志解释。"""

    total_window_messages: int = 0
    candidate_count: int = 0
    selected_count: int = 0
    dropped_count: int = 0
    priority_count: int = 0
    admin_count: int = 0
    overflowed: bool = False
    admin_forced: bool = False
    policy: str = "normal"
    reason: str = ""
    dropped_message_ids: tuple[str, ...] = field(default_factory=tuple)
    selected_message_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_window_messages": self.total_window_messages,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "dropped_count": self.dropped_count,
            "priority_count": self.priority_count,
            "admin_count": self.admin_count,
            "overflowed": self.overflowed,
            "admin_forced": self.admin_forced,
            "policy": self.policy,
            "reason": self.reason,
            "dropped_message_ids": list(self.dropped_message_ids),
            "selected_message_ids": list(self.selected_message_ids),
        }


def message_identity(message: Any) -> str:
    """提取稳定消息标识。"""
    return str(
        getattr(message, "message_id", None)
        or getattr(message, "id", None)
        or f"{getattr(message, 'user_id', '')}:{getattr(message, 'time', '')}:{id(message)}"
    )


def message_time(message: Any) -> float:
    try:
        return float(getattr(message, "time", 0.0) or getattr(message, "timestamp", 0.0) or 0.0)
    except Exception:
        return 0.0


def deduplicate_messages(messages: Iterable[Any], processed_ids: set[str]) -> list[Any]:
    """过滤已经处理过的消息，保持原顺序。"""
    unique: list[Any] = []
    seen: set[str] = set()
    for message in messages or []:
        mid = message_identity(message)
        if not mid or mid in seen or mid in processed_ids:
            continue
        seen.add(mid)
        unique.append(message)
    return unique


def _sorted_unique(messages: Sequence[Any]) -> list[Any]:
    seen: set[str] = set()
    ordered: list[Any] = []
    for message in sorted(messages, key=message_time):
        mid = message_identity(message)
        if mid in seen:
            continue
        seen.add(mid)
        ordered.append(message)
    return ordered


def _tail(messages: Sequence[Any], limit: int) -> list[Any]:
    if limit <= 0:
        return []
    if len(messages) <= limit:
        return list(messages)
    return list(messages[-limit:])


def schedule_turn_messages(
    messages: Sequence[Any],
    *,
    total_window_messages: int,
    is_admin_message: Callable[[Any], bool],
    is_priority_message: Callable[[Any], bool],
    config: TurnScheduleConfig | None = None,
) -> tuple[list[Any], TurnScheduleStats]:
    """在单频道时间窗内选择本轮需要进入 Heartflow 的消息。

    规则：
    - 未溢出时保持原行为，全部候选进入本轮。
    - 溢出时优先保留管理员、@/提及、命令等关键消息。
    - 普通刷屏消息只保留最新上下文，避免形成不可控 backlog。
    """
    config = config or TurnScheduleConfig()
    total_window_messages = max(int(total_window_messages or 0), len(messages or []))
    candidates = _sorted_unique(list(messages or []))
    overflowed = total_window_messages > config.normal_limit or len(candidates) > config.normal_limit

    admin_messages = [msg for msg in candidates if is_admin_message(msg)]
    priority_messages = [msg for msg in candidates if is_priority_message(msg) or is_admin_message(msg)]
    admin_forced = bool(admin_messages)

    if not overflowed:
        selected = candidates
        dropped: list[Any] = []
        policy = "normal"
        reason = "消息未超过单轮上限，保持原顺序处理"
    else:
        admin_ids = {message_identity(msg) for msg in admin_messages}
        non_admin_priority = [msg for msg in priority_messages if message_identity(msg) not in admin_ids]
        priority_budget = max(config.priority_keep_limit, len(admin_messages))
        kept_priority = _sorted_unique(
            [
                *admin_messages,
                *_tail(non_admin_priority, max(0, priority_budget - len(admin_messages))),
            ]
        )
        priority_ids = {message_identity(msg) for msg in priority_messages}
        regular_candidates = [msg for msg in candidates if message_identity(msg) not in priority_ids]
        regular_budget = max(0, min(config.regular_keep_limit, config.normal_limit - len(kept_priority)))
        kept_regular = _tail(regular_candidates, regular_budget)
        selected = _sorted_unique([*kept_priority, *kept_regular])
        selected_ids = {message_identity(msg) for msg in selected}
        dropped = [msg for msg in candidates if message_identity(msg) not in selected_ids]
        policy = "overflow_admin_priority" if admin_forced else "overflow_priority_latest"
        reason = (
            f"时间窗消息 {total_window_messages} 条超过单轮上限 {config.normal_limit}，"
            f"保留关键消息 {len(kept_priority)} 条和最新普通上下文 {len(kept_regular)} 条"
        )

    stats = TurnScheduleStats(
        total_window_messages=total_window_messages,
        candidate_count=len(candidates),
        selected_count=len(selected),
        dropped_count=len(dropped),
        priority_count=len(priority_messages),
        admin_count=len(admin_messages),
        overflowed=overflowed,
        admin_forced=admin_forced,
        policy=policy,
        reason=reason,
        dropped_message_ids=tuple(message_identity(msg) for msg in dropped),
        selected_message_ids=tuple(message_identity(msg) for msg in selected),
    )
    return selected, stats
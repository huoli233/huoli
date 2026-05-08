# ruff: noqa: E402
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat.heart_flow.turn_scheduler import (
    TurnScheduleConfig,
    deduplicate_messages,
    schedule_turn_messages,
)
from src.memory_system.memory_health import assess_database_health, get_memory_health_status


class FakeChat:
    stream_id = "regression-channel"
    chat_stream = SimpleNamespace(group_info=object())

    def __init__(self) -> None:
        self.admin_ids = {"admin-1"}

    def _admin_force_wake_ids(self) -> set[str]:
        return self.admin_ids

    def _is_human_message_obj(self, msg: Any) -> bool:
        return not getattr(msg, "is_bot", False)

    def _is_message_pinged(self, msg: Any) -> bool:
        return bool(getattr(msg, "is_at", False) or getattr(msg, "is_mentioned", False))

    def _is_message_from_force_wake_admin(self, msg: Any) -> bool:
        return bool(
            msg is not None
            and self._is_human_message_obj(msg)
            and str(getattr(msg, "user_id", "") or "") in self._admin_force_wake_ids()
        )

    def _is_priority_turn_message(self, msg: Any) -> bool:
        if self._is_message_from_force_wake_admin(msg):
            return True
        if self._is_message_pinged(msg):
            return True
        if bool(getattr(msg, "is_command", False)):
            return True
        if getattr(self.chat_stream, "group_info", None) is None:
            return self._is_human_message_obj(msg)
        return False


def msg(mid: str, uid: str, ts: float, **flags: Any) -> SimpleNamespace:
    return SimpleNamespace(message_id=mid, user_id=uid, time=ts, **flags)


def check_turn_scheduler_overflow_contract() -> dict[str, Any]:
    chat = FakeChat()
    messages = [msg(f"m{i}", f"u{i % 5}", float(i)) for i in range(45)]
    messages.insert(4, msg("admin-old", "admin-1", 4.5))
    messages.insert(16, msg("ping-mid", "u-ping", 16.5, is_at=True))
    messages.insert(25, msg("cmd-mid", "u-cmd", 25.5, is_command=True))

    selected, stats = schedule_turn_messages(
        messages,
        total_window_messages=150,
        is_admin_message=chat._is_message_from_force_wake_admin,
        is_priority_message=chat._is_priority_turn_message,
        config=TurnScheduleConfig(normal_limit=20, priority_keep_limit=8, regular_keep_limit=12),
    )
    ids = [item.message_id for item in selected]
    assert stats.overflowed is True
    assert stats.admin_forced is True
    assert stats.policy == "overflow_admin_priority"
    assert "admin-old" in ids
    assert "ping-mid" in ids
    assert "cmd-mid" in ids
    assert len(selected) <= 20
    assert stats.dropped_count > 0
    assert all(f"m{i}" in ids for i in range(33, 45))
    return {
        "overflowed": stats.overflowed,
        "admin_preserved": "admin-old" in ids,
        "priority_preserved": {"ping": "ping-mid" in ids, "command": "cmd-mid" in ids},
        "selected_count": len(selected),
        "dropped_count": stats.dropped_count,
    }


def check_dedup_contract() -> dict[str, Any]:
    processed = {"m1"}
    messages = [msg("m1", "u1", 1.0), msg("m2", "u2", 2.0), msg("m2", "u2", 2.0)]
    unique = deduplicate_messages(messages, processed)
    assert [item.message_id for item in unique] == ["m2"]
    return {"deduplicated": True, "remaining": len(unique)}


def check_private_priority_contract() -> dict[str, Any]:
    chat = FakeChat()
    chat.chat_stream = SimpleNamespace(group_info=None)
    private_msg = msg("private", "user-private", 1.0)
    assert chat._is_priority_turn_message(private_msg) is True
    return {"private_priority": True}


def check_memory_health_contract() -> dict[str, Any]:
    status = assess_database_health(raise_on_error=False)
    cached = get_memory_health_status()
    assert cached.checked_at == status.checked_at
    assert status.quick_check in {"ok", "error"} or bool(status.quick_check)
    assert isinstance(status.degraded, bool)
    return {
        "checked": status.checked_at > 0,
        "quick_check": status.quick_check,
        "degraded": status.degraded,
        "table_count": status.table_count,
    }


def main() -> None:
    results = {
        "turn_scheduler_overflow": check_turn_scheduler_overflow_contract(),
        "dedup": check_dedup_contract(),
        "private_priority": check_private_priority_contract(),
        "memory_health": check_memory_health_contract(),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
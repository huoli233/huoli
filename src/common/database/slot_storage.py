import json
import time
from typing import Any

from src.common.logger import get_logger

logger = get_logger("槽位存储")

_TABLE_READY = False


def _ensure_slot_table():
    """确保通用槽位表可用。"""
    global _TABLE_READY
    from src.common.database.database import db
    from src.common.database.database_model import PersistenceSlot

    if not _TABLE_READY:
        db.connect(reuse_if_open=True)
        db.create_tables([PersistenceSlot], safe=True)
        _TABLE_READY = True
    return PersistenceSlot


def load_slot(slot_key: str, default: Any = None) -> Any:
    """从 Huoli.db 读取一个结构化槽位。"""
    try:
        Slot = _ensure_slot_table()
        row = Slot.get_or_none(Slot.slot_key == slot_key)
        if row is None or not row.slot_value:
            return default
        return json.loads(row.slot_value)
    except Exception as exc:
        logger.debug(f"读取槽位失败 key={slot_key}: {exc}")
        return default


def save_slot(slot_key: str, payload: Any, ttl_days: int = 365) -> bool:
    """写入一个结构化槽位到 Huoli.db。"""
    try:
        Slot = _ensure_slot_table()
        now = time.time()
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        Slot.insert(
            slot_key=slot_key,
            slot_value=serialized,
            birth_ts=now,
            modify_ts=now,
            ttl_days=ttl_days,
        ).on_conflict(
            conflict_target=[Slot.slot_key],
            update={
                Slot.slot_value: serialized,
                Slot.modify_ts: now,
                Slot.ttl_days: ttl_days,
            },
        ).execute()
        return True
    except Exception as exc:
        logger.error(f"写入槽位失败 key={slot_key}: {exc}")
        return False


def delete_slot(slot_key: str) -> bool:
    """删除一个结构化槽位。"""
    try:
        Slot = _ensure_slot_table()
        Slot.delete().where(Slot.slot_key == slot_key).execute()
        return True
    except Exception as exc:
        logger.error(f"删除槽位失败 key={slot_key}: {exc}")
        return False


def slot_exists(slot_key: str) -> bool:
    """判断槽位是否存在。"""
    try:
        Slot = _ensure_slot_table()
        return Slot.select().where(Slot.slot_key == slot_key).exists()
    except Exception as exc:
        logger.debug(f"槽位存在性检查失败 key={slot_key}: {exc}")
        return False


def list_slot_keys(prefix: str = "") -> list[str]:
    """按前缀列出槽位键。"""
    try:
        Slot = _ensure_slot_table()
        query = Slot.select(Slot.slot_key)
        if prefix:
            query = query.where(Slot.slot_key.startswith(prefix))
        return [row.slot_key for row in query]
    except Exception as exc:
        logger.warning(f"列出槽位失败 prefix={prefix}: {exc}")
        return []


def delete_slots_by_prefix(prefix: str) -> int:
    """删除指定前缀下的槽位，返回删除数量。"""
    if not prefix:
        return 0
    try:
        Slot = _ensure_slot_table()
        return Slot.delete().where(Slot.slot_key.startswith(prefix)).execute()
    except Exception as exc:
        logger.error(f"批量删除槽位失败 prefix={prefix}: {exc}")
        return 0

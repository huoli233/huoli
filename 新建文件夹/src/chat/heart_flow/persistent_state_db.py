import json as _json
import time as _tm
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger

logger = get_logger("心流持久化")


# ---------------------------------------------------------------------------
#  状态分区枚举
# ---------------------------------------------------------------------------

_PARTITIONS = (
    "ignore_list",
    "emotion_snapshot",
    "psychology",
    "engagement",
    "tool_usage",
)


# ---------------------------------------------------------------------------
#  内存层
# ---------------------------------------------------------------------------


@dataclass
class _MemorySlot:
    """单条缓存"""

    value: Any = None
    dirty: bool = False
    updated_at: float = 0.0


class _MemoryLayer:
    """按分区名组织的内存缓存"""

    def __init__(self):
        self._store: Dict[str, Dict[str, _MemorySlot]] = {
            p: {} for p in _PARTITIONS
        }

    def read(self, partition: str, key: str) -> Optional[Any]:
        slot = self._store.get(partition, {}).get(key)
        return slot.value if slot else None

    def read_updated_at(self, partition: str, key: str) -> float:
        """读取条目的最后更新时间，不存在则返回 0"""
        slot = self._store.get(partition, {}).get(key)
        return slot.updated_at if slot else 0.0

    def write(self, partition: str, key: str, value: Any) -> None:
        bucket = self._store.setdefault(partition, {})
        slot = bucket.get(key)
        if slot is None:
            bucket[key] = _MemorySlot(
                value=value, dirty=True, updated_at=_tm.time()
            )
        else:
            slot.value = value
            slot.dirty = True
            slot.updated_at = _tm.time()

    def write_if_absent_or_older(
        self, partition: str, key: str, value: Any, db_ts: float
    ) -> bool:
        """仅当内存中没有该键或内存中的时间戳比 db_ts 更旧时才写入。

        返回是否实际写入。用于 DB 加载时避免覆盖更新的内存数据。
        """
        bucket = self._store.setdefault(partition, {})
        slot = bucket.get(key)
        if slot is not None and slot.updated_at >= db_ts:
            return False
        bucket[key] = _MemorySlot(value=value, dirty=False, updated_at=db_ts)
        return True

    def dirty_entries(self) -> List[tuple]:
        """返回所有脏数据 (partition, key, value)"""
        results = []
        for part, bucket in self._store.items():
            for key, slot in bucket.items():
                if slot.dirty:
                    results.append((part, key, slot.value))
        return results

    def mark_clean(self, partition: str, key: str) -> None:
        slot = self._store.get(partition, {}).get(key)
        if slot:
            slot.dirty = False

    def drop(self, partition: str, key: str) -> bool:
        """删除一个键，返回是否存在并删除"""
        bucket = self._store.get(partition, {})
        if key in bucket:
            del bucket[key]
            return True
        return False

    def all_keys(self, partition: str) -> List[str]:
        return list(self._store.get(partition, {}).keys())


# ---------------------------------------------------------------------------
#  数据库层
# ---------------------------------------------------------------------------


def _ensure_table() -> None:
    """确保持久化表存在"""
    from src.common.database import db

    db.execute_sql(
        """
        CREATE TABLE IF NOT EXISTS heartflow_state (
            partition_name TEXT NOT NULL,
            state_key TEXT NOT NULL,
            state_value TEXT NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (partition_name, state_key)
        )
    """
    )


def _flush_to_db(entries: List[tuple]) -> int:
    """将脏数据写入数据库"""
    if not entries:
        return 0
    from src.common.database import db

    written = 0
    with db.atomic():
        for partition, key, value in entries:
            serialized = (
                _json.dumps(value, ensure_ascii=False)
                if not isinstance(value, str)
                else value
            )
            db.execute_sql(
                """
                INSERT OR REPLACE INTO heartflow_state
                    (partition_name, state_key, state_value, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (partition, key, serialized, _tm.time()),
            )
            written += 1
    return written


def _load_from_db(partition: str) -> Dict[str, Any]:
    """从数据库加载某分区全部数据，返回 {key: (value, updated_at)}"""
    from src.common.database import db

    rows = db.execute_sql(
        "SELECT state_key, state_value, updated_at FROM heartflow_state WHERE partition_name = ?",
        (partition,),).fetchall()
    result = {}
    for row in rows:
        key = row[0]
        raw = row[1]
        ts = float(row[2]) if len(row) > 2 else 0.0
        try:
            result[key] = (_json.loads(raw), ts)
        except (ValueError, TypeError):
            result[key] = (raw, ts)
    return result


# ---------------------------------------------------------------------------
#  持久化状态数据库
# ---------------------------------------------------------------------------


class PersistentStateStore:
    """心流状态持久化管理器（单例）

    读写先走内存缓存，定期或手动调用 flush() 同步到 SQLite。
    """

    _sole_ref = None

    def __new__(cls):
        if cls._sole_ref is None:
            inst = super().__new__(cls)
            inst._mem = _MemoryLayer()
            inst._table_ready = False
            inst._last_flush_ts = _tm.time()
            cls._sole_ref = inst
        return cls._sole_ref

    def _ensure_ready(self) -> None:
        if not self._table_ready:
            try:
                _ensure_table()
                self._table_ready = True
            except Exception as exc:
                logger.warning(f"[持久化] 建表失败: {exc}")

    # ---- 读 ----

    def get_state(self, partition: str, key: str, default: Any = None) -> Any:
        """读取状态值，优先走内存缓存"""
        cached = self._mem.read(partition, key)
        if cached is not None:
            return cached
        self._ensure_ready()
        try:
            db_data = _load_from_db(partition)
            for k, (v, ts) in db_data.items():
                self._mem.write_if_absent_or_older(partition, k, v, ts)
            loaded_pair = db_data.get(key)
            return loaded_pair[0] if loaded_pair else default
        except Exception as exc:
            logger.debug(f"[持久化] 读取失败 {partition}/{key}: {exc}")
            return default

    # ---- 写 ----

    def set_state(self, partition: str, key: str, value: Any) -> None:
        """设置状态值（写入缓存，标记脏）"""
        self._mem.write(partition, key, value)

    # ---- 批量操作 ----

    def get_all_keys(self, partition: str) -> List[str]:
        """获取某分区所有键"""
        self._ensure_ready()
        cached_keys = set(self._mem.all_keys(partition))
        try:
            db_data = _load_from_db(partition)
            cached_keys.update(db_data.keys())  # dict keys (str) stay the same
        except Exception as e:
            logger.debug(f"[持久化] 获取分区键列表失败({partition}): {e}")
        return list(cached_keys)

    # ---- 刷盘 ----

    def flush(self) -> int:
        """将所有脏数据同步到数据库"""
        self._ensure_ready()
        dirty = self._mem.dirty_entries()
        if not dirty:
            return 0
        try:
            count = _flush_to_db(dirty)
            for partition, key, _ in dirty:
                self._mem.mark_clean(partition, key)
            self._last_flush_ts = _tm.time()
            logger.info(f"[持久化] 刷盘 {count} 条")
            return count
        except Exception as exc:
            logger.error(f"[持久化] 刷盘失败: {exc}")
            return 0

    def auto_flush(self, interval_sec: float = 60.0) -> int:
        """如果距上次刷盘超过 interval_sec 则自动刷盘"""
        if _tm.time() - self._last_flush_ts >= interval_sec:
            return self.flush()
        return 0

    # ---- 便捷方法 ----

    def is_ignored(self, user_id: str) -> bool:
        ignored = self.get_state("ignore_list", "global", [])
        return user_id in ignored if isinstance(ignored, list) else False

    def add_to_ignore(self, user_id: str) -> None:
        ignored = self.get_state("ignore_list", "global", [])
        if not isinstance(ignored, list):
            ignored = []
        if user_id not in ignored:
            ignored.append(user_id)
            self.set_state("ignore_list", "global", ignored)

    def remove_from_ignore(self, user_id: str) -> None:
        ignored = self.get_state("ignore_list", "global", [])
        if isinstance(ignored, list) and user_id in ignored:
            ignored.remove(user_id)
            self.set_state("ignore_list", "global", ignored)

    def snapshot_emotion(self, channel_id: str, emotion_data: dict) -> None:
        self.set_state("emotion_snapshot", channel_id, emotion_data)

    def get_emotion_snapshot(self, channel_id: str) -> Optional[dict]:
        val = self.get_state("emotion_snapshot", channel_id)
        return val if isinstance(val, dict) else None

    def log_tool_usage(self, tool_name: str) -> None:
        current = self.get_state("tool_usage", tool_name, 0)
        self.set_state("tool_usage", tool_name, int(current) + 1)

    # ---- engagement ----

    def save_engagement(self, user_key: str, report_dict: dict) -> None:
        """保存用户参与度快照"""
        self.set_state("engagement", user_key, report_dict)

    def load_engagement(self, user_key: str) -> Optional[dict]:
        """读取用户参与度快照"""
        val = self.get_state("engagement", user_key)
        return val if isinstance(val, dict) else None

    # ---- 通用删除 ----

    def remove_state(self, partition: str, key: str) -> None:
        """删除某个键（内存+标记删除用于下次刷盘时处理）"""
        self._mem.drop(partition, key)


# ---------------------------------------------------------------------------
#  便捷访问
# ---------------------------------------------------------------------------


def acquire_persistent_store() -> PersistentStateStore:
    return PersistentStateStore()

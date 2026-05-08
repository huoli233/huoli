import json
import time
import asyncio
import threading
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger
from src.modules.social_value.models import SocialValueRecord

logger = get_logger("社交存储")


class SocialStorage:
    """社交值存储 - 内存缓存 + 数据库持久化"""

    def __init__(self, storage_dir: str = ""):
        del storage_dir
        self._storage_dir = "Huoli.db:social_value"
        self._cache: Dict[str, SocialValueRecord] = {}
        self._dirty: set = set()
        self._lock: Optional[asyncio.Lock] = None
        self._sync_lock = threading.Lock()
        self._adapter = None

    def _ensure_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def bind_adapter(self, adapter) -> None:
        """绑定适配器，启用数据库持久化"""
        self._adapter = adapter

    async def get(
        self, user_id: str, channel_id: str
    ) -> Optional[SocialValueRecord]:
        """获取社交值记录"""
        key = self._make_key(user_id, channel_id)
        if key in self._cache:
            return self._cache[key]
        if self._adapter:
            try:
                data = await self._adapter.get_social_value(
                    user_id, channel_id
                )
                if data:
                    record = SocialValueRecord(
                        user_id=user_id,
                        channel_id=channel_id,
                        value=data.get("value", 0.0),
                        positive_dim=data.get("positive_dim", 0.0),
                        negative_dim=data.get("negative_dim", 0.0),
                        trust_value=data.get("trust_value", 0.0),
                        annoyance_value=data.get("annoyance_value", 0.0),
                        interaction_count=data.get("interaction_count", 0),
                        last_interaction=data.get("last_interaction", 0),
                        created_at=data.get("created_at", 0),
                        updated_at=data.get("updated_at", 0),
                    )
                    self._cache[key] = record
                    return record
            except Exception as e:
                logger.debug(f"Adapter 获取社交值失败: {e}")
        record = self._load_from_db(user_id, channel_id)
        if record:
            self._cache[key] = record
        return record

    def get_sync(
        self, user_id: str, channel_id: str
    ) -> Optional[SocialValueRecord]:
        """同步获取社交值记录（从内存缓存或数据库直接读取，不使用async）"""
        key = self._make_key(user_id, channel_id)
        with self._sync_lock:
            if key in self._cache:
                return self._cache[key]
            record = self._load_from_db(user_id, channel_id)
            if record:
                self._cache[key] = record
                return record
            return None

    async def set(self, user_id: str, channel_id: str, value: float) -> None:
        """设置社交值（带边界保护）"""
        value = max(-100.0, min(100.0, value))
        key = self._make_key(user_id, channel_id)
        async with self._ensure_lock():
            record = self._cache.get(key)
            if record is None:
                record = SocialValueRecord(
                    user_id=user_id,
                    channel_id=channel_id,
                    value=value,
                    interaction_count=1,
                    last_interaction=time.time(),
                    created_at=time.time(),
                    updated_at=time.time(),
                )
            else:
                record.value = value
                record.interaction_count += 1
                record.last_interaction = time.time()
                record.updated_at = time.time()
                if value >= 0:
                    record.positive_dim = max(record.positive_dim, value)
                else:
                    record.negative_dim = min(record.negative_dim, value)
            self._cache[key] = record
            self._dirty.add(key)
        if self._adapter:
            try:
                await self._adapter.set_social_value(
                    user_id,
                    channel_id,
                    {
                        "value": record.value,
                        "positive_dim": record.positive_dim,
                        "negative_dim": record.negative_dim,
                        "trust_value": record.trust_value,
                        "annoyance_value": record.annoyance_value,
                        "interaction_count": record.interaction_count,
                        "last_interaction": record.last_interaction,
                    },
                )
                self._dirty.discard(key)
                return
            except Exception as e:
                logger.debug(f"Adapter 设置社交值失败: {e}")
        await self._persist(key)

    async def get_all_for_user(self, user_id: str) -> List[SocialValueRecord]:
        """获取用户在所有频道的社交值"""
        records = []
        try:
            from src.common.database.database import db
            from src.common.database.database_model import PersistenceSlot

            db.create_tables([PersistenceSlot], safe=True)
            prefix = f"{self._slot_prefix(user_id)}:"
            rows = PersistenceSlot.select().where(PersistenceSlot.slot_key.startswith(prefix))
            for row in rows:
                record = self._record_from_payload(json.loads(row.slot_value))
                if record:
                    records.append(record)
        except Exception as exc:
            logger.debug(f"数据库读取用户社交值失败: {exc}")
        return records

    async def get_all_for_channel(
        self, channel_id: str
    ) -> List[SocialValueRecord]:
        """获取频道内所有用户的社交值"""
        records = []
        try:
            from src.common.database.database import db
            from src.common.database.database_model import PersistenceSlot

            db.create_tables([PersistenceSlot], safe=True)
            suffix = f":{self._safe_filename(channel_id)}"
            rows = PersistenceSlot.select().where(
                PersistenceSlot.slot_key.startswith("social_value:")
            )
            for row in rows:
                if not str(row.slot_key).endswith(suffix):
                    continue
                record = self._record_from_payload(json.loads(row.slot_value))
                if record:
                    records.append(record)
        except Exception as exc:
            logger.debug(f"数据库读取频道社交值失败: {exc}")
        return records

    async def flush_all(self) -> int:
        """将所有脏数据写入数据库"""
        count = 0
        dirty_keys = list(self._dirty)
        for key in dirty_keys:
            await self._persist(key)
            count += 1
        return count

    async def check_and_fix_time_sync(self) -> Dict[str, int]:
        """检查并修复时间同步问题"""
        now = time.time()
        thirty_days = 30 * 24 * 3600
        stats = {
            "long_gap_fixed": 0,
            "future_fixed": 0,
            "expired_cleaned": 0,
        }
        async with self._ensure_lock():
            for key, record in list(self._cache.items()):
                fixed = False
                if record.last_interaction > 0:
                    time_gap = now - record.last_interaction
                    if time_gap > thirty_days:
                        new_last = (
                            now
                            - (7 * 24 * 3600)
                            + (hash(key) % (7 * 24 * 3600))
                        )
                        record.last_interaction = new_last
                        if record.created_at > new_last:
                            record.created_at = new_last - 3600
                        stats["long_gap_fixed"] += 1
                        fixed = True
                if record.last_interaction > now + 3600:
                    record.last_interaction = now - 3600
                    stats["future_fixed"] += 1
                    fixed = True
                if record.created_at > now + 3600:
                    record.created_at = now - 3600
                    stats["future_fixed"] += 1
                    fixed = True
                if record.updated_at > now + 3600:
                    record.updated_at = now
                    stats["future_fixed"] += 1
                    fixed = True
                if fixed:
                    self._dirty.add(key)
        if stats["long_gap_fixed"] > 0 or stats["future_fixed"] > 0:
            logger.info(
                f"时间同步修复完成 | "
                f"长间隔修复:{stats['long_gap_fixed']} | "
                f"未来时间修复:{stats['future_fixed']} | "
                f"过期清理:{stats['expired_cleaned']}"
            )
            await self.flush_all()
        return stats

    async def cleanup_expired_records(self, max_age_days: int = 90) -> int:
        """清理过期记录"""
        now = time.time()
        max_age = max_age_days * 24 * 3600
        cleaned = 0
        async with self._ensure_lock():
            keys_to_remove = []
            for key, record in self._cache.items():
                if record.last_interaction > 0:
                    age = now - record.last_interaction
                    if age > max_age:
                        keys_to_remove.append(key)
            for key in keys_to_remove:
                del self._cache[key]
                self._dirty.discard(key)
                cleaned += 1
        if cleaned > 0:
            logger.info(f"过期记录清理完成 | 清理数:{cleaned}")
        return cleaned

    async def _persist(self, key: str) -> None:
        """持久化单条记录"""
        await asyncio.to_thread(self._persist_sync, key)

    def _load_from_db(self, user_id: str, channel_id: str) -> Optional[SocialValueRecord]:
        try:
            from src.common.database.database import db
            from src.common.database.database_model import PersistenceSlot

            db.create_tables([PersistenceSlot], safe=True)
            row = PersistenceSlot.get_or_none(
                PersistenceSlot.slot_key == self._slot_key(user_id, channel_id)
            )
            if row is None or not row.slot_value:
                return None
            return self._record_from_payload(json.loads(row.slot_value))
        except Exception as exc:
            logger.debug(f"数据库读取社交值失败: {exc}")
            return None

    def _persist_sync(self, key: str) -> None:
        record = self._cache.get(key)
        if record is None:
            return
        try:
            from src.common.database.database import db
            from src.common.database.database_model import PersistenceSlot

            db.create_tables([PersistenceSlot], safe=True)
            now = time.time()
            data = self._record_to_payload(record)
            serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            PersistenceSlot.insert(
                slot_key=self._slot_key(record.user_id, record.channel_id),
                slot_value=serialized,
                birth_ts=now,
                modify_ts=now,
                ttl_days=365,
            ).on_conflict(
                conflict_target=[PersistenceSlot.slot_key],
                update={
                    PersistenceSlot.slot_value: serialized,
                    PersistenceSlot.modify_ts: now,
                    PersistenceSlot.ttl_days: 365,
                },
            ).execute()
            self._dirty.discard(key)
        except Exception as exc:
            logger.error(f"保存社交值到数据库失败: {exc}")

    @staticmethod
    def _record_to_payload(record: SocialValueRecord) -> Dict[str, Any]:
        return {
            "user_id": record.user_id,
            "channel_id": record.channel_id,
            "value": record.value,
            "positive_dim": record.positive_dim,
            "negative_dim": record.negative_dim,
            "trust_value": record.trust_value,
            "annoyance_value": record.annoyance_value,
            "interaction_count": record.interaction_count,
            "last_interaction": record.last_interaction,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    @staticmethod
    def _record_from_payload(data: Dict[str, Any]) -> Optional[SocialValueRecord]:
        try:
            return SocialValueRecord(
                user_id=str(data.get("user_id", "") or ""),
                channel_id=str(data.get("channel_id", "") or ""),
                value=max(-100.0, min(100.0, float(data.get("value", 0.0)))),
                positive_dim=float(data.get("positive_dim", 0.0)),
                negative_dim=float(data.get("negative_dim", 0.0)),
                trust_value=max(
                    -100.0, min(100.0, float(data.get("trust_value", 0.0)))
                ),
                annoyance_value=max(
                    0.0, min(100.0, float(data.get("annoyance_value", 0.0)))
                ),
                interaction_count=int(data.get("interaction_count", 0)),
                last_interaction=float(data.get("last_interaction", 0)),
                created_at=float(data.get("created_at", 0)),
                updated_at=float(data.get("updated_at", 0)),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _make_key(user_id: str, channel_id: str) -> str:
        return f"{user_id}:{channel_id}"

    @classmethod
    def _slot_key(cls, user_id: str, channel_id: str) -> str:
        return f"{cls._slot_prefix(user_id)}:{cls._safe_filename(channel_id)}"

    @classmethod
    def _slot_prefix(cls, user_id: str) -> str:
        return f"social_value:{cls._safe_filename(user_id)}"

    @staticmethod
    def _safe_filename(s: str) -> str:
        """将字符串转为安全的文件名"""
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)

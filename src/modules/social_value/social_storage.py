import os
import json
import time
import asyncio
import threading
from typing import Dict, List, Optional
from src.common.logger import get_logger
from src.modules.social_value.models import SocialValueRecord

logger = get_logger("社交存储")


class SocialStorage:
    """社交值存储 - 内存缓存 + 文件持久化"""

    def __init__(self, storage_dir: str = "data/huoli/social"):
        self._storage_dir = storage_dir
        self._cache: Dict[str, SocialValueRecord] = {}
        self._dirty: set = set()
        self._lock: Optional[asyncio.Lock] = None
        self._sync_lock = threading.Lock()
        self._adapter = None
        os.makedirs(self._storage_dir, exist_ok=True)

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
        record = await self._load_from_disk(user_id, channel_id)
        if record:
            self._cache[key] = record
        return record

    def get_sync(
        self, user_id: str, channel_id: str
    ) -> Optional[SocialValueRecord]:
        """同步获取社交值记录（从内存缓存或磁盘直接读取，不使用async）"""
        key = self._make_key(user_id, channel_id)
        with self._sync_lock:
            if key in self._cache:
                return self._cache[key]
            filepath = os.path.join(
                self._storage_dir,
                self._safe_filename(user_id),
                self._safe_filename(channel_id) + ".json",
            )
            if not os.path.exists(filepath):
                return None
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                record = SocialValueRecord(
                    user_id=data.get("user_id", ""),
                    channel_id=data.get("channel_id", ""),
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
                self._cache[key] = record
                return record
            except (json.JSONDecodeError, KeyError, TypeError):
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
        user_dir = os.path.join(
            self._storage_dir, self._safe_filename(user_id)
        )
        if not os.path.exists(user_dir):
            return records
        for filename in os.listdir(user_dir):
            if filename.endswith(".json"):
                channel_id = filename[:-5]
                record = await self.get(user_id, channel_id)
                if record:
                    records.append(record)
        return records

    async def get_all_for_channel(
        self, channel_id: str
    ) -> List[SocialValueRecord]:
        """获取频道内所有用户的社交值"""
        records = []
        if not os.path.exists(self._storage_dir):
            return records
        for user_dir_name in os.listdir(self._storage_dir):
            user_dir = os.path.join(self._storage_dir, user_dir_name)
            if os.path.isdir(user_dir):
                filepath = os.path.join(
                    user_dir, self._safe_filename(channel_id) + ".json"
                )
                if os.path.exists(filepath):
                    record = await self._load_file(filepath)
                    if record:
                        records.append(record)
        return records

    async def flush_all(self) -> int:
        """将所有脏数据写入磁盘"""
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
        record = self._cache.get(key)
        if record is None:
            return
        user_dir = os.path.join(
            self._storage_dir, self._safe_filename(record.user_id)
        )
        os.makedirs(user_dir, exist_ok=True)
        filepath = os.path.join(
            user_dir, self._safe_filename(record.channel_id) + ".json"
        )
        data = {
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

        def _write_file():
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        await asyncio.to_thread(_write_file)
        self._dirty.discard(key)

    async def _load_from_disk(
        self, user_id: str, channel_id: str
    ) -> Optional[SocialValueRecord]:
        """从磁盘加载记录"""
        filepath = os.path.join(
            self._storage_dir,
            self._safe_filename(user_id),
            self._safe_filename(channel_id) + ".json",
        )
        return await self._load_file(filepath)

    async def _load_file(self, filepath: str) -> Optional[SocialValueRecord]:
        """从文件加载记录"""
        if not os.path.exists(filepath):
            return None

        def _read_json():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)

        try:
            data = await asyncio.to_thread(_read_json)
            return SocialValueRecord(
                user_id=data.get("user_id", ""),
                channel_id=data.get("channel_id", ""),
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
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    @staticmethod
    def _make_key(user_id: str, channel_id: str) -> str:
        return f"{user_id}:{channel_id}"

    @staticmethod
    def _safe_filename(s: str) -> str:
        """将字符串转为安全的文件名"""
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)

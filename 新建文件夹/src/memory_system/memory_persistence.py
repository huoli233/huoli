"""
记忆持久化子系统 —— 提供键值存储、版本追踪、批量迁移、过期清扫和容量诊断
融合三源设计:
  - XBcore: 脏标记优化 + 适配器可扩展架构
  - MaiBot: 版本管理 + JSON文件迁移回退
  - MIMiaoCore: 分层存储 + 单例工厂模式
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import OrderedDict
from src.common.logger import get_logger

logger = get_logger("persistence")

# 备份文件根目录(用于JSON迁移和导出)
_FALLBACK_DIR = Path(__file__).resolve().parents[2] / "data" / "memory_vault"


class _LRUWriteCache:
    """最近写入缓存，减少频繁DB写操作(XBcore脏标记思路)"""

    def __init__(self, capacity: int = 128):
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._capacity = capacity
        self._dirty_keys: set = set()

    def peek(self, key: str) -> Any:
        # 查询缓存，命中则提升位置
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: str, value: Any, mark_dirty: bool = True):
        # 写入缓存并标记脏位
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if mark_dirty:
            self._dirty_keys.add(key)
        # 容量溢出时驱逐最早条目
        while len(self._store) > self._capacity:
            evicted_key, _ = self._store.popitem(last=False)
            self._dirty_keys.discard(evicted_key)

    def pop(self, key: str):
        # 移除缓存条目
        self._store.pop(key, None)
        self._dirty_keys.discard(key)

    def drain_dirty(self) -> Dict[str, Any]:
        # 取出所有脏数据(一次性刷盘用)
        result = {}
        for k in list(self._dirty_keys):
            if k in self._store:
                result[k] = self._store[k]
        self._dirty_keys.clear()
        return result

    def has_dirty(self) -> bool:
        return len(self._dirty_keys) > 0

    def wipe(self):
        self._store.clear()
        self._dirty_keys.clear()

    @property
    def size(self) -> int:
        return len(self._store)


class KeyVaultManager:
    """
    通用键值持久化管理器
    负责将任意JSON可序列化数据以键名为索引存入数据库
    集成LRU写缓存减少IO + JSON文件自动迁移
    """

    def __init__(self, fallback_dir: str = "", cache_capacity: int = 128):
        self._fallback_path = (
            Path(fallback_dir) if fallback_dir else _FALLBACK_DIR
        )
        self._write_cache = _LRUWriteCache(capacity=cache_capacity)
        self._last_flush_ts = time.time()
        self._flush_interval = 60.0  # 脏缓存自动刷盘间隔(秒)

    def _fetch_slot_model(self):
        # 延迟导入避免循环依赖
        from src.common.database.database_model import PersistenceSlot

        return PersistenceSlot

    def _attempt_json_migration(self, key: str):
        # 检查是否有遗留JSON文件需要迁移到数据库
        if not self._fallback_path.exists():
            return
        safe_name = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        json_file = self._fallback_path / f"{safe_name}.json"
        if not json_file.exists():
            return
        try:
            raw_text = json_file.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
            self.store(key, payload)
            backup_file = json_file.with_suffix(".json.migrated")
            json_file.rename(backup_file)
            logger.info(f"键 '{key}' 已从本地JSON迁移到数据库")
        except Exception as exc:
            logger.debug(f"JSON迁移跳过(键={key}): {exc}")

    def store(self, key: str, payload: Any):
        """存储键值对到数据库(先写缓存，定期落盘)"""
        self._write_cache.put(key, payload, mark_dirty=True)
        # 判断是否需要立即刷盘
        elapsed = time.time() - self._last_flush_ts
        if elapsed >= self._flush_interval or self._write_cache.size > 64:
            self._flush_dirty_to_db()

    def _flush_dirty_to_db(self):
        """将所有脏缓存条目批量写入数据库"""
        dirty_items = self._write_cache.drain_dirty()
        if not dirty_items:
            return
        Slot = self._fetch_slot_model()
        now = time.time()
        flushed_count = 0
        for slot_key, value in dirty_items.items():
            try:
                json_str = json.dumps(value, ensure_ascii=False)
                Slot.insert(
                    slot_key=slot_key,
                    slot_value=json_str,
                    birth_ts=now,
                    modify_ts=now,
                    ttl_days=30,
                ).on_conflict(
                    conflict_target=[Slot.slot_key],
                    update={
                        Slot.slot_value: json_str,
                        Slot.modify_ts: now,
                    },
                ).execute()
                flushed_count += 1
            except Exception as exc:
                logger.error(f"持久化槽写入失败(key={slot_key}): {exc}")
        if flushed_count > 0:
            logger.debug(f"批量刷盘: {flushed_count}条脏数据已写入")
        self._last_flush_ts = time.time()

    def retrieve(self, key: str, fallback: Any = None) -> Any:
        """按键名读取数据(优先缓存→数据库→JSON迁移)"""
        # 缓存命中
        cached = self._write_cache.peek(key)
        if cached is not None:
            return cached
        # 数据库查询
        Slot = self._fetch_slot_model()
        try:
            row = Slot.get_or_none(Slot.slot_key == key)
            if row:
                parsed = json.loads(row.slot_value)
                self._write_cache.put(key, parsed, mark_dirty=False)
                return parsed
        except Exception as exc:
            logger.debug(f"持久化槽读取异常(key={key}): {exc}")
        # 尝试JSON迁移
        self._attempt_json_migration(key)
        try:
            row = Slot.get_or_none(Slot.slot_key == key)
            if row:
                parsed = json.loads(row.slot_value)
                self._write_cache.put(key, parsed, mark_dirty=False)
                return parsed
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return fallback

    def remove(self, key: str):
        """删除指定键"""
        self._write_cache.pop(key)
        Slot = self._fetch_slot_model()
        try:
            Slot.delete().where(Slot.slot_key == key).execute()
        except Exception as exc:
            logger.error(f"持久化槽删除失败(key={key}): {exc}")

    def contains(self, key: str) -> bool:
        """检查键是否存在"""
        if self._write_cache.peek(key) is not None:
            return True
        Slot = self._fetch_slot_model()
        try:
            return Slot.select().where(Slot.slot_key == key).exists()
        except Exception:
            return False

    def enumerate_keys(self) -> List[str]:
        """列出所有已存储的键名"""
        Slot = self._fetch_slot_model()
        try:
            return [
                row.slot_key for row in Slot.select(Slot.slot_key).distinct()
            ]
        except Exception:
            return []

    def force_flush(self):
        """强制将缓存中所有脏数据写入数据库"""
        self._flush_dirty_to_db()


class RevisionTracker:
    """
    版本追踪器 —— 为键值存储提供版本历史功能
    每次保存自动创建新版本快照，超出上限自动淘汰最旧版本
    """

    def __init__(self, vault: KeyVaultManager, max_snapshots: int = 5):
        self._vault = vault
        self._snapshot_cap = max(1, max_snapshots)

    def _fetch_revision_model(self):
        from src.common.database.database_model import RevisionSnapshot

        return RevisionSnapshot

    def _fetch_catalog_model(self):
        from src.common.database.database_model import SlotCatalog

        return SlotCatalog

    def _try_migrate_legacy_versions(self, key: str):
        # 检查是否有旧格式版本数据(以 _versions 后缀存储在KeyVault中)
        legacy_key = f"{key}__revisions"
        if not self._vault.contains(legacy_key):
            return
        try:
            old_versions = self._vault.retrieve(legacy_key)
            if not old_versions or not isinstance(old_versions, list):
                return
            Snap = self._fetch_revision_model()
            Cat = self._fetch_catalog_model()
            now = time.time()
            for entry in old_versions:
                Snap.insert(
                    slot_key=key,
                    revision_num=entry.get("seq", 1),
                    snapshot_value=json.dumps(
                        entry.get("payload", {}), ensure_ascii=False
                    ),
                    birth_ts=entry.get("ts", now),
                ).execute()
            Cat.insert(
                slot_key=key,
                total_revisions=len(old_versions),
                latest_access_ts=now,
                birth_ts=now,
            ).on_conflict(
                conflict_target=[Cat.slot_key],
                update={
                    Cat.total_revisions: len(old_versions),
                    Cat.latest_access_ts: now,
                },
            ).execute()
            self._vault.remove(legacy_key)
            logger.info(
                f"版本数据 '{key}' 从旧格式迁移完成({len(old_versions)}个版本)"
            )
        except Exception as exc:
            logger.debug(f"版本迁移跳过(key={key}): {exc}")

    def commit(self, key: str, payload: Any):
        """为指定键提交新版本快照"""
        Snap = self._fetch_revision_model()
        Cat = self._fetch_catalog_model()
        now = time.time()
        try:
            # 统计当前版本数量
            existing_count = Snap.select().where(Snap.slot_key == key).count()
            # 超出上限时删除最早的快照
            if existing_count >= self._snapshot_cap:
                oldest = (
                    Snap.select()
                    .where(Snap.slot_key == key)
                    .order_by(Snap.birth_ts.asc())
                    .first()
                )
                if oldest:
                    oldest.delete_instance()
                    existing_count -= 1
            next_num = existing_count + 1
            json_str = json.dumps(payload, ensure_ascii=False)
            Snap.insert(
                slot_key=key,
                revision_num=next_num,
                snapshot_value=json_str,
                birth_ts=now,
            ).execute()
            # 更新目录
            Cat.insert(
                slot_key=key,
                total_revisions=next_num,
                latest_access_ts=now,
                birth_ts=now,
            ).on_conflict(
                conflict_target=[Cat.slot_key],
                update={
                    Cat.total_revisions: next_num,
                    Cat.latest_access_ts: now,
                },
            ).execute()
        except Exception as exc:
            logger.debug(f"版本提交失败(key={key}): {exc}")

    def _load_all_snapshots(self, key: str) -> List[dict]:
        """加载指定键的全部版本列表"""
        Snap = self._fetch_revision_model()
        try:
            rows = (
                Snap.select()
                .where(Snap.slot_key == key)
                .order_by(Snap.revision_num.asc())
            )
            result = []
            for row in rows:
                try:
                    parsed = json.loads(row.snapshot_value)
                except (json.JSONDecodeError, ValueError):
                    parsed = {}
                result.append(
                    {
                        "payload": parsed,
                        "ts": row.birth_ts,
                        "seq": row.revision_num,
                    }
                )
            if not result:
                self._try_migrate_legacy_versions(key)
                rows = (
                    Snap.select()
                    .where(Snap.slot_key == key)
                    .order_by(Snap.revision_num.asc())
                )
                for row in rows:
                    try:
                        parsed = json.loads(row.snapshot_value)
                    except (json.JSONDecodeError, ValueError):
                        parsed = {}
                    result.append(
                        {
                            "payload": parsed,
                            "ts": row.birth_ts,
                            "seq": row.revision_num,
                        }
                    )
            return result
        except Exception:
            return []

    def fetch_by_number(self, key: str, revision_num: int) -> Any:
        """按版本号获取特定快照"""
        Snap = self._fetch_revision_model()
        try:
            row = Snap.get_or_none(
                (Snap.slot_key == key) & (Snap.revision_num == revision_num)
            )
            if row:
                return json.loads(row.snapshot_value)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return None

    def fetch_newest(self, key: str) -> Any:
        """获取最新版本快照"""
        Snap = self._fetch_revision_model()
        try:
            row = (
                Snap.select()
                .where(Snap.slot_key == key)
                .order_by(Snap.revision_num.desc())
                .first()
            )
            if row:
                return json.loads(row.snapshot_value)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return None

    def list_history(self, key: str) -> List[dict]:
        """列出指定键的所有版本摘要(序号+时间戳)"""
        snapshots = self._load_all_snapshots(key)
        return [{"seq": s["seq"], "ts": s["ts"]} for s in snapshots]


class BulkTransporter:
    """
    批量导入导出器 —— 将键值存储整体导出为JSON文件或从文件恢复
    """

    def __init__(self, vault: KeyVaultManager):
        self._vault = vault

    def export_to_file(self, destination: str):
        """将全部键值数据导出为单个JSON文件"""
        all_keys = self._vault.enumerate_keys()
        export_dict = {}
        for k in all_keys:
            val = self._vault.retrieve(k)
            if val is not None:
                export_dict[k] = val
        try:
            dest_path = Path(destination)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_text(
                json.dumps(export_dict, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"导出完成: {len(export_dict)}条记忆 → {destination}")
        except Exception as exc:
            logger.error(f"导出失败: {exc}")

    def import_from_file(self, source: str):
        """从JSON文件恢复全部键值数据"""
        source_path = Path(source)
        if not source_path.exists():
            logger.error(f"导入源文件不存在: {source}")
            return
        try:
            raw = source_path.read_text(encoding="utf-8")
            import_dict = json.loads(raw)
            if not isinstance(import_dict, dict):
                logger.error("导入文件格式异常: 顶层结构应为字典")
                return
            imported_count = 0
            for k, v in import_dict.items():
                self._vault.store(k, v)
                imported_count += 1
            # 确保全部写入
            self._vault.force_flush()
            logger.info(f"导入完成: {imported_count}条记忆 ← {source}")
        except Exception as exc:
            logger.error(f"导入失败: {exc}")


class StalenessReaper:
    """
    过期数据清扫器 —— 按TTL策略清除超龄键值记录
    同时支持按修改时间阈值进行精细化清理
    """

    def __init__(self, vault: KeyVaultManager, default_ttl_days: int = 30):
        self._vault = vault
        self._ttl_days = max(1, default_ttl_days)

    def reap(self) -> int:
        """执行一轮过期清扫，返回清除条数"""
        Slot = self._vault._fetch_slot_model()
        now = time.time()
        cutoff_ts = now - (self._ttl_days * 86400)
        try:
            # 查找过期且TTL已超的条目
            expired_query = Slot.select().where(Slot.modify_ts < cutoff_ts)
            candidates = list(expired_query)
            reaped = 0
            for item in candidates:
                actual_ttl = (
                    getattr(item, "ttl_days", self._ttl_days) or self._ttl_days
                )
                actual_cutoff = now - (actual_ttl * 86400)
                if item.modify_ts < actual_cutoff:
                    # 从缓存中也清除
                    self._vault._write_cache.pop(item.slot_key)
                    item.delete_instance()
                    reaped += 1
            if reaped > 0:
                logger.info(f"过期清扫: 移除{reaped}条超龄记忆")
            return reaped
        except Exception as exc:
            logger.debug(f"过期清扫执行异常: {exc}")
            return 0

    def adjust_ttl(self, days: int):
        """动态调整TTL天数"""
        self._ttl_days = max(1, days)


class VaultDiagnostics:
    """
    存储诊断工具 —— 统计键值仓库的容量、大小分布和健康度
    """

    def __init__(self, vault: KeyVaultManager):
        self._vault = vault

    def summarize(self) -> Dict[str, Any]:
        """生成仓库统计摘要"""
        Slot = self._vault._fetch_slot_model()
        try:
            all_rows = list(Slot.select())
            total_keys = len(all_rows)
            total_bytes = 0
            size_buckets = {"tiny": 0, "small": 0, "medium": 0, "large": 0}
            oldest_ts = time.time()
            newest_ts = 0.0
            for row in all_rows:
                val_size = len((row.slot_value or "").encode("utf-8"))
                total_bytes += val_size
                if val_size < 256:
                    size_buckets["tiny"] += 1
                elif val_size < 4096:
                    size_buckets["small"] += 1
                elif val_size < 65536:
                    size_buckets["medium"] += 1
                else:
                    size_buckets["large"] += 1
                if row.birth_ts < oldest_ts:
                    oldest_ts = row.birth_ts
                if row.modify_ts > newest_ts:
                    newest_ts = row.modify_ts
            return {
                "total_keys": total_keys,
                "total_bytes": total_bytes,
                "total_kb": round(total_bytes / 1024, 2),
                "size_distribution": size_buckets,
                "oldest_record_ts": oldest_ts if total_keys > 0 else 0,
                "newest_modify_ts": newest_ts if total_keys > 0 else 0,
                "cache_entries": self._vault._write_cache.size,
                "has_pending_writes": self._vault._write_cache.has_dirty(),
            }
        except Exception:
            return {"total_keys": 0, "total_bytes": 0, "total_kb": 0}

    def estimate_bloat_ratio(self) -> float:
        """估算存储膨胀率(JSON冗余比例)"""
        Slot = self._vault._fetch_slot_model()
        try:
            sample_rows = list(Slot.select().limit(50))
            if not sample_rows:
                return 0.0
            raw_total = 0
            compact_total = 0
            for row in sample_rows:
                raw_size = len((row.slot_value or "").encode("utf-8"))
                raw_total += raw_size
                try:
                    parsed = json.loads(row.slot_value)
                    compact_str = json.dumps(
                        parsed, ensure_ascii=False, separators=(",", ":")
                    )
                    compact_total += len(compact_str.encode("utf-8"))
                except (json.JSONDecodeError, ValueError):
                    compact_total += raw_size
            if compact_total == 0:
                return 0.0
            return (
                round((raw_total - compact_total) / raw_total, 4)
                if raw_total > 0
                else 0.0
            )
        except Exception:
            return 0.0


# ═══════════════════════════════════════════
# 单例管理 —— 提供模块级便捷访问入口
# ═══════════════════════════════════════════

_key_vault_ref: Optional[KeyVaultManager] = None
_revision_tracker_ref: Optional[RevisionTracker] = None
_bulk_transporter_ref: Optional[BulkTransporter] = None
_staleness_reaper_ref: Optional[StalenessReaper] = None
_vault_diagnostics_ref: Optional[VaultDiagnostics] = None


def acquire_key_vault() -> KeyVaultManager:
    """获取全局键值仓库管理器实例"""
    global _key_vault_ref
    if _key_vault_ref is None:
        _key_vault_ref = KeyVaultManager()
    return _key_vault_ref


def acquire_revision_tracker() -> RevisionTracker:
    """获取全局版本追踪器实例"""
    global _revision_tracker_ref
    if _revision_tracker_ref is None:
        _revision_tracker_ref = RevisionTracker(acquire_key_vault())
    return _revision_tracker_ref


def acquire_bulk_transporter() -> BulkTransporter:
    """获取全局批量导入导出器实例"""
    global _bulk_transporter_ref
    if _bulk_transporter_ref is None:
        _bulk_transporter_ref = BulkTransporter(acquire_key_vault())
    return _bulk_transporter_ref


def acquire_staleness_reaper() -> StalenessReaper:
    """获取全局过期清扫器实例"""
    global _staleness_reaper_ref
    if _staleness_reaper_ref is None:
        _staleness_reaper_ref = StalenessReaper(acquire_key_vault())
    return _staleness_reaper_ref


def acquire_vault_diagnostics() -> VaultDiagnostics:
    """获取全局存储诊断工具实例"""
    global _vault_diagnostics_ref
    if _vault_diagnostics_ref is None:
        _vault_diagnostics_ref = VaultDiagnostics(acquire_key_vault())
    return _vault_diagnostics_ref


# 短别名(兼容其他模块导入习惯)
get_memory_storage = acquire_key_vault
get_version_manager = acquire_revision_tracker
get_memory_exporter = acquire_bulk_transporter
get_garbage_collector = acquire_staleness_reaper
get_memory_statistics = acquire_vault_diagnostics

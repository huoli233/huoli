import json
import os
import time
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("memory")


class MemoryStorage:
    def __init__(self, storage_dir: str = "data/memories"):
        self._storage_dir = storage_dir
        self._ensure_dir()

    def _ensure_dir(self):
        import os
        if not os.path.exists(self._storage_dir):
            os.makedirs(self._storage_dir, exist_ok=True)

    def _get_model(self):
        from src.common.database.database_model import MemoryEntry
        return MemoryEntry

    def _migrate_json_if_exists(self, key: str):
        import os
        import json
        file_path = os.path.join(self._storage_dir, f"{key}.json")
        if not os.path.exists(file_path):
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.save(key, data)
            os.rename(file_path, file_path + ".bak")
            logger.info(f"记忆 {key} 从JSON迁移到数据库")
        except Exception as e:
            logger.debug(f"记忆 {key} JSON迁移跳过: {e}")

    def save(self, key: str, data):
        try:
            ME = self._get_model()
            import json
            import time
            json_str = json.dumps(data, ensure_ascii=False)
            now = time.time()
            ME.insert(
                memory_key=key,
                memory_value=json_str,
                created_at=now,
                updated_at=now,
                ttl_days=30
            ).on_conflict(
                conflict_target=[ME.memory_key],
                update={
                    ME.memory_value: json_str,
                    ME.updated_at: now,
                }
            ).execute()
        except Exception as e:
            logger.error(f"保存记忆失败: {e}")

    def load(self, key: str, default=None):
        try:
            ME = self._get_model()
            row = ME.get_or_none(ME.memory_key == key)
            if not row:
                self._migrate_json_if_exists(key)
                row = ME.get_or_none(ME.memory_key == key)
                if not row:
                    return default
            import json
            return json.loads(row.memory_value)
        except Exception as e:
            logger.error(f"加载记忆失败: {e}")
            return default

    def delete(self, key: str):
        try:
            ME = self._get_model()
            ME.delete().where(ME.memory_key == key).execute()
        except Exception as e:
            logger.error(f"删除记忆失败: {e}")

    def exists(self, key: str) -> bool:
        try:
            ME = self._get_model()
            return ME.select().where(ME.memory_key == key).exists()
        except Exception:
            return False

    def list_keys(self) -> List[str]:
        try:
            ME = self._get_model()
            return [row.memory_key for row in ME.select(ME.memory_key).distinct()]
        except Exception:
            return []


class MemoryVersionManager:
    def __init__(self, storage: MemoryStorage):
        self._storage = storage
        self._max_versions = 5

    def _get_version_model(self):
        from src.common.database.database_model import MemoryVersion
        return MemoryVersion

    def _get_meta_model(self):
        from src.common.database.database_model import MemoryMetadata
        return MemoryMetadata

    def _migrate_json_versions(self, key: str):
        versions_key = f"{key}_versions"
        if not self._storage.exists(versions_key):
            return
        try:
            versions = self._storage.load(versions_key)
            if not versions:
                return
            MV = self._get_version_model()
            MM = self._get_meta_model()
            import time
            now = time.time()
            for v in versions:
                MV.insert(
                    memory_key=key,
                    version=v.get("version", 1),
                    memory_value=json.dumps(v.get("data", {}), ensure_ascii=False),
                    created_at=v.get("timestamp", now)
                ).execute()
            MM.insert(
                memory_key=key,
                total_versions=len(versions),
                last_accessed_at=now,
                created_at=now
            ).on_conflict(
                conflict_target=[MM.memory_key],
                update={MM.total_versions: len(versions), MM.last_accessed_at: now}
            ).execute()
            self._storage.delete(versions_key)
            logger.info(f"版本记忆 {key} 从JSON迁移到数据库")
        except Exception as e:
            logger.debug(f"版本记忆 {key} JSON迁移跳过: {e}")

    def save_version(self, key: str, data):
        try:
            MV = self._get_version_model()
            MM = self._get_meta_model()
            import time
            now = time.time()
            # 获取当前版本数
            current_count = MV.select().where(MV.memory_key == key).count()
            if current_count >= self._max_versions:
                # 删除最旧的版本
                oldest = MV.select().where(MV.memory_key == key).order_by(MV.created_at.asc()).first()
                if oldest:
                    oldest.delete_instance()
                    current_count -= 1
            new_version = current_count + 1
            json_data = json.dumps(data, ensure_ascii=False)
            MV.insert(
                memory_key=key,
                version=new_version,
                memory_value=json_data,
                created_at=now
            ).execute()
            MM.insert(
                memory_key=key,
                total_versions=new_version,
                last_accessed_at=now,
                created_at=now
            ).on_conflict(
                conflict_target=[MM.memory_key],
                update={MM.total_versions: new_version, MM.last_accessed_at: now}
            ).execute()
        except Exception as e:
            logger.debug(f"保存版本失败: {e}")

    def _get_versions(self, key: str):
        try:
            MV = self._get_version_model()
            rows = MV.select().where(MV.memory_key == key).order_by(MV.version.asc())
            versions = []
            for row in rows:
                try:
                    data = json.loads(row.memory_value)
                except:
                    data = {}
                versions.append({
                    "data": data,
                    "timestamp": row.created_at,
                    "version": row.version,
                })
            if not versions:
                self._migrate_json_versions(key)
                # 重新获取
                rows = MV.select().where(MV.memory_key == key).order_by(MV.version.asc())
                for row in rows:
                    try:
                        data = json.loads(row.memory_value)
                    except:
                        data = {}
                    versions.append({
                        "data": data,
                        "timestamp": row.created_at,
                        "version": row.version,
                    })
            return versions
        except Exception:
            return []

    def get_version(self, key: str, version: int):
        try:
            MV = self._get_version_model()
            row = MV.get_or_none((MV.memory_key == key) & (MV.version == version))
            if row:
                return json.loads(row.memory_value)
        except Exception:
            pass
        return None

    def get_latest_version(self, key: str):
        try:
            MV = self._get_version_model()
            row = MV.select().where(MV.memory_key == key).order_by(MV.version.desc()).first()
            if row:
                return json.loads(row.memory_value)
        except Exception:
            pass
        return None

    def list_versions(self, key: str):
        versions = self._get_versions(key)
        return [{"version": v["version"], "timestamp": v["timestamp"]} for v in versions]


class MemoryExporter:
    def __init__(self, storage: MemoryStorage):
        self._storage = storage

    def export_all(self, output_path: str):
        all_data = {}
        for key in self._storage.list_keys():
            all_data[key] = self._storage.load(key)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            logger.info(f"导出 {len(all_data)} 条记忆到 {output_path}")
        except Exception as e:
            logger.error(f"导出记忆失败: {e}")

    def import_all(self, input_path: str):
        if not os.path.exists(input_path):
            logger.error(f"导入文件不存在: {input_path}")
            return
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            for key, data in all_data.items():
                self._storage.save(key, data)
            logger.info(f"导入 {len(all_data)} 条记忆")
        except Exception as e:
            logger.error(f"导入记忆失败: {e}")


class MemoryGarbageCollector:
    def __init__(self, storage: MemoryStorage, ttl_days: int = 30):
        self._storage = storage
        self._ttl_days = ttl_days

    def collect(self):
        try:
            ME = self._storage._get_model()
            import time
            cutoff = time.time() - (self._ttl_days * 86400)
            deleted = ME.delete().where(ME.updated_at < cutoff).execute()
            if deleted > 0:
                logger.info(f"清理 {deleted} 条过期记忆")
            return deleted
        except Exception as e:
            logger.debug(f"清理失败: {e}")
            return 0

    def set_ttl_days(self, days: int):
        self._ttl_days = max(1, days)


class MemoryStatistics:
    def __init__(self, storage: MemoryStorage):
        self._storage = storage

    def get_stats(self):
        try:
            ME = self._storage._get_model()
            total_keys = ME.select(ME.memory_key).distinct().count()
            total_size = 0
            for row in ME.select():
                total_size += len(row.memory_value.encode('utf-8'))
            return {
                "total_keys": total_keys,
                "total_size_bytes": total_size,
                "total_size_kb": total_size / 1024,
            }
        except Exception:
            return {"total_keys": 0, "total_size_bytes": 0, "total_size_kb": 0}


_memory_storage: Optional[MemoryStorage] = None
_memory_version_manager: Optional[MemoryVersionManager] = None
_memory_exporter: Optional[MemoryExporter] = None
_memory_garbage_collector: Optional[MemoryGarbageCollector] = None
_memory_statistics: Optional[MemoryStatistics] = None


def get_memory_storage() -> MemoryStorage:
    global _memory_storage
    if _memory_storage is None:
        _memory_storage = MemoryStorage()
    return _memory_storage


def get_memory_version_manager() -> MemoryVersionManager:
    global _memory_version_manager
    if _memory_version_manager is None:
        _memory_version_manager = MemoryVersionManager(get_memory_storage())
    return _memory_version_manager


def get_memory_exporter() -> MemoryExporter:
    global _memory_exporter
    if _memory_exporter is None:
        _memory_exporter = MemoryExporter(get_memory_storage())
    return _memory_exporter


def get_memory_garbage_collector() -> MemoryGarbageCollector:
    global _memory_garbage_collector
    if _memory_garbage_collector is None:
        _memory_garbage_collector = MemoryGarbageCollector(get_memory_storage())
    return _memory_garbage_collector


def get_memory_statistics() -> MemoryStatistics:
    global _memory_statistics
    if _memory_statistics is None:
        _memory_statistics = MemoryStatistics(get_memory_storage())
    return _memory_statistics

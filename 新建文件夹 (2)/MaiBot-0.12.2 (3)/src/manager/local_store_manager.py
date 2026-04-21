import json
import time
import os
from pathlib import Path
from src.common.logger import get_logger

logger = get_logger("local_storage")

JSON_LEGACY_PATH = "data/local_store.json"


class LocalStoreManager:
    store: dict[str, str | list | dict | int | float | bool]

    def __init__(self, legacy_path: str | None = None):
        self._legacy_path = legacy_path or JSON_LEGACY_PATH
        self.store = {}
        self._load_from_db()
        self._migrate_json_if_exists()

    def _get_model(self):
        from src.common.database.database_model import KeyValueStore
        return KeyValueStore

    def _serialize(self, value) -> tuple:
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False), "dict"
        elif isinstance(value, list):
            return json.dumps(value, ensure_ascii=False), "list"
        elif isinstance(value, bool):
            return str(value), "bool"
        elif isinstance(value, int):
            return str(value), "int"
        elif isinstance(value, float):
            return str(value), "float"
        return str(value), "str"

    def _deserialize(self, raw: str, vtype: str):
        if vtype == "dict":
            return json.loads(raw)
        elif vtype == "list":
            return json.loads(raw)
        elif vtype == "bool":
            return raw.lower() in ("true", "1")
        elif vtype == "int":
            return int(raw)
        elif vtype == "float":
            return float(raw)
        return raw

    def _load_from_db(self):
        try:
            KV = self._get_model()
            for row in KV.select():
                self.store[row.key] = self._deserialize(row.value, row.value_type)
            if self.store:
                logger.info(f"从数据库加载了 {len(self.store)} 条记录")
        except Exception as e:
            logger.warning(f"数据库加载失败，将使用空存储: {e}")
            self.store = {}

    def _migrate_json_if_exists(self):
        json_path = Path(self._legacy_path)
        if not json_path.exists():
            return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            if not old_data:
                json_path.unlink(missing_ok=True)
                return
            migrated = 0
            KV = self._get_model()
            for k, v in old_data.items():
                if k not in self.store:
                    self.store[k] = v
                    raw_val, vtype = self._serialize(v)
                    KV.insert(
                        key=k, value=raw_val, value_type=vtype, updated_at=time.time()
                    ).on_conflict(
                        conflict_target=[KV.key],
                        update={KV.value: raw_val, KV.value_type: vtype, KV.updated_at: time.time()}
                    ).execute()
                    migrated += 1
            if migrated > 0:
                logger.info(f"从JSON迁移了 {migrated} 条记录到数据库")
            backup = json_path.with_suffix(".json.bak")
            json_path.rename(backup)
            logger.info(f"旧JSON文件已备份为 {backup.name}")
        except Exception as e:
            logger.warning(f"JSON迁移失败: {e}")

    def __getitem__(self, item: str) -> str | list | dict | int | float | bool | None:
        return self.store.get(item)

    def __setitem__(self, key: str, value: str | list | dict | int | float | bool):
        self.store[key] = value
        self._save_key(key, value)

    def __delitem__(self, key: str):
        if key in self.store:
            del self.store[key]
            try:
                KV = self._get_model()
                KV.delete().where(KV.key == key).execute()
            except Exception as e:
                logger.warning(f"删除键 {key} 失败: {e}")
        else:
            logger.warning(f"尝试删除不存在的键: {key}")

    def __contains__(self, item: str) -> bool:
        return item in self.store

    def _save_key(self, key: str, value):
        try:
            KV = self._get_model()
            raw_val, vtype = self._serialize(value)
            KV.insert(
                key=key, value=raw_val, value_type=vtype, updated_at=time.time()
            ).on_conflict(
                conflict_target=[KV.key],
                update={KV.value: raw_val, KV.value_type: vtype, KV.updated_at: time.time()}
            ).execute()
        except Exception as e:
            logger.warning(f"保存键 {key} 失败: {e}")

    def load_local_store(self):
        self._load_from_db()

    def save_local_store(self):
        try:
            KV = self._get_model()
            for k, v in self.store.items():
                raw_val, vtype = self._serialize(v)
                KV.insert(
                    key=k, value=raw_val, value_type=vtype, updated_at=time.time()
                ).on_conflict(
                    conflict_target=[KV.key],
                    update={KV.value: raw_val, KV.value_type: vtype, KV.updated_at: time.time()}
                ).execute()
        except Exception as e:
            logger.warning(f"批量保存失败: {e}")


local_storage = LocalStoreManager(JSON_LEGACY_PATH)

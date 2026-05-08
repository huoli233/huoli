from src.common.database.slot_storage import load_slot, save_slot
from src.common.logger import get_logger

LOCAL_STORE_SLOT_KEY = "runtime_config:local_store"
logger = get_logger("本地存储")


class LocalStoreManager:
    file_path: str
    store: dict[str, str | list | dict | int | float | bool]

    def __init__(self, local_store_path: str | None = None):
        del local_store_path
        self.file_path = f"Huoli.db:{LOCAL_STORE_SLOT_KEY}"
        self.store = {}
        self.load_local_store()

    def __getitem__(
        self, item: str
    ) -> str | list | dict | int | float | bool | None:
        return self.store.get(item)

    def __setitem__(
        self, key: str, value: str | list | dict | int | float | bool
    ):
        self.store[key] = value
        self.save_local_store()

    def __delitem__(self, key: str):
        if key in self.store:
            del self.store[key]
            self.save_local_store()
        else:
            logger.warning(f"尝试删除不存在的键: {key}")

    def __contains__(self, item: str) -> bool:
        return item in self.store

    def get(self, key: str, default=None):
        return self.store.get(key, default)

    def set(self, key: str, value: str | list | dict | int | float | bool):
        self.store[key] = value
        self.save_local_store()

    def load_local_store(self):
        logger.debug(f"从数据库加载本地存储: {self.file_path}")
        payload = load_slot(LOCAL_STORE_SLOT_KEY, {})
        self.store = payload if isinstance(payload, dict) else {}
        if self.store:
            logger.info("本地存储已从数据库加载")

    def save_local_store(self):
        logger.debug(f"保存本地存储数据: {self.file_path}")
        save_slot(LOCAL_STORE_SLOT_KEY, self.store, ttl_days=3650)

    def clear(self):
        self.store = {}
        self.save_local_store()
        logger.info("本地存储已清空")

    def keys(self):
        return list(self.store.keys())

    def values(self):
        return list(self.store.values())

    def items(self):
        return list(self.store.items())


local_storage = LocalStoreManager()

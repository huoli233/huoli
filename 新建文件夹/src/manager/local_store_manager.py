import json
import os
from src.common.atomic_io import atomic_json_dump
from src.common.logger import get_logger

LOCAL_STORE_FILE_PATH = "data/local_store.json"
logger = get_logger("本地存储")


class LocalStoreManager:
    file_path: str
    store: dict[str, str | list | dict | int | float | bool]

    def __init__(self, local_store_path: str | None = None):
        self.file_path = local_store_path or LOCAL_STORE_FILE_PATH
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
        if os.path.exists(self.file_path):
            logger.info("正在阅读记事本......我在看，我真的在看！")
            logger.debug(f"加载本地存储数据: {self.file_path}")
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.store = json.load(f)
                    logger.info("全都记起来了！")
            except json.JSONDecodeError:
                logger.warning("啊咧？记事本被弄脏了，正在重建记事本......")
                self.store = {}
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False, indent=4)
                logger.info("记事本重建成功！")
        else:
            logger.warning("啊咧？记事本不存在，正在创建新的记事本......")
            dir_path = os.path.dirname(self.file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=4)
            logger.info("记事本创建成功！")

    def save_local_store(self):
        logger.debug(f"保存本地存储数据: {self.file_path}")
        dir_path = os.path.dirname(self.file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        atomic_json_dump(
            self.store, self.file_path, ensure_ascii=False, indent=4
        )

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


local_storage = LocalStoreManager("data/local_store.json")

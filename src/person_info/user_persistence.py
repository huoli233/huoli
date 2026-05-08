import asyncio
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional

from src.common.database.slot_storage import (
    delete_slot,
    list_slot_keys,
    load_slot,
    save_slot,
    slot_exists,
)
from src.common.logger import get_logger, sanitize_log_input
from src.person_info.runtime_config import identity_module_view

logger = get_logger("用户持久化")

_file_locks: Dict[str, Lock] = {}
_locks_lock = Lock()
_lock_access_times: Dict[str, float] = {}
_last_cleanup_ts: float = 0.0
_USER_SLOT_PREFIX = "identity_user"


def _user_persistence_config() -> Dict[str, Any]:
    return identity_module_view("identity_user_persistence")


def _get_file_lock(file_path: str) -> Lock:
    """获取文件锁（线程安全），定期清理过期锁"""
    global _last_cleanup_ts
    with _locks_lock:
        now = time.time()
        _lock_access_times[file_path] = now
        if file_path not in _file_locks:
            _file_locks[file_path] = Lock()
        cleanup_interval_sec = float(
            _user_persistence_config().get(
                "file_lock_cleanup_interval_seconds", 600.0
            )
        )
        if now - _last_cleanup_ts > cleanup_interval_sec:
            _last_cleanup_ts = now
        else:
            return _file_locks[file_path]
    # 在_locks_lock外执行清理，避免持锁过久
    _cleanup_stale_locks()
    with _locks_lock:
        return _file_locks.get(file_path) or Lock()


def _cleanup_stale_locks() -> int:
    """清理长期未访问的文件锁，返回清理数量"""
    removed = 0
    with _locks_lock:
        cutoff = time.time() - float(
            _user_persistence_config().get(
                "file_lock_stale_seconds", 3600.0
            )
        )
        stale_keys = [k for k, t in _lock_access_times.items() if t < cutoff]
        for k in stale_keys:
            lk = _file_locks.get(k)
            if lk and not lk.locked():
                del _file_locks[k]
                del _lock_access_times[k]
                removed += 1
    if removed:
        logger.debug(f"清理了 {removed} 个空闲文件锁")
    return removed


@dataclass
class UserActivity:
    """用户活动记录"""

    activity_type: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class UserDataStorage:
    """用户数据存储"""

    def __init__(
        self,
        config_engine: Optional[Any] = None,
        storage_dir: str = "",
    ):
        del config_engine
        config = _user_persistence_config()
        self._storage_dir = storage_dir or str(
            config.get("storage_dir", "data/users")
        )
        self._user_index: set[str] = set()
        self._index_loaded = False

    def _slot_key(self, user_id: str) -> str:
        safe_id = (
            str(user_id)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("..", "__")
        )
        return f"{_USER_SLOT_PREFIX}:{safe_id}"

    def _get_file_path(self, user_id: str) -> str:
        """兼容旧调用：返回数据库槽位标识。"""
        return f"Huoli.db:{self._slot_key(user_id)}"

    def save_user(self, user_id: str, data: Dict[str, Any]) -> bool:
        """保存用户数据"""
        slot_key = self._slot_key(user_id)
        data["last_updated"] = time.time()
        lock = _get_file_lock(slot_key)
        with lock:
            try:
                if not save_slot(slot_key, data, ttl_days=3650):
                    return False
                self._user_index.add(user_id)
                return True
            except Exception as e:
                logger.error(f"保存用户数据失败: {e}")
                return False

    def load_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """加载用户数据"""
        slot_key = self._slot_key(user_id)
        lock = _get_file_lock(slot_key)
        with lock:
            try:
                payload = load_slot(slot_key)
                return payload if isinstance(payload, dict) else None
            except Exception as e:
                logger.error(f"加载用户数据失败: {e}")
                return None

    def delete_user(self, user_id: str) -> bool:
        """删除用户数据"""
        slot_key = self._slot_key(user_id)
        lock = _get_file_lock(slot_key)
        with lock:
            try:
                removed = delete_slot(slot_key)
                self._user_index.discard(user_id)
                return removed
            except Exception as e:
                logger.error(f"删除用户数据失败: {e}")
                return False

    def user_exists(self, user_id: str) -> bool:
        """检查用户是否存在"""
        return slot_exists(self._slot_key(user_id))

    def list_users(self) -> List[str]:
        """列出所有用户（首次调用后使用内存索引）"""
        if not self._index_loaded:
            prefix = f"{_USER_SLOT_PREFIX}:"
            self._user_index = {
                key.removeprefix(prefix)
                for key in list_slot_keys(prefix)
            }
            self._index_loaded = True
        return list(self._user_index)

    async def async_save_user(
        self, user_id: str, data: Dict[str, Any]
    ) -> bool:
        """异步保存用户数据（不阻塞事件循环）"""
        return await asyncio.to_thread(self.save_user, user_id, data)

    async def async_load_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """异步加载用户数据（不阻塞事件循环）"""
        return await asyncio.to_thread(self.load_user, user_id)

    def get_user_count(self) -> int:
        """获取用户数量"""
        return len(self.list_users())


class UserDataExporter:
    """用户数据导出器"""

    def __init__(self, storage: UserDataStorage):
        self._storage = storage

    def _export_slot_key(self, label: str) -> str:
        safe = (
            str(label or "default")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("..", "__")
        )
        return f"identity_user_export:{safe[:120]}"

    def export_all(self, output_path: str) -> bool:
        """导出所有用户数据到数据库槽位，避免生成文件。"""
        all_data = {}
        for user_id in self._storage.list_users():
            user_data = self._storage.load_user(user_id)
            if user_data:
                all_data[user_id] = user_data
        try:
            slot_key = self._export_slot_key(output_path)
            ok = save_slot(slot_key, all_data, ttl_days=365)
            logger.info(
                f"导出 {len(all_data)} 个用户数据到数据库槽位 {slot_key}"
            )
            return ok
        except Exception as e:
            logger.error(f"导出用户数据失败: {e}")
            return False

    def import_all(self, input_path: str) -> int:
        """从数据库导出槽位导入用户数据。"""
        all_data = load_slot(self._export_slot_key(input_path))
        if not isinstance(all_data, dict):
            logger.error(f"导入槽位不存在或格式无效: {input_path}")
            return 0
        try:
            count = 0
            for user_id, user_data in all_data.items():
                if isinstance(user_data, dict) and self._storage.save_user(
                    user_id, user_data
                ):
                    count += 1
            logger.info(f"导入 {count} 个用户数据")
            return count
        except Exception as e:
            logger.error(f"导入用户数据失败: {e}")
            return 0

    def export_user(self, user_id: str, output_path: str) -> bool:
        """导出单个用户数据到数据库槽位，避免生成文件。"""
        user_data = self._storage.load_user(user_id)
        if not user_data:
            logger.warning("用户 %s 不存在", sanitize_log_input(user_id))
            return False
        try:
            return save_slot(self._export_slot_key(output_path), user_data)
        except Exception as e:
            logger.error(f"导出用户数据失败: {e}")
            return False


class UserDataMerger:
    """用户数据合并器"""

    def __init__(self, storage: UserDataStorage):
        self._storage = storage

    def merge_users(self, source_id: str, target_id: str) -> bool:
        """合并用户数据"""
        source_data = self._storage.load_user(source_id)
        target_data = self._storage.load_user(target_id)

        if not source_data:
            logger.warning(f"源用户 {source_id} 不存在")
            return False

        if not target_data:
            target_data = {}

        for key, value in source_data.items():
            if key not in target_data:
                target_data[key] = value
            elif isinstance(value, list) and isinstance(
                target_data[key], list
            ):
                target_data[key].extend(value)
            elif isinstance(value, dict) and isinstance(
                target_data[key], dict
            ):
                target_data[key].update(value)

        self._storage.save_user(target_id, target_data)
        logger.info(f"合并用户 {source_id} 到 {target_id}")
        return True


class UserActivityTracker:
    """用户活动追踪器"""

    def __init__(self, storage: UserDataStorage):
        self._storage = storage
        self._max_activities = int(
            _user_persistence_config().get("max_activities", 100)
        )

    def record_activity(
        self,
        user_id: str,
        activity_type: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录活动"""
        user_data = self._storage.load_user(user_id) or {}
        if "activities" not in user_data:
            user_data["activities"] = []

        activity = {
            "type": activity_type,
            "details": details or {},
            "timestamp": time.time(),
        }
        user_data["activities"].append(activity)

        if len(user_data["activities"]) > self._max_activities:
            user_data["activities"] = user_data["activities"][
                -self._max_activities:
            ]

        self._storage.save_user(user_id, user_data)

    def get_activities(
        self, user_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """获取活动记录"""
        user_data = self._storage.load_user(user_id)
        if not user_data:
            return []
        activities = user_data.get("activities", [])
        return activities[-limit:]

    def get_activity_summary(self, user_id: str) -> Dict[str, int]:
        """获取活动摘要"""
        user_data = self._storage.load_user(user_id)
        if not user_data:
            return {}
        activities = user_data.get("activities", [])
        summary: Dict[str, int] = {}
        for a in activities:
            activity_type = a.get("type", "unknown")
            summary[activity_type] = summary.get(activity_type, 0) + 1
        return summary


class UserStatisticsCalculator:
    """用户统计计算器"""

    def __init__(self, storage: UserDataStorage):
        self._storage = storage

    def calculate_global_stats(self) -> Dict[str, Any]:
        """计算全局统计"""
        users = self._storage.list_users()
        total_activities = 0
        active_users_24h = 0
        cutoff_24h = time.time() - 86400

        for user_id in users:
            user_data = self._storage.load_user(user_id)
            if user_data:
                activities = user_data.get("activities", [])
                total_activities += len(activities)
                last_updated = user_data.get("last_updated", 0)
                if last_updated >= cutoff_24h:
                    active_users_24h += 1

        return {
            "total_users": len(users),
            "total_activities": total_activities,
            "active_users_24h": active_users_24h,
        }


_user_data_storage: Optional[UserDataStorage] = None
_user_data_exporter: Optional[UserDataExporter] = None
_user_data_merger: Optional[UserDataMerger] = None
_user_activity_tracker: Optional[UserActivityTracker] = None
_user_statistics_calculator: Optional[UserStatisticsCalculator] = None
_singleton_lock = Lock()


def get_user_data_storage(
    config_engine: Optional[Any] = None,
) -> UserDataStorage:
    """获取用户数据存储单例"""
    global _user_data_storage
    if _user_data_storage is None:
        with _singleton_lock:
            if _user_data_storage is None:
                _user_data_storage = UserDataStorage(config_engine)
    return _user_data_storage


def get_user_data_exporter() -> UserDataExporter:
    """获取用户数据导出器单例"""
    global _user_data_exporter
    if _user_data_exporter is None:
        with _singleton_lock:
            if _user_data_exporter is None:
                _user_data_exporter = UserDataExporter(get_user_data_storage())
    return _user_data_exporter


def get_user_data_merger() -> UserDataMerger:
    """获取用户数据合并器单例"""
    global _user_data_merger
    if _user_data_merger is None:
        with _singleton_lock:
            if _user_data_merger is None:
                _user_data_merger = UserDataMerger(get_user_data_storage())
    return _user_data_merger


def get_user_activity_tracker() -> UserActivityTracker:
    """获取用户活动追踪器单例"""
    global _user_activity_tracker
    if _user_activity_tracker is None:
        with _singleton_lock:
            if _user_activity_tracker is None:
                _user_activity_tracker = UserActivityTracker(
                    get_user_data_storage()
                )
    return _user_activity_tracker


def get_user_statistics_calculator() -> UserStatisticsCalculator:
    """获取用户统计计算器单例"""
    global _user_statistics_calculator
    if _user_statistics_calculator is None:
        with _singleton_lock:
            if _user_statistics_calculator is None:
                _user_statistics_calculator = UserStatisticsCalculator(
                    get_user_data_storage()
                )
    return _user_statistics_calculator

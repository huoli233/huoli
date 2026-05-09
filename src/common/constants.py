"""
统一常量定义模块

集中管理项目中的硬编码值，提高可维护性和可读性。
"""

from datetime import timezone
from typing import Final

# ==================== 重试与阈值常量 ====================

# 重试相关
MAX_RETRY_COUNT: Final[int] = 3
RETRY_BACKOFF_BASE: Final[int] = 4

# 冲突与交互阈值
CONFLICT_THRESHOLD_LOW: Final[int] = 3
CONFLICT_THRESHOLD_MEDIUM: Final[int] = 5
CONFLICT_THRESHOLD_HIGH: Final[int] = 10
INTERACTION_THRESHOLD_MIN: Final[int] = 3
INTERACTION_THRESHOLD_LOW: Final[int] = 5
INTERACTION_THRESHOLD_MEDIUM: Final[int] = 20
INTERACTION_THRESHOLD_HIGH: Final[int] = 50

# 重复检测阈值
REPETITION_THRESHOLD: Final[int] = 3
MAX_EMOJI_COUNT: Final[int] = 3

# ==================== 超时常量 (秒) ====================

# 网络请求超时
TIMEOUT_HTTP_SHORT: Final[float] = 5.0
TIMEOUT_HTTP_MEDIUM: Final[float] = 10.0
TIMEOUT_HTTP_LONG: Final[float] = 15.0
TIMEOUT_HTTP_EXTENDED: Final[float] = 30.0
TIMEOUT_HTTP_VERY_LONG: Final[float] = 60.0
TIMEOUT_HTTP_MAX: Final[float] = 300.0

# 任务管理超时
TIMEOUT_TASK_CANCEL: Final[float] = 5.0
TIMEOUT_TASK_SHUTDOWN: Final[float] = 10.0

# WebSocket 超时
TIMEOUT_WEBSOCKET_RECEIVE: Final[float] = 60.0
TIMEOUT_WEBSOCKET_IDLE: Final[float] = 300.0

# Git 操作超时
TIMEOUT_GIT_SHORT: Final[float] = 5.0
TIMEOUT_GIT_LONG: Final[float] = 300.0

# ==================== 缓存与容量常量 ====================

# 缓存大小
CACHE_MAX_ENTRIES: Final[int] = 10000
CACHE_BACKUP_COUNT: Final[int] = 30
CACHE_MAX_BYTES: Final[int] = 5 * 1024 * 1024

# 消息处理
MESSAGE_MAX_LENGTH: Final[int] = 200
MESSAGE_MERGE_WINDOW: Final[float] = 30.0
MESSAGE_SHORT_LENGTH: Final[int] = 50

# 标签限制
MAX_CHARACTER_TAGS: Final[int] = 15

# ==================== 路径常量 ====================

# 数据目录
DATA_DIR: Final[str] = "data"
LOGS_DIR: Final[str] = "logs"
TEMP_DIR: Final[str] = "data/temp"
USERS_DIR: Final[str] = "data/users"

# 存储路径
SOCIAL_STORAGE_DIR: Final[str] = "data/huoli/social"
GROUP_PERSONA_DIR: Final[str] = "data/group_persona"
MEMOIR_VAULT_DIR: Final[str] = "data/memoir_vault"
EMBEDDING_DIR: Final[str] = "data/embedding"

# ==================== 时区工具 =================---

LOCAL_TIMEZONE: Final[timezone] = timezone.utc


def get_timezone_aware_now():
    """获取时区感知的当前时间"""
    from datetime import datetime
    return datetime.now(LOCAL_TIMEZONE)


def get_local_now():
    """获取本地时区的当前时间（用于显示）"""
    from datetime import datetime
    return datetime.now()


def timestamp_to_datetime(ts: float):
    """将时间戳转换为时区感知的datetime"""
    from datetime import datetime
    return datetime.fromtimestamp(ts, tz=LOCAL_TIMEZONE)


# ==================== WebUI 相关常量 ====================

# CORS 允许的源
CORS_ALLOWED_ORIGINS: Final[tuple] = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:7999",
    "http://127.0.0.1:7999",
)

# 默认服务地址
DEFAULT_WEBUI_HOST: Final[str] = "127.0.0.1"
DEFAULT_WEBUI_PORT: Final[int] = 8001

__all__ = [
    # 重试与阈值
    "MAX_RETRY_COUNT",
    "RETRY_BACKOFF_BASE",
    "CONFLICT_THRESHOLD_LOW",
    "CONFLICT_THRESHOLD_MEDIUM",
    "CONFLICT_THRESHOLD_HIGH",
    "INTERACTION_THRESHOLD_MIN",
    "INTERACTION_THRESHOLD_LOW",
    "INTERACTION_THRESHOLD_MEDIUM",
    "INTERACTION_THRESHOLD_HIGH",
    "REPETITION_THRESHOLD",
    "MAX_EMOJI_COUNT",
    # 超时
    "TIMEOUT_HTTP_SHORT",
    "TIMEOUT_HTTP_MEDIUM",
    "TIMEOUT_HTTP_LONG",
    "TIMEOUT_HTTP_EXTENDED",
    "TIMEOUT_HTTP_VERY_LONG",
    "TIMEOUT_HTTP_MAX",
    "TIMEOUT_TASK_CANCEL",
    "TIMEOUT_TASK_SHUTDOWN",
    "TIMEOUT_WEBSOCKET_RECEIVE",
    "TIMEOUT_WEBSOCKET_IDLE",
    "TIMEOUT_GIT_SHORT",
    "TIMEOUT_GIT_LONG",
    # 缓存与容量
    "CACHE_MAX_ENTRIES",
    "CACHE_BACKUP_COUNT",
    "CACHE_MAX_BYTES",
    "MESSAGE_MAX_LENGTH",
    "MESSAGE_MERGE_WINDOW",
    "MESSAGE_SHORT_LENGTH",
    "MAX_CHARACTER_TAGS",
    # 路径
    "DATA_DIR",
    "LOGS_DIR",
    "TEMP_DIR",
    "USERS_DIR",
    "SOCIAL_STORAGE_DIR",
    "GROUP_PERSONA_DIR",
    "MEMOIR_VAULT_DIR",
    "EMBEDDING_DIR",
    # 时区
    "LOCAL_TIMEZONE",
    "get_timezone_aware_now",
    "get_local_now",
    "timestamp_to_datetime",
    # WebUI
    "CORS_ALLOWED_ORIGINS",
    "DEFAULT_WEBUI_HOST",
    "DEFAULT_WEBUI_PORT",
]

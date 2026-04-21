import sys
import os
import json
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from src.common.logger import get_logger

logger = get_logger("日志API")


class LogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEntry:
    def __init__(
        self,
        level: str,
        message: str,
        source: str = "",
        timestamp: Optional[datetime] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.level = level
        self.message = message
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "extra": self.extra,
        }

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        extra_str = (
            f" | {
                json.dumps(
                    self.extra,
                    ensure_ascii=False)}"
            if self.extra
            else ""
        )
        return f"[{time_str}] [{self.level}] [{self.source}] {self.message}{extra_str}"


class PluginLogger:
    def __init__(self, plugin_name: str):
        self.plugin_name = plugin_name
        self._log_buffer: List[LogEntry] = []
        self._max_buffer_size = 1000
        self._callbacks: List[Callable[[LogEntry], None]] = []
        self._file_handler: Optional[Any] = None
        self._log_file_path: Optional[str] = None

    def _add_to_buffer(self, entry: LogEntry):
        self._log_buffer.append(entry)
        if len(self._log_buffer) > self._max_buffer_size:
            self._log_buffer = self._log_buffer[-self._max_buffer_size:]
        for callback in self._callbacks:
            try:
                callback(entry)
            except Exception as e:
                logger.error(f"[LoggingAPI] 回调执行失败: {e}")

    def add_callback(self, callback: Callable[[LogEntry], None]):
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[LogEntry], None]):
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def set_log_file(self, file_path: str):
        try:
            self._log_file_path = file_path
            log_dir = os.path.dirname(file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"[LoggingAPI] 设置日志文件失败: {e}")

    def _write_to_file(self, entry: LogEntry):
        if not self._log_file_path:
            return
        try:
            with open(self._log_file_path, "a", encoding="utf-8") as f:
                f.write(str(entry) + "\n")
        except Exception as e:
            logger.error(f"[LoggingAPI] 写入日志文件失败: {e}")

    def debug(self, message: str, **extra):
        entry = LogEntry(
            level=LogLevel.DEBUG,
            message=message,
            source=self.plugin_name,
            extra=extra if extra else None,
        )
        self._add_to_buffer(entry)
        self._write_to_file(entry)
        logger.debug(f"[{self.plugin_name}] {message}")

    def info(self, message: str, **extra):
        entry = LogEntry(
            level=LogLevel.INFO,
            message=message,
            source=self.plugin_name,
            extra=extra if extra else None,
        )
        self._add_to_buffer(entry)
        self._write_to_file(entry)
        logger.info(f"[{self.plugin_name}] {message}")

    def warning(self, message: str, **extra):
        entry = LogEntry(
            level=LogLevel.WARNING,
            message=message,
            source=self.plugin_name,
            extra=extra if extra else None,
        )
        self._add_to_buffer(entry)
        self._write_to_file(entry)
        logger.warning(f"[{self.plugin_name}] {message}")

    def error(self, message: str, **extra):
        entry = LogEntry(
            level=LogLevel.ERROR,
            message=message,
            source=self.plugin_name,
            extra=extra if extra else None,
        )
        self._add_to_buffer(entry)
        self._write_to_file(entry)
        logger.error(f"[{self.plugin_name}] {message}")

    def critical(self, message: str, **extra):
        entry = LogEntry(
            level=LogLevel.CRITICAL,
            message=message,
            source=self.plugin_name,
            extra=extra if extra else None,
        )
        self._add_to_buffer(entry)
        self._write_to_file(entry)
        logger.critical(f"[{self.plugin_name}] {message}")

    def get_recent_logs(
        self, count: int = 100, level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        logs = self._log_buffer
        if level:
            logs = [log for log in logs if log.level == level]
        return [log.to_dict() for log in logs[-count:]]

    def clear_buffer(self):
        self._log_buffer.clear()


_plugin_loggers: Dict[str, PluginLogger] = {}


def get_plugin_logger(plugin_name: str) -> PluginLogger:
    if plugin_name not in _plugin_loggers:
        _plugin_loggers[plugin_name] = PluginLogger(plugin_name)
    return _plugin_loggers[plugin_name]


def log_debug(plugin_name: str, message: str, **extra):
    get_plugin_logger(plugin_name).debug(message, **extra)


def log_info(plugin_name: str, message: str, **extra):
    get_plugin_logger(plugin_name).info(message, **extra)


def log_warning(plugin_name: str, message: str, **extra):
    get_plugin_logger(plugin_name).warning(message, **extra)


def log_error(plugin_name: str, message: str, **extra):
    get_plugin_logger(plugin_name).error(message, **extra)


def log_critical(plugin_name: str, message: str, **extra):
    get_plugin_logger(plugin_name).critical(message, **extra)


def get_logs(
    plugin_name: str, count: int = 100, level: Optional[str] = None
) -> List[Dict[str, Any]]:
    return get_plugin_logger(plugin_name).get_recent_logs(count, level)


def clear_logs(plugin_name: str):
    get_plugin_logger(plugin_name).clear_buffer()


def set_plugin_log_file(plugin_name: str, file_path: str):
    get_plugin_logger(plugin_name).set_log_file(file_path)


def add_log_callback(plugin_name: str, callback: Callable[[LogEntry], None]):
    get_plugin_logger(plugin_name).add_callback(callback)


def remove_log_callback(
    plugin_name: str, callback: Callable[[LogEntry], None]
):
    get_plugin_logger(plugin_name).remove_callback(callback)

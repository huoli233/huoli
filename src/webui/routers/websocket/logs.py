import asyncio
import time
import threading
from collections import deque
from typing import Optional
from datetime import datetime
from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    Query,
    Cookie,
    Header,
    Depends,
)
from src.common.logger import get_logger
from src.webui.core.auth import verify_auth_token_from_cookie_or_header
from src.webui.runtime_config import webui_module_view

logger = get_logger("WS日志")
router = APIRouter(tags=["websocket"])


def _websocket_config() -> dict:
    return webui_module_view("webui_websocket")


def require_auth(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
) -> bool:
    """认证依赖：验证用户是否已登录"""
    return verify_auth_token_from_cookie_or_header(
        huoli_session, authorization
    )


class LogEntry:
    """日志条目"""

    def __init__(
        self, level: str, message: str, source: str, timestamp: float = None
    ):
        self.level = level
        self.message = message
        self.source = source
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
        }


class LogBuffer:
    """日志缓冲区"""

    def __init__(self, max_size: Optional[int] = None):
        config = _websocket_config()
        if max_size is None:
            max_size = int(config.get("logs_buffer_max_size", 1000))
        self._buffer: deque[LogEntry] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._subscribers: list[WebSocket] = []
        self._subscriber_lock = threading.Lock()
        self._closing = False
        self._pending_tasks: set[asyncio.Task] = set()
        self._close_timeout_seconds = float(
            config.get("logs_close_timeout_seconds", 5.0)
        )

    def add_log(self, level: str, message: str, source: str = "system"):
        """添加日志条目"""
        if self._closing:
            return
        entry = LogEntry(level, message, source)
        with self._lock:
            self._buffer.append(entry)
        try:
            task = asyncio.get_running_loop().create_task(self._broadcast_log(entry))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
        except RuntimeError:
            pass

    async def close(self, timeout: Optional[float] = None) -> None:
        """优雅关闭：停止接收新日志，等待所有待完成的广播任务"""
        timeout = (
            self._close_timeout_seconds if timeout is None else timeout
        )
        self._closing = True
        if not self._pending_tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*list(self._pending_tasks), return_exceptions=True),
                timeout=timeout,
            )
        except (asyncio.TimeoutError, Exception):
            for t in list(self._pending_tasks):
                if not t.done():
                    t.cancel()
            if self._pending_tasks:
                await asyncio.gather(*list(self._pending_tasks), return_exceptions=True)
        self._pending_tasks.clear()

    def get_recent_logs(self, count: int = 100) -> list[dict]:
        """获取最近的日志"""
        with self._lock:
            entries = list(self._buffer)[-count:]
        return [entry.to_dict() for entry in entries]

    def get_logs_by_level(self, level: str, count: int = 100) -> list[dict]:
        """按级别获取日志"""
        with self._lock:
            entries = [e for e in self._buffer if e.level == level][-count:]
        return [entry.to_dict() for entry in entries]

    def subscribe(self, websocket: WebSocket):
        """订阅日志推送"""
        with self._subscriber_lock:
            if websocket not in self._subscribers:
                self._subscribers.append(websocket)
                logger.info(
                    f"WebSocket 订阅日志: 当前订阅数 {len(self._subscribers)}"
                )

    def unsubscribe(self, websocket: WebSocket):
        """取消订阅"""
        with self._subscriber_lock:
            if websocket in self._subscribers:
                self._subscribers.remove(websocket)
                logger.info(
                    f"WebSocket 取消订阅日志: 当前订阅数 {len(self._subscribers)}"
                )

    async def _broadcast_log(self, entry: LogEntry):
        """广播日志到所有订阅者"""
        message = {
            "type": "log",
            "data": entry.to_dict(),
        }
        disconnected = []
        with self._subscriber_lock:
            for ws in self._subscribers:
                try:
                    await ws.send_json(message)
                except Exception:
                    disconnected.append(ws)
        for ws in disconnected:
            self.unsubscribe(ws)


log_buffer = LogBuffer()


class WebSocketLogHandler:
    """WebSocket 日志处理器"""

    def __init__(self, buffer: LogBuffer):
        self.buffer = buffer

    def emit(self, record):
        """发送日志记录"""
        try:
            level = record.levelname.lower()
            message = self.format(record)
            source = record.name
            self.buffer.add_log(level, message, source)
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def format(self, record):
        """格式化日志记录"""
        return record.getMessage()


class LogConnectionManager:
    """日志 WebSocket 连接管理器"""

    def __init__(self):
        self._active_connections: dict[str, WebSocket] = {}
        self._connection_filters: dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """接受连接"""
        try:
            await websocket.accept()
            self._active_connections[session_id] = websocket
            self._connection_filters[session_id] = {
                "levels": [],
                "sources": [],
                "keywords": [],
            }
            log_buffer.subscribe(websocket)
            logger.info(f"日志 WebSocket 连接已建立: {session_id}")
            return True
        except Exception as e:
            logger.error(f"日志 WebSocket 连接失败: {e}")
            return False

    def disconnect(self, session_id: str):
        """断开连接"""
        if session_id in self._active_connections:
            websocket = self._active_connections[session_id]
            log_buffer.unsubscribe(websocket)
            del self._active_connections[session_id]
        if session_id in self._connection_filters:
            del self._connection_filters[session_id]
        logger.info(f"日志 WebSocket 连接已断开: {session_id}")

    def set_filter(self, session_id: str, filter_config: dict):
        """设置连接的日志过滤器"""
        if session_id in self._connection_filters:
            self._connection_filters[session_id].update(filter_config)

    def get_filter(self, session_id: str) -> Optional[dict]:
        """获取连接的过滤器配置"""
        return self._connection_filters.get(session_id)

    async def send_history(self, session_id: str, count: int = 100):
        """发送历史日志"""
        if session_id not in self._active_connections:
            return
        websocket = self._active_connections[session_id]
        filter_config = self._connection_filters.get(session_id, {})
        logs = log_buffer.get_recent_logs(count)
        filtered_logs = self._apply_filter(logs, filter_config)
        try:
            await websocket.send_json(
                {
                    "type": "history",
                    "data": {
                        "logs": filtered_logs,
                        "count": len(filtered_logs),
                    },
                }
            )
        except Exception as e:
            logger.warning(f"发送历史日志失败: {e}")

    def _apply_filter(
        self, logs: list[dict], filter_config: dict
    ) -> list[dict]:
        """应用过滤器"""
        levels = filter_config.get("levels", [])
        sources = filter_config.get("sources", [])
        keywords = filter_config.get("keywords", [])
        result = []
        for log in logs:
            if levels and log["level"] not in levels:
                continue
            if sources and log["source"] not in sources:
                continue
            if keywords:
                matched = any(
                    kw.lower() in log["message"].lower() for kw in keywords
                )
                if not matched:
                    continue
            result.append(log)
        return result


log_connection_manager = LogConnectionManager()


@router.websocket("/ws/logs")
async def websocket_logs_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    levels: Optional[str] = Query(None, description="日志级别过滤，逗号分隔"),
    sources: Optional[str] = Query(None, description="日志来源过滤，逗号分隔"),
):
    """WebSocket 日志流端点"""
    from src.webui.core.security import get_token_manager

    session_id = f"log_{int(time.time() * 1000)}_{id(websocket)}"
    if not token:
        await websocket.close(code=4001, reason="缺少认证令牌")
        return
    token_manager = get_token_manager()
    if not token_manager.verify_token(token):
        await websocket.close(code=4002, reason="无效的认证令牌")
        return
    connected = await log_connection_manager.connect(websocket, session_id)
    if not connected:
        return
    if levels:
        log_connection_manager.set_filter(
            session_id,
            {"levels": [level.strip().lower() for level in levels.split(",")]},
        )
    if sources:
        log_connection_manager.set_filter(
            session_id, {"sources": [s.strip() for s in sources.split(",")]}
        )
    try:
        config = _websocket_config()
        initial_history_count = int(config.get("logs_initial_history_count", 50))
        receive_timeout_seconds = float(
            config.get("logs_receive_timeout_seconds", 60.0)
        )
        default_history_count = int(
            config.get("logs_default_history_count", 100)
        )
        max_history_count = int(config.get("logs_max_history_count", 200))
        await log_connection_manager.send_history(session_id, initial_history_count)
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(), timeout=receive_timeout_seconds
                )
                message_type = data.get("type", "unknown")
                if message_type == "ping":
                    await websocket.send_json(
                        {"type": "pong", "data": {"timestamp": time.time()}}
                    )
                elif message_type == "set_filter":
                    filter_config = data.get("data", {})
                    log_connection_manager.set_filter(
                        session_id, filter_config
                    )
                    await websocket.send_json(
                        {
                            "type": "filter_updated",
                            "data": log_connection_manager.get_filter(
                                session_id
                            ),
                        }
                    )
                elif message_type == "get_history":
                    count = int(
                        data.get("data", {}).get(
                            "count", default_history_count
                        )
                    )
                    count = max(1, min(max_history_count, count))
                    await log_connection_manager.send_history(
                        session_id, count
                    )
            except asyncio.TimeoutError:
                await websocket.send_json(
                    {"type": "heartbeat", "data": {"timestamp": time.time()}}
                )
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"日志 WebSocket 消息处理错误: {e}")
                break
    except WebSocketDisconnect:
        pass
    finally:
        log_connection_manager.disconnect(session_id)


@router.get("/logs/recent")
async def get_recent_logs(
    count: int = 100, _auth: bool = Depends(require_auth)
):
    """获取最近日志（HTTP 接口）"""
    return {
        "success": True,
        "logs": log_buffer.get_recent_logs(count),
        "count": count,
    }


@router.get("/logs/level/{level}")
async def get_logs_by_level(
    level: str, count: int = 100, _auth: bool = Depends(require_auth)
):
    """按级别获取日志"""
    return {
        "success": True,
        "logs": log_buffer.get_logs_by_level(level.lower(), count),
        "level": level,
        "count": count,
    }

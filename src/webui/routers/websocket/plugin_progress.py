import asyncio
import time
import threading
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
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
from src.common.task_utils import safe_create_task
from src.webui.core.auth import verify_auth_token_from_cookie_or_header
from src.webui.runtime_config import webui_module_view

logger = get_logger("WS插件进度")
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


class ProgressStatus(str, Enum):
    """进度状态"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressInfo:
    """进度信息"""

    task_id: str
    task_type: str
    plugin_id: str
    plugin_name: str
    status: ProgressStatus = ProgressStatus.PENDING
    progress: float = 0.0
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    error_message: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    extra_data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "plugin_id": self.plugin_id,
            "plugin_name": self.plugin_name,
            "status": self.status.value,
            "progress": round(self.progress, 2),
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "error_message": self.error_message,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_time": (self.end_time or time.time()) - self.start_time,
            "extra_data": self.extra_data,
        }


class ProgressManager:
    """进度管理器"""

    def __init__(self):
        self._tasks: dict[str, ProgressInfo] = {}
        self._subscribers: dict[str, list[WebSocket]] = {}
        self._task_subscribers: dict[str, list[WebSocket]] = {}
        self._lock = threading.Lock()
        self._subscriber_lock = threading.Lock()

    def create_task(
        self,
        task_id: str,
        task_type: str,
        plugin_id: str,
        plugin_name: str,
        total_steps: int = 0,
    ) -> ProgressInfo:
        """创建进度任务"""
        task = ProgressInfo(
            task_id=task_id,
            task_type=task_type,
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            total_steps=total_steps,
            status=ProgressStatus.PENDING,
        )
        with self._lock:
            self._tasks[task_id] = task
        logger.info(
            f"创建进度任务: task_id={task_id}, type={task_type}, plugin={plugin_name}"
        )
        safe_create_task(self._broadcast_progress(task), name="broadcast_progress")
        return task

    def update_progress(
        self,
        task_id: str,
        progress: float = None,
        current_step: str = None,
        completed_steps: int = None,
        status: ProgressStatus = None,
        extra_data: dict = None,
    ):
        """更新进度"""
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"任务不存在: {task_id}")
                return
            task = self._tasks[task_id]
            if progress is not None:
                task.progress = min(100.0, max(0.0, progress))
            if current_step is not None:
                task.current_step = current_step
            if completed_steps is not None:
                task.completed_steps = completed_steps
                if task.total_steps > 0:
                    task.progress = (completed_steps / task.total_steps) * 100
            if status is not None:
                task.status = status
                if status in (
                    ProgressStatus.COMPLETED,
                    ProgressStatus.FAILED,
                    ProgressStatus.CANCELLED,
                ):
                    task.end_time = time.time()
                    if status == ProgressStatus.COMPLETED:
                        task.progress = 100.0
            if extra_data is not None:
                task.extra_data.update(extra_data)
        safe_create_task(self._broadcast_progress(task), name="broadcast_progress")

    def complete_task(self, task_id: str, extra_data: dict = None):
        """完成任务"""
        self.update_progress(
            task_id,
            progress=100.0,
            status=ProgressStatus.COMPLETED,
            extra_data=extra_data,
        )

    def fail_task(
        self, task_id: str, error_message: str, extra_data: dict = None
    ):
        """任务失败"""
        with self._lock:
            if task_id not in self._tasks:
                return
            task = self._tasks[task_id]
            task.status = ProgressStatus.FAILED
            task.error_message = error_message
            task.end_time = time.time()
            if extra_data:
                task.extra_data.update(extra_data)
        safe_create_task(self._broadcast_progress(task), name="broadcast_progress")

    def cancel_task(self, task_id: str):
        """取消任务"""
        with self._lock:
            if task_id not in self._tasks:
                return
            task = self._tasks[task_id]
            task.status = ProgressStatus.CANCELLED
            task.end_time = time.time()
        safe_create_task(self._broadcast_progress(task), name="broadcast_progress")

    def get_task(self, task_id: str) -> Optional[ProgressInfo]:
        """获取任务信息"""
        with self._lock:
            return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[dict]:
        """获取所有任务"""
        with self._lock:
            return [task.to_dict() for task in self._tasks.values()]

    def get_active_tasks(self) -> list[dict]:
        """获取活动任务"""
        with self._lock:
            return [
                task.to_dict()
                for task in self._tasks.values()
                if task.status
                in (ProgressStatus.PENDING, ProgressStatus.RUNNING)
            ]

    def subscribe_all(self, websocket: WebSocket):
        """订阅所有进度更新"""
        with self._subscriber_lock:
            session_id = f"ws_{id(websocket)}"
            if session_id not in self._subscribers:
                self._subscribers[session_id] = []
            self._subscribers[session_id].append(websocket)
            logger.info(f"WebSocket 订阅所有进度: session={session_id}")

    def subscribe_task(self, task_id: str, websocket: WebSocket):
        """订阅特定任务进度"""
        with self._subscriber_lock:
            if task_id not in self._task_subscribers:
                self._task_subscribers[task_id] = []
            self._task_subscribers[task_id].append(websocket)
            logger.info(f"WebSocket 订阅任务进度: task_id={task_id}")

    def unsubscribe_all(self, websocket: WebSocket):
        """取消订阅所有进度"""
        with self._subscriber_lock:
            session_id = f"ws_{id(websocket)}"
            if session_id in self._subscribers:
                del self._subscribers[session_id]
            for task_id in list(self._task_subscribers.keys()):
                if websocket in self._task_subscribers[task_id]:
                    self._task_subscribers[task_id].remove(websocket)

    async def _broadcast_progress(self, task: ProgressInfo):
        """广播进度更新"""
        message = {
            "type": "progress",
            "data": task.to_dict(),
        }
        disconnected = []
        with self._subscriber_lock:
            for session_id, websockets in self._subscribers.items():
                for ws in websockets:
                    try:
                        await ws.send_json(message)
                    except Exception:
                        disconnected.append((session_id, ws))
            if task.task_id in self._task_subscribers:
                for ws in self._task_subscribers[task.task_id]:
                    try:
                        await ws.send_json(message)
                    except Exception:
                        disconnected.append((task.task_id, ws))
        for key, ws in disconnected:
            try:
                with self._subscriber_lock:
                    if (
                        key in self._subscribers
                        and ws in self._subscribers.get(key, [])
                    ):
                        self._subscribers[key].remove(ws)
                    if (
                        key in self._task_subscribers
                        and ws in self._task_subscribers.get(key, [])
                    ):
                        self._task_subscribers[key].remove(ws)
            except Exception as _cleanup_err:
                import logging as _lg

                _lg.getLogger("plugin_progress").debug(
                    f"清理断连订阅者时出错: {_cleanup_err}"
                )


progress_manager = ProgressManager()


async def update_progress(
    task_id: str,
    progress: float = None,
    current_step: str = None,
    completed_steps: int = None,
    status: ProgressStatus = None,
    extra_data: dict = None,
):
    """更新进度的模块级函数"""
    progress_manager.update_progress(
        task_id=task_id,
        progress=progress,
        current_step=current_step,
        completed_steps=completed_steps,
        status=status,
        extra_data=extra_data,
    )


class PluginProgressConnectionManager:
    """插件进度 WebSocket 连接管理器"""

    def __init__(self):
        self._active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """接受连接"""
        try:
            await websocket.accept()
            self._active_connections[session_id] = websocket
            progress_manager.subscribe_all(websocket)
            logger.info(f"插件进度 WebSocket 连接已建立: {session_id}")
            return True
        except Exception as e:
            logger.error(f"插件进度 WebSocket 连接失败: {e}")
            return False

    def disconnect(self, session_id: str):
        """断开连接"""
        if session_id in self._active_connections:
            websocket = self._active_connections[session_id]
            progress_manager.unsubscribe_all(websocket)
            del self._active_connections[session_id]
        logger.info(f"插件进度 WebSocket 连接已断开: {session_id}")

    async def send_active_tasks(self, session_id: str):
        """发送活动任务列表"""
        if session_id not in self._active_connections:
            return
        websocket = self._active_connections[session_id]
        tasks = progress_manager.get_active_tasks()
        try:
            await websocket.send_json(
                {
                    "type": "active_tasks",
                    "data": {
                        "tasks": tasks,
                        "count": len(tasks),
                    },
                }
            )
        except Exception as e:
            logger.warning(f"发送活动任务失败: {e}")


plugin_progress_manager = PluginProgressConnectionManager()


@router.websocket("/ws/plugin-progress")
async def websocket_plugin_progress_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    task_id: Optional[str] = Query(None, description="订阅特定任务"),
):
    """WebSocket 插件进度端点"""
    from src.webui.core.security import get_token_manager

    session_id = f"progress_{int(time.time() * 1000)}_{id(websocket)}"
    if not token:
        await websocket.close(code=4001, reason="缺少认证令牌")
        return
    token_manager = get_token_manager()
    if not token_manager.verify_token(token):
        await websocket.close(code=4002, reason="无效的认证令牌")
        return
    connected = await plugin_progress_manager.connect(websocket, session_id)
    if not connected:
        return
    if task_id:
        progress_manager.subscribe_task(task_id, websocket)
        task = progress_manager.get_task(task_id)
        if task:
            await websocket.send_json(
                {
                    "type": "task_info",
                    "data": task.to_dict(),
                }
            )
    try:
        config = _websocket_config()
        receive_timeout_seconds = float(
            config.get("plugin_progress_receive_timeout_seconds", 60.0)
        )
        await plugin_progress_manager.send_active_tasks(session_id)
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
                elif message_type == "subscribe_task":
                    subscribe_task_id = data.get("data", {}).get("task_id")
                    if subscribe_task_id:
                        progress_manager.subscribe_task(
                            subscribe_task_id, websocket
                        )
                        task = progress_manager.get_task(subscribe_task_id)
                        if task:
                            await websocket.send_json(
                                {
                                    "type": "task_info",
                                    "data": task.to_dict(),
                                }
                            )
                elif message_type == "get_active_tasks":
                    await plugin_progress_manager.send_active_tasks(session_id)
                elif message_type == "cancel_task":
                    cancel_task_id = data.get("data", {}).get("task_id")
                    if cancel_task_id:
                        progress_manager.cancel_task(cancel_task_id)
            except asyncio.TimeoutError:
                await websocket.send_json(
                    {"type": "heartbeat", "data": {"timestamp": time.time()}}
                )
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"插件进度 WebSocket 消息处理错误: {e}")
                break
    except WebSocketDisconnect:
        pass
    finally:
        plugin_progress_manager.disconnect(session_id)


@router.get("/plugin-progress/tasks")
async def get_all_progress_tasks(_auth: bool = Depends(require_auth)):
    """获取所有进度任务"""
    return {
        "success": True,
        "tasks": progress_manager.get_all_tasks(),
    }


@router.get("/plugin-progress/tasks/active")
async def get_active_progress_tasks(_auth: bool = Depends(require_auth)):
    """获取活动进度任务"""
    return {
        "success": True,
        "tasks": progress_manager.get_active_tasks(),
    }


@router.get("/plugin-progress/tasks/{task_id}")
async def get_progress_task(task_id: str, _auth: bool = Depends(require_auth)):
    """获取特定进度任务"""
    task = progress_manager.get_task(task_id)
    if not task:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="任务不存在")
    return {"success": True, "task": task.to_dict()}


def get_progress_router() -> APIRouter:
    """获取进度路由器"""
    return router

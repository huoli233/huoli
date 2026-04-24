import time
import asyncio
from typing import Optional
from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    Query,
    HTTPException,
    Depends,
    Cookie,
    Header,
)
from pydantic import BaseModel
from src.common.logger import get_logger
from src.webui.core.security import get_token_manager
from src.webui.core.auth import verify_auth_token_from_cookie_or_header
from src.webui.runtime_config import webui_module_view

logger = get_logger("WS认证")
router = APIRouter(tags=["websocket"])


class WebSocketAuthResponse(BaseModel):
    """WebSocket 认证响应"""

    success: bool
    message: str
    session_id: Optional[str] = None


class WebSocketMessage(BaseModel):
    """WebSocket 消息"""

    type: str
    data: dict
    timestamp: float = time.time()


class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self._active_connections: dict[str, WebSocket] = {}
        self._connection_times: dict[str, float] = {}
        self._user_sessions: dict[str, str] = {}
        self._load_config()

    def _load_config(self) -> None:
        config = webui_module_view("webui_websocket")
        self._max_connections_per_user = int(
            config.get("auth_max_connections_per_user", 5)
        )
        self._auth_receive_timeout_seconds = float(
            config.get("auth_receive_timeout_seconds", 300.0)
        )

    async def connect(
        self, websocket: WebSocket, session_id: str, user_id: str = "anonymous"
    ) -> bool:
        """接受 WebSocket 连接"""
        user_conn_count = sum(
            1 for uid in self._user_sessions.values() if uid == user_id
        )
        if user_conn_count >= self._max_connections_per_user:
            logger.warning(
                f"用户 {user_id} 连接数达到上限 {
                    self._max_connections_per_user}，拒绝新连接"
            )
            await websocket.close(code=4008, reason="连接数达到上限")
            return False
        try:
            await websocket.accept()
            self._active_connections[session_id] = websocket
            self._connection_times[session_id] = time.time()
            self._user_sessions[session_id] = user_id
            logger.info(
                f"WebSocket 连接已建立: session_id={session_id}, user_id={user_id}"
            )
            return True
        except Exception as e:
            logger.error(f"WebSocket 连接失败: {e}")
            return False

    def disconnect(self, session_id: str):
        """断开 WebSocket 连接"""
        if session_id in self._active_connections:
            del self._active_connections[session_id]
        if session_id in self._connection_times:
            del self._connection_times[session_id]
        if session_id in self._user_sessions:
            del self._user_sessions[session_id]
        logger.info(f"WebSocket 连接已断开: session_id={session_id}")

    async def send_message(self, session_id: str, message: dict):
        """发送消息到指定连接"""
        if session_id in self._active_connections:
            try:
                await self._active_connections[session_id].send_json(message)
            except Exception as e:
                logger.warning(
                    f"发送消息失败: session_id={session_id}, error={e}"
                )
                self.disconnect(session_id)

    async def broadcast(
        self, message: dict, target_user_ids: Optional[list[str]] = None
    ):
        """广播消息到所有连接，可选按用户ID过滤"""
        disconnected = []
        for session_id, websocket in self._active_connections.items():
            if target_user_ids is not None:
                uid = self._user_sessions.get(session_id)
                if uid not in target_user_ids:
                    continue
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(
                    f"广播消息失败: session_id={session_id}, error={e}"
                )
                disconnected.append(session_id)
        for session_id in disconnected:
            self.disconnect(session_id)

    def get_connection_count(self) -> int:
        """获取当前连接数"""
        return len(self._active_connections)

    def get_connection_info(self, session_id: str) -> Optional[dict]:
        """获取连接信息"""
        if session_id not in self._active_connections:
            return None
        return {
            "session_id": session_id,
            "user_id": self._user_sessions.get(session_id, "unknown"),
            "connected_at": self._connection_times.get(session_id, 0),
            "uptime": time.time()
            - self._connection_times.get(session_id, time.time()),
        }

    def get_all_connections(self) -> list[dict]:
        """获取所有连接信息"""
        return [
            self.get_connection_info(session_id)
            for session_id in self._active_connections.keys()
        ]


connection_manager = ConnectionManager()


def verify_ws_token(token: str) -> bool:
    """验证 WebSocket 令牌"""
    token_manager = get_token_manager()
    return token_manager.verify_token(token)


def require_ws_auth(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
) -> bool:
    """WebSocket HTTP 端点认证依赖"""
    return verify_auth_token_from_cookie_or_header(
        huoli_session, authorization
    )


@router.websocket("/ws/auth")
async def websocket_auth_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """WebSocket 认证端点"""
    session_id = f"ws_{int(time.time() * 1000)}_{id(websocket)}"
    if not token:
        await websocket.close(code=4001, reason="缺少认证令牌")
        return
    token_manager = get_token_manager()
    if not token_manager.verify_token(token):
        await websocket.close(code=4002, reason="无效的认证令牌")
        return
    connected = await connection_manager.connect(websocket, session_id)
    if not connected:
        return
    try:
        await connection_manager.send_message(
            session_id,
            {
                "type": "auth_success",
                "data": {
                    "session_id": session_id,
                    "message": "WebSocket 认证成功",
                },
            },
        )
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=connection_manager._auth_receive_timeout_seconds,
                )
                message_type = data.get("type", "unknown")
                if message_type == "ping":
                    await connection_manager.send_message(
                        session_id,
                        {"type": "pong", "data": {"timestamp": time.time()}},
                    )
                elif message_type == "heartbeat":
                    await connection_manager.send_message(
                        session_id,
                        {
                            "type": "heartbeat_ack",
                            "data": {"timestamp": time.time()},
                        },
                    )
                elif message_type == "subscribe":
                    channel = data.get("data", {}).get("channel")
                    if channel:
                        await connection_manager.send_message(
                            session_id,
                            {
                                "type": "subscribed",
                                "data": {"channel": channel},
                            },
                        )
                elif message_type == "unsubscribe":
                    channel = data.get("data", {}).get("channel")
                    if channel:
                        await connection_manager.send_message(
                            session_id,
                            {
                                "type": "unsubscribed",
                                "data": {"channel": channel},
                            },
                        )
                else:
                    await connection_manager.send_message(
                        session_id, {"type": "echo", "data": data}
                    )
            except asyncio.TimeoutError:
                await connection_manager.send_message(
                    session_id,
                    {
                        "type": "heartbeat_request",
                        "data": {"timestamp": time.time()},
                    },
                )
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"WebSocket 消息处理错误: {e}")
                break
    except WebSocketDisconnect:
        pass
    finally:
        connection_manager.disconnect(session_id)


@router.get("/ws/connections")
async def get_websocket_connections(_auth: bool = Depends(require_ws_auth)):
    """获取当前 WebSocket 连接列表"""
    return {
        "success": True,
        "count": connection_manager.get_connection_count(),
        "connections": connection_manager.get_all_connections(),
    }


@router.get("/ws/connections/{session_id}")
async def get_websocket_connection(session_id: str, _auth: bool = Depends(require_ws_auth)):
    """获取指定 WebSocket 连接信息"""
    info = connection_manager.get_connection_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="连接不存在")
    return {"success": True, "connection": info}


@router.post("/ws/broadcast")
async def broadcast_to_websockets(message: dict, _auth: bool = Depends(require_ws_auth)):
    """广播消息到所有 WebSocket 连接"""
    await connection_manager.broadcast(
        {
            "type": "broadcast",
            "data": message,
            "timestamp": time.time(),
        }
    )
    return {
        "success": True,
        "message": f"已广播到 {connection_manager.get_connection_count()} 个连接",
    }

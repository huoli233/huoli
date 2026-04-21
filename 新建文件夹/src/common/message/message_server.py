import asyncio
import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocketState

from src.common.logger import get_logger

logger = get_logger("MessageServer")


@dataclass
class ConnectionInfo:
    websocket: WebSocket
    connection_id: str
    connected_at: float
    platform: str = ""
    remote_addr: str = ""
    authenticated: bool = False


class MessageServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        app: Optional[FastAPI] = None,
        custom_logger: Optional[Any] = None,
        enable_token: bool = False,
    ):
        self.host = host
        self.port = port
        self.app = app
        self.logger = custom_logger or logger
        self.enable_token = enable_token
        self._valid_tokens: Set[str] = set()
        self._message_handlers: List[Callable] = []
        self._connections: Dict[str, ConnectionInfo] = {}
        self._connection_lock = asyncio.Lock()
        self._running = False
        self._server_task: Optional[asyncio.Task] = None
        self._ws_endpoint = "/ws"

    def add_valid_token(self, token: str) -> None:
        if token:
            self._valid_tokens.add(token)
            self.logger.info(f"已添加认证令牌: {token[:8]}...")

    def remove_valid_token(self, token: str) -> None:
        self._valid_tokens.discard(token)

    def register_message_handler(self, handler: Callable) -> None:
        self._message_handlers.append(handler)
        self.logger.debug(f"已注册消息处理器: {handler.__name__}")

    async def _authenticate(self, token: Optional[str]) -> bool:
        if not self.enable_token:
            return True
        if not token:
            return False
        return token in self._valid_tokens

    async def _handle_websocket(self, websocket: WebSocket, token: Optional[str] = None) -> None:
        connection_id = str(uuid.uuid4())[:8]
        remote_addr = ""
        try:
            if websocket.client:
                remote_addr = f"{websocket.client.host}:{websocket.client.port}"
        except Exception:
            pass
        await websocket.accept()
        if not await self._authenticate(token):
            await websocket.send_json({"type": "error", "code": 4001, "message": "Unauthorized"})
            await websocket.close(code=4001, reason="Unauthorized")
            self.logger.warning(f"连接认证失败: {remote_addr}")
            return
        conn_info = ConnectionInfo(
            websocket=websocket,
            connection_id=connection_id,
            connected_at=time.time(),
            remote_addr=remote_addr,
            authenticated=True,
        )
        async with self._connection_lock:
            self._connections[connection_id] = conn_info
        self.logger.info(f"WebSocket 连接建立: {connection_id} from {remote_addr}")
        try:
            while True:
                try:
                    raw_data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=300.0
                    )
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping", "time": time.time()})
                    continue
                try:
                    message = json.loads(raw_data)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"无效的 JSON 数据: {e}")
                    continue
                await self._dispatch_message(message, conn_info)
        except WebSocketDisconnect:
            self.logger.info(f"WebSocket 断开: {connection_id}")
        except Exception as e:
            self.logger.error(f"WebSocket 错误: {connection_id} - {e}")
        finally:
            async with self._connection_lock:
                self._connections.pop(connection_id, None)
            self.logger.info(f"连接已清理: {connection_id}")

    async def _dispatch_message(self, message: Dict[str, Any], conn_info: ConnectionInfo) -> None:
        if not message:
            return
        message["_connection_id"] = conn_info.connection_id
        message["_remote_addr"] = conn_info.remote_addr
        for handler in self._message_handlers:
            try:
                result = handler(message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self.logger.error(f"消息处理器错误: {handler.__name__} - {e}")

    async def send_message(self, message: Dict[str, Any], connection_id: Optional[str] = None) -> bool:
        target_conns: List[ConnectionInfo] = []
        async with self._connection_lock:
            if connection_id:
                if connection_id in self._connections:
                    target_conns = [self._connections[connection_id]]
            else:
                target_conns = list(self._connections.values())
        if not target_conns:
            self.logger.warning("没有可用的连接")
            return False
        success = False
        for conn in target_conns:
            try:
                if conn.websocket.client_state == WebSocketState.CONNECTED:
                    await conn.websocket.send_json(message)
                    success = True
            except Exception as e:
                self.logger.error(f"发送消息失败: {conn.connection_id} - {e}")
        return success

    async def broadcast(self, message: Dict[str, Any]) -> int:
        sent_count = 0
        async with self._connection_lock:
            connections = list(self._connections.values())
        for conn in connections:
            try:
                if conn.websocket.client_state == WebSocketState.CONNECTED:
                    await conn.websocket.send_json(message)
                    sent_count += 1
            except Exception as e:
                self.logger.error(f"广播失败: {conn.connection_id} - {e}")
        return sent_count

    def setup_routes(self, app: FastAPI) -> None:
        @app.websocket(self._ws_endpoint)
        async def websocket_endpoint(
            websocket: WebSocket,
            token: Optional[str] = Query(None),
        ):
            await self._handle_websocket(websocket, token)
        self.logger.info(f"WebSocket 路由已注册: {self._ws_endpoint}")

    async def run(self) -> None:
        if self.app:
            self.setup_routes(self.app)
        self._running = True
        self.logger.info(f"消息服务器已启动: ws://{self.host}:{self.port}{self._ws_endpoint}")
        while self._running:
            await asyncio.sleep(1)

    async def start(self) -> None:
        self._server_task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        self._running = False
        async with self._connection_lock:
            for conn in list(self._connections.values()):
                try:
                    await conn.websocket.close()
                except Exception:
                    pass
            self._connections.clear()
        if self._server_task:
            self._server_task.cancel()
        self.logger.info("消息服务器已停止")

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    @property
    def is_running(self) -> bool:
        return self._running


_global_message_server: Optional[MessageServer] = None


def get_message_server() -> MessageServer:
    global _global_message_server
    if _global_message_server is None:
        _global_message_server = MessageServer()
    return _global_message_server


async def create_message_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    app: Optional[FastAPI] = None,
    enable_token: bool = False,
    tokens: Optional[List[str]] = None,
) -> MessageServer:
    server = MessageServer(
        host=host,
        port=port,
        app=app,
        enable_token=enable_token,
    )
    if tokens:
        for token in tokens:
            server.add_valid_token(token)
    if app:
        server.setup_routes(app)
    return server

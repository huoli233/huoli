import asyncio
import os
from typing import Optional, Callable, Awaitable, Any, List

from src.common.logger import get_logger
from src.common.server import get_global_server
from src.config.config import global_config
from .message_server import MessageServer, create_message_server

logger = get_logger("消息API")
_global_api = None


class MessageAPIClient:
    """消息 API 客户端"""

    def __init__(self):
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._connected = False
        self._message_handlers: dict[str, Callable] = {}
        self._server: Optional[MessageServer] = None

    async def connect(self, host: str, port: int) -> bool:
        self._host = host
        self._port = port
        self._connected = True
        logger.info(f"消息 API 已连接: {host}:{port}")
        return True

    async def disconnect(self) -> None:
        self._connected = False
        if self._server:
            await self._server.stop()
        logger.info("消息 API 已断开")

    async def send_message(self, message) -> bool:
        if not self._connected:
            logger.warning("消息 API 未连接")
            return False
        if hasattr(message, 'to_dict'):
            message = message.to_dict()
        if not isinstance(message, dict):
            logger.warning(f"消息格式错误，期望 dict，得到 {type(message)}")
            return False
        if self._server:
            return await self._server.send_message(message)
        logger.debug(f"发送消息: {message.get('message_type', 'unknown')}")
        return True

    def register_handler(
        self, message_type: str, handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        self._message_handlers[message_type] = handler

    def register_message_handler(
        self, handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        """注册消息处理器（通用）"""
        self._message_handlers["default"] = handler
        if self._server:
            self._server.register_message_handler(handler)

    def register_custom_message_handler(
        self, message_type: str, handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        """注册自定义消息处理器"""
        self._message_handlers[message_type] = handler

    async def run(self) -> None:
        """运行消息服务器"""
        if self._server:
            await self._server.run()
        elif self._connected:
            logger.info("消息服务器正在运行...")
            while self._connected:
                await asyncio.sleep(1)
        else:
            logger.warning("消息服务器未连接，无法运行")

    async def start(self) -> None:
        """启动消息服务器"""
        if self._server:
            await self._server.start()
        else:
            await self.connect(self._host or "127.0.0.1", self._port or 8080)

    async def process_message(self, message: dict) -> None:
        msg_type = message.get("message_type", "unknown")
        handler = self._message_handlers.get(msg_type)
        if handler:
            await handler(message)
        else:
            logger.debug(f"未找到消息处理器: {msg_type}")

    @property
    def is_connected(self) -> bool:
        return self._connected


def get_message_api() -> MessageAPIClient:
    global _global_api
    if _global_api is None:
        _global_api = MessageAPIClient()
    return _global_api


get_global_api = get_message_api


async def initialize_message_api() -> MessageAPIClient:
    api = get_message_api()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8080"))
    await api.connect(host, port)
    return api


async def setup_message_server() -> Any:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    maim_config = getattr(global_config, "maim_message", None)
    enable_token = False
    tokens: List[str] = []
    if maim_config:
        auth_tokens = getattr(maim_config, "auth_token", [])
        if auth_tokens:
            enable_token = True
            tokens = auth_tokens if isinstance(auth_tokens, list) else [auth_tokens]
    server = await create_message_server(
        host=host,
        port=port,
        app=get_global_server().get_app(),
        enable_token=enable_token,
        tokens=tokens,
    )
    api = get_message_api()
    api._server = server
    api._host = host
    api._port = port
    api._connected = True
    logger.info(f"自研消息服务器已初始化: {host}:{port}")
    return server

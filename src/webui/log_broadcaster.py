"""日志广播桥接 —— 供 logger 模块将日志推送到 WebUI WebSocket."""
from src.common.logger import get_logger, register_ws_broadcast

logger = get_logger("日志广播")


async def broadcast_log(log_data: dict) -> None:
    """将日志条目转发给 WebSocket 日志缓冲区"""
    try:
        from src.webui.routers.websocket.logs import log_buffer

        log_buffer.add_log(
            level=log_data.get("level", "INFO"),
            message=log_data.get("message", ""),
            source=log_data.get("module", "system"),
        )
    except Exception as _e:
        logger.debug(f"异常: {_e}")


# 模块加载时自动注册广播回调
register_ws_broadcast(broadcast_log)

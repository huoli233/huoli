import asyncio
import time
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from src.common.logger import get_logger
from src.webui.services.state_monitor import build_channel_monitor_state, build_monitor_overview
from src.webui.runtime_config import webui_module_view

logger = get_logger("WS状态监控")
router = APIRouter(tags=["websocket"])


def _websocket_config() -> dict:
    return webui_module_view("webui_websocket")


@router.websocket("/ws/state-monitor")
async def websocket_state_monitor_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    channel_id: Optional[str] = Query(default=None),
    interval: float = Query(default=1.0, ge=0.1, le=30.0),
):
    session_id = f"state_{int(time.time() * 1000)}_{id(websocket)}"
    await websocket.accept()
    logger.info(f"状态监控 WebSocket 已建立: {session_id}, 会话={channel_id or '全部群聊/私聊'}")
    config = _websocket_config()
    min_interval = float(config.get("state_monitor_min_interval_seconds", 0.5))
    max_interval = float(config.get("state_monitor_max_interval_seconds", 10.0))
    default_interval = float(config.get("state_monitor_default_interval_seconds", 1.0))
    interval = default_interval if interval is None else interval
    interval = max(min_interval, min(max_interval, interval))

    async def send_snapshot(target_channel: Optional[str] = None) -> None:
        chosen_channel = target_channel if target_channel is not None else channel_id
        if chosen_channel:
            payload = await build_channel_monitor_state(chosen_channel)
            if payload is None:
                await websocket.send_json(
                    {
                        "type": "state_error",
                        "data": {
                            "channel_id": chosen_channel,
                            "message": "Heartflow chat is not active",
                        },
                    }
                )
                return
            await websocket.send_json({"type": "state_snapshot", "data": payload})
        else:
            await websocket.send_json({"type": "state_overview", "data": await build_monitor_overview()})

    try:
        await send_snapshot()
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=interval)
                msg_type = data.get("type", "unknown")
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong", "data": {"timestamp": time.time()}})
                elif msg_type == "refresh":
                    requested_channel = data.get("data", {}).get("channel_id")
                    await send_snapshot(requested_channel)
                elif msg_type == "switch_channel":
                    channel_id = data.get("data", {}).get("channel_id")
                    await send_snapshot()
                else:
                    await websocket.send_json({"type": "echo", "data": data})
            except asyncio.TimeoutError:
                await send_snapshot()
            except WebSocketDisconnect:
                break
    finally:
        logger.info(f"状态监控 WebSocket 已断开: {session_id}")


@router.get("/ws/state-monitor/status")
async def get_state_monitor_status():
    return {"success": True, "message": "state monitor websocket router ready"}

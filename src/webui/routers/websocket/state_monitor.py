import asyncio
import time
from typing import Optional

from fastapi import APIRouter, Cookie, Depends, Header, Query, WebSocket, WebSocketDisconnect

from src.common.logger import get_logger
from src.webui.core.auth import verify_auth_token_from_cookie_or_header
from src.webui.core.security import get_token_manager
from src.webui.services.state_monitor import build_channel_monitor_state, build_monitor_overview

logger = get_logger("WS状态监控")
router = APIRouter(tags=["websocket"])


def require_auth(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
) -> bool:
    return verify_auth_token_from_cookie_or_header(huoli_session, authorization)


@router.websocket("/ws/state-monitor")
async def websocket_state_monitor_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    channel_id: Optional[str] = Query(default=None),
    interval: float = Query(default=1.0, ge=0.5, le=10.0),
):
    token_manager = get_token_manager()
    cookie_token = websocket.cookies.get("huoli_session")
    active_token = cookie_token or token
    if not active_token or not token_manager.verify_token(active_token):
        await websocket.close(code=4002, reason="无效的认证令牌")
        return

    session_id = f"state_{int(time.time() * 1000)}_{id(websocket)}"
    await websocket.accept()
    logger.info(f"状态监控 WebSocket 已建立: {session_id}, channel={channel_id or 'all'}")

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
async def get_state_monitor_status(_auth: bool = Depends(require_auth)):
    return {"success": True, "message": "state monitor websocket router ready"}

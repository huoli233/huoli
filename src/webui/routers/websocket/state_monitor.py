import asyncio
import time
from typing import Optional

from fastapi import APIRouter, Cookie, Header, HTTPException, Query, WebSocket, WebSocketDisconnect

from src.common.logger import get_logger
from src.webui.core.auth import verify_auth_token_from_cookie_or_header
from src.webui.core.security import get_token_manager
from src.webui.runtime_config import webui_module_view
from src.webui.services.state_monitor import build_channel_monitor_state, build_monitor_overview

logger = get_logger("WS状态监控")
router = APIRouter(tags=["websocket"])


def _websocket_config() -> dict:
    return webui_module_view("webui_websocket")


def _close_code_for_auth_error(detail: object) -> int:
    detail_text = str(detail or "")
    if "无效" in detail_text or "过期" in detail_text:
        return 4002
    return 4001


async def _authorize_state_monitor_websocket(
    websocket: WebSocket,
    token: Optional[str],
    huoli_session: Optional[str],
    authorization: Optional[str],
) -> bool:
    token_manager = get_token_manager()

    if token and token_manager.verify_token(token):
        return True

    try:
        verify_auth_token_from_cookie_or_header(huoli_session, authorization)
        return True
    except HTTPException as exc:
        reason = str(exc.detail or "未提供有效的认证信息")
        close_code = _close_code_for_auth_error(exc.detail)
        if token:
            reason = "无效的认证令牌"
            close_code = 4002
        logger.warning(f"状态监控 WebSocket 认证失败: {reason}")
        await websocket.close(code=close_code, reason=reason)
        return False


@router.websocket("/ws/state-monitor")
async def websocket_state_monitor_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    channel_id: Optional[str] = Query(default=None),
    interval: float = Query(default=1.0, ge=0.1, le=30.0),
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    session_id = f"state_{int(time.time() * 1000)}_{id(websocket)}"
    if not await _authorize_state_monitor_websocket(
        websocket=websocket,
        token=token,
        huoli_session=huoli_session,
        authorization=authorization,
    ):
        return
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

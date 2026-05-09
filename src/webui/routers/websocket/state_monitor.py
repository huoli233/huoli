import asyncio
import time
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from src.common.logger import get_logger
from src.webui.services.state_monitor import build_channel_monitor_state
from src.webui.services.runtime_state_hub import get_runtime_state_hub

logger = get_logger("WS状态监控")
router = APIRouter(tags=["websocket"])


@router.websocket("/ws/state-monitor")
async def websocket_state_monitor_endpoint(
    websocket: WebSocket,
    channel_id: Optional[str] = Query(default=None),
):
    session_id = f"state_{int(time.time() * 1000)}_{id(websocket)}"
    await websocket.accept()
    if not channel_id:
        await websocket.send_json(
            {
                "type": "state_error",
                "data": {
                    "message": "state monitor websocket requires channel_id",
                },
            }
        )
        await websocket.close(code=1008, reason="channel_id required")
        logger.info(f"状态监控 WebSocket 拒绝未指定会话: {session_id}")
        return

    logger.debug(f"状态实时事件通道已建立: {session_id}, 会话={channel_id}")
    hub = get_runtime_state_hub()
    subscription: Optional[asyncio.Queue] = None
    client_queue: asyncio.Queue = asyncio.Queue(maxsize=20)

    async def read_client_messages() -> None:
        try:
            while True:
                data = await websocket.receive_json()
                if client_queue.full():
                    try:
                        client_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await client_queue.put(data)
        except WebSocketDisconnect:
            await client_queue.put({"type": "__disconnect__"})

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

    async def rebuild_subscription(old_channel: Optional[str], target_channel: Optional[str]) -> None:
        nonlocal subscription
        if subscription is not None:
            await hub.unsubscribe(subscription, old_channel)
        subscription = await hub.subscribe(target_channel)

    try:
        await send_snapshot()
        subscription = await hub.subscribe(channel_id)
        reader_task = asyncio.create_task(read_client_messages())
        while True:
            event_task = asyncio.create_task(subscription.get())
            client_task = asyncio.create_task(client_queue.get())
            try:
                done, pending = await asyncio.wait(
                    {client_task, event_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if client_task in done:
                    data = client_task.result()
                    msg_type = data.get("type", "unknown")
                    if msg_type == "__disconnect__":
                        break
                    if msg_type == "ping":
                        await websocket.send_json(
                            {
                                "type": "pong",
                                "data": {
                                    "timestamp": time.time(),
                                    "server_time": time.time(),
                                },
                            }
                        )
                    elif msg_type == "switch_channel":
                        target_channel = data.get("data", {}).get("channel_id")
                        if not target_channel:
                            await websocket.send_json(
                                {
                                    "type": "state_error",
                                    "data": {
                                        "message": "state monitor switch_channel requires channel_id",
                                    },
                                }
                            )
                            continue
                        old_channel = channel_id
                        channel_id = target_channel
                        await send_snapshot()
                        await rebuild_subscription(old_channel, channel_id)
                    else:
                        await websocket.send_json({"type": "echo", "data": data})
                elif event_task in done:
                    event = event_task.result()
                    await websocket.send_json({"type": event.get("type", "state_delta"), "data": event})
            except WebSocketDisconnect:
                break
    finally:
        if "reader_task" in locals():
            reader_task.cancel()
        if subscription is not None:
            await hub.unsubscribe(subscription, channel_id)
        logger.debug(f"状态实时事件通道已断开: {session_id}")


@router.get("/ws/state-monitor/status")
async def get_state_monitor_status():
    return {"success": True, "message": "state monitor websocket router ready"}

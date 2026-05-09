from fastapi import APIRouter, HTTPException, Query

from src.chat.heart_flow.heartfc_state_exporter import (
    export_heartfc_state,
    list_heartfc_chats,
)
from src.chat.heart_flow.heartfc_thresholds import get_heartfc_thresholds
from src.webui.services.state_monitor import (
    build_channel_monitor_state,
    build_monitor_overview,
    is_channel_monitor_packet,
)
from src.webui.services.runtime_state_hub import MONITOR_OVERVIEW_CHANNEL_ID, get_runtime_state_hub

router = APIRouter(prefix="/api/heartflow", tags=["heartflow"])


@router.get("/chats")
async def get_heartflow_chats():
    """List active Heartflow chat instances without creating new ones."""

    return {"success": True, **list_heartfc_chats()}


@router.get("/state/{channel_id}")
async def get_heartflow_state(channel_id: str):
    """Return canonical state domains for an active Heartflow channel."""

    state = await export_heartfc_state(channel_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Heartflow chat is not active")
    return {"success": True, "state": state}


@router.get("/monitor")
async def get_heartflow_monitor_overview():
    """Return monitor overview for all active Heartflow channels."""

    return {"success": True, "monitor": await build_monitor_overview()}


@router.get("/monitor/live")
async def wait_heartflow_monitor_overview(
    after_version: int = Query(default=0, ge=0),
    timeout: float = Query(default=25.0, ge=0.1, le=30.0),
):
    """Long-poll monitor overview and return as soon as the channel list changes."""

    hub = get_runtime_state_hub()
    overview = await build_monitor_overview() if int(after_version or 0) <= 0 else await hub.get_snapshot(MONITOR_OVERVIEW_CHANNEL_ID)
    live_chats = list_heartfc_chats()
    if overview is None or (
        int(overview.get("active_count", 0) or 0) <= 0
        and int(live_chats.get("active_count", 0) or 0) > 0
    ):
        overview = await build_monitor_overview()
    current_version = int(overview.get("state_version", 0) or 0)
    if current_version > int(after_version or 0):
        return {"success": True, "changed": True, "monitor": overview}
    event = await hub.wait_for_update(
        MONITOR_OVERVIEW_CHANNEL_ID,
        after_version=after_version,
        timeout_seconds=timeout,
    )
    if event is not None:
        return {
            "success": True,
            "changed": True,
            "monitor": event.get("data", overview),
            "event": {
                "type": event.get("type", "state_snapshot"),
                "version": event.get("version", 0),
                "reason": event.get("reason", ""),
            },
        }
    latest = await hub.get_snapshot(MONITOR_OVERVIEW_CHANNEL_ID)
    return {"success": True, "changed": False, "monitor": latest or overview}


@router.get("/monitor/{channel_id}")
async def get_heartflow_monitor_state(channel_id: str):
    """Return canonical monitor packet for a single Heartflow channel."""

    monitor = await build_channel_monitor_state(channel_id)
    if monitor is None:
        raise HTTPException(status_code=404, detail="Heartflow chat is not active")
    return {"success": True, "monitor": monitor}


@router.get("/monitor/{channel_id}/live")
async def wait_heartflow_monitor_state(
    channel_id: str,
    after_version: int = Query(default=0, ge=0),
    timeout: float = Query(default=25.0, ge=0.1, le=30.0),
):
    """Long-poll one Heartflow monitor packet without using WebSocket."""

    hub = get_runtime_state_hub()
    cached = await hub.get_snapshot(channel_id)
    if (
        is_channel_monitor_packet(cached, channel_id)
        and int(cached.get("state_version", 0) or 0) > int(after_version or 0)
    ):
        return {"success": True, "changed": True, "monitor": cached}
    if not is_channel_monitor_packet(cached, channel_id):
        monitor = await build_channel_monitor_state(channel_id)
        if monitor is None:
            raise HTTPException(status_code=404, detail="Heartflow chat is not active")
        return {"success": True, "changed": True, "monitor": monitor}
    event = await hub.wait_for_update(channel_id, after_version=after_version, timeout_seconds=timeout)
    if event is None:
        latest = await hub.get_snapshot(channel_id)
        if latest is None:
            raise HTTPException(status_code=404, detail="Heartflow chat is not active")
        return {"success": True, "changed": False, "monitor": latest}
    monitor = None
    if (
        event.get("type") == "state_snapshot"
        and str(event.get("reason", "") or "").startswith("monitor_")
        and is_channel_monitor_packet(event.get("data"), channel_id)
    ):
        monitor = event.get("data")
    else:
        monitor = await build_channel_monitor_state(channel_id)
    if not is_channel_monitor_packet(monitor, channel_id):
        monitor = await hub.get_snapshot(channel_id)
    if not is_channel_monitor_packet(monitor, channel_id):
        monitor = await build_channel_monitor_state(channel_id)
    if monitor is None:
        raise HTTPException(status_code=404, detail="Heartflow chat is not active")
    return {
        "success": True,
        "changed": True,
        "monitor": monitor,
        "event": {
            "type": event.get("type", "state_delta"),
            "version": event.get("version", 0),
            "path": event.get("path", ""),
            "reason": event.get("reason", ""),
        },
    }


@router.get("/thresholds")
async def get_heartflow_thresholds():
    """Return effective HeartFC threshold configuration."""

    return {"success": True, "thresholds": get_heartfc_thresholds().to_dict()}

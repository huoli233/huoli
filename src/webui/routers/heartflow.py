from fastapi import APIRouter, HTTPException

from src.chat.heart_flow.heartfc_state_exporter import (
    export_heartfc_state,
    list_heartfc_chats,
)
from src.chat.heart_flow.heartfc_thresholds import get_heartfc_thresholds
from src.webui.services.state_monitor import build_channel_monitor_state, build_monitor_overview

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


@router.get("/monitor/{channel_id}")
async def get_heartflow_monitor_state(channel_id: str):
    """Return canonical monitor packet for a single Heartflow channel."""

    monitor = await build_channel_monitor_state(channel_id)
    if monitor is None:
        raise HTTPException(status_code=404, detail="Heartflow chat is not active")
    return {"success": True, "monitor": monitor}


@router.get("/thresholds")
async def get_heartflow_thresholds():
    """Return effective HeartFC threshold configuration."""

    return {"success": True, "thresholds": get_heartfc_thresholds().to_dict()}

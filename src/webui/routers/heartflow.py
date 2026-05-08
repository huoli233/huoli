from fastapi import APIRouter, HTTPException
import asyncio
import time

from src.chat.heart_flow.heartfc_state_exporter import (
    export_heartfc_state,
    list_heartfc_chats,
)
from src.chat.heart_flow.heartfc_thresholds import get_heartfc_thresholds
from src.common.logger import get_logger
from src.webui.services.config_scope import build_config_scope_snapshot
from src.webui.services.state_monitor import build_channel_monitor_state, build_monitor_overview

router = APIRouter(prefix="/api/heartflow", tags=["heartflow"])
logger = get_logger("WebUI状态页")


def _config_scope_fallback(reason: str) -> dict:
    return {
        "updated_at": time.time(),
        "summary": {
            "total": 0,
            "system": 0,
            "mixed": 0,
            "user": 0,
            "editable": 0,
        },
        "modules": [],
        "degraded": True,
        "message": reason,
    }


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


@router.get("/config-scope")
async def get_heartflow_config_scope():
    """Return config scope metadata without blocking live status sync."""

    try:
        config_scope = await asyncio.wait_for(
            asyncio.to_thread(build_config_scope_snapshot),
            timeout=2.0,
        )
    except asyncio.TimeoutError:
        logger.warning("状态页配置分级构建超时(2s)，返回降级快照")
        config_scope = _config_scope_fallback("配置分级构建超时，实时状态同步不受影响")
    except Exception as exc:
        logger.warning(f"状态页配置分级构建失败，返回降级快照: {exc}")
        config_scope = _config_scope_fallback("配置分级构建失败，实时状态同步不受影响")
    return {"success": True, "config_scope": config_scope}

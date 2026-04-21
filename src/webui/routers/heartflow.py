from typing import Optional

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException

from src.chat.heart_flow.heartfc_state_exporter import (
    export_heartfc_state,
    list_heartfc_chats,
)
from src.chat.heart_flow.heartfc_thresholds import get_heartfc_thresholds
from src.webui.core.auth import verify_auth_token_from_cookie_or_header

router = APIRouter(prefix="/api/heartflow", tags=["heartflow"])


def require_auth(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
) -> bool:
    return verify_auth_token_from_cookie_or_header(huoli_session, authorization)


@router.get("/chats")
async def get_heartflow_chats(_auth: bool = Depends(require_auth)):
    """List active Heartflow chat instances without creating new ones."""

    return {"success": True, **list_heartfc_chats()}


@router.get("/state/{channel_id}")
async def get_heartflow_state(channel_id: str, _auth: bool = Depends(require_auth)):
    """Export a read-only state snapshot for an active Heartflow channel."""

    state = await export_heartfc_state(channel_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Heartflow chat is not active")
    return {"success": True, "state": state}


@router.get("/thresholds")
async def get_heartflow_thresholds(_auth: bool = Depends(require_auth)):
    """Return effective HeartFC threshold configuration."""

    return {"success": True, "thresholds": get_heartfc_thresholds().to_dict()}


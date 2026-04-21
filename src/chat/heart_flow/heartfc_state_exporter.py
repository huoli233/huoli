import enum
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple

from src.chat.heart_flow.heartfc_thresholds import get_heartfc_thresholds


def _plain(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if is_dataclass(value):
        try:
            return _plain(asdict(value), depth + 1)
        except Exception:
            return repr(value)
    if isinstance(value, dict):
        return {str(k): _plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item, depth + 1) for item in value]
    if hasattr(value, "__dict__"):
        safe: Dict[str, Any] = {}
        for key, item in vars(value).items():
            if str(key).startswith("_"):
                continue
            safe[str(key)] = _plain(item, depth + 1)
        return safe or repr(value)
    return repr(value)


def _find_key(mapping: Dict[Any, Any], channel_id: str) -> Optional[Any]:
    if channel_id in mapping:
        return channel_id
    for key in mapping.keys():
        if str(key) == str(channel_id):
            return key
    return None


def _find_active_chat(channel_id: str) -> Tuple[Optional[Any], Optional[Any]]:
    from src.chat.heart_flow.heartflow import heartflow

    key = _find_key(heartflow.heartflow_chat_list, channel_id)
    if key is None:
        return None, None
    return key, heartflow.heartflow_chat_list.get(key)


def list_heartfc_chats() -> Dict[str, Any]:
    from src.chat.heart_flow.heartflow import heartflow
    from src.chat.message_receive.chat_stream import get_chat_manager

    now = time.time()
    streams = get_chat_manager()
    chats = []
    for chat_id, chat in heartflow.heartflow_chat_list.items():
        stream = streams.get_stream(str(chat_id))
        last_active = float(heartflow._active_since.get(chat_id, 0.0) or 0.0)
        chats.append(
            {
                "channel_id": str(chat_id),
                "chat_type": "group" if stream and stream.group_info else "private",
                "class_name": chat.__class__.__name__,
                "last_active": last_active,
                "idle_seconds": round(max(0.0, now - last_active), 3) if last_active else None,
                "platform": getattr(stream, "platform", "") if stream else "",
                "group_id": getattr(getattr(stream, "group_info", None), "group_id", None) if stream else None,
                "user_id": getattr(getattr(stream, "user_info", None), "user_id", None) if stream else None,
            }
        )
    return {
        "active_count": len(chats),
        "channels": chats,
    }


def _cached_world_snapshot(chat: Any) -> Dict[str, Any]:
    snapshot = getattr(chat, "_tick_world_snapshot", None)
    if snapshot is None:
        return {"available": False, "reason": "no_cached_snapshot"}
    return {"available": True, "snapshot": _plain(snapshot)}


def _energy_snapshot(channel_id: str) -> Dict[str, Any]:
    from src.chat.heart_flow.energy_manager import EnergyChainDimension

    dimension = getattr(EnergyChainDimension, "_singleton", None)
    if dimension is None:
        return {"available": False, "reason": "energy_dimension_not_initialized"}
    channels = getattr(dimension, "_channels", {}) or {}
    key = _find_key(channels, channel_id)
    if key is None:
        return {"available": False, "reason": "no_channel_energy_state"}
    state = channels[key]
    payload = _plain(state.to_snapshot())
    if not isinstance(payload, dict):
        payload = {"raw": payload}
    payload.update(
        {
            "chat_ratio": round(state.chat_ratio(), 3),
            "thinking_ratio": round(state.thinking_ratio(), 3),
            "combined_ratio": round(state.combined_ratio(), 3),
        }
    )
    return {"available": True, "snapshot": payload}


def _emotion_snapshot(channel_id: str) -> Dict[str, Any]:
    import src.chat.heart_flow.emotion_driven_core as emotion_module

    core = getattr(emotion_module, "_emotion_core", None)
    if core is None:
        return {"available": False, "reason": "emotion_core_not_initialized"}
    states = getattr(core, "_states", {}) or {}
    key = _find_key(states, channel_id)
    if key is None:
        return {"available": False, "reason": "no_channel_emotion_state"}
    return {"available": True, "snapshot": _plain(core.get_state_snapshot(key))}


def _dormancy_snapshot(channel_id: str) -> Dict[str, Any]:
    from src.chat.heart_flow.rest_handler import DormancySupervisor

    supervisor = getattr(DormancySupervisor, "_solo", None)
    if supervisor is None:
        return {"available": False, "reason": "dormancy_supervisor_not_initialized"}
    return {"available": True, "snapshot": _plain(supervisor.snapshot(channel_id))}


def _planner_snapshot(channel_id: str) -> Dict[str, Any]:
    import src.core.unified_planner as planner_module

    planner = getattr(planner_module, "_unified_planner", None)
    if planner is None:
        return {"available": False, "reason": "planner_not_initialized"}
    return {
        "available": True,
        "status": _plain(planner.get_status(channel_id)),
        "context": _plain(planner.get_context_stats(channel_id)),
    }


def _chat_runtime_snapshot(chat: Any) -> Dict[str, Any]:
    fields = {
        "pipeline_ready": "_pipeline_ready",
        "enhanced_round": "_enhanced_round",
        "last_dashboard_status_line": "_last_dashboard_status_line",
        "last_gateway_verdict": "_last_gateway_verdict",
        "last_reactive_plan": "_last_reactive_plan",
        "last_rest_governor": "_last_rest_governor_verdict",
        "last_behavior_governor": "_last_behavior_governor_verdict",
        "last_model_governor": "_last_model_governor_verdict",
        "cached_night_phase": "_cached_night_phase",
        "cached_metabolism_constraints": "_cached_metabolism_constraints",
        "last_relation_snapshot": "_last_relation_snapshot",
    }
    return {name: _plain(getattr(chat, attr, None)) for name, attr in fields.items()}


async def export_heartfc_state(channel_id: str) -> Optional[Dict[str, Any]]:
    key, chat = _find_active_chat(channel_id)
    if chat is None:
        return None
    resolved_id = str(key)
    return {
        "channel_id": resolved_id,
        "class_name": chat.__class__.__name__,
        "exported_at": time.time(),
        "runtime": _chat_runtime_snapshot(chat),
        "world_snapshot": _cached_world_snapshot(chat),
        "energy": _energy_snapshot(resolved_id),
        "emotion": _emotion_snapshot(resolved_id),
        "dormancy": _dormancy_snapshot(resolved_id),
        "planner": _planner_snapshot(resolved_id),
        "thresholds": get_heartfc_thresholds().to_dict(),
    }

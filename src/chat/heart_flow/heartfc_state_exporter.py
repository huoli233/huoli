import enum
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple

from src.chat.heart_flow.heartfc_thresholds import get_heartfc_thresholds
from src.common.logger import get_logger

logger = get_logger("heartflow_state_exporter")


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
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        safe: Dict[str, Any] = {}
        for key, item in vars(value).items():
            if str(key).startswith("_"):
                continue
            safe[str(key)] = _plain(item, depth + 1)
        return safe or repr(value)
    return repr(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


def _resource_phase(combined_ratio: float) -> str:
    if combined_ratio >= 0.85:
        return "full"
    if combined_ratio >= 0.60:
        return "adequate"
    if combined_ratio >= 0.35:
        return "low"
    if combined_ratio >= 0.15:
        return "critical"
    return "depleted"


def _derive_scene_heat(active_users: int, vexation: float, weariness: float) -> tuple[str, float]:
    heat_score = _clamp(active_users / 12.0 + vexation * 0.15 - weariness * 0.08, 0.0, 1.0)
    if heat_score >= 0.78:
        return "heated", heat_score
    if heat_score >= 0.55:
        return "lively", heat_score
    if heat_score >= 0.28:
        return "normal", heat_score
    if heat_score >= 0.12:
        return "quiet", heat_score
    return "dead", heat_score


def _extract_world_snapshot(chat: Any) -> Dict[str, Any]:
    snapshot = getattr(chat, "_tick_world_snapshot", None)
    if snapshot is None:
        return {}
    if hasattr(snapshot, "to_canonical_state"):
        try:
            return _plain(snapshot.to_canonical_state())
        except Exception as exc:
            logger.debug(f"世界快照 canonical 导出失败: {exc}")
    return _plain(snapshot)


def _extract_runtime_snapshot(chat: Any) -> Dict[str, Any]:
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
        "participant_summary": "_cached_participant_summary",
        "last_user_id": "_last_user_id",
        "last_proactive_target_user_id": "_last_proactive_target_user_id",
    }
    return {name: _plain(getattr(chat, attr, None)) for name, attr in fields.items()}


def _extract_energy_domain(channel_id: str) -> Dict[str, Any]:
    from src.chat.heart_flow.energy_manager import EnergyChainDimension

    dimension = getattr(EnergyChainDimension, "_singleton", None)
    if dimension is None:
        return {}
    channels = getattr(dimension, "_channels", {}) or {}
    key = _find_key(channels, channel_id)
    if key is None:
        return {}
    state = channels[key]
    combined_ratio = state.combined_ratio()
    return {
        "energy_reserve_ratio": round(combined_ratio, 3),
        "energy_phase": _resource_phase(combined_ratio),
        "chat_reserve": round(state.chat_pool, 2),
        "chat_capacity": round(state.chat_ceiling, 2),
        "thinking_reserve": round(state.thinking_value, 2),
        "thinking_capacity": round(state.thinking_ceiling, 2),
        "activity_index": round(state.activity_level, 2),
        "irritation_load": round(state.annoyance_level, 2),
        "social_field_score": round(state.social_value, 2),
        "chain_count": state.chain_count,
        "night_reply_count": state.night_reply_count,
        "total_consumed_today": round(state.total_consumed_today, 2),
        "last_recovery_source": state.last_recovery_source,
        "last_irritation_relief_source": state.last_annoyance_relief_source,
    }


def _extract_emergence_state(channel_id: str) -> Dict[str, Any]:
    import src.chat.heart_flow.emotion_driven_core as emotion_module

    core = getattr(emotion_module, "_emotion_core", None)
    if core is None:
        return {}
    states = getattr(core, "_states", {}) or {}
    key = _find_key(states, channel_id)
    if key is None:
        return {}
    state = states[key]
    return {
        "boredom_load": round(_safe_float(getattr(state, "boredom", 0.0)), 3),
        "loneliness_load": round(_safe_float(getattr(state, "loneliness", 0.0)), 3),
        "environment_fatigue_load": round(
            _safe_float(getattr(state, "environmental_fatigue", 0.0)), 3
        ),
        "social_desire": round(_safe_float(getattr(state, "social_desire", 0.0)), 3),
        "mood_bias": round(_safe_float(getattr(state, "mood", 0.5)), 3),
        "energy_ratio": round(_safe_float(getattr(state, "energy", 0.5)), 3),
        "curiosity_drive": round(_safe_float(getattr(state, "curiosity", 0.0)), 3),
        "initiative_drive": round(_safe_float(getattr(state, "proactive_willingness", 0.0)), 3),
        "withdrawal_drive": round(_safe_float(getattr(state, "withdrawal_tendency", 0.0)), 3),
        "silence_seconds": round(_safe_float(getattr(state, "silence_duration", 0.0)), 2),
        "unanswered_count": _safe_int(getattr(state, "unanswered_count", 0)),
        "feeling_text": str(getattr(state, "feeling_description", "") or ""),
    }


def _extract_relationship_profile(chat: Any, world_state: Dict[str, Any]) -> Dict[str, Any]:
    world_target = world_state.get("target", {}) if isinstance(world_state, dict) else {}
    runtime = _extract_runtime_snapshot(chat)
    target_uid = str(
        runtime.get("last_proactive_target_user_id")
        or runtime.get("last_user_id")
        or world_target.get("group_friend_id")
        or ""
    ).strip()
    target_name = str(
        world_target.get("group_friend_name")
        or world_target.get("display_name")
        or ""
    ).strip()

    state = None
    if target_uid and getattr(chat, "stream_id", ""):
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker

            tracker = get_emotion_tracker(str(getattr(chat, "stream_id", "") or ""))
            state = tracker.get_user_state(target_uid, create_if_missing=False)
        except Exception as exc:
            logger.debug(f"提取关系档案失败: {exc}")

    rapport_score = _safe_float(getattr(state, "affection", world_target.get("affection", 0.0)))
    trust_score = _safe_float(
        getattr(state, "trust_value", world_target.get("trust_value", world_target.get("trust_score", 0.0)))
    )
    irritation_load = _safe_float(
        getattr(state, "annoyance", world_target.get("annoyance_value", 0.0))
    )
    trauma_load = _safe_float(getattr(state, "trauma_score", world_target.get("trauma_score", 0.0)))
    pressure_load = _safe_float(
        getattr(state, "psychological_pressure", world_target.get("psychological_pressure", 0.0))
    )
    chaos_load = _safe_float(getattr(state, "inner_chaos", 0.0))
    mask_load = _safe_float(getattr(state, "surface_mask", 0.0))
    compliance_index = _safe_float(getattr(state, "submission_level", 0.0))
    conditioning_progress = _safe_float(getattr(state, "training_progress", 0.0))
    resistance_index = _safe_float(getattr(state, "training_resistance", 100.0))
    authority_imprint = _safe_float(getattr(state, "authority_level", 0.0))
    intimacy_depth = _safe_float(getattr(state, "intimacy_level", 0.0))
    boundary_overstep_index = _safe_float(getattr(state, "excess_level", 0.0))
    interaction_count = _safe_int(
        getattr(state, "interaction_count", world_target.get("interaction_count", 0))
    )
    relationship_label = str(
        getattr(state, "relationship", world_target.get("custom_label", "")) or world_target.get("mood", "陌生人")
    )
    display_name = str(
        getattr(state, "nickname", "") or target_name or target_uid or "当前目标"
    )
    return {
        "user_id": target_uid,
        "display_name": display_name,
        "relationship_label": relationship_label,
        "rapport_score": round(rapport_score, 2),
        "trust_score": round(trust_score, 2),
        "irritation_load": round(irritation_load, 2),
        "trauma_load": round(trauma_load, 2),
        "pressure_load": round(pressure_load, 2),
        "chaos_load": round(chaos_load, 2),
        "mask_load": round(mask_load, 2),
        "compliance_index": round(compliance_index, 2),
        "conditioning_progress": round(conditioning_progress, 2),
        "resistance_index": round(resistance_index, 2),
        "authority_imprint": round(authority_imprint, 2),
        "intimacy_depth": round(intimacy_depth, 2),
        "boundary_overstep_index": round(boundary_overstep_index, 2),
        "interaction_count": interaction_count,
        "is_blocked": bool(getattr(state, "is_blocked", False)),
        "is_protected": bool(getattr(state, "is_protected", False)),
    }


def _extract_domains(chat: Any, channel_id: str) -> Dict[str, Any]:
    world_state = _extract_world_snapshot(chat)
    runtime = _extract_runtime_snapshot(chat)
    energy = _extract_energy_domain(channel_id)
    emergence = _extract_emergence_state(channel_id)
    relation = _extract_relationship_profile(chat, world_state)
    subject = world_state.get("subject", {}) if isinstance(world_state, dict) else {}
    scene = world_state.get("scene", {}) if isinstance(world_state, dict) else {}
    behavior = world_state.get("behavior", {}) if isinstance(world_state, dict) else {}
    meta = world_state.get("meta", {}) if isinstance(world_state, dict) else {}
    participant_summary = runtime.get("participant_summary", {}) if isinstance(runtime.get("participant_summary"), dict) else {}
    active_users = scene.get("active_users", []) if isinstance(scene.get("active_users", []), list) else []
    topic_focus = list(scene.get("current_topics", []) or [])
    scene_heat, heat_score = _derive_scene_heat(
        len(active_users),
        _safe_float(scene.get("vexation", 0.0)),
        _safe_float(scene.get("weariness", 0.0)),
    )
    readiness_score = _clamp(
        0.5
        + relation.get("rapport_score", 0.0) / 220.0
        + relation.get("trust_score", 0.0) / 220.0
        - relation.get("irritation_load", 0.0) / 140.0
        - relation.get("pressure_load", 0.0) / 220.0,
        0.0,
        1.0,
    )
    hostility_detected = bool(
        relation.get("irritation_load", 0.0) >= 60
        or behavior.get("category") in {"negative", "harassment"}
    )
    engagement_state = "neutral"
    if readiness_score >= 0.72:
        engagement_state = "eager"
    elif hostility_detected:
        engagement_state = "hostile"
    elif readiness_score <= 0.3:
        engagement_state = "reluctant"
    trauma_load = relation.get("trauma_load", 0.0)
    pressure_load = relation.get("pressure_load", 0.0)
    chaos_load = relation.get("chaos_load", 0.0)
    flashback_probability = _clamp(
        trauma_load / 10.0 * 0.45 + pressure_load / 100.0 * 0.2 + chaos_load / 10.0 * 0.25,
        0.0,
        1.0,
    )
    cognitive_drag = _clamp(
        trauma_load / 10.0 * 0.7 + pressure_load / 100.0 * 0.25,
        0.0,
        1.0,
    )
    energy_ratio = _safe_float(energy.get("energy_reserve_ratio", 0.5))
    irritation_load = _safe_float(energy.get("irritation_load", 0.0))
    fatigue_level = _clamp((1.0 - energy_ratio) + emergence.get("environment_fatigue_load", 0.0) * 0.35, 0.0, 1.0)
    patience_level = _clamp(100.0 - irritation_load - fatigue_level * 30.0, 0.0, 100.0)
    pending_seconds = _safe_float(subject.get("pending_elapsed_sec", 0.0))
    pending_active = bool(subject.get("pending_active", False))
    cooling_down = bool(subject.get("phase") in {"dormant", "light_rest"} or relation.get("is_blocked", False))
    should_reengage = bool(
        emergence.get("initiative_drive", 0.0) >= 0.72
        or pending_active
        or runtime.get("last_reactive_plan")
    )
    return {
        "emotion_axis": {
            "patience_level": round(patience_level, 2),
            "irritation_load": round(irritation_load, 2),
            "fatigue_level": round(fatigue_level, 3),
            "initiative_pull": round(_safe_float(emergence.get("initiative_drive", 0.0)), 3),
        },
        "rapport_trust": {
            "rapport_score": relation.get("rapport_score", 0.0),
            "trust_score": relation.get("trust_score", 0.0),
            "relationship_label": relation.get("relationship_label", "陌生人"),
            "interaction_count": relation.get("interaction_count", 0),
        },
        "boundary_guard": {
            "blocked": bool(relation.get("is_blocked", False)),
            "protected": bool(relation.get("is_protected", False)),
            "guard_severity": round(
                _clamp(relation.get("irritation_load", 0.0) / 100.0 + trauma_load / 10.0 * 0.4, 0.0, 1.0),
                3,
            ),
            "hostility_detected": hostility_detected,
        },
        "tempo": {
            "cooling_down": cooling_down,
            "pending_seconds": round(pending_seconds, 2),
            "pending_progress": round(_safe_float(subject.get("pending_progress", 0.0)), 3),
            "reply_pause_window": round(_safe_float(energy.get("night_reply_count", 0.0)), 2),
        },
        "counterparty_intent": {
            "engagement_state": engagement_state,
            "readiness_score": round(readiness_score, 3),
            "intent_label": str(meta.get("last_user_intent", "") or behavior.get("category", "neutral")),
            "hostility_detected": hostility_detected,
        },
        "group_climate": {
            "scene_heat": scene_heat,
            "scene_heat_score": round(heat_score, 3),
            "active_user_count": len(active_users),
            "topic_focus": topic_focus,
            "session_phase": str(scene.get("session_phase", "") or ""),
            "vexation": round(_safe_float(scene.get("vexation", 0.0)), 3),
            "weariness": round(_safe_float(scene.get("weariness", 0.0)), 3),
            "hot_count": _safe_int(participant_summary.get("hot_count", 0)),
            "warm_count": _safe_int(participant_summary.get("warm_count", 0)),
        },
        "resource_ledger": energy,
        "social_field": {
            "social_field_score": round(_safe_float(energy.get("social_field_score", 0.0)), 2),
            "rapport_score": relation.get("rapport_score", 0.0),
            "trust_score": relation.get("trust_score", 0.0),
            "interaction_count": relation.get("interaction_count", 0),
        },
        "pending_response": {
            "pending_active": pending_active,
            "pending_seconds": round(pending_seconds, 2),
            "pending_progress": round(_safe_float(subject.get("pending_progress", 0.0)), 3),
            "should_reengage": should_reengage,
            "flow_phase": str(subject.get("phase", "") or ""),
        },
        "runtime_calibration": {
            "threshold_profile": get_heartfc_thresholds().to_dict(),
            "pipeline_ready": bool(runtime.get("pipeline_ready", False)),
            "round_index": _safe_int(runtime.get("enhanced_round", 0)),
        },
        "trauma_load": {
            "trauma_load": round(trauma_load, 3),
            "pressure_load": round(pressure_load, 3),
            "chaos_load": round(chaos_load, 3),
            "flashback_probability": round(flashback_probability, 3),
            "cognitive_drag": round(cognitive_drag, 3),
        },
        "surface_mask": {
            "mask_load": round(relation.get("mask_load", 0.0), 3),
            "chaos_load": round(chaos_load, 3),
            "reply_quality_cap": round(_clamp(1.0 - relation.get("mask_load", 0.0) / 12.0, 0.25, 1.0), 3),
        },
        "relationship_profile": relation,
        "emergence_core": emergence,
        "flow_runtime": {
            "phase": str(subject.get("phase", "standby") or "standby"),
            "phase_label": str(subject.get("phase_label", "待命") or "待命"),
            "watch_state": str(subject.get("watch_level", "peek") or "peek"),
            "night_phase": str(subject.get("night_phase", "") or runtime.get("cached_night_phase", "")),
            "sleeping": bool(subject.get("is_sleeping", False)),
            "reply_allowed": bool(subject.get("can_reply", True)),
            "pending_active": pending_active,
            "last_target_user_id": relation.get("user_id", ""),
            "topic_focus": topic_focus,
            "last_reactive_plan": runtime.get("last_reactive_plan") or {},
        },
    }


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
                "target_user_id": str(getattr(chat, "_last_user_id", "") or ""),
            }
        )
    return {
        "active_count": len(chats),
        "channels": chats,
    }


async def export_heartfc_state(channel_id: str) -> Optional[Dict[str, Any]]:
    key, chat = _find_active_chat(channel_id)
    if chat is None:
        return None
    resolved_id = str(key)
    domains = _extract_domains(chat, resolved_id)
    return {
        "channel_id": resolved_id,
        "updated_at": time.time(),
        "domains": domains,
    }

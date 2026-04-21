import time
from typing import Any, Dict, Optional, Tuple

from src.chat.heart_flow.heartfc_state_exporter import export_heartfc_state, list_heartfc_chats


def _find_active_chat(channel_id: str) -> Tuple[Optional[Any], Optional[Any]]:
    from src.chat.heart_flow.heartflow import heartflow

    if channel_id in heartflow.heartflow_chat_list:
        return channel_id, heartflow.heartflow_chat_list.get(channel_id)
    for key, chat in heartflow.heartflow_chat_list.items():
        if str(key) == str(channel_id):
            return key, chat
    return None, None


def _build_dashboard_snapshot(chat: Any) -> Dict[str, Any]:
    try:
        builder = getattr(chat, "_build_state_dashboard", None)
        if not callable(builder):
            return {"available": False, "reason": "dashboard_builder_missing"}
        snapshot = builder(force=True)
        if snapshot is None:
            return {"available": False, "reason": "dashboard_not_ready"}
        return {"available": True, "snapshot": snapshot}
    except Exception as exc:
        return {"available": False, "reason": f"dashboard_error: {exc}"}


def _build_prediction_snapshot(state: Dict[str, Any], dashboard_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    dashboard = dashboard_snapshot.get("snapshot", {}) if dashboard_snapshot.get("available") else {}
    verdict = dashboard.get("verdict", {}) if isinstance(dashboard, dict) else {}
    social = dashboard.get("social", {}) if isinstance(dashboard, dict) else {}
    attention = dashboard.get("attention", {}) if isinstance(dashboard, dict) else {}
    vitality = dashboard.get("vitality", {}) if isinstance(dashboard, dict) else {}
    safety = dashboard.get("safety", {}) if isinstance(dashboard, dict) else {}

    urgency_score = {
        "立即回复": 0.95,
        "尽快回复": 0.78,
        "可稍后回": 0.52,
        "跳过": 0.18,
        "不回复": 0.05,
    }.get(str(verdict.get("urgency", "") or ""), 0.35)

    emotion = state.get("emotion", {}).get("snapshot", {}) if isinstance(state.get("emotion"), dict) else {}
    energy = state.get("energy", {}).get("snapshot", {}) if isinstance(state.get("energy"), dict) else {}
    world_snapshot = state.get("world_snapshot", {}).get("snapshot", {}) if isinstance(state.get("world_snapshot"), dict) else {}
    scene = world_snapshot.get("scene", {}) if isinstance(world_snapshot, dict) else {}

    proactive_willingness = float(emotion.get("proactive_willingness", 0.0) or 0.0)
    silence_duration = float(emotion.get("silence_duration", 0.0) or 0.0)
    combined_ratio = float(energy.get("combined_ratio", 0.5) or 0.5)
    social_willingness = float(social.get("willingness", 0.5) or 0.5)
    process_ratio_raw = attention.get("process_ratio", "50%")
    if isinstance(process_ratio_raw, str) and process_ratio_raw.endswith("%"):
        try:
            process_ratio = float(process_ratio_raw.rstrip("%")) / 100.0
        except Exception:
            process_ratio = 0.5
    else:
        process_ratio = float(process_ratio_raw or 0.5)
    blocked = bool(safety.get("blocked", False))

    probability = (
        urgency_score * 0.35
        + proactive_willingness * 0.25
        + combined_ratio * 0.15
        + social_willingness * 0.15
        + min(1.0, silence_duration / 300.0) * 0.10
    )
    probability *= max(0.25, min(1.0, process_ratio + 0.2))
    if blocked:
        probability *= 0.25
    probability = max(0.0, min(1.0, probability))

    if probability >= 0.80:
        eta_seconds = 30
    elif probability >= 0.65:
        eta_seconds = 120
    elif probability >= 0.45:
        eta_seconds = 300
    else:
        eta_seconds = 900

    current_topics = list(scene.get("current_topics", []) or []) if isinstance(scene, dict) else []
    runtime = state.get("runtime", {}) if isinstance(state.get("runtime"), dict) else {}
    last_plan = runtime.get("last_reactive_plan") if isinstance(runtime, dict) else None
    content_direction = ""
    if isinstance(last_plan, dict):
        content_direction = str(last_plan.get("content_plan", "") or "").strip()
    if not content_direction and current_topics:
        content_direction = f"围绕「{current_topics[0]}」延续发言"
    if not content_direction:
        content_direction = "根据当前群聊状态自然接话"

    vitality_percent = int(vitality.get("percent", 50) or 50)
    reply_length = "中等"
    if vitality_percent < 25:
        reply_length = "简短"
    elif vitality_percent > 75 and probability > 0.7:
        reply_length = "中长"

    return {
        "available": True,
        "prediction": {
            "speak_probability": round(probability, 3),
            "eta_seconds": eta_seconds,
            "eta_label": "立即" if eta_seconds <= 30 else f"约 {round(eta_seconds / 60)} 分钟后",
            "tone": str(verdict.get("tone", "") or "正常回应"),
            "content_direction": content_direction,
            "reply_length": reply_length,
            "generated_at": time.time(),
        },
    }


async def build_monitor_overview() -> Dict[str, Any]:
    overview = list_heartfc_chats()
    return {
        "generated_at": time.time(),
        "active_count": overview.get("active_count", 0),
        "channels": overview.get("channels", []),
    }


async def build_channel_monitor_state(channel_id: str) -> Optional[Dict[str, Any]]:
    state = await export_heartfc_state(channel_id)
    if state is None:
        return None
    _, chat = _find_active_chat(channel_id)
    dashboard = _build_dashboard_snapshot(chat) if chat is not None else {"available": False, "reason": "chat_not_found"}
    prediction = _build_prediction_snapshot(state, dashboard)
    return {
        "generated_at": time.time(),
        "channel_id": str(channel_id),
        "dashboard": dashboard,
        "prediction": prediction,
        "state": state,
    }

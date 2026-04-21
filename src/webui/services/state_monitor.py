import time
from typing import Any, Dict, Optional, Tuple

from src.chat.heart_flow.heartfc_state_exporter import export_heartfc_state, list_heartfc_chats
from src.chat.heart_flow.speak_prediction_engine import get_speak_prediction_engine


def _find_active_chat(channel_id: str) -> Tuple[Optional[Any], Optional[Any]]:
    from src.chat.heart_flow.heartflow import heartflow

    if channel_id in heartflow.heartflow_chat_list:
        return channel_id, heartflow.heartflow_chat_list.get(channel_id)
    for key, chat in heartflow.heartflow_chat_list.items():
        if str(key) == str(channel_id):
            return key, chat
    return None, None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _signal_color(severity: str) -> str:
    return {
        "info": "#2a9d8f",
        "low": "#e9c46a",
        "medium": "#f4a261",
        "high": "#e76f51",
        "critical": "#c1121f",
    }.get(severity, "#457b9d")


def _signal_card(
    *,
    key: str,
    label: str,
    family: str,
    severity: str,
    value: float,
    display_value: str,
    trend: str,
    source_domain: str,
    icon: str,
) -> Dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "family": family,
        "severity": severity,
        "value": round(value, 3),
        "display_value": display_value,
        "trend": trend,
        "source_domain": source_domain,
        "icon": icon,
        "color": _signal_color(severity),
    }


def _build_dashboard_snapshot(chat: Any) -> Dict[str, Any]:
    try:
        builder = getattr(chat, "_build_state_dashboard", None)
        if not callable(builder):
            return {"available": False, "reason": "dashboard_builder_missing"}
        display = builder(force=True)
        if display is None:
            return {"available": False, "reason": "dashboard_not_ready"}
        change_events = []
        try:
            from src.core.state_dashboard import get_state_dashboard

            engine = get_state_dashboard(str(getattr(chat, "stream_id", "") or ""))
            cached = getattr(engine, "_snapshot_cache", None)
            if cached is not None:
                change_events = list(getattr(cached, "change_events", []) or [])
        except Exception:
            change_events = []
        return {"available": True, "snapshot": display, "change_events": change_events}
    except Exception as exc:
        return {"available": False, "reason": f"dashboard_error: {exc}"}


def _build_active_signals(domains: Dict[str, Any]) -> list[Dict[str, Any]]:
    emergence = domains.get("emergence_core", {})
    relationship = domains.get("relationship_profile", {})
    trauma = domains.get("trauma_load", {})
    surface_mask = domains.get("surface_mask", {})
    pending = domains.get("pending_response", {})
    tempo = domains.get("tempo", {})

    signals: list[Dict[str, Any]] = []

    boredom = _safe_float(emergence.get("boredom_load", 0.0))
    if boredom >= 0.35:
        signals.append(
            _signal_card(
                key="boredom_load",
                label="无聊负荷",
                family="emotion",
                severity="medium" if boredom < 0.7 else "high",
                value=boredom,
                display_value=f"{boredom:.2f}",
                trend="rising",
                source_domain="emergence_core",
                icon="🥱",
            )
        )

    loneliness = _safe_float(emergence.get("loneliness_load", 0.0))
    if loneliness >= 0.35:
        signals.append(
            _signal_card(
                key="loneliness_load",
                label="孤独负荷",
                family="emotion",
                severity="medium" if loneliness < 0.7 else "high",
                value=loneliness,
                display_value=f"{loneliness:.2f}",
                trend="rising",
                source_domain="emergence_core",
                icon="🫥",
            )
        )

    fatigue = _safe_float(emergence.get("environment_fatigue_load", 0.0))
    if fatigue >= 0.35:
        signals.append(
            _signal_card(
                key="environment_fatigue_load",
                label="环境疲劳",
                family="emotion",
                severity="medium" if fatigue < 0.75 else "high",
                value=fatigue,
                display_value=f"{fatigue:.2f}",
                trend="rising",
                source_domain="emergence_core",
                icon="🌫️",
            )
        )

    withdrawal = _safe_float(emergence.get("withdrawal_drive", 0.0))
    if withdrawal >= 0.25:
        signals.append(
            _signal_card(
                key="withdrawal_drive",
                label="撤离倾向",
                family="emotion",
                severity="medium" if withdrawal < 0.55 else "high",
                value=withdrawal,
                display_value=f"{withdrawal:.2f}",
                trend="rising",
                source_domain="emergence_core",
                icon="🚪",
            )
        )

    initiative = _safe_float(emergence.get("initiative_drive", 0.0))
    if initiative >= 0.70 or initiative <= 0.25:
        signals.append(
            _signal_card(
                key="initiative_drive",
                label="主动意愿",
                family="drive",
                severity="high" if initiative >= 0.70 else "low",
                value=initiative,
                display_value=f"{initiative:.2f}",
                trend="rising" if initiative >= 0.70 else "falling",
                source_domain="emergence_core",
                icon="🗣️" if initiative >= 0.70 else "🤐",
            )
        )

    irritation = _safe_float(relationship.get("irritation_load", 0.0))
    if irritation >= 15:
        signals.append(
            _signal_card(
                key="irritation_load",
                label="烦躁负荷",
                family="relationship",
                severity="medium" if irritation < 45 else "high",
                value=irritation,
                display_value=f"{irritation:.1f}",
                trend="rising",
                source_domain="relationship_profile",
                icon="😒",
            )
        )

    pressure = _safe_float(relationship.get("pressure_load", 0.0))
    if pressure >= 15:
        signals.append(
            _signal_card(
                key="pressure_load",
                label="心理压力",
                family="relationship",
                severity="medium" if pressure < 45 else "high",
                value=pressure,
                display_value=f"{pressure:.1f}",
                trend="rising",
                source_domain="relationship_profile",
                icon="🧠",
            )
        )

    trauma_load = _safe_float(trauma.get("trauma_load", 0.0))
    if trauma_load >= 0.8:
        signals.append(
            _signal_card(
                key="trauma_load",
                label="创伤负荷",
                family="trauma",
                severity="high" if trauma_load < 2.5 else "critical",
                value=trauma_load,
                display_value=f"{trauma_load:.2f}",
                trend="persistent",
                source_domain="trauma_load",
                icon="🩹",
            )
        )

    chaos_load = _safe_float(trauma.get("chaos_load", 0.0))
    if chaos_load >= 1.0:
        signals.append(
            _signal_card(
                key="chaos_load",
                label="内心混乱",
                family="trauma",
                severity="medium" if chaos_load < 4.5 else "critical",
                value=chaos_load,
                display_value=f"{chaos_load:.2f}",
                trend="rising",
                source_domain="trauma_load",
                icon="🌀",
            )
        )

    flashback = _safe_float(trauma.get("flashback_probability", 0.0))
    if flashback >= 0.15:
        signals.append(
            _signal_card(
                key="flashback_probability",
                label="闪回风险",
                family="trauma",
                severity="medium" if flashback < 0.45 else "high",
                value=flashback,
                display_value=f"{flashback:.2f}",
                trend="volatile",
                source_domain="trauma_load",
                icon="⚡",
            )
        )

    cognitive_drag = _safe_float(trauma.get("cognitive_drag", 0.0))
    if cognitive_drag >= 0.15:
        signals.append(
            _signal_card(
                key="cognitive_drag",
                label="认知拖拽",
                family="trauma",
                severity="medium" if cognitive_drag < 0.45 else "high",
                value=cognitive_drag,
                display_value=f"{cognitive_drag:.2f}",
                trend="persistent",
                source_domain="trauma_load",
                icon="🧩",
            )
        )

    mask_load = _safe_float(surface_mask.get("mask_load", 0.0))
    if mask_load >= 1.0:
        signals.append(
            _signal_card(
                key="mask_load",
                label="伪装负荷",
                family="mask",
                severity="medium" if mask_load < 4 else "high",
                value=mask_load,
                display_value=f"{mask_load:.2f}",
                trend="persistent",
                source_domain="surface_mask",
                icon="🎭",
            )
        )

    compliance = _safe_float(relationship.get("compliance_index", 0.0))
    if compliance >= 5:
        signals.append(
            _signal_card(
                key="compliance_index",
                label="顺从指数",
                family="conditioning",
                severity="low" if compliance < 25 else "medium",
                value=compliance,
                display_value=f"{compliance:.1f}",
                trend="rising",
                source_domain="relationship_profile",
                icon="🪢",
            )
        )

    conditioning = _safe_float(relationship.get("conditioning_progress", 0.0))
    if conditioning >= 5:
        signals.append(
            _signal_card(
                key="conditioning_progress",
                label="调教进度",
                family="conditioning",
                severity="low" if conditioning < 25 else "medium",
                value=conditioning,
                display_value=f"{conditioning:.1f}%",
                trend="rising",
                source_domain="relationship_profile",
                icon="📈",
            )
        )

    resistance = _safe_float(relationship.get("resistance_index", 100.0))
    if abs(resistance - 100.0) >= 10:
        signals.append(
            _signal_card(
                key="resistance_index",
                label="抵抗指数",
                family="conditioning",
                severity="medium" if resistance >= 90 else "low",
                value=resistance,
                display_value=f"{resistance:.1f}",
                trend="falling" if resistance < 100 else "rising",
                source_domain="relationship_profile",
                icon="🧱",
            )
        )

    authority = _safe_float(relationship.get("authority_imprint", 0.0))
    if authority >= 5:
        signals.append(
            _signal_card(
                key="authority_imprint",
                label="权威印记",
                family="conditioning",
                severity="low" if authority < 25 else "medium",
                value=authority,
                display_value=f"{authority:.1f}",
                trend="rising",
                source_domain="relationship_profile",
                icon="👑",
            )
        )

    intimacy = _safe_float(relationship.get("intimacy_depth", 0.0))
    if intimacy >= 5:
        signals.append(
            _signal_card(
                key="intimacy_depth",
                label="亲密深度",
                family="relationship",
                severity="low" if intimacy < 25 else "medium",
                value=intimacy,
                display_value=f"{intimacy:.1f}",
                trend="rising",
                source_domain="relationship_profile",
                icon="🤝",
            )
        )

    overstep = _safe_float(relationship.get("boundary_overstep_index", 0.0))
    if overstep >= 5:
        signals.append(
            _signal_card(
                key="boundary_overstep_index",
                label="越界指数",
                family="relationship",
                severity="medium" if overstep < 25 else "high",
                value=overstep,
                display_value=f"{overstep:.1f}",
                trend="rising",
                source_domain="relationship_profile",
                icon="🚧",
            )
        )

    if bool(tempo.get("cooling_down", False)):
        signals.append(
            _signal_card(
                key="cooling_down",
                label="冷却窗口",
                family="runtime",
                severity="info",
                value=1.0,
                display_value="生效中",
                trend="holding",
                source_domain="tempo",
                icon="⏳",
            )
        )

    pending_seconds = _safe_float(pending.get("pending_seconds", 0.0))
    if pending_seconds > 0:
        signals.append(
            _signal_card(
                key="pending_seconds",
                label="等待时长",
                family="runtime",
                severity="info" if pending_seconds < 120 else "medium",
                value=pending_seconds,
                display_value=f"{round(pending_seconds)}秒",
                trend="rising",
                source_domain="pending_response",
                icon="⌛",
            )
        )

    if bool(pending.get("should_reengage", False)):
        signals.append(
            _signal_card(
                key="should_reengage",
                label="重新接入倾向",
                family="runtime",
                severity="info",
                value=1.0,
                display_value="已激活",
                trend="ready",
                source_domain="pending_response",
                icon="🔁",
            )
        )

    return signals


def _build_timeline(
    domains: Dict[str, Any],
    dashboard_snapshot: Dict[str, Any],
    active_signals: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    change_events = dashboard_snapshot.get("change_events", []) if isinstance(dashboard_snapshot, dict) else []
    if change_events:
        timeline = []
        for event in change_events[-10:]:
            timeline.append(
                {
                    "at": event.get("timestamp", time.time()),
                    "label": str(event.get("type", "state_change") or "state_change"),
                    "detail": str(event.get("detail", "") or event.get("summary", "") or "状态发生变化"),
                    "family": "dashboard",
                }
            )
        return timeline

    generated = []
    for signal in active_signals[:8]:
        generated.append(
            {
                "at": time.time(),
                "label": signal["label"],
                "detail": f"{signal['label']} 当前处于 {signal['display_value']}",
                "family": signal["family"],
            }
        )
    if not generated:
        scene_heat = domains.get("group_climate", {}).get("scene_heat", "normal")
        generated.append(
            {
                "at": time.time(),
                "label": "scene_heat",
                "detail": f"当前群聊热度为 {scene_heat}",
                "family": "scene",
            }
        )
    return generated


def _build_presentation(
    *,
    channel_id: str,
    domains: Dict[str, Any],
    dashboard_snapshot: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    dashboard = dashboard_snapshot.get("snapshot", {}) if dashboard_snapshot.get("available") else {}
    vitality = dashboard.get("energy_reserve", {}) if isinstance(dashboard, dict) else {}
    mood = dashboard.get("inner_mood", {}) if isinstance(dashboard, dict) else {}
    attention = dashboard.get("attention", {}) if isinstance(dashboard, dict) else {}
    social = dashboard.get("social_attitude", {}) if isinstance(dashboard, dict) else {}
    safety = dashboard.get("safety_shield", {}) if isinstance(dashboard, dict) else {}

    flow_runtime = domains.get("flow_runtime", {})
    group_climate = domains.get("group_climate", {})
    relationship = domains.get("relationship_profile", {})

    active_signals = _build_active_signals(domains)

    resident_overview = {
        "energy_reserve": {
            "label": "精力储备",
            "icon": str(vitality.get("icon", "🔋") or "🔋"),
            "state": str(vitality.get("level", domains.get("resource_ledger", {}).get("energy_phase", "未知")) or "未知"),
            "value": _safe_float(domains.get("resource_ledger", {}).get("energy_reserve_ratio", 0.0)),
            "display_value": f"{int(_safe_float(vitality.get('percent', 0), 0.0))}%",
            "color": str(vitality.get("color", "#22c55e") or "#22c55e"),
        },
        "inner_mood": {
            "label": "内在心情",
            "icon": str(mood.get("icon", "😊") or "😊"),
            "state": str(mood.get("mood_state", "平静") or "平静"),
            "value": _safe_float(mood.get("mood", 0.5)),
            "display_value": str(mood.get("mood_state", "平静") or "平静"),
            "color": "#e9c46a",
        },
        "attention_state": {
            "label": "注意状态",
            "icon": str(attention.get("icon", "📡") or "📡"),
            "state": str(attention.get("attention_state", "扫描模式") or "扫描模式"),
            "value": _safe_float(domains.get("resource_ledger", {}).get("activity_index", 50.0)) / 100.0,
            "display_value": str(attention.get("attention_state", "扫描模式") or "扫描模式"),
            "color": "#457b9d",
        },
        "social_attitude": {
            "label": "社交姿态",
            "icon": str(social.get("icon", "👀") or "👀"),
            "state": str(social.get("posture", "中立观察") or "中立观察"),
            "value": _safe_float(social.get("willingness", 0.5)),
            "display_value": str(social.get("posture", "中立观察") or "中立观察"),
            "color": "#2a9d8f",
        },
        "safety_shield": {
            "label": "安全护盾",
            "icon": str(safety.get("icon", "🛡️") or "🛡️"),
            "state": str(safety.get("level", "安全") or "安全"),
            "value": _safe_float(safety.get("score", 0.0)),
            "display_value": str(safety.get("level", "安全") or "安全"),
            "color": "#264653" if not safety.get("blocked") else "#c1121f",
        },
        "flow_phase": {
            "label": "流转阶段",
            "icon": "🫀",
            "state": str(flow_runtime.get("phase_label", flow_runtime.get("phase", "待命")) or "待命"),
            "value": 1.0 if flow_runtime.get("reply_allowed", True) else 0.0,
            "display_value": str(flow_runtime.get("phase_label", flow_runtime.get("phase", "待命")) or "待命"),
            "color": "#577590",
        },
        "scene_heat": {
            "label": "场景热度",
            "icon": "🔥" if group_climate.get("scene_heat") == "heated" else "🌡️",
            "state": str(group_climate.get("scene_heat", "normal") or "normal"),
            "value": _safe_float(group_climate.get("scene_heat_score", 0.0)),
            "display_value": str(group_climate.get("scene_heat", "normal") or "normal"),
            "color": "#f4a261",
        },
        "prediction_readiness": {
            "label": "发言预测",
            "icon": "🔮",
            "state": "高" if _safe_float(prediction.get("speak_probability", 0.0)) >= 0.7 else "中" if _safe_float(prediction.get("speak_probability", 0.0)) >= 0.45 else "低",
            "value": _safe_float(prediction.get("speak_probability", 0.0)),
            "display_value": prediction.get("eta_label", "未知"),
            "color": "#8d99ae",
        },
    }

    scene_context = {
        "channel_id": channel_id,
        "scene_heat": str(group_climate.get("scene_heat", "normal") or "normal"),
        "scene_heat_score": round(_safe_float(group_climate.get("scene_heat_score", 0.0)), 3),
        "topic_focus": list(group_climate.get("topic_focus", []) or []),
        "active_user_count": int(_safe_float(group_climate.get("active_user_count", 0), 0)),
        "session_phase": str(group_climate.get("session_phase", "") or ""),
        "hot_count": int(_safe_float(group_climate.get("hot_count", 0), 0)),
        "warm_count": int(_safe_float(group_climate.get("warm_count", 0), 0)),
    }

    participant_impacts = []
    if relationship.get("user_id"):
        active_labels = [
            signal["label"]
            for signal in active_signals
            if signal["source_domain"] in {"relationship_profile", "trauma_load", "surface_mask"}
        ]
        impact_rank = _safe_float(relationship.get("rapport_score", 0.0)) * 0.2 + _safe_float(
            relationship.get("trust_score", 0.0)
        ) * 0.15 - _safe_float(relationship.get("irritation_load", 0.0)) * 0.1
        participant_impacts.append(
            {
                "user_id": relationship.get("user_id", ""),
                "display_name": relationship.get("display_name", "当前目标"),
                "relationship_label": relationship.get("relationship_label", "陌生人"),
                "rapport_score": relationship.get("rapport_score", 0.0),
                "trust_score": relationship.get("trust_score", 0.0),
                "irritation_load": relationship.get("irritation_load", 0.0),
                "trauma_load": relationship.get("trauma_load", 0.0),
                "pressure_load": relationship.get("pressure_load", 0.0),
                "active_signals": active_labels,
                "impact_rank": round(impact_rank, 3),
            }
        )

    timeline = _build_timeline(domains, dashboard_snapshot, active_signals)

    return {
        "resident_overview": resident_overview,
        "active_signals": active_signals,
        "scene_context": scene_context,
        "participant_impacts": participant_impacts,
        "timeline": timeline,
    }


async def build_monitor_overview() -> Dict[str, Any]:
    overview = list_heartfc_chats()
    return {
        "updated_at": time.time(),
        "active_count": overview.get("active_count", 0),
        "channels": overview.get("channels", []),
    }


async def build_channel_monitor_state(channel_id: str) -> Optional[Dict[str, Any]]:
    state = await export_heartfc_state(channel_id)
    if state is None:
        return None
    _, chat = _find_active_chat(channel_id)
    dashboard = _build_dashboard_snapshot(chat) if chat is not None else {"available": False, "reason": "chat_not_found"}
    prediction = get_speak_prediction_engine().predict(
        channel_id=str(channel_id),
        domains=state.get("domains", {}),
        dashboard_snapshot=dashboard,
    )
    presentation = _build_presentation(
        channel_id=str(channel_id),
        domains=state.get("domains", {}),
        dashboard_snapshot=dashboard,
        prediction=prediction,
    )
    return {
        "channel_id": str(channel_id),
        "updated_at": time.time(),
        "domains": state.get("domains", {}),
        "presentation": presentation,
        "prediction": prediction,
    }

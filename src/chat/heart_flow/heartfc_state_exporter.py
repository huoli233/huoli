from collections import deque
import enum
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    if isinstance(value, (list, tuple, set, deque)):
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


def _scene_heat_label(heat_score: float) -> str:
    if heat_score >= 0.78:
        return "heated"
    if heat_score >= 0.55:
        return "lively"
    if heat_score >= 0.28:
        return "normal"
    if heat_score >= 0.12:
        return "quiet"
    return "dead"


def _derive_scene_heat_from_snapshot(snapshot: Dict[str, Any], vexation: float, weariness: float) -> tuple[str, float]:
    msg_rate = _clamp(_safe_float(snapshot.get("messages_per_minute", 0.0)) / 8.0, 0.0, 1.0)
    active_users = _clamp(_safe_float(snapshot.get("unique_speakers_5min", 0.0)) / 6.0, 0.0, 1.0)
    interaction_quality = _clamp(_safe_float(snapshot.get("interaction_quality", 0.0)), 0.0, 1.0)
    social_density = _clamp(_safe_float(snapshot.get("social_density", 0.0)), 0.0, 1.0)
    complexity_level = _clamp(_safe_float(snapshot.get("complexity_level", 0.0)), 0.0, 1.0)
    score = (
        msg_rate * 0.28
        + active_users * 0.22
        + interaction_quality * 0.18
        + social_density * 0.12
        + complexity_level * 0.10
        + _clamp(vexation / 100.0, 0.0, 1.0) * 0.12
        - _clamp(weariness / 100.0, 0.0, 1.0) * 0.08
    )
    atmosphere = str(snapshot.get("atmosphere", "") or "").strip().lower()
    if atmosphere in {"heated_discussion", "argument", "celebration", "social_gaming"}:
        score += 0.08
    elif atmosphere in {"quiet", "mourning"}:
        score -= 0.06
    score = _clamp(score, 0.0, 1.0)
    return _scene_heat_label(score), score


def _normalize_night_phase(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return {
        "daytime": "awake",
        "active_twilight": "active_twilight",
        "drowsy": "drowsy",
        "deep_valley": "deep_sleep",
        "dawn_recovery": "dawn_recover",
        "dawn_recover": "dawn_recover",
        "burned_out": "burnthrough",
        "burnthrough": "burnthrough",
        "light_sleep": "light_sleep",
        "deep_sleep": "deep_sleep",
        "night_active": "night_active",
        "midnight_reflect": "midnight_reflect",
        "social_night": "social_night",
        "quiet_contemplate": "quiet_contemplate",
        "awake": "awake",
    }.get(raw, raw or "awake")


def _night_phase_label(value: Any) -> str:
    phase = _normalize_night_phase(value)
    return {
        "awake": "清醒",
        "active_twilight": "黄昏微倦",
        "drowsy": "犯困",
        "light_sleep": "浅睡",
        "deep_sleep": "深睡",
        "deep_valley": "深睡沉寂",
        "burnthrough": "熬穿",
        "dawn_recover": "清晨恢复",
        "night_active": "晚上阶段",
        "midnight_reflect": "凌晨阶段",
        "social_night": "夜间社交",
        "quiet_contemplate": "凌晨沉思",
    }.get(phase, "清醒")


def _time_band_label(value: Any) -> str:
    return {
        "afternoon": "下午",
        "early_evening": "傍晚",
        "evening": "晚上",
        "late_evening": "晚上",
        "late_night": "半夜",
        "midnight": "凌晨",
        "dawn": "清晨",
        "morning": "上午",
        "noon": "中午",
        "daytime": "白天",
    }.get(str(value or "").strip().lower(), str(value or "") or "白天")


def _resolve_time_band(hour: Optional[int] = None) -> Dict[str, Any]:
    import time as _time

    h = _time.localtime().tm_hour if hour is None else int(hour) % 24
    if 13 <= h < 17:
        key, desc = "afternoon", "午后低谷"
    elif 20 <= h < 22:
        key, desc = "evening", "夜间社交窗口"
    elif 22 <= h < 23:
        key, desc = "late_evening", "熬夜压力预热"
    elif 23 <= h:
        key, desc = "late_night", "睡眠窗口开始"
    elif 0 <= h < 4:
        key, desc = "midnight", "凌晨反思窗口"
    elif 4 <= h < 7:
        key, desc = "dawn", "清晨恢复窗口"
    elif 7 <= h < 11:
        key, desc = "morning", "上午清醒"
    elif 11 <= h < 13:
        key, desc = "noon", "中午平稳"
    else:
        key, desc = "daytime", "白天平稳"
    return {"key": key, "label": _time_band_label(key), "description": desc, "hour": h}


def _body_state_label(value: Any) -> str:
    return {
        "deep_half_asleep": "半梦半醒很深",
        "light_half_asleep": "半梦半醒",
        "irritated_woken": "被吵醒有点恼",
        "slightly_grumpy": "醒来有点烦",
        "stubborn_lazy": "赖床抗拒",
        "reluctant_wake": "不太想醒",
        "frequently_yawning": "频繁打哈欠",
        "occasionally_yawning": "偶尔打哈欠",
        "on_verge_collapse": "快撑不住",
        "heavily_strained": "明显疲惫",
        "noticeably_tired": "有点累",
        "normal": "正常",
    }.get(str(value or "").strip(), str(value or "") or "正常")


def _expression_style_label(value: Any) -> str:
    return {
        "normal": "正常",
        "drowsy": "困倦短句",
        "stubborn": "硬撑克制",
        "arousal": "短时亢奋",
        "burnthrough": "熬穿失衡",
    }.get(str(value or "").strip().lower(), str(value or "") or "正常")


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
        "last_decision_runtime": "_last_decision_runtime",
        "last_execution_runtime": "_last_execution_runtime",
        "last_timing_gate_runtime": "_last_timing_gate_runtime",
        "timing_gate_history": "_timing_gate_history",
        "cached_night_phase": "_cached_night_phase",
        "cached_night_summary": "_cached_night_summary",
        "cached_metabolism_constraints": "_cached_metabolism_constraints",
        "cached_emotion_state": "_cached_emotion_state",
        "cached_emotion_feedback_report": "_cached_emotion_feedback_report",
        "cached_awareness": "_cached_awareness",
        "cached_presence_verdict": "_cached_presence_verdict",
        "cached_attention_snapshot": "_cached_attention_snapshot",
        "cached_safety_assessment": "_cached_safety_assessment",
        "last_relation_snapshot": "_last_relation_snapshot",
        "participant_summary": "_cached_participant_summary",
        "last_user_id": "_last_user_id",
        "last_proactive_target_user_id": "_last_proactive_target_user_id",
        "last_turn_schedule": "_last_turn_schedule_stats",
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


def _extract_group_climate(
    channel_id: str,
    world_scene: Dict[str, Any],
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    participant_summary = runtime.get("participant_summary", {})
    if not isinstance(participant_summary, dict):
        participant_summary = {}
    topic_focus = list(world_scene.get("current_topics", []) or [])
    session_phase = str(world_scene.get("session_phase", "") or "")
    vexation = _safe_float(world_scene.get("vexation", 0.0))
    weariness = _safe_float(world_scene.get("weariness", 0.0))
    active_user_count = 0
    messages_per_minute = 0.0
    participant_diversity = 0.0
    interaction_quality = 0.0
    complexity_level = 0.0
    social_density = 0.0
    suitable_to_join = True
    join_unsuitable_reason = ""
    dominant_speaker = ""
    active_topic_count = 0
    thread_count = 0
    atmosphere = ""
    atmosphere_label = ""
    vitality = 100.0
    mood_category = "calm"

    try:
        from src.core.group_scene_state import get_group_scene

        scene_engine = get_group_scene(channel_id)
        scene_snapshot = getattr(scene_engine, "current_scene_snapshot", None) or scene_engine.snapshot()
        if scene_snapshot is not None and hasattr(scene_snapshot, "to_dict"):
            scene_dict = scene_snapshot.to_dict()
            active_user_count = _safe_int(scene_dict.get("unique_speakers_5min", 0))
            messages_per_minute = _safe_float(scene_dict.get("messages_per_minute", 0.0))
            participant_diversity = _safe_float(scene_dict.get("participant_diversity", 0.0))
            interaction_quality = _safe_float(scene_dict.get("interaction_quality", 0.0))
            complexity_level = _safe_float(scene_dict.get("complexity_level", 0.0))
            social_density = _safe_float(scene_dict.get("social_density", 0.0))
            suitable_to_join = bool(scene_dict.get("suitable_to_join", True))
            join_unsuitable_reason = str(scene_dict.get("join_unsuitable_reason", "") or "")
            dominant_speaker = str(scene_dict.get("dominant_speaker", "") or "")
            active_topic_count = _safe_int(scene_dict.get("active_topic_count", 0))
            thread_count = _safe_int(scene_dict.get("thread_count", 0))
            atmosphere = str(scene_dict.get("atmosphere", "") or "")
            atmosphere_label = str(scene_dict.get("atmosphere_label", "") or "")
            try:
                topic_focus = scene_engine.active_topics(limit=5) or topic_focus
            except Exception:
                pass
            scene_heat, heat_score = _derive_scene_heat_from_snapshot(scene_dict, vexation, weariness)
        else:
            raise RuntimeError("scene_snapshot_unavailable")
    except Exception as exc:
        logger.debug(f"群场景快照导出失败，回退世界快照: {exc}")
        active_users = world_scene.get("active_users", []) if isinstance(world_scene.get("active_users", []), list) else []
        active_user_count = len(active_users)
        scene_heat, heat_score = _derive_scene_heat(active_user_count, vexation, weariness)

    try:
        from src.chat.heart_flow.emotion_stream import get_channel_mood_tracker

        mood_snapshot = get_channel_mood_tracker().dump_snapshot().get(channel_id, {})
        if isinstance(mood_snapshot, dict):
            vexation = _safe_float(mood_snapshot.get("vexation", vexation), vexation)
            weariness = _safe_float(mood_snapshot.get("weariness", weariness), weariness)
            vitality = _safe_float(mood_snapshot.get("vitality", vitality), vitality)
            mood_category = str(mood_snapshot.get("category", mood_category) or mood_category)
            if messages_per_minute <= 0 and active_user_count <= 0:
                scene_heat, heat_score = _derive_scene_heat(active_user_count, vexation, weariness)
    except Exception as exc:
        logger.debug(f"频道氛围账本导出失败: {exc}")

    return {
        "scene_heat": scene_heat,
        "scene_heat_score": round(_clamp(_safe_float(heat_score), 0.0, 1.0), 3),
        "messages_per_minute": round(messages_per_minute, 3),
        "active_user_count": int(active_user_count),
        "participant_diversity": round(participant_diversity, 3),
        "interaction_quality": round(interaction_quality, 3),
        "complexity_level": round(complexity_level, 3),
        "social_density": round(social_density, 3),
        "suitable_to_join": suitable_to_join,
        "join_unsuitable_reason": join_unsuitable_reason,
        "dominant_speaker": dominant_speaker,
        "active_topic_count": int(active_topic_count),
        "topic_focus": topic_focus,
        "thread_count": int(thread_count),
        "session_phase": session_phase,
        "vexation": round(vexation, 3),
        "weariness": round(weariness, 3),
        "vitality": round(vitality, 3),
        "mood_category": mood_category,
        "atmosphere": atmosphere,
        "atmosphere_label": atmosphere_label,
        "hot_count": _safe_int(participant_summary.get("hot_count", 0)),
        "warm_count": _safe_int(participant_summary.get("warm_count", 0)),
    }


def _extract_attention_runtime(channel_id: str, runtime: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = runtime.get("cached_attention_snapshot", {})
    if not isinstance(snapshot, dict):
        snapshot = {}
    try:
        from src.core.subjective_attention_flow import get_attention_flow_controller

        controller = get_attention_flow_controller(channel_id)
        live_snapshot = controller.get_snapshot()
        if hasattr(live_snapshot, "to_dict"):
            snapshot = dict(live_snapshot.to_dict() or {}) or snapshot
    except Exception as exc:
        logger.debug(f"注意力运行态导出失败，回退缓存: {exc}")
    return {
        "state": str(snapshot.get("state", "") or snapshot.get("attention_mode", "") or ""),
        "state_label": str(snapshot.get("state_label", "") or ""),
        "visibility_threshold": round(_safe_float(snapshot.get("visibility_threshold", 0.0)), 3),
        "process_ratio": round(_safe_float(snapshot.get("process_ratio", 0.0)), 3),
        "peek_desire": round(_safe_float(snapshot.get("peek_desire", 0.0)), 3),
        "interrupt_tolerance": round(_safe_float(snapshot.get("interrupt_tolerance", 0.0)), 3),
        "silence_tolerance": round(_safe_float(snapshot.get("silence_tolerance", 0.0)), 3),
        "time_since_last_look": round(_safe_float(snapshot.get("since_last_look_sec", 0.0)), 2),
        "consecutive_peeks_without_action": _safe_int(snapshot.get("consecutive_empty_peeks", 0)),
        "look_budget_state": round(_safe_float(snapshot.get("look_budget_state", 0.0)), 3),
        "process_budget_state": round(_safe_float(snapshot.get("process_budget_state", 0.0)), 3),
        "openness": round(_safe_float(snapshot.get("openness", 0.0)), 3),
    }


def _extract_safety_runtime(runtime: Dict[str, Any]) -> Dict[str, Any]:
    safety = runtime.get("cached_safety_assessment", {})
    if not isinstance(safety, dict):
        safety = {}
    level = safety.get("level", safety.get("overall_level_label", safety.get("safety_level", "")))
    score = safety.get("score", safety.get("overall_score", 0.0))
    dominant = safety.get("dominant_threat", safety.get("dominant", safety.get("threat", "")))
    blocked = safety.get("block_reply") if "block_reply" in safety else safety.get("blocked", False)
    bar_penalty = safety.get("bar_delta", safety.get("bar_penalty", 0.0))
    evidence = safety.get("threat_evidence_summary", safety.get("action", dominant))
    return {
        "level": str(level or ""),
        "score": round(_safe_float(score), 4),
        "dominant_threat": str(dominant or ""),
        "blocked": bool(blocked),
        "bar_penalty": round(_safe_float(bar_penalty), 4),
        "threat_evidence_summary": str(evidence or ""),
    }


def _extract_circadian_rhythm(
    channel_id: str,
    runtime: Dict[str, Any],
    subject: Dict[str, Any],
) -> Dict[str, Any]:
    summary = runtime.get("cached_night_summary", {})
    if not isinstance(summary, dict):
        summary = {}

    phase = _normalize_night_phase(
        summary.get("phase")
        or subject.get("night_phase")
        or runtime.get("cached_night_phase")
        or "awake"
    )
    phase_label = str(summary.get("phase_label") or _night_phase_label(phase))
    can_reply = bool(summary.get("can_reply", True))
    remaining_replies = _safe_int(summary.get("remaining_replies", 0), 0)
    sleep_reply_used = _safe_int(summary.get("sleep_reply_used", 0), 0)
    sleep_reply_cap = _safe_int(summary.get("night_reply_cap", 0), 0)
    peek_window = bool(summary.get("is_peek_window", summary.get("peek_window_open", False)))
    sleep_debt = _safe_float(summary.get("sleep_debt", 0.0))
    drowsiness = _safe_float(summary.get("drowsiness", summary.get("drowsiness_value", 0.0)))
    overnight_pressure = _safe_float(summary.get("overnight_pressure", 0.0))
    sleep_reserve = _safe_float(summary.get("sleep_reserve", 100.0), 100.0)
    dawn_progress = _safe_float(summary.get("dawn_recovery_progress", 0.0))
    response_suppression = _safe_float(summary.get("response_suppression", 0.0))
    body_state_tag = str(summary.get("body_state_tag", "") or "")
    mood_hint = str(summary.get("monologue_mood_hint", "") or "")
    expression_style = str(summary.get("expression_style", "normal") or "normal")
    time_band = str(summary.get("time_band", "") or "")
    time_band_label = str(summary.get("time_band_label", "") or "")
    time_band_description = str(summary.get("time_band_description", "") or "")
    current_hour = _safe_int(summary.get("current_hour", time.localtime().tm_hour), time.localtime().tm_hour)
    pressure_breakdown = summary.get("pressure_breakdown", {})
    if not isinstance(pressure_breakdown, dict):
        pressure_breakdown = {}
    mechanism_windows = summary.get("mechanism_windows", {})
    if not isinstance(mechanism_windows, dict):
        mechanism_windows = {}
    system_started_at = _safe_float(summary.get("system_started_at", 0.0), 0.0)
    last_evaluated_at = _safe_float(summary.get("last_evaluated_at", 0.0), 0.0)
    sync_label = str(summary.get("sync_label", "") or "")
    sync_source = str(summary.get("sync_source", "") or "")
    is_sleep_window = bool(summary.get("is_sleep_window", False))
    is_night_social_window = bool(summary.get("is_night_social_window", False))
    is_pressure_window = bool(summary.get("is_pressure_window", False))
    is_burnthrough = bool(summary.get("is_burnthrough", False)) or phase == "burnthrough"
    is_sleeping = bool(summary.get("is_sleeping", subject.get("is_sleeping", False))) or phase in {
        "light_sleep",
        "deep_sleep",
    }
    is_night = bool(summary.get("is_night", False)) or phase not in {"awake"}

    try:
        import src.core.night_cycle_system as night_module

        instances = getattr(night_module, "_night_system_instances", {}) or {}
        night_cycle = instances.get(channel_id) or instances.get(str(channel_id))
        if night_cycle is not None:
            behavior = night_cycle.night_behavior_summary()
            state = night_cycle.state_snapshot
            phase = _normalize_night_phase(behavior.get("phase", phase))
            phase_label = str(behavior.get("phase_label") or _night_phase_label(phase))
            time_band = str(behavior.get("time_band", time_band) or time_band)
            time_band_label = str(behavior.get("time_band_label", time_band_label) or time_band_label)
            time_band_description = str(
                behavior.get("time_band_description", time_band_description) or time_band_description
            )
            current_hour = _safe_int(behavior.get("current_hour", current_hour), current_hour)
            mechanism_windows = behavior.get("mechanism_windows", mechanism_windows)
            if not isinstance(mechanism_windows, dict):
                mechanism_windows = {}
            system_started_at = _safe_float(behavior.get("system_started_at", system_started_at), system_started_at)
            last_evaluated_at = _safe_float(behavior.get("last_evaluated_at", last_evaluated_at), last_evaluated_at)
            sync_label = str(behavior.get("sync_label", sync_label) or sync_label)
            sync_source = str(behavior.get("sync_source", sync_source) or sync_source)
            is_sleep_window = bool(behavior.get("is_sleep_window", is_sleep_window))
            is_night_social_window = bool(behavior.get("is_night_social_window", is_night_social_window))
            is_pressure_window = bool(behavior.get("is_pressure_window", is_pressure_window))
            can_reply = bool(behavior.get("can_reply", can_reply))
            remaining_replies = _safe_int(behavior.get("remaining_replies", remaining_replies))
            sleep_debt = _safe_float(behavior.get("sleep_debt", sleep_debt))
            drowsiness = _safe_float(behavior.get("drowsiness_value", drowsiness))
            overnight_pressure = _safe_float(behavior.get("overnight_pressure", overnight_pressure))
            sleep_reserve = _safe_float(behavior.get("sleep_reserve", sleep_reserve))
            response_suppression = _safe_float(behavior.get("response_suppression", response_suppression))
            body_state_tag = str(behavior.get("body_state_tag", body_state_tag) or "")
            mood_hint = str(behavior.get("monologue_mood_hint", mood_hint) or "")
            peek_window = bool(behavior.get("is_peek_window", peek_window))
            is_burnthrough = bool(behavior.get("burned_out", is_burnthrough)) or phase == "burnthrough"
            is_sleeping = bool(behavior.get("is_sleeping", is_sleeping)) or phase in {"light_sleep", "deep_sleep"}
            is_night = bool(behavior.get("is_night", phase not in {"awake"}))
            dawn_progress = _safe_float(getattr(state, "dawn_recovery_progress", dawn_progress))
            sleep_reply_used = _safe_int(getattr(state, "sleep_reply_used", sleep_reply_used))
            sleep_reply_cap = _safe_int(getattr(state, "night_reply_cap", sleep_reply_cap))
            behavior_breakdown = behavior.get("pressure_breakdown", {})
            if isinstance(behavior_breakdown, dict):
                pressure_breakdown = behavior_breakdown
            try:
                style = night_cycle.get_expression_deformation()
                expression_style = str(style.get("active_template", expression_style) or expression_style)
            except Exception:
                pass
    except Exception as exc:
        logger.debug(f"夜间状态导出失败: {exc}")

    if not time_band:
        resolved_band = _resolve_time_band(current_hour)
        time_band = str(resolved_band["key"])
        time_band_label = str(resolved_band["label"])
        time_band_description = str(resolved_band["description"])
        current_hour = _safe_int(resolved_band["hour"], current_hour)
    else:
        time_band_label = time_band_label or _time_band_label(time_band)
        time_band_description = time_band_description or time_band_label

    if sleep_reply_cap <= 0:
        sleep_reply_cap = max(remaining_replies, 0)
    if remaining_replies <= 0 and sleep_reply_cap > 0:
        remaining_replies = max(0, sleep_reply_cap - sleep_reply_used)

    return {
        "phase": phase,
        "phase_label": phase_label or _night_phase_label(phase),
        "time_band": time_band,
        "time_band_label": time_band_label,
        "time_band_description": time_band_description,
        "current_hour": current_hour,
        "is_night": is_night,
        "is_sleep_window": is_sleep_window,
        "is_night_social_window": is_night_social_window,
        "is_pressure_window": is_pressure_window,
        "mechanism_windows": mechanism_windows,
        "system_started_at": round(system_started_at, 3),
        "last_evaluated_at": round(last_evaluated_at, 3),
        "sync_label": sync_label or "运行时实时计算",
        "sync_source": sync_source or "heartfc_state_exporter",
        "is_sleeping": is_sleeping,
        "is_burnthrough": is_burnthrough,
        "can_reply": can_reply,
        "drowsiness_value": round(drowsiness, 3),
        "sleep_debt": round(sleep_debt, 3),
        "overnight_pressure": round(overnight_pressure, 3),
        "sleep_reserve": round(sleep_reserve, 3),
        "dawn_recovery_progress": round(dawn_progress, 3),
        "peek_window_open": peek_window,
        "sleep_reply_used": sleep_reply_used,
        "sleep_reply_cap": sleep_reply_cap,
        "remaining_sleep_replies": remaining_replies,
        "response_suppression": round(response_suppression, 3),
        "pressure_breakdown": {
            "total": round(_safe_float(pressure_breakdown.get("total", 0.0)), 4),
            "chat_minutes": round(_safe_float(pressure_breakdown.get("chat_minutes", 0.0)), 2),
            "peek_minutes": round(_safe_float(pressure_breakdown.get("peek_minutes", 0.0)), 2),
            "think_intensity": round(_safe_float(pressure_breakdown.get("think_intensity", 0.0)), 3),
            "interrupt_count": _safe_int(pressure_breakdown.get("interrupt_count", 0), 0),
            "chat_component": round(_safe_float(pressure_breakdown.get("chat_component", 0.0)), 4),
            "peek_component": round(_safe_float(pressure_breakdown.get("peek_component", 0.0)), 4),
            "think_component": round(_safe_float(pressure_breakdown.get("think_component", 0.0)), 4),
            "interrupt_component": round(_safe_float(pressure_breakdown.get("interrupt_component", 0.0)), 4),
        },
        "body_state_tag": body_state_tag,
        "body_state_label": _body_state_label(body_state_tag),
        "mood_hint": mood_hint,
        "expression_style": expression_style,
        "expression_style_label": _expression_style_label(expression_style),
    }


def _extract_memory_stack(channel_id: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ephemeral_short_term_count": 0,
        "long_term_total": 0,
        "long_term_by_category": {},
        "planner_short_term_count": 0,
        "planner_long_term_count": 0,
        "memoir_phase": "open",
        "memoir_exchange_tally": 0,
        "memoir_consecutive_timeouts": 0,
        "memoir_last_topic": "",
        "memoir_last_mood": "",
        "journal_count": 0,
        "governance": {},
        "health": {},
        "overload": {},
        "reactivation": {},
        "knowledge_entry_count": 0,
    }
    try:
        from src.core.memory_governance_engine import get_memory_governance_engine

        payload["governance"] = get_memory_governance_engine(channel_id).get_snapshot().to_dict()
    except Exception as exc:
        logger.debug(f"记忆治理导出失败: {exc}")
    try:
        from src.memory_system.memory_consolidator import acquire_ephemeral_bank

        bank = acquire_ephemeral_bank()
        buffers = getattr(bank, "_channel_buffers", {}) or {}
        payload["ephemeral_short_term_count"] = len(buffers.get(channel_id, []) or [])
    except Exception as exc:
        logger.debug(f"短期记忆导出失败: {exc}")
    try:
        from src.memory_system.memory_core import get_memory_core

        hub = get_memory_core()
        long_term_stats = hub.compile_statistics(channel_id)
        payload["long_term_total"] = _safe_int(long_term_stats.get("total", 0))
        payload["long_term_by_category"] = dict(long_term_stats.get("by_category", {}) or {})
        payload["health"] = hub.generate_health_assessment(channel_id)
        payload["overload"] = hub.retrieve_overload_status(channel_id)
    except Exception as exc:
        logger.debug(f"长期记忆导出失败: {exc}")
    try:
        from src.core.memory_reactivation_gate import get_reactivation_gate

        payload["reactivation"] = get_reactivation_gate(channel_id).recent_activation_summary()
    except Exception as exc:
        logger.debug(f"记忆激活导出失败: {exc}")
    try:
        from src.core.unified_planner import get_unified_planner

        planner_stats = get_unified_planner().get_context_stats(channel_id)
        payload["planner_short_term_count"] = _safe_int(planner_stats.get("short_term_count", 0))
        payload["planner_long_term_count"] = _safe_int(planner_stats.get("long_term_count", 0))
    except Exception as exc:
        logger.debug(f"规划器上下文导出失败: {exc}")
    try:
        from src.chat.proactive.session_tracker import get_memoir_cabinet

        memoir = get_memoir_cabinet().lookup_by_channel(channel_id)
        if memoir is not None:
            payload["memoir_phase"] = str(getattr(memoir, "phase", "open") or "open")
            payload["memoir_exchange_tally"] = _safe_int(getattr(memoir, "exchange_tally", 0))
            payload["memoir_consecutive_timeouts"] = _safe_int(getattr(memoir, "consecutive_timeouts", 0))
            payload["memoir_last_topic"] = str(getattr(memoir, "last_topic", "") or "")
            payload["memoir_last_mood"] = str(getattr(memoir, "last_mood", "") or "")
            try:
                payload["journal_count"] = len(memoir.latest_entries(999))
            except Exception:
                payload["journal_count"] = 0
    except Exception as exc:
        logger.debug(f"回忆录导出失败: {exc}")
    try:
        from src.memory_system.memory_health import get_memory_health_status

        payload["database_health"] = get_memory_health_status().to_dict()
    except Exception as exc:
        logger.debug(f"记忆库健康状态导出失败: {exc}")
    return payload


def _extract_autonomy_runtime(chat: Any, channel_id: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "active_intentions": [],
        "intention_alive_count": 0,
        "intention_drive": 0.0,
        "max_intention_urgency": 0.0,
        "reward_score": 0.0,
        "pending_proactive_events": 0,
        "self_recent_messages_count": 0,
        "self_recent_actions_count": 0,
        "self_recent_events_count": 0,
        "background_event": {},
    }
    try:
        from src.chat.proactive.intention_pool import get_intention_pool

        pool = get_intention_pool()
        payload["active_intentions"] = pool.build_planner_intent_payload(channel_id, limit=3)
        snapshot = pool.snapshot(channel_id)
        payload["intention_alive_count"] = _safe_int(snapshot.get("alive_count", 0))
        payload["max_intention_urgency"] = round(_safe_float(snapshot.get("max_urgency", 0.0)), 3)
        payload["intention_drive"] = round(_safe_float(pool.build_decision_signal(channel_id), 0.0), 3)
    except Exception as exc:
        logger.debug(f"意图池导出失败: {exc}")
    try:
        from src.chat.proactive.proactive_decider import get_proactive_decider

        decider = get_proactive_decider()
        payload["reward_score"] = round(_safe_float(decider.ledger.channel_reward(channel_id), 0.0), 3)
        pending = getattr(decider.ledger, "_pending", []) or []
        payload["pending_proactive_events"] = sum(
            1
            for evt in pending
            if not bool(getattr(evt, "resolved", False))
            and str(getattr(evt, "channel_id", "") or "") == str(channel_id)
        )
    except Exception as exc:
        logger.debug(f"主动账本导出失败: {exc}")
    try:
        from src.chat.proactive.proactive_integration_hub import get_proactive_integration_hub

        state = get_proactive_integration_hub().get_state_snapshot(channel_id)
        if isinstance(state, dict) and "error" not in state:
            self_awareness = state.get("self_awareness_data", {}) or {}
            payload["self_recent_messages_count"] = _safe_int(self_awareness.get("recent_messages_count", 0))
            payload["self_recent_actions_count"] = _safe_int(self_awareness.get("recent_actions_count", 0))
            payload["self_recent_events_count"] = _safe_int(self_awareness.get("recent_events_count", 0))
    except Exception as exc:
        logger.debug(f"主动整合快照导出失败: {exc}")
    try:
        background_event = getattr(chat, "_last_background_event", {}) if chat is not None else {}
        if isinstance(background_event, dict):
            payload["background_event"] = {
                "type": str(background_event.get("type", "") or ""),
                "priority": round(_safe_float(background_event.get("priority", 0.0)), 3),
                "budget_remaining": _safe_int(background_event.get("budget_remaining", 0)),
                "scheduled_at": _safe_float(background_event.get("scheduled_at", 0.0)),
            }
    except Exception as exc:
        logger.debug(f"后台事件导出失败: {exc}")
    return payload


def _extract_context_awareness(
    chat: Any,
    runtime: Dict[str, Any],
    subject: Dict[str, Any],
    relation: Dict[str, Any],
    memory_stack: Dict[str, Any],
    autonomy_runtime: Dict[str, Any],
) -> Dict[str, Any]:
    behavior = runtime.get("last_behavior_governor", {})
    if not isinstance(behavior, dict):
        behavior = {}
    rest = runtime.get("last_rest_governor", {})
    if not isinstance(rest, dict):
        rest = {}
    awareness = runtime.get("cached_awareness", {})
    if not isinstance(awareness, dict):
        awareness = _plain(awareness) if awareness else {}
        if not isinstance(awareness, dict):
            awareness = {}
    presence = runtime.get("cached_presence_verdict", {})
    if not isinstance(presence, dict):
        presence = {}
    behavior_reasons = [str(code) for code in list(behavior.get("reason_codes", []) or []) if str(code).strip()]
    rest_reasons = [str(code) for code in list(rest.get("reason_codes", []) or []) if str(code).strip()]
    direct_target = bool(
        "direct_relevance" in behavior_reasons
        or "direct_target_override" in rest_reasons
        or str(behavior.get("interrupt_level", "")) == "engage"
    )
    quote_anchor = bool(any("quote" in code or "anchor" in code for code in behavior_reasons))
    recent_human_activity = bool(
        "recent_human_activity" in behavior_reasons
        or "ambient_human_signal" in behavior_reasons
        or "weak_human_signal" in behavior_reasons
    )
    watch_state = str(subject.get("watch_level", "") or "")
    watch_label = str(subject.get("watch_level_label", "") or "")
    attention_rank = {
        "blackout": 0,
        "peek": 1,
        "skim_window": 2,
        "active_watch": 3,
        "engaged": 4,
    }.get(watch_state, 1)
    scene_suitable = str(behavior.get("reply_mode", "observe") or "observe") not in {"rest", "defer"}
    return {
        "direct_target": direct_target,
        "quote_anchor": quote_anchor,
        "recent_human_activity": recent_human_activity,
        "scene_suitable": scene_suitable,
        "watch_state": watch_state,
        "watch_state_label": watch_label or "瞥一眼",
        "attention_level": attention_rank,
        "perception_engagement_pull": round(_safe_float(awareness.get("engagement_pull", 0.0)), 3),
        "self_recent_messages_count": _safe_int(autonomy_runtime.get("self_recent_messages_count", 0)),
        "self_recent_actions_count": _safe_int(autonomy_runtime.get("self_recent_actions_count", 0)),
        "self_recent_events_count": _safe_int(autonomy_runtime.get("self_recent_events_count", 0)),
        "memoir_phase": str(memory_stack.get("memoir_phase", "open") or "open"),
        "memoir_consecutive_timeouts": _safe_int(memory_stack.get("memoir_consecutive_timeouts", 0)),
        "current_target_user_id": str(relation.get("user_id", "") or ""),
        "behavior_reason_codes": behavior_reasons,
        "rest_reason_codes": rest_reasons,
        "reply_mode": str(behavior.get("reply_mode", "observe") or "observe"),
        "interrupt_level": str(behavior.get("interrupt_level", "ignore") or "ignore"),
        "silence_policy": str(behavior.get("silence_policy", "silent") or "silent"),
        "presence_summary": presence,
    }


def _dominant_emotion_label(state: Any) -> str:
    try:
        emotion_name, intensity = state.get_dominant_emotion()
    except Exception:
        return "状态平稳"
    emotion_key = str(emotion_name or "neutral").strip().lower()
    intensity_value = abs(_safe_float(intensity, 0.0))
    if intensity_value < 0.15:
        return "状态平稳"
    return {
        "joy": "明显愉快",
        "anticipation": "期待偏高",
        "surprise": "注意被勾起",
        "sadness": "情绪低落",
        "disgust": "排斥增强",
        "anger": "烦躁升高",
        "fear": "警惕紧张",
        "trust": "信任放松",
        "pride": "自我肯定",
        "guilt": "内疚波动",
        "shame": "羞耻防御",
        "envy": "比较心起伏",
        "gratitude": "感激放松",
    }.get(emotion_key, "情绪波动")


def _relationship_entry_from_state(
    user_id: str,
    state: Any,
    *,
    target_user_id: str = "",
    in_current_scene: bool = False,
    recent_speaker: bool = False,
    recent_targeted_interaction: bool = False,
    source_scope: str = "recent_history",
) -> Dict[str, Any]:
    rapport_score = _safe_float(getattr(state, "affection", 0.0))
    trust_score = _safe_float(getattr(state, "trust_value", getattr(state, "trust_score", 0.0)))
    irritation_load = _safe_float(getattr(state, "annoyance", 0.0))
    trauma_load = _safe_float(getattr(state, "trauma_score", 0.0))
    pressure_load = _safe_float(getattr(state, "psychological_pressure", 0.0))
    chaos_load = _safe_float(getattr(state, "inner_chaos", 0.0))
    mask_load = _safe_float(getattr(state, "surface_mask", 0.0))
    interaction_count = _safe_int(getattr(state, "interaction_count", 0))
    last_interaction = _safe_float(getattr(state, "last_interaction", 0.0))
    last_seen_age = 999999.0
    if last_interaction > 0:
        last_seen_age = max(0.0, time.time() - last_interaction)
    recency_score = _clamp(1.0 - (last_seen_age / 3600.0), 0.0, 1.0)
    emotional_pressure = (
        abs(rapport_score) * 0.16
        + abs(trust_score) * 0.14
        + irritation_load * 0.20
        + pressure_load * 0.18
        + trauma_load * 4.0
        + chaos_load * 0.35
        + mask_load * 0.20
        + interaction_count * 0.08
        + recency_score * 8.0
    )
    if str(user_id) == str(target_user_id):
        emotional_pressure += 120.0
    if in_current_scene:
        emotional_pressure += 28.0
    if recent_speaker:
        emotional_pressure += 24.0
    if recent_targeted_interaction:
        emotional_pressure += 16.0
    return {
        "user_id": str(user_id or ""),
        "display_name": str(getattr(state, "nickname", "") or user_id or "用户"),
        "relationship_label": str(getattr(state, "relationship", "陌生人") or "陌生人"),
        "rapport_score": round(rapport_score, 2),
        "trust_score": round(trust_score, 2),
        "irritation_load": round(irritation_load, 2),
        "pressure_load": round(pressure_load, 2),
        "trauma_load": round(trauma_load, 2),
        "chaos_load": round(chaos_load, 2),
        "mask_load": round(mask_load, 2),
        "interaction_count": interaction_count,
        "current_mood_hint": _dominant_emotion_label(state),
        "last_interaction_age_sec": round(last_seen_age, 1),
        "is_current_target": bool(str(user_id) == str(target_user_id)),
        "in_current_scene": bool(in_current_scene),
        "recent_speaker": bool(recent_speaker),
        "recent_targeted_interaction": bool(recent_targeted_interaction),
        "source_scope": str(source_scope or "recent_history"),
        "is_blocked": bool(getattr(state, "is_blocked", False)),
        "is_protected": bool(getattr(state, "is_protected", False)),
        "impact_rank": round(emotional_pressure, 3),
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
        world_target.get("personal_impression", "")
        or getattr(state, "relationship", world_target.get("custom_label", ""))
        or world_target.get("mood", "")
    )
    legacy_relationship_label = str(
        world_target.get("legacy_relationship_label", "")
        or world_target.get("custom_label", "")
        or getattr(state, "relationship", "")
        or ""
    )
    personal_impression = str(world_target.get("personal_impression", "") or relationship_label or "")
    impression_labels = world_target.get("impression_labels", [])
    if not isinstance(impression_labels, list):
        impression_labels = []
    display_name = str(
        getattr(state, "nickname", "") or target_name or target_uid or "当前目标"
    )
    return {
        "user_id": target_uid,
        "display_name": display_name,
        "relationship_label": relationship_label,
        "personal_impression": personal_impression,
        "impression_labels": impression_labels[:8],
        "impression_source": str(world_target.get("impression_source", "") or ""),
        "impression_updated_at": _safe_float(world_target.get("impression_updated_at", 0.0)),
        "legacy_relationship_label": legacy_relationship_label,
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


def _extract_relationship_population(chat: Any, target_user_id: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "total_users": 0,
        "relationship_distribution": {},
        "participants": [],
    }
    if chat is None or not getattr(chat, "stream_id", ""):
        return payload
    try:
        candidate_sources: Dict[str, set[str]] = {}

        def add_candidate(user_id: str, scope: str) -> None:
            normalized = str(user_id or "").strip()
            if not normalized:
                return
            candidate_sources.setdefault(normalized, set()).add(str(scope or "recent_history"))

        if target_user_id:
            add_candidate(target_user_id, "mentioned_target")

        try:
            from src.core.group_scene_state import get_group_scene

            scene_engine = get_group_scene(str(getattr(chat, "stream_id", "") or ""))
            for user_id in scene_engine.recent_speakers(window_sec=300.0, limit=12):
                add_candidate(user_id, "current_scene")
            for user_id in scene_engine.alive_thread_participants(limit=12):
                add_candidate(user_id, "thread_participant")
            for user_id in scene_engine.recent_mentioned_users(window_sec=300.0, limit=12):
                add_candidate(user_id, "mentioned_target")
        except Exception as exc:
            logger.debug(f"关系候选用户场景过滤失败: {exc}")

        from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker

        tracker = get_emotion_tracker(str(getattr(chat, "stream_id", "") or ""))
        stats = tracker.get_statistics()
        all_states = tracker.get_all_user_states()
        payload["total_users"] = _safe_int(stats.get("total", 0))
        payload["relationship_distribution"] = dict(stats.get("relationship_distribution", {}) or {})
        participants: List[Dict[str, Any]] = []
        for user_id, state in list(all_states.items()):
            normalized_user_id = str(user_id or "").strip()
            if not normalized_user_id:
                continue
            last_interaction = _safe_float(getattr(state, "last_interaction", 0.0))
            last_interaction_age_sec = 999999.0
            if last_interaction > 0:
                last_interaction_age_sec = max(0.0, time.time() - last_interaction)
            if last_interaction_age_sec <= 3600.0:
                add_candidate(normalized_user_id, "recent_history")
            scopes = candidate_sources.get(normalized_user_id, set())
            if not scopes and normalized_user_id != str(target_user_id or "").strip():
                continue
            if "current_scene" in scopes:
                source_scope = "current_scene"
            elif "thread_participant" in scopes:
                source_scope = "thread_participant"
            elif "mentioned_target" in scopes:
                source_scope = "mentioned_target"
            else:
                source_scope = "recent_history"
            entry = _relationship_entry_from_state(
                normalized_user_id,
                state,
                target_user_id=target_user_id,
                in_current_scene=bool(scopes & {"current_scene", "thread_participant", "mentioned_target"}),
                recent_speaker=bool("current_scene" in scopes),
                recent_targeted_interaction=bool(
                    "mentioned_target" in scopes or normalized_user_id == str(target_user_id or "").strip()
                ),
                source_scope=source_scope,
            )
            if not entry.get("user_id"):
                continue
            participants.append(entry)
        participants.sort(
            key=lambda item: (
                0 if item.get("is_current_target") else 1,
                0 if item.get("in_current_scene") else 1,
                0 if item.get("recent_targeted_interaction") else 1,
                -_safe_float(item.get("impact_rank", 0.0)),
                _safe_float(item.get("last_interaction_age_sec", 999999.0)),
            )
        )
        payload["participants"] = participants[:8]
    except Exception as exc:
        logger.debug(f"关系群体导出失败: {exc}")
    return payload


def _extract_domains(chat: Any, channel_id: str) -> Dict[str, Any]:
    world_state = _extract_world_snapshot(chat)
    runtime = _extract_runtime_snapshot(chat)
    energy = _extract_energy_domain(channel_id)
    emergence = _extract_emergence_state(channel_id)
    relation = _extract_relationship_profile(chat, world_state)
    relationship_population = _extract_relationship_population(chat, relation.get("user_id", ""))
    memory_stack = _extract_memory_stack(channel_id)
    autonomy_runtime = _extract_autonomy_runtime(chat, channel_id)
    attention_runtime = _extract_attention_runtime(channel_id, runtime)
    safety_runtime = _extract_safety_runtime(runtime)
    subject = world_state.get("subject", {}) if isinstance(world_state, dict) else {}
    scene = world_state.get("scene", {}) if isinstance(world_state, dict) else {}
    behavior = world_state.get("behavior", {}) if isinstance(world_state, dict) else {}
    meta = world_state.get("meta", {}) if isinstance(world_state, dict) else {}
    circadian = _extract_circadian_rhythm(channel_id, runtime, subject)
    group_climate = _extract_group_climate(channel_id, scene, runtime)
    context_awareness = _extract_context_awareness(
        chat,
        runtime,
        subject,
        relation,
        memory_stack,
        autonomy_runtime,
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
            "relationship_label": relation.get("personal_impression", "") or relation.get("relationship_label", ""),
            "personal_impression": relation.get("personal_impression", ""),
            "impression_labels": list(relation.get("impression_labels", []) or []),
            "legacy_relationship_label": relation.get("legacy_relationship_label", ""),
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
        "group_climate": group_climate,
        "resource_ledger": energy,
        "social_field": {
            "social_field_score": round(_safe_float(energy.get("social_field_score", 0.0)), 2),
            "rapport_score": relation.get("rapport_score", 0.0),
            "trust_score": relation.get("trust_score", 0.0),
            "interaction_count": relation.get("interaction_count", 0),
        },
        "decision_runtime": runtime.get("last_decision_runtime") or {},
        "execution_runtime": runtime.get("last_execution_runtime") or {},
        "timing_gate": {
            "current": runtime.get("last_timing_gate_runtime") or {},
            "history": list(runtime.get("timing_gate_history") or [])[-20:],
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
        "memory_stack": memory_stack,
        "autonomy_runtime": autonomy_runtime,
        "attention_runtime": attention_runtime,
        "safety_runtime": safety_runtime,
        "context_awareness": context_awareness,
        "relationship_profile": relation,
        "relationship_population": relationship_population,
        "emergence_core": emergence,
        "circadian_rhythm": circadian,
        "flow_runtime": {
            "phase": str(subject.get("phase", "standby") or "standby"),
            "phase_label": str(subject.get("phase_label", "待命") or "待命"),
            "watch_state": str(subject.get("watch_level", "peek") or "peek"),
            "watch_state_label": str(subject.get("watch_level_label", "瞥一眼") or "瞥一眼"),
            "night_phase": circadian.get("phase", "awake"),
            "night_phase_label": circadian.get("phase_label", "清醒"),
            "sleeping": bool(circadian.get("is_sleeping", False)),
            "reply_allowed": bool(subject.get("can_reply", True)) and bool(circadian.get("can_reply", True)),
            "pending_active": pending_active,
            "last_target_user_id": relation.get("user_id", ""),
            "topic_focus": list(group_climate.get("topic_focus", []) or []),
            "last_reactive_plan": runtime.get("last_reactive_plan") or {},
            "turn_schedule": runtime.get("last_turn_schedule") or {},
        },
    }


def list_heartfc_chats() -> Dict[str, Any]:
    from src.chat.heart_flow.heartflow import heartflow
    from src.chat.message_receive.chat_stream import get_chat_manager

    now = time.time()
    streams = get_chat_manager()
    raw_chats = []
    for chat_id, chat in heartflow.heartflow_chat_list.items():
        stream = streams.get_stream(str(chat_id))
        last_active = float(heartflow._active_since.get(chat_id, 0.0) or 0.0)
        chat_type = "group" if stream and stream.group_info else "private"
        platform = str(getattr(stream, "platform", "") if stream else "" or "").strip()
        platform_label = {
            "qq": "QQ",
            "webui": "本地测试",
        }.get(platform.lower(), platform.upper() if platform else "本地")
        group_info = getattr(stream, "group_info", None) if stream else None
        user_info = getattr(stream, "user_info", None) if stream else None
        group_name = str(getattr(group_info, "group_name", "") or getattr(group_info, "group_id", "") or "")
        user_name = str(
            getattr(user_info, "user_nickname", "")
            or getattr(user_info, "user_cardname", "")
            or getattr(user_info, "user_id", "")
            or ""
        )
        display_name = group_name if chat_type == "group" else user_name
        is_internal_webui = platform.lower() == "webui"
        raw_chats.append(
            {
                "channel_id": str(chat_id),
                "chat_type": chat_type,
                "chat_type_label": "群聊" if chat_type == "group" else "私聊",
                "display_name": display_name,
                "platform_label": platform_label,
                "is_internal_webui": is_internal_webui,
                "class_name": chat.__class__.__name__,
                "last_active": last_active,
                "idle_seconds": round(max(0.0, now - last_active), 3) if last_active else None,
                "platform": platform,
                "group_id": getattr(group_info, "group_id", None) if group_info else None,
                "group_name": group_name,
                "user_id": getattr(user_info, "user_id", None) if user_info else None,
                "user_name": user_name,
                "target_user_id": str(getattr(chat, "_last_user_id", "") or ""),
            }
        )
    public_chats = [item for item in raw_chats if not item.get("is_internal_webui")]
    chats = public_chats or raw_chats
    chats.sort(
        key=lambda item: (
            1 if item.get("is_internal_webui") else 0,
            0 if item.get("chat_type") == "group" else 1,
            str(item.get("platform_label", "")),
            str(item.get("display_name", "")),
        )
    )
    return {
        "active_count": len(chats),
        "hidden_internal_count": max(0, len(raw_chats) - len(chats)),
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

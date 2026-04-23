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
        "dawn_recover": "黎明恢复",
        "night_active": "夜间活跃",
        "midnight_reflect": "午夜反思",
        "social_night": "社交夜",
        "quiet_contemplate": "安静沉思",
    }.get(phase, "清醒")


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
        "cached_night_phase": "_cached_night_phase",
        "cached_night_summary": "_cached_night_summary",
        "cached_metabolism_constraints": "_cached_metabolism_constraints",
        "cached_emotion_state": "_cached_emotion_state",
        "cached_emotion_feedback_report": "_cached_emotion_feedback_report",
        "cached_awareness": "_cached_awareness",
        "cached_presence_verdict": "_cached_presence_verdict",
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
            is_night = phase not in {"awake"}
            dawn_progress = _safe_float(getattr(state, "dawn_recovery_progress", dawn_progress))
            sleep_reply_used = _safe_int(getattr(state, "sleep_reply_used", sleep_reply_used))
            sleep_reply_cap = _safe_int(getattr(state, "night_reply_cap", sleep_reply_cap))
            try:
                style = night_cycle.get_expression_deformation()
                expression_style = str(style.get("active_template", expression_style) or expression_style)
            except Exception:
                pass
    except Exception as exc:
        logger.debug(f"昼夜节律导出失败: {exc}")

    if sleep_reply_cap <= 0:
        sleep_reply_cap = max(remaining_replies, 0)
    if remaining_replies <= 0 and sleep_reply_cap > 0:
        remaining_replies = max(0, sleep_reply_cap - sleep_reply_used)

    return {
        "phase": phase,
        "phase_label": phase_label or _night_phase_label(phase),
        "is_night": is_night,
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
        from src.modules.modcore.social_cognition.knowledge_graph import get_knowledge_graph_manager

        kg = get_knowledge_graph_manager(channel_id)
        payload["knowledge_entry_count"] = _safe_int(getattr(kg, "entry_count", 0))
    except Exception as exc:
        logger.debug(f"知识图谱导出失败: {exc}")
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
    memory_stack = _extract_memory_stack(channel_id)
    autonomy_runtime = _extract_autonomy_runtime(chat, channel_id)
    subject = world_state.get("subject", {}) if isinstance(world_state, dict) else {}
    scene = world_state.get("scene", {}) if isinstance(world_state, dict) else {}
    behavior = world_state.get("behavior", {}) if isinstance(world_state, dict) else {}
    meta = world_state.get("meta", {}) if isinstance(world_state, dict) else {}
    circadian = _extract_circadian_rhythm(channel_id, runtime, subject)
    context_awareness = _extract_context_awareness(
        chat,
        runtime,
        subject,
        relation,
        memory_stack,
        autonomy_runtime,
    )
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
        "decision_runtime": runtime.get("last_decision_runtime") or {},
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
        "context_awareness": context_awareness,
        "relationship_profile": relation,
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
        chat_type = "group" if stream and stream.group_info else "private"
        chats.append(
            {
                "channel_id": str(chat_id),
                "chat_type": chat_type,
                "chat_type_label": "群聊" if chat_type == "group" else "私聊",
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

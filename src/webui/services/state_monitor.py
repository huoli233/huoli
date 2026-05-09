import copy
import time
from typing import Any, Dict, Optional, Tuple

from src.chat.heart_flow.heartfc_state_exporter import export_heartfc_state, list_heartfc_chats
from src.chat.heart_flow.speak_prediction_engine import get_speak_prediction_engine
from src.webui.services.runtime_state_hub import MONITOR_OVERVIEW_CHANNEL_ID, get_runtime_state_hub
from src.webui.runtime_config import webui_module_view

_DYNAMIC_TRACE_CACHE: Dict[str, Dict[str, float]] = {}
_STATE_MONITOR_DEFAULTS: Dict[str, Dict[str, float]] = {
    "emotion": {
        "boredom_show": 0.35,
        "boredom_high": 0.70,
        "loneliness_show": 0.35,
        "loneliness_high": 0.70,
        "fatigue_show": 0.35,
        "fatigue_high": 0.75,
        "withdrawal_show": 0.25,
        "withdrawal_high": 0.55,
        "initiative_high": 0.70,
        "initiative_low": 0.25,
        "curiosity_show": 0.65,
        "mood_low": 0.25,
        "social_desire_high": 0.75,
        "social_desire_low": 0.25,
        "mood_happy": 0.72,
        "mood_positive": 0.55,
        "mood_calm": 0.40,
        "mood_low_label": 0.25,
    },
    "relationship": {
        "irritation_show": 15.0,
        "irritation_high": 45.0,
        "pressure_show": 15.0,
        "pressure_high": 45.0,
        "rapport_high": 30.0,
        "rapport_low": -10.0,
        "trust_high": 70.0,
        "trust_low": 25.0,
        "overstep_show": 5.0,
        "intimacy_show": 5.0,
        "compliance_show": 5.0,
        "compliance_high": 25.0,
        "conditioning_show": 5.0,
        "conditioning_high": 25.0,
        "resistance_delta_show": 10.0,
        "resistance_low": 90.0,
        "authority_show": 5.0,
        "authority_high": 25.0,
        "current_user_irritation_high": 60.0,
        "current_user_pressure_high": 45.0,
        "current_user_trauma_high": 1.2,
        "current_user_rapport_warm": 35.0,
        "current_user_trust_warm": 35.0,
        "current_user_rapport_steady": 15.0,
        "current_user_trust_steady": 20.0,
    },
    "trauma": {
        "trauma_show": 0.8,
        "trauma_critical": 2.5,
        "chaos_show": 1.0,
        "chaos_critical": 4.5,
        "flashback_show": 0.15,
        "flashback_high": 0.45,
        "cognitive_drag_show": 0.15,
        "cognitive_drag_high": 0.45,
        "mask_show": 1.0,
        "mask_high": 4.0,
    },
    "scene": {
        "scene_heat_show": 0.55,
        "scene_heat_high": 0.78,
    },
    "pending": {
        "pending_seconds_medium": 120.0,
    },
    "memory": {
        "overload_show": 0.8,
        "overload_critical": 0.92,
        "memoir_timeout_show": 3.0,
        "memoir_timeout_high": 5.0,
    },
    "autonomy": {
        "intention_show": 0.70,
        "intention_high": 0.90,
        "pending_proactive_medium": 3.0,
    },
    "attention": {
        "peek_desire_show": 0.65,
        "process_low": 0.12,
        "visibility_threshold": 0.55,
        "empty_peeks_show": 3.0,
        "empty_peeks_high": 5.0,
    },
    "safety": {
        "safety_show": 0.45,
        "safety_critical": 0.75,
    },
    "circadian": {
        "drowsiness_show": 60.0,
        "drowsiness_high": 80.0,
        "sleep_debt_show": 0.45,
        "sleep_debt_high": 0.75,
        "sleep_reserve_low": 25.0,
    },
    "prediction": {
        "probability_high": 0.70,
        "probability_medium": 0.45,
    },
}


def _overlay_dicts(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _overlay_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _state_monitor_thresholds() -> Dict[str, Dict[str, float]]:
    override = webui_module_view("webui_state_monitor_thresholds")
    if not isinstance(override, dict):
        override = {}
    return _overlay_dicts(_STATE_MONITOR_DEFAULTS, override)


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


def _format_percent(value: float) -> str:
    return f"{max(0.0, min(1.0, value)) * 100:.1f}%"


def _ratio_metric(value: float, *, precision: int = 1) -> Dict[str, Any]:
    ratio = max(0.0, min(1.0, float(value or 0.0)))
    return {
        "raw": round(ratio, 4),
        "percent": round(ratio * 100.0, precision),
        "display_precision": precision,
        "display_value": f"{ratio * 100.0:.{precision}f}%",
    }


def _label_energy_phase(value: Any) -> str:
    return {
        "full": "精力充沛",
        "adequate": "储备充足",
        "low": "储备偏低",
        "critical": "接近见底",
        "depleted": "已经耗尽",
        "unknown": "未知",
    }.get(str(value or "").strip().lower(), "未知")


def _label_expression_style(value: Any) -> str:
    return {
        "normal": "正常",
        "drowsy": "困倦短句",
        "stubborn": "硬撑克制",
        "irritable": "被扰烦躁",
        "soft_night": "夜间放轻",
        "arousal": "短时亢奋",
        "burnthrough": "熬穿失衡",
    }.get(str(value or "").strip().lower(), str(value or "") or "正常")


def _label_scene_heat(value: Any) -> str:
    return {
        "heated": "高热",
        "lively": "活跃",
        "normal": "正常",
        "quiet": "偏安静",
        "dead": "低活跃",
    }.get(str(value or "").strip().lower(), str(value or "") or "-")


def _label_family(value: Any) -> str:
    return {
        "emotion": "情绪",
        "drive": "主动性",
        "relationship": "关系",
        "trauma": "创伤",
        "mask": "伪装",
        "conditioning": "调教",
        "runtime": "运行",
        "scene": "会话氛围",
        "circadian": "夜间",
    }.get(str(value or "").strip().lower(), str(value or "") or "状态")


def _label_trend(value: Any) -> str:
    return {
        "rising": "上升",
        "falling": "下降",
        "persistent": "持续",
        "volatile": "波动",
        "holding": "保持",
        "ready": "已就绪",
        "warm": "升温",
        "cold": "降温",
        "stable": "稳定",
        "fragile": "脆弱",
        "active": "激活",
        "recovering": "恢复中",
    }.get(str(value or "").strip().lower(), str(value or "") or "稳定")


def _label_source(value: Any) -> str:
    return {
        "emergence_core": "涌现核心",
        "relationship_profile": "关系档案",
        "trauma_load": "创伤负荷",
        "surface_mask": "表层伪装",
        "tempo": "节奏控制",
        "group_climate": "会话氛围",
        "pending_response": "待回应状态",
        "circadian_rhythm": "夜间状态",
        "memory_stack": "记忆栈",
        "autonomy_runtime": "自主运行",
        "context_awareness": "上下文感知",
        "attention_runtime": "注意力运行态",
        "safety_runtime": "安全护盾",
    }.get(str(value or "").strip(), str(value or "") or "状态源")


def _label_reply_mode(value: Any) -> str:
    return {
        "observe": "观察",
        "reply": "回复",
        "proactive": "主动开口",
        "defer": "延后",
        "rest": "休息",
    }.get(str(value or "").strip().lower(), str(value or "") or "观察")


def _label_interrupt_level(value: Any) -> str:
    return {
        "ignore": "忽略",
        "peek": "窥屏",
        "skim": "扫视",
        "engage": "参与",
    }.get(str(value or "").strip().lower(), str(value or "") or "忽略")


def _label_quote_policy(value: Any) -> str:
    return {
        "none": "不引用",
        "soft_reference": "软引用",
        "quote": "引用",
    }.get(str(value or "").strip().lower(), str(value or "") or "不引用")


def _label_silence_policy(value: Any) -> str:
    return {
        "silent": "静默",
        "ambient": "随群",
        "reengage": "重新接入",
    }.get(str(value or "").strip().lower(), str(value or "") or "静默")


def _label_model_tier(value: Any) -> str:
    return {
        "skip": "跳过",
        "small": "小模型",
        "large": "大模型",
    }.get(str(value or "").strip().lower(), str(value or "") or "跳过")


def _label_memoir_phase(value: Any) -> str:
    return {
        "open": "开放会话",
        "anticipating": "等待回应",
    }.get(str(value or "").strip().lower(), str(value or "") or "开放会话")


def _label_background_event(value: Any) -> str:
    return {
        "proactive_greeting": "主动打招呼",
        "random_observation": "随机观察",
        "mood_expression": "心情表达",
        "memory_recall": "记忆触发",
        "topic_resurrection": "旧话题复燃",
    }.get(str(value or "").strip().lower(), str(value or "") or "无后台事件")


def _label_rest_posture(value: Any) -> str:
    return {
        "active": "活跃",
        "peek_only": "只窥屏",
        "loaf": "摸鱼",
        "rest": "休息",
    }.get(str(value or "").strip().lower(), str(value or "") or "活跃")


def _label_rest_interrupt(value: Any) -> str:
    return {
        "allow": "允许",
        "peek_only": "只窥屏",
        "block": "阻断",
    }.get(str(value or "").strip().lower(), str(value or "") or "允许")


def _label_behavior_reason(code: Any) -> str:
    mapping = {
        "admin_force": "管理员强制",
        "night_rest": "夜间休息",
        "resource_exhausted": "资源不足",
        "high_loafing_quiet": "高摸鱼高安静偏好",
        "soft_loafing_guard": "轻度摸鱼保护",
        "ignored_chain_rest": "连续未回应后休息",
        "watch_low_quiet": "关注度低且偏安静",
        "direct_target_override": "直接目标覆盖",
        "quiet_idle_loaf": "安静空窗摸鱼",
        "direct_relevance": "直接相关",
        "weak_human_signal": "弱人类信号",
        "ambient_human_signal": "环境信号",
        "recent_human_activity": "最近有人说话",
        "interest_rebound": "兴趣回弹",
        "default_silence": "默认静默",
        "no_anchor_quote_none": "无锚点不引用",
        "explicit_quote_anchor": "显式引用锚点",
        "anchor_soft_reference": "软引用锚点",
        "anchor_timeout_quote_none": "锚点超时",
        "recent_targeted_quote": "近期目标引用",
        "stale_context_soft_reference": "旧上下文软引用",
        "context_timeout_quote_none": "上下文超时",
        "night_freeze": "夜间冻结",
        "ignored_chain": "连续未回应",
        "chatterbox_guard": "话痨保护",
        "air_chat_guard": "空气对话保护",
        "consecutive_speaks_guard": "连续发言保护",
        "idle_backoff": "空窗退避",
        "scene_allows_reengage": "场景允许重新接入",
        "scene_unsuitable": "场景不适合",
        "long_silence": "长时间沉默",
        "risk_large_candidate": "高风险升级大模型",
        "behavior_model_skip": "行为层建议跳过模型",
        "behavior_blocked": "行为层阻断",
        "low_signal_skip": "低信号跳过",
        "large_demoted_noncritical": "非关键场景降级",
        "large_high_value": "高价值场景升级",
        "behavior_prefers_skip": "行为层偏向跳过",
    }
    payload = str(code or "").strip()
    return mapping.get(payload, payload or "未命名原因")


def _label_final_action(value: Any) -> str:
    return {
        "continue": "继续管线",
        "reply": "已回复",
        "upgrade": "升级处理",
        "observe": "继续观察",
        "rest": "转入休息",
        "disengage": "放下会话",
        "lurk": "退回潜水",
        "skip": "本轮跳过",
    }.get(str(value or "").strip().lower(), str(value or "") or "继续观察")


def _label_execution_stage(value: Any) -> str:
    return {
        "self_echo_gate": "自身消息回流跳过",
        "short_batch_skip": "消息未达阈值",
        "watch_gate_blackout": "黑屏观看阻断",
        "peek_gate_observe": "窥屏后继续观察",
        "night_gate_skip": "夜间节律阻断",
        "autonomy_guard_skip": "自治守卫阻断",
        "planner_cooldown_skip": "规划器冷却跳过",
        "pipeline_early_exit": "输入早退出",
        "dashboard_hard_block": "仪表盘硬阻断",
        "pattern_route_skip": "群体模式硬路由",
        "scene_constraint_skip": "会话场景硬约束",
        "full_pipeline_entry": "完整管线入口",
        "decision_runtime_skip": "初裁直接跳过",
        "voice_action_rest": "内心要求休息",
        "voice_action_disengage": "内心要求放下会话",
        "voice_action_lurk": "内心要求潜水观察",
        "gateway_block": "门控明确阻断",
        "llm_autonomous_allow": "自主大模型放行",
        "llm_autonomous_hold": "自主大模型保持观察",
        "inner_voice_priority": "内心优先规则修正",
        "post_voice_reluctant": "独白抗拒保持观察",
        "post_gate_hold": "证据不足保持观察",
        "post_gate_reply": "门控放行后转入回复",
        "post_relation_block": "关系极端阻断",
        "post_relation_guard": "关系情绪收缩",
        "post_wait_intent_block": "等待意图压制",
        "post_psychology_block": "心理门槛收缩",
        "post_probability_block": "频率概率收缩",
        "post_burst_block": "连续发言保护",
        "post_negative_emotion_block": "负面情绪回避",
        "pre_execution": "前置裁定阶段",
        "reply_sent": "已成功发送",
        "reply_aborted": "进入执行后中止",
        "final_no_action": "最终未执行动作",
    }.get(str(value or "").strip().lower(), str(value or "") or "执行中")


def _label_timing_gate_result(value: Any) -> str:
    return {
        "continue": "继续完整管线",
        "no_reply": "不回复",
        "wait": "等待",
        "observe": "观察",
        "task_only": "仅处理任务",
    }.get(str(value or "").strip().lower(), str(value or "") or "观察")


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
        "family_label": _label_family(family),
        "severity": severity,
        "value": round(value, 3),
        "display_value": display_value,
        "trend": trend,
        "trend_label": _label_trend(trend),
        "source_domain": source_domain,
        "source_domain_label": _label_source(source_domain),
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
    group_climate = domains.get("group_climate", {})
    circadian = domains.get("circadian_rhythm", {})
    memory_stack = domains.get("memory_stack", {})
    autonomy = domains.get("autonomy_runtime", {})
    attention_runtime = domains.get("attention_runtime", {})
    safety_runtime = domains.get("safety_runtime", {})
    execution_runtime = domains.get("execution_runtime", {})
    thresholds = _state_monitor_thresholds()
    emotion_cfg = thresholds["emotion"]
    relationship_cfg = thresholds["relationship"]
    trauma_cfg = thresholds["trauma"]
    scene_cfg = thresholds["scene"]
    pending_cfg = thresholds["pending"]
    memory_cfg = thresholds["memory"]
    autonomy_cfg = thresholds["autonomy"]
    attention_cfg = thresholds["attention"]
    safety_cfg = thresholds["safety"]
    circadian_cfg = thresholds["circadian"]

    signals: list[Dict[str, Any]] = []

    boredom = _safe_float(emergence.get("boredom_load", 0.0))
    if boredom >= emotion_cfg["boredom_show"]:
        signals.append(
            _signal_card(
                key="boredom_load",
                label="无聊度",
                family="emotion",
                severity="medium" if boredom < emotion_cfg["boredom_high"] else "high",
                value=boredom,
                display_value=_format_percent(boredom),
                trend="rising",
                source_domain="emergence_core",
                icon="🥱",
            )
        )

    loneliness = _safe_float(emergence.get("loneliness_load", 0.0))
    if loneliness >= emotion_cfg["loneliness_show"]:
        signals.append(
            _signal_card(
                key="loneliness_load",
                label="孤独感",
                family="emotion",
                severity="medium" if loneliness < emotion_cfg["loneliness_high"] else "high",
                value=loneliness,
                display_value=_format_percent(loneliness),
                trend="rising",
                source_domain="emergence_core",
                icon="🫥",
            )
        )

    fatigue = _safe_float(emergence.get("environment_fatigue_load", 0.0))
    if fatigue >= emotion_cfg["fatigue_show"]:
        signals.append(
            _signal_card(
                key="environment_fatigue_load",
                label="疲劳感",
                family="emotion",
                severity="medium" if fatigue < emotion_cfg["fatigue_high"] else "high",
                value=fatigue,
                display_value=_format_percent(fatigue),
                trend="rising",
                source_domain="emergence_core",
                icon="🌫️",
            )
        )

    withdrawal = _safe_float(emergence.get("withdrawal_drive", 0.0))
    if withdrawal >= emotion_cfg["withdrawal_show"]:
        signals.append(
            _signal_card(
                key="withdrawal_drive",
                label="退场倾向",
                family="emotion",
                severity="medium" if withdrawal < emotion_cfg["withdrawal_high"] else "high",
                value=withdrawal,
                display_value=_format_percent(withdrawal),
                trend="rising",
                source_domain="emergence_core",
                icon="🚪",
            )
        )

    initiative = _safe_float(emergence.get("initiative_drive", 0.0))
    if initiative >= emotion_cfg["initiative_high"] or initiative <= emotion_cfg["initiative_low"]:
        signals.append(
            _signal_card(
                key="initiative_drive",
                label="主动意愿",
                family="drive",
                severity="high" if initiative >= emotion_cfg["initiative_high"] else "low",
                value=initiative,
                display_value=_format_percent(initiative),
                trend="rising" if initiative >= emotion_cfg["initiative_high"] else "falling",
                source_domain="emergence_core",
                icon="🗣️" if initiative >= emotion_cfg["initiative_high"] else "🤐",
            )
        )

    irritation = _safe_float(relationship.get("irritation_load", 0.0))
    if irritation >= relationship_cfg["irritation_show"]:
        signals.append(
            _signal_card(
                key="irritation_load",
                label="烦躁负荷",
                family="relationship",
                severity="medium" if irritation < relationship_cfg["irritation_high"] else "high",
                value=irritation,
                display_value=f"{irritation:.1f}",
                trend="rising",
                source_domain="relationship_profile",
                icon="😒",
            )
        )

    pressure = _safe_float(relationship.get("pressure_load", 0.0))
    if pressure >= relationship_cfg["pressure_show"]:
        signals.append(
            _signal_card(
                key="pressure_load",
                label="心理压力",
                family="relationship",
                severity="medium" if pressure < relationship_cfg["pressure_high"] else "high",
                value=pressure,
                display_value=f"{pressure:.1f}",
                trend="rising",
                source_domain="relationship_profile",
                icon="🧠",
            )
        )

    trauma_load = _safe_float(trauma.get("trauma_load", 0.0))
    if trauma_load >= trauma_cfg["trauma_show"]:
        signals.append(
            _signal_card(
                key="trauma_load",
                label="创伤负荷",
                family="trauma",
                severity="high" if trauma_load < trauma_cfg["trauma_critical"] else "critical",
                value=trauma_load,
                display_value=f"{trauma_load:.2f}",
                trend="persistent",
                source_domain="trauma_load",
                icon="🩹",
            )
        )

    chaos_load = _safe_float(trauma.get("chaos_load", 0.0))
    if chaos_load >= trauma_cfg["chaos_show"]:
        signals.append(
            _signal_card(
                key="chaos_load",
                label="内心混乱",
                family="trauma",
                severity="medium" if chaos_load < trauma_cfg["chaos_critical"] else "critical",
                value=chaos_load,
                display_value=f"{chaos_load:.2f}",
                trend="rising",
                source_domain="trauma_load",
                icon="🌀",
            )
        )

    flashback = _safe_float(trauma.get("flashback_probability", 0.0))
    if flashback >= trauma_cfg["flashback_show"]:
        signals.append(
            _signal_card(
                key="flashback_probability",
                label="闪回风险",
                family="trauma",
                severity="medium" if flashback < trauma_cfg["flashback_high"] else "high",
                value=flashback,
                display_value=_format_percent(flashback),
                trend="volatile",
                source_domain="trauma_load",
                icon="⚡",
            )
        )

    cognitive_drag = _safe_float(trauma.get("cognitive_drag", 0.0))
    if cognitive_drag >= trauma_cfg["cognitive_drag_show"]:
        signals.append(
            _signal_card(
                key="cognitive_drag",
                label="认知拖拽",
                family="trauma",
                severity="medium" if cognitive_drag < trauma_cfg["cognitive_drag_high"] else "high",
                value=cognitive_drag,
                display_value=_format_percent(cognitive_drag),
                trend="persistent",
                source_domain="trauma_load",
                icon="🧩",
            )
        )

    mask_load = _safe_float(surface_mask.get("mask_load", 0.0))
    if mask_load >= trauma_cfg["mask_show"]:
        signals.append(
            _signal_card(
                key="mask_load",
                label="伪装负荷",
                family="mask",
                severity="medium" if mask_load < trauma_cfg["mask_high"] else "high",
                value=mask_load,
                display_value=f"{mask_load:.2f}",
                trend="persistent",
                source_domain="surface_mask",
                icon="🎭",
            )
        )

    compliance = _safe_float(relationship.get("compliance_index", 0.0))
    if compliance >= relationship_cfg["compliance_show"]:
        signals.append(
            _signal_card(
                key="compliance_index",
                label="顺从指数",
                family="conditioning",
                severity="low" if compliance < relationship_cfg["compliance_high"] else "medium",
                value=compliance,
                display_value=f"{compliance:.1f}",
                trend="rising",
                source_domain="relationship_profile",
                icon="🪢",
            )
        )

    conditioning = _safe_float(relationship.get("conditioning_progress", 0.0))
    if conditioning >= relationship_cfg["conditioning_show"]:
        signals.append(
            _signal_card(
                key="conditioning_progress",
                label="调教进度",
                family="conditioning",
                severity="low" if conditioning < relationship_cfg["conditioning_high"] else "medium",
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
    if authority >= relationship_cfg["authority_show"]:
        signals.append(
            _signal_card(
                key="authority_imprint",
                label="权威印记",
                family="conditioning",
                severity="low" if authority < relationship_cfg["authority_high"] else "medium",
                value=authority,
                display_value=f"{authority:.1f}",
                trend="rising",
                source_domain="relationship_profile",
                icon="👑",
            )
        )

    intimacy = _safe_float(relationship.get("intimacy_depth", 0.0))
    if intimacy >= relationship_cfg["intimacy_show"]:
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

    rapport = _safe_float(relationship.get("rapport_score", 0.0))
    if rapport >= relationship_cfg["rapport_high"] or rapport <= relationship_cfg["rapport_low"]:
        signals.append(
            _signal_card(
                key="rapport_score",
                label="好感关系",
                family="relationship",
                severity="info" if rapport >= relationship_cfg["rapport_high"] else "medium",
                value=rapport,
                display_value=f"{rapport:.1f}",
                trend="warm" if rapport >= relationship_cfg["rapport_high"] else "cold",
                source_domain="relationship_profile",
                icon="🤝" if rapport >= relationship_cfg["rapport_high"] else "🧊",
            )
        )

    trust = _safe_float(relationship.get("trust_score", 0.0))
    if trust >= relationship_cfg["trust_high"] or (0 < trust <= relationship_cfg["trust_low"]):
        signals.append(
            _signal_card(
                key="trust_score",
                label="信任关系",
                family="relationship",
                severity="info" if trust >= relationship_cfg["trust_high"] else "high",
                value=trust,
                display_value=f"{trust:.1f}",
                trend="stable" if trust >= relationship_cfg["trust_high"] else "fragile",
                source_domain="relationship_profile",
                icon="🟢" if trust >= relationship_cfg["trust_high"] else "🟠",
            )
        )

    overstep = _safe_float(relationship.get("boundary_overstep_index", 0.0))
    if overstep >= relationship_cfg["overstep_show"]:
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

    scene_heat_score = _safe_float(group_climate.get("scene_heat_score", 0.0))
    if scene_heat_score >= scene_cfg["scene_heat_show"]:
        signals.append(
            _signal_card(
                key="scene_heat_score",
                label="会话升温",
                family="scene",
                severity="medium" if scene_heat_score < scene_cfg["scene_heat_high"] else "high",
                value=scene_heat_score,
                display_value=_format_percent(scene_heat_score),
                trend="rising",
                source_domain="group_climate",
                icon="🔥",
            )
        )

    pending_seconds = _safe_float(pending.get("pending_seconds", 0.0))
    if pending_seconds > 0:
        signals.append(
            _signal_card(
                key="pending_seconds",
                label="等待时长",
                family="runtime",
                severity="info" if pending_seconds < pending_cfg["pending_seconds_medium"] else "medium",
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

    execution_blocker = str(execution_runtime.get("blocker", "") or execution_runtime.get("execution_reason", "") or "")
    if bool(execution_runtime.get("verdict_id")) and not bool(execution_runtime.get("should_act", False)):
        signals.append(
            _signal_card(
                key="execution_blocked",
                label="执行裁定收口",
                family="runtime",
                severity="medium",
                value=1.0,
                display_value=execution_blocker or "本轮决定观察",
                trend="holding",
                source_domain="execution_runtime",
                icon="🧭",
            )
        )

    memory_overload = memory_stack.get("overload", {}) if isinstance(memory_stack.get("overload"), dict) else {}
    load_ratio = _safe_float(memory_overload.get("load_ratio", 0.0))
    if bool(memory_overload.get("emergency_needed", False)) or load_ratio >= memory_cfg["overload_show"]:
        signals.append(
            _signal_card(
                key="memory_overload",
                label="记忆过载",
                family="runtime",
                severity="high" if load_ratio < memory_cfg["overload_critical"] else "critical",
                value=load_ratio,
                display_value=_format_percent(load_ratio),
                trend="rising",
                source_domain="memory_stack",
                icon="🧠",
            )
        )

    memoir_timeouts = _safe_float(memory_stack.get("memoir_consecutive_timeouts", 0.0))
    if memoir_timeouts >= memory_cfg["memoir_timeout_show"]:
        signals.append(
            _signal_card(
                key="memoir_consecutive_timeouts",
                label="连续未回应",
                family="runtime",
                severity="medium" if memoir_timeouts < memory_cfg["memoir_timeout_high"] else "high",
                value=memoir_timeouts,
                display_value=f"{int(memoir_timeouts)}次",
                trend="rising",
                source_domain="memory_stack",
                icon="🕰️",
            )
        )

    intention_drive = _safe_float(autonomy.get("intention_drive", 0.0))
    if intention_drive >= autonomy_cfg["intention_show"]:
        signals.append(
            _signal_card(
                key="intention_drive",
                label="主动意图堆积",
                family="drive",
                severity="medium" if intention_drive < autonomy_cfg["intention_high"] else "high",
                value=intention_drive,
                display_value=_format_percent(intention_drive),
                trend="rising",
                source_domain="autonomy_runtime",
                icon="🧭",
            )
        )

    pending_proactive = _safe_float(autonomy.get("pending_proactive_events", 0.0))
    if pending_proactive > 0:
        signals.append(
            _signal_card(
                key="pending_proactive_events",
                label="主动事件待结算",
                family="runtime",
                severity="info" if pending_proactive < autonomy_cfg["pending_proactive_medium"] else "medium",
                value=pending_proactive,
                display_value=f"{int(pending_proactive)}条",
                trend="holding",
                source_domain="autonomy_runtime",
                icon="📌",
            )
        )

    background_event = autonomy.get("background_event", {}) if isinstance(autonomy.get("background_event"), dict) else {}
    if str(background_event.get("type", "") or "").strip():
        signals.append(
            _signal_card(
                key="background_event",
                label="后台事件排队",
                family="runtime",
                severity="info",
                value=_safe_float(background_event.get("priority", 0.0)),
                display_value=_label_background_event(background_event.get("type", "")),
                trend="ready",
                source_domain="autonomy_runtime",
                icon="🎲",
            )
        )

    peek_desire = _safe_float(attention_runtime.get("peek_desire", 0.0))
    if peek_desire >= attention_cfg["peek_desire_show"]:
        signals.append(
            _signal_card(
                key="peek_desire_high",
                label="窥屏欲望偏高",
                family="runtime",
                severity="info",
                value=peek_desire,
                display_value=_format_percent(peek_desire),
                trend="rising",
                source_domain="attention_runtime",
                icon="👁️",
            )
        )

    process_ratio = _safe_float(attention_runtime.get("process_ratio", 0.0))
    visibility_threshold = _safe_float(attention_runtime.get("visibility_threshold", 0.0))
    if process_ratio <= attention_cfg["process_low"] and visibility_threshold >= attention_cfg["visibility_threshold"]:
        signals.append(
            _signal_card(
                key="attention_suppressed",
                label="注意力受抑",
                family="runtime",
                severity="medium",
                value=1.0 - process_ratio,
                display_value=_format_percent(1.0 - process_ratio),
                trend="holding",
                source_domain="attention_runtime",
                icon="📡",
            )
        )

    empty_peeks = _safe_float(attention_runtime.get("consecutive_peeks_without_action", 0.0))
    if empty_peeks >= attention_cfg["empty_peeks_show"]:
        signals.append(
            _signal_card(
                key="consecutive_peeks_without_action",
                label="连续窥屏未行动",
                family="runtime",
                severity="info" if empty_peeks < attention_cfg["empty_peeks_high"] else "medium",
                value=empty_peeks,
                display_value=f"{int(empty_peeks)}次",
                trend="holding",
                source_domain="attention_runtime",
                icon="🫣",
            )
        )

    safety_score = _safe_float(safety_runtime.get("score", 0.0))
    if bool(safety_runtime.get("blocked", False)) or safety_score >= safety_cfg["safety_show"]:
        signals.append(
            _signal_card(
                key="safety_threat",
                label="安全护盾收紧",
                family="runtime",
                severity="high" if safety_score < safety_cfg["safety_critical"] else "critical",
                value=safety_score,
                display_value=str(safety_runtime.get("dominant_threat", "") or "已触发风险"),
                trend="holding",
                source_domain="safety_runtime",
                icon="🛡️",
            )
        )

    if bool(circadian.get("is_burnthrough", False)):
        signals.append(
            _signal_card(
                key="burnthrough",
                label="熬穿状态",
                family="circadian",
                severity="critical",
                value=1.0,
                display_value="已熬穿",
                trend="volatile",
                source_domain="circadian_rhythm",
                icon="🔥",
            )
        )
    elif str(circadian.get("phase", "")) == "deep_sleep":
        signals.append(
            _signal_card(
                key="deep_sleep",
                label="深睡中",
                family="circadian",
                severity="high",
                value=1.0,
                display_value="深睡",
                trend="holding",
                source_domain="circadian_rhythm",
                icon="💤",
            )
        )
    elif str(circadian.get("phase", "")) == "light_sleep":
        signals.append(
            _signal_card(
                key="light_sleep",
                label="浅睡中",
                family="circadian",
                severity="medium",
                value=1.0,
                display_value="浅睡",
                trend="holding",
                source_domain="circadian_rhythm",
                icon="😴",
            )
        )
    elif str(circadian.get("phase", "")) == "dawn_recover":
        signals.append(
            _signal_card(
                key="dawn_recovery",
                label="黎明恢复",
                family="circadian",
                severity="info",
                value=_safe_float(circadian.get("dawn_recovery_progress", 0.0)),
                display_value=_format_percent(_safe_float(circadian.get("dawn_recovery_progress", 0.0))),
                trend="recovering",
                source_domain="circadian_rhythm",
                icon="🌅",
            )
        )

    drowsiness = _safe_float(circadian.get("drowsiness_value", 0.0))
    if drowsiness >= circadian_cfg["drowsiness_show"]:
        signals.append(
            _signal_card(
                key="drowsiness_value",
                label="困意过高",
                family="circadian",
                severity="medium" if drowsiness < circadian_cfg["drowsiness_high"] else "high",
                value=drowsiness,
                display_value=f"{drowsiness:.0f}",
                trend="rising",
                source_domain="circadian_rhythm",
                icon="🌙",
            )
        )

    sleep_debt = _safe_float(circadian.get("sleep_debt", 0.0))
    if sleep_debt >= circadian_cfg["sleep_debt_show"]:
        signals.append(
            _signal_card(
                key="sleep_debt",
                label="睡眠债偏高",
                family="circadian",
                severity="medium" if sleep_debt < circadian_cfg["sleep_debt_high"] else "high",
                value=sleep_debt,
                display_value=_format_percent(sleep_debt),
                trend="rising",
                source_domain="circadian_rhythm",
                icon="🛌",
            )
        )

    sleep_reserve = _safe_float(circadian.get("sleep_reserve", 100.0), 100.0)
    if sleep_reserve <= circadian_cfg["sleep_reserve_low"]:
        signals.append(
            _signal_card(
                key="sleep_reserve",
                label="睡眠储备偏低",
                family="circadian",
                severity="high",
                value=sleep_reserve,
                display_value=f"{sleep_reserve:.0f}",
                trend="falling",
                source_domain="circadian_rhythm",
                icon="🔋",
            )
        )

    if bool(circadian.get("peek_window_open", False)):
        signals.append(
            _signal_card(
                key="peek_window_open",
                label="浅睡窥屏窗口",
                family="circadian",
                severity="info",
                value=1.0,
                display_value="打开",
                trend="active",
                source_domain="circadian_rhythm",
                icon="🫣",
            )
        )

    mood_bias = _safe_float(emergence.get("mood_bias", 0.5), 0.5)
    if mood_bias <= emotion_cfg["mood_low"]:
        signals.append(
            _signal_card(
                key="mood_bias",
                label="情绪低落",
                family="emotion",
                severity="medium",
                value=mood_bias,
                display_value=_format_percent(mood_bias),
                trend="falling",
                source_domain="emergence_core",
                icon="🌧️",
            )
        )

    curiosity = _safe_float(emergence.get("curiosity_drive", 0.0))
    if curiosity >= emotion_cfg["curiosity_show"]:
        signals.append(
            _signal_card(
                key="curiosity_drive",
                label="好奇心升高",
                family="emotion",
                severity="info",
                value=curiosity,
                display_value=_format_percent(curiosity),
                trend="rising",
                source_domain="emergence_core",
                icon="✨",
            )
        )

    social_desire = _safe_float(emergence.get("social_desire", 0.0))
    if social_desire >= emotion_cfg["social_desire_high"] or social_desire <= emotion_cfg["social_desire_low"]:
        signals.append(
            _signal_card(
                key="social_desire",
                label="社交欲",
                family="emotion",
                severity="info" if social_desire >= emotion_cfg["social_desire_high"] else "low",
                value=social_desire,
                display_value=_format_percent(social_desire),
                trend="rising" if social_desire >= emotion_cfg["social_desire_high"] else "falling",
                source_domain="emergence_core",
                icon="💬" if social_desire >= emotion_cfg["social_desire_high"] else "🤫",
            )
        )

    return signals


def _build_timeline(
    domains: Dict[str, Any],
    dashboard_snapshot: Dict[str, Any],
    active_signals: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    execution_runtime = domains.get("execution_runtime", {})
    timing_gate = domains.get("timing_gate", {}) if isinstance(domains.get("timing_gate"), dict) else {}
    gate_history = timing_gate.get("history", []) if isinstance(timing_gate.get("history"), list) else []
    change_events = dashboard_snapshot.get("change_events", []) if isinstance(dashboard_snapshot, dict) else []
    generated = []
    for event in change_events[-6:]:
        generated.append(
            {
                "at": event.get("timestamp", time.time()),
                "label": str(event.get("type", "state_change") or "state_change"),
                "detail": str(event.get("detail", "") or event.get("summary", "") or "状态发生变化"),
                "family": "dashboard",
            }
        )

    for gate in gate_history[-8:]:
        if not isinstance(gate, dict):
            continue
        stage = str(gate.get("stage", "") or "")
        result = str(gate.get("gate_result", "") or "observe")
        generated.append(
            {
                "at": _safe_float(gate.get("at", time.time()), time.time()),
                "label": f"Timing Gate · {_label_timing_gate_result(result)}",
                "detail": f"{_label_execution_stage(stage)}：{str(gate.get('reason', '') or '暂无门控原因')}",
                "family": "门控",
            }
        )

    if bool(execution_runtime.get("verdict_id")):
        generated.append(
            {
                "at": time.time(),
                "label": "最终执行裁定",
                "detail": str(
                    execution_runtime.get("execution_reason", "")
                    or execution_runtime.get("blocker", "")
                    or "本轮已有执行裁定"
                ),
                "family": "执行",
            }
        )
    for signal in active_signals[:8]:
        generated.append(
            {
                "at": time.time(),
                "label": signal["label"],
                "detail": f"{signal['label']} 当前处于 {signal['display_value']}",
                "family": signal.get("family_label", signal["family"]),
            }
        )
    if not generated:
        scene_heat = _label_scene_heat(domains.get("group_climate", {}).get("scene_heat", "normal"))
        generated.append(
            {
                "at": time.time(),
                "label": "场景热度",
                "detail": f"当前会话热度为 {scene_heat}",
                "family": "会话",
            }
        )
    return generated


def _build_resource_detail(domains: Dict[str, Any]) -> Dict[str, Any]:
    resource = domains.get("resource_ledger", {})
    chat_capacity = max(1.0, _safe_float(resource.get("chat_capacity", 100.0), 100.0))
    thinking_capacity = max(1.0, _safe_float(resource.get("thinking_capacity", 100.0), 100.0))
    chat_reserve = _safe_float(resource.get("chat_reserve", 0.0))
    thinking_reserve = _safe_float(resource.get("thinking_reserve", 0.0))
    return {
        "energy_reserve_ratio": round(_safe_float(resource.get("energy_reserve_ratio", 0.0)), 3),
        "energy_phase": str(resource.get("energy_phase", "unknown") or "unknown"),
        "energy_phase_label": _label_energy_phase(resource.get("energy_phase", "unknown")),
        "chat_reserve": round(chat_reserve, 2),
        "chat_capacity": round(chat_capacity, 2),
        "chat_percent": round(chat_reserve / chat_capacity, 3),
        "thinking_reserve": round(thinking_reserve, 2),
        "thinking_capacity": round(thinking_capacity, 2),
        "thinking_percent": round(thinking_reserve / thinking_capacity, 3),
        "activity_index": round(_safe_float(resource.get("activity_index", 0.0)), 2),
        "irritation_load": round(_safe_float(resource.get("irritation_load", 0.0)), 2),
        "social_field_score": round(_safe_float(resource.get("social_field_score", 0.0)), 2),
        "last_recovery_source": str(resource.get("last_recovery_source", "") or ""),
        "last_irritation_relief_source": str(resource.get("last_irritation_relief_source", "") or ""),
    }


def _build_circadian_detail(domains: Dict[str, Any]) -> Dict[str, Any]:
    circadian = domains.get("circadian_rhythm", {})
    sleep_used = int(_safe_float(circadian.get("sleep_reply_used", 0), 0))
    sleep_cap = int(_safe_float(circadian.get("sleep_reply_cap", 0), 0))
    remaining = int(_safe_float(circadian.get("remaining_sleep_replies", 0), 0))
    current_hour = int(_safe_float(circadian.get("current_hour", time.localtime().tm_hour), time.localtime().tm_hour)) % 24
    time_band = str(circadian.get("time_band", "") or "")
    time_band_label = str(circadian.get("time_band_label", "") or "")
    time_band_description = str(circadian.get("time_band_description", "") or "")
    if not time_band:
        try:
            from src.core.night_cycle_system import resolve_system_time_band

            timeline = resolve_system_time_band(current_hour)
            time_band = str(timeline.get("key", "") or "daytime")
            time_band_label = str(timeline.get("label", "") or "白天")
            time_band_description = str(timeline.get("description", "") or "系统时间线")
        except Exception:
            time_band, time_band_label, time_band_description = "daytime", "白天", "系统时间线"
    pressure_breakdown = circadian.get("pressure_breakdown", {})
    if not isinstance(pressure_breakdown, dict):
        pressure_breakdown = {}
    mechanism_windows = circadian.get("mechanism_windows", {})
    if not isinstance(mechanism_windows, dict):
        mechanism_windows = {}
    phase = str(circadian.get("phase", "awake") or "awake")
    phase_label = str(circadian.get("phase_label", "清醒") or "清醒")
    phase_label = {
        "night_active": "晚上阶段",
        "midnight_reflect": "凌晨阶段",
        "quiet_contemplate": "凌晨沉思",
        "social_night": "夜间社交",
        "dawn_recover": "清晨恢复",
        "burned_out": "熬穿",
    }.get(phase, phase_label)
    return {
        "phase": phase,
        "phase_label": phase_label,
        "time_band": time_band,
        "time_band_label": time_band_label,
        "time_band_description": time_band_description,
        "current_hour": current_hour,
        "is_night": bool(circadian.get("is_night", False)),
        "is_sleep_window": bool(circadian.get("is_sleep_window", False)),
        "is_night_social_window": bool(circadian.get("is_night_social_window", False)),
        "is_pressure_window": bool(circadian.get("is_pressure_window", False)),
        "mechanism_windows": mechanism_windows,
        "system_started_at": round(_safe_float(circadian.get("system_started_at", 0.0)), 3),
        "last_evaluated_at": round(_safe_float(circadian.get("last_evaluated_at", 0.0)), 3),
        "sync_label": str(circadian.get("sync_label", "") or "运行时实时计算"),
        "sync_source": str(circadian.get("sync_source", "") or "state_monitor"),
        "is_sleeping": bool(circadian.get("is_sleeping", False)),
        "is_burnthrough": bool(circadian.get("is_burnthrough", False)),
        "can_reply": bool(circadian.get("can_reply", True)),
        "drowsiness_value": round(_safe_float(circadian.get("drowsiness_value", 0.0)), 2),
        "sleep_debt": round(_safe_float(circadian.get("sleep_debt", 0.0)), 3),
        "overnight_pressure": round(_safe_float(circadian.get("overnight_pressure", 0.0)), 2),
        "sleep_reserve": round(_safe_float(circadian.get("sleep_reserve", 100.0), 100.0), 2),
        "dawn_recovery_progress": round(_safe_float(circadian.get("dawn_recovery_progress", 0.0)), 3),
        "peek_window_open": bool(circadian.get("peek_window_open", False)),
        "sleep_reply_used": sleep_used,
        "sleep_reply_cap": sleep_cap,
        "remaining_sleep_replies": remaining,
        "reply_quota_label": f"{remaining}/{sleep_cap}" if sleep_cap > 0 else "不限",
        "response_suppression": round(_safe_float(circadian.get("response_suppression", 0.0)), 3),
        "pressure_breakdown": {
            "total": round(_safe_float(pressure_breakdown.get("total", 0.0)), 4),
            "chat_minutes": round(_safe_float(pressure_breakdown.get("chat_minutes", 0.0)), 2),
            "peek_minutes": round(_safe_float(pressure_breakdown.get("peek_minutes", 0.0)), 2),
            "think_intensity": round(_safe_float(pressure_breakdown.get("think_intensity", 0.0)), 3),
            "interrupt_count": int(_safe_float(pressure_breakdown.get("interrupt_count", 0), 0)),
            "chat_component": round(_safe_float(pressure_breakdown.get("chat_component", 0.0)), 4),
            "peek_component": round(_safe_float(pressure_breakdown.get("peek_component", 0.0)), 4),
            "think_component": round(_safe_float(pressure_breakdown.get("think_component", 0.0)), 4),
            "interrupt_component": round(_safe_float(pressure_breakdown.get("interrupt_component", 0.0)), 4),
        },
        "body_state_label": str(circadian.get("body_state_label", "正常") or "正常"),
        "mood_hint": str(circadian.get("mood_hint", "") or "状态平稳"),
        "expression_style_label": _label_expression_style(
            circadian.get("expression_style_label") or circadian.get("expression_style") or "normal"
        ),
    }


def _build_emotion_detail(domains: Dict[str, Any]) -> Dict[str, Any]:
    emergence = domains.get("emergence_core", {})
    emotion_cfg = _state_monitor_thresholds()["emotion"]
    mood = _safe_float(emergence.get("mood_bias", 0.5), 0.5)
    if mood >= emotion_cfg["mood_happy"]:
        mood_label = "愉快"
    elif mood >= emotion_cfg["mood_positive"]:
        mood_label = "平稳偏好"
    elif mood >= emotion_cfg["mood_calm"]:
        mood_label = "平静"
    elif mood >= emotion_cfg["mood_low_label"]:
        mood_label = "低落"
    else:
        mood_label = "明显低落"
    feeling = str(emergence.get("feeling_text", "") or "").strip()
    return {
        "mood_bias": round(mood, 3),
        "mood_label": mood_label,
        "curiosity_drive": round(_safe_float(emergence.get("curiosity_drive", 0.0)), 3),
        "social_desire": round(_safe_float(emergence.get("social_desire", 0.0)), 3),
        "boredom_load": round(_safe_float(emergence.get("boredom_load", 0.0)), 3),
        "loneliness_load": round(_safe_float(emergence.get("loneliness_load", 0.0)), 3),
        "environment_fatigue_load": round(_safe_float(emergence.get("environment_fatigue_load", 0.0)), 3),
        "initiative_drive": round(_safe_float(emergence.get("initiative_drive", 0.0)), 3),
        "withdrawal_drive": round(_safe_float(emergence.get("withdrawal_drive", 0.0)), 3),
        "energy_ratio": round(_safe_float(emergence.get("energy_ratio", 0.5), 0.5), 3),
        "silence_seconds": round(_safe_float(emergence.get("silence_seconds", 0.0)), 2),
        "unanswered_count": int(_safe_float(emergence.get("unanswered_count", 0), 0)),
        "feeling_text": feeling or "当前没有明显情绪波动",
    }


def _build_behavior_detail(
    chat: Any,
    domains: Dict[str, Any],
    dashboard_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    flow_runtime = domains.get("flow_runtime", {})
    execution_runtime = domains.get("execution_runtime", {})
    reply_decision = {}
    if isinstance(dashboard_snapshot, dict) and dashboard_snapshot.get("available"):
        snapshot = dashboard_snapshot.get("snapshot", {})
        if isinstance(snapshot, dict):
            reply_decision = snapshot.get("reply_decision", {}) or {}

    behavior = getattr(chat, "_last_behavior_governor_verdict", None) if chat is not None else None
    rest = getattr(chat, "_last_rest_governor_verdict", None) if chat is not None else None
    model = getattr(chat, "_last_model_governor_verdict", None) if chat is not None else None

    behavior_codes = [str(code).strip() for code in list(getattr(behavior, "reason_codes", []) or []) if str(code).strip()]
    rest_codes = [str(code).strip() for code in list(getattr(rest, "reason_codes", []) or []) if str(code).strip()]
    model_codes = [str(code).strip() for code in list(getattr(model, "upgrade_reason_codes", []) or []) if str(code).strip()]

    return {
        "watch_state": str(flow_runtime.get("watch_state", "") or ""),
        "watch_state_label": str(flow_runtime.get("watch_state_label", "瞥一眼") or "瞥一眼"),
        "flow_phase_label": str(flow_runtime.get("phase_label", "待命") or "待命"),
        "reply_decision": {
            "reply_urgency": str(reply_decision.get("reply_urgency", "可稍后回") or "可稍后回"),
            "should_reply": bool(reply_decision.get("reply", False)),
            "decision_reason": str(reply_decision.get("decision_reason", "") or "暂无裁定理由"),
            "detail_notes": [str(item) for item in list(reply_decision.get("detail_notes", []) or []) if str(item).strip()],
            "suggested_tone": str(reply_decision.get("suggested_tone", "") or "正常回应"),
            "decision_stage": str(reply_decision.get("decision_stage", "") or ""),
            "confidence": round(_safe_float(reply_decision.get("confidence", 0.0)), 2),
        },
        "execution_runtime": {
            "should_act": bool(execution_runtime.get("should_act", False)),
            "reply_sent": bool(execution_runtime.get("reply_sent", False)),
            "final_action": str(execution_runtime.get("final_action", "") or ""),
            "final_action_label": _label_final_action(execution_runtime.get("final_action", "")),
            "execution_stage": str(execution_runtime.get("execution_stage", "") or ""),
            "execution_stage_label": _label_execution_stage(execution_runtime.get("execution_stage", "")),
            "execution_reason": str(execution_runtime.get("execution_reason", "") or "暂无最终执行原因"),
            "confidence": round(_safe_float(execution_runtime.get("confidence", 0.0)), 2),
            "model_path": str(execution_runtime.get("model_path", "") or ""),
            "source": str(execution_runtime.get("source", "") or ""),
            "planner_action": str(execution_runtime.get("planner_action", "") or ""),
            "blocker": str(execution_runtime.get("blocker", "") or ""),
            "blocking_factors": [
                str(item) for item in list(execution_runtime.get("blocking_factors", []) or []) if str(item).strip()
            ],
            "driving_factors": [
                str(item) for item in list(execution_runtime.get("driving_factors", []) or []) if str(item).strip()
            ],
        },
        "behavior_governor": {
            "reply_mode": str(getattr(behavior, "reply_mode", "observe") or "observe"),
            "reply_mode_label": _label_reply_mode(getattr(behavior, "reply_mode", "observe")),
            "interrupt_level": str(getattr(behavior, "interrupt_level", "ignore") or "ignore"),
            "interrupt_level_label": _label_interrupt_level(getattr(behavior, "interrupt_level", "ignore")),
            "quote_policy": str(getattr(behavior, "quote_policy", "none") or "none"),
            "quote_policy_label": _label_quote_policy(getattr(behavior, "quote_policy", "none")),
            "silence_policy": str(getattr(behavior, "silence_policy", "silent") or "silent"),
            "silence_policy_label": _label_silence_policy(getattr(behavior, "silence_policy", "silent")),
            "allow_generation": bool(getattr(behavior, "allow_generation", True)),
            "model_tier": str(getattr(behavior, "model_tier", "small") or "small"),
            "model_tier_label": _label_model_tier(getattr(behavior, "model_tier", "small")),
            "reason_codes": behavior_codes,
            "reason_labels": [_label_behavior_reason(code) for code in behavior_codes],
        },
        "rest_governor": {
            "posture": str(getattr(rest, "posture", "active") or "active"),
            "posture_label": _label_rest_posture(getattr(rest, "posture", "active")),
            "interruption_policy": str(getattr(rest, "interruption_policy", "allow") or "allow"),
            "interruption_policy_label": _label_rest_interrupt(getattr(rest, "interruption_policy", "allow")),
            "should_rest": bool(getattr(rest, "should_rest", False)),
            "should_loaf": bool(getattr(rest, "should_loaf", False)),
            "reason_codes": rest_codes,
            "reason_labels": [_label_behavior_reason(code) for code in rest_codes],
        },
        "model_governor": {
            "tier": str(getattr(model, "tier", "skip") or "skip"),
            "tier_label": _label_model_tier(getattr(model, "tier", "skip")),
            "rate_limited": bool(getattr(model, "rate_limited", False)),
            "fallback_to_small": bool(getattr(model, "fallback_to_small", False)),
            "dynamic_cooldown_sec": round(_safe_float(getattr(model, "dynamic_cooldown_sec", 0.0)), 1),
            "dynamic_hourly_cap": int(_safe_float(getattr(model, "dynamic_hourly_cap", 0), 0)),
            "reason_codes": model_codes,
            "reason_labels": [_label_behavior_reason(code) for code in model_codes],
        },
    }


def _build_timing_gate_detail(domains: Dict[str, Any]) -> Dict[str, Any]:
    timing_gate = domains.get("timing_gate", {})
    if not isinstance(timing_gate, dict):
        timing_gate = {}
    current = timing_gate.get("current", {}) if isinstance(timing_gate.get("current"), dict) else {}
    history_raw = timing_gate.get("history", []) if isinstance(timing_gate.get("history"), list) else []
    history = []
    for item in history_raw[-20:]:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage", "") or "")
        result = str(item.get("gate_result", "") or "observe")
        history.append(
            {
                "verdict_id": str(item.get("verdict_id", "") or ""),
                "at": _safe_float(item.get("at", 0.0)),
                "gate_result": result,
                "gate_result_label": str(item.get("gate_result_label", "") or _label_timing_gate_result(result)),
                "stage": stage,
                "stage_label": _label_execution_stage(stage),
                "reason": str(item.get("reason", "") or "暂无门控原因"),
                "source": str(item.get("source", "") or "timing_gate"),
                "final_action": str(item.get("final_action", "") or "observe"),
                "final_action_label": _label_final_action(item.get("final_action", "")),
                "next_action": str(item.get("next_action", "") or ""),
                "model_path": str(item.get("model_path", "") or "skip"),
                "blocker": str(item.get("blocker", "") or ""),
                "confidence": round(_safe_float(item.get("confidence", 0.0)), 3),
            }
        )
    current_stage = str(current.get("stage", "") or "")
    current_result = str(current.get("gate_result", "") or "observe")
    return {
        "current": {
            "verdict_id": str(current.get("verdict_id", "") or ""),
            "at": _safe_float(current.get("at", 0.0)),
            "gate_result": current_result,
            "gate_result_label": str(current.get("gate_result_label", "") or _label_timing_gate_result(current_result)),
            "stage": current_stage,
            "stage_label": _label_execution_stage(current_stage),
            "reason": str(current.get("reason", "") or "暂无门控原因"),
            "source": str(current.get("source", "") or "timing_gate"),
            "final_action": str(current.get("final_action", "") or "observe"),
            "final_action_label": _label_final_action(current.get("final_action", "")),
            "next_action": str(current.get("next_action", "") or ""),
            "model_path": str(current.get("model_path", "") or "skip"),
            "blocker": str(current.get("blocker", "") or ""),
            "confidence": round(_safe_float(current.get("confidence", 0.0)), 3),
        },
        "history": history,
        "history_count": len(history),
    }


def _build_memory_detail(domains: Dict[str, Any]) -> Dict[str, Any]:
    memory_stack = domains.get("memory_stack", {})
    governance = memory_stack.get("governance", {}) if isinstance(memory_stack.get("governance"), dict) else {}
    health = memory_stack.get("health", {}) if isinstance(memory_stack.get("health"), dict) else {}
    overload = memory_stack.get("overload", {}) if isinstance(memory_stack.get("overload"), dict) else {}
    reactivation = memory_stack.get("reactivation", {}) if isinstance(memory_stack.get("reactivation"), dict) else {}
    return {
        "ephemeral_short_term_count": int(_safe_float(memory_stack.get("ephemeral_short_term_count", 0), 0)),
        "planner_short_term_count": int(_safe_float(memory_stack.get("planner_short_term_count", 0), 0)),
        "planner_long_term_count": int(_safe_float(memory_stack.get("planner_long_term_count", 0), 0)),
        "long_term_total": int(_safe_float(memory_stack.get("long_term_total", 0), 0)),
        "long_term_by_category": dict(memory_stack.get("long_term_by_category", {}) or {}),
        "memoir_phase": str(memory_stack.get("memoir_phase", "open") or "open"),
        "memoir_phase_label": _label_memoir_phase(memory_stack.get("memoir_phase", "open")),
        "memoir_exchange_tally": int(_safe_float(memory_stack.get("memoir_exchange_tally", 0), 0)),
        "memoir_consecutive_timeouts": int(_safe_float(memory_stack.get("memoir_consecutive_timeouts", 0), 0)),
        "memoir_last_topic": str(memory_stack.get("memoir_last_topic", "") or ""),
        "memoir_last_mood": str(memory_stack.get("memoir_last_mood", "") or ""),
        "journal_count": int(_safe_float(memory_stack.get("journal_count", 0), 0)),
        "governance_total": int(_safe_float(governance.get("total", 0), 0)),
        "governance_utilization": round(_safe_float(governance.get("utilization", 0.0)), 3),
        "governance_pending_evict": int(_safe_float(governance.get("pending_evict", 0), 0)),
        "governance_last_op": str(governance.get("last_op", "") or "无操作"),
        "health_score": int(_safe_float(health.get("health_score", 100), 100)),
        "health_grade": str(health.get("health_grade", "优秀") or "优秀"),
        "overload_gauge_level": str(overload.get("gauge_level", "") or ""),
        "overload_load_ratio": round(_safe_float(overload.get("load_ratio", 0.0)), 3),
        "overload_amnesia_pressure": round(_safe_float(overload.get("amnesia_pressure", 0.0)), 3),
        "overload_emergency_needed": bool(overload.get("emergency_needed", False)),
        "knowledge_entry_count": int(_safe_float(memory_stack.get("knowledge_entry_count", 0), 0)),
        "reactivation_summary": dict(reactivation),
    }


def _build_autonomy_detail(domains: Dict[str, Any]) -> Dict[str, Any]:
    autonomy = domains.get("autonomy_runtime", {})
    background_event = autonomy.get("background_event", {}) if isinstance(autonomy.get("background_event"), dict) else {}
    return {
        "active_intentions": list(autonomy.get("active_intentions", []) or []),
        "intention_alive_count": int(_safe_float(autonomy.get("intention_alive_count", 0), 0)),
        "intention_drive": round(_safe_float(autonomy.get("intention_drive", 0.0)), 3),
        "max_intention_urgency": round(_safe_float(autonomy.get("max_intention_urgency", 0.0)), 3),
        "reward_score": round(_safe_float(autonomy.get("reward_score", 0.0)), 3),
        "pending_proactive_events": int(_safe_float(autonomy.get("pending_proactive_events", 0), 0)),
        "self_recent_messages_count": int(_safe_float(autonomy.get("self_recent_messages_count", 0), 0)),
        "self_recent_actions_count": int(_safe_float(autonomy.get("self_recent_actions_count", 0), 0)),
        "self_recent_events_count": int(_safe_float(autonomy.get("self_recent_events_count", 0), 0)),
        "background_event_type": str(background_event.get("type", "") or ""),
        "background_event_label": _label_background_event(background_event.get("type", "")),
        "background_event_priority": round(_safe_float(background_event.get("priority", 0.0)), 3),
        "background_budget_remaining": int(_safe_float(background_event.get("budget_remaining", 0), 0)),
    }


def _build_context_detail(domains: Dict[str, Any]) -> Dict[str, Any]:
    context = domains.get("context_awareness", {})
    if not isinstance(context, dict):
        context = {}
    attention = domains.get("attention_runtime", {})
    if not isinstance(attention, dict):
        attention = {}
    return {
        "direct_target": bool(context.get("direct_target", False)),
        "quote_anchor": bool(context.get("quote_anchor", False)),
        "recent_human_activity": bool(context.get("recent_human_activity", False)),
        "scene_suitable": bool(context.get("scene_suitable", True)),
        "watch_state": str(context.get("watch_state", "") or ""),
        "watch_state_label": str(context.get("watch_state_label", "瞥一眼") or "瞥一眼"),
        "attention_level": int(_safe_float(context.get("attention_level", 0), 0)),
        "perception_engagement_pull": round(_safe_float(context.get("perception_engagement_pull", 0.0)), 3),
        "self_recent_messages_count": int(_safe_float(context.get("self_recent_messages_count", 0), 0)),
        "self_recent_actions_count": int(_safe_float(context.get("self_recent_actions_count", 0), 0)),
        "self_recent_events_count": int(_safe_float(context.get("self_recent_events_count", 0), 0)),
        "memoir_phase": str(context.get("memoir_phase", "open") or "open"),
        "memoir_phase_label": _label_memoir_phase(context.get("memoir_phase", "open")),
        "memoir_consecutive_timeouts": int(_safe_float(context.get("memoir_consecutive_timeouts", 0), 0)),
        "current_target_user_id": str(context.get("current_target_user_id", "") or ""),
        "reply_mode": str(context.get("reply_mode", "observe") or "observe"),
        "reply_mode_label": _label_reply_mode(context.get("reply_mode", "observe")),
        "interrupt_level": str(context.get("interrupt_level", "ignore") or "ignore"),
        "interrupt_level_label": _label_interrupt_level(context.get("interrupt_level", "ignore")),
        "silence_policy": str(context.get("silence_policy", "silent") or "silent"),
        "silence_policy_label": _label_silence_policy(context.get("silence_policy", "silent")),
        "behavior_reason_labels": [_label_behavior_reason(code) for code in list(context.get("behavior_reason_codes", []) or [])],
        "rest_reason_labels": [_label_behavior_reason(code) for code in list(context.get("rest_reason_codes", []) or [])],
        "visibility_threshold": round(_safe_float(attention.get("visibility_threshold", 0.0)), 3),
        "process_ratio": round(_safe_float(attention.get("process_ratio", 0.0)), 3),
        "peek_desire": round(_safe_float(attention.get("peek_desire", 0.0)), 3),
        "interrupt_tolerance": round(_safe_float(attention.get("interrupt_tolerance", 0.0)), 3),
        "silence_tolerance": round(_safe_float(attention.get("silence_tolerance", 0.0)), 3),
        "time_since_last_look": round(_safe_float(attention.get("time_since_last_look", 0.0)), 2),
        "consecutive_peeks_without_action": int(_safe_float(attention.get("consecutive_peeks_without_action", 0), 0)),
        "look_budget_state": round(_safe_float(attention.get("look_budget_state", 0.0)), 3),
        "process_budget_state": round(_safe_float(attention.get("process_budget_state", 0.0)), 3),
    }


def _derive_current_user_mood_hint(relationship: Dict[str, Any]) -> str:
    relationship_cfg = _state_monitor_thresholds()["relationship"]
    irritation = _safe_float(relationship.get("irritation_load", 0.0))
    pressure = _safe_float(relationship.get("pressure_load", 0.0))
    trauma = _safe_float(relationship.get("trauma_load", 0.0))
    rapport = _safe_float(relationship.get("rapport_score", 0.0))
    trust = _safe_float(relationship.get("trust_score", 0.0))
    if irritation >= relationship_cfg["current_user_irritation_high"]:
        return "明显不耐烦"
    if (
        pressure >= relationship_cfg["current_user_pressure_high"]
        or trauma >= relationship_cfg["current_user_trauma_high"]
    ):
        return "防御绷紧"
    if (
        rapport >= relationship_cfg["current_user_rapport_warm"]
        and trust >= relationship_cfg["current_user_trust_warm"]
    ):
        return "放松熟络"
    if (
        rapport >= relationship_cfg["current_user_rapport_steady"]
        or trust >= relationship_cfg["current_user_trust_steady"]
    ):
        return "平稳接触"
    return "谨慎观察"


def _build_group_state_detail(domains: Dict[str, Any]) -> Dict[str, Any]:
    group_climate = domains.get("group_climate", {})
    return {
        "scene_heat": str(group_climate.get("scene_heat", "normal") or "normal"),
        "scene_heat_label": _label_scene_heat(group_climate.get("scene_heat", "normal")),
        "scene_heat_score": round(_safe_float(group_climate.get("scene_heat_score", 0.0)), 3),
        "messages_per_minute": round(_safe_float(group_climate.get("messages_per_minute", 0.0)), 3),
        "active_user_count": int(_safe_float(group_climate.get("active_user_count", 0), 0)),
        "participant_diversity": round(_safe_float(group_climate.get("participant_diversity", 0.0)), 3),
        "interaction_quality": round(_safe_float(group_climate.get("interaction_quality", 0.0)), 3),
        "complexity_level": round(_safe_float(group_climate.get("complexity_level", 0.0)), 3),
        "social_density": round(_safe_float(group_climate.get("social_density", 0.0)), 3),
        "suitable_to_join": bool(group_climate.get("suitable_to_join", True)),
        "join_unsuitable_reason": str(group_climate.get("join_unsuitable_reason", "") or ""),
        "dominant_speaker": str(group_climate.get("dominant_speaker", "") or ""),
        "active_topic_count": int(_safe_float(group_climate.get("active_topic_count", 0), 0)),
        "topic_focus": list(group_climate.get("topic_focus", []) or []),
        "thread_count": int(_safe_float(group_climate.get("thread_count", 0), 0)),
        "session_phase": str(group_climate.get("session_phase", "") or ""),
        "vexation": round(_safe_float(group_climate.get("vexation", 0.0)), 3),
        "weariness": round(_safe_float(group_climate.get("weariness", 0.0)), 3),
        "vitality": round(_safe_float(group_climate.get("vitality", 100.0), 100.0), 3),
        "mood_category": str(group_climate.get("mood_category", "") or ""),
        "atmosphere": str(group_climate.get("atmosphere", "") or ""),
        "atmosphere_label": str(group_climate.get("atmosphere_label", "") or ""),
        "hot_count": int(_safe_float(group_climate.get("hot_count", 0), 0)),
        "warm_count": int(_safe_float(group_climate.get("warm_count", 0), 0)),
    }


def _build_attention_runtime_detail(domains: Dict[str, Any]) -> Dict[str, Any]:
    attention = domains.get("attention_runtime", {})
    if not isinstance(attention, dict):
        attention = {}
    return {
        "state": str(attention.get("state", "") or ""),
        "state_label": str(attention.get("state_label", "") or "扫描模式"),
        "visibility_threshold": round(_safe_float(attention.get("visibility_threshold", 0.0)), 3),
        "process_ratio": round(_safe_float(attention.get("process_ratio", 0.0)), 3),
        "peek_desire": round(_safe_float(attention.get("peek_desire", 0.0)), 3),
        "interrupt_tolerance": round(_safe_float(attention.get("interrupt_tolerance", 0.0)), 3),
        "silence_tolerance": round(_safe_float(attention.get("silence_tolerance", 0.0)), 3),
        "time_since_last_look": round(_safe_float(attention.get("time_since_last_look", 0.0)), 2),
        "consecutive_peeks_without_action": int(_safe_float(attention.get("consecutive_peeks_without_action", 0), 0)),
        "look_budget_state": round(_safe_float(attention.get("look_budget_state", 0.0)), 3),
        "process_budget_state": round(_safe_float(attention.get("process_budget_state", 0.0)), 3),
        "openness": round(_safe_float(attention.get("openness", 0.0)), 3),
    }


def _build_safety_detail(domains: Dict[str, Any]) -> Dict[str, Any]:
    safety = domains.get("safety_runtime", {})
    if not isinstance(safety, dict):
        safety = {}
    level = safety.get("level", safety.get("overall_level_label", safety.get("safety_level", "安全")))
    score = safety.get("score", safety.get("overall_score", 0.0))
    blocked = safety.get("blocked") if "blocked" in safety else safety.get("block_reply", False)
    dominant = safety.get("dominant_threat", safety.get("dominant", safety.get("threat", "")))
    bar_penalty = safety.get("bar_penalty", safety.get("bar_delta", 0.0))
    evidence = safety.get("threat_evidence_summary", safety.get("action", dominant))
    return {
        "safety_level": str(level or "安全"),
        "safety_score": round(_safe_float(score), 4),
        "blocked": bool(blocked),
        "dominant_threat": str(dominant or ""),
        "bar_penalty": round(_safe_float(bar_penalty), 4),
        "threat_evidence_summary": str(evidence or "暂无显著风险"),
    }


def _build_current_user_detail(domains: Dict[str, Any]) -> Dict[str, Any]:
    relationship = domains.get("relationship_profile", {})
    return {
        "user_id": str(relationship.get("user_id", "") or ""),
        "display_name": str(relationship.get("display_name", "当前目标") or "当前目标"),
        "relationship_label": str(
            relationship.get("personal_impression", "")
            or relationship.get("relationship_label", "")
            or relationship.get("legacy_relationship_label", "")
            or ""
        ),
        "personal_impression": str(relationship.get("personal_impression", "") or ""),
        "impression_labels": list(relationship.get("impression_labels", []) or []),
        "legacy_relationship_label": str(relationship.get("legacy_relationship_label", "") or ""),
        "rapport_score": round(_safe_float(relationship.get("rapport_score", 0.0)), 2),
        "trust_score": round(_safe_float(relationship.get("trust_score", 0.0)), 2),
        "irritation_load": round(_safe_float(relationship.get("irritation_load", 0.0)), 2),
        "pressure_load": round(_safe_float(relationship.get("pressure_load", 0.0)), 2),
        "trauma_load": round(_safe_float(relationship.get("trauma_load", 0.0)), 2),
        "chaos_load": round(_safe_float(relationship.get("chaos_load", 0.0)), 2),
        "mask_load": round(_safe_float(relationship.get("mask_load", 0.0)), 2),
        "interaction_count": int(_safe_float(relationship.get("interaction_count", 0), 0)),
        "current_mood_hint": _derive_current_user_mood_hint(relationship),
    }


def _build_participant_impacts(
    domains: Dict[str, Any],
    active_signals: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    thresholds = _state_monitor_thresholds()
    relationship_cfg = thresholds["relationship"]
    trauma_cfg = thresholds["trauma"]
    relationship = domains.get("relationship_profile", {})
    relationship_population = domains.get("relationship_population", {})
    raw_participants = []
    if isinstance(relationship_population, dict):
        raw_participants = list(relationship_population.get("participants", []) or [])

    impacts: list[Dict[str, Any]] = []
    focus_user_id = str(relationship.get("user_id", "") or "")
    focus_signal_labels = [
        signal["label"]
        for signal in active_signals
        if signal.get("source_domain") in {"relationship_profile", "trauma_load", "surface_mask"}
    ]

    for participant in raw_participants:
        if not isinstance(participant, dict):
            continue
        user_id = str(participant.get("user_id", "") or "")
        if not user_id:
            continue
        in_current_scene = bool(participant.get("in_current_scene", False))
        recent_speaker = bool(participant.get("recent_speaker", False))
        recent_targeted_interaction = bool(participant.get("recent_targeted_interaction", False))
        source_scope = str(participant.get("source_scope", "recent_history") or "recent_history")
        labels: list[str] = []
        if bool(participant.get("is_current_target", False)):
            labels.append("当前焦点")
        if in_current_scene:
            labels.append("当前会话相关")
        if recent_speaker:
            labels.append("近5分钟发言")
        if recent_targeted_interaction:
            labels.append("近期目标互动")
        if _safe_float(participant.get("irritation_load", 0.0)) >= relationship_cfg["irritation_show"]:
            labels.append("烦躁偏高")
        if _safe_float(participant.get("pressure_load", 0.0)) >= relationship_cfg["pressure_show"]:
            labels.append("压力偏高")
        if _safe_float(participant.get("trauma_load", 0.0)) >= trauma_cfg["trauma_show"]:
            labels.append("创伤警惕")
        if _safe_float(participant.get("chaos_load", 0.0)) >= trauma_cfg["chaos_show"]:
            labels.append("内心混乱")
        if _safe_float(participant.get("mask_load", 0.0)) >= trauma_cfg["mask_show"]:
            labels.append("表层伪装")
        if bool(participant.get("is_blocked", False)):
            labels.append("已阻断")
        if bool(participant.get("is_protected", False)):
            labels.append("保护中")
        if user_id == focus_user_id:
            labels.extend(focus_signal_labels)
        deduped_labels = list(dict.fromkeys(label for label in labels if str(label).strip()))
        impacts.append(
            {
                "user_id": user_id,
                "display_name": str(participant.get("display_name", "当前目标") or "当前目标"),
                "relationship_label": str(
                    participant.get("personal_impression", "")
                    or participant.get("relationship_label", "")
                    or participant.get("legacy_relationship_label", "")
                    or ""
                ),
                "personal_impression": str(participant.get("personal_impression", "") or ""),
                "impression_labels": list(participant.get("impression_labels", []) or []),
                "legacy_relationship_label": str(participant.get("legacy_relationship_label", "") or ""),
                "rapport_score": round(_safe_float(participant.get("rapport_score", 0.0)), 2),
                "trust_score": round(_safe_float(participant.get("trust_score", 0.0)), 2),
                "irritation_load": round(_safe_float(participant.get("irritation_load", 0.0)), 2),
                "trauma_load": round(_safe_float(participant.get("trauma_load", 0.0)), 2),
                "pressure_load": round(_safe_float(participant.get("pressure_load", 0.0)), 2),
                "chaos_load": round(_safe_float(participant.get("chaos_load", 0.0)), 2),
                "mask_load": round(_safe_float(participant.get("mask_load", 0.0)), 2),
                "interaction_count": int(_safe_float(participant.get("interaction_count", 0), 0)),
                "current_mood_hint": str(participant.get("current_mood_hint", "") or "状态平稳"),
                "last_interaction_age_sec": round(_safe_float(participant.get("last_interaction_age_sec", 999999.0)), 1),
                "active_signals": deduped_labels,
                "impact_rank": round(_safe_float(participant.get("impact_rank", 0.0)), 3),
                "is_current_target": bool(participant.get("is_current_target", False)),
                "in_current_scene": in_current_scene,
                "recent_speaker": recent_speaker,
                "recent_targeted_interaction": recent_targeted_interaction,
                "source_scope": source_scope,
            }
        )

    if not impacts and (relationship.get("user_id") or relationship.get("display_name")):
        impacts.append(
            {
                "user_id": relationship.get("user_id", ""),
                "display_name": relationship.get("display_name", "当前目标"),
                "relationship_label": (
                    relationship.get("personal_impression", "")
                    or relationship.get("relationship_label", "")
                    or relationship.get("legacy_relationship_label", "")
                    or ""
                ),
                "personal_impression": relationship.get("personal_impression", ""),
                "impression_labels": list(relationship.get("impression_labels", []) or []),
                "legacy_relationship_label": relationship.get("legacy_relationship_label", ""),
                "rapport_score": relationship.get("rapport_score", 0.0),
                "trust_score": relationship.get("trust_score", 0.0),
                "irritation_load": relationship.get("irritation_load", 0.0),
                "trauma_load": relationship.get("trauma_load", 0.0),
                "pressure_load": relationship.get("pressure_load", 0.0),
                "chaos_load": relationship.get("chaos_load", 0.0),
                "mask_load": relationship.get("mask_load", 0.0),
                "interaction_count": relationship.get("interaction_count", 0),
                "current_mood_hint": _derive_current_user_mood_hint(relationship),
                "last_interaction_age_sec": 999999.0,
                "active_signals": list(dict.fromkeys(focus_signal_labels)),
                "impact_rank": round(
                    _safe_float(relationship.get("rapport_score", 0.0)) * 0.2
                    + _safe_float(relationship.get("trust_score", 0.0)) * 0.15
                    - _safe_float(relationship.get("irritation_load", 0.0)) * 0.1,
                    3,
                ),
                "is_current_target": True,
                "in_current_scene": False,
                "recent_speaker": False,
                "recent_targeted_interaction": True,
                "source_scope": "mentioned_target",
            }
        )

    impacts.sort(
        key=lambda item: (
            0 if item.get("is_current_target") else 1,
            0 if item.get("in_current_scene") else 1,
            0 if item.get("recent_targeted_interaction") else 1,
            -_safe_float(item.get("impact_rank", 0.0)),
            _safe_float(item.get("last_interaction_age_sec", 999999.0)),
        )
    )
    return impacts[:6]


def _trace_item(
    *,
    key: str,
    label: str,
    current: float,
    previous: float,
    reason: str,
) -> Dict[str, Any]:
    delta = current - previous
    if abs(delta) < 0.001:
        trend = "平稳"
    elif delta > 0:
        trend = "上升"
    else:
        trend = "下降"
    return {
        "key": key,
        "label": label,
        "current": round(current, 4),
        "previous": round(previous, 4),
        "delta": round(delta, 4),
        "trend_label": trend,
        "last_change_reason": reason,
    }


def _build_dynamic_trace(
    *,
    channel_id: str,
    domains: Dict[str, Any],
    resource_detail: Dict[str, Any],
    group_state_detail: Dict[str, Any],
    safety_detail: Dict[str, Any],
) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    emergence = domains.get("emergence_core", {})
    current_values = {
        "boredom_load": _safe_float(emergence.get("boredom_load", 0.0)),
        "loneliness_load": _safe_float(emergence.get("loneliness_load", 0.0)),
        "environment_fatigue_load": _safe_float(emergence.get("environment_fatigue_load", 0.0)),
        "initiative_drive": _safe_float(emergence.get("initiative_drive", 0.0)),
        "scene_heat_score": _safe_float(group_state_detail.get("scene_heat_score", 0.0)),
        "safety_score": _safe_float(safety_detail.get("safety_score", 0.0)),
        "chat_percent": _safe_float(resource_detail.get("chat_percent", 0.0)),
        "thinking_percent": _safe_float(resource_detail.get("thinking_percent", 0.0)),
    }
    labels = {
        "boredom_load": "无聊度",
        "loneliness_load": "孤独感",
        "environment_fatigue_load": "疲劳感",
        "initiative_drive": "主动意愿",
        "scene_heat_score": "会话热度",
        "safety_score": "安全风险",
        "chat_percent": "聊天值",
        "thinking_percent": "思考值",
    }
    reasons = {
        "boredom_load": "静默时间、内容丰富度和心情共同影响",
        "loneliness_load": "未回应次数与群聊活跃度共同影响",
        "environment_fatigue_load": "话题重复度、静默时长和精力共同影响",
        "initiative_drive": "内在时钟、情绪驱动、好奇心和疲劳共同影响",
        "scene_heat_score": "会话消息频率、活跃人数、互动质量和氛围共同影响",
        "safety_score": "安全边界融合的威胁信号共同影响",
        "chat_percent": "聊天值的消耗与恢复事件共同影响",
        "thinking_percent": "思考值的消耗与恢复事件共同影响",
    }
    previous = _DYNAMIC_TRACE_CACHE.get(channel_id, {})
    trace = []
    trace_map: Dict[str, Dict[str, Any]] = {}
    for key, value in current_values.items():
        prev_value = previous.get(key, value)
        item = _trace_item(
            key=key,
            label=labels[key],
            current=value,
            previous=prev_value,
            reason=reasons[key],
        )
        trace.append(item)
        trace_map[key] = item
    _DYNAMIC_TRACE_CACHE[channel_id] = current_values
    return trace, trace_map


def _build_initiative_state(domains: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    emergence = domains.get("emergence_core", {})
    emotion_cfg = _state_monitor_thresholds()["emotion"]
    boredom = _safe_float(emergence.get("boredom_load", 0.0))
    loneliness = _safe_float(emergence.get("loneliness_load", 0.0))
    fatigue = _safe_float(emergence.get("environment_fatigue_load", 0.0))
    initiative = _safe_float(emergence.get("initiative_drive", 0.0))
    withdrawal = _safe_float(emergence.get("withdrawal_drive", 0.0))
    if initiative >= emotion_cfg["initiative_high"]:
        label = "主动想聊"
    elif initiative <= emotion_cfg["initiative_low"]:
        label = "主动性偏低"
    else:
        label = "自然观望"
    return {
        "boredom_load": round(boredom, 3),
        "loneliness_load": round(loneliness, 3),
        "environment_fatigue_load": round(fatigue, 3),
        "initiative_drive": round(initiative, 3),
        "withdrawal_drive": round(withdrawal, 3),
        "effect_label": label,
        "chat_timing_label": str(prediction.get("decision_label", "继续观察") or "继续观察"),
        "effect_summary": (
            "无聊度和孤独感会推高主动开口；疲劳感、退场倾向、压力和低资源会压低开口。"
        ),
    }


def _build_display_policy() -> Dict[str, list[str]]:
    return {
        "resident": [
            "精力储备",
            "内在心情",
            "注意状态",
            "社交姿态",
            "安全护盾",
            "流转阶段",
            "场景热度",
            "发言预测",
        ],
        "active": [
            "无聊度/孤独感/疲劳感/退场倾向/主动意愿",
            "情绪低落/好奇心/社交欲显著变化",
            "浅睡/深睡/熬穿/清晨恢复/睡眠债",
            "烦躁/压力/创伤/混乱/伪装",
            "关系好感/信任显著偏高或偏低",
            "冷却窗口/等待时长/重新接入/会话升温",
        ],
        "detail": [
            "运行资源的聊天值和思考值",
            "夜间机制窗口、睡眠债、困意和熬夜压力",
            "情绪状态、主动倾向和当前感受",
            "记忆栈、记忆过载、回忆录和知识条目",
            "主动意图、待结算事件和后台随机事件",
            "会话感知、话题焦点、活跃人数",
            "目标用户关系、好感、信任、压力",
            "发言预测的驱动和抑制因素",
            "Timing Gate 当前裁定和最近门控历史",
        ],
        "hidden": [
            "内部阈值",
            "调试原因",
            "缓存字段",
            "旧命名残留",
            "纯计数器原值",
        ],
    }


def _build_presentation(
    *,
    chat: Any,
    channel_id: str,
    domains: Dict[str, Any],
    dashboard_snapshot: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    prediction_cfg = _state_monitor_thresholds()["prediction"]
    dashboard = dashboard_snapshot.get("snapshot", {}) if dashboard_snapshot.get("available") else {}
    vitality = dashboard.get("energy_reserve", {}) if isinstance(dashboard, dict) else {}
    mood = dashboard.get("inner_mood", {}) if isinstance(dashboard, dict) else {}
    attention = dashboard.get("attention", {}) if isinstance(dashboard, dict) else {}
    social = dashboard.get("social_attitude", {}) if isinstance(dashboard, dict) else {}
    safety = dashboard.get("safety_shield", {}) if isinstance(dashboard, dict) else {}

    flow_runtime = domains.get("flow_runtime", {})
    group_climate = domains.get("group_climate", {})
    circadian_detail = _build_circadian_detail(domains)
    emotion_detail = _build_emotion_detail(domains)
    behavior_detail = _build_behavior_detail(chat, domains, dashboard_snapshot)
    timing_gate_detail = _build_timing_gate_detail(domains)
    memory_detail = _build_memory_detail(domains)
    autonomy_detail = _build_autonomy_detail(domains)
    context_detail = _build_context_detail(domains)
    group_state_detail = _build_group_state_detail(domains)
    current_user_detail = _build_current_user_detail(domains)
    attention_runtime_detail = _build_attention_runtime_detail(domains)
    safety_detail = _build_safety_detail(domains)

    active_signals = _build_active_signals(domains)
    resource_detail = _build_resource_detail(domains)
    initiative_state = _build_initiative_state(domains, prediction)
    dynamic_trace, value_deltas = _build_dynamic_trace(
        channel_id=channel_id,
        domains=domains,
        resource_detail=resource_detail,
        group_state_detail=group_state_detail,
        safety_detail=safety_detail,
    )
    energy_ratio = _safe_float(domains.get("resource_ledger", {}).get("energy_reserve_ratio", 0.0))
    vitality_percent = _safe_float(vitality.get("percent", energy_ratio * 100), energy_ratio * 100)

    resident_overview = {
        "energy_reserve": {
            "label": "精力储备",
            "icon": str(vitality.get("icon", "🔋") or "🔋"),
            "state": str(vitality.get("level") or _label_energy_phase(domains.get("resource_ledger", {}).get("energy_phase"))),
            "value": energy_ratio,
            "display_value": f"{int(vitality_percent)}%",
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
            "state": _label_scene_heat(group_climate.get("scene_heat", "normal")),
            "value": _safe_float(group_climate.get("scene_heat_score", 0.0)),
            "display_value": _label_scene_heat(group_climate.get("scene_heat", "normal")),
            "color": "#f4a261",
        },
        "prediction_readiness": {
            "label": "发言预测",
            "icon": "🔮",
            "state": "高" if _safe_float(prediction.get("speak_probability", 0.0)) >= prediction_cfg["probability_high"] else "中" if _safe_float(prediction.get("speak_probability", 0.0)) >= prediction_cfg["probability_medium"] else "低",
            "value": _safe_float(prediction.get("speak_probability", 0.0)),
            "display_value": f"{int(_safe_float(prediction.get('probability_percent', 0), 0.0))}%",
            "color": "#8d99ae",
        },
    }

    scene_context = {
        "channel_id": channel_id,
        "scene_heat": str(group_climate.get("scene_heat", "normal") or "normal"),
        "scene_heat_label": _label_scene_heat(group_climate.get("scene_heat", "normal")),
        "scene_heat_score": round(_safe_float(group_climate.get("scene_heat_score", 0.0)), 3),
        "messages_per_minute": round(_safe_float(group_climate.get("messages_per_minute", 0.0)), 3),
        "topic_focus": list(group_climate.get("topic_focus", []) or []),
        "active_user_count": int(_safe_float(group_climate.get("active_user_count", 0), 0)),
        "session_phase": str(group_climate.get("session_phase", "") or ""),
        "dominant_speaker": str(group_climate.get("dominant_speaker", "") or ""),
        "suitable_to_join": bool(group_climate.get("suitable_to_join", True)),
        "join_unsuitable_reason": str(group_climate.get("join_unsuitable_reason", "") or ""),
        "thread_count": int(_safe_float(group_climate.get("thread_count", 0), 0)),
        "hot_count": int(_safe_float(group_climate.get("hot_count", 0), 0)),
        "warm_count": int(_safe_float(group_climate.get("warm_count", 0), 0)),
    }

    participant_impacts = _build_participant_impacts(domains, active_signals)

    timeline = _build_timeline(domains, dashboard_snapshot, active_signals)

    return {
        "resident_overview": resident_overview,
        "active_signals": active_signals,
        "resource_detail": resource_detail,
        "circadian_detail": circadian_detail,
        "emotion_detail": emotion_detail,
        "behavior_detail": behavior_detail,
        "timing_gate_detail": timing_gate_detail,
        "context_detail": context_detail,
        "attention_runtime_detail": attention_runtime_detail,
        "safety_detail": safety_detail,
        "group_state_detail": group_state_detail,
        "current_user_detail": current_user_detail,
        "dynamic_trace": dynamic_trace,
        "value_deltas": value_deltas,
        "memory_detail": memory_detail,
        "autonomy_detail": autonomy_detail,
        "initiative_state": initiative_state,
        "scene_context": scene_context,
        "participant_impacts": participant_impacts,
        "timeline": timeline,
        "display_policy": _build_display_policy(),
    }


async def build_monitor_overview() -> Dict[str, Any]:
    overview = list_heartfc_chats()
    now = time.time()
    packet = {
        "snapshot_kind": "overview",
        "updated_at": now,
        "server_time": now,
        "state_version": int(now * 1000),
        "active_count": overview.get("active_count", 0),
        "hidden_internal_count": overview.get("hidden_internal_count", 0),
        "channels": overview.get("channels", []),
    }
    packet = await get_runtime_state_hub().set_snapshot(MONITOR_OVERVIEW_CHANNEL_ID, packet, reason="monitor_overview")
    return packet


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
        chat=chat,
        channel_id=str(channel_id),
        domains=state.get("domains", {}),
        dashboard_snapshot=dashboard,
        prediction=prediction,
    )
    now = time.time()
    packet = {
        "channel_id": str(channel_id),
        "updated_at": now,
        "server_time": now,
        "domains": state.get("domains", {}),
        "presentation": presentation,
        "prediction": prediction,
    }
    packet["state_version"] = int(now * 1000)
    await get_runtime_state_hub().set_snapshot(str(channel_id), packet, reason="monitor_build")
    return packet

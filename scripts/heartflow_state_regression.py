# ruff: noqa: E402
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
REMOVED_STATE_MONITOR_WS_PATH = "/ws/" + "state" + "-monitor"
REMOVED_STATE_MONITOR_ROUTER_NAME = "ws_" + "state_monitor"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat.heart_flow.heartfc_state_exporter import (
    _extract_circadian_rhythm,
    _extract_relationship_profile,
    _extract_safety_runtime,
    list_heartfc_chats,
)
from src.chat.heart_flow.enhanced_modules.proactive_context_prompt_mixin import ProactiveContextPromptMixin
from src.chat.heart_flow.enhanced_modules.proactive_reactive_flow_mixin import ProactiveReactiveFlowMixin
from src.chat.heart_flow.enhanced_modules.scene_planner_bridge_mixin import ScenePlannerBridgeMixin
from src.chat.heart_flow.enhanced_modules.scene_context_analysis_mixin import SceneContextAnalysisMixin
from src.chat.heart_flow.enhanced_modules.strategy_relation_style_mixin import StrategyRelationStyleMixin
from src.chat.heart_flow.enhanced_modules.interaction_core_mixin import EnhancedInteractionCoreMixin
from src.chat.heart_flow.enhanced_modules.shared_runtime import BehaviorGovernorVerdict
from src.chat.heart_flow.speak_prediction_engine import SpeakPredictionEngine
from src.chat.proactive.perception_engine import _RelationGauge
from src.core.content_state_tracker import (
    ContentStatus,
    IgnoreReason,
    get_content_state_tracker,
)
from src.core.impression_evolution_hub import get_impression_hub, remove_impression_hub
from src.core.night_cycle_system import NightCycleSystem, remove_night_cycle
from src.core.unified_planner import PlanningContext, UnifiedPlanner
from src.core.world_snapshot import (
    build_relation_rapport_snapshot,
    TargetUserState,
    WorldSnapshot,
    get_relation_number,
    _reconcile_overlapping_values,
    _set_relation_value,
)
from src.webui.app import create_app
from src.webui.services.state_monitor import (
    _build_circadian_detail,
    _build_current_user_detail,
    _build_participant_impacts,
    _build_safety_detail,
    _build_timing_gate_detail,
)


REQUIRED_EARLY_STAGES = [
    "self_echo_gate",
    "short_batch_skip",
    "watch_gate_blackout",
    "peek_gate_observe",
    "night_gate_skip",
    "autonomy_guard_skip",
    "autonomy_guard_hard_skip",
    "planner_cooldown_skip",
    "pipeline_early_exit",
    "dashboard_hard_block",
    "pattern_route_skip",
    "scene_constraint_skip",
]


def _dashboard_snapshot() -> Dict[str, Any]:
    return {
        "available": True,
        "snapshot": {
            "reply_decision": {
                "reply_urgency": "可稍后回",
                "suggested_tone": "正常回应",
                "confidence": 0.6,
            },
            "social_attitude": {"willingness": 0.55},
            "attention": {"process_ratio": "65%"},
            "energy_reserve": {"percent": 68},
            "safety_shield": {"blocked": False},
        },
    }


def _base_domains() -> Dict[str, Any]:
    return {
        "emergence_core": {
            "initiative_drive": 0.62,
            "boredom_load": 0.42,
            "loneliness_load": 0.24,
            "environment_fatigue_load": 0.12,
            "withdrawal_drive": 0.08,
        },
        "resource_ledger": {
            "energy_reserve_ratio": 0.72,
            "irritation_load": 0.0,
        },
        "group_climate": {
            "scene_heat_score": 0.58,
            "active_user_count": 4,
            "topic_focus": ["测试话题"],
        },
        "trauma_load": {
            "trauma_load": 0.0,
            "pressure_load": 0.0,
            "cognitive_drag": 0.0,
        },
        "relationship_profile": {
            "rapport_score": 25.0,
            "trust_score": 55.0,
            "irritation_load": 0.0,
            "pressure_load": 0.0,
            "trauma_load": 0.0,
        },
        "counterparty_intent": {
            "readiness_score": 0.65,
            "hostility_detected": False,
        },
        "boundary_guard": {
            "blocked": False,
            "hostility_detected": False,
        },
        "tempo": {
            "cooling_down": False,
        },
        "circadian_rhythm": {
            "phase": "awake",
            "phase_label": "清醒",
            "is_sleeping": False,
            "is_burnthrough": False,
            "drowsiness_value": 10,
            "sleep_debt": 0.1,
            "overnight_pressure": 0,
            "sleep_reserve": 80,
            "response_suppression": 0.0,
            "remaining_sleep_replies": 1,
        },
        "pending_response": {
            "pending_active": False,
            "pending_seconds": 0,
            "should_reengage": False,
        },
        "flow_runtime": {
            "last_reactive_plan": {},
        },
        "memory_stack": {
            "memoir_phase": "open",
            "memoir_consecutive_timeouts": 0,
            "overload": {
                "load_ratio": 0.1,
                "amnesia_pressure": 0.05,
                "emergency_needed": False,
            },
            "reactivation": {},
        },
        "autonomy_runtime": {
            "active_intentions": [],
            "intention_drive": 0.0,
            "max_intention_urgency": 0.0,
            "pending_proactive_events": 0,
            "background_event": {},
        },
    }


def check_source_contract() -> Dict[str, Any]:
    loop_source = (ROOT / "src/chat/heart_flow/enhanced_modules/loop_main_driver_mixin.py").read_text(
        encoding="utf-8"
    )
    prediction_source = (ROOT / "src/chat/heart_flow/speak_prediction_engine.py").read_text(
        encoding="utf-8"
    )
    message_server_source = (ROOT / "src/common/message/message_server.py").read_text(
        encoding="utf-8"
    )
    missing = [stage for stage in REQUIRED_EARLY_STAGES if stage not in loop_source]
    assert not missing, f"缺少 early gate stage: {missing}"
    assert "_store_gate_runtime(" in loop_source, "主循环未调用 gate_runtime 写入层"
    assert "pre_execution" in prediction_source, "预测引擎未区分前置裁定"
    assert "execution_has_verdict" in prediction_source, "预测引擎未显式区分 execution_runtime"
    assert "WebSocket 连接建立" not in message_server_source
    assert "WebSocket 断开" not in message_server_source
    assert "消息适配器通道已连接" in message_server_source
    return {"early_stages": len(REQUIRED_EARLY_STAGES)}


def check_remote_telemetry_removed_contract() -> Dict[str, Any]:
    main_source = (ROOT / "src/main.py").read_text(encoding="utf-8")
    assert not (ROOT / "src/manager/remote.py").exists()
    assert "TelemetryHeartBeatTask" not in main_source
    assert "src.manager.remote" not in main_source
    assert "远程连接" not in main_source
    assert "TELEMETRY_SERVER_URL" not in main_source
    return {
        "remote_telemetry_module_removed": True,
        "startup_telemetry_task_removed": True,
    }


def check_prediction_runtime_contract() -> Dict[str, Any]:
    engine = SpeakPredictionEngine()

    execution_domains = _base_domains()
    execution_domains["execution_runtime"] = {
        "verdict_id": "exec-1",
        "reply_sent": False,
        "should_act": False,
        "final_action": "observe",
        "execution_stage": "night_gate_skip",
        "execution_reason": "夜间节律阻断",
        "model_path": "skip",
        "complexity_label": "普通",
    }
    result = engine.predict(
        channel_id="regression",
        domains=execution_domains,
        dashboard_snapshot=_dashboard_snapshot(),
    )
    assert result["shared_verdict_id"] == "exec-1"
    assert result["should_reply"] is False
    assert result["execution_stage"] == "night_gate_skip"
    assert result["execution_action"] == "observe"

    decision_domains = _base_domains()
    decision_domains["decision_runtime"] = {
        "verdict_id": "dec-1",
        "should_reply": True,
        "next_action": "reply",
        "decision_stage": "state_consensus",
        "decision_reason": "状态链达成回复倾向",
        "model_path": "small",
        "complexity_label": "普通",
    }
    result = engine.predict(
        channel_id="regression",
        domains=decision_domains,
        dashboard_snapshot=_dashboard_snapshot(),
    )
    assert result["shared_verdict_id"] == "dec-1"
    assert result["should_reply"] is True
    assert result["execution_stage"] == "pre_execution"
    assert result["execution_action"] == "reply"
    assert "尚未进入最终执行" in result["decision_reason"]

    return {"execution_runtime": True, "decision_runtime": True}


def check_memory_and_autonomy_contract() -> Dict[str, Any]:
    engine = SpeakPredictionEngine()
    dashboard = _dashboard_snapshot()

    normal_domains = _base_domains()
    normal = engine.predict(channel_id="memory-normal", domains=normal_domains, dashboard_snapshot=dashboard)

    overload_domains = _base_domains()
    overload_domains["memory_stack"]["overload"] = {
        "load_ratio": 0.92,
        "amnesia_pressure": 0.74,
        "emergency_needed": True,
    }
    overloaded = engine.predict(channel_id="memory-overload", domains=overload_domains, dashboard_snapshot=dashboard)
    assert overloaded["suppression_score"] > normal["suppression_score"]
    assert overloaded["reply_length"] == "简短"
    assert any("记忆过载" in item or "记忆负载" in item for item in overloaded["suppressing_factors"])

    memoir_domains = _base_domains()
    memoir_domains["memory_stack"]["memoir_phase"] = "anticipating"
    memoir_domains["memory_stack"]["memoir_consecutive_timeouts"] = 1
    memoir = engine.predict(channel_id="memoir", domains=memoir_domains, dashboard_snapshot=dashboard)
    assert memoir["drive_score"] > normal["drive_score"]
    assert any("回忆录" in item for item in memoir["driving_factors"])

    autonomy_domains = _base_domains()
    autonomy_domains["autonomy_runtime"] = {
        "active_intentions": [{"intent_id": "i1", "kind": "followup", "effective_urgency": 0.8}],
        "intention_drive": 0.62,
        "max_intention_urgency": 0.8,
        "pending_proactive_events": 2,
        "background_event": {"type": "memory_recall", "priority": 0.7},
    }
    autonomy = engine.predict(channel_id="autonomy", domains=autonomy_domains, dashboard_snapshot=dashboard)
    assert autonomy["drive_score"] > normal["drive_score"]
    assert any("主动意图" in item for item in autonomy["driving_factors"])
    assert any("后台事件" in item for item in autonomy["driving_factors"])

    return {
        "memory_overload": True,
        "memoir_anticipating": True,
        "autonomy_runtime": True,
    }


def check_deprecated_small_module_cleanup_contract() -> Dict[str, Any]:
    assert not (ROOT / "src/config" / ("threshold_" + "config.py")).exists()

    vector_probe_source = (
        ROOT / "src/memory_system/retrieval_tools/query_embedding_memory.py"
    ).read_text(encoding="utf-8")
    for token in (
        "_Cached" + "Vector",
        "Embedding" + "Reservoir",
        "_vector_" + "reservoir",
        "_resolve_" + "record_vector",
        "_probe_" + "memory_by_vector_legacy",
    ):
        assert token not in vector_probe_source
    assert "class StreamAwareVectorProbe" in vector_probe_source
    assert "probe_memory_by_vector" in vector_probe_source

    registry_source = (
        ROOT / "src/memory_system/retrieval_tools/tool_registry.py"
    ).read_text(encoding="utf-8")
    assert ("get_" + "action_types_list") not in registry_source
    assert "get_tool_definitions" in registry_source

    lifecycle_source = (
        ROOT / "src/chat/heart_flow/enhanced_modules/scene_bot_lifecycle_mixin.py"
    ).read_text(encoding="utf-8")
    assert ("_ensure_" + "night_cycle") not in lifecycle_source
    assert "_apply_night_cycle_modulation" in lifecycle_source

    relation_source = (
        ROOT / "src/chat/heart_flow/enhanced_modules/strategy_relation_style_mixin.py"
    ).read_text(encoding="utf-8")
    assert ("_evaluate_" + "impulse_factors") not in relation_source
    assert ("_log_" + "impulse_factors") not in relation_source
    assert "_update_emotion_state" in relation_source
    return {
        "old_threshold_module_removed": True,
        "legacy_vector_probe_removed": True,
        "deprecated_registry_action_types_removed": True,
        "deprecated_night_cycle_stub_removed": True,
        "deprecated_impulse_stubs_removed": True,
    }


def check_social_compat_bridge_cleanup_contract() -> Dict[str, Any]:
    fuser_source = (
        ROOT / "src/modules/social_value/social_affect_fuser.py"
    ).read_text(encoding="utf-8")
    settlement_source = (
        ROOT / "src/modules/social_value/settlement_engine.py"
    ).read_text(encoding="utf-8")
    for token in (
        "compat_" + "update",
        "compat_" + "get_value",
        "compat_" + "get_stage",
        "compat_" + "get_custom_label",
        "compat_" + "record_value",
        "compat_" + "get_level_factor",
        "compat_" + "favor_of",
        "compat_" + "capture_snapshot",
    ):
        assert token not in fuser_source
    assert ("Social" + "UpdateResult") not in fuser_source
    assert ("build_" + "legacy_update_result") not in settlement_source
    assert ("Social" + "UpdateResult") not in settlement_source
    assert "def collect_diagnostics" in settlement_source
    return {
        "social_affect_compat_bridge_removed": True,
        "settlement_legacy_result_builder_removed": True,
    }


def check_participant_contract() -> Dict[str, Any]:
    domains = {
        "relationship_profile": {"user_id": "u1", "display_name": "甲", "relationship_label": "熟人"},
        "relationship_population": {
            "participants": [
                {
                    "user_id": "u1",
                    "display_name": "甲",
                    "relationship_label": "熟人",
                    "rapport_score": 30,
                    "trust_score": 40,
                    "irritation_load": 0,
                    "pressure_load": 0,
                    "trauma_load": 0,
                    "chaos_load": 0,
                    "mask_load": 0,
                    "interaction_count": 5,
                    "current_mood_hint": "平稳接触",
                    "last_interaction_age_sec": 10,
                    "impact_rank": 140,
                    "is_current_target": True,
                    "in_current_scene": True,
                    "recent_speaker": True,
                    "recent_targeted_interaction": True,
                    "source_scope": "current_scene",
                }
            ]
        },
    }
    impacts = _build_participant_impacts(domains, [])
    assert impacts and impacts[0]["is_current_target"] is True
    for key in (
        "chaos_load",
        "mask_load",
        "current_mood_hint",
        "is_current_target",
        "in_current_scene",
        "recent_speaker",
        "recent_targeted_interaction",
        "source_scope",
    ):
        assert key in impacts[0], key
    return {"participant_impacts": len(impacts)}


def check_world_snapshot_relation_contract() -> Dict[str, Any]:
    snap = WorldSnapshot(target_user=TargetUserState())
    user = snap.target_user

    assert _set_relation_value(user, "social_value", 0.0, "test.social_zero") is True
    assert _set_relation_value(user, "trust_value", -12.5, "test.trust_negative") is True
    assert _set_relation_value(user, "annoyance_value", 0.0, "test.annoyance_zero") is True
    assert _set_relation_value(user, "annoyance_value", 33.0, "test.annoyance_fallback", overwrite=False) is False
    _reconcile_overlapping_values(snap)

    assert user.social_value == 0.0
    assert user.favorability == 0.0
    assert user.trust_value == -12.5
    assert user.trust_score == -12.5
    assert user.annoyance_value == 0.0
    assert user.relation_value_sources["social_value"] == "test.social_zero"
    assert user.relation_value_sources["favorability"] == "alias:test.social_zero"
    assert user.relation_value_sources["trust_score"] == "alias:test.trust_negative"
    assert user.relation_value_sources["annoyance_value"] == "test.annoyance_zero"

    alias_only = WorldSnapshot(target_user=TargetUserState())
    alias_user = alias_only.target_user
    assert _set_relation_value(alias_user, "trust_score", -7.0, "test.emotion_trust") is True
    _reconcile_overlapping_values(alias_only)
    assert alias_user.trust_score == -7.0
    assert alias_user.trust_value == 0.0
    assert alias_user.relation_value_sources["trust_score"] == "test.emotion_trust"
    assert "trust_value" not in alias_user.relation_value_sources

    canonical = snap.to_canonical_state()["target"]
    rapport = snap.to_rapport_dict()
    assert canonical["relation_value_sources"]["social_value"] == "test.social_zero"
    assert rapport["relation_value_sources"]["annoyance_value"] == "test.annoyance_zero"
    assert get_relation_number(rapport, "social_value", aliases=("favorability",)) == 0.0
    assert get_relation_number(rapport, "trust_value", aliases=("trust_score",)) == -12.5
    assert get_relation_number(rapport, "annoyance_value", aliases=("annoyance",)) == 0.0

    explicit_zero_relation = {
        "social_value": 0.0,
        "affection": 80.0,
        "trust_score": -9.0,
        "annoyance": 0.0,
        "relation_value_sources": {
            "social_value": "test.explicit_zero_social",
            "affection": "test.affection",
            "trust_score": "test.trust_alias",
            "annoyance": "test.annoyance_alias",
        },
    }
    receptiveness, summary = _RelationGauge.evaluate(explicit_zero_relation)
    assert "好感度高" not in summary
    assert get_relation_number(explicit_zero_relation, "trust_value", aliases=("trust_score",)) == -9.0
    assert get_relation_number(explicit_zero_relation, "annoyance_value", aliases=("annoyance",)) == 0.0
    assert receptiveness < 0.6
    return {"zero_values_preserved": True, "negative_values_preserved": True}


def check_dynamic_personal_impression_contract() -> Dict[str, Any]:
    channel_id = "dynamic-impression-regression"
    isolated_channel_id = "dynamic-impression-regression-isolated"
    user_id = "impression-user"
    remove_impression_hub(channel_id)
    remove_impression_hub(isolated_channel_id)
    try:
        hub = get_impression_hub(channel_id)
        hub.update_free_impression(
            user_id,
            "活泼可爱，喜欢突然抛梗",
            labels=["活泼", "会接梗", "活泼", "  "],
            user_name="印象用户",
            source="regression",
        )

        rapport = build_relation_rapport_snapshot(channel_id=channel_id, user_id=user_id, user_name="印象用户")
        assert rapport["relation_contract_version"] == 3
        assert rapport["relationship"] == "活泼可爱，喜欢突然抛梗"
        assert rapport["personal_impression"] == "活泼可爱，喜欢突然抛梗"
        assert rapport["impression_labels"] == ["活泼", "会接梗"]
        assert rapport["impression_source"] == "regression"
        assert rapport["legacy_relationship_label"] != rapport["personal_impression"]
        assert "群内关系级别" not in rapport["profile_summary"]
        assert "个人印象=活泼可爱，喜欢突然抛梗" in rapport["profile_summary"]

        world_state = {
            "target": {
                "group_friend_id": user_id,
                "group_friend_name": "印象用户",
                "personal_impression": rapport["personal_impression"],
                "impression_labels": rapport["impression_labels"],
                "impression_source": rapport["impression_source"],
                "impression_updated_at": rapport["impression_updated_at"],
                "custom_label": "好友",
                "mood": "平静",
                "affection": 12.0,
                "trust_value": 7.0,
                "annoyance_value": 0.0,
                "interaction_count": 3,
            }
        }
        chat = type(
            "FakeChat",
            (),
            {"stream_id": "", "_last_user_id": user_id, "_last_proactive_target_user_id": ""},
        )()
        relation = _extract_relationship_profile(chat, world_state)
        assert relation["relationship_label"] == "活泼可爱，喜欢突然抛梗"
        assert relation["personal_impression"] == "活泼可爱，喜欢突然抛梗"
        assert relation["impression_labels"] == ["活泼", "会接梗"]
        assert relation["legacy_relationship_label"] == "好友"

        detail = _build_current_user_detail({"relationship_profile": relation})
        assert detail["relationship_label"] == "活泼可爱，喜欢突然抛梗"
        assert detail["personal_impression"] == "活泼可爱，喜欢突然抛梗"
        assert detail["legacy_relationship_label"] == "好友"

        impacts = _build_participant_impacts(
            {
                "relationship_profile": {"user_id": user_id},
                "relationship_population": {"participants": [dict(relation, is_current_target=True)]},
            },
            [],
        )
        assert impacts[0]["relationship_label"] == "活泼可爱，喜欢突然抛梗"
        assert impacts[0]["personal_impression"] == "活泼可爱，喜欢突然抛梗"

        isolated_rapport = build_relation_rapport_snapshot(
            channel_id=isolated_channel_id,
            user_id=user_id,
            user_name="印象用户",
        )
        assert isolated_rapport["personal_impression"] == ""
        assert isolated_rapport["relationship"] == ""
        assert isolated_rapport["impression_labels"] == []

        planner_ctx = PlanningContext(channel_id=channel_id, user_id=user_id)
        planner_ctx.personal_impression = "活泼可爱，喜欢突然抛梗"
        planner_ctx.custom_label = "好友"
        planner_prompt = UnifiedPlanner()._build_decision_prompt(planner_ctx)
        assert "- 对当前这位的印象: 活泼可爱，喜欢突然抛梗" in planner_prompt
        assert "- 对当前这位的印象: 好友" not in planner_prompt

        style_probe = type(
            "StyleProbe",
            (StrategyRelationStyleMixin,),
            {
                "stream_id": channel_id,
                "log_prefix": "[regression]",
                "_last_user_id": "",
                "_resolve_relation_view": lambda self, relation_snapshot=None: dict(relation_snapshot or {}),
                "_get_self_behavior_style_hints": lambda self, relation_snapshot=None: [],
                "_is_acute_spamming": lambda self: False,
            },
        )()
        style_context = style_probe._build_reply_style_context(
            {
                "personal_impression": "活泼可爱，喜欢突然抛梗",
                "relationship": "好友",
                "legacy_relationship_label": "好友",
                "custom_label": "好友",
                "trust_value": 7.0,
                "affection": 12.0,
            }
        )
        assert "你对对方的个人印象是活泼可爱，喜欢突然抛梗" in style_context
        assert "你对对方的个人印象是好友" not in style_context
        assert "你和对方算" not in style_context
        return {
            "dynamic_personal_impression": True,
            "context_isolated": True,
            "consumer_priority": True,
        }
    finally:
        remove_impression_hub(channel_id)
        remove_impression_hub(isolated_channel_id)


def check_statusbar_export_contract() -> Dict[str, Any]:
    engine = SpeakPredictionEngine()

    baseline = engine.predict(
        channel_id="statusbar-baseline",
        domains=_base_domains(),
        dashboard_snapshot=_dashboard_snapshot(),
    )
    assert baseline["runtime_sync_state"] == "baseline"
    assert baseline["runtime_sync_label"] == "仅按基线估算"
    assert baseline["content_source"] == "topic_focus"
    assert baseline["content_source_label"] == "按话题焦点估算"
    assert "群聊" not in baseline["content_direction"]
    assert all("群聊" not in item for item in baseline["driving_factors"])

    planned_domains = _base_domains()
    planned_domains["flow_runtime"] = {"last_reactive_plan": {"content_plan": "顺着上一轮继续回应"}}
    planned = engine.predict(
        channel_id="statusbar-planned",
        domains=planned_domains,
        dashboard_snapshot=_dashboard_snapshot(),
    )
    assert planned["content_source"] == "reactive_plan"
    assert planned["content_source_label"] == "已同步内容规划"

    safety_runtime = _extract_safety_runtime(
        {
            "cached_safety_assessment": {
                "overall_level_label": "高风险",
                "overall_score": 0.83,
                "threat": "越界",
                "blocked": True,
                "bar_penalty": 0.31,
                "threat_evidence_summary": "命中越界规则",
            }
        }
    )
    assert safety_runtime == {
        "level": "高风险",
        "score": 0.83,
        "dominant_threat": "越界",
        "blocked": True,
        "bar_penalty": 0.31,
        "threat_evidence_summary": "命中越界规则",
    }
    safety_detail = _build_safety_detail({"safety_runtime": safety_runtime})
    assert safety_detail["safety_level"] == "高风险"
    assert safety_detail["safety_score"] == 0.83
    assert safety_detail["blocked"] is True
    assert safety_detail["dominant_threat"] == "越界"
    assert safety_detail["bar_penalty"] == 0.31
    assert safety_detail["threat_evidence_summary"] == "命中越界规则"

    service_source = (ROOT / "src/webui/services/state_monitor.py").read_text(encoding="utf-8")
    dashboard_source = (ROOT / "web/dashboard/src/components/EmotionDashboard/index.tsx").read_text(encoding="utf-8")
    forbidden_statusbar_text = [
        "群聊 / 私聊",
        "免登录查看",
        "状态页无需登录凭证",
        "群聊升温",
        "群聊整体状态",
        "群场景硬约束",
        "当前群聊热度",
        "群聊烦躁 / 疲劳",
        "群聊里其他人的影响",
        "群聊影响",
    ]
    for text in forbidden_statusbar_text:
        assert text not in service_source, text
        assert text not in dashboard_source, text
    assert "会话升温" in service_source
    assert "会话整体状态" in dashboard_source
    assert "runtime_sync_label" in dashboard_source

    timing_gate_detail = _build_timing_gate_detail(
        {
            "timing_gate": {
                "current": {
                    "verdict_id": "gate-current",
                    "at": 123.0,
                    "gate_result": "continue",
                    "stage": "full_pipeline_entry",
                    "reason": "状态门控放行，进入完整回复管线",
                    "source": "timing_gate",
                    "final_action": "continue",
                    "next_action": "continue",
                    "model_path": "pending",
                    "confidence": 0.72,
                },
                "history": [
                    {
                        "verdict_id": "gate-observe",
                        "at": 122.0,
                        "gate_result": "observe",
                        "stage": "peek_gate_observe",
                        "reason": "窥屏态观察完成，内心无波澜",
                        "source": "peek_gate",
                        "final_action": "observe",
                        "next_action": "observe",
                        "model_path": "skip",
                        "confidence": 0.81,
                    },
                    {
                        "verdict_id": "gate-current",
                        "at": 123.0,
                        "gate_result": "continue",
                        "stage": "full_pipeline_entry",
                        "reason": "状态门控放行，进入完整回复管线",
                        "source": "timing_gate",
                        "final_action": "continue",
                        "next_action": "continue",
                        "model_path": "pending",
                        "confidence": 0.72,
                    },
                ],
            }
        }
    )
    assert timing_gate_detail["current"]["gate_result_label"] == "继续完整管线"
    assert timing_gate_detail["current"]["stage_label"] == "完整管线入口"
    assert timing_gate_detail["current"]["final_action_label"] == "继续管线"
    assert timing_gate_detail["history_count"] == 2
    assert timing_gate_detail["history"][0]["stage_label"] == "窥屏后继续观察"
    assert "timing_gate_detail" in service_source
    assert "Timing Gate" in dashboard_source
    assert "timingGateDetail" in dashboard_source
    return {"prediction_sync": True, "safety_aliases": True, "statusbar_copy": True, "timing_gate_detail": True}


def check_monitor_overview_contract() -> Dict[str, Any]:
    from src.chat.heart_flow.heartflow import heartflow
    from src.chat.message_receive.chat_stream import ChatStream, get_chat_manager
    from src.common.message_types.group_info import GroupInfo
    from src.common.message_types.user_info import UserInfo

    manager = get_chat_manager()
    old_chats = dict(heartflow.heartflow_chat_list)
    old_active_since = dict(heartflow._active_since)
    old_streams = dict(manager.streams)
    now = 123456.0
    try:
        public_id = "statusbar-public"
        internal_id = "statusbar-internal"
        public_chat = type("PublicChat", (), {"_last_user_id": "u-public"})()
        internal_chat = type("InternalChat", (), {"_last_user_id": "u-internal"})()
        heartflow.heartflow_chat_list.clear()
        heartflow.heartflow_chat_list.update({public_id: public_chat, internal_id: internal_chat})
        heartflow._active_since.clear()
        heartflow._active_since.update({public_id: now, internal_id: now})
        manager.streams[public_id] = ChatStream(
            stream_id=public_id,
            platform="qq",
            user_info=UserInfo(user_id="u-public", user_nickname="公聊用户", platform="qq"),
            group_info=GroupInfo(group_id="g-public", group_name="公开会话", platform="qq"),
        )
        manager.streams[internal_id] = ChatStream(
            stream_id=internal_id,
            platform="webui",
            user_info=UserInfo(user_id="u-internal", user_nickname="本地测试", platform="webui"),
        )

        overview = list_heartfc_chats()
        assert overview["active_count"] == 1
        assert overview["hidden_internal_count"] == 1
        assert len(overview["channels"]) == 1
        channel = overview["channels"][0]
        assert channel["channel_id"] == public_id
        assert channel["chat_type"] == "group"
        assert channel["chat_type_label"] == "群聊"
        assert channel["platform_label"] == "QQ"
        assert channel["display_name"] == "公开会话"
        assert channel["is_internal_webui"] is False
        assert channel["target_user_id"] == "u-public"
    finally:
        heartflow.heartflow_chat_list.clear()
        heartflow.heartflow_chat_list.update(old_chats)
        heartflow._active_since.clear()
        heartflow._active_since.update(old_active_since)
        manager.streams.clear()
        manager.streams.update(old_streams)
    return {"hidden_internal_count": True, "public_channel_naming": True}


def check_night_status_label_contract() -> Dict[str, Any]:
    state_monitor_source = (ROOT / "src/webui/services/state_monitor.py").read_text(encoding="utf-8")
    night_source = (ROOT / "src/core/night_cycle_system.py").read_text(encoding="utf-8")
    runtime_source = (
        ROOT / "src/chat/heart_flow/enhanced_modules/runtime_integration_lifecycle_mixin.py"
    ).read_text(encoding="utf-8")
    exporter_source = (ROOT / "src/chat/heart_flow/heartfc_state_exporter.py").read_text(encoding="utf-8")
    assert '"circadian": "夜间"' in state_monitor_source
    assert '"circadian_rhythm": "夜间状态"' in state_monitor_source
    assert "夜间机制窗口、睡眠债、困意和熬夜压力" in state_monitor_source
    assert "作息分区、晚上/半夜/凌晨阶段、睡眠债、困意和熬夜压力" not in state_monitor_source
    assert '"昼夜节律"' not in state_monitor_source
    assert 'get_logger("夜间状态")' in night_source
    assert 'NightPhase.NIGHT_ACTIVE: "晚上阶段"' in night_source
    assert 'NightPhase.MIDNIGHT_REFLECT: "凌晨阶段"' in night_source
    assert "def resolve_time_band" in night_source
    assert "def mechanism_windows" in night_source
    assert "def persist_state" in night_source
    assert "_load_persisted_state" in night_source
    assert "_night_state_slot_key" in night_source
    assert "_apply_daytime_state_calibration" in night_source
    assert 'resolve_module_view("night_cycle_runtime")' in night_source
    assert "daytime_drowsiness_cap" in night_source
    assert "MAX_ITERATION_PASSES" not in night_source
    assert "CONVERGENCE_EPSILON" not in night_source
    assert "data/night_cycle_state" not in night_source
    assert "is_night=clock_is_night" in night_source
    assert "self._state.in_night_mode = clock_is_night" in night_source
    assert "self._state.last_stimulus_at = now" in night_source
    assert "get_night_cycle(channel_id)" in exporter_source
    assert "runtime_has_live_night" in exporter_source
    assert "has_cached_time_band" in exporter_source
    assert "evaluate_current(" in exporter_source
    assert '"late_night", "半夜", "睡眠窗口开始"' in night_source
    assert '"midnight", "凌晨", "凌晨反思窗口"' in night_source
    assert "_ncs.evaluate(" in runtime_source
    assert "record_overnight_activity(\"peek\"" in runtime_source
    assert "record_overnight_activity(\"interrupt\"" in runtime_source
    assert '"last_evaluated_at"' in runtime_source
    assert '"mechanism_windows"' in runtime_source
    assert '"pressure_breakdown"' in exporter_source
    assert '"mechanism_windows"' in exporter_source
    assert '"last_evaluated_at"' in exporter_source
    assert '"irritable": "被扰烦躁"' in exporter_source
    assert '"soft_night": "夜间放轻"' in exporter_source
    feedback_source = (ROOT / "src/chat/heart_flow/enhanced_modules/loop_resource_feedback_mixin.py").read_text(
        encoding="utf-8"
    )
    dashboard_source = (ROOT / "src/core/state_dashboard.py").read_text(encoding="utf-8")
    bot_lifecycle_source = (ROOT / "src/chat/heart_flow/enhanced_modules/scene_bot_lifecycle_mixin.py").read_text(
        encoding="utf-8"
    )
    assert "record_overnight_activity" in feedback_source and '"think"' in feedback_source
    assert "record_overnight_activity" in feedback_source and '"peek"' in feedback_source
    assert "D{_drowsy_val" not in feedback_source
    assert "P{_pressure_val" not in feedback_source
    assert "疲{_daily_fatigue_val" not in feedback_source
    assert "储{_reserve_val" not in feedback_source
    assert "困倦(D" not in feedback_source
    assert "{_mood_icon}{_np_label}" not in feedback_source
    assert "{_time_icon}{_np_label}" not in feedback_source
    assert "_body_status_label" in feedback_source
    assert "_time_period_label" in feedback_source
    assert '"drowsy": "有点困"' in dashboard_source
    assert '"drowsy": "困倦昏沉"' not in dashboard_source
    assert 'record_overnight_activity("chat", 1.0)' in bot_lifecycle_source

    detail = _build_circadian_detail(
        {
            "circadian_rhythm": {
                "phase": "midnight_reflect",
                "phase_label": "午夜反思",
                "time_band": "midnight",
                "time_band_label": "凌晨",
                "time_band_description": "凌晨反思窗口",
                "current_hour": 3,
                "mechanism_windows": {
                    "midnight_reflect": {"label": "凌晨反思", "range": "00:00-04:00", "active": True}
                },
                "last_evaluated_at": 1778258282.0,
                "pressure_breakdown": {
                    "total": 0.42,
                    "chat_minutes": 12,
                    "peek_minutes": 6,
                    "think_intensity": 2.5,
                    "interrupt_count": 2,
                },
                "sleep_reply_cap": 2,
                "remaining_sleep_replies": 1,
                "expression_style": "irritable",
            }
        }
    )
    assert detail["phase_label"] == "凌晨阶段"
    assert detail["time_band_label"] == "凌晨"
    assert detail["mechanism_windows"]["midnight_reflect"]["active"] is True
    assert detail["last_evaluated_at"] == 1778258282.0
    assert detail["pressure_breakdown"]["interrupt_count"] == 2
    assert detail["reply_quota_label"] == "1/2"
    assert detail["expression_style_label"] == "被扰烦躁"
    exported = _extract_circadian_rhythm(
        "regression-night",
        {
            "cached_night_summary": {
                "phase": "night_active",
                "phase_label": "晚上阶段",
                "time_band": "late_night",
                "time_band_label": "半夜",
                "time_band_description": "睡眠窗口开始",
                "current_hour": 23,
                "is_sleep_window": True,
                "is_pressure_window": True,
                "mechanism_windows": {
                    "sleep_window": {"label": "睡眠窗口", "range": "23:00-07:00", "active": True},
                    "overnight_pressure": {"label": "熬夜压力", "range": "22:00-07:00", "active": True},
                },
                "pressure_breakdown": {"total": 0.3, "chat_minutes": 9, "interrupt_count": 1},
            }
        },
        {},
    )
    assert exported["time_band_label"] == "半夜"
    assert exported["is_sleep_window"] is True
    assert exported["is_pressure_window"] is True
    assert exported["mechanism_windows"]["sleep_window"]["active"] is True
    assert exported["pressure_breakdown"]["chat_minutes"] == 9
    assert _extract_circadian_rhythm(
        "regression-night-style",
        {"cached_night_summary": {"expression_style": "irritable"}},
        {},
    )["expression_style_label"] == "被扰烦躁"
    direct_exported = _extract_circadian_rhythm(
        "regression-night-created",
        {},
        {"energy_ratio": 0.45, "boredom": 0.0},
    )
    try:
        assert direct_exported["last_evaluated_at"] > 0
        assert direct_exported["system_started_at"] > 0
        assert direct_exported["mechanism_windows"]
    finally:
        remove_night_cycle("regression-night-created")
    daytime = NightCycleSystem("regression-night-daytime")
    try:
        from src.core.night_cycle_system import NightPhase

        daytime._state.current_phase = NightPhase.DEEP_SLEEP
        daytime._state.drowsiness_value = 100.0
        daytime._state.sleepiness = 100.0
        daytime._state.half_asleep_level = 1.0
        daytime._state.body_state_tag = "deep_half_asleep"
        daytime._state.response_suppression_coef = 0.96
        daytime._state.last_drowsiness_tick = 0.0
        daytime._state.pressure_last_tick = 0.0
        daytime._state.derivative_timestamp = 0.0
        daytime.evaluate(
            energy_ratio=1.0,
            activity_level=35.0,
            boredom=0.0,
            social_stimulus=0.0,
            consecutive_active_minutes=0.0,
            current_hour=9,
        )
        behavior = daytime.night_behavior_summary()
        assert behavior["current_hour"] == 9
        assert daytime.state_snapshot.last_evaluated_hour == 9
        assert behavior["time_band"] == "morning"
        assert behavior["phase"] == "awake"
        assert behavior["phase_label"] == "清醒"
        assert behavior["drowsiness_value"] < 60.0
        assert behavior["body_state_tag"] != "deep_half_asleep"
        assert behavior["response_suppression"] < 0.4
    finally:
        remove_night_cycle("regression-night-daytime")
    from src.common.database.database_model import PersistenceSlot
    from src.core.night_cycle_system import _night_state_slot_key

    persist_slot_key = _night_state_slot_key("regression-night-persist")
    PersistenceSlot.delete().where(PersistenceSlot.slot_key == persist_slot_key).execute()
    persistent = NightCycleSystem("regression-night-persist")
    try:
        persistent._last_persist_at = 0.0
        persistent.evaluate(
            energy_ratio=0.25,
            activity_level=70.0,
            boredom=40.0,
            social_stimulus=0.2,
            consecutive_active_minutes=180.0,
            current_hour=2,
        )
        persistent.record_overnight_activity("chat", 0.8)
        saved_debt = persistent.state_snapshot.sleep_debt
        saved_chat_pressure = persistent.state_snapshot.pressure_chat_minutes
        persistent.persist_state(force=True)
        restored = NightCycleSystem("regression-night-persist")
        restored._load_persisted_state()
        assert restored.state_snapshot.last_evaluated_at > 0
        assert restored.state_snapshot.sleep_debt >= saved_debt
        assert restored.state_snapshot.pressure_chat_minutes >= saved_chat_pressure
    finally:
        remove_night_cycle("regression-night-persist")
        PersistenceSlot.delete().where(PersistenceSlot.slot_key == persist_slot_key).execute()
    return {
        "night_labels": True,
        "midnight_phase_label": detail["phase_label"],
        "time_band_label": exported["time_band_label"],
        "daytime_realtime_calibrated": True,
        "expression_style_label": detail["expression_style_label"],
        "persistent_night_state": True,
    }



def check_content_state_scope_contract() -> Dict[str, Any]:
    tracker = get_content_state_tracker()
    old_state_cache = dict(tracker._state_cache)
    old_hash_index = dict(tracker._hash_index)
    old_channel_index = {key: list(value) for key, value in tracker._channel_index.items()}
    old_user_index = {key: list(value) for key, value in tracker._user_index.items()}
    try:
        tracker._state_cache.clear()
        tracker._hash_index.clear()
        tracker._channel_index.clear()
        tracker._user_index.clear()

        content = "同一句内容"
        user_id = "content-scope-user"
        channel_a = "content-scope-a"
        channel_b = "content-scope-b"
        state_a = tracker.track_content(content, channel_id=channel_a, user_id=user_id)
        state_b = tracker.track_content(content, channel_id=channel_b, user_id=user_id)
        assert state_a.content_hash != state_b.content_hash

        probe = type(
            "ContentScopeProbe",
            (SceneContextAnalysisMixin,),
            {"stream_id": channel_a, "log_prefix": "[regression]"},
        )()
        msg = type(
            "FakeMessage",
            (),
            {"processed_plain_text": content, "user_id": user_id},
        )()
        probe._mark_message_content_processed(msg, "regression_reply", quality=0.9)
        assert state_a.status == ContentStatus.PROCESSED
        assert state_b.status == ContentStatus.NEW
        assert tracker.should_process(content, channel_id=channel_a, user_id=user_id).reason == "already_well_replied"
        assert tracker.should_process(content, channel_id=channel_b, user_id=user_id).should_process is True

        probe._apply_content_state_skip(
            {
                "target_content": content,
                "decision_reason": "low_interest",
                "channel_id": channel_b,
                "target_user_id": user_id,
            }
        )
        assert state_a.status == ContentStatus.PROCESSED
        assert state_b.status == ContentStatus.IGNORED
        assert state_b.ignore_reason == IgnoreReason.LOW_INTEREST.value
        assert tracker.should_process(content, channel_id=channel_b, user_id=user_id).reason == "already_ignored:low_interest"
        return {"channel_scoped_hash": True, "scoped_status_writeback": True}
    finally:
        tracker._state_cache.clear()
        tracker._state_cache.update(old_state_cache)
        tracker._hash_index.clear()
        tracker._hash_index.update(old_hash_index)
        tracker._channel_index.clear()
        tracker._channel_index.update(old_channel_index)
        tracker._user_index.clear()
        tracker._user_index.update(old_user_index)


def check_force_guard_contract() -> Dict[str, Any]:
    probe = type(
        "ForceGuardProbe",
        (ProactiveContextPromptMixin,),
        {
            "stream_id": "force-guard-regression",
            "log_prefix": "[regression]",
            "_has_targeted_bot_message": lambda self, messages: False,
            "_get_human_message_candidates": lambda self, messages: list(messages),
        },
    )()
    msg = type(
        "FakeMessage",
        (),
        {"processed_plain_text": "这条消息需要进入强制保护测试", "user_id": "admin-user"},
    )()
    high_desire_voice = type("Voice", (), {"reply_desire_level": 10})()

    hard_guard = probe._evaluate_autonomy_guard(
        decision_messages=[msg],
        pinged_msg=None,
        identity_context={"identity": "default", "response_mode": "normal", "has_conflict": False},
        self_reply_risk={"is_self_reply": False, "similarity": 0.0},
        group_sense_result={},
        relation_result={},
        voice_conclusion=high_desire_voice,
        message_salience={"score": 5},
        content_state_signal={
            "should_skip": True,
            "decision_reason": "already_well_replied",
            "reason": "这类内容刚处理过且回应质量足够",
        },
    )
    assert hard_guard["should_skip"] is True
    assert hard_guard["source"] == "content_state"
    assert hard_guard["hard_skip"] is True

    soft_guard = probe._evaluate_autonomy_guard(
        decision_messages=[msg],
        pinged_msg=None,
        identity_context={"identity": "default", "response_mode": "normal", "has_conflict": False},
        self_reply_risk={"is_self_reply": False, "similarity": 0.0},
        group_sense_result={},
        relation_result={},
        voice_conclusion=high_desire_voice,
        message_salience={"score": 5},
        content_state_signal={
            "should_skip": True,
            "decision_reason": "low_interest",
            "reason": "当前内容信息量偏低，先不重复接话",
        },
    )
    assert soft_guard["should_skip"] is False
    assert soft_guard["hard_skip"] is False

    loop_source = (ROOT / "src/chat/heart_flow/enhanced_modules/loop_main_driver_mixin.py").read_text(
        encoding="utf-8"
    )
    lifecycle_source = (ROOT / "src/chat/heart_flow/enhanced_modules/scene_bot_lifecycle_mixin.py").read_text(
        encoding="utf-8"
    )
    assert "autonomy_guard[\"should_skip\"] and not _admin_force_active" in loop_source
    assert "管理员测试通道已绕过不回复限制" in loop_source
    assert "_action in (\"deep_sleep\", \"hard_block\")" in loop_source
    assert "_action == \"sleep_resist\" and not self._is_force_wake_admin" in loop_source
    assert "voice_conclusion = self._cached_voice" in loop_source
    assert "管理员测试通道: 绕过回复限制，但保留感知/独白/关系/语气链路" in loop_source
    assert "管理员强制唤醒-无视仪表盘硬阻断" in loop_source
    assert "管理员强制唤醒-无视低信息重复抑制" in loop_source
    assert "force_reply_message = pinged_msg or self._get_latest_human_message(decision_messages)" in loop_source
    assert "低信息重复施压" not in (ROOT / "src/chat/heart_flow/enhanced_modules/runtime_state_trace_mixin.py").read_text(
        encoding="utf-8"
    )

    assert "_apply_night_cycle_modulation(_now_hr, incoming_batch, pinged_msg)" in loop_source
    assert "深睡阶段禁止普通夜间回复" in lifecycle_source
    hard_block_idx = lifecycle_source.index("if not _ncs_cap.evaluate_sleep_reply_budget()")
    deep_sleep_idx = lifecycle_source.index("if _composite >= _collapse * 1.1")
    admin_wake_idx = lifecycle_source.index("if _admin_force_wake:")
    assert admin_wake_idx < hard_block_idx
    assert admin_wake_idx < deep_sleep_idx
    return {
        "admin_content_hard_guard": True,
        "admin_soft_guard_bypass": True,
        "admin_night_gate_preempts_sleep_block": True,
        "admin_dashboard_hard_block_bypass": True,
        "admin_low_info_repeat_bypass": True,
    }


def check_night_cycle_persistence_contract() -> Dict[str, Any]:
    from src.common.database.database_model import PersistenceSlot
    from src.core.night_cycle_system import _ensure_persistence_slot_table, _night_state_slot_key

    channel_id = "regression-night-db-only"
    slot_key = _night_state_slot_key(channel_id)
    _ensure_persistence_slot_table()
    PersistenceSlot.delete().where(PersistenceSlot.slot_key == slot_key).execute()
    remove_night_cycle(channel_id)

    system = NightCycleSystem(channel_id)
    system.persist_state(force=True)
    row = PersistenceSlot.get_or_none(PersistenceSlot.slot_key == slot_key)
    assert row is not None
    assert row.slot_value
    assert not list((ROOT / "data").rglob("*.json"))
    remove_night_cycle(channel_id)
    PersistenceSlot.delete().where(PersistenceSlot.slot_key == slot_key).execute()

    source = (ROOT / "src/core/night_cycle_system.py").read_text(encoding="utf-8")
    assert "_night_state_slot_key" in source
    assert "_ensure_persistence_slot_table" in source
    assert "Slot.insert(" in source
    assert "_safe_state_filename" not in source
    assert "_load_state_from_legacy_json" not in source
    assert ".write_text(" not in source
    return {
        "night_cycle_state_db_slot": True,
        "night_cycle_no_new_json_file": True,
    }


def check_night_soul_prompt_contract() -> Dict[str, Any]:
    from src.core.night_cycle_system import NightPhase, get_night_cycle
    from src.chat.replyer.group_generator import DefaultReplyer

    channel_id = "regression-night-soul"
    ncs = get_night_cycle(channel_id)
    ncs._state.current_phase = NightPhase.DAWN_RECOVER
    ncs._state.drowsiness_value = 72.0
    ncs._state.overnight_pressure = 88.0
    ncs._state.half_asleep_level = 0.62
    ncs._state.sleep_debt = 0.58
    ncs._state.body_state_tag = "半醒"
    ncs._state.monologue_mood_hint = "刚醒，反应慢"
    ncs._state.last_evaluated_at = time.time()
    probe = type(
        "NightSoulProbe",
        (ScenePlannerBridgeMixin,),
        {
            "stream_id": channel_id,
            "log_prefix": "[regression]",
        },
    )()
    parts: list[str] = []
    probe._inject_night_soul_state(parts)
    joined = "\n".join(parts)
    assert "[夜间身体状态]" in joined
    assert "困意=72" in joined
    assert "熬夜压力=88" in joined
    assert "夜间语气约束" in joined
    assert "夜间心境提示" in joined
    ensured = probe._ensure_soul_data_in_extra_info("[当前心理状态] 烦躁度0，回复平静。")
    assert "[夜间身体状态]" in ensured
    assert "夜间语气约束" in ensured
    group_reply_source = (ROOT / "src/chat/replyer/group_generator.py").read_text(encoding="utf-8")
    reply_message = type("ReplyMessageProbe", (), {"processed_plain_text": "你不困吗", "display_message": ""})()
    guarded = DefaultReplyer._apply_sleepy_fast_reply_guard(
        content="不太困，想再聊会儿",
        extra_info=ensured,
        reply_message=reply_message,
    )
    assert "不太困" not in guarded
    assert "想再聊" not in guarded
    assert "困" in guarded
    sleepy_short = DefaultReplyer._apply_sleepy_fast_reply_guard(
        content="又咋了？",
        extra_info=ensured,
        reply_message=type("ReplyMessageProbe", (), {"processed_plain_text": "我的刀盾", "display_message": ""})(),
    )
    assert sleepy_short == "又咋了？"
    assert "又咋了...我有点没反应过来" not in sleepy_short
    assert "脑子慢着" not in sleepy_short
    sleepy_confused = DefaultReplyer._apply_sleepy_fast_reply_guard(
        content="啥玩意儿？",
        extra_info=ensured,
        reply_message=type("ReplyMessageProbe", (), {"processed_plain_text": "曹氏你", "display_message": ""})(),
    )
    assert sleepy_confused == "啥玩意儿？"
    assert "反应有点慢" not in group_reply_source
    remove_night_cycle(channel_id)
    source = (ROOT / "src/chat/heart_flow/enhanced_modules/scene_planner_bridge_mixin.py").read_text(
        encoding="utf-8"
    )
    assert "self._inject_night_soul_state(extra_parts)" in source
    assert "fallback_lines[:6]" in source
    assert "不要套固定开头" in group_reply_source
    assert "困死了，" not in group_reply_source
    assert "低信息短句跳过工具链" in group_reply_source
    assert "_should_skip_tools_for_low_info_input" in group_reply_source
    assert DefaultReplyer._should_skip_tools_for_low_info_input("草饲你", True) is True
    assert DefaultReplyer._should_skip_tools_for_low_info_input("草饲你是什么意思？", True) is False
    return {
        "night_soul_prompt_injected": True,
        "dawn_sleepy_tone_guard": True,
        "fast_reply_sleep_denial_guard": True,
        "sleepy_short_reply_preserves_persona": True,
    }


def check_inner_voice_low_info_tone_contract() -> Dict[str, Any]:
    from src.chat.heart_flow.inner_voice import BoundaryContext, SelfDialogueEngine
    from src.common.data_models.heartflow_models import VoiceVerdict

    engine = SelfDialogueEngine("regression-low-info-tone")
    ctx = BoundaryContext(raw_text="刀盾刀盾", text_length=4, mentioned_me=False)
    verdict = VoiceVerdict(
        thinking="又发疯。",
        reply_desire_level=3,
        should_reply=False,
        current_mood="无聊",
        thinking_source="llm_json",
    )
    fixed = engine._stabilize_low_info_reflection(verdict, ctx)
    assert "发疯" not in fixed.thinking
    assert any(marker in fixed.thinking for marker in ("重复", "看懂", "没说清楚", "太短", "别乱猜"))
    assert "low_info_guard" in fixed.thinking_source
    overguess = VoiceVerdict(
        thinking="又来这句，他是不是没活了，有点无聊。",
        reply_desire_level=6,
        should_reply=True,
        current_mood="无聊",
        thinking_source="llm_json",
    )
    low_info_fixed = engine._stabilize_low_info_reflection(
        overguess,
        BoundaryContext(raw_text="草饲你", text_length=3, mentioned_me=False),
    )
    assert "没活" not in low_info_fixed.thinking
    assert "low_info_guard" in low_info_fixed.thinking_source
    multi_ref = VoiceVerdict(
        thinking="这俩人又在打什么哑谜，感觉有点莫名其妙的。",
        reply_desire_level=4,
        should_reply=False,
        current_mood="困惑",
        thinking_source="llm_json",
    )
    normalized = engine._stabilize_single_speaker_reference(
        multi_ref,
        BoundaryContext(raw_text="我的刀盾", speaker_id="3138039548"),
    )
    assert "这俩人" not in normalized.thinking
    assert "这人" in normalized.thinking
    assert "single_speaker_guard" in normalized.thinking_source
    prompt_source = (ROOT / "src/chat/prompts/catalog.py").read_text(encoding="utf-8")
    group_reply_source = (ROOT / "src/chat/replyer/group_generator.py").read_text(encoding="utf-8")
    assert "不要写\"发疯\"" in prompt_source
    assert "不要脑补他\"没活了\"" in prompt_source
    assert "不要把\"他和我\"误写成\"这俩人/他们俩\"" in prompt_source
    assert "不要说“没活了”" in group_reply_source
    return {
        "hostile_low_info_thought_softened": True,
        "underexplained_short_overguess_softened": True,
        "single_speaker_reference_normalized": True,
    }


def check_inner_voice_single_message_dedupe_contract() -> Dict[str, Any]:
    enhanced_source = (ROOT / "src/chat/heart_flow/heartFC_chat_enhanced.py").read_text(encoding="utf-8")
    voice_source = (ROOT / "src/chat/heart_flow/enhanced_modules/voice_pipeline_mixin.py").read_text(
        encoding="utf-8"
    )
    loop_source = (ROOT / "src/chat/heart_flow/enhanced_modules/loop_main_driver_mixin.py").read_text(
        encoding="utf-8"
    )

    assert "_inner_voice_inflight_tasks" in enhanced_source
    assert "_inner_voice_result_cache" in enhanced_source
    assert "_inner_voice_intention_ingested" in enhanced_source
    assert "def _get_voice_trigger_message" in voice_source
    assert "self._get_latest_human_message(messages or [])" in voice_source
    assert "def _build_inner_voice_message_key" in voice_source
    assert "mid:{message_id}" in voice_source
    assert "asyncio.shield(existing_task)" in voice_source
    assert "asyncio.shield(task)" in voice_source
    assert "task.add_done_callback(_finalize_inner_voice_task)" in voice_source
    assert "self._remember_inner_voice_result(voice_key, verdict, now)" in voice_source
    assert "def _mark_voice_intention_ingested" in voice_source
    assert "if self._mark_voice_intention_ingested(voice_key):" in loop_source
    assert "本轮内心独白意图已写入，跳过重复写入" in loop_source
    assert "直接快回后台独白意图已由主链写入" in loop_source
    assert "latest_human = self._get_voice_trigger_message(decision_messages)" in loop_source
    assert "latest = self._get_voice_trigger_message(messages)" in voice_source
    assert "latest = messages[-1] if messages else None" not in voice_source
    return {
        "single_message_voice_inflight_reused": True,
        "single_message_voice_cache_reused": True,
        "single_message_intention_ingest_deduped": True,
        "voice_trigger_prefers_latest_human": True,
    }


def check_webui_internal_chat_disabled_contract() -> Dict[str, Any]:
    from src.chat.message_receive.chat_stream import (
        LEGACY_WEBUI_LOCAL_GROUP_ID,
        is_internal_webui_stream,
        is_webui_virtual_stream,
    )

    chat_route_source = (ROOT / "src/webui/routers/chat.py").read_text(encoding="utf-8")
    stream_source = (ROOT / "src/chat/message_receive/chat_stream.py").read_text(encoding="utf-8")
    heartflow_source = (ROOT / "src/chat/heart_flow/heartflow.py").read_text(encoding="utf-8")

    legacy_const = "WEBUI" + "_CHAT_GROUP_ID"
    legacy_label = "WebUI" + "本地聊天室"
    assert legacy_const not in chat_route_source
    assert legacy_label not in chat_route_source
    assert "WebUI 独立本地聊天已停用" in chat_route_source
    assert "跳过聊天流创建与心流预热" in chat_route_source
    assert "ChatStreams.delete()" in stream_source
    assert "Messages.delete()" in stream_source
    assert "拒绝为 WebUI 内部管理连接创建心流实例" in heartflow_source
    assert is_internal_webui_stream(platform="webui", group_id=None) is True
    assert is_internal_webui_stream(platform="qq", group_id=LEGACY_WEBUI_LOCAL_GROUP_ID) is True
    assert is_internal_webui_stream(platform="qq", group_id="real-group") is False
    assert is_webui_virtual_stream("webui_virtual_group_abc") is True
    return {
        "webui_default_local_chat_disabled": True,
        "legacy_webui_stream_pruned": True,
        "startup_prewarm_skips_webui_virtual": True,
    }


def check_huoli_naming_contract() -> Dict[str, Any]:
    forbidden = (
        "Mai" + "Core",
        "Mai" + "Bot",
        "MIMiao" + "Core",
        "XB" + "core",
        "麦" + "麦",
        "mai" + "m_message",
        "品鉴" + "配置",
        "非常的" + "新鲜",
        "神经元" + "放电",
    )
    scanned = []
    for folder in ("src", "config", "template", "scripts"):
        base = ROOT / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix not in {".py", ".toml", ".md"}:
                continue
            if "web/dashboard/dist" in path.as_posix():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                assert token not in text, f"{path.relative_to(ROOT)} 仍包含旧命名 {token}"
            scanned.append(str(path.relative_to(ROOT)))
    config_source = (ROOT / "src/config/config.py").read_text(encoding="utf-8")
    assert "HuoLiCore当前版本" in config_source
    assert "huoli_message" in config_source
    assert "_legacy_message_section_name" in config_source
    return {"scanned_files": len(scanned), "huoli_message_section": True}


def check_admin_identity_not_relationship_contract() -> Dict[str, Any]:
    from src.modules.modcore.dynamic_persona.emotion_tracker import EmotionTracker

    source = (ROOT / "src/modules/modcore/dynamic_persona/emotion_tracker.py").read_text(
        encoding="utf-8"
    )
    assert 'state.relationship = "管理员"' not in source
    assert 'state.impression = "管理员"' not in source
    assert 'return "管理员"' not in source
    assert "return 100.0" not in source[source.index("def _derive_relationship_score") : source.index("def _derive_relationship_label")]
    assert "管理员身份，给予信任" not in source

    tracker = EmotionTracker("regression-admin-identity")
    state = tracker._dict_to_state(
        {
            "user_id": "admin-user",
            "stream_id": "regression-admin-identity",
            "relationship": "管理员",
            "last_relationship": "熟人",
            "impression": "管理员",
            "trust_score": 88.0,
        }
    )
    assert state.relationship == "熟人"
    assert state.impression != "管理员"
    return {
        "admin_permission_not_relationship": True,
        "legacy_admin_relationship_normalized": True,
    }


def check_db_backed_json_storage_contract() -> Dict[str, Any]:
    import asyncio

    from src.common.database.database_model import PersistenceSlot
    from src.hippo_memorizer.summary_storage import SummaryStorage, TopicSummary
    from src.memory_system.chat_history_summarizer import ChatHistorySummarizer, TopicCacheItem
    from src.modules.social_value.social_storage import SocialStorage

    hippo_source = (ROOT / "src/hippo_memorizer/summary_storage.py").read_text(encoding="utf-8")
    summarizer_source = (ROOT / "src/memory_system/chat_history_summarizer.py").read_text(encoding="utf-8")
    social_source = (ROOT / "src/modules/social_value/social_storage.py").read_text(encoding="utf-8")
    assert "_save_to_db" in hippo_source
    assert "_load_from_legacy_json" not in hippo_source
    assert "_file_path" not in hippo_source
    assert "atomic_json_dump" not in hippo_source
    assert "_save_topic_cache_payload" in summarizer_source
    assert "_topic_cache_file" not in summarizer_source
    assert "_load_topic_cache_from_legacy_json" not in summarizer_source
    assert "json.dump(data" not in summarizer_source
    assert "_persist_sync" in social_source
    assert "json.dump(data" not in social_source
    assert "os.makedirs(self._storage_dir" not in social_source

    summary_chat = "regression-hippo-summary"
    topic_chat = "regression-hippo-topic"
    social_user = "regression-social-user"
    social_channel = "regression-social-channel"
    slot_keys = [
        f"hippo_summary:{summary_chat}",
        f"hippo_topic_cache:{topic_chat}",
        f"social_value:{social_user}:{social_channel}",
    ]
    PersistenceSlot.delete().where(PersistenceSlot.slot_key.in_(slot_keys)).execute()

    storage = SummaryStorage(summary_chat, max_summaries=5)
    storage.add_summary(
        TopicSummary(
            topic="测试话题",
            summary="测试摘要",
            keywords=["测试"],
            key_points=["重点"],
            participants=["用户"],
            start_time=1.0,
            end_time=2.0,
        )
    )
    assert PersistenceSlot.get_or_none(PersistenceSlot.slot_key == slot_keys[0]) is not None

    summarizer = ChatHistorySummarizer(topic_chat)
    summarizer.topic_cache["测试话题"] = TopicCacheItem(
        topic="测试话题",
        messages=["1. 用户: 内容"],
        participants={"用户"},
    )
    summarizer._persist_topic_cache()
    assert PersistenceSlot.get_or_none(PersistenceSlot.slot_key == slot_keys[1]) is not None

    social = SocialStorage(storage_dir="ignored-json-dir")
    asyncio.run(social.set(social_user, social_channel, 12.5))
    social_row = PersistenceSlot.get_or_none(PersistenceSlot.slot_key == slot_keys[2])
    assert social_row is not None
    assert '"value":12.5' in social_row.slot_value
    assert not list((ROOT / "data").rglob("*.json"))

    PersistenceSlot.delete().where(PersistenceSlot.slot_key.in_(slot_keys)).execute()
    return {
        "hippo_summary_db_slot": True,
        "hippo_topic_cache_db_slot": True,
        "social_value_db_slot": True,
    }


def check_memoir_channel_key_contract() -> Dict[str, Any]:
    from src.chat.proactive.session_tracker import DialogueMemoir, MemoirCabinet

    cabinet = MemoirCabinet.__new__(MemoirCabinet)
    cabinet._memoirs = {}
    cabinet._user_locks = {}
    user_id = "memoir-user-regression"
    channel_id = "memoir-channel-regression"
    memoir = DialogueMemoir(user_id=user_id, channel_id=channel_id)
    cabinet.commit(channel_id, memoir)
    assert cabinet.lookup_sync(user_id) is memoir
    assert cabinet.lookup_sync(channel_id) is None
    assert cabinet.lookup_by_channel(channel_id) is memoir
    return {
        "memoir_key_uses_user_id": True,
        "memoir_channel_lookup_preserved": True,
    }


def check_heartflow_startup_no_channel_prewarm_contract() -> Dict[str, Any]:
    source = (ROOT / "src/chat/heart_flow/heartflow.py").read_text(encoding="utf-8")
    chat_core_source = (ROOT / "src/chat/chat_core_base.py").read_text(encoding="utf-8")
    loop_source = (ROOT / "src/chat/heart_flow/enhanced_modules/loop_main_driver_mixin.py").read_text(
        encoding="utf-8"
    )
    main_source = (ROOT / "src/main.py").read_text(encoding="utf-8")
    logger_source = (ROOT / "src/common/logger.py").read_text(encoding="utf-8")
    base_chat_source = (ROOT / "src/chat/heart_flow/heartFC_chat.py").read_text(encoding="utf-8")
    enhanced_source = (ROOT / "src/chat/heart_flow/heartFC_chat_enhanced.py").read_text(encoding="utf-8")
    gateway_source = (ROOT / "src/chat/heart_flow/decision_gateway.py").read_text(encoding="utf-8")
    startup_block = source[source.index("async def startup") : source.index("async def shutdown")]
    warmup_block = source[
        source.index("async def _run_startup_warmup") : source.index("    @staticmethod\n    def _should_skip_prewarm")
    ]
    create_block = source[source.index("async def _create_chat_instance") : source.index("    # ---- 清理 ----")]
    assert "self._warm_core_services()" in startup_block
    assert "self._schedule_startup_warmup()" in startup_block
    assert "get_chat_manager().streams.values()" in warmup_block
    assert "startup_all" in warmup_block
    assert "prewarm_chat(stream.stream_id" in warmup_block
    assert "启动频道预热完成" in warmup_block
    assert "notify_message_arrived()" in source
    assert "def notify_message_arrived" in chat_core_source
    assert "await self._sleep_or_wake" in loop_source
    assert "get_self_behavior_learner" in source
    assert "get_perception_generator" in source
    assert "get_bot_identity_manager" in source
    assert "get_group_persona_manager" in source
    assert "回复发送前耗时" in loop_source
    assert "结算收尾耗时" in loop_source
    assert "启动频道运行实例" in create_block
    assert "绑定已存在聊天流" in create_block
    assert "创建聊天实例" not in create_block
    assert "1000 * (time.time() - init_start_time)" not in main_source
    assert "init_seconds:.2f" in main_source
    assert "datetime.fromtimestamp(log_file.stat().st_mtime)" in logger_source
    assert "file_time = timestamp_to_datetime(log_file.stat().st_mtime)" not in logger_source
    assert "心流频道运行基类" in base_chat_source
    assert "async def _loopbody" not in base_chat_source
    assert ("旧版" + "循环体") not in base_chat_source
    readiness_token = "readiness_" + "evaluator"
    assert readiness_token not in enhanced_source
    assert readiness_token not in gateway_source
    assert ("_delegate_" + "readiness_evaluation") not in enhanced_source
    assert ("_apply_" + "readiness_constraint") not in enhanced_source
    return {
        "startup_prewarms_known_streams": True,
        "startup_warmup_scans_real_streams": True,
        "message_arrival_wakes_loop": True,
        "startup_warms_first_reply_singletons": True,
        "reply_timing_split_before_settlement": True,
        "startup_duration_seconds_unit": True,
        "log_cleanup_timezone_compare_fixed": True,
        "heartflow_runtime_log_disambiguated": True,
        "heartflow_base_legacy_loopbody_removed": True,
    }


def check_heartflow_gateway_naming_contract() -> Dict[str, Any]:
    heartflow_files = [
        ROOT / "src/chat/heart_flow/heartFC_chat_enhanced.py",
        ROOT / "src/chat/heart_flow/enhanced_modules/loop_main_driver_mixin.py",
        ROOT / "src/chat/heart_flow/enhanced_modules/loop_reply_execution_mixin.py",
        ROOT / "src/chat/heart_flow/enhanced_modules/loop_resource_feedback_mixin.py",
        ROOT / "src/chat/heart_flow/enhanced_modules/loop_state_flow_mixin.py",
        ROOT / "src/chat/heart_flow/enhanced_modules/proactive_reactive_flow_mixin.py",
        ROOT / "src/chat/heart_flow/enhanced_modules/runtime_state_trace_mixin.py",
        ROOT / "src/chat/heart_flow/enhanced_modules/runtime_subjective_signal_mixin.py",
        ROOT / "src/chat/heart_flow/enhanced_modules/runtime_watch_governor_mixin.py",
        ROOT / "src/chat/heart_flow/enhanced_modules/scene_bot_lifecycle_mixin.py",
        ROOT / "src/chat/heart_flow/enhanced_modules/strategy_action_state_mixin.py",
        ROOT / "src/chat/heart_flow/enhanced_modules/voice_pipeline_mixin.py",
        ROOT / "src/chat/heart_flow/reply_coordinator.py",
        ROOT / "src/webui/services/state_monitor.py",
    ]
    merged_source = "\n".join(path.read_text(encoding="utf-8") for path in heartflow_files)
    core_config_source = (ROOT / "config/core_config.toml").read_text(encoding="utf-8")

    banned_tokens = [
        "legacy" + "_gate",
        "_last_" + "legacy",
        "legacy" + "_constraint",
        "legacy" + "_block",
        "legacy" + "_high_desire",
        "legacy" + "_breakdown",
        "legacy" + "_hits",
        "legacy" + "_penalty",
        "legacy" + "_reason",
    ]
    for token in banned_tokens:
        assert token not in merged_source
    assert ("heartfc_" + "legacy" + "_gate") not in core_config_source
    assert "_last_gateway_gate" in merged_source
    assert "gateway_gate" in merged_source
    assert "gateway_block" in merged_source
    return {
        "heartflow_gateway_old_names_removed": True,
        "heartflow_gateway_runtime_names_preserved": True,
        "core_config_old_gateway_block_removed": True,
    }


def check_force_reply_generation_failure_contract() -> Dict[str, Any]:
    msg = type(
        "FakeMessage",
        (),
        {
            "processed_plain_text": "模型生成失败时不应该发送本地话术",
            "plain_text": "模型生成失败时不应该发送本地话术",
            "content": "模型生成失败时不应该发送本地话术",
            "user_id": "admin-user",
        },
    )()
    force_probe = ProactiveReactiveFlowMixin()
    assert force_probe._should_use_direct_fast_reply_generation(force_bypass=True)
    force_probe._is_admin_forced = True
    force_probe._direct_fast_reply_generation = True
    assert force_probe._should_use_direct_fast_reply_generation(force_bypass=False)
    force_probe._force_full_reply_generation = True
    assert not force_probe._should_use_direct_fast_reply_generation(force_bypass=True)
    direct_probe = ProactiveReactiveFlowMixin()
    direct_probe._direct_fast_reply_generation = True
    assert direct_probe._should_use_direct_fast_reply_generation(force_bypass=False)

    probe = type(
        "ModelGovernorProbe",
        (EnhancedInteractionCoreMixin,),
        {
            "stream_id": "force-reply-governor-regression",
            "log_prefix": "[regression]",
            "_is_admin_forced": True,
            "_has_targeted_bot_message": lambda self, messages: False,
            "_is_force_wake_admin": lambda self, incoming_batch, pinged_msg=None: True,
            "_current_hourly_proactive_reply_count": lambda self, now=None: 0,
            "_has_recent_human_activity": lambda self, window_sec=180.0: True,
        },
    )()
    probe._unanswered_bot_turns = 0
    probe._chatterbox_penalty = 0.0
    probe._consecutive_speaks = 0.0
    probe._model_large_hour_window_start = 0.0
    probe._model_large_last_ts = 0.0
    probe._model_large_proactive_hour_calls = 0
    verdict = probe._evaluate_model_governor(
        now=123.0,
        desired_level=0,
        incoming_batch=[msg],
        pinged_msg=None,
        is_proactive=False,
        source="reactive_decision",
        behavior_verdict=BehaviorGovernorVerdict(
            reply_mode="observe",
            allow_generation=False,
            model_tier="skip",
        ),
    )
    assert verdict.tier == "small"
    assert "admin_force_model_bypass" in verdict.upgrade_reason_codes
    assert "behavior_model_skip" not in verdict.upgrade_reason_codes

    flow_source = (ROOT / "src/chat/heart_flow/enhanced_modules/proactive_reactive_flow_mixin.py").read_text(
        encoding="utf-8"
    )
    interaction_source = (ROOT / "src/chat/heart_flow/enhanced_modules/interaction_core_mixin.py").read_text(
        encoding="utf-8"
    )
    loop_source = (ROOT / "src/chat/heart_flow/enhanced_modules/loop_main_driver_mixin.py").read_text(
        encoding="utf-8"
    )
    scene_source = (ROOT / "src/chat/heart_flow/enhanced_modules/scene_planner_bridge_mixin.py").read_text(
        encoding="utf-8"
    )
    coordinator_source = (ROOT / "src/chat/heart_flow/reply_coordinator.py").read_text(encoding="utf-8")
    generator_source = (ROOT / "src/plugin_system/apis/generator_api.py").read_text(encoding="utf-8")
    group_reply_source = (ROOT / "src/chat/replyer/group_generator.py").read_text(encoding="utf-8")
    private_reply_source = (ROOT / "src/chat/replyer/private_generator.py").read_text(encoding="utf-8")
    tool_executor_source = (ROOT / "src/plugin_system/core/tool_use.py").read_text(encoding="utf-8")
    shared_runtime_source = (ROOT / "src/chat/heart_flow/enhanced_modules/shared_runtime.py").read_text(encoding="utf-8")
    core_config_source = (ROOT / "config/core_config.toml").read_text(encoding="utf-8")
    core_template_source = (ROOT / "template/core_config_template.toml").read_text(encoding="utf-8")
    assert "heartfc_force_reply_generation_timeout_seconds" in flow_source
    assert "heartfc_direct_fast_reply_generation_timeout_seconds" in shared_runtime_source
    assert "tool_executor_llm_timeout_seconds" in tool_executor_source
    assert "tool_executor_llm_timeout_seconds" in core_config_source
    assert "tool_executor_llm_timeout_seconds" in core_template_source
    assert "timeout=30.0" not in tool_executor_source
    assert "heartfc_post_send_focus_audit_timeout_seconds" in interaction_source
    assert "_post_finalize_focus_audit" in interaction_source
    assert "asyncio.wait_for" in interaction_source
    assert "回复后台巡查超时" in interaction_source
    assert "direct_fast_reply_generation_timeout" in flow_source
    assert "force_reply_generation_timeout" in flow_source
    assert "self._last_flow_blocker = f\"voice回复生成失败:{generation_failure_reason}\"" in flow_source
    assert "self._mark_message_content_deferred(target_message, generation_failure_reason)" in flow_source
    assert "回复生成失败，取消发送" in flow_source
    assert "未发送本地兜底" not in flow_source
    assert "本地短兜底" not in flow_source
    assert "local_force_reply_fallback" not in flow_source
    assert "_build_forced_reply_fallback_text" not in flow_source
    assert "_build_forced_reply_fallback_response" not in flow_source
    assert "allow_local_generation_fallback" not in flow_source
    assert "heartfc_force_reply_fallback" not in flow_source
    assert "heartfc_force_reply_fallback" not in core_config_source
    assert "heartfc_force_reply_fallback" not in core_template_source
    assert "听着呢，说事。" not in core_config_source
    assert "听着呢，说事。" not in core_template_source
    assert "看到了，怎么了？" not in flow_source
    assert "我看见问题了，你具体指哪块" not in flow_source
    assert "continuity_context" in flow_source
    assert "_build_context_execution_block" in flow_source
    assert "def _decide_subject_reply_mode" in loop_source
    assert "[主体裁定/preflight]" in loop_source
    assert "_preflight_subject_mode == \"fast_reply\"" in loop_source
    assert "_preflight_subject_mode == \"deep_think\"" in loop_source
    assert "subject_mode_observe" in loop_source
    assert "[主体裁定/voice]" in loop_source
    assert "voice_conclusion.needs_upgrade = True" in loop_source
    assert "if _target_uid and not _is_direct_reply_fastlane" in loop_source
    assert "heartfc_stage25_fast_reply_timeout_seconds" in loop_source
    assert "1.2 if _is_direct_reply_fastlane else 6.0" in loop_source
    assert "跳过阶段2.5b深度集成" in loop_source
    assert "跳过记忆预取" in loop_source
    assert "直接快回链路" in loop_source
    assert "self._direct_fast_reply_generation = bool(_is_direct_reply_fastlane)" in loop_source
    assert "直接快回，不等待前台感知/独白" in loop_source
    assert "direct_fastlane_synthetic" in loop_source
    assert "使用缓存关系快照，不等待关系计算" in loop_source
    assert "直接快回跳过维度网关" in loop_source
    assert "src={_gateway_src}" in loop_source
    assert "_refresh_direct_fastlane_inner_state" in loop_source
    assert "_queue_direct_fastlane_inner_state_refresh" in loop_source
    assert "回复成功后后台补感知/独白已排队" in loop_source
    assert "self._refresh_direct_fastlane_inner_state(decision_messages, ambient_info, now)" not in loop_source
    assert "_voice_reply_timeout = _direct_fast_reply_total_timeout()" in loop_source
    assert "_direct_fast_reply_generation_timeout" in flow_source
    assert "def _model_response_timeout_budget" in shared_runtime_source
    assert "heartfc_direct_fast_reply_generation_timeout_seconds = 20.0" in core_config_source
    assert "heartfc_direct_fast_reply_generation_timeout_seconds = 20.0" in core_template_source
    assert "direct_fast_reply_total_timeout" in loop_source
    assert "直接快回整体超时" in loop_source
    assert "direct_fast_deadline = time.monotonic() + direct_fast_timeout" in flow_source
    assert (
        "if direct_fast_reply_generation:\n"
        "                direct_fast_timeout = max(0.5, _rt_float(\"heartfc_direct_fast_reply_generation_timeout_seconds\"))\n"
        "                direct_fast_deadline = time.monotonic() + direct_fast_timeout"
    ) not in flow_source
    assert "fast_path_deadline=direct_fast_deadline" in flow_source
    assert "直接快回预算已耗尽" in flow_source
    assert "来源=voice_driven_fast" in flow_source
    assert "def _append_direct_fast_soul_state" in flow_source
    assert "self._append_direct_fast_soul_state(extra_info_parts, relation_view)" in flow_source
    assert "def _append_direct_fast_relation_style" in flow_source
    assert "self._append_direct_fast_relation_style(extra_info_parts, relation_view)" in flow_source
    assert (
        "self._append_direct_fast_relation_style(extra_info_parts, relation_view)\n"
        "                self._append_direct_fast_soul_state(extra_info_parts, relation_view)"
    ) in flow_source
    assert "[当前对象关系]" in flow_source
    assert "[直接快回关系语气]" in flow_source
    assert "不要回“好的，你说”“你说”“继续说”“收到”“请问”“当然”" in flow_source
    assert "relation_view = self._inject_realtime_emotion(cached_relation)" in flow_source
    assert "self._inject_fallback_soul_state(extra_info_parts)" not in flow_source
    assert "if not direct_fast_reply_generation:\n                append_reply_style(extra_info_parts, style_route)" in flow_source
    assert "if direct_fast_reply_generation:\n                extra_info = \"\\n\".join" in flow_source
    assert "heartfc_direct_fastlane_background_voice_timeout_seconds" in loop_source
    assert "direct_fastlane_background" in loop_source
    assert "get_intention_pool().ingest_voice_verdict" in loop_source
    assert '_gateway_src = "admin_force" if _is_admin_force_wake else "direct_fastlane"' in loop_source
    assert "heartfc_background_proactive_min_silence_seconds" in flow_source
    assert "heartfc_background_proactive_cooldown_seconds" in flow_source
    assert "heartfc_background_proactive_success_cooldown_min_seconds" in flow_source
    assert "if gw_ctx.admin_force" in scene_source
    assert "跳过被动策略规划" in scene_source
    assert "_should_use_direct_fast_reply_generation" in flow_source
    assert "return bool(getattr(self, \"_direct_fast_reply_generation\", False) or force_bypass)" in flow_source
    assert "直接快回跳过自省深检" in flow_source
    assert "forced_reply_generation = bool(force_bypass or getattr(self, \"_is_admin_forced\", False))" in flow_source
    assert "direct_fast_reply_generation = self._should_use_direct_fast_reply_generation" in flow_source
    assert "fast_path=direct_fast_reply_generation" in flow_source
    assert "enable_chinese_typo=not direct_fast_reply_generation" in flow_source
    assert "强制回复完整生成预算" in flow_source
    assert "强制回复完整生成超时" in flow_source
    assert "取消发送" in flow_source
    assert "[直接快回]" in flow_source
    assert "直接快回跳过慢上下文构建" in flow_source
    assert "target_text[:120]" in flow_source
    assert "if direct_fast_reply_generation:\n                extra_info_parts.append(\"[直接快回]" in flow_source
    assert "self_reference_parts = self._build_self_reference_parts(target_message)\n                if self_reference_parts.get(\"self_memory\")" not in flow_source
    assert "fast_path: bool = False" in coordinator_source
    assert "fast_path: bool = False" in generator_source
    assert "fast_path_deadline: Optional[float] = None" in coordinator_source
    assert "fast_path_deadline: Optional[float] = None" in generator_source
    assert "fast_path_deadline: Optional[float] = None" in group_reply_source
    assert "fast_path_deadline: Optional[float] = None" in private_reply_source
    assert "直接快回跳过动作与工具准备" in coordinator_source
    assert "直接快回跳过临时动作记录" in coordinator_source
    assert "enable_tool=(False if fast_path else global_config.tool.enable_tool)" in coordinator_source
    assert "if fast_path:\n            logger.debug(f\"[回复协调] {channel_id} 直接快回跳过临时动作记录\")" in coordinator_source
    assert "self._tool_executor = None" in group_reply_source
    assert "self._tool_executor = None" in private_reply_source
    assert "self.tool_executor" not in group_reply_source
    assert "self.tool_executor" not in private_reply_source
    assert "if fast_path:\n                person_name = self._message_sender_name(reply_message) or user_id" in group_reply_source
    assert "chat_talking_prompt_fast = \"\"" in group_reply_source
    assert "fast_low_info_text = text_part if has_text else target" in group_reply_source
    assert "low_info_guard = self._build_low_info_input_guard(fast_low_info_text)" in group_reply_source
    assert "low_info_input=bool(low_info_guard)" in group_reply_source
    assert "identity=fast_identity_hint" in group_reply_source
    assert "禁回: 好的你说、你说、继续说、收到、请问、当然" in group_reply_source
    assert "extra_info_block = fast_extra_info" in group_reply_source
    assert "不要复读、不要照抄、不要同音改写对方原句" in group_reply_source
    assert "不要把对方词尾改成“我/你”当回复" in group_reply_source
    assert "信息不完整就自然追问或轻接一句" in group_reply_source
    assert "_looks_reception_fast_reply" in group_reply_source
    assert "fast_path_reception_reply" in group_reply_source
    assert "绝对不要回“好的，你说”“你说”“继续说”“收到”“请问”“当然”" in group_reply_source
    assert "DefaultReplyer._looks_reception_fast_reply(\"好的，你说。\") is True" not in group_reply_source
    assert "[reply_postprocess] 跳过自然化错字和二次改写" in group_reply_source
    assert "_apply_adaptive_recall(content" not in group_reply_source
    assert "await self._apply_rewrite_if_needed(" not in group_reply_source
    assert "跳过补充回复判断" in group_reply_source
    assert "管理员强制快回" not in group_reply_source
    assert "直接快回通道" in group_reply_source
    assert "fast_path=fast_path" in private_reply_source
    assert "if fast_path:\n            fast_low_info_text = text_part if has_text else target" in private_reply_source
    assert "low_info_guard = self._build_low_info_input_guard(fast_low_info_text)" in private_reply_source
    assert "low_info_input=bool(low_info_guard)" in private_reply_source
    assert "identity=fast_identity_hint" in private_reply_source
    assert "禁回: 好的你说、你说、继续说、收到、请问、当然" in private_reply_source
    assert "extra_info_block = fast_extra_info" in private_reply_source
    assert "不要复读、不要照抄、不要同音改写对方原句" in private_reply_source
    assert "_looks_reception_fast_reply" in private_reply_source
    assert "fast_path_reception_reply" in private_reply_source
    assert "绝对不要回“好的，你说”“你说”“继续说”“收到”“请问”“当然”" in private_reply_source
    context_block_source = (ROOT / "src/chat/replyer/context_block_builder.py").read_text(encoding="utf-8")
    assert "[当前对象关系]" in context_block_source
    assert "[直接快回关系语气]" in context_block_source
    assert "厌烦=" in context_block_source
    assert "压力=" in context_block_source
    reply_prompt_source = (ROOT / "src/chat/replyer/prompt/replyer_prompt.py").read_text(encoding="utf-8")
    private_prompt_source = (ROOT / "src/chat/replyer/prompt/replyer_private_prompt.py").read_text(encoding="utf-8")
    assert "\"好感\", \"信任\", \"厌烦\", \"压力\", \"禁回\"" in reply_prompt_source
    assert "priority_markers = (\"关系\", \"好感\", \"信任\", \"厌烦\", \"压力\", \"禁回\"" in private_prompt_source
    from src.chat.replyer.group_generator import DefaultReplyer
    from src.chat.replyer.private_generator import PrivateReplyer

    assert DefaultReplyer._looks_reception_fast_reply("好的，你说。") is True
    assert DefaultReplyer._looks_reception_fast_reply("嗯你说") is True
    assert DefaultReplyer._looks_reception_fast_reply("有点没看懂") is False
    assert PrivateReplyer._looks_reception_fast_reply("继续说吧") is True
    assert PrivateReplyer._looks_reception_fast_reply("啥意思") is False
    assert "_cond_tasks" not in group_reply_source
    assert "_reply_context_tasks" in group_reply_source
    assert "asyncio.gather(*_reply_context_tasks)" in group_reply_source
    assert "条件任务并行超时(30s)" not in group_reply_source
    return {
        "local_force_reply_fallback_removed": True,
        "force_reply_generation_failure_aborts": True,
        "force_reply_fallback_config_removed": True,
        "admin_model_governor_bypass": True,
        "stale_voice_cache_cleared": True,
        "direct_fast_reply_path": True,
        "force_reply_uses_direct_fast_path": True,
        "explicit_full_generation_override": True,
        "direct_fast_deadline_enforced": True,
        "direct_fast_tool_chain_skipped": True,
        "direct_fast_db_history_skipped": True,
        "direct_fast_soul_state_uses_cache": True,
        "direct_fast_budget_starts_at_generation": True,
        "direct_fast_temp_record_skipped": True,
        "direct_fast_budget_matches_model_timeout": True,
        "direct_fast_low_info_guard_restored": True,
        "direct_fast_relation_style_restored": True,
        "direct_fast_reception_reply_blocked": True,
    }


def check_webui_contract() -> Dict[str, Any]:
    dashboard_source = (ROOT / "web/dashboard/src/components/EmotionDashboard/index.tsx").read_text(encoding="utf-8")
    dashboard_style = (ROOT / "web/dashboard/src/components/EmotionDashboard/styles.css").read_text(encoding="utf-8")
    assert "fallbackPollTimerRef" not in dashboard_source
    assert "pollMonitorFallback" not in dashboard_source
    assert "config-scope" not in dashboard_source
    assert "配置分级" not in dashboard_source
    assert "时间线状态" in dashboard_source
    assert "时间线窗口、睡眠债、困意和熬夜压力" in dashboard_source
    assert "当前时段" in dashboard_source
    assert "机制窗口" in dashboard_source or "mechanism_windows" in dashboard_source
    assert "压力分解" in dashboard_source
    assert "下午 / 傍晚 / 晚上 / 半夜 / 凌晨 / 清晨" not in dashboard_source
    assert "社交窗口" in dashboard_source
    assert "睡眠窗口" in dashboard_source
    assert "状态同步" in dashboard_source
    assert "function resolveTimeBand" not in dashboard_source
    assert "等待后端时间线" in dashboard_source
    assert "夜间影响" in dashboard_source
    assert "昼夜节律" not in dashboard_source
    assert "Promise.allSettled" not in dashboard_source
    assert "void loadMonitorOverview();" in dashboard_source
    assert "void loadConfigScope();" not in dashboard_source
    assert "has" + "EverConnectedRef" not in dashboard_source
    assert "new " + "WebSocket" not in dashboard_source
    assert "Web" + "Socket" not in dashboard_source
    assert REMOVED_STATE_MONITOR_WS_PATH not in dashboard_source
    assert "/api/heartflow/monitor/live" in dashboard_source
    assert "/api/heartflow/monitor/" in dashboard_source
    assert "monitorUrl(" in dashboard_source
    assert 'if (!selectedChannel) {' in dashboard_source
    assert 'const displayConnectionState = selectedChannel ? (usingCachedSnapshot ? "cached" : connectionState) : "waiting";' in dashboard_source
    assert 'window.addEventListener("blur"' not in dashboard_source
    assert 'document.addEventListener("visibilitychange"' not in dashboard_source
    assert 'window.addEventListener("focus"' not in dashboard_source
    assert 'waiting: "等待会话"' in dashboard_source
    assert 'connecting: "监听中"' in dashboard_source
    assert 'reconnecting: "续连中"' in dashboard_source
    assert 'live: "API实时同步"' in dashboard_source
    assert 'cached: "缓存快速恢复"' in dashboard_source
    assert "来自最近快照" in dashboard_source
    assert "实时通道暂不可用，轮询同步失败" not in dashboard_source
    assert 'polling: "补偿同步"' not in dashboard_source
    assert ".live-pill.is-polling" not in dashboard_style
    assert ".live-pill.is-waiting" in dashboard_style
    assert ".live-pill.is-cached" in dashboard_style
    assert "下一步发言概率" in dashboard_source
    assert "后端当前时间" in dashboard_source
    assert "formatClockSecond" in dashboard_source
    assert "durationSecond(syncAgeSeconds)" in dashboard_source
    assert "formatClockMs" not in dashboard_source
    assert "preciseDuration" not in dashboard_source
    assert "心理负荷" in dashboard_source
    assert "创伤负荷" in dashboard_source
    assert "表层伪装" in dashboard_source
    assert "低于触发阈值的心理值也显示原始数值" in dashboard_source
    assert "Number(predictionPercent).toFixed(1)" in dashboard_source
    assert "requestAnimationFrame" in dashboard_source
    assert "timestamp - last" not in dashboard_source
    assert "state_delta" not in dashboard_source
    assert "state_heartbeat" not in dashboard_source
    assert 'type: "refresh"' not in dashboard_source
    assert "requires_refresh" not in dashboard_source
    assert "liveEtaLabel" in dashboard_source
    assert "function expressionStyleLabel" in dashboard_source
    assert "function circadianExpressionStyle" in dashboard_source
    assert "irritable: \"被扰烦躁\"" in dashboard_source
    assert "soft_night: \"夜间放轻\"" in dashboard_source
    assert "circadianExpressionStyle(packet, circadianDetail)" in dashboard_source
    assert "<strong>{circadianDetail?.expression_style_label ?? \"正常\"}</strong>" not in dashboard_source

    heartflow_router_source = (ROOT / "src/webui/routers/heartflow.py").read_text(encoding="utf-8")
    router_registry_source = (ROOT / "src/webui/routers/router_registry.py").read_text(encoding="utf-8")
    assert "asyncio.to_thread(build_config_scope_snapshot)" not in heartflow_router_source
    assert "状态页配置分级构建超时" not in heartflow_router_source
    assert "_config_scope_fallback" not in heartflow_router_source
    assert "wait_for_update" in heartflow_router_source
    assert '"/monitor/live"' in heartflow_router_source
    assert '"/monitor/{channel_id}/live"' in heartflow_router_source
    assert REMOVED_STATE_MONITOR_ROUTER_NAME not in router_registry_source
    assert not (ROOT / "src/webui/routers/websocket/state_monitor.py").exists()

    client = TestClient(create_app())
    dashboard = client.get("/dashboard")
    monitor = client.get("/api/heartflow/monitor")
    monitor_live = client.get("/api/heartflow/monitor/live?timeout=0.1")
    config_scope = client.get("/api/heartflow/config-scope")
    assert dashboard.status_code == 200
    assert monitor.status_code == 200
    assert monitor_live.status_code == 200
    assert config_scope.status_code == 404
    channels = monitor.json().get("monitor", {}).get("channels", [])
    checked_channel = False
    if channels:
        channel_id = channels[0]["channel_id"]
        packet = client.get(f"/api/heartflow/monitor/{channel_id}").json().get("monitor", {})
        impacts = packet.get("presentation", {}).get("participant_impacts", [])
        for item in impacts:
            for key in (
                "chaos_load",
                "mask_load",
                "current_mood_hint",
                "is_current_target",
                "in_current_scene",
                "recent_speaker",
                "recent_targeted_interaction",
                "source_scope",
            ):
                assert key in item, key
        checked_channel = True
    return {
        "dashboard": dashboard.status_code,
        "monitor": monitor.status_code,
        "monitor_live": monitor_live.status_code,
        "channel_checked": checked_channel,
        "startup_http_fallback_polling_removed": True,
        "config_scope_removed_from_status_page": True,
        "state_monitor_websocket_removed": True,
    }


def check_runtime_state_hub_contract() -> Dict[str, Any]:
    import asyncio

    from src.webui.services.runtime_state_hub import get_runtime_state_hub

    async def _exercise() -> Dict[str, Any]:
        channel_id = "regression-runtime-hub"
        empty_channel_id = "regression-runtime-hub-empty"
        hub = get_runtime_state_hub()
        empty_delta = await hub.update_path(
            empty_channel_id,
            module="regression",
            path="presentation.resource_detail.chat_percent",
            value=0.5,
            reason="no_partial_seed",
        )
        empty_cached = await hub.get_snapshot(empty_channel_id)
        assert empty_delta is None
        assert empty_cached is None
        queue = await hub.subscribe(channel_id)
        try:
            snapshot = await hub.set_snapshot(
                channel_id,
                {
                    "channel_id": channel_id,
                    "presentation": {"resource_detail": {"chat_percent": 0.731}},
                    "prediction": {"eta_seconds": 720},
                },
                reason="regression",
            )
            first = await asyncio.wait_for(queue.get(), timeout=1.0)
            delta = await hub.update_path(
                channel_id,
                module="regression",
                path="presentation.resource_detail.chat_percent",
                value=0.735,
                reason="value_changed",
            )
            second = await asyncio.wait_for(queue.get(), timeout=1.0)
            cached = await hub.get_snapshot(channel_id)
            assert snapshot["state_version"] >= 1
            assert first["type"] == "state_snapshot"
            assert delta is not None
            assert second["type"] == "state_delta"
            assert second["path"] == "presentation.resource_detail.chat_percent"
            assert second["new"] == 0.735
            assert cached is not None
            assert cached["presentation"]["resource_detail"]["chat_percent"] == 0.735
            waiter = asyncio.create_task(
                hub.wait_for_update(channel_id, after_version=int(cached["state_version"]), timeout_seconds=1.0)
            )
            await asyncio.sleep(0)
            refreshed = await hub.set_snapshot(
                channel_id,
                {
                    "channel_id": channel_id,
                    "presentation": {"resource_detail": {"chat_percent": 0.812}},
                    "prediction": {"eta_seconds": 360},
                },
                reason="full_refresh",
            )
            third = await asyncio.wait_for(queue.get(), timeout=1.0)
            waited = await waiter
            timed_out = await hub.wait_for_update(
                channel_id,
                after_version=int(refreshed["state_version"]),
                timeout_seconds=0.1,
            )
            assert refreshed["state_version"] > snapshot["state_version"]
            assert third["type"] == "state_snapshot"
            assert third["data"]["presentation"]["resource_detail"]["chat_percent"] == 0.812
            assert waited is not None
            assert waited["type"] == "state_snapshot"
            assert timed_out is None
            assert "requires_refresh" not in third
            return {
                "snapshot": True,
                "delta": True,
                "no_partial_seed": True,
                "event_driven_snapshot": True,
                "http_wait_for_update": True,
                "delta_rebuilds_monitor_packet": True,
            }
        finally:
            await hub.unsubscribe(queue, channel_id)

    hub_source = (ROOT / "src/webui/services/runtime_state_hub.py").read_text(encoding="utf-8")
    heartflow_router_source = (ROOT / "src/webui/routers/heartflow.py").read_text(encoding="utf-8")
    state_monitor_source = (ROOT / "src/webui/services/state_monitor.py").read_text(encoding="utf-8")
    core_config_source = (ROOT / "config/core_config.toml").read_text(encoding="utf-8")
    core_template_source = (ROOT / "template/core_config_template.toml").read_text(encoding="utf-8")
    assert not (ROOT / "src/webui/routers/websocket/state_monitor.py").exists()
    assert "class RuntimeStateHub" in hub_source
    assert "emit_runtime_delta" in hub_source
    assert "wait_for_update" in hub_source
    assert "MONITOR_OVERVIEW_CHANNEL_ID" in hub_source
    assert "await build_monitor_overview() if int(after_version or 0) <= 0" in heartflow_router_source
    assert "live_chats = list_heartfc_chats()" in heartflow_router_source
    assert "startswith(\"monitor_\")" in heartflow_router_source
    assert "monitor = await build_channel_monitor_state(channel_id)" in heartflow_router_source
    assert "if not isinstance(existing, dict) or not existing:" in hub_source
    assert "is_channel_monitor_packet" in heartflow_router_source
    assert "warm_monitor_snapshots" in state_monitor_source
    assert "_MONITOR_CACHE_OVERVIEW_SLOT" in state_monitor_source
    assert "_load_cached_channel_packet" in state_monitor_source
    assert "requires_refresh" not in hub_source
    assert "def heartbeat(" not in hub_source
    assert "state_monitor_default_interval_seconds" not in core_config_source
    assert "state_monitor_min_interval_seconds" not in core_config_source
    assert "state_monitor_max_interval_seconds" not in core_config_source
    assert "state_monitor_default_interval_seconds" not in core_template_source
    assert "state_monitor_min_interval_seconds" not in core_template_source
    assert "state_monitor_max_interval_seconds" not in core_template_source
    return asyncio.run(_exercise())


def main() -> None:
    results = {
        "source": check_source_contract(),
        "remote_telemetry_removed": check_remote_telemetry_removed_contract(),
        "prediction_runtime": check_prediction_runtime_contract(),
        "memory_autonomy": check_memory_and_autonomy_contract(),
        "deprecated_small_module_cleanup": check_deprecated_small_module_cleanup_contract(),
        "social_compat_bridge_cleanup": check_social_compat_bridge_cleanup_contract(),
        "participants": check_participant_contract(),
        "world_snapshot_relation": check_world_snapshot_relation_contract(),
        "dynamic_personal_impression": check_dynamic_personal_impression_contract(),
        "content_state_scope": check_content_state_scope_contract(),
        "force_guard": check_force_guard_contract(),
        "night_cycle_persistence": check_night_cycle_persistence_contract(),
        "night_soul_prompt": check_night_soul_prompt_contract(),
        "inner_voice_low_info_tone": check_inner_voice_low_info_tone_contract(),
        "inner_voice_single_message_dedupe": check_inner_voice_single_message_dedupe_contract(),
        "webui_internal_chat_disabled": check_webui_internal_chat_disabled_contract(),
        "huoli_naming": check_huoli_naming_contract(),
        "admin_identity": check_admin_identity_not_relationship_contract(),
        "db_backed_json_storage": check_db_backed_json_storage_contract(),
        "memoir_channel_key": check_memoir_channel_key_contract(),
        "heartflow_startup_prewarm": check_heartflow_startup_no_channel_prewarm_contract(),
        "heartflow_gateway_naming": check_heartflow_gateway_naming_contract(),
        "force_reply_generation_failure": check_force_reply_generation_failure_contract(),
        "statusbar_export": check_statusbar_export_contract(),
        "monitor_overview": check_monitor_overview_contract(),
        "night_status_labels": check_night_status_label_contract(),
        "runtime_state_hub": check_runtime_state_hub_contract(),
        "webui": check_webui_contract(),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

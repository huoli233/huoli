# ruff: noqa: E402
import json
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
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
    missing = [stage for stage in REQUIRED_EARLY_STAGES if stage not in loop_source]
    assert not missing, f"缺少 early gate stage: {missing}"
    assert "_store_gate_runtime(" in loop_source, "主循环未调用 gate_runtime 写入层"
    assert "pre_execution" in prediction_source, "预测引擎未区分前置裁定"
    assert "execution_has_verdict" in prediction_source, "预测引擎未显式区分 execution_runtime"
    return {"early_stages": len(REQUIRED_EARLY_STAGES)}


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
    assert "is_night=clock_is_night" in night_source
    assert "self._state.in_night_mode = clock_is_night" in night_source
    assert "self._state.last_stimulus_at = now" in night_source
    assert '"late_night", "半夜", "睡眠窗口开始"' in night_source
    assert '"midnight", "凌晨", "凌晨反思窗口"' in night_source
    assert "_ncs.evaluate(" in runtime_source
    assert "record_overnight_activity(\"peek\"" in runtime_source
    assert "record_overnight_activity(\"interrupt\"" in runtime_source
    assert '"pressure_breakdown"' in exporter_source
    assert '"mechanism_windows"' in exporter_source
    assert '"last_evaluated_at"' in exporter_source
    feedback_source = (ROOT / "src/chat/heart_flow/enhanced_modules/loop_resource_feedback_mixin.py").read_text(
        encoding="utf-8"
    )
    bot_lifecycle_source = (ROOT / "src/chat/heart_flow/enhanced_modules/scene_bot_lifecycle_mixin.py").read_text(
        encoding="utf-8"
    )
    assert "record_overnight_activity" in feedback_source and '"think"' in feedback_source
    assert "record_overnight_activity" in feedback_source and '"peek"' in feedback_source
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
            }
        }
    )
    assert detail["phase_label"] == "凌晨阶段"
    assert detail["time_band_label"] == "凌晨"
    assert detail["mechanism_windows"]["midnight_reflect"]["active"] is True
    assert detail["last_evaluated_at"] == 1778258282.0
    assert detail["pressure_breakdown"]["interrupt_count"] == 2
    assert detail["reply_quota_label"] == "1/2"
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
    return {
        "night_labels": True,
        "midnight_phase_label": detail["phase_label"],
        "time_band_label": exported["time_band_label"],
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
        {"processed_plain_text": "我的刀盾", "user_id": "admin-user"},
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
    assert "and (not _admin_force_active or _autonomy_hard_skip)" in loop_source
    assert "管理员强制唤醒受硬保护限制" in loop_source
    assert "_action in (\"deep_sleep\", \"hard_block\")" in loop_source
    assert "_action == \"sleep_resist\" and not self._is_force_wake_admin" in loop_source
    assert "voice_conclusion = None if _is_admin_msg else self._cached_voice" in loop_source

    hard_block_idx = lifecycle_source.index("if not _ncs_cap.evaluate_sleep_reply_budget()")
    deep_sleep_idx = lifecycle_source.index("if _composite >= _collapse * 1.1")
    admin_wake_idx = lifecycle_source.index("if _admin_force_wake:")
    assert hard_block_idx < admin_wake_idx
    assert deep_sleep_idx < admin_wake_idx
    return {
        "admin_content_hard_guard": True,
        "admin_soft_guard_bypass": True,
        "night_hard_block_before_admin": True,
    }


def check_force_reply_generation_fallback_contract() -> Dict[str, Any]:
    msg = type(
        "FakeMessage",
        (),
        {
            "processed_plain_text": "我的刀盾",
            "plain_text": "我的刀盾",
            "content": "我的刀盾",
            "user_id": "admin-user",
        },
    )()
    fallback_response = ProactiveReactiveFlowMixin()._build_forced_reply_fallback_response(
        msg,
        "force_reply_generation_timeout",
    )
    assert fallback_response.model == "local_force_reply_fallback"
    assert fallback_response.content == "看到了，怎么了？"
    assert fallback_response.processed_output == ["看到了，怎么了？"]
    assert fallback_response.reply_set is not None
    assert fallback_response.reply_set.reply_data[0].content == "看到了，怎么了？"
    assert fallback_response.timing["fallback_reason"] == "force_reply_generation_timeout"
    direct_fallback = ProactiveReactiveFlowMixin()._build_forced_reply_fallback_response(
        msg,
        "direct_fast_reply_generation_timeout",
    )
    assert direct_fallback.timing["fallback_reason"] == "direct_fast_reply_generation_timeout"

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
    loop_source = (ROOT / "src/chat/heart_flow/enhanced_modules/loop_main_driver_mixin.py").read_text(
        encoding="utf-8"
    )
    scene_source = (ROOT / "src/chat/heart_flow/enhanced_modules/scene_planner_bridge_mixin.py").read_text(
        encoding="utf-8"
    )
    coordinator_source = (ROOT / "src/chat/heart_flow/reply_coordinator.py").read_text(encoding="utf-8")
    generator_source = (ROOT / "src/plugin_system/apis/generator_api.py").read_text(encoding="utf-8")
    group_reply_source = (ROOT / "src/chat/replyer/group_generator.py").read_text(encoding="utf-8")
    assert "heartfc_force_reply_generation_timeout_seconds" in flow_source
    assert "heartfc_direct_fast_reply_generation_timeout_seconds" in flow_source
    assert "direct_fast_reply_generation_timeout" in flow_source
    assert "强制回复生成失败，已启用本地短兜底" in flow_source
    assert "_is_direct_reply_fastlane = bool(_is_admin_force_wake or getattr(self, \"_cached_targeted_to_bot\", False))" in loop_source
    assert "if _target_uid and not _is_direct_reply_fastlane" in loop_source
    assert "heartfc_stage25_fast_reply_timeout_seconds" in loop_source
    assert "1.2 if _is_direct_reply_fastlane else 6.0" in loop_source
    assert "跳过阶段2.5b深度集成" in loop_source
    assert "跳过记忆预取" in loop_source
    assert "直接快回链路" in loop_source
    assert "_refresh_direct_fastlane_inner_state" in loop_source
    assert "heartfc_direct_fastlane_background_voice_timeout_seconds" in loop_source
    assert "direct_fastlane_background" in loop_source
    assert "get_intention_pool().ingest_voice_verdict" in loop_source
    assert "src=admin_force" in loop_source
    assert "heartfc_background_proactive_min_silence_seconds" in flow_source
    assert "heartfc_background_proactive_cooldown_seconds" in flow_source
    assert "heartfc_background_proactive_success_cooldown_min_seconds" in flow_source
    assert "if gw_ctx.admin_force" in scene_source
    assert "跳过被动策略规划" in scene_source
    assert "fast_path=force_generation_fallback" in flow_source
    assert "enable_chinese_typo=not force_generation_fallback" in flow_source
    assert "[直接快回]" in flow_source
    assert "fast_path: bool = False" in coordinator_source
    assert "fast_path: bool = False" in generator_source
    assert "if fast_path:" in group_reply_source
    assert "跳过自然化错字和二次改写" in group_reply_source
    assert "跳过补充回复判断" in group_reply_source
    return {
        "local_force_reply_fallback": True,
        "admin_model_governor_bypass": True,
        "stale_voice_cache_cleared": True,
        "direct_fast_reply_path": True,
    }


def check_webui_contract() -> Dict[str, Any]:
    dashboard_source = (ROOT / "web/dashboard/src/components/EmotionDashboard/index.tsx").read_text(encoding="utf-8")
    dashboard_style = (ROOT / "web/dashboard/src/components/EmotionDashboard/styles.css").read_text(encoding="utf-8")
    assert "fallbackPollTimerRef" in dashboard_source
    assert "pollMonitorFallback" in dashboard_source
    assert "夜间状态" in dashboard_source
    assert "夜间机制窗口、睡眠债、困意和熬夜压力" in dashboard_source
    assert "当前时段" in dashboard_source
    assert "机制窗口" in dashboard_source or "mechanism_windows" in dashboard_source
    assert "压力分解" in dashboard_source
    assert "下午 / 傍晚 / 晚上 / 半夜 / 凌晨 / 清晨" not in dashboard_source
    assert "社交窗口" in dashboard_source
    assert "睡眠窗口" in dashboard_source
    assert "状态同步" in dashboard_source
    assert "function resolveTimeBand" in dashboard_source
    assert 'return { hour, label: "凌晨", description: "凌晨反思窗口" }' in dashboard_source
    assert "夜间影响" in dashboard_source
    assert "昼夜节律" not in dashboard_source
    assert "Promise.allSettled" not in dashboard_source
    assert "void loadMonitorOverview();" in dashboard_source
    assert "void loadConfigScope();" in dashboard_source
    assert "hasEverConnectedRef" in dashboard_source
    assert "const shouldSyncRealtime = true;" in dashboard_source
    assert 'const displayConnectionState = "live";' in dashboard_source
    assert 'window.addEventListener("blur"' not in dashboard_source
    assert 'connecting: "实时同步"' in dashboard_source
    assert 'reconnecting: "实时同步"' in dashboard_source
    assert "实时通道暂不可用，轮询同步失败" not in dashboard_source
    assert 'polling: "实时同步"' in dashboard_source
    assert ".live-pill.is-polling" in dashboard_style

    heartflow_router_source = (ROOT / "src/webui/routers/heartflow.py").read_text(encoding="utf-8")
    assert "asyncio.to_thread(build_config_scope_snapshot)" in heartflow_router_source
    assert "timeout=2.0" in heartflow_router_source
    assert "_config_scope_fallback" in heartflow_router_source

    client = TestClient(create_app())
    dashboard = client.get("/dashboard")
    monitor = client.get("/api/heartflow/monitor")
    config_scope = client.get("/api/heartflow/config-scope")
    assert dashboard.status_code == 200
    assert monitor.status_code == 200
    assert config_scope.status_code == 200
    assert "config_scope" in config_scope.json()
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
        "channel_checked": checked_channel,
        "http_fallback_polling": True,
    }


def main() -> None:
    results = {
        "source": check_source_contract(),
        "prediction_runtime": check_prediction_runtime_contract(),
        "memory_autonomy": check_memory_and_autonomy_contract(),
        "participants": check_participant_contract(),
        "world_snapshot_relation": check_world_snapshot_relation_contract(),
        "dynamic_personal_impression": check_dynamic_personal_impression_contract(),
        "content_state_scope": check_content_state_scope_contract(),
        "force_guard": check_force_guard_contract(),
        "force_reply_generation_fallback": check_force_reply_generation_fallback_contract(),
        "statusbar_export": check_statusbar_export_contract(),
        "monitor_overview": check_monitor_overview_contract(),
        "night_status_labels": check_night_status_label_contract(),
        "webui": check_webui_contract(),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

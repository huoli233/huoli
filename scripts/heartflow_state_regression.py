# ruff: noqa: E402
import json
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat.heart_flow.speak_prediction_engine import SpeakPredictionEngine
from src.chat.proactive.perception_engine import _RelationGauge
from src.core.world_snapshot import (
    TargetUserState,
    WorldSnapshot,
    get_relation_number,
    _reconcile_overlapping_values,
    _set_relation_value,
)
from src.webui.app import create_app
from src.webui.services.state_monitor import _build_participant_impacts


REQUIRED_EARLY_STAGES = [
    "self_echo_gate",
    "short_batch_skip",
    "watch_gate_blackout",
    "peek_gate_observe",
    "night_gate_skip",
    "autonomy_guard_skip",
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



def check_webui_contract() -> Dict[str, Any]:
    client = TestClient(create_app())
    dashboard = client.get("/dashboard")
    monitor = client.get("/api/heartflow/monitor")
    assert dashboard.status_code == 200
    assert monitor.status_code == 200
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
    return {"dashboard": dashboard.status_code, "monitor": monitor.status_code, "channel_checked": checked_channel}


def main() -> None:
    results = {
        "source": check_source_contract(),
        "prediction_runtime": check_prediction_runtime_contract(),
        "memory_autonomy": check_memory_and_autonomy_contract(),
        "participants": check_participant_contract(),
        "world_snapshot_relation": check_world_snapshot_relation_contract(),
        "webui": check_webui_contract(),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

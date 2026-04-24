# ruff: noqa: E402
import json
import os
import sys
import tempfile
from pathlib import Path
from time import sleep

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.core_config_engine import CoreSettingsHub


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _config(value: float) -> str:
    return f"""
[proactive_schedule]
min_cooldown_seconds = 60.0

[message_processor]
hippocampus_min_importance = 0.2

[profile_mapping.base_parameters.defaults]
enabled = true
timeout_seconds = 30.0

[profile_mapping.semantic_domains.memory]
short_term_limit = 64
importance_floor = 0.2
overload_ratio_threshold = 0.76
reactivation_cooldown_seconds = 88.0

[profile_mapping.semantic_domains.memory_capacity]
review_interval_seconds = 333

[profile_mapping.semantic_domains.memory_capacity.max_count_by_tier]
working = 222

[profile_mapping.semantic_domains.memory_retrieval]
rrf_k = 77

[profile_mapping.semantic_domains.memory_decay]
forget_threshold = 0.22

[profile_mapping.semantic_domains.memory_dedup]
similarity_threshold = 0.88

[profile_mapping.semantic_domains.context]
reply_context_max_tokens = 96
extra_info_max_tokens = 72

[profile_mapping.semantic_domains.model_routing]
model_cooldown_seconds = 77.0
model_fail_threshold = 4
disable_thinking_for = ["qwen"]

[profile_mapping.semantic_domains.schedule]
proactive_reply_cooldown_seconds = 45.0
task_cancel_timeout_seconds = 7.0
telemetry_heartbeat_interval_seconds = 123.0
expression_learning_interval_seconds = 456.0

[profile_mapping.semantic_domains.phase_timing]
max_idle_wait_seconds = 99
phase_idle_timeout_seconds = 222.0

[profile_mapping.semantic_domains.proactive_decider]
activation_bar = 0.61
max_pending_events = 12

[profile_mapping.semantic_domains.heartflow_decision]
heated_pulse_limit = 9

[profile_mapping.semantic_domains.module_switches]
heartflow_enabled = true

[profile_mapping.semantic_domains.heartflow_runtime]
initiative_probability = 0.44
focus_channels = ["demo"]

[profile_mapping.semantic_domains.inner_voice]
mention_min_desire = 7
private_chat_min_desire = 6
resource_penalty_max = 4

[profile_mapping.semantic_domains.inner_voice.time_bands.deep_night]
start = 2
end = 6
adjustment = -3

[profile_mapping.semantic_domains.inner_voice.waiting_thoughts]
short = ["稍等"]

[profile_mapping.semantic_domains.frequency_control]
baseline_multiplier = 1.2
skip_limit = 7
tick_interval_seconds = 4.0

[profile_mapping.semantic_domains.global_shield]
default_duration = 321.0

[profile_mapping.semantic_domains.harassment_detection]
block_threshold = 0.66

[profile_mapping.semantic_domains.injection_detection]
cache_max_entries = 123

[profile_mapping.semantic_domains.user_protection]
max_escalation_count = 7

[profile_mapping.semantic_domains.psychological_core]
response_irritated_annoyance = 66.0
stamina_trauma_scale = 0.55
flashback_intensity_max = 0.95

[profile_mapping.semantic_domains.trauma_complex]
escape_chaos_threshold = 6.0
negative_intensity_threshold = 0.75

[profile_mapping.semantic_domains.trauma_fragment]
prompt_key = "trauma_shard_stimulus"
fallback_negative_keywords = ["阴影", "恐惧"]

[profile_mapping.semantic_domains.trauma_layers]
max_triggers_per_layer = 11

[profile_mapping.semantic_domains.trauma_layers.severity_weights]
medium = 0.33

[profile_mapping.semantic_domains.trauma_system]
stress_threshold = 6.5
fragment_bonus_scale = 0.15

[profile_mapping.semantic_domains.trauma_timeline]
default_query_limit = 77

[profile_mapping.semantic_domains.trauma_triggers]
keyword_confidence = 0.81
default_patterns = [
    {{ pattern_id = "echo", keywords = ["回声"], severity = 2.2, layer_type = "浅层", response_type = "freeze", description = "回声类触发" }},
]

[profile_mapping.semantic_domains.trauma_worldview]
max_beliefs_tracked = 12
reality_keywords = ["虚假", "欺骗"]

[profile_mapping.semantic_domains.recall_post_send]
analysis_interval_seconds = 6.0
psychology_chaos_threshold = 8.5

[profile_mapping.semantic_domains.recall_typo]
max_typo_length = 66
prompt_key = "typo_decision_custom"

[profile_mapping.semantic_domains.recall_typo.wrong_char_map]
"你" = "尼"

[profile_mapping.semantic_domains.recall_self_behavior]
min_samples = 4
quick_learn_trigger_count = 5

[profile_mapping.semantic_domains.recall_self_awareness]
max_history = 88
default_cleanup_max_age_seconds = 7200.0

[profile_mapping.semantic_domains.recall_dimension]
typo_limit_per_window = 4
default_main_personality = "克制"

[profile_mapping.semantic_domains.recall_correction]
timeout_seconds = 166.0

[profile_mapping.semantic_domains.recall_shuffle]
probability = 0.22

[profile_mapping.semantic_domains.recall_shuffle.wrong_word_map]
"真" = "针"

[profile_mapping.semantic_domains.webui_rate_limit]
auth_request_limit = 12
api_request_limit = 144

[profile_mapping.semantic_domains.webui_websocket]
auth_max_connections_per_user = 7
state_monitor_default_interval_seconds = 1.5

[profile_mapping.semantic_domains.webui_git_mirror]
max_retries = 4
clone_timeout_seconds = 420

[profile_mapping.semantic_domains.webui_state_monitor_thresholds.emotion]
boredom_show = 0.41

[profile_mapping.semantic_domains.webui_state_monitor_thresholds.prediction]
probability_high = 0.74

[profile_mapping.semantic_domains.skill]
trial_to_active_invocations = 6
runtime_skill_cost = 0.4
sticker_dispatch_cooldown_seconds = 150.0

[profile_mapping.semantic_domains.adaptive_learning]
max_learning_items = 210
unknown_term_min_count = 4
vocabulary_limit = 18
expression_learn_temperature = 0.25
jargon_infer_thresholds = [2, 5, 10]
message_extract_min_messages = 12

[profile_mapping.scenario_profiles.image_high_risk.vision]
reply_suppressed = true
risk_score = {value}

[profile_mapping.module_views.memory]
semantic_domains = ["memory"]
long_term_limit = 4096

[profile_mapping.module_views.memory_capacity]
semantic_domains = ["memory_capacity"]

[profile_mapping.module_views.memory_retrieval]
semantic_domains = ["memory_retrieval"]

[profile_mapping.module_views.memory_decay]
semantic_domains = ["memory_decay"]

[profile_mapping.module_views.memory_dedup]
semantic_domains = ["memory_dedup"]

[profile_mapping.module_views.vision]
timeout_seconds = 12.0
vlm_temperature = 0.2
gif_frame_limit = 4

[profile_mapping.module_views.model_routing]
semantic_domains = ["model_routing"]

[profile_mapping.module_views.schedule]
semantic_domains = ["schedule"]

[profile_mapping.module_views.phase_timing]
semantic_domains = ["phase_timing"]

[profile_mapping.module_views.proactive_decider]
semantic_domains = ["proactive_decider"]

[profile_mapping.module_views.heartflow_decision]
semantic_domains = ["heartflow_decision"]

[profile_mapping.module_views.module_switches]
semantic_domains = ["module_switches"]

[profile_mapping.module_views.heartflow_runtime]
semantic_domains = ["heartflow_runtime"]

[profile_mapping.module_views.inner_voice]
semantic_domains = ["inner_voice"]

[profile_mapping.module_views.frequency_control]
semantic_domains = ["frequency_control"]

[profile_mapping.module_views.global_shield]
semantic_domains = ["global_shield"]

[profile_mapping.module_views.harassment_detection]
semantic_domains = ["harassment_detection"]

[profile_mapping.module_views.injection_detection]
semantic_domains = ["injection_detection"]

[profile_mapping.module_views.user_protection]
semantic_domains = ["user_protection"]

[profile_mapping.module_views.psychological_core]
semantic_domains = ["psychological_core"]

[profile_mapping.module_views.trauma_complex]
semantic_domains = ["trauma_complex"]

[profile_mapping.module_views.trauma_fragment]
semantic_domains = ["trauma_fragment"]

[profile_mapping.module_views.trauma_layers]
semantic_domains = ["trauma_layers"]

[profile_mapping.module_views.trauma_system]
semantic_domains = ["trauma_system"]

[profile_mapping.module_views.trauma_timeline]
semantic_domains = ["trauma_timeline"]

[profile_mapping.module_views.trauma_triggers]
semantic_domains = ["trauma_triggers"]

[profile_mapping.module_views.trauma_worldview]
semantic_domains = ["trauma_worldview"]

[profile_mapping.module_views.recall_post_send]
semantic_domains = ["recall_post_send"]

[profile_mapping.module_views.recall_typo]
semantic_domains = ["recall_typo"]

[profile_mapping.module_views.recall_self_behavior]
semantic_domains = ["recall_self_behavior"]

[profile_mapping.module_views.recall_self_awareness]
semantic_domains = ["recall_self_awareness"]

[profile_mapping.module_views.recall_dimension]
semantic_domains = ["recall_dimension"]

[profile_mapping.module_views.recall_correction]
semantic_domains = ["recall_correction"]

[profile_mapping.module_views.recall_shuffle]
semantic_domains = ["recall_shuffle"]

[profile_mapping.module_views.webui_rate_limit]
semantic_domains = ["webui_rate_limit"]

[profile_mapping.module_views.webui_websocket]
semantic_domains = ["webui_websocket"]

[profile_mapping.module_views.webui_git_mirror]
semantic_domains = ["webui_git_mirror"]

[profile_mapping.module_views.webui_state_monitor_thresholds]
semantic_domains = ["webui_state_monitor_thresholds"]

[profile_mapping.module_views.context]
semantic_domains = ["context"]

[profile_mapping.module_views.skill]
semantic_domains = ["skill"]

[profile_mapping.module_views.adaptive_learning]
semantic_domains = ["adaptive_learning"]
""".strip()


def main() -> None:
    CoreSettingsHub.teardown()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config_dir = tmp_path / "config"
        template_dir = tmp_path / "template"
        config_path = config_dir / "core_config.toml"
        template_path = template_dir / "core_config_template.toml"
        _write(config_path, _config(0.8))
        _write(template_path, _config(0.8))

        hub = CoreSettingsHub.boot(config_dir=config_dir, template_dir=template_dir)
        memory = hub.resolve_module_view("memory")
        assert memory.module == "memory"
        assert memory.scenario is None
        assert memory.values["enabled"] is True
        assert memory.values["short_term_limit"] == 64
        assert memory.values["long_term_limit"] == 4096
        assert memory.values["reactivation_cooldown_seconds"] == 88.0
        assert memory.fallback_used is False
        assert hub.resolve_module_view("memory_capacity").values["review_interval_seconds"] == 333
        assert hub.resolve_module_view("memory_capacity").values["max_count_by_tier"]["working"] == 222
        assert hub.resolve_module_view("memory_retrieval").values["rrf_k"] == 77
        assert hub.resolve_module_view("memory_decay").values["forget_threshold"] == 0.22
        assert hub.resolve_module_view("memory_dedup").values["similarity_threshold"] == 0.88

        vision = hub.resolve_module_view("vision", scenario="image_high_risk")
        assert vision.values["reply_suppressed"] is True
        assert vision.values["risk_score"] == 0.8
        assert vision.values["vlm_temperature"] == 0.2
        assert vision.values["gif_frame_limit"] == 4
        assert hub.resolve_module_view("context").values["reply_context_max_tokens"] == 96

        schedule = hub.resolve_module_view("schedule")
        assert schedule.fallback_used is False
        assert schedule.values["proactive_reply_cooldown_seconds"] == 45.0
        assert hub.assemble_scheduler_config()["cooldown_sec"] == 45.0
        assert hub.assemble_scheduler_config()["task_cancel_timeout_sec"] == 7.0
        assert hub.resolve_module_view("model_routing").values["model_fail_threshold"] == 4
        assert hub.resolve_module_view("phase_timing").values["max_idle_wait_seconds"] == 99
        assert hub.resolve_module_view("proactive_decider").values["activation_bar"] == 0.61
        assert hub.resolve_module_view("heartflow_decision").values["heated_pulse_limit"] == 9
        assert hub.resolve_module_view("module_switches").values["heartflow_enabled"] is True
        assert hub.resolve_module_view("heartflow_runtime").values["initiative_probability"] == 0.44
        assert hub.resolve_module_view("inner_voice").values["mention_min_desire"] == 7
        assert hub.resolve_module_view("frequency_control").values["skip_limit"] == 7
        assert hub.resolve_module_view("global_shield").values["default_duration"] == 321.0
        assert hub.resolve_module_view("harassment_detection").values["block_threshold"] == 0.66
        assert hub.resolve_module_view("injection_detection").values["cache_max_entries"] == 123
        assert hub.resolve_module_view("user_protection").values["max_escalation_count"] == 7
        assert hub.resolve_module_view("psychological_core").values["response_irritated_annoyance"] == 66.0
        assert hub.resolve_module_view("psychological_core").values["flashback_intensity_max"] == 0.95
        assert hub.resolve_module_view("trauma_complex").values["escape_chaos_threshold"] == 6.0
        assert hub.resolve_module_view("trauma_fragment").values["prompt_key"] == "trauma_shard_stimulus"
        assert hub.resolve_module_view("trauma_fragment").values["fallback_negative_keywords"] == ["阴影", "恐惧"]
        assert hub.resolve_module_view("trauma_layers").values["severity_weights"]["medium"] == 0.33
        assert hub.resolve_module_view("trauma_system").values["stress_threshold"] == 6.5
        assert hub.resolve_module_view("trauma_timeline").values["default_query_limit"] == 77
        assert hub.resolve_module_view("trauma_triggers").values["keyword_confidence"] == 0.81
        assert hub.resolve_module_view("trauma_triggers").values["default_patterns"][0]["pattern_id"] == "echo"
        assert hub.resolve_module_view("trauma_worldview").values["max_beliefs_tracked"] == 12
        assert hub.resolve_module_view("recall_post_send").values["analysis_interval_seconds"] == 6.0
        assert hub.resolve_module_view("recall_typo").values["max_typo_length"] == 66
        assert hub.resolve_module_view("recall_typo").values["wrong_char_map"]["你"] == "尼"
        assert hub.resolve_module_view("recall_self_behavior").values["min_samples"] == 4
        assert hub.resolve_module_view("recall_self_behavior").values["quick_learn_trigger_count"] == 5
        assert hub.resolve_module_view("recall_self_awareness").values["max_history"] == 88
        assert hub.resolve_module_view("recall_dimension").values["typo_limit_per_window"] == 4
        assert hub.resolve_module_view("recall_correction").values["timeout_seconds"] == 166.0
        assert hub.resolve_module_view("recall_shuffle").values["probability"] == 0.22
        assert hub.resolve_module_view("recall_shuffle").values["wrong_word_map"]["真"] == "针"
        assert hub.resolve_module_view("webui_rate_limit").values["auth_request_limit"] == 12
        assert hub.resolve_module_view("webui_websocket").values["auth_max_connections_per_user"] == 7
        assert hub.resolve_module_view("webui_git_mirror").values["clone_timeout_seconds"] == 420
        assert hub.resolve_module_view("webui_state_monitor_thresholds").values["emotion"]["boredom_show"] == 0.41
        assert hub.resolve_module_view("webui_state_monitor_thresholds").values["prediction"]["probability_high"] == 0.74
        assert hub.resolve_module_view("skill").values["trial_to_active_invocations"] == 6
        assert hub.resolve_module_view("adaptive_learning").values["unknown_term_min_count"] == 4
        assert hub.resolve_module_view("adaptive_learning").values["expression_learn_temperature"] == 0.25
        assert hub.resolve_module_view("adaptive_learning").values["jargon_infer_thresholds"] == [2, 5, 10]

        sleep(1.1)
        _write(config_path, _config(0.35))
        os.utime(config_path, None)
        assert hub.hot_reload() is True
        reloaded = hub.resolve_module_view("vision", scenario="image_high_risk")
        assert reloaded.values["risk_score"] == 0.35

        result = {
            "memory_keys": sorted(memory.values.keys()),
            "vision_risk_score": reloaded.values["risk_score"],
            "schedule_fallback_used": schedule.fallback_used,
            "schedule_cooldown_sec": hub.assemble_scheduler_config()["cooldown_sec"],
            "model_fail_threshold": hub.resolve_module_view("model_routing").values["model_fail_threshold"],
            "context_reply_tokens": hub.resolve_module_view("context").values["reply_context_max_tokens"],
            "phase_idle_timeout": hub.resolve_module_view("phase_timing").values["phase_idle_timeout_seconds"],
            "memory_rrf_k": hub.resolve_module_view("memory_retrieval").values["rrf_k"],
            "inner_voice_mention_floor": hub.resolve_module_view("inner_voice").values["mention_min_desire"],
            "frequency_skip_limit": hub.resolve_module_view("frequency_control").values["skip_limit"],
            "harassment_block_threshold": hub.resolve_module_view("harassment_detection").values["block_threshold"],
            "psych_irritated_threshold": hub.resolve_module_view("psychological_core").values["response_irritated_annoyance"],
            "trauma_query_limit": hub.resolve_module_view("trauma_timeline").values["default_query_limit"],
            "recall_typo_limit": hub.resolve_module_view("recall_typo").values["max_typo_length"],
            "recall_correction_timeout": hub.resolve_module_view("recall_correction").values["timeout_seconds"],
            "webui_auth_limit": hub.resolve_module_view("webui_rate_limit").values["auth_request_limit"],
            "webui_prediction_high": hub.resolve_module_view("webui_state_monitor_thresholds").values["prediction"]["probability_high"],
            "skill_trial_invocations": hub.resolve_module_view("skill").values["trial_to_active_invocations"],
            "adaptive_learning_limit": hub.resolve_module_view("adaptive_learning").values["max_learning_items"],
            "jargon_thresholds": hub.resolve_module_view("adaptive_learning").values["jargon_infer_thresholds"],
            "trace_layers": sorted({item.layer for item in vision.trace}),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    CoreSettingsHub.teardown()


if __name__ == "__main__":
    main()

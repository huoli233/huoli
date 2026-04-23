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
            "inner_voice_mention_floor": hub.resolve_module_view("inner_voice").values["mention_min_desire"],
            "frequency_skip_limit": hub.resolve_module_view("frequency_control").values["skip_limit"],
            "skill_trial_invocations": hub.resolve_module_view("skill").values["trial_to_active_invocations"],
            "adaptive_learning_limit": hub.resolve_module_view("adaptive_learning").values["max_learning_items"],
            "jargon_thresholds": hub.resolve_module_view("adaptive_learning").values["jargon_infer_thresholds"],
            "trace_layers": sorted({item.layer for item in vision.trace}),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    CoreSettingsHub.teardown()


if __name__ == "__main__":
    main()

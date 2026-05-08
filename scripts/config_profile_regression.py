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
from src.config.api_ada_configs import APIProvider, ModelInfo, TaskConfig
from src.llm_models.payload_content.message import Message, RoleType
from src.llm_models.utils_model import LLMRequest, RequestType


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _social_calculator_file(
    hostile_acceleration_multiplier: float,
    flirt_multiplier: float,
    existence_threat_bonus: float,
) -> str:
    return f"""
[categories.hostile]
acceleration_multiplier = {hostile_acceleration_multiplier}

[intents]
flirt = {flirt_multiplier}

[types.existence_threat]
category = "hostile"
bonus = {existence_threat_bonus}
""".strip()


def _check_llm_routing_runtime_contract() -> dict[str, object]:
    request = LLMRequest(
        TaskConfig(model_list=["fast-openai"]),
        request_type="replyer",
    )
    openai_provider = APIProvider(
        name="fake-openai",
        base_url="http://127.0.0.1/v1",
        api_key="test-key",
        max_retry=3,
        timeout=120,
        retry_interval=9,
    )
    openai_model = ModelInfo(
        name="fast-openai",
        model_identifier="fast-openai",
        api_provider="fake-openai",
        client_type="openai",
    )
    gemini_model = ModelInfo(
        name="fast-gemini",
        model_identifier="gemini-2.5-flash",
        api_provider="fake-gemini",
        client_type="gemini",
    )
    explicit_thinking_model = ModelInfo(
        name="deep-openai",
        model_identifier="deep-openai",
        api_provider="fake-openai",
        client_type="openai",
        extra_params={"enable_thinking": True},
    )
    disabled_by_pattern_model = ModelInfo(
        name="qwen-deep",
        model_identifier="qwen-deep",
        api_provider="fake-openai",
        client_type="openai",
        extra_params={"enable_thinking": True},
    )

    openai_params = request._effective_extra_params(openai_model)
    gemini_params = request._effective_extra_params(gemini_model)
    explicit_params = request._effective_extra_params(explicit_thinking_model)
    disabled_params = request._effective_extra_params(disabled_by_pattern_model)
    guarded_messages = request._with_no_thinking_guard(
        [Message(RoleType.User, "hello")],
        openai_model,
    )

    assert openai_params["enable_thinking"] is False
    assert openai_params["chat_template_kwargs"]["enable_thinking"] is False
    assert openai_params["reasoning_effort"] == "minimal"
    assert gemini_params["enable_thinking"] is False
    assert gemini_params["include_thoughts"] is False
    assert gemini_params["thinking_budget"] == 0
    assert explicit_params["enable_thinking"] is True
    assert explicit_params["chat_template_kwargs"]["enable_thinking"] is True
    assert "reasoning_effort" not in explicit_params
    assert disabled_params["enable_thinking"] is False
    assert disabled_params["chat_template_kwargs"]["enable_thinking"] is False
    assert guarded_messages[0].role == RoleType.System
    assert "常规快速推理" in str(guarded_messages[0].content)
    assert request._request_timeout_budget(openai_provider, RequestType.RESPONSE) == 17.0
    assert request._request_timeout_budget(openai_provider, RequestType.EMBEDDING) == 9.0
    assert request._request_timeout_budget(openai_provider, RequestType.AUDIO) == 11.0
    assert request._retry_budget(openai_provider, RequestType.RESPONSE) == 1
    assert request._retry_budget(openai_provider, RequestType.EMBEDDING) == 2
    assert request._retry_sleep_seconds(openai_provider, RequestType.RESPONSE) == 1.25
    return {
        "default_no_thinking": True,
        "gemini_thinking_budget": gemini_params["thinking_budget"],
        "response_timeout": request._request_timeout_budget(openai_provider, RequestType.RESPONSE),
        "response_retry": request._retry_budget(openai_provider, RequestType.RESPONSE),
    }


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
response_request_timeout_seconds = 17.0
response_request_timeout_floor_seconds = 4.0
embedding_request_timeout_seconds = 9.0
audio_request_timeout_seconds = 11.0
response_max_retry = 1
background_max_retry = 2
response_retry_interval_cap_seconds = 1.25

[profile_mapping.semantic_domains.schedule]
proactive_reply_cooldown_seconds = 45.0
task_cancel_timeout_seconds = 7.0
telemetry_heartbeat_interval_seconds = 123.0
expression_learning_interval_seconds = 456.0

[profile_mapping.semantic_domains.runtime_tuning]
autosave_interval_seconds = 321.0
heartfc_voice_cooldown_sec = 9.0

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

[profile_mapping.semantic_domains.brain_chat_runtime]
parallel_action_timeout_seconds = 88.0
main_loop_max_retries = 4

[profile_mapping.semantic_domains.brain_planner]
planner_timeout_seconds = 22.0

[profile_mapping.semantic_domains.brain_pfc_action]
llm_timeout_seconds = 26.0
high_risk_threshold = 0.41
farewell_timeout_seconds = 19.0

[profile_mapping.semantic_domains.brain_pfc_reply]
duplicate_similarity_threshold = 0.91
generation_timeout_seconds = 44.0

[profile_mapping.semantic_domains.brain_waiter]
desired_timeout_seconds = 222

[profile_mapping.semantic_domains.brain_pfc_goal]
analysis_timeout_seconds = 12.0
max_goals = 5

[profile_mapping.semantic_domains.identity_bot]
nickname = "测试爱丽丝"
alias_names = ["测试爱丽丝", "测试小爱"]
bot_id = "10001"

[profile_mapping.semantic_domains.identity_bot.identity_templates]
default_persona = "你叫{{name}}。"

[profile_mapping.semantic_domains.identity_anchor]
max_history = 55

[profile_mapping.semantic_domains.identity_anchor.prompt_templates]
default_persona = "anchor:{{name}}"

[profile_mapping.semantic_domains.identity_user_persistence]
storage_dir = "tmp/users"
max_activities = 77

[profile_mapping.semantic_domains.energy_runtime.chat_value]
ceiling = 123.0

[profile_mapping.semantic_domains.energy_runtime.decay]
annoyance_decay_per_min = 4.2

[profile_mapping.semantic_domains.energy_runtime.penalty_caps]
trauma_cap = 0.71

[profile_mapping.semantic_domains.trigger_runtime]
energy_overflow_ratio = 0.91

[profile_mapping.semantic_domains.personality_factors]
social_warmth = 0.77

[profile_mapping.semantic_domains.personality]
persona_pool_capacity = 12
default_blend_duration_sec = 44.0
persona_auto_revert_sec = 333.0

[profile_mapping.semantic_domains.social_calculator]
offset_decay_per_hour = 0.23
positive_streak_coeff = 1.9
file_refs = ["profile_domains/social/social_calculator.toml"]

[profile_mapping.semantic_domains.social_settlement]
absolute_step_ceil = 12.0
reversal_required_streak = 5

[profile_mapping.semantic_domains.social_phase_tracker]
trend_window_sec = 2222.0
impression_update_interval = 4444.0

[profile_mapping.semantic_domains.social_phase_tracker.milestones]
first_trust = 33.0

[profile_mapping.semantic_domains.social_phase_tracker.phase_bands]
trusted = [66.0, 101.0]

[profile_mapping.semantic_domains.social_phase_tracker.phase_weights]
trusted = 1.23

[profile_mapping.semantic_domains.social_phase_tracker.phase_descriptions]
trusted = "超信任"

[profile_mapping.semantic_domains.social_affect_fuser]
support_relief_cap = 0.25
group_window_limit = 18
support_keywords = ["站你"]

[profile_mapping.semantic_domains.social_value_core]
social_new_user_initial = 6.5
social_single_step_max = 3.3

[profile_mapping.semantic_domains.perception_group_sense]
burst_msg_count = 21

[profile_mapping.semantic_domains.perception_self_sense]
max_history_size = 66

[profile_mapping.semantic_domains.perception_user_relation]
familiar_threshold = 9

[profile_mapping.semantic_domains.perception_user_state]
model_weight = 0.7

[profile_mapping.semantic_domains.perception_user_state.verdict_thresholds]
want_to_chat = 0.68

[profile_mapping.semantic_domains.affection_dynamics]
max_annoyance = 88.0

[profile_mapping.semantic_domains.persona_controller]
max_personas = 12

[profile_mapping.semantic_domains.persona_generator]
switch_cooldown = 240.0

[profile_mapping.semantic_domains.persona_switcher]
transition_duration_seconds = 44.0
primary_base_weight = 0.55

[profile_mapping.semantic_domains.perception_behavior]
memory_cap = 77

[profile_mapping.semantic_domains.perception_buffer]
recent_show_count = 8

[profile_mapping.semantic_domains.perception_group_atmosphere]
max_history = 66

[profile_mapping.semantic_domains.perception_interest]
high_threshold = 0.82

[profile_mapping.semantic_domains.perception_message_preprocessor]
message_text_max_length = 88
bot_aliases = ["阿活"]

[profile_mapping.semantic_domains.perception_signal_detector]
strong_at_threshold = 2

[profile_mapping.semantic_domains.perception_understand]
cache_ttl = 88.0

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
edit_scope = "mixed"
user_editable_keys = ["short_term_limit", "temporary_ttl_seconds", "reactivation_cooldown_seconds", "reactivation_max_candidates", "emotion_reactivation_threshold", "boredom_reactivation_threshold"]
system_only_keys = ["overload_ratio_threshold", "overload_amnesia_threshold", "overload_assessment_ttl_seconds", "long_term_batch_size", "memoir_timeout_seconds", "reactivation_cache_limit", "reactivation_score_threshold", "reactivation_topic_scan_limit", "reactivation_visibility_scan_limit", "boredom_high_threshold"]

[profile_mapping.module_views.memory_capacity]
semantic_domains = ["memory_capacity"]
edit_scope = "mixed"
user_editable_keys = ["review_interval_seconds", "max_count_by_tier"]
system_only_keys = ["llm_timeout_seconds"]

[profile_mapping.module_views.memory_retrieval]
semantic_domains = ["memory_retrieval"]
edit_scope = "mixed"
user_editable_keys = ["bm25_weight", "vector_weight"]
system_only_keys = ["rrf_k"]

[profile_mapping.module_views.memory_decay]
semantic_domains = ["memory_decay"]
edit_scope = "mixed"
user_editable_keys = ["decay_rate_per_hour", "forget_threshold", "min_age_for_decay_seconds"]
system_only_keys = ["cycle_interval_seconds", "max_batch_for_llm"]

[profile_mapping.module_views.memory_dedup]
semantic_domains = ["memory_dedup"]
edit_scope = "mixed"
user_editable_keys = ["similarity_threshold"]
system_only_keys = ["scan_interval_seconds", "max_pairs_per_scan", "merge_top_k"]

[profile_mapping.module_views.vision]
timeout_seconds = 12.0
vlm_temperature = 0.2
gif_frame_limit = 4
edit_scope = "mixed"
user_editable_keys = ["image_recognition_enabled", "gif_convert_enabled"]
system_only_keys = ["vlm_temperature", "vlm_timeout_seconds", "gif_frame_limit", "gif_similarity_threshold", "emoji_vlm_temperature", "emoji_emotion_temperature", "emoji_emotion_timeout_seconds", "record_only_confidence_threshold", "degrade_to_placeholder"]

[profile_mapping.module_views.model_routing]
semantic_domains = ["model_routing"]
edit_scope = "mixed"
user_editable_keys = ["complexity_threshold", "skip_low_value_threshold", "high_risk_threshold"]
system_only_keys = ["default_model_path", "large_model_path", "model_cooldown_seconds", "model_fail_threshold", "tool_incompat_cooldown_seconds"]

[profile_mapping.module_views.schedule]
semantic_domains = ["schedule"]
edit_scope = "mixed"
user_editable_keys = ["default_idle_minutes", "idle_trigger_seconds", "proactive_reply_cooldown_seconds", "priority_boost_on_mention", "llm_perception_enabled", "perception_engagement_threshold"]
system_only_keys = ["active_user_cleanup_interval_seconds", "backend_event_cooldown_seconds", "backend_status_log_enabled", "backend_status_log_interval_sec", "cache_cleanup_interval_seconds", "chatter_cleanup_interval_seconds", "cleanup_chat_instances_interval_seconds", "cleanup_error_backoff_seconds", "cleanup_initial_delay_seconds", "cleanup_loop_interval_seconds", "context_cleanup_interval_seconds", "expression_learning_interval_seconds", "max_concurrent_thinks", "memory_cleanup_interval_seconds", "memory_governance_interval_seconds", "perception_max_tokens", "queue_capacity", "retry_max_count", "session_cleanup_interval_seconds", "session_max_age_days", "session_storage_dir", "skill_patrol_interval_seconds", "task_cancel_timeout_seconds", "task_shutdown_timeout_seconds", "telemetry_heartbeat_interval_seconds", "temp_files_cleanup_interval_seconds", "think_timeout_seconds"]

[profile_mapping.module_views.runtime_tuning]
semantic_domains = ["runtime_tuning"]
edit_scope = "mixed"
user_editable_keys = ["autosave_interval_seconds", "flow_planner_slow_threshold_seconds", "heartfc_perception_cooldown_sec", "heartfc_voice_cooldown_sec"]
system_only_keys = ["flow_planner_timeout_seconds", "heartfc_dormant_poll_sec", "heartfc_energy_drain_floor", "heartfc_post_message_retry_sec", "heartfc_tick_floor_sec", "vlm_concurrency_limit"]

[profile_mapping.module_views.phase_timing]
semantic_domains = ["phase_timing"]
edit_scope = "mixed"
user_editable_keys = ["rest_default_seconds", "peek_chance_ratio", "max_idle_wait_seconds", "max_chase_count", "phase_idle_timeout_seconds"]
system_only_keys = ["reflect_interval_seconds", "max_reflect_rounds", "transition_cooldown_ms", "pending_expire_sec", "max_engaged_duration_sec", "awaken_skip_threshold", "auto_dormant_after_sec"]

[profile_mapping.module_views.proactive_decider]
semantic_domains = ["proactive_decider"]
edit_scope = "mixed"
user_editable_keys = ["activation_bar", "reply_grace_period_sec", "silence_full_score_sec", "max_pending_events"]
system_only_keys = ["reward_on_reply", "penalty_on_ignore", "penalty_on_negative", "reward_decay_per_hour", "reply_watch_window_sec", "reward_floor", "reward_ceiling", "wt_content_freshness", "wt_emotional_readiness", "wt_historical_reward", "wt_inner_desire", "wt_intention_drive", "wt_social_standing", "wt_vitality_reserve"]

[profile_mapping.module_views.heartflow_decision]
semantic_domains = ["heartflow_decision"]
edit_scope = "mixed"
user_editable_keys = ["bond", "tactics", "mood_factors"]
system_only_keys = ["dimension_weights", "barrier_seeds", "barrier_tuning", "threat_weights", "fusion", "immediacy", "credibility", "heated_pulse_limit", "pulse_span_seconds", "heated_slot_ttl", "calm_slot_ttl", "slot_capacity_cap"]

[profile_mapping.module_views.module_switches]
semantic_domains = ["module_switches"]
edit_scope = "mixed"
user_editable_keys = ["heartflow_enabled", "proactive_enabled", "silence_watcher_enabled", "inner_voice_enabled"]
system_only_keys = ["energy_system_enabled", "emotion_stream_enabled", "enhanced_focus_route_enabled"]

[profile_mapping.module_views.heartflow_runtime]
semantic_domains = ["heartflow_runtime"]
edit_scope = "mixed"
user_editable_keys = ["focus_channels", "quiet_threshold_sec", "max_initiative_per_hour"]
system_only_keys = ["initiative_probability", "initiative_interval_sec", "monitor_interval_sec", "stamina_max"]

[profile_mapping.module_views.inner_voice]
semantic_domains = ["inner_voice"]
edit_scope = "mixed"
user_editable_keys = ["waiting_thoughts", "fallback_action"]
system_only_keys = ["mention_min_desire", "private_chat_min_desire", "resource_penalty_max", "resource_thresholds", "time_bands", "max_output_actions", "desire_scale_min", "desire_scale_max"]

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
edit_scope = "mixed"
user_editable_keys = ["probability", "extra_chars", "wrong_char_map"]
system_only_keys = ["generation_task", "max_typo_length", "prompt_key", "random_confidence", "random_reasoning", "skip_reasoning"]

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
edit_scope = "mixed"
user_editable_keys = ["probability", "wrong_word_map", "shuffle_types"]
system_only_keys = ["max_length", "prompt_key"]

[profile_mapping.module_views.webui_rate_limit]
semantic_domains = ["webui_rate_limit"]

[profile_mapping.module_views.webui_websocket]
semantic_domains = ["webui_websocket"]

[profile_mapping.module_views.webui_git_mirror]
semantic_domains = ["webui_git_mirror"]

[profile_mapping.module_views.webui_state_monitor_thresholds]
semantic_domains = ["webui_state_monitor_thresholds"]
edit_scope = "mixed"
user_editable_keys = ["emotion", "relationship", "trauma", "memory", "prediction", "scene", "attention", "circadian"]
system_only_keys = ["autonomy", "pending", "safety"]

[profile_mapping.module_views.brain_chat_runtime]
semantic_domains = ["brain_chat_runtime"]

[profile_mapping.module_views.brain_planner]
semantic_domains = ["brain_planner"]

[profile_mapping.module_views.brain_pfc_action]
semantic_domains = ["brain_pfc_action"]

[profile_mapping.module_views.brain_pfc_reply]
semantic_domains = ["brain_pfc_reply"]

[profile_mapping.module_views.brain_waiter]
semantic_domains = ["brain_waiter"]

[profile_mapping.module_views.brain_pfc_goal]
semantic_domains = ["brain_pfc_goal"]

[profile_mapping.module_views.identity_bot]
semantic_domains = ["identity_bot"]
edit_scope = "mixed"
user_editable_keys = ["nickname", "alias_names", "character_age", "interests", "lore", "personality_core", "relationships", "reply_style", "visual_features", "identity_templates", "role_visual_profiles", "skill_prompt_templates"]
system_only_keys = ["bot_id"]

[profile_mapping.module_views.identity_anchor]
semantic_domains = ["identity_anchor"]
edit_scope = "mixed"
user_editable_keys = ["prompt_templates"]
system_only_keys = ["max_history"]

[profile_mapping.module_views.identity_user_persistence]
semantic_domains = ["identity_user_persistence"]
edit_scope = "mixed"
user_editable_keys = ["max_activities"]
system_only_keys = ["storage_dir", "file_lock_cleanup_interval_seconds", "file_lock_stale_seconds"]

[profile_mapping.module_views.energy_runtime]
semantic_domains = ["energy_runtime"]

[profile_mapping.module_views.trigger_runtime]
semantic_domains = ["trigger_runtime"]

[profile_mapping.module_views.personality_factors]
semantic_domains = ["personality_factors"]
edit_scope = "user"
user_editable_keys = ["sensitivity", "tolerance", "reactiveness", "recovery_speed", "social_warmth", "curiosity_drive", "reply_eagerness"]

[profile_mapping.module_views.personality]
semantic_domains = ["personality"]
edit_scope = "mixed"
user_editable_keys = ["default_blend_duration_sec", "persona_auto_revert_sec", "persona_pool_capacity"]
system_only_keys = ["bond_saturation_steepness", "bond_reversal_multiplier", "bond_reversal_min_streak", "bond_large_delta_threshold", "bond_large_delta_dampen_rate", "bond_step_ceiling", "chaos_sensitivity", "chaos_natural_decay_per_hour", "mask_base_armor", "mask_strain_rate", "mask_recovery_rate", "crisis_persona_chaos_threshold"]

[profile_mapping.module_views.social_calculator]
semantic_domains = ["social_calculator"]
edit_scope = "system"

[profile_mapping.module_views.social_settlement]
semantic_domains = ["social_settlement"]

[profile_mapping.module_views.social_phase_tracker]
semantic_domains = ["social_phase_tracker"]

[profile_mapping.module_views.social_affect_fuser]
semantic_domains = ["social_affect_fuser"]

[profile_mapping.module_views.social_value_core]
semantic_domains = ["social_value_core"]

[profile_mapping.module_views.perception_group_sense]
semantic_domains = ["perception_group_sense"]

[profile_mapping.module_views.perception_self_sense]
semantic_domains = ["perception_self_sense"]

[profile_mapping.module_views.perception_user_relation]
semantic_domains = ["perception_user_relation"]

[profile_mapping.module_views.perception_user_state]
semantic_domains = ["perception_user_state"]

[profile_mapping.module_views.affection_dynamics]
semantic_domains = ["affection_dynamics"]
edit_scope = "mixed"
user_editable_keys = ["max_affection", "max_trust", "max_annoyance"]

[profile_mapping.module_views.persona_controller]
semantic_domains = ["persona_controller"]
edit_scope = "mixed"
user_editable_keys = ["max_personas", "default_duration"]

[profile_mapping.module_views.persona_generator]
semantic_domains = ["persona_generator"]
edit_scope = "mixed"
user_editable_keys = ["max_history", "switch_cooldown"]

[profile_mapping.module_views.persona_switcher]
semantic_domains = ["persona_switcher"]
edit_scope = "mixed"
user_editable_keys = ["auto_revert_default_seconds", "transition_duration_seconds", "trauma_switch_duration_seconds"]
system_only_keys = ["transition_revert_buffer_seconds", "time_balance_max_seconds", "primary_base_weight", "primary_time_balance_scale", "trauma_impact_cap", "trauma_impact_scale", "primary_weight_floor", "primary_weight_ceiling", "transition_phase_1_ratio", "transition_phase_2_ratio", "transition_phase_3_ratio"]

[profile_mapping.module_views.perception_behavior]
semantic_domains = ["perception_behavior"]

[profile_mapping.module_views.perception_buffer]
semantic_domains = ["perception_buffer"]

[profile_mapping.module_views.perception_group_atmosphere]
semantic_domains = ["perception_group_atmosphere"]

[profile_mapping.module_views.perception_interest]
semantic_domains = ["perception_interest"]

[profile_mapping.module_views.perception_message_preprocessor]
semantic_domains = ["perception_message_preprocessor"]

[profile_mapping.module_views.perception_signal_detector]
semantic_domains = ["perception_signal_detector"]

[profile_mapping.module_views.perception_understand]
semantic_domains = ["perception_understand"]

[profile_mapping.module_views.context]
semantic_domains = ["context"]
edit_scope = "mixed"
user_editable_keys = ["reply_context_max_tokens", "extra_info_max_tokens", "max_context_messages", "max_extra_blocks", "compressed_context_target_chars"]
system_only_keys = ["compact_line_max_tokens", "compression_trigger_chars", "pollution_threshold", "repeat_topic_threshold", "selection_saturation_ratio"]

[profile_mapping.module_views.skill]
semantic_domains = ["skill"]
edit_scope = "mixed"
user_editable_keys = ["sticker_daily_ceiling", "sticker_dispatch_threshold"]
system_only_keys = ["trigger_threshold", "cooldown_seconds", "failure_backoff_seconds", "priority_boost_from_learning", "runtime_skill_cost", "analysis_skill_cost", "dev_skill_cost"]

[profile_mapping.module_views.adaptive_learning]
semantic_domains = ["adaptive_learning"]
edit_scope = "mixed"
user_editable_keys = ["priority_feedback_enabled", "vocabulary_limit", "mood_vocabulary_limit"]
system_only_keys = ["expression_learn_temperature", "expression_learn_timeout_seconds", "jargon_infer_temperature", "jargon_infer_timeout_seconds", "behavior_check_interval_seconds", "max_learning_items"]
""".strip()


def main() -> None:
    CoreSettingsHub.teardown()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config_dir = tmp_path / "config"
        template_dir = tmp_path / "template"
        config_path = config_dir / "core_config.toml"
        template_path = template_dir / "core_config_template.toml"
        config_sidecar_path = config_dir / "profile_domains" / "social" / "social_calculator.toml"
        template_sidecar_path = template_dir / "profile_domains" / "social" / "social_calculator.toml"
        _write(config_path, _config(0.8))
        _write(template_path, _config(0.8))
        _write(template_sidecar_path, _social_calculator_file(3.8, 0.95, 2.2))
        _write(config_sidecar_path, _social_calculator_file(3.8, 0.95, 2.2))

        hub = CoreSettingsHub.boot(config_dir=config_dir, template_dir=template_dir)
        memory = hub.resolve_module_view("memory")
        assert memory.module == "memory"
        assert memory.scenario is None
        assert memory.values["enabled"] is True
        assert memory.values["short_term_limit"] == 64
        assert memory.values["long_term_limit"] == 4096
        assert memory.values["reactivation_cooldown_seconds"] == 88.0
        assert memory.fallback_used is False
        assert memory.edit_scope == "mixed"
        assert memory.user_editable_keys == ("short_term_limit", "temporary_ttl_seconds", "reactivation_cooldown_seconds", "reactivation_max_candidates", "emotion_reactivation_threshold", "boredom_reactivation_threshold")
        memory_capacity = hub.resolve_module_view("memory_capacity")
        assert memory_capacity.values["review_interval_seconds"] == 333
        assert memory_capacity.values["max_count_by_tier"]["working"] == 222
        assert memory_capacity.edit_scope == "mixed"
        assert memory_capacity.user_editable_keys == ("review_interval_seconds", "max_count_by_tier")
        memory_retrieval = hub.resolve_module_view("memory_retrieval")
        assert memory_retrieval.values["rrf_k"] == 77
        assert memory_retrieval.edit_scope == "mixed"
        assert memory_retrieval.user_editable_keys == ("bm25_weight", "vector_weight")
        memory_decay = hub.resolve_module_view("memory_decay")
        assert memory_decay.values["forget_threshold"] == 0.22
        assert memory_decay.edit_scope == "mixed"
        assert memory_decay.user_editable_keys == ("decay_rate_per_hour", "forget_threshold", "min_age_for_decay_seconds")
        memory_dedup = hub.resolve_module_view("memory_dedup")
        assert memory_dedup.values["similarity_threshold"] == 0.88
        assert memory_dedup.edit_scope == "mixed"
        assert memory_dedup.user_editable_keys == ("similarity_threshold",)

        vision = hub.resolve_module_view("vision", scenario="image_high_risk")
        assert vision.values["reply_suppressed"] is True
        assert vision.values["risk_score"] == 0.8
        assert vision.values["vlm_temperature"] == 0.2
        assert vision.values["gif_frame_limit"] == 4
        assert vision.edit_scope == "mixed"
        assert vision.user_editable_keys == ("image_recognition_enabled", "gif_convert_enabled")
        context = hub.resolve_module_view("context")
        assert context.values["reply_context_max_tokens"] == 96
        assert context.edit_scope == "mixed"
        assert context.user_editable_keys == ("reply_context_max_tokens", "extra_info_max_tokens", "max_context_messages", "max_extra_blocks", "compressed_context_target_chars")

        schedule = hub.resolve_module_view("schedule")
        assert schedule.fallback_used is False
        assert schedule.edit_scope == "mixed"
        assert schedule.user_editable_keys == ("default_idle_minutes", "idle_trigger_seconds", "proactive_reply_cooldown_seconds", "priority_boost_on_mention", "llm_perception_enabled", "perception_engagement_threshold")
        assert schedule.values["proactive_reply_cooldown_seconds"] == 45.0
        assert hub.assemble_scheduler_config()["cooldown_sec"] == 45.0
        assert hub.assemble_scheduler_config()["task_cancel_timeout_sec"] == 7.0
        model_routing = hub.resolve_module_view("model_routing")
        assert model_routing.values["model_fail_threshold"] == 4
        assert model_routing.values["response_request_timeout_seconds"] == 17.0
        assert model_routing.values["embedding_request_timeout_seconds"] == 9.0
        assert model_routing.values["audio_request_timeout_seconds"] == 11.0
        assert model_routing.values["response_max_retry"] == 1
        assert model_routing.values["background_max_retry"] == 2
        assert model_routing.values["response_retry_interval_cap_seconds"] == 1.25
        assert model_routing.edit_scope == "mixed"
        assert model_routing.user_editable_keys == ("complexity_threshold", "skip_low_value_threshold", "high_risk_threshold")
        llm_runtime = _check_llm_routing_runtime_contract()
        runtime_tuning = hub.resolve_module_view("runtime_tuning")
        assert runtime_tuning.values["autosave_interval_seconds"] == 321.0
        assert runtime_tuning.edit_scope == "mixed"
        assert runtime_tuning.user_editable_keys == ("autosave_interval_seconds", "flow_planner_slow_threshold_seconds", "heartfc_perception_cooldown_sec", "heartfc_voice_cooldown_sec")
        phase_timing = hub.resolve_module_view("phase_timing")
        assert phase_timing.values["max_idle_wait_seconds"] == 99
        assert phase_timing.edit_scope == "mixed"
        assert phase_timing.user_editable_keys == ("rest_default_seconds", "peek_chance_ratio", "max_idle_wait_seconds", "max_chase_count", "phase_idle_timeout_seconds")
        proactive_decider = hub.resolve_module_view("proactive_decider")
        assert proactive_decider.values["activation_bar"] == 0.61
        assert proactive_decider.edit_scope == "mixed"
        assert proactive_decider.user_editable_keys == ("activation_bar", "reply_grace_period_sec", "silence_full_score_sec", "max_pending_events")
        heartflow_decision = hub.resolve_module_view("heartflow_decision")
        assert heartflow_decision.values["heated_pulse_limit"] == 9
        assert heartflow_decision.edit_scope == "mixed"
        assert heartflow_decision.user_editable_keys == ("bond", "tactics", "mood_factors")
        module_switches = hub.resolve_module_view("module_switches")
        assert module_switches.values["heartflow_enabled"] is True
        assert module_switches.edit_scope == "mixed"
        assert module_switches.user_editable_keys == ("heartflow_enabled", "proactive_enabled", "silence_watcher_enabled", "inner_voice_enabled")
        heartflow_runtime = hub.resolve_module_view("heartflow_runtime")
        assert heartflow_runtime.values["initiative_probability"] == 0.44
        assert heartflow_runtime.edit_scope == "mixed"
        assert heartflow_runtime.user_editable_keys == ("focus_channels", "quiet_threshold_sec", "max_initiative_per_hour")
        assert heartflow_runtime.system_only_keys == ("initiative_probability", "initiative_interval_sec", "monitor_interval_sec", "stamina_max")
        inner_voice = hub.resolve_module_view("inner_voice")
        assert inner_voice.values["mention_min_desire"] == 7
        assert inner_voice.edit_scope == "mixed"
        assert inner_voice.user_editable_keys == ("waiting_thoughts", "fallback_action")
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
        recall_typo = hub.resolve_module_view("recall_typo")
        assert recall_typo.values["max_typo_length"] == 66
        assert recall_typo.values["wrong_char_map"]["你"] == "尼"
        assert recall_typo.edit_scope == "mixed"
        assert recall_typo.user_editable_keys == ("probability", "extra_chars", "wrong_char_map")
        assert hub.resolve_module_view("recall_self_behavior").values["min_samples"] == 4
        assert hub.resolve_module_view("recall_self_behavior").values["quick_learn_trigger_count"] == 5
        assert hub.resolve_module_view("recall_self_awareness").values["max_history"] == 88
        assert hub.resolve_module_view("recall_dimension").values["typo_limit_per_window"] == 4
        assert hub.resolve_module_view("recall_correction").values["timeout_seconds"] == 166.0
        recall_shuffle = hub.resolve_module_view("recall_shuffle")
        assert recall_shuffle.values["probability"] == 0.22
        assert recall_shuffle.values["wrong_word_map"]["真"] == "针"
        assert recall_shuffle.edit_scope == "mixed"
        assert recall_shuffle.user_editable_keys == ("probability", "wrong_word_map", "shuffle_types")
        assert hub.resolve_module_view("webui_rate_limit").values["auth_request_limit"] == 12
        assert hub.resolve_module_view("webui_websocket").values["auth_max_connections_per_user"] == 7
        assert hub.resolve_module_view("webui_git_mirror").values["clone_timeout_seconds"] == 420
        webui_thresholds = hub.resolve_module_view("webui_state_monitor_thresholds")
        assert webui_thresholds.values["emotion"]["boredom_show"] == 0.41
        assert webui_thresholds.values["prediction"]["probability_high"] == 0.74
        assert webui_thresholds.edit_scope == "mixed"
        assert webui_thresholds.user_editable_keys == ("emotion", "relationship", "trauma", "memory", "prediction", "scene", "attention", "circadian")
        assert hub.resolve_module_view("brain_chat_runtime").values["parallel_action_timeout_seconds"] == 88.0
        assert hub.resolve_module_view("brain_planner").values["planner_timeout_seconds"] == 22.0
        assert hub.resolve_module_view("brain_pfc_action").values["llm_timeout_seconds"] == 26.0
        assert hub.resolve_module_view("brain_pfc_action").values["high_risk_threshold"] == 0.41
        assert hub.resolve_module_view("brain_pfc_action").values["farewell_timeout_seconds"] == 19.0
        assert hub.resolve_module_view("brain_pfc_reply").values["duplicate_similarity_threshold"] == 0.91
        assert hub.resolve_module_view("brain_pfc_reply").values["generation_timeout_seconds"] == 44.0
        assert hub.resolve_module_view("brain_waiter").values["desired_timeout_seconds"] == 222
        assert hub.resolve_module_view("brain_pfc_goal").values["max_goals"] == 5
        identity_bot = hub.resolve_module_view("identity_bot")
        assert identity_bot.values["nickname"] == "测试爱丽丝"
        assert identity_bot.values["identity_templates"]["default_persona"] == "你叫{name}。"
        assert identity_bot.edit_scope == "mixed"
        assert identity_bot.system_only_keys == ("bot_id",)
        identity_anchor = hub.resolve_module_view("identity_anchor")
        assert identity_anchor.values["max_history"] == 55
        assert identity_anchor.values["prompt_templates"]["default_persona"] == "anchor:{name}"
        assert identity_anchor.edit_scope == "mixed"
        assert identity_anchor.user_editable_keys == ("prompt_templates",)
        identity_user_persistence = hub.resolve_module_view("identity_user_persistence")
        assert identity_user_persistence.values["storage_dir"] == "tmp/users"
        assert identity_user_persistence.edit_scope == "mixed"
        assert identity_user_persistence.user_editable_keys == ("max_activities",)
        assert hub.resolve_module_view("energy_runtime").values["chat_value"]["ceiling"] == 123.0
        assert hub.resolve_module_view("energy_runtime").values["decay"]["annoyance_decay_per_min"] == 4.2
        assert hub.resolve_module_view("energy_runtime").values["penalty_caps"]["trauma_cap"] == 0.71
        assert hub.resolve_module_view("trigger_runtime").values["energy_overflow_ratio"] == 0.91
        personality_factors = hub.resolve_module_view("personality_factors")
        assert personality_factors.values["social_warmth"] == 0.77
        assert personality_factors.edit_scope == "user"
        assert "social_warmth" in personality_factors.user_editable_keys
        personality = hub.resolve_module_view("personality")
        assert personality.values["persona_pool_capacity"] == 12
        assert personality.edit_scope == "mixed"
        assert personality.user_editable_keys == ("default_blend_duration_sec", "persona_auto_revert_sec", "persona_pool_capacity")
        social_calculator = hub.resolve_module_view("social_calculator")
        assert social_calculator.values["positive_streak_coeff"] == 1.9
        assert social_calculator.values["categories"]["hostile"]["acceleration_multiplier"] == 3.8
        assert social_calculator.values["intents"]["flirt"] == 0.95
        assert social_calculator.values["types"]["existence_threat"]["bonus"] == 2.2
        assert social_calculator.edit_scope == "system"
        assert hub.resolve_module_view("social_settlement").values["absolute_step_ceil"] == 12.0
        assert hub.resolve_module_view("social_phase_tracker").values["trend_window_sec"] == 2222.0
        assert hub.resolve_module_view("social_phase_tracker").values["phase_weights"]["trusted"] == 1.23
        assert hub.resolve_module_view("social_affect_fuser").values["support_relief_cap"] == 0.25
        assert hub.resolve_module_view("social_value_core").values["social_new_user_initial"] == 6.5
        assert hub.resolve_module_view("perception_group_sense").values["burst_msg_count"] == 21
        assert hub.resolve_module_view("perception_self_sense").values["max_history_size"] == 66
        assert hub.resolve_module_view("perception_user_relation").values["familiar_threshold"] == 9
        assert hub.resolve_module_view("perception_user_state").values["model_weight"] == 0.7
        assert hub.resolve_module_view("perception_user_state").values["verdict_thresholds"]["want_to_chat"] == 0.68
        affection_dynamics = hub.resolve_module_view("affection_dynamics")
        assert affection_dynamics.values["max_annoyance"] == 88.0
        assert affection_dynamics.edit_scope == "mixed"
        persona_controller = hub.resolve_module_view("persona_controller")
        assert persona_controller.values["max_personas"] == 12
        assert persona_controller.edit_scope == "mixed"
        persona_generator = hub.resolve_module_view("persona_generator")
        assert persona_generator.values["switch_cooldown"] == 240.0
        assert persona_generator.edit_scope == "mixed"
        persona_switcher = hub.resolve_module_view("persona_switcher")
        assert persona_switcher.values["transition_duration_seconds"] == 44.0
        assert persona_switcher.edit_scope == "mixed"
        assert hub.resolve_module_view("perception_behavior").values["memory_cap"] == 77
        assert hub.resolve_module_view("perception_buffer").values["recent_show_count"] == 8
        assert hub.resolve_module_view("perception_group_atmosphere").values["max_history"] == 66
        assert hub.resolve_module_view("perception_interest").values["high_threshold"] == 0.82
        assert hub.resolve_module_view("perception_message_preprocessor").values["message_text_max_length"] == 88
        assert hub.resolve_module_view("perception_message_preprocessor").values["bot_aliases"] == ["阿活"]
        assert hub.resolve_module_view("perception_signal_detector").values["strong_at_threshold"] == 2
        assert hub.resolve_module_view("perception_understand").values["cache_ttl"] == 88.0
        skill = hub.resolve_module_view("skill")
        assert skill.values["trial_to_active_invocations"] == 6
        assert skill.edit_scope == "mixed"
        assert skill.user_editable_keys == ("sticker_daily_ceiling", "sticker_dispatch_threshold")
        adaptive_learning = hub.resolve_module_view("adaptive_learning")
        assert adaptive_learning.values["unknown_term_min_count"] == 4
        assert adaptive_learning.values["expression_learn_temperature"] == 0.25
        assert adaptive_learning.values["jargon_infer_thresholds"] == [2, 5, 10]
        assert adaptive_learning.edit_scope == "mixed"
        assert adaptive_learning.user_editable_keys == ("priority_feedback_enabled", "vocabulary_limit", "mood_vocabulary_limit")

        sleep(1.1)
        _write(config_path, _config(0.35))
        os.utime(config_path, None)
        assert hub.hot_reload() is True
        reloaded = hub.resolve_module_view("vision", scenario="image_high_risk")
        assert reloaded.values["risk_score"] == 0.35
        sleep(1.1)
        _write(config_sidecar_path, _social_calculator_file(4.1, 1.05, 2.4))
        os.utime(config_sidecar_path, None)
        assert hub.hot_reload() is True
        social_reloaded = hub.resolve_module_view("social_calculator")
        assert social_reloaded.values["categories"]["hostile"]["acceleration_multiplier"] == 4.1
        assert social_reloaded.values["intents"]["flirt"] == 1.05
        assert social_reloaded.values["types"]["existence_threat"]["bonus"] == 2.4

        result = {
            "memory_keys": sorted(memory.values.keys()),
            "vision_risk_score": reloaded.values["risk_score"],
            "schedule_fallback_used": schedule.fallback_used,
            "schedule_edit_scope": schedule.edit_scope,
            "schedule_cooldown_sec": hub.assemble_scheduler_config()["cooldown_sec"],
            "model_fail_threshold": model_routing.values["model_fail_threshold"],
            "llm_default_no_thinking": llm_runtime["default_no_thinking"],
            "llm_response_timeout": llm_runtime["response_timeout"],
            "llm_response_retry": llm_runtime["response_retry"],
            "model_routing_scope": model_routing.edit_scope,
            "runtime_tuning_scope": runtime_tuning.edit_scope,
            "context_reply_tokens": hub.resolve_module_view("context").values["reply_context_max_tokens"],
            "context_scope": context.edit_scope,
            "phase_idle_timeout": phase_timing.values["phase_idle_timeout_seconds"],
            "phase_timing_scope": phase_timing.edit_scope,
            "memory_rrf_k": memory_retrieval.values["rrf_k"],
            "memory_retrieval_scope": memory_retrieval.edit_scope,
            "memory_capacity_scope": memory_capacity.edit_scope,
            "memory_decay_scope": memory_decay.edit_scope,
            "memory_dedup_scope": memory_dedup.edit_scope,
            "inner_voice_mention_floor": hub.resolve_module_view("inner_voice").values["mention_min_desire"],
            "frequency_skip_limit": hub.resolve_module_view("frequency_control").values["skip_limit"],
            "harassment_block_threshold": hub.resolve_module_view("harassment_detection").values["block_threshold"],
            "psych_irritated_threshold": hub.resolve_module_view("psychological_core").values["response_irritated_annoyance"],
            "trauma_query_limit": hub.resolve_module_view("trauma_timeline").values["default_query_limit"],
            "recall_typo_limit": hub.resolve_module_view("recall_typo").values["max_typo_length"],
            "recall_correction_timeout": hub.resolve_module_view("recall_correction").values["timeout_seconds"],
            "webui_auth_limit": hub.resolve_module_view("webui_rate_limit").values["auth_request_limit"],
            "webui_prediction_high": webui_thresholds.values["prediction"]["probability_high"],
            "webui_threshold_scope": webui_thresholds.edit_scope,
            "brain_planner_timeout": hub.resolve_module_view("brain_planner").values["planner_timeout_seconds"],
            "brain_waiter_timeout": hub.resolve_module_view("brain_waiter").values["desired_timeout_seconds"],
            "identity_bot_name": hub.resolve_module_view("identity_bot").values["nickname"],
            "identity_bot_scope": identity_bot.edit_scope,
            "identity_anchor_scope": identity_anchor.edit_scope,
            "identity_storage_dir": identity_user_persistence.values["storage_dir"],
            "identity_user_persistence_scope": identity_user_persistence.edit_scope,
            "energy_chat_ceiling": hub.resolve_module_view("energy_runtime").values["chat_value"]["ceiling"],
            "trigger_overflow_ratio": hub.resolve_module_view("trigger_runtime").values["energy_overflow_ratio"],
            "social_streak_coeff": hub.resolve_module_view("social_calculator").values["positive_streak_coeff"],
            "social_hostile_accel": social_reloaded.values["categories"]["hostile"]["acceleration_multiplier"],
            "social_flirt_multiplier": social_reloaded.values["intents"]["flirt"],
            "social_threat_bonus": social_reloaded.values["types"]["existence_threat"]["bonus"],
            "social_settlement_cap": hub.resolve_module_view("social_settlement").values["absolute_step_ceil"],
            "social_phase_trend_window": hub.resolve_module_view("social_phase_tracker").values["trend_window_sec"],
            "social_trusted_weight": hub.resolve_module_view("social_phase_tracker").values["phase_weights"]["trusted"],
            "social_support_cap": hub.resolve_module_view("social_affect_fuser").values["support_relief_cap"],
            "social_core_initial": hub.resolve_module_view("social_value_core").values["social_new_user_initial"],
            "personality_scope": personality_factors.edit_scope,
            "persona_runtime_scope": personality.edit_scope,
            "memory_scope": memory.edit_scope,
            "heartflow_decision_scope": heartflow_decision.edit_scope,
            "proactive_scope": proactive_decider.edit_scope,
            "perception_burst_count": hub.resolve_module_view("perception_group_sense").values["burst_msg_count"],
            "persona_pool_cap": hub.resolve_module_view("persona_controller").values["max_personas"],
            "perception_behavior_cap": hub.resolve_module_view("perception_behavior").values["memory_cap"],
            "skill_trial_invocations": hub.resolve_module_view("skill").values["trial_to_active_invocations"],
            "adaptive_learning_limit": hub.resolve_module_view("adaptive_learning").values["max_learning_items"],
            "jargon_thresholds": hub.resolve_module_view("adaptive_learning").values["jargon_infer_thresholds"],
            "trace_layers": sorted({item.layer for item in vision.trace}),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        CoreSettingsHub.teardown()
        stamped_config_dir = tmp_path / "stamped_config"
        stamped_hub = CoreSettingsHub.boot(
            config_dir=stamped_config_dir,
            template_dir=template_dir,
        )
        assert (stamped_config_dir / "core_config.toml").exists()
        assert (stamped_config_dir / "profile_domains" / "social" / "social_calculator.toml").exists()
        assert stamped_hub.resolve_module_view("social_calculator").values["types"]["existence_threat"]["bonus"] == 2.2
    CoreSettingsHub.teardown()


if __name__ == "__main__":
    main()

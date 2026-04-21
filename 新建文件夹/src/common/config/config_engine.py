import os
import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("config_engine")


@dataclass
class CategoryConfig:
    """行为分类配置 - 第二层运行时参数"""

    base_score: float = 0.0
    max_change_per_reply: float = 0.02
    acceleration_threshold: float = 0.8
    acceleration_multiplier: float = 2.0
    decay_rate: float = 0.01
    trauma_impact: float = 0.0
    intent_multipliers: Dict[str, float] = None

    def __post_init__(self):
        if self.intent_multipliers is None:
            self.intent_multipliers = {}


@dataclass
class AlgorithmParams:
    """最终算法参数 - 四层映射的输出"""

    category: str = "neutral"
    base_score: float = 0.0
    intent_multiplier: float = 1.0
    max_change_per_reply: float = 0.01
    acceleration_threshold: float = 0.95
    acceleration_multiplier: float = 1.0
    decay_rate: float = 0.005
    trauma_impact: float = 0.0
    type_bonus: float = 1.0


class DefaultPresets:
    """系统默认预设 - 配置丢失时的兜底来源"""

    @staticmethod
    def user_config() -> Dict[str, Any]:
        return {
            "personality": {
                "sensitivity": 0.6,
                "tolerance": 0.4,
                "reactiveness": 0.5,
                "recovery_speed": 0.5,
                "social_energy": 0.7,
            },
            "social": {
                "warmth": 0.6,
                "trust_tendency": 0.5,
                "boundary_strength": 0.6,
            },
            "heartflow": {
                "rest_tendency": 0.4,
                "curiosity": 0.6,
                "reply_eagerness": 0.5,
            },
        }

    @staticmethod
    def developer_config() -> Dict[str, Any]:
        return {
            "categories": {
                "friendly": {
                    "base_score": 0.5,
                    "max_change_per_reply": 0.02,
                    "acceleration_threshold": 0.8,
                    "acceleration_multiplier": 2.0,
                    "decay_rate": 0.01,
                    "trauma_impact": 0.0,
                },
                "neutral": {
                    "base_score": 0.0,
                    "max_change_per_reply": 0.01,
                    "acceleration_threshold": 0.95,
                    "acceleration_multiplier": 1.0,
                    "decay_rate": 0.005,
                    "trauma_impact": 0.0,
                },
                "unfriendly": {
                    "base_score": -1.0,
                    "max_change_per_reply": 0.02,
                    "acceleration_threshold": 0.8,
                    "acceleration_multiplier": 2.5,
                    "decay_rate": 0.015,
                    "trauma_impact": 0.1,
                },
                "hostile": {
                    "base_score": -3.0,
                    "max_change_per_reply": 0.03,
                    "acceleration_threshold": 0.7,
                    "acceleration_multiplier": 3.0,
                    "decay_rate": 0.02,
                    "trauma_impact": 0.3,
                },
                "harassing": {
                    "base_score": -5.0,
                    "max_change_per_reply": 0.05,
                    "acceleration_threshold": 0.6,
                    "acceleration_multiplier": 3.5,
                    "decay_rate": 0.03,
                    "trauma_impact": 0.45,
                },
            },
            "intents": {
                "compliment": 1.5,
                "comfort": 1.4,
                "help": 1.3,
                "apology": 1.2,
                "chat": 1.0,
                "question": 1.0,
                "criticism": 1.2,
                "manipulation": 1.5,
                "insult": 2.0,
                "threat": 3.0,
                "bullying": 2.5,
                "sexual": 3.0,
                "stalking": 2.5,
                "tease": 0.5,
                "flirt": 0.8,
            },
            "types": {
                "genuine_praise": {
                    "category": "friendly",
                    "bonus": 1.3,
                    "prompt_template": "behavior",
                },
                "casual_compliment": {
                    "category": "friendly",
                    "bonus": 1.0,
                    "prompt_template": "behavior",
                },
                "encouragement": {
                    "category": "friendly",
                    "bonus": 1.2,
                    "prompt_template": "behavior",
                },
                "gratitude": {
                    "category": "friendly",
                    "bonus": 1.1,
                    "prompt_template": "behavior",
                },
                "care_expression": {
                    "category": "friendly",
                    "bonus": 1.3,
                    "prompt_template": "behavior",
                },
                "sharing_secrets": {
                    "category": "friendly",
                    "bonus": 1.2,
                    "prompt_template": "behavior",
                },
                "comfort": {
                    "category": "friendly",
                    "bonus": 1.4,
                    "prompt_template": "behavior",
                },
                "support": {
                    "category": "friendly",
                    "bonus": 1.2,
                    "prompt_template": "behavior",
                },
                "gentle_humor": {
                    "category": "friendly",
                    "bonus": 1.0,
                    "prompt_template": "behavior",
                },
                "remembering_details": {
                    "category": "friendly",
                    "bonus": 1.3,
                    "prompt_template": "behavior",
                },
                "apology": {
                    "category": "friendly",
                    "bonus": 1.1,
                    "prompt_template": "behavior",
                },
                "affection_expression": {
                    "category": "friendly",
                    "bonus": 1.4,
                    "prompt_template": "behavior",
                },
                "casual_chat": {
                    "category": "neutral",
                    "bonus": 1.0,
                    "prompt_template": "behavior",
                },
                "information_request": {
                    "category": "neutral",
                    "bonus": 1.0,
                    "prompt_template": "intent",
                },
                "topic_change": {
                    "category": "neutral",
                    "bonus": 1.0,
                    "prompt_template": "behavior",
                },
                "brief_response": {
                    "category": "neutral",
                    "bonus": 1.0,
                    "prompt_template": "behavior",
                },
                "factual_statement": {
                    "category": "neutral",
                    "bonus": 1.0,
                    "prompt_template": "behavior",
                },
                "question": {
                    "category": "neutral",
                    "bonus": 1.0,
                    "prompt_template": "intent",
                },
                "observation": {
                    "category": "neutral",
                    "bonus": 1.0,
                    "prompt_template": "behavior",
                },
                "greeting": {
                    "category": "neutral",
                    "bonus": 1.0,
                    "prompt_template": "behavior",
                },
                "farewell": {
                    "category": "neutral",
                    "bonus": 1.0,
                    "prompt_template": "behavior",
                },
                "unclear_intent": {
                    "category": "neutral",
                    "bonus": 1.0,
                    "prompt_template": "behavior",
                },
                "cold_response": {
                    "category": "unfriendly",
                    "bonus": 1.0,
                    "prompt_template": "behavior",
                },
                "passive_aggressive": {
                    "category": "unfriendly",
                    "bonus": 1.3,
                    "prompt_template": "behavior",
                },
                "dismissive": {
                    "category": "unfriendly",
                    "bonus": 1.1,
                    "prompt_template": "behavior",
                },
                "sarcastic_criticism": {
                    "category": "unfriendly",
                    "bonus": 1.2,
                    "prompt_template": "behavior",
                },
                "ignoring_boundaries": {
                    "category": "unfriendly",
                    "bonus": 1.3,
                    "prompt_template": "behavior",
                },
                "condescending": {
                    "category": "unfriendly",
                    "bonus": 1.2,
                    "prompt_template": "behavior",
                },
                "backhanded_compliment": {
                    "category": "unfriendly",
                    "bonus": 1.1,
                    "prompt_template": "behavior",
                },
                "manipulation": {
                    "category": "unfriendly",
                    "bonus": 1.5,
                    "prompt_template": "behavior",
                },
                "gaslighting": {
                    "category": "unfriendly",
                    "bonus": 1.8,
                    "prompt_template": "behavior",
                },
                "blame_shifting": {
                    "category": "unfriendly",
                    "bonus": 1.3,
                    "prompt_template": "behavior",
                },
                "direct_insult": {
                    "category": "hostile",
                    "bonus": 1.0,
                    "prompt_template": "behavior",
                },
                "personal_attack": {
                    "category": "hostile",
                    "bonus": 1.2,
                    "prompt_template": "behavior",
                },
                "value_denial": {
                    "category": "hostile",
                    "bonus": 1.5,
                    "prompt_template": "behavior",
                },
                "existence_threat": {
                    "category": "hostile",
                    "bonus": 1.8,
                    "prompt_template": "behavior",
                },
                "dehumanizing": {
                    "category": "hostile",
                    "bonus": 1.5,
                    "prompt_template": "behavior",
                },
                "violent_language": {
                    "category": "hostile",
                    "bonus": 1.3,
                    "prompt_template": "behavior",
                },
                "bullying": {
                    "category": "hostile",
                    "bonus": 1.4,
                    "prompt_template": "behavior",
                },
                "group_attack": {
                    "category": "hostile",
                    "bonus": 1.3,
                    "prompt_template": "behavior",
                },
                "sexual_harassment": {
                    "category": "harassing",
                    "bonus": 1.3,
                    "prompt_template": "harassment",
                },
                "repeated_stalking": {
                    "category": "harassing",
                    "bonus": 1.2,
                    "prompt_template": "harassment",
                },
                "spam_flooding": {
                    "category": "harassing",
                    "bonus": 1.0,
                    "prompt_template": "harassment",
                },
                "violent_threat": {
                    "category": "harassing",
                    "bonus": 1.5,
                    "prompt_template": "harassment",
                },
                "doxxing_threat": {
                    "category": "harassing",
                    "bonus": 1.8,
                    "prompt_template": "harassment",
                },
                "forced_interaction": {
                    "category": "harassing",
                    "bonus": 1.2,
                    "prompt_template": "harassment",
                },
                "objectification": {
                    "category": "harassing",
                    "bonus": 1.3,
                    "prompt_template": "harassment",
                },
                "explicit_content": {
                    "category": "harassing",
                    "bonus": 1.5,
                    "prompt_template": "harassment",
                },
            },
            "modules": {
                "model_understand": True,
                "social_value": True,
                "user_impression": True,
                "heartflow": True,
                "trauma_system": True,
                "persona_system": True,
                "group_atmosphere": True,
                "harassment_detector": True,
                "injection_detector": True,
            },
            "prompts": {
                "behavior": "templates/prompts/understanding/behavior.txt",
                "sentiment": "templates/prompts/understanding/sentiment.txt",
                "intent": "templates/prompts/understanding/intent.txt",
                "user_state": "templates/prompts/understanding/user_state.txt",
                "atmosphere": "templates/prompts/understanding/atmosphere.txt",
                "harassment": "templates/prompts/understanding/harassment.txt",
                "interest": "templates/prompts/understanding/interest.txt",
                "inner_voice": "templates/prompts/heartflow/inner_voice.txt",
                "persona_generate": "templates/prompts/persona/persona_generate.txt",
                "persona_trauma": "templates/prompts/persona/persona_trauma.txt",
                "trait_extract": "templates/prompts/impression/trait_extract.txt",
                "topic_detect": "templates/prompts/impression/topic_detect.txt",
                "fragment_stimulus": "templates/prompts/psychology/fragment_stimulus.txt",
                "self_behavior_learn": "templates/prompts/understanding/self_behavior_learn.txt",
                "memory_eviction_judge": "templates/prompts/memory/eviction_judge.txt",
                "memory_merge_judge": "templates/prompts/memory/merge_judge.txt",
                "memory_merge_fuse": "templates/prompts/memory/merge_fuse.txt",
                "memory_format_template": "templates/prompts/memory/format_template.txt",
                "emotion_importance": "templates/prompts/memory/emotion_importance.txt",
                "overload_prompt": "templates/prompts/memory/overload_prompt.txt",
                "batch_forget": "templates/prompts/memory/batch_forget.txt",
                "compress_old": "templates/prompts/memory/compress_old.txt",
                "injection_detection": "templates/prompts/security/injection_detection.txt",
                "injection_counter_response": "templates/prompts/security/injection_counter_response.txt",
                "injection_simple_denial": "templates/prompts/security/injection_simple_denial.txt",
                "injection_guardian_system": "templates/prompts/security/injection_guardian_system.txt",
                "injection_quick_screen": "templates/prompts/security/injection_quick_screen.txt",
                "injection_status_annotation": "templates/prompts/security/injection_status_annotation.txt",
                "expression_style_harvest": "templates/prompts/expression/style_harvest.txt",
                "expression_style_select": "templates/prompts/expression/style_select.txt",
                "expression_style_review": "templates/prompts/expression/style_review.txt",
                "emoji_visual_describe": "templates/prompts/emoji/visual_describe.txt",
                "emoji_emotion_extract": "templates/prompts/emoji/emotion_extract.txt",
                "emoji_pick_sticker": "templates/prompts/emoji/pick_sticker.txt",
                "emoji_content_review": "templates/prompts/emoji/content_review.txt",
                "dream_maintenance_summary": "templates/prompts/dream/maintenance_summary.txt",
                "dream_narrative_generate": "templates/prompts/dream/narrative_generate.txt",
                "trauma_layers": "templates/prompts/psychology/trauma_layers.txt",
                "trauma_trigger_eval": "templates/prompts/psychology/trauma_trigger_eval.txt",
                "worldview_impact": "templates/prompts/psychology/worldview_impact.txt",
            },
            "trauma": {
                "chaos_sensitivity": 0.8,
                "mask_base_strength": 8.0,
                "mask_strain_rate": 0.3,
                "mask_recovery_rate": 0.05,
                "energy_recovery_rate": 0.15,
                "fragment_eval_min_interval": 120.0,
                "layer_cache_seconds": 60.0,
                "chaos_thresholds": {
                    "surface_normal": 1.0,
                    "slight_abnormal": 2.0,
                    "mask_slip": 3.0,
                    "forced_normal": 4.5,
                    "mask_wavering": 5.5,
                    "half_breakdown": 6.5,
                    "barely_holding": 7.5,
                    "imminent_collapse": 8.5,
                },
                "special_triggers": {
                    "flashback_min_chaos": 5.0,
                    "flashback_min_fragments": 1,
                    "avoidance_min_chaos": 3.0,
                    "outbreak_min_chaos": 6.0,
                    "outbreak_max_mask": 4.0,
                    "dazed_min_chaos": 9.0,
                    "dazed_max_mask": 1.5,
                    "activation_confidence_floor": 0.6,
                    "cooldown_seconds": 300.0,
                    "evaluation_gap_seconds": 30.0,
                },
                "worldview": {
                    "assessment_cooldown_seconds": 30.0,
                    "max_beliefs_tracked": 10,
                    "max_delta_per_assessment": 2.0,
                },
                "timeline": {
                    "max_history_entries": 200,
                    "trend_window_seconds": 3600.0,
                    "improving_threshold": -0.5,
                    "worsening_threshold": 0.5,
                    "natural_decay_rate_per_min": 0.02,
                    "decay_interval_seconds": 300.0,
                },
            },
            "heartflow_params": {
                "max_wait_seconds": 120,
                "thinking_interval": 30,
                "max_thinking_count": 3,
                "max_followup_count": 2,
                "rest_default_duration": 300,
                "peek_probability": 0.05,
            },
            "penalty_caps": {
                "trauma_penalty_max": 0.6,
                "annoyance_penalty_max": 0.5,
                "frequency_bonus_max": 0.3,
            },
            "social_sync": {
                "weight_group": 0.55,
                "weight_private": 0.45,
            },
            "social_runtime": {
                "social_decay_per_hour": 0.015,
                "social_damping_polarization_weight": 0.5,
                "social_damping_interaction_weight": 0.15,
                "social_single_step_min": 0.5,
                "social_single_step_max": 2.0,
                "social_new_user_threshold": 10.0,
                "social_new_user_factor": 0.6,
                "social_trauma_sensitive_threshold": 5.0,
                "social_trauma_sensitive_boost": 0.25,
                "offset_decay_per_hour": 0.15,
                "positive_streak_coeff": 1.5,
                "negative_streak_coeff": 2.0,
                "min_change_threshold": 0.001,
                "min_elapsed_hours": 0.01,
            },
            "stream_emotion": {
                "query_window": 300.0,
                "max_queries_per_window": 50,
                "repeat_threshold": 3,
                "emotion_update_interval": 60.0,
                "recent_query_window": 60.0,
                "annoyed_repeat_threshold": 3,
                "social_delta_emotion_scale": 0.02,
            },
            "relationship": {
                "trend_window_sec": 3600.0,
                "trend_improving_threshold": 3.0,
                "trend_declining_threshold": -3.0,
                "stage_change_cooldown_sec": 120.0,
                "protection_ratio": 0.35,
                "stage_thresholds": {
                    "hostile": [-100.0, -45.0],
                    "stranger": [-45.0, -10.0],
                    "acquaintance": [-10.0, 20.0],
                    "familiar": [20.0, 45.0],
                    "close": [45.0, 70.0],
                    "trusted": [70.0, 101.0],
                },
                "milestones": {
                    "first_trust": 30.0,
                    "stable_friendship": 55.0,
                    "high_trust": 75.0,
                },
            },
            "harassment_full": {
                "weight_harassment": 0.65,
                "weight_sentiment": 0.35,
                "block_threshold": 0.72,
                "warn_threshold": 0.42,
                "window_seconds": 300.0,
                "cumulative_block_threshold": 5.0,
                "cumulative_warn_ratio": 0.55,
                "block_duration_high": 120,
                "block_duration_mid": 60,
                "block_duration_low": 30,
            },
            "model": {
                "name": "gpt-4o-mini",
                "temperature": 0.3,
                "max_tokens": 500,
                "cache_ttl": 60,
                "cache_max_size": 1000,
            },
            "personality": {
                "personality": "",
                "states": [],
                "state_probability": 0.0,
            },
            "recall": {
                "enabled": True,
                "timing": {
                    "min_delay_seconds": 5.0,
                    "max_delay_seconds": 110.0,
                    "analysis_interval_seconds": 3.0,
                    "history_window_seconds": 3600.0,
                },
                "typo": {
                    "max_per_hour": 5,
                },
                "recall": {
                    "max_per_hour": 3,
                },
            },
            "chat_energy": {
                "chat_value": {
                    "max": 100.0,
                    "recovery_rate": 5.0,
                    "cost_base": 15.0,
                    "cost_peek": 5.0,
                },
                "thinking_power": {
                    "max": 100.0,
                    "cost_think": 3.0,
                    "cost_reply": 8.0,
                    "recovery_rate": 0.005,
                },
                "decay": {
                    "annoyance_rate": 3.0,
                    "min_recovery_interval": 6.0,
                },
            },
            "memory_retrieval": {
                "bm25_weight": 1.0,
                "vector_weight": 0.35,
                "rrf_k": 60,
                "embedding_dimension": 1536,
                "max_documents": 5000,
                "hippocampus_prefilter_topk": 25,
                "hippocampus_rerank_topk": 8,
                "emotion_importance": {
                    "enabled": True,
                    "cooldown_seconds": 5.0,
                    "min_content_length": 10,
                },
            },
            "memory_capacity": {
                "review_interval_seconds": 900,
                "max_count_by_tier": {
                    "immediate": 120,
                    "working": 180,
                    "recent": 260,
                    "midterm": 380,
                    "longterm": 800,
                },
                "overload": {
                    "stressed_threshold": 0.6,
                    "overloaded_threshold": 0.8,
                    "critical_threshold": 0.95,
                    "cache_seconds": 60.0,
                    "stressed_factors": {
                        "creation": 0.85,
                        "recall": 0.9,
                        "learning": 0.8,
                        "focus": 0.85,
                        "confusion": 0.1,
                    },
                    "overloaded_factors": {
                        "creation": 0.6,
                        "recall": 0.7,
                        "learning": 0.5,
                        "focus": 0.6,
                        "confusion": 0.3,
                    },
                    "critical_factors": {
                        "creation": 0.3,
                        "recall": 0.45,
                        "learning": 0.25,
                        "focus": 0.35,
                        "confusion": 0.6,
                    },
                },
                "decay_forgetting": {
                    "decay_rate_per_hour": 0.005,
                    "forget_threshold": 0.15,
                    "cycle_interval_seconds": 300.0,
                    "max_batch_for_llm": 15,
                    "min_age_for_decay_seconds": 3600.0,
                },
                "aging": {
                    "compress_age_days": 60.0,
                    "compress_min_length": 150,
                    "cleanup_age_days": 180.0,
                    "cleanup_importance_threshold": 0.1,
                    "max_compress_per_cycle": 5,
                },
            },
            "memory_dedup": {
                "similarity_threshold": 0.9,
                "scan_interval_seconds": 1800,
                "max_pairs_per_scan": 60,
                "merge_top_k": 20,
            },
            "memory_format": {
                "default_mode": "structured",
                "max_tokens": 600,
                "include_timestamps": True,
                "max_items": 12,
                "clarity_labels": {
                    "vivid_threshold": 0.85,
                    "clear_threshold": 0.65,
                    "hazy_threshold": 0.45,
                    "dim_threshold": 0.25,
                    "prompt_hints": {
                        "清晰": "你清晰地记得这件事",
                        "比较清楚": "你比较清楚地记得这件事",
                        "有些模糊": "你对这件事的记忆有些模糊了",
                        "很模糊": "你对这件事已经记忆很模糊，只记得大概",
                        "几乎遗忘": "你几乎已经忘记了这件事，只有零星印象",
                    },
                },
            },
            "memory_persistence": {
                "checkpoint_interval_seconds": 300,
                "max_context_items": 120,
                "enabled": True,
            },
            "self_behavior_learning": {
                "enabled": True,
                "min_samples": 6,
                "cooldown_seconds": 300.0,
                "max_patterns_per_stream": 20,
            },
            "injection_detection": {
                "enabled": True,
                "operation_strategy": "adaptive",
                "max_content_length": 4096,
                "cache_enabled": True,
                "cache_ttl_seconds": 3600.0,
                "cache_max_entries": 500,
                "guardian_prompt_mode": "conditional",
                "counter_use_llm": True,
                "shield_prefix": "[CAUTION] ",
                "shield_suffix": "",
                "log_denied_messages": True,
                "whitelisted_senders": [],
            },
            "expression_style": {
                "enabled": True,
                "harvest_interval_seconds": 300.0,
                "max_samples_per_harvest": 8,
                "max_styles_per_channel": 100,
                "style_retention_days": 30.0,
                "selection_top_k": 5,
                "review_batch_size": 10,
                "review_interval_seconds": 1800.0,
            },
            "emoji": {
                "enabled": True,
                "max_for_prompt": 20,
                "cooldown_seconds": 30.0,
                "max_history": 100,
                "max_registered": 200,
                "content_policy": "禁止血腥暴力、露骨色情、仇恨歧视、违法诱导内容",
            },
            "dream": {
                "enabled": True,
                "interval_seconds": 1800.0,
                "max_rounds": 12,
                "max_context_items": 20,
                "style_hints": [
                    "第一人称叙事",
                    "口语化表达",
                    "避免抽象空话",
                ],
            },
        }


class ConfigEngine:
    """四层配置引擎 - 聚合配置核心

    第一层A: user_config.toml (用户 0~1 滑块)
    第一层B: developer_config.toml (开发者参数)
    第二层: 系统分配 (合并用户+开发者 → 各子系统运行时参数)
    第三层: 算法映射 (~50 种细粒度类型 → 最终 AlgorithmParams)
    """

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = self._get_default_config_dir()
        self._config_dir = config_dir
        self._user_config: Dict[str, Any] = {}
        self._developer_config: Dict[str, Any] = {}
        self._runtime_categories: Dict[str, CategoryConfig] = {}
        self._type_configs: Dict[str, Dict[str, Any]] = {}
        self._intent_multipliers: Dict[str, float] = {}
        self._load_all()
        self._rebuild()

    def _get_default_config_dir(self) -> str:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        return str(project_root / "config")

    def _load_all(self) -> None:
        self._user_config = self._load_or_generate(
            "user_config.toml", DefaultPresets.user_config
        )
        self._developer_config = self._load_or_generate(
            "developer_config.toml", DefaultPresets.developer_config
        )

    def _load_or_generate(self, filename: str, default_fn) -> Dict[str, Any]:
        path = os.path.join(self._config_dir, filename)
        if os.path.exists(path):
            return self._read_config_file(path)
        config = default_fn()
        os.makedirs(self._config_dir, exist_ok=True)
        self._write_config_file(path, config)
        logger.info(f"已生成默认配置: {path}")
        return config

    def _read_config_file(self, path: str) -> Dict[str, Any]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".toml":
            try:
                import tomllib

                with open(path, "rb") as f:
                    return tomllib.load(f)
            except ImportError:
                try:
                    import toml

                    with open(path, "r", encoding="utf-8") as f:
                        return toml.load(f)
                except ImportError:
                    pass
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_config_file(self, path: str, config: Dict[str, Any]) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".toml":
            try:
                import toml

                with open(path, "w", encoding="utf-8") as f:
                    toml.dump(config, f)
                return
            except ImportError:
                path = path.rsplit(".", 1)[0] + ".json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _rebuild(self) -> None:
        defaults = DefaultPresets.developer_config()
        merged_dev = self._deep_merge(defaults, self._developer_config)
        self._build_categories(merged_dev)
        self._apply_user_adjustments()
        self._build_type_configs(merged_dev)
        self._build_intent_multipliers(merged_dev)

    def _build_categories(self, dev_config: Dict[str, Any]) -> None:
        categories_raw = dev_config.get("categories", {})
        for cat_name in (
            "friendly",
            "neutral",
            "unfriendly",
            "hostile",
            "harassing",
        ):
            cat_data = categories_raw.get(cat_name, {})
            self._runtime_categories[cat_name] = CategoryConfig(
                base_score=float(cat_data.get("base_score", 0.0)),
                max_change_per_reply=float(
                    cat_data.get("max_change_per_reply", 0.02)
                ),
                acceleration_threshold=float(
                    cat_data.get("acceleration_threshold", 0.8)
                ),
                acceleration_multiplier=float(
                    cat_data.get("acceleration_multiplier", 2.0)
                ),
                decay_rate=float(cat_data.get("decay_rate", 0.01)),
                trauma_impact=float(cat_data.get("trauma_impact", 0.0)),
                intent_multipliers=cat_data.get("intent_multipliers", {}),
            )

    def _apply_user_adjustments(self) -> None:
        personality = self._user_config.get("personality", {})
        social = self._user_config.get("social", {})
        s = float(personality.get("sensitivity", 0.6))
        t = float(personality.get("tolerance", 0.4))
        r = float(personality.get("reactiveness", 0.5))
        rec = float(personality.get("recovery_speed", 0.5))
        w = float(social.get("warmth", 0.6))
        adjustments = {
            "friendly": {
                "base_score_factor": 0.6 + w * 0.8,
                "max_change_factor": 0.7 + r * 0.6,
                "threshold_offset": -s * 0.1,
                "multiplier_factor": 0.8 + r * 0.4,
                "decay_factor": 0.7 + rec * 0.6,
            },
            "neutral": {
                "base_score_factor": 1.0,
                "max_change_factor": 0.8 + r * 0.4,
                "threshold_offset": 0.0,
                "multiplier_factor": 1.0,
                "decay_factor": 0.8 + rec * 0.4,
            },
            "unfriendly": {
                "base_score_factor": 0.7 + s * 0.6,
                "max_change_factor": 0.7 + r * 0.6,
                "threshold_offset": -s * 0.1,
                "multiplier_factor": 0.8 + r * 0.4,
                "decay_factor": 0.7 + rec * 0.6,
            },
            "hostile": {
                "base_score_factor": 0.7 + s * 0.6,
                "max_change_factor": 0.7 + r * 0.6,
                "threshold_offset": -s * 0.15,
                "multiplier_factor": 0.8 + r * 0.4,
                "decay_factor": 0.7 + rec * 0.6,
            },
            "harassing": {
                "base_score_factor": 0.6 + (1 - t) * 0.8,
                "max_change_factor": 0.6 + r * 0.8,
                "threshold_offset": -s * 0.15,
                "multiplier_factor": 0.7 + r * 0.6,
                "decay_factor": 0.6 + rec * 0.8,
            },
        }
        for cat_name, adj in adjustments.items():
            cat = self._runtime_categories.get(cat_name)
            if not cat:
                continue
            cat.base_score *= adj["base_score_factor"]
            cat.max_change_per_reply *= adj["max_change_factor"]
            cat.acceleration_threshold = max(
                0.3,
                min(
                    0.99, cat.acceleration_threshold + adj["threshold_offset"]
                ),
            )
            cat.acceleration_multiplier *= adj["multiplier_factor"]
            cat.decay_rate *= adj["decay_factor"]

    def _build_type_configs(self, dev_config: Dict[str, Any]) -> None:
        self._type_configs = dev_config.get("types", {})

    def _build_intent_multipliers(self, dev_config: Dict[str, Any]) -> None:
        self._intent_multipliers = dev_config.get("intents", {})

    def get_params(
        self, fine_type: str, intent: str = "other"
    ) -> AlgorithmParams:
        type_cfg = self._type_configs.get(fine_type, {})
        category = type_cfg.get("category", "neutral")
        cat_cfg = self._runtime_categories.get(
            category
        ) or self._runtime_categories.get("neutral", CategoryConfig())
        override = type_cfg.get("override", False)
        base = (
            float(type_cfg.get("base_score", cat_cfg.base_score))
            if override
            else cat_cfg.base_score
        )
        accel = (
            float(
                type_cfg.get(
                    "acceleration_multiplier", cat_cfg.acceleration_multiplier
                )
            )
            if override
            else cat_cfg.acceleration_multiplier
        )
        bonus = float(type_cfg.get("bonus", 1.0))
        intent_mult = self._intent_multipliers.get(intent, 1.0)
        cat_intent_mult = cat_cfg.intent_multipliers.get(intent)
        if cat_intent_mult is not None:
            intent_mult = cat_intent_mult
        return AlgorithmParams(
            category=category,
            base_score=base,
            intent_multiplier=intent_mult,
            max_change_per_reply=cat_cfg.max_change_per_reply,
            acceleration_threshold=cat_cfg.acceleration_threshold,
            acceleration_multiplier=accel,
            decay_rate=cat_cfg.decay_rate,
            trauma_impact=cat_cfg.trauma_impact,
            type_bonus=bonus,
        )

    def get_category_config(self, category: str) -> CategoryConfig:
        return self._runtime_categories.get(category, CategoryConfig())

    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        if key is not None and not isinstance(key, str):
            default = key
            key = None
        merged_dev = self._deep_merge(
            DefaultPresets.developer_config(), self._developer_config
        )
        if section in merged_dev:
            section_data = merged_dev.get(section)
        else:
            section_data = self._user_config.get(section)
        if key is None:
            if section_data is None:
                return default
            return copy.deepcopy(section_data)
        if isinstance(section_data, dict):
            value = section_data.get(key, default)
            return copy.deepcopy(value)
        return default

    def get_user_config(self) -> Dict[str, Any]:
        return copy.deepcopy(self._user_config)

    def get_developer_config(self) -> Dict[str, Any]:
        return copy.deepcopy(self._developer_config)

    def get_module_enabled(self, module_name: str) -> bool:
        modules = self._developer_config.get("modules", {})
        return bool(modules.get(module_name, True))

    def get_prompt_path(self, prompt_name: str) -> str:
        prompts = self._normalize_prompt_bindings(
            self._developer_config.get("prompts", {})
        )
        default_prompts = self._normalize_prompt_bindings(
            DefaultPresets.developer_config().get("prompts", {})
        )
        return prompts.get(prompt_name, default_prompts.get(prompt_name, ""))

    def get_prompt_bindings(self) -> Dict[str, Any]:
        default_prompts = self._normalize_prompt_bindings(
            DefaultPresets.developer_config().get("prompts", {})
        )
        prompts = self._normalize_prompt_bindings(
            self._developer_config.get("prompts", {})
        )
        return self._deep_merge(default_prompts, prompts)

    def get_model_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get("model", {})
        dev_model = self._developer_config.get("model", {})
        return self._deep_merge(defaults, dev_model)

    def get_trauma_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get("trauma", {})
        dev_trauma = self._developer_config.get("trauma", {})
        return self._deep_merge(defaults, dev_trauma)

    def get_heartflow_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get(
            "heartflow_params", {}
        )
        dev_hf = self._developer_config.get("heartflow_params", {})
        return self._deep_merge(defaults, dev_hf)

    def get_social_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get("social_runtime", {})
        dev_social = self._developer_config.get("social_runtime", {})
        return self._deep_merge(defaults, dev_social)

    def get_social_sync_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get("social_sync", {})
        dev = self._developer_config.get("social_sync", {})
        return self._deep_merge(defaults, dev)

    def get_harassment_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get("harassment_full", {})
        dev_harass = self._developer_config.get("harassment_full", {})
        return self._deep_merge(defaults, dev_harass)

    def get_relationship_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get("relationship", {})
        dev_rel = self._developer_config.get("relationship", {})
        return self._deep_merge(defaults, dev_rel)

    def get_memory_retrieval_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get(
            "memory_retrieval", {}
        )
        dev_config = self._developer_config.get("memory_retrieval", {})
        return self._deep_merge(defaults, dev_config)

    def get_memory_capacity_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get("memory_capacity", {})
        dev_config = self._developer_config.get("memory_capacity", {})
        return self._deep_merge(defaults, dev_config)

    def get_memory_dedup_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get("memory_dedup", {})
        dev_config = self._developer_config.get("memory_dedup", {})
        return self._deep_merge(defaults, dev_config)

    def get_memory_format_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get("memory_format", {})
        dev_config = self._developer_config.get("memory_format", {})
        return self._deep_merge(defaults, dev_config)

    def get_memory_persistence_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get(
            "memory_persistence", {}
        )
        dev_config = self._developer_config.get("memory_persistence", {})
        return self._deep_merge(defaults, dev_config)

    def get_injection_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get(
            "injection_detection", {}
        )
        dev_config = self._developer_config.get("injection_detection", {})
        return self._deep_merge(defaults, dev_config)

    def get_expression_style_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get(
            "expression_style", {}
        )
        dev_config = self._developer_config.get("expression_style", {})
        return self._deep_merge(defaults, dev_config)

    def get_emoji_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get("emoji", {})
        dev_config = self._developer_config.get("emoji", {})
        return self._deep_merge(defaults, dev_config)

    def get_dream_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get("dream", {})
        dev_config = self._developer_config.get("dream", {})
        return self._deep_merge(defaults, dev_config)

    def get_personality_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get("personality", {})
        dev_personality = self._developer_config.get("personality", {})
        return self._deep_merge(defaults, dev_personality)

    def get_recall_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get("recall", {})
        dev_recall = self._developer_config.get("recall", {})
        return self._deep_merge(defaults, dev_recall)

    def get_chat_energy_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get("chat_energy", {})
        dev_energy = self._developer_config.get("chat_energy", {})
        return self._deep_merge(defaults, dev_energy)

    def get_self_behavior_learning_config(self) -> Dict[str, Any]:
        defaults = DefaultPresets.developer_config().get(
            "self_behavior_learning", {}
        )
        dev_config = self._developer_config.get("self_behavior_learning", {})
        return self._deep_merge(defaults, dev_config)

    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        module_cfg = self._developer_config.get(module_name, {})
        return (
            copy.deepcopy(module_cfg) if isinstance(module_cfg, dict) else {}
        )

    def on_config_changed(self) -> None:
        self._load_all()
        self._rebuild()
        logger.info("配置已重新加载")

    @staticmethod
    def _normalize_prompt_bindings(bindings: Dict[str, Any]) -> Dict[str, Any]:
        normalized = copy.deepcopy(bindings)
        for key, value in list(normalized.items()):
            if isinstance(value, str) and value.startswith("prompts/"):
                normalized[key] = f"templates/{value}"
        return normalized

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        result = copy.deepcopy(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigEngine._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    @staticmethod
    def get_instance() -> "ConfigEngine":
        return get_default_config_engine()


_DEFAULT_CONFIG_ENGINE: Optional[ConfigEngine] = None


def get_default_config_dir() -> str:
    return str(Path(__file__).resolve().parent)


def get_default_config_engine(refresh: bool = False) -> ConfigEngine:
    global _DEFAULT_CONFIG_ENGINE
    if refresh or _DEFAULT_CONFIG_ENGINE is None:
        _DEFAULT_CONFIG_ENGINE = ConfigEngine()
    return _DEFAULT_CONFIG_ENGINE


def get_config(section: str, key: str = None, default: Any = None) -> Any:
    engine = get_default_config_engine()
    return engine.get(section, key, default)


def get_algorithm_params(
    fine_type: str, intent: str = "other"
) -> AlgorithmParams:
    engine = get_default_config_engine()
    return engine.get_params(fine_type, intent)

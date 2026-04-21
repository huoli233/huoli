from typing import Dict, Any, Optional
from src.common.logger import get_logger

logger = get_logger("behavior_tuner")


def _lerp(low: float, high: float, t: float) -> float:
    """线性插值，t在0-1之间"""
    return low + (high - low) * max(0.0, min(1.0, t))


def _lerp_int(low: int, high: int, t: float) -> int:
    """整数线性插值"""
    return int(round(_lerp(float(low), float(high), t)))


class BehaviorTuner:
    """统一行为调谐器，将10个主控旋钮(0-1)映射到30+个下游参数"""

    def __init__(self):
        self._cfg = None
        self._derived: Dict[str, Any] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            from src.config.config import global_config
            self._cfg = global_config.behavior
        except Exception:
            from src.config.official_configs import BehaviorConfig
            self._cfg = BehaviorConfig()
        self._compute_all()
        self._loaded = True

    def reload(self):
        """重新加载配置并重算所有派生参数"""
        self._loaded = False
        self._ensure_loaded()

    def _compute_all(self):
        c = self._cfg
        d = {}
        d.update(self._compute_activeness(c.activeness))
        d.update(self._compute_endurance(c.endurance))
        d.update(self._compute_sensitivity(c.sensitivity))
        d.update(self._compute_emotionality(c.emotionality))
        d.update(self._compute_caution(c.caution))
        d.update(self._compute_creativity(c.creativity))
        d.update(self._compute_memory_depth(c.memory_depth))
        d.update(self._compute_sociability(c.sociability))
        d.update(self._compute_learning_rate(c.learning_rate))
        d.update(self._compute_rest_tendency(c.rest_tendency))
        self._derived = d
        logger.debug(f"行为参数已计算完成，共{len(d)}个派生参数")

    def _compute_activeness(self, t: float) -> Dict[str, Any]:
        return {
            "base_reply_probability": _lerp(0.005, 0.15, t),
            "proactive_probability": _lerp(0.02, 0.3, t),
            "peek_interval_min": _lerp(15.0, 5.0, t),
            "peek_interval_max": _lerp(25.0, 10.0, t),
            "reply_threshold": _lerp(0.7, 0.3, t),
            "action_threshold": _lerp(0.6, 0.25, t),
            "mention_boost": _lerp(0.3, 0.7, t),
            "post_reply_boost_max": _lerp_int(1, 5, t),
            "threshold_decay_per_skip": _lerp(0.02, 0.08, t),
        }

    def _compute_endurance(self, t: float) -> Dict[str, Any]:
        return {
            "chat_value_max": _lerp(60.0, 150.0, t),
            "brain_power_max": _lerp(60.0, 150.0, t),
            "fatigue_threshold_min": _lerp_int(2, 6, t),
            "fatigue_threshold_max": _lerp_int(5, 15, t),
            "max_consecutive_replies": _lerp_int(2, 10, t),
            "chat_recovery_idle": _lerp(0.15, 0.6, t),
            "chat_recovery_rest": _lerp(1.0, 3.0, t),
            "brain_recovery_rate": _lerp(0.15, 0.6, t),
            "chat_cost_base": _lerp(18.0, 8.0, t),
            "brain_cost_reply": _lerp(8.0, 3.0, t),
            "consecutive_penalty_factor": _lerp(0.6, 0.2, t),
            "stamina_cost_reply": _lerp(8.0, 3.0, t),
            "stamina_cost_peek": _lerp(3.0, 1.0, t),
            "stamina_low_threshold": _lerp(30.0, 15.0, t),
            "stamina_critical_threshold": _lerp(8.0, 3.0, t),
            "mask_energy_max": _lerp(6.0, 14.0, t),
            "mask_recovery_rate": _lerp(0.03, 0.08, t),
        }

    def _compute_sensitivity(self, t: float) -> Dict[str, Any]:
        return {
            "at_boost": _lerp(0.3, 1.0, t),
            "wake_word_boost": _lerp(0.2, 0.8, t),
            "interest_weight": _lerp(0.2, 0.7, t),
            "admin_boost": _lerp(1.2, 3.0, t),
            "spam_penalty_factor": _lerp(0.5, 0.15, t),
            "scoring_semantic_weight": _lerp(0.2, 0.4, t),
            "scoring_keyword_weight": _lerp(0.15, 0.25, t),
            "scoring_sentiment_weight": _lerp(0.1, 0.2, t),
            "scoring_topic_weight": _lerp(0.1, 0.2, t),
            "scoring_relationship_weight": _lerp(0.05, 0.15, t),
            "scoring_mention_weight": _lerp(0.05, 0.15, t),
            "sentiment_positive_threshold": _lerp(0.2, 0.05, t),
            "sentiment_negative_threshold": _lerp(-0.05, -0.2, t),
        }

    def _compute_emotionality(self, t: float) -> Dict[str, Any]:
        return {
            "affection_increase_rate": _lerp(0.03, 0.25, t),
            "affection_decrease_rate": _lerp(0.02, 0.12, t),
            "trauma_recovery_rate": _lerp(0.003, 0.03, t),
            "annoyance_decay_rate": _lerp(0.5, 2.0, t),
            "block_threshold": _lerp(90.0, 60.0, t),
            "block_duration_base": _lerp(30.0, 120.0, t),
            "block_duration_max": _lerp(1800.0, 7200.0, t),
            "global_rage_threshold": _lerp(200.0, 100.0, t),
            "threshold_offset_decay_rate": _lerp(0.01, 0.04, t),
            "annoyance_per_harass": _lerp(8.0, 18.0, t),
            "trauma_per_harass_multiplier": _lerp(0.1, 0.25, t),
            "persona_switch_duration": _lerp(300.0, 900.0, t),
            "mask_fatigue_rate": _lerp(0.15, 0.5, t),
            "mask_breakdown_threshold": _lerp(8.0, 5.5, t),
            "chaos_decay_rate": _lerp(0.15, 0.06, t),
            "flashback_frequency": _lerp(0.05, 0.2, t),
            "awakening_chaos_threshold": _lerp(7.5, 4.5, t),
            "awakening_stress_threshold": _lerp(8.5, 5.5, t),
        }

    def _compute_caution(self, t: float) -> Dict[str, Any]:
        return {
            "risk_threshold": _lerp(0.7, 0.3, t),
            "recall_review_weight": _lerp(0.1, 0.6, t),
            "min_pattern_confidence": _lerp(0.4, 0.8, t),
            "harassment_block_low": _lerp(20.0, 45.0, t),
            "harassment_block_mid": _lerp(40.0, 80.0, t),
            "harassment_block_high": _lerp(80.0, 180.0, t),
            "harassment_switch_threshold_high": _lerp(10.0, 7.0, t),
            "harassment_switch_threshold_mid": _lerp(8.0, 5.0, t),
            "harassment_switch_threshold_low": _lerp(5.0, 3.0, t),
            "confidence_learning_rate": _lerp(0.005, 0.02, t),
            "adaptation_sensitivity": _lerp(0.05, 0.2, t),
            "min_adaptation_interval": _lerp_int(90, 30, t),
            "interaction_time_window": _lerp(90.0, 45.0, t),
            "interaction_max_per_window": _lerp_int(3, 8, t),
            "interaction_cooldown": _lerp(450.0, 180.0, t),
            "interaction_max_warnings": _lerp_int(1, 4, t),
        }

    def _compute_creativity(self, t: float) -> Dict[str, Any]:
        return {
            "multiple_style_probability": _lerp(0.0, 0.6, t),
            "state_probability": _lerp(0.0, 0.6, t),
            "emoji_chance": _lerp(0.1, 0.7, t),
            "typo_error_rate": _lerp(0.0, 0.025, t),
            "typo_word_replace_rate": _lerp(0.0, 0.015, t),
            "response_max_length": _lerp_int(100, 300, t),
            "response_max_sentences": _lerp_int(3, 10, t),
            "playfulness_base": _lerp(0.2, 0.8, t),
        }

    def _compute_memory_depth(self, t: float) -> Dict[str, Any]:
        return {
            "context_size_multiplier": _lerp(0.6, 1.5, t),
            "memory_importance_threshold": _lerp(0.8, 0.4, t),
            "agent_max_iterations": _lerp_int(2, 8, t),
            "agent_timeout_seconds": _lerp(90.0, 240.0, t),
            "memory_save_interval": _lerp(60.0, 20.0, t),
            "consolidation_keep_threshold": _lerp(0.7, 0.5, t),
            "consolidation_delete_threshold": _lerp(0.4, 0.2, t),
            "consolidation_max_per_day": _lerp_int(60, 150, t),
            "deep_optimizer_batch_size": _lerp_int(5, 15, t),
        }

    def _compute_sociability(self, t: float) -> Dict[str, Any]:
        return {
            "proactive_enabled": t > 0.3,
            "proactive_min_interval": _lerp_int(900, 300, t),
            "proactive_daily_max": _lerp_int(2, 10, t),
            "silence_threshold_seconds": _lerp_int(3600, 600, t),
            "proactive_thinking_interval": _lerp_int(180, 60, t),
            "proactive_quiet_start": _lerp_int(23, 24, t),
            "proactive_quiet_end": _lerp_int(7, 6, t),
            "post_reply_threshold_reduction": _lerp(0.05, 0.15, t),
            "post_reply_decay_rate": _lerp(0.8, 0.5, t),
            "relationship_cache_ttl": _lerp_int(600, 180, t),
        }

    def _compute_learning_rate(self, t: float) -> Dict[str, Any]:
        return {
            "learning_trigger_interval": _lerp_int(40, 8, t),
            "max_vocabulary": _lerp_int(200, 800, t),
            "jargon_inference_limit": _lerp_int(5, 20, t),
            "expression_similarity_threshold": _lerp(0.85, 0.65, t),
            "expression_min_count": _lerp_int(3, 1, t),
            "expression_select_limit": _lerp_int(3, 8, t),
            "expression_auto_check_interval": _lerp_int(1200, 300, t),
            "expression_auto_check_count": _lerp_int(10, 30, t),
        }

    def _compute_rest_tendency(self, t: float) -> Dict[str, Any]:
        return {
            "fatigue_per_reply": _lerp(20.0, 45.0, t),
            "fatigue_accumulated_threshold": _lerp(160.0, 80.0, t),
            "fatigue_max_duration": _lerp(120.0, 300.0, t),
            "fatigue_decay_rate": _lerp(1.5, 0.5, t),
            "mental_fatigue_cost_base": _lerp(3.0, 8.0, t),
            "mental_fatigue_recovery_amount": _lerp(8.0, 3.0, t),
            "mental_fatigue_recovery_interval": _lerp(30.0, 90.0, t),
            "sleep_consolidation_hour_start": _lerp_int(3, 1, t),
            "sleep_consolidation_hour_end": _lerp_int(5, 7, t),
        }

    def get(self, key: str, default: Any = None) -> Any:
        """获取派生参数值"""
        self._ensure_loaded()
        return self._derived.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        """获取所有派生参数"""
        self._ensure_loaded()
        return dict(self._derived)

    def get_raw(self, knob_name: str) -> float:
        """获取原始旋钮值"""
        self._ensure_loaded()
        return getattr(self._cfg, knob_name, 0.5)

    def dump_summary(self) -> str:
        """输出所有参数摘要"""
        self._ensure_loaded()
        lines = ["=== 行为旋钮 ==="]
        for attr in [
            'activeness', 'endurance', 'sensitivity', 'emotionality',
            'caution', 'creativity', 'memory_depth', 'sociability',
            'learning_rate', 'rest_tendency',
        ]:
            lines.append(f"  {attr}: {getattr(self._cfg, attr, 0.5):.2f}")
        lines.append(f"=== 派生参数({len(self._derived)}个) ===")
        for k, v in sorted(self._derived.items()):
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


_tuner: Optional[BehaviorTuner] = None


def get_behavior_tuner() -> BehaviorTuner:
    global _tuner
    if _tuner is None:
        _tuner = BehaviorTuner()
    return _tuner

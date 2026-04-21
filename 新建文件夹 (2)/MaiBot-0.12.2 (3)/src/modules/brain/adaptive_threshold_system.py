import time
import math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from src.common.logger import get_logger

logger = get_logger("adaptive_threshold")


@dataclass
class ThresholdMetrics:
    threshold_name: str
    current_value: float
    baseline_value: float
    adaptation_rate: float
    last_updated: float
    performance_score: float = 0.0
    adjustment_count: int = 0
    stability_score: float = 1.0


@dataclass
class PerformanceData:
    decision_quality: float = 0.0
    response_time: float = 0.0
    user_satisfaction: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ThresholdAdjustment:
    threshold_name: str
    old_value: float
    new_value: float
    reason: str
    performance_impact: float
    timestamp: float = field(default_factory=time.time)


class AdaptiveThresholdSystem:
    def __init__(self):
        from src.config.config import global_config
        brain_cfg = global_config.brain
        self._thresholds: Dict[str, ThresholdMetrics] = {}
        self._performance_history: deque = deque(maxlen=brain_cfg.max_decision_history)
        self._adjustments_history: List[ThresholdAdjustment] = []
        self._max_adjustments_history = 500
        self._initialize_default_thresholds()
        try:
            from src.config.behavior_tuner import get_behavior_tuner
            _bt = get_behavior_tuner()
            self._adaptation_sensitivity = _bt.get("adaptation_sensitivity", brain_cfg.adaptation_sensitivity)
            self._min_adaptation_interval = _bt.get("min_adaptation_interval", brain_cfg.min_adaptation_interval)
        except Exception:
            self._adaptation_sensitivity = brain_cfg.adaptation_sensitivity
            self._min_adaptation_interval = brain_cfg.min_adaptation_interval
        self._max_threshold_change = 0.2
        self._stability_window = 10
        logger.info("自适应阈值系统初始化完成")

    def _initialize_default_thresholds(self):
        default_thresholds = {
            "reply_confidence_threshold": {"baseline": 0.6},
            "risk_tolerance_threshold": {"baseline": 0.4},
            "engagement_threshold": {"baseline": 0.5},
            "creativity_threshold": {"baseline": 0.3},
            "reply_frequency_threshold": {"baseline": 0.7},
            "cooldown_threshold": {"baseline": 30.0},
            "content_quality_threshold": {"baseline": 0.6},
            "relevance_threshold": {"baseline": 0.5},
            "emotional_response_threshold": {"baseline": 0.4},
            "mood_stability_threshold": {"baseline": 0.7},
        }
        for name, config in default_thresholds.items():
            self._thresholds[name] = ThresholdMetrics(
                threshold_name=name,
                current_value=config["baseline"],
                baseline_value=config["baseline"],
                adaptation_rate=0.05,
                last_updated=time.time()
            )

    def get_threshold(self, name: str) -> float:
        if name in self._thresholds:
            return self._thresholds[name].current_value
        return 0.5

    def record_performance(self, performance_data: PerformanceData):
        self._performance_history.append(performance_data)
        self._check_and_adapt_thresholds()

    def _check_and_adapt_thresholds(self):
        current_time = time.time()
        for threshold_name, threshold in self._thresholds.items():
            if current_time - threshold.last_updated < self._min_adaptation_interval:
                continue
            performance_trend = self._calculate_performance_trend(threshold_name)
            stability = self._calculate_stability(threshold_name)
            if abs(performance_trend) > 0.1 and stability > 0.7:
                adjustment = self._calculate_optimal_adjustment(threshold_name, performance_trend, stability)
                if abs(adjustment) > 0.01:
                    self._apply_threshold_adjustment(threshold_name, adjustment, performance_trend)

    def _calculate_performance_trend(self, threshold_name: str) -> float:
        if len(self._performance_history) < 10:
            return 0.0
        recent_data = list(self._performance_history)[-20:]
        n = len(recent_data)
        if n < 2:
            return 0.0
        values = [data.decision_quality for data in recent_data]
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * val for i, val in enumerate(values))
        sum_x2 = sum(i * i for i in range(n))
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return max(-1.0, min(1.0, slope * 10))

    def _calculate_stability(self, threshold_name: str) -> float:
        if len(self._performance_history) < self._stability_window:
            return 0.5
        recent_data = list(self._performance_history)[-self._stability_window:]
        qualities = [data.decision_quality for data in recent_data]
        if len(qualities) < 2:
            return 0.5
        mean_quality = sum(qualities) / len(qualities)
        variance = sum((q - mean_quality) ** 2 for q in qualities) / len(qualities)
        std_dev = math.sqrt(variance)
        return max(0.0, 1.0 - min(1.0, std_dev * 2))

    def _calculate_optimal_adjustment(self, threshold_name: str, performance_trend: float, stability: float) -> float:
        threshold = self._thresholds[threshold_name]
        base_adjustment = performance_trend * threshold.adaptation_rate * self._adaptation_sensitivity
        stability_multiplier = stability
        distance_from_baseline = abs(threshold.current_value - threshold.baseline_value)
        baseline_multiplier = max(0.1, 1.0 - distance_from_baseline)
        adjustment = base_adjustment * stability_multiplier * baseline_multiplier
        return max(-self._max_threshold_change, min(self._max_threshold_change, adjustment))

    def _apply_threshold_adjustment(self, threshold_name: str, adjustment: float, performance_trend: float):
        threshold = self._thresholds[threshold_name]
        old_value = threshold.current_value
        new_value = max(0.0, min(1.0, old_value + adjustment))
        threshold.current_value = new_value
        threshold.last_updated = time.time()
        threshold.adjustment_count += 1
        reason = "性能提升，增加阈值" if performance_trend > 0 else "性能下降，降低阈值"
        record = ThresholdAdjustment(
            threshold_name=threshold_name, old_value=old_value,
            new_value=new_value, reason=reason,
            performance_impact=abs(adjustment) * 0.1
        )
        self._adjustments_history.append(record)
        if len(self._adjustments_history) > self._max_adjustments_history:
            self._adjustments_history = self._adjustments_history[-self._max_adjustments_history:]
        logger.debug(f"阈值调整: {threshold_name} {old_value:.3f} → {new_value:.3f} ({reason})")

    def force_threshold_adjustment(self, threshold_name: str, new_value: float, reason: str = "manual"):
        if threshold_name not in self._thresholds:
            return
        threshold = self._thresholds[threshold_name]
        old_value = threshold.current_value
        threshold.current_value = max(0.0, min(1.0, new_value))
        threshold.last_updated = time.time()
        threshold.adjustment_count += 1
        self._adjustments_history.append(ThresholdAdjustment(
            threshold_name=threshold_name, old_value=old_value,
            new_value=threshold.current_value, reason=f"手动: {reason}",
            performance_impact=0.0
        ))

    def reset_threshold(self, threshold_name: str):
        if threshold_name not in self._thresholds:
            return
        threshold = self._thresholds[threshold_name]
        threshold.current_value = threshold.baseline_value
        threshold.last_updated = time.time()

    def get_threshold_info(self, threshold_name: str) -> Optional[Dict[str, Any]]:
        if threshold_name not in self._thresholds:
            return None
        threshold = self._thresholds[threshold_name]
        return {
            "name": threshold.threshold_name,
            "current_value": threshold.current_value,
            "baseline_value": threshold.baseline_value,
            "adaptation_rate": threshold.adaptation_rate,
            "last_updated": threshold.last_updated,
            "adjustment_count": threshold.adjustment_count,
            "performance_trend": self._calculate_performance_trend(threshold_name),
            "stability_score": self._calculate_stability(threshold_name),
            "deviation_from_baseline": threshold.current_value - threshold.baseline_value
        }

    def get_all_thresholds_info(self) -> Dict[str, Dict[str, Any]]:
        return {name: self.get_threshold_info(name) for name in self._thresholds if self.get_threshold_info(name)}

    def get_system_health(self) -> Dict[str, Any]:
        if not self._performance_history:
            return {"status": "insufficient_data"}
        recent_data = list(self._performance_history)[-50:]
        avg_quality = sum(d.decision_quality for d in recent_data) / len(recent_data)
        avg_time = sum(d.response_time for d in recent_data) / len(recent_data)
        avg_satisfaction = sum(d.user_satisfaction for d in recent_data) / len(recent_data)
        error_rate = sum(d.error_rate for d in recent_data) / len(recent_data)
        health_score = avg_quality * 0.4 + (1 - avg_time / 10) * 0.3 + avg_satisfaction * 0.2 + (1 - error_rate) * 0.1
        if health_score > 0.8:
            status = "excellent"
        elif health_score > 0.6:
            status = "good"
        elif health_score > 0.4:
            status = "fair"
        else:
            status = "poor"
        return {
            "status": status, "health_score": health_score,
            "metrics": {"decision_quality": avg_quality, "response_time": avg_time,
                        "user_satisfaction": avg_satisfaction, "error_rate": error_rate},
            "adaptation_stats": {"total_adjustments": len(self._adjustments_history),
                                 "thresholds_count": len(self._thresholds)}
        }

    def export_configuration(self) -> Dict[str, Any]:
        config = {"thresholds": {}, "system_settings": {
            "adaptation_sensitivity": self._adaptation_sensitivity,
            "min_adaptation_interval": self._min_adaptation_interval,
            "max_threshold_change": self._max_threshold_change,
            "stability_window": self._stability_window
        }, "exported_at": time.time()}
        for name, threshold in self._thresholds.items():
            config["thresholds"][name] = {
                "current_value": threshold.current_value,
                "baseline_value": threshold.baseline_value,
                "adaptation_rate": threshold.adaptation_rate
            }
        return config

    def import_configuration(self, config: Dict[str, Any]):
        if "thresholds" in config:
            for name, tc in config["thresholds"].items():
                if name in self._thresholds:
                    t = self._thresholds[name]
                    t.current_value = tc.get("current_value", t.baseline_value)
                    t.baseline_value = tc.get("baseline_value", t.baseline_value)
                    t.adaptation_rate = tc.get("adaptation_rate", 0.05)
                    t.last_updated = time.time()
        if "system_settings" in config:
            s = config["system_settings"]
            self._adaptation_sensitivity = s.get("adaptation_sensitivity", 0.1)
            self._min_adaptation_interval = s.get("min_adaptation_interval", 60)
            self._max_threshold_change = s.get("max_threshold_change", 0.2)
            self._stability_window = s.get("stability_window", 10)


_adaptive_threshold_system: Optional[AdaptiveThresholdSystem] = None


def get_adaptive_threshold_system() -> AdaptiveThresholdSystem:
    global _adaptive_threshold_system
    if _adaptive_threshold_system is None:
        _adaptive_threshold_system = AdaptiveThresholdSystem()
    return _adaptive_threshold_system

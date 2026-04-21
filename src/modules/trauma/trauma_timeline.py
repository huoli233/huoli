import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.common.config.config_engine import ConfigEngine
from src.common.logger import get_logger

logger = get_logger("创伤时间线")


class TraumaEventType(Enum):
    """创伤事件类型"""

    TRIGGER = "触发"
    EPISODE = "发作"
    RECOVERY = "恢复"
    HEALING = "治愈"
    RELAPSE = "复发"
    MILESTONE = "里程碑"
    INSIGHT = "洞察"


@dataclass
class TraumaEvent:
    """创伤事件"""

    event_id: str = ""
    event_type: TraumaEventType = TraumaEventType.TRIGGER
    timestamp: float = field(default_factory=time.time)
    severity: float = 0.0
    description: str = ""
    triggers: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    recovery_time: float = 0.0
    impact_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "description": self.description,
            "triggers": self.triggers,
            "symptoms": self.symptoms,
            "context": self.context,
            "duration_seconds": self.duration_seconds,
            "recovery_time": self.recovery_time,
            "impact_score": self.impact_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraumaEvent":
        event_type = TraumaEventType.TRIGGER
        for et in TraumaEventType:
            if et.value == data.get("event_type", "触发"):
                event_type = et
                break
        return cls(
            event_id=data.get("event_id", ""),
            event_type=event_type,
            timestamp=data.get("timestamp", time.time()),
            severity=data.get("severity", 0.0),
            description=data.get("description", ""),
            triggers=data.get("triggers", []),
            symptoms=data.get("symptoms", []),
            context=data.get("context", {}),
            duration_seconds=data.get("duration_seconds", 0.0),
            recovery_time=data.get("recovery_time", 0.0),
            impact_score=data.get("impact_score", 0.0),
        )


@dataclass
class TraumaPattern:
    """创伤模式"""

    pattern_id: str = ""
    pattern_type: str = ""
    frequency: int = 0
    avg_severity: float = 0.0
    avg_duration: float = 0.0
    common_triggers: List[str] = field(default_factory=list)
    time_distribution: Dict[int, int] = field(default_factory=dict)
    first_occurrence: float = 0.0
    last_occurrence: float = 0.0


class TraumaTimeline:
    """创伤时间线

    记录创伤事件的时间序列，支持回溯分析和模式识别
    """

    def __init__(self, config_engine: Optional[ConfigEngine] = None):
        self._config = config_engine or ConfigEngine.get_instance()
        self._events: List[TraumaEvent] = []
        self._patterns: Dict[str, TraumaPattern] = {}
        self._max_events = 500
        self._load_config()

    def _load_config(self):
        """加载配置"""
        trauma_cfg = self._config.get("trauma", {})
        self._max_events = trauma_cfg.get("max_timeline_events", 500)
        self._pattern_window_hours = trauma_cfg.get(
            "pattern_window_hours", 168
        )
        self._min_pattern_occurrences = trauma_cfg.get(
            "min_pattern_occurrences", 3
        )

    def record_event(
        self,
        event_type: TraumaEventType,
        severity: float = 0.0,
        description: str = "",
        triggers: Optional[List[str]] = None,
        symptoms: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TraumaEvent:
        """记录创伤事件"""
        event_id = f"{int(time.time())}_{len(self._events)}"

        event = TraumaEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            description=description,
            triggers=triggers or [],
            symptoms=symptoms or [],
            context=context or {},
        )

        self._events.append(event)

        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

        self._update_patterns(event)
        logger.debug(f"记录创伤事件: {event_type.value}, 严重度={severity}")

        return event

    def _update_patterns(self, event: TraumaEvent):
        """更新模式识别"""
        for trigger in event.triggers:
            if trigger not in self._patterns:
                self._patterns[trigger] = TraumaPattern(
                    pattern_id=trigger,
                    pattern_type="trigger_based",
                    common_triggers=[trigger],
                )

            pattern = self._patterns[trigger]
            pattern.frequency += 1
            pattern.avg_severity = (
                pattern.avg_severity * (pattern.frequency - 1) + event.severity
            ) / pattern.frequency
            pattern.last_occurrence = event.timestamp

            if pattern.first_occurrence == 0:
                pattern.first_occurrence = event.timestamp

            hour = time.localtime(event.timestamp).tm_hour
            pattern.time_distribution[hour] = (
                pattern.time_distribution.get(hour, 0) + 1
            )

    def get_events(
        self,
        event_type: Optional[TraumaEventType] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        min_severity: Optional[float] = None,
        limit: int = 50,
    ) -> List[TraumaEvent]:
        """获取事件列表"""
        filtered = self._events

        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        if start_time is not None:
            filtered = [e for e in filtered if e.timestamp >= start_time]
        if end_time is not None:
            filtered = [e for e in filtered if e.timestamp <= end_time]
        if min_severity is not None:
            filtered = [e for e in filtered if e.severity >= min_severity]

        return filtered[-limit:]

    def get_recent_events(
        self, hours: float = 24.0, limit: int = 20
    ) -> List[TraumaEvent]:
        """获取最近事件"""
        cutoff = time.time() - (hours * 3600)
        return [e for e in self._events if e.timestamp >= cutoff][-limit:]

    def get_event_by_id(self, event_id: str) -> Optional[TraumaEvent]:
        """根据ID获取事件"""
        for event in self._events:
            if event.event_id == event_id:
                return event
        return None

    def update_event_recovery(
        self, event_id: str, recovery_time: float
    ) -> bool:
        """更新事件恢复时间"""
        event = self.get_event_by_id(event_id)
        if event:
            event.recovery_time = recovery_time
            return True
        return False

    def get_patterns(self, min_frequency: int = 2) -> List[TraumaPattern]:
        """获取识别的模式"""
        return [
            p for p in self._patterns.values() if p.frequency >= min_frequency
        ]

    def get_frequent_triggers(self, limit: int = 10) -> List[Tuple[str, int]]:
        """获取高频触发"""
        trigger_counts: Dict[str, int] = {}
        for event in self._events:
            for trigger in event.triggers:
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

        sorted_triggers = sorted(
            trigger_counts.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_triggers[:limit]

    def get_severity_trend(self, days: int = 7) -> List[Tuple[float, float]]:
        """获取严重度趋势"""
        cutoff = time.time() - (days * 86400)
        daily_severity: Dict[str, List[float]] = {}

        for event in self._events:
            if event.timestamp >= cutoff:
                day_key = time.strftime(
                    "%Y-%m-%d", time.localtime(event.timestamp)
                )
                if day_key not in daily_severity:
                    daily_severity[day_key] = []
                daily_severity[day_key].append(event.severity)

        trend = []
        for day_key in sorted(daily_severity.keys()):
            severities = daily_severity[day_key]
            avg = sum(severities) / len(severities) if severities else 0.0
            day_ts = time.mktime(time.strptime(day_key, "%Y-%m-%d"))
            trend.append((day_ts, avg))

        return trend

    def get_time_distribution(self) -> Dict[int, int]:
        """获取时间分布"""
        distribution: Dict[int, int] = {}
        for event in self._events:
            hour = time.localtime(event.timestamp).tm_hour
            distribution[hour] = distribution.get(hour, 0) + 1
        return distribution

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._events:
            return {
                "total_events": 0,
                "event_types": {},
                "avg_severity": 0.0,
                "max_severity": 0.0,
            }

        event_types: Dict[str, int] = {}
        total_severity = 0.0
        max_severity = 0.0

        for event in self._events:
            type_key = event.event_type.value
            event_types[type_key] = event_types.get(type_key, 0) + 1
            total_severity += event.severity
            max_severity = max(max_severity, event.severity)

        return {
            "total_events": len(self._events),
            "event_types": event_types,
            "avg_severity": total_severity / len(self._events),
            "max_severity": max_severity,
            "patterns_identified": len(self._patterns),
            "time_range_hours": (
                (time.time() - self._events[0].timestamp) / 3600
                if self._events
                else 0
            ),
        }

    def get_summary(self) -> str:
        """获取摘要"""
        stats = self.get_statistics()
        parts = [
            f"总事件: {stats['total_events']}",
            f"平均严重度: {stats['avg_severity']:.2f}",
        ]

        if stats["event_types"]:
            type_strs = [f"{k}:{v}" for k, v in stats["event_types"].items()]
            parts.append(f"类型分布: {', '.join(type_strs[:3])}")

        return " | ".join(parts)

    def clear_old_events(self, days: int = 30) -> int:
        """清除旧事件"""
        cutoff = time.time() - (days * 86400)
        original_count = len(self._events)
        self._events = [e for e in self._events if e.timestamp >= cutoff]
        return original_count - len(self._events)

    def reset(self) -> None:
        """重置时间线"""
        self._events.clear()
        self._patterns.clear()

    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "events": [e.to_dict() for e in self._events],
            "patterns": {
                k: {
                    "pattern_id": p.pattern_id,
                    "frequency": p.frequency,
                    "avg_severity": p.avg_severity,
                    "common_triggers": p.common_triggers,
                }
                for k, p in self._patterns.items()
            },
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """反序列化"""
        self._events = [
            TraumaEvent.from_dict(e) for e in data.get("events", [])
        ]
        for k, p_data in data.get("patterns", {}).items():
            self._patterns[k] = TraumaPattern(
                pattern_id=p_data.get("pattern_id", k),
                frequency=p_data.get("frequency", 0),
                avg_severity=p_data.get("avg_severity", 0.0),
                common_triggers=p_data.get("common_triggers", []),
            )


_trauma_timeline: Optional[TraumaTimeline] = None


def get_trauma_timeline(
    config_engine: Optional[ConfigEngine] = None,
) -> TraumaTimeline:
    """获取创伤时间线单例"""
    global _trauma_timeline
    if _trauma_timeline is None:
        _trauma_timeline = TraumaTimeline(config_engine)
    return _trauma_timeline

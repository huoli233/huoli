import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.modules.trauma.runtime_config import trauma_module_view

logger = get_logger("创伤触发")


@dataclass
class TriggerPattern:
    """触发模式"""

    pattern_id: str = ""
    keywords: List[str] = field(default_factory=list)
    regex_pattern: str = ""
    severity: float = 1.0
    layer_type: str = "表层"
    response_type: str = "avoidance"
    description: str = ""
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "keywords": self.keywords,
            "regex_pattern": self.regex_pattern,
            "severity": self.severity,
            "layer_type": self.layer_type,
            "response_type": self.response_type,
            "description": self.description,
            "created_at": self.created_at,
            "hit_count": self.hit_count,
        }


@dataclass
class TriggerResult:
    """触发结果"""

    triggered: bool = False
    pattern_id: str = ""
    matched_content: str = ""
    severity: float = 0.0
    layer_type: str = "表层"
    response_type: str = "avoidance"
    confidence: float = 0.0


class TraumaTriggerEvaluator:
    """创伤触发评估器

    评估文本是否触发创伤响应
    """

    def __init__(self):
        self._patterns: Dict[str, TriggerPattern] = {}
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._hit_history: List[Tuple[str, float, str]] = []
        self._load_config()

    def _load_config(self):
        """加载配置"""
        trauma_cfg = trauma_module_view("trauma_triggers")

        self._default_severity = trauma_cfg.get(
            "default_trigger_severity", 1.0
        )
        self._max_history = trauma_cfg.get("max_trigger_history", 100)
        self._cooldown_seconds = trauma_cfg.get(
            "trigger_cooldown_seconds", 60.0
        )
        self._sensitivity = trauma_cfg.get("trigger_sensitivity", 0.7)
        self._keyword_confidence = trauma_cfg.get("keyword_confidence", 0.8)
        self._regex_confidence = trauma_cfg.get("regex_confidence", 0.9)
        self._history_content_chars = int(trauma_cfg.get("history_content_chars", 50))
        self._recent_trigger_limit = int(trauma_cfg.get("recent_trigger_limit", 10))
        self._frequency_window_hours = int(trauma_cfg.get("frequency_window_hours", 24))
        self._default_patterns = trauma_cfg.get("default_patterns", [])

        self._initialize_default_patterns()

    def _initialize_default_patterns(self):
        """初始化默认触发模式"""
        default_patterns = self._default_patterns or [
            {
                "pattern_id": "rejection",
                "keywords": ["不要你", "讨厌你", "滚", "离开", "不想理你"],
                "severity": 3.0,
                "layer_type": "中层",
                "response_type": "withdrawal",
                "description": "拒绝类触发",
            },
            {
                "pattern_id": "abandonment",
                "keywords": ["抛弃", "丢下", "不管你", "没人要"],
                "severity": 4.0,
                "layer_type": "深层",
                "response_type": "panic",
                "description": "遗弃类触发",
            },
            {
                "pattern_id": "humiliation",
                "keywords": ["蠢", "笨", "废物", "没用", "丢人"],
                "severity": 3.5,
                "layer_type": "中层",
                "response_type": "shame",
                "description": "羞辱类触发",
            },
            {
                "pattern_id": "threat",
                "keywords": ["打死", "伤害", "报复", "让你后悔"],
                "severity": 5.0,
                "layer_type": "核心",
                "response_type": "fear",
                "description": "威胁类触发",
            },
            {
                "pattern_id": "invalidation",
                "keywords": ["你想多了", "太敏感", "无理取闹", "矫情"],
                "severity": 2.5,
                "layer_type": "浅层",
                "response_type": "numbness",
                "description": "否定类触发",
            },
        ]

        for item in default_patterns:
            if not isinstance(item, dict):
                continue
            pattern = TriggerPattern(
                pattern_id=str(item.get("pattern_id", "")),
                keywords=list(item.get("keywords", [])),
                regex_pattern=str(item.get("regex_pattern", "")),
                severity=float(item.get("severity", self._default_severity)),
                layer_type=str(item.get("layer_type", "表层")),
                response_type=str(item.get("response_type", "avoidance")),
                description=str(item.get("description", "")),
            )
            self.add_pattern(pattern)

    def add_pattern(self, pattern: TriggerPattern) -> None:
        """添加触发模式"""
        if not pattern.pattern_id:
            pattern.pattern_id = f"custom_{int(time.time())}"

        self._patterns[pattern.pattern_id] = pattern

        if pattern.regex_pattern:
            try:
                self._compiled_patterns[pattern.pattern_id] = re.compile(
                    pattern.regex_pattern, re.IGNORECASE
                )
            except re.error:
                logger.warning(f"无效正则模式: {pattern.regex_pattern}")

    def remove_pattern(self, pattern_id: str) -> bool:
        """移除触发模式"""
        if pattern_id in self._patterns:
            del self._patterns[pattern_id]
            self._compiled_patterns.pop(pattern_id, None)
            return True
        return False

    def evaluate(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> TriggerResult:
        """评估文本是否触发创伤"""
        if not text:
            return TriggerResult()

        text_lower = text.lower()
        best_match: Optional[TriggerPattern] = None
        best_confidence = 0.0
        matched_content = ""

        for pattern_id, pattern in self._patterns.items():
            confidence = 0.0
            matched = ""

            for keyword in pattern.keywords:
                if keyword.lower() in text_lower:
                    confidence = max(confidence, self._keyword_confidence)
                    matched = keyword

            if pattern_id in self._compiled_patterns:
                match = self._compiled_patterns[pattern_id].search(text)
                if match:
                    confidence = max(confidence, self._regex_confidence)
                    matched = match.group()

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = pattern
                matched_content = matched

        if best_match and best_confidence >= self._sensitivity:
            best_match.hit_count += 1
            self._record_hit(
                best_match.pattern_id, best_match.severity, matched_content
            )

            return TriggerResult(
                triggered=True,
                pattern_id=best_match.pattern_id,
                matched_content=matched_content,
                severity=best_match.severity,
                layer_type=best_match.layer_type,
                response_type=best_match.response_type,
                confidence=best_confidence,
            )

        return TriggerResult()

    def _record_hit(self, pattern_id: str, severity: float, content: str):
        """记录触发历史"""
        self._hit_history.append((pattern_id, time.time(), content[: self._history_content_chars]))
        if len(self._hit_history) > self._max_history:
            self._hit_history = self._hit_history[-self._max_history:]

    def get_recent_triggers(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取最近触发记录"""
        limit = self._recent_trigger_limit if limit is None else limit
        recent = self._hit_history[-limit:]
        return [
            {
                "pattern_id": p_id,
                "timestamp": ts,
                "content": content,
                "pattern": self._patterns.get(
                    p_id, TriggerPattern()
                ).to_dict(),
            }
            for p_id, ts, content in reversed(recent)
        ]

    def get_trigger_frequency(self, hours: Optional[int] = None) -> Dict[str, int]:
        """获取触发频率"""
        hours = self._frequency_window_hours if hours is None else hours
        cutoff = time.time() - hours * 3600
        frequency: Dict[str, int] = {}
        for pattern_id, ts, _ in self._hit_history:
            if ts >= cutoff:
                frequency[pattern_id] = frequency.get(pattern_id, 0) + 1
        return frequency

    def get_pattern_stats(self) -> Dict[str, Any]:
        """获取模式统计"""
        return {
            "total_patterns": len(self._patterns),
            "total_hits": len(self._hit_history),
            "pattern_hits": {
                p_id: p.hit_count for p_id, p in self._patterns.items()
            },
        }

    def clear_history(self) -> None:
        """清除历史"""
        self._hit_history.clear()

    def reset_patterns(self) -> None:
        """重置模式"""
        self._patterns.clear()
        self._compiled_patterns.clear()
        self._initialize_default_patterns()


_trauma_trigger_evaluator: Optional[TraumaTriggerEvaluator] = None


def get_trauma_trigger_evaluator() -> TraumaTriggerEvaluator:
    """获取创伤触发评估器单例"""
    global _trauma_trigger_evaluator
    if _trauma_trigger_evaluator is None:
        _trauma_trigger_evaluator = TraumaTriggerEvaluator()
    return _trauma_trigger_evaluator

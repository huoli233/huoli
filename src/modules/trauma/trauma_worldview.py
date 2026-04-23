"""
创伤世界观冲击处理器

评估对话对主体世界观的损伤与修复，"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.modules.trauma.runtime_config import trauma_module_view

logger = get_logger("创伤世界观")


@dataclass
class WorldviewImpactReport:
    """世界观冲击评估报告"""

    trust_shift: float = 0.0
    reality_shift: float = 0.0
    meaning_shift: float = 1.0
    cognition_shift: float = 0.5
    newly_damaged: List[str] = field(default_factory=list)
    healing_detected: bool = False
    note: str = ""
    assessed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trust_shift": self.trust_shift,
            "reality_shift": self.reality_shift,
            "meaning_shift": self.meaning_shift,
            "cognition_shift": self.cognition_shift,
            "newly_damaged": self.newly_damaged,
            "healing_detected": self.healing_detected,
            "note": self.note,
            "assessed_at": self.assessed_at,
        }


class WorldviewImpactProcessor:
    """世界观冲击处理器

    评估对话对主体世界观的损伤与修复
    """

    def __init__(self):
        self._impact_history: List[WorldviewImpactReport] = []
        self._last_assessment_at: float = 0.0
        self._load_config()

    def _load_config(self):
        """加载配置"""
        worldview_cfg = trauma_module_view("trauma_worldview")
        self._assessment_cooldown = worldview_cfg.get(
            "assessment_cooldown_seconds", 30.0
        )
        self._max_beliefs_tracked = worldview_cfg.get(
            "max_beliefs_tracked", 10
        )
        self._max_delta_per_assessment = worldview_cfg.get(
            "max_delta_per_assessment", 2.0
        )
        self._healing_threshold = worldview_cfg.get("healing_threshold", 0.3)
        self._impact_history_limit = int(worldview_cfg.get("impact_history_limit", 100))
        self._trust_negative_scale = worldview_cfg.get("trust_negative_scale", 0.5)
        self._trust_positive_scale = worldview_cfg.get("trust_positive_scale", 0.3)
        self._reality_severity_scale = worldview_cfg.get("reality_severity_scale", 0.3)
        self._reality_keyword_bonus = worldview_cfg.get("reality_keyword_bonus", 0.2)
        self._meaning_severity_scale = worldview_cfg.get("meaning_severity_scale", 0.4)
        self._meaning_keyword_bonus = worldview_cfg.get("meaning_keyword_bonus", 0.3)
        self._cognition_severity_scale = worldview_cfg.get("cognition_severity_scale", 0.2)
        self._cognition_keyword_bonus = worldview_cfg.get("cognition_keyword_bonus", 0.2)
        self._damaged_belief_min_severity = worldview_cfg.get("damaged_belief_min_severity", 0.5)
        self._damaged_belief_limit = int(worldview_cfg.get("damaged_belief_limit", 3))
        self._healing_min_keywords = int(worldview_cfg.get("healing_min_keywords", 2))
        self._worldview_cap = worldview_cfg.get("worldview_cap", 10.0)
        self._severe_trust_threshold = worldview_cfg.get("severe_trust_threshold", 3.0)
        self._severe_reality_threshold = worldview_cfg.get("severe_reality_threshold", 7.0)
        self._severe_meaning_threshold = worldview_cfg.get("severe_meaning_threshold", 7.0)
        self._medium_trust_threshold = worldview_cfg.get("medium_trust_threshold", 5.0)
        self._medium_reality_threshold = worldview_cfg.get("medium_reality_threshold", 4.0)
        self._medium_meaning_threshold = worldview_cfg.get("medium_meaning_threshold", 4.0)
        self._reality_keywords = worldview_cfg.get(
            "reality_keywords", ["假的", "不真实", "欺骗", "谎言", "虚伪"]
        )
        self._meaning_keywords = worldview_cfg.get(
            "meaning_keywords", ["没意义", "无所谓", "没用", "徒劳"]
        )
        self._confusion_keywords = worldview_cfg.get(
            "confusion_keywords", ["不明白", "搞不懂", "混乱", "矛盾"]
        )
        self._belief_patterns = worldview_cfg.get(
            "belief_patterns",
            [
                {"belief": "信任他人", "triggers": ["不可信", "不能信", "骗我"]},
                {"belief": "世界是安全的", "triggers": ["危险", "不安全", "威胁"]},
                {"belief": "人是善良的", "triggers": ["恶意", "坏人", "伤害"]},
                {"belief": "未来有希望", "triggers": ["绝望", "没希望", "完蛋"]},
            ],
        )
        self._healing_keywords = worldview_cfg.get(
            "healing_keywords", ["理解", "支持", "关心", "爱", "温暖", "感谢"]
        )

    def assess_impact(
        self,
        dialogue_text: str,
        severity: float = 0.0,
        sentiment: str = "neutral",
        current_worldview: Optional[Dict] = None,
    ) -> WorldviewImpactReport:
        """评估世界观冲击"""
        now = time.time()

        if (now - self._last_assessment_at) < self._assessment_cooldown:
            return WorldviewImpactReport()

        self._last_assessment_at = now

        trust_shift = self._calculate_trust_shift(
            dialogue_text, severity, sentiment
        )
        reality_shift = self._calculate_reality_shift(dialogue_text, severity)
        meaning_shift = self._calculate_meaning_shift(
            dialogue_text, severity, sentiment
        )
        cognition_shift = self._calculate_cognition_shift(
            dialogue_text, severity
        )

        newly_damaged = self._identify_damaged_beliefs(dialogue_text, severity)
        healing_detected = self._detect_healing(
            dialogue_text, sentiment, severity
        )

        report = WorldviewImpactReport(
            trust_shift=trust_shift,
            reality_shift=reality_shift,
            meaning_shift=meaning_shift,
            cognition_shift=cognition_shift,
            newly_damaged=newly_damaged,
            healing_detected=healing_detected,
            note=self._generate_assessment_note(
                trust_shift, reality_shift, meaning_shift
            ),
        )

        self._impact_history.append(report)
        if len(self._impact_history) > self._impact_history_limit:
            self._impact_history = self._impact_history[-self._impact_history_limit:]

        return report

    def _calculate_trust_shift(
        self, text: str, severity: float, sentiment: str
    ) -> float:
        """计算信任变化"""
        base_shift = severity * self._trust_negative_scale
        if sentiment == "negative":
            return -min(self._max_delta_per_assessment, base_shift)
        elif sentiment == "positive":
            return min(self._max_delta_per_assessment, base_shift * self._trust_positive_scale)
        return 0.0

    def _calculate_reality_shift(self, text: str, severity: float) -> float:
        """计算现实扭曲变化"""
        keyword_bonus = sum(self._reality_keyword_bonus for kw in self._reality_keywords if kw in text)
        return min(
            self._max_delta_per_assessment,
            severity * self._reality_severity_scale + keyword_bonus,
        )

    def _calculate_meaning_shift(
        self, text: str, severity: float, sentiment: str
    ) -> float:
        """计算意义崩塌变化"""
        keyword_bonus = sum(self._meaning_keyword_bonus for kw in self._meaning_keywords if kw in text)
        base = severity * self._meaning_severity_scale
        if sentiment == "negative":
            return min(self._max_delta_per_assessment, base + keyword_bonus)
        return min(self._max_delta_per_assessment, keyword_bonus)

    def _calculate_cognition_shift(self, text: str, severity: float) -> float:
        """计算认知碎片化变化"""
        keyword_bonus = sum(self._cognition_keyword_bonus for kw in self._confusion_keywords if kw in text)
        return min(
            self._max_delta_per_assessment,
            severity * self._cognition_severity_scale + keyword_bonus,
        )

    def _identify_damaged_beliefs(
        self, text: str, severity: float
    ) -> List[str]:
        """识别受损信念"""
        if severity < self._damaged_belief_min_severity:
            return []

        damaged = []
        for pattern in self._belief_patterns:
            if not isinstance(pattern, dict):
                continue
            belief = str(pattern.get("belief", ""))
            triggers = pattern.get("triggers", [])
            for trigger in triggers:
                if trigger in text:
                    damaged.append(belief)
                    break

        return damaged[: self._damaged_belief_limit]

    def _detect_healing(
        self, text: str, sentiment: str, severity: float
    ) -> bool:
        """检测治愈"""
        if sentiment != "positive":
            return False

        healing_count = sum(1 for kw in self._healing_keywords if kw in text)

        return healing_count >= self._healing_min_keywords and severity < self._healing_threshold

    def _generate_assessment_note(
        self,
        trust_shift: float,
        reality_shift: float,
        meaning_shift: float,
    ) -> str:
        """生成评估备注"""
        parts = []

        if abs(trust_shift) > 0.0:
            direction = "下降" if trust_shift < 0.0 else "上升"
            parts.append(f"信任{direction}{abs(trust_shift):.1f}")

        if abs(reality_shift) > 1.0:
            parts.append(f"现实扭曲+{reality_shift:.1f}")

        if abs(meaning_shift) > 1.0:
            parts.append(f"意义感{'下降' if meaning_shift > 1.0 else '稳定'}")

        return " | ".join(parts) if parts else "无明显变化"

    def apply_to_worldview(
        self, worldview: Dict, report: WorldviewImpactReport
    ) -> Dict:
        """应用冲击到世界观"""
        if not worldview:
            worldview = {
                "trust_level": 5.0,
                "reality_distortion": 0.0,
                "meaning_collapse": 0.0,
                "cognitive_fragmentation": 0.0,
                "damaged_beliefs": [],
            }

        def _safe_shift(current, shift, lo=0.0, hi=self._worldview_cap, default=0.0):
            try:
                return max(lo, min(hi, float(current) + float(shift)))
            except (TypeError, ValueError):
                return default

        worldview["trust_level"] = _safe_shift(
            worldview.get("trust_level", 5.0), report.trust_shift, default=5.0
        )
        worldview["reality_distortion"] = _safe_shift(
            worldview.get("reality_distortion", 0.0), report.reality_shift
        )
        worldview["meaning_collapse"] = _safe_shift(
            worldview.get("meaning_collapse", 0.0), report.meaning_shift
        )
        worldview["cognitive_fragmentation"] = _safe_shift(
            worldview.get("cognitive_fragmentation", 0.0),
            report.cognition_shift,
        )

        damaged = worldview.get("damaged_beliefs", [])
        for belief in report.newly_damaged:
            if belief not in damaged:
                damaged.append(belief)
        worldview["damaged_beliefs"] = damaged[: self._max_beliefs_tracked]

        return worldview

    def get_impact_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取冲击历史"""
        return [r.to_dict() for r in self._impact_history[-limit:]]

    def get_recent_impacts(
        self, hours: float = 24.0
    ) -> List[WorldviewImpactReport]:
        """获取最近冲击"""
        cutoff = time.time() - (hours * 3600)
        return [r for r in self._impact_history if r.assessed_at >= cutoff]

    def get_worldview_status(self, worldview: Dict) -> str:
        """获取世界观状态描述"""
        trust = worldview.get("trust_level", 5.0)
        reality = worldview.get("reality_distortion", 0.0)
        meaning = worldview.get("meaning_collapse", 0.0)
        damaged_count = len(worldview.get("damaged_beliefs", []))

        if (
            trust < self._severe_trust_threshold
            or reality > self._severe_reality_threshold
            or meaning > self._severe_meaning_threshold
        ):
            return "严重受损"
        elif (
            trust < self._medium_trust_threshold
            or reality > self._medium_reality_threshold
            or meaning > self._medium_meaning_threshold
        ):
            return "中度受损"
        elif damaged_count > 0:
            return "轻微受损"
        return "基本健康"


_worldview_impact_processor: Optional[WorldviewImpactProcessor] = None


def get_worldview_impact_processor() -> WorldviewImpactProcessor:
    """获取世界观冲击处理器单例"""
    global _worldview_impact_processor
    if _worldview_impact_processor is None:
        _worldview_impact_processor = WorldviewImpactProcessor()
    return _worldview_impact_processor

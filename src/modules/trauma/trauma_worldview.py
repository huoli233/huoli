"""
创伤世界观冲击处理器

评估对话对主体世界观的损伤与修复，"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.common.config.config_engine import ConfigEngine
from src.common.logger import get_logger

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

    def __init__(self, config_engine: Optional[ConfigEngine] = None):
        self._config = config_engine or ConfigEngine.get_instance()
        self._impact_history: List[WorldviewImpactReport] = []
        self._last_assessment_at: float = 0.0
        self._load_config()

    def _load_config(self):
        """加载配置"""
        worldview_cfg = self._config.get("worldview", {})
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
        if len(self._impact_history) > 100:
            self._impact_history = self._impact_history[-100:]

        return report

    def _calculate_trust_shift(
        self, text: str, severity: float, sentiment: str
    ) -> float:
        """计算信任变化"""
        base_shift = severity * 0.5
        if sentiment == "negative":
            return -min(self._max_delta_per_assessment, base_shift)
        elif sentiment == "positive":
            return min(self._max_delta_per_assessment, base_shift * 0.3)
        return 0.0

    def _calculate_reality_shift(self, text: str, severity: float) -> float:
        """计算现实扭曲变化"""
        reality_keywords = ["假的", "不真实", "欺骗", "谎言", "虚伪"]
        keyword_bonus = sum(0.2 for kw in reality_keywords if kw in text)
        return min(
            self._max_delta_per_assessment, severity * 0.3 + keyword_bonus
        )

    def _calculate_meaning_shift(
        self, text: str, severity: float, sentiment: str
    ) -> float:
        """计算意义崩塌变化"""
        meaning_keywords = ["没意义", "无所谓", "没用", "徒劳"]
        keyword_bonus = sum(0.3 for kw in meaning_keywords if kw in text)
        base = severity * 0.4
        if sentiment == "negative":
            return min(self._max_delta_per_assessment, base + keyword_bonus)
        return min(self._max_delta_per_assessment, keyword_bonus)

    def _calculate_cognition_shift(self, text: str, severity: float) -> float:
        """计算认知碎片化变化"""
        confusion_keywords = ["不明白", "搞不懂", "混乱", "矛盾"]
        keyword_bonus = sum(0.2 for kw in confusion_keywords if kw in text)
        return min(
            self._max_delta_per_assessment, severity * 0.2 + keyword_bonus
        )

    def _identify_damaged_beliefs(
        self, text: str, severity: float
    ) -> List[str]:
        """识别受损信念"""
        if severity < 0.5:
            return []

        belief_patterns = [
            ("信任他人", ["不可信", "不能信", "骗我"]),
            ("世界是安全的", ["危险", "不安全", "威胁"]),
            ("人是善良的", ["恶意", "坏人", "伤害"]),
            ("未来有希望", ["绝望", "没希望", "完蛋"]),
        ]

        damaged = []
        for belief, triggers in belief_patterns:
            for trigger in triggers:
                if trigger in text:
                    damaged.append(belief)
                    break

        return damaged[:3]

    def _detect_healing(
        self, text: str, sentiment: str, severity: float
    ) -> bool:
        """检测治愈"""
        if sentiment != "positive":
            return False

        healing_keywords = ["理解", "支持", "关心", "爱", "温暖", "感谢"]
        healing_count = sum(1 for kw in healing_keywords if kw in text)

        return healing_count >= 2 and severity < self._healing_threshold

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

        def _safe_shift(current, shift, lo=0.0, hi=10.0, default=0.0):
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

        if trust < 3.0 or reality > 7.0 or meaning > 7.0:
            return "严重受损"
        elif trust < 5.0 or reality > 4.0 or meaning > 4.0:
            return "中度受损"
        elif damaged_count > 0:
            return "轻微受损"
        return "基本健康"


_worldview_impact_processor: Optional[WorldviewImpactProcessor] = None


def get_worldview_impact_processor(
    config_engine: Optional[ConfigEngine] = None,
) -> WorldviewImpactProcessor:
    """获取世界观冲击处理器单例"""
    global _worldview_impact_processor
    if _worldview_impact_processor is None:
        _worldview_impact_processor = WorldviewImpactProcessor(config_engine)
    return _worldview_impact_processor

import random
from typing import Dict, Any, Optional, Tuple

from src.common.logger import get_logger

logger = get_logger("adaptive_recall")


class AdaptiveRecallIntegrator:
    """自适应回忆集成器：根据心理状态在回复中引入"打字错误"等自然化效果

    模拟真实人在心理压力/疲劳/创伤状态下的语言表现：
    - 表面伪装低 + 内心混乱高 → 更容易出现口误/重复/遗漏
    - 心理疲劳高 → 反应变慢，可能出现重复字
    - 创伤分数高 → 语言不稳定
    """

    def __init__(self):
        self._base_recall_prob = 0.05
        self._base_typo_prob = 0.15

    def calculate_adjusted_probabilities(
        self,
        surface_mask: float = 10.0,
        inner_chaos: float = 0.0,
        mental_fatigue: float = 0.0,
        trauma_score: float = 0.0,
    ) -> Dict[str, float]:
        mask_factor = 1.0 - (surface_mask / 10.0)
        chaos_factor = inner_chaos / 10.0
        fatigue_factor = mental_fatigue / 100.0
        trauma_factor = trauma_score / 10.0
        multiplier = (
            1.0
            + mask_factor * 0.5
            + chaos_factor * 0.4
            + fatigue_factor * 0.3
            + trauma_factor * 0.5
        )
        multiplier = min(multiplier, 2.5)
        recall_prob = min(self._base_recall_prob * multiplier, 0.3)
        typo_prob = min(self._base_typo_prob * multiplier, 0.4)
        return {
            "recall_probability": recall_prob,
            "typo_probability": typo_prob,
            "multiplier": multiplier,
            "base_recall": self._base_recall_prob,
            "base_typo": self._base_typo_prob,
        }

    def should_trigger_recall(
        self,
        surface_mask: float,
        inner_chaos: float,
        mental_fatigue: float,
        trauma_score: float,
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        probs = self.calculate_adjusted_probabilities(
            surface_mask, inner_chaos, mental_fatigue, trauma_score
        )
        typo_prob = probs["typo_probability"]
        if random.random() > typo_prob:
            return False, None, probs
        error_types = ["typo", "swap", "duplicate", "pinyin", "wrong"]
        weights = [0.30, 0.20, 0.20, 0.15, 0.15]
        selected_error = random.choices(error_types, weights=weights)[0]
        return True, selected_error, probs

    def _generate_simple_error(self, text: str, error_type: str) -> str:
        if not text or len(text) < 2:
            return text
        chars = list(text)
        if error_type == "typo":
            idx = random.randint(0, len(chars) - 1)
            if "\u4e00" <= chars[idx] <= "\u9fff":
                similar_chars = [
                    "的",
                    "了",
                    "是",
                    "在",
                    "有",
                    "不",
                    "我",
                    "他",
                    "这",
                    "那",
                ]
                chars[idx] = random.choice(similar_chars)
        elif error_type == "swap" and len(chars) >= 3:
            idx = random.randint(0, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif error_type == "omit" and len(chars) >= 4:
            idx = random.randint(1, len(chars) - 2)
            chars.pop(idx)
        elif error_type == "duplicate" and len(chars) >= 2:
            idx = random.randint(0, len(chars) - 1)
            chars.insert(idx, chars[idx])
        return "".join(chars)


_adaptive_recall_instance: Optional[AdaptiveRecallIntegrator] = None


def get_adaptive_recall_integrator() -> AdaptiveRecallIntegrator:
    global _adaptive_recall_instance
    if _adaptive_recall_instance is None:
        _adaptive_recall_instance = AdaptiveRecallIntegrator()
    return _adaptive_recall_instance

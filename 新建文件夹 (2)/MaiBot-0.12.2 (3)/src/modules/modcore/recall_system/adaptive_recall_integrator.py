import random
from typing import Dict, Any, Optional, Tuple, Callable, Awaitable
from src.common.logger import get_logger

logger = get_logger("adaptive_recall")


class AdaptiveRecallIntegrator:
    def __init__(self):
        self._base_recall_prob = 0.05
        self._base_typo_prob = 0.15
        logger.info("自适应撤回集成器初始化完成")

    def calculate_adjusted_probabilities(
        self, surface_mask: float = 10.0, inner_chaos: float = 0.0,
        mental_fatigue: float = 0.0, trauma_score: float = 0.0,
    ) -> Dict[str, float]:
        mask_factor = 1.0 - (surface_mask / 10.0)
        chaos_factor = inner_chaos / 10.0
        fatigue_factor = mental_fatigue / 100.0
        trauma_factor = trauma_score / 10.0
        multiplier = 1.0 + mask_factor * 0.5 + chaos_factor * 0.4 + fatigue_factor * 0.3 + trauma_factor * 0.5
        multiplier = min(multiplier, 2.5)
        recall_prob = min(self._base_recall_prob * multiplier, 0.3)
        typo_prob = min(self._base_typo_prob * multiplier, 0.4)
        return {
            "recall_probability": recall_prob, "typo_probability": typo_prob,
            "multiplier": multiplier,
            "base_recall": self._base_recall_prob, "base_typo": self._base_typo_prob,
        }

    def should_trigger_recall(
        self, surface_mask: float, inner_chaos: float,
        mental_fatigue: float, trauma_score: float,
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        probs = self.calculate_adjusted_probabilities(
            surface_mask, inner_chaos, mental_fatigue, trauma_score
        )
        typo_prob = probs["typo_probability"]
        if random.random() > typo_prob:
            return False, None, probs
        error_types = ["typo", "swap", "omit", "duplicate", "pinyin", "wrong"]
        weights = [0.25, 0.15, 0.15, 0.15, 0.15, 0.15]
        selected_error = random.choices(error_types, weights=weights)[0]
        return True, selected_error, probs

    async def process_with_adaptive_recall(
        self, original_text: str,
        surface_mask: float, inner_chaos: float,
        mental_fatigue: float, trauma_score: float,
        group_id: str, stream_id: str,
        send_func: Callable[[str, str], Any],
        delete_func: Callable[[str], Any],
    ) -> Tuple[str, bool, Dict[str, Any]]:
        should_recall, error_type, probs = self.should_trigger_recall(
            surface_mask, inner_chaos, mental_fatigue, trauma_score
        )
        if not should_recall:
            logger.debug(f"未触发撤回 (概率{probs['typo_probability']:.2%})")
            return original_text, False, probs
        logger.info(f"触发撤回 (类型:{error_type}, 概率{probs['typo_probability']:.2%}, 倍率{probs['multiplier']:.2f}x)")
        try:
            from src.modules.modcore.recall_system.recall_controller import get_recall_controller
            controller = get_recall_controller()
            error_text, error_source = await controller.generate_error_text(original_text, error_type, stream_id)
            logger.debug(f"生成错误文本({error_source}): {error_text[:30]}...")
            msg_id = await send_func(group_id, error_text)
            await controller.schedule_recall(
                msg_id=msg_id, group_id=group_id, original_text=original_text,
                delete_func=delete_func, send_func=send_func,
                stream_id=stream_id, error_text=error_text, error_type=error_type,
            )
            return error_text, True, probs
        except Exception as e:
            logger.error(f"自适应撤回处理失败: {e}")
            return original_text, False, probs

    def _generate_simple_error(self, text: str, error_type: str) -> str:
        if not text or len(text) < 2:
            return text
        chars = list(text)
        if error_type == "typo":
            idx = random.randint(0, len(chars) - 1)
            if '\u4e00' <= chars[idx] <= '\u9fff':
                similar_chars = ["的", "了", "是", "在", "有", "不", "我", "他", "这", "那"]
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

import random
from typing import List, Optional
from src.common.logger import get_logger

logger = get_logger("affection_emoji")


class AffectionEmojiSelector:
    EMOJI_RULES = [
        (80, ["😊", "🥰", "💕", "✨", "🌸", "💖", "😘", "🤗"]),
        (50, ["😄", "👍", "😉", "🙂", "😌", "🌟"]),
        (20, ["😐", "🙂", "😶", "🤔"]),
        (0, ["😑", "😒", "😐"]),
        (-float('inf'), ["😒", "😑", "💢", "😤", "😠"]),
    ]

    def __init__(self):
        logger.info("表情符号选择器已初始化（基于好感度阈值）")

    def select_emoji(self, affection: float) -> str:
        try:
            emoji_list = self._get_emoji_list(affection)
            if emoji_list:
                emoji = random.choice(emoji_list)
                logger.debug(f"[表情符号] 好感度={affection}, 选择={emoji}")
                return emoji
            else:
                logger.warning(f"[表情符号] 好感度={affection}, 未找到匹配的表情符号列表")
                return ""
        except Exception as e:
            logger.error(f"[表情符号] 选择失败: {e}")
            return ""

    def _get_emoji_list(self, affection: float) -> List[str]:
        for threshold, emoji_list in self.EMOJI_RULES:
            if affection >= threshold:
                return emoji_list
        return self.EMOJI_RULES[-1][1]

    def should_add_emoji(self, affection: float, probability: float = 0.7) -> bool:
        if affection >= 80:
            probability = 0.9
        elif affection >= 50:
            probability = 0.7
        elif affection >= 20:
            probability = 0.5
        elif affection >= 0:
            probability = 0.3
        else:
            probability = 0.2
        return random.random() < probability


_emoji_selector_instance: Optional[AffectionEmojiSelector] = None


def get_affection_emoji_selector() -> AffectionEmojiSelector:
    global _emoji_selector_instance
    if _emoji_selector_instance is None:
        _emoji_selector_instance = AffectionEmojiSelector()
    return _emoji_selector_instance

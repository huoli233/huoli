import random
from typing import Optional, Dict, Tuple
from src.common.logger import get_logger

logger = get_logger("recall")


class ContentShuffler:
    GAME_SWAP = {
        '原神': ['崩坏星穹铁道', '明日方舟', '王者荣耀'],
        '蔚蓝档案': ['碧蓝航线', '少女前线', '公主连结'],
        '崩坏星穹铁道': ['原神', '崩坏3', '幻塔'],
        '明日方舟': ['原神', '少女前线', '碧蓝航线'],
        '英雄联盟': ['王者荣耀', 'DOTA2', '永劫无间'],
        '绝区零': ['崩坏星穹铁道', '原神', '鸣潮'],
    }
    TIME_SWAP = {
        '今年': ['去年', '前年'], '昨天': ['前天', '大前天'], '上个月': ['上上个月', '三个月前'],
        '2024': ['2023', '2022'], '2025': ['2024', '2023'], '最近': ['之前', '早些时候'],
    }
    TERM_SWAP = {
        '机器学习': ['深度学习', '神经网络'], '深度学习': ['机器学习', '强化学习'],
        'Python': ['Java', 'JavaScript'], 'Java': ['C#', 'Kotlin'],
        'API': ['SDK', 'RPC'], 'GPT': ['Claude', 'Gemini'],
    }
    NOUN_SWAP = {
        '技能': ['天赋', '被动'], '角色': ['英雄', '干员'], '武器': ['装备', '圣遗物'],
        '属性': ['数值', '能力'], '活动': ['副本', '关卡'], '任务': ['委托', '挑战'],
    }

    def __init__(self):
        self._swap_tables = [
            ('game', self.GAME_SWAP),
            ('time', self.TIME_SWAP),
            ('term', self.TERM_SWAP),
            ('noun', self.NOUN_SWAP),
        ]

    def shuffle(self, text: str, prob: float = 0.35) -> Tuple[str, Optional[str]]:
        if not text or random.random() > prob:
            return text, None
        random.shuffle(self._swap_tables)
        for swap_type, table in self._swap_tables:
            shuffled = self._apply_swap(text, table)
            if shuffled != text:
                logger.debug(f"内容混淆[{swap_type}]: {text[:20]} -> {shuffled[:20]}")
                return shuffled, swap_type
        return text, None

    def _apply_swap(self, text: str, table: Dict) -> str:
        for original, replacements in table.items():
            if original in text:
                return text.replace(original, random.choice(replacements), 1)
        return text

    def shuffle_with_info(self, text: str, prob: float = 0.35) -> Dict:
        shuffled, swap_type = self.shuffle(text, prob)
        return {
            'original': text,
            'shuffled': shuffled,
            'modified': shuffled != text,
            'type': swap_type,
        }


_shuffler: Optional[ContentShuffler] = None


def get_content_shuffler() -> ContentShuffler:
    global _shuffler
    if _shuffler is None:
        _shuffler = ContentShuffler()
    return _shuffler

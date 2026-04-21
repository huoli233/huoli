import random
import re
from typing import List, Tuple, Optional
from src.common.logger import get_logger

logger = get_logger("chinese_typo")


class ChineseTypoGenerator:
    """中文错别字生成器

    模拟人类打字时的常见错误：
    - 同音字替换
    - 形近字替换
    - 声调错误
    - 词语替换
    """

    COMMON_HOMOPHONES = {
        "的": ["地", "得"],
        "地": ["的", "得"],
        "得": ["的", "地"],
        "在": ["再"],
        "再": ["在"],
        "有": ["又"],
        "又": ["有"],
        "是": ["事", "式"],
        "事": ["是", "式"],
        "我": ["卧", "窝"],
        "你": ["泥", "拟"],
        "他": ["她", "它"],
        "她": ["他", "它"],
        "它": ["他", "她"],
        "这": ["者", "折"],
        "那": ["哪"],
        "哪": ["那"],
        "了": ["乐"],
        "乐": ["了"],
        "不": ["布", "步"],
        "就": ["旧", "舅"],
        "都": ["斗"],
        "还": ["孩", "海"],
        "会": ["汇", "绘"],
        "能": ["嫩"],
        "想": ["向", "象"],
        "要": ["药", "耀"],
        "去": ["趣", "取"],
        "来": ["莱", "赖"],
        "说": ["硕"],
        "看": ["砍", "侃"],
        "听": ["停", "廷"],
        "做": ["作", "坐"],
        "作": ["做", "坐"],
        "坐": ["做", "作"],
        "很": ["狠", "痕"],
        "好": ["号", "耗"],
        "多": ["躲", "朵"],
        "少": ["绍", "哨"],
        "大": ["打", "达"],
        "小": ["晓", "肖"],
        "上": ["尚", "伤"],
        "下": ["吓", "夏"],
        "里": ["理", "李"],
        "外": ["歪"],
        "前": ["钱", "乾"],
        "后": ["候", "厚"],
        "左": ["佐"],
        "右": ["又", "佑"],
        "中": ["钟", "忠"],
        "人": ["仁", "任"],
        "心": ["新", "欣"],
        "头": ["投", "透"],
        "手": ["首", "守"],
        "眼": ["演", "掩"],
        "口": ["扣", "叩"],
        "耳": ["尔", "饵"],
        "面": ["免", "棉"],
        "身": ["深", "申"],
        "自": ["字", "紫"],
        "己": ["几", "机"],
        "已": ["以", "乙"],
        "以": ["已", "乙"],
        "为": ["位", "未"],
        "因": ["音", "阴"],
        "但": ["蛋", "淡"],
        "而": ["儿", "耳"],
        "或": ["货", "获"],
        "和": ["合", "河"],
        "与": ["予", "雨"],
        "也": ["野", "冶"],
        "没": ["美", "眉", "梅", "煤"],
        "把": ["吧", "罢"],
        "被": ["备", "背"],
        "给": ["跟", "根"],
        "从": ["丛", "匆"],
        "到": ["道", "盗"],
        "用": ["永", "泳"],
        "对": ["队", "兑"],
        "让": ["嚷", "壤"],
        "比": ["笔", "彼"],
        "等": ["登", "灯"],
        "起": ["气", "弃"],
        "过": ["国", "果"],
        "进": ["近", "金"],
        "出": ["初", "除"],
        "开": ["凯", "楷"],
        "关": ["观", "官"],
        "点": ["电", "店"],
        "长": ["常", "场"],
        "高": ["搞", "稿"],
        "低": ["底", "抵"],
        "快": ["块", "筷"],
        "慢": ["满", "曼"],
        "新": ["心", "欣"],
        "老": ["脑", "恼"],
        "美": ["没", "每"],
        "丑": ["瞅", "愁"],
        "真": ["针", "珍"],
        "假": ["价", "架"],
        "错": ["措", "挫"],
    }

    COMMON_CONFUSABLE = {
        "己": "已",
        "已": "己",
        "戊": "戌",
        "戌": "戊",
        "戎": "戒",
        "戒": "戎",
        "日": "曰",
        "曰": "日",
        "未": "末",
        "末": "未",
        "甲": "由",
        "由": "甲",
        "人": "入",
        "入": "人",
        "土": "士",
        "士": "土",
        "牛": "午",
        "午": "牛",
        "免": "兔",
        "兔": "免",
        "拆": "折",
        "折": "拆",
        "拔": "拨",
        "拨": "拔",
        "亨": "享",
        "享": "亨",
        "幻": "幼",
        "幼": "幻",
        "弓": "引",
        "引": "弓",
        "户": "尸",
        "尸": "户",
        "爪": "瓜",
        "瓜": "爪",
        "乌": "鸟",
        "鸟": "乌",
        "斤": "斥",
        "斥": "斤",
        "今": "令",
        "令": "今",
        "仓": "仑",
        "仑": "仓",
        "见": "贝",
        "贝": "见",
        "内": "肉",
        "肉": "内",
        "冈": "岗",
        "岗": "冈",
        "气": "乞",
        "乞": "气",
        "天": "夫",
        "夫": "天",
        "夭": "天",
        "币": "巾",
        "巾": "币",
        "干": "于",
        "于": "干",
        "亏": "亏",
        "王": "玉",
        "玉": "王",
        "主": "王",
        "本": "木",
        "木": "本",
        "术": "木",
        "禾": "木",
        "灭": "灰",
        "灰": "灭",
        "太": "大",
        "犬": "大",
        "大": "太",
    }

    def __init__(
        self,
        error_rate: float = 0.02,
        min_freq: int = 1,
        tone_error_rate: float = 0.3,
        word_replace_rate: float = 0.3,
    ):
        self.error_rate = error_rate
        self.min_freq = min_freq
        self.tone_error_rate = tone_error_rate
        self.word_replace_rate = word_replace_rate

    def create_typo_sentence(self, sentence: str) -> Tuple[str, str]:
        """生成带有错别字的句子

        Args:
            sentence: 原始句子

        Returns:
            Tuple[str, str]: (错别字句子, 更正文本)
        """
        if not sentence or len(sentence) < 2:
            return sentence, ""

        if random.random() > self.error_rate:
            return sentence, ""

        chars = list(sentence)
        typo_positions = []
        corrections = []

        for i, char in enumerate(chars):
            if not self._is_chinese_char(char):
                continue

            typo_char = self._get_typo_char(char)
            if typo_char and typo_char != char:
                if random.random() < self._get_char_error_rate(
                    char, i, len(chars)
                ):
                    chars[i] = typo_char
                    typo_positions.append(i)
                    corrections.append(f"{char}→{typo_char}")

        if not typo_positions:
            return sentence, ""

        typo_sentence = "".join(chars)
        correction_text = "更正：" + "，".join(corrections)

        logger.debug(
            f"生成错别字: {sentence} -> {typo_sentence}, {correction_text}"
        )

        return typo_sentence, correction_text

    def _is_chinese_char(self, char: str) -> bool:
        """检查是否为中文字符"""
        return "\u4e00" <= char <= "\u9fff"

    def _get_char_error_rate(
        self, char: str, position: int, length: int
    ) -> float:
        """获取字符的错误概率

        根据位置和字符特性调整错误概率
        """
        base_rate = 1.0

        if position == 0 or position == length - 1:
            base_rate *= 0.5

        if char in "的地得了着过":
            base_rate *= 1.5

        return base_rate

    def _get_typo_char(self, char: str) -> Optional[str]:
        """获取字符的错别字

        优先级：
        1. 同音字替换
        2. 形近字替换
        """
        if char in self.COMMON_HOMOPHONES:
            homophones = self.COMMON_HOMOPHONES[char]
            if homophones:
                return random.choice(homophones)

        if char in self.COMMON_CONFUSABLE:
            return self.COMMON_CONFUSABLE[char]

        if random.random() < self.word_replace_rate:
            similar_char = self._find_similar_char(char)
            if similar_char:
                return similar_char

        return None

    def _find_similar_char(self, char: str) -> Optional[str]:
        """查找形近字

        使用简单的笔画相似性判断
        """
        similar_chars_map = {
            "人": ["入", "八"],
            "入": ["人", "八"],
            "大": ["太", "犬"],
            "太": ["大", "犬"],
            "犬": ["大", "太"],
            "日": ["曰", "目"],
            "曰": ["日", "目"],
            "目": ["日", "曰"],
            "田": ["由", "甲"],
            "由": ["田", "甲"],
            "甲": ["田", "由"],
            "白": ["自", "百"],
            "自": ["白", "百"],
            "百": ["白", "自"],
            "干": ["于", "千"],
            "于": ["干", "千"],
            "千": ["干", "于"],
            "土": ["士", "工"],
            "士": ["土", "工"],
            "工": ["土", "士"],
            "王": ["玉", "主"],
            "玉": ["王", "主"],
            "主": ["王", "玉"],
            "天": ["夫", "无"],
            "夫": ["天", "无"],
            "无": ["天", "夫"],
            "开": ["井", "并"],
            "井": ["开", "并"],
            "并": ["开", "井"],
            "月": ["用", "同"],
            "用": ["月", "同"],
            "同": ["月", "用"],
            "山": ["出", "仙"],
            "出": ["山", "仙"],
            "仙": ["山", "出"],
            "水": ["冰", "永"],
            "冰": ["水", "永"],
            "永": ["水", "冰"],
            "火": ["灭", "灰"],
            "灭": ["火", "灰"],
            "灰": ["火", "灭"],
        }

        if char in similar_chars_map:
            return random.choice(similar_chars_map[char])

        return None

    def add_custom_homophone(self, char: str, homophones: List[str]) -> None:
        """添加自定义同音字映射"""
        if char in self.COMMON_HOMOPHONES:
            self.COMMON_HOMOPHONES[char].extend(homophones)
        else:
            self.COMMON_HOMOPHONES[char] = homophones

    def add_custom_confusable(self, char: str, confusable: str) -> None:
        """添加自定义形近字映射"""
        self.COMMON_CONFUSABLE[char] = confusable

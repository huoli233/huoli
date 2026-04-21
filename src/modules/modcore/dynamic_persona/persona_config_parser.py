"""
人格配置解析器

结构化解析人格配置中的核心属性（年龄、性别、职业等）
让模型自主理解人格特征，而不是硬编码
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.common.config.config_engine import ConfigEngine
from src.common.logger import get_logger

logger = get_logger("人格配置解析")


@dataclass
class PersonaAttributes:
    """人格核心属性"""

    name: str = ""
    age: Optional[int] = None
    gender: str = ""
    occupation: str = ""
    education: str = ""
    personality_traits: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    speaking_style: str = ""
    background: str = ""
    raw_description: str = ""

    def to_prompt_fragment(self) -> str:
        """生成人格提示片段"""
        parts = []
        if self.name:
            parts.append(f"你的名字是{self.name}")
        if self.age:
            parts.append(f"你今年{self.age}岁")
        if self.gender:
            parts.append(f"性别{self.gender}")
        if self.occupation:
            parts.append(f"职业是{self.occupation}")
        if self.education:
            parts.append(f"学历{self.education}")
        if self.personality_traits:
            parts.append(f"性格特点：{'、'.join(self.personality_traits)}")
        if self.interests:
            parts.append(f"兴趣爱好：{'、'.join(self.interests)}")
        if self.speaking_style:
            parts.append(f"说话风格：{self.speaking_style}")
        if self.background:
            parts.append(f"背景：{self.background}")
        return "，".join(parts) + "。"


class PersonaConfigParser:
    """人格配置解析器

    从配置文件中结构化解析人格属性
    支持自然语言描述的智能解析
    """

    AGE_PATTERNS = [
        r"(\d+)岁",
        r"(\d+)年级",
        r"大二|大三|大四|大一",
        r"研一|研二|研三",
        r"高中生|初中生|小学生",
    ]

    GENDER_PATTERNS = {
        "女": ["女", "女生", "女性", "女孩", "少女", "妹子"],
        "男": ["男", "男生", "男性", "男孩", "少年", "汉子"],
    }

    OCCUPATION_PATTERNS = {
        "学生": ["学生", "大学生", "研究生", "博士生", "高中生", "初中生"],
        "程序员": ["程序员", "开发", "工程师", "码农"],
        "教师": ["老师", "教师", "教授", "讲师"],
        "医生": ["医生", "医师", "护士"],
        "设计师": ["设计师", "UI", "UX"],
        "作家": ["作家", "作者", "写手"],
    }

    EDUCATION_PATTERNS = {
        "本科": ["本科", "大学", "大二", "大三", "大四", "大一"],
        "硕士": ["硕士", "研究生", "研一", "研二", "研三"],
        "博士": ["博士", "博士生"],
        "高中": ["高中", "高中生"],
        "初中": ["初中", "初中生"],
    }

    PERSONALITY_KEYWORDS = {
        "活泼": ["活泼", "开朗", "外向", "热情"],
        "温柔": ["温柔", "温和", "柔和", "体贴"],
        "傲娇": ["傲娇", "口是心非", "别扭"],
        "毒舌": ["毒舌", "刻薄", "尖锐", "攻击性"],
        "内向": ["内向", "害羞", "腼腆", "社恐"],
        "幽默": ["幽默", "风趣", "搞笑", "逗比"],
        "理性": ["理性", "冷静", "理智", "逻辑"],
        "感性": ["感性", "情绪化", "多愁善感"],
        "傲慢": ["傲慢", "高傲", "自大"],
        "腹黑": ["腹黑", "心机", "城府"],
    }

    INTEREST_KEYWORDS = {
        "游戏": ["游戏", "打游戏", "玩游戏", "电竞"],
        "动漫": ["动漫", "动画", "番剧", "二次元"],
        "音乐": ["音乐", "唱歌", "听歌", "乐器"],
        "阅读": ["阅读", "看书", "读书", "小说"],
        "运动": ["运动", "健身", "跑步", "打球"],
        "美食": ["美食", "吃", "做饭", "烹饪"],
        "旅行": ["旅行", "旅游", "出游"],
        "摄影": ["摄影", "拍照", "相机"],
        "编程": ["编程", "代码", "程序"],
    }

    def __init__(self, config_engine: Optional[ConfigEngine] = None):
        self._config = config_engine

    def parse(
        self, personality_text: str, bot_name: str = ""
    ) -> PersonaAttributes:
        """解析人格描述文本

        Args:
            personality_text: 人格描述文本
            bot_name: 机器人名称

        Returns:
            PersonaAttributes: 结构化的人格属性
        """
        attrs = PersonaAttributes(
            name=bot_name,
            raw_description=personality_text,
        )

        attrs.age = self._extract_age(personality_text)
        attrs.gender = self._extract_gender(personality_text)
        attrs.occupation = self._extract_occupation(personality_text)
        attrs.education = self._extract_education(personality_text)
        attrs.personality_traits = self._extract_personality(personality_text)
        attrs.interests = self._extract_interests(personality_text)
        attrs.speaking_style = self._extract_speaking_style(personality_text)
        attrs.background = self._extract_background(personality_text)

        return attrs

    def _extract_age(self, text: str) -> Optional[int]:
        """提取年龄"""
        for pattern in self.AGE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                if pattern == r"(\d+)岁":
                    return int(match.group(1))
                elif pattern == r"(\d+)年级":
                    grade = int(match.group(1))
                    return grade + 6
                elif "大一" in text:
                    return 19
                elif "大二" in text:
                    return 20
                elif "大三" in text:
                    return 21
                elif "大四" in text:
                    return 22
                elif "研一" in text:
                    return 23
                elif "研二" in text:
                    return 24
                elif "研三" in text:
                    return 25
                elif "高中生" in text:
                    return 17
                elif "初中生" in text:
                    return 14
                elif "小学生" in text:
                    return 10
        return None

    def _extract_gender(self, text: str) -> str:
        """提取性别"""
        for gender, keywords in self.GENDER_PATTERNS.items():
            for kw in keywords:
                if kw in text:
                    return gender
        return ""

    def _extract_occupation(self, text: str) -> str:
        """提取职业"""
        for occupation, keywords in self.OCCUPATION_PATTERNS.items():
            for kw in keywords:
                if kw in text:
                    return occupation
        return ""

    def _extract_education(self, text: str) -> str:
        """提取学历"""
        for education, keywords in self.EDUCATION_PATTERNS.items():
            for kw in keywords:
                if kw in text:
                    return education
        return ""

    def _extract_personality(self, text: str) -> List[str]:
        """提取性格特点"""
        traits = []
        for trait, keywords in self.PERSONALITY_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    traits.append(trait)
                    break
        return traits

    def _extract_interests(self, text: str) -> List[str]:
        """提取兴趣爱好"""
        interests = []
        for interest, keywords in self.INTEREST_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    interests.append(interest)
                    break
        return interests

    def _extract_speaking_style(self, text: str) -> str:
        """提取说话风格"""
        style_patterns = [
            (r"有时有点攻击性", "有时有点攻击性"),
            (r"有时比较温柔", "有时比较温柔"),
            (r"说话(.{2,10})", r"说话\1"),
            (r"回复风格(.{2,20})", r"\1"),
        ]
        for pattern, _replacement in style_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return ""

    def _extract_background(self, text: str) -> str:
        """提取背景信息"""
        background_patterns = [
            r"是(.{2,15}学生)",
            r"是(.{2,15}工程师)",
            r"是(.{2,15}设计师)",
            r"现在正在(.{2,20})",
        ]
        for pattern in background_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return ""

    async def generate_dynamic_persona_fragment(
        self,
        attrs: PersonaAttributes,
        context: str = "",
        model_client: Any = None,
    ) -> str:
        """让模型生成动态人格片段

        根据当前状态和上下文，让模型自主生成人格表现
        """
        if model_client is None:
            return attrs.to_prompt_fragment()

        try:
            prompt = f"""你是一个角色扮演者，需要根据以下人格设定生成一个简短的人格表现描述。

## 人格设定

{attrs.to_prompt_fragment()}

原始描述：{attrs.raw_description}

## 当前上下文

{context[:200] if context else '无特定上下文'}

## 任务

请生成一个简短的人格表现描述（1-2句话），体现当前状态下这个角色会如何表现。
不要重复人格设定，而是描述当前的心境、态度或行为倾向。

直接输出描述，不要有任何前缀或格式。"""

            if hasattr(model_client, "generate"):
                response = await model_client.generate(prompt)
            else:
                return attrs.to_prompt_fragment()

            if response and len(response) > 5:
                return response.strip()
        except Exception as e:
            logger.debug(f"生成动态人格片段失败: {e}")

        return attrs.to_prompt_fragment()


_persona_parser: Optional[PersonaConfigParser] = None


def get_persona_parser(
    config_engine: Optional[ConfigEngine] = None,
) -> PersonaConfigParser:
    """获取人格配置解析器单例"""
    global _persona_parser
    if _persona_parser is None:
        _persona_parser = PersonaConfigParser(config_engine)
    return _persona_parser

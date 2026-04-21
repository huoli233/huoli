import random
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from src.common.logger import get_logger
from src.common.config.config_engine import ConfigEngine

logger = get_logger("感知生成")


@dataclass
class PerceptionContext:
    """感知上下文数据类"""

    physical_sensation: str = ""
    cognitive_state: str = ""
    emotional_pressure: str = ""
    social_willingness: str = ""
    subconscious_interference: str = ""
    trauma_flashback: str = ""
    fatigue_effect: str = ""

    def to_prompt_text(self) -> str:
        parts = []
        if self.physical_sensation:
            parts.append(f"【生理感受】{self.physical_sensation}")
        if self.cognitive_state:
            parts.append(f"【认知状态】{self.cognitive_state}")
        if self.emotional_pressure:
            parts.append(f"【情绪压力】{self.emotional_pressure}")
        if self.social_willingness:
            parts.append(f"【社交意愿】{self.social_willingness}")
        if self.subconscious_interference:
            parts.append(f"【潜意识干扰】{self.subconscious_interference}")
        if self.trauma_flashback:
            parts.append(f"【创伤闪回】{self.trauma_flashback}")
        if self.fatigue_effect:
            parts.append(f"【疲劳影响】{self.fatigue_effect}")
        if not parts:
            return ""
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, str]:
        return {
            "physical_sensation": self.physical_sensation,
            "cognitive_state": self.cognitive_state,
            "emotional_pressure": self.emotional_pressure,
            "social_willingness": self.social_willingness,
            "subconscious_interference": self.subconscious_interference,
            "trauma_flashback": self.trauma_flashback,
            "fatigue_effect": self.fatigue_effect,
        }


DESCRIPTOR_ATOMS: Dict[str, Dict[str, List[str]]] = {
    "trauma_physical": {
        "0-2": ["呼吸平稳", "身体放松", "肌肉没有紧张感", "心跳正常"],
        "2-4": [
            "隐隐的不安感",
            "肩膀有些紧绷",
            "手指不自觉地蜷缩",
            "略微僵硬",
        ],
        "4-6": ["心跳开始加快", "手心微微出汗", "喉咙有些发紧", "胃部不适"],
        "6-8": ["胸口发闷", "呼吸变得急促", "后背开始冒冷汗", "身体微微发抖"],
        "8-10": ["心脏狂跳", "全身肌肉紧绷到发抖", "呼吸困难", "手脚冰凉"],
    },
    "trauma_cognitive": {
        "0-2": ["思维清晰", "能够专注", "逻辑连贯", "反应敏捷"],
        "2-4": ["偶尔走神", "注意力略有分散", "思绪有些飘忽", "反应稍慢"],
        "4-6": [
            "思维开始混乱",
            "难以集中注意力",
            "脑海中闪过一些念头",
            "记忆模糊",
        ],
        "6-8": [
            "思绪非常混乱",
            "几乎无法思考",
            "脑子里像有嗡嗡声",
            "意识飘忽",
        ],
        "8-10": [
            "意识支离破碎",
            "完全无法控制思维",
            "大脑一片空白",
            "失去方向感",
        ],
    },
    "trauma_emotional": {
        "0-2": ["情绪平静", "内心安宁", "没有压抑感", "心情舒畅"],
        "2-4": ["内心有些不安", "隐约的焦虑", "情绪略显波动", "有些烦躁"],
        "4-6": ["明显的焦虑感", "内心充满压抑", "情绪难以控制", "想要逃避"],
        "6-8": ["强烈的恐慌", "濒临崩溃的边缘", "情绪如同风暴", "极度不安"],
        "8-10": ["彻底的绝望", "情感完全失控", "内心支离破碎", "陷入黑暗"],
    },
    "fatigue_physical": {
        "0-20": ["精力充沛", "身体轻盈", "没有疲惫感", "状态极佳"],
        "20-40": ["略感疲惫", "眼皮有些沉重", "想要打哈欠", "身体微沉"],
        "40-60": ["明显的疲劳", "身体沉重", "思维变慢", "想要休息"],
        "60-80": ["非常疲惫", "每一个动作都很费力", "眼睛酸涩", "浑身乏力"],
        "80-100": ["精疲力竭", "几乎无法支撑", "只想倒头就睡", "意识模糊"],
    },
    "fatigue_cognitive": {
        "0-20": ["思维敏捷", "反应迅速", "注意力集中", "记忆清晰"],
        "20-40": ["思维略慢", "偶尔走神", "反应稍迟钝", "注意力下降"],
        "40-60": ["思维迟缓", "难以集中", "反应明显变慢", "记忆力下降"],
        "60-80": ["思维困难", "几乎无法专注", "反应迟钝", "健忘"],
        "80-100": ["无法思考", "意识模糊", "完全无法集中", "失去判断力"],
    },
    "affection_social": {
        "negative": ["不想搭理", "想要远离", "保持距离", "冷淡应对"],
        "low": ["有些勉强", "不太情愿", "敷衍应对", "想要结束"],
        "neutral": ["正常应对", "适度交流", "保持礼貌", "一般态度"],
        "positive": ["愿意交流", "期待互动", "态度友好", "想要更多"],
        "high": ["非常期待", "热情回应", "想要亲近", "充满期待"],
    },
    "annoyance_emotional": {
        "0-20": ["心情平和", "没有烦躁", "情绪稳定", "耐心充足"],
        "20-40": ["略有不满", "有些不耐烦", "情绪微波动", "耐心下降"],
        "40-60": ["明显烦躁", "难以忍受", "情绪波动大", "想要发作"],
        "60-80": ["非常愤怒", "濒临爆发", "情绪失控边缘", "极度不耐烦"],
        "80-100": ["彻底爆发", "无法控制", "情绪崩溃", "完全失控"],
    },
}


class PerceptionGenerator:
    """感知生成器

    根据心理状态生成感知上下文描述。
    支持创伤、疲劳、情感等多维度感知生成。
    """

    _instance: Optional["PerceptionGenerator"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._config = ConfigEngine.get_instance()
        self._descriptor_atoms = DESCRIPTOR_ATOMS
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._cache_ttl = 60.0
        self._initialized = True
        logger.info("感知生成器初始化完成")

    def generate_perception(
        self,
        trauma_score: float = 0.0,
        inner_chaos: float = 0.0,
        surface_mask: float = 0.0,
        pressure_level: float = 0.0,
        affection: float = 0.0,
        annoyance: float = 0.0,
        trust: float = 0.0,
        context_seed: str = "",
    ) -> PerceptionContext:
        """生成感知上下文"""
        physical = self._generate_physical_sensation(
            trauma_score, pressure_level, context_seed
        )
        cognitive = self._generate_cognitive_state(
            trauma_score, pressure_level, inner_chaos, context_seed
        )
        emotional = self._generate_emotional_pressure(
            trauma_score, annoyance, inner_chaos, context_seed
        )
        social = self._generate_social_willingness(
            affection, annoyance, pressure_level, context_seed
        )
        subconscious = self._generate_subconscious_interference(
            trauma_score, inner_chaos, context_seed
        )
        flashback = self._generate_trauma_flashback(
            trauma_score, inner_chaos, surface_mask, context_seed
        )
        fatigue_effect = self._generate_pressure_effect(
            pressure_level, context_seed
        )
        return PerceptionContext(
            physical_sensation=physical,
            cognitive_state=cognitive,
            emotional_pressure=emotional,
            social_willingness=social,
            subconscious_interference=subconscious,
            trauma_flashback=flashback,
            fatigue_effect=fatigue_effect,
        )

    def generate_from_state(
        self, state: Dict[str, Any], context_seed: str = ""
    ) -> PerceptionContext:
        """从状态字典生成感知上下文"""
        return self.generate_perception(
            trauma_score=state.get("trauma_score", 0.0),
            inner_chaos=state.get("inner_chaos_level", 0.0),
            surface_mask=state.get("surface_mask_strength", 0.0),
            pressure_level=0.0,
            affection=state.get("affection", 0.0),
            annoyance=state.get("annoyance", 0.0),
            trust=state.get("trust", 0.0),
            context_seed=context_seed,
        )

    def _generate_physical_sensation(
        self, trauma: float, pressure: float, seed: str
    ) -> str:
        trauma_desc = self._select_descriptor(
            "trauma_physical", trauma, seed + "_trauma"
        )
        pressure_desc = self._select_descriptor(
            "fatigue_physical", pressure, seed + "_pressure"
        )
        if trauma > 5 and pressure > 40:
            return f"{trauma_desc}，同时{pressure_desc}"
        elif trauma > 5:
            return trauma_desc
        elif pressure > 40:
            return pressure_desc
        return ""

    def _generate_cognitive_state(
        self, trauma: float, pressure: float, chaos: float, seed: str
    ) -> str:
        trauma_desc = self._select_descriptor(
            "trauma_cognitive", trauma, seed + "_cog_trauma"
        )
        pressure_desc = self._select_descriptor(
            "fatigue_cognitive", pressure, seed + "_cog_pressure"
        )
        if chaos > 5:
            return trauma_desc
        elif pressure > 50:
            return pressure_desc
        elif trauma > 3:
            return trauma_desc
        return ""

    def _generate_emotional_pressure(
        self, trauma: float, annoyance: float, chaos: float, seed: str
    ) -> str:
        trauma_desc = self._select_descriptor(
            "trauma_emotional", trauma, seed + "_emo_trauma"
        )
        annoyance_desc = self._select_descriptor(
            "annoyance_emotional", annoyance, seed + "_annoyance"
        )
        if trauma > 6:
            return trauma_desc
        elif annoyance > 50:
            return annoyance_desc
        elif trauma > 3:
            return trauma_desc
        return ""

    def _generate_social_willingness(
        self, affection: float, annoyance: float, pressure: float, seed: str
    ) -> str:
        if pressure > 70:
            level = "negative"
        elif annoyance > 60:
            level = "negative"
        elif affection > 60:
            level = "high"
        elif affection > 30:
            level = "positive"
        elif affection < -30:
            level = "low"
        elif annoyance > 30:
            level = "low"
        else:
            level = "neutral"
        return self._select_descriptor(
            "affection_social", level, seed + "_social"
        )

    def _generate_subconscious_interference(
        self, trauma: float, chaos: float, seed: str
    ) -> str:
        if trauma < 3 or chaos < 3:
            return ""
        interferences = [
            "脑海中闪过一些模糊的画面",
            "内心深处有声音在低语",
            "一些被压抑的记忆在翻涌",
            "潜意识里的恐惧在蠢蠢欲动",
            "过去的阴影若隐若现",
        ]
        idx = self._deterministic_random(seed + "_sub", len(interferences))
        return interferences[idx]

    def _generate_trauma_flashback(
        self, trauma: float, chaos: float, mask: float, seed: str
    ) -> str:
        if trauma < 5 or chaos < 4:
            return ""
        if mask > 7:
            return ""
        flashbacks = [
            "过去的痛苦记忆突然涌现",
            "熟悉的恐惧感再次袭来",
            "那些画面在眼前闪过",
            "心跳加速，仿佛回到了那一刻",
            "无法控制地想起那些事",
        ]
        idx = self._deterministic_random(seed + "_flash", len(flashbacks))
        return flashbacks[idx]

    def _generate_pressure_effect(self, pressure: float, seed: str) -> str:
        if pressure < 50:
            return ""
        effects = [
            "感到有些疲惫，想要休息",
            "精力不济，反应变慢",
            "身体沉重，想要躺下",
            "眼皮沉重，难以集中",
        ]
        idx = self._deterministic_random(seed + "_pressure", len(effects))
        return effects[idx]

    def _select_descriptor(
        self, category: str, value: float, seed: str
    ) -> str:
        atoms = self._descriptor_atoms.get(category, {})
        if not atoms:
            return ""
        range_key = self._find_range_key(value, atoms)
        descriptors = atoms.get(range_key, [])
        if not descriptors:
            return ""
        idx = self._deterministic_random(seed, len(descriptors))
        return descriptors[idx]

    def _find_range_key(
        self, value: float, atoms: Dict[str, List[str]]
    ) -> str:
        for key in atoms.keys():
            if "-" in key:
                try:
                    low, high = map(float, key.split("-"))
                    if low <= value < high:
                        return key
                except ValueError:
                    continue
        for key in atoms.keys():
            if "-" not in key:
                return key
        return list(atoms.keys())[0] if atoms else ""

    def _deterministic_random(self, seed: str, max_val: int) -> int:
        if not seed:
            return random.randint(0, max_val - 1)
        hash_val = int(hashlib.md5(seed.encode()).hexdigest(), 16)
        return hash_val % max_val

    def get_quick_perception(
        self, trauma: float, pressure: float, affection: float
    ) -> str:
        """快速获取感知描述"""
        ctx = self.generate_perception(
            trauma_score=trauma, pressure_level=pressure, affection=affection
        )
        return ctx.to_prompt_text()


_perception_generator_instance: Optional[PerceptionGenerator] = None


def get_perception_generator() -> PerceptionGenerator:
    """获取感知生成器单例"""
    global _perception_generator_instance
    if _perception_generator_instance is None:
        _perception_generator_instance = PerceptionGenerator()
    return _perception_generator_instance

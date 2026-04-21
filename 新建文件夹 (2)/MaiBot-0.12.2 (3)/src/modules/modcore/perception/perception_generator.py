import random
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("perception_generator")


@dataclass
class PerceptionContext:
    physical_sensation: str = ""
    cognitive_state: str = ""
    emotional_pressure: str = ""
    social_willingness: str = ""
    subconscious_interference: str = ""

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
        if not parts:
            return ""
        return "\n".join(parts)


class PerceptionGenerator:
    _instance: Optional["PerceptionGenerator"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._descriptor_atoms = self._build_descriptor_atoms()
        self._initialized = True
        logger.info("感知生成器初始化完成")

    def _build_descriptor_atoms(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "trauma_physical": {
                "0-2": ["呼吸平稳", "身体放松", "肌肉没有紧张感"],
                "2-4": ["隐隐的不安感", "肩膀有些紧绷", "手指不自觉地蜷曲"],
                "4-6": ["心跳开始加快", "手心微微出汗", "喉咙有些发紧"],
                "6-8": ["胸口发闷", "呼吸变得急促", "后背开始冒冷汗"],
                "8-10": ["心脏狂跳", "全身肌肉紧绷到发抖", "呼吸困难"],
            },
            "trauma_cognitive": {
                "0-2": ["思维清晰", "能够专注", "逻辑连贯"],
                "2-4": ["偶尔走神", "注意力略有分散", "思绪有些飘忽"],
                "4-6": ["思维开始混乱", "难以集中注意力", "脑海中闪过一些不相关的念头"],
                "6-8": ["思绪非常混乱", "几乎无法思考", "脑子里像有嗡嗡声"],
                "8-10": ["意识支离破碎", "完全无法控制思维", "大脑一片空白"],
            },
            "trauma_emotional": {
                "0-2": ["情绪平静", "内心安宁", "没有压抑感"],
                "2-4": ["内心有些不安", "隐约的焦虑", "情绪略显波动"],
                "4-6": ["明显的焦虑感", "内心充满压抑", "情绪难以控制"],
                "6-8": ["强烈的恐慌", "濒临崩溃的边缘", "情绪如同风暴"],
                "8-10": ["彻底的绝望", "情感完全失控", "内心支离破碎"],
            },
            "fatigue_physical": {
                "0-20": ["精力充沛", "身体轻盈", "没有疲惫感"],
                "20-40": ["略感疲惫", "眼皮有些沉重", "想要打哈欠"],
                "40-60": ["明显的疲劳", "身体沉重", "思维变慢"],
                "60-80": ["非常疲惫", "每一个动作都很费力", "眼睛酸涩"],
                "80-100": ["精疲力竭", "几乎无法支撑", "只想倒头就睡"],
            },
            "fatigue_cognitive": {
                "0-20": ["思维敏捷", "反应迅速", "逻辑清晰"],
                "20-40": ["思考速度略慢", "需要更多时间理解", "偶尔反应迟钝"],
                "40-60": ["思维明显迟缓", "理解困难", "记忆力下降"],
                "60-80": ["几乎无法思考", "大脑像生锈了一样", "记不住刚才说了什么"],
                "80-100": ["意识模糊", "思维完全停滞", "只想让一切停止"],
            },
            "fatigue_social": {
                "0-20": ["愿意交流", "享受对话", "充满活力"],
                "20-40": ["对话开始感到疲惫", "不太想说长句", "更倾向于简短回复"],
                "40-60": ["明显不想多说话", "只想安静一会儿", "社交成为负担"],
                "60-80": ["极度抗拒交流", "每句话都是煎熬", "只想独处"],
                "80-100": ["完全无法应对社交", "只想逃离", "任何互动都让人崩溃"],
            },
            "mask_deterioration": {
                "0-2": ["能够完美地控制表情和语气", "外表毫无破绽", "保持得体的形象"],
                "2-4": ["偶尔会露出一些真实情绪", "伪装开始有些吃力", "需要刻意维持表面的平静"],
                "4-6": ["明显难以维持伪装", "情绪不时泄露", "表情管理开始失控"],
                "6-8": ["伪装几乎崩溃", "真实情绪不断涌现", "几乎无法掩饰内心的痛苦"],
                "8-10": ["完全无法伪装", "情绪彻底暴露", "已经放弃维持表面形象"],
            },
            "annoyance_tone": {
                "0-20": ["心平气和", "没有烦躁感", "语气平和"],
                "20-40": ["略显不耐烦", "语气开始变硬", "有些烦躁"],
                "40-60": ["明显的厌烦", "语气冷淡", "不想多费口舌"],
                "60-80": ["非常烦躁", "语气充满敌意", "几乎要忍不住发火"],
                "80-100": ["愤怒到极点", "语气尖锐刺耳", "完全失去耐心"],
            },
        }

    def _get_range_key(self, value: float, ranges: List[Tuple[float, float, str]]) -> str:
        for min_val, max_val, key in ranges:
            if min_val <= value < max_val:
                return key
        return ranges[-1][2]

    def _select_descriptor(self, category: str, value: float,
                            ranges: List[Tuple[float, float, str]], seed: str) -> str:
        range_key = self._get_range_key(value, ranges)
        descriptors = self._descriptor_atoms.get(category, {}).get(range_key, [""])
        if not descriptors:
            return ""
        random.seed(hashlib.md5(seed.encode()).hexdigest())
        selected = random.choice(descriptors)
        random.seed()
        return selected

    def generate(self, trauma_score: float = 0.0, mental_fatigue: float = 0.0,
                 surface_mask: float = 10.0, inner_chaos: float = 0.0,
                 annoyance: float = 0.0, affection: float = 0.0,
                 flashback_fragments: Optional[List[str]] = None,
                 user_id: str = "", channel_id: str = "") -> PerceptionContext:
        current_hour = int(time.time() / 3600)
        seed_base = f"{user_id}_{channel_id}_{current_hour}"
        trauma_ranges = [(0, 2, "0-2"), (2, 4, "2-4"), (4, 6, "4-6"), (6, 8, "6-8"), (8, 11, "8-10")]
        fatigue_ranges = [(0, 20, "0-20"), (20, 40, "20-40"), (40, 60, "40-60"), (60, 80, "60-80"), (80, 101, "80-100")]
        mask_ranges = [(0, 2, "8-10"), (2, 4, "6-8"), (4, 6, "4-6"), (6, 8, "2-4"), (8, 11, "0-2")]
        annoyance_ranges = fatigue_ranges
        perception = PerceptionContext()
        if trauma_score > 1.5 or mental_fatigue > 15:
            physical_parts = []
            if trauma_score > 1.5:
                trauma_phys = self._select_descriptor("trauma_physical", trauma_score, trauma_ranges, f"{seed_base}_trauma_phys")
                if trauma_phys:
                    physical_parts.append(trauma_phys)
            if mental_fatigue > 15:
                fatigue_phys = self._select_descriptor("fatigue_physical", mental_fatigue, fatigue_ranges, f"{seed_base}_fatigue_phys")
                if fatigue_phys:
                    physical_parts.append(fatigue_phys)
            perception.physical_sensation = "，".join(physical_parts) + "。"
        if trauma_score > 1.5 or mental_fatigue > 15:
            cognitive_parts = []
            if trauma_score > 1.5:
                trauma_cog = self._select_descriptor("trauma_cognitive", trauma_score, trauma_ranges, f"{seed_base}_trauma_cog")
                if trauma_cog:
                    cognitive_parts.append(trauma_cog)
            if mental_fatigue > 15:
                fatigue_cog = self._select_descriptor("fatigue_cognitive", mental_fatigue, fatigue_ranges, f"{seed_base}_fatigue_cog")
                if fatigue_cog:
                    cognitive_parts.append(fatigue_cog)
            perception.cognitive_state = "，".join(cognitive_parts) + "。"
        if trauma_score > 2.0:
            trauma_emo = self._select_descriptor("trauma_emotional", trauma_score, trauma_ranges, f"{seed_base}_trauma_emo")
            perception.emotional_pressure = trauma_emo + "。" if trauma_emo else ""
        if mental_fatigue > 20 or annoyance > 20:
            social_parts = []
            if mental_fatigue > 20:
                fatigue_social = self._select_descriptor("fatigue_social", mental_fatigue, fatigue_ranges, f"{seed_base}_fatigue_social")
                if fatigue_social:
                    social_parts.append(fatigue_social)
            if annoyance > 20:
                annoyance_tone = self._select_descriptor("annoyance_tone", annoyance, annoyance_ranges, f"{seed_base}_annoyance_tone")
                if annoyance_tone:
                    social_parts.append(annoyance_tone)
            perception.social_willingness = "，".join(social_parts) + "。"
        mask_value = 10.0 - surface_mask
        if mask_value > 2.0:
            mask_desc = self._select_descriptor("mask_deterioration", mask_value, mask_ranges, f"{seed_base}_mask")
            if mask_desc and not perception.emotional_pressure:
                perception.emotional_pressure = mask_desc + "。"
            elif mask_desc:
                perception.emotional_pressure += f" {mask_desc}。"
        if flashback_fragments and len(flashback_fragments) > 0 and trauma_score > 3.0:
            random.seed(hashlib.md5(f"{seed_base}_flashback".encode()).hexdigest())
            selected_fragment = random.choice(flashback_fragments)
            random.seed()
            faded_fragment = self._fade_memory(selected_fragment, fade_level=0.3)
            perception.subconscious_interference = f"脑海中闪过一些破碎的画面：{faded_fragment}..."
        return perception

    def _fade_memory(self, content: str, fade_level: float = 0.3) -> str:
        if not content or fade_level <= 0:
            return content
        chars = list(content)
        fade_count = int(len(chars) * fade_level)
        if fade_count == 0:
            return content
        fade_indices = random.sample(range(len(chars)), min(fade_count, len(chars)))
        fade_markers = ["…", "□", "▓"]
        for idx in fade_indices:
            chars[idx] = random.choice(fade_markers)
        return "".join(chars)

    def generate_persona_perception(self, persona_name: str, persona_tone: str = "",
                                     trauma_score: float = 0.0, mental_fatigue: float = 0.0,
                                     surface_mask: float = 10.0) -> str:
        persona_descriptors = {
            "愤怒": "你现在非常愤怒，感到胸口发闷，想要发泄。你的耐心已经耗尽，不想再伪装礼貌。",
            "创伤": "你感到受伤和脆弱，心里很痛。你不想说话，只想躲起来。",
            "防御": "你处于高度警惕状态，不信任任何人。你的回复简短而冷漠。",
            "崩溃": "你的思维一片混乱，说不出完整的话。你感到绝望和无力。",
            "厌恶": "你对一切感到反感和厌倦。你的语气充满讽刺和冷淡。",
            "冒犯": "你感到被冒犯了，想要反击。你不会再委婉客气。",
        }
        base_desc = ""
        for key, desc in persona_descriptors.items():
            if key in persona_name or key in persona_tone:
                base_desc = desc
                break
        if not base_desc:
            base_desc = f"你处于{persona_name}状态。"
        if trauma_score > 5:
            base_desc += " 创伤记忆在折磨你。"
        if mental_fatigue > 70:
            base_desc += " 你非常疲惫，不想多说话。"
        if surface_mask < 5:
            base_desc += " 你无法再伪装，真实情绪会直接表现出来。"
        return base_desc


    def generate_perception(self, stream_id: str, user_id: str) -> Dict[str, Any]:
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(stream_id)
            state = tracker.get_user_state(user_id, create_if_missing=False)
            if not state:
                return {}
            ctx = self.generate(
                trauma_score=state.trauma_score,
                mental_fatigue=state.mental_fatigue if hasattr(state, 'mental_fatigue') else 0.0,
                surface_mask=state.surface_mask if hasattr(state, 'surface_mask') else 10.0,
                inner_chaos=state.inner_chaos if hasattr(state, 'inner_chaos') else 0.0,
                annoyance=state.annoyance,
                affection=state.affection,
                user_id=user_id,
                channel_id=stream_id,
            )
            hint = ctx.to_prompt_text() if ctx else ""
            return {"hint": hint, "context": ctx} if hint else {}
        except Exception as e:
            logger.debug(f"感知生成失败: {e}")
            return {}


_perception_generator_instance: Optional[PerceptionGenerator] = None


def get_perception_generator() -> PerceptionGenerator:
    global _perception_generator_instance
    if _perception_generator_instance is None:
        _perception_generator_instance = PerceptionGenerator()
    return _perception_generator_instance

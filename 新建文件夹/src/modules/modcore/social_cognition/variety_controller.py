import time
import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("多样性控制")


@dataclass
class StyleProfile:
    name: str
    desc: str
    weight: float = 1.0


@dataclass
class PatternProfile:
    name: str
    desc: str
    weight: float = 1.0


class VarietyController:
    STYLE_POOL: List[StyleProfile] = [
        StyleProfile("standard", "自然、随性、不做作", 1.0),
        StyleProfile("witty", "幽默风趣，喜欢玩梗，语气轻快", 1.0),
        StyleProfile("spicy", "略带毒舌，一针见血，不留情面", 0.8),
        StyleProfile("gentle", "温柔体贴，如沐春风，治愈系", 1.0),
        StyleProfile("logical", "理性客观，逻辑严密，像个理科生", 0.9),
        StyleProfile("impatient", "略显不耐烦，说话简短有力，酷酷的", 0.7),
        StyleProfile("playful", "调皮捣蛋，喜欢恶作剧，古灵精怪", 0.8),
        StyleProfile("philosophical", "深沉思考，喜欢引经据典，文艺范", 0.6),
    ]

    PATTERN_POOL: List[PatternProfile] = [
        PatternProfile("direct", "直接回答型：不绕弯子，直击要点", 1.0),
        PatternProfile("question", "反问引导型：通过反问引发对方思考", 0.9),
        PatternProfile("empathy", "情绪共鸣型：先认同情绪，再给建议", 1.0),
        PatternProfile("tease", "调侃吐槽型：对现状进行幽默的解构", 0.8),
        PatternProfile("minimal", "极简风格型：能用一个字绝不用两个字", 0.7),
        PatternProfile("story", "故事叙述型：用小故事或比喻来表达", 0.6),
        PatternProfile("socratic", "苏格拉底式：层层递进，引导发现", 0.5),
    ]

    def __init__(self, channel_id: str):
        self.channel_id = channel_id
        self.current_style_index = 0
        self.current_pattern_index = 0
        self.last_rotation_time = 0.0
        self._rotation_base_period = 1800
        self._min_rotation_period = 300
        self._state_cache: Dict[str, any] = {}
        self._recent_bot_phrases: List[str] = []
        self._max_recent_phrases = 20
        self._load_state()
        logger.debug(
            f"VarietyController 初始化: channel={channel_id}, style={
                self.current_style_name}"
        )

    @property
    def current_style_name(self) -> str:
        return self.STYLE_POOL[self.current_style_index].name

    @property
    def current_style_desc(self) -> str:
        return self.STYLE_POOL[self.current_style_index].desc

    @property
    def current_pattern_name(self) -> str:
        return self.PATTERN_POOL[self.current_pattern_index].name

    @property
    def current_pattern_desc(self) -> str:
        return self.PATTERN_POOL[self.current_pattern_index].desc

    def get_dynamic_temperature(self, global_rage: float = 0.0) -> float:
        base_temp = 0.75
        rage_mod = min(0.4, (global_rage / 50.0) * 0.1)
        noise = random.uniform(-0.05, 0.05)
        final_temp = base_temp + rage_mod + noise
        return max(0.2, min(1.4, final_temp))

    async def get_diversity_instruction(
        self, global_rage: float = 0.0
    ) -> Tuple[str, float]:
        self._rotate_style_if_needed(global_rage)
        current_style = self.STYLE_POOL[self.current_style_index]
        current_pattern = self.PATTERN_POOL[self.current_pattern_index]
        anti_rep = await self._generate_anti_repetition_prompt()
        temp = self.get_dynamic_temperature(global_rage)
        instruction = f"""【多态性控制系统】
- 当前人格面具：{current_style.name} ({current_style.desc})
- 推荐回复模式：{current_pattern.desc}
- 动态熵值调节：已激活 (当前系统表现出较低的确定性，请尝试更多样化的表达)

{anti_rep}
"""
        return instruction.strip(), temp

    def _rotate_style_if_needed(self, global_rage: float = 0.0):
        now = time.time()
        period = self._rotation_base_period
        if global_rage > 50:
            period = self._min_rotation_period
        if now - self.last_rotation_time > period:
            old_style_idx = self.current_style_index
            old_pattern_idx = self.current_pattern_index
            available_styles = [
                i for i in range(len(self.STYLE_POOL)) if i != old_style_idx
            ]
            weights = [self.STYLE_POOL[i].weight for i in available_styles]
            total_weight = sum(weights)
            if total_weight > 0:
                r = random.uniform(0, total_weight)
                cumulative = 0.0
                for i, idx in enumerate(available_styles):
                    cumulative += weights[i]
                    if r <= cumulative:
                        self.current_style_index = idx
                        break
            else:
                self.current_style_index = random.choice(available_styles)
            available_patterns = [
                i
                for i in range(len(self.PATTERN_POOL))
                if i != old_pattern_idx
            ]
            pattern_weights = [
                self.PATTERN_POOL[i].weight for i in available_patterns
            ]
            total_pattern_weight = sum(pattern_weights)
            if total_pattern_weight > 0:
                r = random.uniform(0, total_pattern_weight)
                cumulative = 0.0
                for i, idx in enumerate(available_patterns):
                    cumulative += pattern_weights[i]
                    if r <= cumulative:
                        self.current_pattern_index = idx
                        break
            else:
                self.current_pattern_index = random.choice(available_patterns)
            self.last_rotation_time = now
            self._save_state()
            logger.info(
                f"[{self.channel_id}] 风格轮转: {self.STYLE_POOL[old_style_idx].name} -> {self.current_style_name}"
            )

    async def _generate_anti_repetition_prompt(self) -> str:
        try:
            if not self._recent_bot_phrases:
                return ""
            forbidden_phrases = []
            for msg in self._recent_bot_phrases[-10:]:
                if len(msg) > 5:
                    forbidden_phrases.append(msg[:4])
                    if len(msg) > 10:
                        forbidden_phrases.append(msg[-4:])
            forbidden_phrases = list(set(forbidden_phrases))[:15]
            if not forbidden_phrases:
                return ""
            forbidden_list = json.dumps(forbidden_phrases, ensure_ascii=False)
            prompt_lines = [
                "【即时反重复通过协议】",
                "- 检测到近期已使用过以下表达，本次回复**严禁**再次使用相同的开头或结尾：",
                f"  禁止列表: {forbidden_list}",
                "- 请强制切换句式结构，尝试全新的表达方式。",
            ]
            return "\n".join(prompt_lines)
        except Exception as e:
            logger.debug(f"反重复检测失败: {e}")
            return ""

    def record_bot_message(self, message: str) -> None:
        if not message or len(message) < 3:
            return
        self._recent_bot_phrases.append(message)
        if len(self._recent_bot_phrases) > self._max_recent_phrases:
            self._recent_bot_phrases = self._recent_bot_phrases[
                -self._max_recent_phrases:
            ]

    def force_rotate_style(self) -> str:
        old_style = self.current_style_name
        available = [
            i
            for i in range(len(self.STYLE_POOL))
            if i != self.current_style_index
        ]
        self.current_style_index = random.choice(available)
        available_patterns = [
            i
            for i in range(len(self.PATTERN_POOL))
            if i != self.current_pattern_index
        ]
        self.current_pattern_index = random.choice(available_patterns)
        self.last_rotation_time = time.time()
        self._save_state()
        logger.info(
            f"[{self.channel_id}] 强制风格轮转: {old_style} -> {self.current_style_name}"
        )
        return self.current_style_name

    def set_style(self, style_name: str) -> bool:
        for i, style in enumerate(self.STYLE_POOL):
            if style.name.lower() == style_name.lower():
                self.current_style_index = i
                self.last_rotation_time = time.time()
                self._save_state()
                logger.info(f"[{self.channel_id}] 设置风格: {style_name}")
                return True
        return False

    def set_pattern(self, pattern_name: str) -> bool:
        for i, pattern in enumerate(self.PATTERN_POOL):
            if pattern.name.lower() == pattern_name.lower():
                self.current_pattern_index = i
                self._save_state()
                logger.info(f"[{self.channel_id}] 设置模式: {pattern_name}")
                return True
        return False

    def get_available_styles(self) -> List[Dict]:
        return [
            {"name": s.name, "desc": s.desc, "weight": s.weight}
            for s in self.STYLE_POOL
        ]

    def get_available_patterns(self) -> List[Dict]:
        return [
            {"name": p.name, "desc": p.desc, "weight": p.weight}
            for p in self.PATTERN_POOL
        ]

    def set_rotation_period(self, seconds: int) -> None:
        self._rotation_base_period = max(60, seconds)

    def _get_state_key(self) -> str:
        return f"variety_state_{self.channel_id}"

    def _save_state(self) -> None:
        try:
            from src.common.database.database_model import PersistenceSlot

            data = {
                "style_idx": self.current_style_index,
                "pattern_idx": self.current_pattern_index,
                "last_rotation": self.last_rotation_time,
                "recent_phrases": self._recent_bot_phrases[-10:],
            }
            slot, _ = PersistenceSlot.get_or_create(
                slot_key=self._get_state_key(),
                defaults={
                    "slot_value": json.dumps(data, ensure_ascii=False),
                    "birth_ts": time.time(),
                    "modify_ts": time.time(),
                    "ttl_days": 30,
                },
            )
            slot.slot_value = json.dumps(data, ensure_ascii=False)
            slot.modify_ts = time.time()
            slot.save()
        except Exception as e:
            logger.debug(f"保存状态失败: {e}")

    def _load_state(self) -> None:
        try:
            from src.common.database.database_model import PersistenceSlot

            slot = PersistenceSlot.get_or_none(
                PersistenceSlot.slot_key == self._get_state_key()
            )
            if slot and slot.slot_value:
                data = json.loads(slot.slot_value)
                self.current_style_index = data.get("style_idx", 0)
                self.current_pattern_index = data.get("pattern_idx", 0)
                self.last_rotation_time = data.get("last_rotation", 0.0)
                self._recent_bot_phrases = data.get("recent_phrases", [])
        except Exception as e:
            logger.debug(f"加载状态失败: {e}")

    def get_statistics(self) -> Dict:
        return {
            "channel_id": self.channel_id,
            "current_style": self.current_style_name,
            "current_pattern": self.current_pattern_name,
            "last_rotation": self.last_rotation_time,
            "rotation_period": self._rotation_base_period,
            "recent_phrases_count": len(self._recent_bot_phrases),
        }


_controllers: Dict[str, VarietyController] = {}


def get_variety_controller(channel_id: str) -> VarietyController:
    if channel_id not in _controllers:
        _controllers[channel_id] = VarietyController(channel_id)
    return _controllers[channel_id]


def reset_variety_controller(channel_id: str) -> None:
    if channel_id in _controllers:
        del _controllers[channel_id]


def get_all_controllers() -> Dict[str, VarietyController]:
    return _controllers.copy()


async def get_diversity_instruction(
    channel_id: str, global_rage: float = 0.0
) -> Tuple[str, float]:
    controller = get_variety_controller(channel_id)
    return await controller.get_diversity_instruction(global_rage)


def record_bot_message(channel_id: str, message: str) -> None:
    controller = get_variety_controller(channel_id)
    controller.record_bot_message(message)

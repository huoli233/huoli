import time
import json
import random
import traceback
from typing import Dict, List, Any, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger("variety_controller")


class VarietyController:
    STYLE_POOL = [
        {"name": "standard", "desc": "自然、随性、不做作"},
        {"name": "witty", "desc": "幽默风趣，喜欢玩梗，语气轻快"},
        {"name": "spicy", "desc": "略带毒舌，一针见血，不留情面"},
        {"name": "gentle", "desc": "温柔体贴，如沐春风，治愈系"},
        {"name": "logical", "desc": "理性客观，逻辑严密，像个理科生"},
        {"name": "impatient", "desc": "略显不耐烦，说话简短有力，酷酷的"},
    ]
    PATTERN_POOL = [
        "直接回答型：不绕弯子，直击要点",
        "反问引导型：通过反问引发对方思考",
        "情绪共鸣型：先认同情绪，再给建议",
        "调侃吐槽型：对现状进行幽默的解构",
        "极简风格型：能用一个字绝不用两个字",
    ]

    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.current_style_index = 0
        self.last_rotation_time = 0
        self.current_pattern = self.PATTERN_POOL[0]
        self._load_state()

    def get_dynamic_temperature(self, global_rage: float = 0.0) -> float:
        base_temp = 0.75
        rage_mod = min(0.4, (global_rage / 50.0) * 0.1)
        noise = random.uniform(-0.05, 0.05)
        final_temp = base_temp + rage_mod + noise
        return max(0.2, min(1.4, final_temp))

    async def get_diversity_instruction(self, global_rage: float, user_id: str = "") -> Tuple[str, float]:
        self._rotate_style_if_needed(global_rage)
        current_style = self.STYLE_POOL[self.current_style_index]
        current_style = self._filter_style_by_relationship(current_style, user_id)
        anti_rep = await self._generate_anti_repetition_prompt()
        temp = self.get_dynamic_temperature(global_rage)
        instruction = f"""【多态性控制系统】
- 当前人格面具：{current_style['name']} ({current_style['desc']})
- 推荐回复模式：{self.current_pattern}
- 动态熵值调节：已激活 (当前系统表现出较低的确定性，请尝试更多样化的表达)

{anti_rep}"""
        return instruction.strip(), temp

    def _filter_style_by_relationship(self, style: Dict[str, str], user_id: str) -> Dict[str, str]:
        aggressive_styles = {"spicy", "impatient"}
        if not user_id:
            return style
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(self.stream_id)
            state = tracker.get_user_state(user_id, create_if_missing=False)
            if not state:
                if style["name"] in aggressive_styles:
                    safe_styles = [s for s in self.STYLE_POOL if s["name"] not in aggressive_styles]
                    return random.choice(safe_styles) if safe_styles else self.STYLE_POOL[0]
                return style
            if style["name"] in aggressive_styles:
                if state.affection < 30 or state.relationship in ("陌生人", "冷淡", "stranger"):
                    safe_styles = [s for s in self.STYLE_POOL if s["name"] not in aggressive_styles]
                    return random.choice(safe_styles) if safe_styles else self.STYLE_POOL[0]
        except Exception:
            if style["name"] in aggressive_styles:
                return self.STYLE_POOL[0]
        return style

    def _rotate_style_if_needed(self, global_rage: float):
        now = time.time()
        period = 2400
        if global_rage > 50:
            period = 600
        if now - self.last_rotation_time > period:
            old_idx = self.current_style_index
            available = [i for i in range(len(self.STYLE_POOL)) if i != old_idx]
            self.current_style_index = random.choice(available)
            self.current_pattern = random.choice(self.PATTERN_POOL)
            self.last_rotation_time = now
            self._save_state()
            logger.info(f"[{self.stream_id[:8]}] 风格已轮转 -> {self.STYLE_POOL[self.current_style_index]['name']}")

    async def _generate_anti_repetition_prompt(self) -> str:
        try:
            from src.api import message_api
            recent_msgs = message_api.get_messages_by_count_in_chat(
                chat_id=self.stream_id, count=20, filter_mai=False
            )
            bot_msgs = []
            for msg in recent_msgs:
                ui = getattr(msg, 'user_info', None)
                if ui and str(ui.user_id) == "0":
                    content = getattr(msg, 'processed_plain_text', '') or ''
                    if content and len(content) > 3:
                        bot_msgs.append(content)
                if len(bot_msgs) >= 5:
                    break
            if not bot_msgs:
                return ""
            forbidden_phrases = []
            for msg in bot_msgs:
                if len(msg) > 5:
                    forbidden_phrases.append(msg[:4])
                    if len(msg) > 10:
                        forbidden_phrases.append(msg[-4:])
            forbidden_phrases = list(set(forbidden_phrases))
            forbidden_list = json.dumps(forbidden_phrases, ensure_ascii=False)
            prompt_lines = [
                "【即时反重复通过协议】",
                "- 检测到近期已使用过以下表达，本次回复**严禁**再次使用相同的开头或结尾："
            ]
            prompt_lines.append(f"  禁止列表: {forbidden_list}")
            prompt_lines.append("- 请强制切换句式结构，尝试全新的表达方式。")
            return "\n".join(prompt_lines)
        except Exception as e:
            logger.debug(f"反重复检测失败: {e}")
            return ""

    def _get_state_key(self) -> str:
        return f"variety_state_{self.stream_id}"

    def _save_state(self):
        try:
            from src.chat.heart_flow.persistent_state_db import get_persistent_state_db
            db = get_persistent_state_db()
            data = {
                "style_index": self.current_style_index,
                "pattern": self.current_pattern,
                "rotation_time": self.last_rotation_time,
            }
            db.save_engagement_state(self._get_state_key(), data)
        except Exception:
            pass

    def _load_state(self):
        try:
            from src.chat.heart_flow.persistent_state_db import get_persistent_state_db
            db = get_persistent_state_db()
            data = db.load_engagement_state(self._get_state_key())
            if data:
                self.current_style_index = data.get("style_index", 0)
                self.current_pattern = data.get("pattern", self.PATTERN_POOL[0])
                self.last_rotation_time = data.get("rotation_time", 0)
        except Exception:
            pass

    def get_current_style(self) -> Dict[str, str]:
        return self.STYLE_POOL[self.current_style_index]


_variety_controllers: Dict[str, VarietyController] = {}


def get_variety_controller(stream_id: str) -> VarietyController:
    if stream_id not in _variety_controllers:
        _variety_controllers[stream_id] = VarietyController(stream_id)
    return _variety_controllers[stream_id]

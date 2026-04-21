import time
import random
import asyncio
from typing import Optional
from src.common.logger import get_logger
from src.config.prompt_loader import get_prompt, PromptCategory

logger = get_logger("rest_skill")


class RestDecisionSkill:

    @staticmethod
    def evaluate(chat_value: float, brain_power: float,
                 consecutive_replies: int, mood: str = "") -> int:
        now_hour = time.localtime().tm_hour
        is_late_night = 2 <= now_hour < 6
        is_evening = 22 <= now_hour or now_hour < 2
        if chat_value > 50 and consecutive_replies < 3:
            return 0
        if chat_value > 20:
            base_min = 30
            base_max = 90
        elif chat_value > 0:
            base_min = 60
            base_max = 180
        else:
            base_min = 90
            base_max = 240
        if brain_power < 10:
            base_min = max(base_min, 120)
            base_max = max(base_max, 300)
        if consecutive_replies >= 6:
            base_min = max(base_min, 90)
            base_max = max(base_max, 240)
        if is_late_night:
            base_min = max(base_min, 120)
            base_max = max(base_max, 300)
        elif is_evening:
            base_min = int(base_min * 1.2)
            base_max = int(base_max * 1.2)
        if mood in ("无聊", "疲惫", "烦躁", "tired", "bored"):
            base_min = int(base_min * 1.2)
            base_max = int(base_max * 1.2)
        rest_sec = random.randint(base_min, base_max)
        rest_sec = max(30, min(300, rest_sec))
        logger.info(
            f"🛏️ [RestSkill] 规则预筛: 休息{rest_sec}s | "
            f"聊天值={chat_value:.0f} | 脑力={brain_power:.0f} | "
            f"连续回复={consecutive_replies} | 时段={'深夜' if is_late_night else '晚间' if is_evening else '白天'}"
        )
        return rest_sec

    @staticmethod
    async def evaluate_with_model(
        chat_value: float,
        brain_power: float,
        consecutive_replies: int,
        mood: str = "",
        thought: str = "",
        dialogue_summary: str = "",
    ) -> int:
        rule_rest = RestDecisionSkill.evaluate(chat_value, brain_power, consecutive_replies, mood)
        if rule_rest == 0:
            return 0
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            now_hour = time.localtime().tm_hour
            time_desc = "深夜" if 2 <= now_hour < 6 else "晚间" if (22 <= now_hour or now_hour < 2) else "白天"
            prompt = get_prompt(
                PromptCategory.HEARTFLOW, "skills", "rest_decision.template",
                chat_value=f"{chat_value:.0f}", brain_power=f"{brain_power:.0f}",
                consecutive_replies=consecutive_replies,
                mood=mood or '一般', time_desc=time_desc, now_hour=now_hour,
                thought=thought[:100] if thought else '无',
                dialogue_summary=dialogue_summary[:100] if dialogue_summary else '无',
                rule_rest=rule_rest, rule_rest_max=min(rule_rest * 2, 300),
            )
            request = LLMRequest(model_config.lightweight, request_type="rest_decision")
            response, _ = await request.generate_response_async(prompt)
            if response:
                try:
                    raw_val = int(response.strip().split()[0])
                    if raw_val < 10:
                        raw_val = raw_val * 60
                    rest_sec = max(30, min(300, raw_val))
                    logger.info(f"🛏️ [RestSkill] 模型决定: 休息{rest_sec}s")
                    return rest_sec
                except (ValueError, IndexError):
                    pass
        except Exception as e:
            logger.debug(f"[RestSkill] 模型调用失败: {e}")
        return rule_rest

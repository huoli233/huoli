import random
import time
from typing import Optional, Tuple, Dict, Any
from src.common.logger import get_logger

logger = get_logger("sticker_skill")


class StickerDecisionSkill:
    def __init__(self):
        self._last_sticker_time: float = 0.0
        self._sticker_cooldown: float = 120.0
        self._daily_sticker_count: int = 0
        self._daily_sticker_date: str = ""
        self._max_daily_stickers: int = 20

    async def should_send_sticker(
        self,
        dialogue_context: str,
        current_mood: str,
        scene_type: str,
        user_id: str,
        channel_id: str,
        chat_value: float = 100.0,
    ) -> Tuple[bool, str]:
        now = time.time()
        if now - self._last_sticker_time < self._sticker_cooldown:
            return False, "冷却中"
        today = time.strftime("%Y-%m-%d")
        if self._daily_sticker_date != today:
            self._daily_sticker_date = today
            self._daily_sticker_count = 0
        if self._daily_sticker_count >= self._max_daily_stickers:
            return False, "今日表情包额度用完"
        if chat_value < 20:
            return False, "聊天值不足"
        try:
            from src.chat.heart_flow.skills.scoring_system import (
                get_scoring_system, DecisionType, ScoringFactors, ScoreLevel
            )
            scoring = get_scoring_system()
            context_relevance = await self._analyze_sticker_timing(dialogue_context, current_mood)
            affection = 0.0
            trust = 0.0
            annoyance = 0.0
            try:
                from src.modules.modcore.psychological_core import get_psychological_core
                psy = get_psychological_core()
                if psy:
                    state = psy.get_user_psychological_state(channel_id, user_id)
                    affection = state.get("favor", 0.0)
                    trust = state.get("trust", 0.0)
                    annoyance = state.get("annoyance", 0.0)
            except Exception:
                pass
            factors = ScoringFactors(
                favor=affection, trust=trust, annoyance=annoyance,
                scene_type=scene_type, context_relevance=context_relevance,
                timing_score=context_relevance,
                dialogue_length=len(dialogue_context.split('\n')),
                user_activity=0.7, chat_value=chat_value,
            )
            result = scoring.calculate_score(DecisionType.SEND_STICKER, factors)
            logger.info(f"[表情包决策] 评分={result.total_score:.1f} 等级={result.level.value} 理由={result.reasoning}")
            if result.level in (ScoreLevel.HIGHLY_RECOMMENDED, ScoreLevel.RECOMMENDED):
                self._last_sticker_time = now
                self._daily_sticker_count += 1
                return True, result.reasoning
            return False, result.reasoning
        except Exception as e:
            logger.error(f"[表情包决策] 异常: {e}")
            return False, f"决策异常: {e}"

    async def select_sticker(
        self,
        mood: str,
        dialogue_context: str,
    ) -> Optional[Tuple[str, str, str]]:
        try:
            from src.chat.emoji_system.emoji_manager import StickerVault
            vault = StickerVault()
            if not vault._initialized:
                await vault.load_registry()
            result = await vault.find_sticker_by_mood(mood)
            if result:
                logger.info(f"[表情包选择] 找到: {result[2]} → {result[1]}")
                return result
            result = await vault.get_random_sticker()
            if result:
                logger.info(f"[表情包选择] 随机选择: {result[1]}")
                return result
            return None
        except Exception as e:
            logger.error(f"[表情包选择] 失败: {e}")
            return None

    async def collect_sticker_from_message(self, image_url: str) -> bool:
        try:
            from src.chat.emoji_system.emoji_manager import StickerVault
            vault = StickerVault()
            if not vault._initialized:
                await vault.load_registry()
            success = await vault.enroll_sticker_from_url(image_url)
            if success:
                logger.info(f"[表情包收集] 成功收集: {image_url[:50]}")
            return success
        except Exception as e:
            logger.error(f"[表情包收集] 失败: {e}")
            return False

    async def _analyze_sticker_timing(self, dialogue_context: str, mood: str) -> float:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "heart_flow",
                "sticker_timing.template",
                dialogue_context=dialogue_context[-200:],
                mood=mood
            )
            request = LLMRequest(model_config.lightweight, request_type="sticker_timing")
            response, _ = await request.generate_response_async(prompt)
            if response:
                try:
                    score = float(response.strip())
                    return max(0.0, min(1.0, score))
                except ValueError:
                    return 0.5
            return 0.5
        except Exception:
            return 0.5


_sticker_skill_instance: Optional[StickerDecisionSkill] = None


def get_sticker_skill() -> StickerDecisionSkill:
    global _sticker_skill_instance
    if _sticker_skill_instance is None:
        _sticker_skill_instance = StickerDecisionSkill()
    return _sticker_skill_instance

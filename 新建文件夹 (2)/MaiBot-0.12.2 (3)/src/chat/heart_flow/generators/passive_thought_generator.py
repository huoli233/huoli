from typing import Dict, Optional, List
from dataclasses import dataclass
from src.common.logger import get_logger
from src.config.prompt_loader import get_prompt, PromptCategory

logger = get_logger("passive_thought")


@dataclass
class PassiveThought:
    thought: str
    mood: str
    confidence: float
    thought_type: str = "passive"


class PassiveThoughtGenerator:
    def __init__(self):
        self._llm_bridge = None

    def set_llm_bridge(self, llm_bridge):
        self._llm_bridge = llm_bridge

    async def generate_thought(
        self,
        user_message: str,
        user_id: str,
        emotion_state: Dict,
        recent_messages: List = None,
        context: Dict = None
    ) -> PassiveThought:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            prompt = self._build_thought_prompt(
                user_message, user_id, emotion_state, recent_messages, context
            )
            request = LLMRequest(model_config.focus_chat, request_type="passive_thought")
            response_text, _ = await request.generate_response_async(prompt)
            if not response_text:
                logger.warning("[内心想法-普通] LLM返回空结果，使用默认想法")
                return self._get_default_thought(user_message, emotion_state)
            thought_text = response_text.strip()
            mood = self._extract_mood(emotion_state)
            confidence = 0.8
            logger.info(f"[内心想法-普通] {thought_text[:50]}...")
            return PassiveThought(
                thought=thought_text,
                mood=mood,
                confidence=confidence,
                thought_type="passive"
            )
        except Exception as e:
            logger.error(f"[内心想法-普通] 生成失败: {e}")
            return self._get_default_thought(user_message, emotion_state)

    def _build_thought_prompt(
        self,
        user_message: str,
        user_id: str,
        emotion_state: Dict,
        recent_messages: List,
        context: Dict
    ) -> str:
        affection = emotion_state.get("favor", emotion_state.get("affection", 0))
        annoyance = emotion_state.get("annoyance", 0)
        avg_mood = emotion_state.get("average_mood_score", 0.5)
        energy = emotion_state.get("energy", 50)
        recent_context = ""
        if recent_messages:
            recent_context = "\n".join([
                f"{msg.get('sender', '未知')}: {msg.get('content', '')[:50]}"
                for msg in recent_messages[-5:]
            ])
        
        prompt = get_prompt(
            PromptCategory.HEARTFLOW, "passive_thought", "thought_prompt.template",
            user_message=user_message,
            affection=affection,
            annoyance=annoyance,
            avg_mood=avg_mood,
            energy=energy,
            recent_context=recent_context if recent_context else "（无）"
        )
        return prompt

    def _extract_mood(self, emotion_state: Dict) -> str:
        avg_mood = emotion_state.get("average_mood_score", 0.5)
        if avg_mood >= 0.7:
            return get_prompt(PromptCategory.HEARTFLOW, "passive_thought", "mood.happy")
        elif avg_mood >= 0.4:
            return get_prompt(PromptCategory.HEARTFLOW, "passive_thought", "mood.calm")
        elif avg_mood >= 0.2:
            return get_prompt(PromptCategory.HEARTFLOW, "passive_thought", "mood.low")
        return get_prompt(PromptCategory.HEARTFLOW, "passive_thought", "mood.bad")

    def _get_default_thought(self, user_message: str, emotion_state: Dict) -> PassiveThought:
        affection = emotion_state.get("favor", emotion_state.get("affection", 0))
        if affection >= 50:
            thought = get_prompt(PromptCategory.HEARTFLOW, "passive_thought", "default_thought.high_affection")
        elif affection >= 0:
            thought = get_prompt(PromptCategory.HEARTFLOW, "passive_thought", "default_thought.neutral_affection")
        else:
            thought = get_prompt(PromptCategory.HEARTFLOW, "passive_thought", "default_thought.low_affection")
        mood = self._extract_mood(emotion_state)
        return PassiveThought(
            thought=thought,
            mood=mood,
            confidence=0.5,
            thought_type="passive"
        )


_passive_thought_gen: Optional[PassiveThoughtGenerator] = None


def get_passive_thought_generator() -> PassiveThoughtGenerator:
    global _passive_thought_gen
    if _passive_thought_gen is None:
        _passive_thought_gen = PassiveThoughtGenerator()
    return _passive_thought_gen

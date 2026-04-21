from typing import Dict, Any, Optional, Tuple
from src.common.logger import get_logger
from src.modules.modcore.decision_engine.scoring_system import (
    get_scoring_system,
    DecisionType,
    ScoringFactors,
    ScoreLevel
)

logger = get_logger("group_decision")


class GroupReplyDecisionEngine:
    def __init__(self):
        self._scoring_system = get_scoring_system()

    async def decide_group_sticker(
        self,
        dialogue_context: str,
        current_mood: str,
        user_id: str,
        stream_id: str
    ) -> Tuple[bool, Optional[str], Optional[str], str]:
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(stream_id)
            state = tracker.get_user_state(user_id, create_if_missing=False)
            affection = state.affection if state else 0
            trust = state.trust_score if state else 0
            annoyance = state.annoyance if state else 0
            context_relevance = self._estimate_context_relevance(dialogue_context)
            timing_score = self._estimate_timing(dialogue_context)
            dialogue_length = len(dialogue_context.split('\n'))
            factors = ScoringFactors(
                favor=affection, trust=trust, annoyance=annoyance,
                scene_type="group", context_relevance=context_relevance,
                timing_score=timing_score, risk_level=0.1,
                dialogue_length=dialogue_length, user_activity=0.8
            )
            decision_score = self._scoring_system.calculate_score(DecisionType.SEND_STICKER, factors)
            logger.debug(f"[群聊表情包] 评分={decision_score.total_score:.1f} 等级={decision_score.level.value}")
            if decision_score.level in [ScoreLevel.HIGHLY_RECOMMENDED, ScoreLevel.RECOMMENDED]:
                sticker_path, caption, tag = await self._select_group_sticker(dialogue_context, current_mood)
                if sticker_path:
                    return True, sticker_path, caption, decision_score.reasoning
                return False, None, None, "未找到合适的表情包"
            return False, None, None, decision_score.reasoning
        except Exception as e:
            logger.error(f"[群聊表情包] 决策失败: {e}")
            return False, None, None, f"决策异常: {e}"

    def _estimate_context_relevance(self, dialogue_context: str) -> float:
        if not dialogue_context:
            return 0.5
        lines = dialogue_context.strip().split('\n')
        if len(lines) > 5:
            return 0.7
        elif len(lines) > 2:
            return 0.6
        return 0.5

    def _estimate_timing(self, dialogue_context: str) -> float:
        if not dialogue_context:
            return 0.5
        lines = dialogue_context.strip().split('\n')
        if len(lines) > 3:
            return 0.7
        return 0.6

    async def _select_group_sticker(
        self, dialogue_context: str, current_mood: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        try:
            from src.chat.emoji_system.emoji_manager import get_sticker_selector
            selector = get_sticker_selector()
            if not selector:
                return None, None, None
            result = await selector.select_with_llm_decision(dialogue_context, current_mood, "group")
            if result:
                sticker_path, caption, tag, is_temp = result
                return sticker_path, caption, tag
            return None, None, None
        except Exception as e:
            logger.error(f"[群聊表情包] 选择失败: {e}")
            return None, None, None


_group_decision_instance: Optional[GroupReplyDecisionEngine] = None


def get_group_decision() -> GroupReplyDecisionEngine:
    global _group_decision_instance
    if _group_decision_instance is None:
        _group_decision_instance = GroupReplyDecisionEngine()
    return _group_decision_instance

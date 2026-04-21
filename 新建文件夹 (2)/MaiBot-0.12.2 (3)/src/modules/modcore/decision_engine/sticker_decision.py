import re
from typing import Dict, Any, Optional, Tuple
from src.common.logger import get_logger
from src.modules.modcore.decision_engine.scoring_system import (
    get_scoring_system, DecisionType, ScoringFactors, ScoreLevel,
)

logger = get_logger("sticker_decision")

_POSITIVE_KEYWORDS = {
    "哈哈", "笑", "开心", "好玩", "有趣", "搞笑", "可爱", "喜欢", "爱",
    "棒", "厉害", "赞", "666", "牛", "太好了", "好耶", "恭喜", "庆祝",
    "嘻嘻", "嘿嘿", "呜呜", "emmm", "啊啊", "哇", "nice", "cool",
}
_NEGATIVE_KEYWORDS = {
    "死", "去世", "丧", "抑郁", "悲伤", "难过", "哭", "痛苦", "绝望",
    "严肃", "认真", "讨论", "分析", "问题", "错误", "bug", "故障",
    "对不起", "抱歉", "遗憾", "失望",
}
_EMOJI_PATTERN = re.compile(r'[😀-😻🤣🥰😂😭💀👀🎉✨❤️💕]')


class ModelDrivenStickerDecision:
    def __init__(self):
        self._scoring_system = get_scoring_system()

    async def decide_sticker_send(
        self, dialogue_context: str, current_mood: str,
        scene_type: str, user_id: str, stream_id: str,
    ) -> Tuple[bool, Optional[str], Optional[str], str]:
        try:
            from src.chat.emoji_system.emoji_manager import get_emoji_manager
            emoji_mgr = get_emoji_manager()
            if not emoji_mgr.emoji_objects:
                return False, None, None, "内存中无可用表情包"
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(stream_id)
            emotion_state = tracker.get_user_emotion(user_id) if tracker else {}
            affection = emotion_state.get("affection", 0)
            trust = emotion_state.get("trust", 0)
            annoyance = emotion_state.get("annoyance", 0)
            context_relevance = self._analyze_context_relevance(dialogue_context)
            timing_score = self._analyze_timing(dialogue_context, current_mood)
            dialogue_length = len(dialogue_context.split('\n'))
            factors = ScoringFactors(
                favor=affection, trust=trust, annoyance=annoyance,
                scene_type=scene_type, context_relevance=context_relevance,
                timing_score=timing_score, risk_level=0.0,
                dialogue_length=dialogue_length, user_activity=0.7,
            )
            decision_score = self._scoring_system.calculate_score(DecisionType.SEND_STICKER, factors)
            logger.info(f"[表情包决策] 评分: {decision_score.total_score:.1f} 等级: {decision_score.level.value}")
            logger.info(f"[表情包决策] 理由: {decision_score.reasoning}")
            if decision_score.level in [ScoreLevel.HIGHLY_RECOMMENDED, ScoreLevel.RECOMMENDED]:
                sticker_path, caption, tag = await self._select_sticker_with_llm(
                    dialogue_context, current_mood, scene_type
                )
                if sticker_path:
                    return True, sticker_path, caption, decision_score.reasoning
                else:
                    return False, None, None, "未找到合适的表情包"
            else:
                return False, None, None, decision_score.reasoning
        except Exception as e:
            logger.error(f"[表情包决策] 决策失败: {e}")
            return False, None, None, f"决策异常: {e}"

    def _analyze_context_relevance(self, dialogue_context: str) -> float:
        text = dialogue_context[-400:].lower()
        positive_count = sum(1 for kw in _POSITIVE_KEYWORDS if kw in text)
        negative_count = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text)
        emoji_count = len(_EMOJI_PATTERN.findall(text))
        exclamation_count = text.count('！') + text.count('!')
        question_count = text.count('？') + text.count('?')
        base_score = 0.5
        base_score += min(positive_count * 0.08, 0.35)
        base_score -= min(negative_count * 0.1, 0.4)
        base_score += min(emoji_count * 0.05, 0.15)
        base_score += min(exclamation_count * 0.03, 0.1)
        base_score -= min(question_count * 0.02, 0.1)
        return max(0.0, min(1.0, base_score))

    def _analyze_timing(self, dialogue_context: str, current_mood: str) -> float:
        lines = dialogue_context.strip().split('\n')
        line_count = len(lines)
        mood_lower = current_mood.lower() if current_mood else ""
        base_score = 0.5
        if line_count >= 6:
            base_score += 0.15
        elif line_count >= 3:
            base_score += 0.05
        elif line_count <= 1:
            base_score -= 0.2
        positive_moods = {"开心", "高兴", "愉快", "兴奋", "期待", "happy", "excited", "joy"}
        negative_moods = {"悲伤", "难过", "愤怒", "生气", "焦虑", "sad", "angry", "anxious"}
        if any(m in mood_lower for m in positive_moods):
            base_score += 0.2
        elif any(m in mood_lower for m in negative_moods):
            base_score -= 0.25
        recent_text = '\n'.join(lines[-3:]) if len(lines) >= 3 else dialogue_context
        if any(kw in recent_text for kw in _POSITIVE_KEYWORDS):
            base_score += 0.1
        return max(0.0, min(1.0, base_score))

    async def _select_sticker_with_llm(
        self, dialogue_context: str, current_mood: str, scene_type: str,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        try:
            from src.chat.emoji_system.emoji_manager import get_emoji_manager
            manager = get_emoji_manager()
            if not manager or not manager.emoji_objects:
                return None, None, None
            result = await manager.get_emoji_for_text(dialogue_context[-200:])
            if result:
                return result.get("path"), result.get("caption"), result.get("tag")
            return None, None, None
        except Exception as e:
            logger.error(f"[表情包决策] 选择表情包失败: {e}")
            return None, None, None


_sticker_decision_instance: Optional[ModelDrivenStickerDecision] = None


def get_sticker_decision() -> ModelDrivenStickerDecision:
    global _sticker_decision_instance
    if _sticker_decision_instance is None:
        _sticker_decision_instance = ModelDrivenStickerDecision()
    return _sticker_decision_instance

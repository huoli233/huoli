import time
import hashlib
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from src.common.logger import get_logger

logger = get_logger("repeat_query_detector")


@dataclass
class ToolQueryRecord:
    tool_type: str
    content_hash: str
    user_id: str
    channel_id: str
    timestamp: float = field(default_factory=time.time)
    prompt_hash: str = ""


class RepeatQueryDetector:
    def __init__(self, repeat_query_window: int = 1800):
        self._repeat_query_window = repeat_query_window
        self._query_history: Dict[str, List[ToolQueryRecord]] = defaultdict(list)

    def record_query(
        self,
        tool_type: str,
        content_hash: str,
        user_id: str,
        channel_id: str,
        prompt_hash: str = ""
    ):
        query_key = f"{tool_type}_{content_hash}_{user_id}_{channel_id}"
        record = ToolQueryRecord(
            tool_type=tool_type,
            content_hash=content_hash,
            user_id=user_id,
            channel_id=channel_id,
            prompt_hash=prompt_hash,
            timestamp=time.time()
        )
        self._query_history[query_key].append(record)
        now = time.time()
        self._query_history[query_key] = [
            r for r in self._query_history[query_key]
            if now - r.timestamp < self._repeat_query_window
        ]

    def get_repeat_count(
        self,
        tool_type: str,
        content_hash: str,
        user_id: str,
        channel_id: str
    ) -> int:
        query_key = f"{tool_type}_{content_hash}_{user_id}_{channel_id}"
        if query_key not in self._query_history:
            return 0
        now = time.time()
        recent_queries = [
            r for r in self._query_history[query_key]
            if now - r.timestamp < self._repeat_query_window
        ]
        return len(recent_queries)

    async def should_reject_repeat_query(
        self,
        tool_type: str,
        content_hash: str,
        user_id: str,
        channel_id: str,
        repeat_count: int,
        tool_description: str = ""
    ) -> Tuple[bool, str]:
        if repeat_count < 2:
            return False, ""
        try:
            emotion_data = {}
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
                tracker = get_emotion_tracker(channel_id)
                if tracker:
                    state = tracker.get_user_state(user_id, create_if_missing=False)
                    if state:
                        emotion_data = {
                            "annoyance": state.annoyance,
                            "affection": state.affection,
                            "trust_score": state.trust_score,
                            "impression": getattr(state, 'impression', ''),
                            "relationship": state.relationship,
                        }
            except Exception as e:
                logger.debug(f"[重复检测] 获取emotion_tracker失败: {e}")
            if not emotion_data:
                try:
                    from src.modules.emotion_state import get_emotion_state_manager
                    manager = get_emotion_state_manager()
                    channel_state = manager.get_channel_state(channel_id)
                    if channel_state:
                        emotion = channel_state._emotion
                        emotion_data = {
                            "patience": emotion.patience,
                            "annoyance": emotion.annoyance,
                            "fatigue": emotion.fatigue,
                        }
                except Exception as e:
                    logger.debug(f"[重复检测] 获取emotion_state失败: {e}")
            emotion_summary = ""
            if emotion_data:
                parts = []
                if "annoyance" in emotion_data:
                    parts.append(f"讨厌度: {emotion_data['annoyance']:.1f}/100")
                if "affection" in emotion_data:
                    parts.append(f"好感度: {emotion_data['affection']:.1f}/100")
                if "patience" in emotion_data:
                    parts.append(f"耐心值: {emotion_data['patience']:.1f}/100")
                if "trust_score" in emotion_data:
                    parts.append(f"信任分: {emotion_data['trust_score']:.1f}/100")
                if "fatigue" in emotion_data:
                    parts.append(f"疲劳值: {emotion_data['fatigue']:.1f}/100")
                emotion_summary = ", ".join(parts) if parts else "情绪状态未知"
            tool_desc = tool_description or tool_type
            from src.config.prompt_loader import get_prompt, PromptCategory
            judgment_prompt = get_prompt(
                PromptCategory.MODULE,
                "repeat_query_detector",
                "repeat_query_judge.template",
                user_id=user_id[:8],
                tool_desc=tool_desc,
                repeat_count=repeat_count,
                emotion_summary=emotion_summary if emotion_summary else "情绪状态平稳，但依然觉得无聊"
            )
            try:
                from src.llm_models.utils_model import LLMRequest
                from src.config.config import model_config
                request = LLMRequest(model_config.lightweight, request_type="repeat_query_judge")
                response_text, _ = await request.generate_response_async(judgment_prompt, max_tokens=300)
                if response_text:
                    response_text = response_text.strip()
                    json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                        should_reject = result.get("should_reject", False)
                        reason = result.get("reason", "")
                        if should_reject and reason:
                            logger.info(f"[重复检测] 用户{user_id[:8]}...重复提问{tool_desc}{repeat_count}次，AI决定拒绝: {reason}")
                            return True, reason
            except Exception as e:
                logger.debug(f"[重复检测] LLM判断失败: {e}")
            return False, ""
        except Exception as e:
            logger.debug(f"[重复检测] 判断过程出错: {e}")
            return False, ""


_global_detector: Optional[RepeatQueryDetector] = None


def get_repeat_query_detector() -> RepeatQueryDetector:
    global _global_detector
    if _global_detector is None:
        _global_detector = RepeatQueryDetector()
    return _global_detector

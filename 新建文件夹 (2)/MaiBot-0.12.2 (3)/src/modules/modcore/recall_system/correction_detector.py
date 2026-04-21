import time
import asyncio
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("recall")


@dataclass
class CorrectionEvent:
    stream_id: str
    bot_msg_id: str = ""
    bot_content: str = ""
    sent_at: float = 0.0
    user_reply: str = ""
    detected_at: float = 0.0
    need_recall: bool = False
    reason: str = ""


class CorrectionDetector:
    HINT_PHRASES = [
        "理解错", "搞错", "不是这个意思", "误会", "错了", "不对",
        "我说的是", "我指的是", "不是那个", "不是说", "不是指", "我的意思是",
        "你理解错了", "不是这样", "我没说", "搞混了", "弄错了",
    ]
    STRONG_PHRASES = [
        "你理解错了", "你说错了", "不是这个意思", "我的意思是",
        "你搞错了", "你弄错了", "你误会了",
    ]

    def __init__(self, history_limit: int = 12, timeout_sec: float = 150.0):
        self._history: Dict[str, List[CorrectionEvent]] = {}
        self._history_limit = history_limit
        self._timeout_sec = timeout_sec
        self._lock = asyncio.Lock()

    def log_bot_send(self, stream_id: str, msg_id: str, content: str):
        if stream_id not in self._history:
            self._history[stream_id] = []
        event = CorrectionEvent(
            stream_id=stream_id, bot_msg_id=msg_id,
            bot_content=content, sent_at=time.time(),
        )
        self._history[stream_id].append(event)
        if len(self._history[stream_id]) > self._history_limit:
            self._history[stream_id] = self._history[stream_id][-self._history_limit:]
        logger.debug(f"记录发送: {stream_id[:8]} - {content[:30]}")

    async def check_correction(self, stream_id: str, user_msg: str) -> Optional[CorrectionEvent]:
        async with self._lock:
            if stream_id not in self._history or not self._history[stream_id]:
                return None
            if not self._has_hint_phrase(user_msg):
                return None
            recent = self._history[stream_id][-1]
            elapsed = time.time() - recent.sent_at
            if elapsed > self._timeout_sec:
                logger.debug(f"超时: {elapsed:.1f}s > {self._timeout_sec}s")
                return None
            if self._is_strong_match(user_msg):
                is_correction = True
            else:
                is_correction = await self._verify_with_llm(recent.bot_content, user_msg)
            if is_correction:
                recent.user_reply = user_msg
                recent.detected_at = time.time()
                recent.need_recall = True
                recent.reason = "用户指出理解偏差"
                logger.info(f"检测到纠正: {user_msg[:30]}")
                return recent
            return None

    def _has_hint_phrase(self, text: str) -> bool:
        return any(phrase in text for phrase in self.HINT_PHRASES)

    def _is_strong_match(self, text: str) -> bool:
        return any(phrase in text for phrase in self.STRONG_PHRASES)

    async def _verify_with_llm(self, bot_msg: str, user_msg: str) -> bool:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "recall_controller",
                "correction_verify.template",
                bot_msg=bot_msg[:100],
                user_msg=user_msg[:100]
            )
            request = LLMRequest(model_config.focus_chat, request_type="correction_verify")
            response_text, _ = await request.generate_response_async(prompt)
            if response_text:
                return 'YES' in response_text.upper()
        except Exception as e:
            logger.warning(f"LLM验证失败: {e}")
        return True

    def within_time_limit(self, event: CorrectionEvent) -> bool:
        elapsed = time.time() - event.sent_at
        return elapsed <= self._timeout_sec

    def clear_stream(self, stream_id: str):
        if stream_id in self._history:
            self._history[stream_id].clear()


_detector: Optional[CorrectionDetector] = None


def get_correction_detector() -> CorrectionDetector:
    global _detector
    if _detector is None:
        _detector = CorrectionDetector()
    return _detector

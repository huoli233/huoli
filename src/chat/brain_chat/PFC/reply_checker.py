import asyncio
from collections import deque
import difflib
import re
import traceback
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.chat.brain_chat.PFC.chat_observer import ChatObserver
from src.chat.brain_chat.runtime_config import brainchat_module_view
from src.chat.brain_chat.PFC.pfc_utils import parse_json_payload
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.llm_models.utils_model import LLMRequest

import src.chat.prompts.catalog  # noqa: F401 注册提示词
from src.chat.utils.prompt_builder import global_prompt_manager

logger = get_logger("回复检查")


class ReplyChecker:
    """回复检查器，负责检查生成的回复是否合适。"""

    _recent_checked_replies: Dict[str, Deque[str]] = {}

    def __init__(self, stream_id: str, private_name: str):
        self.llm = LLMRequest(
            model_set=model_config.model_task_config.utils,
            request_type="reply_check",
        )
        self.bot_name = global_config.bot.nickname
        self.private_name = private_name
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)
        self.bot_account = str(
            getattr(global_config.bot, "qq_account", "") or ""
        )
        self._history_key = f"{stream_id}:{private_name}"
        self._load_config()

    def _load_config(self) -> None:
        config = brainchat_module_view("brain_pfc_reply")
        self.max_retries = int(config.get("check_max_retries", 3))
        self._recent_checked_maxlen = int(
            config.get("checked_history_maxlen", 6)
        )
        self._recent_bot_limit = int(config.get("recent_bot_limit", 4))
        self._duplicate_similarity_threshold = float(
            config.get("duplicate_similarity_threshold", 0.9)
        )
        self._checked_similarity_threshold = float(
            config.get("checked_similarity_threshold", 0.85)
        )
        self._tail_streak_limit = int(config.get("tail_streak_limit", 3))
        self._llm_timeout_seconds = float(
            config.get("check_timeout_seconds", 30.0)
        )

    def _ensure_reply_history(self) -> Deque[str]:
        history = self._recent_checked_replies.get(self._history_key)
        if history is None or history.maxlen != self._recent_checked_maxlen:
            history = deque(history or [], maxlen=self._recent_checked_maxlen)
            self._recent_checked_replies[self._history_key] = history
        return history

    @staticmethod
    def _normalize_reply_text(text: str) -> str:
        lowered = (text or "").strip().lower()
        lowered = re.sub(r"[\s\t\r\n]+", "", lowered)
        lowered = re.sub(r"[，。！？!?,.:;；、~\-\[\](){}'\"`]+", "", lowered)
        return lowered

    def _append_recent_reply(self, text: str) -> None:
        normalized = self._normalize_reply_text(text)
        if normalized:
            self._ensure_reply_history().appendleft(normalized)

    def _extract_recent_bot_messages(
        self, chat_history: List[Dict[str, Any]], limit: Optional[int] = None
    ) -> List[str]:
        limit = self._recent_bot_limit if limit is None else limit
        bot_messages: List[str] = []
        for msg in reversed(chat_history):
            user_id = str(msg.get("user_id", "") or "")
            role = str(msg.get("role", "") or "")
            is_bot_sender = user_id == self.bot_account or role == "assistant"
            if not is_bot_sender:
                continue
            content = (
                msg.get("text", "")
                or msg.get("processed_plain_text", "")
                or msg.get("detailed_plain_text", "")
                or ""
            )
            if content:
                bot_messages.append(content)
            if len(bot_messages) >= limit:
                break
        return bot_messages

    def _count_tail_bot_streak(
        self, chat_history: List[Dict[str, Any]]
    ) -> int:
        streak = 0
        for msg in reversed(chat_history):
            user_id = str(msg.get("user_id", "") or "")
            role = str(msg.get("role", "") or "")
            is_bot_sender = user_id == self.bot_account or role == "assistant"
            if not is_bot_sender:
                break
            streak += 1
        return streak

    def _check_local_duplicate_rule(
        self, reply: str, chat_history: List[Dict[str, Any]]
    ) -> Optional[str]:
        normalized_reply = self._normalize_reply_text(reply)
        if not normalized_reply:
            return "回复为空，无法发送"

        recent_bot_messages = self._extract_recent_bot_messages(chat_history)
        for recent_message in recent_bot_messages:
            normalized_recent = self._normalize_reply_text(recent_message)
            if not normalized_recent:
                continue
            if normalized_recent == normalized_reply:
                return "被逻辑检查拒绝：回复内容与你上一条发言完全相同，可以选择深入话题或先等待。"

            similarity_ratio = difflib.SequenceMatcher(
                None, normalized_reply, normalized_recent
            ).ratio()
            if similarity_ratio >= self._duplicate_similarity_threshold:
                return f"被逻辑检查拒绝：回复内容与你近期发言高度相似（相似度 {
                    similarity_ratio:.2f}），建议换一种推进方式。"

        recent_checked = self._ensure_reply_history()
        for checked in list(recent_checked)[:3]:
            similarity_ratio = difflib.SequenceMatcher(
                None, normalized_reply, checked
            ).ratio()
            if similarity_ratio >= self._checked_similarity_threshold:
                return "被逻辑检查拒绝：这条回复在当前会话里已出现过近似表达，建议避免复读。"

        tail_streak = self._count_tail_bot_streak(chat_history)
        if tail_streak >= self._tail_streak_limit:
            return "被逻辑检查拒绝：你已经连续发送多条消息且对方尚未回应，建议等待或调整目标。"

        return None

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "是"}:
                return True
            if lowered in {"false", "0", "no", "否"}:
                return False
        return default

    async def check(
        self,
        reply: str,
        goal: str,
        chat_history: List[Dict[str, Any]],
        chat_history_text: str,
        retry_count: int = 0,
    ) -> Tuple[bool, str, bool]:
        """检查生成的回复是否合适。"""
        try:
            duplicate_issue = self._check_local_duplicate_rule(
                reply, chat_history
            )
            if duplicate_issue:
                logger.warning(
                    f"[私聊][{self.private_name}]ReplyChecker 本地规则拒绝: {duplicate_issue}"
                )
                return False, duplicate_issue, True
        except Exception as e:
            logger.error(
                f"[私聊][{
                    self.private_name}]检查回复时出错: 类型={
                    type(e)}, 值={e}"
            )
            logger.error(
                f"[私聊][{self.private_name}]{traceback.format_exc()}"
            )

        _tpl = global_prompt_manager.get_prompt("pfc_reply_check")
        if not _tpl:
            logger.warning(f"[私聊][{self.private_name}] pfc_reply_check 模板未注册")
            return True, reply, False
        prompt = _tpl.render(
            goal=goal,
            chat_history_text=chat_history_text,
            reply=reply,
        )

        try:
            content, _ = await asyncio.wait_for(
                self.llm.generate_response_async(
                    prompt,
                    usage_stream_id=self.chat_observer.stream_id,
                ),
                timeout=self._llm_timeout_seconds,
            )
            logger.debug(
                f"[私聊][{self.private_name}]检查回复的原始返回: {content}"
            )
            content = (content or "").strip()
            parsed_payload = parse_json_payload(content, allow_array=False)

            if isinstance(parsed_payload, dict):
                suitable = self._coerce_bool(
                    parsed_payload.get("suitable"), default=False
                )
                reason = str(
                    parsed_payload.get("reason", "未提供原因") or "未提供原因"
                )
                need_replan = self._coerce_bool(
                    parsed_payload.get("need_replan"), default=False
                )
            else:
                safe_text = content.lower()
                suitable = (
                    "不合适" not in safe_text and "违规" not in safe_text
                )
                reason = content[:100] if content else "无法解析响应"
                need_replan = (
                    "重新规划" in safe_text or "目标不适合" in safe_text
                )

            if not isinstance(suitable, bool):
                suitable = (
                    "不合适" not in reason.lower()
                    and "违规" not in reason.lower()
                )

            if not suitable and retry_count < self.max_retries:
                return False, reason, False

            if not suitable and retry_count >= self.max_retries:
                return False, f"多次重试后仍不合适: {reason}", True

            self._append_recent_reply(reply)

            return suitable, reason, need_replan

        except Exception as e:
            logger.error(f"[私聊][{self.private_name}]检查回复时出错: {e}")
            if retry_count >= self.max_retries:
                return False, "多次检查失败，建议重新规划", True
            return False, f"检查过程出错，建议重试: {str(e)}", False

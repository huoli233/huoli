import asyncio
import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger
from src.config.prompt_loader import get_prompt, PromptCategory
from src.chat.heart_flow.models.heart_state import (
    HeartState, HeartStatus, MentalEvent, EventType, ActionIntent
)

logger = get_logger("flow_planner")


class SituationType(Enum):
    NEW_MESSAGE = "new_message"
    REPLY_IN_TIME = "reply_in_time"
    REPLY_LATE = "reply_late"
    TIMEOUT = "timeout"
    PROACTIVE = "proactive"

    def __str__(self) -> str:
        return self.value


@dataclass
class FlowResponse:
    thought: str
    actions: List[ActionIntent]
    expected_reaction: str = ""
    max_wait_seconds: int = 0
    mood: str = ""
    success: bool = True
    error: Optional[str] = None
    situation_type: SituationType = SituationType.NEW_MESSAGE

    def has_reply(self) -> bool:
        return any(a.action_type in ("reply", "respond") for a in self.actions)

    def get_reply_content(self) -> str:
        for action in self.actions:
            if action.action_type in ("reply", "respond"):
                return action.params.get("content", "")
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought,
            "actions": [a.to_dict() for a in self.actions],
            "expected_reaction": self.expected_reaction,
            "max_wait_seconds": self.max_wait_seconds,
            "mood": self.mood, "success": self.success, "error": self.error,
        }


class FlowPlanner:
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self._log_tag = f"[心流规划:{stream_id[:8]}]"

    def determine_situation_type(self, heart_state: HeartState) -> SituationType:
        if heart_state.status == HeartStatus.WAITING:
            if heart_state.waiting_config.max_wait_seconds <= 0:
                return SituationType.NEW_MESSAGE
            if heart_state.waiting_config.is_timeout():
                return SituationType.REPLY_LATE
            return SituationType.REPLY_IN_TIME
        return SituationType.NEW_MESSAGE

    def _truncate_to_limit(self, text: str, min_len: int = 20, max_len: int = 50, hard_max: int = 256) -> str:
        if not text:
            return text
        text = text.strip()
        char_count = len(text)
        if min_len <= char_count <= max_len:
            return text
        if char_count > hard_max:
            for i in range(min(hard_max - 3, len(text) - 1), -1, -1):
                if text[i] in "。！？.?!":
                    truncated = text[:i + 1]
                    if len(truncated) >= min_len:
                        return truncated
            return text[:hard_max - 3] + "..."
        return text

    async def generate_flow_response(
        self, heart_state: HeartState, user_message: str,
        user_name: str, user_id: str,
        conversation_context: List[Dict],
        available_actions: Optional[List[str]] = None,
        extra_context: Optional[Dict] = None,
    ) -> FlowResponse:
        try:
            situation_type = self.determine_situation_type(heart_state)
            prompt = self._build_planner_prompt(
                heart_state=heart_state, user_message=user_message,
                user_name=user_name, situation_type=situation_type,
                conversation_context=conversation_context,
                available_actions=available_actions, extra_context=extra_context,
            )
            raw_response = await self._call_llm(prompt)
            if raw_response == "__SAFETY_VIOLATION__":
                logger.warning(f"{self._log_tag} 检测到安全违规，触发心理防御")
                try:
                    from src.modules.modcore.psychological_core import get_psychological_core
                    psy_core = get_psychological_core()
                    if psy_core:
                        safety_ratings = [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "HIGH"}]
                        await psy_core.handle_safety_violation(
                            user_id=user_id, stream_id=self.stream_id,
                            content=user_message, safety_ratings=safety_ratings
                        )
                except Exception as e:
                    logger.error(f"{self._log_tag} 心理防御触发失败: {e}")
                return self._create_safety_violation_response(situation_type)
            if not raw_response:
                return self._create_fallback_response("LLM调用失败", situation_type)
            result = self._parse_response(raw_response, situation_type)
            logger.info(f"{self._log_tag} 内心: {result.thought[:60]}...")
            if result.has_reply():
                logger.info(f"{self._log_tag} 动作: 回复 - {result.get_reply_content()[:40]}...")
            return result
        except Exception as e:
            logger.error(f"{self._log_tag} 规划失败: {e}")
            return self._create_fallback_response(str(e), SituationType.NEW_MESSAGE)

    async def generate_proactive_response(
        self, heart_state: HeartState, trigger_context: str,
        available_actions: Optional[List[str]] = None,
    ) -> FlowResponse:
        try:
            prompt = self._build_proactive_prompt(
                heart_state=heart_state, trigger_context=trigger_context,
                available_actions=available_actions,
            )
            raw_response = await self._call_llm(prompt)
            if not raw_response:
                return self._create_fallback_response("LLM调用失败", SituationType.PROACTIVE)
            result = self._parse_response(raw_response, SituationType.PROACTIVE)
            logger.info(f"{self._log_tag} 主动想法: {result.thought[:60]}...")
            return result
        except Exception as e:
            logger.error(f"{self._log_tag} 主动规划失败: {e}")
            return self._create_fallback_response(str(e), SituationType.PROACTIVE)

    def _build_planner_prompt(
        self, heart_state: HeartState, user_message: str,
        user_name: str, situation_type: SituationType,
        conversation_context: List[Dict],
        available_actions: Optional[List[str]] = None,
        extra_context: Optional[Dict] = None,
    ) -> str:
        persona_block = self._get_persona_block()
        mental_history = self._format_mental_history(heart_state.get_recent_events(8))
        context_block = self._format_conversation_context(conversation_context[-10:])
        action_list = available_actions or ["reply", "do_nothing", "poke", "emoji"]
        action_block = self._format_available_actions(action_list)
        situation_desc = self._get_situation_description(situation_type, heart_state)
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        emotion_block = ""
        if extra_context:
            if extra_context.get("my_feeling"):
                emotion_block = f"\n【我现在的感受】{extra_context['my_feeling']}"
            if extra_context.get("relationship"):
                emotion_block += f"\n【我与{user_name}的关系】{extra_context['relationship']}"
            if extra_context.get("memory_context"):
                emotion_block += f"\n【相关记忆】{extra_context['memory_context']}"
        prompt = get_prompt(
            PromptCategory.HEARTFLOW, "flow_planner", "reactive_prompt.template",
            persona_block=persona_block,
            current_time=current_time,
            situation_desc=situation_desc,
            emotion_block=emotion_block,
            mental_history=mental_history,
            context_block=context_block,
            user_name=user_name,
            user_message=user_message,
            action_block=action_block
        )
        return prompt

    def _build_proactive_prompt(self, heart_state: HeartState,
                                 trigger_context: str,
                                 available_actions: Optional[List[str]] = None) -> str:
        persona_block = self._get_persona_block()
        mental_history = self._format_mental_history(heart_state.get_recent_events(5))
        action_list = available_actions or ["reply", "do_nothing"]
        action_block = self._format_available_actions(action_list)
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        prompt = get_prompt(
            PromptCategory.HEARTFLOW, "flow_planner", "proactive_prompt.template",
            persona_block=persona_block,
            action_block=action_block,
            current_time=current_time,
            trigger_context=trigger_context,
            mental_history=mental_history
        )
        return prompt

    def _get_persona_block(self) -> str:
        try:
            from src.config.config import global_config
            name = global_config.bot.nickname
            personality = global_config.personality.personality
            if name and personality:
                return get_prompt(PromptCategory.HEARTFLOW, "flow_planner", "persona.full", name=name, personality=personality[:200])
            elif name:
                return get_prompt(PromptCategory.HEARTFLOW, "flow_planner", "persona.name_only", name=name)
            return get_prompt(PromptCategory.HEARTFLOW, "flow_planner", "persona.default")
        except Exception:
            return get_prompt(PromptCategory.HEARTFLOW, "flow_planner", "persona.default")

    def _get_situation_description(self, situation_type: SituationType, heart_state: HeartState) -> str:
        if situation_type == SituationType.NEW_MESSAGE:
            return get_prompt(PromptCategory.HEARTFLOW, "flow_planner", "situation.new_message")
        elif situation_type == SituationType.REPLY_IN_TIME:
            elapsed = heart_state.waiting_config.get_elapsed_minutes()
            expected = heart_state.waiting_config.expected_reaction
            return get_prompt(PromptCategory.HEARTFLOW, "flow_planner", "situation.reply_in_time", elapsed=f"{elapsed:.1f}", expected=expected)
        elif situation_type == SituationType.REPLY_LATE:
            elapsed = heart_state.waiting_config.get_elapsed_minutes()
            expected = heart_state.waiting_config.expected_reaction
            return get_prompt(PromptCategory.HEARTFLOW, "flow_planner", "situation.reply_late", elapsed=f"{elapsed:.1f}", expected=expected)
        elif situation_type == SituationType.TIMEOUT:
            return get_prompt(PromptCategory.HEARTFLOW, "flow_planner", "situation.timeout")
        elif situation_type == SituationType.PROACTIVE:
            return get_prompt(PromptCategory.HEARTFLOW, "flow_planner", "situation.proactive")
        return get_prompt(PromptCategory.HEARTFLOW, "flow_planner", "situation.default")

    def _format_mental_history(self, events: List[MentalEvent]) -> str:
        if not events:
            return "（无历史记录）"
        lines = []
        for event in events[-6:]:
            time_str = event.get_time_str("%H:%M")
            if event.event_type == EventType.USER_MESSAGE:
                lines.append(f"[{time_str}] 收到消息 {event.user_name}: {event.content[:40]}")
            elif event.event_type == EventType.BOT_THINKING:
                lines.append(f"[{time_str}] 我想: {event.thought[:40]}")
            elif event.event_type == EventType.WAITING_UPDATE:
                lines.append(f"[{time_str}] 等待中想: {event.waiting_thought[:40]}")
            elif event.event_type == EventType.WAIT_TIMEOUT:
                lines.append(f"[{time_str}] 等待超时")
        return "\n".join(lines) if lines else "（无历史记录）"

    def _format_conversation_context(self, context: List[Dict]) -> str:
        if not context:
            return "（无上下文）"
        lines = []
        for msg in context[-8:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:60]
            prefix = "我" if role == "assistant" else "用户"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines) if lines else "（无上下文）"

    def _format_available_actions(self, actions: List[str]) -> str:
        action_desc = {
            "reply": "reply - 发送文字回复，需要content参数",
            "do_nothing": "do_nothing - 什么都不做，保持沉默",
            "poke": "poke - 戳一戳对方",
            "emoji": "emoji - 发送表情包，需要keyword参数",
            "wait": "wait - 继续等待，需要seconds参数",
            "analyze_image": "analyze_image - 分析图片内容",
            "search": "search - 搜索信息，需要query参数",
        }
        lines = []
        for action in actions:
            desc = action_desc.get(action, f"{action} - 执行{action}动作")
            lines.append(f"- {desc}")
        return "\n".join(lines)

    async def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            request = LLMRequest(model_config.focus_chat, request_type="flow_planner")
            response_text, _ = await request.generate_response_async(prompt)
            if not response_text:
                return None
            return response_text
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            if "EmptyContent" in error_type or "Safety violation" in error_msg or "HARASSMENT" in error_msg:
                logger.warning(f"{self._log_tag} 检测到安全违规: {error_msg}")
                return "__SAFETY_VIOLATION__"
            logger.error(f"{self._log_tag} LLM调用异常: {e}")
            return None

    def _parse_response(self, raw_response: str, situation_type: SituationType) -> FlowResponse:
        try:
            raw_response = raw_response.strip()
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            if raw_response.startswith("```"):
                raw_response = raw_response[3:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]
            raw_response = raw_response.strip()
            data = json.loads(raw_response)
            thought = data.get("thought", "")
            actions_data = data.get("actions", [])
            actions = []
            for a in actions_data:
                if isinstance(a, dict):
                    actions.append(ActionIntent.from_dict(a))
            if not actions:
                actions = [ActionIntent(action_type="do_nothing")]
            max_wait = data.get("max_wait_seconds", 0)
            try:
                max_wait = int(max_wait)
                max_wait = max(0, min(max_wait, 300))
            except (ValueError, TypeError):
                max_wait = 0
            thought = self._truncate_to_limit(thought, min_len=20, max_len=50, hard_max=256)
            mood = data.get("mood", "")
            mood = self._truncate_to_limit(mood, min_len=5, max_len=15, hard_max=50)
            for action in actions:
                if action.action_type in ("reply", "respond", "tts"):
                    content = action.params.get("content", "")
                    content = self._truncate_to_limit(content, min_len=20, max_len=100, hard_max=256)
                    action.params["content"] = content
            return FlowResponse(
                thought=thought, actions=actions,
                expected_reaction=data.get("expected_reaction", ""),
                max_wait_seconds=max_wait, mood=mood,
                success=True, situation_type=situation_type,
            )
        except json.JSONDecodeError as e:
            logger.warning(f"{self._log_tag} JSON解析失败: {e}")
            return FlowResponse(
                thought=raw_response[:80] if raw_response else "思考中...",
                actions=[ActionIntent(action_type="do_nothing")],
                success=False, error=f"JSON解析失败: {e}",
                situation_type=situation_type,
            )

    def _create_fallback_response(self, error: str, situation_type: SituationType) -> FlowResponse:
        return FlowResponse(
            thought="让我想想怎么回应...",
            actions=[ActionIntent(action_type="do_nothing")],
            success=False, error=error, situation_type=situation_type,
        )

    def _create_safety_violation_response(self, situation_type: SituationType) -> FlowResponse:
        if situation_type == SituationType.REPLY_LATE:
            thought = "...（默默看着你，不知道该说什么）"
        elif situation_type == SituationType.REPLY_IN_TIME:
            thought = "......"
        else:
            thought = "......"
        return FlowResponse(
            thought=thought,
            actions=[ActionIntent(action_type="do_nothing")],
            success=True, situation_type=situation_type,
        )


_flow_planners: Dict[str, FlowPlanner] = {}


def get_flow_planner(stream_id: str) -> FlowPlanner:
    if stream_id not in _flow_planners:
        _flow_planners[stream_id] = FlowPlanner(stream_id)
    return _flow_planners[stream_id]

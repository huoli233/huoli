import json
import re
from typing import Any, Optional, Dict, List
from src.common.logger import get_logger
from src.chat.proactive.models import GeneratorResult, BehaviorModel, ActivityKind
from src.chat.proactive.session_manager import MiaoSession
from src.chat.proactive.topic_extractor import extract_group_topics, TopicInfo

logger = get_logger("unified_mode")


async def generate_unified(
    session: MiaoSession, user_name: str, situation: str = "new_message",
    available_behaviors: Optional[Dict] = None, extra_params: Optional[Dict] = None,
) -> GeneratorResult:
    try:
        prompt = _build_unified_prompt(
            session=session, user_name=user_name, situation=situation,
            available_behaviors=available_behaviors, extra_params=extra_params,
        )
        from src.llm_models.utils_model import LLMRequest
        from src.config.config import model_config
        request = LLMRequest(model_config.focus_chat, request_type="unified_mode")
        response_text, _ = await request.generate_response_async(prompt, max_tokens=800)
        if not response_text:
            logger.error("[Unified] LLM响应为空")
            return GeneratorResult.create_fallback("LLM响应为空")
        logger.debug(f"[Unified] LLM响应: {response_text[:300]}...")
        return _parse_unified_response(response_text)
    except Exception as e:
        logger.error(f"[Unified] 生成失败: {e}")
        return GeneratorResult.create_fallback(str(e))


    from src.config.prompt_loader import get_prompt, PromptCategory
    template = get_prompt(PromptCategory.MODULE, "proactive", "unified_template.template")
    response_format = get_prompt(PromptCategory.MODULE, "proactive", "unified_format.template")
    notes = get_prompt(PromptCategory.MODULE, "proactive", "unified_notes.template")
    
    extra_params = extra_params or {}
    recent_records = session.fetch_recent_records(limit=10)
    history_text = _format_history(recent_records, user_name)
    behaviors_text = _format_behaviors(available_behaviors)
    situation_context = _build_context(session, situation, extra_params)
    try:
        if template:
            prompt_body = template.format(
                user_name=user_name,
                history=history_text,
                context=situation_context,
                behaviors=behaviors_text
            )
        else:
            prompt_body = f"你正在与 {user_name} 对话。\n\n{history_text}\n\n{situation_context}"
    except (KeyError, ValueError):
        prompt_body = f"你正在与 {user_name} 对话。\n\n{history_text}\n\n{situation_context}"
    prompt = f"{prompt_body}\n\n{response_format}\n\n{notes}\n\n请只输出JSON格式的回复："
    return prompt


def _format_history(records: List, user_name: str) -> str:
    if not records:
        return "（这是你们的第一次对话）"
    lines = ["最近的对话："]
    for record in records[-8:]:
        if record.kind == ActivityKind.INCOMING_MSG:
            content = record.text_content[:100]
            if len(record.text_content) > 100:
                content += "..."
            sender = record.sender_name or user_name
            lines.append(f"  {sender}: {content}")
        elif record.kind == ActivityKind.BOT_DECISION:
            for behavior in record.behavior_list:
                if behavior.get("type") in ("miao_reply", "respond"):
                    content = behavior.get("content", "")[:100]
                    if len(behavior.get("content", "")) > 100:
                        content += "..."
                    lines.append(f"  你: {content}")
                    break
        elif record.kind == ActivityKind.AWAIT_SHIFT:
            if record.reflect_thought:
                lines.append(f"  （你的内心：{record.reflect_thought[:50]}...）")
    return "\n".join(lines)


def _format_behaviors(behaviors: Optional[Dict]) -> str:
    if not behaviors:
        return """- miao_reply: 发送文字回复 (参数: content - 回复内容)
- idle: 保持沉默，不做任何事
- nudge_user: 轻戳对方
- send_sticker: 发送表情 (参数: sticker - 表情名称)"""
    lines = []
    for name, info in behaviors.items():
        if isinstance(info, dict):
            desc = info.get("description", "")
            params = info.get("params", [])
            params_str = f" (参数: {', '.join(params)})" if params else ""
            lines.append(f"- {name}: {desc}{params_str}")
        else:
            lines.append(f"- {name}: {info}")
    return "\n".join(lines) if lines else "- miao_reply: 发送文字回复\n- idle: 保持沉默"


def _build_context(session: MiaoSession, situation: str, extra_params: Dict) -> str:
    from src.config.prompt_loader import get_prompt, PromptCategory
    if situation == "new_message":
        return get_prompt(
            PromptCategory.MODULE, "proactive", "unified_context_new_message.template",
            user_message=extra_params.get("user_message", "")
        )
    elif situation == "reply_in_time":
        return get_prompt(
            PromptCategory.MODULE, "proactive", "unified_context_reply_in_time.template",
            elapsed_seconds=f"{session.pending_config.elapsed_seconds():.0f}",
            anticipated=session.pending_config.anticipated_reply
        )
    elif situation == "reply_late":
        return get_prompt(
            PromptCategory.MODULE, "proactive", "unified_context_reply_late.template",
            elapsed_seconds=f"{session.pending_config.elapsed_seconds():.0f}",
            timeout=session.pending_config.timeout_seconds,
            anticipated=session.pending_config.anticipated_reply
        )
    elif situation == "timeout":
        return get_prompt(
            PromptCategory.MODULE, "proactive", "unified_context_timeout.template",
            elapsed_seconds=f"{session.pending_config.elapsed_seconds():.0f}",
            timeout=session.pending_config.timeout_seconds,
            chase_count=extra_params.get("chase_count", 0),
            sequential_timeout=extra_params.get("sequential_timeout_count", 0),
            time_since_user=extra_params.get("time_since_user_reply_str", "未知")
        )
    elif situation == "proactive":
        return get_prompt(
            PromptCategory.MODULE, "proactive", "unified_context_proactive.template",
            silence_duration=extra_params.get("silence_duration", "一段时间"),
            trigger_reason=extra_params.get("trigger_reason", "")
        )
    elif situation == "proactive_direct":
        topic_info = extra_params.get("topic_info")
        topic_context = extra_params.get("topic_context", "")
        unanswered_count = extra_params.get("unanswered_count", 0)
        unanswered_hint = ""
        topic_hint = ""
        
        if unanswered_count > 0:
            unanswered_hint = f"\n这是你第 {unanswered_count + 1} 次尝试联系对方。"
            if unanswered_count == 1:
                unanswered_hint += "\n对方之前没有回复，可以换个话题或方式。"
            elif unanswered_count >= 2:
                unanswered_hint += "\n对方多次没有回复，语气可以更随意一些，不要显得太刻意。"
        if topic_context:
            topic_hint = f"\n\n【群里最近在聊】\n{topic_context}\n你可以基于这些话题自然地加入讨论，或者发表自己的看法。"
        elif isinstance(topic_info, TopicInfo) and topic_info.is_valid():
            if topic_info.keywords:
                kw_str = "、".join(topic_info.keywords[:5])
                topic_hint += f"\n\n群友最近在讨论的关键词：{kw_str}"
            if topic_info.participants:
                topic_hint += f"\n参与讨论的人：{'、'.join(topic_info.participants[:3])}"
        return get_prompt(
            PromptCategory.MODULE, "proactive", "unified_context_proactive_direct.template",
            silence_duration=extra_params.get("silence_duration", "一段时间"),
            unanswered_hint=unanswered_hint,
            topic_hint=topic_hint
        )
    return f"当前情况：{situation}"


def _parse_unified_response(raw_response: str) -> GeneratorResult:
    text = raw_response.strip()
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            result = GeneratorResult.from_dict(data)
            if result.inner_thought:
                logger.info(f"[Unified] {result.inner_thought[:60]}...")
            for behavior in result.behaviors:
                if behavior.action_type in ("miao_reply", "respond"):
                    content = behavior.action_params.get("content", "")
                    if content:
                        logger.info(f"[Unified] {content[:60]}...")
                    break
            behaviors_str = ", ".join(b.action_type for b in result.behaviors)
            logger.debug(f"[Unified] behaviors={behaviors_str}, wait={result.timeout_seconds}s")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"[Unified] JSON解析失败: {e}")
    logger.warning(f"[Unified] 无法解析响应: {text[:200]}...")
    return GeneratorResult.create_fallback("无法解析响应格式")


class UnifiedGenerator:
    def __init__(self, default_timeout: int = 300):
        self.default_timeout = default_timeout
        self._stats = {"total_calls": 0, "successful_calls": 0, "failed_calls": 0}

    async def generate(
        self, session: MiaoSession, user_name: str, situation: str = "new_message",
        available_behaviors: Optional[Dict] = None, extra_params: Optional[Dict] = None,
    ) -> GeneratorResult:
        self._stats["total_calls"] += 1
        result = await generate_unified(
            session=session, user_name=user_name, situation=situation,
            available_behaviors=available_behaviors, extra_params=extra_params,
        )
        if result.inner_thought and "失败" not in result.inner_thought:
            self._stats["successful_calls"] += 1
        else:
            self._stats["failed_calls"] += 1
        return result

    def get_statistics(self) -> Dict:
        return self._stats.copy()


_unified_generator: Optional[UnifiedGenerator] = None


def get_unified_generator() -> UnifiedGenerator:
    global _unified_generator
    if _unified_generator is None:
        _unified_generator = UnifiedGenerator()
    return _unified_generator

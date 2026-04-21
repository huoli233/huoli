import random
import re
from typing import Optional, Dict, Any, List, Tuple
from src.common.logger import get_logger
from src.config.prompt_loader import get_prompt, get_prompt_raw, PromptCategory
from src.chat.proactive.session_manager import MiaoSession
from src.chat.proactive.models import ActivityKind

logger = get_logger("proactive_replyer")


async def generate_reply(plan: Dict, context_info: Dict = None) -> Dict:
    try:
        if not plan or not plan.get("success", False):
            logger.warning("[主动回复] 计划无效或失败")
            return {"success": False, "reason": "invalid_plan"}
        if context_info and context_info.get("is_proactive", False):
            return await _generate_proactive_reply(plan, context_info)
        user_name = plan.get("user_name", "")
        inner_thought = plan.get("inner_thought", "")
        situation = plan.get("situation", "new_message")
        extra_params = plan.get("extra_params", {})
        prompt = _build_replyer_prompt(
            session=None, user_name=user_name, inner_thought=inner_thought,
            situation=situation, extra_params=extra_params,
        )
        response_text = await _call_llm(prompt)
        if not response_text:
            logger.error("[Replyer] LLM响应为空")
            return {"success": False, "reason": "llm_empty"}
        reply_text = _sanitize_reply(response_text)
        logger.info(f"[Replyer] {reply_text[:60]}...")
        return {"success": True, "reply_text": reply_text, "source": "llm_generated"}
    except Exception as e:
        logger.error(f"[主动回复] 生成回复失败: {e}")
        return {"success": False, "error": str(e)}


async def _call_llm(prompt: str, max_tokens: int = 500) -> Optional[str]:
    try:
        from src.llm_models.utils_model import LLMRequest
        from src.config.config import model_config
        request = LLMRequest(model_config.focus_chat, request_type="proactive_replyer")
        response_text, _ = await request.generate_response_async(prompt, max_tokens=max_tokens)
        return response_text
    except Exception as e:
        logger.error(f"[Replyer] LLM调用失败: {e}")
        return None


async def _generate_proactive_reply(plan: Dict, context_info: Dict) -> Dict:
    try:
        actions = plan.get("actions", [])
        if not actions:
            return {"success": False, "reason": "no_actions"}
        reply_action = None
        for action in actions:
            if action.get("type") in ["miao_reply", "respond"]:
                reply_action = action
                break
        if not reply_action:
            return {"success": False, "reason": "no_reply_action"}
        existing_content = reply_action.get("content")
        if existing_content and existing_content.strip():
            return {"success": True, "reply_text": existing_content.strip(), "source": "plan_content"}
        return await _generate_llm_proactive_reply(plan, context_info)
    except Exception as e:
        logger.error(f"[主动回复] 生成主动回复失败: {e}")
        return _get_fallback_reply(context_info)


async def _generate_llm_proactive_reply(plan: Dict, context_info: Dict) -> Dict:
    try:
        prompt = _build_proactive_reply_prompt(plan, context_info)
        logger.info(f"[主动回复] LLM生成开始 | 提示词长度: {len(prompt)}字符")
        response_text = await _call_llm(prompt)
        if not response_text or not response_text.strip():
            logger.warning("[主动回复] LLM响应为空")
            return _get_fallback_reply(context_info)
        cleaned_reply = _clean_reply_text(response_text)
        if not cleaned_reply or len(cleaned_reply.strip()) < 2:
            logger.warning("[主动回复] 生成内容太短")
            return _get_fallback_reply(context_info)
        return {"success": True, "reply_text": cleaned_reply, "source": "llm_generated"}
    except Exception as e:
        logger.error(f"[主动回复] LLM生成失败: {e}")
        return _get_fallback_reply(context_info)


def _build_proactive_reply_prompt(plan: Dict, context_info: Dict) -> str:
    try:
        thought = plan.get("thought", "")
        silence_duration = context_info.get("silence_duration", 0)
        persona = context_info.get("current_persona", {})
        emotion_state = context_info.get("emotion_state", {})
        recent_topics = context_info.get("recent_topics", "")
        persona_info = ""
        if persona.get("name"):
            persona_info = f"你是 {persona['name']}"
            if persona.get("tone"):
                persona_info += f"（{persona['tone']}语调）"
            if persona.get("description"):
                persona_info += f"\n人格描述：{persona['description']}"
        emotion_info = ""
        if emotion_state:
            mood_score = emotion_state.get("average_mood_score", 0.5)
            energy = emotion_state.get("energy", 50)
            emotion_info = f"当前心情指数：{mood_score:.2f}，能量值：{energy}/100"
            if emotion_state.get("global_shield_active"):
                emotion_info += "\n（当前处于保护状态）"
        topics_info = ""
        if recent_topics.strip():
            topics_info = f"\n近期对话内容：\n{recent_topics[:150]}"
        from src.config.prompt_loader import get_prompt, PromptCategory
        prompt = get_prompt(
            PromptCategory.MODULE,
            "proactive",
            "proactive_reply.template",
            persona_info=persona_info,
            silence_duration_minutes=f"{silence_duration/60:.0f}",
            thought=thought,
            emotion_info=emotion_info,
            topics_info=topics_info
        )
        return prompt
    except Exception as e:
        logger.debug(f"构建提示词失败: {e}")
        return "请生成一条简单的问候消息。"


def _clean_reply_text(text: str) -> str:
    try:
        if not text:
            return ""
        cleaned = text.strip()
        prefixes_to_remove = [
            "消息内容：", "回复：", "生成的消息：", "主动消息：",
            "AI:", "机器人：", "Bot:",
        ]
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        elif cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1].strip()
        if len(cleaned) > 200:
            cleaned = cleaned[:200].rstrip() + "..."
        return cleaned
    except Exception:
        return text


def _get_fallback_reply(context_info: Dict) -> Dict:
    try:
        persona = context_info.get("current_persona", {}) if context_info else {}
        persona_tone = persona.get("tone", "").lower() if persona else ""
        if "活泼" in persona_tone or "开朗" in persona_tone:
            fallback_messages = [
                "大家好啊~ 最近怎么样？",
                "呀！好像有点安静呢，来聊聊天吧！",
                "大家都在忙什么呢？有什么有趣的事情吗？",
            ]
        elif "温和" in persona_tone or "友好" in persona_tone:
            fallback_messages = [
                "大家好，最近过得怎么样？",
                "想和大家聊聊天呢，有什么新鲜事吗？",
                "好久没聊天了，大家都好吗？",
            ]
        elif "冷静" in persona_tone or "理性" in persona_tone:
            fallback_messages = [
                "观察到频道有些安静，最近有什么值得关注的话题吗？",
                "最近有什么有趣的现象或观点吗？",
                "大家对最近的事情有什么看法？",
            ]
        else:
            fallback_messages = [
                "大家好，最近怎么样？",
                "有点想念大家了，最近都在做什么呢？",
                "在吗？聊聊天吧~",
                "啊，好像有点安静呢，大家都在忙什么？",
            ]
        message = random.choice(fallback_messages)
        return {"success": True, "reply_text": message, "source": "fallback"}
    except Exception:
        return {"success": True, "reply_text": "大家好，最近怎么样？", "source": "default_fallback"}


def _get_user_analysis_context_replyer(user_id: str, extra_params: Dict) -> str:
    context_parts = []
    try:
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker("")
            if tracker:
                state = tracker.get_user_state(user_id, create_if_missing=False)
                if state:
                    affection = state.affection
                    relationship = state.relationship
                    if affection <= -80:
                        context_parts.append("用户对你非常厌恶")
                    elif affection <= -50:
                        context_parts.append("用户对你有敌意")
                    elif affection <= -15:
                        context_parts.append("用户对你态度冷淡")
                    elif affection > 50:
                        context_parts.append("用户对你很有好感")
                    if relationship:
                        context_parts.append(f"你们的关系是：{relationship}")
        except Exception:
            pass
    except Exception:
        pass
    return "；".join(context_parts) if context_parts else ""


def _get_current_persona_reply_hint(session_id: str, extra_params: Dict) -> str:
    try:
        from src.modules.modcore.dynamic_persona.persona_switcher import get_persona_switcher
        switcher = get_persona_switcher()
        active = switcher.get_active_persona(session_id)
        if not active:
            return ""
        persona_name = getattr(active, 'name', '') or ''
        persona_tone = getattr(active, 'tone', '') or ''
        try:
            from src.modules.modcore.perception.perception_generator import get_perception_generator
            from src.modules.modcore.psychological_core import get_psychological_core
            psych_core = get_psychological_core()
            perception_gen = get_perception_generator()
            if psych_core and perception_gen:
                perception_desc = perception_gen.generate_persona_perception(
                    persona_name=persona_name,
                    persona_tone=persona_tone,
                )
                if perception_desc:
                    return perception_desc
        except Exception:
            pass
        return f"你当前的人格特征：{persona_name}，音调：{persona_tone}"
    except Exception:
        return ""


def _build_replyer_prompt(
    session: Optional[MiaoSession], user_name: str, inner_thought: str,
    situation: str, extra_params: Optional[Dict],
) -> str:
    extra_params = extra_params or {}
    history_text = "（暂无对话历史）"
    if session:
        recent_records = session.fetch_recent_records(limit=5)
        history_text = _format_brief_history(recent_records, user_name)
    situation_desc = _get_situation_description(situation, extra_params)
    user_analysis = ""
    if session:
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(session.stream_id)
            state = tracker.get_user_state(session.user_id, create_if_missing=False)
            if state:
                if state.affection <= -50:
                    user_analysis = "用户对你有敌意"
                elif state.affection <= -15:
                    user_analysis = "用户对你态度冷淡"
                elif state.affection > 50:
                    user_analysis = "用户对你很有好感"
                if state.relationship:
                    user_analysis += f"，你们的关系是：{state.relationship}"
        except Exception:
            pass
    analysis_hint = f"\n用户状态信息：{user_analysis}\n请根据这些信息调整回复的语气和内容。" if user_analysis else ""
    persona_hint = ""
    if session:
        try:
            from src.modules.modcore.dynamic_persona.persona_switcher import get_persona_switcher
            switcher = get_persona_switcher()
            active = switcher.get_active_persona(session.stream_id)
            if active:
                persona_hint = f"\n你当前的人格特征：{getattr(active, 'name', '')}"
        except Exception:
            pass
    from src.config.prompt_loader import get_prompt, PromptCategory
    prompt = get_prompt(
        PromptCategory.MODULE,
        "proactive",
        "replyer_chat.template",
        user_name=user_name,
        history_text=history_text,
        situation_desc=situation_desc,
        analysis_hint=analysis_hint,
        persona_hint=persona_hint,
        inner_thought=inner_thought
    )
    return prompt


def _format_brief_history(records: List, user_name: str) -> str:
    if not records:
        return "（这是你们的第一次对话）"
    lines = ["最近的对话："]
    for record in records[-3:]:
        if record.kind == ActivityKind.INCOMING_MSG:
            content = record.text_content[:50]
            if len(record.text_content) > 50:
                content += "..."
            lines.append(f"  {user_name}: {content}")
        elif record.kind == ActivityKind.BOT_DECISION:
            for behavior in record.behavior_list:
                if behavior.get("type") in ("miao_reply", "respond"):
                    content = behavior.get("content", "")[:50]
                    if len(behavior.get("content", "")) > 50:
                        content += "..."
                    lines.append(f"  你: {content}")
                    break
    return "\n".join(lines)


def _get_situation_description(situation: str, extra_params: Dict) -> str:
    descriptions = {
        "new_message": "用户发来了新消息",
        "reply_in_time": "用户在你期待的时间内回复了",
        "reply_late": "用户超出预期时间才回复",
        "timeout": "等待超时，你决定追问",
        "proactive": "你主动发起对话",
    }
    return descriptions.get(situation, situation)


def _sanitize_reply(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.split("\n")
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    prefixes_to_remove = [
        "你说：", "你说:",
        "回复：", "回复:",
        "我说：", "我说:",
        "消息：", "消息:",
        "回复内容：", "回复内容:",
    ]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    try:
        from src.config.config import global_config
        forbidden_phrases = global_config.get("forbidden_phrases", "")
        if forbidden_phrases:
            for phrase in forbidden_phrases.split(","):
                phrase = phrase.strip()
                if phrase:
                    text = text.replace(phrase, "")
        forbidden_patterns = global_config.get("forbidden_patterns", "")
        if forbidden_patterns:
            for pattern in forbidden_patterns.split(","):
                pattern = pattern.strip()
                if pattern:
                    try:
                        text = re.sub(pattern, "", text)
                    except re.error:
                        pass
    except Exception:
        pass
    while text.endswith("。。") or text.endswith("！！") or text.endswith("？？"):
        text = text[:-1]
    return text.strip()

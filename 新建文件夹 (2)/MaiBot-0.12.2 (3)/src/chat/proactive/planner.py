import json
import re
import time
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
from src.common.logger import get_logger
from src.config.prompt_loader import get_prompt, PromptCategory
from src.chat.proactive.models import GeneratorResult, BehaviorModel
from src.chat.proactive.session_manager import MiaoSession

logger = get_logger("proactive_planner")


@dataclass
class ProactiveScores:
    mood_score: float = 5.0
    energy_score: float = 5.0
    willingness_score: float = 5.0
    silence_appropriateness: float = 5.0
    topic_availability: float = 5.0
    shield_status: float = 10.0
    timing_score: float = 5.0
    user_receptiveness: float = 5.0
    next_check_delay: int = 600
    confidence: float = 0.5
    dynamic_weights: Optional[Dict] = None

    def get_overall(self, channel_type: str = "group") -> float:
        if self.dynamic_weights:
            weights = self.dynamic_weights
        elif channel_type == "private":
            weights = {
                "mood_score": 0.20, "energy_score": 0.15, "willingness_score": 0.25,
                "silence_appropriateness": 0.10, "topic_availability": 0.05,
                "timing_score": 0.10, "user_receptiveness": 0.15,
            }
        else:
            weights = {
                "mood_score": 0.25, "energy_score": 0.20, "willingness_score": 0.20,
                "silence_appropriateness": 0.15, "topic_availability": 0.10,
                "shield_status": 0.05, "timing_score": 0.05,
            }
        total = sum(getattr(self, k, 5.0) * v for k, v in weights.items())
        return total / 10.0

    def get_user_state_summary(self, user_state: Dict) -> Dict:
        return {
            "favor": user_state.get("favor", user_state.get("affection", 0)),
            "annoyance": user_state.get("annoyance", 0),
            "trauma": user_state.get("trauma", user_state.get("trauma_score", 0)),
            "user_receptiveness": self.user_receptiveness,
            "shield_active": self.shield_status < 5.0,
            "mood_score": self.mood_score,
            "energy_score": self.energy_score,
        }


def _calculate_user_receptiveness(user_state: Dict, context_info: Dict) -> float:
    base_score = 5.0
    affection = user_state.get("favor", user_state.get("affection", 0))
    affection_factor = affection / 50.0
    annoyance = user_state.get("annoyance", 0)
    annoyance_factor = -annoyance / 33.3
    final_score = base_score + affection_factor + annoyance_factor
    return max(0.0, min(10.0, final_score))


async def _call_llm(prompt: str, max_tokens: int = 1000) -> Optional[str]:
    try:
        from src.llm_models.utils_model import LLMRequest
        from src.config.config import model_config
        request = LLMRequest(model_config.focus_chat, request_type="proactive_planner")
        response_text, _ = await request.generate_response_async(prompt, max_tokens=max_tokens)
        return response_text
    except Exception as e:
        logger.error(f"[Planner] LLM调用失败: {e}")
        return None


async def create_plan(
    session: MiaoSession, user_name: str, situation: str = "new_message",
    available_behaviors: Optional[Dict] = None, extra_params: Optional[Dict] = None,
) -> GeneratorResult:
    """
    生成行动计划

    Args:
        session: 会话对象
        user_name: 用户名称
        situation: 情况类型 (new_message, reply_in_time, reply_late, timeout, proactive)
        available_behaviors: 可用行为字典
        extra_params: 额外参数

    Returns:
        GeneratorResult 对象，包含计划信息
    """
    try:
        prompt = _build_planner_prompt(
            session=session,
            user_name=user_name,
            situation=situation,
            available_behaviors=available_behaviors,
            extra_params=extra_params,
        )

        response_text = await _call_llm(prompt)

        if not response_text:
            logger.error("[Planner] LLM响应为空")
            return GeneratorResult.create_fallback("LLM响应为空")

        logger.debug(f"[Planner] LLM响应:\n{response_text[:200]}...")

        return _parse_plan_response(response_text)

    except Exception as e:
        logger.error(f"[Planner] 生成计划失败: {e}")
        import traceback
        traceback.print_exc()
        return GeneratorResult.create_fallback(str(e))


async def create_proactive_plan(channel_id: str, situation_desc: str, context_info: Optional[Dict] = None) -> Dict:
    """
    创建主动对话的行动计划（增强版，支持情感和人格集成）

    Args:
        channel_id: 频道ID
        situation_desc: 情况描述
        context_info: 额外上下文信息（包含情感状态、人格信息等）

    Returns:
        包含行动计划的字典
    """
    try:
        context_info = context_info or {}

        if context_info.get("is_proactive", False):
            return await _create_proactive_plan(channel_id, situation_desc, context_info)

        raw_response = await _call_llm(_build_proactive_prompt(situation_desc, context_info))

        if not raw_response:
            logger.info("[Planner] LLM响应为空，不主动")
            return {"success": False, "reason": "llm_empty"}

        logger.debug(f"[Planner] LLM响应:\n{raw_response[:200]}...")

        return _parse_proactive_response(raw_response, context_info)

    except Exception as e:
        logger.error(f"[主动规划] 创建计划失败: {e}")
        return {"success": False, "error": str(e)}


async def _create_simple_proactive_plan(channel_id: str, situation_desc: str, context_info: Dict) -> Dict:
    try:
        emotion_state = context_info.get("emotion_state", {})
        avg_mood = emotion_state.get("average_mood_score", 0.5)
        current_energy = emotion_state.get("energy", 50)
        shield_active = emotion_state.get("global_shield_active", False)
        silence_duration = context_info.get("silence_duration", 0)
        if shield_active:
            logger.info(f"[主动规划-普通] 保护盾激活，拒绝主动")
            return {"success": False, "reason": "shield_active", "score": 0.0}
        silence_hours = silence_duration / 3600
        if silence_hours < 0.5:
            logger.info(f"[主动规划-普通] 沉默时间过短({silence_hours:.1f}h)，拒绝主动")
            return {"success": False, "reason": "silence_too_short", "score": 0.0}
        state_score = (avg_mood + 1.0) / 2.0 * 0.5 + (current_energy / 100.0) * 0.5
        if state_score < 0.3:
            logger.info(f"[主动规划-普通] 状态评分过低({state_score:.2f})，拒绝主动")
            return {"success": False, "reason": "state_too_low", "score": 0.0}
        logger.info(f"[主动规划-普通] 条件满足，允许主动")
        return {"success": True, "thought": "", "decision": "proactive", "reason": "普通模式简化决策", "score": state_score, "scores": {}}
    except Exception as e:
        logger.error(f"[主动规划-普通] 失败: {e}")
        return {"success": False, "error": str(e), "score": 0.0}


async def _create_private_proactive_plan(channel_id: str, situation_desc: str, context_info: Dict) -> Dict:
    try:
        emotion_state = context_info.get("emotion_state", {})
        user_state = context_info.get("user_state", {})
        affection = user_state.get("favor", user_state.get("affection", 0))
        annoyance = user_state.get("annoyance", 0)
        trauma_score = user_state.get("trauma", user_state.get("trauma_score", 0))
        avg_mood = emotion_state.get("average_mood_score", 0.5)
        current_energy = emotion_state.get("energy", 50)
        shield_active = emotion_state.get("global_shield_active", False)
        logger.info(
            f"[主动规划-私聊] 用户状态 | 好感度={affection} | 烦躁度={annoyance} | "
            f"创伤={trauma_score:.1f} | 心情={avg_mood:.2f} | 能量={current_energy} | "
            f"保护盾={'激活' if shield_active else '未激活'}"
        )
        context_info["user_receptiveness"] = _calculate_user_receptiveness(user_state, context_info)
        prompt = _build_private_proactive_prompt(situation_desc, context_info)
        response_text = await _call_llm(prompt)
        if not response_text:
            logger.error(f"[主动规划-私聊] LLM调用失败，拒绝主动")
            return {"success": False, "reason": "llm_failed", "next_check_delay": 600}
        plan_result = _parse_proactive_response(response_text, context_info)
        scores_dict = plan_result.get("scores", {})
        next_check_delay = plan_result.get("next_check_delay", 600)
        confidence = plan_result.get("confidence", 0.5)
        user_strategy = plan_result.get("user_strategy", {})
        next_check_delay = max(60, min(next_check_delay, 86400))
        if scores_dict:
            scores = ProactiveScores(
                mood_score=scores_dict.get("mood_score", 5.0),
                energy_score=scores_dict.get("energy_score", 5.0),
                willingness_score=scores_dict.get("willingness_score", 5.0),
                silence_appropriateness=scores_dict.get("silence_appropriateness", 5.0),
                topic_availability=scores_dict.get("topic_availability", 5.0),
                shield_status=10.0 if not shield_active else 0.0,
                timing_score=scores_dict.get("timing_score", 5.0),
                user_receptiveness=scores_dict.get("user_receptiveness", context_info["user_receptiveness"]),
                next_check_delay=next_check_delay,
                confidence=confidence,
            )
            overall_score = scores.get_overall("private")
            plan_result["score"] = overall_score
            logger.info(
                f"[主动规划-私聊] 评分详情 | 综合={overall_score:.2f} | 信心度={confidence:.2f} | "
                f"心情={scores.mood_score:.1f} | 能量={scores.energy_score:.1f} | "
                f"意愿={scores.willingness_score:.1f} | 用户接受度={scores.user_receptiveness:.1f} | "
                f"下次检查={next_check_delay}s"
            )
        plan_result["next_check_delay"] = next_check_delay
        plan_result["confidence"] = confidence
        plan_result["user_strategy"] = user_strategy
        return plan_result
    except Exception as e:
        logger.error(f"[主动规划-私聊] 决策失败: {e}")
        return {"success": False, "error": str(e), "next_check_delay": 600}


async def _create_proactive_plan(channel_id: str, situation_desc: str, context_info: Dict) -> Dict:
    try:
        chat_mode = context_info.get("chat_mode", "normal")
        if chat_mode == "normal":
            return await _create_simple_proactive_plan(channel_id, situation_desc, context_info)
        channel_type = context_info.get("channel_type", "group")
        if channel_type == "private":
            return await _create_private_proactive_plan(channel_id, situation_desc, context_info)
        emotion_state = context_info.get("emotion_state", {})
        avg_mood = emotion_state.get("average_mood_score", 0.5)
        current_energy = context_info.get("energy", 50)
        shield_active = emotion_state.get("global_shield_active", False)
        silence_duration = context_info.get("silence_duration", 0)
        silence_hours = silence_duration / 3600
        logger.info(
            f"[主动规划] {channel_id[:8]} | 心情={avg_mood:.2f} 能量={current_energy} "
            f"盾={'开' if shield_active else '关'} 沉默={silence_duration/60:.0f}分钟"
        )
        if shield_active:
            logger.info(f"[主动规划] {channel_id[:8]} 保护盾激活，拒绝主动")
            return {"success": False, "reason": "保护盾激活", "skipped": True}
        if avg_mood < 0.3:
            logger.info(f"[主动规划] {channel_id[:8]} 心情不好({avg_mood:.2f})，不想说话")
            return {"success": False, "reason": f"心情不好({avg_mood:.2f})", "skipped": True}
        if current_energy < 30:
            logger.info(f"[主动规划] {channel_id[:8]} 能量不足({current_energy})，不想说话")
            return {"success": False, "reason": f"能量不足({current_energy})", "skipped": True}
        if silence_duration < 60:
            logger.debug(f"[主动规划] {channel_id[:8]} 沉默不足1分钟，跳过")
            return {"success": False, "reason": "刚说过话", "skipped": True}
        if silence_hours < 0.5 and avg_mood < 0.7:
            logger.info(f"[主动规划] {channel_id[:8]} 沉默太短({silence_hours:.1f}h)+心情一般({avg_mood:.2f})，算了")
            return {"success": False, "reason": "沉默太短且心情一般", "skipped": True}
        current_persona = context_info.get("current_persona", {})
        strategy = _determine_proactive_strategy(emotion_state, current_persona, silence_duration)
        enhanced_situation = _build_enhanced_situation(situation_desc, context_info, strategy)
        response_text = await _call_llm(_build_proactive_prompt(enhanced_situation, context_info))
        if not response_text:
            logger.info(f"[主动规划] {channel_id[:8]} LLM调用失败，算了")
            return {"success": False, "reason": "llm_failed"}
        plan_result = _parse_proactive_response(response_text, context_info)
        if plan_result.get("skipped") or not plan_result.get("success"):
            logger.info(f"[主动规划] {channel_id[:8]} LLM决定不主动: {plan_result.get('reason', '')}")
            return plan_result
        plan_result = _adjust_plan_by_strategy(plan_result, strategy, context_info)
        logger.info(f"🎯 [主动规划] {channel_id[:8]} LLM决定主动 | 策略={strategy} | 原因={plan_result.get('reason', '')[:50]}")
        return plan_result
    except Exception as e:
        logger.error(f"[主动规划] 失败: {e}")
        return {"success": False, "error": str(e)}


def _determine_proactive_strategy(emotion_state: Dict, persona: Dict, silence_duration: float) -> str:
    try:
        avg_mood = emotion_state.get("average_mood_score", 0.5)
        shield_active = emotion_state.get("global_shield_active", False)
        persona_tone = persona.get("tone", "").lower()
        silence_hours = silence_duration / 3600
        if shield_active:
            return "cautious"
        if silence_hours > 2 and avg_mood < 0.3:
            return "gentle_care"
        if silence_hours < 1 and avg_mood > 0.7:
            return "cheerful_chat"
        if persona_tone in ['活泼', '开朗', '热情']:
            return "energetic"
        elif persona_tone in ['温和', '友好']:
            return "friendly"
        elif persona_tone in ['冷静', '理性']:
            return "analytical"
        return "balanced"
    except Exception:
        return "balanced"


def _build_enhanced_situation(base_desc: str, context_info: Dict, strategy: str) -> str:
    try:
        enhanced = base_desc + "\n\n"
        strategy_guides = {
            "cautious": "当前处于保护状态，需要小心谨慎地处理。",
            "gentle_care": "需要温和关心，可能需要安慰或鼓励。",
            "cheerful_chat": "适合轻松快乐的聊天，可以分享有趣的事情。",
            "energetic": "可以更加活跃和充满能量地互动。",
            "friendly": "保持友好温和的气氛。",
            "analytical": "可以分享一些有趣的知识或观点。",
            "balanced": "保持平衡，根据情况灵活处理。",
        }
        enhanced += strategy_guides.get(strategy, "")
        persona = context_info.get("current_persona", {})
        if persona.get("name"):
            enhanced += f"\n\n当前人格: {persona['name']}"
            if persona.get("tone"):
                enhanced += f"（{persona['tone']}语调）"
        recent_topics = context_info.get("recent_topics", "")
        if recent_topics.strip():
            enhanced += f"\n\n近期话题:\n{recent_topics[:200]}"
        return enhanced
    except Exception:
        return base_desc


def _build_private_proactive_prompt(situation_desc: str, context_info: Dict) -> str:
    silence_duration = context_info.get("silence_duration", 0)
    emotion_state = context_info.get("emotion_state", {})
    current_persona = context_info.get("current_persona", {})
    user_state = context_info.get("user_state", {})
    silence_mins = silence_duration / 60
    avg_mood = emotion_state.get("average_mood_score", 0.5)
    current_energy = emotion_state.get("energy", 50)
    affection = user_state.get("favor", user_state.get("affection", 0))
    annoyance = user_state.get("annoyance", 0)
    trauma_score = user_state.get("trauma", user_state.get("trauma_score", 0))
    last_message = context_info.get("last_user_message", "")
    last_message_preview = last_message[:100] + "..." if len(last_message) > 100 else last_message
    persona_name = current_persona.get("name", "我")
    persona_tone = current_persona.get("tone", "随意")
    recent_messages = context_info.get("recent_messages", [])
    conversation_history = ""
    if recent_messages:
        for msg in recent_messages[-10:]:
            sender = msg.get("sender", "未知")
            content = msg.get("content", "")[:50]
            conversation_history += f"- [{sender}] {content}\n"
    else:
        conversation_history = "（无最近对话记录）"
    prompt = get_prompt(
        PromptCategory.PROACTIVE, "planner", "private.template",
        persona_name=persona_name, persona_tone=persona_tone,
        situation_desc=situation_desc, conversation_history=conversation_history,
        silence_mins=f"{silence_mins:.0f}",
        last_message_preview=last_message_preview if last_message_preview else "（无）",
        avg_mood=f"{avg_mood:.2f}", current_energy=f"{current_energy:.0f}",
        affection=affection, annoyance=annoyance, trauma_score=trauma_score,
    )
    return prompt


def _build_proactive_prompt(situation_desc: str, context_info: Dict) -> str:
    silence_duration = context_info.get("silence_duration", 0)
    emotion_state = context_info.get("emotion_state", {})
    current_persona = context_info.get("current_persona", {})
    recent_topics = context_info.get("recent_topics", "")
    silence_mins = silence_duration / 60
    avg_mood = emotion_state.get("average_mood_score", 0.5)
    current_energy = emotion_state.get("energy", 50)
    shield_active = emotion_state.get("global_shield_active", False)
    persona_name = current_persona.get("name", "我")
    persona_tone = current_persona.get("tone", "随意")
    recent_messages = context_info.get("recent_messages", [])
    conversation_history = ""
    if recent_messages:
        for msg in recent_messages[-10:]:
            sender = msg.get("sender", "未知")
            content = msg.get("content", "")[:50]
            conversation_history += f"- [{sender}] {content}\n"
    else:
        conversation_history = "（无最近对话记录）"
    topic_context = f"近期话题：{recent_topics[:200]}" if recent_topics and recent_topics.strip() else "近期话题：无明显话题"
    prompt = get_prompt(
        PromptCategory.PROACTIVE, "planner", "group.template",
        persona_name=persona_name, persona_tone=persona_tone,
        conversation_history=conversation_history,
        silence_mins=f"{silence_mins:.0f}", topic_context=topic_context,
        avg_mood=f"{avg_mood:.2f}", current_energy=f"{current_energy:.0f}",
    )
    return prompt


def _parse_proactive_response(response_text: str, context_info: Dict) -> Dict:
    try:
        cleaned = response_text.strip()
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not json_match:
            logger.warning("[主动规划] 响应中未找到JSON格式")
            return {"success": False, "reason": "invalid_format"}
        try:
            result = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"[主动规划] JSON解析失败: {e}")
            try:
                fixed = re.sub(r',\s*}', '}', json_match.group())
                fixed = re.sub(r',\s*]', ']', fixed)
                result = json.loads(fixed)
            except Exception:
                return {"success": False, "reason": "json_parse_error"}
        if not isinstance(result, dict):
            return {"success": False, "reason": "invalid_format"}
        thought = result.get("thought", "")
        should_speak = result.get("should_speak", False)
        confidence = result.get("confidence", 0.5)
        next_check_delay = result.get("next_check_delay", 600)
        reason = result.get("reason", "")
        suggested_content = result.get("suggested_content", "")
        scores_dict = result.get("scores", {})
        user_strategy = result.get("user_strategy", {})
        conversation_ended = result.get("conversation_ended", False)
        user_wants_continue = result.get("user_wants_continue", True)
        decision = result.get("decision", "wait")
        if not should_speak and decision.lower() not in ["proactive", "积极主动", "谨慎主动"]:
            logger.info(f"[主动规划] 模型决定不主动 | 想法: {thought[:50]}... | 理由: {reason} | 下次检查: {next_check_delay}s")
            return {
                "success": False, "thought": thought, "reason": reason,
                "scores": scores_dict, "next_check_delay": next_check_delay,
                "confidence": confidence, "user_strategy": user_strategy,
                "conversation_ended": conversation_ended,
                "user_wants_continue": user_wants_continue, "skipped": True,
            }
        logger.info(f"[主动规划] 模型决定主动 | 想法: {thought[:50]}... | 理由: {reason} | 信心度: {confidence:.2f} | 下次检查: {next_check_delay}s")
        return {
            "success": True, "thought": thought, "reason": reason,
            "scores": scores_dict, "next_check_delay": next_check_delay,
            "confidence": confidence, "suggested_content": suggested_content,
            "user_strategy": user_strategy, "conversation_ended": conversation_ended,
            "user_wants_continue": user_wants_continue,
            "actions": [{"type": "miao_reply", "content": suggested_content}],
            "expected_reaction": "希望有人回应并开启对话",
            "max_wait_seconds": 600,
        }
    except Exception as e:
        logger.warning(f"[主动规划] 解析响应失败: {e}")
        return {"success": False, "reason": "parse_error", "error": str(e), "next_check_delay": 600}


def _try_fix_json(json_str: str) -> Dict:
    try:
        fixed = re.sub(r',\s*}', '}', json_str)
        fixed = re.sub(r',\s*]', ']', fixed)
        return json.loads(fixed)
    except Exception:
        raise ValueError("无法修复JSON")


def _get_user_analysis_context(user_id: str, extra_params: Dict) -> str:
    context_parts = []
    try:
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker("")
            if tracker:
                state = tracker.get_user_state(user_id, create_if_missing=False)
                if state:
                    affection = state.affection
                    trust = state.trust_score
                    relationship = state.relationship
                    affection_desc = "喜爱" if affection > 50 else "友好" if affection > 0 else "冷淡" if affection > -50 else "厌恶"
                    trust_desc = "信任" if trust > 50 else "中性" if trust > 0 else "不信任"
                    context_parts.append(f"当前关系状态：{relationship}（好感：{affection_desc}，信任：{trust_desc}）")
                    if affection <= -50:
                        context_parts.append(get_prompt(PromptCategory.PROACTIVE, "planner", "context_parts.relation_hostile"))
                    elif affection <= -15:
                        context_parts.append(get_prompt(PromptCategory.PROACTIVE, "planner", "context_parts.relation_cold"))
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"[Planner] 获取用户分析信息失败: {e}")
    return "\n".join(context_parts) if context_parts else ""


def _adjust_plan_by_strategy(plan: Dict, strategy: str, context_info: Dict) -> Dict:
    try:
        strategy_wait_times = {
            "cautious": 1200, "gentle_care": 900, "cheerful_chat": 300,
            "energetic": 180, "friendly": 600, "analytical": 900, "balanced": 600,
        }
        if strategy in strategy_wait_times:
            plan["max_wait_seconds"] = strategy_wait_times[strategy]
        return plan
    except Exception:
        return plan


def _build_planner_prompt(
    session: MiaoSession, user_name: str, situation: str,
    available_behaviors: Optional[Dict], extra_params: Optional[Dict],
) -> str:
    extra_params = extra_params or {}
    recent_records = session.fetch_recent_records(limit=10)
    history_text = _format_conversation_history(recent_records, user_name)
    behaviors_text = _format_available_behaviors(available_behaviors)
    situation_context = _build_situation_context(session, situation, extra_params)
    user_analysis = _get_user_analysis_context(session.user_id, extra_params)
    analysis_influence = ""
    if user_analysis:
        if "用户对你有敌意" in user_analysis or "关系状态：仇人" in user_analysis:
            analysis_influence = "\n用户对你的态度敌对，需要采取防御性回复策略。"
        elif "关系状态：冷淡" in user_analysis:
            analysis_influence += "\n用户对你态度冷淡，需要慢慢重建信任。"
        elif "积极" in user_analysis:
            analysis_influence = "\n用户近期处于积极情绪，可以更开放地交流。"
    persona_influence = _get_current_persona_influence(session.stream_id, extra_params)
    
    # 从配置文件加载提示词模板
    prompt_template = get_prompt(PromptCategory.PROACTIVE, "planner", "planner.template")
    
    prompt = prompt_template.format(
        user_name=user_name,
        history_text=history_text,
        situation_context=situation_context,
        user_analysis=user_analysis,
        behaviors_text=behaviors_text,
        analysis_influence=analysis_influence,
        persona_influence=persona_influence
    )
    return prompt


def _get_current_persona_influence(stream_id: str, extra_params: Dict) -> str:
    try:
        from src.modules.modcore.dynamic_persona.persona_switcher import get_persona_switcher
        switcher = get_persona_switcher()
        active = switcher.get_active_persona(stream_id)
        if not active:
            return ""
        persona_name = getattr(active, 'name', '') or ''
        persona_tone = getattr(active, 'tone', '') or ''
        is_main = getattr(active, 'is_main', True)
        if is_main:
            return ""
        
        # 从配置文件加载人格状态提示词
        if "愤怒" in persona_name or "furious" in persona_tone:
            return "\n" + get_prompt(PromptCategory.PSYCHOLOGY, "persona_states", "persona_influence.furious")
        elif "创伤" in persona_name or "trauma" in persona_tone:
            return "\n" + get_prompt(PromptCategory.PSYCHOLOGY, "persona_states", "persona_influence.trauma")
        elif "防御" in persona_name or "defensive" in persona_tone:
            return "\n" + get_prompt(PromptCategory.PSYCHOLOGY, "persona_states", "persona_influence.defensive")
        elif "崩溃" in persona_name or "breakdown" in persona_tone:
            return "\n" + get_prompt(PromptCategory.PSYCHOLOGY, "persona_states", "persona_influence.breakdown")
        elif "厌恶" in persona_name or "disgusted" in persona_tone:
            return "\n" + get_prompt(PromptCategory.PSYCHOLOGY, "persona_states", "persona_influence.disgusted")
        else:
            default_template = get_prompt(PromptCategory.PSYCHOLOGY, "persona_states", "persona_influence.default")
            return f"\n{default_template.format(persona_name=persona_name)}"
    except Exception:
        return ""


def _format_conversation_history(records: List, user_name: str) -> str:
    if not records:
        return "（暂无对话历史）"
    from src.chat.proactive.models import ActivityKind
    lines = ["最近的对话："]
    for record in records[-8:]:
        if record.kind == ActivityKind.INCOMING_MSG:
            content = record.text_content[:80]
            if len(record.text_content) > 80:
                content += "..."
            lines.append(f"  {record.sender_name or user_name}: {content}")
        elif record.kind == ActivityKind.BOT_DECISION:
            for behavior in record.behavior_list:
                if behavior.get("type") in ("miao_reply", "respond"):
                    content = behavior.get("content", "")[:80]
                    if len(behavior.get("content", "")) > 80:
                        content += "..."
                    lines.append(f"  你: {content}")
                    break
    return "\n".join(lines)


def _format_available_behaviors(behaviors: Optional[Dict]) -> str:
    if not behaviors:
        return "- miao_reply: 发送文字回复 (参数: content)\n- idle: 保持沉默，不做任何事\n- nudge_user: 轻戳对方"
    lines = []
    for name, info in behaviors.items():
        desc = info.get("description", "") if isinstance(info, dict) else str(info)
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines) if lines else "- miao_reply: 发送文字回复\n- idle: 保持沉默"


def _build_situation_context(session: MiaoSession, situation: str, extra_params: Dict) -> str:
    context_parts = []
    if situation == "new_message":
        user_message = extra_params.get("user_message", "")
        context_parts.append(f"当前情况：你收到了新消息。")
        context_parts.append(f"用户说: {user_message}")
    elif situation == "reply_in_time":
        elapsed = session.pending_config.elapsed_seconds()
        context_parts.append(f"当前情况：用户在你期待的时间内回复了（等待了{elapsed:.0f}秒）。")
    elif situation == "reply_late":
        elapsed = session.pending_config.elapsed_seconds()
        timeout = session.pending_config.timeout_seconds
        context_parts.append(f"当前情况：用户超出预期时间才回复（等待了{elapsed:.0f}秒，预期{timeout}秒）。")
    elif situation == "timeout":
        elapsed = session.pending_config.elapsed_seconds()
        timeout = session.pending_config.timeout_seconds
        chase_count = extra_params.get("chase_count", 0)
        context_parts.append(f"当前情况：等待超时。已等待{elapsed:.0f}秒（预期{timeout}秒），已追问{chase_count}次。")
    elif situation == "proactive":
        silence_duration = extra_params.get("silence_duration", "一段时间")
        context_parts.append(f"当前情况：你们已经有{silence_duration}没有交流了，你可以考虑主动发起话题。")
    else:
        context_parts.append(f"当前情况：{situation}")
    return "\n".join(context_parts)


def _parse_plan_response(raw_response: str) -> GeneratorResult:
    text = raw_response.strip()
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            result = GeneratorResult.from_dict(data)
            if result.inner_thought:
                logger.info(f"[Planner] {result.inner_thought[:60]}...")
                behaviors_str = ", ".join(b.action_type for b in result.behaviors)
                logger.debug(f"[Planner] behaviors={behaviors_str}")
            else:
                logger.warning("[Planner] 响应缺少thought字段")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"[Planner] JSON解析失败: {e}")
    logger.warning(f"[Planner] 无法解析响应: {text[:200]}...")
    return GeneratorResult.create_fallback("无法解析响应格式")

import json
import time
import re
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from src.common.logger import get_logger
from src.config.prompt_loader import get_prompt, PromptCategory
from src.chat.heart_flow.models.heart_state import (
    HeartState, MentalEvent, ActionIntent, EventType
)

logger = get_logger("inner_voice")


class SceneType(Enum):
    PRIVATE = "private"
    GROUP = "group"


class TopicType(Enum):
    UNKNOWN = "unknown"
    GREETING = "greeting"
    QUESTION = "question"
    COMMAND = "command"
    EMOTIONAL = "emotional"
    KNOWLEDGE = "knowledge"
    CASUAL = "casual"


TOPIC_PATTERNS = {
    TopicType.GREETING: [r"^(你好|嗨|哈喽|早|晚|在吗|在？|有人吗)", r"(早上好|晚上好|下午好|晚安)"],
    TopicType.QUESTION: [r"[?？]$", r"(什么|怎么|为什么|哪里|谁|多少|几)", r"(吗|呢|吧)[?？]$"],
    TopicType.COMMAND: [r"^(请|帮我|给我|告诉我|说|做)", r"(帮我|给我|告诉我|说一下)"],
    TopicType.EMOTIONAL: [r"(开心|难过|伤心|生气|烦|累|爽|哈哈|呜呜|嘤嘤)", r"[！!]{2,}"],
    TopicType.KNOWLEDGE: [r"(知道|了解|懂|会|能|怎么|如何|为什么)", r"(是什么|什么意思|怎么回事)"],
    TopicType.CASUAL: [r"(嗯|哦|啊|呢|吧|嘛)", r"(我觉得|我认为|我想)"],
}

POSITIVE_THOUGHT_KEYWORDS = [
    ("好奇", 2.5), ("有趣", 2.5), ("有意思", 2.5), ("喜欢", 2.0),
    ("开心", 1.5), ("期待", 2.0), ("聊聊", 1.5), ("说说", 1.0),
    ("听听", 1.5), ("想回", 2.0), ("想聊", 2.0), ("想说", 1.5),
]

NEGATIVE_THOUGHT_KEYWORDS = [
    ("不想理", -3.0), ("不想回", -3.0), ("不想聊", -2.5),
    ("没兴趣", -2.0), ("没意思", -2.0), ("无聊", -2.0),
    ("烦死", -2.5), ("懒得", -2.0), ("算了", -1.5),
    ("不管", -1.5), ("沉默", -1.0), ("窥屏", -0.5),
]

REST_THOUGHT_KEYWORDS = [
    ("休息", 3.0), ("累了", 2.5), ("疲惫", 2.5), ("烦死了", 2.5),
    ("不想理", 1.5), ("没精力", 2.0), ("受不了", 2.0),
]

TIME_MODIFIERS = {
    "deep_night": (2, 6, -2),
    "late_night": (0, 2, -1),
    "evening": (22, 24, -1),
    "morning": (6, 9, 0),
    "daytime": (9, 18, 1),
    "afternoon": (18, 22, 0),
}

CONFIDENCE_WEIGHTS = {
    "data_quality": 0.25,
    "context_clarity": 0.20,
    "pattern_match": 0.20,
    "topic_relevance": 0.15,
    "uncertainty": 0.10,
    "time_pressure": 0.10,
}


@dataclass
class InnerVoiceResult:
    thought: str
    actions: List[ActionIntent] = field(default_factory=list)
    expected_reaction: str = ""
    max_wait_seconds: int = 0
    mood: str = ""
    success: bool = True
    error: Optional[str] = None
    reply_desire: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought,
            "actions": [a.to_dict() for a in self.actions],
            "expected_reaction": self.expected_reaction,
            "max_wait_seconds": self.max_wait_seconds,
            "mood": self.mood,
            "success": self.success,
            "error": self.error,
            "reply_desire": self.reply_desire,
        }


@dataclass
class GuardrailContext:
    scene: SceneType = SceneType.GROUP
    is_at_me: bool = False
    is_reply_to_me: bool = False
    user_message: str = ""
    message_length: int = 0
    relationship: str = "陌生人"
    favor: float = 0.0
    trust: float = 0.0
    mental_fatigue: float = 0.0
    stamina: float = 100.0
    annoyance: float = 0.0
    trauma: float = 0.0
    willingness: float = 1.0
    chat_value: float = 100.0
    brain_power: float = 100.0
    consecutive_replies: int = 0
    hour: int = -1
    topic: TopicType = TopicType.UNKNOWN
    confidence: float = 0.5
    engagement: float = 0.5


class InnerVoiceGenerator:
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self._log_tag = "[内心独白]"

    async def generate_thought_and_action(
        self, heart_state: HeartState, user_message: str,
        user_name: str, conversation_context: List[Dict],
        available_actions: Optional[List[str]] = None,
        confidence_score: float = 0.5, engagement_score: float = 0.5,
        mental_fatigue: float = 0.0, stamina: float = 100.0,
        annoyance: float = 0.0, trauma: float = 0.0,
        willingness: float = 1.0, user_id: str = "",
    ) -> InnerVoiceResult:
        try:
            is_at_me = self._detect_at_me(user_message)
            rel_data = self._get_relationship_data(user_name, user_id)
            chat_value, brain_power, consecutive = self._get_resource_state()
            topic = self._detect_topic(user_message)
            is_group = bool(self.stream_id and self.stream_id.startswith("group"))
            ctx = GuardrailContext(
                scene=SceneType.GROUP if is_group else SceneType.PRIVATE,
                is_at_me=is_at_me,
                user_message=user_message,
                message_length=len(user_message),
                relationship=rel_data.get("relationship", "陌生人"),
                favor=rel_data.get("favor", 0.0),
                trust=rel_data.get("trust", 0.0),
                mental_fatigue=mental_fatigue,
                stamina=stamina,
                annoyance=annoyance,
                trauma=trauma,
                willingness=willingness,
                chat_value=chat_value,
                brain_power=brain_power,
                consecutive_replies=consecutive,
                hour=time.localtime().tm_hour,
                topic=topic,
                confidence=confidence_score,
                engagement=engagement_score,
            )
            prompt = self._build_focus_prompt(
                heart_state, user_message, user_name,
                conversation_context, ctx,
            )
            raw = await self._call_llm(prompt)
            if raw == "__SAFETY_VIOLATION__":
                return InnerVoiceResult(thought="...", reply_desire=1)
            if not raw:
                return InnerVoiceResult(thought="没听清", reply_desire=5, success=False, error="LLM失败")
            result = self._parse_json_response(raw, user_message, user_name)
            original_desire = result.reply_desire
            result = self._apply_guardrails(result, ctx)
            if result.reply_desire != original_desire:
                logger.info(f"{self._log_tag} 模型{original_desire}→修正{result.reply_desire}")
            logger.info(f"{self._log_tag} 生成完成 | 欲望={result.reply_desire}")
            return result
        except Exception as e:
            logger.error(f"{self._log_tag} 生成失败: {e}")
            return InnerVoiceResult(thought="...", reply_desire=5, success=False, error=str(e))

    def _build_focus_prompt(
        self, heart_state: HeartState, user_message: str, user_name: str,
        conversation_context: List[Dict], ctx: GuardrailContext,
    ) -> str:
        persona = self._get_persona_block()
        context_lines = []
        for msg in conversation_context[-8:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:100]
            if content:
                prefix = "我" if role == "assistant" else user_name
                context_lines.append(f"{prefix}: {content}")
        context_str = "\n".join(context_lines) if context_lines else "（刚开始聊天）"
        rel_feeling = self._build_relationship_description(user_name, ctx)
        state_line = self._build_state_description(ctx)
        scene_tag = "群聊" if ctx.scene == SceneType.GROUP else "私聊"
        at_tag = ""
        if ctx.is_at_me:
            at_tag = "\n" + get_prompt(PromptCategory.HEARTFLOW, "descriptions", "at_response.at_tag")

        # 简单的内心独白生成（真正的判断在heartFC_chat.py中）
        prompt = get_prompt(
            PromptCategory.HEARTFLOW, "inner_voice", "focus.template",
            persona=persona, scene_tag=scene_tag, context_str=context_str,
            user_name=user_name, user_message=user_message,
            rel_feeling=rel_feeling, state_line=state_line,
            chat_value=f"{ctx.chat_value:.0f}", brain_power=f"{ctx.brain_power:.0f}",
            at_tag=at_tag,
        )
        return prompt

    def _build_relationship_description(self, user_name: str, ctx: GuardrailContext) -> str:
        aff = ctx.favor
        rel = ctx.relationship
        if aff < -30:
            return get_prompt(PromptCategory.HEARTFLOW, "descriptions", "relationship.hate", user_name=user_name, aff=aff, rel=rel)
        elif aff < -10:
            return get_prompt(PromptCategory.HEARTFLOW, "descriptions", "relationship.dislike", user_name=user_name, aff=aff, rel=rel)
        elif aff < 10:
            return get_prompt(PromptCategory.HEARTFLOW, "descriptions", "relationship.neutral", user_name=user_name, aff=aff, rel=rel)
        elif aff < 40:
            return get_prompt(PromptCategory.HEARTFLOW, "descriptions", "relationship.like", user_name=user_name, aff=aff, rel=rel)
        return get_prompt(PromptCategory.HEARTFLOW, "descriptions", "relationship.love", user_name=user_name, aff=aff, rel=rel)

    def _build_state_description(self, ctx: GuardrailContext) -> str:
        parts = []
        if ctx.chat_value < 30:
            parts.append(get_prompt(PromptCategory.HEARTFLOW, "descriptions", "state.low_energy"))
        elif ctx.chat_value < 60:
            parts.append(get_prompt(PromptCategory.HEARTFLOW, "descriptions", "state.medium_energy"))
        if ctx.brain_power < 30:
            parts.append(get_prompt(PromptCategory.HEARTFLOW, "descriptions", "state.low_brain"))
        if ctx.annoyance > 50:
            parts.append(get_prompt(PromptCategory.HEARTFLOW, "descriptions", "state.high_annoyance"))
        elif ctx.annoyance > 30:
            parts.append(get_prompt(PromptCategory.HEARTFLOW, "descriptions", "state.medium_annoyance"))
        if ctx.trauma > 5:
            parts.append(get_prompt(PromptCategory.HEARTFLOW, "descriptions", "state.trauma"))
        return "，".join(parts) if parts else get_prompt(PromptCategory.HEARTFLOW, "descriptions", "state.default")

    def _apply_guardrails(self, result: InnerVoiceResult, ctx: GuardrailContext) -> InnerVoiceResult:
        desire = result.reply_desire
        resource_penalty = self._calculate_resource_penalty(ctx)
        if resource_penalty > 0 and desire > 1:
            old = desire
            desire = max(1, desire - resource_penalty)
            if desire != old:
                logger.info(f"{self._log_tag} [精力惩罚] 聊天值={ctx.chat_value:.0f} 脑力={ctx.brain_power:.0f} 连续={ctx.consecutive_replies} | {old}→{desire}")
        fatigue_penalty = self._calculate_fatigue_penalty(ctx)
        if fatigue_penalty > 0 and desire > 1:
            old = desire
            desire = max(1, desire - fatigue_penalty)
            if desire != old:
                # 详细的惩罚原因已经在 _calculate_fatigue_penalty 中记录
                logger.info(f"{self._log_tag} [疲劳惩罚] 总惩罚={fatigue_penalty} | {old}→{desire}")
        if ctx.is_at_me and desire < 6:
            old = desire
            desire = 6
            logger.info(f"{self._log_tag} [强制回应] 被@或叫名字 {old}→{desire}")
        if ctx.scene == SceneType.PRIVATE and desire < 5:
            old = desire
            desire = 5
            logger.info(f"{self._log_tag} [私聊保底] {old}→{desire}")
        thought_score = self._analyze_thought_sentiment(result.thought)
        if thought_score > 4.0 and desire < 4:
            old = desire
            desire = min(10, desire + 2)
            logger.info(f"{self._log_tag} [一致性] thought积极但desire低 {old}→{desire}")
        elif thought_score < -4.0 and desire > 7:
            old = desire
            desire = max(1, desire - 2)
            logger.info(f"{self._log_tag} [一致性] thought消极但desire高 {old}→{desire}")
        rest_score = self._analyze_rest_intent(result.thought)
        if rest_score > 4.0 and desire > 3:
            old = desire
            desire = min(desire, 2)
            logger.info(f"{self._log_tag} [一致性] 明确休息意图 {old}→{desire}")
        time_adj = self._calculate_time_adjustment(ctx.hour)
        if time_adj != 0 and desire > 1:
            old = desire
            desire = max(1, min(10, desire + time_adj))
            if desire != old:
                logger.info(f"{self._log_tag} [时段调整] hour={ctx.hour} adj={time_adj} | {old}→{desire}")
        desire = max(1, min(10, desire))
        result.reply_desire = desire
        return result

    def _analyze_thought_sentiment(self, thought: str) -> float:
        score = 0.0
        for keyword, weight in POSITIVE_THOUGHT_KEYWORDS:
            if keyword in thought:
                score += weight
        for keyword, weight in NEGATIVE_THOUGHT_KEYWORDS:
            if keyword in thought:
                score += weight
        return score

    def _analyze_rest_intent(self, thought: str) -> float:
        score = 0.0
        for keyword, weight in REST_THOUGHT_KEYWORDS:
            if keyword in thought:
                score += abs(weight)
        return score

    def _calculate_fatigue_penalty(self, ctx: GuardrailContext) -> int:
        """计算疲劳惩罚
        
        考虑因素：
        1. mental_fatigue (心理疲劳) - 越高惩罚越大
        2. stamina (耐力) - 越低惩罚越大
        3. willingness (意愿度) - 越低惩罚越大
        """
        penalty = 0
        fatigue_penalty = 0
        stamina_penalty = 0
        willingness_penalty = 0
        
        # 疲劳度惩罚
        if ctx.mental_fatigue > 80:
            fatigue_penalty = 2
        elif ctx.mental_fatigue > 60:
            fatigue_penalty = 1
        
        # 耐力惩罚
        if ctx.stamina < 20:
            stamina_penalty = 2
        elif ctx.stamina < 40:
            stamina_penalty = 1
        
        # 意愿度惩罚
        willingness_factor = max(0.0, 1.0 - ctx.willingness)
        if willingness_factor > 0.5:
            willingness_penalty = 1
        
        penalty = fatigue_penalty + stamina_penalty + willingness_penalty
        
        # 如果有惩罚，记录详细信息
        if penalty > 0:
            details = []
            if fatigue_penalty > 0:
                details.append(f"疲劳={ctx.mental_fatigue:.0f}(-{fatigue_penalty})")
            if stamina_penalty > 0:
                details.append(f"耐力={ctx.stamina:.0f}(-{stamina_penalty})")
            if willingness_penalty > 0:
                details.append(f"意愿={ctx.willingness:.2f}(-{willingness_penalty})")
            logger.debug(f"{self._log_tag} [疲劳分析] {' '.join(details)} 总惩罚={penalty}")
        
        return min(penalty, 4)

    def _calculate_resource_penalty(self, ctx: GuardrailContext) -> int:
        penalty = 0
        if ctx.chat_value < 30:
            penalty += 2
        elif ctx.chat_value < 50:
            penalty += 1
        if ctx.brain_power < 30:
            penalty += 2
        elif ctx.brain_power < 50:
            penalty += 1
        if ctx.consecutive_replies >= 5:
            penalty += 2
        elif ctx.consecutive_replies >= 3:
            penalty += 1
        return min(penalty, 5)

    def _calculate_time_adjustment(self, hour: int) -> int:
        if hour < 0:
            hour = time.localtime().tm_hour
        for name, (start, end, adj) in TIME_MODIFIERS.items():
            if start <= hour < end:
                return adj
        return 0

    def _detect_at_me(self, user_message: str) -> bool:
        try:
            from src.config.config import global_config
            bot_name = global_config.bot.nickname or ""
            alias_names = global_config.bot.alias_names or []
            for n in [bot_name] + alias_names:
                if n and n in user_message:
                    return True
            if "@" in user_message:
                return True
        except Exception:
            pass
        return False

    def _detect_topic(self, text: str) -> TopicType:
        if not text:
            return TopicType.UNKNOWN
        best_topic = TopicType.UNKNOWN
        best_count = 0
        for topic, patterns in TOPIC_PATTERNS.items():
            count = sum(1 for p in patterns if re.search(p, text))
            if count > best_count:
                best_count = count
                best_topic = topic
        return best_topic

    def _get_resource_state(self) -> Tuple[float, float, int]:
        chat_value = 100.0
        brain_power = 100.0
        consecutive = 0
        try:
            from src.chat.heart_flow.heartflow import get_heartflow
            hf = get_heartflow(self.stream_id)
            if hf and hasattr(hf, 'chatting') and hf.chatting:
                chat_value = getattr(hf.chatting, '_chat_value', 100.0)
                brain_power = getattr(hf.chatting, '_brain_power', 100.0)
                consecutive = getattr(hf.chatting, '_consecutive_replies', 0)
        except Exception:
            pass
        return chat_value, brain_power, consecutive

    def calculate_confidence(self, user_message: str, conversation_context: List[Dict]) -> float:
        scores: Dict[str, float] = {}
        msg_len = len(user_message.strip())
        if msg_len < 3:
            scores["data_quality"] = 0.2
        elif msg_len < 10:
            scores["data_quality"] = 0.4
        elif msg_len < 50:
            scores["data_quality"] = 0.6
        else:
            scores["data_quality"] = 0.8
        special_ratio = sum(1 for c in user_message if not c.isalnum() and not c.isspace()) / max(1, msg_len)
        if special_ratio > 0.4:
            scores["data_quality"] = max(0.1, scores["data_quality"] - 0.2)
        ctx_len = len(conversation_context)
        if ctx_len == 0:
            scores["context_clarity"] = 0.3
        elif ctx_len < 3:
            scores["context_clarity"] = 0.5
        elif ctx_len < 8:
            scores["context_clarity"] = 0.7
        else:
            scores["context_clarity"] = 0.8
        topic = self._detect_topic(user_message)
        if topic in (TopicType.QUESTION, TopicType.COMMAND, TopicType.GREETING):
            scores["pattern_match"] = 0.8
        elif topic in (TopicType.EMOTIONAL, TopicType.KNOWLEDGE):
            scores["pattern_match"] = 0.7
        elif topic == TopicType.CASUAL:
            scores["pattern_match"] = 0.5
        else:
            scores["pattern_match"] = 0.4
        scores["topic_relevance"] = 0.6 if topic != TopicType.UNKNOWN else 0.3
        uncertain_words = ["可能", "大概", "也许", "或许", "不清楚", "不知道"]
        uncertain_count = sum(1 for w in uncertain_words if w in user_message)
        scores["uncertainty"] = max(0.2, 0.7 - uncertain_count * 0.15)
        hour = time.localtime().tm_hour
        if 8 <= hour < 22:
            scores["time_pressure"] = 0.6
        elif 22 <= hour or hour < 2:
            scores["time_pressure"] = 0.4
        else:
            scores["time_pressure"] = 0.3
        total = sum(scores.get(k, 0.5) * w for k, w in CONFIDENCE_WEIGHTS.items())
        return max(0.05, min(0.95, total))

    def calculate_willingness(self, ctx: GuardrailContext) -> Tuple[float, str]:
        base = 1.0
        fatigue_pen = min(ctx.mental_fatigue / 100.0, 0.8)
        trauma_pen = min(ctx.trauma / 10.0 * 0.6, 0.6)
        annoy_pen = min(ctx.annoyance / 100.0 * 0.5, 0.5)
        resource_pen = max(0, (50 - ctx.chat_value) / 200) + max(0, (50 - ctx.brain_power) / 200)
        result = base - fatigue_pen - trauma_pen - annoy_pen - resource_pen
        result = max(0.05, min(1.0, result))
        parts = []
        if ctx.mental_fatigue > 60:
            parts.append(f"疲劳{ctx.mental_fatigue:.0f}")
        if ctx.trauma > 5:
            parts.append(f"创伤{ctx.trauma:.1f}")
        if ctx.annoyance > 50:
            parts.append(f"烦躁{ctx.annoyance:.0f}")
        reason = f"意愿{result:.2f}" + (f"({','.join(parts)})" if parts else "(正常)")
        return result, reason

    def suggest_response_length(self, ctx: GuardrailContext) -> str:
        pressure = ctx.mental_fatigue * 0.4 + ctx.trauma * 10 * 0.3 + ctx.annoyance * 0.2
        if pressure < 20:
            return "normal"
        elif pressure < 40:
            return "moderate"
        elif pressure < 60:
            return "brief"
        elif pressure < 80:
            return "minimal"
        return "ultra_minimal"

    def _get_persona_block(self) -> str:
        try:
            from src.config.config import global_config
            bot_name = global_config.bot.nickname or "AI"
            personality = global_config.personality.personality or ""
            plan_style = global_config.personality.plan_style or ""
            alias_names = global_config.bot.alias_names or []
            parts = [f"你是{bot_name}"]
            if alias_names:
                parts[0] += f"(别名{'、'.join(alias_names)})"
            if personality:
                parts.append(personality)
            if plan_style:
                parts.append(plan_style)
            return "，".join(parts) if len(parts) > 1 else parts[0]
        except Exception:
            pass
        return ""

    def _get_relationship_data(self, user_name: str, user_id: str) -> Dict[str, Any]:
        try:
            from src.modules.modcore.psychological_core import get_psychological_core
            psy_core = get_psychological_core()
            if psy_core:
                lookup_id = user_id if user_id else user_name
                return psy_core.get_user_psychological_state(self.stream_id, lookup_id)
        except Exception:
            pass
        return {"relationship": "陌生人", "favor": 0.0, "trust": 0.0}

    async def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            request = LLMRequest(model_config.focus_chat, request_type="inner_voice")
            response_text, _ = await request.generate_response_async(prompt)
            return response_text or None
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            if "EmptyContent" in error_type or "Safety violation" in error_msg or "HARASSMENT" in error_msg:
                logger.warning(f"{self._log_tag} 安全违规: {error_msg}")
                return "__SAFETY_VIOLATION__"
            logger.error(f"{self._log_tag} LLM调用失败: {e}")
            return None

    def _parse_json_response(self, raw: str, user_message: str, user_name: str) -> InnerVoiceResult:
        text = raw.strip()
        think_match = re.search(r'</think>\s*(.+)', text, re.DOTALL)
        if think_match:
            text = think_match.group(1).strip()
        json_match = re.search(r'\{[^{}]*"thought"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                thought = data.get("thought", "").strip()
                desire_raw = data.get("desire", data.get("reply_desire", 5))
                desire = max(1, min(10, int(desire_raw)))
                mood = data.get("mood", "")
                if thought and len(thought) >= 2:
                    return InnerVoiceResult(thought=thought[:120], reply_desire=desire, mood=mood)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        num_match = re.search(r'"?desire"?\s*[:=]\s*(\d+)', text)
        thought_match = re.search(r'"?thought"?\s*[:=]\s*"([^"]+)"', text)
        if thought_match and num_match:
            desire = max(1, min(10, int(num_match.group(1))))
            thought = thought_match.group(1).strip()
            if thought and len(thought) >= 2:
                return InnerVoiceResult(thought=thought[:120], reply_desire=desire)
        plain_match = re.match(r'(\d+)\s*(.*)', text, re.DOTALL)
        if plain_match:
            desire = max(1, min(10, int(plain_match.group(1))))
            remainder = plain_match.group(2).strip()
            if remainder and len(remainder) >= 2:
                return InnerVoiceResult(thought=remainder[:120], reply_desire=desire)
            return InnerVoiceResult(thought=self._desire_to_thought(desire, user_message, user_name), reply_desire=desire)
        any_num = re.search(r'(\d+)', text)
        if any_num:
            desire = max(1, min(10, int(any_num.group(1))))
        else:
            low_kw = ["不想", "无聊", "没兴趣", "懒", "烦"]
            high_kw = ["想", "有趣", "好奇", "回复", "聊"]
            desire = 5
            for kw in low_kw:
                if kw in text:
                    desire = 3
                    break
            for kw in high_kw:
                if kw in text:
                    desire = 7
                    break
        if len(text) >= 2:
            return InnerVoiceResult(thought=text[:120], reply_desire=desire)
        return InnerVoiceResult(thought=self._desire_to_thought(desire, user_message, user_name), reply_desire=desire)

    async def generate_waiting_thought(self, heart_state: HeartState,
                                        user_name: str, progress: float) -> str:
        if progress < 0.4:
            options = ["对方可能在忙吧...", "再等等看", "不知道在做什么呢"]
        elif progress < 0.7:
            options = ["等了一会了...", "是不是忘记回复了？", "嗯...还没消息"]
        else:
            options = ["等了挺久了", "要不要主动说点什么...", "快到时间了"]
        return random.choice(options)

    def _desire_to_thought(self, desire: int, user_message: str, user_name: str) -> str:
        msg_preview = user_message[:20]
        if desire >= 8:
            return f"{user_name}说的'{msg_preview}'很有意思，想回复"
        elif desire >= 6:
            return f"看到{user_name}的消息，有点想聊"
        elif desire >= 4:
            return f"看了看{user_name}的消息，还行"
        elif desire >= 2:
            return f"{user_name}的消息不太感兴趣"
        return "不想理"


_inner_voice_generators: Dict[str, InnerVoiceGenerator] = {}


def get_inner_voice_generator(stream_id: str) -> InnerVoiceGenerator:
    if stream_id not in _inner_voice_generators:
        _inner_voice_generators[stream_id] = InnerVoiceGenerator(stream_id)
    return _inner_voice_generators[stream_id]

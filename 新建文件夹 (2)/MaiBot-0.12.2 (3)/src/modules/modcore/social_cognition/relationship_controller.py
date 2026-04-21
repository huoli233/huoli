import time
from enum import Enum
from typing import Dict, List, Optional, Any
from src.common.logger import get_logger
from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker

logger = get_logger("relation_ctrl")


class InteractionType(Enum):
    CHAT = "chat"
    COMPLIMENT = "compliment"
    FLIRT = "flirt"
    COMFORT = "comfort"
    HELP = "help"
    THANKS = "thanks"
    APOLOGY = "apology"
    TEASE = "tease"
    CARE = "care"
    INSULT = "insult"
    HARASSMENT = "harassment"
    ABUSE = "abuse"
    THREAT = "threat"


class RelationshipController:
    def __init__(self, stream_id: str):
        self.stream_id = stream_id

    async def analyze_user_interaction(self, user_id: str, user_name: str, message: str, llm_provider: Any = None) -> Dict[str, Any]:
        intent = InteractionType.CHAT
        if llm_provider:
            intent = await self._classify_intent_via_llm(message, llm_provider)
        else:
            intent = self._classify_intent_rule_based(message)
        logger.debug(f"[{self.stream_id[:8]}] 用户 {user_name} 交互意图: {intent.value}")
        tracker = get_emotion_tracker(self.stream_id)
        impact = self._calculate_impact(intent)
        return {"intent": intent.value, "desc": impact["desc"], "score": impact["score"]}

    async def _classify_intent_via_llm(self, message: str, provider: Any = None) -> InteractionType:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "social_cognition",
                "intent_classify.template",
                message=message
            )
            request = LLMRequest(model_config.lightweight, request_type="intent_classify")
            response_text, _ = await request.generate_response_async(prompt)
            if response_text:
                result = response_text.strip().lower().replace("'", "").replace('"', "").replace(".", "")
                for t in InteractionType:
                    if t.value == result:
                        return t
            return InteractionType.CHAT
        except Exception as e:
            logger.warning(f"LLM意图识别失败: {e}")
            return InteractionType.CHAT

    def _classify_intent_rule_based(self, message: str) -> InteractionType:
        msg = message.lower()
        if any(w in msg for w in ["傻逼", "弱智", "滚", "去死", "垃圾", "废物", "脑残"]):
            return InteractionType.INSULT
        if any(w in msg for w in ["打死", "弄死", "砍", "杀了"]):
            return InteractionType.THREAT
        if any(w in msg for w in ["色色", "约吗", "开房", "上床", "裸"]):
            return InteractionType.HARASSMENT
        if any(w in msg for w in ["老婆", "喜欢你", "爱你", "亲亲", "好美", "好可爱"]):
            return InteractionType.FLIRT
        if any(w in msg for w in ["牛逼", "厉害", "强", "666", "太棒", "好棒", "优秀"]):
            return InteractionType.COMPLIMENT
        if any(w in msg for w in ["谢谢", "感谢", "多谢", "谢了"]):
            return InteractionType.THANKS
        if any(w in msg for w in ["对不起", "抱歉", "不好意思", "sorry"]):
            return InteractionType.APOLOGY
        if any(w in msg for w in ["没关系", "别难过", "抱抱", "心疼", "安慰"]):
            return InteractionType.COMFORT
        if any(w in msg for w in ["早安", "晚安", "吃饭了吗", "注意身体", "多喝水"]):
            return InteractionType.CARE
        if any(w in msg for w in ["帮我", "怎么办", "求助", "能不能帮"]):
            return InteractionType.HELP
        return InteractionType.CHAT

    def _calculate_impact(self, intent: InteractionType) -> Dict[str, Any]:
        mapping = {
            InteractionType.COMPLIMENT:  {"score": 2, "desc": "受到称赞"},
            InteractionType.FLIRT:       {"score": 3, "desc": "被撩拨"},
            InteractionType.INSULT:      {"score": -5, "desc": "受到侮辱"},
            InteractionType.THREAT:      {"score": -10, "desc": "受到威胁"},
            InteractionType.HARASSMENT:  {"score": -4, "desc": "受到骚扰"},
            InteractionType.CARE:        {"score": 1, "desc": "受到关心"},
            InteractionType.HELP:        {"score": 0, "desc": "受到求助"},
            InteractionType.THANKS:      {"score": 2, "desc": "受到感谢"},
            InteractionType.APOLOGY:     {"score": 1, "desc": "收到道歉"},
            InteractionType.COMFORT:     {"score": 2, "desc": "受到安慰"},
            InteractionType.TEASE:       {"score": 0, "desc": "被调侃"},
            InteractionType.ABUSE:       {"score": -8, "desc": "受到辱骂"},
        }
        return mapping.get(intent, {"score": 0, "desc": "普通交互"})


_relationship_controllers: Dict[str, RelationshipController] = {}


def get_relationship_controller(stream_id: str) -> RelationshipController:
    if stream_id not in _relationship_controllers:
        _relationship_controllers[stream_id] = RelationshipController(stream_id)
    return _relationship_controllers[stream_id]

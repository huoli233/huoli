import time
from typing import Optional, Dict, Any
from src.common.logger import get_logger

logger = get_logger("heartflow_decision")

HEARTFLOW_DECISION_PROMPT = """你是一个聊天决策助手。请根据以下信息，判断机器人是否应该回复这条消息。

【基本信息】
- 机器人名字: {bot_name}
- 发送者: {sender_name}
- 是否@机器人: {is_mentioned}
- 是否私聊: {is_private}

【消息内容】
{message}

【最近对话摘要】
{recent_context}

【判断标准】
1. 如果消息明确@机器人或叫机器人的名字，应该回复
2. 如果是私聊消息，通常应该回复
3. 如果消息是在问问题或寻求帮助，应该回复
4. 如果消息是无意义的闲聊且没有针对机器人，可以不回复
5. 如果消息是群里其他人之间的对话，通常不应该回复
6. 如果消息涉及机器人感兴趣的话题，可以考虑回复

请只回答"是"或"否"，不要有其他内容。"""


class HeartflowDecision:
    def __init__(self, context=None):
        self._context = context
        self._last_decision_time = 0.0
        self._decision_cache: Dict[str, Dict] = {}
        self._cache_ttl = 5.0

    async def should_reply(self, message: str, sender_name: str = "",
                           is_mentioned: bool = False, is_private: bool = False,
                           recent_context: str = "", stream_id: str = "") -> Dict[str, Any]:
        try:
            now = time.time()
            cache_key = f"{stream_id}:{hash(message[:50])}"
            if cache_key in self._decision_cache:
                cached = self._decision_cache[cache_key]
                if now - cached.get("timestamp", 0) < self._cache_ttl:
                    return {
                        "should_reply": cached.get("should_reply", False),
                        "confidence": cached.get("confidence", 0.5),
                        "reason": cached.get("reason", "缓存命中"),
                        "source": "cache",
                    }
            bot_name = self._get_bot_name()
            prompt = HEARTFLOW_DECISION_PROMPT.format(
                bot_name=bot_name,
                sender_name=sender_name or "未知用户",
                is_mentioned="是" if is_mentioned else "否",
                is_private="是" if is_private else "否",
                message=message[:500] if message else "(空消息)",
                recent_context=recent_context[:300] if recent_context else "(无)",
            )
            start_time = time.time()
            response_text = await self._call_llm(prompt)
            elapsed = time.time() - start_time
            should_reply = False
            confidence = 0.5
            reason = "LLM决策"
            if response_text:
                answer = response_text.strip().lower()
                if answer in ["是", "yes", "true", "1"]:
                    should_reply = True
                    confidence = 0.8
                    reason = "LLM判断应该回复"
                elif answer in ["否", "no", "false", "0"]:
                    should_reply = False
                    confidence = 0.8
                    reason = "LLM判断不应回复"
                else:
                    should_reply = "是" in answer or "yes" in answer
                    confidence = 0.6
                    reason = f"LLM回复解析: {answer[:20]}"
            else:
                return self._fallback_decision(message, is_mentioned, is_private)
            logger.info(f"心流决策: {'是' if should_reply else '否'} | 耗时{elapsed:.2f}s | {reason}")
            self._decision_cache[cache_key] = {
                "should_reply": should_reply, "confidence": confidence,
                "reason": reason, "timestamp": now,
            }
            self._cleanup_cache()
            return {
                "should_reply": should_reply, "confidence": confidence,
                "reason": reason, "source": "llm",
            }
        except Exception as e:
            logger.error(f"心流决策异常: {e}")
            return self._fallback_decision(message, is_mentioned, is_private)

    async def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            request = LLMRequest(model_config.lightweight, request_type="heartflow_decision")
            response_text, _ = await request.generate_response_async(prompt)
            return response_text
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return None

    def _get_bot_name(self) -> str:
        try:
            from src.config.config import global_config
            return global_config.bot.nickname
        except Exception:
            return "AI"

    def _fallback_decision(self, message: str, is_mentioned: bool, is_private: bool) -> Dict[str, Any]:
        should_reply = is_mentioned or is_private
        return {
            "should_reply": should_reply, "confidence": 0.5,
            "reason": "规则回退决策（is_mentioned或私聊）", "source": "fallback",
        }

    def _cleanup_cache(self):
        now = time.time()
        expired_keys = [
            k for k, v in self._decision_cache.items()
            if now - v.get("timestamp", 0) > self._cache_ttl * 2
        ]
        for k in expired_keys:
            del self._decision_cache[k]
        if len(self._decision_cache) > 100:
            sorted_items = sorted(
                self._decision_cache.items(),
                key=lambda x: x[1].get("timestamp", 0),
            )
            self._decision_cache = dict(sorted_items[-50:])


_heartflow_decision: Optional[HeartflowDecision] = None


def get_heartflow_decision() -> HeartflowDecision:
    global _heartflow_decision
    if _heartflow_decision is None:
        _heartflow_decision = HeartflowDecision()
    return _heartflow_decision


def reset_heartflow_decision():
    global _heartflow_decision
    _heartflow_decision = None


def _get_heartflow_provider():
    return get_heartflow_decision()


def set_context(context=None):
    decision = get_heartflow_decision()
    if context and hasattr(decision, '_context'):
        decision._context = context

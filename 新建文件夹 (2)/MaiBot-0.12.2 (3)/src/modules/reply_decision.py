import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.common.logger import get_logger

logger = get_logger("reply_decision")


@dataclass
class ReplyDecision:
    """回复决策结果"""
    probability: float  # 回复概率 0.0-1.0
    reply_style: str  # 回复方式: direct/quote
    reason: str  # 决策原因


DECISION_PROMPT = """你是一个群聊回复决策模型。分析以下聊天记录和最新消息，判断回复的必要性。

--- 最近的聊天记录 ---
{message_queue}

--- 最新消息 ---
{message_text}

--- 回复标准 ---
1. 被@或直接提问 → probability: 1.0
2. 轻松有趣话题 → probability: 0.6-0.8
3. AI/代码/技术相关 → probability: 0.5-0.7
4. 私人对话或@其他人 → probability: 0.0
5. 无意义闲聊/表情符号 → probability: 0.0-0.2

--- 回复方式 ---
- direct: 对话延续或开放性陈述
- quote: 明确问题或避免误解

请返回JSON格式：
{{"probability": <0.0-1.0>, "reply_style": "<direct或quote>", "reason": "<简短原因>"}}
"""


class ReplyDecisionMaker:
    """回复决策器"""
    
    def __init__(self):
        self._context = None
        from src.config.config import global_config
        self._bot_name = global_config.bot.nickname
        self._enabled = True
    
    def set_context(self, context):
        """设置上下文"""
        self._context = context
    
    def set_bot_name(self, name: str):
        """设置机器人名称"""
        self._bot_name = name
    
    async def decide(
        self,
        message_text: str,
        message_queue: List[str],
        sender_name: str = "",
    ) -> ReplyDecision:
        """
        判断是否需要回复以及回复方式
        
        Args:
            message_text: 最新消息
            message_queue: 最近的聊天记录
            sender_name: 发送者名称
            
        Returns:
            ReplyDecision: 回复决策结果
        """

        if not self._enabled:
            return ReplyDecision(
                probability=0.0,
                reply_style="normal",
                reason="LLM决策未启用，使用规则决策系统"
            )

        try:
            decision = await self._llm_decide(message_text, message_queue)
            if decision:
                return decision
        except Exception as e:
            logger.warning(f"LLM决策失败: {e}")
        
        # 默认决策
        return ReplyDecision(
            probability=0.3,
            reply_style="direct",
            reason="默认概率"
        )
    
    async def _llm_decide(
        self,
        message_text: str,
        message_queue: List[str],
    ) -> Optional[ReplyDecision]:
        """调用LLM进行决策"""
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            
            request = LLMRequest(model_config.lightweight, request_type="reply_decision")
            
            # 构建prompt
            queue_text = "\n".join(message_queue[-10:]) if message_queue else "无"
            prompt = DECISION_PROMPT.format(
                message_queue=queue_text,
                message_text=message_text,
            )
            
            response, _ = await request.generate_response_async(prompt, max_tokens=100)
            
            if not response:
                return None
            
            # 解析JSON
            text = response.strip()
            # 提取JSON部分
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
                return ReplyDecision(
                    probability=float(data.get("probability", 0.3)),
                    reply_style=data.get("reply_style", "direct"),
                    reason=data.get("reason", "LLM判断")
                )
        except json.JSONDecodeError:
            logger.warning("JSON解析失败")
        except Exception as e:
            logger.warning(f"LLM决策错误: {e}")
        
        return None
    
    def should_reply(self, decision: ReplyDecision, threshold: float = 0.5) -> bool:
        """根据决策结果判断是否应该回复"""
        return decision.probability >= threshold


_reply_decision_maker = None


def get_reply_decision_maker() -> ReplyDecisionMaker:
    """获取回复决策器单例"""
    global _reply_decision_maker
    if _reply_decision_maker is None:
        _reply_decision_maker = ReplyDecisionMaker()
    return _reply_decision_maker

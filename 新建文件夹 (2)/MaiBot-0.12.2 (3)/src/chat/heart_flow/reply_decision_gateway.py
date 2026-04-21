import time
import random
from typing import Tuple, Dict, Any, Optional
from src.common.logger import get_logger

logger = get_logger("decision_gateway")


class ReplyDecisionGateway:
    # 统一回复决策网关，所有回复决策必须经过此网关
    # 决策流程：强制触发 → 强制抑制 → 概率决策
    _instance: Optional["ReplyDecisionGateway"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._base_probability = 0.3
        self._last_reply_time: Dict[str, float] = {}
        self._reply_count: Dict[str, int] = {}
        self._consecutive_count: Dict[str, int] = {}
        self._cooldown_time = 5.0
        self._max_consecutive = 3
        self._initialized = True
        logger.info("统一决策网关已初始化 | 基础概率30%")

    def set_base_probability(self, probability: float):
        self._base_probability = max(0.0, min(1.0, probability))
        logger.info(f"基础回复概率已更新: {self._base_probability*100:.0f}%")

    async def should_reply(
        self,
        session_id: str,
        context: Dict[str, Any]
    ) -> Tuple[bool, str, float]:
        # 统一决策入口：强制触发 → 强制抑制 → 概率决策
        force_decision = self._check_force_trigger(context)
        if force_decision:
            return force_decision
        suppress_decision = self._check_suppress(session_id, context)
        if suppress_decision:
            return suppress_decision
        return self._probability_decision(session_id, context)

    def _check_force_trigger(self, context: Dict) -> Optional[Tuple[bool, str, float]]:
        # @提及、回复消息、管理员命令 → 100%回复
        if context.get("is_at_bot"):
            logger.debug("强制触发 | @提及")
            return True, "@提及", 1.0
        if context.get("is_reply_to_bot"):
            logger.debug("强制触发 | 回复消息")
            return True, "回复消息", 1.0
        return None

    def _check_suppress(self, session_id: str, context: Dict) -> Optional[Tuple[bool, str, float]]:
        # 冷却时间、连续回复上限、屏蔽 → 禁止回复
        now = time.time()
        last_time = self._last_reply_time.get(session_id, 0)
        if now - last_time < self._cooldown_time:
            remaining = self._cooldown_time - (now - last_time)
            logger.debug(f"冷却中 | 剩余{remaining:.1f}秒")
            return False, f"冷却中({remaining:.1f}s)", 0.0
        consecutive = self._consecutive_count.get(session_id, 0)
        if consecutive >= self._max_consecutive:
            logger.debug(f"连续回复上限 | 已连续{consecutive}次")
            return False, f"连续上限({consecutive}次)", 0.0
        if context.get("is_blocked"):
            logger.debug("用户已屏蔽")
            return False, "用户屏蔽", 0.0
        return None

    def _probability_decision(self, session_id: str, context: Dict) -> Tuple[bool, str, float]:
        base_prob = self._base_probability
        interest_score = context.get("interest_score", 0.5)
        interest_boost = interest_score * 0.2
        affection = context.get("favor", context.get("affection", 50))
        relationship_boost = (affection - 50) / 100 * 0.1
        boost_factor = context.get("boost_factor", 0.0)
        boost_addition = boost_factor * 0.1
        annoyance = context.get("annoyance", 0)
        annoyance_penalty = -annoyance * 0.003
        trauma = context.get("trauma", context.get("trauma_score", 0))
        trauma_penalty = -trauma * 0.03
        stamina = context.get("stamina", 100)
        stamina_penalty = -(100 - stamina) * 0.002
        final_prob = base_prob + interest_boost + relationship_boost + boost_addition + annoyance_penalty + trauma_penalty + stamina_penalty
        final_prob = max(0.0, min(1.0, final_prob))
        roll = random.random()
        should_reply = roll < final_prob
        if should_reply:
            logger.info(
                f"概率触发 | 概率{final_prob*100:.0f}% "
                f"(基础{base_prob*100:.0f}% +兴趣{interest_boost*100:.0f}% "
                f"+关系{relationship_boost*100:+.0f}% +优先级{boost_addition*100:.0f}%) "
                f"| 随机值{roll:.2f}"
            )
            return True, f"概率触发({final_prob:.2f})", final_prob
        else:
            logger.debug(f"概率未触发 | 概率{final_prob*100:.0f}% | 随机值{roll:.2f}")
            return False, f"概率未触发({final_prob:.2f})", 0.0

    def record_reply(self, session_id: str, user_replied: bool = False):
        now = time.time()
        self._last_reply_time[session_id] = now
        if user_replied:
            self._consecutive_count[session_id] = 0
            logger.debug("用户回复 | 重置连续计数")
        else:
            count = self._consecutive_count.get(session_id, 0) + 1
            self._consecutive_count[session_id] = count
            logger.debug(f"机器人回复 | 连续计数{count}")


def get_reply_decision_gateway() -> ReplyDecisionGateway:
    return ReplyDecisionGateway()

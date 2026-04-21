import random
import time as _tm
from typing import Optional

from src.common.logger import get_logger

logger = get_logger("rest_arbiter")


# ---------------------------------------------------------------------------
#  时段标签 & 系数
# ---------------------------------------------------------------------------

_PERIOD_TABLE = [
    # (start_hour, end_hour, label, multiplier)
    (2, 6, "deep_night", 1.4),
    (0, 2, "late_night", 1.2),
    (22, 24, "evening", 1.2),
    (6, 9, "morning", 1.0),
    (9, 18, "daytime", 1.0),
    (18, 22, "dusk", 1.0),
]

_LOW_ENERGY_MOODS = frozenset(["tired", "bored", "疲惫", "无聊", "烦躁"])


def _resolve_period(hour: int) -> tuple:
    """根据当前小时返回 (标签, 乘数)"""
    for lo, hi, label, mul in _PERIOD_TABLE:
        if lo <= hour < hi:
            return label, mul
    return "daytime", 1.0


# ---------------------------------------------------------------------------
#  规则基线估算
# ---------------------------------------------------------------------------


def _rule_baseline(
    chat_stamina: float,
    brain_reserve: float,
    streak_count: int,
    mood_tag: str,
) -> int:
    """纯规则计算休息秒数。返回 0 表示无需休息。"""
    # 精力充足且连续回复少 → 不需要休息
    if chat_stamina > 50 and streak_count < 3:
        return 0
    # 按精力分档
    if chat_stamina > 20:
        lo, hi = 30, 90
    elif chat_stamina > 0:
        lo, hi = 60, 180
    else:
        lo, hi = 90, 240
    # 思考值过低
    if brain_reserve < 10:
        lo = max(lo, 120)
        hi = max(hi, 300)
    # 连续回复过多
    if streak_count >= 6:
        lo = max(lo, 90)
        hi = max(hi, 240)
    # 时段修正
    hour = _tm.localtime().tm_hour
    _label, multiplier = _resolve_period(hour)
    lo = int(lo * multiplier)
    hi = int(hi * multiplier)
    # 低能量心情修正
    if mood_tag in _LOW_ENERGY_MOODS:
        lo = int(lo * 1.2)
        hi = int(hi * 1.2)
    duration = random.randint(lo, hi)
    return max(30, min(300, duration))


# ---------------------------------------------------------------------------
#  仲裁器
# ---------------------------------------------------------------------------


class RestArbiter:
    """休息时长仲裁器"""

    @staticmethod
    def estimate(
        chat_stamina: float,
        brain_reserve: float,
        streak_count: int,
        mood_tag: str = "",
    ) -> int:
        """纯规则估算休息秒数（0 = 不休息）"""
        seconds = _rule_baseline(
            chat_stamina, brain_reserve, streak_count, mood_tag
        )
        hour = _tm.localtime().tm_hour
        period_label, _ = _resolve_period(hour)
        logger.info(
            f"[休息仲裁] 规则估算: {seconds}s | "
            f"精力={chat_stamina:.0f} 思考值={brain_reserve:.0f} "
            f"连续={streak_count} 时段={period_label}"
        )
        return seconds

    @staticmethod
    async def estimate_with_llm(
        chat_stamina: float,
        brain_reserve: float,
        streak_count: int,
        mood_tag: str = "",
        recent_thought: str = "",
        dialogue_digest: str = "",
    ) -> int:
        """先走规则基线，再可选用 LLM 精化"""
        baseline = _rule_baseline(
            chat_stamina, brain_reserve, streak_count, mood_tag
        )
        if baseline == 0:
            return 0
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            hour = _tm.localtime().tm_hour
            period_label, _ = _resolve_period(hour)
            stimulus = (
                "请估算应该休息多少秒（30~300之间），只回复一个整数。\n"
                f"聊天精力: {chat_stamina:.0f}, 思考值: {brain_reserve:.0f}\n"
                f"连续回复: {streak_count}, 心情: {mood_tag or '一般'}\n"
                f"时段: {period_label}({hour}时)\n"
                f"近期想法: {(recent_thought or '无')[:100]}\n"
                f"对话摘要: {(dialogue_digest or '无')[:100]}\n"
                f"规则建议: {baseline}s (参考范围 {baseline}~{min(baseline * 2, 300)}s)"
            )
            req = LLMRequest(model_config.model_task_config.utils, request_type="rest_duration")
            raw, _ = await req.generate_response_async(stimulus)
            if raw:
                token = raw.strip().split()[0]
                val = int(token)
                # 如果 LLM 返回的是分钟数
                if val < 10:
                    val = val * 60
                result = max(30, min(300, val))
                logger.info(f"[休息仲裁] LLM精化: {result}s")
                return result
        except Exception as exc:
            logger.warning(f"[休息仲裁] LLM精化失败: {exc}，使用规则基线")
        return baseline


# ---------------------------------------------------------------------------
#  模块单例
# ---------------------------------------------------------------------------

_arbiter_instance: Optional[RestArbiter] = None


def acquire_rest_arbiter() -> RestArbiter:
    """获取全局 RestArbiter 单例"""
    global _arbiter_instance
    if _arbiter_instance is None:
        _arbiter_instance = RestArbiter()
    return _arbiter_instance

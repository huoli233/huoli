import random
import time as _tm
from typing import Optional

from src.common.logger import get_logger

logger = get_logger("rest_timing")


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
#  休息时长评估器
# ---------------------------------------------------------------------------


class RestTimingEngine:
    """休息时长评估器"""

    @staticmethod
    def estimate_reply_cooldown(
        unanswered_turns: int,
        voice_desire: int,
        emotional_drive: float,
        recent_reply_burst: int = 0,
    ) -> float:
        """估算主动说话冷却，统一承接主链的高频发言约束。"""
        if emotional_drive > 0.6 and voice_desire >= 5:
            return max(90.0, 240.0 / max(voice_desire / 10.0, 0.1))
        if voice_desire >= 7:
            return max(120.0, 120.0 * max(1, unanswered_turns))
        if unanswered_turns >= 8 and voice_desire < 4:
            return 600.0
        base_cooldown = unanswered_turns**2 * 30.0
        desire_factor = max(0.3, voice_desire / 10.0)
        emotion_factor = max(0.5, 0.5 + emotional_drive * 0.5)
        adaptive_cooldown = base_cooldown / (desire_factor * emotion_factor)
        if recent_reply_burst >= 4:
            adaptive_cooldown *= 1.5
        return max(90.0, min(900.0, adaptive_cooldown * 1.8))

    @staticmethod
    def estimate(
        chat_stamina: float,
        brain_reserve: float,
        streak_count: int,
        mood_tag: str = "",
    ) -> int:
        """纯规则估算休息秒数（0 = 不休息）"""
        seconds = _rule_baseline(chat_stamina, brain_reserve, streak_count, mood_tag)
        hour = _tm.localtime().tm_hour
        period_label, _ = _resolve_period(hour)
        logger.info(
            f"[休息评估] 规则估算: {seconds}s | "
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
        """先走规则基线，再用本地规则微调，不调LLM。"""
        baseline = _rule_baseline(chat_stamina, brain_reserve, streak_count, mood_tag)
        if baseline == 0:
            return 0
        result = baseline
        hour = _tm.localtime().tm_hour
        if hour >= 23 or hour < 6:
            result = min(300, int(result * 1.5))
        elif hour >= 22:
            result = min(300, int(result * 1.2))
        if chat_stamina < 30:
            result = min(300, int(result * 1.3))
        if streak_count >= 4:
            result = min(300, int(result * 1.2))
        result = max(30, min(300, result))
        logger.info(f"[休息评估] 本地规则精化: {result}s (基线={baseline}s)")
        return result


# ---------------------------------------------------------------------------
#  模块单例
# ---------------------------------------------------------------------------

_timing_engine_instance: Optional[RestTimingEngine] = None


def acquire_rest_timing_engine() -> RestTimingEngine:
    """获取全局 RestTimingEngine 单例"""
    global _timing_engine_instance
    if _timing_engine_instance is None:
        _timing_engine_instance = RestTimingEngine()
    return _timing_engine_instance

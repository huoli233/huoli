import random
import time as _tm
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger

logger = get_logger("idle_reflection")


# ---------------------------------------------------------------------------
#  数据载体
# ---------------------------------------------------------------------------


@dataclass
class IdleReflection:
    """一次被动反射的结果"""

    reflection_text: str
    emotional_tone: str
    certainty: float = 0.7
    category: str = "passive"
    generated_at: float = field(default_factory=_tm.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.reflection_text,
            "tone": self.emotional_tone,
            "certainty": self.certainty,
            "category": self.category,
            "ts": self.generated_at,
        }


# ---------------------------------------------------------------------------
#  情绪音调档位映射
# ---------------------------------------------------------------------------

_TONE_THRESHOLDS: List[tuple] = [
    (0.75, "upbeat"),
    (0.50, "neutral"),
    (0.25, "subdued"),
]
_TONE_FLOOR = "gloomy"


def _gauge_emotional_tone(emotion_snapshot: Dict) -> str:
    """从情绪快照中推断一个简单的音调标签

    使用 average_mood_score 查表，找不到就回落到最低档。
    """
    score = emotion_snapshot.get("average_mood_score", 0.5)
    for threshold, label in _TONE_THRESHOLDS:
        if score >= threshold:
            return label
    return _TONE_FLOOR


# ---------------------------------------------------------------------------
#  静态兜底反射库（按音调分组）
# ---------------------------------------------------------------------------

_FALLBACK_POOL: Dict[str, List[str]] = {
    "upbeat": [
        "心情不错，随便看看消息",
        "今天精神还挺好的",
        "有什么有趣的消息吗",
    ],
    "neutral": [
        "随便看看吧",
        "嗯，消息来了",
        "先瞅一眼",
    ],
    "subdued": [
        "有点累了不过还是看看",
        "心情一般般",
        "先看看，不一定回",
    ],
    "gloomy": [
        "不太想理人",
        "没什么精力",
        "算了先不管了",
    ],
}


def _pick_fallback(tone: str, favor: float) -> str:
    """根据音调和好感选一条兜底碎念"""
    pool = _FALLBACK_POOL.get(tone, _FALLBACK_POOL["neutral"])
    if favor >= 50 and tone in ("subdued", "gloomy"):
        # 好感高时稍微积极一点
        pool = _FALLBACK_POOL.get("neutral", pool)
    return random.choice(pool)


# ---------------------------------------------------------------------------
#  生成器主体
# ---------------------------------------------------------------------------


class IdleReflectionProducer:
    """被动反射思维生产者

    典型调用场景：
    - 收到消息但当前状态不需要正式回复（窥屏/旁观）
    - 主循环切到等待或休息阶段的过渡时刻
    """

    def __init__(self):
        self._llm_bridge = None

    def attach_llm_bridge(self, bridge):
        """注入外部 LLM 桥接对象（用于扩展）"""
        self._llm_bridge = bridge

    async def produce_reflection(
        self,
        incoming_text: str,
        sender_id: str,
        emotion_snapshot: Dict,
        recent_dialogue: Optional[List] = None,
        extra_ctx: Optional[Dict] = None,
    ) -> IdleReflection:
        """生成一条被动碎念反射

        优先走 LLM；LLM 不可用或返回空则走兜底逻辑。
        """
        tone = _gauge_emotional_tone(emotion_snapshot)
        favor = emotion_snapshot.get(
            "favor", emotion_snapshot.get("affection", 0)
        )
        try:
            stimulus = self._compose_stimulus(
                incoming_text,
                sender_id,
                emotion_snapshot,
                recent_dialogue,
                extra_ctx,
            )
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            req = LLMRequest(
                model_config.model_task_config.focus_chat,
                request_type="idle_reflection",
            )
            raw_output, _ = await req.generate_response_async(stimulus)
            if not raw_output or not raw_output.strip():
                logger.warning("[被动反射] LLM回空，走兜底")
                return self._fallback_reflection(incoming_text, tone, favor)
            cleaned = raw_output.strip()[:120]
            logger.info(f"[被动反射] {cleaned[:40]}...")
            return IdleReflection(
                reflection_text=cleaned,
                emotional_tone=tone,
                certainty=0.8,
                category="passive",
            )
        except Exception as exc:
            logger.error(f"[被动反射] 生成异常: {exc}")
            return self._fallback_reflection(incoming_text, tone, favor)

    # ---- 内部方法 ----

    def _compose_stimulus(
        self,
        incoming_text: str,
        sender_id: str,
        emotion_snapshot: Dict,
        recent_dialogue: Optional[List],
        extra_ctx: Optional[Dict],
    ) -> str:
        """拼装发给 LLM 的激励文本"""
        favor = emotion_snapshot.get(
            "favor", emotion_snapshot.get("affection", 0)
        )
        annoyance = emotion_snapshot.get("annoyance", 0)
        avg_mood = emotion_snapshot.get("average_mood_score", 0.5)
        energy_pct = emotion_snapshot.get("energy", 50)
        # 近期对话摘要
        dialogue_clip = ""
        if recent_dialogue:
            snippet_lines = []
            for msg in recent_dialogue[-5:]:
                who = msg.get("sender", "?")
                what = msg.get("content", "")[:50]
                snippet_lines.append(f"  {who}: {what}")
            dialogue_clip = "\n".join(snippet_lines)
        parts = [
            "你现在不需要正式回复，只需要产出一段简短的内心碎念（一句话）。",
            f"收到的消息: {incoming_text[:80]}",
            f"当前心情: {avg_mood:.1f}/1.0，好感度: {favor}，烦躁度: {annoyance}，精力: {energy_pct}%",
        ]
        if dialogue_clip:
            parts.append(f"最近对话:\n{dialogue_clip}")
        parts.append("请只输出一句话的碎念，不要加任何标记或JSON。")
        return "\n".join(parts)

    @staticmethod
    def _fallback_reflection(
        incoming_text: str,
        tone: str,
        favor: float,
    ) -> IdleReflection:
        """无 LLM 可用时的兜底生成"""
        text = _pick_fallback(tone, favor)
        return IdleReflection(
            reflection_text=text,
            emotional_tone=tone,
            certainty=0.5,
            category="passive",
        )


# ---------------------------------------------------------------------------
#  模块单例
# ---------------------------------------------------------------------------

_idle_producer: Optional[IdleReflectionProducer] = None


def acquire_idle_reflection_producer() -> IdleReflectionProducer:
    """获取全局 IdleReflectionProducer 单例"""
    global _idle_producer
    if _idle_producer is None:
        _idle_producer = IdleReflectionProducer()
    return _idle_producer

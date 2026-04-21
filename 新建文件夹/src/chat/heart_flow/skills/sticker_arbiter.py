import random
import time as _tm
from typing import Optional, Tuple

from src.common.logger import get_logger

logger = get_logger("sticker_arbiter")


class StickerArbiter:
    """表情包发送仲裁器"""

    # 默认参数
    _COOLDOWN_SEC: float = 120.0
    _DAILY_CEILING: int = 20

    def __init__(self):
        self._prev_dispatch_ts: float = 0.0
        self._today_dispatched: int = 0
        self._today_date_tag: str = ""

    # ----------------------------------------------------------------
    #  核心决策
    # ----------------------------------------------------------------

    async def should_dispatch(
        self,
        dialogue_excerpt: str,
        current_mood: str,
        scene_kind: str,
        peer_id: str,
        channel_id: str,
        chat_stamina: float = 100.0,
    ) -> Tuple[bool, str]:
        """判断此刻是否应该发送表情包

        返回 (是否发送, 理由文本)。
        """
        now = _tm.time()
        # 冷却检测
        if now - self._prev_dispatch_ts < self._COOLDOWN_SEC:
            return False, "冷却中"
        # 日限额检测
        date_tag = _tm.strftime("%Y-%m-%d")
        if self._today_date_tag != date_tag:
            self._today_date_tag = date_tag
            self._today_dispatched = 0
        if self._today_dispatched >= self._DAILY_CEILING:
            return False, "今日额度已用尽"
        # 精力过低
        if chat_stamina < 20:
            return False, "聊天精力不足"
        # 接入行为评判系统
        try:
            from src.chat.heart_flow.skills.action_evaluator import (
                acquire_action_judge,
                ActionCategory,
                EvaluationInputs,
                VerdictGrade,
            )

            judge = acquire_action_judge()
            ctx_score = await self._probe_timing_relevance(
                dialogue_excerpt, current_mood
            )
            # 尝试获取心理数据
            aff, trs, irr = 0.0, 0.0, 0.0
            try:
                from src.chat.heart_flow.fondness_trust import FondnessTrustDimension
from src.chat.heart_flow.social_value_dim import SocialValueDimension

                _fuser = FondnessTrustDimension.get_instance()
                _snap = await _fuser.get_snapshot(peer_id, channel_id)
                _sc = _snap.social_score
                if _sc >= 0:
                    aff = _sc
                    trs = _sc * _snap.phase_weight
                    irr = 0.0
                else:
                    aff = 0.0
                    trs = 0.0
                    irr = abs(_sc)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} 异常: {_e}")
            inp = EvaluationInputs(
                affection=aff,
                trust_level=trs,
                irritation=irr,
                scene_kind=scene_kind,
                relevance=ctx_score,
                timing_quality=ctx_score,
                dialogue_rounds=len(dialogue_excerpt.split("\n")),
                peer_activity=0.7,
                chat_stamina=chat_stamina,
            )
            verdict = judge.evaluate(ActionCategory.EMIT_STICKER, inp)
            logger.info(
                f"[表情仲裁] 评分={verdict.composite_score:.1f} "
                f"等级={verdict.grade.value} 理由={verdict.rationale}"
            )
            if verdict.grade in (
                VerdictGrade.IMPERATIVE,
                VerdictGrade.FAVORABLE,
            ):
                self._prev_dispatch_ts = now
                self._today_dispatched += 1
                return True, verdict.rationale
            return False, verdict.rationale
        except Exception as exc:
            logger.error(f"[表情仲裁] 决策异常: {exc}")
            return False, f"决策异常: {exc}"

    # ----------------------------------------------------------------
    #  表情包选取
    # ----------------------------------------------------------------

    async def pick_sticker(
        self,
        mood: str,
        dialogue_excerpt: str,
    ) -> Optional[Tuple[str, str, str]]:
        """从表情库中选择一张合适的表情包

        返回 (文件路径, 描述, 标签) 或 None。
        """
        try:
            from src.chat.emoji_system.emoji_manager import StickerVault

            vault = StickerVault()
            if not vault._initialized:
                await vault.load_registry()
            match = await vault.find_sticker_by_mood(mood)
            if match:
                logger.info(f"[表情选取] 按心情匹配: {match[2]} → {match[1]}")
                return match
            fallback = await vault.get_random_sticker()
            if fallback:
                logger.info(f"[表情选取] 随机选取: {fallback[1]}")
                return fallback
            return None
        except Exception as exc:
            logger.error(f"[表情选取] 失败: {exc}")
            return None

    # ----------------------------------------------------------------
    #  表情包收集
    # ----------------------------------------------------------------

    async def harvest_from_message(self, image_url: str) -> bool:
        """从消息中收集一张新的表情包"""
        try:
            from src.chat.emoji_system.emoji_manager import StickerVault

            vault = StickerVault()
            if not vault._initialized:
                await vault.load_registry()
            ok = await vault.enroll_sticker_from_url(image_url)
            if ok:
                logger.info(f"[表情收集] 成功: {image_url[:50]}")
            return ok
        except Exception as exc:
            logger.error(f"[表情收集] 失败: {exc}")
            return False

    # ----------------------------------------------------------------
    #  内部
    # ----------------------------------------------------------------

    async def _probe_timing_relevance(
        self, dialogue_excerpt: str, mood: str
    ) -> float:
        """用 LLM 探测当前是否适合发表情（返回 0~1）"""
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            stimulus = (
                "请评估当前对话中发出表情包的适合度，只回复一个 0~1 的小数。\n"
                f"心情: {mood}\n最近对话:\n{dialogue_excerpt[-200:]}"
            )
            req = LLMRequest(model_config.model_task_config.utils, request_type="sticker_timing")
            raw, _ = await req.generate_response_async(stimulus)
            if raw:
                return max(0.0, min(1.0, float(raw.strip())))
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return 0.5


# ---------------------------------------------------------------------------
#  模块单例
# ---------------------------------------------------------------------------

_arbiter: Optional[StickerArbiter] = None


def acquire_sticker_arbiter() -> StickerArbiter:
    """获取全局 StickerArbiter 单例"""
    global _arbiter
    if _arbiter is None:
        _arbiter = StickerArbiter()
    return _arbiter

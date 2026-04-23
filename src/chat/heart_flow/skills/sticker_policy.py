import random
import time as _tm
import asyncio
import base64
import urllib.request
from typing import Optional, Tuple

from src.common.logger import get_logger

logger = get_logger("sticker_policy")


def _skill_view() -> dict:
    try:
        from src.config.core_config_engine import get_core_config

        return get_core_config().resolve_module_view("skill").values
    except Exception:
        return {}


class StickerPolicyEngine:
    """表情包发送策略引擎"""

    def __init__(self):
        self._prev_dispatch_ts: float = 0.0
        self._today_dispatched: int = 0
        self._today_date_tag: str = ""
        skill_view = _skill_view()
        self._cooldown_sec: float = float(skill_view.get("sticker_dispatch_cooldown_seconds", 120.0))
        self._daily_ceiling: int = int(skill_view.get("sticker_daily_ceiling", 20))
        self._trigger_threshold: float = float(skill_view.get("sticker_dispatch_threshold", 0.55))

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
        if now - self._prev_dispatch_ts < self._cooldown_sec:
            return False, "冷却中"
        # 日限额检测
        date_tag = _tm.strftime("%Y-%m-%d")
        if self._today_date_tag != date_tag:
            self._today_date_tag = date_tag
            self._today_dispatched = 0
        if self._today_dispatched >= self._daily_ceiling:
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
            ctx_score = await self._probe_timing_relevance(dialogue_excerpt, current_mood)
            if ctx_score < self._trigger_threshold:
                return False, f"触发分不足({ctx_score:.2f}<{self._trigger_threshold:.2f})"
            # 尝试获取心理数据
            aff, trs, irr = 0.0, 0.0, 0.0
            try:
                from src.chat.heart_flow.fondness_trust import FondnessTrustDimension

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
                logger.debug(f"[表情策略] 异常: {_e}")
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
                f"[表情策略] 评分={verdict.composite_score:.1f} 等级={verdict.grade.value} 理由={verdict.rationale}"
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
            logger.error(f"[表情策略] 决策异常: {exc}")
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
            from src.chat.emoji_system.emoji_manager import get_emoji_manager

            manager = get_emoji_manager()
            if not getattr(manager, "_initialized", False):
                manager.initialize()
            if not getattr(manager, "emoji_objects", None):
                await manager.get_all_emoji_from_db()
            match = await manager.get_emoji_for_text(mood)
            if match:
                logger.info(f"[表情选取] 按心情匹配: {match[2]} → {match[1]}")
                return match
            valid_pool = [item for item in manager.emoji_objects if not getattr(item, "is_deleted", False)]
            fallback = None
            if valid_pool:
                picked = random.choice(valid_pool)
                matched_emotion = random.choice(picked.emotion) if picked.emotion else "随机表情"
                fallback = (picked.full_path, f"[ {picked.description} ]", matched_emotion)
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
            from src.plugin_system.apis.emoji_api import register_emoji

            if not str(image_url or "").strip():
                return False

            def _download_bytes(url: str) -> bytes:
                with urllib.request.urlopen(url, timeout=8) as resp:
                    return resp.read()

            image_bytes = await asyncio.to_thread(
                _download_bytes,
                str(image_url).strip(),
            )
            if not image_bytes:
                return False
            encoded = base64.b64encode(image_bytes).decode("ascii")
            result = await register_emoji(encoded)
            ok = bool(result.get("success", False)) if isinstance(result, dict) else False
            if ok:
                logger.info(f"[表情收集] 成功: {image_url[:50]}")
            return ok
        except Exception as exc:
            logger.error(f"[表情收集] 失败: {exc}")
            return False

    # ----------------------------------------------------------------
    #  内部
    # ----------------------------------------------------------------

    async def _probe_timing_relevance(self, dialogue_excerpt: str, mood: str) -> float:
        """用本地规则评估当前是否适合发表情（返回 0~1），不调LLM。"""
        try:
            score = 0.5
            if not dialogue_excerpt:
                return 0.3
            _len = len(dialogue_excerpt)
            if _len > 50:
                score += 0.1
            if _len > 150:
                score += 0.1
            _mood_lower = (mood or "").lower()
            if any(w in _mood_lower for w in ["开心", "快乐", "笑", "happy", "joy"]):
                score += 0.2
            elif any(w in _mood_lower for w in ["无聊", "bored", "平淡"]):
                score += 0.1
            elif any(w in _mood_lower for w in ["生气", "angry", "烦躁"]):
                score -= 0.15
            _exc_marks = dialogue_excerpt.count("！") + dialogue_excerpt.count("!")
            if _exc_marks >= 2:
                score += 0.1
            _emoji_count = sum(1 for c in dialogue_excerpt if ord(c) > 0x1F000)
            if _emoji_count >= 2:
                score += 0.1
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5


# ---------------------------------------------------------------------------
#  模块单例
# ---------------------------------------------------------------------------

_policy_engine: Optional[StickerPolicyEngine] = None


def acquire_sticker_policy_engine() -> StickerPolicyEngine:
    """获取全局 StickerPolicyEngine 单例"""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = StickerPolicyEngine()
    return _policy_engine

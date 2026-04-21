import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger

logger = get_logger("内在驱动")


# ---------------------------------------------------------------------------
#  脉搏快照
# ---------------------------------------------------------------------------


@dataclass
class UrgeSnapshot:
    """单频道的情感脉搏快照。"""

    tedium: float = 0.0
    isolation_buildup: float = 0.0
    companionship_pull: float = 0.5
    temper_index: float = 0.5
    vitality: float = 1.0
    assertiveness: float = 0.5
    last_touch_ts: float = 0.0


# ---------------------------------------------------------------------------
#  描述文本生成
# ---------------------------------------------------------------------------


def _describe_tedium(val: float) -> str:
    if val >= 0.8:
        return "极度无聊，急需新鲜刺激"
    if val >= 0.6:
        return "比较无聊，想找人聊天"
    if val >= 0.4:
        return "有点闲，但还能忍"
    if val >= 0.2:
        return "稍微有点无事可做"
    return "完全不觉得无聊"


def _describe_isolation(val: float) -> str:
    if val >= 0.8:
        return "非常孤独，渴望交流"
    if val >= 0.6:
        return "比较寂寞，希望有人陪"
    if val >= 0.4:
        return "有一点孤独感"
    if val >= 0.2:
        return "略感寂寞"
    return "不觉得孤独"


def _describe_companionship(val: float) -> str:
    if val >= 0.8:
        return "很想找人说话"
    if val >= 0.6:
        return "比较想社交"
    if val >= 0.4:
        return "对聊天有些兴趣"
    if val >= 0.2:
        return "社交兴趣不高"
    return "不太想社交"


def _describe_temper(val: float) -> str:
    if val >= 0.8:
        return "心情很棒，充满活力"
    if val >= 0.6:
        return "心情不错"
    if val >= 0.4:
        return "心情平淡，夹杂小情绪"
    if val >= 0.2:
        return "心情有些低落"
    return "心情很差"


def _describe_vitality(val: float) -> str:
    if val >= 0.8:
        return "精力充沛"
    if val >= 0.6:
        return "精力还行"
    if val >= 0.4:
        return "精力一般"
    if val >= 0.2:
        return "有点疲倦"
    return "非常疲惫"


def _compose_inner_monologue(snap: UrgeSnapshot) -> str:
    """根据脉搏快照生成一段内心独白式描述。"""
    parts: List[str] = []
    if snap.tedium > 0.6:
        parts.append("感觉有些无聊，想找点有趣的事")
    if snap.isolation_buildup > 0.5:
        parts.append("有点寂寞，想和人聊天")
    if snap.companionship_pull > 0.6:
        parts.append("很想和大家说说话")
    if snap.temper_index > 0.7:
        parts.append("心情不错，想把开心分享出去")
    elif snap.temper_index < 0.3:
        parts.append("心情不太好，或许需要一点温暖")
    if snap.vitality < 0.3:
        parts.append("感觉有点累，精力不太够")
    return "；".join(parts) if parts else "内心平静，没有特别强烈的感受"


# ---------------------------------------------------------------------------
#  提示词构建
# ---------------------------------------------------------------------------


def _build_quick_feeling_prompt(snap: UrgeSnapshot, silence_min: float) -> str:
    """快速 LLM 决策：'我现在想说话吗？'"""
    try:
        from src.config.config import global_config

        bot_name = global_config.bot.nickname
        persona_desc = global_config.personality.personality
    except Exception:
        bot_name = "AI"
        persona_desc = ""
    from src.config.prompt_loader import get_prompt, PromptCategory

    return get_prompt(
        PromptCategory.MODULE,
        "proactive",
        "emotion_proactive.template",
        character_name=bot_name,
        personality=persona_desc,
        boredom_level=snap.tedium,
        loneliness_accumulation=snap.isolation_buildup,
        social_desire=snap.companionship_pull,
        mood_score=snap.temper_index,
        energy_level=snap.vitality,
        silence_minutes=silence_min,
    )


def _build_detailed_decision_prompt(
    snap: UrgeSnapshot,
    silence_min: float,
    channel_type: str,
    recent_topics: str,
    vibe: str,
    **extra,
) -> str:
    """增强版决策提示词（含人格/话题/氛围）。"""
    try:
        from src.config.config import global_config

        bot_name = global_config.bot.nickname
        persona_desc = global_config.personality.personality
        bot_age = getattr(global_config.personality, "character_age", 0)
        bot_hobbies = getattr(
            global_config.personality, "character_hobbies", ""
        )
    except Exception:
        bot_name, persona_desc, bot_age, bot_hobbies = "AI", "", 0, ""
    monologue = _compose_inner_monologue(snap)
    from src.config.prompt_loader import get_prompt, PromptCategory

    return get_prompt(
        PromptCategory.MODULE,
        "proactive",
        "enhanced_emotion_decision.template",
        character_name=bot_name,
        character_age=bot_age,
        personality=persona_desc,
        hobbies=bot_hobbies,
        boredom_level=f"{snap.tedium:.2f}",
        boredom_description=_describe_tedium(snap.tedium),
        loneliness_accumulation=f"{snap.isolation_buildup:.2f}",
        loneliness_description=_describe_isolation(snap.isolation_buildup),
        social_desire=f"{snap.companionship_pull:.2f}",
        social_description=_describe_companionship(snap.companionship_pull),
        mood_score=f"{snap.temper_index:.2f}",
        mood_description=_describe_temper(snap.temper_index),
        energy_level=f"{snap.vitality:.2f}",
        energy_description=_describe_vitality(snap.vitality),
        chat_type=channel_type,
        silence_minutes=f"{silence_min:.1f}",
        recent_topics=recent_topics,
        atmosphere=vibe,
        inner_state_analysis=monologue,
    )


def _build_private_resonance_prompt(
    snap: UrgeSnapshot,
    silence_min: float,
    user_name: str,
    **extra,
) -> str:
    """私聊场景的情感共鸣提示词。"""
    from src.config.prompt_loader import get_prompt, PromptCategory

    return get_prompt(
        PromptCategory.MODULE,
        "proactive",
        "private_emotional_resonance.template",
        user_name=user_name,
        affection=extra.get("favor", extra.get("affection", 0)),
        trust=extra.get("trust_value", extra.get("trust", 0)),
        relationship=extra.get("relationship", "普通"),
        last_interaction=extra.get("last_interaction", "普通对话"),
        silence_minutes=f"{silence_min:.1f}",
        missing_level=min(10, int(snap.isolation_buildup * 10)),
        worry_level=min(10, int((1.0 - snap.temper_index) * 10)),
        sharing_desire=min(10, int(snap.companionship_pull * 10)),
        social_need=min(10, int((snap.tedium + snap.isolation_buildup) * 5)),
        recent_mood_state=_describe_temper(snap.temper_index),
        emotional_reflection=_compose_inner_monologue(snap),
    )


def assemble_urge_prompt(
    snap: UrgeSnapshot,
    silence_min: float,
    channel_type: str = "group",
    recent_topics: str = "",
    user_name: str = "",
    vibe: str = "平静",
    **kw,
) -> str:
    """高层入口：根据频道类型选择最合适的提示词模板。"""
    if channel_type == "private":
        return _build_private_resonance_prompt(
            snap, silence_min, user_name, **kw
        )
    return _build_detailed_decision_prompt(
        snap,
        silence_min,
        channel_type,
        recent_topics,
        vibe,
        **kw,
    )


# ---------------------------------------------------------------------------
#  群聊氛围探测（对标 MaiBot GROUP_ATMOSPHERE_SENSING）
# ---------------------------------------------------------------------------


def compose_vibe_probe_prompt(
    recent_msgs: str,
    activity_lv: str,
    participant_ct: int,
    topic_kind: str,
    silence_min: float,
) -> str:
    """生成群聊氛围感知提示词。"""
    from src.config.prompt_loader import get_prompt, PromptCategory

    return get_prompt(
        PromptCategory.MODULE,
        "proactive",
        "group_atmosphere_sensing.template",
        recent_messages=recent_msgs,
        activity_level=activity_lv,
        participant_count=participant_ct,
        topic_type=topic_kind,
        silence_duration=f"{silence_min:.0f}",
    )


# ---------------------------------------------------------------------------
#  内在驱力追踪器
# ---------------------------------------------------------------------------


class InnerUrgeTracker:
    """基于情感脉搏的自然主动发言系统。

    核心理念：
    1. 用户停止互动后延迟数分钟开始情感监控
    2. 定期累积厌倦/孤独等指标并整合外部数据
    3. 综合意愿超过阈值后由 LLM 最终确认
    """

    _sole: Optional["InnerUrgeTracker"] = None

    @classmethod
    def sole(cls) -> "InnerUrgeTracker":
        if cls._sole is None:
            cls._sole = cls()
        return cls._sole

    def __init__(self):
        self._snapshots: Dict[str, UrgeSnapshot] = {}
        self._watch_tasks: Dict[str, asyncio.Task] = {}
        self._channel_locks: Dict[str, asyncio.Lock] = {}
        self._alive = True
        # 可调参数
        self.ONSET_DELAY_MIN = 3
        self.POLL_INTERVAL_SEC = 60
        self.TEDIUM_RATE = 0.1
        self.ISOLATION_RATE = 0.05
        self.ASSERTIVENESS_BAR = 0.7
        logger.info("[驱力追踪] 内在驱力追踪器已初始化")

    # ==================================================================
    #  外部事件接口
    # ==================================================================

    async def on_user_input(self, channel_id: str, user_id: str = "") -> None:
        """用户发消息时重置情感状态并重启监控。"""
        lock = self._channel_locks.get(channel_id)
        if lock is None:
            lock = asyncio.Lock()
            self._channel_locks[channel_id] = lock
        async with lock:
            try:
                await self._halt_watch(channel_id)
                snap = self._touch(channel_id)
                snap.last_touch_ts = time.time()
                snap.tedium = 0.0
                snap.isolation_buildup = 0.0
                delay_s = self.ONSET_DELAY_MIN * 60
                task = asyncio.create_task(
                    self._deferred_watch(channel_id, delay_s),
                    name=f"urge_watch_{channel_id[:8]}",
                )
                self._watch_tasks[channel_id] = task
                logger.debug(
                    f"[驱力追踪] {channel_id[:8]} 收到输入，{self.ONSET_DELAY_MIN}min 后启动监控"
                )
            except Exception as exc:
                logger.error(f"[驱力追踪] on_user_input 异常: {exc}")

    async def on_bot_output(
        self, channel_id: str, was_proactive: bool = False
    ) -> None:
        """Bot 发送消息后调整脉搏并重启监控。"""
        lock = self._channel_locks.get(channel_id)
        if lock is None:
            lock = asyncio.Lock()
            self._channel_locks[channel_id] = lock
        async with lock:
            try:
                snap = self._touch(channel_id)
                snap.last_touch_ts = time.time()
                if was_proactive:
                    snap.companionship_pull = max(
                        0.0, snap.companionship_pull - 0.3
                    )
                    snap.tedium = max(0.0, snap.tedium - 0.5)
                    snap.isolation_buildup = max(
                        0.0, snap.isolation_buildup - 0.4
                    )
                    logger.debug(
                        f"[驱力追踪] {channel_id[:8]} 主动发言后情感已回调"
                    )
                    await self._halt_watch(channel_id)
                    delay_s = self.ONSET_DELAY_MIN * 60
                    task = asyncio.create_task(
                        self._deferred_watch(channel_id, delay_s),
                        name=f"urge_watch_{channel_id[:8]}",
                    )
                    self._watch_tasks[channel_id] = task
            except Exception as exc:
                logger.error(f"[驱力追踪] on_bot_output 异常: {exc}")

    # ==================================================================
    #  监控循环
    # ==================================================================

    async def _deferred_watch(self, channel_id: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            await self._watch_loop(channel_id)
        except asyncio.CancelledError:
            logger.debug(f"[驱力追踪] {channel_id[:8]} 监控被取消")
        except Exception as exc:
            logger.error(f"[驱力追踪] {channel_id[:8]} 监控异常: {exc}")

    async def _watch_loop(self, channel_id: str) -> None:
        logger.info(f"[驱力追踪] {channel_id[:8]} 开始情感监控")
        while self._alive:
            try:
                await asyncio.sleep(self.POLL_INTERVAL_SEC)
                self._evolve_snapshot(channel_id)
                await self._absorb_external(channel_id)
                snap = self._snapshots.get(channel_id)
                if snap is None:
                    break
                self._recalc_assertiveness(snap, channel_id)
                verdict, motive = await self._llm_check(channel_id)
                if verdict:
                    logger.info(
                        f"[驱力追踪] {channel_id[:8]} 决定主动发言: {motive}"
                    )
                    await self._dispatch_proactive(channel_id, motive)
                    break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[驱力追踪] {channel_id[:8]} 循环错误: {exc}")
                break

    # ==================================================================
    #  脉搏演进
    # ==================================================================

    def _evolve_snapshot(self, channel_id: str) -> None:
        snap = self._snapshots.get(channel_id)
        if snap is None:
            return
        silence_min = (time.time() - snap.last_touch_ts) / 60.0
        tedium_add = min(silence_min * self.TEDIUM_RATE, 1.0)
        snap.tedium = min(1.0, snap.tedium + tedium_add)
        iso_add = min(silence_min * self.ISOLATION_RATE, 1.0)
        snap.isolation_buildup = min(1.0, snap.isolation_buildup + iso_add)

    async def _absorb_external(self, channel_id: str) -> None:
        """整合外部情绪追踪器 + 能量管理器数据。"""
        snap = self._snapshots.get(channel_id)
        if snap is None:
            return
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            tracker = get_emotion_tracker(channel_id)
            if tracker:
                avg_mood = getattr(tracker, "average_mood_score", 0.5)
                snap.temper_index = avg_mood
                try:
                    from src.chat.heart_flow.energy_manager import EnergyChainDimension

                    d6 = EnergyChainDimension.get_instance()
                    ch = d6._ensure_channel(channel_id)
                    if ch:
                        snap.vitality = max(0.0, min(1.0, float(ch.combined_ratio())))
                except Exception as _e:
                    logger.debug(f"活力状态获取异常: {_e}")
                mood_factor = (avg_mood + 1.0) / 2.0
                energy_factor = snap.vitality
                snap.companionship_pull = (
                    mood_factor * 0.6 + energy_factor * 0.4
                )
        except Exception as exc:
            logger.debug(f"[驱力追踪] 外部数据整合异常: {exc}")

    def _recalc_assertiveness(
        self, snap: UrgeSnapshot, channel_id: str
    ) -> None:
        """计算综合主动发言意愿。"""
        silence_min = (time.time() - snap.last_touch_ts) / 60.0
        tedium_w = snap.tedium * 0.4
        iso_w = snap.isolation_buildup * 0.3
        social_w = snap.companionship_pull * 0.2
        mood_w = snap.temper_index * 0.1
        time_bonus = min(silence_min / 60.0, 1.0) * 0.1
        raw = tedium_w + iso_w + social_w + mood_w + time_bonus
        if snap.vitality < 0.3:
            raw *= 0.5
        snap.assertiveness = min(1.0, raw)
        logger.debug(
            f"[驱力追踪] {channel_id[:8]} 脉搏: "
            f"厌倦={snap.tedium:.2f} 孤独={snap.isolation_buildup:.2f} "
            f"意愿={snap.assertiveness:.2f} 沉默={silence_min:.1f}min"
        )

    # ==================================================================
    #  LLM 确认
    # ==================================================================

    async def _llm_check(self, channel_id: str) -> Tuple[bool, str]:
        snap = self._snapshots.get(channel_id)
        if snap is None:
            return False, "无快照"
        if snap.assertiveness < self.ASSERTIVENESS_BAR:
            return False, f"意愿不足({snap.assertiveness:.2f})"
        if snap.vitality < 0.2:
            return False, f"精力过低({snap.vitality:.2f})"
        ok = await self._llm_quick_ask(channel_id, snap)
        if ok:
            reason = (
                f"情感驱动(厌倦={snap.tedium:.2f}, "
                f"孤独={snap.isolation_buildup:.2f}, "
                f"社交欲={snap.companionship_pull:.2f})"
            )
            return True, reason
        return False, "LLM 决定暂不说话"

    async def _llm_quick_ask(
        self, channel_id: str, snap: UrgeSnapshot
    ) -> bool:
        """快速 LLM 判断：现在想说话吗？"""
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            cfg = model_config.model_task_config.lightweight
            if cfg is None:
                return False
            silence_min = (time.time() - snap.last_touch_ts) / 60.0
            prompt = _build_quick_feeling_prompt(snap, silence_min)
            req = LLMRequest(cfg, "intrinsic_drive")
            try:
                resp, _ = await asyncio.wait_for(
                    req.generate_response_async(prompt, max_tokens=10),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                logger.debug("[驱力追踪] LLM快速问询超时(15s)")
                return False
            if isinstance(resp, str) and "想" in resp.strip():
                return True
            return False
        except Exception as exc:
            logger.error(f"[驱力追踪] LLM 快速问询异常: {exc}")
            return False

    # ==================================================================
    #  触发主动发言
    # ==================================================================

    async def _dispatch_proactive(self, channel_id: str, motive: str) -> None:
        try:
            from src.chat.heart_flow.heartflow import heartflow

            await heartflow.get_or_create_heartflow_chat(channel_id)
            heartflow.touch(channel_id)
            logger.info(
                f"[驱力追踪] {channel_id[:8]} 已唤醒主链主动评估: {motive}"
            )
        except Exception as exc:
            logger.error(f"[驱力追踪] 触发主动发言失败: {exc}")

    # ==================================================================
    #  辅助
    # ==================================================================

    async def _halt_watch(self, channel_id: str) -> None:
        task = self._watch_tasks.pop(channel_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _touch(self, channel_id: str) -> UrgeSnapshot:
        if channel_id not in self._snapshots:
            self._snapshots[channel_id] = UrgeSnapshot()
        return self._snapshots[channel_id]

    # ==================================================================
    #  查询 / 关闭
    # ==================================================================

    def peek_status(self, channel_id: str) -> Dict[str, Any]:
        snap = self._touch(channel_id)
        silence_min = (
            (time.time() - snap.last_touch_ts) / 60.0
            if snap.last_touch_ts
            else 0.0
        )
        return {
            "tedium": snap.tedium,
            "isolation_buildup": snap.isolation_buildup,
            "companionship_pull": snap.companionship_pull,
            "temper_index": snap.temper_index,
            "vitality": snap.vitality,
            "assertiveness": snap.assertiveness,
            "silence_minutes": silence_min,
            "monitoring": channel_id in self._watch_tasks,
        }

    async def teardown(self) -> None:
        self._alive = False
        for t in list(self._watch_tasks.values()):
            if not t.done():
                t.cancel()
        self._watch_tasks.clear()
        self._channel_locks.clear()
        logger.info("[驱力追踪] 内在驱力追踪器已关闭")


# ---------------------------------------------------------------------------
#  模块级入口
# ---------------------------------------------------------------------------


def get_inner_urge_tracker() -> InnerUrgeTracker:
    """获取内在驱力追踪器单例。"""
    return InnerUrgeTracker.sole()

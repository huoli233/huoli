import time
from typing import Dict, Optional
from src.common.logger import get_logger
from src.core.group_scene_state import (
    get_group_scene,
    GroupSceneState,
    AtmosphereType,
    SceneSnapshot,
)
from src.chat.heart_flow.dimension_protocol import (
    DimensionBase,
    DimensionScope,
    EventContext,
    TickResult,
)
from src.chat.heart_flow.vote_types import (
    GroupAtmosphereVote,
    AtmosphereTier,
    AIPositioning,
)

logger = get_logger("group_atmosphere")


# ============================================================
# 18种原始氛围 → 5档映射
# ============================================================
_ATMOSPHERE_TIER_MAP: Dict[AtmosphereType, AtmosphereTier] = {
    AtmosphereType.QUIET: AtmosphereTier.QUIET,
    AtmosphereType.CASUAL_CHAT: AtmosphereTier.NORMAL,
    AtmosphereType.HEATED_DISCUSSION: AtmosphereTier.HEATED,
    AtmosphereType.ARGUMENT: AtmosphereTier.HEATED,
    AtmosphereType.CELEBRATION: AtmosphereTier.LIVELY,
    AtmosphereType.MOURNING: AtmosphereTier.QUIET,
    AtmosphereType.BORED_DRIFT: AtmosphereTier.QUIET,
    AtmosphereType.NEWCOMER_WELCOME: AtmosphereTier.LIVELY,
    AtmosphereType.SPECTATING: AtmosphereTier.NORMAL,
    AtmosphereType.MEMEING: AtmosphereTier.LIVELY,
    AtmosphereType.DEEP_DISCUSSION: AtmosphereTier.HEATED,
    AtmosphereType.STORYTELLING: AtmosphereTier.NORMAL,
    AtmosphereType.PROBLEM_SOLVING: AtmosphereTier.LIVELY,
    AtmosphereType.SOCIAL_GAMING: AtmosphereTier.LIVELY,
    AtmosphereType.EMOTIONAL_SUPPORT: AtmosphereTier.NORMAL,
    AtmosphereType.NIGHT_OWLS: AtmosphereTier.QUIET,
    AtmosphereType.MORNING_COFFEE: AtmosphereTier.NORMAL,
    AtmosphereType.SPAM_FLOOD: AtmosphereTier.DEAD,
}


# ============================================================
# 五档对应的基础概率调整和回复风格约束
# ============================================================
_TIER_CONFIG = {
    AtmosphereTier.HEATED: {
        "boost": 0.25,
        "style": "简短有力、跟上节奏",
        "token_cap": 150,
    },
    AtmosphereTier.LIVELY: {
        "boost": 0.15,
        "style": "言简意赅、活泼参与",
        "token_cap": 250,
    },
    AtmosphereTier.NORMAL: {
        "boost": 0.0,
        "style": "",
        "token_cap": 0,
    },
    AtmosphereTier.QUIET: {
        "boost": -0.1,
        "style": "轻声、不打扰",
        "token_cap": 400,
    },
    AtmosphereTier.DEAD: {
        "boost": -0.3,
        "style": "谨慎发言、避免尴尬",
        "token_cap": 200,
    },
}


# ============================================================
# AI定位判定：被提及/被讨论/围观/局外
# ============================================================
def _determine_ai_position(
    snap: SceneSnapshot,
    is_at_bot: bool,
    bot_mentioned_in_topic: bool,
) -> AIPositioning:
    """判定AI在当前群聊场景中的位置"""
    if is_at_bot:
        return AIPositioning.CORE
    if bot_mentioned_in_topic:
        return AIPositioning.PARTICIPANT
    if snap.suitable_to_join and snap.unique_speakers_5min >= 3:
        return AIPositioning.PARTICIPANT
    if snap.atmosphere in (
        AtmosphereType.QUIET,
        AtmosphereType.MOURNING,
        AtmosphereType.BORED_DRIFT,
    ):
        return AIPositioning.OUTSIDER
    return AIPositioning.OBSERVER


# ============================================================
# 频道级别的氛围缓存（减少重复计算）
# ============================================================
class _ChannelAtmosphereCache:
    """缓存最近一次快照结果（30秒有效期）"""

    def __init__(self):
        self._snap: Optional[SceneSnapshot] = None
        self._ts: float = 0.0

    def get(self, scene: GroupSceneState, ttl: float = 30.0) -> SceneSnapshot:
        now = time.time()
        if self._snap and (now - self._ts) < ttl:
            return self._snap
        self._snap = scene.snapshot()
        self._ts = now
        return self._snap

    def invalidate(self):
        self._snap = None
        self._ts = 0.0


# ============================================================
# D5 维度实现：群体氛围
# ============================================================
class GroupAtmosphereDimension(DimensionBase):
    """
    D5 群体氛围维度。
    包装现有 group_scene_state.py 的 GroupSceneState 单例，
    将18种精细氛围映射为5档 + AI定位 + 参与度加成。
    per-channel 粒度。
    """

    _singleton: Optional["GroupAtmosphereDimension"] = None

    @classmethod
    def get_instance(cls) -> "GroupAtmosphereDimension":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        # channel_id → 氛围缓存
        self._caches: Dict[str, _ChannelAtmosphereCache] = {}
        # 每个频道最近一次 AI 被提及的时间
        self._bot_mention_ts: Dict[str, float] = {}
        # 频道粒度活跃汇总
        self._last_tick_ts: float = time.time()

    @property
    def dimension_name(self) -> str:
        return "group_atmosphere"

    @property
    def scope(self) -> DimensionScope:
        return DimensionScope.PER_CHANNEL

    @property
    def needs_persistence(self) -> bool:
        return False

    @property
    def tick_interval_sec(self) -> float:
        return 15.0

    def _get_cache(self, channel_id: str) -> _ChannelAtmosphereCache:
        if channel_id not in self._caches:
            self._caches[channel_id] = _ChannelAtmosphereCache()
        return self._caches[channel_id]

    def tick(self, elapsed_sec: float) -> TickResult:
        """定期清理过期缓存"""
        now = time.time()
        stale_channels = []
        for ch_id, cache in self._caches.items():
            if cache._ts > 0 and (now - cache._ts) > 600.0:
                stale_channels.append(ch_id)
        for ch_id in stale_channels:
            del self._caches[ch_id]
        return TickResult(
            dimension_name=self.dimension_name,
            updated=len(stale_channels) > 0,
            summary=f"清理过期缓存={len(stale_channels)}" if stale_channels else "",
        )

    def on_event(self, ctx: EventContext):
        """
        消息到达时:
          1. 将消息录入底层 GroupSceneState
          2. 如果是@bot，记录时间戳
          3. 刷新缓存
        """
        if not ctx.channel_id:
            return
        if ctx.event_type == "message_received":
            scene = get_group_scene(ctx.channel_id)
            scene.record_message(
                user_id=ctx.user_id or "unknown",
                text=ctx.message_text,
                topics=ctx.raw_extras.get("topics"),
                thread_hint=ctx.raw_extras.get("thread_hint", ""),
                mentioned_users=ctx.raw_extras.get("mentioned_users"),
                reply_anchor=ctx.raw_extras.get("reply_anchor", ""),
            )
            # @bot 记录
            if ctx.is_at_bot:
                self._bot_mention_ts[ctx.channel_id] = time.time()
            # 刷新缓存
            cache = self._get_cache(ctx.channel_id)
            cache.invalidate()

    def vote(self, ctx: EventContext) -> GroupAtmosphereVote:
        """
        拉取频道场景快照，映射为五档氛围 + AI定位，
        计算参与度加成和风格约束。
        """
        if not ctx.channel_id:
            return GroupAtmosphereVote()
        scene = get_group_scene(ctx.channel_id)
        cache = self._get_cache(ctx.channel_id)
        snap = cache.get(scene, ttl=15.0)
        # 氛围五档映射
        tier = _ATMOSPHERE_TIER_MAP.get(snap.atmosphere, AtmosphereTier.NORMAL)
        # AI定位
        bot_mention_recent = False
        mention_ts = self._bot_mention_ts.get(ctx.channel_id, 0.0)
        if mention_ts > 0 and (time.time() - mention_ts) < 120.0:
            bot_mention_recent = True
        position = _determine_ai_position(
            snap,
            is_at_bot=ctx.is_at_bot,
            bot_mentioned_in_topic=bot_mention_recent,
        )
        # 参与度加成
        tier_cfg = _TIER_CONFIG.get(tier, _TIER_CONFIG[AtmosphereTier.NORMAL])
        base_boost = tier_cfg["boost"]
        # AI是核心时额外加成
        if position == AIPositioning.CORE:
            base_boost += 0.3
        elif position == AIPositioning.PARTICIPANT:
            base_boost += 0.1
        elif position == AIPositioning.OUTSIDER:
            base_boost -= 0.1
        # 概率因子 = 1.0 + boost（限制在0.2~2.0）
        prob = max(0.2, min(2.0, 1.0 + base_boost))
        # 话题关键词
        topic_keywords = scene.active_topics(limit=5)
        # 话题相关度（简单启发式：如果AI被提及或@，相关度高）
        relevance = 0.5
        if position == AIPositioning.CORE:
            relevance = 0.9
        elif position == AIPositioning.PARTICIPANT:
            relevance = 0.7
        elif position == AIPositioning.OUTSIDER:
            relevance = 0.2
        # 风格约束
        style = str(tier_cfg.get("style", ""))
        # 氛围特殊风格覆盖
        if snap.atmosphere == AtmosphereType.ARGUMENT:
            style = "冷静客观、避免火上浇油"
        elif snap.atmosphere == AtmosphereType.MOURNING:
            style = "温和安静、少说多听"
        elif snap.atmosphere == AtmosphereType.MEMEING:
            style = "跟梗适度、不要刻意"
        elif snap.atmosphere == AtmosphereType.EMOTIONAL_SUPPORT:
            style = "共情理解、温暖简洁"
        # 态度标签
        attitude = ""
        if snap.atmosphere == AtmosphereType.SPAM_FLOOD:
            attitude = "spam_detected"
        elif snap.atmosphere == AtmosphereType.ARGUMENT:
            attitude = "conflict_zone"
        # token上限
        token_cap = tier_cfg.get("token_cap", 0)
        return GroupAtmosphereVote(
            probability_factor=round(prob, 3),
            attitude_tag=attitude,
            style_hint=style,
            atmosphere_tier=tier,
            ai_position=position,
            engagement_boost=round(base_boost, 3),
            active_user_count=snap.unique_speakers_5min,
            topic_keywords=topic_keywords,
            style_constraint=style,
            topic_relevance=round(relevance, 2),
            min_tokens=0,
            max_tokens=token_cap if token_cap > 0 else 0,
            debug_reason=(
                f"atm={snap.atmosphere.value} tier={tier.value} "
                f"pos={position.value} boost={base_boost:.2f} "
                f"speakers={snap.unique_speakers_5min}"
            ),
        )

    def serialize(self) -> dict:
        return {}

    def deserialize(self, data: dict):
        pass

    def calibrate(self, offline_seconds: float):
        """离线后清空所有缓存，场景从零积累"""
        self._caches.clear()
        self._bot_mention_ts.clear()

    def get_state_summary(self) -> dict:
        channels = {}
        for ch_id, cache in list(self._caches.items())[:10]:
            if cache._snap:
                channels[ch_id[:12]] = {
                    "atmosphere": cache._snap.atmosphere.value,
                    "msg_per_min": round(cache._snap.messages_per_minute, 2),
                    "speakers": cache._snap.unique_speakers_5min,
                }
        return {
            "dimension": self.dimension_name,
            "scope": self.scope.value,
            "tracked_channels": len(self._caches),
            "channels": channels,
        }

    def reset(self, user_id: str = "", channel_id: str = ""):
        if channel_id:
            self._caches.pop(channel_id, None)
            self._bot_mention_ts.pop(channel_id, None)

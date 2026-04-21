import time
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from src.common.logger import get_logger

logger = get_logger("participant_roster")

_roster_instances: Dict[str, "ActiveParticipantRoster"] = {}
_refresh_scheduler_instance: "TierRefreshScheduler" = None


class ParticipantTier(Enum):
    """用户活跃分层"""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    ABSENT = "absent"

    def label(self) -> str:
        labels = {
            ParticipantTier.HOT: "热活跃",
            ParticipantTier.WARM: "温活跃",
            ParticipantTier.COLD: "冷用户",
            ParticipantTier.ABSENT: "离线",
        }
        return labels.get(self, "未知")


@dataclass
class ParticipantEntry:
    """单个用户的活跃记录，第三阶段升级：增加更多活跃度指标"""

    user_id: str = ""
    user_name: str = ""
    tier: ParticipantTier = ParticipantTier.COLD
    last_active_at: float = field(default_factory=time.time)
    message_count_5min: int = 0
    message_count_total: int = 0
    first_seen_at: float = field(default_factory=time.time)
    last_message_text_hint: str = ""
    is_bot_interacted: bool = False
    consecutive_messages: int = 0
    engagement_score: float = 0.0
    social_influence: float = 0.0
    topic_contribution: float = 0.0
    emotional_engagement: float = 0.0
    time_spent_active: float = 0.0
    interaction_quality: float = 0.5
    response_patterns: Dict[str, int] = field(default_factory=dict)
    # USR-09: 最近活跃快照
    recent_messages: List[str] = field(default_factory=list)
    current_topic: str = ""
    mood_snapshot: str = "平静"
    _max_recent_messages: int = 8
    # 表情包偏好追踪
    sticker_send_count: int = 0
    sticker_receive_count: int = 0
    preferred_sticker_categories: Dict[str, int] = field(default_factory=dict)
    emoji_frequency: float = 0.0
    last_sticker_at: float = 0.0

    def idle_seconds(self) -> float:
        return time.time() - self.last_active_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "tier": self.tier.value,
            "tier_label": self.tier.label(),
            "idle_seconds": round(self.idle_seconds(), 1),
            "message_count_5min": self.message_count_5min,
            "message_count_total": self.message_count_total,
            "is_bot_interacted": self.is_bot_interacted,
            "consecutive_messages": self.consecutive_messages,
            "engagement_score": round(self.engagement_score, 3),
            "social_influence": round(self.social_influence, 3),
            "topic_contribution": round(self.topic_contribution, 3),
            "emotional_engagement": round(self.emotional_engagement, 3),
            "time_spent_active": round(self.time_spent_active, 1),
            "interaction_quality": round(self.interaction_quality, 3),
            "recent_messages": self.recent_messages[-5:],
            "current_topic": self.current_topic,
            "mood_snapshot": self.mood_snapshot,
            "sticker_send_count": self.sticker_send_count,
            "sticker_receive_count": self.sticker_receive_count,
            "preferred_sticker_categories": dict(
                self.preferred_sticker_categories
            ),
            "emoji_frequency": round(self.emoji_frequency, 3),
        }


class ActiveParticipantRoster:
    """活跃用户分层缓存，第三阶段升级：智能活跃度计算和社交影响分析
    维护四层缓存：
    - 热层（HOT）：高活跃度用户，综合考虑发言频率、质量、影响力
    - 温层（WARM）：中等活跃度，包含近期互动和潜在贡献者
    - 冷层（COLD）：低活跃度但有历史记录
    - 离线层（ABSENT）：长期不活跃
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._entries: Dict[str, ParticipantEntry] = {}
        self._hot_cap = 20
        self._warm_cap = 50
        self._hot_window = 300.0
        self._warm_window = 900.0
        self._cold_purge_sec = 3600.0
        self._engagement_decay_rate = 0.95
        self._influence_boost_factor = 1.2
        self._quality_weight = 0.4
        self._frequency_weight = 0.3
        self._influence_weight = 0.3

    def touch(
        self,
        user_id: str,
        *,
        user_name: str = "",
        text_hint: str = "",
        bot_interacted: bool = False,
        message_quality: float = 0.5,
        topic_relevance: float = 0.5,
        emotional_content: float = 0.0,
        response_type: str = "general",
        current_topic: str = "",
        mood_label: str = "",
    ) -> ParticipantEntry:
        """记录用户活动，第三阶段升级：加入质量和影响力评估"""
        now = time.time()
        if user_id not in self._entries:
            self._entries[user_id] = ParticipantEntry(
                user_id=user_id,
                user_name=user_name or user_id,
                first_seen_at=now,
            )
        entry = self._entries[user_id]
        previous_active_at = entry.last_active_at
        if user_name:
            entry.user_name = user_name
        entry.last_active_at = now
        entry.message_count_total += 1
        entry.consecutive_messages += 1
        if text_hint:
            entry.last_message_text_hint = text_hint[:50]
            # USR-09/10: 追踪最近消息内容
            entry.recent_messages.append(text_hint[:80])
            if len(entry.recent_messages) > entry._max_recent_messages:
                entry.recent_messages = entry.recent_messages[
                    -entry._max_recent_messages:
                ]
        if current_topic:
            entry.current_topic = current_topic[:30]
        if mood_label:
            entry.mood_snapshot = mood_label[:20]
        if bot_interacted:
            entry.is_bot_interacted = True
        # 第三阶段新增：更新活跃度指标
        self._update_engagement_metrics(
            entry,
            message_quality,
            topic_relevance,
            emotional_content,
            response_type,
            now,
            previous_active_at,
        )
        # 更新分层
        self._update_tier(entry, now)
        return entry

    def _update_engagement_metrics(
        self,
        entry: ParticipantEntry,
        quality: float,
        relevance: float,
        emotion: float,
        response_type: str,
        now: float,
        previous_active_at: float,
    ) -> None:
        """更新用户活跃度指标，第三阶段新增"""
        # 消息质量综合评分
        entry.interaction_quality = (
            entry.interaction_quality * 0.7 + quality * 0.3
        )
        # 话题贡献度
        entry.topic_contribution = min(
            1.0, entry.topic_contribution + relevance * 0.1
        )
        # 情感参与度
        entry.emotional_engagement = min(
            1.0, entry.emotional_engagement + abs(emotion) * 0.15
        )
        # 活跃时间累积
        time_gap = now - previous_active_at if previous_active_at > 0 else 0
        if time_gap < 300:  # 5分钟内连续活跃
            entry.time_spent_active += time_gap
        # 响应模式统计
        if response_type not in entry.response_patterns:
            entry.response_patterns[response_type] = 0
        entry.response_patterns[response_type] += 1
        # 综合活跃度评分
        quality_score = entry.interaction_quality * self._quality_weight
        frequency_score = (
            min(1.0, entry.consecutive_messages / 10.0)
            * self._frequency_weight
        )
        influence_score = (
            (entry.topic_contribution + entry.emotional_engagement)
            / 2.0
            * self._influence_weight
        )
        entry.engagement_score = (
            quality_score + frequency_score + influence_score
        ) * self._influence_boost_factor
        # 社交影响力（基于回复数量和质量）
        entry.social_influence = entry.engagement_score * (
            1.0 + entry.is_bot_interacted * 0.5
        )

    def _update_tier(self, entry: ParticipantEntry, now: float) -> None:
        """智能分层更新，第三阶段升级"""
        idle_time = now - entry.last_active_at
        # 热层条件：高活跃度 + 近期活跃 + 高质量互动
        if (
            entry.engagement_score >= 0.7
            and idle_time < self._hot_window
            and entry.interaction_quality >= 0.6
        ):
            entry.tier = ParticipantTier.HOT
        # 温层条件：中等活跃度或近期互动或bot互动历史
        elif (
            entry.engagement_score >= 0.3
            or idle_time < self._warm_window
            or entry.is_bot_interacted
        ):
            entry.tier = ParticipantTier.WARM
        # 冷层条件：有记录但不活跃
        elif idle_time < self._cold_purge_sec:
            entry.tier = ParticipantTier.COLD
        else:
            entry.tier = ParticipantTier.ABSENT

    def get_top_contributors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取顶级贡献者，第三阶段新增"""
        active_entries = [
            e
            for e in self._entries.values()
            if e.tier in (ParticipantTier.HOT, ParticipantTier.WARM)
        ]
        sorted_entries = sorted(
            active_entries, key=lambda e: e.engagement_score, reverse=True
        )
        return [self._entry_to_dict(e) for e in sorted_entries[:limit]]

    def get_social_influencers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取社交影响力用户，第三阶段新增"""
        active_entries = [
            e
            for e in self._entries.values()
            if e.tier != ParticipantTier.ABSENT
        ]
        sorted_entries = sorted(
            active_entries, key=lambda e: e.social_influence, reverse=True
        )
        return [self._entry_to_dict(e) for e in sorted_entries[:limit]]

    def _entry_to_dict(self, entry: ParticipantEntry) -> Dict[str, Any]:
        """将条目转换为字典，包含新字段"""
        base = entry.to_dict()
        base.update(
            {
                "engagement_score": round(entry.engagement_score, 3),
                "social_influence": round(entry.social_influence, 3),
                "topic_contribution": round(entry.topic_contribution, 3),
                "emotional_engagement": round(entry.emotional_engagement, 3),
                "time_spent_active": round(entry.time_spent_active, 1),
                "interaction_quality": round(entry.interaction_quality, 3),
                "response_patterns": entry.response_patterns,
            }
        )
        return base

    def mark_bot_reply(self, user_id: str) -> None:
        """标记bot回复了该用户"""
        if user_id in self._entries:
            self._entries[user_id].is_bot_interacted = True

    def reset_consecutive(self, user_id: str) -> None:
        """重置连续消息计数（当其他用户说话时）"""
        for uid, entry in self._entries.items():
            if uid != user_id:
                entry.consecutive_messages = 0

    def refresh_tiers(self) -> None:
        """刷新全体用户分层"""
        now = time.time()
        hot_candidates = []
        warm_candidates = []
        for entry in self._entries.values():
            idle = now - entry.last_active_at
            # 计算5分钟消息数
            entry.message_count_5min = self._estimate_5min_count(entry, now)
            if idle < self._hot_window and entry.message_count_5min >= 2:
                hot_candidates.append(entry)
            elif idle < self._warm_window or entry.is_bot_interacted:
                warm_candidates.append(entry)
            elif idle > self._cold_purge_sec:
                entry.tier = ParticipantTier.ABSENT
            else:
                entry.tier = ParticipantTier.COLD
        # 热层排序：按消息数降序
        hot_candidates.sort(key=lambda e: e.message_count_5min, reverse=True)
        for i, entry in enumerate(hot_candidates):
            if i < self._hot_cap:
                entry.tier = ParticipantTier.HOT
            else:
                warm_candidates.append(entry)
        # 温层排序：按最后活跃时间降序
        warm_candidates.sort(key=lambda e: e.last_active_at, reverse=True)
        for i, entry in enumerate(warm_candidates):
            if i < self._warm_cap:
                entry.tier = ParticipantTier.WARM
            else:
                entry.tier = ParticipantTier.COLD

    def get_hot_users(self) -> List[ParticipantEntry]:
        self.refresh_tiers()
        return [
            e for e in self._entries.values() if e.tier == ParticipantTier.HOT
        ]

    def get_warm_users(self) -> List[ParticipantEntry]:
        self.refresh_tiers()
        return [
            e for e in self._entries.values() if e.tier == ParticipantTier.WARM
        ]

    def get_cold_users(self) -> List[ParticipantEntry]:
        return [
            e for e in self._entries.values() if e.tier == ParticipantTier.COLD
        ]

    def get_user_recent_snapshot(
        self, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """USR-09/10: 获取指定用户的最近活跃状态快照"""
        entry = self._entries.get(user_id)
        if entry is None:
            return None
        return {
            "user_id": entry.user_id,
            "user_name": entry.user_name,
            "tier": entry.tier.value,
            "idle_seconds": round(entry.idle_seconds(), 1),
            "recent_messages": entry.recent_messages[-5:],
            "current_topic": entry.current_topic,
            "mood_snapshot": entry.mood_snapshot,
            "engagement_score": round(entry.engagement_score, 3),
            "consecutive_messages": entry.consecutive_messages,
            "message_count_total": entry.message_count_total,
            "is_bot_interacted": entry.is_bot_interacted,
        }

    def get_active_snapshots(self, limit: int = 10) -> List[Dict[str, Any]]:
        """USR-10: 获取所有活跃用户的最近状态概览"""
        self.refresh_tiers()
        active = [
            e
            for e in self._entries.values()
            if e.tier in (ParticipantTier.HOT, ParticipantTier.WARM)
        ]
        active.sort(key=lambda e: e.last_active_at, reverse=True)
        return [
            {
                "user_id": e.user_id,
                "user_name": e.user_name,
                "tier": e.tier.value,
                "last_msg": e.recent_messages[-1] if e.recent_messages else "",
                "current_topic": e.current_topic,
                "mood": e.mood_snapshot,
                "idle_sec": round(e.idle_seconds(), 0),
            }
            for e in active[:limit]
        ]

    def get_user(self, user_id: str) -> Optional[ParticipantEntry]:
        return self._entries.get(user_id)

    def active_count(self) -> int:
        """热层+温层用户数"""
        self.refresh_tiers()
        return sum(
            1
            for e in self._entries.values()
            if e.tier in (ParticipantTier.HOT, ParticipantTier.WARM)
        )

    def summary(self) -> Dict[str, Any]:
        self.refresh_tiers()
        hot = [
            e for e in self._entries.values() if e.tier == ParticipantTier.HOT
        ]
        warm = [
            e for e in self._entries.values() if e.tier == ParticipantTier.WARM
        ]
        cold = [
            e for e in self._entries.values() if e.tier == ParticipantTier.COLD
        ]
        return {
            "hot_count": len(hot),
            "warm_count": len(warm),
            "cold_count": len(cold),
            "total": len(self._entries),
            "hot_users": [e.user_name for e in hot[:10]],
        }

    def purge_absent(self) -> int:
        """清除已离线用户"""
        absent_ids = [
            uid
            for uid, e in self._entries.items()
            if e.tier == ParticipantTier.ABSENT
        ]
        for uid in absent_ids:
            del self._entries[uid]
        return len(absent_ids)

    def note_sticker_usage(
        self,
        user_id: str,
        *,
        direction: str = "receive",
        category: str = "",
    ) -> None:
        """记录用户表情包使用，direction: "receive"(用户发的) / "send"(bot回复的)"""
        entry = self._entries.get(user_id)
        if entry is None:
            return
        now = time.time()
        if direction == "receive":
            entry.sticker_receive_count += 1
        else:
            entry.sticker_send_count += 1
        entry.last_sticker_at = now
        if category:
            entry.preferred_sticker_categories[category] = (
                entry.preferred_sticker_categories.get(category, 0) + 1
            )
        # 滚动计算 emoji 使用频率
        total_msg = max(1, entry.message_count_total)
        entry.emoji_frequency = entry.sticker_receive_count / total_msg

    def user_sticker_preference(self, user_id: str) -> Dict[str, Any]:
        """查询用户表情包偏好概况"""
        entry = self._entries.get(user_id)
        if entry is None:
            return {"user_id": user_id, "has_data": False}
        sorted_cats = sorted(
            entry.preferred_sticker_categories.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return {
            "user_id": user_id,
            "has_data": True,
            "send_count": entry.sticker_send_count,
            "receive_count": entry.sticker_receive_count,
            "emoji_frequency": round(entry.emoji_frequency, 3),
            "top_categories": [
                {"category": c, "count": n} for c, n in sorted_cats[:5]
            ],
            "favorite_category": sorted_cats[0][0] if sorted_cats else "",
        }

    # ────────────────── 内部方法 ──────────────────

    def _estimate_5min_count(self, entry: ParticipantEntry, now: float) -> int:
        """估算5分钟内的消息数。简化实现：根据总消息数和活跃时间估算"""
        if entry.last_active_at < now - self._hot_window:
            return 0
        idle = now - entry.last_active_at
        if idle < 10:
            return max(1, entry.consecutive_messages)
        active_span = now - entry.first_seen_at
        if active_span < 1:
            return 1
        rate_per_sec = entry.message_count_total / active_span
        return max(1, int(rate_per_sec * min(300.0, active_span)))


# ═══════════════════════════════════════════════════════
#  后台刷新调度器
# ═══════════════════════════════════════════════════════

# 调度常量
_FAST_REFRESH_INTERVAL_SEC = 60.0
_SLOW_REFRESH_INTERVAL_SEC = 300.0
_PURGE_INTERVAL_SEC = 600.0
_ENGAGEMENT_DECAY_INTERVAL_SEC = 120.0
_ACTIVITY_THRESHOLD_FOR_FAST = 3


class TierRefreshScheduler:
    """分层刷新调度器
    根据频道活跃度自动切换快慢刷新节奏：
    - 活跃频道（近期有 ≥ 3 条消息）：每 60 秒刷新分层 + 衰减参与度
    - 安静频道：每 300 秒刷新分层 + 衰减参与度
    - 每 600 秒执行一次离线用户清理
    """

    def __init__(self):
        self._channel_last_refresh: Dict[str, float] = {}
        self._channel_last_purge: Dict[str, float] = {}
        self._channel_last_decay: Dict[str, float] = {}
        self._running = False
        self._loop_task: asyncio.Task = None
        self._tick_interval = 15.0

    async def start(self) -> None:
        """启动后台调度循环"""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.ensure_future(self._schedule_loop())
        logger.debug("[分层调度] 后台刷新调度器已启动")

    async def stop(self) -> None:
        """停止调度循环"""
        self._running = False
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                logger.debug("[分层调度] 调度循环已取消")
        self._loop_task = None

    async def _schedule_loop(self) -> None:
        """主循环：每 tick 检查所有已注册频道是否需要刷新"""
        while self._running:
            try:
                await asyncio.sleep(self._tick_interval)
                await self._tick_all_channels()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"[分层调度] tick异常: {exc}")

    async def _tick_all_channels(self) -> None:
        """遍历所有已注册频道，按需执行刷新/衰减/清理"""
        now = time.time()
        for channel_id, roster in list(_roster_instances.items()):
            self._tick_single_channel(channel_id, roster, now)

    def _tick_single_channel(
        self,
        channel_id: str,
        roster: "ActiveParticipantRoster",
        now: float,
    ) -> None:
        """单频道的周期维护"""
        # 判断频道是否活跃
        recent_activity = self._estimate_channel_activity(roster)
        refresh_interval = (
            _FAST_REFRESH_INTERVAL_SEC
            if recent_activity >= _ACTIVITY_THRESHOLD_FOR_FAST
            else _SLOW_REFRESH_INTERVAL_SEC
        )
        # 分层刷新
        last_refresh = self._channel_last_refresh.get(channel_id, 0.0)
        if now - last_refresh >= refresh_interval:
            roster.refresh_tiers()
            self._channel_last_refresh[channel_id] = now
        # 参与度衰减
        last_decay = self._channel_last_decay.get(channel_id, 0.0)
        if now - last_decay >= _ENGAGEMENT_DECAY_INTERVAL_SEC:
            self._decay_engagement_scores(roster, now)
            self._channel_last_decay[channel_id] = now
        # 离线用户清理
        last_purge = self._channel_last_purge.get(channel_id, 0.0)
        if now - last_purge >= _PURGE_INTERVAL_SEC:
            purged = roster.purge_absent()
            if purged > 0:
                logger.debug(
                    f"[分层调度] {channel_id} 清理离线用户 {purged} 人"
                )
            self._channel_last_purge[channel_id] = now

    @staticmethod
    def _estimate_channel_activity(roster: "ActiveParticipantRoster") -> int:
        """估算频道近期活跃消息数（不触发 refresh_tiers 以避免递归）"""
        now = time.time()
        cutoff = now - 300.0
        active_entries = [
            e for e in roster._entries.values() if e.last_active_at > cutoff
        ]
        return sum(e.consecutive_messages for e in active_entries)

    @staticmethod
    def _decay_engagement_scores(
        roster: "ActiveParticipantRoster", now: float
    ) -> None:
        """衰减所有用户的参与度指标，活跃越久衰减越少"""
        for entry in roster._entries.values():
            idle = now - entry.last_active_at
            if idle < 60:
                decay_factor = 0.98
            elif idle < 300:
                decay_factor = 0.95
            elif idle < 900:
                decay_factor = 0.90
            else:
                decay_factor = 0.80
            entry.engagement_score *= decay_factor
            entry.social_influence *= decay_factor
            entry.topic_contribution *= max(0.85, decay_factor)
            entry.emotional_engagement *= max(0.90, decay_factor)

    def force_refresh(self, channel_id: str) -> bool:
        """手动触发指定频道的立即刷新"""
        roster = _roster_instances.get(channel_id)
        if roster is None:
            return False
        now = time.time()
        roster.refresh_tiers()
        self._decay_engagement_scores(roster, now)
        self._channel_last_refresh[channel_id] = now
        self._channel_last_decay[channel_id] = now
        return True

    def schedule_status(self) -> Dict[str, Any]:
        """调度器状态概览"""
        now = time.time()
        channel_info = {}
        for ch_id in _roster_instances:
            last_r = self._channel_last_refresh.get(ch_id, 0.0)
            last_d = self._channel_last_decay.get(ch_id, 0.0)
            last_p = self._channel_last_purge.get(ch_id, 0.0)
            channel_info[ch_id] = {
                "seconds_since_refresh": (
                    round(now - last_r, 1) if last_r > 0 else -1
                ),
                "seconds_since_decay": (
                    round(now - last_d, 1) if last_d > 0 else -1
                ),
                "seconds_since_purge": (
                    round(now - last_p, 1) if last_p > 0 else -1
                ),
            }
        return {
            "running": self._running,
            "tracked_channels": len(_roster_instances),
            "channels": channel_info,
        }

    def untrack_channel(self, channel_id: str) -> None:
        """移除已关闭频道的调度记录"""
        self._channel_last_refresh.pop(channel_id, None)
        self._channel_last_purge.pop(channel_id, None)
        self._channel_last_decay.pop(channel_id, None)


def get_participant_roster(channel_id: str) -> ActiveParticipantRoster:
    if channel_id not in _roster_instances:
        _roster_instances[channel_id] = ActiveParticipantRoster(channel_id)
    return _roster_instances[channel_id]


def remove_participant_roster(channel_id: str) -> None:
    _roster_instances.pop(channel_id, None)
    scheduler = get_tier_refresh_scheduler()
    scheduler.untrack_channel(channel_id)


def get_tier_refresh_scheduler() -> TierRefreshScheduler:
    global _refresh_scheduler_instance
    if _refresh_scheduler_instance is None:
        _refresh_scheduler_instance = TierRefreshScheduler()
    return _refresh_scheduler_instance

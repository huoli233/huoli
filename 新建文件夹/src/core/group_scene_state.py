import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from src.common.logger import get_logger

logger = get_logger("group_scene")

_scene_instances: Dict[str, "GroupSceneState"] = {}


class AtmosphereType(Enum):
    """群氛围类型，第三阶段升级：增加更多精细化氛围类型"""

    QUIET = "quiet"
    CASUAL_CHAT = "casual_chat"
    HEATED_DISCUSSION = "heated_discussion"
    ARGUMENT = "argument"
    CELEBRATION = "celebration"
    MOURNING = "mourning"
    BORED_DRIFT = "bored_drift"
    NEWCOMER_WELCOME = "newcomer_welcome"
    SPECTATING = "spectating"
    MEMEING = "memeing"
    DEEP_DISCUSSION = "deep_discussion"
    STORYTELLING = "storytelling"
    PROBLEM_SOLVING = "problem_solving"
    SOCIAL_GAMING = "social_gaming"
    EMOTIONAL_SUPPORT = "emotional_support"
    NIGHT_OWLS = "night_owls"
    MORNING_COFFEE = "morning_coffee"
    SPAM_FLOOD = "spam_flood"

    def label(self) -> str:
        labels = {
            AtmosphereType.QUIET: "安静",
            AtmosphereType.CASUAL_CHAT: "随意闲聊",
            AtmosphereType.HEATED_DISCUSSION: "热烈讨论",
            AtmosphereType.ARGUMENT: "争论",
            AtmosphereType.CELEBRATION: "庆祝",
            AtmosphereType.MOURNING: "低落",
            AtmosphereType.BORED_DRIFT: "无聊漫游",
            AtmosphereType.NEWCOMER_WELCOME: "迎新",
            AtmosphereType.SPECTATING: "围观",
            AtmosphereType.MEMEING: "玩梗",
            AtmosphereType.DEEP_DISCUSSION: "深度讨论",
            AtmosphereType.STORYTELLING: "讲故事",
            AtmosphereType.PROBLEM_SOLVING: "解决问题",
            AtmosphereType.SOCIAL_GAMING: "社交游戏",
            AtmosphereType.EMOTIONAL_SUPPORT: "情感支持",
            AtmosphereType.NIGHT_OWLS: "夜猫子",
            AtmosphereType.MORNING_COFFEE: "晨间咖啡",
            AtmosphereType.SPAM_FLOOD: "刷屏",
        }
        return labels.get(self, "未知")


class TopicHeat(Enum):
    """话题热度"""

    COLD = "cold"
    WARM = "warm"
    HOT = "hot"
    EXPLOSIVE = "explosive"


@dataclass
class TopicSlot:
    """话题槽位"""

    keyword: str = ""
    contributors: Set[str] = field(default_factory=set)
    first_seen: float = field(default_factory=time.time)
    last_mentioned: float = field(default_factory=time.time)
    mention_count: int = 1
    heat: TopicHeat = TopicHeat.WARM
    bot_contributed: bool = False

    def age_seconds(self) -> float:
        return time.time() - self.last_mentioned

    def is_stale(self, ttl: float = 300.0) -> bool:
        return self.age_seconds() > ttl

    def ownership_type(self, bot_id: str = "bot") -> str:
        """话题所有权分类：bot_topic / target_user_topic / shared_group_topic / external_topic / ambiguous_topic"""
        if (
            self.bot_contributed
            and len(self.contributors) == 1
            and bot_id in self.contributors
        ):
            return "self_topic"
        if len(self.contributors) == 1:
            return "target_user_topic"
        if len(self.contributors) >= 3 or self.mention_count >= 5:
            return "shared_group_topic"
        if self.mention_count <= 1 and self.age_seconds() > 900:
            return "external_topic"
        if len(self.contributors) == 2 and self.mention_count <= 3:
            return "target_user_topic"
        return "ambiguous_topic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keyword": self.keyword,
            "contributors": list(self.contributors),
            "mention_count": self.mention_count,
            "heat": self.heat.value,
            "age_seconds": round(self.age_seconds(), 1),
            "ownership": self.ownership_type(),
        }


@dataclass
class ThreadMarker:
    """线程标记：同时进行的多条对话线"""

    thread_id: str = ""
    participants: Set[str] = field(default_factory=set)
    target_users: Set[str] = field(default_factory=set)
    topic_hint: str = ""
    started_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    message_count: int = 0

    def is_alive(self, ttl: float = 180.0) -> bool:
        return (time.time() - self.last_active) < ttl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "participants": list(self.participants),
            "target_users": list(self.target_users),
            "topic_hint": self.topic_hint,
            "message_count": self.message_count,
            "alive": self.is_alive(),
        }


@dataclass
class RecentSceneMessage:
    """最近消息线索，用于隐式线程推断。"""

    user_id: str = ""
    text: str = ""
    topics: List[str] = field(default_factory=list)
    mentioned_users: List[str] = field(default_factory=list)
    thread_id: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class SceneSnapshot:
    """场景快照，第三阶段升级：增加更多场景分析维度"""

    atmosphere: AtmosphereType = AtmosphereType.QUIET
    active_topic_count: int = 0
    thread_count: int = 0
    messages_per_minute: float = 0.0
    unique_speakers_5min: int = 0
    suitable_to_join: bool = True
    emotional_tone: str = "neutral"
    complexity_level: float = 0.0
    social_density: float = 0.0
    time_of_day_factor: float = 0.0
    participant_diversity: float = 0.0
    interaction_quality: float = 0.5
    suitable_to_stay_silent: bool = False
    dominant_speaker: str = ""
    interaction_pairs: List[str] = field(default_factory=list)
    auto_thread_inference: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atmosphere": self.atmosphere.value,
            "atmosphere_label": self.atmosphere.label(),
            "active_topic_count": self.active_topic_count,
            "thread_count": self.thread_count,
            "messages_per_minute": round(self.messages_per_minute, 2),
            "unique_speakers_5min": self.unique_speakers_5min,
            "suitable_to_join": self.suitable_to_join,
            "suitable_to_stay_silent": self.suitable_to_stay_silent,
            "dominant_speaker": self.dominant_speaker,
            "emotional_tone": self.emotional_tone,
            "complexity_level": round(self.complexity_level, 2),
            "social_density": round(self.social_density, 2),
            "time_of_day_factor": round(self.time_of_day_factor, 2),
            "participant_diversity": round(self.participant_diversity, 2),
            "interaction_quality": round(self.interaction_quality, 2),
            "interaction_pairs": self.interaction_pairs[:8],
            "auto_thread_inference": self.auto_thread_inference,
        }


class GroupSceneState:
    """群场景状态层
    聚合当前群聊的氛围、话题、线程、活跃度等信息，
    为主链和可见性层提供群环境输入。
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._topics: Dict[str, TopicSlot] = {}
        self._threads: Dict[str, ThreadMarker] = {}
        self._message_timestamps: List[float] = []
        self._speaker_timestamps: Dict[str, List[float]] = {}
        self._interaction_pairs: Dict[str, float] = {}
        self._recent_messages: List[RecentSceneMessage] = []
        self._max_timestamps = 200
        self._topic_ttl = 300.0
        self._thread_ttl = 180.0
        # 围观/玩梗追踪
        self._known_online_count: int = 0
        self._recent_meme_hits: List[float] = []
        self._meme_keywords = {
            "哈哈哈",
            "笑死",
            "绷不住",
            "乐了",
            "草",
            "6",
            "xswl",
            "hhh",
            "2333",
            "哈哈",
            "笑了",
            "awsl",
            "太草了",
            "wtf",
            "yyds",
        }

    def record_message(
        self,
        *,
        user_id: str,
        text: str = "",
        topics: Optional[List[str]] = None,
        thread_hint: str = "",
        mentioned_users: Optional[List[str]] = None,
        reply_anchor: str = "",
    ) -> None:
        """记录一条消息到场景"""
        now = time.time()
        normalized_text = (text or "").strip()
        resolved_mentions = [
            str(uid).strip()
            for uid in (mentioned_users or [])
            if str(uid).strip() and str(uid).strip() != user_id
        ]
        resolved_topics = topics or []
        if not resolved_topics and not resolved_mentions and not reply_anchor:
            resolved_topics = self._extract_topic_hints(normalized_text)
        inferred_thread = thread_hint or self._infer_thread_hint(
            user_id=user_id,
            text=normalized_text,
            topics=resolved_topics,
            mentioned_users=resolved_mentions,
            reply_anchor=reply_anchor,
        )
        self._message_timestamps.append(now)
        if len(self._message_timestamps) > self._max_timestamps:
            self._message_timestamps = self._message_timestamps[
                -self._max_timestamps:
            ]
        if user_id not in self._speaker_timestamps:
            self._speaker_timestamps[user_id] = []
        self._speaker_timestamps[user_id].append(now)
        if len(self._speaker_timestamps[user_id]) > 50:
            self._speaker_timestamps[user_id] = self._speaker_timestamps[
                user_id
            ][-50:]
        # 话题更新
        if resolved_topics:
            for kw in resolved_topics:
                self._touch_topic(kw, user_id)
        # 线程更新
        if inferred_thread:
            self._touch_thread(
                inferred_thread, user_id, normalized_text, resolved_mentions
            )
        # 互动关系追踪
        for target_user in resolved_mentions:
            pair_key = self._make_pair_key(user_id, target_user)
            self._interaction_pairs[pair_key] = now
        # 玩梗命中追踪
        if normalized_text and any(
            kw in normalized_text for kw in self._meme_keywords
        ):
            self._recent_meme_hits.append(now)
            if len(self._recent_meme_hits) > 60:
                self._recent_meme_hits = self._recent_meme_hits[-60:]
        self._recent_messages.append(
            RecentSceneMessage(
                user_id=user_id,
                text=normalized_text,
                topics=list(resolved_topics),
                mentioned_users=list(resolved_mentions),
                thread_id=inferred_thread,
                timestamp=now,
            )
        )
        if len(self._recent_messages) > self._max_timestamps:
            self._recent_messages = self._recent_messages[
                -self._max_timestamps:
            ]

    def set_known_online_count(self, count: int) -> None:
        """设置外部提供的当前已知在线用户数"""
        self._known_online_count = max(0, count)

    def snapshot(self) -> SceneSnapshot:
        """生成当前场景快照"""
        self._prune_stale()
        now = time.time()
        snap = SceneSnapshot()
        # 消息频率（5分钟窗口）
        window_start = now - 300.0
        recent_msgs = [t for t in self._message_timestamps if t > window_start]
        snap.messages_per_minute = (
            len(recent_msgs) / 5.0 if recent_msgs else 0.0
        )
        # 活跃发言者
        recent_speakers = set()
        for uid, ts_list in self._speaker_timestamps.items():
            if any(t > window_start for t in ts_list):
                recent_speakers.add(uid)
        snap.unique_speakers_5min = len(recent_speakers)
        # 主导发言者（5分钟内消息最多的人）
        if self._speaker_timestamps:
            speaker_counts = {}
            for uid, ts_list in self._speaker_timestamps.items():
                count = sum(1 for t in ts_list if t > window_start)
                if count > 0:
                    speaker_counts[uid] = count
            if speaker_counts:
                snap.dominant_speaker = max(
                    speaker_counts, key=speaker_counts.get
                )
        # 话题和线程
        active_topics = [
            t for t in self._topics.values() if not t.is_stale(self._topic_ttl)
        ]
        snap.active_topic_count = len(active_topics)
        alive_threads = [
            t for t in self._threads.values() if t.is_alive(self._thread_ttl)
        ]
        snap.thread_count = len(alive_threads)
        snap.interaction_pairs = self._collect_recent_pairs(window_start)
        snap.auto_thread_inference = True
        # 氛围判定
        snap.atmosphere = self._judge_atmosphere(
            msg_rate=snap.messages_per_minute,
            speaker_count=snap.unique_speakers_5min,
            topic_count=snap.active_topic_count,
        )
        # 是否适合插话
        snap.suitable_to_join = self._evaluate_join_suitability(snap)
        snap.suitable_to_stay_silent = snap.atmosphere in (
            AtmosphereType.QUIET,
            AtmosphereType.MOURNING,
        )
        return snap

    def get_active_topics(self) -> List[Dict[str, Any]]:
        self._prune_stale()
        return [
            t.to_dict()
            for t in self._topics.values()
            if not t.is_stale(self._topic_ttl)
        ]

    def active_topics(self, limit: int = 5) -> List[str]:
        """按热度返回当前活跃话题关键词。"""
        self._prune_stale()
        topic_list = [
            t for t in self._topics.values() if not t.is_stale(self._topic_ttl)
        ]
        topic_list.sort(
            key=lambda item: (item.mention_count, item.last_mentioned),
            reverse=True,
        )
        safe_limit = max(1, limit)
        return [item.keyword for item in topic_list[:safe_limit]]

    def get_alive_threads(self) -> List[Dict[str, Any]]:
        self._prune_stale()
        return [
            t.to_dict()
            for t in self._threads.values()
            if t.is_alive(self._thread_ttl)
        ]

    # ────────────────── 内部方法 ──────────────────

    def _touch_topic(self, keyword: str, user_id: str) -> None:
        normalized = keyword.strip().lower()
        if not normalized:
            return
        if normalized in self._topics:
            slot = self._topics[normalized]
            slot.last_mentioned = time.time()
            slot.mention_count += 1
            slot.contributors.add(user_id)
            # 更新热度
            if slot.mention_count >= 10:
                slot.heat = TopicHeat.EXPLOSIVE
            elif slot.mention_count >= 5:
                slot.heat = TopicHeat.HOT
            elif slot.mention_count >= 2:
                slot.heat = TopicHeat.WARM
        else:
            self._topics[normalized] = TopicSlot(
                keyword=normalized,
                contributors={user_id},
            )

    def _touch_thread(
        self,
        thread_hint: str,
        user_id: str,
        text: str,
        mentioned_users: Optional[List[str]] = None,
    ) -> None:
        if thread_hint not in self._threads:
            self._threads[thread_hint] = ThreadMarker(
                thread_id=thread_hint,
                participants={user_id},
                topic_hint=text[:30],
            )
        marker = self._threads[thread_hint]
        marker.participants.add(user_id)
        for target_user in mentioned_users or []:
            marker.target_users.add(target_user)
            marker.participants.add(target_user)
        marker.last_active = time.time()
        marker.message_count += 1

    def _prune_stale(self) -> None:
        stale_topics = [
            k
            for k, v in self._topics.items()
            if v.is_stale(self._topic_ttl * 2)
        ]
        for k in stale_topics:
            del self._topics[k]
        dead_threads = [
            k
            for k, v in self._threads.items()
            if not v.is_alive(self._thread_ttl * 2)
        ]
        for k in dead_threads:
            del self._threads[k]
        # 清除不活跃发言者
        cutoff = time.time() - 600
        inactive = [
            uid
            for uid, ts in self._speaker_timestamps.items()
            if not ts or ts[-1] < cutoff
        ]
        for uid in inactive:
            del self._speaker_timestamps[uid]
        stale_pairs = [
            pair
            for pair, ts in self._interaction_pairs.items()
            if ts < time.time() - self._thread_ttl * 2
        ]
        for pair in stale_pairs:
            del self._interaction_pairs[pair]
        self._recent_messages = [
            msg
            for msg in self._recent_messages
            if msg.timestamp >= time.time() - self._thread_ttl * 2
        ]

    def _extract_topic_hints(self, text: str) -> List[str]:
        """从文本中抽取轻量话题提示，避免场景层完全依赖外部话题标注。"""
        if not text:
            return []
        compact = "".join(ch for ch in text if not ch.isspace())
        hints: List[str] = []
        if len(compact) >= 4:
            hints.append(compact[:6].lower())
        if "#" in text:
            for piece in text.split("#"):
                normalized = piece.strip()[:10].lower()
                if normalized:
                    hints.append(normalized)
        deduped: List[str] = []
        for hint in hints:
            if hint and hint not in deduped:
                deduped.append(hint)
        return deduped[:3]

    def _infer_thread_hint(
        self,
        *,
        user_id: str,
        text: str,
        topics: List[str],
        mentioned_users: List[str],
        reply_anchor: str,
    ) -> str:
        """在线索不足时推断线程键，减少对外部 thread_hint 的硬依赖。"""
        if mentioned_users:
            participants = sorted({user_id, *mentioned_users})
            return f"mention::{','.join(participants)}"
        if reply_anchor:
            return f"reply::{reply_anchor[:24]}"
        implicit_reply_thread = self._infer_thread_from_recent_context(
            user_id=user_id,
            text=text,
            topics=topics,
        )
        if implicit_reply_thread:
            return implicit_reply_thread
        recent_thread = self._find_recent_thread_for_user(
            user_id, max_gap_sec=90.0
        )
        if recent_thread:
            return recent_thread
        for topic in topics:
            normalized = topic.strip().lower()
            if not normalized:
                continue
            for marker in self._threads.values():
                if not marker.is_alive(self._thread_ttl):
                    continue
                if (
                    normalized in marker.thread_id
                    or normalized in marker.topic_hint.lower()
                ):
                    return marker.thread_id
            return f"topic::{normalized}"
        return ""

    def _infer_thread_from_recent_context(
        self,
        *,
        user_id: str,
        text: str,
        topics: List[str],
    ) -> str:
        """从最近消息中推断隐式回复线程。"""
        now = time.time()
        for recent in reversed(self._recent_messages[-12:]):
            if not recent.thread_id:
                continue
            if recent.user_id == user_id:
                continue
            if now - recent.timestamp > 75.0:
                continue
            score = 0.0
            if topics and recent.topics and set(topics) & set(recent.topics):
                score += 0.7
            score += self._text_overlap_score(text, recent.text) * 0.8
            if user_id in recent.mentioned_users:
                score += 0.9
            if score >= 0.75:
                return recent.thread_id
        return ""

    def _text_overlap_score(self, left: str, right: str) -> float:
        """计算两段短文本的轻量重叠度。"""
        left_compact = "".join(ch for ch in (left or "") if not ch.isspace())
        right_compact = "".join(ch for ch in (right or "") if not ch.isspace())
        if not left_compact or not right_compact:
            return 0.0
        if left_compact in right_compact or right_compact in left_compact:
            return 1.0
        left_grams = {
            left_compact[i: i + 2]
            for i in range(max(0, len(left_compact) - 1))
        }
        right_grams = {
            right_compact[i: i + 2]
            for i in range(max(0, len(right_compact) - 1))
        }
        if not left_grams or not right_grams:
            return 0.0
        intersection = len(left_grams & right_grams)
        union = len(left_grams | right_grams)
        if union <= 0:
            return 0.0
        return intersection / union

    def _find_recent_thread_for_user(
        self, user_id: str, max_gap_sec: float = 180.0
    ) -> str:
        """复用用户最近参与过的活跃线程。"""
        newest_thread_id = ""
        newest_ts = 0.0
        now = time.time()
        for thread_id, marker in self._threads.items():
            if user_id not in marker.participants:
                continue
            if not marker.is_alive(self._thread_ttl):
                continue
            if now - marker.last_active > max_gap_sec:
                continue
            if marker.last_active > newest_ts:
                newest_thread_id = thread_id
                newest_ts = marker.last_active
        return newest_thread_id

    def _make_pair_key(self, user_id: str, target_user: str) -> str:
        return f"{user_id}->{target_user}"

    def _collect_recent_pairs(self, window_start: float) -> List[str]:
        pairs = [
            pair
            for pair, ts in self._interaction_pairs.items()
            if ts > window_start
        ]
        pairs.sort()
        return pairs[:8]

    def _judge_atmosphere(
        self,
        *,
        msg_rate: float,
        speaker_count: int,
        topic_count: int,
    ) -> AtmosphereType:
        now = time.time()
        hour = time.localtime(now).tm_hour
        # 第三阶段新增：时间段特殊氛围
        if 6 <= hour < 9:
            if msg_rate > 1.0 and speaker_count >= 3:
                return AtmosphereType.MORNING_COFFEE
        elif 22 <= hour < 24 or 0 <= hour < 4:
            if msg_rate > 0.5 and speaker_count >= 2:
                return AtmosphereType.NIGHT_OWLS
        # 玩梗检测：5分钟内梗词命中 >= 4 次
        window_start = now - 300.0
        meme_hits_recent = sum(
            1 for t in self._recent_meme_hits if t > window_start
        )
        if meme_hits_recent >= 4 and speaker_count >= 2:
            return AtmosphereType.MEMEING
        # 围观检测：少数人说话 + 已知在线用户多
        if (
            self._known_online_count >= 5
            and speaker_count <= 2
            and msg_rate >= 1.0
            and self._known_online_count > speaker_count * 3
        ):
            return AtmosphereType.SPECTATING
        # 深度讨论：话题多 + 发言频率适中 + 发言者稳定
        if topic_count >= 4 and 2.0 <= msg_rate <= 6.0 and speaker_count >= 4:
            return AtmosphereType.DEEP_DISCUSSION
        # 讲故事：单一话题 + 某人主导发言
        if topic_count == 1 and msg_rate >= 3.0 and speaker_count == 1:
            return AtmosphereType.STORYTELLING
        # 解决问题：话题集中 + 快速交互
        if topic_count <= 2 and msg_rate >= 4.0 and speaker_count >= 3:
            return AtmosphereType.PROBLEM_SOLVING
        # 社交游戏：高频 + 多话题 + 娱乐关键词
        if msg_rate >= 5.0 and topic_count >= 3 and meme_hits_recent >= 2:
            return AtmosphereType.SOCIAL_GAMING
        # 情感支持：低频 + 特定关键词
        support_keywords = {"安慰", "难过", "支持", "加油", "没事", "抱抱"}
        recent_texts = [
            msg.text
            for msg in self._recent_messages
            if msg.timestamp > window_start and msg.text
        ]
        support_hit_count = sum(
            1 for txt in recent_texts for kw in support_keywords if kw in txt
        )
        if msg_rate < 2.0 and speaker_count >= 2 and support_hit_count >= 2:
            return AtmosphereType.EMOTIONAL_SUPPORT
        # 原有逻辑
        if msg_rate < 0.5:
            return AtmosphereType.QUIET
        if msg_rate < 2.0:
            if speaker_count <= 2:
                return AtmosphereType.CASUAL_CHAT
            return AtmosphereType.BORED_DRIFT
        if msg_rate < 5.0:
            if topic_count >= 3:
                return AtmosphereType.HEATED_DISCUSSION
            return AtmosphereType.CASUAL_CHAT
        # 高频消息
        if speaker_count <= 2 and msg_rate > 8.0:
            return AtmosphereType.ARGUMENT
        if topic_count <= 1 and msg_rate > 6.0:
            return AtmosphereType.HEATED_DISCUSSION
        return AtmosphereType.HEATED_DISCUSSION

    def _evaluate_join_suitability(self, snap: SceneSnapshot) -> bool:
        if snap.atmosphere == AtmosphereType.ARGUMENT:
            return False
        if snap.atmosphere == AtmosphereType.QUIET:
            return True
        if snap.messages_per_minute > 10.0:
            return False
        return True


def get_group_scene(channel_id: str) -> GroupSceneState:
    if channel_id not in _scene_instances:
        _scene_instances[channel_id] = GroupSceneState(channel_id)
    return _scene_instances[channel_id]


def remove_group_scene(channel_id: str) -> None:
    _scene_instances.pop(channel_id, None)

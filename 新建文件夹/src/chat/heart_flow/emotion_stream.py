"""
频道氛围追踪器 — 管理每条频道的情绪脉搏

融合查询频率监测、重复内容检测、工具调用追踪与社交反馈四维度，
维护 vitality/weariness/vexation 三轴数值随时间自然衰减恢复。
所有阈值参数从 CoreSettingsHub 读取，禁止硬编码。
"""

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Optional, Tuple
from src.common.logger import get_logger
from src.config.core_config_engine import get_core_config

logger = get_logger("mood_tracker")


class MoodCategory(Enum):
    """氛围类别枚举"""

    CALM = "calm"
    INQUISITIVE = "inquisitive"
    LIVELY = "lively"
    FATIGUED = "fatigued"
    IRRITATED = "irritated"
    SWAMPED = "swamped"


@dataclass
class InquiryStamp:
    """单条查询印记"""

    moment: float = 0.0
    author_id: str = ""
    digest: str = ""
    excerpt: str = ""
    privileged: bool = False


@dataclass
class ToolCallStamp:
    """工具调用印记"""

    moment: float = 0.0
    tool_label: str = ""
    privileged: bool = False


@dataclass
class MoodLedger:
    """频道氛围账本 — 记录一条频道的完整情绪状态"""

    category: MoodCategory = MoodCategory.CALM
    inquiry_tally: int = 0
    tool_call_tally: int = 0
    latest_inquiry_at: float = 0.0
    duplicate_hits: int = 0
    privileged_ratio: float = 0.0
    inquiry_archive: Deque[InquiryStamp] = field(
        default_factory=lambda: deque(maxlen=80)
    )
    tool_archive: Deque[ToolCallStamp] = field(
        default_factory=lambda: deque(maxlen=40)
    )
    last_category_refresh: float = 0.0
    intensity: float = 0.0
    vitality: float = 0.0
    weariness: float = 0.0
    vexation: float = 0.0
    last_recovery_tick: float = 0.0


class ChannelMoodTracker:
    """
    频道氛围追踪器
    单例管理所有频道的查询频率、重复检测、工具调用记录与社交反馈。
    内部维护 vitality/weariness/vexation 三轴数值，随时间自动回复。
    """

    _solo: Optional["ChannelMoodTracker"] = None

    @classmethod
    def instance(cls) -> "ChannelMoodTracker":
        if cls._solo is None:
            cls._solo = cls()
        return cls._solo

    @classmethod
    def teardown(cls) -> None:
        if cls._solo is not None:
            cls._solo._books.clear()
        cls._solo = None

    def __init__(self):
        self._books: Dict[str, MoodLedger] = {}
        self._cfg_cache: Dict[str, Any] = {}
        self._cfg_ts: float = 0.0
        self._reload_params()

    # ---- 配置加载 ----

    def _reload_params(self) -> None:
        """从核心配置重新加载所有阈值参数"""
        now = time.time()
        if now - self._cfg_ts < 60.0:
            return
        hub = get_core_config()
        blk = hub.emotion_stream_block()
        self._cfg_cache = {
            "window_sec": float(blk.get("query_window_seconds", 300.0)),
            "max_per_window": int(blk.get("max_queries_per_window", 50)),
            "dup_threshold": int(blk.get("repeat_detection_threshold", 3)),
            "refresh_gap": float(blk.get("emotion_refresh_interval", 60.0)),
            "recent_sec": float(blk.get("recent_query_window_seconds", 60.0)),
            "annoyed_dup": int(blk.get("annoyed_repeat_count", 3)),
            "social_scale": float(blk.get("social_delta_scale", 0.02)),
        }
        pool = hub.dual_pool_energy()
        decay_section = pool.get("decay", {})
        self._cfg_cache["vita_regen"] = float(
            decay_section.get("chat_recovery_per_min", 5.0)
        )
        self._cfg_cache["wear_decay"] = float(
            decay_section.get("fatigue_decay_per_min", 3.0)
        )
        self._cfg_cache["vex_decay"] = float(
            decay_section.get("annoyance_decay_per_min", 3.0)
        )
        self._cfg_ts = now

    def _p(self, key: str) -> Any:
        """快速取参数"""
        self._reload_params()
        return self._cfg_cache[key]

    # ---- 账本管理 ----

    def _obtain_book(self, channel_id: str) -> MoodLedger:
        """获取或创建频道账本"""
        if channel_id not in self._books:
            now = time.time()
            self._books[channel_id] = MoodLedger(
                vitality=100.0,
                last_recovery_tick=now,
            )
        return self._books[channel_id]

    # ---- 时间恢复 ----

    def _apply_time_restoration(self, book: MoodLedger) -> None:
        """按流逝时间恢复 vitality 并衰减 weariness/vexation"""
        now = time.time()
        gap_min = (now - book.last_recovery_tick) / 60.0
        if gap_min < 0.1:
            return
        book.vitality = min(
            100.0, book.vitality + gap_min * self._p("vita_regen")
        )
        book.weariness = max(
            0.0, book.weariness - gap_min * self._p("wear_decay")
        )
        book.vexation = max(
            0.0, book.vexation - gap_min * self._p("vex_decay")
        )
        book.last_recovery_tick = now

    # ---- 文本相似度 ----

    @staticmethod
    def _text_resemblance(text_a: str, text_b: str) -> float:
        """基于词集合 Jaccard 的简易相似度"""
        if not text_a or not text_b:
            return 0.0
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            tokens_a = set(text_a.lower())
            tokens_b = set(text_b.lower())
        overlap = len(tokens_a & tokens_b)
        total = len(tokens_a | tokens_b)
        return overlap / total if total else 0.0

    # ---- 公共接口：记录查询 ----

    def record_inquiry(
        self,
        channel_id: str,
        content: str,
        author_id: str,
        privileged: bool = False,
    ) -> Tuple[int, float]:
        """
        记录一次查询
        返回 (重复次数, 最高相似度)
        重复检测采用 MD5摘要 + Jaccard 文本相似度双重策略
        """
        book = self._obtain_book(channel_id)
        self._apply_time_restoration(book)
        now = time.time()
        trimmed = content[:60]
        digest = hashlib.md5(trimmed.encode()).hexdigest()[:12]
        stamp = InquiryStamp(
            moment=now,
            author_id=author_id,
            digest=digest,
            excerpt=trimmed,
            privileged=privileged,
        )
        book.inquiry_archive.append(stamp)
        book.inquiry_tally += 1
        book.latest_inquiry_at = now
        # 跳过极短文本的重复检测
        dup_count = 0
        peak_sim = 0.0
        if len(content.strip()) > 3:
            cutoff_recent = now - self._p("recent_sec")
            cutoff_window = now - self._p("window_sec")
            for older in book.inquiry_archive:
                if older.moment >= now or older.moment < cutoff_window:
                    continue
                # 摘要匹配
                if older.digest == digest:
                    dup_count += 1
                    peak_sim = 1.0
                    continue
                # 文本相似匹配（仅近期窗口）
                if older.moment >= cutoff_recent and older.excerpt:
                    sim = self._text_resemblance(trimmed, older.excerpt)
                    peak_sim = max(peak_sim, sim)
                    if sim >= 0.85:
                        dup_count += 1
        # 累计重复命中
        if dup_count >= self._p("dup_threshold"):
            book.duplicate_hits += 1
        # 非特权用户的重复惩罚
        if not privileged and dup_count > 0:
            vita_cost = dup_count * 2
            wear_gain = dup_count * 2
            vex_gain = dup_count * 3
            book.vitality = max(0.0, book.vitality - vita_cost)
            book.weariness = min(100.0, book.weariness + wear_gain)
            book.vexation = min(100.0, book.vexation + vex_gain)
            logger.debug(
                f"[{channel_id[:8]}] 重复查询x{dup_count} "
                f"vitality-{vita_cost} weariness+{wear_gain} vexation+{vex_gain}"
            )
        self._recalculate_category(book)
        return dup_count, peak_sim

    # ---- 公共接口：记录工具调用 ----

    def record_tool_call(
        self,
        channel_id: str,
        tool_label: str,
        privileged: bool = False,
    ) -> Tuple[int, bool]:
        """
        记录工具调用
        返回 (短期内调用次数, 是否应拒绝执行)
        """
        book = self._obtain_book(channel_id)
        self._apply_time_restoration(book)
        now = time.time()
        book.tool_archive.append(
            ToolCallStamp(
                moment=now, tool_label=tool_label, privileged=privileged
            )
        )
        book.tool_call_tally += 1
        if privileged:
            return 0, False
        short_window = 300.0
        recent = sum(
            1
            for s in book.tool_archive
            if now - s.moment < short_window and s.tool_label == tool_label
        )
        should_refuse = False
        annoy_bump = 0
        wear_bump = 0
        if recent >= 10:
            should_refuse = True
            annoy_bump, wear_bump = 25, 15
        elif recent >= 7:
            annoy_bump, wear_bump = 15, 10
        elif recent >= 5:
            annoy_bump, wear_bump = 8, 5
        elif recent >= 3:
            annoy_bump, wear_bump = 3, 2
        if annoy_bump:
            book.vexation = min(100.0, book.vexation + annoy_bump)
            book.weariness = min(100.0, book.weariness + wear_bump)
        self._recalculate_category(book)
        return recent, should_refuse

    # ---- 公共接口：社交反馈 ----

    def record_social_feedback(self, channel_id: str, delta: float) -> None:
        """社交值变化映射到氛围变化"""
        if abs(delta) < 1e-6:
            return
        book = self._obtain_book(channel_id)
        boost = min(0.35, abs(delta) * self._p("social_scale"))
        if delta > 0:
            book.category = (
                MoodCategory.LIVELY
                if boost > 0.12
                else MoodCategory.INQUISITIVE
            )
            book.intensity = max(book.intensity, min(1.0, 0.5 + boost))
        else:
            book.category = (
                MoodCategory.IRRITATED
                if boost > 0.12
                else MoodCategory.FATIGUED
            )
            book.intensity = max(book.intensity, min(1.0, 0.45 + boost))
        book.last_category_refresh = time.time()

    # ---- 类别重评估 ----

    def _recalculate_category(self, book: MoodLedger) -> None:
        """根据频率、重复、工具使用综合评估氛围类别"""
        now = time.time()
        if now - book.last_category_refresh < self._p("refresh_gap"):
            return
        book.last_category_refresh = now
        window = self._p("window_sec")
        recent_stamps = [
            s for s in book.inquiry_archive if now - s.moment < window
        ]
        rate = len(recent_stamps) / (window / 60.0) if window else 0
        priv_n = sum(1 for s in recent_stamps if s.privileged)
        book.privileged_ratio = priv_n / max(1, len(recent_stamps))
        ceiling = self._p("max_per_window") / 5.0
        if rate > ceiling:
            book.category = MoodCategory.SWAMPED
            book.intensity = min(1.0, rate / 20.0)
        elif book.duplicate_hits > self._p("annoyed_dup"):
            book.category = MoodCategory.IRRITATED
            book.intensity = min(1.0, book.duplicate_hits / 10.0)
        elif rate > ceiling / 2.0:
            book.category = MoodCategory.FATIGUED
            book.intensity = 0.6
        elif book.tool_call_tally > 10:
            book.category = MoodCategory.INQUISITIVE
            book.intensity = 0.7
        elif rate > 2:
            book.category = MoodCategory.LIVELY
            book.intensity = 0.8
        else:
            book.category = MoodCategory.CALM
            book.intensity = 0.5

    # ---- 查询接口 ----

    def fetch_mood(self, channel_id: str) -> MoodLedger:
        """返回频道当前氛围账本（自动触发恢复计算）"""
        book = self._obtain_book(channel_id)
        self._apply_time_restoration(book)
        return book

    def compose_mood_hint(self, channel_id: str) -> str:
        """生成可插入提示词的氛围描述"""
        book = self.fetch_mood(channel_id)
        if (
            book.vitality >= 80
            and book.weariness <= 20
            and book.vexation <= 20
        ):
            if book.category == MoodCategory.CALM:
                return ""
        segments: list = []
        tool_usage: Dict[str, int] = {}
        for stamp in book.tool_archive:
            tool_usage[stamp.tool_label] = (
                tool_usage.get(stamp.tool_label, 0) + 1
            )
        for label, cnt in tool_usage.items():
            if cnt >= 2:
                segments.append(f"最近使用了{cnt}次「{label}」")
        if book.duplicate_hits > 0:
            segments.append(f"检测到{book.duplicate_hits}次重复内容")
        if not segments:
            return ""
        qualifier = ""
        if book.vexation > 70 or book.weariness > 70:
            qualifier = "当前状态疲惫且烦躁。"
        elif book.vexation > 50 or book.weariness > 50:
            qualifier = "有些疲倦和不耐烦。"
        elif book.vexation > 30 or book.weariness > 30:
            qualifier = "略微不耐烦。"
        if book.vitality < 30:
            qualifier += "聊天精力很低。"
        category_desc = {
            MoodCategory.CALM: "",
            MoodCategory.INQUISITIVE: "对话活跃，可以深入探索话题。",
            MoodCategory.LIVELY: "讨论热烈，可以积极参与互动。",
            MoodCategory.FATIGUED: "对话频率较高，可以适当放慢节奏。",
            MoodCategory.IRRITATED: "检测到重复内容，可以尝试新话题。",
            MoodCategory.SWAMPED: "消息量过大，建议简短回复。",
        }
        desc = category_desc.get(book.category, "")
        return f"[氛围感知] {qualifier}{desc}\n" + "\n".join(segments)

    def compose_repeat_alert(self, channel_id: str) -> str:
        """生成重复查询警告文本"""
        book = self.fetch_mood(channel_id)
        if book.duplicate_hits >= 3:
            return f"已经问了{book.duplicate_hits}次类似的问题，都回答过了"
        if book.duplicate_hits >= 2:
            return "这个问题刚才好像问过了"
        return ""

    # ---- 状态维护 ----

    def wipe_channel(self, channel_id: str) -> None:
        """清空频道全部氛围数据"""
        if channel_id in self._books:
            del self._books[channel_id]
            logger.info(f"频道氛围已重置 | {channel_id[:8]}")

    def dump_snapshot(self) -> Dict[str, Any]:
        """导出所有频道的氛围快照"""
        result = {}
        for ch, bk in self._books.items():
            result[ch] = {
                "category": bk.category.value,
                "inquiry_tally": bk.inquiry_tally,
                "tool_call_tally": bk.tool_call_tally,
                "duplicate_hits": bk.duplicate_hits,
                "privileged_ratio": round(bk.privileged_ratio, 4),
                "intensity": round(bk.intensity, 4),
                "vitality": round(bk.vitality, 2),
                "weariness": round(bk.weariness, 2),
                "vexation": round(bk.vexation, 2),
            }
        return result

    def load_snapshot(self, data: Dict[str, Any]) -> None:
        """从快照恢复所有频道氛围"""
        self._books.clear()
        now = time.time()
        for ch, vals in data.items():
            if not isinstance(vals, dict):
                continue
            book = MoodLedger(
                inquiry_tally=vals.get("inquiry_tally", 0),
                tool_call_tally=vals.get("tool_call_tally", 0),
                duplicate_hits=vals.get("duplicate_hits", 0),
                privileged_ratio=vals.get("privileged_ratio", 0.0),
                intensity=vals.get("intensity", 0.5),
                vitality=vals.get("vitality", 100.0),
                weariness=vals.get("weariness", 0.0),
                vexation=vals.get("vexation", 0.0),
                last_recovery_tick=now,
            )
            try:
                book.category = MoodCategory(vals.get("category", "calm"))
            except ValueError:
                book.category = MoodCategory.CALM
            self._books[ch] = book
        logger.info(f"氛围状态恢复完成 | 频道数:{len(self._books)}")

    def save_to_persistent_store(self) -> None:
        """将氛围快照写入 persistent_state_db"""
        snap = self.dump_snapshot()
        if not snap:
            return
        try:
            from src.chat.heart_flow.persistent_state_db import (
                acquire_persistent_store,
            )

            acquire_persistent_store().set_state(
                "mood", "__all_channels__", snap
            )
        except Exception as exc:
            logger.debug(f"氛围持久化失败: {exc}")

    def restore_from_persistent_store(self) -> bool:
        """从 persistent_state_db 恢复氛围快照"""
        try:
            from src.chat.heart_flow.persistent_state_db import (
                acquire_persistent_store,
            )

            data = acquire_persistent_store().get_state(
                "mood", "__all_channels__"
            )
            if not data or not isinstance(data, dict):
                return False
            self.load_snapshot(data)
            return True
        except Exception as exc:
            logger.debug(f"氛围恢复失败: {exc}")
            return False


def get_channel_mood_tracker() -> ChannelMoodTracker:
    """获取频道氛围追踪器单例"""
    return ChannelMoodTracker.instance()


# ============================================================
# D1 维度接口：情绪三轴（patience/annoyance/fatigue）
# 包装 ChannelMoodTracker 使其符合 DimensionBase 统一协议
# ============================================================
from src.chat.heart_flow.dimension_protocol import (
    DimensionBase,
    DimensionScope,
    EventContext,
    TickResult,
)
from src.chat.heart_flow.vote_types import EmotionAxisVote


class EmotionAxisDimension(DimensionBase):
    """
    D1 情绪三轴维度。
    将 ChannelMoodTracker 的 vitality/weariness/vexation
    映射到 V2 架构的 patience/annoyance/fatigue。
    映射关系:
      patience  = vitality (满值100=完全耐心)
      annoyance = vexation (0=平静, 100=极度烦躁)
      fatigue   = weariness (0=精力充沛, 100=极度疲劳)
    """

    _singleton: Optional["EmotionAxisDimension"] = None

    @classmethod
    def get_instance(cls) -> "EmotionAxisDimension":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        self._tracker = ChannelMoodTracker.instance()

    @property
    def dimension_name(self) -> str:
        return "emotion_axis"

    @property
    def scope(self) -> DimensionScope:
        return DimensionScope.PER_CHANNEL

    @property
    def needs_persistence(self) -> bool:
        return True

    @property
    def tick_interval_sec(self) -> float:
        return 6.0

    def tick(self, elapsed_sec: float) -> TickResult:
        """周期性驱动所有频道的氛围恢复/衰减"""
        updated = False
        for channel_id in list(self._tracker._books.keys()):
            book = self._tracker._books.get(channel_id)
            if book is None:
                continue
            old_v = book.vitality
            old_w = book.weariness
            old_x = book.vexation
            self._tracker._apply_time_restoration(book)
            if (
                abs(book.vitality - old_v) > 0.01
                or abs(book.weariness - old_w) > 0.01
                or abs(book.vexation - old_x) > 0.01
            ):
                updated = True
        return TickResult(
            dimension_name=self.dimension_name,
            updated=updated,
            summary=f"频道数={len(self._tracker._books)}" if updated else "",
        )

    def on_event(self, ctx: EventContext):
        """
        接收消息事件并更新情绪状态。
        message_received: 记录查询（触发重复检测和情绪消耗）
        reply_completed: 回复完成后增加疲劳
        """
        if ctx.event_type == "message_received" and ctx.channel_id:
            self._tracker.record_inquiry(
                channel_id=ctx.channel_id,
                content=ctx.message_text,
                author_id=ctx.user_id,
                privileged=ctx.is_admin,
            )
        elif ctx.event_type == "reply_completed" and ctx.channel_id:
            book = self._tracker._obtain_book(ctx.channel_id)
            fatigue_cost = 2.0 + ctx.reply_tokens * 0.01
            book.weariness = min(100.0, book.weariness + fatigue_cost)

    def vote(self, ctx: EventContext) -> EmotionAxisVote:
        """
        根据当前频道的情绪三轴生成投票。
        patience(vitality) 越低越不耐烦，annoyance(vexation) 越高越烦躁，
        fatigue(weariness) 越高越疲劳，各自独立影响概率和态度。
        """
        if not ctx.channel_id:
            return EmotionAxisVote()
        book = self._tracker.fetch_mood(ctx.channel_id)
        patience = book.vitality
        annoyance = book.vexation
        fatigue = book.weariness
        # 概率乘数由三轴共同决定（乘法组合）
        prob = 1.0
        # 耐心影响
        if patience < 10:
            prob *= 0.1
        elif patience < 30:
            prob *= 0.5
        elif patience < 50:
            prob *= 0.8
        # 烦躁影响
        if annoyance > 85:
            prob *= 0.2
        elif annoyance > 70:
            prob *= 0.5
        elif annoyance > 50:
            prob *= 0.8
        elif annoyance > 30:
            prob *= 0.9
        # 疲劳影响
        if fatigue > 80:
            prob *= 0.7
        elif fatigue > 60:
            prob *= 0.85
        # 态度标签
        attitude = ""
        if patience < 10:
            attitude = "refusing"
        elif patience < 30:
            attitude = "impatient"
        elif fatigue > 80:
            attitude = "exhausted"
        # 语气修饰
        mood_mod = ""
        if annoyance > 85:
            mood_mod = "furious"
        elif annoyance > 70:
            mood_mod = "very_annoyed"
        elif annoyance > 50:
            mood_mod = "annoyed"
        elif annoyance > 30:
            mood_mod = "slightly_irritated"
        # 是否建议拒绝
        should_refuse = patience < 10 or annoyance > 85
        # token上限（疲劳影响回复长度）
        tokens_cap = 0
        if fatigue > 80:
            tokens_cap = 100
        elif fatigue > 60:
            tokens_cap = 300
        return EmotionAxisVote(
            probability_factor=max(0.05, prob),
            force_refuse=should_refuse,
            attitude_tag=attitude,
            mood_modifier=mood_mod,
            should_refuse=should_refuse,
            patience_snapshot=round(patience, 1),
            annoyance_snapshot=round(annoyance, 1),
            fatigue_snapshot=round(fatigue, 1),
            max_tokens_cap=tokens_cap,
            debug_reason=f"p={patience:.0f} a={annoyance:.0f} f={fatigue:.0f}",
        )

    def serialize(self) -> dict:
        """导出所有频道的情绪状态"""
        return self._tracker.dump_snapshot()

    def deserialize(self, data: dict):
        """恢复所有频道的情绪状态"""
        if isinstance(data, dict) and data:
            self._tracker.load_snapshot(data)

    def calibrate(self, offline_seconds: float):
        """离线校准: 离线期间情绪自动恢复"""
        offline_minutes = offline_seconds / 60.0
        for book in self._tracker._books.values():
            book.vitality = min(100.0, book.vitality + offline_minutes * 2.0)
            book.weariness = max(0.0, book.weariness - offline_minutes * 3.0)
            book.vexation = max(0.0, book.vexation - offline_minutes * 3.0)
            book.last_recovery_tick = time.time()

    def get_state_summary(self) -> dict:
        """调试面板用的状态摘要"""
        summaries = {}
        for ch_id, book in self._tracker._books.items():
            summaries[ch_id[:8]] = {
                "patience": round(book.vitality, 1),
                "annoyance": round(book.vexation, 1),
                "fatigue": round(book.weariness, 1),
                "category": book.category.value,
            }
        return {
            "dimension": self.dimension_name,
            "scope": self.scope.value,
            "channel_count": len(self._tracker._books),
            "channels": summaries,
        }

    def reset(self, user_id: str = "", channel_id: str = ""):
        """重置指定频道的情绪状态"""
        if channel_id:
            self._tracker.wipe_channel(channel_id)

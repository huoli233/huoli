import time
import json
import hashlib
import threading
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from src.common.logger import get_logger

logger = get_logger("ctx_keeper")


# ==================== 时间标注体系 ====================

WEEKDAY_LABELS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


class TemporalTag(Enum):
    """时间感知标签（8级精度）。"""

    INSTANT = "instant"
    MINUTES_AGO = "minutes_ago"
    WITHIN_HOUR = "within_hour"
    SAME_DAY = "same_day"
    PRIOR_DAY = "prior_day"
    TWO_DAYS = "two_days"
    SAME_WEEK = "same_week"
    LONG_AGO = "long_ago"


def annotate_temporal(ts: float) -> Tuple[TemporalTag, str]:
    """为时间戳生成(标签, 人类可读描述)对。"""
    now = time.time()
    gap = now - ts
    dt = datetime.fromtimestamp(ts)
    wday = WEEKDAY_LABELS[dt.weekday()]
    hour_text = f"{dt.hour}点"
    date_text = dt.strftime("%m月%d日")
    if gap < 300:
        return TemporalTag.INSTANT, f"刚刚 ({hour_text})"
    elif gap < 1800:
        return TemporalTag.MINUTES_AGO, f"{int(gap / 60)}分钟前"
    elif gap < 3600:
        return TemporalTag.WITHIN_HOUR, f"一小时内 ({hour_text})"
    elif gap < 86400:
        return TemporalTag.SAME_DAY, f"今天 {hour_text}"
    elif gap < 172800:
        return TemporalTag.PRIOR_DAY, f"昨天 {wday} {hour_text}"
    elif gap < 259200:
        return TemporalTag.TWO_DAYS, f"前天 {wday} {hour_text}"
    elif gap < 604800:
        days = int(gap / 86400)
        return TemporalTag.SAME_WEEK, f"{days}天前 {wday}"
    return TemporalTag.LONG_AGO, f"{date_text} {wday}"


# ==================== 数据类 ====================


@dataclass
class ToolInvocationLog:
    """工具调用记录。"""

    tool_name: str
    tool_args: Dict[str, Any]
    outcome: Optional[str]
    succeeded: bool
    invoked_at: float
    elapsed_ms: int
    stream_id: str
    invoker_id: Optional[str]


@dataclass
class ContextSlice:
    """单条上下文片段。"""

    slice_id: str
    stream_id: str
    author_id: Optional[str]
    content_role: str
    body: str
    ts: float
    temporal_tag: str
    temporal_desc: str
    weekday: str
    hour: int
    date_str: str
    weight: float = 0.5
    tool_payload: Optional[Dict] = None
    attachments: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class AuthorContextBucket:
    """按作者分桶的上下文集合。"""

    author_id: str
    stream_id: str
    slices: List[ContextSlice] = field(default_factory=list)
    tool_logs: List[ToolInvocationLog] = field(default_factory=list)
    latest_activity_ts: float = 0.0
    msg_tally: int = 0
    tool_tally: int = 0


# ==================== 禁言守卫 ====================


class SilenceGuard:
    """管理频道禁言状态检测与自动解除。"""

    # 可疑关键词
    _MUTE_SIGNALS = ["禁言", "muted", "banned", "silence", "120", "1200"]

    def __init__(self):
        self._silenced: Dict[str, float] = {}  # {stream_id: until_ts}
        self._detection_tally: Dict[str, int] = {}
        self._rw_lock = threading.Lock()

    def is_silenced(self, stream_id: str) -> bool:
        """判断指定频道是否处于禁言状态。"""
        with self._rw_lock:
            if stream_id not in self._silenced:
                return False
            if time.time() >= self._silenced[stream_id]:
                del self._silenced[stream_id]
                self._detection_tally.pop(stream_id, None)
                return False
            return True

    def impose_silence(self, stream_id: str, duration: float = 3600.0):
        """手动设置禁言。"""
        with self._rw_lock:
            self._silenced[stream_id] = time.time() + duration
            self._detection_tally[stream_id] = (
                self._detection_tally.get(stream_id, 0) + 1
            )
            logger.info(f"频道 {stream_id[:8]}... 进入禁言 ({duration:.0f}秒)")

    def detect_from_error(self, stream_id: str, error_text: str) -> bool:
        """从错误消息中检测禁言信号，自动递增禁言时长。"""
        lowered = error_text.lower()
        for signal in self._MUTE_SIGNALS:
            if signal.lower() in lowered:
                count = self._detection_tally.get(stream_id, 0) + 1
                escalated_duration = min(3600 * count, 86400)
                self.impose_silence(stream_id, escalated_duration)
                return True
        return False

    def lift_silence(self, stream_id: str):
        """手动解除禁言。"""
        with self._rw_lock:
            self._silenced.pop(stream_id, None)
            self._detection_tally.pop(stream_id, None)
            logger.info(f"频道 {stream_id[:8]}... 禁言已解除")

    def remaining_seconds(self, stream_id: str) -> float:
        """获取剩余禁言秒数。"""
        if stream_id not in self._silenced:
            return 0.0
        return max(0.0, self._silenced[stream_id] - time.time())

    def list_all_silenced(self) -> Dict[str, float]:
        """列出所有被禁言的频道及剩余时长。"""
        result = {}
        now = time.time()
        with self._rw_lock:
            for sid, until_ts in list(self._silenced.items()):
                if until_ts > now:
                    result[sid] = until_ts - now
        return result


# ==================== 上下文保管器 ====================


class InstantContextKeeper:
    """即时上下文保管器 —— 双区（活跃区+归档区）管理对话上下文。

    特性：
    - 活跃区（active）：当前正在进行的对话内容
    - 归档区（archived）：高重要性条目自动归档，供长期检索
    - 自动节流持久化（最短间隔5秒）
    - 工具调用记录追踪
    - 容量溢出时自动转移至归档区
    """

    SLICE_CAP_PER_AUTHOR = 500
    TOOL_LOG_CAP_PER_AUTHOR = 200
    SAVE_THROTTLE = 5.0  # 最短保存间隔（秒）
    ARCHIVED_CAP = 300

    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self._active: Dict[str, AuthorContextBucket] = {}
        self._archived: Dict[str, AuthorContextBucket] = {}
        self._prev_save_ts = time.time()
        self._io_lock = threading.Lock()
        self._bootstrap_from_db()

    # ==================== 内部工具 ====================

    def _mint_slice_id(self, body: str) -> str:
        seed = f"{self.stream_id}:{body}:{time.time()}"
        return hashlib.md5(seed.encode()).hexdigest()[:16]

    @staticmethod
    def _resolve_cache_model():
        """惰性加载ContextCacheRecord模型。"""
        try:
            from src.common.database.database_model import ContextCacheRecord

            return ContextCacheRecord
        except Exception:
            return None

    # ==================== 消息存入 ====================

    def record_message(
        self,
        body: str,
        author_id: Optional[str] = None,
        content_role: str = "message",
        weight: float = 0.5,
        attachments: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """存入一条消息上下文。"""
        attachments = attachments or []
        metadata = metadata or {}
        weight = max(0.0, min(weight, 1.0))
        # 带附件/工具/系统事件 → 自动提升重要性
        if attachments:
            weight = max(weight, 0.7)
        is_tool_or_system = content_role in (
            "tool_call",
            "system",
            "bot_reply",
        )
        event_type = metadata.get("event_type", "")
        if is_tool_or_system or event_type in ("ban", "mute", "join", "leave"):
            weight = max(weight, 0.8)
        now = time.time()
        dt = datetime.fromtimestamp(now)
        tag, desc = annotate_temporal(now)
        ctx_slice = ContextSlice(
            slice_id=self._mint_slice_id(body),
            stream_id=self.stream_id,
            author_id=author_id,
            content_role=content_role,
            body=body,
            ts=now,
            temporal_tag=tag.value,
            temporal_desc=desc,
            weekday=WEEKDAY_LABELS[dt.weekday()],
            hour=dt.hour,
            date_str=dt.strftime("%Y-%m-%d"),
            weight=weight,
            attachments=attachments,
            metadata=metadata,
        )
        key = author_id or "system"
        if key not in self._active:
            self._active[key] = AuthorContextBucket(
                author_id=key, stream_id=self.stream_id, latest_activity_ts=now
            )
        bucket = self._active[key]
        bucket.slices.append(ctx_slice)
        bucket.latest_activity_ts = now
        bucket.msg_tally += 1
        # 容量溢出处理
        if len(bucket.slices) > self.SLICE_CAP_PER_AUTHOR:
            overflow = bucket.slices[
                : len(bucket.slices) - self.SLICE_CAP_PER_AUTHOR
            ]
            bucket.slices = bucket.slices[-self.SLICE_CAP_PER_AUTHOR:]
            self._migrate_to_archive(key, overflow)
        # 高重要性自动归档
        if weight >= 0.6:
            self._auto_archive_slice(key, ctx_slice)
        self._throttled_save()
        return ctx_slice.slice_id

    # ==================== 工具调用记录 ====================

    def record_tool_invocation(
        self,
        tool_name: str,
        tool_args: Dict,
        outcome: Optional[str],
        succeeded: bool,
        elapsed_ms: int,
        invoker_id: Optional[str] = None,
    ):
        """记录一次工具调用。"""
        now = time.time()
        log_entry = ToolInvocationLog(
            tool_name=tool_name,
            tool_args=tool_args,
            outcome=(
                outcome[:500] if outcome and len(outcome) > 500 else outcome
            ),
            succeeded=succeeded,
            invoked_at=now,
            elapsed_ms=elapsed_ms,
            stream_id=self.stream_id,
            invoker_id=invoker_id,
        )
        key = invoker_id or "system"
        if key not in self._active:
            self._active[key] = AuthorContextBucket(
                author_id=key, stream_id=self.stream_id, latest_activity_ts=now
            )
        bucket = self._active[key]
        bucket.tool_logs.append(log_entry)
        bucket.tool_tally += 1
        bucket.latest_activity_ts = now
        if len(bucket.tool_logs) > self.TOOL_LOG_CAP_PER_AUTHOR:
            bucket.tool_logs = bucket.tool_logs[
                -self.TOOL_LOG_CAP_PER_AUTHOR:
            ]
        self._throttled_save()

    # ==================== 归档管理 ====================

    def _migrate_to_archive(self, author_key: str, slices: List[ContextSlice]):
        """将溢出的片段迁移至归档区。"""
        if author_key not in self._archived:
            self._archived[author_key] = AuthorContextBucket(
                author_id=author_key, stream_id=self.stream_id
            )
        archive_bucket = self._archived[author_key]
        for s in slices:
            if s.weight >= 0.6 and all(
                e.slice_id != s.slice_id for e in archive_bucket.slices
            ):
                archive_bucket.slices.append(s)
        # 归档区容量控制
        if len(archive_bucket.slices) > self.ARCHIVED_CAP:
            keep = int(self.ARCHIVED_CAP * 0.7)
            archive_bucket.slices = archive_bucket.slices[-keep:]

    def _auto_archive_slice(self, author_key: str, ctx_slice: ContextSlice):
        """高重要性片段自动归档。"""
        if author_key not in self._archived:
            self._archived[author_key] = AuthorContextBucket(
                author_id=author_key, stream_id=self.stream_id
            )
        archive = self._archived[author_key]
        if all(e.slice_id != ctx_slice.slice_id for e in archive.slices):
            archive.slices.append(ctx_slice)
            if len(archive.slices) > self.ARCHIVED_CAP:
                keep = int(self.ARCHIVED_CAP * 0.7)
                archive.slices = archive.slices[-keep:]

    # ==================== 查询 ====================

    def fetch_author_context(
        self, author_id: str, ceiling: int = 50
    ) -> List[ContextSlice]:
        """获取指定作者的上下文片段（合并活跃区+归档区，按时间倒序）。"""
        key = author_id or "system"
        combined = []
        if key in self._active:
            combined.extend(self._active[key].slices)
        if key in self._archived:
            combined.extend(self._archived[key].slices)
        combined.sort(key=lambda s: s.ts, reverse=True)
        return combined[:ceiling]

    def list_known_authors(self) -> List[str]:
        """列出所有已知作者。"""
        authors = set()
        authors.update(self._active.keys())
        authors.update(self._archived.keys())
        return list(authors)

    def compile_stats(self) -> Dict[str, Any]:
        """编译统计摘要。"""
        active_slices = sum(len(b.slices) for b in self._active.values())
        active_tools = sum(len(b.tool_logs) for b in self._active.values())
        archived_slices = sum(len(b.slices) for b in self._archived.values())
        archived_tools = sum(len(b.tool_logs) for b in self._archived.values())
        return {
            "stream_id": self.stream_id,
            "authors": len(self.list_known_authors()),
            "active_slices": active_slices,
            "active_tools": active_tools,
            "archived_slices": archived_slices,
            "archived_tools": archived_tools,
            "last_save_ts": self._prev_save_ts,
        }

    # ==================== 持久化 ====================

    def _throttled_save(self):
        """节流保存：距上次保存<5秒则跳过。"""
        now = time.time()
        if now - self._prev_save_ts < self.SAVE_THROTTLE:
            return
        with self._io_lock:
            self._persist_zone(
                "cache", self._active, self.SLICE_CAP_PER_AUTHOR
            )
            self._persist_zone("pending", self._archived, self.ARCHIVED_CAP)
            self._prev_save_ts = now

    def force_save(self):
        """强制立即保存。"""
        with self._io_lock:
            self._persist_zone(
                "cache", self._active, self.SLICE_CAP_PER_AUTHOR
            )
            self._persist_zone("pending", self._archived, self.ARCHIVED_CAP)
            self._prev_save_ts = time.time()

    def _persist_zone(
        self,
        zone_type: str,
        zone_data: Dict[str, AuthorContextBucket],
        cap: int,
    ):
        """将指定区域写入数据库（事务保护，delete+insert 原子执行）。"""
        CacheModel = self._resolve_cache_model()
        if not CacheModel:
            return
        try:
            from src.common.database.database import db

            now = time.time()
            batch_rows = []
            for _key, bucket in zone_data.items():
                trimmed = (
                    bucket.slices[-cap:]
                    if len(bucket.slices) > cap
                    else bucket.slices
                )
                for s in trimmed:
                    batch_rows.append(
                        {
                            "stream_id": s.stream_id or self.stream_id,
                            "user_id": s.author_id or "",
                            "cache_type": zone_type,
                            "role": s.content_role,
                            "content": s.body,
                            "significance": s.weight,
                            "birth_ts": s.ts,
                            "expire_ts": None,
                            "refresh_ts": now,
                        }
                    )
            with db.atomic():
                CacheModel.delete().where(
                    (CacheModel.stream_id == self.stream_id)
                    & (CacheModel.cache_type == zone_type)
                ).execute()
                if batch_rows:
                    CacheModel.insert_many(batch_rows).execute()
        except Exception as exc:
            logger.error(f"上下文持久化失败 ({zone_type}): {exc}")

    def _bootstrap_from_db(self):
        """从数据库恢复活跃区和归档区。"""
        self._restore_zone("cache", self._active)
        self._restore_zone("pending", self._archived)

    def _restore_zone(
        self, zone_type: str, zone_target: Dict[str, AuthorContextBucket]
    ):
        """从数据库恢复指定区域。"""
        CacheModel = self._resolve_cache_model()
        if not CacheModel:
            return
        try:
            count = (
                CacheModel.select()
                .where(
                    (CacheModel.stream_id == self.stream_id)
                    & (CacheModel.cache_type == zone_type)
                )
                .count()
            )
            if count == 0:
                return
            rows = (
                CacheModel.select()
                .where(
                    (CacheModel.stream_id == self.stream_id)
                    & (CacheModel.cache_type == zone_type)
                )
                .order_by(CacheModel.birth_ts.asc())
            )
            for row in rows:
                uid = row.user_id or "unknown"
                if uid not in zone_target:
                    zone_target[uid] = AuthorContextBucket(
                        author_id=uid, stream_id=row.stream_id
                    )
                tag, desc = annotate_temporal(row.birth_ts)
                dt = datetime.fromtimestamp(row.birth_ts)
                ctx_slice = ContextSlice(
                    slice_id=f"{row.stream_id}_{uid}_{int(row.birth_ts)}",
                    stream_id=row.stream_id,
                    author_id=row.user_id,
                    content_role=row.role,
                    body=row.content,
                    ts=row.birth_ts,
                    temporal_tag=tag.value,
                    temporal_desc=desc,
                    weekday=WEEKDAY_LABELS[dt.weekday()],
                    hour=dt.hour,
                    date_str=dt.strftime("%Y-%m-%d"),
                    weight=row.significance,
                )
                zone_target[uid].slices.append(ctx_slice)
                zone_target[uid].latest_activity_ts = max(
                    zone_target[uid].latest_activity_ts, row.birth_ts
                )
                zone_target[uid].msg_tally += 1
            total = sum(len(b.slices) for b in zone_target.values())
            if total > 0:
                logger.debug(f"上下文恢复({zone_type}): {total}条")
        except Exception as exc:
            logger.debug(f"上下文恢复跳过({zone_type}): {exc}")

    # ==================== Prompt生成 ====================

    def compose_context_prompt(
        self, target_author: Optional[str] = None, token_budget: int = 600
    ) -> str:
        """为LLM生成上下文注入区块。"""
        sections = []
        if target_author:
            # 归档区高重要性条目
            archived_bucket = self._archived.get(target_author)
            if archived_bucket and archived_bucket.slices:
                notable = [
                    s for s in archived_bucket.slices[-20:] if s.weight >= 0.6
                ][:2]
                if notable:
                    section_lines = ["[用户历史行为]"]
                    for s in notable:
                        preview = s.body[:60] if len(s.body) > 60 else s.body
                        section_lines.append(f"  {s.temporal_desc}: {preview}")
                    sections.append("\n".join(section_lines))
            # 活跃区最近消息
            active_bucket = self._active.get(target_author)
            if active_bucket and active_bucket.slices:
                past = (
                    active_bucket.slices[-10:-1]
                    if len(active_bucket.slices) > 1
                    else active_bucket.slices[:-1]
                )
                past = past[-3:] if len(past) > 3 else past
                if past:
                    section_lines = ["[用户之前的行为]"]
                    for s in past:
                        preview = s.body[:60] if len(s.body) > 60 else s.body
                        section_lines.append(f"  {s.temporal_desc}: {preview}")
                    sections.append("\n".join(section_lines))
            # 其他用户最新群聊
            for other_key, other_bucket in self._active.items():
                if other_key != target_author and other_bucket.slices:
                    chat_msgs = [
                        s
                        for s in other_bucket.slices[-10:]
                        if s.content_role == "message"
                    ][:2]
                    if chat_msgs:
                        section_lines = ["[最新群聊内容]"]
                        for s in chat_msgs:
                            preview = (
                                s.body[:60] if len(s.body) > 60 else s.body
                            )
                            section_lines.append(
                                f"  [{s.author_id or '?'}] {preview}"
                            )
                        sections.append("\n".join(section_lines))
                    break
            # 用户最新发言
            if active_bucket and active_bucket.slices:
                newest = active_bucket.slices[-1]
                if newest.content_role == "message":
                    preview = (
                        newest.body[:80]
                        if len(newest.body) > 80
                        else newest.body
                    )
                    sections.append(
                        f"[用户最新发言]\n  {
                            newest.temporal_desc}: {preview}"
                    )
            # 心理状态注入
            psych_block = self._inject_psychological_state(target_author)
            if psych_block:
                sections.append(psych_block)
        combined = "\n".join(sections)
        if len(combined) > token_budget * 2:
            combined = combined[: token_budget * 2] + "...(已截断)"
        return combined

    def _inject_psychological_state(self, author_id: str) -> str:
        """尝试注入用户心理状态信息。"""
        try:
            from src.modules.modcore.psychological_core import (
                get_psychological_core,
            )

            tracker = get_psychological_core()
            if not tracker:
                return ""
            psych = (
                tracker.get_user_psychological_state(self.stream_id, author_id)
                if hasattr(tracker, "get_user_psychological_state")
                else None
            )
            if not psych:
                return ""
            indicators = []
            chaos = getattr(psych, "inner_chaos_level", 0)
            mask = getattr(psych, "surface_mask_strength", 10)
            stress = getattr(psych, "stress_accumulation", 0)
            trauma_count = len(getattr(psych, "trauma_timeline", []))
            current_state = getattr(psych, "current_state", None)
            state_label = current_state.value if current_state else "未知"
            has_signal = (
                chaos > 3 or mask < 7 or stress > 3 or trauma_count > 5
            )
            if not has_signal:
                return ""
            if chaos > 3:
                indicators.append(f"混乱{chaos:.0f}/10")
            if stress > 3:
                indicators.append(f"压力{stress:.0f}/10")
            if trauma_count > 3:
                indicators.append(f"累积{trauma_count}次")
            if mask < 6:
                indicators.append(f"伪装弱{mask:.0f}/10")
            if state_label in ("闪回发作", "彻底崩溃", "即将崩溃"):
                indicators.append(state_label)
            # 趋势分析
            if hasattr(psych, "get_state_trend"):
                trend = psych.get_state_trend(hours=24.0)
                if trend.get("entry_count", 0) > 3:
                    t = trend.get("trend", "")
                    if t == "worsening":
                        indicators.append("状态恶化")
                    elif t == "improving":
                        indicators.append("状态好转")
                    dominant = trend.get("dominant_state", "")
                    if dominant:
                        indicators.insert(0, dominant)
            if indicators:
                return f"[心理状态] {' | '.join(indicators)}"
            return ""
        except Exception:
            return ""


# ==================== 单例池管理 ====================

_keeper_pool: Dict[str, InstantContextKeeper] = {}
_silence_guard_solo: Optional[SilenceGuard] = None


def get_context_saver(stream_id: str) -> InstantContextKeeper:
    """获取指定会话流的上下文保管器。"""
    if stream_id not in _keeper_pool:
        _keeper_pool[stream_id] = InstantContextKeeper(stream_id)
    return _keeper_pool[stream_id]


def get_mute_manager() -> SilenceGuard:
    """获取全局禁言守卫单例。"""
    global _silence_guard_solo
    if _silence_guard_solo is None:
        _silence_guard_solo = SilenceGuard()
    return _silence_guard_solo


def should_respond(stream_id: str) -> bool:
    """判断指定频道是否可以发言（未被禁言）。"""
    guard = get_mute_manager()
    if guard.is_silenced(stream_id):
        remaining = guard.remaining_seconds(stream_id)
        logger.debug(
            f"频道 {stream_id[:8]}... 禁言中，剩余 {remaining:.0f} 秒"
        )
        return False
    return True


async def save_context_on_shutdown():
    """关闭时强制保存所有会话流的上下文。"""
    for keeper in _keeper_pool.values():
        keeper.force_save()
    logger.info(f"已保存 {len(_keeper_pool)} 个频道的上下文")

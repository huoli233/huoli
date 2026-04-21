import time
import json
import hashlib
import threading
from pathlib import Path
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from src.common.logger import get_logger

logger = get_logger("context_saver")


class TimeLabel(Enum):
    JUST_NOW = "just_now"
    RECENT = "recent"
    THIS_HOUR = "this_hour"
    TODAY = "today"
    YESTERDAY = "yesterday"
    BEFORE_YESTERDAY = "before_yesterday"
    THIS_WEEK = "this_week"
    EARLIER = "earlier"


WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def get_time_label(timestamp: float) -> Tuple[TimeLabel, str]:
    now = time.time()
    age_seconds = now - timestamp
    dt = datetime.fromtimestamp(timestamp)
    weekday = WEEKDAY_NAMES[dt.weekday()]
    hour_str = f"{dt.hour}点"
    date_str = dt.strftime("%m月%d日")
    if age_seconds < 300:
        return TimeLabel.JUST_NOW, f"刚才 ({hour_str})"
    elif age_seconds < 1800:
        mins = int(age_seconds / 60)
        return TimeLabel.RECENT, f"{mins}分钟前"
    elif age_seconds < 3600:
        return TimeLabel.THIS_HOUR, f"本小时 ({hour_str})"
    elif age_seconds < 86400:
        return TimeLabel.TODAY, f"今天 {hour_str}"
    elif age_seconds < 172800:
        return TimeLabel.YESTERDAY, f"昨天 {weekday} {hour_str}"
    elif age_seconds < 259200:
        return TimeLabel.BEFORE_YESTERDAY, f"前天 {weekday} {hour_str}"
    elif age_seconds < 604800:
        days_ago = int(age_seconds / 86400)
        return TimeLabel.THIS_WEEK, f"{days_ago}天前 {weekday}"
    return TimeLabel.EARLIER, f"{date_str} {weekday}"


@dataclass
class ToolCallRecord:
    tool_name: str
    tool_params: Dict[str, Any]
    result: Optional[str]
    success: bool
    called_at: float
    duration_ms: int
    stream_id: str
    user_id: Optional[str]


@dataclass
class ContextEntry:
    entry_id: str
    stream_id: str
    user_id: Optional[str]
    content_type: str
    content: str
    timestamp: float
    time_label: str
    time_desc: str
    weekday: str
    hour: int
    date_str: str
    importance: float = 0.5
    tool_data: Optional[Dict] = None
    media_urls: List[str] = field(default_factory=list)
    extra_data: Dict = field(default_factory=dict)


@dataclass
class UserContextGroup:
    user_id: str
    stream_id: str
    entries: List[ContextEntry] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    last_activity: float = 0.0
    total_messages: int = 0
    total_tool_calls: int = 0


class MuteStateManager:
    def __init__(self):
        self._muted_streams: Dict[str, float] = {}
        self._mute_detection_count: Dict[str, int] = {}
        self._lock = threading.Lock()

    def is_muted(self, stream_id: str) -> bool:
        with self._lock:
            if stream_id not in self._muted_streams:
                return False
            mute_until = self._muted_streams[stream_id]
            if time.time() >= mute_until:
                del self._muted_streams[stream_id]
                self._mute_detection_count.pop(stream_id, None)
                return False
            return True

    def set_muted(self, stream_id: str, duration_seconds: float = 3600.0):
        with self._lock:
            self._muted_streams[stream_id] = time.time() + duration_seconds
            self._mute_detection_count[stream_id] = self._mute_detection_count.get(stream_id, 0) + 1
            logger.info(f"频道 {stream_id[:8]}... 进入禁言状态 ({duration_seconds}秒)")

    def detect_mute_from_error(self, stream_id: str, error_msg: str) -> bool:
        mute_keywords = ["禁言", "muted", "banned", "silence", "120", "1200"]
        error_lower = error_msg.lower()
        for keyword in mute_keywords:
            if keyword.lower() in error_lower:
                count = self._mute_detection_count.get(stream_id, 0) + 1
                duration = min(3600 * count, 86400)
                self.set_muted(stream_id, duration)
                return True
        return False

    def clear_mute(self, stream_id: str):
        with self._lock:
            self._muted_streams.pop(stream_id, None)
            self._mute_detection_count.pop(stream_id, None)
            logger.info(f"频道 {stream_id[:8]}... 禁言状态已清除")

    def get_mute_remaining(self, stream_id: str) -> float:
        if stream_id not in self._muted_streams:
            return 0.0
        remaining = self._muted_streams[stream_id] - time.time()
        return max(0.0, remaining)

    def get_all_muted(self) -> Dict[str, float]:
        result = {}
        now = time.time()
        with self._lock:
            for sid, until_ts in list(self._muted_streams.items()):
                if until_ts > now:
                    result[sid] = until_ts - now
        return result


class InstantContextSaver:
    MAX_ENTRIES_PER_USER = 500
    MAX_TOOL_CALLS_PER_USER = 200
    AUTO_SAVE_INTERVAL = 30.0
    MAX_PENDING_FILE_SIZE = 2 * 1024 * 1024
    MAX_CACHE_FILE_SIZE = 2 * 1024 * 1024

    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self._cache: Dict[str, UserContextGroup] = {}
        self._pending: Dict[str, UserContextGroup] = {}
        self._last_save_time = time.time()
        self._save_lock = threading.Lock()
        self._load_from_disk()

    def _get_cache_key(self) -> str:
        safe_id = hashlib.md5(self.stream_id.encode()).hexdigest()[:16]
        return f"cache_{safe_id}"

    def _get_pending_key(self) -> str:
        safe_id = hashlib.md5(self.stream_id.encode()).hexdigest()[:16]
        return f"pending_{safe_id}"

    def _move_tool_calls_to_pending(self, entries: List[Dict]) -> List[Dict]:
        moved = []
        remaining = []
        for entry in entries:
            content = entry.get("content", "")
            if any(marker in content for marker in ["[工具调用]", "[tool_call]", "tool_use"]):
                moved.append(entry)
            else:
                remaining.append(entry)
        if moved:
            pending = self._load_pending()
            pending.extend(moved)
            self._save_pending(pending)
        return remaining

    def _get_entry_model(self):
        from src.common.database.database_model import ContextCacheEntry
        return ContextCacheEntry

    def _get_cache_file(self) -> Path:
        safe_id = hashlib.md5(self.stream_id.encode()).hexdigest()[:16]
        data_dir = Path(__file__).resolve().parents[2] / "data" / "context_cache"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / f"cache_{safe_id}.json"

    def _get_pending_file(self) -> Path:
        safe_id = hashlib.md5(self.stream_id.encode()).hexdigest()[:16]
        data_dir = Path(__file__).resolve().parents[2] / "data" / "context_cache"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / f"pending_{safe_id}.json"

    def _generate_entry_id(self, content: str) -> str:
        unique_str = f"{self.stream_id}:{content}:{time.time()}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]

    def save_message(self, content: str, user_id: Optional[str] = None,
                     content_type: str = "message", importance: float = 0.5,
                     media_urls: Optional[List[str]] = None,
                     extra_data: Optional[Dict] = None) -> str:
        media_urls = media_urls or []
        extra_data = extra_data or {}
        importance = max(0.0, min(importance, 1.0))
        has_media = bool(media_urls)
        is_tool_event = content_type == "tool_call" or extra_data.get("is_tool_event")
        is_system_event = content_type in ("system", "bot_reply") or extra_data.get("event_type") in {"ban", "mute", "join", "leave"}
        if has_media:
            importance = max(importance, 0.7)
        if is_tool_event or is_system_event:
            importance = max(importance, 0.8)
        now = time.time()
        dt = datetime.fromtimestamp(now)
        time_label, time_desc = get_time_label(now)
        entry = ContextEntry(
            entry_id=self._generate_entry_id(content),
            stream_id=self.stream_id, user_id=user_id,
            content_type=content_type, content=content,
            timestamp=now, time_label=time_label.value,
            time_desc=time_desc, weekday=WEEKDAY_NAMES[dt.weekday()],
            hour=dt.hour, date_str=dt.strftime("%Y-%m-%d"),
            importance=importance, media_urls=media_urls,
            extra_data=extra_data
        )
        user_key = user_id or "system"
        if user_key not in self._cache:
            self._cache[user_key] = UserContextGroup(
                user_id=user_key, stream_id=self.stream_id, last_activity=now
            )
        group = self._cache[user_key]
        group.entries.append(entry)
        group.last_activity = now
        group.total_messages += 1
        if len(group.entries) > self.MAX_ENTRIES_PER_USER:
            group.entries = group.entries[-self.MAX_ENTRIES_PER_USER:]
        if importance >= 0.6:
            self._push_high_importance_to_pending(user_key, entry)
        self._instant_save()
        return entry.entry_id

    def save_tool_call(self, tool_name: str, tool_params: Dict,
                       result: Optional[str], success: bool,
                       duration_ms: int, user_id: Optional[str] = None) -> None:
        now = time.time()
        record = ToolCallRecord(
            tool_name=tool_name, tool_params=tool_params,
            result=result[:500] if result and len(result) > 500 else result,
            success=success, called_at=now, duration_ms=duration_ms,
            stream_id=self.stream_id, user_id=user_id
        )
        user_key = user_id or "system"
        if user_key not in self._cache:
            self._cache[user_key] = UserContextGroup(
                user_id=user_key, stream_id=self.stream_id, last_activity=now
            )
        group = self._cache[user_key]
        group.tool_calls.append(record)
        group.total_tool_calls += 1
        group.last_activity = now
        if len(group.tool_calls) > self.MAX_TOOL_CALLS_PER_USER:
            group.tool_calls = group.tool_calls[-self.MAX_TOOL_CALLS_PER_USER:]
        self._instant_save()

    def _move_to_pending(self, user_key: str, entries: List[ContextEntry]):
        if user_key not in self._pending:
            self._pending[user_key] = UserContextGroup(user_id=user_key, stream_id=self.stream_id)
        pending_group = self._pending[user_key]
        for entry in entries:
            if entry.importance >= 0.6:
                if all(e.entry_id != entry.entry_id for e in pending_group.entries):
                    pending_group.entries.append(entry)
        self._trim_pending_if_needed()

    def _trim_pending_if_needed(self):
        pending_file = self._get_pending_file()
        if not pending_file.exists():
            return
        try:
            file_size = pending_file.stat().st_size
            if file_size <= self.MAX_PENDING_FILE_SIZE:
                return
            for user_key, group in self._pending.items():
                if len(group.entries) > 50:
                    keep_count = int(len(group.entries) * 0.7)
                    group.entries = group.entries[-keep_count:]
                if len(group.tool_calls) > 30:
                    keep_count = int(len(group.tool_calls) * 0.7)
                    group.tool_calls = group.tool_calls[-keep_count:]
            self._save_pending()
        except Exception as e:
            logger.error(f"删减待处理区失败: {e}")

    def _trim_cache_if_needed(self):
        cache_file = self._get_cache_file()
        if not cache_file.exists():
            return
        try:
            file_size = cache_file.stat().st_size
            if file_size <= self.MAX_CACHE_FILE_SIZE:
                return
            for user_key, group in self._cache.items():
                if len(group.entries) > 100:
                    old_entries = group.entries[:50]
                    group.entries = group.entries[50:]
                    self._move_to_pending(user_key, old_entries)
                if len(group.tool_calls) > 50:
                    group.tool_calls = group.tool_calls[30:]
        except Exception as e:
            logger.error(f"删减缓存区失败: {e}")

    def _push_high_importance_to_pending(self, user_key: str, entry: ContextEntry) -> None:
        if user_key not in self._pending:
            self._pending[user_key] = UserContextGroup(user_id=user_key, stream_id=self.stream_id)
        pending_group = self._pending[user_key]
        if all(e.entry_id != entry.entry_id for e in pending_group.entries):
            pending_group.entries.append(entry)
            self._trim_pending_if_needed()
            self._save_pending()

    def get_user_context(self, user_id: str, limit: int = 50) -> List[ContextEntry]:
        user_key = user_id or "system"
        entries = []
        if user_key in self._cache:
            entries.extend(self._cache[user_key].entries)
        if user_key in self._pending:
            entries.extend(self._pending[user_key].entries)
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    def get_all_users(self) -> List[str]:
        users = set()
        for user_key in self._cache.keys():
            users.add(user_key)
        for user_key in self._pending.keys():
            users.add(user_key)
        return list(users)

    def get_stats(self) -> Dict[str, Any]:
        cache_entries = sum(len(g.entries) for g in self._cache.values())
        cache_tools = sum(len(g.tool_calls) for g in self._cache.values())
        pending_entries = sum(len(g.entries) for g in self._pending.values())
        pending_tools = sum(len(g.tool_calls) for g in self._pending.values())
        cache_file = self._get_cache_file()
        pending_file = self._get_pending_file()
        cache_size = cache_file.stat().st_size if cache_file.exists() else 0
        pending_size = pending_file.stat().st_size if pending_file.exists() else 0
        return {
            "stream_id": self.stream_id, "users": len(self.get_all_users()),
            "cache_entries": cache_entries, "cache_tool_calls": cache_tools,
            "pending_entries": pending_entries, "pending_tool_calls": pending_tools,
            "cache_size_kb": round(cache_size / 1024, 1),
            "pending_size_kb": round(pending_size / 1024, 1),
            "last_save": self._last_save_time
        }

    def _instant_save(self):
        now = time.time()
        if now - self._last_save_time < 5.0:
            return
        with self._save_lock:
            self._save_cache()
            self._save_pending()
            self._last_save_time = now

    def force_save(self):
        with self._save_lock:
            self._save_cache()
            self._save_pending()
            self._last_save_time = time.time()

    def _save_entries_to_db(self, entries: List[ContextEntry], cache_type: str, max_per_user: int):
        CCE = self._get_entry_model()
        now = time.time()
        CCE.delete().where(
            (CCE.stream_id == self.stream_id) & (CCE.cache_type == cache_type)
        ).execute()
        if not entries:
            return
        trimmed = entries[-max_per_user:]
        batch = []
        for entry in trimmed:
            batch.append({
                "stream_id": entry.stream_id or self.stream_id,
                "user_id": entry.user_id or "",
                "cache_type": cache_type,
                "role": entry.content_type,
                "content": entry.content,
                "importance": entry.importance,
                "created_at": entry.timestamp,
                "expires_at": None,
                "updated_at": now,
            })
        if batch:
            CCE.insert_many(batch).execute()

    def _save_cache(self):
        try:
            all_entries = []
            for user_key, group in self._cache.items():
                trimmed = group.entries[-500:] if len(group.entries) > 500 else group.entries
                all_entries.extend(trimmed)
            self._save_entries_to_db(all_entries, "cache", 500)
        except Exception as e:
            logger.error(f"缓存保存失败: {e}")

    def _save_pending(self):
        try:
            all_entries = []
            for user_key, group in self._pending.items():
                trimmed = group.entries[-300:] if len(group.entries) > 300 else group.entries
                all_entries.extend(trimmed)
            self._save_entries_to_db(all_entries, "pending", 300)
        except Exception as e:
            logger.error(f"待处理区保存失败: {e}")

    def _load_from_disk(self):
        self._load_cache()
        self._load_pending()

    def _load_entries_from_db(self, cache_type: str) -> Dict[str, UserContextGroup]:
        CCE = self._get_entry_model()
        rows = CCE.select().where(
            (CCE.stream_id == self.stream_id) & (CCE.cache_type == cache_type)
        ).order_by(CCE.created_at.asc())
        groups: Dict[str, UserContextGroup] = {}
        for row in rows:
            uid = row.user_id or "unknown"
            if uid not in groups:
                groups[uid] = UserContextGroup(
                    user_id=uid, stream_id=row.stream_id,
                )
            label, desc = get_time_label(row.created_at)
            dt = datetime.fromtimestamp(row.created_at)
            entry = ContextEntry(
                entry_id=f"{row.stream_id}_{uid}_{int(row.created_at)}",
                stream_id=row.stream_id,
                user_id=row.user_id,
                content_type=row.role,
                content=row.content,
                timestamp=row.created_at,
                time_label=label.value,
                time_desc=desc,
                weekday=WEEKDAY_NAMES[dt.weekday()],
                hour=dt.hour,
                date_str=dt.strftime("%Y-%m-%d"),
                importance=row.importance,
            )
            groups[uid].entries.append(entry)
            groups[uid].last_activity = max(groups[uid].last_activity, row.created_at)
            groups[uid].total_messages += 1
        return groups

    def _load_cache(self):
        try:
            CCE = self._get_entry_model()
            count = CCE.select().where(
                (CCE.stream_id == self.stream_id) & (CCE.cache_type == "cache")
            ).count()
            if count > 0:
                self._cache = self._load_entries_from_db("cache")
                total = sum(len(g.entries) for g in self._cache.values())
                if total > 0:
                    logger.debug(f"已恢复 {total} 条缓存记录")
            else:
                self._migrate_json_file("cache")
        except Exception as e:
            logger.debug(f"缓存加载跳过: {e}")

    def _load_pending(self):
        try:
            CCE = self._get_entry_model()
            count = CCE.select().where(
                (CCE.stream_id == self.stream_id) & (CCE.cache_type == "pending")
            ).count()
            if count > 0:
                self._pending = self._load_entries_from_db("pending")
                total = sum(len(g.entries) for g in self._pending.values())
                if total > 0:
                    logger.debug(f"已恢复 {total} 条待处理记录")
            else:
                self._migrate_json_file("pending")
        except Exception as e:
            logger.debug(f"待处理区加载跳过: {e}")

    def _migrate_json_file(self, cache_type: str):
        if cache_type == "cache":
            target_file = self._get_cache_file()
            target_dict = self._cache
        else:
            target_file = self._get_pending_file()
            target_dict = self._pending
        if not target_file.exists():
            return
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for user_key, group_data in data.items():
                entries = [ContextEntry(**e) for e in group_data.get("entries", [])]
                target_dict[user_key] = UserContextGroup(
                    user_id=group_data["user_id"],
                    stream_id=group_data.get("stream_id", self.stream_id),
                    entries=entries,
                    last_activity=group_data.get("last_activity", 0),
                    total_messages=group_data.get("total_messages", len(entries)),
                )
            if cache_type == "cache":
                self._save_cache()
            else:
                self._save_pending()
            target_file.rename(target_file.with_suffix('.json.bak'))
            logger.info(f"上下文{cache_type}已从JSON迁移到结构化数据库")
        except Exception as e:
            logger.debug(f"JSON迁移跳过({cache_type}): {e}")

    def generate_context_prompt(self, user_id: Optional[str] = None, max_tokens: int = 600) -> str:
        lines = []
        if user_id:
            user_pending = self._pending.get(user_id)
            if user_pending and user_pending.entries:
                negative_entries = [e for e in user_pending.entries[-20:] if e.importance >= 0.6][:2]
                if negative_entries:
                    lines.append("[用户历史行为]")
                    for entry in negative_entries:
                        content = entry.content[:60] if len(entry.content) > 60 else entry.content
                        lines.append(f"- {entry.time_desc}: {content}")
            user_cache = self._cache.get(user_id)
            if user_cache and user_cache.entries:
                past_entries = user_cache.entries[-10:-1] if len(user_cache.entries) > 1 else user_cache.entries[:-1]
                past_entries = past_entries[-3:] if len(past_entries) > 3 else past_entries
                if past_entries:
                    lines.append("[用户之前的行为]")
                    for entry in past_entries:
                        content = entry.content[:60] if len(entry.content) > 60 else entry.content
                        lines.append(f"- {entry.time_desc}: {content}")
            for other_key, group in self._cache.items():
                if other_key != user_id and group.entries:
                    other_entries = [e for e in group.entries[-10:] if e.content_type == "message"][:2]
                    if other_entries:
                        lines.append("[最新群聊内容]")
                        for entry in other_entries:
                            content = entry.content[:60] if len(entry.content) > 60 else entry.content
                            lines.append(f"- [{entry.user_id or '?'}] {content}")
                    break
            if user_cache and user_cache.entries:
                latest_entry = user_cache.entries[-1]
                if latest_entry.content_type == "message":
                    lines.append("[用户最新发言]")
                    content = latest_entry.content[:80] if len(latest_entry.content) > 80 else latest_entry.content
                    lines.append(f"- {latest_entry.time_desc}: {content}")
            psych_context = self._get_psychological_context(user_id)
            if psych_context:
                lines.append(psych_context)
        result = "\n".join(lines)
        if len(result) > max_tokens * 2:
            result = result[:max_tokens * 2] + "...(已截断)"
        return result

    def _get_psychological_context(self, user_id: str) -> str:
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(self.stream_id)
            psych_state = tracker.get_psychology_state(user_id)
            if not psych_state:
                return ""
            trend = psych_state.get_state_trend(hours=24.0)
            trauma_count = len(psych_state.trauma_timeline)
            has_significant_state = (
                psych_state.inner_chaos_level > 3.0 or
                psych_state.surface_mask_strength < 7.0 or
                psych_state.stress_accumulation > 3.0 or
                trauma_count > 5 or
                psych_state.current_state.value != "表面正常"
            )
            if has_significant_state:
                state_parts = []
                if trend["entry_count"] > 0:
                    state_parts.append(trend["dominant_state"])
                else:
                    state_parts.append(psych_state.current_state.value)
                if psych_state.inner_chaos_level > 3.0:
                    state_parts.append(f"混乱{psych_state.inner_chaos_level:.0f}/10")
                if psych_state.stress_accumulation > 3.0:
                    state_parts.append(f"压力{psych_state.stress_accumulation:.0f}/10")
                if trauma_count > 3:
                    state_parts.append(f"累积{trauma_count}次")
                if psych_state.surface_mask_strength < 6.0:
                    state_parts.append(f"伪装弱{psych_state.surface_mask_strength:.0f}/10")
                if psych_state.current_state.value in ["闪回发作", "彻底崩溃", "即将崩溃"]:
                    state_parts.append(psych_state.current_state.value)
                if trend["entry_count"] > 3:
                    if trend["trend"] == "worsening":
                        state_parts.append("状态恶化")
                    elif trend["trend"] == "improving":
                        state_parts.append("状态好转")
                if state_parts:
                    return f"[心理状态] {' | '.join(state_parts)}"
            return ""
        except Exception:
            return ""


_context_savers: Dict[str, InstantContextSaver] = {}
_mute_manager: Optional[MuteStateManager] = None


def get_context_saver(stream_id: str) -> InstantContextSaver:
    if stream_id not in _context_savers:
        _context_savers[stream_id] = InstantContextSaver(stream_id)
    return _context_savers[stream_id]


def get_mute_manager() -> MuteStateManager:
    global _mute_manager
    if _mute_manager is None:
        _mute_manager = MuteStateManager()
    return _mute_manager


def should_respond(stream_id: str) -> bool:
    mute_mgr = get_mute_manager()
    if mute_mgr.is_muted(stream_id):
        remaining = mute_mgr.get_mute_remaining(stream_id)
        logger.debug(f"频道 {stream_id[:8]}... 处于禁言状态，剩余 {remaining:.0f} 秒")
        return False
    return True


async def save_context_on_shutdown():
    for saver in _context_savers.values():
        saver.force_save()
    logger.info(f"已保存 {len(_context_savers)} 个频道的上下文")

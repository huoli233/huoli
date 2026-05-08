import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from src.common.database.slot_storage import (
    delete_slot,
    load_slot,
    save_slot,
)
from src.common.logger import get_logger
from src.config.core_config_engine import get_core_config
from src.common.data_models.proactive_models import (
    DialogueStage,
    JournalKind,
    MindJournalEntry,
)

logger = get_logger("回忆录柜")

# 心理日志最大保留条数
_MAX_JOURNAL_CAPACITY = 60
_MEMOIR_SLOT_PREFIX = "memoir_vault"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class AnticipationSetup:
    """[已弃用] 等待配置 — 等待判断权威已迁移至 PendingOrchestrator。
    此结构仅为反序列化兼容保留，不再承担任何决策功能。"""

    expected_reply_text: str = ""
    max_wait_sec: int = 0
    commenced_at: float = 0.0
    latest_ponder_at: float = 0.0
    ponder_tally: int = 0

    def is_engaged(self) -> bool:
        return self.commenced_at > 0 and self.max_wait_sec > 0

    def elapsed_sec(self) -> float:
        if self.commenced_at <= 0:
            return 0.0
        return time.time() - self.commenced_at

    def wipe(self) -> None:
        self.expected_reply_text = ""
        self.max_wait_sec = 0
        self.commenced_at = 0.0
        self.latest_ponder_at = 0.0
        self.ponder_tally = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_reply_text": self.expected_reply_text,
            "max_wait_sec": self.max_wait_sec,
            "commenced_at": self.commenced_at,
            "latest_ponder_at": self.latest_ponder_at,
            "ponder_tally": self.ponder_tally,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnticipationSetup":
        return cls(
            expected_reply_text=str(data.get("expected_reply_text", "") or ""),
            max_wait_sec=_safe_int(data.get("max_wait_sec"), 0),
            commenced_at=_safe_float(data.get("commenced_at"), 0.0),
            latest_ponder_at=_safe_float(data.get("latest_ponder_at"), 0.0),
            ponder_tally=_safe_int(data.get("ponder_tally"), 0),
        )


class DialogueMemoir:
    """
    单用户对话回忆录
    维护状态(OPEN / ANTICIPATING)、心理日志、等待配置、统计计数器。
    """

    def __init__(self, user_id: str, channel_id: str):
        self.user_id: str = user_id
        self.channel_id: str = channel_id
        # 会话阶段
        self._phase: DialogueStage = DialogueStage.OPEN
        # 心理日志
        self._journal: List[MindJournalEntry] = []
        # 等待配置
        self.anticipation: AnticipationSetup = AnticipationSetup()
        # 时间戳
        self.born_at: float = time.time()
        self.touched_at: float = time.time()
        # 计数
        self.exchange_tally: int = 0
        self.last_initiative_at: float = 0.0
        self.consecutive_timeouts: int = 0
        self.last_user_spoke_at: float = 0.0
        self.last_topic: str = ""
        self.last_mood: str = ""

    # 兼容旧字段名：部分模块仍使用 last_human_ts
    @property
    def last_human_ts(self) -> float:
        return self.last_user_spoke_at

    @last_human_ts.setter
    def last_human_ts(self, value: float) -> None:
        self.last_user_spoke_at = float(value or 0.0)

    # ---- 阶段访问 ----
    @property
    def phase(self) -> DialogueStage:
        return self._phase

    @phase.setter
    def phase(self, new_phase: DialogueStage) -> None:
        if self._phase != new_phase:
            logger.debug(f"回忆录 {self.user_id[:8]} 阶段变更: {self._phase} -> {new_phase}")
        self._phase = new_phase

    def _pending_elapsed_sec(self) -> float:
        """从 PendingOrchestrator 读取等待时长（唯一权威），不再回退旧 anticipation。"""
        try:
            from src.chat.heart_flow.waiting_handler import (
                get_pending_orchestrator,
            )

            return float(get_pending_orchestrator().watch_elapsed_sec(self.channel_id) or 0.0)
        except Exception:
            return 0.0

    def _pending_limit_sec(self) -> int:
        """从 PendingOrchestrator 读取等待上限（唯一权威），不再回退旧 anticipation。"""
        try:
            from src.chat.heart_flow.waiting_handler import (
                get_pending_orchestrator,
            )

            setup = get_pending_orchestrator().fetch_setup(self.channel_id)
            if setup is not None:
                return int(getattr(setup, "max_wait_sec", 0) or 0)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return 0

    def is_waiting(self) -> bool:
        """等待判断唯一委托给 PendingOrchestrator，不再回退本地 anticipation。"""
        try:
            from src.chat.heart_flow.waiting_handler import (
                get_pending_orchestrator,
            )

            return bool(get_pending_orchestrator().is_watching(self.channel_id))
        except Exception:
            return False

    # ---- 日志写入 ----

    def _append_entry(self, entry: MindJournalEntry) -> None:
        self._journal.append(entry)
        self.touched_at = time.time()
        if len(self._journal) > _MAX_JOURNAL_CAPACITY:
            self._journal = self._journal[-_MAX_JOURNAL_CAPACITY:]

    def record_user_utterance(
        self,
        content: str,
        sender_name: str,
        sender_id: str,
        msg_time: Optional[float] = None,
    ) -> MindJournalEntry:
        """记录一条用户发言"""
        ts = msg_time or time.time()
        entry = MindJournalEntry(
            kind=JournalKind.USER_UTTERANCE,
            written_at=ts,
            body_text=content,
            speaker_name=sender_name,
            speaker_id=sender_id,
        )
        self.consecutive_timeouts = 0
        self.last_user_spoke_at = ts
        self.last_topic = (content or "").strip()[:120]
        # 如果正在等待，标注回复的时效性
        if self.is_waiting():
            elapsed = self._pending_elapsed_sec()
            limit = self._pending_limit_sec()
            tag = "timely" if elapsed <= limit else "delayed"
            entry.extras["reply_timeliness"] = tag
            entry.extras["wait_elapsed"] = round(elapsed, 1)
            entry.extras["wait_limit"] = limit
        self._append_entry(entry)
        return entry

    def log_bot_decision(
        self,
        inner_thought: str,
        behaviors: List[Dict],
        anticipated_reply: str = "",
        timeout_seconds: int = 0,
        decision_time: Optional[float] = None,
    ) -> MindJournalEntry:
        """记录机器人决策"""
        entry = MindJournalEntry(
            kind=JournalKind.BOT_VERDICT,
            written_at=decision_time or time.time(),
            inner_monologue=inner_thought,
            action_list=behaviors,
            expected_response=anticipated_reply,
            wait_limit_sec=timeout_seconds,
        )
        self._append_entry(entry)
        self.exchange_tally += 1
        return entry

    def record_reflection_shift(
        self,
        thought: str,
        mood_label: str = "",
        shift_time: Optional[float] = None,
    ) -> MindJournalEntry:
        """记录等待期间心理变化"""
        elapsed = self._pending_elapsed_sec()
        entry = MindJournalEntry(
            kind=JournalKind.IDLE_SHIFT,
            written_at=shift_time or time.time(),
            reflection=thought,
            mood_label=mood_label,
            elapsed_wait=elapsed,
        )
        if mood_label:
            self.last_mood = str(mood_label).strip()[:60]
        self._append_entry(entry)
        return entry

    # ---- 等待管控（已降级为纯日志，等待权威统一由 PendingOrchestrator 管理） ----

    def begin_awaiting(
        self,
        anticipated_reply: str,
        timeout_seconds: int,
    ) -> None:
        """记录等待开始事件（仅写日志，不维护独立等待状态）"""
        if timeout_seconds <= 0:
            return
        # 写一条日志留痕，不再修改 phase 和 anticipation
        entry = MindJournalEntry(
            kind=JournalKind.IDLE_SHIFT,
            written_at=time.time(),
            reflection=f"开始等待: {anticipated_reply[:60]}",
            mood_label="anticipation",
            elapsed_wait=0.0,
        )
        entry.extras["await_limit_sec"] = timeout_seconds
        entry.extras["await_expected"] = anticipated_reply[:80]
        self._append_entry(entry)
        logger.debug(
            f"回忆录 {self.user_id[:8]} 记录等待开始: limit={timeout_seconds}s expect={anticipated_reply[:30]}..."
        )

    def finish_awaiting(self) -> None:
        """记录等待结束事件（仅写日志）"""
        entry = MindJournalEntry(
            kind=JournalKind.IDLE_SHIFT,
            written_at=time.time(),
            reflection="等待结束",
            mood_label="neutral",
            elapsed_wait=self._pending_elapsed_sec(),
        )
        self._append_entry(entry)
        # 清理残余状态（兼容旧数据反序列化后的遗留值）
        if self._phase == DialogueStage.ANTICIPATING:
            self._phase = DialogueStage.OPEN
        self.anticipation.wipe()
        self.touched_at = time.time()

    # ---- 查询 ----

    def latest_entries(self, count: int = 20) -> List[MindJournalEntry]:
        """获取最近 N 条日志"""
        if count <= 0:
            return list(self._journal)
        return self._journal[-count:]

    def fetch_last_bot_reply(self) -> Optional[str]:
        """查找最近一条机器人发送的文本消息"""
        for entry in reversed(self._journal):
            if entry.kind == JournalKind.BOT_VERDICT:
                for act in entry.action_list:
                    if isinstance(act, dict) and act.get("type") in (
                        "respond",
                        "miao_reply",
                    ):
                        text = act.get("content", "")
                        if text:
                            return text
        return None

    # ---- 序列化 ----

    def serialize(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "phase": str(self._phase),
            "journal": [e.to_dict() for e in self._journal],
            "anticipation": self.anticipation.to_dict(),
            "born_at": self.born_at,
            "touched_at": self.touched_at,
            "exchange_tally": self.exchange_tally,
            "last_initiative_at": self.last_initiative_at,
            "consecutive_timeouts": self.consecutive_timeouts,
            "last_user_spoke_at": self.last_user_spoke_at,
            "last_topic": self.last_topic,
            "last_mood": self.last_mood,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "DialogueMemoir":
        obj = cls(
            user_id=data.get("user_id", ""),
            channel_id=data.get("channel_id", ""),
        )
        phase_str = data.get("phase", "open")
        try:
            obj._phase = DialogueStage(phase_str)
        except ValueError:
            obj._phase = DialogueStage.OPEN
        raw_journal = data.get("journal", [])
        journal: List[MindJournalEntry] = []
        for row in raw_journal:
            if not isinstance(row, dict):
                continue
            try:
                journal.append(MindJournalEntry.from_dict(row))
            except Exception as exc:
                logger.debug(f"回忆录日志恢复失败，已跳过: {exc}")
        obj._journal = journal
        obj.anticipation = AnticipationSetup.from_dict(data.get("anticipation", {}))
        obj.born_at = _safe_float(data.get("born_at"), time.time())
        obj.touched_at = _safe_float(data.get("touched_at"), time.time())
        obj.exchange_tally = _safe_int(data.get("exchange_tally"), 0)
        obj.last_initiative_at = _safe_float(
            data.get("last_initiative_at"), 0.0
        )
        obj.consecutive_timeouts = _safe_int(
            data.get("consecutive_timeouts"), 0
        )
        obj.last_user_spoke_at = _safe_float(
            data.get("last_user_spoke_at"), 0.0
        )
        obj.last_topic = str(data.get("last_topic", "") or "")
        obj.last_mood = str(data.get("last_mood", "") or "")
        return obj


class MemoirCabinet:
    """
    回忆录管理中心 — 管理所有用户的 DialogueMemoir
    提供获取、持久化、清理功能，使用用户级锁保证并发安全。
    """

    _solo: Optional["MemoirCabinet"] = None

    @classmethod
    def instance(cls) -> "MemoirCabinet":
        if cls._solo is None:
            cls._solo = cls()
        return cls._solo

    @classmethod
    def teardown(cls) -> None:
        cls._solo = None

    def __init__(self):
        hub = get_core_config()
        schedule_blk = hub.resolve_module_view("schedule").values
        self._vault_dir = f"Huoli.db:{_MEMOIR_SLOT_PREFIX}"
        self._stale_days = int(schedule_blk.get("session_max_age_days", 30))
        self._memoirs: Dict[str, DialogueMemoir] = {}
        self._user_locks: Dict[str, asyncio.Lock] = {}
        logger.info(f"回忆录柜初始化: {self._vault_dir}")

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        lock = self._user_locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._user_locks[user_id] = lock
        return lock

    def _build_filename(self, user_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
        return f"{_MEMOIR_SLOT_PREFIX}:{safe}"

    # ---- 获取 ----

    async def fetch_memoir(self, user_id: str, channel_id: str) -> DialogueMemoir:
        """获取或创建用户回忆录"""
        async with self._get_lock(user_id):
            if user_id in self._memoirs:
                existing = self._memoirs[user_id]
                existing.channel_id = channel_id
                return existing
            loaded = await self._load_from_disk(user_id)
            if loaded:
                loaded.channel_id = channel_id
                self._memoirs[user_id] = loaded
                return loaded
            fresh = DialogueMemoir(user_id=user_id, channel_id=channel_id)
            self._memoirs[user_id] = fresh
            logger.info(f"新建回忆录: {user_id[:8]}")
            return fresh

    def lookup_sync(self, user_id: str) -> Optional[DialogueMemoir]:
        """同步从内存中查找，不加载磁盘"""
        return self._memoirs.get(user_id)

    def lookup_by_channel(self, channel_id: str) -> Optional[DialogueMemoir]:
        """按频道ID查找最近的回忆录（群聊回退）"""
        _best = None
        _best_ts = 0.0
        for m in self._memoirs.values():
            if getattr(m, "channel_id", "") == channel_id:
                _ts = getattr(m, "last_user_spoke_at", 0.0) or 0.0
                if _ts > _best_ts:
                    _best_ts = _ts
                    _best = m
        return _best

    # ---- 兼容旧接口（同步）----
    def retrieve(self, user_id: str) -> Optional[DialogueMemoir]:
        """兼容旧调用：同步读取内存中的回忆录。"""
        return self.lookup_sync(user_id)

    def create(self, user_id: str, channel_id: Optional[str] = None) -> DialogueMemoir:
        """兼容旧调用：同步创建回忆录。"""
        memoir = self._memoirs.get(user_id)
        if memoir is None:
            memoir = DialogueMemoir(user_id=user_id, channel_id=channel_id or user_id)
            self._memoirs[user_id] = memoir
        elif channel_id:
            memoir.channel_id = channel_id
        return memoir

    def commit(self, user_id: str, memoir: DialogueMemoir) -> None:
        """兼容旧调用：同步写回内存。"""
        memoir.touched_at = time.time()
        self._memoirs[user_id] = memoir

    # ---- 持久化 ----
    async def persist_memoir(self, user_id: str) -> bool:
        """将指定用户回忆录写入磁盘"""
        async with self._get_lock(user_id):
            if user_id not in self._memoirs:
                return False
            memoir = self._memoirs[user_id]
            slot_key = self._build_filename(user_id)
            try:
                data = memoir.serialize()
                return save_slot(slot_key, data, ttl_days=max(1, self._stale_days))
            except Exception as exc:
                logger.error(f"持久化失败 {user_id[:8]}: {exc}")
                return False

    async def persist_all(self) -> int:
        """持久化全部回忆录，返回成功数"""
        ok_count = 0
        for uid in list(self._memoirs):
            if await self.persist_memoir(uid):
                ok_count += 1
        return ok_count

    async def _load_from_disk(self, user_id: str) -> Optional[DialogueMemoir]:
        slot_key = self._build_filename(user_id)
        data = load_slot(slot_key)
        if not isinstance(data, dict):
            return None
        try:
            memoir = DialogueMemoir.deserialize(data)
            logger.debug(f"从数据库加载回忆录: {user_id[:8]}")
            return memoir
        except Exception as exc:
            logger.error(f"加载失败 {user_id[:8]}: {exc}")
            return None

    # ---- 查询 ----

    async def list_anticipating(self) -> List[DialogueMemoir]:
        """列出处于等待阶段的回忆录（委托 PendingOrchestrator 判断真实等待状态）"""
        try:
            from src.chat.heart_flow.waiting_handler import (
                get_pending_orchestrator,
            )

            watching_channels = set(get_pending_orchestrator().all_watching_channels())
            return [m for m in self._memoirs.values() if m.channel_id in watching_channels]
        except Exception:
            # 回退：使用兼容等待判定
            return [m for m in self._memoirs.values() if m.is_waiting()]

    async def list_all(self) -> List[DialogueMemoir]:
        return list(self._memoirs.values())

    # ---- 清理 ----

    async def discard_memoir(self, user_id: str) -> bool:
        """移除用户回忆录（内存+磁盘）"""
        async with self._get_lock(user_id):
            if user_id in self._memoirs:
                del self._memoirs[user_id]
            slot_key = self._build_filename(user_id)
            if not delete_slot(slot_key):
                logger.debug(f"回忆录槽位不存在或删除失败: {user_id[:8]}")
        # 回忆录已删除，锁不再需要
        self._user_locks.pop(user_id, None)
        return True

    async def purge_stale(self) -> int:
        """清理超龄回忆录"""
        now = time.time()
        cutoff = self._stale_days * 86400
        removed = 0
        for uid in list(self._memoirs):
            m = self._memoirs[uid]
            if now - m.touched_at > cutoff:
                await self.discard_memoir(uid)
                removed += 1
                logger.info(f"清理超龄回忆录: {uid[:8]}")
        # 清理孤儿锁（对应回忆录已不在内存中）
        orphan_keys = [k for k in self._user_locks if k not in self._memoirs]
        for k in orphan_keys:
            self._user_locks.pop(k, None)
        return removed

    def aggregate_stats(self) -> Dict[str, Any]:
        total = len(self._memoirs)
        try:
            from src.chat.heart_flow.waiting_handler import (
                get_pending_orchestrator,
            )

            watching_channels = set(get_pending_orchestrator().all_watching_channels())
            waiting = sum(1 for m in self._memoirs.values() if m.channel_id in watching_channels)
        except Exception:
            waiting = sum(1 for m in self._memoirs.values() if m.is_waiting())
        return {
            "total_memoirs": total,
            "open_memoirs": total - waiting,
            "anticipating_memoirs": waiting,
            "vault_dir": str(self._vault_dir),
        }


def get_memoir_cabinet() -> MemoirCabinet:
    """获取回忆录管理中心单例"""
    return MemoirCabinet.instance()

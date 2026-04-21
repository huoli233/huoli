import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, List, Any
from src.common.logger import get_logger
from src.chat.proactive.models import (
    ActivityKind, ConversationState, MindLogRecord, PendingConfig,
)

logger = get_logger("proactive_session")


class MiaoSession:
    MAX_LOG_CAPACITY = 50
    DEFAULT_MAX_UNANSWERED = 3

    def __init__(self, user_id: str, stream_id: str):
        self.user_id = user_id
        self.stream_id = stream_id
        self._state: ConversationState = ConversationState.READY
        self.mind_log: List[MindLogRecord] = []
        self.pending_config: PendingConfig = PendingConfig()
        self.created_at: float = time.time()
        self.last_active_at: float = time.time()
        self.interaction_count: int = 0
        self.last_spontaneous_at: Optional[float] = None
        self.sequential_timeout_count: int = 0
        self.last_user_msg_at: Optional[float] = None
        self.unanswered_count: int = 0
        self.max_unanswered: int = self.DEFAULT_MAX_UNANSWERED
        self.last_bot_message_at: Optional[float] = None
        self.daily_proactive_count: int = 0
        self.daily_date: str = ""
        self.proactive_paused: bool = False

    @property
    def state(self) -> ConversationState:
        return self._state

    @state.setter
    def state(self, value: ConversationState) -> None:
        old_state = self._state
        self._state = value
        if old_state != value:
            logger.debug(f"Session {self.user_id} 状态变更: {old_state} -> {value}")

    def append_record(self, record: MindLogRecord) -> None:
        self.mind_log.append(record)
        self.last_active_at = time.time()
        if len(self.mind_log) > self.MAX_LOG_CAPACITY:
            self.mind_log = self.mind_log[-self.MAX_LOG_CAPACITY:]

    def log_incoming_message(
        self, content: str, sender_name: str, sender_id: str,
        msg_time: Optional[float] = None,
    ) -> MindLogRecord:
        recorded_at = msg_time or time.time()
        record = MindLogRecord(
            kind=ActivityKind.INCOMING_MSG,
            recorded_at=recorded_at,
            text_content=content,
            sender_name=sender_name,
            sender_id=sender_id,
        )
        self.sequential_timeout_count = 0
        self.last_user_msg_at = recorded_at
        if self.state == ConversationState.AWAITING and self.pending_config.is_active():
            elapsed = self.pending_config.elapsed_seconds()
            timeout = self.pending_config.timeout_seconds
            if elapsed <= timeout:
                record.extra_data["reply_status"] = "timely"
                record.extra_data["elapsed_seconds"] = elapsed
                record.extra_data["timeout_seconds"] = timeout
            else:
                record.extra_data["reply_status"] = "delayed"
                record.extra_data["elapsed_seconds"] = elapsed
                record.extra_data["timeout_seconds"] = timeout
        self.append_record(record)
        return record

    def log_user_message(self, sender_id: str, sender_name: str, text_content: str):
        self.log_incoming_message(text_content, sender_name, sender_id)

    def log_bot_decision(
        self, inner_thought: str, behaviors: List[Dict],
        anticipated_reply: str = "", timeout_seconds: int = 0,
        decision_time: Optional[float] = None,
    ) -> MindLogRecord:
        record = MindLogRecord(
            kind=ActivityKind.BOT_DECISION,
            recorded_at=decision_time or time.time(),
            inner_thought=inner_thought,
            behavior_list=behaviors,
            anticipated_reply=anticipated_reply,
            timeout_seconds=timeout_seconds,
        )
        self.append_record(record)
        self.interaction_count += 1
        return record

    def log_await_shift(
        self, reflect_thought: str, current_mood: str = "",
        shift_time: Optional[float] = None,
    ) -> MindLogRecord:
        record = MindLogRecord(
            kind=ActivityKind.AWAIT_SHIFT,
            recorded_at=shift_time or time.time(),
            reflect_thought=reflect_thought,
            current_mood=current_mood,
            elapsed_time=self.pending_config.elapsed_seconds(),
        )
        self.append_record(record)
        return record

    def begin_awaiting(self, anticipated_reply: str, timeout_seconds: int) -> None:
        if timeout_seconds <= 0:
            self.state = ConversationState.READY
            self.pending_config.clear()
            return
        self.state = ConversationState.AWAITING
        self.pending_config = PendingConfig(
            anticipated_reply=anticipated_reply,
            timeout_seconds=timeout_seconds,
            initiated_at=time.time(),
            last_reflect_at=0.0,
            reflect_count=0,
        )
        logger.debug(
            f"Session {self.user_id} 开始等待: "
            f"timeout={timeout_seconds}s, anticipated={anticipated_reply[:30]}..."
        )

    def finish_awaiting(self) -> None:
        self.state = ConversationState.READY
        self.pending_config.clear()
        self.last_active_at = time.time()

    def fetch_recent_records(self, limit: int = 20) -> List[MindLogRecord]:
        return self.mind_log[-limit:] if self.mind_log else []

    def fetch_last_bot_reply(self) -> Optional[str]:
        for record in reversed(self.mind_log):
            if record.kind == ActivityKind.BOT_DECISION:
                for behavior in record.behavior_list:
                    if behavior.get("type") in ("miao_reply", "respond"):
                        return behavior.get("content", "")
        return None

    def on_bot_sent(self) -> None:
        from datetime import datetime
        self.unanswered_count += 1
        self.last_bot_message_at = time.time()
        today = datetime.now().strftime("%Y-%m-%d")
        if self.daily_date != today:
            self.daily_date = today
            self.daily_proactive_count = 0
        self.daily_proactive_count += 1
        logger.debug(
            f"Session {self.user_id} Bot已发送消息，"
            f"未回复计数: {self.unanswered_count}"
        )

    def on_user_reply(self) -> None:
        if self.unanswered_count > 0:
            logger.debug(
                f"Session {self.user_id} 用户已回复，"
                f"重置未回复计数 {self.unanswered_count} -> 0"
            )
        self.unanswered_count = 0
        self.proactive_paused = False
        self.sequential_timeout_count = 0

    def should_give_up_proactive(self) -> bool:
        return self.unanswered_count >= self.max_unanswered

    def pause_proactive(self, reason: str = "") -> None:
        self.proactive_paused = True
        if reason:
            logger.info(f"Session {self.user_id} 暂停主动消息: {reason}")

    def resume_proactive(self) -> None:
        self.proactive_paused = False
        self.unanswered_count = 0
        logger.debug(f"Session {self.user_id} 恢复主动消息")

    def get_proactive_status(self) -> Dict:
        return {
            "unanswered_count": self.unanswered_count,
            "max_unanswered": self.max_unanswered,
            "should_give_up": self.should_give_up_proactive(),
            "is_paused": self.proactive_paused,
            "daily_count": self.daily_proactive_count,
            "last_bot_message_at": self.last_bot_message_at,
        }

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "stream_id": self.stream_id,
            "state": str(self.state),
            "mind_log": [r.to_dict() for r in self.mind_log],
            "pending_config": self.pending_config.to_dict(),
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
            "interaction_count": self.interaction_count,
            "last_spontaneous_at": self.last_spontaneous_at,
            "sequential_timeout_count": self.sequential_timeout_count,
            "last_user_msg_at": self.last_user_msg_at,
            "unanswered_count": self.unanswered_count,
            "max_unanswered": self.max_unanswered,
            "last_bot_message_at": self.last_bot_message_at,
            "daily_proactive_count": self.daily_proactive_count,
            "daily_date": self.daily_date,
            "proactive_paused": self.proactive_paused,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MiaoSession":
        session = cls(
            user_id=data.get("user_id", ""),
            stream_id=data.get("stream_id", ""),
        )
        state_str = data.get("state", "ready")
        try:
            session._state = ConversationState(state_str)
        except ValueError:
            session._state = ConversationState.READY
        log_data = data.get("mind_log", [])
        session.mind_log = [MindLogRecord.from_dict(r) for r in log_data]
        pending_data = data.get("pending_config", {})
        session.pending_config = PendingConfig.from_dict(pending_data)
        session.created_at = data.get("created_at", time.time())
        session.last_active_at = data.get("last_active_at", time.time())
        session.interaction_count = data.get("interaction_count", 0)
        session.last_spontaneous_at = data.get("last_spontaneous_at")
        session.sequential_timeout_count = data.get("sequential_timeout_count", 0)
        session.last_user_msg_at = data.get("last_user_msg_at")
        session.unanswered_count = data.get("unanswered_count", 0)
        session.max_unanswered = data.get("max_unanswered", cls.DEFAULT_MAX_UNANSWERED)
        session.last_bot_message_at = data.get("last_bot_message_at")
        session.daily_proactive_count = data.get("daily_proactive_count", 0)
        session.daily_date = data.get("daily_date", "")
        session.proactive_paused = data.get("proactive_paused", False)
        return session


class SessionHub:
    _instance: Optional["SessionHub"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, storage_dir: str = "data/miao_sessions", max_age_days: int = 30):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self.storage_dir = Path(storage_dir)
        self.max_age_days = max_age_days
        self._sessions: Dict[str, MiaoSession] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SessionHub 初始化完成: {self.storage_dir}")

    @asynccontextmanager
    async def _acquire_lock(self, user_id: str):
        async with self._locks_lock:
            if user_id not in self._locks:
                self._locks[user_id] = asyncio.Lock()
            lock = self._locks[user_id]
        async with lock:
            yield lock

    def _resolve_file_path(self, user_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
        return self.storage_dir / f"{safe_id}.json"

    async def get_session(self, user_id: str, stream_id: str) -> MiaoSession:
        async with self._acquire_lock(user_id):
            if user_id in self._sessions:
                session = self._sessions[user_id]
                session.stream_id = stream_id
                return session
            session = await self._load_from_storage(user_id)
            if session:
                session.stream_id = stream_id
                self._sessions[user_id] = session
                return session
            session = MiaoSession(user_id=user_id, stream_id=stream_id)
            self._sessions[user_id] = session
            logger.info(f"创建新会话: {user_id}")
            return session

    def _get_session_model(self):
        from src.common.database.database_model import SessionRecord
        return SessionRecord

    async def _load_from_storage(self, user_id: str) -> Optional[MiaoSession]:
        try:
            SR = self._get_session_model()
            row = SR.get_or_none(SR.session_id == user_id)
            if not row:
                return self._migrate_json_session(user_id)
            session = MiaoSession(user_id=user_id, stream_id=row.stream_id)
            try:
                session._state = ConversationState(row.status)
            except ValueError:
                session._state = ConversationState.READY
            session.created_at = row.started_at
            session.last_active_at = row.last_active_at
            session.interaction_count = row.message_count
            if row.metadata_text:
                meta = json.loads(row.metadata_text)
                session.unanswered_count = meta.get("unanswered_count", 0)
                session.daily_proactive_count = meta.get("daily_proactive_count", 0)
                session.daily_date = meta.get("daily_date", "")
                session.proactive_paused = meta.get("proactive_paused", False)
                session.sequential_timeout_count = meta.get("sequential_timeout_count", 0)
            logger.debug(f"从数据库加载会话: {user_id}")
            return session
        except Exception as e:
            logger.error(f"加载会话失败 {user_id}: {e}")
            return None

    def _migrate_json_session(self, user_id: str) -> Optional[MiaoSession]:
        file_path = self._resolve_file_path(user_id)
        if not file_path.exists():
            return None
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            session = MiaoSession.from_dict(data)
            self._sessions[user_id] = session
            asyncio.get_event_loop().create_task(self.save_session(user_id))
            file_path.rename(file_path.with_suffix('.json.bak'))
            logger.info(f"会话 {user_id} 已从JSON迁移到数据库")
            return session
        except Exception as e:
            logger.debug(f"JSON会话迁移跳过 {user_id}: {e}")
            return None

    async def save_session(self, user_id: str) -> bool:
        async with self._acquire_lock(user_id):
            if user_id not in self._sessions:
                return False
            session = self._sessions[user_id]
            try:
                SR = self._get_session_model()
                meta = json.dumps({
                    "unanswered_count": session.unanswered_count,
                    "daily_proactive_count": session.daily_proactive_count,
                    "daily_date": session.daily_date,
                    "proactive_paused": session.proactive_paused,
                    "sequential_timeout_count": session.sequential_timeout_count,
                }, ensure_ascii=False)
                SR.insert(
                    session_id=user_id,
                    stream_id=session.stream_id,
                    session_type="proactive",
                    status=str(session.state),
                    trigger_reason="",
                    message_count=session.interaction_count,
                    started_at=session.created_at,
                    last_active_at=session.last_active_at,
                    ended_at=0.0,
                    metadata_text=meta,
                ).on_conflict(
                    conflict_target=[SR.session_id],
                    update={
                        SR.stream_id: session.stream_id,
                        SR.status: str(session.state),
                        SR.message_count: session.interaction_count,
                        SR.last_active_at: session.last_active_at,
                        SR.metadata_text: meta,
                    }
                ).execute()
                return True
            except Exception as e:
                logger.error(f"保存会话失败 {user_id}: {e}")
                return False

    async def save_all_sessions(self) -> int:
        saved_count = 0
        for user_id in list(self._sessions.keys()):
            if await self.save_session(user_id):
                saved_count += 1
        return saved_count

    async def get_awaiting_sessions(self) -> List[MiaoSession]:
        return [s for s in self._sessions.values() if s.state == ConversationState.AWAITING]

    async def get_all_sessions(self) -> List[MiaoSession]:
        return list(self._sessions.values())

    def get_session_sync(self, user_id: str) -> Optional[MiaoSession]:
        return self._sessions.get(user_id)

    async def remove_session(self, user_id: str) -> bool:
        async with self._acquire_lock(user_id):
            if user_id in self._sessions:
                del self._sessions[user_id]
                try:
                    SR = self._get_session_model()
                    SR.delete().where(SR.session_id == user_id).execute()
                except Exception as e:
                    logger.error(f"删除会话记录失败 {user_id}: {e}")
                return True
            return False

    async def cleanup_expired_sessions(self) -> int:
        cleaned_count = 0
        current_time = time.time()
        max_age_seconds = self.max_age_days * 24 * 3600
        for user_id in list(self._sessions.keys()):
            session = self._sessions[user_id]
            if current_time - session.last_active_at > max_age_seconds:
                if await self.remove_session(user_id):
                    cleaned_count += 1
                    logger.info(f"清理过期会话: {user_id}")
        return cleaned_count

    def get_statistics(self) -> Dict:
        total = len(self._sessions)
        awaiting = sum(1 for s in self._sessions.values() if s.state == ConversationState.AWAITING)
        ready = total - awaiting
        return {
            "total_sessions": total,
            "ready_sessions": ready,
            "awaiting_sessions": awaiting,
            "storage_dir": str(self.storage_dir),
        }


_session_hub: Optional[SessionHub] = None


def get_session_hub() -> SessionHub:
    global _session_hub
    if _session_hub is None:
        _session_hub = SessionHub()
    return _session_hub

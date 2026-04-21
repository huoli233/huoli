import asyncio
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Dict, List
from src.common.logger import get_logger
from src.chat.proactive.models import GeneratorResult
from src.chat.proactive.session_manager import MiaoSession, get_session_hub
from src.chat.proactive.planner import create_plan
from src.chat.proactive.unified_mode import generate_unified

logger = get_logger("affinity_chatter")


@dataclass
class ChatContext:
    stream_id: str
    user_id: str
    user_name: str
    message_content: str
    message_id: str = ""
    is_private: bool = True
    is_mentioned: bool = False
    extra_data: Dict = field(default_factory=dict)


@dataclass
class ChatResult:
    success: bool
    stream_id: str
    behaviors_count: int = 0
    reply_content: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0


class BondingChatter:
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.session_hub = get_session_hub()
        self._lock = asyncio.Lock()
        self._work_mode = "unified"
        self._metrics = {
            "messages_handled": 0,
            "plans_generated": 0,
            "behaviors_executed": 0,
            "successful_runs": 0,
            "failed_runs": 0,
        }
        self._last_active_at = time.time()

    async def handle_message(self, context: ChatContext) -> ChatResult:
        async with self._lock:
            start_time = time.time()
            try:
                session = await self.session_hub.get_session(context.user_id, context.stream_id)
                session.log_user_message(
                    sender_id=context.user_id,
                    sender_name=context.user_name,
                    text_content=context.message_content,
                )
                decision = await self._generate_decision(session, context)
                reply_content = self._extract_reply(decision)
                session.log_bot_decision(
                    inner_thought=decision.inner_thought,
                    behaviors=[b.to_dict() for b in decision.behaviors],
                    anticipated_reply=decision.anticipated_reply,
                    timeout_seconds=decision.timeout_seconds,
                )
                if decision.timeout_seconds > 0:
                    session.begin_awaiting(
                        anticipated_reply=decision.anticipated_reply,
                        timeout_seconds=decision.timeout_seconds,
                    )
                await self.session_hub.save_session(context.user_id)
                self._metrics["messages_handled"] += 1
                self._metrics["plans_generated"] += 1
                self._metrics["behaviors_executed"] += len(decision.behaviors)
                self._metrics["successful_runs"] += 1
                self._last_active_at = time.time()
                processing_time = time.time() - start_time
                logger.debug(
                    f"[BondingChatter] 消息处理完成: stream={self.stream_id}, "
                    f"behaviors={len(decision.behaviors)}, time={processing_time:.2f}s"
                )
                return ChatResult(
                    success=True, stream_id=self.stream_id,
                    behaviors_count=len(decision.behaviors),
                    reply_content=reply_content, processing_time=processing_time,
                )
            except asyncio.CancelledError:
                logger.info(f"[BondingChatter] 处理被取消: stream={self.stream_id}")
                self._metrics["failed_runs"] += 1
                raise
            except Exception as e:
                logger.error(
                    f"[BondingChatter] 处理失败: stream={self.stream_id}, error={e}\n"
                    f"{traceback.format_exc()}"
                )
                self._metrics["failed_runs"] += 1
                self._last_active_at = time.time()
                return ChatResult(
                    success=False, stream_id=self.stream_id,
                    error_message=str(e), processing_time=time.time() - start_time,
                )

    async def _generate_decision(self, session: MiaoSession, context: ChatContext) -> GeneratorResult:
        extra_params = {
            "user_message": context.message_content,
            "is_private": context.is_private,
            "is_mentioned": context.is_mentioned,
        }
        if self._work_mode == "unified":
            return await generate_unified(
                session=session, user_name=context.user_name,
                situation="new_message", extra_params=extra_params,
            )
        return await create_plan(
            session=session, user_name=context.user_name,
            situation="new_message", extra_params=extra_params,
        )

    def _extract_reply(self, decision: GeneratorResult) -> Optional[str]:
        for behavior in decision.behaviors:
            if behavior.action_type in ("miao_reply", "respond"):
                return behavior.action_params.get("content")
        return None

    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self._metrics,
            "stream_id": self.stream_id,
            "work_mode": self._work_mode,
            "last_active_at": self._last_active_at,
        }

    def is_active(self, max_idle_minutes: int = 60) -> bool:
        idle_seconds = time.time() - self._last_active_at
        return idle_seconds < max_idle_minutes * 60

    def reset_metrics(self) -> None:
        self._metrics = {
            "messages_handled": 0, "plans_generated": 0,
            "behaviors_executed": 0, "successful_runs": 0, "failed_runs": 0,
        }

    def __str__(self) -> str:
        return f"BondingChatter(stream={self.stream_id}, msgs={self._metrics['messages_handled']})"

    def __repr__(self) -> str:
        return (
            f"BondingChatter(stream_id={self.stream_id}, "
            f"messages={self._metrics['messages_handled']}, "
            f"last_active={datetime.fromtimestamp(self._last_active_at)})"
        )


class ChatterRegistry:
    def __init__(self):
        self._chatters: Dict[str, BondingChatter] = {}
        self._lock = asyncio.Lock()

    async def get_chatter(self, stream_id: str) -> BondingChatter:
        async with self._lock:
            if stream_id not in self._chatters:
                self._chatters[stream_id] = BondingChatter(stream_id)
                logger.debug(f"[ChatterRegistry] 创建聊天器: stream={stream_id}")
            return self._chatters[stream_id]

    async def remove_chatter(self, stream_id: str) -> bool:
        async with self._lock:
            if stream_id in self._chatters:
                del self._chatters[stream_id]
                logger.debug(f"[ChatterRegistry] 移除聊天器: stream={stream_id}")
                return True
            return False

    def list_chatters(self) -> List[str]:
        return list(self._chatters.keys())

    def get_active_chatters(self, max_idle_minutes: int = 60) -> List[BondingChatter]:
        return [c for c in self._chatters.values() if c.is_active(max_idle_minutes)]

    async def cleanup_inactive(self, max_idle_minutes: int = 120) -> int:
        async with self._lock:
            inactive = [
                sid for sid, c in self._chatters.items()
                if not c.is_active(max_idle_minutes)
            ]
            for sid in inactive:
                del self._chatters[sid]
            if inactive:
                logger.info(f"[ChatterRegistry] 清理 {len(inactive)} 个不活跃聊天器")
            return len(inactive)


_chatter_registry: Optional[ChatterRegistry] = None


def get_chatter_registry() -> ChatterRegistry:
    global _chatter_registry
    if _chatter_registry is None:
        _chatter_registry = ChatterRegistry()
    return _chatter_registry


async def get_bonding_chatter(stream_id: str) -> BondingChatter:
    registry = get_chatter_registry()
    return await registry.get_chatter(stream_id)

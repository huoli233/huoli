import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
from src.common.logger import get_logger
from src.chat.proactive.models import (
    ActivityKind, BehaviorModel, ConversationState, GeneratorResult, MindLogRecord,
)
from src.chat.proactive.session_manager import MiaoSession, get_session_hub
from src.chat.proactive.think_scheduler import ThinkCategory, ThinkTask

logger = get_logger("think_executor")


@dataclass
class ExecutionContext:
    session: MiaoSession
    user_name: str
    situation: str
    extra_params: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    decision: Optional[GeneratorResult] = None
    error: Optional[str] = None
    started_at: float = 0.0
    completed_at: float = 0.0


class ThinkExecutor:
    def __init__(self):
        self.session_hub = get_session_hub()
        self._prompt_builders: Dict[str, callable] = {
            "expire": self._build_expire_prompt,
            "reflect": self._build_reflect_prompt,
            "spontaneous": self._build_spontaneous_prompt,
            "new_message": self._build_new_message_prompt,
        }
        self._stats = {
            "total_executions": 0, "successful_executions": 0,
            "failed_executions": 0, "llm_calls": 0, "behaviors_executed": 0,
        }

    async def execute_think_task(self, task: ThinkTask) -> Dict[str, Any]:
        self._stats["total_executions"] += 1
        try:
            session = await self.session_hub.get_session(task.target_user_id, task.target_stream_id)
            user_name = await self._resolve_user_name(task.target_user_id)
            situation = self._map_category_to_situation(task.category)
            context = ExecutionContext(
                session=session, user_name=user_name, situation=situation,
                extra_params=task.params, started_at=time.time(),
            )
            await self._execute_think_flow(context)
            context.completed_at = time.time()
            if context.success:
                self._stats["successful_executions"] += 1
            else:
                self._stats["failed_executions"] += 1
            return {
                "success": context.success,
                "decision": context.decision.to_dict() if context.decision else None,
                "error": context.error,
                "duration": context.completed_at - context.started_at,
            }
        except Exception as e:
            self._stats["failed_executions"] += 1
            logger.error(f"[ThinkExecutor] 执行任务失败: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_think_flow(self, context: ExecutionContext) -> None:
        try:
            prompt = self._build_prompt(context)
            if not prompt:
                context.error = "无法构建提示词"
                return
            response_text = await self._call_llm(prompt)
            self._stats["llm_calls"] += 1
            if not response_text:
                context.error = "LLM响应为空"
                return
            decision = self._parse_response(response_text)
            context.decision = decision
            await self._execute_behaviors(context)
            await self._update_session(context)
            context.success = True
        except Exception as e:
            context.error = str(e)
            logger.error(f"[ThinkExecutor] 思考流程异常: {e}")

    def _build_prompt(self, context: ExecutionContext) -> Optional[str]:
        builder = self._prompt_builders.get(context.situation)
        if not builder:
            logger.warning(f"[ThinkExecutor] 未找到情境 {context.situation} 的提示词构建器")
            return None
        base_prompt = builder(context)
        if not base_prompt:
            return None
        return base_prompt

    def _build_expire_prompt(self, context: ExecutionContext) -> str:
        session = context.session
        user_name = context.user_name
        params = context.extra_params
        last_bot_message = session.fetch_last_bot_reply() or "（未知）"
        anticipated = session.pending_config.anticipated_reply or "对方能回复点什么"
        from src.config.prompt_loader import get_prompt, PromptCategory
        return get_prompt(
            PromptCategory.MODULE,
            "proactive",
            "think_expire.template",
            user_name=user_name,
            last_bot_message=last_bot_message[:100],
            anticipated=anticipated,
            sequential_timeout_count=params.get('sequential_timeout_count', 0),
            chase_count=params.get('chase_count', 0),
            time_since_user_reply_str=params.get('time_since_user_reply_str', '未知')
        )

    def _build_reflect_prompt(self, context: ExecutionContext) -> str:
        session = context.session
        user_name = context.user_name
        params = context.extra_params
        last_bot_message = session.fetch_last_bot_reply() or "（未知）"
        if len(last_bot_message) > 100:
            last_bot_message = last_bot_message[:100] + "..."
        elapsed_minutes = session.pending_config.elapsed_minutes()
        max_wait_minutes = session.pending_config.timeout_seconds / 60
        anticipated = session.pending_config.anticipated_reply or "对方能回复点什么"
        progress = params.get("progress_percent", 50)
        from src.config.prompt_loader import get_prompt, PromptCategory
        return get_prompt(
            PromptCategory.MODULE,
            "proactive",
            "think_reflect.template",
            user_name=user_name,
            last_bot_message=last_bot_message,
            anticipated=anticipated,
            elapsed_minutes=f"{elapsed_minutes:.0f}",
            max_wait_minutes=f"{max_wait_minutes:.0f}",
            progress=progress
        )

    def _build_spontaneous_prompt(self, context: ExecutionContext) -> str:
        user_name = context.user_name
        params = context.extra_params
        silence_duration = params.get("silence_duration", "一段时间")
        trigger_reason = params.get("trigger_reason", "")
        from src.config.prompt_loader import get_prompt, PromptCategory
        return get_prompt(
            PromptCategory.MODULE,
            "proactive",
            "think_spontaneous.template",
            user_name=user_name,
            silence_duration=silence_duration,
            trigger_reason=trigger_reason
        )

    def _build_new_message_prompt(self, context: ExecutionContext) -> str:
        session = context.session
        user_name = context.user_name
        params = context.extra_params
        user_message = params.get("user_message", "")
        recent_records = session.fetch_recent_records(limit=10)
        history_text = self._format_history(recent_records)
        from src.config.prompt_loader import get_prompt, PromptCategory
        return get_prompt(
            PromptCategory.MODULE,
            "proactive",
            "think_new_message.template",
            user_name=user_name,
            history_text=history_text,
            user_message=user_message
        )

    def _format_history(self, records: List[MindLogRecord]) -> str:
        if not records:
            return "（暂无对话历史）"
        lines = ["最近的对话："]
        for record in records[-5:]:
            if record.kind == ActivityKind.INCOMING_MSG:
                lines.append(f"  {record.sender_name}: {record.text_content[:50]}...")
            elif record.kind == ActivityKind.BOT_DECISION:
                for behavior in record.behavior_list:
                    if behavior.get("type") == "miao_reply":
                        content = behavior.get("content", "")[:50]
                        lines.append(f"  你: {content}...")
        return "\n".join(lines)

    async def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            request = LLMRequest(model_config.focus_chat, request_type="think_executor")
            response_text, _ = await request.generate_response_async(prompt, max_tokens=500)
            return response_text
        except Exception as e:
            logger.error(f"[ThinkExecutor] LLM调用失败: {e}")
            return None

    def _parse_response(self, raw_response: str) -> GeneratorResult:
        text = raw_response.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return GeneratorResult.from_dict(data)
            except json.JSONDecodeError:
                pass
        return GeneratorResult(
            inner_thought=text[:200],
            behaviors=[BehaviorModel(action_type="idle")],
        )

    async def _execute_behaviors(self, context: ExecutionContext) -> None:
        if not context.decision:
            return
        for behavior in context.decision.behaviors:
            if behavior.action_type == "idle":
                continue
            self._stats["behaviors_executed"] += 1
            if behavior.action_type == "miao_reply":
                content = behavior.action_params.get("content", "")
                if content:
                    await self._send_message(context.session.stream_id, content)
            elif behavior.action_type == "nudge_user":
                await self._send_nudge(context.session.stream_id, context.session.user_id)
            elif behavior.action_type == "send_sticker":
                sticker = behavior.action_params.get("sticker", "")
                if sticker:
                    await self._send_sticker(context.session.stream_id, sticker)

    async def _send_message(self, stream_id: str, content: str) -> bool:
        try:
            logger.info(f"[ThinkExecutor] 准备发送主动消息到 {stream_id[:8]}: {content[:50]}...")
            from src.chat.message_receive.chat_stream import get_chat_manager
            chat_mgr = get_chat_manager()
            chat_stream = chat_mgr.get_stream(stream_id)
            if not chat_stream:
                logger.warning(f"[ThinkExecutor] 未找到聊天流: {stream_id}")
                return False
            from src.chat.message_receive.message import MessageSending
            from maim_message import Seg
            msg = MessageSending(chat_stream)
            msg.processed_plain_text = content
            msg.message_segment = Seg(type="seglist", data=[Seg(type="text", data=content)])
            from src.chat.message_receive.uni_message_sender import MessageSender
            sender = MessageSender()
            success = await sender.send_message(msg, typing=True, set_reply=False, storage_message=True, show_log=True)
            if success:
                logger.info(f"[ThinkExecutor] 主动消息已发送到 {stream_id[:8]}")
                try:
                    from src.chat.proactive.message_integration import get_proactive_integration
                    proactive = get_proactive_integration()
                    if proactive:
                        await proactive.on_reply_sent(stream_id, content, None, is_proactive=True)
                except Exception as e:
                    logger.debug(f"[ThinkExecutor] 通知proactive系统失败: {e}")
            else:
                logger.warning(f"[ThinkExecutor] 主动消息发送失败到 {stream_id[:8]}")
            return success
        except Exception as e:
            logger.warning(f"[ThinkExecutor] 发送消息失败: {e}")
            return False

    async def _send_nudge(self, stream_id: str, user_id: str) -> bool:
        try:
            logger.info(f"[ThinkExecutor] 发送戳一戳到 {user_id}")
            return True
        except Exception as e:
            logger.error(f"[ThinkExecutor] 发送戳一戳失败: {e}")
            return False

    async def _send_sticker(self, stream_id: str, sticker: str) -> bool:
        try:
            logger.info(f"[ThinkExecutor] 发送表情到 {stream_id}: {sticker}")
            return True
        except Exception as e:
            logger.error(f"[ThinkExecutor] 发送表情失败: {e}")
            return False

    async def _resolve_user_name(self, user_id: str) -> str:
        try:
            from src.person_info.person_info import Person
            person = Person(platform="qq", user_id=user_id)
            if person.person_name:
                return person.person_name
        except Exception:
            pass
        return user_id[:8] if len(user_id) > 8 else user_id

    async def _update_session(self, context: ExecutionContext) -> None:
        if not context.decision:
            return
        session = context.session
        decision = context.decision
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
        else:
            if session.state == ConversationState.AWAITING:
                session.finish_awaiting()
        await self.session_hub.save_session(session.user_id)

    def _map_category_to_situation(self, category: ThinkCategory) -> str:
        mapping = {
            ThinkCategory.EXPIRE_HANDLE: "expire",
            ThinkCategory.REFLECT_UPDATE: "reflect",
            ThinkCategory.SPONTANEOUS_INIT: "spontaneous",
            ThinkCategory.CUSTOM_TASK: "new_message",
        }
        return mapping.get(category, "new_message")

    def get_statistics(self) -> Dict[str, Any]:
        return self._stats.copy()


_think_executor: Optional[ThinkExecutor] = None


def get_think_executor() -> ThinkExecutor:
    global _think_executor
    if _think_executor is None:
        _think_executor = ThinkExecutor()
    return _think_executor


async def setup_think_handlers() -> None:
    from src.chat.proactive.think_scheduler import get_think_scheduler
    scheduler = get_think_scheduler()
    executor = get_think_executor()
    scheduler.register_handler(ThinkCategory.EXPIRE_HANDLE, executor.execute_think_task)
    scheduler.register_handler(ThinkCategory.REFLECT_UPDATE, executor.execute_think_task)
    scheduler.register_handler(ThinkCategory.SPONTANEOUS_INIT, executor.execute_think_task)
    scheduler.register_handler(ThinkCategory.CUSTOM_TASK, executor.execute_think_task)
    logger.info("[ThinkExecutor] 思考处理器已注册")

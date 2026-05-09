import asyncio
import time as _tm
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine, Dict, Optional

from src.common.logger import get_logger

logger = get_logger("reply_coord")


# ---------------------------------------------------------------------------
#  优先级枚举
# ---------------------------------------------------------------------------


class ReplyUrgency(IntEnum):
    """回复紧急程度，数值越大优先级越高"""

    PASSIVE = 1  # 被动闲聊
    MONITOR = 2  # 监控触发
    ACTIVE = 3  # 主动/@ 触发


# ---------------------------------------------------------------------------
#  协调器
# ---------------------------------------------------------------------------


@dataclass
class _ChannelSlot:
    """单频道的回复锁与元数据"""

    mutex: asyncio.Lock = field(default_factory=asyncio.Lock)
    current_urgency: ReplyUrgency = ReplyUrgency.PASSIVE
    executing: bool = False
    last_finish_ts: float = 0.0
    followup_tasks: Dict[str, asyncio.Task] = field(default_factory=dict)
    recent_bot_utterances: list = field(default_factory=list)


class ReplyCoordinator:
    """频道级回复协调器

    确保同一频道同一时刻只有一个回复任务在执行。
    高优先级任务可以跳过等待。
    """

    def __init__(self):
        self._slots: Dict[str, _ChannelSlot] = {}

    def _slot(self, channel_id: str) -> _ChannelSlot:
        if channel_id not in self._slots:
            self._slots[channel_id] = _ChannelSlot()
        return self._slots[channel_id]

    async def acquire_permission(
        self,
        channel_id: str,
        urgency: ReplyUrgency = ReplyUrgency.PASSIVE,
        wait_timeout: float = 15.0,
    ) -> bool:
        """尝试获取回复许可

        低优先级任务在锁被占用时等候 wait_timeout 秒；
        高优先级任务直接获得许可（不会阻塞）。
        """
        slot = self._slot(channel_id)
        if slot.executing and urgency <= slot.current_urgency:
            try:
                acquired = await asyncio.wait_for(slot.mutex.acquire(), timeout=wait_timeout)
                if acquired:
                    slot.mutex.release()
            except asyncio.TimeoutError:
                logger.debug(f"[回复协调] {channel_id} 等待超时，放弃本次回复")
                return False
        slot.current_urgency = urgency
        return True

    async def execute_reply(
        self,
        channel_id: str,
        reply_coro: Coroutine,
        urgency: ReplyUrgency = ReplyUrgency.PASSIVE,
    ) -> Optional[Any]:
        """在锁保护下执行回复协程"""
        slot = self._slot(channel_id)
        async with slot.mutex:
            slot.executing = True
            slot.current_urgency = urgency
            try:
                result = await reply_coro
                return result
            except Exception as exc:
                logger.error(f"[回复协调] {channel_id} 回复执行失败: {exc}")
                return None
            finally:
                slot.executing = False
                slot.last_finish_ts = _tm.time()
                # 回复完成后通知网关记录一次机器人回复
                try:
                    from src.chat.heart_flow.reply_decision_gateway import (
                        acquire_reply_gateway,
                    )

                    acquire_reply_gateway().record_reply(channel_id)
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")

    def seconds_since_last(self, channel_id: str) -> float:
        """距离上次回复完成的秒数"""
        slot = self._slot(channel_id)
        if slot.last_finish_ts == 0.0:
            return float("inf")
        return _tm.time() - slot.last_finish_ts

    def register_followup_task(self, channel_id: str, task_name: str, task: asyncio.Task) -> None:
        slot = self._slot(channel_id)
        slot.followup_tasks[task_name] = task

        def _cleanup(done_task: asyncio.Task) -> None:
            slot.followup_tasks.pop(task_name, None)

        task.add_done_callback(_cleanup)

    async def cancel_followup_tasks(self, channel_id: str, reason: str = "") -> int:
        slot = self._slot(channel_id)
        pending = [task for task in slot.followup_tasks.values() if not task.done()]
        if not pending:
            return 0
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        slot.followup_tasks.clear()
        logger.info(f"[回复协调] {channel_id} 已取消{len(pending)}个补充任务 reason={reason}")
        return len(pending)

    def resolve_style_route(
        self,
        *,
        delivery_form: str,
        target_message: Any,
        reference_user_name: str,
        mention_user_name: str,
        fallback_selector: Callable[[Any, Dict[str, Any], Optional[Dict[str, Any]]], Dict[str, Any]],
        relation_view: Dict[str, Any],
        harassment_signal: Optional[Dict[str, Any]],
        is_bot_message: Optional[Callable[[Any], bool]] = None,
        governor_quote_policy: str = "",
    ) -> Dict[str, Any]:
        """统一解析回复形式，避免主循环文件继续承载这层分支逻辑。"""
        need_quote = "quote" in (delivery_form or "")
        need_mention = "mention" in (delivery_form or "")
        reasons = []
        fallback_route = fallback_selector(target_message, relation_view, harassment_signal) or {}
        target_is_bot = bool(is_bot_message(target_message)) if (is_bot_message and target_message is not None) else False
        explicit_anchor = bool(
            need_quote
            or reference_user_name.strip()
            or getattr(target_message, "is_quote_reply", False)
            or getattr(target_message, "is_reply_to_bot", False)
            or getattr(target_message, "reply_to", "")
            or getattr(target_message, "reply_to_message_id", None)
        )
        target_age_sec = float("inf")
        if target_message is not None:
            try:
                target_age_sec = max(
                    0.0,
                    _tm.time()
                    - float(
                        getattr(target_message, "timestamp", 0.0)
                        or getattr(target_message, "time", 0.0)
                        or _tm.time()
                    ),
                )
            except Exception:
                target_age_sec = float("inf")
        quote_fresh_sec = 300.0
        quote_soft_sec = 1200.0

        quote_policy = str(governor_quote_policy or "").strip()
        if not quote_policy:
            if target_message is None or target_is_bot:
                quote_policy = "none"
            elif explicit_anchor and target_age_sec <= quote_fresh_sec:
                quote_policy = "quote_reply"
            elif explicit_anchor and target_age_sec <= quote_soft_sec:
                quote_policy = "soft_reference"
            elif explicit_anchor:
                quote_policy = "none"
            elif bool(fallback_route.get("quote_message", False)) and target_age_sec <= quote_fresh_sec:
                quote_policy = "quote_reply"
            elif target_message is not None and target_age_sec <= quote_soft_sec:
                quote_policy = "soft_reference"
            else:
                quote_policy = "none"

        if need_quote:
            if quote_policy == "quote_reply":
                reasons.append(f"规划器指定引用{reference_user_name or '某人'}的消息")
            else:
                reasons.append("没有稳定锚点，规划器引用降级为软续接")
        if need_mention:
            reasons.append(f"规划器指定@{mention_user_name}")
        if quote_policy == "quote_reply" and target_age_sec > quote_fresh_sec:
            if target_age_sec <= quote_soft_sec:
                quote_policy = "soft_reference"
                reasons.append("引用窗口过新鲜期，降级为软引用")
            else:
                quote_policy = "none"
                reasons.append("引用超窗，降级为无引用")
        if quote_policy == "soft_reference" and target_age_sec > quote_soft_sec:
            quote_policy = "none"
            reasons.append("前情超窗，降级为无引用")
        if quote_policy == "none" and target_is_bot:
            reasons.append("不把自己的消息当引用锚点")
        if quote_policy != "none" and (target_message is None or target_is_bot):
            quote_policy = "none"
            reasons.append("目标锚点无效，降级为无引用")

        reply_style = str(fallback_route.get("reply_style", "direct") or "direct")
        if quote_policy == "quote_reply":
            reply_style = "quote"
        elif quote_policy == "soft_reference":
            reply_style = "soft_reference"
        elif reply_style == "quote":
            reply_style = "direct"

        route_reason = "；".join(part for part in [str(fallback_route.get("reason", "") or "").strip(), *reasons] if part)
        return {
            "reply_style": reply_style,
            "quote_message": quote_policy == "quote_reply",
            "quote_policy": quote_policy,
            "target_user_id": str(getattr(target_message, "user_id", "") or "").strip() if target_message is not None else "",
            "reason": route_reason,
        }

    async def generate_reply(
        self,
        *,
        channel_id: str,
        chat_stream: Any,
        action_modifier: Any,
        action_manager: Any,
        target_message: Any,
        reply_reason: str,
        extra_info: str,
        request_type: str,
        think_level: int = 1,
        fast_path: bool = False,
        enable_splitter: bool = True,
        enable_chinese_typo: bool = True,
    ):
        from src.plugin_system.apis import generator_api
        from src.config.config import global_config
        from src.chat.utils.utils import record_replyer_action_temp

        record_replyer_action_temp(
            chat_id=channel_id,
            reason=reply_reason,
            think_level=think_level,
        )

        available_actions: Dict[str, Any] = {}
        try:
            await action_modifier.modify_actions()
            available_actions = action_manager.get_using_actions()
        except Exception as exc:
            logger.warning(f"[回复协调] {channel_id} 获取可用动作失败: {exc}")

        return await generator_api.generate_reply(
            chat_stream=chat_stream,
            reply_message=target_message,
            available_actions=available_actions,
            chosen_actions=[],
            reply_reason=reply_reason,
            unknown_words=None,
            enable_tool=global_config.tool.enable_tool,
            request_type=request_type,
            from_plugin=False,
            reply_time_point=_tm.time(),
            think_level=think_level,
            extra_info=extra_info,
            enable_splitter=enable_splitter,
            enable_chinese_typo=enable_chinese_typo,
            fast_path=fast_path,
        )

    def select_reply_target(
        self,
        *,
        messages: list,
        reference_user_name: str,
        delivery_form: str,
        is_bot_message: Callable[[Any], bool],
        preferred_selector: Callable[[Any, list], Any],
    ):
        """统一选择回复目标消息，减少主循环重复逻辑。"""
        target_message = None
        need_quote = "quote" in (delivery_form or "")
        if need_quote and reference_user_name:
            ref_lower = reference_user_name.lower().strip()
            for msg in reversed(messages[-15:]):
                if is_bot_message(msg):
                    continue
                msg_nick = (
                    str(getattr(msg, "user_nickname", "") or getattr(msg, "user_cardname", "") or "").lower().strip()
                )
                if ref_lower and ref_lower in msg_nick:
                    target_message = msg
                    break
        if target_message is None:
            for msg in reversed(messages[-10:]):
                if not is_bot_message(msg):
                    target_message = msg
                    break
        target_message = preferred_selector(target_message, list(messages[-10:]))
        if target_message is not None and is_bot_message(target_message):
            target_message = None
        return target_message

    def remember_bot_utterance(self, channel_id: str, content: str, message_id: str = "") -> None:
        text = str(content or "").strip()
        if not text:
            return
        slot = self._slot(channel_id)
        now = _tm.time()
        if slot.recent_bot_utterances:
            latest_text = str(slot.recent_bot_utterances[-1].get("text", "") or "").strip()
            latest_ts = float(slot.recent_bot_utterances[-1].get("ts", 0.0) or 0.0)
            if latest_text == text and latest_ts > 0 and (now - latest_ts) <= 5.0:
                if message_id and not str(slot.recent_bot_utterances[-1].get("message_id", "") or "").strip():
                    slot.recent_bot_utterances[-1]["message_id"] = str(message_id).strip()
                return
        slot.recent_bot_utterances.append({"text": text, "ts": now, "message_id": str(message_id or "").strip()})
        self.prune_recent_bot_utterances(channel_id, now)

    def bind_utterance_message_id(self, channel_id: str, content: str, message_id: str) -> bool:
        text = str(content or "").strip()
        bound_id = str(message_id or "").strip()
        if not text or not bound_id:
            return False
        slot = self._slot(channel_id)
        for item in reversed(slot.recent_bot_utterances):
            if str(item.get("text", "") or "").strip() == text:
                item["message_id"] = bound_id
                return True
        return False

    def prune_recent_bot_utterances(self, channel_id: str, now: Optional[float] = None) -> None:
        slot = self._slot(channel_id)
        current = now if now is not None else _tm.time()
        ttl_sec = 180.0
        recent_items = []
        for item in slot.recent_bot_utterances[-5:]:
            text = str(item.get("text", "") or "").strip()
            ts = float(item.get("ts", 0.0) or 0.0)
            if not text:
                continue
            if ts > 0 and (current - ts) <= ttl_sec:
                recent_items.append(
                    {
                        "text": text,
                        "ts": ts,
                        "message_id": str(item.get("message_id", "") or ""),
                    }
                )
        slot.recent_bot_utterances = recent_items[-3:]

    def recent_bot_utterance_entries(self, channel_id: str, limit: int = 3) -> list[Dict[str, Any]]:
        try:
            from src.core.self_reply_recognizer import get_self_reply_recognizer

            entries = get_self_reply_recognizer().build_recent_message_snapshots(channel_id, limit=limit)
            if entries:
                return entries[-limit:]
        except Exception as exc:
            logger.debug(f"[回复协调] {channel_id} 读取统一 bot 快照失败: {exc}")

        self.prune_recent_bot_utterances(channel_id)
        slot = self._slot(channel_id)
        return [
            {
                "text": str(item.get("text", "") or "").strip(),
                "ts": float(item.get("ts", 0.0) or 0.0),
                "message_id": str(item.get("message_id", "") or ""),
            }
            for item in slot.recent_bot_utterances[-limit:]
            if str(item.get("text", "") or "").strip()
        ]

    def recent_bot_texts(self, channel_id: str) -> list[str]:
        return [
            str(item.get("text", "") or "").strip()
            for item in self.recent_bot_utterance_entries(channel_id)
            if str(item.get("text", "") or "").strip()
        ]

    def compose_reply_reason(
        self,
        *,
        base_reason: str,
        content_plan: str = "",
        gateway_gate: str = "",
        voice_reason: str = "",
    ) -> str:
        reason = str(base_reason or "").strip() or "回复"
        if content_plan:
            reason = f"{reason} | 内容规划: {content_plan[:100]}"
        if voice_reason:
            reason = f"{reason} | {voice_reason[:140]}"
        if gateway_gate == "hesitate":
            reason = f"{reason} | 当前处于低可回复度，只允许简短澄清或低风险短回"
        return reason


# ---------------------------------------------------------------------------
#  单例
# ---------------------------------------------------------------------------

_coordinator_instance: Optional[ReplyCoordinator] = None


def acquire_reply_coordinator() -> ReplyCoordinator:
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = ReplyCoordinator()
    return _coordinator_instance

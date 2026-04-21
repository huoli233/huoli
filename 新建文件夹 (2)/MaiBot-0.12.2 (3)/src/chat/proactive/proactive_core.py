import time
import asyncio
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("proactive_core")


@dataclass
class ProactiveEvent:
    event_type: str
    channel_id: str
    user_id: str = ""
    content: str = ""
    interest_level: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProactiveDecision:
    should_act: bool = False
    action_type: str = ""
    content: str = ""
    delay_seconds: float = 0.0
    energy_cost: float = 0.0
    energy_remaining: float = 0.0
    reason: str = ""
    priority: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProactiveCore:
    _instance: Optional["ProactiveCore"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._session_hub = None
        self._think_scheduler = None
        self._energy_manager = None
        self._silence_detector = None
        self._unified_scheduler = None
        self._topic_extractor = None
        self._initialized = True
        self._is_running = False
        self._load_subsystems()
        logger.info("主动行为核心初始化完成")

    def _load_subsystems(self):
        try:
            from src.chat.proactive.session_manager import get_session_hub
            self._session_hub = get_session_hub()
        except Exception as e:
            logger.debug(f"会话管理器加载跳过: {e}")
        try:
            from src.chat.proactive.think_scheduler import get_think_scheduler
            self._think_scheduler = get_think_scheduler()
        except Exception as e:
            logger.debug(f"思考调度器加载跳过: {e}")
        try:
            from src.chat.proactive.engagement_energy import get_engagement_energy_manager
            self._energy_manager = get_engagement_energy_manager()
        except Exception as e:
            logger.debug(f"能量管理器加载跳过: {e}")
        try:
            from src.chat.proactive.silence_detector import get_silence_detector
            self._silence_detector = get_silence_detector()
        except Exception as e:
            logger.debug(f"沉默检测器加载跳过: {e}")
        try:
            from src.chat.proactive.schedule_strategy import get_unified_scheduler
            self._unified_scheduler = get_unified_scheduler()
        except Exception as e:
            logger.debug(f"统一调度器加载跳过: {e}")
        try:
            from src.chat.proactive.topic_extractor import get_topic_extractor
            self._topic_extractor = get_topic_extractor()
        except Exception as e:
            logger.debug(f"话题提取器加载跳过: {e}")

    async def start(self) -> bool:
        if self._is_running:
            return True
        try:
            if self._think_scheduler and not self._think_scheduler._is_active:
                await self._think_scheduler.start()
            if self._unified_scheduler and not getattr(self._unified_scheduler, '_is_active', False):
                await self._unified_scheduler.start()
            self._is_running = True
            logger.info("[主动核心] 系统已启动")
            return True
        except Exception as e:
            logger.error(f"[主动核心] 启动失败: {e}")
            return False

    async def stop(self) -> bool:
        if not self._is_running:
            return True
        try:
            if self._think_scheduler:
                await self._think_scheduler.stop()
            if self._unified_scheduler:
                await self._unified_scheduler.stop()
            self._is_running = False
            logger.info("[主动核心] 系统已停止")
            return True
        except Exception as e:
            logger.error(f"[主动核心] 停止失败: {e}")
            return False

    async def process_event(self, event: ProactiveEvent) -> ProactiveDecision:
        decision = ProactiveDecision()
        event_type = event.event_type
        if event_type == "check_proactive":
            decision = await self._check_should_proactive(event)
        elif event_type == "silence_detected":
            decision = await self._handle_silence(event)
        elif event_type == "user_message":
            decision = await self._handle_user_message(event)
        elif event_type == "schedule_task":
            decision = await self._schedule_think_task(event)
        elif event_type == "generate_proactive":
            decision = await self._generate_proactive_content(event)
        else:
            logger.debug(f"未知主动事件类型: {event_type}")
        return decision

    async def _check_should_proactive(self, event: ProactiveEvent) -> ProactiveDecision:
        decision = ProactiveDecision()
        channel_id = event.channel_id
        interest_level = event.interest_level
        if self._energy_manager:
            can_speak, reason, energy = self._energy_manager.can_speak(channel_id, interest_level)
            decision.energy_remaining = energy
            if not can_speak:
                decision.should_act = False
                decision.reason = reason
                return decision
        from src.chat.proactive.schedule_strategy import is_quiet_time
        if is_quiet_time("22-7"):
            decision.should_act = False
            decision.reason = "当前为安静时段"
            return decision
        if self._session_hub:
            session = self._session_hub.get_session_sync(event.user_id)
            if session:
                if session.should_give_up_proactive():
                    decision.should_act = False
                    decision.reason = "用户未回复次数过多"
                    return decision
                if session.proactive_paused:
                    decision.should_act = False
                    decision.reason = "主动消息已暂停"
                    return decision
        decision.should_act = True
        decision.reason = "可以执行主动行为"
        return decision

    async def _handle_silence(self, event: ProactiveEvent) -> ProactiveDecision:
        decision = ProactiveDecision()
        check_result = await self._check_should_proactive(event)
        if not check_result.should_act:
            return check_result
        decision.should_act = True
        decision.action_type = "proactive_message"
        silence_duration = event.metadata.get("silence_duration", 0)
        decision.reason = f"沉默{silence_duration}秒后触发主动发言"
        return decision

    async def _handle_user_message(self, event: ProactiveEvent) -> ProactiveDecision:
        decision = ProactiveDecision()
        channel_id = event.channel_id
        user_id = event.user_id
        if self._energy_manager:
            self._energy_manager.on_user_message(channel_id)
        if self._session_hub and user_id:
            session = self._session_hub.get_session_sync(user_id)
            if session:
                session.on_user_reply()
        decision.should_act = False
        decision.reason = "用户消息已处理"
        return decision

    async def _schedule_think_task(self, event: ProactiveEvent) -> ProactiveDecision:
        decision = ProactiveDecision()
        if not self._think_scheduler:
            decision.should_act = False
            decision.reason = "思考调度器未加载"
            return decision
        from src.chat.proactive.think_scheduler import ThinkCategory, ThinkPriority
        task_type = event.metadata.get("task_type", "custom")
        priority_value = event.metadata.get("priority", 3)
        delay = event.metadata.get("delay_seconds", 0)
        category_map = {
            "expire_handle": ThinkCategory.EXPIRE_HANDLE,
            "reflect_update": ThinkCategory.REFLECT_UPDATE,
            "spontaneous_init": ThinkCategory.SPONTANEOUS_INIT,
            "custom": ThinkCategory.CUSTOM_TASK,
        }
        priority_map = {
            1: ThinkPriority.URGENT, 2: ThinkPriority.HIGH, 3: ThinkPriority.NORMAL,
            4: ThinkPriority.LOW, 5: ThinkPriority.IDLE,
        }
        category = category_map.get(task_type, ThinkCategory.CUSTOM_TASK)
        priority = priority_map.get(priority_value, ThinkPriority.NORMAL)
        task_id = await self._think_scheduler.schedule_task(
            category=category, user_id=event.user_id, stream_id=event.channel_id,
            priority=priority, params=event.metadata.get("params", {}), delay_seconds=delay,
        )
        if task_id:
            decision.should_act = True
            decision.action_type = "task_scheduled"
            decision.metadata["task_id"] = task_id
            decision.reason = f"任务已调度: {task_type}"
        else:
            decision.should_act = False
            decision.reason = "任务调度失败或被去重"
        return decision

    async def _generate_proactive_content(self, event: ProactiveEvent) -> ProactiveDecision:
        decision = ProactiveDecision()
        channel_id = event.channel_id
        check_result = await self._check_should_proactive(event)
        if not check_result.should_act:
            return check_result
        try:
            from src.chat.proactive.planner import create_proactive_plan
            plan = await create_proactive_plan(
                channel_id=channel_id,
                situation_desc=event.content or "主动发起对话",
                context_info=event.metadata.get("context", {}),
            )
            if plan and plan.get("success"):
                content = plan.get("suggested_content", "")
                if content:
                    decision.should_act = True
                    decision.action_type = "proactive_message"
                    decision.content = content
                    if self._energy_manager:
                        result = self._energy_manager.on_speak(channel_id, event.interest_level)
                        decision.energy_cost = result.get("cost", 0)
                        decision.energy_remaining = result.get("energy", 0)
                    if self._session_hub and event.user_id:
                        session = self._session_hub.get_session_sync(event.user_id)
                        if session:
                            session.on_bot_sent()
                    decision.reason = "主动内容已生成"
                else:
                    decision.should_act = False
                    decision.reason = "内容为空"
            else:
                decision.should_act = False
                decision.reason = plan.get("reason", "规划失败") if plan else "规划失败"
        except Exception as e:
            decision.should_act = False
            decision.reason = f"生成失败: {e}"
            logger.error(f"[主动核心] 内容生成失败: {e}")
        return decision

    def can_proactive(self, channel_id: str, interest_level: float = 0.5) -> tuple:
        if self._energy_manager:
            return self._energy_manager.can_speak(channel_id, interest_level)
        return True, "能量系统未加载", 0.0

    def on_bot_message_sent(self, channel_id: str, user_id: str, interest_level: float = 0.5) -> Dict[str, Any]:
        result = {"energy": 0.0, "cost": 0.0}
        if self._energy_manager:
            result = self._energy_manager.on_speak(channel_id, interest_level)
        if self._session_hub and user_id:
            session = self._session_hub.get_session_sync(user_id)
            if session:
                session.on_bot_sent()
        return result

    def on_user_message_received(self, channel_id: str, user_id: str) -> None:
        if self._energy_manager:
            self._energy_manager.on_user_message(channel_id)
        if self._session_hub and user_id:
            session = self._session_hub.get_session_sync(user_id)
            if session:
                session.on_user_reply()

    async def get_session(self, user_id: str, stream_id: str):
        if self._session_hub:
            return await self._session_hub.get_session(user_id, stream_id)
        return None

    def get_energy_status(self, channel_id: str) -> Dict[str, Any]:
        if self._energy_manager:
            return self._energy_manager.get_status(channel_id)
        return {"energy": 0.0, "is_resting": False}

    def get_proactive_status(self, user_id: str) -> Dict[str, Any]:
        if self._session_hub:
            session = self._session_hub.get_session_sync(user_id)
            if session:
                return session.get_proactive_status()
        return {"unanswered_count": 0, "should_give_up": False}

    def init_hook_trigger(self, context: Any = None) -> bool:
        if self._hook_trigger is not None:
            return True
        try:
            from src.chat.proactive.hook_trigger import init_hook_trigger
            self._hook_trigger = init_hook_trigger(context)
            return True
        except Exception as e:
            logger.debug(f"HookTrigger初始化失败: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        stats = {"is_running": self._is_running, "subsystems_loaded": []}
        if self._session_hub:
            stats["subsystems_loaded"].append("session_hub")
            stats["session_stats"] = self._session_hub.get_statistics()
        if self._think_scheduler:
            stats["subsystems_loaded"].append("think_scheduler")
            stats["scheduler_stats"] = self._think_scheduler.get_statistics()
        if self._energy_manager:
            stats["subsystems_loaded"].append("energy_manager")
        if self._silence_detector:
            stats["subsystems_loaded"].append("silence_detector")
        if self._unified_scheduler:
            stats["subsystems_loaded"].append("unified_scheduler")
        if self._topic_extractor:
            stats["subsystems_loaded"].append("topic_extractor")
        return stats

    def calculate_engagement_unified(self, content: str) -> Any:
        try:
            from src.chat.chatter.interest_calculator import get_engagement_scorer
            scorer = get_engagement_scorer()
            if scorer:
                return scorer.calculate_interest(content)
        except Exception as e:
            logger.debug(f"计算参与度失败: {e}")
        class DummyResult:
            def __init__(self):
                self.score = 0.5
                self.final_decision = "neutral"
        return DummyResult()

    def force_rest(self, channel_id: str, minutes: float = 30.0) -> None:
        if self._energy_manager:
            self._energy_manager.force_rest(channel_id, minutes)

    def force_recover(self, channel_id: str, amount: float = 50.0) -> None:
        if self._energy_manager:
            self._energy_manager.force_recover(channel_id, amount)

    async def cancel_user_tasks(self, user_id: str) -> int:
        if self._think_scheduler:
            return await self._think_scheduler.cancel_tasks_for_user(user_id)
        return 0


_proactive_core: Optional[ProactiveCore] = None


def get_proactive_core() -> ProactiveCore:
    global _proactive_core
    if _proactive_core is None:
        _proactive_core = ProactiveCore()
    return _proactive_core


async def check_proactive_unified(
    channel_id: str, user_id: str = "", interest_level: float = 0.5,
) -> ProactiveDecision:
    core = get_proactive_core()
    event = ProactiveEvent(
        event_type="check_proactive", channel_id=channel_id,
        user_id=user_id, interest_level=interest_level,
    )
    return await core.process_event(event)


async def generate_proactive_unified(
    channel_id: str, user_id: str = "", interest_level: float = 0.5,
    context: Optional[Dict[str, Any]] = None,
) -> ProactiveDecision:
    core = get_proactive_core()
    event = ProactiveEvent(
        event_type="generate_proactive", channel_id=channel_id,
        user_id=user_id, interest_level=interest_level,
        metadata={"context": context or {}},
    )
    return await core.process_event(event)


def on_message_received_unified(channel_id: str, user_id: str) -> None:
    core = get_proactive_core()
    core.on_user_message_received(channel_id, user_id)


def on_message_sent_unified(channel_id: str, user_id: str, interest_level: float = 0.5) -> Dict[str, Any]:
    core = get_proactive_core()
    return core.on_bot_message_sent(channel_id, user_id, interest_level)

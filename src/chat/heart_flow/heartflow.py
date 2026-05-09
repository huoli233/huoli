import asyncio
import traceback
import time as _tm
from typing import Any, Optional, Dict, List

from src.chat.message_receive.chat_stream import (
    get_chat_manager,
    is_internal_webui_stream,
    is_webui_virtual_stream,
)
from src.common.logger import get_logger
from src.common.task_utils import safe_create_task
from src.chat.heart_flow.heartFC_chat_enhanced import EnhancedHeartFChatting
from src.chat.brain_chat.brain_chat import BrainChatting
from src.chat.message_receive.chat_stream import ChatStream

logger = get_logger("心流引擎")


def _module_switch_on(name: str, default: bool = True) -> bool:
    try:
        from src.config.core_config_engine import get_core_config

        values = get_core_config().resolve_module_view("module_switches").values
        return bool(values.get(name, default))
    except Exception:
        return default


def _phase_idle_timeout_seconds() -> float:
    try:
        from src.config.core_config_engine import get_core_config

        return float(get_core_config().resolve_module_view("phase_timing").values.get("phase_idle_timeout_seconds", 300.0))
    except Exception:
        return 300.0


class Heartflow:
    """主心流协调器

    职责:
            - 管理频道和增强主链/BrainChatting 的映射
      - 提供统一的生命周期(启动/停止/清理)
      - 集成状态管理器、独立监控器
      - 定期清理长时间未活跃的聊天实例
    """

    def __init__(self):
        self.log_prefix = "[心流]"
        self.heartflow_chat_list: Dict[
            Any,
            EnhancedHeartFChatting | BrainChatting,
        ] = {}
        self._creation_locks: Dict[Any, asyncio.Lock] = {}
        self._creation_tasks: Dict[Any, asyncio.Task] = {}
        self._active_since: Dict[Any, float] = {}
        self._started = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._warmup_task: Optional[asyncio.Task] = None
        self._proactive_hooks_bound = False
        self._dimension_dispatcher = None

    # ---- 生命周期 ----

    async def startup(self) -> None:
        """启动心流系统及后台监控"""
        if self._started:
            return
        self._started = True
        try:
            from src.config.core_config_engine import boot_core_config

            boot_core_config()
        except Exception as exc:
            logger.warning(f"[心流] 核心配置启动异常: {exc}")
        try:
            from src.chat.heart_flow.system_calibration import (
                acquire_system_calibrator,
            )

            calibrator = acquire_system_calibrator()
            summary = calibrator.calibrate_and_propagate()
            logger.info(f"[心流] 启动校准完成: {summary.level}, 亲密度变化{summary.intimacy_delta:+.1f}")
        except Exception as exc:
            logger.warning(f"[心流] 启动校准异常: {exc}")
        try:
            logger.info("[心流] 独立自主代理已移出启动路径，统一由增强主链决策")
        except Exception as exc:
            logger.warning(f"[心流] 自主代理启动异常: {exc}")
        # 注册多维独立状态系统的所有维度到调度器
        try:
            from src.chat.heart_flow.dimension_bootstrap import (
                bootstrap_all_dimensions,
            )

            self._dimension_dispatcher = await bootstrap_all_dimensions()
            await self._dimension_dispatcher.initialize()
            await self._dimension_dispatcher.start()
            logger.info("[心流] 多维状态调度器已启动")
        except Exception as exc:
            self._dimension_dispatcher = None
            logger.warning(f"[心流] 多维状态调度器启动异常: {exc}")
        self._bind_proactive_hooks()
        self._warm_core_services()
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup_loop())
        logger.info("[心流] 主协调器已启动")

    async def shutdown(self) -> None:
        """优雅关闭所有聊天实例和后台服务"""
        logger.info(f"[心流] 正在关闭, 活跃聊天数: {len(self.heartflow_chat_list)}")
        if self._warmup_task and not self._warmup_task.done():
            self._warmup_task.cancel()
            try:
                await self._warmup_task
            except asyncio.CancelledError:
                pass
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        try:
            logger.info("[心流] 独立自主代理已移出关闭路径")
        except Exception as _e:
            logger.warning(f"[心流] 关闭日志记录异常: {_e}")
        # 关闭多维状态调度器
        if self._dimension_dispatcher is not None:
            try:
                await self._dimension_dispatcher.stop()
                logger.info("[心流] 多维状态调度器已关闭")
            except Exception as _e:
                logger.debug(f"调度器关闭异常: {_e}")
        try:
            from src.chat.heart_flow.user_state_detector import (
                acquire_engagement_probe,
            )

            saved = acquire_engagement_probe().persist_all_active()
            if saved:
                logger.info(f"[心流] 用户参与度快照已保存 {saved} 人")
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.energy_manager import EnergyChainDimension

            _d6 = EnergyChainDimension.get_instance()
            e_saved = _d6.serialize() if _d6 else {}
            if e_saved:
                logger.info(f"[心流] 能量快照已保存 {len(e_saved)} 个频道")
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.state_machine import get_phase_coordinator

            get_phase_coordinator().persist_phases()
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.emotion_stream import (
                get_channel_mood_tracker,
            )

            get_channel_mood_tracker().save_to_persistent_store()
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.persona_engine import CharacterFoundry

            CharacterFoundry.teardown()
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.inner_voice import clear_all_engines

            clear_all_engines()
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.persistent_state_db import (
                acquire_persistent_store,
            )

            flushed = acquire_persistent_store().flush()
            logger.info(f"[心流] 持久化刷盘 {flushed} 条")
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.proactive.session_tracker import get_memoir_cabinet

            saved = await get_memoir_cabinet().persist_all()
            if saved:
                logger.info(f"[心流] 主动回忆录已保存 {saved} 份")
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        for chat_id in list(self.heartflow_chat_list.keys()):
            await self._teardown_chat(chat_id)
        self.heartflow_chat_list.clear()
        self._active_since.clear()
        self._started = False
        logger.info("[心流] 主协调器已关闭")

    # ---- 聊天实例获取/创建 ----

    async def get_or_create_heartflow_chat(self, chat_id: Any) -> Optional[EnhancedHeartFChatting | BrainChatting]:
        """获取或创建聊天实例（线程安全）"""
        existing = self.heartflow_chat_list.get(chat_id)
        if existing is not None:
            self._active_since[chat_id] = _tm.time()
            return existing
        task = self.prewarm_chat(chat_id, reason="on_demand")
        if task is None:
            existing = self.heartflow_chat_list.get(chat_id)
            if existing is not None:
                self._active_since[chat_id] = _tm.time()
            return existing
        try:
            created = await task
            if created is not None:
                self._active_since[chat_id] = _tm.time()
            return created
        except Exception as exc:
            logger.error(f"创建心流聊天 {chat_id} 失败: {exc}", exc_info=True)
            traceback.print_exc()
            return None

    def prewarm_chat(self, chat_id: Any, reason: str = "") -> Optional[asyncio.Task]:
        """后台预热聊天实例，重复调用会复用进行中的启动任务。"""
        stream = get_chat_manager().get_stream(str(chat_id))
        if stream and self._should_skip_prewarm(stream, reason=reason):
            logger.debug(f"[心流] 跳过内部/虚拟会话预热 {chat_id}: {reason or 'unspecified'}")
            return None
        existing = self.heartflow_chat_list.get(chat_id)
        if existing is not None:
            self._active_since[chat_id] = _tm.time()
            return None
        active_task = self._creation_tasks.get(chat_id)
        if active_task and not active_task.done():
            return active_task

        async def _create_and_cache() -> Optional[EnhancedHeartFChatting | BrainChatting]:
            return await self._create_chat_instance(chat_id)

        task = asyncio.create_task(
            _create_and_cache(),
            name=f"heartflow_create_{str(chat_id)[:8]}",
        )
        self._creation_tasks[chat_id] = task

        def _finalize(finished: asyncio.Task, target_id: Any = chat_id) -> None:
            try:
                self._creation_tasks.pop(target_id, None)
                if finished.cancelled():
                    return
                exc = finished.exception()
                if exc is not None:
                    logger.warning(f"[心流] 预热聊天失败 {target_id}: {exc}")
            except Exception as cb_err:
                logger.error(f"[心流] _finalize 回调异常: {cb_err}")

        task.add_done_callback(_finalize)
        if reason:
            logger.debug(f"[心流] 触发聊天预热 {chat_id}: {reason}")
        return task

    async def _create_chat_instance(
        self,
        chat_id: Any,
    ) -> Optional[EnhancedHeartFChatting | BrainChatting]:
        """实际执行聊天实例创建。"""
        _MAX_CHAT_INSTANCES = 2000
        if len(self.heartflow_chat_list) >= _MAX_CHAT_INSTANCES:
            cleaned = await self.cleanup_inactive(max_idle_sec=600.0)
            if len(self.heartflow_chat_list) >= _MAX_CHAT_INSTANCES:
                logger.warning(
                    f"[心流] 聊天实例数已达上限 {_MAX_CHAT_INSTANCES}（清理了{cleaned}个），拒绝创建 {chat_id}"
                )
                return None
        lock = self._creation_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._creation_locks[chat_id] = lock
        async with lock:
            existing = self.heartflow_chat_list.get(chat_id)
            if existing is not None:
                self._active_since[chat_id] = _tm.time()
                return existing
            chat_stream: ChatStream | None = get_chat_manager().get_stream(chat_id)
            if not chat_stream:
                raise ValueError(f"未找到 chat_id={chat_id} 的聊天流")
            if is_internal_webui_stream(
                platform=chat_stream.platform,
                group_id=getattr(chat_stream.group_info, "group_id", None),
            ):
                raise ValueError(f"拒绝为 WebUI 内部管理连接创建心流实例: {chat_id}")
            if chat_stream.group_info:
                new_chat = EnhancedHeartFChatting(chat_id=chat_id)
            else:
                new_chat = BrainChatting(chat_id=chat_id)
            await new_chat.start()
            self.heartflow_chat_list[chat_id] = new_chat
            self._active_since[chat_id] = _tm.time()
            # 向自主运行器注册新频道
            try:
                from src.core.autonomous_core.autonomous_runner import (
                    get_background_orchestrator,
                )

                get_background_orchestrator().register_channel(str(chat_id))
            except Exception as _e:
                logger.debug(f"{self.log_prefix} 异常: {_e}")
            logger.info(f"[心流] 创建聊天实例 {chat_id}, 类型={'群聊' if chat_stream.group_info else '私聊'}")
            return new_chat

    # ---- 清理 ----

    async def cleanup_inactive(self, max_idle_sec: float = 1800.0) -> int:
        """清理长时间未活跃的聊天实例"""
        now = _tm.time()
        stale_ids: List[Any] = []
        for chat_id, last_active in self._active_since.items():
            if now - last_active > max_idle_sec:
                stale_ids.append(chat_id)
        for chat_id in stale_ids:
            await self._teardown_chat(chat_id)
            self.heartflow_chat_list.pop(chat_id, None)
            self._active_since.pop(chat_id, None)
        if stale_ids:
            logger.info(f"[心流] 清理了 {len(stale_ids)} 个不活跃聊天")
        return len(stale_ids)

    async def _teardown_chat(self, chat_id: Any) -> None:
        """安全关闭单个聊天实例"""
        chat = self.heartflow_chat_list.get(chat_id)
        if chat is None:
            return
        self._flush_channel_state(chat_id)
        try:
            from src.core.autonomous_core.autonomous_runner import (
                get_background_orchestrator,
            )

            get_background_orchestrator().unregister_channel(str(chat_id))
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            if hasattr(chat, "stop"):
                await chat.stop()
        except Exception as exc:
            logger.warning(f"[心流] 关闭聊天 {chat_id} 异常: {exc}")
        # 清理关联的锁和任务引用，防止内存泄漏
        self._creation_locks.pop(chat_id, None)
        self._creation_tasks.pop(chat_id, None)
        self._active_since.pop(chat_id, None)

    @staticmethod
    def _flush_channel_state(channel_id: Any) -> None:
        """通知下游模块刷新该频道的缓存或持久化数据"""
        try:
            from src.chat.heart_flow.heartflow_decision import (
                acquire_decision_maker,
            )

            acquire_decision_maker().signal_activity_change(str(channel_id))
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.user_state_detector import (
                acquire_engagement_probe,
            )

            acquire_engagement_probe().persist_report(str(channel_id))
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def note_user_activity(self, stream_id: str, user_id: str, text: str) -> None:
        """向主动链路广播用户消息。"""
        self.touch(stream_id)
        # 多维状态系统事件广播
        if self._dimension_dispatcher is not None:
            try:
                from src.chat.heart_flow.dimension_protocol import EventContext

                evt = EventContext(
                    event_type="message_received",
                    user_id=user_id,
                    channel_id=stream_id,
                    message_text=text,
                    message_length=len(text) if text else 0,
                )
                safe_create_task(
                    self._dimension_dispatcher.broadcast_event(evt),
                    name="dim_broadcast_msg",
                )
            except Exception as _e:
                logger.debug(f"维度广播异常: {_e}")
        try:
            from src.chat.heart_flow.reply_decision_gateway import (
                acquire_reply_gateway,
            )

            acquire_reply_gateway().notify_user_spoke(stream_id)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.proactive.silence_watcher import get_quiet_monitor

            get_quiet_monitor().note_activity(stream_id, speaker_id=user_id, is_bot=False)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # 反馈闭环：用户回复，主动决策账本结算奖励
        if stream_id:
            try:
                from src.chat.proactive.proactive_decider import (
                    get_proactive_decider,
                )

                settled = get_proactive_decider().ledger.acknowledge_user_reply(stream_id, is_negative=False)
                if settled:
                    logger.info(f"[反馈] stream_id={stream_id} 用户回复已确认，主动决策账本结算奖励")
            except Exception as e:
                logger.warning(f"[heartflow] [反馈] 账本结算失败: {e}")

    def note_bot_activity(self, stream_id: str, was_proactive: bool = False) -> None:
        """向主动链路广播机器人发言。"""
        self.touch(stream_id)
        # 多维状态系统回复完成事件广播
        if self._dimension_dispatcher is not None:
            try:
                from src.chat.heart_flow.dimension_protocol import EventContext

                evt = EventContext(
                    event_type="reply_completed",
                    channel_id=stream_id,
                    raw_extras={"was_proactive": was_proactive},
                )
                safe_create_task(
                    self._dimension_dispatcher.broadcast_event(evt),
                    name="dim_broadcast_reply",
                )
            except Exception as _e:
                logger.debug(f"维度广播异常: {_e}")
        try:
            from src.chat.proactive.silence_watcher import get_quiet_monitor

            get_quiet_monitor().note_activity(stream_id, is_bot=True)
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _bind_proactive_hooks(self) -> None:
        """补齐静默巡查回调，仅作为唤醒信号接入。"""
        if self._proactive_hooks_bound:
            return
        try:
            from src.chat.proactive.silence_watcher import get_quiet_monitor

            get_quiet_monitor().attach_quiet_handler(self._handle_quiet_period)
        except Exception as exc:
            logger.warning(f"[心流] 绑定静默巡查回调失败: {exc}")
        self._proactive_hooks_bound = True

    def _schedule_startup_warmup(self) -> None:
        if self._warmup_task is not None and not self._warmup_task.done():
            return
        self._warmup_task = asyncio.create_task(
            self._run_startup_warmup(),
            name="heartflow_startup_warmup",
        )

    async def _run_startup_warmup(self) -> None:
        """后台预热核心模块与全部已知聊天流，避免再等首条消息冷启动。"""
        await asyncio.sleep(0.2)
        self._warm_core_services()
        try:
            streams = list(get_chat_manager().streams.values())
            streams.sort(
                key=lambda item: getattr(item, "last_active_time", 0.0),
                reverse=True,
            )
            warmed = 0
            batch_size = 8
            for offset in range(0, len(streams), batch_size):
                pending = []
                for stream in streams[offset : offset + batch_size]:
                    if self._should_skip_prewarm(stream, reason="startup_all"):
                        continue
                    task = self.prewarm_chat(stream.stream_id, reason="startup_all")
                    if task is not None:
                        pending.append(task)
                if pending:
                    warmed += len(pending)
                    await asyncio.gather(*pending, return_exceptions=True)
            if warmed:
                logger.info(f"[心流] 启动全量预热完成，聊天流数: {warmed}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"[心流] 启动预热异常: {exc}")

    @staticmethod
    def _should_skip_prewarm(stream: ChatStream, *, reason: str = "") -> bool:
        """启动预热只覆盖真实平台聊天流，避免 WebUI 管理会话污染频道。"""
        group_id = getattr(getattr(stream, "group_info", None), "group_id", None)
        if is_internal_webui_stream(platform=getattr(stream, "platform", ""), group_id=group_id):
            return True
        if reason == "startup_all" and is_webui_virtual_stream(group_id):
            return True
        return False

    def _warm_core_services(self) -> None:
        """提前实例化首轮回复常用模块。"""
        try:
            from src.core.module_coordinator import discover_modules

            discover_modules()
        except Exception as exc:
            logger.debug(f"[心流] 模块协调预热异常: {exc}")

        warmers = []
        try:
            from src.core.identity_anchor import get_identity_anchor

            warmers.append(get_identity_anchor)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.core.dynamic_context_window import get_dynamic_context

            warmers.append(get_dynamic_context)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.core.self_reply_recognizer import (
                get_self_reply_recognizer,
            )

            warmers.append(get_self_reply_recognizer)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.core.content_state_tracker import (
                get_content_state_tracker,
            )

            warmers.append(get_content_state_tracker)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.state_machine import get_phase_coordinator

            warmers.append(get_phase_coordinator)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.proactive.silence_watcher import get_quiet_monitor

            warmers.append(get_quiet_monitor)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.llm_autonomous_planner import (
                get_llm_autonomous_planner,
            )

            warmers.append(get_llm_autonomous_planner)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.proactive.session_tracker import get_memoir_cabinet

            warmers.append(get_memoir_cabinet)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.modules.trauma.trauma_system import get_trauma_system

            warmers.append(get_trauma_system)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.modules.modcore.psychological_core import (
                get_psychological_core,
            )

            warmers.append(get_psychological_core)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.emotion_driven_core import (
                get_emotion_driven_core,
            )

            warmers.append(get_emotion_driven_core)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        for warmer in warmers:
            try:
                warmer()
            except Exception as exc:
                logger.debug(f"[心流] 预热模块失败: {exc}")

    async def _handle_quiet_period(self, channel_id: str) -> None:
        """静默巡查触发后仅唤醒频道主链，不再直接派发主动发言。"""
        await self.get_or_create_heartflow_chat(channel_id)
        self.touch(channel_id)

    # ---- 状态查询 ----

    def active_chat_count(self) -> int:
        return len(self.heartflow_chat_list)

    def list_active_chats(self) -> List[Any]:
        return list(self.heartflow_chat_list.keys())

    def touch(self, chat_id: Any) -> None:
        """手动标记频道活跃"""
        self._active_since[chat_id] = _tm.time()

    def _sweep_phase_expirations(self) -> None:
        timeout = _phase_idle_timeout_seconds()
        try:
            from src.chat.heart_flow.heartflow_decision import (
                acquire_decision_maker,
            )
            from src.chat.heart_flow.phase_governor import (
                acquire_phase_governor,
            )
        except Exception as exc:
            logger.debug(f"[心流] 阶段巡检模块加载失败: {exc}")
            return

        for chat_id in list(self.heartflow_chat_list.keys()):
            channel_id = str(chat_id)
            try:
                governor = acquire_phase_governor(channel_id)
                if governor.check_expiration(timeout):
                    acquire_decision_maker().signal_activity_change(channel_id)
                    logger.info(f"[心流] {channel_id} 状态过期，已回落")
            except Exception as exc:
                logger.debug(f"[心流] 状态巡检异常 {channel_id}: {exc}")

    async def _periodic_cleanup_loop(self) -> None:
        """后台协程，巡检阶段过期并定期回收不活跃聊天实例。"""
        try:
            from src.config.core_config_engine import get_core_config

            cleanup_interval = float(
                get_core_config().resolve_module_view("schedule").values.get("cleanup_chat_instances_interval_seconds", 300.0)
            )
        except Exception:
            cleanup_interval = 300.0
        next_cleanup_at = _tm.time() + cleanup_interval
        while True:
            try:
                await asyncio.sleep(5.0)
                self._sweep_phase_expirations()
                now = _tm.time()
                if now >= next_cleanup_at:
                    await self.cleanup_inactive()
                    next_cleanup_at = now + cleanup_interval
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug(f"[心流] 周期清理异常: {exc}")


heartflow = Heartflow()

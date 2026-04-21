import asyncio
import contextlib
from typing import List, Dict, Optional, Type, Tuple, Union, TYPE_CHECKING, Any

from src.common.logger import get_logger
from src.plugin_system.base.component_types import (
    EventType,
    EventHandlerInfo,
    MaiMessages,
    CustomEventHandlerResult,
)
from src.plugin_system.base.base_events_handler import BaseEventHandler

if TYPE_CHECKING:
    from src.chat.message_receive.message import MessageRecv, MessageSending

logger = get_logger("事件管理")


class EventsManager:
    """事件管理器，负责事件分发与处理器调度"""

    def __init__(self):
        self._events_subscribers: Dict[
            Union[EventType, str], List[BaseEventHandler]
        ] = {}
        self._handler_mapping: Dict[str, Type[BaseEventHandler]] = {}
        self._handler_tasks: Dict[str, List[asyncio.Task]] = {}
        self._events_result_history: Dict[
            Union[EventType, str], List[CustomEventHandlerResult]
        ] = {}
        self._history_enable_map: Dict[Union[EventType, str], bool] = {}
        for event in EventType:
            self.register_event(event, enable_history_result=False)

    def register_event(
        self,
        event_type: Union[EventType, str],
        enable_history_result: bool = False,
    ):
        """注册事件类型"""
        if event_type in self._events_subscribers:
            raise ValueError(f"事件类型 {event_type} 已存在")
        self._events_subscribers[event_type] = []
        self._history_enable_map[event_type] = enable_history_result
        if enable_history_result:
            self._events_result_history[event_type] = []

    def register_event_subscriber(
        self,
        handler_info: EventHandlerInfo,
        handler_class: Type[BaseEventHandler],
    ) -> bool:
        """注册事件处理器"""
        if not issubclass(handler_class, BaseEventHandler):
            logger.error(
                f"类 {handler_class.__name__} 不是 BaseEventHandler 的子类"
            )
            return False
        handler_name = handler_info.name
        if handler_name in self._handler_mapping:
            logger.warning(f"事件处理器 {handler_name} 已存在，跳过注册")
            return False
        if handler_info.event_type not in self._history_enable_map:
            logger.error(f"事件类型 {handler_info.event_type} 未注册")
            return False
        self._handler_mapping[handler_name] = handler_class
        return self._insert_event_handler(handler_class, handler_info)

    async def handle_mai_events(
        self,
        event_type: EventType,
        message: Optional[Union["MessageRecv", "MessageSending"]] = None,
        llm_prompt: Optional[str] = None,
        llm_response: Optional[Any] = None,
        stream_id: Optional[str] = None,
        action_usage: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[MaiMessages]]:
        """处理所有事件，根据事件类型分发给订阅的处理器"""
        from src.plugin_system.core import component_registry

        continue_flag = True
        transformed_message = self._prepare_message(
            event_type,
            message,
            llm_prompt,
            llm_response,
            stream_id,
            action_usage,
        )
        if transformed_message:
            transformed_message = transformed_message.deepcopy()
        handlers = self._events_subscribers.get(event_type, [])
        if not handlers:
            return True, None
        modified_message: Optional[MaiMessages] = None
        for handler in handlers:
            plugin_config = (
                component_registry.get_plugin_config(handler.plugin_name) or {}
            )
            handler.set_plugin_config(plugin_config)
            if handler.intercept_message or event_type == EventType.ON_STOP:
                should_continue, modified_message = (
                    await self._dispatch_intercepting_handler_task(
                        handler,
                        event_type,
                        modified_message or transformed_message,
                    )
                )
                continue_flag = continue_flag and should_continue
            else:
                self._dispatch_handler_task(
                    handler, event_type, transformed_message
                )
        return continue_flag, modified_message

    async def cancel_handler_tasks(self, handler_name: str) -> None:
        """取消事件处理器的所有任务"""
        tasks_to_be_cancelled = self._handler_tasks.get(handler_name, [])
        if remaining_tasks := [
            task for task in tasks_to_be_cancelled if not task.done()
        ]:
            for task in remaining_tasks:
                task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*remaining_tasks, return_exceptions=True),
                    timeout=5,
                )
                logger.info(f"已取消事件处理器 {handler_name} 的所有任务")
            except asyncio.TimeoutError:
                logger.warning(f"取消事件处理器 {handler_name} 的任务超时")
            except Exception as e:
                logger.error(
                    f"取消事件处理器 {handler_name} 的任务时发生异常: {e}"
                )
        if handler_name in self._handler_tasks:
            del self._handler_tasks[handler_name]

    async def unregister_event_subscriber(self, handler_name: str) -> bool:
        """取消注册事件处理器"""
        if handler_name not in self._handler_mapping:
            logger.warning(f"事件处理器 {handler_name} 不存在，无法取消注册")
            return False
        await self.cancel_handler_tasks(handler_name)
        handler_class = self._handler_mapping.pop(handler_name)
        if not self._remove_event_handler_instance(handler_class):
            return False
        logger.info(f"事件处理器 {handler_name} 已成功取消注册")
        return True

    async def get_event_result_history(
        self, event_type: Union[EventType, str]
    ) -> List[CustomEventHandlerResult]:
        """获取事件的结果历史记录"""
        if event_type == EventType.UNKNOWN:
            raise ValueError("未知事件类型")
        if event_type not in self._history_enable_map:
            raise ValueError(f"事件类型 {event_type} 未注册")
        if not self._history_enable_map[event_type]:
            raise ValueError(f"事件类型 {event_type} 的历史记录未启用")
        return self._events_result_history[event_type]

    async def clear_event_result_history(
        self, event_type: Union[EventType, str]
    ) -> None:
        """清空事件的结果历史记录"""
        if event_type == EventType.UNKNOWN:
            raise ValueError("未知事件类型")
        if event_type not in self._history_enable_map:
            raise ValueError(f"事件类型 {event_type} 未注册")
        if not self._history_enable_map[event_type]:
            raise ValueError(f"事件类型 {event_type} 的历史记录未启用")
        self._events_result_history[event_type] = []

    def _insert_event_handler(
        self,
        handler_class: Type[BaseEventHandler],
        handler_info: EventHandlerInfo,
    ) -> bool:
        """插入事件处理器到对应的事件类型列表中"""
        if handler_class.event_type == EventType.UNKNOWN:
            logger.error(f"事件处理器 {handler_class.__name__} 的事件类型未知")
            return False
        if handler_class.event_type not in self._events_subscribers:
            self._events_subscribers[handler_class.event_type] = []
        handler_instance = handler_class()
        handler_instance.set_plugin_name(handler_info.plugin_name or "unknown")
        self._events_subscribers[handler_class.event_type].append(
            handler_instance
        )
        self._events_subscribers[handler_class.event_type].sort(
            key=lambda x: x.weight, reverse=True
        )
        return True

    def _remove_event_handler_instance(
        self, handler_class: Type[BaseEventHandler]
    ) -> bool:
        """从事件类型列表中移除事件处理器"""
        display_handler_name = (
            handler_class.handler_name or handler_class.__name__
        )
        if handler_class.event_type == EventType.UNKNOWN:
            logger.warning(f"事件处理器 {display_handler_name} 的事件类型未知")
            return False
        handlers = self._events_subscribers[handler_class.event_type]
        for i, handler in enumerate(handlers):
            if isinstance(handler, handler_class):
                del handlers[i]
                logger.debug(f"事件处理器 {display_handler_name} 已移除")
                return True
        logger.warning(f"未找到事件处理器 {display_handler_name}")
        return False

    def _transform_event_message(
        self,
        message: Union["MessageRecv", "MessageSending"],
        llm_prompt: Optional[str] = None,
        llm_response: Optional[Any] = None,
    ) -> MaiMessages:
        """转换事件消息格式"""
        transformed_message = MaiMessages(
            llm_prompt=llm_prompt,
            llm_response_content=(
                llm_response.content if llm_response else None
            ),
            llm_response_reasoning=(
                llm_response.reasoning if llm_response else None
            ),
            llm_response_model=llm_response.model if llm_response else None,
            llm_response_tool_call=(
                llm_response.tool_calls if llm_response else None
            ),
            raw_message=getattr(message, "raw_message", None),
            additional_data=(
                getattr(message.message_info, "additional_config", {})
                if hasattr(message, "message_info")
                else {}
            ),
        )
        if hasattr(message, "message_segment"):
            if message.message_segment.type == "seglist":
                transformed_message.message_segments = list(
                    message.message_segment.data
                )
            else:
                transformed_message.message_segments = [
                    message.message_segment
                ]
        if hasattr(message, "chat_stream") and message.chat_stream:
            transformed_message.stream_id = message.chat_stream.stream_id
        transformed_message.plain_text = getattr(
            message, "processed_plain_text", ""
        )
        if hasattr(message, "message_info") and message.message_info:
            if message.message_info.platform:
                transformed_message.message_base_info["platform"] = (
                    message.message_info.platform
                )
            if message.message_info.group_info:
                transformed_message.is_group_message = True
                transformed_message.message_base_info.update(
                    {"group_id": message.message_info.group_info.group_id,
                     "group_name": message.message_info.group_info.group_name, })
            if message.message_info.user_info:
                if not transformed_message.is_group_message:
                    transformed_message.is_private_message = True
                transformed_message.message_base_info.update(
                    {
                        "user_id": message.message_info.user_info.user_id,
                        "user_cardname": message.message_info.user_info.user_cardname,
                        "user_nickname": message.message_info.user_info.user_nickname,
                    }
                )
        return transformed_message

    def _build_message_from_stream(
        self,
        stream_id: str,
        llm_prompt: Optional[str] = None,
        llm_response: Optional[Any] = None,
    ) -> MaiMessages:
        """从流ID构建消息"""
        from src.chat.message_receive.chat_stream import get_chat_manager

        chat_stream = get_chat_manager().get_stream(stream_id)
        if not chat_stream:
            raise ValueError(f"未找到流ID为 {stream_id} 的聊天流")

        # 空闲主动场景下，流可能尚未绑定上下文消息；此时降级为无 message 模式
        if chat_stream.context is None:
            logger.debug(
                f"事件转换缺少上下文，降级构建消息: stream_id={stream_id}"
            )
            return self._transform_event_without_message(
                stream_id, llm_prompt, llm_response
            )

        message = chat_stream.context.get_last_message()
        return self._transform_event_message(message, llm_prompt, llm_response)

    def _transform_event_without_message(
        self,
        stream_id: str,
        llm_prompt: Optional[str] = None,
        llm_response: Optional[Any] = None,
        action_usage: Optional[List[str]] = None,
    ) -> MaiMessages:
        """没有message对象时进行转换"""
        from src.chat.message_receive.chat_stream import get_chat_manager

        chat_stream = get_chat_manager().get_stream(stream_id)
        if not chat_stream:
            raise ValueError(f"未找到流ID为 {stream_id} 的聊天流")
        return MaiMessages(
            stream_id=stream_id,
            llm_prompt=llm_prompt,
            llm_response_content=(
                llm_response.content if llm_response else None
            ),
            llm_response_reasoning=(
                llm_response.reasoning if llm_response else None
            ),
            llm_response_model=llm_response.model if llm_response else None,
            llm_response_tool_call=(
                llm_response.tool_calls if llm_response else None
            ),
            is_group_message=bool(chat_stream.group_info),
            is_private_message=not chat_stream.group_info,
            action_usage=action_usage,
            additional_data={"response_is_processed": True},
        )

    def _prepare_message(
        self,
        event_type: EventType,
        message: Optional[Union["MessageRecv", "MessageSending"]] = None,
        llm_prompt: Optional[str] = None,
        llm_response: Optional[Any] = None,
        stream_id: Optional[str] = None,
        action_usage: Optional[List[str]] = None,
    ) -> Optional[MaiMessages]:
        """根据事件类型和输入，准备和转换消息对象"""
        if message:
            return self._transform_event_message(
                message, llm_prompt, llm_response
            )
        if event_type not in [EventType.ON_START, EventType.ON_STOP]:
            if not stream_id:
                raise ValueError("如果没有消息，必须为非启动/关闭事件提供流ID")
            if event_type in [
                EventType.ON_MESSAGE,
                EventType.ON_PLAN,
                EventType.POST_LLM,
                EventType.AFTER_LLM,
            ]:
                return self._build_message_from_stream(
                    stream_id, llm_prompt, llm_response
                )
            else:
                return self._transform_event_without_message(
                    stream_id, llm_prompt, llm_response, action_usage
                )
        return None

    def _dispatch_handler_task(
        self,
        handler: BaseEventHandler,
        event_type: Union[EventType, str],
        message: Optional[MaiMessages] = None,
    ):
        """分发一个非阻塞的事件处理任务"""
        if event_type == EventType.UNKNOWN:
            raise ValueError("未知事件类型")
        try:
            task = asyncio.create_task(handler.execute(message))
            task_name = f"{handler.plugin_name}-{handler.handler_name}"
            task.set_name(task_name)

            def _safe_done_cb(finished_task, _evt=event_type, _name=task_name):
                try:
                    self._task_done_callback(finished_task, _evt)
                except Exception as cb_err:
                    logger.error(f"事件回调异常 [{_name}]: {cb_err}")

            task.add_done_callback(_safe_done_cb)
            self._handler_tasks.setdefault(handler.handler_name, []).append(
                task
            )
        except Exception as e:
            logger.error(
                f"创建事件处理器任务 {
                    handler.handler_name} 时发生异常: {e}",
                exc_info=True,
            )

    async def _dispatch_intercepting_handler_task(
        self,
        handler: BaseEventHandler,
        event_type: Union[EventType, str],
        message: Optional[MaiMessages] = None,
    ) -> Tuple[bool, Optional[MaiMessages]]:
        """分发并等待一个阻塞的事件处理器"""
        if event_type == EventType.UNKNOWN:
            raise ValueError("未知事件类型")
        if event_type not in self._history_enable_map:
            raise ValueError(f"事件类型 {event_type} 未注册")
        try:
            result = await handler.execute(message)
            if not isinstance(result, tuple) or len(result) != 5:
                logger.error(
                    f"EventHandler {handler.handler_name} 返回值不符合预期"
                )
                return True, None
            (
                success,
                continue_processing,
                return_message,
                custom_result,
                modified_message,
            ) = result
            if not success:
                logger.error(
                    f"EventHandler {
                        handler.handler_name} 执行失败: {return_message}"
                )
            else:
                logger.debug(f"EventHandler {handler.handler_name} 执行成功")
            if self._history_enable_map[event_type] and custom_result:
                self._events_result_history[event_type].append(custom_result)
            return continue_processing, modified_message
        except KeyError:
            logger.error(
                f"事件 {event_type} 注册的历史记录启用情况与实际不符合"
            )
            return True, None
        except Exception as e:
            logger.error(
                f"EventHandler {
                    handler.handler_name} 发生异常: {e}",
                exc_info=True,
            )
            return True, None

    def _task_done_callback(
        self, task: asyncio.Task, event_type: Union[EventType, str]
    ):
        """任务完成回调"""
        task_name = task.get_name() or "Unknown Task"
        if event_type == EventType.UNKNOWN:
            raise ValueError("未知事件类型")
        if event_type not in self._history_enable_map:
            raise ValueError(f"事件类型 {event_type} 未注册")
        try:
            success, _, result, custom_result, _ = task.result()
            if success:
                logger.debug(f"事件处理任务 {task_name} 已成功完成")
            else:
                logger.error(f"事件处理任务 {task_name} 执行失败: {result}")
            if self._history_enable_map[event_type] and custom_result:
                self._events_result_history[event_type].append(custom_result)
        except asyncio.CancelledError:
            pass
        except KeyError:
            logger.error(
                f"事件 {event_type} 注册的历史记录启用情况与实际不符合"
            )
        except Exception as e:
            logger.error(f"事件处理任务 {task_name} 发生异常: {e}")
        finally:
            with contextlib.suppress(ValueError, KeyError):
                self._handler_tasks[task_name].remove(task)


events_manager = EventsManager()

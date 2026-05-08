from src.chat.chat_core_base import ChatCoreBase
import asyncio
import time
import traceback
import random
from typing import List, Optional, TYPE_CHECKING
from src.config.config import global_config
from src.common.logger import get_logger
from src.common.data_models.message_data_model import ReplyContentType
from src.plugin_system.apis import send_api, message_api
from src.memory_system.chat_history_summarizer import ChatHistorySummarizer

if TYPE_CHECKING:
    from src.common.data_models.database_data_model import DatabaseMessages
    from src.common.data_models.message_data_model import ReplySetModel

logger = get_logger("心流核心")


class HeartFChatting(ChatCoreBase):
    """管理一个连续的 Focus Chat 循环，用于在特定聊天流中生成回复。
    其生命周期现在由其关联的 SubHeartflow 的 FOCUSED 状态控制。

    继承耦合评估（供后续退役参考）：
    ─────────────────────────────────
    【Enhanced 仍在使用的能力】
      ChatCoreBase.__init__: stream_id, chat_stream, log_prefix,
                             action_manager, action_modifier, running,
                             _loop_task, last_read_time
      ChatCoreBase._handle_loop_completion: 循环任务完成回调
      HeartFChatting.__init__: last_active_time, chat_history_summarizer
      HeartFChatting.start/stop: 循环生命周期管理
      HeartFChatting._main_chat_loop: 循环运行器
      HeartFChatting._send_response: 消息发送

    【Enhanced 已覆盖（不使用父版本）】
      _classify_behavior_signal, _decide_reply_style,
      _select_preferred_reply_message, _capture_reply_behavior_learning,
      _loopbody

    【Enhanced 完全未使用（安全移除候选）】
      ChatCoreBase: expression_learner, start_cycle, end_cycle,
                    print_cycle_info, _invoke_unified_planner_gate,
                    _convert_unified_decision_to_actions,
                    _build_fallback_actions, _build_relation_style_hint,
                    _build_relationship_target_hint
      HeartFChatting: is_mute, question_probability_multiplier,
                      questioned, consecutive_no_reply_count,
                      _history_loop_limit

    结论：旧链可压缩为"初始化+循环管理+发送"三能力最小壳，
          但目前不建议拆分，因为 action_manager/action_modifier
          初始化深度耦合 ChatCoreBase，风险收益比不合适。
    """

    def __init__(self, chat_id: str):
        """
        HeartFChatting 初始化函数

        参数:
            chat_id: 聊天流唯一标识符(如stream_id)
            on_stop_focus_chat: 当收到stop_focus_chat命令时调用的回调函数
            performance_version: 性能记录版本号，用于区分不同启动版本
        """
        super().__init__(chat_id)

        self.is_mute = False

        self.last_active_time = time.time()  # 记录上一次非noreply时间

        self.question_probability_multiplier = 1
        self.questioned = False

        # 跟踪连续 no_reply 次数，用于动态调整阈值
        self.consecutive_no_reply_count = 0
        self._history_loop_limit = 200

        # 聊天内容概括器
        self.chat_history_summarizer = ChatHistorySummarizer(
            chat_id=self.stream_id
        )

    async def start(self):
        """检查是否需要启动主循环，如果未激活则启动。"""

        # 如果循环已经激活，直接返回
        if self.running:
            logger.debug(
                f"{self.log_prefix} HeartFChatting 已激活，无需重复启动"
            )
            return

        try:
            # 标记为活动状态，防止重复启动
            self.running = True

            self._loop_task = asyncio.create_task(self._main_chat_loop())
            self._loop_task.add_done_callback(self._handle_loop_completion)

            # 启动聊天内容概括器的后台定期检查循环
            await self.chat_history_summarizer.start()

            logger.info(f"{self.log_prefix} HeartFChatting 启动完成")

        except Exception as e:
            # 启动失败时重置状态
            self.running = False
            self._loop_task = None
            logger.error(f"{self.log_prefix} HeartFChatting 启动失败: {e}")
            raise

    async def _loopbody(self):
        """旧版循环体已废弃，群聊职责已移交 EnhancedHeartFChatting。
        此方法仅作为虚方法保留，由子类完全覆盖。"""
        logger.warning(f"{self.log_prefix} 旧版循环体被调用，应由子类覆盖")
        await asyncio.sleep(3.0)
        return True

    async def _main_chat_loop(self):
        """主循环，调用子类覆盖的 _loopbody 直到被取消。"""
        try:
            while self.running:
                success = await self._loopbody()
                await asyncio.sleep(0.1)
                if not success:
                    break
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} 活力已关闭聊天")
            return
        except Exception:
            logger.error(
                f"{self.log_prefix} 活力聊天意外错误，将于3s后尝试重新启动"
            )
            logger.error(traceback.format_exc())
            await asyncio.sleep(3)
            self._loop_task = asyncio.create_task(self._main_chat_loop())
            self._loop_task.add_done_callback(self._handle_loop_completion)
            return
        logger.info(f"{self.log_prefix} 结束了当前聊天循环")

    async def _send_response(
        self,
        reply_set: "ReplySetModel",
        message_data: Optional["DatabaseMessages"],
        selected_expressions: Optional[List[int]] = None,
        quote_message: Optional[bool] = None,
    ) -> str:
        """发送回复到聊天流，由增强链复用。"""
        if message_data is None:
            need_reply = False
        elif global_config.chat.llm_quote:
            if quote_message is None:
                logger.warning(
                    f"{self.log_prefix} quote_message 参数为空，不引用"
                )
                need_reply = False
            else:
                need_reply = quote_message
                if need_reply:
                    logger.info(f"{self.log_prefix} LLM 决定使用引用回复")
        else:
            new_message_count = message_api.count_new_messages(
                chat_id=self.chat_stream.stream_id,
                start_time=self.last_read_time,
                end_time=time.time(),
            )
            need_reply = (
                new_message_count >= random.randint(2, 3)
                or time.time() - self.last_read_time > 90
            )
            if need_reply:
                logger.info(
                    f"{self.log_prefix} 从思考到回复，共有{new_message_count}条新消息，使用引用回复，或者上次回复时间超过90秒"
                )
        reply_text = ""
        for reply_content in reply_set.reply_data:
            if reply_content.content_type != ReplyContentType.TEXT:
                continue
            data: str = reply_content.content  # type: ignore
            reply_text += data
        if reply_text.strip():
            await send_api.text_to_stream(
                text=reply_text,
                stream_id=self.chat_stream.stream_id,
                reply_message=message_data,
                set_reply=need_reply,
                typing=False,
                selected_expressions=selected_expressions,
            )
        return reply_text

    async def stop(self) -> None:
        """停止主循环。"""
        self.running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        try:
            await self.chat_history_summarizer.stop()
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        self.history_loop.clear()
        logger.info(f"{self.log_prefix} HeartFChatting 已停止")

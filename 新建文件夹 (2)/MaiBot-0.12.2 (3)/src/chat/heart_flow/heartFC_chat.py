import asyncio
import time
import traceback
import random
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING
from rich.traceback import install

from src.config.config import global_config
from src.chat.heart_flow.config import get_heartflow_config
from src.common.logger import get_logger
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.common.data_models.message_data_model import ReplyContentType
from src.chat.message_receive.chat_stream import ChatStream, get_chat_manager
from src.chat.utils.prompt_builder import global_prompt_manager
from src.chat.utils.timer_calculator import Timer
from src.chat.planner_actions.planner import ActionPlanner
from src.chat.planner_actions.action_modifier import ActionModifier
from src.chat.planner_actions.action_manager import ActionManager
from src.chat.heart_flow.hfc_utils import CycleDetail
from src.bw_learner.expression_learner import get_expression_learner
from src.chat.heart_flow.frequency_control import frequency_control_manager
from src.bw_learner.reflect_tracker import reflect_tracker_manager
from src.bw_learner.expression_reflector import expression_reflector_manager
from src.bw_learner.message_recorder import extract_and_distribute_messages
from src.person_info.person_info import Person
from src.plugin_system.base.component_types import EventType, ActionInfo
from src.plugin_system.core.events_manager import events_manager
from src.plugin_system.apis import generator_api, send_api, message_api, database_api
from src.chat.utils.chat_message_builder import (
    build_readable_messages_with_id,
    get_raw_msg_before_timestamp_with_chat,
)
from src.chat.utils.utils import record_replyer_action_temp
from src.memory_system.chat_history_summarizer import ChatHistorySummarizer

if TYPE_CHECKING:
    from src.common.data_models.database_data_model import DatabaseMessages
    from src.common.data_models.message_data_model import ReplySetModel


ERROR_LOOP_INFO = {
    "loop_plan_info": {
        "action_result": {
            "action_type": "error",
            "action_data": {},
            "reasoning": "循环处理失败",
        },
    },
    "loop_action_info": {
        "action_taken": False,
        "reply_text": "",
        "command": "",
        "taken_time": time.time(),
    },
}


install(extra_lines=3)

# 注释：原来的动作修改超时常量已移除，因为改为顺序执行

logger = get_logger("hfc")  # Logger Name Changed


class HeartFChatting:
    """
    管理一个连续的Focus Chat循环
    用于在特定聊天流中生成回复。
    其生命周期现在由其关联的 SubHeartflow 的 FOCUSED 状态控制。
    """

    def __init__(self, chat_id: str):
        """
        HeartFChatting 初始化函数

        参数:
            chat_id: 聊天流唯一标识符(如stream_id)
            on_stop_focus_chat: 当收到stop_focus_chat命令时调用的回调函数
            performance_version: 性能记录版本号，用于区分不同启动版本
        """
        # 基础属性
        self.stream_id: str = chat_id  # 聊天流ID
        self.chat_stream: ChatStream = get_chat_manager().get_stream(self.stream_id)  # type: ignore
        if not self.chat_stream:
            raise ValueError(f"无法找到聊天流: {self.stream_id}")
        self.log_prefix = f"[{get_chat_manager().get_stream_name(self.stream_id) or self.stream_id}]"

        self.expression_learner = get_expression_learner(self.stream_id)

        self.action_manager = ActionManager()
        self.action_planner = ActionPlanner(chat_id=self.stream_id, action_manager=self.action_manager)
        self.action_modifier = ActionModifier(action_manager=self.action_manager, chat_id=self.stream_id)

        # 循环控制内部状态
        self.running: bool = False
        self._loop_task: Optional[asyncio.Task] = None  # 主循环任务

        # 回复协调器（三通道共享）
        from src.chat.heart_flow.reply_coordinator import ReplyCoordinator
        self._reply_coordinator = ReplyCoordinator(
            reply_cooldown=5.0,
            lock_timeout=30.0
        )

        # 添加循环信息管理相关的属性
        self.history_loop: List[CycleDetail] = []
        self._cycle_counter = 0
        self._current_cycle_detail: CycleDetail = None  # type: ignore

        self.last_read_time = time.time() - 2
        try:
            import jieba
            jieba.initialize()
        except Exception:
            pass

        self.is_mute = False

        self.last_active_time = time.time()  # 记录上一次非noreply时间

        self.question_probability_multiplier = 1
        self.questioned = False

        # 跟踪连续 no_reply 次数，用于动态调整阈值
        self.consecutive_no_reply_count = 0

        self.chat_history_summarizer = ChatHistorySummarizer(chat_id=self.stream_id)
        self._active = True
        self._new_message_event = asyncio.Event()
        self._fatigue_level: float = 0.0
        self._consecutive_reply_count: int = 0
        self._reply_cooldown_base: float = 5.0
        self._reply_cooldown_multiplier: float = 1.5
        self._last_reply_time: float = 0.0
        self._message_queue: list = []
        self._last_memory_save_time: float = time.time()
        self._talk_value: float = global_config.chat.talk_frequency if hasattr(global_config.chat, 'talk_frequency') else 0.2
        self._bt = None
        try:
            from src.config.behavior_tuner import get_behavior_tuner
            self._bt = get_behavior_tuner()
        except Exception:
            pass
        _hf_cfg = get_heartflow_config()
        self._max_consecutive_replies: int = _hf_cfg.fatigue.threshold_max
        self._engage_base_prob: float = _hf_cfg.normal_chat.base_reply_probability
        self._heartflow_mode: str = self._detect_heartflow_mode()

        self._state_manager = None
        try:
            from src.chat.heart_flow.state_manager import get_heart_state_manager
            self._state_manager = get_heart_state_manager(self.stream_id, self)
        except Exception:
            pass

        self._last_proactive_time: float = 0.0
        self._proactive_daily_count: int = 0
        self._proactive_daily_date: str = ""
        self._last_proactive_decision: dict = {}
        self._proactive_check_count: int = 0
        self._proactive_attempted: bool = False
        self._inner_voice_generator = None
        
        self._focus_engaged: bool = False
        self._processed_msg_ids: set = set()
        self._last_thought_result = None
        self._last_reply_time: float = 0.0
        self._chat_value: float = _hf_cfg.focus_chat.chat_value_max
        self._brain_power: float = _hf_cfg.focus_chat.brain_power_max
        self._last_resource_recovery_time: float = time.time()
        self._fatigue_threshold_min: int = _hf_cfg.fatigue.threshold_min
        self._fatigue_threshold_max: int = _hf_cfg.fatigue.threshold_max
        self._fatigue_per_reply: float = _hf_cfg.fatigue.per_reply
        self._fatigue_accumulated_threshold: float = _hf_cfg.fatigue.accumulated_threshold
        self._fatigue_decay_rate: float = _hf_cfg.fatigue.decay_rate
        self._fatigue_max_duration: float = _hf_cfg.fatigue.max_duration
        self._fatigue_penalty: float = _hf_cfg.normal_chat.fatigue_penalty
        self._accumulated_fatigue: float = 0.0
        self._is_fatigued: bool = False
        self._fatigue_start_time: float = 0.0
        self._fatigue_duration: float = 0.0
        self._fatigue_reply_count: int = 0
        self._last_fatigue_decay_time: float = time.time()
        self._fatigue_threshold: int = self._calc_dynamic_fatigue_threshold()
        self._desire_level: str = "normal"
        self._desire_start_time: float = time.time()
        self._mention_count: int = 0
        self._last_mention_time: float = 0.0
        self._spam_penalty_factor: float = self._bt.get("spam_penalty_factor", 0.3) if self._bt else 0.3
        self._interest_weight: float = self._bt.get("interest_weight", 0.4) if self._bt else 0.4
        self._admin_boost: float = _hf_cfg.normal_chat.admin_boost
        self._independent_monitor = None
        self._model_driven_monitor = None
        self._is_ignoring: bool = False
        self._ignore_until_time: float = 0.0
        self._consecutive_replies: int = 0
        self._last_peek_time: float = 0.0
        self._state_db = None
        try:
            from src.chat.heart_flow.persistent_state_db import get_persistent_state_db
            self._state_db = get_persistent_state_db()
        except Exception:
            pass
        self._load_persistent_state()

        if self._heartflow_mode == "focus":
            try:
                # 初始化心流模式监控系统
                from src.chat.heart_flow.model_driven_monitor import ModelDrivenMonitor
                self._model_driven_monitor = ModelDrivenMonitor(self.stream_id, self)
                logger.info(f"{self.log_prefix} 心流模式初始化成功")
            except Exception as e:
                logger.error(f"{self.log_prefix} 心流模式初始化失败: {e}", exc_info=True)

    def _load_focus_config(self):
        try:
            from src.chat.heart_flow.config import get_heartflow_config
            return get_heartflow_config().focus_chat
        except Exception:
            from src.chat.heart_flow.config import FocusChatConfig
            return FocusChatConfig()

    def _detect_heartflow_mode(self) -> str:
        try:
            from src.chat.heart_flow.config import get_heartflow_config, is_focus_mode_for_stream
            is_private = not bool(getattr(self.chat_stream, 'group_info', None))
            stream_type = "private" if is_private else "group"
            if is_focus_mode_for_stream(stream_type):
                return "focus"
            config = get_heartflow_config()
            return config.mode.value
        except Exception:
            return "normal"

    @property
    def is_focus_mode(self) -> bool:
        return self._heartflow_mode == "focus"

    def _get_psychological_context(self, user_id: str) -> dict:
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(self.stream_id)
            state = tracker.get_user_state(user_id, create_if_missing=True)
            return {
                "affection": state.affection,
                "annoyance": state.annoyance,
                "trauma_score": state.trauma_score,
                "relationship": state.relationship,
                "fatigue": self._fatigue_level,
            }
        except Exception:
            return {"affection": 0, "annoyance": 0, "trauma_score": 0, "relationship": "陌生人", "fatigue": 0}



    async def _save_conversation_memory(self, content: str, user_id: str, importance: float = 0.5):
        try:
            now = time.time()
            if now - getattr(self, '_last_memory_save_time', 0) < 30:
                return
            self._last_memory_save_time = now
            from src.memory_system.memory_core import get_memory_manager
            mgr = get_memory_manager()
            mgr.create_memory(
                stream_id=self.stream_id, content=content,
                memory_type="group_chat", user_id=user_id,
                importance=importance,
            )
        except Exception as e:
            logger.debug(f"保存群聊记忆失败: {e}")

    def _record_self_behavior(self, action_type: str, content: str, user_id: str = ""):
        try:
            from src.memory_system.self_behavior_recorder import get_self_recorder
            recorder = get_self_recorder()
            recorder.record(
                stream_id=self.stream_id,
                action_type=action_type,
                content=content,
                target_user_id=user_id,
            )
        except Exception:
            pass

    def _update_resources(self, now: float):
        elapsed = now - self._last_resource_recovery_time
        if elapsed < 1.0:
            return
        self._last_resource_recovery_time = now
        _hf = get_heartflow_config()
        is_resting = getattr(self, '_is_resting', False)
        cv_rate = _hf.focus_chat.chat_rest_recovery_rate if is_resting else _hf.focus_chat.chat_recovery_rate
        bp_rate = _hf.stamina.recovery_rate if is_resting else _hf.focus_chat.brain_recovery_rate
        cv_max = _hf.focus_chat.chat_value_max
        bp_max = _hf.focus_chat.brain_power_max
        self._chat_value = min(cv_max, self._chat_value + cv_rate * elapsed)
        self._brain_power = min(bp_max, self._brain_power + bp_rate * elapsed)
        if self._model_driven_monitor:
            self._brain_power = min(bp_max, getattr(self._model_driven_monitor, '_stamina', self._brain_power))
        since_reply = now - self._last_reply_time
        if since_reply > 30.0 and self._consecutive_reply_count > 0:
            self._consecutive_reply_count = 0

    def _consume_reply_resources(self):
        _hf = get_heartflow_config()
        # 将消耗修改为类似旧版设计的动态浮动：1% - 5%，最大不超过10%
        cv_cost = min(random.uniform(1.0, 5.0), 10.0)
        bp_cost = min(random.uniform(1.0, 5.0), 10.0)
        self._chat_value = max(0.0, self._chat_value - cv_cost)
        self._brain_power = max(0.0, self._brain_power - bp_cost)
        self._consecutive_reply_count += 1
        logger.debug(
            f"{self.log_prefix} [资源消耗] 聊天值={self._chat_value:.0f} 脑力={self._brain_power:.0f} "
            f"连续回复={self._consecutive_reply_count}"
        )

    def can_reply(self, is_at_me: bool = False) -> tuple:
        if is_at_me:
            return True, "被@强制回复"
        _hf = get_heartflow_config()
        cv_floor = _hf.focus_chat.chat_value_max * 0.1
        bp_floor = _hf.focus_chat.brain_power_max * 0.05
        if self._chat_value < cv_floor:
            return False, f"聊天值耗尽({self._chat_value:.0f})"
        if self._brain_power < bp_floor:
            return False, f"脑力耗尽({self._brain_power:.0f})"
        if self._consecutive_reply_count >= self._max_consecutive_replies:
            return False, f"连续回复过多({self._consecutive_reply_count})"
        return True, "资源充足"

    def _display_modcore_status(self, user_id: str, user_name: str, state):
        try:
            uid_short = user_id[:2] + '****' if len(user_id) > 2 else user_id
            name_short = user_name[:6] if len(user_name) > 6 else user_name
            aff = max(-100.0, min(100.0, getattr(state, 'affection', 0)))
            aff_delta = getattr(state, '_last_affection_delta', 0)
            aff_arrow = "↑" if aff_delta > 0 else "↓" if aff_delta < 0 else ""
            trust = max(-100.0, min(100.0, getattr(state, 'trust_score', 0)))
            trust_delta = getattr(state, '_last_trust_delta', 0)
            trust_arrow = "↑" if trust_delta > 0 else "↓" if trust_delta < 0 else ""
            ann = max(0.0, min(100.0, getattr(state, 'annoyance', 0)))
            trauma = max(0.0, min(10.0, getattr(state, 'trauma_score', 0.0)))
            trauma_lv = 3 if trauma >= 7 else 2 if trauma >= 4 else 1 if trauma >= 1.0 else 0
            rel = getattr(state, 'relationship', '陌生人')
            trauma_display = f"创伤Lv{trauma_lv}({trauma:.1f})" if trauma_lv > 0 else f"创伤:0"
            core_parts = [
                f"{name_short}({uid_short})",
                f"好感:{aff:+.0f}{aff_arrow}",
                f"信任:{trust:.0f}{trust_arrow}",
                f"烦恼:{ann:.0f}" if ann > 0.1 else "烦恼:0",
                trauma_display,
                f"{rel}",
            ]
            surface_mask = getattr(state, 'surface_mask', -1)
            inner_chaos = getattr(state, 'inner_chaos', -1)
            psych_pressure = getattr(state, 'psychological_pressure', 0)
            mental_fatigue = getattr(state, 'mental_fatigue', 0)
            active_flags = []
            inactive_flags = []
            hidden_flags = []
            if surface_mask >= 0:
                if surface_mask >= 5.0:
                    active_flags.append(f"伪装:{surface_mask:.1f}/10")
                else:
                    inactive_flags.append(f"伪装:{surface_mask:.1f}/10")
            else:
                hidden_flags.append("伪装")
            if inner_chaos >= 0:
                if inner_chaos >= 3.0:
                    active_flags.append(f"混乱:{inner_chaos:.1f}/10")
                else:
                    inactive_flags.append(f"混乱:{inner_chaos:.1f}/10")
            else:
                hidden_flags.append("混乱")
            if psych_pressure > 5:
                active_flags.append(f"压力:{psych_pressure:.0f}")
            elif psych_pressure > 0:
                inactive_flags.append(f"压力:{psych_pressure:.0f}")
            if mental_fatigue > 10:
                active_flags.append(f"疲劳:{mental_fatigue:.0f}")
            elif mental_fatigue > 0:
                inactive_flags.append(f"疲劳:{mental_fatigue:.0f}")
            is_breakdown = getattr(state, 'is_system_breakdown', False)
            if is_breakdown:
                active_flags.append("心理过载")
            if hasattr(state, 'is_blocked') and state.is_blocked:
                block_until = getattr(state, 'block_until', 0)
                if block_until > time.time():
                    rem = int(block_until - time.time())
                    active_flags.append(f"保护中({rem}s)")
                else:
                    inactive_flags.append("保护(已过期)")
            if hasattr(state, 'is_protected') and state.is_protected:
                protect_until = getattr(state, 'protect_until', 0)
                if protect_until > time.time():
                    active_flags.append(f"用户保护({int(protect_until - time.time())}s)")
            if hasattr(state, 'training_stage') and getattr(state, 'training_stage', 0) > 0:
                sub = getattr(state, 'submission_level', 0)
                shy = getattr(state, 'shyness_level', 0)
                active_flags.append(f"调教Lv{state.training_stage}(服从{sub:.0f}%|羞耻{shy:.0f}%)")
            else:
                hidden_flags.append("调教")
            persona_info = ""
            try:
                from src.modules.modcore.dynamic_persona import get_persona_controller
                ctrl = get_persona_controller()
                active_p = ctrl.get_active_persona(self.stream_id)
                if active_p:
                    rem_str = ""
                    try:
                        ss = getattr(ctrl, '_stream_states', {}).get(self.stream_id, {})
                        et = ss.get("end_time", 0)
                        if et > 0:
                            rem_min = max(0, int((et - time.time()) / 60))
                            rem_str = f"({rem_min}m)" if rem_min > 0 else ""
                    except Exception:
                        pass
                    active_flags.append(f"人格:{active_p.name}{rem_str}")
                else:
                    inactive_flags.append("人格:默认")
            except Exception:
                hidden_flags.append("人格")
            _hf = get_heartflow_config()
            cv_max = _hf.focus_chat.chat_value_max
            bp_max = _hf.focus_chat.brain_power_max
            cv_pct = (self._chat_value / cv_max * 100) if cv_max > 0 else 0
            bp_pct = (self._brain_power / bp_max * 100) if bp_max > 0 else 0
            resource_parts = [f"聊天值:{cv_pct:.0f}%", f"脑力:{bp_pct:.0f}%"]
            if self._consecutive_reply_count > 0:
                resource_parts.append(f"连续x{self._consecutive_reply_count}")
            mode_label = "专注" if self._heartflow_mode == "focus" else "普通"
            resource_parts.append(f"模式:{mode_label}")
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
                tracker = get_emotion_tracker(self.stream_id)
                rage = tracker.get_global_rage_level()
                if rage > 50:
                    active_flags.append(f"全局激怒:{rage:.0f}")
                elif rage > 10:
                    inactive_flags.append(f"全局怒气:{rage:.0f}")
            except Exception:
                pass
            try:
                is_private_chat = not bool(getattr(self.chat_stream, 'group_info', None))
                if not is_private_chat:
                    from src.modules.stream_emotion_state import get_stream_emotion_manager
                    emotion_mgr = get_stream_emotion_manager()
                    stream_emotion = emotion_mgr.get_emotion_state(self.stream_id)
                    if stream_emotion:
                        fatigue_pct = stream_emotion.accumulated_fatigue
                        annoy_pct = stream_emotion.annoyance
                        if fatigue_pct > 50 or annoy_pct > 50:
                            active_flags.append(f"群聊疲劳:{fatigue_pct:.0f}|烦躁:{annoy_pct:.0f}")
                        elif fatigue_pct > 20 or annoy_pct > 20:
                            inactive_flags.append(f"群聊累:{fatigue_pct:.0f}|烦:{annoy_pct:.0f}")
            except Exception:
                pass
            main_line = " | ".join(core_parts)
            logger.info(f"{self.log_prefix} {main_line}")
            res_line = " | ".join(resource_parts)
            act_str = " ".join(f"[{f}]" for f in active_flags) if active_flags else ""
            ext_parts = [res_line]
            if act_str:
                ext_parts.append(act_str)
            logger.info(f"{self.log_prefix} {' | '.join(ext_parts)}")
        except Exception as e:
            logger.debug(f"状态显示错误: {e}")

    async def _notify_heart_state(self, event_type: str, data: dict = None):
        if not self._state_manager:
            return
        try:
            if event_type == "message":
                is_at_me = data.get('is_at_me', False) if data else False
                self._state_manager.on_message_received(data or {}, is_at_me)
            elif event_type == "reply":
                self._state_manager.on_reply_sent()
            elif event_type == "no_reply":
                self._state_manager.on_no_reply()
        except Exception:
            pass

    def enqueue_message(self, message_dict: dict):
        self._message_queue.append(message_dict)
        if len(self._message_queue) > 100:
            self._message_queue = self._message_queue[-50:]
        self._proactive_attempted = False
        self._new_message_event.set()
        
        # 唤醒心流监控循环
        if self._model_driven_monitor and getattr(self._model_driven_monitor, 'is_running', False):
            self._model_driven_monitor.on_new_message()

    def get_pending_messages(self) -> list:
        msgs = list(self._message_queue)
        self._message_queue.clear()
        return msgs

    def get_stats(self) -> dict:
        stats = {
            "stream_id": self.stream_id,
            "cycle_count": self._cycle_counter,
            "consecutive_no_reply": self.consecutive_no_reply_count,
            "consecutive_reply": self._consecutive_reply_count,
            "is_mute": self.is_mute,
            "running": self.running,
            "pending_messages": len(self._message_queue),
        }

        if self._model_driven_monitor:
            stats["model_driven_monitor"] = self._model_driven_monitor.get_status()

        if self._independent_monitor:
            stats["independent_monitor"] = self._independent_monitor.get_status()

        if self._state_manager:
            stats["state_manager"] = self._state_manager.get_status()

        return stats

    async def start(self):
        """检查是否需要启动主循环，如果未激活则启动。"""

        # 如果循环已经激活，直接返回
        if self.running:
            logger.debug(f"{self.log_prefix} 聊天节奏已激活，无需重复启动")
            return

        try:
            # 标记为活动状态，防止重复启动
            self.running = True

            # 专注模式：启动心流模式监控系统（新版）
            if self._heartflow_mode == "focus" and self._model_driven_monitor:
                await self._model_driven_monitor.start()
                logger.info(f"{self.log_prefix} 心流模式已启动")
            # 旧版监控已移除，完全由心流模式接管
            
            # 再启动主循环
            self._loop_task = asyncio.create_task(self._main_chat_loop())
            self._loop_task.add_done_callback(self._handle_loop_completion)

            await self.chat_history_summarizer.start()

            logger.info(f"{self.log_prefix} 聊天节奏启动完成")

        except Exception as e:
            self.running = False
            self._loop_task = None
            logger.error(f"{self.log_prefix} 聊天节奏启动失败: {e}")
            raise

    def _handle_loop_completion(self, task: asyncio.Task):
        """当 _hfc_loop 任务完成时执行的回调。"""
        try:
            if exception := task.exception():
                logger.error(f"{self.log_prefix} 聊天节奏: 脱离了聊天(异常): {exception}")
                logger.error(traceback.format_exc())  # Log full traceback for exceptions
            else:
                logger.info(f"{self.log_prefix} 聊天节奏: 脱离了聊天(外部停止)")
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} 聊天节奏: 结束了聊天")
        
        # 注意：心流模式不会在这里停止
        # 它会继续运行，直到明确调用 stop_all()
        if self._model_driven_monitor and self._model_driven_monitor.is_running:
            logger.info(f"{self.log_prefix} 心流模式继续运行")
    
    async def stop_all(self):
        """
        完全停止所有任务
        """
        logger.info(f"{self.log_prefix} 开始停止所有任务...")

        # 停止主循环
        self.running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

        # 停止心流模式监控系统
        if self._model_driven_monitor:
            await self._model_driven_monitor.stop()

        logger.info(f"{self.log_prefix} 所有任务已停止")

    def start_cycle(self) -> Tuple[Dict[str, float], str]:
        self._cycle_counter += 1
        self._current_cycle_detail = CycleDetail(self._cycle_counter)
        self._current_cycle_detail.thinking_id = f"tid{str(round(time.time(), 2))}"
        cycle_timers = {}
        return cycle_timers, self._current_cycle_detail.thinking_id

    def end_cycle(self, loop_info, cycle_timers):
        self._current_cycle_detail.set_loop_info(loop_info)
        self.history_loop.append(self._current_cycle_detail)
        self._current_cycle_detail.timers = cycle_timers
        self._current_cycle_detail.end_time = time.time()

    def print_cycle_info(self, cycle_timers):
        # 记录循环信息和计时器结果
        timer_strings = []
        for name, elapsed in cycle_timers.items():
            if elapsed < 0.1:
                # 不显示小于0.1秒的计时器
                continue
            formatted_time = f"{elapsed:.2f}秒"
            timer_strings.append(f"{name}: {formatted_time}")

        logger.info(
            f"{self.log_prefix} 第{self._current_cycle_detail.cycle_id}次思考,"
            f"耗时: {self._current_cycle_detail.end_time - self._current_cycle_detail.start_time:.1f}秒;"  # type: ignore
            + (f"详情: {'; '.join(timer_strings)}" if timer_strings else "")
        )



    async def _loopbody(self):
        now = time.time()
        self._update_resources(now)

        if self._state_manager:
            # 状态管理器现在由 model_driven_monitor 驱动，这里只检查状态到期
            if self._state_manager:
                expired = self._state_manager.check_state_expiration()
                if expired:
                    logger.debug(f"{self.log_prefix} 状态到期: {expired['state']}")
        from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
        from src.modules.modcore.group_impression.group_impression_analyzer import get_group_impression_analyzer
        emotion_tracker = get_emotion_tracker(self.stream_id)
        if not hasattr(self, '_last_emotion_recovery_time'):
            self._last_emotion_recovery_time = 0.0
        if now - self._last_emotion_recovery_time >= 60.0:
            emotion_tracker.apply_time_recovery_to_all()
            self._last_emotion_recovery_time = now
        recent_messages_list = message_api.get_messages_by_time_in_chat(
            chat_id=self.stream_id,
            start_time=self.last_read_time,
            end_time=time.time(),
            limit=20,
            limit_mode="latest",
            filter_mai=True,
            filter_command=False,
            filter_intercept_message_level=0,
        )
        from src.modules.modcore.psychological_core import get_psychological_core, handle_message_unified
        psy_core = get_psychological_core()
        impression_analyzer = get_group_impression_analyzer()
        recorded_count = 0
        last_psy_response = None
        for msg in recent_messages_list:
            msg_id = getattr(msg, 'message_id', None) or id(msg)
            if msg_id in self._processed_msg_ids:
                continue
            self._processed_msg_ids.add(msg_id)
            msg_ui = getattr(msg, 'user_info', None)
            msg_sender = str(msg_ui.user_id) if msg_ui else ""
            msg_text = getattr(msg, 'processed_plain_text', '') or ''
            if msg_sender and msg_text:
                impression_analyzer.record_message(self.stream_id, msg_sender, msg_text)
                recorded_count += 1
                msg_nickname = (getattr(msg_ui, 'user_nickname', '') or msg_sender[:8]) if msg_ui else msg_sender[:8]
                try:
                    from src.modules.stream_emotion_state import get_stream_emotion_manager
                    emotion_mgr = get_stream_emotion_manager()
                    is_admin = False
                    try:
                        from src.manager.permission_checker import is_admin_user
                        is_admin = is_admin_user(msg_sender)
                    except Exception:
                        pass
                    emotion_mgr.record_query(self.stream_id, msg_text, msg_sender, is_admin)
                except Exception as e:
                    logger.debug(f"记录群聊情绪失败: {e}")
                if '[picid:' in msg_text:
                    interested = await self._evaluate_image_interest_simple(msg_text, msg_nickname)
                    if interested:
                        logger.info(f"{self.log_prefix} [识图] 对 {msg_nickname} 的图片感兴趣，触发VLM分析")
                        await self._trigger_lazy_image_analysis(msg_text)
                    else:
                        logger.info(f"{self.log_prefix} [识图] 对 {msg_nickname} 的图片不感兴趣，跳过分析")
                last_psy_response = await handle_message_unified(
                    stream_id=self.stream_id, user_id=msg_sender,
                    content=msg_text,
                )
                _usr_state = emotion_tracker.get_user_state(msg_sender, create_if_missing=True)
                if _usr_state:
                    self._display_modcore_status(msg_sender, msg_nickname, _usr_state)
                if last_psy_response.should_block:
                    logger.info(f"{self.log_prefix} [modcore] 用户被拦截: uid={msg_sender[:8]}, 原因={last_psy_response.block_reason}")
                    if self.chat_stream.group_info:
                        platform_msg_id = str(getattr(msg, 'message_id', ''))
                        if platform_msg_id and not platform_msg_id.startswith("tid"):
                            asyncio.create_task(self._try_admin_recall_user_msg(platform_msg_id, msg_sender, last_psy_response.block_reason))
                    continue
                if last_psy_response.should_switch_persona:
                    logger.info(f"{self.log_prefix} [modcore] 人格切换触发: 目标={last_psy_response.target_persona}")
                    try:
                        from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
                        _sw_tracker = get_emotion_tracker(self.stream_id)
                        _sw_state = _sw_tracker.get_user_state(msg_sender, create_if_missing=False)
                        _sw_trauma = getattr(_sw_state, 'trauma_score', 0.0) if _sw_state else 0.0
                        _sw_target = last_psy_response.target_persona or "创伤反应"
                        _sw_type = "general"
                        if "骚扰" in str(last_psy_response.block_reason or ""):
                            _sw_type = "harassment"
                        from src.modules.modcore.dynamic_persona.persona_generator import get_persona_generator
                        from src.modules.modcore.dynamic_persona.persona_switcher import get_persona_switcher
                        from src.modules.modcore.dynamic_persona.persona_controller import get_persona_controller
                        _sw_generator = get_persona_generator()
                        _sw_mood_ctx = {
                            "user_id": msg_sender,
                            "channel_id": self.stream_id,
                            "stream_id": self.stream_id,
                            "current_content": msg_text[:200],
                            "trauma_level": _sw_trauma,
                            "trauma_score": _sw_trauma,
                            "intensity": 7 if _sw_trauma >= 7 else 5 if _sw_trauma > 5 else 3,
                            "triggers": [_sw_type],
                            "primary_emotion": "devastated" if _sw_trauma >= 7 else "hurt" if _sw_trauma < 5 else "distressed",
                            "mood_level": _sw_trauma,
                        }
                        _sw_persona = await _sw_generator.generate_trauma_sensitive_persona(_sw_type, _sw_mood_ctx)
                        if _sw_persona:
                            _sw_persona.name = _sw_target
                            _sw_ctrl = get_persona_controller()
                            _sw_ctrl.add_persona(_sw_persona)
                            _sw_dur = 420.0 if _sw_trauma >= 7 else 360.0 if _sw_trauma > 5 else 240.0 if _sw_trauma > 2 else 180.0
                            _sw_switcher = get_persona_switcher()
                            _sw_ok = await _sw_switcher.switch_with_auto_revert(
                                self.stream_id, _sw_persona.persona_id,
                                _sw_dur, reason=f"创伤触发: {_sw_type} ({_sw_target})"
                            )
                            if _sw_ok:
                                logger.info(f"{self.log_prefix} [modcore] 人格注入完成: '{_sw_target}' (持续{_sw_dur:.0f}秒)")
                    except Exception as _sw_e:
                        logger.debug(f"{self.log_prefix} [modcore] 人格预注入失败: {_sw_e}")
        if len(self._processed_msg_ids) > 200:
            self._processed_msg_ids = set(list(self._processed_msg_ids)[-100:])
        if recorded_count > 0:
            group_state = impression_analyzer.analyze_group_impression(self.stream_id)
            logger.debug(f"{self.log_prefix} [modcore] 群体印象: 记录{recorded_count}条, 氛围={group_state.atmosphere.value}, 定位={group_state.ai_position.value}")
        if self.consecutive_no_reply_count >= 5:
            threshold = 2
        elif self.consecutive_no_reply_count >= 3:
            threshold = 2 if random.random() < 0.5 else 1
        else:
            threshold = 1
        if len(recent_messages_list) >= threshold:
            self.last_read_time = time.time()
            # 专注模式：使用心流模式决策
            if self._heartflow_mode == "focus":
                # 在此只需依赖 _decide_after_observation(或者新的独立决策流) 
                # 旧版的连发消耗机制等随着分数系统均已移除
                await asyncio.sleep(5)
                return True
            
            
            # 非专注模式：使用概率判断
            else:
                # 非专注模式原来也是走到 observe 然后做决策，现在这块统一由新的逻辑处理。
                # 假设以后这里会整合到一个单独的处理方法，现在用挂起顶替。
                await asyncio.sleep(5)
                return True
        else:
            await asyncio.sleep(0.2)
            return True
        return True




    async def _send_and_store_reply(
        self,
        response_set: "ReplySetModel",
        action_message: "DatabaseMessages",
        cycle_timers: Dict[str, float],
        thinking_id,
        actions,
        selected_expressions: Optional[List[int]] = None,
        quote_message: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, float]]:
        with Timer("回复发送", cycle_timers):
            reply_text = await self._send_response(
                reply_set=response_set,
                message_data=action_message,
                selected_expressions=selected_expressions,
                quote_message=quote_message,
            )

            # --- [新增] 防刷屏机制：根据即将发出的文字长度决定延迟 ---
            if reply_text:
                # 假设每秒能打字或者想内容大概 10 字符，加点基础延迟
                delay_sec = max(2.0, len(reply_text) * 0.1)
                # 避免等得过分长（限制最大12秒)
                delay_sec = min(delay_sec, 12.0)
                logger.info(f"{self.log_prefix} [防刷屏] 准备回复的内容较长，模拟思考与打字延迟 {delay_sec:.1f} 秒...")
                try:
                    await asyncio.sleep(delay_sec)
                except asyncio.CancelledError:
                    pass
            # --------------------------------------------------------

        # 获取 platform，如果不存在则从 chat_stream 获取，如果还是 None 则使用默认值
        platform = action_message.chat_info.platform
        if platform is None:
            platform = getattr(self.chat_stream, "platform", "unknown")

        person = Person(platform=platform, user_id=action_message.user_info.user_id)
        person_name = person.person_name
        action_prompt_display = f"你对{person_name}进行了回复：{reply_text}"

        await database_api.store_action_info(
            chat_stream=self.chat_stream,
            action_build_into_prompt=False,
            action_prompt_display=action_prompt_display,
            action_done=True,
            thinking_id=thinking_id,
            action_data={"reply_text": reply_text},
            action_name="reply",
        )

        if reply_text and len(reply_text) >= 8:
            try:
                from src.modules.modcore.decision_engine.recall_monitor import get_recall_monitor
                monitor = get_recall_monitor()
                msg_id = str(thinking_id or id(reply_text))
                user_id = str(action_message.user_info.user_id) if action_message.user_info else ""
                review_weight = 0.3
                psy_state = {}
                try:
                    from src.modules.modcore.psychological_core import get_psychological_state
                    psy_state = get_psychological_state(self.chat_stream.stream_id, user_id)
                except Exception:
                    pass
                annoyance = psy_state.get("annoyance", 0)
                trauma = psy_state.get("trauma", 0)
                surface_mask = 10.0
                inner_chaos = 0.0
                mental_fatigue = psy_state.get("mental_fatigue", 0)
                try:
                    from src.chat.heart_flow.skills.recall_skill import get_focus_recall_skill
                    recall_skill = get_focus_recall_skill()
                    triggered, error_type, recall_probs = recall_skill.should_trigger_with_psychology(
                        surface_mask=surface_mask, inner_chaos=inner_chaos,
                        mental_fatigue=mental_fatigue, trauma_score=trauma,
                    )
                    if recall_probs:
                        review_weight = min(0.9, recall_probs.get("typo_probability", 0.15) * 3)
                except Exception:
                    pass
                if annoyance > 50 or trauma > 3:
                    review_weight = max(review_weight, 0.7)
                elif annoyance > 20:
                    review_weight = max(review_weight, 0.5)

                recall_stream_id = self.chat_stream.stream_id
                recall_reply_text = reply_text

                async def _do_recall(mid: str) -> bool:
                    try:
                        from src.chat.message_receive.uni_message_sender import UniversalMessageSender
                        from src.chat.message_receive.chat_stream import get_chat_manager
                        from src.common.database.database_model import Messages
                        from maim_message import Seg, UserInfo
                        from src.config.config import global_config
                        from src.llm_models.utils_model import LLMRequest
                        from src.config.config import model_config
                        import time as _time

                        target_stream = get_chat_manager().get_stream(recall_stream_id)
                        if not target_stream:
                            logger.warning(f"[撤回] 未找到目标stream: {recall_stream_id[:16]}")
                            return False

                        bot_user_info = UserInfo(
                            user_id=global_config.bot.qq_account,
                            user_nickname=global_config.bot.nickname,
                            platform=target_stream.platform,
                        )
                        from src.chat.message_receive.message import MessageSending
                        message_sender = UniversalMessageSender()

                        # 如果提供了具体的消息ID，需要先获取真实的平台消息ID
                        if mid:
                            actual_msg_id = mid
                            if mid.startswith("tid"):
                                # tid消息需要查找对应的真实消息ID，但只撤回第一条匹配的
                                await asyncio.sleep(1.0)
                                bot_qq = str(global_config.bot.qq_account)
                                try:
                                    matched_rows = (
                                        Messages.select(Messages.message_id, Messages.processed_plain_text)
                                        .where(
                                            (Messages.chat_id == recall_stream_id) &
                                            (Messages.user_id == bot_qq)
                                        )
                                        .order_by(Messages.time.desc())
                                        .limit(12)
                                    )
                                    # 只找第一条匹配的消息，不批量撤回
                                    found_msg_id = None
                                    for row in matched_rows:
                                        row_id = str(row.message_id)
                                        if row_id.startswith("tid") or row_id.startswith("recall_") or row_id.startswith("send_api_"):
                                            continue
                                        row_text = row.processed_plain_text or ""
                                        if row_text and row_text in recall_reply_text:
                                            found_msg_id = row_id
                                            logger.info(f"[撤回] 找到tid对应的消息: {row_id[:12]}")
                                            break
                                    
                                    if found_msg_id:
                                        actual_msg_id = found_msg_id
                                    else:
                                        logger.warning(f"[撤回] 未找到tid {mid[:8]} 对应的消息")
                                        return False
                                except Exception as db_err:
                                    logger.error(f"[撤回] 查询消息失败: {db_err}")
                                    return False
                            try:
                                cmd = {"action": "recall_message", "message_id": actual_msg_id, "timestamp": _time.time()}
                                msg = MessageSending(
                                    message_id=f"recall_{int(_time.time()*1000)}",
                                    chat_stream=target_stream, bot_user_info=bot_user_info,
                                    sender_info=target_stream.user_info,
                                    message_segment=Seg(type="command", data=cmd),
                                    display_message="", is_head=True, is_emoji=False,
                                    thinking_start_time=_time.time(),
                                )
                                ok = await message_sender.send_message(msg, typing=False, storage_message=False)
                                if ok:
                                    logger.info(f"[撤回] 成功撤回消息: {actual_msg_id[:16]}")
                                    return True
                                else:
                                    logger.warning(f"[撤回] 撤回消息失败: {actual_msg_id[:16]}")
                                    return False
                            except Exception as e:
                                logger.error(f"[撤回] 撤回消息出错: {e}")
                                return False

                        # 如果没有提供消息ID，则进行复杂的多消息审查流程
                        bot_qq = str(global_config.bot.qq_account)
                        segments = []
                        try:
                            matched = (
                                Messages.select(Messages.message_id, Messages.processed_plain_text)
                                .where(
                                    (Messages.chat_id == recall_stream_id) &
                                    (Messages.user_id == bot_qq)
                                )
                                .order_by(Messages.time.desc())
                                .limit(8)
                            )
                            for row in matched:
                                msg_id_str = str(row.message_id)
                                if msg_id_str.startswith("tid") or msg_id_str.startswith("recall_"):
                                    continue
                                txt = row.processed_plain_text or ""
                                if txt and (recall_reply_text[:20] in txt or txt in recall_reply_text):
                                    segments.append((msg_id_str, txt))
                        except Exception as db_err:
                            logger.warning(f"[撤回] 查询分段消息失败: {db_err}")

                        if not segments:
                            logger.warning(f"[撤回] 未找到可审查的分段消息")
                            return False

                        target_stream = get_chat_manager().get_stream(recall_stream_id)
                        if not target_stream:
                            return False

                        bot_user_info = UserInfo(
                            user_id=global_config.bot.qq_account,
                            user_nickname=global_config.bot.nickname,
                            platform=target_stream.platform,
                        )
                        from src.chat.message_receive.message import MessageSending
                        message_sender = UniversalMessageSender()

                        bad_ids = []
                        for seg_id, seg_text in segments:
                            try:
                                from src.config.prompt_loader import get_prompt, PromptCategory
                                check_prompt = get_prompt(
                                    PromptCategory.MODULE,
                                    "heart_flow",
                                    "segment_review.template",
                                    seg_text=seg_text
                                )
                                req = LLMRequest(model_config.lightweight, request_type="segment_review")
                                resp, _ = await req.generate_response_async(check_prompt, temperature=0.2, max_tokens=500)
                                if resp:
                                    import re as _re
                                    r = resp.strip()
                                    tm = _re.search(r'</think>\s*(.+)', r, _re.DOTALL)
                                    if tm:
                                        r = tm.group(1).strip()
                                    if "有问题" in r.split('\n')[0] and "没问题" not in r.split('\n')[0]:
                                        bad_ids.append(seg_id)
                                        logger.info(f"[撤回] 分段审查: {seg_text[:30]}... → 有问题")
                                    else:
                                        logger.debug(f"[撤回] 分段审查: {seg_text[:30]}... → 没问题")
                            except Exception:
                                pass

                        if not bad_ids:
                            logger.info(f"[撤回] 逐条审查完毕: {len(segments)}条均正常，不撤回")
                            return False

                        recalled = 0
                        for bad_id in bad_ids:
                            try:
                                cmd = {"action": "recall_message", "message_id": bad_id, "timestamp": _time.time()}
                                msg = MessageSending(
                                    message_id=f"recall_{int(_time.time()*1000)}_{recalled}",
                                    chat_stream=target_stream, bot_user_info=bot_user_info,
                                    sender_info=target_stream.user_info,
                                    message_segment=Seg(type="command", data=cmd),
                                    display_message="", is_head=True, is_emoji=False,
                                    thinking_start_time=_time.time(),
                                )
                                ok = await message_sender.send_message(msg, typing=False, storage_message=False)
                                if ok:
                                    recalled += 1
                                if len(bad_ids) > 1:
                                    await asyncio.sleep(0.5)
                            except Exception:
                                pass
                        logger.info(f"[撤回] 撤回{recalled}/{len(segments)}条(有问题{len(bad_ids)}条)")
                        return recalled > 0
                    except Exception as e:
                        logger.error(f"[撤回] 撤回出错: {e}", exc_info=True)
                        return False

                resend_stream_id = self.chat_stream.stream_id

                async def _do_resend(sid: str, reason: str) -> str:
                    try:
                        target_sid = sid or resend_stream_id
                        logger.info(f"{self.log_prefix} [撤回重发] 原因: {reason}")
                        
                        # 构建重发的内心想法，让模型自然地重新表达
                        thought_label = (
                            f"你刚才说的话被撤回了，原因是：{reason}。"
                            f"现在你想重新组织语言，用更自然、更符合你性格的方式表达刚才想说的意思。"
                            f"不要重复之前的话，要完全重新表达，但保持原本想表达的核心意思。"
                        )
                        
                        coordinator = getattr(self, "_reply_coordinator", None)
                        if coordinator:
                            await coordinator.execute_reply(
                                channel="active",
                                reply_func=self._execute_urgent_reply,
                                message=action_message,
                                thought=thought_label
                            )
                        else:
                            await self._execute_urgent_reply(
                                message=action_message,
                                thought=thought_label
                            )
                        
                        return "ok"
                    except Exception as resend_err:
                        logger.error(f"{self.log_prefix} [撤回重发] 失败: {resend_err}", exc_info=True)
                        return ""

                user_context = ""
                try:
                    user_msg = action_message.processed_plain_text if action_message else ""
                    if user_msg:
                        sender_name = person_name if person_name else "用户"
                        user_context = f"{sender_name}: {user_msg}"
                except Exception:
                    pass

                is_focus = self._heartflow_mode == "focus"
                is_private = not bool(self.chat_stream.group_info)
                await monitor.start_monitoring(
                    msg_id=msg_id,
                    content=reply_text,
                    stream_id=self.chat_stream.stream_id,
                    user_id=user_id,
                    review_weight=review_weight,
                    recall_func=_do_recall,
                    resend_func=_do_resend,
                    conversation_context=user_context,
                    urgent=is_focus,
                    is_private=is_private,
                )
            except Exception:
                pass

        if reply_text:
            try:
                from src.modules.modcore.decision_engine.sticker_decision import get_sticker_decision
                sticker_engine = get_sticker_decision()
                stream_id = self.chat_stream.stream_id
                user_id = str(action_message.user_info.user_id) if action_message.user_info else ""
                mood = "neutral"
                sticker_pre_check = True
                try:
                    from src.chat.heart_flow.skills.sticker_skill import get_sticker_skill
                    sticker_sk = get_sticker_skill()
                    cv = getattr(self, '_chat_value', 100.0)
                    scene_type = "group" if self.chat_stream.group_info else "private"
                    pre_ok, pre_reason = await sticker_sk.should_send_sticker(
                        dialogue_context=reply_text[:100], current_mood=mood,
                        scene_type=scene_type, user_id=user_id,
                        channel_id=stream_id, chat_value=cv,
                    )
                    sticker_pre_check = pre_ok
                    if not pre_ok:
                        logger.debug(f"[StickerSkill] 前置拒绝: {pre_reason}")
                except Exception:
                    pass
                try:
                    from src.modules.modcore.psychological_core import get_psychological_state
                    ps = get_psychological_state(stream_id, user_id)
                    annoyance = ps.get("annoyance", 0)
                    affection = ps.get("favor", 0)
                    if annoyance > 40:
                        mood = "irritated"
                    elif affection > 30:
                        mood = "happy"
                except Exception:
                    pass
                scene = "group" if self.chat_stream.group_info else "private"

                async def _try_send_sticker():
                    if not sticker_pre_check:
                        return
                    try:
                        should_send, path, caption, reason = await sticker_engine.decide_sticker_send(
                            reply_text, mood, scene, user_id, stream_id
                        )
                        if should_send and path:
                            logger.info(f"[表情包决策] 发送表情包: {reason}")
                    except Exception:
                        pass

                asyncio.create_task(_try_send_sticker())
            except Exception:
                pass

        # 构建循环信息
        loop_info: Dict[str, Any] = {
            "loop_plan_info": {
                "action_result": actions,
            },
            "loop_action_info": {
                "action_taken": True,
                "reply_text": reply_text,
                "command": "",
                "taken_time": time.time(),
            },
        }

        return loop_info, reply_text, cycle_timers



    async def _main_chat_loop(self):
        """主循环，持续进行计划并可能回复消息，直到被外部取消。"""
        try:
            while self.running:
                # 主循环
                success = await self._loopbody()
                await asyncio.sleep(0.1)
                if not success:
                    break
        except asyncio.CancelledError:
            # 设置了关闭标志位后被取消是正常流程
            logger.info(f"{self.log_prefix} 麦麦已关闭聊天")
        except Exception:
            logger.error(f"{self.log_prefix} 麦麦聊天意外错误，将于3s后尝试重新启动")
            print(traceback.format_exc())
            await asyncio.sleep(3)
            self._loop_task = asyncio.create_task(self._main_chat_loop())
        logger.error(f"{self.log_prefix} 结束了当前聊天循环")

    async def _handle_action(
        self,
        action: str,
        action_reasoning: str,
        action_data: dict,
        cycle_timers: Dict[str, float],
        thinking_id: str,
        action_message: Optional["DatabaseMessages"] = None,
    ) -> tuple[bool, str, str]:
        """
        处理规划动作，使用动作工厂创建相应的动作处理器

        参数:
            action: 动作类型
            action_reasoning: 决策理由
            action_data: 动作数据，包含不同动作需要的参数
            cycle_timers: 计时器字典
            thinking_id: 思考ID
            action_message: 消息数据
        返回:
            tuple[bool, str, str]: (是否执行了动作, 思考消息ID, 命令)
        """
        try:
            # 使用工厂创建动作处理器实例
            try:
                action_handler = self.action_manager.create_action(
                    action_name=action,
                    action_data=action_data,
                    cycle_timers=cycle_timers,
                    thinking_id=thinking_id,
                    chat_stream=self.chat_stream,
                    log_prefix=self.log_prefix,
                    action_reasoning=action_reasoning,
                    action_message=action_message,
                )
            except Exception as e:
                logger.error(f"{self.log_prefix} 创建动作处理器时出错: {e}")
                traceback.print_exc()
                return False, ""

            # 处理动作并获取结果（固定记录一次动作信息）
            result = await action_handler.execute()
            success, action_text = result

            return success, action_text

        except Exception as e:
            logger.error(f"{self.log_prefix} 处理{action}时出错: {e}")
            traceback.print_exc()
            return False, ""

    async def _send_response(
        self,
        reply_set: "ReplySetModel",
        message_data: "DatabaseMessages",
        selected_expressions: Optional[List[int]] = None,
        quote_message: Optional[bool] = None,
    ) -> str:
        # 根据 llm_quote 配置决定是否使用 quote_message 参数
        if global_config.chat.llm_quote:
            # 如果配置为 true，使用 llm_quote 参数决定是否引用回复
            if quote_message is None:
                logger.warning(f"{self.log_prefix} quote_message 参数为空，不引用")
                need_reply = False
            else:
                need_reply = quote_message
                if need_reply:
                    logger.info(f"{self.log_prefix} LLM 决定使用引用回复")
        else:
            # 如果配置为 false，使用原来的模式
            new_message_count = message_api.count_new_messages(
                chat_id=self.chat_stream.stream_id, start_time=self.last_read_time, end_time=time.time()
            )
            need_reply = new_message_count >= random.randint(2, 3) or time.time() - self.last_read_time > 90
            if need_reply:
                logger.debug(f"{self.log_prefix} 从思考到回复，原本应触发引用回复(新消息数:{new_message_count} 或超90s)")
        
        # 强制取消所有的系统引用机制，营造纯聊天的读空气氛围，不再显得像机器人
        need_reply = False

        reply_text = ""
        first_replied = False
        seg_count = 0
        for reply_content in reply_set.reply_data:
            if reply_content.content_type != ReplyContentType.TEXT:
                continue
            data: str = reply_content.content  # type: ignore
            if not data or not data.strip():
                logger.debug(f"{self.log_prefix} 跳过空白回复段")
                continue
            if not first_replied:
                await send_api.text_to_stream(
                    text=data,
                    stream_id=self.chat_stream.stream_id,
                    reply_message=message_data,
                    set_reply=need_reply,
                    typing=False,
                    selected_expressions=selected_expressions,
                )
                first_replied = True
            else:
                use_typing = random.random() < 0.6
                if use_typing:
                    gap = random.uniform(0.5, 2.5)
                    await asyncio.sleep(gap)
                await send_api.text_to_stream(
                    text=data,
                    stream_id=self.chat_stream.stream_id,
                    reply_message=message_data,
                    set_reply=False,
                    typing=use_typing,
                    selected_expressions=selected_expressions,
                )
            seg_count += 1
            reply_text += data

        return reply_text

    async def _try_admin_recall_user_msg(self, platform_msg_id: str, user_id: str, reason: str):
        try:
            from src.chat.message_receive.uni_message_sender import UniversalMessageSender
            from src.chat.message_receive.chat_stream import get_chat_manager
            from maim_message import Seg, UserInfo
            from src.config.config import global_config
            import time as _time
            target_stream = get_chat_manager().get_stream(self.chat_stream.stream_id)
            if not target_stream or not target_stream.group_info:
                return
            bot_user_info = UserInfo(
                user_id=global_config.bot.qq_account,
                user_nickname=global_config.bot.nickname,
                platform=target_stream.platform,
            )
            from src.chat.message_receive.message import MessageSending
            sender = UniversalMessageSender()
            cmd = {"action": "recall_message", "message_id": platform_msg_id, "timestamp": _time.time()}
            msg = MessageSending(
                message_id=f"admin_recall_{int(_time.time()*1000)}",
                chat_stream=target_stream, bot_user_info=bot_user_info,
                sender_info=target_stream.user_info,
                message_segment=Seg(type="command", data=cmd),
                display_message="", is_head=True, is_emoji=False,
                thinking_start_time=_time.time(),
            )
            result = await sender.send_message(msg, typing=False, storage_message=False)
            if result:
                logger.info(f"{self.log_prefix} [管理员撤回] 已撤回用户{user_id[:8]}的消息 | 原因: {reason}")
            else:
                logger.debug(f"{self.log_prefix} [管理员撤回] 撤回失败(可能不是管理员)")
        except Exception as e:
            logger.debug(f"{self.log_prefix} [管理员撤回] 异常: {e}")

    async def _execute_action(
        self,
        action_planner_info: ActionPlannerInfo,
        chosen_action_plan_infos: List[ActionPlannerInfo],
        thinking_id: str,
        available_actions: Dict[str, ActionInfo],
        cycle_timers: Dict[str, float],
    ):
        """执行单个动作的通用函数"""
        try:
            with Timer(f"动作{action_planner_info.action_type}", cycle_timers):
                # 直接当场执行no_reply逻辑
                if action_planner_info.action_type == "no_reply":
                    # 直接处理no_reply逻辑，不再通过动作系统
                    reason = action_planner_info.reasoning or "选择不回复"
                    # logger.info(f"{self.log_prefix} 选择不回复，原因: {reason}")

                    # 增加连续 no_reply 计数
                    self.consecutive_no_reply_count += 1

                    await database_api.store_action_info(
                        chat_stream=self.chat_stream,
                        action_build_into_prompt=False,
                        action_prompt_display=reason,
                        action_done=True,
                        thinking_id=thinking_id,
                        action_data={},
                        action_name="no_reply",
                        action_reasoning=reason,
                    )

                    return {"action_type": "no_reply", "success": True, "result": "选择不回复", "command": ""}

                elif action_planner_info.action_type == "reply":
                    # 直接当场执行reply逻辑
                    self.questioned = False
                    # 刷新主动发言状态
                    # 重置连续 no_reply 计数
                    self.consecutive_no_reply_count = 0

                    reason = action_planner_info.reasoning or ""
                    # 根据 think_mode 配置决定 think_level 的值
                    think_mode = global_config.chat.think_mode
                    if think_mode == "default":
                        think_level = 0
                    elif think_mode == "deep":
                        think_level = 1
                    elif think_mode == "dynamic":
                        # dynamic 模式：从 planner 返回的 action_data 中获取
                        think_level = action_planner_info.action_data.get("think_level", 1)
                    else:
                        # 默认使用 default 模式
                        think_level = 0
                    # 使用 action_reasoning（planner 的整体思考理由）作为 reply_reason
                    planner_reasoning = action_planner_info.action_reasoning or reason
                    focus_desire = getattr(self, '_focus_last_desire', 5)
                    focus_cv = getattr(self, '_chat_value', 100)
                    if focus_desire <= 4:
                        planner_reasoning += "\n[回复要求] 兴趣不高，简短回复，15字以内，1段即可"
                    elif focus_desire <= 6 or focus_cv < 40:
                        planner_reasoning += "\n[回复要求] 日常对话，20-30字，最多2段"
                    elif focus_desire <= 8:
                        planner_reasoning += "\n[回复要求] 有兴趣，30-50字，最多2-3段"
                    try:
                        from src.modules.modcore.psychological_core import get_psychological_core
                        psy_core = get_psychological_core()
                        if psy_core:
                            user_id_for_tone = str(getattr(action_planner_info.action_message, 'user_info', None) and action_planner_info.action_message.user_info.user_id or "")
                            if user_id_for_tone:
                                psy_s = psy_core.get_user_psychological_state(self.stream_id, user_id_for_tone)
                                if psy_s.get("trauma", 0) >= 5:
                                    planner_reasoning += "\n[语气约束] 即使生气也要克制，禁止说脏话、人身攻击、让人滚之类的话。可以冷淡、疏远、拒绝，但不能辱骂。"
                    except Exception:
                        pass

                    record_replyer_action_temp(
                        chat_id=self.stream_id,
                        reason=reason,
                        think_level=think_level,
                    )

                    await database_api.store_action_info(
                        chat_stream=self.chat_stream,
                        action_build_into_prompt=False,
                        action_prompt_display=reason,
                        action_done=True,
                        thinking_id=thinking_id,
                        action_data={},
                        action_name="reply",
                        action_reasoning=reason,
                    )

                    # 从 Planner 的 action_data 中提取未知词语列表（仅在 reply 时使用）
                    unknown_words = None
                    quote_message = None
                    if isinstance(action_planner_info.action_data, dict):
                        uw = action_planner_info.action_data.get("unknown_words")
                        if isinstance(uw, list):
                            cleaned_uw: List[str] = []
                            for item in uw:
                                if isinstance(item, str):
                                    s = item.strip()
                                    if s:
                                        cleaned_uw.append(s)
                            if cleaned_uw:
                                unknown_words = cleaned_uw
                        
                        # 从 Planner 的 action_data 中提取 quote_message 参数
                        qm = action_planner_info.action_data.get("quote")
                        if qm is not None:
                            # 支持多种格式：true/false, "true"/"false", 1/0
                            if isinstance(qm, bool):
                                quote_message = qm
                            elif isinstance(qm, str):
                                quote_message = qm.lower() in ("true", "1", "yes")
                            elif isinstance(qm, (int, float)):
                                quote_message = bool(qm)
                                
                        if qm is not None:
                            logger.info(f"{self.log_prefix} 引用回复设置: {quote_message}")

                    success, llm_response = await generator_api.generate_reply(
                        chat_stream=self.chat_stream,
                        reply_message=action_planner_info.action_message,
                        available_actions=available_actions,
                        chosen_actions=chosen_action_plan_infos,
                        reply_reason=planner_reasoning,
                        unknown_words=unknown_words,
                        enable_tool=global_config.tool.enable_tool,
                        enable_splitter=True,
                        request_type="replyer",
                        from_plugin=False,
                        reply_time_point=action_planner_info.action_data.get("loop_start_time", time.time()),
                        think_level=think_level,
                    )

                    if not success or not llm_response or not llm_response.reply_set:
                        if action_planner_info.action_message:
                            logger.info(f"对 {action_planner_info.action_message.processed_plain_text} 的回复生成失败")
                        else:
                            logger.info("回复生成失败")
                        return {"action_type": "reply", "success": False, "result": "回复生成失败", "loop_info": None}

                    response_set = llm_response.reply_set
                    selected_expressions = llm_response.selected_expressions
                    loop_info, reply_text, _ = await self._send_and_store_reply(
                        response_set=response_set,
                        action_message=action_planner_info.action_message,  # type: ignore
                        cycle_timers=cycle_timers,
                        thinking_id=thinking_id,
                        actions=chosen_action_plan_infos,
                        selected_expressions=selected_expressions,
                        quote_message=quote_message,
                    )
                    self.last_active_time = time.time()
                    self._last_reply_time = time.time()
                    self._consume_reply_resources()
                    return {
                        "action_type": "reply",
                        "success": True,
                        "result": f"你使用reply动作，对' {action_planner_info.action_message.processed_plain_text} '这句话进行了回复，回复内容为: '{reply_text}'",
                        "loop_info": loop_info,
                    }

                else:
                    # 执行普通动作
                    with Timer("动作执行", cycle_timers):
                        success, result = await self._handle_action(
                            action=action_planner_info.action_type,
                            action_reasoning=action_planner_info.action_reasoning or "",
                            action_data=action_planner_info.action_data or {},
                            cycle_timers=cycle_timers,
                            thinking_id=thinking_id,
                            action_message=action_planner_info.action_message,
                        )

                    self.last_active_time = time.time()
                    return {
                        "action_type": action_planner_info.action_type,
                        "success": success,
                        "result": result,
                    }

        except Exception as e:
            logger.error(f"{self.log_prefix} 执行动作时出错: {e}")
            logger.error(f"{self.log_prefix} 错误信息: {traceback.format_exc()}")
            return {
                "action_type": action_planner_info.action_type,
                "success": False,
                "result": "",
                "loop_info": None,
                "error": str(e),
            }

    def _load_persistent_state(self):
        """从持久化数据库加载疲劳和忽略状态，校准离线时间"""
        if not self._state_db:
            return
        try:
            fatigue_state = self._state_db.load_fatigue_state(self.stream_id)
            if fatigue_state:
                self._is_fatigued = fatigue_state.get("is_fatigued", False)
                self._fatigue_start_time = fatigue_state.get("fatigue_cooldown", 0.0)
                self._fatigue_duration = fatigue_state.get("fatigue_duration", 0.0)
                self._fatigue_reply_count = fatigue_state.get("fatigue_reply_count", 0)
                self._consecutive_replies = fatigue_state.get("consecutive_replies", 0)
                self._last_reply_time = fatigue_state.get("last_reply_time", 0.0)
                now = time.time()
                if self._is_fatigued and self._fatigue_start_time > 0:
                    elapsed = now - self._fatigue_start_time
                    remaining = max(0.0, self._fatigue_duration - elapsed)
                    if remaining <= 0:
                        self._is_fatigued = False
                        self._fatigue_duration = 0.0
                        self._fatigue_start_time = 0.0
                        logger.info(f"{self.log_prefix} 疲劳已自动恢复")
                    else:
                        minutes = int(remaining // 60)
                        seconds = int(remaining % 60)
                        logger.info(f"{self.log_prefix} 疲劳恢复中 剩余{minutes}分{seconds}秒")
            ignore_state = self._state_db.load_ignore_state(self.stream_id)
            if ignore_state:
                self._is_ignoring = ignore_state.get("is_ignoring", False)
                self._ignore_until_time = ignore_state.get("ignore_until_time", 0.0)
                now = time.time()
                if self._is_ignoring and self._ignore_until_time > now:
                    remaining = self._ignore_until_time - now
                    minutes = int(remaining // 60)
                    seconds = int(remaining % 60)
                    logger.info(f"{self.log_prefix} 忽略恢复中 剩余{minutes}分{seconds}秒")
                elif self._is_ignoring:
                    self._is_ignoring = False
                    self._ignore_until_time = 0.0
                    logger.info(f"{self.log_prefix} 忽略已自动恢复")
        except Exception as e:
            logger.warning(f"{self.log_prefix} 持久化状态加载失败: {e}")

    def _save_fatigue_state(self):
        """保存疲劳状态到持久化数据库"""
        if not self._state_db:
            return
        try:
            self._state_db.save_fatigue_state(
                stream_id=self.stream_id,
                is_fatigued=self._is_fatigued,
                fatigue_cooldown=self._fatigue_start_time,
                fatigue_duration=self._fatigue_duration,
                fatigue_reply_count=self._fatigue_reply_count,
                consecutive_replies=self._consecutive_replies,
                last_reply_time=self._last_reply_time,
            )
        except Exception as e:
            logger.warning(f"{self.log_prefix} 疲劳状态保存失败: {e}")

    def _save_ignore_state(self):
        """保存忽略状态到持久化数据库"""
        if not self._state_db:
            return
        try:
            self._state_db.save_ignore_state(
                stream_id=self.stream_id,
                is_ignoring=self._is_ignoring,
                ignore_until_time=self._ignore_until_time,
            )
        except Exception as e:
            logger.warning(f"{self.log_prefix} 忽略状态保存失败: {e}")

    def _update_reply_state(self, now: float, increment: bool = True):
        """更新回复计数并持久化保存"""
        self._last_reply_time = now
        if increment:
            self._consecutive_replies += 1
        else:
            self._consecutive_replies = 0
        self._save_fatigue_state()

    def _maybe_peek_and_think(self, recent_messages: List[Dict]) -> bool:
        """窥屏节流：短时间内不重复处理"""
        now = time.time()
        if now - self._last_peek_time < 10:
            return False
        self._last_peek_time = now
        return True

    def _is_quiet_hours(self, stream_type: str = "group") -> bool:
        """检查当前是否处于勿扰时段"""
        try:
            from datetime import datetime as _dt
            from src.config.config import global_config
            if stream_type == "private":
                start_hour = getattr(global_config, 'quiet_start_private', 22)
                end_hour = getattr(global_config, 'quiet_end_private', 9)
            else:
                start_hour = getattr(global_config, 'quiet_start', 23)
                end_hour = getattr(global_config, 'quiet_end', 7)
            current_hour = _dt.now().hour
            if start_hour > end_hour:
                return current_hour >= start_hour or current_hour < end_hour
            else:
                return start_hour <= current_hour < end_hour
        except Exception:
            return False

    def _calc_dynamic_fatigue_threshold(self) -> int:
        """根据时间段动态计算疲劳阈值"""
        base = random.randint(self._fatigue_threshold_min, self._fatigue_threshold_max)
        hour = time.localtime().tm_hour
        if 0 <= hour < 6:
            base = max(self._fatigue_threshold_min, base - 2)
        elif 22 <= hour <= 23:
            base = max(self._fatigue_threshold_min, base - 1)
        return base

    def _calculate_dynamic_fatigue_threshold(self, confidence_score: float) -> float:
        """根据置信度动态调整疲劳阈值"""
        base_threshold = self._fatigue_accumulated_threshold
        if confidence_score > 0.7:
            return base_threshold * 1.2
        elif confidence_score < 0.4:
            return base_threshold * 0.8
        return base_threshold

    async def _check_model_fatigue(self, last_response: str):
        """累积疲劳机制：回复增加疲劳值，空闲时衰减，达到阈值触发休息"""
        try:
            now = time.time()
            time_since_decay = now - self._last_fatigue_decay_time
            if time_since_decay > 1.0:
                decay_amount = time_since_decay * self._fatigue_decay_rate
                old_fatigue = self._accumulated_fatigue
                self._accumulated_fatigue = max(0.0, self._accumulated_fatigue - decay_amount)
                if old_fatigue > 0 and self._accumulated_fatigue < old_fatigue:
                    logger.debug(f"{self.log_prefix} 疲劳衰减 {old_fatigue:.0f}->{self._accumulated_fatigue:.0f}")
                self._last_fatigue_decay_time = now
            self._accumulated_fatigue += self._fatigue_per_reply
            logger.debug(f"{self.log_prefix} 疲劳+{self._fatigue_per_reply:.0f} 总计{self._accumulated_fatigue:.0f}")
            if self._accumulated_fatigue >= self._fatigue_accumulated_threshold:
                self._fatigue_duration = min(self._accumulated_fatigue, self._fatigue_max_duration)
                self._is_fatigued = True
                self._fatigue_start_time = now
                self._accumulated_fatigue = 0.0
                self._consecutive_reply_count = 0
                self._fatigue_reply_count = 0
                logger.info(f"{self.log_prefix} 疲劳触发 休息{int(self._fatigue_duration)}s")
        except Exception as e:
            logger.debug(f"{self.log_prefix} 疲劳检查错误: {e}")

    async def _evaluate_message_interest(self, recent_messages: List[Dict]) -> float:
        """使用LLM评估消息兴趣度，返回0.0-1.0评分"""
        if not recent_messages:
            return 0.0
        try:
            import re as _re
            latest = recent_messages[-1]
            user_message = latest.get("processed_plain_text") or latest.get("display_text", "")
            user_name = latest.get("sender_name") or latest.get("nickname", "用户")
            if not user_message or len(user_message.strip()) < 2:
                return 0.2
            context_lines = []
            for m in recent_messages[-5:]:
                txt = m.get("processed_plain_text") or m.get("display_text", "")
                name = m.get("sender_name") or m.get("nickname", "用户")
                if txt:
                    context_lines.append(f"{name}: {txt}")
            context_str = "\n".join(context_lines[-3:]) if context_lines else user_message
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "heart_flow",
                "interest_eval.template",
                context_str=context_str,
                user_name=user_name,
                user_message=user_message
            )
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            req = LLMRequest(model_config.lightweight, request_type="interest_eval")
            resp, _ = await req.generate_response_async(prompt, max_tokens=10)
            if resp:
                match = _re.search(r'(\d+\.?\d*)', resp.strip())
                if match:
                    return max(0.0, min(1.0, float(match.group(1))))
            return 0.5
        except Exception as e:
            logger.error(f"{self.log_prefix} 消息兴趣评分异常: {e}")
            return 0.5

    def _calc_engage_prob(self, recent_messages: List[Dict], is_admin: bool = False) -> float:
        """计算普通模式下的回复概率"""
        talk_factor = self._talk_value
        base_prob = self._engage_base_prob
        desire_mults = {"high": 1.5, "normal": 1.0, "low": 0.5, "silent": 0.1}
        desire_mult = desire_mults.get(self._desire_level, 1.0)
        admin_boost = self._admin_boost if is_admin else 1.0
        interest_boost = 0.0
        if recent_messages:
            for m in recent_messages[-3:]:
                raw = m.get("weight_score")
                if raw is not None:
                    try:
                        score = float(raw)
                        interest_boost += (score - 0.5) * self._interest_weight
                    except (TypeError, ValueError):
                        pass
        interest_factor = 1.0 + max(-0.3, min(0.5, interest_boost))
        consecutive_penalty = self._consecutive_reply_count * self._spam_penalty_factor
        now = time.time()
        since_reply = now - self._last_reply_time if self._last_reply_time > 0 else 60.0
        time_decay = min(1.0, since_reply / 30.0)
        fatigue_penalty = 0.0
        if self._is_fatigued:
            fatigue_penalty = self._fatigue_penalty
        elif self._accumulated_fatigue > 60:
            fatigue_penalty = self._accumulated_fatigue / 240.0
        combined = base_prob * 0.3 + talk_factor * 0.7
        final = combined * desire_mult * admin_boost * interest_factor * time_decay - fatigue_penalty
        final = max(0.01, min(0.99, final))
        return final

    def _check_and_switch_desire(self):
        """检查并切换欲望等级"""
        now = time.time()
        since_start = now - self._desire_start_time
        old_level = self._desire_level
        if since_start > 600 and self._desire_level == "low":
            self._desire_level = "normal"
            self._desire_start_time = now
        elif since_start > 1200 and self._desire_level == "normal":
            self._desire_level = "high"
            self._desire_start_time = now
        if self._mention_count >= 3:
            if self._desire_level in ("low", "normal"):
                self._desire_level = "high"
            self._mention_count = 0
            self._desire_start_time = now
        if self._consecutive_reply_count >= self._max_consecutive_replies:
            if self._desire_level == "high":
                self._desire_level = "normal"
            elif self._desire_level == "normal":
                self._desire_level = "low"
            self._desire_start_time = now
        if old_level != self._desire_level:
            logger.debug(f"{self.log_prefix} 欲望等级变化: {old_level} -> {self._desire_level}")

    def _detect_message_type(self, message_dict: dict) -> str:
        """检测消息类型：image/link/text"""
        if not message_dict:
            return "text"
        segments = message_dict.get("message_segment") or message_dict.get("segments", [])
        if isinstance(segments, list):
            for seg in segments:
                seg_type = seg.get("type", "") if isinstance(seg, dict) else getattr(seg, "type", "")
                if seg_type in ("image", "img"):
                    return "image"
                if seg_type in ("url", "link", "share"):
                    return "link"
        text = message_dict.get("processed_plain_text", "") or message_dict.get("display_text", "")
        if text:
            import re as _re
            if _re.search(r'https?://\S+', text):
                return "link"
            if "[图片]" in text or "[image]" in text.lower():
                return "image"
        return "text"

    def _build_conversation_context(self, recent_messages: List[Dict]) -> str:
        """构建对话上下文字符串"""
        lines = []
        for m in recent_messages[-10:]:
            name = m.get("sender_name") or m.get("nickname", "用户")
            text = m.get("processed_plain_text") or m.get("display_text", "")
            if text:
                lines.append(f"{name}: {text}")
        return "\n".join(lines)

    async def _evaluate_media_interest(self, user_message: str, user_name: str, media_type: str) -> bool:
        """使用LLM判断是否对媒体内容感兴趣"""
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "heart_flow",
                "media_interest.template",
                user_name=user_name,
                media_type=media_type,
                user_message=user_message[:100]
            )
            req = LLMRequest(model_config.utils, request_type="media_interest")
            resp, _ = await req.generate_response_async(prompt, max_tokens=10)
            if resp:
                answer = resp.strip().lower()
                return "是" in answer or "yes" in answer
            return False
        except Exception as e:
            logger.error(f"{self.log_prefix} {media_type}兴趣判断异常: {e}")
            return False

    async def _evaluate_image_interest_simple(self, user_message: str, user_name: str) -> bool:
        return await self._evaluate_media_interest(user_message, user_name, "图片")

    async def _evaluate_video_interest(self, user_message: str, user_name: str) -> bool:
        return await self._evaluate_media_interest(user_message, user_name, "视频")

    async def _evaluate_voice_interest(self, user_message: str, user_name: str) -> bool:
        return await self._evaluate_media_interest(user_message, user_name, "语音")

    async def _trigger_lazy_image_analysis(self, msg_text: str):
        import re
        picid_match = re.search(r'\[picid:([^\]]+)\]', msg_text)
        if not picid_match:
            return
        pic_id = picid_match.group(1)
        try:
            from src.common.database.database_model import Images
            record = Images.get_or_none(Images.image_id == pic_id)
            if not record:
                return
            if record.vlm_processed and record.description and len(record.description.strip()) > 10:
                logger.debug(f"{self.log_prefix} [识图] 图片已有描述，跳过: {record.description[:30]}")
                return
            img_path = record.path
            if not img_path:
                return
            import base64 as b64
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            img_b64 = b64.b64encode(img_bytes).decode('utf-8')
            from src.chat.utils.utils_image import get_image_manager
            mgr = get_image_manager()
            await mgr._process_image_with_vlm(pic_id, img_b64)
            logger.info(f"{self.log_prefix} [识图] VLM分析完成: picid={pic_id[:8]}")
        except Exception as e:
            logger.error(f"{self.log_prefix} [识图] 触发VLM分析失败: {e}")

    async def _analyze_pending_images(self, pending_images: Dict) -> Optional[str]:
        """识别待处理的图片"""
        try:
            image_data = pending_images.get('data', [])
            if not image_data:
                return None
            first_image = image_data[0]
            img_b64 = first_image.get('base64')
            if not img_b64:
                img_bytes = first_image.get('data')
                if img_bytes:
                    import base64
                    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                else:
                    img_url = first_image.get('url')
                    if img_url:
                        import aiohttp
                        import base64
                        async with aiohttp.ClientSession() as session:
                            async with session.get(img_url, ssl=False, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                                if resp.status == 200:
                                    content = await resp.read()
                                    img_b64 = base64.b64encode(content).decode('utf-8')
            if not img_b64:
                return None
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            req = LLMRequest(model_config.lightweight, request_type="image_describe")
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "heart_flow",
                "image_describe.template"
            )
            resp, _ = await req.generate_response_async(prompt, max_tokens=100)
            if resp:
                return resp.strip()
            return None
        except Exception as e:
            logger.error(f"{self.log_prefix} 图片识别异常: {e}")
            return None

    async def _execute_urgent_reply(self, message: Optional["DatabaseMessages"] = None, thought: str = "") -> bool:
        """提供给外部监控器(如model_driven_monitor)在遇到极其紧急的被艾特或必需回复时直接触发的函数"""
        if not message:
            return False
            
        logger.info(f"{self.log_prefix} [即时回复] 对 '{getattr(message, 'processed_plain_text', '')[:20]}' 生成回复")

        cycle_timers: Dict[str, float] = {}
        thinking_id = f"tid_urgent_{str(round(time.time(), 2))}"
        
        reply_reason = "有人在叫你或者说了跟你有关的话，你觉得应该回应一下"
        if thought:
            reply_reason += f"\n【你的内心想法/情绪】(请务必将以下情绪与内心想法转化为你即将说出口的语气，做到言行合一): {thought}"

        try:
            # 跳过冗长的ActionPlanner规划过程，直接走到LLM生成回复这一步
            success, llm_response = await generator_api.generate_reply(
                chat_stream=self.chat_stream,
                reply_message=message,
                available_actions={},
                chosen_actions=[],  # 因为是直接插队回复，没有经过多步规划动作池
                reply_reason=reply_reason,
                unknown_words=None,
                enable_tool=False,
                enable_splitter=True,
                request_type="replyer",
                from_plugin=False,
                reply_time_point=time.time(),
                think_level=0,
            )

            if not success or not llm_response or not llm_response.reply_set:
                logger.info(f"{self.log_prefix} [即时回复] 生成失败")
                return False

            # 分发并展示聊天表现
            response_set = llm_response.reply_set
            selected_expressions = getattr(llm_response, 'selected_expressions', None)
            
            loop_info, reply_text, _ = await self._send_and_store_reply(
                response_set=response_set,
                action_message=message,
                cycle_timers=cycle_timers,
                thinking_id=thinking_id,
                actions=[],
                selected_expressions=selected_expressions,
                quote_message=None,
            )
            
            self.last_active_time = time.time()
            self._last_reply_time = time.time()
            self.consecutive_no_reply_count = 0
            self._consume_reply_resources()
            msg_ui = getattr(message, 'user_info', None)
            reply_uid = str(msg_ui.user_id) if msg_ui else ""
            msg_content = getattr(message, 'processed_plain_text', '') or ''
            if reply_uid and msg_content:
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
                    tracker = get_emotion_tracker(self.stream_id)
                    await tracker.process_interaction(reply_uid, msg_content, use_llm=True)
                except Exception:
                    pass
            logger.info(f"{self.log_prefix} [即时回复] 完成")
            return True
            
        except Exception as e:
            logger.error(f"{self.log_prefix} [即时回复] 异常: {e}")
            traceback.print_exc()
            return False

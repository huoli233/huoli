import time
import random
from typing import Optional, Dict, Callable, Any
from dataclasses import dataclass
from enum import Enum

from src.common.logger import get_logger
from src.chat.heart_flow.config import HeartFlowState

logger = get_logger('心流状态')


@dataclass
class StateInfo:
    state: HeartFlowState
    start_time: float
    expected_duration: int
    reason: str
    thought: str


class HeartFlowStateManager:
    """
    心流状态管理器

    单层架构，负责：
    1. 存储当前状态
    2. 记录状态持续时间
    3. 执行模型决策的状态切换
    4. 状态到期检测

    状态切换由 model_driven_monitor 的模型决策驱动
    """

    def __init__(self, chat_id: str, heartfc_instance):
        self.chat_id = chat_id
        self.heartfc = heartfc_instance
        self.log_prefix = f"[状态层-{chat_id[-6:] if len(chat_id) > 6 else chat_id}]"

        self._current_state = HeartFlowState.OBSERVING
        self._state_info: Optional[StateInfo] = None
        self._last_state_change_time = 0.0

        self._stamina = 100.0
        self._stamina_max = 100.0
        self._last_stamina_update = time.time()

        self._state_changed_callbacks: list[Callable] = []
        self._peeking_msg_count = 0

        self._transition_to(HeartFlowState.OBSERVING, 120, "初始化", "开始读空气")

        logger.info(f"{self.log_prefix} 状态管理器初始化，当前状态: {self._current_state.value}")

    @property
    def current_state(self) -> HeartFlowState:
        return self._current_state

    @property
    def state_info(self) -> Optional[StateInfo]:
        return self._state_info

    @property
    def is_observing(self) -> bool:
        return self._current_state == HeartFlowState.OBSERVING

    @property
    def is_peeking(self) -> bool:
        return self._current_state == HeartFlowState.PEEKING

    @property
    def is_slacking(self) -> bool:
        return self._current_state == HeartFlowState.SLACKING

    @property
    def can_process_messages(self) -> bool:
        return self._current_state != HeartFlowState.SLACKING

    @property
    def can_reply(self) -> bool:
        if self._current_state == HeartFlowState.SLACKING:
            return False
        if self._current_state == HeartFlowState.PEEKING:
            return False
        return True

    @property
    def state_duration(self) -> float:
        if self._state_info:
            return time.time() - self._state_info.start_time
        return 0.0

    @property
    def remaining_duration(self) -> float:
        if self._state_info:
            elapsed = time.time() - self._state_info.start_time
            return max(0, self._state_info.expected_duration - elapsed)
        return 0.0

    def _update_stamina(self):
        now = time.time()
        elapsed = now - self._last_stamina_update

        if self._current_state == HeartFlowState.SLACKING:
            recovery_rate = 1.0
        elif self._current_state == HeartFlowState.PEEKING:
            recovery_rate = 0.5
        else:
            recovery_rate = 0.2

        self._stamina = min(self._stamina_max, self._stamina + elapsed * recovery_rate)
        self._last_stamina_update = now

    def consume_stamina(self, cost: float) -> bool:
        self._update_stamina()
        if self._stamina < cost:
            return False
        self._stamina -= cost
        return True

    def _transition_to(self, new_state: HeartFlowState, duration: int, reason: str, thought: str) -> bool:
        if new_state == self._current_state and self._state_info:
            self._state_info.expected_duration += duration
            return True
        if new_state == HeartFlowState.PEEKING:
            self._peeking_msg_count = 0

        old_state = self._current_state
        self._current_state = new_state
        self._last_state_change_time = time.time()

        self._state_info = StateInfo(
            state=new_state,
            start_time=time.time(),
            expected_duration=duration,
            reason=reason,
            thought=thought
        )

        logger.info(f"{self.log_prefix} 状态切换: {old_state.value} -> {new_state.value} | "
                   f"持续{duration}秒 | 原因: {reason} | 想法: {thought}")

        for callback in self._state_changed_callbacks:
            try:
                callback(old_state, new_state, reason, thought)
            except Exception:
                pass

        return True

    def force_terminate_state(self, reason: str) -> bool:
        """
        强制终止当前状态

        由心流层调用，当模型感觉不对时使用
        """
        if not self._state_info:
            return False

        logger.info(f"{self.log_prefix} 强制终止当前状态({self._current_state.value}): {reason}")

        # 将状态持续时间设为0，使其立即到期
        self._state_info.expected_duration = int(self.state_duration)

        return True

    def apply_model_decision(self, action: str, duration: int, reason: str, thought: str) -> bool:
        """
        应用模型决策的状态切换

        Args:
            action: 模型决策的动作 (观察/窥屏/摸鱼/待机)
            duration: 模型决定的持续时间(秒)
            reason: 切换原因
            thought: 内心想法

        Returns:
            是否切换成功
        """
        state_map = {
            "读空气": HeartFlowState.OBSERVING,
            "观察": HeartFlowState.OBSERVING,
            "默默读空气": HeartFlowState.PEEKING,
            "窥屏": HeartFlowState.PEEKING,
            "潜水休眠": HeartFlowState.SLACKING,
            "摸鱼": HeartFlowState.SLACKING,
            "休眠": HeartFlowState.SLACKING,
            "放下手机": HeartFlowState.SLACKING,
        }

        new_state = state_map.get(action)
        if not new_state:
            new_state = HeartFlowState.OBSERVING

        if duration <= 0:
            if new_state == HeartFlowState.OBSERVING:
                duration = random.randint(60, 600)
            elif new_state == HeartFlowState.PEEKING:
                duration = random.randint(30, 300)
            else:
                duration = random.randint(300, 1200)

        return self._transition_to(new_state, duration, reason, thought)

    def check_state_expiration(self) -> Optional[Dict]:
        """
        检查状态是否到期

        Returns:
            如果状态到期返回状态信息，否则返回 None
        """
        if not self._state_info:
            return None

        elapsed = time.time() - self._state_info.start_time
        if elapsed >= self._state_info.expected_duration:
            return {
                "state": self._current_state.value,
                "duration": int(elapsed),
                "thought": self._state_info.thought,
                "reason": self._state_info.reason,
            }
        return None

    def on_message_received(self, is_at_me: bool = False) -> Dict:
        self._update_stamina()
        interrupted = False
        if self._current_state == HeartFlowState.PEEKING:
            self._peeking_msg_count += 1
            should_interrupt = False
            if is_at_me:
                should_interrupt = True
            elif self._peeking_msg_count >= 10:
                should_interrupt = True
            else:
                interrupt_chance = self._peeking_msg_count * 0.1
                if random.random() < interrupt_chance:
                    should_interrupt = True
            if should_interrupt:
                reason = "被@唤醒" if is_at_me else f"累积{self._peeking_msg_count}条消息触发关注"
                logger.info(f"{self.log_prefix} 默默读空气被打断: {reason}")
                self._transition_to(HeartFlowState.OBSERVING, 60, f"默默读空气打断({reason})", "有人在聊天，该关注一下空气了")
                self._peeking_msg_count = 0
                interrupted = True
        return {
            "current_state": self._current_state.value,
            "can_reply": self.can_reply if not is_at_me else True,
            "can_process": self.can_process_messages,
            "stamina": self._stamina,
            "state_duration": self.state_duration,
            "remaining_duration": self.remaining_duration,
            "interrupted": interrupted,
        }

    def check_rest_trigger(self) -> bool:
        self._update_stamina()
        if self._current_state == HeartFlowState.SLACKING:
            return False
        rest_chance = 0.0
        if self._stamina <= 10:
            rest_chance = 0.8
        elif self._stamina <= 25:
            rest_chance = 0.4
        elif self._stamina <= 40:
            rest_chance = 0.15
        elif self._stamina <= 60:
            rest_chance = 0.05
        if rest_chance <= 0:
            return False
        if random.random() >= rest_chance:
            return False
        if self._stamina <= 10:
            duration = random.randint(600, 1800)
        elif self._stamina <= 25:
            duration = random.randint(180, 600)
        elif self._stamina <= 40:
            duration = random.randint(60, 300)
        else:
            duration = random.randint(60, 180)
        logger.info(f"{self.log_prefix} 脑力不足({self._stamina:.0f})，进入休息 {duration//60}分钟")
        self._transition_to(HeartFlowState.SLACKING, duration, f"脑力不足({self._stamina:.0f})", "累了，想休息一会儿")
        return True

    def get_status(self) -> Dict:
        """获取当前状态信息"""
        if self._state_info:
            return {
                "state": self._current_state.value,
                "state_duration": self.state_duration,
                "expected_duration": self._state_info.expected_duration,
                "remaining": self.remaining_duration,
                "stamina": self._stamina,
                "reason": self._state_info.reason,
                "thought": self._state_info.thought,
                "can_reply": self.can_reply,
                "can_process": self.can_process_messages,
            }
        return {
            "state": self._current_state.value,
            "stamina": self._stamina,
            "can_reply": self.can_reply,
            "can_process": self.can_process_messages,
        }

    def register_state_change_callback(self, callback: Callable):
        self._state_changed_callbacks.append(callback)

    def unregister_state_change_callback(self, callback: Callable):
        if callback in self._state_changed_callbacks:
            self._state_changed_callbacks.remove(callback)


_heart_state_managers: Dict[str, HeartFlowStateManager] = {}


def get_heart_state_manager(chat_id: str, heartfc_instance=None) -> Optional[HeartFlowStateManager]:
    if chat_id in _heart_state_managers:
        return _heart_state_managers[chat_id]

    if heartfc_instance is None:
        return None

    manager = HeartFlowStateManager(chat_id, heartfc_instance)
    _heart_state_managers[chat_id] = manager
    return manager


def remove_heart_state_manager(chat_id: str):
    if chat_id in _heart_state_managers:
        del _heart_state_managers[chat_id]


def get_all_managers() -> Dict[str, HeartFlowStateManager]:
    return _heart_state_managers.copy()

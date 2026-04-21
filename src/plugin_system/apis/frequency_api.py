from typing import Optional, Any
from src.common.logger import get_logger
from src.chat.heart_flow.frequency_control import frequency_control_manager
from src.config.config import global_config

logger = get_logger("频率API")


def get_current_talk_value(chat_id: str) -> float:
    try:
        fc = frequency_control_manager.get_or_create_frequency_control(chat_id)
        base_value = 1.0
        if hasattr(global_config, "chat") and hasattr(
            global_config.chat, "get_talk_value"
        ):
            base_value = global_config.chat.get_talk_value(chat_id)
        return fc.get_talk_frequency_adjust() * base_value
    except Exception as e:
        logger.error(f"[FrequencyAPI] 获取当前说话值失败: {e}")
        return 0.0


def set_talk_frequency_adjust(
    chat_id: str, talk_frequency_adjust: float
) -> None:
    try:
        fc = frequency_control_manager.get_or_create_frequency_control(chat_id)
        fc.set_talk_frequency_adjust(talk_frequency_adjust)
        logger.debug(
            f"[FrequencyAPI] 设置说话频率调整: {chat_id} -> {talk_frequency_adjust}"
        )
    except Exception as e:
        logger.error(f"[FrequencyAPI] 设置说话频率调整失败: {e}")


def get_talk_frequency_adjust(chat_id: str) -> float:
    try:
        fc = frequency_control_manager.get_or_create_frequency_control(chat_id)
        return fc.get_talk_frequency_adjust()
    except Exception as e:
        logger.error(f"[FrequencyAPI] 获取说话频率调整失败: {e}")
        return 1.0


def get_frequency_control(chat_id: str) -> Optional[Any]:
    try:
        return frequency_control_manager.get_or_create_frequency_control(
            chat_id
        )
    except Exception as e:
        logger.error(f"[FrequencyAPI] 获取频率控制失败: {e}")
        return None


def reset_frequency_control(chat_id: str) -> bool:
    try:
        fc = frequency_control_manager.get_or_create_frequency_control(chat_id)
        fc.reset()
        logger.debug(f"[FrequencyAPI] 重置频率控制: {chat_id}")
        return True
    except Exception as e:
        logger.error(f"[FrequencyAPI] 重置频率控制失败: {e}")
        return False


def update_last_reply_time(chat_id: str) -> None:
    try:
        fc = frequency_control_manager.get_or_create_frequency_control(chat_id)
        fc.update_last_reply_time()
    except Exception as e:
        logger.error(f"[FrequencyAPI] 更新最后回复时间失败: {e}")


def get_time_since_last_reply(chat_id: str) -> float:
    try:
        fc = frequency_control_manager.get_or_create_frequency_control(chat_id)
        return fc.get_time_since_last_reply()
    except Exception as e:
        logger.error(f"[FrequencyAPI] 获取距离上次回复的时间失败: {e}")
        return float("inf")


def should_reply(chat_id: str, threshold: float = 0.5) -> bool:
    try:
        current_value = get_current_talk_value(chat_id)
        return current_value >= threshold
    except Exception as e:
        logger.error(f"[FrequencyAPI] 检查是否应该回复失败: {e}")
        return False


def boost_frequency(chat_id: str, multiplier: float = 1.5) -> None:
    try:
        current = get_talk_frequency_adjust(chat_id)
        set_talk_frequency_adjust(chat_id, current * multiplier)
        logger.debug(
            f"[FrequencyAPI] 提升频率: {chat_id} -> {current * multiplier}"
        )
    except Exception as e:
        logger.error(f"[FrequencyAPI] 提升频率失败: {e}")


def reduce_frequency(chat_id: str, divisor: float = 2.0) -> None:
    try:
        current = get_talk_frequency_adjust(chat_id)
        set_talk_frequency_adjust(chat_id, current / divisor)
        logger.debug(
            f"[FrequencyAPI] 降低频率: {chat_id} -> {current / divisor}"
        )
    except Exception as e:
        logger.error(f"[FrequencyAPI] 降低频率失败: {e}")

from collections import deque
from typing import Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("EmojiHistory")

MAX_HISTORY_SIZE = 5

_history_cache: Dict[str, deque] = {}


def add_emoji_to_history(chat_id: str, emoji_description: str):
    if not chat_id or not emoji_description:
        return
    if chat_id not in _history_cache:
        _history_cache[chat_id] = deque(maxlen=MAX_HISTORY_SIZE)
    history = _history_cache[chat_id]
    history.append(emoji_description)
    logger.debug(f"已将表情 '{emoji_description}' 添加到聊天 {chat_id} 的内存历史中")


def get_recent_emojis(chat_id: str, limit: int = 5) -> List[str]:
    if not chat_id or chat_id not in _history_cache:
        return []
    history = _history_cache[chat_id]
    num_to_get = min(limit, len(history))
    recent_emojis = [history[-i] for i in range(1, num_to_get + 1)]
    logger.debug(f"为聊天 {chat_id} 从内存中获取到最近 {len(recent_emojis)} 个表情: {recent_emojis}")
    return recent_emojis


def clear_history(chat_id: str):
    if chat_id in _history_cache:
        del _history_cache[chat_id]


def get_all_history() -> Dict[str, List[str]]:
    result = {}
    for chat_id, history in _history_cache.items():
        result[chat_id] = list(history)
    return result

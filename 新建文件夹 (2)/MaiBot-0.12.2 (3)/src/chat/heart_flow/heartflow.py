import traceback
from typing import Any, Optional, Dict, List

from src.chat.message_receive.chat_stream import get_chat_manager
from src.common.logger import get_logger
from src.chat.heart_flow.heartFC_chat import HeartFChatting
from src.chat.brain_chat.brain_chat import BrainChatting
from src.chat.message_receive.chat_stream import ChatStream

logger = get_logger("heartflow")


class Heartflow:
    def __init__(self):
        self.heartflow_chat_list: Dict[Any, HeartFChatting | BrainChatting] = {}

    async def get_or_create_heartflow_chat(self, chat_id: Any) -> Optional[HeartFChatting | BrainChatting]:
        try:
            if chat_id in self.heartflow_chat_list:
                existing = self.heartflow_chat_list.get(chat_id)
                is_active = getattr(existing, '_active', False) if existing else False
                if existing and is_active:
                    return existing
                elif existing:
                    logger.info(f"心流不活跃，重建 | {str(chat_id)[:12]}...")
                    del self.heartflow_chat_list[chat_id]
            chat_stream: ChatStream | None = get_chat_manager().get_stream(chat_id)
            if not chat_stream:
                raise ValueError(f"未找到 chat_id={chat_id} 的聊天流")
            if chat_stream.group_info:
                new_chat = HeartFChatting(chat_id=chat_id)
            else:
                new_chat = BrainChatting(chat_id=chat_id)
            await new_chat.start()
            self.heartflow_chat_list[chat_id] = new_chat
            return new_chat
        except Exception as e:
            logger.error(f"创建心流聊天 {chat_id} 失败: {e}", exc_info=True)
            traceback.print_exc()
            return None

    def get_chat(self, chat_id: Any) -> Optional[HeartFChatting | BrainChatting]:
        return self.heartflow_chat_list.get(chat_id)

    def is_active(self, chat_id: Any) -> bool:
        chat = self.heartflow_chat_list.get(chat_id)
        if not chat:
            return False
        return getattr(chat, '_active', False)

    def list_active_chats(self) -> List[str]:
        return [str(cid) for cid in self.heartflow_chat_list.keys()]

    def remove_chat(self, chat_id: Any):
        if chat_id in self.heartflow_chat_list:
            del self.heartflow_chat_list[chat_id]

    def clear_all(self):
        self.heartflow_chat_list.clear()

    def get_stats(self) -> Dict[str, Any]:
        total = len(self.heartflow_chat_list)
        group_count = 0
        private_count = 0
        for chat in self.heartflow_chat_list.values():
            if isinstance(chat, HeartFChatting):
                group_count += 1
            elif isinstance(chat, BrainChatting):
                private_count += 1
        return {
            "total_chats": total,
            "group_chats": group_count,
            "private_chats": private_count,
        }


heartflow = Heartflow()

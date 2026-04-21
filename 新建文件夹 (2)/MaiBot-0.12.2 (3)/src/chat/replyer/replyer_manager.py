from typing import Dict, Optional, Any

from src.common.logger import get_logger
from src.chat.message_receive.chat_stream import ChatStream, get_chat_manager
from src.chat.replyer.group_generator import DefaultReplyer
from src.chat.replyer.private_generator import PrivateReplyer

logger = get_logger("ReplyerManager")


class ReplyerManager:
    def __init__(self):
        self._repliers: Dict[str, DefaultReplyer | PrivateReplyer] = {}

    def get_replyer(
        self,
        chat_stream: Optional[ChatStream] = None,
        chat_id: Optional[str] = None,
        request_type: str = "replyer",
    ) -> Optional[DefaultReplyer | PrivateReplyer]:
        stream_id = chat_stream.stream_id if chat_stream else chat_id
        if not stream_id:
            logger.warning("缺少 stream_id，无法获取回复器")
            return None
        if stream_id in self._repliers:
            cached = self._repliers[stream_id]
            if cached and hasattr(cached, 'generate_reply_with_context'):
                return cached
            logger.warning(f"缓存无效，重建回复器")
            del self._repliers[stream_id]
        target_stream = chat_stream
        if not target_stream:
            if chat_manager := get_chat_manager():
                target_stream = chat_manager.get_stream(stream_id)
        if not target_stream:
            logger.warning(f"未找到 stream_id='{stream_id}' 的聊天流")
            return None
        if target_stream.group_info:
            replyer = DefaultReplyer(
                chat_stream=target_stream,
                request_type=request_type,
            )
        else:
            replyer = PrivateReplyer(
                chat_stream=target_stream,
                request_type=request_type,
            )
        self._repliers[stream_id] = replyer
        return replyer

    def clear_cache(self, stream_id: Optional[str] = None):
        if stream_id:
            self._repliers.pop(stream_id, None)
        else:
            self._repliers.clear()

    async def generate_group_reply(
        self,
        stream_id: str,
        packet_data: Dict,
        is_priority: bool = False,
        think_level: int = 1,
        mental_state: Optional[Dict] = None,
        is_trauma_blocked: bool = False,
    ) -> str:
        replyer = self.get_replyer(chat_id=stream_id)
        if not replyer:
            return ""
        try:
            result = await replyer.generate_reply(
                packet_data,
                is_priority=is_priority,
                think_level=think_level,
                mental_state=mental_state,
                is_trauma_blocked=is_trauma_blocked,
            )
            if result:
                logger.info(f"群聊回复: {len(result)}字符")
                self._log_emotion_status(stream_id, packet_data)
            return result or ""
        except Exception as e:
            logger.error(f"群聊回复生成失败: {e}")
            self.clear_cache(stream_id)
            return ""

    async def generate_private_reply(
        self,
        stream_id: str,
        packet_data: Dict,
        think_level: int = 1,
        mental_state: Optional[Dict] = None,
    ) -> str:
        replyer = self.get_replyer(chat_id=stream_id)
        if not replyer:
            return ""
        try:
            result = await replyer.generate_reply(
                packet_data,
                think_level=think_level,
                mental_state=mental_state,
            )
            return result or ""
        except Exception as e:
            logger.error(f"私聊回复生成失败: {e}")
            self.clear_cache(stream_id)
            return ""

    def _log_emotion_status(self, stream_id: str, packet_data: Dict):
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            user_id = packet_data.get("sender_id") or packet_data.get("user_id", "")
            if not user_id:
                return
            tracker = get_emotion_tracker(stream_id)
            if not tracker:
                return
            state = tracker.get_user_state(user_id, create_if_missing=False)
            if not state:
                return
            status_parts = []
            affection = int(state.affection)
            if affection >= 70:
                status_parts.append(f"好感{affection}")
            elif affection >= 0:
                status_parts.append(f"好感{affection}")
            else:
                status_parts.append(f"好感{affection}")
            trust = int(state.trust_score)
            status_parts.append(f"信任{trust}")
            status_parts.append(f"{state.relationship}")
            annoyance = int(state.annoyance)
            if annoyance > 0:
                status_parts.append(f"烦恼{annoyance}")
            trauma = state.trauma_score
            if trauma > 0:
                status_parts.append(f"创伤{trauma:.1f}")
            if status_parts:
                logger.info(f"状态 {' '.join(status_parts)}")
        except Exception as e:
            logger.debug(f"状态显示失败: {e}")


replyer_manager = ReplyerManager()


def get_replyer_manager() -> ReplyerManager:
    return replyer_manager


class ResponseFactory:
    def __init__(self):
        self._responders: Dict[str, Any] = {}

    def acquire_responder(self, chat_stream=None, stream_id: str = "",
                           request_type: str = "replyer"):
        sid = stream_id or (chat_stream.stream_id if chat_stream else "")
        if not sid:
            return None
        if sid in self._responders:
            return self._responders[sid]
        try:
            from src.chat.message_receive.chat_stream import get_chat_manager
            cs = chat_stream or get_chat_manager().get_stream(sid)
            if not cs:
                return None
            if cs.group_info:
                from src.chat.replyer.group_generator import DefaultReplyer
                responder = DefaultReplyer(chat_stream=cs, request_type=request_type)
            else:
                from src.chat.replyer.private_generator import PrivateReplyer
                responder = PrivateReplyer(chat_stream=cs, request_type=request_type)
            self._responders[sid] = responder
            return responder
        except Exception as e:
            logger.error(f"创建响应器失败: {e}")
            return None

    async def fabricate_group_response(self, chat_stream, reply_message=None,
                                        reply_reason: str = "", **kwargs):
        responder = self.acquire_responder(chat_stream=chat_stream)
        if not responder:
            return False, None
        return await responder.generate_reply_with_context(
            reply_message=reply_message, reply_reason=reply_reason, **kwargs
        )

    async def fabricate_solo_response(self, chat_stream, reply_message=None,
                                       reply_reason: str = "", **kwargs):
        responder = self.acquire_responder(chat_stream=chat_stream)
        if not responder:
            return False, None
        return await responder.generate_reply_with_context(
            reply_message=reply_message, reply_reason=reply_reason, **kwargs
        )


_response_factory: Optional[ResponseFactory] = None


def get_response_factory() -> ResponseFactory:
    global _response_factory
    if _response_factory is None:
        _response_factory = ResponseFactory()
    return _response_factory


def reset_response_factory():
    global _response_factory
    _response_factory = None

from typing import List, Dict, Any, Optional
from enum import Enum
from src.common.logger import get_logger
from src.chat.message_receive.chat_stream import ChatStream, get_chat_manager

logger = get_logger("chat_api")


class SpecialTypes(Enum):
    ALL_PLATFORMS = "all_platforms"


class ChatManager:
    @staticmethod
    def get_all_streams(
        platform: Optional[str] | SpecialTypes = "qq",
    ) -> List[ChatStream]:
        if not isinstance(platform, (str, SpecialTypes)):
            raise TypeError("platform 必须是字符串或是 SpecialTypes 枚举")
        streams = []
        try:
            for _, stream in get_chat_manager().streams.items():
                if (
                    platform == SpecialTypes.ALL_PLATFORMS
                    or stream.platform == platform
                ):
                    streams.append(stream)
            logger.debug(
                f"[ChatAPI] 获取到 {len(streams)} 个 {platform} 平台的聊天流"
            )
        except Exception as e:
            logger.error(f"[ChatAPI] 获取聊天流失败: {e}")
        return streams

    @staticmethod
    def get_group_streams(
        platform: Optional[str] | SpecialTypes = "qq",
    ) -> List[ChatStream]:
        if not isinstance(platform, (str, SpecialTypes)):
            raise TypeError("platform 必须是字符串或是 SpecialTypes 枚举")
        streams = []
        try:
            for _, stream in get_chat_manager().streams.items():
                if (
                    platform == SpecialTypes.ALL_PLATFORMS
                    or stream.platform == platform
                ) and stream.group_info:
                    streams.append(stream)
            logger.debug(
                f"[ChatAPI] 获取到 {len(streams)} 个 {platform} 平台的群聊流"
            )
        except Exception as e:
            logger.error(f"[ChatAPI] 获取群聊流失败: {e}")
        return streams

    @staticmethod
    def get_private_streams(
        platform: Optional[str] | SpecialTypes = "qq",
    ) -> List[ChatStream]:
        if not isinstance(platform, (str, SpecialTypes)):
            raise TypeError("platform 必须是字符串或是 SpecialTypes 枚举")
        streams = []
        try:
            for _, stream in get_chat_manager().streams.items():
                if (
                    platform == SpecialTypes.ALL_PLATFORMS
                    or stream.platform == platform
                ) and not stream.group_info:
                    streams.append(stream)
            logger.debug(
                f"[ChatAPI] 获取到 {len(streams)} 个 {platform} 平台的私聊流"
            )
        except Exception as e:
            logger.error(f"[ChatAPI] 获取私聊流失败: {e}")
        return streams

    @staticmethod
    def get_group_stream_by_group_id(
        group_id: str, platform: Optional[str] | SpecialTypes = "qq"
    ) -> Optional[ChatStream]:
        if not isinstance(group_id, str):
            raise TypeError("group_id 必须是字符串类型")
        if not isinstance(platform, (str, SpecialTypes)):
            raise TypeError("platform 必须是字符串或是 SpecialTypes 枚举")
        if not group_id:
            raise ValueError("group_id 不能为空")
        try:
            for _, stream in get_chat_manager().streams.items():
                if (
                    stream.group_info
                    and str(stream.group_info.group_id) == str(group_id)
                    and (
                        platform == SpecialTypes.ALL_PLATFORMS
                        or stream.platform == platform
                    )
                ):
                    logger.debug(f"[ChatAPI] 找到群ID {group_id} 的聊天流")
                    return stream
            logger.warning(f"[ChatAPI] 未找到群ID {group_id} 的聊天流")
        except Exception as e:
            logger.error(f"[ChatAPI] 查找群聊流失败: {e}")
        return None

    @staticmethod
    def get_private_stream_by_user_id(
        user_id: str, platform: Optional[str] | SpecialTypes = "qq"
    ) -> Optional[ChatStream]:
        if not isinstance(user_id, str):
            raise TypeError("user_id 必须是字符串类型")
        if not isinstance(platform, (str, SpecialTypes)):
            raise TypeError("platform 必须是字符串或是 SpecialTypes 枚举")
        if not user_id:
            raise ValueError("user_id 不能为空")
        try:
            for _, stream in get_chat_manager().streams.items():
                if (
                    not stream.group_info
                    and str(stream.user_info.user_id) == str(user_id)
                    and (
                        platform == SpecialTypes.ALL_PLATFORMS
                        or stream.platform == platform
                    )
                ):
                    logger.debug(f"[ChatAPI] 找到用户ID {user_id} 的私聊流")
                    return stream
            logger.warning(f"[ChatAPI] 未找到用户ID {user_id} 的私聊流")
        except Exception as e:
            logger.error(f"[ChatAPI] 查找私聊流失败: {e}")
        return None

    @staticmethod
    def get_stream_type(chat_stream: ChatStream) -> str:
        if not isinstance(chat_stream, ChatStream):
            raise TypeError("chat_stream 必须是 ChatStream 类型")
        if not chat_stream:
            raise ValueError("chat_stream 不能为 None")
        if hasattr(chat_stream, "group_info"):
            return "group" if chat_stream.group_info else "private"
        return "unknown"

    @staticmethod
    def get_stream_info(chat_stream: ChatStream) -> Dict[str, Any]:
        if not chat_stream:
            raise ValueError("chat_stream 不能为 None")
        if not isinstance(chat_stream, ChatStream):
            raise TypeError("chat_stream 必须是 ChatStream 类型")
        try:
            info: Dict[str, Any] = {
                "stream_id": chat_stream.stream_id,
                "platform": chat_stream.platform,
                "type": ChatManager.get_stream_type(chat_stream),
            }
            if chat_stream.group_info:
                info.update(
                    {
                        "group_id": chat_stream.group_info.group_id,
                        "group_name": getattr(
                            chat_stream.group_info, "group_name", "未知群聊"
                        ),
                    }
                )
            if chat_stream.user_info:
                info.update(
                    {
                        "user_id": chat_stream.user_info.user_id,
                        "user_name": chat_stream.user_info.user_nickname,
                    }
                )
            return info
        except Exception as e:
            logger.error(f"[ChatAPI] 获取聊天流信息失败: {e}")
            return {}

    @staticmethod
    def get_streams_summary() -> Dict[str, int]:
        try:
            all_streams = ChatManager.get_all_streams(
                SpecialTypes.ALL_PLATFORMS
            )
            group_streams = ChatManager.get_group_streams(
                SpecialTypes.ALL_PLATFORMS
            )
            private_streams = ChatManager.get_private_streams(
                SpecialTypes.ALL_PLATFORMS
            )
            summary = {
                "total_streams": len(all_streams),
                "group_streams": len(group_streams),
                "private_streams": len(private_streams),
                "qq_streams": len(
                    [s for s in all_streams if s.platform == "qq"]
                ),
            }
            logger.debug(f"[ChatAPI] 聊天流统计: {summary}")
            return summary
        except Exception as e:
            logger.error(f"[ChatAPI] 获取聊天流统计失败: {e}")
            return {
                "total_streams": 0,
                "group_streams": 0,
                "private_streams": 0,
                "qq_streams": 0,
            }


def get_all_streams(
    platform: Optional[str] | SpecialTypes = "qq",
) -> List[ChatStream]:
    return ChatManager.get_all_streams(platform)


def get_group_streams(
    platform: Optional[str] | SpecialTypes = "qq",
) -> List[ChatStream]:
    return ChatManager.get_group_streams(platform)


def get_private_streams(
    platform: Optional[str] | SpecialTypes = "qq",
) -> List[ChatStream]:
    return ChatManager.get_private_streams(platform)


def get_stream_by_group_id(
    group_id: str, platform: Optional[str] | SpecialTypes = "qq"
) -> Optional[ChatStream]:
    return ChatManager.get_group_stream_by_group_id(group_id, platform)


def get_stream_by_user_id(
    user_id: str, platform: Optional[str] | SpecialTypes = "qq"
) -> Optional[ChatStream]:
    return ChatManager.get_private_stream_by_user_id(user_id, platform)


def get_stream_type(chat_stream: ChatStream) -> str:
    return ChatManager.get_stream_type(chat_stream)


def get_stream_info(chat_stream: ChatStream) -> Dict[str, Any]:
    return ChatManager.get_stream_info(chat_stream)


def get_streams_summary() -> Dict[str, int]:
    return ChatManager.get_streams_summary()

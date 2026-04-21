import asyncio
from typing import Optional, Any, List, Dict
from src.common.logger import get_logger

logger = get_logger("hook_trigger")


class HookTrigger:
    def __init__(self):
        self._enabled = True
        logger.debug("[HookTrigger] 钩子触发器已初始化")

    async def trigger_decorating_hooks(
        self, stream_id: str, content: str, platform: str = "qq",
    ) -> str:
        try:
            from src.plugin_system.core.events_manager import events_manager
            from src.plugin_system.base.component_types import EventType
            continue_flag, modified = await events_manager.handle_mai_events(
                EventType.POST_SEND_PRE_PROCESS, stream_id=stream_id
            )
            if not continue_flag:
                logger.debug(f"[HookTrigger] 消息被插件取消: {stream_id[:8]}")
                return ""
            if modified and hasattr(modified, '_modify_flags') and modified._modify_flags.modify_plain_text:
                return modified.plain_text
            return content
        except Exception as e:
            logger.debug(f"[HookTrigger] 钩子触发失败: {e}")
            return content

    async def process_proactive_message(
        self, stream_id: str, content: str, user_id: str = "",
    ) -> Optional[str]:
        try:
            processed = await self.trigger_decorating_hooks(stream_id, content)
            if not processed:
                return None
            return processed
        except Exception as e:
            logger.error(f"[HookTrigger] 处理主动消息失败: {e}")
            return content

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled


_hook_trigger: Optional[HookTrigger] = None


def get_hook_trigger() -> HookTrigger:
    global _hook_trigger
    if _hook_trigger is None:
        _hook_trigger = HookTrigger()
    return _hook_trigger


def init_hook_trigger(context: Any = None) -> HookTrigger:
    global _hook_trigger
    _hook_trigger = HookTrigger()
    logger.info("[HookTrigger] 已初始化")
    return _hook_trigger

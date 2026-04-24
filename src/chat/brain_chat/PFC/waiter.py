from src.common.logger import get_logger
from .chat_observer import ChatObserver
from .conversation_info import ConversationInfo
from src.chat.brain_chat.runtime_config import brainchat_module_view

# from src.individuality.individuality import Individuality # 不再需要
from src.config.config import global_config
import time
import asyncio
from typing import Dict

logger = get_logger("等待器")


class Waiter:
    """等待处理类"""

    def __init__(self, stream_id: str, private_name: str):
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)
        self.name = (
            getattr(global_config, "BOT_NICKNAME", None)
            or getattr(global_config.bot, "nickname", "")
        )
        self.private_name = private_name
        self._stream_id = stream_id
        self._load_config()

    def _load_config(self) -> None:
        config = brainchat_module_view("brain_waiter")
        self._desired_timeout_seconds = int(
            config.get("desired_timeout_seconds", 300)
        )
        self._listening_timeout_seconds = int(
            config.get("listening_timeout_seconds", self._desired_timeout_seconds)
        )
        self._fast_retry_timeout_seconds = int(
            config.get("fast_retry_timeout_seconds", 120)
        )
        self._goal_list_fast_retry_threshold = int(
            config.get("goal_list_fast_retry_threshold", 4)
        )
        self._wait_poll_interval_seconds = float(
            config.get("wait_poll_interval_seconds", 5.0)
        )

    def _pick_timeout_seconds(
        self, listening_mode: bool, conversation_info: ConversationInfo
    ) -> int:
        if (
            conversation_info.goal_list
            and len(conversation_info.goal_list)
            >= self._goal_list_fast_retry_threshold
        ):
            return self._fast_retry_timeout_seconds

        if listening_mode:
            return self._listening_timeout_seconds
        return self._desired_timeout_seconds

    @staticmethod
    def _has_similar_goal(
        conversation_info: ConversationInfo, candidate_goal: str
    ) -> bool:
        normalized_candidate = candidate_goal.strip()
        if not normalized_candidate:
            return True

        for item in conversation_info.goal_list:
            if not isinstance(item, dict):
                continue
            existing_goal = str(item.get("goal", "")).strip()
            if not existing_goal:
                continue
            if existing_goal == normalized_candidate:
                return True
            if (
                normalized_candidate in existing_goal
                or existing_goal in normalized_candidate
            ):
                return True
        return False

    @staticmethod
    def _build_timeout_goal(
        elapsed_seconds: float, listening_mode: bool
    ) -> Dict[str, str]:
        elapsed_minutes = elapsed_seconds / 60
        if listening_mode:
            return {
                "goal": (
                    f"你等待了{elapsed_minutes:.1f}分钟，对方像是暂时离开了，"
                    "可以考虑继续等待、轻量追问，或体面结束当前对话"
                ),
                "reasoning": "倾听阶段超时，对方尚未继续表达",
            }
        return {
            "goal": f"你等待了{elapsed_minutes:.1f}分钟，对方暂未回应，思考是继续推进还是改为等待",
            "reasoning": "常规等待超时，需要重新判断互动节奏",
        }

    async def _record_wait_runtime_state(
        self, state_name: str, elapsed_seconds: float = 0.0
    ) -> None:
        try:
            from src.chat.utils.statistic import record_runtime_state_event

            thinking_score = min(
                max(
                    elapsed_seconds / max(self._desired_timeout_seconds, 1),
                    0.0,
                ),
                1.0,
            )
            record_runtime_state_event(
                channel_id=self._stream_id,
                mode=state_name,
                thinking_score=thinking_score,
            )
        except Exception:
            logger.debug(f"[私聊][{self.private_name}]记录等待状态失败")

    async def _wait_core(
        self, conversation_info: ConversationInfo, listening_mode: bool
    ) -> bool:
        wait_start_time = time.time()
        timeout_seconds = self._pick_timeout_seconds(
            listening_mode, conversation_info
        )
        state_label = "listening_wait" if listening_mode else "normal_wait"
        status_name = "倾听等待" if listening_mode else "常规等待"

        logger.info(
            f"[私聊][{self.private_name}]进入{status_name}状态 (超时: {timeout_seconds} 秒)..."
        )
        await self._record_wait_runtime_state(f"{state_label}_start")

        while True:
            if self.chat_observer.new_message_after(wait_start_time):
                elapsed_time = time.time() - wait_start_time
                logger.info(
                    f"[私聊][{self.private_name}]{status_name}结束，收到新消息"
                )
                await self._record_wait_runtime_state(
                    f"{state_label}_interrupt", elapsed_time
                )
                return False

            elapsed_time = time.time() - wait_start_time
            if elapsed_time > timeout_seconds:
                timeout_goal = self._build_timeout_goal(
                    elapsed_time, listening_mode
                )
                if not self._has_similar_goal(
                    conversation_info, timeout_goal["goal"]
                ):
                    conversation_info.goal_list.append(timeout_goal)
                    logger.info(
                        f"[私聊][{self.private_name}]添加目标: {timeout_goal}"
                    )
                else:
                    logger.debug(
                        f"[私聊][{self.private_name}]超时目标已存在，跳过重复追加"
                    )

                await self._record_wait_runtime_state(
                    f"{state_label}_timeout", elapsed_time
                )
                return True

            await asyncio.sleep(self._wait_poll_interval_seconds)
            logger.debug(f"[私聊][{self.private_name}]{status_name}中...")

    async def wait(self, conversation_info: ConversationInfo) -> bool:
        """等待用户新消息或超时"""
        return await self._wait_core(conversation_info, listening_mode=False)

    async def wait_listening(
        self, conversation_info: ConversationInfo
    ) -> bool:
        """倾听用户发言或超时"""
        return await self._wait_core(conversation_info, listening_mode=True)

from src.common.logger import get_logger
from .chat_observer import ChatObserver
from .conversation_info import ConversationInfo

# from src.individuality.individuality import Individuality # 不再需要
from src.config.config import global_config
import time
import asyncio
from typing import Any, Dict, Tuple

logger = get_logger("等待器")

# --- 在这里设定你想要的超时时间（秒） ---
# 例如： 120 秒 = 2 分钟
DESIRED_TIMEOUT_SECONDS = 300

WAIT_PROFILE_DEFAULTS: Dict[str, int] = {
    "normal": DESIRED_TIMEOUT_SECONDS,
    "listening": DESIRED_TIMEOUT_SECONDS,
    "fast_retry": 120,
}


class Waiter:
    """等待处理类"""

    def __init__(self, stream_id: str, private_name: str):
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)
        self.name = global_config.BOT_NICKNAME
        self.private_name = private_name
        self._stream_id = stream_id

    def _pick_timeout_seconds(
        self, listening_mode: bool, conversation_info: ConversationInfo
    ) -> int:
        configured = getattr(global_config, "wait_timeout_seconds", None)
        if isinstance(configured, int) and configured > 0:
            return configured

        if (
            conversation_info.goal_list
            and len(conversation_info.goal_list) >= 4
        ):
            return WAIT_PROFILE_DEFAULTS["fast_retry"]

        profile_name = "listening" if listening_mode else "normal"
        return WAIT_PROFILE_DEFAULTS.get(profile_name, DESIRED_TIMEOUT_SECONDS)

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
                max(elapsed_seconds / max(DESIRED_TIMEOUT_SECONDS, 1), 0.0),
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

            await asyncio.sleep(5)
            logger.debug(f"[私聊][{self.private_name}]{status_name}中...")

    async def wait(self, conversation_info: ConversationInfo) -> bool:
        """等待用户新消息或超时"""
        return await self._wait_core(conversation_info, listening_mode=False)

    async def wait_listening(
        self, conversation_info: ConversationInfo
    ) -> bool:
        """倾听用户发言或超时"""
        return await self._wait_core(conversation_info, listening_mode=True)

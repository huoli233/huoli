import asyncio
import random
import time
from typing import Dict, Any, Optional, Callable, Awaitable, Tuple
from src.common.logger import get_logger

logger = get_logger("recall_skill")


class FocusRecallSkill:
    def __init__(self):
        self._review_delay_min = 1
        self._review_delay_max = 115

    def _get_controller(self):
        try:
            from src.modules.modcore.recall_system.recall_controller import get_recall_controller
            return get_recall_controller()
        except Exception:
            return None

    def _get_adaptive(self):
        try:
            from src.modules.modcore.recall_system.adaptive_recall_integrator import get_adaptive_recall_integrator
            return get_adaptive_recall_integrator()
        except Exception:
            return None

    def should_trigger_with_psychology(
        self,
        surface_mask: float = 10.0,
        inner_chaos: float = 0.0,
        mental_fatigue: float = 0.0,
        trauma_score: float = 0.0,
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        adaptive = self._get_adaptive()
        if adaptive:
            return adaptive.should_trigger_recall(surface_mask, inner_chaos, mental_fatigue, trauma_score)
        controller = self._get_controller()
        if controller:
            return (*controller.should_trigger(), {})
        return False, None, {}

    async def process_with_recall(
        self,
        original_text: str,
        group_id: str,
        stream_id: str,
        send_func: Callable[[str, str], Awaitable[str]],
        delete_func: Callable[[str], Awaitable[bool]],
        surface_mask: float = 10.0,
        inner_chaos: float = 0.0,
        mental_fatigue: float = 0.0,
        trauma_score: float = 0.0,
    ) -> Tuple[str, bool]:
        adaptive = self._get_adaptive()
        if adaptive:
            text, recalled, _ = await adaptive.process_with_adaptive_recall(
                original_text=original_text,
                surface_mask=surface_mask,
                inner_chaos=inner_chaos,
                mental_fatigue=mental_fatigue,
                trauma_score=trauma_score,
                group_id=group_id,
                stream_id=stream_id,
                send_func=send_func,
                delete_func=delete_func,
            )
            return text, recalled
        controller = self._get_controller()
        if controller:
            should, error_type = controller.should_trigger()
            if not should or not error_type:
                return original_text, False
            error_text, _ = await controller.generate_error_text(original_text, error_type, stream_id)
            msg_id = await send_func(group_id, error_text)
            await controller.schedule_recall(
                msg_id=msg_id, group_id=group_id, original_text=original_text,
                delete_func=delete_func, send_func=send_func,
                stream_id=stream_id, error_text=error_text, error_type=error_type,
            )
            return error_text, True
        return original_text, False

    async def start_post_send_review(
        self,
        msg_id: str,
        channel_id: str,
        content: str,
        delete_func: Callable[[str], Awaitable[bool]],
        send_func: Optional[Callable[[str, str], Awaitable[str]]] = None,
    ):
        delay = random.uniform(self._review_delay_min, self._review_delay_max)
        async def _review_once():
            try:
                await asyncio.sleep(delay)
                should_recall, reason = await self._review_sent_message(content)
                if should_recall:
                    logger.info(f"[专注模式审查] {delay:.1f}s后决定撤回: {reason}")
                    success = await delete_func(msg_id)
                    if success and send_func:
                        new_content = await self._regenerate_content(content, reason)
                        if new_content:
                            await asyncio.sleep(random.uniform(1.0, 4.0))
                            await send_func(channel_id, new_content)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[专注模式审查] 异常: {e}")
        asyncio.create_task(_review_once())

    async def _review_sent_message(self, content: str) -> Tuple[bool, str]:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "heart_flow",
                "focus_review.template",
                content=content
            )
            request = LLMRequest(model_config.lightweight, request_type="focus_review")
            response, _ = await request.generate_response_async(prompt)
            if not response:
                return False, ""
            text = response.strip()
            should = "是" in text.split('\n')[0]
            reason = ""
            for line in text.split('\n'):
                if line.startswith("理由"):
                    reason = line.split("：", 1)[-1].strip()
                    break
            return should, reason
        except Exception:
            return False, ""

    async def _regenerate_content(self, original: str, reason: str) -> Optional[str]:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            import re
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "heart_flow",
                "focus_regenerate.template",
                original=original[:120],
                reason=reason
            )
            request = LLMRequest(model_config.lightweight, request_type="focus_regenerate")
            response, _ = await request.generate_response_async(prompt, max_tokens=60, temperature=0.7)
            if response:
                text = response.strip()
                think_match = re.search(r'</think>\s*(.+)', text, re.DOTALL)
                if think_match:
                    text = think_match.group(1).strip()
                if len(text) > 50:
                    text = text[:50]
                return text
            return None
        except Exception:
            return None


_focus_recall_instance: Optional[FocusRecallSkill] = None


def get_focus_recall_skill() -> FocusRecallSkill:
    global _focus_recall_instance
    if _focus_recall_instance is None:
        _focus_recall_instance = FocusRecallSkill()
    return _focus_recall_instance

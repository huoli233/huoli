import asyncio
import random
import re
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from src.common.logger import get_logger
from src.common.task_utils import safe_create_task

logger = get_logger("focus_patrol")


class FocusPatrol:
    """专注模式的消息巡查与召回技能"""

    _REVIEW_DELAY_LO: float = 1.0
    _REVIEW_DELAY_HI: float = 115.0

    def __init__(self):
        self._post_send_analyzer_ref = None
        self._typo_generator_ref = None

    def _lazy_post_send_analyzer(self):
        """懒加载发送后分析器"""
        if self._post_send_analyzer_ref is None:
            try:
                from src.modules.recall.post_send_analyzer import (
                    get_post_send_analyzer,
                )

                self._post_send_analyzer_ref = get_post_send_analyzer()
            except Exception:
                self._post_send_analyzer_ref = False
        return self._post_send_analyzer_ref if self._post_send_analyzer_ref is not False else None

    def _lazy_typo_generator(self):
        """懒加载打错字生成器"""
        if self._typo_generator_ref is None:
            try:
                from src.modules.recall.typo_generator import (
                    get_typo_generator,
                )

                self._typo_generator_ref = get_typo_generator()
            except Exception:
                self._typo_generator_ref = False
        return self._typo_generator_ref if self._typo_generator_ref is not False else None

    def should_trigger_with_psychology(
        self,
        surface_mask: float = 10.0,
        inner_chaos: float = 0.0,
        trauma_score: float = 0.0,
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """综合心理指标判断是否应触发召回机制

        返回 (是否触发, 错误类型, 附加信息)。
        """
        analyzer = self._lazy_post_send_analyzer()
        if analyzer:
            try:
                decision = analyzer.analyze_recall_need(
                    surface_mask=surface_mask,
                    inner_chaos=inner_chaos,
                    trauma_score=trauma_score,
                )
                if decision and decision.should_recall:
                    return (
                        True,
                        (decision.recall_reason.value if decision.recall_reason else "unknown"),
                        {},
                    )
            except Exception as e:
                logger.debug(f"分析召回需求失败: {e}")
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
        trauma_score: float = 0.0,
    ) -> Tuple[str, bool]:
        """在发送前走召回分支，返回 (最终文本, 是否触发了召回)"""
        should, err_kind, _ = self.should_trigger_with_psychology(surface_mask, inner_chaos, trauma_score)
        if not should:
            return original_text, False

        typo_gen = self._lazy_typo_generator()
        if typo_gen:
            try:
                from src.modules.recall.typo_generator import TypoType

                decision = await typo_gen.decide(
                    original_content=original_text,
                    user_id="",
                    channel_id=group_id,
                    stream_id=stream_id,
                )
                if decision.should_typo and decision.error_content:
                    err_text = decision.error_content
                else:
                    err_text = typo_gen._simple_typo(original_text, TypoType.TYPO)
            except Exception as e:
                logger.debug(f"生成错误文本失败: {e}")
                err_text = original_text
        else:
            err_text = original_text

        msg_id = await send_func(group_id, err_text)

        analyzer = self._lazy_post_send_analyzer()
        if analyzer:
            try:
                await analyzer.schedule_recall(
                    msg_id=msg_id,
                    channel_id=group_id,
                    stream_id=stream_id,
                    sent_content=err_text,
                    original_content=original_text,
                    delete_func=delete_func,
                    send_func=send_func,
                )
            except Exception as e:
                logger.debug(f"调度撤回失败: {e}")

        return err_text, True

    async def launch_post_send_audit(
        self,
        msg_id: str,
        channel_id: str,
        content: str,
        delete_func: Callable[[str], Awaitable[bool]],
        send_func: Optional[Callable[[str, str], Awaitable[str]]] = None,
    ):
        """在后台延迟审阅已发送的消息，必要时撤回并重写"""
        delay = random.uniform(self._REVIEW_DELAY_LO, self._REVIEW_DELAY_HI)

        async def _audit_task():
            try:
                await asyncio.sleep(delay)
                should_retract, reason = await self._audit_content(content)
                if not should_retract:
                    return
                logger.info(f"[专注巡查] {delay:.1f}s后决定撤回: {reason}")
                ok = await delete_func(msg_id)
                if ok and send_func:
                    replacement = await self._rewrite_content(content, reason)
                    if replacement:
                        await asyncio.sleep(random.uniform(1.0, 4.0))
                        await send_func(channel_id, replacement)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error(f"[专注巡查] 审查异常: {exc}")

        safe_create_task(_audit_task(), name="focus_audit")

    async def audit_high_risk_reply(self, content: str, risk_note: str = "") -> Tuple[bool, str]:
        """对高风险回复做即时发送后审查，不等待长延迟。"""
        try:
            should_retract, reason = await self._audit_content(
                f"{content}\n[风险提示]{risk_note[:120]}" if risk_note else content
            )
            return should_retract, reason
        except Exception as exc:
            logger.warning(f"高风险回复审查异常: {exc}")
            return False, ""

    async def review_before_send(self, content: str, risk_note: str = "") -> Tuple[str, bool, str]:
        """发送前审查并在必要时重写，避免高风险内容先发后悔。"""
        original = str(content or "").strip()
        if not original:
            return "", False, ""
        should_retract, reason = await self.audit_high_risk_reply(original, risk_note=risk_note)
        if not should_retract:
            return original, False, ""
        replacement = await self._rewrite_content(original, reason)
        replacement = str(replacement or "").strip()
        if not replacement:
            replacement = "我换个说法，慢慢说。"
        if replacement == original:
            replacement = "我换个说法。"
        return replacement[:50], True, reason

    async def _audit_content(self, content: str) -> Tuple[bool, str]:
        """用 LLM 审阅内容，判断是否应该撤回"""
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            stimulus = (
                "审阅以下已发送的消息，判断是否需要撤回。\n"
                "第一行回答'是'或'否'，第二行以'理由：'开头写明原因。\n"
                f"消息内容: {content}"
            )
            req = LLMRequest(model_config.model_task_config.utils, request_type="focus_audit")
            raw, _ = await req.generate_response_async(stimulus)
            if not raw:
                return False, ""
            lines = raw.strip().split("\n")
            should = "是" in lines[0]
            reason = ""
            for ln in lines:
                if ln.startswith("理由"):
                    reason = ln.split("：", 1)[-1].strip()
                    break
            return should, reason
        except Exception as _exc:
            logger.warning(f"撤回判断异常: {_exc}")
            return False, ""

    async def _rewrite_content(self, original: str, reason: str) -> Optional[str]:
        """撤回后用 LLM 重写内容"""
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            stimulus = f"请根据撤回理由重写以下消息（50字以内）。\n原消息: {original[:120]}\n撤回理由: {reason}"
            req = LLMRequest(model_config.model_task_config.utils, request_type="focus_rewrite")
            raw, _ = await req.generate_response_async(stimulus, temperature=0.7)
            if not raw:
                return None
            cleaned = raw.strip()
            think_hit = re.search(r"```(?:json)?\s*(.+?)\s*```", cleaned, re.DOTALL)
            if think_hit:
                cleaned = think_hit.group(1).strip()
            return cleaned[:50] if cleaned else None
        except Exception as _exc:
            logger.warning(f"内容重写异常: {_exc}")
            return None


_patrol: Optional[FocusPatrol] = None


def acquire_focus_patrol() -> FocusPatrol:
    """获取全局 FocusPatrol 单例"""
    global _patrol
    if _patrol is None:
        _patrol = FocusPatrol()
    return _patrol

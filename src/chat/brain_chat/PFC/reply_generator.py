import asyncio
from typing import Any, Dict, List, Tuple

from src.chat.brain_chat.PFC.chat_observer import ChatObserver
from src.chat.brain_chat.PFC.conversation_info import ConversationInfo
from src.chat.brain_chat.PFC.observation_info import ObservationInfo
from src.chat.brain_chat.PFC.reply_checker import ReplyChecker
from src.chat.brain_chat.runtime_config import brainchat_module_view
from src.chat.utils.chat_message_builder import build_readable_messages
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.llm_models.utils_model import LLMRequest
from src.person_info.bot_identity import get_bot_identity_manager

import src.chat.prompts.catalog  # noqa: F401 注册提示词
from src.chat.utils.prompt_builder import global_prompt_manager

logger = get_logger("回复生成")


class ReplyGenerator:
    """回复文本生成器 —— 根据行动类型与对话上下文调用LLM生成回复并做质量后处理。"""

    def __init__(self, stream_id: str, private_name: str):
        self.llm = LLMRequest(
            model_set=model_config.model_task_config.replyer,
            request_type="reply_generation",
        )
        self.bot_name, self.personality_info = self._compose_persona_trait()
        self.private_name = private_name
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)
        self.reply_checker = ReplyChecker(stream_id, private_name)
        self._load_config()

    def _load_config(self) -> None:
        config = brainchat_module_view("brain_pfc_reply")
        self._response_length_soft_cap = int(
            config.get("response_length_soft_cap", 200)
        )
        self._max_generation_attempts = int(
            config.get("max_generation_attempts", 2)
        )
        self._generation_timeout_seconds = float(
            config.get("generation_timeout_seconds", 60.0)
        )
        self._generation_retry_backoff_seconds = float(
            config.get("generation_retry_backoff_seconds", 0.35)
        )
        self._force_replan_retry_count = int(
            config.get("force_replan_retry_count", 2)
        )

    @staticmethod
    def _compose_persona_trait() -> Tuple[str, str]:
        """从统一人格源构造私聊回复阶段使用的人设描述。"""
        identity = get_bot_identity_manager()
        bot_name = identity.get_display_name() or global_config.bot.nickname
        trait = identity.build_persona_brief(
            include_name_prefix=False,
            include_style=True,
            include_interests=True,
            include_lore=True,
            fallback_text="自然说话，别像客服，也别像在念设定。",
        )
        return bot_name, trait

    @staticmethod
    def _normalize_check_reason(reason: str) -> str:
        text = (reason or "").strip()
        if not text:
            return "未提供检查原因"
        while "  " in text:
            text = text.replace("  ", " ")
        if len(text) > 140:
            text = text[:140].rstrip() + "..."
        return text

    @staticmethod
    def _is_hard_replan_reason(reason: str) -> bool:
        if not reason:
            return False
        hard_markers = (
            "完全相同",
            "高度相似",
            "复读",
            "消息轰炸",
            "连续发送多条",
            "需要重新规划",
            "不适合发消息",
            "违规",
        )
        return any(marker in reason for marker in hard_markers)

    # ==================== 主生成入口 ====================

    async def generate(
        self,
        observation_info: ObservationInfo,
        conversation_info: ConversationInfo,
        action_type: str = "direct_reply",
    ) -> str:
        """根据行动类型生成回复文本。支持自动重试与后处理清洗。"""
        logger.debug(
            f"[私聊][{self.private_name}]启动回复生成 (action={action_type})"
        )
        # 准备上下文参数
        goals_text = self._compose_goals_block(conversation_info)
        knowledge_text = self._compose_knowledge_block(conversation_info)
        dialogue_text = self._compose_dialogue_block(observation_info)
        persona_text = f"你的名字是{self.bot_name}，{self.personality_info}"
        # 渲染提示词
        rendered_prompt = self._render_prompt_for_action(
            action_type,
            persona_text,
            goals_text,
            knowledge_text,
            dialogue_text,
        )
        logger.debug(
            f"[私聊][{
                self.private_name}]生成提示词长度={
                len(rendered_prompt)}"
        )
        # 带重试的生成循环
        last_error = None
        for attempt in range(self._max_generation_attempts):
            try:
                raw_output, _ = await asyncio.wait_for(
                    self.llm.generate_response_async(rendered_prompt),
                    timeout=self._generation_timeout_seconds,
                )
                logger.debug(
                    f"[私聊][{self.private_name}]第{attempt + 1}次生成原始输出: "
                    f"{raw_output[:200]}"
                )
                # 后处理：清洗LLM输出伪影、控制长度
                cleaned = self._sanitize_raw_output(raw_output)
                if cleaned.strip():
                    logger.info(
                        f"[私聊][{self.private_name}]回复生成成功 "
                        f"(attempt={attempt + 1}, len={len(cleaned)})"
                    )
                    return cleaned
                logger.warning(
                    f"[私聊][{self.private_name}]第{attempt + 1}次生成结果清洗后为空，准备重试"
                )
            except Exception as exc:
                last_error = exc
                logger.error(
                    f"[私聊][{self.private_name}]第{attempt + 1}次生成异常: {exc}"
                )

            if attempt < self._max_generation_attempts - 1:
                backoff = self._generation_retry_backoff_seconds * (attempt + 1)
                await asyncio.sleep(backoff)
        # 全部尝试失败，返回兜底回复
        logger.error(
            f"[私聊][{self.private_name}]回复生成彻底失败。最后异常: {last_error}"
        )
        return "抱歉，我现在有点混乱，让我重新思考一下..."

    # ==================== 提示词渲染 ====================

    def _render_prompt_for_action(
        self,
        action_type: str,
        persona_text: str,
        goals_text: str,
        knowledge_text: str,
        dialogue_text: str,
    ) -> str:
        """根据行动类型选择模板并注入上下文参数，返回完整提示词。"""
        name_dispatch = {
            "send_new_message": "pfc_send_new_message",
            "say_goodbye": "pfc_farewell",
        }
        prompt_name = name_dispatch.get(action_type, "pfc_direct_reply")
        if action_type == "send_new_message":
            logger.info(f"[私聊][{self.private_name}]选用追问消息模板")
        elif action_type == "say_goodbye":
            logger.info(f"[私聊][{self.private_name}]选用告别语模板")
        else:
            logger.info(f"[私聊][{self.private_name}]选用首次回复模板")
        tpl = global_prompt_manager.get_prompt(prompt_name)
        if not tpl:
            return ""
        return tpl.render(
            persona_text=persona_text,
            goals_str=goals_text,
            chat_history_text=dialogue_text,
            knowledge_info_str=knowledge_text,
        )

    # ==================== 上下文组装 ====================

    def _compose_goals_block(self, conv: ConversationInfo) -> str:
        """渲染目标列表为可读文本。"""
        parts = []
        try:
            items = getattr(conv, "goal_list", None) or []
            if not items:
                return "- 目前没有明确对话目标\n"
            for item in items:
                if isinstance(item, dict):
                    g = str(item.get("goal", "目标缺失"))
                    r = str(item.get("reasoning", "无原因"))
                else:
                    g, r = str(item), "无原因"
                parts.append(f"- 目标：{g}\n  原因：{r}")
        except Exception as exc:
            logger.error(f"[私聊][{self.private_name}]目标渲染失败: {exc}")
            return "- 渲染目标时出错\n"
        return "\n".join(parts) + "\n"

    def _compose_knowledge_block(self, conv: ConversationInfo) -> str:
        """渲染供参考的知识片段，每条最多2000字。"""
        header = "【供参考的相关知识】\n"
        try:
            k_items = getattr(conv, "knowledge_list", None) or []
            if not k_items:
                return header + "- 暂无相关知识。\n"
            recent = k_items[-5:]
            lines = []
            for seq, item in enumerate(recent, start=1):
                if isinstance(item, dict):
                    topic = str(item.get("query", "未知"))
                    body = str(item.get("knowledge", "无内容"))[:2000]
                    origin = str(item.get("source", "未知"))
                    lines.append(
                        f"{seq}. 关于'{topic}'(来源:{origin}): {body}"
                    )
                else:
                    lines.append(f"{seq}. {str(item)[:2000]}")
            return header + "\n".join(lines) + "\n"
        except Exception as exc:
            logger.error(f"[私聊][{self.private_name}]知识渲染失败: {exc}")
            return header + "- 获取知识时出错。\n"

    def _compose_dialogue_block(self, obs: ObservationInfo) -> str:
        """组合已有聊天记录和新到达的未处理消息。"""
        try:
            base = getattr(obs, "chat_history_str", "") or "还没有聊天记录。"
            pending_cnt = getattr(obs, "new_messages_count", 0) or 0
            pending_msgs = getattr(obs, "unprocessed_messages", None)
            if pending_cnt > 0 and pending_msgs:
                appended = build_readable_messages(
                    pending_msgs,
                    replace_bot_name=True,
                    timestamp_mode="relative",
                )
                base += f"\n--- {pending_cnt}条新消息 ---\n{appended}"
            return base
        except Exception as exc:
            logger.error(f"[私聊][{self.private_name}]对话文本组合失败: {exc}")
            return "获取聊天记录时出错。"

    # ==================== 后处理 ====================

    def _sanitize_raw_output(self, raw: str) -> str:
        """清洗LLM原始输出，去除各类生成伪影。

        处理步骤：
        1. 去除常见角色前缀（如 "bot:" "assistant:" 等）
        2. 去除引号包裹
        3. 合并连续空白行
        4. 如果超长则截断到最近一个句子边界
        """
        if not raw:
            return ""
        text = raw.strip()
        # 去除角色前缀
        role_prefixes = (
            f"{self.bot_name}：",
            f"{self.bot_name}:",
            "bot:",
            "Bot:",
            "BOT:",
            "assistant:",
            "Assistant:",
            "ASSISTANT:",
        )
        for prefix in role_prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        # 去除外层引号
        if len(text) >= 2:
            open_quotes = ('"', "\u201c", "\u2018", "'")
            close_quotes = ('"', "\u201d", "\u2019", "'")
            if text[0] in open_quotes and text[-1] in close_quotes:
                text = text[1:-1].strip()
        # 合并连续空行
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        # 长度治理
        if len(text) > self._response_length_soft_cap:
            text = self._truncate_at_sentence_boundary(
                text, self._response_length_soft_cap
            )
        return text

    @staticmethod
    def _truncate_at_sentence_boundary(text: str, max_len: int) -> str:
        """将文本截断到不超过max_len，尝试在句子边界处切割。"""
        if len(text) <= max_len:
            return text
        # 在max_len附近寻找句末标点
        search_zone = text[: max_len + 20]
        best_cut = -1
        for marker in ("。", "！", "？", ".", "!", "?", "\n"):
            pos = search_zone.rfind(marker, max_len // 2, max_len + 10)
            if pos > best_cut:
                best_cut = pos
        if best_cut > max_len // 2:
            return text[: best_cut + 1].strip()
        # 找不到句子边界就硬截
        return text[:max_len].strip()

    def _split_into_natural_segments(self, text: str) -> List[str]:
        """将长文本在自然断点处拆分为多段。

        优先识别 [BREAK] 标记，
        其次按双换行拆分，最后原样返回。
        """
        if not text:
            return [text]
        # 优先识别 [BREAK] 标记
        if "[BREAK]" in text:
            segments = [
                seg.strip() for seg in text.split("[BREAK]") if seg.strip()
            ]
            if segments:
                return segments
        # 尝试按双换行拆分
        if "\n\n" in text:
            segments = [
                seg.strip() for seg in text.split("\n\n") if seg.strip()
            ]
            if len(segments) > 1:
                return segments
        return [text]

    # ==================== 回复质检 ====================

    async def check_reply(
        self,
        reply: str,
        goal: str,
        chat_history: List[Dict[str, Any]],
        chat_history_str: str,
        retry_count: int = 0,
    ) -> Tuple[bool, str, bool]:
        """委托ReplyChecker执行回复质量检查。"""
        if not (reply or "").strip():
            return False, "生成内容为空，建议重新生成", False

        suitable, reason, need_replan = await self.reply_checker.check(
            reply, goal, chat_history, chat_history_str, retry_count
        )
        normalized_reason = self._normalize_check_reason(reason)

        if not suitable and self._is_hard_replan_reason(normalized_reason):
            return False, normalized_reason, True

        if not suitable and retry_count >= self._force_replan_retry_count:
            return False, normalized_reason, True

        return suitable, normalized_reason, need_replan

import asyncio
from typing import Any, Dict, List, Tuple, TYPE_CHECKING
from src.common.logger import get_logger
from src.chat.brain_chat.runtime_config import brainchat_module_view
from src.llm_models.utils_model import LLMRequest
from src.config.config import global_config, model_config
from src.chat.utils.prompt_builder import global_prompt_manager
from .chat_observer import ChatObserver
from .pfc_utils import (
    extract_json_objects,
    get_items_from_json,
    parse_json_payload,
    truncate_text,
)
from .conversation_info import ConversationInfo
from .observation_info import ObservationInfo
from src.chat.utils.chat_message_builder import build_readable_messages
from src.person_info.bot_identity import get_bot_identity_manager
from rich.traceback import install

install(extra_lines=3)

if TYPE_CHECKING:
    pass

logger = get_logger("PFC模块")


def _calculate_similarity(goal1: str, goal2: str) -> float:
    """简单计算两个目标之间的相似度

    使用词级 Jaccard 相似度，避免字符级误判

    Args:
        goal1: 第一个目标
        goal2: 第二个目标

    Returns:
        float: 相似度得分 (0-1)
    """
    words1 = set(goal1.split())
    words2 = set(goal2.split())
    overlap = len(words1.intersection(words2))
    total = len(words1.union(words2))
    return overlap / total if total > 0 else 0


class GoalAnalyzer:
    """对话目标分析器"""

    def __init__(self, stream_id: str, private_name: str):
        config = brainchat_module_view("brain_pfc_goal")
        self.llm = LLMRequest(
            model_set=model_config.model_task_config.lightweight,
            request_type="conversation_goal",
        )

        self.name, self.personality_info = self._get_personality_prompt()
        self.nick_name = (
            getattr(global_config, "BOT_ALIAS_NAMES", None)
            or getattr(global_config.bot, "alias_names", [])
        )
        self.private_name = private_name
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)

        # 多目标存储结构
        self.goals = []  # 存储多个目标
        self.max_goals = int(config.get("max_goals", 3))  # 同时保持的最大目标数量
        self.current_goal_and_reason = None
        self._analysis_timeout_seconds = float(
            config.get("analysis_timeout_seconds", 18.0)
        )
        self._goal_similarity_threshold = float(
            config.get("goal_similarity_threshold", 0.72)
        )

    @staticmethod
    def _get_personality_prompt() -> Tuple[str, str]:
        """从统一人格源获取目标分析所需的人设信息。"""
        identity = get_bot_identity_manager()
        bot_name = identity.get_display_name() or global_config.bot.nickname
        trait = identity.build_persona_brief(
            include_name_prefix=True,
            include_style=True,
            include_interests=False,
            include_lore=True,
            fallback_text="自然参与聊天，不装系统，不装模板。",
        )
        return bot_name, trait

    @staticmethod
    def _normalize_text(value: str) -> str:
        return (value or "").strip().replace("\n", " ")

    @staticmethod
    def _safe_goal_item(item: Dict[str, Any]) -> Dict[str, str]:
        return {
            "goal": str(item.get("goal", "")).strip(),
            "reasoning": str(item.get("reasoning", "")).strip()
            or "未提供原因",
        }

    def _deduplicate_and_rank_goals(
        self, goal_items: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        ranked_buffer: List[Tuple[float, Dict[str, str]]] = []

        for index, item in enumerate(goal_items):
            goal_text = self._normalize_text(item.get("goal", ""))
            reasoning_text = self._normalize_text(item.get("reasoning", ""))
            if not goal_text:
                continue

            candidate = {
                "goal": goal_text,
                "reasoning": reasoning_text or "未提供原因",
            }

            duplicated = False
            for _, existing in ranked_buffer:
                if (
                    _calculate_similarity(goal_text, existing["goal"])
                    >= self._goal_similarity_threshold
                ):
                    duplicated = True
                    break
            if duplicated:
                continue

            base_score = 1.0 / (index + 1)
            if "结束对话" in goal_text:
                base_score += 0.3
            if "等待" in goal_text:
                base_score += 0.05
            if "回复" in goal_text or "回答" in goal_text:
                base_score += 0.1

            ranked_buffer.append((base_score, candidate))

        ranked_buffer.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in ranked_buffer[: self.max_goals]]

    def _build_goal_prompt(
        self,
        persona_text: str,
        goals_text: str,
        chat_history_text: str,
        action_history_text: str,
    ) -> str:
        return f"""{persona_text}。现在你在参与一场QQ聊天，请分析以下聊天记录，并根据你的性格特征确定多个明确的对话目标。
这些目标应该反映出对话的不同方面和意图。

{action_history_text}
当前对话目标：
{goals_text}

聊天记录：
{chat_history_text}

请分析当前对话并确定最适合的对话目标。你可以：
1. 保持现有目标不变
2. 修改现有目标
3. 添加新目标
4. 删除不再相关的目标
5. 如果你想结束对话，请设置一个目标，目标goal为"结束对话"，原因reasoning为你希望结束对话

请以JSON数组格式输出当前的所有对话目标，每个目标包含以下字段：
1. goal: 对话目标（简短的一句话）
2. reasoning: 对话原因，为什么设定这个目标（简要解释）

输出格式示例：
[
{{
    "goal": "回答用户关于Python编程的具体问题",
    "reasoning": "用户提出了关于Python的技术问题，需要专业且准确的解答"
}},
{{
    "goal": "给出安装步骤并确认对方环境",
    "reasoning": "用户不仅问了概念，还需要可执行的下一步"
}}
]"""

    async def _call_llm_with_timeout(self, prompt: str) -> str:
        try:
            content, _ = await asyncio.wait_for(
                self.llm.generate_response_async(
                    prompt, usage_stream_id=self.chat_observer.stream_id
                ),
                timeout=self._analysis_timeout_seconds,
            )
            return (content or "").strip()
        except asyncio.TimeoutError:
            logger.warning(
                f"[私聊][{
                    self.private_name}]目标分析超时({
                    self._analysis_timeout_seconds}s)"
            )
            return ""
        except Exception as e:
            logger.error(
                f"[私聊][{self.private_name}]调用目标分析LLM失败: {e}"
            )
            return ""

    def _extract_goal_items(self, content: str) -> List[Dict[str, str]]:
        candidates: List[Dict[str, str]] = []
        if not content:
            return candidates

        success, parsed = get_items_from_json(
            content,
            self.private_name,
            "goal",
            "reasoning",
            required_types={"goal": str, "reasoning": str},
            allow_array=True,
        )
        if success:
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        candidates.append(self._safe_goal_item(item))
            elif isinstance(parsed, dict):
                candidates.append(self._safe_goal_item(parsed))

        if candidates:
            return candidates

        for item in extract_json_objects(content):
            if isinstance(item, dict) and "goal" in item:
                candidates.append(self._safe_goal_item(item))
        return candidates

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "是"}:
                return True
            if lowered in {"false", "0", "no", "否"}:
                return False
        return default

    @staticmethod
    def _format_goals_text(goal_list: List[Any]) -> str:
        if not goal_list:
            return "目标：目前没有明确对话目标，产生该对话目标的原因：最好思考一个对话目标\n"

        lines: List[str] = []
        for raw_goal in goal_list:
            if isinstance(raw_goal, dict):
                goal = str(raw_goal.get("goal", "目标内容缺失")).strip()
                reasoning = str(
                    raw_goal.get("reasoning", "没有明确原因")
                ).strip()
            else:
                goal = str(raw_goal).strip() or "目标内容缺失"
                reasoning = "没有明确原因"
            lines.append(f"目标：{goal}，产生该对话目标的原因：{reasoning}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_action_history(actions: List[Any]) -> str:
        if not actions:
            return "你之前做的事情是：无\n"
        rows = ["你之前做的事情是："]
        for action in actions:
            rows.append(str(action))
        return "\n".join(rows) + "\n"

    async def analyze_goal(
        self,
        conversation_info: ConversationInfo,
        observation_info: ObservationInfo,
    ):
        """分析对话历史并设定目标

        Args:
            conversation_info: 对话信息
            observation_info: 观察信息

        Returns:
            Tuple[str, str, str]: (目标, 方法, 原因)
        """
        goals_str = self._format_goals_text(conversation_info.goal_list)

        # 获取聊天历史记录
        chat_history_text = observation_info.chat_history_str

        if observation_info.new_messages_count > 0:
            new_messages_list = observation_info.unprocessed_messages
            new_messages_str = await build_readable_messages(
                new_messages_list,
                replace_bot_name=True,
                merge_messages=False,
                timestamp_mode="relative",
                read_mark=0.0,
            )
            chat_history_text += f"\n--- 以下是 {
                observation_info.new_messages_count} 条新消息 ---\n{new_messages_str}"

            # await observation_info.clear_unprocessed_messages()

        persona_text = f"你的名字是{self.name}，{self.personality_info}。"
        action_history_text = self._format_action_history(
            conversation_info.done_action
        )
        prompt = self._build_goal_prompt(
            persona_text=persona_text,
            goals_text=goals_str,
            chat_history_text=chat_history_text,
            action_history_text=action_history_text,
        )

        logger.debug(
            f"[私聊][{
                self.private_name}]目标分析提示词: {
                truncate_text(
                    prompt,
                    1600)}"
        )
        content = await self._call_llm_with_timeout(prompt)
        if content:
            logger.debug(
                f"[私聊][{
                    self.private_name}]目标分析原始返回: {
                    truncate_text(
                        content,
                        1000)}"
            )

        model_goals = self._extract_goal_items(content)
        preserved_goals = [
            self._safe_goal_item(item)
            for item in conversation_info.goal_list
            if isinstance(item, dict)
        ]
        merged_goals = self._deduplicate_and_rank_goals(
            model_goals + preserved_goals
        )

        if merged_goals:
            conversation_info.goal_list = merged_goals
            self.goals = [
                (goal_item["goal"], "", goal_item["reasoning"])
                for goal_item in merged_goals
            ]
            primary_goal = merged_goals[0]
            return primary_goal["goal"], "", primary_goal["reasoning"]

        if conversation_info.goal_list:
            first_goal = conversation_info.goal_list[0]
            if isinstance(first_goal, dict):
                return (
                    str(first_goal.get("goal", "")),
                    "",
                    str(first_goal.get("reasoning", "")),
                )

        return "", "", ""

    async def _update_goals(self, new_goal: str, method: str, reasoning: str):
        """更新目标列表

        Args:
            new_goal: 新的目标
            method: 实现目标的方法
            reasoning: 目标的原因
        """
        current = [
            {"goal": goal, "reasoning": reason}
            for goal, _, reason in self.goals
        ]
        current.insert(0, {"goal": new_goal, "reasoning": reasoning})
        merged = self._deduplicate_and_rank_goals(current)
        self.goals = [
            (item["goal"], method, item["reasoning"]) for item in merged
        ]

    async def get_all_goals(self) -> List[Tuple[str, str, str]]:
        """获取所有当前目标

        Returns:
            List[Tuple[str, str, str]]: 目标列表，每项为(目标, 方法, 原因)
        """
        return self.goals.copy()

    async def get_alternative_goals(self) -> List[Tuple[str, str, str]]:
        """获取除了当前主要目标外的其他备选目标

        Returns:
            List[Tuple[str, str, str]]: 备选目标列表
        """
        if len(self.goals) <= 1:
            return []
        return self.goals[1:].copy()

    async def analyze_conversation(self, goal, reasoning):
        messages = self.chat_observer.get_cached_messages()
        chat_history_text = await build_readable_messages(
            messages,
            replace_bot_name=True,
            merge_messages=False,
            timestamp_mode="relative",
            read_mark=0.0,
        )

        persona_text = f"你的名字是{self.name}，{self.personality_info}。"
        # ===> Persona 文本构建结束 <===

        prompt = await global_prompt_manager.format_prompt(
            "pfc_conversation_analysis",
            persona_text=persona_text,
            goal=goal,
            reasoning=reasoning,
            chat_history_text=chat_history_text,
        )

        try:
            content = await self._call_llm_with_timeout(prompt)
            logger.debug(
                f"[私聊][{
                    self.private_name}]对话分析原始返回: {
                    truncate_text(
                        content,
                        800)}"
            )

            parsed = parse_json_payload(content, allow_array=False)
            if isinstance(parsed, dict):
                goal_achieved = self._coerce_bool(
                    parsed.get("goal_achieved"), default=False
                )
                stop_conversation = self._coerce_bool(
                    parsed.get("stop_conversation"), default=False
                )
                parsed_reason = str(
                    parsed.get("reason", "未提供原因") or "未提供原因"
                )
                return goal_achieved, stop_conversation, parsed_reason

            success, result = get_items_from_json(
                content,
                self.private_name,
                "goal_achieved",
                "stop_conversation",
                "reason",
                required_types={
                    "goal_achieved": bool,
                    "stop_conversation": bool,
                    "reason": str,
                },
                allow_array=False,
            )
            if success and isinstance(result, dict):
                return (
                    result["goal_achieved"],
                    result["stop_conversation"],
                    result["reason"],
                )

            reason_text = truncate_text(content or "解析结果失败", 120)
            fallback_stop = "结束" in reason_text and "继续" not in reason_text
            return False, fallback_stop, reason_text

        except Exception as e:
            logger.error(
                f"[私聊][{self.private_name}]分析对话状态时出错: {str(e)}"
            )
            return False, False, f"分析出错: {str(e)}"

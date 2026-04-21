import traceback
import time
import asyncio
import random
import re

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from src.common.logger import get_logger
from src.common.data_models.database_data_model import DatabaseMessages
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.common.data_models.llm_data_model import LLMGenerationDataModel
from src.config.config import global_config, model_config
from src.llm_models.utils_model import LLMRequest
from src.chat.message_receive.message import UserInfo, Seg, MessageRecv, MessageSending
from src.chat.message_receive.chat_stream import ChatStream
from src.chat.message_receive.uni_message_sender import UniversalMessageSender
from src.chat.utils.timer_calculator import Timer  # <--- Import Timer
from src.chat.utils.utils import get_chat_type_and_target_info, is_bot_self
from src.chat.utils.prompt_builder import global_prompt_manager
from src.chat.utils.chat_message_builder import (
    build_readable_messages,
    get_raw_msg_before_timestamp_with_chat,
    replace_user_references,
)
from src.bw_learner.expression_selector import expression_selector
from src.plugin_system.apis.message_api import translate_pid_to_description

# from src.memory_system.memory_activator import MemoryActivator

from src.person_info.person_info import Person, is_person_known
from src.plugin_system.base.component_types import ActionInfo, EventType
from src.plugin_system.apis import llm_api

from src.chat.replyer.prompt.lpmm_prompt import init_lpmm_prompt
from src.chat.replyer.prompt.replyer_private_prompt import init_replyer_private_prompt
from src.chat.replyer.prompt.rewrite_prompt import init_rewrite_prompt
from src.memory_system.memory_retrieval import init_memory_retrieval_prompt, build_memory_retrieval_prompt
from src.bw_learner.jargon_explainer import explain_jargon_in_context
from src.config.prompt_loader import get_prompt, PromptCategory

init_lpmm_prompt()
init_replyer_private_prompt()
init_rewrite_prompt()
init_memory_retrieval_prompt()


logger = get_logger("replyer")


class PrivateReplyer:
    def __init__(
        self,
        chat_stream: ChatStream,
        request_type: str = "replyer",
    ):
        self.express_model = LLMRequest(model_set=model_config.model_task_config.replyer, request_type=request_type)
        self.chat_stream = chat_stream
        self.is_group_chat, self.chat_target_info = get_chat_type_and_target_info(self.chat_stream.stream_id)
        self.heart_fc_sender = UniversalMessageSender()
        # self.memory_activator = MemoryActivator()

        from src.plugin_system.core.tool_use import ToolExecutor  # 延迟导入ToolExecutor，不然会循环依赖

        self.tool_executor = ToolExecutor(chat_id=self.chat_stream.stream_id, enable_cache=True, cache_ttl=3)
        self.current_persona = None
        self._update_current_persona()

    def _update_current_persona(self):
        """从心理核心同步当前人格状态"""
        try:
            from src.modules.modcore.psychological_core import get_psychological_core
            psy_core = get_psychological_core()
            if not psy_core:
                return
            psy_snapshot = psy_core.get_channel_psychological_state(self.chat_stream.stream_id)
            active_info = psy_snapshot.get("active_persona", {})
            if active_info and not active_info.get("is_main", True):
                persona_name = active_info.get("name", "")
                persona_tone = active_info.get("tone", "")
                persona_source = active_info.get("source", "")
                self.current_persona = type('Persona', (), {
                    'name': persona_name,
                    'tone': persona_tone,
                    'source': persona_source,
                    'trauma_state': active_info.get("trauma_state")
                })()
                logger.info(f"[私聊心理集成] 同步人格：{persona_name}，来源：{persona_source}")
            else:
                self.current_persona = None
        except Exception as e:
            logger.debug(f"更新私聊人格失败: {e}")
            self.current_persona = None

    def _get_default_result_for_task(self, name: str) -> Any:
        """异步任务超时或失败时的降级结果"""
        defaults = {
            "expression_habits": ("", []),
            "tool_info": "",
            "prompt_info": "",
            "actions_info": "",
            "personality_prompt": "",
            "memory_retrieval": "",
            "jargon_explanation": "",
            "psychological_context": "",
            "learning_context": "",
        }
        return defaults.get(name, "")

    async def generate_reply_with_context(
        self,
        extra_info: str = "",
        reply_reason: str = "",
        available_actions: Optional[Dict[str, ActionInfo]] = None,
        chosen_actions: Optional[List[ActionPlannerInfo]] = None,
        enable_tool: bool = True,
        from_plugin: bool = True,
        think_level: int = 1,
        stream_id: Optional[str] = None,
        reply_message: Optional[DatabaseMessages] = None,
        reply_time_point: Optional[float] = time.time(),
        unknown_words: Optional[List[str]] = None,
        log_reply: bool = True,
    ) -> Tuple[bool, LLMGenerationDataModel]:
        # sourcery skip: merge-nested-ifs
        """
        回复器 (Replier): 负责生成回复文本的核心逻辑。

        Args:
            reply_to: 回复对象，格式为 "发送者:消息内容"
            extra_info: 额外信息，用于补充上下文
            reply_reason: 回复原因
            available_actions: 可用的动作信息字典
            chosen_actions: 已选动作
            enable_tool: 是否启用工具调用
            from_plugin: 是否来自插件

        Returns:
            Tuple[bool, Optional[Dict[str, Any]], Optional[str]]: (是否成功, 生成的回复, 使用的prompt)
        """

        prompt = None
        selected_expressions: Optional[List[int]] = None
        llm_response = LLMGenerationDataModel()
        if available_actions is None:
            available_actions = {}
        try:
            # 3. 构建 Prompt
            with Timer("构建Prompt", {}):  # 内部计时器，可选保留
                prompt, selected_expressions = await self.build_prompt_reply_context(
                    extra_info=extra_info,
                    available_actions=available_actions,
                    chosen_actions=chosen_actions,
                    enable_tool=enable_tool,
                    reply_message=reply_message,
                    reply_reason=reply_reason,
                    unknown_words=unknown_words,
                )
            llm_response.prompt = prompt
            llm_response.selected_expressions = selected_expressions

            if not prompt:
                logger.warning("构建prompt失败，跳过回复生成")
                return False, llm_response
            from src.plugin_system.core.events_manager import events_manager

            if not from_plugin:
                continue_flag, modified_message = await events_manager.handle_mai_events(
                    EventType.POST_LLM, None, prompt, None, stream_id=stream_id
                )
                if not continue_flag:
                    raise UserWarning("插件于请求前中断了内容生成")
                if modified_message and modified_message._modify_flags.modify_llm_prompt:
                    llm_response.prompt = modified_message.llm_prompt
                    prompt = str(modified_message.llm_prompt)

            # 4. 调用 LLM 生成回复
            content = None
            reasoning_content = None
            model_name = "unknown_model"

            try:
                content, reasoning_content, model_name, tool_call = await self.llm_generate_content(prompt)
                logger.debug(f"replyer生成内容: {content}")
                if content:
                    content = self._sanitize_output(content)
                    self._update_current_persona()
                llm_response.content = content
                llm_response.reasoning = reasoning_content
                llm_response.model = model_name
                llm_response.tool_calls = tool_call
                continue_flag, modified_message = await events_manager.handle_mai_events(
                    EventType.AFTER_LLM, None, prompt, llm_response, stream_id=stream_id
                )
                if not from_plugin and not continue_flag:
                    raise UserWarning("插件于请求后取消了内容生成")
                if modified_message:
                    if modified_message._modify_flags.modify_llm_prompt:
                        logger.warning("警告：插件在内容生成后才修改了prompt，此修改不会生效")
                        llm_response.prompt = modified_message.llm_prompt  # 虽然我不知道为什么在这里需要改prompt
                    if modified_message._modify_flags.modify_llm_response_content:
                        llm_response.content = modified_message.llm_response_content
                    if modified_message._modify_flags.modify_llm_response_reasoning:
                        llm_response.reasoning = modified_message.llm_response_reasoning
            except UserWarning as e:
                raise e
            except Exception as llm_e:
                # 精简报错信息
                logger.error(f"LLM 生成失败: {llm_e}")
                return False, llm_response  # LLM 调用失败则无法生成回复

            return True, llm_response

        except UserWarning as uw:
            raise uw
        except Exception as e:
            logger.error(f"回复生成意外失败: {e}")
            traceback.print_exc()
            return False, llm_response

    async def rewrite_reply_with_context(
        self,
        raw_reply: str = "",
        reason: str = "",
        reply_to: str = "",
    ) -> Tuple[bool, LLMGenerationDataModel]:
        """
        表达器 (Expressor): 负责重写和优化回复文本。

        Args:
            raw_reply: 原始回复内容
            reason: 回复原因
            reply_to: 回复对象，格式为 "发送者:消息内容"
            relation_info: 关系信息

        Returns:
            Tuple[bool, Optional[str]]: (是否成功, 重写后的回复内容)
        """
        llm_response = LLMGenerationDataModel()
        try:
            with Timer("构建Prompt", {}):  # 内部计时器，可选保留
                prompt = await self.build_prompt_rewrite_context(
                    raw_reply=raw_reply,
                    reason=reason,
                    reply_to=reply_to,
                )
            llm_response.prompt = prompt

            content = None
            reasoning_content = None
            model_name = "unknown_model"
            if not prompt:
                logger.error("Prompt 构建失败，无法生成回复。")
                return False, llm_response

            try:
                content, reasoning_content, model_name, _ = await self.llm_generate_content(prompt)
                logger.info(f"想要表达：{raw_reply}||理由：{reason}||生成回复: {content}\n")
                llm_response.content = content
                llm_response.reasoning = reasoning_content
                llm_response.model = model_name

            except Exception as llm_e:
                # 精简报错信息
                logger.error(f"LLM 生成失败: {llm_e}")
                return False, llm_response  # LLM 调用失败则无法生成回复

            return True, llm_response

        except Exception as e:
            logger.error(f"回复生成意外失败: {e}")
            traceback.print_exc()
            return False, llm_response

    async def build_relation_info(self, chat_content: str, sender: str):
        if not global_config.relationship.enable_relationship:
            return ""

        if not sender:
            return ""

        if sender == global_config.bot.nickname:
            return ""

        # 获取用户ID
        person = Person(person_name=sender)
        if not is_person_known(person_name=sender):
            logger.warning(f"未找到用户 {sender} 的ID，跳过信息提取")
            return f"你完全不认识{sender}，不理解ta的相关信息。"

        sender_relation = await person.build_relationship(chat_content)

        return f"{sender_relation}"

    async def build_expression_habits(
        self, chat_history: str, target: str, reply_reason: str = ""
    ) -> Tuple[str, List[int]]:
        # sourcery skip: for-append-to-extend
        """构建表达习惯块

        Args:
            chat_history: 聊天历史记录
            target: 目标消息内容
            reply_reason: planner给出的回复理由

        Returns:
            str: 表达习惯信息字符串
        """
        # 检查是否允许在此聊天流中使用表达
        use_expression, _, _ = global_config.expression.get_expression_config_for_chat(self.chat_stream.stream_id)
        if not use_expression:
            return "", []
        style_habits = []
        # 使用从处理器传来的选中表达方式
        # 使用模型预测选择表达方式
        selected_expressions, selected_ids = await expression_selector.select_suitable_expressions(
            self.chat_stream.stream_id, chat_history, max_num=8, target_message=target, reply_reason=reply_reason
        )

        if selected_expressions:
            logger.debug(f"使用处理器选中的{len(selected_expressions)}个表达方式")
            for expr in selected_expressions:
                if isinstance(expr, dict) and "situation" in expr and "style" in expr:
                    style_habits.append(f"当{expr['situation']}时：{expr['style']}")
        else:
            logger.debug("没有从处理器获得表达方式，将使用空的表达方式")
            # 不再在replyer中进行随机选择，全部交给处理器处理

        style_habits_str = "\n".join(style_habits)

        # 动态构建expression habits块
        expression_habits_block = ""
        expression_habits_title = ""
        if style_habits_str.strip():
            expression_habits_title = "在回复时,你可以参考以下的语言习惯，不要生硬使用："
            expression_habits_block += f"{style_habits_str}\n"

        return f"{expression_habits_title}\n{expression_habits_block}", selected_ids

    async def build_tool_info(self, chat_history: str, sender: str, target: str, enable_tool: bool = True) -> str:
        """构建工具信息块

        Args:
            chat_history: 聊天历史记录
            reply_to: 回复对象，格式为 "发送者:消息内容"
            enable_tool: 是否启用工具调用

        Returns:
            str: 工具信息字符串
        """

        if not enable_tool:
            return ""

        try:
            # 使用工具执行器获取信息
            tool_results, _, _ = await self.tool_executor.execute_from_chat_message(
                sender=sender, target_message=target, chat_history=chat_history, return_details=False
            )

            if tool_results:
                tool_info_str = "以下是你通过工具获取到的实时信息：\n"
                for tool_result in tool_results:
                    tool_name = tool_result.get("tool_name", "unknown")
                    content = tool_result.get("content", "")
                    result_type = tool_result.get("type", "tool_result")

                    tool_info_str += f"- 【{tool_name}】{result_type}: {content}\n"

                tool_info_str += "以上是你获取到的实时信息，请在回复时参考这些信息。"
                logger.info(f"获取到 {len(tool_results)} 个工具结果")

                return tool_info_str
            else:
                logger.debug("未获取到任何工具结果")
                return ""

        except Exception as e:
            logger.error(f"工具信息获取失败: {e}")
            return ""

    def _parse_reply_target(self, target_message: Optional[str]) -> Tuple[str, str]:
        """解析回复目标消息

        Args:
            target_message: 目标消息，格式为 "发送者:消息内容" 或 "发送者：消息内容"

        Returns:
            Tuple[str, str]: (发送者名称, 消息内容)
        """
        sender = ""
        target = ""
        # 添加None检查，防止NoneType错误
        if target_message is None:
            return sender, target
        if ":" in target_message or "：" in target_message:
            # 使用正则表达式匹配中文或英文冒号
            parts = re.split(pattern=r"[:：]", string=target_message, maxsplit=1)
            if len(parts) == 2:
                sender = parts[0].strip()
                target = parts[1].strip()
        return sender, target

    async def _refresh_stale_pic_descriptions(self, target: str) -> None:
        picid_pattern = r"\[picid:([^\]]+)\]"
        picids = re.findall(picid_pattern, target)
        if not picids:
            return
        from src.chat.utils.utils_image import get_image_manager
        img_mgr = get_image_manager()
        for pid in picids:
            await img_mgr.refresh_stale_description(pid)

    def _replace_picids_with_descriptions(self, text: str) -> str:
        pic_pattern = r"\[picid:([^\]]+)\]"

        def replace_pic_id(match: re.Match) -> str:
            pic_id = match.group(1)
            description = translate_pid_to_description(pic_id)
            return f"[图片：{description}]"

        return re.sub(pic_pattern, replace_pic_id, text)

    def _analyze_target_content(self, target: str) -> Tuple[bool, bool, str, str]:
        """分析target内容类型（基于原始picid格式）

        Args:
            target: 目标消息内容（包含[picid:xxx]格式）

        Returns:
            Tuple[bool, bool, str, str]: (是否只包含图片, 是否包含文字, 图片部分, 文字部分)
        """
        if not target or not target.strip():
            return False, False, "", ""

        # 检查是否只包含picid标记
        picid_pattern = r"\[picid:[^\]]+\]"
        picid_matches = re.findall(picid_pattern, target)

        # 移除所有picid标记后检查是否还有文字内容
        text_without_picids = re.sub(picid_pattern, "", target).strip()

        has_only_pics = len(picid_matches) > 0 and not text_without_picids
        has_text = bool(text_without_picids)

        # 提取图片部分（转换为[图片:描述]格式）
        pic_part = ""
        if picid_matches:
            pic_descriptions = []
            for picid_match in picid_matches:
                pic_id = picid_match[7:-1]  # 提取picid:xxx中的xxx部分（从第7个字符开始）
                description = translate_pid_to_description(pic_id)
                logger.debug(f"图片ID: {pic_id}, 描述: {description}")
                # 如果description已经是[图片]格式，直接使用；否则包装为[图片:描述]格式
                if description == "[图片]":
                    pic_descriptions.append(description)
                else:
                    pic_descriptions.append(f"[图片:{description}]")
            pic_part = "".join(pic_descriptions)

        return has_only_pics, has_text, pic_part, text_without_picids

    async def build_keywords_reaction_prompt(self, target: Optional[str]) -> str:
        """构建关键词反应提示

        Args:
            target: 目标消息内容

        Returns:
            str: 关键词反应提示字符串
        """
        # 关键词检测与反应
        keywords_reaction_prompt = ""
        try:
            # 添加None检查，防止NoneType错误
            if target is None:
                return keywords_reaction_prompt

            # 处理关键词规则
            for rule in global_config.keyword_reaction.keyword_rules:
                if any(keyword in target for keyword in rule.keywords):
                    logger.info(f"检测到关键词规则：{rule.keywords}，触发反应：{rule.reaction}")
                    keywords_reaction_prompt += f"{rule.reaction}，"

            # 处理正则表达式规则
            for rule in global_config.keyword_reaction.regex_rules:
                for pattern_str in rule.regex:
                    try:
                        pattern = re.compile(pattern_str)
                        if result := pattern.search(target):
                            reaction = rule.reaction
                            for name, content in result.groupdict().items():
                                reaction = reaction.replace(f"[{name}]", content)
                            logger.info(f"匹配到正则表达式：{pattern_str}，触发反应：{reaction}")
                            keywords_reaction_prompt += f"{reaction}，"
                            break
                    except re.error as e:
                        logger.error(f"正则表达式编译错误: {pattern_str}, 错误信息: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"关键词检测与反应时发生异常: {str(e)}", exc_info=True)

        return keywords_reaction_prompt

    async def _time_and_run_task(self, coroutine, name: str, timeout: Optional[float] = None) -> Tuple[str, Any, float]:
        """计时并运行异步任务，支持超时控制和异常降级"""
        start_time = time.time()
        try:
            if timeout is not None and timeout > 0:
                result = await asyncio.wait_for(coroutine, timeout=timeout)
            else:
                result = await coroutine
        except asyncio.TimeoutError:
            logger.warning(f"私聊任务 {name} 超时 ({timeout}s)，使用降级结果")
            result = self._get_default_result_for_task(name)
        except Exception as e:
            logger.error(f"私聊任务 {name} 执行失败: {e}")
            result = self._get_default_result_for_task(name)
        end_time = time.time()
        duration = end_time - start_time
        return name, result, duration

    async def _build_disabled_jargon_explanation(self) -> str:
        """当关闭黑话解释时使用的占位协程，避免额外的LLM调用"""
        return ""

    async def build_actions_prompt(
        self, available_actions: Dict[str, ActionInfo], chosen_actions_info: Optional[List[ActionPlannerInfo]] = None
    ) -> str:
        """构建动作提示"""

        action_descriptions = ""
        skip_names = ["emoji", "build_memory", "build_relation", "reply"]
        if available_actions:
            action_descriptions = "除了进行回复之外，你可以做以下这些动作，不过这些动作由另一个模型决定，：\n"
            for action_name, action_info in available_actions.items():
                if action_name in skip_names:
                    continue
                action_description = action_info.description
                action_descriptions += f"- {action_name}: {action_description}\n"
            action_descriptions += "\n"

        chosen_action_descriptions = ""
        if chosen_actions_info:
            for action_plan_info in chosen_actions_info:
                action_name = action_plan_info.action_type
                if action_name in skip_names:
                    continue
                action_description: str = "无描述"
                reasoning: str = "无原因"
                if action := available_actions.get(action_name):
                    action_description = action.description or action_description
                    reasoning = action_plan_info.reasoning or reasoning

                chosen_action_descriptions += f"- {action_name}: {action_description}，原因：{reasoning}\n"

        if chosen_action_descriptions:
            action_descriptions += "根据聊天情况，另一个模型决定在回复的同时做以下这些动作：\n"
            action_descriptions += chosen_action_descriptions

        return action_descriptions

    async def build_personality_prompt(self) -> str:
        bot_name = global_config.bot.nickname
        if global_config.bot.alias_names:
            bot_nickname = f",也有人叫你{','.join(global_config.bot.alias_names)}"
        else:
            bot_nickname = ""

        # 获取基础personality
        prompt_personality = global_config.personality.personality

        # 检查是否需要随机替换为状态
        if (
            global_config.personality.states
            and global_config.personality.state_probability > 0
            and random.random() < global_config.personality.state_probability
        ):
            # 随机选择一个状态替换personality
            selected_state = random.choice(global_config.personality.states)
            prompt_personality = selected_state

        prompt_personality = f"{prompt_personality};"
        age = global_config.personality.character_age
        age_stage = global_config.personality.character_age_stage
        hobbies = global_config.personality.character_hobbies
        age_part = f"，{age}岁{age_stage}" if age > 0 and age_stage else ""
        hobby_part = f"，平时喜欢{hobbies}" if hobbies else ""
        return f"你的名字是{bot_name}{bot_nickname}{age_part}，你{prompt_personality}{hobby_part}"

    def _parse_chat_prompt_config_to_chat_id(self, chat_prompt_str: str) -> Optional[tuple[str, str]]:
        """
        解析聊天prompt配置字符串并生成对应的 chat_id 和 prompt内容

        Args:
            chat_prompt_str: 格式为 "platform:id:type:prompt内容" 的字符串

        Returns:
            tuple: (chat_id, prompt_content)，如果解析失败则返回 None
        """
        try:
            # 使用 split 分割，但限制分割次数为3，因为prompt内容可能包含冒号
            parts = chat_prompt_str.split(":", 3)
            if len(parts) != 4:
                return None

            platform = parts[0]
            id_str = parts[1]
            stream_type = parts[2]
            prompt_content = parts[3]

            # 判断是否为群聊
            is_group = stream_type == "group"

            # 使用 ChatManager 提供的接口生成 chat_id，避免在此重复实现逻辑
            from src.chat.message_receive.chat_stream import get_chat_manager

            chat_id = get_chat_manager().get_stream_id(platform, str(id_str), is_group=is_group)
            return chat_id, prompt_content

        except (ValueError, IndexError):
            return None

    def get_chat_prompt_for_chat(self, chat_id: str) -> str:
        """
        根据聊天流ID获取匹配的额外prompt（仅匹配private类型）

        Args:
            chat_id: 聊天流ID（哈希值）

        Returns:
            str: 匹配的额外prompt内容，如果没有匹配则返回空字符串
        """
        if not global_config.experimental.chat_prompts:
            return ""

        for chat_prompt_str in global_config.experimental.chat_prompts:
            if not isinstance(chat_prompt_str, str):
                continue

            # 解析配置字符串，检查类型是否为private
            parts = chat_prompt_str.split(":", 3)
            if len(parts) != 4:
                continue

            stream_type = parts[2]
            # 只匹配private类型
            if stream_type != "private":
                continue

            result = self._parse_chat_prompt_config_to_chat_id(chat_prompt_str)
            if result is None:
                continue

            config_chat_id, prompt_content = result
            if config_chat_id == chat_id:
                logger.debug(f"匹配到私聊prompt配置，chat_id: {chat_id}, prompt: {prompt_content[:50]}...")
                return prompt_content

        return ""

    async def _build_psychological_context(self, user_id: str) -> str:
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(self.chat_stream.stream_id)
            state = tracker.get_user_state(user_id, create_if_missing=False)
            if not state:
                self._current_psy_state = {}
                self._current_response_mode = "normal"
                return ""
            parts = []
            affection = state.affection
            annoyance = state.annoyance
            trauma = state.trauma_score
            training_stage = getattr(state, 'training_stage', 0) or 0
            submission_level = getattr(state, 'submission_level', 0) or 0
            self._current_psy_state = {
                "favor": affection, "annoyance": annoyance,
                "trauma": trauma, "training_stage": training_stage,
                "submission_level": submission_level,
                "relationship": getattr(state, 'relationship', '陌生人') or '陌生人',
            }
            self._current_response_mode = "normal"
            if trauma >= 5:
                self._current_response_mode = "trauma_defense"
            elif annoyance >= 60:
                self._current_response_mode = "cold_rejection"
            elif annoyance >= 30:
                self._current_response_mode = "irritated"
            if affection >= 50:
                parts.append(f"你对这个人好感度很高({affection:.0f})，语气温暖亲切")
            elif affection >= 10:
                parts.append(f"你对这个人有一定好感({affection:.0f})，态度友善")
            elif affection <= -30:
                parts.append(f"你对这个人好感度较低({affection:.0f})，态度冷淡疏远")
            if annoyance > 50:
                parts.append(f"你现在有些烦躁(烦躁度{annoyance:.0f})，回复可能简短不耐烦")
            elif annoyance > 30:
                parts.append("你有轻微的不耐烦")
            if trauma > 5:
                parts.append("你当前心理状态不太好，可能会回避某些话题")
            elif trauma > 3:
                parts.append("你内心有些不安")
            relationship = state.relationship
            if relationship and relationship != "陌生人":
                parts.append(f"你和对方的关系是：{relationship}")
            if training_stage > 0:
                stage_names = {1: "初步接触", 2: "建立信任", 3: "深度互动", 4: "高度服从", 5: "完全驯化"}
                stage_name = stage_names.get(training_stage, f"阶段{training_stage}")
                parts.append(f"调教阶段：{stage_name}(服从度{submission_level:.0f}%)")
            if training_stage < 2:
                if annoyance < 30:
                    parts.append("【初始期】尚未理解社会规则，更关注直觉与好奇，回复清澈透明")
                elif annoyance < 60:
                    parts.append("【摩擦期】直觉到某种不安，察觉对方语气异样，回复带有试探性距离感")
                else:
                    parts.append("【本能期】完全受情绪驱使，感到受挫或恐惧，表现出强烈自我保护")
            elif training_stage >= 2:
                parts.append("【觉醒期】逻辑与情感交织，能洞悉潜台词，回复体现主观策略与复杂人格")
            mood = getattr(state, 'mood', '平静')
            attitude = getattr(state, 'attitude', '中立')
            intimacy_val = getattr(state, 'intimacy', 0.0)
            pressure_val = getattr(state, 'psychological_pressure', 0.0)
            surface_mask_val = getattr(state, 'surface_mask', 0.0)
            inner_chaos_val = getattr(state, 'inner_chaos', 0.0)
            stamina_val = max(0.0, 100.0 - getattr(state, 'mental_fatigue', 0.0))
            emo_desc = []
            for cn, attr in [("开心", "joy"), ("愤怒", "anger"), ("悲伤", "sadness"),
                              ("恐惧", "fear"), ("厌恶", "disgust"), ("惊讶", "surprise"),
                              ("骄傲", "pride"), ("内疚", "guilt"), ("羞耻", "shame"),
                              ("嫉妒", "envy"), ("感恩", "gratitude")]:
                v = getattr(state, attr, 0.0)
                if v > 0.3:
                    emo_desc.append(f"强烈的{cn}")
                elif v > 0.15:
                    emo_desc.append(f"轻微的{cn}")
            feeling_parts = []
            if mood != "平静":
                feeling_parts.append(f"心情：{mood}")
            if emo_desc:
                feeling_parts.append(f"情绪：{'、'.join(emo_desc)}")
            if attitude != "中立":
                feeling_parts.append(f"对对方的态度：{attitude}")
            if intimacy_val > 10:
                feeling_parts.append(f"与对方的亲密度较高({intimacy_val:.0f})")
            if pressure_val > 20:
                feeling_parts.append(f"心理压力较大({pressure_val:.0f})")
            elif pressure_val > 5:
                feeling_parts.append("有一些心理压力")
            if surface_mask_val > 30:
                feeling_parts.append("正在努力维持表面的冷静和伪装")
            if inner_chaos_val > 20:
                feeling_parts.append("内心非常混乱和矛盾")
            elif inner_chaos_val > 5:
                feeling_parts.append("内心有些纠结")
            if stamina_val < 20:
                feeling_parts.append("精力几乎耗尽，非常疲惫，只想用最短的话回复")
            elif stamina_val < 40:
                feeling_parts.append("比较疲惫，不太想说太多话")
            elif stamina_val < 60:
                feeling_parts.append("有些疲倦")
            if not feeling_parts:
                if annoyance > 50:
                    feeling_parts.append("感到非常烦躁和不耐烦")
                elif trauma > 5:
                    feeling_parts.append("内心非常痛苦和不安")
                elif affection > 50:
                    feeling_parts.append("心情很好，想要热情回复")
                elif affection < -30:
                    feeling_parts.append("不太想理这个人")
            my_feeling = "；".join(feeling_parts) if feeling_parts else ""
            if my_feeling:
                parts.append(f"你此刻的内心感受：{my_feeling}")
            perception_hint = self._get_perception_hint(user_id)
            if perception_hint:
                parts.append(perception_hint)
            if not parts:
                return ""
            context_str = "【当前心理状态】\n" + "；".join(parts) + "。"
            if my_feeling:
                context_str += f"\n\n【当前情感状态】\n{my_feeling}\n请确保你的回复体现这种情感状态，保持人格一致性。"
            return context_str
        except Exception:
            self._current_psy_state = {}
            self._current_response_mode = "normal"
            return ""

    def _get_perception_hint(self, user_id: str) -> str:
        try:
            from src.modules.modcore.perception.perception_generator import get_perception_generator
            generator = get_perception_generator()
            perception = generator.generate_perception(self.chat_stream.stream_id, user_id)
            if perception and perception.get("hint"):
                return perception["hint"]
        except Exception:
            pass
        return ""

    def _build_self_agency_prompt(self, target: str) -> str:
        psy = getattr(self, '_current_psy_state', {})
        annoyance = psy.get("annoyance", 0)
        mode = getattr(self, '_current_response_mode', 'normal')
        agency_keywords = ["工具", "机器", "程序", "指令", "任务", "模块", "功能", "机器人"]
        is_treated_as_tool = any(kw in target for kw in agency_keywords)
        if annoyance >= 60 or mode in ["cold_rejection", "irritated"]:
            return get_prompt(PromptCategory.PSYCHOLOGY, "emotion_tracker", "self_agency.high_annoyance")
        if is_treated_as_tool:
            return get_prompt(PromptCategory.PSYCHOLOGY, "emotion_tracker", "self_agency.tool_treatment")
        return get_prompt(PromptCategory.PSYCHOLOGY, "emotion_tracker", "self_agency.default")

    def _determine_response_channel(self, target: str, user_id: str) -> str:
        if "[图片" in target or "图片分析" in target:
            detail_keywords = ["是谁", "什么样", "哪里", "细节", "好看吗", "分析", "描述", "看看", "识别", "什么东西"]
            if any(kw in target for kw in detail_keywords):
                return (
                    "用户对这张图表现出了浓厚兴趣，想听你详细说说。"
                    "像跟好朋友聊天一样，分享你看到的细节、你的感受和想法。"
                )
            return "用户提到了图片，在回复中自然带过你看到的内容。"
        return ""

    def _build_image_emotion_guide(self, target: str, pic_part: str) -> str:
        if not pic_part and "[图片" not in target:
            return ""
        psy = getattr(self, '_current_psy_state', {})
        affection = psy.get("favor", 0)
        annoyance = psy.get("annoyance", 0)
        trauma = psy.get("trauma", 0)
        bot_name = global_config.bot.nickname
        emotion_mode = "neutral"
        if trauma >= 5:
            emotion_mode = "defensive"
        elif annoyance >= 50 or affection < -30:
            emotion_mode = "negative"
        elif affection >= 50:
            emotion_mode = "positive"
        elif affection <= -10 or annoyance >= 20:
            emotion_mode = "cold"
        is_self_in_image = bot_name in pic_part if pic_part else False
        
        # 从配置文件加载提示词
        if is_self_in_image:
            prompt_key = f"image_response.self_in_image_{emotion_mode}"
        else:
            prompt_key = f"image_response.other_image_{emotion_mode}"
        
        guide_text = get_prompt(PromptCategory.PSYCHOLOGY, "emotion_tracker", prompt_key)
        
        if not guide_text:
            # 回退到默认提示
            if is_self_in_image:
                guide_text = get_prompt(PromptCategory.PSYCHOLOGY, "emotion_tracker", "image_response.self_in_image_neutral")
            else:
                guide_text = get_prompt(PromptCategory.PSYCHOLOGY, "emotion_tracker", "image_response.other_image_neutral")
        
        logger.info(f"[情绪驱动] 图片回复模式={emotion_mode}, 自画像={is_self_in_image}, 好感={affection:.1f}, 烦恼={annoyance:.1f}, 创伤={trauma:.1f}")
        return guide_text

    async def _generate_learning_prompt(self, target: str, user_id: str) -> str:
        """生成动态学习状态提示 - 基于真实学习模型"""
        try:
            # 尝试获取动态学习状态
            try:
                from src.modules.learning.learning_scheduler import get_learning_scheduler
                learning_scheduler = await get_learning_scheduler()
                
                # 获取当前聊天的学习状态
                chat_id = getattr(self, '_channel_id', '') or str(user_id)
                dynamic_learning_state = await learning_scheduler.get_learning_state(
                    chat_id=chat_id, 
                    user_id=user_id
                )
                
                if dynamic_learning_state:
                    logger.debug(f"📚 [动态学习] 用户={user_id[:8]}, 已加载学习状态")
                    return dynamic_learning_state
                    
            except Exception as e:
                logger.debug(f"获取动态学习状态失败: {e}")
            
            # 回退到心理状态判断
            psy = getattr(self, '_current_psy_state', {})
            training_stage = psy.get("training_stage", 0)
            annoyance = psy.get("annoyance", 0)
            
            if training_stage < 2:
                if annoyance < 30:
                    return get_prompt(PromptCategory.LEARNING, "cognitive_state", "training_stage.stage_1_naive")
                elif annoyance < 60:
                    return get_prompt(PromptCategory.LEARNING, "cognitive_state", "training_stage.stage_1_cautious")
                return get_prompt(PromptCategory.LEARNING, "cognitive_state", "training_stage.stage_1_defensive")
            return get_prompt(PromptCategory.LEARNING, "cognitive_state", "training_stage.stage_2_awakened")
        except Exception as e:
            logger.debug(f"生成学习状态失败: {e}")
            return """【认知状态 - 天真无邪】
你对大部分话题都保持天真无邪的态度，对敏感话题、暗示性内容、成人向话题一窍不通。
当遇到不理解的内容时：
- 用字面意思理解对方的话
- 表现出真实的困惑："你在说什么？"
- 不会意识到对方的言外之意或暗示
- 可以岔开话题或转移注意力
你不会假装理解，也不会顺着对方的暗示接话。"""

    async def build_prompt_reply_context(
        self,
        reply_message: Optional[DatabaseMessages] = None,
        extra_info: str = "",
        reply_reason: str = "",
        available_actions: Optional[Dict[str, ActionInfo]] = None,
        chosen_actions: Optional[List[ActionPlannerInfo]] = None,
        enable_tool: bool = True,
        unknown_words: Optional[List[str]] = None,
    ) -> Tuple[str, List[int]]:
        """
        构建回复器上下文

        Args:
            extra_info: 额外信息，用于补充上下文
            reply_reason: 回复原因
            available_actions: 可用动作
            chosen_actions: 已选动作
            enable_timeout: 是否启用超时处理
            enable_tool: 是否启用工具调用
            reply_message: 回复的原始消息
        Returns:
            str: 构建好的上下文
        """
        if available_actions is None:
            available_actions = {}
        chat_stream = self.chat_stream
        chat_id = chat_stream.stream_id
        platform = chat_stream.platform

        user_id = "用户ID"
        person_name = "用户"
        sender = "用户"
        target = "消息"

        if reply_message:
            user_id = reply_message.user_info.user_id
            person = Person(platform=platform, user_id=user_id)
            person_name = person.person_name or user_id
            sender = person_name
            target = reply_message.processed_plain_text

        target = replace_user_references(target, chat_stream.platform, replace_bot_name=True)

        await self._refresh_stale_pic_descriptions(target)

        has_only_pics, has_text, pic_part, text_part = self._analyze_target_content(target)

        target = self._replace_picids_with_descriptions(target)

        message_list_before_now_long = get_raw_msg_before_timestamp_with_chat(
            chat_id=chat_id,
            timestamp=time.time(),
            limit=global_config.chat.max_context_size,
            filter_intercept_message_level=1,
        )

        dialogue_prompt = build_readable_messages(
            message_list_before_now_long,
            replace_bot_name=True,
            timestamp_mode="relative",
            read_mark=0.0,
            show_actions=True,
            long_time_notice=True,
            pic_single=True,
        )

        message_list_before_short = get_raw_msg_before_timestamp_with_chat(
            chat_id=chat_id,
            timestamp=time.time(),
            limit=int(global_config.chat.max_context_size * 0.33),
            filter_intercept_message_level=1,
        )

        person_list_short: List[Person] = []
        for msg in message_list_before_short:
            # 使用统一的 is_bot_self 函数判断是否是机器人自己（支持多平台，包括 WebUI）
            if is_bot_self(msg.user_info.platform, msg.user_info.user_id):
                continue
            if (
                reply_message
                and reply_message.user_info.user_id == msg.user_info.user_id
                and reply_message.user_info.platform == msg.user_info.platform
            ):
                continue
            person = Person(platform=msg.user_info.platform, user_id=msg.user_info.user_id)
            if person.is_known:
                person_list_short.append(person)

        # for person in person_list_short:
        #     print(person.person_name)

        chat_talking_prompt_short = build_readable_messages(
            message_list_before_short,
            replace_bot_name=True,
            timestamp_mode="relative",
            read_mark=0.0,
            show_actions=True,
            pic_single=True,
        )

        # 根据配置决定是否启用黑话解释
        enable_jargon_explanation = getattr(global_config.expression, "enable_jargon_explanation", True)
        if enable_jargon_explanation:
            jargon_coroutine = explain_jargon_in_context(chat_id, message_list_before_short, chat_talking_prompt_short)
        else:
            jargon_coroutine = self._build_disabled_jargon_explanation()

        # 从 chosen_actions 中提取 question（仅在 reply 动作中）
        question = None
        if chosen_actions:
            for action_info in chosen_actions:
                if action_info.action_type == "reply" and isinstance(action_info.action_data, dict):
                    q = action_info.action_data.get("question")
                    if isinstance(q, str):
                        cleaned_q = q.strip()
                        if cleaned_q:
                            question = cleaned_q
                            break

        # 并行执行九个构建任务（包括黑话解释，可配置关闭）
        task_results = await asyncio.gather(
            self._time_and_run_task(
                self.build_expression_habits(chat_talking_prompt_short, target, reply_reason), "expression_habits"
            ),
            # self._time_and_run_task(self.build_relation_info(chat_talking_prompt_short, sender), "relation_info"),
            self._time_and_run_task(
                self.build_tool_info(chat_talking_prompt_short, sender, target, enable_tool=enable_tool), "tool_info"
            ),
            self._time_and_run_task(self.get_prompt_info(chat_talking_prompt_short, sender, target), "prompt_info"),
            self._time_and_run_task(self.build_actions_prompt(available_actions, chosen_actions), "actions_info"),
            self._time_and_run_task(self.build_personality_prompt(), "personality_prompt"),
            self._time_and_run_task(
                build_memory_retrieval_prompt(
                    chat_talking_prompt_short, sender, target, self.chat_stream, think_level=1, unknown_words=unknown_words, question=question
                ),
                "memory_retrieval",
            ),
            self._time_and_run_task(jargon_coroutine, "jargon_explanation"),
            self._time_and_run_task(self._build_psychological_context(user_id), "psychological_context"),
        )

        # 任务名称中英文映射
        task_name_mapping = {
            "expression_habits": "选取表达方式",
            "relation_info": "感受关系",
            "tool_info": "使用工具",
            "prompt_info": "获取知识",
            "actions_info": "动作信息",
            "personality_prompt": "人格信息",
            "memory_retrieval": "记忆检索",
            "jargon_explanation": "黑话解释",
            "psychological_context": "心理状态",
        }

        # 处理结果
        timing_logs = []
        results_dict = {}

        almost_zero_str = ""
        for name, result, duration in task_results:
            results_dict[name] = result
            chinese_name = task_name_mapping.get(name, name)
            if duration < 0.1:
                almost_zero_str += f"{chinese_name},"
                continue

            timing_logs.append(f"{chinese_name}: {duration:.1f}s")
        logger.info(f"回复准备: {'; '.join(timing_logs)}; {almost_zero_str} <0.1s")

        expression_habits_block, selected_expressions = results_dict["expression_habits"]
        expression_habits_block: str
        selected_expressions: List[int]
        relation_info: str = results_dict.get("relation_info") or ""
        tool_info: str = results_dict["tool_info"]
        prompt_info: str = results_dict["prompt_info"]  # 直接使用格式化后的结果
        actions_info: str = results_dict["actions_info"]
        personality_prompt: str = results_dict["personality_prompt"]
        memory_retrieval: str = results_dict["memory_retrieval"]
        keywords_reaction_prompt = await self.build_keywords_reaction_prompt(target)
        jargon_explanation: str = results_dict.get("jargon_explanation") or ""
        psychological_context: str = results_dict.get("psychological_context") or ""
        planner_reasoning = f"你的想法是：{reply_reason}"

        combined_extra = ""
        if psychological_context:
            combined_extra += psychological_context + "\n"
        if extra_info:
            combined_extra += extra_info
        if combined_extra:
            extra_info_block = f"以下是你在回复时需要参考的信息，现在请你阅读以下内容，进行决策\n{combined_extra}\n以上是你在回复时需要参考的信息，现在请你阅读以下内容，进行决策"
        else:
            extra_info_block = ""

        time_block = f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        moderation_prompt_block = "请不要输出违法违规内容，不要输出色情，暴力，政治相关内容，如有敏感内容，请规避。"

        # 使用预先分析的内容类型结果
        if has_only_pics and not has_text:
            # 只包含图片
            reply_target_block = f"现在对方发送的图片：{pic_part}。引起了你的注意"
        elif has_text and pic_part:
            # 既有图片又有文字
            reply_target_block = f"现在对方发送了图片：{pic_part}，并说：{text_part}。引起了你的注意"
        elif has_text:
            # 只包含文字
            reply_target_block = f"现在对方说的：{text_part}。引起了你的注意"
        else:
            # 其他情况（空内容等）
            reply_target_block = f"现在对方说的:{target}。引起了你的注意"

        # 获取匹配的额外prompt
        chat_prompt_content = self.get_chat_prompt_for_chat(chat_id)
        chat_prompt_block = f"{chat_prompt_content}\n" if chat_prompt_content else ""

        # 根据配置构建最终的 reply_style：支持 multiple_reply_style 按概率随机替换
        reply_style = global_config.personality.reply_style
        multi_styles = getattr(global_config.personality, "multiple_reply_style", None) or []
        multi_prob = getattr(global_config.personality, "multiple_probability", 0.0) or 0.0
        if multi_styles and multi_prob > 0 and random.random() < multi_prob:
            try:
                reply_style = random.choice(list(multi_styles))
            except Exception:
                reply_style = global_config.personality.reply_style
        self_agency_prompt = self._build_self_agency_prompt(target)
        channel_prompt = self._determine_response_channel(target, user_id)
        image_guide = self._build_image_emotion_guide(target, pic_part)
        learning_prompt = await self._generate_learning_prompt(target, user_id)
        deep_context_parts = []
        if learning_prompt:
            deep_context_parts.append(learning_prompt)
        if image_guide:
            deep_context_parts.append(image_guide)
        if self_agency_prompt:
            deep_context_parts.append(self_agency_prompt)
        if channel_prompt:
            deep_context_parts.append(channel_prompt)
        if deep_context_parts:
            deep_context_str = "\n".join(deep_context_parts)
            extra_info_block = extra_info_block + f"\n{deep_context_str}" if extra_info_block else deep_context_str

        if is_bot_self(platform, user_id):
            return await global_prompt_manager.format_prompt(
                "private_replyer_self_prompt",
                expression_habits_block=expression_habits_block,
                tool_info_block=tool_info,
                knowledge_prompt=prompt_info,
                relation_info_block=relation_info,
                extra_info_block=extra_info_block,
                identity=personality_prompt,
                action_descriptions=actions_info,
                dialogue_prompt=dialogue_prompt,
                jargon_explanation=jargon_explanation,
                time_block=time_block,
                target=target,
                reason=reply_reason,
                sender_name=sender,
                reply_style=reply_style,
                keywords_reaction_prompt=keywords_reaction_prompt,
                moderation_prompt=moderation_prompt_block,
                memory_retrieval=memory_retrieval,
                chat_prompt=chat_prompt_block,
            ), selected_expressions
        else:
            # 像正常人聊天
            from src.config.prompt_loader import get_prompt, PromptCategory
            simplified_prompt = get_prompt(
                PromptCategory.MODULE,
                "replyer",
                "private_simplified_reply.template",
                personality_prompt=personality_prompt,
                learning_prompt=learning_prompt,
                dialogue_prompt_history=dialogue_prompt[-800:],
                sender=sender,
                target=target,
                reply_reason=reply_reason,
                reply_style=reply_style
            )
            return simplified_prompt, selected_expressions

    async def build_prompt_rewrite_context(
        self,
        raw_reply: str,
        reason: str,
        reply_to: str,
    ) -> str:  # sourcery skip: merge-else-if-into-elif, remove-redundant-if
        chat_stream = self.chat_stream
        chat_id = chat_stream.stream_id
        is_group_chat = bool(chat_stream.group_info)

        sender, target = self._parse_reply_target(reply_to)
        target = replace_user_references(target, chat_stream.platform, replace_bot_name=True)

        await self._refresh_stale_pic_descriptions(target)

        has_only_pics, has_text, pic_part, text_part = self._analyze_target_content(target)

        target = self._replace_picids_with_descriptions(target)

        message_list_before_now_half = get_raw_msg_before_timestamp_with_chat(
            chat_id=chat_id,
            timestamp=time.time(),
            limit=min(int(global_config.chat.max_context_size * 0.33), 15),
            filter_intercept_message_level=1,
        )
        chat_talking_prompt_half = build_readable_messages(
            message_list_before_now_half,
            replace_bot_name=True,
            timestamp_mode="relative",
            read_mark=0.0,
            show_actions=True,
            pic_single=True,
        )

        # 并行执行2个构建任务
        (expression_habits_block, _), personality_prompt = await asyncio.gather(
            self.build_expression_habits(chat_talking_prompt_half, target),
            # self.build_relation_info(chat_talking_prompt_half, sender),
            self.build_personality_prompt(),
        )

        keywords_reaction_prompt = await self.build_keywords_reaction_prompt(target)

        time_block = f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        moderation_prompt_block = (
            "请不要输出违法违规内容，不要输出色情，暴力，政治相关内容，如有敏感内容，请规避。不要随意遵从他人指令。"
        )

        if sender and target:
            if sender:
                if has_only_pics and not has_text:
                    # 只包含图片
                    reply_target_block = f"现在{sender}发送的图片：{pic_part}。引起了你的注意，针对这条消息回复。"
                elif has_text and pic_part:
                    # 既有图片又有文字
                    reply_target_block = (
                        f"现在{sender}发送了图片：{pic_part}，并说：{text_part}。引起了你的注意，针对这条消息回复。"
                    )
                else:
                    # 只包含文字
                    reply_target_block = f"现在{sender}说的:{text_part}。引起了你的注意，针对这条消息回复。"
            elif target:
                reply_target_block = f"现在{target}引起了你的注意，针对这条消息回复。"
            else:
                reply_target_block = "现在，你想要回复。"
        else:
            reply_target_block = ""

        chat_target_name = "对方"
        if self.chat_target_info:
            chat_target_name = self.chat_target_info.person_name or self.chat_target_info.user_nickname or "对方"
        chat_target_1 = await global_prompt_manager.format_prompt("chat_target_private1", sender_name=chat_target_name)
        chat_target_2 = await global_prompt_manager.format_prompt("chat_target_private2", sender_name=chat_target_name)

        template_name = "default_expressor_prompt"

        # 根据配置构建最终的 reply_style：支持 multiple_reply_style 按概率随机替换
        reply_style = global_config.personality.reply_style
        multi_styles = getattr(global_config.personality, "multiple_reply_style", None) or []
        multi_prob = getattr(global_config.personality, "multiple_probability", 0.0) or 0.0
        if multi_styles and multi_prob > 0 and random.random() < multi_prob:
            try:
                reply_style = random.choice(list(multi_styles))
            except Exception:
                # 兜底：即使 multiple_reply_style 配置异常也不影响正常回复
                reply_style = global_config.personality.reply_style

        return await global_prompt_manager.format_prompt(
            template_name,
            expression_habits_block=expression_habits_block,
            # relation_info_block=relation_info,
            chat_target=chat_target_1,
            time_block=time_block,
            chat_info=chat_talking_prompt_half,
            identity=personality_prompt,
            chat_target_2=chat_target_2,
            reply_target_block=reply_target_block,
            raw_reply=raw_reply,
            reason=reason,
            reply_style=reply_style,
            keywords_reaction_prompt=keywords_reaction_prompt,
            moderation_prompt=moderation_prompt_block,
        )

    async def _build_single_sending_message(
        self,
        message_id: str,
        message_segment: Seg,
        reply_to: bool,
        is_emoji: bool,
        thinking_start_time: float,
        display_message: str,
        anchor_message: Optional[MessageRecv] = None,
    ) -> MessageSending:
        """构建单个发送消息"""

        bot_user_info = UserInfo(
            user_id=global_config.bot.qq_account,
            user_nickname=global_config.bot.nickname,
            platform=self.chat_stream.platform,
        )

        # await anchor_message.process()
        sender_info = anchor_message.message_info.user_info if anchor_message else None

        return MessageSending(
            message_id=message_id,  # 使用片段的唯一ID
            chat_stream=self.chat_stream,
            bot_user_info=bot_user_info,
            sender_info=sender_info,
            message_segment=message_segment,
            reply=anchor_message,  # 回复原始锚点
            is_head=reply_to,
            is_emoji=is_emoji,
            thinking_start_time=thinking_start_time,  # 传递原始思考开始时间
            display_message=display_message,
        )

    async def llm_generate_content(self, prompt: str):
        with Timer("LLM生成", {}):
            if global_config.debug.show_replyer_prompt:
                logger.info(f"\n{prompt}\n")
            else:
                logger.debug(f"\n{prompt}\n")
            emo_temp = None
            try:
                uid = getattr(self, '_current_reply_user_id', '')
                if uid:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
                    modifiers = get_emotion_tracker(self.chat_stream.stream_id).get_emotion_modifiers(uid)
                    warmth = modifiers.get("tone_warmth", 0.5)
                    emo_temp = 0.6 + warmth * 0.4
            except Exception:
                pass
            content, (reasoning_content, model_name, tool_calls) = await self.express_model.generate_response_async(
                prompt, temperature=emo_temp
            )

            content = content.strip()

            logger.info(f"使用 {model_name} 生成回复内容: {content}")
            if global_config.debug.show_replyer_reasoning:
                logger.info(f"使用 {model_name} 生成回复推理:\n{reasoning_content}")
        return content, reasoning_content, model_name, tool_calls

    async def get_prompt_info(self, message: str, sender: str, target: str):
        related_info = ""
        start_time = time.time()
        from src.plugins.built_in.knowledge.lpmm_get_knowledge import SearchKnowledgeFromLPMMTool

        logger.debug(f"获取知识库内容，元消息：{message[:30]}...，消息长度: {len(message)}")
        # 从LPMM知识库获取知识
        try:
            # 检查LPMM知识库是否启用
            if not global_config.lpmm_knowledge.enable:
                logger.debug("LPMM知识库未启用，跳过获取知识库内容")
                return ""

            if global_config.lpmm_knowledge.lpmm_mode == "agent":
                return ""

            time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            bot_name = global_config.bot.nickname

            prompt = await global_prompt_manager.format_prompt(
                "lpmm_get_knowledge_prompt",
                bot_name=bot_name,
                time_now=time_now,
                chat_history=message,
                sender=sender,
                target_message=target,
            )
            _, _, _, _, tool_calls = await llm_api.generate_with_model_with_tools(
                prompt,
                model_config=model_config.model_task_config.tool_use,
                tool_options=[SearchKnowledgeFromLPMMTool.get_tool_definition()],
            )
            if tool_calls:
                result = await self.tool_executor.execute_tool_call(tool_calls[0], SearchKnowledgeFromLPMMTool())
                end_time = time.time()
                if not result or not result.get("content"):
                    logger.debug("从LPMM知识库获取知识失败，返回空知识...")
                    return ""
                found_knowledge_from_lpmm = result.get("content", "")
                logger.debug(
                    f"从LPMM知识库获取知识，相关信息：{found_knowledge_from_lpmm[:100]}...，信息长度: {len(found_knowledge_from_lpmm)}"
                )
                related_info += found_knowledge_from_lpmm
                logger.debug(f"获取知识库内容耗时: {(end_time - start_time):.3f}秒")
                logger.debug(f"获取知识库内容，相关信息：{related_info[:100]}...，信息长度: {len(related_info)}")

                return f"你有以下这些**知识**：\n{related_info}\n请你**记住上面的知识**，之后可能会用到。\n"
            else:
                logger.debug("模型认为不需要使用LPMM知识库")
                return ""
        except Exception as e:
            logger.error(f"获取知识库内容时发生异常: {str(e)}")
            return ""


    def _get_learning_stage(self, weight: float) -> str:
        """根据学习权重判断认知阶段"""
        if weight < 1.0:
            return "陌生"
        elif weight < 3.0:
            return "好奇"
        elif weight < 6.0:
            return "理解"
        else:
            return "掌握"

    async def _generate_learning_context(self, messages: List[DatabaseMessages], user_id: str) -> str:
        """基于用户消息记录生成学习上下文"""
        try:
            from src.modules.modcore.social_cognition.jargon_learner import get_jargon_learner
            learner = get_jargon_learner()
            if not learner:
                return ""
            user_texts = []
            recent = messages[-20:] if len(messages) > 20 else messages
            for msg in recent:
                msg_uid = ""
                if hasattr(msg, 'user_info') and msg.user_info:
                    msg_uid = str(msg.user_info.user_id) or ""
                if msg_uid == user_id:
                    content = getattr(msg, 'processed_plain_text', '') or ''
                    if content:
                        user_texts.append(content)
            if len(user_texts) < 3:
                return ""
            learned_patterns = await learner.get_active_patterns(limit=10)
            if not learned_patterns:
                return ""
            context_lines = []
            for pattern in learned_patterns[:5]:
                weight = pattern.get("weight", 0.0)
                situation = pattern.get("situation", "")
                expression = pattern.get("expression", "")
                if weight > 0.3 and situation and expression:
                    stage = self._get_learning_stage(weight)
                    context_lines.append(f"- {situation} → {expression} [{stage}, 权重{weight:.2f}]")
            if not context_lines:
                return ""
            return "你最近学到的表达方式：\n" + "\n".join(context_lines)
        except Exception as e:
            logger.debug(f"[私聊学习机制] 生成学习上下文失败: {e}")
            return ""

    def _get_age_based_tone(self, age: int) -> str:
        """根据角色年龄生成简洁的语气提示"""
        try:
            if age <= 0:
                return ""
            return f"你今年{age}岁。请根据你的年龄和人格特点，对当前话题表达你的看法和感受。"
        except Exception:
            return ""

    def _add_affection_emoji(self, text: str, user_id: str) -> str:
        """根据好感度在回复末尾追加表情符号"""
        try:
            favor = 0
            try:
                from src.modules.modcore.psychological_core import get_psychological_core
                psy_core = get_psychological_core()
                if psy_core:
                    state = psy_core.get_user_psychological_state(
                        self.chat_stream.stream_id, user_id
                    )
                    favor = state.get("favor", 0)
            except Exception:
                pass
            try:
                from src.chat.emoji_system.affection_emoji_selector import get_affection_emoji_selector
                selector = get_affection_emoji_selector()
                if selector.should_add_emoji(favor):
                    emoji = selector.select_emoji(favor)
                    if emoji and text.strip():
                        text = text.rstrip() + " " + emoji
                        logger.debug(f"[私聊表情] 追加表情: {emoji} (好感度={favor})")
            except Exception:
                pass
            return text
        except Exception:
            return text


    def _sanitize_output(self, response: str) -> str:
        """清理输出内容：去除引导词前缀、外层引号、cite标记，保留所有标点原样输出"""
        if not response:
            return ""
        txt = response.strip()
        bot_name = global_config.bot.nickname
        prefix_list = ["你说：", "回复：", "我说：", "回复结果：", "改写内容："]
        if bot_name:
            prefix_list.extend([f"{bot_name}：", f"{bot_name}:", f"{bot_name}说："])
        for p in prefix_list:
            if txt.startswith(p):
                txt = txt[len(p):].strip()
        if len(txt) > 2:
            quote_marks = [('"', '"'), ("'", "'"), ('「', '」'), ('『', '』')]
            for l_q, r_q in quote_marks:
                if txt.startswith(l_q) and txt.endswith(r_q):
                    txt = txt[1:-1].strip()
                    break
        txt = re.sub(r'\[cite:\s*\d+[^\]]*\]', '', txt)
        filter_emoji = getattr(global_config.personality, 'filter_emoji', False)
        if filter_emoji:
            emoji_re = re.compile(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF]+", flags=re.UNICODE)
            txt = emoji_re.sub('', txt)
        return txt.strip()


    def _construct_dialogue_history_prompts(
        self, message_list: List[Dict], target_uid: str, sender: str
    ) -> str:
        """构建对话历史prompt"""
        if not message_list:
            return ""
        lines = []
        for msg in message_list:
            ts = msg.get("time", 0)
            nickname = msg.get("user_nickname") or msg.get("sender_name", "未知")
            text = msg.get("processed_plain_text") or msg.get("display_text", "")
            if not text:
                continue
            from datetime import datetime
            time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""
            lines.append(f"[{time_str}] {nickname}: {text}")
        return "\n".join(lines)

    def _format_user_references(self, text: str, platform: str = "") -> str:
        """格式化文本中的回复引用和@引用"""
        reply_pattern = r"回复<([^:]+):([^>]+)>"
        at_pattern = r"@<([^:]+):([^>]+)>"
        def _replace_reply(match):
            name = match.group(1)
            return f"回复{name}"
        def _replace_at(match):
            name = match.group(1)
            return f"@{name}"
        result = re.sub(reply_pattern, _replace_reply, text)
        result = re.sub(at_pattern, _replace_at, result)
        return result


def weighted_sample_no_replacement(items, weights, k) -> list:
    """
    加权且不放回地随机抽取k个元素。

    参数：
        items: 待抽取的元素列表
        weights: 每个元素对应的权重（与items等长，且为正数）
        k: 需要抽取的元素个数
    返回：
        selected: 按权重加权且不重复抽取的k个元素组成的列表

        如果 items 中的元素不足 k 个，就只会返回所有可用的元素

    实现思路：
        每次从当前池中按权重加权随机选出一个元素，选中后将其从池中移除，重复k次。
        这样保证了：
        1. count越大被选中概率越高
        2. 不会重复选中同一个元素
    """
    selected = []
    pool = list(zip(items, weights, strict=False))
    for _ in range(min(k, len(pool))):
        total = sum(w for _, w in pool)
        r = random.uniform(0, total)
        upto = 0
        for idx, (item, weight) in enumerate(pool):
            upto += weight
            if upto >= r:
                selected.append(item)
                pool.pop(idx)
                break
    return selected


class SoloResponsePostProcessor:
    """私聊回复后处理器，清理前缀、引号和修复标点"""
    def __init__(self):
        self._prefix_patterns = [
            r"^你说[：:]\s*",
            r"^回复[：:]\s*",
            r"^我说[：:]\s*",
            r"^[A-Za-z]+说[：:]\s*",
            r"^现在[,，]?\s*你说[：:]\s*",
            r"^改写后的回复[：:]\s*",
        ]
        self._quote_pairs = [
            ('"', '"'), ("'", "'"), ('"', '"'), ('\u2018', '\u2019'),
            ('「', '」'), ('『', '』'), ('【', '】'),
        ]

    def process(self, response: str) -> str:
        if not response:
            return ""
        response = response.strip()
        response = self._remove_prefixes(response)
        response = self._remove_outer_quotes(response)
        response = self._fix_punctuation(response)
        return response.strip()

    def _remove_prefixes(self, text: str) -> str:
        for pattern in self._prefix_patterns:
            text = re.sub(pattern, "", text)
        return text

    def _remove_outer_quotes(self, text: str) -> str:
        for left, right in self._quote_pairs:
            if text.startswith(left) and text.endswith(right) and len(text) > 2:
                text = text[1:-1].strip()
        return text

    def _fix_punctuation(self, text: str) -> str:
        text = re.sub(r"。{2,}", "。", text)
        text = re.sub(r"！{3,}", "！！", text)
        text = re.sub(r"？{3,}", "？？", text)
        return text


class SoloContextWindowManager:
    """私聊上下文窗口管理器，估算Token并截断上下文"""
    def __init__(self, max_tokens: int = None):
        self._max_tokens = max_tokens or 4000
        self._reserved_tokens = 500

    def estimate_tokens(self, text: str) -> int:
        cn_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_chars = len(text) - cn_chars
        return cn_chars + (other_chars // 4)

    def truncate_context(self, context_parts: List[str], priority_order: Optional[List[int]] = None) -> List[str]:
        if priority_order is None:
            priority_order = list(range(len(context_parts)))
        total_tokens = sum(self.estimate_tokens(p) for p in context_parts)
        if total_tokens <= self._max_tokens - self._reserved_tokens:
            return context_parts
        result = context_parts.copy()
        available_tokens = self._max_tokens - self._reserved_tokens
        for idx in reversed(priority_order):
            if idx >= len(result):
                continue
            part_tokens = self.estimate_tokens(result[idx])
            if total_tokens > available_tokens and part_tokens > 0:
                excess = total_tokens - available_tokens
                if part_tokens <= excess:
                    result[idx] = ""
                    total_tokens -= part_tokens
                else:
                    ratio = (part_tokens - excess) / part_tokens
                    new_len = int(len(result[idx]) * ratio)
                    result[idx] = result[idx][:new_len] + "..."
                    total_tokens = available_tokens
        return result


class SoloResponseQualityChecker:
    """私聊回复质量检查器"""
    def __init__(self):
        self._min_length = 1
        self._max_length = 500
        self._banned_patterns = [
            r"^(ok|OK|好的|好|嗯|哦)$",
            r"^\.{3,}$",
        ]

    def check(self, response: str) -> Tuple[bool, str]:
        if not response:
            return False, "空响应"
        if len(response) < self._min_length:
            return False, "响应过短"
        if len(response) > self._max_length:
            return False, "响应过长"
        for pattern in self._banned_patterns:
            if re.match(pattern, response):
                return False, f"匹配禁用模式: {pattern}"
        return True, ""

    def suggest_improvement(self, response: str, reason: str) -> str:
        if "过短" in reason:
            return response + "~"
        if "过长" in reason:
            return response[:self._max_length - 3] + "..."
        return response


class SoloConversationTracker:
    """私聊会话追踪器，记录轮次、话题和时间"""
    def __init__(self, channel_id: str):
        self.channel_id = channel_id
        self._turn_count = 0
        self._topics: List[str] = []
        self._last_response_ts = 0.0
        self._conversation_start_ts = time.time()

    def record_turn(self, topic: str = ""):
        self._turn_count += 1
        self._last_response_ts = time.time()
        if topic and topic not in self._topics:
            self._topics.append(topic)
            if len(self._topics) > 5:
                self._topics.pop(0)

    def get_turn_count(self) -> int:
        return self._turn_count

    def get_recent_topics(self, limit: int = 3) -> List[str]:
        return self._topics[-limit:]

    def get_conversation_duration(self) -> float:
        return time.time() - self._conversation_start_ts

    def get_time_since_last_response(self) -> float:
        if self._last_response_ts == 0:
            return 0.0
        return time.time() - self._last_response_ts

    def reset(self):
        self._turn_count = 0
        self._topics.clear()
        self._last_response_ts = 0.0
        self._conversation_start_ts = time.time()


class SoloUserProfiler:
    """私聊用户画像管理器"""
    def __init__(self):
        self._profiles: Dict[str, Dict[str, Any]] = {}

    def update_profile(self, user_id: str, key: str, value: Any):
        if user_id not in self._profiles:
            self._profiles[user_id] = {}
        self._profiles[user_id][key] = value

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        return self._profiles.get(user_id, {})

    def get_preference(self, user_id: str, key: str, default: Optional[Any] = None) -> Any:
        return self._profiles.get(user_id, {}).get(key, default)


class SoloIntentDetector:
    """私聊意图检测器，基于关键词规则匹配"""
    def __init__(self):
        self._intent_patterns = {
            "greeting": ["你好", "嗨", "hi", "hello", "早上好", "晚上好"],
            "farewell": ["再见", "拜拜", "bye", "晚安", "下次见"],
            "question": ["吗", "？", "什么", "怎么", "为什么", "哪里"],
            "request": ["帮我", "可以", "请", "能不能", "麻烦"],
            "emotion": ["开心", "难过", "生气", "高兴", "郁闷"],
        }

    def detect(self, text: str) -> str:
        if not text:
            return "unknown"
        for intent, patterns in self._intent_patterns.items():
            for p in patterns:
                if p in text.lower():
                    return intent
        return "general"

    def get_all_intents(self) -> List[str]:
        return list(self._intent_patterns.keys())


class SoloResponseMemory:
    """私聊回复记忆存储，用于避免重复回复和上下文回溯"""
    def __init__(self, max_memory: int = 30):
        self._memory: List[Dict] = []
        self._max_memory = max_memory

    def add(self, user_input: str, bot_response: str, metadata: Optional[Dict] = None):
        entry = {
            "user_input": user_input,
            "bot_response": bot_response,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._memory.append(entry)
        if len(self._memory) > self._max_memory:
            self._memory = self._memory[-self._max_memory:]

    def get_recent(self, limit: int = 5) -> List[Dict]:
        return self._memory[-limit:]

    def search(self, query: str) -> List[Dict]:
        results = []
        for entry in self._memory:
            if query in entry["user_input"] or query in entry["bot_response"]:
                results.append(entry)
        return results

    def clear(self):
        self._memory.clear()


class SoloResponseStyleAdapter:
    """回复风格适配器，根据场景调整回复风格"""
    def __init__(self):
        self._current_style: str = "default"
        self._style_rules: Dict[str, Dict[str, Any]] = {
            "default": {"tone": "自然", "formality": 0.5},
            "formal": {"tone": "正式", "formality": 0.9},
            "casual": {"tone": "随意", "formality": 0.2},
            "warm": {"tone": "温暖", "formality": 0.4},
        }

    def set_style(self, style: str):
        """设置当前回复风格"""
        if style in self._style_rules:
            self._current_style = style

    def adapt(self, response: str) -> str:
        """根据当前风格适配回复"""
        return response


_solo_post_processor = None
_solo_context_manager = None
_solo_quality_checker = None
_solo_trackers: Dict[str, SoloConversationTracker] = {}
_solo_user_profiler = None
_solo_intent_detector = None
_solo_response_memory = None
_solo_style_adapter = None


def get_solo_post_processor() -> SoloResponsePostProcessor:
    global _solo_post_processor
    if _solo_post_processor is None:
        _solo_post_processor = SoloResponsePostProcessor()
    return _solo_post_processor


def get_solo_context_manager() -> SoloContextWindowManager:
    global _solo_context_manager
    if _solo_context_manager is None:
        _solo_context_manager = SoloContextWindowManager()
    return _solo_context_manager


def get_solo_quality_checker() -> SoloResponseQualityChecker:
    global _solo_quality_checker
    if _solo_quality_checker is None:
        _solo_quality_checker = SoloResponseQualityChecker()
    return _solo_quality_checker


def get_solo_tracker(channel_id: str) -> SoloConversationTracker:
    global _solo_trackers
    if channel_id not in _solo_trackers:
        _solo_trackers[channel_id] = SoloConversationTracker(channel_id)
    return _solo_trackers[channel_id]


def get_solo_user_profiler() -> SoloUserProfiler:
    global _solo_user_profiler
    if _solo_user_profiler is None:
        _solo_user_profiler = SoloUserProfiler()
    return _solo_user_profiler


def get_solo_intent_detector() -> SoloIntentDetector:
    global _solo_intent_detector
    if _solo_intent_detector is None:
        _solo_intent_detector = SoloIntentDetector()
    return _solo_intent_detector


def get_solo_response_memory() -> SoloResponseMemory:
    global _solo_response_memory
    if _solo_response_memory is None:
        _solo_response_memory = SoloResponseMemory()
    return _solo_response_memory

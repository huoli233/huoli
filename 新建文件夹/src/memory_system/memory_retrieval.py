import time
import json
import asyncio
import inspect
import re
from typing import List, Dict, Any, Optional, Tuple
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.plugin_system.apis import llm_api
from src.common.database.database_model import ThinkingBack
from src.memory_system.retrieval_tools import get_tool_registry, init_all_tools
from src.llm_models.payload_content.message import (
    MessageBuilder,
    RoleType,
    Message,
)
from src.chat.message_receive.chat_stream import get_chat_manager
from src.bw_learner.jargon_explainer import retrieve_concepts_with_jargon

logger = get_logger("记忆检索")

THINKING_BACK_NOT_FOUND_RETENTION_SECONDS = 36000  # 未找到答案记录保留时长
THINKING_BACK_CLEANUP_INTERVAL_SECONDS = 3000  # 清理频率
_last_not_found_cleanup_ts: float = 0.0

_INVALID_MEMORY_MARKERS = frozenset(
    [
        "未找到",
        "没有找到",
        "not found",
        "无匹配",
        "不存在",
        "查询文本为空",
        "未指定查询参数",
        "查询失败",
        "参数错误",
        "用户名称为空",
        "查询线索为空",
        "LPMM知识库未启用",
        "LPMM知识库未初始化",
        "无法获取查询向量",
        "没有可检索的记忆记录",
        "未提供有效的词语",
        "未提供要查询的词语",
        "工具执行失败",
        "missing 1 required positional argument",
        "return_information",
        "search_chat_history",
        "query_words",
        "probe_chat_history",
    ]
)

_INTERNAL_MEMORY_PATTERNS = (
    re.compile(r"第\s*\d+\s*次迭代.*?工具执行失败", re.IGNORECASE),
    re.compile(
        r"missing\s+\d+\s+required\s+positional\s+argument", re.IGNORECASE),
    re.compile(
        r"\b(?:return_information|search_chat_history|query_words|probe_chat_history)\b",
        re.IGNORECASE,),
    re.compile(r"工具执行失败", re.IGNORECASE),)

_LOW_VALUE_SHORT_REPLIES = frozenset(
    [
        "咋了",
        "咋啦",
        "干嘛",
        "然后",
        "活力",
        "嗯",
        "啊",
        "哦",
        "在吗",
        "啥",
        "什么",
    ]
)

_QUERY_PATH_RE = re.compile(
    r"[A-Za-z]:\\|[A-Za-z]:/|[/\\].+[/\\]|\.py\b|\.md\b", re.IGNORECASE
)


def _squash_text_for_match(text: str) -> str:
    """压缩文本中的空白和常见标点，便于做低价值判定。"""
    if not text:
        return ""
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).lower()


def _extract_answer_payload(text: str) -> str:
    """从“问题/答案”格式中提取答案正文。"""
    if not text:
        return ""
    matched = re.search(r"答案[:：]\s*(.+)", text, re.S)
    if matched:
        return matched.group(1).strip()
    return text.strip()


def _looks_like_low_value_short_reply(text: str) -> bool:
    """判断文本是否像“咋了/干嘛”这类低信息量短答。"""
    normalized = _squash_text_for_match(text)
    if not normalized:
        return True
    if normalized in _LOW_VALUE_SHORT_REPLIES:
        return True
    return len(normalized) <= 3


def _build_compound_query_context(message: str, target: str) -> str:
    """为短句或重复句构造带前情的检索上下文，避免只拿一句低信息文本裸查。"""
    history = str(message or "").strip()
    latest = str(target or "").strip()
    if not latest:
        return history

    normalized_latest = _squash_text_for_match(latest)
    if len(normalized_latest) > 8:
        return history

    history_lines = [
        line.strip() for line in history.splitlines() if line.strip()
    ]
    recent_lines = history_lines[-8:]
    if not recent_lines:
        return latest

    recent_context = "\n".join(recent_lines)
    return (
        f"以下是最近对话前情，请重点理解最后这句短消息可能在延续什么上下文：\n"
        f"{recent_context}\n"
        f"当前最新短消息：{latest}"
    )


def _is_valid_memory_query_text(text: str) -> bool:
    payload = str(text or "").strip()
    if not payload:
        return False
    if len(payload) > 120:
        return False
    if _QUERY_PATH_RE.search(payload):
        return False
    if payload.count("\n") >= 3:
        return False
    return True


def _compress_memory_query_text(message: str, target: str) -> str:
    """压缩检索问题，优先保留最新用户句和必要前情，避免整段元消息污染检索。"""
    latest = str(target or "").strip()
    if latest and len(_squash_text_for_match(latest)) <= 12:
        history_lines = [
            line.strip()
            for line in str(message or "").splitlines()
            if line.strip()
        ]
        tail = history_lines[-4:]
        if tail:
            compact_tail = " | ".join(line[:40] for line in tail)
            candidate = f"前情: {compact_tail} | 最新: {latest}"
            if _is_valid_memory_query_text(candidate):
                return candidate
        return latest

    candidate = str(message or "").strip()
    if len(candidate) > 120:
        candidate = candidate[:120]
    return candidate


def should_skip_memory_retrieval(
    message: str, target: str, extra_context: str = ""
) -> bool:
    """当当前上下文已足够解释话头时，跳过记忆检索，避免再次污染 prompt。"""
    latest = str(target or "").strip()
    history = str(message or "").strip()
    context = str(extra_context or "").strip()

    normalized_latest = _squash_text_for_match(latest)
    if not normalized_latest:
        return True

    if len(normalized_latest) >= 15:
        return False

    compact_context = f"{history}\n{context}".strip()
    if not compact_context:
        return False

    normalized_context = _squash_text_for_match(compact_context)
    if not normalized_context:
        return False

    if normalized_latest in normalized_context:
        return False

    continuation_markers = (
        "前情执行块",
        "决策上下文包",
        "续接提示",
        "重复短句判定",
        "最近对话上下文",
        "相关前情",
    )
    if any(marker in compact_context for marker in continuation_markers):
        return True

    return False


def _sanitize_unknown_concepts(concepts: Optional[List[str]]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for word in concepts or []:
        if not isinstance(word, str):
            continue
        item = word.strip()
        if not item:
            continue
        if len(item) > 20:
            continue
        if _QUERY_PATH_RE.search(item):
            continue
        if any(sep in item for sep in ("\n", "\r", "\t")):
            continue
        if item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def _is_high_value_memory_text(text: Optional[str]) -> bool:
    """判断一段检索结果是否值得进入记忆缓存或回复提示词。"""
    if not text or not text.strip():
        return False
    payload = _extract_answer_payload(text)
    if not payload:
        return False
    if any(marker in payload for marker in _INVALID_MEMORY_MARKERS):
        return False
    normalized = _squash_text_for_match(payload)
    if not normalized:
        return False
    if _looks_like_low_value_short_reply(payload):
        return False
    if "问题：" in text and "答案：" in text and len(normalized) < 24:
        return False
    return True


def _sanitize_memory_payload(text: Optional[str]) -> str:
    """清理工具层元信息和失败文本，只保留可给回复模型看的结果。"""
    if not text:
        return ""

    cleaned_lines: List[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(pattern.search(line) for pattern in _INTERNAL_MEMORY_PATTERNS):
            continue
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines).strip()
    if not cleaned_text:
        return ""
    if not _is_high_value_memory_text(cleaned_text):
        return ""
    return cleaned_text


def _cleanup_stale_not_found_thinking_back() -> None:
    """定期清理过期的未找到答案记录"""
    global _last_not_found_cleanup_ts

    now = time.time()
    if (
        now - _last_not_found_cleanup_ts
        < THINKING_BACK_CLEANUP_INTERVAL_SECONDS
    ):
        return

    threshold_time = now - THINKING_BACK_NOT_FOUND_RETENTION_SECONDS
    try:
        deleted_rows = (
            ThinkingBack.delete()
            .where(
                (ThinkingBack.found_answer == 0)
                & (ThinkingBack.update_time < threshold_time)
            )
            .execute()
        )
        if deleted_rows:
            logger.info(
                f"清理过期的未找到答案thinking_back记录 {deleted_rows} 条"
            )
        _last_not_found_cleanup_ts = now
    except Exception as e:
        logger.error(f"清理未找到答案的thinking_back记录失败: {e}")


def invalidate_memory_cache(stream_id: str) -> int:
    """
    使指定聊天流的记忆缓存失效

    在回复成功后调用，防止模型重复回复相同内容

    Args:
        stream_id: 聊天流ID

    Returns:
        删除的记录数
    """
    try:
        deleted_rows = (
            ThinkingBack.delete()
            .where(ThinkingBack.chat_id == stream_id)
            .execute()
        )
        logger.info(
            f"[记忆缓存] 清除 {stream_id[:16]} 的缓存，删除 {deleted_rows} 条记录"
        )
        return deleted_rows
    except Exception as e:
        logger.error(f"清除记忆缓存失败: {e}")
        return 0


def init_memory_retrieval_prompt():
    """初始化记忆检索相关的 prompt 模板和工具"""
    # 首先注册所有工具
    init_all_tools()

    # ReAct Agent prompt（使用function calling，要求先思考再行动）
    Prompt(
        """你的名字是{bot_name}。现在是{time_now}。
你正在参与聊天，你需要搜集信息来帮助你进行回复。
当前聊天记录：
{chat_history}

已收集的信息：
{collected_info}

**工具说明：**
- search_knowledge：查询过往对话记忆、历史事件、之前提到过的概念
- query_words：查询不熟悉词语/缩写/黑话的含义

**重要检索规则（必须遵守）：**
1. **短句/模糊消息必须检索**：如果用户的消息很短（少于10字）、是简称/代词/省略语
   （如"我的那个""上次说的""它""刀盾"等），这通常是在引用之前聊过的内容，
   **必须调用 search_knowledge 工具搜索相关记忆**，不能直接跳过！
2. **重复出现的内容更要检索**：如果当前消息的关键词在聊天记录中出现过，
   说明这是延续性话题，**必须检索之前的完整上下文**。
3. **宁可多搜不可漏搜**：即使你不确定是否有相关记忆，也应该尝试搜索。
   搜索不到不会造成任何问题，但漏掉记忆会导致回复质量下降。

**思考流程：**
- 先判断：这条消息是否可能是对之前内容的引用？（短句、代词、重复关键词=很可能是）
- 如果是 → 立即用 search_knowledge 搜索相关记忆
- 同时可以用 query_words 查询不确定的词汇
- 收集完信息后，调用 return_information 返回总结并结束
""",
        name="memory_retrieval_react_prompt_head",
    )

    # 额外，如果最后一轮迭代：ReAct Agent prompt（使用function calling，要求先思考再行动）
    Prompt(
        """你的名字是{bot_name}。现在是{time_now}。
你正在参与聊天，你需要根据搜集到的信息总结信息。
如果搜集到的信息对于参与聊天，回答问题有帮助，请加入总结，如果无关，请不要加入到总结。

当前聊天记录：
{chat_history}

已收集的信息：
{collected_info}


分析：
- 基于已收集的信息，总结出对当前聊天有帮助的相关信息
- **如果收集的信息对当前聊天有帮助**，在思考中直接给出总结信息，格式为：return_information(information="你的总结信息")
- **如果信息无关或没有帮助**，在思考中给出：return_information(information="")

**重要规则：**
- 必须严格使用检索到的信息回答问题，不要编造信息
- 答案必须精简，不要过多解释
""",
        name="memory_retrieval_react_final_prompt",
    )


def _log_conversation_messages(
    conversation_messages: List[Message],
    head_prompt: Optional[str] = None,
    final_status: Optional[str] = None,
) -> None:
    """输出对话消息列表的日志

    Args:
        conversation_messages: 对话消息列表
        head_prompt: 第一条系统消息（head_prompt）的内容，可选
        final_status: 最终结果状态描述（例如：找到答案/未找到答案），可选
    """
    if not global_config.debug.show_memory_prompt:
        return

    log_lines: List[str] = []

    # 如果有head_prompt，先添加为第一条消息
    if head_prompt:
        msg_info = "========================================\n[消息 1] 角色: System\n-----------------------------"
        msg_info += f"\n{head_prompt}"
        log_lines.append(msg_info)
        start_idx = 2
    else:
        start_idx = 1

    if not conversation_messages and not head_prompt:
        return

    for idx, msg in enumerate(conversation_messages, start_idx):
        role_name = (
            msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        )

        # 构建单条消息的日志信息
        # msg_info = f"\n========================================\n[消息 {idx}] 角色: {role_name} 内容类型: {content_type}\n-----------------------------"
        msg_info = f"\n========================================\n[消息 {idx}] 角色: {role_name}\n-----------------------------"

        # if full_content:
        #     msg_info += f"\n{full_content}"
        if msg.content:
            msg_info += f"\n{msg.content}"

        if msg.tool_calls:
            msg_info += f"\n  工具调用: {len(msg.tool_calls)}个"
            for tool_call in msg.tool_calls:
                msg_info += f"\n    - {
                    tool_call.func_name}: {
                    json.dumps(
                        tool_call.args,
                        ensure_ascii=False)}"

        # if msg.tool_call_id:
        # msg_info += f"\n  工具调用ID: {msg.tool_call_id}"

        log_lines.append(msg_info)

    total_count = len(conversation_messages) + (1 if head_prompt else 0)
    log_text = f"消息列表 (共{total_count}条):{''.join(log_lines)}"
    if final_status:
        log_text += f"\n\n[最终结果] {final_status}"
    logger.info(log_text)


async def _react_agent_solve_question(
    chat_id: str,
    max_iterations: int = 5,
    timeout: float = 30.0,
    initial_info: str = "",
    chat_history: str = "",
) -> Tuple[bool, str, List[Dict[str, Any]], bool]:
    """使用ReAct架构的Agent来解决问题

    Args:
        chat_id: 聊天ID
        max_iterations: 最大迭代次数
        timeout: 超时时间（秒）
        initial_info: 初始信息，将作为collected_info的初始值
        chat_history: 聊天记录，将传递给 ReAct Agent prompt

    Returns:
        Tuple[bool, str, List[Dict[str, Any]], bool]: (是否找到答案, 答案内容, 思考步骤列表, 是否超时)
    """
    start_time = time.time()
    collected_info = initial_info if initial_info else ""
    # 构造日志前缀：[聊天流名称]，用于在日志中标识聊天流
    try:
        chat_name = get_chat_manager().get_stream_name(chat_id) or chat_id
    except Exception:
        chat_name = chat_id
    react_log_prefix = f"[{chat_name}] "
    thinking_steps = []
    is_timeout = False
    conversation_messages: List[Message] = []
    first_head_prompt: Optional[str] = (
        None  # 保存第一次使用的head_prompt（用于日志显示）
    )

    iteration = 0
    while iteration < max_iterations:
        # 检查超时
        if time.time() - start_time > timeout:
            logger.warning(f"ReAct Agent超时，已迭代{iteration}次")
            is_timeout = True
            break

        # 获取工具注册器
        tool_registry = get_tool_registry()

        # 获取bot_name
        bot_name = global_config.bot.nickname

        # 获取当前时间
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 计算剩余迭代次数
        current_iteration = iteration + 1
        remaining_iterations = max_iterations - current_iteration

        # 提取函数调用中参数的值，支持单引号和双引号
        def _extract_quoted_content_v2(text, func_name, param_name):
            """从文本中提取函数调用中参数的值，支持单引号和双引号

            Args:
                text: 要搜索的文本
                func_name: 函数名，如 'return_information'
                param_name: 参数名，如 'information'

            Returns:
                提取的参数值，如果未找到则返回None
            """
            if not text:
                return None

            # 查找函数调用位置（不区分大小写）
            func_pattern = func_name.lower()
            text_lower = text.lower()
            func_pos = text_lower.find(func_pattern)
            if func_pos == -1:
                return None

            # 查找参数名和等号
            param_pattern = f"{param_name}="
            param_pos = text_lower.find(param_pattern, func_pos)
            if param_pos == -1:
                return None

            # 跳过参数名、等号和空白
            start_pos = param_pos + len(param_pattern)
            while start_pos < len(text) and text[start_pos] in " \t\n":
                start_pos += 1

            if start_pos >= len(text):
                return None

            # 确定引号类型
            quote_char = text[start_pos]
            if quote_char not in ['"', "'"]:
                return None

            # 查找匹配的结束引号（考虑转义）
            end_pos = start_pos + 1
            while end_pos < len(text):
                if text[end_pos] == quote_char:
                    # 检查是否是转义的引号
                    if end_pos > start_pos + 1 and text[end_pos - 1] == "\\":
                        end_pos += 1
                        continue
                    # 找到匹配的引号
                    content = text[start_pos + 1: end_pos]
                    # 处理转义字符
                    content = (
                        content.replace('\\"', '"')
                        .replace("\\'", "'")
                        .replace("\\\\", "\\")
                    )
                    return content
                end_pos += 1

            return None

        # 正常迭代：使用head_prompt决定调用哪些工具（包含return_information工具）
        tool_definitions = tool_registry.get_tool_definitions()
        # tool_names = [tool_def["name"] for tool_def in tool_definitions]
        # logger.debug(f"ReAct Agent 第 {iteration + 1} 次迭代，问题: {question}|可用工具: {', '.join(tool_names)} (共{len(tool_definitions)}个)")

        # head_prompt应该只构建一次，使用初始的collected_info，后续迭代都复用同一个
        if first_head_prompt is None:
            # 第一次构建，使用初始的collected_info（即initial_info）
            initial_collected_info = initial_info if initial_info else ""
            first_head_prompt = await global_prompt_manager.format_prompt(
                "memory_retrieval_react_prompt_head",
                bot_name=bot_name,
                time_now=time_now,
                chat_history=chat_history,
                collected_info=initial_collected_info,
                current_iteration=current_iteration,
                remaining_iterations=remaining_iterations,
                max_iterations=max_iterations,
            )

        # 后续迭代都复用第一次构建的head_prompt
        head_prompt = first_head_prompt

        def message_factory(
            _client,
            *,
            _head_prompt: str = head_prompt,
            _conversation_messages: List[Message] = conversation_messages,
        ) -> List[Message]:
            messages: List[Message] = []

            system_builder = MessageBuilder()
            system_builder.set_role(RoleType.System)
            system_builder.add_text_content(_head_prompt)
            messages.append(system_builder.build())

            messages.extend(_conversation_messages)

            return messages

        (
            success,
            response,
            reasoning_content,
            model_name,
            tool_calls,
        ) = await llm_api.generate_with_model_with_message_factory(
            message_factory,
            model_set=model_config.model_task_config.tool_use,
            tool_options=tool_definitions,
            request_type="memory.react",
        )

        # logger.info(
        # f"ReAct Agent 第 {iteration + 1} 次迭代 模型: {model_name} ，调用工具数量: {len(tool_calls) if tool_calls else 0} ，调用工具响应: {response}"
        # )

        if not success:
            logger.error(f"ReAct Agent LLM调用失败: {response}")
            break

        # 注意：这里会检查return_information工具调用，如果检测到return_information工具，会根据information参数决定返回信息或退出查询

        assistant_message: Optional[Message] = None
        if tool_calls:
            assistant_builder = MessageBuilder()
            assistant_builder.set_role(RoleType.Assistant)
            if response and response.strip():
                assistant_builder.add_text_content(response)
            assistant_builder.set_tool_calls(tool_calls)
            assistant_message = assistant_builder.build()
        elif response and response.strip():
            assistant_builder = MessageBuilder()
            assistant_builder.set_role(RoleType.Assistant)
            assistant_builder.add_text_content(response)
            assistant_message = assistant_builder.build()

        # 记录思考步骤
        step = {
            "iteration": iteration + 1,
            "thought": response,
            "actions": [],
            "observations": [],
        }

        if assistant_message:
            conversation_messages.append(assistant_message)

        # 记录思考过程到collected_info中
        if reasoning_content or response:
            thought_summary = reasoning_content or (
                response[:200] if response else ""
            )
            if thought_summary:
                collected_info += f"\n[思考] {thought_summary}\n"

        # 处理工具调用
        if not tool_calls:
            # 如果没有工具调用，检查响应文本中是否包含return_information函数调用格式或JSON格式
            if response and response.strip():
                # 首先尝试解析JSON格式的return_information
                def parse_json_return_information(text: str):
                    """从文本中解析JSON格式的return_information，返回information字符串，如果未找到则返回None"""
                    if not text:
                        return None, None

                    try:
                        # 尝试提取JSON对象（可能包含在代码块中或直接是JSON）
                        json_text = text.strip()

                        # 如果包含代码块标记，提取JSON部分
                        if "```json" in json_text:
                            start = json_text.find("```json") + 7
                            end = json_text.find("```", start)
                            if end != -1:
                                json_text = json_text[start:end].strip()
                        elif "```" in json_text:
                            start = json_text.find("```") + 3
                            end = json_text.find("```", start)
                            if end != -1:
                                json_text = json_text[start:end].strip()

                        # 尝试解析JSON
                        data = json.loads(json_text)

                        # 检查是否包含return_information字段
                        if (
                            isinstance(data, dict)
                            and "return_information" in data
                        ):
                            information = data.get("information", "")
                            return information
                    except (json.JSONDecodeError, ValueError, TypeError):
                        # 如果JSON解析失败，尝试在文本中查找JSON对象
                        try:
                            # 查找第一个 { 和最后一个 } 之间的内容（更健壮的JSON提取）
                            first_brace = text.find("{")
                            if first_brace != -1:
                                # 从第一个 { 开始，找到匹配的 }
                                brace_count = 0
                                json_end = -1
                                for i in range(first_brace, len(text)):
                                    if text[i] == "{":
                                        brace_count += 1
                                    elif text[i] == "}":
                                        brace_count -= 1
                                        if brace_count == 0:
                                            json_end = i + 1
                                            break

                                if json_end != -1:
                                    json_text = text[first_brace:json_end]
                                    data = json.loads(json_text)
                                    if (
                                        isinstance(data, dict)
                                        and "return_information" in data
                                    ):
                                        information = data.get(
                                            "information", ""
                                        )
                                        return information
                        except (json.JSONDecodeError, ValueError, TypeError):
                            pass

                    return None

                # 尝试从文本中解析return_information函数调用
                def parse_return_information_from_text(text: str):
                    """从文本中解析return_information函数调用，返回information字符串，如果未找到则返回None"""
                    if not text:
                        return None

                    # 查找return_information函数调用位置（不区分大小写）
                    func_pattern = "return_information"
                    text_lower = text.lower()
                    func_pos = text_lower.find(func_pattern)
                    if func_pos == -1:
                        return None

                    # 解析information参数（字符串，使用extract_quoted_content）
                    information = extract_quoted_content(
                        text, "return_information", "information"
                    )

                    # 如果information存在（即使是空字符串），也返回它
                    return information

                # 首先尝试解析JSON格式
                parsed_information_json = parse_json_return_information(
                    response
                )
                is_json_format = parsed_information_json is not None

                # 如果JSON解析成功，使用JSON结果
                if is_json_format:
                    parsed_information = parsed_information_json
                else:
                    # 如果JSON解析失败，尝试解析函数调用格式
                    parsed_information = parse_return_information_from_text(
                        response
                    )

                if parsed_information is not None or is_json_format:
                    # 检测到return_information格式（可能是JSON格式或函数调用格式）
                    format_type = (
                        "JSON格式" if is_json_format else "函数调用格式"
                    )
                    # 返回信息（即使为空字符串也返回）
                    step["actions"].append(
                        {
                            "action_type": "return_information",
                            "action_params": {
                                "information": parsed_information or ""
                            },
                        }
                    )
                    if parsed_information and parsed_information.strip():
                        step["observations"] = [
                            f"检测到return_information{format_type}调用，返回信息"
                        ]
                        thinking_steps.append(step)
                        logger.info(
                            f"{react_log_prefix}第 {iteration +
                                                   1} 次迭代 通过return_information{format_type}返回信息: {parsed_information[:100]}..."
                        )

                        _log_conversation_messages(
                            conversation_messages,
                            head_prompt=first_head_prompt,
                            final_status=f"返回信息：{parsed_information}",
                        )

                        return True, parsed_information, thinking_steps, False
                    else:
                        # 信息为空，直接退出查询
                        step["observations"] = [
                            f"检测到return_information{format_type}调用，信息为空"
                        ]
                        thinking_steps.append(step)
                        logger.info(
                            f"{react_log_prefix}第 {
                                iteration +
                                1} 次迭代 通过return_information{format_type}判断信息为空"
                        )

                        _log_conversation_messages(
                            conversation_messages,
                            head_prompt=first_head_prompt,
                            final_status="信息为空：通过return_information文本格式判断信息为空",
                        )

                        return False, "", thinking_steps, False

                # 如果没有检测到return_information格式，记录思考过程，继续下一轮迭代
                step["observations"] = [
                    f"思考完成，但未调用工具。响应: {response}"
                ]
                logger.info(
                    f"{react_log_prefix}第 {
                        iteration +
                        1} 次迭代 思考完成但未调用工具: {response}"
                )
                collected_info += f"思考: {response}"
            else:
                logger.warning(
                    f"{react_log_prefix}第 {
                        iteration + 1} 次迭代 无工具调用且无响应"
                )
                step["observations"] = ["无响应且无工具调用"]
            thinking_steps.append(step)
            iteration += (
                1  # 在continue之前增加迭代计数，避免跳过iteration += 1
            )
            continue

        # 处理工具调用
        # 首先检查是否有return_information工具调用，如果有则立即返回，不再处理其他工具
        return_information_info = None
        for tool_call in tool_calls:
            tool_name = tool_call.func_name
            tool_args = tool_call.args or {}

            if tool_name == "return_information":
                return_information_info = tool_args.get("information", "")

                # 返回信息（即使为空也返回）
                step["actions"].append(
                    {
                        "action_type": "return_information",
                        "action_params": {
                            "information": return_information_info
                        },
                    }
                )
                if return_information_info and return_information_info.strip():
                    # 有信息，返回
                    step["observations"] = [
                        "检测到return_information工具调用，返回信息"
                    ]
                    thinking_steps.append(step)
                    logger.info(
                        f"{react_log_prefix}第 {
                            iteration +
                            1} 次迭代 通过return_information工具返回信息: {return_information_info}"
                    )

                    _log_conversation_messages(
                        conversation_messages,
                        head_prompt=first_head_prompt,
                        final_status=f"返回信息：{return_information_info}",
                    )

                    return True, return_information_info, thinking_steps, False
                else:
                    # 信息为空，直接退出查询
                    step["observations"] = [
                        "检测到return_information工具调用，信息为空"
                    ]
                    thinking_steps.append(step)
                    logger.info(
                        f"{react_log_prefix}第 {
                            iteration +
                            1} 次迭代 通过return_information工具判断信息为空"
                    )

                    _log_conversation_messages(
                        conversation_messages,
                        head_prompt=first_head_prompt,
                        final_status="信息为空：通过return_information工具判断信息为空",
                    )

                    return False, "", thinking_steps, False

        # 如果没有return_information工具调用，继续处理其他工具
        tool_tasks = []
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.func_name
            tool_args = tool_call.args or {}

            logger.debug(
                f"{react_log_prefix}第 {iteration + 1} 次迭代 工具调用 {i + 1}/{len(tool_calls)}: {tool_name}({tool_args})"
            )

            # 跳过return_information工具调用（已经在上面处理过了）
            if tool_name == "return_information":
                continue

            # 普通工具调用
            tool = tool_registry.get_tool(tool_name)
            if tool:
                # 准备工具参数（需要添加chat_id如果工具需要）
                import inspect

                sig = inspect.signature(tool.execute_func)
                tool_params = tool_args.copy()
                if "chat_id" in sig.parameters:
                    tool_params["chat_id"] = chat_id

                # 创建异步任务
                async def execute_single_tool(
                    tool_instance, params, tool_name_str, iter_num
                ):
                    try:
                        observation = await tool_instance.execute(**params)
                        param_str = ", ".join(
                            [
                                f"{k}={v}"
                                for k, v in params.items()
                                if k != "chat_id"
                            ]
                        )
                        return f"查询{tool_name_str}({param_str})的结果：{observation}"
                    except Exception as e:
                        error_msg = f"工具执行失败: {str(e)}"
                        logger.error(
                            f"{react_log_prefix}第 {
                                iter_num +
                                1} 次迭代 工具 {tool_name_str} {error_msg}"
                        )
                        return f"查询{tool_name_str}失败: {error_msg}"

                tool_tasks.append(
                    execute_single_tool(
                        tool, tool_params, tool_name, iteration
                    )
                )
                step["actions"].append(
                    {"action_type": tool_name, "action_params": tool_args}
                )
            else:
                error_msg = f"未知的工具类型: {tool_name}"
                logger.warning(
                    f"{react_log_prefix}第 {iteration + 1} 次迭代 工具 {i + 1}/{len(tool_calls)} {error_msg}"
                )
                tool_tasks.append(
                    asyncio.create_task(
                        asyncio.sleep(
                            0, result=f"查询{tool_name}失败: {error_msg}"
                        )
                    )
                )

        # 并行执行所有工具
        if tool_tasks:
            observations = await asyncio.gather(
                *tool_tasks, return_exceptions=True
            )

            # 处理执行结果
            for i, (tool_call_item, observation) in enumerate(
                zip(tool_calls, observations, strict=False)
            ):
                if isinstance(observation, Exception):
                    observation = f"工具执行异常: {str(observation)}"
                    logger.error(
                        f"{react_log_prefix}第 {
                            iteration +
                            1} 次迭代 工具 {
                            i +
                            1} 执行异常: {observation}"
                    )

                observation_text = (
                    observation
                    if isinstance(observation, str)
                    else str(observation)
                )
                stripped_observation = observation_text.strip()
                step["observations"].append(observation_text)
                collected_info += f"\n{observation_text}\n"
                if stripped_observation:
                    # 不再自动检测工具输出中的jargon，改为通过 query_words 工具主动查询
                    tool_builder = MessageBuilder()
                    tool_builder.set_role(RoleType.Tool)
                    tool_builder.add_text_content(observation_text)
                    tool_builder.add_tool_call(tool_call_item.call_id)
                    conversation_messages.append(tool_builder.build())

        thinking_steps.append(step)

        iteration += 1

    # 正常迭代结束后，如果达到最大迭代次数或超时，执行最终评估
    # 最终评估单独处理，不算在迭代中
    should_do_final_evaluation = False
    if is_timeout:
        should_do_final_evaluation = True
        logger.warning(
            f"{react_log_prefix}超时，已迭代{iteration}次，进入最终评估"
        )
    elif iteration >= max_iterations:
        should_do_final_evaluation = True
        logger.info(
            f"{react_log_prefix}达到最大迭代次数（已迭代{iteration}次），进入最终评估"
        )

    if should_do_final_evaluation:
        # 获取必要变量用于最终评估
        tool_registry = get_tool_registry()
        bot_name = global_config.bot.nickname
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        current_iteration = iteration + 1
        remaining_iterations = 0

        # 提取函数调用中参数的值，支持单引号和双引号
        def extract_quoted_content(text, func_name, param_name):
            """从文本中提取函数调用中参数的值，支持单引号和双引号

            Args:
                text: 要搜索的文本
                func_name: 函数名，如 'return_information'
                param_name: 参数名，如 'information'

            Returns:
                提取的参数值，如果未找到则返回None
            """
            if not text:
                return None

            # 查找函数调用位置（不区分大小写）
            func_pattern = func_name.lower()
            text_lower = text.lower()
            func_pos = text_lower.find(func_pattern)
            if func_pos == -1:
                return None

            # 查找参数名和等号
            param_pattern = f"{param_name}="
            param_pos = text_lower.find(param_pattern, func_pos)
            if param_pos == -1:
                return None

            # 跳过参数名、等号和空白
            start_pos = param_pos + len(param_pattern)
            while start_pos < len(text) and text[start_pos] in " \t\n":
                start_pos += 1

            if start_pos >= len(text):
                return None

            # 确定引号类型
            quote_char = text[start_pos]
            if quote_char not in ['"', "'"]:
                return None

            # 查找匹配的结束引号（考虑转义）
            end_pos = start_pos + 1
            while end_pos < len(text):
                if text[end_pos] == quote_char:
                    # 检查是否是转义的引号
                    if end_pos > start_pos + 1 and text[end_pos - 1] == "\\":
                        end_pos += 1
                        continue
                    # 找到匹配的引号
                    content = text[start_pos + 1: end_pos]
                    # 处理转义字符
                    content = (
                        content.replace('\\"', '"')
                        .replace("\\'", "'")
                        .replace("\\\\", "\\")
                    )
                    return content
                end_pos += 1

            return None

        # 执行最终评估
        evaluation_prompt = await global_prompt_manager.format_prompt(
            "memory_retrieval_react_final_prompt",
            bot_name=bot_name,
            time_now=time_now,
            chat_history=chat_history,
            collected_info=collected_info if collected_info else "暂无信息",
            current_iteration=current_iteration,
            remaining_iterations=remaining_iterations,
            max_iterations=max_iterations,
        )

        (
            eval_success,
            eval_response,
            eval_reasoning_content,
            eval_model_name,
            eval_tool_calls,
        ) = await llm_api.generate_with_model_with_tools(
            evaluation_prompt,
            model_set=model_config.model_task_config.tool_use,
            tool_options=[],  # 最终评估阶段不提供工具
            request_type="memory.react.final",
        )

        if not eval_success:
            logger.error(
                f"ReAct Agent 最终评估阶段 LLM调用失败: {eval_response}"
            )
            _log_conversation_messages(
                conversation_messages,
                head_prompt=first_head_prompt,
                final_status="未找到答案：最终评估阶段LLM调用失败",
            )
            return False, "最终评估阶段LLM调用失败", thinking_steps, is_timeout

        if global_config.debug.show_memory_prompt:
            logger.info(
                f"{react_log_prefix}最终评估Prompt: {evaluation_prompt}"
            )
            logger.info(f"{react_log_prefix}最终评估响应: {eval_response}")

        # 从最终评估响应中提取return_information
        return_information_content = None

        if eval_response:
            return_information_content = extract_quoted_content(
                eval_response, "return_information", "information"
            )

        # 如果提取到信息，返回（无论是否超时，都视为成功完成）
        if return_information_content is not None:
            eval_step = {
                "iteration": current_iteration,
                "thought": f"[最终评估] {eval_response}",
                "actions": [
                    {
                        "action_type": "return_information",
                        "action_params": {
                            "information": return_information_content
                        },
                    }
                ],
                "observations": ["最终评估阶段检测到return_information"],
            }
            thinking_steps.append(eval_step)
            if (
                return_information_content
                and return_information_content.strip()
            ):
                logger.info(
                    f"ReAct Agent 最终评估阶段返回信息: {return_information_content}"
                )
                _log_conversation_messages(
                    conversation_messages,
                    head_prompt=first_head_prompt,
                    final_status=f"返回信息：{return_information_content}",
                )
                return True, return_information_content, thinking_steps, False
            else:
                logger.info("ReAct Agent 最终评估阶段判断信息为空")
                _log_conversation_messages(
                    conversation_messages,
                    head_prompt=first_head_prompt,
                    final_status="信息为空：最终评估阶段判断信息为空",
                )
                return False, "", thinking_steps, False

        # 如果没有明确判断，视为not_enough_info，返回空字符串（不返回任何信息）
        eval_step = {
            "iteration": current_iteration,
            "thought": f"[最终评估] {eval_response}",
            "actions": [
                {
                    "action_type": "return_information",
                    "action_params": {"information": ""},
                }
            ],
            "observations": ["已到达最大迭代次数，信息为空"],
        }
        thinking_steps.append(eval_step)
        logger.info("ReAct Agent 已到达最大迭代次数，信息为空")

        _log_conversation_messages(
            conversation_messages,
            head_prompt=first_head_prompt,
            final_status="未找到答案：已到达最大迭代次数，无法找到答案",
        )

        return False, "", thinking_steps, is_timeout

    # 如果正常迭代过程中提前找到答案返回，不会到达这里
    # 如果正常迭代结束但没有触发最终评估（理论上不应该发生），直接返回
    logger.warning("ReAct Agent正常迭代结束，但未触发最终评估")
    _log_conversation_messages(
        conversation_messages,
        head_prompt=first_head_prompt,
        final_status="未找到答案：正常迭代结束",
    )

    return False, "", thinking_steps, is_timeout


def _get_recent_query_history(
    chat_id: str, time_window_seconds: float = 600.0
) -> str:
    """获取最近一段时间内的查询历史（用于避免重复查询）

    Args:
        chat_id: 聊天ID
        time_window_seconds: 时间窗口（秒），默认10分钟

    Returns:
        str: 格式化的查询历史字符串
    """
    try:
        current_time = time.time()
        start_time = current_time - time_window_seconds

        # 查询最近时间窗口内的记录，按更新时间倒序
        records = (
            ThinkingBack.select()
            .where(
                (ThinkingBack.chat_id == chat_id)
                & (ThinkingBack.update_time >= start_time)
            )
            .order_by(ThinkingBack.update_time.desc())
            .limit(5)  # 最多返回5条最近的记录
        )

        if not records.exists():
            return ""

        history_lines = []
        history_lines.append("最近已查询的问题和结果：")

        for record in records:
            status = "✓ 已找到答案" if record.found_answer else "✗ 未找到答案"
            answer_preview = ""
            # 只有找到答案时才显示答案内容
            if record.found_answer and record.answer:
                # 截取答案前100字符
                answer_preview = record.answer[:100]
                if len(record.answer) > 100:
                    answer_preview += "..."

            history_lines.append(f"- 问题：{record.question}")
            history_lines.append(f"  状态：{status}")
            if answer_preview:
                history_lines.append(f"  答案：{answer_preview}")
            history_lines.append("")  # 空行分隔

        return "\n".join(history_lines)

    except Exception as e:
        logger.error(f"获取查询历史失败: {e}")
        return ""


def _get_recent_found_answers(
    chat_id: str, time_window_seconds: float = 600.0
) -> List[str]:
    """获取最近一段时间内已找到答案的查询记录（用于返回给 replyer）

    Args:
        chat_id: 聊天ID
        time_window_seconds: 时间窗口（秒），默认10分钟

    Returns:
        List[str]: 格式化的答案列表，每个元素格式为 "问题：xxx\n答案：xxx"
    """
    try:
        current_time = time.time()
        start_time = current_time - time_window_seconds

        # 查询最近时间窗口内已找到答案的记录，按更新时间倒序
        records = (
            ThinkingBack.select()
            .where(
                (ThinkingBack.chat_id == chat_id)
                & (ThinkingBack.update_time >= start_time)
                & (ThinkingBack.found_answer == 1)
                & (ThinkingBack.answer.is_null(False))
                & (ThinkingBack.answer != "")
            )
            .order_by(ThinkingBack.update_time.desc())
            .limit(3)  # 最多返回5条最近的记录
        )

        if not records.exists():
            return []

        found_answers = []
        for record in records:
            if record.answer and _is_high_value_memory_text(record.answer):
                found_answers.append(
                    f"问题：{
                        record.question}\n答案：{
                        record.answer}"
                )

        return found_answers

    except Exception as e:
        logger.error(f"获取最近已找到答案的记录失败: {e}")
        return []


def _store_thinking_back(
    chat_id: str,
    question: str,
    context: str,
    found_answer: bool,
    answer: str,
    thinking_steps: List[Dict[str, Any]],
) -> None:
    """存储或更新思考过程到数据库（如果已存在则更新，否则创建）

    Args:
        chat_id: 聊天ID
        question: 问题
        context: 上下文信息
        found_answer: 是否找到答案
        answer: 答案内容
        thinking_steps: 思考步骤列表
    """
    try:
        now = time.time()
        if found_answer and not _is_high_value_memory_text(answer):
            logger.debug(
                f"过滤低价值记忆结果，不写入有效缓存: {question[:50]}..."
            )
            found_answer = False
            answer = ""

        # 先查询是否已存在相同chat_id和问题的记录
        existing = (
            ThinkingBack.select()
            .where(
                (ThinkingBack.chat_id == chat_id)
                & (ThinkingBack.question == question)
            )
            .order_by(ThinkingBack.update_time.desc())
            .limit(1)
        )

        if existing.exists():
            # 更新现有记录
            record = existing.get()
            record.context = context
            record.found_answer = found_answer
            record.answer = answer
            record.thinking_steps = json.dumps(
                thinking_steps, ensure_ascii=False
            )
            record.update_time = now
            record.save()
            logger.info(f"已更新思考过程到数据库，问题: {question[:50]}...")
        else:
            # 创建新记录
            ThinkingBack.create(
                chat_id=chat_id,
                question=question,
                context=context,
                found_answer=found_answer,
                answer=answer,
                thinking_steps=json.dumps(thinking_steps, ensure_ascii=False),
                create_time=now,
                update_time=now,
            )
            # logger.info(f"已创建思考过程到数据库，问题: {question[:50]}...")
    except Exception as e:
        logger.error(f"存储思考过程失败: {e}")


# ==================== 缓存探测器 ====================


class ThinkingRecordCache:
    """ThinkingBack缓存探测器 —— 在发起LLM调用前先行检查历史结果。

    查找同一聊天流中、时间窗口内、已找到答案的记录，
    通过词粒度重叠率判断问题相似性。命中时直接复用历史答案。
    融合XBcore的混合检索思想（词级+语义级），但只保留轻量词级匹配。
    """

    def __init__(
        self,
        window_seconds: float = 3600.0,
        overlap_bar: float = 0.6,
        scan_limit: int = 20,
    ):
        self._window = window_seconds
        self._overlap_bar = overlap_bar
        self._scan_limit = scan_limit

    def probe(self, question: str, stream_id: str) -> Optional[str]:
        """在ThinkingBack表中搜索可复用的历史答案。

        返回:
            命中的答案字符串，或None
        """
        try:
            cutoff = time.time() - self._window
            rows = (
                ThinkingBack.select()
                .where(
                    (ThinkingBack.chat_id == stream_id)
                    & (ThinkingBack.found_answer)
                    & (ThinkingBack.update_time >= cutoff)
                )
                .order_by(ThinkingBack.update_time.desc())
                .limit(self._scan_limit)
            )
            q_lower = question.lower().strip()
            q_tokens = set(q_lower.split())
            for row in rows:
                if not _is_high_value_memory_text(row.answer):
                    continue
                r_lower = (row.question or "").lower().strip()
                # 精确匹配（大小写不敏感）
                if r_lower == q_lower:
                    logger.info(f"[缓存命中] 精确匹配: {question[:40]}")
                    return row.answer
                # 词重叠度匹配
                r_tokens = set(r_lower.split())
                if q_tokens and r_tokens:
                    overlap = len(q_tokens & r_tokens) / max(
                        len(q_tokens), len(r_tokens)
                    )
                    if overlap >= self._overlap_bar:
                        logger.info(
                            f"[缓存命中] 重叠度{overlap:.2f}: {question[:40]} ~ {row.question[:40]}"
                        )
                        return row.answer
        except Exception as exc:
            logger.debug(f"[缓存探测] 失败: {exc}")
        return None


_thinking_cache = ThinkingRecordCache()


# ==================== 关键词直驱工具探针 ====================


class DirectToolProbe:
    """直接工具探针 —— 绕过LLM，用关键词驱动注册工具快速扫描。

    从问题中提取中英文关键词，直接注入工具的第一个参数执行查询。
    命中时跳过昂贵的多轮ReAct推理。
    融合MIMiaoCore的HistoryQueryTool直查概念。
    """

    # 判定"没找到"的标志词
    _MISS_MARKERS = _INVALID_MEMORY_MARKERS
    _SKIP_TOOL_NAMES = frozenset(
        [
            "get_chat_history_detail",
            "query_person_info",
            "search_chat_history",
            "query_words",
        ]
    )

    @staticmethod
    def _prepare_probe_text(question: str) -> str:
        """保留自然语言原貌，只做最轻量的压缩清洗。"""
        if not question:
            return ""
        lines = [
            line.strip() for line in question.splitlines() if line.strip()
        ]
        compact = " ".join(lines)
        compact = re.sub(r"\s+", " ", compact).strip()
        return compact[:200]

    @staticmethod
    def _build_probe_kwargs(
        tool_name: str,
        signature: inspect.Signature,
        query_text: str,
        stream_id: str,
    ) -> Optional[Dict[str, Any]]:
        """按照工具签名构造参数，避免把查询词错误塞进 stream_id 一类字段。"""
        if tool_name in DirectToolProbe._SKIP_TOOL_NAMES:
            return None

        parameters = signature.parameters
        kwargs: Dict[str, Any] = {}
        if "chat_id" in parameters:
            kwargs["chat_id"] = stream_id
        if "stream_id" in parameters:
            kwargs["stream_id"] = stream_id

        if tool_name in ("search_memory_by_embedding", "search_knowledge"):
            kwargs["query"] = query_text
            if "top_k" in parameters:
                kwargs["top_k"] = 3
            if "limit" in parameters:
                kwargs["limit"] = 3
            return kwargs

        remaining_required = [
            name
            for name, param in parameters.items()
            if name not in kwargs and param.default is inspect._empty
        ]
        if len(remaining_required) != 1:
            return None
        target_name = remaining_required[0]
        if target_name not in {"query", "keyword", "words"}:
            return None
        kwargs[target_name] = query_text
        return kwargs

    @staticmethod
    async def attempt(question: str, stream_id: str) -> Optional[str]:
        """尝试直接用关键词驱动工具获取答案。

        返回:
            汇总结果字符串（命中时），或None（未命中）。
        """
        try:
            registry = get_tool_registry()
            all_tools = registry.get_all_tools()
            if not all_tools:
                return None
            query_str = DirectToolProbe._prepare_probe_text(question)
            if not query_str:
                return None
            fragments: List[str] = []
            for tool_tag, tool_obj in all_tools.items():
                # 跳过终结工具
                if tool_tag in ("found_answer", "return_information"):
                    continue
                try:
                    sig = inspect.signature(tool_obj.execute_func)
                    kwargs = DirectToolProbe._build_probe_kwargs(
                        tool_name=tool_tag,
                        signature=sig,
                        query_text=query_str,
                        stream_id=stream_id,
                    )
                    if not kwargs:
                        continue
                    observation = await tool_obj.execute(**kwargs)
                    obs_text = str(observation).strip() if observation else ""
                    if _is_high_value_memory_text(obs_text):
                        fragments.append(obs_text)
                except Exception:
                    continue
            if fragments:
                merged = "\n".join(fragments)
                if len(merged) > 30:
                    logger.info(
                        f"[直驱探针] 命中: {question[:40]}，结果{len(merged)}字符"
                    )
                    return merged[:800]
        except Exception as exc:
            logger.debug(f"[直驱探针] 失败: {exc}")
        return None


# ==================== 近期发现收集器 ====================


def _collect_recent_findings(
    stream_id: str,
    current_answer: Optional[str],
    window_seconds: float = 600.0,
    ceiling: int = 3,
) -> List[str]:
    """收集当前结果 + 近期已找到答案的缓存记录，去重合并。

    返回格式化的答案字符串列表。
    """
    all_items: List[str] = []
    current_question_text: Optional[str] = None
    # 先加入当前结果
    sanitized_current = _sanitize_memory_payload(current_answer)
    if sanitized_current:
        all_items.append(sanitized_current)
    # 拉取近期缓存
    recent = _get_recent_found_answers(stream_id, window_seconds)
    for cached_item in recent:
        sanitized_cached = _sanitize_memory_payload(cached_item)
        if not sanitized_cached:
            continue
        # 去重：跳过与当前结果相同问题的缓存
        if sanitized_current and sanitized_cached == sanitized_current:
            continue
        all_items.append(sanitized_cached)
        if len(all_items) >= ceiling + 1:
            break
    return all_items


# ==================== 三阶段检索管线 ====================


async def _process_memory_retrieval(
    chat_id: str,
    context: str,
    initial_info: str = "",
    max_iterations: Optional[int] = None,
    chat_history: str = "",
) -> Optional[str]:
    """三阶段记忆检索管线: 1)缓存探测 → 2)直驱工具扫描 → 3)ReAct Agent推理。

    任何一阶段命中即短路返回，避免不必要的LLM调用开销。

    参数:
        chat_id: 聊天流ID
        context: 上下文信息
        initial_info: 初始信息（jargon解释等），传递给ReAct Agent
        max_iterations: 最大迭代次数
        chat_history: 聊天记录文本
    返回:
        找到的答案字符串，或None
    """
    _cleanup_stale_not_found_thinking_back()

    # —— 第1阶段：缓存探测 ——
    # 从ThinkingBack中查找近1小时内、同一聊天流的相似问题答案
    cached_hit = _thinking_cache.probe(chat_history[:200], chat_id)
    if cached_hit:
        return cached_hit

    # —— 第2阶段：直驱工具扫描 ——
    # 绕过LLM，用关键词直接驱动注册工具查询
    probe_hit = await DirectToolProbe.attempt(chat_history[:200], chat_id)
    if probe_hit:
        # 命中时存储到ThinkingBack以便后续缓存复用
        _store_thinking_back(
            chat_id=chat_id,
            question=chat_history[:200],
            context=context,
            found_answer=True,
            answer=probe_hit,
            thinking_steps=[{"iteration": 0, "method": "direct_tool_probe"}],
        )
        return probe_hit

    # —— 第3阶段：ReAct Agent推理 ——
    question_initial_info = initial_info or ""
    if max_iterations is None:
        max_iterations = global_config.memory.max_agent_iterations

    found_answer, answer, thinking_steps, is_timeout = (
        await _react_agent_solve_question(
            chat_id=chat_id,
            max_iterations=max_iterations,
            timeout=global_config.memory.agent_timeout_seconds,
            initial_info=question_initial_info,
            chat_history=chat_history,
        )
    )

    # 存储推理结果到ThinkingBack（超时时除外）
    if not is_timeout:
        _store_thinking_back(
            chat_id=chat_id,
            question=chat_history[:200],
            context=context,
            found_answer=found_answer,
            answer=answer if found_answer else "",
            thinking_steps=thinking_steps,
        )
    else:
        logger.info("ReAct Agent超时，不存储结果")

    sanitized_answer = _sanitize_memory_payload(answer)
    if found_answer and sanitized_answer:
        return sanitized_answer
    return None


async def build_memory_retrieval_prompt(
    message: str,
    sender: str,
    target: str,
    chat_stream,
    think_level: int = 1,
    unknown_words: Optional[List[str]] = None,
    relation_context: str = "",
) -> str:
    """构建记忆检索提示
    Args:
        message: 聊天历史记录
        sender: 发送者名称
        target: 目标消息内容
        chat_stream: 聊天流对象
        think_level: 思考深度等级
        unknown_words: Planner 提供的未知词语列表，优先使用此列表而不是从聊天记录匹配
        relation_context: 关系优先级提示（F14：高好感用户优先检索相关记忆）

    Returns:
        str: 记忆检索结果字符串
    """
    start_time = time.time()

    # 构造日志前缀：[聊天流名称]，用于在日志中标识聊天流（优先群名称/用户昵称）
    try:
        group_info = chat_stream.group_info
        user_info = chat_stream.user_info
        # 群聊优先使用群名称
        if group_info is not None and getattr(group_info, "group_name", None):
            stream_name = group_info.group_name.strip() or str(
                group_info.group_id
            )
        # 私聊使用用户昵称
        elif user_info is not None and getattr(
            user_info, "user_nickname", None
        ):
            stream_name = user_info.user_nickname.strip() or str(
                user_info.user_id
            )
        # 兜底使用 stream_id
        else:
            stream_name = chat_stream.stream_id
    except Exception:
        stream_name = chat_stream.stream_id
    log_prefix = f"[{stream_name}] " if stream_name else ""

    logger.info(
        f"{log_prefix}检测是否需要回忆，元消息：{message[:30]}...，消息长度: {len(message)}"
    )
    try:
        chat_id = chat_stream.stream_id
        retrieval_context = _build_compound_query_context(message, target)
        # F14：注入关系优先级提示，影响记忆检索方向
        if relation_context:
            retrieval_context = f"{relation_context}\n{retrieval_context}"
        compressed_query = _compress_memory_query_text(message, target)
        if _is_valid_memory_query_text(compressed_query):
            retrieval_context = compressed_query

        # 初始阶段：使用 Planner 提供的 unknown_words 进行检索（如果提供）
        initial_info = ""
        if unknown_words and len(unknown_words) > 0:
            # 清理和去重 unknown_words
            cleaned_concepts = _sanitize_unknown_concepts(unknown_words)
            if cleaned_concepts:
                # 对匹配到的概念进行jargon检索，作为初始信息
                concept_info = await retrieve_concepts_with_jargon(
                    cleaned_concepts, chat_id
                )
                if concept_info:
                    initial_info += concept_info
                    logger.info(
                        f"{log_prefix}使用 Planner 提供的 unknown_words，共 {len(cleaned_concepts)} 个概念，检索结果: {concept_info[:100]}..."
                    )
                else:
                    logger.debug(
                        f"{log_prefix}unknown_words 检索未找到任何结果"
                    )

        # 直接使用 ReAct Agent 进行记忆检索（跳过问题生成步骤）
        base_max_iterations = global_config.memory.max_agent_iterations
        # 根据think_level调整迭代次数：think_level=1时不变，think_level=0时减半
        if think_level == 0:
            max_iterations = max(1, base_max_iterations // 2)  # 至少为1
        else:
            max_iterations = base_max_iterations
        timeout_seconds = global_config.memory.agent_timeout_seconds
        logger.debug(
            f"{log_prefix}直接使用 ReAct Agent 进行记忆检索，think_level={think_level}，设置最大迭代次数: {max_iterations}（基础值: {base_max_iterations}），超时时间: {timeout_seconds}秒"
        )

        # 直接调用 ReAct Agent 处理记忆检索
        try:
            result = await _process_memory_retrieval(
                chat_id=chat_id,
                context=retrieval_context,
                initial_info=initial_info,
                max_iterations=max_iterations,
                chat_history=retrieval_context,
            )
        except Exception as e:
            logger.error(f"{log_prefix}处理记忆检索时发生异常: {e}")
            result = None

        end_time = time.time()

        sanitized_result = _sanitize_memory_payload(result)
        if sanitized_result:
            # 合并当前结果与近期缓存答案（去重）
            all_findings = _collect_recent_findings(
                chat_id, sanitized_result, window_seconds=600.0, ceiling=3
            )
            if all_findings:
                retrieved_memory = "\n\n".join(all_findings)
                current_cnt = 1
                cached_cnt = len(all_findings) - current_cnt
                logger.info(
                    f"{log_prefix}记忆检索成功，耗时: {(end_time - start_time):.3f}秒，"
                    f"当前查询 {current_cnt} 条，缓存 {cached_cnt} 条，共 {len(all_findings)} 条"
                )
                return f"你回忆起了以下信息：\n{retrieved_memory}\n如果与回复内容相关，可以参考这些回忆的信息。\n"
            logger.info(
                f"{log_prefix}记忆检索成功，耗时: {(end_time - start_time):.3f}秒"
            )
            return f"你回忆起了以下信息：\n{sanitized_result}\n如果与回复内容相关，可以参考这些回忆的信息。\n"
        else:
            # 即使当次没查到，也尝试返回近期缓存答案
            cached_only = _collect_recent_findings(
                chat_id, None, window_seconds=600.0, ceiling=3
            )
            if cached_only:
                retrieved_memory = "\n\n".join(cached_only)
                logger.info(
                    f"{log_prefix}当次未找到新结果，但有 {len(cached_only)} 条近期缓存答案"
                )
                return f"你回忆起了以下信息：\n{retrieved_memory}\n如果与回复内容相关，可以参考这些回忆的信息。\n"
            logger.debug(f"{log_prefix}记忆检索未找到相关信息")
            return ""

    except Exception as e:
        logger.error(f"{log_prefix}记忆检索时发生异常: {str(e)}")
        return ""

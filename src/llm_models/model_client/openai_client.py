import asyncio
import io
import json
import re
import base64
from collections.abc import Iterable
from typing import Callable, Any, Coroutine, Optional
from json_repair import repair_json

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIStatusError,
    NOT_GIVEN,
    AsyncStream,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from src.config.api_ada_configs import ModelInfo, APIProvider  # noqa: E402
from src.common.logger import get_logger  # noqa: E402

from ._registry import client_registry
try:
    from .base_client import (
        APIResponse,
        UsageRecord,
        BaseClient,
    )  # noqa: E402
except ImportError as e:
    raise ImportError(
        f"无法导入 base_client 模块，请检查模块路径是否正确: {e}"
    ) from e  # noqa: E402

try:
    from ..exceptions import (  # noqa: E402
        RespParseException,
        NetworkConnectionError,
        RespNotOkException,
        ReqAbortException,
        EmptyResponseException,
    )
except ImportError as e:
    raise ImportError(
        f"无法导入 exceptions 模块，请检查模块路径是否正确: {e}"
    ) from e  # noqa: E402

try:
    from ..payload_content.message import Message, RoleType  # noqa: E402
except ImportError as e:
    raise ImportError(
        f"无法导入 message 模块，请检查模块路径是否正确: {e}"
    ) from e  # noqa: E402

try:
    from ..payload_content.resp_format import ResponseFormat  # noqa: E402
except ImportError as e:
    raise ImportError(
        f"无法导入 resp_format 模块，请检查模块路径是否正确: {e}"
    ) from e  # noqa: E402

try:
    from ..payload_content.tool_option import (
        ToolOption,
        ToolParam,
        ToolCall,
    )  # noqa: E402
except ImportError as e:
    raise ImportError(
        f"无法导入 tool_option 模块，请检查模块路径是否正确: {e}"
    ) from e  # noqa: E402

logger = get_logger("LLM模型")

def _convert_messages(
    messages: list[Message],
    allow_tool_context: bool = True,
) -> list[ChatCompletionMessageParam]:
    """
    转换消息格式 - 将消息转换为OpenAI API所需的格式
    :param messages: 消息列表
    :return: 转换后的消息列表
    """

    def _convert_message_item(
        message: Message, *, _allow_tool_context: bool
    ) -> Optional[ChatCompletionMessageParam]:
        """
        转换单个消息格式
        :param message: 消息对象
        :return: 转换后的消息字典
        """

        # 添加Content
        content: str | list[dict[str, Any]]
        if isinstance(message.content, str):
            content = message.content
        elif isinstance(message.content, list):
            content = []
            for item in message.content:
                if isinstance(item, tuple):
                    image_format = item[0].lower()
                    # 规范 JPEG MIME 类型后缀，统一使用 image/jpeg
                    if image_format in ("jpg", "jpeg"):
                        mime_suffix = "jpeg"
                    else:
                        mime_suffix = image_format
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime_suffix};base64,{item[1]}"
                            },
                        }
                    )
                elif isinstance(item, str):
                    content.append({"type": "text", "text": item})
        else:
            raise RuntimeError(
                "无法触及的代码：请使用MessageBuilder类构建消息对象"
            )

        ret = {
            "role": message.role.value,
            "content": content,
        }

        if (
            _allow_tool_context
            and message.role == RoleType.Assistant
            and getattr(message, "tool_calls", None)
        ):
            tool_calls_payload: list[dict[str, Any]] = []
            for call in message.tool_calls or []:
                tool_calls_payload.append(
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.func_name,
                            "arguments": json.dumps(
                                call.args or {}, ensure_ascii=False
                            ),
                        },
                    }
                )
            ret["tool_calls"] = tool_calls_payload
            if ret["content"] == []:
                ret["content"] = ""
        elif (
            not _allow_tool_context
            and message.role == RoleType.Assistant
            and getattr(message, "tool_calls", None)
        ):
            # 无工具模式下剥离历史工具调用回合，避免代理报“function call turn顺序错误”
            if ret["content"] == []:
                return None
            if ret["content"] == "":
                return None

        # 添加工具调用ID
        if message.role == RoleType.Tool:
            if not _allow_tool_context:
                # 无工具模式下忽略 tool 回合
                return None
            if not message.tool_call_id:
                raise ValueError(
                    "无法触及的代码：请使用MessageBuilder类构建消息对象"
                )
            ret["tool_call_id"] = message.tool_call_id

        return ret  # type: ignore

    converted_messages: list[ChatCompletionMessageParam] = []
    for message in messages:
        converted = _convert_message_item(
            message, _allow_tool_context=allow_tool_context
        )
        if converted is not None:
            converted_messages.append(converted)
    return converted_messages


def _convert_tool_options(
    tool_options: list[ToolOption],
) -> list[dict[str, Any]]:
    """
    转换工具选项格式 - 将工具选项转换为OpenAI API所需的格式
    :param tool_options: 工具选项列表
    :return: 转换后的工具选项列表
    """

    def _convert_tool_param(tool_option_param: ToolParam) -> dict[str, Any]:
        """
        转换单个工具参数格式
        :param tool_option_param: 工具参数对象
        :return: 转换后的工具参数字典
        """
        # JSON Schema 类型名称修正：
        # - 布尔类型使用 "boolean" 而不是 "bool"
        # - 浮点数使用 "number" 而不是 "float"
        param_type_value = tool_option_param.param_type.value
        if param_type_value == "bool":
            param_type_value = "boolean"
        elif param_type_value == "float":
            param_type_value = "number"

        return_dict: dict[str, Any] = {
            "type": param_type_value,
            "description": tool_option_param.description,
        }
        if tool_option_param.enum_values:
            return_dict["enum"] = tool_option_param.enum_values
        return return_dict

    def _convert_tool_option_item(tool_option: ToolOption) -> dict[str, Any]:
        """
        转换单个工具项格式
        :param tool_option: 工具选项对象
        :return: 转换后的工具选项字典
        """
        ret: dict[str, Any] = {
            "name": tool_option.name,
            "description": tool_option.description,
        }
        if tool_option.params:
            ret["parameters"] = {
                "type": "object",
                "properties": {
                    param.name: _convert_tool_param(param)
                    for param in tool_option.params
                },
                "required": [
                    param.name
                    for param in tool_option.params
                    if param.required
                ],
            }
        return ret

    return [
        {
            "type": "function",
            "function": _convert_tool_option_item(tool_option),
        }
        for tool_option in tool_options
    ]


def _process_delta(
    delta: ChoiceDelta,
    has_rc_attr_flag: bool,
    in_rc_flag: bool,
    rc_delta_buffer: io.StringIO,
    fc_delta_buffer: io.StringIO,
    tool_calls_buffer: list[tuple[str, str, io.StringIO]],
) -> bool:
    # 接收content
    if has_rc_attr_flag:
        # 有独立的推理内容块，则无需考虑content内容的判读
        if (
            hasattr(delta, "reasoning_content") and delta.reasoning_content
        ):  # type: ignore
            # 如果有推理内容，则将其写入推理内容缓冲区
            assert isinstance(delta.reasoning_content, str)  # type: ignore
            rc_delta_buffer.write(delta.reasoning_content)  # type: ignore
        elif delta.content:
            # 如果有正式内容，则将其写入正式内容缓冲区
            fc_delta_buffer.write(delta.content)
    elif hasattr(delta, "content") and delta.content is not None:
        # 没有独立的推理内容块，但有正式内容
        if in_rc_flag:
            # 当前在推理内容块中
            if delta.content == "</think>":
                # 如果当前内容是</think>，则将其视为推理内容的结束标记，退出推理内容块
                in_rc_flag = False
            else:
                # 其他情况视为推理内容，加入推理内容缓冲区
                rc_delta_buffer.write(delta.content)
        elif delta.content == "<think>" and not fc_delta_buffer.getvalue():
            # 如果当前内容是<think>，且正式内容缓冲区为空，说明<think>为输出的首个token
            # 则将其视为推理内容的开始标记，进入推理内容块
            in_rc_flag = True
        else:
            # 其他情况视为正式内容，加入正式内容缓冲区
            fc_delta_buffer.write(delta.content)
    # 接收tool_calls
    if hasattr(delta, "tool_calls") and delta.tool_calls:
        tool_call_delta = delta.tool_calls[0]

        if tool_call_delta.index >= len(tool_calls_buffer):
            # 调用索引号大于等于缓冲区长度，说明是新的工具调用
            if (
                tool_call_delta.id
                and tool_call_delta.function
                and tool_call_delta.function.name
            ):
                tool_calls_buffer.append(
                    (
                        tool_call_delta.id,
                        tool_call_delta.function.name,
                        io.StringIO(),
                    )
                )
            else:
                logger.warning(
                    "工具调用索引号大于等于缓冲区长度，但缺少ID或函数信息。"
                )

        if tool_call_delta.function and tool_call_delta.function.arguments:
            # 如果有工具调用参数，则添加到对应的工具调用的参数串缓冲区中
            tool_calls_buffer[tool_call_delta.index][2].write(
                tool_call_delta.function.arguments
            )

    return in_rc_flag


def _build_stream_api_resp(
    _fc_delta_buffer: io.StringIO,
    _rc_delta_buffer: io.StringIO,
    _tool_calls_buffer: list[tuple[str, str, io.StringIO]],
    finish_reason: str | None = None,
) -> APIResponse:
    resp = APIResponse()

    if _rc_delta_buffer.tell() > 0:
        # 如果推理内容缓冲区不为空，则将其写入APIResponse对象
        resp.reasoning_content = _rc_delta_buffer.getvalue()
    _rc_delta_buffer.close()
    if _fc_delta_buffer.tell() > 0:
        # 如果正式内容缓冲区不为空，则将其写入APIResponse对象
        resp.content = _fc_delta_buffer.getvalue()
    _fc_delta_buffer.close()
    if _tool_calls_buffer:
        # 如果工具调用缓冲区不为空，则将其解析为ToolCall对象列表
        resp.tool_calls = []
        for call_id, function_name, arguments_buffer in _tool_calls_buffer:
            if arguments_buffer.tell() > 0:
                # 如果参数串缓冲区不为空，则解析为JSON对象
                raw_arg_data = arguments_buffer.getvalue()
                arguments_buffer.close()
                try:
                    arguments = json.loads(repair_json(raw_arg_data))
                    if not isinstance(arguments, dict):
                        raise RespParseException(
                            None,
                            f"响应解析失败，工具调用参数无法解析为字典类型。工具调用参数原始响应：\n{raw_arg_data}",
                        )
                except json.JSONDecodeError as e:
                    raise RespParseException(
                        None,
                        f"响应解析失败，无法解析工具调用参数。工具调用参数原始响应：{raw_arg_data}",
                    ) from e
            else:
                arguments_buffer.close()
                arguments = None

            resp.tool_calls.append(ToolCall(call_id, function_name, arguments))

    # 检查 max_tokens 截断（流式的告警改由处理函数统一输出，这里不再输出）
    # 保留 finish_reason 仅用于上层判断

    if not resp.content and not resp.tool_calls:
        raise EmptyResponseException()

    return resp


def _apply_model_reasoning_policy(model_info: ModelInfo, response: APIResponse) -> None:
    if not bool(getattr(model_info, "suppress_reasoning", False)):
        return
    content = str(response.content or "").strip()
    if content.startswith("Thinking Process:") or content.startswith("We need answer"):
        response.content = ""
    response.reasoning_content = ""


async def _default_stream_response_handler(
    resp_stream: AsyncStream[ChatCompletionChunk],
    interrupt_flag: asyncio.Event | None,
) -> tuple[APIResponse, Optional[tuple[int, int, int]]]:
    """
    流式响应处理函数 - 处理OpenAI API的流式响应
    :param resp_stream: 流式响应对象
    :return: APIResponse对象
    """

    _has_rc_attr_flag = False  # 标记是否有独立的推理内容块
    _in_rc_flag = False  # 标记是否在推理内容块中
    _rc_delta_buffer = (
        io.StringIO()
    )  # 推理内容缓冲区，用于存储接收到的推理内容
    _fc_delta_buffer = (
        io.StringIO()
    )  # 正式内容缓冲区，用于存储接收到的正式内容
    # 工具调用缓冲区，用于存储接收到的工具调用
    _tool_calls_buffer: list[tuple[str, str, io.StringIO]] = []
    _usage_record = None  # 使用情况记录
    finish_reason: str | None = None  # 记录最后的 finish_reason
    _model_name: str | None = None  # 记录模型名

    def _insure_buffer_closed():
        # 确保缓冲区被关闭
        if _rc_delta_buffer and not _rc_delta_buffer.closed:
            _rc_delta_buffer.close()
        if _fc_delta_buffer and not _fc_delta_buffer.closed:
            _fc_delta_buffer.close()
        for _, _, buffer in _tool_calls_buffer:
            if buffer and not buffer.closed:
                buffer.close()

    async for event in resp_stream:
        if interrupt_flag and interrupt_flag.is_set():
            # 如果中断量被设置，则抛出ReqAbortException
            _insure_buffer_closed()
            raise ReqAbortException("请求被外部信号中断")
        # 空 choices / usage-only 帧的防御
        if not hasattr(event, "choices") or not event.choices:
            if hasattr(event, "usage") and event.usage:
                _usage_record = (
                    event.usage.prompt_tokens or 0,
                    event.usage.completion_tokens or 0,
                    event.usage.total_tokens or 0,
                )
            continue  # 跳过本帧，避免访问 choices[0]
        delta = event.choices[0].delta  # 获取当前块的delta内容

        if (
            hasattr(event.choices[0], "finish_reason")
            and event.choices[0].finish_reason
        ):
            finish_reason = event.choices[0].finish_reason

        if hasattr(event, "model") and event.model and not _model_name:
            _model_name = event.model  # 记录模型名

        if (
            hasattr(delta, "reasoning_content") and delta.reasoning_content
        ):  # type: ignore
            # 标记：有独立的推理内容块
            _has_rc_attr_flag = True

        _in_rc_flag = _process_delta(
            delta,
            _has_rc_attr_flag,
            _in_rc_flag,
            _rc_delta_buffer,
            _fc_delta_buffer,
            _tool_calls_buffer,
        )

        if event.usage:
            # 如果有使用情况，则将其存储在APIResponse对象中
            _usage_record = (
                event.usage.prompt_tokens or 0,
                event.usage.completion_tokens or 0,
                event.usage.total_tokens or 0,
            )

    try:
        resp = _build_stream_api_resp(
            _fc_delta_buffer,
            _rc_delta_buffer,
            _tool_calls_buffer,
            finish_reason=finish_reason,
        )
        # 统一在这里输出 max_tokens 截断的警告，并从 resp 中读取
        if finish_reason == "length":
            # 把模型名塞到 resp.raw_data，后续严格“从 resp 提取”
            try:
                if _model_name:
                    resp.raw_data = {"model": _model_name}
            except Exception as e:
                logger.debug(f"openai日志采集-模型名写入失败: {e}")
            model_dbg = None
            try:
                if isinstance(resp.raw_data, dict):
                    model_dbg = resp.raw_data.get("model")
            except Exception:
                model_dbg = None

            # 统一日志格式
            logger.info(
                "模型%s因为超过最大max_token限制，可能仅输出部分内容，可视情况调整"
                % (model_dbg or "")
            )

        return resp, _usage_record
    except Exception:
        # 确保缓冲区被关闭
        _insure_buffer_closed()
        raise


pattern = re.compile(
    r"<think>(?P<think>.*?)</think>(?P<content>.*)|<think>(?P<think_unclosed>.*)|(?P<content_only>.+)",
    re.DOTALL,)
"""用于解析推理内容的正则表达式"""


def _default_normal_response_parser(
    resp: ChatCompletion,
) -> tuple[APIResponse, Optional[tuple[int, int, int]]]:
    """
    解析对话补全响应 - 将OpenAI API响应解析为APIResponse对象
    :param resp: 响应对象
    :return: APIResponse对象
    """
    api_response = APIResponse()

    # 兼容部分 OpenAI 兼容服务在空回复时返回 choices=None 的情况
    choices = getattr(resp, "choices", None)
    if not choices:
        try:
            model_dbg = getattr(resp, "model", None)
            id_dbg = getattr(resp, "id", None)
            usage_dbg = None
            if hasattr(resp, "usage") and resp.usage:
                usage_dbg = {
                    "prompt": getattr(resp.usage, "prompt_tokens", None),
                    "completion": getattr(
                        resp.usage, "completion_tokens", None
                    ),
                    "total": getattr(resp.usage, "total_tokens", None),
                }
            try:
                raw_snippet = str(resp)[:300]
            except Exception:
                raw_snippet = "<unserializable>"
            logger.warning(
                f"[响应解析] empty choices: model={model_dbg} id={id_dbg} usage={usage_dbg} raw≈{raw_snippet}"
            )
        except Exception as e:
            logger.warning(f"日志采集失败: {e}")
        raise EmptyResponseException("响应解析失败，choices 为空或缺失")
    message_part = choices[0].message

    if (
        hasattr(message_part, "reasoning_content")
        and message_part.reasoning_content
    ):  # type: ignore
        # 有有效的推理字段
        api_response.content = message_part.content
        api_response.reasoning_content = message_part.reasoning_content  # type: ignore
    elif message_part.content:
        # 提取推理和内容
        match = pattern.match(message_part.content)
        if not match:
            raise RespParseException(
                resp, "响应解析失败，无法捕获推理内容和输出内容"
            )
        if match.group("think") is not None:
            result = (
                match.group("think").strip(),
                match.group("content").strip(),
            )
        elif match.group("think_unclosed") is not None:
            result = match.group("think_unclosed").strip(), None
        else:
            result = None, match.group("content_only").strip()
        api_response.reasoning_content, api_response.content = result

    # 提取工具调用
    if message_part.tool_calls:
        api_response.tool_calls = []
        for call in message_part.tool_calls:
            try:
                arguments = json.loads(repair_json(call.function.arguments))
                # 【新增修复逻辑】如果解析出来还是字符串，说明发生了双重编码，尝试二次解析
                if isinstance(arguments, str):
                    try:
                        # 尝试对字符串内容再次进行修复和解析
                        arguments = json.loads(repair_json(arguments))
                    except Exception as e:
                        logger.debug(f"二次解析工具参数失败: {e}")
                if not isinstance(arguments, dict):
                    # 此时为了调试方便，建议打印出 arguments 的类型
                    raise RespParseException(
                        resp,
                        f"响应解析失败，工具调用参数无法解析为字典类型 type={
                            type(arguments)} arguments={arguments}",
                    )
                api_response.tool_calls.append(
                    ToolCall(call.id, call.function.name, arguments)
                )
            except json.JSONDecodeError as e:
                raise RespParseException(
                    resp, "响应解析失败，无法解析工具调用参数"
                ) from e

    # 提取Usage信息
    if resp.usage:
        _usage_record = (
            resp.usage.prompt_tokens or 0,
            resp.usage.completion_tokens or 0,
            resp.usage.total_tokens or 0,
        )
    else:
        _usage_record = None

    # 将原始响应存储在原始数据中
    api_response.raw_data = resp

    # 检查 max_tokens 截断
    try:
        choice0 = resp.choices[0]
        reason = getattr(choice0, "finish_reason", None)
        if reason and reason == "length":
            # print(resp)
            _model_name = resp.model
            # 统一日志格式
            logger.info(
                "模型%s因为超过最大max_token限制，可能仅输出部分内容，可视情况调整"
                % (_model_name or "")
            )
            return api_response, _usage_record
    except Exception as e:
        logger.debug(f"检查 MAX_TOKENS 截断时异常: {e}")

    if not api_response.content and not api_response.tool_calls:
        # 当 content 为空但 reasoning_content 有内容时（如推理模型只返回思考内容），
        # 将 reasoning_content 作为 content 使用，避免无谓的空响应重试
        if api_response.reasoning_content and str(api_response.reasoning_content).strip():
            api_response.content = api_response.reasoning_content
            api_response.reasoning_content = ""
            return api_response, _usage_record
        # 收集诊断信息
        try:
            _finish_reason = getattr(resp.choices[0], "finish_reason", None) if resp.choices else None
            _model_name_dbg = getattr(resp, "model", None)
            _completion_tokens = (_usage_record[1] if _usage_record else None)
        except Exception:
            _finish_reason = None
            _model_name_dbg = None
            _completion_tokens = None
        logger.warning(
            f"[非流式响应解析] 空响应: content='{api_response.content}', tool_calls={api_response.tool_calls}, "
            f"reasoning_content='{api_response.reasoning_content}', "
            f"model={_model_name_dbg}, finish_reason={_finish_reason}, "
            f"completion_tokens={_completion_tokens}"
        )
        raise EmptyResponseException()

    return api_response, _usage_record


@client_registry.register_client_class("openai")
class OpenaiClient(BaseClient):
    def __init__(self, api_provider: APIProvider):
        super().__init__(api_provider)
        # 客户端超时设为 None，完全依赖 asyncio.wait_for 控制超时
        # 避免 OpenAI 内部 httpx 超时与 asyncio 超时冲突
        self.client: AsyncOpenAI = AsyncOpenAI(
            base_url=api_provider.base_url,
            api_key=api_provider.api_key,
            max_retries=2,
            timeout=None,
        )

    async def get_response(
        self,
        model_info: ModelInfo,
        message_list: list[Message],
        tool_options: list[ToolOption] | None = None,
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.7,
        response_format: ResponseFormat | None = None,
        stream_response_handler: Optional[
            Callable[
                [AsyncStream[ChatCompletionChunk], asyncio.Event | None],
                Coroutine[
                    Any,
                    Any,
                    tuple[APIResponse, Optional[tuple[int, int, int]]],
                ],
            ]
        ] = None,
        async_response_parser: Optional[
            Callable[
                [ChatCompletion],
                tuple[APIResponse, Optional[tuple[int, int, int]]],
            ]
        ] = None,
        interrupt_flag: asyncio.Event | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """
        获取对话响应
        Args:
            model_info: 模型信息
            message_list: 对话体
            tool_options: 工具选项（可选，默认为None）
            max_tokens: 最大token数（可选，默认为1024）
            temperature: 温度（可选，默认为0.7）
            response_format: 响应格式（可选，默认为 NotGiven ）
            stream_response_handler: 流式响应处理函数（可选，默认为default_stream_response_handler）
            async_response_parser: 响应解析函数（可选，默认为default_response_parser）
            interrupt_flag: 中断信号量（可选，默认为None）
        Returns:
            (响应文本, 推理文本, 工具调用, 其他数据)
        """
        if stream_response_handler is None:
            stream_response_handler = _default_stream_response_handler
        if async_response_parser is None:
            async_response_parser = _default_normal_response_parser
        if max_tokens is None or max_tokens <= 0:
            max_tokens = getattr(model_info, "max_tokens", None) or 1024

        # 将messages构造为OpenAI API所需的格式
        _allow_tool_context = bool(tool_options)
        messages: Iterable[ChatCompletionMessageParam] = _convert_messages(
            message_list, allow_tool_context=_allow_tool_context
        )
        # 将tool_options转换为OpenAI API所需的格式
        tools: Iterable[ChatCompletionToolParam] = (
            _convert_tool_options(tool_options) if tool_options else NOT_GIVEN
        )  # type: ignore
        if tool_options:
            logger.debug(
                f"[工具调用格式] client=openai chat.completions "
                f"stream={'true' if model_info.force_stream_mode else 'false'} "
                f"tools={len(tool_options)} model={model_info.model_identifier}"
            )

        async def _do_stream_request():
            """执行流式请求并处理响应"""
            stream_resp = await self.client.chat.completions.create(
                model=model_info.model_identifier,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                response_format=response_format if response_format is not None else NOT_GIVEN,
                extra_body=extra_params,
            )
            parsed_response, usage_record = await stream_response_handler(stream_resp, interrupt_flag)
            _apply_model_reasoning_policy(model_info, parsed_response)
            return parsed_response, usage_record

        async def _do_normal_request():
            """执行非流式请求并解析响应"""
            normal_resp = await self.client.chat.completions.create(
                model=model_info.model_identifier,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                response_format=response_format if response_format is not None else NOT_GIVEN,
                extra_body=extra_params,
            )
            parsed_response, usage_record = async_response_parser(normal_resp)
            _apply_model_reasoning_policy(model_info, parsed_response)
            return parsed_response, usage_record

        async def _do_request_with_interrupt():
            """包装请求以支持中断信号检查"""
            if model_info.force_stream_mode:
                return await _do_stream_request()
            else:
                try:
                    return await _do_normal_request()
                except EmptyResponseException:
                    # 某些代理（如GeminiProxy）在非流式模式下返回 content=null，
                    # 但流式模式正常。当非流式空响应时自动回退到流式重试
                    logger.info(
                        f"[非流式→流式回退] 模型 {model_info.model_identifier} 非流式响应为空，"
                        f"自动使用流式模式重试"
                    )
                    stream_resp = await self.client.chat.completions.create(
                        model=model_info.model_identifier,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        response_format=response_format if response_format is not None else NOT_GIVEN,
                        extra_body=extra_params,
                    )
                    parsed_response, usage_record = await stream_response_handler(stream_resp, interrupt_flag)
                    _apply_model_reasoning_policy(model_info, parsed_response)
                    return parsed_response, usage_record

        try:
            # 使用 asyncio.wait_for 包装整个请求，确保超时后能正确取消
            # 同时定期检查中断信号
            req_task = asyncio.create_task(_do_request_with_interrupt())
            while not req_task.done():
                if interrupt_flag and interrupt_flag.is_set():
                    req_task.cancel()
                    try:
                        await req_task
                    except asyncio.CancelledError:
                        logger.debug("openai请求被取消")
                    raise ReqAbortException("请求被外部信号中断")
                await asyncio.sleep(0.1)

            resp, usage_record = await req_task
        except APIConnectionError as e:
            # 重封装APIConnectionError为NetworkConnectionError
            raise NetworkConnectionError() from e
        except APIStatusError as e:
            # 重封装APIError为RespNotOkException
            raise RespNotOkException(e.status_code, e.message) from e

        if usage_record:
            resp.usage = UsageRecord(
                model_name=model_info.name,
                provider_name=model_info.api_provider,
                prompt_tokens=usage_record[0],
                completion_tokens=usage_record[1],
                total_tokens=usage_record[2],
            )

        # logger.debug(f"OpenAI API响应: {resp}")

        return resp

    async def get_embedding(
        self,
        model_info: ModelInfo,
        embedding_input: str,
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """
        获取文本嵌入
        :param model_info: 模型信息
        :param embedding_input: 嵌入输入文本
        :return: 嵌入响应
        """
        try:
            raw_response = await self.client.embeddings.create(
                model=model_info.model_identifier,
                input=embedding_input,
                extra_body=extra_params,
            )
        except APIConnectionError as e:
            # 添加详细的错误信息以便调试
            logger.error(f"OpenAI API连接错误（嵌入模型）: {str(e)}")
            logger.error(f"错误类型: {type(e)}")
            if hasattr(e, "__cause__") and e.__cause__:
                logger.error(f"底层错误: {str(e.__cause__)}")
            raise NetworkConnectionError() from e
        except APIStatusError as e:
            # 重封装APIError为RespNotOkException
            raise RespNotOkException(e.status_code) from e

        response = APIResponse()

        # 解析嵌入响应
        if len(raw_response.data) > 0:
            response.embedding = raw_response.data[0].embedding
        else:
            raise RespParseException(
                raw_response,
                "响应解析失败，缺失嵌入数据。",
            )

        # 解析使用情况
        if hasattr(raw_response, "usage"):
            response.usage = UsageRecord(
                model_name=model_info.name,
                provider_name=model_info.api_provider,
                prompt_tokens=raw_response.usage.prompt_tokens or 0,
                completion_tokens=getattr(
                    raw_response.usage, "completion_tokens", 0
                ),
                total_tokens=raw_response.usage.total_tokens or 0,
            )

        return response

    async def get_audio_transcriptions(
        self,
        model_info: ModelInfo,
        audio_base64: str,
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """
        获取音频转录
        :param model_info: 模型信息
        :param audio_base64: base64编码的音频数据
        :extra_params: 附加的请求参数
        :return: 音频转录响应
        """
        try:
            raw_response = await self.client.audio.transcriptions.create(
                model=model_info.model_identifier,
                file=("audio.wav", io.BytesIO(base64.b64decode(audio_base64))),
                extra_body=extra_params,
            )
        except APIConnectionError as e:
            raise NetworkConnectionError() from e
        except APIStatusError as e:
            # 重封装APIError为RespNotOkException
            raise RespNotOkException(e.status_code) from e
        response = APIResponse()
        # 解析转录响应
        if hasattr(raw_response, "text"):
            response.content = raw_response.text
        else:
            raise RespParseException(
                raw_response,
                "响应解析失败，缺失转录文本。",
            )
        return response

    def get_support_image_formats(self) -> list[str]:
        """
        获取支持的图片格式
        :return: 支持的图片格式列表
        """
        return ["jpg", "jpeg", "png", "webp", "gif"]

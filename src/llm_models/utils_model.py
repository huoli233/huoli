import threading as _threading
import re
import asyncio
import time
import random

from enum import Enum
from rich.traceback import install
from typing import Tuple, List, Dict, Optional, Callable, Any, Set
import traceback

from src.common.logger import get_logger
from src.common.task_utils import safe_create_task
from src.config.config import model_config
from src.config.api_ada_configs import APIProvider, ModelInfo, TaskConfig
from .payload_content.message import MessageBuilder, Message, RoleType
from .payload_content.resp_format import RespFormat
from .payload_content.tool_option import (
    ToolOption,
    ToolCall,
    ToolOptionBuilder,
    ToolParamType,
)
from .model_client.base_client import BaseClient, APIResponse
from .model_client._registry import client_registry
import src.llm_models.model_client.openai_client as _openai_reg  # noqa: F401
import src.llm_models.model_client.gemini_client as _gemini_reg  # noqa: F401
from .utils import compress_messages, llm_usage_recorder
from .exceptions import (
    NetworkConnectionError,
    RespNotOkException,
    EmptyResponseException,
    ModelAttemptFailed,
)

install(extra_lines=3)

logger = get_logger("模型工具")

_DEFAULT_NO_THINKING_SYSTEM_PROMPT = (
    "运行模式：常规快速推理。不要进入或输出思考模式，不要生成 <think>...</think>、"
    "Thinking Process、reasoning_content 或任何推理过程；只输出最终答复。"
    "只有模型配置显式 enable_thinking=true / thinking_enabled=true，或 model_routing.enable_thinking_for 命中时，才允许使用思考模式。"
)
_NO_THINKING_GUARD_MARKER = "运行模式：常规快速推理"

# 全局模型可用性追踪器：记录各模型的失败计数和冷却截止时间
_model_health_ledger: Dict[str, Dict[str, Any]] = {}
_health_ledger_lock = _threading.Lock()
_tool_call_unsupported_models: Set[str] = set()
_provider_auth_ledger: Dict[str, Dict[str, float]] = {}


def _health_cooldown_sec() -> float:
    try:
        from src.config.core_config_engine import get_core_config

        view = get_core_config().resolve_module_view("model_routing")
        return float(view.values.get("model_cooldown_seconds", 120.0))
    except Exception:
        return 120.0


def _health_fail_threshold() -> int:
    try:
        from src.config.core_config_engine import get_core_config

        view = get_core_config().resolve_module_view("model_routing")
        return int(view.values.get("model_fail_threshold", 3))
    except Exception:
        return 3


def _tool_incompat_cooldown_sec() -> float:
    try:
        from src.config.core_config_engine import get_core_config

        view = get_core_config().resolve_module_view("model_routing")
        return float(view.values.get("tool_incompat_cooldown_seconds", 600.0))
    except Exception:
        return 600.0


def _mark_model_failure(model_name: str) -> None:
    """记录模型失败，连续失败达到阈值后进入冷却（线程安全）"""
    with _health_ledger_lock:
        entry = _model_health_ledger.setdefault(model_name, {"consecutive_fails": 0, "cooldown_until": 0.0})
        entry["consecutive_fails"] += 1
        if entry["consecutive_fails"] >= _health_fail_threshold():
            new_cooldown_until = time.time() + _health_cooldown_sec()
            entry["cooldown_until"] = max(float(entry.get("cooldown_until", 0.0) or 0.0), new_cooldown_until)
            cooldown_remain = max(0.0, entry["cooldown_until"] - time.time())
            logger.warning(
                f"模型 '{model_name}' 连续失败 {entry['consecutive_fails']} 次，冷却 {cooldown_remain:.0f} 秒"
            )


def _mark_model_success(model_name: str) -> None:
    """模型请求成功，重置失败计数（线程安全）"""
    with _health_ledger_lock:
        if model_name in _model_health_ledger:
            _model_health_ledger[model_name] = {
                "consecutive_fails": 0,
                "cooldown_until": 0.0,
            }


def _provider_auth_fail_threshold(status_code: int, error_detail_text: str = "") -> int:
    """不同状态码采用不同鉴权熔断阈值。对于明确包含API key无效致命信息的异常，1次即熔断。"""
    if status_code in (401, 403):
        return 1
    fatal_api_key_markers = (
        "api key not valid",
        "invalid api key",
        "incorrect api key",
        "api_key not valid",
    )
    if error_detail_text and any(marker in error_detail_text.lower() for marker in fatal_api_key_markers):
        return 1
    return 2


def _register_provider_auth_failure(provider_name: str, *, status_code: int) -> Tuple[int, bool]:
    """记录 provider 鉴权失败，返回 (当前计数, 是否触发熔断)。"""
    if not provider_name:
        return 0, False
    now = time.time()
    with _health_ledger_lock:
        entry = _provider_auth_ledger.setdefault(
            provider_name,
            {
                "consecutive_fails": 0.0,
                "last_fail_ts": 0.0,
                "auth_block_until": 0.0,
            },
        )
        if now - float(entry.get("last_fail_ts", 0.0) or 0.0) > 180.0:
            entry["consecutive_fails"] = 0.0
        entry["consecutive_fails"] = float(entry.get("consecutive_fails", 0.0) or 0.0) + 1.0
        entry["last_fail_ts"] = now
        count = int(entry["consecutive_fails"])
        should_trip = count >= _provider_auth_fail_threshold(int(status_code or 0))
        return count, should_trip


def _set_provider_auth_block(provider_name: str, *, cooldown_sec: float) -> None:
    """设置 provider 级鉴权熔断窗口。"""
    if not provider_name:
        return
    now = time.time()
    with _health_ledger_lock:
        entry = _provider_auth_ledger.setdefault(
            provider_name,
            {
                "consecutive_fails": 0.0,
                "last_fail_ts": 0.0,
                "auth_block_until": 0.0,
            },
        )
        entry["auth_block_until"] = max(
            float(entry.get("auth_block_until", 0.0) or 0.0),
            now + max(1.0, float(cooldown_sec or 1.0)),
        )
        entry["consecutive_fails"] = 0.0


def _is_provider_auth_blocked(provider_name: str) -> bool:
    if not provider_name:
        return False
    with _health_ledger_lock:
        entry = _provider_auth_ledger.get(provider_name)
        if not entry:
            return False
        return float(entry.get("auth_block_until", 0.0) or 0.0) > time.time()


def _provider_auth_block_remaining(provider_name: str) -> float:
    if not provider_name:
        return 0.0
    with _health_ledger_lock:
        entry = _provider_auth_ledger.get(provider_name)
        if not entry:
            return 0.0
        return max(0.0, float(entry.get("auth_block_until", 0.0) or 0.0) - time.time())


def _clear_provider_auth_failures(provider_name: str) -> None:
    """provider 出现成功调用时清空鉴权失败计数和熔断。"""
    if not provider_name:
        return
    with _health_ledger_lock:
        if provider_name not in _provider_auth_ledger:
            return
        _provider_auth_ledger[provider_name] = {
            "consecutive_fails": 0.0,
            "last_fail_ts": 0.0,
            "auth_block_until": 0.0,
        }


def _cooldown_model_now(model_name: str, *, reason: str = "", cooldown_sec: Optional[float] = None) -> None:
    """让明显不兼容的模型立即进入冷却，减少重复探测。"""
    effective_cooldown = max(1.0, float(cooldown_sec if cooldown_sec is not None else _health_cooldown_sec()))
    with _health_ledger_lock:
        entry = _model_health_ledger.setdefault(model_name, {"consecutive_fails": 0, "cooldown_until": 0.0})
        entry["consecutive_fails"] = max(int(entry.get("consecutive_fails", 0)), _health_fail_threshold())
        entry["cooldown_until"] = time.time() + effective_cooldown
    if reason:
        logger.warning(f"模型 '{model_name}' 因 {reason} 立即冷却 {effective_cooldown:.0f} 秒")


def _is_model_in_cooldown(model_name: str) -> bool:
    """检查模型是否处于冷却期"""
    with _health_ledger_lock:
        entry = _model_health_ledger.get(model_name)
        if entry and entry["cooldown_until"] > time.time():
            return True
        return False


def _mark_tool_calling_unsupported(model_name: str, *, detail: str = "") -> None:
    with _health_ledger_lock:
        _tool_call_unsupported_models.add(model_name)
    _cooldown_model_now(
        model_name,
        reason=f"工具调用不可用{(':' + detail) if detail else ''}",
        cooldown_sec=max(_tool_incompat_cooldown_sec(), 1800.0),
    )


def _is_tool_calling_unsupported(model_name: str) -> bool:
    with _health_ledger_lock:
        return model_name in _tool_call_unsupported_models


def _cooldown_provider_models(
    provider_name: str,
    *,
    reason: str,
    cooldown_sec: Optional[float] = None,
) -> None:
    """按 provider 冷却全部模型，避免鉴权错误时同组模型反复撞400。"""
    if not provider_name:
        return
    effective_cooldown = max(1.0, float(cooldown_sec if cooldown_sec is not None else _health_cooldown_sec()))
    _set_provider_auth_block(provider_name, cooldown_sec=effective_cooldown)
    cooled_models: List[str] = []
    try:
        for model in getattr(model_config, "models", []) or []:
            if str(getattr(model, "api_provider", "") or "") != provider_name:
                continue
            model_name = str(getattr(model, "name", "") or "").strip()
            if not model_name:
                continue
            _cooldown_model_now(
                model_name,
                reason=reason,
                cooldown_sec=effective_cooldown,
            )
            cooled_models.append(model_name)
    except Exception as exc:
        logger.debug(f"按 provider 冷却模型失败 provider={provider_name}: {exc}")
    if cooled_models:
        logger.warning(
            f"API提供商 '{provider_name}' 触发鉴权异常，已冷却同组模型: {', '.join(cooled_models)}"
        )


def _resolve_model_client_type(model_info: ModelInfo, api_provider: APIProvider) -> str:
    """解析当前模型应使用的客户端类型（模型级覆盖优先，provider兜底）。"""
    provider_client_type = str(getattr(api_provider, "client_type", "") or "").strip().lower()
    model_client_type = str(getattr(model_info, "client_type", "") or "").strip().lower()
    chosen_client_type = model_client_type or provider_client_type or "openai"

    registered_clients = client_registry.get_registered_clients()
    if chosen_client_type in registered_clients:
        return chosen_client_type

    if provider_client_type in registered_clients:
        logger.warning(
            f"模型 '{getattr(model_info, 'name', '?')}' 配置的 client_type='{chosen_client_type}' 未注册，"
            f"回退到 provider client_type='{provider_client_type}'"
        )
        return provider_client_type

    if "openai" in registered_clients:
        logger.warning(
            f"模型 '{getattr(model_info, 'name', '?')}' 的客户端类型无效 "
            f"(model='{chosen_client_type}', provider='{provider_client_type}')，回退到 openai"
        )
        return "openai"

    raise ValueError(
        f"未注册的客户端类型: model={chosen_client_type}, provider={provider_client_type}, "
        f"registered={list(registered_clients.keys())}"
    )


class RequestType(Enum):
    """请求类型枚举"""

    RESPONSE = "response"
    EMBEDDING = "embedding"
    AUDIO = "audio"


class LLMRequest:
    """LLM请求类"""

    def __init__(self, model_set: TaskConfig, request_type: str = "") -> None:
        self.task_name = request_type
        self.model_for_task = model_set
        self.request_type = request_type
        self.model_usage: Dict[str, Tuple[int, int, int]] = {
            model: (0, 0, 0) for model in self.model_for_task.model_list
        }
        """模型使用量记录，用于进行负载均衡，对应为(total_tokens, penalty, usage_penalty)，惩罚值是为了能在某个模型请求不给力或正在被使用的时候进行调整"""

    @staticmethod
    def _model_routing_values() -> Dict[str, Any]:
        try:
            from src.config.core_config_engine import get_core_config

            return get_core_config().resolve_module_view("model_routing").values
        except Exception:
            return {}

    @staticmethod
    def _model_matches(model_info: ModelInfo, patterns: Any) -> bool:
        if isinstance(patterns, str):
            pattern_list = [patterns]
        elif isinstance(patterns, list):
            pattern_list = patterns
        else:
            pattern_list = []
        haystack = " ".join(
            str(item or "").lower()
            for item in (
                getattr(model_info, "name", ""),
                getattr(model_info, "model_identifier", ""),
                getattr(model_info, "api_provider", ""),
                getattr(model_info, "client_type", ""),
            )
        )
        return any(str(pattern).strip().lower() in haystack for pattern in pattern_list if str(pattern).strip())

    def _thinking_explicitly_enabled(self, model_info: ModelInfo) -> bool:
        extra_params = dict(getattr(model_info, "extra_params", {}) or {})
        if bool(extra_params.get("enable_thinking") is True or extra_params.get("thinking_enabled") is True):
            return True
        routing = self._model_routing_values()
        return self._model_matches(model_info, routing.get("enable_thinking_for", []))

    def _thinking_enabled_for_request(self, model_info: ModelInfo) -> bool:
        if bool(getattr(model_info, "suppress_reasoning", False)):
            return False
        routing = self._model_routing_values()
        if self._model_matches(model_info, routing.get("disable_thinking_for", [])):
            return False
        return self._thinking_explicitly_enabled(model_info)

    def _should_suppress_reasoning(self, model_info: ModelInfo) -> bool:
        return not self._thinking_enabled_for_request(model_info)

    def _with_no_thinking_guard(self, messages: List[Message], model_info: ModelInfo) -> List[Message]:
        if not self._should_suppress_reasoning(model_info):
            return messages
        for index, item in enumerate(messages):
            if item.role == RoleType.System and isinstance(item.content, str):
                if _NO_THINKING_GUARD_MARKER in item.content:
                    return messages
                guarded_system = Message(
                    RoleType.System,
                    f"{_DEFAULT_NO_THINKING_SYSTEM_PROMPT}\n\n{item.content}",
                )
                return [*messages[:index], guarded_system, *messages[index + 1 :]]
        return [
            Message(RoleType.System, _DEFAULT_NO_THINKING_SYSTEM_PROMPT),
            *messages,
        ]

    def _effective_extra_params(self, model_info: ModelInfo) -> Dict[str, Any]:
        extra_params = dict(model_info.extra_params or {})
        thinking_enabled = self._thinking_enabled_for_request(model_info)
        try:
            provider = model_config.get_provider(model_info.api_provider)
            client_type = _resolve_model_client_type(model_info, provider)
        except Exception:
            client_type = str(getattr(model_info, "client_type", "") or "openai").lower()

        if client_type == "gemini":
            extra_params["enable_thinking"] = thinking_enabled
            if not thinking_enabled:
                extra_params["include_thoughts"] = False
                extra_params["thinking_budget"] = 0
        else:
            extra_params["enable_thinking"] = thinking_enabled
            chat_template_kwargs = dict(extra_params.get("chat_template_kwargs") or {})
            chat_template_kwargs["enable_thinking"] = thinking_enabled
            extra_params["chat_template_kwargs"] = chat_template_kwargs
            if not thinking_enabled:
                extra_params["reasoning_effort"] = "minimal"
        return extra_params

    def _request_timeout_budget(self, api_provider: APIProvider, request_type: RequestType) -> float:
        routing = self._model_routing_values()
        try:
            configured_timeout = max(1.0, float(api_provider.timeout or 1.0))
        except Exception:
            configured_timeout = 30.0

        if request_type == RequestType.EMBEDDING:
            cap = float(routing.get("embedding_request_timeout_seconds", 12.0) or 12.0)
            return max(3.0, min(configured_timeout, cap))
        if request_type == RequestType.AUDIO:
            cap = float(routing.get("audio_request_timeout_seconds", 20.0) or 20.0)
            return max(5.0, min(configured_timeout, cap))

        cap = float(routing.get("response_request_timeout_seconds", 18.0) or 18.0)
        floor = float(routing.get("response_request_timeout_floor_seconds", 5.0) or 5.0)
        return max(floor, min(configured_timeout, cap))

    def _retry_budget(self, api_provider: APIProvider, request_type: RequestType) -> int:
        try:
            configured_retry = int(api_provider.max_retry or 1)
        except Exception:
            configured_retry = 1

        routing = self._model_routing_values()
        if request_type == RequestType.RESPONSE:
            cap = int(routing.get("response_max_retry", 1) or 1)
        else:
            cap = int(routing.get("background_max_retry", 2) or 2)
        return max(1, min(configured_retry, cap))

    def _retry_sleep_seconds(self, api_provider: APIProvider, request_type: RequestType) -> float:
        routing = self._model_routing_values()
        if request_type == RequestType.RESPONSE:
            cap = float(routing.get("response_retry_interval_cap_seconds", 2.0) or 2.0)
        else:
            cap = 5.0
        try:
            return max(0.0, min(float(api_provider.retry_interval or 0.0), cap))
        except Exception:
            return cap

    def _finalize_response_content(self, content: str | None, reasoning_content: str | None, model_info: ModelInfo) -> Tuple[str, str]:
        text = content or ""
        reasoning = reasoning_content or ""
        if self._should_suppress_reasoning(model_info):
            if text:
                text = self._strip_reasoning_blocks(text)
            return text, ""
        if not reasoning and text:
            text, extracted_reasoning = self._extract_reasoning(text)
            reasoning = extracted_reasoning
        return text, reasoning

    def _filter_models_for_tools(self, model_names: List[str]) -> List[str]:
        """为工具调用任务过滤当前已知高风险不兼容模型。"""
        if not model_names:
            return model_names
        compatible_models: List[str] = []
        skipped_models: List[str] = []
        for model_name in model_names:
            if _is_tool_calling_unsupported(model_name):
                skipped_models.append(model_name)
                continue
            try:
                model_info = model_config.get_model_info(model_name)
                declared_support = getattr(model_info, "supports_tool_calling", None)
                if declared_support is False:
                    skipped_models.append(model_name)
                    continue
                api_provider = model_config.get_provider(model_info.api_provider)
                resolved_client_type = _resolve_model_client_type(model_info, api_provider)
                client = client_registry.get_client_class_instance(
                    resolved_client_type,
                    api_provider=api_provider,
                    force_new=False,
                )
                if not client.supports_function_calling():
                    skipped_models.append(model_name)
                    continue
                compatible_models.append(model_name)
            except Exception as exc:
                logger.debug(f"工具调用模型兼容性检查失败 {model_name}: {exc}")
                skipped_models.append(model_name)
        if compatible_models:
            if skipped_models:
                logger.info(
                    f"任务 '{self.request_type or '未知任务'}' 跳过疑似不兼容工具调用的模型: {', '.join(skipped_models)}"
                )
            return compatible_models
        logger.warning(
            f"任务 '{self.request_type or '未知任务'}' 未找到明确兼容工具调用的模型，"
            "将进入无工具兜底，不再回退原始工具模型列表。"
        )
        return []

    def _check_slow_request(self, time_cost: float, model_name: str) -> None:
        """检查请求是否过慢并输出警告日志

        Args:
            time_cost: 请求耗时（秒）
            model_name: 使用的模型名称
        """
        threshold = self.model_for_task.slow_threshold
        if time_cost > threshold:
            request_type_display = self.request_type or "未知任务"
            logger.warning(
                f"LLM请求耗时过长: {request_type_display} 使用模型 {model_name} 耗时 {time_cost:.1f}s（阈值: {threshold}s），请考虑使用更快的模型\n"
                f"  如果你认为该警告出现得过于频繁，请调整model_config.toml中对应任务的slow_threshold至符合你实际情况的合理值"
            )

    async def generate_response_for_image(
        self,
        prompt: str,
        image_base64: str,
        image_format: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Tuple[str, str, Optional[List[ToolCall]]]]:
        """
        为图像生成响应
        Args:
            prompt (str): 提示词
            image_base64 (str): 图像的Base64编码字符串
            image_format (str): 图像格式（如 'png', 'jpeg' 等）
        Returns:
            (Tuple[str, str, str, Optional[List[ToolCall]]]): 响应内容、推理内容、模型名称、工具调用列表
        """
        start_time = time.time()

        def message_factory(client: BaseClient) -> List[Message]:
            message_builder = MessageBuilder()
            message_builder.add_text_content(prompt)
            message_builder.add_image_content(
                image_base64=image_base64,
                image_format=image_format,
                support_formats=client.get_support_image_formats(),
            )
            return [message_builder.build()]

        response, model_info = await self._execute_request(
            request_type=RequestType.RESPONSE,
            message_factory=message_factory,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content, reasoning_content = self._finalize_response_content(
            response.content,
            response.reasoning_content,
            model_info,
        )
        tool_calls = response.tool_calls
        time_cost = time.time() - start_time
        self._check_slow_request(time_cost, model_info.name)
        if usage := response.usage:
            llm_usage_recorder.record_usage_to_database(
                model_info=model_info,
                model_usage=usage,
                user_id="system",
                request_type=self.request_type,
                endpoint="/chat/completions",
                time_cost=time_cost,
            )
        return content, (reasoning_content, model_info.name, tool_calls)

    async def generate_response_for_voice(self, voice_base64: str) -> Optional[str]:
        """
        为语音生成响应
        Args:
            voice_base64 (str): 语音的Base64编码字符串
        Returns:
            (Optional[str]): 生成的文本描述或None
        """
        response, _ = await self._execute_request(
            request_type=RequestType.AUDIO,
            audio_base64=voice_base64,
        )
        return response.content or None

    async def generate_response_async(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        raise_when_empty: bool = True,
    ) -> Tuple[str, Tuple[str, str, Optional[List[ToolCall]]]]:
        """
        异步生成响应
        Args:
            prompt (str): 提示词
            temperature (float, optional): 温度参数
            max_tokens (int, optional): 最大token数
            tools (Optional[List[Dict[str, Any]]]): 工具列表
            raise_when_empty (bool): 当响应为空时是否抛出异常
        Returns:
            (Tuple[str, str, str, Optional[List[ToolCall]]]): 响应内容、推理内容、模型名称、工具调用列表
        """
        start_time = time.time()

        def message_factory(client: BaseClient) -> List[Message]:
            message_builder = MessageBuilder()
            message_builder.add_text_content(prompt)
            return [message_builder.build()]

        tool_built = self._build_tool_options(tools)

        response, model_info = await self._execute_request(
            request_type=RequestType.RESPONSE,
            message_factory=message_factory,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_options=tool_built,
        )

        logger.debug(f"LLM请求总耗时: {time.time() - start_time}")
        logger.debug(f"LLM生成内容: {response}")

        content, reasoning_content = self._finalize_response_content(
            response.content,
            response.reasoning_content,
            model_info,
        )
        tool_calls = response.tool_calls
        if usage := response.usage:
            llm_usage_recorder.record_usage_to_database(
                model_info=model_info,
                model_usage=usage,
                user_id="system",
                request_type=self.request_type,
                endpoint="/chat/completions",
                time_cost=time.time() - start_time,
            )
        return content, (reasoning_content, model_info.name, tool_calls)

    async def generate_response_with_message_async(
        self,
        message_factory: Callable[[BaseClient], List[Message]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        raise_when_empty: bool = True,
    ) -> Tuple[str, Tuple[str, str, Optional[List[ToolCall]]]]:
        """
        异步生成响应
        Args:
            message_factory (Callable[[BaseClient], List[Message]]): 已构建好的消息工厂
            temperature (float, optional): 温度参数
            max_tokens (int, optional): 最大token数
            tools (Optional[List[Dict[str, Any]]]): 工具列表
            raise_when_empty (bool): 当响应为空时是否抛出异常
        Returns:
            (Tuple[str, str, str, Optional[List[ToolCall]]]): 响应内容、推理内容、模型名称、工具调用列表
        """
        start_time = time.time()

        tool_built = self._build_tool_options(tools)

        response, model_info = await self._execute_request(
            request_type=RequestType.RESPONSE,
            message_factory=message_factory,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_options=tool_built,
        )

        time_cost = time.time() - start_time
        logger.debug(f"LLM请求总耗时: {time_cost}")
        logger.debug(f"LLM生成内容: {response}")

        content, reasoning_content = self._finalize_response_content(
            response.content,
            response.reasoning_content,
            model_info,
        )
        tool_calls = response.tool_calls
        self._check_slow_request(time_cost, model_info.name)
        if usage := response.usage:
            llm_usage_recorder.record_usage_to_database(
                model_info=model_info,
                model_usage=usage,
                user_id="system",
                request_type=self.request_type,
                endpoint="/chat/completions",
                time_cost=time_cost,
            )
        return content, (reasoning_content, model_info.name, tool_calls)

    async def get_embedding(self, embedding_input: str) -> Tuple[List[float], str]:
        """
        获取嵌入向量

        返回值说明：
            Tuple[List[float], str]: (嵌入向量, 使用的模型名称)
            与 BaseClient.get_embedding()（仅返回 list[float]）和
            utils.get_embedding()（返回 Optional[List[float]]）不同，
            本方法额外返回模型名称用于用量追踪。

        Args:
            embedding_input (str): 获取嵌入的目标
        Returns:
            (Tuple[List[float], str]): (嵌入向量，使用的模型名称)
        """
        start_time = time.time()
        response, model_info = await self._execute_request(
            request_type=RequestType.EMBEDDING,
            embedding_input=embedding_input,
        )
        embedding = response.embedding
        if usage := response.usage:
            llm_usage_recorder.record_usage_to_database(
                model_info=model_info,
                model_usage=usage,
                user_id="system",
                request_type=self.request_type,
                endpoint="/embeddings",
                time_cost=time.time() - start_time,
            )
        if not embedding:
            raise RuntimeError("获取embedding失败")
        return embedding, model_info.name

    def _select_model(
        self,
        exclude_models: Optional[Set[str]] = None,
        tool_options: Optional[list[ToolOption]] = None,
    ) -> Tuple[ModelInfo, APIProvider, BaseClient]:
        """
        根据配置的策略选择模型：balance（负载均衡）或 random（随机选择）
        """
        candidate_models = [model for model in self.model_usage if not exclude_models or model not in exclude_models]
        if tool_options:
            candidate_models = self._filter_models_for_tools(candidate_models)
        available_models = {model: scores for model, scores in self.model_usage.items() if model in candidate_models}
        if not available_models:
            raise RuntimeError("没有可用的模型可供选择。所有模型均已尝试失败。")

        strategy = self.model_for_task.selection_strategy.lower()

        if strategy == "random":
            # 随机选择策略
            selected_model_name = random.choice(list(available_models.keys()))
        elif strategy == "balance":
            # 负载均衡策略：根据总tokens和惩罚值选择
            selected_model_name = min(
                available_models,
                key=lambda k: available_models[k][0] + available_models[k][1] * 300 + available_models[k][2] * 1000,
            )
        else:
            # 默认使用负载均衡策略
            logger.warning(f"未知的选择策略 '{strategy}'，使用默认的负载均衡策略")
            selected_model_name = min(
                available_models,
                key=lambda k: available_models[k][0] + available_models[k][1] * 300 + available_models[k][2] * 1000,
            )

        model_info = model_config.get_model_info(selected_model_name)
        api_provider = model_config.get_provider(model_info.api_provider)
        force_new_client = self.request_type == "embedding"
        resolved_client_type = _resolve_model_client_type(model_info, api_provider)
        client = client_registry.get_client_class_instance(
            resolved_client_type,
            api_provider=api_provider,
            force_new=force_new_client,
        )
        logger.debug(f"选择请求模型: {model_info.name} (策略: {strategy})")
        total_tokens, penalty, usage_penalty = self.model_usage[model_info.name]
        self.model_usage[model_info.name] = (
            total_tokens,
            penalty,
            usage_penalty + 1,
        )
        return model_info, api_provider, client

    async def _attempt_request_on_model(
        self,
        model_info: ModelInfo,
        api_provider: APIProvider,
        client: BaseClient,
        request_type: RequestType,
        message_list: List[Message],
        tool_options: list[ToolOption] | None,
        response_format: RespFormat | None,
        stream_response_handler: Optional[Callable],
        async_response_parser: Optional[Callable],
        temperature: Optional[float],
        max_tokens: Optional[int],
        embedding_input: str | None,
        audio_base64: str | None,
    ) -> APIResponse:
        """
        在单个模型上执行请求，包含针对临时错误的重试逻辑。
        如果成功，返回APIResponse。如果失败（重试耗尽或硬错误），则抛出ModelAttemptFailed异常。
        """
        retry_remain = self._retry_budget(api_provider, request_type)
        compressed_messages: Optional[List[Message]] = None
        # 标记是否已尝试过降级（去掉response_format/tool_options）
        _format_degraded = False
        _call_timeout = self._request_timeout_budget(api_provider, request_type)
        _retry_sleep = self._retry_sleep_seconds(api_provider, request_type)

        while retry_remain > 0:
            try:
                if request_type == RequestType.RESPONSE:
                    # 温度优先级：参数传入 > 模型级别配置 > extra_params > 任务配置
                    effective_temperature = temperature
                    if effective_temperature is None:
                        effective_temperature = model_info.temperature
                    if effective_temperature is None:
                        effective_temperature = (model_info.extra_params or {}).get("temperature")
                    if effective_temperature is None:
                        effective_temperature = self.model_for_task.temperature

                    # max_tokens 优先级：参数传入 > 模型级别配置 > extra_params > 任务配置
                    effective_max_tokens = max_tokens
                    if effective_max_tokens is None:
                        effective_max_tokens = model_info.max_tokens
                    if effective_max_tokens is None:
                        effective_max_tokens = (model_info.extra_params or {}).get("max_tokens")
                    if effective_max_tokens is None:
                        effective_max_tokens = self.model_for_task.max_tokens

                    # 如果已降级，剥离高级格式参数
                    _active_response_format = None if _format_degraded else response_format
                    _active_tool_options = None if _format_degraded else tool_options

                    return await asyncio.wait_for(
                        client.get_response(
                            model_info=model_info,
                            message_list=self._with_no_thinking_guard(
                                compressed_messages or message_list,
                                model_info,
                            ),
                            tool_options=_active_tool_options,
                            max_tokens=effective_max_tokens,
                            temperature=effective_temperature,
                            response_format=_active_response_format,
                            stream_response_handler=stream_response_handler,
                            async_response_parser=async_response_parser,
                            extra_params=self._effective_extra_params(model_info),
                        ),
                        timeout=_call_timeout,
                    )
                elif request_type == RequestType.EMBEDDING:
                    assert embedding_input is not None, "嵌入输入不能为空"
                    return await asyncio.wait_for(
                        client.get_embedding(
                            model_info=model_info,
                            embedding_input=embedding_input,
                            extra_params=self._effective_extra_params(model_info),
                        ),
                        timeout=_call_timeout,
                    )
                elif request_type == RequestType.AUDIO:
                    assert audio_base64 is not None, "音频Base64不能为空"
                    return await asyncio.wait_for(
                        client.get_audio_transcriptions(
                            model_info=model_info,
                            audio_base64=audio_base64,
                            extra_params=self._effective_extra_params(model_info),
                        ),
                        timeout=_call_timeout,
                    )

            except asyncio.TimeoutError as err:
                # 请求超时：视为可重试的临时错误
                retry_remain -= 1
                task_display = self.request_type or "未知任务"
                if retry_remain <= 0:
                    logger.error(
                        f"任务 '{task_display}' 的模型 '{model_info.name}' 请求超时 ({_call_timeout:.0f}s) 且重试已耗尽"
                    )
                    raise ModelAttemptFailed(
                        f"模型 '{model_info.name}' 请求超时，重试耗尽",
                        original_exception=TimeoutError(f"超时 {_call_timeout}s"),
                    ) from err
                logger.warning(
                    f"任务 '{task_display}' 的模型 '{model_info.name}' 请求超时 ({_call_timeout:.0f}s)，剩余重试: {
                        retry_remain
                    }"
                )
                await asyncio.sleep(_retry_sleep)

            except EmptyResponseException as e:
                # 空回复：通常为临时问题，单独记录并重试
                original_error_info = self._get_original_error_info(e)
                retry_remain -= 1
                task_display = self.request_type or "未知任务"
                if retry_remain <= 0:
                    logger.error(
                        f"任务 '{task_display}' 的模型 '{model_info.name}' 在多次出现空回复后仍然失败。{
                            original_error_info
                        }"
                    )
                    raise ModelAttemptFailed(
                        f"模型 '{model_info.name}' 重试耗尽",
                        original_exception=e,
                    ) from e

                logger.warning(
                    f"任务 '{task_display}' 的模型 '{model_info.name}' 返回空回复(可重试){
                        original_error_info
                    }。剩余重试次数: {retry_remain}"
                )
                await asyncio.sleep(_retry_sleep)

            except NetworkConnectionError as e:
                # 网络错误：单独记录并重试
                original_error_info = self._get_original_error_info(e)

                retry_remain -= 1
                task_display = self.request_type or "未知任务"
                if retry_remain <= 0:
                    logger.error(
                        f"任务 '{task_display}' 的模型 '{model_info.name}' 在网络错误重试用尽后仍然失败。{
                            original_error_info
                        }"
                    )
                    raise ModelAttemptFailed(
                        f"模型 '{model_info.name}' 重试耗尽",
                        original_exception=e,
                    ) from e

                logger.warning(
                    f"任务 '{task_display}' 的模型 '{model_info.name}' 遇到网络错误(可重试): {str(e)}{original_error_info}\n"
                    f"  常见原因: 如请求的API正常但APITimeoutError类型错误过多，请尝试调整模型配置中对应API Provider的timeout值\n"
                    f"  其它可能原因: 网络波动、DNS 故障、连接超时、防火墙限制或代理问题\n"
                    f"  剩余重试次数: {retry_remain}"
                )
                await asyncio.sleep(_retry_sleep)

            except RespNotOkException as e:
                original_error_info = self._get_original_error_info(e)
                task_display = self.request_type or "未知任务"
                error_detail_text = self._get_error_detail_text(e)
                auth_error_markers = (
                    "api key not valid",
                    "invalid api key",
                    "invalid api-key",
                    "invalid api_key",
                    "incorrect api key",
                    "authentication failed",
                    "unauthorized",
                    "api_key not valid",
                )
                quota_markers = (
                    "insufficient_quota",
                    "quota",
                    "余额",
                    "credit",
                )

                # 鉴权错误：整组provider快速冷却，避免同provider模型轮询风暴
                if (
                    e.status_code in (400, 401, 403)
                    and any(marker in error_detail_text for marker in auth_error_markers)
                    and not any(marker in error_detail_text for marker in quota_markers)
                ):
                    provider_name = str(getattr(model_info, "api_provider", "") or "")
                    auth_count, should_trip = _register_provider_auth_failure(
                        provider_name,
                        status_code=e.status_code,
                    )
                    if should_trip:
                        _cooldown_provider_models(
                            provider_name,
                            reason="API Key不可用",
                            cooldown_sec=max(_tool_incompat_cooldown_sec(), 1800.0),
                        )
                        logger.error(
                            f"任务 '{task_display}' 的模型 '{model_info.name}' 鉴权失败，"
                            f"provider熔断已触发(count={auth_count})，已对同provider模型执行快速冷却。"
                            f"{original_error_info}"
                        )
                    else:
                        logger.warning(
                            f"任务 '{task_display}' 的模型 '{model_info.name}' 疑似鉴权异常"
                            f"(count={auth_count}/{_provider_auth_fail_threshold(e.status_code, error_detail_text)})，"
                            "暂不熔断provider，先切换到下一个模型。"
                            f"{original_error_info}"
                        )
                    raise ModelAttemptFailed(
                        f"模型 '{model_info.name}' API Key不可用",
                        original_exception=e,
                    ) from e

                # 频率限制（429）：提取 Retry-After 头并等待
                if e.status_code == 429:
                    raw_retry_after = getattr(e, "retry_after", 0.0) or _retry_sleep
                    try:
                        retry_after = min(float(raw_retry_after), _retry_sleep)
                    except Exception:
                        retry_after = _retry_sleep
                    retry_remain -= 1
                    if retry_remain <= 0:
                        logger.error(
                            f"任务 '{task_display}' 的模型 '{model_info.name}' 触发频率限制且重试已耗尽。{
                                original_error_info
                            }"
                        )
                        raise ModelAttemptFailed(
                            f"模型 '{model_info.name}' 频率受限，重试耗尽",
                            original_exception=e,
                        ) from e
                    logger.warning(
                        f"任务 '{task_display}' 的模型 '{model_info.name}' 触发频率限制(429)，等待 {
                            retry_after
                        }s 后重试。"
                        f"剩余重试: {retry_remain}"
                    )
                    await asyncio.sleep(retry_after)
                    continue

                # 配额不足（402/403余额）：直接切换模型，不再重试当前模型
                if e.status_code in (402, 403):
                    logger.error(
                        f"任务 '{task_display}' 的模型 '{model_info.name}' 可能配额不足或余额耗尽 "
                        f"(HTTP {e.status_code})。{original_error_info}。切换到备用模型"
                    )
                    raise ModelAttemptFailed(
                        f"模型 '{model_info.name}' 配额/权限不足({e.status_code})",
                        original_exception=e,
                    ) from e

                # 服务端错误（5xx）：可重试
                if e.status_code >= 500:
                    retry_remain -= 1
                    if retry_remain <= 0:
                        logger.error(
                            f"任务 '{task_display}' 的模型 '{model_info.name}' 在遇到 {
                                e.status_code
                            } 错误并用尽重试次数后仍然失败。{original_error_info}"
                        )
                        raise ModelAttemptFailed(
                            f"模型 '{model_info.name}' 重试耗尽",
                            original_exception=e,
                        ) from e
                    logger.warning(
                        f"任务 '{task_display}' 的模型 '{model_info.name}' 遇到服务端错误({e.status_code})。"
                        f"剩余重试: {retry_remain}"
                    )
                    await asyncio.sleep(_retry_sleep)
                    continue

                # 特殊处理413，尝试压缩
                if e.status_code == 413 and message_list and not compressed_messages:
                    logger.warning(
                        f"任务 '{task_display}' 的模型 '{model_info.name}' 返回413请求体过大，尝试压缩后重试..."
                    )
                    compressed_messages = compress_messages(message_list)
                    continue

                # 特殊处理400：可能是模型不支持 response_format 或 tool_options
                if e.status_code == 400 and not _format_degraded and (response_format or tool_options):
                    # 如果有tool_options，优先切换模型而非降级（降级会丢失工具调用能力）
                    if tool_options:
                        marked_unsupported = False
                        if (
                            "function calling is not enabled" in error_detail_text
                            or "function call turn comes immediately after a user turn" in error_detail_text
                            or "tools are not supported" in error_detail_text
                            or "tool_calls are not supported" in error_detail_text
                        ):
                            _mark_tool_calling_unsupported(
                                model_info.name,
                                detail="provider明确返回不支持function-calling",
                            )
                            marked_unsupported = True
                        if not marked_unsupported:
                            _cooldown_model_now(
                                model_info.name,
                                reason="tool_options 格式不兼容",
                                cooldown_sec=max(_tool_incompat_cooldown_sec(), _health_cooldown_sec()),
                            )
                        logger.warning(
                            f"任务 '{task_display}' 的模型 '{model_info.name}' 返回400，"
                            f"该任务依赖工具调用，切换到备用模型而非降级。"
                        )
                        raise ModelAttemptFailed(
                            f"模型 '{model_info.name}' 不兼容当前tool_options格式(400)",
                            original_exception=e,
                        ) from e
                    logger.warning(
                        f"任务 '{task_display}' 的模型 '{model_info.name}' 返回400，"
                        f"尝试降级（去除 response_format）后重试。"
                    )
                    _format_degraded = True
                    continue

                # 其他不可重试的HTTP错误
                logger.warning(
                    f"任务 '{task_display}' 的模型 '{model_info.name}' 遇到不可重试的HTTP错误: {str(e)}{
                        original_error_info
                    }"
                )
                raise ModelAttemptFailed(
                    f"模型 '{model_info.name}' 遇到硬错误",
                    original_exception=e,
                ) from e

            except Exception as e:
                logger.error(traceback.format_exc())

                original_error_info = self._get_original_error_info(e)
                task_display = self.request_type or "未知任务"

                logger.warning(
                    f"任务 '{task_display}' 的模型 '{model_info.name}' 遇到未知的不可重试错误: {str(e)}{
                        original_error_info
                    }"
                )
                raise ModelAttemptFailed(
                    f"模型 '{model_info.name}' 遇到硬错误",
                    original_exception=e,
                ) from e

        raise ModelAttemptFailed(
            f"任务 '{self.request_type or '未知任务'}' 的模型 '{
                model_info.name
            }' 未被尝试，因为重试次数已配置为0或更少。"
        )

    async def _execute_request(
        self,
        request_type: RequestType,
        message_factory: Optional[Callable[[BaseClient], List[Message]]] = None,
        tool_options: list[ToolOption] | None = None,
        response_format: RespFormat | None = None,
        stream_response_handler: Optional[Callable] = None,
        async_response_parser: Optional[Callable] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        embedding_input: str | None = None,
        audio_base64: str | None = None,
        allow_tool_degrade_fallback: bool = True,
    ) -> Tuple[APIResponse, ModelInfo]:
        """
        调度器函数，负责模型选择、故障切换。
        """
        failed_models_this_request: Set[str] = set()
        max_attempts = len(self.model_for_task.model_list)
        last_exception: Optional[Exception] = None
        saw_non_tool_incompat_error = False

        for _ in range(max_attempts):
            try:
                model_info, api_provider, client = self._select_model(
                    exclude_models=failed_models_this_request,
                    tool_options=tool_options,
                )
            except RuntimeError as select_err:
                if tool_options is not None:
                    last_exception = select_err
                    logger.warning(
                        f"任务 '{self.request_type or '未知任务'}' 工具模式无可用模型，"
                        "将尝试无工具兜底。"
                    )
                    break
                raise

            # 跳过处于冷却期的模型（除非已无其他选择）
            if _is_model_in_cooldown(model_info.name):
                remaining_healthy = [
                    m
                    for m in self.model_for_task.model_list
                    if m not in failed_models_this_request and not _is_model_in_cooldown(m)
                ]
                if remaining_healthy:
                    logger.debug(f"模型 '{model_info.name}' 处于冷却期，跳过")
                    failed_models_this_request.add(model_info.name)
                    continue
                provider_name = str(getattr(model_info, "api_provider", "") or "")
                if _is_provider_auth_blocked(provider_name):
                    remain = _provider_auth_block_remaining(provider_name)
                    logger.warning(
                        f"模型 '{model_info.name}' 所属provider '{provider_name}' 处于鉴权熔断期"
                        f"({remain:.0f}s)，跳过强制尝试。"
                    )
                    failed_models_this_request.add(model_info.name)
                    last_exception = RuntimeError(
                        f"provider '{provider_name}' 处于鉴权熔断期，暂不重试模型 '{model_info.name}'"
                    )
                    continue
                if tool_options is not None:
                    logger.warning(
                        f"模型 '{model_info.name}' 处于冷却期且当前任务依赖工具调用，"
                        "跳过本轮请求并进入无工具兜底。"
                    )
                    failed_models_this_request.add(model_info.name)
                    continue
                logger.warning(f"模型 '{model_info.name}' 处于冷却期但无其他可用模型，强制尝试")

            message_list = []
            if message_factory:
                message_list = message_factory(client)

            try:
                response = await self._attempt_request_on_model(
                    model_info,
                    api_provider,
                    client,
                    request_type,
                    message_list=message_list,
                    tool_options=tool_options,
                    response_format=response_format,
                    stream_response_handler=stream_response_handler,
                    async_response_parser=async_response_parser,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    embedding_input=embedding_input,
                    audio_base64=audio_base64,
                )
                total_tokens, penalty, usage_penalty = self.model_usage[model_info.name]
                if response_usage := response.usage:
                    total_tokens += response_usage.total_tokens
                self.model_usage[model_info.name] = (
                    total_tokens,
                    penalty,
                    usage_penalty - 1,
                )
                _mark_model_success(model_info.name)
                _clear_provider_auth_failures(str(getattr(model_info, "api_provider", "") or ""))
                return response, model_info

            except ModelAttemptFailed as e:
                last_exception = e.original_exception or e
                logger.warning(f"模型 '{model_info.name}' 尝试失败，切换到下一个模型。原因: {e}")
                total_tokens, penalty, usage_penalty = self.model_usage[model_info.name]
                self.model_usage[model_info.name] = (
                    total_tokens,
                    penalty + 1,
                    usage_penalty - 1,
                )
                failed_models_this_request.add(model_info.name)
                _mark_model_failure(model_info.name)

                tool_incompat = (
                    tool_options is not None
                    and isinstance(last_exception, RespNotOkException)
                    and last_exception.status_code == 400
                )
                if not tool_incompat:
                    saw_non_tool_incompat_error = True

                if isinstance(last_exception, RespNotOkException) and last_exception.status_code == 400:
                    logger.warning("收到客户端错误 (400)，跳过当前模型并继续尝试其他模型。")
                    continue

        if tool_options is not None and allow_tool_degrade_fallback and not saw_non_tool_incompat_error:
            request_name = (self.request_type or "").lower()
            if request_name.startswith("memory.react"):
                utils_task_cfg = getattr(model_config.model_task_config, "utils", None)
                if utils_task_cfg and getattr(utils_task_cfg, "model_list", None):
                    logger.warning(
                        f"任务 '{self.request_type or '未知任务'}' 的工具调用模型均不兼容当前tool_options，"
                        "降级为无工具模式并切换到 utils 模型池重试一次。"
                    )
                    fallback_request = LLMRequest(
                        model_set=utils_task_cfg,
                        request_type=f"{self.request_type}.fallback",
                    )
                    return await fallback_request._execute_request(
                        request_type=request_type,
                        message_factory=message_factory,
                        tool_options=None,
                        response_format=response_format,
                        stream_response_handler=stream_response_handler,
                        async_response_parser=async_response_parser,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        embedding_input=embedding_input,
                        audio_base64=audio_base64,
                        allow_tool_degrade_fallback=False,
                    )
            logger.warning(
                f"任务 '{self.request_type or '未知任务'}' 的工具调用模型均不兼容当前tool_options，"
                "降级为无工具模式重试一次。"
            )
            return await self._execute_request(
                request_type=request_type,
                message_factory=message_factory,
                tool_options=None,
                response_format=response_format,
                stream_response_handler=stream_response_handler,
                async_response_parser=async_response_parser,
                temperature=temperature,
                max_tokens=max_tokens,
                embedding_input=embedding_input,
                audio_base64=audio_base64,
                allow_tool_degrade_fallback=False,
            )

        logger.error(f"所有 {max_attempts} 个模型均尝试失败。")
        if last_exception:
            raise last_exception
        raise RuntimeError("请求失败，所有可用模型均已尝试失败。")

    def _build_tool_options(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[ToolOption]]:
        # sourcery skip: extract-method
        """构建工具选项列表"""
        if not tools:
            return None
        tool_options: List[ToolOption] = []
        for tool in tools:
            tool_legal = True
            tool_options_builder = ToolOptionBuilder()
            tool_options_builder.set_name(tool.get("name", ""))
            tool_options_builder.set_description(tool.get("description", ""))
            parameters: List[Tuple[str, str, str, bool, List[str] | None]] = tool.get("parameters", [])
            for param in parameters:
                try:
                    assert isinstance(param, tuple) and len(param) == 5, "参数必须是包含5个元素的元组"
                    assert isinstance(param[0], str), "参数名称必须是字符串"
                    assert isinstance(param[1], ToolParamType), "参数类型必须是ToolParamType枚举"
                    assert isinstance(param[2], str), "参数描述必须是字符串"
                    assert isinstance(param[3], bool), "参数是否必填必须是布尔值"
                    assert isinstance(param[4], list) or param[4] is None, "参数枚举值必须是列表或None"
                    tool_options_builder.add_param(
                        name=param[0],
                        param_type=param[1],
                        description=param[2],
                        required=param[3],
                        enum_values=param[4],
                    )
                except AssertionError as ae:
                    tool_legal = False
                    logger.error(f"{param[0]} 参数定义错误: {str(ae)}")
                except Exception as e:
                    tool_legal = False
                    logger.error(f"构建工具参数失败: {str(e)}")
            if tool_legal:
                tool_options.append(tool_options_builder.build())
        return tool_options or None

    @staticmethod
    def _extract_reasoning(content: str) -> Tuple[str, str]:
        """CoT思维链提取，向后兼容"""
        match = re.search(r"(?:<think>)?(.*?)</think>", content, re.DOTALL)
        content = re.sub(r"(?:<think>)?.*?</think>", "", content, flags=re.DOTALL, count=1).strip()
        reasoning = match.group(1).strip() if match and match.group(1) else ""
        return content, reasoning

    @staticmethod
    def _strip_reasoning_blocks(content: str) -> str:
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    @staticmethod
    def _get_original_error_info(e: Exception) -> str:
        """获取原始错误信息"""
        if e.__cause__:
            original_error_type = type(e.__cause__).__name__
            original_error_msg = str(e.__cause__)
            return f"\n  底层异常类型: {original_error_type} \n  底层异常信息: {original_error_msg} "
        return ""

    @staticmethod
    def _get_error_detail_text(e: Exception) -> str:
        """提取可用于规则匹配的错误详情文本（小写）。"""
        parts: List[str] = []
        try:
            msg = getattr(e, "message", None)
            if msg:
                parts.append(str(msg))
        except Exception:
            pass
        try:
            parts.append(str(e))
        except Exception:
            pass
        try:
            cause = getattr(e, "__cause__", None)
            if cause:
                parts.append(str(cause))
        except Exception:
            pass
        return " | ".join(parts).lower()


class FlowContextRegistry:
    """流上下文注册器
    负责为每个聊天流ID维护独立的并发锁和状态信息
    采用注册表模式确保心流管道中voice_driven_reply、planned_reply和主动执行器能正确关联上下文
    防止并发过载引发GeminiProxy空响应问题
    """

    def __init__(self):
        self._stream_contexts: Dict[str, Dict[str, Any]] = {}
        self._stream_locks: Dict[str, asyncio.Semaphore] = {}
        self._global_lock: asyncio.Lock | None = None
        self._access_history: Dict[str, List[float]] = {}
        self._max_concurrent = 2
        self._cleanup_threshold = 1800.0

    async def register_stream(self, stream_id: str) -> None:
        if self._global_lock is None:
            self._global_lock = asyncio.Lock()
        async with self._global_lock:
            if stream_id not in self._stream_contexts:
                self._stream_contexts[stream_id] = {
                    "id": stream_id,
                    "bind_time": time.time(),
                    "request_count": 0,
                    "last_access": time.time(),
                    "status": "active",
                    "success_count": 0,
                    "fail_count": 0,
                }
                self._stream_locks[stream_id] = asyncio.Semaphore(self._max_concurrent)
                self._access_history[stream_id] = []
                logger.debug(f"流上下文注册成功: {stream_id}")

    def get_stream_context(self, stream_id: str | None = None) -> Dict[str, Any] | None:
        if stream_id is None:
            try:
                import contextvars

                var = contextvars.ContextVar("current_flow_stream", default=None)
                current = var.get()
                if current and isinstance(current, str):
                    stream_id = current
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
            if not stream_id and self._stream_contexts:
                stream_id = next(iter(self._stream_contexts.keys()), None)
        if stream_id and stream_id in self._stream_contexts:
            ctx = self._stream_contexts[stream_id]
            ctx["last_access"] = time.time()
            ctx["request_count"] = ctx.get("request_count", 0) + 1
            self._access_history.setdefault(stream_id, []).append(time.time())
            return ctx
        return None

    async def bind_stream(self, stream_id: str) -> None:
        await self.register_stream(stream_id)
        try:
            import contextvars

            var = contextvars.ContextVar("current_flow_stream", default=None)
            var.set(stream_id)
            logger.info(f"流 {stream_id} 上下文绑定完成，并发上限 {self._max_concurrent}")
        except Exception as bind_err:
            logger.debug(f"上下文变量设置失败，使用注册表模式: {bind_err}")

    async def acquire_slot(self, stream_id: str) -> bool:
        await self.register_stream(stream_id)
        sem = self._stream_locks.get(stream_id)
        if not sem:
            return False
        try:
            await asyncio.wait_for(sem.acquire(), timeout=6.0)
            if stream_id in self._stream_contexts:
                self._stream_contexts[stream_id]["status"] = "locked"
            return True
        except asyncio.TimeoutError:
            logger.warning(f"流 {stream_id} 并发槽获取超时，可能代理负载较高")
            return False

    def release_slot(self, stream_id: str) -> None:
        sem = self._stream_locks.get(stream_id)
        if sem:
            try:
                if sem._value < self._max_concurrent:
                    sem.release()
                if stream_id in self._stream_contexts:
                    self._stream_contexts[stream_id]["status"] = "active"
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")

    def get_stats(self, stream_id: str) -> Dict[str, Any]:
        ctx = self.get_stream_context(stream_id)
        if not ctx:
            return {"status": "unregistered", "requests": 0}
        hist = self._access_history.get(stream_id, [])
        recent = sum(1 for t in hist if time.time() - t < 300)
        return {
            "stream_id": stream_id,
            "total_requests": ctx.get("request_count", 0),
            "recent_activity": recent,
            "status": ctx.get("status", "unknown"),
            "uptime_seconds": int(time.time() - ctx.get("bind_time", 0)),
            "success_rate": round(ctx.get("success_count", 0) / max(1, ctx.get("request_count", 1)), 2),
        }

    def cleanup_stream(self, stream_id: str) -> bool:
        if stream_id in self._stream_contexts:
            self._stream_contexts.pop(stream_id, None)
            self._stream_locks.pop(stream_id, None)
            self._access_history.pop(stream_id, None)
            logger.debug(f"流 {stream_id} 上下文已清理")
            return True
        return False

    def cleanup_inactive(self) -> int:
        now = time.time()
        inactive = [
            sid
            for sid, c in list(self._stream_contexts.items())
            if now - c.get("last_access", 0) > self._cleanup_threshold
        ]
        for sid in inactive:
            self.cleanup_stream(sid)
        if inactive:
            logger.info(f"自动清理了 {len(inactive)} 个不活跃流上下文")
        return len(inactive)

    async def reset_registry(self) -> None:
        async with self._global_lock:
            self._stream_contexts.clear()
            for sem in self._stream_locks.values():
                while not sem.locked():
                    try:
                        sem.release()
                    except Exception:
                        break
            self._stream_locks.clear()
            self._access_history.clear()
            logger.warning("流上下文注册器已完全重置")


_stream_registry = FlowContextRegistry()


def bind_stream_context(stream_id: str) -> None:
    """绑定聊天流ID到当前任务上下文
    必须在_execute_voice_driven_reply、_execute_planned_reply和主动回复执行器入口处调用
    确保LLMRequest能感知具体流ID以应用独立并发守卫
    这是解决memory.react和voice_driven_reply空响应超时问题的关键环节
    """
    if not stream_id or not isinstance(stream_id, str):
        logger.warning("bind_stream_context收到无效stream_id")
        return
    try:
        safe_create_task(_stream_registry.bind_stream(stream_id), name="bind_stream")
    except RuntimeError as re_err:
        logger.debug(f"create_task失败: {re_err}，尝试运行循环方式")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_stream_registry.bind_stream(stream_id))
        except Exception as loop_err:
            logger.debug(f"loop创建任务失败: {loop_err}")
            try:
                asyncio.run(_stream_registry.bind_stream(stream_id))
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
    try:
        import contextvars

        ctx_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_flow_stream", default=None)
        ctx_var.set(stream_id)
    except Exception as ctx_err:
        logger.debug(f"contextvars绑定辅助失败: {ctx_err}")
    logger.debug(f"流上下文绑定成功: {stream_id}")


def get_current_flow_stream() -> str | None:
    """查询当前绑定的流上下文ID
    可被LLMRequest._attempt_request_on_model等方法调用以获取stream_id
    """
    try:
        import contextvars

        var: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_flow_stream", default=None)
        value = var.get()
        if value and isinstance(value, str):
            return value
    except Exception as _exc:
        logger.debug(f"非关键异常: {_exc}")
    ctx = _stream_registry.get_stream_context()
    return ctx.get("id") if ctx else None


def apply_context_to_llm_request(llm_req: Any, stream_id: str | None = None) -> None:
    """将上下文应用到LLMRequest实例
    如果提供stream_id则绑定，否则尝试自动获取
    """
    if stream_id is None:
        stream_id = get_current_flow_stream()
    if stream_id:
        try:
            bind_stream_context(stream_id)
            if hasattr(llm_req, "task_name"):
                logger.debug(f"LLMRequest {llm_req.task_name} 已关联流 {stream_id}")
        except Exception as app_err:
            logger.debug(f"应用上下文到请求时异常: {app_err}")


def log_all_stream_stats() -> None:
    """输出所有活跃流的统计信息"""
    for sid in list(_stream_registry._stream_contexts.keys()):
        stats = _stream_registry.get_stats(sid)
        logger.info(
            f"流上下文统计 [{sid}] 请求={stats.get('total_requests', 0)} "
            f"最近={stats.get('recent_activity', 0)} 状态={stats.get('status')} "
            f"在线={stats.get('uptime_seconds', 0)}s"
        )


__all__ = [
    "bind_stream_context",
    "get_current_flow_stream",
    "FlowContextRegistry",
    "apply_context_to_llm_request",
    "log_all_stream_stats",
]

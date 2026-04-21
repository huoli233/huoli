from typing import Tuple, Dict, List, Any, Optional, Callable
from src.common.logger import get_logger
from src.llm_models.payload_content.tool_option import ToolCall
from src.llm_models.payload_content.message import Message
from src.llm_models.model_client.base_client import BaseClient
from src.llm_models.utils_model import LLMRequest
from src.config.config import model_config

logger = get_logger("LLMAPI")


def get_available_models() -> Dict[str, Any]:
    try:
        models = {}
        if hasattr(model_config, "model_task_config"):
            task_config = model_config.model_task_config
            attrs = dir(task_config)
            for attr in attrs:
                if not attr.startswith("__"):
                    try:
                        value = getattr(task_config, attr)
                        if not callable(value):
                            models[attr] = value
                    except Exception as e:
                        logger.debug(f"[LLMAPI] 获取属性 {attr} 失败: {e}")
                        continue
        return models
    except Exception as e:
        logger.error(f"[LLMAPI] 获取可用模型失败: {e}")
        return {}


async def generate_with_model(
    prompt: str,
    model_set: Any,
    request_type: str = "plugin.generate",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[bool, str, str, str]:
    try:
        logger.debug(f"[LLMAPI] 使用模型生成内容，请求类型: {request_type}")
        llm_request = LLMRequest(
            model_set=model_set, request_type=request_type
        )
        response, (reasoning_content, model_name, _) = (
            await llm_request.generate_response_async(
                prompt, temperature=temperature, max_tokens=max_tokens
            )
        )
        return True, response, reasoning_content or "", model_name or ""
    except Exception as e:
        error_msg = f"生成内容时出错: {str(e)}"
        logger.error(f"[LLMAPI] {error_msg}")
        return False, error_msg, "", ""


async def generate_with_model_with_tools(
    prompt: str,
    model_set: Any,
    tool_options: Optional[List[Dict[str, Any]]] = None,
    request_type: str = "plugin.generate",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[bool, str, str, str, Optional[List[ToolCall]]]:
    try:
        logger.debug(
            f"[LLMAPI] 使用模型和工具生成内容，请求类型: {request_type}"
        )
        llm_request = LLMRequest(
            model_set=model_set, request_type=request_type
        )
        response, (reasoning_content, model_name, tool_call) = (
            await llm_request.generate_response_async(
                prompt,
                tools=tool_options,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        return (
            True,
            response,
            reasoning_content or "",
            model_name or "",
            tool_call,
        )
    except Exception as e:
        error_msg = f"生成内容时出错: {str(e)}"
        logger.error(f"[LLMAPI] {error_msg}")
        return False, error_msg, "", "", None


async def generate_with_model_with_message_factory(
    message_factory: Callable[[BaseClient], List[Message]],
    model_set: Any,
    tool_options: Optional[List[Dict[str, Any]]] = None,
    request_type: str = "plugin.generate",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[bool, str, str, str, Optional[List[ToolCall]]]:
    try:
        logger.debug(
            f"[LLMAPI] 使用消息工厂生成内容，请求类型: {request_type}"
        )
        llm_request = LLMRequest(
            model_set=model_set, request_type=request_type
        )
        response, (reasoning_content, model_name, tool_call) = (
            await llm_request.generate_response_with_message_async(
                message_factory=message_factory,
                tools=tool_options,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        return (
            True,
            response,
            reasoning_content or "",
            model_name or "",
            tool_call,
        )
    except Exception as e:
        error_msg = f"生成内容时出错: {str(e)}"
        logger.error(f"[LLMAPI] {error_msg}")
        return False, error_msg, "", "", None


async def generate_stream_with_model(
    prompt: str,
    model_set: Any,
    request_type: str = "plugin.generate_stream",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    try:
        logger.debug(
            f"[LLMAPI] 使用流式模型生成内容，请求类型: {request_type}"
        )
        llm_request = LLMRequest(
            model_set=model_set, request_type=request_type
        )
        async for chunk in llm_request.generate_stream_response_async(
            prompt, temperature=temperature, max_tokens=max_tokens
        ):
            yield chunk
    except Exception as e:
        error_msg = f"流式生成内容时出错: {str(e)}"
        logger.error(f"[LLMAPI] {error_msg}")
        yield error_msg


def get_model_config(model_name: str) -> Optional[Any]:
    try:
        models = get_available_models()
        return models.get(model_name)
    except Exception as e:
        logger.error(f"[LLMAPI] 获取模型配置失败: {model_name}, error={e}")
        return None


def estimate_tokens(text: str) -> int:
    try:
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return chinese_chars * 2 + other_chars // 4
    except Exception as e:
        logger.error(f"[LLMAPI] 估算token数失败: {e}")
        return len(text)

import asyncio
from abc import ABC, abstractmethod
from typing import Callable, Any, Optional

from src.config.api_ada_configs import ModelInfo, APIProvider
from ..payload_content.message import Message
from ..payload_content.resp_format import ResponseFormat
from ..payload_content.tool_option import ToolOption, ToolCall
from ._registry import UsageRecord, APIResponse


class BaseClient(ABC):
    """Base class for all API clients"""

    def __init__(self, api_provider: APIProvider):
        self.api_provider = api_provider

    @abstractmethod
    async def get_response(
        self,
        model_info: ModelInfo,
        message_list: list[Message],
        tool_options: list[ToolOption] | None = None,
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.7,
        response_format: ResponseFormat | None = None,
        interrupt_flag: asyncio.Event | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """获取对话响应"""
        pass

    @abstractmethod
    async def get_embedding(
        self,
        model_info: ModelInfo,
        text: str,
    ) -> list[float]:
        """获取文本嵌入向量"""
        pass

    @abstractmethod
    async def get_audio_transcriptions(
        self,
        model_info: ModelInfo,
        audio_base64: str,
    ) -> str:
        """获取音频转录文本"""
        pass

    def supports_reasoning(self) -> bool:
        return False

    def supports_function_calling(self) -> bool:
        return True

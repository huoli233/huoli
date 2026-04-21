from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING, Dict, Any
from src.common.data_models.base_data_model import BaseDataModel

if TYPE_CHECKING:
    from .message_data_model import ReplySetModel


@dataclass
class LLMGenerationDataModel(BaseDataModel):
    content: Optional[str] = None
    reasoning: Optional[str] = None
    model: Optional[str] = None
    tool_calls: Optional[List] = None
    prompt: Optional[str] = None
    selected_expressions: Optional[List[int]] = None
    reply_set: Optional["ReplySetModel"] = None
    timing: Optional[Dict[str, Any]] = None
    processed_output: Optional[List[str]] = None
    timing_logs: Optional[List[str]] = None

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, TypeVar

ModelT = TypeVar("ModelT", bound="BaseDataModel")


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_payload(item) for key, item in value.items()}
    return value


class BaseDataModel:
    """数据模型的最小公共基类。"""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self) if is_dataclass(self) else dict(self.__dict__)
        return _normalize_payload(payload)

    @classmethod
    def from_dict(cls: type[ModelT], data: Dict[str, Any] | None) -> ModelT:
        return cls(**(data or {}))

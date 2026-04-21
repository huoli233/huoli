from copy import copy as _shallow_copy, deepcopy as _deep_copy
from dataclasses import asdict, is_dataclass, replace as _dataclass_replace, fields as _dataclass_fields
from enum import Enum
from types import UnionType
from typing import Any, Dict, Iterable, TypeVar, Union, get_args, get_origin, get_type_hints

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


def _strip_none_recursive(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_none_recursive(item) for item in value if item is not None]
    if isinstance(value, dict):
        return {
            key: _strip_none_recursive(item)
            for key, item in value.items()
            if item is not None
        }
    return value


def _apply_dump_filters(
    payload: Dict[str, Any],
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    exclude_none: bool = False,
) -> Dict[str, Any]:
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude or [])
    filtered: Dict[str, Any] = {}
    for key, value in payload.items():
        if include_set is not None and key not in include_set:
            continue
        if key in exclude_set:
            continue
        filtered[key] = value
    if exclude_none:
        return _strip_none_recursive(filtered)
    return filtered


def _matches_annotation(value: Any, annotation: Any) -> bool:
    if annotation in (None, Any):
        return True
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return any(_matches_annotation(value, arg) for arg in get_args(annotation) if arg is not type(None))
    if origin is list:
        return isinstance(value, list)
    if origin is dict:
        return isinstance(value, dict)
    if origin is tuple:
        return isinstance(value, tuple)
    if origin is set:
        return isinstance(value, set)
    if isinstance(annotation, type):
        try:
            return isinstance(value, annotation)
        except TypeError:
            return False
    return False


def _coerce_primitive(value: Any, annotation: type[Any]) -> Any:
    if isinstance(value, annotation):
        return value
    try:
        if annotation is bool and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            return value
        return annotation(value)
    except (TypeError, ValueError):
        return value


def _coerce_model_value(value: Any, annotation: Any) -> Any:
    if annotation in (None, Any) or value is None:
        return value

    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        candidates = [arg for arg in get_args(annotation) if arg is not type(None)]
        for candidate in candidates:
            coerced = _coerce_model_value(value, candidate)
            if _matches_annotation(coerced, candidate):
                return coerced
        return value

    if origin is list and isinstance(value, list):
        inner = get_args(annotation)
        inner_type = inner[0] if inner else Any
        return [_coerce_model_value(item, inner_type) for item in value]

    if origin is dict and isinstance(value, dict):
        args = get_args(annotation)
        key_type = args[0] if len(args) > 0 else Any
        value_type = args[1] if len(args) > 1 else Any
        return {
            _coerce_model_value(key, key_type): _coerce_model_value(item, value_type)
            for key, item in value.items()
        }

    if origin is tuple and isinstance(value, (list, tuple)):
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_model_value(item, args[0]) for item in value)
        if args:
            return tuple(
                _coerce_model_value(item, args[min(idx, len(args) - 1)])
                for idx, item in enumerate(value)
            )
        return tuple(value)

    if origin is set and isinstance(value, (list, tuple, set)):
        inner = get_args(annotation)
        inner_type = inner[0] if inner else Any
        return {_coerce_model_value(item, inner_type) for item in value}

    if isinstance(annotation, type):
        if issubclass(annotation, Enum):
            if isinstance(value, annotation):
                return value
            try:
                return annotation(value)
            except (TypeError, ValueError):
                try:
                    return annotation[str(value)]
                except (KeyError, TypeError):
                    return value
        if issubclass(annotation, BaseDataModel) and isinstance(value, dict):
            return annotation.from_dict(value)
        if is_dataclass(annotation) and isinstance(value, dict):
            try:
                return annotation(**value)
            except TypeError:
                return value
        if annotation in (str, int, float, bool):
            return _coerce_primitive(value, annotation)

    return value


class BaseDataModel:
    """数据模型的最小公共基类。"""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self) if is_dataclass(self) else dict(self.__dict__)
        return _normalize_payload(payload)

    def model_dump(
        self,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        exclude_none: bool = False,
        **_kwargs,
    ) -> Dict[str, Any]:
        """兼容 Pydantic 风格导出接口。"""
        return _apply_dump_filters(
            self.to_dict(),
            include=include,
            exclude=exclude,
            exclude_none=exclude_none,
        )

    def dict(
        self,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        exclude_none: bool = False,
        **_kwargs,
    ) -> Dict[str, Any]:
        """兼容旧式 .dict() 导出接口。"""
        return self.model_dump(
            include=include,
            exclude=exclude,
            exclude_none=exclude_none,
            **_kwargs,
        )

    def model_copy(self: ModelT, *, update: Dict[str, Any] | None = None) -> ModelT:
        """兼容 Pydantic v2 的 model_copy(update=...) 调用风格。"""
        update = dict(update or {})
        if is_dataclass(self):
            try:
                return _dataclass_replace(self, **update)
            except TypeError:
                pass
        cloned = _shallow_copy(self)
        for key, value in update.items():
            setattr(cloned, key, value)
        return cloned

    def copy(self: ModelT, *, update: Dict[str, Any] | None = None, deep: bool = False) -> ModelT:
        """兼容旧式 copy(update=..., deep=...) 调用。"""
        update = dict(update or {})
        if deep:
            cloned = _deep_copy(self)
            for key, value in update.items():
                setattr(cloned, key, value)
            return cloned
        return self.model_copy(update=update)

    @classmethod
    def from_dict(cls: type[ModelT], data: Dict[str, Any] | None) -> ModelT:
        data = data or {}
        converted = data
        if is_dataclass(cls):
            try:
                type_hints = get_type_hints(cls)
            except Exception:
                type_hints = {field.name: field.type for field in _dataclass_fields(cls)}
            converted = {
                key: _coerce_model_value(value, type_hints.get(key))
                for key, value in data.items()
            }
        try:
            return cls(**converted)
        except TypeError:
            obj = cls.__new__(cls)
            if is_dataclass(cls):
                for field in _dataclass_fields(cls):
                    if field.name not in converted:
                        default = field.default
                        if default is not _MISSING:
                            setattr(obj, field.name, default)
                        elif field.default_factory is not _MISSING:
                            setattr(obj, field.name, field.default_factory())
                for key, value in converted.items():
                    setattr(obj, key, value)
                post_init = getattr(obj, "__post_init__", None)
                if post_init is not None and callable(post_init):
                    try:
                        post_init()
                    except Exception:
                        pass
            else:
                for key, value in converted.items():
                    setattr(obj, key, value)
            return obj


_MISSING = object()

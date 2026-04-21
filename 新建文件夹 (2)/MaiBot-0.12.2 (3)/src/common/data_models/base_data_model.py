import copy
from typing import Any


class BaseDataModel:
    def deep_clone(self):
        return copy.deepcopy(self)


def convert_model_to_dict(obj: Any) -> Any:
    def _convert(value: Any) -> Any:
        if isinstance(value, type) and issubclass(value, BaseDataModel):
            return {k: _convert(v) for k, v in value.__dict__.items() if not k.startswith("__") and not callable(v)}
        if isinstance(value, BaseDataModel):
            return {k: _convert(v) for k, v in vars(value).items()}
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_convert(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_convert(v) for v in value)
        if isinstance(value, set):
            return {_convert(v) for v in value}
        return value
    result = _convert(obj)
    def flatten_dict(target_dict: dict):
        flat_dict = {}
        for k, v in target_dict.items():
            if isinstance(v, dict):
                sub_flat = flatten_dict(v)
                flat_dict.update(sub_flat)
            else:
                flat_dict[k] = v
        return flat_dict
    return flatten_dict(result) if isinstance(result, dict) else result

import re
from typing import Any
import tomlkit  # noqa: 第三方库依赖：tomlkit（TOML文件处理）
from tomlkit.items import AoT, Table, Array  # noqa: 第三方库依赖：tomlkit


def _format_value(obj: Any, threshold: int, depth: int = 0) -> Any:
    """递归格式化 TOML 值"""
    if isinstance(obj, AoT):
        for item in obj:
            _format_value(item, threshold, depth)
        return obj
    if isinstance(obj, (dict, Table)):
        for k, v in obj.items():
            _format_value(v, threshold, depth + 1)
        return obj
    if isinstance(obj, list):
        should_multiline = depth == 0 and len(obj) > threshold
        if isinstance(obj, Array):
            obj.multiline(should_multiline)
            for i, item in enumerate(obj):
                obj[i] = _format_value(item, threshold, depth + 1)
            return obj
        arr = tomlkit.array()
        arr.multiline(should_multiline)
        for item in obj:
            arr.append(_format_value(item, threshold, depth + 1))
        return arr
    return obj


def _merge_dict(target: Any, source: Any) -> None:
    """递归合并字典，保留目标格式和注释"""
    if (
        isinstance(source, list)
        or not isinstance(source, dict)
        or not isinstance(target, dict)
    ):
        return
    for key, value in source.items():
        if key == "version":
            continue
        if key in target:
            target_value = target[key]
            if isinstance(value, dict) and isinstance(target_value, dict):
                _merge_dict(target_value, value)
            else:
                try:
                    target[key] = tomlkit.item(value)
                except (TypeError, ValueError):
                    target[key] = value
        else:
            try:
                target[key] = tomlkit.item(value)
            except (TypeError, ValueError):
                target[key] = value


def save_toml_formatted(
    data: Any,
    file_path: str,
    multiline_threshold: int = 1,
    preserve_comments: bool = True,
) -> None:
    """格式化保存 TOML 文件"""
    import os
    from tomlkit import TOMLDocument

    if (
        preserve_comments
        and os.path.exists(file_path)
        and not isinstance(data, TOMLDocument)
    ):
        with open(file_path, "r", encoding="utf-8") as f:
            doc = tomlkit.load(f)
        _merge_dict(doc, data)
        data = doc
    formatted = (
        _format_value(data, multiline_threshold)
        if multiline_threshold >= 0
        else data
    )
    output = tomlkit.dumps(formatted)
    output = re.sub(r"\n{3,}", "\n\n", output)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(output)


save_toml_with_format = save_toml_formatted


def format_toml(data: Any, multiline_threshold: int = 1) -> str:
    """格式化 TOML 数据为字符串"""
    formatted = (
        _format_value(data, multiline_threshold)
        if multiline_threshold >= 0
        else data
    )
    output = tomlkit.dumps(formatted)
    return re.sub(r"\n{3,}", "\n\n", output)


def load_toml(file_path: str) -> dict:
    """加载 TOML 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return tomlkit.load(f)


def dump_toml(data: dict) -> str:
    """将字典序列化为 TOML 字符串"""
    return tomlkit.dumps(data)


def format_toml_string(data: Any) -> str:
    """格式化 TOML 数据为字符串（别名）"""
    return format_toml(data)


def get_short_hash(text: str, length: int = 8) -> str:
    """生成短哈希值"""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]

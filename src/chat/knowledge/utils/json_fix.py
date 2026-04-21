import json
from typing import Any, Optional, Union


def find_unclosed_brackets(json_str: str) -> list:
    """查找未闭合的括号

    参数:
        json_str: JSON 字符串

    返回:
        未闭合括号列表，按出现顺序排列
    """
    unclosed = []
    inside_string = False
    escape_next = False
    for char in json_str:
        if inside_string:
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == '"':
                inside_string = False
        else:
            if char == '"':
                inside_string = True
            elif char in "{[":
                unclosed.append(char)
            elif char in "}]":
                if unclosed:
                    last = unclosed[-1]
                    if (char == "}" and last == "{") or (
                        char == "]" and last == "["
                    ):
                        unclosed.pop()
    return unclosed


def fix_broken_json(json_str: str) -> str:
    """修复格式错误的 JSON 字符串

    尝试修复常见的 JSON 格式问题：
    - 移除末尾多余的逗号
    - 补全缺失的闭合括号

    参数:
        json_str: 可能格式错误的 JSON 字符串

    返回:
        修复后的 JSON 字符串
    """
    if not json_str:
        return json_str
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass
    last_comma = json_str.rfind(",")
    if last_comma != -1:
        json_str = json_str[:last_comma]
    unclosed = find_unclosed_brackets(json_str)
    closing_map = {"{": "}", "[": "]"}
    for open_char in reversed(unclosed):
        json_str += closing_map[open_char]
    return json_str


def safe_json_loads(
    json_str: str,
    default: Any = None,
    repair: bool = True,
) -> Optional[Union[dict, list]]:
    """安全的 JSON 解析

    尝试解析 JSON 字符串，失败时尝试修复后重新解析。

    参数:
        json_str: JSON 字符串
        default: 解析失败时的默认返回值
        repair: 是否尝试修复格式错误

    返回:
        解析后的对象，或默认值
    """
    if not json_str:
        return default
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        if not repair:
            return default
    try:
        fixed = fix_broken_json(json_str)
        return json.loads(fixed)
    except json.JSONDecodeError:
        return default


def extract_json_from_text(text: str) -> Optional[Union[dict, list]]:
    """从文本中提取 JSON

    查找文本中的 JSON 对象或数组并解析。

    参数:
        text: 可能包含 JSON 的文本

    返回:
        解析后的 JSON 对象，或 None
    """
    if not text:
        return None
    text = text.strip()
    start_idx = text.find("{")
    arr_start = text.find("[")
    if start_idx == -1 and arr_start == -1:
        return None
    if arr_start != -1 and (start_idx == -1 or arr_start < start_idx):
        start_idx = arr_start
        end_char = "]"
    else:
        end_char = "}"
    end_idx = text.rfind(end_char)
    if end_idx == -1 or end_idx < start_idx:
        return None
    json_str = text[start_idx: end_idx + 1]
    return safe_json_loads(json_str)


def extract_json_list(text: str) -> list:
    """从文本中提取 JSON 列表

    参数:
        text: 可能包含 JSON 列表的文本

    返回:
        解析后的列表，失败返回空列表
    """
    result = extract_json_from_text(text)
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ["items", "data", "list", "result", "entities", "triples"]:
            if key in result and isinstance(result[key], list):
                return result[key]
    return []


def extract_json_dict(text: str) -> dict:
    """从文本中提取 JSON 对象

    参数:
        text: 可能包含 JSON 对象的文本

    返回:
        解析后的字典，失败返回空字典
    """
    result = extract_json_from_text(text)
    if isinstance(result, dict):
        return result
    return {}


def ensure_json_serializable(obj: Any) -> Any:
    """确保对象可序列化为 JSON

    将不可序列化的对象转换为可序列化形式。

    参数:
        obj: 任意对象

    返回:
        可序列化的对象
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): ensure_json_serializable(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return ensure_json_serializable(obj.__dict__)
    return str(obj)

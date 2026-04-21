import json
import re
from typing import Dict, Any, Optional, Tuple, List, Union
from src.common.logger import get_logger

logger = get_logger("PFC工具")


def _repair_json_text(raw_text: str) -> str:
    fixed = raw_text.strip()
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    fixed = re.sub(r"//.*?\n", "\n", fixed)
    fixed = re.sub(r"/\*.*?\*/", "", fixed, flags=re.DOTALL)
    fixed = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', fixed)
    return fixed


def _scan_json_chunks(content: str, allow_array: bool = True) -> List[str]:
    chunks: List[str] = []
    if not content:
        return chunks

    fenced_pattern = r"```(?:json)?\s*(.*?)\s*```"
    fenced_matches = re.findall(
        fenced_pattern, content, re.DOTALL | re.IGNORECASE
    )
    for block in fenced_matches:
        if block.strip():
            chunks.append(block.strip())

    if allow_array:
        array_pattern = r"\[[\s\S]*?\]"
        for match in re.findall(array_pattern, content):
            if match.strip():
                chunks.append(match.strip())

    object_pattern = r"\{[\s\S]*?\}"
    for match in re.findall(object_pattern, content):
        if match.strip():
            chunks.append(match.strip())

    if content.strip() and content.strip() not in chunks:
        chunks.append(content.strip())

    return chunks


def _safe_load_json(
    raw_text: str,
) -> Optional[Union[Dict[str, Any], List[Any]]]:
    for candidate in (raw_text, _repair_json_text(raw_text)):
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, (dict, list)):
                return loaded
        except json.JSONDecodeError:
            continue
    return None


def _coerce_field_value(value: Any, expected_type: type) -> Any:
    if isinstance(value, expected_type):
        return value
    if expected_type is bool and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "是"}:
            return True
        if lowered in {"false", "0", "no", "否"}:
            return False
    if (
        expected_type is int
        and isinstance(value, str)
        and value.strip().isdigit()
    ):
        return int(value.strip())
    if expected_type is float and isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    return value


def _validate_record(
    raw_item: Dict[str, Any],
    required_fields: Tuple[str, ...],
    required_types: Optional[Dict[str, type]] = None,
) -> Optional[Dict[str, Any]]:
    if not all(field in raw_item for field in required_fields):
        return None

    normalized: Dict[str, Any] = {
        field: raw_item[field] for field in required_fields
    }
    for key, value in raw_item.items():
        if key not in normalized:
            normalized[key] = value

    if required_types:
        for field, expected_type in required_types.items():
            if field not in normalized:
                continue
            converted = _coerce_field_value(normalized[field], expected_type)
            if not isinstance(converted, expected_type):
                return None
            normalized[field] = converted

    for field in required_fields:
        if (
            isinstance(normalized[field], str)
            and not normalized[field].strip()
        ):
            return None

    return normalized


def parse_json_payload(
    content: str, allow_array: bool = True
) -> Optional[Union[Dict[str, Any], List[Any]]]:
    for chunk in _scan_json_chunks(content, allow_array=allow_array):
        parsed = _safe_load_json(chunk)
        if parsed is not None:
            return parsed
    return None


def get_items_from_json(
    content: str,
    private_name: str,
    *items: str,
    default_values: Optional[Dict[str, Any]] = None,
    required_types: Optional[Dict[str, type]] = None,
    allow_array: bool = True,
) -> Tuple[bool, Union[Dict[str, Any], List[Dict[str, Any]]]]:
    content = (content or "").strip()
    result = {}

    if default_values:
        result.update(default_values)

    json_data = parse_json_payload(content, allow_array=allow_array)
    if json_data is None:
        logger.error(f"[{private_name}]无法在返回内容中找到有效的JSON")
        return False, result

    if allow_array and isinstance(json_data, list):
        valid_items: List[Dict[str, Any]] = []
        for candidate in json_data:
            if not isinstance(candidate, dict):
                continue
            validated = _validate_record(candidate, items, required_types)
            if validated:
                valid_items.append(validated)
        if valid_items:
            return True, valid_items
        logger.error(f"[{private_name}]JSON数组中未找到包含必要字段的对象")
        return False, result

    if not isinstance(json_data, dict):
        logger.error(f"[{private_name}]JSON不是对象结构")
        return False, result

    validated_result = _validate_record(json_data, items, required_types)
    if validated_result is None:
        logger.error(
            f"[{private_name}]JSON缺少必要字段或字段类型不正确，实际内容: {json_data}"
        )
        return False, result

    for item in items:
        result[item] = validated_result[item]

    for key, value in validated_result.items():
        if key not in result:
            result[key] = value

    return True, result


def extract_json_objects(content: str) -> List[Dict[str, Any]]:
    json_objects: List[Dict[str, Any]] = []
    for chunk in _scan_json_chunks(content, allow_array=True):
        parsed = _safe_load_json(chunk)
        if isinstance(parsed, dict):
            json_objects.append(parsed)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    json_objects.append(item)
    return json_objects


def truncate_text(
    text: str, max_length: int = 500, suffix: str = "..."
) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_timestamp(
    timestamp: float, format_str: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    from datetime import datetime

    try:
        return datetime.fromtimestamp(timestamp).strftime(format_str)
    except Exception:
        return ""


def calculate_time_diff(start_time: float, end_time: float) -> str:
    diff = end_time - start_time
    if diff < 1:
        return f"{diff * 1000:.2f}毫秒"
    elif diff < 60:
        return f"{diff:.2f}秒"
    else:
        minutes = int(diff // 60)
        seconds = diff % 60
        return f"{minutes}分{seconds:.2f}秒"

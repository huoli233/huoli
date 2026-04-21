from enum import Enum
from typing import Optional, Any, Required
from typing_extensions import TypedDict


class ResponseFormatType(Enum):
    """响应格式类型"""

    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class JsonSchemaDef(TypedDict, total=False):
    """JSON Schema 定义"""

    name: Required[str]
    description: Optional[str]
    schema: dict[str, object]
    strict: Optional[bool]


def _validate_json_schema(schema: dict) -> Optional[str]:
    """验证 JSON Schema 格式"""
    if "name" not in schema:
        return "schema 必须包含 'name' 字段"
    if not isinstance(schema["name"], str) or not schema["name"].strip():
        return "schema 的 'name' 字段必须是非空字符串"
    if "description" in schema:
        if (
            not isinstance(schema["description"], str)
            or not schema["description"].strip()
        ):
            return "schema 的 'description' 字段必须是非空字符串"
    if "schema" not in schema:
        return "schema 必须包含 'schema' 字段"
    if not isinstance(schema["schema"], dict):
        return "schema 的 'schema' 字段必须是字典"
    if "strict" in schema and not isinstance(schema["strict"], bool):
        return "schema 的 'strict' 字段必须是布尔值"
    return None


def _strip_title_fields(schema: Any) -> Any:
    """递归移除 title 字段"""
    if isinstance(schema, list):
        return [_strip_title_fields(item) for item in schema]
    if isinstance(schema, dict):
        result = {}
        for key, value in schema.items():
            if key != "title":
                result[key] = _strip_title_fields(value)
        return result
    return schema


def _resolve_definitions(schema: dict) -> dict:
    """解析并内联 $defs 定义"""
    definitions = {}

    def extract_defs(obj: Any, path: str = "#"):
        if isinstance(obj, dict):
            if "$defs" in obj:
                for def_name, def_value in obj["$defs"].items():
                    definitions[f"{path}/$defs/{def_name}"] = def_value
                del obj["$defs"]
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    extract_defs(value, f"{path}/{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    extract_defs(item, f"{path}/{i}")

    def resolve_refs(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref = obj["$ref"]
                if ref in definitions:
                    resolved = definitions[ref].copy()
                    for key, value in obj.items():
                        if key != "$ref":
                            resolved[key] = value
                    return resolve_refs(resolved)
            return {k: resolve_refs(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve_refs(item) for item in obj]
        return obj

    schema_copy = schema.copy()
    extract_defs(schema_copy)
    return resolve_refs(schema_copy)


class ResponseFormat:
    """响应格式配置"""

    def __init__(
        self,
        format_type: ResponseFormatType = ResponseFormatType.TEXT,
        schema: type | JsonSchemaDef | None = None,
    ):
        self.format_type = format_type
        self.schema: Optional[dict] = None
        if format_type == ResponseFormatType.JSON_SCHEMA:
            if schema is None:
                raise ValueError(
                    "format_type 为 JSON_SCHEMA 时，schema 不能为空"
                )
            if isinstance(schema, dict):
                error = _validate_json_schema(schema)
                if error:
                    raise ValueError(error)
                self.schema = schema
            elif hasattr(schema, "model_json_schema"):
                self.schema = self._build_schema_from_model(schema)
            else:
                raise ValueError("schema 必须是字典或 Pydantic 模型类")

    def _build_schema_from_model(self, model_class: type) -> dict:
        """从 Pydantic 模型构建 JSON Schema"""
        try:
            raw_schema = model_class.model_json_schema()
            processed = _strip_title_fields(raw_schema)
            resolved = _resolve_definitions(processed)
            result = {
                "name": model_class.__name__,
                "schema": resolved,
                "strict": False,
            }
            if model_class.__doc__:
                result["description"] = model_class.__doc__
            return result
        except Exception as e:
            raise ValueError(f"生成 JSON Schema 失败: {e}") from e

    def to_dict(self) -> dict:
        """转换为字典格式"""
        if self.schema:
            return {
                "type": self.format_type.value,
                "json_schema": self.schema,
            }
        return {"type": self.format_type.value}

    def to_openai_format(self) -> dict:
        """转换为 OpenAI API 格式"""
        if self.format_type == ResponseFormatType.TEXT:
            return {"type": "text"}
        if self.format_type == ResponseFormatType.JSON_OBJECT:
            return {"type": "json_object"}
        if self.format_type == ResponseFormatType.JSON_SCHEMA:
            return {
                "type": "json_schema",
                "json_schema": self.schema,
            }
        return {"type": "text"}


def text_format() -> ResponseFormat:
    """创建文本格式"""
    return ResponseFormat(format_type=ResponseFormatType.TEXT)


def json_object_format() -> ResponseFormat:
    """创建 JSON 对象格式"""
    return ResponseFormat(format_type=ResponseFormatType.JSON_OBJECT)


def json_schema_format(schema: type | JsonSchemaDef) -> ResponseFormat:
    """创建 JSON Schema 格式"""
    return ResponseFormat(
        format_type=ResponseFormatType.JSON_SCHEMA, schema=schema
    )


RespFormat = ResponseFormat

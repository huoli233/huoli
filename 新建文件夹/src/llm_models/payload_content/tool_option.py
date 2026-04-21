from enum import Enum
from typing import Optional, Dict, List, Any


class ToolParamType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "bool"
    ARRAY = "array"
    OBJECT = "object"


class ToolParam:
    def __init__(
        self,
        name: str,
        param_type: ToolParamType,
        description: str,
        required: bool = False,
        enum_values: Optional[List[str]] = None,
        default: Optional[Any] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        self.name: str = name
        self.param_type: ToolParamType = param_type
        self.description: str = description
        self.required: bool = required
        self.enum_values: Optional[List[str]] = enum_values
        self.default: Optional[Any] = default
        self.min_value: Optional[float] = min_value
        self.max_value: Optional[float] = max_value

    def to_json_schema(self) -> Dict[str, Any]:
        schema: Dict[str, Any] = {
            "type": self.param_type.value,
            "description": self.description,
        }
        if self.enum_values:
            schema["enum"] = self.enum_values
        if self.default is not None:
            schema["default"] = self.default
        if self.min_value is not None and self.param_type in (
            ToolParamType.INTEGER,
            ToolParamType.FLOAT,
        ):
            schema["minimum"] = self.min_value
        if self.max_value is not None and self.param_type in (
            ToolParamType.INTEGER,
            ToolParamType.FLOAT,
        ):
            schema["maximum"] = self.max_value
        return schema


class ToolOption:
    def __init__(
        self,
        name: str,
        description: str,
        params: Optional[List[ToolParam]] = None,
    ):
        self.name: str = name
        self.description: str = description
        self.params: Optional[List[ToolParam]] = params

    def to_openai_tool(self) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        required: List[str] = []
        if self.params:
            for param in self.params:
                properties[param.name] = param.to_json_schema()
                if param.required:
                    required.append(param.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolOptionBuilder:
    def __init__(self):
        self._name: str = ""
        self._description: str = ""
        self._params: List[ToolParam] = []

    def set_name(self, name: str) -> "ToolOptionBuilder":
        if not name:
            raise ValueError("工具名称不能为空")
        self._name = name
        return self

    def set_description(self, description: str) -> "ToolOptionBuilder":
        if not description:
            raise ValueError("工具描述不能为空")
        self._description = description
        return self

    def add_param(
        self,
        name: str,
        param_type: ToolParamType,
        description: str,
        required: bool = False,
        enum_values: Optional[List[str]] = None,
        default: Optional[Any] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> "ToolOptionBuilder":
        if not name or not description:
            raise ValueError("参数名称/描述不能为空")
        self._params.append(
            ToolParam(
                name=name,
                param_type=param_type,
                description=description,
                required=required,
                enum_values=enum_values,
                default=default,
                min_value=min_value,
                max_value=max_value,
            )
        )
        return self

    def add_string_param(
        self,
        name: str,
        description: str,
        required: bool = False,
        enum_values: Optional[List[str]] = None,
        default: Optional[str] = None,
    ) -> "ToolOptionBuilder":
        return self.add_param(
            name=name,
            param_type=ToolParamType.STRING,
            description=description,
            required=required,
            enum_values=enum_values,
            default=default,
        )

    def add_integer_param(
        self,
        name: str,
        description: str,
        required: bool = False,
        default: Optional[int] = None,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> "ToolOptionBuilder":
        return self.add_param(
            name=name,
            param_type=ToolParamType.INTEGER,
            description=description,
            required=required,
            default=default,
            min_value=float(min_value) if min_value is not None else None,
            max_value=float(max_value) if max_value is not None else None,
        )

    def add_float_param(
        self,
        name: str,
        description: str,
        required: bool = False,
        default: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> "ToolOptionBuilder":
        return self.add_param(
            name=name,
            param_type=ToolParamType.FLOAT,
            description=description,
            required=required,
            default=default,
            min_value=min_value,
            max_value=max_value,
        )

    def add_boolean_param(
        self,
        name: str,
        description: str,
        required: bool = False,
        default: Optional[bool] = None,
    ) -> "ToolOptionBuilder":
        return self.add_param(
            name=name,
            param_type=ToolParamType.BOOLEAN,
            description=description,
            required=required,
            default=default,
        )

    def build(self) -> ToolOption:
        if self._name == "" or self._description == "":
            raise ValueError("工具名称/描述不能为空")
        return ToolOption(
            name=self._name,
            description=self._description,
            params=self._params if self._params else None,
        )


class ToolCall:
    def __init__(
        self,
        call_id: str,
        func_name: str,
        args: Optional[Dict[str, Any]] = None,
    ):
        self.call_id: str = call_id
        self.func_name: str = func_name
        self.args: Optional[Dict[str, Any]] = args

    @classmethod
    def from_openai_response(
        cls, tool_call_data: Dict[str, Any]
    ) -> "ToolCall":
        call_id = tool_call_data.get("id", "")
        function_data = tool_call_data.get("function", {})
        func_name = function_data.get("name", "")
        import json

        args_str = function_data.get("arguments", "{}")
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = {}
        return cls(call_id=call_id, func_name=func_name, args=args)

    def to_openai_tool_call(self) -> Dict[str, Any]:
        import json

        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.func_name,
                "arguments": json.dumps(self.args or {}, ensure_ascii=False),
            },
        }


class ToolCallResult:
    def __init__(
        self,
        call_id: str,
        name: str,
        content: str,
        is_error: bool = False,
    ):
        self.call_id: str = call_id
        self.name: str = name
        self.content: str = content
        self.is_error: bool = is_error

    def to_openai_tool_message(self) -> Dict[str, Any]:
        return {
            "tool_call_id": self.call_id,
            "role": "tool",
            "name": self.name,
            "content": self.content,
        }

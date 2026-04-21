import re
import asyncio
import contextvars
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Union
from src.common.logger import get_logger

logger = get_logger("提示词构建")


class PromptScope:
    """提示词作用域管理器，支持协程级别的上下文隔离"""

    def __init__(self):
        self._scoped_prompts: Dict[str, Dict[str, "PromptTemplate"]] = {}
        self._active_scope: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar("prompt_scope", default=None)
        )
        self._scope_lock = asyncio.Lock()

    @property
    def current_scope_id(self) -> Optional[str]:
        return self._active_scope.get()

    @asynccontextmanager
    async def enter_scope(self, scope_id: Optional[str] = None):
        """进入指定作用域"""
        previous_scope = self._active_scope.get()
        token = None
        if scope_id is not None:
            try:
                async with asyncio.timeout(5.0):
                    async with self._scope_lock:
                        if scope_id not in self._scoped_prompts:
                            self._scoped_prompts[scope_id] = {}
            except asyncio.TimeoutError:
                logger.warning(f"获取作用域锁超时: {scope_id}")
                scope_id = None
            if scope_id:
                token = self._active_scope.set(scope_id)
        try:
            yield self
        finally:
            if token is not None:
                try:
                    self._active_scope.reset(token)
                except Exception as e:
                    logger.warning(f"重置作用域失败: {e}")
                    self._active_scope.set(previous_scope)

    async def fetch_prompt(self, name: str) -> Optional["PromptTemplate"]:
        """从当前作用域获取提示词"""
        scope_id = self._active_scope.get()
        if scope_id and scope_id in self._scoped_prompts:
            return self._scoped_prompts[scope_id].get(name)
        return None

    async def store_prompt(
        self, prompt: "PromptTemplate", scope_id: Optional[str] = None
    ):
        """存储提示词到作用域"""
        target_scope = scope_id or self._active_scope.get()
        if target_scope:
            if target_scope not in self._scoped_prompts:
                self._scoped_prompts[target_scope] = {}
            self._scoped_prompts[target_scope][prompt.prompt_name] = prompt


class PromptRegistry:
    """提示词注册表，管理全局和作用域提示词"""

    def __init__(self):
        self._global_prompts: Dict[str, "PromptTemplate"] = {}
        self._scope_manager = PromptScope()
        self._name_counter = 0
        self._registry_lock = asyncio.Lock()

    @asynccontextmanager
    async def message_context(self, message_id: Optional[str] = None):
        """创建消息处理上下文"""
        async with self._scope_manager.enter_scope(message_id):
            yield self

    async def retrieve_prompt(self, name: str) -> "PromptTemplate":
        """获取提示词，优先从作用域获取"""
        scoped = await self._scope_manager.fetch_prompt(name)
        if scoped is not None:
            return scoped
        async with self._registry_lock:
            if name not in self._global_prompts:
                raise KeyError(f"提示词 '{name}' 未找到")
            return self._global_prompts[name]

    def _generate_name(self, template: str) -> str:
        """自动生成提示词名称"""
        self._name_counter += 1
        return f"auto_prompt_{self._name_counter}"

    def register(self, prompt: "PromptTemplate"):
        """注册全局提示词"""
        if not prompt.prompt_name:
            prompt.prompt_name = self._generate_name(prompt.raw_template)
        self._global_prompts[prompt.prompt_name] = prompt

    def create_prompt(self, name: str, template: str) -> "PromptTemplate":
        """创建并注册提示词"""
        prompt = PromptTemplate(template, name=name)
        self._global_prompts[name] = prompt
        return prompt

    async def render(self, name: str, **kwargs) -> str:
        """渲染提示词"""
        prompt = await self.retrieve_prompt(name)
        return prompt.render(**kwargs)

    def get_prompt(self, name: str) -> "PromptTemplate | None":
        """同步获取已注册的提示词模板，未找到返回 None"""
        return self._global_prompts.get(name)

    async def get_prompt_async(self, name: str) -> "PromptTemplate":
        """异步获取提示词模板"""
        return await self.retrieve_prompt(name)

    async def format_prompt(self, name: str, **kwargs) -> str:
        """格式化提示词"""
        return await self.render(name, **kwargs)

    @asynccontextmanager
    async def async_message_scope(self, scope_id: Optional[str] = None):
        """创建异步消息作用域上下文"""
        async with self._scope_manager.enter_scope(scope_id):
            yield self


global_prompt_registry = PromptRegistry()


class PromptTemplate(str):
    """提示词模板类，支持变量插值和嵌套"""

    _PLACEHOLDER_LEFT = "\x00LBRACE\x00"
    _PLACEHOLDER_RIGHT = "\x00RBRACE\x00"

    @staticmethod
    def _escape_braces(template) -> str:
        """转义模板中的花括号"""
        if isinstance(template, list):
            template = "\n".join(str(item) for item in template)
        elif not isinstance(template, str):
            template = str(template)
        return template.replace(
            "\\{", PromptTemplate._PLACEHOLDER_LEFT
        ).replace("\\}", PromptTemplate._PLACEHOLDER_RIGHT)

    @staticmethod
    def _unescape_braces(template: str) -> str:
        """还原转义的花括号"""
        return template.replace(PromptTemplate._PLACEHOLDER_LEFT, "{").replace(
            PromptTemplate._PLACEHOLDER_RIGHT, "}"
        )

    def __new__(
        cls,
        template_str,
        name: Optional[str] = None,
        positional_args: Union[List[Any], tuple] = None,
        **kwargs,
    ):
        if isinstance(positional_args, tuple):
            positional_args = list(positional_args)
        auto_register = kwargs.pop("_auto_register", True)
        escaped_template = cls._escape_braces(template_str)
        variable_names = []
        found_vars = re.findall(r"\{(.*?)}", escaped_template)
        for var in found_vars:
            if var and var not in variable_names:
                variable_names.append(var)
        if kwargs or positional_args:
            rendered = cls._render_template(
                template_str, args=positional_args, kwargs=kwargs
            )
            instance = super().__new__(cls, rendered)
        else:
            instance = super().__new__(cls, "")
        instance.raw_template = template_str
        instance.prompt_name = name
        instance.variable_names = variable_names
        instance._positional_args = positional_args or []
        instance._keyword_args = kwargs
        if (
            auto_register
            and not global_prompt_registry._scope_manager.current_scope_id
        ):
            global_prompt_registry.register(instance)
        return instance

    @classmethod
    async def create_in_scope(
        cls,
        template_str,
        name: Optional[str] = None,
        args: Union[List[Any], tuple] = None,
        **kwargs,
    ):
        """在当前作用域内创建提示词"""
        prompt = cls(template_str, name, args, **kwargs)
        if global_prompt_registry._scope_manager.current_scope_id:
            await global_prompt_registry._scope_manager.store_prompt(prompt)
        return prompt

    @classmethod
    def _render_template(
        cls, template, args: List[Any] = None, kwargs: Dict[str, Any] = None
    ) -> str:
        escaped = cls._escape_braces(template)
        variable_names = []
        found = re.findall(r"\{(.*?)}", escaped)
        for var in found:
            if var and var not in variable_names:
                variable_names.append(var)
        resolved_args = {}
        resolved_kwargs = {}
        if args:
            for i, arg in enumerate(args):
                if i < len(variable_names):
                    if isinstance(arg, PromptTemplate):
                        resolved_args[variable_names[i]] = arg.render(**kwargs)
                    else:
                        resolved_args[variable_names[i]] = arg
                else:
                    logger.error(
                        f"模板参数数量不匹配: 需要 {
                            len(variable_names)} 个，提供了 {
                            len(args)} 个"
                    )
                    raise ValueError("模板参数数量不匹配")
        if kwargs:
            for key, value in kwargs.items():
                if isinstance(value, PromptTemplate):
                    remaining = {k: v for k, v in kwargs.items() if k != key}
                    resolved_kwargs[key] = value.render(**remaining)
                else:
                    resolved_kwargs[key] = value
        try:
            if args:
                escaped = escaped.format(**resolved_args)
            if kwargs:
                escaped = escaped.format(**resolved_kwargs)
            return cls._unescape_braces(escaped)
        except (IndexError, KeyError) as e:
            raise ValueError(f"模板渲染失败: {template}") from e

    def render(self, *args, **kwargs) -> str:
        """渲染模板"""
        result = type(self)(
            self.raw_template,
            self.prompt_name,
            args=list(args) if args else self._positional_args,
            _auto_register=False,
            **kwargs or self._keyword_args,
        )
        return str(result)

    def __str__(self) -> str:
        if self._keyword_args or self._positional_args:
            return super().__str__()
        return self.raw_template

    def __repr__(self) -> str:
        return f"PromptTemplate(name='{
            self.prompt_name}', template='{
            self.raw_template[
                :50]}...')"


Prompt = PromptTemplate
global_prompt_manager = global_prompt_registry

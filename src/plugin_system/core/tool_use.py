import asyncio
import hashlib
import inspect
import time
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger  # noqa: E402

try:
    from src.config.config import global_config, model_config  # noqa: E402
except ImportError as e:
    raise ImportError(f"无法导入 config 模块，请检查模块路径是否正确: {e}") from e  # noqa: E402

try:
    from src.llm_models.utils_model import LLMRequest  # noqa: E402
except ImportError as e:
    raise ImportError(f"无法导入 utils_model 模块，请检查模块路径是否正确: {e}") from e  # noqa: E402

try:
    from src.plugin_system.apis.tool_api import (
        get_llm_available_tool_definitions,
        get_tool_instance,
    )  # noqa: E402
except ImportError as e:
    raise ImportError(f"无法导入 tool_api 模块，请检查模块路径是否正确: {e}") from e  # noqa: E402

try:
    from src.plugin_system.base.base_tool import BaseTool  # noqa: E402
except ImportError as e:
    raise ImportError(f"无法导入 base_tool 模块，请检查模块路径是否正确: {e}") from e  # noqa: E402

try:
    import src.chat.prompts.catalog  # noqa: F401, E401 注册提示词
except ImportError as e:
    raise ImportError(f"无法导入 prompts.catalog 模块，请检查模块路径是否正确: {e}") from e  # noqa: E402

try:
    from src.chat.utils.prompt_builder import (
        global_prompt_manager,
    )  # noqa: E402
except ImportError as e:
    raise ImportError(f"无法导入 prompt_builder 模块，请检查模块路径是否正确: {e}") from e  # noqa: E402

logger = get_logger("工具使用")

_MEMORY_TOOL_EXCLUDED_FROM_REPLY = {"return_information"}
_TOOL_ARG_ALIASES = {
    "record_ids": ("memory_ids", "ids", "id"),
    "query": ("keyword", "question", "content"),
    "words": ("word", "query", "keyword"),
    "person_name": ("name", "nickname", "user_name"),
}


class ToolExecutor:
    """独立的工具执行器组件"""

    def __init__(self, chat_id: str, enable_cache: bool = True, cache_ttl: int = 3):
        self.chat_id = chat_id
        self.log_prefix = f"[ToolExecutor:{chat_id[:8]}]"
        self.llm_model = LLMRequest(
            model_set=model_config.model_task_config.tool_use,
            request_type="tool_executor",
        )
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.tool_cache: Dict[str, Dict[str, Any]] = {}
        logger.info(f"{self.log_prefix} 工具执行器初始化完成")

    async def execute_from_chat_message(
        self,
        target_message: str,
        chat_history: str,
        sender: str,
        return_details: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str], str]:
        """从聊天消息执行工具"""
        cache_key = self._generate_cache_key(target_message, chat_history, sender)
        if cached_result := self._get_from_cache(cache_key):
            logger.info(f"{self.log_prefix} 使用缓存结果")
            if not return_details:
                return cached_result, [], ""
            used_tools = [result.get("tool_name", "unknown") for result in cached_result]
            return cached_result, used_tools, ""
        tools = self._get_tool_definitions()
        if not tools:
            logger.warning(f"{self.log_prefix} 没有可用工具，跳过工具调用分析")
            if return_details:
                return [], [], ""
            return [], [], ""
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        bot_name = global_config.bot.nickname
        prompt = global_prompt_manager.get_prompt("tool_executor_system").render(
            target_message=target_message,
            chat_history=chat_history,
            sender=sender,
            bot_name=bot_name,
            time_now=time_now,
        )
        logger.debug(f"{self.log_prefix} 开始LLM工具调用分析")
        try:
            _response, (_reasoning_content, _model_name, tool_calls) = await asyncio.wait_for(
                self.llm_model.generate_response_async(
                    prompt=prompt,
                    tools=tools,
                    raise_when_empty=False,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"{self.log_prefix} LLM工具调用超时(30s)")
            tool_calls = []
        tool_results, used_tools = await self.execute_tool_calls(tool_calls)
        if tool_results:
            self._set_cache(cache_key, tool_results)
        if used_tools:
            logger.info(f"{self.log_prefix} 工具执行完成，共执行{len(used_tools)}个工具: {used_tools}")
        if return_details:
            return tool_results, used_tools, prompt
        return tool_results, [], ""

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """获取可用工具定义"""
        plugin_tools = self._get_plugin_tool_definitions()
        memory_tools = self._get_memory_tool_definitions()
        all_tools = [*plugin_tools, *memory_tools]
        if not all_tools:
            self._log_empty_tool_registry(plugin_count=0, memory_count=0, disabled_count=0)
            return []
        try:
            from src.plugin_system.core.global_announcement_manager import (
                global_announcement_manager,
            )

            disabled_names = [
                name for name, _definition in all_tools if global_announcement_manager.is_tool_disabled(self.chat_id, name)
            ]
            visible_tools = [
                (name, definition)
                for name, definition in all_tools
                if name not in disabled_names
            ]
            if not visible_tools:
                self._log_empty_tool_registry(
                    plugin_count=len(plugin_tools),
                    memory_count=len(memory_tools),
                    disabled_count=len(disabled_names),
                )
            return self._dedupe_tool_definitions(visible_tools)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 工具禁用状态读取失败，使用全量工具: {exc}")
            return self._dedupe_tool_definitions(all_tools)

    def _get_plugin_tool_definitions(self) -> List[Tuple[str, Dict[str, Any]]]:
        """获取插件系统注册的工具定义。"""
        try:
            return get_llm_available_tool_definitions()
        except Exception as exc:
            logger.warning(f"{self.log_prefix} 插件工具注册表读取失败: {exc}")
            return []

    def _get_memory_tool_definitions(self) -> List[Tuple[str, Dict[str, Any]]]:
        """将记忆检索工具桥接到回复器工具执行器。"""
        try:
            from src.memory_system.retrieval_tools.tool_loader import init_all_tools
            from src.memory_system.retrieval_tools.tool_registry import get_tool_registry

            init_all_tools()
            registry = get_tool_registry()
            tools = registry.get_all_tools()
            definitions: List[Tuple[str, Dict[str, Any]]] = []
            for name, tool in tools.items():
                if name in _MEMORY_TOOL_EXCLUDED_FROM_REPLY:
                    continue
                definitions.append((name, tool.get_tool_definition()))
            return definitions
        except Exception as exc:
            logger.warning(f"{self.log_prefix} 记忆检索工具注册表读取失败: {exc}")
            return []

    def _dedupe_tool_definitions(self, tools: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """按工具名去重，插件工具优先，避免重复 function name 让模型混乱。"""
        deduped: Dict[str, Dict[str, Any]] = {}
        duplicate_names: List[str] = []
        for name, definition in tools:
            if name in deduped:
                duplicate_names.append(name)
                continue
            deduped[name] = definition
        if duplicate_names:
            logger.warning(f"{self.log_prefix} 检测到重复工具名，已保留首个定义: {sorted(set(duplicate_names))}")
        return list(deduped.values())

    def _log_empty_tool_registry(self, plugin_count: int, memory_count: int, disabled_count: int) -> None:
        """输出工具为空时的明确诊断，区分没注册、被禁用和桥接失败。"""
        try:
            from src.plugin_system.core.component_registry import component_registry
            from src.plugin_system.core.plugin_manager import plugin_manager

            stats = component_registry.get_registry_stats()
            logger.warning(
                f"{self.log_prefix} 工具注册诊断: 可用插件工具={plugin_count}, "
                f"可用记忆工具={memory_count}, 当前聊天禁用={disabled_count}, "
                f"已加载插件={len(plugin_manager.loaded_plugins)}, "
                f"注册组件={stats.get('total_components', 0)}, "
                f"Tool组件={stats.get('tool_components', 0)}"
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 工具注册诊断采集失败: {exc}")

    async def execute_tool_calls(
        self,
        tool_calls: Optional[List[Any]],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """执行工具调用"""
        tool_results: List[Dict[str, Any]] = []
        used_tools = []
        if not tool_calls:
            logger.debug(f"{self.log_prefix} 无需执行工具")
            return [], []
        func_names = [call.func_name for call in tool_calls if getattr(call, "func_name", None)]
        logger.info(f"{self.log_prefix} 开始执行工具调用: {func_names}")
        for tool_call in tool_calls:
            tool_name = getattr(tool_call, "func_name", "") or "unknown"
            try:
                logger.debug(f"{self.log_prefix} 执行工具: {tool_name}")
                result = await self.execute_tool_call(tool_call)
                if result:
                    tool_info = {
                        "type": result.get("type", "unknown_type"),
                        "id": result.get("id", f"tool_exec_{time.time()}"),
                        "content": result.get("content", ""),
                        "tool_name": tool_name,
                        "timestamp": time.time(),
                    }
                    content = tool_info["content"]
                    if not isinstance(content, (str, list, tuple)):
                        tool_info["content"] = str(content)
                    content_check = tool_info["content"]
                    if (isinstance(content_check, str) and not content_check.strip()) or (
                        isinstance(content_check, (list, tuple)) and len(content_check) == 0
                    ):
                        logger.debug(f"{self.log_prefix} 工具{tool_name}无有效内容，跳过")
                        continue
                    tool_results.append(tool_info)
                    used_tools.append(tool_name)
            except Exception as e:
                logger.error(f"{self.log_prefix} 工具{tool_name}执行失败: {e}")
                error_info = {
                    "type": "tool_error",
                    "id": f"tool_error_{time.time()}",
                    "content": f"工具{tool_name}执行失败: {str(e)}",
                    "tool_name": tool_name,
                    "timestamp": time.time(),
                }
                tool_results.append(error_info)
        return tool_results, used_tools

    async def execute_tool_call(
        self,
        tool_call: Any,
        tool_instance: Optional[BaseTool] = None,
    ) -> Optional[Dict[str, Any]]:
        """执行单个工具调用"""
        try:
            function_name = getattr(tool_call, "func_name", "") or ""
            if not function_name:
                logger.warning(f"{self.log_prefix} 工具调用缺少函数名，已跳过")
                return None
            function_args = dict(getattr(tool_call, "args", None) or {})
            try:
                from src.plugin_system.core.global_announcement_manager import (
                    global_announcement_manager,
                )

                if global_announcement_manager.is_tool_disabled(self.chat_id, function_name):
                    logger.info(f"{self.log_prefix} 工具已禁用，跳过执行: {function_name}")
                    return None
            except Exception:
                pass
            tool_instance = tool_instance or get_tool_instance(function_name, None)
            if tool_instance:
                function_args["llm_called"] = True
                result = await tool_instance.execute(function_args)
                if result:
                    return {
                        "tool_call_id": tool_call.call_id,
                        "role": "tool",
                        "name": function_name,
                        "type": "function",
                        "content": result.get("content", ""),
                    }
                return None

            memory_result = await self._execute_memory_tool(function_name, function_args)
            if memory_result is not None:
                return {
                    "tool_call_id": tool_call.call_id,
                    "role": "tool",
                    "name": function_name,
                    "type": "function",
                    "content": memory_result,
                }

            logger.warning(f"{self.log_prefix} 未知工具名称: {function_name}")
            return None
        except Exception as e:
            logger.error(f"执行工具调用时发生错误: {str(e)}")
            raise e

    async def _execute_memory_tool(self, function_name: str, function_args: Dict[str, Any]) -> Optional[str]:
        """执行记忆检索工具，并按函数签名注入当前聊天流上下文。"""
        try:
            from src.memory_system.retrieval_tools.tool_loader import init_all_tools
            from src.memory_system.retrieval_tools.tool_registry import get_tool_registry

            init_all_tools()
            memory_tool = get_tool_registry().get_tool(function_name)
            if memory_tool is None or function_name in _MEMORY_TOOL_EXCLUDED_FROM_REPLY:
                return None
            prepared_args = self._prepare_memory_tool_args(memory_tool, function_args)
            result = await memory_tool.execute(**prepared_args)
            if result:
                return str(result)
            return None
        except Exception as exc:
            logger.error(f"{self.log_prefix} 记忆工具 {function_name} 执行失败: {exc}")
            raise

    def _prepare_memory_tool_args(self, memory_tool: Any, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        """按原始执行函数签名整理参数，避免把系统注入字段传给不接收的函数。"""
        params = dict(raw_args)
        signature = inspect.signature(memory_tool.execute_func)
        parameters = signature.parameters

        for target_name, aliases in _TOOL_ARG_ALIASES.items():
            if target_name in parameters and target_name not in params:
                for alias in aliases:
                    if alias in params:
                        params[target_name] = params[alias]
                        break

        if "chat_id" in parameters and not params.get("chat_id"):
            params["chat_id"] = self.chat_id
        if "stream_id" in parameters and not params.get("stream_id"):
            params["stream_id"] = self.chat_id

        accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
        if accepts_var_kwargs:
            return params
        return {key: value for key, value in params.items() if key in parameters}

    def _generate_cache_key(self, target_message: str, chat_history: str, sender: str) -> str:
        """生成缓存键"""
        content = f"{target_message}_{chat_history}_{sender}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """从缓存获取结果"""
        if not self.enable_cache or cache_key not in self.tool_cache:
            return None
        cache_item = self.tool_cache[cache_key]
        if cache_item["ttl"] <= 0:
            del self.tool_cache[cache_key]
            logger.debug(f"{self.log_prefix} 缓存过期")
            return None
        cache_item["ttl"] -= 1
        logger.debug(f"{self.log_prefix} 使用缓存结果，剩余TTL: {cache_item['ttl']}")
        return cache_item["result"]

    def _set_cache(self, cache_key: str, result: List[Dict]):
        """设置缓存"""
        if not self.enable_cache:
            return
        if len(self.tool_cache) >= 500:
            self._cleanup_expired_cache()
        if len(self.tool_cache) >= 500:
            oldest_key = min(self.tool_cache, key=lambda k: self.tool_cache[k]["timestamp"])
            del self.tool_cache[oldest_key]
        self.tool_cache[cache_key] = {
            "result": result,
            "ttl": self.cache_ttl,
            "timestamp": time.time(),
        }
        logger.debug(f"{self.log_prefix} 设置缓存，TTL: {self.cache_ttl}")

    def _cleanup_expired_cache(self):
        """清理过期的缓存"""
        if not self.enable_cache:
            return
        expired_keys = [cache_key for cache_key, cache_item in self.tool_cache.items() if cache_item["ttl"] <= 0]
        for key in expired_keys:
            del self.tool_cache[key]
        if expired_keys:
            logger.debug(f"{self.log_prefix} 清理了{len(expired_keys)}个过期缓存")

    async def execute_specific_tool_simple(
        self,
        tool_name: str,
        tool_args: Dict,
    ) -> Optional[Dict]:
        """直接执行指定工具"""
        try:
            from src.llm_models.payload_content.message import ToolCall

            tool_call = ToolCall(
                call_id=f"direct_tool_{time.time()}",
                func_name=tool_name,
                args=tool_args,
            )
            logger.info(f"{self.log_prefix} 直接执行工具: {tool_name}")
            result = await self.execute_tool_call(tool_call)
            if result:
                tool_info = {
                    "type": result.get("type", "unknown_type"),
                    "id": f"direct_tool_{time.time()}",
                    "content": result.get("content", ""),
                    "tool_name": tool_name,
                    "timestamp": time.time(),
                }
                logger.info(f"{self.log_prefix} 直接工具执行成功: {tool_name}")
                return tool_info
        except Exception as e:
            logger.error(f"{self.log_prefix} 直接工具执行失败 {tool_name}: {e}")
        return None

    def clear_cache(self):
        """清空所有缓存"""
        if self.enable_cache:
            cache_count = len(self.tool_cache)
            self.tool_cache.clear()
            logger.info(f"{self.log_prefix} 清空了{cache_count}个缓存项")

    def get_cache_status(self) -> Dict:
        """获取缓存状态信息"""
        if not self.enable_cache:
            return {"enabled": False, "cache_count": 0}
        self._cleanup_expired_cache()
        total_count = len(self.tool_cache)
        ttl_distribution = {}
        for cache_item in self.tool_cache.values():
            ttl = cache_item["ttl"]
            ttl_distribution[ttl] = ttl_distribution.get(ttl, 0) + 1
        return {
            "enabled": True,
            "cache_count": total_count,
            "cache_ttl": self.cache_ttl,
            "ttl_distribution": ttl_distribution,
        }

    def set_cache_config(self, enable_cache: Optional[bool] = None, cache_ttl: int = -1):
        """动态修改缓存配置"""
        if enable_cache is not None:
            self.enable_cache = enable_cache
            logger.info(f"{self.log_prefix} 缓存状态修改为: {'启用' if enable_cache else '禁用'}")
        if cache_ttl > 0:
            self.cache_ttl = cache_ttl
            logger.info(f"{self.log_prefix} 缓存TTL修改为: {cache_ttl}")

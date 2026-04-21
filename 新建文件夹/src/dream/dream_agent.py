import asyncio
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from src.common.logger import get_logger
from src.dream.config_loader import (
    get_max_iterations,
    get_cooldown_seconds,
    get_tool_definitions,
    get_goals,
    get_merge_rules,
    get_prompt_template,
    get_instruction_template,
)

logger = get_logger("梦境代理")


class ToolParamType(Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    name: str
    param_type: ToolParamType
    description: str
    required: bool = False
    default: Any = None


@dataclass
class DreamTool:
    name: str
    description: str
    parameters: List[ToolParameter]
    execute_func: Callable[..., Awaitable[str]]

    def get_tool_definition(self) -> Dict[str, Any]:
        properties = {}
        required = []
        for param in self.parameters:
            properties[param.name] = {
                "type": param.param_type.value,
                "description": param.description,
            }
            if param.required:
                required.append(param.name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    async def execute(self, **kwargs) -> str:
        return await self.execute_func(**kwargs)


class DreamToolRegistry:
    def __init__(self):
        self._tools: Dict[str, DreamTool] = {}

    def register_tool(self, tool: DreamTool) -> None:
        self._tools[tool.name] = tool
        logger.debug(f"注册 dream 工具: {tool.name}")

    def unregister_tool(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get_tool(self, name: str) -> Optional[DreamTool]:
        return self._tools.get(name)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [tool.get_tool_definition() for tool in self._tools.values()]

    def clear_tools(self) -> None:
        self._tools.clear()


_dream_tool_registry = DreamToolRegistry()


def get_dream_tool_registry() -> DreamToolRegistry:
    return _dream_tool_registry


@dataclass
class MaintenanceContext:
    chat_id: str
    start_memory_id: Optional[int] = None
    current_iteration: int = 0
    max_iterations: int = 10
    tools_used: List[str] = field(default_factory=list)
    records_modified: int = 0
    start_time: float = field(default_factory=time.time)
    is_finished: bool = False
    finish_reason: str = ""


class DreamAgent:
    def __init__(self, llm_bridge=None, db_adapter=None):
        self._llm_bridge = llm_bridge
        self._db_adapter = db_adapter
        self._tool_registry = DreamToolRegistry()
        self._last_run_time: Dict[str, float] = {}
        self._running_contexts: Dict[str, MaintenanceContext] = {}

    def set_llm_bridge(self, bridge):
        self._llm_bridge = bridge

    def set_db_adapter(self, adapter):
        self._db_adapter = adapter

    def _init_tools_for_chat(self, chat_id: str) -> None:
        self._tool_registry.clear_tools()
        tool_defs = get_tool_definitions()
        for tool_name, tool_config in tool_defs.items():
            params = []
            for param_name, param_config in tool_config.get(
                "params", {}
            ).items():
                param_type = ToolParamType(param_config.get("type", "string"))
                params.append(
                    ToolParameter(
                        name=param_name,
                        param_type=param_type,
                        description=param_config.get("description", ""),
                        required=param_config.get("required", False),
                    )
                )
            execute_func = self._make_tool_executor(tool_name, chat_id)
            self._tool_registry.register_tool(
                DreamTool(
                    name=tool_name,
                    description=tool_config.get("description", ""),
                    parameters=params,
                    execute_func=execute_func,
                )
            )

    def _make_tool_executor(
        self, tool_name: str, chat_id: str
    ) -> Callable[..., Awaitable[str]]:
        async def executor(**kwargs) -> str:
            if not self._db_adapter:
                return f"错误: 数据库适配器未初始化"
            try:
                if tool_name == "search_chat_history":
                    return await self._db_adapter.search_chat_history(
                        chat_id=chat_id,
                        keyword=kwargs.get("clue", kwargs.get("keyword", "")),
                        participant=kwargs.get("participant", ""),
                    )
                elif tool_name == "get_chat_history_detail":
                    return await self._db_adapter.get_chat_history_detail(
                        memory_id=kwargs.get("memory_id"),
                    )
                elif tool_name == "delete_chat_history":
                    return await self._db_adapter.delete_chat_history(
                        memory_id=kwargs.get("memory_id"),
                        keywords=kwargs.get(
                            "clues", kwargs.get("keywords", "[]")
                        ),
                    )
                elif tool_name == "create_chat_history":
                    return await self._db_adapter.create_chat_history(
                        chat_id=chat_id,
                        theme=kwargs.get("theme", ""),
                        summary=kwargs.get("summary", ""),
                        keywords=kwargs.get("keywords", "[]"),
                        key_point=kwargs.get("key_point", "[]"),
                        end_time=kwargs.get("end_time", ""),
                    )
                elif tool_name == "update_chat_history":
                    return await self._db_adapter.update_chat_history(
                        memory_id=kwargs.get("memory_id"),
                        keyword=kwargs.get("clue", kwargs.get("keyword", "")),
                        summary=kwargs.get("summary"),
                        keywords=kwargs.get("keywords"),
                        key_point=kwargs.get("key_point"),
                    )
                elif tool_name == "finish_maintenance":
                    return f"维护结束: {kwargs.get('reason', '完成')}"
                elif tool_name == "search_jargon":
                    return await self._db_adapter.search_jargon(
                        chat_id=chat_id,
                        keyword=kwargs.get("keyword", ""),
                    )
                else:
                    return f"未知工具: {tool_name}"
            except Exception as e:
                logger.error(f"工具执行失败 {tool_name}: {e}")
                return f"工具执行错误: {str(e)}"

        return executor

    def can_run(self, chat_id: str) -> Tuple[bool, str]:
        cooldown = get_cooldown_seconds()
        last_run = self._last_run_time.get(chat_id, 0)
        elapsed = time.time() - last_run
        if elapsed < cooldown:
            remaining = int(cooldown - elapsed)
            return False, f"冷却中，还需等待 {remaining} 秒"
        if chat_id in self._running_contexts:
            return False, "正在运行中"
        return True, ""

    async def run_maintenance(
        self,
        chat_id: str,
        start_memory_id: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        can_run, reason = self.can_run(chat_id)
        if not can_run:
            return {"success": False, "reason": reason}
        if max_iterations is None:
            max_iterations = get_max_iterations()
        self._init_tools_for_chat(chat_id)
        context = MaintenanceContext(
            chat_id=chat_id,
            start_memory_id=start_memory_id,
            max_iterations=max_iterations,
        )
        self._running_contexts[chat_id] = context
        self._last_run_time[chat_id] = time.time()
        logger.info(
            f"[dream] 开始维护 chat_id={chat_id}，最多 {max_iterations} 轮"
        )
        try:
            result = await self._run_maintenance_loop(context)
            return result
        except Exception as e:
            logger.error(f"[dream] 维护失败: {e}")
            return {"success": False, "reason": str(e)}
        finally:
            self._running_contexts.pop(chat_id, None)

    async def _run_maintenance_loop(
        self, context: MaintenanceContext
    ) -> Dict[str, Any]:
        while (
            context.current_iteration < context.max_iterations
            and not context.is_finished
        ):
            context.current_iteration += 1
            iteration_result = await self._run_single_iteration(context)
            if iteration_result.get("finished", False):
                context.is_finished = True
                context.finish_reason = iteration_result.get("reason", "完成")
                break
            await asyncio.sleep(0.5)
        duration = time.time() - context.start_time
        return {
            "success": True,
            "chat_id": context.chat_id,
            "iterations": context.current_iteration,
            "tools_used": context.tools_used,
            "records_modified": context.records_modified,
            "duration_seconds": round(duration, 2),
            "finish_reason": context.finish_reason,
        }

    async def _run_single_iteration(
        self, context: MaintenanceContext
    ) -> Dict[str, Any]:
        if not self._llm_bridge:
            return {"finished": True, "reason": "LLM 桥接未初始化"}
        goals = get_goals()
        merge_rules = get_merge_rules()
        tool_defs = self._tool_registry.get_tool_definitions()
        iteration_prompt = self._build_iteration_prompt(
            context, goals, merge_rules, tool_defs
        )
        try:
            response = await self._llm_bridge.chat(
                messages=[{"role": "user", "content": iteration_prompt}],
                tools=tool_defs,
            )
            if response.get("tool_calls"):
                for tool_call in response["tool_calls"]:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    tool = self._tool_registry.get_tool(tool_name)
                    if tool:
                        context.tools_used.append(tool_name)
                        if tool_name == "finish_maintenance":
                            return {
                                "finished": True,
                                "reason": tool_args.get("reason", "完成"),
                            }
                        result = await tool.execute(**tool_args)
                        logger.debug(
                            f"[dream] 工具 {tool_name} 结果: {result[:100]}"
                        )
                        if tool_name in [
                            "create_chat_history",
                            "update_chat_history",
                            "delete_chat_history",
                        ]:
                            context.records_modified += 1
            return {"finished": False}
        except Exception as e:
            logger.error(f"[dream] 迭代 {context.current_iteration} 失败: {e}")
            return {"finished": True, "reason": str(e)}

    def _build_iteration_prompt(
        self,
        context: MaintenanceContext,
        goals: List[str],
        merge_rules: List[str],
        tool_defs: List[Dict],
    ) -> str:
        goals_text = "\n".join([f"- {g}" for g in goals])
        merge_text = "\n".join([f"- {r}" for r in merge_rules])
        tools_text = "\n".join(
            [f"- {t['name']}: {t['description']}" for t in tool_defs]
        )
        template = get_prompt_template("iteration_prompt")
        instruction = get_instruction_template("think_first")
        if template:
            return template.format(
                iteration_info=f"当前是第 {context.current_iteration}/{context.max_iterations} 轮维护。",
                chat_info=f"本轮维护的 chat_id: {context.chat_id}",
                start_memory_info=f"起始记忆 ID: {context.start_memory_id or '无（自由选择切入点）'}",
                goals_block=f"工作目标:\n{goals_text}",
                merge_rules_block=f"合并准则:\n{merge_text}",
                tools_block=f"可用工具:\n{tools_text}",
                instruction=instruction,
            )
        parts = [
            f"当前是第 {context.current_iteration}/{context.max_iterations} 轮维护。",
            f"本轮维护的 chat_id: {context.chat_id}",
            f"起始记忆 ID: {context.start_memory_id or '无（自由选择切入点）'}",
            "",
            f"工作目标:\n{goals_text}",
            "",
            f"合并准则:\n{merge_text}",
            "",
            f"可用工具:\n{tools_text}",
        ]
        if instruction:
            parts.append("")
            parts.append(instruction)
        return "\n".join(parts)


_dream_agent_instance: Optional[DreamAgent] = None


def get_dream_agent(llm_bridge=None, db_adapter=None) -> DreamAgent:
    global _dream_agent_instance
    if _dream_agent_instance is None:
        _dream_agent_instance = DreamAgent(llm_bridge=llm_bridge, db_adapter=db_adapter)
    elif db_adapter is not None and _dream_agent_instance._db_adapter is None:
        _dream_agent_instance.set_db_adapter(db_adapter)
    return _dream_agent_instance


def init_dream_agent(llm_bridge=None, db_adapter=None) -> DreamAgent:
    """初始化梦境代理，必须在使用前调用"""
    global _dream_agent_instance
    _dream_agent_instance = DreamAgent(llm_bridge=llm_bridge, db_adapter=db_adapter)
    return _dream_agent_instance


def reset_dream_agent():
    global _dream_agent_instance
    _dream_agent_instance = None

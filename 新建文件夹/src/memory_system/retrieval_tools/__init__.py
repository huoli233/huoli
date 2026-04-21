from importlib import import_module

from .tool_registry import (
    MemoryRetrievalTool,
    MemoryRetrievalToolRegistry,
    get_tool_registry,
    register_memory_retrieval_tool,
)

_REGISTERED_MODULES = (
    "query_chat_history",
    "query_embedding_memory",
    "query_lpmm_knowledge",
    "query_person_info",
    "query_words",
    "return_information",
)
_tools_initialized = False


def init_all_tools() -> None:
    global _tools_initialized

    if _tools_initialized:
        return

    for module_name in _REGISTERED_MODULES:
        module = import_module(f"{__name__}.{module_name}")
        register_tool = getattr(module, "register_tool", None)
        if callable(register_tool):
            register_tool()

    _tools_initialized = True


__all__ = [
    "MemoryRetrievalTool",
    "MemoryRetrievalToolRegistry",
    "get_tool_registry",
    "init_all_tools",
    "register_memory_retrieval_tool",
]

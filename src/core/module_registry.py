import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable, Type
from enum import Enum
from src.common.logger import get_logger

logger = get_logger("模块注册表")


class ModuleStatus(Enum):
    """模块状态"""

    REGISTERED = "registered"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"


class ModulePriority(Enum):
    """模块优先级"""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    OPTIONAL = 4


@dataclass
class ModuleInfo:
    """模块信息"""

    name: str
    version: str = "1.0.0"
    description: str = ""
    functions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: ModuleStatus = ModuleStatus.REGISTERED
    priority: ModulePriority = ModulePriority.NORMAL
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    use_count: int = 0
    instance: Any = None
    factory: Optional[Callable] = None


@dataclass
class ConflictReport:
    """冲突报告"""

    function_name: str
    existing_modules: List[str]
    new_module: str
    severity: str  # "warning", "error"
    suggestion: str


class ModuleRegistry:
    """
    模块注册系统

    防止模块冗余，追踪依赖关系。
    """

    _instance: Optional["ModuleRegistry"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._modules: Dict[str, ModuleInfo] = {}
        self._function_map: Dict[str, List[str]] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._conflicts: List[ConflictReport] = []
        self._initialization_order: List[str] = []
        self._lock = threading.RLock()
        logger.info("[模块注册] 模块注册系统初始化完成")

    def register(
        self,
        name: str,
        functions: List[str],
        version: str = "1.0.0",
        description: str = "",
        dependencies: Optional[List[str]] = None,
        priority: ModulePriority = ModulePriority.NORMAL,
        instance: Any = None,
        factory: Optional[Callable] = None,
        allow_override: bool = False,
    ) -> bool:
        """注册模块"""
        with self._lock:
            if name in self._modules:
                existing = self._modules[name]
                if existing.status == ModuleStatus.ACTIVE and not allow_override:
                    logger.warning(f"[模块注册] 模块已存在: {name}")
                    return False
            conflicts = self._check_function_conflicts(name, functions)
            if conflicts and not allow_override:
                for conflict in conflicts:
                    self._conflicts.append(conflict)
                    if conflict.severity == "error":
                        logger.error(
                            f"[模块注册] 功能冲突: {
                                conflict.function_name} 已存在于 {
                                conflict.existing_modules}"
                        )
                        return False
                    else:
                        logger.warning(
                            f"[模块注册] 功能重叠警告: {conflict.function_name}"
                        )
            module = ModuleInfo(
                name=name,
                version=version,
                description=description,
                functions=functions,
                dependencies=dependencies or [],
                priority=priority,
                instance=instance,
                factory=factory,
            )
            self._modules[name] = module
            for func in functions:
                if func not in self._function_map:
                    self._function_map[func] = []
                if name not in self._function_map[func]:
                    self._function_map[func].append(name)
            if dependencies:
                self._dependency_graph[name] = set(dependencies)
            else:
                self._dependency_graph[name] = set()
            self._update_initialization_order()
            logger.info(
                f"[模块注册] 注册模块: {name} (v{version}), 功能: {len(functions)}"
            )
            return True

    def unregister(self, name: str) -> bool:
        """注销模块"""
        with self._lock:
            if name not in self._modules:
                return False
            module = self._modules.pop(name)
            for func in module.functions:
                if func in self._function_map:
                    if name in self._function_map[func]:
                        self._function_map[func].remove(name)
                    if not self._function_map[func]:
                        del self._function_map[func]
            self._dependency_graph.pop(name, None)
            if name in self._initialization_order:
                self._initialization_order.remove(name)
            logger.info(f"[模块注册] 注销模块: {name}")
            return True

    def get_module(self, name: str) -> Optional[ModuleInfo]:
        """获取模块"""
        with self._lock:
            module = self._modules.get(name)
            if module:
                module.last_used = time.time()
                module.use_count += 1
            return module

    def get_module_instance(self, name: str) -> Any:
        """获取模块实例"""
        with self._lock:
            module = self.get_module(name)
            if module is None:
                return None
            if module.instance is not None:
                return module.instance
            if module.factory is not None:
                module.instance = module.factory()
                return module.instance
            return None

    def get_module_for_function(self, function_name: str) -> Optional[str]:
        """获取实现某功能的模块"""
        with self._lock:
            modules = self._function_map.get(function_name, [])
            if not modules:
                return None
            active = [
                m
                for m in modules
                if self._modules.get(
                    m, ModuleInfo(status=ModuleStatus.DISABLED)
                ).status
                == ModuleStatus.ACTIVE
            ]
            if not active:
                return modules[0]
            return active[0]

    def get_all_modules_for_function(self, function_name: str) -> List[str]:
        """获取所有实现某功能的模块"""
        with self._lock:
            return self._function_map.get(function_name, [])

    def find_modules_by_pattern(self, pattern: str) -> List[str]:
        """按模式查找模块"""
        import re

        with self._lock:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                return [name for name in self._modules if regex.search(name)]
            except re.error:
                return [
                    name
                    for name in self._modules
                    if pattern.lower() in name.lower()
                ]

    def _check_function_conflicts(
        self, new_module: str, functions: List[str]
    ) -> List[ConflictReport]:
        """检查功能冲突"""
        conflicts = []
        for func in functions:
            if func in self._function_map:
                existing = self._function_map[func]
                if existing:
                    conflicts.append(
                        ConflictReport(
                            function_name=func,
                            existing_modules=existing.copy(),
                            new_module=new_module,
                            severity="warning",
                            suggestion=f"考虑合并 {existing} 和 {new_module} 的 {func} 功能",
                        )
                    )
        return conflicts

    def _update_initialization_order(self) -> None:
        """更新初始化顺序（拓扑排序）"""
        visited: Set[str] = set()
        temp_visited: Set[str] = set()
        order: List[str] = []

        def visit(name: str) -> None:
            if name in temp_visited:
                logger.warning(f"检测到循环依赖: {name}，初始化顺序可能不正确")
                return
            if name in visited:
                return
            temp_visited.add(name)
            for dep in self._dependency_graph.get(name, set()):
                if dep in self._modules:
                    visit(dep)
            temp_visited.remove(name)
            visited.add(name)
            order.append(name)

        sorted_modules = sorted(
            self._modules.keys(),
            key=lambda x: self._modules[x].priority.value,
        )
        for name in sorted_modules:
            visit(name)
        self._initialization_order = order

    def get_initialization_order(self) -> List[str]:
        """获取初始化顺序"""
        return self._initialization_order.copy()

    def check_circular_dependencies(self) -> List[List[str]]:
        """检查循环依赖"""
        with self._lock:
            cycles: List[List[str]] = []
            visited: Set[str] = set()
            rec_stack: Set[str] = set()

            def dfs(node: str, path: List[str]) -> None:
                visited.add(node)
                rec_stack.add(node)
                for neighbor in self._dependency_graph.get(node, set()):
                    if neighbor not in visited:
                        dfs(neighbor, path + [neighbor])
                    elif neighbor in rec_stack:
                        cycle_start = (
                            path.index(neighbor) if neighbor in path else len(path)
                        )
                        cycle = path[cycle_start:] + [neighbor]
                        cycles.append(cycle)
                rec_stack.remove(node)

            for node in list(self._modules.keys()):
                if node not in visited:
                    dfs(node, [node])
            return cycles

    def get_redundancy_report(self) -> Dict[str, Any]:
        """获取冗余报告"""
        with self._lock:
            redundant_functions: Dict[str, List[str]] = {}
            for func, modules in self._function_map.items():
                if len(modules) > 1:
                    redundant_functions[func] = modules
            return {
                "total_modules": len(self._modules),
                "total_functions": len(self._function_map),
                "redundant_functions": redundant_functions,
                "redundancy_count": len(redundant_functions),
                "conflicts": [
                    {
                        "function": c.function_name,
                        "modules": c.existing_modules + [c.new_module],
                        "suggestion": c.suggestion,
                    }
                    for c in self._conflicts
                ],
            }

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        circular = self.check_circular_dependencies()
        redundancy = self.get_redundancy_report()
        missing_deps: Dict[str, List[str]] = {}
        with self._lock:
            for name, deps in self._dependency_graph.items():
                missing = [d for d in deps if d not in self._modules]
                if missing:
                    missing_deps[name] = missing
            return {
                "healthy": len(circular) == 0 and len(missing_deps) == 0,
                "circular_dependencies": circular,
                "missing_dependencies": missing_deps,
                "redundancy": redundancy,
                "module_count": len(self._modules),
                "function_count": len(self._function_map),
            }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            by_status: Dict[str, int] = {}
            by_priority: Dict[str, int] = {}
            for module in self._modules.values():
                st = module.status.value
                by_status[st] = by_status.get(st, 0) + 1
                pr = module.priority.name
                by_priority[pr] = by_priority.get(pr, 0) + 1
            return {
                "total_modules": len(self._modules),
                "total_functions": len(self._function_map),
                "by_status": by_status,
                "by_priority": by_priority,
                "conflicts_recorded": len(self._conflicts),
            }


_module_registry: Optional[ModuleRegistry] = None


def get_module_registry() -> ModuleRegistry:
    """获取模块注册系统单例"""
    global _module_registry
    if _module_registry is None:
        _module_registry = ModuleRegistry()
    return _module_registry


def register_module(
    name: str,
    functions: List[str],
    **kwargs,
) -> bool:
    """便捷注册函数"""
    return get_module_registry().register(name, functions, **kwargs)

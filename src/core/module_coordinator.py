from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import time

from src.common.logger import get_logger

logger = get_logger("模块协调")


# ---------------------------------------------------------------------------
#  枚举定义
# ---------------------------------------------------------------------------


class ModuleCategory(Enum):
    """模块类别"""

    COGNITION = "cognition"  # 认知系统
    DECISION = "decision"  # 决策系统
    ACTION = "action"  # 行为执行
    MEMORY = "memory"  # 记忆系统
    INTERACTION = "interaction"  # 交互系统
    PLANNING = "planning"  # 规划系统
    PERCEPTION = "perception"  # 感知系统
    ENERGY = "energy"  # 能量系统
    UTILITY = "utility"  # 工具类模块


# ---------------------------------------------------------------------------
#  数据模型
# ---------------------------------------------------------------------------


@dataclass
class ModuleCapability:
    """模块能力定义"""

    name: str  # 能力名称，如 "emotion.assess"
    description: str = ""  # 能力描述
    thinking_cost: float = 10.0  # 该能力消耗的思考值
    priority: int = 5  # 优先级（0=最高，10=最低）
    version: str = "1.0"  # 能力版本
    is_active: bool = True


@dataclass
class ModuleProfile:
    """模块档案"""

    module_id: str
    name: str
    category: ModuleCategory
    capabilities: List[ModuleCapability]
    instance: Any = None
    factory: Optional[Callable] = None
    is_active: bool = True
    registered_at: float = field(default_factory=time.time)
    last_called: float = 0.0
    call_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    total_thinking_consumed: float = 0.0


@dataclass
class CallContext:
    """模块调用上下文"""

    channel_id: str = ""
    user_id: str = ""
    caller: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModuleCoordinator:
    """
    模块协调器

    管理所有模块的注册和发现。
    """

    _instance: Optional["ModuleCoordinator"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._modules: Dict[str, ModuleProfile] = {}
        self._capability_map: Dict[str, List[str]] = {}
        self._category_modules: Dict[ModuleCategory, List[str]] = {}
        self._integration_status: Dict[str, bool] = {}
        logger.info("[模块协调] 初始化完成")

    def register(
        self,
        module_id: str,
        name: str,
        category: ModuleCategory,
        capabilities: List[ModuleCapability],
        instance: Any = None,
        factory: Optional[Callable] = None,
    ) -> bool:
        """注册模块"""
        if module_id in self._modules:
            logger.warning(f"[模块协调] 模块已存在: {module_id}")
            return False
        for cap in capabilities:
            if cap.name in self._capability_map:
                existing = self._capability_map[cap.name]
                if existing:
                    logger.warning(
                        f"[模块协调] 能力冲突: {cap.name} 已存在于 {existing}"
                    )
        profile = ModuleProfile(
            module_id=module_id,
            name=name,
            category=category,
            capabilities=capabilities,
            instance=instance,
            factory=factory,
        )
        self._modules[module_id] = profile
        for cap in capabilities:
            if cap.name not in self._capability_map:
                self._capability_map[cap.name] = []
            self._capability_map[cap.name].append(module_id)
        if category not in self._category_modules:
            self._category_modules[category] = []
        self._category_modules[category].append(module_id)
        logger.info(
            f"[模块协调] 注册成功: {name} ({category.value}), "
            f"能力: {[c.name for c in capabilities]}"
        )
        return True

    def get_module(self, module_id: str) -> Optional[ModuleProfile]:
        """获取模块档案"""
        return self._modules.get(module_id)

    def discover_and_register_existing_modules(self) -> Dict[str, bool]:
        """发现并注册现有模块"""
        results = {}
        results.update(self._register_planning_modules())
        results.update(self._register_memory_modules())
        results.update(self._register_energy_modules())
        results.update(self._register_perception_modules())
        results.update(self._register_social_modules())
        self._integration_status = results
        return results

    def _register_planning_modules(self) -> Dict[str, bool]:
        """注册规划模块"""
        results = {}
        try:
            from src.core.unified_planner import get_unified_planner

            planner = get_unified_planner()
            self.register(
                module_id="unified_planner",
                name="统一规划器",
                category=ModuleCategory.PLANNING,
                capabilities=[
                    ModuleCapability(
                        name="plan",
                        description="执行规划流程",
                        thinking_cost=3.0,
                        priority=1,
                    ),
                    ModuleCapability(
                        name="perceive",
                        description="感知环境",
                        thinking_cost=1.0,
                        priority=2,
                    ),
                    ModuleCapability(
                        name="decide",
                        description="做出决策",
                        thinking_cost=2.0,
                        priority=2,
                    ),
                ],
                instance=planner,
            )
            results["unified_planner"] = True
        except Exception as e:
            logger.warning(f"[模块协调] 注册统一规划器失败: {e}")
            results["unified_planner"] = False
        return results

    def _register_memory_modules(self) -> Dict[str, bool]:
        """注册记忆模块"""
        results = {}
        try:
            from src.memory_system.memory_core import acquire_recollection_hub

            hub = acquire_recollection_hub()
            self.register(
                module_id="recollection_hub",
                name="记忆核心",
                category=ModuleCategory.MEMORY,
                capabilities=[
                    ModuleCapability(
                        name="deposit_memory",
                        description="存入记忆",
                        thinking_cost=0.5,
                        priority=3,
                    ),
                    ModuleCapability(
                        name="query_memories",
                        description="查询记忆",
                        thinking_cost=1.0,
                        priority=3,
                    ),
                    ModuleCapability(
                        name="execute_routine_maintenance",
                        description="执行维护",
                        thinking_cost=2.0,
                        priority=5,
                    ),
                ],
                instance=hub,
            )
            results["recollection_hub"] = True
        except Exception as e:
            logger.warning(f"[模块协调] 注册记忆核心失败: {e}")
            results["recollection_hub"] = False
        try:
            from src.core.dynamic_context_window import get_dynamic_context

            ctx = get_dynamic_context()
            self.register(
                module_id="dynamic_context",
                name="动态上下文",
                category=ModuleCategory.MEMORY,
                capabilities=[
                    ModuleCapability(
                        name="add_message",
                        description="添加消息",
                        thinking_cost=0.3,
                        priority=2,
                    ),
                    ModuleCapability(
                        name="get_context",
                        description="获取上下文",
                        thinking_cost=0.5,
                        priority=2,
                    ),
                ],
                instance=ctx,
            )
            results["dynamic_context"] = True
        except Exception as e:
            logger.warning(f"[模块协调] 注册动态上下文失败: {e}")
            results["dynamic_context"] = False
        return results

    def _register_energy_modules(self) -> Dict[str, bool]:
        """注册能量模块"""
        results = {}
        try:
            from src.chat.heart_flow.energy_manager import EnergyChainDimension

            _d6 = EnergyChainDimension.get_instance()
            self.register(
                module_id="energy_chain",
                name="D6能量链条",
                category=ModuleCategory.ENERGY,
                capabilities=[
                    ModuleCapability(
                        name="capture_snapshot",
                        description="获取快照",
                        thinking_cost=0.1,
                        priority=1,
                    ),
                    ModuleCapability(
                        name="deplete_on_reply",
                        description="回复消耗",
                        thinking_cost=0.1,
                        priority=2,
                    ),
                    ModuleCapability(
                        name="deplete_on_think",
                        description="思考消耗",
                        thinking_cost=0.1,
                        priority=2,
                    ),
                    ModuleCapability(
                        name="compute_eagerness",
                        description="计算意愿度",
                        thinking_cost=0.2,
                        priority=2,
                    ),
                ],
                instance=_d6,
            )
            results["energy_chain"] = True
        except Exception as e:
            logger.warning(f"[模块协调] 注册能量管理失败: {e}")
            results["energy_chain"] = False
        return results

    def _register_perception_modules(self) -> Dict[str, bool]:
        """注册感知模块"""
        results = {}
        try:
            from src.chat.heart_flow.inner_voice import (
                get_self_dialogue_engine,
            )

            self.register(
                module_id="inner_voice",
                name="内心独白",
                category=ModuleCategory.PERCEPTION,
                capabilities=[
                    ModuleCapability(
                        name="generate_reflection",
                        description="生成内心想法",
                        thinking_cost=2.0,
                        priority=2,
                    ),
                ],
                factory=lambda cid="default": get_self_dialogue_engine(cid),
            )
            results["inner_voice"] = True
        except Exception as e:
            logger.warning(f"[模块协调] 注册内心独白失败: {e}")
            results["inner_voice"] = False
        return results

    def _register_social_modules(self) -> Dict[str, bool]:
        """注册社交模块"""
        results = {}
        try:
            from src.chat.heart_flow.energy_manager import EnergyChainDimension

            _d6 = EnergyChainDimension.get_instance()
            self.register(
                module_id="social_value_core",
                name="社交值核心(D6)",
                category=ModuleCategory.INTERACTION,
                capabilities=[
                    ModuleCapability(
                        name="get_social_value",
                        description="获取社交值",
                        thinking_cost=0.5,
                        priority=2,
                    ),
                ],
                instance=_d6,
            )
            results["social_value_core"] = True
        except Exception as e:
            logger.warning(f"[模块协调] 注册社交值核心失败: {e}")
            results["social_value_core"] = False
        return results

    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告"""
        total = len(self._modules)
        active = sum(1 for p in self._modules.values() if p.is_active)
        total_calls = sum(p.call_count for p in self._modules.values())
        total_errors = sum(p.error_count for p in self._modules.values())
        avg_latency = 0.0
        if total_calls > 0:
            total_latency = sum(
                p.total_latency for p in self._modules.values()
            )
            avg_latency = total_latency / total_calls
        redundant_capabilities = [
            cap
            for cap, modules in self._capability_map.items()
            if len(modules) > 1
        ]
        return {
            "total_modules": total,
            "active_modules": active,
            "total_calls": total_calls,
            "total_errors": total_errors,
            "error_rate": total_errors / max(1, total_calls),
            "avg_latency": avg_latency,
            "redundant_capabilities": redundant_capabilities,
            "integration_status": self._integration_status,
        }


_coordinator: Optional[ModuleCoordinator] = None


def get_module_coordinator() -> ModuleCoordinator:
    """获取模块协调器单例"""
    global _coordinator
    if _coordinator is None:
        _coordinator = ModuleCoordinator()
    return _coordinator


def discover_modules() -> Dict[str, bool]:
    """发现并注册所有模块"""
    return get_module_coordinator().discover_and_register_existing_modules()

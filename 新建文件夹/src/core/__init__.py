from typing import Dict, Any, Optional, List

from src.core.identity_anchor import (
    IdentityAnchor,
    IdentityMode,
    ResponseMode,
    IdentityContext,
    IdentityConflict,
    get_identity_anchor,
)

from src.core.dynamic_context_window import (
    DynamicContextWindow,
    ContextWindowConfig,
    MessageImportance,
    TopicContext,
    get_dynamic_context,
)

from src.core.self_reply_recognizer import (
    SelfReplyRecognizer,
    MessageSource,
    ResponseQuality,
    BotMessageRecord,
    ReplyChain,
    get_self_reply_recognizer,
)

from src.core.content_state_tracker import (
    ContentStateTracker,
    ContentStatus,
    IgnoreReason,
    ContentProcessingState,
    ContentDecision,
    get_content_state_tracker,
)

from src.core.module_registry import (
    ModuleRegistry,
    ModuleInfo,
    ModuleStatus,
    ModulePriority,
    ConflictReport,
    get_module_registry,
    register_module,
)

from src.core.unified_planner import (
    UnifiedPlanner,
    PlanningPhase,
    ActionType,
    ThinkingBudget,
    ContextSegment,
    PlanningContext,
    PlanningDecision,
    ContextIndexer,
    ContextFilter,
    get_unified_planner,
)

from src.core.module_coordinator import (
    ModuleCoordinator,
    ModuleCategory,
    ModuleCapability,
    ModuleProfile,
    CallContext,
    get_module_coordinator,
    discover_modules,
)

from src.modules.social_value import (
    SocialValueCore,
    SocialCalculator,
    SocialAffectFuser,
    get_social_affect_fuser,
)


class HuoLiCore:
    """HuoLi核心管理器

    统一管理所有核心模块的访问和初始化。
    采用懒加载模式，按需初始化各模块。
    """

    _instance: Optional["HuoLiCore"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._social_value: Optional[SocialValueCore] = None
        self._unified_planner: Optional[UnifiedPlanner] = None
        self._module_coordinator: Optional[ModuleCoordinator] = None
        self._identity_anchor: Optional[IdentityAnchor] = None
        self._dynamic_context: Optional[DynamicContextWindow] = None
        self._self_reply_recognizer: Optional[SelfReplyRecognizer] = None
        self._content_state_tracker: Optional[ContentStateTracker] = None
        self._module_registry: Optional[ModuleRegistry] = None
        self._initialized = True

    @property
    def social_value(self) -> SocialValueCore:
        """社交值核心"""
        if self._social_value is None:
            from src.core.compat_bridge import (
                SocialCalculator,
            )
            from src.core.compat_bridge import SocialStorage

            calculator = SocialCalculator()
            storage = SocialStorage()
            self._social_value = SocialValueCore(calculator, storage)
        return self._social_value

    @property
    def unified_planner(self) -> UnifiedPlanner:
        """统一规划器"""
        if self._unified_planner is None:
            self._unified_planner = get_unified_planner()
        return self._unified_planner

    @property
    def module_coordinator(self) -> ModuleCoordinator:
        """模块协调器"""
        if self._module_coordinator is None:
            self._module_coordinator = get_module_coordinator()
        return self._module_coordinator

    @property
    def identity(self) -> IdentityAnchor:
        """身份锚点"""
        if self._identity_anchor is None:
            self._identity_anchor = get_identity_anchor()
        return self._identity_anchor

    @property
    def dynamic_context(self) -> DynamicContextWindow:
        """动态上下文窗口"""
        if self._dynamic_context is None:
            self._dynamic_context = get_dynamic_context()
        return self._dynamic_context

    @property
    def self_reply_recognizer(self) -> SelfReplyRecognizer:
        """自回复识别器"""
        if self._self_reply_recognizer is None:
            self._self_reply_recognizer = get_self_reply_recognizer()
        return self._self_reply_recognizer

    @property
    def content_state(self) -> ContentStateTracker:
        """内容状态追踪器"""
        if self._content_state_tracker is None:
            self._content_state_tracker = get_content_state_tracker()
        return self._content_state_tracker

    @property
    def module_registry(self) -> ModuleRegistry:
        """模块注册表"""
        if self._module_registry is None:
            self._module_registry = get_module_registry()
        return self._module_registry

    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有模块统计信息"""
        return {
            "social_value": {
                "available": self._social_value is not None,
            },
        }

    def get_module_list(self) -> List[str]:
        """获取所有可用模块列表"""
        return [
            "social_value",
            "unified_planner",
            "module_coordinator",
            "identity",
            "dynamic_context",
            "self_reply_recognizer",
            "content_state",
            "module_registry",
        ]


def get_core() -> HuoLiCore:
    """获取HuoLi核心管理器单例"""
    return HuoLiCore()


__all__ = [
    "IdentityAnchor",
    "IdentityMode",
    "ResponseMode",
    "IdentityContext",
    "IdentityConflict",
    "get_identity_anchor",
    "DynamicContextWindow",
    "ContextWindowConfig",
    "MessageImportance",
    "TopicContext",
    "get_dynamic_context",
    "SelfReplyRecognizer",
    "MessageSource",
    "ResponseQuality",
    "BotMessageRecord",
    "ReplyChain",
    "get_self_reply_recognizer",
    "ContentStateTracker",
    "ContentStatus",
    "IgnoreReason",
    "ContentProcessingState",
    "ContentDecision",
    "get_content_state_tracker",
    "ModuleRegistry",
    "ModuleInfo",
    "ModuleStatus",
    "ModulePriority",
    "ConflictReport",
    "get_module_registry",
    "register_module",
    "UnifiedPlanner",
    "PlanningPhase",
    "ActionType",
    "ThinkingBudget",
    "ContextSegment",
    "PlanningContext",
    "PlanningDecision",
    "ContextIndexer",
    "ContextFilter",
    "get_unified_planner",
    "ModuleCoordinator",
    "ModuleCategory",
    "ModuleCapability",
    "ModuleProfile",
    "CallContext",
    "get_module_coordinator",
    "discover_modules",
    "SocialValueCore",
    "SocialCalculator",
    "RelationshipTracker",
    "HuoLiCore",
    "get_core",
]


def initialize_core_systems() -> Dict[str, Any]:
    """初始化所有核心系统"""
    from src.common.logger import get_logger

    logger = get_logger("core_systems")
    logger.info("[核心系统] 开始初始化核心系统...")
    coordinator = get_module_coordinator()
    integration_results = coordinator.discover_and_register_existing_modules()
    health = coordinator.get_health_report()
    successful = sum(1 for v in integration_results.values() if v)
    total = len(integration_results)
    if health.get("total_errors", 0) == 0:
        logger.info(
            f"[核心系统] 核心系统初始化完成，成功注册 {successful}/{total} 个模块"
        )
    else:
        logger.warning(f"[核心系统] 核心系统初始化完成，存在问题: {health}")
    return {
        "coordinator": coordinator,
        "unified_planner": get_unified_planner(),
        "integration_results": integration_results,
        "health": health,
        "core": get_core(),
    }

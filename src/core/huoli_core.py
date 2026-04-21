from typing import Dict, Any, Optional, List
from src.common.logger import get_logger
from src.core.identity_anchor import (
    IdentityAnchor,
    get_identity_anchor,
)
from src.core.dynamic_context_window import (
    DynamicContextWindow,
    get_dynamic_context,
)
from src.core.self_reply_recognizer import (
    SelfReplyRecognizer,
    get_self_reply_recognizer,
)
from src.core.content_state_tracker import (
    ContentStateTracker,
    get_content_state_tracker,
)
from src.core.module_registry import (
    ModuleRegistry,
    get_module_registry,
)
from src.core.unified_planner import (
    UnifiedPlanner,
    get_unified_planner,
)
from src.core.module_coordinator import (
    ModuleCoordinator,
    get_module_coordinator,
)

logger = get_logger("huoli_core")

try:
    from src.modules.social_value.social_value_core import SocialValueCore
except ImportError:
    SocialValueCore = None


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
        self._social_value = None
        self._unified_planner: Optional[UnifiedPlanner] = None
        self._module_coordinator: Optional[ModuleCoordinator] = None
        self._identity_anchor: Optional[IdentityAnchor] = None
        self._dynamic_context: Optional[DynamicContextWindow] = None
        self._self_reply_recognizer: Optional[SelfReplyRecognizer] = None
        self._content_state_tracker: Optional[ContentStateTracker] = None
        self._module_registry: Optional[ModuleRegistry] = None
        self._initialized = True

    @property
    def social_value(self):
        """社交值核心"""
        if self._social_value is None:
            try:
                from src.modules.safety.social_calculator import get_social_calculator
                calculator = get_social_calculator()
                self._social_value = calculator
            except Exception as exc:
                self._social_value = None
                logger.warning(f"社交值核心初始化失败: {exc}")
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

    def shutdown(self):
        """关闭核心管理器，依次清理所有子模块资源"""
        logger.info("HuoLiCore 开始关闭子模块...")
        shutdown_order = [
            ("module_coordinator", self._module_coordinator),
            ("unified_planner", self._unified_planner),
            ("content_state_tracker", self._content_state_tracker),
            ("dynamic_context", self._dynamic_context),
            ("self_reply_recognizer", self._self_reply_recognizer),
            ("identity_anchor", self._identity_anchor),
            ("module_registry", self._module_registry),
            ("social_value", self._social_value),
        ]
        for name, module in shutdown_order:
            if module is not None and hasattr(module, "shutdown") and callable(module.shutdown):
                try:
                    module.shutdown()
                    logger.info(f"子模块 {name} 已关闭")
                except Exception as exc:
                    logger.warning(f"子模块 {name} 关闭异常: {exc}")
        self._module_coordinator = None
        self._unified_planner = None
        self._content_state_tracker = None
        self._dynamic_context = None
        self._self_reply_recognizer = None
        self._identity_anchor = None
        self._module_registry = None
        self._social_value = None
        logger.info("HuoLiCore 所有子模块已关闭")


def get_core() -> HuoLiCore:
    """获取HuoLi核心管理器单例"""
    return HuoLiCore()

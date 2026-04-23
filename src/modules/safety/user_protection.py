import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple, TYPE_CHECKING

from src.common.logger import get_logger
from src.modules.safety.runtime_config import safety_float, safety_int

if TYPE_CHECKING:
    from src.modules.safety.social_calculator import SocialCalculator

logger = get_logger("用户保护")


@dataclass
class ProtectionState:
    """保护状态数据结构"""

    is_protected: bool = False
    protect_until: float = 0.0
    protect_reason: str = ""
    protect_count: int = 0
    last_protect_time: float = 0.0
    total_protected_seconds: float = 0.0
    severity_history: list = field(default_factory=list)


@dataclass
class ProtectionConfig:
    """保护配置参数"""

    default_duration: float = 300.0
    max_duration: float = 3600.0
    auto_protect_threshold: float = -80.0
    harassment_protect_duration: float = 600.0
    harassment_severity_threshold: float = 0.8
    cooldown_after_expire: float = 60.0
    max_severity_history: int = 10
    escalation_factor: float = 1.5
    max_escalation_count: int = 5


class UserProtectionManager:
    """
    用户保护管理器

    功能:
    1. 管理用户级别的保护状态
    2. 在用户受到过度伤害时自动激活保护
    3. 提供保护状态查询和解除接口
    4. 支持保护升级机制（多次触发延长保护时间）

    互联:
    - 依赖 SocialCalculator: 获取用户社交值
    - 依赖 ConfigEngine: 获取保护参数
    - 被 HarassmentDetector 依赖: 触发保护
    """

    _instance: Optional["UserProtectionManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        social_calculator: Optional["SocialCalculator"] = None,
        config_engine=None,
    ):
        if self._initialized:
            if social_calculator:
                self._social_calculator = social_calculator
            self._config = config_engine if config_engine is not None else self._config
            self._load_config()
            return

        self._social_calculator = social_calculator
        self._config = config_engine
        self._protection_states: Dict[str, ProtectionState] = {}
        self._config_data = ProtectionConfig()
        self._last_cleanup_time = time.time()
        self._cleanup_interval = 3600.0

        self._load_config()
        self._initialized = True
        logger.info("用户保护管理器初始化完成")

    def _load_config(self) -> None:
        """从配置引擎加载参数"""
        if self._config is not None:
            self._config_data.default_duration = self._config.get(
                "user_protection", "default_duration", 300.0
            )
            self._config_data.max_duration = self._config.get(
                "user_protection", "max_duration", 3600.0
            )
            self._config_data.auto_protect_threshold = self._config.get(
                "user_protection", "auto_protect_threshold", -80.0
            )
            self._config_data.harassment_protect_duration = self._config.get(
                "user_protection", "harassment_protect_duration", 600.0
            )
            self._config_data.harassment_severity_threshold = self._config.get(
                "user_protection", "harassment_severity_threshold", 0.8
            )
            self._config_data.cooldown_after_expire = self._config.get(
                "user_protection", "cooldown_after_expire", 60.0
            )
            self._config_data.max_severity_history = self._config.get(
                "user_protection", "max_severity_history", 10
            )
            self._config_data.escalation_factor = self._config.get(
                "user_protection", "escalation_factor", 1.5
            )
            self._config_data.max_escalation_count = self._config.get(
                "user_protection", "max_escalation_count", 5
            )
            return
        self._config_data.default_duration = safety_float("user_protection", "default_duration", 300.0)
        self._config_data.max_duration = safety_float("user_protection", "max_duration", 3600.0)
        self._config_data.auto_protect_threshold = safety_float("user_protection", "auto_protect_threshold", -80.0)
        self._config_data.harassment_protect_duration = safety_float("user_protection", "harassment_protect_duration", 600.0)
        self._config_data.harassment_severity_threshold = safety_float("user_protection", "harassment_severity_threshold", 0.8)
        self._config_data.cooldown_after_expire = safety_float("user_protection", "cooldown_after_expire", 60.0)
        self._config_data.max_severity_history = safety_int("user_protection", "max_severity_history", 10)
        self._config_data.escalation_factor = safety_float("user_protection", "escalation_factor", 1.5)
        self._config_data.max_escalation_count = safety_int("user_protection", "max_escalation_count", 5)

    def set_dependencies(
        self,
        social_calculator: "SocialCalculator",
        config_engine=None,
    ) -> None:
        """设置依赖"""
        self._social_calculator = social_calculator
        self._config = config_engine if config_engine is not None else self._config
        self._load_config()

    def _get_key(self, user_id: str, channel_id: str) -> str:
        """生成存储键"""
        return f"{channel_id}:{user_id}"

    def _get_state(self, key: str) -> ProtectionState:
        """获取或创建保护状态"""
        state = self._protection_states.get(key)
        if state is None:
            state = ProtectionState()
            self._protection_states[key] = state
        return state

    def _cleanup_expired_states(self) -> None:
        """清理过期的保护状态"""
        now = time.time()
        if now - self._last_cleanup_time < self._cleanup_interval:
            return
        self._last_cleanup_time = now
        expired_keys = []
        for key, state in self._protection_states.items():
            if (
                not state.is_protected
                and now - state.last_protect_time > 86400
            ):
                expired_keys.append(key)
        for key in expired_keys:
            del self._protection_states[key]
        if expired_keys:
            logger.debug(f"清理过期保护状态: {len(expired_keys)}个")

    def activate_protection(
        self,
        user_id: str,
        channel_id: str,
        duration: Optional[float] = None,
        reason: str = "",
        severity: float = 0.0,
    ) -> bool:
        """
        激活用户保护

        Args:
            user_id: 用户ID
            channel_id: 频道ID
            duration: 保护时长（秒），None使用默认值
            reason: 保护原因
            severity: 触发严重程度

        Returns:
            bool: 是否成功激活
        """
        key = self._get_key(user_id, channel_id)
        now = time.time()
        state = self._get_state(key)

        if state.is_protected and now < state.protect_until:
            remaining = state.protect_until - now
            logger.debug(
                f"用户已在保护中 | 用户:{user_id[:8]} | 剩余:{remaining:.0f}秒"
            )
            return False

        base_duration = (
            duration
            if duration is not None
            else self._config_data.default_duration
        )
        escalation_count = min(
            state.protect_count, self._config_data.max_escalation_count
        )
        escalation_multiplier = (
            self._config_data.escalation_factor**escalation_count
        )
        final_duration = min(
            base_duration * escalation_multiplier,
            self._config_data.max_duration,
        )

        state.is_protected = True
        state.protect_until = now + final_duration
        state.protect_reason = reason
        state.protect_count += 1
        state.last_protect_time = now
        state.total_protected_seconds += final_duration

        if severity > 0:
            state.severity_history.append(
                {"time": now, "severity": severity, "reason": reason}
            )
            if (
                len(state.severity_history)
                > self._config_data.max_severity_history
            ):
                state.severity_history = state.severity_history[
                    -self._config_data.max_severity_history:
                ]

        logger.warning(
            f"用户保护激活 | 用户:{user_id[:8]} | 原因:{reason} | 持续:{final_duration:.0f}秒 | 累计次数:{state.protect_count}"
        )
        return True

    def is_protected(
        self,
        user_id: str,
        channel_id: str,
    ) -> Tuple[bool, str]:
        """
        检查用户是否处于保护状态

        Args:
            user_id: 用户ID
            channel_id: 频道ID

        Returns:
            Tuple[bool, str]: (是否保护中, 保护信息)
        """
        key = self._get_key(user_id, channel_id)
        state = self._protection_states.get(key)

        if not state or not state.is_protected:
            return False, ""

        now = time.time()
        if now >= state.protect_until:
            state.is_protected = False
            state.protect_until = 0.0
            state.protect_reason = ""
            return False, ""

        remaining = int(state.protect_until - now)
        info = f"保护中，还剩{remaining}秒。原因: {state.protect_reason}"
        return True, info

    def deactivate_protection(
        self,
        user_id: str,
        channel_id: str,
        force: bool = False,
    ) -> bool:
        """
        解除用户保护

        Args:
            user_id: 用户ID
            channel_id: 频道ID
            force: 是否强制解除（忽略冷却期）

        Returns:
            bool: 是否成功解除
        """
        key = self._get_key(user_id, channel_id)
        state = self._protection_states.get(key)

        if state and state.is_protected:
            state.is_protected = False
            state.protect_until = 0.0
            state.protect_reason = ""
            logger.info(f"用户保护解除 | 用户:{user_id[:8]} | 强制:{force}")
            return True

        return False

    def get_remaining_time(
        self,
        user_id: str,
        channel_id: str,
    ) -> float:
        """
        获取保护剩余时间

        Args:
            user_id: 用户ID
            channel_id: 频道ID

        Returns:
            float: 剩余时间（秒），未保护返回0
        """
        key = self._get_key(user_id, channel_id)
        state = self._protection_states.get(key)

        if not state or not state.is_protected:
            return 0.0

        now = time.time()
        if now >= state.protect_until:
            return 0.0

        return state.protect_until - now

    def check_auto_protection(
        self,
        user_id: str,
        channel_id: str,
        social_value: float,
        harassment_severity: float = 0.0,
    ) -> bool:
        """
        检查是否需要自动保护

        Args:
            user_id: 用户ID
            channel_id: 频道ID
            social_value: 当前社交值
            harassment_severity: 骚扰严重程度

        Returns:
            bool: 是否触发了自动保护
        """
        is_protected, _ = self.is_protected(user_id, channel_id)
        if is_protected:
            return False

        if social_value <= self._config_data.auto_protect_threshold:
            self.activate_protection(
                user_id,
                channel_id,
                duration=self._config_data.default_duration,
                reason="社交值过低自动保护",
                severity=abs(social_value) / 100.0,
            )
            return True

        if (
            harassment_severity
            > self._config_data.harassment_severity_threshold
        ):
            self.activate_protection(
                user_id,
                channel_id,
                duration=self._config_data.harassment_protect_duration,
                reason="严重骚扰自动保护",
                severity=harassment_severity,
            )
            return True

        return False

    def get_protection_state(
        self,
        user_id: str,
        channel_id: str,
    ) -> ProtectionState:
        """
        获取保护状态

        Args:
            user_id: 用户ID
            channel_id: 频道ID

        Returns:
            ProtectionState: 保护状态
        """
        key = self._get_key(user_id, channel_id)
        return self._get_state(key)

    def get_protection_summary(
        self,
        user_id: str,
        channel_id: str,
    ) -> Dict[str, Any]:
        """
        获取保护状态摘要

        Args:
            user_id: 用户ID
            channel_id: 频道ID

        Returns:
            Dict: 保护状态摘要
        """
        state = self.get_protection_state(user_id, channel_id)
        is_protected, info = self.is_protected(user_id, channel_id)
        return {
            "is_protected": is_protected,
            "remaining_seconds": self.get_remaining_time(user_id, channel_id),
            "protect_count": state.protect_count,
            "total_protected_seconds": state.total_protected_seconds,
            "last_protect_time": state.last_protect_time,
            "current_reason": state.protect_reason if is_protected else "",
            "severity_history": state.severity_history[-5:],
        }

    def extend_protection(
        self,
        user_id: str,
        channel_id: str,
        additional_seconds: float,
        reason: str = "",
    ) -> bool:
        """
        延长保护时间

        Args:
            user_id: 用户ID
            channel_id: 频道ID
            additional_seconds: 额外保护时间
            reason: 延长原因

        Returns:
            bool: 是否成功延长
        """
        is_protected, _ = self.is_protected(user_id, channel_id)
        if not is_protected:
            return False

        key = self._get_key(user_id, channel_id)
        state = self._protection_states.get(key)
        if not state:
            return False

        now = time.time()
        current_remaining = state.protect_until - now
        new_remaining = min(
            current_remaining + additional_seconds,
            self._config_data.max_duration,
        )
        state.protect_until = now + new_remaining

        if reason:
            state.protect_reason = f"{state.protect_reason} + {reason}"

        logger.info(
            f"保护时间延长 | 用户:{user_id[:8]} | +{additional_seconds:.0f}秒 | 新剩余:{new_remaining:.0f}秒"
        )
        return True

    def reset_protection_count(
        self,
        user_id: str,
        channel_id: str,
    ) -> None:
        """
        重置保护计数（用于长时间无负面行为后）

        Args:
            user_id: 用户ID
            channel_id: 频道ID
        """
        key = self._get_key(user_id, channel_id)
        state = self._protection_states.get(key)
        if state:
            state.protect_count = 0
            state.severity_history = []
            logger.debug(f"保护计数重置 | 用户:{user_id[:8]}")

    def export_state(self) -> Dict[str, Any]:
        """导出状态（用于持久化）"""
        return {
            "protection_states": {
                k: {
                    "is_protected": v.is_protected,
                    "protect_until": v.protect_until,
                    "protect_reason": v.protect_reason,
                    "protect_count": v.protect_count,
                    "last_protect_time": v.last_protect_time,
                    "total_protected_seconds": v.total_protected_seconds,
                    "severity_history": v.severity_history,
                }
                for k, v in self._protection_states.items()
            },
            "config": {
                "default_duration": self._config_data.default_duration,
                "max_duration": self._config_data.max_duration,
                "auto_protect_threshold": self._config_data.auto_protect_threshold,
            },
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """导入状态（用于恢复）"""
        self._protection_states.clear()
        for k, v in state.get("protection_states", {}).items():
            self._protection_states[k] = ProtectionState(
                is_protected=v.get("is_protected", False),
                protect_until=v.get("protect_until", 0.0),
                protect_reason=v.get("protect_reason", ""),
                protect_count=v.get("protect_count", 0),
                last_protect_time=v.get("last_protect_time", 0.0),
                total_protected_seconds=v.get("total_protected_seconds", 0.0),
                severity_history=v.get("severity_history", []),
            )
        logger.info(
            f"用户保护状态恢复完成 | 用户数:{len(self._protection_states)}"
        )


_user_protection_manager: Optional[UserProtectionManager] = None


def get_user_protection_manager(
    social_calculator: Optional["SocialCalculator"] = None,
    config_engine=None,
) -> UserProtectionManager:
    """获取用户保护管理器单例"""
    global _user_protection_manager
    if _user_protection_manager is None:
        _user_protection_manager = UserProtectionManager(
            social_calculator, config_engine
        )
    return _user_protection_manager

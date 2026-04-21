import time
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple
from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine

logger = get_logger("全局防护")


@dataclass
class GlobalShieldState:
    """全局护盾状态"""

    is_active: bool = False
    shield_until: float = 0.0
    reason: str = ""
    global_irritation: float = 0.0
    last_impact_time: float = 0.0
    activation_count: int = 0


class GlobalShieldManager:
    """全局护盾管理器"""

    _instance: Optional["GlobalShieldManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_engine=None):
        if self._initialized:
            new_config = config_engine or self._config or get_default_config_engine()
            if new_config is not self._config:
                self._config = new_config
                self._load_config()
            return
        self._config = config_engine or get_default_config_engine()
        self._state = GlobalShieldState()
        self._default_duration = 300.0
        self._max_duration = 1800.0
        self._irritation_threshold = 150.0
        self._irritation_decay_rate = 0.1
        self._auto_shield_count_threshold = 5
        self._auto_shield_severity_threshold = 3.0
        self._load_config()
        self._initialized = True
        logger.info("全局护盾管理器初始化完成")

    def _load_config(self) -> None:
        """从配置引擎加载参数"""
        self._default_duration = self._config.get(
            "global_shield", "default_duration", 300.0
        )
        self._max_duration = self._config.get(
            "global_shield", "max_duration", 1800.0
        )
        self._irritation_threshold = self._config.get(
            "global_shield", "irritation_threshold", 150.0
        )
        self._irritation_decay_rate = self._config.get(
            "global_shield", "irritation_decay_rate", 0.1
        )
        self._auto_shield_count_threshold = self._config.get(
            "global_shield", "auto_shield_count_threshold", 5
        )
        self._auto_shield_severity_threshold = self._config.get(
            "global_shield", "auto_shield_severity_threshold", 3.0
        )

    def activate_shield(
        self, duration: Optional[float] = None, reason: str = ""
    ) -> bool:
        """激活全局护盾"""
        now = time.time()
        if duration is None:
            duration = self._default_duration
        duration = min(duration, self._max_duration)
        self._state.is_active = True
        self._state.shield_until = now + duration
        self._state.reason = reason
        self._state.activation_count += 1
        logger.critical(
            f"全局护盾激活 | 原因:{reason} | 持续:{duration:.0f}秒"
        )
        return True

    def is_active(self) -> Tuple[bool, str]:
        """检查全局护盾是否激活"""
        if not self._state.is_active:
            return False, ""
        now = time.time()
        if now >= self._state.shield_until:
            self._state.is_active = False
            self._state.shield_until = 0.0
            self._state.reason = ""
            logger.info("全局护盾自动解除")
            return False, ""
        remaining = int(self._state.shield_until - now)
        return (
            True,
            f"系统保护中，还剩{remaining}秒。原因: {self._state.reason}",
        )

    def deactivate_shield(self) -> bool:
        """解除全局护盾"""
        if self._state.is_active:
            self._state.is_active = False
            self._state.shield_until = 0.0
            self._state.reason = ""
            logger.info("全局护盾手动解除")
            return True
        return False

    def get_remaining_time(self) -> float:
        """获取护盾剩余时间"""
        if not self._state.is_active:
            return 0.0
        now = time.time()
        if now >= self._state.shield_until:
            return 0.0
        return self._state.shield_until - now

    def add_irritation(self, amount: float, source: str = "") -> None:
        """增加全局激怒值"""
        now = time.time()
        self._decay_irritation(now)
        self._state.global_irritation += amount
        self._state.last_impact_time = now
        logger.warning(
            f"全局激怒值增加 | +{amount:.1f} | 当前:{self._state.global_irritation:.1f} | 来源:{source}"
        )
        if self._state.global_irritation >= self._irritation_threshold:
            self.activate_shield(
                duration=self._default_duration,
                reason=f"全局激怒值过高 ({
                    self._state.global_irritation:.0f})",
            )
            self._state.global_irritation = 0.0

    def _decay_irritation(self, now: float) -> None:
        """衰减激怒值"""
        if self._state.last_impact_time > 0:
            hours = (now - self._state.last_impact_time) / 3600.0
            if hours > 0:
                decay = (
                    self._irritation_decay_rate
                    * hours
                    * self._state.global_irritation
                )
                self._state.global_irritation = max(
                    0.0, self._state.global_irritation - decay
                )

    def get_irritation_level(self) -> float:
        """获取当前激怒值"""
        now = time.time()
        self._decay_irritation(now)
        return self._state.global_irritation

    def reset_irritation(self) -> None:
        """重置激怒值"""
        self._state.global_irritation = 0.0
        self._state.last_impact_time = 0.0
        logger.info("全局激怒值已重置")

    def check_auto_shield(
        self, harassment_count: int, harassment_severity_sum: float
    ) -> bool:
        """检查是否需要自动护盾"""
        is_active, _ = self.is_active()
        if is_active:
            return False
        if (
            harassment_count >= self._auto_shield_count_threshold
            and harassment_severity_sum >= self._auto_shield_severity_threshold
        ):
            self.activate_shield(
                duration=self._default_duration,
                reason=f"频繁骚扰自动保护 (次数:{harassment_count})",
            )
            return True
        return False

    def export_state(self) -> Dict[str, Any]:
        """导出状态（用于持久化）"""
        return {
            "is_active": self._state.is_active,
            "shield_until": self._state.shield_until,
            "reason": self._state.reason,
            "global_irritation": self._state.global_irritation,
            "last_impact_time": self._state.last_impact_time,
            "activation_count": self._state.activation_count,
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """导入状态（用于恢复）"""
        self._state = GlobalShieldState(
            is_active=state.get("is_active", False),
            shield_until=state.get("shield_until", 0.0),
            reason=state.get("reason", ""),
            global_irritation=state.get("global_irritation", 0.0),
            last_impact_time=state.get("last_impact_time", 0.0),
            activation_count=state.get("activation_count", 0),
        )
        logger.info(f"全局护盾状态恢复完成 | 激活:{self._state.is_active}")


def get_global_shield_manager(config_engine=None) -> GlobalShieldManager:
    """获取全局护盾管理器单例"""
    return GlobalShieldManager(config_engine)

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DimensionScope(Enum):
    """维度作用域枚举"""
    PER_USER_CHANNEL = "per_user_channel"   # 每用户×每频道独立
    PER_USER = "per_user"                   # 每用户全局（跨频道共享）
    PER_CHANNEL = "per_channel"             # 每频道（所有用户共享）
    GLOBAL = "global"                       # 全局单例


@dataclass
class DimensionVote:
    """维度投票基础结构，所有具体投票类型继承此类"""
    dimension_name: str = ""
    # 概率乘数：1.0表示无影响，<1.0降低回复概率，>1.0提高概率
    probability_factor: float = 1.0
    # 是否建议强制拒绝（True时决策网关直接拒绝，不进入概率计算）
    force_refuse: bool = False
    # 是否建议强制触发（True时决策网关直接通过，不进入概率计算）
    force_trigger: bool = False
    # 态度标签（注入LLM提示词，影响回复语气）
    attitude_tag: str = ""
    # 风格修饰（注入风格系统，影响回复风格）
    style_hint: str = ""
    # 回复长度上限（0表示不限制，单位token）
    max_tokens_cap: int = 0
    # 调试信息（不参与决策，仅用于日志追踪）
    debug_reason: str = ""


@dataclass
class EventContext:
    """事件上下文，用于维度间传递消息事件信息"""
    event_type: str = ""           # 事件类型标识（message_received/reply_completed/etc）
    user_id: str = ""              # 触发事件的用户ID
    channel_id: str = ""           # 事件所在频道ID
    message_text: str = ""         # 消息文本（消息事件时有效）
    message_length: int = 0        # 消息长度
    is_at_bot: bool = False        # 是否@了机器人
    is_reply_to_bot: bool = False  # 是否回复了机器人消息
    is_admin: bool = False         # 是否管理员
    reply_tokens: int = 0          # 回复使用的token数（回复完成事件时有效）
    reply_duration_sec: float = 0  # 回复耗时秒数
    sentiment_score: float = 0.0   # 消息情感倾向分数（-1.0负面~1.0正面）
    raw_extras: dict = field(default_factory=dict)  # 额外自定义参数


@dataclass
class TickResult:
    """维度 tick 执行结果"""
    dimension_name: str = ""
    updated: bool = False          # 本次tick是否产生了状态变更
    persisted: bool = False        # 本次tick是否触发了持久化
    summary: str = ""              # 变更摘要（调试用）


class DimensionBase(ABC):
    """
    多维独立状态系统的维度基类。
    每个维度实现此接口后注册到 DimensionDispatcher，
    由调度器统一管理 tick 周期、持久化和投票收集。
    维度内部自行维护自己的状态，不依赖其他维度的计算结果。
    """

    @property
    @abstractmethod
    def dimension_name(self) -> str:
        """维度唯一标识（如 "emotion_axis", "frequency_control"）"""
        raise NotImplementedError

    @property
    @abstractmethod
    def scope(self) -> DimensionScope:
        """维度作用域"""
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_persistence(self) -> bool:
        """是否需要持久化到数据库"""
        raise NotImplementedError

    @property
    def tick_interval_sec(self) -> float:
        """tick 调用间隔（秒），默认1秒"""
        return 1.0

    @property
    def persist_interval_sec(self) -> float:
        """持久化间隔（秒），默认30秒"""
        return 30.0

    def initialize(self):
        """
        初始化维度。
        调度器启动时调用，用于加载持久化数据、初始化内部结构等。
        子类按需覆盖。
        """
        return

    @abstractmethod
    def tick(self, elapsed_sec: float) -> TickResult:
        """
        周期性更新：执行恢复/衰减/时间相关计算。
        由调度器按 tick_interval_sec 间隔调用。
        elapsed_sec: 距上次tick的实际经过秒数（可能大于tick_interval_sec）
        """
        raise NotImplementedError

    @abstractmethod
    def on_event(self, ctx: EventContext):
        """
        接收事件通知并更新内部状态。
        事件类型包括:
          - message_received: 收到用户消息
          - reply_completed: 机器人完成回复
          - reply_skipped: 机器人决定不回复
          - user_joined: 用户加入频道
          - user_left: 用户离开频道
          - bot_mentioned: 机器人被@
        """
        raise NotImplementedError

    @abstractmethod
    def vote(self, ctx: EventContext) -> DimensionVote:
        """
        为决策网关提供独立投票。
        每个维度根据自身状态给出概率乘数、态度标签等建议，
        由决策网关按乘法方式组合各维度投票（不是加权求和）。
        """
        raise NotImplementedError

    def serialize(self) -> dict:
        """
        将当前维度状态序列化为字典，用于持久化。
        仅在 needs_persistence=True 时由调度器调用。
        """
        return {}

    def deserialize(self, data: dict):
        """
        从字典恢复维度状态。
        系统启动时由调度器调用，传入上次持久化的数据。
        """
        return

    def calibrate(self, offline_seconds: float):
        """
        离线校准：系统重启后补算离线期间的衰减/恢复。
        由 D9（系统校准）协调调用。
        offline_seconds: 距离上次运行的秒数
        """
        return

    def get_state_summary(self) -> dict:
        """
        返回当前维度状态摘要（用于调试面板和日志）。
        子类按需覆盖，返回关键指标的快照。
        """
        return {"dimension": self.dimension_name, "scope": self.scope.value}

    def reset(self, user_id: str = "", channel_id: str = ""):
        """
        重置指定用户/频道的维度状态到初始值。
        用户从互动列表移除、管理员手动重置时调用。
        """
        return


class DimensionKeyBuilder:
    """
    复合键生成器：根据维度作用域和用户/频道ID生成唯一存储键。
    """

    @staticmethod
    def build(scope: DimensionScope, user_id: str = "", channel_id: str = "") -> str:
        """根据作用域组合存储键"""
        if scope == DimensionScope.PER_USER_CHANNEL:
            return f"{user_id}:{channel_id}"
        if scope == DimensionScope.PER_USER:
            return user_id
        if scope == DimensionScope.PER_CHANNEL:
            return channel_id
        return "__global__"

    @staticmethod
    def parse(scope: DimensionScope, composite_key: str) -> tuple:
        """从复合键中解析出 (user_id, channel_id)"""
        if scope == DimensionScope.PER_USER_CHANNEL:
            parts = composite_key.split(":", 1)
            if len(parts) == 2:
                return (parts[0], parts[1])
            return ("", "")
        if scope == DimensionScope.PER_USER:
            return (composite_key, "")
        if scope == DimensionScope.PER_CHANNEL:
            return ("", composite_key)
        return ("", "")


class DimensionRegistry:
    """
    维度注册表：存储所有已注册维度实例的全局索引。
    不做业务逻辑，只负责增删查。
    """

    def __init__(self):
        self._dimensions: dict[str, DimensionBase] = {}

    def register(self, dim: DimensionBase):
        """注册一个维度实例"""
        name = dim.dimension_name
        if name in self._dimensions:
            raise ValueError(f"维度 '{name}' 已注册，不允许重复注册")
        self._dimensions[name] = dim

    def unregister(self, name: str):
        """移除一个已注册的维度"""
        self._dimensions.pop(name, None)

    def get(self, name: str) -> DimensionBase | None:
        """按名称获取维度实例"""
        return self._dimensions.get(name)

    def all_dimensions(self) -> list[DimensionBase]:
        """返回所有已注册维度的列表（注册顺序）"""
        return list(self._dimensions.values())

    def persistent_dimensions(self) -> list[DimensionBase]:
        """返回所有需要持久化的维度"""
        return [d for d in self._dimensions.values() if d.needs_persistence]

    def dimension_names(self) -> list[str]:
        """返回所有已注册维度的名称列表"""
        return list(self._dimensions.keys())

    @property
    def count(self) -> int:
        return len(self._dimensions)

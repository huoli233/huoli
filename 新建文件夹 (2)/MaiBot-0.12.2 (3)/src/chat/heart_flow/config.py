from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class HeartFlowMode(str, Enum):
    NORMAL = "normal"
    FOCUS = "focus"

    @classmethod
    def from_str(cls, value: str) -> "HeartFlowMode":
        value = value.lower().strip()
        if value == "focus":
            return cls.FOCUS
        return cls.NORMAL


class HeartFlowState(str, Enum):
    """心流运行状态 - 描述机器人在当前会话中的行为模式"""
    OBSERVING = "读空气中"
    PEEKING = "默默读空气"
    SLACKING = "潜水休眠"

    @classmethod
    def from_str(cls, value: str) -> "HeartFlowState":
        value = value.lower().strip()
        if value in ("默默读空气", "窥屏", "peek", "peeking"):
            return cls.PEEKING
        if value in ("潜水休眠", "潜水", "摸鱼", "slack", "slacking", "休息"):
            return cls.SLACKING
        return cls.OBSERVING


@dataclass
class NormalChatConfig:
    # 概率回复模式参数
    base_reply_probability: float = 0.03
    at_boost: float = 1.0
    wake_word_boost: float = 1.0
    topic_interest_boost: float = 0.2
    fatigue_decay: float = 0.1
    fatigue_penalty: float = 0.8
    max_fatigue: float = 1.0
    no_reply_threshold_3: int = 2
    no_reply_threshold_5: int = 3
    admin_boost: float = 2.0


@dataclass
class FocusChatConfig:
    enabled_stream_types: List[str] = field(default_factory=lambda: ["private"])
    max_wait_seconds: int = 300
    min_wait_seconds: int = 30
    max_chase_count: int = 2
    silence_threshold_hours: float = 2.0
    quiet_hours_start: str = "23:00"
    quiet_hours_end: str = "07:00"
    trigger_probability: float = 0.3
    brain_power_max: float = 100.0
    brain_cost_per_think: float = 3.0
    brain_cost_per_reply: float = 8.0
    brain_recovery_rate: float = 0.005
    chat_value_max: float = 100.0
    chat_cost_base: float = 15.0
    chat_cost_peek: float = 5.0
    chat_recovery_rate: float = 0.002
    chat_rest_recovery_rate: float = 1.5
    peek_interval_min: float = 8.0
    peek_interval_max: float = 15.0
    consecutive_penalty_factor: float = 0.4
    reply_desire_threshold: int = 7


@dataclass
class ProactiveConfig:
    # 主动回复参数
    enabled: bool = False
    thinking_interval: int = 120
    silence_threshold: int = 1800
    probability: float = 0.1
    min_interval: int = 600
    daily_max: int = 9999
    quiet_start: int = 23
    quiet_end: int = 7


@dataclass
class FatigueConfig:
    # 疲劳系统参数
    threshold_min: int = 3
    threshold_max: int = 8
    per_reply: float = 30.0
    accumulated_threshold: float = 120.0
    decay_rate: float = 1.0
    max_duration: float = 180.0


@dataclass
class StaminaConfig:
    # 耐力系统参数
    cost_reply: float = 5.0
    cost_peek: float = 2.0
    recovery_rate: float = 0.05
    low_threshold: float = 20.0
    critical_threshold: float = 5.0


@dataclass
class PsychologyConfig:
    # 心理系统参数
    affection_enabled: bool = True
    trauma_enabled: bool = True
    emotion_tracking_enabled: bool = True
    affection_increase_rate: float = 0.1
    affection_decrease_rate: float = 0.05
    trauma_recovery_rate: float = 0.01


@dataclass
class ModelDrivenMonitorConfig:
    """模型驱动监控系统配置"""
    enabled: bool = True
    """是否启用模型驱动监控（永久运行的独立循环）"""
    
    check_interval: float = 5.0
    """检查间隔（秒），每隔多久检查一次新消息"""
    
    auto_restart: bool = True
    """崩溃后是否自动重启"""
    
    restart_delay: float = 3.0
    """重启延迟（秒）"""
    
    max_restart_attempts: int = 5
    """最大重启尝试次数（0表示无限重启）"""
    
    model_name: str = "qwen3-0.6b"
    """小模型名称，从model_config读取"""
    
    model_task: str = "model_monitor"
    """模型任务类型，用于从model_config获取模型配置"""
    
    temperature: float = 0.7
    """模型温度"""
    
    max_duration_seconds: int = 600
    """最大状态持续时长（秒），10分钟"""
    
    min_duration_seconds: int = 60
    """最小状态持续时长（秒），1分钟"""
    
    max_messages_for_long_duration: int = 100
    """超过最大时长时限制的最大消息数"""
    
    save_state_interval: float = 30.0
    """状态保存间隔（秒），定期保存到数据库"""
    
    decision_prompt_template: str = ""
    """决策提示词模板，为空则使用默认模板"""
    
    thought_prompt_template: str = ""
    """内心独白提示词模板，为空则使用默认模板"""
    
    summary_prompt_template: str = ""
    """总结提示词模板，为空则使用默认模板"""


@dataclass
class HeartFlowCoreConfig:
    # 心流系统总配置
    enabled: bool = False
    threshold: int = 50
    mode: HeartFlowMode = HeartFlowMode.NORMAL
    normal_chat: NormalChatConfig = field(default_factory=NormalChatConfig)
    focus_chat: FocusChatConfig = field(default_factory=FocusChatConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    fatigue: FatigueConfig = field(default_factory=FatigueConfig)
    stamina: StaminaConfig = field(default_factory=StaminaConfig)
    psychology: PsychologyConfig = field(default_factory=PsychologyConfig)
    model_driven_monitor: ModelDrivenMonitorConfig = field(default_factory=ModelDrivenMonitorConfig)
    admin_always_reply: bool = True


_config: Optional[HeartFlowCoreConfig] = None


def get_heartflow_config() -> HeartFlowCoreConfig:
    global _config
    if _config is None:
        _config = load_heartflow_config()
    return _config


def load_heartflow_config() -> HeartFlowCoreConfig:
    config = HeartFlowCoreConfig()
    try:
        from src.config.config import global_config
        config.enabled = getattr(global_config.chat, 'enable_heartflow', False)
        config.threshold = getattr(global_config.chat, 'heartflow_threshold', 50)
        mode_str = getattr(global_config.chat, 'heartflow_mode', 'normal')
        config.mode = HeartFlowMode.from_str(str(mode_str))
        config.normal_chat.base_reply_probability = getattr(
            global_config.chat, 'base_reply_probability', 0.03
        )
        config.admin_always_reply = getattr(
            global_config.chat, 'at_auto_reply', True
        )
    except Exception:
        pass
    try:
        from src.config.behavior_tuner import get_behavior_tuner
        tuner = get_behavior_tuner()
        config.normal_chat.base_reply_probability = tuner.get("base_reply_probability", config.normal_chat.base_reply_probability)
        config.normal_chat.at_boost = tuner.get("at_boost", config.normal_chat.at_boost)
        config.normal_chat.wake_word_boost = tuner.get("wake_word_boost", config.normal_chat.wake_word_boost)
        config.normal_chat.admin_boost = tuner.get("admin_boost", config.normal_chat.admin_boost)
        config.normal_chat.fatigue_penalty = tuner.get("consecutive_penalty_factor", config.normal_chat.fatigue_penalty)
        config.focus_chat.peek_interval_min = tuner.get("peek_interval_min", config.focus_chat.peek_interval_min)
        config.focus_chat.peek_interval_max = tuner.get("peek_interval_max", config.focus_chat.peek_interval_max)
        config.focus_chat.chat_value_max = tuner.get("chat_value_max", config.focus_chat.chat_value_max)
        config.focus_chat.brain_power_max = tuner.get("brain_power_max", config.focus_chat.brain_power_max)
        config.focus_chat.chat_cost_base = tuner.get("chat_cost_base", config.focus_chat.chat_cost_base)
        config.focus_chat.chat_recovery_rate = tuner.get("chat_recovery_idle", config.focus_chat.chat_recovery_rate)
        config.focus_chat.chat_rest_recovery_rate = tuner.get("chat_recovery_rest", config.focus_chat.chat_rest_recovery_rate)
        config.focus_chat.brain_recovery_rate = tuner.get("brain_recovery_rate", config.focus_chat.brain_recovery_rate)
        config.focus_chat.consecutive_penalty_factor = tuner.get("consecutive_penalty_factor", config.focus_chat.consecutive_penalty_factor)
        config.fatigue.threshold_min = tuner.get("fatigue_threshold_min", config.fatigue.threshold_min)
        config.fatigue.threshold_max = tuner.get("fatigue_threshold_max", config.fatigue.threshold_max)
        config.fatigue.per_reply = tuner.get("fatigue_per_reply", config.fatigue.per_reply)
        config.fatigue.accumulated_threshold = tuner.get("fatigue_accumulated_threshold", config.fatigue.accumulated_threshold)
        config.fatigue.max_duration = tuner.get("fatigue_max_duration", config.fatigue.max_duration)
        config.fatigue.decay_rate = tuner.get("fatigue_decay_rate", config.fatigue.decay_rate)
        config.stamina.recovery_rate = tuner.get("brain_recovery_rate", config.stamina.recovery_rate)
        config.stamina.cost_reply = tuner.get("brain_cost_reply", config.stamina.cost_reply)
        config.proactive.probability = tuner.get("proactive_probability", config.proactive.probability)
        config.proactive.enabled = tuner.get("proactive_enabled", config.proactive.enabled)
        config.proactive.min_interval = tuner.get("proactive_min_interval", config.proactive.min_interval)
        config.proactive.daily_max = tuner.get("proactive_daily_max", config.proactive.daily_max)
        config.psychology.affection_increase_rate = tuner.get("affection_increase_rate", config.psychology.affection_increase_rate)
        config.psychology.affection_decrease_rate = tuner.get("affection_decrease_rate", config.psychology.affection_decrease_rate)
        config.psychology.trauma_recovery_rate = tuner.get("trauma_recovery_rate", config.psychology.trauma_recovery_rate)
    except Exception:
        pass
    return config


def reload_heartflow_config() -> HeartFlowCoreConfig:
    global _config
    _config = load_heartflow_config()
    return _config


def is_focus_mode_for_stream(stream_type: str) -> bool:
    config = get_heartflow_config()
    if config.mode == HeartFlowMode.FOCUS:
        return True
    if stream_type in config.focus_chat.enabled_stream_types:
        return True
    return False


def get_fatigue_config() -> FatigueConfig:
    return get_heartflow_config().fatigue


def get_proactive_config() -> ProactiveConfig:
    return get_heartflow_config().proactive


def get_stamina_config() -> StaminaConfig:
    return get_heartflow_config().stamina


def get_psychology_config() -> PsychologyConfig:
    return get_heartflow_config().psychology


def get_model_driven_monitor_config() -> ModelDrivenMonitorConfig:
    """获取模型驱动监控配置"""
    return get_heartflow_config().model_driven_monitor

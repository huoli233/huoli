from dataclasses import dataclass, field
from typing import Dict, Optional
import time


@dataclass
class SocialValueRecord:
    """社交值持久化记录"""

    user_id: str
    channel_id: str
    value: float = 0.0
    positive_dim: float = 0.0
    negative_dim: float = 0.0
    trust_value: float = 0.0
    annoyance_value: float = 0.0
    interaction_count: int = 0
    last_interaction: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class SocialUpdateResult:
    """社交值更新结果"""

    old_value: float
    new_value: float
    delta: float
    behavior_type: str
    intent: str = ""
    severity: float = 0.0
    category: str = ""
    params_used: Optional[Dict] = None


@dataclass
class AlgorithmParams:
    """最终算法参数 - 四层映射的输出

    由 social_calculator 配置画像生成，包含计算社交值变化所需的全部参数。
    """

    category: str
    base_score: float
    intent_multiplier: float
    max_change_per_reply: float
    acceleration_threshold: float
    acceleration_multiplier: float
    decay_rate: float
    trauma_impact: float
    type_bonus: float = 1.0


@dataclass
class CategoryConfig:
    """分类配置 - 第二层运行时参数

    5个分类各一套配置：friendly, neutral, unfriendly, hostile, harassing
    由 social_calculator 配置画像构建并应用人格因子调整。
    """

    base_score: float = 0.0
    max_change_per_reply: float = 0.02
    acceleration_threshold: float = 0.8
    acceleration_multiplier: float = 2.0
    decay_rate: float = 0.01
    trauma_impact: float = 0.0
    intent_multipliers: Dict[str, float] = field(default_factory=dict)

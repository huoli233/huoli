from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger

logger = get_logger("记忆模型")


@dataclass
class BufferedMemory:
    """记忆缓冲区条目模型。

    字段映射说明：
    - importance: 记忆重要性（0~1），向后兼容保留
    - significance: importance 的标准别名，与数据库 MemoryRecord.significance 对齐
      两者指向同一概念（记忆重要性），统一使用 significance 作为标准字段名
    """

    memory_id: str
    content: str
    created_at: float
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = 0.0
    compressed_content: str = ""
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    channel_id: str = ""
    user_id: str = ""
    tier: Any = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

    @property
    def significance(self) -> float:
        """记忆重要性（importance 的标准别名，与 DB 字段 significance 对齐）。"""
        return self.importance

    @significance.setter
    def significance(self, value: float) -> None:
        self.importance = value


class UnifiedMemoryTier(Enum):
    """统一记忆层级 —— 三套层级系统的公共基准。

    映射关系：
    ┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
    │ UnifiedTier      │ RetentionLayer   │ CapacityTier     │ DedupTier        │
    ├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ SENSORY          │ SENSORY(0)       │ IMMEDIATE        │ SHORT_TERM       │
    │ WORKING          │ WORKING(1)       │ WORKING          │ SHORT_TERM       │
    │ SHORT_TERM       │ SHORT_TERM(2)    │ RECENT           │ LONG_TERM        │
    │ MID_TERM         │ MID_TERM(3)      │ MIDTERM          │ LONG_TERM        │
    │ CONSOLIDATED     │ CONSOLIDATED(4)  │ LONGTERM         │ ARCHIVE          │
    └──────────────────┴──────────────────┴──────────────────┴──────────────────┘
    """

    SENSORY = "sensory"
    WORKING = "working"
    SHORT_TERM = "short_term"
    MID_TERM = "mid_term"
    CONSOLIDATED = "consolidated"


_RETENTION_TO_UNIFIED = {
    0: UnifiedMemoryTier.SENSORY,
    1: UnifiedMemoryTier.WORKING,
    2: UnifiedMemoryTier.SHORT_TERM,
    3: UnifiedMemoryTier.MID_TERM,
    4: UnifiedMemoryTier.CONSOLIDATED,
}

_CAPACITY_TO_UNIFIED: Dict[str, UnifiedMemoryTier] = {
    "immediate": UnifiedMemoryTier.SENSORY,
    "working": UnifiedMemoryTier.WORKING,
    "recent": UnifiedMemoryTier.SHORT_TERM,
    "midterm": UnifiedMemoryTier.MID_TERM,
    "longterm": UnifiedMemoryTier.CONSOLIDATED,
}

_DEDUP_TO_UNIFIED: Dict[str, UnifiedMemoryTier] = {
    "short_term": UnifiedMemoryTier.WORKING,
    "long_term": UnifiedMemoryTier.MID_TERM,
    "archive": UnifiedMemoryTier.CONSOLIDATED,
}

_UNIFIED_TO_CAPACITY: Dict[UnifiedMemoryTier, str] = {
    UnifiedMemoryTier.SENSORY: "immediate",
    UnifiedMemoryTier.WORKING: "working",
    UnifiedMemoryTier.SHORT_TERM: "recent",
    UnifiedMemoryTier.MID_TERM: "midterm",
    UnifiedMemoryTier.CONSOLIDATED: "longterm",
}

_UNIFIED_TO_DEDUP: Dict[UnifiedMemoryTier, str] = {
    UnifiedMemoryTier.SENSORY: "short_term",
    UnifiedMemoryTier.WORKING: "short_term",
    UnifiedMemoryTier.SHORT_TERM: "long_term",
    UnifiedMemoryTier.MID_TERM: "long_term",
    UnifiedMemoryTier.CONSOLIDATED: "archive",
}


def retention_to_capacity_tier(retention_layer_index: int):
    """将海马体RetentionLayer层级索引转换为CapacityGovernor的MemoryTier枚举。"""
    unified = _RETENTION_TO_UNIFIED.get(retention_layer_index)
    if unified is None:
        return None
    capacity_name = _UNIFIED_TO_CAPACITY.get(unified)
    if capacity_name is None:
        return None
    try:
        from src.memory_system.capacity_governor import MemoryTier
        return MemoryTier(capacity_name)
    except Exception as exc:
        logger.warning(f"capacity层级转换失败 ({capacity_name}): {exc}")
        return None


def retention_to_dedup_tier(retention_layer_index: int):
    """将海马体RetentionLayer层级索引转换为DedupFuser的MemoryTier枚举。"""
    unified = _RETENTION_TO_UNIFIED.get(retention_layer_index)
    if unified is None:
        return None
    dedup_name = _UNIFIED_TO_DEDUP.get(unified)
    if dedup_name is None:
        return None
    try:
        from src.memory_system.dedup_fuser import MemoryTier
        return MemoryTier(dedup_name)
    except Exception as exc:
        logger.warning(f"dedup层级转换失败 ({dedup_name}): {exc}")
        return None


def capacity_to_retention_index(capacity_tier_name: str) -> Optional[int]:
    """将CapacityGovernor的MemoryTier名称转换为RetentionLayer索引。"""
    unified = _CAPACITY_TO_UNIFIED.get(capacity_tier_name)
    if unified is None:
        return None
    for idx, ut in _RETENTION_TO_UNIFIED.items():
        if ut == unified:
            return idx
    return None


def dedup_to_retention_index(dedup_tier_name: str) -> Optional[int]:
    """将DedupFuser的MemoryTier名称转换为RetentionLayer索引（取该分组的最低层）。"""
    unified = _DEDUP_TO_UNIFIED.get(dedup_tier_name)
    if unified is None:
        return None
    tier_order = list(UnifiedMemoryTier)
    # 防御性检查：避免index()方法抛出ValueError
    try:
        idx = tier_order.index(unified)
        return idx
    except ValueError:
        logger.warning(f"无法在UnifiedMemoryTier中找到对应的层级: {unified}")
        return None


def unify_tier(tier_obj: Any) -> Optional[UnifiedMemoryTier]:
    """将任意层级对象统一转换为UnifiedMemoryTier。"""
    if isinstance(tier_obj, UnifiedMemoryTier):
        return tier_obj
    if isinstance(tier_obj, int):
        return _RETENTION_TO_UNIFIED.get(tier_obj)
    if hasattr(tier_obj, "value"):
        val = str(tier_obj.value)
        if val in _CAPACITY_TO_UNIFIED:
            return _CAPACITY_TO_UNIFIED[val]
        if val in _DEDUP_TO_UNIFIED:
            return _DEDUP_TO_UNIFIED[val]
    if isinstance(tier_obj, str):
        if tier_obj in _CAPACITY_TO_UNIFIED:
            return _CAPACITY_TO_UNIFIED[tier_obj]
        if tier_obj in _DEDUP_TO_UNIFIED:
            return _DEDUP_TO_UNIFIED[tier_obj]
    return None

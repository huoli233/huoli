import time
from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("EngineOrchestrator")

# 缓存条目数据结构
@dataclass
class _CacheEntry:
    value: Any
    expires_at: float
    group: str

class EngineOrchestrator:
    """统一引擎编排器
    将 30+ 个分散的单例引擎整合为分组管理,
    提供统一缓存层（TTL + 分组失效）和懒加载引用。
    设计目标：非侵入式，可逐步迁移，不影响现有代码。
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._engine_refs: Dict[str, Any] = {}
        self._cache: Dict[str, _CacheEntry] = {}
        self._default_ttl: float = 5.0
        self._group_ttls: Dict[str, float] = {
            "perception": 5.0,
            "energy": 3.0,
            "social": 10.0,
            "decision": 2.0,
            "memory": 15.0,
            "scene": 8.0,
        }
        # 引擎名 → (获取函数, 分组名) 懒加载注册表
        self._registry: Dict[str, tuple] = {}
        self._register_builtin_engines()

    def _register_builtin_engines(self) -> None:
        """注册全部内建引擎的懒加载获取函数"""
        # 能量组
        self.register("energy_chain", self._load_energy_chain, "energy")
        self.register("metabolism", self._load_metabolism, "energy")
        self.register("night_cycle", self._load_night_cycle, "energy")
        # 感知组
        self.register("watch_machine", self._load_watch_machine, "perception")
        self.register("deep_visibility", self._load_deep_visibility, "perception")
        self.register("scene_state", self._load_scene_state, "scene")
        self.register("pattern_detector", self._load_pattern_detector, "perception")
        self.register("semantic_router", self._load_semantic_router, "perception")
        self.register("visibility_gate", self._load_visibility_gate, "perception")
        self.register("understanding_gate", self._load_understanding_gate, "perception")
        # 社交组
        self.register("impression_hub", self._load_impression_hub, "social")
        self.register("impression_engine", self._load_impression_engine, "social")
        self.register("skill_hub", self._load_skill_hub, "social")
        self.register("participant_roster", self._load_participant_roster, "social")
        self.register("learning_hub", self._load_learning_hub, "social")
        # 决策组
        self.register("unified_planner", self._load_unified_planner, "decision")
        self.register("narration_planner", self._load_narration_planner, "decision")
        self.register("decision_engine", self._load_decision_engine, "decision")
        self.register("state_coupling", self._load_state_coupling, "decision")
        # 记忆组
        self.register("memory_governance", self._load_memory_governance, "memory")
        self.register("memory_reactivation", self._load_memory_reactivation, "memory")
        self.register("freshness_decay", self._load_freshness_decay, "memory")
        self.register("subjective_time", self._load_subjective_time, "memory")
        # 安全组
        self.register("safety_fusion", self._load_safety_fusion, "decision")
        self.register("cross_validator", self._load_cross_validator, "decision")
        # 注意力与情感
        self.register("attention_flow", self._load_attention_flow, "perception")
        self.register("emotion_feedback", self._load_emotion_feedback, "social")
        # 辅助工具
        self.register("multimodal_budget", self._load_multimodal_budgeter, "perception")
        self.register("self_reference", self._load_self_reference_detector, "perception")
        self.register("panel_adapter", self._load_panel_adapter, "perception")
        self.register("state_dashboard", self._load_state_dashboard, "perception")

    def register(self, name: str, loader: Callable, group: str) -> None:
        """注册一个引擎的懒加载函数"""
        self._registry[name] = (loader, group)

    def get(self, name: str) -> Any:
        """获取引擎实例（懒加载，首次调用时初始化）"""
        if name in self._engine_refs:
            return self._engine_refs[name]
        entry = self._registry.get(name)
        if entry is None:
            logger.warning(f"未注册的引擎: {name}")
            return None
        loader, _group = entry
        try:
            instance = loader()
            self._engine_refs[name] = instance
            return instance
        except Exception as exc:
            logger.debug(f"加载引擎 {name} 失败: {exc}")
            return None

    # ── 统一缓存层 ──

    def cache_get(self, key: str) -> Any:
        """获取缓存值，过期返回 None"""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() > entry.expires_at:
            del self._cache[key]
            return None
        return entry.value

    def cache_set(self, key: str, value: Any, group: str = "default", ttl: float = 0.0) -> None:
        """设置缓存值，ttl=0 时使用分组默认 TTL"""
        actual_ttl = ttl if ttl > 0 else self._group_ttls.get(group, self._default_ttl)
        self._cache[key] = _CacheEntry(
            value=value,
            expires_at=time.time() + actual_ttl,
            group=group,
        )

    def invalidate_group(self, group: str) -> int:
        """使某个分组的所有缓存立即失效"""
        keys_to_remove = [k for k, v in self._cache.items() if v.group == group]
        for k in keys_to_remove:
            del self._cache[k]
        return len(keys_to_remove)

    def invalidate_all(self) -> None:
        """清空全部缓存"""
        self._cache.clear()

    def cache_stats(self) -> Dict[str, int]:
        """返回各分组的缓存条目数"""
        stats: Dict[str, int] = {}
        now = time.time()
        for entry in self._cache.values():
            if now <= entry.expires_at:
                stats[entry.group] = stats.get(entry.group, 0) + 1
        return stats

    # ── 引擎加载函数 ──

    def _load_energy_chain(self) -> Any:
        from src.chat.heart_flow.energy_manager import EnergyChainDimension
        return EnergyChainDimension.get_instance()

    def _load_metabolism(self) -> Any:
        from src.core.metabolism_engine import get_metabolism_engine
        return get_metabolism_engine(self._channel_id)

    def _load_night_cycle(self) -> Any:
        from src.core.night_cycle_system import get_night_cycle
        return get_night_cycle(self._channel_id)

    def _load_watch_machine(self) -> Any:
        from src.core.watch_state_machine import get_watch_machine
        return get_watch_machine(self._channel_id)

    def _load_deep_visibility(self) -> Any:
        from src.core.deep_visibility_scorer import get_deep_visibility_scorer
        return get_deep_visibility_scorer(self._channel_id)

    def _load_scene_state(self) -> Any:
        from src.core.group_scene_state import get_group_scene
        return get_group_scene(self._channel_id)

    def _load_pattern_detector(self) -> Any:
        from src.core.group_pattern_detector import get_group_pattern_detector
        return get_group_pattern_detector(self._channel_id)

    def _load_semantic_router(self) -> Any:
        from src.core.message_semantic_router import get_semantic_router
        return get_semantic_router()

    def _load_visibility_gate(self) -> Any:
        from src.core.visibility_gate import get_visibility_gate
        return get_visibility_gate(self._channel_id)

    def _load_understanding_gate(self) -> Any:
        from src.core.understanding_gate import get_understanding_gate
        return get_understanding_gate()

    def _load_impression_hub(self) -> Any:
        from src.core.impression_evolution_hub import get_impression_hub
        return get_impression_hub(self._channel_id)

    def _load_impression_engine(self) -> Any:
        # 已迁移至 impression_evolution_hub，委托到 hub
        from src.core.impression_evolution_hub import get_impression_hub
        return get_impression_hub(self._channel_id)

    def _load_skill_hub(self) -> Any:
        from src.core.skill_lifecycle_hub import get_skill_lifecycle_hub
        return get_skill_lifecycle_hub(self._channel_id)

    def _load_participant_roster(self) -> Any:
        from src.core.active_participant_roster import get_participant_roster
        return get_participant_roster(self._channel_id)

    def _load_learning_hub(self) -> Any:
        from src.core.learning_hub import get_learning_hub
        return get_learning_hub(self._channel_id)

    def _load_unified_planner(self) -> Any:
        from src.core.unified_planner import get_unified_planner
        return get_unified_planner()

    def _load_narration_planner(self) -> Any:
        from src.core.inner_narration_planner import get_narration_planner
        return get_narration_planner(self._channel_id)

    def _load_decision_engine(self) -> Any:
        from src.core.multi_factor_decision_engine import get_multi_factor_engine
        return get_multi_factor_engine()

    def _load_state_coupling(self) -> Any:
        from src.core.state_coupling_matrix import get_coupling_engine
        return get_coupling_engine()

    def _load_memory_governance(self) -> Any:
        from src.core.memory_governance_engine import get_memory_governance_engine
        return get_memory_governance_engine(self._channel_id)

    def _load_memory_reactivation(self) -> Any:
        from src.core.memory_reactivation_gate import get_reactivation_gate
        return get_reactivation_gate(self._channel_id)

    def _load_freshness_decay(self) -> Any:
        from src.core.freshness_decay_engine import get_freshness_decay_engine
        return get_freshness_decay_engine(self._channel_id)

    def _load_subjective_time(self) -> Any:
        from src.core.subjective_time_stream import get_time_stream
        return get_time_stream(self._channel_id)

    def _load_safety_fusion(self) -> Any:
        from src.core.safety_boundary_fusion import get_safety_fusion_engine
        return get_safety_fusion_engine()

    def _load_cross_validator(self) -> Any:
        from src.core.cross_engine_validator import get_cross_engine_validator
        return get_cross_engine_validator()

    def _load_attention_flow(self) -> Any:
        from src.core.subjective_attention_flow import get_attention_flow_controller
        return get_attention_flow_controller(self._channel_id)

    def _load_emotion_feedback(self) -> Any:
        from src.core.emotion_feedback_loop import get_emotion_feedback_loop
        return get_emotion_feedback_loop()

    def _load_multimodal_budgeter(self) -> Any:
        from src.core.multimodal_budgeter import get_multimodal_budgeter
        return get_multimodal_budgeter()

    def _load_self_reference_detector(self) -> Any:
        from src.core.self_reference_detector import get_self_reference_detector
        return get_self_reference_detector()

    def _load_panel_adapter(self) -> Any:
        from src.core.panel_state_adapter import get_panel_adapter
        return get_panel_adapter()

    def _load_state_dashboard(self) -> Any:
        from src.core.state_dashboard import get_state_dashboard
        return get_state_dashboard(self._channel_id)


# 频道级实例管理
_orchestrators: Dict[str, EngineOrchestrator] = {}

def get_engine_orchestrator(channel_id: str) -> EngineOrchestrator:
    """获取或创建频道级引擎编排器"""
    if channel_id not in _orchestrators:
        _orchestrators[channel_id] = EngineOrchestrator(channel_id)
    return _orchestrators[channel_id]

def remove_engine_orchestrator(channel_id: str) -> None:
    """移除频道级编排器（频道关闭时调用）"""
    orch = _orchestrators.pop(channel_id, None)
    if orch:
        orch.invalidate_all()
        orch._engine_refs.clear()

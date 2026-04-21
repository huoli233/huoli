import time
import random
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from src.common.logger import get_logger

logger = get_logger("memory_overload")


@dataclass
class MemoryOverloadState:
    total_memories: int = 0
    total_size_mb: float = 0.0
    overload_level: float = 0.0
    saturation_effects: Optional[Dict[str, float]] = None
    forgetting_pressure: float = 0.0
    last_cleanup: float = 0.0

    def __post_init__(self):
        if self.saturation_effects is None:
            self.saturation_effects = {}


class MemoryOverloadProtector:
    CAPACITY_THRESHOLDS = {
        "normal": {"memories": 5000, "size_mb": 50},
        "stressed": {"memories": 8000, "size_mb": 80},
        "overloaded": {"memories": 12000, "size_mb": 120},
        "critical": {"memories": 15000, "size_mb": 150}
    }
    OVERLOAD_EFFECTS = {
        "memory_creation": 0.7,
        "memory_recall": 0.8,
        "learning_ability": 0.6,
        "focus_span": 0.5,
        "confusion_rate": 0.3
    }
    FORGETTING_WEIGHTS = {
        "age": 0.3,
        "access_frequency": 0.4,
        "importance": 0.2,
        "redundancy": 0.1
    }

    def __init__(self, stream_id: str = "default"):
        self.stream_id = stream_id
        self.current_state = MemoryOverloadState()
        self.overload_history: List[Tuple[float, float]] = []
        self.forced_forgetting_count = 0
        self.adaptive_threshold_modifier = 1.0

    def _get_memory_model(self):
        try:
            from src.common.database.database_model import MemoryEntry
            return MemoryEntry
        except Exception:
            return None

    def assess_memory_overload(self) -> MemoryOverloadState:
        try:
            MemoryEntry = self._get_memory_model()
            if not MemoryEntry:
                return self.current_state
            query = MemoryEntry.select().where(MemoryEntry.stream_id == self.stream_id)
            memories = list(query)
            total_memories = len(memories)
            total_size = sum(len(m.content.encode('utf-8')) for m in memories)
            total_size_mb = total_size / (1024 * 1024)
            overload_level = self._calculate_overload_level(total_memories, total_size_mb)
            forgetting_pressure = self._calculate_forgetting_pressure(overload_level)
            self.current_state = MemoryOverloadState(
                total_memories=total_memories,
                total_size_mb=total_size_mb,
                overload_level=overload_level,
                saturation_effects=self._calculate_saturation_effects(overload_level),
                forgetting_pressure=forgetting_pressure,
                last_cleanup=time.time()
            )
            self.overload_history.append((time.time(), overload_level))
            if len(self.overload_history) > 100:
                self.overload_history.pop(0)
            self._log_overload_status()
            return self.current_state
        except Exception as e:
            logger.error(f"状态评估失败: {e}")
            return self.current_state

    def _calculate_overload_level(self, total_memories: int, total_size_mb: float) -> float:
        thresholds = self.CAPACITY_THRESHOLDS
        memory_overload = 0.0
        if total_memories >= thresholds["critical"]["memories"]:
            memory_overload = 1.0 + (total_memories - thresholds["critical"]["memories"]) / 5000
        elif total_memories >= thresholds["overloaded"]["memories"]:
            memory_overload = 0.8 + 0.2 * ((total_memories - thresholds["overloaded"]["memories"]) /
                                             (thresholds["critical"]["memories"] - thresholds["overloaded"]["memories"]))
        elif total_memories >= thresholds["stressed"]["memories"]:
            memory_overload = 0.4 + 0.4 * ((total_memories - thresholds["stressed"]["memories"]) /
                                             (thresholds["overloaded"]["memories"] - thresholds["stressed"]["memories"]))
        elif total_memories >= thresholds["normal"]["memories"]:
            memory_overload = 0.2 * ((total_memories - thresholds["normal"]["memories"]) /
                                      (thresholds["stressed"]["memories"] - thresholds["normal"]["memories"]))
        size_overload = 0.0
        if total_size_mb >= thresholds["critical"]["size_mb"]:
            size_overload = 1.0 + (total_size_mb - thresholds["critical"]["size_mb"]) / 50
        elif total_size_mb >= thresholds["overloaded"]["size_mb"]:
            size_overload = 0.8 + 0.2 * ((total_size_mb - thresholds["overloaded"]["size_mb"]) /
                                           (thresholds["critical"]["size_mb"] - thresholds["overloaded"]["size_mb"]))
        elif total_size_mb >= thresholds["stressed"]["size_mb"]:
            size_overload = 0.4 + 0.4 * ((total_size_mb - thresholds["stressed"]["size_mb"]) /
                                           (thresholds["overloaded"]["size_mb"] - thresholds["stressed"]["size_mb"]))
        elif total_size_mb >= thresholds["normal"]["size_mb"]:
            size_overload = 0.2 * ((total_size_mb - thresholds["normal"]["size_mb"]) /
                                    (thresholds["stressed"]["size_mb"] - thresholds["normal"]["size_mb"]))
        combined_overload = max(memory_overload, size_overload) * 0.7 + (memory_overload + size_overload) * 0.15
        return min(2.0, combined_overload * self.adaptive_threshold_modifier)

    def _calculate_forgetting_pressure(self, overload_level: float) -> float:
        if overload_level < 0.4:
            return 0.0
        elif overload_level < 0.7:
            return (overload_level - 0.4) * 2
        return 0.6 + (overload_level - 0.7) * 1.3

    def _calculate_saturation_effects(self, overload_level: float) -> Dict[str, float]:
        effects = {}
        for effect_name, base_impact in self.OVERLOAD_EFFECTS.items():
            if overload_level < 0.3:
                effects[effect_name] = 1.0 - (overload_level * 0.1)
            elif overload_level < 0.7:
                effects[effect_name] = 1.0 - (0.03 + (overload_level - 0.3) * base_impact * 0.5)
            else:
                effects[effect_name] = 1.0 - (base_impact * 0.2 + (overload_level - 0.7) * base_impact)
            effects[effect_name] = max(0.1, effects[effect_name])
        return effects

    def should_trigger_emergency_forgetting(self) -> bool:
        return (self.current_state.overload_level > 0.8 and
                self.current_state.forgetting_pressure > 0.7)

    def execute_intelligent_forgetting(self, target_reduction: float = 0.3) -> Dict[str, Any]:
        logger.info(f"大脑容量不足，开始智能遗忘... 目标减少{target_reduction * 100:.0f}%记忆")
        try:
            MemoryEntry = self._get_memory_model()
            if not MemoryEntry:
                return {"forgotten_count": 0, "space_freed_mb": 0.0}
            memories = list(MemoryEntry.select().where(MemoryEntry.stream_id == self.stream_id))
            if not memories:
                return {"forgotten_count": 0, "space_freed_mb": 0.0}
            forgetting_candidates = []
            now = time.time()
            for memory in memories:
                score = self._calculate_forgetting_priority(memory, now)
                forgetting_candidates.append((score, memory))
            forgetting_candidates.sort(key=lambda x: x[0], reverse=True)
            target_count = int(len(memories) * target_reduction)
            forgotten_memories = forgetting_candidates[:target_count]
            forgotten_count = 0
            space_freed = 0
            forgotten_types = {"conversation": 0, "fact": 0, "compressed": 0, "other": 0}
            for score, memory in forgotten_memories:
                try:
                    content_size = len(memory.content)
                    mem_type = memory.memory_type
                    memory.delete_instance()
                    forgotten_count += 1
                    space_freed += content_size
                    forgotten_types[mem_type] = forgotten_types.get(mem_type, 0) + 1
                    if forgotten_count % 500 == 0:
                        logger.info(f"已遗忘 {forgotten_count} 条记忆...")
                except Exception as e:
                    logger.debug(f"遗忘单个记忆失败: {e}")
            space_freed_mb = space_freed / (1024 * 1024)
            self.forced_forgetting_count += forgotten_count
            self.adaptive_threshold_modifier = max(0.8, self.adaptive_threshold_modifier - 0.1)
            result = {
                "forgotten_count": forgotten_count,
                "space_freed_mb": space_freed_mb,
                "forgotten_by_type": forgotten_types,
                "forgetting_score_range": (
                    forgotten_memories[-1][0] if forgotten_memories else 0,
                    forgotten_memories[0][0] if forgotten_memories else 0
                )
            }
            logger.warning(f"智能遗忘完成：遗忘了 {forgotten_count} 条记忆，释放空间 {space_freed_mb:.2f}MB")
            return result
        except Exception as e:
            logger.error(f"智能遗忘执行失败: {e}")
            return {"forgotten_count": 0, "space_freed_mb": 0.0, "error": str(e)}

    def _calculate_forgetting_priority(self, memory, current_time: float) -> float:
        score = 0.0
        age_days = (current_time - memory.created_at) / 86400
        age_score = min(100, age_days) * self.FORGETTING_WEIGHTS["age"]
        score += age_score
        access_score = max(0, 20 - memory.access_count)
        score += access_score
        importance_score = (1.0 - memory.importance) * 30 * self.FORGETTING_WEIGHTS["importance"]
        score += importance_score
        redundancy_score = 0
        if len(memory.content) < 50:
            redundancy_score = 10
        elif memory.memory_type == "compressed":
            redundancy_score = 5
        score += redundancy_score * self.FORGETTING_WEIGHTS["redundancy"]
        if memory.importance > 0.8:
            score *= 0.3
        if memory.access_count > 10:
            score *= 0.4
        if (current_time - memory.created_at) < 86400:
            score *= 0.5
        return score

    def apply_overload_effects(self, operation: str, **kwargs) -> Dict[str, Any]:
        if self.current_state.overload_level < 0.3:
            return {"success": True, "effect_applied": False}
        effects = self.current_state.saturation_effects or {}
        result = {"success": True, "effect_applied": True, "effects": {}}
        if operation == "create_memory":
            creation_success_rate = effects.get("memory_creation", 1.0)
            if random.random() > creation_success_rate:
                result["success"] = False
                result["reason"] = "memory_overload_creation_failed"
                result["effects"]["creation_blocked"] = True
                logger.debug("记忆创建被阻止（大脑饱和）")
        elif operation == "recall_memory":
            recall_accuracy = effects.get("memory_recall", 1.0)
            confusion_rate = 1.0 - effects.get("confusion_rate", 0.0)
            if random.random() > recall_accuracy:
                result["success"] = False
                result["reason"] = "memory_overload_recall_failed"
                result["effects"]["recall_blocked"] = True
            elif random.random() > confusion_rate:
                result["effects"]["confusion"] = True
                result["confused_content"] = self._generate_confused_response()
        elif operation == "learning":
            learning_rate = effects.get("learning_ability", 1.0)
            if random.random() > learning_rate:
                result["success"] = False
                result["reason"] = "memory_overload_learning_impaired"
                result["effects"]["learning_blocked"] = True
        return result

    def _generate_confused_response(self) -> str:
        confused_responses = [
            "我...我好像有点想不起来了...",
            "咦？刚才说的是什么来着？",
            "诶...感觉脑子有点混乱...",
            "等等，让我想想...好像记不太清了",
            "嗯？我是不是忘了什么重要的事情？",
            "有点头晕...记忆好像混在一起了",
            "抱歉，我现在有点分不清了..."
        ]
        return random.choice(confused_responses)

    def _log_overload_status(self):
        state = self.current_state
        if state.overload_level < 0.3:
            status_text = "正常"
        elif state.overload_level < 0.6:
            status_text = "轻度过载"
        elif state.overload_level < 0.9:
            status_text = "严重过载"
        else:
            status_text = "临界过载"
        logger.info(f"大脑状态: {status_text} | 记忆数: {state.total_memories} | "
                     f"大小: {state.total_size_mb:.1f}MB | 过载度: {state.overload_level:.2f} | "
                     f"遗忘压力: {state.forgetting_pressure:.2f}")
        if state.overload_level > 0.8:
            logger.warning("大脑严重过载，可能出现健忘、困惑症状")
        if self.should_trigger_emergency_forgetting():
            logger.warning("达到紧急遗忘阈值，建议立即清理记忆")

    def get_overload_behavioral_prompt(self) -> str:
        if self.current_state.overload_level < 0.4:
            return ""
        effects = self.current_state.saturation_effects or {}
        overload_level = self.current_state.overload_level
        prompts = []
        if overload_level > 0.6:
            prompts.append("你现在感到大脑有些混乱和疲惫，可能会忘记一些事情或想不起来某些细节。")
        if effects.get("confusion_rate", 0) > 0.3:
            prompts.append("你偶尔会感到困惑，可能会说'让我想想...'或'我有点想不起来了'。")
        if effects.get("focus_span", 1.0) < 0.7:
            prompts.append("你的注意力不太集中，可能会在话题中走神或忘记刚才在说什么。")
        if overload_level > 0.8:
            prompts.append("你感觉脑袋很胀，像是装了太多东西，学习新事物变得困难。")
        if prompts:
            return "【记忆过载状态】" + "；".join(prompts)
        return ""

    def get_overload_statistics(self) -> Dict[str, Any]:
        return {
            "current_overload_level": self.current_state.overload_level,
            "total_memories": self.current_state.total_memories,
            "total_size_mb": self.current_state.total_size_mb,
            "forgetting_pressure": self.current_state.forgetting_pressure,
            "saturation_effects": self.current_state.saturation_effects,
            "forced_forgetting_count": self.forced_forgetting_count,
            "adaptive_threshold_modifier": self.adaptive_threshold_modifier,
            "overload_history_recent": self.overload_history[-10:] if self.overload_history else [],
            "emergency_forgetting_needed": self.should_trigger_emergency_forgetting()
        }


_overload_protectors: Dict[str, MemoryOverloadProtector] = {}


def get_memory_overload_protector(stream_id: str = "default") -> MemoryOverloadProtector:
    if stream_id not in _overload_protectors:
        _overload_protectors[stream_id] = MemoryOverloadProtector(stream_id)
    return _overload_protectors[stream_id]


def cleanup_overload_protectors():
    global _overload_protectors
    _overload_protectors.clear()

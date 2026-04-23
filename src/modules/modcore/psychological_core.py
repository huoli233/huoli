import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.modules.trauma.runtime_config import trauma_module_view

logger = get_logger("心理核心")


@dataclass
class PsychologicalEvent:
    """心理事件数据类"""

    event_type: str
    stream_id: str
    user_id: str
    content: str = ""
    intensity: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PsychologicalResponse:
    """心理响应数据类"""

    should_block: bool = False
    block_duration: float = 0.0
    block_reason: str = ""
    should_switch_persona: bool = False
    target_persona: str = ""
    persona_duration: float = 0.0
    trauma_layers: Dict[str, Any] = field(default_factory=dict)
    emotion_changes: Dict[str, float] = field(default_factory=dict)
    memory_stored: bool = False
    response_mode: str = "normal"
    actions: List[Dict[str, Any]] = field(default_factory=list)


class PsychologicalCore:
    """心理核心

    聚合情绪追踪数据，输出用户心理画像。
    处理各类心理事件，管理思考值消耗和恢复。
    """

    _instance: Optional["PsychologicalCore"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._trust_ledger: Dict[str, float] = {}
        self._emotion_trackers: Dict[str, Any] = {}
        self._trauma_systems: Dict[str, Any] = {}
        self._persona_generator = None
        self._memory_system = None
        self._harassment_detector = None
        self._initialized = True
        self._load_config()
        self._load_subsystems()
        logger.info("心理核心初始化完成")

    def _load_config(self):
        core_cfg = trauma_module_view("psychological_core")
        self._max_pressure = core_cfg.get("max_pressure", 100.0)
        self._recovery_rate = core_cfg.get("recovery_rate", 0.1)
        self._harassment_block_high = core_cfg.get("harassment_block_high", 120.0)
        self._harassment_block_mid = core_cfg.get("harassment_block_mid", 60.0)
        self._harassment_block_low = core_cfg.get("harassment_block_low", 30.0)
        self._annoyance_per_harass = core_cfg.get("annoyance_per_harass", 8.0)
        self._persona_switch_duration = core_cfg.get("persona_switch_duration", 600.0)
        self._harassment_intensity_high = core_cfg.get("harassment_intensity_high", 10.0)
        self._harassment_intensity_mid = core_cfg.get("harassment_intensity_mid", 5.0)
        self._harassment_emotion_divisor = core_cfg.get("harassment_emotion_divisor", 5.0)
        self._harassment_emotion_min = core_cfg.get("harassment_emotion_min", 0.5)
        self._harassment_emotion_max = core_cfg.get("harassment_emotion_max", 2.5)
        self._persona_switch_severe = core_cfg.get("persona_switch_severe", 9.0)
        self._persona_switch_medium = core_cfg.get("persona_switch_medium", 7.0)
        self._persona_switch_light = core_cfg.get("persona_switch_light", 5.0)
        self._positive_interaction_min = core_cfg.get("positive_interaction_min", 0.5)
        self._positive_interaction_multiplier = core_cfg.get("positive_interaction_multiplier", 4.0)
        self._negative_interaction_min = core_cfg.get("negative_interaction_min", 0.5)
        self._recovery_intensity = core_cfg.get("recovery_intensity", 1.0)
        self._stamina_default = core_cfg.get("stamina_default", 100.0)
        self._stamina_trauma_scale = core_cfg.get("stamina_trauma_scale", 0.5)
        self._stamina_annoyance_scale = core_cfg.get("stamina_annoyance_scale", 0.3)
        self._stamina_affection_bonus_scale = core_cfg.get("stamina_affection_bonus_scale", 0.1)
        self._response_severe_trauma_threshold = core_cfg.get("response_severe_trauma_threshold", 8.0)
        self._response_trauma_defense_threshold = core_cfg.get("response_trauma_defense_threshold", 5.0)
        self._response_cold_rejection_annoyance = core_cfg.get("response_cold_rejection_annoyance", 80.0)
        self._response_irritated_annoyance = core_cfg.get("response_irritated_annoyance", 50.0)
        self._response_intimate_affection = core_cfg.get("response_intimate_affection", 80.0)
        self._response_friendly_affection = core_cfg.get("response_friendly_affection", 50.0)
        self._response_hostile_affection = core_cfg.get("response_hostile_affection", -50.0)
        self._hint_high_affection = core_cfg.get("hint_high_affection", 50.0)
        self._hint_low_affection = core_cfg.get("hint_low_affection", -30.0)
        self._hint_high_annoyance = core_cfg.get("hint_high_annoyance", 40.0)
        self._hint_high_trauma = core_cfg.get("hint_high_trauma", 5.0)
        self._surface_shock_cap = core_cfg.get("surface_shock_cap", 10.0)
        self._surface_fear_cap = core_cfg.get("surface_fear_cap", 8.0)
        self._deep_violation_cap = core_cfg.get("deep_violation_cap", 10.0)
        self._deep_helplessness_cap = core_cfg.get("deep_helplessness_cap", 9.0)
        self._surface_shock_scale = core_cfg.get("surface_shock_scale", 0.8)
        self._surface_fear_scale = core_cfg.get("surface_fear_scale", 0.6)
        self._deep_violation_scale = core_cfg.get("deep_violation_scale", 0.9)
        self._deep_helplessness_scale = core_cfg.get("deep_helplessness_scale", 0.7)
        self._flashback_content_chars = int(core_cfg.get("flashback_content_chars", 100))
        self._flashback_intensity_min = core_cfg.get("flashback_intensity_min", 0.5)
        self._flashback_intensity_max = core_cfg.get("flashback_intensity_max", 1.0)
        self._trust_min = core_cfg.get("trust_min", -50.0)
        self._trust_max = core_cfg.get("trust_max", 100.0)
        self._computed_trust_min = core_cfg.get("computed_trust_min", 0.0)
        self._computed_trust_max = core_cfg.get("computed_trust_max", 100.0)
        self._trust_affection_weight = core_cfg.get("trust_affection_weight", 0.4)
        self._trust_interaction_weight = core_cfg.get("trust_interaction_weight", 0.5)
        self._trust_interaction_cap = core_cfg.get("trust_interaction_cap", 30.0)

    def _load_subsystems(self):
        try:
            from src.modules.modcore.dynamic_persona.persona_generator import (
                get_persona_generator,
            )

            self._persona_generator = get_persona_generator()
        except Exception as e:
            logger.debug(f"人格生成器加载跳过: {e}")
        try:
            from src.memory_system.memory_core import get_memory_core

            self._memory_system = get_memory_core()
        except Exception as e:
            logger.debug(f"记忆系统加载跳过: {e}")
        try:
            from src.modules.safety.harassment_detector import (
                HarassmentDetector,
            )

            self._harassment_detector = HarassmentDetector()
            logger.info("骚扰检测器加载成功")
        except Exception as e:
            logger.debug(f"骚扰检测器加载跳过: {e}")

    def _get_emotion_tracker(self, stream_id: str):
        if stream_id not in self._emotion_trackers:
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import (
                    get_emotion_tracker,
                )

                self._emotion_trackers[stream_id] = get_emotion_tracker(stream_id)
            except Exception as e:
                logger.debug(f"情绪追踪器获取失败: {e}")
                return None
        return self._emotion_trackers[stream_id]

    def _get_trauma_system(self, user_id: str):
        if user_id not in self._trauma_systems:
            try:
                from src.modules.trauma.complex_psychology import (
                    get_complex_psychology,
                )

                self._trauma_systems[user_id] = get_complex_psychology()
            except Exception as e:
                logger.debug(f"创伤系统获取失败: {e}")
                return None
        return self._trauma_systems[user_id]

    async def process_event(self, event: PsychologicalEvent) -> PsychologicalResponse:
        """处理心理事件"""
        response = PsychologicalResponse()
        event_type = event.event_type
        if event_type == "harassment":
            response = await self._handle_harassment(event)
        elif event_type == "message":
            response = await self._handle_message(event)
        elif event_type == "positive_interaction":
            response = await self._handle_positive_interaction(event)
        elif event_type == "negative_interaction":
            response = await self._handle_negative_interaction(event)
        elif event_type == "trauma_trigger":
            response = await self._handle_trauma_trigger(event)
        elif event_type == "recovery":
            response = await self._handle_recovery(event)
        else:
            logger.debug(f"未知心理事件类型: {event_type}")
        return response

    async def _handle_harassment(self, event: PsychologicalEvent) -> PsychologicalResponse:
        """处理骚扰事件"""
        response = PsychologicalResponse()
        stream_id = event.stream_id
        user_id = event.user_id
        content = event.content
        harassment_details = event.metadata.get("harassment_details", {})
        harassment_intensity = harassment_details.get("total_intensity", event.intensity)
        harassment_count = harassment_details.get("total_similar", 3)
        harassment_type = harassment_details.get("harassment_type", "unknown")
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            if hasattr(tracker, "apply_harassment"):
                state = tracker.get_user_state(user_id, create_if_missing=False)
                old_annoyance = getattr(state, "annoyance", 0) if state else 0
                old_trauma = getattr(state, "trauma_score", 0) if state else 0
                result = tracker.apply_harassment(
                    user_id,
                    intensity=max(
                        self._harassment_emotion_min,
                        min(
                            self._harassment_emotion_max,
                            harassment_intensity / self._harassment_emotion_divisor,
                        ),
                    ),
                    reason=f"{harassment_type}: {content[:40]}",
                    content=content,
                )
                response.emotion_changes = {
                    "annoyance_delta": result.get("annoyance_delta", 0.0),
                    "trauma_delta": result.get("trauma_delta", 0.0),
                    "annoyance": result.get("annoyance", old_annoyance),
                    "trauma_score": result.get("trauma_score", old_trauma),
                }
        if harassment_intensity >= self._harassment_intensity_high:
            block_duration = self._harassment_block_high
            block_reason = f"严重性骚扰/攻击: {harassment_type}"
        elif harassment_intensity >= self._harassment_intensity_mid:
            block_duration = self._harassment_block_mid
            block_reason = f"持续骚扰: {harassment_type}"
        else:
            block_duration = self._harassment_block_low
            block_reason = f"轻微骚扰: {harassment_type}"
        if block_duration > 0:
            response.should_block = True
            response.block_duration = block_duration
            response.block_reason = block_reason
            response.actions.append(
                {
                    "type": "user_blocked_by_harassment",
                    "duration": block_duration,
                    "reason": block_reason,
                }
            )
            uid_masked = self._mask_id(user_id)
            logger.warning(f"心理防御 {uid_masked} 触发拦截 | 强度:{harassment_intensity:.1f} 原因:{harassment_type}")
        trauma_system = self._get_trauma_system(user_id)
        if trauma_system:
            trauma_layers = self._generate_harassment_trauma_layers(harassment_intensity, harassment_count)
            response.trauma_layers = trauma_layers
            response.actions.append({"type": "trauma_layers_generated", "layers": trauma_layers})
        should_switch_persona = False
        target_state = ""
        if harassment_intensity >= self._persona_switch_severe:
            should_switch_persona = True
            target_state = "应激爆发状态"
        elif harassment_intensity >= self._persona_switch_medium:
            should_switch_persona = True
            target_state = "伪装动摇状态"
        elif harassment_intensity >= self._persona_switch_light:
            should_switch_persona = True
            target_state = "勉强伪装状态"
        if should_switch_persona and self._persona_generator:
            try:
                response.should_switch_persona = True
                response.target_persona = target_state
                response.persona_duration = self._persona_switch_duration
                response.actions.append({"type": "trauma_persona_switch", "target": target_state})
                logger.info(f"心理防御人格切换至 {target_state}")
            except Exception as persona_e:
                logger.debug(f"骚扰触发人格切换失败: {persona_e}")
        response.response_mode = "trauma_defense"
        return response

    async def _handle_message(self, event: PsychologicalEvent) -> PsychologicalResponse:
        """处理消息事件"""
        response = PsychologicalResponse()
        stream_id = event.stream_id
        user_id = event.user_id
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            if hasattr(tracker, "is_user_blocked"):
                blocked, block_info = tracker.is_user_blocked(user_id)
                if blocked:
                    response.should_block = True
                    response.block_reason = (
                        block_info.get("reason", "用户被屏蔽") if isinstance(block_info, dict) else str(block_info)
                    )
                    response.actions.append(
                        {
                            "type": "blocked_by_existing",
                            "reason": response.block_reason,
                        }
                    )
                    return response
            if hasattr(tracker, "is_user_protected"):
                protected, protect_reason = tracker.is_user_protected(user_id)
                if protected:
                    response.should_block = True
                    response.block_reason = protect_reason
                    response.actions.append(
                        {
                            "type": "protected_mode_active",
                            "reason": protect_reason,
                        }
                    )
                    return response
            if hasattr(tracker, "process_interaction_with_llm"):
                emotion_result = await tracker.process_interaction_with_llm(
                    user_id,
                    event.content,
                    use_llm=False,
                )
                response.emotion_changes = {
                    "affection": emotion_result.get("affection", 0),
                    "annoyance": emotion_result.get("annoyance", 0),
                    "trauma_score": emotion_result.get("trauma_score", 0),
                    "affection_delta": emotion_result.get("affection_delta", 0),
                    "trust_delta": emotion_result.get("trust_delta", 0),
                }
            response.response_mode = self.get_response_mode(stream_id, user_id)
        return response

    async def _handle_positive_interaction(self, event: PsychologicalEvent) -> PsychologicalResponse:
        """处理正面交互事件"""
        response = PsychologicalResponse()
        stream_id = event.stream_id
        user_id = event.user_id
        intensity = event.intensity
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            if hasattr(tracker, "apply_positive_interaction"):
                state = tracker.get_user_state(user_id, create_if_missing=False)
                old_affection = getattr(state, "affection", 0) if state else 0
                result = tracker.apply_positive_interaction(
                    user_id,
                    intensity=max(
                        self._positive_interaction_min,
                        intensity * self._positive_interaction_multiplier,
                    ),
                    reason=event.content[:40] or "心理核心正向交互",
                )
                response.emotion_changes = {
                    "affection_delta": result.get("affection_delta", 0.0),
                    "affection": result.get("affection", old_affection),
                }
                response.actions.append(
                    {
                        "type": "positive_emotion_update",
                        "changes": response.emotion_changes,
                    }
                )
        return response

    async def _handle_negative_interaction(self, event: PsychologicalEvent) -> PsychologicalResponse:
        """处理负面交互事件"""
        response = PsychologicalResponse()
        stream_id = event.stream_id
        user_id = event.user_id
        intensity = event.intensity
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            if hasattr(tracker, "apply_negative_interaction"):
                state = tracker.get_user_state(user_id, create_if_missing=False)
                old_annoyance = getattr(state, "annoyance", 0) if state else 0
                result = tracker.apply_negative_interaction(
                    user_id,
                    intensity=max(self._negative_interaction_min, intensity),
                    reason=event.content[:40] or "心理核心负向交互",
                    content=event.content,
                )
                response.emotion_changes = {
                    "annoyance_delta": result.get("annoyance_delta", 0.0),
                    "annoyance": result.get("annoyance", old_annoyance),
                }
                response.actions.append(
                    {
                        "type": "negative_emotion_update",
                        "changes": response.emotion_changes,
                    }
                )
        return response

    async def _handle_trauma_trigger(self, event: PsychologicalEvent) -> PsychologicalResponse:
        """处理创伤触发事件"""
        response = PsychologicalResponse()
        stream_id = event.stream_id
        user_id = event.user_id
        trigger_content = event.metadata.get("trigger_content", "")
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            state = (
                tracker.get_user_state(user_id, create_if_missing=False) if hasattr(tracker, "get_user_state") else None
            )
            if state and getattr(state, "trauma_score", 0) > 0:
                trauma_system = self._get_trauma_system(user_id)
                if trauma_system:
                    flashback_result = self._trigger_flashback(trauma_system, trigger_content)
                    response.trauma_layers = flashback_result
                    response.actions.append(
                        {
                            "type": "flashback_triggered",
                            "layers": flashback_result,
                        }
                    )
        return response

    async def _handle_recovery(self, event: PsychologicalEvent) -> PsychologicalResponse:
        """处理恢复事件"""
        response = PsychologicalResponse()
        stream_id = event.stream_id
        user_id = event.user_id
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            state = tracker.get_user_state(user_id, create_if_missing=False) if hasattr(tracker, "get_user_state") else None
            if state and hasattr(tracker, "apply_apology_repair"):
                tracker.apply_apology_repair(
                    user_id,
                    intensity=self._recovery_intensity,
                    reason=event.content[:40] or "心理核心恢复事件",
                )
                updated = tracker.get_user_state(user_id, create_if_missing=False)
                response.emotion_changes = {
                    "trauma_score": getattr(updated, "trauma_score", 0),
                    "annoyance": getattr(updated, "annoyance", 0),
                }
                response.actions.append(
                    {
                        "type": "recovery_applied",
                        "changes": response.emotion_changes,
                    }
                )
        return response

    def get_user_psychological_state(self, stream_id: str, user_id: str) -> Dict[str, Any]:
        """获取用户心理状态"""
        result = {
            "favor": 0.0,
            "trust": 0.0,
            "trust_value": 0.0,
            "annoyance": 0.0,
            "trauma": 0.0,
            "stamina": self._stamina_default,
            "is_blocked": False,
            "is_protected": False,
            "relationship": "陌生人",
            "impression": "初次见面",
        }
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            state = (
                tracker.get_user_state(user_id, create_if_missing=False) if hasattr(tracker, "get_user_state") else None
            )
            if state:
                result["favor"] = getattr(state, "affection", 0)
                canonical_trust = getattr(
                    state, "trust_value", getattr(state, "trust_score", 0)
                )
                result["trust"] = canonical_trust
                result["trust_value"] = canonical_trust
                result["annoyance"] = getattr(state, "annoyance", 0)
                result["trauma"] = getattr(state, "trauma_score", 0)
                result["relationship"] = getattr(state, "relationship", "陌生人")
                result["impression"] = getattr(state, "impression", "初次见面")
            if hasattr(tracker, "is_user_blocked"):
                blocked, _ = tracker.is_user_blocked(user_id)
                result["is_blocked"] = blocked
            if hasattr(tracker, "is_user_protected"):
                protected, _ = tracker.is_user_protected(user_id)
                result["is_protected"] = protected
        return result

    def update_stamina(
        self,
        stream_id: str,
        user_id: str,
        cost: float,
        action_type: str = "think",
    ) -> Dict[str, float]:
        """更新思考值消耗"""
        tracker = self._get_emotion_tracker(stream_id)
        if not tracker:
            return {"stamina": self._stamina_default, "cost": 0.0}
        state = tracker.get_user_state(user_id, create_if_missing=True) if hasattr(tracker, "get_user_state") else None
        if not state:
            return {"stamina": self._stamina_default, "cost": 0.0}
        current_stamina = getattr(state, "stamina", self._stamina_default)
        trauma_weight = 1.0 + (getattr(state, "trauma_score", 0) / 10.0) * self._stamina_trauma_scale
        annoyance_weight = 1.0 + (getattr(state, "annoyance", 0) / 100.0) * self._stamina_annoyance_scale
        affection_bonus = (
            (getattr(state, "affection", 0) / 100.0) * self._stamina_affection_bonus_scale
            if getattr(state, "affection", 0) > 0
            else 0.0
        )
        final_cost = cost * trauma_weight * annoyance_weight * (1.0 - affection_bonus)
        stamina = max(0.0, current_stamina - final_cost)
        if hasattr(state, "stamina"):
            state.stamina = stamina
        uid_masked = self._mask_id(user_id)
        logger.debug(
            f"思考值消耗 [{uid_masked}] {action_type} | 基础:{cost:.1f} 实际:{final_cost:.1f} | 原始:{
                current_stamina:.1f} 剩余:{stamina:.1f}"
        )
        return {"stamina": stamina, "cost": final_cost}

    def recover_stamina(self, stream_id: str, user_id: str, amount: float) -> Dict[str, float]:
        """恢复思考值"""
        tracker = self._get_emotion_tracker(stream_id)
        if not tracker:
            return {"stamina": self._stamina_default, "recovered": 0.0}
        state = tracker.get_user_state(user_id, create_if_missing=False) if hasattr(tracker, "get_user_state") else None
        if not state:
            return {"stamina": self._stamina_default, "recovered": 0.0}
        current_stamina = getattr(state, "stamina", self._stamina_default)
        actual_recovered = max(0.0, amount)
        stamina = min(self._stamina_default, current_stamina + actual_recovered)
        if hasattr(state, "stamina"):
            state.stamina = stamina
        if actual_recovered > 0:
            logger.debug(f"思考值恢复 | 恢复:{actual_recovered:.1f} | 原始:{current_stamina:.1f} 剩余:{stamina:.1f}")
        return {"stamina": stamina, "recovered": actual_recovered}

    def get_response_mode(self, stream_id: str, user_id: str) -> str:
        """获取响应模式"""
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            state = (
                tracker.get_user_state(user_id, create_if_missing=False) if hasattr(tracker, "get_user_state") else None
            )
            if state:
                trauma_score = getattr(state, "trauma_score", 0)
                annoyance = getattr(state, "annoyance", 0)
                affection = getattr(state, "affection", 0)
                if trauma_score >= self._response_severe_trauma_threshold:
                    return "severe_trauma"
                elif trauma_score >= self._response_trauma_defense_threshold:
                    return "trauma_defense"
                elif annoyance >= self._response_cold_rejection_annoyance:
                    return "cold_rejection"
                elif annoyance >= self._response_irritated_annoyance:
                    return "irritated"
                elif affection >= self._response_intimate_affection:
                    return "intimate"
                elif affection >= self._response_friendly_affection:
                    return "friendly"
                elif affection <= self._response_hostile_affection:
                    return "hostile"
        return "normal"

    def is_user_blocked(self, stream_id: str, user_id: str) -> Tuple[bool, str]:
        """检查用户是否被屏蔽"""
        tracker = self._get_emotion_tracker(stream_id)
        if tracker and hasattr(tracker, "is_user_blocked"):
            blocked, info = tracker.is_user_blocked(user_id)
            reason = info.get("reason", "") if isinstance(info, dict) else str(info)
            return blocked, reason
        return False, ""

    def is_interaction_rejected(self, stream_id: str, user_id: str) -> bool:
        """检查交互是否被拒绝"""
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            if hasattr(tracker, "is_user_blocked"):
                blocked, _ = tracker.is_user_blocked(user_id)
                if blocked:
                    return True
            if hasattr(tracker, "is_user_protected"):
                protected, _ = tracker.is_user_protected(user_id)
                if protected:
                    return True
        return False

    def get_stream_psychological_state(self, stream_id: str) -> Dict[str, Any]:
        """获取流心理状态"""
        result = {
            "active_persona": {},
            "global_rage": 0.0,
            "atmosphere": "normal",
        }
        try:
            from src.modules.modcore.dynamic_persona.persona_switcher import (
                get_persona_switcher,
            )

            switcher = get_persona_switcher()
            active_info = switcher.get_active_info(stream_id)
            if active_info:
                result["active_persona"] = active_info
                persona_obj = active_info.get("persona_obj")
                if persona_obj:
                    result["active_persona"]["inner_chaos_level"] = getattr(persona_obj, "inner_chaos_level", 0)
            tracker = self._get_emotion_tracker(stream_id)
            if tracker:
                result["global_rage"] = getattr(tracker, "global_rage_level", 0.0)
        except Exception as e:
            logger.debug(f"获取频道心理状态快照失败: {e}")
        return result

    def get_combined_psychological_hint(self, channel_id: str, user_id: str) -> str:
        """获取综合心理提示"""
        tracker = self._get_emotion_tracker(channel_id)
        if not tracker:
            return ""
        state = tracker.get_user_state(user_id, create_if_missing=False) if hasattr(tracker, "get_user_state") else None
        if not state:
            return ""
        parts = []
        affection = getattr(state, "affection", 0)
        annoyance = getattr(state, "annoyance", 0)
        trauma_score = getattr(state, "trauma_score", 0)
        if affection >= self._hint_high_affection:
            parts.append(f"好感度很高({affection:.0f})，态度温暖亲切")
        elif affection <= self._hint_low_affection:
            parts.append(f"好感度较低({affection:.0f})，态度冷淡")
        if annoyance > self._hint_high_annoyance:
            parts.append(f"烦躁度较高({annoyance:.0f})，可能不耐烦")
        if trauma_score > self._hint_high_trauma:
            parts.append(f"创伤分数较高({trauma_score:.1f})，回避敏感话题")
        return "；".join(parts) if parts else ""

    def detect_harassment_unified(self, user_id: str, channel_id: str, content: str) -> Tuple[bool, Dict[str, Any]]:
        """统一骚扰检测"""
        if self._harassment_detector:
            if hasattr(self._harassment_detector, "detect_harassment"):
                is_harassment, details = self._harassment_detector.detect_harassment(user_id, channel_id, content)
                return is_harassment, details
        return False, {}

    def _generate_harassment_trauma_layers(self, intensity: float, count: int) -> Dict[str, Any]:
        """生成骚扰创伤层"""
        return {
            "surface": {
                "shock": min(self._surface_shock_cap, intensity * self._surface_shock_scale),
                "fear": min(self._surface_fear_cap, intensity * self._surface_fear_scale),
            },
            "deep": {
                "violation": min(self._deep_violation_cap, intensity * self._deep_violation_scale),
                "helplessness": min(self._deep_helplessness_cap, intensity * self._deep_helplessness_scale),
            },
            "count": count,
            "timestamp": time.time(),
        }

    def _trigger_flashback(self, trauma_system: Any, trigger_content: str) -> Dict[str, Any]:
        """触发闪回"""
        return {
            "triggered": True,
            "trigger_content": trigger_content[: self._flashback_content_chars],
            "intensity": random.uniform(self._flashback_intensity_min, self._flashback_intensity_max),
            "timestamp": time.time(),
        }

    def _clamp_value(self, value: float, min_val: float, max_val: float) -> float:
        """限制值范围"""
        return max(min_val, min(max_val, value))

    def _mask_id(self, id_str: str) -> str:
        """遮蔽ID"""
        if not id_str:
            return ""
        if len(id_str) <= 8:
            return id_str[:2] + "****"
        return f"{id_str[:4]}****{id_str[-4:]}"

    def record_trust_delta(self, user_id: str, delta: float) -> None:
        """记录信任变化"""
        current = self._trust_ledger.get(user_id, 0.0)
        self._trust_ledger[user_id] = max(self._trust_min, min(self._trust_max, current + delta))

    def inject_emotion_event(
        self,
        channel_id: str,
        emotion_name: str = "",
        emotion_type: str = "",
        intensity: float = 0.0,
        source: str = "",
        event_tag: str = "",
    ) -> None:
        """将情绪事件注入频道氛围追踪器

        兼容两种调用签名：
        - waiting_handler:  emotion_name + source
        - proactive_decider: emotion_type + event_tag
        """
        tag = emotion_name or emotion_type or "neutral"
        label = source or event_tag or "unknown"
        try:
            from src.chat.heart_flow.emotion_stream import (
                get_channel_mood_tracker,
            )

            mood_tracker = get_channel_mood_tracker()
            # 正面情绪→正向社交反馈，负面→负向
            _POLARITY = {
                "joy": 1.0,
                "calm": 0.15,
                "gratitude": 0.8,
                "pride": 0.6,
                "sadness": -0.6,
                "pain": -1.0,
                "anger": -0.8,
                "fear": -0.5,
                "disgust": -0.7,
            }
            polarity = _POLARITY.get(tag, 0.0)
            delta = polarity * intensity * 10.0
            if abs(delta) > 1e-6:
                mood_tracker.record_social_feedback(channel_id, delta)
            logger.debug(
                f"情绪事件注入: channel={self._mask_id(channel_id)} "
                f"tag={tag} intensity={intensity:.2f} delta={delta:.2f} label={label}"
            )
        except Exception as exc:
            logger.debug(f"情绪事件注入失败: {exc}")

    def _compute_trust(self, user_id: str, affection: float, interactions: int) -> float:
        """计算信任值"""
        base_trust = self._trust_ledger.get(user_id, 0.0)
        affection_part = max(0.0, affection) * self._trust_affection_weight
        interaction_part = min(self._trust_interaction_cap, interactions * self._trust_interaction_weight)
        combined = base_trust + affection_part + interaction_part
        return max(self._computed_trust_min, min(self._computed_trust_max, combined))


_psychological_core: Optional[PsychologicalCore] = None


def get_psychological_core() -> PsychologicalCore:
    """获取心理核心单例"""
    global _psychological_core
    if _psychological_core is None:
        _psychological_core = PsychologicalCore()
    return _psychological_core


async def handle_harassment_unified(
    stream_id: str,
    user_id: str,
    content: str,
    harassment_details: Dict[str, Any],
) -> PsychologicalResponse:
    """统一骚扰处理接口"""
    core = get_psychological_core()
    event = PsychologicalEvent(
        event_type="harassment",
        stream_id=stream_id,
        user_id=user_id,
        content=content,
        intensity=harassment_details.get("total_intensity", 1.0),
        metadata={"harassment_details": harassment_details},
    )
    return await core.process_event(event)


async def handle_message_unified(stream_id: str, user_id: str, content: str) -> PsychologicalResponse:
    """统一消息处理接口"""
    core = get_psychological_core()
    event = PsychologicalEvent(
        event_type="message",
        stream_id=stream_id,
        user_id=user_id,
        content=content,
        metadata={},
    )
    return await core.process_event(event)


def get_psychological_state(stream_id: str, user_id: str) -> Dict[str, Any]:
    """获取心理状态"""
    core = get_psychological_core()
    return core.get_user_psychological_state(stream_id, user_id)


def get_response_mode(stream_id: str, user_id: str) -> str:
    """获取响应模式"""
    core = get_psychological_core()
    return core.get_response_mode(stream_id, user_id)

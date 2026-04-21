import time
import random
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("psychological_core")


@dataclass
class PsychologicalEvent:
    event_type: str
    stream_id: str
    user_id: str
    content: str = ""
    intensity: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PsychologicalResponse:
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
    _instance: Optional["PsychologicalCore"] = None

    @staticmethod
    def _bt_get(key: str, default):
        try:
            from src.config.behavior_tuner import get_behavior_tuner
            return get_behavior_tuner().get(key, default)
        except Exception:
            return default

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._emotion_trackers: Dict[str, Any] = {}
        self._trauma_systems: Dict[str, Any] = {}
        self._persona_generator = None
        self._memory_system = None
        self._harassment_detector = None
        self._initialized = True
        self._load_subsystems()
        logger.info("心理核心初始化完成")

    def _load_subsystems(self):
        try:
            from src.modules.modcore.dynamic_persona.persona_generator import get_persona_generator
            self._persona_generator = get_persona_generator()
        except Exception as e:
            logger.debug(f"人格生成器加载跳过: {e}")
        try:
            from src.memory_system.memory_core import get_memory_core
            self._memory_system = get_memory_core()
        except Exception as e:
            logger.debug(f"记忆系统加载跳过: {e}")
        try:
            from src.chat.utils.harassment_detector import get_harassment_detector
            self._harassment_detector = get_harassment_detector()
            logger.info("骚扰检测器加载成功")
        except Exception as e:
            logger.debug(f"骚扰检测器加载跳过: {e}")

    def _get_emotion_tracker(self, stream_id: str):
        if stream_id not in self._emotion_trackers:
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
                self._emotion_trackers[stream_id] = get_emotion_tracker(stream_id)
            except Exception as e:
                logger.debug(f"情绪追踪器获取失败: {e}")
                return None
        return self._emotion_trackers[stream_id]

    def _calculate_psychological_depth(self, stream_id: str, user_id: str) -> Dict[str, Any]:
        tracker = self._get_emotion_tracker(stream_id)
        if not tracker:
            return {}
        state = tracker.get_user_state(user_id, False)
        if not state:
            return {}
        flashback_prob = min(0.4, (state.trauma_score / 10.0) * 0.5)
        is_flashback = random.random() < flashback_prob
        submission_factor = state.submission_level / 100.0
        resistance_factor = state.training_resistance / 100.0
        inner_chaos = (state.trauma_score * 0.4 + state.annoyance * 0.3 + (1.0 - state.trust_score) * 30.0) / 10.0
        return {
            "is_flashback": is_flashback,
            "submission_factor": submission_factor,
            "resistance_factor": resistance_factor,
            "inner_chaos": inner_chaos,
            "training_stage": state.training_stage
        }

    def _get_trauma_system(self, user_id: str):
        if user_id not in self._trauma_systems:
            try:
                from src.modules.modcore.dynamic_persona.complex_trauma_psychology import ComplexTraumaPsychology
                self._trauma_systems[user_id] = ComplexTraumaPsychology()
            except Exception as e:
                logger.debug(f"创伤系统获取失败: {e}")
                return None
        return self._trauma_systems[user_id]

    async def process_event(self, event: PsychologicalEvent) -> PsychologicalResponse:
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
            from src.modules.modcore.dynamic_persona.emotion_tracker import EmotionTracker
            state = tracker.get_user_state(user_id, False)
            if state:
                old_annoyance = state.annoyance
                old_trauma = state.trauma_score
                trauma_increase = min(2.0, harassment_intensity * 0.15)
                state.trauma_score = EmotionTracker.clamp_trauma(state.trauma_score + trauma_increase)
                trauma_record_level = max(1, min(6, int(harassment_intensity / 2)))
                tracker.add_trauma_record(state, content, trauma_record_level, user_id, trigger_type="harassment")
                _ann_base = self._bt_get("annoyance_per_harass", 10.0)
                annoyance_increase = _ann_base + harassment_intensity * 2.5
                state.annoyance = EmotionTracker.clamp_annoyance(state.annoyance + annoyance_increase)
                response.emotion_changes = {
                    "annoyance_delta": state.annoyance - old_annoyance,
                    "trauma_delta": state.trauma_score - old_trauma,
                    "annoyance": state.annoyance,
                    "trauma_score": state.trauma_score,
                }
                tracker._save_states()
            if harassment_intensity >= 10:
                block_duration = self._bt_get("harassment_block_high", 120.0)
                block_reason = f"严重性骚扰/攻击: {harassment_type}"
            elif harassment_intensity >= 5:
                block_duration = self._bt_get("harassment_block_mid", 60.0)
                block_reason = f"持续骚扰: {harassment_type}"
            else:
                block_duration = self._bt_get("harassment_block_low", 30.0)
                block_reason = f"轻微骚扰: {harassment_type}"
            if block_duration > 0:
                tracker.block_user(user_id, block_duration, f"{block_reason} | 心理防线已触发")
                response.should_block = True
                response.block_duration = block_duration
                response.block_reason = block_reason
                response.actions.append({"type": "user_blocked_by_harassment", "duration": block_duration, "reason": block_reason})
                from src.modules.modcore.dynamic_persona.emotion_tracker import EmotionTracker
                uid_masked = EmotionTracker.mask_id(user_id)
                logger.warning(f"心理防御 {uid_masked} 触发拦截 | 强度:{harassment_intensity:.1f} 原因:{harassment_type}")
        trauma_system = self._get_trauma_system(user_id)
        if trauma_system:
            trauma_layers = trauma_system.generate_harassment_trauma_layers(harassment_intensity, harassment_count)
            response.trauma_layers = trauma_layers
            response.actions.append({"type": "trauma_layers_generated", "layers": trauma_layers})
        if self._memory_system:
            try:
                extra_metadata = {
                    "is_harassment": True, "harassment_type": harassment_type,
                    "harassment_intensity": harassment_intensity,
                    "harassment_count": harassment_count,
                    "trauma_layers": response.trauma_layers,
                }
                await self._memory_system.store_memory(
                    stream_id=stream_id, content=content, user_id=user_id,
                    importance=0.95, is_bot=False, extra_metadata=extra_metadata
                )
                response.memory_stored = True
                response.actions.append({"type": "harassment_memory_stored"})
            except Exception as mem_e:
                logger.debug(f"存储骚扰记忆失败: {mem_e}")
        tracker = self._get_emotion_tracker(stream_id)
        should_switch_persona = False
        if self._persona_generator:
            if tracker:
                trauma_score = 0.0
                try:
                    user_state = tracker.get_user_state(user_id, False)
                    if user_state:
                        trauma_score = user_state.trauma_score
                except Exception:
                    pass
                if harassment_intensity >= 9:
                    should_switch_persona = True
                elif harassment_intensity >= 7 and trauma_score >= 2.0:
                    should_switch_persona = True
                elif harassment_intensity >= 5 and trauma_score >= 3.0:
                    should_switch_persona = True
                else:
                    logger.debug(f"人格切换跳过 | harassment_intensity={harassment_intensity:.1f}, trauma_score={trauma_score:.1f}")
            else:
                should_switch_persona = (harassment_intensity >= 5)
        if should_switch_persona:
            try:
                if harassment_intensity >= 9:
                    target_state = "应激爆发状态"
                elif harassment_intensity >= 7:
                    target_state = "伪装动摇状态"
                else:
                    target_state = "勉强伪装状态"
                switched = await self._persona_generator.switch_persona_state(
                    stream_id=stream_id,
                    state_name=target_state,
                    trigger_reason=f"骚扰防卫: {harassment_type} (强度:{harassment_intensity:.1f})",
                    extra_data={
                        "harassment_details": harassment_details,
                        "trauma_response": response.trauma_layers,
                        "intensity": harassment_intensity
                    }
                )
                if switched:
                    response.should_switch_persona = True
                    response.target_persona = target_state
                    response.persona_duration = self._bt_get("persona_switch_duration", 600.0)
                    response.actions.append({"type": "trauma_persona_switch", "target": target_state})
                    logger.info(f"心理防御人格切换至 {target_state}")
            except Exception as persona_e:
                logger.debug(f"骚扰触发人格切换失败: {persona_e}")
        response.response_mode = "trauma_defense"
        return response

    async def _handle_message(self, event: PsychologicalEvent) -> PsychologicalResponse:
        response = PsychologicalResponse()
        stream_id = event.stream_id
        user_id = event.user_id
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            blocked, block_info = tracker.is_user_blocked(user_id)
            if blocked:
                response.should_block = True
                response.block_reason = block_info.get("reason", "用户被屏蔽") if isinstance(block_info, dict) else str(block_info)
                response.actions.append({"type": "blocked_by_existing", "reason": response.block_reason})
                return response
            protected, protect_reason = tracker.is_user_protected(user_id)
            if protected:
                response.should_block = True
                response.block_reason = protect_reason
                response.actions.append({"type": "protected_mode_active", "reason": protect_reason})
                return response
            emotion_result = await tracker.process_interaction(user_id, event.content, use_llm=False)
            response.emotion_changes = {
                "affection": emotion_result.get("affection", 0),
                "annoyance": emotion_result.get("annoyance", 0),
                "trauma_score": emotion_result.get("trauma_score", 0),
                "affection_delta": emotion_result.get("affection_delta", 0),
                "trust_delta": emotion_result.get("trust_delta", 0),
            }
            response.response_mode = self.get_response_mode(stream_id, user_id)
            if emotion_result.get("should_block"):
                response.should_block = True
                response.block_reason = emotion_result.get("block_reason", "触发屏蔽阈值")
                response.actions.append({"type": "blocked_by_threshold", "reason": response.block_reason})
            if emotion_result.get("persona_switch"):
                persona_info = emotion_result["persona_switch"]
                response.should_switch_persona = True
                response.target_persona = persona_info.get("type", "")
                response.persona_duration = float(persona_info.get("duration", 180))
                response.actions.append({"type": "persona_switch_triggered", "target": response.target_persona, "reason": persona_info.get("reason", "")})
        return response

    async def _handle_positive_interaction(self, event: PsychologicalEvent) -> PsychologicalResponse:
        response = PsychologicalResponse()
        stream_id = event.stream_id
        user_id = event.user_id
        intensity = event.intensity
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            state = tracker.get_user_state(user_id, False)
            if state:
                old_affection = state.affection
                affection_gain = intensity * 2.0
                state.affection = min(100, state.affection + affection_gain)
                if state.annoyance > 0:
                    annoyance_decay = intensity * 1.0
                    state.annoyance = max(0, state.annoyance - annoyance_decay)
                response.emotion_changes = {"affection_delta": state.affection - old_affection, "affection": state.affection}
                tracker._save_states()
                response.actions.append({"type": "positive_emotion_update", "changes": response.emotion_changes})
        return response

    async def _handle_negative_interaction(self, event: PsychologicalEvent) -> PsychologicalResponse:
        response = PsychologicalResponse()
        stream_id = event.stream_id
        user_id = event.user_id
        intensity = event.intensity
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            state = tracker.get_user_state(user_id, False)
            if state:
                old_annoyance = state.annoyance
                annoyance_gain = intensity * 3.0
                state.annoyance = min(100, state.annoyance + annoyance_gain)
                if state.affection > 0:
                    affection_decay = intensity * 0.5
                    state.affection = max(-100, state.affection - affection_decay)
                response.emotion_changes = {"annoyance_delta": state.annoyance - old_annoyance, "annoyance": state.annoyance}
                tracker._save_states()
                response.actions.append({"type": "negative_emotion_update", "changes": response.emotion_changes})
        return response

    async def _handle_trauma_trigger(self, event: PsychologicalEvent) -> PsychologicalResponse:
        response = PsychologicalResponse()
        stream_id = event.stream_id
        user_id = event.user_id
        trigger_content = event.metadata.get("trigger_content", "")
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            state = tracker.get_user_state(user_id, False)
            if state and state.trauma_score > 0:
                trauma_system = self._get_trauma_system(user_id)
                if trauma_system:
                    flashback_result = trauma_system.trigger_flashback(trigger_content)
                    response.trauma_layers = flashback_result
                    response.actions.append({"type": "flashback_triggered", "layers": flashback_result})
        return response

    async def _handle_recovery(self, event: PsychologicalEvent) -> PsychologicalResponse:
        response = PsychologicalResponse()
        stream_id = event.stream_id
        user_id = event.user_id
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            state = tracker.get_user_state(user_id, False)
            if state:
                if state.trauma_score > 0:
                    state.trauma_score = max(0, state.trauma_score - 0.5)
                if state.annoyance > 0:
                    state.annoyance = max(0, state.annoyance - 1.0)
                response.emotion_changes = {"trauma_score": state.trauma_score, "annoyance": state.annoyance}
                tracker._save_states()
                response.actions.append({"type": "recovery_applied", "changes": response.emotion_changes})
        return response

    def get_user_psychological_state(self, stream_id: str, user_id: str) -> Dict[str, Any]:
        result = {
            "favor": 0.0, "trust": 0.0, "annoyance": 0.0,
            "trauma": 0.0, "mental_fatigue": 0.0, "stamina": 100.0,
            "is_blocked": False, "is_protected": False,
            "relationship": "陌生人", "impression": "初次见面",
        }
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            state = tracker.get_user_state(user_id, False)
            if state:
                result["favor"] = state.affection
                result["trust"] = state.trust_score
                result["annoyance"] = state.annoyance
                result["trauma"] = state.trauma_score
                result["mental_fatigue"] = getattr(state, 'mental_fatigue', 0.0)
                result["stamina"] = 100.0 - result["mental_fatigue"]
                result["relationship"] = state.relationship
                result["impression"] = state.impression
            blocked, _ = tracker.is_user_blocked(user_id)
            result["is_blocked"] = blocked
            protected, _ = tracker.is_user_protected(user_id)
            result["is_protected"] = protected
        return result

    def update_stamina(self, stream_id: str, user_id: str, cost: float, action_type: str = "think") -> Dict[str, float]:
        tracker = self._get_emotion_tracker(stream_id)
        if not tracker:
            return {"stamina": 100.0, "mental_fatigue": 0.0, "cost": 0.0}
        state = tracker.get_user_state(user_id, create_if_missing=True)
        if not state:
            return {"stamina": 100.0, "mental_fatigue": 0.0, "cost": 0.0}
        
        # ========== 心理状态约束权重 ==========
        # 1. 创伤权重：创伤越高，维持自我的成本越高
        trauma_weight = 1.0 + (state.trauma_score / 10.0) * 0.5  # 最高 +50% 消耗
        # 2. 厌恶权重：如果讨厌对方，被迫互动的阻力越大
        annoyance_weight = 1.0 + (state.annoyance / 100.0) * 0.3  # 最高 +30% 消耗
        # 3. 好感权重：好感越高，互动越自然省力
        affection_bonus = (state.affection / 100.0) * 0.1 if state.affection > 0 else 0.0 # 最高 -10% 消耗
        
        final_cost = cost * trauma_weight * annoyance_weight * (1.0 - affection_bonus)
        
        current_fatigue = getattr(state, 'mental_fatigue', 0.0)
        new_fatigue = min(100.0, current_fatigue + final_cost)
        state.mental_fatigue = new_fatigue
        stamina = 100.0 - new_fatigue
        
        uid_masked = f"{user_id[:4]}****{user_id[-4:]}" if len(user_id) > 8 else user_id
        logger.debug(f"脑力消耗 [{uid_masked}] {action_type} | 基础:{cost:.1f} 实际:{final_cost:.1f} | 疲劳:{new_fatigue:.1f} 剩余潜力:{stamina:.1f}")
        return {"stamina": stamina, "mental_fatigue": new_fatigue, "cost": final_cost}

    def recover_stamina(self, stream_id: str, user_id: str, amount: float) -> Dict[str, float]:
        tracker = self._get_emotion_tracker(stream_id)
        if not tracker:
            return {"stamina": 100.0, "mental_fatigue": 0.0, "recovered": 0.0}
        state = tracker.get_user_state(user_id, create_if_missing=False)
        if not state:
            return {"stamina": 100.0, "mental_fatigue": 0.0, "recovered": 0.0}
        current_fatigue = getattr(state, 'mental_fatigue', 0.0)
        new_fatigue = max(0.0, current_fatigue - amount)
        state.mental_fatigue = new_fatigue
        stamina = 100.0 - new_fatigue
        actual_recovered = current_fatigue - new_fatigue
        if actual_recovered > 0:
            logger.debug(f"聊天值恢复 | 恢复:{actual_recovered:.1f} | 疲劳度:{current_fatigue:.1f}->{new_fatigue:.1f} | 聊天值:{stamina:.1f}")
        return {"stamina": stamina, "mental_fatigue": new_fatigue, "recovered": actual_recovered}

    def is_user_blocked(self, stream_id: str, user_id: str) -> Tuple[bool, str]:
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            blocked, info = tracker.is_user_blocked(user_id)
            reason = info.get("reason", "") if isinstance(info, dict) else str(info)
            return blocked, reason
        return False, ""

    def get_response_mode(self, stream_id: str, user_id: str) -> str:
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            state = tracker.get_user_state(user_id, False)
            if state:
                if state.trauma_score >= 8:
                    return "severe_trauma"
                elif state.trauma_score >= 5:
                    return "trauma_defense"
                elif state.annoyance >= 80:
                    return "cold_rejection"
                elif state.annoyance >= 50:
                    return "irritated"
                elif state.affection >= 80:
                    return "intimate"
                elif state.affection >= 50:
                    return "friendly"
                elif state.affection <= -50:
                    return "hostile"
        fatigue_info = self._get_fatigue_info(stream_id)
        if fatigue_info.get("fatigue_level", 0) > 80:
            return "exhausted"
        return "normal"

    def _get_fatigue_info(self, stream_id: str) -> dict:
        try:
            from src.modules.modcore.fatigue.fatigue_controller import get_fatigue_controller
            controller = get_fatigue_controller()
            return controller.get_fatigue_state(stream_id) or {}
        except Exception:
            return {}

    def _get_perception_context(self, stream_id: str, user_id: str) -> dict:
        try:
            from src.modules.modcore.perception.perception_generator import get_perception_generator
            generator = get_perception_generator()
            return generator.generate_perception(stream_id, user_id) or {}
        except Exception:
            return {}

    def is_interaction_rejected(self, stream_id: str, user_id: str) -> bool:
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            blocked, _ = tracker.is_user_blocked(user_id)
            if blocked:
                return True
            protected, _ = tracker.is_user_protected(user_id)
            if protected:
                return True
        return False

    def get_stream_psychological_state(self, stream_id: str) -> Dict[str, Any]:
        result = {"active_persona": {}, "global_rage": 0.0, "atmosphere": "normal"}
        try:
            from src.modules.modcore.dynamic_persona.persona_switcher import get_persona_switcher
            switcher = get_persona_switcher()
            active_info = switcher.get_active_info(stream_id)
            if active_info:
                result["active_persona"] = active_info
                persona_obj = active_info.get("persona_obj")
                if persona_obj:
                    result["active_persona"]["inner_chaos_level"] = getattr(persona_obj, "inner_chaos_level", 0)
                    trauma_state = getattr(persona_obj, "trauma_state", None)
                    result["active_persona"]["trauma_state_name"] = str(trauma_state.value) if hasattr(trauma_state, 'value') else str(trauma_state)
            tracker = self._get_emotion_tracker(stream_id)
            if tracker:
                result["global_rage"] = getattr(tracker, "global_rage_level", 0.0)
        except Exception as e:
            logger.debug(f"获取频道心理状态快照失败: {e}")
        return result

    async def handle_safety_violation(self, user_id: str, stream_id: str, content: str,
                                       safety_ratings: List[Dict[str, Any]]) -> PsychologicalResponse:
        response = PsychologicalResponse()
        harassment_probability = "NEGLIGIBLE"
        for rating in safety_ratings:
            if rating.get("category") == "HARM_CATEGORY_HARASSMENT":
                harassment_probability = rating.get("probability", "NEGLIGIBLE")
                break
        if harassment_probability not in ["HIGH", "MEDIUM"]:
            return response
        intensity_map = {"HIGH": 8.0, "MEDIUM": 5.0, "LOW": 2.0, "NEGLIGIBLE": 0.0}
        harassment_intensity = intensity_map.get(harassment_probability, 3.0)
        logger.warning(f"安全检测 {user_id[:8]} 触发HARASSMENT={harassment_probability} 强度={harassment_intensity}")
        tracker = self._get_emotion_tracker(stream_id)
        if tracker:
            try:
                state = tracker.get_user_state(user_id, False)
                if state:
                    old_trauma = state.trauma_score
                    old_annoyance = state.annoyance
                    trauma_increase = min(2.0, harassment_intensity * 0.15)
                    state.trauma_score = tracker.clamp_trauma(state.trauma_score + trauma_increase)
                    _ann_base3 = self._bt_get("annoyance_per_harass", 15.0)
                    annoyance_increase = _ann_base3 + harassment_intensity * 3.0
                    state.annoyance = tracker.clamp_annoyance(state.annoyance + annoyance_increase)
                    response.emotion_changes = {
                        "annoyance_delta": state.annoyance - old_annoyance,
                        "trauma_delta": state.trauma_score - old_trauma,
                        "annoyance": state.annoyance, "trauma_score": state.trauma_score,
                    }
                    trauma_record_level = max(1, min(6, int(harassment_intensity / 2)))
                    tracker.add_trauma_record(state, content, trauma_record_level, user_id, trigger_type="safety_violation")
                    tracker._save_states()
            except Exception as e:
                logger.debug(f"更新情绪状态失败: {e}")
        trauma_score = 0.0
        if tracker:
            try:
                state = tracker.get_user_state(user_id, False)
                if state:
                    trauma_score = state.trauma_score
            except Exception:
                pass
        should_switch = False
        target_state = ""
        _sw_high2 = self._bt_get("harassment_switch_threshold_high", 8.0)
        _sw_mid2 = self._bt_get("harassment_switch_threshold_mid", 5.0)
        _sw_low2 = self._bt_get("harassment_switch_threshold_low", 3.0)
        if harassment_intensity >= _sw_high2:
            should_switch = True
            target_state = "应激爆发状态"
        elif harassment_intensity >= _sw_mid2 and trauma_score >= 2.0:
            should_switch = True
            target_state = "伪装动摇状态"
        elif harassment_intensity >= _sw_low2 and trauma_score >= 4.0:
            should_switch = True
            target_state = "勉强伪装状态"
        if should_switch and self._persona_generator:
            try:
                switched = await self._persona_generator.switch_persona_state(
                    stream_id=stream_id,
                    state_name=target_state,
                    trigger_reason=f"安全防御: HARASSMENT={harassment_probability} (强度:{harassment_intensity:.1f})",
                    extra_data={"safety_source": "api", "harassment_probability": harassment_probability, "intensity": harassment_intensity}
                )
                if switched:
                    response.should_switch_persona = True
                    response.target_persona = target_state
                    response.persona_duration = self._bt_get("persona_switch_duration", 600.0)
                    response.actions.append({"type": "safety_persona_switch", "target": target_state})
                    logger.info(f"安全检测触发人格切换 -> {target_state}")
            except Exception as e:
                logger.debug(f"人格切换失败: {e}")
        response.response_mode = "defensive"
        return response

    def detect_harassment_unified(self, user_id: str, channel_id: str, content: str) -> Tuple[bool, Dict[str, Any]]:
        if self._harassment_detector:
            is_harassment, details = self._harassment_detector.detect_harassment(user_id, channel_id, content)
            if is_harassment and details:
                try:
                    sensitive_words = details.get("matched_patterns", [])
                    total_intensity = details.get("total_intensity", 0)
                    from src.memory_system.memory_consolidator import get_short_term_memory
                    stm = get_short_term_memory()
                    if hasattr(stm, 'add_user_cross_context_learning'):
                        for word in sensitive_words:
                            stage = "第3次+" if total_intensity >= 15 else "第2次" if total_intensity >= 8 else "第1次"
                            stm.add_user_cross_context_learning(
                                user_id=user_id, topic=f"敏感词:{word}",
                                count=int(total_intensity), stage=stage
                            )
                except Exception as e:
                    logger.debug(f"学习机制更新失败: {e}")
            return is_harassment, details
        return False, {}

    def get_channel_psychological_state(self, channel_id: str) -> Dict[str, Any]:
        result = {"active_persona": {}, "global_rage": 0.0, "atmosphere": "normal"}
        try:
            from src.modules.modcore.dynamic_persona.persona_switcher import get_persona_switcher
            switcher = get_persona_switcher()
            active_info = switcher.get_active_info(channel_id)
            if active_info:
                result["active_persona"] = active_info
                persona_obj = active_info.get("persona_obj")
                if persona_obj:
                    result["active_persona"]["inner_chaos_level"] = getattr(persona_obj, "inner_chaos_level", 0)
                    trauma_state = getattr(persona_obj, "trauma_state", None)
                    result["active_persona"]["trauma_state_name"] = str(trauma_state.value) if hasattr(trauma_state, 'value') else str(trauma_state)
            tracker = self._get_emotion_tracker(channel_id)
            if tracker:
                result["global_rage"] = getattr(tracker, "global_rage_level", 0.0)
        except Exception as e:
            logger.debug(f"获取频道心理状态快照失败: {e}")
        return result

    def get_combined_psychological_hint(self, channel_id: str, user_id: str) -> str:
        try:
            tracker = self._get_emotion_tracker(channel_id)
            if not tracker:
                return ""
            state = tracker.get_user_state(user_id, False)
            if not state:
                return ""
            parts = []
            if state.affection >= 50:
                parts.append(f"好感度很高({state.affection:.0f})，态度温暖亲切")
            elif state.affection <= -30:
                parts.append(f"好感度较低({state.affection:.0f})，态度冷淡")
            if state.annoyance > 40:
                parts.append(f"烦躁度较高({state.annoyance:.0f})，可能不耐烦")
            if state.trauma_score > 5:
                parts.append(f"创伤分数较高({state.trauma_score:.1f})，回避敏感话题")
            return "；".join(parts) if parts else ""
        except Exception:
            return ""

    async def handle_gemini_safety_violation(self, user_id: str, channel_id: str,
                                              content: str, safety_ratings: List[Dict[str, Any]]) -> PsychologicalResponse:
        response = PsychologicalResponse()
        harassment_probability = "NEGLIGIBLE"
        for rating in safety_ratings:
            if rating.get("category") == "HARM_CATEGORY_HARASSMENT":
                harassment_probability = rating.get("probability", "NEGLIGIBLE")
                break
        if harassment_probability not in ["HIGH", "MEDIUM"]:
            return response
        intensity_map = {"HIGH": 8.0, "MEDIUM": 5.0, "LOW": 2.0, "NEGLIGIBLE": 0.0}
        harassment_intensity = intensity_map.get(harassment_probability, 3.0)
        logger.warning(f"安全检测 用户{user_id[:8]}触发HARASSMENT={harassment_probability}，强度={harassment_intensity}")
        tracker = self._get_emotion_tracker(channel_id)
        if tracker:
            try:
                state = tracker.get_user_state(user_id, False)
                if state:
                    old_trauma = state.trauma_score
                    old_annoyance = state.annoyance
                    trauma_increase = min(2.0, harassment_intensity * 0.15)
                    state.trauma_score = tracker.clamp_trauma(state.trauma_score + trauma_increase)
                    _ann_base2 = self._bt_get("annoyance_per_harass", 15.0)
                    annoyance_increase = _ann_base2 + harassment_intensity * 3.0
                    state.annoyance = tracker.clamp_annoyance(state.annoyance + annoyance_increase)
                    response.emotion_changes = {
                        "annoyance_delta": state.annoyance - old_annoyance,
                        "trauma_delta": state.trauma_score - old_trauma,
                        "annoyance": state.annoyance, "trauma_score": state.trauma_score,
                    }
                    trauma_record_level = max(1, min(6, int(harassment_intensity / 2)))
                    tracker.add_trauma_record(state, content, trauma_record_level, user_id, trigger_type="gemini_safety")
                    tracker._save_states()
            except Exception as e:
                logger.debug(f"更新情绪状态失败: {e}")
        trauma_score = 0.0
        if tracker:
            try:
                state = tracker.get_user_state(user_id, False)
                if state:
                    trauma_score = state.trauma_score
            except Exception:
                pass
        should_switch = False
        target_state = ""
        _sw_high = self._bt_get("harassment_switch_threshold_high", 8.0)
        _sw_mid = self._bt_get("harassment_switch_threshold_mid", 5.0)
        _sw_low = self._bt_get("harassment_switch_threshold_low", 3.0)
        if harassment_intensity >= _sw_high:
            should_switch = True
            target_state = "应激爆发状态"
        elif harassment_intensity >= _sw_mid and trauma_score >= 2.0:
            should_switch = True
            target_state = "伪装动摇状态"
        elif harassment_intensity >= _sw_low and trauma_score >= 4.0:
            should_switch = True
            target_state = "勉强伪装状态"
        if should_switch and self._persona_generator:
            try:
                _duration = self._bt_get("safety_violation_persona_duration", 600.0)
                switched = await self._persona_generator.switch_persona_state(
                    channel_id=channel_id, state_name=target_state,
                    trigger_reason=f"安全防御: HARASSMENT={harassment_probability} (强度:{harassment_intensity:.1f})",
                    extra_data={"safety_source": "gemini", "harassment_probability": harassment_probability, "intensity": harassment_intensity}
                )
                if switched:
                    response.should_switch_persona = True
                    response.target_persona = target_state
                    response.persona_duration = self._bt_get("persona_switch_duration", 600.0)
                    response.actions.append({"type": "safety_persona_switch", "target": target_state})
            except Exception as e:
                logger.debug(f"人格切换失败: {e}")
        response.response_mode = "defensive"
        return response


_psychological_core: Optional[PsychologicalCore] = None


def get_psychological_core() -> PsychologicalCore:
    global _psychological_core
    if _psychological_core is None:
        _psychological_core = PsychologicalCore()
    return _psychological_core


async def handle_harassment_unified(stream_id: str, user_id: str, content: str,
                                     harassment_details: Dict[str, Any]) -> PsychologicalResponse:
    core = get_psychological_core()
    event = PsychologicalEvent(
        event_type="harassment", stream_id=stream_id, user_id=user_id,
        content=content, intensity=harassment_details.get("total_intensity", 5.0),
        metadata={"harassment_details": harassment_details}
    )
    return await core.process_event(event)


async def handle_message_unified(stream_id: str, user_id: str, content: str) -> PsychologicalResponse:
    core = get_psychological_core()
    event = PsychologicalEvent(
        event_type="message", stream_id=stream_id, user_id=user_id,
        content=content, metadata={}
    )
    return await core.process_event(event)


def get_psychological_state(stream_id: str, user_id: str) -> Dict[str, Any]:
    core = get_psychological_core()
    return core.get_user_psychological_state(stream_id, user_id)


def get_response_mode(stream_id: str, user_id: str) -> str:
    core = get_psychological_core()
    return core.get_response_mode(stream_id, user_id)


def generate_perception_context(stream_id: str, user_id: str,
                                 include_flashback: bool = True) -> Dict[str, Any]:
    core = get_psychological_core()
    tracker = core._get_emotion_tracker(stream_id)
    if not tracker:
        return {"physical_sensation": "", "cognitive_state": "", "emotional_pressure": "", "social_willingness": "", "subconscious_interference": "", "prompt_text": ""}
    state = tracker.get_user_state(user_id, False)
    if not state:
        return {"physical_sensation": "", "cognitive_state": "", "emotional_pressure": "", "social_willingness": "", "subconscious_interference": "", "prompt_text": ""}
    trauma_system = core._get_trauma_system(user_id)
    inner_chaos = 0.0
    surface_mask = 10.0
    flashback_fragments = []
    if trauma_system:
        try:
            inner_chaos = getattr(trauma_system, 'inner_chaos_level', 0.0)
            surface_mask = getattr(trauma_system, 'surface_mask_strength', 10.0)
            if include_flashback and hasattr(trauma_system, 'trauma_fragments'):
                fragments = getattr(trauma_system, 'trauma_fragments', [])
                flashback_fragments = []
                for f in fragments:
                    tw = getattr(f, 'trigger_words', [])
                    if tw:
                        flashback_fragments.append(tw[0])
                    if len(flashback_fragments) >= 3:
                        break
        except Exception as e:
            logger.debug(f"获取创伤系统数据失败: {e}")
    physical = ""
    cognitive = ""
    emotional = ""
    social = ""
    subconscious = ""
    if state.mental_fatigue > 70:
        physical = "身体感到沉重，反应变慢，需要休息"
    elif state.mental_fatigue > 40:
        physical = "有些疲惫，注意力偶尔分散"
    if state.trauma_score >= 7:
        cognitive = "思维混乱，难以集中注意力，过去的痛苦记忆不断涌来"
    elif state.trauma_score >= 4:
        cognitive = "心思不宁，偶尔闪过不好的回忆"
    if state.annoyance > 70:
        emotional = "内心极度烦躁，压力即将爆发"
    elif state.annoyance > 40:
        emotional = "感到不耐烦，情绪有些波动"
    if state.affection > 60:
        social = "很愿意交流，对对方充满好感"
    elif state.affection < -30:
        social = "不太想搭理，想要保持距离"
    elif state.mental_fatigue > 60:
        social = "有些累了，不太想多说话"
    if flashback_fragments:
        subconscious = f"脑海中隐约浮现模糊的不好回忆：{'、'.join(flashback_fragments[:2])}"
    parts = []
    if physical:
        parts.append(f"[生理感受] {physical}")
    if cognitive:
        parts.append(f"[认知状态] {cognitive}")
    if emotional:
        parts.append(f"[情绪压力] {emotional}")
    if social:
        parts.append(f"[社交意愿] {social}")
    if subconscious:
        parts.append(f"[潜意识] {subconscious}")
    prompt_text = "\n".join(parts) if parts else ""
    return {
        "physical_sensation": physical, "cognitive_state": cognitive,
        "emotional_pressure": emotional, "social_willingness": social,
        "subconscious_interference": subconscious, "prompt_text": prompt_text
    }


def check_social_fatigue(stream_id: str, user_id: str,
                          force_response: bool = False) -> Dict[str, Any]:
    core = get_psychological_core()
    tracker = core._get_emotion_tracker(stream_id)
    default_result = {"should_respond": True, "willingness_score": 1.0, "reason": "默认回复", "suggested_length": "normal", "length_description": "正常长度"}
    if not tracker:
        return default_result
    state = tracker.get_user_state(user_id, False)
    if not state:
        return default_result
    try:
        from src.modules.modcore.fatigue.fatigue_controller import FatigueBasedResponseController
        fatigue_ctrl = FatigueBasedResponseController()
        should, reason = fatigue_ctrl.should_respond(
            stream_id=stream_id, user_id=user_id,
            mental_fatigue=state.mental_fatigue,
            trauma_score=state.trauma_score,
            annoyance=state.annoyance, force=force_response,
        )
        willingness_result = fatigue_ctrl.calculate_response_willingness(
            stream_id=stream_id, user_id=user_id,
            mental_fatigue=state.mental_fatigue,
            trauma_score=state.trauma_score,
            annoyance=state.annoyance,
        )
        willingness = willingness_result[0] if isinstance(willingness_result, tuple) else willingness_result
        length_cat, length_desc = fatigue_ctrl.suggest_response_length(
            mental_fatigue=state.mental_fatigue, trauma_score=state.trauma_score,
        )
        return {
            "should_respond": should, "willingness_score": willingness,
            "reason": reason, "suggested_length": length_cat,
            "length_description": length_desc
        }
    except Exception as e:
        logger.error(f"检查社交疲劳失败: {e}")
        return default_result



def detect_harassment_unified(user_id: str, channel_id: str, content: str) -> Tuple[bool, Dict[str, Any]]:
    """统一的骚扰检测接口
    
    Args:
        user_id: 用户ID
        channel_id: 频道/群组ID
        content: 消息内容
    
    Returns:
        (是否为骚扰, 骚扰详情)
    """
    core = get_psychological_core()
    if not core._harassment_detector:
        return False, {}
    
    return core._harassment_detector.detect_harassment(user_id, channel_id, content)


async def handle_harassment_unified(
    stream_id: str,
    user_id: str,
    content: str,
    harassment_details: Dict[str, Any]
) -> PsychologicalResponse:
    """统一的骚扰处理接口
    
    Args:
        stream_id: 聊天流ID
        user_id: 用户ID
        content: 消息内容
        harassment_details: 骚扰检测详情
    
    Returns:
        心理响应对象
    """
    core = get_psychological_core()
    event = PsychologicalEvent(
        event_type="harassment",
        stream_id=stream_id,
        user_id=user_id,
        content=content,
        intensity=harassment_details.get("total_intensity", 1.0),
        metadata={"harassment_details": harassment_details}
    )
    return await core.process_event(event)

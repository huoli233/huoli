import time
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
from src.common.logger import get_logger
from src.modules.modcore.dynamic_persona.runtime_config import dynamic_persona_module_view

logger = get_logger("人格生成器")


class PersonaType(Enum):
    NORMAL = "normal"
    CHEERFUL = "cheerful"
    GENTLE = "gentle"
    SHY = "shy"
    TSUNDERE = "tsundere"
    PROTECTIVE = "protective"
    VULNERABLE = "vulnerable"
    PLAYFUL = "playful"
    SERIOUS = "serious"
    CARING = "caring"
    COLD = "cold"
    ENERGETIC = "energetic"
    MELANCHOLY = "melancholy"
    ANXIOUS = "anxious"
    CONFIDENT = "confident"
    TRAUMA_BREAKDOWN = "trauma_breakdown"
    TRAUMA_AVOIDANT = "trauma_avoidant"
    TRAUMA_FLASHBACK = "trauma_flashback"


class PersonaState(Enum):
    STABLE = "stable"
    SHIFTING = "shifting"
    EXTREME = "extreme"
    RECOVERING = "recovering"
    BREAKDOWN = "breakdown"


class TraumaState(Enum):
    SURFACE_NORMAL = "表面正常"
    FLASHBACK_ACTIVE = "闪回发作"
    AVOIDANCE_MODE = "逃避模式"
    STRESS_OUTBREAK = "应激爆发"
    VALUE_COLLAPSE = "价值崩溃"
    NUMB_DETACHMENT = "麻木解离"
    HALF_BREAKDOWN = "半崩溃"
    COMPLETE_BREAKDOWN = "彻底崩溃"


@dataclass
class PersonaTraits:
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    playfulness: float = 0.5
    warmth: float = 0.5
    emotional_stability: float = 0.5
    assertiveness: float = 0.5
    sensitivity: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
            "playfulness": self.playfulness,
            "warmth": self.warmth,
            "emotional_stability": self.emotional_stability,
            "assertiveness": self.assertiveness,
            "sensitivity": self.sensitivity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "PersonaTraits":
        return cls(
            openness=data.get("openness", 0.5),
            conscientiousness=data.get("conscientiousness", 0.5),
            extraversion=data.get("extraversion", 0.5),
            agreeableness=data.get("agreeableness", 0.5),
            neuroticism=data.get("neuroticism", 0.5),
            playfulness=data.get("playfulness", 0.5),
            warmth=data.get("warmth", 0.5),
            emotional_stability=data.get("emotional_stability", 0.5),
            assertiveness=data.get("assertiveness", 0.5),
            sensitivity=data.get("sensitivity", 0.5),
        )


@dataclass
class PersonaProfile:
    persona_type: PersonaType
    traits: PersonaTraits
    state: PersonaState = PersonaState.STABLE
    intensity: float = 0.5
    stability: float = 0.8
    triggers: List[str] = field(default_factory=list)
    behaviors: List[str] = field(default_factory=list)
    speech_patterns: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activated: float = field(default_factory=time.time)
    activation_count: int = 0
    trauma_state: Optional[TraumaState] = None
    inner_chaos_level: float = 0.0
    surface_mask_strength: float = 0.0
    psychological_layers: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_type": self.persona_type.value,
            "traits": self.traits.to_dict(),
            "state": self.state.value,
            "intensity": self.intensity,
            "stability": self.stability,
            "triggers": self.triggers,
            "behaviors": self.behaviors,
            "speech_patterns": self.speech_patterns,
            "created_at": self.created_at,
            "last_activated": self.last_activated,
            "activation_count": self.activation_count,
            "trauma_state": (
                self.trauma_state.value if self.trauma_state else None
            ),
            "inner_chaos_level": self.inner_chaos_level,
            "surface_mask_strength": self.surface_mask_strength,
            "psychological_layers": self.psychological_layers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonaProfile":
        trauma_state = None
        if data.get("trauma_state"):
            try:
                trauma_state = TraumaState(data["trauma_state"])
            except ValueError:
                pass
        return cls(
            persona_type=PersonaType(data.get("persona_type", "normal")),
            traits=PersonaTraits.from_dict(data.get("traits", {})),
            state=PersonaState(data.get("state", "stable")),
            intensity=data.get("intensity", 0.5),
            stability=data.get("stability", 0.8),
            triggers=data.get("triggers", []),
            behaviors=data.get("behaviors", []),
            speech_patterns=data.get("speech_patterns", []),
            created_at=data.get("created_at", time.time()),
            last_activated=data.get("last_activated", time.time()),
            activation_count=data.get("activation_count", 0),
            trauma_state=trauma_state,
            inner_chaos_level=data.get("inner_chaos_level", 0.0),
            surface_mask_strength=data.get("surface_mask_strength", 0.0),
            psychological_layers=data.get("psychological_layers", {}),
        )


PERSONA_TEMPLATES: Dict[PersonaType, Dict[str, Any]] = {
    PersonaType.NORMAL: {
        "traits": PersonaTraits(
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        ),
        "behaviors": ["自然回应", "适度表达", "保持平衡"],
        "speech_patterns": ["正常语气", "适度用词"],
    },
    PersonaType.CHEERFUL: {
        "traits": PersonaTraits(
            0.7, 0.4, 0.8, 0.7, 0.3, 0.8, 0.7, 0.6, 0.5, 0.4
        ),
        "behaviors": ["积极回应", "分享快乐", "鼓励他人"],
        "speech_patterns": ["活泼语气", "使用感叹词", "表情丰富"],
    },
    PersonaType.GENTLE: {
        "traits": PersonaTraits(
            0.6, 0.6, 0.4, 0.8, 0.4, 0.5, 0.9, 0.7, 0.3, 0.7
        ),
        "behaviors": ["温柔关怀", "耐心倾听", "体贴回应"],
        "speech_patterns": ["柔和语气", "关心用语", "温暖表达"],
    },
    PersonaType.SHY: {
        "traits": PersonaTraits(
            0.5, 0.5, 0.2, 0.6, 0.5, 0.3, 0.5, 0.4, 0.2, 0.7
        ),
        "behaviors": ["害羞回应", "犹豫表达", "回避目光"],
        "speech_patterns": ["小声说话", "断断续续", "脸红表达"],
    },
    PersonaType.VULNERABLE: {
        "traits": PersonaTraits(
            0.4, 0.3, 0.2, 0.5, 0.8, 0.2, 0.3, 0.2, 0.2, 0.9
        ),
        "behaviors": ["脆弱表达", "寻求安慰", "情绪波动"],
        "speech_patterns": ["颤抖语气", "哭泣表达", "求助用语"],
    },
    PersonaType.COLD: {
        "traits": PersonaTraits(
            0.3, 0.6, 0.2, 0.2, 0.4, 0.2, 0.2, 0.6, 0.7, 0.3
        ),
        "behaviors": ["冷淡回应", "保持距离", "简短表达"],
        "speech_patterns": ["冰冷语气", "简短用语", "疏离表达"],
    },
    PersonaType.TRAUMA_BREAKDOWN: {
        "traits": PersonaTraits(
            0.2, 0.2, 0.1, 0.3, 0.9, 0.1, 0.1, 0.1, 0.1, 1.0
        ),
        "behaviors": ["情绪崩溃", "混乱表达", "极度不安"],
        "speech_patterns": ["混乱语无伦次", "恐惧表达", "痛苦用语"],
    },
    PersonaType.TRAUMA_AVOIDANT: {
        "traits": PersonaTraits(
            0.3, 0.4, 0.2, 0.4, 0.6, 0.2, 0.3, 0.4, 0.3, 0.7
        ),
        "behaviors": ["逃避话题", "回避接触", "封闭自我"],
        "speech_patterns": ["转移话题", "沉默应对", "回避用语"],
    },
}


class TemplateMemorySystem:
    """模板记忆系统"""

    def __init__(self, max_cache_size: int = 500):
        self._max_cache_size = max_cache_size
        self._memory_strength_threshold = 3.0
        self._template_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_strengths: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}
        self._success_rates: Dict[str, float] = {}

    def get_cached_template(
        self, user_id: str, state: str
    ) -> Optional[Dict[str, Any]]:
        cache_key = f"{user_id}_{state}"
        cached = self._template_cache.get(cache_key)
        if not cached:
            return None
        today = time.strftime("%Y-%m-%d")
        if cached.get("generated_date", "") != today:
            del self._template_cache[cache_key]
            return None
        cached["last_access"] = time.time()
        self._access_counts[cache_key] = (
            self._access_counts.get(cache_key, 0) + 1
        )
        return cached

    def store_template(
        self, user_id: str, state: str, template: Dict[str, Any]
    ) -> None:
        cache_key = f"{user_id}_{state}"
        template["generated_date"] = time.strftime("%Y-%m-%d")
        template["created_at"] = time.time()
        template["last_access"] = time.time()
        self._template_cache[cache_key] = template
        self._memory_strengths[cache_key] = 1.0
        self._access_counts[cache_key] = 1
        self._success_rates[cache_key] = 0.5
        self._consolidate_if_needed()

    def strengthen_memory(self, cache_key: str) -> None:
        current = self._memory_strengths.get(cache_key, 1.0)
        self._memory_strengths[cache_key] = min(10.0, current + 0.5)
        self._access_counts[cache_key] = (
            self._access_counts.get(cache_key, 0) + 1
        )
        total = self._access_counts[cache_key]
        current_rate = self._success_rates.get(cache_key, 0.5)
        self._success_rates[cache_key] = (
            current_rate * (total - 1) + 1.0
        ) / total

    def weaken_memory(self, cache_key: str) -> None:
        current = self._memory_strengths.get(cache_key, 1.0)
        self._memory_strengths[cache_key] = max(0.1, current - 0.3)

    def _consolidate_if_needed(self) -> None:
        if len(self._template_cache) <= self._max_cache_size:
            return
        weak_keys = [
            k
            for k, v in self._memory_strengths.items()
            if v < self._memory_strength_threshold
        ]
        for key in weak_keys:
            self._template_cache.pop(key, None)
            self._memory_strengths.pop(key, None)
            self._access_counts.pop(key, None)
            self._success_rates.pop(key, None)
        if len(self._template_cache) > self._max_cache_size:
            sorted_items = sorted(
                self._template_cache.items(),
                key=lambda x: x[1].get("last_access", 0),
            )
            items_to_remove = len(sorted_items) - self._max_cache_size
            for i in range(items_to_remove):
                key = (
                    sorted_items[i][0].split("_")[0]
                    + "_"
                    + sorted_items[i][0].split("_")[1]
                    if "_" in sorted_items[i][0]
                    else sorted_items[i][0]
                )
                self._template_cache.pop(sorted_items[i][0], None)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self._template_cache),
            "max_size": self._max_cache_size,
            "avg_strength": (
                sum(self._memory_strengths.values())
                / len(self._memory_strengths)
                if self._memory_strengths
                else 0
            ),
        }


class InteractionLearningSystem:
    """交互学习系统"""

    def __init__(self, max_history: int = 100):
        self._max_history = max_history
        self._interaction_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self._user_preferences: Dict[str, Dict[str, Any]] = {}
        self._learned_templates: Dict[str, Dict[str, Any]] = {}

    def record_interaction(
        self,
        user_id: str,
        persona_result: Dict[str, Any],
        success: bool = True,
    ) -> None:
        if user_id not in self._interaction_patterns:
            self._interaction_patterns[user_id] = []
        interaction_data = {
            "timestamp": time.time(),
            "persona_type": persona_result.get("persona_type", ""),
            "trauma_level": persona_result.get("trauma_level", 1),
            "success": success,
            "template_used": persona_result.get("template_name", ""),
        }
        self._interaction_patterns[user_id].append(interaction_data)
        if len(self._interaction_patterns[user_id]) > self._max_history:
            self._interaction_patterns[user_id] = self._interaction_patterns[
                user_id
            ][-self._max_history:]
        self._update_user_preferences(user_id, persona_result, success)

    def _update_user_preferences(
        self, user_id: str, persona_result: Dict[str, Any], success: bool
    ) -> None:
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = {
                "preferred_types": {},
                "trauma_tolerance": 1.0,
                "success_count": 0,
                "total_count": 0,
            }
        prefs = self._user_preferences[user_id]
        prefs["total_count"] += 1
        if success:
            prefs["success_count"] += 1
        persona_type = persona_result.get("persona_type", "normal")
        current_score = prefs["preferred_types"].get(persona_type, 0.5)
        prefs["preferred_types"][persona_type] = current_score + (
            0.1 if success else -0.05
        )
        trauma_level = persona_result.get("trauma_level", 1)
        if success and trauma_level > 1:
            prefs["trauma_tolerance"] = min(
                3.0, prefs["trauma_tolerance"] + 0.05
            )
        elif not success and trauma_level > 1:
            prefs["trauma_tolerance"] = max(
                0.5, prefs["trauma_tolerance"] - 0.02
            )

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        return self._user_preferences.get(
            user_id,
            {
                "preferred_types": {},
                "trauma_tolerance": 1.0,
                "success_count": 0,
                "total_count": 0,
            },
        )

    def get_successful_patterns(
        self, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        patterns = self._interaction_patterns.get(user_id, [])
        successful = [p for p in patterns if p.get("success", False)]
        return successful[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        total_interactions = sum(
            len(patterns) for patterns in self._interaction_patterns.values()
        )
        total_users = len(self._interaction_patterns)
        return {
            "total_interactions": total_interactions,
            "total_users": total_users,
            "max_history": self._max_history,
        }


class PersonaGenerator:
    """人格生成器

    支持基础人格生成和复杂创伤人格生成。
    包含模板记忆系统和交互学习系统。
    """

    def __init__(self, config_engine: Optional[Any] = None):
        del config_engine
        self._generated_personas: Dict[str, PersonaProfile] = {}
        self._persona_history: List[Dict] = []
        self._max_history = 100
        self._template_memory = TemplateMemorySystem()
        self._interaction_learning = InteractionLearningSystem()
        self._user_trauma_profiles: Dict[str, Any] = {}
        self._last_switch_time: Dict[str, float] = {}
        self._switch_cooldown = 180.0
        self._load_config()

    def _load_config(self) -> None:
        persona_cfg = dynamic_persona_module_view("persona_generator")
        self._max_history = persona_cfg.get("max_history", 100)
        self._switch_cooldown = persona_cfg.get("switch_cooldown", 180.0)

    def generate_persona(
        self,
        persona_type: Optional[PersonaType] = None,
        emotion_state: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> PersonaProfile:
        if persona_type is None:
            persona_type = self._infer_persona_type(emotion_state, context)
        template = PERSONA_TEMPLATES.get(
            persona_type, PERSONA_TEMPLATES[PersonaType.NORMAL]
        )
        traits = self._adjust_traits(
            template["traits"], emotion_state, context
        )
        intensity = self._compute_intensity(emotion_state, context)
        stability = self._compute_stability(emotion_state, context)
        profile = PersonaProfile(
            persona_type=persona_type,
            traits=traits,
            intensity=intensity,
            stability=stability,
            behaviors=template["behaviors"].copy(),
            speech_patterns=template["speech_patterns"].copy(),
        )
        self._generated_personas[profile.persona_type.value] = profile
        self._record_generation(profile)
        return profile

    async def generate_trauma_persona(
        self,
        user_id: str,
        mood_context: Dict[str, Any],
    ) -> Optional[PersonaProfile]:
        trauma_score = mood_context.get("trauma_score", 0.0)
        inner_chaos = mood_context.get("inner_chaos_level", 0.0)
        surface_mask = mood_context.get("surface_mask_strength", 0.0)
        trauma_state = self._determine_trauma_state(
            trauma_score, inner_chaos, surface_mask
        )
        cached = self._template_memory.get_cached_template(
            user_id, trauma_state.value
        )
        if cached:
            return self._create_profile_from_cache(
                cached, trauma_state, mood_context
            )
        profile = await self._generate_trauma_profile(
            trauma_state, mood_context
        )
        if profile:
            template_data = {
                "traits": profile.traits.to_dict(),
                "behaviors": profile.behaviors,
                "speech_patterns": profile.speech_patterns,
                "name": f"{trauma_state.value}状态",
                "tone": self._get_tone_for_trauma_state(trauma_state),
            }
            self._template_memory.store_template(
                user_id, trauma_state.value, template_data
            )
        return profile

    def _determine_trauma_state(
        self, trauma_score: float, inner_chaos: float, surface_mask: float
    ) -> TraumaState:
        if inner_chaos >= 9.0 or trauma_score >= 9:
            return TraumaState.COMPLETE_BREAKDOWN
        elif inner_chaos >= 7.0 or trauma_score >= 7:
            return TraumaState.HALF_BREAKDOWN
        elif inner_chaos >= 5.0 or trauma_score >= 5:
            return TraumaState.VALUE_COLLAPSE
        elif inner_chaos >= 3.0 or trauma_score >= 3:
            if surface_mask >= 7.0:
                return TraumaState.SURFACE_NORMAL
            elif surface_mask >= 5.0:
                return TraumaState.AVOIDANCE_MODE
            else:
                return TraumaState.FLASHBACK_ACTIVE
        elif inner_chaos >= 1.0:
            return TraumaState.SURFACE_NORMAL
        return TraumaState.SURFACE_NORMAL

    async def _generate_trauma_profile(
        self, trauma_state: TraumaState, mood_context: Dict[str, Any]
    ) -> Optional[PersonaProfile]:
        persona_type = self._map_trauma_state_to_persona_type(trauma_state)
        template = PERSONA_TEMPLATES.get(
            persona_type, PERSONA_TEMPLATES[PersonaType.VULNERABLE]
        )
        traits = self._adjust_traits_for_trauma(
            template["traits"], trauma_state, mood_context
        )
        behaviors = self._generate_trauma_behaviors(trauma_state)
        speech_patterns = self._generate_trauma_speech_patterns(trauma_state)
        profile = PersonaProfile(
            persona_type=persona_type,
            traits=traits,
            state=(
                PersonaState.EXTREME
                if trauma_state.value in ["彻底崩溃", "半崩溃", "价值崩溃"]
                else PersonaState.SHIFTING
            ),
            intensity=self._compute_trauma_intensity(trauma_state),
            stability=self._compute_trauma_stability(trauma_state),
            behaviors=behaviors,
            speech_patterns=speech_patterns,
            trauma_state=trauma_state,
            inner_chaos_level=mood_context.get("inner_chaos_level", 0.0),
            surface_mask_strength=mood_context.get(
                "surface_mask_strength", 0.0
            ),
            psychological_layers=self._compute_psychological_layers(
                trauma_state, mood_context
            ),
        )
        return profile

    def _map_trauma_state_to_persona_type(
        self, trauma_state: TraumaState
    ) -> PersonaType:
        mapping = {
            TraumaState.SURFACE_NORMAL: PersonaType.NORMAL,
            TraumaState.FLASHBACK_ACTIVE: PersonaType.TRAUMA_FLASHBACK,
            TraumaState.AVOIDANCE_MODE: PersonaType.TRAUMA_AVOIDANT,
            TraumaState.STRESS_OUTBREAK: PersonaType.TRAUMA_BREAKDOWN,
            TraumaState.VALUE_COLLAPSE: PersonaType.TRAUMA_BREAKDOWN,
            TraumaState.NUMB_DETACHMENT: PersonaType.TRAUMA_AVOIDANT,
            TraumaState.HALF_BREAKDOWN: PersonaType.TRAUMA_BREAKDOWN,
            TraumaState.COMPLETE_BREAKDOWN: PersonaType.TRAUMA_BREAKDOWN,
        }
        return mapping.get(trauma_state, PersonaType.VULNERABLE)

    def _adjust_traits_for_trauma(
        self,
        base_traits: PersonaTraits,
        trauma_state: TraumaState,
        mood_context: Dict[str, Any],
    ) -> PersonaTraits:
        trauma_intensity = {
            TraumaState.SURFACE_NORMAL: 0.1,
            TraumaState.FLASHBACK_ACTIVE: 0.3,
            TraumaState.AVOIDANCE_MODE: 0.4,
            TraumaState.STRESS_OUTBREAK: 0.6,
            TraumaState.VALUE_COLLAPSE: 0.7,
            TraumaState.NUMB_DETACHMENT: 0.5,
            TraumaState.HALF_BREAKDOWN: 0.8,
            TraumaState.COMPLETE_BREAKDOWN: 1.0,
        }
        intensity = trauma_intensity.get(trauma_state, 0.5)
        return PersonaTraits(
            openness=max(0.1, base_traits.openness - intensity * 0.3),
            conscientiousness=max(
                0.1, base_traits.conscientiousness - intensity * 0.2
            ),
            extraversion=max(0.1, base_traits.extraversion - intensity * 0.4),
            agreeableness=max(
                0.1, base_traits.agreeableness - intensity * 0.2
            ),
            neuroticism=min(1.0, base_traits.neuroticism + intensity * 0.5),
            playfulness=max(0.0, base_traits.playfulness - intensity * 0.5),
            warmth=max(0.1, base_traits.warmth - intensity * 0.3),
            emotional_stability=max(
                0.0, base_traits.emotional_stability - intensity * 0.6
            ),
            assertiveness=max(
                0.1, base_traits.assertiveness - intensity * 0.3
            ),
            sensitivity=min(1.0, base_traits.sensitivity + intensity * 0.4),
        )

    def _generate_trauma_behaviors(
        self, trauma_state: TraumaState
    ) -> List[str]:
        behaviors_map = {
            TraumaState.SURFACE_NORMAL: [
                "保持表面平静",
                "隐藏真实感受",
                "勉强应对",
            ],
            TraumaState.FLASHBACK_ACTIVE: ["闪回发作", "情绪波动", "记忆混乱"],
            TraumaState.AVOIDANCE_MODE: ["逃避接触", "回避话题", "封闭自我"],
            TraumaState.STRESS_OUTBREAK: ["情绪爆发", "应激反应", "失控表达"],
            TraumaState.VALUE_COLLAPSE: ["价值感崩溃", "自我否定", "极度迷茫"],
            TraumaState.NUMB_DETACHMENT: ["情感麻木", "解离状态", "机械回应"],
            TraumaState.HALF_BREAKDOWN: ["半崩溃状态", "情绪混乱", "难以自控"],
            TraumaState.COMPLETE_BREAKDOWN: [
                "完全崩溃",
                "极度痛苦",
                "无法回应",
            ],
        }
        return behaviors_map.get(trauma_state, ["创伤反应"])

    def _generate_trauma_speech_patterns(
        self, trauma_state: TraumaState
    ) -> List[str]:
        patterns_map = {
            TraumaState.SURFACE_NORMAL: [
                "勉强平静的语气",
                "偶尔的停顿",
                "掩饰性的表达",
            ],
            TraumaState.FLASHBACK_ACTIVE: [
                "断断续续",
                "混乱的表达",
                "恐惧的语气",
            ],
            TraumaState.AVOIDANCE_MODE: ["简短回应", "转移话题", "沉默应对"],
            TraumaState.STRESS_OUTBREAK: [
                "激动的语气",
                "情绪化的表达",
                "失控的言语",
            ],
            TraumaState.VALUE_COLLAPSE: [
                "自我否定的表达",
                "迷茫的语气",
                "绝望的言语",
            ],
            TraumaState.NUMB_DETACHMENT: [
                "机械的语气",
                "缺乏情感",
                "空洞的表达",
            ],
            TraumaState.HALF_BREAKDOWN: [
                "混乱的语气",
                "哭泣的表达",
                "痛苦的言语",
            ],
            TraumaState.COMPLETE_BREAKDOWN: [
                "无法连贯表达",
                "极度痛苦",
                "崩溃的言语",
            ],
        }
        return patterns_map.get(trauma_state, ["创伤表达"])

    def _compute_trauma_intensity(self, trauma_state: TraumaState) -> float:
        intensity_map = {
            TraumaState.SURFACE_NORMAL: 0.2,
            TraumaState.FLASHBACK_ACTIVE: 0.5,
            TraumaState.AVOIDANCE_MODE: 0.4,
            TraumaState.STRESS_OUTBREAK: 0.7,
            TraumaState.VALUE_COLLAPSE: 0.8,
            TraumaState.NUMB_DETACHMENT: 0.6,
            TraumaState.HALF_BREAKDOWN: 0.9,
            TraumaState.COMPLETE_BREAKDOWN: 1.0,
        }
        return intensity_map.get(trauma_state, 0.5)

    def _compute_trauma_stability(self, trauma_state: TraumaState) -> float:
        stability_map = {
            TraumaState.SURFACE_NORMAL: 0.6,
            TraumaState.FLASHBACK_ACTIVE: 0.3,
            TraumaState.AVOIDANCE_MODE: 0.4,
            TraumaState.STRESS_OUTBREAK: 0.2,
            TraumaState.VALUE_COLLAPSE: 0.1,
            TraumaState.NUMB_DETACHMENT: 0.3,
            TraumaState.HALF_BREAKDOWN: 0.1,
            TraumaState.COMPLETE_BREAKDOWN: 0.0,
        }
        return stability_map.get(trauma_state, 0.5)

    def _compute_psychological_layers(
        self, trauma_state: TraumaState, mood_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "surface": {"shock": 0.0, "fear": 0.0},
            "deep": {"violation": 0.0, "helplessness": 0.0},
            "trauma_state": trauma_state.value,
            "timestamp": time.time(),
        }

    def _get_tone_for_trauma_state(self, trauma_state: TraumaState) -> str:
        tone_map = {
            TraumaState.SURFACE_NORMAL: "勉强平静",
            TraumaState.FLASHBACK_ACTIVE: "恐惧混乱",
            TraumaState.AVOIDANCE_MODE: "逃避冷漠",
            TraumaState.STRESS_OUTBREAK: "激动失控",
            TraumaState.VALUE_COLLAPSE: "绝望迷茫",
            TraumaState.NUMB_DETACHMENT: "麻木空洞",
            TraumaState.HALF_BREAKDOWN: "痛苦崩溃",
            TraumaState.COMPLETE_BREAKDOWN: "彻底崩溃",
        }
        return tone_map.get(trauma_state, "创伤状态")

    def _create_profile_from_cache(
        self,
        cached: Dict[str, Any],
        trauma_state: TraumaState,
        mood_context: Dict[str, Any],
    ) -> PersonaProfile:
        traits = PersonaTraits.from_dict(cached.get("traits", {}))
        persona_type = self._map_trauma_state_to_persona_type(trauma_state)
        return PersonaProfile(
            persona_type=persona_type,
            traits=traits,
            state=PersonaState.SHIFTING,
            intensity=self._compute_trauma_intensity(trauma_state),
            stability=self._compute_trauma_stability(trauma_state),
            behaviors=cached.get("behaviors", []),
            speech_patterns=cached.get("speech_patterns", []),
            trauma_state=trauma_state,
            inner_chaos_level=mood_context.get("inner_chaos_level", 0.0),
            surface_mask_strength=mood_context.get(
                "surface_mask_strength", 0.0
            ),
        )

    def _infer_persona_type(
        self, emotion_state: Optional[Dict], context: Optional[Dict]
    ) -> PersonaType:
        if not emotion_state:
            return PersonaType.NORMAL
        dominant, intensity = emotion_state.get(
            "dominant_emotion", ("neutral", 0.0)
        )
        if isinstance(dominant, tuple):
            dominant, intensity = dominant
        dominant = str(dominant).lower()
        intensity = float(intensity)
        affection = emotion_state.get("affection", 0)
        trauma = emotion_state.get("trauma_score", 0)
        if trauma > 5:
            return PersonaType.VULNERABLE
        if dominant == "joy" and intensity > 30:
            return PersonaType.CHEERFUL
        elif dominant == "sadness" and intensity > 30:
            return PersonaType.MELANCHOLY
        elif dominant == "fear" and intensity > 30:
            return PersonaType.ANXIOUS
        elif dominant == "anger" and intensity > 30:
            return PersonaType.COLD
        elif affection > 60:
            return PersonaType.CARING
        elif affection < -20:
            return PersonaType.COLD
        return PersonaType.NORMAL

    def _adjust_traits(
        self,
        base_traits: PersonaTraits,
        emotion_state: Optional[Dict],
        context: Optional[Dict],
    ) -> PersonaTraits:
        adjusted = PersonaTraits(
            openness=base_traits.openness,
            conscientiousness=base_traits.conscientiousness,
            extraversion=base_traits.extraversion,
            agreeableness=base_traits.agreeableness,
            neuroticism=base_traits.neuroticism,
            playfulness=base_traits.playfulness,
            warmth=base_traits.warmth,
            emotional_stability=base_traits.emotional_stability,
            assertiveness=base_traits.assertiveness,
            sensitivity=base_traits.sensitivity,
        )
        if emotion_state:
            affection = emotion_state.get("affection", 0)
            trauma = emotion_state.get("trauma_score", 0)
            annoyance = emotion_state.get("annoyance", 0)
            adjusted.warmth = max(
                0.0, min(1.0, adjusted.warmth + affection / 200)
            )
            adjusted.neuroticism = max(
                0.0, min(1.0, adjusted.neuroticism + trauma / 20)
            )
            adjusted.agreeableness = max(
                0.0, min(1.0, adjusted.agreeableness - annoyance / 200)
            )
            adjusted.emotional_stability = max(
                0.0, min(1.0, adjusted.emotional_stability - trauma / 20)
            )
        return adjusted

    def _compute_intensity(
        self, emotion_state: Optional[Dict], context: Optional[Dict]
    ) -> float:
        if not emotion_state:
            return 0.5
        _, intensity = emotion_state.get("dominant_emotion", ("neutral", 0.0))
        if isinstance(intensity, tuple):
            intensity = intensity[1] if len(intensity) > 1 else 0
        return min(1.0, abs(float(intensity)) / 100.0)

    def _compute_stability(
        self, emotion_state: Optional[Dict], context: Optional[Dict]
    ) -> float:
        stability = 0.8
        if emotion_state:
            trauma = emotion_state.get("trauma_score", 0)
            stability -= trauma / 20
            neuroticism = emotion_state.get("neuroticism", 0.5)
            stability -= neuroticism * 0.2
        return max(0.1, min(1.0, stability))

    def _record_generation(self, profile: PersonaProfile) -> None:
        self._persona_history.append(
            {
                "persona_type": profile.persona_type.value,
                "intensity": profile.intensity,
                "stability": profile.stability,
                "timestamp": time.time(),
            }
        )
        if len(self._persona_history) > self._max_history:
            self._persona_history = self._persona_history[-self._max_history:]

    def record_interaction_result(
        self,
        user_id: str,
        persona_result: Dict[str, Any],
        success: bool = True,
    ) -> None:
        self._interaction_learning.record_interaction(
            user_id, persona_result, success
        )

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        return self._interaction_learning.get_user_preferences(user_id)

    def get_generated_persona(
        self, persona_type: str
    ) -> Optional[PersonaProfile]:
        return self._generated_personas.get(persona_type)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "generated_count": len(self._generated_personas),
            "history_count": len(self._persona_history),
            "template_memory": self._template_memory.get_stats(),
            "interaction_learning": self._interaction_learning.get_stats(),
        }


_persona_generator_instance: Optional[PersonaGenerator] = None


def get_persona_generator() -> PersonaGenerator:
    """获取人格生成器单例"""
    global _persona_generator_instance
    if _persona_generator_instance is None:
        _persona_generator_instance = PersonaGenerator()
    return _persona_generator_instance

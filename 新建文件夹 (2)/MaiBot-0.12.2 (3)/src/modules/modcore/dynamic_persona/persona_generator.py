import time
import uuid
import random
import json
from typing import Optional, Dict, List, Any
from src.common.logger import get_logger
from src.modules.modcore.dynamic_persona.persona_controller import DynamicPersona, get_persona_controller
from src.modules.modcore.dynamic_persona.complex_trauma_psychology import ComplexTraumaPsychology, TraumaState

logger = get_logger("persona")


class PersonaGenerator:
    SAFETY_RULES = """【安全限制】
- 禁止讨论成人/色情内容
- 禁止讨论政治敏感话题
- 可以生气但不能说脏话
- 可以表达不满但要有分寸
- 保持少女角色的纯真感
"""

    def __init__(self, config: Optional[Dict] = None):
        self._cfg = config or {}
        self._llm_bridge = None
        self._generated_today: List[str] = []
        self._last_reset_date: str = ""
        self.trauma_psychology = ComplexTraumaPsychology()
        self._user_trauma_profiles: Dict[str, ComplexTraumaPsychology] = {}
        self._template_memory_cache: Dict[str, Dict[str, Any]] = {}
        self._interaction_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self._learned_templates: Dict[str, List[Dict[str, Any]]] = {}
        self._user_preference_memory: Dict[str, Dict[str, Any]] = {}
        self._initialize_template_memory_system()
        self._last_switch_time: Dict[str, float] = {}
        self._last_switched_state: Dict[str, str] = {}
        self._switch_cooldown_seconds: float = 180.0
        self._state_transition_history: Dict[str, List[Dict[str, Any]]] = {}

    def _initialize_template_memory_system(self):
        try:
            self._template_memory_config = {
                "max_cache_size": 500,
                "memory_strength_threshold": 3.0,
                "pattern_similarity_threshold": 0.7,
                "consolidation_interval": 100,
                "max_learned_templates": 50
            }
            self._template_memory_strengths: Dict[str, float] = {}
            self._template_access_count: Dict[str, int] = {}
            self._template_success_rate: Dict[str, float] = {}
            self._user_response_patterns: Dict[str, Dict[str, Any]] = {}
            self._user_persona_preferences: Dict[str, List[str]] = {}
            logger.info("模板记忆系统初始化完成")
        except Exception as e:
            logger.error(f"模板记忆系统初始化失败: {e}")

    def _learn_from_interaction(self, user_id: str, persona_result: Dict[str, Any], success_feedback: bool = True):
        try:
            interaction_key = f"{user_id}_{persona_result.get('persona_type', 'unknown')}"
            current_time = time.time()
            if user_id not in self._interaction_patterns:
                self._interaction_patterns[user_id] = []
            interaction_data = {
                "timestamp": current_time,
                "persona_type": persona_result.get("persona_type", ""),
                "progression_stage": persona_result.get("progression_stage", ""),
                "trauma_level": persona_result.get("trauma_level", 1),
                "success": success_feedback,
                "template_used": persona_result.get("template_name", ""),
                "response_quality": persona_result.get("quality_score", 0.5)
            }
            self._interaction_patterns[user_id].append(interaction_data)
            if len(self._interaction_patterns[user_id]) > 100:
                self._interaction_patterns[user_id] = self._interaction_patterns[user_id][-100:]
            if success_feedback:
                self._strengthen_template_memory(interaction_key, persona_result)
            else:
                self._weaken_template_memory(interaction_key)
            self._update_user_preference_memory(user_id, persona_result, success_feedback)
        except Exception as e:
            logger.error(f"交互学习失败: {e}")

    def _strengthen_template_memory(self, memory_key: str, persona_result: Dict[str, Any]):
        current_strength = self._template_memory_strengths.get(memory_key, 1.0)
        self._template_memory_strengths[memory_key] = min(10.0, current_strength + 0.5)
        self._template_access_count[memory_key] = self._template_access_count.get(memory_key, 0) + 1
        total_access = self._template_access_count[memory_key]
        current_success_rate = self._template_success_rate.get(memory_key, 0.5)
        new_success_rate = (current_success_rate * (total_access - 1) + 1.0) / total_access
        self._template_success_rate[memory_key] = new_success_rate

    def _weaken_template_memory(self, memory_key: str):
        current_strength = self._template_memory_strengths.get(memory_key, 1.0)
        self._template_memory_strengths[memory_key] = max(0.1, current_strength - 0.3)
        total_access = self._template_access_count.get(memory_key, 1)
        current_success_rate = self._template_success_rate.get(memory_key, 0.5)
        new_success_rate = (current_success_rate * total_access + 0.0) / (total_access + 1)
        self._template_success_rate[memory_key] = new_success_rate

    def _update_user_preference_memory(self, user_id: str, persona_result: Dict[str, Any], success: bool):
        if user_id not in self._user_preference_memory:
            self._user_preference_memory[user_id] = {
                "preferred_stages": {}, "preferred_traits": {},
                "response_style_prefs": {}, "trauma_tolerance": 1.0
            }
        user_prefs = self._user_preference_memory[user_id]
        stage = persona_result.get("progression_stage", "unknown")
        if stage in user_prefs["preferred_stages"]:
            current_score = user_prefs["preferred_stages"][stage]
            user_prefs["preferred_stages"][stage] = current_score + (0.2 if success else -0.1)
        else:
            user_prefs["preferred_stages"][stage] = 0.5 if success else 0.3
        trauma_level = persona_result.get("trauma_level", 1)
        if success:
            user_prefs["trauma_tolerance"] = min(3.0, user_prefs["trauma_tolerance"] + 0.1)
        else:
            user_prefs["trauma_tolerance"] = max(0.5, user_prefs["trauma_tolerance"] - 0.05)

    def _generate_dynamic_template(self, user_id: str, base_template: Dict[str, Any],
                                    progression_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            user_prefs = self._user_preference_memory.get(user_id, {})
            user_patterns = self._interaction_patterns.get(user_id, [])
            if not user_patterns:
                return base_template
            recent_patterns = user_patterns[-10:]
            successful_patterns = [p for p in recent_patterns if p["success"]]
            if not successful_patterns:
                return base_template
            dynamic_template = base_template.copy()
            successful_traits = []
            for pattern in successful_patterns:
                template_name = pattern.get("template_used", "")
                if template_name in self._learned_templates.get(user_id, {}):
                    learned_template = self._learned_templates[user_id][template_name]
                    successful_traits.extend(learned_template.get("traits", []))
            if successful_traits:
                trait_counts = {}
                for trait in successful_traits:
                    trait_counts[trait] = trait_counts.get(trait, 0) + 1
                top_traits = sorted(trait_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                dynamic_template["traits"] = [trait for trait, _ in top_traits]
            trauma_tolerance = user_prefs.get("trauma_tolerance", 1.0)
            progression_level = progression_state.get("progression_level", 0.5)
            adjusted_progression = progression_level * (1.0 / max(0.5, trauma_tolerance))
            adjusted_progression = min(1.0, max(0.0, adjusted_progression))
            dynamic_template["adjusted_progression_level"] = adjusted_progression
            dynamic_template["trauma_tolerance"] = trauma_tolerance
            return dynamic_template
        except Exception as e:
            logger.error(f"动态模板生成失败: {e}")
            return base_template

    def _consolidate_template_memory(self):
        try:
            weak_memories = []
            for memory_key, strength in self._template_memory_strengths.items():
                if strength < self._template_memory_config["memory_strength_threshold"]:
                    weak_memories.append(memory_key)
            for weak_key in weak_memories:
                self._template_memory_strengths.pop(weak_key, None)
                self._template_access_count.pop(weak_key, None)
                self._template_success_rate.pop(weak_key, None)
            if len(self._template_memory_cache) > self._template_memory_config["max_cache_size"]:
                sorted_items = sorted(
                    self._template_memory_cache.items(),
                    key=lambda x: x[1].get("last_access", 0)
                )
                items_to_remove = len(sorted_items) - self._template_memory_config["max_cache_size"]
                for i in range(items_to_remove):
                    del self._template_memory_cache[sorted_items[i][0]]
            logger.debug(f"模板记忆巩固 | 移除弱记忆: {len(weak_memories)}个")
        except Exception as e:
            logger.error(f"模板记忆巩固失败: {e}")

    async def generate_from_mood(self, mood_context: Dict[str, Any]) -> Optional[DynamicPersona]:
        mood = mood_context.get("mood", "unknown")
        return await self.generate_trauma_sensitive_persona(mood, mood_context)

    async def generate_trauma_sensitive_persona(self, persona_type: str,
                                                 mood_context: Dict[str, Any]) -> Optional[DynamicPersona]:
        try:
            if not mood_context:
                logger.warning("mood_context为空")
                return None
            user_id = mood_context.get("user_id", "default")
            current_input = mood_context.get("current_input", "") or mood_context.get("current_content", "")
            if user_id not in self._user_trauma_profiles:
                self._user_trauma_profiles[user_id] = ComplexTraumaPsychology()
                self._initialize_user_trauma_fragments(user_id)
            user_trauma = self._user_trauma_profiles[user_id]
            psychological_response = user_trauma.process_stimulus(current_input, mood_context)
            trauma_score = mood_context.get("trauma_score", 0.0) or mood_context.get("trauma_level", 0.0)
            stamina_val = 100.0
            affection_val = 0.0
            trust_val = 0.0
            annoyance_val = 0.0
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
                stream_id = mood_context.get("stream_id", f"private_{user_id}")
                tracker = get_emotion_tracker(stream_id)
                user_state = tracker.get_user_state(user_id, False)
                if user_state:
                    stamina_val = 100.0 - getattr(user_state, 'mental_fatigue', 0.0)
                    affection_val = getattr(user_state, 'affection', 0.0)
                    trust_val = getattr(user_state, 'trust_score', 0.0)
                    annoyance_val = getattr(user_state, 'annoyance', 0.0)
            except Exception as e:
                logger.debug(f"获取情感状态失败: {e}")
            mood_context["stamina"] = stamina_val
            mood_context["affection"] = affection_val
            mood_context["trust"] = trust_val
            mood_context["annoyance"] = annoyance_val
            chaos_level = psychological_response['inner_chaos_level']
            mask_level = psychological_response['surface_mask_strength']
            apparent_state = psychological_response['apparent_state']
            
            # ========== 独立双轨系统图标 ==========
            # 系统1: 内心混乱 🌀 - 独立计算
            chaos_emoji = "🌀" if chaos_level >= 8 else "💫" if chaos_level >= 5 else "🌪" if chaos_level >= 3 else "✨"
            # 系统2: 表面伪装 😊 - 独立计算
            mask_emoji = "😊" if mask_level >= 8 else "🙂" if mask_level >= 7 else "😐" if mask_level >= 5 else "😕" if mask_level >= 3 else "😢" if mask_level >= 1 else "😱"

            # 综合状态图标
            if apparent_state == "应激爆发":
                combined_emoji = "💥"
            elif apparent_state == "呆滞麻木":
                combined_emoji = "😶"
            elif apparent_state in ["彻底崩溃", "即将崩溃"]:
                combined_emoji = "😱"
            elif apparent_state in ["半崩溃", "伪装动摇", "强撑"]:
                combined_emoji = "😢"
            elif apparent_state in ["闪回发作", "内心混乱"]:
                combined_emoji = "🌀"
            elif apparent_state == "逃避模式":
                combined_emoji = "🏃"
            elif apparent_state in ["勉强伪装", "轻微异常", "露出破绽"]:
                combined_emoji = "😕"
            else:
                combined_emoji = "😊"

            # 创伤分图标
            trauma_emoji = "💔" if trauma_score >= 8 else "😢" if trauma_score >= 5 else "💖"
            # 耐力图标
            stamina_emoji = "🔋" if stamina_val >= 70 else "🪫" if stamina_val <= 30 else "⚡"

            uid_masked = self._mask_uid(user_id)
            # 🚀 极简卡片式心境模板 (彻底修复双重前缀)
            # 格式: [心境] [ID] 综合描述 (内心/表面)
            logger.info(f"[心境] [{uid_masked}] {combined_emoji}{apparent_state} ({chaos_emoji}{chaos_level:.1f}/{mask_emoji}{mask_level:.1f})")
            logger.info(f"[数据] [{uid_masked}] {trauma_emoji}{trauma_score:.1f} | {stamina_emoji}{stamina_val:.1f} | ❤️{affection_val:.0f} 🤝{trust_val:.0f} 💢{annoyance_val:.0f}")

            persona = await self._generate_complex_trauma_persona(
                user_trauma, psychological_response, mood_context
            )
            return persona
        except Exception as e:
            logger.error(f"人格生成失败: {e}")
            return self._create_emergency_persona(mood_context)

    @staticmethod
    def _mask_uid(user_id: str) -> str:
        if len(user_id) > 8:
            return f"{user_id[:4]}****{user_id[-4:]}"
        elif len(user_id) > 3:
            return f"{user_id[:3]}***"
        return user_id

    def _initialize_user_trauma_fragments(self, user_id: str):
        pass

    def _get_cached_persona_template(self, user_id: str, state_value: str) -> Optional[Dict[str, Any]]:
        cache_key = f"{user_id}_{state_value}"
        cached = self._template_memory_cache.get(cache_key)
        if not cached:
            return None
        today = time.strftime("%Y-%m-%d")
        if cached.get("generated_date", "") != today:
            del self._template_memory_cache[cache_key]
            return None
        cached["last_access"] = time.time()
        return cached

    def _store_persona_template(self, user_id: str, state_value: str,
                                 traits: List[str], name: str,
                                 speech_style: str, tone: str):
        cache_key = f"{user_id}_{state_value}"
        self._template_memory_cache[cache_key] = {
            "traits": traits,
            "name": name,
            "speech_style": speech_style,
            "tone": tone,
            "generated_date": time.strftime("%Y-%m-%d"),
            "created_at": time.time(),
            "last_access": time.time(),
        }
        if len(self._template_memory_cache) > self._template_memory_config["max_cache_size"]:
            self._consolidate_template_memory()

    def _compute_psychological_layers(self, user_trauma: ComplexTraumaPsychology,
                                       psychological_response: Dict[str, Any],
                                       mood_context: Dict[str, Any]) -> Dict[str, Any]:
        psychological_layers = dict(psychological_response["psychological_layers"])
        is_harassment = mood_context.get("is_harassment", False)
        is_insult = mood_context.get("is_insult", False)
        harassment_count = mood_context.get("harassment_count", 0)
        intensity = mood_context.get("intensity", 0.0)
        if is_harassment or is_insult:
            harassment_intensity = 10.0 if is_harassment else 7.0 if is_insult else intensity * 10
            harassment_layers = user_trauma.generate_harassment_trauma_layers(
                harassment_intensity=harassment_intensity,
                harassment_count=harassment_count
            )
            for key, value in harassment_layers.items():
                if value:
                    psychological_layers[key] = value
        is_comforting = mood_context.get("is_comforting", False)
        healer_affection = mood_context.get("healer_affection", mood_context.get("affection", 0.0))
        previous_trauma_depth = user_trauma.inner_chaos_level
        if is_comforting and previous_trauma_depth >= 3.0:
            healing_intensity = intensity * 10 if intensity > 0 else 5.0
            healing_layers = user_trauma.generate_redemption_healing_layers(
                healer_affection=healer_affection,
                healing_intensity=healing_intensity,
                previous_trauma_depth=previous_trauma_depth
            )
            for key, value in healing_layers.items():
                if value:
                    psychological_layers[key] = value
        affection_level = mood_context.get("affection", 0.0)
        trust_level = mood_context.get("trust_score", mood_context.get("trust", 0.0))
        relationship_type = mood_context.get("relationship", "陌生人")
        sentiment = mood_context.get("sentiment", "neutral")
        if affection_level >= 20 and sentiment == "positive" and not (is_harassment or is_insult):
            positive_layers = user_trauma.generate_positive_relationship_layers(
                affection_level=affection_level,
                trust_level=trust_level,
                relationship_type=relationship_type
            )
            if user_trauma.inner_chaos_level < 5.0:
                for key, value in positive_layers.items():
                    if value:
                        psychological_layers[key] = value
        return psychological_layers

    async def _generate_complex_trauma_persona(self, user_trauma: ComplexTraumaPsychology,
                                                psychological_response: Dict[str, Any],
                                                mood_context: Dict[str, Any]) -> DynamicPersona:
        current_state = TraumaState(psychological_response["apparent_state"])
        worldview_status = psychological_response["worldview_status"]
        user_id = mood_context.get("user_id", "default")
        psychological_layers = self._compute_psychological_layers(
            user_trauma, psychological_response, mood_context
        )
        cached = self._get_cached_persona_template(user_id, current_state.value)
        if cached:
            traits = cached["traits"]
            persona_name = cached["name"]
            speech_style = cached["speech_style"]
            tone = cached["tone"]
            logger.debug(f"复用缓存人格模板 | {persona_name} state={current_state.value}")
        else:
            traits = await self._llm_generate_trauma_traits(current_state, psychological_response, worldview_status, user_id)
            persona_name = await self._llm_generate_trauma_name(current_state, traits, psychological_response, user_id)
            speech_style = await self._llm_generate_trauma_speech(current_state, traits, psychological_layers, user_id)
            tone = await self._llm_generate_trauma_tone(current_state, psychological_response, user_id)
            if not traits:
                traits = ["极度不安", "情绪崩溃", "内心混乱", "表面伪装"]
            if not persona_name:
                persona_name = f"严重创伤_{current_state.value}"
            if not speech_style:
                speech_style = "混乱的表达，充满恐惧和痛苦，偶尔语无伦次"
            if not tone:
                tone = "崩溃边缘"
            self._store_persona_template(user_id, current_state.value,
                                          traits, persona_name, speech_style, tone)
        persona = DynamicPersona(
            persona_id=str(uuid.uuid4())[:12],
            name=persona_name,
            tone=tone,
            traits=traits,
            speech_style=speech_style,
            source=f"llm_complex_trauma_{current_state.value}",
            extra={"generated_date": time.strftime("%Y-%m-%d")},
        )
        persona.psychological_layers = psychological_layers
        persona.trauma_state = current_state
        persona.trauma_state_name = current_state.value
        persona.inner_chaos_level = psychological_response["inner_chaos_level"]
        persona.surface_mask_strength = psychological_response["surface_mask_strength"]
        persona.worldview_status = worldview_status
        logger.debug(f"复杂创伤人格 | {persona_name} state={current_state.value} chaos={persona.inner_chaos_level:.1f} mask={persona.surface_mask_strength:.1f}")
        return persona

    def _get_character_config(self) -> Dict[str, str]:
        try:
            from src.config.config import global_config
            name = global_config.bot.nickname
            personality = global_config.personality.personality
            style = global_config.personality.reply_style
            
            if not name or not personality:
                raise ValueError("缺少必要的角色配置信息：bot.nickname 或 personality.personality 为空")
            
            return {
                'name': name, 
                'personality': personality,
                'age': "", 
                'lore': "", 
                'relationships': "", 
                'style': style
            }
        except AttributeError as e:
            logger.error(f"配置文件缺少必要字段: {e}")
            raise RuntimeError(f"角色配置文件缺少必要字段，请检查 bot_config.toml 中的 bot.nickname 和 personality.personality") from e
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise RuntimeError(f"角色配置文件加载失败: {e}") from e

    async def _llm_generate_trauma_traits(self, trauma_state: TraumaState,
                                           psychological_response: Dict[str, Any],
                                           worldview_status: Dict[str, Any],
                                           user_id: str) -> List[str]:
        inner_chaos = psychological_response["inner_chaos_level"]
        surface_mask = psychological_response["surface_mask_strength"]
        damaged_beliefs = worldview_status.get("damaged_beliefs", [])
        try:
            character_info = self._get_character_config()
        except RuntimeError:
            return [f"{trauma_state.value}状态", "心理创伤反应", "情绪调节困难"]
        bg_block = (f"【角色背景】\n- 性格: {character_info['personality']}\n"
                    f"- 年龄: {character_info['age']}岁\n- 背景: {character_info['lore']}\n"
                    f"- 关系: {character_info['relationships']}")
        gen_block = (f"请以我（{character_info['name']}）的视角，描述我现在会有什么心理特点，8-12个：\n"
                     "1. 基于我的性格和背景\n2. 体现真实的创伤反应\n"
                     "3. 用第一视角的日常话语，4-8个字\n\n直接输出特点，用逗号分开，不要编号，不要思考过程。")
        from src.config.prompt_loader import get_prompt, PromptCategory
        
        damaged_beliefs_str = ', '.join(damaged_beliefs[:3]) if damaged_beliefs else '一些基本想法'
        
        prompt_key_map = {
            TraumaState.SURFACE_NORMAL: "traits_surface_normal.template",
            TraumaState.FLASHBACK_ACTIVE: "traits_flashback_active.template",
            TraumaState.AVOIDANCE_MODE: "traits_avoidance_mode.template",
            TraumaState.STRESS_OUTBREAK: "traits_stress_outbreak.template"
        }
        
        prompt_key = prompt_key_map.get(trauma_state, "traits_value_collapse.template")
        
        prompt = get_prompt(
            PromptCategory.MODULE,
            "persona_generator",
            prompt_key,
            character_name=character_info['name'],
            character_personality=character_info['personality'],
            inner_chaos=inner_chaos,
            surface_mask=surface_mask,
            damaged_beliefs=damaged_beliefs_str,
            bg_block=bg_block,
            gen_block=gen_block
        )
        try:
            generated_traits = await self._call_llm_for_traits(prompt, user_id)
            if generated_traits and len(generated_traits) > 0:
                logger.info(f"创伤特征 {trauma_state.value}: {len(generated_traits)}个")
                return generated_traits
        except Exception as e:
            logger.warning(f"创伤特征生成失败: {e}")
        return [f"{trauma_state.value}状态", "心理创伤反应", "情绪调节困难"]

    async def _llm_generate_trauma_name(self, trauma_state: TraumaState,
                                         traits: List[str],
                                         psychological_response: Dict[str, Any],
                                         user_id: str) -> str:
        inner_chaos = psychological_response["inner_chaos_level"]
        surface_mask = psychological_response["surface_mask_strength"]
        try:
            character_info = self._get_character_config()
        except RuntimeError:
            return f"{trauma_state.value}状态"
        bg_block = (f"【角色背景】\n- 性格: {character_info['personality']}\n"
                    f"- 年龄: {character_info['age']}岁\n- 说话风格: {character_info['style']}")
        from src.config.prompt_loader import get_prompt, PromptCategory
        
        prompt_key_map = {
            TraumaState.SURFACE_NORMAL: "name_surface_normal.template",
            TraumaState.FLASHBACK_ACTIVE: "name_flashback_active.template",
            TraumaState.AVOIDANCE_MODE: "name_avoidance_mode.template",
            TraumaState.STRESS_OUTBREAK: "name_stress_outbreak.template"
        }
        
        prompt_key = prompt_key_map.get(trauma_state, "name_value_collapse.template")
        
        prompt = get_prompt(
            PromptCategory.MODULE,
            "persona_generator",
            prompt_key,
            character_name=character_info['name'],
            character_personality=character_info['personality'],
            traits=', '.join(traits[:8]),
            inner_chaos=inner_chaos,
            surface_mask=surface_mask,
            bg_block=bg_block,
            name_req=name_req
        )
        try:
            generated_name = await self._call_llm_for_name(prompt, user_id)
            if generated_name and len(generated_name.strip()) > 0:
                logger.info(f"创伤名称 {trauma_state.value}: {generated_name}")
                return generated_name.strip()
        except Exception as e:
            logger.warning(f"创伤名称生成失败: {e}")
        return f"{trauma_state.value}状态"

    async def _llm_generate_trauma_speech(self, trauma_state: TraumaState,
                                           traits: List[str],
                                           psychological_layers: Dict[str, Any],
                                           user_id: str) -> str:
        try:
            character_info = self._get_character_config()
        except RuntimeError:
            return f"{trauma_state.value}状态的语言表达特征"
        surface_layer = psychological_layers.get('surface_layer', '努力保持正常')
        conscious_layer = psychological_layers.get('conscious_layer', '内心不安')
        subconscious_layer = psychological_layers.get('subconscious_layer', '深层混乱')
        bg_block = (f"【角色背景】\n- 性格: {character_info['personality']}\n"
                    f"- 年龄: {character_info['age']}岁\n- 平时说话风格: {character_info['style']}\n"
                    f"- 背景: {character_info['lore']}")
        from src.config.prompt_loader import get_prompt, PromptCategory
        
        prompt_key_map = {
            TraumaState.SURFACE_NORMAL: "speech_surface_normal.template",
            TraumaState.FLASHBACK_ACTIVE: "speech_flashback_active.template",
            TraumaState.AVOIDANCE_MODE: "speech_avoidance_mode.template",
            TraumaState.STRESS_OUTBREAK: "speech_stress_outbreak.template"
        }
        
        prompt_key = prompt_key_map.get(trauma_state, "speech_value_collapse.template")
        trauma_echoes_str = ', '.join(psychological_layers.get('trauma_echoes', [])) if psychological_layers.get('trauma_echoes') else '极度痛苦的创伤记忆'
        
        prompt = get_prompt(
            PromptCategory.MODULE,
            "persona_generator",
            prompt_key,
            character_name=character_info['name'],
            character_personality=character_info['personality'],
            traits=', '.join(traits[:8]),
            surface_layer=surface_layer,
            conscious_layer=conscious_layer,
            subconscious_layer=subconscious_layer,
            trauma_echoes_str=trauma_echoes_str,
            bg_block=bg_block,
            speech_req=speech_req
        )
        try:
            generated_speech = await self._call_llm_for_speech_style(prompt, user_id)
            if generated_speech and len(generated_speech.strip()) > 0:
                return generated_speech.strip()
        except Exception as e:
            logger.warning(f"创伤语言生成失败: {e}")
        return f"{trauma_state.value}状态的语言表达特征"

    async def _llm_generate_trauma_tone(self, trauma_state: TraumaState,
                                         psychological_response: Dict[str, Any],
                                         user_id: str) -> str:
        inner_chaos = psychological_response["inner_chaos_level"]
        surface_mask = psychological_response["surface_mask_strength"]
        try:
            character_info = self._get_character_config()
        except RuntimeError:
            return "complex_trauma"
        from src.config.prompt_loader import get_prompt, PromptCategory
        prompt = get_prompt(
            PromptCategory.MODULE,
            "persona_generator",
            "trauma_tone.template",
            character_name=character_info['name'],
            character_personality=character_info['personality'],
            state_value=trauma_state.value,
            inner_chaos=inner_chaos,
            surface_mask=surface_mask,
            character_age=character_info['age'],
            character_style=character_info['style']
        )
        try:
            generated_tone = await self._call_llm_for_name(prompt, user_id)
            if generated_tone and len(generated_tone.strip()) > 0:
                return generated_tone.strip()
        except Exception as e:
            logger.warning(f"创伤语调生成失败: {e}")
        return "complex_trauma"

    async def _call_llm_for_traits(self, prompt: str, user_id: str) -> List[str]:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            request = LLMRequest(model_config.lightweight, request_type="persona_traits")
            response_text, _ = await request.generate_response_async(prompt, max_tokens=600)
            if not response_text:
                return []
            content = response_text.strip()
            traits = [t.strip() for t in content.split('，') if t.strip()]
            if len(traits) <= 1:
                traits = [t.strip() for t in content.split(',') if t.strip()]
            cleaned_traits = []
            for trait in traits[:12]:
                clean_trait = trait.strip('0123456789.、- ')
                if 2 <= len(clean_trait) <= 15 and clean_trait:
                    cleaned_traits.append(clean_trait)
            return cleaned_traits
        except Exception as e:
            logger.warning(f"LLM特征生成失败: {e}")
            return []

    async def _call_llm_for_name(self, prompt: str, user_id: str) -> str:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            request = LLMRequest(model_config.lightweight, request_type="persona_name")
            response_text, _ = await request.generate_response_async(prompt, max_tokens=150)
            if not response_text:
                return ""
            name = response_text.strip().strip('""''《》「」【】()（）')
            if 2 <= len(name) <= 8:
                return name
            return ""
        except Exception as e:
            logger.warning(f"LLM名称生成失败: {e}")
            return ""

    async def _call_llm_for_speech_style(self, prompt: str, user_id: str) -> str:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            request = LLMRequest(model_config.lightweight, request_type="persona_speech")
            response_text, _ = await request.generate_response_async(prompt, max_tokens=800)
            if not response_text:
                return ""
            style = response_text.strip()
            if 50 <= len(style) <= 500:
                return style
            return ""
        except Exception as e:
            logger.warning(f"LLM语言风格生成失败: {e}")
            return ""

    def _fallback_traits(self, trauma_level: int, emotion: str) -> List[str]:
        if trauma_level >= 8:
            return ["极度疲惫感", "情绪完全崩溃", "强烈逃避冲动", "内心极度痛苦", "撑不下去了", "想要逃离一切", "感觉被彻底摧毁", "身心俱疲"]
        elif trauma_level >= 5:
            return ["感到很累很痛苦", "想要逃避现实", "心情很低落", "被伤害后很难过", "觉得很委屈", "不想面对任何人", "感觉心很痛", "想要被保护"]
        return ["心情不好想哭", "感觉很委屈", "被伤害后难过", "情绪低落", "需要安慰和关怀", "感到孤单脆弱"]

    def _fallback_name(self, trauma_level: int, emotion: str) -> str:
        if trauma_level >= 8:
            return "精神崩溃"
        elif trauma_level >= 5:
            return "心累了"
        return "很委屈"

    def _fallback_speech_style(self, trauma_level: int) -> str:
        if trauma_level >= 8:
            return "声音颤抖地说'好累，真的累了'，经常说'撑不下去了'、'想要逃避一切'，语气充满绝望和无助，说话断断续续经常停顿"
        elif trauma_level >= 5:
            return "语气低沉地说'好痛苦'、'不想面对了'、'心好累'，经常叹气，说话时带着哭腔，表达很委屈"
        return "语气委屈地说'好难过'、'想哭'、'为什么这样对我'，声音带着颤音，需要安慰"

    def _generate_masked_normal_traits(self, psychological_response: Dict, worldview_status: Dict) -> List[str]:
        inner_chaos = psychological_response["inner_chaos_level"]
        surface_mask = psychological_response["surface_mask_strength"]
        damaged_beliefs = worldview_status.get("damaged_beliefs", [])
        surface_traits = ["努力表现正常", "强迫性微笑", "过度礼貌", "回避深度话题", "快速转移注意力", "保持距离感"]
        inner_traits = []
        if inner_chaos > 8.0:
            inner_traits.extend(["内心价值观正在崩塌", "脑海中有扭曲画面", "负面情绪在攻击大脑", "对世界认知出现裂痕", "存在感正在消失", "现实感变得模糊"])
        elif inner_chaos > 5.0:
            inner_traits.extend(["价值观开始动摇", "内心充满困惑", "对自己产生怀疑", "世界观出现裂缝", "安全感正在流失"])
        else:
            inner_traits.extend(["内心有些不安", "对未来感到担心", "需要额外的安全感"])
        for belief in damaged_beliefs[:3]:
            if "人性本善" in belief:
                inner_traits.append("对他人失去基本信任")
            elif "世界是安全的" in belief:
                inner_traits.append("感觉世界充满威胁")
            elif "我是有价值的" in belief:
                inner_traits.append("自我价值感严重受损")
        return surface_traits[:4] + inner_traits[:6]

    def _generate_flashback_traits(self, trauma_echoes: List[str]) -> List[str]:
        base_traits = ["创伤记忆突然闪现", "脑海中出现扭曲画面", "时间感变得模糊", "现实与记忆混淆", "情绪被过去控制", "身体出现应激反应"]
        if trauma_echoes:
            for echo in trauma_echoes[:3]:
                base_traits.append(f"反复想起'{echo}'相关的痛苦")
        return base_traits

    def _generate_avoidance_traits(self, psychological_response: Dict) -> List[str]:
        return ["强烈想要逃离当前情况", "在寻找退出的借口", "注意力无法集中", "试图转移话题", "身体想要远离", "心理防御机制全开", "但努力保持表面平静", "不想让人看出异常", "内心在倒计时逃离"]

    def _generate_stress_outbreak_traits(self, psychological_response: Dict) -> List[str]:
        return ["内心压力突然爆发", "情绪控制出现裂缝", "身体出现应激反应", "但立即试图掩饰", "声音可能有细微颤抖", "呼吸节奏改变", "快速进入自我保护模式", "努力恢复表面镇定", "内心正在紧急修补伪装"]

    def _generate_inner_chaos_traits(self, worldview_status: Dict) -> List[str]:
        base_traits = ["价值观正在发生动摇", "对世界的认知出现裂痕", "内心充满混乱", "原有信念正在破碎", "现实感开始模糊", "自我认知产生分裂"]
        damaged_beliefs = worldview_status.get("damaged_beliefs", [])
        for belief in damaged_beliefs[:2]:
            base_traits.append(f"'{belief}'这一信念正在崩塌")
        return base_traits

    def _determine_complex_tone(self, state: TraumaState, psychological_response: Dict) -> str:
        surface_mask = psychological_response["surface_mask_strength"]
        if state == TraumaState.SURFACE_NORMAL:
            return "forced_normal" if surface_mask > 7.0 else "struggling_normal"
        elif state == TraumaState.FLASHBACK_ACTIVE:
            return "dissociative"
        elif state == TraumaState.AVOIDANCE_MODE:
            return "evasive_polite"
        elif state == TraumaState.STRESS_OUTBREAK:
            return "momentary_crack"
        return "internally_conflicted"

    def _create_emergency_persona(self, mood_context: Dict[str, Any]) -> Optional[DynamicPersona]:
        try:
            trauma_level = mood_context.get("trauma_level", 1)
            if trauma_level >= 8.0:
                emergency_traits = ["极度疲惫感", "情绪完全崩溃", "强烈逃避冲动", "内心极度痛苦", "感觉被彻底摧毁", "精神高度敏感脆弱", "对一切失去信心", "身心俱疲想要逃离"]
                emergency_name = "精神崩溃"
                emergency_speech = "感觉好累，真的累了，不想再承受任何伤害了，内心已经崩溃到极点，只想逃避一切"
            elif trauma_level >= 5.0:
                emergency_traits = ["感到很累很痛苦", "想要逃避现实", "情绪低落沮丧", "被伤害后很难过", "感觉心很痛", "不想面对任何人", "觉得自己很可怜", "想要被保护和安慰"]
                emergency_name = "痛苦逃避"
                emergency_speech = "觉得好累好痛苦，不想再被伤害了，只想找个地方躲起来"
            else:
                emergency_traits = ["感觉很委屈", "心情不好想哭", "被伤害后很难过", "情绪低落", "需要安慰和关怀", "感到孤单脆弱"]
                emergency_name = "委屈难过"
                emergency_speech = "心情很不好，感觉很委屈很难过，希望有人能安慰我一下"
            emergency_persona = DynamicPersona(
                persona_id=str(uuid.uuid4())[:12],
                name=emergency_name,
                tone="depressive_protective",
                traits=emergency_traits,
                speech_style=emergency_speech,
                source="emergency_fallback_depressive"
            )
            logger.warning(f"应急人格: '{emergency_persona.name}' | 创伤: {trauma_level}")
            return emergency_persona
        except Exception as e:
            logger.error(f"应急人格生成也失败: {e}")
            return DynamicPersona(
                persona_id="emergency_fallback",
                name="心理保护",
                tone="withdrawn",
                traits=["需要保护", "情感脆弱"],
                speech_style="简短而带有防御性的表达，避免进一步伤害",
                source="final_fallback"
            )

    def _calculate_quality_boost(self, trauma_level: float, intensity: int) -> float:
        base_quality = 1.0
        trauma_boost = 1.0 + (trauma_level / 10.0) * 2.0
        intensity_boost = 1.0 + (min(intensity, 10) / 10.0) * 1.0
        total_boost = base_quality * trauma_boost * intensity_boost
        return min(total_boost, 5.0)

    async def _generate_adaptive_traits(self, user_id: str, mood_context: Dict[str, Any],
                                         quality_boost: float) -> List[str]:
        trauma_level = mood_context.get("trauma_level", 1)
        primary_emotion = mood_context.get("primary_emotion", "hurt")
        intensity = mood_context.get("intensity", 1)
        triggers = mood_context.get("triggers", [])
        from src.config.prompt_loader import get_prompt, PromptCategory
        triggers_str = ', '.join(triggers) if triggers else '被伤害'
        prompt = get_prompt(
            PromptCategory.MODULE,
            "persona_generator",
            "adaptive_traits.template",
            trauma_level=trauma_level,
            primary_emotion=primary_emotion,
            intensity=intensity,
            triggers_str=triggers_str
        )
        try:
            generated_traits = await self._call_llm_for_traits(prompt, user_id)
            if generated_traits:
                return generated_traits
        except Exception as e:
            logger.warning(f"LLM特征生成失败: {e}")
        return self._fallback_traits(trauma_level, primary_emotion)

    async def _generate_adaptive_name(self, traits: List[str], mood_context: Dict[str, Any],
                                       quality_boost: float) -> str:
        trauma_level = mood_context.get("trauma_level", 1)
        primary_emotion = mood_context.get("primary_emotion", "hurt")
        from src.config.prompt_loader import get_prompt, PromptCategory
        prompt = get_prompt(
            PromptCategory.MODULE,
            "persona_generator",
            "adaptive_name.template",
            traits_str=', '.join(traits[:8]),
            trauma_level=trauma_level,
            primary_emotion=primary_emotion
        )
        try:
            name = await self._call_llm_for_name(prompt, mood_context.get("user_id", ""))
            if name:
                return name
        except Exception as e:
            logger.warning(f"LLM名称生成失败: {e}")
        return self._fallback_name(trauma_level, primary_emotion)

    async def _generate_adaptive_speech_style(self, traits: List[str], mood_context: Dict[str, Any],
                                               quality_boost: float) -> str:
        trauma_level = mood_context.get("trauma_level", 1)
        primary_emotion = mood_context.get("primary_emotion", "hurt")
        intensity = mood_context.get("intensity", 1)
        from src.config.prompt_loader import get_prompt, PromptCategory
        prompt = get_prompt(
            PromptCategory.MODULE,
            "persona_generator",
            "adaptive_speech.template",
            traits_str=', '.join(traits[:8]),
            trauma_level=trauma_level,
            primary_emotion=primary_emotion,
            intensity=intensity
        )
        try:
            style = await self._call_llm_for_speech_style(prompt, mood_context.get("user_id", ""))
            if style:
                return style
        except Exception as e:
            logger.warning(f"LLM语言风格生成失败: {e}")
        return self._fallback_speech_style(trauma_level)

    async def _optimize_persona_quality(self, name: str, traits: List[str], speech_style: str,
                                         user_id: str, mood_context: Dict[str, Any],
                                         quality_boost: float) -> Dict[str, Any]:
        user_prefs = self._user_preference_memory.get(user_id, {})
        successful_patterns = user_prefs.get("successful_patterns", {})
        if quality_boost >= 2.0:
            optimized_name = self._refine_name_quality(name, successful_patterns, mood_context)
            optimized_traits = self._refine_traits_quality(traits, successful_patterns, quality_boost)
            optimized_speech = self._refine_speech_quality(speech_style, mood_context, quality_boost)
            tone = "深层情感表达"
        else:
            optimized_name = name
            optimized_traits = traits
            optimized_speech = speech_style
            tone = "情感表达"
        return {"name": optimized_name, "traits": optimized_traits, "speech_style": optimized_speech, "tone": tone}

    def _refine_name_quality(self, name: str, successful_patterns: Dict, mood_context: Dict) -> str:
        if successful_patterns.get("preferred_name_style") == "complex":
            trauma_level = mood_context.get("trauma_level", 1)
            if trauma_level >= 8:
                return f"{name}且无助"
            elif trauma_level >= 5:
                return f"{name}中挣扎"
            else:
                return f"{name}寻慰藉"
        return name

    def _refine_traits_quality(self, traits: List[str], successful_patterns: Dict, quality_boost: float) -> List[str]:
        if quality_boost >= 3.0:
            refined_traits = traits.copy()
            refined_traits.extend(["情感层次丰富", "反应真实自然", "表达深度情感", "心理变化细腻", "情绪转换流畅"])
            return list(set(refined_traits))[:8]
        return traits

    def _refine_speech_quality(self, speech_style: str, mood_context: Dict, quality_boost: float) -> str:
        if quality_boost >= 4.0:
            return speech_style + "，具有明显的心理防御机制，情感表达层次分明，内心活动复杂多变"
        elif quality_boost >= 3.0:
            return speech_style + "，情感表达真实细腻，心理状态变化明显"
        return speech_style

    def _record_dynamic_generation(self, user_id: str, persona: DynamicPersona,
                                    mood_context: Dict[str, Any], quality_boost: float):
        generation_record = {
            "timestamp": time.time(), "persona_name": persona.name,
            "quality_boost": quality_boost, "trauma_level": mood_context.get("trauma_level", 1),
            "traits_count": len(persona.traits), "generation_method": "fully_dynamic"
        }
        if user_id not in self._user_preference_memory:
            self._user_preference_memory[user_id] = {}
        if "generation_history" not in self._user_preference_memory[user_id]:
            self._user_preference_memory[user_id]["generation_history"] = []
        self._user_preference_memory[user_id]["generation_history"].append(generation_record)
        if len(self._user_preference_memory[user_id]["generation_history"]) > 50:
            self._user_preference_memory[user_id]["generation_history"] = self._user_preference_memory[user_id]["generation_history"][-30:]

    def _generate_fallback_dynamic_persona(self, mood_context: Dict[str, Any]) -> DynamicPersona:
        return DynamicPersona(
            persona_id=str(uuid.uuid4())[:12],
            name="应急动态", tone="adaptable",
            traits=["动态适应", "情感表达", "稳定可靠"],
            speech_style="根据情况动态调整语言风格和情感表达",
            source="fallback_dynamic"
        )

    async def background_trauma_processing(self, user_id: str):
        if user_id not in self._user_trauma_profiles:
            return
        user_trauma = self._user_trauma_profiles[user_id]
        user_trauma.background_processing()
        if random.random() < 0.05:
            logger.info(f"后台创伤 {user_id} 触发间歇性应激")
            user_trauma.stress_accumulation += random.uniform(1.0, 3.0)
        if random.random() < 0.08:
            if user_trauma.trauma_fragments:
                fragment = random.choice(user_trauma.trauma_fragments)
                fragment.activation_count += 1

    def get_user_trauma_profile(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self._user_trauma_profiles:
            return {"status": "无创伤档案"}
        user_trauma = self._user_trauma_profiles[user_id]
        profile = user_trauma.get_current_psychological_profile()
        profile.update({
            "trauma_fragments_count": len(user_trauma.trauma_fragments),
            "most_triggered_fragments": self._get_most_triggered_fragments(user_trauma),
            "psychological_complexity": self._calculate_psychological_complexity(user_trauma),
            "long_term_prognosis": self._assess_long_term_prognosis(user_trauma)
        })
        return profile

    def _get_most_triggered_fragments(self, user_trauma: ComplexTraumaPsychology) -> List[Dict]:
        fragments_with_counts = []
        for fragment in user_trauma.trauma_fragments:
            fragments_with_counts.append({
                "triggers": fragment.trigger_words[:2],
                "activation_count": fragment.activation_count,
                "emotional_charge": fragment.emotional_charge,
            })
        fragments_with_counts.sort(key=lambda x: x["activation_count"], reverse=True)
        return fragments_with_counts[:5]

    def _calculate_psychological_complexity(self, user_trauma: ComplexTraumaPsychology) -> float:
        fragments_complexity = len(user_trauma.trauma_fragments) * 0.1
        chaos_complexity = user_trauma.inner_chaos_level * 0.15
        worldview_complexity = user_trauma.worldview_collapse.reality_distortion * 0.1
        trust_complexity = (10.0 - user_trauma.worldview_collapse.trust_level) * 0.1
        total_complexity = fragments_complexity + chaos_complexity + worldview_complexity + trust_complexity
        return min(10.0, total_complexity)

    def _assess_long_term_prognosis(self, user_trauma: ComplexTraumaPsychology) -> str:
        complexity = self._calculate_psychological_complexity(user_trauma)
        inner_chaos = user_trauma.inner_chaos_level
        surface_mask = user_trauma.surface_mask_strength
        if complexity > 8.0 and inner_chaos > 8.0:
            return "高度复杂创伤状态，需要长期稳定的支持环境"
        elif complexity > 6.0 and surface_mask < 5.0:
            return "中度创伤状态，伪装能力下降，可能需要干预"
        elif complexity > 4.0:
            return "轻中度创伤状态，目前能够维持表面功能"
        return "相对稳定状态，有一定的自我修复能力"

    async def switch_persona_state(self, stream_id: str, state_name: str,
                                    trigger_reason: str = "",
                                    extra_data: Optional[Dict[str, Any]] = None) -> bool:
        try:
            current_time = time.time()
            key = f"{stream_id}"
            last_switch_time = self._last_switch_time.get(key, 0)
            time_since_last = current_time - last_switch_time
            if time_since_last < self._switch_cooldown_seconds:
                last_state = self._last_switched_state.get(key, "")
                if last_state == state_name:
                    return True
                else:
                    logger.debug(f"心境切换冷却中: {last_state} -> {state_name} ({time_since_last:.1f}s)")
                    return False
            last_state = self._last_switched_state.get(key, "")
            if last_state == state_name:
                self._last_switch_time[key] = current_time
                return True
            from src.modules.modcore.dynamic_persona.persona_switcher import get_persona_switcher
            from src.modules.modcore.dynamic_persona.persona_controller import get_persona_controller
            switcher = get_persona_switcher()
            controller = get_persona_controller()
            extra_data = extra_data or {}
            state_to_trauma_type = {
                "应激爆发状态": "stress_outbreak", "伪装动摇状态": "mask_wavering",
                "勉强伪装状态": "forced_normal", "表面正常状态": "surface_normal",
                "半崩溃状态": "half_breakdown", "即将崩溃状态": "imminent_collapse",
                "彻底崩溃状态": "total_breakdown", "闪回发作状态": "flashback_active",
                "逃避模式状态": "avoidance_mode",
            }
            trauma_type = state_to_trauma_type.get(state_name, "general")
            intensity = extra_data.get("intensity", 5.0)
            harassment_count = extra_data.get("count", 1)
            mood_context = {
                "user_id": extra_data.get("user_id", "default"),
                "current_content": trigger_reason[:200] if trigger_reason else "",
                "trauma_level": intensity, "intensity": intensity,
                "triggers": [trauma_type],
                "primary_emotion": "devastated" if intensity >= 7 else "hurt",
                "mood_level": intensity, "harassment_count": harassment_count,
                "state_name": state_name, "stream_id": stream_id,
            }
            if extra_data.get("harassment_details"):
                harassment_details = extra_data["harassment_details"]
                mood_context["is_harassment"] = True
                mood_context["harassment_type"] = harassment_details.get("harassment_type", "unknown")
            dynamic_persona = await self.generate_trauma_sensitive_persona(trauma_type, mood_context)
            if dynamic_persona:
                dynamic_persona.name = state_name
                controller.add_persona(dynamic_persona)
                duration = 360.0 if intensity >= 8 else 300.0 if intensity >= 5 else 240.0
                success = await switcher.switch_with_auto_revert(
                    stream_id, dynamic_persona.persona_id, duration, reason=trigger_reason
                )
                if success:
                    self._last_switch_time[key] = current_time
                    self._last_switched_state[key] = state_name
                    if key not in self._state_transition_history:
                        self._state_transition_history[key] = []
                    self._state_transition_history[key].append({
                        "timestamp": current_time, "from_state": last_state,
                        "to_state": state_name, "intensity": intensity,
                        "reason": trigger_reason[:50]
                    })
                    if len(self._state_transition_history[key]) > 20:
                        self._state_transition_history[key] = self._state_transition_history[key][-20:]
                    logger.info(f"心境觉醒 {stream_id[:8]} 切换至: {state_name} 原因: {trigger_reason}")
                    return True
            logger.warning(f"心境切换人格生成失败: {state_name}")
            return False
        except Exception as e:
            logger.warning(f"心境切换失败: {e}")
            return False

    def get_stats(self) -> Dict:
        return {"awakened_today": len(self._generated_today)}

    def _generate_masked_normal_name(self, psychological_response: Dict) -> str:
        surface_mask = psychological_response.get("surface_mask_strength", 5.0)
        if surface_mask > 8.0:
            names = ["努力正常中", "强迫微笑", "表面平静", "伪装完美"]
        elif surface_mask > 6.0:
            names = ["勉强正常", "挣扎伪装", "表面镇定", "努力掩饰"]
        else:
            names = ["伪装破绽", "表面裂痕", "勉强支撑", "伪装疲劳"]
        return random.choice(names)

    def _generate_flashback_speech(self, psychological_layers: Dict) -> str:
        base_patterns = [
            "声音突然变得很小，像是在自言自语，'又想起那些事了...'",
            "说话断断续续，眼神变得空洞，'脑海里又出现那些画面...'",
            "语速变慢，带着困惑，'为什么总是会想起...不对，我在说什么'",
            "突然停顿，然后强迫自己继续，'没事，我很好，只是...'"
        ]
        trauma_echoes = psychological_layers.get("trauma_echoes", [])
        if trauma_echoes:
            echo = trauma_echoes[0]
            base_patterns.append(f"突然想起'{echo}'，声音带着颤抖，'怎么又想起这个了...'")
        return random.choice(base_patterns)

    def _generate_stress_outbreak_speech(self, psychological_layers: Dict) -> str:
        patterns = [
            "声音突然提高然后立刻压低，'我...不，没事，我很好'",
            "语气短暂失控后快速恢复，'为什么要...算了，当我没说'",
            "呼吸急促地说，'我受够了...不对，我是说我有点累'",
            "情绪瞬间泄露后立即修正，'真的很...没关系，我能处理'"
        ]
        return random.choice(patterns)

    def _add_emergency_persona_method(self):
        import types
        def _emergency_impl(self_ref, mood_context: Dict[str, Any]) -> Optional[Any]:
            try:
                from src.modules.modcore.dynamic_persona.persona_controller import DynamicPersona
                import uuid
                trauma_level = mood_context.get("trauma_level", 1)
                if trauma_level >= 8.0:
                    emergency_traits = [
                        "深度抑郁症状", "情感表达严重受阻", "认知功能显著下降",
                        "社交完全回避", "存在感消失", "内心极度空洞",
                        "生理功能紊乱", "心理保护性解离",
                    ]
                    emergency_name = "极度创伤解离"
                    emergency_speech = (
                        "语言表达极度困难，内心充满绝望和空虚，倾向于简短的自我保护性回应，"
                        "话语中透露出深度的心理创伤和无助感，避免任何可能引起二次伤害的交流，"
                        "表达方式内向、退缩、带有明显的心理防御机制"
                    )
                elif trauma_level >= 5.0:
                    emergency_traits = [
                        "中度抑郁反应", "情感表达困难", "注意力分散",
                        "人际关系敏感", "自我价值感下降", "情绪调节障碍",
                    ]
                    emergency_name = "创伤应激"
                    emergency_speech = (
                        "情感表达有一定困难，语调带有抑郁色彩，"
                        "回应时显得犹豫和不确定，话语中透露出心理压力，"
                        "对外界保持防御性，倾向于简单而谨慎的表达"
                    )
                else:
                    emergency_traits = ["轻度情绪波动", "需要理解支持", "情感敏感"]
                    emergency_name = "情绪不稳"
                    emergency_speech = "情绪化表达，语气带有明显的情感色彩，需要理解和安慰"
                persona = DynamicPersona(
                    persona_id=str(uuid.uuid4())[:12],
                    name=emergency_name,
                    tone="depressive_protective",
                    traits=emergency_traits,
                    speech_style=emergency_speech,
                    source="emergency_fallback_depressive",
                )
                logger.warning(f"应急生成抑郁式人格: '{persona.name}' | 创伤: {trauma_level}")
                return persona
            except Exception as e:
                logger.error(f"应急人格生成失败: {e}")
                return None
        self._create_emergency_persona = types.MethodType(_emergency_impl, self)

    async def _create_emergency_persona_impl(self, persona_type: str, context: Dict) -> Optional[Any]:
        try:
            from src.modules.modcore.dynamic_persona.extreme_breakdown_persona import get_extreme_breakdown_generator
            generator = get_extreme_breakdown_generator()
            persona = await generator.generate_emergency_persona(persona_type, context)
            if persona:
                return persona
        except Exception:
            pass
        if hasattr(self, '_create_emergency_persona'):
            return self._create_emergency_persona(context)
        self._add_emergency_persona_method()
        if hasattr(self, '_create_emergency_persona'):
            return self._create_emergency_persona(context)
        return None


_generator: Optional[PersonaGenerator] = None

def get_persona_generator() -> PersonaGenerator:
    global _generator
    if _generator is None:
        _generator = PersonaGenerator()
    return _generator

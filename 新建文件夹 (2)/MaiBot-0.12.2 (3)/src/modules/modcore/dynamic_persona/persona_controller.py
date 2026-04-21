import time
import random
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("persona")


@dataclass
class DynamicPersona:
    # 由认知觉醒激发的临时意识形态
    persona_id: str
    name: str
    tone: str
    traits: List[str]
    speech_style: str
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    source: str = "generated"
    extra: Dict[str, Any] = field(default_factory=dict)
    psychological_layers: Optional[Dict[str, Any]] = None
    trauma_state: Optional[Any] = None
    inner_chaos_level: Optional[float] = None
    surface_mask_strength: Optional[float] = None
    worldview_status: Optional[str] = None

    def to_prompt_addition(self) -> str:
        traits_str = "、".join(self.traits) if self.traits else ""
        return f"""【当前临时意识形态：{self.name}】
- 说话风格：{self.speech_style}
- 核心特质：{traits_str}"""

    def to_dict(self) -> Dict:
        trauma_state_value = None
        if self.trauma_state is not None:
            if hasattr(self.trauma_state, 'value'):
                trauma_state_value = self.trauma_state.value
            else:
                trauma_state_value = str(self.trauma_state)
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "tone": self.tone,
            "traits": self.traits,
            "speech_style": self.speech_style,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "use_count": self.use_count,
            "source": self.source,
            "extra": self.extra,
            "psychological_layers": self.psychological_layers,
            "trauma_state": trauma_state_value,
            "inner_chaos_level": self.inner_chaos_level,
            "surface_mask_strength": self.surface_mask_strength,
            "worldview_status": self.worldview_status,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DynamicPersona":
        trauma_state = None
        trauma_state_val = data.get("trauma_state")
        if trauma_state_val is not None:
            try:
                from src.modules.modcore.dynamic_persona.complex_trauma_psychology import TraumaState
                if isinstance(trauma_state_val, str):
                    trauma_state = TraumaState(trauma_state_val)
            except Exception:
                trauma_state = None
        return cls(
            persona_id=data.get("persona_id", ""),
            name=data.get("name", ""),
            tone=data.get("tone", ""),
            traits=data.get("traits", []),
            speech_style=data.get("speech_style", ""),
            created_at=data.get("created_at", time.time()),
            last_used=data.get("last_used", time.time()),
            use_count=data.get("use_count", 0),
            source=data.get("source", "generated"),
            extra=data.get("extra", {}),
            psychological_layers=data.get("psychological_layers"),
            trauma_state=trauma_state,
            inner_chaos_level=data.get("inner_chaos_level"),
            surface_mask_strength=data.get("surface_mask_strength"),
            worldview_status=data.get("worldview_status"),
        )


class PersonaController:
    MAX_PERSONAS = 10
    DEFAULT_DURATION = 300.0
    TRAINING_PERSONA_THRESHOLD = 40
    CHARM_LEVELS = {
        40: {"charm": 0.2, "desc": "初步顺从", "style": "偶尔会听话"},
        50: {"charm": 0.4, "desc": "逐渐适应", "style": "会害羞地配合"},
        60: {"charm": 0.6, "desc": "明显依赖", "style": "主动想要讨好"},
        70: {"charm": 0.8, "desc": "高度服从", "style": "很在意对方的看法"},
        80: {"charm": 0.9, "desc": "完全信赖", "style": "只想让对方开心"},
        90: {"charm": 1.0, "desc": "专属状态", "style": "完全属于对方"},
    }

    def __init__(self, config: Optional[Dict] = None, stream_id: str = "default"):
        self._cfg = config or {}
        self._stream_id = stream_id
        self._main_persona: Optional[DynamicPersona] = None
        self._persona_pool: Dict[str, DynamicPersona] = {}
        self._stream_states: Dict[str, Dict[str, Any]] = {}
        self._last_daily_refresh: str = ""

    def check_daily_refresh(self) -> bool:
        today = time.strftime("%Y-%m-%d")
        if self._last_daily_refresh != today:
            self._last_daily_refresh = today
            now = time.time()
            expired = [pid for pid, p in self._persona_pool.items()
                       if p.extra.get("generated_date", "") and p.extra["generated_date"] != today]
            for pid in expired:
                del self._persona_pool[pid]
            if expired:
                logger.info(f"每日刷新: 移除了 {len(expired)} 个过期人格")
            return True
        return False

    def set_main_persona(self, persona: DynamicPersona):
        self._main_persona = persona
        logger.info(f"核心自我设定已更新: {persona.name}")

    def get_main_persona(self) -> Optional[DynamicPersona]:
        return self._main_persona

    def add_persona(self, persona: DynamicPersona) -> bool:
        if len(self._persona_pool) >= self.MAX_PERSONAS:
            least_used = min(self._persona_pool.values(), key=lambda p: (p.use_count, p.last_used))
            self._persona_pool.pop(least_used.persona_id, None)
            logger.debug(f"人格池已满，移除最少使用的: {least_used.name}")
        self._persona_pool[persona.persona_id] = persona
        return True

    def replace_persona(self, old_persona_id: str, new_persona: DynamicPersona) -> bool:
        if old_persona_id in self._persona_pool:
            del self._persona_pool[old_persona_id]
            logger.info(f"移除旧人格: {old_persona_id[:8]}")
        new_persona.extra["generated_date"] = time.strftime("%Y-%m-%d")
        self._persona_pool[new_persona.persona_id] = new_persona
        logger.info(f"新人格: {new_persona.name}")
        return True

    def get_persona(self, persona_id: str) -> Optional[DynamicPersona]:
        return self._persona_pool.get(persona_id)

    def list_personas(self) -> List[DynamicPersona]:
        return list(self._persona_pool.values())

    def get_persona_stats(self) -> Dict[str, Any]:
        return {
            "total": len(self._persona_pool),
            "max": self.MAX_PERSONAS,
            "last_refresh": self._last_daily_refresh,
        }

    def switch_persona(self, stream_id: str, persona_id: str,
                       duration: Optional[float] = None, reason: str = "") -> bool:
        persona = self._persona_pool.get(persona_id)
        if not persona:
            return False
        now = time.time()
        self._stream_states[stream_id] = {
            "active_persona_id": persona_id,
            "end_time": now + (duration or self.DEFAULT_DURATION),
            "reason": reason,
        }
        persona.use_count += 1
        persona.last_used = now
        logger.info(f"心境切换 {stream_id[:8]} → {persona.name} | 原因: {reason}")
        return True

    def revert_to_main(self, stream_id: str):
        if stream_id in self._stream_states:
            del self._stream_states[stream_id]

    def get_active_persona(self, stream_id: str) -> Optional[DynamicPersona]:
        state = self._stream_states.get(stream_id)
        if not state:
            return None
        if time.time() > state["end_time"]:
            self.revert_to_main(stream_id)
            return None
        return self._persona_pool.get(state["active_persona_id"])

    def is_persona_active(self, stream_id: str) -> bool:
        return self.get_active_persona(stream_id) is not None

    def get_remaining_time(self, stream_id: str) -> float:
        state = self._stream_states.get(stream_id)
        if not state:
            return 0.0
        return max(0.0, state["end_time"] - time.time())

    def get_persona_prompt_addition(self, stream_id: str, user_id: str = "guest") -> str:
        prompt_parts = []
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(stream_id)
            state = tracker.get_user_state(user_id, create_if_missing=False)
            if state:
                if state.affection >= 85:
                    prompt_parts.append("你很喜欢这个人")
                elif state.affection >= 50:
                    prompt_parts.append("你对他有好感")
                elif state.affection <= -60:
                    prompt_parts.append("你讨厌这个人")
                elif state.affection <= -30:
                    prompt_parts.append("你不太喜欢这个人")
                if state.trust_score >= 50:
                    prompt_parts.append("你很信任他")
                elif state.trust_score < 0:
                    prompt_parts.append("你对他有戒心")
                if state.annoyance > 50:
                    prompt_parts.append("你很烦")
                if state.trauma_score > 3:
                    prompt_parts.append("你对他有些敏感")
                if hasattr(state, 'relationship') and state.relationship and state.relationship not in ["陌生人", "stranger"]:
                    prompt_parts.append(f"关系：{state.relationship}")
                if hasattr(state, 'impression') and state.impression and state.impression not in ["初次见面", "未知"]:
                    prompt_parts.append(f"印象：{state.impression}")
            global_rage = tracker.get_global_rage_level() if hasattr(tracker, 'get_global_rage_level') else 0.0
            if global_rage >= 80.0:
                prompt_parts.append("你情绪波动很大")
        except Exception as e:
            logger.debug(f"获取情绪状态失败: {e}")
        active = self.get_active_persona(stream_id)
        if active:
            prompt_parts.append(f"{active.name}：{active.speech_style}")
            if active.source == "progression_based" and active.extra.get("voice_samples"):
                voice_samples = active.extra["voice_samples"]
                import random
                prompt_parts.append(f"回复示例：{random.choice(voice_samples) if voice_samples else ''}")
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(stream_id)
            psych_state = tracker.get_psychology_state(user_id) if hasattr(tracker, 'get_psychology_state') else None
            if psych_state:
                trend = psych_state.get_state_trend(hours=24.0)
                if trend["entry_count"] > 0:
                    prompt_parts.append(f"{trend['dominant_state']}，{trend['recommendation']}")
                if psych_state.inner_chaos_level > 5.0:
                    prompt_parts.append("内心很乱，努力维持冷静")
                if psych_state.surface_mask_strength < 6.0:
                    prompt_parts.append("伪装快撑不住了")
                if psych_state.current_state.value == "闪回发作":
                    prompt_parts.append("脑海中闪过痛苦的记忆")
                if psych_state.stress_accumulation > 5.0:
                    prompt_parts.append("压力很大")
                if psych_state.mask_fatigue > 5.0:
                    prompt_parts.append("装久了很累")
                damaged_beliefs = psych_state.worldview_collapse.core_beliefs_damaged
                if damaged_beliefs:
                    prompt_parts.append(f"信念动摇：{damaged_beliefs[0]}")
                if len(psych_state.trauma_timeline) > 10:
                    prompt_parts.append(f"经历了{len(psych_state.trauma_timeline)}次创伤")
        except Exception:
            pass
        if not prompt_parts:
            return ""
        return "；".join(prompt_parts)

    async def should_switch_persona(self, stream_id: str, content: str,
                                     user_id: str, context: Optional[Dict] = None) -> Tuple[bool, Optional[str], str]:
        # 判断是否需要切换人格
        if not self._persona_pool:
            return False, None, ""
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(stream_id)
            state = tracker.get_user_state(user_id, create_if_missing=False)
            if not state:
                return False, None, ""
            if state.annoyance > 60 and state.trauma_score >= 3:
                for p in self._persona_pool.values():
                    if any(kw in p.name for kw in ["创伤", "动摇", "崩溃", "委屈", "防御"]):
                        return True, p.persona_id, f"烦躁度{state.annoyance:.0f}+创伤{state.trauma_score:.1f}触发"
        except Exception as e:
            logger.debug(f"人格切换判断失败: {e}")
        # 承受累积机制 - 检查用户交互历史
        try:
            # 获取最近5次交互记录进行累积分析
            interaction_history = tracker.get_recent_interactions(user_id, limit=5)
            negative_interaction_count = sum(1 for interaction in interaction_history if interaction.get("sentiment") == "negative")
            
            # 连续负面交互触发机制
            if negative_interaction_count >= 3 and state.trauma_score >= 5.0:
                # 连续3次负面交互且创伤分≥5，触发负面人格为主
                logger.info(f"承受累积触发: 连续{negative_interaction_count}次负面交互，创伤分{state.trauma_score:.1f}")
                
                # 查找合适的负面人格
                for p in self._persona_pool.values():
                    if any(keyword in p.name for keyword in ["创伤", "动摇", "崩溃", "委屈"]):
                        return True, p.persona_id, f"连续{negative_interaction_count}次负面交互触发负面人格为主"
        except Exception as history_e:
            logger.debug(f"交互历史获取失败: {history_e}")
            
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            personas_desc = "\n".join([
                f"- {p.name}: {p.tone} ({p.speech_style})"
                for p in list(self._persona_pool.values())[:10]
            ])
            # 读取人格配置（使用正确的路径）
            try:
                from src.config.config import global_config
                logger.debug(f"全局配置对象类型: {type(global_config)}")
                logger.debug(f"bot配置对象类型: {type(global_config.bot) if hasattr(global_config, 'bot') else 'MISSING'}")
                logger.debug(f"personality配置对象类型: {type(global_config.personality) if hasattr(global_config, 'personality') else 'MISSING'}")
                
                character_name = global_config.bot.nickname
                personality = global_config.personality.personality
                
                logger.debug(f"成功读取人格配置: {character_name}, {personality[:20]}...")
            except AttributeError as e:
                logger.error(f"人格配置属性访问失败: {e}")
                logger.error(f"global_config可用属性: {dir(global_config) if 'global_config' in locals() else 'global_config未定义'}")
                character_name = '爱丽丝'  # 使用正确的配置值作为回退
                personality = '是个有点傲娇又社恐的女孩子，喜欢刷B站看番追动漫，平时爱和朋友聊QQ水群'
            except Exception as e:
                logger.error(f"读取人格配置失败: {type(e).__name__}: {e}")
                character_name = '爱丽丝'  # 使用正确的配置值作为回退
                personality = '是个有点傲娇又社恐的女孩子，喜欢刷B站看番追动漫，平时爱和朋友聊QQ水群'
                
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "persona_controller",
                "cognitive_awakening.template",
                character_name=character_name,
                personality=personality,
                relationship=state.relationship if state else '未知',
                impression=state.impression if state else '未知',
                affection=state.affection if state else 0,
                trust_score=state.trust_score if state else 0,
                annoyance=state.annoyance if state else 0,
                trauma_score=state.trauma_score if state else 0,
                content=content[:300],
                personas_desc=personas_desc
            )
            request = LLMRequest(model_config.lightweight, request_type="persona_switch")
            response_text, _ = await request.generate_response_async(prompt)
            if response_text and "YES|" in response_text.upper():
                parts = response_text.strip().split("|")
                if len(parts) >= 2:
                    target_name = parts[1].strip()
                    for p in self._persona_pool.values():
                        if p.name == target_name:
                            reason = parts[2] if len(parts) > 2 else "认知感官触发"
                            return True, p.persona_id, reason
        except Exception as e:
            logger.debug(f"LLM人格切换分析失败: {e}")
        return False, None, ""


    def _get_db(self):
        if self._db is None:
            try:
                from src.modules.modcore.dynamic_persona.emotion_database import get_emotion_database
                self._db = get_emotion_database(self._stream_id if hasattr(self, '_stream_id') else "default")
            except Exception as e:
                logger.warning(f"数据库初始化失败: {e}")
        return self._db

    def _load_personas(self):
        try:
            db = self._get_db()
            if not db:
                return
            today = time.strftime("%Y-%m-%d")
            loaded_count = 0
            expired_count = 0
            for p_data in db.load_all_personas():
                persona = DynamicPersona.from_dict(p_data)
                generated_date = persona.extra.get("generated_date", "")
                if generated_date and generated_date != today:
                    db.delete_persona(persona.persona_id)
                    expired_count += 1
                    continue
                self._persona_pool[persona.persona_id] = persona
                loaded_count += 1
            if expired_count > 0:
                logger.info(f"每日刷新: 移除了 {expired_count} 个过期人格")
            if loaded_count > 0:
                logger.info(f"加载了 {loaded_count} 个动态人格")
            self._last_daily_refresh = today
        except Exception as e:
            logger.debug(f"加载人格失败: {e}")

    def _load_stream_states(self):
        try:
            db = self._get_db()
            if not db:
                return
            stream_states_data = db.load_stream_states()
            now = time.time()
            loaded_count = 0
            for stream_id, state_data in stream_states_data.items():
                end_time = state_data.get("end_time", 0)
                if end_time > now:
                    self._stream_states[stream_id] = state_data
                    loaded_count += 1
            if loaded_count > 0:
                logger.info(f"恢复了 {loaded_count} 个活跃人格状态")
        except Exception as e:
            logger.debug(f"流状态加载失败: {e}")

    def _save_persona(self, persona: "DynamicPersona"):
        try:
            db = self._get_db()
            if db:
                db.save_persona(persona.to_dict())
        except Exception as e:
            logger.debug(f"保存人格失败: {e}")

    def _get_llm(self):
        if self._llm_bridge is None:
            try:
                from src.llm_models.utils_model import LLMRequest
                from src.config.config import model_config
                self._llm_bridge = LLMRequest(model_set=model_config.model_task_config.utils, request_type="persona")
            except Exception:
                pass
        return self._llm_bridge

    def _get_accumulated_state_info(self, stream_id: str, user_id: str = "") -> Dict[str, Any]:
        state_data = self._stream_states.get(stream_id, {})
        if not state_data:
            return {"active": False, "persona_id": None, "remaining": 0}
        now = time.time()
        end_time = state_data.get("end_time", 0)
        remaining = max(0, end_time - now)
        return {
            "active": remaining > 0,
            "persona_id": state_data.get("active_persona_id"),
            "remaining": remaining,
            "reason": state_data.get("reason", ""),
        }

    def _save_stream_states(self):
        try:
            db = self._get_db()
            if not db:
                return
            now = time.time()
            active_states = {sid: st for sid, st in self._stream_states.items() if st.get("end_time", 0) > now}
            db.save_stream_states(active_states)
        except Exception as e:
            logger.debug(f"流状态保存失败: {e}")

    def can_trigger_training_persona(self, submission_level: float) -> bool:
        return submission_level >= getattr(self, 'TRAINING_PERSONA_THRESHOLD', 60.0)

    def get_charm_level(self, submission_level: float) -> Dict[str, Any]:
        result = {"charm": 0.0, "desc": "未触发", "style": "正常状态"}
        threshold = getattr(self, 'TRAINING_PERSONA_THRESHOLD', 60.0)
        if submission_level < threshold:
            return result
        charm_levels = getattr(self, 'CHARM_LEVELS', {})
        for t in sorted(charm_levels.keys(), reverse=True):
            if submission_level >= t:
                return charm_levels[t]
        return result

    def should_accumulate_psychological_state(self, stream_id: str, user_id: str = "", response_data: dict = None) -> bool:
        if not response_data:
            state_info = self._get_accumulated_state_info(stream_id)
            return state_info.get("active", False)
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(stream_id)
            psych_state = tracker.get_psychology_state(user_id) if hasattr(tracker, 'get_psychology_state') else None
            if not psych_state:
                return False
            if hasattr(psych_state, 'accumulate_state'):
                psych_state.accumulate_state(response_data, source="interaction")
            if hasattr(psych_state, 'should_trigger_awakening'):
                should_awaken, reason = psych_state.should_trigger_awakening()
                if should_awaken:
                    logger.info(f"累积状态触发心境觉醒: {reason}")
                    return True
        except Exception as e:
            logger.debug(f"累积心理状态失败: {e}")
        return False

    def replace_persona(self, old_persona_id: str, new_persona: "DynamicPersona") -> bool:
        if old_persona_id in self._persona_pool:
            db = self._get_db()
            if db:
                db.delete_persona(old_persona_id)
            del self._persona_pool[old_persona_id]
        new_persona.extra["generated_date"] = time.strftime("%Y-%m-%d")
        self._persona_pool[new_persona.persona_id] = new_persona
        self._save_persona(new_persona)
        return True


_controller: Optional[PersonaController] = None


def get_persona_controller() -> PersonaController:
    global _controller
    if _controller is None:
        _controller = PersonaController()
    return _controller

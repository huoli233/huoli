import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.common.config.config_engine import ConfigEngine
from src.common.logger import get_logger

logger = get_logger("人格控制器")


@dataclass
class DynamicPersona:
    """动态人格数据类"""

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
    trauma_state: Optional[str] = None
    inner_chaos_level: Optional[float] = None
    surface_mask_strength: Optional[float] = None

    def to_prompt_addition(self) -> str:
        traits_str = "、".join(self.traits) if self.traits else ""
        return f"""【当前临时意识形态：{self.name}】
- 说话风格：{self.speech_style}
- 核心特质：{traits_str}"""

    def to_dict(self) -> Dict[str, Any]:
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
            "trauma_state": self.trauma_state,
            "inner_chaos_level": self.inner_chaos_level,
            "surface_mask_strength": self.surface_mask_strength,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DynamicPersona":
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
            trauma_state=data.get("trauma_state"),
            inner_chaos_level=data.get("inner_chaos_level"),
            surface_mask_strength=data.get("surface_mask_strength"),
        )


class PersonaController:
    """人格控制器

    管理动态人格的创建、切换、恢复。
    支持基于情绪状态的自动人格切换。
    """

    MAX_PERSONAS = 10
    DEFAULT_DURATION = 300.0
    CHARM_LEVELS = {
        40: {"charm": 0.2, "desc": "初步顺从", "style": "偶尔会听话"},
        50: {"charm": 0.4, "desc": "逐渐适应", "style": "会害羞地配合"},
        60: {"charm": 0.6, "desc": "明显依赖", "style": "主动想要讨好"},
        70: {"charm": 0.8, "desc": "高度服从", "style": "很在意对方的看法"},
        80: {"charm": 0.9, "desc": "完全信赖", "style": "只想让对方开心"},
        90: {"charm": 1.0, "desc": "专属状态", "style": "完全属于对方"},
    }

    def __init__(
        self,
        config_engine: Optional[ConfigEngine] = None,
        stream_id: str = "default",
    ):
        self._config = config_engine or ConfigEngine.get_instance()
        self._stream_id = stream_id
        self._main_persona: Optional[DynamicPersona] = None
        self._persona_pool: Dict[str, DynamicPersona] = {}
        self._stream_states: Dict[str, Dict[str, Any]] = {}
        self._last_daily_refresh: str = ""
        self._load_config()

    def _load_config(self):
        persona_cfg = self._config.get("persona", {})
        self._max_personas = persona_cfg.get("max_personas", self.MAX_PERSONAS)
        self._default_duration = persona_cfg.get(
            "default_duration", self.DEFAULT_DURATION
        )

    def check_daily_refresh(self) -> bool:
        today = time.strftime("%Y-%m-%d")
        if self._last_daily_refresh != today:
            self._last_daily_refresh = today
            expired = [
                pid
                for pid, p in self._persona_pool.items()
                if p.extra.get("generated_date", "")
                and p.extra["generated_date"] != today
            ]
            for pid in expired:
                del self._persona_pool[pid]
            if expired:
                logger.info(f"每日刷新: 移除了 {len(expired)} 个过期人格")
            return True
        return False

    def set_main_persona(self, persona: DynamicPersona) -> None:
        self._main_persona = persona
        logger.info(f"核心自我设定已更新: {persona.name}")

    def get_main_persona(self) -> Optional[DynamicPersona]:
        return self._main_persona

    def add_persona(self, persona: DynamicPersona) -> bool:
        if len(self._persona_pool) >= self._max_personas:
            least_used = min(
                self._persona_pool.values(),
                key=lambda p: (p.use_count, p.last_used),
            )
            self._persona_pool.pop(least_used.persona_id, None)
            logger.debug(f"人格池已满，移除最少使用的: {least_used.name}")
        self._persona_pool[persona.persona_id] = persona
        return True

    def replace_persona(
        self, old_persona_id: str, new_persona: DynamicPersona
    ) -> bool:
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
            "max": self._max_personas,
            "last_refresh": self._last_daily_refresh,
        }

    def switch_persona(
        self,
        stream_id: str,
        persona_id: str,
        duration: Optional[float] = None,
        reason: str = "",
    ) -> bool:
        persona = self._persona_pool.get(persona_id)
        if not persona:
            return False
        now = time.time()
        self._stream_states[stream_id] = {
            "active_persona_id": persona_id,
            "end_time": now + (duration or self._default_duration),
            "reason": reason,
        }
        persona.use_count += 1
        persona.last_used = now
        logger.info(
            f"心境切换 {stream_id[:8]} → {persona.name} | 原因: {reason}"
        )
        return True

    def revert_to_main(self, stream_id: str) -> None:
        if stream_id in self._stream_states:
            del self._stream_states[stream_id]
            logger.debug(f"恢复主人格: {stream_id[:8]}")

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

    def get_persona_prompt_addition(
        self, stream_id: str, user_id: str = "guest"
    ) -> str:
        prompt_parts = []
        active = self.get_active_persona(stream_id)
        if active:
            prompt_parts.append(f"{active.name}：{active.speech_style}")
            if active.traits:
                prompt_parts.append(f"特质：{'、'.join(active.traits[:3])}")
        if not prompt_parts:
            return ""
        return "；".join(prompt_parts)

    async def should_switch_persona(
        self,
        stream_id: str,
        content: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str], str]:
        if not self._persona_pool:
            return False, None, ""
        ctx = context or {}
        annoyance = ctx.get("annoyance", 0.0)
        trauma_score = ctx.get("trauma_score", 0.0)
        sentiment = ctx.get("sentiment", "neutral")
        if annoyance > 60 and trauma_score >= 3:
            for p in self._persona_pool.values():
                if any(
                    kw in p.name
                    for kw in ["创伤", "动摇", "崩溃", "委屈", "防御"]
                ):
                    return (
                        True,
                        p.persona_id,
                        f"烦躁度{
                            annoyance:.0f}+创伤{
                            trauma_score:.1f}触发",
                    )
        if sentiment == "negative" and trauma_score >= 5:
            for p in self._persona_pool.values():
                if any(kw in p.name for kw in ["创伤", "委屈", "防御"]):
                    return (
                        True,
                        p.persona_id,
                        f"负面情绪+创伤{trauma_score:.1f}触发",
                    )
        return False, None, ""

    def get_charm_level(self, submission_level: float) -> Dict[str, Any]:
        result = {"charm": 0.0, "desc": "未触发", "style": "正常状态"}
        for threshold in sorted(self.CHARM_LEVELS.keys(), reverse=True):
            if submission_level >= threshold:
                return self.CHARM_LEVELS[threshold]
        return result


_persona_controller_instance: Optional[PersonaController] = None


def get_persona_controller() -> PersonaController:
    """获取人格控制器单例"""
    global _persona_controller_instance
    if _persona_controller_instance is None:
        _persona_controller_instance = PersonaController()
    return _persona_controller_instance

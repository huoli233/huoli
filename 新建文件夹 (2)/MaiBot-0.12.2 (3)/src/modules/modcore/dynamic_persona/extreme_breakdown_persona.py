import random
from typing import Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("extreme_breakdown")


class ExtremeBreakdownPersona:
    # 极度崩溃人格生成器，基于LLM动态生成
    def __init__(self):
        self.used_personas = []

    async def generate_extreme_breakdown_persona(self, user_id: str, trauma_score: float, context: str = "") -> Optional[Dict]:
        # 创伤分>=9.0时触发极度崩溃人格
        if trauma_score < 9.0:
            return None
        from src.modules.modcore.dynamic_persona.persona_generator import get_persona_generator
        mood_context = {
            "user_id": user_id,
            "current_content": context[:200] if context else "极度创伤触发",
            "trauma_level": min(trauma_score, 10.0),
            "intensity": 10,
            "triggers": ["extreme_breakdown", "severe_trauma"],
            "primary_emotion": "devastated",
            "mood_level": trauma_score,
            "special_handling": "extreme_breakdown",
        }
        generator = get_persona_generator()
        dynamic_persona = await generator.generate_trauma_sensitive_persona("extreme_trauma", mood_context)
        if not dynamic_persona:
            return self._create_fallback_persona(user_id, trauma_score, context)
        persona_data = {
            "persona_id": dynamic_persona.persona_id,
            "name": dynamic_persona.name,
            "description": f"极度创伤动态人格: {dynamic_persona.name}",
            "traits": dynamic_persona.traits,
            "response_style": {
                "speech_style": dynamic_persona.speech_style,
                "tone": dynamic_persona.tone,
                "special_handling": "extreme_breakdown_dynamic",
                "generation_method": "llm_dynamic"
            },
            "trigger_conditions": {
                "trauma_threshold": 9.0,
                "current_trauma": trauma_score,
                "user_id": user_id,
                "context": context
            },
            "duration": random.randint(300, 600),
            "priority": "extreme",
            "source": "dynamic_llm_generation"
        }
        return persona_data

    def _create_fallback_persona(self, user_id: str, trauma_score: float, context: str) -> Dict:
        return {
            "persona_id": f"emergency_breakdown_{random.randint(1000, 9999)}",
            "name": "应急极度创伤",
            "description": "动态生成失败时的应急人格",
            "traits": ["极度创伤", "应急状态", "系统保护"],
            "response_style": {
                "speech_style": "语言表达严重受阻，情感状态极度压抑",
                "tone": "traumatized",
                "special_handling": "emergency_fallback",
            },
            "trigger_conditions": {
                "trauma_threshold": 9.0,
                "current_trauma": trauma_score,
                "user_id": user_id,
                "context": context
            },
            "duration": 300,
            "priority": "extreme",
            "source": "emergency_fallback"
        }

    async def generate_emergency_persona(self, persona_type: str, context: Dict) -> Optional[Dict]:
        trauma_score = context.get("trauma_level", 9.0)
        user_id = context.get("user_id", "unknown")
        context_str = context.get("context", persona_type)
        return await self.generate_extreme_breakdown_persona(user_id, trauma_score, context_str)

    def get_breakdown_response(self, persona_data: Dict, message: str) -> str:
        # 生成极度崩溃回复
        if persona_data.get("source") == "dynamic_llm_generation":
            traits = persona_data.get("traits", [])
            speech_style = persona_data.get("response_style", {}).get("speech_style", "")
            response_guide = f"基于特征{traits}和语言风格{speech_style}的深度表达"
            return self._generate_dynamic_response(response_guide, persona_data)
        return "内心极度痛苦...无法言喻..."

    def _generate_dynamic_response(self, guide: str, persona_data: Dict) -> str:
        traits = persona_data.get("traits", [])
        if "深度抑郁倾向" in traits:
            return "内心深处的抑郁如潮水般涌来...我感觉自己正在被吞没..."
        elif "情感麻木与敏感交替" in traits:
            return "有时感觉不到任何东西...有时又被情感的洪流冲击得无法呼吸..."
        elif "存在感消失" in traits:
            return "我感觉自己正在消失...像是从未存在过一样..."
        else:
            return "心理状态极度不稳定...需要专业的心理支持..."


_extreme_breakdown_generator = None


def get_extreme_breakdown_generator() -> ExtremeBreakdownPersona:
    global _extreme_breakdown_generator
    if _extreme_breakdown_generator is None:
        _extreme_breakdown_generator = ExtremeBreakdownPersona()
    return _extreme_breakdown_generator

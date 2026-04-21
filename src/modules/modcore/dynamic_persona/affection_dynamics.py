import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.common.config.config_engine import ConfigEngine
from src.common.logger import get_logger

logger = get_logger("affection_dynamics")


class EmotionContagion:
    """情绪感染分析"""

    EMOTION_ANALYSIS_PROMPT = """分析用户消息中的情绪，返回JSON格式：
{"emotions": {"joy": 0-100, "sadness": 0-100, "anger": 0-100, "fear": 0-100, "surprise": 0-100, "disgust": 0-100, "anticipation": 0-100, "gratitude": 0-100}, "dominant": "主要情绪", "intensity": "low/medium/high"}
只返回JSON，不要其他内容。"""

    CONTAGION_PROMPT = """[情绪感知]
用户当前情绪: {user_emotion}
情绪强度: {intensity}

你能感受到用户的情绪。根据你们的关系和你的性格，这种情绪会以某种方式影响你。
- 如果用户开心，你可能也会感到愉悦
- 如果用户难过，你可能会想要安慰
- 如果用户生气，你可能会紧张或想要解决问题
- 如果用户害怕，你可能会想要保护
- 如果用户感谢你，你会感到被认可"""

    @classmethod
    def build_emotion_analysis_prompt(cls, user_message: str) -> str:
        return f"{cls.EMOTION_ANALYSIS_PROMPT}\n\n用户消息: {user_message}"

    @classmethod
    def build_contagion_context(
        cls, detected_emotion: str, intensity: str
    ) -> str:
        return cls.CONTAGION_PROMPT.format(
            user_emotion=detected_emotion, intensity=intensity
        )

    @classmethod
    def parse_emotion_result(cls, llm_response: str) -> Dict[str, Any]:
        try:
            result = json.loads(llm_response.strip())
            return result
        except Exception:
            return {"emotions": {}, "dominant": "neutral", "intensity": "low"}


class ConversationContextAnalyzer:
    """对话情境分析"""

    CONTEXT_ANALYSIS_PROMPT = """分析这段对话的情境类型，返回JSON格式：
{"context_type": "类型", "description": "情境描述", "response_suggestion": "回应建议"}

可能的情境类型:
- casual_chat: 闲聊
- seeking_help: 求助
- emotional_support: 需要情感支持
- flirting: 调情或表达亲密
- argument: 争论或冲突
- apology: 道歉
- gratitude: 感谢
- curiosity: 好奇询问
- other: 其他

只返回JSON，不要其他内容。"""

    @classmethod
    def build_analysis_prompt(cls, user_message: str) -> str:
        return f"{cls.CONTEXT_ANALYSIS_PROMPT}\n\n用户消息: {user_message}"

    @classmethod
    def parse_context_result(cls, llm_response: str) -> Dict[str, Any]:
        try:
            result = json.loads(llm_response.strip())
            return result
        except Exception:
            return {
                "context_type": "casual_chat",
                "description": "普通对话",
                "response_suggestion": "正常回应",
            }

    @classmethod
    def build_context_injection(
        cls, context_type: str, description: str, suggestion: str
    ) -> str:
        return f"""[对话情境感知]
当前情境: {description}
回应建议: {suggestion}"""


class AffectionTriggers:
    """好感度触发分析"""

    AFFECTION_ANALYSIS_PROMPT = """分析用户消息对好感度的影响，返回JSON格式：
{"sentiment": "positive/negative/neutral", "intensity": "low/medium/high", "trigger_type": "类型", "description": "描述", "affection_change": -100到100的整数, "trust_change": -100到100的整数, "annoyance_change": 0到100的整数}

可能的trigger_type:
正面: genuine_praise(真诚赞美), care_expression(关心表达), sharing_secrets(分享秘密), apology(道歉), remembering(记住之前的事), support(支持鼓励)
负面: insult(侮辱), disrespect(不尊重), manipulation(操纵), harassment(骚扰), ignoring_boundaries(无视边界), criticism(批评指责)
中性: question(提问), statement(陈述), greeting(问候)

只返回JSON，不要其他内容。"""

    @classmethod
    def build_analysis_prompt(
        cls, user_message: str, current_affection: float
    ) -> str:
        return f"""{cls.AFFECTION_ANALYSIS_PROMPT}

当前好感度: {current_affection}/100
用户消息: {user_message}"""

    @classmethod
    def parse_trigger_result(cls, llm_response: str) -> Dict[str, Any]:
        try:
            result = json.loads(llm_response.strip())
            return {
                "sentiment": result.get("sentiment", "neutral"),
                "intensity": result.get("intensity", "low"),
                "trigger_type": result.get("trigger_type", "statement"),
                "description": result.get("description", ""),
                "affection_delta": result.get("affection_change", 0),
                "trust_delta": result.get("trust_change", 0),
                "annoyance_delta": result.get("annoyance_change", 0),
            }
        except Exception:
            return {
                "sentiment": "neutral",
                "intensity": "low",
                "trigger_type": "statement",
                "description": "",
                "affection_delta": 0,
                "trust_delta": 0,
                "annoyance_delta": 0,
            }

    @classmethod
    def build_trigger_context(cls, trigger_result: Dict[str, Any]) -> str:
        if trigger_result["sentiment"] == "neutral":
            return ""
        sentiment_desc = (
            "正面" if trigger_result["sentiment"] == "positive" else "负面"
        )
        return f"""[好感度变化感知]
用户行为: {trigger_result['description']}
情感倾向: {sentiment_desc}
影响强度: {trigger_result['intensity']}"""


@dataclass
class RelationshipMemory:
    """关系记忆"""

    user_id: str
    stream_id: str
    shared_experiences: List[Dict] = field(default_factory=list)
    inside_jokes: List[str] = field(default_factory=list)
    conflicts: List[Dict] = field(default_factory=list)
    milestones: List[Dict] = field(default_factory=list)
    preferences_learned: Dict[str, Any] = field(default_factory=dict)
    conversation_topics: Dict[str, int] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

    def add_shared_experience(
        self, description: str, sentiment: str = "neutral"
    ):
        self.shared_experiences.append(
            {
                "description": description[:200],
                "sentiment": sentiment,
                "time": time.time(),
            }
        )
        if len(self.shared_experiences) > 50:
            self.shared_experiences = self.shared_experiences[-50:]
        self.last_updated = time.time()

    def add_inside_joke(self, joke: str):
        if joke not in self.inside_jokes:
            self.inside_jokes.append(joke[:100])
            if len(self.inside_jokes) > 20:
                self.inside_jokes = self.inside_jokes[-20:]
        self.last_updated = time.time()

    def add_conflict(self, description: str, resolved: bool = False):
        self.conflicts.append(
            {
                "description": description[:200],
                "resolved": resolved,
                "time": time.time(),
            }
        )
        if len(self.conflicts) > 30:
            self.conflicts = self.conflicts[-30:]
        self.last_updated = time.time()

    def add_milestone(self, milestone_type: str, description: str):
        self.milestones.append(
            {
                "type": milestone_type,
                "description": description[:100],
                "time": time.time(),
            }
        )
        if len(self.milestones) > 30:
            self.milestones = self.milestones[-30:]
        self.last_updated = time.time()

    def learn_preference(self, key: str, value: Any):
        self.preferences_learned[key] = value
        self.last_updated = time.time()

    def record_topic(self, topic: str):
        topic_lower = topic.lower()[:50]
        self.conversation_topics[topic_lower] = (
            self.conversation_topics.get(topic_lower, 0) + 1
        )
        self.last_updated = time.time()

    def get_top_topics(self, limit: int = 5) -> List[Tuple[str, int]]:
        return sorted(
            self.conversation_topics.items(), key=lambda x: x[1], reverse=True
        )[:limit]

    def get_recent_shared_experiences(self, limit: int = 5) -> List[Dict]:
        return self.shared_experiences[-limit:]

    def get_unresolved_conflicts(self) -> List[Dict]:
        return [c for c in self.conflicts if not c.get("resolved", False)]

    def build_memory_context(self) -> str:
        lines = ["[关系记忆]"]
        if self.milestones:
            recent_milestones = self.milestones[-3:]
            milestone_strs = [m["description"] for m in recent_milestones]
            lines.append(f"重要时刻: {', '.join(milestone_strs)}")
        if self.inside_jokes:
            lines.append(f"我们之间的梗: {', '.join(self.inside_jokes[:3])}")
        top_topics = self.get_top_topics(3)
        if top_topics:
            topic_strs = [t[0] for t in top_topics]
            lines.append(f"常聊话题: {', '.join(topic_strs)}")
        unresolved = self.get_unresolved_conflicts()
        if unresolved:
            lines.append(f"未解决的矛盾: {len(unresolved)}个")
        if self.preferences_learned:
            pref_strs = [
                f"{k}:{v}"
                for k, v in list(self.preferences_learned.items())[:3]
            ]
            lines.append(f"了解到的偏好: {', '.join(pref_strs)}")
        return "\n".join(lines) if len(lines) > 1 else ""


class RelationshipMemoryManager:
    """关系记忆管理器"""

    _memories: Dict[str, RelationshipMemory] = {}

    @classmethod
    def get_memory(cls, user_id: str, stream_id: str) -> RelationshipMemory:
        key = f"{stream_id}:{user_id}"
        if key not in cls._memories:
            cls._memories[key] = RelationshipMemory(
                user_id=user_id, stream_id=stream_id
            )
        return cls._memories[key]

    @classmethod
    def save_all(cls):
        logger.debug(f"保存 {len(cls._memories)} 条关系记忆")

    @classmethod
    def load_all(cls):
        logger.debug("加载关系记忆")

    @classmethod
    def get_all_memories(cls) -> Dict[str, RelationshipMemory]:
        return cls._memories.copy()


_affection_dynamics_instance: Optional["AffectionDynamicsManager"] = None


class AffectionDynamicsManager:
    """情感动力学管理器"""

    def __init__(self, config_engine: Optional[ConfigEngine] = None):
        self._config = config_engine or ConfigEngine.get_instance()
        self._emotion_contagion = EmotionContagion()
        self._context_analyzer = ConversationContextAnalyzer()
        self._affection_triggers = AffectionTriggers()
        self._memory_manager = RelationshipMemoryManager()
        self._load_config()

    def _load_config(self):
        affection_cfg = self._config.get("affection_dynamics", {})
        self._max_affection = affection_cfg.get("max_affection", 100.0)
        self._max_trust = affection_cfg.get("max_trust", 100.0)
        self._max_annoyance = affection_cfg.get("max_annoyance", 100.0)

    def get_memory(self, user_id: str, stream_id: str) -> RelationshipMemory:
        return self._memory_manager.get_memory(user_id, stream_id)

    def build_emotion_analysis_prompt(self, user_message: str) -> str:
        return self._emotion_contagion.build_emotion_analysis_prompt(
            user_message
        )

    def build_contagion_context(
        self, detected_emotion: str, intensity: str
    ) -> str:
        return self._emotion_contagion.build_contagion_context(
            detected_emotion, intensity
        )

    def parse_emotion_result(self, llm_response: str) -> Dict[str, Any]:
        return self._emotion_contagion.parse_emotion_result(llm_response)

    def build_context_analysis_prompt(self, user_message: str) -> str:
        return self._context_analyzer.build_analysis_prompt(user_message)

    def parse_context_result(self, llm_response: str) -> Dict[str, Any]:
        return self._context_analyzer.parse_context_result(llm_response)

    def build_affection_analysis_prompt(
        self, user_message: str, current_affection: float
    ) -> str:
        return self._affection_triggers.build_analysis_prompt(
            user_message, current_affection
        )

    def parse_affection_result(self, llm_response: str) -> Dict[str, Any]:
        return self._affection_triggers.parse_trigger_result(llm_response)

    def apply_affection_change(
        self,
        current_affection: float,
        current_trust: float,
        current_annoyance: float,
        affection_delta: float,
        trust_delta: float,
        annoyance_delta: float,
    ) -> Tuple[float, float, float]:
        new_affection = max(
            -self._max_affection,
            min(self._max_affection, current_affection + affection_delta),
        )
        new_trust = max(
            -self._max_trust, min(self._max_trust, current_trust + trust_delta)
        )
        new_annoyance = max(
            0, min(self._max_annoyance, current_annoyance + annoyance_delta)
        )
        return new_affection, new_trust, new_annoyance


def get_affection_dynamics() -> AffectionDynamicsManager:
    """获取情感动力学管理器单例"""
    global _affection_dynamics_instance
    if _affection_dynamics_instance is None:
        _affection_dynamics_instance = AffectionDynamicsManager()
    return _affection_dynamics_instance

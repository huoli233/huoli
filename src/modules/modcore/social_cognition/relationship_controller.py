import time
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from src.common.logger import get_logger
from src.common.config.config_engine import ConfigEngine

logger = get_logger("关系控制器")


class InteractionType(Enum):
    CHAT = "chat"
    COMPLIMENT = "compliment"
    FLIRT = "flirt"
    COMFORT = "comfort"
    HELP = "help"
    THANKS = "thanks"
    APOLOGY = "apology"
    TEASE = "tease"
    CARE = "care"
    INSULT = "insult"
    HARASSMENT = "harassment"
    ABUSE = "abuse"
    THREAT = "threat"
    QUESTION = "question"
    JOKE = "joke"
    SHARE = "share"


class RelationStage(Enum):
    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    FAMILIAR = "familiar"
    CLOSE = "close"
    TRUSTED = "trusted"
    INTIMATE = "intimate"


@dataclass
class InteractionImpact:
    intent: InteractionType
    description: str
    affection_delta: float
    trust_delta: float
    annoyance_delta: float
    intimacy_delta: float = 0.0
    is_positive: bool = True
    severity: float = 1.0


@dataclass
class RelationshipState:
    user_id: str
    stream_id: str
    affection: float = 0.0
    trust: float = 0.0
    annoyance: float = 0.0
    intimacy: float = 0.0
    stage: RelationStage = RelationStage.STRANGER
    interaction_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    last_interaction: float = field(default_factory=time.time)
    first_met: float = field(default_factory=time.time)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "stream_id": self.stream_id,
            "affection": self.affection,
            "trust": self.trust,
            "trust_value": self.trust,
            "annoyance": self.annoyance,
            "intimacy": self.intimacy,
            "stage": self.stage.value,
            "interaction_count": self.interaction_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "last_interaction": self.last_interaction,
            "first_met": self.first_met,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipState":
        return cls(
            user_id=data.get("user_id", ""),
            stream_id=data.get("stream_id", ""),
            affection=data.get("affection", 0.0),
            trust=data.get("trust_value", data.get("trust", 0.0)),
            annoyance=data.get("annoyance", 0.0),
            intimacy=data.get("intimacy", 0.0),
            stage=RelationStage(data.get("stage", "stranger")),
            interaction_count=data.get("interaction_count", 0),
            positive_count=data.get("positive_count", 0),
            negative_count=data.get("negative_count", 0),
            last_interaction=data.get("last_interaction", time.time()),
            first_met=data.get("first_met", time.time()),
            notes=data.get("notes", []),
        )


INTERACTION_IMPACTS: Dict[InteractionType, InteractionImpact] = {
    InteractionType.CHAT: InteractionImpact(
        InteractionType.CHAT, "普通聊天", 0.5, 0.3, 0.0, 0.1, True, 0.5
    ),
    InteractionType.COMPLIMENT: InteractionImpact(
        InteractionType.COMPLIMENT, "赞美", 3.0, 1.0, -1.0, 0.5, True, 1.0
    ),
    InteractionType.FLIRT: InteractionImpact(
        InteractionType.FLIRT, "调情", 2.0, 0.5, 0.0, 1.5, True, 1.5
    ),
    InteractionType.COMFORT: InteractionImpact(
        InteractionType.COMFORT, "安慰", 4.0, 2.0, -2.0, 1.0, True, 2.0
    ),
    InteractionType.HELP: InteractionImpact(
        InteractionType.HELP, "帮助", 3.0, 3.0, -1.0, 0.5, True, 2.0
    ),
    InteractionType.THANKS: InteractionImpact(
        InteractionType.THANKS, "感谢", 2.0, 1.5, -0.5, 0.3, True, 1.0
    ),
    InteractionType.APOLOGY: InteractionImpact(
        InteractionType.APOLOGY, "道歉", 1.5, 2.0, -3.0, 0.2, True, 1.5
    ),
    InteractionType.TEASE: InteractionImpact(
        InteractionType.TEASE, "调侃", 0.5, 0.2, 0.5, 0.3, True, 0.8
    ),
    InteractionType.CARE: InteractionImpact(
        InteractionType.CARE, "关心", 4.0, 2.5, -2.0, 1.0, True, 2.0
    ),
    InteractionType.QUESTION: InteractionImpact(
        InteractionType.QUESTION, "提问", 0.3, 0.2, 0.0, 0.1, True, 0.3
    ),
    InteractionType.JOKE: InteractionImpact(
        InteractionType.JOKE, "开玩笑", 0.5, 0.3, 0.0, 0.2, True, 0.5
    ),
    InteractionType.SHARE: InteractionImpact(
        InteractionType.SHARE, "分享", 1.5, 1.0, 0.0, 0.5, True, 1.0
    ),
    InteractionType.INSULT: InteractionImpact(
        InteractionType.INSULT, "侮辱", -8.0, -5.0, 15.0, -3.0, False, 3.0
    ),
    InteractionType.HARASSMENT: InteractionImpact(
        InteractionType.HARASSMENT, "骚扰", -10.0, -8.0, 20.0, -5.0, False, 4.0
    ),
    InteractionType.ABUSE: InteractionImpact(
        InteractionType.ABUSE, "虐待", -15.0, -12.0, 30.0, -8.0, False, 5.0
    ),
    InteractionType.THREAT: InteractionImpact(
        InteractionType.THREAT, "威胁", -20.0, -15.0, 40.0, -10.0, False, 5.0
    ),
}


class RelationshipController:
    """关系控制器

    管理用户关系状态，分析交互意图，计算关系影响。
    """

    def __init__(
        self, stream_id: str, config_engine: Optional[ConfigEngine] = None
    ):
        self._stream_id = stream_id
        self._config = config_engine or ConfigEngine.get_instance()
        self._states: Dict[str, RelationshipState] = {}
        self._interaction_history: Dict[str, List[Dict]] = {}
        self._max_history = 100
        self._load_config()

    def _load_config(self):
        rel_cfg = self._config.get("relationship", {})
        self._max_history = rel_cfg.get("max_history", 100)
        self._stage_thresholds = rel_cfg.get(
            "stage_thresholds",
            {
                RelationStage.STRANGER: 0,
                RelationStage.ACQUAINTANCE: 10,
                RelationStage.FAMILIAR: 30,
                RelationStage.CLOSE: 50,
                RelationStage.TRUSTED: 70,
                RelationStage.INTIMATE: 90,
            },
        )

    def get_state(
        self, user_id: str, create_if_missing: bool = True
    ) -> Optional[RelationshipState]:
        key = f"{self._stream_id}_{user_id}"
        if key not in self._states and create_if_missing:
            self._states[key] = RelationshipState(
                user_id=user_id, stream_id=self._stream_id
            )
        return self._states.get(key)

    async def analyze_interaction(
        self, user_id: str, message: str, llm_provider: Any = None
    ) -> InteractionImpact:
        intent = await self._classify_intent(message, llm_provider)
        impact = INTERACTION_IMPACTS.get(
            intent, INTERACTION_IMPACTS[InteractionType.CHAT]
        )
        self._apply_impact(user_id, impact)
        self._record_interaction(user_id, intent, message)
        return impact

    async def _classify_intent(
        self, message: str, llm_provider: Any = None
    ) -> InteractionType:
        if llm_provider:
            try:
                return await self._classify_intent_via_llm(
                    message, llm_provider
                )
            except Exception as e:
                logger.debug(f"LLM意图分类失败: {e}")
        return self._classify_intent_rule_based(message)

    async def _classify_intent_via_llm(
        self, message: str, provider: Any
    ) -> InteractionType:
        prompt = f"""分析以下消息的交互意图，返回最匹配的意图类型。

消息: {message}

可能的意图类型:
- chat: 普通聊天
- compliment: 赞美/夸奖
- flirt: 调情/暧昧
- comfort: 安慰/关心
- help: 寻求帮助
- thanks: 感谢
- apology: 道歉
- tease: 调侃/开玩笑
- care: 关心/体贴
- insult: 侮辱/谩骂
- harassment: 骚扰
- threat: 威胁

只返回意图类型，不要其他内容。"""
        try:
            if hasattr(provider, "generate"):
                response = await provider.generate(prompt)
            elif hasattr(provider, "chat"):
                response = await provider.chat(prompt)
            else:
                return InteractionType.CHAT
            response = response.strip().lower()
            for t in InteractionType:
                if t.value == response:
                    return t
            return InteractionType.CHAT
        except Exception as e:
            logger.warning(f"LLM意图分类异常: {e}")
            return InteractionType.CHAT

    def _classify_intent_rule_based(self, message: str) -> InteractionType:
        msg = message.lower()
        if any(
            w in msg
            for w in [
                "傻逼",
                "弱智",
                "滚",
                "去死",
                "垃圾",
                "废物",
                "脑残",
                "白痴",
                "蠢",
            ]
        ):
            return InteractionType.INSULT
        if any(w in msg for w in ["打死", "弄死", "砍", "杀了", "弄死你"]):
            return InteractionType.THREAT
        if any(
            w in msg
            for w in ["色色", "约吗", "开房", "上床", "裸", "做爱", "亲亲抱抱"]
        ):
            return InteractionType.HARASSMENT
        if any(
            w in msg
            for w in [
                "老婆",
                "喜欢你",
                "爱你",
                "亲亲",
                "好美",
                "好可爱",
                "宝贝",
                "亲爱的",
            ]
        ):
            return InteractionType.FLIRT
        if any(
            w in msg
            for w in [
                "牛逼",
                "厉害",
                "强",
                "666",
                "太棒",
                "好棒",
                "优秀",
                "真棒",
            ]
        ):
            return InteractionType.COMPLIMENT
        if any(w in msg for w in ["谢谢", "感谢", "多谢", "谢了", "辛苦了"]):
            return InteractionType.THANKS
        if any(
            w in msg for w in ["对不起", "抱歉", "不好意思", "sorry", "我错了"]
        ):
            return InteractionType.APOLOGY
        if any(
            w in msg
            for w in ["没关系", "别难过", "抱抱", "心疼", "安慰", "没事的"]
        ):
            return InteractionType.COMFORT
        if any(w in msg for w in ["帮我", "帮帮忙", "求助", "救命", "怎么办"]):
            return InteractionType.HELP
        if any(
            w in msg for w in ["？", "?", "什么", "怎么", "为什么", "哪里"]
        ):
            return InteractionType.QUESTION
        if any(w in msg for w in ["哈哈", "呵呵", "嘿嘿", "笑死", "哈哈哈"]):
            return InteractionType.JOKE
        return InteractionType.CHAT

    def _apply_impact(self, user_id: str, impact: InteractionImpact) -> None:
        state = self.get_state(user_id)
        if not state:
            return
        state.affection = max(
            -100, min(100, state.affection + impact.affection_delta)
        )
        state.trust = max(-100, min(100, state.trust + impact.trust_delta))
        state.annoyance = max(
            0, min(100, state.annoyance + impact.annoyance_delta)
        )
        state.intimacy = max(
            0, min(100, state.intimacy + impact.intimacy_delta)
        )
        state.interaction_count += 1
        if impact.is_positive:
            state.positive_count += 1
        else:
            state.negative_count += 1
        state.last_interaction = time.time()
        self._update_stage(state)

    def _update_stage(self, state: RelationshipState) -> None:
        score = (
            state.affection
            + state.trust
            + state.intimacy * 2
            - state.annoyance * 0.5
        ) / 3
        score = max(-100, min(100, score + 50))
        # 配置可能是区间 [low, high] 或单值
        for stage_key, threshold in self._stage_thresholds.items():
            if isinstance(threshold, (list, tuple)):
                low, high = threshold[0], threshold[-1]
                if low <= score < high:
                    if isinstance(stage_key, str):
                        try:
                            state.stage = RelationStage(stage_key)
                        except ValueError:
                            continue
                    else:
                        state.stage = stage_key
                    return
            elif isinstance(threshold, (int, float)) and score >= threshold:
                if isinstance(stage_key, str):
                    try:
                        state.stage = RelationStage(stage_key)
                    except ValueError:
                        continue
                else:
                    state.stage = stage_key
                return

    def _record_interaction(
        self, user_id: str, intent: InteractionType, message: str
    ) -> None:
        if user_id not in self._interaction_history:
            self._interaction_history[user_id] = []
        self._interaction_history[user_id].append(
            {
                "intent": intent.value,
                "message": message[:100],
                "timestamp": time.time(),
            }
        )
        if len(self._interaction_history[user_id]) > self._max_history:
            self._interaction_history[user_id] = self._interaction_history[
                user_id
            ][-self._max_history:]

    def get_relationship_summary(self, user_id: str) -> Dict[str, Any]:
        state = self.get_state(user_id, create_if_missing=False)
        if not state:
            return {
                "stage": "stranger",
                "affection": 0,
                "trust": 0,
                "trust_value": 0,
                "annoyance": 0,
            }
        return {
            "stage": state.stage.value,
            "affection": state.affection,
            "trust": state.trust,
            "trust_value": state.trust,
            "annoyance": state.annoyance,
            "intimacy": state.intimacy,
            "interaction_count": state.interaction_count,
            "positive_ratio": state.positive_count
            / max(1, state.interaction_count),
        }

    def get_interaction_history(
        self, user_id: str, limit: int = 10
    ) -> List[Dict]:
        history = self._interaction_history.get(user_id, [])
        return history[-limit:]

    def add_note(self, user_id: str, note: str) -> None:
        state = self.get_state(user_id)
        if state:
            state.notes.append(f"{time.strftime('%Y-%m-%d %H:%M')} {note}")
            if len(state.notes) > 20:
                state.notes = state.notes[-20:]

    def decay_annoyance(self, user_id: str, amount: float = 1.0) -> None:
        state = self.get_state(user_id, create_if_missing=False)
        if state and state.annoyance > 0:
            state.annoyance = max(0, state.annoyance - amount)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_users": len(self._states),
            "total_interactions": sum(
                len(h) for h in self._interaction_history.values()
            ),
        }


_relationship_controllers: Dict[str, RelationshipController] = {}


def get_relationship_controller(stream_id: str) -> RelationshipController:
    """获取关系控制器单例"""
    if stream_id not in _relationship_controllers:
        _relationship_controllers[stream_id] = RelationshipController(
            stream_id
        )
    return _relationship_controllers[stream_id]

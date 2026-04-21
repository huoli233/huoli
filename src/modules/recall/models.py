from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RecallReason(Enum):
    """撤回原因"""

    TYPO = "typo"
    AWKWARD = "awkward"
    INAPPROPRIATE = "inappropriate"
    REGRET = "regret"
    MISUNDERSTOOD = "misunderstood"
    TOO_HASTY = "too_hasty"
    USER_NEGATIVE_REACTION = "user_negative_reaction"
    CORRECTION_NEEDED = "correction_needed"
    NULL = "null"


class AfterRecallAction(Enum):
    """撤回后行为"""

    CORRECT = "correct"
    SUPPLEMENT = "supplement"
    CASUAL_CORRECT = "casual_correct"
    SILENT = "silent"
    REGRET = "regret"
    SHY = "shy"
    APOLOGIZE = "apologize"
    REDIRECT = "redirect"
    SELF_DEPRECATE = "self_deprecate"
    RESEND = "resend"


class TypoType(Enum):
    """打错字类型"""

    TYPO = "typo"
    SWAP = "swap"
    OMIT = "omit"
    DUPLICATE = "duplicate"
    WRONG_CHAR = "wrong_char"
    EXTRA_CHAR = "extra_char"
    HOMOPHONE = "homophone"


@dataclass
class DimensionFactors:
    """维度因素集合"""

    social_value: float = 0.5
    relationship_depth: float = 0.5
    trust_value: float = 0.0
    annoyance_value: float = 0.0
    interaction_count: int = 0
    emotional_state: str = "平静"
    trauma_score: float = 0.0
    inner_chaos: float = 0.0
    persona_state: str = "default"
    is_private: bool = False
    group_atmosphere: float = 0.5
    group_member_count: int = 0
    group_activity: float = 0.5
    bystander_effect: float = 0.0
    time_of_day: str = "白天"
    time_elapsed: float = 0.0
    recent_typo_count: int = 0
    recent_recall_count: int = 0
    message_importance: float = 0.5
    topic_type: str = "日常"
    content_sensitivity: float = 0.0
    is_question: bool = False
    content_length: int = 0
    user_reaction: str = ""
    reaction_type: str = "无"
    reaction_content: str = ""
    conversation_context: str = ""
    main_personality: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "social_value": self.social_value,
            "relationship_depth": self.relationship_depth,
            "trust_value": self.trust_value,
            "annoyance_value": self.annoyance_value,
            "interaction_count": self.interaction_count,
            "emotional_state": self.emotional_state,
            "trauma_score": self.trauma_score,
            "inner_chaos": self.inner_chaos,
            "persona_state": self.persona_state,
            "is_private": self.is_private,
            "group_atmosphere": self.group_atmosphere,
            "group_member_count": self.group_member_count,
            "group_activity": self.group_activity,
            "bystander_effect": self.bystander_effect,
            "time_of_day": self.time_of_day,
            "time_elapsed": self.time_elapsed,
            "recent_typo_count": self.recent_typo_count,
            "recent_recall_count": self.recent_recall_count,
            "message_importance": self.message_importance,
            "topic_type": self.topic_type,
            "content_sensitivity": self.content_sensitivity,
            "is_question": self.is_question,
            "content_length": self.content_length,
            "user_reaction": self.user_reaction,
            "reaction_type": self.reaction_type,
            "reaction_content": self.reaction_content,
            "conversation_context": self.conversation_context,
            "main_personality": self.main_personality,
        }


@dataclass
class RecallDecision:
    """撤回决策"""

    should_recall: bool = False
    recall_reason: Optional[RecallReason] = None
    delay_seconds: float = 5.0
    after_action: AfterRecallAction = AfterRecallAction.CORRECT
    confidence: float = 0.5
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "should_recall": self.should_recall,
            "recall_reason": (
                self.recall_reason.value if self.recall_reason else None
            ),
            "delay_seconds": self.delay_seconds,
            "after_action": self.after_action.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class TypoDecision:
    """打错字决策"""

    should_typo: bool = False
    typo_type: Optional[TypoType] = None
    error_content: str = ""
    original_content: str = ""
    confidence: float = 0.5
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "should_typo": self.should_typo,
            "typo_type": self.typo_type.value if self.typo_type else None,
            "error_content": self.error_content,
            "original_content": self.original_content,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class PendingRecall:
    """待处理撤回"""

    msg_id: str = ""
    stream_id: str = ""
    channel_id: str = ""
    user_id: str = ""
    sent_content: str = ""
    original_content: str = ""
    scheduled_at: float = 0.0
    recall_at: float = 0.0
    recalled: bool = False
    completed: bool = False
    recall_decision: Optional[RecallDecision] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "msg_id": self.msg_id,
            "stream_id": self.stream_id,
            "channel_id": self.channel_id,
            "user_id": self.user_id,
            "sent_content": self.sent_content,
            "original_content": self.original_content,
            "scheduled_at": self.scheduled_at,
            "recall_at": self.recall_at,
            "recalled": self.recalled,
            "completed": self.completed,
            "recall_decision": (
                self.recall_decision.to_dict()
                if self.recall_decision
                else None
            ),
            "extra": self.extra,
        }


@dataclass
class CorrectionEvent:
    """纠正事件"""

    stream_id: str = ""
    bot_msg_id: str = ""
    bot_content: str = ""
    sent_at: float = 0.0
    user_reply: str = ""
    detected_at: float = 0.0
    need_recall: bool = False
    reason: str = ""
    confidence: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "stream_id": self.stream_id,
            "bot_msg_id": self.bot_msg_id,
            "bot_content": self.bot_content,
            "sent_at": self.sent_at,
            "user_reply": self.user_reply,
            "detected_at": self.detected_at,
            "need_recall": self.need_recall,
            "reason": self.reason,
            "confidence": self.confidence,
            "extra": self.extra,
        }


@dataclass
class SentMessage:
    """已发送消息记录"""

    msg_id: str = ""
    stream_id: str = ""
    channel_id: str = ""
    content: str = ""
    content_type: str = "text"
    sent_at: float = 0.0
    recalled: bool = False
    recall_reason: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "msg_id": self.msg_id,
            "stream_id": self.stream_id,
            "channel_id": self.channel_id,
            "content": self.content,
            "content_type": self.content_type,
            "sent_at": self.sent_at,
            "recalled": self.recalled,
            "recall_reason": self.recall_reason,
            "context": self.context,
        }


@dataclass
class ActionRecord:
    """行为记录"""

    stream_id: str = ""
    action_type: str = ""
    target: str = ""
    content: str = ""
    timestamp: float = 0.0
    result: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "stream_id": self.stream_id,
            "action_type": self.action_type,
            "target": self.target,
            "content": self.content,
            "timestamp": self.timestamp,
            "result": self.result,
            "extra": self.extra,
        }


@dataclass
class UserEvent:
    """用户事件"""

    stream_id: str = ""
    user_id: str = ""
    event_type: str = ""
    content: str = ""
    timestamp: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "stream_id": self.stream_id,
            "user_id": self.user_id,
            "event_type": self.event_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "extra": self.extra,
        }


@dataclass
class LearnedPattern:
    """学习到的模式"""

    stream_id: str = ""
    relation_stage: str = ""
    action_type: str = ""
    style_hint: str = ""
    confidence: float = 0.5
    sample_count: int = 0
    learned_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "stream_id": self.stream_id,
            "relation_stage": self.relation_stage,
            "action_type": self.action_type,
            "style_hint": self.style_hint,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "learned_at": self.learned_at,
        }

from enum import Enum, auto


class ConversationState(Enum):
    """对话状态枚举。"""

    INIT = auto()
    IDLE = auto()
    ANALYZING = auto()
    PLANNING = auto()
    GENERATING = auto()
    CHECKING = auto()
    WAITING = auto()
    LISTENING = auto()
    FETCHING = auto()
    RETHINKING = auto()
    SENDING = auto()
    IGNORED = auto()
    ENDED = auto()


class ActionType(Enum):
    """动作类型枚举。"""

    DIRECT_REPLY = "direct_reply"
    SEND_NEW_MESSAGE = "send_new_message"
    FETCH_KNOWLEDGE = "fetch_knowledge"
    RETHINK_GOAL = "rethink_goal"
    LISTENING = "listening"
    SAY_GOODBYE = "say_goodbye"
    END_CONVERSATION = "end_conversation"
    BLOCK_AND_IGNORE = "block_and_ignore"
    WAIT = "wait"


class GoalStatus(Enum):
    """目标状态枚举。"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"


class ConversationPhase(Enum):
    """对话阶段枚举。"""

    GREETING = "greeting"
    EXPLORING = "exploring"
    DEEPENING = "deepening"
    CLOSING = "closing"
    ENDED = "ended"

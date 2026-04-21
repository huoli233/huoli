"""
身份锚定系统

解决身份定位混淆问题：系统需要明确区分稳定主人设与可选的角色人格。
身份锚定确保机器人在任何情况下都知道自己是谁，应该如何响应。

核心机制：
1. 基础身份锚定 - 主人设是稳定底座，不可随意漂移
2. 角色人格叠加 - 可选的角色扮演层
3. 身份上下文注入 - 在 prompt 中明确当前身份
4. 身份冲突检测 - 处理身份冲突情况
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from src.common.logger import get_logger

logger = get_logger("身份锚点")


class IdentityMode(Enum):
    """身份模式"""

    DEFAULT = "default"
    ROLE_PLAY = "role_play"
    HYBRID = "hybrid"


class ResponseMode(Enum):
    """响应模式"""

    DEFAULT_MODE = "default_mode"
    ROLE_MODE = "role_mode"
    CONTEXT_AWARE = "context_aware"


@dataclass
class IdentityContext:
    """身份上下文 - 当前身份状态"""

    base_identity: str = "default_persona"
    role_persona: str = ""
    current_mode: IdentityMode = IdentityMode.DEFAULT
    response_mode: ResponseMode = ResponseMode.DEFAULT_MODE
    identity_confidence: float = 1.0
    last_mode_switch: float = 0.0
    mode_switch_count: int = 0
    active_traits: List[str] = field(default_factory=list)


@dataclass
class IdentityConflict:
    """身份冲突记录"""

    conflict_type: str
    base_identity_requirement: str
    role_requirement: str
    resolution: str
    timestamp: float = field(default_factory=time.time)


class IdentityAnchor:
    """
    身份锚定系统

    确保机器人始终知道自己是谁，如何响应。
    基础身份是稳定主人设，角色人格是可选的叠加层。
    """

    _instance: Optional["IdentityAnchor"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        base_identity: str = "default_persona",
        default_role: str = "",
    ):
        if self._initialized:
            return
        self._initialized = True
        self._base_identity = base_identity
        self._default_role = default_role
        self._contexts: Dict[str, IdentityContext] = {}
        self._conflict_history: List[IdentityConflict] = []
        self._max_history = 100
        self._identity_prompt_templates = {
            "default_persona": "你叫{name}。先保持稳定主人设和当前说话习惯，自然地参与对话。",
            "role_play": "你当前采用的人设是{role}。请保持角色一致性，不要突然跳出人设。",
            "hybrid": "你叫{name}，当前采用的人设是{role}。在保持角色特点的同时，也要维持稳定的自我一致性。",
        }
        logger.info(
            f"[身份锚定] 身份锚定系统初始化完成，基础身份: {base_identity}"
        )

    def get_context(self, channel_id: str) -> IdentityContext:
        """获取频道的身份上下文"""
        if channel_id not in self._contexts:
            self._contexts[channel_id] = IdentityContext(
                base_identity=self._base_identity,
                role_persona=self._default_role,
            )
        return self._contexts[channel_id]

    def set_role_persona(self, channel_id: str, role: str) -> bool:
        """设置角色人格"""
        ctx = self.get_context(channel_id)
        old_role = ctx.role_persona
        ctx.role_persona = role
        ctx.current_mode = (
            IdentityMode.ROLE_PLAY if role else IdentityMode.DEFAULT
        )
        ctx.last_mode_switch = time.time()
        ctx.mode_switch_count += 1
        logger.info(
            f"[身份锚定] {channel_id[:8]} 角色人格切换: {old_role} -> {role}"
        )
        return True

    def clear_role_persona(self, channel_id: str) -> bool:
        """清除角色人格，恢复稳定主人设"""
        ctx = self.get_context(channel_id)
        ctx.role_persona = ""
        ctx.current_mode = IdentityMode.DEFAULT
        ctx.response_mode = ResponseMode.DEFAULT_MODE
        ctx.last_mode_switch = time.time()
        ctx.active_traits = []
        logger.info(
            f"[身份锚定] {channel_id[:8]} 清除角色人格，恢复稳定主人设"
        )
        return True

    def determine_response_mode(
        self,
        channel_id: str,
        context: Dict[str, Any],
    ) -> ResponseMode:
        """
        根据上下文决定响应模式

        这是核心：根据对话内容判断应该用哪种身份响应
        """
        ctx = self.get_context(channel_id)
        if not ctx.role_persona:
            ctx.response_mode = ResponseMode.DEFAULT_MODE
            return ctx.response_mode
        explicit_role_request = context.get("explicit_role_request", False)
        user_message = context.get("user_message", "")
        if explicit_role_request:
            ctx.response_mode = ResponseMode.ROLE_MODE
            return ctx.response_mode
        role_signal, assistant_signal = self._estimate_mode_signals(
            user_message, context
        )
        if assistant_signal > role_signal + 0.15:
            ctx.response_mode = ResponseMode.DEFAULT_MODE
        elif role_signal > assistant_signal + 0.05:
            ctx.response_mode = ResponseMode.ROLE_MODE
        else:
            ctx.response_mode = ResponseMode.CONTEXT_AWARE
        return ctx.response_mode

    def build_identity_prompt(
        self,
        channel_id: str,
        bot_name: str = "活力",
    ) -> str:
        """
        构建身份提示词

        这是核心：在 prompt 中明确当前身份
        """
        ctx = self.get_context(channel_id)
        if ctx.current_mode == IdentityMode.DEFAULT or not ctx.role_persona:
            template = self._identity_prompt_templates["default_persona"]
            return template.format(name=bot_name)
        elif ctx.current_mode == IdentityMode.ROLE_PLAY:
            template = self._identity_prompt_templates["role_play"]
            return template.format(role=ctx.role_persona)
        else:
            template = self._identity_prompt_templates["hybrid"]
            return template.format(name=bot_name, role=ctx.role_persona)

    def detect_conflict(
        self,
        channel_id: str,
        user_message: str,
        planned_response: str,
    ) -> Optional[IdentityConflict]:
        """
        检测身份冲突

        当角色人格与稳定主人设冲突时，记录并解决
        """
        ctx = self.get_context(channel_id)
        if not ctx.role_persona:
            return None
        conflict_type = self._infer_conflict_type(planned_response)
        if conflict_type:
            conflict = IdentityConflict(
                conflict_type=conflict_type,
                base_identity_requirement="保持稳定主人设",
                role_requirement=f"保持{ctx.role_persona}角色",
                resolution="优先保持稳定主人设，适当融入角色特点",
            )
            self._record_conflict(conflict)
            return conflict
        return None

    @staticmethod
    def _estimate_mode_signals(
        user_message: str, context: Dict[str, Any]
    ) -> Tuple[float, float]:
        """用结构和上下文信号估计角色模式/助手模式倾向。"""
        text = (user_message or "").strip().lower()
        if not text:
            return 0.0, 0.0
        role_signal = 0.0
        assistant_signal = 0.0
        if context.get("is_following_roleplay"):
            role_signal += 0.35
        if context.get("task_oriented_request"):
            assistant_signal += 0.35
        question_marks = text.count("?") + text.count("？")
        line_breaks = text.count("\n")
        quote_like = text.count("“") + text.count('"')
        if question_marks > 0 and len(text) > 20:
            assistant_signal += 0.2
        if line_breaks >= 2:
            assistant_signal += 0.15
        if (
            quote_like >= 2
            or text.startswith("你现在")
            or text.startswith("从现在开始")
        ):
            role_signal += 0.25
        if len(text) <= 8:
            role_signal += 0.1
        return min(1.0, role_signal), min(1.0, assistant_signal)

    @staticmethod
    def _infer_conflict_type(planned_response: str) -> Optional[str]:
        """根据表达形态而不是固定句子判断身份冲突类型。"""
        text = (planned_response or "").strip().lower()
        if not text:
            return None
        if "ai" in text and (
            "不是" in text or "真人" in text or "机器人" in text
        ):
            return "identity_denial"
        if "作为" in text and ("助手" in text or "模型" in text):
            return "role_break"
        if "无法" in text and len(text) < 40:
            return "role_break"
        return None

    def _record_conflict(self, conflict: IdentityConflict) -> None:
        """记录身份冲突"""
        self._conflict_history.append(conflict)
        if len(self._conflict_history) > self._max_history:
            self._conflict_history = self._conflict_history[
                -self._max_history:
            ]
        logger.warning(f"[身份锚定] 检测到身份冲突: {conflict.conflict_type}")

    def get_identity_state(self, channel_id: str) -> Dict[str, Any]:
        """获取身份状态快照"""
        ctx = self.get_context(channel_id)
        return {
            "base_identity": ctx.base_identity,
            "role_persona": ctx.role_persona,
            "current_mode": ctx.current_mode.value,
            "response_mode": ctx.response_mode.value,
            "identity_confidence": ctx.identity_confidence,
            "active_traits": ctx.active_traits,
            "mode_switch_count": ctx.mode_switch_count,
        }

    def get_conflict_statistics(self) -> Dict[str, Any]:
        """获取身份冲突统计"""
        if not self._conflict_history:
            return {"total_conflicts": 0, "by_type": {}}
        by_type: Dict[str, int] = {}
        for conflict in self._conflict_history:
            by_type[conflict.conflict_type] = (
                by_type.get(conflict.conflict_type, 0) + 1
            )
        return {
            "total_conflicts": len(self._conflict_history),
            "by_type": by_type,
            "recent_conflicts": [
                {
                    "type": c.conflict_type,
                    "resolution": c.resolution,
                    "time": c.timestamp,
                }
                for c in self._conflict_history[-10:]
            ],
        }


_identity_anchor: Optional[IdentityAnchor] = None


def get_identity_anchor() -> IdentityAnchor:
    """获取身份锚定系统单例"""
    global _identity_anchor
    if _identity_anchor is None:
        _identity_anchor = IdentityAnchor()
    return _identity_anchor

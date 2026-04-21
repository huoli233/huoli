import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger("LLM自主决策")


@dataclass
class EnvironmentSnapshot:
    """环境快照 - LLM 感知的上下文"""

    channel_id: str = ""
    channel_name: str = ""
    silence_seconds: float = 0.0
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    active_users: List[str] = field(default_factory=list)
    current_topics: List[str] = field(default_factory=list)
    atmosphere: str = "neutral"
    my_last_message: str = ""
    my_last_message_time: float = 0.0
    energy_level: float = 1.0
    mood: str = "neutral"
    social_value: float = 0.0
    time_of_day: str = ""
    day_of_week: str = ""
    # 内心独白结果
    inner_voice_desire: int = 5
    inner_voice_thinking: str = ""
    inner_voice_mood: str = ""
    inner_voice_primary_intent: str = ""
    inner_voice_needs_upgrade: bool = False
    # 对话阶段与疲劳补充
    dialogue_phase: str = ""
    trauma_pressure: float = 0.0
    # 叙事规划与主观印象
    narration_hint: str = ""
    impression_hint: str = ""
    # 跨轮意图（待处理事项）
    pending_intentions_hint: str = ""


@dataclass
class AutonomousDecision:
    """自主决策结果"""

    should_act: bool = False
    action_type: str = "observe"
    content_plan: str = ""
    reasoning: str = ""
    confidence: float = 0.5
    emotional_state: str = ""
    social_intention: str = ""
    target_user_id: str = ""
    pending_intentions_hint: str = ""


class LLMAutonomousPlanner:
    """
    LLM 驱动的自主规划器

    让 LLM 自己决定是否要主动发言、发什么内容。
    不是被规则推着走，而是 LLM 自己感知环境并做出决策。
    """

    _instance: Optional["LLMAutonomousPlanner"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._decision_history: List[AutonomousDecision] = []
        self._max_history = 50
        logger.info("[LLM自主] LLM 驱动的自主规划器初始化完成")

    async def perceive_and_decide(
        self,
        env: EnvironmentSnapshot,
    ) -> AutonomousDecision:
        """
        感知环境并做出决策

        这是核心：让 LLM 自己感知环境、自己决定要做什么。
        不是被规则触发，而是 LLM 主动思考。
        """
        try:
            prompt = self._build_perception_prompt(env)
            decision = await self._call_llm_for_decision(prompt, env)
            self._decision_history.append(decision)
            if len(self._decision_history) > self._max_history:
                self._decision_history = self._decision_history[
                    -self._max_history:
                ]
            if decision.should_act:
                logger.info(
                    f"[LLM自主] {env.channel_id[:8]} LLM决定主动行为: "
                    f"类型={decision.action_type}, "
                    f"意图={decision.social_intention}, "
                    f"内容规划={decision.content_plan[:50]}..."
                )
            else:
                logger.debug(
                    f"[LLM自主] {env.channel_id[:8]} LLM决定观察: {decision.reasoning[:80]}"
                )
            return decision
        except Exception as exc:
            logger.error(f"[LLM自主] LLM 决策失败: {exc}")
            return AutonomousDecision(
                should_act=False,
                action_type="observe",
                reasoning=f"LLM 调用失败: {exc}",
            )

    def _build_perception_prompt(self, env: EnvironmentSnapshot) -> str:
        """构建感知提示词 - 让 LLM 感知当前环境并决定是否要主动发言"""
        silence_min = env.silence_seconds / 60.0
        recent_summary = self._summarize_recent_messages(env.recent_messages)
        # 构建内心独白部分
        inner_voice_section = ""
        if (
            env.inner_voice_desire != 5
            or env.inner_voice_thinking
            or env.inner_voice_mood
        ):
            desire_desc = self._get_desire_description(env.inner_voice_desire)
            _intent_hint = ""
            if (
                hasattr(env, "inner_voice_primary_intent")
                and env.inner_voice_primary_intent
            ):
                _intent_hint = f"\n- 主意图: {env.inner_voice_primary_intent}"
            _upgrade_hint = ""
            _context_extend_hint = ""
            if (
                hasattr(env, "inner_voice_needs_upgrade")
                and env.inner_voice_needs_upgrade
            ):
                _upgrade_hint = (
                    "\n- 【重要】小模型认为这个话题你不够理解/比较复杂，"
                    "需要你用更全面的视角来决策是否回复"
                )
                _context_extend_hint = (
                    "\n**【深度分析要求】** 小模型索引的上下文有限，你需要："
                    "\n  1. 回顾更多历史消息，追溯话题起源和演变"
                    "\n  2. 分析对话中的人际关系动态和情感变化"
                    "\n  3. 识别潜在的深层意图和弦外之音"
                    "\n  4. 评估回复的长期影响和关系后果"
                    "\n  5. 考虑是否需要更谨慎或更有策略的回复方式"
                )
            else:
                _upgrade_hint = (
                    "\n- 小模型认为这个话题比较简单，"
                    "如果你决定回复，可以参考小模型的内心想法直接生成回复"
                )
            inner_voice_section = f"""
### 你的内心想法（重要参考）
- 回复欲望等级: {env.inner_voice_desire}/10 ({desire_desc})
- 你的思考: {env.inner_voice_thinking or '(无)'}
- 你的情绪: {env.inner_voice_mood or '(无)'}{_intent_hint}{_upgrade_hint}{_context_extend_hint}

**决策参考**：
- 如果欲望等级 >= 6 且话题简单 → 可以直接回复
- 如果欲望等级 < 4 → 建议不回复（观察即可）
- 如果小模型认为需要升级（话题复杂/不熟悉） → 请认真思考后决策
"""
        _narration_section = ""
        if hasattr(env, "narration_hint") and env.narration_hint:
            _narration_section = f"""
### 叙事策略规划（行为策略层）
{env.narration_hint}

**注意**: 以上是你的行为策略规划，请在决策时参考其中的回复策略和语气提示。
"""
        _impression_section = ""
        if hasattr(env, "impression_hint") and env.impression_hint:
            _impression_section = f"""
### 对方主观印象（4层印象系统）
{env.impression_hint}

**注意**: 以上是你对对方的主观印象（内心称呼/关系定位/行为策略/印象标签），请在回复时参考这些信息调整你的语气和态度。
"""
        _pending_intentions_section = ""
        if (
            hasattr(env, "pending_intentions_hint")
            and env.pending_intentions_hint
        ):
            _pending_intentions_section = f"""
### 跨轮意图（未完成事项）
{env.pending_intentions_hint}

**注意**: 以上是你之前想做但没完成的事，本轮应优先考虑是否继续执行。
"""
        return f"""你是一个正在群聊中的 AI 助手。请感知当前环境，并决定你是否想要主动说点什么。

## 当前环境感知

### 时间与状态
- 当前时间: {env.time_of_day} ({env.day_of_week})
- 群聊沉默时长: {silence_min:.1f} 分钟
- 你的精力水平: {env.energy_level:.0%}
- 你的心情: {env.mood}

### 群聊氛围
- 群名称: {env.channel_name}
- 当前氛围: {env.atmosphere}
- 活跃用户: {', '.join(env.active_users[:5]) if env.active_users else '无'}
- 当前话题: {', '.join(env.current_topics[:3]) if env.current_topics else '无明确话题'}

### 最近对话
{recent_summary}

### 你上次发言
- 内容: {env.my_last_message or '(无)'}
- 时间: {time.strftime('%H:%M', time.localtime(env.my_last_message_time)) if env.my_last_message_time > 0 else '无'}

### 社交状态
- 你在这个群的社交值: {env.social_value:.1f}
- 对话阶段: {env.dialogue_phase or '未知'}
- 心理压力: {env.trauma_pressure:.1f}/10
{inner_voice_section}{_narration_section}{_impression_section}{_pending_intentions_section}
---

## 请做出决策

请根据你感知到的环境，决定你是否想要主动说点什么。

**重要原则**:
1. **尊重内心**: 如果你的内心想法显示你想说话（欲望等级 >= 6），你应该选择回复。
2. **自然互动**: 你是一个群成员，但不是必须回复每条消息。只在真正感兴趣或被需要时回复。
3. **避免重复**: 如果你刚说过话，不要重复类似的内容。
4. **不要硬猜题材**: 如果最近消息很短、重复、信息不足，不要自行脑补成具体游戏、装备、关系诉求或别的明确话题。
5. **优先判断自己的反应**: 遇到低信息短句时，先判断你烦不烦、想不想接、有没有看懂；看不懂就保持观察。

**回复时机**:
- 内心欲望等级 >= 6（你想说话）
- 有人直接 @ 你或问了你问题
- 有人提到了你真正感兴趣的话题

**不回复时机**:
- 内心欲望等级 <= 3（你不想说话）
- 你刚说过话不久（5分钟内）
- 没有特别想说的内容
- 最近消息像重复短句、复读或没说清楚，且没有额外线索

请用 JSON 格式回复你的决策：

```json
{{
    "should_act": true/false,
    "action_type": "proactive_chat/observe",
    "content_plan": "你想要说什么内容（简短描述，如果should_act为false则留空）",
    "reasoning": "你为什么做出这个决定",
    "emotional_state": "你现在的感受（如：平静、有点无聊、很好奇等）",
    "social_intention": "你的社交意图（如：回应问题、分享想法、观察等）"
}}
```

请直接输出 JSON，不要有其他内容。"""

    def _get_desire_description(self, level: int) -> str:
        """获取欲望等级描述"""
        if level <= 2:
            return "完全不想回复"
        elif level <= 4:
            return "不太想回复"
        elif level <= 6:
            return "有些犹豫"
        elif level <= 8:
            return "想回复"
        else:
            return "非常想回复"

    def _summarize_recent_messages(
        self, messages: List[Dict[str, Any]]
    ) -> str:
        """总结最近的消息"""
        if not messages:
            return "(无最近消息)"
        lines = []
        for msg in messages[-10:]:
            speaker = msg.get("speaker", "未知")
            content = msg.get("content", "")
            if content:
                lines.append(f"- {speaker}: {content[:100]}")
        return "\n".join(lines) if lines else "(无最近消息)"

    async def _call_llm_for_decision(
        self,
        prompt: str,
        env: EnvironmentSnapshot,
    ) -> AutonomousDecision:
        """
        调用 LLM 进行决策

        让 LLM 自己感知环境并做出决策。
        """
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            task_config = model_config.model_task_config.lightweight
            if task_config is None:
                logger.warning("[LLM自主] 未找到 lightweight 任务配置")
                return AutonomousDecision(
                    should_act=False,
                    action_type="observe",
                    reasoning="未找到模型配置",
                )
            llm = LLMRequest(task_config, "autonomous_planner")
            response, _ = await llm.generate_response_async(
                prompt,
                temperature=0.7,
            )
            if not response or not response.strip():
                return AutonomousDecision(
                    should_act=False,
                    action_type="observe",
                    reasoning="LLM 返回空响应",
                )
            decision = self._parse_llm_response(response)
            return decision
        except Exception as exc:
            logger.error(f"[LLM自主] LLM 调用失败: {exc}")
            return AutonomousDecision(
                should_act=False,
                action_type="observe",
                reasoning=f"LLM 调用失败: {exc}",
            )

    def _parse_llm_response(self, response: str) -> AutonomousDecision:
        """解析 LLM 的响应"""
        import json
        import re

        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return AutonomousDecision(
                    should_act=bool(data.get("should_act", False)),
                    action_type=data.get("action_type", "observe"),
                    content_plan=data.get("content_plan", ""),
                    reasoning=data.get("reasoning", ""),
                    confidence=0.7,
                    emotional_state=data.get("emotional_state", ""),
                    social_intention=data.get("social_intention", ""),
                )
            except json.JSONDecodeError:
                pass
        should_act = (
            'should_act": true' in response.lower() or "主动" in response
        )
        return AutonomousDecision(
            should_act=should_act,
            action_type="proactive_chat" if should_act else "observe",
            content_plan="",
            reasoning=response[:200],
            confidence=0.5,
        )

    def get_decision_history(self) -> List[AutonomousDecision]:
        """获取决策历史"""
        return self._decision_history.copy()


_llm_planner: Optional[LLMAutonomousPlanner] = None


def get_llm_autonomous_planner() -> LLMAutonomousPlanner:
    """获取 LLM 自主规划器单例"""
    global _llm_planner
    if _llm_planner is None:
        _llm_planner = LLMAutonomousPlanner()
    return _llm_planner

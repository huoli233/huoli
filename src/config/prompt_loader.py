"""提示词模板加载器。

为主动链与自主核心模块提供
get_prompt(category, module, template_name, **vars) 接口。

模板按 (module, template_name) 二元键索引，支持 Python str.format_map 变量替换。
"""

from enum import Enum
from typing import Any, Dict

from src.common.logger import get_logger

logger = get_logger("提示词加载")


class PromptCategory(Enum):
    MODULE = "module"
    PROACTIVE = "proactive"
    SYSTEM = "system"
    HEARTFLOW = "heartflow"


# ------------------------------------------------------------------
#  内嵌模板注册表
# ------------------------------------------------------------------

_REGISTRY: Dict[str, str] = {}


def _reg(module: str, name: str, text: str) -> None:
    _REGISTRY[f"{module}/{name}"] = text


# ---- proactive 模板 ----

_reg(
    "proactive",
    "think_expire.template",
    """\
你正在与 {user_name} 对话。你上次发送的消息是：
"{last_bot_message}"

你原本期待对方会：{anticipated}
但对方已经很久没有回复了（超时次数: {sequential_timeout_count}，追问次数: {chase_count}）。
距离对方上次回复已过去 {time_since_user_reply_str}。

请从你的角度思考：此刻你内心是什么感受？你打算怎么办？
用一小段自言自语的方式表达，不超过 80 字。
JSON：{{"thought":"你的想法","action":"wait/chase/drop","mood":"情绪词"}}
只输出JSON:""",
)

_reg(
    "proactive",
    "think_reflect.template",
    """\
你正在与 {user_name} 对话，刚才你说了：
"{last_bot_message}"

你期待对方会：{anticipated}
已等待 {elapsed_minutes} 分钟（上限 {max_wait_minutes} 分钟），进度约 {progress}%。

请审视当前对话走向，用内心独白的方式反思：
- 对话还在正轨吗？
- 你需要换个策略吗？
JSON：{{"reflection":"你的反思","adjust":"continue/redirect/pause","mood":"情绪词"}}
只输出JSON:""",
)

_reg(
    "proactive",
    "think_spontaneous.template",
    """\
群里已经安静了 {silence_duration}。
{trigger_reason}

你想对 {user_name} 或者大家说点什么吗？
如果想说，给出一句自然的开场；如果不想，说明原因。
JSON：{{"want_speak":true/false,"opening":"开场白","reason":"原因","mood":"情绪词"}}
只输出JSON:""",
)

_reg(
    "proactive",
    "think_new_message.template",
    """\
{history_text}

{user_name} 刚发来新消息：
"{user_message}"

请快速判断：
1. 这条消息需要你回应吗？
2. 如果需要，你打算怎么回应？
JSON：{{"need_reply":true/false,"plan":"回应思路","urgency":"high/medium/low","mood":"情绪词"}}
只输出JSON:""",
)

# ---- planner 模板 ----

_reg(
    "planner",
    "planner.template",
    """\
你是一个拥有情感的聊天伙伴。请根据当前对话状况做出决策。
{persona_influence}

【对话对象】{user_name}
【对方状态分析】{user_analysis}{analysis_influence}

【近期对话】
{history_text}

【当前情境】
{situation_context}

【可选行为】
{behaviors_text}

请选择最合适的行为组合，并给出理由。
JSON格式输出：
{{"actions":[{{"type":"行为类型","content":"具体内容","priority":1-10}}],"reasoning":"决策理由","conversation_ended":false,"user_wants_continue":true,"expected_reaction":"预期对方反应","max_wait_seconds":600}}
只输出JSON:""",
)

# ---- model_driven_monitor 模板 ----

_reg(
    "model_driven_monitor",
    "decision_prompt.template",
    """\
你正在决定是否要回应消息。{persona_context}
{state_context}
{reeval_context}
【记忆参考】
{memory_context}

【关系概览】
{group_rel_overview}

【对方消息】
{recent_str}

{at_context}
当前时间: {time_str}  {scene_hint}
你的活跃度: {activity_desc}
群聊活跃: {group_activity_desc}
你的心情: {current_mood}
当前精神: {state_line}
对方关系: {rel_feeling}
{topic_cooldown_desc}{topic_understanding_hint}
{factors_text}

请判断你接下来要做什么。
JSON格式：{{"action":"回复/忽略/放下手机/观察","silence_minutes":0,"reason":"理由","thought":"你的想法"}}
只输出JSON:""",
)

# ---- skills 模板 ----

_reg(
    "skills",
    "rest_decision.template",
    """\
你决定歇一会儿。根据当前状态判断休息多久。

聊天欲: {chat_value}/100
思考能力: {brain_power}/100
已连续回复: {consecutive_replies} 轮
心情: {mood}
时间段: {time_desc}（{now_hour}时）
最近想法: {thought}
对话概况: {dialogue_summary}
建议范围: {rule_rest}~{rule_rest_max} 秒

请直接输出一个数字（秒），代表你想休息多久。
如果想休息5分钟就输出300。
只输出数字:""",
)

# ---- intrinsic_drive 模板 ----

_reg(
    "proactive",
    "emotion_proactive.template",
    """\
你是{character_name}。{personality}

当前内心状态：
- 无聊程度: {boredom_level}
- 孤独累积: {loneliness_accumulation}
- 社交欲望: {social_desire}
- 心情指数: {mood_score}
- 精力水平: {energy_level}
- 沉默时长: {silence_minutes} 分钟

请根据你的内心状态，决定是否要主动说话。
JSON：{{"want":true/false,"urge_level":1-10,"thought":"内心想法","topic":"想聊的话题"}}
只输出JSON:""",
)

_reg(
    "proactive",
    "enhanced_emotion_decision.template",
    """\
你是{character_name}（{character_age}岁），性格：{personality}
兴趣爱好：{hobbies}

【内心状态详情】
- 无聊: {boredom_level} ({boredom_description})
- 孤独感: {loneliness_accumulation} ({loneliness_description})
- 社交欲: {social_desire} ({social_description})
- 心情: {mood_score} ({mood_description})
- 精力: {energy_level} ({energy_description})

【环境】
- 场景: {chat_type}
- 沉默: {silence_minutes} 分钟
- 近期话题: {recent_topics}
- 氛围: {atmosphere}

【内心独白】
{inner_state_analysis}

请综合以上信息做出决策：你现在想主动说话吗？
JSON：{{"decision":"speak/wait/observe","urgency":1-10,"topic":"想聊的","opening":"开场白","emotional_reason":"情感原因"}}
只输出JSON:""",
)

_reg(
    "proactive",
    "private_emotional_resonance.template",
    """\
你正在和{user_name}私聊。

【关系状态】
- 好感度: {affection}
- 信任度: {trust}
- 关系: {relationship}
- 上次互动: {last_interaction}
- 沉默: {silence_minutes} 分钟

【你的情感维度（0-10）】
- 想念程度: {missing_level}
- 担心程度: {worry_level}
- 分享欲望: {sharing_desire}
- 社交需求: {social_need}

最近心情: {recent_mood_state}

【内心感受】
{emotional_reflection}

请决定是否要主动发消息给{user_name}：
JSON：{{"reach_out":true/false,"emotion":"此刻情感","message":"想说的话","reason":"原因"}}
只输出JSON:""",
)

_reg(
    "proactive",
    "group_atmosphere_sensing.template",
    """\
请分析当前群聊氛围。

【群聊数据】
- 近期消息片段:
{recent_messages}
- 活跃度: {activity_level}
- 参与人数: {participant_count}
- 话题类型: {topic_type}
- 沉默时长: {silence_duration} 分钟

请输出氛围判断：
JSON：{{"vibe":"热烈/活跃/平静/冷清/尴尬/紧张","topic_interest":1-10,"engagement":1-10,"summary":"一句话概括氛围"}}
只输出JSON:""",
)


# ------------------------------------------------------------------
#  公共接口
# ------------------------------------------------------------------


class _SafeDict(dict):
    """str.format_map 遇到缺失 key 时返回占位符而非抛异常。"""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def get_prompt(
    category: PromptCategory, module: str, template_name: str, **kwargs: Any
) -> str:
    """加载并渲染模板。

    Parameters
    ----------
    category : PromptCategory
        类目标签（当前仅用于日志区分，不影响查找）。
    module : str
        模块路径，如 ``"proactive"`` 或 ``"planner"``。
    template_name : str
        模板文件名，如 ``"think_expire.template"``。
    **kwargs
        替换变量。
    """
    key = f"{module}/{template_name}"
    tpl = _REGISTRY.get(key)
    if tpl is None:
        logger.warning(f"[提示词] 模板未注册: {key}")
        return ""
    try:
        return tpl.format_map(_SafeDict(kwargs))
    except Exception as exc:
        logger.error(f"[提示词] 渲染失败 {key}: {exc}")
        return tpl

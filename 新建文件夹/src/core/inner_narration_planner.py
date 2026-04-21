import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("narration_planner")

_planner_instances: Dict[str, "InnerNarrationPlanner"] = {}


@dataclass
class NarrationPlan:
    """旁白与行为策略规划"""

    inner_thought: str = ""
    action_intent: str = ""
    reply_strategy: str = ""
    tone_hint: str = ""
    should_reply: bool = False
    should_observe: bool = False
    should_defer: bool = False
    confidence: float = 0.5
    context_summary: str = ""
    planned_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inner_thought": self.inner_thought,
            "action_intent": self.action_intent,
            "reply_strategy": self.reply_strategy,
            "tone_hint": self.tone_hint,
            "should_reply": self.should_reply,
            "should_observe": self.should_observe,
            "should_defer": self.should_defer,
            "confidence": round(self.confidence, 3),
            "context_summary": self.context_summary[:200],
        }

    def to_prompt_block(self) -> str:
        """转换为可注入 prompt 的文本块"""
        lines = []
        if self.inner_thought:
            lines.append(f"[内心独白] {self.inner_thought}")
        if self.action_intent:
            lines.append(f"[行为意图] {self.action_intent}")
        if self.reply_strategy:
            lines.append(f"[回复策略] {self.reply_strategy}")
        if self.tone_hint:
            lines.append(f"[语气提示] {self.tone_hint}")
        return "\n".join(lines) if lines else ""


class InnerNarrationPlanner:
    """旁白前置规划层
    在最终回复生成之前，先产出一段旁白与行为策略，
    让回复具备连续性和主观一致性。
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._recent_plans: List[NarrationPlan] = []
        self._max_history = 15

    @property
    def latest_plan(self) -> Optional[NarrationPlan]:
        return self._recent_plans[-1] if self._recent_plans else None

    def plan(
        self,
        *,
        understanding_level: str = "can_understand_no_reply",
        visibility_level: str = "noticed",
        watch_level: str = "skim_window",
        presence_mode: str = "一般",
        behavior_category: str = "neutral",
        energy_ratio: float = 1.0,
        social_willingness: float = 0.5,
        user_relationship_level: int = 2,
        user_trust: float = 0.0,
        is_at_bot: bool = False,
        is_question: bool = False,
        recent_bot_texts: Optional[List[str]] = None,
        message_summary: str = "",
        dominant_mood: str = "",
        topic_interest_level: float = 0.0,
        want_learn_first: bool = False,
        self_ref_type: str = "",
        self_ref_strength: float = 0.0,
        metabolism_reply_hint: str = "normal",
        metabolism_is_perfunctory: bool = False,
        metabolism_gate_open: bool = True,
        scene_suitable_to_join: bool = True,
        dominant_pattern: str = "",
        msg_has_voice_or_music: bool = False,
        trauma_score: float = 0.0,
        channel_annoyance: float = 0.0,
        misunderstanding_risk: float = 0.0,
        user_negative_emotion: float = 0.0,
        user_impression_tags: Optional[List[str]] = None,
    ) -> NarrationPlan:
        """根据上下文生成旁白规划"""
        plan = NarrationPlan(planned_at=time.time())
        plan.context_summary = message_summary[:200]
        # 判断是否回复
        reply_score = self._compute_reply_score(
            understanding_level=understanding_level,
            visibility_level=visibility_level,
            watch_level=watch_level,
            energy_ratio=energy_ratio,
            social_willingness=social_willingness,
            user_relationship_level=user_relationship_level,
            is_at_bot=is_at_bot,
            is_question=is_question,
        )
        plan.confidence = reply_score
        # 情绪调制：烦躁/疲倦降低得分，好奇/想参与提升得分
        mood_adjust = {
            "烦躁": -0.12,
            "疲倦": -0.08,
            "无聊": -0.03,
            "好奇": 0.08,
            "想参与": 0.1,
            "想围观": 0.03,
        }.get(dominant_mood, 0.0)
        reply_score = max(0.0, min(1.0, reply_score + mood_adjust))
        # 话题兴趣调制
        if topic_interest_level > 0.6:
            reply_score = min(1.0, reply_score + 0.06)
        elif topic_interest_level < 0.2 and not is_at_bot:
            reply_score = max(0.0, reply_score - 0.05)
        # 自引用强度调制（被引用/被讨论时提升回复意愿）
        if self_ref_strength > 0.5 and self_ref_type != "none":
            reply_score = min(1.0, reply_score + self_ref_strength * 0.15)
        # 学习优先：想先学习时压制回复冲动，转为观察
        if want_learn_first and not is_at_bot:
            reply_score = min(reply_score, 0.35)
        # 代谢约束：能量门关闭时压低得分，敷衍模式时限制上限
        if not metabolism_gate_open and not is_at_bot:
            reply_score = max(0.0, reply_score - 0.2)
        if metabolism_is_perfunctory:
            reply_score = min(reply_score, 0.55)
        # 场景适合度：不适合插话时压低得分
        if not scene_suitable_to_join and not is_at_bot:
            reply_score = max(0.0, reply_score - 0.15)
        # 群体模式调制
        _pattern_adjust = {
            "newcomer_welcome": 0.12,
            "ritual_greeting": 0.08,
            "copycat_chain": -0.1,
            "pile_on": -0.15,
            "conflict_escalation": -0.12,
            "spectating": -0.05,
        }.get(dominant_pattern, 0.0)
        if _pattern_adjust:
            reply_score = max(0.0, min(1.0, reply_score + _pattern_adjust))
        # 语音/音乐消息：转为文字无关特殊策略
        if msg_has_voice_or_music and not is_at_bot:
            reply_score = min(reply_score, 0.4)
        # 创伤/烦躁联合调制：长期创伤叠加短期烦躁压低意愿
        if trauma_score > 0.3 and channel_annoyance > 0.5:
            reply_score = max(0.0, reply_score - 0.18)
        elif trauma_score > 0.5:
            reply_score = max(0.0, reply_score - 0.1)
        # 误解风险：上轮回复可能被误解时，提升回复意愿以主动澄清
        if misunderstanding_risk > 0.4:
            reply_score = min(1.0, reply_score + 0.2)
        elif misunderstanding_risk > 0.2:
            reply_score = min(1.0, reply_score + 0.1)
        # 用户负面情绪高时降低回复攻击性、提升温和倾向
        if user_negative_emotion > 60:
            reply_score = min(1.0, reply_score + 0.05)
        # 印象标签驱动的回复调整
        _tags = user_impression_tags or []
        _conflict_tags = {"容易起冲突", "冲突型", "爱抬杠", "攻击性强"}
        _friendly_tags = {"积极友善", "热情", "话题丰富", "有趣"}
        if _tags and _conflict_tags & set(_tags):
            reply_score = max(0.0, reply_score - 0.1)
        elif _tags and _friendly_tags & set(_tags):
            reply_score = min(1.0, reply_score + 0.05)
        plan.confidence = reply_score
        if reply_score > 0.6:
            plan.should_reply = True
        elif reply_score > 0.3:
            plan.should_observe = True
        else:
            plan.should_defer = True
        # 生成内心独白
        plan.inner_thought = self._generate_inner_thought(
            presence_mode=presence_mode,
            should_reply=plan.should_reply,
            reply_score=reply_score,
            is_at_bot=is_at_bot,
            is_question=is_question,
            energy_ratio=energy_ratio,
            dominant_mood=dominant_mood,
            want_learn_first=want_learn_first,
            dominant_pattern=dominant_pattern,
            msg_has_voice_or_music=msg_has_voice_or_music,
            trauma_score=trauma_score,
            misunderstanding_risk=misunderstanding_risk,
            user_negative_emotion=user_negative_emotion,
            user_impression_tags=user_impression_tags,
        )
        # 行为意图
        if plan.should_reply:
            plan.action_intent = self._determine_action_intent(
                behavior_category=behavior_category,
                user_relationship_level=user_relationship_level,
                is_question=is_question,
            )
        elif plan.should_observe:
            plan.action_intent = "继续观察，暂不插话"
        else:
            plan.action_intent = "先放一放"
        # 回复策略
        if plan.should_reply:
            plan.reply_strategy = self._determine_reply_strategy(
                energy_ratio=energy_ratio,
                user_relationship_level=user_relationship_level,
                social_willingness=social_willingness,
                recent_bot_texts=recent_bot_texts,
            )
        # 学习优先时策略切换
        if plan.should_reply and want_learn_first and not is_at_bot:
            plan.reply_strategy = "先搞清楚再说，简短回应"
        # 代谢回复长度约束覆盖
        if plan.should_reply and metabolism_reply_hint in (
            "brief",
            "perfunctory",
        ):
            if metabolism_reply_hint == "perfunctory":
                plan.reply_strategy = "敷衍了事，能短则短"
            elif metabolism_reply_hint == "brief":
                plan.reply_strategy = "简要回应，不展开"
        if (
            plan.should_reply
            and metabolism_reply_hint == "silent"
            and not is_at_bot
        ):
            plan.should_reply = False
            plan.should_defer = True
            plan.action_intent = "体力不支，不说了"
        # 仪式模式策略覆盖
        if plan.should_reply and dominant_pattern == "newcomer_welcome":
            plan.reply_strategy = "简短欢迎，不要抢话"
            plan.tone_hint = "友好"
        if plan.should_reply and dominant_pattern == "ritual_greeting":
            plan.reply_strategy = "跟随群体礼仪，简短附和"
        # 被误解时主动澄清策略覆盖
        if plan.should_reply and misunderstanding_risk > 0.4:
            plan.reply_strategy = "主动澄清，纠正对方的理解偏差"
        elif plan.should_reply and misunderstanding_risk > 0.2:
            plan.reply_strategy = "温和补充说明，避免加深误会"
        # 用户负面情绪高时切换安抚策略
        if plan.should_reply and user_negative_emotion > 60:
            plan.reply_strategy = "温柔安慰，回避刺激性话题"
        elif plan.should_reply and user_negative_emotion > 40:
            plan.reply_strategy = "语气放软，适当关心"
        # 印象标签驱动的回复策略调整
        if plan.should_reply and _tags and _conflict_tags & set(_tags):
            plan.reply_strategy = "谨慎措辞，避免激化冲突"
        elif plan.should_reply and _tags and _friendly_tags & set(_tags):
            plan.reply_strategy = "轻松互动，积极回应"
        # 语气提示
        plan.tone_hint = self._determine_tone(
            presence_mode=presence_mode,
            energy_ratio=energy_ratio,
            user_relationship_level=user_relationship_level,
            dominant_mood=dominant_mood,
            trauma_score=trauma_score,
            misunderstanding_risk=misunderstanding_risk,
            user_negative_emotion=user_negative_emotion,
            user_impression_tags=user_impression_tags,
        )
        self._recent_plans.append(plan)
        if len(self._recent_plans) > self._max_history:
            self._recent_plans = self._recent_plans[-self._max_history:]
        return plan

    def continuity_context(self) -> str:
        """从最近旁白中提取连续性上下文"""
        if not self._recent_plans:
            return ""
        recent = self._recent_plans[-3:]
        lines = []
        for idx, p in enumerate(recent, 1):
            if p.inner_thought:
                lines.append(f"第{idx}轮心声：{p.inner_thought[:80]}")
        return "\n".join(lines) if lines else ""

    # ────────────────── 内部方法 ──────────────────

    def _compute_reply_score(
        self,
        *,
        understanding_level: str,
        visibility_level: str,
        watch_level: str,
        energy_ratio: float,
        social_willingness: float,
        user_relationship_level: int,
        is_at_bot: bool,
        is_question: bool,
    ) -> float:
        score = 0.3
        # 被@直接加大分
        if is_at_bot:
            score += 0.4
        if is_question:
            score += 0.2
        # 理解层级加分
        understanding_bonus = {
            "can_reply_deep": 0.25,
            "can_reply_simple": 0.15,
            "need_clarification": 0.1,
            "can_understand_no_reply": 0.0,
            "cannot_understand": -0.15,
        }
        score += understanding_bonus.get(understanding_level, 0.0)
        # 可见性层级
        visibility_bonus = {
            "focused": 0.15,
            "noticed": 0.05,
            "glanced": -0.05,
            "received": -0.15,
        }
        score += visibility_bonus.get(visibility_level, 0.0)
        # 观看层级
        watch_bonus = {
            "engaged": 0.15,
            "active_watch": 0.08,
            "skim_window": 0.0,
            "peek": -0.1,
            "blackout": -0.3,
        }
        score += watch_bonus.get(watch_level, 0.0)
        # 精力和社交意愿
        score += (energy_ratio - 0.5) * 0.15
        score += (social_willingness - 0.5) * 0.1
        # 关系加分
        if user_relationship_level >= 4:
            score += 0.1
        elif user_relationship_level >= 3:
            score += 0.05
        return max(0.0, min(1.0, score))

    def _generate_inner_thought(
        self,
        *,
        presence_mode: str,
        should_reply: bool,
        reply_score: float,
        is_at_bot: bool,
        is_question: bool,
        energy_ratio: float,
        dominant_mood: str = "",
        want_learn_first: bool = False,
        dominant_pattern: str = "",
        msg_has_voice_or_music: bool = False,
        trauma_score: float = 0.0,
        misunderstanding_risk: float = 0.0,
        user_negative_emotion: float = 0.0,
        user_impression_tags: Optional[List[str]] = None,
    ) -> str:
        # 印象标签驱动的内心独白（冲突预警优先）
        _itags = user_impression_tags or []
        _conflict_set = {"容易起冲突", "冲突型", "爱抬杠", "攻击性强"}
        if _itags and _conflict_set & set(_itags):
            return "这人容易起冲突，说话注意别踩雷"
        # 误解检测最先输出：发现上轮可能被误解时优先澄清
        if misunderstanding_risk > 0.4:
            return "好像说错了什么，得赶紧解释一下"
        if misunderstanding_risk > 0.2:
            return "他们好像没get到我的意思"
        # 用户负面情绪高时优先安抚
        if user_negative_emotion > 60:
            return "对方情绪不太好，说话小心点"
        if user_negative_emotion > 40:
            return "感觉对面有点低落，温柔一些吧"
        # 学习优先判定最先输出
        if want_learn_first and not is_at_bot:
            return "这个梗不太懂，先看看再说"
        # 群体仪式模式
        if dominant_pattern == "newcomer_welcome":
            return "有新人来了，可以打个招呼"
        if dominant_pattern == "copycat_chain":
            return "他们在复读...忍住别跟风"
        if dominant_pattern == "pile_on":
            return "气氛有点不对，先别掺和"
        if dominant_pattern == "conflict_escalation":
            return "好像在吵架，离远点"
        # 语音/音乐消息判定
        if msg_has_voice_or_music and not is_at_bot:
            return "发了语音/音乐，听不太清，先不管"
        # 创伤积累
        if trauma_score > 0.5:
            return "跟这人相处总出问题，保持距离"
        if is_at_bot:
            if energy_ratio < 0.3:
                return "被叫到了...但我好累，简单回一下吧"
            return "有人叫我，看看说了什么"
        if is_question and should_reply:
            return "被问到了，想想怎么回"
        # 情绪驱动的内心独白
        if dominant_mood == "烦躁":
            return "有点烦，不想搭理"
        if dominant_mood == "疲倦":
            return "精力不够了，歇一下"
        if dominant_mood == "好奇":
            if should_reply:
                return "挺好奇的，想聊一聊"
            return "好奇但不确定怎么接"
        if dominant_mood == "想围观":
            return "看看他们在说什么"
        if dominant_mood == "想参与":
            if should_reply:
                return "有兴趣聊一聊"
            return "想聊天但不太好接话"
        if dominant_mood == "无聊":
            if should_reply:
                return "有点无聊，找个话茬聊聊"
            return "没什么意思，随便看看"
        # 原有兜底
        if presence_mode == "想安静":
            return "不太想说话，安静看看"
        if presence_mode == "想社交":
            if should_reply:
                return "有兴趣聊一聊"
            return "想聊天但不太好接话"
        if presence_mode == "纯围观":
            return "看看他们在说什么"
        if energy_ratio < 0.3:
            return "精力不够了，歇一下"
        if should_reply:
            return "可以接一下"
        return "随便看看"

    def _determine_action_intent(
        self,
        *,
        behavior_category: str,
        user_relationship_level: int,
        is_question: bool,
    ) -> str:
        if is_question:
            return "回答问题"
        intent_map = {
            "greeting": "打招呼",
            "chat": "日常聊天",
            "emotional": "情感回应",
            "help": "提供帮助",
            "humor": "开玩笑",
            "neutral": "随意回复",
        }
        return intent_map.get(behavior_category, "适当回应")

    def _determine_reply_strategy(
        self,
        *,
        energy_ratio: float,
        user_relationship_level: int,
        social_willingness: float,
        recent_bot_texts: Optional[List[str]] = None,
    ) -> str:
        if energy_ratio < 0.3:
            return "简短回复，省精力"
        if user_relationship_level >= 4 and social_willingness > 0.6:
            return "可以多聊一点"
        if recent_bot_texts and len(recent_bot_texts) >= 2:
            return "顺着之前说的继续，不要重复"
        return "正常回复"

    def _determine_tone(
        self,
        *,
        presence_mode: str,
        energy_ratio: float,
        user_relationship_level: int,
        dominant_mood: str = "",
        trauma_score: float = 0.0,
        misunderstanding_risk: float = 0.0,
        user_negative_emotion: float = 0.0,
        user_impression_tags: Optional[List[str]] = None,
    ) -> str:
        # 印象标签驱动的语气调整（冲突用户谨慎、友善用户热情）
        _t = user_impression_tags or []
        _conflict_check = {"容易起冲突", "冲突型", "爱抬杠", "攻击性强"}
        _friendly_check = {"积极友善", "热情", "话题丰富", "有趣"}
        if _t and _conflict_check & set(_t):
            return "克制"
        if _t and _friendly_check & set(_t):
            return "热情"
        # 被误解时语气优先切换
        if misunderstanding_risk > 0.4:
            return "歉意"
        if misunderstanding_risk > 0.2:
            return "耐心"
        # 用户负面情绪高时切换语气
        if user_negative_emotion > 60:
            return "温柔"
        if user_negative_emotion > 40:
            return "关怀"
        # 创伤积累优先
        if trauma_score > 0.5:
            return "疏离"
        if trauma_score > 0.3:
            return "戒备"
        # 情绪标签优先
        if dominant_mood == "烦躁":
            return "不耐烦"
        if dominant_mood == "疲倦":
            return "慵懒"
        if dominant_mood == "好奇":
            return "好奇"
        if dominant_mood == "无聊":
            return "随意"
        if presence_mode == "想安静":
            return "平淡"
        if energy_ratio < 0.3:
            return "慵懒"
        if user_relationship_level >= 4:
            return "亲近"
        if presence_mode == "想社交" or dominant_mood == "想参与":
            return "友好"
        return "自然"


def get_narration_planner(channel_id: str) -> InnerNarrationPlanner:
    if channel_id not in _planner_instances:
        _planner_instances[channel_id] = InnerNarrationPlanner(channel_id)
    return _planner_instances[channel_id]


def remove_narration_planner(channel_id: str) -> None:
    _planner_instances.pop(channel_id, None)

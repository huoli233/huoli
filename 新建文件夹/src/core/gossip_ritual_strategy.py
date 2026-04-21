import time
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from src.common.logger import get_logger

logger = get_logger("gossip_ritual")

_strategy_instances: Dict[str, "GossipRitualStrategyEngine"] = {}

# ═══════════════════════════════════════════════════
#  事件类型与行为枚举
# ═══════════════════════════════════════════════════


class EventType(Enum):
    """事件类型"""

    NONE = "none"
    GOSSIP_MELON = "gossip_melon"
    NEWCOMER_WELCOME = "newcomer_welcome"
    BIRTHDAY_CELEBRATION = "birthday_celebration"
    HOLIDAY_GREETING = "holiday_greeting"
    CONGRATULATIONS = "congratulations"
    MEME_CHAIN = "meme_chain"
    GROUP_CONFLICT = "group_conflict"
    SUPPORT_CIRCLE = "support_circle"
    COLLECTIVE_STORYTELLING = "collective_storytelling"
    DEBATE_CIRCLE = "debate_circle"
    RITUAL_REPLY_CHAIN = "ritual_reply_chain"
    BOT_DISCUSSION = "bot_discussion"

    def label(self) -> str:
        return {
            "none": "无事件",
            "gossip_melon": "吃瓜事件",
            "newcomer_welcome": "迎新",
            "birthday_celebration": "生日祝福",
            "holiday_greeting": "节日问候",
            "congratulations": "恭喜",
            "meme_chain": "梗接龙",
            "group_conflict": "群内冲突",
            "support_circle": "支持圈",
            "collective_storytelling": "集体叙事",
            "debate_circle": "辩论圈",
            "ritual_reply_chain": "礼仪回复链",
            "bot_discussion": "讨论机器人",
        }.get(self.value, "")


class ParticipationPosture(Enum):
    """参与姿态"""

    SILENT_OBSERVER = "silent_observer"
    CASUAL_SPECTATOR = "casual_spectator"
    LIGHT_CHIME_IN = "light_chime_in"
    NORMAL_JOIN = "normal_join"
    ACTIVE_ENGAGE = "active_engage"
    COMFORT_SUPPORT = "comfort_support"
    DEFEND_COUNTER = "defend_counter"
    AVOID_COMPLETELY = "avoid_completely"

    def label(self) -> str:
        return {
            "silent_observer": "安静围观",
            "casual_spectator": "随意看看",
            "light_chime_in": "轻插一句",
            "normal_join": "正常参与",
            "active_engage": "积极卷入",
            "comfort_support": "安慰支持",
            "defend_counter": "辩护反驳",
            "avoid_completely": "完全回避",
        }.get(self.value, "")


@dataclass
class EventContext:
    """事件上下文输入"""

    event_type: EventType = EventType.NONE
    confidence: float = 0.0
    involved_users: Set[str] = field(default_factory=set)
    third_party_target: str = ""
    is_bot_relevant: bool = False
    bot_mentioned_directly: bool = False
    event_intensity: float = 0.5
    message_count_in_event: int = 0
    time_span_sec: float = 0.0
    emotional_valence: float = 0.0
    has_media_storm: bool = False
    group_atmosphere_friendly: bool = True
    newcomer_id: Optional[str] = None
    ritual_keyword: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "event_label": self.event_type.label(),
            "confidence": round(self.confidence, 3),
            "involved_count": len(self.involved_users),
            "bot_relevant": self.is_bot_relevant,
            "intensity": round(self.event_intensity, 3),
            "msg_count": self.message_count_in_event,
        }


@dataclass
class SubjectiveReadiness:
    """主体就绪状态"""

    energy_ratio: float = 1.0
    social_willingness: float = 0.5
    boredom_level: float = 0.0
    loafing_level: float = 0.0
    watch_state_rank: int = 2
    current_mood: str = ""
    understanding_level: str = "can_reply_simple"
    relation_to_involved_avg: float = 0.5
    trauma_score: float = 0.0
    annoyance_to_group: float = 0.0
    recent_reply_count: int = 0
    interest_in_topic: float = 0.5
    want_learn_first: bool = False


@dataclass
class StrategyVerdict:
    """策略裁定结果"""

    posture: ParticipationPosture = ParticipationPosture.SILENT_OBSERVER
    should_act: bool = False
    priority: float = 0.0
    reply_strategy_hint: str = ""
    tone_hint: str = ""
    inner_narration_hint: str = ""
    defer_reason: str = ""
    confidence: float = 0.0
    decided_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "posture": self.posture.value,
            "posture_label": self.posture.label(),
            "should_act": self.should_act,
            "priority": round(self.priority, 3),
            "reply_hint": self.reply_strategy_hint[:60],
            "tone_hint": self.tone_hint,
            "narration": self.inner_narration_hint[:80],
            "defer_reason": (
                self.defer_reason[:60] if self.defer_reason else ""
            ),
            "confidence": round(self.confidence, 3),
        }


# ═══════════════════════════════════════════════════
#  事件特征关键词库
# ═══════════════════════════════════════════════════

_GOSSIP_REACTION_KW = {
    "卧槽",
    "真的假的",
    "不是吧",
    "天哪",
    "离谱",
    "啊这",
    "我靠",
    "震惊",
    "什么情况",
    "发生了什么",
    "太可怕了",
    "牛逼",
    "好家伙",
    "绝了",
    "我的天",
    "笑死",
    "细说",
}
_RITUAL_GREETING_KW = {
    "早安",
    "晚安",
    "早上好",
    "晚上好",
    "早",
    "晚",
    "新年好",
    "新年快乐",
    "中秋快乐",
    "国庆快乐",
    "生日快乐",
    "恭喜",
    "祝贺",
    "生日快乐",
    "happy birthday",
}
_CONFLICT_KW = {
    "滚",
    "傻",
    "蠢",
    "垃圾",
    "烦死",
    "闭嘴",
    "吵死了",
    "你有没有病",
    "神经病",
    "脑残",
    "智障",
}
_SUPPORT_KW = {"加油", "没事", "安慰", "支持", "抱抱", "坚强", "理解", "心疼"}
_MEME_KW = {"哈哈哈", "笑死", "绷不住", "乐了", "草", "xswl", "hhh", "2333"}


class GossipRitualStrategyEngine:
    """吃瓜/仪式行为策略引擎
    在群体模式检测器识别出模式后，进一步判断：
    1. 这是不是一个值得参与的事件型话题？
    2. 主体当前状态适合什么程度的参与？
    3. 具体应该采取哪种行为姿态？
    输出从"安静围观"到"辩护反驳"的8种参与姿态。
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._recent_verdicts: List[StrategyVerdict] = []
        self._max_history = 20
        self._event_participation_log: Dict[str, int] = {}
        self._max_event_log = 50

    def evaluate(
        self,
        event_ctx: EventContext,
        readiness: SubjectiveReadiness,
    ) -> StrategyVerdict:
        """评估并输出策略裁定"""
        verdict = StrategyVerdict()
        if (
            event_ctx.event_type == EventType.NONE
            or event_ctx.confidence < 0.30
        ):
            verdict.posture = ParticipationPosture.SILENT_OBSERVER
            verdict.should_act = False
            verdict.defer_reason = "未检测到有效事件或置信度过低"
            verdict.confidence = 0.9
            self._log_verdict(verdict)
            return verdict
        participation_score = self._compute_participation_score(
            event_ctx, readiness
        )
        posture = self._select_posture(
            participation_score, event_ctx, readiness
        )
        verdict.posture = posture
        verdict.confidence = min(
            1.0,
            event_ctx.confidence * 0.7 + abs(participation_score - 0.5) * 0.4,
        )
        if posture in (
            ParticipationPosture.LIGHT_CHIME_IN,
            ParticipationPosture.NORMAL_JOIN,
            ParticipationPosture.ACTIVE_ENGAGE,
            ParticipationPosture.COMFORT_SUPPORT,
            ParticipationPosture.DEFEND_COUNTER,
        ):
            verdict.should_act = True
            verdict.priority = participation_score
            verdict.reply_strategy_hint = self._derive_reply_hint(
                posture, event_ctx, readiness
            )
            verdict.tone_hint = self._derive_tone_hint(
                posture, event_ctx, readiness
            )
            verdict.inner_narration_hint = self._derive_narration_hint(
                posture, event_ctx, readiness
            )
        else:
            verdict.should_act = False
            verdict.defer_reason = self._derive_defer_reason(
                posture, event_ctx, readiness
            )
            verdict.inner_narration_hint = self._derive_narration_hint(
                posture, event_ctx, readiness
            )
        self._log_verdict(verdict)
        self._record_participation(
            event_ctx.event_type.value, verdict.should_act
        )
        return verdict

    def evaluate_batch(
        self,
        events: List[EventContext],
        readiness: SubjectiveReadiness,
    ) -> List[StrategyVerdict]:
        return [self.evaluate(ev, readiness) for ev in events]

    def recent_stats(self) -> Dict[str, Any]:
        ev_counts: Dict[str, int] = {}
        act_counts: Dict[str, int] = {}
        for v in self._recent_verdicts[-15:]:
            key = v.posture.value
            act_counts[key] = act_counts.get(key, 0) + 1
        for evt, cnt in list(self._event_participation_log.items())[-10:]:
            ev_counts[evt] = cnt
        return {
            "posture_distribution": act_counts,
            "event_participation": ev_counts,
        }

    # ────────────────── 核心评分逻辑 ──────────────────

    def _compute_participation_score(
        self, ctx: EventContext, rd: SubjectiveReadiness
    ) -> float:
        score = 0.35
        if ctx.is_bot_relevant:
            score += 0.20
        if ctx.bot_mentioned_directly:
            score += 0.25
        score += ctx.event_intensity * 0.12
        score += ctx.confidence * 0.08
        if ctx.emotional_valence > 0.3 and ctx.group_atmosphere_friendly:
            score += 0.06
        elif ctx.emotional_valence < -0.3:
            score += 0.04
        score += (rd.energy_ratio - 0.5) * 0.12
        score += (rd.social_willingness - 0.5) * 0.10
        if rd.boredom_level > 0.5:
            score += rd.boredom_level * 0.08
        if rd.loafing_level > 0.55:
            score -= rd.loafing_level * 0.18
        if rd.watch_state_rank >= 4:
            score += 0.10
        elif rd.watch_state_rank <= 1:
            score -= 0.20
        score += (rd.relation_to_involved_avg - 0.5) * 0.10
        if rd.trauma_score > 0.4:
            score -= 0.08
        if rd.annoyance_to_group > 0.5:
            score -= 0.12
        if rd.recent_reply_count >= 6:
            score -= 0.08
        elif rd.recent_reply_count >= 3:
            score -= 0.03
        score += (rd.interest_in_topic - 0.5) * 0.10
        if rd.want_learn_first:
            score = min(score, 0.38)
        _evt_bonus = {
            EventType.NEWCOMER_WELCOME: 0.10,
            EventType.BIRTHDAY_CELEBRATION: 0.08,
            EventType.HOLIDAY_GREETING: 0.06,
            EventType.CONGRATULATIONS: 0.07,
            EventType.GOSSIP_MELON: 0.05,
            EventType.SUPPORT_CIRCLE: 0.09,
            EventType.BOT_DISCUSSION: 0.15,
            EventType.GROUP_CONFLICT: -0.05,
        }
        score += _evt_bonus.get(ctx.event_type, 0.0)
        return max(0.02, min(0.98, score))

    def _select_posture(
        self,
        score: float,
        ctx: EventContext,
        rd: SubjectiveReadiness,
    ) -> ParticipationPosture:
        et = ctx.event_type
        if et == EventType.GROUP_CONFlict:
            if ctx.is_bot_relevant and score > 0.45:
                return ParticipationPosture.DEFEND_COUNTER
            if rd.annoyance_to_group < 0.3 and score > 0.40:
                return ParticipationPosture.COMFORT_SUPPORT
            return ParticipationPosture.AVOID_COMPLETELY
        if et == EventType.SUPPORT_CIRCLE:
            if score > 0.35 and rd.social_willingness > 0.3:
                return ParticipationPosture.COMFORT_SUPPORT
            if score > 0.25:
                return ParticipationPosture.LIGHT_CHIME_IN
            return ParticipationPosture.SILENT_OBSERVER
        if et == EventType.NEWCOMER_WELCOME:
            if score > 0.42:
                return ParticipationPosture.LIGHT_CHIME_IN
            return ParticipationPosture.SILENT_OBSERVER
        if et in (
            EventType.BIRTHDAY_CELEBRATION,
            EventType.HOLIDAY_GREETING,
            EventType.CONGRATULATIONS,
        ):
            if score > 0.40:
                return ParticipationPosture.LIGHT_CHIME_IN
            if score > 0.28:
                return ParticipationPosture.CASUAL_SPECTATOR
            return ParticipationPosture.SILENT_OBSERVER
        if et == EventType.GOSSIP_MELON:
            if score > 0.55:
                return ParticipationPosture.LIGHT_CHIME_IN
            if score > 0.35:
                return ParticipationPosture.CASUAL_SPECTATOR
            return ParticipationPosture.SILENT_OBSERVER
        if et == EventType.BOT_DISCUSSION:
            if ctx.bot_mentioned_directly and score > 0.40:
                return ParticipationPosture.ACTIVE_ENGAGE
            if score > 0.48:
                return ParticipationPosture.NORMAL_JOIN
            if score > 0.30:
                return ParticipationPosture.CASUAL_SPECTATOR
            return ParticipationPosture.SILENT_OBSERVER
        if et in (EventType.MEME_CHAIN, EventType.RITUAL_REPLY_CHAIN):
            if score > 0.52:
                return ParticipationPosture.LIGHT_CHIME_IN
            return ParticipationPosture.SILENT_OBSERVER
        if et in (EventType.COLLECTIVE_STORYTELLING, EventType.DEBATE_CIRCLE):
            if score > 0.58 and not rd.want_learn_first:
                return ParticipationPosture.NORMAL_JOIN
            if score > 0.38:
                return ParticipationPosture.CASUAL_SPECTATOR
            return ParticipationPosture.SILENT_OBSERVER
        if score > 0.62:
            return ParticipationPosture.ACTIVE_ENGAGE
        if score > 0.48:
            return ParticipationPosture.NORMAL_JOIN
        if score > 0.32:
            return ParticipationPosture.LIGHT_CHIME_IN
        if score > 0.18:
            return ParticipationPosture.CASUAL_SPECTATOR
        return ParticipationPosture.SILENT_OBSERVER

    # ────────────────── 提示词派生方法 ──────────────────

    @staticmethod
    def _derive_reply_hint(
        posture: ParticipationPosture,
        ctx: EventContext,
        rd: SubjectiveReadiness,
    ) -> str:
        et = ctx.event_type
        if posture == ParticipationPosture.LIGHT_CHIME_IN:
            if et == EventType.NEWCOMER_WELCOME:
                return "简短欢迎，不要抢话，一句话就好"
            if et in (
                EventType.BIRTHDAY_CELEBRATION,
                EventType.HOLIDAY_GREETING,
                EventType.CONGRATULATIONS,
            ):
                return "跟随群体礼仪，简短附和一句即可"
            if et == EventType.GOSSIP_MELON:
                return "轻度插话表达兴趣，比如'啥情况'/'我也想听'"
            if et == EventType.MEME_CHAIN:
                return "可以跟一个表情包或简短反应，但不要带头复读"
            return "轻插一句，不要太长，不要抢焦点"
        if posture == ParticipationPosture.NORMAL_JOIN:
            if et == EventType.COLLECTIVE_STORYTELLING:
                return "可以分享自己的相关经历或看法，自然融入"
            if et == EventType.DEBATE_CIRCLE:
                return "可以发表观点，但保持理性不激化矛盾"
            return "正常参与讨论，可以展开说几句"
        if posture == ParticipationPosture.ACTIVE_ENGAGE:
            if et == EventType.BOT_DISCUSSION:
                return "关于自己的话题可以多说一些，解释或回应"
            return "积极参与，可以主导一小段对话"
        if posture == ParticipationPosture.COMFORT_SUPPORT:
            return "温柔安慰或表示支持，语气要软，不要讲大道理"
        if posture == ParticipationPosture.DEFEND_COUNTER:
            return "为自己辩解或反驳，但要控制情绪，不要升级冲突"
        return "正常回复"

    @staticmethod
    def _derive_tone_hint(
        posture: ParticipationPosture,
        ctx: EventContext,
        rd: SubjectiveReadiness,
    ) -> str:
        if posture == ParticipationPosture.COMFORT_SUPPORT:
            return "温柔"
        if posture == ParticipationPosture.DEFEND_COUNTER:
            return "认真但不攻击" if rd.annoyance_to_group < 0.5 else "克制"
        if posture == ParticipationPosture.LIGHT_CHIME_IN:
            if ctx.event_type in (
                EventType.NEWCOMER_WELCOME,
                EventType.BIRTHDAY_CELEBRATION,
                EventType.HOLIDAY_GREETING,
            ):
                return "友好"
            return "随意"
        if posture == ParticipationPosture.ACTIVE_ENGAGE:
            return "热情" if rd.social_willingness > 0.6 else "认真"
        if posture == ParticipationPosture.NORMAL_JOIN:
            return "自然"
        return "平淡"

    @staticmethod
    def _derive_narration_hint(
        posture: ParticipationPosture,
        ctx: EventContext,
        rd: SubjectiveReadiness,
    ) -> str:
        et = ctx.event_type
        label = et.label()
        if posture == ParticipationPosture.AVOID_COMPLETELY:
            if et == EventType.GROUP_CONFLICT:
                return "气氛不太对，别掺和进去"
            return f"{label}跟我关系不大，算了不参与了"
        if posture == ParticipationPosture.SILENT_OBSERVER:
            if et == EventType.GOSSIP_MELON:
                return "好像有瓜，先看看再说"
            if et == EventType.NEWCOMER_WELCOME:
                return "有人来了，不过应该不用我打招呼"
            return f"群里在{label}，我先看看"
        if posture == ParticipationPosture.CASUAL_SPECTATOR:
            return f"{label}有点意思，随便瞄两眼"
        if posture == ParticipationPosture.LIGHT_CHIME_IN:
            if et == EventType.GOSSIP_MELON:
                return "这瓜看起来不错，凑过去听听"
            if et == EventType.NEWCOMER_WELCOME:
                return "新人来了，打个招呼吧"
            return f"{label}，我可以凑个热闹说一句"
        if posture == ParticipationPosture.NORMAL_JOIN:
            return f"他们在{label}，我也可以聊两句"
        if posture == ParticipationPosture.ACTIVE_ENGAGE:
            if et == EventType.BOT_DISCUSSION:
                return "他们在说我，我得回应一下"
            return f"{label}挺感兴趣的，我想多聊聊"
        if posture == ParticipationPosture.COMFORT_SUPPORT:
            return "感觉有人需要安慰，说两句暖心的话"
        if posture == ParticipationPosture.DEFEND_COUNTER:
            return "他们说的不太对，我得解释一下"
        return "看看情况再决定"

    @staticmethod
    def _derive_defer_reason(
        posture: ParticipationPosture,
        ctx: EventContext,
        rd: SubjectiveReadiness,
    ) -> str:
        reasons = []
        if rd.energy_ratio < 0.25:
            reasons.append("精力不足")
        if rd.loafing_level > 0.6:
            reasons.append("不太想理群")
        if rd.watch_state_rank <= 1:
            reasons.append("没在看群")
        if rd.annoyance_to_group > 0.5:
            reasons.append("对群有点烦")
        if rd.trauma_score > 0.4:
            reasons.append("状态不适合社交")
        if rd.want_learn_first:
            reasons.append("想先观察学习")
        if (
            ctx.event_type == EventType.GROUP_CONFLICT
            and not ctx.is_bot_relevant
        ):
            reasons.append("别人的事不掺和")
        if not ctx.is_bot_relevant and ctx.confidence < 0.5:
            reasons.append("跟自己关系不大")
        return "；".join(reasons) if reasons else "暂时不想参与"

    # ────────────────── 内部记录方法 ──────────────────

    def _log_verdict(self, verdict: StrategyVerdict) -> None:
        self._recent_verdicts.append(verdict)
        if len(self._recent_verdicts) > self._max_history:
            self._recent_verdicts = self._recent_verdicts[-self._max_history:]

    def _record_participation(self, event_type: str, did_act: bool) -> None:
        key = f"{event_type}:{'act' if did_act else 'skip'}"
        self._event_participation_log[key] = (
            self._event_participation_log.get(key, 0) + 1
        )
        if len(self._event_participation_log) > self._max_event_log:
            items = sorted(
                self._event_participation_log.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            self._event_participation_log = dict(items[: self._max_event_log])


def get_gossip_ritual_engine(channel_id: str) -> GossipRitualStrategyEngine:
    if channel_id not in _strategy_instances:
        _strategy_instances[channel_id] = GossipRitualStrategyEngine(
            channel_id
        )
    return _strategy_instances[channel_id]


def remove_gossip_ritual_engine(channel_id: str) -> None:
    _strategy_instances.pop(channel_id, None)

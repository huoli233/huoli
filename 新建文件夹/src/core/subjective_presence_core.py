import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from src.common.logger import get_logger

logger = get_logger("presence_core")

_presence_instances: Dict[str, "SubjectivePresenceCore"] = {}


@dataclass
class PresenceState:
    """主观存在态快照
    表达主体当前对外界的关注意愿与内在倾向。
    所有字段范围 0.0~1.0。
    """

    outward_attention: float = 0.5
    social_willingness: float = 0.5
    watch_willingness: float = 0.5
    quiet_preference: float = 0.2
    curiosity_level: float = 0.3
    avoidance_tendency: float = 0.0
    computed_at: float = field(default_factory=time.time)

    def dominant_mode(self) -> str:
        """当前主导模式标签"""
        if self.quiet_preference > 0.7:
            return "想安静"
        if self.avoidance_tendency > 0.6:
            return "回避中"
        if self.social_willingness > 0.7:
            return "想社交"
        if self.watch_willingness > 0.7 and self.social_willingness < 0.4:
            return "纯围观"
        if self.curiosity_level > 0.6:
            return "有点好奇"
        return "一般"

    def participation_verdict(self) -> str:
        """综合6维存在态，给出参与群聊意愿的一词判定
        返回值：eager（渴望参与）/ willing（愿意参与）
               / reluctant（不太情愿）/ refusing（拒绝参与）
        """
        # 正向驱动力 = 社交意愿 + 外向注意力 + 好奇水平
        positive_drive = (
            self.social_willingness * 0.45
            + self.outward_attention * 0.30
            + self.curiosity_level * 0.25
        )
        # 负向抑制力 = 安静偏好 + 回避倾向
        negative_brake = (
            self.quiet_preference * 0.55 + self.avoidance_tendency * 0.45
        )
        net = positive_drive - negative_brake * 0.7
        if net > 0.45:
            return "eager"
        if net > 0.2:
            return "willing"
        if net > 0.0:
            return "reluctant"
        return "refusing"

    def mood_label(self) -> str:
        """从已有维度推导当前情绪标签
        覆盖：无聊 / 烦躁 / 好奇 / 想围观 / 想参与 / 疲倦 / 平静
        """
        # 优先判断负面情绪
        if self.avoidance_tendency > 0.6:
            return "烦躁"
        if self.quiet_preference > 0.65 and self.social_willingness < 0.3:
            return "疲倦"
        # 正向情绪
        if self.curiosity_level > 0.55:
            if self.social_willingness > 0.55:
                return "想参与"
            return "好奇"
        if self.watch_willingness > 0.6 and self.social_willingness < 0.4:
            return "想围观"
        if self.social_willingness > 0.65:
            return "想参与"
        # 中性: 判断是否无聊（低注意力 + 低好奇 + 低社交意愿）
        if self.outward_attention < 0.35 and self.curiosity_level < 0.3:
            return "无聊"
        return "平静"

    def topic_interest_score(
        self, topic: str, known_topics: Optional[list] = None
    ) -> float:
        """对给定话题的兴趣评分 0.0~1.0
        受好奇水平、社交意愿、话题熟悉度影响。
        known_topics 为主体已知/偏好的话题列表，命中则加分。
        """
        base_curiosity = (
            self.curiosity_level * 0.5 + self.social_willingness * 0.2
        )
        # 话题匹配加分
        familiarity_bonus = 0.0
        if known_topics and topic:
            topic_lower = topic.lower()
            for kt in known_topics:
                if isinstance(kt, str) and (
                    topic_lower in kt.lower() or kt.lower() in topic_lower
                ):
                    familiarity_bonus = 0.25
                    break
        raw = (
            base_curiosity + familiarity_bonus + self.outward_attention * 0.15
        )
        # 回避/安静抑制
        raw -= self.avoidance_tendency * 0.2 + self.quiet_preference * 0.1
        return max(0.0, min(1.0, raw))

    def should_learn_before_reply(
        self,
        comprehension_level: str = "",
        has_unknown_meme: bool = False,
    ) -> bool:
        """判断是否应该先学习再发言
        当理解水平低且存在未知梗/话题时，好奇心高的主体倾向先学。
        """
        comprehension_deficit = comprehension_level in (
            "cannot_understand",
            "need_clarification",
        )
        if not comprehension_deficit and not has_unknown_meme:
            return False
        # 好奇心高 + 非回避状态 → 先学习
        if self.curiosity_level > 0.4 and self.avoidance_tendency < 0.5:
            return True
        # 即使好奇心不高，完全不理解时也建议学习
        if (
            comprehension_level == "cannot_understand"
            and self.avoidance_tendency < 0.6
        ):
            return True
        return False

    def full_verdict_dict(
        self,
        comprehension_level: str = "",
        has_unknown_meme: bool = False,
        current_topic: str = "",
        known_topics: Optional[list] = None,
    ) -> Dict[str, Any]:
        """组装完整 SELF 判定字典，供面板和旁白消费"""
        t_interest = (
            self.topic_interest_score(current_topic, known_topics)
            if current_topic
            else 0.0
        )
        learn_first = self.should_learn_before_reply(
            comprehension_level, has_unknown_meme
        )
        return {
            "participation_verdict": self.participation_verdict(),
            "mood_label": self.mood_label(),
            "dominant_mode": self.dominant_mode(),
            "topic_interest_score": round(t_interest, 3),
            "want_learn_first": learn_first,
            "outward_attention": round(self.outward_attention, 3),
            "social_willingness": round(self.social_willingness, 3),
            "watch_willingness": round(self.watch_willingness, 3),
            "quiet_preference": round(self.quiet_preference, 3),
            "curiosity_level": round(self.curiosity_level, 3),
            "avoidance_tendency": round(self.avoidance_tendency, 3),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outward_attention": round(self.outward_attention, 3),
            "social_willingness": round(self.social_willingness, 3),
            "watch_willingness": round(self.watch_willingness, 3),
            "quiet_preference": round(self.quiet_preference, 3),
            "curiosity_level": round(self.curiosity_level, 3),
            "avoidance_tendency": round(self.avoidance_tendency, 3),
            "dominant_mode": self.dominant_mode(),
            "participation_verdict": self.participation_verdict(),
            "mood_label": self.mood_label(),
        }


class SubjectivePresenceCore:
    """主观存在层核心
    接收资源态、情绪、关系和群环境信号，计算主体当前的主观存在状态。
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._latest_state = PresenceState()
        self._history: list = []
        self._max_history = 20

    @property
    def state(self) -> PresenceState:
        return self._latest_state

    def compute(
        self,
        *,
        chat_energy_ratio: float = 1.0,
        thinking_energy_ratio: float = 1.0,
        activity_level: float = 50.0,
        boredom: float = 0.0,
        loneliness: float = 0.0,
        social_desire: float = 0.0,
        silence_seconds: float = 0.0,
        channel_annoyance: float = 0.0,
        vexation: float = 0.0,
        weariness: float = 0.0,
        has_interesting_topic: bool = False,
        has_close_friend_active: bool = False,
    ) -> PresenceState:
        """综合多因子计算主观存在态"""
        # 外向注意力：精力越充沛、活跃越高 → 越关注外界
        energy_factor = chat_energy_ratio * 0.6 + thinking_energy_ratio * 0.4
        activity_norm = min(1.0, activity_level / 100.0)
        outward = _sigmoid_blend(
            energy_factor * 0.4 + activity_norm * 0.3 + (1.0 - boredom * 0.3)
        )
        # 沉默越久越不关注
        silence_decay = 1.0 / (1.0 + silence_seconds / 300.0)
        outward *= silence_decay
        # 社交意愿：社交欲望 + 孤独感 - 烦躁 - 疲惫
        social_raw = (
            social_desire * 0.35
            + loneliness * 0.25
            + (0.3 if has_close_friend_active else 0.0)
            - channel_annoyance * 0.2
            - weariness * 0.15
        )
        social_will = _clamp01(social_raw + 0.3)
        # 看群意愿：好奇 + 无聊驱动 - 疲惫
        watch_raw = (
            boredom * 0.3
            + (0.25 if has_interesting_topic else 0.0)
            + activity_norm * 0.15
            - weariness * 0.2
            - vexation * 0.1
        )
        watch_will = _clamp01(watch_raw + 0.35)
        # 安静偏好：疲惫 + 烦躁 + 低精力
        quiet_raw = (
            weariness * 0.3
            + channel_annoyance * 0.25
            + vexation * 0.15
            + (1.0 - energy_factor) * 0.2
            - social_desire * 0.15
        )
        quiet_pref = _clamp01(quiet_raw)
        # 好奇水平
        curiosity_raw = (
            (0.35 if has_interesting_topic else 0.1)
            + boredom * 0.2
            + social_desire * 0.1
            - weariness * 0.15
        )
        curiosity = _clamp01(curiosity_raw)
        # 回避倾向：高烦躁 + 高压力 + 高创伤残留
        avoidance_raw = (
            channel_annoyance * 0.35
            + vexation * 0.25
            + weariness * 0.2
            - social_desire * 0.15
        )
        avoidance = _clamp01(avoidance_raw)
        new_state = PresenceState(
            outward_attention=round(outward, 4),
            social_willingness=round(social_will, 4),
            watch_willingness=round(watch_will, 4),
            quiet_preference=round(quiet_pref, 4),
            curiosity_level=round(curiosity, 4),
            avoidance_tendency=round(avoidance, 4),
            computed_at=time.time(),
        )
        self._latest_state = new_state
        self._history.append(new_state)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        return new_state

    def compute_from_snapshot(self, snapshot) -> PresenceState:
        """直接从 WorldSnapshot 计算主观存在态"""
        res = snapshot.self_resources
        sc = snapshot.scene
        return self.compute(
            chat_energy_ratio=res.chat_ratio(),
            thinking_energy_ratio=res.thinking_ratio(),
            activity_level=res.activity_level,
            boredom=res.boredom,
            loneliness=res.loneliness,
            social_desire=res.social_desire,
            silence_seconds=res.silence_seconds,
            channel_annoyance=res.channel_annoyance,
            vexation=sc.vexation,
            weariness=sc.weariness,
            has_interesting_topic=bool(sc.current_topics),
            has_close_friend_active=False,
        )

    def trend_direction(self) -> str:
        """最近存在态变化趋势"""
        if len(self._history) < 3:
            return "稳定"
        recent = self._history[-3:]
        att_delta = recent[-1].outward_attention - recent[0].outward_attention
        if att_delta > 0.1:
            return "关注上升"
        if att_delta < -0.1:
            return "关注下降"
        return "稳定"


# ---------------------------------------------------------------------------
#  辅助函数
# ---------------------------------------------------------------------------


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _sigmoid_blend(
    x: float, midpoint: float = 0.5, steepness: float = 6.0
) -> float:
    """Sigmoid 映射到 0~1"""
    z = steepness * (x - midpoint)
    z = max(-12.0, min(12.0, z))
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
#  全局访问接口
# ---------------------------------------------------------------------------


def get_presence_core(channel_id: str) -> SubjectivePresenceCore:
    """获取或创建频道的主观存在层实例"""
    if channel_id not in _presence_instances:
        _presence_instances[channel_id] = SubjectivePresenceCore(channel_id)
    return _presence_instances[channel_id]


def remove_presence_core(channel_id: str) -> None:
    _presence_instances.pop(channel_id, None)

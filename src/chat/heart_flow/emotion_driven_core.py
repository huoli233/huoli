import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger("情感驱动")


@dataclass
class InternalClock:
    """内在时钟 - 机器人自己的时间感"""

    circadian_phase: float = 12.0  # 0-24小时内在节律，默认中午
    energy_cycle: float = 1.0  # 能量周期（高涨-低谷）
    continuous_active_min: float = 0.0  # 连续活跃分钟（真实累积）
    mental_fatigue: float = 0.0  # 精神疲劳（0-1，真实累积，不会自动恢复）
    social_satiation: float = 0.0  # 社交饱腹感（说多了就不想说了）
    last_deep_rest: float = 0.0  # 上次真正休息的时间戳

    def tick(self, is_active: bool, now: float):
        """时间流逝 - 真正的内在时间"""
        if is_active:
            # 活跃时：疲劳真实累积
            self.continuous_active_min += 1
            # 精神疲劳：越活跃越累，不会自动恢复，必须休息
            fatigue_growth = 0.015 * (1 + self.continuous_active_min / 60)
            self.mental_fatigue = min(1.0, self.mental_fatigue + fatigue_growth)
            # 社交饱腹感：说多了就饱
            self.social_satiation = min(1.0, self.social_satiation + 0.04)
        else:
            # 不活跃时：缓慢恢复，但必须达到一定时间才真正恢复
            if self.continuous_active_min > 3:  # 至少休息3分钟才开始恢复
                recovery = 0.008 * (self.continuous_active_min / 20)
                self.mental_fatigue = max(0.0, self.mental_fatigue - recovery)
                self.social_satiation = max(0.0, self.social_satiation - 0.02)
            self.continuous_active_min = max(0, self.continuous_active_min - 1.5)

        # 更新昼夜节律（每分钟推进）
        self.circadian_phase = (self.circadian_phase + 1 / 60) % 24

    def wants_to_speak(self) -> float:
        """自然的发言欲望（0-1，连续值）"""
        base = 0.35

        # 疲劳抑制（真正的抑制，不是惩罚系数）
        fatigue_inhibition = self.mental_fatigue ** 2  # 平方让高疲劳时更不想说
        satiation_inhibition = self.social_satiation ** 1.5

        # 昼夜节律影响
        circadian_boost = 0.0
        if 9 <= self.circadian_phase <= 12:  # 上午活跃
            circadian_boost = 0.15
        elif 14 <= self.circadian_phase <= 17:  # 下午也可以
            circadian_boost = 0.08
        elif 20 <= self.circadian_phase <= 22:  # 晚上还行
            circadian_boost = 0.05
        elif 22 <= self.circadian_phase or self.circadian_phase <= 6:  # 深夜到清晨不想说
            circadian_boost = -0.25

        desire = base + circadian_boost - fatigue_inhibition * 0.6 - satiation_inhibition * 0.4
        return max(0.0, min(1.0, desire))


@dataclass
class EmotionState:
    """情感状态 - 机器人的内在感受"""

    boredom: float = 0.0
    """无聊感 (0-3)：可以超过1.0，真实累积"""
    environmental_fatigue: float = 0.0
    """环境疲劳 (0-1)：对当前环境/节奏看腻了，倾向不看不参与"""
    loneliness: float = 0.0
    """孤独感 (0-1)：随着未回复次数累积，感到孤独"""
    social_desire: float = 0.5
    """社交欲望 (0-1)：内在的想聊天的冲动"""
    mood: float = 0.5
    """心情 (0-1)：当前心情好坏"""
    energy: float = 1.0
    """精力 (0-1)：当前精力水平"""
    curiosity: float = 0.3
    """好奇心 (0-1)：对周围发生的事情的好奇程度"""
    proactive_willingness: float = 0.0
    """主动意愿 (0-1)：综合计算出的主动行为意愿"""
    last_update: float = 0.0
    """上次更新时间"""
    last_interaction: float = 0.0
    """上次交互时间"""
    silence_duration: float = 0.0
    """当前沉默时长（秒）"""
    unanswered_count: int = 0
    """连续未回复次数"""
    monitoring: bool = False
    """是否在监控状态"""
    monitoring_start: float = 0.0
    """监控开始时间"""
    feeling_description: str = ""
    """当前感受描述（自然语言）"""
    internal_clock: InternalClock = field(default_factory=InternalClock)
    """内在时钟 - 机器人自己的时间感"""
    withdrawal_tendency: float = 0.0
    """放弃倾向 (0-1)：太无聊了，不想参与了"""


@dataclass
class EmotionEvent:
    """情感事件 - 影响情感状态的事件"""

    event_type: str
    """事件类型：user_message, bot_message, silence_tick, etc."""
    impact: Dict[str, float]
    """对情感状态的影响"""
    description: str = ""
    """事件描述"""


class EmotionDrivenCore:
    """
    情感驱动核心

    这是真正的自主行为系统的核心。机器人不是被规则推着走，
    而是被内在的情感状态驱动。

    工作原理：
    1. 情感状态持续累积/衰减（无聊感、孤独感等）
    2. 当情感状态达到"感受阈值"时，机器人会"想要"做某事
    3. 这种"想要"是内在驱动的，不是外部规则触发的

    举例：
    - 规则驱动：沉默超过1200秒 → 触发主动行为
    - 情感驱动：无聊感累积到0.6 → 机器人感到"有点无聊，想找人聊聊"
    """

    _instance: Optional["EmotionDrivenCore"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        boredom_growth_rate: float = 0.07,
        loneliness_growth_rate: float = 0.22,
        boredom_decay_on_interaction: float = 0.35,
        monitoring_start_silence: float = 90.0,
    ):
        if self._initialized:
            return
        self._initialized = True
        self._boredom_growth_rate = boredom_growth_rate
        self._loneliness_growth_rate = loneliness_growth_rate
        self._boredom_decay_on_interaction = boredom_decay_on_interaction
        self._monitoring_start_silence = monitoring_start_silence
        self._states: Dict[str, EmotionState] = {}
        self._event_history: List[EmotionEvent] = []
        self._max_history = 100
        logger.info("[情感驱动] 情感驱动核心初始化完成")

    def _get_state(self, channel_id: str) -> EmotionState:
        """获取或创建频道的情感状态"""
        if channel_id not in self._states:
            now = time.time()
            self._states[channel_id] = EmotionState(
                last_update=now,
                last_interaction=now,
            )
        return self._states[channel_id]

    def get_state(self, channel_id: str) -> "EmotionState":
        """获取频道的情感状态（公开接口，不存在时自动创建）"""
        return self._get_state(channel_id)

    def tick(self, channel_id: str, silence_sec: float, unanswered: int = 0,
             context_richness: float = 0.5, group_activity: float = 0.5,
             topic_repetitiveness: float = 0.5, is_active: bool = False) -> EmotionState:
        """
        情感时钟滴答 - 每次循环调用，更新情感状态

        这是情感驱动的核心：情感状态会随着时间自然变化，
        不是被外部规则触发，而是内在累积。
        """
        state = self._get_state(channel_id)
        now = time.time()
        state.silence_duration = max(0.0, float(silence_sec))
        state.unanswered_count = max(0, int(unanswered))
        silence_min = state.silence_duration / 60.0

        # 更新内在时钟
        state.internal_clock.tick(is_active, now)

        if state.silence_duration >= self._monitoring_start_silence and not state.monitoring:
            state.monitoring = True
            state.monitoring_start = now
            logger.info(f"[情感驱动] {channel_id[:8]} 开始监控情感状态，沉默{state.silence_duration:.0f}秒")
        if state.monitoring:
            self._accumulate_boredom(state, silence_min, context_richness)
            self._accumulate_loneliness(state, silence_min, unanswered, group_activity)
            self._update_environmental_fatigue(state, silence_min, topic_repetitiveness)
        self._update_social_desire(state)
        self._calculate_proactive_willingness(state)
        state.feeling_description = self._describe_feeling(state)
        state.last_update = now

        # 只在意愿较高时记录日志
        if state.proactive_willingness >= 0.5:
            logger.info(
                f"[情感驱动] {channel_id[:8]} 感受到: {state.feeling_description} "
                f"(无聊={state.boredom:.2f}, 孤独={state.loneliness:.2f}, 环境疲劳={state.environmental_fatigue:.2f}, "
                f"疲劳={state.internal_clock.mental_fatigue:.2f}, 饱腹={state.internal_clock.social_satiation:.2f}, 意愿={state.proactive_willingness:.2f})"
            )
        return state

    def _accumulate_boredom(self, state: EmotionState, silence_min: float, context_richness: float = 0.5) -> None:
        """
        真正累积无聊感 - 取决于内容质量，不只是时间

        无聊感是真实的体验：
        - 沉默时间长 + 群里冷清 → 无聊加速累积
        - 沉默时间长 + 群里热闹 → 我在观察，不无聊
        - 无聊太久 → 转化为"放弃"倾向
        """
        # 基础增长：每分钟0.05
        base_growth = silence_min * 0.05

        # 内容质量影响：群里有意思就不无聊
        if context_richness > 0.6:
            # 有有意思的事 → 我在观察，不无聊
            base_growth *= 0.15
        elif context_richness < 0.2:
            # 群里很冷清 → 无聊加速
            base_growth *= 1.8

        # 心情影响
        if state.mood < 0.3:
            base_growth *= 1.2  # 心情差更容易无聊
        if state.mood > 0.7:
            base_growth *= 0.7  # 心情好不容易无聊

        # 精力影响：精力低时没力气无聊
        if state.energy < 0.3:
            base_growth *= 0.4

        # 真正累积（可以超过1.0，但超过后增长放缓）
        if state.boredom < 1.0:
            state.boredom = min(2.0, state.boredom + base_growth)
        else:
            # 超过1.0后，边际效应（但不会停止增长）
            state.boredom = min(3.0, state.boredom + base_growth * 0.25)

        # 无聊太久会转化为"放弃"倾向（不是更想说话）
        if state.boredom > 1.5:
            withdrawal = min(1.0, (state.boredom - 1.5) * 0.8)
            # 降低社交欲望（太无聊了，不想参与了）
            state.social_desire = max(0.1, state.social_desire - withdrawal * 0.1)

    def _accumulate_loneliness(self, state: EmotionState, silence_min: float, unanswered: int, group_activity: float = 0.5) -> None:
        """
        真正累积孤独感 - 取决于群体活跃度

        孤独感是真实的体验：
        - 群里很热闹但没人理我 → 孤独感加速（被忽视的感觉）
        - 群里很冷清 → 大家一起沉默，不孤独
        - 有人回复我 → 孤独感大幅降低
        - 孤独太久 → 转化为"独立"（习惯一个人）
        """
        # 基础增长：未回复次数贡献
        base_growth = max(0, unanswered) * 0.08

        # 沉默时间贡献（但前5分钟不算，给自己时间）
        if silence_min > 5:
            base_growth += (silence_min - 5) * 0.015

        # 关键：群体活跃度影响
        if group_activity > 0.6 and unanswered > 0:
            # 群里很热闹但没人理我 → 被忽视的感觉，孤独加速
            base_growth *= 1.5
        elif group_activity < 0.2:
            # 群里很冷清 → 大家一起沉默，不孤独
            base_growth *= 0.3

        # 心情影响：心情差时更容易感到孤独
        if state.mood < 0.35:
            base_growth *= 1.3
        if state.mood > 0.75:
            base_growth *= 0.7

        # 真正累积（可以超过1.0）
        if state.loneliness < 1.0:
            state.loneliness = min(1.8, state.loneliness + base_growth)
        else:
            # 超过1.0后增长放缓
            state.loneliness = min(2.5, state.loneliness + base_growth * 0.2)

        # 孤独太久会转化为"独立"（习惯一个人，反而不想参与）
        if state.loneliness > 1.3:
            independence = min(1.0, (state.loneliness - 1.3) * 0.6)
            # 降低社交欲望（习惯一个人了）
            state.social_desire = max(0.1, state.social_desire - independence * 0.08)

    def _update_environmental_fatigue(self, state: EmotionState, silence_min: float, topic_repetitiveness: float = 0.5) -> None:
        """
        真正累积环境疲劳 - 对当前场域失去兴趣

        环境疲劳是真实的体验：
        - 话题重复无聊 → 加速疲劳
        - 话题新鲜有趣 → 不疲劳
        - 长时间同一话题 → 边际效应
        - 疲劳太久 → 想离开这个环境
        """
        # 基础增长：长时间在同一环境
        base_growth = max(0.0, silence_min - 10.0) * 0.02

        # 话题重复度影响（最关键）
        if topic_repetitiveness > 0.7:
            # 话题很重复 → 快速疲劳
            base_growth += 0.08
            base_growth *= 1.6
        elif topic_repetitiveness < 0.3:
            # 话题很新鲜 → 不疲劳，甚至恢复
            base_growth = -0.05

        # 精力影响：精力低时更容易疲劳
        if state.energy < 0.4:
            base_growth *= 1.4

        # 心情影响：心情差时更容易疲劳
        if state.mood < 0.35:
            base_growth *= 1.25
        if state.mood > 0.8:
            base_growth *= 0.6

        # 好奇心抑制：好奇时不容易疲劳
        if state.curiosity > 0.6:
            base_growth *= 0.7

        # 真正累积（可以增长也可以减少）
        if base_growth > 0:
            if state.environmental_fatigue < 1.0:
                state.environmental_fatigue = min(1.5, state.environmental_fatigue + base_growth)
            else:
                state.environmental_fatigue = min(2.0, state.environmental_fatigue + base_growth * 0.15)
        else:
            # 负值表示恢复
            state.environmental_fatigue = max(0.0, state.environmental_fatigue + base_growth)

        # 环境疲劳太高 → 想离开（降低参与意愿）
        if state.environmental_fatigue > 0.85:
            leave_desire = min(1.0, (state.environmental_fatigue - 0.85) * 2.0)
            state.withdrawal_tendency = max(state.withdrawal_tendency, leave_desire * 0.5)

    def _update_social_desire(self, state: EmotionState) -> None:
        """
        更新社交欲望 - 更自然的内在驱动

        社交欲望是内在的想聊天的冲动，但受到：
        - 疲劳抑制（累了就不想说话）
        - 饱腹感抑制（说多了就不想再说）
        - 环境疲劳（看腻了就不想参与）
        """
        # 基础欲望降低
        base_desire = 0.18

        # 无聊和孤独的贡献降低（不会无脑想聊）
        desire_from_boredom = min(0.3, state.boredom * 0.15)  # 无聊太多反而不想聊
        desire_from_loneliness = state.loneliness * 0.18
        desire_from_mood = state.mood * 0.08
        desire_from_curiosity = state.curiosity * 0.06

        # 负面因素
        fatigue_penalty = state.internal_clock.mental_fatigue * 0.4
        satiation_penalty = state.internal_clock.social_satiation * 0.35
        env_fatigue_penalty = state.environmental_fatigue * 0.25

        _raw_desire = (
            base_desire
            + desire_from_boredom
            + desire_from_loneliness
            + desire_from_mood
            + desire_from_curiosity
            - fatigue_penalty
            - satiation_penalty
            - env_fatigue_penalty
        )

        # 精力低时大幅降低社交欲望
        if state.energy < 0.25:
            _raw_desire *= 0.3
        elif state.energy < 0.45:
            _raw_desire *= 0.6

        # 放弃倾向抑制
        if state.withdrawal_tendency > 0.3:
            _raw_desire *= (1 - state.withdrawal_tendency * 0.7)

        state.social_desire = min(1.0, max(0.0, _raw_desire))

    def _calculate_proactive_willingness(self, state: EmotionState) -> None:
        """
        计算主动意愿 - 真正的内在涌现

        不是简单的加权，而是：
        1. 内在时钟：我现在想说话吗？
        2. 情感状态：我有强烈的情感驱动吗？
        3. 疲劳状态：我有精力去做吗？
        """
        # 1. 内在时钟的发言欲望（最重要）
        internal_desire = state.internal_clock.wants_to_speak()

        # 2. 情感驱动（但不是简单的加权）
        emotional_drive = 0.0
        if state.boredom > 0.8 and state.boredom < 1.8:
            # 适度无聊 → 想参与
            emotional_drive += 0.15
        elif state.boredom >= 1.8:
            # 太无聊了 → 放弃倾向
            emotional_drive -= 0.2

        if state.loneliness > 0.5:
            emotional_drive += 0.12

        if state.social_desire > 0.6:
            emotional_drive += 0.1

        # 3. 疲劳抑制（真实的抑制）
        fatigue_factor = 1.0 - state.internal_clock.mental_fatigue ** 1.5
        satiation_factor = 1.0 - state.internal_clock.social_satiation ** 1.2

        # 4. 综合（内在欲望占主导）
        willingness = (
            internal_desire * 0.55 +      # 内在时钟最重要
            emotional_drive * 0.25 +       # 情感驱动
            state.curiosity * 0.08 +       # 好奇心
            state.mood * 0.05              # 心情
        ) * fatigue_factor * satiation_factor

        # 放弃倾向抑制
        if state.withdrawal_tendency > 0.4:
            willingness *= (1 - state.withdrawal_tendency)

        state.proactive_willingness = max(0.0, min(1.0, willingness))

    def _describe_feeling(self, state: EmotionState) -> str:
        """
        用自然语言描述当前感受

        这让机器人的感受变得可见、可理解。
        """
        feelings = []
        if state.boredom >= 0.7:
            feelings.append("非常无聊")
        elif state.boredom >= 0.4:
            feelings.append("有点无聊")
        if state.loneliness >= 0.6:
            feelings.append("感到孤独")
        elif state.loneliness >= 0.3:
            feelings.append("有点寂寞")
        if state.social_desire >= 0.7:
            feelings.append("很想找人聊天")
        elif state.social_desire >= 0.4:
            feelings.append("想聊点什么")
        if state.curiosity >= 0.6:
            feelings.append("对群里发生的事很好奇")
        if state.environmental_fatigue >= 0.65:
            feelings.append("对眼下环境有点看腻了")
        if state.mood >= 0.7:
            feelings.append("心情不错")
        elif state.mood <= 0.3:
            feelings.append("心情不太好")
        if state.energy <= 0.3:
            feelings.append("有点累")
        if not feelings:
            return "平静"
        return "，".join(feelings)

    def on_user_message(self, channel_id: str, user_id: str = "") -> EmotionState:
        """
        用户消息事件 - 用户说话了，情感状态会变化

        这不是规则，而是自然的情感反应：
        - 有人说话了，无聊感降低
        - 孤独感消失
        - 社交欲望得到满足
        """
        state = self._get_state(channel_id)
        state.boredom = max(0.0, state.boredom - self._boredom_decay_on_interaction)
        state.environmental_fatigue = max(0.0, state.environmental_fatigue - 0.2)
        state.loneliness = 0.0
        state.social_desire = min(1.0, state.social_desire + 0.1)
        state.mood = min(1.0, state.mood + 0.05)
        state.last_interaction = time.time()
        state.silence_duration = 0.0
        state.unanswered_count = 0
        state.monitoring = False
        state.monitoring_start = 0.0
        state.last_update = time.time()
        self._calculate_proactive_willingness(state)
        state.feeling_description = self._describe_feeling(state)
        logger.debug(f"[情感驱动] {channel_id[:8]} 用户消息，情感重置: 无聊={state.boredom:.2f}")
        return state

    def on_bot_message(self, channel_id: str) -> EmotionState:
        """
        机器人消息事件 - 机器人说话了，情感状态会变化

        主动发言后：
        - 无聊感降低（但不是清零）
        - 孤独感降低
        - 社交欲望降低（得到了满足）
        - 社交饱腹感增加（说多了）
        """
        state = self._get_state(channel_id)

        # 无聊感降低，但不会清零（真实体验）
        state.boredom = max(0.3, state.boredom - 0.15)
        state.environmental_fatigue = max(0.0, state.environmental_fatigue - 0.08)
        state.loneliness = max(0.0, state.loneliness - 0.15)
        state.social_desire = max(0.15, state.social_desire - 0.12)

        # 更新内在时钟 - 标记为活跃
        now = time.time()
        state.internal_clock.tick(is_active=True, now=now)

        state.last_interaction = now
        state.silence_duration = 0.0
        state.monitoring = False
        state.monitoring_start = 0.0
        state.last_update = now
        self._calculate_proactive_willingness(state)
        state.feeling_description = self._describe_feeling(state)
        logger.debug(f"[情感驱动] {channel_id[:8]} 机器人发言，疲劳={state.internal_clock.mental_fatigue:.2f}, 饱腹={state.internal_clock.social_satiation:.2f}, 意愿={state.proactive_willingness:.2f}")
        return state

    def integrate_external_state(
        self,
        channel_id: str,
        mood: Optional[float] = None,
        energy: Optional[float] = None,
        curiosity: Optional[float] = None,
    ) -> EmotionState:
        """
        整合外部状态 - 从其他系统获取状态信息

        情感状态不是孤立的，会受到：
        - 心情系统的影响
        - 能量系统的影响
        - 觉察系统的影响（好奇心）
        """
        state = self._get_state(channel_id)
        if mood is not None:
            state.mood = max(0.0, min(1.0, float(mood)))
        if energy is not None:
            state.energy = max(0.0, min(1.0, float(energy)))
        if curiosity is not None:
            state.curiosity = max(0.0, min(1.0, float(curiosity)))
        self._calculate_proactive_willingness(state)
        state.feeling_description = self._describe_feeling(state)
        state.last_update = time.time()
        return state

    def should_proactive(self, channel_id: str, context_richness: float = 0.5) -> Tuple[bool, str]:
        """
        判断是否应该主动行为 - 真正的自主决策

        不是简单的阈值判断，而是：
        1. 我有足够的内在欲望吗？
        2. 现在是个好时机吗？
        3. 我有精力去做吗？

        返回 (是否应该主动, 原因描述)
        """
        state = self._get_state(channel_id)

        # 1. 内在欲望必须足够强烈（提高阈值）
        if state.proactive_willingness < 0.72:  # 从0.65提高到0.72
            return False, f"内在意愿不够强烈({state.proactive_willingness:.2f})"

        # 2. 疲劳检查 - 太累了就不说
        if state.internal_clock.mental_fatigue > 0.7:
            return False, f"精神疲劳较高({state.internal_clock.mental_fatigue:.2f})，需要休息"

        # 3. 饱腹感检查 - 说多了就不说
        if state.internal_clock.social_satiation > 0.6:
            return False, f"社交饱腹感较高({state.internal_clock.social_satiation:.2f})，不想再说"

        # 4. 放弃倾向检查 - 太无聊了就不想参与
        if state.withdrawal_tendency > 0.5:
            return False, f"放弃倾向较高({state.withdrawal_tendency:.2f})，不想参与"

        # 5. 内容质量检查 - 群里太冷清就不说
        if context_richness < 0.25:
            return False, f"群里比较冷清({context_richness:.2f})，没有参与的动力"

        # 6. 综合判断通过，生成原因
        reasons = []
        if state.proactive_willingness >= 0.85:
            reasons.append("现在很有表达欲")
        elif state.proactive_willingness >= 0.72:
            reasons.append("感觉想说点什么")

        if 0.8 <= state.boredom < 1.5:
            reasons.append("有点无聊想参与")

        if state.loneliness > 0.5:
            reasons.append("感到有点孤独")

        if state.curiosity > 0.6:
            reasons.append("对话题很好奇")

        reason_str = "，".join(reasons) if reasons else "就是想参与一下"
        return True, reason_str

    def get_state_snapshot(self, channel_id: str) -> Dict[str, Any]:
        """获取情感状态快照"""
        state = self._get_state(channel_id)
        return {
            "boredom": round(state.boredom, 3),
            "environmental_fatigue": round(state.environmental_fatigue, 3),
            "loneliness": round(state.loneliness, 3),
            "social_desire": round(state.social_desire, 3),
            "mood": round(state.mood, 3),
            "energy": round(state.energy, 3),
            "curiosity": round(state.curiosity, 3),
            "proactive_willingness": round(state.proactive_willingness, 3),
            "silence_duration": round(state.silence_duration, 1),
            "unanswered_count": state.unanswered_count,
            "monitoring": state.monitoring,
            "feeling": state.feeling_description,
        }

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """获取所有频道的情感状态"""
        return {cid: self.get_state_snapshot(cid) for cid in self._states}


_emotion_core: Optional[EmotionDrivenCore] = None


def get_emotion_driven_core() -> EmotionDrivenCore:
    """获取情感驱动核心单例"""
    global _emotion_core
    if _emotion_core is None:
        _emotion_core = EmotionDrivenCore()
    return _emotion_core

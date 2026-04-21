import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from src.common.logger import get_logger

logger = get_logger("group_pattern")

_detector_instances: Dict[str, "GroupPatternDetector"] = {}


class GroupPattern(Enum):
    """群体行为模式，第三阶段升级：增加更多精细化模式"""

    NONE = "none"
    NEWCOMER_WELCOME = "newcomer_welcome"
    COPYCAT_CHAIN = "copycat_chain"
    SPECTATOR_MODE = "spectator_mode"
    PILE_ON = "pile_on"
    RITUAL_GREETING = "ritual_greeting"
    RAPID_EXCHANGE = "rapid_exchange"
    TOPIC_SHIFT = "topic_shift"
    SUDDEN_SILENCE = "sudden_silence"
    GOSSIP_EVENT = "gossip_event"
    EMOTIONAL_CONTAGION = "emotional_contagion"
    COLLECTIVE_STORYTELLING = "collective_storytelling"
    DEBATE_CIRCLE = "debate_circle"
    MEME_STORM = "meme_storm"
    SUPPORT_CIRCLE = "support_circle"
    CELEBRATION_WAVE = "celebration_wave"
    CONFLICT_ESCALATION = "conflict_escalation"
    QUIET_REFLECTION = "quiet_reflection"

    def label(self) -> str:
        labels = {
            GroupPattern.NONE: "无明显模式",
            GroupPattern.NEWCOMER_WELCOME: "迎新",
            GroupPattern.COPYCAT_CHAIN: "复读",
            GroupPattern.SPECTATOR_MODE: "围观",
            GroupPattern.PILE_ON: "围攻/附议",
            GroupPattern.RITUAL_GREETING: "礼仪问候",
            GroupPattern.RAPID_EXCHANGE: "快速对话",
            GroupPattern.TOPIC_SHIFT: "话题切换",
            GroupPattern.SUDDEN_SILENCE: "突然安静",
            GroupPattern.GOSSIP_EVENT: "吃瓜事件",
            GroupPattern.EMOTIONAL_CONTAGION: "情感传染",
            GroupPattern.COLLECTIVE_STORYTELLING: "集体叙事",
            GroupPattern.DEBATE_CIRCLE: "辩论圈",
            GroupPattern.MEME_STORM: "梗风暴",
            GroupPattern.SUPPORT_CIRCLE: "支持圈",
            GroupPattern.CELEBRATION_WAVE: "庆祝浪潮",
            GroupPattern.CONFLICT_ESCALATION: "冲突升级",
            GroupPattern.QUIET_REFLECTION: "安静反思",
        }
        return labels.get(self, "未知")


@dataclass
class PatternEvidence:
    """模式证据"""

    pattern: GroupPattern = GroupPattern.NONE
    confidence: float = 0.0
    involved_users: Set[str] = field(default_factory=set)
    supporting_texts: List[str] = field(default_factory=list)
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern.value,
            "pattern_label": self.pattern.label(),
            "confidence": round(self.confidence, 3),
            "involved_users": list(self.involved_users)[:10],
            "supporting_count": len(self.supporting_texts),
        }


@dataclass
class RecentMessage:
    """近期消息简记"""

    user_id: str = ""
    text: str = ""
    timestamp: float = field(default_factory=time.time)
    is_new_member: bool = False
    is_at_someone: bool = False
    at_target: str = ""


class GroupPatternDetector:
    """群体行为模式识别器
    基于近期消息滑动窗口检测：
    - 迎新：新成员发言后多人回复
    - 复读：连续相同或高度相似文本
    - 围观：大量人看但极少人说
    - 围攻/附议：多人集中回复同一人
    - 礼仪问候：早安/晚安/新年好等集体问候
    - 快速对话：两人高频对答
    - 话题切换：关键词突然变化
    - 突然安静：从高频消息骤降
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._recent_messages: List[RecentMessage] = []
        self._max_window = 60
        self._last_patterns: List[PatternEvidence] = []
        # 礼仪关键词
        self._ritual_keywords = {
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
        }
        # 吃瓜事件识别用情绪关键词
        self._gossip_reaction_keywords = {
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
            "求解释",
            "什么操作",
            "乐了",
        }
        self._meme_keywords = {
            "哈哈哈",
            "笑死",
            "绷不住",
            "乐了",
            "草",
            "6",
            "xswl",
            "hhh",
            "2333",
            "哈哈",
            "笑了",
            "awsl",
            "太草了",
            "yyds",
            "绝绝子",
            "离谱",
        }
        # 已知活跃用户数（由外部设置，用于围观检测）
        self._known_active_count: int = 0

    def feed(self, msg: RecentMessage) -> None:
        """喂入一条新消息"""
        self._recent_messages.append(msg)
        if len(self._recent_messages) > self._max_window:
            self._recent_messages = self._recent_messages[-self._max_window:]

    def feed_simple(
        self,
        user_id: str,
        text: str,
        *,
        is_new_member: bool = False,
        is_at_someone: bool = False,
        at_target: str = "",
    ) -> None:
        """简化入口"""
        self.feed(
            RecentMessage(
                user_id=user_id,
                text=text,
                timestamp=time.time(),
                is_new_member=is_new_member,
                is_at_someone=is_at_someone,
                at_target=at_target,
            )
        )

    def set_known_active_count(self, count: int) -> None:
        """设置外部提供的已知在线/活跃用户总数"""
        self._known_active_count = max(0, count)

    def detect(self) -> List[PatternEvidence]:
        """检测当前窗口中的所有模式"""
        self._last_patterns = []
        window = self._get_active_window()
        if len(window) < 2:
            return self._last_patterns
        self._detect_copycat(window)
        self._detect_newcomer_welcome(window)
        self._detect_pile_on(window)
        self._detect_ritual_greeting(window)
        self._detect_rapid_exchange(window)
        self._detect_sudden_silence(window)
        self._detect_spectator(window)
        self._detect_gossip_event(window)
        # 第三阶段新增模式检测
        self._detect_emotional_contagion(window)
        self._detect_collective_storytelling(window)
        self._detect_debate_circle(window)
        self._detect_meme_storm(window)
        self._detect_support_circle(window)
        self._detect_celebration_wave(window)
        self._detect_conflict_escalation(window)
        self._detect_quiet_reflection(window)
        # 按置信度排序
        self._last_patterns.sort(key=lambda e: e.confidence, reverse=True)
        return self._last_patterns

    def dominant_pattern(self) -> Optional[PatternEvidence]:
        patterns = self.detect()
        if patterns and patterns[0].confidence > 0.4:
            return patterns[0]
        return None

    def has_pattern(self, pattern: GroupPattern) -> bool:
        for ev in self._last_patterns:
            if ev.pattern == pattern and ev.confidence > 0.4:
                return True
        return False

    # ────────────────── 检测方法 ──────────────────

    def _detect_copycat(self, window: List[RecentMessage]) -> None:
        """检测复读链：连续N条消息文本相同"""
        if len(window) < 3:
            return
        chain_text = ""
        chain_users: Set[str] = set()
        chain_count = 0
        for msg in reversed(window):
            normalized = msg.text.strip()
            if not normalized:
                continue
            if chain_text == "":
                chain_text = normalized
                chain_users.add(msg.user_id)
                chain_count = 1
            elif normalized == chain_text:
                chain_users.add(msg.user_id)
                chain_count += 1
            else:
                break
        if chain_count >= 3 and len(chain_users) >= 2:
            self._last_patterns.append(
                PatternEvidence(
                    pattern=GroupPattern.COPYCAT_CHAIN,
                    confidence=min(1.0, chain_count / 5.0),
                    involved_users=chain_users,
                    supporting_texts=[chain_text],
                )
            )

    def _detect_newcomer_welcome(self, window: List[RecentMessage]) -> None:
        """检测迎新：新成员发言后多人回复"""
        newcomer_msgs = [m for m in window if m.is_new_member]
        if not newcomer_msgs:
            return
        for nm in newcomer_msgs:
            after_msgs = [
                m
                for m in window
                if m.timestamp > nm.timestamp and m.user_id != nm.user_id
            ]
            responders = set(m.user_id for m in after_msgs[:10])
            if len(responders) >= 2:
                self._last_patterns.append(
                    PatternEvidence(
                        pattern=GroupPattern.NEWCOMER_WELCOME,
                        confidence=min(1.0, len(responders) / 4.0),
                        involved_users=responders | {nm.user_id},
                        supporting_texts=[nm.text[:30]],
                    )
                )

    def _detect_pile_on(self, window: List[RecentMessage]) -> None:
        """检测围攻/附议：多人集中@或回复同一人"""
        at_targets: Dict[str, Set[str]] = {}
        for msg in window:
            if msg.is_at_someone and msg.at_target:
                if msg.at_target not in at_targets:
                    at_targets[msg.at_target] = set()
                at_targets[msg.at_target].add(msg.user_id)
        for target, attackers in at_targets.items():
            if len(attackers) >= 3:
                self._last_patterns.append(
                    PatternEvidence(
                        pattern=GroupPattern.PILE_ON,
                        confidence=min(1.0, len(attackers) / 5.0),
                        involved_users=attackers | {target},
                    )
                )

    def _detect_ritual_greeting(self, window: List[RecentMessage]) -> None:
        """检测礼仪问候"""
        ritual_users: Set[str] = set()
        ritual_texts: List[str] = []
        for msg in window:
            text_lower = msg.text.strip()
            if any(kw in text_lower for kw in self._ritual_keywords):
                ritual_users.add(msg.user_id)
                ritual_texts.append(text_lower[:20])
        if len(ritual_users) >= 3:
            self._last_patterns.append(
                PatternEvidence(
                    pattern=GroupPattern.RITUAL_GREETING,
                    confidence=min(1.0, len(ritual_users) / 5.0),
                    involved_users=ritual_users,
                    supporting_texts=ritual_texts[:5],
                )
            )

    def _detect_rapid_exchange(self, window: List[RecentMessage]) -> None:
        """检测快速对话：两人在短时间内频繁对答"""
        if len(window) < 6:
            return
        recent_six = window[-6:]
        speakers = set(m.user_id for m in recent_six)
        if len(speakers) == 2:
            time_span = recent_six[-1].timestamp - recent_six[0].timestamp
            if time_span < 60 and time_span > 0:
                self._last_patterns.append(
                    PatternEvidence(
                        pattern=GroupPattern.RAPID_EXCHANGE,
                        confidence=min(1.0, 6.0 / max(1.0, time_span / 10.0)),
                        involved_users=speakers,
                    )
                )

    def _detect_sudden_silence(self, window: List[RecentMessage]) -> None:
        """检测突然安静：前半窗口消息密集，后半窗口消息稀疏"""
        if len(window) < 8:
            return
        mid = len(window) // 2
        first_half = window[:mid]
        second_half = window[mid:]
        if not first_half or not second_half:
            return
        first_span = first_half[-1].timestamp - first_half[0].timestamp
        second_span = second_half[-1].timestamp - second_half[0].timestamp
        if first_span < 1 or second_span < 1:
            return
        first_rate = len(first_half) / (first_span / 60.0)
        second_rate = len(second_half) / (second_span / 60.0)
        if first_rate > 5.0 and second_rate < 1.0:
            self._last_patterns.append(
                PatternEvidence(
                    pattern=GroupPattern.SUDDEN_SILENCE,
                    confidence=min(1.0, first_rate / second_rate / 10.0),
                    involved_users=set(m.user_id for m in window),
                )
            )

    def _detect_spectator(self, window: List[RecentMessage]) -> None:
        """检测围观模式：少数人在说话、大量用户沉默围观
        判据：
          - 窗口内发言者 <= 2 人
          - 已知活跃用户数 >= 5 人（由 set_known_active_count 提供）
          - 消息频率中等（有人在聊，但观众远多于演员）
        """
        if len(window) < 4:
            return
        speakers = set(m.user_id for m in window)
        speaker_count = len(speakers)
        known_active = self._known_active_count
        # 需要外部已知活跃人数，否则无法判断"沉默围观"
        if known_active < 5:
            return
        if speaker_count > 2:
            return
        # 沉默比 = 围观者 / 发言者
        silent_ratio = (known_active - speaker_count) / max(1, speaker_count)
        if silent_ratio < 2.0:
            return
        # 消息频率需要在中等以上（不是单纯安静）
        time_span = window[-1].timestamp - window[0].timestamp
        if time_span < 1:
            return
        msg_per_min = len(window) / (time_span / 60.0)
        if msg_per_min < 1.0:
            return
        confidence = min(1.0, silent_ratio / 5.0) * min(1.0, msg_per_min / 3.0)
        if confidence > 0.25:
            self._last_patterns.append(
                PatternEvidence(
                    pattern=GroupPattern.SPECTATOR_MODE,
                    confidence=confidence,
                    involved_users=speakers,
                    supporting_texts=[
                        f"发言{speaker_count}人/围观约{known_active - speaker_count}人"
                    ],
                )
            )

    def _detect_gossip_event(self, window: List[RecentMessage]) -> None:
        """检测吃瓜事件：多人用惊讶/猎奇语气讨论第三方
        判据：
          - 窗口内 >= 3 条消息含吃瓜反应关键词
          - 反应消息来自 >= 2 个不同用户
          - 可选加分：有人提及不在发言列表中的第三人
        """
        if len(window) < 4:
            return
        reaction_users: Set[str] = set()
        reaction_texts: List[str] = []
        # 收集含吃瓜关键词的消息
        for msg in window:
            text = msg.text.strip()
            if any(kw in text for kw in self._gossip_reaction_keywords):
                reaction_users.add(msg.user_id)
                reaction_texts.append(text[:30])
        if len(reaction_texts) < 3 or len(reaction_users) < 2:
            return
        # 识别被讨论的第三方（@但目标不在发言列表中）
        speakers = set(m.user_id for m in window)
        third_party_mentioned = False
        for msg in window:
            if (
                msg.is_at_someone
                and msg.at_target
                and msg.at_target not in speakers
            ):
                third_party_mentioned = True
                break
        base_conf = min(1.0, len(reaction_texts) / 6.0) * min(
            1.0, len(reaction_users) / 3.0
        )
        if third_party_mentioned:
            base_conf = min(1.0, base_conf + 0.2)
        if base_conf > 0.3:
            self._last_patterns.append(
                PatternEvidence(
                    pattern=GroupPattern.GOSSIP_EVENT,
                    confidence=base_conf,
                    involved_users=reaction_users,
                    supporting_texts=reaction_texts[:5],
                )
            )

    # 第三阶段新增检测方法
    def _detect_emotional_contagion(self, window: List[RecentMessage]) -> None:
        """检测情感传染：多人表达相似情绪"""
        if len(window) < 4:
            return
        emotion_keywords = {
            "positive": [
                "开心",
                "高兴",
                "快乐",
                "棒",
                "好",
                "赞",
                "厉害",
                "牛",
            ],
            "negative": [
                "难过",
                "伤心",
                "生气",
                "烦",
                "累",
                "讨厌",
                "糟糕",
                "坑",
            ],
            "surprise": ["震惊", "意外", "哇", "天哪", "真的吗", "不会吧"],
        }
        emotion_counts = {"positive": 0, "negative": 0, "surprise": 0}
        emotion_users = {
            "positive": set(),
            "negative": set(),
            "surprise": set(),
        }
        for msg in window[-10:]:  # 检查最近10条消息
            text = msg.text.lower()
            for emotion, keywords in emotion_keywords.items():
                if any(kw in text for kw in keywords):
                    emotion_counts[emotion] += 1
                    emotion_users[emotion].add(msg.user_id)
                    break
        for emotion, count in emotion_counts.items():
            if count >= 3 and len(emotion_users[emotion]) >= 3:
                self._last_patterns.append(
                    PatternEvidence(
                        pattern=GroupPattern.EMOTIONAL_CONTAGION,
                        confidence=min(1.0, count / 5.0),
                        involved_users=emotion_users[emotion],
                        supporting_texts=[f"{emotion}_contagion"],
                    )
                )

    def _detect_collective_storytelling(
        self, window: List[RecentMessage]
    ) -> None:
        """检测集体叙事：多人围绕同一话题展开故事"""
        if len(window) < 6:
            return
        # 简化检测：连续消息中话题相关性高（实际需要更复杂的NLP）
        speakers = [m.user_id for m in window[-8:]]
        if len(set(speakers)) >= 4:  # 至少4人参与
            self._last_patterns.append(
                PatternEvidence(
                    pattern=GroupPattern.COLLECTIVE_STORYTELLING,
                    confidence=0.6,
                    involved_users=set(speakers),
                )
            )

    def _detect_debate_circle(self, window: List[RecentMessage]) -> None:
        """检测辩论圈：观点交锋，@对方回复"""
        if len(window) < 5:
            return
        at_count = sum(1 for m in window if m.is_at_someone)
        if at_count >= 3 and len(window) >= 5:
            speakers = set(m.user_id for m in window)
            self._last_patterns.append(
                PatternEvidence(
                    pattern=GroupPattern.DEBATE_CIRCLE,
                    confidence=min(1.0, at_count / 4.0),
                    involved_users=speakers,
                )
            )

    def _detect_meme_storm(self, window: List[RecentMessage]) -> None:
        """检测梗风暴：大量梗词使用"""
        meme_count = sum(
            1
            for m in window
            if any(kw in m.text for kw in self._meme_keywords)
        )
        if meme_count >= 5 and len(window) >= 5:
            users = set(m.user_id for m in window)
            self._last_patterns.append(
                PatternEvidence(
                    pattern=GroupPattern.MEME_STORM,
                    confidence=min(1.0, meme_count / 8.0),
                    involved_users=users,
                )
            )

    def _detect_support_circle(self, window: List[RecentMessage]) -> None:
        """检测支持圈：安慰鼓励类消息集中出现"""
        support_keywords = [
            "加油",
            "没事",
            "安慰",
            "支持",
            "抱抱",
            "坚强",
            "理解",
        ]
        support_count = sum(
            1 for m in window if any(kw in m.text for kw in support_keywords)
        )
        if support_count >= 3 and len(window) >= 4:
            users = set(m.user_id for m in window)
            self._last_patterns.append(
                PatternEvidence(
                    pattern=GroupPattern.SUPPORT_CIRCLE,
                    confidence=min(1.0, support_count / 4.0),
                    involved_users=users,
                )
            )

    def _detect_celebration_wave(self, window: List[RecentMessage]) -> None:
        """检测庆祝浪潮：庆祝类关键词集中"""
        celebration_keywords = [
            "庆祝",
            "恭喜",
            "厉害",
            "牛逼",
            "太棒了",
            "赢了",
        ]
        celebration_count = sum(
            1
            for m in window
            if any(kw in m.text for kw in celebration_keywords)
        )
        if celebration_count >= 3 and len(window) >= 4:
            users = set(m.user_id for m in window)
            self._last_patterns.append(
                PatternEvidence(
                    pattern=GroupPattern.CELEBRATION_WAVE,
                    confidence=min(1.0, celebration_count / 4.0),
                    involved_users=users,
                )
            )

    def _detect_conflict_escalation(self, window: List[RecentMessage]) -> None:
        """检测冲突升级：负面情绪递增"""
        if len(window) < 4:
            return
        negative_words = ["滚", "傻", "蠢", "垃圾", "烦死", "闭嘴", "吵死了"]
        negative_count = sum(
            1 for m in window if any(kw in m.text for kw in negative_words)
        )
        if negative_count >= 3:
            users = set(m.user_id for m in window)
            self._last_patterns.append(
                PatternEvidence(
                    pattern=GroupPattern.CONFLICT_ESCALATION,
                    confidence=min(1.0, negative_count / 4.0),
                    involved_users=users,
                )
            )

    def _detect_quiet_reflection(self, window: List[RecentMessage]) -> None:
        """检测安静反思：消息稀疏，内容深沉"""
        if len(window) < 3:
            return
        # 消息间隔大，关键词包含反思性词语
        reflection_keywords = [
            "思考",
            "觉得",
            "也许",
            "可能",
            "原来",
            "反思",
            "感悟",
        ]
        reflection_count = sum(
            1
            for m in window
            if any(kw in m.text for kw in reflection_keywords)
        )
        time_span = window[-1].timestamp - window[0].timestamp if window else 0
        msg_rate = len(window) / max(1, time_span / 60.0)
        if reflection_count >= 2 and msg_rate < 2.0:  # 消息频率低
            users = set(m.user_id for m in window)
            self._last_patterns.append(
                PatternEvidence(
                    pattern=GroupPattern.QUIET_REFLECTION,
                    confidence=min(1.0, reflection_count / 3.0),
                    involved_users=users,
                )
            )

    # ────────────────── 工具方法 ──────────────────
    def _get_active_window(
        self, max_age_sec: float = 300.0
    ) -> List[RecentMessage]:
        cutoff = time.time() - max_age_sec
        return [m for m in self._recent_messages if m.timestamp > cutoff]


def get_group_pattern_detector(channel_id: str) -> GroupPatternDetector:
    if channel_id not in _detector_instances:
        _detector_instances[channel_id] = GroupPatternDetector(channel_id)
    return _detector_instances[channel_id]


def remove_group_pattern_detector(channel_id: str) -> None:
    _detector_instances.pop(channel_id, None)

import asyncio
import time
import math
import random
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from src.common.logger import get_logger

logger = get_logger("adaptive_threshold_v2")


@dataclass
class FactorContext:
    """当前上下文状态（用于因子联动计算）"""

    energy_level: float = 0.8
    affection_level: float = 0.0
    annoyance_level: float = 0.0
    trust_level: float = 0.0
    stress_level: float = 0.0
    trauma_level: float = 0.0
    is_night: bool = False
    conversation_density: float = 0.0
    group_tension: float = 0.0
    familiarity: float = 0.0
    dominance: float = 0.5
    mood_valence: float = 0.5
    mood_arousal: float = 0.5


class FactorInteractionNetwork:
    """因子交互网络 — 计算因子之间的联动效应"""

    @staticmethod
    def compute_interaction_matrix(ctx: FactorContext) -> Dict[str, float]:
        """根据上下文计算因子交互矩阵

        返回每个因子在该上下文中的动态权重
        """
        m = {}

        m["affection_weight"] = (
            0.5
            + (ctx.affection_level / 100.0) * 0.4
            - ctx.energy_level * 0.2
            + ctx.familiarity * 0.2
        )

        m["energy_weight"] = (
            0.6
            + (1.0 - ctx.energy_level) * 0.4
            + (0.3 if ctx.is_night else 0.0) * 0.3
        )

        m["annoyance_weight"] = (
            0.4 + (ctx.annoyance_level / 100.0) * 0.4 + ctx.stress_level * 0.2
        )

        m["trust_weight"] = (
            0.3 + (ctx.trust_level / 100.0) * 0.4 - ctx.group_tension * 0.2
        )

        m["complexity_tolerance"] = (
            0.5
            - ctx.mood_arousal * 0.3
            - ctx.trauma_level * 0.2
            + ctx.trust_level / 100.0 * 0.2
        )

        m["urgency_sensitivity"] = (
            0.4
            + (ctx.trust_level / 100.0) * 0.3
            - ctx.affection_level / 100.0 * 0.2
        )

        m["night_multiplier"] = 1.0 + (0.4 if ctx.is_night else 0.0)
        m["stress_multiplier"] = 1.0 + ctx.stress_level * 0.5
        m["fatigue_multiplier"] = 1.0 + (1.0 - ctx.energy_level) * 0.6

        for key in m:
            m[key] = max(0.0, min(2.0, m[key]))

        return m

    @staticmethod
    def interact_skip_factors(
        energy: float,
        affection: float,
        annoyance: float,
        trust: float,
        is_night: bool,
        familiarity: float,
        density: float,
        interaction_weights: Dict[str, float],
    ) -> float:
        """计算跳过因子（考虑因子联动）

        核心思想：
        - 能量低时：是否跳过取决于喜不喜欢这个人（为喜欢的人愿意花能量）
        - 能量高时：是否跳过主要看话题和情绪，不那么看喜不喜欢
        - 夜间模式：放大跳过倾向
        - 高密度对话：增加观察倾向
        """
        base_skip = 0.0

        energy_factor = (1.0 - energy) * interaction_weights.get(
            "energy_weight", 0.6
        )

        affection_modulated = (affection / 100.0) * (1.0 - energy * 0.5)
        affection_skip = (
            -affection_modulated
            * 0.5
            * interaction_weights.get("affection_weight", 0.5)
        )

        annoyance_skip = (
            (annoyance / 100.0)
            * 0.4
            * interaction_weights.get("annoyance_weight", 0.4)
        )

        familiarity_skip = familiarity * 0.1 * (1.0 - energy)

        night_skip = (
            0.25 * interaction_weights.get("night_multiplier", 1.0)
            if is_night
            else 0.0
        )

        density_skip = (
            density * 0.2 * interaction_weights.get("fatigue_multiplier", 1.0)
        )

        trust_forgiving = (trust / 100.0) * 0.15 * (1.0 - annoyance / 100.0)

        total = (
            base_skip
            + energy_factor * 0.4
            + affection_skip
            + annoyance_skip
            + familiarity_skip * 0.3
            + night_skip
            + density_skip
            + trust_forgiving * 0.2
        )

        return max(0.0, min(1.0, total))

    @staticmethod
    def interact_escalate_factors(
        energy: float,
        affection: float,
        annoyance: float,
        trust: float,
        complexity: float,
        controversy: float,
        emotion_intensity: float,
        group_tension: float,
        question_marks: float,
        has_forward: float,
        trauma: float,
        interaction_weights: Dict[str, float],
    ) -> float:
        """计算升级因子（考虑因子联动）

        核心思想：
        - 高能量时：更愿意处理复杂问题
        - 低能量+高好感：愿意为喜欢的人处理复杂问题
        - 低能量+低好感：不愿意为讨厌的人处理复杂问题
        - 高争议/争议话题：需要大模型处理
        - 有转发内容：需要大模型理解
        """
        base_escalate = 0.0

        energy_bonus = (
            energy * 0.3 * interaction_weights.get("energy_weight", 0.6)
        )

        affection_escalate = (
            (affection / 100.0) * energy * 0.25
        ) * interaction_weights.get("affection_weight", 0.5)

        complexity_escalate = (
            complexity
            * (1.0 - interaction_weights.get("complexity_tolerance", 0.5))
        ) * 0.4

        controversy_escalate = (
            controversy
            * 0.35
            * interaction_weights.get("complexity_tolerance", 0.5)
        )

        emotion_escalate = (
            emotion_intensity
            * 0.2
            * interaction_weights.get("stress_multiplier", 1.0)
        )

        tension_escalate = (
            group_tension
            * 0.25
            * interaction_weights.get("stress_multiplier", 1.0)
        )

        question_bonus = (
            question_marks
            * 0.15
            * interaction_weights.get("urgency_sensitivity", 0.4)
        )

        forward_bonus = has_forward * 0.25

        trauma_penalty = -trauma * 0.3

        annoyance_escalate = (
            -(annoyance / 100.0) * 0.2
        ) * interaction_weights.get("annoyance_weight", 0.4)

        total = (
            base_escalate
            + energy_bonus
            + affection_escalate
            + complexity_escalate
            + controversy_escalate
            + emotion_escalate
            + tension_escalate
            + question_bonus
            + forward_bonus
            + trauma_penalty
            + annoyance_escalate
        )

        return max(0.0, min(1.0, total))

    @staticmethod
    def compute_contextual_threshold(
        base_skip: float,
        base_escalate: float,
        energy: float,
        affection: float,
        annoyance: float,
        trust: float,
        is_night: bool,
        fatigue: float,
        streak_skip: int,
        streak_llm: int,
    ) -> Tuple[float, float]:
        """计算动态阈值（根据上下文调整）

        核心思想：
        - 连续跳过时：提高跳过阈值（防止过度跳过导致错过重要消息）
        - 连续调用LLM时：降低升级阈值（防止过度话痨）
        - 能量极低时：大幅提高跳过阈值
        - 夜间模式：提高跳过阈值
        """
        skip_threshold = base_skip
        escalate_threshold = base_escalate

        if streak_skip > 3:
            skip_threshold -= min(0.15, (streak_skip - 3) * 0.03)

        if streak_llm > 2:
            escalate_threshold += min(0.15, (streak_llm - 2) * 0.05)

        if energy < 0.3:
            skip_threshold -= 0.2
        elif energy < 0.5:
            skip_threshold -= 0.1

        if is_night:
            skip_threshold -= 0.15

        if fatigue > 0.7:
            skip_threshold -= 0.2
        elif fatigue > 0.5:
            skip_threshold -= 0.1

        if affection > 50:
            skip_threshold += 0.15
        elif affection < -20:
            skip_threshold -= 0.15

        if annoyance > 60:
            skip_threshold -= 0.2

        skip_threshold = max(0.3, min(0.85, skip_threshold))
        escalate_threshold = max(0.1, min(0.6, escalate_threshold))

        return skip_threshold, escalate_threshold


@dataclass
class LearnedBias:
    """根据历史学习到的偏置"""

    skip_bias: float = 0.0
    escalate_bias: float = 0.0
    context_biases: Dict[str, float] = field(default_factory=dict)
    version: int = 0
    last_update: float = 0.0


@dataclass
class DecisionRecord:
    """决策记录"""

    timestamp: float
    context_signature: str
    skip_score: float
    escalate_score: float
    final_level: int
    actual_llm_called: bool
    quality: float
    feedback: float


class AdaptiveThresholdLearnerV2:
    """自适应动态阈值学习器 v2 — 多层级智能决策系统

    核心机制：
    1. 多层级决策：skip / 小模型 / 大模型 三级决策
    2. 检测：1分钟内超10条消息 → 进入智能决策
    3. 自适应学习：从历史反馈学习什么时候该融入/跳过
    4. 小模型调用限制：每小时有限次数（保护token）
    5. 动态概率：根据对话状态自动调整

    决策流程：
    1. 检测刷屏（Burst Mode）
    2. 计算融入意愿（多维度因子）
    3. 自适应决策（跳过/融入/小模型协助）
    4. 学习反馈（根据结果调整参数）
    """

    BASE_SKIP_THRESHOLD = 0.60
    BASE_ESCALATE_THRESHOLD = 0.30

    # 小模型调用限制
    LLM_HOURLY_LIMIT = 10
    LLM_MIN_INTERVAL_SEC = 120

    # 刷屏检测参数
    BURST_WINDOW_SEC = 60
    BURST_THRESHOLD_LOW = 5
    BURST_THRESHOLD_HIGH = 15

    # 学习参数
    HISTORY_WINDOW = 200
    MIN_SAMPLES = 10
    LEARNING_RATE = 0.1

    # 决策权重（从数据中学习）
    DECISION_WEIGHTS = {
        "energy": 0.2,
        "affection": 0.15,
        "interest": 0.15,
        "burst_severity": 0.25,
        "momentum": 0.15,
        "quality_history": 0.1,
    }

    def __init__(self, stream_id: str = "default"):
        self.stream_id = stream_id
        self._bias = LearnedBias()
        self._history: deque = deque(maxlen=self.HISTORY_WINDOW)
        self._streak_skip = 0
        self._streak_llm = 0
        self._decision_count = 0
        self._network = FactorInteractionNetwork()
        # 行为状态
        self._unanswered_turns = 0
        self._consecutive_speaks = 0
        self._last_user_activity_ts = time.time()
        self._last_bot_speak_ts = time.time()
        self._silence_since_bot = 0.0
        # 自适应参数
        self._learned_silence_threshold = None
        self._learned_speak_threshold = None
        self._silence_effectiveness = 1.0
        self._speak_effectiveness = 1.0
        self._interaction_pattern_score = 0.5
        # 多层级决策状态
        self._burst_mode = False
        self._burst_start_time = 0.0
        self._burst_msg_count = 0
        self._融入意愿 = 0.5
        self._last_llm_call_time = 0.0
        self._llm_calls_this_hour = 0
        self._llm_hour_start = time.time()
        self._decision_outcomes: deque = deque(maxlen=50)
        self._learned_skip_probability = 0.5
        self._learned_join_probability = 0.5
        self._user_msg_timeline: deque = deque(maxlen=100)
        self._bot_reply_timeline: deque = deque(maxlen=100)
        self._momentum_factor = 1.0
        # 自适应偏好模板（从历史中学习）
        self._topic_preferences: Dict[str, float] = {}
        self._topic_history: deque = deque(maxlen=200)
        self._preference_update_count = 0
        self._last_preference_update = time.time()
        # 模型拒绝接管机制（算法严格模式）
        self._model_rejection_count = 0
        self._model_rejection_streak = 0
        self._last_model_rejection_time = 0.0
        self._strict_mode_active = False
        self._strict_mode_until = 0.0
        self._model_rejection_cooldown = 300.0
        # 无聊驱动状态（与emotion_driven_core联动）
        self._boredom_level = 0.0
        self._loneliness_level = 0.0
        self._social_desire_level = 0.5
        self._curiosity_level = 0.3
        self._boredom_accumulation_rate = 0.03
        self._loneliness_accumulation_rate = 0.15
        self._boredom_decay_on_interaction = 0.4
        self._last_boredom_update = time.time()
        # 沉默监控开关
        self._monitoring_start_delay = 240.0
        self._is_monitoring = False
        self._monitoring_start_time = 0.0
        # 状态到期询问机制
        self._state_expired = False
        self._state_expired_time = 0.0
        self._last_state_check = 0.0
        self._state_check_interval = 10.0
        # 思考值系统 - 与energy_manager.VitalityPoolManager联动
        # 思考值越低越想休息摸鱼
        self._energy_manager = None
        # 模型拒绝接管后的休息规划
        self._rest_until = 0.0
        self._rest_duration_base = 300.0
        self._wants_to_takeover = True
        self._takeover_offered = False
        self._takeover_offer_time = 0.0

    def compute_context(
        self, messages, social, energy, group, time_metrics
    ) -> FactorContext:
        """从各系统提取上下文状态"""
        ctx = FactorContext()

        ctx.energy_level = energy.get("energy_ratio", 0.8)
        ctx.affection_level = social.get("affection", 0.0)
        ctx.annoyance_level = social.get("annoyance_value", 0.0)
        ctx.trust_level = social.get("trust_value", 0.0)
        ctx.stress_level = energy.get("stress_accumulation", 0.0)
        ctx.trauma_level = energy.get("trauma_score", 0.0)
        ctx.mood_valence = energy.get("mood_valence", 0.5)
        ctx.mood_arousal = energy.get("mood_arousal", 0.5)
        ctx.conversation_density = (
            min(1.0, len(messages) / 20.0) if messages else 0.0
        )
        ctx.group_tension = group.get("atmosphere_tension", 0.0)
        ctx.familiarity = social.get("familiarity", 0.0)
        ctx.dominance = social.get("dominance", 0.5)

        try:
            _hour = time.localtime().tm_hour
            ctx.is_night = _hour >= 23 or _hour < 6
        except Exception:
            ctx.is_night = False

        return ctx

    def compute_factor_values(
        self, messages, ctx: FactorContext
    ) -> Dict[str, float]:
        """计算各因子值"""
        f = {}

        _msg_count = len(messages) if messages else 0
        f["msg_density"] = min(1.0, _msg_count / 20.0)

        if messages:
            texts = [
                getattr(m, "plain_text", "") or "" for m in messages[-10:]
            ]
            total_len = sum(len(t) for t in texts)
            f["msg_complexity"] = min(1.0, total_len / 500.0)
            f["msg_emotion"] = self._calc_emotion(texts)
            f["msg_urgency"] = self._calc_urgency(texts)
            f["msg_controversy"] = self._calc_controversy(texts)
            f["msg_has_forward"] = (
                1.0
                if any(getattr(m, "has_forward", False) for m in messages)
                else 0.0
            )
            f["msg_question"] = sum(1 for t in texts if "?" in t) / max(
                1, _msg_count
            )
        else:
            f["msg_complexity"] = 0.0
            f["msg_emotion"] = 0.0
            f["msg_urgency"] = 0.0
            f["msg_controversy"] = 0.0
            f["msg_has_forward"] = 0.0
            f["msg_question"] = 0.0

        f["energy_level"] = ctx.energy_level
        f["affection"] = ctx.affection_level / 100.0
        f["annoyance"] = ctx.annoyance_level / 100.0
        f["trust"] = ctx.trust_level / 100.0
        f["stress"] = ctx.stress_level
        f["trauma"] = ctx.trauma_level
        f["is_night"] = 1.0 if ctx.is_night else 0.0
        f["familiarity"] = ctx.familiarity
        f["group_tension"] = ctx.group_tension

        return f

    def _calc_emotion(self, texts: List[str]) -> float:
        _emotion_words = [
            "好",
            "喜欢",
            "开心",
            "棒",
            "哈哈",
            "呵",
            "滚",
            "烦",
            "讨厌",
            "气",
            "怒",
            "哭",
            "笑",
            "爱",
            "恨",
        ]
        score = 0.0
        for text in texts:
            lower = text.lower()
            for w in _emotion_words:
                if w in lower:
                    score += 0.15
        return min(1.0, score / max(1, len(texts)))

    def _calc_urgency(self, texts: List[str]) -> float:
        _urgency_keywords = [
            "快",
            "快说",
            "赶紧",
            "马上",
            "立刻",
            "急",
            "现在",
        ]
        score = 0.0
        for text in texts:
            for kw in _urgency_keywords:
                if kw in text:
                    score += 0.2
        return min(1.0, score)

    def _calc_controversy(self, texts: List[str]) -> float:
        _controversy_words = [
            "不对",
            "不是",
            "但是",
            "可是",
            "虽然",
            "其实",
            "应该",
            "不一定",
            "可能",
            "未必",
            "反对",
            "同意",
            "傻",
            "蠢",
        ]
        score = 0.0
        for text in texts:
            for w in _controversy_words:
                if w in text:
                    score += 0.2
        return min(1.0, score)

    def _detect_burst_mode(
        self, current_time: float
    ) -> Tuple[bool, int, float]:
        """检测刷屏模式

        返回: (is_burst, msg_count_in_window, burst_intensity)
        - is_burst: 是否在刷屏模式
        - msg_count_in_window: 窗口内的消息数
        - burst_intensity: 刷屏强度 0.0-1.0
        """
        window_start = current_time - self.BURST_WINDOW_SEC
        self._user_msg_timeline = deque(
            [t for t in self._user_msg_timeline if t >= window_start],
            maxlen=100,
        )
        msg_count = len(self._user_msg_timeline)

        if msg_count >= self.BURST_THRESHOLD_HIGH:
            self._burst_mode = True
            self._burst_start_time = current_time
            self._burst_msg_count = msg_count
        elif msg_count < self.BURST_THRESHOLD_LOW:
            self._burst_mode = False
            self._burst_msg_count = 0

        burst_intensity = min(1.0, msg_count / self.BURST_THRESHOLD_HIGH)
        return self._burst_mode, msg_count, burst_intensity

    def _compute_user_interest_scores(
        self,
        messages: List,
        social: Dict[str, float],
        energy: float,
    ) -> List[Dict[str, Any]]:
        """计算多用户消息的兴趣分数

        每个用户的消息都会被评估：
        1. 话题兴趣：消息内容是否吸引机器人
        2. 关系权重：与该用户的好感度/信任度
        3. 时效性：消息是否新鲜
        4. 紧急度：消息是否有紧迫性

        返回按兴趣分数排序的用户消息列表
        """
        if not messages:
            return []

        user_messages: Dict[str, Dict[str, Any]] = {}

        for msg in messages:
            user_id = getattr(msg, "user_id", "") or "unknown"
            if user_id == "bot":
                continue

            text = getattr(msg, "plain_text", "") or ""
            if not text:
                continue

            ts = getattr(msg, "timestamp", 0.0) or time.time()

            if user_id not in user_messages:
                user_messages[user_id] = {
                    "user_id": user_id,
                    "messages": [],
                    "total_text": "",
                    "latest_ts": ts,
                    "msg_count": 0,
                }

            user_messages[user_id]["messages"].append(text)
            user_messages[user_id]["total_text"] += text + " "
            user_messages[user_id]["latest_ts"] = max(
                user_messages[user_id]["latest_ts"], ts
            )
            user_messages[user_id]["msg_count"] += 1

        interest_scores = []
        now = time.time()

        for user_id, data in user_messages.items():
            text = data["total_text"].strip()

            topic_score = self._calc_topic_interest(text)
            relation_weight = self._calc_relation_weight(user_id, social)
            recency_score = self._calc_recency_score(data["latest_ts"], now)
            urgency_score = self._calc_urgency(text)
            emotion_score = self._calc_emotion([text])

            confidence = min(
                1.0,
                topic_score * 0.4
                + relation_weight * 0.2
                + recency_score * 0.2
                + urgency_score * 0.2,
            )

            interest_score = (
                topic_score * 0.35
                + relation_weight * 0.20
                + recency_score * 0.15
                + urgency_score * 0.15
                + emotion_score * 0.15
            )

            interest_scores.append(
                {
                    "user_id": user_id,
                    "interest_score": interest_score,
                    "confidence": confidence,
                    "topic_score": topic_score,
                    "relation_weight": relation_weight,
                    "recency_score": recency_score,
                    "urgency_score": urgency_score,
                    "msg_count": data["msg_count"],
                    "text": text[:100],
                }
            )

        interest_scores.sort(key=lambda x: x["interest_score"], reverse=True)

        for _i, score in enumerate(interest_scores):
            logger.debug(
                f"[自适应v2] 用户{
                    score['user_id']} "
                f"兴趣={
                    score['interest_score']:.3f}(自信={
                    score['confidence']:.2f}) "
                f"话题={
                    score['topic_score']:.2f} 关系={
                        score['relation_weight']:.2f} "
                f"时效={
                    score['recency_score']:.2f} 紧急={
                    score['urgency_score']:.2f}"
            )

        return interest_scores

    def _calc_topic_interest(self, text: str) -> float:
        """计算话题兴趣分数（基于自适应偏好模板）

        不使用硬编码的话题关键词，而是：
        1. 从历史交互中学习机器人对什么话题感兴趣
        2. 根据回复质量更新偏好模板
        3. 新话题有探索奖励
        """
        if not text:
            return 0.5

        if not self._topic_preferences:
            self._initialize_topic_preferences(text)

        words = self._extract_keywords(text)
        topic_scores = {}

        for word in words:
            word_lower = word.lower()
            if word_lower in self._topic_preferences:
                topic_scores[word_lower] = self._topic_preferences[word_lower]
            else:
                topic_scores[word_lower] = 0.5

        if not topic_scores:
            return 0.5

        max_word = max(topic_scores.items(), key=lambda x: x[1])
        base_score = max_word[1]

        is_new_topic = all(w not in self._topic_preferences for w in words)
        if is_new_topic:
            exploration_bonus = 0.1 * min(1.0, len(words) / 5.0)
            base_score = min(0.9, base_score + exploration_bonus)

        decay_factor = (
            0.95 ** (time.time() - self._last_preference_update)
            if self._last_preference_update > 0
            else 1.0
        )
        base_score *= decay_factor

        return min(1.0, max(0.1, base_score))

    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词

        使用简单的中分词逻辑：
        - 长度 >= 2 的连续字符
        - 去除常见停用词
        """
        if not text:
            return []

        stopwords = {
            "的",
            "了",
            "在",
            "是",
            "我",
            "有",
            "和",
            "就",
            "不",
            "人",
            "都",
            "一",
            "一个",
            "上",
            "也",
            "很",
            "到",
            "说",
            "要",
            "去",
            "你",
            "会",
            "着",
            "没有",
            "看",
            "好",
            "这",
            "那",
            "吗",
            "吧",
            "呢",
        }

        words = []
        current_word = []
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                current_word.append(char)
            else:
                if current_word:
                    word = "".join(current_word)
                    if len(word) >= 2 and word not in stopwords:
                        words.append(word)
                    current_word = []
                if char.isalnum() and len(char) > 1:
                    words.append(char.lower())

        if current_word:
            word = "".join(current_word)
            if len(word) >= 2 and word not in stopwords:
                words.append(word)

        return list(set(words))

    def _initialize_topic_preferences(self, seed_text: str):
        """初始化话题偏好模板

        从种子文本中提取初始话题
        """
        words = self._extract_keywords(seed_text)
        for word in words[:10]:
            self._topic_preferences[word] = 0.5 + (hash(word) % 100) / 200.0

        if not self._topic_preferences:
            default_topics = ["聊天", "游戏", "技术", "问题", "学习"]
            for topic in default_topics:
                self._topic_preferences[topic] = 0.5

        logger.debug(
            f"[自适应v2] 初始化偏好模板: {len(self._topic_preferences)}个话题"
        )

    def _update_topic_preferences(
        self,
        topic_word: str,
        quality: float,
        was_engaging: bool,
    ):
        """更新话题偏好模板

        根据回复质量更新偏好：
        - 质量高 + 参与度高 → 增加偏好
        - 质量低 + 参与度低 → 减少偏好
        """
        if not topic_word:
            return

        topic_word = topic_word.lower()

        if topic_word not in self._topic_preferences:
            self._topic_preferences[topic_word] = 0.5

        current_pref = self._topic_preferences[topic_word]

        if quality > 0.6 and was_engaging:
            delta = 0.05 + (quality - 0.6) * 0.1
            new_pref = current_pref + delta
        elif quality < 0.3 or not was_engaging:
            delta = 0.03 + (0.4 - quality) * 0.05
            new_pref = current_pref - delta
        else:
            new_pref = current_pref

        self._topic_preferences[topic_word] = max(0.1, min(1.0, new_pref))
        self._preference_update_count += 1
        self._last_preference_update = time.time()

        if self._preference_update_count % 20 == 0:
            self._decay_topic_preferences()

        logger.debug(
            f"[自适应v2] 更新偏好: '{topic_word}' "
            f"{current_pref:.3f}→{self._topic_preferences[topic_word]:.3f} "
            f"(质量={quality:.2f} 参与={was_engaging})"
        )

    def _decay_topic_preferences(self):
        """衰减话题偏好

        长时间未触及的话题会逐渐衰减，保持模板新鲜度
        """
        if not self._topic_preferences:
            return

        decay_rate = 0.98
        words_to_remove = []

        for word, pref in self._topic_preferences.items():
            decayed = pref * decay_rate
            if decayed < 0.15:
                words_to_remove.append(word)
            else:
                self._topic_preferences[word] = decayed

        for word in words_to_remove:
            del self._topic_preferences[word]

        logger.debug(
            f"[自适应v2] 偏好衰减: {len(words_to_remove)}个话题移除, "
            f"剩余{len(self._topic_preferences)}个话题"
        )

    def _get_top_interests(self) -> List[str]:
        """获取当前最感兴趣的话题

        用于调试和日志
        """
        if not self._topic_preferences:
            return []
        sorted_topics = sorted(
            self._topic_preferences.items(), key=lambda x: x[1], reverse=True
        )
        return [f"{w}({s:.2f})" for w, s in sorted_topics[:5]]

    def _calc_relation_weight(
        self, user_id: str, social: Dict[str, float]
    ) -> float:
        """计算与用户的关系权重

        - 好感高 → 更容易被吸引
        - 信任高 → 更容易回应
        - 熟悉度高 → 更容易融入
        """
        affection = social.get("affection", 0.0) / 100.0
        trust = social.get("trust_value", 0.0) / 100.0
        familiarity = social.get("familiarity", 0.0)

        relation = affection * 0.4 + trust * 0.3 + familiarity * 0.3
        return min(1.0, relation)

    def _calc_recency_score(self, msg_ts: float, now: float) -> float:
        """计算时效性分数

        越新鲜的消息分数越高
        """
        age_sec = now - msg_ts
        if age_sec < 10:
            return 1.0
        elif age_sec < 60:
            return 0.9
        elif age_sec < 300:
            return 0.7
        elif age_sec < 600:
            return 0.5
        else:
            return max(0.1, 0.3 - age_sec / 3600 * 0.2)

    def _should_fallback_to_llm(
        self,
        top_interest_score: float,
        confidence: float,
        second_score: float,
        burst_intensity: float,
    ) -> bool:
        """判断是否应该fallback到小模型

        条件：
        1. 最高兴趣分数不够高（< 0.5）
        2. 置信度低（< 0.4）
        3. 前两名分数接近（差异 < 0.1）→ 难以抉择
        4. 刷屏严重 + 置信度中等 → 需要小模型判断
        """
        if confidence < 0.35:
            logger.debug(
                f"[自适应v2] 置信度不足({confidence:.2f}<0.35)，fallback小模型"
            )
            return True

        if top_interest_score < 0.4:
            logger.debug(
                f"[自适应v2] 兴趣分数不足({
                    top_interest_score:.2f}<0.4)，fallback小模型"
            )
            return True

        if second_score > 0 and (top_interest_score - second_score) < 0.08:
            logger.debug(
                f"[自适应v2] 前两名分数接近({top_interest_score:.2f}-{second_score:.2f}<0.08)，"
                f"难以抉择，fallback小模型"
            )
            return True

        if burst_intensity > 0.7 and confidence < 0.6:
            logger.debug(
                f"[自适应v2] 刷屏严重({burst_intensity:.2f}) + 置信度不足({confidence:.2f}<0.6)，"
                f"fallback小模型"
            )
            return True

        return False

    def _compute_融入意愿(
        self,
        energy: float,
        affection: float,
        trust: float,
        burst_intensity: float,
        msg_emotion: float,
        desire_level: float,
    ) -> float:
        """计算融入意愿（0.0-1.0）

        综合多维度因子计算是否想融入对话
        - 能量高 + 好感高 → 想融入
        - 刷屏严重 → 降低融入意愿
        - 情绪激动 → 需要判断
        - 欲望高 → 增加融入意愿
        """
        energy_factor = energy * 0.25
        affection_factor = affection * 0.2
        trust_factor = trust * 0.15
        desire_factor = desire_level / 10.0 * 0.2

        burst_penalty = burst_intensity * 0.25

        emotion_factor = msg_emotion * 0.1

        融入意愿 = (
            energy_factor
            + affection_factor
            + trust_factor
            + desire_factor
            + emotion_factor
            - burst_penalty
        )

        融入意愿 = max(0.0, min(1.0, 融入意愿))

        logger.debug(
            f"[自适应v2] 融入意愿={
                融入意愿:.3f} "
            f"(能量={
                energy:.2f} 好感={
                affection:.2f} 信任={
                    trust:.2f} "
            f"欲望={
                desire_level:.1f} 刷屏={
                burst_intensity:.2f} 情绪={
                msg_emotion:.2f})"
        )

        return 融入意愿

    def _adaptive_decide(
        self,
        融入意愿: float,
        burst_intensity: float,
        current_time: float,
    ) -> str:
        """自适应决策：skip / 融入 / 小模型协助

        根据融入意愿和刷屏强度做决策
        - 融入意愿高 + 刷屏不严重 → 融入
        - 融入意愿低 + 刷屏严重 → skip
        - 融入意愿中等 + 刷屏中等 → 小模型协助
        - 融入意愿高 + 刷屏严重 → 小模型协助
        """
        if self._llm_calls_this_hour >= self.LLM_HOURLY_LIMIT:
            logger.debug(
                f"[自适应v2] 小模型调用已达上限({self.LLM_HOURLY_LIMIT}/小时)，强制决策"
            )
            if 融入意愿 >= 0.6:
                return "join"
            else:
                return "skip"

        time_since_last_llm = current_time - self._last_llm_call_time
        if (
            time_since_last_llm < self.LLM_MIN_INTERVAL_SEC
            and self._llm_calls_this_hour > 0
        ):
            logger.debug(
                f"[自适应v2] 小模型调用间隔不足({
                    time_since_last_llm:.0f}s<{
                    self.LLM_MIN_INTERVAL_SEC}s)"
            )
            if 融入意愿 >= 0.7:
                return "join"
            elif 融入意愿 <= 0.3:
                return "skip"
            else:
                return "skip"

        if 融入意愿 >= 0.7 and burst_intensity <= 0.5:
            return "join"
        elif 融入意愿 <= 0.3:
            return "skip"
        elif 融入意愿 >= 0.5 and burst_intensity >= 0.7:
            return "llm_assist"
        elif 融入意愿 >= 0.6 and burst_intensity <= 0.3:
            return "join"
        elif 融入意愿 <= 0.5 and burst_intensity <= 0.4:
            return "skip"
        else:
            return "llm_assist"

    def _learn_from_decision(
        self,
        decision: str,
        actual_outcome: str,
        quality: float,
    ):
        """从决策结果中学习

        decision: 原始决策（skip/join/llm_assist）
        actual_outcome: 实际结果（success/fail/neutral）
        quality: 回复质量 0.0-1.0
        """
        outcome_record = {
            "decision": decision,
            "outcome": actual_outcome,
            "quality": quality,
            "timestamp": time.time(),
        }
        self._decision_outcomes.append(outcome_record)

        if len(self._decision_outcomes) < 5:
            return

        recent = list(self._decision_outcomes)[-20:]

        skip_quality = [
            r["quality"] for r in recent if r["decision"] == "skip"
        ]
        join_quality = [
            r["quality"] for r in recent if r["decision"] == "join"
        ]
        llm_quality = [r["quality"] for r in recent if r["decision"] == "llm_assist"]

        if skip_quality:
            avg_skip = sum(skip_quality) / len(skip_quality)
            if avg_skip < 0.4:
                self._learned_skip_probability = max(
                    0.2, self._learned_skip_probability - 0.05
                )
            elif avg_skip > 0.6:
                self._learned_skip_probability = min(
                    0.8, self._learned_skip_probability + 0.05
                )

        if join_quality:
            avg_join = sum(join_quality) / len(join_quality)
            if avg_join < 0.4:
                self._learned_join_probability = max(
                    0.2, self._learned_join_probability - 0.05
                )
            elif avg_join > 0.6:
                self._learned_join_probability = min(
                    0.8, self._learned_join_probability + 0.05
                )

        if quality > 0.6:
            if decision == "skip":
                self._learned_skip_probability += 0.02
            elif decision == "join":
                self._learned_join_probability += 0.02
        elif quality < 0.3:
            if decision == "skip":
                self._learned_skip_probability -= 0.02
            elif decision == "join":
                self._learned_join_probability -= 0.02

        self._learned_skip_probability = max(
            0.1, min(0.9, self._learned_skip_probability)
        )
        self._learned_join_probability = max(
            0.1, min(0.9, self._learned_join_probability)
        )

        logger.debug(
            f"[自适应v2] 学习结果: skip概率={self._learned_skip_probability:.2f} "
            f"join概率={self._learned_join_probability:.2f} "
            f"llm质量={((sum(llm_quality) / len(llm_quality)) if llm_quality else 0.0):.2f} "
            f"(最近{len(recent)}次决策)"
        )

    def record_outcome(
        self,
        level: int,
        skip_score: float,
        escalate_score: float,
        actual_llm_called: bool,
        quality: float,
        group: Dict[str, float],
        time_metrics: Dict[str, float],
        history_metrics: Dict[str, Any],
        pinged: bool = False,
        messages: Optional[List] = None,
        social: Optional[Dict] = None,
        energy: Optional[Dict] = None,
    ) -> Tuple[int, float, float, bool]:
        """决定调用等级

        返回 (level, skip_score, escalate_score, tentative_speak)

        level:
        - 0: skip（完全沉默）
        - 1: 小模型直接回复
        - 2: 大模型深度分析

        tentative_speak:
        - True: 试探性发牢骚（长时间沉默后的一种轻量试探）
        - False: 正常回复决策

        核心算法：
        1. 提取上下文状态
        2. 计算因子交互矩阵
        3. 通过因子联动计算skip_score和escalate_score
        4. 计算动态阈值
        5. 根据net_score和阈值决定level
        6. 判断是否适合试探性发牢骚
        """
        self._decision_count += 1

        desire_level = (
            float(history_metrics.get("desire_level", 5.0))
            if history_metrics
            else 5.0
        )

        if pinged and desire_level >= 7:
            return 1, 0.0, 0.0, False

        ctx = self.compute_context(
            messages, social, energy, group, time_metrics
        )
        f = self.compute_factor_values(messages, ctx)

        interaction_weights = self._network.compute_interaction_matrix(ctx)

        skip_score = self._network.interact_skip_factors(
            energy=f["energy_level"],
            affection=f["affection"] * 100.0,
            annoyance=f["annoyance"] * 100.0,
            trust=f["trust"] * 100.0,
            is_night=bool(f["is_night"]),
            familiarity=f["familiarity"],
            density=f["msg_density"],
            interaction_weights=interaction_weights,
        )

        escalate_score = self._network.interact_escalate_factors(
            energy=f["energy_level"],
            affection=f["affection"] * 100.0,
            annoyance=f["annoyance"] * 100.0,
            trust=f["trust"] * 100.0,
            complexity=f["msg_complexity"],
            controversy=f["msg_controversy"],
            emotion_intensity=f["msg_emotion"],
            group_tension=f["group_tension"],
            question_marks=f["msg_question"],
            has_forward=f["msg_has_forward"],
            trauma=f["trauma"],
            interaction_weights=interaction_weights,
        )

        skip_threshold, escalate_threshold = (
            self._network.compute_contextual_threshold(
                base_skip=self.BASE_SKIP_THRESHOLD,
                base_escalate=self.BASE_ESCALATE_THRESHOLD,
                energy=f["energy_level"],
                affection=f["affection"] * 100.0,
                annoyance=f["annoyance"] * 100.0,
                trust=f["trust"] * 100.0,
                is_night=bool(f["is_night"]),
                fatigue=1.0 - f["energy_level"],
                streak_skip=self._streak_skip,
                streak_llm=self._streak_llm,
            )
        )

        silence_mod, speak_mod, chatter_penalty = (
            self.compute_behavior_adjustment(time.time())
        )
        skip_threshold = min(0.95, skip_threshold + silence_mod)
        escalate_threshold = max(0.05, escalate_threshold - speak_mod)

        desire_factor = (desire_level - 5.0) / 10.0
        skip_score -= desire_factor * 0.5
        escalate_score += desire_factor * 0.3

        if desire_level >= 8:
            skip_score -= 0.3
        elif desire_level <= 3:
            skip_score += 0.4

        net_score = skip_score - escalate_score

        if skip_score >= skip_threshold:
            level = 0
        elif escalate_score >= escalate_threshold:
            level = 2
        else:
            level = 1

        logger.debug(
            f"[自适应v2] skip={
                skip_score:.3f}(>{
                skip_threshold:.2f}) "
            f"esc={
                escalate_score:.3f}(>{
                    escalate_threshold:.2f}) "
            f"net={
                net_score:.3f} → level={level} "
            f"energy={
                f['energy_level']:.2f} aff={
                f['affection']:.2f} desire={
                desire_level:.1f} "
            f"沉默调整={
                silence_mod:.3f} 说话调整={
                speak_mod:.3f} 话痨={
                chatter_penalty:.3f}"
        )

        tentative_speak = self._should_tentative_speak(
            level=level,
            skip_score=skip_score,
            social=f,
            energy=f["energy_level"],
            since_bot_hours=(
                (time.time() - self._last_bot_speak_ts) / 3600.0
                if self._last_bot_speak_ts > 0
                else 999.0
            ),
        )

        return level, skip_score, escalate_score, tentative_speak

    def _should_tentative_speak(
        self,
        level: int,
        skip_score: float,
        social: Dict[str, float],
        energy: float,
        since_bot_hours: float,
    ) -> bool:
        """判断是否应该试探性发牢骚

        试探性发牢骚 = 长时间沉默后的轻量试探，不是正式发言

        条件：
        1. 当前决策是skip（level=0）
        2. 沉默时间够长（since_bot_hours > 学习到的阈值）
        3. 不是害羞状态（能量+社交值都足够）
        4. 沉默有效性够高（之前的沉默策略是有效的）
        5. 学到的试探阈值允许
        6. 有足够的孤独感或想社交的冲动

        害羞状态判断：
        - 低能量 + 低好感度 → 害羞，不发牢骚
        - 低能量 + 高好感度 → 可能发一点
        - 高能量 + 低好感度 → 不想发
        - 高能量 + 高好感度 → 可以发牢骚
        """
        if level != 0:
            return False

        if self._learned_silence_threshold is None:
            return False

        silence_threshold_hours = max(
            1.0, (self._learned_silence_threshold - 0.5) * 48.0
        )
        if since_bot_hours < silence_threshold_hours:
            return False

        if self._silence_effectiveness < 0.6:
            return False

        mood_valence = social.get("mood_valence", 0.5)
        mood_arousal = social.get("mood_arousal", 0.5)
        stress = social.get("stress", 0.0)

        social_states = self._compute_shyness(
            social=social,
            energy=energy,
            mood_valence=mood_valence,
            mood_arousal=mood_arousal,
            stress=stress,
        )

        shyness = social_states["shyness"]
        loneliness = social_states["loneliness"]
        fatigue = social_states["fatigue"]
        social_fear = social_states["social_fear"]

        if shyness > 0.6:
            return False

        if fatigue > 0.5:
            return False

        social_impulse = loneliness * 2.0 - social_fear

        if shyness < 0.3 and since_bot_hours > silence_threshold_hours * 1.5:
            if social_impulse > 0.3:
                return True

        if self._speak_effectiveness > 0.7 and shyness < 0.5:
            if social_impulse > 0.1:
                return True

        return False

    def _record_outcome_log(
        self,
        level: int,
        skip_score: float,
        escalate_score: float,
        actual_llm_called: bool,
        quality: float,
        feedback: float,
    ):
        """记录决策结果用于学习"""
        self._decision_count += 1

        record = DecisionRecord(
            timestamp=time.time(),
            context_signature=f"{self._streak_skip}_{self._streak_llm}",
            skip_score=skip_score,
            escalate_score=escalate_score,
            final_level=level,
            actual_llm_called=actual_llm_called,
            quality=quality,
            feedback=feedback,
        )
        self._history.append(record)

        if level == 0:
            self._streak_skip += 1
            self._streak_llm = 0
        else:
            self._streak_llm += 1
            self._streak_skip = 0

        if len(self._history) >= self.MIN_SAMPLES:
            self._learn_from_feedback()

    def _update_bias(self):
        """基于历史反馈更新偏置"""
        try:
            _success = [r for r in self._history if r.quality > 0.6]
            _fail = [r for r in self._history if r.quality <= 0.4]

            if not _success or not _fail:
                return

            _success_avg_skip = sum(r.skip_score for r in _success) / max(1, len(
                _success
            ))
            _fail_avg_skip = sum(r.skip_score for r in _fail) / max(1, len(_fail))

            _skip_delta = (
                _fail_avg_skip - _success_avg_skip
            ) * self.LEARNING_RATE

            self._bias.skip_bias += _skip_delta
            self._bias.skip_bias = max(-0.2, min(0.2, self._bias.skip_bias))
            self._bias.version += 1
            self._bias.last_update = time.time()

            logger.debug(
                f"[自适应v2] 偏置更新 v={self._bias.version} "
                f"skip_bias={self._bias.skip_bias:.4f} delta={_skip_delta:.4f}"
            )
        except Exception as e:
            logger.debug(f"[自适应v2] 偏置更新异常: {e}")

    def update_behavior_state(
        self,
        unanswered_turns: int = 0,
        consecutive_speaks: int = 0,
        user_activity_ts: float = 0.0,
        bot_speak_ts: float = 0.0,
    ):
        """更新行为状态"""
        self._unanswered_turns = unanswered_turns
        self._consecutive_speaks = consecutive_speaks
        if user_activity_ts > 0:
            self._last_user_activity_ts = user_activity_ts
        if bot_speak_ts > 0:
            self._last_bot_speak_ts = bot_speak_ts
            self._silence_since_bot = 0.0

    def _analyze_silence_patterns(self) -> Tuple[float, float]:
        """分析历史沉默模式，返回(沉默倾向调整,说话倾向调整)

        数据驱动学习：
        - 如果沉默后用户有反馈 → 沉默是有效的，增加沉默倾向
        - 如果沉默后用户没反馈 → 沉默是无效的，减少沉默倾向
        - 如果沉默太久用户都跑了 → 减少沉默倾向
        """
        if len(self._history) < 5:
            return 0.0, 0.0

        recent = list(self._history)[-20:]
        silence_after_skip = []
        for i, record in enumerate(recent):
            if record.final_level == 0 and i + 1 < len(recent):
                next_record = recent[i + 1]
                silence_after_skip.append(
                    {
                        "skip_score": record.skip_score,
                        "next_user_active": next_record.actual_llm_called,
                        "quality": record.quality,
                    }
                )

        if not silence_after_skip:
            return 0.0, 0.0

        effective_silence = sum(
            1
            for s in silence_after_skip
            if s["next_user_active"] and s["quality"] > 0.5
        )

        total = len(silence_after_skip)
        effectiveness_ratio = effective_silence / max(1, total)

        if effectiveness_ratio > 0.6:
            return 0.08, -0.05
        elif effectiveness_ratio < 0.3:
            return -0.1, 0.08
        return 0.0, 0.0

    def _analyze_speak_patterns(self) -> Tuple[float, float]:
        """分析历史说话模式，返回(沉默倾向调整,说话倾向调整)

        数据驱动学习：
        - 如果说话后用户回应 → 说话是受欢迎的，增加说话倾向
        - 如果说话后用户沉默 → 说话是打扰的，减少说话倾向
        """
        if len(self._history) < 5:
            return 0.0, 0.0

        recent = list(self._history)[-20:]
        speak_records = [
            r for r in recent if r.final_level > 0 and r.actual_llm_called
        ]

        if not speak_records:
            return 0.0, 0.0

        welcomed = sum(1 for s in speak_records if s.quality > 0.6)

        total = len(speak_records)
        welcome_ratio = welcomed / max(1, total)

        if welcome_ratio > 0.7:
            return -0.05, 0.08
        elif welcome_ratio < 0.4:
            return 0.08, -0.1
        return 0.0, 0.0

    def _analyze_interaction_pattern(self) -> float:
        """分析群聊互动模式

        数据驱动学习：
        - 高互动群：用户经常主动说话，bot应该更活跃
        - 低互动群：用户沉默，bot应该更沉默
        - 混合群：根据当前状态动态调整
        """
        if len(self._history) < 10:
            return 0.5

        recent = list(self._history)[-30:]
        user_active_count = sum(1 for r in recent if r.actual_llm_called)
        total = len(recent)

        activity_ratio = user_active_count / max(1, total)

        self._interaction_pattern_score = (
            self._interaction_pattern_score * 0.9 + activity_ratio * 0.1
        )

        if self._interaction_pattern_score > 0.7:
            return 0.1
        elif self._interaction_pattern_score < 0.3:
            return -0.1
        return 0.0

    def _learn_from_feedback(self):
        """从反馈中学习，动态调整参数"""
        if len(self._history) < self.MIN_SAMPLES:
            return

        silence_adj, speak_adj = self._analyze_silence_patterns()
        interaction_adj = self._analyze_interaction_pattern()

        self._silence_effectiveness = max(
            0.3, min(1.5, self._silence_effectiveness + silence_adj)
        )
        self._speak_effectiveness = max(
            0.3, min(1.5, self._speak_effectiveness + speak_adj)
        )

        base_threshold = self.BASE_SKIP_THRESHOLD
        interaction_mod = interaction_adj
        effectiveness_mod = (1.0 - self._silence_effectiveness) * 0.1

        self._learned_silence_threshold = max(
            0.3,
            min(0.85, base_threshold + interaction_mod + effectiveness_mod),
        )
        self._learned_speak_threshold = max(
            0.1, min(0.5, self.BASE_ESCALATE_THRESHOLD - speak_adj * 0.5)
        )

        self._bias.skip_bias = max(
            -0.2, min(0.2, self._bias.skip_bias + silence_adj * 0.5)
        )
        self._bias.version += 1
        self._bias.last_update = time.time()

        logger.debug(
            f"[自适应v2-L] 沉默有效性={
                self._silence_effectiveness:.3f} "
            f"说话有效性={
                self._speak_effectiveness:.3f} "
            f"互动模式={
                self._interaction_pattern_score:.3f} "
            f"学习阈值=沉默{
                self._learned_silence_threshold:.2f} 说话{
                self._learned_speak_threshold:.2f}"
        )

    def compute_behavior_adjustment(
        self, current_ts: float
    ) -> Tuple[float, float, float]:
        """计算行为调整因子（基于数据驱动学习）

        返回 (沉默倾向调整, 说话倾向调整, 话痨惩罚)

        根据历史学习结果动态计算
        """
        silence_since_bot_hours = 0.0
        if self._last_bot_speak_ts > 0:
            silence_since_bot_hours = (
                current_ts - self._last_bot_speak_ts
            ) / 3600.0

        silence_hours_factor = min(1.0, silence_since_bot_hours / 48.0)

        silence_mod = 0.0
        if self._learned_silence_threshold is not None:
            base_mod = (
                self._learned_silence_threshold - self.BASE_SKIP_THRESHOLD
            )
            silence_mod = (
                base_mod * self._silence_effectiveness * silence_hours_factor
            )
        else:
            silence_mod = silence_hours_factor * 0.1

        speak_mod = 0.0
        if self._learned_speak_threshold is not None:
            base_mod = (
                self.BASE_ESCALATE_THRESHOLD - self._learned_speak_threshold
            )
            speak_mod = base_mod * self._speak_effectiveness
        else:
            speak_mod = 0.0

        chatter_penalty = 0.0
        if self._consecutive_speaks > 2:
            chatter_penalty = min(
                0.25,
                (self._consecutive_speaks - 2)
                * 0.03
                * (2.0 - self._speak_effectiveness),
            )

        unanswered_mod = 0.0
        if self._unanswered_turns > 0:
            unanswered_mod = min(
                0.15,
                self._unanswered_turns
                * 0.03
                * (2.0 - self._speak_effectiveness),
            )

        interaction_mod = 0.0
        if self._interaction_pattern_score < 0.3:
            interaction_mod = -0.1
        elif self._interaction_pattern_score > 0.7:
            interaction_mod = 0.1

        total_silence_mod = silence_mod + unanswered_mod + interaction_mod
        total_speak_mod = speak_mod - chatter_penalty

        return total_silence_mod, total_speak_mod, chatter_penalty

    def on_model_rejection(self, current_time: float) -> None:
        """当模型拒绝回复时调用

        记录模型拒绝事件，连续拒绝会触发算法严格模式
        """
        self._model_rejection_count += 1
        self._model_rejection_streak += 1
        self._last_model_rejection_time = current_time

        if self._model_rejection_streak >= 2:
            self._strict_mode_active = True
            self._strict_mode_until = (
                current_time
                + self._model_rejection_cooldown * self._model_rejection_streak
            )
            logger.info(
                f"[自适应v2] 模型拒绝第{self._model_rejection_streak}次，"
                f"激活算法严格模式至{self._strict_mode_until - current_time:.0f}秒后"
            )
        else:
            logger.debug(
                f"[自适应v2] 模型拒绝 streak={self._model_rejection_streak} "
                f"累计={self._model_rejection_count}"
            )

    def on_model_accepted(self, current_time: float) -> None:
        """当模型接受回复任务时调用

        重置模型拒绝连续计数
        """
        if self._model_rejection_streak > 0:
            logger.debug(
                f"[自适应v2] 模型接受回复，重置拒绝streak "
                f"(原={self._model_rejection_streak})"
            )
        self._model_rejection_streak = 0
        if (
            self._strict_mode_active
            and current_time >= self._strict_mode_until
        ):
            self._strict_mode_active = False

    def is_strict_mode(self, current_time: float) -> bool:
        """检查是否在算法严格模式

        严格模式下：
        - 跳过阈值提高
        - 需要更高的融入意愿才能join
        - 刷屏检测更敏感
        """
        if self._strict_mode_active and current_time < self._strict_mode_until:
            return True
        if self._strict_mode_active:
            self._strict_mode_active = False
        return False

    def get_strict_mode_info(self) -> Dict[str, Any]:
        """获取严格模式状态信息"""
        return {
            "strict_mode_active": self._strict_mode_active,
            "model_rejection_count": self._model_rejection_count,
            "model_rejection_streak": self._model_rejection_streak,
            "time_until_strict_mode_end": (
                max(0.0, self._strict_mode_until - time.time())
                if self._strict_mode_active
                else 0.0
            ),
        }

    def update_boredom_state(
        self,
        silence_seconds: float,
        unanswered_count: int = 0,
        mood: float = 0.5,
        energy: float = 1.0,
    ) -> None:
        """更新无聊驱动状态

        与emotion_driven_core的boredom累积逻辑对齐
        - 无聊感随沉默时间累积
        - 孤独感随沉默时间和未回复次数累积
        - 心情和精力影响累积速度
        """
        now = time.time()
        silence_min = silence_seconds / 60.0

        if (
            silence_seconds >= self._monitoring_start_delay
            and not self._is_monitoring
        ):
            self._is_monitoring = True
            self._monitoring_start_time = now

        if self._is_monitoring:
            target_boredom = min(
                1.0, silence_min * self._boredom_accumulation_rate
            )
            if mood < 0.3:
                target_boredom *= 1.3
            if energy < 0.3:
                target_boredom *= 0.7
            self._boredom_level = max(
                self._boredom_level * 0.7, target_boredom
            )

            target_loneliness = min(
                1.0,
                max(0, unanswered_count) * self._loneliness_accumulation_rate
                + max(0.0, silence_min - 3.0) * 0.04,
            )
            self._loneliness_level = max(
                self._loneliness_level * 0.75, target_loneliness
            )
        else:
            self._boredom_level *= 0.95
            self._loneliness_level *= 0.95

        base_desire = 0.3
        desire_from_boredom = self._boredom_level * 0.3
        desire_from_loneliness = self._loneliness_level * 0.25
        desire_from_mood = mood * 0.15
        desire_from_curiosity = self._curiosity_level * 0.1
        _raw_desire = (
            base_desire
            + desire_from_boredom
            + desire_from_loneliness
            + desire_from_mood
            + desire_from_curiosity
        )
        self._social_desire_level = min(1.0, max(0.0, _raw_desire))

        self._last_boredom_update = now

        logger.debug(
            f"[自适应v2] 无聊状态: boredom={self._boredom_level:.3f} "
            f"loneliness={self._loneliness_level:.3f} "
            f"social_desire={self._social_desire_level:.3f}"
        )

    def reset_boredom_on_interaction(self) -> None:
        """用户交互时重置无聊状态

        有人说话时：
        - 无聊感降低
        - 孤独感清零
        - 社交欲望提升
        """
        self._boredom_level = max(
            0.0, self._boredom_level - self._boredom_decay_on_interaction
        )
        self._loneliness_level = 0.0
        self._social_desire_level = min(1.0, self._social_desire_level + 0.1)
        self._is_monitoring = False
        self._monitoring_start_time = 0.0
        logger.debug(
            f"[自适应v2] 交互重置无聊: boredom={self._boredom_level:.3f} "
            f"loneliness={self._loneliness_level:.3f}"
        )

    def get_boredom_driven_willingness(self, energy: float = 1.0) -> float:
        """获取无聊驱动的主动发言意愿

        基于emotion_driven_core的proactive_willingness公式：
        willingness = boredom*0.30 + loneliness*0.25 + social_desire*0.20
                    + mood*0.10 + time_factor*0.10 + curiosity*0.05

        这个意愿值会影响算法决策：
        - 当模型不想回复时，算法接管
        - 无聊意愿高时会降低skip倾向
        """
        time_since_update = (time.time() - self._last_boredom_update) / 60.0
        time_factor = min(1.0, time_since_update / 60.0)

        willingness = (
            self._boredom_level * 0.30
            + self._loneliness_level * 0.25
            + self._social_desire_level * 0.20
            + time_factor * 0.10
            + self._curiosity_level * 0.05
        )

        if self._is_monitoring:
            willingness += 0.08

        if energy < 0.3:
            willingness *= 0.5
        elif energy < 0.5:
            willingness *= 0.75

        return max(0.0, min(1.0, willingness))

    def _compute_strict_mode_adjustment(self) -> Tuple[float, float]:
        """计算严格模式下的阈值调整

        严格模式时：
        - skip_threshold提高（更难跳过，要更积极参与）
        - 融入意愿门槛提高

        返回 (skip_threshold调整, 融入意愿门槛调整)
        """
        if not self._strict_mode_active:
            return 0.0, 0.0

        rejection_boost = min(self._model_rejection_streak * 0.1, 0.3)
        skip_adjustment = rejection_boost

        boredom_bonus = self._boredom_level * 0.15
        loneliness_bonus = self._loneliness_level * 0.1
        willingness_threshold_adjustment = -(boredom_bonus + loneliness_bonus)

        logger.debug(
            f"[自适应v2] 严格模式调整: skip+{skip_adjustment:.3f} "
            f"意愿门槛{willingness_threshold_adjustment:+.3f} "
            f"(rejection_streak={self._model_rejection_streak})"
        )

        return skip_adjustment, willingness_threshold_adjustment

    def compute_integrated_decision(
        self,
        energy: float,
        affection: float,
        trust: float,
        burst_intensity: float,
        msg_emotion: float,
        desire_level: float,
        current_time: float,
    ) -> Tuple[str, float, float]:
        """综合决策入口：整合无聊驱动 + 算法接管 + 严格模式

        返回 (decision, skip_threshold_adjustment, willingness)

        这是模型拒绝后的算法接管核心：
        1. 当模型不想回复时，调用此方法
        2. 算法会根据无聊状态、严格模式综合判断
        3. 返回接管决策（skip/join/llm_assist）
        """
        self.update_boredom_state(
            silence_seconds=(
                current_time - self._last_bot_speak_ts
                if self._last_bot_speak_ts > 0
                else 0.0
            ),
            mood=energy,
            energy=energy,
        )

        boredom_willingness = self.get_boredom_driven_willingness(energy)

        strict_skip_adj, strict_will_adj = (
            self._compute_strict_mode_adjustment()
        )

        base_willingness = self._compute_融入意愿(
            energy=energy,
            affection=affection,
            trust=trust,
            burst_intensity=burst_intensity,
            msg_emotion=msg_emotion,
            desire_level=desire_level,
        )

        integrated_willingness = (
            base_willingness * 0.6
            + boredom_willingness * 0.3
            + (desire_level / 10.0) * 0.1
        )

        if strict_will_adj != 0.0:
            integrated_willingness += strict_will_adj

        integrated_willingness = max(0.0, min(1.0, integrated_willingness))

        if integrated_willingness >= 0.7 and burst_intensity <= 0.5:
            decision = "join"
        elif integrated_willingness <= 0.3:
            decision = "skip"
        elif integrated_willingness >= 0.5 and burst_intensity >= 0.7:
            decision = "llm_assist"
        elif integrated_willingness >= 0.6 and burst_intensity <= 0.3:
            decision = "join"
        elif integrated_willingness <= 0.5 and burst_intensity <= 0.4:
            decision = "skip"
        else:
            decision = "llm_assist"

        if self._strict_mode_active:
            if decision == "join" and integrated_willingness < 0.75:
                decision = "skip"
                logger.debug(
                    f"[自适应v2] 严格模式抑制join: "
                    f"意愿{integrated_willingness:.3f} < 0.75"
                )

        logger.info(
            f"[自适应v2] 综合决策: {decision} "
            f"(融入意愿={integrated_willingness:.3f} "
            f"无聊意愿={boredom_willingness:.3f} "
            f"严格模式={self._strict_mode_active})"
        )

        return decision, strict_skip_adj, integrated_willingness

    async def decide_with_model(
        self,
        message_text: str,
        user_name: str,
        user_id: str,
        psych: dict,
        is_at_me: bool = False,
        is_private: bool = False,
        current_state_info: str = "",
        is_state_reevaluation: bool = False,
        dialogue_history: List[Dict] = None,
        rapport_desc: str = "",
        profile_desc: str = "",
        memory_desc: str = "",
        condition_desc: str = "",
    ) -> Optional[Dict]:
        """让小模型自主决策（类似旧版model_driven_monitor）

        调用提示词模板，让模型根据当前状态决定：
        - action: 观望/回复/待机/休息/放下手机
        - duration: 持续时长
        - thought: 内心想法
        - reason: 决策原因
        - confidence: 置信度

        这是一个严谨的判断逻辑，比简单if-else更智能
        """
        try:
            from src.modules.brain.decision_brain import DecisionContext

            decision_brain = getattr(self, "_decision_brain", None)
            if decision_brain is None:
                try:
                    from src.modules.brain.decision_brain import (
                        get_decision_brain,
                    )

                    decision_brain = get_decision_brain()
                    self._decision_brain = decision_brain
                except Exception as e:
                    logger.debug(f"加载决策大脑失败: {e}")

            factors_text = ""
            if decision_brain:
                try:
                    ctx = DecisionContext(
                        stream_id=self.stream_id,
                        user_id=user_id,
                        message_content=message_text,
                        current_mood=psych.get("mood", "neutral"),
                        social_context={
                            "is_targeted": is_at_me,
                            "is_private": is_private,
                        },
                    )
                    factors = await decision_brain._calculate_factors(ctx)
                    factors_text = (
                        f"【环境研判数据】:\n"
                        f"- 置信度: {factors.confidence_score:.2f} | 风险值: {factors.risk_score:.2f}\n"
                        f"- 紧迫感: {factors.urgency_score:.2f} | 社交热度: {factors.social_score:.2f}\n"
                        f"- 心情指数: {factors.mood_score:.2f}\n"
                    )
                except Exception as e:
                    logger.debug(f"计算环境研判因子失败: {e}")

            reeval_hint = ""
            if is_state_reevaluation:
                reeval_hint = (
                    "【注意】你的状态刚刚到期，需要重新决定接下来要做什么。"
                )

            prompt = await self._build_model_decision_prompt(
                message_text=message_text,
                user_name=user_name,
                user_id=user_id,
                psych=psych,
                current_state_info=current_state_info,
                reeval_hint=reeval_hint,
                is_at_me=is_at_me,
                is_private=is_private,
                factors_text=factors_text,
                dialogue_history=dialogue_history,
                rapport_desc=rapport_desc,
                profile_desc=profile_desc,
                memory_desc=memory_desc,
                condition_desc=condition_desc,
            )

            response = await self._call_small_model(prompt)
            if not response:
                return None

            decision = self._parse_model_decision(response)

            can_act = await self._check_resource_constraints_model(psych)
            if not can_act:
                decision["action"] = "放下手机"
                decision["silence_minutes"] = random.randint(5, 15)
                decision["reason"] = "思考值不足，即便想回也回不动"

            logger.debug(
                f"[自适应v2] 模型决策: {decision.get('action')} | "
                f"原因: {decision.get('reason')} | 想法: {decision.get('thought')}"
            )
            return decision
        except Exception as e:
            logger.debug(f"[自适应v2] 模型决策失败: {e}")
            return None

    async def _build_model_decision_prompt(
        self,
        message_text: str,
        user_name: str,
        user_id: str,
        psych: dict,
        current_state_info: str,
        reeval_hint: str,
        is_at_me: bool,
        is_private: bool,
        factors_text: str,
        dialogue_history: List[Dict] = None,
        rapport_desc: str = "",
        profile_desc: str = "",
        memory_desc: str = "",
        condition_desc: str = "",
    ) -> str:
        """构建模型决策提示词（含完整上下文）"""
        from src.config.prompt_loader import get_prompt, PromptCategory

        now_hour = time.localtime().tm_hour
        time_desc = "深夜" if 2 <= now_hour < 6 else "晚间" if (22 <= now_hour or now_hour < 2) else "白天"

        mental_fatigue = psych.get("mental_fatigue", 0.0)
        annoyance = psych.get("annoyance", 0.0)
        favor = psych.get("favor", 0)

        state_parts = []
        if mental_fatigue > 60:
            state_parts.append("心理疲劳")
        if annoyance > 50:
            state_parts.append("很烦躁")
        state_line = "，".join(state_parts) if state_parts else "精神状态平稳"

        activity_desc = (
            f"聊天欲{int(self._chat_value)}/100"
            if hasattr(self, "_chat_value")
            else "一般"
        )
        if 23 <= now_hour or now_hour < 6:
            activity_desc += "现在是深夜，有点困"

        if is_private:
            scene_hint = "这是私聊，就你们两个人。没必要端着，有话就说。"
        else:
            scene_hint = "这是群聊，想说就说，不想说就刷刷看看。"

        current_state_hint = (
            f"\n{current_state_info}\n" if current_state_info else ""
        )
        reeval_context = f"\n{reeval_hint}\n" if reeval_hint else ""

        dialogue_block = ""
        if dialogue_history and len(dialogue_history) > 0:
            lines = []
            for msg in dialogue_history[-8:]:
                role = msg.get("role", "user")
                body = msg.get("content", "")[:150]
                if body:
                    prefix = "我" if role == "assistant" else user_name
                    lines.append(f"{prefix}: {body}")
            dialogue_block = "\n".join(lines)

        memory_context = memory_desc or "无相关记忆"
        group_rel_overview = rapport_desc or f"{user_name}对你好感{favor}"
        profile_context = profile_desc or "(暂无稳定画像)"
        condition_context = condition_desc or f"当前状态：{state_line}"
        state_context = (
            current_state_hint
            + "\n"
            + condition_context
            + f"\n时段：{time_desc}"
            + f"\n对象画像：{profile_context}"
        )

        return get_prompt(
            PromptCategory.SYSTEM,
            "model_driven_monitor",
            "decision_prompt.template",
            persona_context="",
            state_context=state_context,
            reeval_context=reeval_context,
            memory_context=memory_context,
            group_rel_overview=group_rel_overview,
            profile_context=profile_context,
            recent_str=message_text[:200],
            message_text=message_text[:100],
            time_str=f"{now_hour}:00",
            activity_desc=activity_desc,
            group_activity_desc=dialogue_block or "一般",
            topic_cooldown_desc="",
            topic_understanding_hint="",
            state_line=state_line,
            at_context="有人@你" if is_at_me else "",
            rel_feeling=f"你对{user_name}印象={favor}",
            current_mood=psych.get("mood", "一般"),
            factors_text=factors_text,
            scene_hint=scene_hint,
        )

    async def _call_small_model(self, prompt: str) -> Optional[str]:
        """调用小模型"""
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            request = LLMRequest(
                model_config.model_task_config.lightweight, request_type="model_decision"
            )
            try:
                response, _ = await asyncio.wait_for(
                    request.generate_response_async(prompt),
                    timeout=20.0,
                )
            except asyncio.TimeoutError:
                logger.debug("[自适应v2] 小模型调用超时(20s)")
                return None
            return response
        except Exception as e:
            logger.debug(f"[自适应v2] 小模型调用失败: {e}")
            return None

    def _parse_model_decision(self, response: str) -> Dict:
        """解析模型决策响应"""
        import json
        import re

        try:
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                if len(lines) > 1:
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].startswith("```"):
                        lines = lines[:-1]
                    response = "".join(lines).strip()

            start_idx = response.find("{")
            end_idx = response.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx: end_idx + 1]
                data = json.loads(json_str)
            else:
                raise ValueError("JSON未找到")

            action = data.get("action", "待机")
            duration = int(data.get("duration_seconds", 60))
            confidence = float(data.get("confidence", 0.5))

            if action == "窥屏":
                duration = max(1, min(300, duration))
            elif action in ["待机", "观望"]:
                duration = max(5, min(600, duration))
            elif action == "休息":
                duration = max(10, min(300, duration))
            elif action == "回复":
                duration = 0

            return {
                "action": action,
                "duration_seconds": duration,
                "thought": data.get("thought", "..."),
                "reason": data.get("reason", ""),
                "confidence": confidence,
                "silence_minutes": int(data.get("silence_minutes", 0)),
            }
        except Exception as e:
            logger.debug(f"[自适应v2] 解析模型决策失败: {e}")
            return {
                "action": "待机",
                "duration_seconds": 60,
                "thought": "...",
                "reason": "解析失败",
                "confidence": 0.0,
                "silence_minutes": 0,
            }

    async def _check_resource_constraints_model(self, psych: dict) -> bool:
        """检查思考值约束（供模型决策使用）"""
        try:
            mgr = self._get_energy_manager()
            if mgr is None:
                return True

            snap = mgr.capture_snapshot(self.stream_id)
            thinking_ratio = snap.thinking_ratio()

            mental_fatigue = psych.get("mental_fatigue", 0.0) if psych else 0.0
            annoyance = psych.get("annoyance", 0.0) if psych else 0.0

            pressure = (
                mental_fatigue * 0.4
                + annoyance * 0.2
                + (1.0 - thinking_ratio) * 0.4
            )

            if pressure > 100.0 or thinking_ratio < 0.1:
                return False

            return True
        except Exception:
            return True

    def get_boredom_status(self) -> Dict[str, Any]:
        """获取无聊状态快照"""
        return {
            "boredom_level": round(self._boredom_level, 3),
            "loneliness_level": round(self._loneliness_level, 3),
            "social_desire_level": round(self._social_desire_level, 3),
            "curiosity_level": round(self._curiosity_level, 3),
            "is_monitoring": self._is_monitoring,
            "monitoring_duration_sec": (
                time.time() - self._monitoring_start_time
                if self._is_monitoring
                else 0.0
            ),
            "boredom_driven_willingness": round(
                self.get_boredom_driven_willingness(), 3
            ),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "decision_count": self._decision_count,
            "history_size": len(self._history),
            "bias_version": self._bias.version,
            "streak_skip": self._streak_skip,
            "streak_llm": self._streak_llm,
            "unanswered_turns": self._unanswered_turns,
            "consecutive_speaks": self._consecutive_speaks,
            "silence_effectiveness": self._silence_effectiveness,
            "speak_effectiveness": self._speak_effectiveness,
            "interaction_pattern": self._interaction_pattern_score,
            "learned_silence_threshold": self._learned_silence_threshold,
            "learned_speak_threshold": self._learned_speak_threshold,
            "model_rejection_streak": self._model_rejection_streak,
            "strict_mode_active": self._strict_mode_active,
            "boredom_level": self._boredom_level,
            "loneliness_level": self._loneliness_level,
            "social_desire_level": self._social_desire_level,
            "boredom_driven_willingness": self.get_boredom_driven_willingness(),
            "rest_until": (
                max(0.0, self._rest_until - time.time())
                if self._rest_until > time.time()
                else 0.0
            ),
            "wants_to_takeover": self._wants_to_takeover,
        }

    def _get_energy_manager(self):
        """获取能量管理器（懒加载）→ D6 EnergyChainDimension"""
        if self._energy_manager is None:
            try:
                from src.chat.heart_flow.energy_manager import EnergyChainDimension

                self._energy_manager = EnergyChainDimension.get_instance()
            except Exception as e:
                logger.debug(f"加载能量管理器失败: {e}")
        return self._energy_manager

    def get_thinking_value(self, channel_id: str) -> float:
        """获取思考值（从energy_manager）"""
        mgr = self._get_energy_manager()
        if mgr is None:
            return 100.0
        try:
            snap = mgr.capture_snapshot(channel_id)
            return snap.thinking_value
        except Exception:
            return 100.0

    def get_thinking_ratio(self, channel_id: str) -> float:
        """获取思考值比例（0-1）"""
        mgr = self._get_energy_manager()
        if mgr is None:
            return 1.0
        try:
            snap = mgr.capture_snapshot(channel_id)
            return snap.thinking_ratio()
        except Exception:
            return 1.0

    def wants_to_rest(self, channel_id: str) -> bool:
        """判断是否想休息/摸鱼

        基于思考值判断：
        - 思考值低于阈值 → 想休息
        - 思考值越低越想休息摸鱼
        """
        thinking_ratio = self.get_thinking_ratio(channel_id)

        if thinking_ratio <= 0.1:
            return True
        if thinking_ratio <= 0.25 and self._boredom_level > 0.6:
            return True
        if self._boredom_level > 0.8:
            return True

        return False

    def wants_to_work(self, channel_id: str) -> bool:
        """判断是否想工作/主动说话

        条件：
        1. 思考值恢复到一定程度（>=50%）
        2. 无聊值足够高（>=0.4）
        3. 不是休息状态
        """
        thinking_ratio = self.get_thinking_ratio(channel_id)

        if thinking_ratio < 0.5:
            return False

        if self._boredom_level < 0.4:
            return False

        boredom_driven = self.get_boredom_driven_willingness()
        if boredom_driven < 0.35:
            return False

        return True

    def should_offer_takeover(self) -> bool:
        """是否应该询问模型是否要接管

        当状态到期后，算法询问模型是否要接管
        """
        if self._rest_until > time.time():
            return False

        if not self._takeover_offered:
            return True

        if self._takeover_offered and not self._wants_to_takeover:
            elapsed = time.time() - self._takeover_offer_time
            if elapsed > 60.0:
                return True

        return False

    def offer_takeover_to_model(self) -> None:
        """向模型发出接管询问"""
        self._takeover_offered = True
        self._takeover_offer_time = time.time()
        logger.info(
            f"[自适应v2] 算法询问模型是否接管 "
            f"(无聊={self._boredom_level:.2f} 孤独={self._loneliness_level:.2f})"
        )

    def model_takeover_response(self, wants_to_takeover: bool) -> None:
        """模型对接管询问的回应

        Args:
            wants_to_takeover: 模型是否想接管
        """
        self._wants_to_takeover = wants_to_takeover
        if wants_to_takeover:
            self._rest_until = 0.0
            self._strict_mode_active = True
            self._strict_mode_until = time.time() + 300.0
            logger.info("[自适应v2] 模型选择接管，激活严格模式")
        else:
            planned_rest = self._plan_rest_duration()
            self._rest_until = time.time() + planned_rest
            logger.info(f"[自适应v2] 模型拒绝接管，休息{planned_rest:.0f}秒")

    def _plan_rest_duration(self) -> float:
        """规划休息时长（算法预筛选，返回模型决策参考值）

        返回建议休息时长（秒），实际由模型最终决定
        范围：60-600秒（1-10分钟）
        特殊情况可延长到1800秒（30分钟）
        """
        boredom_ratio = self._boredom_level

        if boredom_ratio >= 0.8:
            base_min = 180
            base_max = 600
        elif boredom_ratio >= 0.6:
            base_min = 120
            base_max = 300
        elif boredom_ratio >= 0.4:
            base_min = 60
            base_max = 180
        else:
            base_min = 30
            base_max = 120

        if self._loneliness_level > 0.7:
            base_min = int(base_min * 1.3)
            base_max = int(base_max * 1.3)

        import random

        suggested = random.randint(base_min, base_max)
        logger.debug(
            f"[自适应v2] 算法预筛选建议休息: {suggested}秒 "
            f"(无聊={boredom_ratio:.2f} 孤独={self._loneliness_level:.2f})"
        )
        return suggested

    async def plan_rest_with_model(
        self,
        chat_value: float,
        thinking_value: float,
        consecutive_replies: int,
        mood: str = "",
        thought: str = "",
        dialogue_summary: str = "",
    ) -> int:
        """让模型自主规划休息时长

        调用模型的提示词模板，让模型根据当前状态决定休息多久
        """
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            rule_rest = self._plan_rest_duration()

            import time

            now_hour = time.localtime().tm_hour
            time_desc = (
                "深夜"
                if 2 <= now_hour < 6
                else "晚间" if (22 <= now_hour or now_hour < 2) else "白天"
            )

            prompt = self._build_rest_decision_prompt(
                chat_value=chat_value,
                thinking_value=thinking_value,
                consecutive_replies=consecutive_replies,
                mood=mood,
                time_desc=time_desc,
                now_hour=now_hour,
                thought=thought,
                dialogue_summary=dialogue_summary,
                rule_rest=rule_rest,
            )

            request = LLMRequest(
                model_config.model_task_config.lightweight, request_type="rest_decision"
            )
            try:
                response, _ = await asyncio.wait_for(
                    request.generate_response_async(prompt),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                logger.debug("[自适应v2] 休息决策超时(15s)")
                response = None

            if response:
                try:
                    raw_val = int(response.strip().split()[0])
                    if raw_val < 10:
                        raw_val = raw_val * 60
                    rest_sec = max(60, min(1800, raw_val))
                    logger.info(f"[自适应v2] 模型决定休息: {rest_sec}秒")
                    return rest_sec
                except (ValueError, IndexError):
                    pass
        except Exception as e:
            logger.debug(f"[自适应v2] 模型休息规划失败: {e}")

        return rule_rest

    def _build_rest_decision_prompt(
        self,
        chat_value: float,
        thinking_value: float,
        consecutive_replies: int,
        mood: str,
        time_desc: str,
        now_hour: int,
        thought: str,
        dialogue_summary: str,
        rule_rest: int,
    ) -> str:
        """构建休息决策提示词"""
        from src.config.prompt_loader import get_prompt, PromptCategory

        return get_prompt(
            PromptCategory.HEARTFLOW,
            "skills",
            "rest_decision.template",
            chat_value=f"{chat_value:.0f}",
            brain_power=f"{thinking_value:.0f}",
            consecutive_replies=consecutive_replies,
            mood=mood or "一般",
            time_desc=time_desc,
            now_hour=now_hour,
            thought=thought[:100] if thought else "无",
            dialogue_summary=(
                dialogue_summary[:100] if dialogue_summary else "无"
            ),
            rule_rest=rule_rest,
            rule_rest_max=min(rule_rest * 2, 300),
        )

    def check_state_expiration(self, current_time: float) -> Optional[Dict]:
        """检查状态是否到期

        状态到期后，算法会询问模型是否要接管
        """
        if self._last_state_check == 0.0:
            self._last_state_check = current_time
            return None

        if current_time - self._last_state_check < self._state_check_interval:
            return None

        self._last_state_check = current_time

        if self._rest_until > current_time:
            return None

        if self._state_expired:
            return None

        self._state_expired = True
        self._state_expired_time = current_time

        return {
            "expired": True,
            "boredom": self._boredom_level,
            "loneliness": self._loneliness_level,
            "social_desire": self._social_desire_level,
        }

    def reset_state_expiration(self) -> None:
        """重置状态到期标记"""
        self._state_expired = False
        self._state_expired_time = 0.0

    def is_resting(self) -> bool:
        """是否正在休息"""
        return self._rest_until > time.time()

    def get_full_status(self) -> Dict[str, Any]:
        """获取完整状态（用于调试和日志）"""
        return {
            "boredom_level": round(self._boredom_level, 3),
            "loneliness_level": round(self._loneliness_level, 3),
            "social_desire_level": round(self._social_desire_level, 3),
            "social_desire": round(self._social_desire_level, 3),
            "is_resting": self.is_resting(),
            "rest_remaining": (
                max(0.0, self._rest_until - time.time())
                if self._rest_until > time.time()
                else 0.0
            ),
            "boredom_driven_willingness": round(
                self.get_boredom_driven_willingness(), 3
            ),
            "strict_mode_active": self._strict_mode_active,
            "model_rejection_streak": self._model_rejection_streak,
        }


_global_learner_v2: Optional[AdaptiveThresholdLearnerV2] = None


def get_adaptive_threshold_learner_v2(
    stream_id: str = "default",
) -> AdaptiveThresholdLearnerV2:
    global _global_learner_v2
    if _global_learner_v2 is None:
        _global_learner_v2 = AdaptiveThresholdLearnerV2(stream_id)
    return _global_learner_v2

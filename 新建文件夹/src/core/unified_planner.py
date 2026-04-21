import time
import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Callable,
    Set,
    TYPE_CHECKING,
)
from enum import Enum
from src.common.logger import get_logger

if TYPE_CHECKING:
    from src.core.world_snapshot import WorldSnapshot

logger = get_logger("统一规划")


class PlanningPhase(Enum):
    """规划阶段"""

    IDLE = "idle"
    PERCEIVING = "perceiving"
    PLANNING = "planning"
    INDEXING = "indexing"
    FILTERING = "filtering"
    DECIDING = "deciding"
    EXECUTING = "executing"


class ActionType(Enum):
    """行动类型"""

    OBSERVE = "observe"
    REPLY = "reply"
    PROACTIVE_SPEAK = "proactive_speak"
    JOIN_CONVERSATION = "join_conversation"
    CHECK_LATER = "check_later"
    WAIT = "wait"
    NO_REPLY = "no_reply"
    COMPLETE_TALK = "complete_talk"
    DEEP_THINK = "deep_think"
    REST = "rest"

    def label(self) -> str:
        return {
            "observe": "观察",
            "reply": "回复",
            "proactive_speak": "主动发言",
            "join_conversation": "加入对话",
            "check_later": "稍后再看",
            "wait": "等待",
            "no_reply": "不回复",
            "complete_talk": "完成对话",
            "deep_think": "深度思考",
            "rest": "休息",
        }.get(self.value, self.value)


@dataclass
class ThinkingBudget:
    """思考预算"""

    thinking_value: float = 100.0
    thinking_ceiling: float = 100.0
    min_for_planning: float = 20.0
    min_for_proactive: float = 40.0
    min_for_deep_think: float = 60.0
    cost_per_planning: float = 0.3
    cost_per_proactive: float = 1.5
    cost_per_deep_think: float = 3.0

    def can_plan(self) -> bool:
        return self.thinking_value >= self.min_for_planning

    def can_proactive(self) -> bool:
        return self.thinking_value >= self.min_for_proactive

    def can_deep_think(self) -> bool:
        return self.thinking_value >= self.min_for_deep_think

    def consume(self, amount: float) -> float:
        consumed = min(amount, self.thinking_value)
        self.thinking_value = max(0.0, self.thinking_value - consumed)
        return consumed

    def ratio(self) -> float:
        return (
            self.thinking_value / self.thinking_ceiling
            if self.thinking_ceiling > 0
            else 0.0
        )


@dataclass
class ContextSegment:
    """上下文片段"""

    segment_id: str
    content: str
    source: str
    importance: float = 0.5
    relevance: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanningContext:
    """规划上下文"""

    channel_id: str
    user_id: str = ""
    silence_seconds: float = 0.0
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    preferred_reply_style: str = ""
    preferred_quote: Optional[bool] = None
    active_users: List[str] = field(default_factory=list)
    current_topics: List[str] = field(default_factory=list)
    relationship_candidates: List[Tuple[str, float]] = field(
        default_factory=list
    )
    atmosphere: str = "neutral"
    my_last_message: str = ""
    my_last_message_time: float = 0.0
    thinking_budget: ThinkingBudget = field(default_factory=ThinkingBudget)
    chat_value: float = 100.0
    activity_level: float = 50.0
    social_value: float = 0.0
    trust_value: float = 0.0
    relationship_level: int = 2
    custom_label: str = ""
    annoyance_value: float = 0.0
    affection: float = 0.0
    trust_score: float = 0.0
    psychological_pressure: float = 0.0
    shared_social_value: float = 0.0
    trauma_score: float = 0.0
    repeated_topic_pressure: float = 0.0
    last_user_intent: str = ""
    memory_hint: str = ""
    context_execution_block: str = ""
    intention_hint: str = ""
    active_intentions: List[Dict[str, Any]] = field(default_factory=list)
    group_pattern_hint: str = ""
    behavior_category: str = "neutral"
    behavior_severity: float = 0.0
    behavior_reason: str = ""
    is_new_user: bool = False
    short_term_context: List[ContextSegment] = field(default_factory=list)
    long_term_context: List[ContextSegment] = field(default_factory=list)
    filtered_context: List[ContextSegment] = field(default_factory=list)


@dataclass
class PlanningDecision:
    """规划决策"""

    action: ActionType
    reason: str = ""
    content_plan: str = ""
    target_user_id: str = ""
    preferred_reply_style: str = ""
    preferred_quote: Optional[bool] = None
    next_check_seconds: float = 60.0
    confidence: float = 0.5
    thinking_consumed: float = 0.0
    context_used: List[str] = field(default_factory=list)
    internal_state: str = ""
    planning_process: str = ""
    matched_intent_id: str = ""
    matched_intent_kind: str = ""
    matched_intent_urgency: float = 0.0


class ContextIndexer:
    """
    上下文索引器

    动态索引和检索上下文片段。
    根据模型输出需求选择上下文。
    """

    def __init__(self):
        self._short_term_cache: Dict[str, List[ContextSegment]] = {}
        self._long_term_index: Dict[str, Dict[str, ContextSegment]] = {}
        self._max_short_term = 20
        self._max_long_term = 100
        self._importance_threshold = 0.3

    def index_short_term(
        self,
        channel_id: str,
        messages: List[Dict[str, Any]],
    ) -> List[ContextSegment]:
        """索引短期上下文（最近消息）"""
        if channel_id not in self._short_term_cache:
            self._short_term_cache[channel_id] = []
        segments = []
        for msg in messages[-self._max_short_term:]:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if not content:
                continue
            segment_id = self._generate_segment_id(content)
            importance = self._calculate_importance(content, msg)
            segment = ContextSegment(
                segment_id=segment_id,
                content=content,
                source="recent_message",
                importance=importance,
                timestamp=msg.get("timestamp", time.time()),
                metadata={
                    "speaker": msg.get("speaker", ""),
                    "is_bot": msg.get("is_bot", False),
                },
            )
            segments.append(segment)
        self._short_term_cache[channel_id] = segments
        return segments

    def index_long_term(
        self,
        channel_id: str,
        memories: List[Any],
    ) -> List[ContextSegment]:
        """索引长期上下文（记忆）"""
        if channel_id not in self._long_term_index:
            self._long_term_index[channel_id] = {}
        segments = []
        for mem in memories[: self._max_long_term]:
            content = getattr(mem, "content", str(mem))
            if not content:
                continue
            segment_id = getattr(
                mem, "record_id", self._generate_segment_id(content)
            )
            importance = getattr(mem, "significance", 0.5)
            segment = ContextSegment(
                segment_id=segment_id,
                content=content,
                source="memory",
                importance=importance,
                timestamp=getattr(mem, "birth_ts", time.time()),
                metadata={
                    "category": getattr(mem, "entry_category", "unknown"),
                    "visit_count": getattr(mem, "visit_count", 0),
                },
            )
            segments.append(segment)
            self._long_term_index[channel_id][segment_id] = segment
        return segments

    def retrieve_by_relevance(
        self,
        channel_id: str,
        query: str,
        max_segments: int = 10,
        min_relevance: float = 0.2,
    ) -> List[ContextSegment]:
        """根据片段相似度与结构信号检索上下文"""
        normalized_query = self._normalize_text(query)
        short_term = self._short_term_cache.get(channel_id, [])
        long_term = list(self._long_term_index.get(channel_id, {}).values())
        all_segments = short_term + long_term
        scored = []
        for segment in all_segments:
            relevance = self._calculate_relevance(segment, normalized_query)
            if relevance >= min_relevance:
                segment.relevance = relevance
                scored.append((segment, relevance))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:max_segments]]

    def retrieve_random_sample(
        self,
        channel_id: str,
        sample_size: int = 5,
        importance_weighted: bool = True,
    ) -> List[ContextSegment]:
        """随机采样上下文（可选重要性加权）"""
        import random

        short_term = self._short_term_cache.get(channel_id, [])
        long_term = list(self._long_term_index.get(channel_id, {}).values())
        all_segments = short_term + long_term
        if not all_segments:
            return []
        if importance_weighted:
            weights = [s.importance for s in all_segments]
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                return random.choices(
                    all_segments,
                    weights=weights,
                    k=min(sample_size, len(all_segments)),
                )
            # 全部权重为0，降级为均匀采样
        return random.sample(all_segments, min(sample_size, len(all_segments)))

    def retrieve_by_duration(
        self,
        channel_id: str,
        duration_seconds: float,
        max_segments: int = 10,
    ) -> List[ContextSegment]:
        """根据时间范围检索上下文"""
        if duration_seconds < 0:
            duration_seconds = 0.0
        now = time.time()
        cutoff = now - duration_seconds
        short_term = self._short_term_cache.get(channel_id, [])
        long_term = list(self._long_term_index.get(channel_id, {}).values())
        all_segments = short_term + long_term
        filtered = [s for s in all_segments if s.timestamp >= cutoff]
        filtered.sort(key=lambda x: x.importance, reverse=True)
        return filtered[:max_segments]

    def _generate_segment_id(self, content: str) -> str:
        """生成片段ID"""
        seed = f"{content}:{time.time()}"
        return hashlib.md5(seed.encode()).hexdigest()[:12]

    def _calculate_importance(
        self, content: str, msg: Dict[str, Any]
    ) -> float:
        """计算重要性，尽量依赖结构而不是固定词表"""
        importance = 0.5
        question_marks = content.count("?") + content.count("？")
        exclamations = content.count("!") + content.count("！")
        line_breaks = content.count("\n")
        if question_marks > 0:
            importance += 0.15
        if exclamations >= 2:
            importance += 0.1
        if len(content) > 100:
            importance += 0.1
        if line_breaks >= 2:
            importance += 0.05
        if msg.get("is_bot"):
            importance -= 0.1
        return max(0.0, min(1.0, importance))

    def _calculate_relevance(
        self, segment: ContextSegment, normalized_query: str
    ) -> float:
        """计算相关性，结合字符片段相似度与结构接近度"""
        normalized_segment = self._normalize_text(segment.content)
        if not normalized_query or not normalized_segment:
            return 0.0
        query_ngrams = self._build_ngrams(normalized_query)
        segment_ngrams = self._build_ngrams(normalized_segment)
        overlap = len(query_ngrams & segment_ngrams)
        union = len(query_ngrams | segment_ngrams)
        ngram_score = overlap / union if union > 0 else 0.0
        query_questions = normalized_query.count("?") + normalized_query.count(
            "？"
        )
        segment_questions = normalized_segment.count(
            "?"
        ) + normalized_segment.count("？")
        structure_score = 0.0
        if query_questions and segment_questions:
            structure_score += 0.15
        length_gap = abs(
            len(normalized_query) - len(normalized_segment)
        ) / max(len(normalized_query), len(normalized_segment), 1)
        structure_score += max(0.0, 0.2 - length_gap * 0.2)
        weighted = (
            ngram_score * 0.7 + structure_score + segment.importance * 0.15
        )
        return max(0.0, min(1.0, weighted))

    @staticmethod
    def _normalize_text(text: str) -> str:
        """对文本做轻量归一化，保留基本句式结构。"""
        if not text:
            return ""
        normalized = text.lower()
        for char in [
            " ",
            "\n",
            "\t",
            "，",
            "。",
            "、",
            "；",
            "：",
            '"',
            "'",
            "（",
            "）",
            "(",
            ")",
            "[",
            "]",
            "【",
            "】",
        ]:
            normalized = normalized.replace(char, "")
        return normalized.strip()

    @staticmethod
    def _build_ngrams(text: str, width: int = 2) -> Set[str]:
        """构建字符 n-gram 集合，用于弱词面相似度比较。"""
        if not text:
            return set()
        if len(text) <= width:
            return {text}
        return {
            text[index: index + width]
            for index in range(len(text) - width + 1)
        }


class ContextFilter:
    """
    上下文过滤器

    根据模型容量和输出需求过滤上下文。
    """

    def __init__(self):
        self._model_capacity_tokens = 4000
        self._reserve_tokens = 500
        self._max_segments = 15

    def filter_for_output(
        self,
        segments: List[ContextSegment],
        output_type: str = "reply",
        max_tokens: Optional[int] = None,
    ) -> List[ContextSegment]:
        """根据输出类型过滤上下文"""
        if not segments:
            return []
        max_tokens = max_tokens or (
            self._model_capacity_tokens - self._reserve_tokens
        )
        if output_type == "reply":
            return self._filter_for_reply(segments, max_tokens)
        elif output_type == "proactive":
            return self._filter_for_proactive(segments, max_tokens)
        elif output_type == "deep_think":
            return self._filter_for_deep_think(segments, max_tokens)
        else:
            return self._filter_generic(segments, max_tokens)

    def _filter_for_reply(
        self, segments: List[ContextSegment], max_tokens: int
    ) -> List[ContextSegment]:
        """为回复过滤上下文"""
        recent = sorted(segments, key=lambda s: s.timestamp, reverse=True)[:10]
        scored = []
        for seg in recent:
            score = (
                seg.importance * 0.4
                + seg.relevance * 0.4
                + (1.0 - (time.time() - seg.timestamp) / 3600) * 0.2
            )
            scored.append((seg, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        result = []
        total_tokens = 0
        for seg, _ in scored:
            seg_tokens = len(seg.content) // 2
            if total_tokens + seg_tokens <= max_tokens:
                result.append(seg)
                total_tokens += seg_tokens
            if len(result) >= self._max_segments:
                break
        return result

    def _filter_for_proactive(
        self, segments: List[ContextSegment], max_tokens: int
    ) -> List[ContextSegment]:
        """为主动发言过滤上下文"""
        high_importance = [s for s in segments if s.importance > 0.5]
        scored = []
        for seg in high_importance:
            score = seg.importance * 0.5 + seg.relevance * 0.5
            scored.append((seg, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        result = []
        total_tokens = 0
        for seg, _ in scored:
            seg_tokens = len(seg.content) // 2
            if total_tokens + seg_tokens <= max_tokens:
                result.append(seg)
                total_tokens += seg_tokens
            if len(result) >= self._max_segments:
                break
        return result

    def _filter_for_deep_think(
        self, segments: List[ContextSegment], max_tokens: int
    ) -> List[ContextSegment]:
        """为深度思考过滤上下文"""
        return sorted(segments, key=lambda s: s.importance, reverse=True)[
            : self._max_segments
        ]

    def _filter_generic(
        self, segments: List[ContextSegment], max_tokens: int
    ) -> List[ContextSegment]:
        """通用过滤"""
        return segments[: self._max_segments]


class UnifiedPlanner:
    """
    统一规划系统

    整合所有规划能力，基于思考值进行决策。
    结构化工作流：规划→索引→过滤→返回→决策
    """

    _instance: Optional["UnifiedPlanner"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._context_indexer = ContextIndexer()
        self._context_filter = ContextFilter()
        self._channel_states: Dict[str, PlanningPhase] = {}
        self._channel_contexts: Dict[str, PlanningContext] = {}
        self._decision_history: Dict[str, List[PlanningDecision]] = {}
        self._max_history = 50
        self._module_registry: Dict[str, Any] = {}
        self._action_callbacks: Dict[str, Callable] = {}
        self._planning_cooldown = 30.0
        self._last_planning_time: Dict[str, float] = {}
        self._max_cached_channels = 500
        self._channel_stale_sec = 3600.0
        self._plan_locks: Dict[str, asyncio.Lock] = {}
        logger.info("[统一规划] 初始化完成")

    def _evict_stale_channels(self) -> None:
        """清理长时间无活动的频道数据，防止字典无限增长"""
        if len(self._last_planning_time) <= self._max_cached_channels:
            return
        now = time.time()
        stale_ids = [
            cid
            for cid, ts in self._last_planning_time.items()
            if now - ts > self._channel_stale_sec
        ]
        for cid in stale_ids:
            self._channel_states.pop(cid, None)
            self._channel_contexts.pop(cid, None)
            self._decision_history.pop(cid, None)
            self._last_planning_time.pop(cid, None)
        if stale_ids:
            logger.debug(f"[统一规划] 清理 {len(stale_ids)} 个过期频道缓存")

    def register_module(
        self, name: str, module: Any, functions: List[str]
    ) -> None:
        """注册模块"""
        self._module_registry[name] = {
            "module": module,
            "functions": functions,
        }
        logger.info(f"[统一规划] 注册模块: {name}, 功能: {functions}")

    def register_action_callback(
        self, action_type: ActionType, callback: Callable
    ) -> None:
        """注册行动回调"""
        self._action_callbacks[action_type.value] = callback

    def _get_thinking_budget(self, channel_id: str) -> ThinkingBudget:
        """获取思考预算"""
        try:
            from src.chat.heart_flow.energy_manager import get_vitality_pool

            pool = get_vitality_pool()
            snap = pool.capture_snapshot(channel_id)
            return ThinkingBudget(
                thinking_value=snap.thinking_value,
                thinking_ceiling=snap.thinking_ceiling,
            )
        except Exception:
            return ThinkingBudget()

    def _consume_thinking(
        self, channel_id: str, amount: float, action_type: str = "plan"
    ) -> float:
        """
        动态消耗思考值

        参考旧版动态计算：
        - 基础消耗
        - 连续发言惩罚
        - 频率因子（短时间发言消耗更多）
        - 兴趣折扣
        """
        try:
            from src.chat.heart_flow.energy_manager import get_vitality_pool

            pool = get_vitality_pool()
            snap = pool.capture_snapshot(channel_id)
            base_cost = amount
            if action_type == "proactive":
                base_cost = 8.0
            elif action_type == "deep_think":
                base_cost = 15.0
            else:
                base_cost = 3.0
            consecutive_penalty = 0.0
            history = self._decision_history.get(channel_id, [])
            if history:
                last = history[-1]
                time_since_last = time.time() - self._last_planning_time.get(
                    channel_id, 0
                )
                if time_since_last < 60:
                    consecutive_penalty = 5.0
                elif time_since_last < 300:
                    consecutive_penalty = 2.0
            frequency_factor = 1.0
            dynamic_cost = (base_cost + consecutive_penalty) * frequency_factor
            dynamic_cost = max(1.0, min(30.0, dynamic_cost))
            result = pool.deplete_on_think(channel_id)
            actual_cost = result.get("brain_cost", dynamic_cost)
            logger.debug(
                f"[统一规划] {channel_id[:8]} 思考消耗: "
                f"基础={base_cost:.1f}, 连续惩罚={consecutive_penalty:.1f}, "
                f"频率因子={frequency_factor:.1f}, 实际={actual_cost:.1f}"
            )
            return actual_cost
        except Exception as e:
            logger.warning(f"[统一规划] 思考消耗计算失败: {e}")
            return amount

    async def plan(
        self,
        channel_id: str,
        user_id: str = "",
        trigger: str = "timer",
        hints: Optional[Dict[str, Any]] = None,
        world_snapshot: Optional["WorldSnapshot"] = None,
    ) -> PlanningDecision:
        """
        执行规划流程

        结构化工作流：
        1. 感知环境 → 规划阶段
        2. 索引上下文 → 索引阶段
        3. 过滤上下文 → 过滤阶段
        4. LLM决策 → 决策阶段
        5. 执行行动 → 执行阶段
        """
        plan_lock = self._plan_locks.setdefault(channel_id, asyncio.Lock())
        if plan_lock.locked():
            return PlanningDecision(
                action=ActionType.CHECK_LATER,
                reason="该频道正在规划中",
                next_check_seconds=5,
            )
        async with plan_lock:
            return await self._plan_inner(
                channel_id, user_id, trigger, hints, world_snapshot
            )

    async def _plan_inner(
        self,
        channel_id: str,
        user_id: str = "",
        trigger: str = "unknown",
        hints: Optional[Dict[str, Any]] = None,
        world_snapshot: Optional[Dict[str, Any]] = None,
    ) -> PlanningDecision:
        """规划核心逻辑（受 plan_lock 保护）"""
        now = time.time()
        self._evict_stale_channels()
        last_time = self._last_planning_time.get(channel_id, 0)
        if (
            now - last_time < self._planning_cooldown
            and trigger != "user_message"
        ):
            return PlanningDecision(
                action=ActionType.CHECK_LATER,
                reason=f"冷却中({self._planning_cooldown - (now - last_time):.0f}秒)",
                next_check_seconds=self._planning_cooldown - (now - last_time),
            )
        self._channel_states[channel_id] = PlanningPhase.PERCEIVING
        ctx = await self._perceive_environment(
            channel_id, user_id, hints or {}, world_snapshot
        )
        self._channel_contexts[channel_id] = ctx
        budget = ctx.thinking_budget
        if not budget.can_plan():
            self._channel_states[channel_id] = PlanningPhase.IDLE
            return PlanningDecision(
                action=ActionType.REST,
                reason=f"思考值不足({
                    budget.thinking_value:.0f} < {
                    budget.min_for_planning})",
                next_check_seconds=120,
            )
        self._channel_states[channel_id] = PlanningPhase.INDEXING
        ctx = await self._index_context(ctx)
        self._channel_states[channel_id] = PlanningPhase.FILTERING
        ctx = await self._filter_context(ctx)
        self._channel_states[channel_id] = PlanningPhase.DECIDING
        decision = await self._make_decision(ctx)
        action_type = "plan"
        if decision.action == ActionType.PROACTIVE_SPEAK:
            action_type = "proactive"
        elif decision.action == ActionType.DEEP_THINK:
            action_type = "deep_think"
        consumed = self._consume_thinking(
            channel_id, decision.thinking_consumed, action_type
        )
        decision.thinking_consumed = consumed
        self._channel_states[channel_id] = PlanningPhase.EXECUTING
        await self._execute_decision(channel_id, decision)
        self._last_planning_time[channel_id] = time.time()
        self._channel_states[channel_id] = PlanningPhase.IDLE
        self._record_decision(channel_id, decision)
        return decision

    async def _perceive_environment(
        self,
        channel_id: str,
        user_id: str,
        hints: Dict[str, Any],
        world_snapshot: Optional["WorldSnapshot"] = None,
    ) -> PlanningContext:
        """感知环境：优先从 WorldSnapshot 读取，否则按需构建"""
        ctx = PlanningContext(channel_id=channel_id, user_id=user_id)
        ctx.thinking_budget = self._get_thinking_budget(channel_id)
        # 如果调用方未传入快照，发出警告后就地构建（正常流程应由主链提前构建）
        if world_snapshot is None:
            logger.warning(
                f"[规划器] plan() 未收到预构建的 WorldSnapshot (channel={channel_id[:8]}), "
                "退回自行构建——请检查调用链是否遗漏 snapshot 传递"
            )
            from src.core.world_snapshot import build_world_snapshot

            world_snapshot = await build_world_snapshot(channel_id, user_id)
        # 从快照填充资源状态
        res = world_snapshot.self_resources
        ctx.chat_value = res.chat_energy
        ctx.activity_level = res.activity_level
        ctx.silence_seconds = res.silence_seconds
        # 从快照填充用户状态
        tu = world_snapshot.target_user
        ctx.social_value = tu.social_value
        ctx.trust_value = tu.trust_value
        ctx.shared_social_value = tu.social_value
        ctx.affection = tu.affection
        ctx.trust_score = tu.trust_score
        ctx.annoyance_value = tu.annoyance_value
        ctx.trauma_score = tu.trauma_score
        ctx.psychological_pressure = tu.psychological_pressure
        ctx.relationship_level = tu.relationship_level
        ctx.custom_label = tu.custom_label

        # 外部注入：重复压力与用户意图
        ctx.repeated_topic_pressure = float(
            hints.get("repeated_topic_pressure", 0.0) or 0.0
        )
        ctx.last_user_intent = str(hints.get("last_user_intent", "") or "")
        ctx.memory_hint = str(hints.get("memory_hint", "") or "")
        ctx.context_execution_block = str(
            hints.get("context_execution_block", "") or ""
        )
        ctx.group_pattern_hint = str(hints.get("group_pattern_hint", "") or "")
        ctx.intention_hint = str(hints.get("intention_hint", "") or "")
        raw_active_intentions = hints.get("active_intentions") or []
        if isinstance(raw_active_intentions, list):
            ctx.active_intentions = [
                item
                for item in raw_active_intentions
                if isinstance(item, dict)
            ]
        ctx.behavior_category = str(
            hints.get("behavior_category", "neutral") or "neutral"
        )
        ctx.behavior_severity = float(
            hints.get("behavior_severity", 0.0) or 0.0
        )
        ctx.behavior_reason = str(hints.get("behavior_reason", "") or "")
        ctx.is_new_user = bool(hints.get("is_new_user", False))
        raw_active_users = hints.get("active_users") or []
        if isinstance(raw_active_users, list):
            ctx.active_users = [
                str(user_id or "").strip()
                for user_id in raw_active_users[:10]
                if str(user_id or "").strip()
            ]
        raw_current_topics = hints.get("current_topics") or []
        if isinstance(raw_current_topics, list):
            ctx.current_topics = [
                str(topic or "").strip()
                for topic in raw_current_topics[:8]
                if str(topic or "").strip()
            ]
        raw_relationship_candidates = (
            hints.get("relationship_candidates") or []
        )
        if isinstance(raw_relationship_candidates, list):
            for item in raw_relationship_candidates[:5]:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                candidate_id = str(item[0] or "").strip()
                if not candidate_id:
                    continue
                try:
                    candidate_weight = float(item[1] or 0.0)
                except Exception:
                    candidate_weight = 0.0
                ctx.relationship_candidates.append(
                    (candidate_id, candidate_weight)
                )
        ctx.recent_messages = await self._get_recent_messages(channel_id)
        return ctx

    async def _get_recent_messages(
        self, channel_id: str
    ) -> List[Dict[str, Any]]:
        """获取最近消息"""
        try:
            from src.common.message_repository import get_message_repository

            repo = get_message_repository()
            messages = repo.get_recent_messages(channel_id, limit=20)
            return [
                {
                    "content": msg.content,
                    "speaker": msg.user_nickname or msg.user_id,
                    "is_bot": msg.user_id == "bot",
                    "timestamp": msg.timestamp,
                }
                for msg in messages
            ]
        except Exception:
            return []

    async def _index_context(self, ctx: PlanningContext) -> PlanningContext:
        """索引上下文"""
        short_term = self._context_indexer.index_short_term(
            ctx.channel_id,
            ctx.recent_messages,
        )
        ctx.short_term_context = short_term
        try:
            from src.memory_system.memory_core import acquire_recollection_hub

            hub = acquire_recollection_hub()
            memories = hub.query_memories(ctx.channel_id, ceiling=20)
            long_term = self._context_indexer.index_long_term(
                ctx.channel_id, memories
            )
            ctx.long_term_context = long_term
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return ctx

    async def _filter_context(self, ctx: PlanningContext) -> PlanningContext:
        """过滤上下文"""
        all_segments = ctx.short_term_context + ctx.long_term_context
        output_type = self._determine_output_type(ctx)
        filtered = self._context_filter.filter_for_output(
            all_segments, output_type
        )
        ctx.filtered_context = filtered
        return ctx

    def _determine_output_type(self, ctx: PlanningContext) -> str:
        """确定输出类型"""
        if ctx.silence_seconds > 300:
            return "proactive"
        if ctx.thinking_budget.can_deep_think():
            return "deep_think"
        return "reply"

    async def _make_decision(self, ctx: PlanningContext) -> PlanningDecision:
        """LLM决策"""
        budget = ctx.thinking_budget
        prompt = self._build_decision_prompt(ctx)
        response = await self._call_lightweight_llm(prompt)
        decision = self._parse_decision(response, ctx)
        if decision.action == ActionType.PROACTIVE_SPEAK:
            decision.thinking_consumed = budget.cost_per_proactive
        elif decision.action == ActionType.DEEP_THINK:
            decision.thinking_consumed = budget.cost_per_deep_think
        else:
            decision.thinking_consumed = budget.cost_per_planning
        decision.internal_state = (
            f"思考值={
                budget.thinking_value:.0f}, 聊天值={
                ctx.chat_value:.0f}, "
            f"活跃度={
                ctx.activity_level:.0f}, 社交值={
                    ctx.social_value:.0f}, 沉默={
                        ctx.silence_seconds:.0f}s"
        )
        return decision

    def _build_decision_prompt(self, ctx: PlanningContext) -> str:
        """构建决策提示词"""
        from src.config.config import global_config
        from src.person_info.bot_identity import get_bot_identity_manager

        context_summary = self._summarize_context(ctx.filtered_context)
        budget = ctx.thinking_budget
        relationship_label = ctx.custom_label or self._get_level_description(
            ctx.relationship_level
        )
        identity_manager = get_bot_identity_manager()
        identity_prompt = identity_manager.get_identity_prompt()
        reply_style = str(
            getattr(global_config.personality, "reply_style", "") or ""
        ).strip()
        plan_style = str(
            getattr(global_config.personality, "plan_style", "") or ""
        ).strip()
        relationship_candidates = (
            ", ".join(
                f"{candidate_id}({weight:.2f})"
                for candidate_id, weight in ctx.relationship_candidates[:5]
            )
            if ctx.relationship_candidates
            else "无"
        )
        structured_intentions = (
            "\n".join(
                (
                    f"- [{str(intent.get('kind', 'unknown') or 'unknown')}] "
                    f"urgency={float(intent.get('effective_urgency', 0.0) or 0.0):.2f} "
                    f"target={str(intent.get('target_user', '') or 'any')[:8]} "
                    f"src={str(intent.get('source', '') or 'unknown')} "
                    f"desc={str(intent.get('description', '') or '')[:60]}"
                )
                for intent in ctx.active_intentions[:5]
            )
            if ctx.active_intentions
            else "(当前无结构化活跃意图)"
        )
        persona_block = (
            "\n".join(
                part
                for part in [
                    identity_prompt,
                    f"- 表达基调: {reply_style}" if reply_style else "",
                    f"- 行动准则: {plan_style}" if plan_style else "",
                ]
                if part
            )
            or "- 人设要求: 维持自然、口语化、像群成员一样接话"
        )
        return f"""你现在就在这个聊天现场里，需要按自己的主人设判断要不要接话、怎么接话。

## 角色约束

{persona_block}

## 当前状态

    - 思考值: {budget.thinking_value:.0f}/{budget.thinking_ceiling:.0f} ({budget.ratio():.0%})
- 聊天值: {ctx.chat_value:.1f}
- 活跃度: {ctx.activity_level:.1f}
- 群聊沉默: {ctx.silence_seconds:.0f} 秒
- 社交值: {ctx.social_value:.1f}
- 信任值: {ctx.trust_value:.1f}
- 厌烦值: {ctx.annoyance_value:.1f}
- 好感值: {ctx.affection:.1f}
- 心理压力: {ctx.psychological_pressure:.1f}
- 创伤值: {ctx.trauma_score:.2f}
- 重复话题压力: {ctx.repeated_topic_pressure:.2f}
- 对当前这位的印象: {relationship_label}
- 当前活跃成员: {', '.join(ctx.active_users[:6]) if ctx.active_users else '无'}
- 当前话题线索: {', '.join(ctx.current_topics[:5]) if ctx.current_topics else '无'}
- 关系网候选对象: {relationship_candidates}
- 当前行为分类: {ctx.behavior_category}
- 当前行为强度: {ctx.behavior_severity:.2f}
- 是否刚接触这位: {'是' if ctx.is_new_user else '否'}
- 行为分类原因: {ctx.behavior_reason or '未知'}
- 对方最近原话/线索: {ctx.last_user_intent or '未知'}
- 相关记忆提示: {ctx.memory_hint or '无'}

## 可用上下文

{context_summary}

## 前情执行依据

{ctx.context_execution_block or '(无额外前情执行块)'}

## 群体行为模式（场景感知）

{ctx.group_pattern_hint or '(未检测到特定群体模式)'}

## 跨轮意图

{ctx.intention_hint or '(当前无活跃意图)'}

## 结构化活跃意图（优先用于动作竞争）

{structured_intentions}

## 可选行动

1. **observe** - 继续观察，不行动
2. **reply** - 回复最近的消息
3. **proactive_speak** - 主动发言（消耗较多思考值）
4. **join_conversation** - 融入对话
5. **check_later** - 稍后再检查
6. **wait** - 等待一段时间，可被新消息打断
7. **no_reply** - 明确选择本轮不回复
8. **complete_talk** - 结束当前轮次，等待下一次新消息
9. **deep_think** - 深度思考（消耗大量思考值）
10. **rest** - 休息（思考值不足时）

## 决策原则

- 思考值 < 20%: 选择 rest
- 聊天值 < 25: 避免频繁接话，优先 no_reply/wait
- 活跃度 < 20: 降低介入频率，优先 wait
- 思考值 < 40%: 避免 proactive_speak 和 deep_think
- 沉默 < 60秒: 通常选择 wait 或 no_reply
- 重复话题压力 > 0.5: 优先 wait/no_reply，拉长 next_check_seconds
- 厌烦值 > 30、聊天值过低或思考压力过高: 避免主动推进，优先 wait/no_reply
- 当前行为分类为 harassing 且强度较高: 优先 wait/no_reply，避免正面升级
- 当前行为分类为 hostile 且强度高: 降低主动接话倾向，优先克制或短回复
- 当前行为分类为 friendly 且不是刚接触这位: 可适度提高 reply / join_conversation 倾向
- 如果有相关记忆提示，优先把当前短句理解为前情延续，不要误判成全新话题
- 如果决定 reply 或 join_conversation，优先考虑最近发言者，以及关系网候选对象里仍在当前活跃范围内且权重更高的对象
- 沉默 > 300秒: 可以考虑 proactive_speak
- 如果本轮已经完成承接，且更适合等待新消息再继续，可以选择 complete_talk
- 聊天值、活跃度、社交值整体较高且有重要上下文: 可以考虑 reply 或 join_conversation
- 输出理由时不要把自己写成系统、助手、监控器，要像在心里判断自己要不要接这句。

## 意图裁决（必须执行）

如果上方「结构化活跃意图」非空，你**必须**在 thinking 中逐条回应：
- 对每个活跃意图说明：承接/推迟/放弃/完成，以及理由
- urgency >= 0.70 的意图：除非有更强理由否则应优先选择 proactive_speak 或 reply 来响应该意图
- 如果你的 action 与最高紧迫意图的方向矛盾，必须在 thinking 中明确解释为什么压过它
- 不允许完全忽略结构化活跃意图，即使最终选择 wait/no_reply 也必须说明对意图的处理态度

请用JSON格式返回决策：

```json
{{
    "action": "observe/reply/proactive_speak/join_conversation/check_later/wait/no_reply/complete_talk/deep_think/rest",
    "reason": "决策原因",
    "content_plan": "如果发言，计划说什么",
    "target_user_id": "如果应优先接某个对象，填其对象标识，没有则留空",
    "preferred_reply_style": "可选：direct/quote",
    "preferred_quote": true,
    "next_check_seconds": 60,
    "thinking": "你的思考过程"
}}
```

直接输出JSON。"""

    def _get_level_description(self, level: int) -> str:
        """获取关系等级的默认描述"""
        descriptions = {
            0: "关系紧张",
            1: "不太熟悉",
            2: "有些熟悉",
            3: "比较熟悉",
            4: "关系亲密",
            5: "非常信任",
        }
        return descriptions.get(level, "有些熟悉")

    def _summarize_context(self, segments: List[ContextSegment]) -> str:
        """总结上下文"""
        if not segments:
            return "(无可用上下文)"
        lines = []
        for i, seg in enumerate(segments[:10]):
            source = "消息" if seg.source == "recent_message" else "记忆"
            lines.append(f"{i + 1}. [{source}] {seg.content[:80]}...")
        return "\n".join(lines)

    def _pick_priority_target(self, ctx: PlanningContext) -> str:
        """从最近发言者与关系网候选中挑出当前优先关注对象。"""
        active_user_set = {
            str(user_id or "").strip()
            for user_id in ctx.active_users
            if str(user_id or "").strip()
        }
        for candidate_id, _weight in ctx.relationship_candidates:
            normalized_id = str(candidate_id or "").strip()
            if normalized_id and normalized_id in active_user_set:
                return normalized_id
        for message in reversed(ctx.recent_messages):
            speaker = str(message.get("speaker", "") or "").strip()
            if speaker and speaker.lower() != "bot":
                return speaker
        if ctx.user_id:
            return str(ctx.user_id)
        return ""

    def _build_default_content_plan(
        self, ctx: PlanningContext, action: ActionType
    ) -> str:
        """在 LLM 缺少稳定输出时，给出结构化默认内容规划。"""
        target_user = self._pick_priority_target(ctx)
        if action == ActionType.PROACTIVE_SPEAK:
            if target_user:
                return f"优先围绕 {target_user} 刚才的语境自然接话，避免偏题硬切。"
            return "基于当前话题自然接入，不强行打断现有节奏。"
        if action in (ActionType.REPLY, ActionType.JOIN_CONVERSATION):
            if target_user:
                return f"优先回应 {target_user} 当前这轮表达，并保持和当前话题连续。"
            return "优先回应最近一条有效表达，保持和当前话题连续。"
        return ""

    def _extract_target_user_from_plan(
        self, content_plan: str, ctx: PlanningContext
    ) -> str:
        normalized_plan = str(content_plan or "").strip()
        if normalized_plan:
            for candidate_id, _weight in ctx.relationship_candidates:
                normalized_id = str(candidate_id or "").strip()
                if normalized_id and normalized_id in normalized_plan:
                    return normalized_id

            for message in reversed(ctx.recent_messages):
                speaker = str(message.get("speaker", "") or "").strip()
                if (
                    speaker
                    and speaker.lower() != "bot"
                    and speaker in normalized_plan
                ):
                    return speaker

        return self._pick_priority_target(ctx)

    def _build_fallback_decision(
        self, ctx: PlanningContext, response: str = ""
    ) -> PlanningDecision:
        """当 LLM 不可用或输出不可解析时，使用代码级状态回退。"""
        target_user = self._pick_priority_target(ctx)
        if (
            ctx.thinking_budget.thinking_value
            < ctx.thinking_budget.min_for_planning
        ):
            return PlanningDecision(
                action=ActionType.REST,
                reason="思考值不足，进入保守回退",
                target_user_id=target_user,
                next_check_seconds=120,
            )
        if (
            ctx.behavior_category == "harassing"
            and ctx.behavior_severity >= 0.6
        ):
            return PlanningDecision(
                action=ActionType.WAIT,
                reason="高压骚扰信号，回退为延迟观察",
                target_user_id=target_user,
                next_check_seconds=180,
            )
        if (
            ctx.repeated_topic_pressure > 0.5
            or ctx.annoyance_value > 30
            or ctx.chat_value < 25
        ):
            return PlanningDecision(
                action=ActionType.NO_REPLY,
                reason="当前压力或重复信号偏高，回退为稍后检查",
                target_user_id=target_user,
                next_check_seconds=120,
            )
        if ctx.silence_seconds > 300 and ctx.thinking_budget.can_proactive():
            return PlanningDecision(
                action=ActionType.PROACTIVE_SPEAK,
                reason=(
                    f"沉默较久，优先面向 {target_user} 重新接入对话"
                    if target_user
                    else "沉默较久，尝试重新接入对话"
                ),
                content_plan=self._build_default_content_plan(
                    ctx, ActionType.PROACTIVE_SPEAK
                ),
                target_user_id=target_user,
                next_check_seconds=120,
                confidence=0.55,
            )
        action = (
            ActionType.REPLY if target_user else ActionType.JOIN_CONVERSATION
        )
        fallback_reason = "代码级回退：优先延续最近活跃目标"
        if response:
            fallback_reason += "，LLM输出不可解析"
        return PlanningDecision(
            action=action,
            reason=(
                f"{fallback_reason}，当前目标={target_user}"
                if target_user
                else fallback_reason
            ),
            content_plan=self._build_default_content_plan(ctx, action),
            target_user_id=target_user,
            next_check_seconds=90,
            confidence=0.55,
        )

    async def _call_lightweight_llm(self, prompt: str) -> str:
        """调用规划模型 - 使用 planner 配置，与 lightweight 分开处理"""
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            task_config = model_config.model_task_config.planner
            if task_config is None:
                task_config = model_config.model_task_config.lightweight
            if task_config is None:
                logger.warning(
                    "[统一规划] 未找到 planner 或 lightweight 任务配置"
                )
                return ""
            llm = LLMRequest(task_config, "unified_planning")
            response, _ = await llm.generate_response_async(
                prompt,
                temperature=0.7,
            )
            return response or ""
        except Exception as e:
            logger.error(f"[统一规划] LLM调用失败: {e}")
            return ""

    def _parse_decision(
        self, response: str, ctx: PlanningContext
    ) -> PlanningDecision:
        """解析决策"""
        import json
        import re

        top_intent: Optional[Dict[str, Any]] = (
            ctx.active_intentions[0] if ctx.active_intentions else None
        )
        top_intent_id = str((top_intent or {}).get("intent_id", "") or "")
        top_intent_kind = str((top_intent or {}).get("kind", "") or "")
        top_intent_urgency = float(
            (top_intent or {}).get("effective_urgency", 0.0) or 0.0
        )
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                action_str = data.get("action", "observe")
                try:
                    action = ActionType(action_str)
                except ValueError:
                    action = ActionType.OBSERVE
                content_plan = str(data.get("content_plan", "") or "").strip()
                if not content_plan:
                    content_plan = self._build_default_content_plan(
                        ctx, action
                    )
                target_user_id = str(
                    data.get("target_user_id", "") or ""
                ).strip()
                if not target_user_id:
                    target_user_id = self._extract_target_user_from_plan(
                        content_plan, ctx
                    )
                reason = str(data.get("reason", "") or "").strip()
                model_confidence = 0.7
                if action in (
                    ActionType.WAIT,
                    ActionType.NO_REPLY,
                    ActionType.OBSERVE,
                ):
                    urgency_bar = 0.75
                    if top_intent_kind in (
                        "reengage_user",
                        "followup_question",
                    ):
                        urgency_bar = 0.68
                    elif top_intent_kind in (
                        "clarify_misunderstanding",
                        "wait_for_reply",
                    ):
                        urgency_bar = 0.72
                    elif top_intent_kind in ("self_expression",):
                        urgency_bar = 0.82
                    if (
                        top_intent_urgency >= urgency_bar
                        and ctx.silence_seconds >= 120
                    ):
                        action = (
                            ActionType.JOIN_CONVERSATION
                            if target_user_id
                            else ActionType.PROACTIVE_SPEAK
                        )
                        model_confidence = min(0.88, model_confidence + 0.12)
                        reason = (reason + " | " if reason else "") + (
                            f"意图竞争加权触发: {top_intent_kind} urgency={
                                top_intent_urgency:.2f} bar={
                                urgency_bar:.2f}"
                        )
                        if not content_plan:
                            content_plan = self._build_default_content_plan(
                                ctx, action
                            )
                return PlanningDecision(
                    action=action,
                    reason=reason,
                    content_plan=content_plan,
                    target_user_id=target_user_id,
                    preferred_reply_style=str(
                        data.get("preferred_reply_style", "") or ""
                    ).strip(),
                    preferred_quote=(
                        data.get("preferred_quote")
                        if isinstance(data.get("preferred_quote"), bool)
                        else None
                    ),
                    next_check_seconds=float(
                        data.get("next_check_seconds", 120)
                    ),
                    confidence=model_confidence,
                    planning_process=data.get("thinking", ""),
                    context_used=[
                        s.segment_id for s in ctx.filtered_context[:5]
                    ],
                    matched_intent_id=top_intent_id,
                    matched_intent_kind=top_intent_kind,
                    matched_intent_urgency=top_intent_urgency,
                )
            except json.JSONDecodeError:
                pass
        if "主动" in response or "发言" in response:
            return PlanningDecision(
                action=ActionType.PROACTIVE_SPEAK,
                reason="从响应推断",
                content_plan=self._build_default_content_plan(
                    ctx, ActionType.PROACTIVE_SPEAK
                ),
                target_user_id=self._pick_priority_target(ctx),
                next_check_seconds=120,
            )
        if "等待" in response or "稍后" in response:
            return PlanningDecision(
                action=ActionType.WAIT,
                reason="从响应推断为等待",
                target_user_id=self._pick_priority_target(ctx),
                next_check_seconds=120,
            )
        if "结束" in response or "收尾" in response:
            return PlanningDecision(
                action=ActionType.COMPLETE_TALK,
                reason="从响应推断为结束当前轮次",
                target_user_id=self._pick_priority_target(ctx),
                next_check_seconds=120,
            )
        return self._build_fallback_decision(ctx, response=response)

    async def _execute_decision(
        self, channel_id: str, decision: PlanningDecision
    ) -> None:
        """执行决策"""
        action_value = decision.action.value
        if action_value in self._action_callbacks:
            callback = self._action_callbacks[action_value]
            try:
                await callback(channel_id, decision)
            except Exception as e:
                logger.error(f"[统一规划] 执行回调失败: {e}")
        else:
            logger.info(
                f"[统一规划] {channel_id[:8]} 执行: {decision.action.label()}, "
                f"原因: {decision.reason[:50] if decision.reason else '无'}, "
                f"消耗思考值: {decision.thinking_consumed:.1f}"
            )

    def _record_decision(
        self, channel_id: str, decision: PlanningDecision
    ) -> None:
        """记录决策历史"""
        if channel_id not in self._decision_history:
            self._decision_history[channel_id] = []
        self._decision_history[channel_id].append(decision)
        if decision.matched_intent_id:
            logger.info(
                f"[统一规划] {channel_id[:8]} 命中意图 "
                f"id={decision.matched_intent_id[:10]} "
                f"kind={decision.matched_intent_kind or 'unknown'} "
                f"urgency={decision.matched_intent_urgency:.2f} "
                f"action={decision.action.label()}"
            )
        if len(self._decision_history[channel_id]) > self._max_history:
            self._decision_history[channel_id] = self._decision_history[
                channel_id
            ][-self._max_history:]

    def get_status(self, channel_id: str) -> Dict[str, Any]:
        """获取状态"""
        state = self._channel_states.get(channel_id, PlanningPhase.IDLE)
        ctx = self._channel_contexts.get(channel_id)
        budget = ctx.thinking_budget if ctx else ThinkingBudget()
        history = self._decision_history.get(channel_id, [])
        return {
            "phase": state.value,
            "thinking_budget": budget.thinking_value,
            "chat_value": ctx.chat_value if ctx else 100.0,
            "activity_level": ctx.activity_level if ctx else 50.0,
            "thinking_ratio": budget.ratio(),
            "can_plan": budget.can_plan(),
            "can_proactive": budget.can_proactive(),
            "can_deep_think": budget.can_deep_think(),
            "decision_count": len(history),
            "last_action": history[-1].action.value if history else None,
        }

    def get_context_stats(self, channel_id: str) -> Dict[str, Any]:
        """获取上下文统计"""
        ctx = self._channel_contexts.get(channel_id)
        if not ctx:
            return {"error": "no_context"}
        return {
            "short_term_count": len(ctx.short_term_context),
            "long_term_count": len(ctx.long_term_context),
            "filtered_count": len(ctx.filtered_context),
            "silence_seconds": ctx.silence_seconds,
            "chat_value": ctx.chat_value,
            "activity_level": ctx.activity_level,
            "social_value": ctx.social_value,
        }


_unified_planner: Optional[UnifiedPlanner] = None


def get_unified_planner() -> UnifiedPlanner:
    """获取统一规划系统单例"""
    global _unified_planner
    if _unified_planner is None:
        _unified_planner = UnifiedPlanner()
    return _unified_planner

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("理解门控")

_gate_singleton = None


class ComprehensionLevel(Enum):
    """理解层级"""

    CANNOT_UNDERSTAND = "cannot_understand"
    CAN_UNDERSTAND_NO_REPLY = "can_understand_no_reply"
    CAN_REPLY_SIMPLE = "can_reply_simple"
    CAN_REPLY_DEEP = "can_reply_deep"
    NEED_CLARIFICATION = "need_clarification"

    def should_reply(self) -> bool:
        return self in (
            ComprehensionLevel.CAN_REPLY_SIMPLE,
            ComprehensionLevel.CAN_REPLY_DEEP,
        )

    def label(self) -> str:
        _labels = {
            "cannot_understand": "看不懂",
            "can_understand_no_reply": "懂了但不回",
            "can_reply_simple": "能简单回",
            "can_reply_deep": "值得深聊",
            "need_clarification": "需要追问",
        }
        return _labels.get(self.value, "")


@dataclass
class UnderstandingResult:
    """理解门控判定结果"""

    level: ComprehensionLevel = ComprehensionLevel.CAN_UNDERSTAND_NO_REPLY
    comprehension_score: float = 0.5
    reply_worthiness: float = 0.0
    upgrade_analysis: bool = False
    misunderstanding_risk: float = 0.0
    reasons: List[str] = field(default_factory=list)
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "level_label": self.level.label(),
            "comprehension_score": round(self.comprehension_score, 3),
            "reply_worthiness": round(self.reply_worthiness, 3),
            "upgrade_analysis": self.upgrade_analysis,
            "misunderstanding_risk": round(self.misunderstanding_risk, 3),
            "reasons": self.reasons[:5],
        }


@dataclass
class MisunderstandingSignal:
    """误解风险信号"""

    risk_score: float = 0.0
    detected_cues: List[str] = field(default_factory=list)
    correction_fragment: str = ""
    signal_type: str = ""
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 3),
            "detected_cues": self.detected_cues[:5],
            "signal_type": self.signal_type,
        }


# ═══════════════════════════════════════════════════════
#  小模型预理解层
# ═══════════════════════════════════════════════════════


@dataclass
class PreUnderstandingSpec:
    """预理解层输入规格
    定义了轻量模型在主LLM之前需要处理的输入字段。
    调用者构造此规格后交给预理解层执行。
    """

    raw_text: str = ""
    message_length: int = 0
    has_media: bool = False
    is_reply: bool = False
    sender_tier: str = "cold"
    channel_activity_level: int = 0
    recent_bot_replied: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text_len": self.message_length or len(self.raw_text),
            "has_media": self.has_media,
            "is_reply": self.is_reply,
            "sender_tier": self.sender_tier,
            "channel_activity": self.channel_activity_level,
            "recent_bot_replied": self.recent_bot_replied,
        }


@dataclass
class PreUnderstandingResult:
    """预理解层输出
    轻量模型/规则引擎的快速判断结果，供后续深度理解门控参考。
    """

    intent_tag: str = ""
    sentiment_polarity: float = 0.0
    topic_hint: str = ""
    complexity_estimate: float = 0.3
    skip_deep_analysis: bool = False
    skip_reason: str = ""
    extra_signals: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_tag": self.intent_tag,
            "sentiment_polarity": round(self.sentiment_polarity, 3),
            "topic_hint": self.topic_hint,
            "complexity_estimate": round(self.complexity_estimate, 3),
            "skip_deep_analysis": self.skip_deep_analysis,
            "skip_reason": self.skip_reason,
            "latency_ms": round(self.latency_ms, 1),
        }


# 意图标签常量
_INTENT_GREETING = "greeting"
_INTENT_QUESTION = "question"
_INTENT_COMMAND = "command"
_INTENT_CHITCHAT = "chitchat"
_INTENT_EMOTION = "emotion_expression"
_INTENT_MEDIA_SHARE = "media_share"
_INTENT_UNKNOWN = "unknown"

# 意图判定关键词组
_GREETING_PATTERNS = {
    "早",
    "晚安",
    "你好",
    "在吗",
    "hello",
    "hi",
    "嗨",
    "早上好",
    "下午好",
    "晚上好",
}
_QUESTION_PATTERNS = {
    "吗",
    "呢",
    "？",
    "?",
    "怎么",
    "什么",
    "为什么",
    "哪",
    "谁",
    "几",
    "多少",
    "如何",
}
_COMMAND_PATTERNS = {
    "帮我",
    "请",
    "能不能",
    "可以",
    "给我",
    "发一下",
    "查一下",
    "看看",
}

# 情感极性关键词（简化版）
_POSITIVE_SENTIMENT_HINTS = {
    "开心",
    "高兴",
    "棒",
    "好",
    "赞",
    "牛",
    "厉害",
    "加油",
    "漂亮",
    "感谢",
    "谢谢",
}
_NEGATIVE_SENTIMENT_HINTS = {
    "难过",
    "烦",
    "累",
    "哭",
    "怒",
    "差",
    "垃圾",
    "讨厌",
    "无聊",
    "失望",
    "生气",
}


class PreUnderstandingEngine:
    """规则驱动的预理解引擎
    在主LLM调用前，以极低成本完成：
    1. 意图粗分类（问候/提问/指令/闲聊/情绪表达/媒体分享）
    2. 情感极性速判（正/负/中性）
    3. 复杂度估算（决定是否需要深度分析）
    4. 快速跳过判定（明确不需要LLM的场景）
    可选：挂载外部小模型回调替换规则引擎。
    """

    def __init__(self):
        self._external_model: Optional[
            Callable[[PreUnderstandingSpec], PreUnderstandingResult]
        ] = None
        self._call_count = 0
        self._skip_count = 0
        self._total_latency_ms = 0.0

    def register_external_model(
        self,
        model_fn: Callable[[PreUnderstandingSpec], PreUnderstandingResult],
    ) -> None:
        """注册外部小模型回调。注册后 pre_evaluate 会优先调用此回调。"""
        self._external_model = model_fn
        logger.debug("[预理解] 已注册外部小模型回调")

    def unregister_external_model(self) -> None:
        self._external_model = None

    def pre_evaluate(
        self, spec: PreUnderstandingSpec
    ) -> PreUnderstandingResult:
        """执行预理解。若已注册外部模型则优先使用，否则走内置规则引擎。"""
        start = time.monotonic()
        self._call_count += 1
        if self._external_model is not None:
            try:
                result = self._external_model(spec)
                result.latency_ms = (time.monotonic() - start) * 1000
                self._total_latency_ms += result.latency_ms
                if result.skip_deep_analysis:
                    self._skip_count += 1
                return result
            except Exception as exc:
                logger.warning(f"[预理解] 外部模型异常，回退规则引擎: {exc}")
        result = self._rule_based_evaluate(spec)
        result.latency_ms = (time.monotonic() - start) * 1000
        self._total_latency_ms += result.latency_ms
        if result.skip_deep_analysis:
            self._skip_count += 1
        return result

    def _rule_based_evaluate(
        self, spec: PreUnderstandingSpec
    ) -> PreUnderstandingResult:
        """内置规则引擎：零模型依赖的快速预理解"""
        result = PreUnderstandingResult()
        text = spec.raw_text.strip()
        text_lower = text.lower()
        text_len = len(text)
        # 意图粗分类
        result.intent_tag = self._classify_intent(text_lower, spec.has_media)
        # 情感极性
        result.sentiment_polarity = self._estimate_sentiment(text_lower)
        # 复杂度估算
        result.complexity_estimate = self._estimate_complexity(
            text_len=text_len,
            has_media=spec.has_media,
            is_reply=spec.is_reply,
            intent=result.intent_tag,
        )
        # 快速跳过判定
        if self._should_skip(spec, result, text_len):
            result.skip_deep_analysis = True
        return result

    def _classify_intent(self, text_lower: str, has_media: bool) -> str:
        """规则意图分类"""
        if has_media and len(text_lower) < 5:
            return _INTENT_MEDIA_SHARE
        for kw in _GREETING_PATTERNS:
            if kw in text_lower:
                return _INTENT_GREETING
        question_score = sum(
            1 for kw in _QUESTION_PATTERNS if kw in text_lower
        )
        if question_score >= 2:
            return _INTENT_QUESTION
        command_score = sum(1 for kw in _COMMAND_PATTERNS if kw in text_lower)
        if command_score >= 1:
            return _INTENT_COMMAND
        # 纯情绪表达：文本短 + 含情绪词
        sentiment = self._estimate_sentiment(text_lower)
        if abs(sentiment) > 0.3 and len(text_lower) < 20:
            return _INTENT_EMOTION
        if question_score == 1:
            return _INTENT_QUESTION
        if len(text_lower) > 3:
            return _INTENT_CHITCHAT
        return _INTENT_UNKNOWN

    @staticmethod
    def _estimate_sentiment(text_lower: str) -> float:
        """简单正负极性估算 [-1, 1]"""
        pos_hits = sum(
            1 for kw in _POSITIVE_SENTIMENT_HINTS if kw in text_lower
        )
        neg_hits = sum(
            1 for kw in _NEGATIVE_SENTIMENT_HINTS if kw in text_lower
        )
        total = pos_hits + neg_hits
        if total == 0:
            return 0.0
        return (pos_hits - neg_hits) / max(1, total)

    @staticmethod
    def _estimate_complexity(
        text_len: int,
        has_media: bool,
        is_reply: bool,
        intent: str,
    ) -> float:
        """复杂度估算 [0, 1]"""
        base = 0.2
        if text_len > 200:
            base += 0.3
        elif text_len > 50:
            base += 0.15
        elif text_len < 5:
            base -= 0.1
        if has_media:
            base += 0.15
        if is_reply:
            base += 0.05
        if intent == _INTENT_QUESTION:
            base += 0.15
        elif intent == _INTENT_COMMAND:
            base += 0.1
        elif intent in (_INTENT_GREETING, _INTENT_EMOTION):
            base -= 0.1
        return max(0.0, min(1.0, base))

    @staticmethod
    def _should_skip(
        spec: PreUnderstandingSpec,
        result: PreUnderstandingResult,
        text_len: int,
    ) -> bool:
        """判定是否可跳过深度分析"""
        # 纯问候 + 短文本 → 不需要LLM
        if result.intent_tag == _INTENT_GREETING and text_len < 10:
            result.skip_reason = "简短问候"
            return True
        # 纯媒体分享无文字 → 交给多模态预算器
        if result.intent_tag == _INTENT_MEDIA_SHARE and text_len < 3:
            result.skip_reason = "纯媒体无文字"
            return True
        # 极短模糊输入
        if text_len <= 1 and not spec.has_media:
            result.skip_reason = "内容过短"
            return True
        # 频道极度繁忙 + 低复杂度 + 非bot相关 → 节省算力
        if (
            spec.channel_activity_level > 20
            and result.complexity_estimate < 0.25
            and not spec.recent_bot_replied
        ):
            result.skip_reason = "高流量低复杂度"
            return True
        return False

    def engine_stats(self) -> Dict[str, Any]:
        """引擎统计"""
        return {
            "total_calls": self._call_count,
            "skip_count": self._skip_count,
            "skip_rate": round(self._skip_count / max(1, self._call_count), 3),
            "avg_latency_ms": round(
                self._total_latency_ms / max(1, self._call_count), 2
            ),
            "external_model_loaded": self._external_model is not None,
        }


class UnderstandingGate:
    """理解门控
    在消息通过可见性层之后，判断主体能否理解该消息、
    是否值得回复、是否需要升级到深度分析链。
    """

    def __init__(self):
        self._recent_results: List[UnderstandingResult] = []
        self._max_cache = 30

    def evaluate(
        self,
        *,
        text_content: str = "",
        message_category: str = "plain_text",
        visibility_score: float = 0.5,
        user_relationship_level: int = 2,
        user_trust: float = 0.0,
        topic_familiarity: float = 0.5,
        bot_relevant: bool = False,
        is_question: bool = False,
        has_media: bool = False,
        energy_ratio: float = 1.0,
        social_willingness: float = 0.5,
    ) -> UnderstandingResult:
        """判断消息理解层级"""
        reasons: List[str] = []
        # 理解分数：基础 + 话题熟悉度 + 可见性
        comprehension = 0.3 + topic_familiarity * 0.35 + visibility_score * 0.2
        text_len = len(text_content)
        if text_len < 2:
            comprehension -= 0.15
            reasons.append("内容过短")
        elif text_len > 500:
            comprehension += 0.1
            reasons.append("长消息")
        if has_media:
            comprehension -= 0.05
            reasons.append("含媒体需额外理解")
        comprehension = max(0.0, min(1.0, comprehension))
        # 回复价值分数
        worthiness = 0.15  # 基础回复价值，确保简单消息不会被完全忽略
        if bot_relevant:
            worthiness += 0.35
            reasons.append("与bot相关")
        if is_question:
            worthiness += 0.25
            reasons.append("提问")
        if user_relationship_level >= 4:
            worthiness += 0.15
            reasons.append("亲密用户")
        elif user_relationship_level >= 3:
            worthiness += 0.08
        if social_willingness > 0.6:
            worthiness += 0.1
            reasons.append("社交意愿高")
        if visibility_score > 0.6:
            worthiness += 0.08
        # 简单问候/确认类消息检测
        text_lower = text_content.lower().strip()
        simple_greetings = ["在", "在吗", "在不在", "呢", "？", "?", "哈", "哈哈", "哈哈哈", "嗯", "哦", "好", "好的", "ok", "嗯嗯", "行", "可以"]
        if text_lower in simple_greetings or len(text_lower) <= 2:
            worthiness += 0.12
            reasons.append("简单问候/确认")
        # 精力影响
        if energy_ratio < 0.3:
            worthiness -= 0.15
            reasons.append("精力不足")
        worthiness = max(0.0, min(1.0, worthiness))
        # 判定层级
        need_upgrade = False
        if comprehension < 0.25:
            level = ComprehensionLevel.CANNOT_UNDERSTAND
            reasons.append("理解度低")
        elif worthiness < 0.12:
            level = ComprehensionLevel.CAN_UNDERSTAND_NO_REPLY
            reasons.append("不值得回复")
        elif worthiness > 0.6 and energy_ratio > 0.4:
            level = ComprehensionLevel.CAN_REPLY_DEEP
            need_upgrade = True
            reasons.append("值得深入")
        elif worthiness > 0.3:
            level = ComprehensionLevel.CAN_REPLY_SIMPLE
        else:
            level = ComprehensionLevel.CAN_UNDERSTAND_NO_REPLY
        result = UnderstandingResult(
            level=level,
            comprehension_score=comprehension,
            reply_worthiness=worthiness,
            upgrade_analysis=need_upgrade,
            reasons=reasons,
        )
        self._recent_results.append(result)
        if len(self._recent_results) > self._max_cache:
            self._recent_results = self._recent_results[-self._max_cache:]
        return result

    def batch_evaluate(
        self,
        messages_with_context: List[Dict[str, Any]],
    ) -> List[UnderstandingResult]:
        """批量评估消息理解层级
        每个元素应包含: text_content, message_category, visibility_score 等字段
        """
        results = []
        for ctx in messages_with_context:
            result = self.evaluate(**ctx)
            results.append(result)
        return results

    def any_worth_replying(self, results: List[UnderstandingResult]) -> bool:
        """批次中是否有值得回复的消息"""
        return any(r.level.should_reply() for r in results)

    # ────────────────── 误解风险评估 ──────────────────
    # 纠正型关键词（用户在修正 bot 的错误理解）
    _CORRECTION_CUES = [
        "不是这个意思",
        "你理解错了",
        "你搞错了",
        "不对不对",
        "我说的是",
        "我的意思是",
        "不是那个意思",
        "你误解了",
        "想说的是",
        "搞混了吧",
        "弄错了",
        "不是这样",
        "我不是说",
        "你搞岔了",
        "答非所问",
    ]
    # 拒绝型信号（用户不满 bot 回复的表达）
    _REJECTION_CUES = [
        "啊？",
        "？？？",
        "什么鬼",
        "无语",
        "离谱",
        "牛头不对马嘴",
        "没说这个",
        "跑题了",
        "别乱说",
        "瞎说",
        "胡说",
    ]
    # 困惑型短句（用户对 bot 回复表示费解）
    _CONFUSION_SHORT = {"?", "？", "??", "？？", "???", "啊？", "嗯？", "啥"}

    def evaluate_misunderstanding_risk(
        self,
        *,
        bot_reply_text: str = "",
        user_followup_text: str = "",
        is_repeated_question: bool = False,
        reply_was_rejected: bool = False,
        time_gap_sec: float = 60.0,
    ) -> MisunderstandingSignal:
        """评估 bot 上一条回复被误解的风险
        从用户后续消息中检测纠正、拒绝、困惑信号。
        time_gap_sec 越短，信号越有参考价值（排除跨话题噪声）。
        """
        risk = 0.0
        cues: List[str] = []
        signal_type = ""
        correction_frag = ""
        followup_stripped = user_followup_text.strip()
        followup_lower = followup_stripped.lower()
        # 时间衰减系数：间隔越长，风险衰减越大
        time_factor = (
            max(0.3, 1.0 - time_gap_sec / 300.0) if time_gap_sec > 0 else 1.0
        )
        # 纠正型检测
        for kw in self._CORRECTION_CUES:
            if kw in followup_lower:
                risk += 0.35
                cues.append(f"纠正:{kw}")
                signal_type = "correction"
                # 尝试提取纠正内容
                _idx = followup_lower.find(kw)
                _after = followup_stripped[_idx + len(kw):].strip()
                if _after:
                    correction_frag = _after[:60]
                break
        # 拒绝型检测
        for kw in self._REJECTION_CUES:
            if kw in followup_lower:
                risk += 0.25
                cues.append(f"拒绝:{kw}")
                if not signal_type:
                    signal_type = "rejection"
                break
        # 困惑型短句检测（仅当后续消息很短时）
        if followup_stripped in self._CONFUSION_SHORT:
            risk += 0.20
            cues.append("困惑短句")
            if not signal_type:
                signal_type = "confusion"
        # 重复提问检测
        if is_repeated_question:
            risk += 0.30
            cues.append("重复提问")
            if not signal_type:
                signal_type = "repetition"
        # 回复被直接拒绝
        if reply_was_rejected:
            risk += 0.25
            cues.append("回复被拒")
            if not signal_type:
                signal_type = "explicit_rejection"
        # 打分修正
        risk = risk * time_factor
        # 跨不相关话题的误识别保护：如果用户后续文本很长且没有任何纠正信号，可能只是换话题
        if len(followup_stripped) > 100 and not cues:
            risk = max(0.0, risk - 0.1)
        risk = max(0.0, min(1.0, risk))
        signal = MisunderstandingSignal(
            risk_score=risk,
            detected_cues=cues,
            correction_fragment=correction_frag,
            signal_type=signal_type,
        )
        # 回填到最近理解结果
        if self._recent_results and risk > 0.15:
            self._recent_results[-1].misunderstanding_risk = risk
        if risk > 0.2 and cues:
            logger.debug(
                f"[理解门控] 误解风险={risk:.2f} 信号={signal_type} "
                f"线索={cues[:3]} 纠正片段={correction_frag[:30]}"
            )
        return signal


def get_understanding_gate() -> UnderstandingGate:
    global _gate_singleton
    if _gate_singleton is None:
        _gate_singleton = UnderstandingGate()
    return _gate_singleton


_pre_engine_singleton = None


def get_pre_understanding_engine() -> PreUnderstandingEngine:
    global _pre_engine_singleton
    if _pre_engine_singleton is None:
        _pre_engine_singleton = PreUnderstandingEngine()
    return _pre_engine_singleton

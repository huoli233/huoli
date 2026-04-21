import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from src.common.config.config_engine import ConfigEngine
from src.common.logger import get_logger

logger = get_logger("用户状态")


@dataclass
class RuleSignals:
    """规则信号"""

    avg_interval_sec: float = 0.0
    messages_per_hour: float = 0.0
    avg_message_length: float = 0.0
    user_initiation_ratio: float = 0.0
    contains_question_ratio: float = 0.0
    emoji_usage_ratio: float = 0.0
    reply_speed_avg: float = 0.0
    rule_score: float = 0.5
    rule_verdict: str = "uncertain"


@dataclass
class UserStateResult:
    """用户状态结果"""

    state: str = "uncertain"
    willingness: float = 0.5
    interest_level: float = 0.5
    attention_level: float = 0.5
    confidence: float = 0.5
    reasoning: str = ""
    rule_signals: Optional[RuleSignals] = None
    model_contribution: float = 0.0
    rule_contribution: float = 0.0


@dataclass
class ScoringThresholds:
    """评分阈值配置"""

    interval_tiers: Dict[str, float] = field(
        default_factory=lambda: {"tier1": 60, "tier2": 300, "tier3": 600}
    )
    interval_scores: Dict[str, float] = field(
        default_factory=lambda: {
            "tier1": 0.9,
            "tier2": 0.7,
            "tier3": 0.5,
            "tier4": 0.25,
        }
    )
    freq_tiers: Dict[str, float] = field(
        default_factory=lambda: {"tier1": 10, "tier2": 5, "tier3": 3}
    )
    freq_scores: Dict[str, float] = field(
        default_factory=lambda: {
            "tier1": 1.0,
            "tier2": 0.8,
            "tier3": 0.6,
            "tier4": 0.3,
        }
    )
    length_tiers: Dict[str, float] = field(
        default_factory=lambda: {"tier1": 30, "tier2": 10, "tier3": 5}
    )
    length_scores: Dict[str, float] = field(
        default_factory=lambda: {
            "tier1": 0.85,
            "tier2": 0.6,
            "tier3": 0.4,
            "tier4": 0.25,
        }
    )
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "interval": 0.25,
            "freq": 0.20,
            "length": 0.15,
            "init": 0.25,
            "question": 0.15,
        }
    )
    verdict_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"want_to_chat": 0.6, "not_want_to_chat": 0.35}
    )


@dataclass
class InteractionRecord:
    """交互记录"""

    timestamp: float = 0.0
    msg_length: int = 0
    has_question: bool = False
    is_user_initiated: bool = True
    has_emoji: bool = False
    reply_speed: float = 0.0


class _InteractionLog:
    """交互日志"""

    def __init__(self, window_size: int = 200):
        self._records: Deque[InteractionRecord] = deque(maxlen=window_size)
        self._thresholds: Optional[ScoringThresholds] = None

    def set_thresholds(self, thresholds: ScoringThresholds) -> None:
        self._thresholds = thresholds

    def append(
        self,
        msg_length: int,
        has_question: bool = False,
        is_user_initiated: bool = True,
        has_emoji: bool = False,
        reply_speed: float = 0.0,
    ) -> None:
        self._records.append(
            InteractionRecord(
                timestamp=time.time(),
                msg_length=msg_length,
                has_question=has_question,
                is_user_initiated=is_user_initiated,
                has_emoji=has_emoji,
                reply_speed=reply_speed,
            )
        )

    def compute_signals(self) -> RuleSignals:
        if not self._thresholds:
            return RuleSignals()

        n = len(self._records)
        if n == 0:
            return RuleSignals()

        records = list(self._records)
        intervals = [
            records[i].timestamp - records[i - 1].timestamp
            for i in range(1, n)
        ]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0.0

        now = time.time()
        one_hour_ago = now - 3600.0
        recent_count = sum(1 for r in records if r.timestamp >= one_hour_ago)
        freq = recent_count

        avg_len = sum(r.msg_length for r in records) / n
        init_ratio = sum(1 for r in records if r.is_user_initiated) / n
        q_ratio = sum(1 for r in records if r.has_question) / n
        emoji_ratio = sum(1 for r in records if r.has_emoji) / n
        reply_speeds = [r.reply_speed for r in records if r.reply_speed > 0]
        avg_reply_speed = (
            sum(reply_speeds) / len(reply_speeds) if reply_speeds else 0.0
        )

        score = self._evaluate(
            avg_interval, freq, avg_len, init_ratio, q_ratio
        )

        want_threshold = self._thresholds.verdict_thresholds.get(
            "want_to_chat", 0.6
        )
        not_want_threshold = self._thresholds.verdict_thresholds.get(
            "not_want_to_chat", 0.35
        )
        verdict = (
            "want_to_chat"
            if score >= want_threshold
            else (
                "not_want_to_chat"
                if score <= not_want_threshold
                else "uncertain"
            )
        )

        return RuleSignals(
            avg_interval_sec=round(avg_interval, 1),
            messages_per_hour=float(freq),
            avg_message_length=round(avg_len, 1),
            user_initiation_ratio=round(init_ratio, 3),
            contains_question_ratio=round(q_ratio, 3),
            emoji_usage_ratio=round(emoji_ratio, 3),
            reply_speed_avg=round(avg_reply_speed, 1),
            rule_score=round(score, 3),
            rule_verdict=verdict,
        )

    def _evaluate(
        self,
        interval: float,
        freq: float,
        avg_len: float,
        init_ratio: float,
        q_ratio: float,
    ) -> float:
        th = self._thresholds

        interval_tiers = th.interval_tiers
        interval_scores = th.interval_scores
        if interval < interval_tiers.get("tier1", 60):
            interval_sc = interval_scores.get("tier1", 0.9)
        elif interval < interval_tiers.get("tier2", 300):
            interval_sc = interval_scores.get("tier2", 0.7)
        elif interval < interval_tiers.get("tier3", 600):
            interval_sc = interval_scores.get("tier3", 0.5)
        else:
            interval_sc = interval_scores.get("tier4", 0.25)

        freq_tiers = th.freq_tiers
        freq_scores = th.freq_scores
        if freq >= freq_tiers.get("tier1", 10):
            freq_sc = freq_scores.get("tier1", 1.0)
        elif freq >= freq_tiers.get("tier2", 5):
            freq_sc = freq_scores.get("tier2", 0.8)
        elif freq >= freq_tiers.get("tier3", 3):
            freq_sc = freq_scores.get("tier3", 0.6)
        else:
            freq_sc = freq_scores.get("tier4", 0.3)

        length_tiers = th.length_tiers
        length_scores = th.length_scores
        if avg_len >= length_tiers.get("tier1", 30):
            len_sc = length_scores.get("tier1", 0.85)
        elif avg_len >= length_tiers.get("tier2", 10):
            len_sc = length_scores.get("tier2", 0.6)
        elif avg_len >= length_tiers.get("tier3", 5):
            len_sc = length_scores.get("tier3", 0.4)
        else:
            len_sc = length_scores.get("tier4", 0.25)

        init_sc = min(1.0, init_ratio * 1.2)
        q_sc = min(1.0, 0.5 + q_ratio)

        weights = th.weights
        weighted = (
            interval_sc * weights.get("interval", 0.25)
            + freq_sc * weights.get("freq", 0.20)
            + len_sc * weights.get("length", 0.15)
            + init_sc * weights.get("init", 0.25)
            + q_sc * weights.get("question", 0.15)
        )
        return max(0.0, min(1.0, weighted))


class UserStateDetector:
    """用户状态推断 — 双层架构（规则 + 模型）"""

    def __init__(
        self,
        config_engine: Optional[ConfigEngine] = None,
        model_understand: Optional[Any] = None,
    ):
        self._config = config_engine or ConfigEngine.get_instance()
        self._model = model_understand
        self._logs: Dict[str, _InteractionLog] = {}
        self._model_weight: float = 0.6
        self._rule_weight: float = 0.4
        self._thresholds: Optional[ScoringThresholds] = None
        self._window_size: int = 200
        self._load_config()

    def _load_config(self):
        """加载配置"""
        self._model_weight = self._config.get(
            "user_state", "model_weight", 0.6
        )
        self._rule_weight = self._config.get("user_state", "rule_weight", 0.4)
        self._window_size = self._config.get("user_state", "window_size", 200)

        self._thresholds = ScoringThresholds(
            interval_tiers=self._config.get(
                "user_state",
                "interval_tiers",
                {"tier1": 60, "tier2": 300, "tier3": 600},
            ),
            interval_scores=self._config.get(
                "user_state",
                "interval_scores",
                {"tier1": 0.9, "tier2": 0.7, "tier3": 0.5, "tier4": 0.25},
            ),
            freq_tiers=self._config.get(
                "user_state",
                "freq_tiers",
                {"tier1": 10, "tier2": 5, "tier3": 3},
            ),
            freq_scores=self._config.get(
                "user_state",
                "freq_scores",
                {"tier1": 1.0, "tier2": 0.8, "tier3": 0.6, "tier4": 0.3},
            ),
            length_tiers=self._config.get(
                "user_state",
                "length_tiers",
                {"tier1": 30, "tier2": 10, "tier3": 5},
            ),
            length_scores=self._config.get(
                "user_state",
                "length_scores",
                {"tier1": 0.85, "tier2": 0.6, "tier3": 0.4, "tier4": 0.25},
            ),
            weights=self._config.get(
                "user_state",
                "weights",
                {
                    "interval": 0.25,
                    "freq": 0.20,
                    "length": 0.15,
                    "init": 0.25,
                    "question": 0.15,
                },
            ),
            verdict_thresholds=self._config.get(
                "user_state",
                "verdict_thresholds",
                {"want_to_chat": 0.6, "not_want_to_chat": 0.35},
            ),
        )

    def set_model(self, model_understand: Any):
        """设置模型理解器"""
        self._model = model_understand

    def feed_interaction(
        self,
        uid: str,
        msg_length: int,
        has_question: bool = False,
        is_user_initiated: bool = True,
        has_emoji: bool = False,
        reply_speed: float = 0.0,
    ) -> None:
        """
        记录交互

        Args:
            uid: 用户ID
            msg_length: 消息长度
            has_question: 是否包含问题
            is_user_initiated: 是否用户主动发起
            has_emoji: 是否包含表情
            reply_speed: 回复速度（秒）
        """
        if uid not in self._logs:
            log = _InteractionLog(window_size=self._window_size)
            log.set_thresholds(self._thresholds)
            self._logs[uid] = log
        self._logs[uid].append(
            msg_length, has_question, is_user_initiated, has_emoji, reply_speed
        )

    def get_rule_signals(self, uid: str) -> RuleSignals:
        """获取规则信号"""
        log = self._logs.get(uid)
        if not log:
            return RuleSignals()
        return log.compute_signals()

    async def infer_model(
        self, content: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """模型推断"""
        if not self._model:
            return {
                "state": "uncertain",
                "willingness": 0.5,
                "interest_level": 0.5,
                "attention_level": 0.5,
                "confidence": 0.3,
                "reasoning": "模型未配置",
            }

        try:
            if hasattr(self._model, "analyze"):
                result = await self._model.analyze(
                    {
                        "task": "user_state_inference",
                        "content": content,
                        "context": context,
                    }
                )
                return {
                    "state": result.get("state", "uncertain"),
                    "willingness": float(result.get("willingness", 0.5)),
                    "interest_level": float(result.get("interest_level", 0.5)),
                    "attention_level": float(
                        result.get("attention_level", 0.5)
                    ),
                    "confidence": float(result.get("confidence", 0.5)),
                    "reasoning": result.get("reasoning", ""),
                }
        except Exception as e:
            logger.debug(f"模型推断失败: {e}")

        return {
            "state": "uncertain",
            "willingness": 0.5,
            "interest_level": 0.5,
            "attention_level": 0.5,
            "confidence": 0.3,
            "reasoning": "模型推断失败",
        }

    async def infer_fused(
        self, uid: str, content: str, context: Dict[str, Any]
    ) -> UserStateResult:
        """
        融合推断

        Args:
            uid: 用户ID
            content: 内容
            context: 上下文

        Returns:
            UserStateResult: 融合后的用户状态结果
        """
        if not self._thresholds:
            return UserStateResult(reasoning="阈值未加载")

        model_result = await self.infer_model(content, context)
        rule_signals = self.get_rule_signals(uid)

        fused_willingness = (
            model_result["willingness"] * self._model_weight
            + rule_signals.rule_score * self._rule_weight
        )
        fused_willingness = max(0.0, min(1.0, fused_willingness))

        want_threshold = self._thresholds.verdict_thresholds.get(
            "want_to_chat", 0.6
        )
        not_want_threshold = self._thresholds.verdict_thresholds.get(
            "not_want_to_chat", 0.35
        )

        if fused_willingness >= want_threshold:
            fused_state = "want_to_chat"
        elif fused_willingness <= not_want_threshold:
            fused_state = "not_want_to_chat"
        else:
            fused_state = "uncertain"

        return UserStateResult(
            state=fused_state,
            willingness=round(fused_willingness, 3),
            interest_level=model_result["interest_level"],
            attention_level=model_result["attention_level"],
            confidence=model_result["confidence"],
            reasoning=model_result["reasoning"],
            rule_signals=rule_signals,
            model_contribution=self._model_weight,
            rule_contribution=self._rule_weight,
        )

    def get_state_description(self, state: str) -> str:
        """获取状态描述"""
        descriptions = {
            "want_to_chat": "想聊天",
            "not_want_to_chat": "不想聊天",
            "uncertain": "状态不确定",
        }
        return descriptions.get(state, state)

    def get_user_summary(self, uid: str) -> Dict[str, Any]:
        """获取用户状态摘要"""
        signals = self.get_rule_signals(uid)
        return {
            "user_id": uid,
            "rule_score": signals.rule_score,
            "rule_verdict": signals.rule_verdict,
            "messages_per_hour": signals.messages_per_hour,
            "avg_message_length": signals.avg_message_length,
            "avg_interval_sec": signals.avg_interval_sec,
            "initiation_ratio": signals.user_initiation_ratio,
            "question_ratio": signals.contains_question_ratio,
            "emoji_ratio": signals.emoji_usage_ratio,
            "interaction_count": len(
                self._logs.get(uid, _InteractionLog())._records
            ),
        }

    def clear_user(self, uid: str):
        """清除用户数据"""
        self._logs.pop(uid, None)

    def clear_all(self):
        """清除所有数据"""
        self._logs.clear()


_user_state_detector_instance: Optional[UserStateDetector] = None


def get_user_state_detector(
    config_engine: Optional[ConfigEngine] = None,
    model_understand: Optional[Any] = None,
) -> UserStateDetector:
    """获取用户状态检测器单例"""
    global _user_state_detector_instance
    if _user_state_detector_instance is None:
        _user_state_detector_instance = UserStateDetector(
            config_engine, model_understand
        )
    return _user_state_detector_instance

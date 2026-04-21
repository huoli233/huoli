"""
深度骚扰检测器

合并骚扰检测与情感分析结果，按配置权重合成综合风险值；
支持时间窗口内风险累积与建议屏蔽时长。

依赖: ConfigEngine, ModelAnalyzer（抽象接口）
被依赖: 消息处理管线, GlobalShieldManager
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Callable

from src.common.logger import get_logger
from src.common.config.config_engine import get_default_config_engine

logger = get_logger("骚扰检测")


@dataclass
class HarassmentResult:
    """骚扰检测结果"""

    is_harassment: bool = False
    severity: float = 0.0
    harassment_type: str = "none"
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class SentimentResult:
    """情感分析结果"""

    sentiment: str = "neutral"
    intensity: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class HarassmentFullResult:
    """深度骚扰分析结果"""

    is_safe: bool = True
    combined_risk: float = 0.0
    harassment_severity: float = 0.0
    is_harassment: bool = False
    harassment_type: str = "none"
    sentiment: str = "neutral"
    sentiment_intensity: float = 0.0
    suggested_action: str = "ignore"
    confidence: float = 0.0
    reasoning: str = ""
    cumulative_window_score: float = 0.0
    block_duration_seconds: int = 0
    analysis_timestamp: float = field(default_factory=time.time)


class ModelAnalyzerInterface(ABC):
    """模型分析器抽象接口"""

    @abstractmethod
    async def analyze_harassment(
        self, content: str, context: Dict[str, Any]
    ) -> HarassmentResult:
        """分析骚扰内容"""
        pass

    @abstractmethod
    async def analyze_sentiment(
        self, content: str, context: Dict[str, Any]
    ) -> SentimentResult:
        """分析情感"""
        pass

    async def analyze_batch(
        self,
        content: str,
        context: Dict[str, Any],
        tasks: List[str],
    ) -> Dict[str, Any]:
        """批量分析"""
        results = {}
        if "harassment_detection" in tasks:
            results["harassment_detection"] = await self.analyze_harassment(
                content, context
            )
        if "sentiment_analysis" in tasks:
            results["sentiment_analysis"] = await self.analyze_sentiment(
                content, context
            )
        return results


class DefaultModelAnalyzer(ModelAnalyzerInterface):
    """默认模型分析器（基于规则的简单实现）"""

    def __init__(self, config_engine=None):
        self._config = config_engine or get_default_config_engine()
        self._harassment_keywords = self._load_harassment_keywords()
        self._negative_sentiment_words = self._load_negative_words()

    def _load_harassment_keywords(self) -> Dict[str, float]:
        """加载骚扰关键词及权重"""
        keywords = self._config.get("harassment_keywords", "words", {})
        if keywords:
            return keywords
        return {
            "傻逼": 0.8,
            "傻": 0.3,
            "蠢": 0.3,
            "笨蛋": 0.3,
            "滚": 0.5,
            "闭嘴": 0.4,
            "讨厌你": 0.4,
            "恶心": 0.5,
            "去死": 0.9,
            "打死": 0.7,
            "打你": 0.4,
            "杀了你": 0.9,
            "骂你": 0.4,
            "废物": 0.6,
            "垃圾": 0.5,
        }

    def _load_negative_words(self) -> List[str]:
        """加载负面情感词"""
        words = self._config.get("sentiment_words", "negative", [])
        if words:
            return words
        return [
            "不爽",
            "生气",
            "愤怒",
            "讨厌",
            "烦",
            "累",
            "难过",
            "伤心",
            "失望",
        ]

    async def analyze_harassment(
        self, content: str, context: Dict[str, Any]
    ) -> HarassmentResult:
        """基于关键词的骚扰检测"""
        content_lower = content.lower()
        max_severity = 0.0
        matched_type = "none"
        matched_keywords = []

        for keyword, weight in self._harassment_keywords.items():
            if keyword in content_lower:
                if weight > max_severity:
                    max_severity = weight
                    matched_type = "verbal_abuse"
                matched_keywords.append(keyword)

        # 阈值提高到0.4，避免低权重词（如"傻"0.3）单独触发骚扰判定
        is_harassment = max_severity > 0.4
        confidence = min(1.0, len(matched_keywords) * 0.2 + max_severity * 0.5)

        return HarassmentResult(
            is_harassment=is_harassment,
            severity=max_severity,
            harassment_type=matched_type,
            confidence=confidence,
            reasoning=f"关键词匹配: {
                ', '.join(matched_keywords) if matched_keywords else '无'}",
        )

    async def analyze_sentiment(
        self, content: str, context: Dict[str, Any]
    ) -> SentimentResult:
        """基于关键词的情感分析"""
        content_lower = content.lower()
        negative_count = sum(
            1
            for word in self._negative_sentiment_words
            if word in content_lower
        )

        if negative_count >= 3:
            sentiment = "negative"
            intensity = min(1.0, negative_count * 0.25)
        elif negative_count >= 1:
            sentiment = "negative"
            intensity = negative_count * 0.2
        else:
            positive_words = [
                "好",
                "棒",
                "喜欢",
                "爱",
                "开心",
                "高兴",
                "谢谢",
                "感谢",
            ]
            positive_count = sum(
                1 for word in positive_words if word in content_lower
            )
            if positive_count >= 2:
                sentiment = "positive"
                intensity = min(1.0, positive_count * 0.2)
            else:
                sentiment = "neutral"
                intensity = 0.0

        return SentimentResult(
            sentiment=sentiment,
            intensity=intensity,
            confidence=0.6 if sentiment != "neutral" else 0.8,
            reasoning=f"情感词计数: 负面={negative_count}",
        )


class CumulativeWindow:
    """滑动时间窗口内风险累积"""

    def __init__(self, window_seconds: float = 300.0, max_events: int = 100):
        self._window_seconds = max(30.0, float(window_seconds))
        self._max_events = max_events
        self._events: List[Tuple[float, float]] = []

    def add(self, now: float, score: float) -> float:
        """添加事件并返回窗口内累积分数"""
        self._events.append((now, max(0.0, float(score))))
        self._events = [
            (t, s) for t, s in self._events if now - t <= self._window_seconds
        ]
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        return sum(s for _, s in self._events)

    def get_recent_count(self, now: float, threshold: float = 0.0) -> int:
        """获取窗口内超过阈值的事件数"""
        return sum(
            1
            for t, s in self._events
            if now - t <= self._window_seconds and s > threshold
        )

    def get_average(self, now: float) -> float:
        """获取窗口内平均分数"""
        valid_events = [
            (t, s) for t, s in self._events if now - t <= self._window_seconds
        ]
        if not valid_events:
            return 0.0
        return sum(s for _, s in valid_events) / len(valid_events)

    def clear(self) -> None:
        """清空窗口"""
        self._events.clear()


@dataclass
class HarassmentConfig:
    """骚扰检测配置"""

    weight_harassment: float = 0.65
    weight_sentiment: float = 0.35
    block_threshold: float = 0.72
    warn_threshold: float = 0.42
    window_seconds: float = 300.0
    cumulative_block_threshold: float = 5.0
    cumulative_warn_ratio: float = 0.55
    cumulative_high_threshold: float = 10.0
    combined_high_threshold: float = 0.9
    cumulative_mid_threshold: float = 5.0
    combined_mid_threshold: float = 0.55
    block_duration_high: int = 120
    block_duration_mid: int = 60
    block_duration_low: int = 30
    enabled: bool = True


class HarassmentDetector:
    """
    骚扰检测器

    功能:
    1. 结合骚扰检测与情感分析计算综合风险
    2. 时间窗口内风险累积追踪
    3. 自动生成建议动作和屏蔽时长

    重要：所有阈值从配置加载，禁止硬编码
    """

    def __init__(
        self,
        config_engine=None,
        model_analyzer: Optional[ModelAnalyzerInterface] = None,
    ):
        self._config = config_engine or get_default_config_engine()
        self._model = model_analyzer or DefaultModelAnalyzer(self._config)
        self._cum_windows: Dict[str, CumulativeWindow] = {}
        self._config_data = HarassmentConfig()
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600.0
        self._load_config()

    def _load_config(self) -> None:
        """从配置加载所有阈值"""
        self._config_data.weight_harassment = self._config.get(
            "harassment_detector", "weight_harassment", 0.65
        )
        self._config_data.weight_sentiment = self._config.get(
            "harassment_detector", "weight_sentiment", 0.35
        )
        self._config_data.block_threshold = self._config.get(
            "harassment_detector", "block_threshold", 0.72
        )
        self._config_data.warn_threshold = self._config.get(
            "harassment_detector", "warn_threshold", 0.42
        )
        self._config_data.window_seconds = self._config.get(
            "harassment_detector", "window_seconds", 300.0
        )
        self._config_data.cumulative_block_threshold = self._config.get(
            "harassment_detector", "cumulative_block_threshold", 5.0
        )
        self._config_data.cumulative_warn_ratio = self._config.get(
            "harassment_detector", "cumulative_warn_ratio", 0.55
        )
        self._config_data.cumulative_high_threshold = self._config.get(
            "harassment_detector", "cumulative_high_threshold", 10.0
        )
        self._config_data.combined_high_threshold = self._config.get(
            "harassment_detector", "combined_high_threshold", 0.9
        )
        self._config_data.cumulative_mid_threshold = self._config.get(
            "harassment_detector", "cumulative_mid_threshold", 5.0
        )
        self._config_data.combined_mid_threshold = self._config.get(
            "harassment_detector", "combined_mid_threshold", 0.55
        )
        self._config_data.block_duration_high = self._config.get(
            "harassment_detector", "block_duration_high", 120
        )
        self._config_data.block_duration_mid = self._config.get(
            "harassment_detector", "block_duration_mid", 60
        )
        self._config_data.block_duration_low = self._config.get(
            "harassment_detector", "block_duration_low", 30
        )
        self._config_data.enabled = self._config.get(
            "harassment_detector", "enabled", True
        )

    def set_model_analyzer(self, analyzer: ModelAnalyzerInterface) -> None:
        """设置模型分析器"""
        self._model = analyzer

    def _stream_key(self, context: Dict[str, Any]) -> str:
        """生成流标识键"""
        channel = context.get("channel_id", context.get("stream_id", "global"))
        user = context.get("user_id", "anonymous")
        return f"{channel}:{user}"

    def _get_window(self, key: str) -> CumulativeWindow:
        """获取或创建累积窗口"""
        w = self._cum_windows.get(key)
        if w is None:
            w = CumulativeWindow(self._config_data.window_seconds)
            self._cum_windows[key] = w
        return w

    def _cleanup_old_windows(self) -> None:
        """清理过期的窗口"""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        expired_keys = []
        for key, window in self._cum_windows.items():
            if not window._events or (
                window._events
                and now - window._events[-1][0]
                > self._config_data.window_seconds * 2
            ):
                expired_keys.append(key)
        for key in expired_keys:
            del self._cum_windows[key]
        if expired_keys:
            logger.debug(f"清理过期累积窗口: {len(expired_keys)}个")

    async def analyze(
        self, content: str, context: Dict[str, Any]
    ) -> HarassmentFullResult:
        """
        分析内容的骚扰风险

        Args:
            content: 待分析文本
            context: 上下文信息（user_id, channel_id等）

        Returns:
            HarassmentFullResult: 完整分析结果
        """
        if not self._config_data.enabled:
            return HarassmentFullResult(
                is_safe=True,
                reasoning="骚扰检测模块未启用",
            )

        if not content or not content.strip():
            return HarassmentFullResult(
                is_safe=True,
                reasoning="空内容无需检测",
            )

        self._cleanup_old_windows()

        results = await self._model.analyze_batch(
            content,
            context,
            ["harassment_detection", "sentiment_analysis"],
        )

        h = results.get("harassment_detection")
        s = results.get("sentiment_analysis")

        if h is None:
            h = HarassmentResult()
        if s is None:
            s = SentimentResult()

        h_sev = float(h.severity) if h.is_harassment else 0.0
        neg_boost = (
            float(s.intensity)
            if s.sentiment == "negative"
            else float(s.intensity) * 0.2
        )

        combined = (
            self._config_data.weight_harassment * h_sev
            + self._config_data.weight_sentiment * neg_boost
        )
        if h.is_harassment:
            combined = max(combined, h_sev * 0.92)
        combined = max(0.0, min(1.0, combined))

        key = self._stream_key(context)
        tracker = self._get_window(key)
        now = time.time()
        cumulative = tracker.add(now, combined)

        warn_cum = (
            self._config_data.cumulative_block_threshold
            * self._config_data.cumulative_warn_ratio
        )

        if (
            combined >= self._config_data.block_threshold
            or cumulative >= self._config_data.cumulative_block_threshold
        ):
            action = "block"
        elif (
            combined >= self._config_data.warn_threshold
            or cumulative >= warn_cum
        ):
            action = "warn"
        else:
            action = "ignore"

        block_dur = 0
        if action == "block":
            if (
                cumulative >= self._config_data.cumulative_high_threshold
                or combined >= self._config_data.combined_high_threshold
            ):
                block_dur = self._config_data.block_duration_high
            elif (
                cumulative >= self._config_data.cumulative_mid_threshold
                or combined >= self._config_data.combined_mid_threshold
            ):
                block_dur = self._config_data.block_duration_mid
            else:
                block_dur = self._config_data.block_duration_low

        is_safe = (not h.is_harassment) and action == "ignore"
        conf = (float(h.confidence) + float(s.confidence)) / 2.0

        result = HarassmentFullResult(
            is_safe=is_safe,
            combined_risk=combined,
            harassment_severity=float(h.severity),
            is_harassment=bool(h.is_harassment),
            harassment_type=str(h.harassment_type),
            sentiment=str(s.sentiment),
            sentiment_intensity=float(s.intensity),
            suggested_action=action,
            confidence=conf,
            reasoning=f"骚扰: {h.reasoning[:100]} | 情感: {s.reasoning[:100]}",
            cumulative_window_score=float(cumulative),
            block_duration_seconds=block_dur,
            analysis_timestamp=now,
        )

        if action != "ignore":
            logger.warning(
                f"骚扰检测触发 | 用户:{context.get('user_id', 'unknown')[:8]} | "
                f"风险:{combined:.2f} | 累积:{cumulative:.2f} | 动作:{action}"
            )

        return result

    def get_user_risk_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """获取用户风险摘要"""
        key = self._stream_key(context)
        window = self._cum_windows.get(key)
        now = time.time()

        if window is None:
            return {
                "cumulative_score": 0.0,
                "recent_count": 0,
                "average_score": 0.0,
                "has_history": False,
            }

        return {
            "cumulative_score": window.get_average(now)
            * window.get_recent_count(now),
            "recent_count": window.get_recent_count(now),
            "average_score": window.get_average(now),
            "has_history": len(window._events) > 0,
        }

    def reset_user_window(self, context: Dict[str, Any]) -> None:
        """重置用户累积窗口"""
        key = self._stream_key(context)
        if key in self._cum_windows:
            self._cum_windows[key].clear()
            logger.info(f"用户累积窗口已重置 | 键:{key}")

    def export_state(self) -> Dict[str, Any]:
        """导出状态"""
        return {
            "windows": {
                k: {"events": w._events} for k, w in self._cum_windows.items()
            },
            "config": {
                "enabled": self._config_data.enabled,
                "block_threshold": self._config_data.block_threshold,
                "warn_threshold": self._config_data.warn_threshold,
            },
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """导入状态"""
        self._cum_windows.clear()
        for k, v in state.get("windows", {}).items():
            window = CumulativeWindow(self._config_data.window_seconds)
            window._events = v.get("events", [])
            self._cum_windows[k] = window
        logger.info(f"骚扰检测状态恢复完成 | 窗口数:{len(self._cum_windows)}")


_harassment_detector: Optional[HarassmentDetector] = None


def get_harassment_detector(
    config_engine=None,
    model_analyzer: Optional[ModelAnalyzerInterface] = None,
) -> HarassmentDetector:
    """获取骚扰检测器单例"""
    global _harassment_detector
    if _harassment_detector is None:
        _harassment_detector = HarassmentDetector(
            config_engine, model_analyzer
        )
    return _harassment_detector

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.common.config.config_engine import ConfigEngine
from src.common.logger import get_logger

logger = get_logger("群感知")


@dataclass
class GroupSenseResult:
    """群聊感知结果"""

    activity_level: str = "冷清"
    message_count_5min: int = 0
    active_user_count: int = 0
    silence_duration_seconds: float = 0.0
    controversy_detected: bool = False
    needs_topic: bool = False
    description: str = ""
    burst_detected: bool = False
    burst_density: float = 0.0
    dominant_users: List[str] = field(default_factory=list)
    topic_hints: List[str] = field(default_factory=list)


@dataclass
class BurstAnalysis:
    """爆发分析结果"""

    is_burst: bool = False
    density: float = 0.0
    peak_rate: float = 0.0
    dominant_users: List[str] = field(default_factory=list)


@dataclass
class ControversySignals:
    """争议信号"""

    has_controversy: bool = False
    involved_pairs: List[Tuple[str, str]] = field(default_factory=list)
    exchange_rate: float = 0.0
    avg_message_length: float = 0.0


class GroupSense:
    """群聊态势感知 - 纯代码分析，零LLM调用"""

    def __init__(self, config_engine: Optional[ConfigEngine] = None):
        self._config = config_engine or ConfigEngine.get_instance()
        self._last_message_times: Dict[str, float] = {}
        self._last_bot_reply_times: Dict[str, float] = {}
        self._recent_speakers: Dict[str, List[Tuple[str, float]]] = (
            defaultdict(list)
        )
        self._message_history: Dict[str, List[Tuple[str, float, str]]] = (
            defaultdict(list)
        )
        self._load_config()

    def _load_config(self):
        """加载配置参数"""
        self._burst_msg_count = self._config.get(
            "group_sense", "burst_msg_count", 15
        )
        self._burst_msg_count_alt = self._config.get(
            "group_sense", "burst_msg_count_alt", 8
        )
        self._burst_user_count = self._config.get(
            "group_sense", "burst_user_count", 4
        )
        self._burst_density_threshold = self._config.get(
            "group_sense", "burst_density_threshold", 0.5
        )
        self._burst_msg_count_density = self._config.get(
            "group_sense", "burst_msg_count_density", 10
        )
        self._active_msg_count = self._config.get(
            "group_sense", "active_msg_count", 6
        )
        self._active_user_count = self._config.get(
            "group_sense", "active_user_count", 3
        )
        self._stable_msg_count = self._config.get(
            "group_sense", "stable_msg_count", 2
        )
        self._silence_cold_threshold = self._config.get(
            "group_sense", "silence_cold_threshold", 600.0
        )
        self._silence_dead_threshold = self._config.get(
            "group_sense", "silence_dead_threshold", 1800.0
        )
        self._controversy_short_msg_length = self._config.get(
            "group_sense", "controversy_short_msg_length", 30
        )
        self._controversy_exchange_rate = self._config.get(
            "group_sense", "controversy_exchange_rate", 0.1
        )
        self._controversy_min_exchanges = self._config.get(
            "group_sense", "controversy_min_exchanges", 3
        )
        self._activity_window_seconds = self._config.get(
            "group_sense", "activity_window_seconds", 300
        )
        self._silence_topic_threshold = self._config.get(
            "group_sense", "silence_topic_threshold", 600
        )
        self._speaker_window_seconds = self._config.get(
            "group_sense", "speaker_window_seconds", 300
        )
        self._history_max_size = self._config.get(
            "group_sense", "history_max_size", 500
        )

    def analyze(
        self,
        stream_id: str,
        recent_messages: List[dict],
        last_reply_time: float = 0.0,
    ) -> GroupSenseResult:
        """
        分析群聊态势

        Args:
            stream_id: 流ID（群ID）
            recent_messages: 最近消息列表，每条消息包含 sender_id, timestamp, text
            last_reply_time: 机器人上次回复时间

        Returns:
            GroupSenseResult: 群聊感知结果
        """
        result = GroupSenseResult()
        now = time.time()

        if not recent_messages:
            result.activity_level = "死寂"
            result.silence_duration_seconds = self._calc_silence(
                stream_id, now
            )
            result.needs_topic = (
                result.silence_duration_seconds
                >= self._silence_topic_threshold
            )
            result.description = self._build_description(result)
            return result

        timestamps = [
            m.get("timestamp", 0)
            for m in recent_messages
            if m.get("timestamp", 0) > 0
        ]
        if not timestamps:
            result.activity_level = "冷清"
            result.description = self._build_description(result)
            return result

        latest_ts = max(timestamps)
        self._last_message_times[stream_id] = latest_ts

        if last_reply_time > 0:
            self._last_bot_reply_times[stream_id] = last_reply_time

        for msg in recent_messages:
            sender_id = msg.get("sender_id")
            ts = msg.get("timestamp", 0)
            text = msg.get("text", "")
            if sender_id:
                self._recent_speakers[stream_id].append((sender_id, ts))
                self._message_history[stream_id].append((sender_id, ts, text))

        cutoff = now - self._speaker_window_seconds
        self._recent_speakers[stream_id] = [
            (uid, ts)
            for uid, ts in self._recent_speakers[stream_id]
            if ts >= cutoff
        ]
        if len(self._message_history[stream_id]) > self._history_max_size:
            self._message_history[stream_id] = self._message_history[
                stream_id
            ][-self._history_max_size:]

        result.silence_duration_seconds = now - latest_ts

        window_msgs = [
            m
            for m in recent_messages
            if now - m.get("timestamp", 0) <= self._activity_window_seconds
        ]
        result.message_count_5min = len(window_msgs)
        result.active_user_count = len(
            set(m.get("sender_id") for m in window_msgs if m.get("sender_id"))
        )

        time_span = 0.0
        if len(window_msgs) >= 2:
            window_ts = [m.get("timestamp", 0) for m in window_msgs]
            time_span = max(window_ts) - min(window_ts)

        result.activity_level = self._calc_activity_level(
            msg_count=result.message_count_5min,
            user_count=result.active_user_count,
            silence_seconds=result.silence_duration_seconds,
            time_span=time_span,
        )

        burst_analysis = self._analyze_burst(window_msgs, time_span)
        result.burst_detected = burst_analysis.is_burst
        result.burst_density = burst_analysis.density
        if burst_analysis.dominant_users:
            result.dominant_users = burst_analysis.dominant_users[:3]

        result.controversy_detected = self._detect_controversy(
            window_msgs, stream_id, now
        )

        result.needs_topic = (
            result.silence_duration_seconds >= self._silence_topic_threshold
        )

        result.topic_hints = self._extract_topic_hints(window_msgs)

        result.description = self._build_description(result)
        return result

    def _calc_silence(self, stream_id: str, now: float) -> float:
        """计算沉默时长"""
        last = self._last_message_times.get(stream_id, 0.0)
        if last > 0:
            return now - last
        return 0.0

    def _calc_activity_level(
        self,
        msg_count: int,
        user_count: int,
        silence_seconds: float,
        time_span: float,
    ) -> str:
        """计算活跃度等级"""
        if silence_seconds > self._silence_dead_threshold:
            return "死寂"
        if silence_seconds > self._silence_cold_threshold:
            return "冷清"

        density = msg_count / max(1, time_span) if time_span > 0 else 0

        if msg_count >= self._burst_msg_count or (
            msg_count >= self._burst_msg_count_alt
            and user_count >= self._burst_user_count
        ):
            return "爆发"
        if (
            density > self._burst_density_threshold
            and msg_count >= self._burst_msg_count_density
        ):
            return "爆发"

        if (
            msg_count >= self._active_msg_count
            or user_count >= self._active_user_count
        ):
            return "活跃"

        if msg_count >= self._stable_msg_count:
            return "平稳"

        return "冷清"

    def _analyze_burst(
        self, messages: List[dict], time_span: float
    ) -> BurstAnalysis:
        """分析消息爆发"""
        result = BurstAnalysis()

        if len(messages) < 5:
            return result

        density = len(messages) / max(1, time_span) if time_span > 0 else 0
        result.density = density

        if density > self._burst_density_threshold:
            result.is_burst = True

        if time_span > 0:
            window_size = 10.0
            peak_count = 0
            timestamps = sorted(m.get("timestamp", 0) for m in messages)
            for i, ts in enumerate(timestamps):
                count = sum(
                    1 for t in timestamps if ts <= t < ts + window_size
                )
                peak_count = max(peak_count, count)
            result.peak_rate = peak_count / window_size

        user_msg_counts = Counter(
            m.get("sender_id") for m in messages if m.get("sender_id")
        )
        if user_msg_counts:
            top_user, top_count = user_msg_counts.most_common(1)[0]
            if top_count >= len(messages) * 0.5:
                result.dominant_users = [top_user]

        return result

    def _detect_controversy(
        self,
        messages: List[dict],
        stream_id: str,
        now: float,
    ) -> bool:
        """检测争议/争吵"""
        if len(messages) < 4:
            return False

        user_pair_counts: Counter = Counter()
        user_pair_timestamps: Dict[tuple, List[float]] = defaultdict(list)
        prev_user = None

        for msg in messages:
            curr_user = msg.get("sender_id")
            if curr_user and prev_user and curr_user != prev_user:
                pair = tuple(sorted([curr_user, prev_user]))
                user_pair_counts[pair] += 1
                user_pair_timestamps[pair].append(msg.get("timestamp", 0))
            prev_user = curr_user

        frequent_pairs = [
            (pair, count)
            for pair, count in user_pair_counts.items()
            if count >= self._controversy_min_exchanges
        ]
        if not frequent_pairs:
            return False

        for pair, count in frequent_pairs:
            pair_user_ids = set(pair)
            pair_msgs = [
                m
                for m in messages
                if m.get("sender_id") in pair_user_ids and m.get("text")
            ]
            if len(pair_msgs) < self._controversy_min_exchanges:
                continue

            avg_length = sum(len(m.get("text", "")) for m in pair_msgs) / len(
                pair_msgs
            )
            is_short = avg_length < self._controversy_short_msg_length

            timestamps = sorted(m.get("timestamp", 0) for m in pair_msgs)
            if len(timestamps) >= 2:
                time_span = timestamps[-1] - timestamps[0]
                exchange_rate = len(timestamps) / max(time_span, 1.0)
                is_rapid = exchange_rate > self._controversy_exchange_rate
            else:
                is_rapid = False

            if count >= self._controversy_min_exchanges and (
                is_short or is_rapid
            ):
                logger.debug(
                    f"检测到争议: 互动对={pair}, 次数={count}, "
                    f"平均长度={avg_length:.0f}, 快节奏={is_rapid}"
                )
                return True

        return False

    def _extract_topic_hints(self, messages: List[dict]) -> List[str]:
        """提取话题提示"""
        hints = []
        keywords = Counter()

        for msg in messages:
            text = msg.get("text", "")
            if not text or len(text) < 2:
                continue
            words = text.split()
            for word in words:
                if len(word) >= 2:
                    keywords[word] += 1

        for word, count in keywords.most_common(5):
            if count >= 2:
                hints.append(word)

        return hints

    def _build_description(self, result: GroupSenseResult) -> str:
        """构建描述文本"""
        parts = []

        if result.message_count_5min > 0:
            parts.append(
                f"活跃度: {result.activity_level}({result.message_count_5min}条/{result.active_user_count}人)"
            )
        else:
            parts.append(f"活跃度: {result.activity_level}")

        silence = result.silence_duration_seconds
        if silence < 30:
            parts.append("刚刚有人在说话")
        elif silence < 120:
            parts.append(f"已有{int(silence)}秒没人说话")
        elif silence < 600:
            parts.append(f"已有{int(silence // 60)}分钟没人说话")
        elif silence < 3600:
            parts.append(f"已安静{int(silence // 60)}分钟")
        else:
            parts.append(f"已安静{silence / 3600:.1f}小时")

        if result.burst_detected:
            parts.append(f"消息爆发(密度={result.burst_density:.2f})")

        if result.controversy_detected:
            parts.append("检测到争议/吵架")

        if result.needs_topic:
            parts.append("群聊需要话题")

        if result.topic_hints:
            parts.append(f"热门词: {', '.join(result.topic_hints[:3])}")

        return ", ".join(parts)

    def on_message_received(self, stream_id: str):
        """消息接收回调"""
        self._last_message_times[stream_id] = time.time()

    def on_bot_replied(self, stream_id: str):
        """机器人回复回调"""
        self._last_bot_reply_times[stream_id] = time.time()

    def get_silence_duration(self, stream_id: str) -> float:
        """获取沉默时长"""
        return self._calc_silence(stream_id, time.time())

    def get_recent_speakers(
        self, stream_id: str, window_seconds: float = 300
    ) -> List[str]:
        """获取最近发言者列表"""
        now = time.time()
        cutoff = now - window_seconds
        speakers = [
            uid
            for uid, ts in self._recent_speakers.get(stream_id, [])
            if ts >= cutoff
        ]
        return list(set(speakers))

    def get_activity_summary(self, stream_id: str) -> dict:
        """获取活跃度摘要"""
        now = time.time()
        speakers = self._recent_speakers.get(stream_id, [])
        recent_speakers = [
            (uid, ts)
            for uid, ts in speakers
            if now - ts <= self._activity_window_seconds
        ]
        unique_users = set(uid for uid, _ in recent_speakers)

        return {
            "stream_id": stream_id,
            "silence_seconds": self._calc_silence(stream_id, now),
            "recent_speaker_count": len(unique_users),
            "recent_speakers": list(unique_users),
            "last_message_time": self._last_message_times.get(stream_id, 0),
            "last_bot_reply_time": self._last_bot_reply_times.get(
                stream_id, 0
            ),
        }

    def reset(self, stream_id: Optional[str] = None):
        """重置状态"""
        if stream_id:
            self._last_message_times.pop(stream_id, None)
            self._last_bot_reply_times.pop(stream_id, None)
            self._recent_speakers.pop(stream_id, None)
            self._message_history.pop(stream_id, None)
        else:
            self._last_message_times.clear()
            self._last_bot_reply_times.clear()
            self._recent_speakers.clear()
            self._message_history.clear()


_group_sense_instance: Optional[GroupSense] = None


def get_group_sense(
    config_engine: Optional[ConfigEngine] = None,
) -> GroupSense:
    """获取群聊感知单例"""
    global _group_sense_instance
    if _group_sense_instance is None:
        _group_sense_instance = GroupSense(config_engine)
    return _group_sense_instance

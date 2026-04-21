import time
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from src.common.logger import get_logger

logger = get_logger("stream_emotion")


CHAT_VALUE_RECOVERY_RATE = 5.0
FATIGUE_DECAY_RATE = 3.0
ANNOYANCE_DECAY_RATE = 3.0

REPEAT_SIMILARITY_THRESHOLD = 0.85
REPEAT_TIME_WINDOW = 300

QUERY_HISTORY_MAXLEN = 50

MIN_RECOVERY_INTERVAL = 6


def _text_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        words1 = set(text1.lower())
        words2 = set(text2.lower())
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


@dataclass
class StreamEmotionSnapshot:
    chat_value: float = 100.0
    accumulated_fatigue: float = 0.0
    annoyance: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "chat_value": self.chat_value,
            "accumulated_fatigue": self.accumulated_fatigue,
            "annoyance": self.annoyance,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StreamEmotionSnapshot":
        return cls(
            chat_value=data.get("chat_value", 100.0),
            accumulated_fatigue=data.get("accumulated_fatigue", 0.0),
            annoyance=data.get("annoyance", 0.0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class QueryRecord:
    content: str
    content_hash: str
    user_id: str
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def create(cls, content: str, user_id: str) -> "QueryRecord":
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        return cls(content=content, content_hash=content_hash, user_id=user_id)


class StreamEmotionState:
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self._emotion = StreamEmotionSnapshot()
        self._query_history: deque = deque(maxlen=QUERY_HISTORY_MAXLEN)
        self._tool_usage: Dict[str, List[float]] = {}
        self._last_interaction = time.time()
        self._chat_value_recovery_rate = CHAT_VALUE_RECOVERY_RATE
        self._fatigue_decay_rate = FATIGUE_DECAY_RATE
        self._annoyance_decay_rate = ANNOYANCE_DECAY_RATE
        self._repeat_similarity_threshold = REPEAT_SIMILARITY_THRESHOLD
        self._repeat_time_window = REPEAT_TIME_WINDOW

    def _apply_time_recovery(self):
        now = time.time()
        elapsed_minutes = (now - self._emotion.timestamp) / 60.0
        if elapsed_minutes > (MIN_RECOVERY_INTERVAL / 60.0):
            self._emotion.chat_value = min(100.0,
                self._emotion.chat_value + elapsed_minutes * self._chat_value_recovery_rate)
            self._emotion.accumulated_fatigue = max(0.0,
                self._emotion.accumulated_fatigue - elapsed_minutes * self._fatigue_decay_rate)
            self._emotion.annoyance = max(0.0,
                self._emotion.annoyance - elapsed_minutes * self._annoyance_decay_rate)
            self._emotion.timestamp = now

    def record_query(self, content: str, user_id: str, is_admin: bool = False) -> Tuple[int, float]:
        self._apply_time_recovery()
        record = QueryRecord.create(content, user_id)
        now = time.time()
        cutoff = now - self._repeat_time_window
        repeat_count = 0
        max_similarity = 0.0
        content_stripped = content.strip()
        skip_repeat_check = len(content_stripped) <= 3
        if not skip_repeat_check:
            for old_record in self._query_history:
                if old_record.timestamp < cutoff:
                    continue
                similarity = _text_similarity(content, old_record.content)
                max_similarity = max(max_similarity, similarity)
                if similarity >= self._repeat_similarity_threshold:
                    repeat_count += 1
        self._query_history.append(record)
        if not is_admin and repeat_count > 0:
            chat_value_cost = repeat_count * 2
            fatigue_gain = repeat_count * 2
            annoyance_gain = repeat_count * 3
            self._emotion.chat_value = max(0.0, self._emotion.chat_value - chat_value_cost)
            self._emotion.accumulated_fatigue = min(100.0, self._emotion.accumulated_fatigue + fatigue_gain)
            self._emotion.annoyance = min(100.0, self._emotion.annoyance + annoyance_gain)
            logger.debug(f"[{self.stream_id}] 检测到重复问题 x{repeat_count}, "
                        f"聊天值-{chat_value_cost}, 疲劳+{fatigue_gain}, 烦躁+{annoyance_gain}")
        self._last_interaction = now
        return repeat_count, max_similarity

    def record_tool_usage(self, tool_name: str, is_admin: bool = False) -> Tuple[int, bool]:
        if is_admin:
            return 0, False
        self._apply_time_recovery()
        now = time.time()
        if tool_name not in self._tool_usage:
            self._tool_usage[tool_name] = []
        short_term_window = 300
        self._tool_usage[tool_name] = [
            ts for ts in self._tool_usage[tool_name]
            if now - ts < short_term_window
        ]
        self._tool_usage[tool_name].append(now)
        recent_count = len(self._tool_usage[tool_name])
        will_refuse = False
        annoyance_gain = 0
        fatigue_gain = 0
        if recent_count >= 10:
            will_refuse = True
            annoyance_gain = 25
            fatigue_gain = 15
        elif recent_count >= 7:
            annoyance_gain = 15
            fatigue_gain = 10
        elif recent_count >= 5:
            annoyance_gain = 8
            fatigue_gain = 5
        elif recent_count >= 3:
            annoyance_gain = 3
            fatigue_gain = 2
        if annoyance_gain > 0:
            self._emotion.annoyance = min(100.0, self._emotion.annoyance + annoyance_gain)
            self._emotion.accumulated_fatigue = min(100.0, self._emotion.accumulated_fatigue + fatigue_gain)
            logger.debug(f"[{self.stream_id}] {tool_name} 短期调用{recent_count}次, "
                        f"烦躁+{annoyance_gain}, 疲劳+{fatigue_gain}")
        return recent_count, will_refuse

    def get_tool_usage_count(self, tool_name: str) -> int:
        if tool_name not in self._tool_usage:
            return 0
        return len(self._tool_usage[tool_name])

    def reset_tool_usage(self, tool_name: Optional[str] = None):
        if tool_name:
            if tool_name in self._tool_usage:
                self._tool_usage[tool_name] = []
        else:
            self._tool_usage.clear()

    def is_tool_overused(self, tool_name: str, threshold: int = 30) -> bool:
        return self.get_tool_usage_count(tool_name) >= threshold

    def get_emotion_state(self) -> StreamEmotionSnapshot:
        self._apply_time_recovery()
        return self._emotion

    def get_emotion_prompt(self) -> str:
        self._apply_time_recovery()
        chat_value = self._emotion.chat_value
        fatigue = self._emotion.accumulated_fatigue
        annoyance = self._emotion.annoyance
        if chat_value >= 80 and fatigue <= 20 and annoyance <= 20 and not self._tool_usage:
            return ""
        lines = []
        for tool_name, timestamps in self._tool_usage.items():
            count = len(timestamps)
            if count >= 2:
                lines.append(f"你刚才帮忙用了{count}次「{tool_name}」")
        recent_queries = [q for q in self._query_history if time.time() - q.timestamp < 300]
        if len(recent_queries) >= 2:
            last_query = recent_queries[-1]
            repeat_count = 0
            for q in recent_queries[:-1]:
                if _text_similarity(last_query.content, q.content) >= self._repeat_similarity_threshold:
                    repeat_count += 1
            if repeat_count > 0:
                lines.append(f"这个问题刚才已经被问过{repeat_count}次了")
        if not lines:
            return ""
        emotion_hint = ""
        if annoyance > 70 or fatigue > 70:
            emotion_hint = "你现在挺烦的，感觉很累。"
        elif annoyance > 50 or fatigue > 50:
            emotion_hint = "你有点烦了，有点疲劳。"
        elif annoyance > 30 or fatigue > 30:
            emotion_hint = "你稍微有点不耐烦。"
        if chat_value < 30:
            emotion_hint += "你不想聊天了，聊天值很低。"
        return f"[短期记忆] {emotion_hint}\n" + "\n".join(lines)

    def get_repeat_warning(self) -> Optional[str]:
        now = time.time()
        cutoff = now - 300
        recent_queries = [q for q in self._query_history if q.timestamp > cutoff]
        if len(recent_queries) < 2:
            return None
        if not recent_queries:
            return None
        last_query = recent_queries[-1]
        repeat_count = 0
        for q in recent_queries[:-1]:
            if _text_similarity(last_query.content, q.content) >= self._repeat_similarity_threshold:
                repeat_count += 1
        if repeat_count >= 3:
            return "你已经问了好几次类似的问题了，我都回答过了呀"
        elif repeat_count >= 2:
            return "这个问题你刚才好像问过了"
        elif repeat_count >= 1:
            return None
        return None

    def reset_emotion(self):
        self._emotion = StreamEmotionSnapshot()
        self._query_history.clear()
        self._tool_usage.clear()
        logger.info(f"[{self.stream_id}] 情绪状态已重置")

    def to_dict(self) -> Dict:
        return {
            "stream_id": self.stream_id,
            "emotion": self._emotion.to_dict(),
            "last_interaction": self._last_interaction,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StreamEmotionState":
        state = cls(data.get("stream_id", "unknown"))
        state._emotion = StreamEmotionSnapshot.from_dict(data.get("emotion", {}))
        state._last_interaction = data.get("last_interaction", time.time())
        return state


class GlobalStreamEmotionManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._streams: Dict[str, StreamEmotionState] = {}
        self._last_save = time.time()
        self._auto_save_interval = 300
        self._load_all_streams()
        self._initialized = True
        logger.info("[流情绪系统] 初始化完成")

    def _get_stream(self, stream_id: str) -> StreamEmotionState:
        if stream_id not in self._streams:
            self._streams[stream_id] = StreamEmotionState(stream_id)
        return self._streams[stream_id]

    def record_query(self, stream_id: str, content: str, user_id: str,
                     is_admin: bool = False) -> Tuple[int, float]:
        stream = self._get_stream(stream_id)
        result = stream.record_query(content, user_id, is_admin)
        self._auto_save()
        return result

    def record_tool_usage(self, stream_id: str, tool_name: str,
                          is_admin: bool = False) -> Tuple[int, bool]:
        stream = self._get_stream(stream_id)
        result = stream.record_tool_usage(tool_name, is_admin)
        self._auto_save()
        return result

    def get_tool_usage_count(self, stream_id: str, tool_name: str) -> int:
        stream = self._get_stream(stream_id)
        return stream.get_tool_usage_count(tool_name)

    def is_tool_overused(self, stream_id: str, tool_name: str, threshold: int = 30) -> bool:
        stream = self._get_stream(stream_id)
        return stream.is_tool_overused(tool_name, threshold)

    def reset_tool_usage(self, stream_id: str, tool_name: Optional[str] = None):
        stream = self._get_stream(stream_id)
        stream.reset_tool_usage(tool_name)
        self._auto_save()

    def get_emotion_state(self, stream_id: str) -> StreamEmotionSnapshot:
        stream = self._get_stream(stream_id)
        return stream.get_emotion_state()

    def get_emotion_prompt(self, stream_id: str) -> str:
        stream = self._get_stream(stream_id)
        return stream.get_emotion_prompt()

    def get_repeat_warning(self, stream_id: str) -> Optional[str]:
        stream = self._get_stream(stream_id)
        return stream.get_repeat_warning()

    def reset_stream_emotion(self, stream_id: str):
        stream = self._get_stream(stream_id)
        stream.reset_emotion()
        self._save_stream(stream_id)

    def _auto_save(self):
        now = time.time()
        if now - self._last_save > self._auto_save_interval:
            self._save_all_streams()
            self._last_save = now

    def _save_stream(self, stream_id: str):
        if stream_id not in self._streams:
            return
        try:
            from src.common.database.database_model import StreamStateData
            state_key = f"stream_emotion_{stream_id}"
            data = self._streams[stream_id].to_dict()
            StreamStateData.insert(
                stream_id=stream_id,
                state_key=state_key,
                state_value=json.dumps(data, ensure_ascii=False),
                updated_at=time.time()
            ).on_conflict(
                conflict_target=[StreamStateData.stream_id, StreamStateData.state_key],
                update={StreamStateData.state_value: json.dumps(data, ensure_ascii=False),
                        StreamStateData.updated_at: time.time()}
            ).execute()
        except Exception as e:
            logger.debug(f"[流情绪系统] 保存流 {stream_id} 失败: {e}")

    def _save_all_streams(self):
        for stream_id in self._streams:
            self._save_stream(stream_id)
        logger.debug(f"[流情绪系统] 已保存 {len(self._streams)} 个流的情绪数据")

    def _load_all_streams(self):
        try:
            from src.common.database.database_model import StreamStateData
            loaded_count = 0
            records = list(StreamStateData.select().where(
                StreamStateData.state_key.startswith("stream_emotion_")
            ))
            for record in records:
                try:
                    data = json.loads(record.state_value)
                    stream_id = data.get("stream_id", "")
                    if stream_id:
                        self._streams[stream_id] = StreamEmotionState.from_dict(data)
                        loaded_count += 1
                except Exception as e:
                    logger.debug(f"[流情绪系统] 加载记录失败: {e}")
            if loaded_count > 0:
                logger.debug(f"[流情绪系统] 已恢复 {loaded_count} 个流的情绪状态")
        except Exception as e:
            logger.debug(f"[流情绪系统] 数据库未初始化，跳过加载: {e}")


_stream_emotion_manager: Optional[GlobalStreamEmotionManager] = None


def get_stream_emotion_manager() -> GlobalStreamEmotionManager:
    global _stream_emotion_manager
    if _stream_emotion_manager is None:
        _stream_emotion_manager = GlobalStreamEmotionManager()
    return _stream_emotion_manager

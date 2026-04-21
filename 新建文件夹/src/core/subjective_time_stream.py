import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("time_stream")

_stream_instances: Dict[str, "SubjectiveTimeStream"] = {}

# 衰减参数
_FRESHNESS_HALF_LIFE_SEC = 180.0
_STALE_THRESHOLD = 0.15
_REACTIVATION_BOOST = 0.4


@dataclass
class FreshnessRecord:
    """单条消息的心理新鲜度记录"""

    message_id: str = ""
    user_id: str = ""
    content_hash: str = ""
    original_freshness: float = 1.0
    current_freshness: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    reactivation_count: int = 0

    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def is_stale(self) -> bool:
        return self.current_freshness < _STALE_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "freshness": round(self.current_freshness, 3),
            "age_sec": round(self.age_seconds(), 1),
            "stale": self.is_stale(),
            "reactivation_count": self.reactivation_count,
        }


class SubjectiveTimeStream:
    """主观时间流
    维护每条消息的心理新鲜度，实现指数衰减模型。
    支持因话题/用户/情绪而触发的新鲜度重新激活。
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._records: Dict[str, FreshnessRecord] = {}
        self._max_records = 200
        self._last_prune_ts = time.time()
        self._prune_interval = 120.0
        # 主体时间感知
        self._last_glance_ts = time.time()
        self._last_speak_ts = 0.0
        self._last_topic_change_ts = time.time()

    # ────────────────── 消息注册 ──────────────────

    def register_message(
        self, message_id: str, user_id: str = "", content_hash: str = ""
    ) -> FreshnessRecord:
        """注册一条新消息，初始新鲜度为 1.0"""
        now = time.time()
        record = FreshnessRecord(
            message_id=message_id,
            user_id=user_id,
            content_hash=content_hash,
            original_freshness=1.0,
            current_freshness=1.0,
            created_at=now,
            last_accessed_at=now,
        )
        self._records[message_id] = record
        self._maybe_prune(now)
        return record

    def register_batch(self, messages: List[Any]) -> List[FreshnessRecord]:
        """批量注册消息"""
        results = []
        for msg in messages:
            mid = str(getattr(msg, "message_id", "") or "")
            uid = str(getattr(msg, "user_id", "") or "")
            text = str(
                getattr(msg, "processed_plain_text", "")
                or getattr(msg, "plain_text", "")
                or ""
            )
            content_hash = str(hash(text[:100])) if text else ""
            if mid:
                results.append(self.register_message(mid, uid, content_hash))
        return results

    # ────────────────── 新鲜度查询 ──────────────────

    def get_freshness(self, message_id: str) -> float:
        """获取消息当前新鲜度（自动衰减计算）"""
        record = self._records.get(message_id)
        if record is None:
            return 0.0
        self._apply_decay(record)
        return record.current_freshness

    def get_freshness_batch(self, message_ids: List[str]) -> Dict[str, float]:
        """批量获取新鲜度"""
        return {mid: self.get_freshness(mid) for mid in message_ids}

    def fresh_messages(self, threshold: float = 0.3) -> List[FreshnessRecord]:
        """返回仍然新鲜的消息列表"""
        now = time.time()
        result = []
        for record in self._records.values():
            self._apply_decay(record, now)
            if record.current_freshness >= threshold:
                result.append(record)
        result.sort(key=lambda r: r.current_freshness, reverse=True)
        return result

    def stale_message_ids(self) -> List[str]:
        """返回已经过期的消息 ID"""
        now = time.time()
        stale = []
        for mid, record in self._records.items():
            self._apply_decay(record, now)
            if record.is_stale():
                stale.append(mid)
        return stale

    # ────────────────── 重新激活 ──────────────────

    def reactivate(
        self,
        message_id: str,
        boost: float = _REACTIVATION_BOOST,
        reason: str = "",
    ) -> bool:
        """重新激活一条消息的新鲜度"""
        record = self._records.get(message_id)
        if record is None:
            return False
        self._apply_decay(record)
        old = record.current_freshness
        record.current_freshness = min(1.0, record.current_freshness + boost)
        record.reactivation_count += 1
        record.last_accessed_at = time.time()
        logger.debug(
            f"[{self._channel_id}] 消息 {message_id} 重新激活 "
            f"{old:.3f}→{record.current_freshness:.3f} ({reason})"
        )
        return True

    def reactivate_by_user(self, user_id: str, boost: float = 0.3) -> int:
        """激活该用户所有消息"""
        count = 0
        for record in self._records.values():
            if record.user_id == user_id:
                self._apply_decay(record)
                record.current_freshness = min(
                    1.0, record.current_freshness + boost
                )
                record.reactivation_count += 1
                record.last_accessed_at = time.time()
                count += 1
        return count

    def reactivate_by_content(self, keyword: str, boost: float = 0.25) -> int:
        """通过内容关键词激活相关消息（需要外部提供 content_hash 对照）"""
        # 此处简化实现：仅根据 content_hash 前缀匹配
        count = 0
        kw_hash = str(hash(keyword[:50]))
        for record in self._records.values():
            if record.content_hash and kw_hash[:6] in record.content_hash:
                self._apply_decay(record)
                record.current_freshness = min(
                    1.0, record.current_freshness + boost
                )
                record.reactivation_count += 1
                count += 1
        return count

    # ────────────────── 主体时间感知 ──────────────────

    def mark_glance(self) -> None:
        """标记主体看了一眼群"""
        self._last_glance_ts = time.time()

    def mark_speak(self) -> None:
        """标记主体说了话"""
        self._last_speak_ts = time.time()

    def mark_topic_change(self) -> None:
        """标记话题变化"""
        self._last_topic_change_ts = time.time()

    def since_last_glance_sec(self) -> float:
        """距离上次看群的秒数"""
        return time.time() - self._last_glance_ts

    def since_last_speak_sec(self) -> float:
        """距离上次说话的秒数"""
        if self._last_speak_ts <= 0:
            return 9999.0
        return time.time() - self._last_speak_ts

    def since_last_topic_change_sec(self) -> float:
        """距离上次话题变化的秒数"""
        return time.time() - self._last_topic_change_ts

    def time_perception_snapshot(self) -> Dict[str, Any]:
        """主观时间感知快照"""
        return {
            "since_glance_sec": round(self.since_last_glance_sec(), 1),
            "since_speak_sec": round(self.since_last_speak_sec(), 1),
            "since_topic_change_sec": round(
                self.since_last_topic_change_sec(), 1
            ),
            "total_tracked": len(self._records),
            "fresh_count": len(self.fresh_messages()),
        }

    # ────────────────── 内部方法 ──────────────────

    def _apply_decay(
        self, record: FreshnessRecord, now: Optional[float] = None
    ) -> None:
        """应用指数衰减"""
        current = now or time.time()
        elapsed = current - record.last_accessed_at
        if elapsed <= 0:
            return
        decay_factor = math.pow(0.5, elapsed / _FRESHNESS_HALF_LIFE_SEC)
        record.current_freshness = record.current_freshness * decay_factor
        record.last_accessed_at = current

    def _maybe_prune(self, now: float) -> None:
        """定期清除过期记录"""
        if now - self._last_prune_ts < self._prune_interval:
            return
        self._last_prune_ts = now
        if len(self._records) <= self._max_records:
            return
        # 按新鲜度排序，保留较新鲜的
        sorted_items = sorted(
            self._records.items(),
            key=lambda kv: kv[1].current_freshness,
            reverse=True,
        )
        keep = sorted_items[: self._max_records]
        self._records = dict(keep)


# ---------------------------------------------------------------------------
#  全局访问接口
# ---------------------------------------------------------------------------


def get_time_stream(channel_id: str) -> SubjectiveTimeStream:
    if channel_id not in _stream_instances:
        _stream_instances[channel_id] = SubjectiveTimeStream(channel_id)
    return _stream_instances[channel_id]


def remove_time_stream(channel_id: str) -> None:
    _stream_instances.pop(channel_id, None)

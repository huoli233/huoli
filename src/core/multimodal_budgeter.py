import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger
from src.common.singleton import _get_class_lock

logger = get_logger("mm_budgeter")

_budgeter_singleton = None
_budgeter_lock = _get_class_lock("MultimodalBudgeter")

# 预算上限常量
_MAX_HEAVY_ANALYSIS_PER_MINUTE = 3
_IMAGE_CACHE_TTL_SEC = 600.0
_STICKER_CACHE_TTL_SEC = 1800.0


@dataclass
class ImageCacheEntry:
    """图片分析缓存条目"""

    image_hash: str = ""
    description: str = ""
    category: str = ""
    cached_at: float = field(default_factory=time.time)
    hit_count: int = 0

    def is_expired(self, ttl: float = _IMAGE_CACHE_TTL_SEC) -> bool:
        return (time.time() - self.cached_at) > ttl


@dataclass
class BudgetDecision:
    """多模态预算决策"""

    should_analyze: bool = False
    analysis_priority: float = 0.0
    use_cache: bool = False
    cached_description: str = ""
    skip_reason: str = ""
    media_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_analyze": self.should_analyze,
            "analysis_priority": round(self.analysis_priority, 3),
            "use_cache": self.use_cache,
            "skip_reason": self.skip_reason,
            "media_type": self.media_type,
        }


@dataclass
class BudgetBatchSummary:
    """一批多模态输入的预算汇总。"""

    total_media: int = 0
    analyzed_count: int = 0
    cache_hit_count: int = 0
    skipped_count: int = 0
    throttled_count: int = 0
    sticker_count: int = 0
    storm_detected: bool = False
    cached_image: bool = False
    skip_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_media": self.total_media,
            "analyzed_count": self.analyzed_count,
            "cache_hit_count": self.cache_hit_count,
            "skipped_count": self.skipped_count,
            "throttled_count": self.throttled_count,
            "sticker_count": self.sticker_count,
            "storm_detected": self.storm_detected,
            "cached_image": self.cached_image,
            "skip_reasons": dict(self.skip_reasons),
        }


# 发送侧常量
_STICKER_SEND_COOLDOWN_SEC = 120.0
_STICKER_SEND_MAX_PER_WINDOW = 3
_STICKER_SEND_WINDOW_SEC = 600.0
_STICKER_ENERGY_FLOOR = 0.15
_STICKER_MOOD_THRESHOLD = 0.25


@dataclass
class StickerSendDecision:
    """表情包发送策略结果"""

    allow_send: bool = False
    send_priority: float = 0.0
    veto_reason: str = ""
    recommended_category: str = ""
    cooldown_remaining_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow_send": self.allow_send,
            "send_priority": round(self.send_priority, 3),
            "veto_reason": self.veto_reason,
            "recommended_category": self.recommended_category,
            "cooldown_remaining_sec": round(self.cooldown_remaining_sec, 1),
        }


@dataclass
class _ChannelSendRecord:
    """频道级表情包发送记录"""

    send_timestamps: List[float] = field(default_factory=list)
    last_category: str = ""
    consecutive_same_category: int = 0


class MultimodalBudgeter:
    """多模态预算器
    控制图片、表情包、多图消息的分析成本，
    通过判重、缓存命中和预算余量限流，
    避免一见图就走重型识别链路。
    """

    def __init__(self):
        self._image_cache: Dict[str, ImageCacheEntry] = {}
        self._heavy_analysis_timestamps: List[float] = []
        self._max_cache_size = 200
        self._last_batch_summary = BudgetBatchSummary()
        self._send_records: Dict[str, _ChannelSendRecord] = {}
        self._send_records_cap = 100

    def evaluate(
        self,
        *,
        media_type: str = "image",
        image_data: Optional[bytes] = None,
        image_url: str = "",
        image_hash: str = "",
        context_relevance: float = 0.5,
        current_budget_ratio: float = 1.0,
        is_sticker: bool = False,
    ) -> BudgetDecision:
        """评估是否对多模态内容进行分析"""
        fingerprint = self._resolve_fingerprint(
            image_data=image_data,
            image_url=image_url,
            image_hash=image_hash,
        )
        cached_decision = self._try_cache_hit(
            fingerprint=fingerprint,
            media_type=media_type,
            is_sticker=is_sticker,
        )
        if cached_decision is not None:
            return cached_decision
        return self._evaluate_uncached(
            media_type=media_type,
            context_relevance=context_relevance,
            current_budget_ratio=current_budget_ratio,
            is_sticker=is_sticker,
        )

    def evaluate_batch(
        self,
        media_items: List[Dict[str, Any]],
        current_budget_ratio: float = 1.0,
    ) -> List[BudgetDecision]:
        """批量评估多模态内容"""
        results: List[BudgetDecision] = []
        if not media_items:
            self._last_batch_summary = BudgetBatchSummary()
            return results
        cached_flags = [
            self._is_cache_available(
                fingerprint=self._resolve_fingerprint(
                    image_data=item.get("image_data"),
                    image_url=str(item.get("image_url", "") or ""),
                    image_hash=str(item.get("image_hash", "") or ""),
                ),
                is_sticker=bool(item.get("is_sticker", False)),
            )
            for item in media_items
        ]
        sample_limit = self._batch_sample_limit(len(media_items))
        prioritized_indices = sorted(
            [idx for idx, cached in enumerate(cached_flags) if not cached],
            key=lambda idx: (
                1 if media_items[idx].get("is_sticker", False) else 0,
                float(media_items[idx].get("context_relevance", 0.5) or 0.5),
            ),
            reverse=True,
        )
        selected_indices = set(prioritized_indices[:sample_limit])
        storm_detected = len(media_items) >= 5
        for idx, item in enumerate(media_items):
            media_type = str(item.get("media_type", "image") or "image")
            is_sticker = bool(item.get("is_sticker", False))
            fingerprint = self._resolve_fingerprint(
                image_data=item.get("image_data"),
                image_url=str(item.get("image_url", "") or ""),
                image_hash=str(item.get("image_hash", "") or ""),
            )
            cached_decision = self._try_cache_hit(
                fingerprint=fingerprint,
                media_type=media_type,
                is_sticker=is_sticker,
            )
            if cached_decision is not None:
                results.append(cached_decision)
                continue
            if len(media_items) > sample_limit and idx not in selected_indices:
                results.append(
                    BudgetDecision(
                        should_analyze=False,
                        skip_reason=("刷图风暴节流" if storm_detected else "多图抽样节流"),
                        media_type=media_type,
                    )
                )
                continue
            results.append(
                self._evaluate_uncached(
                    media_type=media_type,
                    context_relevance=float(item.get("context_relevance", 0.5) or 0.5),
                    current_budget_ratio=current_budget_ratio,
                    is_sticker=is_sticker,
                )
            )
        self._last_batch_summary = self._summarize_batch(
            results,
            total_media=len(media_items),
            storm_detected=storm_detected,
        )
        return results

    def store_analysis_result(self, fingerprint: str, description: str, category: str = "") -> None:
        """存储分析结果到缓存"""
        if not fingerprint:
            return
        entry = ImageCacheEntry(
            image_hash=fingerprint,
            description=description,
            category=category,
        )
        self._image_cache[fingerprint] = entry
        self._prune_cache()

    def cache_stats(self) -> Dict[str, Any]:
        """缓存统计"""
        total = len(self._image_cache)
        expired = sum(1 for e in self._image_cache.values() if e.is_expired())
        return {
            "total_cached": total,
            "expired": expired,
            "active": total - expired,
            "recent_analyses": len(self._heavy_analysis_timestamps),
            "last_batch": self._last_batch_summary.to_dict(),
        }

    def last_batch_summary(self) -> Dict[str, Any]:
        return self._last_batch_summary.to_dict()

    # ────────────────── 内部方法 ──────────────────

    def _resolve_fingerprint(
        self,
        *,
        image_data: Optional[bytes] = None,
        image_url: str = "",
        image_hash: str = "",
    ) -> str:
        fingerprint = str(image_hash or "").strip()
        if not fingerprint and image_data:
            fingerprint = hashlib.md5(image_data[:4096]).hexdigest()
        if not fingerprint and image_url:
            fingerprint = hashlib.md5(image_url.encode()[:256]).hexdigest()
        return fingerprint

    def _try_cache_hit(
        self,
        *,
        fingerprint: str,
        media_type: str,
        is_sticker: bool,
    ) -> Optional[BudgetDecision]:
        if not self._is_cache_available(fingerprint=fingerprint, is_sticker=is_sticker):
            return None
        cached = self._image_cache.get(fingerprint)
        if cached is not None:
            cached.hit_count += 1
            return BudgetDecision(
                should_analyze=False,
                use_cache=True,
                cached_description=cached.description,
                skip_reason=f"缓存命中(命中{cached.hit_count}次)",
                media_type=media_type,
            )
        return None

    def _is_cache_available(self, *, fingerprint: str, is_sticker: bool) -> bool:
        if not fingerprint:
            return False
        cached = self._image_cache.get(fingerprint)
        ttl = _STICKER_CACHE_TTL_SEC if is_sticker else _IMAGE_CACHE_TTL_SEC
        return bool(cached and not cached.is_expired(ttl))

    def _evaluate_uncached(
        self,
        *,
        media_type: str,
        context_relevance: float,
        current_budget_ratio: float,
        is_sticker: bool,
    ) -> BudgetDecision:
        decision = BudgetDecision(media_type=media_type)
        if not self._has_budget():
            decision.should_analyze = False
            decision.skip_reason = "分析预算耗尽"
            return decision
        if is_sticker:
            if current_budget_ratio < 0.1:
                decision.should_analyze = False
                decision.skip_reason = "精力不足"
                return decision
            decision.should_analyze = True
            decision.analysis_priority = 0.35
            self._record_analysis()
            return decision
        if current_budget_ratio < 0.2:
            decision.should_analyze = False
            decision.skip_reason = "精力不足"
            return decision
        if context_relevance < 0.2:
            decision.should_analyze = False
            decision.skip_reason = "与上下文关联度低"
            return decision
        decision.should_analyze = True
        decision.analysis_priority = min(1.0, context_relevance * 0.8 + current_budget_ratio * 0.2)
        self._record_analysis()
        return decision

    def _batch_sample_limit(self, batch_size: int) -> int:
        if batch_size <= 3:
            return batch_size
        if batch_size <= 6:
            return 2
        return 1

    def _summarize_batch(
        self,
        decisions: List[BudgetDecision],
        *,
        total_media: int,
        storm_detected: bool,
    ) -> BudgetBatchSummary:
        summary = BudgetBatchSummary(total_media=total_media, storm_detected=storm_detected)
        for decision in decisions:
            if decision.should_analyze:
                summary.analyzed_count += 1
            else:
                summary.skipped_count += 1
            if decision.use_cache:
                summary.cache_hit_count += 1
            if decision.media_type == "sticker":
                summary.sticker_count += 1
            if decision.skip_reason:
                summary.skip_reasons[decision.skip_reason] = summary.skip_reasons.get(decision.skip_reason, 0) + 1
            if decision.skip_reason in {"多图抽样节流", "刷图风暴节流"}:
                summary.throttled_count += 1
        summary.cached_image = summary.cache_hit_count > 0
        return summary

    def _has_budget(self) -> bool:
        """检查是否还有重型分析预算"""
        now = time.time()
        cutoff = now - 60.0
        self._heavy_analysis_timestamps = [ts for ts in self._heavy_analysis_timestamps if ts > cutoff]
        return len(self._heavy_analysis_timestamps) < _MAX_HEAVY_ANALYSIS_PER_MINUTE

    def _record_analysis(self) -> None:
        self._heavy_analysis_timestamps.append(time.time())

    def _prune_cache(self) -> None:
        if len(self._image_cache) <= self._max_cache_size:
            return
        expired_keys = [k for k, v in self._image_cache.items() if v.is_expired()]
        for k in expired_keys:
            del self._image_cache[k]
        if len(self._image_cache) > self._max_cache_size:
            sorted_keys = sorted(
                self._image_cache.keys(),
                key=lambda k: self._image_cache[k].cached_at,
            )
            for k in sorted_keys[: len(self._image_cache) - self._max_cache_size]:
                del self._image_cache[k]

    # ═══════════════════════════════════════════════════
    #  发送侧策略判断
    # ═══════════════════════════════════════════════════

    def evaluate_sticker_send(
        self,
        *,
        channel_id: str,
        energy_ratio: float = 1.0,
        mood_valence: float = 0.0,
        social_willingness: float = 0.5,
        conversation_turn_count: int = 0,
        incoming_has_sticker: bool = False,
        desired_category: str = "",
    ) -> StickerSendDecision:
        """判断是否在本轮回复中附带表情包

        参数:
            channel_id: 频道标识
            energy_ratio: 当前精力比例 [0,1]
            mood_valence: 情绪效价 [-1,1]，正值表正面情绪
            social_willingness: 社交意愿 [0,1]
            conversation_turn_count: 近期对话轮次
            incoming_has_sticker: 对方是否发了表情包
            desired_category: 期望的表情类别（happy/sad/funny等）
        """
        now = time.time()
        record = self._get_send_record(channel_id)
        decision = StickerSendDecision()
        # 冷却检查
        cooldown_left = self._check_send_cooldown(record, now)
        if cooldown_left > 0:
            decision.veto_reason = f"发送冷却中({cooldown_left:.0f}s)"
            decision.cooldown_remaining_sec = cooldown_left
            return decision
        # 窗口频次检查
        window_count = self._count_sends_in_window(record, now)
        if window_count >= _STICKER_SEND_MAX_PER_WINDOW:
            decision.veto_reason = f"窗口频次已满({window_count}/{_STICKER_SEND_MAX_PER_WINDOW})"
            return decision
        # 精力门槛
        if energy_ratio < _STICKER_ENERGY_FLOOR:
            decision.veto_reason = "精力不足"
            return decision
        # 计算发送优先级
        priority = self._compute_send_priority(
            energy_ratio=energy_ratio,
            mood_valence=mood_valence,
            social_willingness=social_willingness,
            conversation_turn_count=conversation_turn_count,
            incoming_has_sticker=incoming_has_sticker,
            window_count=window_count,
        )
        # 优先级低于阈值
        if priority < _STICKER_MOOD_THRESHOLD:
            decision.veto_reason = "发送意愿不足"
            decision.send_priority = priority
            return decision
        # 确定推荐类别
        category = self._resolve_send_category(
            desired_category=desired_category,
            mood_valence=mood_valence,
            record=record,
        )
        decision.allow_send = True
        decision.send_priority = priority
        decision.recommended_category = category
        return decision

    def confirm_sticker_sent(self, channel_id: str, category: str = "") -> None:
        """确认表情包已发送，记录时间"""
        now = time.time()
        record = self._get_send_record(channel_id)
        record.send_timestamps.append(now)
        if category:
            if category == record.last_category:
                record.consecutive_same_category += 1
            else:
                record.consecutive_same_category = 1
            record.last_category = category
        self._trim_send_timestamps(record, now)

    def send_stats(self, channel_id: str) -> Dict[str, Any]:
        """频道发送侧统计"""
        now = time.time()
        record = self._send_records.get(channel_id)
        if record is None:
            return {
                "window_count": 0,
                "cooldown_remaining": 0.0,
                "last_category": "",
            }
        return {
            "window_count": self._count_sends_in_window(record, now),
            "cooldown_remaining": max(0.0, self._check_send_cooldown(record, now)),
            "last_category": record.last_category,
            "consecutive_same": record.consecutive_same_category,
        }

    # ────────────────── 发送侧内部方法 ──────────────────

    def _get_send_record(self, channel_id: str) -> _ChannelSendRecord:
        record = self._send_records.get(channel_id)
        if record is None:
            record = _ChannelSendRecord()
            self._send_records[channel_id] = record
            if len(self._send_records) > self._send_records_cap:
                oldest_ch = min(
                    self._send_records,
                    key=lambda ch: (
                        self._send_records[ch].send_timestamps[-1] if self._send_records[ch].send_timestamps else 0.0
                    ),
                )
                del self._send_records[oldest_ch]
        return record

    def _check_send_cooldown(self, record: _ChannelSendRecord, now: float) -> float:
        if not record.send_timestamps:
            return 0.0
        elapsed = now - record.send_timestamps[-1]
        return max(0.0, _STICKER_SEND_COOLDOWN_SEC - elapsed)

    def _count_sends_in_window(self, record: _ChannelSendRecord, now: float) -> int:
        cutoff = now - _STICKER_SEND_WINDOW_SEC
        return sum(1 for ts in record.send_timestamps if ts > cutoff)

    def _compute_send_priority(
        self,
        *,
        energy_ratio: float,
        mood_valence: float,
        social_willingness: float,
        conversation_turn_count: int,
        incoming_has_sticker: bool,
        window_count: int,
    ) -> float:
        # 基础分：情绪积极面加分
        base = 0.15 + max(0.0, mood_valence) * 0.25
        # 社交意愿加分
        base += social_willingness * 0.2
        # 精力充沛加分
        base += max(0.0, energy_ratio - 0.5) * 0.15
        # 对方发了表情包则互动加分
        if incoming_has_sticker:
            base += 0.2
        # 对话轮次少时（刚开场）抑制表情包
        if conversation_turn_count < 3:
            base -= 0.15
        # 频次惩罚：窗口内越多优先级越低
        base -= window_count * 0.1
        return max(0.0, min(1.0, base))

    def _resolve_send_category(
        self,
        *,
        desired_category: str,
        mood_valence: float,
        record: _ChannelSendRecord,
    ) -> str:
        if desired_category:
            # 连续同类别超2次，推荐换一种
            if desired_category == record.last_category and record.consecutive_same_category >= 2:
                return self._mood_based_category(mood_valence)
            return desired_category
        return self._mood_based_category(mood_valence)

    @staticmethod
    def _mood_based_category(mood_valence: float) -> str:
        if mood_valence > 0.4:
            return "happy"
        if mood_valence > 0.1:
            return "friendly"
        if mood_valence < -0.3:
            return "sympathetic"
        return "neutral"

    def _trim_send_timestamps(self, record: _ChannelSendRecord, now: float) -> None:
        cutoff = now - _STICKER_SEND_WINDOW_SEC * 2
        record.send_timestamps = [ts for ts in record.send_timestamps if ts > cutoff]


def get_multimodal_budgeter() -> MultimodalBudgeter:
    global _budgeter_singleton
    if _budgeter_singleton is None:
        with _budgeter_lock:
            if _budgeter_singleton is None:
                _budgeter_singleton = MultimodalBudgeter()
    return _budgeter_singleton

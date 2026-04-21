import time
import asyncio
from typing import List, Optional, Dict
from dataclasses import dataclass
from src.common.logger import get_logger

logger = get_logger("风格反射")


@dataclass
class ReflectEntry:
    style_id: int
    scene: str
    manner: str
    action: str
    reason: str
    timestamp: float


class StyleReflector:
    def __init__(self, channel_id: str) -> None:
        self.channel_id = channel_id
        self._reflect_interval = 3600.0
        self._last_reflect_ts = 0.0
        self._reflect_lock: asyncio.Lock | None = None
        self._pending_reflects: List[Dict] = []
        self._auto_enabled = True
        self._quality_threshold = 0.3

    def should_trigger_reflect(self) -> bool:
        if not self._auto_enabled:
            return False
        elapsed = time.time() - self._last_reflect_ts
        return elapsed >= self._reflect_interval

    async def trigger_reflect_task(self):
        if self._reflect_lock is None:
            self._reflect_lock = asyncio.Lock()
        async with self._reflect_lock:
            if not self.should_trigger_reflect():
                return
            self._last_reflect_ts = time.time()
            try:
                await self._perform_reflection()
            except Exception as e:
                logger.error(f"反思任务失败: {e}")

    async def _perform_reflection(self):
        unchecked = await self._fetch_unchecked_styles(limit=20)
        if not unchecked:
            return
        accepted = 0
        rejected = 0
        for style in unchecked:
            quality = await self._evaluate_style_quality(style)
            if quality < self._quality_threshold:
                await self._mark_style_rejected(style["id"])
                rejected += 1
            else:
                await self._mark_style_checked(style["id"])
                accepted += 1
        logger.info(
            f"频道 {
                self.channel_id} 反思完成: 检查 {
                len(unchecked)} 条，接受 {accepted} 条，拒绝 {rejected} 条"
        )

    async def _fetch_unchecked_styles(self, limit: int = 20) -> List[Dict]:
        try:
            from src.common.database.database_model import Expression

            records = (
                Expression.select()
                .where(Expression.chat_id == self.channel_id)
                .where(Expression.checked is False)
                .order_by(Expression.create_date.asc())
                .limit(limit)
            )
            return [
                {
                    "id": r.id,
                    "situation": r.situation,
                    "style": r.style,
                    "content_list": r.content_list,
                    "count": r.count,
                }
                for r in records
            ]
        except Exception as e:
            logger.error(f"获取未检查风格失败: {e}")
            return []

    async def _evaluate_style_quality(self, style: Dict) -> float:
        scene = style.get("situation", "")
        manner = style.get("style", "")
        if not scene or not manner:
            return 0.0
        if len(scene) < 3 or len(manner) < 2:
            return 0.1
        if len(scene) > 40 or len(manner) > 40:
            return 0.2
        if scene == manner:
            return 0.0
        count = style.get("count", 1)
        if count >= 3:
            return 0.8
        if count >= 2:
            return 0.6
        return 0.5

    async def _mark_style_checked(self, style_id: int):
        try:
            from src.common.database.database_model import Expression

            Expression.update(checked=True).where(
                Expression.id == style_id
            ).execute()
        except Exception as e:
            logger.error(f"标记风格已检查失败: {e}")

    async def _mark_style_rejected(self, style_id: int):
        try:
            from src.common.database.database_model import Expression

            Expression.update(rejected=True, checked=True).where(
                Expression.id == style_id
            ).execute()
        except Exception as e:
            logger.error(f"标记风格被拒绝失败: {e}")

    def add_pending_reflect(self, style_id: int, reason: str):
        self._pending_reflects.append(
            {
                "style_id": style_id,
                "reason": reason,
                "timestamp": time.time(),
            }
        )

    async def process_pending_reflects(self):
        if not self._pending_reflects:
            return
        pending = self._pending_reflects[:]
        self._pending_reflects.clear()
        for item in pending:
            style_id = item["style_id"]
            reason = item["reason"]
            if reason == "negative_feedback":
                await self._mark_style_rejected(style_id)
            elif reason == "positive_feedback":
                await self._boost_style_count(style_id)

    async def _boost_style_count(self, style_id: int, boost: int = 1):
        try:
            from src.common.database.database_model import Expression

            record = Expression.get_or_none(Expression.id == style_id)
            if record:
                record.count += boost
                record.last_active_time = time.time()
                record.save()
        except Exception as e:
            logger.error(f"提升风格计数失败: {e}")

    async def manual_accept(self, style_id: int):
        await self._mark_style_checked(style_id)
        await self._boost_style_count(style_id, boost=1)
        logger.info(f"手动接受风格 {style_id}")

    async def manual_reject(self, style_id: int, reason: str = ""):
        await self._mark_style_rejected(style_id)
        logger.info(f"手动拒绝风格 {style_id}, 原因: {reason}")

    def get_reflect_stats(self) -> Dict:
        return {
            "channel_id": self.channel_id,
            "last_reflect_ts": self._last_reflect_ts,
            "pending_count": len(self._pending_reflects),
            "auto_enabled": self._auto_enabled,
            "quality_threshold": self._quality_threshold,
        }

    def set_quality_threshold(self, value: float):
        self._quality_threshold = max(0.0, min(1.0, value))

    def set_auto_enabled(self, enabled: bool):
        self._auto_enabled = enabled

    def set_reflect_interval(self, seconds: float):
        self._reflect_interval = max(300.0, seconds)


class ReflectScheduler:
    def __init__(self):
        self._running = False
        self._interval = 1800.0

    async def start(self):
        if self._running:
            return
        self._running = True
        logger.info("反思调度器已启动")
        while self._running:
            await asyncio.sleep(self._interval)
            await self._run_all_reflects()

    def stop(self):
        self._running = False
        logger.info("反思调度器已停止")

    async def _run_all_reflects(self):
        for reflector in _reflectors.values():
            await reflector.trigger_reflect_task()

    def set_interval(self, seconds: float):
        self._interval = max(300.0, seconds)


class QualityEvaluator:
    def __init__(self):
        self._min_scene_len = 3
        self._max_scene_len = 40
        self._min_manner_len = 2
        self._max_manner_len = 40
        self._blacklist_words: set = set()

    def evaluate(
        self, scene: str, manner: str, context: Optional[str] = None
    ) -> float:
        if not scene or not manner:
            return 0.0
        score = 0.5
        if len(scene) < self._min_scene_len:
            score -= 0.2
        if len(scene) > self._max_scene_len:
            score -= 0.1
        if len(manner) < self._min_manner_len:
            score -= 0.2
        if len(manner) > self._max_manner_len:
            score -= 0.1
        if scene == manner:
            score -= 0.3
        if context and len(context) > 10:
            score += 0.2
        for word in self._blacklist_words:
            if word in scene.lower() or word in manner.lower():
                score -= 0.3
                break
        return max(0.0, min(1.0, score))

    def add_blacklist_word(self, word: str):
        self._blacklist_words.add(word.lower())

    def remove_blacklist_word(self, word: str):
        self._blacklist_words.discard(word.lower())

    def set_length_limits(
        self, min_scene: int, max_scene: int, min_manner: int, max_manner: int
    ):
        self._min_scene_len = max(1, min_scene)
        self._max_scene_len = max(self._min_scene_len, max_scene)
        self._min_manner_len = max(1, min_manner)
        self._max_manner_len = max(self._min_manner_len, max_manner)


class ReflectHistory:
    def __init__(self):
        self._history: List[ReflectEntry] = []
        self._max_entries = 500

    def add_entry(
        self, style_id: int, scene: str, manner: str, action: str, reason: str
    ):
        entry = ReflectEntry(
            style_id=style_id,
            scene=scene,
            manner=manner,
            action=action,
            reason=reason,
            timestamp=time.time(),
        )
        self._history.append(entry)
        if len(self._history) > self._max_entries:
            self._history = self._history[-self._max_entries:]

    def get_recent(self, limit: int = 20) -> List[Dict]:
        return [
            {
                "style_id": e.style_id,
                "scene": e.scene,
                "manner": e.manner,
                "action": e.action,
                "reason": e.reason,
                "timestamp": e.timestamp,
            }
            for e in self._history[-limit:]
        ]

    def get_by_style(self, style_id: int) -> List[Dict]:
        return [
            {
                "style_id": e.style_id,
                "scene": e.scene,
                "manner": e.manner,
                "action": e.action,
                "reason": e.reason,
                "timestamp": e.timestamp,
            }
            for e in self._history
            if e.style_id == style_id
        ]

    def get_stats(self) -> Dict:
        actions = {}
        for e in self._history:
            actions[e.action] = actions.get(e.action, 0) + 1
        return {"total": len(self._history), "by_action": actions}

    def clear_old_entries(self, days: int = 30):
        cutoff = time.time() - (days * 86400)
        self._history = [e for e in self._history if e.timestamp >= cutoff]


class AutoReflectEngine:
    def __init__(self):
        self._enabled = True
        self._interval = 1800.0
        self._last_run = 0.0
        self._batch_size = 20

    async def run_auto_reflect(self, channel_id: str):
        if not self._enabled:
            return
        if time.time() - self._last_run < self._interval:
            return
        self._last_run = time.time()
        reflector = get_style_reflector(channel_id)
        await reflector.trigger_reflect_task()

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def set_interval(self, seconds: float):
        self._interval = max(300.0, seconds)

    def set_batch_size(self, size: int):
        self._batch_size = max(1, min(100, size))


class StyleFeedbackHandler:
    def __init__(self):
        self._feedback_log: List[Dict] = []
        self._max_log_size = 200

    def record_positive_feedback(
        self, style_id: int, scene: str, manner: str, user_id: str = ""
    ):
        self._feedback_log.append(
            {
                "style_id": style_id,
                "scene": scene,
                "manner": manner,
                "type": "positive",
                "user_id": user_id,
                "timestamp": time.time(),
            }
        )
        self._trim_log()
        logger.debug(f"记录正面反馈: 风格 {style_id}")

    def record_negative_feedback(
        self, style_id: int, scene: str, manner: str, user_id: str = ""
    ):
        self._feedback_log.append(
            {
                "style_id": style_id,
                "scene": scene,
                "manner": manner,
                "type": "negative",
                "user_id": user_id,
                "timestamp": time.time(),
            }
        )
        self._trim_log()
        logger.debug(f"记录负面反馈: 风格 {style_id}")

    def get_feedback_stats(self, hours: int = 24) -> Dict:
        cutoff = time.time() - (hours * 3600)
        recent = [f for f in self._feedback_log if f["timestamp"] >= cutoff]
        positive = sum(1 for f in recent if f["type"] == "positive")
        negative = sum(1 for f in recent if f["type"] == "negative")
        return {
            "total": len(recent),
            "positive": positive,
            "negative": negative,
            "ratio": (
                positive / max(1, (positive + negative))
                if (positive + negative) > 0
                else 0.5
            ),
        }

    def get_style_feedback(self, style_id: int) -> Dict:
        style_feedbacks = [
            f for f in self._feedback_log if f["style_id"] == style_id
        ]
        positive = sum(1 for f in style_feedbacks if f["type"] == "positive")
        negative = sum(1 for f in style_feedbacks if f["type"] == "negative")
        return {
            "style_id": style_id,
            "total": len(style_feedbacks),
            "positive": positive,
            "negative": negative,
        }

    def _trim_log(self):
        if len(self._feedback_log) > self._max_log_size:
            self._feedback_log = self._feedback_log[-self._max_log_size:]


class StyleCleanupManager:
    def __init__(self):
        self._max_age_days = 30
        self._min_count_threshold = 1
        self._cleanup_interval = 86400.0

    async def cleanup_channel_styles(self, channel_id: str) -> Dict:
        try:
            from src.common.database.database_model import Expression

            cutoff_ts = time.time() - (self._max_age_days * 86400)
            delete_query = Expression.delete().where(
                (Expression.chat_id == channel_id)
                & (
                    (Expression.last_active_time < cutoff_ts)
                    | (Expression.rejected)
                    & (Expression.count < self._min_count_threshold)
                )
            )
            deleted = delete_query.execute()
            logger.info(f"频道 {channel_id} 清理了 {deleted} 条过期风格")
            return {"deleted": deleted, "channel_id": channel_id}
        except Exception as e:
            logger.error(f"清理风格失败: {e}")
            return {"deleted": 0, "error": str(e)}

    async def cleanup_all_channels(self) -> Dict:
        try:
            from src.common.database.database_model import Expression

            channels = Expression.select(Expression.chat_id).distinct()
            total_deleted = 0
            for channel in channels:
                result = await self.cleanup_channel_styles(channel.chat_id)
                total_deleted += result.get("deleted", 0)
            return {
                "total_deleted": total_deleted,
                "channels_processed": len(list(channels)),
            }
        except Exception as e:
            logger.error(f"清理所有频道风格失败: {e}")
            return {"total_deleted": 0, "error": str(e)}

    def set_max_age_days(self, days: int):
        self._max_age_days = max(1, days)

    def set_min_count_threshold(self, count: int):
        self._min_count_threshold = max(0, count)


_reflectors: Dict[str, StyleReflector] = {}
_reflect_scheduler: Optional[ReflectScheduler] = None
_quality_evaluator: Optional[QualityEvaluator] = None
_reflect_history: Optional[ReflectHistory] = None
_auto_reflect_engine: Optional[AutoReflectEngine] = None
_style_feedback_handler: Optional[StyleFeedbackHandler] = None
_style_cleanup_manager: Optional[StyleCleanupManager] = None


def get_style_reflector(channel_id: str) -> StyleReflector:
    if channel_id not in _reflectors:
        _reflectors[channel_id] = StyleReflector(channel_id)
    return _reflectors[channel_id]


def reset_style_reflector(channel_id: str) -> None:
    if channel_id in _reflectors:
        del _reflectors[channel_id]


def get_reflect_scheduler() -> ReflectScheduler:
    global _reflect_scheduler
    if _reflect_scheduler is None:
        _reflect_scheduler = ReflectScheduler()
    return _reflect_scheduler


def get_quality_evaluator() -> QualityEvaluator:
    global _quality_evaluator
    if _quality_evaluator is None:
        _quality_evaluator = QualityEvaluator()
    return _quality_evaluator


def get_reflect_history() -> ReflectHistory:
    global _reflect_history
    if _reflect_history is None:
        _reflect_history = ReflectHistory()
    return _reflect_history


def get_auto_reflect_engine() -> AutoReflectEngine:
    global _auto_reflect_engine
    if _auto_reflect_engine is None:
        _auto_reflect_engine = AutoReflectEngine()
    return _auto_reflect_engine


def get_style_feedback_handler() -> StyleFeedbackHandler:
    global _style_feedback_handler
    if _style_feedback_handler is None:
        _style_feedback_handler = StyleFeedbackHandler()
    return _style_feedback_handler


def get_style_cleanup_manager() -> StyleCleanupManager:
    global _style_cleanup_manager
    if _style_cleanup_manager is None:
        _style_cleanup_manager = StyleCleanupManager()
    return _style_cleanup_manager


async def start_reflect_system():
    scheduler = get_reflect_scheduler()
    await scheduler.start()


def stop_reflect_system():
    global _reflect_scheduler
    if _reflect_scheduler:
        _reflect_scheduler.stop()

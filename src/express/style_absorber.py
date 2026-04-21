import time
import json
import re
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("风格吸收")


_INVALID_CHAR_RE = re.compile(r"[\n\r\t]")


def validate_style_pair(scene: str, manner: str) -> bool:
    if not scene or not manner:
        return False
    if len(scene) < 2 or len(manner) < 2:
        return False
    if len(scene) > 50 or len(manner) > 50:
        return False
    if scene.strip() == manner.strip():
        return False
    if _INVALID_CHAR_RE.search(scene) or _INVALID_CHAR_RE.search(manner):
        return False
    return True


def compute_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    words1 = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", text1.lower()))
    words2 = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", text2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def filter_content(text: str) -> str:
    text = re.sub(r"\[CQ:[^\]]+\]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    return text.strip()


@dataclass
class StyleRecord:
    scene: str
    manner: str
    context: str = ""
    up_content: str = ""
    count: int = 1
    last_active_at: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    checked: bool = False
    rejected: bool = False


class StyleAbsorber:
    def __init__(self, channel_id: str) -> None:
        self.channel_id = channel_id
        self.channel_name = channel_id
        self.last_absorb_ts: float = time.time()
        self._absorb_lock: asyncio.Lock | None = None
        self.absorb_enabled = True
        self.absorb_intensity = 1.0
        self.min_packets_for_absorb = 15
        self.min_absorb_interval = 120.0
        self._style_cache: Dict[str, StyleRecord] = {}
        self._max_style_cache = 5000

    def should_trigger_absorb(self) -> bool:
        if not self.absorb_enabled:
            return False
        elapsed = time.time() - self.last_absorb_ts
        if elapsed < self.min_absorb_interval:
            return False
        return True

    async def trigger_absorb_task(self):
        if self._absorb_lock is None:
            self._absorb_lock = asyncio.Lock()
        async with self._absorb_lock:
            if not self.should_trigger_absorb():
                return
            absorb_start_ts = time.time()
            prev_absorb_ts = self.last_absorb_ts
            self.last_absorb_ts = absorb_start_ts
            try:
                logger.info(f"在频道 {self.channel_name} 吸收表达风格")
                absorbed = await self.absorb_and_persist(
                    count=25, since_ts=prev_absorb_ts
                )
                if absorbed:
                    logger.info(
                        f"频道 {
                            self.channel_name} 风格吸收完成，共 {
                            len(absorbed)} 条"
                    )
                else:
                    logger.warning(
                        f"频道 {self.channel_name} 未吸收到有效风格"
                    )
            except Exception as e:
                logger.error(f"频道 {self.channel_name} 吸收失败: {e}")

    async def absorb_and_persist(
        self, count: int = 10, since_ts: Optional[float] = None
    ) -> List[Tuple[str, str, str, str]]:
        absorbed_styles = await self.absorb_styles(count, since_ts=since_ts)
        if absorbed_styles is None:
            logger.info("未吸收到表达风格")
            return []
        display_str = ""
        for scene, manner, _ctx, _up in absorbed_styles:
            display_str += f"{scene}->{manner}\n"
        logger.info(f"在 {self.channel_name} 吸收到风格:\n{display_str}")
        now_ts = time.time()
        for scene, manner, ctx, up in absorbed_styles:
            await self._upsert_style_record(
                scene=scene,
                manner=manner,
                context=ctx,
                up_content=up,
                current_ts=now_ts,
            )
        return absorbed_styles

    async def absorb_styles(
        self, count: int, since_ts: Optional[float] = None
    ) -> Optional[List[Tuple[str, str, str, str]]]:
        try:
            from src.common.database.database_model import Messages

            query = (
                Messages.select()
                .where(Messages.chat_info_group_id == self.channel_id)
                .order_by(Messages.time.desc())
                .limit(count * 3)
            )
            messages = list(query)
            if len(messages) < self.min_packets_for_absorb:
                return []
            styles = await self._extract_styles_from_messages(messages)
            return styles
        except Exception as e:
            logger.error(f"吸收风格失败: {e}")
            return []

    async def _extract_styles_from_messages(
        self, messages: List
    ) -> List[Tuple[str, str, str, str]]:
        styles = []
        seen = set()
        for msg in messages:
            text = msg.processed_plain_text or ""
            if not text or len(text) < 5:
                continue
            text = filter_content(text)
            extracted = self._extract_style_patterns(text)
            for scene, manner in extracted:
                key = f"{scene}|{manner}"
                if key in seen:
                    continue
                seen.add(key)
                if validate_style_pair(scene, manner):
                    styles.append((scene, manner, text[:100], ""))
        return styles[:30]

    def _extract_style_patterns(self, text: str) -> List[Tuple[str, str]]:
        patterns = []
        pattern_regex = re.compile(
            r'当["\"](.+?)["\"]时[，,].*?[使用有]["\"](.+?)["\"]'
        )
        matches = pattern_regex.findall(text)
        for scene, manner in matches:
            scene = scene.strip()
            manner = manner.strip()
            if scene and manner:
                patterns.append((scene, manner))
        return patterns

    async def _upsert_style_record(
        self,
        scene: str,
        manner: str,
        context: str,
        up_content: str,
        current_ts: float,
    ):
        if not validate_style_pair(scene, manner):
            return
        try:
            from src.common.database.database_model import Expression

            existing = Expression.get_or_none(
                (Expression.situation == scene)
                & (Expression.style == manner)
                & (Expression.chat_id == self.channel_id)
            )
            if existing:
                existing.count += 1
                existing.last_active_time = current_ts
                if context:
                    existing.content_list = context
                existing.save()
            else:
                Expression.create(
                    situation=scene,
                    style=manner,
                    content_list=context,
                    count=1,
                    last_active_time=current_ts,
                    chat_id=self.channel_id,
                    create_date=current_ts,
                    checked=False,
                    rejected=False,
                )
            key = f"{scene}|{manner}"
            if key in self._style_cache:
                self._style_cache[key].count += 1
                self._style_cache[key].last_active_at = current_ts
            else:
                if len(self._style_cache) >= self._max_style_cache:
                    oldest_key = min(
                        self._style_cache,
                        key=lambda k: self._style_cache[k].last_active_at,
                    )
                    del self._style_cache[oldest_key]
                self._style_cache[key] = StyleRecord(
                    scene=scene,
                    manner=manner,
                    context=context,
                    up_content=up_content,
                    count=1,
                    last_active_at=current_ts,
                    created_at=current_ts,
                )
        except Exception as e:
            logger.error(f"更新风格记录失败: {e}")

    async def fetch_recent_styles(
        self, limit: int = 50
    ) -> List[Tuple[str, str]]:
        try:
            from src.common.database.database_model import Expression

            records = (
                Expression.select()
                .where(Expression.chat_id == self.channel_id)
                .where(Expression.rejected.is_(False))
                .order_by(Expression.last_active_time.desc())
                .limit(limit)
            )
            return [(r.situation, r.style) for r in records]
        except Exception as e:
            logger.error(f"获取风格记录失败: {e}")
            return []

    async def remove_outdated_styles(self, days: int = 30):
        cutoff_ts = time.time() - (days * 86400)
        try:
            from src.common.database.database_model import Expression

            query = Expression.delete().where(
                (Expression.chat_id == self.channel_id)
                & (Expression.last_active_time < cutoff_ts)
            )
            deleted = query.execute()
            if deleted > 0:
                logger.info(f"清理了 {deleted} 条过期风格记录")
        except Exception as e:
            logger.error(f"清理过期风格失败: {e}")

    def get_absorb_stats(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "last_absorb_ts": self.last_absorb_ts,
            "absorb_enabled": self.absorb_enabled,
            "absorb_intensity": self.absorb_intensity,
            "cache_size": len(self._style_cache),
        }


class StylePatternMatcher:
    def __init__(self):
        self._pattern_cache: Dict[str, List[Tuple[str, str]]] = {}

    def extract_patterns(self, text: str) -> List[Tuple[str, str]]:
        patterns = []
        lines = text.split("\n")
        for line in lines:
            match = re.search(
                r'当["\"](.+?)["\"]时[，,].*?使用["\"](.+?)["\"]', line
            )
            if match:
                scene = match.group(1).strip()
                manner = match.group(2).strip()
                if validate_style_pair(scene, manner):
                    patterns.append((scene, manner))
        return patterns

    def score_pattern_match(
        self, pattern: Tuple[str, str], context: str
    ) -> float:
        scene, manner = pattern
        scene_score = compute_similarity(scene, context)
        return scene_score

    def find_best_matches(
        self, patterns: List[Tuple[str, str]], context: str, top_k: int = 3
    ) -> List[Tuple[Tuple[str, str], float]]:
        scored = [(p, self.score_pattern_match(p, context)) for p in patterns]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class AbsorbMetrics:
    def __init__(self):
        self._total_absorbed = 0
        self._total_rejected = 0
        self._absorb_times: List[float] = []

    def record_absorb(self, count: int, duration: float):
        self._total_absorbed += count
        self._absorb_times.append(duration)

    def record_reject(self, count: int):
        self._total_rejected += count

    def get_stats(self) -> dict:
        avg_time = (
            sum(self._absorb_times) / max(1, len(self._absorb_times))
            if self._absorb_times
            else 0
        )
        return {
            "total_absorbed": self._total_absorbed,
            "total_rejected": self._total_rejected,
            "avg_absorb_time": avg_time,
            "absorb_count": len(self._absorb_times),
        }


class StyleValidator:
    def __init__(self):
        self._blacklist: set = set()
        self._whitelist: set = set()

    def add_to_blacklist(self, pattern: str):
        self._blacklist.add(pattern.lower())

    def add_to_whitelist(self, pattern: str):
        self._whitelist.add(pattern.lower())

    def is_valid(self, scene: str, manner: str) -> bool:
        combined = f"{scene}{manner}".lower()
        for blocked in self._blacklist:
            if blocked in combined:
                return False
        if self._whitelist:
            for allowed in self._whitelist:
                if allowed in combined:
                    return True
            return False
        return True


class StyleEmbedding:
    _MAX_CACHE_SIZE = 3000

    def __init__(self):
        self._cache: Dict[str, List[float]] = {}

    async def compute_embedding(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]
        embedding = [0.0] * 128
        words = re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", text.lower())
        for i, word in enumerate(words[:128]):
            h = hash(word) % 128
            embedding[h] += 1.0 / (i + 1)
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        if len(self._cache) >= self._MAX_CACHE_SIZE:
            evict_count = self._MAX_CACHE_SIZE // 4
            for _k in list(self._cache.keys())[:evict_count]:
                del self._cache[_k]
        self._cache[text] = embedding
        return embedding

    async def find_similar(
        self,
        query_embedding: List[float],
        candidates: List[Tuple[str, List[float]]],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        results = []
        for text, emb in candidates:
            score = self._cosine_similarity(query_embedding, emb)
            results.append((text, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class StyleDatabase:
    def __init__(self):
        self._connection = None

    async def get_all_styles(
        self, channel_id: str, limit: int = 100
    ) -> List[Dict]:
        try:
            from src.common.database.database_model import Expression

            records = (
                Expression.select()
                .where(Expression.chat_id == channel_id)
                .where(Expression.rejected.is_(False))
                .order_by(Expression.count.desc())
                .limit(limit)
            )
            return [
                {
                    "id": r.id,
                    "situation": r.situation,
                    "style": r.style,
                    "content_list": r.content_list,
                    "count": r.count,
                    "last_active_at": r.last_active_time,
                }
                for r in records
            ]
        except Exception as e:
            logger.error(f"获取所有风格失败: {e}")
            return []

    async def delete_style(self, style_id: int) -> bool:
        try:
            from src.common.database.database_model import Expression

            query = Expression.delete().where(Expression.id == style_id)
            deleted = query.execute()
            return deleted > 0
        except Exception as e:
            logger.error(f"删除风格失败: {e}")
            return False

    async def update_style(self, style_id: int, updates: Dict) -> bool:
        try:
            from src.common.database.database_model import Expression

            query = Expression.update(**updates).where(
                Expression.id == style_id
            )
            updated = query.execute()
            return updated > 0
        except Exception as e:
            logger.error(f"更新风格失败: {e}")
            return False

    async def get_style_by_id(self, style_id: int) -> Optional[Dict]:
        try:
            from src.common.database.database_model import Expression

            record = Expression.get_or_none(Expression.id == style_id)
            if record:
                return {
                    "id": record.id,
                    "situation": record.situation,
                    "style": record.style,
                    "content_list": record.content_list,
                    "count": record.count,
                }
            return None
        except Exception as e:
            logger.error(f"获取风格详情失败: {e}")
            return None

    async def search_styles(
        self, channel_id: str, keyword: str, limit: int = 20
    ) -> List[Dict]:
        try:
            from src.common.database.database_model import Expression

            records = (
                Expression.select()
                .where(Expression.chat_id == channel_id)
                .where(Expression.rejected.is_(False))
                .where(
                    (Expression.situation.contains(keyword))
                    | (Expression.style.contains(keyword))
                )
                .order_by(Expression.count.desc())
                .limit(limit)
            )
            return [
                {
                    "id": r.id,
                    "situation": r.situation,
                    "style": r.style,
                    "count": r.count,
                }
                for r in records
            ]
        except Exception as e:
            logger.error(f"搜索风格失败: {e}")
            return []


class StyleExporter:
    def __init__(self):
        self._export_format = "json"

    async def export_to_json(self, channel_id: str) -> str:
        db = get_style_database()
        styles = await db.get_all_styles(channel_id, limit=500)
        return json.dumps(styles, ensure_ascii=False, indent=2)

    async def export_to_csv(self, channel_id: str) -> str:
        db = get_style_database()
        styles = await db.get_all_styles(channel_id, limit=500)
        lines = ["situation,style,count"]
        for s in styles:
            line = f'"{s["situation"]}","{s["style"]}",{s["count"]}'
            lines.append(line)
        return "\n".join(lines)

    async def import_from_json(self, channel_id: str, json_str: str) -> int:
        try:
            data = json.loads(json_str)
            imported = 0
            absorber = get_style_absorber(channel_id)
            for item in data:
                situation = item.get("situation", "")
                style = item.get("style", "")
                if validate_style_pair(situation, style):
                    await absorber._upsert_style_record(
                        scene=situation,
                        manner=style,
                        context=item.get("content_list", ""),
                        up_content="",
                        current_ts=time.time(),
                    )
                    imported += 1
            return imported
        except Exception as e:
            logger.error(f"导入风格失败: {e}")
            return 0


class StyleStatistics:
    def __init__(self):
        self._stats_cache: Dict[str, Dict] = {}

    async def get_channel_stats(self, channel_id: str) -> Dict:
        try:
            from src.common.database.database_model import Expression

            total = (
                Expression.select()
                .where(Expression.chat_id == channel_id)
                .count()
            )
            checked = (
                Expression.select()
                .where(
                    (Expression.chat_id == channel_id)
                    & (Expression.checked.is_(True))
                )
                .count()
            )
            rejected = (
                Expression.select()
                .where(
                    (Expression.chat_id == channel_id)
                    & (Expression.rejected.is_(True))
                )
                .count()
            )
            return {
                "total": total,
                "checked": checked,
                "rejected": rejected,
                "pending": total - checked,
            }
        except Exception as e:
            logger.error(f"获取统计失败: {e}")
            return {"total": 0, "checked": 0, "rejected": 0, "pending": 0}

    async def get_top_styles(
        self, channel_id: str, limit: int = 10
    ) -> List[Dict]:
        db = get_style_database()
        return await db.get_all_styles(channel_id, limit=limit)


_absorbers: Dict[str, StyleAbsorber] = {}
_pattern_matcher: Optional[StylePatternMatcher] = None
_absorb_metrics: Optional[AbsorbMetrics] = None
_style_validator: Optional[StyleValidator] = None
_style_embedding: Optional[StyleEmbedding] = None
_style_database: Optional[StyleDatabase] = None
_style_exporter: Optional[StyleExporter] = None
_style_statistics: Optional[StyleStatistics] = None


def get_style_absorber(channel_id: str) -> StyleAbsorber:
    if channel_id not in _absorbers:
        _absorbers[channel_id] = StyleAbsorber(channel_id)
    return _absorbers[channel_id]


def reset_style_absorber(channel_id: str) -> None:
    if channel_id in _absorbers:
        del _absorbers[channel_id]


async def trigger_all_absorbers():
    for absorber in _absorbers.values():
        await absorber.trigger_absorb_task()


class StyleAbsorbScheduler:
    def __init__(self):
        self._running = False
        self._interval = 300.0

    async def start(self):
        if self._running:
            return
        self._running = True
        logger.info("风格吸收调度器已启动")
        while self._running:
            await asyncio.sleep(self._interval)
            await trigger_all_absorbers()

    def stop(self):
        self._running = False
        logger.info("风格吸收调度器已停止")


_scheduler: Optional[StyleAbsorbScheduler] = None


def get_absorb_scheduler() -> StyleAbsorbScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = StyleAbsorbScheduler()
    return _scheduler


def get_pattern_matcher() -> StylePatternMatcher:
    global _pattern_matcher
    if _pattern_matcher is None:
        _pattern_matcher = StylePatternMatcher()
    return _pattern_matcher


def get_absorb_metrics() -> AbsorbMetrics:
    global _absorb_metrics
    if _absorb_metrics is None:
        _absorb_metrics = AbsorbMetrics()
    return _absorb_metrics


def get_style_validator() -> StyleValidator:
    global _style_validator
    if _style_validator is None:
        _style_validator = StyleValidator()
    return _style_validator


def get_style_embedding() -> StyleEmbedding:
    global _style_embedding
    if _style_embedding is None:
        _style_embedding = StyleEmbedding()
    return _style_embedding


def get_style_database() -> StyleDatabase:
    global _style_database
    if _style_database is None:
        _style_database = StyleDatabase()
    return _style_database


def get_style_exporter() -> StyleExporter:
    global _style_exporter
    if _style_exporter is None:
        _style_exporter = StyleExporter()
    return _style_exporter


def get_style_statistics() -> StyleStatistics:
    global _style_statistics
    if _style_statistics is None:
        _style_statistics = StyleStatistics()
    return _style_statistics

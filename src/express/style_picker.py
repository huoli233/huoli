import time
import random
import re
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from src.common.logger import get_logger

logger = get_logger("风格选择")


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


@dataclass
class StyleCandidate:
    scene: str
    manner: str
    score: float = 0.0
    count: int = 1
    last_active_at: float = 0.0


class StylePicker:
    def __init__(self, channel_id: str) -> None:
        self.channel_id = channel_id
        self._style_cache: List[Tuple[str, str]] = []
        self._cache_ts: float = 0.0
        self._cache_ttl: float = 300.0
        self._min_similarity: float = 0.3
        self._max_results: int = 5
        self._temperature: float = 1.0

    async def refresh_cache(self):
        try:
            from src.common.database.database_model import Expression

            records = (
                Expression.select()
                .where(Expression.chat_id == self.channel_id)
                .where(Expression.rejected is False)
                .order_by(Expression.count.desc())
                .limit(100)
            )
            self._style_cache = [(r.situation, r.style) for r in records]
            self._cache_ts = time.time()
            logger.debug(f"风格缓存刷新，共 {len(self._style_cache)} 条")
        except Exception as e:
            logger.error(f"刷新风格缓存失败: {e}")

    def is_cache_valid(self) -> bool:
        return time.time() - self._cache_ts < self._cache_ttl

    async def pick_styles(
        self, context: str, top_k: int = 3
    ) -> List[Tuple[str, str, float]]:
        if not self.is_cache_valid():
            await self.refresh_cache()
        if not self._style_cache:
            return []
        scored = []
        for scene, manner in self._style_cache:
            score = compute_similarity(scene, context)
            if score >= self._min_similarity:
                scored.append((scene, manner, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    async def pick_random_style(self) -> Optional[Tuple[str, str]]:
        if not self.is_cache_valid():
            await self.refresh_cache()
        if not self._style_cache:
            return None
        return random.choice(self._style_cache)

    async def pick_by_scene(self, scene_keyword: str) -> List[Tuple[str, str]]:
        if not self.is_cache_valid():
            await self.refresh_cache()
        matches = []
        for scene, manner in self._style_cache:
            if scene_keyword.lower() in scene.lower():
                matches.append((scene, manner))
        return matches

    async def pick_weighted_random(
        self, context: str, temperature: float = None
    ) -> Optional[Tuple[str, str]]:
        styles = await self.pick_styles(context, top_k=10)
        if not styles:
            return None
        temp = temperature if temperature is not None else self._temperature
        if temp <= 0:
            return (styles[0][0], styles[0][1])
        import math

        weights = [math.exp(s[2] / temp) for s in styles]
        total = sum(weights)
        if total == 0:
            return random.choice(styles)[:2]
        probs = [w / total for w in weights]
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return (styles[i][0], styles[i][1])
        return (styles[-1][0], styles[-1][1])

    def set_min_similarity(self, value: float):
        self._min_similarity = max(0.0, min(1.0, value))

    def set_cache_ttl(self, seconds: float):
        self._cache_ttl = max(60.0, seconds)

    def set_temperature(self, value: float):
        self._temperature = max(0.01, value)

    def get_cache_stats(self) -> Dict:
        return {
            "cache_size": len(self._style_cache),
            "cache_age": time.time() - self._cache_ts,
            "cache_valid": self.is_cache_valid(),
            "min_similarity": self._min_similarity,
            "temperature": self._temperature,
        }


class GlobalStylePicker:
    def __init__(self) -> None:
        self._global_cache: List[Tuple[str, str]] = []
        self._cache_ts: float = 0.0
        self._cache_ttl: float = 600.0

    async def refresh_global_cache(self):
        try:
            from src.common.database.database_model import Expression

            records = (
                Expression.select()
                .where(Expression.rejected is False)
                .order_by(Expression.count.desc())
                .limit(200)
            )
            self._global_cache = [(r.situation, r.style) for r in records]
            self._cache_ts = time.time()
            logger.debug(f"全局风格缓存刷新，共 {len(self._global_cache)} 条")
        except Exception as e:
            logger.error(f"刷新全局风格缓存失败: {e}")

    def is_cache_valid(self) -> bool:
        return time.time() - self._cache_ts < self._cache_ttl

    async def pick_global_styles(
        self, context: str, top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        if not self.is_cache_valid():
            await self.refresh_global_cache()
        if not self._global_cache:
            return []
        scored = []
        for scene, manner in self._global_cache:
            score = compute_similarity(scene, context)
            if score >= 0.2:
                scored.append((scene, manner, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]


class StyleRanker:
    def __init__(self):
        self._weights = {
            "similarity": 0.5,
            "frequency": 0.3,
            "recency": 0.2,
        }

    def rank_styles(self, candidates: List[Dict], context: str) -> List[Dict]:
        now = time.time()
        for c in candidates:
            sim_score = compute_similarity(c.get("scene", ""), context)
            freq_score = min(1.0, c.get("count", 1) / 100.0)
            last_ts = c.get("last_active_at", now)
            recency_score = max(0.0, 1.0 - (now - last_ts) / (30 * 86400))
            c["rank_score"] = (
                self._weights["similarity"] * sim_score
                + self._weights["frequency"] * freq_score
                + self._weights["recency"] * recency_score
            )
        candidates.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
        return candidates

    def set_weight(self, key: str, value: float):
        if key in self._weights:
            self._weights[key] = max(0.0, min(1.0, value))

    def get_weights(self) -> Dict:
        return self._weights.copy()


class StyleFilter:
    def __init__(self):
        self._min_count = 1
        self._max_age_days = 30
        self._excluded_scenes: set = set()

    def filter_styles(self, styles: List[Dict]) -> List[Dict]:
        now = time.time()
        cutoff_ts = now - (self._max_age_days * 86400)
        filtered = []
        for s in styles:
            if s.get("count", 0) < self._min_count:
                continue
            if s.get("last_active_at", now) < cutoff_ts:
                continue
            if s.get("scene", "").lower() in self._excluded_scenes:
                continue
            filtered.append(s)
        return filtered

    def set_min_count(self, value: int):
        self._min_count = max(1, value)

    def set_max_age_days(self, value: int):
        self._max_age_days = max(1, value)

    def exclude_scene(self, scene: str):
        self._excluded_scenes.add(scene.lower())

    def clear_exclusions(self):
        self._excluded_scenes.clear()


class StyleSampler:
    def __init__(self):
        self._temperature = 1.0

    def sample_with_temperature(
        self, styles: List[Tuple[str, str, float]], temperature: float = None
    ) -> Optional[Tuple[str, str]]:
        if not styles:
            return None
        temp = temperature if temperature is not None else self._temperature
        if temp <= 0:
            return (styles[0][0], styles[0][1])
        import math

        weights = [math.exp(s[2] / temp) for s in styles]
        total = sum(weights)
        if total == 0:
            return random.choice(styles)[:2]
        probs = [w / total for w in weights]
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return (styles[i][0], styles[i][1])
        return (styles[-1][0], styles[-1][1])

    def sample_top_p(
        self, styles: List[Tuple[str, str, float]], p: float = 0.9
    ) -> Optional[Tuple[str, str]]:
        if not styles:
            return None
        import math

        total = sum(math.exp(min(50.0, s[2])) for s in styles)
        probs = [math.exp(min(50.0, s[2])) / total for s in styles]
        cumulative = 0.0
        candidates = []
        for i, prob in enumerate(probs):
            cumulative += prob
            candidates.append(styles[i])
            if cumulative >= p:
                break
        if not candidates:
            return None
        return random.choice(candidates)[:2]

    def set_temperature(self, value: float):
        self._temperature = max(0.01, value)


class StyleContextMatcher:
    def __init__(self):
        self._context_keywords: Dict[str, List[str]] = {}

    def extract_context_keywords(self, text: str) -> List[str]:
        words = re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", text)
        keywords = []
        for w in words:
            if len(w) >= 2:
                keywords.append(w.lower())
        return list(set(keywords))

    def match_style_to_context(
        self, styles: List[Tuple[str, str]], context: str
    ) -> List[Tuple[str, str, float]]:
        context_kw = set(self.extract_context_keywords(context))
        results = []
        for scene, manner in styles:
            scene_kw = set(self.extract_context_keywords(scene))
            common = len(context_kw & scene_kw)
            total = len(context_kw | scene_kw)
            score = common / total if total > 0 else 0
            results.append((scene, manner, score))
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def find_best_match(
        self, styles: List[Tuple[str, str]], context: str
    ) -> Optional[Tuple[str, str]]:
        matches = self.match_style_to_context(styles, context)
        if matches and matches[0][2] > 0.2:
            return (matches[0][0], matches[0][1])
        return None

    def register_context_keywords(
        self, context_name: str, keywords: List[str]
    ):
        self._context_keywords[context_name] = [k.lower() for k in keywords]

    def get_registered_keywords(self, context_name: str) -> List[str]:
        return self._context_keywords.get(context_name, [])


class StyleUsageTracker:
    def __init__(self):
        self._usage_log: Dict[str, List[float]] = {}
        self._max_entries = 100

    def log_usage(self, scene: str, manner: str):
        key = f"{scene}|{manner}"
        if key not in self._usage_log:
            self._usage_log[key] = []
        self._usage_log[key].append(time.time())
        if len(self._usage_log[key]) > self._max_entries:
            self._usage_log[key] = self._usage_log[key][-self._max_entries:]

    def get_usage_count(self, scene: str, manner: str, hours: int = 24) -> int:
        key = f"{scene}|{manner}"
        if key not in self._usage_log:
            return 0
        cutoff = time.time() - (hours * 3600)
        return sum(1 for ts in self._usage_log[key] if ts >= cutoff)

    def get_most_used(self, top_k: int = 10) -> List[Dict]:
        usage_counts = {}
        for key, timestamps in self._usage_log.items():
            usage_counts[key] = len(timestamps)
        sorted_usage = sorted(
            usage_counts.items(), key=lambda x: x[1], reverse=True
        )
        results = []
        for key, count in sorted_usage[:top_k]:
            parts = key.split("|", 1)
            if len(parts) == 2:
                results.append(
                    {"scene": parts[0], "manner": parts[1], "count": count}
                )
        return results

    def clear_old_entries(self, hours: int = 168):
        cutoff = time.time() - (hours * 3600)
        for key in self._usage_log:
            self._usage_log[key] = [
                ts for ts in self._usage_log[key] if ts >= cutoff
            ]


class StyleCooldownManager:
    def __init__(self):
        self._cooldowns: Dict[str, float] = {}
        self._default_cooldown = 60.0

    def set_cooldown(self, scene: str, manner: str, seconds: float = None):
        key = f"{scene}|{manner}"
        cooldown = seconds if seconds is not None else self._default_cooldown
        self._cooldowns[key] = time.time() + cooldown

    def is_on_cooldown(self, scene: str, manner: str) -> bool:
        key = f"{scene}|{manner}"
        if key not in self._cooldowns:
            return False
        return time.time() < self._cooldowns[key]

    def get_remaining_cooldown(self, scene: str, manner: str) -> float:
        key = f"{scene}|{manner}"
        if key not in self._cooldowns:
            return 0.0
        remaining = self._cooldowns[key] - time.time()
        return max(0.0, remaining)

    def clear_cooldown(self, scene: str, manner: str):
        key = f"{scene}|{manner}"
        if key in self._cooldowns:
            del self._cooldowns[key]

    def clear_all_cooldowns(self):
        self._cooldowns.clear()

    def set_default_cooldown(self, seconds: float):
        self._default_cooldown = max(0.0, seconds)


_pickers: Dict[str, StylePicker] = {}
_global_picker: Optional[GlobalStylePicker] = None
_style_ranker: Optional[StyleRanker] = None
_style_filter: Optional[StyleFilter] = None
_style_sampler: Optional[StyleSampler] = None
_style_context_matcher: Optional[StyleContextMatcher] = None
_style_usage_tracker: Optional[StyleUsageTracker] = None
_style_cooldown_manager: Optional[StyleCooldownManager] = None


def get_style_picker(channel_id: str) -> StylePicker:
    if channel_id not in _pickers:
        _pickers[channel_id] = StylePicker(channel_id)
    return _pickers[channel_id]


def reset_style_picker(channel_id: str) -> None:
    if channel_id in _pickers:
        del _pickers[channel_id]


def get_global_style_picker() -> GlobalStylePicker:
    global _global_picker
    if _global_picker is None:
        _global_picker = GlobalStylePicker()
    return _global_picker


def get_style_ranker() -> StyleRanker:
    global _style_ranker
    if _style_ranker is None:
        _style_ranker = StyleRanker()
    return _style_ranker


def get_style_filter() -> StyleFilter:
    global _style_filter
    if _style_filter is None:
        _style_filter = StyleFilter()
    return _style_filter


def get_style_sampler() -> StyleSampler:
    global _style_sampler
    if _style_sampler is None:
        _style_sampler = StyleSampler()
    return _style_sampler


def get_style_context_matcher() -> StyleContextMatcher:
    global _style_context_matcher
    if _style_context_matcher is None:
        _style_context_matcher = StyleContextMatcher()
    return _style_context_matcher


def get_style_usage_tracker() -> StyleUsageTracker:
    global _style_usage_tracker
    if _style_usage_tracker is None:
        _style_usage_tracker = StyleUsageTracker()
    return _style_usage_tracker


def get_style_cooldown_manager() -> StyleCooldownManager:
    global _style_cooldown_manager
    if _style_cooldown_manager is None:
        _style_cooldown_manager = StyleCooldownManager()
    return _style_cooldown_manager


async def pick_best_style(
    channel_id: str, context: str, use_cooldown: bool = True
) -> Optional[Tuple[str, str]]:
    picker = get_style_picker(channel_id)
    cooldown_mgr = get_style_cooldown_manager()
    styles = await picker.pick_styles(context, top_k=10)
    if not styles:
        return None
    for scene, manner, _score in styles:
        if use_cooldown and cooldown_mgr.is_on_cooldown(scene, manner):
            continue
        if use_cooldown:
            cooldown_mgr.set_cooldown(scene, manner)
        usage_tracker = get_style_usage_tracker()
        usage_tracker.log_usage(scene, manner)
        return (scene, manner)
    return (styles[0][0], styles[0][1])

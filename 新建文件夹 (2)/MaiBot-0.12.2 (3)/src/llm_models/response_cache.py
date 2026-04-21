import time
import hashlib
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("llm_cache")


@dataclass
class CacheEntry:
    response: str
    reasoning: str
    model_name: str
    created_at: float
    hit_count: int = 0


class LLMResponseCache:
    _instance: Optional["LLMResponseCache"] = None

    def __new__(cls) -> "LLMResponseCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = 500
        self._default_ttl = 300.0
        self._task_ttl: Dict[str, float] = {
            "sticker_context": 60.0,
            "sticker_timing": 60.0,
            "correction_verify": 120.0,
            "self_awareness": 180.0,
            "amnesia_check": 180.0,
            "intent_classify": 120.0,
            "persona_switch": 60.0,
            "recall_review": 60.0,
            "persona_traits": 600.0,
            "persona_name": 600.0,
            "persona_speech": 600.0,
            "knowledge_extract": 300.0,
        }
        self._no_cache_tasks = {
            "replyer", "chat_history_summarizer", "recall_regenerate",
            "recall_correction", "recall_regret", "recall_shy",
            "typo_maker", "expression.learner", "expression.summary",
        }
        self._cleanup_interval = 120.0
        self._last_cleanup = time.time()
        self._total_hits = 0
        self._total_misses = 0
        self._initialized = True

    def _make_key(self, prompt: str, request_type: str, max_tokens: int = 0) -> str:
        content = f"{request_type}:{max_tokens}:{prompt}"
        return hashlib.md5(content.encode('utf-8', errors='replace')).hexdigest()

    def get(self, prompt: str, request_type: str, max_tokens: int = 0) -> Optional[Tuple[str, str, str]]:
        if request_type in self._no_cache_tasks:
            return None
        self._maybe_cleanup()
        key = self._make_key(prompt, request_type, max_tokens)
        entry = self._cache.get(key)
        if not entry:
            self._total_misses += 1
            return None
        ttl = self._task_ttl.get(request_type, self._default_ttl)
        if time.time() - entry.created_at > ttl:
            del self._cache[key]
            self._total_misses += 1
            return None
        entry.hit_count += 1
        self._total_hits += 1
        logger.debug(f"[缓存命中] {request_type} (累计命中{self._total_hits}次)")
        return entry.response, entry.reasoning, entry.model_name

    def put(self, prompt: str, request_type: str, response: str, reasoning: str = "",
            model_name: str = "", max_tokens: int = 0):
        if request_type in self._no_cache_tasks:
            return
        if not response:
            return
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        key = self._make_key(prompt, request_type, max_tokens)
        self._cache[key] = CacheEntry(
            response=response,
            reasoning=reasoning,
            model_name=model_name,
            created_at=time.time(),
        )

    def _evict_oldest(self):
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]

    def _maybe_cleanup(self):
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        expired_keys = []
        for key, entry in self._cache.items():
            age = now - entry.created_at
            if age > self._default_ttl * 3:
                expired_keys.append(key)
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.debug(f"[缓存清理] 移除{len(expired_keys)}条过期记录，剩余{len(self._cache)}条")

    def invalidate(self, request_type: str = ""):
        if not request_type:
            count = len(self._cache)
            self._cache.clear()
            logger.debug(f"[缓存] 全部清除({count}条)")
            return
        keys_to_remove = [k for k, v in self._cache.items() if request_type in k]
        for k in keys_to_remove:
            del self._cache[k]

    def get_stats(self) -> Dict[str, Any]:
        total = self._total_hits + self._total_misses
        hit_rate = (self._total_hits / total * 100) if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }


def get_llm_cache() -> LLMResponseCache:
    return LLMResponseCache()

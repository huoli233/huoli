import re
import time
import json
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, Counter
from src.common.logger import get_logger

logger = get_logger("deep_optimizer")

_STOPWORDS = frozenset({
    '的', '了', '是', '在', '有', '和', '与', '或', '但', '而', '也', '就',
    '都', '还', '又', '不', '吗', '吧', '呢', '啊', '哦', '哈', '嗯',
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
})

_GARBAGE_PATTERNS = [
    re.compile(r'^[\s\.\,\!\?\;\:\-\_\=\+\*\&\^\%\$\#\@\~\`]+$'),
    re.compile(r'^(\.{3,}|。{3,}|？{3,}|！{3,})$'),
    re.compile(r'^https?://\S+$'),
    re.compile(r'^\[图片\]$'),
    re.compile(r'^\[语音\]$'),
    re.compile(r'^\[视频\]$'),
]


class MemoryDeepOptimizer:
    def __init__(self):
        self._last_run_time: float = 0.0
        self._run_interval: float = 21600.0
        self.content_signatures: Dict[str, List[str]] = defaultdict(list)
        self._compression_stats: Dict[str, int] = {
            "total_processed": 0,
            "duplicates_removed": 0,
            "content_compressed": 0,
            "garbage_cleaned": 0,
            "space_saved_bytes": 0,
            "token_estimate_saved": 0,
        }

    def _get_memory_model(self):
        try:
            from src.common.database.database_model import MemoryEntry
            return MemoryEntry
        except Exception:
            return None

    def should_run(self) -> bool:
        return (time.time() - self._last_run_time) >= self._run_interval

    async def run_optimization(self, stream_id: Optional[str] = None,
                               force_cleanup: bool = False) -> Dict[str, Any]:
        self._compression_stats = {
            "total_processed": 0,
            "duplicates_removed": 0,
            "content_compressed": 0,
            "garbage_cleaned": 0,
            "space_saved_bytes": 0,
            "token_estimate_saved": 0,
        }
        self.content_signatures.clear()
        start_time = time.time()
        self._last_run_time = start_time
        logger.info("[深度优化] 开始执行...")
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            logger.warning("[深度优化] MemoryEntry模型不可用，跳过")
            return self._compression_stats
        try:
            self._analyze_content_signatures(MemoryEntry, stream_id)
            self._remove_garbage(MemoryEntry, stream_id, force_cleanup)
            self._deduplicate_memories(MemoryEntry, stream_id)
            self._compress_long_content(MemoryEntry, stream_id)
            self._optimize_database_structure()
            duration = round(time.time() - start_time, 2)
            logger.info(
                f"[深度优化] 完成({duration}秒) | 去重{self._compression_stats['duplicates_removed']}条 | "
                f"压缩{self._compression_stats['content_compressed']}条 | "
                f"清理{self._compression_stats['garbage_cleaned']}条 | "
                f"节省{format_size(self._compression_stats['space_saved_bytes'])}"
            )
        except Exception as e:
            logger.error(f"[深度优化] 执行失败: {e}")
        return self._compression_stats

    def _analyze_content_signatures(self, MemoryEntry, stream_id: Optional[str] = None):
        query = MemoryEntry.select()
        if stream_id:
            query = query.where(MemoryEntry.stream_id == stream_id)
        memories = list(query)
        self._compression_stats["total_processed"] = len(memories)
        for memory in memories:
            signatures = self._generate_content_signatures(memory.content or "")
            for sig in signatures:
                self.content_signatures[sig].append(memory.memory_id)

    def _generate_content_signatures(self, content: str) -> List[str]:
        signatures = []
        normalized = re.sub(r'\s+', ' ', content.strip().lower())
        signatures.append(hashlib.md5(normalized.encode()).hexdigest())
        keywords = self._extract_keywords(content)
        if keywords:
            keyword_sig = ''.join(sorted(keywords))
            signatures.append(hashlib.md5(keyword_sig.encode()).hexdigest())
        try:
            stripped = content.strip()
            if stripped.startswith('{') and stripped.endswith('}'):
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    key_structure = sorted(parsed.keys())
                    struct_sig = '|'.join(key_structure)
                    signatures.append(hashlib.md5(struct_sig.encode()).hexdigest())
        except (json.JSONDecodeError, ValueError):
            pass
        return signatures

    def _extract_keywords(self, content: str, min_length: int = 2) -> Set[str]:
        words = re.findall(r'\b\w+\b', content.lower())
        return {w for w in words if len(w) >= min_length and w not in _STOPWORDS}

    def _remove_garbage(self, MemoryEntry, stream_id: Optional[str] = None,
                        force_cleanup: bool = False):
        query = MemoryEntry.select()
        if stream_id:
            query = query.where(MemoryEntry.stream_id == stream_id)
        removed = 0
        for mem in query:
            if self._is_garbage_content(mem, force_cleanup):
                content_size = len((mem.content or "").encode('utf-8'))
                self._compression_stats["space_saved_bytes"] += content_size
                self._compression_stats["token_estimate_saved"] += estimate_token_count(mem.content or "")
                mem.delete_instance()
                removed += 1
        self._compression_stats["garbage_cleaned"] = removed
        if removed:
            logger.info(f"[深度优化] 垃圾清理: 删除{removed}条无效记忆")

    def _is_garbage_content(self, memory, force_cleanup: bool = False) -> bool:
        content = (getattr(memory, 'content', '') or "").strip()
        if len(content) < 3:
            return True
        if re.match(r'^[^\w\u4e00-\u9fff]*$', content):
            return True
        if len(set(content)) <= 3 and len(content) > 10:
            return True
        for pattern in _GARBAGE_PATTERNS:
            if pattern.match(content):
                return True
        access_count = getattr(memory, 'access_count', 0) or 0
        importance = getattr(memory, 'importance', 0.5) or 0.5
        created_at = getattr(memory, 'created_at', time.time()) or time.time()
        if access_count == 0 and importance < 0.3 and (time.time() - created_at) > 30 * 86400:
            return True
        if force_cleanup:
            if access_count <= 1 and importance < 0.2:
                return True
            words = content.split()
            if words and len(words) > 5:
                most_common = Counter(words).most_common(1)[0]
                if most_common[1] > len(words) * 0.5:
                    return True
        return False

    def _deduplicate_memories(self, MemoryEntry, stream_id: Optional[str] = None):
        query = MemoryEntry.select()
        if stream_id:
            query = query.where(MemoryEntry.stream_id == stream_id)
        hash_groups: Dict[str, List] = defaultdict(list)
        for mem in query:
            content_hash = self._content_hash(mem.content or "")
            hash_groups[content_hash].append(mem)
        removed = 0
        for content_hash, group in hash_groups.items():
            if len(group) <= 1:
                continue
            best = self._select_best_memory(group)
            for mem in group:
                if mem.memory_id != best.memory_id:
                    content_size = len((mem.content or "").encode('utf-8'))
                    self._compression_stats["space_saved_bytes"] += content_size
                    self._compression_stats["token_estimate_saved"] += len(mem.content or "") // 4
                    mem.delete_instance()
                    removed += 1
        self._compression_stats["duplicates_removed"] = removed
        if removed:
            logger.info(f"[深度优化] 去重: 删除{removed}条重复记忆")

    def _compress_long_content(self, MemoryEntry, stream_id: Optional[str] = None):
        now = time.time()
        age_threshold = 259200.0
        length_threshold = 300
        query = MemoryEntry.select().where(
            (MemoryEntry.memory_type == "conversation") &
            (MemoryEntry.created_at < (now - age_threshold))
        )
        if stream_id:
            query = query.where(MemoryEntry.stream_id == stream_id)
        compressed = 0
        for mem in query:
            content = mem.content or ""
            if len(content) <= length_threshold:
                continue
            shorter = self._extract_key_sentences(content)
            if shorter and len(shorter) < len(content) * 0.7:
                original_size = len(content.encode('utf-8'))
                compressed_size = len(shorter.encode('utf-8'))
                self._compression_stats["space_saved_bytes"] += (original_size - compressed_size)
                self._compression_stats["token_estimate_saved"] += (len(content) - len(shorter)) // 4
                mem.content = shorter
                mem.memory_type = "compressed"
                mem.save()
                compressed += 1
        self._compression_stats["content_compressed"] = compressed
        if compressed:
            logger.info(f"[深度优化] 压缩: 压缩{compressed}条长记忆")

    def _content_hash(self, content: str) -> str:
        normalized = ' '.join(content.strip().lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _select_best_memory(self, memories: list) -> object:
        def score(mem) -> float:
            s = 0.0
            s += (getattr(mem, 'access_count', 0) or 0) * 100
            s += (getattr(mem, 'importance', 0.5) or 0.5) * 50
            age_days = (time.time() - (getattr(mem, 'created_at', 0) or 0)) / 86400
            s += max(0, 30 - age_days)
            if getattr(mem, 'summary', None):
                s += 10
            if len(getattr(mem, 'content', '') or '') > 50:
                s += 5
            return s
        return max(memories, key=score)

    def _extract_key_sentences(self, content: str) -> str:
        sentences = re.split(r'[。！？.!?\n]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 2:
            return content
        _IMPORTANT_KEYWORDS = ['记住', '重要', '提醒', '注意', '必须', '应该', '不要', '禁止']
        scored = []
        for sentence in sentences:
            words = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', sentence.lower()))
            meaningful = words - _STOPWORDS
            s = 0.0
            if 10 <= len(sentence) <= 50:
                s += 2.0
            elif len(sentence) > 50:
                s += 1.0
            s += min(1.0, len(meaningful) / 5) * 0.6
            s += min(1.0, len(sentence) / 80) * 0.4
            for kw in _IMPORTANT_KEYWORDS:
                if kw in sentence:
                    s += 3.0
                    break
            if re.search(r'\d+', sentence):
                s += 1.0
            if re.search(r'[a-zA-Z]+', sentence):
                s += 0.5
            scored.append((s, sentence))
        scored.sort(key=lambda x: x[0], reverse=True)
        keep_count = max(1, len(scored) // 2)
        kept_sentences = [s for _, s in scored[:keep_count]]
        original_order = []
        for sentence in sentences:
            if sentence in kept_sentences:
                original_order.append(sentence)
                kept_sentences.remove(sentence)
        return "。".join(original_order) + "。" if original_order else content[:100] + "..."

    def get_stats(self) -> Dict[str, int]:
        return self._compression_stats.copy()

    def _optimize_database_structure(self):
        try:
            from src.common.database.database import db
            db.execute_sql("REINDEX")
            db.execute_sql("VACUUM")
            db.execute_sql("ANALYZE")
            logger.debug("[深度优化] 数据库结构优化完成")
        except Exception as e:
            logger.debug(f"[深度优化] 数据库优化跳过: {e}")

    def get_optimization_report(self, stream_id: Optional[str] = None) -> Dict[str, Any]:
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return {"error": "MemoryEntry模型不可用"}
        query = MemoryEntry.select()
        if stream_id:
            query = query.where(MemoryEntry.stream_id == stream_id)
        memories = list(query)
        total_size = sum(len((m.content or "").encode('utf-8')) for m in memories)
        type_stats = Counter(getattr(m, 'memory_type', 'unknown') for m in memories)
        size_distribution = {"tiny": 0, "small": 0, "medium": 0, "large": 0}
        for memory in memories:
            size = len((memory.content or "").encode('utf-8'))
            if size < 100:
                size_distribution["tiny"] += 1
            elif size < 1024:
                size_distribution["small"] += 1
            elif size < 10240:
                size_distribution["medium"] += 1
            else:
                size_distribution["large"] += 1
        return {
            "total_memories": len(memories),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "type_distribution": dict(type_stats),
            "size_distribution": size_distribution,
            "avg_size_bytes": round(total_size / len(memories)) if memories else 0,
            "estimated_total_tokens": sum(estimate_token_count(m.content or "") for m in memories),
            "optimization_potential": {
                "large_memories": len([m for m in memories if len(m.content or "") > 1000]),
                "unaccessed_memories": len([m for m in memories if (getattr(m, 'access_count', 0) or 0) == 0]),
                "low_importance_memories": len([m for m in memories if (getattr(m, 'importance', 0.5) or 0.5) < 0.3]),
            },
        }


_deep_optimizer: Optional[MemoryDeepOptimizer] = None


def get_deep_optimizer() -> MemoryDeepOptimizer:
    global _deep_optimizer
    if _deep_optimizer is None:
        _deep_optimizer = MemoryDeepOptimizer()
    return _deep_optimizer


def estimate_token_count(text: str) -> int:
    chinese_chars = len(re.findall('[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return chinese_chars + (other_chars // 4)


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f}KB"
    return f"{size_bytes/(1024*1024):.1f}MB"

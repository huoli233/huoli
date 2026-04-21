import re
import time
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from src.common.logger import get_logger

logger = get_logger("知识图谱")


def _load_stopwords() -> frozenset:
    try:
        from src.modules.modcore.social_cognition.config_loader import (
            get_stopwords,
        )

        return frozenset(get_stopwords())
    except Exception:
        return frozenset(
            [
                "的",
                "了",
                "是",
                "在",
                "我",
                "有",
                "和",
                "就",
                "不",
                "人",
                "都",
                "一",
                "这",
                "中",
                "到",
                "那",
                "个",
                "你",
                "他",
                "她",
                "它",
                "们",
                "会",
                "对",
                "也",
                "很",
                "要",
                "能",
                "上",
                "下",
                "啊",
                "呢",
                "吗",
                "吧",
                "哦",
            ]
        )


def _load_separators() -> str:
    try:
        from src.modules.modcore.social_cognition.config_loader import (
            get_separators,
        )

        separators = get_separators()
        escaped = [re.escape(s) for s in separators]
        return "|".join(escaped)
    except Exception:
        return r"[，。！？；\n,.!?;]+"


def _load_settings() -> dict:
    try:
        from src.modules.modcore.social_cognition.config_loader import (
            get_kg_settings,
        )

        return get_kg_settings()
    except Exception:
        return {
            "capacity": 500,
            "min_text_length": 8,
            "max_body_length": 500,
            "min_keywords": 1,
            "eviction_ratio": 0.25,
        }


_STOPWORDS = _load_stopwords()
_SEPARATOR_PATTERN = _load_separators()
_SEPARATOR = re.compile(_SEPARATOR_PATTERN)
_SETTINGS = _load_settings()


@dataclass
class KnowledgeEntry:
    entry_id: str
    body: str
    fragments: Set[str]
    source_msg_id: str
    channel_id: str
    born_at: float
    access_count: int = 0


class ChannelKnowledgeGraph:
    """单频道的知识图谱"""

    def __init__(self, stream_id: str, capacity: Optional[int] = None):
        self.stream_id = stream_id
        self._entries: Dict[str, KnowledgeEntry] = {}
        self._fragment_index: Dict[str, Set[str]] = defaultdict(set)
        self._capacity = capacity or _SETTINGS.get("capacity", 500)
        self._min_text_length = _SETTINGS.get("min_text_length", 8)
        self._max_body_length = _SETTINGS.get("max_body_length", 500)
        self._min_keywords = _SETTINGS.get("min_keywords", 1)
        self._eviction_ratio = _SETTINGS.get("eviction_ratio", 0.25)

    def learn_from_message(self, text: str, msg_id: str = "") -> Optional[str]:
        """从消息文本中提取知识并存储，返回条目ID或None"""
        if not text or len(text.strip()) < self._min_text_length:
            return None
        fragments = self._extract_fragments(text)
        if len(fragments) < self._min_keywords:
            return None
        eid = hashlib.md5(
            f"{self.stream_id}:{text[:80]}".encode()
        ).hexdigest()[:12]
        if eid in self._entries:
            self._entries[eid].access_count += 1
            return eid
        self._evict_if_full()
        entry = KnowledgeEntry(
            entry_id=eid,
            body=text.strip()[: self._max_body_length],
            fragments=fragments,
            source_msg_id=msg_id or "",
            channel_id=self.stream_id,
            born_at=time.time(),
        )
        self._entries[eid] = entry
        for fragment in fragments:
            self._fragment_index[fragment].add(eid)
        logger.debug(
            f"[知识图谱] 学习条目 {eid} (片段: {
                ', '.join(
                    sorted(fragments)[
                        :5])})"
        )
        return eid

    def search(self, query: str, limit: int = 5) -> List[KnowledgeEntry]:
        query_fragments = self._extract_fragments(query)
        if not query_fragments:
            return []
        candidate_ids: Dict[str, int] = defaultdict(int)
        for fragment in query_fragments:
            for eid in self._fragment_index.get(fragment, set()):
                candidate_ids[eid] += 1
        ranked = sorted(
            candidate_ids.items(), key=lambda x: x[1], reverse=True
        )
        results = []
        for eid, _hits in ranked[:limit]:
            entry = self._entries.get(eid)
            if entry:
                entry.access_count += 1
                results.append(entry)
        return results

    def _extract_fragments(self, text: str) -> Set[str]:
        segments = _SEPARATOR.split(text)
        fragments: Set[str] = set()
        for seg in segments:
            seg = seg.strip()
            if len(seg) < 2:
                continue
            for i in range(0, len(seg) - 1):
                bigram = seg[i: i + 2]
                if bigram not in _STOPWORDS and not bigram.isspace():
                    fragments.add(bigram)
            if 2 <= len(seg) <= 6 and seg not in _STOPWORDS:
                fragments.add(seg)
        return fragments

    def _evict_if_full(self) -> None:
        if len(self._entries) < self._capacity:
            return
        evict_count = int(self._capacity * self._eviction_ratio)
        oldest = sorted(self._entries.values(), key=lambda e: e.born_at)
        to_remove = oldest[:evict_count]
        for entry in to_remove:
            for fragment in entry.fragments:
                self._fragment_index[fragment].discard(entry.entry_id)
                if not self._fragment_index[fragment]:
                    del self._fragment_index[fragment]
            del self._entries[entry.entry_id]

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def retrieve_context(self, query: str, limit: int = 3) -> str:
        """根据查询文本检索相关知识上下文"""
        results = self.search(query, limit=limit)
        if not results:
            return ""
        lines = []
        for entry in results:
            lines.append(f"- {entry.body}")
        return "[知识库上下文]\n" + "\n".join(lines)

    def get_topic_suggestion(
        self, recent_text: str = "", limit: int = 3
    ) -> str:
        """根据最近对话内容生成话题建议，用于冷场时引导回复方向"""
        if not self._entries:
            return ""
        query = recent_text or ""
        if query:
            results = self.search(query, limit=limit * 2)
        else:
            results = sorted(
                self._entries.values(),
                key=lambda e: e.access_count,
                reverse=True,
            )[:limit]
        if not results:
            return ""
        topics = []
        seen_bodies = set()
        for entry in results:
            if entry.body not in seen_bodies:
                topics.append(entry.body[:80])
                seen_bodies.add(entry.body)
                if len(topics) >= limit:
                    break
        if not topics:
            return ""
        return f"[话题建议]\n如果当前对话冷场或无话可说，可以考虑围绕以下话题展开：\n" + "\n".join(
            f"  - {t}" for t in topics
        )


_graph_pool: Dict[str, ChannelKnowledgeGraph] = {}


def get_knowledge_graph_manager(stream_id: str) -> ChannelKnowledgeGraph:
    if stream_id not in _graph_pool:
        _graph_pool[stream_id] = ChannelKnowledgeGraph(stream_id)
    return _graph_pool[stream_id]

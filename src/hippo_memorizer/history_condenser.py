import asyncio
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from src.common.logger import get_logger
from src.hippo_memorizer.config_loader import (
    get_batch_size,
    get_batch_interval,
    get_min_condense_interval,
    get_max_summaries,
    get_condense_prompt,
)

logger = get_logger("历史压缩")


@dataclass
class PacketBatch:
    packets: List[Dict]
    start_ts: float
    end_ts: float


@dataclass
class TopicCacheEntry:
    topic: str
    packet_indices: List[int]
    original_text: str
    participants: Set[str]
    created_at: float
    updated_at: float
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)


class HistoryCondenser:
    def __init__(self, channel_id: str, llm_bridge=None, db_adapter=None):
        self.channel_id = channel_id
        self.channel_name = channel_id
        self._llm_bridge = llm_bridge
        self._db_adapter = db_adapter
        self._batch_size = get_batch_size()
        self._batch_interval = get_batch_interval()
        self._current_batch: Optional[PacketBatch] = None
        self._topic_cache: Dict[str, TopicCacheEntry] = {}
        self._condense_lock: asyncio.Lock | None = None
        self._last_condense_ts = 0.0
        self._min_condense_interval = get_min_condense_interval()
        self._max_summaries = get_max_summaries()

    def set_llm_bridge(self, bridge):
        self._llm_bridge = bridge

    def set_db_adapter(self, adapter):
        self._db_adapter = adapter

    def add_packet(self, packet: Dict):
        now = time.time()
        if self._current_batch is None:
            self._current_batch = PacketBatch(
                packets=[], start_ts=now, end_ts=now
            )
        self._current_batch.packets.append(packet)
        self._current_batch.end_ts = now

    def should_trigger_condense(self) -> bool:
        if self._current_batch is None:
            return False
        if len(self._current_batch.packets) < self._batch_size:
            return False
        elapsed = time.time() - self._last_condense_ts
        if elapsed < self._min_condense_interval:
            return False
        return True

    async def trigger_condense_task(self):
        if self._condense_lock is None:
            self._condense_lock = asyncio.Lock()
        async with self._condense_lock:
            if not self.should_trigger_condense():
                return
            self._last_condense_ts = time.time()
            batch = self._current_batch
            self._current_batch = None
            if batch and batch.packets:
                await self._condense_batch(batch)

    async def _condense_batch(self, batch: PacketBatch):
        try:
            topics = await self._extract_topics(batch)
            for _topic_name, entry in topics.items():
                await self._summarize_topic(entry)
                await self._persist_topic(entry)
            logger.info(
                f"频道 {self.channel_name} 压缩完成，识别到 {len(topics)} 个话题"
            )
        except Exception as e:
            logger.error(f"压缩批次失败: {e}")

    async def _extract_topics(
        self, batch: PacketBatch
    ) -> Dict[str, TopicCacheEntry]:
        topics = {}
        topic_name = f"话题_{int(batch.start_ts)}"
        participants = set()
        texts = []
        for pkt in batch.packets:
            sender = pkt.get("sender_name", "未知")
            content = pkt.get("plain_content", "")
            participants.add(sender)
            texts.append(f"{sender}: {content}")
        entry = TopicCacheEntry(
            topic=topic_name,
            packet_indices=list(range(len(batch.packets))),
            original_text="\n".join(texts),
            participants=participants,
            created_at=time.time(),
            updated_at=time.time(),
        )
        if self._llm_bridge:
            try:
                extracted = await self._llm_extract_topics(entry.original_text)
                if extracted:
                    topics.update(extracted)
            except Exception as e:
                logger.warning(f"LLM 话题提取失败: {e}")
        if not topics:
            topics[topic_name] = entry
        return topics

    async def _llm_extract_topics(
        self, text: str
    ) -> Dict[str, TopicCacheEntry]:
        if not self._llm_bridge:
            return {}
        template = get_condense_prompt("topic_extract")
        if not template:
            return {}
        prompt = template.format(conversation_text=text[:2000])
        try:
            response = await self._llm_bridge.chat(
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.get("content", "{}")
            import json

            data = json.loads(content)
            topics = {}
            for topic_name in data.get("topics", []):
                topics[topic_name] = TopicCacheEntry(
                    topic=topic_name,
                    packet_indices=[],
                    original_text=text,
                    participants=set(),
                    created_at=time.time(),
                    updated_at=time.time(),
                )
            return topics
        except Exception as e:
            logger.warning(f"LLM 话题提取解析失败: {e}")
            return {}

    async def _summarize_topic(self, entry: TopicCacheEntry):
        if not self._llm_bridge:
            entry.summary = entry.original_text[:200]
            return
        template = get_condense_prompt("summarize")
        if not template:
            entry.summary = entry.original_text[:200]
            return
        prompt = template.format(
            topic_name=entry.topic,
            content_text=entry.original_text[:2000],
        )
        try:
            response = await self._llm_bridge.chat(
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.get("content", "{}")
            import json

            data = json.loads(content)
            entry.summary = data.get("summary", "")
            entry.keywords = data.get("keywords", [])
            entry.key_points = data.get("key_points", [])
        except Exception as e:
            logger.warning(f"LLM 摘要生成失败: {e}")
            entry.summary = entry.original_text[:200]

    async def _persist_topic(self, entry: TopicCacheEntry):
        if not self._db_adapter:
            return
        try:
            await self._db_adapter.save_topic_summary(
                channel_id=self.channel_id,
                topic=entry.topic,
                summary=entry.summary or "",
                keywords=entry.keywords,
                key_points=entry.key_points,
                participants=list(entry.participants),
                start_time=entry.created_at,
                end_time=entry.updated_at,
                original_text=entry.original_text,
            )
        except Exception as e:
            logger.error(f"持久化话题失败: {e}")

    def get_current_batch_size(self) -> int:
        if self._current_batch is None:
            return 0
        return len(self._current_batch.packets)

    def get_last_condense_time(self) -> float:
        return self._last_condense_ts


class _SummaryStorageBridge:
    """将SummaryStorageManager适配为HistoryCondenser所需的db_adapter接口"""

    def __init__(self):
        from src.hippo_memorizer.summary_storage import (
            get_summary_storage_manager,
            TopicSummary,
        )
        self._mgr = get_summary_storage_manager()
        self._TopicSummary = TopicSummary

    async def save_topic_summary(
        self, channel_id: str, topic: str, summary: str,
        keywords: list, key_points: list, participants: list,
        start_time: float, end_time: float, original_text: str = "",
    ):
        ts = self._TopicSummary(
            topic=topic,
            summary=summary,
            keywords=keywords or [],
            key_points=key_points or [],
            participants=participants or [],
            start_time=start_time,
            end_time=end_time,
            original_text=original_text,
        )
        self._mgr.add_summary(channel_id, ts)


class CondenserManager:
    def __init__(self):
        self._condensers: Dict[str, HistoryCondenser] = {}
        self._llm_bridge = None
        self._db_adapter = None
        # 自动绑定summary_storage作为默认持久化适配器
        try:
            self._db_adapter = _SummaryStorageBridge()
        except ImportError:
            pass

    def set_llm_bridge(self, bridge):
        self._llm_bridge = bridge
        for condenser in self._condensers.values():
            condenser.set_llm_bridge(bridge)

    def set_db_adapter(self, adapter):
        self._db_adapter = adapter
        for condenser in self._condensers.values():
            condenser.set_db_adapter(adapter)

    def get_condenser(self, channel_id: str) -> HistoryCondenser:
        if channel_id not in self._condensers:
            condenser = HistoryCondenser(
                channel_id, self._llm_bridge, self._db_adapter
            )
            self._condensers[channel_id] = condenser
        return self._condensers[channel_id]

    def add_packet(self, channel_id: str, packet: Dict):
        condenser = self.get_condenser(channel_id)
        condenser.add_packet(packet)

    async def check_and_condense(self, channel_id: str):
        condenser = self.get_condenser(channel_id)
        if condenser.should_trigger_condense():
            await condenser.trigger_condense_task()

    async def check_all(self):
        for _channel_id, condenser in self._condensers.items():
            if condenser.should_trigger_condense():
                await condenser.trigger_condense_task()

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "channels": len(self._condensers),
            "details": {
                cid: {
                    "batch_size": c.get_current_batch_size(),
                    "last_condense": c.get_last_condense_time(),
                }
                for cid, c in self._condensers.items()
            },
        }


_manager_instance: Optional[CondenserManager] = None


def get_condenser_manager() -> CondenserManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = CondenserManager()
    return _manager_instance

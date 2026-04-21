import time
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from src.common.logger import get_logger
from src.common.atomic_io import atomic_json_dump
from src.hippo_memorizer.config_loader import get_max_summaries

logger = get_logger("摘要存储")

_HIPPO_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "hippo_memorizer"
)


@dataclass
class TopicSummary:
    topic: str
    summary: str
    keywords: List[str]
    key_points: List[str]
    participants: List[str]
    start_time: float
    end_time: float
    original_text: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "topic": self.topic,
            "summary": self.summary,
            "keywords": self.keywords,
            "key_points": self.key_points,
            "participants": self.participants,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "original_text": self.original_text,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TopicSummary":
        summary = cls(
            topic=data.get("topic", ""),
            summary=data.get("summary", ""),
            keywords=data.get("keywords", []),
            key_points=data.get("key_points", []),
            participants=data.get("participants", []),
            start_time=data.get("start_time", 0),
            end_time=data.get("end_time", 0),
            original_text=data.get("original_text", ""),
        )
        summary.created_at = data.get("created_at", time.time())
        return summary

    def get_age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600

    def get_duration_str(self) -> str:
        duration = self.end_time - self.start_time
        if duration < 60:
            return f"{int(duration)}秒"
        elif duration < 3600:
            return f"{int(duration / 60)}分钟"
        return f"{int(duration / 3600)}小时"

    @property
    def clue_tags(self) -> List[str]:
        return self.keywords


class SummaryStorage:
    def __init__(self, chat_id: str, max_summaries: Optional[int] = None):
        self.chat_id = chat_id
        self._max_summaries = max_summaries or get_max_summaries()
        self._summaries: List[TopicSummary] = []
        self._file_path = (
            _HIPPO_DATA_DIR / f"{self._sanitize_id(chat_id)}_summaries.json"
        )
        self._load_from_disk()

    def _sanitize_id(self, chat_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", chat_id)

    def _load_from_disk(self):
        try:
            if self._file_path.exists():
                with self._file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data.get("summaries", []):
                    self._summaries.append(TopicSummary.from_dict(item))
                logger.info(f"已加载 {len(self._summaries)} 个话题摘要")
        except Exception as e:
            logger.error(f"加载话题摘要失败: {e}")

    def _save_to_disk(self):
        try:
            _HIPPO_DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "chat_id": self.chat_id,
                "summaries": [s.to_dict() for s in self._summaries],
            }
            atomic_json_dump(
                data, self._file_path, ensure_ascii=False, indent=2
            )
        except Exception as e:
            logger.error(f"保存话题摘要失败: {e}")

    def add_summary(self, summary: TopicSummary):
        self._summaries.append(summary)
        if len(self._summaries) > self._max_summaries:
            self._summaries = self._summaries[-self._max_summaries:]
        self._save_to_disk()

    def get_summaries(
        self, limit: int = 10, max_age_hours: Optional[float] = None
    ) -> List[TopicSummary]:
        result = self._summaries
        if max_age_hours is not None:
            cutoff = time.time() - (max_age_hours * 3600)
            result = [s for s in result if s.created_at >= cutoff]
        return result[-limit:]

    def search_by_clue(self, clue: str) -> List[TopicSummary]:
        clue_lower = clue.lower()
        return [
            s
            for s in self._summaries
            if clue_lower in s.summary.lower()
            or any(clue_lower in k.lower() for k in s.clue_tags)
            or clue_lower in s.topic.lower()
        ]

    def search_by_keyword(self, keyword: str) -> List[TopicSummary]:
        return self.search_by_clue(keyword)

    def search_by_label(self, label: str) -> List[TopicSummary]:
        return self.search_by_clue(label)

    def search_by_participant(self, participant: str) -> List[TopicSummary]:
        participant_lower = participant.lower()
        return [
            s
            for s in self._summaries
            if any(participant_lower in p.lower() for p in s.participants)
        ]

    def get_recent_topics(self, hours: float = 24.0) -> List[str]:
        cutoff = time.time() - (hours * 3600)
        return [s.topic for s in self._summaries if s.created_at >= cutoff]

    def get_all_clues(self) -> List[str]:
        keywords = []
        for s in self._summaries:
            keywords.extend(s.clue_tags)
        return list(set(keywords))

    def get_all_keywords(self) -> List[str]:
        return self.get_all_clues()

    def get_all_labels(self) -> List[str]:
        return self.get_all_clues()

    def clear_old_summaries(self, max_age_hours: float = 168.0):
        cutoff = time.time() - (max_age_hours * 3600)
        original_count = len(self._summaries)
        self._summaries = [
            s for s in self._summaries if s.created_at >= cutoff
        ]
        if len(self._summaries) < original_count:
            self._save_to_disk()
            logger.info(
                f"清理了 {original_count - len(self._summaries)} 个过期摘要"
            )

    def get_statistics(self) -> Dict[str, Any]:
        if not self._summaries:
            return {
                "chat_id": self.chat_id,
                "total_summaries": 0,
                "oldest_summary": None,
                "newest_summary": None,
            }
        return {
            "chat_id": self.chat_id,
            "total_summaries": len(self._summaries),
            "oldest_summary": (
                self._summaries[0].created_at if self._summaries else None
            ),
            "newest_summary": (
                self._summaries[-1].created_at if self._summaries else None
            ),
            "total_clues": len(self.get_all_clues()),
            "total_keywords": len(self.get_all_clues()),
        }


class SummaryStorageManager:
    def __init__(self):
        self._storages: Dict[str, SummaryStorage] = {}

    def get_storage(self, chat_id: str) -> SummaryStorage:
        if chat_id not in self._storages:
            self._storages[chat_id] = SummaryStorage(chat_id)
        return self._storages[chat_id]

    def add_summary(self, chat_id: str, summary: TopicSummary):
        storage = self.get_storage(chat_id)
        storage.add_summary(summary)

    def get_summaries(
        self, chat_id: str, limit: int = 10
    ) -> List[TopicSummary]:
        storage = self._storages.get(chat_id)
        if not storage:
            return []
        return storage.get_summaries(limit)

    def search_all(
        self, keyword: str, limit: int = 20
    ) -> Dict[str, List[TopicSummary]]:
        results = {}
        for chat_id, storage in self._storages.items():
            matches = storage.search_by_clue(keyword)
            if matches:
                results[chat_id] = matches[:limit]
        return results

    def search_all_by_label(
        self, label: str, limit: int = 20
    ) -> Dict[str, List[TopicSummary]]:
        return self.search_all(label, limit)

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        return {
            chat_id: storage.get_statistics()
            for chat_id, storage in self._storages.items()
        }

    def clear_all_old(self, max_age_hours: float = 168.0):
        for storage in self._storages.values():
            storage.clear_old_summaries(max_age_hours)


_storage_manager_instance: Optional[SummaryStorageManager] = None


def get_summary_storage_manager() -> SummaryStorageManager:
    global _storage_manager_instance
    if _storage_manager_instance is None:
        _storage_manager_instance = SummaryStorageManager()
    return _storage_manager_instance

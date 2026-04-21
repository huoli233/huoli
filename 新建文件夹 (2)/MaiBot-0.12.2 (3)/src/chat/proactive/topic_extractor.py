import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from src.common.logger import get_logger
from src.chat.proactive.models import MindLogRecord, ActivityKind

logger = get_logger("topic_extractor")


@dataclass
class TopicInfo:
    keywords: List[str] = field(default_factory=list)
    summary: str = ""
    message_count: int = 0
    participants: List[str] = field(default_factory=list)
    last_activity: float = 0.0
    relevance_score: float = 0.0

    def is_valid(self) -> bool:
        return len(self.keywords) > 0 or bool(self.summary)

    def age_minutes(self) -> float:
        if self.last_activity <= 0:
            return float('inf')
        return (time.time() - self.last_activity) / 60

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keywords": self.keywords,
            "summary": self.summary,
            "message_count": self.message_count,
            "participants": self.participants,
            "last_activity": self.last_activity,
            "relevance_score": self.relevance_score,
        }

    def is_interesting(self, threshold: float = 0.5) -> bool:
        return self.relevance_score >= threshold


class GroupTopicExtractor:
    STOP_WORDS: Set[str] = {
        "的", "了", "是", "在", "我", "有", "和", "就",
        "不", "人", "都", "一", "一个", "上", "也", "很",
        "到", "说", "要", "去", "你", "会", "着", "没有",
        "看", "好", "自己", "这", "那", "这个", "那个",
        "什么", "怎么", "为什么", "吗", "呢", "啊", "吧",
        "哈", "哈哈", "哈哈哈", "嗯", "噢", "哦", "额",
        "然后", "但是", "因为", "所以", "如果", "虽然",
        "可以", "应该", "可能", "还是", "或者", "而且",
        "the", "a", "an", "is", "are", "was", "were",
        "i", "you", "he", "she", "it", "we", "they",
        "and", "or", "but", "if", "then", "so", "for",
    }
    EMOJI_PATTERN = re.compile(
        r'[\U0001F600-\U0001F64F'
        r'\U0001F300-\U0001F5FF'
        r'\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF'
        r'\U00002702-\U000027B0'
        r'\U000024C2-\U0001F251]+'
    )

    def __init__(self, min_word_length: int = 2, max_keywords: int = 10, recency_weight: float = 0.3):
        self.min_word_length = min_word_length
        self.max_keywords = max_keywords
        self.recency_weight = recency_weight

    def extract_from_records(self, records: List[MindLogRecord], max_age_hours: float = 4.0) -> TopicInfo:
        if not records:
            return TopicInfo()
        cutoff_time = time.time() - (max_age_hours * 3600)
        user_messages = [
            r for r in records
            if r.kind == ActivityKind.INCOMING_MSG
            and r.recorded_at >= cutoff_time
            and r.text_content.strip()
        ]
        if not user_messages:
            logger.debug("[TopicExtractor] 没有找到有效的用户消息")
            return TopicInfo()
        participants = list(set(
            r.sender_name or r.sender_id
            for r in user_messages
            if r.sender_name or r.sender_id
        ))
        all_texts = [r.text_content for r in user_messages]
        keywords = self._extract_keywords(all_texts)
        summary = self._generate_summary(all_texts, keywords)
        relevance = self._calculate_relevance(user_messages)
        topic_info = TopicInfo(
            keywords=keywords[:self.max_keywords],
            summary=summary,
            message_count=len(user_messages),
            participants=participants[:5],
            last_activity=max(r.recorded_at for r in user_messages),
            relevance_score=relevance,
        )
        logger.debug(f"[TopicExtractor] 提取到话题: 关键词={keywords[:5]}, 消息数={len(user_messages)}")
        return topic_info

    def extract_from_texts(self, texts: List[str]) -> TopicInfo:
        if not texts:
            return TopicInfo()
        keywords = self._extract_keywords(texts)
        summary = self._generate_summary(texts, keywords)
        return TopicInfo(
            keywords=keywords[:self.max_keywords],
            summary=summary,
            message_count=len(texts),
            last_activity=time.time(),
            relevance_score=min(len(texts) / 10, 1.0),
        )

    def _extract_keywords(self, texts: List[str]) -> List[str]:
        word_counter = Counter()
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                if self._is_valid_keyword(word):
                    word_counter[word] += 1
        sorted_words = [
            word for word, count in word_counter.most_common(self.max_keywords * 2)
            if count >= 1
        ]
        return sorted_words[:self.max_keywords]

    def _tokenize(self, text: str) -> List[str]:
        text = self.EMOJI_PATTERN.sub(' ', text)
        text = re.sub(r'https?://\S+', ' ', text)
        text = re.sub(r'[@#]\S+', ' ', text)
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        english_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
        number_units = re.findall(r'\d+[天小时分钟秒年月日号点]', text)
        return chinese_words + english_words + number_units

    def _is_valid_keyword(self, word: str) -> bool:
        if len(word) < self.min_word_length:
            return False
        if word.lower() in self.STOP_WORDS:
            return False
        if word.isdigit():
            return False
        return True

    def _generate_summary(self, texts: List[str], keywords: List[str]) -> str:
        if not texts:
            return ""
        recent_snippets = []
        for text in texts[-3:]:
            snippet = text[:50].strip()
            if len(text) > 50:
                snippet += "..."
            recent_snippets.append(snippet)
        if keywords:
            keyword_str = "、".join(keywords[:3])
            summary = f"群友正在讨论: {keyword_str}"
        elif recent_snippets:
            summary = f"最近的消息: {recent_snippets[-1]}"
        else:
            summary = "群里有一些讨论"
        return summary

    def _calculate_relevance(self, messages: List[MindLogRecord]) -> float:
        if not messages:
            return 0.0
        count_factor = min(len(messages) / 10, 1.0)
        if len(messages) > 1:
            time_span = messages[-1].recorded_at - messages[0].recorded_at
            concentration = 1.0 / (1.0 + time_span / 3600)
        else:
            concentration = 0.5
        unique_senders = len(set(
            r.sender_id or r.sender_name
            for r in messages
            if r.sender_id or r.sender_name
        ))
        diversity = min(unique_senders / 5, 1.0)
        relevance = (count_factor * 0.4 + concentration * 0.3 + diversity * 0.3)
        return round(relevance, 3)

    def format_for_prompt(self, topic_info: TopicInfo, include_participants: bool = True) -> str:
        if not topic_info.is_valid():
            return "（没有明确的话题）"
        parts = []
        if topic_info.summary:
            parts.append(topic_info.summary)
        if topic_info.keywords:
            kw_str = "、".join(topic_info.keywords[:5])
            parts.append(f"涉及: {kw_str}")
        if include_participants and topic_info.participants:
            ppl_str = "、".join(topic_info.participants[:3])
            if len(topic_info.participants) > 3:
                ppl_str += f" 等{len(topic_info.participants)}人"
            parts.append(f"参与者: {ppl_str}")
        age = topic_info.age_minutes()
        if age < 60:
            time_str = f"{int(age)}分钟前"
        elif age < 1440:
            time_str = f"{int(age / 60)}小时前"
        else:
            time_str = f"{int(age / 1440)}天前"
        parts.append(f"（最后活动: {time_str}）")
        return " | ".join(parts)


_topic_extractor: Optional[GroupTopicExtractor] = None


def get_topic_extractor() -> GroupTopicExtractor:
    global _topic_extractor
    if _topic_extractor is None:
        _topic_extractor = GroupTopicExtractor()
    return _topic_extractor


async def extract_group_topics(
    session, max_messages: int = 30, max_age_hours: float = 4.0,
) -> TopicInfo:
    try:
        records = session.fetch_recent_records(limit=max_messages)
        if not records:
            return TopicInfo()
        extractor = get_topic_extractor()
        topic_info = extractor.extract_from_records(records, max_age_hours=max_age_hours)
        return topic_info
    except Exception as e:
        logger.warning(f"[TopicExtractor] 提取话题失败: {e}")
        return TopicInfo()

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from src.common.logger import get_logger

logger = get_logger("学习中枢")

_learning_instances: Dict[str, "LearningHub"] = {}
_MEME_HINT_KEYWORDS = {
    "哈哈哈",
    "笑死",
    "绷不住",
    "乐了",
    "草",
    "6",
    "xswl",
    "2333",
    "yyds",
    "梗",
    "离谱",
    "绝了",
}
_ETIQUETTE_HINT_KEYWORDS = {
    "早安",
    "晚安",
    "欢迎",
    "欢迎新人",
    "辛苦了",
    "谢谢",
    "生日快乐",
    "新年快乐",
    "恭喜",
    "打扰了",
    "抱歉",
}
# 表情包语义分类映射：名称片段 → (情绪类别, 语义含义)
_STICKER_SEMANTIC_MAP: Dict[str, Tuple[str, str]] = {
    # 开心/正面
    "微笑": ("happy", "温和的善意"),
    "笑": ("happy", "开心愉悦"),
    "哈哈": ("happy", "真心大笑"),
    "开心": ("happy", "高兴满足"),
    "可爱": ("happy", "感到可爱"),
    "比心": ("happy", "表达喜爱"),
    "爱心": ("happy", "表达爱意"),
    "赞": ("happy", "认同鼓励"),
    "耶": ("happy", "庆祝欢呼"),
    "doge": ("funny", "调侃玩梗"),
    "滑稽": ("funny", "故意搞笑"),
    "狗头": ("funny", "反讽暗示"),
    "捂脸": ("funny", "尴尬但好笑"),
    "偷笑": ("funny", "暗中觉得好笑"),
    "坏笑": ("funny", "调皮捣蛋"),
    # 消极/难过
    "哭": ("sad", "难过想哭"),
    "委屈": ("sad", "感到委屈"),
    "难过": ("sad", "心情低落"),
    "失望": ("sad", "感到失望"),
    "叹气": ("sad", "无奈叹息"),
    "心碎": ("sad", "伤心欲绝"),
    # 愤怒/不满
    "怒": ("angry", "生气愤怒"),
    "生气": ("angry", "不满发火"),
    "翻白眼": ("angry", "无语至极"),
    "抓狂": ("angry", "快要崩溃"),
    "摔": ("angry", "暴躁发泄"),
    # 困惑/惊讶
    "问号": ("confused", "疑惑不解"),
    "疑问": ("confused", "心存疑问"),
    "黑人问号": ("confused", "完全懵圈"),
    "震惊": ("surprised", "非常震惊"),
    "惊讶": ("surprised", "意外惊讶"),
    "吃惊": ("surprised", "出乎预料"),
    # 中性/日常
    "ok": ("neutral", "同意确认"),
    "好的": ("neutral", "简单应答"),
    "嗯": ("neutral", "简短回应"),
    "再见": ("neutral", "告别挥手"),
    "加油": ("encouraging", "打气鼓励"),
    "抱拳": ("neutral", "礼貌致意"),
    "握手": ("neutral", "友好协议"),
    # 害羞/尴尬
    "害羞": ("shy", "不好意思"),
    "脸红": ("shy", "感觉害臊"),
    "尴尬": ("awkward", "社交尴尬"),
    "冷汗": ("awkward", "紧张不安"),
    "无语": ("awkward", "哑口无言"),
    # 特殊状态
    "睡觉": ("tired", "想休息了"),
    "困": ("tired", "犯困疲惫"),
    "吃瓜": ("curious", "围观吃瓜"),
    "呲牙": ("happy", "憨笑"),
    "酷": ("cool", "觉得很酷"),
}
# 上下文伴随词 → 表情包含义修正权重
_STICKER_CONTEXT_HINTS: Dict[str, str] = {
    "绝了": "funny",
    "笑死我了": "happy",
    "太离谱了": "funny",
    "无语了": "awkward",
    "呜呜": "sad",
    "气死": "angry",
    "好可爱": "happy",
    "真不错": "happy",
    "吓到了": "surprised",
    "好无聊": "tired",
    "好期待": "happy",
    "心疼": "sad",
}


class LearnableType(Enum):
    """可学习的内容类型"""

    MEME_WORD = "meme_word"
    USER_CATCHPHRASE = "user_catchphrase"
    STICKER_MEANING = "sticker_meaning"
    GROUP_ETIQUETTE = "group_etiquette"
    TOPIC_KNOWLEDGE = "topic_knowledge"
    REACTION_PATTERN = "reaction_pattern"


class LearningStage(Enum):
    """学习阶段"""

    OBSERVED = "observed"
    TENTATIVE = "tentative"
    PRACTICING = "practicing"
    INTERNALIZED = "internalized"
    DEPRECATED = "deprecated"

    def label(self) -> str:
        labels = {
            LearningStage.OBSERVED: "观察到",
            LearningStage.TENTATIVE: "试探中",
            LearningStage.PRACTICING: "练习中",
            LearningStage.INTERNALIZED: "已内化",
            LearningStage.DEPRECATED: "已过时",
        }
        return labels.get(self, "未知")


@dataclass
class LearningEvent:
    """单次学习事件"""

    event_type: LearnableType = LearnableType.MEME_WORD
    content: str = ""
    source_user: str = ""
    context_hint: str = ""
    occurred_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type.value,
            "content": self.content[:80],
            "source_user": self.source_user,
            "context_hint": self.context_hint[:50],
        }


@dataclass
class StickerSemanticResult:
    """表情包语义识别结果"""

    recognized: bool = False
    emotion_category: str = ""
    semantic_meaning: str = ""
    confidence: float = 0.0
    matched_pattern: str = ""
    context_adjusted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recognized": self.recognized,
            "emotion_category": self.emotion_category,
            "semantic_meaning": self.semantic_meaning,
            "confidence": round(self.confidence, 3),
            "matched_pattern": self.matched_pattern,
            "context_adjusted": self.context_adjusted,
        }


@dataclass
class LearnedItem:
    """已学到的条目"""

    item_id: str = ""
    item_type: LearnableType = LearnableType.MEME_WORD
    content: str = ""
    explanation: str = ""
    stage: LearningStage = LearningStage.OBSERVED
    first_seen_at: float = field(default_factory=time.time)
    last_reinforced: float = field(default_factory=time.time)
    observation_count: int = 1
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    source_users: Set[str] = field(default_factory=set)
    related_topics: List[str] = field(default_factory=list)

    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    def should_promote(self) -> bool:
        """是否应该晋升到下一阶段"""
        if self.stage == LearningStage.OBSERVED:
            return self.observation_count >= 3
        if self.stage == LearningStage.TENTATIVE:
            return self.usage_count >= 2 and self.success_rate() > 0.5
        if self.stage == LearningStage.PRACTICING:
            return self.usage_count >= 5 and self.success_rate() > 0.7
        return False

    def should_deprecate(self) -> bool:
        """是否应该标记为过时"""
        if self.stage == LearningStage.DEPRECATED:
            return False
        stale_days = (time.time() - self.last_reinforced) / 86400
        if stale_days > 30 and self.usage_count < 3:
            return True
        if self.failure_count > 5 and self.success_rate() < 0.2:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "type": self.item_type.value,
            "content": self.content[:80],
            "explanation": self.explanation[:60],
            "stage": self.stage.value,
            "stage_label": self.stage.label(),
            "observation_count": self.observation_count,
            "usage_count": self.usage_count,
            "success_rate": round(self.success_rate(), 3),
            "source_users": list(self.source_users)[:5],
        }


class LearningHub:
    """事件驱动学习中心
    当群聊中出现新梗、用户口头禅、表情包用法、礼仪规范等可学习内容时，
    通过事件驱动机制持续学习和积累，经过 观察→试探→练习→内化 的阶段进化。
    """

    def __init__(self, channel_id: str):
        self._channel_id = channel_id
        self._learned: Dict[str, LearnedItem] = {}
        self._event_log: List[LearningEvent] = []
        self._user_phrase_counts: Dict[str, Dict[str, int]] = {}
        self._last_summary: Dict[str, Any] = {
            "learned": False,
            "hint": "",
            "count": 0,
            "items": [],
            "types": [],
        }
        self._max_events = 100
        self._max_items = 200

    def observe(self, event: LearningEvent) -> LearnedItem:
        """接收一个学习事件"""
        self._event_log.append(event)
        if len(self._event_log) > self._max_events:
            self._event_log = self._event_log[-self._max_events:]
        key = self._make_key(event.event_type, event.content)
        if key in self._learned:
            item = self._learned[key]
            item.observation_count += 1
            item.last_reinforced = time.time()
            if event.source_user:
                item.source_users.add(event.source_user)
            self._check_promotion(item)
            return item
        item = LearnedItem(
            item_id=key,
            item_type=event.event_type,
            content=event.content,
            explanation=event.context_hint,
            stage=LearningStage.OBSERVED,
        )
        if event.source_user:
            item.source_users.add(event.source_user)
        self._learned[key] = item
        # 限制总量
        if len(self._learned) > self._max_items:
            self._purge_oldest()
        return item

    def observe_simple(
        self,
        content: str,
        *,
        event_type: LearnableType = LearnableType.MEME_WORD,
        source_user: str = "",
        context: str = "",
    ) -> LearnedItem:
        """简化入口"""
        return self.observe(
            LearningEvent(
                event_type=event_type,
                content=content,
                source_user=source_user,
                context_hint=context,
            )
        )

    def observe_message(
        self,
        *,
        user_id: str,
        text: str,
        semantic_category: str = "plain_text",
    ) -> Dict[str, Any]:
        """从单条消息提取可学习线索。"""
        normalized_text = self._normalize_content(text)
        learned_items: List[LearnedItem] = []
        seen_keys: Set[str] = set()
        for event in self._collect_message_events(
            user_id=user_id,
            text=text,
            normalized_text=normalized_text,
            semantic_category=semantic_category,
        ):
            event_key = self._make_key(event.event_type, event.content)
            if not event.content.strip() or event_key in seen_keys:
                continue
            seen_keys.add(event_key)
            learned_items.append(self.observe(event))
        self._last_summary = self._summarize_items(learned_items)
        return dict(self._last_summary)

    def observe_topic_keywords(
        self,
        keywords: List[str],
        *,
        source_user: str = "",
        context: str = "群内活跃话题",
    ) -> Dict[str, Any]:
        """从上游话题追踪器接收高频话题。"""
        learned_items: List[LearnedItem] = []
        seen_topics: Set[str] = set()
        for keyword in keywords:
            normalized = self._normalize_content(keyword)[:40]
            if (
                not normalized
                or (len(normalized) < 2 and normalized.isascii())
                or normalized in seen_topics
            ):
                continue
            seen_topics.add(normalized)
            learned_items.append(
                self.observe(
                    LearningEvent(
                        event_type=LearnableType.TOPIC_KNOWLEDGE,
                        content=normalized,
                        source_user=source_user,
                        context_hint=context,
                    )
                )
            )
        self._last_summary = self._summarize_items(learned_items)
        return dict(self._last_summary)

    def last_summary(self) -> Dict[str, Any]:
        return dict(self._last_summary)

    def record_usage(
        self,
        content: str,
        *,
        event_type: LearnableType = LearnableType.MEME_WORD,
        success: bool = True,
    ) -> None:
        """记录使用结果"""
        key = self._make_key(event_type, content)
        item = self._learned.get(key)
        if item is None:
            return
        item.usage_count += 1
        if success:
            item.success_count += 1
        else:
            item.failure_count += 1
        item.last_reinforced = time.time()
        self._check_promotion(item)
        self._check_deprecation(item)

    def get_item(
        self, content: str, event_type: LearnableType = LearnableType.MEME_WORD
    ) -> Optional[LearnedItem]:
        key = self._make_key(event_type, content)
        return self._learned.get(key)

    def get_internalized(self) -> List[LearnedItem]:
        """获取所有已内化的条目"""
        return [
            item
            for item in self._learned.values()
            if item.stage == LearningStage.INTERNALIZED
        ]

    def get_practicing(self) -> List[LearnedItem]:
        """获取正在练习的条目"""
        return [
            item
            for item in self._learned.values()
            if item.stage == LearningStage.PRACTICING
        ]

    def get_by_type(self, event_type: LearnableType) -> List[LearnedItem]:
        return [
            item
            for item in self._learned.values()
            if item.item_type == event_type
        ]

    def known_memes(self) -> List[str]:
        """获取所有已学会的梗词"""
        return [
            item.content
            for item in self._learned.values()
            if item.item_type == LearnableType.MEME_WORD
            and item.stage
            in (LearningStage.PRACTICING, LearningStage.INTERNALIZED)
        ]

    def known_etiquette(self) -> List[str]:
        """获取所有已学会的礼仪"""
        return [
            item.content
            for item in self._learned.values()
            if item.item_type == LearnableType.GROUP_ETIQUETTE
            and item.stage
            in (LearningStage.PRACTICING, LearningStage.INTERNALIZED)
        ]

    def summary(self) -> Dict[str, Any]:
        stage_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for item in self._learned.values():
            stage_counts[item.stage.value] = (
                stage_counts.get(item.stage.value, 0) + 1
            )
            type_counts[item.item_type.value] = (
                type_counts.get(item.item_type.value, 0) + 1
            )
        return {
            "total_items": len(self._learned),
            "by_stage": stage_counts,
            "by_type": type_counts,
            "recent_events": len(self._event_log),
            "last_summary": dict(self._last_summary),
        }

    def deprecation_sweep(self) -> int:
        """批量清理过时条目"""
        deprecated = 0
        for item in self._learned.values():
            if (
                item.should_deprecate()
                and item.stage != LearningStage.DEPRECATED
            ):
                item.stage = LearningStage.DEPRECATED
                deprecated += 1
        return deprecated

    # ────────────────── 内部方法 ──────────────────

    def _make_key(self, event_type: LearnableType, content: str) -> str:
        normalized = content.strip().lower()[:60]
        return f"{event_type.value}::{normalized}"

    def _check_promotion(self, item: LearnedItem) -> None:
        if item.should_promote():
            if item.stage == LearningStage.OBSERVED:
                item.stage = LearningStage.TENTATIVE
                logger.debug(
                    f"[学习] {self._channel_id} {item.content[:20]} 晋升→试探"
                )
            elif item.stage == LearningStage.TENTATIVE:
                item.stage = LearningStage.PRACTICING
                logger.debug(
                    f"[学习] {self._channel_id} {item.content[:20]} 晋升→练习"
                )
            elif item.stage == LearningStage.PRACTICING:
                item.stage = LearningStage.INTERNALIZED
                logger.debug(
                    f"[学习] {self._channel_id} {item.content[:20]} 晋升→内化"
                )

    def _check_deprecation(self, item: LearnedItem) -> None:
        if item.should_deprecate():
            item.stage = LearningStage.DEPRECATED
            logger.debug(
                f"[学习] {self._channel_id} {item.content[:20]} 标记过时"
            )

    def _collect_message_events(
        self,
        *,
        user_id: str,
        text: str,
        normalized_text: str,
        semantic_category: str,
    ) -> List[LearningEvent]:
        events: List[LearningEvent] = []
        if not normalized_text:
            return events
        etiquette_keyword = self._match_keyword(
            normalized_text, _ETIQUETTE_HINT_KEYWORDS
        )
        if etiquette_keyword:
            events.append(
                LearningEvent(
                    event_type=LearnableType.GROUP_ETIQUETTE,
                    content=etiquette_keyword,
                    source_user=user_id,
                    context_hint=f"礼仪表达:{text[:50]}",
                )
            )
        meme_keyword = self._match_keyword(
            normalized_text, _MEME_HINT_KEYWORDS
        )
        if meme_keyword:
            events.append(
                LearningEvent(
                    event_type=LearnableType.MEME_WORD,
                    content=meme_keyword,
                    source_user=user_id,
                    context_hint=f"梗线索:{text[:50]}",
                )
            )
        if semantic_category == "sticker":
            sticker_semantic = self._classify_sticker_semantic(
                sticker_text=normalized_text,
                context_text=text,
            )
            content_label = normalized_text[:24]
            if sticker_semantic.recognized:
                content_label = f"{content_label}|{
                    sticker_semantic.emotion_category}"
            events.append(
                LearningEvent(
                    event_type=LearnableType.STICKER_MEANING,
                    content=content_label,
                    source_user=user_id,
                    context_hint=(
                        sticker_semantic.semantic_meaning
                        if sticker_semantic.recognized
                        else "表情包/贴纸语义线索"
                    ),
                )
            )
        phrase_count = self._track_user_phrase(user_id, normalized_text)
        if phrase_count >= 2 and 4 <= len(normalized_text) <= 24:
            events.append(
                LearningEvent(
                    event_type=LearnableType.USER_CATCHPHRASE,
                    content=normalized_text[:40],
                    source_user=user_id,
                    context_hint="同一用户重复表达",
                )
            )
        return events

    def _track_user_phrase(self, user_id: str, normalized_text: str) -> int:
        if (
            not user_id
            or len(normalized_text) < 4
            or len(normalized_text) > 24
        ):
            return 0
        phrase_map = self._user_phrase_counts.setdefault(user_id, {})
        phrase_map[normalized_text] = phrase_map.get(normalized_text, 0) + 1
        if len(phrase_map) > 100:
            sorted_items = sorted(
                phrase_map.items(), key=lambda item: item[1], reverse=True
            )
            self._user_phrase_counts[user_id] = dict(sorted_items[:50])
            phrase_map = self._user_phrase_counts[user_id]
        return phrase_map.get(normalized_text, 0)

    # ────────────────── 表情包语义识别 ──────────────────

    def _classify_sticker_semantic(
        self,
        sticker_text: str,
        context_text: str = "",
    ) -> StickerSemanticResult:
        """对表情包进行语义分类，融合名称匹配与上下文修正"""
        result = StickerSemanticResult()
        clean_name = self._extract_sticker_name(sticker_text)
        if not clean_name:
            return result
        # 按名称片段在映射表中查找
        matched_key, emotion, meaning = self._lookup_sticker_pattern(
            clean_name
        )
        if matched_key:
            result.recognized = True
            result.emotion_category = emotion
            result.semantic_meaning = meaning
            result.matched_pattern = matched_key
            result.confidence = 0.85
        else:
            # 启发式推断：检查名称是否包含情绪类通用字
            guessed = self._heuristic_sticker_guess(clean_name)
            if guessed:
                result.recognized = True
                result.emotion_category = guessed[0]
                result.semantic_meaning = guessed[1]
                result.matched_pattern = ""
                result.confidence = 0.45
        # 上下文修正
        if context_text and result.recognized:
            adjusted_category = self._context_adjust_category(context_text)
            if (
                adjusted_category
                and adjusted_category != result.emotion_category
            ):
                result.emotion_category = adjusted_category
                result.context_adjusted = True
                result.confidence = min(result.confidence + 0.1, 1.0)
        return result

    def classify_sticker(
        self,
        sticker_text: str,
        *,
        context_text: str = "",
        source_user: str = "",
    ) -> Dict[str, Any]:
        """公开接口：对表情包进行语义识别并返回分类结果"""
        semantic = self._classify_sticker_semantic(sticker_text, context_text)
        if semantic.recognized and source_user:
            self.observe_simple(
                f"{sticker_text[:20]}|{semantic.emotion_category}",
                event_type=LearnableType.STICKER_MEANING,
                source_user=source_user,
                context=semantic.semantic_meaning,
            )
        return semantic.to_dict()

    def known_sticker_categories(self) -> Dict[str, int]:
        """汇总已学到的表情包情绪类别分布"""
        category_counts: Dict[str, int] = {}
        for item in self._learned.values():
            if item.item_type != LearnableType.STICKER_MEANING:
                continue
            parts = item.content.split("|")
            cat = parts[-1] if len(parts) >= 2 else "unknown"
            category_counts[cat] = (
                category_counts.get(cat, 0) + item.observation_count
            )
        return category_counts

    @staticmethod
    def _extract_sticker_name(raw: str) -> str:
        """提取表情包核心名称，去除方括号和前后缀"""
        text = raw.strip()
        # 去掉常见的 [xxx] 格式外壳
        while text.startswith("[") and "]" in text:
            text = text[1: text.index("]")]
        while text.startswith("【") and "】" in text:
            text = text[1: text.index("】")]
        # 去掉 face_xxx / emoji_ 等前缀
        for prefix in ("face_", "emoji_", "sticker_", "img_"):
            if text.lower().startswith(prefix):
                text = text[len(prefix):]
        return text.strip().lower()[:30]

    @staticmethod
    def _lookup_sticker_pattern(name: str) -> Tuple[str, str, str]:
        """在语义映射表中查找匹配项，返回 (匹配键, 情绪类别, 语义含义)"""
        # 先精确匹配
        if name in _STICKER_SEMANTIC_MAP:
            cat, meaning = _STICKER_SEMANTIC_MAP[name]
            return name, cat, meaning
        # 再子串匹配（取最长匹配优先）
        best_key = ""
        best_len = 0
        for pattern_key in _STICKER_SEMANTIC_MAP:
            if pattern_key in name and len(pattern_key) > best_len:
                best_key = pattern_key
                best_len = len(pattern_key)
        if best_key:
            cat, meaning = _STICKER_SEMANTIC_MAP[best_key]
            return best_key, cat, meaning
        return "", "", ""

    @staticmethod
    def _heuristic_sticker_guess(name: str) -> Optional[Tuple[str, str]]:
        """启发式猜测表情包类别，覆盖映射表以外的场景"""
        positive_chars = {"笑", "乐", "喜", "爱", "赞", "棒", "好"}
        negative_chars = {"哭", "泪", "怒", "骂", "烦", "恨", "怕"}
        neutral_chars = {"嗯", "哦", "呵", "额", "喂"}
        for ch in name:
            if ch in positive_chars:
                return "happy", "正面情绪表达"
            if ch in negative_chars:
                return "sad", "负面情绪表达"
        for ch in name:
            if ch in neutral_chars:
                return "neutral", "中性表达"
        return None

    @staticmethod
    def _context_adjust_category(context_text: str) -> str:
        """根据伴随文本修正表情包分类"""
        lower = context_text.lower().strip()
        for hint_phrase, adjusted_cat in _STICKER_CONTEXT_HINTS.items():
            if hint_phrase in lower:
                return adjusted_cat
        return ""

    def _normalize_content(self, text: str) -> str:
        compact = "".join(
            ch for ch in str(text or "").strip() if not ch.isspace()
        )
        return compact[:40]

    def _match_keyword(self, text: str, keyword_pool: Set[str]) -> str:
        for keyword in sorted(keyword_pool, key=len, reverse=True):
            if keyword and keyword in text:
                return keyword
        return ""

    def _summarize_items(self, items: List[LearnedItem]) -> Dict[str, Any]:
        deduped: List[Dict[str, Any]] = []
        seen_keys: Set[str] = set()
        for item in items:
            item_key = self._make_key(item.item_type, item.content)
            if item_key in seen_keys:
                continue
            seen_keys.add(item_key)
            deduped.append(
                {
                    "type": item.item_type.value,
                    "content": item.content[:30],
                    "stage": item.stage.value,
                }
            )
        if not deduped:
            return {
                "learned": False,
                "hint": "",
                "count": 0,
                "items": [],
                "types": [],
            }
        return {
            "learned": True,
            "hint": "；".join(item["content"] for item in deduped[:3]),
            "count": len(deduped),
            "items": deduped,
            "types": sorted({item["type"] for item in deduped}),
        }

    def _purge_oldest(self) -> None:
        """清除最老的已弃用条目"""
        deprecated_keys = [
            k
            for k, v in self._learned.items()
            if v.stage == LearningStage.DEPRECATED
        ]
        for k in deprecated_keys[:20]:
            del self._learned[k]
        if len(self._learned) <= self._max_items:
            return
        # 再清除最久未强化的
        sorted_items = sorted(
            self._learned.items(), key=lambda kv: kv[1].last_reinforced
        )
        excess = len(self._learned) - self._max_items
        for k, _ in sorted_items[:excess]:
            del self._learned[k]


def get_learning_hub(channel_id: str) -> LearningHub:
    if channel_id not in _learning_instances:
        _learning_instances[channel_id] = LearningHub(channel_id)
    return _learning_instances[channel_id]


def remove_learning_hub(channel_id: str) -> None:
    _learning_instances.pop(channel_id, None)

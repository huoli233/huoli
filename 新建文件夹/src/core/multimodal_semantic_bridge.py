import time
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from src.common.logger import get_logger

logger = get_logger("mm_bridge")

_bridge_singleton = None


class MediaType(Enum):
    """媒体类型"""

    IMAGE = "image"
    STICKER = "sticker"
    VIDEO = "video"
    VOICE = "voice"
    FILE = "file"
    UNKNOWN = "unknown"


class StickerSentiment(Enum):
    """表情包情感倾向"""

    POSITIVE_AGREE = "positive_agree"
    NEGATIVE_MOCK = "negative_mock"
    SARCASTIC = "sarcastic"
    WELCOME_BLESSING = "welcome_blessing"
    RIDICULE_TEASE = "ridicule_tease"
    CONFUSED_SURPRISED = "confused_surprised"
    NEUTRAL_REACTION = "neutral_reaction"
    GROUP_MEME_REFERENCE = "group_meme_reference"
    EMBARRASSED_SHY = "embarrassed_shy"
    ANGRY_FRUSTRATED = "angry_frustrated"

    def label(self) -> str:
        return {
            "positive_agree": "赞同/开心",
            "negative_mock": "嘲讽/阴阳怪气",
            "sarcastic": "讽刺",
            "welcome_blessing": "欢迎/祝福",
            "ridicule_tease": "调侃/起哄",
            "confused_surprised": "困惑/惊讶",
            "neutral_reaction": "中性反应",
            "group_meme_reference": "群梗引用",
            "embarrassed_shy": "害羞/尴尬",
            "angry_frustrated": "生气/烦躁",
        }.get(self.value, "")


class ImageTopicCategory(Enum):
    """图片话题分类"""

    MEME_FUNNY = "meme_funny"
    CUTE_PET = "cute_pet"
    FOOD_DRINK = "food_drink"
    SCENERY_NATURE = "scenery_nature"
    SELFIE_PERSON = "selfie_person"
    SCREENSHOT_TEXT = "screenshot_text"
    ANIME_COMICS = "anime_comics"
    INFOGRAPHIC = "infographic"
    ART_CREATIVE = "art_creative"
    NSFW_RISKY = "nsfw_risky"
    UNCATEGORIZED = "uncategorized"

    def decision_weight(self) -> float:
        return {
            "meme_funny": 0.55,
            "cute_pet": 0.45,
            "food_drink": 0.40,
            "scenery_nature": 0.25,
            "selfie_person": 0.70,
            "screenshot_text": 0.80,
            "anime_comics": 0.50,
            "infographic": 0.65,
            "art_creative": 0.35,
            "nsfw_risky": 0.95,
            "uncategorized": 0.30,
        }.get(self.value, 0.30)


@dataclass
class SemanticBridgeResult:
    """语义桥接结果——从原始媒体到决策因子的中间产物"""

    media_id: str = ""
    media_type: MediaType = MediaType.UNKNOWN
    original_hash: str = ""
    sticker_sentiment: Optional[StickerSentiment] = None
    image_topic: Optional[ImageTopicCategory] = None
    emotional_valence: float = 0.0
    engagement_suggestion: float = 0.0
    should_respond_to_media: bool = False
    response_hint: str = ""
    context_relevance: float = 0.0
    is_group_common_sticker: bool = False
    confidence: float = 0.5
    raw_description: str = ""
    bridged_at: float = field(default_factory=time.time)
    source_user: str = ""

    def to_decision_factors(self) -> Dict[str, float]:
        factors: Dict[str, float] = {}
        factors["media_emotional_valence"] = self.emotional_valence
        factors["media_engagement_hint"] = self.engagement_suggestion
        factors["media_context_relevance"] = self.context_relevance
        if self.sticker_sentiment:
            _sent_weights = {
                StickerSentiment.POSITIVE_AGREE: 0.35,
                StickerSentiment.NEGATIVE_MOCK: -0.20,
                StickerSentiment.SARCASTIC: -0.10,
                StickerSentiment.WELCOME_BLESSING: 0.25,
                StickerSentiment.RIDICULE_TEASE: 0.10,
                StickerSentiment.CONFUSED_SURPRISED: 0.15,
                StickerSentiment.GROUP_MEME_REFERENCE: 0.30,
                StickerSentiment.EMBARRASSED_SHY: 0.05,
                StickerSentiment.ANGRY_FRUSTRATED: -0.30,
                StickerSentiment.NEUTRAL_REACTION: 0.10,
            }
            factors["sticker_sentiment_weight"] = _sent_weights.get(
                self.sticker_sentiment, 0.1
            )
        if self.image_topic:
            factors["image_topic_priority"] = (
                self.image_topic.decision_weight()
            )
        if self.should_respond_to_media:
            factors["media_response_urgency"] = min(
                1.0, self.context_relevance * self.confidence
            )
        else:
            factors["media_response_urgency"] = 0.0
        return factors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.media_type.value,
            "sentiment": (
                self.sticker_sentiment.label()
                if self.sticker_sentiment
                else ""
            ),
            "topic": (self.image_topic.value if self.image_topic else ""),
            "valence": round(self.emotional_valence, 3),
            "engage": round(self.engagement_suggestion, 3),
            "should_respond": self.should_respond_to_media,
            "hint": self.response_hint[:40],
            "confidence": round(self.confidence, 3),
        }


@dataclass
class BatchBridgeResult:
    """批量桥接结果"""

    results: List[SemanticBridgeResult] = field(default_factory=list)
    total_processed: int = 0
    responded_count: int = 0
    skipped_count: int = 0
    aggregated_valence: float = 0.0
    aggregated_engagement: float = 0.0
    dominant_sentiment: Optional[StickerSentiment] = None
    has_nsfw_risk: bool = False
    group_sticker_detected: bool = False
    processed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total_processed,
            "responded": self.responded_count,
            "skipped": self.skipped_count,
            "avg_valence": round(self.aggregated_valence, 3),
            "avg_engage": round(self.aggregated_engagement, 3),
            "dominant_sent": (
                self.dominant_sentiment.label()
                if self.dominant_sentiment
                else ""
            ),
            "nsfw_risk": self.has_nsfw_risk,
            "group_sticker": self.group_sticker_detected,
        }


# 表情包关键词→情感映射
_STICKER_SENTIMENT_MAP: List[Tuple[List[str], StickerSentiment, float]] = [
    (
        ["哈哈", "笑死", "xswl", "233", "🤣", "😂", "hhh", "绷不住"],
        StickerSentiment.POSITIVE_AGREE,
        0.7,
    ),
    (["呵呵", "就这", "6", "真行", "厉害"], StickerSentiment.SARCASTIC, 0.65),
    (
        ["欢迎", "新人", "来啦", "报到", "加入", "你好呀"],
        StickerSentiment.WELCOME_BLESSING,
        0.8,
    ),
    (
        ["卧槽", "牛逼", "绝了", "离谱", "我靠", "真的假的"],
        StickerSentiment.RIDICULE_TEASE,
        0.6,
    ),
    (
        ["？？", "？", "哈？", "啊？", "什么鬼", "不会吧"],
        StickerSentiment.CONFUSED_SURPRISED,
        0.7,
    ),
    (
        ["滚", "烦", "闭嘴", "傻", "有病", "神经"],
        StickerSentiment.ANGRY_FRUSTRATED,
        0.85,
    ),
    (
        ["害羞", "不好意思", "呜呜", "😳", "🙈"],
        StickerSentiment.EMBARRASSED_SHY,
        0.75,
    ),
    (
        ["嗯嗯", "好的", "👍", "👌", "ok", "收到"],
        StickerSentiment.NEUTRAL_REACTION,
        0.6,
    ),
]

_IMAGE_TOPIC_KEYWORDS: List[Tuple[List[str], ImageTopicCategory]] = [
    (
        ["猫", "狗", "宠物", "猫猫", "狗狗", "🐱", "🐶"],
        ImageTopicCategory.CUTE_PET,
    ),
    (
        ["吃", "美食", "饭", "菜", "奶茶", "咖啡", "蛋糕"],
        ImageTopicCategory.FOOD_DRINK,
    ),
    (
        ["风景", "天空", "日落", "日出", "海", "山", "云"],
        ImageTopicCategory.SCENERY_NATURE,
    ),
    (["自拍", "照片", "我", "今天", "穿搭"], ImageTopicCategory.SELFIE_PERSON),
    (
        ["截图", "聊天记录", "对话", "通知", "报错"],
        ImageTopicCategory.SCREENSHOT_TEXT,
    ),
    (
        ["动漫", "二次元", "漫画", "番", "角色"],
        ImageTopicCategory.ANIME_COMICS,
    ),
    (["表情包", "meme", "梗图", "沙雕"], ImageTopicCategory.MEME_FUNNY),
]


class MultimodalSemanticBridge:
    """多模态语义桥接层
    在 multimodal_budgeter 的预算决策之后、决策链之前，
    将原始媒体内容转换为可被 ACFN / GAP-M / GAP-O 消费的语义因子。

    核心能力：
    1. 表情包情感分类（10类）+ 情感极性值计算
    2. 图片话题分类（10类）+ 决策权重映射
    3. 是否需要回复该媒体的判断（基于上下文相关性）
    4. 群常用表情包识别与去重
    5. 批量聚合——多条媒体的汇总情感/参与度/风险
    6. 输出统一 DecisionFactors 字典，可直接注入 FactorInput
    """

    def __init__(self):
        self._cache: Dict[str, SemanticBridgeResult] = {}
        self._max_cache = 200
        self._group_sticker_hashes: Set[str] = set()
        self._max_group_stickers = 100
        self._user_sticker_preferences: Dict[str, Set[str]] = {}
        self._recent_results: List[SemanticBridgeResult] = []
        self._max_recent = 50

    def bridge(
        self,
        media_id: str,
        media_type_str: str,
        *,
        description: str = "",
        context_text: str = "",
        source_user: str = "",
        is_at_bot: bool = False,
        is_reply_to_bot: bool = False,
        group_sticker_hint: bool = False,
        confidence: float = 0.5,
    ) -> SemanticBridgeResult:
        mtype = self._parse_media_type(media_type_str)
        content_hash = self._compute_hash(
            media_id, description or context_text[:60]
        )
        cached = self._cache.get(content_hash)
        if cached and not cached.is_expired(180):
            return cached
        result = SemanticBridgeResult(
            media_id=media_id,
            media_type=mtype,
            original_hash=content_hash,
            source_user=source_user,
            confidence=confidence,
            raw_description=description,
        )
        if mtype == MediaType.STICKER:
            self._classify_sticker(result, description, context_text)
        elif mtype == MediaType.IMAGE:
            self._classify_image(result, description, context_text)
        elif mtype in (MediaType.VIDEO, MediaType.VOICE):
            self._classify_av_media(result, description, context_text)
        result.should_respond_to_media = self._evaluate_should_respond(
            result, is_at_bot, is_reply_to_bot, context_text
        )
        result.response_hint = self._derive_response_hint(result)
        result.is_group_common_sticker = group_sticker_hint or (
            content_hash in self._group_sticker_hashes
        )
        if result.is_group_common_sticker and mtype == MediaType.STICKER:
            self._group_sticker_hashes.add(content_hash)
            if len(self._group_sticker_hashes) > self._max_group_stickers:
                self._group_sticker_hashes = set(
                    list(self._group_sticker_hashes)[
                        -self._max_group_stickers:
                    ]
                )
        self._cache[content_hash] = result
        if len(self._cache) > self._max_cache:
            cutoff = time.time() - 3600.0
            self._cache = {
                k: v for k, v in self._cache.items() if v.bridged_at > cutoff
            }
        self._recent_results.append(result)
        if len(self._recent_results) > self._max_recent:
            self._recent_results = self._recent_results[-self._max_recent:]
        return result

    def bridge_batch(
        self,
        items: List[Dict[str, Any]],
        *,
        context_text: str = "",
    ) -> BatchBridgeResult:
        batch = BatchBridgeResult()
        sentiment_counts: Dict[StickerSentiment, int] = {}
        valence_sum = 0.0
        engage_sum = 0.0
        for item in items:
            mid = str(item.get("media_id", "") or item.get("id", "") or "")
            mtype = str(item.get("media_type", "") or "image")
            desc = str(item.get("description", "") or "")
            uid = str(item.get("user_id", "") or "")
            is_at = bool(item.get("is_at_bot", False))
            is_reply = bool(item.get("is_reply_to_bot", False))
            r = self.bridge(
                mid,
                mtype,
                description=desc,
                context_text=context_text,
                source_user=uid,
                is_at_bot=is_at,
                is_reply_to_bot=is_reply,
            )
            batch.results.append(r)
            batch.total_processed += 1
            if r.should_respond_to_media:
                batch.responded_count += 1
            else:
                batch.skipped_count += 1
            valence_sum += r.emotional_valence
            engage_sum += r.engagement_suggestion
            if r.sticker_sentiment:
                sentiment_counts[r.sticker_sentiment] = (
                    sentiment_counts.get(r.sticker_sentiment, 0) + 1
                )
            if r.image_topic == ImageTopicCategory.NSFW_RISKY:
                batch.has_nsfw_risk = True
            if r.is_group_common_sticker:
                batch.group_sticker_detected = True
        n = max(1, batch.total_processed)
        batch.aggregated_valence = valence_sum / n
        batch.aggregated_engagement = engage_sum / n
        if sentiment_counts:
            batch.dominant_sentiment = max(
                sentiment_counts.items(), key=lambda x: x[1]
            )[0]
        return batch

    def get_aggregated_factors_for_acfn(
        self, batch_result: BatchBridgeResult
    ) -> Dict[str, float]:
        factors: Dict[str, float] = {}
        factors["mm_batch_avg_valence"] = batch_result.aggregated_valence
        factors["mm_batch_avg_engagement"] = batch_result.aggregated_engagement
        factors["mm_response_ratio"] = batch_result.responded_count / max(
            1, batch_result.total_processed
        )
        if batch_result.has_nsfw_risk:
            factors["mm_nsfw_risk_flag"] = 1.0
        if batch_result.group_sticker_detected:
            factors["mm_group_sticker_wave"] = 0.4
        if batch_result.dominant_sentiment:
            sent = batch_result.dominant_sentiment
            _weights = {
                StickerSentiment.POSITIVE_AGREE: 0.25,
                StickerSentiment.NEGATIVE_MOCK: -0.15,
                StickerSentiment.SARCASTIC: -0.08,
                StickerSentiment.WELCOME_BLESSING: 0.18,
                StickerSentiment.RIDICULE_TEASE: 0.08,
                StickerSentiment.ANGRY_FRUSTRATED: -0.22,
            }
            factors["mm_dominant_sentiment_mod"] = _weights.get(sent, 0.05)
        return factors

    # ────────────────── 分类方法 ──────────────────

    def _classify_sticker(
        self, result: SemanticBridgeResult, desc: str, ctx: str
    ) -> None:
        text = (desc + " " + ctx).lower()
        best_match: Optional[StickerSentiment] = None
        best_score = 0.0
        for keywords, sentiment, base_conf in _STICKER_SENTIMENT_MAP:
            hit_count = sum(1 for kw in keywords if kw in text)
            if hit_count > 0:
                score = base_conf * (0.6 + hit_count * 0.2)
                if score > best_score:
                    best_score = score
                    best_match = sentiment
        result.sticker_sentiment = (
            best_match or StickerSentiment.NEUTRAL_REACTION
        )
        result.confidence = best_score if best_score > 0 else 0.35
        _valence_map = {
            StickerSentiment.POSITIVE_AGREE: 0.6,
            StickerSentiment.WELCOME_BLESSING: 0.5,
            StickerSentiment.RIDICULE_TEASE: 0.2,
            StickerSentiment.NEUTRAL_REACTION: 0.1,
            StickerSentiment.CONFUSED_SURPRISED: 0.05,
            StickerSentiment.EMBARRASSED_SHY: 0.15,
            StickerSentiment.SARCASTIC: -0.15,
            StickerSentiment.NEGATIVE_MOCK: -0.25,
            StickerSentiment.ANGRY_FRUSTRATED: -0.5,
            StickerSentiment.GROUP_MEME_REFERENCE: 0.35,
        }
        result.emotional_valence = _valence_map.get(
            best_match or StickerSentiment.NEUTRAL_REACTION, 0.1
        )
        result.engagement_suggestion = (
            abs(result.emotional_valence) * 0.6 + 0.15
        )

    def _classify_image(
        self, result: SemanticBridgeResult, desc: str, ctx: str
    ) -> None:
        text = (desc + " " + ctx).lower()
        best_cat: Optional[ImageTopicCategory] = None
        best_hits = 0
        for keywords, category in _IMAGE_TOPIC_KEYWORDS:
            hits = sum(1 for kw in keywords if kw in text)
            if hits > best_hits:
                best_hits = hits
                best_cat = category
        result.image_topic = best_cat or ImageTopicCategory.UNCATEGORIZED
        result.confidence = min(0.9, 0.4 + best_hits * 0.15)
        topic_weights = {
            ImageTopicCategory.SELFIE_PERSON: 0.5,
            ImageTopicCategory.SCREENSHOT_TEXT: 0.4,
            ImageTopicCategory.NSFW_RISKY: -0.6,
            ImageTopicCategory.MEME_FUNNY: 0.35,
            ImageTopicCategory.CUTE_PET: 0.3,
            ImageTopicCategory.FOOD_DRINK: 0.2,
            ImageTopicCategory.INFOGRAPHIC: 0.15,
            ImageTopicCategory.SCENERY_NATURE: 0.1,
            ImageTopicCategory.ANIME_COMICS: 0.18,
            ImageTopicCategory.ART_CREATIVE: 0.12,
            ImageTopicCategory.UNCATEGORIZED: 0.05,
        }
        result.emotional_valence = topic_weights.get(
            best_cat or ImageTopicCategory.UNCATEGORIZED, 0.05
        )
        result.engagement_suggestion = (
            best_cat.decision_weight() if best_cat else 0.3
        ) * 0.7

    @staticmethod
    def _classify_av_media(
        result: SemanticBridgeResult, desc: str, ctx: str
    ) -> None:
        text = (desc + " " + ctx).lower()
        result.image_topic = ImageTopicCategory.UNCATEGORIZED
        result.emotional_valence = 0.1
        result.engagement_suggestion = 0.2
        if any(kw in text for kw in ["唱歌", "音乐", "歌", "语音"]):
            result.emotional_valence = 0.25
            result.engagement_suggestion = 0.35
        elif any(kw in text for kw in ["视频", "录屏", "直播"]):
            result.emotional_valence = 0.15
            result.engagement_suggestion = 0.28

    # ────────────────── 决策方法 ──────────────────

    @staticmethod
    def _evaluate_should_respond(
        result: SemanticBridgeResult, is_at: bool, is_reply: bool, ctx: str
    ) -> bool:
        if is_at or is_reply:
            return True
        if result.image_topic == ImageTopicCategory.NSFW_RISKY:
            return False
        if result.image_topic == ImageTopicCategory.SELFIE_PERSON:
            return result.context_relevance > 0.3
        if result.sticker_sentiment in (
            StickerSentiment.POSITIVE_AGREE,
            StickerSentiment.WELCOME_BLESSING,
        ):
            return result.context_relevance > 0.25
        if result.sticker_sentiment in (
            StickerSentiment.ANGRY_FRUSTRATED,
            StickerSentiment.NEGATIVE_MOCK,
        ):
            return result.context_relevance > 0.5
        if result.is_group_common_sticker:
            return False
        return (
            result.context_relevance > 0.4
            and result.engagement_suggestion > 0.3
        )

    @staticmethod
    def _derive_response_hint(result: SemanticBridgeResult) -> str:
        if result.media_type == MediaType.STICKER:
            if result.sticker_sentiment == StickerSentiment.POSITIVE_AGREE:
                return "可以回一个轻松的表情"
            if result.sticker_sentiment == StickerSentiment.WELCOME_BLESSING:
                return "可以简短回应表示欢迎"
            if result.sticker_sentiment == StickerSentiment.RIDICULE_TEASE:
                return "可以跟着起哄或围观"
            if result.sticker_sentiment == StickerSentiment.ANGRY_FRUSTRATED:
                return "避免卷入冲突，保持距离"
            if result.sticker_sentiment == StickerSentiment.SARCASTIC:
                return "注意语气，不要被带节奏"
            return "看情况决定是否跟表情"
        if result.media_type == MediaType.IMAGE:
            if result.image_topic == ImageTopicCategory.CUTE_PET:
                return "可以说句可爱之类的"
            if result.image_topic == ImageTopicCategory.FOOD_DRINK:
                return "可以评论食物"
            if result.image_topic == ImageTopicCategory.SELFIE_PERSON:
                return "可以夸一下或者评论照片"
            if result.image_topic == ImageTopicCategory.MEME_FUNNY:
                return "可以笑一下或接梗"
            return "看图片内容决定是否回应"
        return "非文本媒体，视情况处理"

    # ────────────────── 工具方法 ──────────────────

    @staticmethod
    def _parse_media_type(raw: str) -> MediaType:
        mapping = {
            "image": MediaType.IMAGE,
            "img": MediaType.IMAGE,
            "photo": MediaType.IMAGE,
            "pic": MediaType.IMAGE,
            "sticker": MediaType.STICKER,
            "face": MediaType.STICKER,
            "emoji": MediaType.STICKER,
            "expression": MediaType.STICKER,
            "video": MediaType.VIDEO,
            "voice": MediaType.VOICE,
            "audio": MediaType.VOICE,
            "file": MediaType.FILE,
        }
        return mapping.get(raw.lower().strip(), MediaType.UNKNOWN)

    @staticmethod
    def _compute_hash(media_id: str, content: str) -> str:
        raw = f"{media_id}:{content}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]


def get_multimodal_semantic_bridge() -> MultimodalSemanticBridge:
    global _bridge_singleton
    if _bridge_singleton is None:
        _bridge_singleton = MultimodalSemanticBridge()
    return _bridge_singleton

import re
import time
import math
from collections import Counter
from typing import Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("recall_toolkit")


# ==================== 记忆寿命分级 ====================


class RecordLifeStage:
    """记忆条目寿命阶段（6级体系）"""

    NASCENT = "nascent"
    INTRADAY = "intraday"
    PRIOR_DAY = "prior_day"
    SHORT_SPAN = "short_span"
    WEEKLY = "weekly"
    ARCHAIC = "archaic"


# 各阶段上界（秒）
STAGE_BOUNDARY_SECONDS = {
    RecordLifeStage.NASCENT: 3600,
    RecordLifeStage.INTRADAY: 86400,
    RecordLifeStage.PRIOR_DAY: 172800,
    RecordLifeStage.SHORT_SPAN: 259200,
    RecordLifeStage.WEEKLY: 604800,
}

# 各阶段基线清晰度
STAGE_SHARPNESS_FLOOR = {
    RecordLifeStage.NASCENT: 1.0,
    RecordLifeStage.INTRADAY: 0.88,
    RecordLifeStage.PRIOR_DAY: 0.72,
    RecordLifeStage.SHORT_SPAN: 0.55,
    RecordLifeStage.WEEKLY: 0.35,
    RecordLifeStage.ARCHAIC: 0.18,
}

# 停用词表，排除高频无意义 token
_STOP_TOKENS = frozenset(
    {
        "的",
        "了",
        "是",
        "在",
        "和",
        "就",
        "都",
        "我",
        "你",
        "他",
        "她",
        "它",
        "这",
        "那",
        "然后",
        "但是",
        "因为",
        "所以",
        "可以",
        "应该",
        "可能",
        "这个",
        "那个",
        "什么",
        "没有",
        "不是",
        "已经",
        "还是",
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "you",
        "your",
        "are",
        "was",
    }
)

# 中文分割标记
_SPLIT_ANCHORS = ("大家在聊", "今天在聊", "有人提到", "还有", "以及", "另外")


# ==================== 文本提取工具 ====================


def harvest_terms(raw_text: str, ceiling: int = 10) -> List[str]:
    """从文本中提取关键词列表

    使用加权评分和简单提取，采用权重计数器实现。
    中英文混合分词 + 停用词过滤 + 长度优先。
    """
    if not raw_text:
        return []
    normalized = _normalize_surface(raw_text)
    token_scores: Counter = Counter()
    for tok in _tokenize_mixed(normalized):
        if not _is_meaningful_token(tok):
            continue
        weight = 1.0
        if 2 <= len(tok) <= 4:
            weight = 1.6
        elif len(tok) > 4:
            weight = 1.2
        if tok.isascii():
            weight *= 0.85
        token_scores[tok] += weight
    ranked = [tok for tok, _ in token_scores.most_common(ceiling)]
    return ranked


def gauge_overlap_ratio(query: str, content: str) -> float:
    """计算查询与内容的关键词重叠率（Jaccard系数）"""
    q_set = set(harvest_terms(query))
    c_set = set(harvest_terms(content))
    if not q_set or not c_set:
        return 0.0
    intersection = len(q_set & c_set)
    union = len(q_set | c_set)
    return intersection / union if union > 0 else 0.0


def compose_recall_context(
    entries: List[Dict], char_budget: int = 2000
) -> str:
    """将记忆条目列表拼接为提示词上下文片段"""
    if not entries:
        return ""
    segments = []
    consumed = 0
    for ent in entries:
        body = ent.get("content", "")
        origin = ent.get("source", "未知")
        segment = f"[{origin}] {body}"
        if consumed + len(segment) > char_budget:
            break
        segments.append(segment)
        consumed += len(segment)
    return "\n".join(segments)


# ==================== 衰减模型 ====================


def halflife_decay(stamp: float, half_life_days: float = 7.0) -> float:
    """半衰期时间衰减模型

    返回 [0, 1] 范围的衰减因子，stamp越老结果越小。
    """
    elapsed_sec = time.time() - stamp
    elapsed_days = max(0.0, elapsed_sec / 86400.0)
    return math.pow(0.5, elapsed_days / max(0.01, half_life_days))


def exponential_strength_fade(
    base_val: float,
    delta_sec: float,
    tau: float = 86400.0,
) -> float:
    """指数衰减模型: result = base × exp(-Δt / τ)"""
    if delta_sec <= 0:
        return base_val
    return base_val * math.exp(-delta_sec / tau)


def exponential_emotion_fade(
    base_val: float,
    delta_sec: float,
    tau: float = 3600.0,
) -> float:
    """情绪指数衰减: result = base × exp(-Δt / τ)

    τ默认1小时，情绪衰减远快于记忆强度。
    """
    if delta_sec <= 0:
        return base_val
    return base_val * math.exp(-delta_sec / tau)


def linear_affinity_fade(
    base_val: float,
    delta_sec: float,
    drain_rate_per_min: float = 2.0,
) -> float:
    """亲密度线性衰减: result = base - (Δt/60 × rate)"""
    if delta_sec <= 0:
        return base_val
    minutes_passed = delta_sec / 60.0
    return max(0.0, base_val - minutes_passed * drain_rate_per_min)


# ==================== 清晰度计算 ====================


def _classify_life_stage(elapsed_sec: float) -> str:
    """根据已过秒数判定记忆寿命阶段"""
    for stage, boundary in STAGE_BOUNDARY_SECONDS.items():
        if elapsed_sec < boundary:
            return stage
    return RecordLifeStage.ARCHAIC


def calculate_memory_clarity(
    timestamp: float,
    importance: float = 0.5,
    access_count: int = 0,
) -> float:
    """计算记忆记录的当前清晰度

    清晰度 = 基线(寿命阶段) + 访问加成(最高0.28) + 重要性加成(最高0.12)
    """
    elapsed = time.time() - timestamp
    stage = _classify_life_stage(elapsed)
    baseline = STAGE_SHARPNESS_FLOOR.get(stage, 0.18)
    visit_bonus = min(0.28, access_count * 0.028)
    importance_bonus = importance * 0.12
    return min(1.0, baseline + visit_bonus + importance_bonus)


# ==================== 相似合并 ====================


def collapse_near_duplicates(
    entries: List[Dict],
    overlap_threshold: float = 0.78,
) -> List[Dict]:
    """合并高相似度记忆条目

    将重叠率超过阈值的条目合并为一条，保留所有来源内容。
    """
    if not entries:
        return []
    output: List[Dict] = []
    consumed: set = set()
    for i, ent_a in enumerate(entries):
        if i in consumed:
            continue
        cluster = [ent_a]
        for j, ent_b in enumerate(entries[i + 1:], start=i + 1):
            if j in consumed:
                continue
            overlap = gauge_overlap_ratio(
                ent_a.get("content", ""),
                ent_b.get("content", ""),
            )
            if overlap >= overlap_threshold:
                cluster.append(ent_b)
                consumed.add(j)
        if len(cluster) == 1:
            output.append(ent_a)
        else:
            fused_body = " | ".join(e.get("content", "") for e in cluster)
            merged = dict(ent_a)
            merged["content"] = fused_body
            merged["source"] = "merged"
            merged["count"] = len(cluster)
            output.append(merged)
        consumed.add(i)
    return output


# ==================== 文本工具 ====================


def clip_text(raw: str, budget: int = 500) -> str:
    """截断文本到指定长度，超出部分用省略号替代"""
    if len(raw) <= budget:
        return raw
    return raw[: budget - 3] + "..."


def assemble_recall_prompt(
    query: str,
    entries: List[Dict],
    preamble: str = "",
) -> str:
    """组装记忆检索提示词"""
    ctx = compose_recall_context(entries)
    parts = []
    if preamble:
        parts.append(preamble)
    if ctx:
        parts.append(f"相关记忆:\n{ctx}")
    parts.append(f"当前问题: {query}")
    return "\n\n".join(parts)


# ==================== 内部分词与规范化 ====================


def _normalize_surface(text: str) -> str:
    """文本表面规范化（小写、去URL、去特殊符号）"""
    lowered = (text or "").lower()
    lowered = re.sub(r"https?://\S+", " ", lowered)
    lowered = re.sub(r"[@#]\S+", " ", lowered)
    lowered = re.sub(r"[^\w\u4e00-\u9fff]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _tokenize_mixed(normalized: str) -> List[str]:
    """中英混合分词"""
    english_tokens = re.findall(r"[a-z0-9_]{2,}", normalized)
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    tokens: List[str] = list(english_tokens)
    for chunk in chinese_chunks:
        if len(chunk) <= 4:
            tokens.append(chunk)
        else:
            tokens.extend(_split_long_chinese(chunk))
    return tokens


def _split_long_chinese(chunk: str) -> List[str]:
    """将长中文片段拆分为较短的子片段（ngram + 分割标记）"""
    segments = re.split(f"({'|'.join(_SPLIT_ANCHORS)})", chunk)
    candidates: List[str] = []
    for seg in segments:
        seg = seg.strip()
        if len(seg) < 2 or seg in _STOP_TOKENS or seg in _SPLIT_ANCHORS:
            continue
        if len(seg) <= 4:
            candidates.append(seg)
        else:
            for size in (2, 3, 4):
                for start in range(len(seg) - size + 1):
                    gram = seg[start: start + size]
                    if gram not in _STOP_TOKENS:
                        candidates.append(gram)
    seen: set = set()
    unique: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _is_meaningful_token(tok: str) -> bool:
    """判断token是否有实际语义价值"""
    if len(tok) < 2:
        return False
    if tok in _STOP_TOKENS:
        return False
    if tok.isdigit():
        return False
    return True


# ==================== 多因子相关性评分 ====================


def compute_weighted_relevance(
    query: str,
    content: str,
    created_at: float = 0.0,
    access_count: int = 0,
    importance: float = 0.5,
    summary: str = "",
    tag_list: Optional[List[str]] = None,
) -> float:
    """多因子加权相关性评分（8因子模型）

    因子权重分配：
    - 加权token重叠: 40%
    - 普通token交集: 16%
    - 子串命中奖励: 20% (固定值)
    - 短语奖励: 动态
    - 时间新鲜度: 14%
    - 访问频率: 13%
    - 重要性: 20%
    - 标签命中: 10%
    """
    q_scores = _score_candidate_tokens(query)
    combined_text = " ".join(
        filter(None, [content, summary, " ".join(tag_list or [])])
    )
    c_scores = _score_candidate_tokens(combined_text)
    q_keys = set(q_scores)
    c_keys = set(c_scores)
    if not q_keys or not c_keys:
        return 0.0
    overlap_weighted = _weighted_token_overlap(q_scores, c_scores)
    plain_overlap = len(q_keys & c_keys) / max(1, len(q_keys))
    norm_q = _normalize_surface(query)
    norm_c = _normalize_surface(content)
    substring_hit = 0.2 if norm_q and norm_q in norm_c else 0.0
    phrase_hit = _compute_phrase_bonus(q_keys, c_keys)
    age_hours = (
        max(0.0, (time.time() - created_at) / 3600.0)
        if created_at > 0
        else 24.0
    )
    freshness = max(0.0, 0.14 - min(0.14, age_hours / 24.0 * 0.022))
    access_bonus = min(0.13, access_count * 0.022)
    importance_bonus = max(0.0, min(0.20, float(importance) * 0.20))
    tag_bonus = 0.0
    if tag_list and q_keys:
        tag_set = {t.lower() for t in tag_list}
        if q_keys & tag_set:
            tag_bonus = 0.10
    total = (
        overlap_weighted * 0.40
        + plain_overlap * 0.16
        + substring_hit
        + phrase_hit
        + freshness
        + access_bonus
        + importance_bonus
        + tag_bonus
    )
    return round(max(0.0, min(1.0, total)), 4)


def _score_candidate_tokens(text: str) -> Counter:
    """对文本token加权评分"""
    counter: Counter = Counter()
    norm = _normalize_surface(text)
    for tok in _tokenize_mixed(norm):
        if not _is_meaningful_token(tok):
            continue
        w = 1.0
        if 2 <= len(tok) <= 4:
            w = 1.6
        elif len(tok) > 4:
            w = 1.2
        if tok.isascii():
            w *= 0.85
        counter[tok] += w
    return counter


def _weighted_token_overlap(left: Counter, right: Counter) -> float:
    """加权token重叠得分"""
    shared = 0.0
    total = 0.0
    for tok, w in left.items():
        total += w
        if tok in right:
            shared += min(w, right[tok])
    return shared / max(1.0, total)


def _compute_phrase_bonus(left_keys: set, right_keys: set) -> float:
    """计算短语级别奖励"""
    bonus = 0.0
    for tok in left_keys:
        if len(tok) < 2:
            continue
        if any(tok != other and tok in other for other in right_keys):
            bonus = max(bonus, 0.10 if len(tok) <= 4 else 0.16)
    return bonus


# ==================== 短别名（兼容旧接口） ====================

extract_keywords = harvest_terms
calculate_relevance_score = gauge_overlap_ratio
format_memory_context = compose_recall_context
calculate_time_decay = halflife_decay
calculate_memory_decay_with_time_diff = exponential_strength_fade
calculate_emotion_decay_with_time_diff = exponential_emotion_fade
calculate_intimacy_decay_with_time_diff = linear_affinity_fade
merge_similar_memories = collapse_near_duplicates
truncate_memory = clip_text
build_memory_prompt = assemble_recall_prompt

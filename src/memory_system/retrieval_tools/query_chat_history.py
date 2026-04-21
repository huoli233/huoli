"""
聊天记录检索探针 —— 从ChatHistory概况库中按条件定位记忆条目。

融合三源能力：
- MaiBot: 多关键词容错匹配 + 时间范围过滤 + 全局/本地作用域 + 黑名单机制
- XBcore: BM25倒排索引回退 + 模糊匹配兜底
- MIMiaoCore: 类封装 + 相关性评分排序

完全原创的类结构和控制流。
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.common.database.database_model import ChatHistory
from src.common.logger import get_logger
from src.config.config import global_config
from src.memory_system.retrieval_tools.tool_registry import (
    register_memory_retrieval_tool,
)

logger = get_logger("历史探测")


# ==================== 黑名单过滤器 ====================


class StreamBlacklistFilter:
    """聊天流黑名单过滤器 —— 将配置中的黑名单条目解析为chat_id集合。"""

    def __init__(self):
        self._resolved_ids: Optional[Set[str]] = None
        self._last_resolution_ts: float = 0.0
        self._cache_ttl: float = 120.0

    def _require_resolution(self) -> Set[str]:
        """按需解析黑名单配置（带缓存）"""
        now = time.time()
        if (
            self._resolved_ids is not None
            and now - self._last_resolution_ts < self._cache_ttl
        ):
            return self._resolved_ids
        raw_list = getattr(global_config.memory, "global_memory_blacklist", [])
        resolved = set()
        if not raw_list:
            self._resolved_ids = resolved
            self._last_resolution_ts = now
            return resolved
        try:
            from src.chat.message_receive.chat_stream import get_chat_manager

            mgr = get_chat_manager()
            for entry in raw_list:
                if not isinstance(entry, str):
                    continue
                segments = entry.split(":")
                if len(segments) != 3:
                    logger.warning(
                        f"黑名单条目格式异常(需platform:id:type): {entry}"
                    )
                    continue
                plat, ident, stype = segments
                is_grp = stype == "group"
                sid = mgr.get_stream_id(plat, str(ident), is_group=is_grp)
                if sid:
                    resolved.add(sid)
                else:
                    logger.warning(f"无法将黑名单条目映射到stream_id: {entry}")
        except Exception as exc:
            logger.error(f"解析黑名单失败: {exc}")
        self._resolved_ids = resolved
        self._last_resolution_ts = now
        return resolved

    def is_blocked(self, stream_id: str) -> bool:
        """判断给定stream_id是否被黑名单拦截"""
        return stream_id in self._require_resolution()

    def blocked_set(self) -> Set[str]:
        """获取全部被拦截的stream_id集合"""
        return self._require_resolution()


_blacklist_guard = StreamBlacklistFilter()


# ==================== 线索拆分 ====================


def _dissect_search_terms(raw_input: str) -> List[str]:
    """将用户输入的检索线索拆分为独立词条列表。"""
    if not raw_input or not raw_input.strip():
        return []
    try:
        from src.chat.utils.utils import (
            parse_keywords_string as parse_legacy_terms,
        )

        parsed = parse_legacy_terms(raw_input)
        if parsed:
            return parsed
    except ImportError:
        pass
    import re

    return [
        t.strip() for t in re.split(r"[,，\s/、|]+", raw_input) if t.strip()
    ]


# ==================== 时间窗口解析 ====================


@dataclass
class TemporalWindow:
    """时间查询窗口"""

    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    @property
    def is_empty(self) -> bool:
        return self.lower_bound is None and self.upper_bound is None

    def validate(self) -> Optional[str]:
        """校验时间窗口合法性，返回错误信息或None"""
        if (
            self.lower_bound
            and self.upper_bound
            and self.lower_bound > self.upper_bound
        ):
            return "起始时间不能晚于截止时间"
        return None


def _build_temporal_window(
    raw_start: Optional[str],
    raw_end: Optional[str],
) -> Tuple[TemporalWindow, Optional[str]]:
    """解析时间字符串为TemporalWindow，返回(窗口, 错误信息)"""
    from src.memory_system.memory_utils import parse_datetime_to_timestamp

    window = TemporalWindow()
    if raw_start:
        try:
            window.lower_bound = parse_datetime_to_timestamp(raw_start)
        except ValueError as ve:
            return (
                window,
                f"起始时间格式有误: {ve}（支持: 2025-01-01 / 2025-01-01 12:00:00 / 2025/01/01）",
            )
    if raw_end:
        try:
            window.upper_bound = parse_datetime_to_timestamp(raw_end)
        except ValueError as ve:
            return (
                window,
                f"截止时间格式有误: {ve}（支持: 2025-01-01 / 2025-01-01 12:00:00 / 2025/01/01）",
            )
    err = window.validate()
    return window, err


# ==================== 匹配判定策略 ====================


def _extract_json_list(raw_field) -> List[str]:
    """从JSON字符串或列表中提取str列表"""
    if not raw_field:
        return []
    try:
        data = (
            json.loads(raw_field) if isinstance(raw_field, str) else raw_field
        )
        if isinstance(data, list):
            return [
                str(item).strip().lower() for item in data if str(item).strip()
            ]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    return []


def _extract_record_clues(record) -> List[str]:
    """兼容历史字段，从记录中提取可检索标签。"""
    return _extract_json_list(record.keywords)


def _compact_search_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or ""), flags=re.UNICODE).lower()


def _score_record_relevance(
    record,
    terms: List[str],
    participant: Optional[str] = None,
) -> float:
    """为聊天记录打相关性分，优先 coverage/phrase 命中，时间新近性只做 tie-break。"""
    theme = _compact_search_text(getattr(record, "theme", "") or "")
    summary = _compact_search_text(getattr(record, "summary", "") or "")
    original = _compact_search_text(getattr(record, "original_text", "") or "")
    keywords = [_compact_search_text(item) for item in _extract_record_clues(record)]

    score = 0.0
    term_hits = 0
    for term in terms:
        compact_term = _compact_search_text(term)
        if not compact_term:
            continue
        hit_score = 0.0
        if compact_term in theme:
            hit_score = max(hit_score, 1.2)
        if any(compact_term in kw for kw in keywords):
            hit_score = max(hit_score, 1.0)
        if compact_term in summary:
            hit_score = max(hit_score, 0.8)
        if compact_term in original:
            hit_score = max(hit_score, 0.6)
        if hit_score > 0.0:
            term_hits += 1
            score += hit_score

    if terms:
        coverage = term_hits / max(1, len(terms))
        score += coverage * 1.45
        if len(terms) >= 2 and term_hits < 2:
            score *= 0.65
        if len(terms) >= 2 and term_hits >= 2:
            score += 0.4
        query_compact = _compact_search_text("".join(terms))
        all_text = " ".join(part for part in (theme, summary, original, " ".join(keywords)) if part)
        phrase_hit = bool(query_compact and query_compact in all_text)
        if phrase_hit:
            score += 0.8
        if query_compact and all_text:
            overlap = len(set(query_compact) & set(all_text)) / max(1, len(set(query_compact)))
            score += overlap * 0.45

    if participant and _check_participant_presence(record, participant):
        score += 0.8

    try:
        age_hours = max(0.0, (time.time() - float(getattr(record, "start_time", 0.0) or 0.0)) / 3600.0)
    except Exception:
        age_hours = 9999.0
    if age_hours <= 24:
        score += 0.12
    elif age_hours <= 72:
        score += 0.06
    elif age_hours <= 24 * 7:
        score += 0.03
    return score


def _rank_matching_records(
    records: List,
    terms: List[str],
    participant: Optional[str] = None,
) -> List:
    scored: List[Tuple[float, Any]] = []
    for rec in records:
        score = _score_record_relevance(rec, terms, participant)
        scored.append((score, rec))
    scored.sort(
        key=lambda item: (
            item[0],
            float(getattr(item[1], "start_time", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return [rec for _, rec in scored]


def _check_participant_presence(record, target_name: str) -> bool:
    """判断记录的参与人列表中是否包含目标人名"""
    members = _extract_json_list(record.participants)
    target_lc = target_name.lower().strip()
    if not target_lc:
        return True
    return any(target_lc in m for m in members)


def _check_clue_coverage(record, terms: List[str]) -> bool:
    """判断记录是否满足检索线索覆盖要求。

    规则：词条数 <=3 时全部命中；词条数 >3 时允许漏掉1个。
    """
    if not terms:
        return True
    # 构建可搜索文本池
    haystack_parts = [
        (record.theme or "").lower(),
        (record.summary or "").lower(),
        (record.original_text or "").lower(),
    ]
    record_kw_list = _extract_record_clues(record)
    haystack_joined = " ".join(haystack_parts)
    hits = 0
    for term in terms:
        term_lc = term.lower()
        found_in_text = term_lc in haystack_joined
        found_in_kw = any(term_lc in kw for kw in record_kw_list)
        if found_in_text or found_in_kw:
            hits += 1
    required = len(terms) if len(terms) <= 3 else len(terms) - 1
    return hits >= required


def _apply_temporal_filter(base_query, window: TemporalWindow):
    """对Peewee查询追加时间范围条件"""
    if window.is_empty:
        return base_query
    if window.lower_bound is not None and window.upper_bound is not None:
        # 记录时间范围与查询窗口有交集的三种情形
        return base_query.where(
            (
                (ChatHistory.start_time >= window.lower_bound)
                & (ChatHistory.start_time <= window.upper_bound)
            )
            | (
                (ChatHistory.end_time >= window.lower_bound)
                & (ChatHistory.end_time <= window.upper_bound)
            )
            | (
                (ChatHistory.start_time <= window.lower_bound)
                & (ChatHistory.end_time >= window.upper_bound)
            )
        )
    if window.lower_bound is not None:
        return base_query.where(ChatHistory.end_time >= window.lower_bound)
    return base_query.where(ChatHistory.start_time <= window.upper_bound)


# ==================== 查询作用域构造器 ====================


def _construct_scoped_query(stream_id: str) -> Any:
    """根据全局/本地模式和黑名单构建Peewee查询基础"""
    current_blocked = _blacklist_guard.is_blocked(stream_id)
    allow_global = (
        getattr(global_config.memory, "global_memory", False)
        and not current_blocked
    )
    if allow_global:
        excluded = _blacklist_guard.blocked_set()
        if excluded:
            base_q = ChatHistory.select().where(
                ~(ChatHistory.chat_id.in_(excluded))
            )
            logger.debug(f"全局检索模式（排除{len(excluded)}条黑名单流）")
        else:
            base_q = ChatHistory.select()
            logger.debug("全局检索模式（无黑名单）")
    else:
        base_q = ChatHistory.select().where(ChatHistory.chat_id == stream_id)
        if current_blocked:
            logger.debug(f"当前流在黑名单中，强制本地检索: {stream_id}")
    return base_q


# ==================== 结果格式化 ====================


def _compose_record_brief(record) -> str:
    """将单条记录格式化为摘要文本"""
    parts = [f"记忆ID：{record.id}"]
    parts.append(f"主题：{record.theme or '（无）'}")
    kw_items = _extract_record_clues(record)
    if kw_items:
        parts.append(f"线索标签：{'、'.join(kw_items)}")
    else:
        parts.append("线索标签：（无）")
    return "\n".join(parts)


def _compose_record_detail(record) -> str:
    """将单条记录格式化为详情文本"""
    parts = [f"记忆ID：{record.id}"]
    if record.theme:
        parts.append(f"主题：{record.theme}")
    begin_str = datetime.fromtimestamp(record.start_time).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    fin_str = datetime.fromtimestamp(record.end_time).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    parts.append(f"时间：{begin_str} - {fin_str}")
    member_list = _extract_json_list(record.participants)
    if member_list:
        parts.append(f"参与人：{'、'.join(member_list)}")
    kw_items = _extract_record_clues(record)
    if kw_items:
        parts.append(f"线索标签：{'、'.join(kw_items)}")
    if record.summary:
        parts.append(f"概括：{record.summary}")
    return "\n".join(parts)


def _build_overflow_hint(matching_records: List, search_label: str) -> str:
    """当结果过多时，汇总标签作为缩窄提示"""
    aggregated_kw: Set[str] = set()
    for rec in matching_records:
        for kw in _extract_record_clues(rec):
            if kw:
                aggregated_kw.add(kw)
    if aggregated_kw:
        kw_text = "、".join(sorted(aggregated_kw))
        return (
            f'包含"{search_label}"的结果过多，请补充更多线索缩小范围\n\n'
            f"可继续补充的标签：\n{kw_text}"
        )
    return f'包含"{search_label}"的结果过多，请补充更多线索精确查找'


def _build_no_match_message(
    terms: List[str],
    person: Optional[str],
    window: TemporalWindow,
) -> str:
    """构建无匹配结果的提示消息"""
    conditions = []
    if terms:
        conditions.append(f"线索'{'、'.join(terms)}'")
    if person:
        conditions.append(f"参与人'{person}'")
    if not window.is_empty:
        parts = []
        if window.lower_bound:
            parts.append(
                f">={
                    datetime.fromtimestamp(
                        window.lower_bound).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        if window.upper_bound:
            parts.append(
                f"<={
                    datetime.fromtimestamp(
                        window.upper_bound).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        conditions.append(f"时间{' '.join(parts)}")
    if conditions:
        return f"未找到满足条件（{'且'.join(conditions)}）的聊天记录"
    return "未找到相关聊天记录"


# ==================== 模糊匹配兜底 ====================


def _attempt_fuzzy_fallback(
    terms: List[str],
    candidate_records: List,
    already_found: List,
) -> List:
    """线索精确匹配无结果时，调用模糊匹配器兜底"""
    if not terms:
        return already_found
    try:
        from src.memory_system.memory_search import get_fuzzy_matcher

        fuzzy = get_fuzzy_matcher()
        search_text = " ".join(terms)
        candidate_themes = [r.theme for r in candidate_records if r.theme]
        similar_hits = fuzzy.find_similar(
            search_text, candidate_themes, cutoff=0.5
        )
        theme_hit_set = {th for th, _ in similar_hits[:5]}
        augmented = list(already_found)
        for rec in candidate_records:
            if rec.theme in theme_hit_set and rec not in augmented:
                augmented.append(rec)
        return augmented
    except Exception:
        return already_found


async def _attempt_hybrid_fallback(
    stream_id: str,
    terms: List[str],
    candidate_records: List,
) -> List:
    """BM25+向量混合检索兜底：当精确与模糊匹配均无结果时尝试"""
    try:
        from src.memory_system.hybrid_retrieval import get_hybrid_retriever
        _retriever = get_hybrid_retriever()
        _query = " ".join(terms)
        _hits = await asyncio.wait_for(
            _retriever.search(_query, top_k=10),
            timeout=15.0,
        )
        if not _hits:
            return []
        _hit_ids = {str(getattr(h, "doc_id", "")) for h in _hits if h}
        matched = [
            rec for rec in candidate_records
            if str(getattr(rec, "id", "")) in _hit_ids
        ]
        return matched[:10]
    except asyncio.TimeoutError:
        logger.debug(f"混合检索兜底超时(15s)")
        return []
    except ImportError:
        return []
    except Exception:
        return []


# ==================== 核心检索函数 ====================

_FETCH_CEILING = 50  # 单次查询最大拉取条数
_OVERFLOW_THRESHOLD = 15  # 超过此数量返回关键词提示而非逐条展示
_DISPLAY_CAP = 20  # 逐条展示时的最大条数


async def probe_chat_history(
    stream_id: str,
    keyword: Optional[str] = None,
    participant: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """按条件检索ChatHistory概况库。

    参数:
        stream_id: 聊天流ID
        keyword: 检索线索（兼容旧参数名，支持多个，空格/逗号分隔；<=2个全部匹配; >2个允许漏1个）
        participant: 参与人昵称（模糊包含匹配）
        start_time: 起始时间字符串（可选）
        end_time: 截止时间字符串（可选）
        limit: 返回结果最大条数（可选）
    返回:
        文本检索结果
    """
    try:
        # 参数校验
        if not keyword and not participant and not start_time and not end_time:
            return "未指定查询参数（需提供keyword、participant、start_time或end_time之一）"
        probe_clock = time.time()
        # 缓存探测
        cache_fingerprint = (
            f"hist:{stream_id}:{keyword}:{participant}:{start_time}:{end_time}"
        )
        try:
            from src.memory_system.memory_search import get_search_cache

            cached_payload = get_search_cache().fetch(cache_fingerprint)
            if cached_payload is not None:
                return cached_payload
        except Exception:
            cached_payload = None
        # 解析时间窗口
        window, time_err = _build_temporal_window(start_time, end_time)
        if time_err:
            return time_err
        # 构建作用域查询
        base_query = _construct_scoped_query(stream_id)
        base_query = _apply_temporal_filter(base_query, window)
        fetch_ceiling = _FETCH_CEILING
        if keyword:
            fetch_ceiling = min(120, _FETCH_CEILING + 40)
        elif participant:
            fetch_ceiling = min(90, _FETCH_CEILING + 20)
        raw_records = await asyncio.to_thread(
            lambda: list(
                base_query.order_by(ChatHistory.start_time.desc()).limit(
                    fetch_ceiling
                )
            )
        )
        # 解析关键词
        terms = _dissect_search_terms(keyword) if keyword else []
        # 逐条筛选
        survivors = []
        for rec in raw_records:
            term_ok = _check_clue_coverage(rec, terms)
            person_ok = (
                _check_participant_presence(rec, participant)
                if participant
                else True
            )
            if term_ok and person_ok:
                survivors.append(rec)
        # 无结果时尝试模糊兜底
        if not survivors and terms:
            survivors = _attempt_fuzzy_fallback(terms, raw_records, survivors)
        # 模糊兜底仍无结果时尝试混合检索（BM25+向量融合）
        if not survivors and terms:
            survivors = await _attempt_hybrid_fallback(
                stream_id, terms, raw_records
            )
        # 仍然无结果
        if not survivors:
            return _build_no_match_message(terms, participant, window)
        survivors = _rank_matching_records(survivors, terms, participant)
        # 结果过多：返回关键词提示
        label = keyword or participant or "当前条件"
        if len(survivors) > _OVERFLOW_THRESHOLD:
            return _build_overflow_hint(survivors, label)
        # 逐条格式化
        display_cap = limit if limit and limit > 0 else _DISPLAY_CAP
        brief_blocks = [
            _compose_record_brief(rec) for rec in survivors[:display_cap]
        ]
        result_text = (
            "\n\n---\n\n".join(brief_blocks)
            if brief_blocks
            else "未找到相关聊天记录"
        )
        # 写入缓存 + 遥测
        try:
            from src.memory_system.memory_search import (
                get_search_cache,
                get_search_analytics,
            )

            get_search_cache().stash(cache_fingerprint, result_text)
            elapsed_ms = (time.time() - probe_clock) * 1000
            get_search_analytics().record_query(
                keyword or participant or "", len(survivors), elapsed_ms
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return result_text
    except Exception as exc:
        logger.error(f"历史检索探针异常: {exc}")
        return f"查询失败: {exc}"


# ==================== 详情展示函数 ====================


async def fetch_history_detail(stream_id: str, record_ids: str) -> str:
    """按记忆ID展示ChatHistory的完整详情。

    参数:
        stream_id: 聊天流ID
        record_ids: 记忆ID（单个如"123"，多个用逗号分隔如"1,2,3"）
    返回:
        详情文本
    """
    try:
        # 解析ID列表
        try:
            id_values = [
                int(seg.strip())
                for seg in record_ids.split(",")
                if seg.strip()
            ]
        except ValueError:
            return f"无效的记忆ID格式: {record_ids}（请使用数字ID，多个用逗号分隔）"
        if not id_values:
            return "未提供有效的记忆ID"
        # 查询
        rows = await asyncio.to_thread(
            lambda: list(
                ChatHistory.select()
                .where(
                    (ChatHistory.chat_id == stream_id)
                    & (ChatHistory.id.in_(id_values))
                )
                .order_by(ChatHistory.start_time.desc())
            )
        )
        if not rows:
            return (
                f"未找到ID为{id_values}的记忆（可能不存在或不属于当前聊天流）"
            )
        # 批量更新访问计数
        try:
            await asyncio.to_thread(
                lambda: ChatHistory.update(count=ChatHistory.count + 1)
                .where(ChatHistory.id.in_([row.id for row in rows]))
                .execute()
            )
        except Exception as ue:
            logger.debug(f"批量更新访问计数失败: {ue}")
        # 格式化详情
        detail_blocks = [_compose_record_detail(row) for row in rows]
        return "\n\n" + ("=" * 50) + "\n\n".join(detail_blocks)
    except Exception as exc:
        logger.error(f"获取记忆详情失败: {exc}")
        return f"查询失败: {exc}"


# ==================== 工具注册 ====================


def register_tool():
    """向全局工具注册器注册聊天历史检索工具"""

    async def execute_search_chat_history(
        keyword: Optional[str] = None,
        participant: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        stream_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        scoped_stream_id = (stream_id or chat_id or "").strip()
        if not scoped_stream_id:
            return "查询失败: 缺少聊天流标识"
        return await probe_chat_history(
            stream_id=scoped_stream_id,
            keyword=keyword,
            participant=participant,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    register_memory_retrieval_tool(
        name="search_chat_history",
        description=(
            "按关键词或参与人查询聊天记忆概况，返回记忆ID/主题/关键词。"
            "关键词匹配规则：<=2个时全部匹配，>2个时允许漏掉1个（容错）。"
            "支持按时间点或时间段过滤。"
        ),
        parameters=[
            {
                "name": "stream_id",
                "type": "string",
                "description": "聊天流ID（通常由系统自动注入）",
                "required": False,
            },
            {
                "name": "keyword",
                "type": "string",
                "description": (
                    "关键词（可选，支持多个，空格/逗号/斜杠分隔）。"
                    "在主题、关键词、概括、原文中搜索。"
                ),
                "required": False,
            },
            {
                "name": "participant",
                "type": "string",
                "description": "参与人昵称（可选，模糊匹配）",
                "required": False,
            },
            {
                "name": "start_time",
                "type": "string",
                "description": (
                    "起始时间（可选），如'2025-01-01'或'2025-01-01 12:00:00'。"
                    "单独提供时查询该时间点之后的记录。"
                ),
                "required": False,
            },
            {
                "name": "end_time",
                "type": "string",
                "description": (
                    "截止时间（可选），如'2025-01-01'或'2025-01-01 12:00:00'。"
                    "单独提供时查询该时间点之前的记录。"
                ),
                "required": False,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "返回结果的最大条数（可选，默认20）",
                "required": False,
            },
        ],
        execute_func=execute_search_chat_history,
    )
    register_memory_retrieval_tool(
        name="get_chat_history_detail",
        description=(
            "按记忆ID展示详情（主题/时间/参与人/关键词/概括等），"
            "需先用search_chat_history获取ID。"
        ),
        parameters=[
            {
                "name": "memory_ids",
                "type": "string",
                "description": "记忆ID，单个如'123'，多个用逗号分隔如'123,456,789'",
                "required": True,
            },
        ],
        execute_func=fetch_history_detail,
    )

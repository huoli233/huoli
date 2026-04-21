import time
import json
import hashlib
import asyncio
import os
import re
from collections import OrderedDict
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from src.common.logger import get_logger
from src.common.task_utils import safe_create_task
from src.common.singleton import _get_class_lock

if TYPE_CHECKING:
    from src.common.database.database_model import MemoryRecord

logger = get_logger("回忆中心")


def _normalize_memory_search_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or ""), flags=re.UNICODE).lower()


def _score_memory_query_match(
    content: str,
    keyword: Optional[str] = None,
    keyword_terms: Optional[List[str]] = None,
) -> float:
    """评估记忆内容与查询的相关性，避免只按 significance 排序。"""
    compact_content = _normalize_memory_search_text(content)
    if not compact_content:
        return 0.0

    terms = [
        _normalize_memory_search_text(term)
        for term in (keyword_terms or [])
        if _normalize_memory_search_text(term)
    ]
    normalized_keyword = _normalize_memory_search_text(keyword or "")
    if not terms and normalized_keyword:
        terms = [normalized_keyword]

    query_compact = normalized_keyword or "".join(terms)
    if not terms and not query_compact:
        return 0.0

    phrase_hit = bool(query_compact and query_compact in compact_content)
    hits = 0
    for term in terms:
        if term in compact_content:
            hits += 1

    score = 0.0
    if phrase_hit:
        score += 2.8

    if terms:
        coverage = hits / max(1, len(terms))
        if len(terms) >= 2:
            if hits >= len(terms):
                score += 1.9
            elif hits >= 2:
                score += 1.35 * coverage
            elif hits == 1:
                score += 0.12
        elif hits >= 1:
            score += 0.42
    elif phrase_hit:
        score += 0.2

    if query_compact:
        overlap = len(set(query_compact) & set(compact_content)) / max(1, len(set(query_compact)))
        score += overlap * 0.35

    return score


class RecollectionHub:
    """记忆核心引擎 —— 统管记忆条目的创建、检索、维护与生命周期管理。

    职责：
    - 记忆CRUD（存入、查询、删除、清除）
    - 内容指纹去重
    - 情感权重动态调节
    - 海马体缓冲区协同（如已部署）
    - 定时维护（缓存清理、老旧压缩、去重、深度优化、过载检查）
    - 会话流惰性校准
    - 健康报告与统计
    """

    def __init__(self):
        self._entry_cache: OrderedDict = OrderedDict()
        self._cache_max_size: int = 10000  # 最大缓存条目数
        self._dirty_entries: set = set()
        # 各项维护任务上次执行时间戳
        self._ts_cache_purge = time.time()
        self._ts_condense = time.time()
        self._ts_dedup = time.time()
        self._ts_deep_opt = time.time()
        self._ts_overload_scan = time.time()
        # 过载守卫实例（按需加载）
        self._overload_guardian = None
        self._forget_executor = None
        # 已完成惰性校准的会话流集合
        self._calibrated_streams: set = set()
        # 持久状态仓库（按需加载）
        self._state_repo = None

    # ==================== 缓存容量管理 ====================

    def _add_to_cache(self, key: str, value: Any) -> None:
        """将条目加入缓存，超过容量上限时自动淘汰最旧条目。"""
        if key in self._entry_cache:
            del self._entry_cache[key]
        elif len(self._entry_cache) >= self._cache_max_size:
            self._entry_cache.popitem(last=False)
        self._entry_cache[key] = value

    # ==================== 内容指纹与相似度 ====================

    def _forge_entry_id(self, stream_id: str, content: str) -> str:
        """为新记忆条目生成唯一标识（SHA-256）。"""
        seed = f"{stream_id}:{content}:{time.time()}:{os.getpid()}:{id(self)}"
        return hashlib.sha256(seed.encode()).hexdigest()[:32]

    def _compute_content_fingerprint(self, content: str) -> str:
        """将内容规范化后计算SHA256短指纹（用于快速去重判断）。"""
        normalized = " ".join(content.strip().lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _measure_text_overlap(self, text_a: str, text_b: str) -> float:
        """基于词集Jaccard系数度量两段文本的相似度。"""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        shared = len(words_a & words_b)
        total = len(words_a | words_b)
        return shared / total if total > 0 else 0.0

    # ==================== 数据库模型解析 ====================

    def _resolve_storage_model(self) -> Optional[type["MemoryRecord"]]:
        """惰性获取MemoryRecord ORM模型，避免循环导入。"""
        try:
            from src.common.database.database_model import MemoryRecord

            return MemoryRecord
        except Exception as exc:
            logger.warning(f"记忆存储模型解析失败: {exc}")
            return None

    # ==================== 重复检测 ====================

    def _detect_duplicate_entry(
        self, stream_id: str, content: str, user_id: Optional[str] = None
    ):
        """在同一会话流内检测是否已存在近似内容的记忆条目。

        检测策略1：SHA256指纹精确匹配
        检测策略2：Jaccard相似度 > 0.9 的模糊匹配
        """
        StorageModel = self._resolve_storage_model()
        if not StorageModel:
            return None
        try:
            new_fingerprint = self._compute_content_fingerprint(content)
            now = time.time()
            one_week_ago = now - 7 * 86400
            # 只在最近一周的记忆中查重
            base_query = StorageModel.select().where(
                (StorageModel.stream_id == stream_id)
                & (StorageModel.birth_ts >= one_week_ago)
            )
            if user_id:
                base_query = base_query.where(StorageModel.user_id == user_id)
            # 策略1：指纹精确匹配
            for existing in base_query:
                if (
                    self._compute_content_fingerprint(existing.content)
                    == new_fingerprint
                ):
                    return existing
            # 策略2：模糊匹配
            for existing in base_query:
                if self._measure_text_overlap(content, existing.content) > 0.9:
                    return existing
            return None
        except Exception as exc:
            logger.debug(f"重复检测异常: {exc}")
            return None

    # ==================== 记忆存入 ====================

    def deposit_memory(
        self,
        stream_id: str,
        content: str,
        entry_category: str = "conversation",
        user_id: Optional[str] = None,
        significance: float = 0.5,
        digest: str = "",
        extra_data: Optional[Dict] = None,
    ) -> str:
        """存入一条新记忆。自动执行情感调权、去重检测和海马体同步。

        返回记忆的唯一ID（失败时返回空字符串）。
        """
        StorageModel = self._resolve_storage_model()
        if not StorageModel:
            return ""
        now = time.time()
        # 情感权重调节：从情绪追踪器获取用户状态，动态提升重要性
        adjusted_significance = self._apply_emotion_weight(
            stream_id, user_id, significance
        )
        try:
            # 去重：若已存在近似条目，刷新其访问信息并返回
            existing = self._detect_duplicate_entry(
                stream_id, content, user_id
            )
            if existing:
                existing.last_visit_ts = now
                existing.visit_count += 1
                if adjusted_significance > existing.significance:
                    existing.significance = min(
                        1.0,
                        (existing.significance + adjusted_significance) / 2,
                    )
                existing.save()
                self._add_to_cache(existing.record_id, existing)
                return existing.record_id
            # 生成唯一ID
            entry_id = self._forge_entry_id(stream_id, content)
            # 若海马体缓冲区可用，同步写入
            entry_id = self._sync_to_hippocampus(
                stream_id, content, user_id, adjusted_significance, entry_id
            )
            # 写入长期存储
            auto_digest = digest or (
                content[:80] + "..." if len(content) > 80 else content
            )
            record = StorageModel.create(
                record_id=entry_id,
                stream_id=stream_id,
                user_id=user_id,
                entry_category=entry_category,
                content=content,
                significance=adjusted_significance,
                visit_count=0,
                birth_ts=now,
                last_visit_ts=now,
                storage_tier=0,
                sharpness=1.0,
                digest=auto_digest,
                extra_json=(
                    json.dumps(extra_data, ensure_ascii=False)
                    if extra_data
                    else None
                ),
            )
            self._add_to_cache(entry_id, record)
            logger.debug(
                f"[记忆核心] 新条目已存入: {entry_id[:8]}... 类型={entry_category}"
            )
            return entry_id
        except Exception as exc:
            logger.error(f"存入记忆失败: {exc}")
            return ""

    def _apply_emotion_weight(
        self, stream_id: str, user_id: Optional[str], base_significance: float
    ) -> float:
        """依据用户情感状态动态调节记忆重要性。"""
        if not user_id or not stream_id:
            return base_significance
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            tracker = get_emotion_tracker()
            if not tracker:
                return base_significance
            user_state = tracker.get_user_state(
                user_id, create_if_missing=False
            )
            if not user_state:
                return base_significance
            # 强负面体验 → 大幅提升重要性
            if (
                getattr(user_state, "affection", 0) < -50
                or getattr(user_state, "trauma_score", 0) > 5
            ):
                return min(1.0, base_significance + 0.3)
            # 中度消极 → 小幅提升
            if (
                getattr(user_state, "affection", 0) < -20
                or getattr(user_state, "annoyance", 0) > 50
            ):
                return min(1.0, base_significance + 0.15)
            # 高好感度 → 适度提升
            if getattr(user_state, "affection", 0) > 50:
                return min(1.0, base_significance + 0.2)
        except Exception as _e:
            logger.warning(f"记忆重要性评估异常: {_e}")
        return base_significance

    def _sync_to_hippocampus(
        self,
        stream_id: str,
        content: str,
        user_id: Optional[str],
        significance: float,
        fallback_id: str,
    ) -> str:
        """将记忆同步写入海马体缓冲区（如果已部署），返回缓冲区分配的ID。"""
        try:
            from src.memory_system.hippocampus_buffer import (
                get_hippocampus_buffer,
            )

            hippo = get_hippocampus_buffer(stream_id)
            return hippo.add_memory(
                content, user_id=user_id, importance=significance
            )
        except ImportError:
            return fallback_id
        except Exception as exc:
            logger.debug(f"海马体同步跳过: {exc}")
            return fallback_id

    # ==================== 记忆查询 ====================

    def fetch_memory(self, record_id: str):
        """按ID获取单条记忆，自动更新访问计数。"""
        StorageModel = self._resolve_storage_model()
        if not StorageModel:
            return None
        # 缓存优先
        if record_id in self._entry_cache:
            entry = self._entry_cache[record_id]
        else:
            try:
                entry = StorageModel.get(StorageModel.record_id == record_id)
                self._add_to_cache(record_id, entry)
            except StorageModel.DoesNotExist:
                logger.debug(f"记忆条目不存在，跳过访问计数: {record_id[:8]}...")
                return None
            except Exception as exc:
                logger.warning(f"记忆条目访问查询失败 ({record_id[:8]}...): {exc}")
                return None
        entry.visit_count += 1
        entry.last_visit_ts = time.time()
        self._dirty_entries.add(record_id)
        return entry

    @staticmethod
    def _entry_identifier(entry: Any) -> str:
        """提取条目标识，兼容长期存储与海马体碎片对象。"""
        return str(
            getattr(entry, "record_id", "")
            or getattr(entry, "fragment_id", "")
            or getattr(entry, "memory_id", "")
            or ""
        ).strip()

    def _mark_entry_accessed(self, entry: Any) -> None:
        """统一更新访问热度，兼容 ORM 记录与海马体碎片。"""
        now = time.time()
        entry_id = self._entry_identifier(entry)
        if hasattr(entry, "visit_count"):
            try:
                entry.visit_count = int(getattr(entry, "visit_count", 0) or 0) + 1
            except Exception:
                entry.visit_count = 1
            if hasattr(entry, "last_visit_ts"):
                entry.last_visit_ts = now
            if entry_id and hasattr(entry, "save"):
                self._add_to_cache(entry_id, entry)
                self._dirty_entries.add(entry_id)
            return
        if hasattr(entry, "touch_count"):
            try:
                entry.touch_count = int(getattr(entry, "touch_count", 0) or 0) + 1
            except Exception:
                entry.touch_count = 1
            if hasattr(entry, "touched_at"):
                entry.touched_at = now
            return

    def query_memories(
        self,
        stream_id: str,
        keyword: Optional[str] = None,
        keyword_terms: Optional[List[str]] = None,
        entry_category: Optional[str] = None,
        ceiling: int = 10,
        user_id: Optional[str] = None,
    ) -> List:
        """分层搜索：先搜海马体缓冲区，再搜长期存储，按重要性×清晰度排序。

        检索接口关系说明：
        - RecollectionHub.query_memories(stream_id, query, ...): 本方法，长期存储+海马体分层检索
        - HippoMemoryBuffer.probe(query, ceiling): 海马体缓冲区关键词检索（本方法内部调用）
        - HippoMemoryBuffer.search_memories(query, limit): probe 的兼容别名
        - MemoryRetriever.probe(question, stream_id): 记忆检索器入口（见 memory_retrieval.py）
        """
        StorageModel = self._resolve_storage_model()
        if not StorageModel:
            return []
        found = []
        try:
            # 层1：海马体缓冲区（如果可用）
            normalized_terms = [
                str(term or "").strip()
                for term in (keyword_terms or [])
                if str(term or "").strip()
            ]
            normalized_keyword = str(keyword or "").strip()
            _hippo_keyword = (normalized_terms[0] if normalized_terms else None) or normalized_keyword
            found = self._search_hippocampus(stream_id, _hippo_keyword, ceiling)
            # 层2：长期存储(数据库)
            q = StorageModel.select().where(
                StorageModel.stream_id == stream_id
            )
            if entry_category:
                q = q.where(StorageModel.entry_category == entry_category)
            if user_id:
                q = q.where(StorageModel.user_id == user_id)
            if normalized_terms:
                _term_expr = None
                for term in normalized_terms[:6]:
                    _expr = StorageModel.content.contains(term)
                    _term_expr = _expr if _term_expr is None else (_term_expr | _expr)
                if _term_expr is not None:
                    q = q.where(_term_expr)
            elif normalized_keyword:
                q = q.where(StorageModel.content.contains(normalized_keyword))
            # 排除海马体已返回的
            if found:
                existing_ids = [
                    entry_id
                    for entry_id in (self._entry_identifier(r) for r in found)
                    if entry_id
                ]
                if existing_ids:
                    q = q.where(StorageModel.record_id.not_in(existing_ids))
            q = q.order_by(
                StorageModel.significance.desc(), StorageModel.birth_ts.desc()
            )
            fetch_ceiling = max(ceiling - len(found), 0)
            if fetch_ceiling <= 0:
                for entry in found[:ceiling]:
                    self._mark_entry_accessed(entry)
                return found[:ceiling]
            if normalized_terms:
                if len(normalized_terms) >= 3:
                    db_fetch_limit = min(20, max(fetch_ceiling + 6, 10))
                elif len(normalized_terms) == 2:
                    db_fetch_limit = min(18, max(fetch_ceiling + 5, 9))
                else:
                    db_fetch_limit = min(16, max(fetch_ceiling + 4, 8))
            elif normalized_keyword:
                db_fetch_limit = min(14, max(fetch_ceiling + 4, 8))
            else:
                db_fetch_limit = min(12, max(fetch_ceiling + 3, 6))
            db_candidates = list(q.limit(db_fetch_limit))
            # 按「重要性×清晰度」综合评分排序
            stage_one_scored = []
            stage_two_scored = []
            compact_query = _normalize_memory_search_text(normalized_keyword or "".join(normalized_terms))
            for entry in db_candidates:
                content_text = getattr(entry, "content", "")
                compact_content = _normalize_memory_search_text(content_text)
                if not compact_content:
                    continue
                sharpness = self._estimate_sharpness(
                    entry.birth_ts, entry.significance, entry.visit_count
                )
                match_score = _score_memory_query_match(
                    content_text,
                    keyword=normalized_keyword,
                    keyword_terms=normalized_terms,
                )
                term_hits = sum(
                    1 for term in normalized_terms if _normalize_memory_search_text(term) in compact_content
                )
                term_coverage = term_hits / max(1, len(normalized_terms)) if normalized_terms else 0.0
                phrase_hit = bool(compact_query and compact_query in compact_content)
                high_overlap = match_score >= 0.9
                stage_one_hit = False
                stage_two_hit = False
                if normalized_terms:
                    if len(normalized_terms) >= 3:
                        stage_one_hit = phrase_hit or term_hits >= 2 or term_coverage >= 0.66 or high_overlap
                        stage_two_hit = term_hits >= 1 and match_score >= 0.5
                    elif len(normalized_terms) == 2:
                        stage_one_hit = phrase_hit or term_hits >= 2 or high_overlap
                        stage_two_hit = term_hits >= 1 and match_score >= 0.52
                    else:
                        stage_one_hit = phrase_hit or (term_hits >= 1 and match_score >= 0.75)
                        stage_two_hit = match_score >= 0.5
                elif normalized_keyword:
                    stage_one_hit = phrase_hit or match_score >= 0.78
                    stage_two_hit = match_score >= 0.45
                else:
                    stage_one_hit = match_score >= 0.55
                    stage_two_hit = match_score >= 0.35

                if not stage_one_hit and not stage_two_hit:
                    continue

                composite = (
                    entry.significance * 0.2
                    + sharpness * 0.2
                    + match_score * 0.6
                )
                if stage_one_hit:
                    stage_one_scored.append((composite, entry))
                elif stage_two_hit:
                    stage_two_scored.append((composite * 0.82, entry))

            stage_one_scored.sort(
                key=lambda pair: (
                    pair[0],
                    float(getattr(pair[1], "birth_ts", 0.0) or 0.0),
                ),
                reverse=True,
            )
            stage_two_scored.sort(
                key=lambda pair: (
                    pair[0],
                    float(getattr(pair[1], "birth_ts", 0.0) or 0.0),
                ),
                reverse=True,
            )
            ordered_db_entries = [entry for _, entry in stage_one_scored]
            if len(ordered_db_entries) < fetch_ceiling:
                ordered_db_entries.extend(
                    entry for _, entry in stage_two_scored[: max(0, fetch_ceiling - len(ordered_db_entries))]
                )
            found.extend(ordered_db_entries)
            # 更新访问热度
            for entry in found[:ceiling]:
                self._mark_entry_accessed(entry)
            return found[:ceiling]
        except Exception as exc:
            logger.error(f"查询记忆失败: {exc}")
            return []

    def flush_dirty_entries(self):
        """批量保存脏条目，减少写放大"""
        if not self._dirty_entries:
            return
        StorageModel = self._resolve_storage_model()
        if not StorageModel:
            self._dirty_entries.clear()
            return
        try:
            for record_id in list(self._dirty_entries):
                entry = self._entry_cache.get(record_id)
                if entry:
                    entry.save()
            logger.debug(f"刷盘完成: 保存 {len(self._dirty_entries)} 条脏数据")
            self._dirty_entries.clear()
        except Exception as exc:
            logger.error(f"批量保存脏条目失败: {exc}")
            self._dirty_entries.clear()

    def flush(self):
        """持久化刷盘入口，将所有脏数据写入持久层。

        供外部生命周期管理调用，确保进程退出前数据不丢失。
        """
        self.flush_dirty_entries()
        # 同时将所有校准过的会话流状态快照持久化
        for stream_id in list(self._calibrated_streams):
            self._persist_state_snapshot(stream_id)

    def shutdown(self):
        """关闭记忆核心引擎，刷盘所有脏数据并清理缓存。"""
        logger.info("RecollectionHub 关闭中...")
        self.flush()
        self._entry_cache.clear()
        self._calibrated_streams.clear()
        self._overload_guardian = None
        self._forget_executor = None
        self._state_repo = None
        logger.info("RecollectionHub 已关闭")

    def format_query_for_prompt(
        self,
        stream_id: str,
        keyword: Optional[str] = None,
        ceiling: int = 10,
        mode: Optional[str] = None,
    ) -> str:
        """查询记忆并直接生成可嵌入Prompt的格式化文本"""
        raw_records = self.query_memories(
            stream_id=stream_id, keyword=keyword, ceiling=ceiling
        )
        if not raw_records:
            return ""
        try:
            from src.memory_system.output_formatter import get_output_formatter
            from src.memory_system.memory_models import BufferedMemory
            _buf_items = []
            for rec in raw_records:
                # 字段映射：MemoryRecord.significance → BufferedMemory.importance
                # 两者语义等价，数据库用 significance，内存模型用 importance
                _buf_items.append(BufferedMemory(
                    memory_id=str(getattr(rec, "record_id", "")),
                    content=str(getattr(rec, "content", "") or ""),
                    created_at=float(getattr(rec, "birth_ts", 0) or 0),
                    importance=float(getattr(rec, "significance", 0.5) or 0.5),
                    access_count=int(getattr(rec, "visit_count", 0) or 0),
                    last_accessed=float(getattr(rec, "last_visit_ts", 0) or 0),
                    channel_id=str(getattr(rec, "stream_id", "") or ""),
                    user_id=str(getattr(rec, "user_id", "") or ""),
                ))
            _fmt = get_output_formatter()
            return _fmt.format_sync(_buf_items, mode=mode)
        except ImportError:
            # 降级：简单拼接
            lines = []
            for rec in raw_records[:ceiling]:
                _c = str(getattr(rec, "content", "") or "")[:120]
                lines.append(f"- {_c}")
            return "\n".join(lines)

    def _search_hippocampus(
        self, stream_id: str, keyword: Optional[str], ceiling: int
    ) -> List:
        """尝试从海马体缓冲区搜索，未部署则返回空列表。"""
        if not keyword:
            return []
        try:
            from src.memory_system.hippocampus_buffer import (
                get_hippocampus_buffer,
            )

            hippo = get_hippocampus_buffer(stream_id)
            hippo_hits = hippo.search_memories(keyword, limit=ceiling // 2)
            results = []
            for frag, _score in hippo_hits:
                entry = self.fetch_memory(frag.fragment_id)
                if entry:
                    results.append(entry)
                else:
                    results.append(frag)
            return results
        except ImportError:
            return []
        except Exception as exc:
            logger.debug(f"海马体搜索跳过: {exc}")
            return []

    @staticmethod
    def _estimate_sharpness(
        birth_ts: float, significance: float, visit_count: int
    ) -> float:
        """基于年龄、重要性和访问频次估算记忆清晰度（0~1）。

        时间格式注意：birth_ts 为 Unix秒（与 MemoryRecord.birth_ts / HippocampusEntry.birth_ts 一致），
        可直接与 time.time() 做差值计算。
        """
        age_days = (time.time() - birth_ts) / 86400
        # 基础衰减：每天衰减0.5%
        base = max(0.0, 1.0 - age_days * 0.005)
        # 访问频次加成
        access_bonus = min(0.3, visit_count * 0.03)
        # 重要性加成
        significance_bonus = significance * 0.1
        return min(
            1.0, base * significance + access_bonus + significance_bonus
        )

    def list_recent_entries(self, stream_id: str, ceiling: int = 20) -> List:
        """获取指定会话流最近的记忆条目，按创建时间倒序。"""
        StorageModel = self._resolve_storage_model()
        if not StorageModel:
            return []
        try:
            return list(
                StorageModel.select()
                .where(StorageModel.stream_id == stream_id)
                .order_by(StorageModel.birth_ts.desc())
                .limit(ceiling)
            )
        except Exception as exc:
            logger.error(f"获取近期记忆失败: {exc}")
            return []

    # ==================== 会话流校准 ====================

    def _obtain_state_repository(self):
        """惰性加载持久状态仓库。"""
        if self._state_repo is None:
            try:
                from src.chat.heart_flow.persistent_state_db import (
                    acquire_persistent_store,
                )

                self._state_repo = acquire_persistent_store()
            except Exception as exc:
                logger.debug(f"状态仓库加载失败: {exc}")
        return self._state_repo

    def _lazy_calibrate_stream(self, stream_id: str) -> bool:
        """对指定会话流执行惰性校准（首次访问时触发，后续跳过）。"""
        if not stream_id or stream_id in self._calibrated_streams:
            return stream_id in self._calibrated_streams
        try:
            repo = self._obtain_state_repository()
            if repo:
                calibrated, forgotten = self._recalibrate_stream_entries(
                    stream_id, repo
                )
                self._calibrated_streams.add(stream_id)
                if calibrated > 0 or forgotten > 0:
                    logger.debug(
                        f"惰性校准完成: stream={stream_id[:8]}... "
                        f"校准={calibrated} 遗忘={forgotten}"
                    )
                return True
        except Exception as exc:
            logger.debug(f"惰性校准异常: {exc}")
        self._calibrated_streams.add(stream_id)
        return False

    def _recalibrate_stream_entries(self, stream_id: str, state_repo) -> tuple:
        """根据离线时长重新校准会话流内所有记忆的清晰度，过于模糊的予以遗忘。"""
        try:
            StorageModel = self._resolve_storage_model()
            if not StorageModel:
                return 0, 0
            memory_state = (
                state_repo.load_memory_state(stream_id)
                if hasattr(state_repo, "load_memory_state")
                else None
            )
            offline_duration = (
                memory_state.get("offline_duration", 0.0)
                if memory_state
                else 0.0
            )
            if offline_duration <= 0:
                return 0, 0
            entries = StorageModel.select().where(
                StorageModel.stream_id == stream_id
            )
            forgotten_count = 0
            calibrated_count = 0
            now = time.time()
            for entry in entries:
                sharpness = self._estimate_sharpness(
                    entry.birth_ts, entry.significance, entry.visit_count
                )
                calibrated_count += 1
                # 清晰度极低 + 年龄超半年 + 低访问 + 低重要性 → 遗忘
                if sharpness < 0.1:
                    age_days = (now - entry.birth_ts) / 86400
                    if (
                        age_days >= 180
                        and entry.visit_count < 3
                        and entry.significance < 0.7
                    ):
                        entry.delete_instance()
                        self._entry_cache.pop(entry.record_id, None)
                        forgotten_count += 1
            if hasattr(state_repo, "save_memory_state"):
                state_repo.save_memory_state(stream_id)
            return calibrated_count, forgotten_count
        except Exception as exc:
            logger.debug(f"校准会话流失败: {exc}")
            return 0, 0

    def _persist_state_snapshot(self, stream_id: str):
        """将记忆状态快照持久化。"""
        try:
            repo = self._obtain_state_repository()
            if repo and hasattr(repo, "save_memory_state"):
                repo.save_memory_state(stream_id)
        except Exception as exc:
            logger.debug(f"状态快照持久化失败: {exc}")

    # ==================== 定时维护 ====================

    def execute_routine_maintenance(self, stream_id: Optional[str] = None):
        """执行例行维护任务（按各自独立节奏触发）。"""
        now = time.time()
        # 缓存清理：每30分钟
        if now - self._ts_cache_purge > 1800:
            self._purge_stale_cache()
            self._ts_cache_purge = now
        # 老旧压缩：每2小时
        if now - self._ts_condense > 7200:
            condensed = self._condense_aged_entries(stream_id)
            if condensed > 0:
                logger.info(f"压缩了 {condensed} 条老旧记忆")
            self._ts_condense = now
        # 去重：每6小时
        if now - self._ts_dedup > 21600:
            removed = self._eliminate_duplicates(stream_id)
            if removed > 0:
                logger.info(f"去重了 {removed} 条重复记忆")
            self._ts_dedup = now
        # 深度优化：每24小时
        if now - self._ts_deep_opt > 86400:
            self._request_deep_optimization(stream_id)
            self._ts_deep_opt = now
        # 过载检查：每10分钟
        if now - self._ts_overload_scan > 600:
            self._engage_overload_guardian(stream_id)
            self._ts_overload_scan = now

    def _purge_stale_cache(self):
        """清除缓存中超过1小时未访问的条目。"""
        now = time.time()
        stale_keys = [
            rid
            for rid, entry in self._entry_cache.items()
            if now - getattr(entry, "last_visit_ts", now) > 3600
        ]
        for key in stale_keys:
            del self._entry_cache[key]
        if stale_keys:
            logger.debug(f"缓存清理: 移除 {len(stale_keys)} 条过期条目")

    def _condense_aged_entries(self, stream_id: Optional[str] = None) -> int:
        """将超过30天的对话类记忆标记为已压缩类型。"""
        StorageModel = self._resolve_storage_model()
        if not StorageModel:
            return 0
        try:
            cutoff = time.time() - 30 * 86400
            q = StorageModel.select().where(
                (StorageModel.birth_ts < cutoff)
                & (StorageModel.entry_category == "conversation")
            )
            if stream_id:
                q = q.where(StorageModel.stream_id == stream_id)
            condensed = 0
            for entry in q:
                if len(entry.content) > 200:
                    entry.entry_category = "compressed"
                    entry.save()
                    condensed += 1
            return condensed
        except Exception as exc:
            logger.error(f"老旧压缩失败: {exc}")
            return 0

    def _eliminate_duplicates(self, stream_id: Optional[str] = None) -> int:
        """基于内容指纹去除最近一周内的重复记忆。"""
        StorageModel = self._resolve_storage_model()
        if not StorageModel:
            return 0
        try:
            week_ago = time.time() - 7 * 86400
            q = (
                StorageModel.select()
                .where(StorageModel.birth_ts >= week_ago)
                .order_by(StorageModel.birth_ts.desc())
            )
            if stream_id:
                q = q.where(StorageModel.stream_id == stream_id)
            entries = list(q)
            if len(entries) < 2:
                return 0
            seen_fingerprints = set()
            removed = 0
            for entry in entries:
                fp = self._compute_content_fingerprint(entry.content)
                if fp in seen_fingerprints:
                    entry.delete_instance()
                    self._entry_cache.pop(entry.record_id, None)
                    removed += 1
                else:
                    seen_fingerprints.add(fp)
            return removed
        except Exception as exc:
            logger.error(f"去重处理失败: {exc}")
            return 0

    # ==================== deletion ====================

    def remove_memory(self, record_id: str) -> bool:
        """删除指定记忆条目。"""
        StorageModel = self._resolve_storage_model()
        if not StorageModel:
            return False
        try:
            StorageModel.delete().where(
                StorageModel.record_id == record_id
            ).execute()
            self._entry_cache.pop(record_id, None)
            return True
        except Exception as exc:
            logger.error(f"删除记忆失败: {exc}")
            return False

    def wipe_stream_entries(self, stream_id: str) -> int:
        """清除指定会话流的全部记忆。"""
        StorageModel = self._resolve_storage_model()
        if not StorageModel:
            return 0
        try:
            deleted = (
                StorageModel.delete()
                .where(StorageModel.stream_id == stream_id)
                .execute()
            )
            stale = [
                k
                for k, v in self._entry_cache.items()
                if getattr(v, "stream_id", "") == stream_id
            ]
            for k in stale:
                del self._entry_cache[k]
            self._calibrated_streams.discard(stream_id)
            logger.info(
                f"清除会话流记忆: stream={stream_id[:8]}... 删除{deleted}条"
            )
            return deleted
        except Exception as exc:
            logger.error(f"清除会话流记忆失败: {exc}")
            return 0

    # ==================== 统计与健康 ====================

    def compile_statistics(self, stream_id: str) -> Dict[str, Any]:
        """编译指定会话流的记忆统计摘要。"""
        StorageModel = self._resolve_storage_model()
        if not StorageModel:
            return {"total": 0, "by_category": {}}
        try:
            total = (
                StorageModel.select()
                .where(StorageModel.stream_id == stream_id)
                .count()
            )
            by_cat = {}
            for cat in (
                "conversation",
                "knowledge",
                "user",
                "system",
                "compressed",
                "fact",
                "trauma",
            ):
                count = (
                    StorageModel.select()
                    .where(
                        (StorageModel.stream_id == stream_id)
                        & (StorageModel.entry_category == cat)
                    )
                    .count()
                )
                if count > 0:
                    by_cat[cat] = count
            return {
                "total": total,
                "by_category": by_cat,
                "cache_size": len(self._entry_cache),
            }
        except Exception as exc:
            logger.error(f"统计编译失败: {exc}")
            return {"total": 0, "by_category": {}}

    def compile_detailed_statistics(
        self, stream_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """编译详尽的多维统计信息。"""
        StorageModel = self._resolve_storage_model()
        if not StorageModel:
            return {"total": 0}
        try:
            q = StorageModel.select()
            if stream_id:
                q = q.where(StorageModel.stream_id == stream_id)
            entries = list(q)
            if not entries:
                return {"total": 0}
            now = time.time()
            stats = {
                "total": len(entries),
                "by_category": {},
                "by_age": {"recent": 0, "medium": 0, "old": 0},
                "cache_size": len(self._entry_cache),
                "avg_significance": (
                    sum(e.significance for e in entries) / len(entries)
                    if entries
                    else 0.0
                ),
                "total_visits": sum(e.visit_count for e in entries),
            }
            for entry in entries:
                cat = entry.entry_category
                stats["by_category"][cat] = (
                    stats["by_category"].get(cat, 0) + 1
                )
                age_days = (now - entry.birth_ts) / 86400
                if age_days <= 7:
                    stats["by_age"]["recent"] += 1
                elif age_days <= 30:
                    stats["by_age"]["medium"] += 1
                else:
                    stats["by_age"]["old"] += 1
            return stats
        except Exception as exc:
            logger.error(f"详尽统计编译失败: {exc}")
            return {"total": 0, "error": str(exc)}

    def generate_health_assessment(
        self, stream_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """生成记忆系统健康评估报告（0~100分）。"""
        try:
            stats = self.compile_detailed_statistics(stream_id)
            total = stats.get("total", 0)
            if total == 0:
                return {
                    "health_score": 100,
                    "health_grade": "优秀",
                    "total": 0,
                    "recommendations": [],
                }
            score = 100
            recommendations = []
            # 扣分：老旧记忆占比过高
            age_dist = stats.get("by_age", {})
            old_ratio = age_dist.get("old", 0) / max(1, total)
            if old_ratio > 0.5:
                score -= 15
                recommendations.append("老旧记忆占比过高，可清理无用数据")
            # 扣分：平均重要性过低
            avg_sig = stats.get("avg_significance", 0.5)
            if avg_sig < 0.3:
                score -= 15
                recommendations.append(
                    "平均重要性偏低，可能存在大量低质量记忆"
                )
            # 扣分：总量过大
            if total > 5000:
                score -= 20
                recommendations.append("记忆总量过大，建议执行深度优化")
            elif total > 2000:
                score -= 10
            # 扣分：已压缩比例过高
            compressed = stats.get("by_category", {}).get("compressed", 0)
            if compressed > total * 0.3:
                score -= 10
            score = max(0, min(100, score))
            if score >= 90:
                grade = "优秀"
            elif score >= 70:
                grade = "良好"
            elif score >= 50:
                grade = "一般"
            else:
                grade = "需要优化"
            return {
                "health_score": score,
                "health_grade": grade,
                "total": total,
                "by_category": stats.get("by_category", {}),
                "by_age": age_dist,
                "avg_significance": round(avg_sig, 3),
                "cache_size": stats.get("cache_size", 0),
                "recommendations": recommendations,
            }
        except Exception as exc:
            logger.error(f"健康评估生成失败: {exc}")
            return {"error": str(exc)}

    # ==================== 过载与深度优化 ====================

    def _engage_overload_guardian(self, stream_id: Optional[str] = None):
        """检查记忆是否过载，过载时触发紧急遗忘。"""
        try:
            if self._overload_guardian is None:
                from src.memory_system.memory_overload_system import (
                    get_memory_overload_protector,
                    acquire_forget_executor,
                )

                self._overload_guardian = get_memory_overload_protector(
                    stream_id or "default"
                )
                self._forget_executor = acquire_forget_executor(
                    stream_id or "default"
                )
            self._overload_guardian.scan_brain_load()
            if self._overload_guardian.requires_emergency_purge():
                logger.warning("记忆过载，触发紧急遗忘...")
                result = self._forget_executor.execute_purge(
                    target_reduction=0.25
                )
                logger.info(
                    f"紧急遗忘完成 | 遗忘{result.get('purged_count', 0)}条"
                )
        except ImportError:
            pass
        except Exception as exc:
            logger.error(f"过载检查失败: {exc}")

    def _request_deep_optimization(self, stream_id: Optional[str] = None):
        """委托深度优化器执行全面优化。"""
        try:
            from src.memory_system.deep_optimizer import get_deep_optimizer

            optimizer = get_deep_optimizer()
            try:
                asyncio.get_running_loop()
                safe_create_task(optimizer.run_optimization(stream_id), name="deep_optimization")
            except RuntimeError:
                safe_create_task(optimizer.run_optimization(stream_id), name="deep_optimization")
        except ImportError:
            pass
        except Exception as exc:
            logger.error(f"深度优化委托失败: {exc}")

    def trigger_deep_optimization(
        self, stream_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """手动触发深度优化。"""
        try:
            from src.memory_system.deep_optimizer import get_deep_optimizer

            optimizer = get_deep_optimizer()
            try:
                asyncio.get_running_loop()
                safe_create_task(optimizer.run_optimization(stream_id), name="deep_optimization")
            except RuntimeError:
                safe_create_task(optimizer.run_optimization(stream_id), name="deep_optimization")
            self._ts_deep_opt = time.time()
            return {"status": "deep_optimization_scheduled"}
        except ImportError:
            return {"error": "deep_optimizer_not_available"}
        except Exception as exc:
            logger.error(f"手动深度优化失败: {exc}")
            return {"error": str(exc)}

    def trigger_urgent_forgetting(
        self, stream_id: Optional[str] = None, reduction: float = 0.3
    ) -> Dict[str, Any]:
        """手动触发紧急遗忘。"""
        try:
            if self._overload_guardian is None:
                from src.memory_system.memory_overload_system import (
                    get_memory_overload_protector,
                    acquire_forget_executor,
                )

                self._overload_guardian = get_memory_overload_protector(
                    stream_id or "default"
                )
                self._forget_executor = acquire_forget_executor(
                    stream_id or "default"
                )
            if self._forget_executor is None:
                from src.memory_system.memory_overload_system import (
                    acquire_forget_executor,
                )
                self._forget_executor = acquire_forget_executor(
                    stream_id or "default"
                )
            result = self._forget_executor.execute_purge(
                reduction
            )
            logger.warning(
                f"手动紧急遗忘完成 | 遗忘{result.get('purged_count', 0)}条"
            )
            return result
        except ImportError:
            return {
                "error": "overload_system_not_available",
                "forgotten_count": 0,
            }
        except Exception as exc:
            logger.error(f"手动紧急遗忘失败: {exc}")
            return {"error": str(exc), "forgotten_count": 0}

    def retrieve_overload_status(
        self, stream_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取当前过载状态。"""
        try:
            if self._overload_guardian is None:
                from src.memory_system.memory_overload_system import (
                    get_memory_overload_protector,
                )

                self._overload_guardian = get_memory_overload_protector(
                    stream_id or "default"
                )
            snapshot = self._overload_guardian.scan_brain_load()
            return {
                "gauge_level": snapshot.gauge_level.name,
                "record_count": snapshot.record_count,
                "volume_mb": round(snapshot.volume_mb, 2),
                "load_ratio": round(snapshot.load_ratio, 3),
                "amnesia_pressure": round(snapshot.amnesia_pressure, 3),
                "emergency_needed": self._overload_guardian.requires_emergency_purge(),
            }
        except ImportError:
            return {"status": "overload_system_not_available"}
        except Exception as exc:
            logger.error(f"获取过载状态失败: {exc}")
            return {"error": str(exc)}

    def retrieve_overload_prompt(self, stream_id: Optional[str] = None) -> str:
        """获取过载行为约束提示词（用于注入LLM prompt）。"""
        try:
            if self._overload_guardian is None:
                from src.memory_system.memory_overload_system import (
                    get_memory_overload_protector,
                )

                self._overload_guardian = get_memory_overload_protector(
                    stream_id or "default"
                )
            snapshot = self._overload_guardian.scan_brain_load()
            dampening = snapshot.capability_dampening
            hints = []
            if dampening.get("reply_length", 1.0) < 0.7:
                hints.append("回复应更简短")
            if dampening.get("proactive_rate", 1.0) < 0.5:
                hints.append("减少主动发言")
            if snapshot.amnesia_pressure > 0.6:
                hints.append("记忆压力大，避免引用旧信息")
            return "；".join(hints) if hints else ""
        except ImportError:
            return ""
        except Exception as exc:
            logger.debug(f"获取过载提示词失败: {exc}")
            return ""

    # ==================== 清理与优化 ====================

    def purge_expired_entries(self) -> int:
        """清除带过期时间戳且已过期的记忆条目。"""
        StorageModel = self._resolve_storage_model()
        if not StorageModel:
            return 0
        try:
            now = time.time()
            if not hasattr(StorageModel, "expires_at"):
                return 0
            deleted = (
                StorageModel.delete()
                .where(
                    (StorageModel.expires_at.is_null(False))
                    & (StorageModel.expires_at < now)
                )
                .execute()
            )
            if deleted > 0:
                logger.info(f"清除过期记忆: {deleted}条")
                stale = []
                for rid in list(self._entry_cache.keys()):
                    try:
                        StorageModel.get(StorageModel.record_id == rid)
                    except Exception:
                        stale.append(rid)
                for k in stale:
                    del self._entry_cache[k]
            return deleted
        except Exception as exc:
            logger.error(f"清除过期记忆失败: {exc}")
            return 0

    def optimize_storage_indexes(self):
        """对底层数据库执行索引重建和空间回收。"""
        try:
            from src.common.database.database_model import db

            db.execute_sql("REINDEX")
            db.execute_sql("VACUUM")
            logger.info("数据库索引优化完成")
        except Exception as exc:
            logger.error(f"数据库优化失败: {exc}")

    # ==================== 异步便捷入口 ====================

    async def store_memory_entry(
        self,
        stream_id: str,
        content: str,
        user_id: str,
        significance: float = 0.5,
        is_bot: bool = False,
        extra_metadata: Optional[Dict] = None,
    ) -> str:
        """异步存入记忆的便捷方法，自动推断类型。"""
        category = "bot_response" if is_bot else "conversation"
        if extra_metadata and extra_metadata.get("is_harassment"):
            category = "trauma"
            significance = max(significance, 0.9)
        return self.deposit_memory(
            stream_id=stream_id,
            content=content,
            entry_category=category,
            user_id=user_id,
            significance=significance,
        )


# ==================== 单例管理 ====================

_hub_instance: Optional[RecollectionHub] = None
_hub_lock = _get_class_lock("RecollectionHub")


def acquire_recollection_hub() -> RecollectionHub:
    """获取记忆核心引擎单例（线程安全）。"""
    global _hub_instance
    if _hub_instance is None:
        with _hub_lock:
            if _hub_instance is None:
                _hub_instance = RecollectionHub()
    return _hub_instance


def get_memory_core() -> RecollectionHub:
    """获取记忆核心引擎单例（兼容性别名）。"""
    return acquire_recollection_hub()


def release_recollection_hub():
    """释放单例（用于测试或重启）。"""
    global _hub_instance
    with _hub_lock:
        _hub_instance = None

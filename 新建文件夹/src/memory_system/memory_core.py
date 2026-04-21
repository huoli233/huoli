import time
import json
import hashlib
import asyncio
from typing import List, Dict, Optional, Any
from src.common.logger import get_logger

logger = get_logger("回忆中心")


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
        self._entry_cache: Dict[str, Any] = {}
        # 各项维护任务上次执行时间戳
        self._ts_cache_purge = time.time()
        self._ts_condense = time.time()
        self._ts_dedup = time.time()
        self._ts_deep_opt = time.time()
        self._ts_overload_scan = time.time()
        # 过载守卫实例（按需加载）
        self._overload_guardian = None
        # 已完成惰性校准的会话流集合
        self._calibrated_streams: set = set()
        # 持久状态仓库（按需加载）
        self._state_repo = None

    # ==================== 内容指纹与相似度 ====================

    def _forge_entry_id(self, stream_id: str, content: str) -> str:
        """为新记忆条目生成唯一标识（MD5）。"""
        seed = f"{stream_id}:{content}:{time.time()}"
        return hashlib.md5(seed.encode()).hexdigest()

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

    def _resolve_storage_model(self):
        """惰性获取MemoryRecord ORM模型，避免循环导入。"""
        try:
            from src.common.database.database_model import MemoryRecord

            return MemoryRecord
        except Exception:
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
                self._entry_cache[existing.record_id] = existing
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
            self._entry_cache[entry_id] = record
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
            logger.debug(f"异常: {_e}")
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
                self._entry_cache[record_id] = entry
            except Exception:
                return None
        entry.visit_count += 1
        entry.last_visit_ts = time.time()
        entry.save()
        return entry

    def query_memories(
        self,
        stream_id: str,
        keyword: Optional[str] = None,
        entry_category: Optional[str] = None,
        ceiling: int = 10,
        user_id: Optional[str] = None,
    ) -> List:
        """分层搜索：先搜海马体缓冲区，再搜长期存储，按重要性×清晰度排序。"""
        StorageModel = self._resolve_storage_model()
        if not StorageModel:
            return []
        found = []
        try:
            # 层1：海马体缓冲区（如果可用）
            found = self._search_hippocampus(stream_id, keyword, ceiling)
            # 层2：长期存储(数据库)
            q = StorageModel.select().where(
                StorageModel.stream_id == stream_id
            )
            if entry_category:
                q = q.where(StorageModel.entry_category == entry_category)
            if user_id:
                q = q.where(StorageModel.user_id == user_id)
            if keyword:
                q = q.where(StorageModel.content.contains(keyword))
            # 排除海马体已返回的
            if found:
                existing_ids = [getattr(r, "record_id", "") for r in found]
                q = q.where(StorageModel.record_id.not_in(existing_ids))
            q = q.order_by(
                StorageModel.significance.desc(), StorageModel.birth_ts.desc()
            )
            db_candidates = list(q.limit(ceiling - len(found)))
            # 按「重要性×清晰度」综合评分排序
            scored = []
            for entry in db_candidates:
                sharpness = self._estimate_sharpness(
                    entry.birth_ts, entry.significance, entry.visit_count
                )
                composite = entry.significance * 0.5 + sharpness * 0.5
                scored.append((composite, entry))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            found.extend([entry for _, entry in scored])
            # 更新访问热度
            for entry in found[:ceiling]:
                entry.visit_count += 1
                entry.last_visit_ts = time.time()
                entry.save()
            return found[:ceiling]
        except Exception as exc:
            logger.error(f"查询记忆失败: {exc}")
            return []

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
            for mem, _score in hippo_hits:
                entry = self.fetch_memory(mem.memory_id)
                if entry:
                    results.append(entry)
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
        """基于年龄、重要性和访问频次估算记忆清晰度（0~1）。"""
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
                    get_persistent_state_db,
                )

                self._state_repo = get_persistent_state_db()
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
                "avg_significance": sum(e.significance for e in entries)
                / len(entries),
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
                )

                self._overload_guardian = get_memory_overload_protector(
                    stream_id or "default"
                )
            self._overload_guardian.assess_memory_overload()
            if self._overload_guardian.should_trigger_emergency_forgetting():
                logger.warning("记忆过载，触发紧急遗忘...")
                result = (
                    self._overload_guardian.execute_intelligent_forgetting(
                        target_reduction=0.25
                    )
                )
                logger.info(
                    f"紧急遗忘完成 | 遗忘{result['forgotten_count']}条 "
                    f"释放{result['space_freed_mb']:.1f}MB"
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
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(optimizer.run_optimization(stream_id))
            else:
                loop.run_until_complete(optimizer.run_optimization(stream_id))
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
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(optimizer.run_optimization(stream_id))
                self._ts_deep_opt = time.time()
                return {"status": "deep_optimization_scheduled"}
            result = loop.run_until_complete(
                optimizer.run_optimization(stream_id)
            )
            self._ts_deep_opt = time.time()
            return result
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
                )

                self._overload_guardian = get_memory_overload_protector(
                    stream_id or "default"
                )
            result = self._overload_guardian.execute_intelligent_forgetting(
                reduction
            )
            logger.warning(
                f"手动紧急遗忘完成 | 遗忘{result['forgotten_count']}条 "
                f"释放{result['space_freed_mb']:.1f}MB"
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
            return self._overload_guardian.get_overload_statistics()
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
            return self._overload_guardian.get_overload_behavioral_prompt()
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


def acquire_recollection_hub() -> RecollectionHub:
    """获取记忆核心引擎单例。"""
    global _hub_instance
    if _hub_instance is None:
        _hub_instance = RecollectionHub()
    return _hub_instance


def release_recollection_hub():
    """释放单例（用于测试或重启）。"""
    global _hub_instance
    _hub_instance = None

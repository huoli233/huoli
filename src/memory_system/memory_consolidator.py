import time
import json
import hashlib
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from src.common.logger import get_logger

logger = get_logger("记忆融合")


# ==================== 记忆年龄分级常量 ====================


class MemoryAgeCategory:
    """记忆年龄分级常量（与MaiBot的7级MemoryAgeLevel完全不同的6级体系）。

    ┌─────────────┬──────────────────┬───────────┐
    │ 级别名称     │ 时间范围         │ 清晰度下限 │
    ├─────────────┼──────────────────┼───────────┤
    │ FRESH       │ 0 ~ 5分钟        │ 0.95     │
    │ TODAY       │ 5分钟 ~ 24小时    │ 0.80     │
    │ YESTERDAY   │ 24小时 ~ 48小时   │ 0.60     │
    │ RECENT      │ 2天 ~ 7天        │ 0.40     │
    │ WEEK_OLD    │ 7天 ~ 30天       │ 0.20     │
    │ ARCHIVE     │ >30天            │ 0.05     │
    └─────────────┴──────────────────┴───────────┘
    """

    FRESH = 0
    TODAY = 1
    YESTERDAY = 2
    RECENT = 3
    WEEK_OLD = 4
    ARCHIVE = 5


# 年龄区间上界（秒）
AGE_BOUNDARY_MAP = {
    MemoryAgeCategory.FRESH: 300,
    MemoryAgeCategory.TODAY: 86400,
    MemoryAgeCategory.YESTERDAY: 172800,
    MemoryAgeCategory.RECENT: 604800,
    MemoryAgeCategory.WEEK_OLD: 2592000,
}

# 各级清晰度下限
AGE_SHARPNESS_FLOOR = {
    MemoryAgeCategory.FRESH: 0.95,
    MemoryAgeCategory.TODAY: 0.80,
    MemoryAgeCategory.YESTERDAY: 0.60,
    MemoryAgeCategory.RECENT: 0.40,
    MemoryAgeCategory.WEEK_OLD: 0.20,
    MemoryAgeCategory.ARCHIVE: 0.05,
}


def classify_age_bracket(birth_ts: float) -> int:
    """根据创建时间戳判定记忆所属年龄段。"""
    elapsed = time.time() - birth_ts
    for bracket, ceiling in sorted(
        AGE_BOUNDARY_MAP.items(), key=lambda x: x[1]
    ):
        if elapsed <= ceiling:
            return bracket
    return MemoryAgeCategory.ARCHIVE


# ==================== 记忆融合器 ====================


class MemoryFusion:
    """会话流级别的记忆整合器 —— 合并、去重、与海马体同步。

    核心功能：
    - 对指定会话流执行整合（聚类相似条目后合并）
    - 与海马体缓冲区双向同步
    """

    def __init__(self):
        self._fusion_lock = None

    def _ensure_lock(self):
        if self._fusion_lock is None:
            self._fusion_lock = asyncio.Lock()

    async def consolidate_stream_entries(
        self, stream_id: str
    ) -> Dict[str, Any]:
        """对指定会话流执行完整记忆整合。"""
        self._ensure_lock()
        async with self._fusion_lock:
            report = {
                "stream_id": stream_id,
                "merged_groups": 0,
                "synced": False,
            }
            try:
                StorageModel = self._load_record_model()
                if not StorageModel:
                    return report
                entries = list(
                    StorageModel.select()
                    .where(StorageModel.stream_id == stream_id)
                    .order_by(StorageModel.birth_ts.desc())
                )
                if len(entries) < 3:
                    return report
                # 步骤1：相似聚类
                similarity_groups = self._cluster_similar_entries(entries)
                # 步骤2：合并各组
                for group in similarity_groups:
                    if len(group) >= 2:
                        self._unite_entry_cluster(group, StorageModel)
                        report["merged_groups"] += 1
                # 步骤3：海马体同步
                report["synced"] = self._synchronize_with_hippocampus(
                    stream_id
                )
                logger.info(
                    f"[记忆融合] stream={stream_id[:8]}... "
                    f"合并{report['merged_groups']}组"
                )
            except Exception as exc:
                logger.error(f"记忆整合失败: {exc}")
                report["error"] = str(exc)
            return report

    def _cluster_similar_entries(self, entries: List) -> List[List]:
        """将记忆条目按内容相似度聚类，返回多组列表。"""
        visited = set()
        groups = []
        for i, anchor in enumerate(entries):
            if i in visited:
                continue
            cluster = [anchor]
            visited.add(i)
            for j, candidate in enumerate(entries):
                if j in visited:
                    continue
                similarity = self._calculate_resemblance(
                    anchor.content, candidate.content
                )
                if similarity > 0.75:
                    cluster.append(candidate)
                    visited.add(j)
            if len(cluster) >= 2:
                groups.append(cluster)
        return groups

    @staticmethod
    def _calculate_resemblance(text_a: str, text_b: str) -> float:
        """基于字符级n-gram（3-gram）计算两段文本的相似度。"""
        if not text_a or not text_b:
            return 0.0

        def _trigrams(text: str) -> set:
            t = text.lower().strip()
            if len(t) < 3:
                return {t}
            return {t[k: k + 3] for k in range(len(t) - 2)}

        grams_a = _trigrams(text_a)
        grams_b = _trigrams(text_b)
        shared = len(grams_a & grams_b)
        total = len(grams_a | grams_b)
        return shared / total if total > 0 else 0.0

    def _unite_entry_cluster(self, cluster: List, StorageModel):
        """将一组相似记忆合并为一条，保留最高重要性条目，删除其余。"""
        if len(cluster) < 2:
            return
        cluster.sort(key=lambda e: e.significance, reverse=True)
        survivor = cluster[0]
        combined_visits = sum(e.visit_count for e in cluster)
        max_significance = max(e.significance for e in cluster)
        # 聚合访问次数和重要性
        survivor.visit_count = combined_visits
        survivor.significance = min(1.0, max_significance)
        survivor.last_visit_ts = time.time()
        survivor.save()
        # 移除冗余条目
        for entry in cluster[1:]:
            entry.delete_instance()

    def _synchronize_with_hippocampus(self, stream_id: str) -> bool:
        """与海马体缓冲区执行双向同步。"""
        try:
            from src.memory_system.hippocampus_buffer import (
                get_hippocampus_buffer,
            )

            hippo = get_hippocampus_buffer(stream_id)
            hippo.sync()
            return True
        except ImportError:
            return False
        except Exception as exc:
            logger.debug(f"海马体同步跳过: {exc}")
            return False

    @staticmethod
    def _load_record_model():
        """惰性加载MemoryRecord模型。"""
        try:
            from src.common.database.database_model import MemoryRecord

            return MemoryRecord
        except Exception as exc:
            logger.warning(f"记忆合并器模型加载失败: {exc}")
            return None


# ==================== 短期记忆库 ====================


class EphemeralMemoryBank:
    """短期记忆管理器 —— 内存驻留 + 数据库持久化的双层缓冲。

    特性：
    - 按会话频道分桶存储
    - 基于衰减曲线的智能检索
    - 内容过滤（跳过空白/过短/重复内容）
    - 跨场景用户认知追踪（天真好奇学习系统）
    - 定时持久化与过期回收
    """

    def __init__(
        self, capacity_per_channel: int = 200, persist_interval: float = 180.0
    ):
        # 分频道内存缓冲: {channel_id: [(content, user_id, timestamp, significance), ...]}
        self._channel_buffers: Dict[
            str, List[Tuple[str, str, float, float]]
        ] = {}
        self._capacity = capacity_per_channel
        self._persist_interval = persist_interval
        self._last_persist_ts = time.time()
        # 跨场景用户学习追踪: {user_id: [{category, payload, significance, origin_stream, ts}, ...]}
        self._cross_context_insights: Dict[str, List[Dict]] = {}
        # 是否已从数据库恢复
        self._restored = False

    def deposit(
        self,
        channel_id: str,
        content: str,
        user_id: str = "",
        significance: float = 0.5,
    ):
        """向指定频道存入一条短期记忆。自动执行内容过滤和容量控制。"""
        if not content or len(content.strip()) < 2:
            return
        trimmed = content.strip()
        # 内容去重：与最近5条比较
        if channel_id in self._channel_buffers:
            recent = self._channel_buffers[channel_id][-5:]
            for prev_content, _, _, _ in recent:
                if prev_content == trimmed:
                    return
        # 存入缓冲区
        if channel_id not in self._channel_buffers:
            self._channel_buffers[channel_id] = []
        self._channel_buffers[channel_id].append(
            (trimmed, user_id, time.time(), significance)
        )
        # 容量溢出时淘汰最旧的条目
        if len(self._channel_buffers[channel_id]) > self._capacity:
            overflow = len(self._channel_buffers[channel_id]) - self._capacity
            self._channel_buffers[channel_id] = self._channel_buffers[
                channel_id
            ][overflow:]
        # 定时持久化检查
        if time.time() - self._last_persist_ts > self._persist_interval:
            self._auto_persist_and_evict()

    def retrieve_recent(
        self, channel_id: str, ceiling: int = 10
    ) -> List[Tuple[str, str, float, float]]:
        """获取指定频道最近的短期记忆，按时间倒序。"""
        if channel_id not in self._channel_buffers:
            return []
        buf = self._channel_buffers[channel_id]
        return list(reversed(buf[-ceiling:]))

    def retrieve_user_entries(
        self, channel_id: str, user_id: str, ceiling: int = 20
    ) -> List[Tuple[str, str, float, float]]:
        """获取指定频道中某用户的短期记忆。"""
        if channel_id not in self._channel_buffers:
            return []
        hits = [
            (c, u, t, s)
            for c, u, t, s in self._channel_buffers[channel_id]
            if u == user_id
        ]
        return list(reversed(hits[-ceiling:]))

    def list_known_users(self, channel_id: str) -> List[str]:
        """列出指定频道中所有已知用户ID。"""
        if channel_id not in self._channel_buffers:
            return []
        seen = set()
        for _, uid, _, _ in self._channel_buffers[channel_id]:
            if uid:
                seen.add(uid)
        return list(seen)

    def purge_channel(self, channel_id: str) -> int:
        """清空指定频道的短期记忆。"""
        if channel_id not in self._channel_buffers:
            return 0
        count = len(self._channel_buffers[channel_id])
        del self._channel_buffers[channel_id]
        return count

    def evict_expired(self, max_age_hours: float = 24.0) -> int:
        """清除所有频道中超过指定时长的短期记忆。"""
        cutoff = time.time() - max_age_hours * 3600
        total_evicted = 0
        for channel_id in list(self._channel_buffers.keys()):
            before = len(self._channel_buffers[channel_id])
            self._channel_buffers[channel_id] = [
                (c, u, t, s)
                for c, u, t, s in self._channel_buffers[channel_id]
                if t >= cutoff
            ]
            total_evicted += before - len(self._channel_buffers[channel_id])
            if not self._channel_buffers[channel_id]:
                del self._channel_buffers[channel_id]
        return total_evicted

    def retrieve_by_age_bracket(
        self, channel_id: str, bracket: int
    ) -> List[Tuple[str, str, float, float]]:
        """按年龄段检索指定频道的记忆。"""
        if channel_id not in self._channel_buffers:
            return []
        return [
            (c, u, t, s)
            for c, u, t, s in self._channel_buffers[channel_id]
            if classify_age_bracket(t) == bracket
        ]

    def retrieve_by_days_offset(
        self, channel_id: str, days_ago: int = 1, ceiling: int = 50
    ) -> List[Tuple[str, str, float, float]]:
        """按天数偏移检索（例如days_ago=1表示昨天的记忆）。"""
        if channel_id not in self._channel_buffers:
            return []
        now = time.time()
        day_start = now - (days_ago + 1) * 86400
        day_end = now - days_ago * 86400
        hits = [
            (c, u, t, s)
            for c, u, t, s in self._channel_buffers[channel_id]
            if day_start <= t < day_end
        ]
        return list(reversed(hits[-ceiling:]))

    def compute_entry_sharpness(
        self, birth_ts: float, significance: float = 0.5
    ) -> float:
        """估算单条短期记忆的当前清晰度。"""
        bracket = classify_age_bracket(birth_ts)
        floor = AGE_SHARPNESS_FLOOR.get(bracket, 0.05)
        elapsed_hours = (time.time() - birth_ts) / 3600
        # 指数衰减（半衰期8小时） + 重要性加成
        raw = max(0.0, 1.0 * (0.5 ** (elapsed_hours / 8.0)))
        return max(floor, min(1.0, raw + significance * 0.1))

    # ==================== 跨场景用户学习 ====================

    def register_cross_context_insight(
        self,
        user_id: str,
        log_category: str,
        payload: str,
        significance: float = 0.5,
        origin_stream: str = "",
    ):
        """记录一条跨场景的用户认知片段。

        这些片段在多个会话流之间共享，帮助机器人积累对用户行为模式的理解。
        """
        if not user_id or not payload:
            return
        if user_id not in self._cross_context_insights:
            self._cross_context_insights[user_id] = []
        insight_entry = {
            "category": log_category,
            "payload": payload,
            "significance": significance,
            "origin_stream": origin_stream,
            "ts": time.time(),
        }
        # 限制单用户最多50条认知片段
        self._cross_context_insights[user_id].append(insight_entry)
        if len(self._cross_context_insights[user_id]) > 50:
            self._cross_context_insights[user_id] = (
                self._cross_context_insights[user_id][-50:]
            )
        # 同步写入数据库
        self._persist_cross_context_insights(user_id)

    def query_cross_context_insights(
        self, user_id: str, ceiling: int = 10
    ) -> List[Dict]:
        """查询某用户的跨场景认知片段（最近N条）。"""
        if user_id not in self._cross_context_insights:
            self._restore_cross_context_insights(user_id)
        items = self._cross_context_insights.get(user_id, [])
        return list(reversed(items[-ceiling:]))

    def compose_learning_state_prompt(self, user_id: str) -> str:
        """为指定用户生成「天真好奇」风格的认知学习提示词。

        这段提示词注入LLM prompt，使机器人以「好奇宝宝」的视角
        对用户进行渐进式理解，避免过早做出判断。
        """
        insights = self.query_cross_context_insights(user_id, ceiling=15)
        if not insights:
            return ""
        observation_lines = []
        for idx, insight in enumerate(insights, 1):
            cat = insight.get("category", "未知")
            payload = insight.get("payload", "")
            if len(payload) > 120:
                payload = payload[:120] + "..."
            observation_lines.append(f"  {idx}. [{cat}] {payload}")
        observations_block = "\n".join(observation_lines)
        prompt = (
            f"你对用户({user_id})有以下零散的观察记忆：\n"
            f"{observations_block}\n\n"
            f"你像一个天真无邪的孩子一样对这些信息充满好奇。\n"
            f"你不会对用户做出武断的判断，而是带着温和的好奇心去理解对方。\n"
            f"当发现新的信息片段时，你会结合已有观察进行思考，"
            f"但不会急于下结论。"
        )
        return prompt

    def _persist_cross_context_insights(self, user_id: str):
        """将用户的跨场景认知片段写入数据库。"""
        try:
            from src.common.database.database_model import UserLearningLog

            items = self._cross_context_insights.get(user_id, [])
            for entry in items:
                now = time.time()
                # 尝试更新已有记录，否则新建
                existing = (
                    UserLearningLog.select()
                    .where(
                        (UserLearningLog.user_id == user_id)
                        & (UserLearningLog.log_category == entry["category"])
                        & (UserLearningLog.payload == entry["payload"])
                    )
                    .first()
                )
                if existing:
                    existing.update_ts = now
                    existing.save()
                else:
                    UserLearningLog.create(
                        user_id=user_id,
                        log_category=entry["category"],
                        payload=entry["payload"],
                        significance=entry.get("significance", 0.5),
                        origin_stream=entry.get("origin_stream", ""),
                        birth_ts=entry.get("ts", now),
                        update_ts=now,
                    )
        except ImportError:
            pass
        except Exception as exc:
            logger.debug(f"跨场景认知持久化失败: {exc}")

    def _restore_cross_context_insights(self, user_id: str):
        """从数据库恢复用户的跨场景认知片段。"""
        if user_id in self._cross_context_insights:
            return
        try:
            from src.common.database.database_model import UserLearningLog

            records = list(
                UserLearningLog.select()
                .where(UserLearningLog.user_id == user_id)
                .order_by(UserLearningLog.birth_ts.asc())
                .limit(50)
            )
            items = []
            for rec in records:
                items.append(
                    {
                        "category": rec.log_category,
                        "payload": rec.payload,
                        "significance": rec.significance,
                        "origin_stream": rec.origin_stream,
                        "ts": rec.birth_ts,
                    }
                )
            self._cross_context_insights[user_id] = items
        except ImportError:
            self._cross_context_insights[user_id] = []
        except Exception as exc:
            logger.debug(f"跨场景认知恢复失败: {exc}")
            self._cross_context_insights[user_id] = []

    # ==================== 持久化 ====================

    def _persist_all_channels(self):
        """将所有频道的短期记忆批量写入数据库快照。"""
        try:
            from src.common.database.database_model import StreamStateSnapshot

            now = time.time()
            for channel_id, buf in self._channel_buffers.items():
                serialized = json.dumps(
                    [{"c": c, "u": u, "t": t, "s": s} for c, u, t, s in buf],
                    ensure_ascii=False,
                )
                existing = (
                    StreamStateSnapshot.select()
                    .where(
                        (StreamStateSnapshot.stream_id == channel_id)
                        & (StreamStateSnapshot.state_key == "ephemeral_buffer")
                    )
                    .first()
                )
                if existing:
                    existing.state_payload = serialized
                    existing.update_ts = now
                    existing.save()
                else:
                    StreamStateSnapshot.create(
                        stream_id=channel_id,
                        state_key="ephemeral_buffer",
                        state_payload=serialized,
                        update_ts=now,
                    )
            self._last_persist_ts = now
        except ImportError:
            pass
        except Exception as exc:
            logger.debug(f"短期记忆持久化失败: {exc}")

    def _restore_all_channels(self):
        """从数据库恢复所有频道的短期记忆。"""
        if self._restored:
            return
        try:
            from src.common.database.database_model import StreamStateSnapshot

            snapshots = list(
                StreamStateSnapshot.select().where(
                    StreamStateSnapshot.state_key == "ephemeral_buffer"
                )
            )
            for snap in snapshots:
                try:
                    records = json.loads(snap.state_payload)
                    buf = [(r["c"], r["u"], r["t"], r["s"]) for r in records]
                    self._channel_buffers[snap.stream_id] = buf
                except (json.JSONDecodeError, KeyError):
                    continue
            self._restored = True
            logger.debug(f"短期记忆恢复: {len(snapshots)}个频道")
        except ImportError:
            self._restored = True
        except Exception as exc:
            logger.debug(f"短期记忆恢复失败: {exc}")
            self._restored = True

    def _auto_persist_and_evict(self):
        """定时自动持久化并淘汰过期条目。"""
        evicted = self.evict_expired(max_age_hours=24.0)
        self._persist_all_channels()
        if evicted > 0:
            logger.debug(f"自动清理过期短期记忆: {evicted}条")


# ==================== 长期事实仓库 ====================


class PersistentFactStore:
    """长期记忆薄层封装 —— 提供面向事实/知识的存取接口。"""

    def __init__(self):
        self._hub = None

    def _get_hub(self):
        """惰性获取RecollectionHub实例。"""
        if self._hub is None:
            from src.memory_system.memory_core import acquire_recollection_hub

            self._hub = acquire_recollection_hub()
        return self._hub

    def inscribe(
        self,
        stream_id: str,
        content: str,
        user_id: str = "",
        significance: float = 0.7,
    ) -> str:
        """铭刻一条长期事实。"""
        hub = self._get_hub()
        return hub.deposit_memory(
            stream_id=stream_id,
            content=content,
            entry_category="fact",
            user_id=user_id,
            significance=significance,
        )

    def recall(self, stream_id: str, keyword: str, ceiling: int = 5) -> List:
        """按关键词检索长期事实。"""
        hub = self._get_hub()
        return hub.query_memories(
            stream_id=stream_id,
            keyword=keyword,
            entry_category="fact",
            ceiling=ceiling,
        )

    def recall_all(self, stream_id: str, ceiling: int = 100) -> List:
        """检索指定会话流的全部长期事实。"""
        hub = self._get_hub()
        return hub.query_memories(
            stream_id=stream_id,
            entry_category="fact",
            ceiling=ceiling,
        )


# ==================== 清晰度衰减引擎 ====================


class ClarityDecayEngine:
    """管理记忆清晰度的衰减、标注、清理和压缩。

    功能：
    - 计算瞬时清晰度
    - 为记忆条目附加清晰度标签
    - 清除已完全遗忘的条目
    - 压缩老旧对话记忆
    - 生成描述清晰度状态的提示词
    """

    # 清晰度标签
    LABEL_VIVID = "清晰"
    LABEL_BLURRY = "模糊"
    LABEL_FADING = "褪色"
    LABEL_FORGOTTEN = "遗忘"

    def compute_clarity(
        self, birth_ts: float, significance: float, visit_count: int = 0
    ) -> float:
        """计算记忆的当前清晰度（0~1）。

        衰减模型：
        - 基础半衰期 = 48小时
        - 重要性越高，半衰期越长（重要记忆衰减慢）
        - 访问频次可延缓衰减
        """
        elapsed_hours = (time.time() - birth_ts) / 3600
        # 动态半衰期: 基础48h + 重要性加成（最多+96h）+ 访问加成（每次+2h，上限24h）
        half_life = 48 + significance * 96 + min(24, visit_count * 2)
        base_clarity = 0.5 ** (elapsed_hours / half_life)
        return max(0.0, min(1.0, base_clarity))

    def classify_clarity_label(self, clarity: float) -> str:
        """将清晰度数值映射为人类可读标签。"""
        if clarity >= 0.7:
            return self.LABEL_VIVID
        elif clarity >= 0.4:
            return self.LABEL_BLURRY
        elif clarity >= 0.1:
            return self.LABEL_FADING
        return self.LABEL_FORGOTTEN

    def enrich_with_clarity(self, entry) -> dict:
        """为单条记忆条目计算并附加清晰度信息。"""
        clarity = self.compute_clarity(
            entry.birth_ts, entry.significance, entry.visit_count
        )
        label = self.classify_clarity_label(clarity)
        bracket = classify_age_bracket(entry.birth_ts)
        return {
            "record_id": entry.record_id,
            "content": entry.content,
            "clarity": round(clarity, 4),
            "clarity_label": label,
            "age_bracket": bracket,
            "significance": entry.significance,
        }

    def enrich_batch(self, entries: List) -> List[dict]:
        """批量为记忆条目附加清晰度信息。"""
        return [self.enrich_with_clarity(e) for e in entries]

    def evict_forgotten_entries(
        self, stream_id: Optional[str] = None, min_age_days: float = 30
    ) -> int:
        """清除清晰度极低且年龄达标的记忆条目。"""
        try:
            from src.common.database.database_model import MemoryRecord

            cutoff = time.time() - min_age_days * 86400
            q = MemoryRecord.select().where(MemoryRecord.birth_ts < cutoff)
            if stream_id:
                q = q.where(MemoryRecord.stream_id == stream_id)
            evicted = 0
            for entry in q:
                clarity = self.compute_clarity(
                    entry.birth_ts, entry.significance, entry.visit_count
                )
                if clarity < 0.05 and entry.significance < 0.6:
                    entry.delete_instance()
                    evicted += 1
            if evicted > 0:
                logger.info(f"清除已遗忘记忆: {evicted}条")
            return evicted
        except ImportError:
            return 0
        except Exception as exc:
            logger.error(f"遗忘清除失败: {exc}")
            return 0

    def condense_old_entries(
        self, stream_id: Optional[str] = None, min_age_days: float = 14
    ) -> int:
        """将超过指定天数的对话记忆压缩（截断+标记类型）。"""
        try:
            from src.common.database.database_model import MemoryRecord

            cutoff = time.time() - min_age_days * 86400
            q = MemoryRecord.select().where(
                (MemoryRecord.birth_ts < cutoff)
                & (MemoryRecord.entry_category == "conversation")
            )
            if stream_id:
                q = q.where(MemoryRecord.stream_id == stream_id)
            condensed = 0
            batch_updates = []
            for entry in q:
                if len(entry.content) > 200:
                    batch_updates.append({
                        "record_id": entry.record_id,
                        "content": entry.content[:150] + "...（已压缩）",
                        "entry_category": "compressed",
                    })
                    condensed += 1
            for upd in batch_updates:
                MemoryRecord.update(
                    content=upd["content"],
                    entry_category=upd["entry_category"],
                ).where(
                    MemoryRecord.record_id == upd["record_id"]
                ).execute()
            if condensed > 0:
                logger.info(f"压缩老旧记忆: {condensed}条")
            return condensed
        except ImportError:
            return 0
        except Exception as exc:
            logger.error(f"老旧压缩失败: {exc}")
            return 0

    def compose_clarity_prompt(self, entries: List) -> str:
        """为一组记忆条目生成清晰度描述的提示词块。"""
        if not entries:
            return ""
        enriched = self.enrich_batch(entries)
        lines = []
        for item in enriched:
            clarity_pct = int(item["clarity"] * 100)
            label = item["clarity_label"]
            content_preview = item["content"][:60]
            if len(item["content"]) > 60:
                content_preview += "..."
            lines.append(f"  [{label} {clarity_pct}%] {content_preview}")
        return "相关记忆（清晰度标注）：\n" + "\n".join(lines)


# ==================== 单例管理 ====================

_fusion_solo: Optional[MemoryFusion] = None
_ephemeral_solo: Optional[EphemeralMemoryBank] = None
_fact_store_solo: Optional[PersistentFactStore] = None
_clarity_engine_solo: Optional[ClarityDecayEngine] = None


def acquire_memory_fusion() -> MemoryFusion:
    """获取记忆融合器单例。"""
    global _fusion_solo
    if _fusion_solo is None:
        _fusion_solo = MemoryFusion()
    return _fusion_solo


def acquire_ephemeral_bank() -> EphemeralMemoryBank:
    """获取短期记忆库单例。"""
    global _ephemeral_solo
    if _ephemeral_solo is None:
        _ephemeral_solo = EphemeralMemoryBank()
    return _ephemeral_solo


def acquire_fact_store() -> PersistentFactStore:
    """获取长期事实仓库单例。"""
    global _fact_store_solo
    if _fact_store_solo is None:
        _fact_store_solo = PersistentFactStore()
    return _fact_store_solo


def acquire_clarity_engine() -> ClarityDecayEngine:
    """获取清晰度衰减引擎单例。"""
    global _clarity_engine_solo
    if _clarity_engine_solo is None:
        _clarity_engine_solo = ClarityDecayEngine()
    return _clarity_engine_solo

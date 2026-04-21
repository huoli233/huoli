import re
import time
import json
import hashlib
import asyncio
from enum import IntEnum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("海马缓冲")


# ==================== 记忆分层定义 ====================


class RetentionLayer(IntEnum):
    """记忆留存层级 —— 模拟大脑从感觉记忆到长期记忆的分层结构。

    5级阶梯式留存体系：
    ┌──────────────┬───────────────┬────────────┐
    │ 层级名称      │ 时间跨度       │ 清晰乘数    │
    ├──────────────┼───────────────┼────────────┤
    │ SENSORY      │ 0 ~ 1小时      │ 1.00      │
    │ WORKING      │ 1 ~ 24小时     │ 0.85      │
    │ SHORT_TERM   │ 1 ~ 3天        │ 0.65      │
    │ MID_TERM     │ 3 ~ 7天        │ 0.45      │
    │ CONSOLIDATED │ 7天以上        │ 0.25      │
    └──────────────┴───────────────┴────────────┘
    """

    SENSORY = 0
    WORKING = 1
    SHORT_TERM = 2
    MID_TERM = 3
    CONSOLIDATED = 4


# 各层时间上界（秒）
LAYER_TIME_CEILING = {
    RetentionLayer.SENSORY: 3600,
    RetentionLayer.WORKING: 86400,
    RetentionLayer.SHORT_TERM: 259200,
    RetentionLayer.MID_TERM: 604800,
    RetentionLayer.CONSOLIDATED: float("inf"),
}

# 各层清晰度乘数（降层时使用）
LAYER_SHARPNESS_FACTOR = {
    RetentionLayer.SENSORY: 1.00,
    RetentionLayer.WORKING: 0.85,
    RetentionLayer.SHORT_TERM: 0.65,
    RetentionLayer.MID_TERM: 0.45,
    RetentionLayer.CONSOLIDATED: 0.25,
}


@dataclass
class BufferedFragment:
    """缓冲区中的单条记忆碎片。"""

    fragment_id: str
    stream_id: str
    author_id: Optional[str]
    body: str
    synopsis: Optional[str]
    weight: float
    born_at: float
    touched_at: float
    touch_count: int
    layer: RetentionLayer
    sharpness: float = 1.0
    tags: List[str] = field(default_factory=list)
    condensed_body: Optional[str] = None


# ==================== 海马体缓冲核心 ====================


class HippoMemoryBuffer:
    """海马体记忆缓冲区 —— 管理记忆碎片在多层之间的流转与衰减。

    核心机制：
    - 新记忆进入感觉层（SENSORY），随时间自动向下流转
    - 高频访问的记忆可被「回忆强化」提升至更高层
    - 低重要性+低访问的记忆在流转时可被自然遗忘
    - 进入中期层以上的长内容自动生成压缩副本
    - 搜索相关度受清晰度加权影响
    """

    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        # 分层存储桶: {layer: {fragment_id: BufferedFragment}}
        self._buckets: Dict[RetentionLayer, Dict[str, BufferedFragment]] = {
            layer: {} for layer in RetentionLayer
        }
        self._previous_drain_ts = 0.0
        self._drain_period = 1800.0  # 每30分钟执行一次层间流转
        self._hydrate_from_storage()

    # ==================== ID与关键词 ====================

    def _mint_fragment_id(self, body: str) -> str:
        """铸造碎片唯一标识。"""
        seed = f"{self.stream_id}|{body}|{time.time()}"
        return hashlib.md5(seed.encode()).hexdigest()[:16]

    @staticmethod
    def _distill_tags(text: str) -> List[str]:
        """从文本中提取关键词标签（频率统计法）。"""
        tokens = re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", text)
        frequency_map: Dict[str, int] = {}
        for token in tokens:
            if len(token) >= 2:
                frequency_map[token] = frequency_map.get(token, 0) + 1
        ranked = sorted(
            frequency_map.items(), key=lambda p: p[1], reverse=True
        )
        return [word for word, _ in ranked[:5]]

    @staticmethod
    def _compose_synopsis(body: str, cap: int = 80) -> str:
        """为碎片内容自动生成简短摘要。"""
        if len(body) <= cap:
            return body
        return body[:cap] + "..."

    # ==================== 碎片增删查 ====================

    def ingest(
        self,
        body: str,
        author_id: Optional[str] = None,
        weight: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> str:
        """将新内容注入感觉层，返回碎片ID。"""
        frag_id = self._mint_fragment_id(body)
        now = time.time()
        fragment = BufferedFragment(
            fragment_id=frag_id,
            stream_id=self.stream_id,
            author_id=author_id,
            body=body,
            synopsis=self._compose_synopsis(body),
            weight=weight,
            born_at=now,
            touched_at=now,
            touch_count=0,
            layer=RetentionLayer.SENSORY,
            sharpness=1.0,
            tags=tags or self._distill_tags(body),
        )
        self._buckets[RetentionLayer.SENSORY][frag_id] = fragment
        logger.debug(
            f"[海马体] 碎片注入: {frag_id[:8]} (stream: {self.stream_id[:8]})"
        )
        return frag_id

    def add_memory(
        self,
        content: str,
        user_id: Optional[str] = None,
        importance: float = 0.5,
    ) -> str:
        """兼容接口（供memory_core调用）。"""
        return self.ingest(body=content, author_id=user_id, weight=importance)

    def touch(self, fragment_id: str) -> Optional[BufferedFragment]:
        """访问指定碎片，刷新时间和计数，高频访问时触发回忆强化。"""
        for layer in RetentionLayer:
            if fragment_id in self._buckets[layer]:
                frag = self._buckets[layer][fragment_id]
                frag.touched_at = time.time()
                frag.touch_count += 1
                # 高频访问的碎片回忆强化（向上提升一层）
                if frag.touch_count >= 5 and layer > RetentionLayer.WORKING:
                    self._elevate_fragment(frag, layer)
                return frag
        return None

    def _elevate_fragment(
        self, fragment: BufferedFragment, source_layer: RetentionLayer
    ):
        """将碎片提升至更高层（回忆强化效应）。"""
        if source_layer <= RetentionLayer.WORKING:
            return
        target = RetentionLayer(source_layer - 1)
        del self._buckets[source_layer][fragment.fragment_id]
        fragment.layer = target
        fragment.touched_at = time.time()
        self._buckets[target][fragment.fragment_id] = fragment
        logger.debug(
            f"[海马体] 碎片回忆强化: {fragment.fragment_id[:8]} -> {target.name}"
        )

    # ==================== 检索 ====================

    def fetch_recent(
        self,
        ceiling: int = 20,
        max_layer: RetentionLayer = RetentionLayer.SHORT_TERM,
        author_id: Optional[str] = None,
    ) -> List[BufferedFragment]:
        """获取最近的碎片列表，按创建时间倒序。"""
        pool = []
        for layer in RetentionLayer:
            if layer > max_layer:
                break
            for frag in self._buckets[layer].values():
                if author_id and frag.author_id != author_id:
                    continue
                pool.append(frag)
        pool.sort(key=lambda f: f.born_at, reverse=True)
        return pool[:ceiling]

    def probe(
        self, query: str, ceiling: int = 10
    ) -> List[Tuple[BufferedFragment, float]]:
        """按关键词搜索碎片并返回(碎片, 相关度)列表。"""
        hits = []
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        for layer in RetentionLayer:
            for frag in self._buckets[layer].values():
                relevance = self._gauge_relevance(
                    query_lower, query_tokens, frag
                )
                if relevance > 0.1:
                    hits.append((frag, relevance))
        hits.sort(key=lambda pair: pair[1], reverse=True)
        return hits[:ceiling]

    def search_memories(
        self, query: str, limit: int = 10
    ) -> List[Tuple[BufferedFragment, float]]:
        """兼容接口（供memory_core调用）。"""
        return self.probe(query, ceiling=limit)

    def _gauge_relevance(
        self, query_lower: str, query_tokens: set, fragment: BufferedFragment
    ) -> float:
        """计算搜索相关度，受清晰度加权。"""
        score = 0.0
        body_lower = fragment.body.lower()
        # 完整子串匹配
        if query_lower in body_lower:
            score += 0.5
        # 单词重叠率
        body_tokens = set(body_lower.split())
        if query_tokens:
            overlap = len(query_tokens & body_tokens) / len(query_tokens)
            score += overlap * 0.3
        # 标签匹配
        for tag in fragment.tags:
            if query_lower in tag.lower():
                score += 0.2
                break
        # 清晰度加权
        current_sharpness = self._compute_sharpness(fragment)
        score *= current_sharpness
        return min(1.0, score)

    def _compute_sharpness(self, fragment: BufferedFragment) -> float:
        """实时计算碎片清晰度。"""
        try:
            from src.memory_system.memory_helpers import (
                calculate_memory_clarity,
            )

            return calculate_memory_clarity(
                fragment.born_at,
                importance=fragment.weight,
                access_count=fragment.touch_count,
            )
        except ImportError:
            # 内置简易衰减模型
            age_hours = (time.time() - fragment.born_at) / 3600
            base = max(0.0, 1.0 * (0.5 ** (age_hours / 48)))
            return max(0.05, min(1.0, base + fragment.weight * 0.1))

    # ==================== 时间范围查询 ====================

    def fetch_by_timespan(
        self, start_ts: float, end_ts: float, ceiling: int = 30
    ) -> List[BufferedFragment]:
        """按时间范围检索碎片。"""
        result = []
        for layer in RetentionLayer:
            for frag in self._buckets[layer].values():
                if start_ts <= frag.born_at <= end_ts:
                    result.append(frag)
        result.sort(key=lambda f: f.born_at, reverse=True)
        return result[:ceiling]

    def fetch_by_days_offset(
        self, days_ago: int, ceiling: int = 20
    ) -> List[BufferedFragment]:
        """获取N天前的碎片。"""
        now = time.time()
        start = now - (days_ago + 1) * 86400
        end = now - days_ago * 86400
        return self.fetch_by_timespan(start, end, ceiling)

    # ==================== 层间流转（核心调度） ====================

    async def drain_and_transform(self):
        """执行层间流转（记忆老化 + 遗忘 + 压缩）。"""
        now = time.time()
        if now - self._previous_drain_ts < self._drain_period:
            return
        self._previous_drain_ts = now
        moved_count = forgotten_count = compressed_count = 0
        # 从高层向低层逐层处理（不处理最底层CONSOLIDATED）
        for layer in list(RetentionLayer)[:-1]:
            next_layer = RetentionLayer(layer + 1)
            time_ceiling = LAYER_TIME_CEILING[layer]
            pending_migration = []
            for frag_id, frag in list(self._buckets[layer].items()):
                age = now - frag.born_at
                # 自然衰减：每轮清晰度轻微下降
                decay_rate = 0.99 if frag.touch_count < 3 else 0.995
                frag.sharpness *= decay_rate
                if age >= time_ceiling:
                    pending_migration.append((frag_id, frag))
            for frag_id, frag in pending_migration:
                del self._buckets[layer][frag_id]
                # 遗忘判定
                if self._should_discard(frag, next_layer):
                    forgotten_count += 1
                    continue
                # 清晰度受层级乘数修正
                frag.sharpness *= LAYER_SHARPNESS_FACTOR.get(next_layer, 1.0)
                # 进入中期及以上层级时自动压缩
                if (
                    next_layer >= RetentionLayer.MID_TERM
                    and not frag.condensed_body
                ):
                    frag.condensed_body = self._condense_body(frag.body)
                    compressed_count += 1
                frag.layer = next_layer
                self._buckets[next_layer][frag_id] = frag
                moved_count += 1
        if moved_count > 0 or forgotten_count > 0:
            logger.info(
                f"[海马体] 流转完成: 迁移{moved_count} 压缩{compressed_count} 遗忘{forgotten_count}"
            )
            self._flush_to_storage()

    def _should_discard(
        self, fragment: BufferedFragment, destination: RetentionLayer
    ) -> bool:
        """拟人化遗忘决策 —— 低重要性且低访问频次的碎片自然遗忘。"""
        if fragment.weight >= 0.7:
            return False
        if fragment.touch_count >= 3:
            return False
        discard_rules = {
            RetentionLayer.WORKING: (0.3, 0),
            RetentionLayer.SHORT_TERM: (0.4, 1),
            RetentionLayer.MID_TERM: (0.5, 2),
            RetentionLayer.CONSOLIDATED: (0.6, 2),
        }
        threshold_weight, threshold_touch = discard_rules.get(
            destination, (0.6, 3)
        )
        return (
            fragment.weight < threshold_weight
            and fragment.touch_count <= threshold_touch
        )

    @staticmethod
    def _condense_body(body: str, cap: int = 150) -> str:
        """将长文本压缩为印象级摘要。"""
        if len(body) <= cap:
            return body
        # 按句号/问号/感叹号切分，取前几句
        clauses = re.split(r"[。！？.!?]", body)
        condensed_parts = []
        running_len = 0
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            if running_len + len(clause) > cap:
                break
            condensed_parts.append(clause)
            running_len += len(clause) + 1
        if not condensed_parts:
            return body[:cap] + "..."
        return "。".join(condensed_parts) + "..."

    # ==================== Prompt生成 ====================

    def compose_prompt_context(
        self, query: Optional[str] = None, token_budget: int = 500
    ) -> str:
        """为LLM注入生成记忆上下文区块。"""
        sections = []
        if query:
            probe_hits = self.probe(query, ceiling=5)
            if probe_hits:
                section_lines = ["[相关记忆]"]
                for frag, _score in probe_hits:
                    sharpness = self._compute_sharpness(frag)
                    display = self._render_fragment(frag, sharpness)
                    label = self._sharpness_label(sharpness)
                    section_lines.append(f"  ({label}) {display}")
                sections.append("\n".join(section_lines))
        now = time.time()
        temporal_bands = [
            ("今天的回忆", 0, 86400),
            ("昨天的回忆", 86400, 172800),
            ("前几天的回忆", 172800, 604800),
        ]
        for band_name, lo, hi in temporal_bands:
            band_frags = []
            for layer in RetentionLayer:
                for frag in self._buckets[layer].values():
                    age = now - frag.born_at
                    if lo <= age < hi:
                        band_frags.append(frag)
            if band_frags:
                band_frags.sort(key=lambda f: f.weight, reverse=True)
                section_lines = [f"\n[{band_name}]"]
                for frag in band_frags[:3]:
                    sharpness = self._compute_sharpness(frag)
                    display = self._render_fragment(frag, sharpness)
                    section_lines.append(f"  {display}")
                sections.append("\n".join(section_lines))
        combined = "\n".join(sections)
        if len(combined) > token_budget * 2:
            combined = combined[: token_budget * 2] + "...(记忆过多，已截断)"
        return combined

    def _render_fragment(
        self, fragment: BufferedFragment, sharpness: float
    ) -> str:
        """根据清晰度渲染碎片的展示文本。"""
        if sharpness >= 0.7:
            return fragment.body[:100]
        elif sharpness >= 0.4:
            return fragment.synopsis or fragment.body[:60]
        elif sharpness >= 0.2:
            preview = fragment.condensed_body or fragment.body[:40]
            return f"模糊记得...{preview}"
        return "似乎有过这样的经历，但想不起细节了"

    @staticmethod
    def _sharpness_label(sharpness: float) -> str:
        """清晰度数值转人类标签。"""
        if sharpness >= 0.7:
            return "清晰"
        elif sharpness >= 0.4:
            return "略有模糊"
        elif sharpness >= 0.2:
            return "比较模糊"
        return "几乎遗忘"

    # ==================== 持久化 ====================

    def _flush_to_storage(self):
        """将全部层级碎片写入数据库（事务内批量 upsert）。"""
        try:
            from src.common.database.database_model import HippocampusEntry
            from src.common.database.database import db

            with db.atomic():
                for layer in RetentionLayer:
                    for frag in self._buckets[layer].values():
                        HippocampusEntry.insert(
                            memory_id=frag.fragment_id,
                            stream_id=self.stream_id,
                            user_id=frag.author_id,
                            content=frag.body,
                            digest=frag.synopsis,
                            significance=frag.weight,
                            sharpness=frag.sharpness,
                            birth_ts=frag.born_at,
                            last_touch_ts=frag.touched_at,
                            touch_count=frag.touch_count,
                            layer_index=layer.value,
                            tag_list=json.dumps(frag.tags, ensure_ascii=False),
                            condensed_text=frag.condensed_body,
                        ).on_conflict_replace().execute()
            logger.debug(f"[海马体] 数据库同步完成: {self.stream_id[:8]}")
        except Exception as exc:
            logger.error(f"[海马体] 数据库写入失败: {exc}")

    def _hydrate_from_storage(self):
        """从数据库恢复碎片到内存。"""
        try:
            from src.common.database.database_model import HippocampusEntry

            rows = list(
                HippocampusEntry.select().where(
                    HippocampusEntry.stream_id == self.stream_id
                )
            )
            for row in rows:
                try:
                    frag = BufferedFragment(
                        fragment_id=row.memory_id,
                        stream_id=row.stream_id,
                        author_id=row.user_id,
                        body=row.content,
                        synopsis=row.digest,
                        weight=row.significance,
                        born_at=row.birth_ts,
                        touched_at=row.last_touch_ts,
                        touch_count=row.touch_count,
                        layer=RetentionLayer(row.layer_index),
                        sharpness=getattr(row, "sharpness", 1.0),
                        tags=json.loads(row.tag_list) if row.tag_list else [],
                        condensed_body=row.condensed_text,
                    )
                    self._buckets[frag.layer][frag.fragment_id] = frag
                except Exception:
                    continue
            if rows:
                logger.debug(f"[海马体] 恢复碎片: {len(rows)}条")
        except Exception as exc:
            logger.debug(f"[海马体] 恢复跳过: {exc}")

    # ==================== 统计与清空 ====================

    def compile_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息。"""
        stats = {
            "stream_id": self.stream_id,
            "total_fragments": 0,
            "by_layer": {},
            "last_drain_ts": self._previous_drain_ts,
        }
        for layer in RetentionLayer:
            count = len(self._buckets[layer])
            stats["by_layer"][layer.name] = count
            stats["total_fragments"] += count
        return stats

    def purge(self):
        """清空所有层级并删除数据库记录。"""
        for layer in RetentionLayer:
            self._buckets[layer].clear()
        try:
            from src.common.database.database_model import HippocampusEntry

            HippocampusEntry.delete().where(
                HippocampusEntry.stream_id == self.stream_id
            ).execute()
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def sync(self):
        """兼容接口（供MemoryFusion调用须知同步）。"""
        self._flush_to_storage()


# ==================== 单例池管理 ====================

_buffer_pool: Dict[str, HippoMemoryBuffer] = {}


def get_hippocampus_buffer(stream_id: str) -> HippoMemoryBuffer:
    """获取指定会话流的海马体缓冲区实例。"""
    if stream_id not in _buffer_pool:
        _buffer_pool[stream_id] = HippoMemoryBuffer(stream_id)
    return _buffer_pool[stream_id]


def reset_hippocampus_buffer(stream_id: str):
    """重置并释放指定会话流的缓冲区。"""
    if stream_id in _buffer_pool:
        _buffer_pool[stream_id].purge()
        del _buffer_pool[stream_id]


# ==================== 后台调度器 ====================


class HippoTransformScheduler:
    """海马体碎片流转后台调度器，定时触发所有活跃缓冲区的层间流转。"""

    def __init__(self):
        self._active = False
        self._cycle_seconds = 1800.0
        self._worker_task: Optional[asyncio.Task] = None

    async def activate(self):
        """启动调度循环。"""
        if self._active:
            return
        self._active = True
        self._worker_task = asyncio.create_task(self._scheduling_loop())
        logger.info("[海马体调度] 已激活")

    async def deactivate(self):
        """停止调度循环。"""
        self._active = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("[海马体调度] 已停止")

    async def _scheduling_loop(self):
        """后台循环：每周期遍历所有缓冲区执行流转。"""
        while self._active:
            await asyncio.sleep(self._cycle_seconds)
            await self._drain_all_buffers()

    async def _drain_all_buffers(self):
        """遍历所有活跃缓冲区执行流转。"""
        for buffer in _buffer_pool.values():
            try:
                await buffer.drain_and_transform()
            except Exception as exc:
                logger.debug(f"[海马体调度] 流转异常: {exc}")


_transform_scheduler: Optional[HippoTransformScheduler] = None


def get_hippocampus_scheduler() -> HippoTransformScheduler:
    """获取全局调度器单例。"""
    global _transform_scheduler
    if _transform_scheduler is None:
        _transform_scheduler = HippoTransformScheduler()
    return _transform_scheduler


async def init_hippocampus_system():
    """初始化海马体子系统（启动调度器）。"""
    scheduler = get_hippocampus_scheduler()
    await scheduler.activate()
    return scheduler

import time
import math
import hashlib
import asyncio
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("切片整合")


# ==================== 整合裁决枚举 ====================


class ConsolidationVerdict(Enum):
    """整合裁决（与MaiBot的KEEP/MERGE/PROMOTE/DECAY/DISCARD完全不同的命名体系）"""

    RETAIN = "retain"
    ELEVATE = "elevate"
    WEAKEN = "weaken"
    PURGE = "purge"
    FUSE = "fuse"


# ==================== 记忆切片数据模型 ====================


@dataclass
class TransientSlice:
    """过渡性记忆切片 —— 整合前的原始记忆单元

    与MaiBot MemoryFragment完全不同的字段命名和方法设计。
    """

    slice_id: str
    body: str
    origin_user: str = ""
    birth_ts: float = field(default_factory=time.time)
    salience: float = 0.5
    touch_count: int = 0
    last_touched: float = 0.0
    labels: List[str] = field(default_factory=list)
    vector_repr: Optional[List[float]] = None

    def content_digest(self) -> str:
        """内容摘要指纹（SHA256前14位）"""
        return hashlib.sha256(self.body.encode()).hexdigest()[:14]

    def elapsed_hours(self) -> float:
        """自创建以来经过的小时数"""
        return (time.time() - self.birth_ts) / 3600.0


# ==================== 整合统计 ====================


@dataclass
class ConsolidationTally:
    """一轮整合周期的统计汇总"""

    inspected: int = 0
    elevated: int = 0
    fused: int = 0
    weakened: int = 0
    purged: int = 0
    duration_ms: float = 0.0


# ==================== 调谐参数 ====================


@dataclass
class MergerTuning:
    """整合器调谐参数集

    与MaiBot MergerConfig使用完全不同的参数名称和默认值。
    """

    cycle_interval_sec: float = 2400.0
    batch_ceiling: int = 20
    elevation_bar: float = 0.60
    weakness_bar: float = 0.18
    disposal_bar: float = 0.06
    attenuation_rate: float = 0.90
    age_half_life_hours: float = 20.0
    ephemeral_capacity: int = 120
    archive_capacity: int = 600


# ==================== 瞬时记忆池 ====================


class EphemeralPool:
    """瞬时记忆池 —— 环形缓冲风格的短期存储容器。

    超过容量上限时自动淘汰最老条目。
    支持关键词搜索和批量访问操作。
    """

    def __init__(self, capacity: int = 120):
        self._capacity = capacity
        self._slots: Dict[str, TransientSlice] = {}

    def deposit(self, slc: TransientSlice) -> str:
        """存入一个切片，超容量时淘汰最老的"""
        if len(self._slots) >= self._capacity:
            self._evict_oldest()
        self._slots[slc.slice_id] = slc
        return slc.slice_id

    def touch(self, slice_id: str) -> Optional[TransientSlice]:
        """触摸切片（增加访问计数和显著性）"""
        slc = self._slots.get(slice_id)
        if slc:
            slc.touch_count += 1
            slc.last_touched = time.time()
            slc.salience = min(1.0, slc.salience + 0.04)
        return slc

    def lookup_keywords(
        self, terms: List[str], ceiling: int = 5
    ) -> List[TransientSlice]:
        """按关键词检索匹配的切片"""
        terms_lower = [t.lower() for t in terms]
        candidates: List[Tuple[float, TransientSlice]] = []
        for slc in self._slots.values():
            body_lower = slc.body.lower()
            hits = sum(1 for t in terms_lower if t in body_lower)
            if hits > 0:
                rank = hits * slc.salience * (1.0 + slc.touch_count * 0.08)
                candidates.append((rank, slc))
        candidates.sort(key=lambda p: p[0], reverse=True)
        results = [slc for _, slc in candidates[:ceiling]]
        for slc in results:
            slc.touch_count += 1
            slc.last_touched = time.time()
        return results

    def _evict_oldest(self) -> Optional[str]:
        """淘汰创建时间最早的切片"""
        if not self._slots:
            return None
        oldest_key = min(self._slots, key=lambda k: self._slots[k].birth_ts)
        self._slots.pop(oldest_key)
        logger.debug(f"淘汰瞬时切片: {oldest_key}")
        return oldest_key

    def drain_all(self) -> List[TransientSlice]:
        """清空并返回所有切片"""
        items = list(self._slots.values())
        self._slots.clear()
        return items

    @property
    def occupancy(self) -> int:
        return len(self._slots)

    def snapshot_ids(self) -> List[str]:
        """获取当前所有切片ID列表"""
        return list(self._slots.keys())

    def _raw_entries(self) -> Dict[str, TransientSlice]:
        """内部访问：整合器使用"""
        return self._slots


# ==================== 归档记忆库 ====================


class ArchiveVault:
    """归档记忆库 —— 容量受限的长期存储容器。

    超容量时淘汰显著性最低的条目。
    """

    def __init__(self, capacity: int = 600):
        self._capacity = capacity
        self._depot: Dict[str, TransientSlice] = {}

    def admit(self, slc: TransientSlice) -> None:
        """准入一个切片"""
        if len(self._depot) >= self._capacity:
            self._expel_weakest()
        self._depot[slc.slice_id] = slc

    def access(self, slice_id: str) -> Optional[TransientSlice]:
        """访问归档切片"""
        slc = self._depot.get(slice_id)
        if slc:
            slc.touch_count += 1
            slc.last_touched = time.time()
        return slc

    def _expel_weakest(self) -> Optional[str]:
        """淘汰显著性最低的切片"""
        if not self._depot:
            return None
        weakest_key = min(self._depot, key=lambda k: self._depot[k].salience)
        evicted = self._depot.pop(weakest_key)
        logger.debug(
            f"淘汰归档切片: {weakest_key} (salience={
                evicted.salience:.3f})"
        )
        return weakest_key

    def drain_all(self) -> List[TransientSlice]:
        """清空并返回所有切片"""
        items = list(self._depot.values())
        self._depot.clear()
        return items

    @property
    def occupancy(self) -> int:
        return len(self._depot)

    def enumerate_all(self) -> List[TransientSlice]:
        """枚举所有归档切片"""
        return list(self._depot.values())


# ==================== 切片整合引擎 ====================


class SliceConsolidator:
    """记忆切片整合引擎 —— 评估瞬时记忆并决策提升/衰减/清除。

    核心设计：
    - 清晰度评估采用指数衰减模型（替代MaiBot的线性清晰度）
    - 瞬时池 → 归档库的双层架构（借鉴XBcore的STM/LTM分离）
    - 可注册的提升/丢弃钩子函数
    - 后台守护进程定时执行整合周期
    """

    _shared_instance: Optional["SliceConsolidator"] = None

    @classmethod
    def shared(cls) -> "SliceConsolidator":
        """获取模块级共享实例"""
        if cls._shared_instance is None:
            cls._shared_instance = cls()
        return cls._shared_instance

    def __init__(self, tuning: Optional[MergerTuning] = None):
        self._tuning = tuning or MergerTuning()
        self._ephemeral = EphemeralPool(self._tuning.ephemeral_capacity)
        self._archive = ArchiveVault(self._tuning.archive_capacity)
        self._last_cycle_ts = time.time()
        self._daemon_task: Optional[asyncio.Task] = None
        self._active = False
        self._on_elevated: Optional[Callable[[TransientSlice], None]] = None
        self._on_discarded: Optional[Callable[[TransientSlice], None]] = None

    # -------------------- 切片操作 --------------------

    def ingest_slice(self, slc: TransientSlice) -> str:
        """将切片存入瞬时池"""
        return self._ephemeral.deposit(slc)

    def forge_and_ingest(
        self,
        body: str,
        origin_user: str = "",
        salience: float = 0.5,
        labels: Optional[List[str]] = None,
    ) -> TransientSlice:
        """创建并存入一个新切片"""
        ts_hex = hashlib.md5(body.encode()).hexdigest()[:10]
        unique_id = f"sl_{int(time.time() * 1000)}_{ts_hex}"
        slc = TransientSlice(
            slice_id=unique_id,
            body=body,
            origin_user=origin_user,
            salience=salience,
            labels=labels or [],
        )
        self.ingest_slice(slc)
        return slc

    def touch_slice(self, slice_id: str) -> Optional[TransientSlice]:
        """触摸切片（先查瞬时池，再查归档库）"""
        result = self._ephemeral.touch(slice_id)
        if result:
            return result
        return self._archive.access(slice_id)

    def find_by_keywords(
        self, terms: List[str], ceiling: int = 5
    ) -> List[TransientSlice]:
        """按关键词在瞬时池中检索"""
        return self._ephemeral.lookup_keywords(terms, ceiling)

    # -------------------- 整合核心 --------------------

    async def run_consolidation_cycle(self) -> ConsolidationTally:
        """执行一轮整合评估

        遍历瞬时池中每个切片，计算有效显著性（salience * clarity），
        根据阈值决定：提升归档 / 衰减 / 清除。
        归档库中的条目也会缓慢衰减。
        """
        t0 = time.time()
        tally = ConsolidationTally()
        pending_elevate: List[str] = []
        pending_weaken: List[str] = []
        pending_purge: List[str] = []
        cfg = self._tuning
        raw = self._ephemeral._raw_entries()
        for sid, slc in list(raw.items()):
            tally.inspected += 1
            clarity = self._gauge_clarity(slc)
            effective_salience = slc.salience * clarity
            if effective_salience >= cfg.elevation_bar:
                pending_elevate.append(sid)
            elif effective_salience <= cfg.disposal_bar:
                pending_purge.append(sid)
            elif effective_salience <= cfg.weakness_bar:
                pending_weaken.append(sid)
        for sid in pending_elevate:
            if sid in raw:
                slc = raw.pop(sid)
                self._archive.admit(slc)
                tally.elevated += 1
                if self._on_elevated:
                    self._on_elevated(slc)
        for sid in pending_weaken:
            if sid in raw:
                raw[sid].salience *= cfg.attenuation_rate
                tally.weakened += 1
        for sid in pending_purge:
            if sid in raw:
                discarded = raw.pop(sid)
                tally.purged += 1
                if self._on_discarded:
                    self._on_discarded(discarded)
        for slc in self._archive.enumerate_all():
            slc.salience *= cfg.attenuation_rate**0.5
        self._last_cycle_ts = time.time()
        tally.duration_ms = (time.time() - t0) * 1000.0
        logger.info(
            f"整合周期完成 | 检查={tally.inspected} 提升={tally.elevated} "
            f"衰减={tally.weakened} 清除={tally.purged}"
        )
        return tally

    def _gauge_clarity(self, slc: TransientSlice) -> float:
        """衡量记忆切片当前清晰度（指数衰减模型）

        基于:
        - 年龄衰减：exp(-0.693 * hours / half_life)
        - 触摸加成：最近被访问的切片获得额外清晰度
        """
        hrs = slc.elapsed_hours()
        half_life = self._tuning.age_half_life_hours
        time_factor = math.exp(-0.693 * hrs / max(half_life, 0.1))
        touch_bonus = 0.0
        if slc.last_touched > 0:
            since_touch = (time.time() - slc.last_touched) / 3600.0
            touch_bonus = (
                0.15
                * math.exp(-0.2 * since_touch)
                * min(slc.touch_count, 10)
                / 10.0
            )
        return min(1.0, time_factor + touch_bonus)

    # -------------------- 守护进程 --------------------

    async def launch_daemon(self) -> None:
        """启动后台整合守护进程"""
        if self._active:
            return
        self._active = True
        self._daemon_task = asyncio.create_task(self._cycle_loop())
        logger.info("记忆整合守护进程已启动")

    async def halt_daemon(self) -> None:
        """停止后台整合守护进程"""
        self._active = False
        if self._daemon_task:
            self._daemon_task.cancel()
            try:
                await self._daemon_task
            except asyncio.CancelledError:
                pass
        logger.info("记忆整合守护进程已停止")

    async def _cycle_loop(self) -> None:
        """守护进程主循环"""
        while self._active:
            await asyncio.sleep(self._tuning.cycle_interval_sec)
            if self._active:
                await self.run_consolidation_cycle()

    # -------------------- 钩子注册 --------------------

    def register_elevation_hook(
        self, callback: Callable[[TransientSlice], None]
    ) -> None:
        """注册切片提升到归档时的回调"""
        self._on_elevated = callback

    def register_discard_hook(
        self, callback: Callable[[TransientSlice], None]
    ) -> None:
        """注册切片被丢弃时的回调"""
        self._on_discarded = callback

    # -------------------- 状态查询 --------------------

    @property
    def ephemeral_count(self) -> int:
        return self._ephemeral.occupancy

    @property
    def archive_count(self) -> int:
        return self._archive.occupancy

    def peek_ephemeral(self) -> List[TransientSlice]:
        """窥视瞬时池内容"""
        return list(self._ephemeral._raw_entries().values())

    def peek_archive(self) -> List[TransientSlice]:
        """窥视归档库内容"""
        return self._archive.enumerate_all()

    def flush_ephemeral(self) -> int:
        """清空瞬时池"""
        items = self._ephemeral.drain_all()
        logger.info(f"清空瞬时记忆池: {len(items)}条")
        return len(items)

    def flush_everything(self) -> Tuple[int, int]:
        """清空所有记忆（瞬时 + 归档）"""
        ep = len(self._ephemeral.drain_all())
        ar = len(self._archive.drain_all())
        logger.info(f"清空全部记忆: 瞬时={ep} 归档={ar}")
        return ep, ar


# ==================== 便捷接口 ====================


def acquire_consolidator() -> SliceConsolidator:
    """获取共享整合器实例"""
    return SliceConsolidator.shared()


def deposit_slice(
    body: str, origin: str = "", salience: float = 0.5
) -> TransientSlice:
    """快捷存入一个记忆切片"""
    return SliceConsolidator.shared().forge_and_ingest(body, origin, salience)


def find_slices(terms: List[str], ceiling: int = 5) -> List[TransientSlice]:
    """快捷按关键词检索切片"""
    return SliceConsolidator.shared().find_by_keywords(terms, ceiling)


async def trigger_consolidation() -> ConsolidationTally:
    """快捷触发一轮整合"""
    return await SliceConsolidator.shared().run_consolidation_cycle()


# 短别名（对标MaiBot接口）
get_memory_merger = acquire_consolidator
add_memory = deposit_slice
search_memories = find_slices
run_consolidation = trigger_consolidation

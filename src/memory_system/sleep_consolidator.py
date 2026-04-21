import time
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("nightly_tidy")


# ==================== 夜间整合器 ====================


class NightlyMemoryTidier:
    """夜间记忆整合器 —— 在凌晨时段自动清理和升级当日记忆。

    设计思路：
    - 凌晨窗口内（默认1:30-4:30）自动触发一次整理
    - 逐会话扫描当日conversation记忆，按显著性三级分流：
        * 高显著 → 转为fact类型，提高重要性
        * 中显著 → 压缩内容，保留摘要
        * 低显著 → 直接删除
    - 整理完成后重置情绪状态（降低烦燥、清除疲劳）
    - 每日最多执行一次（幂等保护）

    融合三源：
    - MaiBot: 整理逻辑的功能基线
    - XBcore: 无专用sleep模块，但其consolidator的去重思路可借鉴
    - MIMiaoCore: 恢复patience的额外情绪重置
    """

    def __init__(self, upstream_manager=None):
        self._upstream = upstream_manager
        self._in_progress = False
        self._finished_date: Optional[str] = None
        self._patrol_task: Optional[asyncio.Task] = None
        self._window_begin_hour = 1
        self._window_begin_minute = 30
        self._window_end_hour = 4
        self._window_end_minute = 30
        self._high_bar = 0.58
        self._low_bar = 0.28
        self._daily_scan_cap = 120
        self._last_report: Dict[str, int] = {}

    def _resolve_record_table(self):
        """惰性获取记忆数据表"""
        try:
            from src.common.database.database_model import MemoryRecord

            return MemoryRecord
        except ImportError:
            return None

    @staticmethod
    def _resolve_timestamp_field(rec_table):
        """兼容时间字段：优先新表结构 birth_ts，回退旧字段 created_at。"""
        if hasattr(rec_table, "birth_ts"):
            return rec_table.birth_ts
        if hasattr(rec_table, "created_at"):
            return rec_table.created_at
        return None

    @staticmethod
    def _resolve_category_field(rec_table):
        """兼容类型字段：优先新表结构 entry_category，回退旧字段 memory_type。"""
        if hasattr(rec_table, "entry_category"):
            return rec_table.entry_category
        if hasattr(rec_table, "memory_type"):
            return rec_table.memory_type
        return None

    @staticmethod
    def _read_significance(record: Any) -> float:
        raw = getattr(record, "significance", None)
        if raw is None:
            raw = getattr(record, "importance", 0.5)
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _write_significance(record: Any, value: float) -> None:
        score = max(0.0, min(1.0, float(value)))
        if hasattr(record, "significance"):
            record.significance = score
        if hasattr(record, "importance"):
            record.importance = score

    @staticmethod
    def _write_category(record: Any, value: str) -> None:
        if hasattr(record, "entry_category"):
            record.entry_category = value
        if hasattr(record, "memory_type"):
            record.memory_type = value

    @staticmethod
    def _write_summary(record: Any, value: str) -> None:
        if hasattr(record, "digest"):
            record.digest = value
        if hasattr(record, "summary"):
            record.summary = value

    def within_tidy_window(self) -> bool:
        """判断当前是否处于整理时间窗口"""
        now = datetime.now()
        current_minutes = now.hour * 60 + now.minute
        begin_minutes = (
            self._window_begin_hour * 60 + self._window_begin_minute
        )
        end_minutes = self._window_end_hour * 60 + self._window_end_minute
        return begin_minutes <= current_minutes < end_minutes

    def already_done_today(self) -> bool:
        """判断今天是否已经执行过整理"""
        today_tag = datetime.now().strftime("%Y-%m-%d")
        return self._finished_date == today_tag

    async def attempt_nightly_tidy(self) -> bool:
        """尝试执行夜间整理（返回是否实际执行了）"""
        if not self.within_tidy_window():
            return False
        if self.already_done_today():
            return False
        if self._in_progress:
            return False
        await self.execute_tidy_cycle()
        return True

    async def execute_tidy_cycle(self) -> None:
        """执行一轮完整的夜间整理流程"""
        if self._in_progress:
            return
        self._in_progress = True
        today_tag = datetime.now().strftime("%Y-%m-%d")
        self._finished_date = today_tag
        logger.info("[夜间整理] 开始凌晨记忆清扫...")
        try:
            channel_ids = await self._enumerate_active_channels()
            cumulative = {
                "preserved": 0,
                "condensed": 0,
                "discarded": 0,
                "channels": len(channel_ids),
            }
            for cid in channel_ids:
                ch_result = await self._tidy_single_channel(cid)
                cumulative["preserved"] += ch_result.get("preserved", 0)
                cumulative["condensed"] += ch_result.get("condensed", 0)
                cumulative["discarded"] += ch_result.get("discarded", 0)
            self._last_report = cumulative
            logger.info(
                f"[夜间整理] 清扫完成 — "
                f"频道数={cumulative['channels']} "
                f"保留={cumulative['preserved']} "
                f"压缩={cumulative['condensed']} "
                f"丢弃={cumulative['discarded']}"
            )
            await self._soothe_emotional_states()
        except Exception as exc:
            logger.error(f"[夜间整理] 清扫异常: {exc}")
        finally:
            self._in_progress = False

    async def _enumerate_active_channels(self) -> List[str]:
        """枚举活跃的会话频道ID列表"""
        rec_table = self._resolve_record_table()
        if not rec_table:
            return []
        try:
            rows = rec_table.select(rec_table.stream_id).distinct().limit(50)
            return [r.stream_id for r in rows]
        except Exception as exc:
            logger.error(f"[夜间整理] 枚举频道失败: {exc}")
            return []

    async def _tidy_single_channel(self, channel_id: str) -> Dict[str, int]:
        """清理单个频道的当日conversation记忆"""
        tally = {"preserved": 0, "condensed": 0, "discarded": 0}
        rec_table = self._resolve_record_table()
        if not rec_table:
            return tally
        try:
            ts_field = self._resolve_timestamp_field(rec_table)
            category_field = self._resolve_category_field(rec_table)
            if ts_field is None or category_field is None:
                logger.error(
                    f"[夜间整理] 频道 {channel_id[:8]}... 记忆表字段不完整，缺少时间或类型字段"
                )
                return tally

            now_ts = time.time()
            day_boundary = now_ts - 86400
            day_records = list(
                rec_table.select()
                .where(
                    (rec_table.stream_id == channel_id)
                    & (ts_field >= day_boundary)
                    & (category_field == "conversation")
                )
                .order_by(ts_field.desc())
                .limit(self._daily_scan_cap)
            )
            if not day_records:
                return tally
            logger.debug(
                f"[夜间整理] 频道 {channel_id[:8]}... 有 {len(day_records)} 条待整理"
            )
            for record in day_records:
                sig = self._read_significance(record)
                if sig >= self._high_bar:
                    self._write_category(record, "fact")
                    self._write_significance(record, min(1.0, sig + 0.12))
                    record.save()
                    tally["preserved"] += 1
                elif sig >= self._low_bar:
                    content_text = record.content or ""
                    if len(content_text) > 160:
                        self._write_summary(record, content_text[:110])
                        self._write_category(record, "compressed")
                        record.save()
                        tally["condensed"] += 1
                    else:
                        tally["preserved"] += 1
                else:
                    record.delete_instance()
                    tally["discarded"] += 1
            return tally
        except Exception as exc:
            logger.error(
                f"[夜间整理] 频道 {channel_id[:8]}... 处理异常: {exc}"
            )
            return tally

    async def _soothe_emotional_states(self) -> None:
        """整理完毕后舒缓各频道的情绪状态"""
        try:
            from src.chat.heart_flow.emotion_stream import (
                get_channel_mood_tracker,
            )

            tracker = get_channel_mood_tracker()
            for _sid, ledger in tracker._books.items():
                ledger.vexation = max(0.0, ledger.vexation - 45)
                ledger.weariness = 0.0
                ledger.vitality = min(100.0, ledger.vitality + 25)
            logger.debug("[夜间整理] 情绪舒缓完成")
        except ImportError:
            pass
        except Exception as exc:
            logger.debug(f"[夜间整理] 情绪舒缓失败: {exc}")

    def fetch_last_report(self) -> Dict[str, int]:
        """获取上一次整理的统计报告"""
        return self._last_report.copy()

    def status_overview(self) -> Dict[str, Any]:
        """获取整理器当前状态概览"""
        now = datetime.now()
        return {
            "in_progress": self._in_progress,
            "within_window": self.within_tidy_window(),
            "done_today": self.already_done_today(),
            "finished_date": self._finished_date,
            "current_hour": now.hour,
            "current_minute": now.minute,
            "window": (
                f"{self._window_begin_hour}:{self._window_begin_minute:02d}"
                f" - {self._window_end_hour}:{self._window_end_minute:02d}"
            ),
            "last_report": self._last_report,
        }


# ==================== 定时巡逻器 ====================


class TidyPatroller:
    """夜间整理巡逻器 —— 周期性检查是否该执行整理。"""

    def __init__(self, tidier: NightlyMemoryTidier):
        self._tidier = tidier
        self._alive = False
        self._loop_task: Optional[asyncio.Task] = None
        self._patrol_gap_sec = 2400

    async def activate(self) -> None:
        """启动巡逻"""
        if self._alive:
            return
        self._alive = True
        self._loop_task = asyncio.create_task(self._patrol_loop())
        logger.info("[巡逻器] 夜间整理巡逻已启动，将在凌晨自动执行")

    async def deactivate(self) -> None:
        """停止巡逻"""
        self._alive = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    async def _patrol_loop(self) -> None:
        """巡逻主循环"""
        while self._alive:
            try:
                await self._tidier.attempt_nightly_tidy()
            except Exception as exc:
                logger.error(f"[巡逻器] 周期检查异常: {exc}")
            await asyncio.sleep(self._patrol_gap_sec)


# ==================== 单例管理 ====================

_tidier_ref: Optional[NightlyMemoryTidier] = None
_patroller_ref: Optional[TidyPatroller] = None


def acquire_nightly_tidier(upstream=None) -> Optional[NightlyMemoryTidier]:
    """获取或创建夜间整合器单例"""
    global _tidier_ref
    if _tidier_ref is None and upstream is not None:
        _tidier_ref = NightlyMemoryTidier(upstream)
    return _tidier_ref


def acquire_tidy_patroller(
    tidier: Optional[NightlyMemoryTidier] = None,
) -> Optional[TidyPatroller]:
    """获取或创建巡逻器单例"""
    global _patroller_ref
    if _patroller_ref is None and tidier is not None:
        _patroller_ref = TidyPatroller(tidier)
    return _patroller_ref


async def bootstrap_nightly_system(upstream) -> Optional[TidyPatroller]:
    """一键启动夜间整理系统"""
    tidier = acquire_nightly_tidier(upstream)
    patroller = acquire_tidy_patroller(tidier)
    if patroller:
        await patroller.activate()
    return patroller

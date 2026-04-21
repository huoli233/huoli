import asyncio
import time
import logging
from typing import Callable, Awaitable
from src.chat.heart_flow.dimension_protocol import (
    DimensionBase,
    DimensionRegistry,
    DimensionKeyBuilder,
    DimensionScope,
    EventContext,
)
from src.chat.heart_flow.signal_domains import (
    DecisionSignalBundle,
    RuntimeCalibrationProfile,
    RuntimeCalibrationBand,
)

logger = logging.getLogger("dimension_dispatcher")

# 维度名称 → VoteCollection 字段名的映射
_VOTE_FIELD_MAP: dict[str, str] = {
    "emotion_axis": "emotion_baseline",
    "fondness_trust": "rapport_trust",
    "dislike_registry": "boundary_block",
    "frequency_control": "tempo_control",
    "user_state": "counterparty_readiness",
    "group_atmosphere": "group_context",
    "energy_chain": "resource_ledger",
    "social_value": "social_balance",
    "heart_state": "pending_engagement",
    "trauma": "trauma_load",
    "surface_mask": "surface_mask",
}


class DimensionDispatcher:
    """
    多维状态系统的统一调度器。
    职责：
      1. 管理所有维度实例的生命周期（注册/初始化/销毁）
      2. 按各维度的 tick_interval 周期驱动恢复/衰减
      3. 广播事件（消息到达、回复完成等）给所有维度
      4. 收集所有维度投票，打包为 VoteCollection 交给决策网关
      5. 定期触发需要持久化的维度执行序列化
      6. 系统启动时协调离线校准

    不做任何业务计算，只做调度和转发。
    """

    _instance: "DimensionDispatcher | None" = None
    _singleton_lock: asyncio.Lock | None = None

    @classmethod
    async def get_instance(cls) -> "DimensionDispatcher":
        """获取调度器单例"""
        if cls._instance is not None:
            return cls._instance
        if cls._singleton_lock is None:
            cls._singleton_lock = asyncio.Lock()
        async with cls._singleton_lock:
            if cls._instance is not None:
                return cls._instance
            cls._instance = cls()
            return cls._instance

    def __init__(self):
        self._registry = DimensionRegistry()
        self._running = False
        self._tick_task: asyncio.Task | None = None
        self._persist_task: asyncio.Task | None = None
        self._last_tick_times: dict[str, float] = {}
        self._last_persist_time: float = 0.0
        self._startup_timestamp: float = 0.0
        # 持久化回调：外部注入，用于将维度状态写入数据库
        self._persist_callback: Callable[[str, str, dict], Awaitable[None]] | None = None
        # 加载回调：外部注入，用于从数据库读取维度状态
        self._load_callback: Callable[[str, str], Awaitable[dict | None]] | None = None
        # 事件处理统计
        self._event_counter: int = 0
        self._tick_counter: int = 0

    # ============================================================
    # 维度注册
    # ============================================================
    def register_dimension(self, dim: DimensionBase):
        """注册维度实例到调度器"""
        self._registry.register(dim)
        self._last_tick_times[dim.dimension_name] = time.monotonic()
        logger.info(
            "维度注册成功: %s (作用域=%s, 持久化=%s, tick间隔=%.1fs)",
            dim.dimension_name,
            dim.scope.value,
            dim.needs_persistence,
            dim.tick_interval_sec,
        )

    def get_dimension(self, name: str) -> DimensionBase | None:
        """按名称获取维度实例"""
        return self._registry.get(name)

    @property
    def registered_count(self) -> int:
        return self._registry.count

    @property
    def dimension_names(self) -> list[str]:
        return self._registry.dimension_names()

    # ============================================================
    # 外部回调注入
    # ============================================================
    def set_persist_callback(
        self, callback: Callable[[str, str, dict], Awaitable[None]]
    ):
        """
        设置持久化写入回调。
        callback 签名: async def save(dimension_name, composite_key, data_dict) -> None
        """
        self._persist_callback = callback

    def set_load_callback(
        self, callback: Callable[[str, str], Awaitable[dict | None]]
    ):
        """
        设置持久化加载回调。
        callback 签名: async def load(dimension_name, composite_key) -> dict | None
        """
        self._load_callback = callback

    # ============================================================
    # 初始化和生命周期
    # ============================================================
    async def initialize(self, last_shutdown_time: float = 0.0):
        """
        初始化所有已注册维度。
        last_shutdown_time: 上次关机的Unix时间戳（用于计算离线时长）
        """
        self._startup_timestamp = time.time()
        logger.info(
            "调度器初始化开始，已注册 %d 个维度",
            self._registry.count,
        )
        # 加载持久化数据
        await self._load_all_persistent_states()
        # 调用各维度的 initialize
        for dim in self._registry.all_dimensions():
            try:
                dim.initialize()
            except Exception as exc:
                logger.error("维度 %s 初始化失败: %s", dim.dimension_name, exc)
        # 计算离线时长并执行校准
        if last_shutdown_time > 0:
            offline_sec = self._startup_timestamp - last_shutdown_time
            if offline_sec > 0:
                await self._run_calibration(offline_sec)
        logger.info("调度器初始化完成")

    async def start(self):
        """启动周期性 tick 和持久化任务"""
        if self._running:
            return
        self._running = True
        self._tick_task = asyncio.create_task(self._tick_loop())
        self._persist_task = asyncio.create_task(self._persist_loop())
        logger.info("调度器已启动（tick循环 + 持久化循环）")

    async def stop(self):
        """停止调度器，最终持久化一次"""
        if not self._running:
            return
        self._running = False
        if self._tick_task and not self._tick_task.done():
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                logger.debug("tick循环已取消")
        if self._persist_task and not self._persist_task.done():
            self._persist_task.cancel()
            try:
                await self._persist_task
            except asyncio.CancelledError:
                logger.debug("持久化循环已取消")
        # 最终持久化
        await self._persist_all()
        logger.info(
            "调度器已停止（tick执行 %d 次，事件处理 %d 次）",
            self._tick_counter,
            self._event_counter,
        )

    # ============================================================
    # 事件广播
    # ============================================================
    async def broadcast_event(self, ctx: EventContext):
        """
        将事件广播给所有维度。
        每个维度根据自身作用域过滤是否处理此事件。
        """
        self._event_counter += 1
        for dim in self._registry.all_dimensions():
            if not self._event_matches_scope(dim, ctx):
                continue
            try:
                dim.on_event(ctx)
            except Exception as exc:
                logger.error(
                    "维度 %s 处理事件 %s 失败: %s",
                    dim.dimension_name,
                    ctx.event_type,
                    exc,
                )

    def _event_matches_scope(self, dim: DimensionBase, ctx: EventContext) -> bool:
        """判断事件是否在维度的作用域内"""
        scope = dim.scope
        if scope == DimensionScope.GLOBAL:
            return True
        if scope == DimensionScope.PER_CHANNEL:
            return bool(ctx.channel_id)
        if scope == DimensionScope.PER_USER:
            return bool(ctx.user_id)
        if scope == DimensionScope.PER_USER_CHANNEL:
            return bool(ctx.user_id) and bool(ctx.channel_id)
        return True

    # ============================================================
    # 投票收集（核心方法：决策网关调用此方法）
    # ============================================================
    def collect_votes(self, ctx: EventContext) -> DecisionSignalBundle:
        """
        收集所有维度的独立投票，打包为 DecisionSignalBundle。
        决策网关通过此方法获取 11 个维度的投票，
        然后按乘法组合概率、合并态度标签、检查强制触发/拒绝。
        """
        collection = DecisionSignalBundle()
        for dim in self._registry.all_dimensions():
            try:
                vote = dim.vote(ctx)
                field_name = _VOTE_FIELD_MAP.get(dim.dimension_name)
                if field_name and hasattr(collection, field_name):
                    setattr(collection, field_name, vote)
            except Exception as exc:
                logger.error(
                    "维度 %s 投票失败（使用默认投票）: %s",
                    dim.dimension_name,
                    exc,
                )
        return collection

    # ============================================================
    # 周期性 tick 循环
    # ============================================================
    async def _tick_loop(self):
        """每秒执行一次，驱动各维度的恢复/衰减"""
        while self._running:
            try:
                await asyncio.sleep(1.0)
                now = time.monotonic()
                for dim in self._registry.all_dimensions():
                    last = self._last_tick_times.get(dim.dimension_name, now)
                    elapsed = now - last
                    if elapsed < dim.tick_interval_sec:
                        continue
                    try:
                        result = dim.tick(elapsed)
                        self._last_tick_times[dim.dimension_name] = now
                        self._tick_counter += 1
                        if result and result.updated:
                            logger.debug(
                                "维度 %s tick: %s",
                                dim.dimension_name,
                                result.summary,
                            )
                    except Exception as exc:
                        logger.error(
                            "维度 %s tick 异常: %s",
                            dim.dimension_name,
                            exc,
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("tick循环异常: %s", exc)

    # ============================================================
    # 周期性持久化循环
    # ============================================================
    async def _persist_loop(self):
        """定期将需要持久化的维度状态写入存储"""
        while self._running:
            try:
                # 找出最短的持久化间隔作为循环周期
                intervals = [
                    d.persist_interval_sec
                    for d in self._registry.persistent_dimensions()
                ]
                sleep_sec = min(intervals) if intervals else 30.0
                await asyncio.sleep(sleep_sec)
                await self._persist_all()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("持久化循环异常: %s", exc)

    async def _persist_all(self):
        """执行一次全量持久化"""
        if not self._persist_callback:
            return
        now = time.time()
        for dim in self._registry.persistent_dimensions():
            try:
                data = dim.serialize()
                if not data:
                    continue
                composite_key = DimensionKeyBuilder.build(dim.scope)
                await self._persist_callback(
                    dim.dimension_name,
                    composite_key,
                    data,
                )
            except Exception as exc:
                logger.error(
                    "维度 %s 持久化失败: %s",
                    dim.dimension_name,
                    exc,
                )
        self._last_persist_time = now

    # ============================================================
    # 持久化加载
    # ============================================================
    async def _load_all_persistent_states(self):
        """从存储加载所有需要持久化的维度状态"""
        if not self._load_callback:
            logger.warning("未设置加载回调，跳过持久化状态加载")
            return
        for dim in self._registry.persistent_dimensions():
            try:
                composite_key = DimensionKeyBuilder.build(dim.scope)
                data = await self._load_callback(dim.dimension_name, composite_key)
                if data:
                    dim.deserialize(data)
                    logger.debug("维度 %s 状态加载成功", dim.dimension_name)
            except Exception as exc:
                logger.error(
                    "维度 %s 状态加载失败: %s",
                    dim.dimension_name,
                    exc,
                )

    # ============================================================
    # 离线校准
    # ============================================================
    async def _run_calibration(self, offline_seconds: float):
        """
        根据离线时长计算校准等级，然后协调各维度执行校准。
        """
        report = self._compute_calibration_report(offline_seconds)
        if report.grade == RuntimeCalibrationBand.SKIP:
            logger.info("离线时间 %.0f 秒，跳过校准", offline_seconds)
            return
        logger.info(
            "执行离线校准: 等级=%s, 离线 %.1f 小时",
            report.grade.value,
            offline_seconds / 3600,
        )
        for dim in self._registry.all_dimensions():
            try:
                dim.calibrate(offline_seconds)
            except Exception as exc:
                logger.error(
                    "维度 %s 校准失败: %s",
                    dim.dimension_name,
                    exc,
                )

    @staticmethod
    def _compute_calibration_report(offline_seconds: float) -> RuntimeCalibrationProfile:
        """根据离线时长计算校准参数"""
        report = RuntimeCalibrationProfile(offline_seconds=offline_seconds)
        hours = offline_seconds / 3600
        # 不足5分钟不校准
        if offline_seconds < 300:
            report.grade = RuntimeCalibrationBand.SKIP
            return report
        # 5分钟~2小时：轻度校准
        if hours < 2:
            report.grade = RuntimeCalibrationBand.LIGHT
            ratio = offline_seconds / 7200
            report.emotion_decay_factor = 0.1 * ratio
            report.fondness_decay_factor = 0.02 * ratio
            report.energy_recovery_factor = 0.3 * ratio
            report.trauma_heal_factor = 0.05 * ratio
            report.mask_fatigue_recovery = 0.2 * ratio
            return report
        # 2~12小时：中度校准
        if hours < 12:
            report.grade = RuntimeCalibrationBand.MODERATE
            ratio = min(hours / 12, 1.0)
            report.emotion_decay_factor = 0.3 + 0.4 * ratio
            report.fondness_decay_factor = 0.05 + 0.1 * ratio
            report.energy_recovery_factor = 0.5 + 0.3 * ratio
            report.trauma_heal_factor = 0.1 + 0.15 * ratio
            report.mask_fatigue_recovery = 0.5 + 0.3 * ratio
            report.frequency_should_reset = True
            return report
        # 12~48小时：深度校准
        if hours < 48:
            report.grade = RuntimeCalibrationBand.DEEP
            ratio = min((hours - 12) / 36, 1.0)
            report.emotion_decay_factor = 0.7 + 0.2 * ratio
            report.fondness_decay_factor = 0.15 + 0.15 * ratio
            report.energy_recovery_factor = 0.8 + 0.15 * ratio
            report.trauma_heal_factor = 0.25 + 0.25 * ratio
            report.mask_fatigue_recovery = 0.8 + 0.15 * ratio
            report.frequency_should_reset = True
            return report
        # 超过48小时：完全重置
        report.grade = RuntimeCalibrationBand.FULL_RESET
        report.emotion_decay_factor = 1.0
        report.fondness_decay_factor = 0.3
        report.energy_recovery_factor = 1.0
        report.trauma_heal_factor = 0.5
        report.mask_fatigue_recovery = 1.0
        report.frequency_should_reset = True
        return report

    # ============================================================
    # 状态查询（调试和面板用）
    # ============================================================
    def get_all_summaries(self) -> dict[str, dict]:
        """获取所有维度的状态摘要（用于调试面板）"""
        result = {}
        for dim in self._registry.all_dimensions():
            try:
                result[dim.dimension_name] = dim.get_state_summary()
            except Exception as exc:
                result[dim.dimension_name] = {"error": str(exc)}
        return result

    def get_dispatcher_stats(self) -> dict:
        """获取调度器自身的运行统计"""
        return {
            "running": self._running,
            "registered_dimensions": self._registry.count,
            "dimension_names": self._registry.dimension_names(),
            "total_ticks": self._tick_counter,
            "total_events": self._event_counter,
            "uptime_sec": time.time() - self._startup_timestamp
            if self._startup_timestamp
            else 0,
        }

    # ============================================================
    # 维度重置
    # ============================================================
    async def reset_dimension(
        self, name: str, user_id: str = "", channel_id: str = ""
    ):
        """重置指定维度的状态（管理员指令等场景）"""
        dim = self._registry.get(name)
        if not dim:
            logger.warning("尝试重置不存在的维度: %s", name)
            return
        dim.reset(user_id, channel_id)
        logger.info(
            "维度 %s 已重置 (user=%s, channel=%s)",
            name,
            user_id or "*",
            channel_id or "*",
        )

    async def reset_all_for_user(self, user_id: str, channel_id: str = ""):
        """重置指定用户在指定频道下的所有维度状态"""
        for dim in self._registry.all_dimensions():
            try:
                dim.reset(user_id, channel_id)
            except Exception as exc:
                logger.error(
                    "维度 %s 重置失败 (user=%s): %s",
                    dim.dimension_name,
                    user_id,
                    exc,
                )

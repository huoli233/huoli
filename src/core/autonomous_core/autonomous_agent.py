import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.common.database.slot_storage import load_slot, save_slot
from src.common.logger import get_logger

logger = get_logger("autonomous_agent")


# ---------------------------------------------------------------------------
#  持久化状态
# ---------------------------------------------------------------------------


@dataclass
class AgentPersistence:
    """可序列化的代理快照，用于跨重启恢复。"""

    cycle_ms: int = 850
    monitored_channels: List[str] = field(default_factory=list)
    cumulative_ticks: int = 0
    last_persist_ts: float = 0.0
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_ms": self.cycle_ms,
            "monitored_channels": self.monitored_channels,
            "cumulative_ticks": self.cumulative_ticks,
            "last_persist_ts": self.last_persist_ts,
            "custom_params": self.custom_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPersistence":
        return cls(
            cycle_ms=data.get("cycle_ms", 850),
            monitored_channels=data.get("monitored_channels", []),
            cumulative_ticks=data.get("cumulative_ticks", 0),
            last_persist_ts=data.get("last_persist_ts", 0.0),
            custom_params=data.get("custom_params", {}),
        )


_PERSIST_SLOT_KEY = "autonomous_agent:state"


def _resolve_persist_path() -> str:
    """兼容旧字段：返回数据库槽位标识。"""
    return f"Huoli.db:{_PERSIST_SLOT_KEY}"


# ---------------------------------------------------------------------------
#  频道评估结果
# ---------------------------------------------------------------------------


@dataclass
class ChannelVerdict:
    """单频道周期评估结论。"""

    channel_id: str
    should_act: bool = False
    reason: str = ""
    priority: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  自主代理
# ---------------------------------------------------------------------------


class AutonomousCore:
    """以 cycle_ms 毫秒为节拍的自治轮询代理。

    生命周期：
        create() → boot() → [轮询循环] → halt()
    """

    _instance: Optional["AutonomousCore"] = None

    @classmethod
    def get(cls) -> Optional["AutonomousCore"]:
        return cls._instance

    @classmethod
    def create(cls, cycle_ms: int = 850) -> "AutonomousCore":
        if cls._instance is not None:
            return cls._instance
        obj = cls(cycle_ms)
        cls._instance = obj
        return obj

    def __init__(self, cycle_ms: int = 850):
        self._state = AgentPersistence(cycle_ms=cycle_ms)
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._channel_registry: Set[str] = set()
        self._tick_counter: int = 0
        self._persist_every_n: int = 100
        self._save_path = _resolve_persist_path()

    # ==================================================================
    #  启停
    # ==================================================================

    async def boot(self) -> None:
        """加载持久化状态并启动轮询。"""
        self._load_state()
        self._running = True
        self._loop_task = asyncio.create_task(
            self._main_loop(), name="autonomous_core_loop"
        )
        logger.info(
            f"[自主代理] 启动完成，周期={self._state.cycle_ms}ms，"
            f"恢复 {len(self._state.monitored_channels)} 个频道"
        )

    async def halt(self) -> None:
        """安全停止轮询并持久化。"""
        self._running = False
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._persist_state()
        logger.info("[自主代理] 已停止并持久化")

    @property
    def is_running(self) -> bool:
        return self._running

    # ==================================================================
    #  频道注册
    # ==================================================================

    def register_channel(self, channel_id: str) -> None:
        if channel_id not in self._channel_registry:
            self._channel_registry.add(channel_id)
            if channel_id not in self._state.monitored_channels:
                self._state.monitored_channels.append(channel_id)
            logger.debug(f"[自主代理] 注册频道 {channel_id[:8]}")

    def unregister_channel(self, channel_id: str) -> None:
        self._channel_registry.discard(channel_id)
        if channel_id in self._state.monitored_channels:
            self._state.monitored_channels.remove(channel_id)
        logger.debug(f"[自主代理] 注销频道 {channel_id[:8]}")

    def list_channels(self) -> List[str]:
        return list(self._channel_registry)

    # ==================================================================
    #  主循环
    # ==================================================================

    async def _main_loop(self) -> None:
        interval = self._state.cycle_ms / 1000.0
        while self._running:
            try:
                await asyncio.sleep(interval)
                self._tick_counter += 1
                self._state.cumulative_ticks += 1
                verdicts = await self._evaluate_all_channels()
                actionable = [v for v in verdicts if v.should_act]
                if actionable:
                    actionable.sort(key=lambda v: v.priority, reverse=True)
                    for vd in actionable:
                        await self._execute_verdict(vd)
                if self._tick_counter % self._persist_every_n == 0:
                    self._persist_state()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[自主代理] 主循环异常: {exc}")
                await asyncio.sleep(interval * 5)

    # ==================================================================
    #  频道评估
    # ==================================================================

    async def _evaluate_all_channels(self) -> List[ChannelVerdict]:
        results: List[ChannelVerdict] = []
        for cid in list(self._channel_registry):
            try:
                vd = await self._evaluate_single(cid)
                results.append(vd)
            except Exception as exc:
                logger.debug(f"[自主代理] 评估 {cid[:8]} 异常: {exc}")
        return results

    async def _evaluate_single(self, channel_id: str) -> ChannelVerdict:
        """评估单个频道是否需要主动行为。"""
        vd = ChannelVerdict(channel_id=channel_id)
        # 1. 检查驱力追踪器
        try:
            from src.core.autonomous_core.intrinsic_drive import (
                get_inner_urge_tracker,
            )

            tracker = get_inner_urge_tracker()
            status = tracker.peek_status(channel_id)
            assertiveness = status.get("assertiveness", 0.0)
            monitoring = status.get("monitoring", False)
            if assertiveness > 0.7 and not monitoring:
                vd.should_act = True
                vd.reason = f"驱力意愿高({assertiveness:.2f})但无监控"
                vd.priority = 2
                vd.extra["source"] = "intrinsic_drive"
                return vd
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # 2. 可选：静默时长弱触发
        try:
            from src.chat.proactive.silence_watcher import get_quiet_monitor

            silence_sec = float(
                get_quiet_monitor().measure_silence_sec(channel_id) or 0.0
            )
            if silence_sec >= 900:
                vd.should_act = True
                vd.reason = f"静默时长触发({silence_sec:.0f}s)"
                vd.priority = 1
                vd.extra["source"] = "silence_probe"
                return vd
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return vd

    # ==================================================================
    #  行为执行
    # ==================================================================

    async def _execute_verdict(self, vd: ChannelVerdict) -> None:
        """将评估结论转化为实际主链唤醒。"""
        source = vd.extra.get("source", "")
        logger.info(
            f"[自主代理] 执行 {vd.channel_id[:8]} | {vd.reason} | 来源={source}"
        )
        if source == "intrinsic_drive":
            try:
                from src.core.autonomous_core.intrinsic_drive import (
                    get_inner_urge_tracker,
                )

                await get_inner_urge_tracker().on_user_input(vd.channel_id)
            except Exception as exc:
                logger.error(f"[自主代理] 驱力重启失败: {exc}")
        try:
            from src.chat.heart_flow.heartflow import heartflow

            await heartflow.get_or_create_heartflow_chat(vd.channel_id)
            heartflow.touch(vd.channel_id)
        except Exception as exc:
            logger.error(f"[自主代理] 主链唤醒失败: {exc}")

    # ==================================================================
    #  持久化
    # ==================================================================

    def _persist_state(self) -> None:
        self._state.last_persist_ts = time.time()
        self._state.monitored_channels = list(self._channel_registry)
        try:
            save_slot(_PERSIST_SLOT_KEY, self._state.to_dict(), ttl_days=3650)
            logger.debug(
                f"[自主代理] 状态已保存 (总tick={
                    self._state.cumulative_ticks})"
            )
        except Exception as exc:
            logger.warning(f"[自主代理] 持久化失败: {exc}")

    def _load_state(self) -> None:
        data = load_slot(_PERSIST_SLOT_KEY)
        if not isinstance(data, dict):
            logger.debug("[自主代理] 无历史状态，使用默认值")
            return
        try:
            self._state = AgentPersistence.from_dict(data)
            for cid in self._state.monitored_channels:
                self._channel_registry.add(cid)
            logger.info(
                f"[自主代理] 加载状态完成，周期={self._state.cycle_ms}"
            )
        except Exception as exc:
            logger.warning(f"[自主代理] 状态加载失败: {exc}")

    # ==================================================================
    #  参数调整
    # ==================================================================

    def set_cycle(self, ms: int) -> None:
        self._state.cycle_ms = max(100, ms)
        logger.info(f"[自主代理] 周期调整为 {self._state.cycle_ms}ms")

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "cycle_ms": self._state.cycle_ms,
            "tick_count": self._tick_counter,
            "cumulative_ticks": self._state.cumulative_ticks,
            "channel_count": len(self._channel_registry),
            "channels": list(self._channel_registry)[:20],
        }

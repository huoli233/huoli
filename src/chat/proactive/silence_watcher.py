import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from src.common.logger import get_logger
from src.common.task_utils import safe_create_task
from src.config.core_config_engine import get_core_config

logger = get_logger("quiet_patrol")


@dataclass
class ChannelPulseRecord:
    """单个频道的活动脉搏记录。"""

    channel_id: str = ""
    latest_any_ts: float = 0.0
    countdown_handle: Optional[asyncio.TimerHandle] = None
    countdown_minutes: int = 0
    countdown_reset_ts: float = 0.0
    frozen: bool = False


class QuietPeriodMonitor:
    """静默期巡查器。

    只保留主链实际使用的三项能力：
      1. 记录频道活动
      2. 统计静默秒数
      3. 倒计时结束后回调上层唤醒
    """

    _solo: Optional["QuietPeriodMonitor"] = None

    @classmethod
    def instance(cls) -> "QuietPeriodMonitor":
        if cls._solo is None:
            cls._solo = cls()
        return cls._solo

    @classmethod
    def teardown(cls) -> None:
        if cls._solo is not None:
            cls._solo.disband_all()
        cls._solo = None

    def __init__(self):
        self._pulse_table: Dict[str, ChannelPulseRecord] = {}
        self._on_quiet_callback: Optional[Callable] = None
        self._param_cache: Dict[str, Any] = {}
        self._param_ts: float = 0.0
        self._refresh_params()
        logger.info("[巡查器] 静默期巡查器启动")

    def _refresh_params(self) -> None:
        now = time.time()
        if now - self._param_ts < 60.0:
            return
        hub = get_core_config()
        blk = hub.silence_detection_block()
        self._param_cache = {
            "default_minutes": int(blk.get("default_idle_minutes", 10)),
        }
        self._param_ts = now

    def _cp(self, key: str) -> Any:
        self._refresh_params()
        return self._param_cache[key]

    def attach_quiet_handler(self, handler: Callable) -> None:
        """绑定静默触发回调。"""
        self._on_quiet_callback = handler
        logger.debug("[巡查器] 已绑定静默回调")

    def _ensure_record(self, channel_id: str) -> ChannelPulseRecord:
        if channel_id not in self._pulse_table:
            self._pulse_table[channel_id] = ChannelPulseRecord(
                channel_id=channel_id
            )
        return self._pulse_table[channel_id]

    def note_activity(
        self,
        channel_id: str,
        speaker_id: Optional[str] = None,
        is_bot: bool = False,
        idle_minutes: Optional[int] = None,
        one_shot_callback: Optional[Callable] = None,
    ) -> None:
        """记录频道活动并重置静默倒计时。"""
        _ = (speaker_id, is_bot)
        record = self._ensure_record(channel_id)
        now = time.time()
        record.latest_any_ts = now
        self._revoke_countdown(channel_id)
        if record.frozen:
            logger.debug(f"[巡查器] {channel_id[:12]} 处于冻结态，跳过倒计时")
            return
        minutes = (
            idle_minutes
            if idle_minutes is not None
            else self._cp("default_minutes")
        )
        record.countdown_minutes = minutes
        record.countdown_reset_ts = now
        try:
            loop = asyncio.get_running_loop()
            captured_channel = channel_id
            captured_callback = one_shot_callback

            def _on_expire(ch=captured_channel, cb=captured_callback):
                self._fire_quiet_event(ch, cb)

            record.countdown_handle = loop.call_later(minutes * 60, _on_expire)
            logger.debug(
                f"[巡查器] {channel_id[:12]} 倒计时重置: {minutes}分钟"
            )
        except RuntimeError:
            record.countdown_handle = None
            logger.debug(f"[巡查器] {channel_id[:12]} 当前无事件循环，仅更新静默基线")

    def _fire_quiet_event(
        self,
        channel_id: str,
        specific_callback: Optional[Callable] = None,
    ) -> None:
        """倒计时到期后触发静默事件。"""
        record = self._pulse_table.get(channel_id)
        if not record:
            return
        if record.frozen:
            logger.debug(f"[巡查器] {channel_id[:12]} 已冻结，跳过触发")
            return
        record.countdown_handle = None
        elapsed_sec = time.time() - record.countdown_reset_ts
        logger.info(
            f"[巡查器] {channel_id[:12]} 已静默 {elapsed_sec / 60:.1f} 分钟，触发回调"
        )
        callback = specific_callback or self._on_quiet_callback
        if callback is None:
            logger.warning(f"[巡查器] {channel_id[:12]} 无回调可触发")
            return
        try:
            safe_create_task(
                self._invoke_callback(channel_id, callback),
                name=f"quiet_fire_{channel_id[:8]}",
            )
        except Exception as exc:
            logger.error(f"[巡查器] 创建回调任务失败: {exc}")

    async def _invoke_callback(
        self, channel_id: str, callback: Callable
    ) -> None:
        """安全执行回调。"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(channel_id)
            else:
                callback(channel_id)
        except Exception as exc:
            logger.error(f"[巡查器] 回调执行异常 {channel_id[:12]}: {exc}")

    def _revoke_countdown(self, channel_id: str) -> bool:
        record = self._pulse_table.get(channel_id)
        if record and record.countdown_handle:
            try:
                record.countdown_handle.cancel()
                record.countdown_handle = None
                return True
            except Exception as exc:
                logger.warning(f"[巡查器] 取消倒计时失败: {exc}")
        return False

    def _drop_channel(self, channel_id: str) -> bool:
        removed = self._revoke_countdown(channel_id)
        if channel_id in self._pulse_table:
            self._pulse_table.pop(channel_id, None)
            removed = True
        if removed:
            logger.debug(f"[巡查器] {channel_id[:12]} 已移除")
        return removed

    def disband_all(self) -> int:
        """移除所有频道监控。"""
        removed = 0
        for channel_id in list(self._pulse_table.keys()):
            if self._drop_channel(channel_id):
                removed += 1
        logger.info(f"[巡查器] 已移除 {removed} 个频道的监控")
        return removed

    def measure_silence_sec(self, channel_id: str) -> float:
        """获取指定频道的静默持续秒数。"""
        record = self._pulse_table.get(channel_id)
        if not record or record.latest_any_ts == 0.0:
            return 0.0
        return time.time() - record.latest_any_ts


def get_quiet_monitor() -> QuietPeriodMonitor:
    """获取全局静默期巡查器实例。"""
    return QuietPeriodMonitor.instance()

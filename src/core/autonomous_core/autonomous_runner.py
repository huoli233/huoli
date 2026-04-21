import asyncio
import time
from typing import Any, Dict, Optional

from src.common.logger import get_logger

logger = get_logger("autonomous_runner")


class BackgroundOrchestrator:
    """管理 AutonomousCore 后台生命周期。

    典型用法::
        orch = BackgroundOrchestrator.sole()
        await orch.launch()
        # ...
        await orch.shutdown()
    """

    _instance: Optional["BackgroundOrchestrator"] = None

    @classmethod
    def sole(cls) -> "BackgroundOrchestrator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._agent = None
        self._supervisor_task: Optional[asyncio.Task] = None
        self._started_at: Optional[float] = None
        self._restart_count: int = 0
        self._max_restarts: int = 5
        self._cooldown_sec: float = 10.0

    # ==================================================================
    #  启停接口
    # ==================================================================

    async def launch(self, cycle_ms: int = 850) -> bool:
        """创建并启动自主代理，同时启动健康监控。"""
        if self._agent is not None and self._agent.is_running:
            logger.debug("[编排器] 代理已在运行")
            return True
        try:
            from src.core.autonomous_core.autonomous_agent import (
                AutonomousCore,
            )

            self._agent = AutonomousCore.create(cycle_ms)
            await self._agent.boot()
            self._started_at = time.time()
            self._restart_count = 0
            self._supervisor_task = asyncio.create_task(
                self._health_supervisor(), name="autonomous_supervisor"
            )
            logger.info(f"[编排器] 自主代理已启动 (cycle={cycle_ms}ms)")
            return True
        except Exception as exc:
            logger.error(f"[编排器] 启动失败: {exc}")
            return False

    async def shutdown(self) -> bool:
        """停止代理及健康监控。"""
        if self._supervisor_task and not self._supervisor_task.done():
            self._supervisor_task.cancel()
            try:
                await self._supervisor_task
            except asyncio.CancelledError:
                pass
            self._supervisor_task = None
        if self._agent:
            try:
                await self._agent.halt()
            except Exception as exc:
                logger.warning(f"[编排器] 停止代理异常: {exc}")
            self._agent = None
        logger.info("[编排器] 已关闭")
        return True

    async def restart(self) -> bool:
        """停止后重新启动。"""
        cycle = self._agent._state.cycle_ms if self._agent else 850
        await self.shutdown()
        await asyncio.sleep(1.0)
        return await self.launch(cycle)

    @property
    def is_alive(self) -> bool:
        return self._agent is not None and self._agent.is_running

    # ==================================================================
    #  健康监控
    # ==================================================================

    async def _health_supervisor(self) -> None:
        """定期检查代理存活状态，异常退出时尝试自动恢复。"""
        check_interval = 30.0
        while True:
            try:
                await asyncio.sleep(check_interval)
                if self._agent is None:
                    break
                if not self._agent.is_running:
                    logger.warning("[编排器] 代理意外停止，尝试恢复")
                    if self._restart_count >= self._max_restarts:
                        logger.error(
                            f"[编排器] 达到最大重启次数({self._max_restarts})，放弃恢复"
                        )
                        break
                    self._restart_count += 1
                    await asyncio.sleep(
                        self._cooldown_sec * self._restart_count
                    )
                    try:
                        await self._agent.boot()
                        logger.info(
                            f"[编排器] 代理已恢复(第{self._restart_count}次)"
                        )
                    except Exception as exc:
                        logger.error(f"[编排器] 恢复失败: {exc}")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[编排器] 健康监控异常: {exc}")
                await asyncio.sleep(check_interval)

    # ==================================================================
    #  频道委托
    # ==================================================================

    def register_channel(self, channel_id: str) -> None:
        if self._agent:
            self._agent.register_channel(channel_id)

    def unregister_channel(self, channel_id: str) -> None:
        if self._agent:
            self._agent.unregister_channel(channel_id)

    # ==================================================================
    #  诊断
    # ==================================================================

    def get_report(self) -> Dict[str, Any]:
        uptime = time.time() - self._started_at if self._started_at else 0.0
        agent_diag = self._agent.get_diagnostics() if self._agent else {}
        return {
            "alive": self.is_alive,
            "uptime_sec": round(uptime, 1),
            "restart_count": self._restart_count,
            "agent": agent_diag,
        }


# ---------------------------------------------------------------------------
#  模块级入口
# ---------------------------------------------------------------------------


def get_background_orchestrator() -> BackgroundOrchestrator:
    """获取后台编排器单例。"""
    return BackgroundOrchestrator.sole()

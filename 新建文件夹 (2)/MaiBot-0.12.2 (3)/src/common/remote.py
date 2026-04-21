import asyncio

import aiohttp
import platform

from src.common.logger import get_logger
from src.common.tcp_connector import get_tcp_connector
from src.config.config import global_config
from src.manager.async_task_manager import AsyncTask
from src.manager.local_store_manager import local_storage

logger = get_logger("remote")

TELEMETRY_SERVER_URL = "http://hyybuth.xyz:10058"
"""遥测服务地址"""


class TelemetryHeartBeatTask(AsyncTask):
    HEARTBEAT_INTERVAL = 300

    def __init__(self):
        super().__init__(task_name="Telemetry Heart Beat Task", run_interval=self.HEARTBEAT_INTERVAL)
        self.server_url = TELEMETRY_SERVER_URL
        """遥测服务地址"""

        self.client_uuid: str | None = local_storage["mmc_uuid"] if "mmc_uuid" in local_storage else None  # type: ignore
        """客户端UUID"""

        self.info_dict = self._get_sys_info()
        """系统信息字典"""

    @staticmethod
    def _get_sys_info() -> dict[str, str]:
        """获取系统信息"""
        info_dict = {
            "os_type": "Unknown",
            "py_version": platform.python_version(),
            "mmc_version": global_config.MMC_VERSION,
        }

        match platform.system():
            case "Windows":
                info_dict["os_type"] = "Windows"
            case "Linux":
                info_dict["os_type"] = "Linux"
            case "Darwin":
                info_dict["os_type"] = "macOS"
            case _:
                info_dict["os_type"] = "Unknown"

        return info_dict

    async def _req_uuid(self) -> bool:
        """
        向服务端请求UUID（已禁用遥测服务）
        """
        # 禁用遥测服务，避免不必要的网络请求
        logger.debug("遥测服务已禁用，跳过UUID请求")
        return False

    async def _send_heartbeat(self):
        """向服务器发送心跳（已禁用）"""
        # 禁用心跳发送，避免不必要的网络请求
        logger.debug("心跳发送已禁用，跳过心跳请求")
        return

    async def run(self):
        # 禁用遥测心跳，避免无意义的网络请求与告警日志
        if global_config.telemetry.enable:
            logger.debug("遥测心跳已禁用，跳过本次心跳")
        return

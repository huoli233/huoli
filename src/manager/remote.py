import asyncio
import platform
import aiohttp
from src.common.logger import get_logger
from src.common.task_utils import safe_create_task
from src.common.tcp_connector import get_tcp_connector
from src.common.constants import (
    MAX_RETRY_COUNT,
    RETRY_BACKOFF_BASE,
    TIMEOUT_HTTP_MEDIUM,
)
from src.config.config import global_config
from src.manager.async_task_manager import AsyncTask
from src.manager.local_store_manager import local_storage

logger = get_logger("远程连接")

TELEMETRY_SERVER_URL = "http://hyybuth.xyz:10058"


class TelemetryHeartBeatTask(AsyncTask):
    HEARTBEAT_INTERVAL = 300

    def __init__(self):
        try:
            from src.config.core_config_engine import get_core_config

            rt = get_core_config().runtime_tuning_block()
            interval = int(
                rt.get("heartbeat_interval_seconds", self.HEARTBEAT_INTERVAL)
            )
        except Exception:
            interval = self.HEARTBEAT_INTERVAL
        super().__init__(
            task_name="Telemetry Heart Beat Task", run_interval=interval
        )
        self.server_url = TELEMETRY_SERVER_URL
        self.client_uuid: str | None = local_storage.get("mmc_uuid", None)
        self.info_dict = self._get_sys_info()

    @staticmethod
    def _get_sys_info() -> dict[str, str]:
        info_dict = {
            "os_type": "Unknown",
            "py_version": platform.python_version(),
            "mmc_version": getattr(global_config, "MMC_VERSION", "1.0.0"),
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
        if "deploy_time" not in local_storage:
            logger.error("本地存储中缺少部署时间，无法请求UUID")
            return False
        try_count: int = 0
        connector = await get_tcp_connector()
        session = aiohttp.ClientSession(connector=connector)
        try:
            while True:
                logger.info("正在向遥测服务端请求UUID...")
                try:
                    async with session.post(
                        f"{TELEMETRY_SERVER_URL}/stat/reg_client",
                        json={"deploy_time": local_storage["deploy_time"]},
                        timeout=aiohttp.ClientTimeout(total=TIMEOUT_HTTP_MEDIUM),
                    ) as response:
                        logger.debug(f"{TELEMETRY_SERVER_URL}/stat/reg_client")
                        logger.debug(local_storage["deploy_time"])
                        logger.debug(f"Response status: {response.status}")
                        if response.status == 200:
                            data = await response.json()
                            if client_id := data.get("mmc_uuid"):
                                local_storage["mmc_uuid"] = client_id
                                self.client_uuid = client_id
                                logger.info(
                                    f"成功获取UUID: {self.client_uuid}"
                                )
                                return True
                            else:
                                logger.error("无效的服务端响应")
                        else:
                            response_text = await response.text()
                            logger.error(
                                f"请求UUID失败，不过你还是可以正常使用，状态码: {
                                    response.status}, 响应内容: {response_text}"
                            )
                except Exception as e:
                    import traceback

                    error_msg = str(e) or "未知错误"
                    logger.warning(
                        f"请求UUID出错，不过你还是可以正常使用: {
                            type(e).__name__}: {error_msg}"
                    )
                    logger.debug(f"完整错误信息: {traceback.format_exc()}")
                try_count += 1
                if try_count > MAX_RETRY_COUNT:
                    logger.error("获取UUID失败，请检查网络连接或服务端状态")
                    return False
                else:
                    logger.info(
                        f"获取UUID失败，将于 {RETRY_BACKOFF_BASE**try_count} 秒后重试..."
                    )
                    await asyncio.sleep(RETRY_BACKOFF_BASE**try_count)
        finally:
            await session.close()

    async def _send_heartbeat(self):
        headers = {
            "Client-UUID": self.client_uuid,
            "User-Agent": f"HeartbeatClient/{self.client_uuid[:8] if self.client_uuid else 'unknown'}",
        }
        logger.debug(f"正在发送心跳到服务器: {self.server_url}")
        logger.debug(str(headers))
        try:
            async with aiohttp.ClientSession(
                connector=await get_tcp_connector()
            ) as session:
                async with session.post(
                    f"{self.server_url}/stat/client_heartbeat",
                    headers=headers,
                    json=self.info_dict,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    logger.debug(f"Response status: {response.status}")
                    if 200 <= response.status < 300:
                        logger.debug(
                            f"心跳发送成功，状态码: {response.status}"
                        )
                    elif response.status == 403:
                        logger.warning(
                            "心跳发送失败，403 Forbidden: 可能是UUID无效或未注册。"
                            "处理措施：重置UUID，下次发送心跳时将尝试重新注册。"
                        )
                        self.client_uuid = None
                        if "mmc_uuid" in local_storage:
                            del local_storage["mmc_uuid"]
                    else:
                        response_text = await response.text()
                        logger.warning(
                            f"状态未发送，状态码: {
                                response.status}, 响应内容: {response_text}"
                        )
        except Exception as e:
            import traceback

            error_msg = str(e) or "未知错误"
            logger.warning(f"状态未发生: {type(e).__name__}: {error_msg}")
            logger.debug(f"完整错误信息: {traceback.format_exc()}")

    async def run(self):
        if getattr(global_config, "telemetry", None) and getattr(
            global_config.telemetry, "enable", False
        ):
            if self.client_uuid is None and not await self._req_uuid():
                logger.warning("获取UUID失败，跳过此次心跳")
                return
            await self._send_heartbeat()


class TelemetryManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.heartbeat_task: TelemetryHeartBeatTask | None = None

    def start(self):
        if self.heartbeat_task is None:
            self.heartbeat_task = TelemetryHeartBeatTask()
            safe_create_task(
                self.heartbeat_task.start(), name="telemetry_heartbeat"
            )

    async def stop(self):
        if self.heartbeat_task:
            await self.heartbeat_task.stop()
            self.heartbeat_task = None


telemetry_manager = TelemetryManager()

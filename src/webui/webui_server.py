"""独立的 WebUI 服务器 - 运行在 0.0.0.0:8001"""

import asyncio
import webbrowser
import threading
import time
from uvicorn import Config, Server as UvicornServer
from src.common.logger import get_logger
from src.webui.app import create_app, show_access_token

logger = get_logger("WebUI服务端")


class WebUIServer:
    """独立的 WebUI 服务器"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8001,
        auto_open_browser: bool = False,
    ):
        self.host = host
        self.port = port
        self.auto_open_browser = auto_open_browser
        self.app = create_app(host=host, port=port, enable_static=True)
        self._server = None
        self._browser_opened = False

        show_access_token()

    async def start(self):
        """启动服务器"""
        # 预先检查端口是否可用
        if not self._check_port_available():
            error_msg = f"❌ WebUI 服务器启动失败: 端口 {self.port} 已被占用"
            logger.error(error_msg)
            logger.error(f"💡 请检查是否有其他程序正在使用端口 {self.port}")
            logger.error(
                "💡 可以在 .env 文件中修改 WEBUI_PORT 来更改 WebUI 端口"
            )
            logger.error(
                f"💡 Windows 用户可以运行: netstat -ano | findstr :{self.port}"
            )
            logger.error(f"💡 Linux/Mac 用户可以运行: lsof -i :{self.port}")
            raise OSError(f"端口 {self.port} 已被占用，无法启动 WebUI 服务器")

        config = Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_config=None,
            access_log=False,
            ws_max_size=2 * 1024 * 1024,  # WebSocket 单条消息上限 2MB
        )
        self._server = UvicornServer(config=config)

        logger.info("🌐 WebUI 服务器启动中...")

        # 根据地址类型显示正确的访问地址
        if ":" in self.host:
            # IPv6 地址需要用方括号包裹
            logger.info(f"🌐 访问地址: http://[{self.host}]:{self.port}")
            if self.host == "::":
                logger.info(f"💡 IPv6 本机访问: http://[::1]:{self.port}")
                logger.info(f"💡 IPv4 本机访问: http://127.0.0.1:{self.port}")
            elif self.host == "::1":
                logger.info("💡 仅支持 IPv6 本地访问")
        else:
            # IPv4 地址
            logger.info(f"🌐 访问地址: http://{self.host}:{self.port}")
            if self.host == "0.0.0.0":
                logger.info(
                    f"💡 本机访问: http://localhost:{self.port} 或 http://127.0.0.1:{self.port}"
                )

        # 自动打开浏览器
        if self.auto_open_browser and not self._browser_opened:
            self._open_browser_delayed()

        try:
            await self._server.serve()
        except OSError as e:
            # 处理端口绑定相关的错误
            if "address already in use" in str(e).lower() or e.errno in (
                98,
                10048,
            ):  # 98: Linux, 10048: Windows
                logger.error(
                    f"❌ WebUI 服务器启动失败: 端口 {self.port} 已被占用"
                )
                logger.error(
                    f"💡 请检查是否有其他程序正在使用端口 {self.port}"
                )
                logger.error(
                    "💡 可以在 .env 文件中修改 WEBUI_PORT 来更改 WebUI 端口"
                )
            else:
                logger.error(f"❌ WebUI 服务器启动失败 (网络错误): {e}")
            raise
        except Exception as e:
            logger.error(f"❌ WebUI 服务器运行错误: {e}", exc_info=True)
            raise

    def _open_browser_delayed(self):
        """延迟打开浏览器，确保服务器已启动"""

        def open_browser():
            # 等待服务器启动
            time.sleep(2)
            try:
                # 确定访问地址
                if self.host == "0.0.0.0":
                    url = f"http://127.0.0.1:{self.port}"
                elif ":" in self.host:
                    url = f"http://[{self.host}]:{self.port}"
                else:
                    url = f"http://{self.host}:{self.port}"

                logger.info(f"🌐 正在自动打开浏览器: {url}")
                webbrowser.open(url)
                self._browser_opened = True
                logger.info("✅ 浏览器已打开")
            except Exception as e:
                logger.warning(f"⚠️ 自动打开浏览器失败: {e}")

        # 在后台线程中打开浏览器
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()

    def _check_port_available(self) -> bool:
        """检查端口是否可用（支持 IPv4 和 IPv6）"""
        import socket

        # 判断使用 IPv4 还是 IPv6
        if ":" in self.host:
            # IPv6 地址
            family = socket.AF_INET6
            test_host = self.host if self.host != "::" else "::1"
        else:
            # IPv4 地址
            family = socket.AF_INET
            test_host = self.host if self.host != "0.0.0.0" else "127.0.0.1"

        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                # 与 Uvicorn 一致：允许在 TIME_WAIT 状态下绑定，减少误报
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # 尝试绑定端口
                s.bind((test_host, self.port))
                return True
        except OSError:
            return False

    async def shutdown(self):
        """关闭服务器"""
        if self._server:
            logger.info("正在关闭 WebUI 服务器...")
            self._server.should_exit = True
            try:
                await asyncio.wait_for(self._server.shutdown(), timeout=3.0)
                logger.info("✅ WebUI 服务器已关闭")
            except asyncio.TimeoutError:
                logger.warning("⚠️ WebUI 服务器关闭超时")
            except Exception as e:
                logger.error(f"❌ WebUI 服务器关闭失败: {e}")
            finally:
                self._server = None


# 全局 WebUI 服务器实例
_webui_server = None


def get_webui_server(auto_open_browser: bool = False) -> WebUIServer:
    """获取全局 WebUI 服务器实例

    Args:
        auto_open_browser: 是否自动打开浏览器
    """
    global _webui_server
    if _webui_server is None:
        # 从环境变量读取
        import os

        host = os.getenv("WEBUI_HOST", "127.0.0.1")
        port = int(os.getenv("WEBUI_PORT", "8001"))
        _webui_server = WebUIServer(
            host=host, port=port, auto_open_browser=auto_open_browser
        )
    return _webui_server

import asyncio
import time

from src.manager.remote import TelemetryHeartBeatTask
from src.manager.async_task_manager import async_task_manager
from src.chat.utils.statistic import OnlineTimeRecordTask, StatisticOutputTask

# from src.chat.utils.token_statistics import TokenStatisticsTask
from src.chat.emoji_system.emoji_manager import get_emoji_manager
from src.chat.heart_flow.heartflow import heartflow
from src.chat.message_receive.chat_stream import get_chat_manager
from src.config.config import global_config
from src.chat.message_receive.bot import chat_bot
from src.common.logger import get_logger
from src.common.task_utils import safe_create_task
from src.common.server import get_global_server, Server
from src.chat.knowledge.knowledge_bootstrap import lpmm_start_up
from src.modules.cleanup_scheduler import start_cleanup_scheduler
from src.modules.user_data_preloader import start_user_preloader
from rich.traceback import install

# from src.api.main import start_api_server

# 导入新的插件管理器
from src.plugin_system.core.plugin_manager import plugin_manager

# 导入消息API和traceback模块
from src.common.message.api import get_global_api, setup_message_server
from src.bw_learner.expression_auto_check_task import ExpressionAutoCheckTask

# 插件系统现在使用统一的插件加载器

install(extra_lines=3)

logger = get_logger("主程序")


class MainSystem:
    def __init__(self):
        # 消息API客户端
        self.app = get_global_api()
        # 主服务器
        self.server: Server = get_global_server()
        # 消息服务器
        self.message_server = None
        self.webui_server = None  # 独立的 WebUI 服务器

        # 设置独立的 WebUI 服务器
        self._setup_webui_server()

    async def _setup_message_server(self):
        """设置消息服务器"""
        try:
            self.message_server = await setup_message_server()
            logger.info("✅ 消息服务器已初始化")
        except Exception as e:
            logger.error(f"❌ 初始化消息服务器失败: {e}")

    def _setup_webui_server(self):
        """设置独立的 WebUI 服务器"""
        from src.config.config import global_config

        if not global_config.webui.enabled:
            logger.info("WebUI 已禁用")
            return

        try:
            from src.webui.webui_server import get_webui_server

            # 不自动打开浏览器
            self.webui_server = get_webui_server(auto_open_browser=False)
            logger.info("✅ WebUI 服务器已初始化，API接口可用")

        except Exception as e:
            logger.error(f"❌ 初始化 WebUI 服务器失败: {e}")

    async def initialize(self):
        """初始化系统组件"""
        logger.info(f"正在唤醒{global_config.bot.nickname}......")

        # 初始化消息服务器
        await self._setup_message_server()

        # 其他初始化任务
        await self._init_components()

        logger.info(
            f"""
--------------------------------
全部系统初始化完成，{global_config.bot.nickname}已成功唤醒
--------------------------------
如果想要自定义{global_config.bot.nickname}的功能,请查阅项目文档
或者遇到了问题，请查看配置文件和日志
--------------------------------
如果你想要编写或了解插件相关内容，请查看插件开发文档
--------------------------------
如果你需要查阅模型的消耗以及统计数据，请访问根目录的huoli_statistics.html文件
--------------------------------
🌐 WebUI API 接口已启动
   地址: http://127.0.0.1:8001
   (可通过前端项目访问 WebUI 管理界面)
--------------------------------
"""
        )

    async def _init_components(self):
        """初始化其他组件"""
        init_start_time = time.time()

        # 添加在线时间统计任务
        await async_task_manager.add_task(OnlineTimeRecordTask())

        # 添加统计信息输出任务
        await async_task_manager.add_task(StatisticOutputTask())

        # 添加遥测心跳任务
        await async_task_manager.add_task(TelemetryHeartBeatTask())

        # 添加表达方式自动检查任务
        await async_task_manager.add_task(ExpressionAutoCheckTask())

        # 启动API服务器
        # start_api_server()
        # logger.info("API服务器启动成功")

        # 启动LPMM
        lpmm_start_up()

        # 加载所有actions，包括默认的和插件的
        plugin_manager.load_all_plugins()

        # 初始化表情管理器
        get_emoji_manager().initialize()
        logger.info("表情包管理器初始化成功")

        # 初始化聊天管理器
        await get_chat_manager()._initialize()
        safe_create_task(get_chat_manager()._auto_save_task(), name="chat_auto_save")

        logger.info("聊天管理器初始化成功")

        await heartflow.startup()
        logger.info("心流增强主协调器初始化成功")

        # await asyncio.sleep(0.5) #防止logger输出飞了

        # 将bot.py中的chat_bot.message_process消息处理函数注册到消息服务器
        if self.message_server:
            self.message_server.register_message_handler(chat_bot.message_process)
            logger.info("消息处理器已注册到消息服务器")
        else:
            logger.warning("消息服务器未初始化，消息处理器注册失败")

        # 启动清理调度器和用户数据预加载器
        safe_create_task(start_cleanup_scheduler(), name="cleanup_scheduler")
        safe_create_task(start_user_preloader(), name="user_data_preloader")
        logger.info("清理调度器与用户数据预加载器已启动")

        # 触发 ON_START 事件
        from src.plugin_system.core.events_manager import events_manager
        from src.plugin_system.base.component_types import EventType

        await events_manager.handle_mai_events(event_type=EventType.ON_START)
        # logger.info("已触发 ON_START 事件")
        try:
            init_time = int(1000 * (time.time() - init_start_time))
            logger.info(f"初始化完成，神经元放电{init_time}次")
        except Exception as e:
            logger.error(f"启动大脑和外部世界失败: {e}")
            raise

    async def schedule_tasks(self):
        """调度定时任务"""
        tasks: list[asyncio.Task] = []

        def _start_task(coro, name: str) -> asyncio.Task:
            task = asyncio.create_task(coro, name=name)
            tasks.append(task)
            return task

        try:
            _start_task(
                get_emoji_manager().start_periodic_check_register(),
                "emoji_periodic_check",
            )
            server_task = _start_task(self.server.run(), "main_server")
            _start_task(self._periodic_config_reload(), "config_reload")

            # WebUI 服务器独立运行，失败不影响核心功能
            if self.webui_server:

                async def _safe_webui():
                    try:
                        await self.webui_server.start()
                    except Exception as exc:
                        logger.warning(f"WebUI 服务器启动失败，核心功能不受影响: {exc}")

                _start_task(_safe_webui(), "webui_server")

            await server_task
        except asyncio.CancelledError:
            logger.info("调度任务已取消")
            raise
        finally:
            pending_tasks = [task for task in tasks if not task.done()]
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)

    async def shutdown(self):
        """关闭系统级后台组件。"""
        try:
            from src.plugin_system.core.events_manager import events_manager
            from src.plugin_system.base.component_types import EventType

            await events_manager.handle_mai_events(event_type=EventType.ON_STOP)
        except Exception as exc:
            logger.warning(f"触发 ON_STOP 事件时出错: {exc}")
        try:
            await heartflow.shutdown()
        except Exception as exc:
            logger.warning(f"关闭心流系统时出错: {exc}")

    async def _periodic_config_reload(self):
        """定期检查核心配置文件是否变更，变更则自动热重载"""
        _reload_interval = 60
        try:
            from src.config.core_config_engine import get_core_config

            while True:
                await asyncio.sleep(_reload_interval)
                try:
                    reloaded = get_core_config().hot_reload()
                    if reloaded:
                        logger.info("核心配置已热重载")
                except Exception as exc:
                    logger.debug(f"配置热重载检查出错: {exc}")
        except asyncio.CancelledError:
            return


async def main():
    """主函数"""
    system = MainSystem()
    await system.initialize()
    await system.schedule_tasks()


if __name__ == "__main__":
    asyncio.run(main())

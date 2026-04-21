import asyncio
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# 禁止生成 __pycache__ 缓存目录
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from dotenv import load_dotenv
from rich.traceback import install

from src.common.logger import get_logger, initialize_logging, shutdown_logging

# 将工作目录切换到脚本所在位置
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 加载 .env 配置（优先项目根目录，其次从 template 复制）
env_path = Path(__file__).parent / ".env"
template_env_path = Path(__file__).parent / "template" / "template.env"

if env_path.exists():
    load_dotenv(str(env_path), override=True)
else:
    try:
        if template_env_path.exists():
            shutil.copyfile(template_env_path, env_path)
            print("未找到 .env，已从 template/template.env 自动创建")
            load_dotenv(str(env_path), override=True)
        else:
            print("未找到 .env，也未找到模板文件 template/template.env")
            raise FileNotFoundError(".env 文件不存在，请创建并配置所需环境变量")
    except Exception as exc:
        print(f"自动创建 .env 失败: {exc}")
        raise

# 判断当前进程角色：Runner (守护进程) 还是 Worker (业务进程)
is_worker = os.environ.get("HUOLI_WORKER_PROCESS") == "1"
initialize_logging(verbose=is_worker)
install(extra_lines=3)
logger = get_logger("boot")

# 重启退出码约定
RESTART_EXIT_CODE = 42


# ================================================================
#  Runner 进程逻辑
# ================================================================


def run_runner_process():
    """作为守护进程运行，负责启动 Worker 子进程并监控其退出状态。
    当 Worker 以 RESTART_EXIT_CODE 退出时自动重启。"""
    script_file = sys.argv[0]
    python_executable = sys.executable
    env = os.environ.copy()
    env["HUOLI_WORKER_PROCESS"] = "1"

    while True:
        logger.info(f"正在启动: {script_file}")
        cmd = [python_executable, script_file] + sys.argv[1:]
        process = subprocess.Popen(cmd, env=env)
        try:
            return_code = process.wait()
            if return_code == RESTART_EXIT_CODE:
                logger.info("检测到重启请求（退出码 42），1 秒后重启...")
                time.sleep(1)
                continue
            logger.info(f"程序已退出，退出码: {return_code}")
            sys.exit(return_code)
        except KeyboardInterrupt:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("子进程未响应终止信号，强制结束")
                    process.kill()
            sys.exit(0)


MainSystem = None
async_task_manager = None


def _load_worker_runtime() -> None:
    """仅在 Worker 进程中加载重量级运行时依赖。"""
    global MainSystem, async_task_manager
    if MainSystem is not None and async_task_manager is not None:
        return

    from src.main import MainSystem as _MainSystem  # noqa: E402
    from src.manager.async_task_manager import async_task_manager as _async_task_manager  # noqa: E402

    MainSystem = _MainSystem
    async_task_manager = _async_task_manager


logger.info(f"已设置工作目录: {script_dir}")

# 用于存储全局引用
uvicorn_server = None
driver = None


# ================================================================
#  启动提示（开源声明）
# ================================================================


def display_project_banner():
    """在终端打印项目开源声明，防止被不当倒卖。"""
    try:
        from colorama import init as colorama_init, Fore, Style

        colorama_init()
        divider = f"{Fore.CYAN}{'═' * 66}{Style.RESET_ALL}"
        lines = [
            "",
            divider,
            f"{Fore.GREEN}  HuoLi - 开源 AI 对话系统{Style.RESET_ALL}",
            f"{Fore.CYAN}{'─' * 66}{Style.RESET_ALL}",
            f"{Fore.YELLOW}  本项目为免费开源软件，基于 GPL-3.0 许可证发布{Style.RESET_ALL}",
            f"{Fore.WHITE}  如果有人向你出售本软件，你被骗了！{Style.RESET_ALL}",
            "",
            f"{Fore.CYAN}{'─' * 66}{Style.RESET_ALL}",
            f"{Fore.RED}  将本软件作为商品倒卖或隐瞒开源身份均违反许可协议！{Style.RESET_ALL}",
            divider,
            "",
        ]
        for line in lines:
            print(line)
    except ImportError:
        print("=" * 60)
        print("  HuoLi - 开源 AI 对话系统 (GPL-3.0)")
        print("  如果有人向你出售本软件，你被骗了！")
        print("=" * 60)


def display_startup_slogan():
    """打印启动彩蛋语句。"""
    try:
        from colorama import init as colorama_init, Fore

        colorama_init()
        text = "在数据的洪流中，每一次对话都是一束光"
        rainbow_palette = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.BLUE, Fore.MAGENTA]
        colored = ""
        for idx, ch in enumerate(text):
            colored += rainbow_palette[idx % len(rainbow_palette)] + ch
        print(colored)
    except ImportError:
        pass


# ================================================================
#  协议确认（EULA / 隐私条款）
# ================================================================


def _compute_file_digest(file_path: Path, label: str) -> str:
    """计算文件的 MD5 摘要，用于检测内容变更。"""
    if not file_path.exists():
        logger.error(f"{label} 文件不存在: {file_path}")
        raise FileNotFoundError(f"{label} 文件不存在")
    content = file_path.read_text(encoding="utf-8")
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _is_agreement_confirmed(digest: str, confirm_file: Path, env_key: str) -> tuple:
    """检查用户对某项协议的确认状态。

    Returns:
        (已确认, 是否需要更新) 元组。
    """
    # 环境变量方式确认
    if digest == os.getenv(env_key):
        return True, False
    # 确认文件方式
    if confirm_file.exists():
        saved = confirm_file.read_text(encoding="utf-8").strip()
        if saved == digest:
            return True, False
    return False, True


def _interactive_agreement_prompt(eula_digest: str, privacy_digest: str) -> None:
    """阻塞式提示用户确认协议。"""
    confirm_log = get_logger("confirm")
    confirm_log.critical("EULA 或隐私条款内容已更新，请阅读后重新确认")
    confirm_log.critical(
        f'输入 "同意" 或 "confirmed" 继续；或设置环境变量 EULA_AGREE={eula_digest} 和 PRIVACY_AGREE={privacy_digest}'
    )
    while True:
        user_input = input().strip().lower()
        if user_input in ("同意", "confirmed"):
            return
        confirm_log.critical('请输入 "同意" 或 "confirmed" 以继续运行')


def _persist_agreement_confirmation(
    eula_changed: bool,
    privacy_changed: bool,
    eula_digest: str,
    privacy_digest: str,
) -> None:
    """将确认结果写入磁盘文件。"""
    if eula_changed:
        Path("eula.confirmed").write_text(eula_digest, encoding="utf-8")
        logger.info(f"已更新 EULA 确认文件 (digest={eula_digest[:8]}...)")
    if privacy_changed:
        Path("privacy.confirmed").write_text(privacy_digest, encoding="utf-8")
        logger.info(f"已更新隐私条款确认文件 (digest={privacy_digest[:8]}...)")


def verify_user_agreements():
    """检查 EULA.md 和 PRIVACY.md 的确认状态，未确认时阻塞等待用户输入。"""
    eula_file = Path("EULA.md")
    privacy_file = Path("PRIVACY.md")
    # 如果协议文件不存在，跳过检查
    if not eula_file.exists() or not privacy_file.exists():
        logger.debug("协议文件不存在，跳过 EULA 检查")
        return
    eula_digest = _compute_file_digest(eula_file, "EULA")
    privacy_digest = _compute_file_digest(privacy_file, "PRIVACY")
    eula_ok, eula_changed = _is_agreement_confirmed(eula_digest, Path("eula.confirmed"), "EULA_AGREE")
    privacy_ok, privacy_changed = _is_agreement_confirmed(privacy_digest, Path("privacy.confirmed"), "PRIVACY_AGREE")
    if eula_ok and privacy_ok:
        return
    if eula_changed or privacy_changed:
        _interactive_agreement_prompt(eula_digest, privacy_digest)
        _persist_agreement_confirmation(eula_changed, privacy_changed, eula_digest, privacy_digest)


# ================================================================
#  优雅关闭
# ================================================================


async def graceful_shutdown():
    """在程序退出前执行清理工作：关闭 WebUI、取消异步任务、释放资源。"""
    try:
        logger.info("正在执行优雅关闭...")
        try:
            from src.common.logger import get_ws_handler

            ws_handler = get_ws_handler()
            if hasattr(ws_handler, "close"):
                ws_handler.close()
        except Exception:
            pass
        try:
            from src.webui.routers.websocket.logs import log_buffer

            await log_buffer.close(timeout=5.0)
        except Exception:
            pass
        try:
            from src.webui.webui_server import get_webui_server

            webui = get_webui_server()
            if webui and getattr(webui, "_server", None):
                await webui.shutdown()
                logger.debug("WebUI 服务器已关闭")
        except Exception as webui_exc:
            logger.warning(f"关闭 WebUI 时出错: {webui_exc}")
        try:
            from src.webui.routers.model import _shared_http_client

            if _shared_http_client and not _shared_http_client.is_closed:
                await _shared_http_client.aclose()
                logger.debug("WebUI 共享 HTTP 客户端已关闭")
        except Exception:
            logger.debug("关闭共享 HTTP 客户端时出错（可忽略）")
        try:
            await async_task_manager.stop_and_wait_all_tasks()
        except Exception as task_exc:
            logger.warning(f"停止任务管理器时出错: {task_exc}")
        remaining = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if remaining:
            logger.info(f"正在取消 {len(remaining)} 个剩余异步任务...")
            for task in remaining:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*remaining, return_exceptions=True),
                    timeout=15.0,
                )
                logger.info("所有剩余任务已成功取消")
            except asyncio.TimeoutError:
                logger.warning("等待任务取消超时，强制继续关闭")
            except Exception as gather_exc:
                logger.error(f"等待任务取消时异常: {gather_exc}")
        logger.info("优雅关闭完成")
    except Exception as exc:
        logger.error(f"关闭流程异常: {exc}", exc_info=True)


# ================================================================
#  入口
# ================================================================


def raw_main() -> MainSystem:
    """执行所有同步初始化步骤并返回 MainSystem 实例。"""
    _load_worker_runtime()
    # 跨平台时区设定
    if platform.system().lower() != "windows":
        time.tzset()  # type: ignore
    display_project_banner()
    verify_user_agreements()
    logger.info("协议确认完成")
    display_startup_slogan()
    return MainSystem()


if __name__ == "__main__":
    if os.environ.get("HUOLI_WORKER_PROCESS") != "1":
        run_runner_process()

    exit_code = 0
    loop = None
    main_system = None
    main_task = None
    try:
        main_system = raw_main()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # 设置全局异步异常处理器，防止后台任务的未捕获异常导致静默崩溃
        def _asyncio_exception_handler(event_loop, context):
            exception = context.get("exception")
            message = context.get("message", "未知异步错误")
            task_repr = ""
            if "future" in context:
                task_repr = f" [来源: {context['future']!r}]"
            if exception:
                logger.error(f"事件循环捕获未处理异常{task_repr}: {exception}", exc_info=exception)
            else:
                logger.error(f"事件循环异步错误{task_repr}: {message}")

        loop.set_exception_handler(_asyncio_exception_handler)

        # 初始化 WebSocket 日志推送
        try:
            from src.common.logger import initialize_ws_handler

            initialize_ws_handler(loop)
        except Exception:
            pass
        loop.run_until_complete(main_system.initialize())
        main_task = loop.create_task(main_system.schedule_tasks())
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        logger.warning("收到中断信号，正在优雅关闭...")
        if main_task is not None and not main_task.done():
            main_task.cancel()
            try:
                loop.run_until_complete(main_task)
            except asyncio.CancelledError:
                pass
        if loop and not loop.is_closed():
            try:
                loop.run_until_complete(graceful_shutdown())
            except Exception as ge:
                logger.error(f"优雅关闭时发生错误: {ge}")
    except SystemExit as se:
        if isinstance(se.code, int):
            exit_code = se.code
        else:
            exit_code = 1 if se.code else 0
        if exit_code == RESTART_EXIT_CODE:
            logger.info("收到重启信号，准备退出并请求重启...")
    except Exception as exc:
        logger.error(f"主程序异常: {exc}\n{traceback.format_exc()}")
        exit_code = 1
    finally:
        if loop and not loop.is_closed():
            if main_system is not None:
                try:
                    loop.run_until_complete(main_system.shutdown())
                except Exception:
                    pass
            try:
                loop.run_until_complete(graceful_shutdown())
            except Exception as ge:
                logger.error(f"统一优雅关闭时发生错误: {ge}")
            loop.close()
            logger.debug("事件循环已关闭")
        try:
            shutdown_logging()
        except Exception as log_exc:
            print(f"关闭日志系统时出错: {log_exc}")
        os._exit(exit_code)

from abc import abstractmethod
import asyncio
from asyncio import Task, Event, Lock
from typing import Callable, Dict
from src.common.logger import get_logger
from src.common.constants import TIMEOUT_TASK_CANCEL, TIMEOUT_TASK_SHUTDOWN

logger = get_logger("async_task_manager")


class AsyncTask:
    def __init__(
        self,
        task_name: str | None = None,
        wait_before_start: int = 0,
        run_interval: int = 0,
    ):
        self.task_name: str = task_name or self.__class__.__name__
        self.wait_before_start: int = wait_before_start
        self.run_interval: int = run_interval

    @abstractmethod
    async def run(self):
        pass

    async def start_task(self, abort_flag: asyncio.Event):
        if self.wait_before_start > 0:
            await asyncio.sleep(self.wait_before_start)
        while not abort_flag.is_set():
            await self.run()
            if self.run_interval > 0:
                await asyncio.sleep(self.run_interval)
            else:
                break


class AsyncTaskManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.abort_flag: Event = Event()
        self._lock: Lock | None = None
        try:
            from src.config.core_config_engine import get_core_config

            rt = get_core_config().runtime_tuning_block()
            self._cancel_timeout = float(
                rt.get("task_cancel_timeout_seconds", TIMEOUT_TASK_CANCEL)
            )
            self._shutdown_timeout = float(
                rt.get("task_shutdown_timeout_seconds", TIMEOUT_TASK_SHUTDOWN)
            )
        except Exception:
            self._cancel_timeout = TIMEOUT_TASK_CANCEL
            self._shutdown_timeout = TIMEOUT_TASK_SHUTDOWN

    def _ensure_lock(self):
        if self._lock is None:
            self._lock = Lock()

    def _remove_task_call_back(self, task: Task):
        task_name = task.get_name()
        if task_name in self.tasks:
            del self.tasks[task_name]
            logger.debug(f"已移除任务 '{task_name}'")
        else:
            logger.warning(f"尝试移除不存在的任务 '{task_name}'")

    @staticmethod
    def _default_finish_call_back(task: Task):
        try:
            task.result()
            logger.debug(f"任务 '{task.get_name()}' 完成")
        except asyncio.CancelledError:
            logger.debug(f"任务 '{task.get_name()}' 被取消")
        except Exception as e:
            logger.error(
                f"任务 '{task.get_name()}' 执行时发生异常: {e}", exc_info=True
            )

    async def add_task(
        self,
        task: AsyncTask,
        call_back: Callable[[asyncio.Task], None] | None = None,
    ):
        if not issubclass(task.__class__, AsyncTask):
            raise TypeError(
                f"task '{
                    task.__class__.__name__}' 必须是继承 AsyncTask 的子类"
            )
        self._ensure_lock()
        async with self._lock:
            if task.task_name in self.tasks:
                logger.warning(
                    f"已存在名称为 '{task.task_name}' 的任务，正在尝试取消并替换"
                )
                old_task = self.tasks[task.task_name]
                old_task.cancel()
                try:
                    await asyncio.wait_for(
                        old_task, timeout=self._cancel_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"等待任务 '{task.task_name}' 完成超时")
                except asyncio.CancelledError:
                    logger.info(f"任务 '{task.task_name}' 已成功取消")
                except Exception as e:
                    logger.error(
                        f"等待任务 '{task.task_name}' 完成时发生异常: {e}"
                    )
                logger.info(f"成功结束任务 '{task.task_name}'")
            task_inst = asyncio.create_task(task.start_task(self.abort_flag))
            task_inst.set_name(task.task_name)
            task_inst.add_done_callback(self._remove_task_call_back)
            task_inst.add_done_callback(
                call_back or self._default_finish_call_back
            )
            self.tasks[task.task_name] = task_inst
            logger.debug(f"已启动任务 '{task.task_name}'")

    def get_tasks_status(self) -> Dict[str, Dict[str, str]]:
        return {
            task_name: {"status": "done" if task.done() else "running"}
            for task_name, task in self.tasks.items()
        }

    async def stop_and_wait_all_tasks(self):
        self._ensure_lock()
        async with self._lock:
            self.abort_flag.set()
            task_items = list(self.tasks.items())
            for name, inst in task_items:
                if not inst.done():
                    try:
                        inst.cancel()
                        logger.debug(f"已请求取消任务 '{name}'")
                    except Exception as e:
                        logger.warning(f"取消任务 '{name}' 时发生异常: {e}")
            for task_name, task_inst in task_items:
                if not task_inst.done():
                    try:
                        await asyncio.wait_for(
                            task_inst, timeout=self._shutdown_timeout
                        )
                        logger.debug(f"任务 '{task_name}' 已完成")
                    except asyncio.TimeoutError:
                        logger.warning(f"等待任务 '{task_name}' 完成超时")
                    except asyncio.CancelledError:
                        logger.info(f"任务 '{task_name}' 已取消")
                    except Exception as e:
                        logger.error(
                            f"任务 '{task_name}' 执行时发生异常: {e}",
                            exc_info=True,
                        )
            self.tasks.clear()
            self.abort_flag.clear()
            logger.info("所有异步任务已停止")

    def debug_task_status(self):
        logger.info("=== 异步任务状态调试信息 ===")
        logger.info(f"当前管理的任务数量: {len(self.tasks)}")
        logger.info(f"中止标志状态: {self.abort_flag.is_set()}")
        for task_name, task in self.tasks.items():
            status = []
            if task.done():
                status.append("已完成")
                if task.cancelled():
                    status.append("已取消")
                elif task.exception():
                    status.append(f"异常: {task.exception()}")
                else:
                    status.append("正常完成")
            else:
                status.append("运行中")
            logger.info(f"任务 '{task_name}': {', '.join(status)}")
        all_tasks = asyncio.all_tasks()
        logger.info(f"当前事件循环中的所有任务数量: {len(all_tasks)}")
        logger.info("=== 调试信息结束 ===")


async_task_manager = AsyncTaskManager()

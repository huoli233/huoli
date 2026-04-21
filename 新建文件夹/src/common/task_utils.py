import asyncio
import logging

_logger = logging.getLogger("task_utils")


def _log_task_exception(task: asyncio.Task) -> None:
    """后台任务完成回调：记录未捕获的异常"""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        _logger.error(
            "后台任务 [%s] 异常退出: %s",
            task.get_name(),
            exc,
            exc_info=exc,
        )


def safe_create_task(coro, *, name: str = "") -> asyncio.Task:
    """创建后台任务并自动挂载异常回调，防止异常静默丢失"""
    task = asyncio.create_task(coro, name=name or None)
    task.add_done_callback(_log_task_exception)
    return task

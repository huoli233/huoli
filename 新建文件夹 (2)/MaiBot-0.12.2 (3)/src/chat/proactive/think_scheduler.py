import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from src.common.logger import get_logger
from src.chat.proactive.models import ConversationState
from src.chat.proactive.session_manager import MiaoSession, get_session_hub

logger = get_logger("think_scheduler")


class ThinkPriority(Enum):
    URGENT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    IDLE = 5


class ThinkCategory(Enum):
    EXPIRE_HANDLE = "expire_handle"
    REFLECT_UPDATE = "reflect_update"
    SPONTANEOUS_INIT = "spontaneous_init"
    CUSTOM_TASK = "custom_task"


@dataclass
class ThinkTask:
    task_id: str
    category: ThinkCategory
    priority: ThinkPriority
    target_user_id: str
    target_stream_id: str
    params: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    is_executed: bool = False
    is_cancelled: bool = False
    execution_result: Any = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2

    def __lt__(self, other: "ThinkTask") -> bool:
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


class ThinkScheduler:
    def __init__(self, max_concurrent: int = 5, default_timeout: float = 60.0):
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.session_hub = get_session_hub()
        self._pending_tasks: List[ThinkTask] = []
        self._running_tasks: Dict[str, ThinkTask] = {}
        self._completed_tasks: List[ThinkTask] = []
        self._task_dedup_map: Dict[str, str] = {}
        self._is_active = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._handlers: Dict[ThinkCategory, Callable] = {}
        self._stats = {
            "total_scheduled": 0,
            "total_executed": 0,
            "total_failed": 0,
            "total_cancelled": 0,
        }

    async def start(self) -> None:
        if self._is_active:
            logger.warning("ThinkScheduler 已在运行中")
            return
        self._is_active = True
        self._scheduler_task = asyncio.create_task(
            self._scheduler_loop(), name="think_scheduler_loop"
        )
        logger.info("ThinkScheduler 已启动")

    async def stop(self) -> None:
        if not self._is_active:
            return
        self._is_active = False
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                return
        for task in self._pending_tasks:
            task.is_cancelled = True
            self._stats["total_cancelled"] += 1
        self._pending_tasks.clear()
        self._task_dedup_map.clear()
        logger.info("ThinkScheduler 已停止")

    def register_handler(self, category: ThinkCategory, handler: Callable) -> None:
        self._handlers[category] = handler
        logger.debug(f"注册思考处理器: {category.value}")

    async def schedule_task(
        self, category: ThinkCategory, user_id: str, stream_id: str,
        priority: ThinkPriority = ThinkPriority.NORMAL,
        params: Optional[Dict[str, Any]] = None,
        delay_seconds: float = 0, allow_duplicate: bool = False,
    ) -> Optional[str]:
        if not self._is_active:
            logger.warning("调度器未运行，无法调度任务")
            return None
        dedup_key = f"{user_id}:{category.value}"
        if not allow_duplicate and dedup_key in self._task_dedup_map:
            existing_id = self._task_dedup_map[dedup_key]
            logger.debug(f"任务已存在，跳过: {dedup_key} -> {existing_id[:8]}...")
            return None
        task_id = str(uuid.uuid4())
        task = ThinkTask(
            task_id=task_id, category=category, priority=priority,
            target_user_id=user_id, target_stream_id=stream_id,
            params=params or {},
            scheduled_at=time.time() + delay_seconds if delay_seconds > 0 else None,
        )
        self._pending_tasks.append(task)
        self._pending_tasks.sort()
        self._task_dedup_map[dedup_key] = task_id
        self._stats["total_scheduled"] += 1
        logger.debug(
            f"调度思考任务: id={task_id[:8]}..., "
            f"category={category.value}, user={user_id}, priority={priority.value}"
        )
        return task_id

    async def cancel_task(self, task_id: str) -> bool:
        for task in self._pending_tasks:
            if task.task_id == task_id:
                task.is_cancelled = True
                self._pending_tasks.remove(task)
                self._stats["total_cancelled"] += 1
                dedup_key = f"{task.target_user_id}:{task.category.value}"
                self._task_dedup_map.pop(dedup_key, None)
                logger.debug(f"取消任务: {task_id[:8]}...")
                return True
        if task_id in self._running_tasks:
            self._running_tasks[task_id].is_cancelled = True
            logger.debug(f"标记运行中任务为取消: {task_id[:8]}...")
            return True
        return False

    async def cancel_tasks_for_user(self, user_id: str) -> int:
        cancelled_count = 0
        tasks_to_remove = [t for t in self._pending_tasks if t.target_user_id == user_id]
        for task in tasks_to_remove:
            task.is_cancelled = True
            self._pending_tasks.remove(task)
            cancelled_count += 1
            dedup_key = f"{task.target_user_id}:{task.category.value}"
            self._task_dedup_map.pop(dedup_key, None)
        for task in self._running_tasks.values():
            if task.target_user_id == user_id:
                task.is_cancelled = True
                cancelled_count += 1
        self._stats["total_cancelled"] += cancelled_count
        logger.debug(f"取消用户 {user_id} 的 {cancelled_count} 个任务")
        return cancelled_count

    async def _scheduler_loop(self) -> None:
        logger.debug("思考调度循环已启动")
        while self._is_active:
            try:
                await asyncio.sleep(0.5)
                executable_tasks = self._get_executable_tasks()
                if not executable_tasks:
                    continue
                for task in executable_tasks:
                    if len(self._running_tasks) >= self.max_concurrent:
                        break
                    self._pending_tasks.remove(task)
                    self._running_tasks[task.task_id] = task
                    asyncio.create_task(
                        self._execute_task(task),
                        name=f"think_task_{task.task_id[:8]}"
                    )
            except asyncio.CancelledError:
                logger.debug("思考调度循环被取消")
                break
            except Exception as e:
                logger.error(f"思考调度循环错误: {e}")

    def _get_executable_tasks(self) -> List[ThinkTask]:
        current_time = time.time()
        executable = []
        for task in self._pending_tasks:
            if task.is_cancelled:
                continue
            if task.scheduled_at and current_time < task.scheduled_at:
                continue
            executable.append(task)
            if len(executable) >= self.max_concurrent:
                break
        return executable

    async def _execute_task(self, task: ThinkTask) -> None:
        task.started_at = time.time()
        try:
            async with self._semaphore:
                if task.is_cancelled:
                    logger.debug(f"任务已取消，跳过执行: {task.task_id[:8]}...")
                    return
                handler = self._handlers.get(task.category)
                if not handler:
                    logger.warning(f"未找到处理器: {task.category.value}")
                    task.error_message = f"未找到处理器: {task.category.value}"
                    return
                try:
                    result = await asyncio.wait_for(
                        handler(task), timeout=self.default_timeout,
                    )
                    task.execution_result = result
                    task.is_executed = True
                    self._stats["total_executed"] += 1
                    logger.debug(
                        f"任务执行成功: id={task.task_id[:8]}..., "
                        f"category={task.category.value}"
                    )
                except asyncio.TimeoutError:
                    task.error_message = "执行超时"
                    self._stats["total_failed"] += 1
                    logger.warning(f"任务执行超时: {task.task_id[:8]}...")
                except Exception as e:
                    task.error_message = str(e)
                    self._stats["total_failed"] += 1
                    logger.error(f"任务执行失败: {task.task_id[:8]}..., error={e}")
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        task.started_at = None
                        task.is_executed = False
                        self._pending_tasks.append(task)
                        self._pending_tasks.sort()
                        logger.info(f"任务重试: {task.task_id[:8]}..., retry={task.retry_count}")
                        return
        finally:
            task.completed_at = time.time()
            self._running_tasks.pop(task.task_id, None)
            dedup_key = f"{task.target_user_id}:{task.category.value}"
            self._task_dedup_map.pop(dedup_key, None)
            self._completed_tasks.append(task)
            if len(self._completed_tasks) > 100:
                self._completed_tasks.pop(0)

    def get_pending_count(self) -> int:
        return len(self._pending_tasks)

    def get_running_count(self) -> int:
        return len(self._running_tasks)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "pending_tasks": len(self._pending_tasks),
            "running_tasks": len(self._running_tasks),
            "completed_tasks": len(self._completed_tasks),
            "is_active": self._is_active,
        }


_think_scheduler: Optional[ThinkScheduler] = None


def get_think_scheduler() -> ThinkScheduler:
    global _think_scheduler
    if _think_scheduler is None:
        _think_scheduler = ThinkScheduler()
    return _think_scheduler

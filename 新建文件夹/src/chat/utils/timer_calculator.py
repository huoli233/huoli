import asyncio
from time import perf_counter
from functools import wraps
from typing import Optional, Dict, Callable
from src.common.logger import get_logger

logger = get_logger("计时器")


class TimerTypeError(TypeError):
    """计时器类型错误"""

    __slots__ = ()

    def __init__(self, param_name: str, expected: str, actual: type):
        super().__init__(
            f"参数 '{param_name}' 类型错误，期望 {expected}，实际 {
                actual.__name__}"
        )


class Timer:
    """
    高精度计时器，支持三种模式：
    1. 装饰器模式：测量函数/协程执行时间
    2. 上下文管理器模式：with 语句块计时
    3. 直接实例化：获取当前 perf_counter 值
    """

    __slots__ = (
        "timer_name",
        "result_storage",
        "elapsed_time",
        "auto_unit",
        "start_time",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        storage: Optional[Dict[str, float]] = None,
        auto_unit: bool = True,
        type_check: bool = False,
    ):
        if type_check:
            self._check_types(name, storage)
        self.timer_name = name
        self.result_storage = storage
        self.elapsed_time: Optional[float] = None
        self.auto_unit = auto_unit
        self.start_time: Optional[float] = None

    @staticmethod
    def _check_types(name, storage):
        if name is not None and not isinstance(name, str):
            raise TimerTypeError("name", "Optional[str]", type(name))
        if storage is not None and not isinstance(storage, dict):
            raise TimerTypeError("storage", "Optional[dict]", type(storage))

    def __call__(self, func: Optional[Callable] = None) -> Callable:
        if func is None:
            return lambda f: Timer(
                name=self.timer_name or f.__name__,
                storage=self.result_storage,
                auto_unit=self.auto_unit,
            )(f)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with self:
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        wrapper = (
            async_wrapper
            if asyncio.iscoroutinefunction(func)
            else sync_wrapper
        )
        wrapper.__timer__ = self
        return wrapper

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_time = perf_counter() - self.start_time
        self._save_result()
        return False

    def _save_result(self):
        if self.result_storage is not None and self.timer_name:
            self.result_storage[self.timer_name] = self.elapsed_time

    @property
    def human_readable(self) -> str:
        if self.elapsed_time is None:
            return "未计时"
        if self.auto_unit:
            if self.elapsed_time < 1:
                return f"{self.elapsed_time * 1000:.2f}毫秒"
            return f"{self.elapsed_time:.2f}秒"
        return f"{self.elapsed_time:.4f}秒"

    def __str__(self):
        if self.start_time is not None:
            if self.elapsed_time is None:
                current = perf_counter() - self.start_time
                return f"<Timer {
                    self.timer_name or '匿名'} [计时中: {
                    current:.4f}秒]>"
            return (
                f"<Timer {self.timer_name or '匿名'} [{self.human_readable}]>"
            )
        return f"{perf_counter()}"


class Stopwatch:
    """秒表类，支持多次计时和统计"""

    __slots__ = ("_laps", "_start", "_running", "name")

    def __init__(self, name: Optional[str] = None):
        self.name = name or "秒表"
        self._laps: list[float] = []
        self._start: Optional[float] = None
        self._running = False

    def start(self) -> "Stopwatch":
        if self._running:
            logger.warning(f"{self.name} 已在运行中")
            return self
        self._start = perf_counter()
        self._running = True
        return self

    def lap(self) -> float:
        if not self._running:
            logger.warning(f"{self.name} 未启动")
            return 0.0
        elapsed = perf_counter() - self._start
        self._laps.append(elapsed)
        return elapsed

    def stop(self) -> float:
        if not self._running:
            logger.warning(f"{self.name} 未启动")
            return 0.0
        total = perf_counter() - self._start
        self._laps.append(total)
        self._running = False
        return total

    def reset(self) -> "Stopwatch":
        self._laps.clear()
        self._start = None
        self._running = False
        return self

    @property
    def total_time(self) -> float:
        if not self._laps:
            return 0.0
        return self._laps[-1]

    @property
    def lap_times(self) -> list[float]:
        return self._laps.copy()

    @property
    def average_lap(self) -> float:
        if len(self._laps) < 2:
            return 0.0
        intervals = [
            self._laps[i] - self._laps[i - 1]
            for i in range(1, len(self._laps))
        ]
        return sum(intervals) / len(intervals)

    def __str__(self):
        status = "运行中" if self._running else "已停止"
        return (
            f"<Stopwatch {self.name} [{status}] 总计: {self.total_time:.4f}秒>"
        )


def measure_time(
    name: Optional[str] = None, storage: Optional[Dict[str, float]] = None
):
    """计时装饰器工厂函数"""
    return Timer(name=name, storage=storage)


async def measure_async(func: Callable, *args, **kwargs) -> tuple:
    """测量异步函数执行时间"""
    timer = Timer(name=func.__name__)
    with timer:
        result = await func(*args, **kwargs)
    return result, timer.elapsed_time


def measure_sync(func: Callable, *args, **kwargs) -> tuple:
    """测量同步函数执行时间"""
    timer = Timer(name=func.__name__)
    with timer:
        result = func(*args, **kwargs)
    return result, timer.elapsed_time

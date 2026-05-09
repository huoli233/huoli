from collections import deque
from collections.abc import Coroutine
import asyncio
import copy
import time
from typing import Any, Dict, Optional

from src.common.logger import get_logger

logger = get_logger("运行态总线")
MONITOR_OVERVIEW_CHANNEL_ID = "__monitor_overview__"


def _plain(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, deque)):
        return [_plain(item, depth + 1) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _plain(value.to_dict(), depth + 1)
        except Exception:
            return repr(value)
    return repr(value)


def _deep_get(payload: Dict[str, Any], path: str) -> Any:
    cursor: Any = payload
    for part in path.split("."):
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
    return cursor


def _deep_set(payload: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    cursor = payload
    parts = [part for part in path.split(".") if part]
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    if parts:
        cursor[parts[-1]] = value
    return payload


class RuntimeStateHub:
    """WebUI 运行态总线：集中保存快照并向订阅者推送增量事件。"""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._channels: Dict[str, Dict[str, Any]] = {}
        self._versions: Dict[str, int] = {}
        self._subscribers: Dict[str, set[asyncio.Queue]] = {}
        self._history: Dict[str, deque[Dict[str, Any]]] = {}
        self._max_history = 200
        self._owner_loop: asyncio.AbstractEventLoop | None = None

    def _remember_loop(self) -> None:
        try:
            self._owner_loop = asyncio.get_running_loop()
        except RuntimeError:
            return

    def submit_coroutine(self, coro: Coroutine[Any, Any, Any], *, name: str = "") -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = self._owner_loop
            if loop is None or loop.is_closed() or not loop.is_running():
                coro.close()
                return None
            return asyncio.run_coroutine_threadsafe(coro, loop)
        return loop.create_task(coro, name=name or None)

    async def set_snapshot(self, channel_id: str, snapshot: Dict[str, Any], *, reason: str = "snapshot") -> Dict[str, Any]:
        self._remember_loop()
        channel = str(channel_id or "")
        payload = copy.deepcopy(snapshot or {})
        now = time.time()
        async with self._lock:
            version = self._versions.get(channel, 0) + 1
            payload["server_time"] = now
            payload["state_version"] = version
            self._channels[channel] = payload
            self._versions[channel] = version
            event = {
                "type": "state_snapshot",
                "channel_id": channel,
                "version": version,
                "server_time": now,
                "reason": reason,
                "data": copy.deepcopy(payload),
            }
            await self._publish_locked(channel, event)
            return copy.deepcopy(payload)

    async def update_path(
        self,
        channel_id: str,
        *,
        module: str,
        path: str,
        value: Any,
        reason: str = "",
    ) -> Optional[Dict[str, Any]]:
        self._remember_loop()
        channel = str(channel_id or "")
        if not channel or not path:
            return None
        now = time.time()
        clean_value = _plain(value)
        async with self._lock:
            snapshot = copy.deepcopy(self._channels.get(channel, {}))
            old = _deep_get(snapshot, path)
            if old == clean_value:
                return None
            version = self._versions.get(channel, 0) + 1
            _deep_set(snapshot, path, clean_value)
            snapshot["server_time"] = now
            snapshot["state_version"] = version
            self._channels[channel] = snapshot
            self._versions[channel] = version
            event = {
                "type": "state_delta",
                "channel_id": channel,
                "version": version,
                "server_time": now,
                "module": str(module or "runtime"),
                "path": str(path),
                "old": _plain(old),
                "new": clean_value,
                "reason": str(reason or module or "runtime_update"),
            }
            await self._publish_locked(channel, event)
            return copy.deepcopy(event)

    async def get_snapshot(self, channel_id: str) -> Optional[Dict[str, Any]]:
        self._remember_loop()
        channel = str(channel_id or "")
        async with self._lock:
            snapshot = self._channels.get(channel)
            return copy.deepcopy(snapshot) if snapshot is not None else None

    async def wait_for_update(
        self,
        channel_id: Optional[str],
        *,
        after_version: int = 0,
        timeout_seconds: float = 25.0,
    ) -> Optional[Dict[str, Any]]:
        self._remember_loop()
        channel = str(channel_id or "")
        timeout = max(0.1, min(float(timeout_seconds or 25.0), 30.0))
        async with self._lock:
            current_version = self._versions.get(channel, 0)
            if current_version > int(after_version or 0):
                snapshot = self._channels.get(channel)
                if snapshot is not None:
                    return {
                        "type": "state_snapshot",
                        "channel_id": channel,
                        "version": current_version,
                        "server_time": snapshot.get("server_time", time.time()),
                        "data": copy.deepcopy(snapshot),
                    }
            queue: asyncio.Queue = asyncio.Queue(maxsize=10)
            self._subscribers.setdefault(channel, set()).add(queue)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    return None
                if int(event.get("version", 0) or 0) > int(after_version or 0):
                    return copy.deepcopy(event)
        finally:
            await self.unsubscribe(queue, channel)

    async def subscribe(self, channel_id: Optional[str] = None) -> asyncio.Queue:
        self._remember_loop()
        channel = str(channel_id or "")
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers.setdefault(channel, set()).add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue, channel_id: Optional[str] = None) -> None:
        channel = str(channel_id or "")
        async with self._lock:
            subscribers = self._subscribers.get(channel)
            if subscribers is not None:
                subscribers.discard(queue)

    async def _publish_locked(self, channel_id: str, event: Dict[str, Any]) -> None:
        history = self._history.setdefault(channel_id, deque(maxlen=self._max_history))
        history.append(copy.deepcopy(event))
        targets = list(self._subscribers.get(channel_id, set())) + list(self._subscribers.get("", set()))
        for queue in targets:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(copy.deepcopy(event))
            except asyncio.QueueFull:
                logger.debug(f"运行态订阅队列已满 channel={channel_id}")


_runtime_state_hub: RuntimeStateHub | None = None


def get_runtime_state_hub() -> RuntimeStateHub:
    global _runtime_state_hub
    if _runtime_state_hub is None:
        _runtime_state_hub = RuntimeStateHub()
    return _runtime_state_hub


def emit_runtime_delta(
    channel_id: str,
    *,
    module: str,
    path: str,
    value: Any,
    reason: str = "",
) -> None:
    """从心流同步代码中安全投递运行态增量。"""
    hub = get_runtime_state_hub()
    hub.submit_coroutine(
        hub.update_path(
            str(channel_id or ""),
            module=module,
            path=path,
            value=value,
            reason=reason,
        ),
        name=f"runtime_delta_{module}",
    )

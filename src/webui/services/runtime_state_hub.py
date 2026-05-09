import asyncio
import copy
import time
from collections import deque
from typing import Any, Dict, Optional

from src.common.logger import get_logger

logger = get_logger("运行态总线")


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

    async def set_snapshot(self, channel_id: str, snapshot: Dict[str, Any], *, reason: str = "snapshot") -> Dict[str, Any]:
        channel = str(channel_id or "")
        payload = copy.deepcopy(snapshot or {})
        now = time.time()
        async with self._lock:
            previous = self._channels.get(channel)
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
            if previous is None:
                await self._publish_locked(channel, event)
            else:
                await self._publish_locked(
                    channel,
                    {
                        "type": "state_delta",
                        "channel_id": channel,
                        "version": version,
                        "server_time": now,
                        "module": "monitor",
                        "path": "snapshot",
                        "old": None,
                        "new": None,
                        "reason": reason,
                        "requires_refresh": True,
                    },
                )
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

    async def heartbeat(self, channel_id: Optional[str] = None) -> Dict[str, Any]:
        channel = str(channel_id or "")
        async with self._lock:
            version = self._versions.get(channel, 0) if channel else max(self._versions.values(), default=0)
        return {
            "type": "state_heartbeat",
            "channel_id": channel,
            "version": version,
            "server_time": time.time(),
        }

    async def get_snapshot(self, channel_id: str) -> Optional[Dict[str, Any]]:
        channel = str(channel_id or "")
        async with self._lock:
            snapshot = self._channels.get(channel)
            return copy.deepcopy(snapshot) if snapshot is not None else None

    async def subscribe(self, channel_id: Optional[str] = None) -> asyncio.Queue:
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
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(
        get_runtime_state_hub().update_path(
            str(channel_id or ""),
            module=module,
            path=path,
            value=value,
            reason=reason,
        )
    )

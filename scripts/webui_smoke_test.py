# ruff: noqa: E402
import asyncio
import json
import re
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.routing import WebSocketRoute

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import global_config
from src.webui.app import create_app
from src.webui.core.security import get_token_manager
from src.common.message.api import MessageAPIClient


IGNORE_HTTP_PATHS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
PATH_SAMPLES = {
    "plugin_id": "nonexistent_plugin",
    "mirror_id": "nonexistent-mirror",
    "emoji_id": "999999",
    "expression_id": "999999",
    "jargon_id": "999999",
    "person_id": "999999",
    "channel_id": "test-channel",
    "chat_id": "test-chat",
    "filename": "missing.log",
    "session_id": "missing-session",
    "task_id": "missing-task",
    "level": "info",
    "full_path": "",
}
SAFE_HTTP_OVERRIDES = {
    ("GET", "/api/webui/health"): {"headers": {}},
    ("POST", "/api/webui/auth/verify"): {"headers": {}, "json": None},
    ("POST", "/api/webui/auth/logout"): {},
    ("GET", "/api/webui/auth/check"): {},
    ("GET", "/api/webui/setup/status"): {},
    ("GET", "/system/startup-check"): {},
    ("GET", "/system/status"): {},
    ("POST", "/system/reload-config"): {},
    ("GET", "/plugins/version"): {},
    ("GET", "/plugins/git-status"): {},
    ("GET", "/plugins/mirrors"): {},
    ("GET", "/plugins/installed"): {},
    ("GET", "/ws/state-monitor/status"): {},
    ("GET", "/config/schema/bot"): {},
    ("GET", "/config/schema/model"): {},
    ("GET", "/config/bot"): {},
    ("GET", "/config/model"): {},
    ("GET", "/config/bot/raw"): {},
    ("GET", "/config/adapter-config/path"): {},
    ("GET", "/statistics/dashboard"): {},
    ("GET", "/statistics/summary"): {},
    ("GET", "/statistics/models"): {},
    ("GET", "/api/heartflow/chats"): {},
    ("GET", "/api/heartflow/monitor"): {},
    ("GET", "/dashboard"): {"headers": {}},
}
HTTP_SKIP_PATTERNS = [
    ("POST", re.compile(r"^/system/restart$")),
    ("POST", re.compile(r"^/system/model-capability$")),
    ("POST", re.compile(r"^/api/webui/auth/regenerate$")),
    ("POST", re.compile(r"^/api/webui/auth/update$")),
    ("POST", re.compile(r"^/api/webui/setup/complete$")),
    ("POST", re.compile(r"^/api/webui/setup/reset$")),
    ("POST", re.compile(r"^/config/.+$")),
    ("PUT", re.compile(r"^/plugins/.+$")),
    ("DELETE", re.compile(r"^/plugins/.+$")),
    ("POST", re.compile(r"^/plugins/(?!version|git-status|mirrors$).+$")),
    ("PATCH", re.compile(r"^/(emoji|expression|jargon|person)/.+$")),
    ("DELETE", re.compile(r"^/(emoji|expression|jargon|person|chat)/.+$")),
    ("POST", re.compile(r"^/(emoji|expression|jargon)/.+$")),
    ("DELETE", re.compile(r"^/api/chat/history$")),
]


@contextmanager
def app_context() -> Any:
    original_mode = global_config.webui.anti_crawler_mode
    global_config.webui.anti_crawler_mode = "false"
    try:
        app = create_app(enable_static=True)
        yield app
    finally:
        global_config.webui.anti_crawler_mode = original_mode


def fill_path(path: str) -> str:
    return re.sub(r"\{([^}]+)\}", lambda match: PATH_SAMPLES.get(match.group(1), "test"), path)


def should_skip_http(method: str, path: str) -> bool:
    return any(method == candidate_method and pattern.match(path) for candidate_method, pattern in HTTP_SKIP_PATTERNS)


def run_http_smoke(client: TestClient, token: str) -> dict[str, Any]:
    auth_headers = {"Authorization": f"Bearer {token}"}
    results = []

    for route in client.app.routes:
        if not isinstance(route, APIRoute) or route.path in IGNORE_HTTP_PATHS:
            continue
        methods = sorted(method for method in route.methods if method not in {"HEAD", "OPTIONS"})
        for method in methods:
            if should_skip_http(method, route.path):
                results.append({"method": method, "path": route.path, "status": "skipped", "detail": "side-effect route skipped"})
                continue

            kwargs = {"headers": auth_headers.copy()}
            override = SAFE_HTTP_OVERRIDES.get((method, route.path))
            if override:
                kwargs.update(override)
            if route.path.startswith("/api/heartflow/") or route.path == "/dashboard":
                kwargs["headers"] = {}
            if (method, route.path) == ("POST", "/api/webui/auth/verify"):
                kwargs["json"] = {"token": token}

            response = getattr(client, method.lower())(fill_path(route.path), **kwargs)
            status = "failed" if response.status_code >= 500 else "ok"
            detail = None
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    detail = payload.get("detail") or payload.get("message")
                    if method == "GET" and route.path == "/api/heartflow/monitor":
                        monitor = payload.get("monitor", {})
                        if not isinstance(monitor, dict) or "updated_at" not in monitor or "channels" not in monitor:
                            status = "failed"
                            detail = "monitor overview shape mismatch"
            except Exception:
                detail = None
            if method == "GET" and route.path == "/dashboard":
                body_text = response.text
                if response.status_code < 500 and "<div id=\"root\"></div>" not in body_text:
                    status = "failed"
                    detail = "dashboard html shape mismatch"
            results.append(
                {
                    "method": method,
                    "path": route.path,
                    "status": status,
                    "http_status": response.status_code,
                    "detail": detail,
                }
            )

    summary = Counter(item["status"] for item in results)
    failed = [item for item in results if item["status"] == "failed"]
    return {"summary": dict(summary), "total": len(results), "failed": failed}


def _expect_message(websocket, expected_type: str) -> dict[str, Any]:
    message = websocket.receive_json()
    if message.get("type") != expected_type:
        raise AssertionError(f"Expected websocket message type '{expected_type}', got '{message.get('type')}'")
    return message


def run_websocket_smoke(client: TestClient, token: str) -> dict[str, Any]:
    results = []
    registered = [route.path for route in client.app.routes if isinstance(route, WebSocketRoute)]

    with client.websocket_connect("/ws/state-monitor") as websocket:
        first = websocket.receive_json()
        if first.get("type") not in {"state_overview", "state_snapshot"}:
            raise AssertionError(f"Unexpected public state-monitor init message: {first}")
        payload = first.get("data", {})
        if first.get("type") == "state_overview":
            if "updated_at" not in payload or "channels" not in payload:
                raise AssertionError(f"Unexpected public state-overview shape: {payload}")
        else:
            for required_key in ("domains", "presentation", "prediction"):
                if required_key not in payload:
                    raise AssertionError(f"Unexpected public state-snapshot shape: missing {required_key}")
        websocket.send_json({"type": "ping", "data": {}})
        pong = _expect_message(websocket, "pong")
        results.append({"path": "/ws/state-monitor", "mode": "public", "status": "ok", "init_type": first.get("type"), "response_type": pong["type"]})

    with client.websocket_connect(f"/ws/auth?token={token}") as websocket:
        message = _expect_message(websocket, "auth_success")
        session_id = message["data"]["session_id"]
        websocket.send_json({"type": "ping", "data": {}})
        pong = _expect_message(websocket, "pong")
        results.append({"path": "/ws/auth", "status": "ok", "session_id": session_id, "response_type": pong["type"]})

    with client.websocket_connect(f"/ws/plugin-progress?token={token}") as websocket:
        first = websocket.receive_json()
        if first.get("type") not in {"active_tasks", "task_info"}:
            raise AssertionError(f"Unexpected plugin-progress init message: {first}")
        websocket.send_json({"type": "ping", "data": {}})
        pong = _expect_message(websocket, "pong")
        results.append({"path": "/ws/plugin-progress", "status": "ok", "init_type": first.get("type"), "response_type": pong["type"]})

    with client.websocket_connect(f"/ws/state-monitor?token={token}") as websocket:
        first = websocket.receive_json()
        if first.get("type") not in {"state_overview", "state_snapshot"}:
            raise AssertionError(f"Unexpected state-monitor init message: {first}")
        payload = first.get("data", {})
        if first.get("type") == "state_overview":
            if "updated_at" not in payload or "channels" not in payload:
                raise AssertionError(f"Unexpected state-overview shape: {payload}")
        else:
            for required_key in ("domains", "presentation", "prediction"):
                if required_key not in payload:
                    raise AssertionError(f"Unexpected state-snapshot shape: missing {required_key}")
        websocket.send_json({"type": "ping", "data": {}})
        pong = _expect_message(websocket, "pong")
        results.append({"path": "/ws/state-monitor", "mode": "token", "status": "ok", "init_type": first.get("type"), "response_type": pong["type"]})

    cookie_headers = {"Cookie": f"huoli_session={token}"}
    with client.websocket_connect("/ws/state-monitor", headers=cookie_headers) as websocket:
        first = websocket.receive_json()
        if first.get("type") not in {"state_overview", "state_snapshot"}:
            raise AssertionError(f"Unexpected state-monitor init message: {first}")
        payload = first.get("data", {})
        if first.get("type") == "state_overview":
            if "updated_at" not in payload or "channels" not in payload:
                raise AssertionError(f"Unexpected state-overview shape: {payload}")
        else:
            for required_key in ("domains", "presentation", "prediction"):
                if required_key not in payload:
                    raise AssertionError(f"Unexpected state-snapshot shape: missing {required_key}")
        websocket.send_json({"type": "ping", "data": {}})
        pong = _expect_message(websocket, "pong")
        results.append({"path": "/ws/state-monitor", "mode": "cookie", "status": "ok", "init_type": first.get("type"), "response_type": pong["type"]})

    with client.websocket_connect(f"/api/chat/ws?token={token}") as websocket:
        seen_types = []
        for _ in range(3):
            message = websocket.receive_json()
            seen_types.append(message.get("type"))
            if message.get("type") == "system":
                break
        if "session_info" not in seen_types:
            raise AssertionError(f"Chat websocket missing session_info, got {seen_types}")
        websocket.send_json({"type": "ping"})
        pong = _expect_message(websocket, "pong")
        results.append({"path": "/api/chat/ws", "status": "ok", "init_types": seen_types, "response_type": pong["type"]})

    return {
        "registered_routes": registered,
        "checked_routes": results,
    }


async def check_message_api_send_failure() -> dict[str, Any]:
    api = MessageAPIClient()
    await api.connect("127.0.0.1", 8080)
    try:
        send_result = await api.send_message({"message_type": "smoke", "content": "ping"})
        if send_result is not False:
            raise AssertionError("MessageAPIClient.send_message() should fail when no backend server is attached")
        return {"connected_without_server": True, "send_result": send_result}
    finally:
        await api.disconnect()


def check_state_monitor_activation_contract() -> dict[str, Any]:
    dashboard_source = (PROJECT_ROOT / "web/dashboard/src/components/EmotionDashboard/index.tsx").read_text(
        encoding="utf-8"
    )
    state_monitor_source = (PROJECT_ROOT / "src/webui/routers/websocket/state_monitor.py").read_text(
        encoding="utf-8"
    )

    required_dashboard_fragments = {
        "always_on_gate": "const shouldSyncRealtime = true;",
        "visibility_listener": 'document.addEventListener("visibilitychange", handleActivation);',
        "focus_listener": 'window.addEventListener("focus", handleActivation);',
        "stable_live_display": 'const displayConnectionState = "live";',
        "immediate_refresh": 'socket.send(JSON.stringify({ type: "refresh", data: { channel_id: selectedChannel } }));',
        "fast_silent_reconnect": "}, 800);",
    }
    missing_dashboard = [
        name for name, fragment in required_dashboard_fragments.items() if fragment not in dashboard_source
    ]
    assert not missing_dashboard, f"状态监控前端常驻连接契约缺失: {missing_dashboard}"
    assert "轮询同步失败" not in dashboard_source, "后台轮询失败不应显示到状态页前台"
    assert "状态页保持实时总览待机" not in dashboard_source, "后台 state_error 不应显示成前台错误"
    assert "authReady" not in dashboard_source, "状态页实时同步不应依赖 WebUI 登录态"
    assert "/api/webui/auth/check" not in dashboard_source, "状态页不应在建立实时连接前检查登录"
    assert 'window.addEventListener("blur"' not in dashboard_source, "状态页不应因窗口失焦断开实时连接"
    assert 'setConnectionState("idle")' not in dashboard_source, "状态页不应把临时不可见显示为未连接"

    forbidden_backend_fragments = [
        "verify_auth_token_from_cookie_or_header",
        "get_token_manager",
        "_authorize_state_monitor_websocket",
        "Cookie",
        "Header",
        "HTTPException",
    ]
    remaining_backend_fragments = [fragment for fragment in forbidden_backend_fragments if fragment in state_monitor_source]
    assert not remaining_backend_fragments, f"状态监控 WebSocket 仍残留认证门控: {remaining_backend_fragments}"
    assert "await websocket.accept()" in state_monitor_source, "状态监控 WebSocket 应直接接受只读连接"

    return {
        "activation_gate": True,
        "always_on_connection": True,
        "stable_live_display": True,
        "public_state_monitor": True,
        "no_auth_gate": True,
    }


def main() -> None:
    token = get_token_manager().get_token()
    with app_context() as app:
        with TestClient(app) as client:
            http_report = run_http_smoke(client, token)
            websocket_report = run_websocket_smoke(client, token)
    state_monitor_activation_report = check_state_monitor_activation_contract()
    message_api_report = asyncio.run(check_message_api_send_failure())
    print(
        json.dumps(
            {
                "http": http_report,
                "websocket": websocket_report,
                "state_monitor_activation": state_monitor_activation_report,
                "message_api": message_api_report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from src.common.logger import get_logger

logger = get_logger("webui.routers")


def iter_router_factories():
    """返回 WebUI 路由构造器，避免单个导入失败拖垮整组路由。"""
    return [
        (
            "auth",
            lambda: __import__("src.webui.routes", fromlist=["router"]).router,
        ),
        (
            "config",
            lambda: __import__("src.webui.routers.config", fromlist=["router"]).router,
        ),
        (
            "statistics",
            lambda: __import__("src.webui.routers.statistics", fromlist=["router"]).router,
        ),
        (
            "person",
            lambda: __import__("src.webui.routers.person", fromlist=["router"]).router,
        ),
        (
            "expression",
            lambda: __import__("src.webui.routers.expression", fromlist=["router"]).router,
        ),
        (
            "jargon",
            lambda: __import__("src.webui.routers.jargon", fromlist=["router"]).router,
        ),
        (
            "emoji",
            lambda: __import__("src.webui.routers.emoji", fromlist=["router"]).router,
        ),
        (
            "plugin",
            lambda: __import__("src.webui.routers.plugin", fromlist=["router"]).router,
        ),
        (
            "plugin_progress",
            lambda: __import__(
                "src.webui.routers.websocket.plugin_progress",
                fromlist=["get_progress_router"],
            ).get_progress_router(),
        ),
        (
            "system",
            lambda: __import__("src.webui.routers.system", fromlist=["router"]).router,
        ),
        (
            "model",
            lambda: __import__("src.webui.routers.model", fromlist=["router"]).router,
        ),
        (
            "ws_auth",
            lambda: __import__("src.webui.routers.websocket.auth", fromlist=["router"]).router,
        ),
        (
            "annual_report",
            lambda: __import__("src.webui.routers.annual_report", fromlist=["router"]).router,
        ),
        (
            "chat",
            lambda: __import__("src.webui.routers.chat", fromlist=["router"]).router,
        ),
        (
            "heartflow",
            lambda: __import__("src.webui.routers.heartflow", fromlist=["router"]).router,
        ),
        (
            "knowledge",
            lambda: __import__("src.webui.routers.knowledge", fromlist=["router"]).router,
        ),
    ]


def get_all_routers():
    """返回所有 WebUI 路由列表。"""
    routers = [router_factory() for _, router_factory in iter_router_factories()]
    logger.info("✅ 所有 WebUI 路由已准备好")
    return routers

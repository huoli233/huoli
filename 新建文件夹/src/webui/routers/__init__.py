"""WebUI 路由集合"""

from src.webui.routes import router as auth_router
from src.webui.routers.config import router as config_router
from src.webui.routers.statistics import router as statistics_router
from src.webui.routers.person import router as person_router
from src.webui.routers.expression import router as expression_router
from src.webui.routers.jargon import router as jargon_router
from src.webui.routers.emoji import router as emoji_router
from src.webui.routers.plugin import router as plugin_router
from src.webui.routers.websocket.plugin_progress import get_progress_router
from src.webui.routers.system import router as system_router
from src.webui.routers.model import router as model_router
from src.webui.routers.websocket.auth import router as ws_auth_router
from src.webui.routers.annual_report import router as annual_report_router
from src.webui.routers.chat import router as chat_router
from src.webui.routers.knowledge import router as knowledge_router
from src.common.logger import get_logger

logger = get_logger("webui.routers")


def get_all_routers():
    """返回所有 WebUI 路由列表"""
    routers = [
        auth_router,
        config_router,
        statistics_router,
        person_router,
        expression_router,
        jargon_router,
        emoji_router,
        plugin_router,
        get_progress_router(),
        system_router,
        model_router,
        ws_auth_router,
        annual_report_router,
        chat_router,
        knowledge_router,
    ]
    logger.info("✅ 所有 WebUI 路由已准备好")
    return routers


__all__ = ["get_all_routers"]

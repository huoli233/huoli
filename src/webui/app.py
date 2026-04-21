"""FastAPI 应用工厂 - 创建和配置 WebUI 应用实例"""

import mimetypes
from pathlib import Path

# 第三方库导入（保留原样）
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402

# 项目内部模块导入
from src.common.logger import get_logger  # noqa: E402
from src.common.constants import CORS_ALLOWED_ORIGINS  # noqa: E402

logger = get_logger("WebUI应用")


def _ensure_database_ready():
    """显式初始化数据库，避免仅导入 ORM 模块就触发副作用。"""
    try:
        from src.common.database.database_model import ensure_database_initialized

        ensure_database_initialized(sync_constraints=False)
    except Exception as exc:
        logger.error(f"❌ WebUI 数据库初始化失败: {exc}", exc_info=True)
        raise


def _ensure_core_config_ready():
    """显式启动核心配置中枢，避免 WebUI 独立路径缺少运行时配置。"""
    try:
        from src.config.core_config_engine import boot_core_config

        boot_core_config()
    except Exception as exc:
        logger.error(f"❌ WebUI 核心配置初始化失败: {exc}", exc_info=True)
        raise


def create_app(
    host: str = "0.0.0.0",
    port: int = 8001,
    enable_static: bool = True,
) -> FastAPI:
    """
    创建 WebUI FastAPI 应用实例

    Args:
        host: 服务器主机地址
        port: 服务器端口
        enable_static: 是否启用静态文件服务
    """
    app = FastAPI(title="Huoli Bot WebUI")
    _ensure_database_ready()
    _ensure_core_config_ready()

    _setup_anti_crawler(app)
    _setup_cors(app, port)
    _setup_global_error_handler(app)
    _setup_shutdown_hooks(app)
    _register_api_routes(app)
    _setup_robots_txt(app)

    if enable_static:
        _setup_dashboard_static_files(app)
        _setup_static_files(app)

    return app


def _setup_shutdown_hooks(app: FastAPI):
    """注册应用关闭时的资源清理钩子"""

    @app.on_event("shutdown")
    async def _cleanup_shared_resources():
        try:
            from src.webui.routers.model import shutdown_shared_client

            await shutdown_shared_client()
        except Exception as _e:
            logger.warning(f"关闭共享 HTTP 客户端异常: {_e}")


def _setup_global_error_handler(app: FastAPI):
    """注册全局未捕获异常处理器，防止内部堆栈/路径泄露到 HTTP 响应"""

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception(f"未捕获异常 [{request.method} {request.url.path}]: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误，请查看日志或联系管理员"},
        )


def _setup_cors(app: FastAPI, port: int):
    cors_origins = list(CORS_ALLOWED_ORIGINS) + [
        f"http://localhost:{port}",
        f"http://127.0.0.1:{port}",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "Accept",
            "Origin",
            "X-Requested-With",
        ],
        expose_headers=["Content-Length", "Content-Type"],
    )
    logger.debug("CORS 中间件已配置")


def _setup_anti_crawler(app: FastAPI):
    try:
        from src.webui.middleware.anti_crawler import AntiCrawlerMiddleware
        from src.config.config import global_config

        anti_crawler_mode = global_config.webui.anti_crawler_mode
        app.add_middleware(AntiCrawlerMiddleware, mode=anti_crawler_mode)

        mode_descriptions = {
            "false": "已禁用",
            "strict": "严格模式",
            "loose": "宽松模式",
            "basic": "基础模式",
        }
        mode_desc = mode_descriptions.get(anti_crawler_mode, "基础模式")
        logger.info(f"🛡️ 防爬虫中间件已配置: {mode_desc}")
    except Exception as e:
        logger.error(f"❌ 配置防爬虫中间件失败: {e}", exc_info=True)


def _setup_robots_txt(app: FastAPI):
    try:
        from src.webui.middleware.anti_crawler import create_robots_txt_response

        @app.get("/robots.txt", include_in_schema=False)
        async def robots_txt():
            return create_robots_txt_response()

        logger.debug("✅ robots.txt 路由已注册")
    except Exception as e:
        logger.error(f"❌ 注册robots.txt路由失败: {e}", exc_info=True)


def _register_api_routes(app: FastAPI):
    try:
        from src.webui.routers.router_registry import iter_router_factories

        registered_count = 0
        for router_name, router_factory in iter_router_factories():
            try:
                app.include_router(router_factory())
                registered_count += 1
            except Exception as router_exc:
                logger.error(
                    f"❌ 注册 WebUI 路由失败: {router_name}: {router_exc}",
                    exc_info=True,
                )

        logger.info(f"✅ WebUI API 路由已注册: {registered_count}")
    except Exception as e:
        logger.error(f"❌ 注册 WebUI API 路由失败: {e}", exc_info=True)


def _setup_static_files(app: FastAPI):
    mimetypes.init()
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("application/javascript", ".mjs")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("application/json", ".json")

    base_dir = Path(__file__).parent.parent.parent
    static_path = base_dir / "webui" / "dist"

    if not static_path.exists():
        logger.warning(f"❌ WebUI 静态文件目录不存在: {static_path}")
        logger.warning("💡 WebUI API 接口可正常使用，前端请构建后放入 webui/dist/ 目录")
        return

    if not (static_path / "index.html").exists():
        logger.warning(f"❌ 未找到 index.html: {static_path / 'index.html'}")
        logger.warning("💡 WebUI API 接口可正常使用，前端请构建后放入 webui/dist/ 目录")
        return

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        if not full_path or full_path == "/":
            response = FileResponse(static_path / "index.html", media_type="text/html")
            response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
            return response

        file_path = static_path / full_path
        # 路径遍历防御：确保解析后的实际路径仍在静态资源目录内
        try:
            resolved_file = file_path.resolve()
            if not resolved_file.is_relative_to(static_path.resolve()):
                response = FileResponse(static_path / "index.html", media_type="text/html")
                response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
                return response
        except (ValueError, OSError):
            response = FileResponse(static_path / "index.html", media_type="text/html")
            response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
            return response
        if resolved_file.is_file():
            media_type = mimetypes.guess_type(str(resolved_file))[0]
            response = FileResponse(resolved_file, media_type=media_type)
            if str(resolved_file).endswith(".html"):
                response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
            return response

        response = FileResponse(static_path / "index.html", media_type="text/html")
        response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
        return response

    logger.info(f"✅ WebUI 静态文件服务已配置: {static_path}")


def _setup_dashboard_static_files(app: FastAPI):
    mimetypes.init()
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("application/javascript", ".mjs")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("application/json", ".json")

    base_dir = Path(__file__).parent.parent.parent
    dashboard_path = base_dir / "web" / "dashboard" / "dist"

    if not dashboard_path.exists():
        logger.warning(f"❌ 状态页静态文件目录不存在: {dashboard_path}")
        logger.warning("💡 独立状态页请先构建到 web/dashboard/dist/ 目录")
        return

    if not (dashboard_path / "index.html").exists():
        logger.warning(f"❌ 未找到状态页 index.html: {dashboard_path / 'index.html'}")
        logger.warning("💡 独立状态页请先构建到 web/dashboard/dist/ 目录")
        return

    @app.get("/dashboard", include_in_schema=False)
    @app.get("/dashboard/{full_path:path}", include_in_schema=False)
    async def serve_dashboard_spa(full_path: str = ""):
        requested = full_path or "index.html"
        file_path = dashboard_path / requested
        try:
            resolved_file = file_path.resolve()
            if not resolved_file.is_relative_to(dashboard_path.resolve()):
                response = FileResponse(dashboard_path / "index.html", media_type="text/html")
                response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
                return response
        except (ValueError, OSError):
            response = FileResponse(dashboard_path / "index.html", media_type="text/html")
            response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
            return response

        if resolved_file.is_file():
            media_type = mimetypes.guess_type(str(resolved_file))[0]
            response = FileResponse(resolved_file, media_type=media_type)
            if str(resolved_file).endswith(".html"):
                response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
            return response

        response = FileResponse(dashboard_path / "index.html", media_type="text/html")
        response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
        return response

    logger.info(f"✅ 独立状态页静态服务已配置: {dashboard_path}")


def show_access_token():
    """显示 WebUI Access Token（供启动时调用）"""
    try:
        from src.webui.core.security import get_token_manager, _token_digest

        token_manager = get_token_manager()
        current_token = token_manager.get_token()
        logger.info(f"WebUI Access Token: {_token_digest(current_token)}...")
        logger.info("请使用此 Token 登录 WebUI")
    except Exception as e:
        logger.error(f"❌ 获取 Access Token 失败: {e}")

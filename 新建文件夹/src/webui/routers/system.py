import asyncio
import os
import time
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Cookie, Header
from pydantic import BaseModel, Field

from src.common.database.database import db, ROOT_PATH
from src.common.logger import get_logger
from src.config.config import (
    MMC_VERSION,
    global_config,
    model_config,
    CONFIG_DIR,
    load_config,
    api_ada_load_config,
)
from src.llm_models.model_client._registry import client_registry
from src.llm_models.utils_model import LLMRequest
from src.webui.core import verify_auth_token_from_cookie_or_header

router = APIRouter(prefix="/system", tags=["system"])
logger = get_logger("WebUI系统")

# 记录启动时间
_start_time = time.time()


def require_auth(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
) -> bool:
    """认证依赖：验证用户是否已登录"""
    return verify_auth_token_from_cookie_or_header(
        huoli_session, authorization
    )


class RestartResponse(BaseModel):
    """重启响应"""

    success: bool
    message: str


class StatusResponse(BaseModel):
    """状态响应"""

    running: bool
    uptime: float
    version: str
    start_time: str


class StartupCheckResponse(BaseModel):
    """启动自检结果"""

    success: bool
    checks: dict[str, Any]


class ModelCapabilityRequest(BaseModel):
    """模型能力探测请求"""

    tasks: Optional[list[str]] = Field(
        default=None, description="要探测的任务名列表，不传则探测全部"
    )
    prompt: str = Field(default="请仅回复：OK", description="模型探测提示词")
    max_tokens: int = Field(default=64, ge=8, le=2048)
    timeout_seconds: float = Field(default=30.0, ge=5.0, le=120.0)


class ModelCapabilityResponse(BaseModel):
    """模型能力探测结果"""

    success: bool
    summary: dict[str, int]
    results: list[dict[str, Any]]


def _collect_model_tasks() -> dict[str, Any]:
    """收集 model_task_config 下的任务配置。"""
    task_config = model_config.model_task_config
    if not is_dataclass(task_config):
        return {}
    return {
        item.name: getattr(task_config, item.name)
        for item in fields(task_config)
    }


@router.get("/startup-check", response_model=StartupCheckResponse)
async def startup_check(_auth: bool = Depends(require_auth)):
    """启动后系统自检接口，用于快速确认关键模块是否可用。"""
    checks: dict[str, Any] = {
        "config_loaded": True,
        "model_config_loaded": True,
        "webui_enabled": bool(getattr(global_config.webui, "enabled", False)),
        "project_root_exists": Path(ROOT_PATH).exists(),
    }

    db_ok = False
    db_error = ""
    was_closed = db.is_closed()
    try:
        db.connect(reuse_if_open=True)
        db.execute_sql("SELECT 1")
        db_ok = True
    except Exception as exc:
        db_error = str(exc)
    finally:
        if was_closed and not db.is_closed():
            db.close()

    checks["database_ready"] = db_ok
    if db_error:
        checks["database_error"] = db_error

    success = all(
        bool(v)
        for k, v in checks.items()
        if k.endswith("_loaded")
        or k.endswith("_ready")
        or k.endswith("_exists")
    )
    return StartupCheckResponse(success=success, checks=checks)


@router.post("/model-capability", response_model=ModelCapabilityResponse)
async def model_capability_probe(
    payload: ModelCapabilityRequest, _auth: bool = Depends(require_auth)
):
    """探测模型能力接口，按任务配置逐个发起最小请求验证可用性。"""
    task_map = _collect_model_tasks()
    if not task_map:
        raise HTTPException(status_code=500, detail="未找到模型任务配置")

    selected_tasks = payload.tasks or list(task_map.keys())
    invalid_tasks = [name for name in selected_tasks if name not in task_map]
    if invalid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"无效任务名: {
                ', '.join(invalid_tasks)}",
        )

    async def probe_single(task_name: str) -> dict[str, Any]:
        task_cfg = task_map[task_name]
        checker = LLMRequest(
            model_set=task_cfg, request_type=f"diagnose_{task_name}"
        )
        started = time.time()

        try:
            if task_name == "embedding":
                vector, model_name = await asyncio.wait_for(
                    checker.get_embedding("系统诊断：模型能力探测"),
                    timeout=payload.timeout_seconds,
                )
                return {
                    "task": task_name,
                    "ok": bool(vector),
                    "model": model_name,
                    "latency": round(time.time() - started, 3),
                    "detail": f"embedding_dim={len(vector)}",
                }

            if task_name == "voice":
                return {
                    "task": task_name,
                    "ok": None,
                    "model": "N/A",
                    "latency": round(time.time() - started, 3),
                    "detail": "语音任务需真实音频输入，已跳过在线调用",
                    "skipped": True,
                }

            content, meta = await asyncio.wait_for(
                checker.generate_response_async(
                    prompt=payload.prompt,
                    max_tokens=payload.max_tokens,
                    temperature=0.1,
                ),
                timeout=payload.timeout_seconds,
            )
            if not (isinstance(meta, tuple) and len(meta) > 1):
                logger.warning(
                    f"任务 {task_name} 的模型响应元信息格式异常: {type(meta)}"
                )
            model_name = (
                meta[1]
                if isinstance(meta, tuple) and len(meta) > 1
                else "unknown"
            )
            ok = bool((content or "").strip())
            return {
                "task": task_name,
                "ok": ok,
                "model": model_name,
                "latency": round(time.time() - started, 3),
                "detail": "响应正常" if ok else "响应为空",
            }
        except Exception as exc:
            return {
                "task": task_name,
                "ok": False,
                "model": "unknown",
                "latency": round(time.time() - started, 3),
                "detail": str(exc),
            }

    results = await asyncio.gather(
        *(probe_single(name) for name in selected_tasks)
    )
    skipped_count = sum(1 for item in results if item.get("skipped") is True)
    ok_count = sum(1 for item in results if item.get("ok") is True)
    fail_count = sum(1 for item in results if item.get("ok") is False)

    return ModelCapabilityResponse(
        success=fail_count == 0,
        summary={
            "total": len(results),
            "ok": ok_count,
            "failed": fail_count,
            "skipped": skipped_count,
        },
        results=results,
    )


_RESTART_CODE = 42


@router.post("/restart", response_model=RestartResponse)
async def request_restart(_auth: bool = Depends(require_auth)):
    """请求重启活力进程，配置更改将在重启后生效。"""
    logger.info("收到 WebUI 重启请求，即将退出进程")
    loop = asyncio.get_running_loop()
    loop.call_later(0.5, os._exit, _RESTART_CODE)
    return RestartResponse(success=True, message="活力正在重启，请稍候...")


@router.get("/status", response_model=StatusResponse)
async def get_running_status(_auth: bool = Depends(require_auth)):
    """查询活力运行状态、运行时长和版本号。"""
    elapsed = time.time() - _start_time
    return StatusResponse(
        running=True,
        uptime=elapsed,
        version=MMC_VERSION,
        start_time=datetime.fromtimestamp(_start_time).isoformat(),
    )


@router.post("/reload-config")
async def reload_configuration(_auth: bool = Depends(require_auth)):
    """热重载配置文件，重新读取 bot_config 和 model_config。"""
    try:
        bot_path = os.path.join(CONFIG_DIR, "bot_config.toml")
        model_path = os.path.join(CONFIG_DIR, "model_config.toml")
        new_bot = load_config(config_path=bot_path)
        new_model = api_ada_load_config(config_path=model_path)
        for f in fields(new_bot):
            if f.init:
                setattr(global_config, f.name, getattr(new_bot, f.name))
        for f in fields(new_model):
            if f.init:
                setattr(model_config, f.name, getattr(new_model, f.name))
        client_registry.clear_client_instance_cache()
        logger.info("配置文件已通过 WebUI 热重载")
        return {"success": True, "message": "配置已重新加载"}
    except Exception as exc:
        logger.error(f"配置热重载失败: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"配置重载失败: {
                str(exc)}",
        ) from exc

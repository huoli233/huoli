import copy
import os
import json
import tomlkit
from fastapi import APIRouter, HTTPException, Body, Depends, Cookie, Header
from typing import Any, Annotated, Optional
from src.common.logger import get_logger
from src.webui.core.auth import verify_auth_token_from_cookie_or_header
from src.config.config import CONFIG_DIR, PROJECT_ROOT

ConfigBody = Annotated[dict[str, Any], Body()]
SectionBody = Annotated[Any, Body()]
RawContentBody = Annotated[str, Body(embed=True)]
PathBody = Annotated[dict[str, str], Body()]
logger = get_logger("WebUI配置")
router = APIRouter(prefix="/config", tags=["config"])

# bot_config.toml 允许修改的顶层节白名单
BOT_CONFIG_ALLOWED_SECTIONS = {
    "bot", "personality", "bot_identity", "expression", "chat",
    "memory", "tool", "emoji", "voice", "message_receive",
    "lpmm_knowledge", "keyword_reaction", "response_post_process",
    "chinese_typo", "response_splitter", "log", "debug",
    "maim_message", "telemetry", "webui", "experimental", "relationship",
}

# model_config.toml 允许修改的顶层节白名单
MODEL_CONFIG_ALLOWED_SECTIONS = {
    "api_providers", "models", "model_task_config",
}

# 需要脱敏的敏感字段关键词
_SENSITIVE_KEY_PATTERNS = {"api_key", "apikey", "secret", "password", "token", "auth_token"}


def _is_sensitive_key(key: str) -> bool:
    """判断字段名是否为敏感字段"""
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in _SENSITIVE_KEY_PATTERNS)


def _mask_sensitive_value(value: Any) -> Any:
    """对敏感值进行脱敏处理"""
    if isinstance(value, str) and len(value) > 8:
        return value[:4] + "****" + value[-4:]
    return "****"


def _mask_sensitive_data(data: Any) -> Any:
    """
    递归对配置中的敏感字段进行脱敏处理
    支持字典、列表的嵌套结构
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if _is_sensitive_key(key):
                result[key] = _mask_sensitive_value(value)
            elif isinstance(value, (dict, list)):
                result[key] = _mask_sensitive_data(value)
            else:
                result[key] = value
        return result
    elif isinstance(data, list):
        return [_mask_sensitive_data(item) for item in data]
    return data


def _filter_allowed_sections(
    data: dict, allowed: set[str]
) -> dict:
    """过滤掉不在白名单中的顶层节"""
    filtered = {}
    for key, value in data.items():
        if key in allowed:
            filtered[key] = value
        else:
            logger.warning(f"配置写入被拒绝: 不允许修改的节 '{key}'")
    return filtered


def require_auth(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
) -> bool:
    """认证依赖：验证用户是否已登录"""
    return verify_auth_token_from_cookie_or_header(
        huoli_session, authorization
    )


def save_toml_with_format(data: dict, file_path: str) -> None:
    """保存 TOML 文件，保留格式"""
    doc = tomlkit.document()
    for key, value in data.items():
        if isinstance(value, list):
            table = tomlkit.aot()
            for item in value:
                if isinstance(item, dict):
                    table.append(_dict_to_table(item))
                else:
                    table.append(item)
            doc[key] = table
        elif isinstance(value, dict):
            doc[key] = _dict_to_table(value)
        else:
            doc[key] = value
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(doc))


def _dict_to_table(d: dict) -> tomlkit.table:
    """将字典转换为 TOML 表格"""
    table = tomlkit.table()
    for k, v in d.items():
        if isinstance(v, dict):
            table[k] = _dict_to_table(v)
        elif isinstance(v, list):
            aot = tomlkit.aot()
            for item in v:
                if isinstance(item, dict):
                    aot.append(_dict_to_table(item))
                else:
                    aot.append(item)
            table[k] = aot
        else:
            table[k] = v
    return table


def _update_toml_doc(target: dict, source: dict) -> None:
    """递归更新 TOML 文档，保留注释"""
    for key, value in source.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            _update_toml_doc(target[key], value)
        else:
            target[key] = value


@router.get("/schema/bot")
async def get_bot_config_schema(_auth: bool = Depends(require_auth)):
    """获取主程序配置架构"""
    try:
        schema = _generate_config_schema()
        return {"success": True, "schema": schema}
    except Exception as e:
        logger.error(f"获取配置架构失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取配置架构失败: {
                str(e)}",
        ) from e


@router.get("/schema/model")
async def get_model_config_schema(_auth: bool = Depends(require_auth)):
    """获取模型配置架构"""
    try:
        schema = _generate_model_schema()
        return {"success": True, "schema": schema}
    except Exception as e:
        logger.error(f"获取模型配置架构失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取模型配置架构失败: {
                str(e)}",
        ) from e


def _generate_config_schema() -> dict:
    """生成配置架构"""
    return {
        "type": "object",
        "properties": {
            "bot": {
                "type": "object",
                "properties": {
                    "nickname": {
                        "type": "string",
                        "description": "机器人昵称",
                    },
                    "qq_account": {"type": "string", "description": "QQ账号"},
                },
            },
            "personality": {
                "type": "object",
                "properties": {
                    "basic_info": {
                        "type": "string",
                        "description": "基本信息",
                    },
                    "character": {"type": "string", "description": "性格特点"},
                },
            },
            "chat": {
                "type": "object",
                "properties": {
                    "max_context_length": {
                        "type": "integer",
                        "description": "最大上下文长度",
                    },
                    "reply_probability": {
                        "type": "number",
                        "description": "回复概率",
                    },
                },
            },
        },
    }


def _generate_model_schema() -> dict:
    """生成模型配置架构"""
    return {
        "type": "object",
        "properties": {
            "api_providers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "提供商名称",
                        },
                        "base_url": {
                            "type": "string",
                            "description": "基础URL",
                        },
                        "api_key": {
                            "type": "string",
                            "description": "API密钥",
                        },
                        "client_type": {
                            "type": "string",
                            "description": "客户端类型",
                        },
                    },
                },
            },
            "models": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "模型名称"},
                        "api_provider": {
                            "type": "string",
                            "description": "API提供商",
                        },
                        "model_name": {
                            "type": "string",
                            "description": "模型标识",
                        },
                    },
                },
            },
        },
    }


@router.get("/bot")
async def get_bot_config(_auth: bool = Depends(require_auth)):
    """获取主程序配置（敏感字段已脱敏）"""
    try:
        config_path = os.path.join(CONFIG_DIR, "bot_config.toml")
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail="配置文件不存在")
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = tomlkit.load(f)
        # 对敏感字段进行脱敏处理
        masked_config = _mask_sensitive_data(config_data)
        return {"success": True, "config": masked_config}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"读取配置文件失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"读取配置文件失败: {
                str(e)}",
        ) from e


@router.get("/model")
async def get_model_config(_auth: bool = Depends(require_auth)):
    """获取模型配置（敏感字段已脱敏）"""
    try:
        config_path = os.path.join(CONFIG_DIR, "model_config.toml")
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail="配置文件不存在")
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = tomlkit.load(f)
        # 对敏感字段进行脱敏处理
        masked_config = _mask_sensitive_data(config_data)
        return {"success": True, "config": masked_config}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"读取配置文件失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"读取配置文件失败: {
                str(e)}",
        ) from e


@router.post("/bot")
async def update_bot_config(
    config_data: ConfigBody, _auth: bool = Depends(require_auth)
):
    """更新主程序配置（仅允许白名单中的节）"""
    try:
        # 白名单过滤：只允许修改已知的安全配置节
        filtered = _filter_allowed_sections(config_data, BOT_CONFIG_ALLOWED_SECTIONS)
        if not filtered:
            raise HTTPException(
                status_code=400, detail="无有效配置字段，所有字段均不在白名单中"
            )
        config_path = os.path.join(CONFIG_DIR, "bot_config.toml")
        save_toml_with_format(filtered, config_path)
        logger.info("主程序配置已更新")
        return {"success": True, "message": "配置已保存"}
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"保存配置文件失败: {
                str(e)}",
        ) from e


@router.post("/model")
async def update_model_config(
    config_data: ConfigBody, _auth: bool = Depends(require_auth)
):
    """更新模型配置（仅允许白名单中的节）"""
    try:
        # 白名单过滤：只允许修改已知的安全配置节
        filtered = _filter_allowed_sections(config_data, MODEL_CONFIG_ALLOWED_SECTIONS)
        if not filtered:
            raise HTTPException(
                status_code=400, detail="无有效配置字段，所有字段均不在白名单中"
            )
        config_path = os.path.join(CONFIG_DIR, "model_config.toml")
        save_toml_with_format(filtered, config_path)
        logger.info("模型配置已更新")
        return {"success": True, "message": "配置已保存"}
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"保存配置文件失败: {
                str(e)}",
        ) from e


@router.post("/bot/section/{section_name}")
async def update_bot_config_section(
    section_name: str,
    section_data: SectionBody,
    _auth: bool = Depends(require_auth),
):
    """更新主程序配置的指定节（仅允许白名单中的节）"""
    try:
        # 白名单验证：检查节名是否在允许列表中
        if section_name not in BOT_CONFIG_ALLOWED_SECTIONS:
            raise HTTPException(
                status_code=403, detail=f"不允许修改的配置节 '{section_name}'"
            )
        config_path = os.path.join(CONFIG_DIR, "bot_config.toml")
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail="配置文件不存在")
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = tomlkit.load(f)
        if section_name not in config_data:
            raise HTTPException(
                status_code=404, detail=f"配置节 '{section_name}' 不存在"
            )
        if isinstance(section_data, list):
            config_data[section_name] = section_data
        elif isinstance(section_data, dict) and isinstance(
            config_data[section_name], dict
        ):
            _update_toml_doc(config_data[section_name], section_data)
        else:
            config_data[section_name] = section_data
        save_toml_with_format(config_data, config_path)
        logger.info(f"配置节 '{section_name}' 已更新")
        return {"success": True, "message": f"配置节 '{section_name}' 已保存"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新配置节失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"更新配置节失败: {
                str(e)}",
        ) from e


@router.post("/model/section/{section_name}")
async def update_model_config_section(
    section_name: str,
    section_data: SectionBody,
    _auth: bool = Depends(require_auth),
):
    """更新模型配置的指定节（仅允许白名单中的节）"""
    try:
        # 白名单验证：检查节名是否在允许列表中
        if section_name not in MODEL_CONFIG_ALLOWED_SECTIONS:
            raise HTTPException(
                status_code=403, detail=f"不允许修改的配置节 '{section_name}'"
            )
        config_path = os.path.join(CONFIG_DIR, "model_config.toml")
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail="配置文件不存在")
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = tomlkit.load(f)
        if section_name not in config_data:
            raise HTTPException(
                status_code=404, detail=f"配置节 '{section_name}' 不存在"
            )
        if isinstance(section_data, list):
            config_data[section_name] = section_data
        elif isinstance(section_data, dict) and isinstance(
            config_data[section_name], dict
        ):
            _update_toml_doc(config_data[section_name], section_data)
        else:
            config_data[section_name] = section_data
        save_toml_with_format(config_data, config_path)
        logger.info(f"配置节 '{section_name}' 已更新")
        return {"success": True, "message": f"配置节 '{section_name}' 已保存"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新配置节失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"更新配置节失败: {
                str(e)}",
        ) from e


@router.get("/bot/raw")
async def get_bot_config_raw(_auth: bool = Depends(require_auth)):
    """获取主程序配置的原始 TOML 内容"""
    try:
        config_path = os.path.join(CONFIG_DIR, "bot_config.toml")
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail="配置文件不存在")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
        return {"success": True, "content": raw_content}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"读取配置文件失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"读取配置文件失败: {
                str(e)}",
        ) from e


@router.post("/bot/raw")
async def update_bot_config_raw(
    raw_content: RawContentBody, _auth: bool = Depends(require_auth)
):
    """更新主程序配置（直接保存原始 TOML 内容）"""
    try:
        try:
            tomlkit.loads(raw_content)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"TOML 格式错误: {
                    str(e)}",
            ) from e
        config_path = os.path.join(CONFIG_DIR, "bot_config.toml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(raw_content)
        logger.info("主程序配置已更新（原始模式）")
        return {"success": True, "message": "配置已保存"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"保存配置文件失败: {
                str(e)}",
        ) from e


def _normalize_adapter_path(path: str) -> str:
    """将路径转换为绝对路径，对相对路径做越界检查"""
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.realpath(path)
    resolved = os.path.realpath(os.path.join(PROJECT_ROOT, path))
    real_root = os.path.realpath(PROJECT_ROOT)
    if not resolved.startswith(real_root + os.sep) and resolved != real_root:
        raise ValueError(f"相对路径越界: {path}")
    return resolved


def _to_relative_path(path: str) -> str:
    """尝试将绝对路径转换为相对路径"""
    if not path or not os.path.isabs(path):
        return path
    try:
        rel_path = os.path.relpath(path, PROJECT_ROOT)
        if not rel_path.startswith(".."):
            return rel_path
    except (ValueError, TypeError):
        pass
    return path


@router.get("/adapter-config/path")
async def get_adapter_config_path(_auth: bool = Depends(require_auth)):
    """获取保存的适配器配置文件路径"""
    try:
        webui_data_path = os.path.join("data", "webui.json")
        if not os.path.exists(webui_data_path):
            return {"success": True, "path": None}
        with open(webui_data_path, "r", encoding="utf-8") as f:
            webui_data = json.load(f)
        adapter_config_path = webui_data.get("adapter_config_path")
        if not adapter_config_path:
            return {"success": True, "path": None}
        abs_path = _normalize_adapter_path(adapter_config_path)
        if os.path.exists(abs_path):
            import datetime

            mtime = os.path.getmtime(abs_path)
            last_modified = datetime.datetime.fromtimestamp(mtime).isoformat()
            display_path = _to_relative_path(abs_path)
            return {
                "success": True,
                "path": display_path,
                "lastModified": last_modified,
            }
        else:
            return {
                "success": True,
                "path": adapter_config_path,
                "lastModified": None,
            }
    except ValueError as path_err:
        raise HTTPException(
            status_code=403, detail=str(path_err)
        ) from path_err
    except Exception as e:
        logger.error(f"获取适配器配置路径失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取配置路径失败: {
                str(e)}",
        ) from e


@router.post("/adapter-config/path")
async def save_adapter_config_path(
    data: PathBody, _auth: bool = Depends(require_auth)
):
    """保存适配器配置文件路径偏好"""
    try:
        path = data.get("path")
        if not path:
            raise HTTPException(status_code=400, detail="路径不能为空")
        webui_data_path = os.path.join("data", "webui.json")
        if os.path.exists(webui_data_path):
            with open(webui_data_path, "r", encoding="utf-8") as f:
                webui_data = json.load(f)
        else:
            webui_data = {}
        abs_path = _normalize_adapter_path(path)
        save_path = _to_relative_path(abs_path)
        webui_data["adapter_config_path"] = save_path
        os.makedirs("data", exist_ok=True)
        with open(webui_data_path, "w", encoding="utf-8") as f:
            json.dump(webui_data, f, ensure_ascii=False, indent=2)
        logger.info(f"适配器配置路径已保存: {save_path}")
        return {"success": True, "message": "路径已保存"}
    except HTTPException:
        raise
    except ValueError as path_err:
        raise HTTPException(
            status_code=403, detail=str(path_err)
        ) from path_err
    except Exception as e:
        logger.error(f"保存适配器配置路径失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"保存路径失败: {str(e)}"
        ) from e


@router.get("/adapter-config")
async def get_adapter_config(path: str, _auth: bool = Depends(require_auth)):
    """从指定路径读取适配器配置文件"""
    try:
        if not path:
            raise HTTPException(status_code=400, detail="路径参数不能为空")
        abs_path = _normalize_adapter_path(path)
        if not os.path.exists(abs_path):
            raise HTTPException(
                status_code=404, detail=f"配置文件不存在: {path}"
            )
        if not abs_path.endswith(".toml"):
            raise HTTPException(
                status_code=400, detail="只支持 .toml 格式的配置文件"
            )
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"已读取适配器配置: {path}")
        return {"success": True, "content": content}
    except HTTPException:
        raise
    except ValueError as path_err:
        raise HTTPException(
            status_code=403, detail=str(path_err)
        ) from path_err
    except Exception as e:
        logger.error(f"读取适配器配置失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"读取配置失败: {str(e)}"
        ) from e


@router.post("/adapter-config")
async def save_adapter_config(
    data: PathBody, _auth: bool = Depends(require_auth)
):
    """保存适配器配置到指定路径"""
    try:
        path = data.get("path")
        content = data.get("content")
        if not path:
            raise HTTPException(status_code=400, detail="路径不能为空")
        if content is None:
            raise HTTPException(status_code=400, detail="配置内容不能为空")
        abs_path = _normalize_adapter_path(path)
        if not abs_path.endswith(".toml"):
            raise HTTPException(
                status_code=400, detail="只支持 .toml 格式的配置文件"
            )
        try:
            tomlkit.loads(content)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"TOML 格式错误: {
                    str(e)}",
            ) from e
        dir_path = os.path.dirname(abs_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"适配器配置已保存: {path}")
        return {"success": True, "message": "配置已保存"}
    except HTTPException:
        raise
    except ValueError as path_err:
        raise HTTPException(
            status_code=403, detail=str(path_err)
        ) from path_err
    except Exception as e:
        logger.error(f"保存适配器配置失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"保存配置失败: {str(e)}"
        ) from e

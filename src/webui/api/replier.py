"""
回复器监控API
提供回复器日志数据的查询接口

性能优化：
1. 聊天摘要只统计数据库槽位键和最新时间戳，不读取完整内容
2. 日志列表使用槽位键解析时间戳，只在需要时读取当前页内容
3. 详情按需加载
"""

import re
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from src.common.database.slot_storage import list_slot_keys, load_slot
from src.common.logger import get_logger

logger = get_logger("WebUI回复API")
router = APIRouter(prefix="/api/replier", tags=["replier"])

# 回复器日志槽位前缀，写入端见 PlanReplyLogger._REPLY_DIR
REPLY_LOG_PREFIX = "plan_reply_log:reply"

# 安全校验：路径分段只允许字母数字、下划线、连字符、点（禁止 .. 和路径分隔符）
_SAFE_PATH_SEGMENT = re.compile(r"^[a-zA-Z0-9_\-][a-zA-Z0-9_\-.]{0,254}$")


def _validate_path_segment(value: str, label: str) -> str:
    """校验路径分段合法性，防止目录遍历"""
    if not value or ".." in value or "/" in value or "\\" in value:
        raise HTTPException(status_code=400, detail=f"{label} 包含非法字符")
    if not _SAFE_PATH_SEGMENT.match(value):
        raise HTTPException(status_code=400, detail=f"{label} 格式不合法")
    return value


class ReplierChatSummary(BaseModel):
    """聊天摘要 - 轻量级，不读取文件内容"""

    chat_id: str
    reply_count: int
    latest_timestamp: float
    latest_filename: str


class ReplyLogSummary(BaseModel):
    """回复日志摘要"""

    chat_id: str
    timestamp: float
    filename: str
    model: str
    success: bool
    llm_ms: float
    overall_ms: float
    output_preview: str


class ReplyLogDetail(BaseModel):
    """回复日志详情"""

    type: str
    chat_id: str
    timestamp: float
    prompt: str
    output: str
    processed_output: List[str]
    model: str
    reasoning: str
    think_level: int
    timing: Dict
    error: Optional[str] = None
    success: bool


class ReplierOverview(BaseModel):
    """回复器总览 - 轻量级统计"""

    total_chats: int
    total_replies: int
    chats: List[ReplierChatSummary]


class PaginatedReplyLogs(BaseModel):
    """分页的回复日志列表"""

    data: List[ReplyLogSummary]
    total: int
    page: int
    page_size: int
    chat_id: str


def parse_timestamp_from_filename(filename: str) -> float:
    """从槽位尾名解析时间戳: 1766497488220_af92bdb1 -> 1766497488.220"""
    try:
        timestamp_str = filename.split("_")[0]
        # 时间戳是毫秒级，需要转换为秒
        return float(timestamp_str) / 1000
    except (ValueError, IndexError):
        return 0


def _slot_prefix(chat_id: Optional[str] = None) -> str:
    if chat_id is None:
        return f"{REPLY_LOG_PREFIX}:"
    return f"{REPLY_LOG_PREFIX}:{chat_id}:"


def _parse_log_slot_key(slot_key: str) -> Optional[dict]:
    prefix = _slot_prefix()
    if not slot_key.startswith(prefix):
        return None
    remainder = slot_key[len(prefix):]
    if ":" not in remainder:
        return None
    chat_id, filename = remainder.rsplit(":", 1)
    if not chat_id or not filename:
        return None
    return {
        "slot_key": slot_key,
        "chat_id": chat_id,
        "filename": filename,
        "timestamp": parse_timestamp_from_filename(filename),
    }


def _list_log_refs(chat_id: Optional[str] = None) -> list[dict]:
    keys = list_slot_keys(_slot_prefix(chat_id))
    refs = []
    for key in keys:
        parsed = _parse_log_slot_key(key)
        if parsed:
            refs.append(parsed)
    refs.sort(key=lambda item: item["timestamp"], reverse=True)
    return refs


def _normalize_search(search: Optional[str]) -> Optional[str]:
    return search if isinstance(search, str) and search else None


def _safe_str(value: object) -> str:
    return value if isinstance(value, str) else ""


@router.get("/overview", response_model=ReplierOverview)
async def get_replier_overview():
    """
    获取回复器总览 - 轻量级接口
    只统计数据库槽位数量，不读取完整日志内容
    """
    grouped: dict[str, list[dict]] = {}
    for ref in _list_log_refs():
        grouped.setdefault(ref["chat_id"], []).append(ref)

    chats = [
        ReplierChatSummary(
            chat_id=chat_id,
            reply_count=len(refs),
            latest_timestamp=refs[0]["timestamp"],
            latest_filename=refs[0]["filename"],
        )
        for chat_id, refs in grouped.items()
        if refs
    ]

    # 按最新时间戳排序
    chats.sort(key=lambda x: x.latest_timestamp, reverse=True)

    return ReplierOverview(
        total_chats=len(chats),
        total_replies=sum(chat.reply_count for chat in chats),
        chats=chats,
    )


@router.get("/chat/{chat_id}/logs", response_model=PaginatedReplyLogs)
async def get_chat_reply_logs(
    chat_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(
        None, description="搜索条件，匹配提示词内容"
    ),
):
    """
    获取指定聊天的回复日志列表（分页）
    需要读取数据库槽位内容获取摘要信息
    支持搜索提示词内容
    """
    _validate_path_segment(chat_id, "chat_id")
    search = _normalize_search(search)
    refs = _list_log_refs(chat_id)

    # 如果有搜索条件，需要过滤文件
    if search:
        search_lower = search.lower()
        filtered_refs = []
        for ref in refs:
            try:
                data = load_slot(ref["slot_key"], {})
                prompt = data.get("prompt", "") if isinstance(data, dict) else ""
                if search_lower in prompt.lower():
                    filtered_refs.append(ref)
            except Exception:
                continue
        refs = filtered_refs

    total = len(refs)

    # 分页 - 只读取当前页的文件
    offset = (page - 1) * page_size
    page_refs = refs[offset: offset + page_size]

    logs = []
    for ref in page_refs:
        try:
            data = load_slot(ref["slot_key"], {})
            if not isinstance(data, dict):
                raise ValueError("日志槽位内容不是字典")
            output = data.get("output", "")
            timing = data.get("timing", {})
            timing = timing if isinstance(timing, dict) else {}
            logs.append(
                ReplyLogSummary(
                    chat_id=data.get("chat_id", chat_id),
                    timestamp=data.get("timestamp", ref["timestamp"]),
                    filename=ref["filename"],
                    model=data.get("model", ""),
                    success=data.get("success", True),
                    llm_ms=timing.get("llm_ms", 0),
                    overall_ms=timing.get("overall_ms", 0),
                    output_preview=output[:100] if output else "",
                )
            )
        except Exception:
            # 槽位读取失败时使用键名信息
            logs.append(
                ReplyLogSummary(
                    chat_id=chat_id,
                    timestamp=ref["timestamp"],
                    filename=ref["filename"],
                    model="",
                    success=False,
                    llm_ms=0,
                    overall_ms=0,
                    output_preview="[读取失败]",
                )
            )

    return PaginatedReplyLogs(
        data=logs, total=total, page=page, page_size=page_size, chat_id=chat_id
    )


@router.get("/log/{chat_id}/{filename}", response_model=ReplyLogDetail)
async def get_reply_log_detail(chat_id: str, filename: str):
    """获取回复日志详情 - 按需加载数据库槽位内容"""
    _validate_path_segment(chat_id, "chat_id")
    _validate_path_segment(filename, "filename")
    slot_key = f"{REPLY_LOG_PREFIX}:{chat_id}:{filename}"
    data = load_slot(slot_key)
    if not isinstance(data, dict):
        raise HTTPException(status_code=404, detail="日志不存在")
    try:
        processed_output = data.get("processed_output", [])
        timing = data.get("timing", {})
        return ReplyLogDetail(
            type=_safe_str(data.get("type")) or "reply",
            chat_id=_safe_str(data.get("chat_id")) or chat_id,
            timestamp=data.get("timestamp", parse_timestamp_from_filename(filename)),
            prompt=_safe_str(data.get("prompt")),
            output=_safe_str(data.get("output")),
            processed_output=processed_output if isinstance(processed_output, list) else [],
            model=_safe_str(data.get("model")),
            reasoning=_safe_str(data.get("reasoning")),
            think_level=data.get("think_level", 0) or 0,
            timing=timing if isinstance(timing, dict) else {},
            error=data.get("error") if isinstance(data.get("error"), str) else None,
            success=data.get("success", True),
        )
    except Exception as e:
        logger.exception(f"读取回复日志失败: {filename}")
        raise HTTPException(status_code=500, detail="读取日志失败") from e


# ========== 兼容接口 ==========


@router.get("/stats")
async def get_replier_stats():
    """获取回复器统计信息"""
    overview = await get_replier_overview()

    # 获取最近10条回复的摘要
    recent_replies = []
    for chat in overview.chats[:5]:  # 从最近5个聊天中获取
        try:
            chat_logs = await get_chat_reply_logs(
                chat.chat_id, page=1, page_size=2
            )
            recent_replies.extend(chat_logs.data)
        except Exception:
            continue

    # 按时间排序取前10
    recent_replies.sort(key=lambda x: x.timestamp, reverse=True)
    recent_replies = recent_replies[:10]

    return {
        "total_chats": overview.total_chats,
        "total_replies": overview.total_replies,
        "recent_replies": recent_replies,
    }


@router.get("/chats")
async def get_replier_chat_list():
    """获取所有聊天ID列表"""
    overview = await get_replier_overview()
    return [chat.chat_id for chat in overview.chats]

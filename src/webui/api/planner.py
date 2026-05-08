"""
规划器监控API
提供规划器日志数据的查询接口

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

logger = get_logger("WebUI规划API")
router = APIRouter(prefix="/api/planner", tags=["planner"])

# 规划器日志槽位前缀，写入端见 PlanReplyLogger._PLAN_DIR
PLAN_LOG_PREFIX = "plan_reply_log:plan"

# 安全校验：路径分段只允许字母数字、下划线、连字符、点（禁止 .. 和路径分隔符）
_SAFE_PATH_SEGMENT = re.compile(r"^[a-zA-Z0-9_\-][a-zA-Z0-9_\-.]{0,254}$")


def _validate_path_segment(value: str, label: str) -> str:
    """校验路径分段合法性，防止目录遍历"""
    if not value or ".." in value or "/" in value or "\\" in value:
        raise HTTPException(status_code=400, detail=f"{label} 包含非法字符")
    if not _SAFE_PATH_SEGMENT.match(value):
        raise HTTPException(status_code=400, detail=f"{label} 格式不合法")
    return value


class ChatSummary(BaseModel):
    """聊天摘要 - 轻量级，不读取文件内容"""

    chat_id: str
    plan_count: int
    latest_timestamp: float
    latest_filename: str


class PlanLogSummary(BaseModel):
    """规划日志摘要"""

    chat_id: str
    timestamp: float
    filename: str
    action_count: int
    action_types: List[str]  # 动作类型列表
    total_plan_ms: float
    llm_duration_ms: float
    reasoning_preview: str


class PlanLogDetail(BaseModel):
    """规划日志详情"""

    type: str
    chat_id: str
    timestamp: float
    prompt: str
    reasoning: str
    raw_output: str
    actions: List[Dict]
    timing: Dict
    extra: Optional[Dict] = None


class PlannerOverview(BaseModel):
    """规划器总览 - 轻量级统计"""

    total_chats: int
    total_plans: int
    chats: List[ChatSummary]


class PaginatedChatLogs(BaseModel):
    """分页的聊天日志列表"""

    data: List[PlanLogSummary]
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
        return f"{PLAN_LOG_PREFIX}:"
    return f"{PLAN_LOG_PREFIX}:{chat_id}:"


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


@router.get("/overview", response_model=PlannerOverview)
async def get_planner_overview():
    """
    获取规划器总览 - 轻量级接口
    只统计数据库槽位数量，不读取完整日志内容
    """
    grouped: dict[str, list[dict]] = {}
    for ref in _list_log_refs():
        grouped.setdefault(ref["chat_id"], []).append(ref)

    chats = [
        ChatSummary(
            chat_id=chat_id,
            plan_count=len(refs),
            latest_timestamp=refs[0]["timestamp"],
            latest_filename=refs[0]["filename"],
        )
        for chat_id, refs in grouped.items()
        if refs
    ]

    # 按最新时间戳排序
    chats.sort(key=lambda x: x.latest_timestamp, reverse=True)

    return PlannerOverview(
        total_chats=len(chats),
        total_plans=sum(chat.plan_count for chat in chats),
        chats=chats,
    )


@router.get("/chat/{chat_id}/logs", response_model=PaginatedChatLogs)
async def get_chat_plan_logs(
    chat_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(
        None, description="搜索条件，匹配提示词内容"
    ),
):
    """
    获取指定聊天的规划日志列表（分页）
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
            reasoning = data.get("reasoning", "")
            actions = data.get("actions", [])
            actions = actions if isinstance(actions, list) else []
            action_types = [
                a.get("action_type", "")
                for a in actions
                if isinstance(a, dict) and a.get("action_type")
            ]
            timing = data.get("timing", {})
            timing = timing if isinstance(timing, dict) else {}
            logs.append(
                PlanLogSummary(
                    chat_id=data.get("chat_id", chat_id),
                    timestamp=data.get("timestamp", ref["timestamp"]),
                    filename=ref["filename"],
                    action_count=len(actions),
                    action_types=action_types,
                    total_plan_ms=timing.get("total_plan_ms", 0),
                    llm_duration_ms=timing.get("llm_duration_ms", 0),
                    reasoning_preview=reasoning[:100] if reasoning else "",
                )
            )
        except Exception:
            # 槽位读取失败时使用键名信息
            logs.append(
                PlanLogSummary(
                    chat_id=chat_id,
                    timestamp=ref["timestamp"],
                    filename=ref["filename"],
                    action_count=0,
                    action_types=[],
                    total_plan_ms=0,
                    llm_duration_ms=0,
                    reasoning_preview="[读取失败]",
                )
            )

    return PaginatedChatLogs(
        data=logs, total=total, page=page, page_size=page_size, chat_id=chat_id
    )


@router.get("/log/{chat_id}/{filename}", response_model=PlanLogDetail)
async def get_log_detail(chat_id: str, filename: str):
    """获取规划日志详情 - 按需加载数据库槽位内容"""
    _validate_path_segment(chat_id, "chat_id")
    _validate_path_segment(filename, "filename")
    slot_key = f"{PLAN_LOG_PREFIX}:{chat_id}:{filename}"
    data = load_slot(slot_key)
    if not isinstance(data, dict):
        raise HTTPException(status_code=404, detail="日志不存在")
    try:
        actions = data.get("actions", [])
        timing = data.get("timing", {})
        return PlanLogDetail(
            type=_safe_str(data.get("type")) or "plan",
            chat_id=_safe_str(data.get("chat_id")) or chat_id,
            timestamp=data.get("timestamp", parse_timestamp_from_filename(filename)),
            prompt=_safe_str(data.get("prompt")),
            reasoning=_safe_str(data.get("reasoning")),
            raw_output=_safe_str(data.get("raw_output")),
            actions=actions if isinstance(actions, list) else [],
            timing=timing if isinstance(timing, dict) else {},
            extra=data.get("extra") if isinstance(data.get("extra"), dict) else None,
        )
    except Exception as e:
        logger.exception(f"读取规划日志失败: {filename}")
        raise HTTPException(status_code=500, detail="读取日志失败") from e

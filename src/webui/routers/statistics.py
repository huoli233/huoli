"""统计数据 API 路由"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException
from peewee import fn
from pydantic import BaseModel, Field

from src.common.database.database_model import LLMUsage, OnlineTime, Messages
from src.common.logger import get_logger
from src.webui.core.auth import verify_auth_token_from_cookie_or_header
from src.webui.services.runtime_state_hub import get_runtime_state_hub

logger = get_logger("WebUI统计")

router = APIRouter(prefix="/statistics", tags=["statistics"])

STATISTICS_CHANNEL_ID = "__statistics__"


def require_auth(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
) -> bool:
    """认证依赖：验证用户是否已登录"""
    return verify_auth_token_from_cookie_or_header(
        huoli_session, authorization
    )


class StatisticsSummary(BaseModel):
    """统计数据摘要"""

    total_requests: int = Field(0, description="总请求数")
    total_cost: float = Field(0.0, description="总花费")
    total_tokens: int = Field(0, description="总token数")
    online_time: float = Field(0.0, description="在线时间（秒）")
    total_messages: int = Field(0, description="总消息数")
    total_replies: int = Field(0, description="总回复数")
    avg_response_time: float = Field(0.0, description="平均响应时间")
    cost_per_hour: float = Field(0.0, description="每小时花费")
    tokens_per_hour: float = Field(0.0, description="每小时token数")


class ModelStatistics(BaseModel):
    """模型统计"""

    model_name: str
    request_count: int
    total_cost: float
    total_tokens: int
    avg_response_time: float


class TimeSeriesData(BaseModel):
    """时间序列数据"""

    timestamp: str
    requests: int = 0
    cost: float = 0.0
    tokens: int = 0


class DashboardData(BaseModel):
    """仪表盘数据"""

    summary: StatisticsSummary
    model_stats: List[ModelStatistics]
    hourly_data: List[TimeSeriesData]
    daily_data: List[TimeSeriesData]
    recent_activity: List[Dict[str, Any]]


@router.get("/dashboard", response_model=DashboardData)
async def get_dashboard_data(
    hours: int = 24, _auth: bool = Depends(require_auth)
):
    """
    获取仪表盘统计数据

    Args:
        hours: 统计时间范围（小时），默认24小时

    Returns:
        仪表盘数据
    """
    try:
        return await build_dashboard_statistics(hours)
    except Exception as e:
        logger.error(f"获取仪表盘数据失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取统计数据失败: {
                str(e)}",
        ) from e


async def build_dashboard_statistics(hours: int = 24) -> DashboardData:
    now = datetime.now()
    start_time = now - timedelta(hours=hours)
    summary = await _get_summary_statistics(start_time, now)
    model_stats = await _get_model_statistics(start_time)
    hourly_data = await _get_hourly_statistics(start_time, now)
    daily_data = await _get_daily_statistics(now - timedelta(days=7), now)
    recent_activity = await _get_recent_activity(limit=10)
    return DashboardData(
        summary=summary,
        model_stats=model_stats,
        hourly_data=hourly_data,
        daily_data=daily_data,
        recent_activity=recent_activity,
    )


async def publish_statistics_snapshot(reason: str = "llm_usage") -> None:
    try:
        snapshot = (await build_dashboard_statistics()).model_dump()
        await get_runtime_state_hub().set_snapshot(
            STATISTICS_CHANNEL_ID,
            snapshot,
            reason=f"statistics_{reason}",
        )
    except Exception as exc:
        logger.debug(f"统计实时快照发布失败: {exc}")


@router.get("/dashboard/live")
async def wait_dashboard_statistics(
    hours: int = 24,
    after_version: int = 0,
    timeout: float = 25.0,
    _auth: bool = Depends(require_auth),
):
    hub = get_runtime_state_hub()
    cached = await hub.get_snapshot(STATISTICS_CHANNEL_ID)
    if cached is None:
        await publish_statistics_snapshot("initial")
        cached = await hub.get_snapshot(STATISTICS_CHANNEL_ID)
    if cached is not None and int(cached.get("state_version", 0) or 0) > int(after_version or 0):
        return {"success": True, "changed": True, "statistics": cached}
    event = await hub.wait_for_update(
        STATISTICS_CHANNEL_ID,
        after_version=after_version,
        timeout_seconds=timeout,
    )
    if event is None:
        latest = await hub.get_snapshot(STATISTICS_CHANNEL_ID)
        return {"success": True, "changed": False, "statistics": latest or cached or {}}
    return {
        "success": True,
        "changed": True,
        "statistics": event.get("data", {}),
        "event": {
            "type": event.get("type", "state_snapshot"),
            "version": event.get("version", 0),
            "reason": event.get("reason", ""),
        },
    }


async def _get_summary_statistics(
    start_time: datetime, end_time: datetime
) -> StatisticsSummary:
    """获取摘要统计数据（优化：使用数据库聚合）"""
    summary = StatisticsSummary()

    # 使用聚合查询替代全量加载
    query = LLMUsage.select(
        fn.COUNT(LLMUsage.id).alias("total_requests"),
        fn.COALESCE(fn.SUM(LLMUsage.cost), 0).alias("total_cost"),
        fn.COALESCE(
            fn.SUM(LLMUsage.prompt_tokens + LLMUsage.completion_tokens), 0
        ).alias("total_tokens"),
        fn.COALESCE(fn.AVG(LLMUsage.time_cost), 0).alias("avg_response_time"),
    ).where(
        (LLMUsage.timestamp >= start_time) & (LLMUsage.timestamp <= end_time)
    )

    result = query.dicts().get()
    summary.total_requests = result["total_requests"]
    summary.total_cost = result["total_cost"]
    summary.total_tokens = result["total_tokens"]
    summary.avg_response_time = result["avg_response_time"] or 0.0

    # 查询在线时间 - 这个数据量通常不大，保留原逻辑
    online_records = list(
        OnlineTime.select().where(
            (OnlineTime.start_timestamp >= start_time)
            | (OnlineTime.end_timestamp >= start_time)
        )
    )

    for record in online_records:
        start = max(record.start_timestamp, start_time)
        end = min(record.end_timestamp, end_time)
        if end > start:
            summary.online_time += (end - start).total_seconds()

    # 查询消息数量 - 使用聚合优化
    messages_query = Messages.select(
        fn.COUNT(Messages.id).alias("total")
    ).where(
        (Messages.time >= start_time.timestamp())
        & (Messages.time <= end_time.timestamp())
    )
    summary.total_messages = messages_query.scalar() or 0

    # 统计回复数量
    replies_query = Messages.select(
        fn.COUNT(Messages.id).alias("total")
    ).where(
        (Messages.time >= start_time.timestamp())
        & (Messages.time <= end_time.timestamp())
        & (Messages.reply_to.is_null(False))
    )
    summary.total_replies = replies_query.scalar() or 0

    # 计算派生指标
    if summary.online_time > 0:
        online_hours = summary.online_time / 3600.0
        summary.cost_per_hour = summary.total_cost / online_hours
        summary.tokens_per_hour = summary.total_tokens / online_hours

    return summary


async def _get_model_statistics(start_time: datetime) -> List[ModelStatistics]:
    """获取模型统计数据（优化：使用数据库聚合和分组）"""
    # 使用GROUP BY聚合，避免全量加载
    query = (
        LLMUsage.select(
            fn.COALESCE(
                LLMUsage.model_assign_name, LLMUsage.model_name, "unknown"
            ).alias("model_name"),
            fn.COUNT(LLMUsage.id).alias("request_count"),
            fn.COALESCE(fn.SUM(LLMUsage.cost), 0).alias("total_cost"),
            fn.COALESCE(
                fn.SUM(LLMUsage.prompt_tokens + LLMUsage.completion_tokens), 0
            ).alias("total_tokens"),
            fn.COALESCE(fn.AVG(LLMUsage.time_cost), 0).alias(
                "avg_response_time"
            ),
        )
        .where(LLMUsage.timestamp >= start_time)
        .group_by(
            fn.COALESCE(
                LLMUsage.model_assign_name, LLMUsage.model_name, "unknown"
            )
        )
        .order_by(fn.COUNT(LLMUsage.id).desc())
        .limit(10)  # 只取前10个
    )

    result = []
    for row in query.dicts():
        result.append(
            ModelStatistics(
                model_name=row["model_name"],
                request_count=row["request_count"],
                total_cost=row["total_cost"],
                total_tokens=row["total_tokens"],
                avg_response_time=row["avg_response_time"] or 0.0,
            )
        )

    return result


async def _get_hourly_statistics(
    start_time: datetime, end_time: datetime
) -> List[TimeSeriesData]:
    """获取小时级统计数据（优化：使用数据库聚合）"""
    # SQLite的日期时间函数进行小时分组
    # 使用strftime将timestamp格式化为小时级别
    query = (
        LLMUsage.select(
            fn.strftime("%Y-%m-%dT%H:00:00", LLMUsage.timestamp).alias("hour"),
            fn.COUNT(LLMUsage.id).alias("requests"),
            fn.COALESCE(fn.SUM(LLMUsage.cost), 0).alias("cost"),
            fn.COALESCE(
                fn.SUM(LLMUsage.prompt_tokens + LLMUsage.completion_tokens), 0
            ).alias("tokens"),
        )
        .where(
            (LLMUsage.timestamp >= start_time)
            & (LLMUsage.timestamp <= end_time)
        )
        .group_by(fn.strftime("%Y-%m-%dT%H:00:00", LLMUsage.timestamp))
    )

    # 转换为字典以快速查找
    data_dict = {row["hour"]: row for row in query.dicts()}

    # 填充所有小时（包括没有数据的）
    result = []
    current = start_time.replace(minute=0, second=0, microsecond=0)
    while current <= end_time:
        hour_str = current.strftime("%Y-%m-%dT%H:00:00")
        if hour_str in data_dict:
            row = data_dict[hour_str]
            result.append(
                TimeSeriesData(
                    timestamp=hour_str,
                    requests=row["requests"],
                    cost=row["cost"],
                    tokens=row["tokens"],
                )
            )
        else:
            result.append(
                TimeSeriesData(
                    timestamp=hour_str, requests=0, cost=0.0, tokens=0
                )
            )
        current += timedelta(hours=1)

    return result


async def _get_daily_statistics(
    start_time: datetime, end_time: datetime
) -> List[TimeSeriesData]:
    """获取日级统计数据（优化：使用数据库聚合）"""
    # 使用strftime按日期分组
    query = (
        LLMUsage.select(
            fn.strftime("%Y-%m-%dT00:00:00", LLMUsage.timestamp).alias("day"),
            fn.COUNT(LLMUsage.id).alias("requests"),
            fn.COALESCE(fn.SUM(LLMUsage.cost), 0).alias("cost"),
            fn.COALESCE(
                fn.SUM(LLMUsage.prompt_tokens + LLMUsage.completion_tokens), 0
            ).alias("tokens"),
        )
        .where(
            (LLMUsage.timestamp >= start_time)
            & (LLMUsage.timestamp <= end_time)
        )
        .group_by(fn.strftime("%Y-%m-%dT00:00:00", LLMUsage.timestamp))
    )

    # 转换为字典
    data_dict = {row["day"]: row for row in query.dicts()}

    # 填充所有天
    result = []
    current = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    while current <= end_time:
        day_str = current.strftime("%Y-%m-%dT00:00:00")
        if day_str in data_dict:
            row = data_dict[day_str]
            result.append(
                TimeSeriesData(
                    timestamp=day_str,
                    requests=row["requests"],
                    cost=row["cost"],
                    tokens=row["tokens"],
                )
            )
        else:
            result.append(
                TimeSeriesData(
                    timestamp=day_str, requests=0, cost=0.0, tokens=0
                )
            )
        current += timedelta(days=1)

    return result


async def _get_recent_activity(limit: int = 10) -> List[Dict[str, Any]]:
    """获取最近活动"""
    records = list(
        LLMUsage.select().order_by(LLMUsage.timestamp.desc()).limit(limit)
    )

    activities = []
    for record in records:
        activities.append(
            {
                "timestamp": record.timestamp.isoformat(),
                "model": record.model_assign_name or record.model_name,
                "request_type": record.request_type,
                "tokens": (record.prompt_tokens or 0)
                + (record.completion_tokens or 0),
                "cost": record.cost or 0.0,
                "time_cost": record.time_cost or 0.0,
                "status": record.status,
            }
        )

    return activities


@router.get("/summary")
async def get_summary(hours: int = 24, _auth: bool = Depends(require_auth)):
    """
    获取统计摘要

    Args:
        hours: 统计时间范围（小时）
    """
    try:
        now = datetime.now()
        start_time = now - timedelta(hours=hours)
        summary = await _get_summary_statistics(start_time, now)
        return summary
    except Exception as e:
        logger.error(f"获取统计摘要失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models")
async def get_model_stats(
    hours: int = 24, _auth: bool = Depends(require_auth)
):
    """
    获取模型统计

    Args:
        hours: 统计时间范围（小时）
    """
    try:
        now = datetime.now()
        start_time = now - timedelta(hours=hours)
        stats = await _get_model_statistics(start_time)
        return stats
    except Exception as e:
        logger.error(f"获取模型统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

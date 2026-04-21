import json
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, Cookie, Header, HTTPException, Query
from pydantic import BaseModel

from src.common.logger import get_logger
from src.common.database.database_model import PersonInfo
from src.webui.core import verify_auth_token_from_cookie_or_header

logger = get_logger("WebUI人物")

router = APIRouter(prefix="/person", tags=["Person"])


class PersonInfoResponse(BaseModel):
    """人物信息响应"""

    id: int
    is_known: bool
    person_id: str
    person_name: Optional[str] = None
    name_reason: Optional[str] = None
    platform: str
    user_id: str
    nickname: Optional[str] = None
    group_nick_name: Optional[List[Dict[str, str]]] = None
    memory_points: Optional[str] = None
    know_times: Optional[float] = None
    know_since: Optional[float] = None
    last_know: Optional[float] = None


class PersonListResponse(BaseModel):
    """人物列表响应"""

    success: bool
    total: int
    page: int
    page_size: int
    data: List[PersonInfoResponse]


class PersonDetailResponse(BaseModel):
    """人物详情响应"""

    success: bool
    data: PersonInfoResponse


class PersonUpdateRequest(BaseModel):
    """人物信息更新请求"""

    person_name: Optional[str] = None
    name_reason: Optional[str] = None
    nickname: Optional[str] = None
    memory_points: Optional[str] = None
    is_known: Optional[bool] = None


class PersonUpdateResponse(BaseModel):
    """人物信息更新响应"""

    success: bool
    message: str
    data: Optional[PersonInfoResponse] = None


class PersonDeleteResponse(BaseModel):
    """人物删除响应"""

    success: bool
    message: str


class BatchDeleteRequest(BaseModel):
    """批量删除请求"""

    person_ids: List[str]


class BatchDeleteResponse(BaseModel):
    """批量删除响应"""

    success: bool
    message: str
    deleted_count: int
    failed_count: int
    failed_ids: List[str] = []


def verify_auth_token(
    huoli_session: Optional[str] = None,
    authorization: Optional[str] = None,
) -> bool:
    """验证认证 Token"""
    return verify_auth_token_from_cookie_or_header(
        huoli_session, authorization
    )


def parse_group_nick_name(
    group_nick_name_str: Optional[str],
) -> Optional[List[Dict[str, str]]]:
    """解析群昵称 JSON 字符串"""
    if not group_nick_name_str:
        return None
    try:
        return json.loads(group_nick_name_str)
    except (json.JSONDecodeError, TypeError):
        return None


def person_to_response(person: PersonInfo) -> PersonInfoResponse:
    """将 PersonInfo 模型转换为响应对象"""
    return PersonInfoResponse(
        id=person.id,
        is_known=person.is_known,
        person_id=person.person_id,
        person_name=person.person_name,
        name_reason=person.name_reason,
        platform=person.platform,
        user_id=person.user_id,
        nickname=person.nickname,
        group_nick_name=parse_group_nick_name(person.group_nick_name),
        memory_points=person.memory_points,
        know_times=person.know_times,
        know_since=person.know_since,
        last_know=person.last_know,
    )


@router.get("/list", response_model=PersonListResponse)
async def get_person_list(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    search: Optional[str] = Query(None, description="搜索条件"),
    is_known: Optional[bool] = Query(None, description="是否已认识筛选"),
    platform: Optional[str] = Query(None, description="平台筛选"),
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """获取人物信息列表"""
    try:
        verify_auth_token(huoli_session, authorization)
        query = PersonInfo.select()
        if search:
            query = query.where(
                (PersonInfo.person_name.contains(search))
                | (PersonInfo.nickname.contains(search))
                | (PersonInfo.user_id.contains(search))
            )
        if is_known is not None:
            query = query.where(PersonInfo.is_known == is_known)
        if platform:
            query = query.where(PersonInfo.platform == platform)
        query = query.order_by(PersonInfo.last_know.desc(nulls="LAST"))
        total = query.count()
        offset = (page - 1) * page_size
        persons = list(query.offset(offset).limit(page_size))
        data = [person_to_response(person) for person in persons]
        return PersonListResponse(
            success=True,
            total=total,
            page=page,
            page_size=page_size,
            data=data,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取人物列表失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取人物列表失败: {
                str(e)}",
        ) from e


@router.get("/{person_id}", response_model=PersonDetailResponse)
async def get_person_detail(
    person_id: str,
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """获取人物详细信息"""
    try:
        verify_auth_token(huoli_session, authorization)
        person = PersonInfo.get_or_none(PersonInfo.person_id == person_id)
        if not person:
            raise HTTPException(
                status_code=404, detail=f"未找到 ID 为 {person_id} 的人物信息"
            )
        return PersonDetailResponse(
            success=True, data=person_to_response(person)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取人物详情失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取人物详情失败: {
                str(e)}",
        ) from e


@router.patch("/{person_id}", response_model=PersonUpdateResponse)
async def update_person(
    person_id: str,
    request: PersonUpdateRequest,
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """增量更新人物信息"""
    try:
        verify_auth_token(huoli_session, authorization)
        person = PersonInfo.get_or_none(PersonInfo.person_id == person_id)
        if not person:
            raise HTTPException(
                status_code=404, detail=f"未找到 ID 为 {person_id} 的人物信息"
            )
        update_data = request.model_dump(exclude_unset=True)
        if not update_data:
            raise HTTPException(
                status_code=400, detail="未提供任何需要更新的字段"
            )
        update_data["last_know"] = time.time()
        for field, value in update_data.items():
            setattr(person, field, value)
        person.save()
        logger.info(
            f"人物信息已更新: {person_id}, 字段: {list(update_data.keys())}"
        )
        return PersonUpdateResponse(
            success=True,
            message=f"成功更新 {len(update_data)} 个字段",
            data=person_to_response(person),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新人物信息失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"更新人物信息失败: {
                str(e)}",
        ) from e


@router.delete("/{person_id}", response_model=PersonDeleteResponse)
async def delete_person(
    person_id: str,
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """删除人物信息"""
    try:
        verify_auth_token(huoli_session, authorization)
        person = PersonInfo.get_or_none(PersonInfo.person_id == person_id)
        if not person:
            raise HTTPException(
                status_code=404, detail=f"未找到 ID 为 {person_id} 的人物信息"
            )
        person_name = person.person_name or person.nickname or person.user_id
        person.delete_instance()
        logger.info(f"人物信息已删除: {person_id} ({person_name})")
        return PersonDeleteResponse(
            success=True, message=f"成功删除人物信息: {person_name}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除人物信息失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"删除人物信息失败: {
                str(e)}",
        ) from e


@router.get("/stats/summary")
async def get_person_stats(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """获取人物信息统计数据"""
    try:
        verify_auth_token(huoli_session, authorization)
        total = PersonInfo.select().count()
        known = PersonInfo.select().where(PersonInfo.is_known).count()
        unknown = total - known
        platforms: Dict[str, int] = {}
        for person in PersonInfo.select(PersonInfo.platform):
            platform = person.platform
            platforms[platform] = platforms.get(platform, 0) + 1
        return {
            "success": True,
            "data": {
                "total": total,
                "known": known,
                "unknown": unknown,
                "platforms": platforms,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取统计数据失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取统计数据失败: {
                str(e)}",
        ) from e


@router.post("/batch/delete", response_model=BatchDeleteResponse)
async def batch_delete_persons(
    request: BatchDeleteRequest,
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    """批量删除人物信息"""
    try:
        verify_auth_token(huoli_session, authorization)
        if not request.person_ids:
            raise HTTPException(status_code=400, detail="未提供要删除的人物ID")
        deleted_count = 0
        failed_count = 0
        failed_ids = []
        for person_id in request.person_ids:
            try:
                person = PersonInfo.get_or_none(
                    PersonInfo.person_id == person_id
                )
                if person:
                    person.delete_instance()
                    deleted_count += 1
                    logger.info(f"批量删除: {person_id}")
                else:
                    failed_count += 1
                    failed_ids.append(person_id)
            except Exception as e:
                logger.error(f"删除 {person_id} 失败: {e}")
                failed_count += 1
                failed_ids.append(person_id)
        message = f"成功删除 {deleted_count} 个人物"
        if failed_count > 0:
            message += f"，{failed_count} 个失败"
        return BatchDeleteResponse(
            success=True,
            message=message,
            deleted_count=deleted_count,
            failed_count=failed_count,
            failed_ids=failed_ids,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量删除人物信息失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"批量删除失败: {str(e)}"
        ) from e

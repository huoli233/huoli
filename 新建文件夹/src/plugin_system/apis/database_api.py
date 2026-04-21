import traceback
import time
import json
from typing import Dict, List, Any, Union, Type, Optional
from src.common.logger import get_logger
from peewee import Model, DoesNotExist

logger = get_logger("数据库API")


async def db_query(
    model_class: Type[Model],
    data: Optional[Dict[str, Any]] = None,
    query_type: Optional[str] = "get",
    filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    order_by: Optional[List[str]] = None,
    single_result: Optional[bool] = False,
) -> Union[List[Dict[str, Any]], Dict[str, Any], None]:
    try:
        if query_type not in ["get", "create", "update", "delete", "count"]:
            raise ValueError(
                "query_type must be 'get' or 'create' or 'update' or 'delete' or 'count'"
            )
        if query_type in ["get", "update", "delete", "count"]:
            query = model_class.select()
            if filters:
                for field, value in filters.items():
                    if hasattr(model_class, field):
                        query = query.where(
                            getattr(model_class, field) == value
                        )
        if query_type == "get":
            if order_by:
                for field in order_by:
                    if field.startswith("-"):
                        query = query.order_by(
                            getattr(model_class, field[1:]).desc()
                        )
                    else:
                        query = query.order_by(getattr(model_class, field))
            if limit:
                query = query.limit(limit)
            results = list(query.dicts())
            if single_result:
                return results[0] if results else None
            return results
        elif query_type == "create":
            if not data:
                raise ValueError("创建记录需要提供data参数")
            record = model_class.create(**data)
            return (
                model_class.select()
                .where(model_class.id == record.id)
                .dicts()
                .get()
            )
        elif query_type == "update":
            if not data:
                raise ValueError("更新记录需要提供data参数")
            return query.update(**data).execute()
        elif query_type == "delete":
            return query.delete().execute()
        elif query_type == "count":
            return query.count()
        else:
            raise ValueError(f"不支持的查询类型: {query_type}")
    except DoesNotExist:
        return None if query_type == "get" and single_result else []
    except Exception as e:
        logger.error(f"[DatabaseAPI] 数据库操作出错: {e}")
        traceback.print_exc()
        if query_type == "get":
            return None if single_result else []
        elif query_type in ["create", "update", "delete", "count"]:
            return None
        return None


async def db_save(
    model_class: Type[Model],
    data: Dict[str, Any],
    key_field: Optional[str] = None,
    key_value: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    try:
        if key_field and key_value is not None:
            if existing_records := list(
                model_class.select()
                .where(getattr(model_class, key_field) == key_value)
                .limit(1)
            ):
                existing_record = existing_records[0]
                for field, value in data.items():
                    setattr(existing_record, field, value)
                existing_record.save()
                updated_record = (
                    model_class.select()
                    .where(model_class.id == existing_record.id)
                    .dicts()
                    .get()
                )
                return updated_record
        new_record = model_class.create(**data)
        created_record = (
            model_class.select()
            .where(model_class.id == new_record.id)
            .dicts()
            .get()
        )
        return created_record
    except Exception as e:
        logger.error(f"[DatabaseAPI] 保存数据库记录出错: {e}")
        traceback.print_exc()
        return None


async def db_get(
    model_class: Type[Model],
    filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    order_by: Optional[str] = None,
    single_result: Optional[bool] = False,
) -> Union[List[Dict[str, Any]], Dict[str, Any], None]:
    try:
        query = model_class.select()
        if filters:
            for field, value in filters.items():
                query = query.where(getattr(model_class, field) == value)
        if order_by:
            if order_by.startswith("-"):
                query = query.order_by(
                    getattr(model_class, order_by[1:]).desc()
                )
            else:
                query = query.order_by(getattr(model_class, order_by))
        if limit:
            query = query.limit(limit)
        results = list(query.dicts())
        if single_result:
            return results[0] if results else None
        return results
    except Exception as e:
        logger.error(f"[DatabaseAPI] 获取数据库记录出错: {e}")
        traceback.print_exc()
        return None if single_result else []


async def store_action_info(
    chat_stream=None,
    action_build_into_prompt: bool = False,
    action_prompt_display: str = "",
    action_done: bool = True,
    thinking_id: str = "",
    action_data: Optional[dict] = None,
    action_name: str = "",
    action_reasoning: str = "",
) -> Optional[Dict[str, Any]]:
    try:
        from src.common.database.database_model import ActionRecords

        record_data = {
            "action_id": thinking_id or str(int(time.time() * 1000000)),
            "time": time.time(),
            "action_name": action_name,
            "action_data": json.dumps(action_data or {}, ensure_ascii=False),
            "action_done": action_done,
            "action_reasoning": action_reasoning,
            "action_build_into_prompt": action_build_into_prompt,
            "action_prompt_display": action_prompt_display,
        }
        if chat_stream:
            record_data.update(
                {
                    "chat_id": getattr(chat_stream, "stream_id", ""),
                    "chat_info_stream_id": getattr(
                        chat_stream, "stream_id", ""
                    ),
                    "chat_info_platform": getattr(chat_stream, "platform", ""),
                }
            )
        else:
            record_data.update(
                {
                    "chat_id": "",
                    "chat_info_stream_id": "",
                    "chat_info_platform": "",
                }
            )
        saved_record = await db_save(
            ActionRecords,
            data=record_data,
            key_field="action_id",
            key_value=record_data["action_id"],
        )
        if saved_record:
            logger.debug(
                f"[DatabaseAPI] 成功存储动作信息: {action_name} (ID: {
                    record_data['action_id']})"
            )
        else:
            logger.error(f"[DatabaseAPI] 存储动作信息失败: {action_name}")
        return saved_record
    except Exception as e:
        logger.error(f"[DatabaseAPI] 存储动作信息时发生错误: {e}")
        traceback.print_exc()
        return None


async def db_count(
    model_class: Type[Model], filters: Optional[Dict[str, Any]] = None
) -> int:
    try:
        query = model_class.select()
        if filters:
            for field, value in filters.items():
                query = query.where(getattr(model_class, field) == value)
        return query.count()
    except Exception as e:
        logger.error(f"[DatabaseAPI] 计数失败: {e}")
        return 0


async def db_exists(model_class: Type[Model], filters: Dict[str, Any]) -> bool:
    try:
        query = model_class.select()
        for field, value in filters.items():
            query = query.where(getattr(model_class, field) == value)
        return query.exists()
    except Exception as e:
        logger.error(f"[DatabaseAPI] 检查存在失败: {e}")
        return False

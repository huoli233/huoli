from typing import Any
from src.common.logger import get_logger
from src.person_info.person_info import Person

logger = get_logger("人物API")


def get_person_id(platform: str, user_id: int | str) -> str:
    try:
        return Person(platform=platform, user_id=str(user_id)).person_id
    except Exception as e:
        logger.error(
            f"[PersonAPI] 获取person_id失败: platform={platform}, user_id={user_id}, error={e}"
        )
        return ""


async def get_person_value(
    person_id: str, field_name: str, default: Any = None
) -> Any:
    try:
        person = Person(person_id=person_id)
        value = getattr(person, field_name, None)
        return value if value is not None else default
    except Exception as e:
        logger.error(
            f"[PersonAPI] 获取用户信息失败: person_id={person_id}, field={field_name}, error={e}"
        )
        return default


def get_person_id_by_name(person_name: str) -> str:
    try:
        person = Person(person_name=person_name)
        return person.person_id
    except Exception as e:
        logger.error(
            f"[PersonAPI] 根据用户名获取person_id失败: person_name={person_name}, error={e}"
        )
        return ""


async def set_person_value(
    person_id: str, field_name: str, value: Any
) -> bool:
    try:
        person = Person(person_id=person_id)
        if hasattr(person, field_name):
            setattr(person, field_name, value)
            person.save()
            logger.debug(
                f"[PersonAPI] 设置用户信息成功: person_id={person_id}, field={field_name}"
            )
            return True
        else:
            logger.warning(f"[PersonAPI] 用户信息字段不存在: {field_name}")
            return False
    except Exception as e:
        logger.error(
            f"[PersonAPI] 设置用户信息失败: person_id={person_id}, field={field_name}, error={e}"
        )
        return False


async def get_person_memory_points(person_id: str) -> list:
    try:
        person = Person(person_id=person_id)
        return getattr(person, "memory_points", [])
    except Exception as e:
        logger.error(
            f"[PersonAPI] 获取用户记忆点失败: person_id={person_id}, error={e}"
        )
        return []


async def add_person_memory_point(person_id: str, memory_point: str) -> bool:
    try:
        person = Person(person_id=person_id)
        if hasattr(person, "memory_points"):
            memory_points = person.memory_points or []
            memory_points.append(memory_point)
            person.memory_points = memory_points
            person.save()
            logger.debug(
                f"[PersonAPI] 添加用户记忆点成功: person_id={person_id}"
            )
            return True
        return False
    except Exception as e:
        logger.error(
            f"[PersonAPI] 添加用户记忆点失败: person_id={person_id}, error={e}"
        )
        return False


def get_person_nickname(person_id: str) -> str:
    try:
        person = Person(person_id=person_id)
        return getattr(person, "nickname", "未知用户")
    except Exception as e:
        logger.error(
            f"[PersonAPI] 获取用户昵称失败: person_id={person_id}, error={e}"
        )
        return "未知用户"


def get_person_impression(person_id: str) -> str:
    try:
        person = Person(person_id=person_id)
        return getattr(person, "impression", "")
    except Exception as e:
        logger.error(
            f"[PersonAPI] 获取用户印象失败: person_id={person_id}, error={e}"
        )
        return ""

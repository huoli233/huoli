import re
import json
import time
import hashlib
import traceback
from typing import Union, Optional, List, Dict, Any

from src.common.database.database_model import Messages, Images, KGEntity, KGRelation
from src.common.logger import get_logger
from src.chat.message_receive.chat_stream import ChatStream
from src.chat.message_receive.message import MessageSending, MessageRecv

logger = get_logger("message_storage")


class MessageStorage:
    @staticmethod
    def _serialize_keywords(keywords) -> str:
        """将关键词列表序列化为JSON字符串"""
        if isinstance(keywords, list):
            return json.dumps(keywords, ensure_ascii=False)
        return "[]"

    @staticmethod
    def _deserialize_keywords(keywords_str: str) -> list:
        """将JSON字符串反序列化为关键词列表"""
        if not keywords_str:
            return []
        try:
            return json.loads(keywords_str)
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    async def store_message(message: Union[MessageSending, MessageRecv], chat_stream: ChatStream) -> None:
        """存储消息到数据库"""
        try:
            # 通知消息不存储
            if isinstance(message, MessageRecv) and message.is_notify:
                logger.debug("通知消息，跳过存储")
                return

            pattern = r"<MainRule>.*?</MainRule>|<schedule>.*?</schedule>|<UserMessage>.*?</UserMessage>"

            # print(message)

            processed_plain_text = message.processed_plain_text

            # print(processed_plain_text)

            if processed_plain_text:
                processed_plain_text = MessageStorage.replace_image_descriptions(processed_plain_text)
                filtered_processed_plain_text = re.sub(pattern, "", processed_plain_text, flags=re.DOTALL)
            else:
                filtered_processed_plain_text = ""

            if isinstance(message, MessageSending):
                display_message = message.display_message
                if display_message:
                    filtered_display_message = re.sub(pattern, "", display_message, flags=re.DOTALL)
                else:
                    filtered_display_message = ""
                interest_value = 0
                is_mentioned = False
                is_at = False
                reply_probability_boost = 0.0
                reply_to = message.reply_to
                priority_mode = ""
                priority_info = {}
                is_emoji = False
                is_picid = False
                is_notify = False
                is_command = False
                key_words = ""
                key_words_lite = ""
                selected_expressions = message.selected_expressions
                intercept_message_level = 0
            else:
                filtered_display_message = ""
                interest_value = message.interest_value
                is_mentioned = message.is_mentioned
                is_at = message.is_at
                reply_probability_boost = message.reply_probability_boost
                reply_to = ""
                priority_mode = message.priority_mode
                priority_info = message.priority_info
                is_emoji = message.is_emoji
                is_picid = message.is_picid
                is_notify = message.is_notify
                is_command = message.is_command
                intercept_message_level = getattr(message, "intercept_message_level", 0)
                # 序列化关键词列表为JSON字符串
                key_words = MessageStorage._serialize_keywords(message.key_words)
                key_words_lite = MessageStorage._serialize_keywords(message.key_words_lite)
                selected_expressions = ""

            chat_info_dict = chat_stream.to_dict()
            user_info_dict = message.message_info.user_info.to_dict()  # type: ignore

            # message_id 现在是 TextField，直接使用字符串值
            msg_id = message.message_info.message_id

            # 安全地获取 group_info, 如果为 None 则视为空字典
            group_info_from_chat = chat_info_dict.get("group_info") or {}
            # 安全地获取 user_info, 如果为 None 则视为空字典 (以防万一)
            user_info_from_chat = chat_info_dict.get("user_info") or {}

            Messages.create(
                message_id=msg_id,
                time=float(message.message_info.time),  # type: ignore
                chat_id=chat_stream.stream_id,
                # Flattened chat_info
                reply_to=reply_to,
                is_mentioned=is_mentioned,
                is_at=is_at,
                reply_probability_boost=reply_probability_boost,
                chat_info_stream_id=chat_info_dict.get("stream_id"),
                chat_info_platform=chat_info_dict.get("platform"),
                chat_info_user_platform=user_info_from_chat.get("platform"),
                chat_info_user_id=user_info_from_chat.get("user_id"),
                chat_info_user_nickname=user_info_from_chat.get("user_nickname"),
                chat_info_user_cardname=user_info_from_chat.get("user_cardname"),
                chat_info_group_platform=group_info_from_chat.get("platform"),
                chat_info_group_id=group_info_from_chat.get("group_id"),
                chat_info_group_name=group_info_from_chat.get("group_name"),
                chat_info_create_time=float(chat_info_dict.get("create_time", 0.0)),
                chat_info_last_active_time=float(chat_info_dict.get("last_active_time", 0.0)),
                # Flattened user_info (message sender)
                user_platform=user_info_dict.get("platform"),
                user_id=user_info_dict.get("user_id"),
                user_nickname=user_info_dict.get("user_nickname"),
                user_cardname=user_info_dict.get("user_cardname"),
                # Text content
                processed_plain_text=filtered_processed_plain_text,
                display_message=filtered_display_message,
                interest_value=interest_value,
                priority_mode=priority_mode,
                priority_info=priority_info,
                is_emoji=is_emoji,
                is_picid=is_picid,
                is_notify=is_notify,
                is_command=is_command,
                intercept_message_level=intercept_message_level,
                key_words=key_words,
                key_words_lite=key_words_lite,
                selected_expressions=selected_expressions,
            )
        except Exception:
            logger.exception("存储消息失败")
            logger.error(f"消息：{message}")
            traceback.print_exc()

    # 如果需要其他存储相关的函数，可以在这里添加
    @staticmethod
    def update_message(mmc_message_id: str | None, qq_message_id: str | None) -> bool:
        """实时更新数据库的自身发送消息ID"""
        try:
            if not qq_message_id:
                logger.info("消息不存在message_id，无法更新")
                return False
            if matched_message := (
                Messages.select().where((Messages.message_id == mmc_message_id)).order_by(Messages.time.desc()).first()
            ):
                # 更新找到的消息记录
                Messages.update(message_id=qq_message_id).where(Messages.id == matched_message.id).execute()  # type: ignore
                logger.debug(f"更新消息ID成功: {matched_message.message_id} -> {qq_message_id}")
                return True
            else:
                logger.debug("未找到匹配的消息")
                return False

        except Exception as e:
            logger.error(f"更新消息ID失败: {e}")
            return False

    _recent_content_hashes: Dict[str, float] = {}
    _hashes_cleanup_time: float = 0

    @staticmethod
    def is_meaningful_content(text: str) -> bool:
        if not text or len(text.strip()) < 1:
            return False
        junk_patterns = [
            r'^\[(TTS|PASS|工具|表情包|emoji|THOUGHT|INTERNAL)\]$',
            r'^<(MainRule|schedule|plan|thought)>.*?</\1>$',
            r'^\[(回复|media):[^\]]+\]$',
            r'^[\s\.,!\?\?。，！？]+$'
        ]
        text_strip = text.strip()
        for pattern in junk_patterns:
            if re.match(pattern, text_strip, flags=re.IGNORECASE | re.DOTALL):
                return False
        if "[THOUGHT]" in text_strip.upper() or "<thought>" in text_strip.lower():
            return False
        return True

    @classmethod
    def is_duplicate(cls, chat_id: str, content: str) -> bool:
        now = time.time()
        if now - cls._hashes_cleanup_time > 600:
            cls._recent_content_hashes = {h: t for h, t in cls._recent_content_hashes.items() if now - t < 3600}
            cls._hashes_cleanup_time = now
        content_hash = hashlib.md5(f"{chat_id}:{content.strip()}".encode()).hexdigest()
        if content_hash in cls._recent_content_hashes:
            last_time = cls._recent_content_hashes[content_hash]
            if now - last_time < 300:
                return True
        cls._recent_content_hashes[content_hash] = now
        return False

    @staticmethod
    def replace_image_descriptions(text: str) -> str:
        pattern = r"\[图片：([^\]]+)\]"
        matches = re.findall(pattern, text)
        if not matches:
            return text

        def replace_match(match):
            description = match.group(1).strip()
            try:
                image_record = (
                    Images.select().where(Images.description == description).order_by(Images.timestamp.desc()).first()
                )
                return f"[picid:{image_record.image_id}]" if image_record else match.group(0)
            except Exception:
                return match.group(0)

        return re.sub(r"\[图片：([^\]]+)\]", replace_match, text)

    @staticmethod
    async def query_recent(chat_id: str, limit: int = 50,
                            user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            query = Messages.select().where(Messages.chat_id == chat_id)
            if user_id:
                query = Messages.select().where(
                    (Messages.chat_id == chat_id) | (Messages.user_id == user_id)
                )
            rows = query.order_by(Messages.time.desc()).limit(limit)
            results = []
            for row in rows:
                results.append({
                    "message_id": row.message_id,
                    "time": row.time,
                    "chat_id": row.chat_id,
                    "user_id": row.user_id,
                    "user_nickname": row.user_nickname,
                    "processed_plain_text": row.processed_plain_text,
                    "display_message": row.display_message,
                    "is_mentioned": row.is_mentioned,
                    "is_at": row.is_at,
                    "interest_value": row.interest_value,
                })
            return results
        except Exception as e:
            logger.error(f"查询最近消息失败: {e}")
            return []

    @staticmethod
    async def query_by_time_range(chat_id: str, start_ts: float,
                                   end_ts: float) -> List[Dict[str, Any]]:
        try:
            rows = (Messages.select()
                    .where((Messages.chat_id == chat_id) &
                           (Messages.time >= start_ts) &
                           (Messages.time <= end_ts))
                    .order_by(Messages.time))
            results = []
            for row in rows:
                results.append({
                    "message_id": row.message_id,
                    "time": row.time,
                    "chat_id": row.chat_id,
                    "user_id": row.user_id,
                    "user_nickname": row.user_nickname,
                    "processed_plain_text": row.processed_plain_text,
                    "display_message": row.display_message,
                    "is_mentioned": row.is_mentioned,
                    "is_at": row.is_at,
                })
            return results
        except Exception as e:
            logger.error(f"时间范围查询失败: {e}")
            return []

    @staticmethod
    async def count_by_chat(chat_id: str) -> int:
        try:
            return Messages.select().where(Messages.chat_id == chat_id).count()
        except Exception as e:
            logger.error(f"消息计数失败: {e}")
            return 0

    @staticmethod
    async def purge_old(days: int = 30) -> int:
        cutoff = time.time() - (days * 86400)
        try:
            count = Messages.delete().where(Messages.time < cutoff).execute()
            logger.info(f"清理 {count} 条过期消息 (>{days}天)")
            return count
        except Exception as e:
            logger.error(f"清理旧消息失败: {e}")
            return 0

    @staticmethod
    async def purge_unimportant(chat_id: str, keep_limit: int = 500) -> int:
        try:
            subquery = (Messages.select(Messages.id)
                        .where(Messages.chat_id == chat_id)
                        .order_by(Messages.time.desc())
                        .limit(keep_limit))
            keep_ids = [row.id for row in subquery]
            if not keep_ids:
                return 0
            count = (Messages.delete()
                     .where((Messages.chat_id == chat_id) &
                            (Messages.is_mentioned == False) &
                            (Messages.is_at == False) &
                            ~(Messages.id << keep_ids))
                     .execute())
            logger.info(f"清理 {count} 条非重要消息 (chat={chat_id[:12]})")
            return count
        except Exception as e:
            logger.error(f"清理非重要消息失败: {e}")
            return 0

    @staticmethod
    async def query_history_actions(chat_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            rows = (Messages.select()
                    .where((Messages.chat_id == chat_id) &
                           (Messages.priority_mode != ""))
                    .order_by(Messages.time.desc())
                    .limit(limit))
            results = []
            for row in rows:
                results.append({
                    "message_id": row.message_id,
                    "time": row.time,
                    "action_type": row.priority_mode,
                    "text": row.processed_plain_text,
                    "user_nickname": row.user_nickname,
                })
            return results
        except Exception as e:
            logger.error(f"查询历史动作失败: {e}")
            return []

    @staticmethod
    async def save_bot_message(chat_id: str, message_id: str, text: str) -> None:
        """保存机器人自身发送的消息记录"""
        try:
            Messages.create(
                message_id=message_id,
                time=time.time(),
                chat_id=chat_id,
                reply_to="",
                is_mentioned=False,
                is_at=False,
                reply_probability_boost=0.0,
                chat_info_stream_id=chat_id,
                chat_info_platform="",
                chat_info_user_platform="",
                chat_info_user_id="bot",
                chat_info_user_nickname="bot",
                chat_info_user_cardname="",
                chat_info_group_platform="",
                chat_info_group_id="",
                chat_info_group_name="",
                chat_info_create_time=0.0,
                chat_info_last_active_time=time.time(),
                user_platform="",
                user_id="bot",
                user_nickname="bot",
                user_cardname="",
                processed_plain_text=text,
                display_message=text,
                interest_value=0,
                priority_mode="",
                priority_info={},
                is_emoji=False,
                is_picid=False,
                is_notify=False,
                is_command=False,
                intercept_message_level=0,
                key_words="[]",
                key_words_lite="[]",
                selected_expressions="",
            )
            logger.debug(f"保存机器人消息: chat={chat_id[:12]}, msg_id={message_id}")
        except Exception as e:
            logger.error(f"保存机器人消息失败: {e}")

    @staticmethod
    async def insert_kg_entity(stream_id: str, name: str, entity_type: str = "general") -> None:
        """插入或更新知识图谱实体"""
        try:
            now = time.time()
            existing = (KGEntity.select()
                        .where((KGEntity.name == name) & (KGEntity.stream_id == stream_id))
                        .first())
            if existing:
                KGEntity.update(
                    appear_count=KGEntity.appear_count + 1,
                    last_active_time=now
                ).where(KGEntity.id == existing.id).execute()
            else:
                KGEntity.create(
                    name=name,
                    entity_type=entity_type,
                    appear_count=1,
                    last_active_time=now,
                    stream_id=stream_id,
                )
        except Exception as e:
            logger.error(f"插入KG实体失败: {e}")

    @staticmethod
    async def insert_kg_relation(stream_id: str, subject: str, predicate: str,
                                  object_name: str, confidence: float = 1.0) -> None:
        """插入知识图谱关系三元组"""
        try:
            now = time.time()
            existing = (KGRelation.select()
                        .where((KGRelation.subject == subject) &
                               (KGRelation.predicate == predicate) &
                               (KGRelation.object_name == object_name) &
                               (KGRelation.stream_id == stream_id))
                        .first())
            if not existing:
                KGRelation.create(
                    subject=subject,
                    predicate=predicate,
                    object_name=object_name,
                    confidence=confidence,
                    created_time=now,
                    stream_id=stream_id,
                )
        except Exception as e:
            logger.error(f"插入KG关系失败: {e}")

    @staticmethod
    async def query_kg_triples(stream_id: str, subject: Optional[str] = None,
                                limit: int = 10) -> List[Dict[str, Any]]:
        """查询知识图谱三元组"""
        try:
            query = KGRelation.select().where(KGRelation.stream_id == stream_id)
            if subject:
                query = query.where(
                    (KGRelation.subject == subject) | (KGRelation.object_name == subject)
                )
            rows = query.order_by(KGRelation.confidence.desc()).limit(limit)
            results = []
            for row in rows:
                results.append({
                    "subject": row.subject,
                    "predicate": row.predicate,
                    "object": row.object_name,
                    "confidence": row.confidence,
                    "created_time": row.created_time,
                    "stream_id": row.stream_id,
                })
            return results
        except Exception as e:
            logger.error(f"查询KG三元组失败: {e}")
            return []

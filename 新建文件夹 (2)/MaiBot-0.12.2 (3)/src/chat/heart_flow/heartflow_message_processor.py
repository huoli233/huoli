import re
import time
import traceback
import asyncio

from typing import TYPE_CHECKING, Optional, Dict, Set

from src.chat.message_receive.message import MessageRecv
from src.chat.message_receive.storage import MessageStorage
from src.chat.heart_flow.heartflow import heartflow
from src.chat.utils.utils import is_mentioned_bot_in_message
from src.chat.utils.chat_message_builder import replace_user_references
from src.common.logger import get_logger
from src.person_info.person_info import Person
from src.common.database.database_model import Images

if TYPE_CHECKING:
    pass

logger = get_logger("chat")


class HeartFCMessageReceiver:
    _processed_ids: Set[str] = set()
    _boot_time: float = time.time()

    def __init__(self):
        self.storage = MessageStorage()
        self._startup_mode = True
        try:
            asyncio.get_event_loop().create_task(self._end_startup_protection())
        except RuntimeError:
            self._startup_mode = False

    async def _end_startup_protection(self):
        await asyncio.sleep(30)
        self._startup_mode = False
        logger.info("消息处理层启动保护期结束，恢复全量处理")

    async def process_message(self, message: MessageRecv) -> None:
        try:
            if message.is_notify:
                logger.debug("通知消息，跳过处理")
                return
            userinfo = message.message_info.user_info
            chat = message.chat_stream
            now_ts = time.time()
            msg_ts = getattr(message, 'timestamp', now_ts) or now_ts
            is_old = (now_ts - msg_ts) > 180
            is_private = not chat.group_info
            if is_old and self._startup_mode and not is_private:
                logger.debug(f"启动保护 | 跳过陈旧消息 (偏移 {now_ts - msg_ts:.0f}s)")
                return
            msg_id = getattr(message, 'message_id', None) or id(message)
            msg_id_str = str(msg_id)
            if msg_id_str in HeartFCMessageReceiver._processed_ids:
                logger.debug("消息去重 | 跳过已处理消息")
                return
            HeartFCMessageReceiver._processed_ids.add(msg_id_str)
            if len(HeartFCMessageReceiver._processed_ids) > 500:
                HeartFCMessageReceiver._processed_ids = set(list(HeartFCMessageReceiver._processed_ids)[-250:])
            is_mentioned, is_at, reply_probability_boost = is_mentioned_bot_in_message(message)
            message.is_mentioned = is_mentioned
            message.is_at = is_at
            message.reply_probability_boost = reply_probability_boost
            await self.storage.store_message(message, chat)
            await heartflow.get_or_create_heartflow_chat(chat.stream_id)
            mes_name = chat.group_info.group_name if chat.group_info else "私聊"
            picid_pattern = r"\[picid:([^\]]+)\]"
            picid_list = re.findall(picid_pattern, message.processed_plain_text)
            processed_text = message.processed_plain_text
            if picid_list:
                for picid in picid_list:
                    image = Images.get_or_none(Images.image_id == picid)
                    if image and image.description:
                        processed_text = processed_text.replace(f"[picid:{picid}]", f"[图片：{image.description}]")
                    else:
                        processed_text = processed_text.replace(f"[picid:{picid}]", "[图片]")
            processed_plain_text = replace_user_references(
                processed_text,
                message.message_info.platform,
                replace_bot_name=True,
            )
            logger.info(f"[{mes_name}]{userinfo.user_nickname}:{processed_plain_text}")
            message_text = (processed_plain_text or "").strip()
            message_len = len(message_text)
            min_length = 3
            is_short_message = message_len < min_length
            whitelist = ["?", "？", "!", "！", "在吗", "你在", "干嘛"]
            is_whitelisted = message_text in whitelist
            is_command = message_text.startswith(("/", "！", "!"))
            should_skip = False
            if is_short_message and not is_whitelisted and not is_command:
                if is_mentioned or is_at:
                    should_skip = False
                elif not is_private:
                    should_skip = True
            if should_skip:
                logger.debug(f"短消息过滤 | 跳过长度为 {message_len} 的消息: {message_text}")
            user_id = str(userinfo.user_id) if userinfo else ""
            engagement_score = await self._calculate_engagement(message, processed_plain_text)
            await self._store_to_memory(
                stream_id=chat.stream_id, user_id=user_id,
                content=processed_plain_text,
                engagement_score=engagement_score,
                is_mentioned=is_mentioned, is_at=is_at,
            )
            self._log_user_status(chat.stream_id, user_id, userinfo.user_nickname if userinfo else "")
            group_id = None
            group_nick_name = None
            if chat.group_info:
                group_id = chat.group_info.group_id
                group_nick_name = userinfo.user_cardname
            _ = Person.register_person(
                platform=message.message_info.platform,
                user_id=message.message_info.user_info.user_id,
                nickname=userinfo.user_nickname,
                group_id=group_id,
                group_nick_name=group_nick_name,
            )
            if message_text and len(message_text) >= 10 and user_id:
                try:
                    from src.modules.modcore.social_cognition.knowledge_graph import get_knowledge_graph_manager
                    kg = get_knowledge_graph_manager(chat.stream_id)
                    kg.learn_from_message(message_text, msg_id=msg_id_str)
                except Exception:
                    pass
            if message_text and user_id:
                try:
                    from src.modules.modcore.social_cognition.jargon_learner import get_jargon_learner
                    learner = get_jargon_learner(chat.stream_id)
                    if not hasattr(learner, '_msg_buffer'):
                        learner._msg_buffer = []
                    learner._msg_buffer.append({
                        "sender_name": userinfo.user_nickname if userinfo else "",
                        "sender_id": user_id,
                        "content": message_text,
                        "timestamp": time.time(),
                    })
                    if len(learner._msg_buffer) >= 20:
                        msgs_to_learn = learner._msg_buffer[:]
                        learner._msg_buffer.clear()
                        asyncio.create_task(learner.learn_from_messages(msgs_to_learn))
                except Exception:
                    pass
            if user_id and message_text:
                try:
                    from src.chat.heart_flow.user_state_detector import get_user_state_detector
                    detector = get_user_state_detector()
                    detector.record_interaction(
                        user_id=user_id,
                        message_content=message_text,
                        is_initiated_by_user=(is_mentioned or is_at),
                    )
                except Exception:
                    pass
            if user_id and message_text and len(message_text) >= 3:
                try:
                    from src.modules.modcore.social_cognition.relationship_controller import get_relationship_controller
                    rel_ctrl = get_relationship_controller(chat.stream_id)
                    intent_result = await rel_ctrl.analyze_user_interaction(
                        user_id=user_id,
                        user_name=userinfo.user_nickname if userinfo else "",
                        message=message_text,
                    )
                    intent_score = intent_result.get("score", 0)
                    if intent_score != 0:
                        from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
                        tracker = get_emotion_tracker(chat.stream_id)
                        state = tracker.get_user_state(user_id)
                        if state:
                            delta = float(intent_score) * 0.5
                            state.affection = max(-100, min(100, state.affection + delta))
                            logger.debug(
                                f"[关系分析] {intent_result.get('desc','')}, "
                                f"好感变化: {delta:+.1f} -> {state.affection:.1f}"
                            )
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"消息处理失败: {e}")
            print(traceback.format_exc())

    async def _calculate_engagement(self, message: MessageRecv, text: str) -> float:
        try:
            from src.chat.chatter.interest_calculator import get_engagement_scorer
            scorer = get_engagement_scorer()
            result = scorer.calculate_interest(text)
            return result.score
        except Exception as e:
            logger.debug(f"兴趣度计算失败: {e}")
            return 0.5

    async def _store_to_memory(self, stream_id: str, user_id: str,
                                content: str, engagement_score: float,
                                is_mentioned: bool, is_at: bool):
        try:
            should_store_full = (
                engagement_score >= 0.6 or is_mentioned or is_at
            )
            if should_store_full:
                from src.memory_system.memory_core import get_memory_manager
                memory_mgr = get_memory_manager()
                importance = max(0.5, engagement_score)
                memory_mgr.create_memory(
                    stream_id=stream_id, content=content,
                    memory_type="conversation", user_id=user_id,
                    importance=importance,
                )
            else:
                try:
                    from src.memory_system.hippocampus_buffer import get_hippocampus_buffer
                    hippo = get_hippocampus_buffer(stream_id)
                    hippo.add_memory(content=content, user_id=user_id,
                                     importance=max(0.2, engagement_score))
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"记忆存储失败: {e}")

    def _detect_mention(self, message: MessageRecv) -> tuple:
        """检测消息是否提及了机器人，返回(is_mentioned, is_at, boost)"""
        from src.config.config import global_config
        is_mentioned_flag = False
        is_at_flag = False
        boost = 0.0
        if message.is_mentioned:
            is_mentioned_flag = True
            is_at_flag = True
            boost = 0.5
        if message.is_at:
            is_at_flag = True
            is_mentioned_flag = True
            boost = max(boost, 0.5)
        text = (message.processed_plain_text or "").lower()
        bot_name = global_config.bot.nickname.lower() if global_config.bot.nickname else ""
        alt_names = [n.lower() for n in global_config.bot.alias_names] if global_config.bot.alias_names else []
        all_triggers = [bot_name] + alt_names if bot_name else alt_names
        for trigger in all_triggers:
            if trigger and trigger in text:
                is_mentioned_flag = True
                boost = max(boost, 0.3)
                break
        return is_mentioned_flag, is_at_flag, boost

    def _format_mentions(self, text: str, platform: str = "") -> str:
        """格式化消息中的提及引用，将内部格式转为可读格式"""
        reply_pattern = r"回复<([^:]+):([^>]+)>"
        at_pattern = r"@<([^:]+):([^>]+)>"
        def _replace_reply(match):
            name = match.group(1)
            return f"回复{name}"
        def _replace_at(match):
            name = match.group(1)
            return f"@{name}"
        result = re.sub(reply_pattern, _replace_reply, text)
        result = re.sub(at_pattern, _replace_at, result)
        return result

    def _register_participant(
        self,
        platform: str,
        uid: str,
        display_name: str,
        group_id: Optional[str] = None,
        card_name: Optional[str] = None,
    ):
        """注册参与者信息到本地缓存"""
        try:
            participant_key = f"{platform}:{uid}"
            participant_info = {
                "platform": platform,
                "uid": uid,
                "display_name": display_name,
                "group_id": group_id,
                "card_name": card_name,
                "last_seen": time.time(),
            }
            if not hasattr(self, "_participant_cache"):
                self._participant_cache = {}
            self._participant_cache[participant_key] = participant_info
        except Exception as e:
            logger.debug(f"注册参与者失败: {e}")

    def _log_user_status(self, stream_id: str, user_id: str, display_name: str):
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(stream_id)
            state = tracker.get_user_state(user_id, create_if_missing=False)
            if not state:
                return
            status_parts = []
            affection = int(state.affection)
            status_parts.append(f"好感{affection:+d}")
            status_parts.append(f"{state.relationship}")
            annoyance = int(state.annoyance)
            if annoyance > 30:
                status_parts.append(f"烦{annoyance}")
            trauma = state.trauma_score
            if trauma > 3:
                status_parts.append(f"创{trauma:.1f}")
            if status_parts:
                logger.debug(f"用户状态 {display_name}: {' '.join(status_parts)}")
        except Exception:
            pass

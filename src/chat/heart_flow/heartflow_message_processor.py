import re
import time as _tm
import asyncio
import traceback
from typing import Any, Dict, Optional, Set

from src.chat.heart_flow.heartflow import heartflow
from src.chat.message_receive.message import MessageRecv
from src.chat.message_receive.storage import MessageStorage
from src.chat.utils.chat_message_builder import replace_user_references
from src.chat.utils.utils import is_mentioned_bot_in_message, is_bot_self
from src.common.logger import get_logger
from src.config.core_config_engine import get_core_config
from src.person_info.person_info import Person

logger = get_logger("消息管线")


def _normalize_repeat_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or "").lower(), flags=re.UNICODE)


# ---------------------------------------------------------------------------
#  配置读取
# ---------------------------------------------------------------------------


def _pipe_cfg() -> Dict[str, Any]:
    """从配置中枢读取消息取得参数"""
    try:
        return get_core_config().resolve_module_view("message_processor").values
    except RuntimeError:
        return {}


# ---------------------------------------------------------------------------
#  去重环
# ---------------------------------------------------------------------------


class _DeduplicationRing:
    """固定容量的消息ID去重环"""

    def __init__(self):
        from collections import deque

        self._cap = int(_pipe_cfg().get("dedup_cap", 500))
        self._ring: Set[str] = set()
        self._order: deque = deque(maxlen=self._cap)

    def seen(self, mid: str) -> bool:
        return mid in self._ring

    def register(self, mid: str) -> None:
        if mid in self._ring:
            return
        if len(self._order) == self._order.maxlen:
            evicted = self._order[0]
            self._ring.discard(evicted)
        self._ring.add(mid)
        self._order.append(mid)


# ---------------------------------------------------------------------------
#  图片描述替换器
# ---------------------------------------------------------------------------

_PICID_RE = re.compile(r"\[picid:([^\]]+)\]")


def _resolve_image_tags(raw_text: str) -> str:
    """将 [picid:xxx] 标记替换为图片描述或占位符"""
    c = _pipe_cfg()
    tpl = c.get("img_desc_tpl", "[图片：{desc}]")
    fallback = c.get("img_fallback", "[图片]")
    refs = _PICID_RE.findall(raw_text)
    if not refs:
        return raw_text
    result = raw_text
    try:
        from src.common.database.database_model import Images

        for pid in refs:
            tag = f"[picid:{pid}]"
            record = Images.get_or_none(Images.image_id == pid)
            if record and record.description:
                result = result.replace(tag, tpl.format(desc=record.description))
            else:
                result = result.replace(tag, fallback)
    except Exception:
        for pid in refs:
            result = result.replace(f"[picid:{pid}]", fallback)
    return result


# ---------------------------------------------------------------------------
#  短消息过滤器
# ---------------------------------------------------------------------------


class _ShortMessageGate:
    """判断短消息是否应跳过处理"""

    _BUILTIN_WHITELIST = {
        "？",
        "?",
        "。",
        "!",
        "！",
        "...",
        "…",
        "嗯",
        "啊",
        "哦",
        "嗯嗯",
        "ok",
        "好",
    }

    @staticmethod
    def should_skip(
        text: str,
        is_mentioned: bool,
        is_at: bool,
        is_private: bool,
    ) -> bool:
        c = _pipe_cfg()
        if not c.get("short_filter_on", True):
            return False
        min_chars = int(c.get("short_min", 3))
        whitelist = c.get("short_whitelist", [])
        at_also = c.get("at_also_filter", False)
        stripped = text.strip()
        if len(stripped) >= min_chars:
            return False
        if stripped in whitelist or stripped in _ShortMessageGate._BUILTIN_WHITELIST:
            return False
        if stripped.startswith(("/", "!", "！")):
            return False
        if (is_mentioned or is_at) and not at_also:
            return False
        if is_private:
            return False
        return True


# ---------------------------------------------------------------------------
#  参与度评估
# ---------------------------------------------------------------------------


async def _evaluate_engagement(text: str) -> float:
    """调用兴趣度评估器，失败时返回配置回退值"""
    c = _pipe_cfg()
    fallback = float(c.get("engage_fallback", 0.5))
    try:
        from src.chat.chatter.interest_calculator import get_engagement_scorer

        scorer = get_engagement_scorer()
        result = scorer.calculate_interest(text)
        return float(result.score)
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
#  记忆存储桥
# ---------------------------------------------------------------------------


async def _route_to_memory(
    stream_id: str,
    user_id: str,
    content: str,
    engagement: float,
    is_flagged: bool,
    platform: str = "",
) -> None:
    """根据参与度分流写入完整记忆或海马体临时缓存"""
    if is_bot_self(platform, user_id):
        return
    c = _pipe_cfg()
    full_thr = float(c.get("engage_full_thr", 0.6))
    hippo_min = float(c.get("hippo_min_imp", 0.2))
    should_full = engagement >= full_thr or is_flagged
    try:
        if should_full:
            from src.memory_system.memory_core import acquire_recollection_hub

            mgr = acquire_recollection_hub()
            mgr.deposit_memory(
                stream_id=stream_id,
                content=content,
                entry_category="conversation",
                user_id=user_id,
                significance=max(0.5, engagement),
            )
        else:
            from src.memory_system.hippocampus_buffer import (
                get_hippocampus_buffer,
            )

            hippo = get_hippocampus_buffer(stream_id)
            hippo.add_memory(
                content=content,
                user_id=user_id,
                importance=max(hippo_min, engagement),
            )
    except Exception as exc:
        logger.debug(f"记忆存储异常: {exc}")


# ---------------------------------------------------------------------------
#  用户状态记录桥
# ---------------------------------------------------------------------------


def _record_user_interaction(user_id: str, text: str, explicitly_called: bool) -> None:
    """异步向用户状态检测器提交交互记录"""
    try:
        from src.modules.perception.user_state import (
            get_user_state_detector,
        )

        det = get_user_state_detector()
        payload = str(text or "")
        det.feed_interaction(
            uid=user_id,
            msg_length=len(payload),
            has_question=("?" in payload or "？" in payload),
            is_user_initiated=explicitly_called,
            has_emoji=bool(payload and ("[" in payload and "]" in payload)),
        )
    except Exception as _e:
        logger.debug(f"异常: {_e}")


async def _detect_user_repeat(user_id: str, stream_id: str, text: str) -> None:
    """检测用户重复发送消息，更新AI厌烦度"""
    if not user_id or not text:
        return
    try:
        from src.chat.utils.user_repeat_detector import (
            get_user_repeat_detector,
        )

        detector = get_user_repeat_detector()
        repeat_result = await detector.detect_and_update_annoyance(
            user_id=user_id,
            stream_id=stream_id,
            message_text=text,
            auto_update=True,
        )
        is_repeat = False
        repeat_count = 0
        max_similarity = 0.0
        emotion_state = {}
        if isinstance(repeat_result, tuple):
            if len(repeat_result) == 4:
                is_repeat, repeat_count, max_similarity, emotion_state = repeat_result
            elif len(repeat_result) == 3:
                is_repeat, max_similarity, emotion_state = repeat_result
                repeat_count = int(bool(is_repeat))
        if is_repeat and repeat_count >= 1:
            logger.debug(
                f"[重复检测] 用户{user_id[:8]}...检测到重复"
                f"（{repeat_count}次，相似度={max_similarity:.2f}），"
                f"厌烦度={emotion_state.get('annoyance', 0):.1f}"
            )
    except Exception as _e:
        logger.debug(f"异常: {_e}")


async def _record_dialogue_memoir(
    stream_id: str,
    user_id: str,
    sender_name: str,
    text: str,
    msg_time: float,
) -> None:
    """将真实用户消息写入 proactive 回忆录，供主动链路后续使用。"""
    if not user_id or not text:
        return
    try:
        from src.chat.proactive.session_tracker import get_memoir_cabinet

        cabinet = get_memoir_cabinet()
        memoir = await cabinet.fetch_memoir(user_id=user_id, channel_id=stream_id)
        memoir.record_user_utterance(
            content=text,
            sender_name=sender_name,
            sender_id=user_id,
            msg_time=msg_time,
        )
        await cabinet.persist_memoir(user_id)
    except Exception as exc:
        logger.debug(f"主动回忆录记录异常: {exc}")


def _record_context_message(
    stream_id: str,
    msg_id: str,
    user_id: str,
    user_name: str,
    text: str,
    msg_time: float,
    is_bot: bool,
) -> None:
    """把消息写入旧版风格的稳定上下文层。"""
    if not text:
        return
    try:
        from src.modules.context_manager import (
            create_context_message,
            get_context_manager,
        )

        get_context_manager().add_message(
            stream_id,
            create_context_message(
                content=text,
                user_id=user_id,
                user_name=user_name,
                message_id=msg_id,
                timestamp=msg_time,
                metadata={"is_bot": is_bot},
            ),
        )
    except Exception as exc:
        logger.debug(f"上下文写入异常: {exc}")


# ---------------------------------------------------------------------------
#  知识图谱/梗学习桥
# ---------------------------------------------------------------------------


def _try_learn_knowledge(stream_id: str, text: str, msg_id: str) -> None:
    """长消息投递知识图谱学习"""
    c = _pipe_cfg()
    min_len = int(c.get("kg_min", 10))
    if len(text) < min_len:
        return
    try:
        from src.modules.modcore.social_cognition.knowledge_graph import (
            get_knowledge_graph_manager,
        )

        kg = get_knowledge_graph_manager(stream_id)
        kg.learn_from_message(text, msg_id=msg_id)
    except Exception as _e:
        logger.debug(f"异常: {_e}")


# ---------------------------------------------------------------------------
#  主接收器
# ---------------------------------------------------------------------------


class MessageIntakePipeline:
    """消息摄入管线

    生命周期：
      1. 启动保护期内跳过群内超龄消息
      2. 去重 → 提及检测 → 图片描述替换 → 消息存储
      3. 短消息过滤
      4. 参与度评估 → 记忆分流
      5. 注册参与者 → 知识图谱学习
    6. 广播活跃度给心流与静默链
    """

    _dedup = _DeduplicationRing()

    def __init__(self):
        self._archive = MessageStorage()
        self._shielding = True
        self._boot_ts = _tm.time()
        self._shield_task: Optional[asyncio.Task] = None
        self._ensure_shield_timer()

    def _ensure_shield_timer(self) -> None:
        if not self._shielding:
            return
        if self._shield_task is not None and not self._shield_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._shield_task = loop.create_task(self._expire_shield())

    async def _expire_shield(self) -> None:
        c = _pipe_cfg()
        dur = float(c.get("boot_shield_duration", c.get("boot_shield", 30)))
        await asyncio.sleep(dur)
        self._shielding = False
        logger.info("消息管线启动保护期结束")

    # ---- 主入口 ----

    async def process_message(self, message: MessageRecv) -> None:
        try:
            self._ensure_shield_timer()
            if message.is_notify:
                return
            userinfo = message.message_info.user_info
            chat = message.chat_stream
            if self._should_discard_stale(message, chat):
                return
            mid = self._extract_msg_id(message)
            if self._dedup.seen(mid):
                logger.debug("重复消息，跳过")
                return
            self._dedup.register(mid)
            is_mentioned, is_at, boost = is_mentioned_bot_in_message(message)
            message.is_mentioned = is_mentioned
            message.is_at = is_at
            message.reply_probability_boost = boost
            raw_text = message.processed_plain_text
            enriched_text = _resolve_image_tags(raw_text)
            message.processed_plain_text = enriched_text
            if not getattr(message, "detailed_plain_text", None):
                message.detailed_plain_text = enriched_text
            await self._archive.store_message(message, chat)
            await heartflow.get_or_create_heartflow_chat(chat.stream_id)
            readable = replace_user_references(
                enriched_text,
                message.message_info.platform,
                replace_bot_name=True,
            )
            chan_label = chat.group_info.group_name if chat.group_info else "私聊"
            c = _pipe_cfg()
            log_max = int(c.get("log_max", 120))
            display = readable[:log_max] + "..." if len(readable) > log_max else readable
            logger.info(f"[{chan_label}]{userinfo.user_nickname}: {display}")
            is_private = not chat.group_info
            clean_text = (readable or "").strip()
            self._annotate_repeated_short_input(message, chat.stream_id, clean_text)
            uid = str(userinfo.user_id) if userinfo else ""
            _add_cfg = getattr(message.message_info, "additional_config", None) or {}
            is_self_msg = bool(_add_cfg.get("is_bot_self") or _add_cfg.get("is_bot"))
            if not is_self_msg:
                is_self_msg = is_bot_self(message.message_info.platform, uid)
            _record_context_message(
                stream_id=chat.stream_id,
                msg_id=mid,
                user_id=uid or "bot",
                user_name=str(getattr(userinfo, "user_nickname", "") or "bot"),
                text=clean_text,
                msg_time=float(getattr(message.message_info, "time", _tm.time()) or _tm.time()),
                is_bot=is_self_msg,
            )
            if is_self_msg:
                heartflow.note_bot_activity(chat.stream_id)
                logger.debug("检测到机器人自身回流消息，仅写入上下文与活动状态，不进入用户消息处理链")
                return
            else:
                heartflow.note_user_activity(chat.stream_id, uid, clean_text)
            if _ShortMessageGate.should_skip(clean_text, is_mentioned, is_at, is_private):
                logger.debug(f"短消息过滤: len={len(clean_text)}")
                return
            engagement = await _evaluate_engagement(clean_text)
            await _route_to_memory(
                stream_id=chat.stream_id,
                user_id=uid,
                content=clean_text,
                engagement=engagement,
                is_flagged=(is_mentioned or is_at),
                platform=message.message_info.platform,
            )
            _record_user_interaction(uid, clean_text, is_mentioned or is_at)
            await _detect_user_repeat(uid, chat.stream_id, clean_text)
            await _record_dialogue_memoir(
                stream_id=chat.stream_id,
                user_id=uid,
                sender_name=userinfo.user_nickname,
                text=clean_text,
                msg_time=float(getattr(message.message_info, "time", _tm.time()) or _tm.time()),
            )
            group_id = None
            card_name = None
            if chat.group_info:
                group_id = chat.group_info.group_id
                card_name = userinfo.user_cardname
            Person.register_person(
                platform=message.message_info.platform,
                user_id=userinfo.user_id,
                nickname=userinfo.user_nickname,
                group_id=group_id,
                group_nick_name=card_name,
            )
            _try_learn_knowledge(chat.stream_id, clean_text, mid)
            self._log_participant_snapshot(chat.stream_id, uid, userinfo.user_nickname)
        except Exception as exc:
            logger.error(f"消息管线异常: {exc}")
            logger.error(traceback.format_exc())

    def _annotate_repeated_short_input(self, message: MessageRecv, chat_id: str, clean_text: str) -> None:
        """在入口层给短重复句打标，提示后续主链按前情延续来理解。"""
        normalized = _normalize_repeat_text(clean_text)
        if not normalized or len(normalized) > 12:
            return

        try:
            recent_messages = fetch_recent_messages(
                chat_id=chat_id,
                start_time=_tm.time() - 1800,
                end_time=_tm.time(),
                limit=8,
                limit_mode="latest",
                filter_bot=False,
                filter_command=False,
            )
        except Exception:
            recent_messages = []

        same_user_repeat = 0
        latest_bot_reply = ""
        for item in reversed(recent_messages or []):
            item_user_id = str(getattr(item, "user_id", "") or "")
            item_text = str(getattr(item, "processed_plain_text", "") or "").strip()
            if not item_text:
                continue
            if item_user_id == "bot" and not latest_bot_reply:
                latest_bot_reply = item_text[:80]
                continue
            if item_user_id == str(getattr(message.message_info.user_info, "user_id", "") or ""):
                if _normalize_repeat_text(item_text) == normalized:
                    same_user_repeat += 1

        if same_user_repeat >= 1 and len(normalized) <= 8:
            message.is_repeated_short_input = True
            if latest_bot_reply:
                message.context_continuation_hint = (
                    f"这是同一用户重复提到的短句，优先按前情延续理解；你刚刚已经回过：{latest_bot_reply}"
                )
            else:
                message.context_continuation_hint = "这是同一用户重复提到的短句，优先按前情延续理解，不要当作全新话题。"

    # ---- 辅助方法 ----

    def _should_discard_stale(self, message: MessageRecv, chat) -> bool:
        """启动保护期内丢弃陈旧的群消息"""
        if not self._shielding:
            return False
        is_private = not chat.group_info
        if is_private:
            return False
        c = _pipe_cfg()
        stale_thr = float(c.get("stale_thr", 180))
        now = _tm.time()
        msg_ts = getattr(getattr(message, "message_info", None), "time", None) or now
        age = now - msg_ts
        if age > stale_thr:
            logger.debug(f"启动保护 | 跳过陈旧消息 (偏移 {age:.0f}s)")
            return True
        return False

    @staticmethod
    def _extract_msg_id(message: MessageRecv) -> str:
        raw = getattr(getattr(message, "message_info", None), "message_id", None) or id(message)
        return str(raw)

    @staticmethod
    def _log_participant_snapshot(stream_id: str, uid: str, name: str) -> None:
        """尝试打印用户状态摘要"""
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            tracker = get_emotion_tracker(stream_id)
            state = tracker.get_user_state(uid, create_if_missing=False)
            if not state:
                return
            parts = []
            aff = int(state.affection)
            parts.append(f"好感{aff:+d}")
            parts.append(state.relationship)
            annoy = int(state.annoyance)
            if annoy > 30:
                parts.append(f"烦{annoy}")
            trauma = state.trauma_score
            if trauma > 3:
                parts.append(f"创{trauma:.1f}")
            if parts:
                logger.debug(f"用户快照 {name}: {' '.join(parts)}")
        except Exception as _e:
            logger.debug(f"异常: {_e}")


# ---------------------------------------------------------------------------
#  消息查询工具
# ---------------------------------------------------------------------------


def fetch_recent_messages(
    chat_id: str,
    start_time: float,
    end_time: float,
    limit: int = 20,
    limit_mode: str = "latest",
    filter_bot: bool = True,
    filter_command: bool = False,
) -> list:
    """根据时间范围查询历史消息列表"""
    try:
        from src.chat.utils.chat_message_builder import (
            get_raw_msg_by_timestamp_with_chat,
        )

        return get_raw_msg_by_timestamp_with_chat(
            chat_id=chat_id,
            timestamp_start=start_time,
            timestamp_end=end_time,
            limit=limit,
            limit_mode=limit_mode,
            filter_bot=filter_bot,
            filter_command=filter_command,
        )
    except Exception as exc:
        logger.error(f"查询消息异常: {exc}")
        return []


# ---------------------------------------------------------------------------
#  模块级单例
# ---------------------------------------------------------------------------

message_receiver = MessageIntakePipeline()

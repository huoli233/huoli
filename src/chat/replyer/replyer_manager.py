from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.chat.message_receive.chat_stream import ChatStream, get_chat_manager
from src.chat.replyer.group_generator import DefaultReplyer
from src.chat.replyer.private_generator import PrivateReplyer
from src.chat.replyer.reply_intent import (
    ReplyIntentPacket,
    build_reply_intent_from_message,
    relation_to_reply_intent,
)

logger = get_logger("ReplyerManager")


class ReplyerManager:
    """回复器管理器

    职责:
      - 按 stream_id 缓存 DefaultReplyer / PrivateReplyer 实例
      - 提供缓存失效、批量清理
      - 高层群聊/私聊回复入口
      - 回复后状态日志输出
    """

    def __init__(self):
        self._repliers: Dict[str, DefaultReplyer | PrivateReplyer] = {}

    # ---- 实例获取 ----

    def get_replyer(
        self,
        chat_stream: Optional[ChatStream] = None,
        chat_id: Optional[str] = None,
        request_type: str = "replyer",
    ) -> Optional[DefaultReplyer | PrivateReplyer]:
        """获取或创建回复器实例（带缓存验证）"""
        stream_id = chat_stream.stream_id if chat_stream else chat_id
        if not stream_id:
            logger.warning("[ReplyerManager] 缺少 stream_id")
            return None
        cached = self._repliers.get(stream_id)
        if cached is not None:
            if hasattr(cached, "generate_reply_with_context"):
                return cached
            logger.warning(f"[ReplyerManager] 缓存无效，重建 {stream_id[:12]}")
            del self._repliers[stream_id]
        target_stream = chat_stream
        if not target_stream:
            if chat_manager := get_chat_manager():
                target_stream = chat_manager.get_stream(stream_id)
        if not target_stream:
            logger.warning(
                f"[ReplyerManager] 未找到流 {stream_id[:12]}，无法创建回复器"
            )
            return None
        if target_stream.group_info:
            replyer = DefaultReplyer(
                chat_stream=target_stream,
                request_type=request_type,
            )
        else:
            replyer = PrivateReplyer(
                chat_stream=target_stream,
                request_type=request_type,
            )
        self._repliers[stream_id] = replyer
        return replyer

    # ---- 缓存管理 ----

    def clear_cache(self, stream_id: Optional[str] = None) -> None:
        """清除缓存。传入 stream_id 清除单个，否则清除全部"""
        if stream_id:
            self._repliers.pop(stream_id, None)
        else:
            self._repliers.clear()

    def cached_count(self) -> int:
        return len(self._repliers)

    def cached_stream_ids(self) -> List[str]:
        return list(self._repliers.keys())

    # ---- 高层回复入口 ----

    async def generate_group_reply(
        self,
        stream_id: str,
        packet_data: Dict,
        *,
        is_priority: bool = False,
        think_level: int = 1,
        mental_state: Optional[Dict] = None,
        is_trauma_blocked: bool = False,
    ) -> str:
        """群聊高层回复入口"""
        replyer = self.get_replyer(chat_id=stream_id)
        if not replyer:
            return ""
        try:
            reply_intent = self._build_manager_reply_intent(
                stream_id=stream_id,
                packet_data=packet_data,
                mental_state=mental_state,
                trigger_reason="群聊高层回复入口",
                is_trauma_blocked=is_trauma_blocked,
            )
            success, llm_response = await replyer.generate_reply_with_context(
                think_level=think_level,
                reply_intent=reply_intent,
            )
            if success and llm_response and llm_response.content:
                text = llm_response.content
                logger.info(f"群聊回复: {len(text)}字符")
                self._emit_emotion_status(stream_id, packet_data)
                return text
            return ""
        except Exception as exc:
            logger.error(f"群聊回复生成失败: {exc}")
            self.clear_cache(stream_id)
            return ""

    async def generate_private_reply(
        self,
        stream_id: str,
        packet_data: Dict,
        *,
        think_level: int = 1,
        mental_state: Optional[Dict] = None,
    ) -> str:
        """私聊高层回复入口"""
        replyer = self.get_replyer(chat_id=stream_id)
        if not replyer:
            return ""
        try:
            reply_intent = self._build_manager_reply_intent(
                stream_id=stream_id,
                packet_data=packet_data,
                mental_state=mental_state,
                trigger_reason="私聊高层回复入口",
            )
            success, llm_response = await replyer.generate_reply_with_context(
                think_level=think_level,
                reply_intent=reply_intent,
            )
            if success and llm_response and llm_response.content:
                return llm_response.content
            return ""
        except Exception as exc:
            logger.error(f"私聊回复生成失败: {exc}")
            self.clear_cache(stream_id)
            return ""

    def _build_manager_reply_intent(
        self,
        *,
        stream_id: str,
        packet_data: Dict,
        mental_state: Optional[Dict],
        trigger_reason: str,
        is_trauma_blocked: bool = False,
    ) -> ReplyIntentPacket:
        packet_data = packet_data or {}
        mental_state = mental_state or {}
        user_id = str(packet_data.get("sender_id") or packet_data.get("user_id") or "").strip()
        user_name = str(packet_data.get("sender_name") or packet_data.get("nickname") or user_id or "对方").strip()
        target_text = str(
            packet_data.get("processed_plain_text")
            or packet_data.get("plain_text")
            or packet_data.get("content")
            or packet_data.get("message")
            or ""
        ).strip()
        relation: Dict[str, Any] = {}
        if user_id:
            try:
                from src.core.world_snapshot import build_relation_rapport_snapshot

                relation = build_relation_rapport_snapshot(channel_id=stream_id, user_id=user_id) or {}
            except Exception as exc:
                logger.debug(f"[ReplyerManager] 构造统一回复意图包读取关系失败: {exc}")
        if mental_state:
            relation.update(
                {
                    "affection": mental_state.get("affection", mental_state.get("favor", relation.get("affection", 0))),
                    "trust_value": mental_state.get("trust", mental_state.get("trust_value", relation.get("trust_value", 0))),
                    "annoyance_value": mental_state.get(
                        "annoyance",
                        mental_state.get("annoyance_value", relation.get("annoyance_value", 0)),
                    ),
                    "psychological_pressure": mental_state.get(
                        "pressure",
                        mental_state.get("psychological_pressure", relation.get("psychological_pressure", 0)),
                    ),
                    "mood": mental_state.get("mood", relation.get("mood", "")),
                }
            )
        if is_trauma_blocked:
            relation["trauma_score"] = max(float(relation.get("trauma_score", 0.0) or 0.0), 5.0)
            relation["is_user_blocked"] = True
        return relation_to_reply_intent(
            target=user_name,
            target_text=target_text,
            trigger_reason=trigger_reason,
            relation_view=relation,
            reference_info="通用回复管理入口：没有心流执行块时仍必须按统一回复意图包收敛表达。",
            fast_path=False,
        )

    # ---- 状态日志 ----

    def _emit_emotion_status(self, stream_id: str, packet_data: Dict) -> None:
        """回复后输出情感/社交状态日志"""
        try:
            from src.core.world_snapshot import (
                build_relation_rapport_snapshot,
                get_relation_number,
            )

            user_id = packet_data.get("sender_id") or packet_data.get(
                "user_id", ""
            )
            if not user_id:
                return
            relation = build_relation_rapport_snapshot(
                channel_id=stream_id,
                user_id=user_id,
            )
            parts: List[str] = []
            affection = get_relation_number(relation, "affection")
            trust_value = get_relation_number(
                relation,
                "trust_value",
                aliases=("trust_score",),
            )
            annoyance_value = get_relation_number(
                relation,
                "annoyance_value",
                aliases=("annoyance",),
            )
            relation_label = str(
                relation.get("personal_impression", "")
                or relation.get("relationship", "")
                or relation.get("legacy_relationship_label", "")
                or relation.get("custom_label", "")
                or ""
            )
            parts.append(f"好感{int(affection)}")
            parts.append(f"信任{int(trust_value)}")
            if relation_label:
                parts.append(relation_label)
            if annoyance_value > 0:
                parts.append(f"烦恼{int(annoyance_value)}")
            if parts:
                logger.info(f"状态 {' '.join(parts)}")
        except Exception as exc:
            logger.debug(f"状态输出失败: {exc}")


# ---------------------------------------------------------------------------
#  响应工厂（桥接模式）
# ---------------------------------------------------------------------------


class ResponseFactory:
    """统一响应工厂

    为心流系统、插件系统、XB桥接等不同调用方提供统一的
    回复生成/重写接口。
    """

    def __init__(self):
        self._responders: Dict[str, Any] = {}

    def acquire_responder(
        self,
        chat_stream: Optional[ChatStream] = None,
        stream_id: str = "",
        request_type: str = "replyer",
    ) -> Optional[DefaultReplyer | PrivateReplyer]:
        """获取响应器实例"""
        sid = stream_id or (chat_stream.stream_id if chat_stream else "")
        if not sid:
            return None
        if sid in self._responders:
            return self._responders[sid]
        try:
            cs = chat_stream or get_chat_manager().get_stream(sid)
            if not cs:
                return None
            if cs.group_info:
                responder = DefaultReplyer(
                    chat_stream=cs, request_type=request_type
                )
            else:
                responder = PrivateReplyer(
                    chat_stream=cs, request_type=request_type
                )
            self._responders[sid] = responder
            return responder
        except Exception as exc:
            logger.error(f"创建响应器失败: {exc}")
            return None

    async def fabricate_group_response(
        self,
        chat_stream: ChatStream,
        reply_message=None,
        reply_reason: str = "",
        **kwargs,
    ):
        """群聊响应生成"""
        responder = self.acquire_responder(chat_stream=chat_stream)
        if not responder:
            return False, None
        if kwargs.get("reply_intent") is None:
            kwargs["reply_intent"] = build_reply_intent_from_message(
                channel_id=chat_stream.stream_id,
                message=reply_message,
                trigger_reason=reply_reason or "响应工厂群聊回复入口",
                reference_info="响应工厂入口：调用方未提供心流意图时，显式收敛为统一回复意图包。",
            )
        return await responder.generate_reply_with_context(
            reply_message=reply_message,
            reply_reason=reply_reason,
            **kwargs,
        )

    async def fabricate_solo_response(
        self,
        chat_stream: ChatStream,
        reply_message=None,
        reply_reason: str = "",
        **kwargs,
    ):
        """私聊响应生成"""
        responder = self.acquire_responder(chat_stream=chat_stream)
        if not responder:
            return False, None
        if kwargs.get("reply_intent") is None:
            kwargs["reply_intent"] = build_reply_intent_from_message(
                channel_id=chat_stream.stream_id,
                message=reply_message,
                trigger_reason=reply_reason or "响应工厂私聊回复入口",
                reference_info="响应工厂入口：调用方未提供心流意图时，显式收敛为统一回复意图包。",
            )
        return await responder.generate_reply_with_context(
            reply_message=reply_message,
            reply_reason=reply_reason,
            **kwargs,
        )


# ---------------------------------------------------------------------------
#  全局单例
# ---------------------------------------------------------------------------

replyer_manager = ReplyerManager()

_response_factory: Optional[ResponseFactory] = None


def get_replyer_manager() -> ReplyerManager:
    return replyer_manager


def get_response_factory() -> ResponseFactory:
    global _response_factory
    if _response_factory is None:
        _response_factory = ResponseFactory()
    return _response_factory

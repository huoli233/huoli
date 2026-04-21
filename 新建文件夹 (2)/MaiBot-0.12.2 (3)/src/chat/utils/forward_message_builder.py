import time
from typing import List, Optional, Dict, Union
from dataclasses import dataclass, field
from maim_message import Seg, UserInfo, BaseMessageInfo, GroupInfo, MessageBase
from src.common.logger import get_logger
from src.config.config import global_config

logger = get_logger("forward_builder")


@dataclass
class ForwardNode:
    content_segments: List[Seg] = field(default_factory=list)
    sender_name: str = ""
    sender_id: str = ""

    def add_text(self, text: str) -> "ForwardNode":
        self.content_segments.append(Seg(type="text", data=text))
        return self

    def add_image_base64(self, base64_data: str) -> "ForwardNode":
        self.content_segments.append(Seg(type="image", data=base64_data))
        return self

    def add_image_url(self, url: str) -> "ForwardNode":
        self.content_segments.append(Seg(type="imageurl", data=url))
        return self

    def to_message_base(self) -> dict:
        if len(self.content_segments) == 1:
            seg = self.content_segments[0]
        else:
            seg = Seg(type="seglist", data=self.content_segments)
        user_info = UserInfo(
            user_id=self.sender_id or str(global_config.bot.qq_account),
            user_nickname=self.sender_name or global_config.bot.nickname,
            platform="qq",
        )
        msg_info = BaseMessageInfo(
            platform="qq",
            message_id=f"fwd_{int(time.time()*1000)}",
            user_info=user_info,
        )
        mb = MessageBase(
            message_info=msg_info,
            message_segment=seg,
        )
        return mb.to_dict()


class ForwardMessageBuilder:
    def __init__(self):
        self._nodes: List[ForwardNode] = []

    def add_bot_text(self, text: str) -> "ForwardMessageBuilder":
        node = ForwardNode(
            sender_name=global_config.bot.nickname,
            sender_id=str(global_config.bot.qq_account),
        )
        node.add_text(text)
        self._nodes.append(node)
        return self

    def add_bot_image(self, image_base64: str, caption: str = "") -> "ForwardMessageBuilder":
        node = ForwardNode(
            sender_name=global_config.bot.nickname,
            sender_id=str(global_config.bot.qq_account),
        )
        if caption:
            node.add_text(caption)
        node.add_image_base64(image_base64)
        self._nodes.append(node)
        return self

    def add_bot_image_url(self, url: str, caption: str = "") -> "ForwardMessageBuilder":
        node = ForwardNode(
            sender_name=global_config.bot.nickname,
            sender_id=str(global_config.bot.qq_account),
        )
        if caption:
            node.add_text(caption)
        node.add_image_url(url)
        self._nodes.append(node)
        return self

    def add_custom_node(self, sender_name: str, sender_id: str, text: str = "", image_base64: str = "", image_url: str = "") -> "ForwardMessageBuilder":
        node = ForwardNode(sender_name=sender_name, sender_id=sender_id)
        if text:
            node.add_text(text)
        if image_base64:
            node.add_image_base64(image_base64)
        if image_url:
            node.add_image_url(image_url)
        self._nodes.append(node)
        return self

    def add_node(self, node: ForwardNode) -> "ForwardMessageBuilder":
        self._nodes.append(node)
        return self

    def build_seg(self) -> Optional[Seg]:
        if not self._nodes:
            return None
        node_dicts = [n.to_message_base() for n in self._nodes]
        return Seg(type="forward", data=node_dicts)

    def build_message_sending(self, chat_stream) -> Optional["MessageSending"]:
        from src.chat.message_receive.message import MessageSending
        seg = self.build_seg()
        if not seg:
            return None
        bot_user_info = UserInfo(
            user_id=global_config.bot.qq_account,
            user_nickname=global_config.bot.nickname,
            platform=chat_stream.platform,
        )
        msg = MessageSending(
            message_id=f"fwd_{int(time.time()*1000)}",
            chat_stream=chat_stream,
            bot_user_info=bot_user_info,
            sender_info=chat_stream.user_info,
            message_segment=seg,
            display_message="[合并转发消息]",
            is_head=True,
            is_emoji=False,
            thinking_start_time=time.time(),
        )
        return msg

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def clear(self):
        self._nodes.clear()


async def send_forward_message(
    chat_stream,
    nodes: List[Dict[str, str]],
) -> bool:
    from src.chat.message_receive.uni_message_sender import UniversalMessageSender
    builder = ForwardMessageBuilder()
    for item in nodes:
        text = item.get("text", "")
        image_url = item.get("image_url", "")
        image_base64 = item.get("image_base64", "")
        sender_name = item.get("sender_name", "")
        sender_id = item.get("sender_id", "")
        if sender_name or sender_id:
            builder.add_custom_node(
                sender_name=sender_name or global_config.bot.nickname,
                sender_id=sender_id or str(global_config.bot.qq_account),
                text=text,
                image_base64=image_base64,
                image_url=image_url,
            )
        elif image_base64:
            builder.add_bot_image(image_base64, caption=text)
        elif image_url:
            builder.add_bot_image_url(image_url, caption=text)
        elif text:
            builder.add_bot_text(text)
    msg = builder.build_message_sending(chat_stream)
    if not msg:
        logger.warning("[合并转发] 构建失败：无有效节点")
        return False
    sender = UniversalMessageSender()
    ok = await sender.send_message(msg, typing=False, storage_message=False)
    if ok:
        logger.info(f"[合并转发] 发送成功 ({builder.node_count}条)")
    else:
        logger.warning("[合并转发] 发送失败")
    return ok

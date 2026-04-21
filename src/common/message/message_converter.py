from typing import Any, Dict, List, Optional, Union
from src.common.message_types.seg import Seg
from src.common.message_types.user_info import UserInfo
from src.common.message_types.group_info import GroupInfo
from src.common.message_types.message_base import MessageBase
from src.common.message_types.message_info import BaseMessageInfo
from src.chat.utils.utils import is_bot_self


class MessageConverter:
    @staticmethod
    def seg_to_onebot(seg: Seg) -> Dict[str, Any]:
        seg_type = seg.type
        data = seg.data
        if seg_type == "text":
            return {"type": "text", "data": {"text": str(data)}}
        elif seg_type == "image":
            return {"type": "image", "data": {"file": f"base64://{data}" if data else ""}}
        elif seg_type == "emoji":
            return {"type": "face", "data": {"id": 0, "file": f"base64://{data}" if data else ""}}
        elif seg_type == "voice":
            return {"type": "record", "data": {"file": f"base64://{data}" if data else ""}}
        elif seg_type == "at":
            return {"type": "at", "data": {"qq": str(data)}}
        elif seg_type == "reply":
            return {"type": "reply", "data": {"id": str(data)}}
        elif seg_type == "seglist":
            return {"type": "node", "data": data}
        elif seg_type == "forward":
            return {"type": "forward", "data": data}
        else:
            return {"type": seg_type, "data": data}

    @staticmethod
    def onebot_to_seg(onebot_msg: Dict[str, Any]) -> Seg:
        msg_type = onebot_msg.get("type", "text")
        data = onebot_msg.get("data", {})
        if msg_type == "text":
            return Seg.text(data.get("text", ""))
        elif msg_type == "image":
            file = data.get("file", "")
            if file.startswith("base64://"):
                return Seg.image(file[9:])
            return Seg.image(file)
        elif msg_type == "face":
            return Seg.emoji(data.get("file", ""))
        elif msg_type == "record":
            file = data.get("file", "")
            if file.startswith("base64://"):
                return Seg.voice(file[9:])
            return Seg.voice(file)
        elif msg_type == "at":
            return Seg.at(data.get("qq", ""))
        elif msg_type == "reply":
            return Seg.reply(data.get("id", ""))
        else:
            return Seg(type=msg_type, data=data)

    @staticmethod
    def message_to_onebot(message: MessageBase) -> Dict[str, Any]:
        result = {
            "message_id": message.message_info.message_id,
            "time": int(message.message_info.time),
            "user_id": message.message_info.user_info.user_id if message.message_info.user_info else "",
            "message": [],
        }
        if message.message_info.group_info:
            result["group_id"] = message.message_info.group_info.group_id
            result["message_type"] = "group"
        else:
            result["message_type"] = "private"
        if message.message_segment:
            if message.message_segment.type == "seglist":
                segments = message.message_segment.data
                if isinstance(segments, list):
                    for seg in segments:
                        if isinstance(seg, Seg):
                            result["message"].append(MessageConverter.seg_to_onebot(seg))
                        elif isinstance(seg, dict):
                            result["message"].append(seg)
            else:
                result["message"].append(MessageConverter.seg_to_onebot(message.message_segment))
        return result

    @staticmethod
    def onebot_to_message(onebot_msg: Dict[str, Any]) -> MessageBase:
        user_info = UserInfo(
            user_id=str(onebot_msg.get("user_id", "")),
            user_nickname=str(onebot_msg.get("sender", {}).get("nickname", "")),
            platform="qq",
        )
        group_info = None
        if onebot_msg.get("message_type") == "group":
            group_info = GroupInfo(
                group_id=str(onebot_msg.get("group_id", "")),
                group_name="",
                platform="qq",
            )
        message_segments = onebot_msg.get("message", [])
        if isinstance(message_segments, list):
            if len(message_segments) == 1:
                seg = MessageConverter.onebot_to_seg(message_segments[0])
            else:
                segs = [MessageConverter.onebot_to_seg(m) for m in message_segments]
                seg = Seg.seglist(segs)
        else:
            seg = Seg.text(str(message_segments))
        message_info = BaseMessageInfo(
            platform="qq",
            message_id=str(onebot_msg.get("message_id", "")),
            time=float(onebot_msg.get("time", 0)),
            user_info=user_info,
            group_info=group_info,
            additional_config={},
        )
        try:
            self_id = str(onebot_msg.get("self_id", "") or "")
            user_id = str(onebot_msg.get("user_id", "") or "")
            if (self_id and self_id == user_id) or is_bot_self("qq", user_id):
                message_info.additional_config["is_bot_self"] = True
                message_info.additional_config["is_bot"] = True
        except Exception:
            pass
        return MessageBase(
            message_info=message_info,
            message_segment=seg,
            raw_message=onebot_msg,
        )

    @staticmethod
    def dict_to_message(data: Dict[str, Any]) -> MessageBase:
        return MessageBase.from_dict(data)

    @staticmethod
    def message_to_dict(message: MessageBase) -> Dict[str, Any]:
        return message.to_dict()

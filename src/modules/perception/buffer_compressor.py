import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.modules.perception.runtime_config import perception_module_view

logger = get_logger("buffer_compressor")


@dataclass
class StandardMessage:
    message_id: str = ""
    text: str = ""
    sender_id: str = ""
    sender_name: str = ""
    timestamp: float = 0.0
    is_at_me: bool = False
    has_image: bool = False
    image_urls: List[str] = field(default_factory=list)
    is_bot_self: bool = False


class BufferCompressor:
    def __init__(self, config_engine=None):
        del config_engine
        self._recent_show_count: int = 5
        self._recent_max_char_length: int = 50
        self._load_config()

    def _load_config(self) -> None:
        config = perception_module_view("perception_buffer")
        self._recent_show_count = int(
            config.get("recent_show_count", 5)
        )
        self._recent_max_char_length = int(
            config.get("recent_max_char_length", 50)
        )

    def compress(
        self, buffer: List[StandardMessage], bot_name: str = ""
    ) -> str:
        if not buffer:
            return "休息期间没有新消息"
        now = time.time()
        total = len(buffer)
        users = set(m.sender_id for m in buffer if m.sender_id)
        summary_line = f"积了{total}条消息, {len(users)}人发言"
        signals = []
        at_count = sum(1 for m in buffer if m.is_at_me)
        if at_count:
            signals.append(f"{at_count}次@你")
        if bot_name:
            name_count = sum(
                1 for m in buffer if m.text and bot_name in m.text
            )
            if name_count:
                signals.append(f"{name_count}次提到你名字")
        img_count = sum(1 for m in buffer if m.has_image)
        if img_count:
            signals.append(f"{img_count}张图片")
        speaker_counts = Counter(
            m.sender_name for m in buffer if m.sender_name
        )
        top_speakers = speaker_counts.most_common(2)
        if top_speakers:
            active_parts = [
                f"{name}发了{count}条" for name, count in top_speakers
            ]
            signals.append(", ".join(active_parts))
        recent = buffer[-self._recent_show_count:]
        recent_lines = []
        for m in recent:
            diff = int(now - m.timestamp)
            if diff <= 10:
                time_label = "刚刚"
            elif diff < 60:
                time_label = f"{diff}秒前"
            elif diff < 3600:
                time_label = f"{diff // 60}分钟前"
            else:
                time_label = f"{diff // 3600}小时前"
            text = m.text[: self._recent_max_char_length] if m.text else ""
            if m.has_image and not text:
                text = "[图片]"
            elif m.has_image:
                text = f"[图片] {text}"
            at_mark = "【@你】" if m.is_at_me else ""
            recent_lines.append(
                f"[{time_label}] {at_mark}{m.sender_name}: {text}"
            )
        result = summary_line
        if signals:
            result += "\n" + ", ".join(signals)
        result += "\n最近几条:\n" + "\n".join(recent_lines)
        return result

    def compress_simple(self, buffer: List[StandardMessage]) -> str:
        if not buffer:
            return ""
        lines = []
        for m in buffer[-10:]:
            text = m.text[:50] if m.text else ""
            lines.append(f"{m.sender_name}: {text}")
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "recent_show_count": self._recent_show_count,
            "recent_max_char_length": self._recent_max_char_length,
        }


_buffer_compressor: Optional[BufferCompressor] = None


def get_buffer_compressor(config_engine=None) -> BufferCompressor:
    global _buffer_compressor
    if _buffer_compressor is None:
        _buffer_compressor = BufferCompressor(config_engine)
    return _buffer_compressor

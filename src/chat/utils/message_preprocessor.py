import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from src.common.logger import get_logger
from src.common.constants import (
    REPETITION_THRESHOLD,
    MAX_EMOJI_COUNT,
    MESSAGE_MAX_LENGTH,
    MESSAGE_MERGE_WINDOW,
    MESSAGE_SHORT_LENGTH,
)

logger = get_logger("消息预处理")


@dataclass
class PreprocessedMessage:
    """预处理后的消息"""
    original_content: str
    processed_content: str
    user_id: str
    user_name: str
    timestamp: float
    message_id: str
    is_filtered: bool = False
    filter_reason: str = ""
    merged_count: int = 1


class MessagePreprocessor:
    """消息预处理器：去噪、合并、归一化"""
    def __init__(self):
        self._emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        self._noise_patterns = [
            (r"^[\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]+$", "纯表情包"),
            (r"^[！？。，,.！？\s]+$", "纯标点符号"),
            (r"^[\d\s]+$", "纯数字"),
            (r"^[a-zA-Z\s]{1,2}$", "过短英文"),
        ]

    def denoise(self, content: str) -> Tuple[str, bool, str]:
        """去噪过滤，返回(处理后内容, 是否过滤, 过滤原因)"""
        if not content or not content.strip():
            return "", True, "空内容"
        text = content.strip()
        for pattern, reason in self._noise_patterns:
            if re.match(pattern, text):
                return "", True, reason
        if len(text) < 2:
            return "", True, "过短消息"
        text = self._collapse_repetitions(text)
        text = self._remove_excessive_emoji(text)
        return text, False, ""

    def _collapse_repetitions(self, text: str) -> str:
        """压缩重复字符（哈哈哈... -> 哈×N）"""
        def replace_repetition(match):
            char = match.group(1)
            count = len(match.group(0))
            if count > REPETITION_THRESHOLD:
                return f"{char}×{count}"
            return match.group(0)
        text = re.sub(r"(.)\1{4,}", replace_repetition, text)
        return text

    def _remove_excessive_emoji(self, text: str, max_emoji: int = MAX_EMOJI_COUNT) -> str:
        """移除过多表情符号"""
        emoji_count = len(self._emoji_pattern.findall(text))
        if emoji_count <= max_emoji:
            return text
        cleaned = self._emoji_pattern.sub("", text)
        return cleaned.strip() if cleaned.strip() else text

    def normalize(self, content: str) -> str:
        """归一化处理，LLM 友好格式"""
        if not content:
            return ""
        text = content.strip()
        text = re.sub(r"@\w+\s*", "", text)
        text = re.sub(r"\[图片:.*?\]", "[图]", text)
        text = re.sub(r"\[链接:.*?\]", "[链接]", text)
        text = re.sub(r"\[表情:.*?\]", "[表情]", text)
        text = re.sub(r"\[CQ:.*?\]", "[消息元素]", text)
        text = re.sub(r"\s{3,}", "  ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) > MESSAGE_MAX_LENGTH:
            text = text[:MESSAGE_MAX_LENGTH - 3] + "..."
        return text.strip()

    def merge_messages(
        self,
        messages: List[PreprocessedMessage],
        time_window: float = MESSAGE_MERGE_WINDOW,
    ) -> List[PreprocessedMessage]:
        """合并同一用户的连续短消息"""
        if not messages:
            return []
        merged = []
        current_batch: List[PreprocessedMessage] = []
        for msg in messages:
            if msg.is_filtered:
                continue
            if (
                current_batch
                and msg.user_id == current_batch[-1].user_id
                and msg.timestamp - current_batch[-1].timestamp <= time_window
                and len(msg.processed_content) < MESSAGE_SHORT_LENGTH
            ):
                current_batch.append(msg)
            else:
                if current_batch:
                    merged.append(self._combine_batch(current_batch))
                current_batch = [msg]
        if current_batch:
            merged.append(self._combine_batch(current_batch))
        return merged

    def _combine_batch(self, batch: List[PreprocessedMessage]) -> PreprocessedMessage:
        """合并一批消息"""
        if len(batch) == 1:
            return batch[0]
        combined_content = " ".join(m.processed_content for m in batch)
        return PreprocessedMessage(
            original_content="\n".join(m.original_content for m in batch),
            processed_content=combined_content,
            user_id=batch[0].user_id,
            user_name=batch[0].user_name,
            timestamp=batch[0].timestamp,
            message_id=batch[0].message_id,
            merged_count=len(batch),
        )

    def preprocess_pipeline(
        self,
        messages: List[dict],
        enable_denoise: bool = True,
        enable_normalize: bool = True,
        enable_merge: bool = True,
    ) -> List[PreprocessedMessage]:
        """完整预处理管道"""
        preprocessed = []
        for msg_dict in messages:
            content = str(msg_dict.get("content", "")).strip()
            user_id = str(msg_dict.get("user_id", ""))
            user_name = str(msg_dict.get("user_name", ""))
            timestamp = float(msg_dict.get("timestamp", time.time()))
            message_id = str(msg_dict.get("message_id", ""))
            processed_content = content
            is_filtered = False
            filter_reason = ""
            if enable_denoise:
                processed_content, is_filtered, filter_reason = self.denoise(content)
            if not is_filtered and enable_normalize:
                processed_content = self.normalize(processed_content)
            preprocessed.append(
                PreprocessedMessage(
                    original_content=content,
                    processed_content=processed_content,
                    user_id=user_id,
                    user_name=user_name,
                    timestamp=timestamp,
                    message_id=message_id,
                    is_filtered=is_filtered,
                    filter_reason=filter_reason,
                )
            )
        if enable_merge:
            preprocessed = self.merge_messages(preprocessed)
        valid_messages = [m for m in preprocessed if not m.is_filtered]
        filtered_count = len(preprocessed) - len(valid_messages)
        if filtered_count > 0:
            logger.debug(
                f"[消息预处理] 过滤 {filtered_count} 条无效消息，保留 {len(valid_messages)} 条"
            )
        return valid_messages


_message_preprocessor: Optional[MessagePreprocessor] = None


def get_message_preprocessor() -> MessagePreprocessor:
    """获取消息预处理器单例"""
    global _message_preprocessor
    if _message_preprocessor is None:
        _message_preprocessor = MessagePreprocessor()
    return _message_preprocessor


def preprocess_messages(
    messages: List[dict],
    enable_denoise: bool = True,
    enable_normalize: bool = True,
    enable_merge: bool = True,
) -> List[PreprocessedMessage]:
    """便捷函数：预处理消息列表"""
    preprocessor = get_message_preprocessor()
    return preprocessor.preprocess_pipeline(
        messages,
        enable_denoise=enable_denoise,
        enable_normalize=enable_normalize,
        enable_merge=enable_merge,
    )

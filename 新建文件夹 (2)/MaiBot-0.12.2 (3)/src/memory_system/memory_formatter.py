import time
import re
from typing import List, Dict, Optional
from datetime import datetime
from src.memory_system.context_saver import get_time_label, TimeLabel
from src.common.logger import get_logger

logger = get_logger("memory_formatter")


class MemoryFormatter:
    def __init__(self, max_display: int = 10):
        self.max_display = max_display

    def _clean_memory_content(self, content: str) -> str:
        if not content:
            return ""
        cleaned = content
        cleaned = re.sub(r'\[图片URLs?\]\s*:.*?(?=\n|$)', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'\[语音URLs?\]\s*:.*?(?=\n|$)', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'\[视频URLs?\]\s*:.*?(?=\n|$)', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'\[图片\]\s*:.*?(?=\n|$)', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'\[语音\]\s*:.*?(?=\n|$)', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'\[视频\]\s*:.*?(?=\n|$)', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'图片大小[：:]\s*\d+[KMG]?B?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'文件大小[：:]\s*\d+[KMG]?B?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\d+\s*[xX×]\s*\d+\s*像素', '', cleaned)
        cleaned = re.sub(r'\d+\s*[KMG]?B?\s*\(.*?\)', '', cleaned)
        cleaned = re.sub(r'\{"tool"[^}]*\}', '', cleaned)
        cleaned = re.sub(r'\{"tool_call"[^}]*\}', '', cleaned)
        cleaned = re.sub(r'\{"tool_name"[^}]*\}', '', cleaned)
        cleaned = re.sub(r'\{"name"[^}]*"type"[^}]*\}', '', cleaned)
        cleaned = re.sub(r'图片识别[：:].*?(?=\n|$)', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'识别结果[：:].*?(?=\n|$)', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'图片分析[：:].*?(?=\n|$)', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'语音识别[：:].*?(?=\n|$)', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'语音转文字[：:].*?(?=\n|$)', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'参数[：:]\s*\{[^}]*\}', '', cleaned)
        cleaned = re.sub(r'结果[：:]\s*\{[^}]*\}', '', cleaned)
        cleaned = re.sub(r'params[：:]\s*\{[^}]*\}', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'result[：:]\s*\{[^}]*\}', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'https?://[^\s]+', '', cleaned)
        cleaned = re.sub(r'www\.[^\s]+', '', cleaned)
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = cleaned.strip()
        if not cleaned or len(cleaned.strip()) < 5:
            first_sentence = re.split(r'[。！？\n]', content)[0]
            if first_sentence and len(first_sentence) > 5:
                return first_sentence[:100]
            return content[:100] if content else ""
        return cleaned

    def format_memories(self, memories: List[Dict], stream_name: str = "当前群聊",
                        sender_name: str = "用户") -> str:
        if not memories:
            return ""
        time_groups: Dict[str, List[Dict]] = {}
        for mem in memories:
            timestamp = mem.get("timestamp") or mem.get("created_at") or time.time()
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
                except Exception:
                    timestamp = time.time()
            time_label, time_desc = get_time_label(timestamp)
            time_key = time_label.value
            if time_key not in time_groups:
                time_groups[time_key] = []
            mem_with_time = mem.copy()
            mem_with_time["timestamp"] = timestamp
            time_groups[time_key].append(mem_with_time)
        time_order = [
            "just_now", "recent", "this_hour", "today",
            "yesterday", "before_yesterday", "this_week", "earlier"
        ]
        displayed_count = 0
        memory_blocks = []
        for time_key in time_order:
            if time_key not in time_groups:
                continue
            if displayed_count >= self.max_display:
                break
            mems = time_groups[time_key]
            if not mems:
                continue
            first_mem = mems[0]
            first_timestamp = first_mem.get("timestamp", time.time())
            _, time_desc = get_time_label(first_timestamp)
            block_lines = []
            for mem in mems:
                if displayed_count >= self.max_display:
                    break
                content = mem.get("content", "")
                if not content:
                    continue
                cleaned_content = self._clean_memory_content(content)
                if not cleaned_content:
                    continue
                source = mem.get("source", "未知")
                user_id = mem.get("user_id", "")
                source_label = self._format_source_label(source, mem)
                display_sender = sender_name if user_id else "群聊"
                content_preview = self._truncate_content(cleaned_content, max_length=120)
                block_lines.append(f"  {display_sender}: {content_preview} [{source_label}]")
                displayed_count += 1
            if block_lines:
                memory_blocks.append(f"- 【{time_desc} ({stream_name})】\n" + "\n".join(block_lines))
        if not memory_blocks:
            return ""
        formatted_text = "\n\n".join(memory_blocks)
        return f"### 相关记忆\n\n{formatted_text}\n"

    def _format_source_label(self, source: str, mem: Dict) -> str:
        if source == "short_term":
            return "短期记忆"
        elif source == "hippocampus":
            tier = mem.get("tier", "")
            if tier:
                return f"海马体记忆({tier})"
            return "海马体记忆"
        elif source == "database":
            return "长期记忆"
        return source

    def _truncate_content(self, content: str, max_length: int = 120) -> str:
        if len(content) <= max_length:
            return content
        truncated = content[:max_length]
        last_punct = max(
            truncated.rfind("。"), truncated.rfind("？"),
            truncated.rfind("！"), truncated.rfind("."),
            truncated.rfind("?"), truncated.rfind("!")
        )
        if last_punct > max_length * 0.6:
            return truncated[:last_punct + 1] + "..."
        return truncated + "..."

    def format_memories_compact(self, memories: List[Dict], max_items: int = 5) -> str:
        if not memories:
            return ""
        sorted_memories = sorted(memories, key=lambda x: x.get("score", 0), reverse=True)[:max_items]
        lines = []
        for mem in sorted_memories:
            content = mem.get("content", "")
            if not content:
                continue
            cleaned_content = self._clean_memory_content(content)
            if not cleaned_content:
                continue
            source = mem.get("source", "未知")
            source_label = self._format_source_label(source, mem)
            content_preview = self._truncate_content(cleaned_content, max_length=80)
            lines.append(f"- {content_preview} [{source_label}]")
        if not lines:
            return ""
        return "### 相关记忆\n\n" + "\n".join(lines) + "\n"


_default_formatter = MemoryFormatter(max_display=10)


def format_memory_block(memories: List[Dict], stream_name: str = "当前群聊",
                        sender_name: str = "用户",
                        max_display: Optional[int] = None) -> str:
    if max_display is not None:
        formatter = MemoryFormatter(max_display=max_display)
    else:
        formatter = _default_formatter
    return formatter.format_memories(memories, stream_name, sender_name)

import time
from typing import Any, Dict, List, Optional
from uuid import uuid4
from src.common.database.slot_storage import (
    delete_slot,
    list_slot_keys,
    save_slot,
)
from src.config.config import global_config


class PlanReplyLogger:
    _PLAN_DIR = "plan_reply_log:plan"
    _REPLY_DIR = "plan_reply_log:reply"
    _TRIM_COUNT = 100

    @classmethod
    def _get_max_per_chat(cls) -> int:
        return getattr(global_config.chat, "plan_reply_log_max_per_chat", 1000)

    @classmethod
    def log_plan(
        cls,
        chat_id: str,
        prompt: str,
        reasoning: str,
        raw_output: Optional[str],
        raw_reasoning: Optional[str],
        actions: List[Any],
        timing: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "type": "plan",
            "chat_id": chat_id,
            "timestamp": time.time(),
            "prompt": prompt,
            "reasoning": reasoning,
            "raw_output": raw_output,
            "raw_reasoning": raw_reasoning,
            "actions": [cls._serialize_action(action) for action in actions],
            "timing": timing or {},
            "extra": cls._safe_data(extra),
        }
        cls._write_json(cls._PLAN_DIR, chat_id, payload)

    @classmethod
    def log_reply(
        cls,
        chat_id: str,
        prompt: str,
        output: Optional[str],
        processed_output: Optional[List[Any]],
        model: Optional[str],
        timing: Optional[Dict[str, Any]] = None,
        reasoning: Optional[str] = None,
        think_level: Optional[int] = None,
        error: Optional[str] = None,
        success: bool = True,
    ) -> None:
        payload = {
            "type": "reply",
            "chat_id": chat_id,
            "timestamp": time.time(),
            "prompt": prompt,
            "output": output,
            "processed_output": cls._safe_data(processed_output),
            "model": model,
            "reasoning": reasoning,
            "think_level": think_level,
            "timing": timing or {},
            "error": error if not success else None,
            "success": success,
        }
        cls._write_json(cls._REPLY_DIR, chat_id, payload)

    @classmethod
    def _write_json(
        cls, base_dir: str, chat_id: str, payload: Dict[str, Any]
    ) -> None:
        safe_chat_id = str(chat_id).replace("/", "_").replace("\\", "_")
        slot_key = f"{base_dir}:{safe_chat_id}:{int(time.time() * 1000)}_{uuid4().hex[:8]}"
        try:
            save_slot(slot_key, cls._safe_data(payload), ttl_days=30)
        finally:
            cls._trim_overflow(f"{base_dir}:{safe_chat_id}:")

    @classmethod
    def _trim_overflow(cls, prefix: str) -> None:
        keys = sorted(list_slot_keys(prefix))
        max_per_chat = cls._get_max_per_chat()
        if len(keys) <= max_per_chat:
            return
        for old_key in keys[: cls._TRIM_COUNT]:
            delete_slot(old_key)

    @classmethod
    def _serialize_action(cls, action: Any) -> Dict[str, Any]:
        message_info = None
        action_message = getattr(action, "action_message", None)
        if action_message:
            user_info = getattr(action_message, "user_info", None)
            message_info = {
                "message_id": getattr(action_message, "message_id", None),
                "user_id": (
                    getattr(user_info, "user_id", None) if user_info else None
                ),
                "platform": (
                    getattr(user_info, "platform", None) if user_info else None
                ),
                "text": getattr(action_message, "processed_plain_text", None),
            }
        return {
            "action_type": getattr(action, "action_type", None),
            "reasoning": getattr(action, "reasoning", None),
            "action_data": cls._safe_data(
                getattr(action, "action_data", None)
            ),
            "action_message": message_info,
            "available_actions": cls._safe_data(
                getattr(action, "available_actions", None)
            ),
            "action_reasoning": getattr(action, "action_reasoning", None),
        }

    @classmethod
    def _safe_data(cls, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): cls._safe_data(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._safe_data(v) for v in value]
        try:
            return str(value)
        except Exception:
            return None

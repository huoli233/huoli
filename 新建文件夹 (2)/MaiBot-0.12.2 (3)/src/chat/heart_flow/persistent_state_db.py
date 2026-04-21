import time
import json
from typing import Optional, Dict, Any, List
from src.common.logger import get_logger

logger = get_logger("persistent_state")


class PersistentStateDatabase:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._fatigue_cache: Dict[str, Dict[str, Any]] = {}
        self._ignore_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_state_cache: Dict[str, Dict[str, Any]] = {}
        self._emotion_cache: Dict[str, Dict[str, Any]] = {}
        self._psychology_cache: Dict[str, Dict[str, Any]] = {}
        self._engagement_cache: Dict[str, Dict[str, Any]] = {}
        self._system_index_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("持久化状态数据库初始化完成")

    def _get_fatigue_model(self):
        try:
            from src.common.database.database_model import FatigueState
            return FatigueState
        except Exception:
            return None

    def _get_heartflow_model(self):
        try:
            from src.common.database.database_model import HeartFlowState
            return HeartFlowState
        except Exception:
            return None

    def _get_memory_state_model(self):
        try:
            from src.common.database.database_model import MemoryState
            return MemoryState
        except Exception:
            return None

    def save_fatigue_state(self, stream_id: str, is_fatigued: bool,
                           fatigue_cooldown: float, fatigue_duration: float,
                           fatigue_reply_count: int, consecutive_replies: int,
                           last_reply_time: float) -> bool:
        try:
            now = time.time()
            FatigueState = self._get_fatigue_model()
            if FatigueState:
                existing = FatigueState.get_or_none(FatigueState.stream_id == stream_id)
                if existing:
                    existing.is_fatigued = is_fatigued
                    existing.fatigue_cooldown = fatigue_cooldown
                    existing.fatigue_duration = fatigue_duration
                    existing.fatigue_reply_count = fatigue_reply_count
                    existing.consecutive_replies = consecutive_replies
                    existing.last_reply_time = last_reply_time
                    existing.last_save_time = now
                    existing.updated_at = now
                    existing.save()
                else:
                    FatigueState.create(
                        stream_id=stream_id,
                        is_fatigued=is_fatigued,
                        fatigue_cooldown=fatigue_cooldown,
                        fatigue_duration=fatigue_duration,
                        fatigue_reply_count=fatigue_reply_count,
                        consecutive_replies=consecutive_replies,
                        last_reply_time=last_reply_time,
                        last_save_time=now,
                        updated_at=now,
                    )
            self._fatigue_cache[stream_id] = {
                "is_fatigued": is_fatigued, "fatigue_cooldown": fatigue_cooldown,
                "fatigue_duration": fatigue_duration,
                "fatigue_reply_count": fatigue_reply_count,
                "consecutive_replies": consecutive_replies,
                "last_reply_time": last_reply_time,
                "last_save_time": now
            }
            return True
        except Exception as e:
            logger.error(f"保存疲劳状态失败: {e}")
            return False

    def load_fatigue_state(self, stream_id: str) -> Optional[Dict[str, Any]]:
        try:
            if stream_id in self._fatigue_cache:
                cached = self._fatigue_cache[stream_id]
                now = time.time()
                last_save_time = cached.get("last_save_time", 0.0)
                offline_duration = now - last_save_time if last_save_time > 0 else 0.0
                is_fatigued = cached.get("is_fatigued", False)
                fatigue_cooldown = cached.get("fatigue_cooldown", 0.0)
                fatigue_duration = cached.get("fatigue_duration", 0.0)
                last_reply_time = cached.get("last_reply_time", 0.0)
                if is_fatigued and fatigue_duration > 0 and fatigue_cooldown > 0:
                    elapsed_since_fatigue = now - fatigue_cooldown
                    if elapsed_since_fatigue >= fatigue_duration:
                        is_fatigued = False
                        fatigue_duration = 0.0
                        fatigue_cooldown = 0.0
                if last_reply_time > 0 and (now - last_reply_time) > 7200:
                    cached["consecutive_replies"] = 0
                return {
                    "is_fatigued": is_fatigued, "fatigue_cooldown": fatigue_cooldown,
                    "fatigue_duration": fatigue_duration,
                    "fatigue_reply_count": cached.get("fatigue_reply_count", 0),
                    "consecutive_replies": cached.get("consecutive_replies", 0),
                    "last_reply_time": last_reply_time,
                    "offline_duration": offline_duration
                }
            FatigueState = self._get_fatigue_model()
            if FatigueState:
                try:
                    record = FatigueState.get(
                        (FatigueState.stream_id == stream_id) &
                        (FatigueState.user_id == "system")
                    )
                    now = time.time()
                    offline_duration = now - record.last_update if record.last_update > 0 else 0.0
                    is_fatigued = record.mental_fatigue > 30.0
                    return {
                        "is_fatigued": is_fatigued, "fatigue_cooldown": 0.0,
                        "fatigue_duration": 0.0, "fatigue_reply_count": 0,
                        "consecutive_replies": 0, "last_reply_time": 0.0,
                        "offline_duration": offline_duration
                    }
                except Exception:
                    pass
            return None
        except Exception as e:
            logger.error(f"加载疲劳状态失败: {e}")
            return None

    def save_ignore_state(self, stream_id: str, is_ignoring: bool,
                          ignore_until_time: float) -> bool:
        try:
            now = time.time()
            self._ignore_cache[stream_id] = {
                "is_ignoring": is_ignoring,
                "ignore_until_time": ignore_until_time,
                "last_save_time": now
            }
            return True
        except Exception as e:
            logger.error(f"保存忽略状态失败: {e}")
            return False

    def load_ignore_state(self, stream_id: str) -> Optional[Dict[str, Any]]:
        try:
            if stream_id not in self._ignore_cache:
                return None
            cached = self._ignore_cache[stream_id]
            now = time.time()
            last_save_time = cached.get("last_save_time", 0.0)
            offline_duration = now - last_save_time if last_save_time > 0 else 0.0
            is_ignoring = cached.get("is_ignoring", False)
            ignore_until_time = cached.get("ignore_until_time", 0.0)
            if is_ignoring and ignore_until_time > 0 and now >= ignore_until_time:
                is_ignoring = False
                ignore_until_time = 0.0
            return {
                "is_ignoring": is_ignoring,
                "ignore_until_time": ignore_until_time,
                "offline_duration": offline_duration
            }
        except Exception as e:
            logger.error(f"加载忽略状态失败: {e}")
            return None

    def save_memory_state(self, stream_id: str) -> bool:
        try:
            now = time.time()
            MemoryState = self._get_memory_state_model()
            if MemoryState:
                MemoryState.insert(
                    stream_id=stream_id, last_save_time=now, total_memories=0
                ).on_conflict(
                    conflict_target=[MemoryState.stream_id],
                    update={MemoryState.last_save_time: now}
                ).execute()
            self._memory_state_cache[stream_id] = {"last_save_time": now}
            return True
        except Exception as e:
            logger.error(f"保存记忆库状态失败: {e}")
            return False

    def load_memory_state(self, stream_id: str) -> Optional[Dict[str, Any]]:
        try:
            if stream_id in self._memory_state_cache:
                cached = self._memory_state_cache[stream_id]
                now = time.time()
                last_save_time = cached.get("last_save_time", 0.0)
                offline_duration = now - last_save_time if last_save_time > 0 else 0.0
                return {"last_save_time": last_save_time, "offline_duration": offline_duration}
            MemoryState = self._get_memory_state_model()
            if MemoryState:
                try:
                    record = MemoryState.get(MemoryState.stream_id == stream_id)
                    now = time.time()
                    offline_duration = now - record.last_save_time if record.last_save_time > 0 else 0.0
                    return {"last_save_time": record.last_save_time, "offline_duration": offline_duration}
                except Exception:
                    pass
            return None
        except Exception as e:
            logger.error(f"加载记忆库状态失败: {e}")
            return None

    def save_emotion_state(self, stream_id: str, user_id: str,
                           affection: float, annoyance: float,
                           trust_score: float, trauma_score: float,
                           last_interaction: float,
                           has_trauma_mark: bool = False,
                           trauma_mark_level: int = 0,
                           is_protected: bool = False,
                           protect_until: float = 0.0) -> bool:
        try:
            now = time.time()
            affection = max(-100.0, min(100.0, float(affection)))
            trust_score = max(-100.0, min(100.0, float(trust_score)))
            annoyance = max(0.0, min(100.0, float(annoyance)))
            trauma_score = max(0.0, min(10.0, float(trauma_score)))
            trauma_mark_level = min(3, max(0, int(trauma_mark_level)))
            cache_key = f"{stream_id}:{user_id}"
            self._emotion_cache[cache_key] = {
                "stream_id": stream_id, "user_id": user_id,
                "affection": affection, "annoyance": annoyance,
                "trust_score": trust_score, "trauma_score": trauma_score,
                "has_trauma_mark": has_trauma_mark,
                "trauma_mark_level": trauma_mark_level,
                "is_protected": is_protected, "protect_until": protect_until,
                "last_interaction": last_interaction, "last_save_time": now
            }
            return True
        except Exception as e:
            logger.error(f"保存情感状态失败: {e}")
            return False

    def load_emotion_state(self, stream_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            cache_key = f"{stream_id}:{user_id}"
            if cache_key in self._emotion_cache:
                cached = self._emotion_cache[cache_key]
                now = time.time()
                last_save_time = cached.get("last_save_time", 0.0)
                offline_duration = now - last_save_time if last_save_time > 0 else 0.0
                result = cached.copy()
                result["offline_duration"] = offline_duration
                return result
            return None
        except Exception as e:
            logger.error(f"加载情感状态失败: {e}")
            return None

    def get_all_emotion_states(self, stream_id: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            now = time.time()
            result = []
            for cache_key, cached in self._emotion_cache.items():
                if stream_id and cached.get("stream_id") != stream_id:
                    continue
                last_save_time = cached.get("last_save_time", 0.0)
                offline_duration = now - last_save_time if last_save_time > 0 else 0.0
                entry = cached.copy()
                entry["offline_duration"] = offline_duration
                result.append(entry)
            return result
        except Exception as e:
            logger.error(f"获取情感状态列表失败: {e}")
            return []

    def save_psychology_state(self, stream_id: str, user_id: str,
                               psych_state_data: Dict[str, Any]) -> bool:
        try:
            now = time.time()
            cache_key = f"{stream_id}:{user_id}"
            psych_state_data["last_save_time"] = now
            self._psychology_cache[cache_key] = psych_state_data
            return True
        except Exception as e:
            logger.debug(f"保存心理状态失败: {e}")
            return False

    def get_psychology_state(self, stream_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            cache_key = f"{stream_id}:{user_id}"
            if cache_key in self._psychology_cache:
                return self._psychology_cache[cache_key]
            return None
        except Exception as e:
            logger.debug(f"获取心理状态失败: {e}")
            return None

    def get_all_psychology_states(self, stream_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        try:
            results: Dict[str, Dict[str, Any]] = {}
            for cache_key, data in self._psychology_cache.items():
                parts = cache_key.split(":", 1)
                if len(parts) != 2:
                    continue
                sid, uid = parts
                if stream_id and sid != stream_id:
                    continue
                results[uid] = data
            return results
        except Exception as e:
            logger.debug(f"获取所有心理状态失败: {e}")
            return {}

    def delete_psychology_state(self, stream_id: str, user_id: str) -> bool:
        try:
            cache_key = f"{stream_id}:{user_id}"
            self._psychology_cache.pop(cache_key, None)
            return True
        except Exception as e:
            logger.debug(f"删除心理状态失败: {e}")
            return False

    def save_engagement_state(self, stream_id: str, data: dict) -> bool:
        try:
            now = time.time()
            data["last_save_time"] = now
            self._engagement_cache[stream_id] = data
            return True
        except Exception as e:
            logger.debug(f"保存发言能量状态失败: {e}")
            return False

    def load_engagement_state(self, stream_id: str) -> Optional[dict]:
        try:
            if stream_id in self._engagement_cache:
                return self._engagement_cache[stream_id]
            return None
        except Exception as e:
            logger.debug(f"加载发言能量状态失败: {e}")
            return None

    def load_engagement_states(self) -> Dict[str, dict]:
        return dict(self._engagement_cache)

    def save_system_index(self, system_name: str,
                          last_calibration_time: Optional[float] = None,
                          last_organize_time: Optional[float] = None,
                          **kwargs) -> bool:
        try:
            now = time.time()
            existing = self._system_index_cache.get(system_name, {})
            calibration_count = existing.get("calibration_count", 0)
            organize_count = existing.get("organize_count", 0)
            if last_calibration_time and last_calibration_time > existing.get("last_calibration_time", 0.0):
                calibration_count += 1
            if last_organize_time and last_organize_time > existing.get("last_organize_time", 0.0):
                organize_count += 1
            entry = {
                "system_name": system_name,
                "last_calibration_time": last_calibration_time or existing.get("last_calibration_time", 0.0),
                "last_organize_time": last_organize_time or existing.get("last_organize_time", 0.0),
                "calibration_count": calibration_count,
                "organize_count": organize_count,
                "updated_at": now
            }
            entry.update(kwargs)
            self._system_index_cache[system_name] = entry
            return True
        except Exception as e:
            logger.error(f"保存系统索引失败: {e}")
            return False

    def get_system_index(self, system_name: str, key: Optional[str] = None) -> Any:
        try:
            if system_name not in self._system_index_cache:
                return None
            data = self._system_index_cache[system_name]
            if key:
                return data.get(key)
            return data
        except Exception as e:
            logger.error(f"获取系统索引失败: {e}")
            return None

    def get_all_system_indexes(self) -> List[Dict[str, Any]]:
        return list(self._system_index_cache.values())

    def save_intimacy_state(self, stream_id: str, channel_id: str,
                            interaction_warmth: float,
                            last_interaction_time: float) -> bool:
        try:
            now = time.time()
            self._emotion_cache[f"intimacy:{stream_id}"] = {
                "stream_id": stream_id, "channel_id": channel_id,
                "interaction_warmth": interaction_warmth,
                "last_interaction_time": last_interaction_time,
                "last_save_time": now
            }
            return True
        except Exception as e:
            logger.error(f"保存亲密度状态失败: {e}")
            return False

    def load_intimacy_state(self, stream_id: str) -> Optional[Dict[str, Any]]:
        try:
            cache_key = f"intimacy:{stream_id}"
            if cache_key in self._emotion_cache:
                cached = self._emotion_cache[cache_key]
                now = time.time()
                last_save_time = cached.get("last_save_time", 0.0)
                offline_duration = now - last_save_time if last_save_time > 0 else 0.0
                result = cached.copy()
                result["offline_duration"] = offline_duration
                return result
            return None
        except Exception as e:
            logger.error(f"加载亲密度状态失败: {e}")
            return None

    def get_all_intimacy_states(self, stream_id: Optional[str] = None) -> List[Dict[str, Any]]:
        result = []
        now = time.time()
        for cache_key, cached in self._emotion_cache.items():
            if not cache_key.startswith("intimacy:"):
                continue
            if stream_id and cached.get("stream_id") != stream_id:
                continue
            last_save_time = cached.get("last_save_time", 0.0)
            offline_duration = now - last_save_time if last_save_time > 0 else 0.0
            entry = cached.copy()
            entry["offline_duration"] = offline_duration
            result.append(entry)
        return result

    def delete_stream_state(self, stream_id: str) -> bool:
        try:
            self._fatigue_cache.pop(stream_id, None)
            self._ignore_cache.pop(stream_id, None)
            self._memory_state_cache.pop(stream_id, None)
            self._engagement_cache.pop(stream_id, None)
            keys_to_remove = [k for k in self._emotion_cache if k.startswith(f"{stream_id}:") or k == f"intimacy:{stream_id}"]
            for k in keys_to_remove:
                del self._emotion_cache[k]
            keys_to_remove = [k for k in self._psychology_cache if k.startswith(f"{stream_id}:")]
            for k in keys_to_remove:
                del self._psychology_cache[k]
            return True
        except Exception as e:
            logger.error(f"删除频道状态失败: {e}")
            return False

    def delete_channel_state(self, channel_id: str) -> bool:
        return self.delete_stream_state(channel_id)

    def save_tool_usage_state(self, channel_id: str, tool_name: str,
                               usage_count: int, last_usage_time: float,
                               fatigue_level: float) -> bool:
        try:
            cache_key = f"tool:{channel_id}:{tool_name}"
            self._engagement_cache[cache_key] = {
                "usage_count": usage_count,
                "last_usage_time": last_usage_time,
                "fatigue_level": fatigue_level,
                "last_save_time": time.time(),
            }
            return True
        except Exception as e:
            logger.error(f"保存工具使用状态失败: {e}")
            return False

    def load_tool_usage_states(self, channel_id: str) -> Dict[str, Dict[str, Any]]:
        try:
            prefix = f"tool:{channel_id}:"
            result = {}
            now = time.time()
            for key, data in self._engagement_cache.items():
                if key.startswith(prefix):
                    tool_name = key[len(prefix):]
                    last_save = data.get("last_save_time", 0.0)
                    offline_duration = now - last_save if last_save > 0 else 0.0
                    fatigue = data.get("fatigue_level", 0.0)
                    if offline_duration > 1800:
                        fatigue = max(0.0, fatigue * 0.5)
                    result[tool_name] = {
                        "usage_count": data.get("usage_count", 0),
                        "last_usage_time": data.get("last_usage_time", 0.0),
                        "fatigue_level": fatigue,
                        "offline_duration": offline_duration,
                    }
            return result
        except Exception as e:
            logger.error(f"加载工具使用状态失败: {e}")
            return {}

    def save_expression_learning_state(self, channel_id: str, state_data: Dict[str, Any]) -> bool:
        try:
            cache_key = f"expr_learn:{channel_id}"
            state_data["last_save_time"] = time.time()
            self._engagement_cache[cache_key] = state_data
            return True
        except Exception as e:
            logger.error(f"保存表达学习状态失败: {e}")
            return False

    def load_expression_learning_state(self, channel_id: str) -> Optional[Dict[str, Any]]:
        try:
            cache_key = f"expr_learn:{channel_id}"
            data = self._engagement_cache.get(cache_key)
            if not data:
                return None
            now = time.time()
            last_save = data.get("last_save_time", 0.0)
            data["offline_duration"] = now - last_save if last_save > 0 else 0.0
            return data
        except Exception as e:
            logger.error(f"加载表达学习状态失败: {e}")
            return None

    def save_jargon_learning_state(self, channel_id: str, state_data: Dict[str, Any]) -> bool:
        try:
            cache_key = f"jargon_learn:{channel_id}"
            state_data["last_save_time"] = time.time()
            self._engagement_cache[cache_key] = state_data
            return True
        except Exception as e:
            logger.error(f"保存黑话学习状态失败: {e}")
            return False

    def load_jargon_learning_state(self, channel_id: str) -> Optional[Dict[str, Any]]:
        try:
            cache_key = f"jargon_learn:{channel_id}"
            data = self._engagement_cache.get(cache_key)
            if not data:
                return None
            now = time.time()
            last_save = data.get("last_save_time", 0.0)
            data["offline_duration"] = now - last_save if last_save > 0 else 0.0
            return data
        except Exception as e:
            logger.error(f"加载黑话学习状态失败: {e}")
            return None

    def close(self):
        logger.info("持久化状态数据库已关闭")


_persistent_state_db: Optional[PersistentStateDatabase] = None


def get_persistent_state_db() -> PersistentStateDatabase:
    global _persistent_state_db
    if _persistent_state_db is None:
        _persistent_state_db = PersistentStateDatabase()
    return _persistent_state_db

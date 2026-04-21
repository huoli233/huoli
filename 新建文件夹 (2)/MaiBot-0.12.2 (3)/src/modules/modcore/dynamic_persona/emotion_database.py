import time
import json
from typing import Optional, Dict, List, Any
from src.common.logger import get_logger

logger = get_logger("emotion_db")


class EmotionDatabase:
    def __init__(self, stream_id: str):
        self.stream_id = stream_id

    def _get_state_model(self):
        try:
            from src.common.database.database_model import StreamStateData
            return StreamStateData
        except Exception as e:
            logger.debug(f"StreamStateData模型未初始化: {e}")
            return None

    def _get_trauma_model(self):
        try:
            from src.common.database.database_model import TraumaRecord
            return TraumaRecord
        except Exception as e:
            logger.debug(f"TraumaRecord模型未初始化: {e}")
            return None

    def _state_key(self, user_id: str) -> str:
        return f"emotion:{user_id}"

    def save_user_state(self, user_id: str, state_data: Dict) -> bool:
        try:
            Model = self._get_state_model()
            if not Model:
                return False
            now = time.time()
            save_data = {
                "user_id": user_id,
                "affection": max(-100.0, min(100.0, float(state_data.get("affection", 0.0)))),
                "trust_score": max(-100.0, min(100.0, float(state_data.get("trust_score", 0.0)))),
                "annoyance": max(0.0, min(100.0, float(state_data.get("annoyance", 0.0)))),
                "relationship": state_data.get("relationship", "陌生人"),
                "last_interaction": state_data.get("last_interaction", now),
                "interaction_count": state_data.get("interaction_count", 0),
                "trauma_score": max(0.0, min(10.0, float(state_data.get("trauma_score", 0.0)))),
                "baseline_affection": state_data.get("baseline_affection", 0.0),
                "is_blocked": state_data.get("is_blocked", False),
                "block_until": state_data.get("block_until", 0.0),
                "block_reason": state_data.get("block_reason", ""),
                "has_trauma_mark": state_data.get("has_trauma_mark", False),
                "trauma_mark_level": state_data.get("trauma_mark_level", 0),
            }
            key = self._state_key(user_id)
            record, created = Model.get_or_create(
                stream_id=self.stream_id,
                state_key=key,
                defaults={
                    "state_value": json.dumps(save_data, ensure_ascii=False),
                    "updated_at": now,
                }
            )
            if not created:
                record.state_value = json.dumps(save_data, ensure_ascii=False)
                record.updated_at = now
                record.save()
            trauma_score = save_data["trauma_score"]
            if trauma_score > 0:
                self._save_trauma_state(user_id, state_data, trauma_score, now)
            return True
        except Exception as e:
            logger.error(f"保存用户状态失败: {e}")
            return False

    def _save_trauma_state(self, user_id: str, state_data: Dict, trauma_score: float, now: float):
        TraumaRecord = self._get_trauma_model()
        if not TraumaRecord:
            return
        try:
            record, created = TraumaRecord.get_or_create(
                stream_id=self.stream_id,
                user_id=user_id,
                defaults={
                    "trauma_type": "interaction",
                    "trigger_words": json.dumps(state_data.get("trauma_history", []), ensure_ascii=False),
                    "severity": trauma_score,
                    "has_trauma_mark": state_data.get("has_trauma_mark", False),
                    "trauma_mark_level": state_data.get("trauma_mark_level", 0),
                    "is_protected": state_data.get("is_protected", False),
                    "protect_until": state_data.get("protect_until", 0.0),
                    "created_at": now,
                    "updated_at": now,
                }
            )
            if not created:
                record.severity = trauma_score
                record.trigger_words = json.dumps(state_data.get("trauma_history", []), ensure_ascii=False)
                record.has_trauma_mark = state_data.get("has_trauma_mark", False)
                record.trauma_mark_level = state_data.get("trauma_mark_level", 0)
                record.is_protected = state_data.get("is_protected", False)
                record.protect_until = state_data.get("protect_until", 0.0)
                record.updated_at = now
                record.save()
        except Exception as e:
            logger.error(f"保存创伤状态失败: {e}")

    def load_user_state(self, user_id: str) -> Optional[Dict]:
        try:
            Model = self._get_state_model()
            if not Model:
                return None
            key = self._state_key(user_id)
            record = Model.get_or_none(
                (Model.stream_id == self.stream_id) &
                (Model.state_key == key)
            )
            if not record:
                return None
            data = json.loads(record.state_value)
            data["stream_id"] = self.stream_id
            data.setdefault("trauma_score", 0.0)
            data.setdefault("trauma_history", [])
            TraumaRecord = self._get_trauma_model()
            if TraumaRecord:
                trauma = TraumaRecord.get_or_none(
                    (TraumaRecord.stream_id == self.stream_id) &
                    (TraumaRecord.user_id == user_id)
                )
                if trauma:
                    data["trauma_score"] = trauma.severity
                    data["trauma_history"] = json.loads(trauma.trigger_words or "[]")
                    data["has_trauma_mark"] = trauma.has_trauma_mark
                    data["trauma_mark_level"] = trauma.trauma_mark_level
                    data["is_protected"] = trauma.is_protected
                    data["protect_until"] = trauma.protect_until
            return data
        except Exception as e:
            logger.error(f"加载用户状态失败: {e}")
            return None

    def load_all_states(self) -> Dict[str, Dict]:
        try:
            Model = self._get_state_model()
            if not Model:
                return {}
            prefix = "emotion:"
            records = Model.select().where(
                (Model.stream_id == self.stream_id) &
                (Model.state_key.startswith(prefix))
            )
            result = {}
            for record in records:
                try:
                    data = json.loads(record.state_value)
                    uid = data.get("user_id", record.state_key.replace(prefix, ""))
                    data["stream_id"] = self.stream_id
                    data.setdefault("trauma_score", 0.0)
                    data.setdefault("trauma_history", [])
                    result[uid] = data
                except (json.JSONDecodeError, KeyError):
                    continue
            TraumaRecord = self._get_trauma_model()
            if TraumaRecord:
                traumas = TraumaRecord.select().where(
                    TraumaRecord.stream_id == self.stream_id
                )
                for trauma in traumas:
                    uid = trauma.user_id
                    if uid in result:
                        result[uid]["trauma_score"] = trauma.severity
                        result[uid]["trauma_history"] = json.loads(trauma.trigger_words or "[]")
            logger.debug(f"加载了 {len(result)} 个用户状态")
            return result
        except Exception as e:
            logger.error(f"批量加载失败: {e}")
            return {}

    def delete_user_state(self, user_id: str) -> bool:
        try:
            Model = self._get_state_model()
            if Model:
                key = self._state_key(user_id)
                Model.delete().where(
                    (Model.stream_id == self.stream_id) &
                    (Model.state_key == key)
                ).execute()
            TraumaRecord = self._get_trauma_model()
            if TraumaRecord:
                TraumaRecord.delete().where(
                    (TraumaRecord.stream_id == self.stream_id) &
                    (TraumaRecord.user_id == user_id)
                ).execute()
            return True
        except Exception as e:
            logger.error(f"删除用户状态失败: {e}")
            return False

    def get_statistics(self) -> Dict:
        try:
            all_states = self.load_all_states()
            if not all_states:
                return {"total_users": 0, "avg_affection": 0.0}
            total = len(all_states)
            avg_aff = sum(s.get("affection", 0.0) for s in all_states.values()) / total
            return {"total_users": total, "avg_affection": avg_aff}
        except Exception:
            return {"total_users": 0, "avg_affection": 0.0}

    def save_persona(self, persona_data: Dict) -> bool:
        try:
            from src.common.database.database_model import DynamicPersonaData
            from src.common.database.database import db
            db.connect(reuse_if_open=True)
            now = time.time()
            traits_str = json.dumps(persona_data.get("traits", []), ensure_ascii=False)
            extra_str = json.dumps(persona_data.get("extra", {}), ensure_ascii=False)
            pid = persona_data.get("persona_id", "")
            record, created = DynamicPersonaData.get_or_create(
                persona_id=pid,
                defaults={
                    "stream_id": self.stream_id,
                    "name": persona_data.get("name", ""),
                    "tone": persona_data.get("tone", ""),
                    "traits": traits_str,
                    "speech_style": persona_data.get("speech_style", ""),
                    "created_at": persona_data.get("created_at", now),
                    "last_used": persona_data.get("last_used", now),
                    "use_count": persona_data.get("use_count", 0),
                    "source": persona_data.get("source", "generated"),
                    "extra": extra_str,
                    "updated_at": now,
                }
            )
            if not created:
                record.name = persona_data.get("name", record.name)
                record.tone = persona_data.get("tone", record.tone)
                record.traits = traits_str
                record.speech_style = persona_data.get("speech_style", record.speech_style)
                record.last_used = persona_data.get("last_used", now)
                record.use_count = persona_data.get("use_count", record.use_count)
                record.extra = extra_str
                record.updated_at = now
                record.save()
            return True
        except Exception as e:
            logger.error(f"保存人格失败: {e}")
            return False

    def load_all_personas(self) -> List[Dict]:
        try:
            from src.common.database.database_model import DynamicPersonaData
            from src.common.database.database import db
            db.connect(reuse_if_open=True)
            records = DynamicPersonaData.select().where(
                DynamicPersonaData.stream_id == self.stream_id
            )
            result = []
            for rec in records:
                result.append({
                    "persona_id": rec.persona_id,
                    "stream_id": rec.stream_id,
                    "name": rec.name,
                    "tone": rec.tone,
                    "traits": json.loads(rec.traits or "[]"),
                    "speech_style": rec.speech_style,
                    "created_at": rec.created_at,
                    "last_used": rec.last_used,
                    "use_count": rec.use_count,
                    "source": rec.source,
                    "extra": json.loads(rec.extra or "{}"),
                    "updated_at": rec.updated_at,
                })
            return result
        except Exception as e:
            logger.error(f"加载人格失败: {e}")
            return []

    def delete_persona(self, persona_id: str) -> bool:
        try:
            from src.common.database.database_model import DynamicPersonaData
            from src.common.database.database import db
            db.connect(reuse_if_open=True)
            DynamicPersonaData.delete().where(
                DynamicPersonaData.persona_id == persona_id
            ).execute()
            return True
        except Exception as e:
            logger.error(f"删除人格失败: {e}")
            return False

    def save_relationship_memory(self, user_id: str, memory_type: str,
                                  content: str, sentiment: str = "neutral") -> bool:
        try:
            cache_key = f"rel_mem:{user_id}:{memory_type}:{hash(content)}"
            self._cache[cache_key] = {
                "user_id": user_id, "stream_id": self.stream_id,
                "memory_type": memory_type, "content": content[:500],
                "sentiment": sentiment, "created_at": time.time()
            }
            return True
        except Exception as e:
            logger.error(f"保存关系记忆失败: {e}")
            return False

    def load_relationship_memories(self, user_id: str) -> List[Dict]:
        try:
            prefix = f"rel_mem:{user_id}:"
            result = []
            for key, data in self._cache.items():
                if key.startswith(prefix) and data.get("stream_id") == self.stream_id:
                    result.append(data)
            result.sort(key=lambda x: x.get("created_at", 0), reverse=True)
            return result[:100]
        except Exception as e:
            logger.error(f"加载关系记忆失败: {e}")
            return []

    def save_stream_states(self, stream_states: Dict[str, Dict]) -> bool:
        try:
            from src.common.database.database_model import StreamStateData
            from src.common.database.database import db
            db.connect(reuse_if_open=True)
            now = time.time()
            for state_key, state_val in stream_states.items():
                val_str = json.dumps(state_val, ensure_ascii=False) if isinstance(state_val, dict) else str(state_val)
                record, created = StreamStateData.get_or_create(
                    stream_id=self.stream_id,
                    state_key=state_key,
                    defaults={"state_value": val_str, "updated_at": now}
                )
                if not created:
                    record.state_value = val_str
                    record.updated_at = now
                    record.save()
            return True
        except Exception as e:
            logger.error(f"保存流状态失败: {e}")
            return False

    def load_stream_states(self) -> Dict[str, Dict]:
        try:
            from src.common.database.database_model import StreamStateData
            from src.common.database.database import db
            db.connect(reuse_if_open=True)
            records = StreamStateData.select().where(
                StreamStateData.stream_id == self.stream_id
            )
            result = {}
            for rec in records:
                try:
                    result[rec.state_key] = json.loads(rec.state_value)
                except (json.JSONDecodeError, TypeError):
                    result[rec.state_key] = rec.state_value
            return result
        except Exception as e:
            logger.error(f"加载流状态失败: {e}")
            return {}

    def close(self):
        pass


_db_instances: Dict[str, EmotionDatabase] = {}


def get_emotion_database(stream_id: str) -> EmotionDatabase:
    if stream_id not in _db_instances:
        _db_instances[stream_id] = EmotionDatabase(stream_id)
    return _db_instances[stream_id]

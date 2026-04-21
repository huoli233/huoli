import time
import json
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from src.common.logger import get_logger

logger = get_logger("情绪数据库")


@dataclass
class EmotionRecord:
    user_id: str
    channel_id: str
    affection: float = 0.0
    impression: str = "初次见面"
    relationship: str = "陌生人"
    annoyance: float = 0.0
    trust_score: float = 0.0
    trauma_score: float = 0.0
    trauma_history: List[Dict] = field(default_factory=list)
    trauma_last_recall: float = 0.0
    trauma_recall_count: int = 0
    baseline_affection: float = 0.0
    interaction_count: int = 0
    last_interaction: float = 0.0
    is_blocked: bool = False
    block_until: float = 0.0
    block_reason: str = ""
    volatility: float = 1.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class PersonaRecord:
    persona_id: str
    channel_id: str
    name: str = ""
    tone: str = ""
    traits: List[str] = field(default_factory=list)
    speech_style: str = ""
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    source: str = "generated"
    extra: Dict = field(default_factory=dict)


@dataclass
class RelationshipMemory:
    user_id: str
    channel_id: str
    memory_type: str
    content: str
    sentiment: str = "neutral"
    created_at: float = field(default_factory=time.time)


class EmotionDatabase:
    def __init__(self, channel_id: str):
        self.channel_id = channel_id
        self._user_states: Dict[str, Dict] = {}
        self._personas: Dict[str, PersonaRecord] = {}
        self._relationship_memories: Dict[str, List[RelationshipMemory]] = {}
        self._stream_states: Dict[str, Dict] = {}
        self._initialized = False

    def _get_db_model(self, model_name: str):
        try:
            from src.common.database.database import db

            return db
        except Exception as e:
            logger.debug(f"数据库模型获取失败: {e}")
            return None

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._initialized = True
        self._load_from_database()

    def _load_from_database(self):
        try:
            from src.common.database.database_model import (
                PersistenceSlot,
                MemoryRecord,
                UserLearningLog,
            )

            self._load_user_states_from_slots()
            self._load_personas_from_slots()
            self._load_memories_from_records()
        except Exception as e:
            logger.debug(f"从数据库加载失败: {e}")

    def _load_user_states_from_slots(self):
        try:
            from src.common.database.database_model import PersistenceSlot

            slot = PersistenceSlot.get_or_none(
                PersistenceSlot.slot_key == f"emotion_states_{self.channel_id}"
            )
            if slot and slot.slot_value:
                data = json.loads(slot.slot_value)
                for user_id, state_data in data.items():
                    if isinstance(state_data, dict):
                        self._user_states[user_id] = self._normalize_user_state(
                            user_id, state_data
                        )
                logger.debug(f"加载了 {len(self._user_states)} 个用户情绪状态")
        except Exception as e:
            logger.debug(f"加载用户状态失败: {e}")

    def _load_personas_from_slots(self):
        try:
            from src.common.database.database_model import PersistenceSlot

            slot = PersistenceSlot.get_or_none(
                PersistenceSlot.slot_key == f"personas_{self.channel_id}"
            )
            if slot and slot.slot_value:
                data = json.loads(slot.slot_value)
                for persona_id, persona_data in data.items():
                    self._personas[persona_id] = PersonaRecord(
                        persona_id=persona_id,
                        channel_id=self.channel_id,
                        **{
                            k: v
                            for k, v in persona_data.items()
                            if k != "persona_id" and k != "channel_id"
                        },
                    )
                logger.debug(f"加载了 {len(self._personas)} 个人格记录")
        except Exception as e:
            logger.debug(f"加载人格记录失败: {e}")

    def _load_memories_from_records(self):
        try:
            from src.common.database.database_model import MemoryRecord

            records = (
                MemoryRecord.select()
                .where(
                    (MemoryRecord.entry_category == "relationship_memory")
                    & (MemoryRecord.extra_json.contains(self.channel_id))
                )
                .limit(500)
            )
            for record in records:
                try:
                    extra = json.loads(record.extra_json or "{}")
                    user_id = extra.get("user_id", "")
                    if user_id:
                        if user_id not in self._relationship_memories:
                            self._relationship_memories[user_id] = []
                        self._relationship_memories[user_id].append(
                            RelationshipMemory(
                                user_id=user_id,
                                channel_id=self.channel_id,
                                memory_type=extra.get("memory_type", ""),
                                content=record.content,
                                sentiment=extra.get("sentiment", "neutral"),
                                created_at=record.birth_ts,
                            )
                        )
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
        except Exception as e:
            logger.debug(f"加载关系记忆失败: {e}")

    def _persist_user_states(self):
        try:
            from src.common.database.database_model import PersistenceSlot

            data = {}
            for user_id, state in self._user_states.items():
                data[user_id] = state if isinstance(state, dict) else asdict(state)
            slot, _ = PersistenceSlot.get_or_create(
                slot_key=f"emotion_states_{self.channel_id}",
                defaults={
                    "slot_value": json.dumps(data, ensure_ascii=False),
                    "birth_ts": time.time(),
                    "modify_ts": time.time(),
                    "ttl_days": 365,
                },
            )
            slot.slot_value = json.dumps(data, ensure_ascii=False)
            slot.modify_ts = time.time()
            slot.save()
            return True
        except Exception as e:
            logger.error(f"持久化用户状态失败: {e}")
            return False

    def _persist_personas(self):
        try:
            from src.common.database.database_model import PersistenceSlot

            data = {}
            for persona_id, persona in self._personas.items():
                data[persona_id] = asdict(persona)
            slot, _ = PersistenceSlot.get_or_create(
                slot_key=f"personas_{self.channel_id}",
                defaults={
                    "slot_value": json.dumps(data, ensure_ascii=False),
                    "birth_ts": time.time(),
                    "modify_ts": time.time(),
                    "ttl_days": 365,
                },
            )
            slot.slot_value = json.dumps(data, ensure_ascii=False)
            slot.modify_ts = time.time()
            slot.save()
            return True
        except Exception as e:
            logger.error(f"持久化人格记录失败: {e}")
            return False

    def _normalize_user_state(
        self, user_id: str, state_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        state = dict(self._user_states.get(user_id, {}))
        if state_data:
            state.update(dict(state_data))
        canonical_trust = float(
            state.get(
                "trust_value",
                state.get("trust_score", state.get("trust", 0.0)),
            )
            or 0.0
        )
        now = time.time()
        state.setdefault("user_id", user_id)
        state.setdefault("channel_id", self.channel_id)
        state.setdefault("affection", 0.0)
        state.setdefault("relationship", "陌生人")
        state.setdefault("annoyance", 0.0)
        state.setdefault("trauma_score", 0.0)
        state.setdefault("interaction_count", 0)
        state.setdefault("last_interaction", 0.0)
        state.setdefault("baseline_affection", 0.0)
        state.setdefault("is_blocked", False)
        state.setdefault("block_until", 0.0)
        state.setdefault("block_reason", "")
        state.setdefault("created_at", now)
        state.setdefault("updated_at", now)
        state["trust_value"] = canonical_trust
        state["trust_score"] = canonical_trust
        state["trust"] = canonical_trust
        return state

    def save_user_state(self, user_id: str, state_data: Dict) -> bool:
        self._ensure_initialized()
        try:
            normalized = self._normalize_user_state(user_id, state_data)
            normalized["updated_at"] = time.time()
            self._user_states[user_id] = normalized
            self._persist_user_states()
            return True
        except Exception as e:
            logger.error(f"保存用户状态失败: {e}")
            return False

    def load_user_state(self, user_id: str) -> Optional[Dict]:
        self._ensure_initialized()
        if user_id in self._user_states:
            return self._normalize_user_state(user_id)
        return None

    def load_all_states(self) -> Dict[str, Dict]:
        self._ensure_initialized()
        return {
            uid: self._normalize_user_state(uid, state)
            for uid, state in self._user_states.items()
        }

    def delete_user_state(self, user_id: str) -> bool:
        self._ensure_initialized()
        if user_id in self._user_states:
            del self._user_states[user_id]
            self._persist_user_states()
            return True
        return False

    def get_statistics(self) -> Dict:
        self._ensure_initialized()
        total = len(self._user_states)
        if total == 0:
            return {"total_users": 0, "avg_affection": 0.0}
        total_affection = sum(
            float(state.get("affection", 0.0) or 0.0)
            for state in self._user_states.values()
        )
        return {
            "total_users": total,
            "avg_affection": total_affection / total,
            "total_interactions": sum(
                int(state.get("interaction_count", 0) or 0)
                for state in self._user_states.values()
            ),
            "blocked_users": sum(
                1
                for state in self._user_states.values()
                if bool(state.get("is_blocked", False))
            ),
        }

    def save_persona(self, persona_data: Dict) -> bool:
        self._ensure_initialized()
        try:
            persona_id = persona_data.get("persona_id", "")
            if not persona_id:
                return False
            now = time.time()
            if persona_id in self._personas:
                record = self._personas[persona_id]
                record.name = persona_data.get("name", record.name)
                record.tone = persona_data.get("tone", record.tone)
                record.traits = persona_data.get("traits", record.traits)
                record.speech_style = persona_data.get(
                    "speech_style", record.speech_style
                )
                record.last_used = persona_data.get("last_used", now)
                record.use_count = persona_data.get(
                    "use_count", record.use_count
                )
                record.source = persona_data.get("source", record.source)
                record.extra = persona_data.get("extra", record.extra)
            else:
                record = PersonaRecord(
                    persona_id=persona_id,
                    channel_id=self.channel_id,
                    name=persona_data.get("name", ""),
                    tone=persona_data.get("tone", ""),
                    traits=persona_data.get("traits", []),
                    speech_style=persona_data.get("speech_style", ""),
                    created_at=persona_data.get("created_at", now),
                    last_used=persona_data.get("last_used", now),
                    use_count=persona_data.get("use_count", 0),
                    source=persona_data.get("source", "generated"),
                    extra=persona_data.get("extra", {}),
                )
                self._personas[persona_id] = record
            self._persist_personas()
            return True
        except Exception as e:
            logger.error(f"保存人格失败: {e}")
            return False

    def load_all_personas(self) -> List[Dict]:
        self._ensure_initialized()
        return [asdict(p) for p in self._personas.values()]

    def delete_persona(self, persona_id: str) -> bool:
        self._ensure_initialized()
        if persona_id in self._personas:
            del self._personas[persona_id]
            self._persist_personas()
            return True
        return False

    def save_relationship_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        sentiment: str = "neutral",
    ) -> bool:
        self._ensure_initialized()
        try:
            memory = RelationshipMemory(
                user_id=user_id,
                channel_id=self.channel_id,
                memory_type=memory_type,
                content=content[:500],
                sentiment=sentiment,
                created_at=time.time(),
            )
            if user_id not in self._relationship_memories:
                self._relationship_memories[user_id] = []
            self._relationship_memories[user_id].append(memory)
            if len(self._relationship_memories[user_id]) > 100:
                self._relationship_memories[user_id] = (
                    self._relationship_memories[user_id][-100:]
                )
            try:
                from src.common.database.database_model import MemoryRecord

                MemoryRecord.create(
                    record_id=f"relmem_{user_id}_{int(time.time() * 1000)}",
                    stream_id=self.channel_id,
                    user_id=user_id,
                    entry_category="relationship_memory",
                    content=content[:500],
                    significance=0.5 if sentiment == "positive" else 0.3,
                    birth_ts=time.time(),
                    last_visit_ts=time.time(),
                    extra_json=json.dumps(
                        {
                            "user_id": user_id,
                            "channel_id": self.channel_id,
                            "memory_type": memory_type,
                            "sentiment": sentiment,
                        },
                        ensure_ascii=False,
                    ),
                )
            except Exception as _e:
                logger.debug(f"[情绪数据库:{self.channel_id}] 异常: {_e}")
            return True
        except Exception as e:
            logger.error(f"保存关系记忆失败: {e}")
            return False

    def load_relationship_memories(self, user_id: str) -> List[Dict]:
        self._ensure_initialized()
        if user_id in self._relationship_memories:
            return [asdict(m) for m in self._relationship_memories[user_id]]
        return []

    def save_stream_states(self, stream_states: Dict[str, Dict]) -> bool:
        self._ensure_initialized()
        try:
            self._stream_states = stream_states
            try:
                from src.common.database.database_model import PersistenceSlot

                slot, _ = PersistenceSlot.get_or_create(
                    slot_key=f"stream_states_{self.channel_id}",
                    defaults={
                        "slot_value": json.dumps(
                            stream_states, ensure_ascii=False
                        ),
                        "birth_ts": time.time(),
                        "modify_ts": time.time(),
                        "ttl_days": 30,
                    },
                )
                slot.slot_value = json.dumps(stream_states, ensure_ascii=False)
                slot.modify_ts = time.time()
                slot.save()
            except Exception as _e:
                logger.debug(f"[情绪数据库:{self.channel_id}] 异常: {_e}")
            logger.debug(f"保存了 {len(stream_states)} 个流状态")
            return True
        except Exception as e:
            logger.error(f"保存流状态失败: {e}")
            return False

    def load_stream_states(self) -> Dict[str, Dict]:
        self._ensure_initialized()
        if self._stream_states:
            return self._stream_states
        try:
            from src.common.database.database_model import PersistenceSlot

            slot = PersistenceSlot.get_or_none(
                PersistenceSlot.slot_key == f"stream_states_{self.channel_id}"
            )
            if slot and slot.slot_value:
                self._stream_states = json.loads(slot.slot_value)
                return self._stream_states
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return {}

    def get_user_affection_ranking(self, top_k: int = 10) -> List[Dict]:
        self._ensure_initialized()
        sorted_states = sorted(
            self._user_states.values(),
            key=lambda state: float(state.get("affection", 0.0) or 0.0),
            reverse=True,
        )
        return [
            {
                "user_id": str(state.get("user_id", "") or ""),
                "affection": float(state.get("affection", 0.0) or 0.0),
                "relationship": str(
                    state.get("relationship", "陌生人") or "陌生人"
                ),
            }
            for state in sorted_states[:top_k]
        ]

    def get_users_by_relationship(self, relationship: str) -> List[str]:
        self._ensure_initialized()
        return [
            uid
            for uid, state in self._user_states.items()
            if str(state.get("relationship", "") or "") == relationship
        ]

    def update_affection(
        self, user_id: str, delta: float, reason: str = ""
    ) -> bool:
        self._ensure_initialized()
        if user_id not in self._user_states:
            self.save_user_state(
                user_id,
                {
                    "affection": delta,
                    "interaction_count": 1,
                    "last_interaction": time.time(),
                },
            )
            return True
        state = self._normalize_user_state(user_id)
        new_affection = max(
            -100.0,
            min(100.0, float(state.get("affection", 0.0) or 0.0) + delta),
        )
        state["affection"] = new_affection
        state["interaction_count"] = (
            int(state.get("interaction_count", 0) or 0) + 1
        )
        state["last_interaction"] = time.time()
        state["updated_at"] = time.time()
        self._user_states[user_id] = state
        self._persist_user_states()
        logger.debug(
            f"用户 {user_id} 好感度变化: {delta:+.2f} -> {new_affection:.2f}, 原因: {reason}"
        )
        return True

    def update_trust(
        self, user_id: str, delta: float, reason: str = ""
    ) -> bool:
        self._ensure_initialized()
        if user_id not in self._user_states:
            self.save_user_state(user_id, {"trust_value": delta})
            return True
        state = self._normalize_user_state(user_id)
        new_trust = max(
            -100.0,
            min(100.0, float(state.get("trust_value", 0.0) or 0.0) + delta),
        )
        state["trust_value"] = new_trust
        state["trust_score"] = new_trust
        state["trust"] = new_trust
        state["updated_at"] = time.time()
        self._user_states[user_id] = state
        self._persist_user_states()
        logger.debug(
            f"用户 {user_id} 信任度变化: {delta:+.2f} -> {new_trust:.2f}, 原因: {reason}"
        )
        return True

    def block_user(
        self, user_id: str, duration_seconds: float, reason: str = ""
    ) -> bool:
        self._ensure_initialized()
        if user_id not in self._user_states:
            self.save_user_state(user_id, {})
        state = self._normalize_user_state(user_id)
        state["is_blocked"] = True
        state["block_until"] = time.time() + duration_seconds
        state["block_reason"] = reason
        state["updated_at"] = time.time()
        self._user_states[user_id] = state
        self._persist_user_states()
        logger.info(
            f"用户 {user_id} 被屏蔽 {duration_seconds}秒, 原因: {reason}"
        )
        return True

    def unblock_user(self, user_id: str) -> bool:
        self._ensure_initialized()
        if user_id in self._user_states:
            state = self._normalize_user_state(user_id)
            state["is_blocked"] = False
            state["block_until"] = 0.0
            state["block_reason"] = ""
            state["updated_at"] = time.time()
            self._user_states[user_id] = state
            self._persist_user_states()
            return True
        return False

    def is_user_blocked(self, user_id: str) -> bool:
        self._ensure_initialized()
        if user_id not in self._user_states:
            return False
        state = self._normalize_user_state(user_id)
        if not bool(state.get("is_blocked", False)):
            return False
        block_until = float(state.get("block_until", 0.0) or 0.0)
        if block_until > 0 and time.time() > block_until:
            state["is_blocked"] = False
            state["block_until"] = 0.0
            state["block_reason"] = ""
            self._user_states[user_id] = state
            self._persist_user_states()
            return False
        return True

    def get_emotion_trend(self, user_id: str, hours: int = 24) -> Dict:
        self._ensure_initialized()
        if user_id not in self._user_states:
            return {"trend": "unknown", "current": 0.0}
        state = self._normalize_user_state(user_id)
        elapsed = time.time() - float(
            state.get("created_at", time.time()) or time.time()
        )
        if elapsed < 3600:
            return {
                "trend": "new",
                "current": float(state.get("affection", 0.0) or 0.0),
            }
        current_affection = float(state.get("affection", 0.0) or 0.0)
        baseline_affection = float(state.get("baseline_affection", 0.0) or 0.0)
        change_rate = (current_affection - baseline_affection) / (elapsed / 3600)
        if change_rate > 0.5:
            trend = "rising_fast"
        elif change_rate > 0.1:
            trend = "rising"
        elif change_rate < -0.5:
            trend = "falling_fast"
        elif change_rate < -0.1:
            trend = "falling"
        else:
            trend = "stable"
        return {
            "trend": trend,
            "current": current_affection,
            "baseline": baseline_affection,
            "change_rate": change_rate,
        }

    def close(self):
        self._persist_user_states()
        self._persist_personas()
        logger.debug(f"情绪数据库 {self.channel_id} 已关闭")


_db_instances: Dict[str, EmotionDatabase] = {}


def get_emotion_database(channel_id: str) -> EmotionDatabase:
    if channel_id not in _db_instances:
        _db_instances[channel_id] = EmotionDatabase(channel_id)
    return _db_instances[channel_id]


def reset_emotion_database(channel_id: str) -> None:
    if channel_id in _db_instances:
        _db_instances[channel_id].close()
        del _db_instances[channel_id]


def get_all_databases() -> Dict[str, EmotionDatabase]:
    return _db_instances.copy()

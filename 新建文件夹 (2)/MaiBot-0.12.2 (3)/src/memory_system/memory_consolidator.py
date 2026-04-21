import time
import json
from typing import List, Dict, Optional, Set
from pathlib import Path
from src.common.logger import get_logger

logger = get_logger("memory_consolidator")

STM_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "short_term_memory"


class MemoryAgeLevel:
    FRESH = "fresh"
    TODAY = "today"
    YESTERDAY = "yesterday"
    RECENT = "recent"
    WEEK = "week"
    OLD = "old"


MEMORY_AGE_THRESHOLDS = {
    MemoryAgeLevel.FRESH: 3600,
    MemoryAgeLevel.TODAY: 86400,
    MemoryAgeLevel.YESTERDAY: 172800,
    MemoryAgeLevel.RECENT: 259200,
    MemoryAgeLevel.WEEK: 604800,
}


MEMORY_CLARITY_VALUES = {
    MemoryAgeLevel.FRESH: 1.0,
    MemoryAgeLevel.TODAY: 0.9,
    MemoryAgeLevel.YESTERDAY: 0.75,
    MemoryAgeLevel.RECENT: 0.6,
    MemoryAgeLevel.WEEK: 0.4,
    MemoryAgeLevel.OLD: 0.2,
}


class MemoryConsolidator:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self._consolidation_threshold = 10
        self._similarity_threshold = 0.7
        self._last_consolidation = {}
        self._consolidation_interval = 3600

    async def consolidate_channel_memories(self, stream_id: str):
        now = time.time()
        last_time = self._last_consolidation.get(stream_id, 0)
        if now - last_time < self._consolidation_interval:
            return
        try:
            from src.memory_system.hippocampus_buffer import get_hippocampus_buffer, MemoryTier
            from src.common.database.database_model import MemoryEntry
            hippo = get_hippocampus_buffer(stream_id)
            await hippo.transform_memories()
            sync_count = 0
            for tier in MemoryTier:
                for mid, hippo_mem in hippo._layers[tier].items():
                    try:
                        entry = MemoryEntry.get_or_none(MemoryEntry.memory_id == mid)
                        if entry:
                            entry.tier = hippo_mem.tier.value
                            entry.clarity = hippo.get_memory_clarity(hippo_mem)
                            entry.access_count = hippo_mem.access_count
                            entry.last_accessed = hippo_mem.last_accessed
                            if hippo_mem.compressed_content and not entry.summary:
                                entry.summary = hippo_mem.compressed_content
                            entry.save()
                            sync_count += 1
                    except Exception as e:
                        logger.debug(f"同步单条记忆失败 {mid[:8]}: {e}")
            try:
                recent = self.memory_manager.get_recent_memories(stream_id, limit=50)
                if len(recent) >= self._consolidation_threshold:
                    short_term = [m for m in recent if (now - m.created_at) < 86400]
                    long_term_candidates = [m for m in recent if (now - m.created_at) >= 86400 and m.importance > 0.6]
                    for mem in long_term_candidates:
                        if mem.memory_type == "conversation":
                            mem.memory_type = "fact"
                            mem.importance = min(1.0, mem.importance + 0.1)
                            mem.save()
                    similar_groups = self._group_similar_memories(short_term)
                    for group in similar_groups:
                        if len(group) >= 3:
                            merged = self._merge_memory_group(group, stream_id)
                            if merged:
                                for mem in group:
                                    try:
                                        self.memory_manager.delete_memory(mem.memory_id)
                                    except Exception:
                                        pass
            except Exception as e:
                logger.debug(f"记忆整合跳过: {e}")
            self._last_consolidation[stream_id] = now
            if sync_count > 0:
                logger.debug(f"[固化器] {stream_id[:8]} 同步完成: {sync_count} 条")
        except Exception as e:
            logger.error(f"[固化器] {stream_id[:8]} 整合失败: {e}")

    async def consolidate_stream_memories(self, stream_id: str):
        await self.consolidate_channel_memories(stream_id)

    def _group_similar_memories(self, memories) -> List[List]:
        groups = []
        used: Set[str] = set()
        for i, mem1 in enumerate(memories):
            mid1 = getattr(mem1, 'memory_id', str(i))
            if mid1 in used:
                continue
            group = [mem1]
            used.add(mid1)
            for j, mem2 in enumerate(memories[i+1:], i+1):
                mid2 = getattr(mem2, 'memory_id', str(j))
                if mid2 in used:
                    continue
                c1 = getattr(mem1, 'content', str(mem1))
                c2 = getattr(mem2, 'content', str(mem2))
                if self._calculate_similarity(c1, c2) > self._similarity_threshold:
                    group.append(mem2)
                    used.add(mid2)
            if len(group) > 1:
                groups.append(group)
        return groups

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _merge_memory_group(self, group, stream_id: str) -> Optional[str]:
        try:
            contents = [str(getattr(m, 'content', m)) for m in group]
            merged_content = " | ".join(contents[:3])
            avg_importance = sum(getattr(m, 'importance', 0.5) for m in group) / len(group)
            avg_importance = min(1.0, avg_importance + 0.2)
            memory_id = self.memory_manager.create_memory(
                stream_id=stream_id,
                content=merged_content,
                memory_type="fact",
                importance=avg_importance,
                summary=f"整合自{len(group)}条记忆",
            )
            logger.debug(f"整合{len(group)}条记忆为{str(memory_id)[:8]}")
            return memory_id
        except Exception as e:
            logger.debug(f"记忆合并失败: {e}")
            return None


class ShortTermMemory:
    def __init__(self, capacity: int = 20):
        self._capacity = capacity
        self._memories: Dict[str, Dict[str, List[Dict]]] = {}
        self._channel_index: Dict[str, List[Dict]] = {}
        self._last_save = time.time()
        self._auto_save_interval = 60
        self._cleanup_interval = 300
        self._user_cross_context_memories: Dict[str, List[Dict]] = {}
        STM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load_all_channels()

    def _get_data_file(self):
        return STM_DATA_DIR / "short_term_memory.json"

    def add(self, channel_id: str, content: str, importance: float = 0.5,
            user_id: Optional[str] = None):
        if not content or len(content.strip()) < 3:
            return
        content_stripped = content.strip()
        if (
            content_stripped.startswith("http") or
            len(set(content_stripped)) < 2 or
            content_stripped in ["...", "。。。", "？？？", "！！！"] or
            all(c in ",.!?;:()[]{}\"'-_=+*&^%$#@`~" for c in content_stripped)
        ):
            return
        if importance < 0.3:
            return
        if channel_id not in self._memories:
            self._memories[channel_id] = {}
            self._channel_index[channel_id] = []
        user_key = user_id or "system"
        if user_key not in self._memories[channel_id]:
            self._memories[channel_id][user_key] = []
        now = time.time()
        memory_item = {
            "content": content,
            "importance": importance,
            "timestamp": now,
            "user_id": user_id,
            "channel_id": channel_id,
            "age_level": MemoryAgeLevel.FRESH,
            "access_count": 0,
            "last_accessed": now,
            "summary": content[:80] if len(content) > 80 else content,
        }
        self._memories[channel_id][user_key].append(memory_item)
        self._channel_index[channel_id].append(memory_item)
        if len(self._memories[channel_id][user_key]) > self._capacity:
            removed = self._memories[channel_id][user_key].pop(0)
            if removed in self._channel_index[channel_id]:
                self._channel_index[channel_id].remove(removed)
        if len(self._channel_index[channel_id]) > self._capacity * 5:
            self._channel_index[channel_id] = self._channel_index[channel_id][-self._capacity * 5:]

    def get_recent(self, channel_id: str, limit: int = 10,
                   user_id: Optional[str] = None,
                   max_age_seconds: Optional[float] = None) -> List[Dict]:
        if channel_id not in self._memories:
            return []
        now = time.time()
        effective_max_age = max_age_seconds or 3600
        if user_id:
            user_key = user_id
            if user_key not in self._memories[channel_id]:
                return []
            memories = self._memories[channel_id][user_key]
        else:
            memories = self._channel_index.get(channel_id, [])
        filtered = []
        for mem in memories:
            age = now - mem.get("timestamp", 0)
            if age <= effective_max_age:
                decay_factor = 1.0 - (age / effective_max_age) * 0.5
                mem_copy = mem.copy()
                mem_copy["decay_factor"] = decay_factor
                mem_copy["importance"] = mem.get("importance", 0.5) * decay_factor
                filtered.append(mem_copy)
        return sorted(filtered, key=lambda x: x.get("timestamp", 0))[-limit:]

    def get_user_memories(self, channel_id: str, user_id: str, limit: int = 10) -> List[Dict]:
        return self.get_recent(channel_id, limit, user_id)

    def get_all_users(self, channel_id: str) -> List[str]:
        if channel_id not in self._memories:
            return []
        return [uid for uid in self._memories[channel_id].keys() if uid != "system"]

    def clear(self, channel_id: str, user_id: Optional[str] = None):
        if channel_id not in self._memories:
            return
        if user_id:
            user_key = user_id
            if user_key in self._memories[channel_id]:
                self._channel_index[channel_id] = [
                    m for m in self._channel_index.get(channel_id, [])
                    if m.get("user_id") != user_id
                ]
                del self._memories[channel_id][user_key]
        else:
            del self._memories[channel_id]
            if channel_id in self._channel_index:
                del self._channel_index[channel_id]

    def cleanup_expired(self, max_age_seconds: float = 604800):
        now = time.time()
        cleaned_count = 0
        for channel_id in list(self._memories.keys()):
            for user_key in list(self._memories.get(channel_id, {}).keys()):
                if user_key not in self._memories.get(channel_id, {}):
                    continue
                original_count = len(self._memories[channel_id][user_key])
                self._memories[channel_id][user_key] = [
                    m for m in self._memories[channel_id][user_key]
                    if (now - m.get("timestamp", now)) < max_age_seconds or m.get("importance", 0.5) > 0.7
                ]
                cleaned_count += original_count - len(self._memories[channel_id][user_key])
        if cleaned_count > 0:
            logger.info(f"[记忆清理] 已清理 {cleaned_count} 条过期短期记忆")

    def _get_age_level(self, age_seconds: float) -> str:
        if age_seconds < MEMORY_AGE_THRESHOLDS[MemoryAgeLevel.FRESH]:
            return MemoryAgeLevel.FRESH
        elif age_seconds < MEMORY_AGE_THRESHOLDS[MemoryAgeLevel.TODAY]:
            return MemoryAgeLevel.TODAY
        elif age_seconds < MEMORY_AGE_THRESHOLDS[MemoryAgeLevel.YESTERDAY]:
            return MemoryAgeLevel.YESTERDAY
        elif age_seconds < MEMORY_AGE_THRESHOLDS[MemoryAgeLevel.RECENT]:
            return MemoryAgeLevel.RECENT
        elif age_seconds < MEMORY_AGE_THRESHOLDS[MemoryAgeLevel.WEEK]:
            return MemoryAgeLevel.WEEK
        return MemoryAgeLevel.OLD

    def get_memories_by_age(self, channel_id: str, age_level: str,
                            limit: int = 20) -> List[Dict]:
        if channel_id not in self._channel_index:
            return []
        now = time.time()
        result = []
        for mem in self._channel_index[channel_id]:
            age = now - mem.get("timestamp", 0)
            if self._get_age_level(age) == age_level:
                result.append(mem)
        result.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return result[:limit]

    def get_memories_by_days_ago(self, channel_id: str, days: int,
                                 limit: int = 20) -> List[Dict]:
        if channel_id not in self._channel_index:
            return []
        now = time.time()
        start_time = now - (days + 1) * 86400
        end_time = now - days * 86400
        result = []
        for mem in self._channel_index[channel_id]:
            ts = mem.get("timestamp", 0)
            if start_time <= ts <= end_time:
                result.append(mem)
        result.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return result[:limit]

    def get_memory_clarity(self, memory: Dict) -> float:
        now = time.time()
        age = now - memory.get("timestamp", 0)
        age_level = self._get_age_level(age)
        base_clarity = MEMORY_CLARITY_VALUES.get(age_level, 0.2)
        access_bonus = min(0.3, memory.get("access_count", 0) * 0.03)
        importance_bonus = memory.get("importance", 0.5) * 0.1
        return min(1.0, base_clarity + access_bonus + importance_bonus)

    def add_user_cross_context_learning(self, user_id: str, topic: str,
                                         count: int, learning_stage: str):
        if user_id not in self._user_cross_context_memories:
            self._user_cross_context_memories[user_id] = []
        existing = next((m for m in self._user_cross_context_memories[user_id]
                        if m.get('topic') == topic), None)
        if existing:
            existing['count'] = count
            existing['learning_stage'] = learning_stage
            existing['updated_at'] = time.time()
        else:
            self._user_cross_context_memories[user_id].append({
                'topic': topic,
                'count': count,
                'learning_stage': learning_stage,
                'created_at': time.time(),
                'updated_at': time.time(),
            })
        logger.debug(f"[跨场景记忆] 用户{user_id[:8]}学习进度: {topic}={count}次({learning_stage})")
        self._save_user_cross_context_memories()

    def get_user_cross_context_learning(self, user_id: str) -> List[Dict]:
        return self._user_cross_context_memories.get(user_id, [])

    def generate_user_learning_prompt(self, user_id: str) -> str:
        learnings = self.get_user_cross_context_learning(user_id)
        if not learnings:
            return """【认知状态 - 天真无邪】
你对大部分话题都保持天真无邪的态度，对敏感话题、暗示性内容、成人向话题一窍不通。
当遇到不理解的内容时：
- 用字面意思理解对方的话
- 表现出真实的困惑："你在说什么？"
- 不会意识到对方的言外之意或暗示
- 可以岔开话题或转移注意力
你不会假装理解，也不会顺着对方的暗示接话。"""
        prompt_parts = []
        for learning in learnings:
            topic = learning.get('topic', '')
            count = learning.get('count', 0)
            if count == 1:
                prompt_parts.append(f"- {topic}：第一次听说这个词，完全不理解是什么意思，感到很困惑")
            elif count == 2:
                prompt_parts.append(f"- {topic}：之前听过一次，有些印象，但还是不太懂")
            elif count >= 3:
                prompt_parts.append(f"- {topic}：已经听过{count}次了，开始有所了解，但仍然保持天真的态度")
        if not prompt_parts:
            return """【认知状态 - 天真无邪】
你对大部分话题都保持天真无邪的态度，对敏感话题、暗示性内容、成人向话题一窍不通。
当遇到不理解的内容时：
- 用字面意思理解对方的话
- 表现出真实的困惑："你在说什么？"
- 不会意识到对方的言外之意或暗示
- 可以岔开话题或转移注意力
你不会假装理解，也不会顺着对方的暗示接话。"""
        return "【当前认知状态】\n" + "\n".join(prompt_parts) + "\n\n【重要】对于其他不熟悉的话题（尤其是暗示性、成人向内容），你仍然保持天真的态度，用字面意思理解，不会意识到言外之意。"

    def _get_cross_model(self):
        from src.common.database.database_model import CrossContextMemory
        return CrossContextMemory

    def _get_state_model(self):
        from src.common.database.database_model import StreamStateData
        return StreamStateData

    def _save_user_cross_context_memories(self):
        try:
            CCM = self._get_cross_model()
            now = time.time()
            for user_id, learnings in self._user_cross_context_memories.items():
                for item in learnings:
                    topic = item.get('topic', '')
                    content_str = f"{topic}|{item.get('count',0)}|{item.get('learning_stage','')}"
                    CCM.insert(
                        user_id=user_id, memory_type="cross_context_learning",
                        content=content_str,
                        importance=min(1.0, item.get('count', 0) * 0.1),
                        source_stream="",
                        created_at=item.get('created_at', now),
                        updated_at=item.get('updated_at', now),
                    ).on_conflict(
                        conflict_target=[CCM.user_id, CCM.memory_type],
                        preserve=[CCM.user_id, CCM.memory_type],
                        update={CCM.content: content_str, CCM.updated_at: now}
                    ).execute()
        except Exception as e:
            logger.debug(f"[跨场景记忆] 保存失败: {e}")

    def _load_user_cross_context_memories(self):
        try:
            CCM = self._get_cross_model()
            rows = CCM.select().where(CCM.memory_type == "cross_context_learning")
            loaded = 0
            for row in rows:
                parts = row.content.split("|", 2)
                if len(parts) < 3:
                    continue
                topic, count_str, stage = parts
                uid = row.user_id
                if uid not in self._user_cross_context_memories:
                    self._user_cross_context_memories[uid] = []
                self._user_cross_context_memories[uid].append({
                    'topic': topic, 'count': int(count_str),
                    'learning_stage': stage,
                    'created_at': row.created_at, 'updated_at': row.updated_at,
                })
                loaded += 1
            if loaded == 0:
                self._migrate_json_cross_context()
            elif loaded > 0:
                logger.debug(f"[跨场景记忆] 已加载, 用户数: {len(self._user_cross_context_memories)}")
        except Exception as e:
            logger.debug(f"[跨场景记忆] 加载失败: {e}")

    def _migrate_json_cross_context(self):
        data_file = self._get_data_file()
        if not data_file.exists():
            return
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cross_data = data.get('user_cross_context_memories', {})
            if cross_data:
                self._user_cross_context_memories = cross_data
                self._save_user_cross_context_memories()
                logger.info(f"跨场景记忆已从JSON迁移到结构化数据库")
        except Exception as e:
            logger.debug(f"JSON跨场景记忆迁移跳过: {e}")

    def _load_all_channels(self):
        try:
            SM = self._get_state_model()
            row = SM.get_or_none(
                (SM.stream_id == "__stm_consolidator") & (SM.state_key == "channels_data")
            )
            if row:
                data = json.loads(row.state_value)
                channels_data = data.get('channels', {})
                loaded_count = 0
                for channel_id, channel_data in channels_data.items():
                    if channel_id not in self._memories:
                        self._memories[channel_id] = {}
                        self._channel_index[channel_id] = []
                    for user_key, memories in channel_data.items():
                        self._memories[channel_id][user_key] = memories
                        self._channel_index[channel_id].extend(memories)
                        loaded_count += len(memories)
                    self._channel_index[channel_id].sort(
                        key=lambda x: x.get("timestamp", 0), reverse=True)
                if loaded_count > 0:
                    logger.debug(f"[短期记忆] 已恢复 {loaded_count} 条")
            else:
                self._migrate_json_channels()
            self._load_user_cross_context_memories()
        except Exception as e:
            logger.debug(f"[短期记忆] 加载跳过: {e}")

    def _migrate_json_channels(self):
        data_file = self._get_data_file()
        if not data_file.exists():
            return
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            channels_data = data.get('channels', {})
            loaded_count = 0
            for channel_id, channel_data in channels_data.items():
                if channel_id not in self._memories:
                    self._memories[channel_id] = {}
                    self._channel_index[channel_id] = []
                for user_key, memories in channel_data.items():
                    self._memories[channel_id][user_key] = memories
                    self._channel_index[channel_id].extend(memories)
                    loaded_count += len(memories)
            if loaded_count > 0:
                self._save_all_channels()
                data_file.rename(data_file.with_suffix('.json.bak'))
                logger.info(f"短期记忆已从JSON迁移到数据库({loaded_count}条)")
        except Exception as e:
            logger.debug(f"JSON短期记忆迁移跳过: {e}")

    def _save_all_channels(self):
        try:
            SM = self._get_state_model()
            max_per_channel = 500
            channels_data = {}
            for channel_id, channel_data in self._memories.items():
                channels_data[channel_id] = {}
                for user_key, memories in channel_data.items():
                    recent = sorted(memories, key=lambda x: x.get("timestamp", 0), reverse=True)
                    channels_data[channel_id][user_key] = recent[:max_per_channel]
            data = {'channels': channels_data}
            json_str = json.dumps(data, ensure_ascii=False)
            now = time.time()
            SM.insert(
                stream_id="__stm_consolidator", state_key="channels_data",
                state_value=json_str, updated_at=now
            ).on_conflict(
                conflict_target=[SM.stream_id, SM.state_key],
                update={SM.state_value: json_str, SM.updated_at: now}
            ).execute()
        except Exception as e:
            logger.debug(f"[短期记忆] 保存失败: {e}")

    def save_all_channels(self):
        self._save_all_channels()

    def _auto_save_and_cleanup(self):
        now = time.time()
        if now - getattr(self, '_last_cleanup', 0) > self._cleanup_interval:
            self._cleanup_expired_short_term_memory()
            self._last_cleanup = now
        if now - self._last_save > self._auto_save_interval:
            self._save_all_channels()
            self._last_save = now

    def _cleanup_expired_short_term_memory(self):
        now = time.time()
        max_age_seconds = 604800
        cleaned_count = 0
        for channel_id in list(self._memories.keys()):
            for user_key in list(self._memories.get(channel_id, {}).keys()):
                original_count = len(self._memories[channel_id].get(user_key, []))
                filtered = []
                for mem in self._memories[channel_id].get(user_key, []):
                    age = now - mem.get("timestamp", 0)
                    importance = mem.get("importance", 0.5)
                    access_count = mem.get("access_count", 0)
                    mem["age_level"] = self._get_age_level(age)
                    if age <= max_age_seconds:
                        filtered.append(mem)
                    elif importance >= 0.7 or access_count >= 3:
                        filtered.append(mem)
                    else:
                        cleaned_count += 1
                self._memories[channel_id][user_key] = filtered
                if not self._memories[channel_id][user_key]:
                    del self._memories[channel_id][user_key]
            if channel_id in self._channel_index:
                self._channel_index[channel_id] = [
                    m for m in self._channel_index[channel_id]
                    if (now - m.get("timestamp", 0) <= max_age_seconds or
                        m.get("importance", 0.5) >= 0.7 or
                        m.get("access_count", 0) >= 3)
                ]
            if not self._memories.get(channel_id):
                self._memories.pop(channel_id, None)
                self._channel_index.pop(channel_id, None)
        if cleaned_count > 0:
            logger.debug(f"[短期记忆] 遗忘了 {cleaned_count} 条老旧记忆")

    def force_cleanup_expired(self):
        self._cleanup_expired_short_term_memory()


class LongTermMemory:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    def store(self, stream_id: str, content: str, importance: float = 0.7):
        memory_id = self.memory_manager.create_memory(
            stream_id=stream_id,
            content=content,
            memory_type="fact",
            importance=importance,
        )
        return memory_id

    def retrieve(self, stream_id: str, query: str, limit: int = 5):
        return self.memory_manager.search_memories(
            stream_id=stream_id,
            query=query,
            memory_type="fact",
            limit=limit,
        )


class MemoryAgingSystem:
    AGING_STAGES = [
        (7, 0.80),
        (30, 0.50),
        (90, 0.20),
        (180, 0.05),
    ]
    DELETE_AFTER_DAYS = 180
    DELETE_MIN_ACCESS = 3

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    def calculate_clarity(self, memory) -> float:
        from src.memory_system.memory_helpers import calculate_memory_clarity
        return calculate_memory_clarity(
            memory.created_at,
            importance=memory.importance,
            access_count=memory.access_count,
        )

    def get_memory_with_clarity(self, memory) -> Dict:
        clarity = self.calculate_clarity(memory)
        age_days = (time.time() - memory.created_at) / 86400
        result = {
            "memory_id": memory.memory_id,
            "content": memory.content,
            "clarity": clarity,
            "age_days": round(age_days, 1),
            "access_count": memory.access_count,
            "importance": memory.importance,
        }
        if clarity >= 0.7:
            result["display_content"] = memory.content
            result["clarity_label"] = "清晰"
        elif clarity >= 0.4:
            summary = getattr(memory, 'summary', None) or memory.content[:100]
            result["display_content"] = summary + " (印象模糊)"
            result["clarity_label"] = "有些模糊"
        elif clarity >= 0.2:
            summary = getattr(memory, 'summary', None) or memory.content[:50]
            result["display_content"] = f"隐约记得...{summary[:50]}..."
            result["clarity_label"] = "很模糊"
        else:
            result["display_content"] = "好像有这么回事，但想不起来具体内容了..."
            result["clarity_label"] = "几乎遗忘"
        return result

    def get_memories_with_clarity(self, memories: List) -> List[Dict]:
        return [self.get_memory_with_clarity(m) for m in memories]

    def cleanup_forgotten_memories(self, stream_id: str) -> int:
        now = time.time()
        cutoff_time = now - (self.DELETE_AFTER_DAYS * 86400)
        try:
            from src.common.database.database_model import MemoryEntry
            old_memories = MemoryEntry.select().where(
                (MemoryEntry.stream_id == stream_id) &
                (MemoryEntry.created_at < cutoff_time) &
                (MemoryEntry.access_count < self.DELETE_MIN_ACCESS) &
                (MemoryEntry.importance < 0.7)
            )
            deleted_count = 0
            for memory in old_memories:
                clarity = self.calculate_clarity(memory)
                if clarity < 0.1:
                    memory.delete_instance()
                    deleted_count += 1
            if deleted_count > 0:
                logger.info(f"[记忆衰老] 清理了 {deleted_count} 条被遗忘的记忆")
            return deleted_count
        except Exception as e:
            logger.debug(f"[记忆衰老] 清理失败: {e}")
            return 0

    def compress_old_memories(self, stream_id: str) -> int:
        now = time.time()
        compress_after_days = 60
        cutoff_time = now - (compress_after_days * 86400)
        try:
            from src.common.database.database_model import MemoryEntry
            old_memories = MemoryEntry.select().where(
                (MemoryEntry.stream_id == stream_id) &
                (MemoryEntry.created_at < cutoff_time) &
                (MemoryEntry.memory_type == "conversation")
            )
            compressed_count = 0
            for memory in old_memories:
                if len(memory.content) > 200:
                    if not memory.summary:
                        memory.summary = memory.content[:100]
                    memory.memory_type = "compressed"
                    memory.save()
                    compressed_count += 1
            if compressed_count > 0:
                logger.info(f"[记忆衰老] 压缩了 {compressed_count} 条老旧记忆")
            return compressed_count
        except Exception as e:
            logger.debug(f"[记忆衰老] 压缩失败: {e}")
            return 0

    def get_clarity_prompt(self, memories: List) -> str:
        if not memories:
            return ""
        clear_memories = []
        fuzzy_memories = []
        vague_memories = []
        for mem in memories:
            result = self.get_memory_with_clarity(mem)
            if result["clarity"] >= 0.7:
                clear_memories.append(result)
            elif result["clarity"] >= 0.4:
                fuzzy_memories.append(result)
            else:
                vague_memories.append(result)
        lines = []
        if clear_memories:
            lines.append("[清晰的记忆]")
            for m in clear_memories[:3]:
                lines.append(f"- {m['display_content'][:80]}")
        if fuzzy_memories:
            lines.append("[有些模糊的记忆]")
            for m in fuzzy_memories[:3]:
                lines.append(f"- {m['display_content'][:60]}...")
        if vague_memories:
            lines.append("[非常模糊的记忆]")
            for m in vague_memories[:2]:
                lines.append(f"- {m['display_content']}")
        return "\n".join(lines) if lines else ""


_stm_instance: Optional[ShortTermMemory] = None
_aging_system: Optional[MemoryAgingSystem] = None


def get_short_term_memory() -> ShortTermMemory:
    global _stm_instance
    if _stm_instance is None:
        _stm_instance = ShortTermMemory()
    return _stm_instance


def get_memory_aging_system(memory_manager=None) -> Optional[MemoryAgingSystem]:
    global _aging_system
    if _aging_system is None and memory_manager is not None:
        _aging_system = MemoryAgingSystem(memory_manager)
    return _aging_system


def get_memory_consolidator(memory_manager):
    return MemoryConsolidator(memory_manager)

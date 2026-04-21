import time
import json
import asyncio
import hashlib
import re
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("hippocampus")


class MemoryTier(IntEnum):
    """记忆层级枚举 - 模拟人脑记忆的分层结构"""
    IMMEDIATE = 0    # 即时记忆层：0-1小时
    WORKING = 1      # 工作记忆层：1-24小时
    RECENT = 2       # 近期记忆层：1-3天
    MIDTERM = 3      # 中期记忆层：3-7天
    LONGTERM = 4     # 长期记忆层：7天以上


# 各层级的时间阈值（秒）
TIER_TIME_THRESHOLDS = {
    MemoryTier.IMMEDIATE: 3600,       # 1小时
    MemoryTier.WORKING: 86400,        # 24小时
    MemoryTier.RECENT: 259200,        # 3天
    MemoryTier.MIDTERM: 604800,       # 7天
    MemoryTier.LONGTERM: float('inf') # 永久
}


# 各层级的清晰度晋升系数
TIER_CLARITY_MULTIPLIERS = {
    MemoryTier.IMMEDIATE: 1.0,
    MemoryTier.WORKING: 0.9,
    MemoryTier.RECENT: 0.7,
    MemoryTier.MIDTERM: 0.5,
    MemoryTier.LONGTERM: 0.3
}


@dataclass
class BufferedMemory:
    """缓冲记忆条目"""
    memory_id: str
    stream_id: str
    user_id: Optional[str]
    content: str
    summary: Optional[str]
    importance: float
    created_at: float
    last_accessed: float
    access_count: int
    tier: MemoryTier
    clarity: float = 1.0  # 记忆清晰度 (1.0 满分，随着时间或层级下降)
    keywords: List[str] = field(default_factory=list)
    compressed_content: Optional[str] = None


class HippocampusBuffer:
    """海马体记忆缓冲区 - 管理记忆的分层与转化"""
    
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        # 按层级存储记忆：{tier: {memory_id: BufferedMemory}}
        self._layers: Dict[MemoryTier, Dict[str, BufferedMemory]] = {
            tier: {} for tier in MemoryTier
        }
        self._last_transform_time = 0.0
        self._transform_interval = 1800.0  # 30分钟转化一次
        self._load_from_disk()
    
    def _generate_id(self, content: str) -> str:
        """生成记忆唯一标识"""
        unique_str = f"{self.stream_id}:{content}:{time.time()}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]
    
    def add_memory(self, content: str, user_id: Optional[str] = None,
                   importance: float = 0.5, keywords: Optional[List[str]] = None) -> str:
        """添加新记忆到即时层"""
        memory_id = self._generate_id(content)
        now = time.time()
        memory = BufferedMemory(
            memory_id=memory_id,
            stream_id=self.stream_id,
            user_id=user_id,
            content=content,
            summary=self._auto_summarize(content),
            importance=importance,
            created_at=now,
            last_accessed=now,
            access_count=0,
            tier=MemoryTier.IMMEDIATE,
            clarity=1.0,
            keywords=keywords or self._extract_keywords(content)
        )
        self._layers[MemoryTier.IMMEDIATE][memory_id] = memory
        logger.debug(f"[海马体] 新记忆添加: {memory_id[:8]} (stream: {self.stream_id})")
        return memory_id
    
    def access_memory(self, memory_id: str) -> Optional[BufferedMemory]:
        """访问记忆，更新访问时间和计数"""
        for tier in MemoryTier:
            if memory_id in self._layers[tier]:
                memory = self._layers[tier][memory_id]
                memory.last_accessed = time.time()
                memory.access_count += 1
                # 频繁访问的记忆提升层级（如“工作记忆”提升回“即时记忆”）
                if memory.access_count >= 5 and tier > MemoryTier.WORKING:
                    self._promote_memory(memory, tier)
                return memory
        return None
    
    def _promote_memory(self, memory: BufferedMemory, current_tier: MemoryTier):
        """将高频访问的记忆提升到更高层级"""
        if current_tier <= MemoryTier.WORKING:
            return
        target_tier = MemoryTier(current_tier - 1)
        del self._layers[current_tier][memory.memory_id]
        memory.tier = target_tier
        memory.last_accessed = time.time()
        self._layers[target_tier][memory.memory_id] = memory
        logger.debug(f"[海马体] 记忆 {memory.memory_id[:8]} 活跃提升 -> {target_tier.name}")
    
    def get_recent_memories(self, limit: int = 20, 
                           max_tier: MemoryTier = MemoryTier.RECENT,
                           user_id: Optional[str] = None) -> List[BufferedMemory]:
        """获取最近记忆"""
        all_memories = []
        for tier in MemoryTier:
            if tier > max_tier:
                break
            for memory in self._layers[tier].values():
                if user_id and memory.user_id != user_id:
                    continue
                all_memories.append(memory)
        all_memories.sort(key=lambda m: m.created_at, reverse=True)
        return all_memories[:limit]
    
    def search_memories(self, query: str, limit: int = 10) -> List[Tuple[BufferedMemory, float]]:
        """搜索记忆并返回相关度"""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        for tier in MemoryTier:
            for memory in self._layers[tier].values():
                score = self._calculate_relevance(query_lower, query_words, memory)
                if score > 0.1:
                    results.append((memory, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_memories_by_time_range(self, start_time: float, end_time: float,
                                   limit: int = 30) -> List[BufferedMemory]:
        """按时间范围获取记忆"""
        result = []
        for tier in MemoryTier:
            for memory in self._layers[tier].values():
                if start_time <= memory.created_at <= end_time:
                    result.append(memory)
        result.sort(key=lambda m: m.created_at, reverse=True)
        return result[:limit]

    def get_memories_by_days_ago(self, days: int, limit: int = 20) -> List[BufferedMemory]:
        """获取N天前的记忆"""
        now = time.time()
        start_time = now - (days + 1) * 86400
        end_time = now - days * 86400
        return self.get_memories_by_time_range(start_time, end_time, limit)

    def _determine_tier_by_age(self, age_seconds: float) -> MemoryTier:
        """根据年龄确定记忆应该属于哪个层级"""
        for tier in MemoryTier:
            if age_seconds < TIER_TIME_THRESHOLDS[tier]:
                return tier
        return MemoryTier.LONGTERM

    def _calculate_relevance(self, query_lower: str, query_words: set, 
                             memory: BufferedMemory) -> float:
        """计算记忆相关度，受清晰度影响"""
        score = 0.0
        content_lower = memory.content.lower()
        if query_lower in content_lower:
            score += 0.5
        memory_words = set(content_lower.split())
        common_words = query_words & memory_words
        if query_words:
            word_overlap = len(common_words) / len(query_words)
            score += word_overlap * 0.3
        for kw in memory.keywords:
            if query_lower in kw.lower():
                score += 0.2
                break
        clarity = self.get_memory_clarity(memory)
        score *= clarity
        return min(1.0, score)
    
    def get_memory_clarity(self, memory: BufferedMemory) -> float:
        """获取记忆清晰度"""
        from src.memory_system.memory_helpers import calculate_memory_clarity
        return calculate_memory_clarity(
            memory.created_at,
            importance=memory.importance,
            access_count=memory.access_count
        )
    
    async def transform_memories(self):
        """执行记忆转化与遗忘逻辑"""
        now = time.time()
        if now - self._last_transform_time < self._transform_interval:
            return
        self._last_transform_time = now
        
        transform_count = forget_count = compress_count = 0
        
        for tier in list(MemoryTier)[:-1]:
            next_tier = MemoryTier(tier + 1)
            threshold = TIER_TIME_THRESHOLDS[tier]
            
            memories_to_move = []
            for memory_id, memory in list(self._layers[tier].items()):
                age = now - memory.created_at
                
                # Clarity Decay: 即使不移动，清晰度也会随时间轻微下降
                decay_factor = 0.99 if memory.access_count < 3 else 0.995
                memory.clarity *= decay_factor
                
                if age >= threshold:
                    memories_to_move.append((memory_id, memory))
            
            for memory_id, memory in memories_to_move:
                del self._layers[tier][memory_id]
                
                if self._should_forget(memory, next_tier):
                    forget_count += 1
                    continue
                
                # 晋升层级时，清晰度进一步受层级系数修正
                memory.clarity *= TIER_CLARITY_MULTIPLIERS.get(next_tier, 1.0)
                
                # 进入后期层级时进行摘要压缩
                if next_tier >= MemoryTier.MIDTERM and not memory.compressed_content:
                    memory.compressed_content = self._compress_content(memory.content)
                    compress_count += 1
                
                memory.tier = next_tier
                self._layers[next_tier][memory_id] = memory
                transform_count += 1
        
        if transform_count > 0 or forget_count > 0:
            logger.info(f"[海马体] 转化完成: 移动{transform_count}, 压缩{compress_count}, 遗忘{forget_count}")
            self._save_to_disk()
    
    def _should_forget(self, memory: BufferedMemory, target_tier: MemoryTier) -> bool:
        """拟人化遗忘决策"""
        if memory.importance >= 0.7:
            return False
        if memory.access_count >= 3:
            return False
        if target_tier == MemoryTier.WORKING:
            return memory.importance < 0.3 and memory.access_count == 0
        elif target_tier == MemoryTier.RECENT:
            return memory.importance < 0.4 and memory.access_count <= 1
        elif target_tier == MemoryTier.MIDTERM:
            return memory.importance < 0.5 and memory.access_count <= 2
        elif target_tier == MemoryTier.LONGTERM:
            return memory.importance < 0.6 and memory.access_count < 3
        return False

    def _compress_content(self, content: str, max_length: int = 150) -> str:
        """模拟'印象模糊'的摘要压缩"""
        if len(content) <= max_length:
            return content
        sentences = content.replace('。', '.').replace('！', '.').replace('？', '.').split('.')
        compressed = []
        current_length = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current_length + len(sentence) > max_length:
                break
            compressed.append(sentence)
            current_length += len(sentence) + 1
        if not compressed:
            return content[:max_length] + "..."
        return "。".join(compressed) + "..."

    def _auto_summarize(self, content: str, max_length: int = 80) -> str:
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."

    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', text)
        word_count: Dict[str, int] = {}
        for w in words:
            if len(w) >= 2:
                word_count[w] = word_count.get(w, 0) + 1
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:5]]
    
    def get_context_for_prompt(self, query: Optional[str] = None, max_tokens: int = 500) -> str:
        """生成用于注入到prompt中的记忆上下文"""
        lines = []
        if query:
            search_results = self.search_memories(query, limit=5)
            if search_results:
                lines.append("[相关记忆]")
                for memory, score in search_results:
                    clarity = self.get_memory_clarity(memory)
                    display_content = self._get_display_content(memory, clarity)
                    clarity_label = self._get_clarity_label(clarity)
                    lines.append(f"- ({clarity_label}) {display_content}")
        now = time.time()
        time_sections = [
            ("今天的记忆", 0, 86400),
            ("昨天的记忆", 86400, 172800),
            ("前几天的记忆", 172800, 604800),
        ]
        for section_name, min_age, max_age in time_sections:
            section_memories = []
            for tier in MemoryTier:
                for memory in self._layers[tier].values():
                    age = now - memory.created_at
                    if min_age <= age < max_age:
                        section_memories.append(memory)
            if section_memories:
                section_memories.sort(key=lambda m: m.importance, reverse=True)
                lines.append(f"\n[{section_name}]")
                for memory in section_memories[:3]:
                    clarity = self.get_memory_clarity(memory)
                    display_content = self._get_display_content(memory, clarity)
                    lines.append(f"- {display_content}")
        result = "\n".join(lines)
        if len(result) > max_tokens * 2:
            result = result[:max_tokens * 2] + "...(记忆过多，已截断)"
        return result
    
    def _get_display_content(self, memory: BufferedMemory, clarity: float) -> str:
        if clarity >= 0.7:
            return memory.content[:100]
        elif clarity >= 0.4:
            return memory.summary or memory.content[:60]
        elif clarity >= 0.2:
            if memory.compressed_content:
                return f"隐约记得...{memory.compressed_content[:40]}"
            return f"隐约记得...{memory.content[:40]}"
        return "好像有这么回事，但具体想不起来了"
    
    def _get_clarity_label(self, clarity: float) -> str:
        if clarity >= 0.7:
            return "清晰"
        elif clarity >= 0.4:
            return "有些模糊"
        elif clarity >= 0.2:
            return "很模糊"
        return "几乎遗忘"
    
    def _save_to_disk(self):
        """保存到数据库"""
        try:
            from src.common.database.database_model import HippocampusBuffer as HippoTable
            for tier in MemoryTier:
                for memory in self._layers[tier].values():
                    HippoTable.insert(
                        memory_id=memory.memory_id,
                        stream_id=self.stream_id,
                        user_id=memory.user_id,
                        content=memory.content,
                        summary=memory.summary,
                        importance=memory.importance,
                        clarity=memory.clarity,
                        created_at=memory.created_at,
                        last_accessed=memory.last_accessed,
                        access_count=memory.access_count,
                        tier=tier.value,
                        keywords=json.dumps(memory.keywords, ensure_ascii=False),
                        compressed_content=memory.compressed_content
                    ).on_conflict_replace().execute()
            logger.debug(f"[海马体] 数据库同步完成: {self.stream_id}")
        except Exception as e:
            logger.error(f"[海马体] 数据库保存异常: {e}")
    
    def _load_from_disk(self):
        """从数据库加载"""
        try:
            from src.common.database.database_model import HippocampusBuffer as HippoTable
            query = HippoTable.select().where(HippoTable.stream_id == self.stream_id)
            records = list(query)
            for r in records:
                try:
                    memory = BufferedMemory(
                        memory_id=r.memory_id, stream_id=r.stream_id, user_id=r.user_id,
                        content=r.content, summary=r.summary, importance=r.importance,
                        clarity=getattr(r, 'clarity', 1.0),
                        created_at=r.created_at, last_accessed=r.last_accessed,
                        access_count=r.access_count, tier=MemoryTier(r.tier),
                        keywords=json.loads(r.keywords) if r.keywords else [],
                        compressed_content=r.compressed_content
                    )
                    self._layers[memory.tier][memory.memory_id] = memory
                except: continue
            logger.debug(f"[海马体] 恢复记忆: {len(records)}条")
        except Exception as e:
            logger.debug(f"[海马体] 加载跳过: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        stats = {
            "stream_id": self.stream_id,
            "total_memories": 0,
            "by_tier": {},
            "last_transform": self._last_transform_time,
        }
        for tier in MemoryTier:
            count = len(self._layers[tier])
            stats["by_tier"][tier.name] = count
            stats["total_memories"] += count
        return stats

    def clear(self):
        for tier in MemoryTier:
            self._layers[tier].clear()
        try:
            from src.common.database.database_model import HippocampusBuffer as HippoTable
            HippoTable.delete().where(HippoTable.stream_id == self.stream_id).execute()
        except Exception:
            pass


_buffers: Dict[str, HippocampusBuffer] = {}


def get_hippocampus_buffer(stream_id: str) -> HippocampusBuffer:
    if stream_id not in _buffers:
        _buffers[stream_id] = HippocampusBuffer(stream_id)
    return _buffers[stream_id]


def reset_hippocampus_buffer(stream_id: str):
    if stream_id in _buffers:
        _buffers[stream_id].clear()
        del _buffers[stream_id]


class HippocampusScheduler:
    def __init__(self):
        self._running = False
        self._interval = 1800.0
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        if self._running: return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
    
    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
    
    async def _run_loop(self):
        while self._running:
            await asyncio.sleep(self._interval)
            await self._transform_all_buffers()

    async def _transform_all_buffers(self):
        for buffer in _buffers.values():
            try:
                await buffer.transform_memories()
            except Exception as e:
                logger.debug(f"[海马体调度] 转化失败: {e}")


_scheduler: Optional[HippocampusScheduler] = None


def get_hippocampus_scheduler() -> HippocampusScheduler:
    global _scheduler
    if _scheduler is None: _scheduler = HippocampusScheduler()
    return _scheduler


async def init_hippocampus_system():
    scheduler = get_hippocampus_scheduler()
    await scheduler.start()
    return scheduler

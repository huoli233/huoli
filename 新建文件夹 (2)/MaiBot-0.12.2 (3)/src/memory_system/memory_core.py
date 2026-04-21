import time
import json
import hashlib
from typing import List, Dict, Optional, Any
from src.common.logger import get_logger

logger = get_logger("memory_core")


class MemoryManager:
    def __init__(self):
        self._memory_cache: Dict[str, Any] = {}
        self._last_cleanup = time.time()
        self._last_compression = time.time()
        self._last_deduplication = time.time()
        self._last_deep_optimization = time.time()
        self._last_overload_check = time.time()
        self._overload_protector = None
        self._calibrated_streams: set = set()
        self._state_db = None

    def _generate_memory_id(self, stream_id: str, content: str) -> str:
        unique_str = f"{stream_id}:{content}:{time.time()}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    def _generate_content_hash(self, content: str) -> str:
        normalized = ' '.join(content.strip().lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _get_memory_model(self):
        try:
            from src.common.database.database_model import MemoryEntry
            return MemoryEntry
        except Exception:
            return None

    def _check_duplicate_content(self, stream_id: str, content: str, user_id: Optional[str] = None):
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return None
        try:
            content_hash = self._generate_content_hash(content)
            now = time.time()
            week_ago = now - 7 * 86400
            query = MemoryEntry.select().where(
                (MemoryEntry.stream_id == stream_id) &
                (MemoryEntry.created_at >= week_ago)
            )
            if user_id:
                query = query.where(MemoryEntry.user_id == user_id)
            for existing in query:
                existing_hash = self._generate_content_hash(existing.content)
                if existing_hash == content_hash:
                    return existing
            for existing in query:
                similarity = self._calculate_content_similarity(content, existing.content)
                if similarity > 0.9:
                    return existing
            return None
        except Exception as e:
            logger.debug(f"重复检测失败: {e}")
            return None

    def create_memory(
        self,
        stream_id: str,
        content: str,
        memory_type: str = "conversation",
        user_id: Optional[str] = None,
        importance: float = 0.5,
        summary: str = "",
        extra_data: Optional[Dict] = None,
    ) -> str:
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return ""
        now = time.time()
        
        # 情绪重要性调节 (保持原有逻辑)
        emotion_adjusted_importance = importance
        if user_id and stream_id:
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
                tracker = get_emotion_tracker(stream_id)
                state = tracker.get_user_state(user_id, create_if_missing=False)
                if state:
                    if state.affection < -50 or state.trauma_score > 5:
                        emotion_adjusted_importance = min(1.0, importance + 0.3)
                    elif state.affection < -20 or state.annoyance > 50:
                        emotion_adjusted_importance = min(1.0, importance + 0.15)
                    elif state.affection > 50:
                        emotion_adjusted_importance = min(1.0, importance + 0.2)
            except Exception:
                pass
        importance = emotion_adjusted_importance
        
        try:
            # 1. 重复检测
            existing_memory = self._check_duplicate_content(stream_id, content, user_id)
            if existing_memory:
                existing_memory.last_accessed = now
                existing_memory.access_count += 1
                if importance > existing_memory.importance:
                    existing_memory.importance = min(1.0, (existing_memory.importance + importance) / 2)
                existing_memory.save()
                self._memory_cache[existing_memory.memory_id] = existing_memory
                return existing_memory.memory_id
            
            # 2. 同步存入海马体缓冲区 (HippocampusBuffer)
            from src.memory_system.hippocampus_buffer import get_hippocampus_buffer
            hippo = get_hippocampus_buffer(stream_id)
            memory_id = hippo.add_memory(content, user_id=user_id, importance=importance)
            
            # 3. 创建长期存储条目 (MemoryEntry)，初始设为 IMMEDIATE 层级
            memory = MemoryEntry.create(
                memory_id=memory_id,
                stream_id=stream_id,
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                importance=importance,
                access_count=0,
                created_at=now,
                last_accessed=now,
                tier=0,  # MemoryTier.IMMEDIATE
                clarity=1.0,
                summary=summary or (content[:80] + "..." if len(content) > 80 else content),
                metadata=json.dumps(extra_data, ensure_ascii=False) if extra_data else None
            )
            self._memory_cache[memory_id] = memory
            logger.debug(f"[记忆核心] 新记忆已创建并存入海马体: {memory_id[:8]}... 类型={memory_type}")
            return memory_id
        except Exception as e:
            logger.error(f"创建记忆失败: {e}")
            return ""

    def get_memory(self, memory_id: str):
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return None
        if memory_id in self._memory_cache:
            memory = self._memory_cache[memory_id]
        else:
            try:
                memory = MemoryEntry.get(MemoryEntry.memory_id == memory_id)
                self._memory_cache[memory_id] = memory
            except Exception:
                return None
        memory.access_count += 1
        memory.last_accessed = time.time()
        memory.save()
        return memory

    def search_memories(
        self,
        stream_id: str,
        query: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> List:
        """分层搜索记忆：海马体联想 -> 长期存储检索"""
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return []
        
        results = []
        try:
            # 1. 优先搜索海马体缓冲区 (极清晰/近期记忆)
            from src.memory_system.hippocampus_buffer import get_hippocampus_buffer
            hippo = get_hippocampus_buffer(stream_id)
            if query:
                hippo_results = hippo.search_memories(query, limit=limit // 2)
                for mem, score in hippo_results:
                    # 尝试转换回 MemoryEntry 对象（如果在缓存或 DB 中）
                    entry = self.get_memory(mem.memory_id)
                    if entry:
                        results.append(entry)
            
            # 2. 搜索长期存储 (Database)
            q = MemoryEntry.select().where(MemoryEntry.stream_id == stream_id)
            if memory_type:
                q = q.where(MemoryEntry.memory_type == memory_type)
            if user_id:
                q = q.where(MemoryEntry.user_id == user_id)
            if query:
                q = q.where(MemoryEntry.content.contains(query))
            
            # 排除已在海马体结果中的
            if results:
                existing_ids = [r.memory_id for r in results]
                q = q.where(MemoryEntry.memory_id.not_in(existing_ids))
            
            q = q.order_by(MemoryEntry.importance.desc(), MemoryEntry.created_at.desc())
            candidates = list(q.limit(limit - len(results)))
            
            # 计算结合清晰度的最终评分
            from src.memory_system.memory_helpers import calculate_memory_clarity
            scored_candidates = []
            for mem in candidates:
                clarity = calculate_memory_clarity(
                    mem.created_at, importance=mem.importance, access_count=mem.access_count
                )
                score = mem.importance * 0.5 + clarity * 0.5
                scored_candidates.append((score, mem))
            
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            results.extend([mem for _, mem in scored_candidates])
            
            # 更新访问热度
            for mem in results[:limit]:
                mem.access_count += 1
                mem.last_accessed = time.time()
                mem.save()
            
            return results[:limit]
        except Exception as e:
            logger.error(f"搜索记忆失败: {e}")
            return []

    def get_recent_memories(self, stream_id: str, limit: int = 20) -> List:
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return []
        try:
            return list(
                MemoryEntry.select()
                .where(MemoryEntry.stream_id == stream_id)
                .order_by(MemoryEntry.created_at.desc())
                .limit(limit)
            )
        except Exception as e:
            logger.error(f"获取最近记忆失败: {e}")
            return []

    def _get_state_db(self):
        if self._state_db is None:
            try:
                from src.chat.heart_flow.persistent_state_db import get_persistent_state_db
                self._state_db = get_persistent_state_db()
            except Exception as e:
                logger.debug(f"获取状态数据库失败: {e}")
        return self._state_db

    def _ensure_stream_calibrated(self, stream_id: str) -> bool:
        if not stream_id:
            return False
        if stream_id in self._calibrated_streams:
            return True
        try:
            state_db = self._get_state_db()
            if state_db:
                calibrated, forgotten = self._calibrate_memories_for_stream(stream_id, state_db)
                self._calibrated_streams.add(stream_id)
                if calibrated > 0 or forgotten > 0:
                    logger.debug(f"懒加载校准: stream={stream_id[:8]}... calibrated={calibrated} forgotten={forgotten}")
                return True
        except Exception as e:
            logger.debug(f"懒加载校准失败: {e}")
        self._calibrated_streams.add(stream_id)
        return False

    def _calibrate_memories_for_stream(self, stream_id: str, state_db) -> tuple:
        try:
            from src.memory_system.memory_helpers import calculate_memory_clarity
            MemoryEntry = self._get_memory_model()
            if not MemoryEntry:
                return 0, 0
            memory_state = state_db.load_memory_state(stream_id) if hasattr(state_db, 'load_memory_state') else None
            offline_duration = memory_state.get("offline_duration", 0.0) if memory_state else 0.0
            if offline_duration <= 0:
                return 0, 0
            memories = MemoryEntry.select().where(MemoryEntry.stream_id == stream_id)
            forgotten_count = 0
            calibrated_count = 0
            now = time.time()
            forget_threshold = 0.1
            for memory in memories:
                clarity = calculate_memory_clarity(
                    memory.created_at, importance=memory.importance, access_count=memory.access_count
                )
                calibrated_count += 1
                if clarity < forget_threshold:
                    age_days = (now - memory.created_at) / 86400
                    if age_days >= 180 and memory.access_count < 3 and memory.importance < 0.7:
                        memory.delete_instance()
                        self._memory_cache.pop(memory.memory_id, None)
                        forgotten_count += 1
            if hasattr(state_db, 'save_memory_state'):
                state_db.save_memory_state(stream_id)
            return calibrated_count, forgotten_count
        except Exception as e:
            logger.debug(f"校准频道失败: {e}")
            return 0, 0

    def _save_memory_state(self, stream_id: str):
        try:
            state_db = self._get_state_db()
            if state_db and hasattr(state_db, 'save_memory_state'):
                state_db.save_memory_state(stream_id)
        except Exception as e:
            logger.debug(f"保存记忆库状态失败: {e}")

    def perform_maintenance(self, stream_id: Optional[str] = None):
        now = time.time()
        if now - self._last_cleanup > 1800:
            self._cleanup_expired_cache()
            self._last_cleanup = now
        if now - self._last_compression > 7200:
            compressed = self._compress_old_memories(stream_id)
            if compressed > 0:
                logger.info(f"压缩了 {compressed} 条老旧记忆")
            self._last_compression = now
        if now - self._last_deduplication > 21600:
            deduplicated = self._deduplicate_memories(stream_id)
            if deduplicated > 0:
                logger.info(f"去重了 {deduplicated} 条重复记忆")
            self._last_deduplication = now
        if now - self._last_deep_optimization > 86400:
            self._perform_deep_optimization(stream_id)
            self._last_deep_optimization = now
        if now - self._last_overload_check > 600:
            self._check_memory_overload(stream_id)
            self._last_overload_check = now

    def _cleanup_expired_cache(self):
        now = time.time()
        expired_keys = [
            mid for mid, mem in self._memory_cache.items()
            if now - getattr(mem, 'last_accessed', now) > 3600
        ]
        for key in expired_keys:
            del self._memory_cache[key]
        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 条过期缓存")

    def _compress_old_memories(self, stream_id: Optional[str] = None) -> int:
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return 0
        try:
            now = time.time()
            cutoff_time = now - 30 * 86400
            query = MemoryEntry.select().where(
                (MemoryEntry.created_at < cutoff_time) &
                (MemoryEntry.memory_type == "conversation")
            )
            if stream_id:
                query = query.where(MemoryEntry.stream_id == stream_id)
            compressed_count = 0
            for memory in query:
                if len(memory.content) > 200:
                    memory.memory_type = "compressed"
                    memory.save()
                    compressed_count += 1
            return compressed_count
        except Exception as e:
            logger.error(f"内容压缩失败: {e}")
            return 0

    def _deduplicate_memories(self, stream_id: Optional[str] = None) -> int:
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return 0
        try:
            now = time.time()
            week_ago = now - 7 * 86400
            query = MemoryEntry.select().where(
                MemoryEntry.created_at >= week_ago
            ).order_by(MemoryEntry.created_at.desc())
            if stream_id:
                query = query.where(MemoryEntry.stream_id == stream_id)
            memories = list(query)
            if len(memories) < 2:
                return 0
            duplicates_found = 0
            processed_hashes = set()
            for memory in memories:
                content_hash = self._generate_content_hash(memory.content)
                if content_hash in processed_hashes:
                    memory.delete_instance()
                    self._memory_cache.pop(memory.memory_id, None)
                    duplicates_found += 1
                    continue
                processed_hashes.add(content_hash)
            return duplicates_found
        except Exception as e:
            logger.error(f"去重处理失败: {e}")
            return 0

    def delete_memory(self, memory_id: str) -> bool:
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return False
        try:
            MemoryEntry.delete().where(MemoryEntry.memory_id == memory_id).execute()
            self._memory_cache.pop(memory_id, None)
            return True
        except Exception as e:
            logger.error(f"删除记忆失败: {e}")
            return False

    def clear_stream_memory(self, stream_id: str) -> int:
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return 0
        try:
            deleted = MemoryEntry.delete().where(MemoryEntry.stream_id == stream_id).execute()
            keys_to_remove = [k for k in self._memory_cache]
            for k in keys_to_remove:
                mem = self._memory_cache[k]
                if getattr(mem, 'stream_id', '') == stream_id:
                    del self._memory_cache[k]
            return deleted
        except Exception as e:
            logger.error(f"清理频道记忆失败: {e}")
            return 0

    def get_memory_stats(self, stream_id: str) -> Dict[str, Any]:
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return {"total": 0, "by_type": {}}
        try:
            total = MemoryEntry.select().where(MemoryEntry.stream_id == stream_id).count()
            by_type = {}
            for mem_type in ["conversation", "knowledge", "user", "system", "compressed"]:
                count = MemoryEntry.select().where(
                    (MemoryEntry.stream_id == stream_id) &
                    (MemoryEntry.memory_type == mem_type)
                ).count()
                if count > 0:
                    by_type[mem_type] = count
            return {"total": total, "by_type": by_type, "cache_size": len(self._memory_cache)}
        except Exception as e:
            logger.error(f"获取记忆统计失败: {e}")
            return {"total": 0, "by_type": {}}

    def get_memory_statistics(self, stream_id: Optional[str] = None) -> Dict[str, Any]:
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return {"total": 0}
        try:
            query = MemoryEntry.select()
            if stream_id:
                query = query.where(MemoryEntry.stream_id == stream_id)
            memories = list(query)
            if not memories:
                return {"total": 0}
            now = time.time()
            stats = {
                "total": len(memories),
                "by_type": {},
                "by_age": {"recent": 0, "medium": 0, "old": 0},
                "cache_size": len(self._memory_cache),
                "avg_importance": sum(m.importance for m in memories) / len(memories),
                "total_access": sum(m.access_count for m in memories)
            }
            for memory in memories:
                mem_type = memory.memory_type
                stats["by_type"][mem_type] = stats["by_type"].get(mem_type, 0) + 1
                age_days = (now - memory.created_at) / 86400
                if age_days <= 7:
                    stats["by_age"]["recent"] += 1
                elif age_days <= 30:
                    stats["by_age"]["medium"] += 1
                else:
                    stats["by_age"]["old"] += 1
            return stats
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"total": 0, "error": str(e)}

    def optimize_database_indexes(self):
        try:
            from src.common.database.database_model import db
            db.execute_sql("REINDEX")
            db.execute_sql("VACUUM")
            logger.info("数据库索引优化完成")
        except Exception as e:
            logger.error(f"数据库优化失败: {e}")

    def _perform_deep_optimization(self, stream_id: Optional[str] = None):
        try:
            import asyncio
            from src.memory_system.deep_optimizer import get_deep_optimizer
            optimizer = get_deep_optimizer()
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(optimizer.run_optimization(stream_id))
            else:
                loop.run_until_complete(optimizer.run_optimization(stream_id))
        except Exception as e:
            logger.error(f"深度优化失败: {e}")

    def force_deep_optimization(self, stream_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            import asyncio
            from src.memory_system.deep_optimizer import get_deep_optimizer
            optimizer = get_deep_optimizer()
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(optimizer.run_optimization(stream_id))
                self._last_deep_optimization = time.time()
                return {"status": "deep_optimization_scheduled"}
            else:
                result = loop.run_until_complete(optimizer.run_optimization(stream_id))
                self._last_deep_optimization = time.time()
                return result
        except Exception as e:
            logger.error(f"强制优化失败: {e}")
            return {"error": str(e)}

    def _check_memory_overload(self, stream_id: Optional[str] = None):
        try:
            if self._overload_protector is None:
                from src.memory_system.memory_overload_system import get_memory_overload_protector
                self._overload_protector = get_memory_overload_protector(stream_id or "default")
            overload_state = self._overload_protector.assess_memory_overload()
            if self._overload_protector.should_trigger_emergency_forgetting():
                logger.warning("记忆过载触发紧急遗忘...")
                result = self._overload_protector.execute_intelligent_forgetting(target_reduction=0.25)
                logger.info(f"紧急遗忘完成 | 遗忘{result['forgotten_count']}条 释放{result['space_freed_mb']:.1f}MB")
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"检查过载状态失败: {e}")

    def _apply_memory_overload_effects(self, operation: str, **kwargs) -> Dict[str, Any]:
        try:
            if self._overload_protector is None:
                from src.memory_system.memory_overload_system import get_memory_overload_protector
                sid = kwargs.get("stream_id", "default")
                self._overload_protector = get_memory_overload_protector(sid)
            return self._overload_protector.apply_overload_effects(operation, **kwargs)
        except ImportError:
            return {"success": True, "effect_applied": False}
        except Exception as e:
            logger.debug(f"应用过载效果失败: {e}")
            return {"success": True, "effect_applied": False}

    def get_memory_overload_status(self, stream_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            if self._overload_protector is None:
                from src.memory_system.memory_overload_system import get_memory_overload_protector
                self._overload_protector = get_memory_overload_protector(stream_id or "default")
            return self._overload_protector.get_overload_statistics()
        except ImportError:
            return {"status": "overload_system_not_available"}
        except Exception as e:
            logger.error(f"获取过载状态失败: {e}")
            return {"error": str(e)}

    def get_overload_behavioral_prompt(self, stream_id: Optional[str] = None) -> str:
        try:
            if self._overload_protector is None:
                from src.memory_system.memory_overload_system import get_memory_overload_protector
                self._overload_protector = get_memory_overload_protector(stream_id or "default")
            return self._overload_protector.get_overload_behavioral_prompt()
        except ImportError:
            return ""
        except Exception as e:
            logger.debug(f"获取行为提示词失败: {e}")
            return ""

    def force_emergency_forgetting(self, stream_id: Optional[str] = None, reduction: float = 0.3) -> Dict[str, Any]:
        try:
            if self._overload_protector is None:
                from src.memory_system.memory_overload_system import get_memory_overload_protector
                self._overload_protector = get_memory_overload_protector(stream_id or "default")
            result = self._overload_protector.execute_intelligent_forgetting(reduction)
            logger.warning(f"强制遗忘完成 | 遗忘{result['forgotten_count']}条 释放{result['space_freed_mb']:.1f}MB")
            return result
        except ImportError:
            return {"error": "overload_system_not_available", "forgotten_count": 0}
        except Exception as e:
            logger.error(f"强制遗忘失败: {e}")
            return {"error": str(e), "forgotten_count": 0}

    def get_memory_health_report(self, stream_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            stats = self.get_memory_statistics(stream_id)
            total = stats.get("total", 0)
            if total == 0:
                return {"health_score": 100, "health_level": "优秀", "total": 0, "suggestions": []}
            health_score = 100
            by_age = stats.get("by_age", {})
            old_ratio = by_age.get("old", 0) / max(1, total)
            if old_ratio > 0.5:
                health_score -= 15
            avg_importance = stats.get("avg_importance", 0.5)
            if avg_importance < 0.3:
                health_score -= 15
            if total > 5000:
                health_score -= 20
            elif total > 2000:
                health_score -= 10
            compressed_count = stats.get("by_type", {}).get("compressed", 0)
            if compressed_count > total * 0.3:
                health_score -= 10
            health_score = max(0, min(100, health_score))
            if health_score >= 90:
                health_level = "优秀"
            elif health_score >= 70:
                health_level = "良好"
            elif health_score >= 50:
                health_level = "一般"
            else:
                health_level = "需要优化"
            suggestions = []
            if total > 5000:
                suggestions.append("数据量过大，建议执行深度优化")
            if old_ratio > 0.5:
                suggestions.append("老旧记忆占比过高，可清理无用数据")
            if avg_importance < 0.3:
                suggestions.append("平均重要性偏低，可能存在大量垃圾记忆")
            return {
                "health_score": health_score, "health_level": health_level,
                "total": total, "by_type": stats.get("by_type", {}),
                "by_age": by_age, "avg_importance": round(avg_importance, 3),
                "cache_size": stats.get("cache_size", 0),
                "suggestions": suggestions
            }
        except Exception as e:
            logger.error(f"获取健康报告失败: {e}")
            return {"error": str(e)}

    def cleanup_expired(self) -> int:
        MemoryEntry = self._get_memory_model()
        if not MemoryEntry:
            return 0
        try:
            now = time.time()
            if not hasattr(MemoryEntry, 'expires_at'):
                return 0
            deleted = MemoryEntry.delete().where(
                (MemoryEntry.expires_at.is_null(False)) &
                (MemoryEntry.expires_at < now)
            ).execute()
            if deleted > 0:
                logger.info(f"清理过期记忆: {deleted}条")
                stale_keys = []
                for memory_id in list(self._memory_cache.keys()):
                    try:
                        MemoryEntry.get(MemoryEntry.memory_id == memory_id)
                    except Exception:
                        stale_keys.append(memory_id)
                for k in stale_keys:
                    del self._memory_cache[k]
            return deleted
        except Exception as e:
            logger.error(f"清理过期记忆失败: {e}")
            return 0

    def _initialize_persistent_state(self):
        logger.debug("使用懒加载模式，跳过启动时全量校准")

    def clear_channel_memory(self, stream_id: str) -> int:
        try:
            from src.common.database.database_model import MemoryEntry
            count = MemoryEntry.delete().where(MemoryEntry.stream_id == stream_id).execute()
            keys_to_remove = [k for k in self._memory_cache if self._memory_cache[k].get("stream_id") == stream_id]
            for k in keys_to_remove:
                del self._memory_cache[k]
            self._calibrated_channels.discard(stream_id)
            logger.info(f"清除频道记忆: stream={stream_id[:8]}... 删除{count}条")
            return count
        except Exception as e:
            logger.error(f"清除频道记忆失败: {e}")
            return 0

    def _ensure_db_connected(self):
        try:
            from src.common.database.database import db as peewee_db
            if peewee_db.is_closed():
                peewee_db.connect()
        except Exception as e:
            logger.warning(f"确保数据库连接失败: {e}")

    def _ensure_channel_calibrated(self, stream_id: str) -> bool:
        if not stream_id:
            return False
        if stream_id in self._calibrated_channels:
            return True
        try:
            state_db = self._get_state_db()
            if state_db:
                calibrated, forgotten = self._calibrate_memories_for_channel(stream_id, state_db)
                self._calibrated_channels.add(stream_id)
                if calibrated > 0 or forgotten > 0:
                    logger.debug(f"懒加载校准: stream={stream_id[:8]}... calibrated={calibrated} forgotten={forgotten}")
                return True
        except Exception as e:
            logger.debug(f"懒加载校准失败: {e}")
        self._calibrated_channels.add(stream_id)
        return False

    def _calibrate_memories_for_channel(self, stream_id: str, state_db) -> tuple:
        try:
            from src.common.database.database_model import MemoryEntry
            memory_state = state_db.load_memory_state(stream_id) if hasattr(state_db, 'load_memory_state') else None
            offline_duration = memory_state.get("offline_duration", 0.0) if memory_state else 0.0
            if offline_duration <= 0:
                return 0, 0
            memories = MemoryEntry.select().where(MemoryEntry.stream_id == stream_id)
            forgotten_count = 0
            calibrated_count = 0
            now = time.time()
            forget_threshold = 0.1
            for memory in memories:
                age_days = (now - memory.created_at) / 86400
                clarity = max(0.0, 1.0 - age_days * 0.005) * memory.importance
                calibrated_count += 1
                if clarity < forget_threshold and age_days >= 180 and memory.access_count < 3 and memory.importance < 0.7:
                    memory.delete_instance()
                    self._memory_cache.pop(memory.memory_id, None)
                    forgotten_count += 1
            if hasattr(state_db, 'save_memory_state'):
                state_db.save_memory_state(stream_id)
            return calibrated_count, forgotten_count
        except Exception as e:
            logger.debug(f"校准频道失败: {e}")
            return 0, 0

    async def store_memory(self, stream_id: str, content: str, user_id: str,
                           importance: float = 0.5, is_bot: bool = False,
                           extra_metadata: Optional[Dict] = None) -> str:
        memory_type = "bot_response" if is_bot else "conversation"
        if extra_metadata and extra_metadata.get("is_harassment"):
            memory_type = "trauma"
            importance = max(importance, 0.9)
        return self.create_memory(
            stream_id=stream_id, content=content,
            memory_type=memory_type, user_id=user_id,
            importance=importance
        )


_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def get_memory_core() -> MemoryManager:
    return get_memory_manager()


def reset_memory_manager():
    global _memory_manager
    _memory_manager = None

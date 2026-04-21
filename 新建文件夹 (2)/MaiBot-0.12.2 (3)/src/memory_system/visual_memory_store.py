import time
import asyncio
from typing import Optional
from src.common.logger import get_logger

logger = get_logger("visual_memory")


class VisualMemoryStore:
    """图片Hash→描述缓存，避免重复调用VLM"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VisualMemoryStore, cls).__new__(cls)
            cls._instance._db_ready = False
        return cls._instance

    def _ensure_table(self):
        if self._db_ready:
            return True
        try:
            from src.common.database.database_model import db
            db.execute_sql("""
                CREATE TABLE IF NOT EXISTS visual_memories (
                    img_hash TEXT PRIMARY KEY,
                    description TEXT,
                    source_url TEXT,
                    created_at REAL
                )
            """)
            db.execute_sql("CREATE INDEX IF NOT EXISTS idx_vis_hash ON visual_memories(img_hash)")
            self._db_ready = True
            return True
        except Exception as e:
            logger.debug(f"视觉存储表创建失败: {e}")
            return False

    def get_description(self, img_hash: str) -> Optional[str]:
        if not img_hash or not self._ensure_table():
            return None
        try:
            from src.common.database.database_model import db
            cursor = db.execute_sql(
                "SELECT description FROM visual_memories WHERE img_hash = ?",
                (img_hash,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.debug(f"视觉检索失败: {e}")
            return None

    async def get_description_async(self, img_hash: str) -> Optional[str]:
        if not img_hash:
            return None
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.get_description, img_hash),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            logger.debug(f"视觉检索超时: {img_hash[:12]}")
            return None

    def store_memory(self, img_hash: str, description: str, url: str = ""):
        if not img_hash or not description or not self._ensure_table():
            return
        try:
            from src.common.database.database_model import db
            db.execute_sql(
                "INSERT OR REPLACE INTO visual_memories (img_hash, description, source_url, created_at) VALUES (?, ?, ?, ?)",
                (img_hash, description, url, time.time())
            )
            self._cleanup_old_records()
            logger.debug(f"视觉存储: {img_hash[:16]}...")
        except Exception as e:
            logger.debug(f"视觉存储失败: {e}")

    def _cleanup_old_records(self, max_records: int = 200):
        try:
            from src.common.database.database_model import db
            cursor = db.execute_sql("SELECT COUNT(*) FROM visual_memories")
            count = cursor.fetchone()[0]
            if count > max_records:
                delete_count = count - max_records
                db.execute_sql("""
                    DELETE FROM visual_memories WHERE img_hash IN (
                        SELECT img_hash FROM visual_memories ORDER BY created_at ASC LIMIT ?
                    )
                """, (delete_count,))
                logger.debug(f"视觉存储清理: {delete_count}条")
        except Exception:
            pass

    async def store_memory_async(self, img_hash: str, description: str, url: str = ""):
        if not img_hash or not description:
            return
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self.store_memory, img_hash, description, url),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            logger.debug("视觉存储超时")


_visual_store: Optional[VisualMemoryStore] = None


def get_visual_store() -> VisualMemoryStore:
    global _visual_store
    if _visual_store is None:
        _visual_store = VisualMemoryStore()
    return _visual_store

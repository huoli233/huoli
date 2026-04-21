import time
import json
from typing import Dict, Any, Optional, List
from src.common.logger import get_logger

logger = get_logger("cognitive_store")


class UserCognitiveStore:
    """用户认知画像存储，记录每个用户的长期偏好、性格标签、禁忌点"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UserCognitiveStore, cls).__new__(cls)
            cls._instance._db_ready = False
        return cls._instance

    def _ensure_table(self):
        if self._db_ready:
            return True
        try:
            from src.common.database.database_model import db
            db.execute_sql("""
                CREATE TABLE IF NOT EXISTS user_cognitive (
                    user_id TEXT PRIMARY KEY,
                    likes TEXT,
                    dislikes TEXT,
                    habits TEXT,
                    persona_tags TEXT,
                    last_updated REAL
                )
            """)
            db.execute_sql("""
                CREATE TABLE IF NOT EXISTS user_long_fragments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    content TEXT,
                    importance REAL,
                    created_at REAL
                )
            """)
            db.execute_sql("CREATE INDEX IF NOT EXISTS idx_frag_user ON user_long_fragments(user_id)")
            self._db_ready = True
            return True
        except Exception as e:
            logger.debug(f"认知画像表创建失败: {e}")
            return False

    def get_preferences(self, user_id: str) -> Dict[str, Any]:
        if not self._ensure_table():
            return {"likes": [], "dislikes": [], "habits": "", "tags": []}
        try:
            from src.common.database.database_model import db
            cursor = db.execute_sql(
                "SELECT likes, dislikes, habits, persona_tags FROM user_cognitive WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "likes": json.loads(row[0]) if row[0] else [],
                    "dislikes": json.loads(row[1]) if row[1] else [],
                    "habits": row[2] or "",
                    "tags": (row[3] or "").split(",") if row[3] else [],
                }
        except Exception as e:
            logger.debug(f"获取认知画像失败: {e}")
        return {"likes": [], "dislikes": [], "habits": "", "tags": []}

    def update_preference(self, user_id: str, field: str, value: Any):
        if not self._ensure_table():
            return
        try:
            from src.common.database.database_model import db
            cursor = db.execute_sql(
                "SELECT 1 FROM user_cognitive WHERE user_id = ?", (user_id,)
            )
            exists = cursor.fetchone()
            if not exists:
                db.execute_sql(
                    "INSERT INTO user_cognitive (user_id, last_updated) VALUES (?, ?)",
                    (user_id, time.time())
                )
            if field in ("likes", "dislikes"):
                value = json.dumps(value, ensure_ascii=False)
            db.execute_sql(
                f"UPDATE user_cognitive SET {field} = ?, last_updated = ? WHERE user_id = ?",
                (value, time.time(), user_id)
            )
        except Exception as e:
            logger.debug(f"更新认知画像失败: {e}")

    def add_like(self, user_id: str, item: str):
        prefs = self.get_preferences(user_id)
        likes = prefs["likes"]
        if item not in likes:
            likes.append(item)
            self.update_preference(user_id, "likes", likes)

    def add_dislike(self, user_id: str, item: str):
        prefs = self.get_preferences(user_id)
        dislikes = prefs["dislikes"]
        if item not in dislikes:
            dislikes.append(item)
            self.update_preference(user_id, "dislikes", dislikes)

    def set_habits(self, user_id: str, habits: str):
        self.update_preference(user_id, "habits", habits)

    def add_tag(self, user_id: str, tag: str):
        prefs = self.get_preferences(user_id)
        tags = prefs["tags"]
        if tag not in tags:
            tags.append(tag)
            self.update_preference(user_id, "persona_tags", ",".join(tags))

    def add_fragment(self, user_id: str, content: str, importance: float = 0.5):
        if not self._ensure_table():
            return
        try:
            from src.common.database.database_model import db
            db.execute_sql(
                "INSERT INTO user_long_fragments (user_id, content, importance, created_at) VALUES (?, ?, ?, ?)",
                (user_id, content, importance, time.time())
            )
            db.execute_sql("""
                DELETE FROM user_long_fragments WHERE id IN (
                    SELECT id FROM user_long_fragments WHERE user_id = ?
                    ORDER BY importance ASC, created_at ASC
                    LIMIT MAX(0, (SELECT COUNT(*) FROM user_long_fragments WHERE user_id = ?) - 50)
                )
            """, (user_id, user_id))
        except Exception as e:
            logger.debug(f"添加记忆碎片失败: {e}")

    def get_fragments(self, user_id: str, limit: int = 10) -> List[Dict]:
        if not self._ensure_table():
            return []
        try:
            from src.common.database.database_model import db
            cursor = db.execute_sql(
                "SELECT content, importance, created_at FROM user_long_fragments WHERE user_id = ? ORDER BY importance DESC, created_at DESC LIMIT ?",
                (user_id, limit)
            )
            return [{"content": row[0], "importance": row[1], "created_at": row[2]} for row in cursor.fetchall()]
        except Exception as e:
            logger.debug(f"获取记忆碎片失败: {e}")
            return []

    def build_cognitive_prompt(self, user_id: str) -> str:
        prefs = self.get_preferences(user_id)
        lines = []
        if prefs["likes"]:
            lines.append(f"喜欢: {', '.join(prefs['likes'][:8])}")
        if prefs["dislikes"]:
            lines.append(f"讨厌/禁忌: {', '.join(prefs['dislikes'][:8])}")
        if prefs["habits"]:
            lines.append(f"习惯: {prefs['habits'][:100]}")
        if prefs["tags"]:
            lines.append(f"性格标签: {', '.join(prefs['tags'][:6])}")
        fragments = self.get_fragments(user_id, limit=5)
        if fragments:
            lines.append("重要记忆碎片:")
            for frag in fragments:
                lines.append(f"  - {frag['content'][:80]}")
        if not lines:
            return ""
        return "【用户认知画像】\n" + "\n".join(lines)


_cognitive_store: Optional[UserCognitiveStore] = None


def get_cognitive_store() -> UserCognitiveStore:
    global _cognitive_store
    if _cognitive_store is None:
        _cognitive_store = UserCognitiveStore()
    return _cognitive_store

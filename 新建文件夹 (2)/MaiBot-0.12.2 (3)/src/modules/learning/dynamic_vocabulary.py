import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import aiosqlite
import difflib
from src.common.logger import get_logger
import re

logger = get_logger("vocab_learner")


class ExpressionStyleStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = asyncio.Lock()

    async def init_table(self):
        async with self._lock:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS expression_style (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id TEXT NOT NULL,
                        situation TEXT NOT NULL,
                        style TEXT NOT NULL,
                        context TEXT,
                        content_list TEXT,
                        style_list TEXT,
                        count INTEGER DEFAULT 1,
                        last_active_time REAL,
                        create_time REAL,
                        checked INTEGER DEFAULT 0,
                        rejected INTEGER DEFAULT 0
                    )
                """)
                await db.execute("CREATE INDEX IF NOT EXISTS idx_style_chat ON expression_style(chat_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_style_situation ON expression_style(situation)")
                await db.commit()

    async def upsert_expression(
        self,
        chat_id: str,
        situation: str,
        style: str,
        context: str,
        current_time: float,
    ) -> Tuple[Optional[int], bool]:
        async with self._lock:
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute(
                    "SELECT id, count, content_list, style_list FROM expression_style WHERE chat_id = ? AND style = ?",
                    (chat_id, style)
                )
                row = await cursor.fetchone()
                if row:
                    exp_id = row[0]
                    new_count = row[1] + 1
                    content_list = json.loads(row[2]) if row[2] else []
                    if situation not in content_list:
                        content_list.append(situation)
                    if len(content_list) > 10:
                        content_list = content_list[-10:]
                    await db.execute(
                        """UPDATE expression_style SET count = ?, content_list = ?,
                        last_active_time = ?, checked = 0 WHERE id = ?""",
                        (new_count, json.dumps(content_list, ensure_ascii=False), current_time, exp_id)
                    )
                    await db.commit()
                    return exp_id, False
                await db.execute(
                    """INSERT INTO expression_style
                    (chat_id, situation, style, context, content_list, style_list, count, last_active_time, create_time)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                    (chat_id, situation, style, context, json.dumps([situation], ensure_ascii=False),
                     json.dumps([style], ensure_ascii=False), current_time, current_time)
                )
                await db.commit()
                cursor = await db.execute("SELECT last_insert_rowid()")
                result = await cursor.fetchone()
                if result is not None:
                    return result[0], True
                return None, False

    async def calculate_similarity(self, style1: str, style2: str, threshold: float = 0.7) -> float:
        clean1 = style1.replace("使用", "").replace("句式", "").strip()
        clean2 = style2.replace("使用", "").replace("句式", "").strip()
        return difflib.SequenceMatcher(None, clean1, clean2).ratio()

    async def find_similar_style(self, chat_id: str, style: str, threshold: float = 0.7) -> Optional[Dict]:
        async with aiosqlite.connect(str(self.db_path)) as db:
            cursor = await db.execute(
                "SELECT id, situation, style, context, count FROM expression_style WHERE chat_id = ? AND rejected = 0",
                (chat_id,)
            )
            rows = await cursor.fetchall()
            clean_style = style.replace("使用", "").replace("句式", "").strip()
            for row in rows:
                row_style = row[2]
                clean_row = row_style.replace("使用", "").replace("句式", "").strip()
                similarity = difflib.SequenceMatcher(None, clean_style, clean_row).ratio()
                if similarity >= threshold:
                    return {"id": row[0], "situation": row[1], "style": row_style, "context": row[3], "count": row[4]}
        return None

    async def get_expressions_for_chat(self, chat_id: str, min_count: int = 2, limit: int = 20) -> List[Dict]:
        async with aiosqlite.connect(str(self.db_path)) as db:
            cursor = await db.execute(
                """SELECT id, situation, style, context, count, last_active_time
                FROM expression_style WHERE chat_id = ? AND count >= ? AND rejected = 0
                ORDER BY count DESC LIMIT ?""",
                (chat_id, min_count, limit)
            )
            rows = await cursor.fetchall()
            return [
                {"id": row[0], "situation": row[1], "style": row[2], "context": row[3], "count": row[4], "last_active": row[5]}
                for row in rows
            ]

    async def update_active_time(self, exp_ids: List[int], current_time: float):
        if not exp_ids:
            return
        async with self._lock:
            async with aiosqlite.connect(str(self.db_path)) as db:
                placeholders = ",".join("?" * len(exp_ids))
                await db.execute(
                    f"UPDATE expression_style SET last_active_time = ? WHERE id IN ({placeholders})",
                    [current_time] + exp_ids
                )
                await db.commit()

    async def get_random_expressions(self, chat_id: str, total: int = 5) -> List[Dict]:
        async with aiosqlite.connect(str(self.db_path)) as db:
            cursor = await db.execute(
                """SELECT id, situation, style, context, count, checked
                FROM expression_style WHERE chat_id = ? AND rejected = 0 ORDER BY RANDOM() LIMIT ?""",
                (chat_id, total * 3)
            )
            rows = await cursor.fetchall()
            weighted = []
            for row in rows:
                weight = min(row[4], 5)
                if row[5]:
                    weight *= 3
                weighted.append({
                    "id": row[0], "situation": row[1], "style": row[2],
                    "context": row[3], "count": row[4], "weight": weight,
                })
            selected = []
            for item in weighted:
                if len(selected) >= total:
                    break
                if item["weight"] >= 3 or (item["count"] > 2 and item["weight"] >= 2):
                    selected.append(item)
            if len(selected) < total:
                for item in weighted:
                    if len(selected) >= total:
                        break
                    if item not in selected:
                        selected.append(item)
            return selected[:total]


class JargonStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = asyncio.Lock()

    async def init_table(self):
        async with self._lock:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS jargon_dict (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        raw_content TEXT,
                        meaning TEXT,
                        chat_ids TEXT,
                        is_global INTEGER DEFAULT 0,
                        count INTEGER DEFAULT 0,
                        is_jargon INTEGER,
                        inference_count INTEGER DEFAULT 0,
                        is_complete INTEGER DEFAULT 0,
                        create_time REAL,
                        update_time REAL
                    )
                """)
                await db.execute("CREATE INDEX IF NOT EXISTS idx_jargon_content ON jargon_dict(content)")
                await db.commit()

    async def upsert_jargon(self, content: str, raw_content: str, chat_id: str, current_time: float) -> int:
        async with self._lock:
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute(
                    "SELECT id, chat_ids, count FROM jargon_dict WHERE content = ?", (content,)
                )
                row = await cursor.fetchone()
                if row:
                    jargon_id = row[0]
                    chat_ids = json.loads(row[1]) if row[1] else []
                    new_count = row[2] + 1
                    if chat_id not in chat_ids:
                        chat_ids.append(chat_id)
                    await db.execute(
                        """UPDATE jargon_dict SET count = ?, chat_ids = ?,
                        raw_content = ?, update_time = ?, inference_count = 0, is_complete = 0
                        WHERE id = ?""",
                        (new_count, json.dumps(chat_ids, ensure_ascii=False), raw_content, current_time, jargon_id)
                    )
                    await db.commit()
                    return jargon_id
                await db.execute(
                    """INSERT INTO jargon_dict
                    (content, raw_content, chat_ids, count, create_time, update_time)
                    VALUES (?, ?, ?, 1, ?, ?)""",
                    (content, raw_content, json.dumps([chat_id], ensure_ascii=False), current_time, current_time)
                )
                await db.commit()
                cursor = await db.execute("SELECT last_insert_rowid()")
                result = await cursor.fetchone()
                return result[0] if result is not None else 0

    async def get_unjargoned(self, limit: int = 10) -> List[Dict]:
        async with aiosqlite.connect(str(self.db_path)) as db:
            cursor = await db.execute(
                """SELECT id, content, raw_content, count FROM jargon_dict
                WHERE is_jargon IS NULL AND inference_count < 5 LIMIT ?""",
                (limit,)
            )
            rows = await cursor.fetchall()
            return [{"id": row[0], "content": row[1], "raw": row[2], "count": row[3]} for row in rows]

    async def update_jargon_inference(self, jargon_id: int, meaning: str, is_jargon: bool, current_time: float):
        async with self._lock:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(
                    """UPDATE jargon_dict SET meaning = ?, is_jargon = ?,
                    inference_count = inference_count + 1, is_complete = 1, update_time = ?
                    WHERE id = ?""",
                    (meaning, 1 if is_jargon else 0, current_time, jargon_id)
                )
                await db.commit()

    async def get_known_jargons(self) -> List[str]:
        async with aiosqlite.connect(str(self.db_path)) as db:
            cursor = await db.execute("SELECT content FROM jargon_dict WHERE is_jargon = 1")
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def match_from_text(self, text: str) -> List[str]:
        jargons = await self.get_known_jargons()
        return [j for j in jargons if j in text]


class DynamicVocabulary:
    def __init__(self):
        from src.config.config import global_config
        learn_cfg = global_config.learning
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        db_dir = project_root / "data"
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / "vocabulary.db"
        self.expression_store = ExpressionStyleStore(self.db_path)
        self.jargon_store = JargonStore(self.db_path)
        self.max_vocabulary = learn_cfg.max_vocabulary
        self.enable_expression = learn_cfg.enable_expression_learning
        self.enable_jargon = learn_cfg.enable_jargon_learning
        self.similarity_threshold = learn_cfg.expression_similarity_threshold
        self.expression_min_count = learn_cfg.expression_min_count
        self.expression_select_limit = learn_cfg.expression_select_limit
        self.jargon_inference_limit = learn_cfg.jargon_inference_limit
        self._db_initialized = False
        self._init_lock = None

    async def _init_db(self):
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._db_initialized:
                return
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiosqlite.connect(str(self.db_path)) as db:
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS vocabulary (
                            word TEXT PRIMARY KEY,
                            meaning TEXT,
                            context TEXT,
                            count INTEGER DEFAULT 1,
                            created_at INTEGER,
                            updated_at INTEGER
                        )
                    """)
                    await db.commit()
                await self.expression_store.init_table()
                await self.jargon_store.init_table()
                self._db_initialized = True
                logger.info(f"学习数据库已初始化: {self.db_path}")
            except Exception as e:
                logger.error(f"数据库初始化失败: {e}")

    async def _ensure_db_initialized(self):
        if not self._db_initialized:
            await self._init_db()

    def _get_llm(self):
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            return LLMRequest(model_config.lightweight, request_type="vocab_learning")
        except Exception:
            return None

    async def learn_from_messages(self, messages: List[Dict], chat_id: str, user_id: str = None) -> Dict:
        """从对话消息中学习表达方式和网络用语 - 核心学习引擎"""
        await self._ensure_db_initialized()
        llm = self._get_llm()
        if not llm:
            return {"expressions": 0, "jargons": 0}
        try:
            text_content = self._build_message_text(messages)
            
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.LEARNING,
                "dynamic_vocabulary",
                "learn_from_messages.template",
                text_content=text_content
            )
            
            result, _ = await llm.generate_response_async(prompt, max_tokens=1000, temperature=0.3)
            if not result:
                return {"expressions": 0, "jargons": 0}
            
            # 提取JSON
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.warning(f"JSON解析失败: {json_match.group()[:100]}")
                    data = {"expressions": [], "jargons": []}
            else:
                data = {"expressions": [], "jargons": []}
            
            current_time = time.time()
            exp_count = 0
            jargon_count = 0
            
            # 学习表达方式
            if self.enable_expression:
                for exp in data.get("expressions", []):
                    situation = exp.get("situation", "").strip()
                    style = exp.get("style", "").strip()
                    context = exp.get("context", "").strip()
                    
                    if situation and style and len(style) > 2:
                        await self.expression_store.upsert_expression(
                            chat_id=chat_id,
                            situation=situation,
                            style=style,
                            context=context,
                            current_time=current_time,
                        )
                        exp_count += 1
            
            # 学习黑话词汇
            if self.enable_jargon:
                for j in data.get("jargons", []):
                    word = j.get("word", "").strip()
                    context = j.get("context", "").strip()
                    
                    if word and len(word) >= 2 and len(word) <= 20:
                        await self.jargon_store.upsert_jargon(
                            content=word,
                            raw_content=context,
                            chat_id=chat_id,
                            current_time=current_time,
                        )
                        jargon_count += 1
            
            if exp_count > 0 or jargon_count > 0:
                logger.info(f"📚 学习完成: 表达{exp_count} 黑话{jargon_count}")
            
            return {"expressions": exp_count, "jargons": jargon_count}
        except Exception as e:
            logger.error(f"学习失败: {e}")
            return {"expressions": 0, "jargons": 0}

    def _build_message_text(self, messages: List[Dict]) -> str:
        parts = []
        for msg in messages[-20:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                parts.append(f"{role}: {content}")
        return "\n".join(parts)

    async def infer_jargon_meanings(self) -> int:
        await self._ensure_db_initialized()
        llm = self._get_llm()
        if not llm:
            return 0
        unjargoned = await self.jargon_store.get_unjargoned(limit=self.jargon_inference_limit)
        if not unjargoned:
            return 0
        count = 0
        current_time = time.time()
        for item in unjargoned:
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.LEARNING,
                "dynamic_vocabulary",
                "infer_jargon_meanings.template",
                content=item['content'],
                raw=item['raw']
            )
            try:
                result, _ = await llm.generate_response_async(prompt, max_tokens=150, temperature=0.3)
                if not result:
                    continue
                import re
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    meaning = data.get("meaning", "")
                    is_jargon = data.get("is_jargon", False)
                else:
                    meaning = result.strip()
                    is_jargon = True
                await self.jargon_store.update_jargon_inference(
                    jargon_id=item["id"], meaning=meaning,
                    is_jargon=is_jargon, current_time=current_time,
                )
                count += 1
            except Exception as e:
                logger.debug(f"推断黑话失败: {item['content'][:10]} - {e}")
        if count > 0:
            logger.info(f"推断黑话含义: {count}个")
        return count

    async def select_expressions(self, chat_id: str, chat_info: str, max_num: int = 5) -> List[Dict]:
        await self._ensure_db_initialized()
        expressions = await self.expression_store.get_expressions_for_chat(
            chat_id=chat_id, min_count=self.expression_min_count, limit=max_num * 3,
        )
        if not expressions:
            return []
        if len(expressions) <= max_num:
            return expressions
        llm = self._get_llm()
        if not llm:
            return expressions[:max_num]
        try:
            exp_text = "\n".join([
                f"{i+1}. 情境:{e['situation']} → 表达:{e['style']}"
                for i, e in enumerate(expressions[:10])
            ])
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.LEARNING,
                "dynamic_vocabulary",
                "select_expressions.template",
                chat_info=chat_info,
                max_num=max_num,
                exp_text=exp_text
            )
            result, _ = await llm.generate_response_async(prompt, max_tokens=60, temperature=0.3)
            if not result:
                return expressions[:max_num]
            selected = []
            for part in result.replace(" ", "").split(","):
                try:
                    idx = int(part) - 1
                    if 0 <= idx < len(expressions) and expressions[idx] not in selected:
                        selected.append(expressions[idx])
                except ValueError:
                    continue
            if selected:
                current_time = time.time()
                await self.expression_store.update_active_time([e["id"] for e in selected], current_time)
                return selected[:max_num]
        except Exception as e:
            logger.debug(f"选择表达失败: {e}")
        return expressions[:max_num]

    async def explain_jargons_in_context(self, messages: List[Dict]) -> str:
        await self._ensure_db_initialized()
        text = self._build_message_text(messages)
        matched = await self.jargon_store.match_from_text(text)
        if not matched:
            return ""
        explanations = []
        for word in matched[:5]:
            meaning = await self.get_word_meaning(word)
            if meaning:
                explanations.append(f"{word}: {meaning}")
        if explanations:
            return "黑话解释: " + "；".join(explanations)
        return ""

    async def is_new_slang(self, word: str) -> bool:
        if len(word) < 2 or len(word) > 10:
            return False
        await self._ensure_db_initialized()
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute("SELECT word FROM vocabulary WHERE word = ?", (word,))
                if await cursor.fetchone():
                    return False
        except Exception:
            pass
        llm = self._get_llm()
        if not llm:
            return False
        try:
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.LEARNING,
                "dynamic_vocabulary",
                "is_new_slang.template",
                word=word
            )
            result, _ = await llm.generate_response_async(prompt, max_tokens=10, temperature=0.3)
            if result:
                return "是" in result or "yes" in result.lower()
        except Exception:
            pass
        return False

    async def learn_word(self, word: str, context: str) -> Optional[str]:
        await self._ensure_db_initialized()
        llm = self._get_llm()
        if not llm:
            return None
        try:
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.LEARNING,
                "dynamic_vocabulary",
                "learn_word.template",
                word=word,
                context=context
            )
            meaning, _ = await llm.generate_response_async(prompt, max_tokens=100, temperature=0.3)
            if not meaning:
                return None
            await self._check_vocabulary_limit()
            current_time = int(time.time())
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute("SELECT count FROM vocabulary WHERE word = ?", (word,))
                existing = await cursor.fetchone()
                if existing:
                    await db.execute(
                        "UPDATE vocabulary SET count = count + 1, updated_at = ?, context = ? WHERE word = ?",
                        (current_time, context, word)
                    )
                else:
                    await db.execute(
                        "INSERT INTO vocabulary (word, meaning, context, count, created_at, updated_at) VALUES (?, ?, ?, 1, ?, ?)",
                        (word, meaning, context, current_time, current_time)
                    )
                await db.commit()
            logger.debug(f"📖 新词: {word[:15]}...")
            return meaning
        except Exception as e:
            logger.error(f"学习新词失败: {e}")
            return None

    async def get_word(self, word: str) -> Optional[Dict]:
        await self._ensure_db_initialized()
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute(
                    "SELECT word, meaning, context, count, created_at, updated_at FROM vocabulary WHERE word = ?", (word,)
                )
                result = await cursor.fetchone()
                if result:
                    return {"word": result[0], "meaning": result[1], "context": result[2],
                            "count": result[3], "created_at": result[4], "updated_at": result[5]}
        except Exception as e:
            logger.error(f"查询词条失败: {e}")
        return None

    async def get_word_meaning(self, word: str) -> Optional[str]:
        word_data = await self.get_word(word)
        return word_data.get('meaning') if word_data else None

    async def search_words(self, query: str, limit: int = 3) -> List[Dict]:
        await self._ensure_db_initialized()
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute(
                    """SELECT word, meaning, context, count FROM vocabulary
                    WHERE word LIKE ? OR context LIKE ? OR meaning LIKE ?
                    ORDER BY count DESC, updated_at DESC LIMIT ?""",
                    (f"%{query}%", f"%{query}%", f"%{query}%", limit)
                )
                results = await cursor.fetchall()
                return [{"word": r[0], "meaning": r[1], "context": r[2], "count": r[3]} for r in results]
        except Exception as e:
            logger.error(f"搜索词条失败: {e}")
            return []

    async def _check_vocabulary_limit(self):
        await self._ensure_db_initialized()
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM vocabulary")
                result = await cursor.fetchone()
                count = result[0] if result is not None else 0
                if count >= self.max_vocabulary:
                    delete_count = max(1, self.max_vocabulary // 10)
                    await db.execute(f"""
                        DELETE FROM vocabulary WHERE word IN (
                            SELECT word FROM vocabulary ORDER BY count ASC, updated_at ASC LIMIT {delete_count}
                        )
                    """)
                    await db.commit()
                    logger.info(f"清理旧词: 删除{delete_count}个")
        except Exception as e:
            logger.error(f"清理词汇失败: {e}")

    async def get_vocabulary_stats(self) -> Dict:
        await self._ensure_db_initialized()
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM vocabulary")
                result = await cursor.fetchone()
                total = result[0] if result is not None else 0
                cursor = await db.execute("SELECT word, count FROM vocabulary ORDER BY count DESC LIMIT 5")
                top_words = await cursor.fetchall()
                cursor = await db.execute("SELECT COUNT(*) FROM expression_style")
                result = await cursor.fetchone()
                exp_count = result[0] if result is not None else 0
                cursor = await db.execute("SELECT COUNT(*) FROM jargon_dict")
                result = await cursor.fetchone()
                jargon_count = result[0] if result is not None else 0
                return {
                    "total": total, "max_vocabulary": self.max_vocabulary,
                    "top_words": [{"word": w[0], "count": w[1]} for w in top_words],
                    "expressions": exp_count, "jargons": jargon_count,
                }
        except Exception:
            return {"total": 0, "max_vocabulary": self.max_vocabulary, "top_words": [], "expressions": 0, "jargons": 0}

    async def get_user_learning_state(self, chat_id: str, user_id: str = None) -> str:
        """获取用户的动态学习状态，生成认知提示"""
        await self._ensure_db_initialized()
        try:
            expressions = []
            if self.enable_expression and chat_id:
                expressions = await self.expression_store.get_expressions_for_chat(chat_id=chat_id, min_count=1, limit=5)
            jargons_list = []
            if self.enable_jargon:
                async with aiosqlite.connect(str(self.db_path)) as db:
                    cursor = await db.execute(
                        "SELECT content, meaning FROM jargon_dict WHERE is_jargon = 1 AND is_complete = 1 LIMIT 5"
                    )
                    rows = await cursor.fetchall()
                    jargons_list = [{"word": r[0], "meaning": r[1]} for r in rows]
            
            # 如果没有学到任何内容，保持天真无邪状态
            from src.config.prompt_loader import get_prompt_raw, get_prompt, PromptCategory
            if not expressions and not jargons_list:
                return get_prompt_raw(PromptCategory.LEARNING, "dynamic_vocabulary", "get_user_learning_state.template_naive")

            prompt = get_prompt_raw(PromptCategory.LEARNING, "dynamic_vocabulary", "get_user_learning_state.template_state_header")
            if expressions:
                learned_exps = "".join([f"- {exp.get('situation', '')}时，使用「{exp.get('style', '')}」\n" for exp in expressions[:3] if exp.get('situation') and exp.get('style')])
                prompt += get_prompt(PromptCategory.LEARNING, "dynamic_vocabulary", "get_user_learning_state.template_expressions_learned", learned_expressions=learned_exps)

            if jargons_list:
                learned_jargons = "".join([f"- {j.get('word', '')}：{j.get('meaning', '')}\n" if j.get('meaning') else f"- {j.get('word', '')}\n" for j in jargons_list[:3] if j.get('word')])
                prompt += get_prompt(PromptCategory.LEARNING, "dynamic_vocabulary", "get_user_learning_state.template_jargons_learned", learned_jargons=learned_jargons)
            
            # 添加学习态度描述
            prompt += get_prompt_raw(PromptCategory.LEARNING, "dynamic_vocabulary", "get_user_learning_state.template_learning_attitude")
            return prompt
        except Exception as e:
            logger.error(f"获取学习状态失败: {e}")
            from src.config.prompt_loader import get_prompt_raw, PromptCategory
            return get_prompt_raw(PromptCategory.LEARNING, "dynamic_vocabulary", "get_user_learning_state.template_naive")


_global_dynamic_vocabulary: Optional[DynamicVocabulary] = None


def get_dynamic_vocabulary() -> DynamicVocabulary:
    global _global_dynamic_vocabulary
    if _global_dynamic_vocabulary is None:
        _global_dynamic_vocabulary = DynamicVocabulary()
    return _global_dynamic_vocabulary

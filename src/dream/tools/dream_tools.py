import json
import time
from typing import List, Optional
from src.common.logger import get_logger

logger = get_logger("梦境工具")


def parse_clue_string(raw_text: str) -> List[str]:
    if not raw_text:
        return []
    keywords = []
    for kw in (
        raw_text.replace(",", " ")
        .replace("，", " ")
        .replace("、", " ")
        .split()
    ):
        kw = kw.strip()
        if kw and len(kw) >= 2:
            keywords.append(kw)
    return keywords


def parse_keywords_string(keyword_str: str) -> List[str]:
    return parse_clue_string(keyword_str)


def _extract_record_clues(record) -> List[str]:
    if not record.keywords:
        return []
    try:
        kw_data = (
            json.loads(record.keywords)
            if isinstance(record.keywords, str)
            else record.keywords
        )
        if isinstance(kw_data, list):
            return [str(k).strip() for k in kw_data if str(k).strip()]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    return []


def create_history_search_handler(channel_id: str):
    async def history_search(
        keyword: Optional[str] = None,
        participant: Optional[str] = None,
        clue: Optional[str] = None,
    ) -> str:
        try:
            actual_clue = clue if clue is not None else keyword
            if not actual_clue and not participant:
                return "未指定查询参数（需提供clue或participant之一）"
            logger.info(
                f"[梦境工具] history_search(clue={actual_clue}, participant={participant})"
            )
            from src.common.database.database_model import ChatHistory

            query = ChatHistory.select().where(
                ChatHistory.chat_id == channel_id
            )
            records = list(
                query.order_by(ChatHistory.start_time.desc()).limit(50)
            )
            matched_records: List = []
            for record in records:
                participant_ok = True
                keyword_ok = True
                if participant:
                    participant_ok = False
                    participant_list: List[str] = []
                    if record.participants:
                        try:
                            parsed_participants = (
                                json.loads(record.participants)
                                if isinstance(record.participants, str)
                                else record.participants
                            )
                            if isinstance(parsed_participants, list):
                                participant_list = [
                                    str(p).lower() for p in parsed_participants
                                ]
                        except (json.JSONDecodeError, TypeError, ValueError):
                            participant_list = []
                    search_participant = participant.lower().strip()
                    if search_participant and any(
                        search_participant in p for p in participant_list
                    ):
                        participant_ok = True
                if actual_clue:
                    keyword_ok = False
                    search_keywords = parse_clue_string(actual_clue)
                    if not search_keywords:
                        search_keywords = (
                            [actual_clue.strip()]
                            if actual_clue.strip()
                            else []
                        )
                    search_keywords_lower = [
                        kw.lower() for kw in search_keywords if kw.strip()
                    ]
                    if search_keywords_lower:
                        theme_text = (record.theme or "").lower()
                        summary_text = (record.summary or "").lower()
                        original_text = (record.original_text or "").lower()
                        record_keyword_list = [
                            item.lower()
                            for item in _extract_record_clues(record)
                        ]
                        hit_count = 0
                        for kw in search_keywords_lower:
                            kw_hit = (
                                kw in theme_text
                                or kw in summary_text
                                or kw in original_text
                                or any(kw in k for k in record_keyword_list)
                            )
                            if kw_hit:
                                hit_count += 1
                        total_keywords = len(search_keywords_lower)
                        required_hits = (
                            total_keywords - 1
                            if total_keywords > 2
                            else total_keywords
                        )
                        keyword_ok = hit_count >= required_hits
                if participant_ok and keyword_ok:
                    matched_records.append(record)
            if not matched_records:
                return f"未找到包含线索'{actual_clue}'或参与者'{participant}'的记录"
            if len(matched_records) > 20:
                all_keywords_set = set()
                for record in matched_records:
                    for k_str in _extract_record_clues(record):
                        if k_str:
                            all_keywords_set.add(k_str)
                search_label = actual_clue or participant or "当前条件"
                if all_keywords_set:
                    kw_str = "、".join(sorted(all_keywords_set))
                    return f'包含"{search_label} "的结果过多，请添加更多线索精确查找\n\n可用标签：\n{
                        kw_str} '
                return (
                    f'包含"{search_label}"的结果过多，请添加更多线索精确查找'
                )
            result_items: List[str] = []
            for record in matched_records[:20]:
                item_lines: List[str] = []
                item_lines.append(f"记录ID：{record.id}")
                item_lines.append(f"主题：{record.theme or '（无）'}")
                record_clues = _extract_record_clues(record)
                if record_clues:
                    kw_str = "、".join(record_clues)
                    item_lines.append(f"线索标签：{kw_str}")
                result_items.append("\n".join(item_lines))
            if not result_items:
                return "未找到相关记录"
            response = "\n\n---\n\n".join(result_items)
            logger.info(
                f"[梦境工具] history_search 返回 {len(matched_records)} 条记录"
            )
            return response
        except Exception as exc:
            logger.error(f"history_search 失败: {exc}")
            return f"history_search 执行失败: {exc}"

    return history_search


def create_history_detail_handler(channel_id: str):
    async def history_detail(record_id: int) -> str:
        try:
            logger.info(f"[梦境工具] history_detail(record_id={record_id})")
            from src.common.database.database_model import ChatHistory

            record = ChatHistory.get_or_none(
                (ChatHistory.id == record_id)
                & (ChatHistory.chat_id == channel_id)
            )
            if not record:
                return f"未找到ID为 {record_id} 的记录（或不属于当前频道）"
            lines = [
                f"记录ID：{record.id}",
                f"主题：{record.theme or '（无）'}",
                f"摘要：{record.summary or '（无）'}",
                f"线索标签：{record.keywords or '（无）'}",
                f"起始时间：{record.start_time}",
                f"结束时间：{record.end_time}",
                f"检索次数：{record.count}",
            ]
            if record.original_text:
                original_preview = record.original_text[:500]
                if len(record.original_text) > 500:
                    original_preview += "..."
                lines.append(f"原文预览：\n{original_preview}")
            return "\n".join(lines)
        except Exception as exc:
            logger.error(f"history_detail 失败: {exc}")
            return f"history_detail 执行失败: {exc}"

    return history_detail


def create_history_remove_handler(channel_id: str):
    async def history_remove(record_id: int) -> str:
        try:
            logger.info(f"[梦境工具] history_remove(record_id={record_id})")
            from src.common.database.database_model import ChatHistory

            record = ChatHistory.get_or_none(
                (ChatHistory.id == record_id)
                & (ChatHistory.chat_id == channel_id)
            )
            if not record:
                return f"未找到ID为 {record_id} 的记录（或不属于当前频道）"
            theme_backup = record.theme or "无主题"
            record.delete_instance()
            msg = f"已删除记录 ID={record_id}，原主题：{theme_backup}"
            logger.info(f"[梦境工具] history_remove 完成: {msg}")
            return msg
        except Exception as exc:
            logger.error(f"history_remove 失败: {exc}")
            return f"history_remove 执行失败: {exc}"

    return history_remove


def create_history_create_handler(channel_id: str):
    async def history_create(
        theme: str,
        summary: str,
        keywords: str,
        begin_ts: float,
        finish_ts: float,
    ) -> str:
        try:
            logger.info(
                f"[梦境工具] history_create(theme={
                    bool(theme)}, summary={
                    bool(summary)})"
            )
            from src.common.database.database_model import ChatHistory

            current_ts = time.time()
            parsed_begin = begin_ts if begin_ts else current_ts
            parsed_finish = finish_ts if finish_ts else current_ts
            new_record = ChatHistory.create(
                chat_id=channel_id,
                theme=theme,
                summary=summary,
                keywords=keywords,
                start_time=parsed_begin,
                end_time=parsed_finish,
                original_text="",
                participants="[]",
            )
            msg = f"已创建新记录，ID={new_record.id}，主题={new_record.theme or '无'}"
            logger.info(f"[梦境工具] history_create 完成: {msg}")
            return msg
        except Exception as exc:
            logger.error(f"history_create 失败: {exc}")
            return f"history_create 执行失败: {exc}"

    return history_create


def create_history_modify_handler(channel_id: str):
    async def history_modify(
        record_id: int,
        theme: Optional[str] = None,
        summary: Optional[str] = None,
        keywords: Optional[str] = None,
    ) -> str:
        try:
            logger.info(f"[梦境工具] history_modify(record_id={record_id})")
            from src.common.database.database_model import ChatHistory

            record = ChatHistory.get_or_none(
                (ChatHistory.id == record_id)
                & (ChatHistory.chat_id == channel_id)
            )
            if not record:
                return f"未找到ID为 {record_id} 的记录（或不属于当前频道）"
            updated_fields = []
            if theme is not None:
                record.theme = theme
                updated_fields.append("主题")
            if summary is not None:
                record.summary = summary
                updated_fields.append("摘要")
            if keywords is not None:
                record.keywords = keywords
                updated_fields.append("关键词")
            if not updated_fields:
                return f"记录 ID={record_id} 未指定任何要更新的字段"
            record.save()
            msg = f"已更新记录 ID={record_id}，修改字段：{', '.join(updated_fields)}"
            logger.info(f"[梦境工具] history_modify 完成: {msg}")
            return msg
        except Exception as exc:
            logger.error(f"history_modify 失败: {exc}")
            return f"history_modify 执行失败: {exc}"

    return history_modify


def create_slang_search_handler(channel_id: str):
    async def slang_search(keyword: str) -> str:
        try:
            logger.info(f"[梦境工具] slang_search(keyword={keyword})")
            from src.common.database.database_model import Jargon

            if not keyword or not keyword.strip():
                return "未指定检索关键词"
            search_terms = [
                kw.strip().lower()
                for kw in keyword.replace(",", " ").replace("，", " ").split()
                if kw.strip()
            ]
            if not search_terms:
                return "未指定有效的检索关键词"
            records = (
                Jargon.select()
                .where((Jargon.chat_id == channel_id) | Jargon.is_global)
                .limit(100)
            )
            matched_entries: List = []
            for record in records:
                content_text = str(record.content or "").lower()
                meaning_text = str(record.meaning or "").lower()
                hit = False
                for term in search_terms:
                    if term in content_text or term in meaning_text:
                        hit = True
                        break
                if hit:
                    matched_entries.append(record)
            if not matched_entries:
                return f"未找到与关键词'{keyword}'相关的术语记录"
            result_items: List[str] = []
            for entry in matched_entries[:15]:
                item_lines = [
                    f"术语ID：{entry.id}",
                    f"内容：{entry.content or '无'}",
                    f"含义：{entry.meaning or '无'}",
                    f"是否全局：{'是' if entry.is_global else '否'}",
                ]
                result_items.append("\n".join(item_lines))
            response = "\n\n---\n\n".join(result_items)
            logger.info(
                f"[梦境工具] slang_search 返回 {len(matched_entries)} 条记录"
            )
            return response
        except Exception as exc:
            logger.error(f"slang_search 失败: {exc}")
            return f"slang_search 执行失败: {exc}"

    return slang_search


def create_slang_remove_handler(channel_id: str):
    async def slang_remove(entry_id: int) -> str:
        try:
            logger.info(f"[梦境工具] slang_remove(entry_id={entry_id})")
            from src.common.database.database_model import Jargon

            record = Jargon.get_or_none(
                (Jargon.id == entry_id) & (Jargon.chat_id == channel_id)
            )
            if not record:
                return f"未找到ID为 {entry_id} 的术语记录（或不属于当前频道）"
            content_backup = record.content or "无内容"
            record.delete_instance()
            msg = f"已删除术语记录 ID={entry_id}，原内容：{content_backup}"
            logger.info(f"[梦境工具] slang_remove 完成: {msg}")
            return msg
        except Exception as exc:
            logger.error(f"slang_remove 失败: {exc}")
            return f"slang_remove 执行失败: {exc}"

    return slang_remove


def create_slang_modify_handler(channel_id: str):
    async def slang_modify(
        entry_id: int,
        meaning: Optional[str] = None,
    ) -> str:
        try:
            logger.info(f"[梦境工具] slang_modify(entry_id={entry_id})")
            from src.common.database.database_model import Jargon

            record = Jargon.get_or_none(
                (Jargon.id == entry_id) & (Jargon.chat_id == channel_id)
            )
            if not record:
                return f"未找到ID为 {entry_id} 的术语记录（或不属于当前频道）"
            updated_fields = []
            if meaning is not None:
                record.meaning = meaning
                updated_fields.append("含义")
            if not updated_fields:
                return f"术语记录 ID={entry_id} 未指定任何要更新的字段"
            record.save()
            msg = f"已更新术语记录 ID={entry_id}，修改字段：{', '.join(updated_fields)}"
            logger.info(f"[梦境工具] slang_modify 完成: {msg}")
            return msg
        except Exception as exc:
            logger.error(f"slang_modify 失败: {exc}")
            return f"slang_modify 执行失败: {exc}"

    return slang_modify


def create_session_complete_handler(channel_id: str):
    async def session_complete(summary: str, actions_taken: str) -> str:
        try:
            logger.info(
                f"[梦境工具] session_complete(summary={bool(summary)})"
            )
            from src.common.database.database_model import PersistenceSlot

            slot, _ = PersistenceSlot.get_or_create(
                slot_key=f"dream_session_{channel_id}_{int(time.time())}",
                defaults={
                    "slot_value": json.dumps(
                        {
                            "summary": summary,
                            "actions_taken": actions_taken,
                            "channel_id": channel_id,
                            "completed_at": time.time(),
                        },
                        ensure_ascii=False,
                    ),
                    "birth_ts": time.time(),
                    "modify_ts": time.time(),
                    "ttl_days": 30,
                },
            )
            msg = f"已记录维护会话完成，摘要：{summary[:100]}..."
            logger.info("[梦境工具] session_complete 完成")
            return msg
        except Exception as exc:
            logger.error(f"session_complete 失败: {exc}")
            return f"session_complete 执行失败: {exc}"

    return session_complete


def create_memory_consolidate_handler(channel_id: str):
    async def memory_consolidate(memory_type: str, limit: int = 10) -> str:
        try:
            logger.info(
                f"[梦境工具] memory_consolidate(type={memory_type}, limit={limit})"
            )
            from src.common.database.database_model import (
                MemoryRecord,
                HippocampusEntry,
            )

            consolidated_count = 0
            if memory_type == "hippocampus":
                entries = (
                    HippocampusEntry.select()
                    .where(HippocampusEntry.stream_id == channel_id)
                    .order_by(HippocampusEntry.birth_ts.desc())
                    .limit(limit)
                )
                for entry in entries:
                    try:
                        MemoryRecord.create(
                            record_id=f"consolidated_{entry.memory_id}",
                            stream_id=entry.stream_id,
                            user_id=entry.user_id,
                            entry_category="consolidated",
                            content=entry.content,
                            significance=entry.significance,
                            birth_ts=entry.birth_ts,
                            last_visit_ts=entry.last_touch_ts,
                            storage_tier=entry.layer_index + 1,
                            sharpness=entry.sharpness,
                            digest=entry.digest,
                        )
                        entry.delete_instance()
                        consolidated_count += 1
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
            msg = f"已整合 {consolidated_count} 条记忆记录"
            logger.info(f"[梦境工具] memory_consolidate 完成: {msg}")
            return msg
        except Exception as exc:
            logger.error(f"memory_consolidate 失败: {exc}")
            return f"memory_consolidate 执行失败: {exc}"

    return memory_consolidate


def create_emotion_balance_handler(channel_id: str):
    async def emotion_balance(user_id: Optional[str] = None) -> str:
        try:
            logger.info(f"[梦境工具] emotion_balance(user_id={user_id})")
            from src.modules.modcore.dynamic_persona.emotion_database import (
                get_emotion_database,
            )

            db = get_emotion_database(channel_id)
            stats = db.get_statistics()
            if user_id:
                state = db.load_user_state(user_id)
                if state:
                    return f"用户 {user_id} 情绪状态：好感度={
                        state['affection']:.1f}, 信任度={
                        state['trust_score']:.1f}"
                return f"未找到用户 {user_id} 的情绪记录"
            return f"频道情绪统计：总用户={
                stats['total_users']}, 平均好感度={
                stats['avg_affection']:.2f}"
        except Exception as exc:
            logger.error(f"emotion_balance 失败: {exc}")
            return f"emotion_balance 执行失败: {exc}"

    return emotion_balance


def create_style_refresh_handler(channel_id: str):
    async def style_refresh() -> str:
        try:
            logger.info("[梦境工具] style_refresh()")
            from src.express.style_absorber import get_style_absorber

            absorber = get_style_absorber(channel_id)
            styles = await absorber.fetch_recent_styles(limit=50)
            return f"已刷新风格缓存，共 {len(styles)} 条风格记录"
        except Exception as exc:
            logger.error(f"style_refresh 失败: {exc}")
            return f"style_refresh 执行失败: {exc}"

    return style_refresh


def create_relationship_update_handler(channel_id: str):
    async def relationship_update(
        user_id: str, delta: float, reason: str = ""
    ) -> str:
        try:
            logger.info(
                f"[梦境工具] relationship_update(user_id={user_id}, delta={delta})"
            )
            from src.modules.modcore.dynamic_persona.emotion_database import (
                get_emotion_database,
            )

            db = get_emotion_database(channel_id)
            success = db.update_affection(user_id, delta, reason)
            if success:
                state = db.load_user_state(user_id)
                new_affection = state["affection"] if state else 0
                return f"已更新用户 {user_id}  关系，变化={
                    delta: +.2f} ，当前好感度={
                    new_affection: .1f} "
            return f"更新用户 {user_id} 关系失败"
        except Exception as exc:
            logger.error(f"relationship_update 失败: {exc}")
            return f"relationship_update 执行失败: {exc}"

    return relationship_update


def get_all_dream_tools(channel_id: str) -> dict:
    return {
        "history_search": create_history_search_handler(channel_id),
        "history_detail": create_history_detail_handler(channel_id),
        "history_remove": create_history_remove_handler(channel_id),
        "history_create": create_history_create_handler(channel_id),
        "history_modify": create_history_modify_handler(channel_id),
        "slang_search": create_slang_search_handler(channel_id),
        "slang_remove": create_slang_remove_handler(channel_id),
        "slang_modify": create_slang_modify_handler(channel_id),
        "session_complete": create_session_complete_handler(channel_id),
        "memory_consolidate": create_memory_consolidate_handler(channel_id),
        "emotion_balance": create_emotion_balance_handler(channel_id),
        "style_refresh": create_style_refresh_handler(channel_id),
        "relationship_update": create_relationship_update_handler(channel_id),
    }


def get_dream_tool_definitions() -> List[dict]:
    return [
        {
            "name": "history_search",
            "description": "按关键词或参与者检索历史概要",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "兼容旧字段：搜索线索",
                    },
                    "clue": {"type": "string", "description": "搜索线索"},
                    "participant": {
                        "type": "string",
                        "description": "参与者名称",
                    },
                },
            },
        },
        {
            "name": "history_detail",
            "description": "获取历史记录详情",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "integer", "description": "记录ID"},
                },
                "required": ["record_id"],
            },
        },
        {
            "name": "history_remove",
            "description": "删除历史记录",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "integer", "description": "记录ID"},
                },
                "required": ["record_id"],
            },
        },
        {
            "name": "history_create",
            "description": "创建新的历史记录",
            "parameters": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string", "description": "主题"},
                    "summary": {"type": "string", "description": "摘要"},
                    "keywords": {
                        "type": "string",
                        "description": "线索标签（兼容旧字段名）",
                    },
                    "begin_ts": {
                        "type": "number",
                        "description": "起始时间戳",
                    },
                    "finish_ts": {
                        "type": "number",
                        "description": "结束时间戳",
                    },
                },
                "required": ["theme", "summary"],
            },
        },
        {
            "name": "history_modify",
            "description": "修改历史记录",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "integer", "description": "记录ID"},
                    "theme": {"type": "string", "description": "新主题"},
                    "summary": {"type": "string", "description": "新摘要"},
                    "keywords": {
                        "type": "string",
                        "description": "新的线索标签（兼容旧字段名）",
                    },
                },
                "required": ["record_id"],
            },
        },
        {
            "name": "slang_search",
            "description": "搜索术语/黑话",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "兼容旧字段：搜索线索",
                    },
                },
                "required": ["keyword"],
            },
        },
        {
            "name": "slang_remove",
            "description": "删除术语记录",
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {"type": "integer", "description": "术语ID"},
                },
                "required": ["entry_id"],
            },
        },
        {
            "name": "slang_modify",
            "description": "修改术语记录",
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {"type": "integer", "description": "术语ID"},
                    "meaning": {"type": "string", "description": "新含义"},
                },
                "required": ["entry_id"],
            },
        },
        {
            "name": "session_complete",
            "description": "标记维护会话完成",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "会话摘要"},
                    "actions_taken": {
                        "type": "string",
                        "description": "执行的操作",
                    },
                },
                "required": ["summary"],
            },
        },
        {
            "name": "memory_consolidate",
            "description": "整合记忆",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "description": "记忆类型",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "处理数量限制",
                    },
                },
            },
        },
        {
            "name": "emotion_balance",
            "description": "检查情绪平衡状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "用户ID（可选）",
                    },
                },
            },
        },
        {
            "name": "style_refresh",
            "description": "刷新风格缓存",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "relationship_update",
            "description": "更新用户关系值",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "用户ID"},
                    "delta": {"type": "number", "description": "变化值"},
                    "reason": {"type": "string", "description": "原因"},
                },
                "required": ["user_id", "delta"],
            },
        },
    ]

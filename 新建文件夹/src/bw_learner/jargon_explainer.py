import re
import time
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.common.database.database_model import Jargon
from src.llm_models.utils_model import LLMRequest
from src.config.config import model_config, global_config
from src.bw_learner.jargon_miner import search_jargon
from src.bw_learner.learner_utils import (
    is_bot_message,
    contains_bot_self_name,
    parse_chat_id_list,
    chat_id_list_contains,
)

logger = get_logger("jargon")

JARGON_SUMMARIZE_PROMPT = """上下文聊天内容:
{chat_context}

在上下文中提取到的黑话及其含义:
{jargon_explanations}

请根据上述信息，对黑话解释进行概括和整理。
- 如果上下文中有黑话出现，请简要说明这些黑话在上下文中的使用情况
- 将所有黑话解释整理成简洁、易读的一段话
- 输出格式要自然，适合作为回复参考信息
请输出概括后的黑话解释（直接输出一段平文本，不要标题，无特殊格式或markdown格式，不要使用JSON格式）："""


class JargonExplainer:
    """黑话解释器，用于在回复前识别和解释上下文中的黑话"""

    def __init__(self, chat_id: str) -> None:
        self.chat_id = chat_id
        self.llm = LLMRequest(
            model_set=model_config.model_task_config.tool_use,
            request_type="jargon.explain",
        )

    def match_jargon_from_messages(
        self, messages: List[Any]
    ) -> List[Dict[str, str]]:
        """通过直接匹配数据库中的jargon字符串来提取黑话"""
        start_time = time.time()
        if not messages:
            return []
        message_texts: List[str] = []
        for msg in messages:
            if is_bot_message(msg, self._get_bot_platforms()):
                continue
            msg_text = (
                getattr(msg, "display_message", None)
                or getattr(msg, "processed_plain_text", None)
                or ""
            ).strip()
            if msg_text:
                message_texts.append(msg_text)
        if not message_texts:
            return []
        combined_text = " ".join(message_texts)
        query = Jargon.select().where(
            (Jargon.meaning.is_null(False)) & (Jargon.meaning != "")
        )
        if global_config.expression.all_global_jargon:
            query = query.where(Jargon.is_global)
        query = query.order_by(Jargon.count.desc())
        matched_jargon: Dict[str, Dict[str, str]] = {}
        query_time = time.time()
        for jargon in query:
            content = jargon.content or ""
            if not content or not content.strip():
                continue
            if contains_bot_self_name(
                content,
                getattr(global_config.bot, "nickname", ""),
                getattr(global_config.bot, "alias_names", []),
            ):
                continue
            if not global_config.expression.all_global_jargon:
                if jargon.is_global:
                    pass
                else:
                    chat_id_list = parse_chat_id_list(jargon.chat_id)
                    if not chat_id_list_contains(chat_id_list, self.chat_id):
                        continue
            pattern = re.escape(content)
            if re.search(r"[\u4e00-\u9fff]", content):
                search_pattern = pattern
            else:
                search_pattern = r"\b" + pattern + r"\b"
            if re.search(search_pattern, combined_text, re.IGNORECASE):
                if content not in matched_jargon:
                    matched_jargon[content] = {"content": content}
        match_time = time.time()
        total_time = match_time - start_time
        logger.debug(
            f"黑话匹配完成: 总耗时 {total_time:.3f}s, 匹配到 {len(matched_jargon)} 个黑话"
        )
        return list(matched_jargon.values())

    def _get_bot_platforms(self) -> Dict[str, str]:
        """获取机器人平台账号映射"""
        from src.bw_learner.learner_utils import parse_platform_accounts

        bot_config = getattr(global_config, "bot", None)
        if not bot_config:
            return {}
        platform_accounts = {}
        try:
            platform_accounts = parse_platform_accounts(
                getattr(bot_config, "platforms", []) or []
            )
        except Exception:
            platform_accounts = {}
        bot_accounts: Dict[str, str] = {}
        qq_account = str(getattr(bot_config, "qq_account", "") or "").strip()
        if qq_account:
            bot_accounts["qq"] = qq_account
        telegram_account = str(
            getattr(bot_config, "telegram_account", "") or ""
        ).strip()
        if telegram_account:
            bot_accounts["telegram"] = telegram_account
        for plat, account in platform_accounts.items():
            if account and plat not in bot_accounts:
                bot_accounts[plat] = account
        return bot_accounts

    async def explain_jargon(
        self, messages: List[Any], chat_context: str
    ) -> Optional[str]:
        """解释上下文中的黑话"""
        if not messages:
            return None
        jargon_entries = self.match_jargon_from_messages(messages)
        if not jargon_entries:
            return None
        unique_jargon: Dict[str, Dict[str, str]] = {}
        for entry in jargon_entries:
            content = entry["content"]
            if content not in unique_jargon:
                unique_jargon[content] = entry
        jargon_list = list(unique_jargon.values())
        logger.info(
            f"从上下文中提取到 {len(jargon_list)} 个黑话: {[j['content'] for j in jargon_list]}"
        )
        jargon_explanations: List[str] = []
        for entry in jargon_list:
            content = entry["content"]
            if global_config.expression.all_global_jargon:
                results = search_jargon(
                    keyword=content,
                    chat_id=None,
                    limit=1,
                    case_sensitive=False,
                    fuzzy=False,
                )
            else:
                results = search_jargon(
                    keyword=content,
                    chat_id=self.chat_id,
                    limit=1,
                    case_sensitive=False,
                    fuzzy=False,
                )
            if results and len(results) > 0:
                meaning = results[0].get("meaning", "").strip()
                if meaning:
                    jargon_explanations.append(f"- {content}: {meaning}")
                else:
                    logger.info(f"黑话 {content} 没有找到含义")
            else:
                logger.info(f"黑话 {content} 未在数据库中找到")
        if not jargon_explanations:
            logger.info("没有找到任何黑话的含义，跳过解释")
            return None
        explanations_text = "\n".join(jargon_explanations)
        summarize_prompt = JARGON_SUMMARIZE_PROMPT.format(
            chat_context=chat_context,
            jargon_explanations=explanations_text,
        )
        summary, _ = await self.llm.generate_response_async(
            summarize_prompt, temperature=0.3
        )
        if not summary:
            return f"上下文中的黑话解释：\n{explanations_text}"
        summary = summary.strip()
        if not summary:
            return f"上下文中的黑话解释：\n{explanations_text}"
        return summary


async def explain_jargon_in_context(
    chat_id: str,
    messages: List[Any],
    chat_context: str,
) -> Optional[str]:
    """解释上下文中的黑话（便捷函数）"""
    explainer = JargonExplainer(chat_id)
    return await explainer.explain_jargon(messages, chat_context)


def match_jargon_from_text(chat_text: str, chat_id: str) -> List[str]:
    """直接在聊天文本中匹配已知的jargon，返回出现过的黑话列表"""
    if not chat_text or not chat_text.strip():
        return []
    query = Jargon.select().where(
        (Jargon.meaning.is_null(False)) & (Jargon.meaning != "")
    )
    if global_config.expression.all_global_jargon:
        query = query.where(Jargon.is_global)
    query = query.order_by(Jargon.count.desc())
    matched: Dict[str, None] = {}
    for jargon in query:
        content = (jargon.content or "").strip()
        if not content:
            continue
        if (
            not global_config.expression.all_global_jargon
            and not jargon.is_global
        ):
            chat_id_list = parse_chat_id_list(jargon.chat_id)
            if not chat_id_list_contains(chat_id_list, chat_id):
                continue
        pattern = re.escape(content)
        if re.search(r"[\u4e00-\u9fff]", content):
            search_pattern = pattern
        else:
            search_pattern = r"\b" + pattern + r"\b"
        if re.search(search_pattern, chat_text, re.IGNORECASE):
            matched[content] = None
    logger.info(f"匹配到 {len(matched)} 个黑话")
    return list(matched.keys())


async def retrieve_concepts_with_jargon(
    concepts: List[str], chat_id: str
) -> str:
    """对概念列表进行jargon检索"""
    if not concepts:
        return ""
    results = []
    exact_matches = []
    for concept in concepts:
        concept = concept.strip()
        if not concept:
            continue
        jargon_results = search_jargon(
            keyword=concept,
            chat_id=chat_id,
            limit=10,
            case_sensitive=False,
            fuzzy=False,
        )
        is_fuzzy_match = False
        if not jargon_results:
            jargon_results = search_jargon(
                keyword=concept,
                chat_id=chat_id,
                limit=10,
                case_sensitive=False,
                fuzzy=True,
            )
            is_fuzzy_match = True
        if jargon_results:
            if is_fuzzy_match:
                output_parts = [f"未精确匹配到'{concept}'"]
                for result in jargon_results:
                    found_content = result.get("content", "").strip()
                    meaning = result.get("meaning", "").strip()
                    if found_content and meaning:
                        output_parts.append(
                            f"找到 '{found_content}' 的含义为：{meaning}"
                        )
                results.append("\n".join(output_parts))
                logger.info(f"在jargon库中找到匹配（模糊搜索）: {concept}")
            else:
                output_parts = []
                for result in jargon_results:
                    meaning = result.get("meaning", "").strip()
                    if meaning:
                        output_parts.append(
                            f"'{concept}' 为黑话或者网络简写，含义为：{meaning}"
                        )
                results.append(
                    "\n".join(output_parts)
                    if len(output_parts) > 1
                    else output_parts[0]
                )
                exact_matches.append(concept)
        else:
            logger.info(f"在jargon库中未找到匹配: {concept}")
    if exact_matches:
        logger.info(f"找到黑话: {', '.join(exact_matches)}")
    if results:
        return "你了解以下词语可能的含义：\n" + "\n".join(results) + "\n"
    return ""

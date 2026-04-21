import time
import json
import os
import re
import asyncio
from typing import List, Optional, Tuple, Any, Dict
from src.common.logger import get_logger
from src.common.database.database_model import Expression
from src.llm_models.utils_model import LLMRequest
from src.config.config import model_config, global_config
from src.chat.utils.chat_message_builder import (
    build_anonymous_messages,
)
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.chat.message_receive.chat_stream import get_chat_manager
from src.bw_learner.learner_utils import (
    filter_message_content,
    is_bot_message,
    build_context_paragraph,
    contains_bot_self_name,
    calculate_similarity,
    parse_expression_response,
)
from src.bw_learner.jargon_miner import miner_manager
from src.bw_learner.expression_auto_check_task import (
    single_expression_check,
)
from src.config.prompt_loader import get_prompt_raw, PromptCategory


logger = get_logger("expressor")


def _learning(key: str) -> str:
    return get_prompt_raw(PromptCategory.LEARNING, "expression", key)


def init_prompt() -> None:
    learn_style_prompt = _learning("learn_style")
    Prompt(learn_style_prompt, "learn_style_prompt")


class ExpressionLearner:
    def __init__(self, chat_id: str) -> None:
        self.express_learn_model: LLMRequest = LLMRequest(
            model_set=model_config.model_task_config.utils, request_type="expression.learner"
        )
        self.summary_model: LLMRequest = LLMRequest(
            model_set=model_config.model_task_config.tool_use, request_type="expression.summary"
        )
        self.check_model: Optional[LLMRequest] = None
        self.chat_id = chat_id
        self.chat_stream = get_chat_manager().get_stream(chat_id)
        self.chat_name = get_chat_manager().get_stream_name(chat_id) or chat_id
        self._learning_lock = asyncio.Lock()

    async def learn_and_store(
        self,
        messages: List[Any],
    ) -> List[Tuple[str, str, str]]:
        if not messages:
            return None

        random_msg = messages
        random_msg_str: str = await build_anonymous_messages(random_msg, show_ids=True)

        prompt: str = await global_prompt_manager.format_prompt(
            "learn_style_prompt",
            bot_name=global_config.bot.nickname,
            chat_str=random_msg_str,
        )

        try:
            response, _ = await self.express_learn_model.generate_response_async(prompt, temperature=0.3)
        except Exception as e:
            logger.error(f"学习表达方式失败,模型生成出错: {e}")
            return None

        expressions: List[Tuple[str, str, str]]
        jargon_entries: List[Tuple[str, str]]
        expressions, jargon_entries = parse_expression_response(response)

        cached_jargon_entries = self._check_cached_jargons_in_messages(random_msg)
        if cached_jargon_entries:
            existing_contents = {content for content, _ in jargon_entries}
            for content, source_id in cached_jargon_entries:
                if content not in existing_contents:
                    jargon_entries.append((content, source_id))
                    existing_contents.add(content)
                    logger.info(f"从缓存中检查到黑话: {content}")

        if len(expressions) > 20:
            logger.info(f"表达方式提取数量超过10个（实际{len(expressions)}个），放弃本次表达学习")
            expressions = []

        if len(jargon_entries) > 30:
            logger.info(f"黑话提取数量超过30个（实际{len(jargon_entries)}个），放弃本次黑话学习")
            jargon_entries = []

        if jargon_entries:
            await self._process_jargon_entries(jargon_entries, random_msg)

        if not expressions:
            logger.info("解析后没有可用的表达方式")
            return []

        logger.info(f"学习的prompt: {prompt}")
        logger.info(f"学习的expressions: {expressions}")
        logger.info(f"学习的jargon_entries: {jargon_entries}")
        logger.info(f"学习的response: {response}")

        # 持久化学习到的表达方式
        current_time = time.time()
        for situation, style, source_id in expressions:
            try:
                # 检查是否存在相似表达方式
                # 此处基于 simple similarity 过滤，避免过度存储
                existing_exprs = Expression.select().where(
                    (Expression.chat_id == self.chat_id) | (Expression.chat_id == "global")
                )
                
                is_duplicate = False
                for existing in existing_exprs:
                    if calculate_similarity(situation, existing.situation) > 0.85 and \
                       calculate_similarity(style, existing.style) > 0.85:
                        # 更新活跃时间与计数
                        existing.count += 1
                        existing.last_active_time = current_time
                        existing.save()
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    Expression.create(
                        situation=situation,
                        style=style,
                        content_list=json.dumps([source_id], ensure_ascii=False),
                        count=1,
                        last_active_time=current_time,
                        chat_id=self.chat_id,
                        create_date=current_time,
                        checked=False,
                        rejected=False
                    )
                    logger.info(f"保存新表达方式: [{situation}] -> [{style}]")
            except Exception as e:
                logger.error(f"保存表达方式失败: {e}")

        return expressions

    def _check_cached_jargons_in_messages(self, messages: List[Any]) -> List[Tuple[str, str]]:
        """从缓存中检查已知黑话"""
        miner = miner_manager.get_miner(self.chat_id)
        cached_jargons = miner.get_cached_jargons()
        if not cached_jargons:
            return []

        found_entries = []
        for msg in messages:
            content = filter_message_content(getattr(msg, "processed_plain_text", ""))
            if not content:
                continue
            
            for jargon in cached_jargons:
                if jargon in content:
                    found_entries.append((jargon, getattr(msg, "message_id", "unknown")))
        
        return found_entries

    async def _process_jargon_entries(self, entries: List[Tuple[str, str]], messages: List[Any]) -> None:
        """处理黑话条目并关联上下文"""
        if not entries:
            return

        jargon_data_list = []
        for content, source_id in entries:
            # 查找对应的消息索引以构建上下文
            center_idx = -1
            for i, msg in enumerate(messages):
                if getattr(msg, "message_id", "") == source_id:
                    center_idx = i
                    break
            
            context = build_context_paragraph(messages, center_idx) if center_idx != -1 else ""
            
            jargon_data_list.append({
                "content": content,
                "raw_content": [context] if context else []
            })

        if jargon_data_list:
            miner = miner_manager.get_miner(self.chat_id)
            # 使用 miner 的 process_extracted_entries 进行增量学习
            await miner.process_extracted_entries(jargon_data_list)


expression_learner_manager: Dict[str, ExpressionLearner] = {}


def get_expression_learner(chat_id: str) -> ExpressionLearner:
    """获取或创建表达学习器实例"""
    if chat_id not in expression_learner_manager:
        expression_learner_manager[chat_id] = ExpressionLearner(chat_id)
    return expression_learner_manager[chat_id]
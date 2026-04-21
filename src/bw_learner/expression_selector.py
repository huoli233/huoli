import json
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.common.database.database_model import Expression
from src.llm_models.utils_model import LLMRequest
from src.config.config import global_config, model_config
from src.bw_learner.learner_utils import (
    weighted_sample,
    parse_platform_accounts,
)

logger = get_logger("表达选择")

EXPRESSION_EVALUATION_PROMPT = """{chat_observe_info}

你的名字是{bot_name}{target_message}
{reply_reason_block}

以下是可选的表达情境：
{all_situations}

请你分析聊天内容的语境、情绪、话题类型，从上述情境中选择最适合当前聊天情境的，最多{max_num}个情境。
考虑因素包括：
1.聊天的情绪氛围（轻松、严肃、幽默等）
2.话题类型（日常、技术、游戏、情感等）
3.情境与当前语境的匹配度
{target_message_extra_block}

请以JSON格式输出，只需要输出选中的情境编号：
例如：
{{
    "selected_situations": [2, 3, 5, 7, 19]
}}

请严格按照JSON格式输出，不要包含其他内容："""


class ExpressionSelector:
    """表达方式选择器，用于选择适合当前聊天情境的表达方式"""

    def __init__(self):
        self.llm_model = LLMRequest(
            model_set=model_config.model_task_config.tool_use,
            request_type="expression.selector",
        )

    def can_use_expression_for_chat(self, chat_id: str) -> bool:
        """检查指定聊天流是否允许使用表达"""
        try:
            use_expression, _, _ = (
                global_config.expression.get_expression_config_for_chat(
                    chat_id
                )
            )
            return use_expression
        except Exception as e:
            logger.error(f"检查表达使用权限失败: {e}")
            return False

    def _parse_stream_config_to_chat_id(
        self, stream_config_str: str
    ) -> Optional[str]:
        """解析'platform:id:type'为chat_id"""
        try:
            parts = stream_config_str.split(":")
            if len(parts) != 3:
                return None
            platform = parts[0]
            id_str = parts[1]
            stream_type = parts[2]
            is_group = stream_type == "group"
            from src.chat.message_receive.chat_stream import get_chat_manager

            return get_chat_manager().get_stream_id(
                platform, str(id_str), is_group=is_group
            )
        except Exception as e:
            logger.debug(f"获取流ID失败: {e}")
            return None

    def get_related_chat_ids(self, chat_id: str) -> List[str]:
        """根据expression_groups配置，获取与当前chat_id相关的所有chat_id"""
        groups = global_config.expression.expression_groups
        global_group_exists = any("*" in group for group in groups)
        if global_group_exists:
            all_chat_ids = set()
            for group in groups:
                for stream_config_str in group:
                    chat_id_candidate = self._parse_stream_config_to_chat_id(
                        stream_config_str
                    )
                    if chat_id_candidate:
                        all_chat_ids.add(chat_id_candidate)
            return list(all_chat_ids) if all_chat_ids else [chat_id]
        for group in groups:
            group_chat_ids = []
            for stream_config_str in group:
                chat_id_candidate = self._parse_stream_config_to_chat_id(
                    stream_config_str
                )
                if chat_id_candidate:
                    group_chat_ids.append(chat_id_candidate)
            if chat_id in group_chat_ids:
                return group_chat_ids
        return [chat_id]

    def _select_expressions_simple(
        self,
        chat_id: str,
        max_num: int,
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """简单模式：只选择 count > 1 的项目，随机选5个，不进行LLM选择"""
        try:
            related_chat_ids = self.get_related_chat_ids(chat_id)
            base_conditions = (
                (Expression.chat_id.in_(related_chat_ids))
                & (~Expression.rejected)
                & (Expression.count > 1)
            )
            if global_config.expression.expression_checked_only:
                base_conditions = base_conditions & (Expression.checked)
            style_query = Expression.select().where(base_conditions)
            style_exprs = [
                {
                    "id": expr.id,
                    "situation": expr.situation,
                    "style": expr.style,
                    "last_active_time": expr.last_active_time,
                    "source_id": expr.chat_id,
                    "create_date": (
                        expr.create_date
                        if expr.create_date is not None
                        else expr.last_active_time
                    ),
                    "count": (
                        expr.count
                        if getattr(expr, "count", None) is not None
                        else 1
                    ),
                    "checked": (
                        expr.checked
                        if getattr(expr, "checked", None) is not None
                        else False
                    ),
                }
                for expr in style_query
            ]
            min_required = 8
            if len(style_exprs) < min_required:
                if not style_exprs:
                    logger.info(
                        f"聊天流 {chat_id} 没有满足 count > 1 且未被拒绝的表达方式"
                    )
                    fallback_num = min(3, max_num) if max_num > 0 else 3
                    fallback_selected = self._random_expressions(
                        chat_id, fallback_num
                    )
                    if fallback_selected:
                        self.update_expressions_last_active_time(
                            fallback_selected
                        )
                        selected_ids = [
                            expr["id"] for expr in fallback_selected
                        ]
                        return fallback_selected, selected_ids
                    return [], []
                logger.info(
                    f"聊天流 {chat_id} count > 1 的表达方式不足 {min_required} 个"
                )
                select_count = min(3, len(style_exprs))
            else:
                select_count = 5
            selected_style = random.sample(style_exprs, select_count)
            if selected_style:
                self.update_expressions_last_active_time(selected_style)
            selected_ids = [expr["id"] for expr in selected_style]
            return selected_style, selected_ids
        except Exception as e:
            logger.error(f"简单模式选择表达方式失败: {e}")
            return [], []

    def _random_expressions(
        self, chat_id: str, total_num: int
    ) -> List[Dict[str, Any]]:
        """随机选择表达方式"""
        try:
            related_chat_ids = self.get_related_chat_ids(chat_id)
            base_conditions = (Expression.chat_id.in_(related_chat_ids)) & (
                ~Expression.rejected
            )
            if global_config.expression.expression_checked_only:
                base_conditions = base_conditions & (Expression.checked)
            style_query = Expression.select().where(base_conditions)
            style_exprs = [
                {
                    "id": expr.id,
                    "situation": expr.situation,
                    "style": expr.style,
                    "last_active_time": expr.last_active_time,
                    "source_id": expr.chat_id,
                    "create_date": (
                        expr.create_date
                        if expr.create_date is not None
                        else expr.last_active_time
                    ),
                    "count": (
                        expr.count
                        if getattr(expr, "count", None) is not None
                        else 1
                    ),
                    "checked": (
                        expr.checked
                        if getattr(expr, "checked", None) is not None
                        else False
                    ),
                }
                for expr in style_query
            ]
            if style_exprs:
                selected_style = weighted_sample(style_exprs, total_num)
            else:
                selected_style = []
            return selected_style
        except Exception as e:
            logger.error(f"随机选择表达方式失败: {e}")
            return []

    async def select_suitable_expressions(
        self,
        chat_id: str,
        chat_info: str,
        max_num: int = 10,
        target_message: Optional[str] = None,
        reply_reason: Optional[str] = None,
        think_level: int = 1,
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """选择适合的表达方式"""
        if not self.can_use_expression_for_chat(chat_id):
            logger.debug(f"聊天流 {chat_id} 不允许使用表达")
            return [], []
        expressions, ids = await self._select_expressions_classic(
            chat_id,
            chat_info,
            max_num,
            target_message,
            reply_reason,
            think_level,
        )
        # 用style_picker补充上下文匹配的候选风格（排名/冷却/用量统计增强）
        if len(expressions) < max_num:
            try:
                from src.express.style_picker import pick_best_style
                _context = target_message or chat_info[:200]
                _seen = {(e.get("situation", ""), e.get("style", "")) for e in expressions}
                _supplement_count = max_num - len(expressions)
                for _ in range(_supplement_count):
                    _pick = await pick_best_style(chat_id, _context, use_cooldown=True)
                    if _pick and _pick not in _seen:
                        _seen.add(_pick)
                        expressions.append({"situation": _pick[0], "style": _pick[1]})
            except ImportError:
                pass
            except Exception as _e:
                logger.debug(f"style_picker补充异常: {_e}")
        return expressions, ids

    async def _select_expressions_classic(
        self,
        chat_id: str,
        chat_info: str,
        max_num: int = 10,
        target_message: Optional[str] = None,
        reply_reason: Optional[str] = None,
        think_level: int = 1,
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """classic模式：随机选择+LLM选择"""
        try:
            if think_level == 0:
                return self._select_expressions_simple(chat_id, max_num)
            related_chat_ids = self.get_related_chat_ids(chat_id)
            base_conditions = (Expression.chat_id.in_(related_chat_ids)) & (
                ~Expression.rejected
            )
            if global_config.expression.expression_checked_only:
                base_conditions = base_conditions & (Expression.checked)
            style_query = Expression.select().where(base_conditions)
            all_style_exprs = [
                {
                    "id": expr.id,
                    "situation": expr.situation,
                    "style": expr.style,
                    "last_active_time": expr.last_active_time,
                    "source_id": expr.chat_id,
                    "create_date": (
                        expr.create_date
                        if expr.create_date is not None
                        else expr.last_active_time
                    ),
                    "count": (
                        expr.count
                        if getattr(expr, "count", None) is not None
                        else 1
                    ),
                    "checked": (
                        expr.checked
                        if getattr(expr, "checked", None) is not None
                        else False
                    ),
                }
                for expr in style_query
            ]
            high_count_exprs = [
                expr
                for expr in all_style_exprs
                if (expr.get("count", 1) or 1) > 1
            ]
            min_high_count = 10
            min_total_count = 10
            select_high_count = 5
            select_random_count = 5
            if len(high_count_exprs) < min_high_count:
                logger.info(
                    f"聊天流 {chat_id} count > 1 的表达方式不足 {min_high_count} 个"
                )
                high_count_valid = False
            else:
                high_count_valid = True
            if len(all_style_exprs) < min_total_count:
                logger.info(
                    f"聊天流 {chat_id} 总表达方式不足 {min_total_count} 个"
                )
                return [], []
            if high_count_valid:
                selected_high = weighted_sample(
                    high_count_exprs,
                    min(len(high_count_exprs), select_high_count),
                )
            else:
                selected_high = []
            remaining_num = select_random_count
            selected_random = weighted_sample(
                all_style_exprs, min(len(all_style_exprs), remaining_num)
            )
            candidate_exprs = selected_high.copy()
            candidate_ids = {expr["id"] for expr in candidate_exprs}
            for expr in selected_random:
                if expr["id"] not in candidate_ids:
                    candidate_exprs.append(expr)
                    candidate_ids.add(expr["id"])
            random.shuffle(candidate_exprs)
            all_expressions: List[Dict[str, Any]] = []
            all_situations: List[str] = []
            for expr in candidate_exprs:
                expr_copy = expr.copy()
                all_expressions.append(expr_copy)
                all_situations.append(
                    f"{len(all_expressions)}.当 {expr['situation']} 时，使用 {expr['style']}"
                )
            if not all_expressions:
                logger.warning("没有找到可用的表达方式")
                return [], []
            all_situations_str = "\n".join(all_situations)
            if target_message:
                target_message_str = (
                    f'，现在你想要对这条消息进行回复："{target_message}"'
                )
                target_message_extra_block = "4.考虑你要回复的目标消息"
            else:
                target_message_str = ""
                target_message_extra_block = ""
            chat_context = f"以下是正在进行的聊天内容：{chat_info}"
            if reply_reason:
                reply_reason_block = f"你的回复理由是：{reply_reason}"
                chat_context = ""
            else:
                reply_reason_block = ""
            prompt = EXPRESSION_EVALUATION_PROMPT.format(
                bot_name=global_config.bot.nickname,
                chat_observe_info=chat_context,
                all_situations=all_situations_str,
                max_num=max_num,
                target_message=target_message_str,
                target_message_extra_block=target_message_extra_block,
                reply_reason_block=reply_reason_block,
            )
            content, (reasoning_content, model_name, _) = (
                await self.llm_model.generate_response_async(prompt=prompt)
            )
            if not content:
                logger.warning("LLM返回空结果")
                return [], []
            result = self._parse_json_response(content)
            if (
                not isinstance(result, dict)
                or "selected_situations" not in result
            ):
                logger.error("LLM返回格式错误")
                return [], []
            selected_indices = result["selected_situations"]
            valid_expressions: List[Dict[str, Any]] = []
            selected_ids = []
            for idx in selected_indices:
                if isinstance(idx, int) and 1 <= idx <= len(all_expressions):
                    expression = all_expressions[idx - 1]
                    selected_ids.append(expression["id"])
                    valid_expressions.append(expression)
            if valid_expressions:
                self.update_expressions_last_active_time(valid_expressions)
            logger.debug(
                f"从{len(all_expressions)}个情境中选择了{len(valid_expressions)}个"
            )
            return valid_expressions, selected_ids
        except Exception as e:
            logger.error(f"classic模式处理表达方式选择时出错: {e}")
            return [], []

    def _parse_json_response(self, content: str) -> Any:
        """解析 JSON 响应"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.debug("JSON解析失败，尝试正则提取")
        import re

        json_match = re.search(
            r'\{[^{}]*"selected_situations"[^{}]*\}', content, re.DOTALL
        )
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.debug("正则提取的JSON解析失败，尝试替换引号后重试")
        content = content.replace("'", '"')
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}

    def update_expressions_last_active_time(
        self, expressions_to_update: List[Dict[str, Any]]
    ):
        """对一批表达方式更新last_active_time"""
        if not expressions_to_update:
            return
        updates_by_key = {}
        for expr in expressions_to_update:
            source_id = expr.get("source_id")
            situation = expr.get("situation")
            style = expr.get("style")
            if not source_id or not situation or not style:
                continue
            key = (source_id, situation, style)
            if key not in updates_by_key:
                updates_by_key[key] = expr
        for chat_id, situation, style in updates_by_key:
            query = Expression.select().where(
                (Expression.chat_id == chat_id)
                & (Expression.situation == situation)
                & (Expression.style == style)
            )
            if query.exists():
                expr_obj = query.get()
                expr_obj.last_active_time = time.time()
                expr_obj.save()
                logger.debug("表达方式激活: 更新last_active_time")


try:
    expression_selector = ExpressionSelector()
except Exception as e:
    logger.error(f"ExpressionSelector初始化失败: {e}")
    expression_selector = None

import asyncio
import json
import random
import re
from typing import List, Tuple

from src.common.database.database_model import Expression
from src.common.logger import get_logger
from src.common.task_utils import safe_create_task
from src.config.config import global_config, model_config
from src.llm_models.utils_model import LLMRequest
from src.manager.async_task_manager import AsyncTask

logger = get_logger("表达自动检查")


def _expression_check_interval() -> int:
    try:
        from src.config.core_config_engine import get_core_config

        view = get_core_config().resolve_module_view("schedule").values
        return int(
            view.get(
                "expression_learning_interval_seconds",
                global_config.expression.expression_auto_check_interval,
            )
        )
    except Exception:
        return int(global_config.expression.expression_auto_check_interval)


def create_evaluation_prompt(situation: str, style: str) -> str:
    """创建评估提示词"""
    base_criteria = [
        "表达方式或言语风格 是否与使用条件或使用情景 匹配",
        "允许部分语法错误或口头化或缺省出现",
        "表达方式不能太过特指，需要具有泛用性",
        "一般不涉及具体的人名或名称",
    ]
    custom_criteria = (
        global_config.expression.expression_auto_check_custom_criteria
    )
    all_criteria = base_criteria.copy()
    if custom_criteria:
        all_criteria.extend(custom_criteria)
    criteria_list = "\n".join(
        [f"{i + 1}. {criterion}" for i, criterion in enumerate(all_criteria)]
    )
    prompt = f"""请评估以下表达方式或语言风格以及使用条件或使用情景是否合适：
使用条件或使用情景：{situation}
表达方式或言语风格：{style}

请从以下方面进行评估：
{criteria_list}

请以JSON格式输出评估结果：
{{
    "suitable": true/false,
    "reason": "评估理由（如果不合适，请说明原因）"

}}
如果合适，suitable设为true；如果不合适，suitable设为false，并在reason中说明原因。
请严格按照JSON格式输出，不要包含其他内容。"""
    return prompt


async def single_expression_check(
    situation: str, style: str
) -> Tuple[bool, str, str]:
    """执行单次LLM评估"""
    try:
        judge_llm = LLMRequest(
            model_set=model_config.model_task_config.tool_use,
            request_type="expression_check",
        )
        prompt = create_evaluation_prompt(situation, style)
        logger.debug(f"正在评估表达方式: situation={situation}, style={style}")
        response, (reasoning, model_name, _) = (
            await judge_llm.generate_response_async(
                prompt=prompt,
                temperature=0.6,
            )
        )
        logger.debug(f"LLM响应: {response}")
        try:
            evaluation = json.loads(response)
        except json.JSONDecodeError as e:
            json_match = re.search(
                r'\{[^{}]*"suitable"[^{}]*\}', response, re.DOTALL
            )
            if json_match:
                evaluation = json.loads(json_match.group())
            else:
                raise ValueError("无法从响应中提取JSON格式的评估结果") from e
        suitable = evaluation.get("suitable", False)
        reason = evaluation.get("reason", "未提供理由")
        logger.debug(f"评估结果: {'通过' if suitable else '不通过'}")
        return suitable, reason, None
    except Exception as e:
        logger.error(
            f"评估表达方式 (situation={situation}, style={style}) 时出错: {e}"
        )
        return False, f"评估过程出错: {str(e)}", str(e)


class ExpressionAutoCheckTask(AsyncTask):
    """表达方式自动检查定时任务"""

    def __init__(self):
        interval = _expression_check_interval()
        super().__init__(
            task_name="ExpressionAutoCheckTask",
            wait_before_start=60,
            run_interval=interval,
        )
        self._running = False
        self._task: asyncio.Task = None
        self.check_interval = interval
        self.wait_before_start = 60

    async def run(self):
        """执行一次检查任务"""
        await self.run_once()

    async def _select_expressions(self, count: int) -> List[Expression]:
        """随机选择指定数量的未检查表达方式"""
        try:
            unevaluated_expressions = list(
                Expression.select().where(~Expression.checked)
            )
            if not unevaluated_expressions:
                logger.info("没有未检查的表达方式")
                return []
            selected_count = min(count, len(unevaluated_expressions))
            selected = random.sample(unevaluated_expressions, selected_count)
            logger.info(
                f"从 {
                    len(unevaluated_expressions)} 条未检查表达方式中随机选择了 {selected_count} 条"
            )
            return selected
        except Exception as e:
            logger.error(f"选择表达方式时出错: {e}")
            return []

    async def _evaluate_expression(self, expression: Expression) -> bool:
        """评估单个表达方式"""
        suitable, reason, error = await single_expression_check(
            expression.situation,
            expression.style,
        )
        try:
            expression.checked = True
            expression.rejected = not suitable
            expression.modified_by = "ai"
            expression.save()
            status = "通过" if suitable else "不通过"
            logger.info(
                f"表达方式评估完成 [ID: {expression.id}] - {status} | "
                f"Situation: {expression.situation[:30]}... | "
                f"Style: {expression.style[:30]}... | "
                f"Reason: {reason[:50]}..."
            )
            if error:
                logger.warning(
                    f"表达方式评估时出现错误 [ID: {expression.id}]: {error}"
                )
            return suitable
        except Exception as e:
            logger.error(f"更新表达方式状态失败 [ID: {expression.id}]: {e}")
            return False

    async def run_once(self):
        """执行一次检查任务"""
        try:
            if not global_config.expression.expression_self_reflect:
                logger.debug("表达方式自动检查未启用，跳过本次执行")
                return
            check_count = global_config.expression.expression_auto_check_count
            if check_count <= 0:
                logger.warning(
                    f"检查数量配置无效: {check_count}，跳过本次执行"
                )
                return
            logger.info(
                f"开始执行表达方式自动检查，本次将检查 {check_count} 条"
            )
            expressions = await self._select_expressions(check_count)
            if not expressions:
                logger.info("没有需要检查的表达方式")
                return
            passed_count = 0
            failed_count = 0
            for i, expression in enumerate(expressions, 1):
                logger.info(
                    f"正在评估 [{i}/{len(expressions)}]: ID={expression.id}"
                )
                if await self._evaluate_expression(expression):
                    passed_count += 1
                else:
                    failed_count += 1
                await asyncio.sleep(0.3)
            logger.info(
                f"表达方式自动检查完成: 总计 {len(expressions)} 条，"
                f"通过 {passed_count} 条，不通过 {failed_count} 条"
            )
        except Exception as e:
            logger.error(f"执行表达方式自动检查任务时出错: {e}", exc_info=True)

    async def _run_loop(self):
        """运行循环"""
        await asyncio.sleep(self.wait_before_start)
        while self._running:
            await self.run_once()
            await asyncio.sleep(self.check_interval)

    def start(self):
        """启动任务"""
        if self._running:
            return
        self._running = True
        self._task = safe_create_task(
            self._run_loop(), name="expression_auto_check"
        )
        logger.info("表达方式自动检查任务已启动")

    def stop(self):
        """停止任务"""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("表达方式自动检查任务已停止")


expression_auto_check_task = ExpressionAutoCheckTask()

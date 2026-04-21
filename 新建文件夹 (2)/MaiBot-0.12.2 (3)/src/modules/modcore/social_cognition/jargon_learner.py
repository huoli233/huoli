import time
import json
import random
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.common.logger import get_logger

logger = get_logger("group_language_learner")


class GroupLanguageLearner:
    DECAY_DAYS = 15.0
    DECAY_MIN = 0.01

    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self._cache_patterns: List[Dict] = []
        self._last_cache_update = 0

    async def learn_from_messages(self, messages: List[Dict[str, Any]]) -> int:
        try:
            prompt = self._construct_learning_prompt(messages)
            if not prompt:
                return 0
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            request = LLMRequest(model_config.lightweight, request_type="jargon_learning")
            response_text, _ = await request.generate_response_async(prompt)
            if not response_text:
                return 0
            patterns = self._parse_llm_response(response_text)
            if not patterns:
                return 0
            count = self._save_patterns(patterns)
            self._apply_time_decay()
            self.refresh_cache()
            return count
        except Exception as e:
            logger.error(f"群体语言学习失败: {e}")
            return 0

    def _construct_learning_prompt(self, messages: List[Dict]) -> str:
        if len(messages) < 5:
            return ""
        context_lines = []
        from src.config.config import global_config
        bot_name = global_config.bot.nickname or ""
        bot_qq = global_config.bot.qq_account or ""
        for msg in messages:
            sender = msg.get("sender_name", "Unknown")
            sender_id = str(msg.get("sender_id", ""))
            if sender_id == bot_qq or sender == bot_name:
                continue
            content = msg.get("content", "").strip()
            if len(content) < 2 or content.startswith("[") or "http" in content:
                continue
            ts = msg.get("timestamp", time.time())
            time_str = datetime.fromtimestamp(ts).strftime("%H:%M")
            context_lines.append(f"[{time_str}] {sender}: {content}")
        if not context_lines:
            return ""
        chat_context = "\n".join(context_lines[-30:])
        from src.config.prompt_loader import get_prompt, PromptCategory
        return get_prompt(
            PromptCategory.MODULE,
            "jargon_learner",
            "jargon_learning.template",
            chat_context=chat_context
        )

    def _parse_llm_response(self, text: str) -> List[Dict]:
        try:
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except Exception:
            logger.debug(f"解析语言学习结果失败: {text[:50]}...")
            return []

    def _save_patterns(self, patterns: List[Dict]) -> int:
        from src.common.database.database_model import ExpressionPattern
        count = 0
        now = time.time()
        for p in patterns:
            sit = p.get("situation")
            expr = p.get("expression")
            if not sit or not expr:
                continue
            try:
                existing = ExpressionPattern.get_or_none(
                    (ExpressionPattern.stream_id == self.stream_id) &
                    (ExpressionPattern.situation == sit) &
                    (ExpressionPattern.expression == expr)
                )
                if existing:
                    existing.weight = existing.weight + 1.0
                    existing.last_used = now
                    existing.save()
                else:
                    ExpressionPattern.create(
                        stream_id=self.stream_id,
                        situation=sit,
                        expression=expr,
                        weight=1.0,
                        last_used=now,
                        created_at=now,
                    )
                count += 1
            except Exception as e:
                logger.debug(f"保存语言模式失败: {e}")
        return count

    def _apply_time_decay(self):
        from src.common.database.database_model import ExpressionPattern
        now = time.time()
        try:
            rows = ExpressionPattern.select().where(
                ExpressionPattern.stream_id == self.stream_id
            )
            for row in rows:
                days_inactive = (now - row.last_used) / 86400.0
                if days_inactive <= 0.5:
                    continue
                decay_amount = 0.05 * days_inactive
                new_weight = row.weight - decay_amount
                if new_weight < self.DECAY_MIN:
                    row.delete_instance()
                elif new_weight != row.weight:
                    row.weight = new_weight
                    row.save()
        except Exception as e:
            logger.debug(f"遗忘机制执行失败: {e}")

    def refresh_cache(self):
        from src.common.database.database_model import ExpressionPattern
        try:
            rows = (ExpressionPattern.select()
                    .where(ExpressionPattern.stream_id == self.stream_id)
                    .order_by(ExpressionPattern.weight.desc())
                    .limit(10))
            self._cache_patterns = [
                {"situation": r.situation, "expression": r.expression}
                for r in rows
            ]
            self._last_cache_update = time.time()
        except Exception as e:
            logger.debug(f"刷新缓存失败: {e}")

    def get_style_prompt(self) -> str:
        if not self._cache_patterns:
            return ""
        lines = ["【本群语言风格感知】"]
        lines.append("- 根据近期观察，本群有以下特定的语言习惯：")
        sample = self._cache_patterns
        if len(sample) > 5:
            sample = random.sample(sample, 5)
        for p in sample:
            lines.append(f"  * 当 {p['situation']} 时 -> 习惯说 \"{p['expression']}\"")
        lines.append("(请在回复中自然地模仿这些表达方式，融入群体氛围)")
        return "\n".join(lines)


_learner_instances: Dict[str, GroupLanguageLearner] = {}


def get_jargon_learner(stream_id: str) -> GroupLanguageLearner:
    if stream_id not in _learner_instances:
        _learner_instances[stream_id] = GroupLanguageLearner(stream_id)
    return _learner_instances[stream_id]

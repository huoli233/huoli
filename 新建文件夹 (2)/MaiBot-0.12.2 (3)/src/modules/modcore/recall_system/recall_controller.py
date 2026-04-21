import random
import asyncio
import time
from typing import Optional, Tuple, Dict, Callable, Awaitable, List
from dataclasses import dataclass, field
from src.common.logger import get_logger
from src.modules.modcore.recall_system.typo_maker import get_typo_maker, make_typo, make_typo_async
from src.modules.modcore.recall_system.correction_detector import get_correction_detector, CorrectionEvent
from src.modules.modcore.recall_system.content_shuffler import get_content_shuffler
from src.modules.modcore.recall_system.self_awareness import get_self_awareness

logger = get_logger("recall")


@dataclass
class PendingRecall:
    msg_id: str
    group_id: str
    stream_id: str
    original_text: str
    error_text: str
    error_type: str
    scheduled_at: float
    recall_at: float
    content_type: str = "text"
    task: Optional[asyncio.Task] = None
    recalled: bool = False
    reason: str = ""
    after_action: str = "correct"


class RecallController:
    AFTER_ACTIONS = {
        "correct": 0.60,
        "silent": 0.25,
        "regret": 0.10,
        "shy": 0.05,
    }

    def __init__(self, config: Optional[Dict] = None):
        self._cfg = config or {}
        self._enabled = self._cfg.get("enable_recall", True)
        self._typo_prob = self._cfg.get("typo_probability", 0.15)  # 从0.12提高到0.15
        self._wrong_prob = self._cfg.get("wrong_content_probability", 0.08)
        self._recall_prob = self._cfg.get("recall_probability", 0.35)
        self._delay_min = self._cfg.get("recall_delay_min", 3)
        self._delay_max = self._cfg.get("recall_delay_max", 60)
        self._time_limit = self._cfg.get("time_limit_seconds", 120)
        self._typo_maker = get_typo_maker()
        self._detector = get_correction_detector()
        self._shuffler = get_content_shuffler()
        self._awareness = get_self_awareness()
        self._use_llm = self._cfg.get("use_llm_for_typo", True)
        self._pending: Dict[str, PendingRecall] = {}
        self._history: List[PendingRecall] = []
        self._max_history = 50
        # 打错字频率控制
        self._typo_history: Dict[str, List[float]] = {}  # stream_id -> [timestamps]
        self._min_typo_interval = 300.0  # 最少5分钟间隔
        self._max_typo_per_hour = 3  # 每小时最多3次

    def should_trigger(
        self, 
        stream_id: str = "",
        user_id: str = "",
        message_text: str = "",
        is_question: bool = False,
        is_important: bool = False,
        topic_type: str = "casual",
        is_private: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        严谨的多维度判断是否触发打错字
        
        判断维度：
        1. 规则：重要消息、问题、严肃话题 -> 不打错
        2. 频率限制：间隔太短、次数太多 -> 不打错
        3. 群友关系：陌生人、关系差 -> 不打错
        4. 机器人状态：状态好时少打错，状态差时可能打错
        5. 场景时间：深夜、正式场合 -> 不打错
        """
        if not self._enabled:
            return False, None
        
        # 第一层：硬性规则过滤
        hard_block, reason = self._check_hard_rules(
            message_text, is_question, is_important, topic_type
        )
        if hard_block:
            logger.debug(f"[打错字] 硬性规则阻止: {reason}")
            return False, None
        
        # 第二层：频率限制
        if not self._check_frequency_limit(stream_id):
            logger.debug(f"[打错字] 频率限制")
            return False, None
        
        # 第三层：用户关系检查
        if not self._check_relationship(stream_id, user_id):
            logger.debug(f"[打错字] 关系不适合")
            return False, None
        
        # 第四层：状态和场景综合判断
        state_prob = self._calculate_state_probability(stream_id)
        
        # 私聊时打错字概率降低（更认真打字）
        if is_private:
            state_prob *= 0.5
        
        # 最终概率判断
        final_prob = self._typo_prob * state_prob
        
        if random.random() > final_prob:
            return False, None
        
        # 通过所有检查，允许打错字
        error_types = ["typo", "swap", "omit", "duplicate", "pinyin", "wrong"]
        weights = [0.25, 0.15, 0.15, 0.15, 0.15, 0.15]
        error_type = random.choices(error_types, weights=weights)[0]
        
        # 记录打错字
        self._record_typo(stream_id)
        
        logger.info(f"[打错字] 触发 类型={error_type} 概率={final_prob:.2%}")
        return True, error_type
    
    def _check_hard_rules(
        self, 
        message_text: str,
        is_question: bool,
        is_important: bool,
        topic_type: str,
    ) -> Tuple[bool, str]:
        """硬性规则检查 - 这些情况绝对不打错"""
        # 1. 重要消息不打错
        if is_important:
            return True, "重要消息"
        
        # 2. 问题不打错
        if is_question or "?" in message_text or "？" in message_text:
            return True, "回答问题"
        
        # 3. 严肃话题不打错
        serious_keywords = ["工作", "学习", "帮助", "求助", "问题", "错误", "bug", "帮我", "请"]
        if any(kw in message_text for kw in serious_keywords):
            return True, "严肃话题"
        
        if topic_type in ["serious", "work", "help", "command"]:
            return True, f"话题类型({topic_type})"
        
        # 4. 消息太短不打错
        if len(message_text) < 5:
            return True, "消息太短"
        
        # 5. 消息太长不打错
        if len(message_text) > 100:
            return True, "消息太长"
        
        # 6. 深夜时段不打错
        hour = time.localtime().tm_hour
        if 2 <= hour < 6:
            return True, "深夜时段"
        
        return False, ""
    
    def _check_frequency_limit(self, stream_id: str) -> bool:
        """检查打错字频率限制"""
        if not stream_id:
            return True
        
        now = time.time()
        history = self._typo_history.get(stream_id, [])
        
        # 清理1小时前的记录
        history = [t for t in history if now - t < 3600]
        self._typo_history[stream_id] = history
        
        # 检查最近一次的间隔
        if history:
            last_time = history[-1]
            if now - last_time < self._min_typo_interval:
                return False
        
        # 检查1小时内的次数
        if len(history) >= self._max_typo_per_hour:
            return False
        
        return True
    
    def _check_relationship(self, stream_id: str, user_id: str) -> bool:
        """检查用户关系是否适合打错字"""
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(stream_id)
            state = tracker.get_user_state(user_id, create_if_missing=False)
            
            if not state:
                # 陌生人，不打错
                return False
            
            # 好感度太低，不打错
            if state.affection < -10:
                return False
            
            # 关系不够熟，不打错
            if state.relationship == "陌生人" and state.affection < 10:
                return False
            
            return True
        except Exception:
            # 无法获取关系信息，保守处理
            return False
    
    def _calculate_state_probability(self, stream_id: str) -> float:
        """
        根据机器人状态计算打错字概率
        状态越差，越容易打错（更真实）
        """
        prob = 1.0
        
        try:
            # 获取疲劳状态
            from src.chat.heart_flow.heartflow import get_heartflow
            hf = get_heartflow(stream_id)
            if hf and hasattr(hf, 'chatting') and hf.chatting:
                chatting = hf.chatting
                
                # 精力低时更容易打错
                chat_value = getattr(chatting, '_chat_value', 100.0)
                if chat_value < 30:
                    prob *= 1.5  # 精力很低，打错概率提高50%
                elif chat_value < 50:
                    prob *= 1.2
                
                # 脑力低时更容易打错
                brain_power = getattr(chatting, '_brain_power', 100.0)
                if brain_power < 30:
                    prob *= 1.3
                elif brain_power < 50:
                    prob *= 1.1
                
                # 连续回复多次，更容易打错
                consecutive = getattr(chatting, '_consecutive_replies', 0)
                if consecutive >= 5:
                    prob *= 1.4
                elif consecutive >= 3:
                    prob *= 1.2
        except Exception:
            pass
        
        # 限制最大概率
        return min(prob, 2.0)
    
    def _record_typo(self, stream_id: str):
        """记录打错字时间"""
        if not stream_id:
            return
        
        now = time.time()
        if stream_id not in self._typo_history:
            self._typo_history[stream_id] = []
        
        self._typo_history[stream_id].append(now)
        
        # 只保留最近1小时的记录
        self._typo_history[stream_id] = [
            t for t in self._typo_history[stream_id]
            if now - t < 3600
        ]

    async def generate_error_text(self, original: str, error_type: str = "random", stream_id: str = "") -> Tuple[str, str]:
        if error_type == "wrong":
            result = self._shuffler.shuffle_with_info(original)
            if result["modified"]:
                return result["shuffled"], "wrong_content"
            error_type = "typo"
        if self._use_llm:
            error_text, source = await make_typo_async(original, error_type, stream_id)
            return error_text, f"{error_type}_{source}"
        return self._typo_maker.create_typo(original), f"{error_type}_local"

    def prepare_recall(
        self, msg_id: str, group_id: str, stream_id: str,
        original_text: str, error_text: str, error_type: str,
        content_type: str = "text",
    ) -> PendingRecall:
        now = time.time()
        delay = random.uniform(self._delay_min, min(self._delay_max, self._time_limit - 5))
        recall_at = now + delay
        pending = PendingRecall(
            msg_id=msg_id, group_id=group_id, stream_id=stream_id,
            original_text=original_text, error_text=error_text,
            error_type=error_type, scheduled_at=now, recall_at=recall_at,
            content_type=content_type,
        )
        self._pending[msg_id] = pending
        self._detector.log_bot_send(stream_id, msg_id, error_text)
        self._awareness.record_sent(msg_id, stream_id, group_id, error_text, content_type)
        logger.info(f"撤回准备: {delay:.1f}s后执行")
        return pending

    def record_normal_send(self, msg_id: str, stream_id: str, group_id: str, content: str, content_type: str = "text"):
        self._awareness.record_sent(msg_id, stream_id, group_id, content, content_type)

    async def execute_recall(
        self, pending: PendingRecall,
        delete_func: Callable[[str], Awaitable[bool]],
        send_func: Optional[Callable[[str, str], Awaitable[str]]] = None,
        on_recalled: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        delay = pending.recall_at - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        if pending.recalled:
            return
        try:
            success = await delete_func(pending.msg_id)
            if success:
                pending.recalled = True
                pending.reason = "scheduled"
                pending.after_action = self._decide_after_action()
                logger.info(f"已撤回: {pending.after_action}")
                if on_recalled:
                    await on_recalled()
                if send_func:
                    reaction = self._generate_reaction(pending)
                    if reaction:
                        await send_func(pending.group_id, reaction)
                        logger.debug(f"撤回后发送: {reaction[:20]}")
                self._archive_pending(pending)
        except Exception as e:
            logger.error(f"撤回执行失败: {e}")

    async def schedule_recall(
        self, msg_id: str, group_id: str, original_text: str,
        delete_func: Callable[[str], Awaitable[bool]],
        send_func: Optional[Callable[[str, str], Awaitable[str]]] = None,
        stream_id: str = "",
        on_recalled: Optional[Callable[[], Awaitable[None]]] = None,
        error_text: str = "", error_type: str = "typo",
    ):
        pending = self.prepare_recall(msg_id, group_id, stream_id, original_text, error_text, error_type)
        async def _do_recall():
            await self.execute_recall(pending, delete_func, send_func, on_recalled)
        task = asyncio.create_task(_do_recall())
        pending.task = task

    async def handle_user_correction(
        self, stream_id: str, user_msg: str,
        delete_func: Callable[[str], Awaitable[bool]],
        send_func: Optional[Callable[[str, str], Awaitable[None]]] = None,
    ) -> bool:
        event = await self._detector.check_correction(stream_id, user_msg)
        if not event:
            return False
        msg_id = event.bot_msg_id
        if not msg_id:
            return False
        pending = self._pending.get(msg_id)
        if pending and pending.task and not pending.task.done():
            pending.task.cancel()
        try:
            success = await delete_func(msg_id)
            if success:
                self._awareness.mark_recalled(msg_id, "user_correction")
                if pending:
                    pending.recalled = True
                    pending.reason = "user_correction"
                    self._archive_pending(pending)
                logger.info(f"用户指出错误，已撤回: {msg_id[:12]} 原因={event.reason}")
                if send_func:
                    reaction = self._generate_correction_response()
                    if reaction:
                        gid = pending.group_id if pending else ""
                        await send_func(gid, reaction)
                return True
        except Exception as e:
            logger.error(f"用户纠正撤回失败: {e}")
        return False

    async def _generate_correction_response(self, analysis: Optional[Dict] = None) -> Optional[str]:
        if self._use_llm:
            try:
                from src.llm_models.utils_model import LLMRequest
                from src.config.config import model_config
                reason = analysis.get("reason", "说错了") if analysis else "说错了"
                from src.config.prompt_loader import get_prompt, PromptCategory
                prompt = get_prompt(
                    PromptCategory.MODULE,
                    "recall_controller",
                    "recall_correction.template",
                    reason=reason
                )
                request = LLMRequest(model_config.focus_chat, request_type="recall_correction")
                response_text, _ = await request.generate_response_async(prompt)
                if response_text:
                    response_text = response_text.strip()
                    if "[沉默]" in response_text:
                        return None
                    return response_text
            except Exception:
                pass
        return random.choice(["哦哦", "啊好", "嗯嗯", ""])

    def _decide_after_action(self) -> str:
        roll = random.random()
        cumulative = 0.0
        for action, prob in self.AFTER_ACTIONS.items():
            cumulative += prob
            if roll < cumulative:
                return action
        return "correct"

    async def _generate_reaction(self, pending: PendingRecall) -> Optional[str]:
        action = pending.after_action or self._decide_after_action()
        if action == "correct":
            if self._use_llm and len(pending.original_text) > 3:
                try:
                    from src.llm_models.utils_model import LLMRequest
                    from src.config.config import model_config
                    from src.config.prompt_loader import get_prompt, PromptCategory
                    prompt = get_prompt(
                        PromptCategory.MODULE,
                        "recall_controller",
                        "typo_correct.template",
                        original_text=pending.original_text
                    )
                    request = LLMRequest(model_config.focus_chat, request_type="typo_correct")
                    resp, _ = await request.generate_response_async(prompt, max_tokens=len(pending.original_text) + 30)
                    if resp:
                        resp = resp.strip()
                        if resp and len(resp) > 1 and resp != pending.original_text:
                            return resp
                except Exception:
                    pass
            return pending.original_text
        if action == "silent":
            return None
        if action == "regret":
            if self._use_llm:
                try:
                    from src.llm_models.utils_model import LLMRequest
                    from src.config.config import model_config
                    from src.config.prompt_loader import get_prompt, PromptCategory
                    prompt = get_prompt(
                        PromptCategory.MODULE,
                        "recall_controller",
                        "recall_regret.template"
                    )
                    request = LLMRequest(model_config.focus_chat, request_type="recall_regret")
                    response_text, _ = await request.generate_response_async(prompt)
                    if response_text:
                        response_text = response_text.strip()
                        if "[沉默]" in response_text or not response_text:
                            return None
                        return response_text
                except Exception:
                    pass
            return random.choice(["算了", "没事", "当我没说", ""])
        if action == "shy":
            if self._use_llm:
                try:
                    from src.llm_models.utils_model import LLMRequest
                    from src.config.config import model_config
                    from src.config.prompt_loader import get_prompt, PromptCategory
                    prompt = get_prompt(
                        PromptCategory.MODULE,
                        "recall_controller",
                        "recall_shy.template"
                    )
                    request = LLMRequest(model_config.focus_chat, request_type="recall_shy")
                    response_text, _ = await request.generate_response_async(prompt)
                    if response_text:
                        response_text = response_text.strip()
                        if "[沉默]" in response_text or not response_text:
                            return None
                        return response_text
                except Exception:
                    pass
            return None
        return pending.original_text

    def _archive_pending(self, pending: PendingRecall):
        self._pending.pop(pending.msg_id, None)
        self._history.append(pending)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def cancel_pending(self, msg_id: str) -> bool:
        pending = self._pending.pop(msg_id, None)
        if pending and pending.task and not pending.task.done():
            pending.task.cancel()
            return True
        return False

    def get_pending_count(self) -> int:
        return len(self._pending)

    def get_recall_stats(self) -> Dict:
        total = len(self._history)
        by_type = {}
        by_reason = {}
        for p in self._history:
            by_type[p.error_type] = by_type.get(p.error_type, 0) + 1
            by_reason[p.reason] = by_reason.get(p.reason, 0) + 1
        return {"total": total, "by_type": by_type, "by_reason": by_reason}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool):
        self._enabled = value


_controller: Optional[RecallController] = None


def get_recall_controller(config: Optional[Dict] = None) -> RecallController:
    global _controller
    if _controller is None:
        _controller = RecallController(config)
    return _controller


async def process_with_recall(
    original_text: str, group_id: str, stream_id: str,
    send_func: Callable[[str, str], Awaitable[str]],
    delete_func: Callable[[str], Awaitable[bool]],
    config: Optional[Dict] = None,
) -> Tuple[str, bool]:
    controller = get_recall_controller(config)
    should_recall, error_type = controller.should_trigger()
    if not should_recall:
        return original_text, False
    if error_type is None:
        error_type = "random"
    error_text, source = await controller.generate_error_text(original_text, error_type, stream_id)
    msg_id = await send_func(group_id, error_text)
    await controller.schedule_recall(
        msg_id=msg_id, group_id=group_id, original_text=original_text,
        delete_func=delete_func, send_func=send_func, stream_id=stream_id,
        error_text=error_text, error_type=error_type,
    )
    return error_text, True

import asyncio
import time
from typing import Dict, Optional, Callable, Awaitable
from dataclasses import dataclass
from src.common.logger import get_logger

logger = get_logger("recall_monitor")


@dataclass
class MessageMonitorRecord:
    msg_id: str
    content: str
    sent_at: float
    stream_id: str
    user_id: str
    review_weight: float
    reviewed_times: int = 0
    recalled: bool = False
    recall_reason: str = ""
    conversation_context: str = ""
    is_private: bool = False


class MessageRecallMonitor:
    def __init__(self):
        self._monitoring_records: Dict[str, MessageMonitorRecord] = {}
        self._monitor_duration = 120
        self._review_intervals = [5]
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._recall_history: list = []
        self._max_recalls_per_window = 2
        self._recall_window_seconds = 1800

    def _is_recall_allowed(self) -> bool:
        now = time.time()
        cutoff = now - self._recall_window_seconds
        self._recall_history = [t for t in self._recall_history if t > cutoff]
        return len(self._recall_history) < self._max_recalls_per_window

    def _record_recall(self):
        self._recall_history.append(time.time())
    
    async def start_monitoring(
        self,
        msg_id: str,
        content: str,
        stream_id: str,
        user_id: str,
        review_weight: float,
        recall_func: Callable[[str], Awaitable[bool]],
        resend_func: Optional[Callable[[str, str], Awaitable[str]]] = None,
        conversation_context: str = "",
        urgent: bool = False,
        is_private: bool = False,
    ):
        if not content or len(content) < 5:
            return
        
        record = MessageMonitorRecord(
            msg_id=msg_id,
            content=content,
            sent_at=time.time(),
            stream_id=stream_id,
            user_id=user_id,
            review_weight=review_weight,
            conversation_context=conversation_context,
            is_private=is_private,
        )
        self._monitoring_records[msg_id] = record
        
        # 记录到自我学习器
        try:
            from src.modules.modcore.recall_system.self_behavior_learner import get_self_behavior_learner
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            
            learner = get_self_behavior_learner()
            tracker = get_emotion_tracker(stream_id)
            state = tracker.get_user_state(user_id, create_if_missing=False)
            affection = state.affection if state else 0.0
            
            # 异步学习，不阻塞主流程
            asyncio.create_task(learner.learn_from_message(
                content=content,
                stream_id=stream_id,
                affection=affection,
                was_recalled=False,  # 初始未撤回
                recall_reason="",
            ))
        except Exception as e:
            logger.debug(f"记录到自我学习器失败: {e}")
        
        task = asyncio.create_task(
            self._monitor_loop(record, recall_func, resend_func, conversation_context)
        )
        self._running_tasks[msg_id] = task
        logger.info(f"[撤回审查] 开始监控 {msg_id[:8]} | {content[:30]}... (权重: {review_weight:.2f})")
    
    async def _monitor_loop(
        self,
        record: MessageMonitorRecord,
        recall_func: Callable[[str], Awaitable[bool]],
        resend_func: Optional[Callable[[str, str], Awaitable[str]]],
        conversation_context: str
    ):
        try:
            # 5秒后统一进行一次模型审查
            elapsed = time.time() - record.sent_at
            wait_time = 5.0 - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            if record.recalled:
                if record.msg_id in self._monitoring_records:
                    del self._monitoring_records[record.msg_id]
                if record.msg_id in self._running_tasks:
                    del self._running_tasks[record.msg_id]
                return
            
            import random
            import re
            ai_leak = re.search(r'作为ai|语言模型|我是一个ai|我是人工智能|作为一个大型语言模型', record.content.lower())
            if ai_leak:
                should_recall, reason, delay_seconds = True, "AI身份泄露(底线拦截)", 5
            elif record.review_weight < 0.3:
                logger.debug(f"[撤回审查] 权重低({record.review_weight:.2f})，跳过")
                if record.msg_id in self._monitoring_records:
                    del self._monitoring_records[record.msg_id]
                if record.msg_id in self._running_tasks:
                    del self._running_tasks[record.msg_id]
                return
            elif record.review_weight < 0.7:
                if random.random() > 0.2:
                    logger.debug(f"[撤回审查] 权重中等({record.review_weight:.2f})，概率跳过")
                    if record.msg_id in self._monitoring_records:
                        del self._monitoring_records[record.msg_id]
                    if record.msg_id in self._running_tasks:
                        del self._running_tasks[record.msg_id]
                    return
                should_recall, reason, delay_seconds = await self._review_message(record, conversation_context)
            else:
                should_recall, reason, delay_seconds = await self._review_message(record, conversation_context)
            record.reviewed_times += 1
            
            if should_recall:
                if not self._is_recall_allowed():
                    logger.info(f"[撤回审查] 已达撤回上限({self._max_recalls_per_window}次/{self._recall_window_seconds//60}分钟)，跳过")
                    should_recall = False
            if should_recall:
                self._record_recall()
                logger.info(f"[撤回审查] 决定撤回: {reason} | 动作将延迟到第 {delay_seconds} 秒执行")
                current_elapsed = time.time() - record.sent_at
                remaining_wait = delay_seconds - current_elapsed
                if remaining_wait > 0:
                    await asyncio.sleep(remaining_wait)
                
                if not record.recalled:
                    success = await recall_func(record.msg_id)
                    if success:
                        record.recalled = True
                        record.recall_reason = reason
                        logger.info(f"[撤回审查] 撤回成功: {reason} | {record.content[:30]}")
                        
                        # 更新自我学习器：这条消息被撤回了
                        try:
                            from src.modules.modcore.recall_system.self_behavior_learner import get_self_behavior_learner
                            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
                            
                            learner = get_self_behavior_learner()
                            tracker = get_emotion_tracker(record.stream_id)
                            state = tracker.get_user_state(record.user_id, create_if_missing=False)
                            affection = state.affection if state else 0.0
                            
                            # 异步学习，标记为被撤回
                            asyncio.create_task(learner.learn_from_message(
                                content=record.content,
                                stream_id=record.stream_id,
                                affection=affection,
                                was_recalled=True,
                                recall_reason=reason,
                            ))
                        except Exception as e:
                            logger.debug(f"更新自我学习器失败: {e}")
                        
                        if resend_func:
                            await asyncio.sleep(2)
                            # 不再越权自己生成重置文本，只将撤回原因透传回真正的主聊天回复通道
                            await resend_func(record.stream_id, reason)
                            logger.info(f"[撤回审查] 已要求核心生成域带上情绪原因 ({reason}) 重新组织文案")
                    else:
                        logger.warning(f"[撤回审查] 撤回执行失败: {record.msg_id[:8]}")
            else:
                logger.debug(f"[撤回审查] 第{record.reviewed_times}次审查通过（无需撤回）")
            
            if record.msg_id in self._monitoring_records:
                del self._monitoring_records[record.msg_id]
            if record.msg_id in self._running_tasks:
                del self._running_tasks[record.msg_id]
            logger.debug(f"[撤回审查] 结束监控 {record.msg_id[:8]}")
        except Exception as e:
            logger.error(f"[撤回审查] 监控循环异常: {e}")
    
    async def _review_message(self, record: MessageMonitorRecord, context: str) -> tuple:
        """审查消息是否需要撤回（引入上下文感知，支持动态对攻）"""
        try:
            import re
            content_lower = record.content.lower()
            
            # 严重底线内容（依然保留对AI腔调的严厉禁止）
            severe_patterns = [
                r'作为ai|作为AI|语言模型|我是一个ai|我是人工智能|我并不是人类|作为一个大型语言模型|作为.*助手',
                r'非常抱歉给你带来困扰|我这就为您解答',
                # 只有在非常过分的极端人格攻击词汇（超出吐槽范围）时才保留正则硬拦截
                r'操你妈|艹你妈|操你全家',
            ]
            
            # 检查绝对底线
            for pattern in severe_patterns:
                if re.search(pattern, content_lower):
                    return True, f"触发底线规则（腔调露馅或极端辱骂）", 5
            
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            from src.config.config import global_config
            
            # 获取用户好感度和心理状态
            affection = 0
            annoyance = 0
            trauma = 0
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
                tracker = get_emotion_tracker(record.stream_id)
                state = tracker.get_user_state(record.user_id, create_if_missing=False)
                if state:
                    affection = state.affection
                    annoyance = state.annoyance
                    trauma = state.trauma_score
            except Exception:
                pass
            
            # 根据好感度调整审查严格度
            strictness_level = "正常"
            strictness_hint = ""
            if affection >= 50:
                strictness_level = "宽松"
                strictness_hint = "关系极好或处于暧昧期，允许开大尺度的玩笑或打闹"
            elif affection <= -20 or trauma >= 5:
                strictness_level = "严格"
                strictness_hint = "关系较差或有创伤，必须保持警惕和一定的攻击性自卫"
            else:
                strictness_hint = "严谨判断是否出戏，但不仅盯着词汇"
            
            # 获取上下文背景 (非常关键：对方说了什么才导致了我这一句？)
            context_snippet = record.conversation_context if record.conversation_context else "（未知）"
            
            # 获取自我认知
            self_knowledge = ""
            try:
                from src.modules.modcore.recall_system.self_behavior_learner import get_self_behavior_learner
                learner = get_self_behavior_learner()
                self_knowledge = await learner.get_self_knowledge(affection)
                if self_knowledge:
                    self_knowledge = f"\n\n【我的性格自画像】\n{self_knowledge}"
            except Exception:
                pass
            
            bot_name = global_config.bot.nickname
            
            scene_type = "私聊" if record.is_private else "群聊"
            scene_hint = ""
            if record.is_private:
                scene_hint = "私聊场景下语气可以更随意，但要注意不能暴露AI身份。"
            else:
                scene_hint = "群聊场景下其他人都能看到，注意措辞得体。吐槽和玩梗是正常的。"
            
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "recall_monitor",
                "recall_review.template",
                bot_name=bot_name,
                scene_type=scene_type,
                scene_hint=scene_hint,
                context_snippet=context_snippet,
                content=record.content,
                affection=affection,
                annoyance=annoyance,
                trauma=trauma,
                strictness_level=strictness_level,
                strictness_hint=strictness_hint,
                self_knowledge=self_knowledge
            )
            
            model_set = model_config.focus_chat
            model_list = getattr(model_set, "model_list", []) if model_set else []
            model_name = model_list[0] if model_list else "未配置"
            logger.info(f"[撤回审查] 模型决策: {model_name}")

            request = LLMRequest(model_set, request_type="recall_review")
            response, _ = await request.generate_response_async(prompt)
            
            if not response:
                # 如果LLM调用失败，默认不撤回
                return False, "LLM调用失败", 5
            
            text = response.strip()
            # 移除思考标签
            think_match = re.search(r'</think>\s*(.+)', text, re.DOTALL)
            if think_match:
                text = think_match.group(1).strip()
            
            should_recall = "撤回" in text.split('\n')[0]
            reason = ""
            delay_seconds = 5
            
            for line in text.split('\n'):
                if line.startswith("理由"):
                    reason = line.split("：", 1)[-1].strip()
                elif line.startswith("延迟"):
                    try:
                        delay_str = line.split("：", 1)[-1].strip()
                        nums = re.findall(r'\d+', delay_str)
                        if nums:
                            delay_seconds = int(nums[0])
                    except Exception:
                        pass
            
            # 限制延迟不低于5秒且不能无限大
            if delay_seconds < 5:
                delay_seconds = 5
            elif delay_seconds > 120:
                delay_seconds = 120
            
            return should_recall, reason or "LLM审查不通过", delay_seconds
        except Exception as e:
            logger.error(f"[撤回审查] 审查失败: {e}")
            return False, "", 5
    
    # _regenerate_content 方法已被用户指正废弃：
    # LLM生成过程不能四分五裂。撤回模块只做撤回决策和透传原因，而真正带情绪重构的文字必须交给原生的主生成器统一生成！
    
    def stop_monitoring(self, msg_id: str):
        if msg_id in self._running_tasks:
            self._running_tasks[msg_id].cancel()
            del self._running_tasks[msg_id]
        if msg_id in self._monitoring_records:
            del self._monitoring_records[msg_id]
        logger.debug(f"[撤回审查] 停止监控 {msg_id[:8]}")
    
    def get_monitoring_status(self, msg_id: str) -> Optional[MessageMonitorRecord]:
        return self._monitoring_records.get(msg_id)


_monitor_instance: Optional[MessageRecallMonitor] = None


def get_recall_monitor() -> MessageRecallMonitor:
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = MessageRecallMonitor()
    return _monitor_instance

"""
自我行为学习器 - 从记忆库学习自己的表达风格

功能：
1. 从记忆库提取自己的历史消息
2. 定期用小模型总结表达风格
3. 使用嵌入式模型做相似度匹配
4. 动态更新模式库，遗忘旧模式
5. 为撤回判断提供自我认知基础
"""

import time
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.common.logger import get_logger

logger = get_logger("self_behavior_learner")


@dataclass
class StyleSummary:
    """风格总结"""
    summary_text: str  # 总结文本
    created_at: float  # 创建时间
    message_count: int  # 基于多少条消息
    affection_range: Tuple[float, float]  # 适用的好感度范围
    success_rate: float = 1.0  # 成功率


class SelfBehaviorLearner:
    """自我行为学习器 - 基于记忆库+定期总结"""
    
    def __init__(self, summary_interval: float = 3600.0):
        self._style_summaries: Dict[str, StyleSummary] = {}  # 按好感度范围存储总结
        self._summary_interval = summary_interval  # 总结间隔（秒）
        self._last_summary_time = 0.0
        self._learning_enabled = True
        self._summary_task = None
        
        # 消息计数器（用于判断是否需要重新总结）
        self._message_count_since_summary = 0
        self._min_messages_for_summary = 20  # 至少20条新消息才重新总结
    
    async def start_periodic_summary(self):
        """启动定期总结任务"""
        if self._summary_task and not self._summary_task.done():
            return
        
        self._summary_task = asyncio.create_task(self._summary_loop())
        logger.info("🎓 自我学习定期总结已启动")
    
    async def stop_periodic_summary(self):
        """停止定期总结任务"""
        if self._summary_task and not self._summary_task.done():
            self._summary_task.cancel()
            try:
                await self._summary_task
            except asyncio.CancelledError:
                pass
        logger.info("🎓 自我学习定期总结已停止")
    
    async def _summary_loop(self):
        """定期总结循环"""
        while self._learning_enabled:
            try:
                await asyncio.sleep(self._summary_interval)
                
                # 检查是否需要总结
                if self._message_count_since_summary >= self._min_messages_for_summary:
                    await self._generate_style_summary()
                    self._message_count_since_summary = 0
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"定期总结失败: {e}")
                await asyncio.sleep(60)
    
    async def learn_from_message(
        self,
        content: str,
        stream_id: str,
        affection: float = 0.0,
        was_recalled: bool = False,
        recall_reason: str = "",
    ):
        """记录消息（轻量级，只计数）"""
        if not self._learning_enabled or not content:
            return
        
        self._message_count_since_summary += 1
        
        # 如果被撤回，降低对应风格总结的成功率
        if was_recalled:
            await self._update_summary_success_rate(affection, False)
    
    async def _update_summary_success_rate(self, affection: float, success: bool):
        """更新风格总结的成功率"""
        # 找到适用的总结
        for key, summary in self._style_summaries.items():
            if summary.affection_range[0] <= affection <= summary.affection_range[1]:
                if success:
                    summary.success_rate = min(1.0, summary.success_rate * 1.05 + 0.05)
                else:
                    summary.success_rate = summary.success_rate * 0.9
                break
    
    async def _generate_style_summary(self):
        """从记忆库生成风格总结（使用小模型）"""
        try:
            from src.memory_system.memory_core import get_memory_core
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config, global_config
            
            memory_core = get_memory_core()
            if not memory_core:
                logger.debug("记忆库未初始化")
                return
            
            # 从记忆库提取自己最近的消息
            bot_qq = str(global_config.bot.qq_account)
            
            # 尝试调用记忆库方法
            try:
                recent_messages = await memory_core.get_recent_bot_messages(
                    bot_id=bot_qq,
                    limit=100,
                    time_range=7 * 24 * 3600,
                )
            except AttributeError:
                # 如果方法不存在，使用备用方案
                logger.debug("记忆库方法不存在，跳过总结")
                return
            
            if len(recent_messages) < 10:
                logger.debug("消息数量不足，跳过总结")
                return
            
            # 按好感度分组
            affection_groups = {
                "high": [],  # 好感度 >= 30
                "medium": [],  # -10 < 好感度 < 30
                "low": [],  # 好感度 <= -10
            }
            
            for msg in recent_messages:
                affection = msg.get("affection", 0)
                content = msg.get("content", "")
                
                if not content or len(content) < 3:
                    continue
                
                if affection >= 30:
                    affection_groups["high"].append(content)
                elif affection <= -10:
                    affection_groups["low"].append(content)
                else:
                    affection_groups["medium"].append(content)
            
            # 为每个好感度范围生成总结
            for group_name, messages in affection_groups.items():
                if len(messages) < 5:
                    continue
                
                # 使用小模型总结
                sample_messages = messages[:30]  # 最多30条样本
                messages_text = "\n".join([f"- {msg}" for msg in sample_messages])
                
                from src.config.prompt_loader import get_prompt, PromptCategory
                prompt = get_prompt(
                    PromptCategory.MODULE,
                    "self_awareness",
                    "self_summary.template",
                    msg_count=len(sample_messages),
                    messages_text=messages_text
                )
                
                request = LLMRequest(model_config.focus_chat, request_type="self_summary")
                response, _ = await request.generate_response_async(prompt, max_tokens=200, temperature=0.5)
                
                if response and response.strip():
                    # 保存总结
                    affection_range = {
                        "high": (30, 100),
                        "medium": (-10, 30),
                        "low": (-100, -10),
                    }[group_name]
                    
                    key = f"{group_name}_{int(time.time())}"
                    self._style_summaries[key] = StyleSummary(
                        summary_text=response.strip(),
                        created_at=time.time(),
                        message_count=len(messages),
                        affection_range=affection_range,
                    )
                    
                    logger.info(f"生成风格总结: [{group_name}] 基于{len(messages)}条消息")
            
            # 清理旧总结（保留最新的3个）
            if len(self._style_summaries) > 9:  # 每个范围最多3个
                sorted_summaries = sorted(
                    self._style_summaries.items(),
                    key=lambda x: x[1].created_at,
                    reverse=True
                )
                self._style_summaries = dict(sorted_summaries[:9])
            
            self._last_summary_time = time.time()
            
        except Exception as e:
            logger.error(f"生成风格总结失败: {e}")
    
    async def get_self_knowledge(
        self,
        affection: float = 0.0,
        pattern_types: Optional[List[str]] = None,
    ) -> str:
        """获取自我认知（从缓存的总结中获取）"""
        # 查找适用的总结
        applicable_summaries = [
            s for s in self._style_summaries.values()
            if s.affection_range[0] <= affection <= s.affection_range[1]
            and s.success_rate > 0.3
        ]
        
        if not applicable_summaries:
            # 如果没有总结，触发一次总结
            if time.time() - self._last_summary_time > 300:  # 5分钟后可以重试
                asyncio.create_task(self._generate_style_summary())
            return "我还在学习自己的表达风格"
        
        # 返回最新的总结
        latest = max(applicable_summaries, key=lambda s: s.created_at)
        return latest.summary_text
    
    async def is_normal_for_me(
        self,
        content: str,
        affection: float = 0.0,
    ) -> Tuple[bool, float, str]:
        """
        判断这个表达是否符合我的风格（使用嵌入式模型或LLM）
        
        Returns:
            (是否正常, 置信度, 原因)
        """
        # 获取风格总结
        self_knowledge = await self.get_self_knowledge(affection)
        
        if "还在学习" in self_knowledge:
            return True, 0.5, "知识库不足，无法判断"
        
        try:
            # 尝试使用嵌入式模型做相似度匹配
            similarity = await self._check_similarity_with_embedding(content, affection)
            if similarity is not None:
                if similarity >= 0.7:
                    return True, similarity, "符合我的表达风格"
                elif similarity >= 0.4:
                    return True, similarity, "可能是我的新表达方式"
                else:
                    return False, similarity, "不太符合我的风格"
        except Exception as e:
            logger.debug(f"嵌入式模型判断失败: {e}")
        
        # 如果嵌入式模型不可用，使用LLM快速判断
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "self_awareness",
                "style_check.template",
                self_knowledge=self_knowledge,
                content=content
            )
            
            request = LLMRequest(model_config.focus_chat, request_type="style_check")
            response, _ = await request.generate_response_async(prompt, max_tokens=50, temperature=0.3)
            
            if response and '|' in response:
                parts = response.strip().split('|')
                if len(parts) >= 3:
                    is_normal = "符合" in parts[0]
                    confidence = float(parts[1])
                    reason = parts[2]
                    return is_normal, confidence, reason
        except Exception as e:
            logger.debug(f"LLM判断失败: {e}")
        
        return True, 0.5, "无法判断"
    
    async def _check_similarity_with_embedding(
        self,
        content: str,
        affection: float,
    ) -> Optional[float]:
        """使用嵌入式模型检查相似度"""
        try:
            from src.memory_system.memory_core import get_memory_core
            from src.config.config import global_config
            
            memory_core = get_memory_core()
            if not memory_core or not hasattr(memory_core, 'embedding_model'):
                return None
            
            # 从记忆库获取相似的历史消息
            bot_qq = str(global_config.bot.qq_account)
            similar_messages = await memory_core.search_similar_bot_messages(
                query=content,
                bot_id=bot_qq,
                affection_range=(affection - 30, affection + 30),
                limit=5,
            )
            
            if not similar_messages:
                return None
            
            # 计算平均相似度
            similarities = [msg.get("similarity", 0) for msg in similar_messages]
            avg_similarity = sum(similarities) / len(similarities)
            
            return avg_similarity
            
        except Exception as e:
            logger.debug(f"嵌入式模型检查失败: {e}")
            return None
    
    async def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_summaries": len(self._style_summaries),
            "messages_since_summary": self._message_count_since_summary,
            "last_summary_time": self._last_summary_time,
        }
    
    def set_learning_enabled(self, enabled: bool):
        """启用/禁用学习"""
        self._learning_enabled = enabled
        logger.info(f"自我学习: {'启用' if enabled else '禁用'}")


# 全局实例
_global_learner: Optional[SelfBehaviorLearner] = None


def get_self_behavior_learner() -> SelfBehaviorLearner:
    """获取全局自我行为学习器"""
    global _global_learner
    if _global_learner is None:
        _global_learner = SelfBehaviorLearner()
        # 启动定期总结
        asyncio.create_task(_global_learner.start_periodic_summary())
    return _global_learner

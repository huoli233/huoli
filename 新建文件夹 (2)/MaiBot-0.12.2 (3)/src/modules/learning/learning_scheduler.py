import asyncio
import time
from typing import Dict, List, Optional
from src.common.logger import get_logger
from src.modules.learning.dynamic_vocabulary import get_dynamic_vocabulary

logger = get_logger("learning_scheduler")


class LearningScheduler:
    """学习任务调度器 - 自动触发学习任务"""
    
    def __init__(self):
        self.vocab_learner = None
        self.learning_interval = 300  # 5分钟学习一次
        self.last_learning_times = {}  # 记录每个聊天的上次学习时间
        self._running = False
        self._task = None
    
    async def start(self):
        """启动学习调度器"""
        if self._running:
            return
        
        self._running = True
        try:
            self.vocab_learner = get_dynamic_vocabulary()
            logger.info("🎓 学习调度器已启动")
            self._task = asyncio.create_task(self._learning_loop())
        except Exception as e:
            logger.error(f"启动学习调度器失败: {e}")
            self._running = False
    
    async def stop(self):
        """停止学习调度器"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("🎓 学习调度器已停止")
    
    async def _learning_loop(self):
        """学习循环"""
        while self._running:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次
                
                if not self.vocab_learner:
                    continue
                
                # 推断黑话含义
                await self.vocab_learner.infer_jargon_meanings()
                
                await asyncio.sleep(5)  # 避免过于频繁
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"学习循环异常: {e}")
                await asyncio.sleep(30)
    
    async def trigger_learning(self, chat_id: str, messages: List[Dict], user_id: str = None) -> Dict:
        """
        触发学习任务
        
        Args:
            chat_id: 聊天ID
            messages: 消息列表
            user_id: 用户ID（可选）
            
        Returns:
            学习结果统计
        """
        if not self.vocab_learner:
            try:
                self.vocab_learner = get_dynamic_vocabulary()
            except Exception as e:
                logger.error(f"获取学习引擎失败: {e}")
                return {"expressions": 0, "jargons": 0}
        
        # 检查学习频率限制
        current_time = time.time()
        last_time = self.last_learning_times.get(chat_id, 0)
        
        if current_time - last_time < 30:  # 30秒内不重复学习
            return {"expressions": 0, "jargons": 0}
        
        try:
            # 执行学习
            result = await self.vocab_learner.learn_from_messages(
                messages=messages,
                chat_id=chat_id,
                user_id=user_id
            )
            
            # 更新学习时间
            self.last_learning_times[chat_id] = current_time
            
            # 清理旧记录
            if len(self.last_learning_times) > 100:
                cutoff_time = current_time - 3600  # 保留1小时内的记录
                self.last_learning_times = {
                    cid: t for cid, t in self.last_learning_times.items() 
                    if t > cutoff_time
                }
            
            return result
            
        except Exception as e:
            logger.error(f"学习任务执行失败: {e}")
            return {"expressions": 0, "jargons": 0}
    
    async def get_learning_state(self, chat_id: str, user_id: str = None) -> str:
        """
        获取学习状态
        
        Args:
            chat_id: 聊天ID  
            user_id: 用户ID（可选）
            
        Returns:
            学习状态提示文本
        """
        if not self.vocab_learner:
            try:
                self.vocab_learner = get_dynamic_vocabulary()
            except Exception as e:
                logger.error(f"获取学习引擎失败: {e}")
                return """【认知状态 - 天真无邪】
你对大部分话题都保持天真无邪的态度，对敏感话题、暗示性内容、成人向话题一窍不通。
当遇到不理解的内容时：
- 用字面意思理解对方的话
- 表现出真实的困惑："你在说什么？"
- 不会意识到对方的言外之意或暗示
- 可以岔开话题或转移注意力
你不会假装理解，也不会顺着对方的暗示接话。"""
        
        try:
            return await self.vocab_learner.get_user_learning_state(
                chat_id=chat_id,
                user_id=user_id
            )
        except Exception as e:
            logger.error(f"获取学习状态失败: {e}")
            return """【认知状态 - 天真无邪】
你对大部分话题都保持天真无邪的态度，对敏感话题、暗示性内容、成人向话题一窍不通。
当遇到不理解的内容时：
- 用字面意思理解对方的话
- 表现出真实的困惑："你在说什么？"
- 不会意识到对方的言外之意或暗示
- 可以岔开话题或转移注意力
你不会假装理解，也不会顺着对方的暗示接话。"""
    
    async def explain_jargons(self, messages: List[Dict]) -> str:
        """解释对话中的黑话"""
        if not self.vocab_learner:
            return ""
        
        try:
            return await self.vocab_learner.explain_jargons_in_context(messages)
        except Exception as e:
            logger.debug(f"解释黑话失败: {e}")
            return ""
    
    async def select_expressions(self, chat_id: str, chat_info: str, max_num: int = 5) -> List[Dict]:
        """选择合适的表达方式"""
        if not self.vocab_learner:
            return []
        
        try:
            return await self.vocab_learner.select_expressions(
                chat_id=chat_id,
                chat_info=chat_info,
                max_num=max_num
            )
        except Exception as e:
            logger.debug(f"选择表达方式失败: {e}")
            return []
    
    async def get_stats(self) -> Dict:
        """获取学习统计信息"""
        if not self.vocab_learner:
            return {"total": 0, "expressions": 0, "jargons": 0}
        
        try:
            return await self.vocab_learner.get_vocabulary_stats()
        except Exception as e:
            logger.debug(f"获取统计信息失败: {e}")
            return {"total": 0, "expressions": 0, "jargons": 0}


# 全局学习调度器实例
_global_learning_scheduler: Optional[LearningScheduler] = None


async def get_learning_scheduler() -> LearningScheduler:
    """获取全局学习调度器实例"""
    global _global_learning_scheduler
    
    if _global_learning_scheduler is None:
        _global_learning_scheduler = LearningScheduler()
        await _global_learning_scheduler.start()
    
    return _global_learning_scheduler


async def cleanup_learning_scheduler():
    """清理学习调度器"""
    global _global_learning_scheduler
    
    if _global_learning_scheduler:
        await _global_learning_scheduler.stop()
        _global_learning_scheduler = None

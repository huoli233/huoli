import asyncio
import time
from typing import Callable, Dict, Optional
from src.common.logger import get_logger

logger = get_logger('reply_coordinator')

# 优先级定义
PRIORITY_PASSIVE = 1   # 被动接收（被@）- 最高
PRIORITY_MONITOR = 2   # 独立监控（模型决定）- 中等
PRIORITY_ACTIVE = 3    # 主动循环（主动思考）- 最低


class ReplyCoordinator:
    """
    回复协调器
    
    职责：
    1. 确保同一时间只有一个通道在回复
    2. 根据优先级决定哪个通道可以回复
    3. 记录最近回复信息
    """
    
    def __init__(self, reply_cooldown: float = 5.0, lock_timeout: float = 30.0):
        """
        初始化回复协调器
        
        Args:
            reply_cooldown: 回复冷却时间（秒）
            lock_timeout: 锁超时时间（秒）
        """
        self._reply_lock = asyncio.Lock()
        self._last_reply_time: float = 0.0
        self._last_reply_channel: str = ""
        self._reply_cooldown = reply_cooldown
        self._lock_timeout = lock_timeout
        
        logger.info(f"回复协调器初始化完成 | 冷却={reply_cooldown}s 超时={lock_timeout}s")
    
    def _get_channel_priority(self, channel: str) -> int:
        """获取通道优先级"""
        priority_map = {
            "passive": PRIORITY_PASSIVE,
            "monitor": PRIORITY_MONITOR,
            "active": PRIORITY_ACTIVE,
        }
        return priority_map.get(channel, 999)
    
    async def acquire_reply_permission(
        self, 
        channel: str, 
        priority: int
    ) -> bool:
        """
        请求回复权限
        
        Args:
            channel: 通道名称 ("passive" | "monitor" | "active")
            priority: 优先级 (1=最高, 2=中, 3=低)
            
        Returns:
            是否获得权限
        """
        # 尝试获取锁（带超时）
        try:
            await asyncio.wait_for(
                self._reply_lock.acquire(), 
                timeout=self._lock_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"[{channel}] 获取回复锁超时，放弃本次回复")
            return False
        
        # 检查最近是否刚回复过
        now = time.time()
        if now - self._last_reply_time < self._reply_cooldown:
            # 冷却时间内，只有更高优先级才能打断
            last_priority = self._get_channel_priority(self._last_reply_channel)
            if priority >= last_priority:
                self._reply_lock.release()
                logger.debug(
                    f"[{channel}] 回复冷却中 | "
                    f"上次={self._last_reply_channel}(优先级{last_priority}) "
                    f"本次优先级{priority} | 放弃"
                )
                return False
        
        logger.info(f"[{channel}] 获得回复权限 | 优先级={priority}")
        return True
    
    async def execute_reply(
        self, 
        channel: str, 
        reply_func: Callable,
        *args,
        **kwargs
    ) -> bool:
        """
        执行回复（带锁保护）
        
        Args:
            channel: 通道名称
            reply_func: 回复函数
            *args, **kwargs: 传递给回复函数的参数
            
        Returns:
            是否成功回复
        """
        priority = self._get_channel_priority(channel)
        
        # 请求权限
        if not await self.acquire_reply_permission(channel, priority):
            return False
        
        try:
            # 执行回复
            logger.info(f"[{channel}] 开始执行回复")
            result = await reply_func(*args, **kwargs)
            
            # 更新状态
            self.update_reply_state(channel)
            
            logger.info(f"[{channel}] 回复执行完成")
            return result
        
        except Exception as e:
            logger.error(f"[{channel}] 回复执行失败: {e}", exc_info=True)
            return False
        
        finally:
            # 释放锁
            if self._reply_lock.locked():
                self._reply_lock.release()
    
    def update_reply_state(self, channel: str) -> None:
        """
        更新回复状态
        
        Args:
            channel: 通道名称
        """
        self._last_reply_time = time.time()
        self._last_reply_channel = channel
        logger.debug(f"[{channel}] 更新回复状态 | 时间={self._last_reply_time}")
    
    def get_last_reply_info(self) -> Dict:
        """
        获取最近回复信息
        
        Returns:
            包含最近回复信息的字典
        """
        return {
            "last_reply_time": self._last_reply_time,
            "last_reply_channel": self._last_reply_channel,
            "time_since_last_reply": time.time() - self._last_reply_time,
            "is_in_cooldown": time.time() - self._last_reply_time < self._reply_cooldown,
        }
    
    def can_reply_now(self, channel: str) -> bool:
        """
        检查当前是否可以回复（不获取锁）
        
        Args:
            channel: 通道名称
            
        Returns:
            是否可以回复
        """
        now = time.time()
        
        # 检查冷却时间
        if now - self._last_reply_time < self._reply_cooldown:
            priority = self._get_channel_priority(channel)
            last_priority = self._get_channel_priority(self._last_reply_channel)
            if priority >= last_priority:
                return False
        
        # 检查锁是否被占用
        if self._reply_lock.locked():
            return False
        
        return True

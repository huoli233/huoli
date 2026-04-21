import asyncio
import time
from typing import Optional, List, Dict
from loguru import logger

from src.plugin_system.apis import message_api


class IndependentMonitor:
    """
    完全独立的后台检测系统
    
    特点：
    1. 不依赖主循环的 running 标志
    2. 有自己独立的生命周期
    3. 专注于小模型检测和状态管理
    4. 通过事件通知主循环
    """
    
    def __init__(self, chat_id: str, heartfc_instance):
        self.chat_id = chat_id
        self.heartfc = heartfc_instance
        self.log_prefix = f"[独立监控] [{heartfc_instance.stream_name}]"
        
        # 独立的控制标志 - 不依赖任何其他标志
        self._monitor_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # 状态机
        self._state = "idle"  # idle/peeking/observing
        self._state_start_time = 0.0
        self._state_duration = 0.0
        self._state_messages: List = []
        
        # 最后读取时间
        self._last_check_time = time.time()
        
        # 配置
        self._check_interval = 5.0  # 检查间隔（秒）
        
        # 窥屏时长范围
        self._peek_min_duration = 30.0
        self._peek_max_duration = 120.0
        
        # 观望时长范围
        self._observe_min_duration = 60.0
        self._observe_max_duration = 300.0
        
        logger.info(f"{self.log_prefix} 🔍 独立监控系统初始化完成")
    
    @property
    def is_running(self) -> bool:
        """监控是否正在运行"""
        return self._monitor_active
    
    @property
    def current_state(self) -> str:
        """当前状态"""
        return self._state

    def _get_psych(self, user_id: str) -> dict:
        """统一获取并归一化心理上下文，避免字段名不一致"""
        if not hasattr(self.heartfc, '_get_psychological_context'):
            return {
                "favor": 0,
                "annoyance": 0.0,
                "mental_fatigue": 0.0,
                "trauma": 0.0,
                "relationship": "陌生人",
                "trust": 0,
            }
        raw = self.heartfc._get_psychological_context(user_id)
        return self._normalize_psych_context(raw)

    def _normalize_psych_context(self, raw: dict) -> dict:
        """将外部心理字段映射为内部统一字段"""
        return {
            "favor": raw.get('affection', raw.get('favor', 0)),
            "annoyance": raw.get('annoyance', 0.0),
            "mental_fatigue": raw.get('fatigue', raw.get('mental_fatigue', 0.0)),
            "trauma": raw.get('trauma_score', raw.get('trauma', 0.0)),
            "relationship": raw.get('relationship', '陌生人'),
            "trust": raw.get('trust', 0),
        }
    
    async def start(self):
        """启动独立监控"""
        if self._monitor_active:
            logger.warning(f"{self.log_prefix} 独立监控已在运行")
            return
        
        self._monitor_active = True
        self._last_check_time = time.time()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info(f"{self.log_prefix} 🔍 独立监控系统启动")
    
    async def stop(self):
        """停止独立监控"""
        if not self._monitor_active:
            return
        
        self._monitor_active = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"{self.log_prefix} 🔍 独立监控系统停止")
    
    async def _monitor_loop(self):
        """
        独立监控循环 - 核心
        
        这个循环完全独立运行，不受主循环影响
        """
        logger.info(f"{self.log_prefix} 🔍 监控循环开始")
        
        try:
            while self._monitor_active:
                now = time.time()
                
                # 1. 检查状态是否超时
                await self._check_state_timeout(now)
                
                # 2. 获取新消息
                recent_messages = await self._fetch_new_messages()
                
                # 3. 处理新消息
                if recent_messages:
                    await self._process_messages(recent_messages, now)
                
                # 4. 等待下一次检查
                await asyncio.sleep(self._check_interval)
        
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} 🔍 监控循环被取消")
            raise
        except Exception as e:
            logger.error(f"{self.log_prefix} 🔍 监控循环异常: {e}", exc_info=True)
        finally:
            self._monitor_active = False
            logger.info(f"{self.log_prefix} 🔍 监控循环结束")
    
    async def _check_state_timeout(self, now: float):
        """检查状态是否超时"""
        if self._state in ["peeking", "observing"]:
            elapsed = now - self._state_start_time
            
            if elapsed >= self._state_duration:
                # 状态时间到了
                state_name = "窥屏" if self._state == "peeking" else "观望"
                msg_count = len(self._state_messages)
                
                logger.info(
                    f"{self.log_prefix} 🔍 [{state_name}结束] "
                    f"收集了{msg_count}条消息，回到idle"
                )
                
                # 如果是观望结束，可以进行总结判断
                if self._state == "observing" and msg_count > 0:
                    await self._handle_observing_end()
                
                # 重置状态
                self._state = "idle"
                self._state_messages = []
    
    async def _fetch_new_messages(self) -> List:
        """获取新消息"""
        try:
            messages = message_api.get_messages_by_time_in_chat(
                chat_id=self.chat_id,
                start_time=self._last_check_time,
                end_time=time.time(),
                limit=20,
                limit_mode="latest",
                filter_mai=True,
                filter_command=False,
                filter_intercept_message_level=0,
            )
            
            if messages:
                # 更新最后检查时间
                self._last_check_time = time.time()
            
            return messages
        
        except Exception as e:
            logger.debug(f"{self.log_prefix} 🔍 获取消息失败: {e}")
            return []
    
    async def _process_messages(self, messages: List, now: float):
        """处理新消息"""
        # 检查是否被@
        last_msg = messages[-1]
        is_at_me = getattr(last_msg, 'is_mentioned', False) or getattr(last_msg, 'is_at', False)
        
        # 如果被@，立即中断当前状态
        if is_at_me and self._state in ["peeking", "observing"]:
            logger.info(f"{self.log_prefix} 🔍 [状态中断] 被@了，立即结束{self._state}")
            self._state = "idle"
            self._state_messages = []
            
            # 通知主循环处理这条消息
            await self._notify_urgent_reply(last_msg)
            return
        
        # 根据当前状态处理
        if self._state == "idle":
            # idle状态：运行小模型判断
            await self._decide_next_state(messages, now)
        
        elif self._state in ["peeking", "observing"]:
            # 窥屏/观望状态：只收集消息
            for msg in messages:
                msg_id = getattr(msg, 'message_id', None) or id(msg)
                existing_ids = [
                    getattr(m, 'message_id', None) or id(m) 
                    for m in self._state_messages
                ]
                if msg_id not in existing_ids:
                    self._state_messages.append(msg)
            
            logger.debug(
                f"{self.log_prefix} 🔍 [{self._state}] "
                f"收集消息，当前{len(self._state_messages)}条"
            )
    
    async def _decide_next_state(self, messages: List, now: float):
        """
        运行小模型判断下一个状态
        
        这里是核心决策逻辑
        """
        last_msg = messages[-1]
        user_id = getattr(last_msg, 'sender', {}).get('user_id', '')
        
        # 获取心理状态
        psych = self._get_psych(user_id)
        
        # 简化的决策逻辑（可以调用小模型）
        # 这里先用规则，后续可以替换为小模型调用
        
        # 检查关系
        relationship = psych.get('relationship', '陌生人')
        favor = psych.get('favor', 0)
        
        # 决策
        if favor < 20:
            # 关系不够，进入窥屏
            duration = self._peek_min_duration + (self._peek_max_duration - self._peek_min_duration) * 0.5
            self._enter_state("peeking", duration, "关系不够")
        
        elif favor < 50:
            # 关系一般，可能观望
            duration = self._observe_min_duration + (self._observe_max_duration - self._observe_min_duration) * 0.3
            self._enter_state("observing", duration, "关系一般，观望一下")
        
        else:
            # 关系好，通知主循环回复
            logger.info(f"{self.log_prefix} 🔍 [决策] 关系好，通知回复")
            await self._notify_reply_needed(last_msg)
    
    def _enter_state(self, state: str, duration: float, reason: str):
        """进入新状态"""
        self._state = state
        self._state_start_time = time.time()
        self._state_duration = duration
        self._state_messages = []
        
        state_name = "窥屏" if state == "peeking" else "观望"
        logger.info(
            f"{self.log_prefix} 🔍 [进入{state_name}] {reason} | "
            f"持续{int(duration)}秒"
        )
    
    async def _handle_observing_end(self):
        """处理观望结束"""
        # 这里可以调用小模型总结观望期间的消息
        # 并决定是否需要回复
        
        logger.info(
            f"{self.log_prefix} 🔍 [观望总结] "
            f"观望期间收集了{len(self._state_messages)}条消息"
        )
        
        # 简化：观望结束后不主动回复
        # 如果需要回复，可以调用 _notify_reply_needed
    
    async def _notify_reply_needed(self, message):
        """通知主循环需要回复"""
        # 这里可以通过事件总线或回调通知主循环
        # 暂时只记录日志
        
        logger.info(
            f"{self.log_prefix} 🔍 [通知回复] "
            f"检测到需要回复的消息"
        )
        
        # 如果主循环在运行，可以设置标志让它处理
        if hasattr(self.heartfc, 'running') and self.heartfc.running:
            # 更新 last_read_time 让主循环重新读取
            self.heartfc.last_read_time = time.time() - 1
    
    async def _notify_urgent_reply(self, message):
        """通知紧急回复（被@等）"""
        logger.info(
            f"{self.log_prefix} 🔍 [紧急通知] "
            f"检测到紧急消息（被@）"
        )
        
        # 强制触发主循环处理
        if hasattr(self.heartfc, 'running') and self.heartfc.running:
            self.heartfc.last_read_time = time.time() - 1
    
    def get_status(self) -> Dict:
        """获取监控状态"""
        return {
            "active": self._monitor_active,
            "state": self._state,
            "state_duration": self._state_duration,
            "messages_collected": len(self._state_messages),
            "check_interval": self._check_interval,
        }

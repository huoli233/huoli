import asyncio
import time
import random
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from src.common.logger import get_logger

logger = get_logger("emotion_driven_proactive")


@dataclass
class EmotionState:
    """AI当前情感状态"""
    boredom_level: float = 0.0  # 无聊程度 0-1
    social_desire: float = 0.5  # 社交欲望 0-1  
    mood_score: float = 0.5  # 心情指数 0-1
    energy_level: float = 1.0  # 能量水平 0-1
    last_interaction_time: float = 0.0
    loneliness_accumulation: float = 0.0  # 孤独感累积
    proactive_willingness: float = 0.5  # 主动发言意愿


class EmotionDrivenProactiveSystem:
    """基于AI情感状态的自然主动发言系统
    
    核心理念：
    1. 用户停止聊天几分钟后开始情感监控
    2. AI根据无聊感、孤独感等情感状态自主决定是否想说话
    3. 不再依赖固定时间触发，而是基于真实情感驱动
    """
    
    def __init__(self):
        self._channel_emotions: Dict[str, EmotionState] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._enabled = True
        
        # 情感参数配置
        self.MIN_SILENCE_MINUTES = 3  # 最少沉默3分钟后开始情感监控
        self.EMOTION_CHECK_INTERVAL = 60  # 每分钟检查一次情感状态
        self.BOREDOM_ACCUMULATION_RATE = 0.1  # 无聊感累积速率
        self.LONELINESS_THRESHOLD = 0.6  # 孤独感阈值
        self.PROACTIVE_THRESHOLD = 0.7  # 主动发言阈值
        
        logger.info("🎭 情感驱动主动发言系统已初始化")
    
    async def on_user_message(self, channel_id: str, user_id: str = "") -> None:
        """用户发送消息时调用，重置情感状态并开始监控"""
        try:
            # 停止当前监控
            await self._stop_emotion_monitoring(channel_id)
            
            # 重置情感状态
            emotion_state = self._get_or_create_emotion_state(channel_id)
            emotion_state.last_interaction_time = time.time()
            emotion_state.boredom_level = 0.0
            emotion_state.loneliness_accumulation = 0.0
            
            # 几分钟后开始情感监控
            delay_seconds = self.MIN_SILENCE_MINUTES * 60
            task = asyncio.create_task(
                self._start_emotion_monitoring_after_delay(channel_id, delay_seconds),
                name=f"emotion_monitor_{channel_id[:8]}"
            )
            self._monitoring_tasks[channel_id] = task
            
            logger.debug(f"[情感系统] {channel_id[:8]} 用户消息，{self.MIN_SILENCE_MINUTES}分钟后开始情感监控")
        except Exception as e:
            logger.error(f"[情感系统] 处理用户消息失败: {e}")
    
    async def on_bot_message(self, channel_id: str, is_proactive: bool = False) -> None:
        """Bot发送消息时调用"""
        try:
            emotion_state = self._get_or_create_emotion_state(channel_id)
            emotion_state.last_interaction_time = time.time()
            
            if is_proactive:
                # 主动发言后降低社交欲望和无聊感
                emotion_state.social_desire = max(0.0, emotion_state.social_desire - 0.3)
                emotion_state.boredom_level = max(0.0, emotion_state.boredom_level - 0.5)
                emotion_state.loneliness_accumulation = max(0.0, emotion_state.loneliness_accumulation - 0.4)
                logger.debug(f"[情感系统] {channel_id[:8]} 主动发言后情感状态已调整")
                
                # 重新开始情感监控
                await self._stop_emotion_monitoring(channel_id)
                delay_seconds = self.MIN_SILENCE_MINUTES * 60
                task = asyncio.create_task(
                    self._start_emotion_monitoring_after_delay(channel_id, delay_seconds),
                    name=f"emotion_monitor_{channel_id[:8]}"
                )
                self._monitoring_tasks[channel_id] = task
        except Exception as e:
            logger.error(f"[情感系统] 处理Bot消息失败: {e}")
    
    async def _start_emotion_monitoring_after_delay(self, channel_id: str, delay_seconds: float) -> None:
        """延迟后开始情感监控"""
        try:
            await asyncio.sleep(delay_seconds)
            await self._emotion_monitoring_loop(channel_id)
        except asyncio.CancelledError:
            logger.debug(f"[情感系统] {channel_id[:8]} 情感监控任务被取消")
        except Exception as e:
            logger.error(f"[情感系统] {channel_id[:8]} 情感监控失败: {e}")
    
    async def _emotion_monitoring_loop(self, channel_id: str) -> None:
        """情感监控主循环"""
        logger.info(f"🎭 [情感系统] {channel_id[:8]} 开始情感监控")
        
        while self._enabled:
            try:
                await asyncio.sleep(self.EMOTION_CHECK_INTERVAL)
                
                # 更新情感状态
                await self._update_emotion_state(channel_id)
                
                # 检查是否想要主动发言
                should_speak, reason = await self._evaluate_proactive_desire(channel_id)
                
                if should_speak:
                    logger.info(f"🎯 [情感系统] {channel_id[:8]} AI想要主动发言: {reason}")
                    await self._trigger_proactive_speech(channel_id, reason)
                    break  # 发言后结束当前监控循环
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[情感系统] {channel_id[:8]} 情感监控循环错误: {e}")
                break
    
    async def _update_emotion_state(self, channel_id: str) -> None:
        """更新AI情感状态"""
        emotion_state = self._get_or_create_emotion_state(channel_id)
        current_time = time.time()
        silence_minutes = (current_time - emotion_state.last_interaction_time) / 60.0
        
        # 累积无聊感（基于沉默时长）
        boredom_increase = min(silence_minutes * self.BOREDOM_ACCUMULATION_RATE, 1.0)
        emotion_state.boredom_level = min(1.0, emotion_state.boredom_level + boredom_increase)
        
        # 累积孤独感（非线性增长）
        loneliness_increase = min(silence_minutes * 0.05, 1.0)
        emotion_state.loneliness_accumulation = min(1.0, emotion_state.loneliness_accumulation + loneliness_increase)
        
        # 获取外部情感数据
        await self._integrate_external_emotion_data(channel_id, emotion_state)
        
        # 计算综合主动意愿
        emotion_state.proactive_willingness = self._calculate_proactive_willingness(emotion_state, silence_minutes)
        
        logger.debug(
            f"[情感系统] {channel_id[:8]} 情感更新: "
            f"无聊={emotion_state.boredom_level:.2f} "
            f"孤独={emotion_state.loneliness_accumulation:.2f} "
            f"主动意愿={emotion_state.proactive_willingness:.2f} "
            f"沉默={silence_minutes:.1f}分钟"
        )
    
    async def _integrate_external_emotion_data(self, channel_id: str, emotion_state: EmotionState) -> None:
        """整合外部情感数据（心情、能量等）"""
        try:
            # 获取emotion_tracker的数据
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(channel_id)
            
            if tracker:
                # 获取平均心情
                avg_mood = getattr(tracker, 'average_mood_score', 0.5)
                emotion_state.mood_score = avg_mood
                
                # 获取能量状态
                try:
                    from src.chat.proactive.engagement_energy import get_engagement_energy_manager
                    energy_mgr = get_engagement_energy_manager()
                    if energy_mgr:
                        status = energy_mgr.get_status(channel_id)
                        emotion_state.energy_level = status.get("energy", 50) / 100.0
                except Exception:
                    pass
                
                # 根据心情和能量调整社交欲望
                mood_factor = (avg_mood + 1.0) / 2.0  # 转换为0-1范围
                energy_factor = emotion_state.energy_level
                
                emotion_state.social_desire = (mood_factor * 0.6 + energy_factor * 0.4)
                
        except Exception as e:
            logger.debug(f"[情感系统] 获取外部情感数据失败: {e}")
    
    def _calculate_proactive_willingness(self, emotion_state: EmotionState, silence_minutes: float) -> float:
        """计算综合主动发言意愿"""
        # 基础意愿受无聊感和孤独感驱动
        boredom_factor = emotion_state.boredom_level * 0.4
        loneliness_factor = emotion_state.loneliness_accumulation * 0.3
        social_factor = emotion_state.social_desire * 0.2
        mood_factor = emotion_state.mood_score * 0.1
        
        # 时间因素：越长时间没说话，意愿越强（但有上限）
        time_factor = min(silence_minutes / 60.0, 1.0) * 0.1  # 最多1小时达到最大值
        
        willingness = boredom_factor + loneliness_factor + social_factor + mood_factor + time_factor
        
        # 能量限制
        if emotion_state.energy_level < 0.3:
            willingness *= 0.5
        
        return min(1.0, willingness)
    
    async def _evaluate_proactive_desire(self, channel_id: str) -> Tuple[bool, str]:
        """使用LLM评估AI是否想要主动发言"""
        try:
            emotion_state = self._get_or_create_emotion_state(channel_id)
            
            # 基础条件检查
            if emotion_state.proactive_willingness < self.PROACTIVE_THRESHOLD:
                return False, f"主动意愿不足({emotion_state.proactive_willingness:.2f})"
            
            if emotion_state.energy_level < 0.2:
                return False, f"能量过低({emotion_state.energy_level:.2f})"
            
            # LLM情感决策
            should_speak = await self._llm_emotion_decision(channel_id, emotion_state)
            
            if should_speak:
                reason = (
                    f"情感驱动(无聊={emotion_state.boredom_level:.2f}, "
                    f"孤独={emotion_state.loneliness_accumulation:.2f}, "
                    f"社交欲={emotion_state.social_desire:.2f})"
                )
                return True, reason
            else:
                return False, "LLM决定暂时不想说话"
                
        except Exception as e:
            logger.error(f"[情感系统] 评估主动欲望失败: {e}")
            return False, f"评估失败: {e}"
    
    async def _llm_emotion_decision(self, channel_id: str, emotion_state: EmotionState) -> bool:
        """LLM驱动的情感决策：我现在想说话吗？"""
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config, global_config
            
            request = LLMRequest(model_config.lightweight, request_type="emotion_proactive")
            
            silence_minutes = (time.time() - emotion_state.last_interaction_time) / 60.0
            
            # 读取人格配置（使用正确的路径）
            try:
                character_name = global_config.bot.nickname
                personality = global_config.personality.personality
            except Exception as e:
                logger.error(f"读取人格配置失败: {e}")
                character_name = '爱丽丝'  # 使用正确的配置值作为回退
                personality = '是个有点傲娇又社恐的女孩子，喜欢刷B站看番追动漫，平时爱和朋友聊QQ水群'
            
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "proactive",
                "emotion_proactive.template",
                character_name=character_name,
                personality=personality,
                boredom_level=emotion_state.boredom_level,
                loneliness_accumulation=emotion_state.loneliness_accumulation,
                social_desire=emotion_state.social_desire,
                mood_score=emotion_state.mood_score,
                energy_level=emotion_state.energy_level,
                silence_minutes=silence_minutes
            )

            response, _ = await request.generate_response_async(prompt, max_tokens=10)
            
            if response and "想" in response.strip():
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"[情感系统] LLM情感决策失败: {e}")
            return False
    
    async def _trigger_proactive_speech(self, channel_id: str, reason: str) -> None:
        """触发主动发言"""
        try:
            # 调用现有的主动发言系统
            from src.chat.proactive.think_scheduler import get_think_scheduler, ThinkCategory, ThinkPriority
            scheduler = get_think_scheduler()
            
            if scheduler:
                task_id = await scheduler.schedule_task(
                    category=ThinkCategory.SPONTANEOUS_INIT,
                    user_id="",
                    stream_id=channel_id,
                    priority=ThinkPriority.NORMAL,
                    params={
                        "trigger_reason": "情感驱动主动发言",
                        "emotion_reason": reason,
                        "silence_trigger": False,  # 标记为情感触发，非沉默触发
                    },
                    delay_seconds=0,
                    allow_duplicate=False,
                )
                
                if task_id:
                    logger.info(f"🎯 [情感系统] {channel_id[:8]} 已调度情感驱动主动发言任务: {task_id[:8]}")
                else:
                    logger.warning(f"[情感系统] {channel_id[:8]} 主动发言任务调度失败（可能被去重）")
            
        except Exception as e:
            logger.error(f"[情感系统] 触发主动发言失败: {e}")
    
    async def _stop_emotion_monitoring(self, channel_id: str) -> None:
        """停止指定频道的情感监控"""
        if channel_id in self._monitoring_tasks:
            task = self._monitoring_tasks.pop(channel_id)
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    def _get_or_create_emotion_state(self, channel_id: str) -> EmotionState:
        """获取或创建情感状态"""
        if channel_id not in self._channel_emotions:
            self._channel_emotions[channel_id] = EmotionState()
        return self._channel_emotions[channel_id]
    
    def get_emotion_status(self, channel_id: str) -> Dict:
        """获取情感状态"""
        emotion_state = self._get_or_create_emotion_state(channel_id)
        silence_minutes = (time.time() - emotion_state.last_interaction_time) / 60.0
        
        return {
            "boredom_level": emotion_state.boredom_level,
            "loneliness_accumulation": emotion_state.loneliness_accumulation,
            "social_desire": emotion_state.social_desire,
            "mood_score": emotion_state.mood_score,
            "energy_level": emotion_state.energy_level,
            "proactive_willingness": emotion_state.proactive_willingness,
            "silence_minutes": silence_minutes,
            "is_monitoring": channel_id in self._monitoring_tasks,
        }
    
    async def shutdown(self) -> None:
        """关闭系统"""
        self._enabled = False
        for task in list(self._monitoring_tasks.values()):
            if not task.done():
                task.cancel()
        self._monitoring_tasks.clear()
        logger.info("🎭 情感驱动主动发言系统已关闭")


# 全局单例
_emotion_proactive_system: Optional[EmotionDrivenProactiveSystem] = None


def get_emotion_driven_proactive() -> EmotionDrivenProactiveSystem:
    """获取情感驱动主动发言系统"""
    global _emotion_proactive_system
    if _emotion_proactive_system is None:
        _emotion_proactive_system = EmotionDrivenProactiveSystem()
    return _emotion_proactive_system

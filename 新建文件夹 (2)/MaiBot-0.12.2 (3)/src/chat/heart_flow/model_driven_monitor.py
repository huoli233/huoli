import asyncio
import time
import json
import random
import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from src.common.logger import get_logger

from src.plugin_system.apis import message_api

logger = get_logger('心流模式')


@dataclass
class ModelDecision:
    """模型决策结果"""
    action: str  # "观望", "回复", "待机", "深度休眠", "战略沉默"
    duration_seconds: int  # 持续时长(秒)
    thought: str  # 内心独白
    reason: str  # 决策原因
    confidence: float  # 置信度 0-1
    silence_minutes: int = 0  # 主动闭关惩罚时间(分钟)


class ModelDrivenMonitor:
    """
    完全由模型驱动的独立监控系统
    
    特点:
    2. 模型决定窥屏/观望时长 (0-1分钟)
    3. 模型生成内心独白
    4. 超过3分钟自动限制消息量
    """
    
    def __init__(self, chat_id: str, heartfc_instance):
        self.chat_id = chat_id
        self.heartfc = heartfc_instance # 拟人视角日志名称
        self.log_prefix = "[日常摸鱼]"
        
        # 独立的控制标志
        self._monitor_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # 状态机
        self._state = "待机"
        self._state_start_time = 0.0
        self._state_duration = 0.0
        self._state_messages: List = []
        self._current_thought = ""

        self._last_check_time = time.time()

        self._max_duration = 600
        self._max_messages_for_long_duration = 100

        self._mood = 0.5
        self._boredom_level = 0.0
        self._random_seed = random.random()

        self._proactive_desire: float = 0.0
        self._last_proactive_eval_time: float = time.time()
        self._proactive_cooldown_until: float = 0.0
        self._last_conversation_summary: str = ""

        self._social_state = "刚加入"
        self._participation_count = 0
        self._last_participation_time = 0.0
        self._consecutive_silence = 0

        from src.chat.heart_flow.skills.scoring_system import get_scoring_system
        self._scoring_system = get_scoring_system()

        self._replied_message_ids: set = set()
        self._processed_monitor_ids: set = set()
        self._last_reply_time: float = 0.0
        self._min_reply_interval: float = 10.0

        self._recent_message_contents: List[str] = []
        self._max_recent_messages: int = 10
        
        # 深度频率控制与强制闭关
        self._silence_until: float = 0.0
        self._dynamic_cooldown_until: float = 0.0 # 动态冷却，模拟看一眼不想理

        # 新增/恢复动态脑力系统
        self._stamina_max: float = 100.0
        self._stamina: float = 100.0
        self._last_stamina_recover_time: float = time.time()
        
        # 从数据库加载持久化的脑力数据 (如果存在)
        self._load_stamina_state()
        
        # 事件驱动：用于立即唤醒监控循环
        self._new_message_event = asyncio.Event()

        # 消息聚合窗口（刷屏场景下收集完整批次再统一处理）
        self._aggregate_initial_wait: float = 0.5
        self._aggregate_settle_timeout: float = 1.5
        self._aggregate_max_wait: float = 5.0

        # 话题冷却：记录最近参与话题的时间，避免反复介入同一话题
        self._last_replied_topic_kw: str = ""
        self._last_replied_time: float = 0.0

        # 群活跃度：记录最后一条新消息到达的时间
        self._last_new_message_time: float = time.time()

        # 频率硬限制：两次模型决策之间最少间隔秒数（被@时豁免）
        self._last_decision_time: float = 0.0
        self._min_decision_interval: float = 1.0  # 极大缩短检测冷却，让模型反应更灵敏

        # 心跳日志间隔控制：日常摸鱼模式下减少日志输出频率
        self._last_heartbeat_log_time: float = 0.0
        self._heartbeat_log_interval: float = 300.0  # 每5分钟(300秒)记录一次心跳状态
        self._heartbeat_count_since_last_log: int = 0  # 记录上次日志后的心跳次数

        # 接入决策大脑与参与度评分系统 (Legacy Restoration)
        from src.modules.brain.decision_brain import get_decision_brain
        from src.chat.chatter.interest_calculator import get_engagement_scorer
        self._decision_brain = get_decision_brain()
        self._engagement_scorer = get_engagement_scorer()

        # 避免对同一条消息重复生成内心独白
        self._last_thought_msg_id: str = ""  # 最后生成过内心独白的消息ID

        # 状态管理器
        self._state_manager = None
        try:
            from src.chat.heart_flow.state_manager import get_heart_state_manager
            self._state_manager = get_heart_state_manager(self.chat_id, self.heartfc)
        except Exception as e:
            logger.debug(f"{self.log_prefix} 状态管理器初始化失败: {e}")

        # 状态检查间隔
        self._state_check_interval = 10.0
        self._last_state_check_time = 0.0

        logger.info(f"{self.log_prefix} 初始化完成")
    
    @property
    def is_running(self) -> bool:
        """监控是否正在运行"""
        return self._monitor_active
    
    @property
    def current_state(self) -> str:
        """当前状态"""
        return self._state
    
    @property
    def current_thought(self) -> str:
        """当前内心独白"""
        return self._current_thought

    def _get_psych(self, user_id: str) -> dict:
        if not hasattr(self.heartfc, '_get_psychological_context'):
            return {"favor": 0, "annoyance": 0.0, "mental_fatigue": 0.0, "trauma": 0.0, "relationship": "陌生人", "trust": 0}
        raw = self.heartfc._get_psychological_context(user_id)
        return self._normalize_psych_context(raw)

    def _get_stream_level_psych(self) -> dict:
        stream_emotion = None
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker
            tracker = get_emotion_tracker(self.chat_id)
            stream_emotion = tracker.get_stream_emotion_state()
        except Exception:
            pass
        fatigue = 0.0
        annoyance = 0.0
        if stream_emotion:
            fatigue = getattr(stream_emotion, 'accumulated_fatigue', 0.0)
            annoyance = getattr(stream_emotion, 'annoyance', 0.0)
        return {
            "favor": 0,
            "annoyance": annoyance,
            "mental_fatigue": fatigue,
            "trauma": 0.0,
            "relationship": "群友",
            "trust": 0,
        }

    def _normalize_psych_context(self, raw: dict) -> dict:
        """将 heartFC_chat 提供的原始心理字段映射为内部统一字段名"""
        return {
            "favor": raw.get('affection', raw.get('favor', 0)),
            "annoyance": raw.get('annoyance', 0.0),
            "mental_fatigue": raw.get('fatigue', raw.get('mental_fatigue', 0.0)),
            "trauma": raw.get('trauma_score', raw.get('trauma', 0.0)),
            "relationship": raw.get('relationship', '陌生人'),
            "trust": raw.get('trust', 0),
        }

    def _try_consume_thinking_energy(self, base_cost: float = 2.0) -> bool:
        """能量闸门：检查脑力是否足够，足够则扣除并返回True，否则返回False"""
        self._update_stamina()
        if self._stamina < 10.0:
            logger.debug(f"{self.log_prefix} 脑力不足({self._stamina:.0f})，无法启动思考")
            return False
        
        # 改为模拟随机扣除 1% ~ 5% 的脑力，最大扣除10.0
        cost = min(random.uniform(1.0, 5.0), 10.0)
        self._stamina = max(0.0, self._stamina - cost)
        self._save_stamina_state()
        return True

    async def start(self):
        """启动独立监控"""
        if self._monitor_active:
            logger.warning(f"{self.log_prefix} 监控已在运行")
            return
        
        self._monitor_active = True
        self._last_check_time = time.time()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info(f"{self.log_prefix} 启动")
    
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
                
        logger.info(f"{self.log_prefix} 停止")
    
    async def _evaluate_proactive_desire(self):
        """让模型自主决定是否主动发起话题（完全模型驱动）
        
        动态评估机制：
        - 模型根据沉默时长、群活跃度、自己的状态自主决定
        - 系统只负责调用频率控制，不做任何决策判断
        - 沉默越久，评估间隔越长，但不会完全停止
        """
        now = time.time()
        if getattr(self, 'is_private_chat', False):
            return  # 私聊不需要主动发起
        
        # 基础冷却：避免频繁调用LLM
        if now < self._proactive_cooldown_until:
            return
        
        silence_duration = now - self._last_new_message_time
        silence_minutes = silence_duration / 60.0
        
        # 如果群里刚有人说话，重置评估状态（放宽至 30 秒，容忍短暂停顿）
        if silence_duration < 30:  # 30秒内有消息则不主动
            self._last_proactive_eval_time = now
            return
        
        # 动态评估间隔：根据沉默时长调整检查频率
        # 但永远不会完全停止，只是越来越慢
        elapsed = now - self._last_proactive_eval_time
        if silence_minutes < 10:
            min_interval = 60  # 1分钟检查一次（此前为3分钟）
        elif silence_minutes < 30:
            min_interval = 300  # 5分钟检查一次（此前为10分钟）
        elif silence_minutes < 60:
            min_interval = 1200  # 20分钟检查一次
        else:
            min_interval = 3600  # 1小时检查一次
        
        if elapsed < min_interval:
            return
        
        self._last_proactive_eval_time = now
        
        # 完全由模型决定是否主动，系统不做任何判断
        should_speak = await self._ask_model_about_proactive_speak(silence_minutes)
        
        if should_speak:
            logger.info(f"{self.log_prefix} [内心] 沉默{int(silence_minutes)}分钟，模型决定主动找话题")
            # 冷却时间也由模型的决策隐含决定（通过下次评估间隔）
            self._proactive_cooldown_until = now + min_interval
            await self._trigger_proactive_speak(silence_minutes)
        else:
            logger.debug(f"{self.log_prefix} [内心] 沉默{int(silence_minutes)}分钟，模型决定继续观望")
    
    async def _ask_model_about_proactive_speak(self, silence_minutes: float) -> bool:
        """询问模型是否想主动说话（完全模型驱动）"""
        try:
            from src.config.config import global_config
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            
            # 获取当前状态
            stamina = self._stamina
            mood = self._mood
            
            # 获取最近话题和活跃度
            recent_topics = ""
            last_active_user = ""
            last_active_time = ""
            message_count_last_hour = 0
            message_count_last_day = 0
            try:
                recent_msgs = await self._fetch_recent_messages(limit=50)
                if recent_msgs:
                    now = time.time()
                    # 统计消息数
                    for m in recent_msgs:
                        msg_time = getattr(m, 'time', getattr(m, 'timestamp', now))
                        if now - msg_time < 3600:
                            message_count_last_hour += 1
                        if now - msg_time < 86400:
                            message_count_last_day += 1
                    
                    # 获取最后发言的人和时间
                    if recent_msgs:
                        last_msg = recent_msgs[-1]
                        last_active_user = self._get_msg_sender_name(last_msg)
                        last_msg_time = getattr(last_msg, 'time', getattr(last_msg, 'timestamp', now))
                        last_silence = (now - last_msg_time) / 60.0
                        if last_silence < 60:
                            last_active_time = f"{int(last_silence)}分钟前"
                        else:
                            last_active_time = f"{int(last_silence/60)}小时前"
                    
                    # 获取最近话题
                    topics = []
                    for m in recent_msgs[-5:]:
                        txt = self._get_msg_text(m)
                        if txt and len(txt) > 5:
                            topics.append(txt[:30])
                    if topics:
                        recent_topics = "最近聊的：" + "；".join(topics)
            except Exception:
                pass
            
            bot_name = global_config.bot.nickname
            personality = global_config.personality.personality
            
            # 获取时间信息
            from datetime import datetime
            current_time = datetime.now()
            time_str = current_time.strftime("%H:%M")
            hour = current_time.hour
            
            time_context = ""
            if 0 <= hour <= 6:
                time_context = "（现在是大半夜，大家可能都在睡觉）"
            elif 7 <= hour <= 9:
                time_context = "（现在是早上，大家可能在通勤或刚醒）"
            elif 23 <= hour <= 24:
                time_context = "（现在是深夜，大家准备睡了）"
            
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.SYSTEM,
                "model_driven_monitor",
                "proactive_speak.template",
                bot_name=bot_name,
                personality=personality,
                time_str=time_str,
                time_context=time_context,
                silence_minutes=int(silence_minutes),
                last_active_user=last_active_user if last_active_user else '不记得了',
                last_active_time=last_active_time if last_active_time else '很久前',
                message_count_last_hour=message_count_last_hour,
                message_count_last_day=message_count_last_day,
                recent_topics=recent_topics if recent_topics else '- 之前没什么特别的话题',
                stamina=f"{stamina:.0f}",
                mood_text='不错' if mood > 0.6 else '一般' if mood > 0.3 else '有点低落'
            )
            
            model_set = model_config.focus_chat
            request = LLMRequest(model_set, request_type="proactive_decision")
            response, _ = await request.generate_response_async(prompt)
            
            if not response:
                return False
            
            import json
            import re
            # 提取JSON
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                want_speak = data.get("want_to_speak", False)
                reason = data.get("reason", "")
                
                if want_speak:
                    logger.info(f"{self.log_prefix} [主动意愿] {reason}")
                else:
                    logger.debug(f"{self.log_prefix} [主动意愿] 不想说话：{reason}")
                
                return want_speak
            
            return False
            
        except Exception as e:
            logger.debug(f"{self.log_prefix} 主动意愿评估失败: {e}")
            return False

    async def _trigger_proactive_speak(self, silence_minutes: float):
        import types
        recent_topics = ""
        try:
            recent_msgs = await self._fetch_recent_messages(limit=15)
            if recent_msgs:
                topics = []
                for m in recent_msgs[-8:]:
                    txt = self._get_msg_text(m)
                    if txt and len(txt) > 3:
                        topics.append(txt[:60])
                if topics:
                    recent_topics = "\n".join(f"- {t}" for t in topics[-5:])
        except Exception:
            recent_topics = ""
        context_hint = ""
        if recent_topics:
            context_hint = f"之前群里在聊的内容：\n{recent_topics}\n"
        fake_msg = types.SimpleNamespace()
        fake_msg.raw_message = (
            f"{context_hint}"
            f"群里已经安静了大约{int(silence_minutes)}分钟。"
            f"你可以根据之前的话题接着聊，或者自然地说点什么。"
            f"如果没什么想说的，也可以继续潜水。"
        )
        fake_msg.processed_plain_text = fake_msg.raw_message
        fake_msg.message_id = f"proactive_{int(time.time())}"
        await self._process_messages_with_model([fake_msg], time.time())

    async def _monitor_loop(self):
        """
        独立监控循环 - 完全由模型驱动(永久运行, 自动重启)
        """
        logger.info(f"{self.log_prefix} 循环开始")
        
        # 永久运行, 崩溃自动重启
        while self._monitor_active:
            try:
                await self._monitor_loop_inner()
            except asyncio.CancelledError:
                logger.info(f"{self.log_prefix} 循环取消")
                raise
            except Exception as e:
                logger.error(f"{self.log_prefix} 循环崩溃,3秒后重启: {e}", exc_info=True)
                await asyncio.sleep(3)
                if self._monitor_active:
                    logger.info(f"{self.log_prefix} 循环重启")

        logger.info(f"{self.log_prefix} 循环停止")
    
    async def _monitor_loop_inner(self):
        """内部监控循环"""
        logger.debug(f"{self.log_prefix} 拿起了手机，开始检查有什么新消息...")

        while self._monitor_active:
            now = time.time()

            # 1. 脑力恢复逻辑
            self._update_stamina()

            # 2. 定期心跳日志 (仅在调试时显示)
            if now - self._last_heartbeat_log_time >= self._heartbeat_log_interval:
                state_info = "未知"
                if self._state_manager:
                    state_info = self._state_manager.current_state.value
                logger.debug(f"{self.log_prefix} [静默运行中] 脑力={self._stamina:.0f} 状态={state_info}")
                self._last_heartbeat_log_time = now

            # 3. 检查状态是否到期，需要重新评估
            if self._state_manager and now - self._last_state_check_time >= self._state_check_interval:
                self._last_state_check_time = now
                expired = self._state_manager.check_state_expiration()
                if expired:
                    logger.info(f"{self.log_prefix} 状态到期({expired['state']} {expired['duration']}秒)，准备重新评估...")
                    await self._reevaluate_state_after_expiration()

            # 4. 脑力不足时触发休息
            if self._state_manager and self._state_manager.check_rest_trigger():
                continue

            # 5. 处理深度闭关/丢手机状态 (Hibernation)
            if now < self._silence_until:
                remaining = int(self._silence_until - now)
                try:
                    await asyncio.wait_for(self._new_message_event.wait(), timeout=min(5.0, remaining))
                    self._new_message_event.clear()
                except asyncio.TimeoutError:
                    pass
                continue

            # 5. 监听新消息事件
            try:
                await asyncio.wait_for(self._new_message_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

            # 如果是被消息触发唤醒的，启用消息聚合窗口防止刷屏时回复旧消息
            is_woken_by_message = self._new_message_event.is_set()
            if is_woken_by_message:
                self._new_message_event.clear()
                await asyncio.sleep(self._aggregate_initial_wait)
                # 初始等待后检测是否还有新消息到达（刷屏判定）
                if self._new_message_event.is_set():
                    agg_start = time.time()
                    remaining_budget = self._aggregate_max_wait - self._aggregate_initial_wait
                    wave_count = 1
                    while time.time() - agg_start < remaining_budget:
                        self._new_message_event.clear()
                        try:
                            await asyncio.wait_for(
                                self._new_message_event.wait(),
                                timeout=self._aggregate_settle_timeout,
                            )
                            wave_count += 1
                        except asyncio.TimeoutError:
                            break
                    self._new_message_event.clear()
                    total_wait = time.time() - agg_start + self._aggregate_initial_wait
                    logger.info(
                        f"{self.log_prefix} [消息聚合] 检测到连续{wave_count}波刷屏，"
                        f"等待{total_wait:.1f}秒后统一处理"
                    )

            # 6. 拉取消息
            new_messages = await self._fetch_new_messages()

            # 7. 处理新消息
            if new_messages:
                is_at_me = any(getattr(m, 'is_mentioned', False) or getattr(m, 'is_at', False) for m in new_messages)
                is_name_mention = False
                if not is_at_me:
                    try:
                        from src.config.config import global_config
                        bot_name = global_config.bot.nickname
                        if bot_name:
                            combined = " ".join(self._get_msg_text(m) for m in new_messages)
                            if bot_name in combined:
                                is_name_mention = True
                    except Exception:
                        pass
                if self._state_manager:
                    state_status = self._state_manager.on_message_received(is_at_me=is_at_me or is_name_mention)
                    if state_status.get("interrupted"):
                        self._dynamic_cooldown_until = 0
                        logger.debug(f"{self.log_prefix} 状态被打断，清除冷却时间")
                    if not state_status["can_process"]:
                        logger.debug(f"{self.log_prefix} 当前状态({state_status['current_state']})不处理消息")
                        for m in new_messages:
                            msg_id = self._get_msg_id(m)
                            self._processed_monitor_ids.discard(msg_id)
                        continue

                if now < self._dynamic_cooldown_until:
                    if not is_at_me:
                        for m in new_messages:
                            msg_id = self._get_msg_id(m)
                            self._processed_monitor_ids.discard(msg_id)
                        continue

                # 正常进入模型处理逻辑
                await self._process_messages_with_model(new_messages, time.time())

            elif self._state == "待机":
                if now >= self._dynamic_cooldown_until:
                    if not await self._check_rest_state(now):
                        await self._idle_check_and_decide()

    async def _check_rest_state(self, now: float) -> bool:
        """检查休息状态"""
        try:
            is_resting = getattr(self.heartfc, '_is_resting', False)
            if not is_resting:
                return False

            rest_start = getattr(self.heartfc, '_rest_start', 0.0)
            rest_duration = getattr(self.heartfc, '_rest_duration', 0.0)
            elapsed = now - rest_start
            remaining = max(0, int(rest_duration - elapsed))

            if elapsed >= rest_duration:
                # 休息结束
                self.heartfc._is_resting = False
                logger.info(f"{self.log_prefix} 休息结束")
                return False

            # 还在休息中
            if remaining % 30 == 0 or remaining == int(rest_duration):
                logger.debug(f"{self.log_prefix} 休息中 {remaining}秒 | 内部脑力={self._stamina:.0f}/{self._stamina_max:.0f}")

            return True

        except Exception as e:
            logger.debug(f"{self.log_prefix} 休息检查失败: {e}")
            return False

    async def _initial_check(self):
        """启动时的初始检测"""
        try:
            logger.info(f"{self.log_prefix} 执行初始检测...")

            # 获取最近的消息
            recent_msgs = await self._fetch_recent_messages(limit=10)
            if recent_msgs:
                # 立即将初始消息ID注册到已处理集合，防止主循环重复拉取
                for m in recent_msgs:
                    self._processed_monitor_ids.add(self._get_msg_id(m))
                last_msg = recent_msgs[-1]
                msg_text = self._get_msg_text(last_msg)
                sender_name = self._get_msg_sender_name(last_msg)

                logger.info(f"{self.log_prefix} 初始检测发现消息: {sender_name}: {msg_text[:30]}...")

                # 生成内心所想
                user_id = self._get_msg_user_id(last_msg)
                psych = self._get_psych(user_id)
                interest = await self._analyze_interest_with_model(msg_text, psych)
                await self._generate_instant_thought(msg_text, psych, interest)

                # 直接进行决策
                await self._model_decide_next_action(recent_msgs, time.time())
            else:
                logger.info(f"{self.log_prefix} 初始检测: 无历史消息")

        except Exception as e:
            logger.warning(f"{self.log_prefix} 初始检测异常: {e}")

    async def _idle_check_and_decide(self):
        try:
            recent_msgs = await self._fetch_recent_messages(limit=5)
            if not recent_msgs:
                return

            last_msg = recent_msgs[-1]
            msg_text = self._get_msg_text(last_msg)
            if not msg_text:
                return

            msg_id = self._get_msg_id(last_msg)
            if msg_id and msg_id == self._last_thought_msg_id:
                return

            user_id = self._get_msg_user_id(last_msg)
            psych = self._get_psych(user_id)

            interest = await self._analyze_interest_with_model(msg_text, psych)

            await self._generate_instant_thought(msg_text, psych, interest)
            if msg_id:
                self._last_thought_msg_id = msg_id

            if interest['interest_level'] == "感兴趣":
                logger.info(f"{self.log_prefix} 待机时检测到感兴趣消息,进入决策")
                await self._model_decide_next_action(recent_msgs, time.time())
            else:
                if self._current_thought:
                    logger.info(f"{self.log_prefix}  [内心] {self._current_thought}")
                await self._evaluate_proactive_desire()

        except Exception as e:
            logger.debug(f"{self.log_prefix} 待机检查异常: {e}")

    async def _reevaluate_state_after_expiration(self):
        """状态到期后重新评估"""
        try:
            recent_msgs = await self._fetch_recent_messages(limit=5)
            if recent_msgs:
                await self._model_decide_next_action(recent_msgs, time.time(), is_state_reevaluation=True)
            else:
                # 当状态到期且没有任何新消息时，调用主动发话意愿评估
                await self._evaluate_proactive_desire()
                
                # 无论主动说话是否触发，都将“无消息”的语境抛给_model_decide_next_action
                # 借用模型本身的能力生成当前无聊或发呆心境的真实`thought`
                await self._model_decide_next_action([], time.time(), is_state_reevaluation=True)

        except Exception as e:
            logger.debug(f"{self.log_prefix} 状态重新评估异常: {e}")

    async def _generate_instant_thought(self, message_text: str, psych: dict, interest: dict):
        """使用模型理解用户消息，生成真实的内心想法"""
        try:
            user_id = psych.get('user_id', '用户')
            affection = psych.get('favor', 0)
            annoyance = psych.get('annoyance', 0.0)
            mental_fatigue = psych.get('mental_fatigue', 0.0)
            trust = psych.get('trust', 0.0)
            relationship = psych.get('relationship', '陌生人')
            interest_level = interest.get('interest_level', '一般')
            interest_reason = interest.get('reason', '')
            
            # 根据兴趣程度决定引用消息数量
            if interest_level == "非常感兴趣":
                context_limit = 15
            elif interest_level == "感兴趣":
                context_limit = 10
            else:
                context_limit = 5
            
            # 获取历史消息上下文
            history_messages = await self._fetch_recent_messages(limit=context_limit)
            messages_summary = []
            for msg in history_messages[-context_limit:]:
                sender = getattr(msg, 'sender', {}).get('nickname', '用户')
                content = self._get_msg_text(msg)[:80]
                messages_summary.append(f"{sender}: {content}")
            chat_context = "\n".join(messages_summary) if messages_summary else "（刚开始聊天）"
            
            persona_context = await self._fetch_persona_context()
            
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.SYSTEM,
                "model_driven_monitor",
                "instant_thought.template",
                persona_context=persona_context,
                context_limit=context_limit,
                chat_context=chat_context,
                relationship=relationship,
                affection=f"{affection:.0f}",
                trust=f"{trust:.0f}",
                annoyance=f"{annoyance:.0f}",
                mental_fatigue=f"{mental_fatigue:.0f}",
                interest_level=interest_level,
                interest_reason=interest_reason
            )
            response = await self._call_small_model(prompt)
            if response:
                thought = response.strip().replace('"', '').replace("'", "")
                if len(thought) > 30:
                    thought = thought[:28] + "..."
                self._current_thought = thought
                logger.debug(f"{self.log_prefix} [模型内心] {self._current_thought}")
            else:
                self._current_thought = self._generate_fallback_thought(affection, annoyance, mental_fatigue, interest_level)
        except Exception as e:
            logger.debug(f"{self.log_prefix} 模型生成内心想法失败: {e}")
            self._current_thought = self._generate_fallback_thought(affection, annoyance, mental_fatigue, interest.get('interest_level', '一般'))

    def _generate_fallback_thought(self, affection: float, annoyance: float, mental_fatigue: float, interest_level: str) -> str:
        """生成备选的内心想法"""
        if annoyance > 50:
            return "有点烦..."
        elif affection < -20:
            return "不太想理..."
        elif mental_fatigue > 60:
            return "有点累..."
        elif interest_level == "感兴趣":
            return "这个话题有点意思..."
        else:
            return "看看再说..."
    

    
    def _get_msg_text(self, msg) -> str:
        """兼容 DatabaseMessages 和实时平台消息，安全读取消息文本并打上多模态类型标签"""
        # 1. 先尝试获取可能的图片/表情特定状态
        is_pic = getattr(msg, 'is_picid', False)
        is_emoji = getattr(msg, 'is_emoji', False)
        is_cmd = getattr(msg, 'is_command', False)

        text = ""
        # 2. 从已有解析字段里提核心文字
        if getattr(msg, 'processed_plain_text', None):
            text = getattr(msg, 'processed_plain_text')
        elif getattr(msg, 'raw_message', None):
            text = getattr(msg, 'raw_message')
        elif getattr(msg, 'display_message', None):
            text = getattr(msg, 'display_message')

        # 3. 如果内容为空，且被认定是特定类型（可能是图库/黄脸库），则强行补上标记物防丢
        if not text:
            if is_pic:
                text = "[图片]"
            elif is_emoji:
                text = "[表情]"
            elif getattr(msg, 'key_words', None) and 'file' in str(getattr(msg, 'key_words')):
                text = "[文件分享]"
            else:
                text = ""

        # 4. 图片消息简化为标签，不传递完整描述（减少token占用）
        if is_pic:
            # 移除所有图片描述，只保留[图片]标记
            text = re.sub(r"\[图片：[^\]]+\]", "[图片]", text)
            text = re.sub(r"\[picid:[^\]]+\]", "[图片]", text)
            if not text or text == "[图片]":
                text = "[图片]"

        # 5. 如果有字，但确实带了特定类型，前置一个标签帮助大模型理解氛围
        elif text:
            prefix = ""
            if is_pic and "[图片]" not in text:
                prefix += "[图片]"
            if is_emoji and "[表情]" not in text:
                prefix += "[表情]"
            if is_cmd and "[系统命令]" not in text:
                prefix += "[系统命令]"

            if prefix and not text.startswith(prefix):
                text = f"{prefix} {text}"

        return text or ''

    def _get_msg_sender_name(self, msg) -> str:
        """兼容 DatabaseMessages 和实时平台消息，安全读取发送者昵称"""
        # DatabaseMessages 用 user_info.user_nickname
        user_info = getattr(msg, 'user_info', None)
        if user_info:
            name = getattr(user_info, 'user_nickname', None) or getattr(user_info, 'user_cardname', None)
            if name:
                return name
        # 实时平台消息用 sender (可能是 dict 或对象)
        sender = getattr(msg, 'sender', None)
        if isinstance(sender, dict):
            return sender.get('nickname', '') or sender.get('card', '') or '用户'
        if sender:
            return getattr(sender, 'nickname', '') or getattr(sender, 'card', '') or '用户'
        return '用户'

    def _get_msg_user_id(self, msg) -> str:
        """兼容 DatabaseMessages 和实时平台消息，安全读取用户ID"""
        # DatabaseMessages 用 user_info.user_id
        user_info = getattr(msg, 'user_info', None)
        if user_info:
            uid = getattr(user_info, 'user_id', None)
            if uid:
                return str(uid)
        # 实时平台消息用 user_id 或者 sender.user_id
        uid = getattr(msg, 'user_id', None)
        if uid:
            return str(uid)
        sender = getattr(msg, 'sender', None)
        if isinstance(sender, dict):
            return str(sender.get('user_id', ''))
        if sender:
            return str(getattr(sender, 'user_id', ''))
        return ''

    def _get_msg_id(self, msg) -> str:
        """安全地获取或计算消息唯一ID"""
        m_id = getattr(msg, 'message_id', None) or getattr(msg, 'msg_id', None)
        if m_id:
            return str(m_id)
        # 用消息特征做降级防撞 Hash
        text = self._get_msg_text(msg)
        user_id = self._get_msg_user_id(msg)
        timestamp = getattr(msg, 'time', getattr(msg, 'timestamp', 0))
        return f"{timestamp}_{user_id}_{hash(text)}"
        
    async def _fetch_new_messages(self) -> List:
        """从数据库和内存队列双重拉取最新消息，通过去重机制合并返回真正的新消息"""
        try:
            # 1. 从数据库拉取兜底历史防丢帧（最多看最近20条）
            recent_db = await self._fetch_recent_messages(limit=20)
            
            # 2. 从内存消息队列获取最新实时消息（彻底消除数据库存储延迟导致的时而拉取不到）
            recent_mem = []
            if hasattr(self.heartfc, 'get_pending_messages'):
                # get_pending_messages() 返回的是 message_dict 列表
                mem_msgs = self.heartfc.get_pending_messages()
                if mem_msgs:
                    # 将 dict 转成一个可以通过 getattr(msg, 'xxx') 访问的简易命名空间对象以兼容现在的流程
                    class DictToObject:
                        def __init__(self, d):
                            for k, v in d.items():
                                setattr(self, k, v)
                            # 挂载原始 dict 为方便安全读取
                            self._raw_dict = d
                        def get(self, k, default=None):
                            return self._raw_dict.get(k, default)
                    
                    recent_mem = [DictToObject(m) for m in mem_msgs]
                    logger.debug(f"{self.log_prefix} [双路拉取] 从内存极速队列拉取到 {len(recent_mem)} 条暂存")

            combined_pool = list(recent_db) + recent_mem
            if not combined_pool:
                return []
                
            # 时间排序一下，让时序归位
            combined_pool.sort(key=lambda m: getattr(m, 'time', getattr(m, 'timestamp', 0)))

            new_msgs = []
            for msg in combined_pool:
                msg_id = self._get_msg_id(msg)
                if msg_id not in self._processed_monitor_ids:
                    self._processed_monitor_ids.add(msg_id)
                    new_msgs.append(msg)
                    
            # 防止ID集合无限增长
            if len(self._processed_monitor_ids) > 500:
                self._processed_monitor_ids = set(list(self._processed_monitor_ids)[-200:])
                
            return new_msgs
        except Exception as e:
            logger.debug(f"{self.log_prefix} 获取消息失败: {e}")
            return []
    
    def on_new_message(self):
        """外部通知有新消息到达"""
        if self._monitor_active:
            self._last_new_message_time = time.time()
            self._new_message_event.set()

    async def _process_messages_with_model(self, messages: List, now: float):
        """
        使用模型处理新消息

        这是核心: 让模型决定一切
        每条消息都会触发状态重新评估
        """
        last_msg = messages[-1]
        is_at_me = any(getattr(m, 'is_mentioned', False) or getattr(m, 'is_at', False) for m in messages)
        msg_text = self._get_msg_text(last_msg)
        # 合并本次所有新消息的文本进行长度防抖
        combined_text = " ".join(self._get_msg_text(m) for m in messages)
        sender_name = self._get_msg_sender_name(last_msg)
        user_id = self._get_msg_user_id(last_msg)

        # [预过滤] 仅过滤完全空白消息，中文单字（如"在""好""嗯"）含义完整不应过滤
        stripped = combined_text.strip()
        if not stripped and not is_at_me:
            logger.debug(f"{self.log_prefix} [预过滤] 空白消息，忽略")
            return

        # [名字检测] 消息中包含bot名字：不视为@，但跳过窥屏门控让模型自行判断
        is_name_mention = False
        if not is_at_me:
            from src.config.config import global_config
            bot_name = global_config.bot.nickname
            if bot_name and bot_name in combined_text:
                is_name_mention = True
                logger.debug(f"{self.log_prefix} [名字提及] 消息中出现'{bot_name}'，交由模型判断意图")

        # [窥屏概率门控] 窥屏状态下非@消息30%概率进入模型评估，提及名字100%进入
        if not is_at_me and not is_name_mention and self._state_manager and self._state_manager.is_peeking:
            if random.random() > 0.3:
                for msg in messages:
                    msg_id = self._get_msg_id(msg)
                    self._processed_monitor_ids.discard(msg_id)
                return

        # [频率硬限制] 两次决策最少间隔 5 秒，被@时豁免
        time_since_last = now - self._last_decision_time
        if not is_at_me and time_since_last < self._min_decision_interval:
            logger.debug(f"{self.log_prefix} [频率限制] 距上次决策不到 {self._min_decision_interval}s，将其退回暂存...")
            # 关键修复：将被防洪阀跳过的消息退回未处理池，下次循环再战
            for msg in messages:
                msg_id = self._get_msg_id(msg)
                self._processed_monitor_ids.discard(msg_id)
            return

        if is_at_me and self._state == "观望中":
            logger.info(f"{self.log_prefix}  [被@] {sender_name}: {msg_text[:20]}...")

        await self._model_decide_next_action(messages, now)
    
    async def _model_decide_next_action(self, messages: List, now: float, is_state_reevaluation: bool = False):
        """三段式决策：能量闸门 → 模型决策 → 动作分发"""
        is_real_at = False
        is_name_mention = False
        
        if messages:
            last_msg = messages[-1]
            user_id = self._get_msg_user_id(last_msg)
            user_name = self._get_msg_sender_name(last_msg)
            new_msgs_dialogue = ""
            prev_text = None
            repeat_count = 1
            for m in messages:
                s_name = self._get_msg_sender_name(m)
                m_text = self._get_msg_text(m)
                line = f"{s_name}说: {m_text}"
                if line == prev_text:
                    repeat_count += 1
                else:
                    if repeat_count > 1:
                        new_msgs_dialogue += f" [以上重复了{repeat_count}次]\n"
                    new_msgs_dialogue += f"{line}\n"
                    prev_text = line
                    repeat_count = 1
            if repeat_count > 1:
                new_msgs_dialogue += f" [以上重复了{repeat_count}次]\n"
            message_text = new_msgs_dialogue.strip()
            # 【防爆Token保护】群友可能发超长文或代码，小模型只有4096容量，超过直接截断
            if len(message_text) > 800:
                message_text = message_text[:800] + "\n...(因过长被本能忽略)"
            is_real_at = any(getattr(m, 'is_mentioned', False) or getattr(m, 'is_at', False) for m in messages)
            if not is_real_at:
                from src.config.config import global_config
                bot_name = global_config.bot.nickname
                if bot_name and bot_name in message_text:
                    is_name_mention = True
        else:
            last_msg = None
            user_id = ""
            user_name = ""
            message_text = "[群聊当前很安静，没有新消息]"

        is_at_me = is_real_at or is_name_mention
        stream_psych = self._get_stream_level_psych()

        current_state_info = ""
        if self._state_manager:
            status = self._state_manager.get_status()
            current_state_info = f"当前状态: {status['state']}，已持续{status['state_duration']:.0f}秒"

        if not self._try_consume_thinking_energy():
            return

        self._last_decision_time = now
        decision = await self._call_model_for_decision(
            message_text=message_text,
            user_name=user_name,
            user_id=user_id,
            psych=stream_psych,
            is_at_me=is_at_me,
            is_private=getattr(self, 'is_private_chat', False),
            current_state_info=current_state_info,
            is_state_reevaluation=is_state_reevaluation,
        )
        if not decision:
            logger.warning(f"{self.log_prefix} 模型调用失败,回退待机")
            return

        if is_real_at and decision.action != "回复":
            logger.info(f"{self.log_prefix} [强制回复] 被@但模型选择了'{decision.action}'，覆盖为回复")
            decision.action = "回复"
            decision.thought = f"{user_name}@了我，找我什么事？先看看再说"
            decision.reason = f"被{user_name}@了，应该回应看看对方想说什么"

        self._current_thought = decision.thought
        if self._current_thought:
            logger.info(f"{self.log_prefix}  [内心] {self._current_thought}")

        await self._apply_decision_to_state(decision, now, message_text, last_msg)

    async def _apply_decision_to_state(self, decision, now: float, message_text: str, last_msg):
        """应用模型决策到状态管理器"""
        # 确定状态和持续时间
        if decision.action in ["休眠", "放下手机"]:
            state_action = "潜水休眠"
            duration = decision.silence_minutes * 60 if decision.silence_minutes > 0 else 600
            self._silence_until = now + duration
        elif decision.action in ["窥屏", "默默读空气", "休息", "待机"]:
            state_action = "默默读空气"
            duration = decision.duration_seconds if decision.duration_seconds > 0 else 60
        elif decision.action in ["观望", "读空气"]:
            state_action = "读空气中"
            duration = decision.duration_seconds if decision.duration_seconds > 0 else 60
        elif decision.action == "回复":
            state_action = "读空气中"
            duration = 120  # 回复后继续读空气
            self._last_replied_topic_kw = message_text[:10]
            self._last_replied_time = now
            self._dynamic_cooldown_until = now + random.randint(5, 15)
            raw_vibe = decision.thought.split(')')[0] + ')' if ')' in decision.thought else ''
            directive_message = f"你决定回复。原因: {decision.reason}。{raw_vibe}"
            if last_msg:
                await self._notify_reply_needed(last_msg, thought=directive_message)
        else:
            state_action = "读空气中"
            duration = 120

        # 应用到状态管理器
        if self._state_manager:
            self._state_manager.apply_model_decision(
                action=state_action,
                duration=duration,
                reason=decision.reason,
                thought=decision.thought
            )

        logger.info(f"{self.log_prefix} [决策执行] {state_action} {duration}秒 | {decision.reason}")
    
    async def _call_model_for_decision(
        self,
        message_text: str,
        user_name: str,
        user_id: str,
        psych: dict,
        is_at_me: bool = False,
        is_private: bool = False,
        current_state_info: str = "",
        is_state_reevaluation: bool = False,
    ) -> Optional[ModelDecision]:
        try:
            from src.modules.brain.decision_brain import DecisionContext
            ctx = DecisionContext(
                stream_id=self.chat_id,
                user_id=user_id,
                message_content=message_text,
                current_mood="happy" if self._mood > 0.7 else "angry" if self._mood < 0.3 else "neutral",
                social_context={"is_targeted": is_at_me, "is_private": is_private}
            )
            factors = await self._decision_brain._calculate_factors(ctx)
            engagement = await self._engagement_scorer.calculate(
                message_id=f"dec_{int(time.time())}",
                content=message_text,
                user_id=user_id,
                is_mentioned=is_at_me,
                is_private=is_private,
                extra_data={"rel_score": 0.5}
            )
            factors_text = (
                f"【环境研判数据 (加权因子)】:\n"
                f"- 置信度: {factors.confidence_score:.2f} | 风险值: {factors.risk_score:.2f}\n"
                f"- 紧迫感: {factors.urgency_score:.2f} | 社交热度: {factors.social_score:.2f}\n"
                f"- 心情指数: {factors.mood_score:.2f} | 参与意愿: {engagement.engagement_score:.2f}\n"
                f"风险值过高时建议战略观望；参与意愿极低时考虑放下手机。"
            )

            reeval_hint = ""
            if is_state_reevaluation:
                reeval_hint = "【注意】你的状态刚刚到期，需要重新决定接下来要做什么。"

            prompt = await self._build_decision_prompt(
                message_text=message_text,
                user_name=user_name,
                user_id=user_id,
                psych=psych,
                current_state_info=current_state_info,
                reeval_hint=reeval_hint,
                is_at_me=is_at_me,
                is_private=is_private,
                factors_text=factors_text,
            )
            response = await self._call_small_model(prompt)
            if not response:
                return None
            decision = self._parse_model_decision(response)
            can_act = await self._check_resource_constraints(decision.action, psych)
            if not can_act:
                decision.action = "放下手机"
                decision.silence_minutes = random.randint(5, 15)
                decision.reason = "太累了，即便想回也回不动，先匿了"
            if decision.action == "回复" and engagement.engagement_score < 0.1 and not is_at_me:
                if random.random() > 0.5:
                    decision.action = "观望"
                    decision.reason = "最后还是觉得没啥好回的，再看看吧"
            self._update_social_state(decision.action)
            logger.debug(
                f"{self.log_prefix}  [决策详情] "
                f"行动={decision.action} | "
                f"想法={decision.thought} | "
                f"原因={decision.reason}"
            )
            return decision
        except Exception as e:
            logger.error(f"{self.log_prefix} 模型决策失败: {e}", exc_info=True)
            return None

    async def _check_resource_constraints(self, action: str, psych: dict = None) -> bool:
        """检查心理与情绪压力，结合脑力值判定"""
        try:
            # 恢复时间
            self._update_stamina()

            # 获取心理疲劳相关状态
            mental_fatigue = psych.get('mental_fatigue', 0.0) if psych else 0.0
            trauma_score = psych.get('trauma', 0.0) if psych else 0.0
            annoyance = psych.get('annoyance', 0.0) if psych else 0.0

            # 计算综合压力，加入脑力亏空惩罚
            stamina_deficit = 100 - self._stamina
            pressure = (
                mental_fatigue * 0.4 +
                trauma_score * 10 * 0.3 +
                annoyance * 0.2 +
                (stamina_deficit * 0.5)
            )

            # 根据剧烈压力基础判断（防崩溃底线）
            if pressure > 100.0 or self._stamina < 5.0:
                logger.debug(f"{self.log_prefix} 精力不足({pressure:.1f})，需要休息")
                return action == "待机" or action == "休眠" or action == "放下手机"

            return True

        except Exception as e:
            logger.debug(f"{self.log_prefix} 状态检查失败: {e}")
            return True  # 失败时允许行动

    async def _build_decision_prompt(
        self,
        message_text: str,
        user_name: str,
        user_id: str,
        psych: dict,
        is_at_me: bool = False,
        is_private: bool = False,
        factors_text: str = "",
        current_state_info: str = "",
        reeval_hint: str = "",
    ) -> str:
        # 动态变长上下文获取：默认最多捞 20 条近期记录
        recent_msgs = await self._fetch_recent_messages(limit=20)
        now_ts = time.time()
        
        # 变焦检索截断逻辑 (Dynamic Context Window)
        last_at_idx = -1
        for i, m in enumerate(recent_msgs):
            if getattr(m, 'is_mentioned', False) or getattr(m, 'is_at', False):
                last_at_idx = i
                
        # 1. 如果近期闲聊很少，没到20条，自然就短
        # 2. 如果超过10条，看最近聊得多密
        keep_count = 10  # 基础关注度：10句
        if len(recent_msgs) >= 15:
            # 如果这批消息的第一条距离现在不到3分钟，说明群里爆发了激烈刷屏
            first_msg_time = getattr(recent_msgs[0], 'time', getattr(recent_msgs[0], 'timestamp', now_ts))
            if (now_ts - first_msg_time) < 180:
                keep_count = 20  # 集中关注度，全量吸收吃瓜
                
        # 如果存在专门点名@我们的，确保焦点不被裁剪
        start_idx = max(0, len(recent_msgs) - keep_count)
        if last_at_idx != -1 and last_at_idx < start_idx:
            start_idx = max(0, last_at_idx - 2) # 带上被点名时的前语境两三句
            
        recent_msgs = recent_msgs[start_idx:]
        
        # 2. 个体化群体印象 (分别心)
        from src.config.config import global_config
        bot_qq = str(global_config.bot.qq_account)
        
        unique_users = {}
        for msg in recent_msgs:
            uid = self._get_msg_user_id(msg)
            if uid and uid != bot_qq and uid not in unique_users:
                u_name = self._get_msg_sender_name(msg)
                u_psych = self._get_psych(uid)
                u_favor = u_psych.get('favor', 0)
                u_rel = u_psych.get('relationship', '熟人')
                unique_users[uid] = f"- {u_name}: {u_rel}(好感{u_favor:.0f})"
        group_rel_overview = "\n".join(unique_users.values()) if unique_users else "暂无详细印象"

        recent_str = ""
        for i, msg in enumerate(recent_msgs, 1):
            s_uid = self._get_msg_user_id(msg)
            s_name = "【你】" if s_uid == bot_qq else self._get_msg_sender_name(msg)
            content = self._get_msg_text(msg)[:100] # 放宽截断以容纳图片与引用回复描述
            
            # 【精确时差感知】计算这条消息是多久前发的
            msg_time = getattr(msg, 'time', getattr(msg, 'timestamp', now_ts))
            diff_sec = max(0, int(now_ts - msg_time))
            if diff_sec <= 10:
                time_label = "刚好"
            elif diff_sec < 60:
                time_label = f"{diff_sec}秒前"
            else:
                time_label = f"{diff_sec // 60}分钟前"
            
            # 【关键】如果这句话是专门@你或者点了你的名字，必须要让大模型看清楚！
            directed_at_me = getattr(msg, 'is_mentioned', False) or getattr(msg, 'is_at', False)
            at_prefix = "【专门对你说】" if directed_at_me else ""
            
            recent_str += f"{i}. [{time_label}] {at_prefix}{s_name}: {content}\n"

        # 【针对骚扰刷屏的整体长度防爆截断】即便单条只取 100，20 条表情包描述累加依然可达 3000 多 token。
        if len(group_rel_overview) > 500:
            group_rel_overview = group_rel_overview[:500] + "\n...(人数过多省略)"
        if len(recent_str) > 1500:
            recent_str = "...(过早的对话因太长被大脑抛弃)\n" + recent_str[-1500:]

        relationship = psych.get('relationship', '陌生人')
        favor = psych.get('favor', 0)
        trust = psych.get('trust', 0)
        trauma = psych.get('trauma', 0)

        # 引入真实客观现实时间
        from datetime import datetime
        current_time = datetime.now()
        time_str = current_time.strftime("%H:%M")
        
        # 将机器人的活跃状态转化为提示
        chat_value = getattr(self.heartfc, '_chat_value', 100.0)
        consecutive = getattr(self.heartfc, '_consecutive_reply_count', 0)
        activity_desc = f"你的聊天欲{chat_value:.0f}/100"
        if consecutive > 0:
            activity_desc += f"，刚连着回了{consecutive}条"
        activity_desc += "。"
        if chat_value < 30:
            activity_desc += "你有点不想聊了。"
        if now_ts < self._silence_until:
            remain = int((self._silence_until - now_ts) / 60)
            activity_desc += f" 你之前决定休息一会，还有{remain}分钟才想回来。除非有人叫你，不然就继续歇着。"
        elif now_ts < self._dynamic_cooldown_until:
            remain_sec = int(self._dynamic_cooldown_until - now_ts)
            activity_desc += f" 你刚才觉得没意思划走了，{remain_sec}秒后再看看。"
        if 23 <= current_time.hour or current_time.hour < 6:
            activity_desc += " 现在是深夜了，你有点困。"

        # 获取心理疲劳状态
        mental_fatigue = psych.get('mental_fatigue', 0.0) if psych else 0.0
        trauma_score = psych.get('trauma', 0.0) if psych else 0.0
        annoyance = psych.get('annoyance', 0.0) if psych else 0.0

        at_context = "有人@你,看了一眼" if is_at_me else ""

        memory_context = await self._fetch_memory_context(user_id, message_text)
        persona_context = await self._fetch_persona_context()
        group_summary = await self._fetch_group_summary()

        # 构建关系描述
        if favor < -30:
            rel_feeling = f"你非常讨厌{user_name}（好感{favor:.0f}），很不想搭理"
        elif favor < -10:
            rel_feeling = f"你对{user_name}印象不好（好感{favor:.0f}），有些排斥"
        elif favor < 10:
            rel_feeling = f"你和{user_name}不太熟（好感{favor:.0f}）"
        elif favor < 40:
            rel_feeling = f"你觉得{user_name}还不错（好感{favor:.0f}），"
        else:
            rel_feeling = f"你很喜欢{user_name}（好感{favor:.0f}），想多聊"

        # 构建精神防线状态描述
        state_parts = []
        if mental_fatigue > 60:
            state_parts.append("心理疲劳")
        if annoyance > 50:
            state_parts.append("很烦躁")
        elif annoyance > 30:
            state_parts.append("有点烦")
        if trauma_score > 5:
            state_parts.append("心里有创伤很敏感")
        state_line = "，".join(state_parts) if state_parts else "精神状态平稳"

        # 群活跃度感知：消息积压数量 + 沉默时长
        msg_count = len(self._state_messages)
        if msg_count >= 15:
            group_activity = "非常活跃（消息积压超过15条）"
        elif msg_count >= 6:
            group_activity = "活跃"
        elif msg_count >= 2:
            group_activity = "一般"
        else:
            group_activity = "冷清"
        silence_secs = now_ts - self._last_new_message_time
        
        # 判断是否是长期沉默后的首条消息
        is_breaking_silence = silence_secs > 600  # 10分钟以上算长期沉默
        
        if silence_secs < 30:
            silence_desc = "刚刚有人在说话"
        elif silence_secs < 120:
            silence_desc = f"已有约 {int(silence_secs)} 秒没人说话"
        elif silence_secs < 600:
            silence_desc = f"已有 {int(silence_secs // 60)} 分钟没人说话，群聊冷清下来了"
        else:
            silence_desc = f"群里已经沉默超过 {int(silence_secs // 60)} 分钟"
            if is_breaking_silence:
                silence_desc += "，终于有人打破沉默了"
        
        group_activity_desc = f"【群聊活跃度】: {group_activity}，{silence_desc}。"
        
        # 长期沉默后的话题理解提示
        topic_understanding_hint = ""
        if is_breaking_silence:
            topic_understanding_hint = """
【长期沉默后的决策】
群里沉默很久了，现在有人说话。在决定是否回复前：
1. 先看看对方在聊什么
2. 判断这个话题你能不能接上话
3. 如果不太懂或没兴趣，可以继续观望
4. 如果能理解且有话说，可以尝试回复
5. 不要勉强自己接不懂的话题

决策建议：
- 能理解 + 有话说 → 可以回复
- 不太懂 → 继续观望
- 完全不懂 → 放下手机
"""

        # 话题冷却：120秒内刚参与过话题则提醒模型不要反复介入
        topic_cooldown_desc = ""
        if self._last_replied_time > 0:
            since_last = now_ts - self._last_replied_time
            if since_last < 120 and self._last_replied_topic_kw:
                topic_cooldown_desc = f"【话题冷却】你 {int(since_last)} 秒前刚参与了「{self._last_replied_topic_kw}...」相关话题，如无必要避免立刻再介入，否则显得话痨。"

        # 生成动态的隐性心境偏移因子（结合时间和小随机种子）
        seed = int(now_ts // 3600)  # 每小时变一次心情底色
        random.seed(seed)
        moods = [
            "有点社恐，不太想在人多时说话，更想默默吃瓜",
            "心情有点亢奋，想找存在感，遇到复读机也想跟着复读",
            "比较佛系，不想卷入麻烦的争论，就看看",
            "有点孤单，希望有人跟我搭话或者能遇到好玩的话题",
            "处于吃瓜乐子人状态，看热闹不嫌事大",
            "状态放松，愿意做个暖心或者接话茬的人"
        ]
        current_mood = random.choice(moods)
        # 恢复真实随机数生成器状态以防影响后续机制
        random.seed() 

        if is_private:
            scene_hint = "这是私聊，就你们两个人。别人看不见你们的对话，没必要端着。有话就说，对方找你聊天你就回，冷场了也可以自己找话题。"
        else:
            scene_hint = "这是群聊，你就是群里的一员。想说就说，不想说就刷刷消息看看热闹。看到有意思的可以接话，看到无聊的可以划走。有人复读你也可以跟着玩，觉得没劲了就放下手机干别的去。"

        # 当前状态信息
        state_context = f"\n{current_state_info}\n" if current_state_info else ""
        reeval_context = f"\n{reeval_hint}\n" if reeval_hint else ""

        from src.config.prompt_loader import get_prompt, PromptCategory
        prompt = get_prompt(
            PromptCategory.SYSTEM,
            "model_driven_monitor",
            "decision_prompt.template",
            persona_context=persona_context,
            state_context=state_context,
            reeval_context=reeval_context,
            memory_context=memory_context,
            group_rel_overview=group_rel_overview,
            recent_str=recent_str,
            message_text=message_text,
            time_str=time_str,
            activity_desc=activity_desc,
            group_activity_desc=group_activity_desc,
            topic_cooldown_desc=topic_cooldown_desc,
            topic_understanding_hint=topic_understanding_hint,
            state_line=state_line,
            at_context=at_context,
            rel_feeling=rel_feeling,
            current_mood=current_mood,
            factors_text=factors_text,
            scene_hint=scene_hint
        )

        return prompt

    async def _fetch_memory_context(self, user_id: str, message_text: str) -> str:
        try:
            from src.memory_system.memory_core import get_memory_manager
            mgr = get_memory_manager()
            memories = mgr.search_memories(
                stream_id=self.chat_id,
                query=message_text,
                limit=3,
                user_id=user_id,
            )
            if memories:
                parts = []
                for m in memories:
                    content = getattr(m, 'content', '')
                    if content:
                        parts.append(content[:30])
                if parts:
                    return "; ".join(parts)
            return "无相关记忆"
        except Exception as e:
            logger.debug(f"{self.log_prefix} 获取记忆失败: {e}")
            return "无相关记忆"

    async def _fetch_persona_context(self) -> str:
        from src.config.config import global_config
        bot_name = global_config.bot.nickname
        personality = global_config.personality.personality
        age = global_config.personality.character_age
        age_stage = global_config.personality.character_age_stage
        hobbies = global_config.personality.character_hobbies
        age_part = f"，{age}岁{age_stage}" if age > 0 and age_stage else ""
        hobby_part = f"，平时喜欢{hobbies}" if hobbies else ""
        base = f"你是{bot_name}{age_part}，{personality}{hobby_part}"
        try:
            from src.modules.modcore.dynamic_persona.persona_switcher import get_persona_switcher
            blended = get_persona_switcher().get_blended_persona_prompt(self.chat_id)
            if blended:
                base = f"{base}\n{blended}"
        except Exception:
            pass
        return base

    async def _fetch_group_summary(self) -> str:
        try:
            from src.modules.modcore.group_impression.group_impression_analyzer import get_group_impression_analyzer
            analyzer = get_group_impression_analyzer()
            state = analyzer.analyze_group_impression(self.chat_id)
            if state:
                return f"氛围: {state.atmosphere.value}; 定位: {state.ai_position.value}; 消息数: {state.recent_message_count}"
            return "氛围未知"
        except Exception as e:
            logger.debug(f"{self.log_prefix}  获取群状态失败: {e}")
            return "群状态获取失败"

    async def _call_small_model(self, prompt: str) -> Optional[str]:
        """调用小模型"""
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            
            # 使用model_monitor任务配置
            llm_request = LLMRequest(
                model_set=model_config.model_task_config.model_monitor,
                request_type="model_monitor"
            )
            
            # 不指定 max_tokens 和 temperature, 使用配置文件中的值
            response, _ = await llm_request.generate_response_async(
                prompt=prompt
            )
            return response
        
        except Exception as e:
            logger.error(f"{self.log_prefix} 小模型调用失败: {e}", exc_info=True)
            return None
    
    def _parse_model_decision(self, response: str) -> ModelDecision:
        """解析模型响应(增强容错)"""
        try:
            response = response.strip()
            # 暴力清理大模型偶尔抽风残留的 markdown 块标记
            if response.startswith("```"):
                lines = response.split('\n')
                if len(lines) > 1:
                    if lines[0].startswith("```"): lines = lines[1:]
                    if lines[-1].startswith("```"): lines = lines[:-1]
                response = ''.join(lines).strip()

            # 基于首尾括号定位提取潜在的 JSON 闭环
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx+1]
                data = json.loads(json_str)
            else:
                raise ValueError("未在响应文本中找到结构化的 JSON 闭口")

            action = data.get("action", "待机")
            duration = int(data.get("duration_seconds", 10))
            silence_minutes = int(data.get("silence_minutes", 0))
            group_vibe = data.get("group_vibe", "未知")

            if group_vibe != "未知":
                 logger.debug(f"{self.log_prefix} [氛围] {group_vibe}")

            # 移除硬编码的时长限制，完全信任模型的决策
            # 只做基本的合理性检查，避免极端值
            if action == "窥屏":
                duration = max(1, min(300, duration))  # 最多5分钟
            elif action in ["待机", "观望"]:
                duration = max(5, min(600, duration))  # 5秒到10分钟，让模型自己决定
            elif action == "休息":
                duration = max(10, min(300, duration))  # 10秒到5分钟
            elif action in ["回复", "回复并放下手机"]:
                duration = 0
            elif action in ["休眠", "放下手机"]:
                duration = 0
                silence_minutes = max(1, min(60, silence_minutes))  # 最多1小时
            else:
                duration = max(5, min(600, duration))

            confidence = float(data.get("confidence", 0.5))
            # 移除随机性干扰，完全信任模型的决策
            # if not self._should_reply_with_randomness(confidence):
            #     if action == "回复" and random.random() < 0.15:
            #         action = "观望"
            #         duration = random.randint(10, 30)

            thought = data.get("thought", "...")

            if action == "回复并放下手机":
                action = "回复"
                self._silence_until = time.time() + max(1, min(30, silence_minutes)) * 60
                logger.info(f"{self.log_prefix} [说完就走] 回完这句就潜水 {silence_minutes} 分钟")

            return ModelDecision(
                action=action,
                duration_seconds=duration,
                thought=data.get("thought", "..."),
                reason=data.get("reason", ""),
                confidence=confidence,
                silence_minutes=silence_minutes,
            )

        except Exception as e:
            logger.warning(f"{self.log_prefix} 解析模型响应失败: {e}, 原始响应: {response[:200]}")
            return ModelDecision(
                action="待机",
                duration_seconds=0,
                thought="...",
                reason="解析失败",
                confidence=0.0,
                silence_minutes=0,
            )

    async def _generate_inner_thought_during_state(self):
        if not self._state_messages:
            return

        try:
            # 根据当前状态和心情决定引用消息数量
            if self._mood > 0.7:  # 心情好，更关注群聊
                context_limit = 10
            elif self._mood < 0.3:  # 心情不好，不太关注
                context_limit = 5
            else:
                context_limit = 7
            
            messages_summary = []
            for msg in self._state_messages[-context_limit:]:
                sender = getattr(msg, 'sender', {}).get('nickname', '用户')
                content = getattr(msg, 'raw_message', '')[:80]
                messages_summary.append(f"{sender}: {content}")

            chat_context = "\n".join(messages_summary)
            latest_msg = messages_summary[-1] if messages_summary else ""

            persona_context = await self._fetch_persona_context()
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.SYSTEM,
                "model_driven_monitor",
                "instant_thought_latest_msg.template",
                persona_context=persona_context,
                context_limit=context_limit,
                chat_context=chat_context,
                latest_msg=latest_msg
            )

            response = await self._call_small_model(prompt)
            if response:
                thought = response.strip().replace('"', '').replace("'", "")
                if len(thought) > 30:
                    thought = thought[:28] + "..."
                self._current_thought = thought
                logger.debug(f"{self.log_prefix}  [模型内心] {self._current_thought}")
        
        except Exception as e:
            logger.debug(f"{self.log_prefix}  生成内心独白失败: {e}")

    async def _update_mood_and_interest(self, message_text: str, psych: dict = None):
        interest = await self._analyze_interest_with_model(message_text, psych)

        if interest["interest_level"] == "感兴趣":
            self._boredom_level = max(0.0, self._boredom_level - 0.15)
            self._mood = min(1.0, self._mood + 0.08)
        elif interest["interest_level"] == "不感兴趣":
            self._boredom_level = min(1.0, self._boredom_level + 0.08)
            self._mood = max(0.0, self._mood - 0.03)

        self._mood += (random.random() - 0.5) * 0.05
        self._mood = max(0.0, min(1.0, self._mood))

        return interest

    def _should_reply_with_randomness(self, base_confidence: float) -> bool:
        random_factor = (random.random() - 0.5) * 0.2
        mood_factor = (self._mood - 0.5) * 0.1
        boredom_factor = -self._boredom_level * 0.15

        final_confidence = base_confidence + random_factor + mood_factor + boredom_factor
        return final_confidence > 0.5

    async def _analyze_interest_with_model(self, message_text: str, psych: dict = None) -> Dict[str, any]:
        """使用模型分析对消息的兴趣程度"""

        # 检查重复（这是硬性规则，不是模型判断）
        is_duplicate = self._is_duplicate_message(message_text)
        if is_duplicate:
            return {
                "interest_level": "不感兴趣",
                "reason": "重复消息",
                "is_duplicate": True,
            }

        # 让模型自己判断兴趣
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            # 高维环境预取
            persona_context = await self._fetch_persona_context()
            group_summary = await self._fetch_group_summary()
            
            chat_type = "【私聊/单线环境】(只对你一个人说话)"
            if self.chat_id and ("group" in self.chat_id.lower() or "qun" in self.chat_id.lower()):
                chat_type = f"【群聊环境】(有多人发言、可能吵架或插科打诨) 氛围参考: {group_summary}"

            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.SYSTEM,
                "model_driven_monitor",
                "interest_analysis.template",
                persona_context=persona_context,
                chat_type=chat_type,
                message_text=message_text
            )

            llm_request = LLMRequest(
                model_set=model_config.model_task_config.model_monitor,
                request_type="model_monitor"
            )

            response, _ = await llm_request.generate_response_async(prompt)
            if response:
                interest_level = "一般"
                reason = "模型未明确"
                thought = "看看再说..."
                
                # 容灾提取模型分析结果的 JSON
                response_clean = response.strip()
                if response_clean.startswith("```"):
                    lines = response_clean.split('\n')
                    if len(lines) > 1:
                        if lines[0].startswith("```"): lines = lines[1:]
                        if lines[-1].startswith("```"): lines = lines[:-1]
                    response_clean = ''.join(lines).strip()

                start_idx = response_clean.find('{')
                end_idx = response_clean.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    try:
                        data = json.loads(response_clean[start_idx:end_idx+1])
                        interest_level = data.get("兴趣", "一般")
                        reason = data.get("原因", reason)
                        thought = data.get("感受", thought)[:20]
                        
                        logger.debug(f"{self.log_prefix} [兴趣评估] 环境:{data.get('环境研判','')} | 契合:{data.get('话题契合','')} | 兴趣:{interest_level}")
                        
                        if interest_level not in ["感兴趣", "一般", "不感兴趣"]:
                            if "不" in interest_level or "无聊" in interest_level or "毫无" in interest_level:
                                interest_level = "不感兴趣"
                            elif "感" in interest_level or "有意思" in interest_level:
                                interest_level = "感兴趣"
                            else:
                                interest_level = "一般"
                    except Exception as parse_e:
                        logger.debug(f"{self.log_prefix} [兴趣评估] JSON解析失败: {parse_e}")
                
                return {
                    "interest_level": interest_level,
                    "reason": reason,
                    "is_duplicate": False,
                    "thought": thought,
                }

        except Exception as e:
            logger.debug(f"{self.log_prefix} 模型兴趣分析失败: {e}")

        # 模型失败时回退到简单规则
        return {
            "interest_level": "一般",
            "reason": "模型分析失败",
            "is_duplicate": False,
        }

    def _is_duplicate_message(self, message_text: str) -> bool:
        """检查消息是否与最近的消息重复"""
        if not message_text:
            return False

        cleaned = message_text.strip().lower()

        for recent in self._recent_message_contents:
            if cleaned == recent or (len(cleaned) > 5 and cleaned in recent) or (len(recent) > 5 and recent in cleaned):
                return True

        self._recent_message_contents.append(cleaned)
        if len(self._recent_message_contents) > self._max_recent_messages:
            self._recent_message_contents.pop(0)

        return False

    def _update_social_state(self, action: str):
        now = time.time()

        if action == "回复":
            self._participation_count += 1
            self._last_participation_time = now
            self._consecutive_silence = 0

            if self._participation_count >= 5:
                self._social_state = "活跃"
            elif now - self._last_participation_time > 600:
                self._social_state = "刚加入"

        elif action == "待机":
            self._consecutive_silence += 1

            if self._consecutive_silence >= 3:
                self._social_state = "沉默"
            if now - self._last_participation_time > 1800:
                self._social_state = "疲惫"

    async def _handle_state_end_with_model(self):
        """
        状态结束时调用模型总结并决策
        """
        # 限制消息数量
        messages_to_analyze = self._state_messages
        if len(messages_to_analyze) > self._max_messages_for_long_duration:
            messages_to_analyze = messages_to_analyze[-self._max_messages_for_long_duration:]
            logger.info(
                f"{self.log_prefix} 消息过多, 限制为最近"
                f"{self._max_messages_for_long_duration}条"
            )
        
        # 构建总结提示词
        messages_summary = []
        for msg in messages_to_analyze:
            sender = getattr(msg, 'sender', {}).get('nickname', '用户')
            content = getattr(msg, 'raw_message', '')[:100]
            messages_summary.append(f"{sender}: {content}")
        
        duration = time.time() - self._state_start_time
        
        from src.config.prompt_loader import get_prompt, PromptCategory
        prompt = get_prompt(
            PromptCategory.SYSTEM,
            "model_driven_monitor",
            "state_end_decision.template",
            duration=int(duration),
            msg_count=len(messages_to_analyze),
            messages_summary=chr(10).join(messages_summary)
        )
        
        response = await self._call_small_model(prompt)
        
        if response:
            try:
                # 解析响应
                if "```json" in response:
                    start = response.find("```json") + 7
                    end = response.find("```", start)
                    response = response[start:end].strip()
                
                data = json.loads(response)
                
                if data.get("should_reply", False):
                    logger.info(f"{self.log_prefix} [观望总结] 决定回复")
                    # 直接执行回复
                    if self._state_messages:
                        await self._notify_reply_needed(self._state_messages[-1])
                else:
                    logger.info(f"{self.log_prefix} [观望总结] 不回复, 回到待机")
            
            except Exception as e:
                logger.error(f"{self.log_prefix} 解析总结失败: {e}")
        
        # 重置状态
        self._state = "待机"
        self._state_messages = []
    
    async def _enter_state(self, state: str, duration: float, reason: str):
        self._state = state
        self._state_start_time = time.time()
        self._state_duration = duration

        history_messages = await self._fetch_recent_messages(limit=10)
        self._state_messages = list(history_messages)

        await self._generate_initial_thought()

        logger.info(
            f"{self.log_prefix}  [进入观望] {reason} | "
            f"持续{int(duration)}秒 | "
            f"预载{len(self._state_messages)}条 | "
            f"心想: {self._current_thought}"
        )

    async def _generate_initial_thought(self):
        if not self._state_messages:
            self._current_thought = "闲着没事干，瞅瞅..."
            return

        try:
            # 根据消息数量和心理状态决定引用数量
            msg_count = len(self._state_messages)
            if msg_count >= 10:
                context_limit = 10
            elif msg_count >= 5:
                context_limit = 7
            else:
                context_limit = msg_count
            
            messages_summary = []
            for msg in self._state_messages[-context_limit:]:
                sender = getattr(msg, 'sender', {}).get('nickname', '用户')
                content = self._get_msg_text(msg)[:60]
                messages_summary.append(f"{sender}: {content}")
            
            chat_context = "\n".join(messages_summary)
            last_msg = self._state_messages[-1]
            user_id = self._get_msg_user_id(last_msg)
            psych = self._get_psych(user_id)
            persona_context = await self._fetch_persona_context()

            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.SYSTEM,
                "model_driven_monitor",
                "initial_thought.template",
                persona_context=persona_context,
                context_limit=context_limit,
                chat_context=chat_context,
                favor=f"{psych.get('favor',0):.0f}",
                annoyance=f"{psych.get('annoyance',0):.0f}",
                mental_fatigue=f"{psych.get('mental_fatigue',0):.0f}"
            )

            response = await self._call_small_model(prompt)
            if response:
                thought = response.strip().replace('"', '').replace("'", "")
                if len(thought) > 30:
                    thought = thought[:28] + "..."
                self._current_thought = thought
                logger.debug(f"{self.log_prefix}  [模型初始内心] {self._current_thought}")
            else:
                self._current_thought = "看看他们在聊啥..."

        except Exception as e:
            logger.debug(f"{self.log_prefix}  生成初始想法失败: {e}")
            self._current_thought = "看看他们在聊啥..."

    async def _fetch_recent_messages(self, limit: int = 10) -> List:
        """获取最近的历史消息"""
        try:
            messages = message_api.get_messages_by_time_in_chat(
                chat_id=self.chat_id,
                start_time=time.time() - 300,  # 最近5分钟
                end_time=time.time(),
                limit=limit,
                limit_mode="latest",
                filter_mai=True,
                filter_command=False,
                filter_intercept_message_level=0,
            )
            return messages if messages else []
        except Exception as e:
            logger.debug(f"{self.log_prefix} 获取历史消息失败: {e}")
            return []
    
    async def _get_context_messages(self) -> List[Dict]:
        """获取对话上下文"""
        try:
            messages = message_api.get_messages_by_time_in_chat(
                chat_id=self.chat_id,
                start_time=time.time() - 600,  # 最近10分钟
                end_time=time.time(),
                limit=50,
                limit_mode="latest",
                filter_mai=False,
                filter_command=False,
                filter_intercept_message_level=0,
            )
            
            context = []
            for msg in messages:
                sender = getattr(msg, 'sender', {})
                is_bot = sender.get('user_id', '') == 'bot'  # 需要实际判断
                
                context.append({
                    "role": "assistant" if is_bot else "user",
                    "name": sender.get('nickname', '用户'),
                    "content": getattr(msg, 'raw_message', '')[:100],
                })
            
            return context
        
        except Exception as e:
            logger.debug(f"{self.log_prefix} 获取上下文失败: {e}")
            return []
    
    async def _notify_reply_needed(self, message, thought: str = ""):
        """通过协调器执行回复(监控通道)"""
        msg_id = self._get_msg_id(message)
        msg_text = self._get_msg_text(message)[:50]

        if msg_id in self._replied_message_ids:
            logger.info(f"{self.log_prefix} [跳过回复] 已回复过此消息: {msg_text}...")
            return

        now = time.time()
        if now - self._last_reply_time < self._min_reply_interval:
            logger.info(f"{self.log_prefix} [跳过回复] 回复间隔太短,需等待{self._min_reply_interval:.0f}秒")
            return

        is_at = getattr(message, 'is_mentioned', False) or getattr(message, 'is_at', False)
        if not is_at:
            from src.config.config import global_config
            bot_name = global_config.bot.nickname
            if bot_name and bot_name in msg_text:
                is_at = True
        can, reason = self.heartfc.can_reply(is_at_me=is_at)
        if not can:
            logger.info(f"{self.log_prefix} [资源约束] 拒绝回复: {reason}")
            return

        reply_target = message
        try:
            latest_msgs = await self._fetch_recent_messages(limit=1)
            if latest_msgs:
                reply_target = latest_msgs[-1]
                latest_text = self._get_msg_text(reply_target)[:30]
                if self._get_msg_id(reply_target) != msg_id:
                    logger.debug(f"{self.log_prefix} [回复目标] 更新为最新消息: {latest_text}...")
        except Exception:
            reply_target = message

        logger.info(f"{self.log_prefix} [请求回复] 模型决定需要回复: {msg_text}...")

        try:
            coordinator = self.heartfc._reply_coordinator
            success = await coordinator.execute_reply(
                channel="monitor",
                reply_func=self.heartfc._execute_urgent_reply,
                message=reply_target,
                thought=thought,
            )
            if success:
                self._replied_message_ids.add(msg_id)
                latest_id = self._get_msg_id(reply_target)
                if latest_id != msg_id:
                    self._replied_message_ids.add(latest_id)
                self._last_reply_time = now
                logger.info(f"{self.log_prefix} [回复成功] 已记录消息ID")
            else:
                logger.warning(f"{self.log_prefix} [回复失败] 未获得权限或执行失败")
        except Exception as e:
            logger.error(f"{self.log_prefix} 执行回复失败: {e}", exc_info=True)
    
    async def _notify_urgent_reply(self, message):
        """通过协调器执行紧急回复(被动通道, 最高优先级)"""
        logger.info(f"{self.log_prefix} [紧急回复] 被@了, 立即回复")
        
        try:
            # 紧急情况也必须保持人设连贯性，获取当前心理状态
            user_id = self._get_msg_user_id(message)
            psych = self._get_psych(user_id)
            
            # 如果已有顺手生成的当前想法，结合进去，否则临时造一个紧急状态的想法
            thought_ctx = self._current_thought if self._current_thought else ""
            if not thought_ctx:
                mood_desc = "心情不错" if self._mood > 0.6 else ("有点烦躁" if self._mood < 0.4 else "状态平稳")
                thought_ctx = f"被人突然叫到了，{mood_desc}，赶紧看看是什么事。"
                
            raw_vibe = thought_ctx.split(')')[0] + ')' if ')' in thought_ctx else ''
            directive_message = f"【当前状态指令】动作: 紧急回复。情绪感知: {raw_vibe}。心理疲劳: {psych.get('mental_fatigue', 0):.0f}，对他的好感: {psych.get('favor', 0):.0f}。你的想法: {thought_ctx}"
            
            coordinator = self.heartfc._reply_coordinator
            success = await coordinator.execute_reply(
                channel="passive",
                reply_func=self.heartfc._execute_urgent_reply,
                message=message,
                thought=directive_message,
            )
            if not success:
                logger.warning(f"{self.log_prefix} [紧急回复失败] 未获得权限或执行失败")
        except Exception as e:
            logger.error(f"{self.log_prefix} 执行紧急回复失败: {e}", exc_info=True)
    
    def get_status(self) -> Dict:
        """获取监控状态"""
        now = time.time()
        heart_flow_state = self._get_heart_flow_state()
        return {
            "active": self._monitor_active,
            "state": self._state,
            "heart_flow_state": heart_flow_state.value if heart_flow_state else "未知",
            "state_duration": self._state_duration,
            "messages_collected": len(self._state_messages),
            "current_thought": self._current_thought,
            "stamina": self._stamina,
            "silence_until": max(0, self._silence_until - now) if self._silence_until > now else 0,
            "cooldown_until": max(0, self._dynamic_cooldown_until - now) if self._dynamic_cooldown_until > now else 0,
            "model_task": "model_monitor",
        }

    def _get_heart_flow_state(self):
        """将当前监控状态映射到 HeartFlowState"""
        from src.chat.heart_flow.config import HeartFlowState
        now = time.time()

        if now < self._silence_until:
            return HeartFlowState.SLACKING

        if self._state in ["窥屏", "观望"] or (now < self._dynamic_cooldown_until and self._state != "回复"):
            return HeartFlowState.PEEKING

        return HeartFlowState.OBSERVING

    def _update_stamina(self):
        """动态脑力恢复：根据当前脑力值动态调整恢复速度"""
        now_ts = time.time()
        elapsed = now_ts - getattr(self, '_last_stamina_recover_time', now_ts)
        current_stamina = getattr(self, '_stamina', 100)
        fatigue_ratio = (100 - current_stamina) / 100.0
        interval = 3.0 - fatigue_ratio * 1.5
        interval = max(1.5, min(3.0, interval))
        if elapsed > interval:
            base_recover = 0.3 + fatigue_ratio * 0.5
            base_recover = max(0.3, min(0.8, base_recover))
            idle_bonus = 1.0 + (elapsed / 60.0) * 0.5
            recover_points = base_recover * idle_bonus
            self._stamina = min(self._stamina_max, current_stamina + recover_points)
            self._last_stamina_recover_time = now_ts
            self._save_stamina_state()

    def _load_stamina_state(self):
        """从 HeartFlowState 加载脑力值"""
        try:
            from src.common.database.database_model import HeartFlowState
            state, _ = HeartFlowState.get_or_create(stream_id=self.chat_id)
            # 使用 accumulated_fatigue 来反算 stamina，或者直接增加 stamina 字段。
            # 为了兼容现有数据库模型，这里我们将 accumulated_fatigue 视作疲劳度 (0 = 满脑力，100 = 脑力耗尽)
            # self._stamina = 100 - fatigue
            self._stamina = max(0.0, 100.0 - getattr(state, 'accumulated_fatigue', 0.0))
        except Exception as e:
            logger.debug(f"{self.log_prefix} 加载脑力状态失败: {e}")

    def _save_stamina_state(self):
        """将脑力值保存到 HeartFlowState"""
        try:
            from src.common.database.database_model import HeartFlowState
            state, _ = HeartFlowState.get_or_create(stream_id=self.chat_id)
            state.accumulated_fatigue = 100.0 - self._stamina
            state.updated_at = time.time()
            state.save()
        except Exception as e:
            logger.debug(f"{self.log_prefix} 保存脑力状态失败: {e}")

import time
import asyncio
from typing import Optional, Dict, List, Any, Callable
from src.common.logger import get_logger
from src.modules.modcore.dynamic_persona.persona_controller import get_persona_controller, DynamicPersona
from src.modules.modcore.dynamic_persona.persona_generator import get_persona_generator

logger = get_logger("persona")


class PersonaSwitcher:
    # 人格切换器，管理自动切换/恢复/过渡
    PRIMARY_WEIGHT = 0.5
    AUXILIARY_WEIGHT = 0.5
    TIME_BALANCE_MAX = 1800.0

    def __init__(self, config: Optional[Dict] = None):
        self._cfg = config or {}
        self._pending_reverts: Dict[str, asyncio.Task] = {}
        self._switch_callbacks: List[Callable] = []
        self._transitions: Dict[str, Dict] = {}
        self._last_switch_time: Dict[str, float] = {}

    def register_callback(self, callback: Callable):
        self._switch_callbacks.append(callback)

    async def _notify_callbacks(self, stream_id: str, persona: DynamicPersona, action: str):
        for cb in self._switch_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(stream_id, persona, action)
                else:
                    cb(stream_id, persona, action)
            except Exception as e:
                logger.debug(f"回调执行失败: {e}")

    async def switch_with_auto_revert(
        self,
        stream_id: str,
        persona_id: str,
        duration_seconds: float = 300.0,
        reason: str = "",
    ) -> bool:
        controller = get_persona_controller()
        success = controller.switch_persona(stream_id, persona_id, duration_seconds, reason)
        if not success:
            return False
        if stream_id in self._pending_reverts:
            self._pending_reverts[stream_id].cancel()
        async def auto_revert():
            await asyncio.sleep(duration_seconds)
            controller.revert_to_main(stream_id)
            self._pending_reverts.pop(stream_id, None)
            main_persona = controller.get_main_persona()
            if main_persona is not None:
                await self._notify_callbacks(stream_id, main_persona, "revert")
            logger.debug(f"自动切回主人格: {stream_id[:12]}")
        task = asyncio.create_task(auto_revert())
        self._pending_reverts[stream_id] = task
        self._last_switch_time[stream_id] = time.time()
        persona = controller.get_persona(persona_id)
        if persona:
            await self._notify_callbacks(stream_id, persona, "switch")
        return True

    async def trigger_trauma_switch(
        self,
        stream_id: str,
        trauma_content: str,
        user_id: str = "default",
        duration_seconds: Optional[float] = None,
    ) -> bool:
        generator = get_persona_generator()
        controller = get_persona_controller()
        if duration_seconds is None:
            duration_seconds = 600.0
        uid = user_id if user_id and user_id != "default" else "guest"
        mood_context = {
            "user_id": uid,
            "channel_id": stream_id,
            "current_content": trauma_content[:200],
            "trauma_level": 8.5,
            "intensity": 9,
            "triggers": ["trauma_switch", "emergency"],
            "primary_emotion": "devastated",
            "mood_level": 8.5
        }
        persona = await generator.generate_trauma_sensitive_persona("trauma", mood_context)
        if not persona:
            return False
        controller.add_persona(persona)
        success = await self.switch_with_auto_revert(
            stream_id, persona.persona_id, duration_seconds,
            reason=f"创伤防御: {trauma_content[:30]}",
        )
        return success

    async def trigger_trauma_sensitive_switch(
        self,
        stream_id: str,
        persona_type: str,
        mood_context: Dict[str, Any],
        duration_seconds: float = 180.0,
    ) -> bool:
        # 创伤敏感状态下的人格切换
        generator = get_persona_generator()
        controller = get_persona_controller()
        try:
            persona = await generator.generate_trauma_sensitive_persona(persona_type, mood_context)
            if persona:
                controller.add_persona(persona)
                success = await self.switch_with_auto_revert(
                    stream_id, persona.persona_id, duration_seconds,
                    reason=f"创伤敏感触发: {persona_type}"
                )
                if success:
                    intensity = mood_context.get('intensity', 5)
                    logger.debug(f"人格切换: {persona.name} 强度:{intensity} 持续:{duration_seconds}秒")
                return success
            else:
                logger.warning("创伤人格生成失败")
                return False
        except Exception as e:
            logger.debug(f"创伤人格切换失败: {e}")
            return False

    async def auto_detect_and_switch(
        self,
        stream_id: str,
        content: str,
        user_id: str = "guest",
        context: Optional[Dict] = None,
    ) -> bool:
        controller = get_persona_controller()
        if controller.is_persona_active(stream_id):
            return False
        should_switch, persona_id, reason = await controller.should_switch_persona(
            stream_id, content, user_id, context
        )
        if should_switch and persona_id:
            return await self.switch_with_auto_revert(
                stream_id, persona_id, reason=reason,
            )
        return False

    async def check_and_switch(
        self,
        user_id: str,
        stream_id: str,
        context: Dict[str, Any],
    ) -> bool:
        # 记录状态，由LLM模型决定是否切换（不使用固定阈值）
        trauma_score = context.get("trauma_score", 0.0)
        sentiment = context.get("sentiment", "neutral")
        intensity = context.get("intensity", 0.0)
        state_record = {
            "trauma_score": trauma_score,
            "sentiment": sentiment,
            "intensity": intensity,
        }
        self._transitions[stream_id] = self._transitions.get(stream_id, {})
        self._transitions[stream_id]["state_record"] = state_record
        self._transitions[stream_id]["last_check"] = time.time()
        return False

    def force_revert(self, stream_id: str):
        if stream_id in self._pending_reverts:
            self._pending_reverts[stream_id].cancel()
            self._pending_reverts.pop(stream_id, None)
        controller = get_persona_controller()
        controller.revert_to_main(stream_id)
        logger.debug(f"强制切回主人格: {stream_id[:12]}")

    def get_active_persona(self, stream_id: str):
        controller = get_persona_controller()
        return controller.get_active_persona(stream_id)

    def get_active_info(self, stream_id: str) -> Optional[Dict]:
        controller = get_persona_controller()
        persona = controller.get_active_persona(stream_id)
        if not persona:
            return None
        remaining = controller.get_remaining_time(stream_id)
        return {
            "persona_id": persona.persona_id,
            "name": persona.name,
            "tone": persona.tone,
            "remaining_seconds": remaining,
            "is_main": persona == controller.get_main_persona(),
            "persona_obj": persona,
        }

    def get_stats(self) -> Dict:
        return {
            "pending_reverts": len(self._pending_reverts),
            "callbacks_count": len(self._switch_callbacks),
            "active_transitions": len(self._transitions),
        }

    async def transition_persona(
        self,
        stream_id: str,
        to_persona_id: str,
        transition_duration: float = 60.0,
        reason: str = "",
    ) -> bool:
        # 平滑过渡到新人格（分三阶段）
        controller = get_persona_controller()
        to_persona = controller.get_persona(to_persona_id)
        if not to_persona:
            return False
        from_persona = controller.get_active_persona(stream_id)
        self._transitions[stream_id] = {
            "from_persona": from_persona.to_dict() if from_persona else None,
            "to_persona_id": to_persona_id,
            "start_time": time.time(),
            "duration": transition_duration,
            "reason": reason,
            "phase": "starting",
        }
        success = await self.switch_with_auto_revert(
            stream_id, to_persona_id, transition_duration + 300, reason
        )
        if success:
            async def update_phases():
                phase_1 = transition_duration * 0.2
                phase_2 = transition_duration * 0.6
                await asyncio.sleep(phase_1)
                if stream_id in self._transitions:
                    self._transitions[stream_id]["phase"] = "blending"
                await asyncio.sleep(phase_2)
                if stream_id in self._transitions:
                    self._transitions[stream_id]["phase"] = "completing"
                await asyncio.sleep(transition_duration * 0.2)
                self._transitions.pop(stream_id, None)
            asyncio.create_task(update_phases())
        return success

    def get_transition_progress(self, stream_id: str) -> Optional[Dict]:
        transition = self._transitions.get(stream_id)
        if not transition or "start_time" not in transition:
            return None
        elapsed = time.time() - transition["start_time"]
        duration = transition.get("duration", 60.0)
        progress = min(1.0, elapsed / duration) if duration > 0 else 1.0
        return {
            "progress": progress,
            "phase": transition.get("phase", "unknown"),
            "to_persona_id": transition.get("to_persona_id"),
            "reason": transition.get("reason", ""),
        }

    async def trigger_mood_switch(self, stream_id: str, mood: str,
                                   user_id: str = "default",
                                   duration_seconds: float = 300.0) -> bool:
        from src.modules.modcore.dynamic_persona.persona_generator import get_persona_generator
        generator = get_persona_generator()
        controller = get_persona_controller()
        mood_context = {
            "user_id": user_id if user_id != "default" else "guest",
            "channel_id": stream_id,
            "current_content": f"状态迁移: {mood}",
            "mood": mood
        }
        persona = None
        try:
            if hasattr(generator, 'generate_from_mood'):
                persona = await generator.generate_from_mood(mood_context)
        except Exception as e:
            logger.debug(f"动态人格生成失败: {e}")
        if persona:
            controller.add_persona(persona)
            return await self.switch_with_auto_revert(
                stream_id, persona.persona_id, duration_seconds,
                reason=f"心情变化: {mood}",
            )
        return False

    def get_transition_state(self, stream_id: str) -> Optional[Dict]:
        if stream_id not in self._transitions:
            return None
        state = self._transitions[stream_id]
        progress = state.get("progress", 0.0)
        if progress < 0.5:
            blend_ratio = 2 * progress * progress
        else:
            blend_ratio = 1 - pow(-2 * progress + 2, 2) / 2
        return {
            "from_persona": state.get("from_persona"),
            "to_persona": state.get("to_persona"),
            "progress": progress,
            "stage": state.get("stage", "unknown"),
            "blend_ratio": blend_ratio,
            "reason": state.get("reason", ""),
        }

    def is_transitioning(self, stream_id: str) -> bool:
        return stream_id in self._transitions

    def get_persona_weight(self, stream_id: str, trauma_score: float = 0.0) -> Dict[str, float]:
        controller = get_persona_controller()
        active_persona = controller.get_active_persona(stream_id)
        main_persona = controller.get_main_persona()
        primary = 0.6
        auxiliary = 0.4
        if not active_persona or not main_persona:
            return {"primary": primary, "auxiliary": auxiliary}
        now = time.time()
        last_switch = self._last_switch_time.get(stream_id, now)
        time_diff = min(now - last_switch, 1800)
        time_balance = time_diff / 1800
        is_main = active_persona.persona_id == main_persona.persona_id
        if is_main:
            primary += 0.2 * time_balance
        else:
            primary -= 0.2 * time_balance
        trauma_impact = min(0.5, trauma_score * 0.05)
        if not is_main:
            auxiliary += trauma_impact
            primary -= trauma_impact
        primary = max(0.2, min(0.8, primary))
        auxiliary = 1.0 - primary
        return {"primary": primary, "auxiliary": auxiliary}

    def get_blended_persona_data(self, stream_id: str) -> Dict[str, Any]:
        controller = get_persona_controller()
        active_persona = controller.get_active_persona(stream_id)
        main_persona = controller.get_main_persona()
        if not active_persona:
            return {
                "active_persona": None, "main_persona": main_persona,
                "weight": {"primary": 1.0, "auxiliary": 0.0},
                "display_mode": "only_active"
            }
        trauma_score = 0.0
        transition_data = self._transitions.get(stream_id, {})
        trauma_level = transition_data.get("trauma_level", 0)
        if trauma_level == 3:
            trauma_score = 9.0
        elif trauma_level == 2:
            trauma_score = 6.0
        weight = self.get_persona_weight(stream_id, trauma_score)
        is_main = active_persona.persona_id == main_persona.persona_id
        display_mode = "only_active" if is_main else "blended"
        return {
            "active_persona": active_persona, "main_persona": main_persona,
            "weight": weight, "display_mode": display_mode,
            "is_main": is_main, "trauma_score": trauma_score,
            "trauma_level": trauma_level,
        }

    def cancel_transition(self, stream_id: str) -> bool:
        if stream_id not in self._transition_tasks:
            return False
        self._transition_tasks[stream_id].cancel()
        self._transitions.pop(stream_id, None)
        self._transition_tasks.pop(stream_id, None)
        logger.debug(f"人格过渡已取消: {stream_id[:12]}")
        return True

    def get_blended_persona_prompt(self, stream_id: str) -> str:
        transition = self.get_transition_state(stream_id)
        if transition:
            from_persona = transition["from_persona"]
            to_persona = transition["to_persona"]
            blend_ratio = transition["blend_ratio"]
            stage = transition["stage"]
            if stage == "starting":
                return (f"你目前主要保持{from_persona.name if from_persona else '正常'}的状态，"
                        f"但开始感受到一些{to_persona.name}的情绪波动。")
            elif stage == "mixing":
                return (f"你正处于情绪变化中。{from_persona.name if from_persona else '之前'}特征({(1-blend_ratio)*100:.0f}%)和"
                        f"{to_persona.name}特征({blend_ratio*100:.0f}%)混合。语言可能有些不稳定。")
            elif stage == "ending":
                return f"你已基本进入{to_persona.name}状态。之前的特征几乎消失。"
        controller = get_persona_controller()
        active_persona = controller.get_active_persona(stream_id)
        if active_persona:
            main_persona = controller.get_main_persona()
            is_main = main_persona and active_persona.persona_id == main_persona.persona_id
            prompt = active_persona.to_prompt_addition()
            if not is_main and prompt:
                return prompt
            elif is_main:
                # 主人格也返回基本提示，不要返回空字符串
                return f"你保持{active_persona.name}的状态。"
        # 没有活跃人格时返回默认提示
        return "你保持正常的状态。"

    async def transition_task(self, stream_id: str, to_persona_id: str,
                               transition_duration: float = 60.0):
        try:
            start_time = time.time()
            controller = get_persona_controller()
            while True:
                elapsed = time.time() - start_time
                progress = min(1.0, elapsed / transition_duration)
                self._transitions[stream_id]["progress"] = progress
                if progress < 0.2:
                    self._transitions[stream_id]["stage"] = "starting"
                elif progress < 0.8:
                    self._transitions[stream_id]["stage"] = "mixing"
                elif progress < 1.0:
                    self._transitions[stream_id]["stage"] = "ending"
                else:
                    self._transitions[stream_id]["stage"] = "complete"
                    break
                await asyncio.sleep(1.0)
            controller.switch_persona(stream_id, to_persona_id, 300.0, "过渡完成")
            self._transitions.pop(stream_id, None)
            self._transition_tasks.pop(stream_id, None)
        except asyncio.CancelledError:
            self._transitions.pop(stream_id, None)
        except Exception as e:
            logger.warning(f"人格过渡失败: {e}")
            self._transitions.pop(stream_id, None)


_switcher: Optional[PersonaSwitcher] = None


def get_persona_switcher() -> PersonaSwitcher:
    global _switcher
    if _switcher is None:
        _switcher = PersonaSwitcher()
    return _switcher

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from src.common.logger import get_logger
from src.common.task_utils import safe_create_task
from src.modules.modcore.dynamic_persona.persona_controller import (
    DynamicPersona,
    get_persona_controller,
)
from src.modules.modcore.dynamic_persona.runtime_config import dynamic_persona_module_view

logger = get_logger("人格切换")


class PersonaSwitcher:
    """人格切换器

    管理人格的自动切换、恢复、平滑过渡。
    支持基于创伤状态的紧急切换。
    """

    def __init__(self, config_engine: Optional[Any] = None):
        del config_engine
        self._pending_reverts: Dict[str, asyncio.Task] = {}
        self._switch_callbacks: List[Callable] = []
        self._transitions: Dict[str, Dict[str, Any]] = {}
        self._last_switch_time: Dict[str, float] = {}
        self._transition_tasks: Dict[str, asyncio.Task] = {}
        self._load_config()

    def _load_config(self) -> None:
        config = dynamic_persona_module_view("persona_switcher")
        self._auto_revert_default_seconds = float(
            config.get("auto_revert_default_seconds", 300.0)
        )
        self._trauma_switch_duration_seconds = float(
            config.get("trauma_switch_duration_seconds", 600.0)
        )
        self._transition_duration_seconds = float(
            config.get("transition_duration_seconds", 60.0)
        )
        self._transition_revert_buffer_seconds = float(
            config.get("transition_revert_buffer_seconds", 300.0)
        )
        self._time_balance_max_seconds = float(
            config.get("time_balance_max_seconds", 1800.0)
        )
        self._primary_base_weight = float(
            config.get("primary_base_weight", 0.6)
        )
        self._primary_time_balance_scale = float(
            config.get("primary_time_balance_scale", 0.2)
        )
        self._trauma_impact_cap = float(
            config.get("trauma_impact_cap", 0.5)
        )
        self._trauma_impact_scale = float(
            config.get("trauma_impact_scale", 0.05)
        )
        self._primary_weight_floor = float(
            config.get("primary_weight_floor", 0.2)
        )
        self._primary_weight_ceiling = float(
            config.get("primary_weight_ceiling", 0.8)
        )
        self._transition_phase_1_ratio = float(
            config.get("transition_phase_1_ratio", 0.2)
        )
        self._transition_phase_2_ratio = float(
            config.get("transition_phase_2_ratio", 0.6)
        )
        self._transition_phase_3_ratio = float(
            config.get("transition_phase_3_ratio", 0.2)
        )

    def register_callback(self, callback: Callable) -> None:
        self._switch_callbacks.append(callback)

    async def _notify_callbacks(
        self, stream_id: str, persona: DynamicPersona, action: str
    ) -> None:
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
        duration_seconds = (
            self._auto_revert_default_seconds
            if duration_seconds is None
            else duration_seconds
        )
        success = controller.switch_persona(
            stream_id, persona_id, duration_seconds, reason
        )
        if not success:
            return False
        if stream_id in self._pending_reverts:
            self._pending_reverts[stream_id].cancel()

        async def auto_revert():
            try:
                await asyncio.sleep(duration_seconds)
                controller.revert_to_main(stream_id)
                main_persona = controller.get_main_persona()
                if main_persona is not None:
                    await self._notify_callbacks(stream_id, main_persona, "revert")
                logger.debug(f"自动切回主人格: {stream_id[:12]}")
            finally:
                self._pending_reverts.pop(stream_id, None)

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
        controller = get_persona_controller()
        if duration_seconds is None:
            duration_seconds = self._trauma_switch_duration_seconds
        trauma_personas = [
            p
            for p in controller.list_personas()
            if any(kw in p.name for kw in ["创伤", "防御", "崩溃", "委屈"])
        ]
        if not trauma_personas:
            logger.warning("没有可用的创伤人格")
            return False
        selected = trauma_personas[0]
        return await self.switch_with_auto_revert(
            stream_id,
            selected.persona_id,
            duration_seconds,
            reason=f"创伤防御: {trauma_content[:30]}",
        )

    async def auto_detect_and_switch(
        self,
        stream_id: str,
        content: str,
        user_id: str = "guest",
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        controller = get_persona_controller()
        if controller.is_persona_active(stream_id):
            return False
        should_switch, persona_id, reason = (
            await controller.should_switch_persona(
                stream_id, content, user_id, context
            )
        )
        if should_switch and persona_id:
            return await self.switch_with_auto_revert(
                stream_id, persona_id, reason=reason
            )
        return False

    def force_revert(self, stream_id: str) -> None:
        if stream_id in self._pending_reverts:
            self._pending_reverts[stream_id].cancel()
            self._pending_reverts.pop(stream_id, None)
        controller = get_persona_controller()
        controller.revert_to_main(stream_id)
        logger.debug(f"强制切回主人格: {stream_id[:12]}")

    def get_active_persona(self, stream_id: str) -> Optional[DynamicPersona]:
        controller = get_persona_controller()
        return controller.get_active_persona(stream_id)

    def get_active_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
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

    def get_stats(self) -> Dict[str, Any]:
        return {
            "pending_reverts": len(self._pending_reverts),
            "callbacks_count": len(self._switch_callbacks),
            "active_transitions": len(self._transitions),
        }

    async def transition_persona(
        self,
        stream_id: str,
        to_persona_id: str,
        transition_duration: Optional[float] = None,
        reason: str = "",
    ) -> bool:
        transition_duration = (
            self._transition_duration_seconds
            if transition_duration is None
            else transition_duration
        )
        controller = get_persona_controller()
        to_persona = controller.get_persona(to_persona_id)
        if not to_persona:
            return False
        if stream_id in self._transition_tasks:
            old_task = self._transition_tasks.pop(stream_id, None)
            if old_task and not old_task.done():
                old_task.cancel()
            self._transitions.pop(stream_id, None)
        from_persona = controller.get_active_persona(stream_id)
        self._transitions[stream_id] = {
            "from_persona": from_persona.to_dict() if from_persona else None,
            "to_persona_id": to_persona_id,
            "start_time": time.time(),
            "duration": transition_duration,
            "reason": reason,
            "phase": "starting",
            "progress": 0.0,
            "stage": "starting",
        }
        success = await self.switch_with_auto_revert(
            stream_id,
            to_persona_id,
            transition_duration + self._transition_revert_buffer_seconds,
            reason,
        )
        if success:
            task = safe_create_task(
                self._update_phases(stream_id, transition_duration),
                name=f"persona_phase_{stream_id[:8]}",
            )
            self._transition_tasks[stream_id] = task
        return success

    async def _update_phases(self, stream_id: str, duration: float) -> None:
        try:
            phase_1 = duration * self._transition_phase_1_ratio
            phase_2 = duration * self._transition_phase_2_ratio
            await asyncio.sleep(phase_1)
            if stream_id in self._transitions:
                self._transitions[stream_id]["phase"] = "blending"
                self._transitions[stream_id]["stage"] = "mixing"
            await asyncio.sleep(phase_2)
            if stream_id in self._transitions:
                self._transitions[stream_id]["phase"] = "completing"
                self._transitions[stream_id]["stage"] = "ending"
            await asyncio.sleep(duration * self._transition_phase_3_ratio)
            self._transitions.pop(stream_id, None)
            self._transition_tasks.pop(stream_id, None)
        except asyncio.CancelledError:
            pass

    def get_transition_progress(
        self, stream_id: str
    ) -> Optional[Dict[str, Any]]:
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

    def get_persona_weight(
        self, stream_id: str, trauma_score: float = 0.0
    ) -> Dict[str, float]:
        controller = get_persona_controller()
        active_persona = controller.get_active_persona(stream_id)
        main_persona = controller.get_main_persona()
        primary = self._primary_base_weight
        auxiliary = 1.0 - primary
        if not active_persona or not main_persona:
            return {"primary": primary, "auxiliary": auxiliary}
        now = time.time()
        last_switch = self._last_switch_time.get(stream_id, now)
        time_diff = min(now - last_switch, self._time_balance_max_seconds)
        time_balance = time_diff / max(self._time_balance_max_seconds, 1.0)
        is_main = active_persona.persona_id == main_persona.persona_id
        if is_main:
            primary += self._primary_time_balance_scale * time_balance
        else:
            primary -= self._primary_time_balance_scale * time_balance
        trauma_impact = min(
            self._trauma_impact_cap, trauma_score * self._trauma_impact_scale
        )
        if not is_main:
            auxiliary += trauma_impact
            primary -= trauma_impact
        primary = max(
            self._primary_weight_floor,
            min(self._primary_weight_ceiling, primary),
        )
        auxiliary = 1.0 - primary
        return {"primary": primary, "auxiliary": auxiliary}

    def get_blended_persona_data(self, stream_id: str) -> Dict[str, Any]:
        controller = get_persona_controller()
        active_persona = controller.get_active_persona(stream_id)
        main_persona = controller.get_main_persona()
        if not active_persona and not main_persona:
            return {
                "active_persona": None,
                "main_persona": None,
                "weight": {"primary": 1.0, "auxiliary": 0.0},
                "display_mode": "uninitialized",
            }
        if not active_persona:
            return {
                "active_persona": None,
                "main_persona": main_persona,
                "weight": {"primary": 1.0, "auxiliary": 0.0},
                "display_mode": "only_active",
            }
        weight = self.get_persona_weight(stream_id, 0.0)
        is_main = (
            active_persona.persona_id == main_persona.persona_id
            if main_persona
            else True
        )
        display_mode = "only_active" if is_main else "blended"
        return {
            "active_persona": active_persona,
            "main_persona": main_persona,
            "weight": weight,
            "display_mode": display_mode,
            "is_main": is_main,
        }

    def get_blended_persona_prompt(self, stream_id: str) -> str:
        transition = self.get_transition_progress(stream_id)
        if transition:
            progress = transition["progress"]
            to_id = transition.get("to_persona_id", "")
            controller = get_persona_controller()
            to_persona = controller.get_persona(to_id)
            if progress < 0.3:
                return f"你开始感受到一些{
                    to_persona.name if to_persona else '新'}的情绪波动。"
            elif progress < 0.7:
                return f"你正处于情绪变化中，{
                    to_persona.name if to_persona else '新'}特征逐渐显现。"
            else:
                return f"你已基本进入{to_persona.name if to_persona else '新'}状态。"
        controller = get_persona_controller()
        active_persona = controller.get_active_persona(stream_id)
        if active_persona:
            return active_persona.to_prompt_addition()
        return "你保持正常的状态。"

    def is_transitioning(self, stream_id: str) -> bool:
        return stream_id in self._transitions

    def cancel_transition(self, stream_id: str) -> bool:
        task = self._transition_tasks.pop(stream_id, None)
        if task is None:
            return False
        if not task.done():
            task.cancel()
        self._transitions.pop(stream_id, None)
        logger.debug(f"人格过渡已取消: {stream_id[:12]}")
        return True


_persona_switcher_instance: Optional[PersonaSwitcher] = None


def get_persona_switcher() -> PersonaSwitcher:
    """获取人格切换器单例"""
    global _persona_switcher_instance
    if _persona_switcher_instance is None:
        _persona_switcher_instance = PersonaSwitcher()
    return _persona_switcher_instance

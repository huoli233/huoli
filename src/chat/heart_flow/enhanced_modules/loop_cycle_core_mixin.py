from src.chat.heart_flow.enhanced_modules.loop_main_driver_mixin import LoopMainDriverMixin
from src.chat.heart_flow.enhanced_modules.loop_reply_execution_mixin import LoopReplyExecutionMixin
from src.chat.heart_flow.enhanced_modules.loop_phase_transition_mixin import LoopPhaseTransitionMixin


class LoopCycleCoreMixin(
    LoopMainDriverMixin,
    LoopReplyExecutionMixin,
    LoopPhaseTransitionMixin,
):
    pass

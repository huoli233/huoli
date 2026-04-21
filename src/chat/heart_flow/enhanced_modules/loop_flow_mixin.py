from src.chat.heart_flow.enhanced_modules.loop_cycle_core_mixin import LoopCycleCoreMixin
from src.chat.heart_flow.enhanced_modules.loop_state_flow_mixin import LoopStateFlowMixin
from src.chat.heart_flow.enhanced_modules.loop_resource_feedback_mixin import LoopResourceFeedbackMixin


class EnhancedLoopFlowMixin(
    LoopCycleCoreMixin,
    LoopStateFlowMixin,
    LoopResourceFeedbackMixin,
):
    pass

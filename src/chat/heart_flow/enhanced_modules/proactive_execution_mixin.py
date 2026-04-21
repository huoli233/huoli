from src.chat.heart_flow.enhanced_modules.proactive_context_prompt_mixin import ProactiveContextPromptMixin
from src.chat.heart_flow.enhanced_modules.proactive_reactive_flow_mixin import ProactiveReactiveFlowMixin
from src.chat.heart_flow.enhanced_modules.proactive_idle_reply_mixin import ProactiveIdleReplyMixin


class EnhancedProactiveExecutionMixin(
    ProactiveContextPromptMixin,
    ProactiveReactiveFlowMixin,
    ProactiveIdleReplyMixin,
):
    pass

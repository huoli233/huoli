from src.chat.heart_flow.enhanced_modules.strategy_action_state_mixin import StrategyActionStateMixin
from src.chat.heart_flow.enhanced_modules.strategy_relation_style_mixin import StrategyRelationStyleMixin
from src.chat.heart_flow.enhanced_modules.strategy_proactive_decision_mixin import StrategyProactiveDecisionMixin


class EnhancedStrategyIntegrationMixin(
    StrategyActionStateMixin,
    StrategyRelationStyleMixin,
    StrategyProactiveDecisionMixin,
):
    pass

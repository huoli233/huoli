from src.chat.heart_flow.enhanced_modules.scene_bot_lifecycle_mixin import SceneBotLifecycleMixin
from src.chat.heart_flow.enhanced_modules.scene_context_analysis_mixin import SceneContextAnalysisMixin
from src.chat.heart_flow.enhanced_modules.scene_planner_bridge_mixin import ScenePlannerBridgeMixin


class EnhancedSceneAnalysisMixin(
    SceneBotLifecycleMixin,
    SceneContextAnalysisMixin,
    ScenePlannerBridgeMixin,
):
    pass

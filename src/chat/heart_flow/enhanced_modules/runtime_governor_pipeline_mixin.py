from src.chat.heart_flow.enhanced_modules.runtime_subjective_signal_mixin import RuntimeSubjectiveSignalMixin
from src.chat.heart_flow.enhanced_modules.runtime_watch_governor_mixin import RuntimeWatchGovernorMixin
from src.chat.heart_flow.enhanced_modules.runtime_sensory_pipeline_mixin import RuntimeSensoryPipelineMixin


class RuntimeGovernorPipelineMixin(
    RuntimeSubjectiveSignalMixin,
    RuntimeWatchGovernorMixin,
    RuntimeSensoryPipelineMixin,
):
    pass

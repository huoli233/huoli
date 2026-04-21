from src.chat.utils.prompt_builder import Prompt
from src.config.prompt_loader import get_prompt_raw, PromptCategory


def _load_template(filename: str, key: str) -> str:
    return get_prompt_raw(PromptCategory.REPLY, filename, key)


def init_replyer_prompt():
    Prompt(
        _load_template("group", "light.template"),
        "replyer_prompt_0",
    )
    Prompt(
        _load_template("group", "deep.template"),
        "replyer_prompt",
    )


def get_group_responder_prompt(**kwargs) -> str:
    template = _load_template("group", "standard.template")
    return template.format(**kwargs)


def get_solo_responder_prompt(**kwargs) -> str:
    template = _load_template("private", "standard.template")
    return template.format(**kwargs)


def get_solo_self_continuation_prompt(**kwargs) -> str:
    template = _load_template("private", "self_continuation.template")
    return template.format(**kwargs)


def get_group_self_continuation_prompt(**kwargs) -> str:
    template = _load_template("group", "self_continuation.template")
    return template.format(**kwargs)


def get_group_responder_prompt_by_think_level(think_level: int = 1, **kwargs) -> str:
    defaults = {
        "knowledge_prompt": "", "tool_info_block": "", "extra_info_block": "",
        "expression_habits_block": "", "memory_retrieval": "", "jargon_explanation": "",
        "blended_persona_prompt": "", "bot_name": "", "time_block": "",
        "dialogue_prompt": "", "reply_target_block": "", "planner_reasoning": "",
        "identity": "", "keywords_reaction_prompt": "", "reply_style": "",
    }
    merged_kwargs = {**defaults, **kwargs}
    key = "light.template" if think_level == 0 else "deep.template"
    template = _load_template("group", key)
    return template.format(**merged_kwargs)


def get_solo_responder_prompt_by_think_level(think_level: int = 1, **kwargs) -> str:
    defaults = {
        "knowledge_prompt": "", "tool_info_block": "", "extra_info_block": "",
        "expression_habits_block": "", "memory_retrieval": "", "jargon_explanation": "",
        "sender_name": "", "time_block": "", "dialogue_prompt": "",
        "reply_target_block": "", "identity": "",
        "keywords_reaction_prompt": "", "reply_style": "",
    }
    merged_kwargs = {**defaults, **kwargs}
    key = "light.template" if think_level == 0 else "deep.template"
    template = _load_template("private", key)
    return template.format(**merged_kwargs)

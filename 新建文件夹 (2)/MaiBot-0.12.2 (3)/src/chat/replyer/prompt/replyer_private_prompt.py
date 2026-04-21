from src.chat.utils.prompt_builder import Prompt
from src.config.prompt_loader import get_prompt_raw, PromptCategory


def init_replyer_private_prompt():
    Prompt(
        get_prompt_raw(PromptCategory.REPLY, "private", "replyer_prompt.template"),
        "private_replyer_prompt",
    )
    Prompt(
        get_prompt_raw(PromptCategory.REPLY, "private", "replyer_self_prompt.template"),
        "private_replyer_self_prompt",
    )

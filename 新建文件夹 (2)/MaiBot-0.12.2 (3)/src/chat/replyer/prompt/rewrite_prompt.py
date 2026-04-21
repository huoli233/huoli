from src.chat.utils.prompt_builder import Prompt
from src.config.prompt_loader import get_prompt_raw, PromptCategory


def init_rewrite_prompt():
    Prompt(get_prompt_raw(PromptCategory.REPLY, "rewrite", "chat_target.group1"), "chat_target_group1")
    Prompt(get_prompt_raw(PromptCategory.REPLY, "rewrite", "chat_target.private1"), "chat_target_private1")
    Prompt(get_prompt_raw(PromptCategory.REPLY, "rewrite", "chat_target.group2"), "chat_target_group2")
    Prompt(get_prompt_raw(PromptCategory.REPLY, "rewrite", "chat_target.private2"), "chat_target_private2")
    Prompt(get_prompt_raw(PromptCategory.REPLY, "rewrite", "expressor.template"), "default_expressor_prompt")

from src.chat.utils.prompt_builder import Prompt
from src.config.prompt_loader import get_prompt_raw, PromptCategory


def init_lpmm_prompt():
    Prompt(
        get_prompt_raw(PromptCategory.REPLY, "lpmm", "get_knowledge.template"),
        name="lpmm_get_knowledge_prompt",
    )

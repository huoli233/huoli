import src.chat.prompts.catalog  # noqa: F401  触发提示词注册
from src.chat.utils.prompt_builder import global_prompt_registry


def _fetch_registered_text(name: str) -> str:
    """从全局注册表同步获取提示词原文"""
    tpl = global_prompt_registry._global_prompts.get(name)
    if tpl is None:
        raise KeyError(f"提示词 '{name}' 未在catalog中注册")
    return str(tpl)


def build_entity_extract_context(paragraph: str) -> str:
    """构建实体提取的完整提示文本"""
    system_text = _fetch_registered_text("entity_extract_system")
    return f"""{system_text}

段落：
```
{paragraph}
```"""


def build_rdf_triple_extract_context(paragraph: str, entities: str) -> str:
    """构建RDF三元组提取的完整提示文本"""
    system_text = _fetch_registered_text("rdf_triple_extract_system")
    return f"""{system_text}

段落：
```
{paragraph}
```

实体列表：
```
{entities}
```"""

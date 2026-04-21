from src.config.prompt_loader import get_prompt_raw, PromptCategory


def _sys(key: str) -> str:
    return get_prompt_raw(PromptCategory.SYSTEM, "knowledge", key)


entity_extract_system_prompt = _sys("entity_extract.system")
rdf_triple_extract_system_prompt = _sys("rdf_triple_extract.system")
qa_system_prompt = _sys("qa.system")


def build_entity_extract_context(paragraph: str) -> str:
    """构建实体提取的完整提示文本"""
    sys_prompt = _sys("entity_extract.system")
    return f"""{sys_prompt}

段落：
```
{paragraph}
```"""


def build_rdf_triple_extract_context(paragraph: str, entities: str) -> str:
    """构建RDF三元组提取的完整提示文本"""
    sys_prompt = _sys("rdf_triple_extract.system")
    return f"""{sys_prompt}

段落：
```
{paragraph}
```

实体列表：
```
{entities}
```"""
import os
import yaml
from typing import Dict, List, Any, Optional
from src.common.logger import get_logger

logger = get_logger("梦境配置加载")

_DREAM_CONFIG_CACHE: Dict[str, Any] = {}


def _get_dream_config_path(relative_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)


def load_dream_yaml_config(
    config_name: str, relative_path: str, use_cache: bool = True
) -> Dict[str, Any]:
    if use_cache and config_name in _DREAM_CONFIG_CACHE:
        return _DREAM_CONFIG_CACHE[config_name]
    config_path = _get_dream_config_path(relative_path)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if use_cache:
            _DREAM_CONFIG_CACHE[config_name] = config
        return config
    except FileNotFoundError:
        logger.warning(f"配置文件不存在: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"配置文件解析错误: {config_path} | {e}")
        return {}


def get_dream_config() -> Dict[str, Any]:
    return load_dream_yaml_config("dream_config", "configs/dream_config.yaml")


def get_maintenance_config() -> Dict[str, Any]:
    config = get_dream_config()
    return config.get(
        "maintenance",
        {
            "max_iterations": 10,
            "cooldown_seconds": 3600,
            "batch_size": 5,
            "timeout_seconds": 300,
        },
    )


def get_max_iterations() -> int:
    return get_maintenance_config().get("max_iterations", 10)


def get_cooldown_seconds() -> int:
    return get_maintenance_config().get("cooldown_seconds", 3600)


def get_tool_definitions() -> Dict[str, Any]:
    config = get_dream_config()
    return config.get("tools", {})


def get_goals() -> List[str]:
    config = get_dream_config()
    return config.get("goals", [])


def get_merge_rules() -> List[str]:
    config = get_dream_config()
    return config.get("merge_rules", [])


def get_dream_prompts() -> Dict[str, Any]:
    return load_dream_yaml_config(
        "dream_prompts", "configs/dream_prompts.yaml"
    )


def get_prompt_template(prompt_name: str) -> str:
    prompts = get_dream_prompts()
    template = prompts.get("dream_summary_generate", {}).get("template", "")
    if prompt_name == "summary_generate":
        template = prompts.get("dream_summary_generate", {}).get(
            "template", ""
        )
    elif prompt_name == "iteration_prompt":
        template = prompts.get("dream_iteration_prompt", {}).get(
            "template", ""
        )
    elif prompt_name == "quality_analysis":
        template = prompts.get("dream_quality_analysis", {}).get(
            "template", ""
        )
    return template


def get_output_instructions(instruction_type: str) -> List[str]:
    prompts = get_dream_prompts()
    output_inst = prompts.get("output_instructions", {})
    return output_inst.get(instruction_type, [])


def get_instruction_template(instruction_name: str) -> str:
    prompts = get_dream_prompts()
    templates = prompts.get("instruction_templates", {})
    return templates.get(instruction_name, "")


def clear_dream_config_cache(config_name: Optional[str] = None):
    global _DREAM_CONFIG_CACHE
    if config_name:
        _DREAM_CONFIG_CACHE.pop(config_name, None)
    else:
        _DREAM_CONFIG_CACHE.clear()

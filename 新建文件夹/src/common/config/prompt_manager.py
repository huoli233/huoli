import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
from src.common.logger import get_logger

logger = get_logger("提示词管理")


class PromptManager:
    """提示词管理器"""

    def __init__(self, templates_dir: str = None):
        if templates_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            templates_dir = str(
                project_root / "src" / "chat" / "prompts" / "templates"
            )
        self._templates_dir = templates_dir
        self._cache: Dict[str, str] = {}
        self._toml_cache: Dict[str, Dict[str, Any]] = {}

    def _resolve_path(self, relative_path: str) -> str:
        return os.path.join(self._templates_dir, relative_path)

    def load_prompt(self, prompt_path: str, use_cache: bool = True) -> str:
        if use_cache and prompt_path in self._cache:
            return self._cache[prompt_path]
        full_path = self._resolve_path(prompt_path)
        if not os.path.exists(full_path):
            logger.warning(f"提示词文件不存在: {full_path}")
            return ""
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            if use_cache:
                self._cache[prompt_path] = content
            return content
        except Exception as e:
            logger.error(f"加载提示词失败: {full_path} | {e}")
            return ""

    def load_toml_prompt(
        self, prompt_path: str, use_cache: bool = True
    ) -> Dict[str, Any]:
        if use_cache and prompt_path in self._toml_cache:
            return self._toml_cache[prompt_path]
        full_path = self._resolve_path(prompt_path)
        if not os.path.exists(full_path):
            logger.warning(f"TOML提示词文件不存在: {full_path}")
            return {}
        try:
            import tomllib

            with open(full_path, "rb") as f:
                data = tomllib.load(f)
            if use_cache:
                self._toml_cache[prompt_path] = data
            return data
        except ImportError:
            try:
                import toml

                with open(full_path, "r", encoding="utf-8") as f:
                    data = toml.load(f)
                if use_cache:
                    self._toml_cache[prompt_path] = data
                return data
            except ImportError:
                logger.warning(f"缺少 toml 库，无法加载: {full_path}")
                return {}
        except Exception as e:
            logger.error(f"加载TOML提示词失败: {full_path} | {e}")
            return {}

    def render(self, template: str, variables: Dict[str, Any] = None) -> str:
        if not variables:
            return template
        result = template
        for key, value in variables.items():
            placeholder = "{{" + key + "}}"
            result = result.replace(placeholder, str(value))
            placeholder2 = "{" + key + "}"
            result = result.replace(placeholder2, str(value))
        return result

    def render_from_file(
        self,
        prompt_path: str,
        variables: Dict[str, Any] = None,
        use_cache: bool = True,
    ) -> str:
        template = self.load_prompt(prompt_path, use_cache)
        return self.render(template, variables)

    def get_prompt_for_type(
        self, fine_type: str, default_template: str = "behavior.txt"
    ) -> str:
        from src.common.config.config_engine import get_default_config_engine

        engine = get_default_config_engine()
        type_cfg = engine.get("types", fine_type, {})
        if isinstance(type_cfg, dict):
            template_name = type_cfg.get("prompt_template", default_template)
        else:
            template_name = default_template
        return self.load_prompt(f"understanding/{template_name}")

    def get_prompt_for_category(self, category: str) -> str:
        category_prompt_map = {
            "friendly": "understanding/behavior.txt",
            "neutral": "understanding/behavior.txt",
            "unfriendly": "understanding/behavior.txt",
            "hostile": "understanding/harassment.txt",
            "harassing": "understanding/harassment.txt",
        }
        prompt_path = category_prompt_map.get(
            category, "understanding/behavior.txt"
        )
        return self.load_prompt(prompt_path)

    def clear_cache(self) -> None:
        self._cache.clear()
        self._toml_cache.clear()
        logger.debug("提示词缓存已清空")

    def list_available_prompts(self) -> Dict[str, str]:
        result = {}
        if not os.path.exists(self._templates_dir):
            return result
        for root, dirs, files in os.walk(self._templates_dir):
            for file in files:
                if file.endswith((".txt", ".toml", ".md")):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self._templates_dir)
                    result[rel_path] = full_path
        return result


_DEFAULT_PROMPT_MANAGER: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    global _DEFAULT_PROMPT_MANAGER
    if _DEFAULT_PROMPT_MANAGER is None:
        _DEFAULT_PROMPT_MANAGER = PromptManager()
    return _DEFAULT_PROMPT_MANAGER


def load_prompt(prompt_path: str, use_cache: bool = True) -> str:
    manager = get_prompt_manager()
    return manager.load_prompt(prompt_path, use_cache)


def render_prompt(template: str, variables: Dict[str, Any] = None) -> str:
    manager = get_prompt_manager()
    return manager.render(template, variables)


def render_from_file(
    prompt_path: str, variables: Dict[str, Any] = None, use_cache: bool = True
) -> str:
    manager = get_prompt_manager()
    return manager.render_from_file(prompt_path, variables, use_cache)

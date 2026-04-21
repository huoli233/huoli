import os
import tomlkit
from typing import Dict, Optional, Any
from src.common.logger import get_logger

logger = get_logger("prompt_loader")

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config", "prompts")


class PromptCategory:
    SYSTEM = "system"
    REPLY = "reply"
    HEARTFLOW = "heartflow"
    PROACTIVE = "proactive"
    PSYCHOLOGY = "psychology"
    RECALL = "recall"
    LEARNING = "learning"
    BRAIN = "brain"
    TOOLS = "tools"
    MODULE = "modules"


class PromptStore:
    _cache: Dict[str, Dict[str, Any]] = {}
    _file_mtimes: Dict[str, float] = {}

    @classmethod
    def _resolve_path(cls, category: str, filename: str) -> str:
        if not filename.endswith(".toml"):
            filename = f"{filename}.toml"
        return os.path.join(_PROMPTS_DIR, category, filename)

    @classmethod
    def _load_file(cls, filepath: str) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            logger.warning(f"提示词文件不存在: {filepath}")
            return {}
        try:
            mtime = os.path.getmtime(filepath)
            if filepath in cls._cache and cls._file_mtimes.get(filepath) == mtime:
                return cls._cache[filepath]
            with open(filepath, "r", encoding="utf-8") as f:
                data = tomlkit.load(f)
            cls._cache[filepath] = dict(data)
            cls._file_mtimes[filepath] = mtime
            return cls._cache[filepath]
        except Exception as e:
            logger.error(f"加载提示词文件失败 {filepath}: {e}")
            return {}

    @classmethod
    def get(cls, category: str, filename: str, key: str, **kwargs) -> str:
        filepath = cls._resolve_path(category, filename)
        data = cls._load_file(filepath)
        template = _deep_get(data, key)
        if template is None:
            logger.warning(f"提示词键不存在: {category}/{filename}#{key}")
            return ""
        if not isinstance(template, str):
            logger.warning(f"提示词值非字符串: {category}/{filename}#{key}")
            return str(template)
        if kwargs:
            try:
                return template.format(**kwargs)
            except KeyError as e:
                logger.debug(f"提示词格式化缺少变量 {category}/{filename}#{key}: {e}")
                return template
        return template

    @classmethod
    def get_raw(cls, category: str, filename: str, key: str) -> str:
        filepath = cls._resolve_path(category, filename)
        data = cls._load_file(filepath)
        template = _deep_get(data, key)
        if template is None:
            return ""
        return str(template)

    @classmethod
    def get_section(cls, category: str, filename: str, section: str) -> Dict[str, Any]:
        filepath = cls._resolve_path(category, filename)
        data = cls._load_file(filepath)
        result = data.get(section)
        if result is None:
            return {}
        if isinstance(result, dict):
            return dict(result)
        return {}

    @classmethod
    def get_list(cls, category: str, filename: str, key: str) -> list:
        filepath = cls._resolve_path(category, filename)
        data = cls._load_file(filepath)
        result = _deep_get(data, key)
        if result is None:
            return []
        if isinstance(result, list):
            return list(result)
        return []

    @classmethod
    def reload(cls, category: Optional[str] = None, filename: Optional[str] = None):
        if category and filename:
            filepath = cls._resolve_path(category, filename)
            cls._cache.pop(filepath, None)
            cls._file_mtimes.pop(filepath, None)
            logger.info(f"重新加载提示词: {category}/{filename}")
        elif category:
            prefix = os.path.join(_PROMPTS_DIR, category)
            keys_to_remove = [k for k in cls._cache if k.startswith(prefix)]
            for k in keys_to_remove:
                cls._cache.pop(k, None)
                cls._file_mtimes.pop(k, None)
            logger.info(f"重新加载分类: {category}")
        else:
            cls._cache.clear()
            cls._file_mtimes.clear()
            logger.info("重新加载全部提示词")

    @classmethod
    def exists(cls, category: str, filename: str, key: str) -> bool:
        filepath = cls._resolve_path(category, filename)
        data = cls._load_file(filepath)
        return _deep_get(data, key) is not None


def _deep_get(data: Dict, dotted_key: str) -> Any:
    keys = dotted_key.split(".")
    current = data
    for k in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(k)
        if current is None:
            return None
    return current


def get_prompt(category: str, filename: str, key: str, **kwargs) -> str:
    return PromptStore.get(category, filename, key, **kwargs)


def get_prompt_raw(category: str, filename: str, key: str) -> str:
    return PromptStore.get_raw(category, filename, key)


def get_prompt_section(category: str, filename: str, section: str) -> Dict[str, Any]:
    return PromptStore.get_section(category, filename, section)


def get_prompt_list(category: str, filename: str, key: str) -> list:
    return PromptStore.get_list(category, filename, key)


def reload_prompts(category: Optional[str] = None, filename: Optional[str] = None):
    PromptStore.reload(category, filename)

import os
import tomlkit
import shutil

from datetime import datetime
from tomlkit import TOMLDocument
from tomlkit.items import Table, KeyType
from dataclasses import field, dataclass
from rich.traceback import install
from typing import Any, List, Optional

from src.common.logger import get_logger
from src.common.toml_utils import format_toml_string
from src.config.config_base import ConfigBase
from src.config.official_configs import (
    BotConfig,
    PersonalityConfig,
    ExpressionConfig,
    ChatConfig,
    EmojiConfig,
    ChineseTypoConfig,
    ResponsePostProcessConfig,
    ResponseSplitterConfig,
    TelemetryConfig,
    ExperimentalConfig,
    MessageReceiveConfig,
    HuoliMessageConfig,
    LPMMKnowledgeConfig,
    RelationshipConfig,
    ToolConfig,
    VoiceConfig,
    MemoryConfig,
    DebugConfig,
    WebUIConfig,
)

from .api_ada_configs import (
    ModelTaskConfig,
    ModelInfo,
    APIProvider,
)


install(extra_lines=3)


# 配置主程序日志格式
logger = get_logger("config")

# 获取当前文件所在目录的父目录的父目录（即项目根目录）
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "template")

# 考虑到配置文件中的版本字段不会自动更新，所以采用硬编码
# 对该字段的更新，请严格参照语义化版本规范：https://semver.org/lang/zh-CN/
MMC_VERSION = "0.13.0-sakana.1"
_SYNC_ON_IMPORT_ENV = "HUOLI_SYNC_CONFIG_ON_IMPORT"
_SENSITIVE_CONFIG_KEY_PARTS = ("api_key", "token", "secret", "password", "authorization")
_LEGACY_MODEL_CONFIG_PROVIDER_NAMES = {"AggAPI"}
_LEGACY_MODEL_CONFIG_MODEL_NAMES = {
    "gemma-3-1b",
    "gemma-3-12b",
    "gemma-3-27b",
    "gemma-4-26b",
    "gemma-4-31b",
}


def _legacy_message_section_name() -> str:
    """旧消息配置节名，仅用于历史配置迁移兼容。"""
    return "mai" + "m_message"


def get_key_comment(toml_table, key):
    # 获取key的注释（如果有）
    if hasattr(toml_table, "trivia") and hasattr(toml_table.trivia, "comment"):
        return toml_table.trivia.comment
    if hasattr(toml_table, "value") and isinstance(toml_table.value, dict):
        item = toml_table.value.get(key)
        if item is not None and hasattr(item, "trivia"):
            return item.trivia.comment
    if hasattr(toml_table, "keys"):
        for k in toml_table.keys():
            if isinstance(k, KeyType) and k.key == key:  # type: ignore
                return k.trivia.comment  # type: ignore
    return None


def compare_dicts(new, old, path=None, logs=None):
    # 递归比较两个dict，找出新增和删减项，收集注释
    if path is None:
        path = []
    if logs is None:
        logs = []
    # 新增项
    for key in new:
        if key == "version":
            continue
        if key not in old:
            comment = get_key_comment(new, key)
            logs.append(
                f"新增: {'.'.join(path + [str(key)])}  注释: {comment or '无'}"
            )
        elif isinstance(new[key], (dict, Table)) and isinstance(
            old.get(key), (dict, Table)
        ):
            compare_dicts(new[key], old[key], path + [str(key)], logs)
    # 删减项
    for key in old:
        if key == "version":
            continue
        if key not in new:
            comment = get_key_comment(old, key)
            logs.append(
                f"删减: {'.'.join(path + [str(key)])}  注释: {comment or '无'}"
            )
    return logs


def get_value_by_path(d, path):
    for k in path:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d


def _plain_config_value(value: Any) -> Any:
    """转换为普通 Python 值，便于比较配置内容而不受 tomlkit trivia 影响。"""
    if hasattr(value, "unwrap"):
        try:
            return value.unwrap()
        except Exception:
            pass
    if isinstance(value, dict) or hasattr(value, "items"):
        try:
            return {str(k): _plain_config_value(v) for k, v in value.items()}
        except Exception:
            return value
    if isinstance(value, list) or isinstance(value, tuple):
        return [_plain_config_value(item) for item in value]
    return value


def _is_sensitive_config_path(path: List[str] | tuple[str, ...]) -> bool:
    normalized = ".".join(str(part).lower() for part in path)
    return any(part in normalized for part in _SENSITIVE_CONFIG_KEY_PARTS)


def _safe_config_log_value(value: Any, path: List[str] | tuple[str, ...] | None = None) -> Any:
    """日志输出配置值时隐藏密钥，避免启动日志泄漏真实或模板密钥。"""
    path = list(path or [])
    if _is_sensitive_config_path(path):
        return "<已隐藏>"
    plain = _plain_config_value(value)
    if isinstance(plain, dict):
        return {k: _safe_config_log_value(v, path + [str(k)]) for k, v in plain.items()}
    if isinstance(plain, list):
        return [_safe_config_log_value(item, path) for item in plain]
    return plain


def _model_config_compare_has_legacy_defaults(compare_config: TOMLDocument | dict, new_config: TOMLDocument | dict) -> bool:
    """识别仍停留在 AggAPI/Gemma 旧默认模型的 compare 基线。"""
    compare_plain = _plain_config_value(compare_config) or {}
    new_plain = _plain_config_value(new_config) or {}
    compare_providers = {
        str(item.get("name", ""))
        for item in compare_plain.get("api_providers", [])
        if isinstance(item, dict)
    }
    new_providers = {
        str(item.get("name", ""))
        for item in new_plain.get("api_providers", [])
        if isinstance(item, dict)
    }
    compare_models = {
        str(item.get("name", ""))
        for item in compare_plain.get("models", [])
        if isinstance(item, dict)
    }
    new_models = {
        str(item.get("name", ""))
        for item in new_plain.get("models", [])
        if isinstance(item, dict)
    }
    legacy_provider_left = bool((compare_providers - new_providers) & _LEGACY_MODEL_CONFIG_PROVIDER_NAMES)
    legacy_model_left = bool((compare_models - new_models) & _LEGACY_MODEL_CONFIG_MODEL_NAMES)
    return legacy_provider_left or legacy_model_left


def _remove_legacy_default_array_items(
    config_doc: TOMLDocument | dict,
    compare_config: TOMLDocument | dict,
    new_config: TOMLDocument | dict,
    key: str,
    identity_key: str = "name",
) -> int:
    """只移除仍等于旧模板默认值的数组项，保留用户改过的 provider/model。"""
    target_items = config_doc.get(key)
    if not isinstance(target_items, list):
        return 0

    compare_items = [
        item
        for item in _plain_config_value(compare_config.get(key, []))
        if isinstance(item, dict) and item.get(identity_key)
    ]
    new_names = {
        str(item.get(identity_key, ""))
        for item in _plain_config_value(new_config.get(key, []))
        if isinstance(item, dict) and item.get(identity_key)
    }
    stale_defaults = {
        str(item.get(identity_key, "")): item
        for item in compare_items
        if str(item.get(identity_key, "")) not in new_names
    }
    if not stale_defaults:
        return 0

    removed = 0
    for index in range(len(target_items) - 1, -1, -1):
        item_plain = _plain_config_value(target_items[index])
        if not isinstance(item_plain, dict):
            continue
        item_name = str(item_plain.get(identity_key, ""))
        if item_name in stale_defaults and item_plain == stale_defaults[item_name]:
            del target_items[index]
            removed += 1
    return removed


def _apply_default_value_changes(
    old_config: TOMLDocument | dict,
    changes: List[tuple[List[str], Any, Any]],
    config_name: str,
    *,
    log_changes: bool = True,
) -> bool:
    config_updated = False
    for path, old_default, new_default in changes:
        old_value = get_value_by_path(old_config, path)
        if old_value == old_default:
            set_value_by_path(old_config, path, new_default)
            if log_changes:
                logger.info(
                    f"已自动将{config_name}配置 {'.'.join(path)} 的值从旧默认值 "
                    f"{_safe_config_log_value(old_default, path)} 更新为新默认值 "
                    f"{_safe_config_log_value(new_default, path)}"
                )
            config_updated = True
    return config_updated


def set_value_by_path(d, path, value):
    """设置嵌套字典中指定路径的值"""
    for k in path[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]

    # 使用 tomlkit.item 来保持 TOML 格式
    try:
        d[path[-1]] = tomlkit.item(value)
    except (TypeError, ValueError):
        # 如果转换失败，直接赋值
        d[path[-1]] = value


def compare_default_values(new, old, path=None, logs=None, changes=None):
    # 递归比较两个dict，找出默认值变化项
    if path is None:
        path = []
    if logs is None:
        logs = []
    if changes is None:
        changes = []
    for key in new:
        if key == "version":
            continue
        if key in old:
            if isinstance(new[key], (dict, Table)) and isinstance(
                old[key], (dict, Table)
            ):
                compare_default_values(
                    new[key], old[key], path + [str(key)], logs, changes
                )
            elif new[key] != old[key]:
                logs.append(
                    f"默认值变化: {'.'.join(path + [str(key)])}  "
                    f"旧默认值: {_safe_config_log_value(old[key], path + [str(key)])}  "
                    f"新默认值: {_safe_config_log_value(new[key], path + [str(key)])}"
                )
                changes.append((path + [str(key)], old[key], new[key]))
    return logs, changes


def _get_version_from_toml(toml_path) -> Optional[str]:
    """从TOML文件中获取版本号"""
    if not os.path.exists(toml_path):
        return None
    with open(toml_path, "r", encoding="utf-8") as f:
        doc = tomlkit.load(f)
    if "inner" in doc and "version" in doc["inner"]:  # type: ignore
        return doc["inner"]["version"]  # type: ignore
    return None


def _version_tuple(v):
    """将版本字符串转换为元组以便比较"""
    if v is None:
        return (0,)
    return tuple(
        int(x) if x.isdigit() else 0
        for x in str(v).replace("v", "").split("-")[0].split(".")
    )


def _update_dict(
    target: TOMLDocument | dict | Table, source: TOMLDocument | dict
):
    """
    将source字典的值更新到target字典中（如果target中存在相同的键）
    """
    for key, value in source.items():
        # 跳过version字段的更新
        if key == "version":
            continue
        if key in target:
            target_value = target[key]
            if isinstance(value, dict) and isinstance(
                target_value, (dict, Table)
            ):
                _update_dict(target_value, value)
            else:
                try:
                    # 统一使用 tomlkit.item 来保持原生类型与转义，不对列表做字符串化处理
                    target[key] = tomlkit.item(value)
                except (TypeError, ValueError):
                    # 如果转换失败，直接赋值
                    target[key] = value


def _update_config_generic(config_name: str, template_name: str):
    """
    通用的配置文件更新函数

    Args:
        config_name: 配置文件名（不含扩展名），如 'bot_config' 或 'model_config'
        template_name: 模板文件名（不含扩展名），如 'bot_config_template' 或 'model_config_template'
    """
    # 获取根目录路径
    old_config_dir = os.path.join(CONFIG_DIR, "old")
    compare_dir = os.path.join(TEMPLATE_DIR, "compare")

    # 定义文件路径
    template_path = os.path.join(TEMPLATE_DIR, f"{template_name}.toml")
    old_config_path = os.path.join(CONFIG_DIR, f"{config_name}.toml")
    new_config_path = os.path.join(CONFIG_DIR, f"{config_name}.toml")
    compare_path = os.path.join(compare_dir, f"{template_name}.toml")

    # 创建compare目录（如果不存在）
    os.makedirs(compare_dir, exist_ok=True)

    template_version = _get_version_from_toml(template_path)
    compare_version = _get_version_from_toml(compare_path)

    # 检查配置文件是否存在
    if not os.path.exists(old_config_path):
        logger.info(f"{config_name}.toml配置文件不存在，从模板创建新配置")
        os.makedirs(CONFIG_DIR, exist_ok=True)  # 创建文件夹
        shutil.copy2(template_path, old_config_path)  # 复制模板文件
        logger.info(
            f"已创建新{config_name}配置文件，请填写后重新运行: {old_config_path}"
        )
        return

    compare_config = None
    new_config = None
    old_config = None

    # 先读取 compare 下的模板（如果有），用于默认值变动检测
    if os.path.exists(compare_path):
        with open(compare_path, "r", encoding="utf-8") as f:
            compare_config = tomlkit.load(f)

    # 读取当前模板
    with open(template_path, "r", encoding="utf-8") as f:
        new_config = tomlkit.load(f)

    # 检查默认值变化并处理（只有 compare_config 存在时才做）
    if compare_config:
        # 读取旧配置
        with open(old_config_path, "r", encoding="utf-8") as f:
            old_config = tomlkit.load(f)

        skip_default_compare_log = False
        if config_name == "model_config" and _model_config_compare_has_legacy_defaults(compare_config, new_config):
            _logs, legacy_changes = compare_default_values(new_config, compare_config)
            config_updated = _apply_default_value_changes(
                old_config,
                legacy_changes,
                config_name,
                log_changes=False,
            )
            removed_defaults = 0
            removed_defaults += _remove_legacy_default_array_items(
                old_config,
                compare_config,
                new_config,
                "api_providers",
            )
            removed_defaults += _remove_legacy_default_array_items(
                old_config,
                compare_config,
                new_config,
                "models",
            )
            config_updated = config_updated or removed_defaults > 0
            if config_updated:
                with open(old_config_path, "w", encoding="utf-8") as f:
                    f.write(format_toml_string(old_config))
                logger.info(
                    f"已清理{config_name}中的旧模板默认项: {removed_defaults} 项；"
                    "用户改过的 provider/model 已保留"
                )
            shutil.copy2(template_path, compare_path)
            compare_config = new_config
            skip_default_compare_log = True
            logger.info(
                f"检测到{config_name}模板对比基线仍是 AggAPI/Gemma 旧默认配置，"
                "已同步为当前模板，后续启动不再重复输出旧默认值差异"
            )

        logs, changes = compare_default_values(new_config, compare_config)
        if logs and not skip_default_compare_log:
            logger.info(f"检测到{config_name}模板默认值变动如下：")
            for log in logs:
                logger.info(log)
            # 检查旧配置是否等于旧默认值，如果是则更新为新默认值
            config_updated = _apply_default_value_changes(old_config, changes, config_name)

            # 如果配置有更新，立即保存到文件
            if config_updated:
                with open(old_config_path, "w", encoding="utf-8") as f:
                    f.write(format_toml_string(old_config))
                logger.info(f"已保存更新后的{config_name}配置文件")
        elif not skip_default_compare_log:
            logger.info(f"未检测到{config_name}模板默认值变动")

    # 检查 compare 下没有模板，或新模板版本更高，则复制
    if not os.path.exists(compare_path):
        shutil.copy2(template_path, compare_path)
        logger.info(f"已将{config_name}模板文件复制到: {compare_path}")
    elif _version_tuple(template_version) > _version_tuple(compare_version):
        shutil.copy2(template_path, compare_path)
        logger.info(
            f"{config_name}模板版本较新，已替换compare下的模板: {compare_path}"
        )
    else:
        logger.debug(
            f"compare下的{config_name}模板版本不低于当前模板，无需替换: {compare_path}"
        )

    # 读取旧配置文件和模板文件（如果前面没读过 old_config，这里再读一次）
    if old_config is None:
        with open(old_config_path, "r", encoding="utf-8") as f:
            old_config = tomlkit.load(f)
    # new_config 已经读取

    # 检查version是否相同
    if old_config and "inner" in old_config and "inner" in new_config:
        old_version = old_config["inner"].get("version")  # type: ignore
        new_version = new_config["inner"].get("version")  # type: ignore
        if old_version and new_version and old_version == new_version:
            logger.info(
                f"检测到{config_name}配置文件版本号相同 (v{old_version})，跳过更新"
            )
            return
        else:
            logger.info(
                f"\n----------------------------------------\n检测到{config_name}版本号不同: 旧版本 v{old_version} -> 新版本 v{new_version}\n----------------------------------------"
            )
    else:
        logger.info(
            f"已有{config_name}配置文件未检测到版本号，可能是旧版本。将进行更新"
        )

    # 创建old目录（如果不存在）
    os.makedirs(old_config_dir, exist_ok=True)  # 生成带时间戳的新文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    old_backup_path = os.path.join(
        old_config_dir, f"{config_name}_{timestamp}.toml"
    )

    # 移动旧配置文件到old目录
    shutil.move(old_config_path, old_backup_path)
    logger.info(f"已备份旧{config_name}配置文件到: {old_backup_path}")

    # 复制模板文件到配置目录
    shutil.copy2(template_path, new_config_path)
    logger.info(f"已创建新{config_name}配置文件: {new_config_path}")

    # 输出新增和删减项及注释
    if old_config:
        logger.info(
            f"{config_name}配置项变动如下：\n----------------------------------------"
        )
        if logs := compare_dicts(new_config, old_config):
            for log in logs:
                logger.info(log)
        else:
            logger.info("无新增或删减项")

    # 将旧配置的值更新到新配置中
    logger.info(f"开始合并{config_name}新旧配置...")
    _update_dict(new_config, old_config)

    # 保存更新后的配置（保留注释和格式，数组多行格式化）
    with open(new_config_path, "w", encoding="utf-8") as f:
        f.write(format_toml_string(new_config))
    logger.info(
        f"{config_name}配置文件更新完成，建议检查新配置文件中的内容，以免丢失重要信息"
    )


def update_config():
    """更新bot_config.toml配置文件"""
    _update_config_generic("bot_config", "bot_config_template")


def update_model_config():
    """更新model_config.toml配置文件"""
    _update_config_generic("model_config", "model_config_template")


def ensure_config_files_exist() -> None:
    """仅在配置文件缺失时从模板生成，避免普通导入产生文件更新副作用。"""
    bot_config_path = os.path.join(CONFIG_DIR, "bot_config.toml")
    model_config_path = os.path.join(CONFIG_DIR, "model_config.toml")
    if not os.path.exists(bot_config_path):
        update_config()
    if not os.path.exists(model_config_path):
        update_model_config()


def should_sync_config_on_import() -> bool:
    """是否允许在模块导入时执行模板同步。默认关闭，仅在主程序启动路径显式开启。"""
    return os.environ.get(_SYNC_ON_IMPORT_ENV, "").strip() == "1"


@dataclass
class Config(ConfigBase):
    """总配置类"""

    MMC_VERSION: str = field(
        default=MMC_VERSION, repr=False, init=False
    )  # 硬编码的版本信息

    bot: BotConfig
    personality: PersonalityConfig
    relationship: RelationshipConfig
    chat: ChatConfig
    message_receive: MessageReceiveConfig
    emoji: EmojiConfig
    expression: ExpressionConfig
    chinese_typo: ChineseTypoConfig
    response_post_process: ResponsePostProcessConfig
    response_splitter: ResponseSplitterConfig
    telemetry: TelemetryConfig
    webui: WebUIConfig
    experimental: ExperimentalConfig
    huoli_message: HuoliMessageConfig
    lpmm_knowledge: LPMMKnowledgeConfig
    tool: ToolConfig
    memory: MemoryConfig
    debug: DebugConfig
    voice: VoiceConfig


@dataclass
class APIAdapterConfig(ConfigBase):
    """API Adapter配置类"""

    models: List[ModelInfo]
    """模型列表"""

    model_task_config: ModelTaskConfig
    """模型任务配置"""

    api_providers: List[APIProvider] = field(default_factory=list)
    """API提供商列表"""

    def __post_init__(self):
        if not self.models:
            raise ValueError(
                "模型列表不能为空，请在配置中设置有效的模型列表。"
            )
        if not self.api_providers:
            raise ValueError(
                "API提供商列表不能为空，请在配置中设置有效的API提供商列表。"
            )

        # 检查API提供商名称是否重复
        provider_names = [provider.name for provider in self.api_providers]
        if len(provider_names) != len(set(provider_names)):
            raise ValueError("API提供商名称存在重复，请检查配置文件。")

        # 检查模型名称是否重复
        model_names = [model.name for model in self.models]
        if len(model_names) != len(set(model_names)):
            raise ValueError("模型名称存在重复，请检查配置文件。")

        self.api_providers_dict = {
            provider.name: provider for provider in self.api_providers
        }
        self.models_dict = {model.name: model for model in self.models}

        for model in self.models:
            if not model.model_identifier:
                raise ValueError(
                    f"模型 '{model.name}' 的 model_identifier 不能为空"
                )
            if (
                not model.api_provider
                or model.api_provider not in self.api_providers_dict
            ):
                raise ValueError(
                    f"模型 '{
                        model.name}' 的 api_provider '{
                        model.api_provider}' 不存在"
                )

    def get_model_info(self, model_name: str) -> ModelInfo:
        """根据模型名称获取模型信息"""
        if not model_name:
            raise ValueError("模型名称不能为空")
        if model_name not in self.models_dict:
            raise KeyError(f"模型 '{model_name}' 不存在")
        return self.models_dict[model_name]

    def get_provider(self, provider_name: str) -> APIProvider:
        """根据提供商名称获取API提供商信息"""
        if not provider_name:
            raise ValueError("API提供商名称不能为空")
        if provider_name not in self.api_providers_dict:
            raise KeyError(f"API提供商 '{provider_name}' 不存在")
        return self.api_providers_dict[provider_name]


def load_config(config_path: str) -> Config:
    """
    加载配置文件
    Args:
        config_path: 配置文件路径
    Returns:
        Config对象
    """
    # 读取配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = tomlkit.load(f)
    legacy_message_section = _legacy_message_section_name()
    if "huoli_message" not in config_data and legacy_message_section in config_data:
        config_data["huoli_message"] = config_data[legacy_message_section]

    # 创建Config对象
    try:
        config = Config.from_dict(config_data)
        setattr(config, legacy_message_section, config.huoli_message)
        return config
    except Exception as e:
        logger.critical("配置文件解析失败")
        raise e


def api_ada_load_config(config_path: str) -> APIAdapterConfig:
    """
    加载API适配器配置文件
    Args:
        config_path: 配置文件路径
    Returns:
        APIAdapterConfig对象
    """
    # 读取配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = tomlkit.load(f)

    # 创建APIAdapterConfig对象
    try:
        return APIAdapterConfig.from_dict(config_data)
    except Exception as e:
        logger.critical("API适配器配置文件解析失败")
        raise e


# 获取配置文件路径
logger.info(f"HuoLiCore当前版本: {MMC_VERSION}")
if should_sync_config_on_import():
    update_config()
    update_model_config()
else:
    ensure_config_files_exist()

logger.info("正在检查配置文件...")
global_config = load_config(
    config_path=os.path.join(CONFIG_DIR, "bot_config.toml")
)
model_config = api_ada_load_config(
    config_path=os.path.join(CONFIG_DIR, "model_config.toml")
)
logger.info("配置文件加载完成。")

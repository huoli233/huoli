from typing import List, Tuple, Type

from src.plugin_system.base.base_plugin import BasePlugin
from src.plugin_system.base.component_types import ComponentInfo
from src.plugin_system.base.config_types import ConfigField
from src.plugin_system.apis.plugin_register_api import register_plugin
from src.common.logger import get_logger
from src.plugins.built_in.image_analyze_plugin.analyze_image import AnalyzeImageAction

logger = get_logger("image_analyze_plugin")


@register_plugin
class ImageAnalyzePlugin(BasePlugin):
    plugin_name: str = "image_analyze"
    enable_plugin: bool = True
    dependencies: list[str] = []
    python_dependencies: list[str] = []
    config_file_name: str = "config.toml"

    config_section_descriptions = {
        "plugin": "插件启用配置",
        "components": "图片分析组件配置",
    }

    config_schema: dict = {
        "plugin": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用插件"),
            "config_version": ConfigField(type=str, default="1.0.0", description="配置文件版本"),
        },
        "components": {
            "enable_analyze_image": ConfigField(type=bool, default=True, description="是否启用图片分析动作"),
        },
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        components = []
        if self.get_config("components.enable_analyze_image", True):
            components.append((AnalyzeImageAction.get_action_info(), AnalyzeImageAction))
        return components

import re
from typing import List, Tuple, Type

from src.common.logger import get_logger
from src.plugin_system.apis.plugin_register_api import register_plugin
from src.plugin_system.base.base_action import ActionActivationType, BaseAction
from src.plugin_system.base.base_plugin import BasePlugin
from src.plugin_system.base.component_types import ComponentInfo
from src.plugin_system.base.config_types import ConfigField

logger = get_logger("tts")


class TTSAction(BaseAction):
    """TTS语音转换动作处理类"""

    activation_type = ActionActivationType.KEYWORD
    activation_keywords = [
        "语音",
        "tts",
        "播报",
        "读出来",
        "语音播放",
        "听",
        "朗读",
    ]
    keyword_case_sensitive = False
    parallel_action = False
    action_name = "tts_action"
    action_description = "将文本转换为语音进行播放，适用于需要语音输出的场景"
    action_parameters = {
        "voice_text": "你想用语音表达的内容，这段内容将会以语音形式发出",
    }
    action_require = [
        "当需要发送语音信息时使用",
        "当用户明确要求使用语音功能时使用",
        "当表达内容更适合用语音而不是文字传达时使用",
        "当用户想听到语音回答而非阅读文本时使用",
    ]
    associated_types = ["tts_text"]

    async def execute(self) -> Tuple[bool, str]:
        """处理TTS文本转语音动作"""
        logger.info(f"{self.log_prefix} 执行TTS动作: {self.reasoning}")
        text = self.action_data.get("voice_text")
        if not text:
            logger.error(f"{self.log_prefix} 执行TTS动作时未提供文本内容")
            return False, "执行TTS动作失败：未提供文本内容"
        processed_text = self._process_text_for_tts(text)
        try:
            await self.send_custom(
                message_type="tts_text", content=processed_text
            )
            await self.store_action_info(
                action_build_into_prompt=True,
                action_prompt_display="已经发送了语音消息。",
                action_done=True,
            )
            logger.info(
                f"{self.log_prefix} TTS动作执行成功，文本长度: {len(processed_text)}"
            )
            return True, "TTS动作执行成功"
        except Exception as e:
            logger.error(f"{self.log_prefix} 执行TTS动作时出错: {e}")
            return False, f"执行TTS动作时出错: {e}"

    def _process_text_for_tts(self, text: str) -> str:
        """处理文本使其更适合TTS使用"""
        processed_text = text
        processed_text = re.sub(
            r"([!?,.;:。！？，、；：])\1+", r"\1", processed_text
        )
        if not any(
            processed_text.endswith(end)
            for end in [".", "?", "!", "。", "！", "？"]
        ):
            processed_text = f"{processed_text}。"
        return processed_text


@register_plugin
class TTSPlugin(BasePlugin):
    """TTS插件"""

    plugin_name: str = "tts_plugin"
    enable_plugin: bool = True
    dependencies: list = []
    python_dependencies: list = []
    config_file_name: str = "config.toml"
    config_section_descriptions = {
        "plugin": "插件基本信息配置",
        "components": "组件启用控制",
        "logging": "日志记录相关配置",
    }
    config_schema: dict = {
        "plugin": {
            "name": ConfigField(
                type=str,
                default="tts_plugin",
                description="插件名称",
                required=True,
            ),
            "version": ConfigField(
                type=str, default="0.1.0", description="插件版本号"
            ),
            "enabled": ConfigField(
                type=bool, default=True, description="是否启用插件"
            ),
            "description": ConfigField(
                type=str,
                default="文字转语音插件",
                description="插件描述",
                required=True,
            ),
        },
        "components": {
            "enable_tts": ConfigField(
                type=bool, default=True, description="是否启用TTS Action"
            ),
        },
        "logging": {
            "level": ConfigField(
                type=str,
                default="INFO",
                description="日志记录级别",
                choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            ),
            "prefix": ConfigField(
                type=str, default="[TTS]", description="日志记录前缀"
            ),
        },
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        """返回插件包含的组件列表"""
        enable_tts = self.get_config("components.enable_tts", True)
        components = []
        if enable_tts:
            components.append((TTSAction.get_action_info(), TTSAction))
        return components

from src.config.config import global_config
from src.common.logger import get_logger

logger = get_logger("语音工具")


async def transcribe_voice(voice_base64: str) -> str:
    """将语音转录为文本"""
    if not hasattr(global_config, "voice") or not getattr(
        global_config.voice, "enable_asr", False
    ):
        logger.warning("语音识别未启用")
        return "[语音]"
    try:
        from src.llm_models.utils_model import LLMRequest
        from src.config.config import model_config

        task_config = model_config.model_task_config.voice
        llm_request = LLMRequest(model_set=task_config, request_type="asr")
        text = await llm_request.generate_response_for_voice(
            voice_base64=voice_base64
        )
        return text or "[语音]"
    except Exception as e:
        logger.error(f"语音转录异常: {e}")
        return "[语音]"


async def get_voice_text(voice_base64: str) -> str:
    """获取语音文本（别名）"""
    return await transcribe_voice(voice_base64)


async def get_voice_description(voice_base64: str) -> str:
    """获取语音内容描述"""
    return await transcribe_voice(voice_base64)


def is_voice_message(message_data: dict) -> bool:
    """判断是否为语音消息"""
    msg_type = message_data.get("message_type") or message_data.get("type", "")
    return msg_type in ("voice", "audio", "record")


def extract_voice_base64(message_data: dict) -> str:
    """从消息中提取语音 base64 数据"""
    if "voice" in message_data:
        voice_info = message_data["voice"]
        if isinstance(voice_info, dict):
            return voice_info.get("base64", "") or voice_info.get("url", "")
        return str(voice_info)
    if "audio" in message_data:
        audio_info = message_data["audio"]
        if isinstance(audio_info, dict):
            return audio_info.get("base64", "") or audio_info.get("url", "")
        return str(audio_info)
    for key in ["voice_base64", "audio_base64", "file_base64"]:
        if key in message_data:
            return message_data[key]
    return ""


async def process_voice_in_message(message_data: dict) -> str:
    """处理消息中的语音内容"""
    if not is_voice_message(message_data):
        return ""
    voice_base64 = extract_voice_base64(message_data)
    if not voice_base64:
        return "[语音]"
    return await transcribe_voice(voice_base64)


class VoiceProcessor:
    """语音处理器"""

    def __init__(self):
        self._enabled = False
        self._check_enabled()

    def _check_enabled(self):
        self._enabled = hasattr(global_config, "voice") and getattr(
            global_config.voice, "enable_asr", False
        )

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    async def process(self, voice_base64: str) -> str:
        if not self._enabled:
            return "[语音]"
        return await transcribe_voice(voice_base64)

    def enable(self):
        self._enabled = True
        logger.info("语音处理已启用")

    def disable(self):
        self._enabled = False
        logger.info("语音处理已禁用")


voice_processor = VoiceProcessor()

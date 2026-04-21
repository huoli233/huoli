from typing import Tuple, List

from src.plugin_system.base.base_action import BaseAction
from src.plugin_system.base.component_types import ActionActivationType
from src.common.logger import get_logger
from src.plugin_system.apis import message_api, database_api

logger = get_logger("analyze_image")


class AnalyzeImageAction(BaseAction):
    activation_type = ActionActivationType.ALWAYS
    parallel_action = False

    action_name = "analyze_image"
    action_description = "分析用户发送的图片内容"

    action_parameters = {}

    action_require = [
        "当消息中有[图片]标记且没有描述内容时使用此动作来查看图片",
        "如果图片已经有描述（如[图片：xxx]），说明已识别完毕，直接用reply回复即可，不需要再analyze",
    ]

    associated_types = ["image"]

    async def execute(self) -> Tuple[bool, str]:
        try:
            recent_messages = message_api.get_recent_messages(
                chat_id=self.chat_id, limit=10
            )
            if not recent_messages:
                return False, "没有找到最近的消息"

            target_picid = None
            target_base64 = None
            for msg in reversed(recent_messages):
                seg_list = getattr(msg, 'message_segment', None)
                if seg_list is None:
                    continue
                segments = []
                if hasattr(seg_list, 'data') and isinstance(seg_list.data, list):
                    segments = seg_list.data
                elif isinstance(seg_list, list):
                    segments = seg_list
                for seg in segments:
                    seg_type = None
                    seg_data = None
                    if isinstance(seg, dict):
                        seg_type = seg.get('type')
                        seg_data = seg.get('data')
                    elif hasattr(seg, 'type'):
                        seg_type = getattr(seg, 'type', None)
                        if hasattr(seg_type, 'value'):
                            seg_type = seg_type.value
                        seg_data = getattr(seg, 'data', None)
                    if seg_type == 'image' and seg_data:
                        if isinstance(seg_data, str) and len(seg_data) > 100:
                            target_base64 = seg_data
                            break
                if target_base64:
                    break
                plain_text = getattr(msg, 'processed_plain_text', '') or ''
                if '[picid:' in plain_text:
                    import re
                    match = re.search(r'\[picid:([^\]]+)\]', plain_text)
                    if match:
                        target_picid = match.group(1)
                        break

            if not target_base64 and not target_picid:
                return False, "最近的消息中没有找到图片"

            from src.chat.utils.utils_image import get_image_manager
            image_manager = get_image_manager()

            if target_base64:
                description = await image_manager.get_image_description(target_base64)
                if description:
                    logger.info(f"图片分析完成: {description[:60]}...")
                    await database_api.store_action_info(
                        chat_stream=self.chat_stream,
                        action_build_into_prompt=True,
                        action_prompt_display=f"你查看了图片，看到了: {description}",
                        action_done=True,
                        thinking_id=self.thinking_id,
                        action_data={"image_description": description},
                        action_name="analyze_image",
                    )
                    return True, f"图片分析结果: {description}"
                return False, "图片分析失败"

            if target_picid:
                from src.common.database.database import db
                from src.common.database.database_model import Images
                db.connect(reuse_if_open=True)
                image_record = Images.get_or_none(Images.image_id == target_picid)
                if image_record and image_record.description:
                    desc = image_record.description
                    logger.info(f"从缓存获取图片描述: {desc[:60]}...")
                    await database_api.store_action_info(
                        chat_stream=self.chat_stream,
                        action_build_into_prompt=True,
                        action_prompt_display=f"你查看了图片，看到了: [图片：{desc}]",
                        action_done=True,
                        thinking_id=self.thinking_id,
                        action_data={"image_description": desc},
                        action_name="analyze_image",
                    )
                    return True, f"图片分析结果: [图片：{desc}]"
                elif image_record and image_record.path:
                    import base64 as b64
                    try:
                        with open(image_record.path, 'rb') as f:
                            img_bytes = f.read()
                        img_b64 = b64.b64encode(img_bytes).decode('utf-8')
                        description = await image_manager.get_image_description(img_b64)
                        if description:
                            await database_api.store_action_info(
                                chat_stream=self.chat_stream,
                                action_build_into_prompt=True,
                                action_prompt_display=f"你查看了图片，看到了: {description}",
                                action_done=True,
                                thinking_id=self.thinking_id,
                                action_data={"image_description": description},
                                action_name="analyze_image",
                            )
                            return True, f"图片分析结果: {description}"
                    except Exception as e:
                        logger.warning(f"读取图片文件失败: {e}")
                return False, "找到了图片ID但无法获取描述"

            return False, "图片分析失败"

        except Exception as e:
            logger.error(f"图片分析动作执行失败: {e}")
            return False, f"图片分析失败: {e}"

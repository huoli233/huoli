import traceback
import os
import re
import time
import random

from typing import Dict, Any, Optional, Set
from maim_message import UserInfo, Seg, GroupInfo

from src.common.logger import get_logger
from src.config.config import global_config
from src.chat.message_receive.chat_stream import get_chat_manager
from src.chat.message_receive.message import MessageRecv
from src.chat.message_receive.storage import MessageStorage
from src.chat.heart_flow.heartflow_message_processor import HeartFCMessageReceiver
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.plugin_system.core.component_registry import component_registry
from src.plugin_system.core.events_manager import events_manager
from src.plugin_system.core.global_announcement_manager import global_announcement_manager
from src.plugin_system.base.base_command import BaseCommand
from src.plugin_system.base.component_types import EventType

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

logger = get_logger("chat")


def _check_ban_words(text: str, userinfo: UserInfo, group_info: Optional[GroupInfo] = None) -> bool:
    """检查消息是否包含过滤词

    Args:
        text: 待检查的文本
        chat: 聊天对象
        userinfo: 用户信息

    Returns:
        bool: 是否包含过滤词
    """
    for word in global_config.message_receive.ban_words:
        if word in text:
            chat_name = group_info.group_name if group_info else "私聊"
            logger.info(f"[{chat_name}]{userinfo.user_nickname}:{text}")
            logger.info(f"[过滤词识别]消息中含有{word}，filtered")
            return True
    return False


def _check_ban_regex(text: str, userinfo: UserInfo, group_info: Optional[GroupInfo] = None) -> bool:
    """检查消息是否匹配过滤正则表达式

    Args:
        text: 待检查的文本
        chat: 聊天对象
        userinfo: 用户信息

    Returns:
        bool: 是否匹配过滤正则
    """
    # 检查text是否为None或空字符串
    if text is None or not text:
        return False

    for pattern in global_config.message_receive.ban_msgs_regex:
        if re.search(pattern, text):
            chat_name = group_info.group_name if group_info else "私聊"
            logger.info(f"[{chat_name}]{userinfo.user_nickname}:{text}")
            logger.info(f"[正则表达式过滤]消息匹配到{pattern}，filtered")
            return True
    return False


class ChatBot:
    _processed_events: Set[str] = set()

    def __init__(self):
        self.bot = None
        self._started = False
        self.heartflow_message_receiver = HeartFCMessageReceiver()
        self._recall_response_cooldown: Dict[str, float] = {}

    async def _ensure_started(self):
        if not self._started:
            logger.debug("确保ChatBot所有任务已启动")
            self._started = True

    async def _collect_sticker(self, url: str):
        try:
            from src.chat.emoji_system.emoji_manager import emoji_manager
            if emoji_manager and hasattr(emoji_manager, 'save_emoji'):
                await emoji_manager.save_emoji(url)
        except Exception as e:
            logger.debug(f"收集表情包失败: {e}")

    async def _handle_recall_response(self, message: MessageRecv):
        try:
            chat = message.chat_stream
            if not chat:
                return
            stream_id = chat.stream_id
            now = time.time()
            last_recall = self._recall_response_cooldown.get(stream_id, 0)
            if now - last_recall < 60:
                return
            if random.random() > 0.3:
                return
            self._recall_response_cooldown[stream_id] = now
            user_name = ""
            if message.message_info and message.message_info.user_info:
                user_name = message.message_info.user_info.user_nickname or ""
            logger.info(f"检测到 {user_name} 的撤回，可能触发回复")
        except Exception as e:
            logger.debug(f"撤回响应处理失败: {e}")

    async def _process_commands(self, message: MessageRecv):
        # sourcery skip: use-named-expression
        """使用新插件系统处理命令"""
        try:
            text = message.processed_plain_text

            # 使用新的组件注册中心查找命令
            command_result = component_registry.find_command_by_text(text)
            if command_result:
                command_class, matched_groups, command_info = command_result
                plugin_name = command_info.plugin_name
                command_name = command_info.name
                if (
                    message.chat_stream
                    and message.chat_stream.stream_id
                    and command_name
                    in global_announcement_manager.get_disabled_chat_commands(message.chat_stream.stream_id)
                ):
                    logger.info("用户禁用的命令，跳过处理")
                    return False, None, True

                message.is_command = True

                # 获取插件配置
                plugin_config = component_registry.get_plugin_config(plugin_name)

                # 创建命令实例
                command_instance: BaseCommand = command_class(message, plugin_config)
                command_instance.set_matched_groups(matched_groups)

                try:
                    # 执行命令
                    success, response, intercept_message_level = await command_instance.execute()
                    message.intercept_message_level = intercept_message_level

                    # 记录命令执行结果
                    if success:
                        logger.info(f"命令执行成功: {command_class.__name__} (拦截等级: {intercept_message_level})")
                    else:
                        logger.warning(f"命令执行失败: {command_class.__name__} - {response}")

                    # 根据命令的拦截设置决定是否继续处理消息
                    return (
                        True,
                        response,
                        not bool(intercept_message_level),
                    )  # 找到命令，根据intercept_message决定是否继续

                except Exception as e:
                    logger.error(f"执行命令时出错: {command_class.__name__} - {e}")
                    logger.error(traceback.format_exc())

                    try:
                        await command_instance.send_text(f"命令执行出错: {str(e)}")
                    except Exception as send_error:
                        logger.error(f"发送错误消息失败: {send_error}")

                    # 命令出错时，根据命令的拦截设置决定是否继续处理消息
                    return True, str(e), False  # 出错时继续处理消息

            # 没有找到命令，继续处理消息
            return False, None, True

        except Exception as e:
            logger.error(f"处理命令时出错: {e}")
            return False, None, True  # 出错时继续处理消息

    async def handle_notice_message(self, message: MessageRecv):
        if message.message_info.message_id == "notice":
            message.is_notify = True
            logger.debug("notice消息")
            try:
                seg = message.message_segment
                mi = message.message_info
                sub_type = None
                scene = None
                msg_id = None
                recalled_id = None

                if getattr(seg, "type", None) == "notify" and isinstance(getattr(seg, "data", None), dict):
                    sub_type = seg.data.get("sub_type")
                    scene = seg.data.get("scene")
                    msg_id = seg.data.get("message_id")
                    recalled = seg.data.get("recalled_user_info") or {}
                    if isinstance(recalled, dict):
                        recalled_id = recalled.get("user_id")

                op = mi.user_info
                gid = mi.group_info.group_id if mi.group_info else None

                # 撤回事件打印；无法获取被撤回者则省略
                if sub_type == "recall":
                    op_name = (
                        getattr(op, "user_cardname", None)
                        or getattr(op, "user_nickname", None)
                        or str(getattr(op, "user_id", None))
                    )
                    recalled_name = None
                    try:
                        if isinstance(recalled, dict):
                            recalled_name = (
                                recalled.get("user_cardname")
                                or recalled.get("user_nickname")
                                or str(recalled.get("user_id"))
                            )
                    except Exception:
                        pass

                    if recalled_name and str(recalled_id) != str(getattr(op, "user_id", None)):
                        logger.info(f"{op_name} 撤回了 {recalled_name} 的消息")
                    else:
                        logger.info(f"{op_name} 撤回了消息")
                else:
                    logger.debug(
                        f"[notice] sub_type={sub_type} scene={scene} op={getattr(op, 'user_nickname', None)}({getattr(op, 'user_id', None)}) "
                        f"gid={gid} msg_id={msg_id} recalled={recalled_id}"
                    )
            except Exception:
                logger.info("[notice] (简略) 收到一条通知事件")

            return True

        return

    async def echo_message_process(self, raw_data: Dict[str, Any]) -> None:
        """
        用于专门处理回送消息ID的函数
        """
        message_data: Dict[str, Any] = raw_data.get("content", {})
        if not message_data:
            return
        message_type = message_data.get("type")
        if message_type != "echo":
            return
        mmc_message_id = message_data.get("echo")
        actual_message_id = message_data.get("actual_id")
        if MessageStorage.update_message(mmc_message_id, actual_message_id):
            logger.debug(f"更新消息ID成功: {mmc_message_id} -> {actual_message_id}")
        else:
            logger.warning(f"更新消息ID失败: {mmc_message_id} -> {actual_message_id}")

    async def command_response_process(self, raw_data: Dict[str, Any]) -> None:
        """
        处理命令响应消息（如撤回命令的执行结果）
        """
        message_data: Dict[str, Any] = raw_data.get("content", {})
        if not message_data:
            return

        command_name = message_data.get("command_name")
        success = message_data.get("success")
        error = message_data.get("error")

        if command_name == "recall_message":
            if success:
                logger.debug(f"[command_response] 撤回命令执行成功")
            else:
                logger.warning(f"[command_response] 撤回命令执行失败: {error}")
        else:
            # 其他命令的响应处理
            if success:
                logger.debug(f"[command_response] 命令 {command_name} 执行成功")
            else:
                logger.warning(f"[command_response] 命令 {command_name} 执行失败: {error}")


    async def message_process(self, message_data: Dict[str, Any]) -> None:
        """处理转化后的统一格式消息
        这个函数本质是预处理一些数据，根据配置信息和消息内容，预处理消息，并分发到合适的消息处理器中
        heart_flow模式：使用思维流系统进行回复
        - 包含思维流状态管理
        - 在回复前进行观察和状态更新
        - 回复后更新思维流状态
        - 消息过滤
        - 记忆激活
        - 意愿计算
        - 消息生成和发送
        - 表情包处理
        - 性能计时
        """
        try:
            # 确保所有任务已启动
            await self._ensure_started()

            if message_data["message_info"].get("group_info") is not None:
                message_data["message_info"]["group_info"]["group_id"] = str(
                    message_data["message_info"]["group_info"]["group_id"]
                )
            if message_data["message_info"].get("user_info") is not None:
                message_data["message_info"]["user_info"]["user_id"] = str(
                    message_data["message_info"]["user_info"]["user_id"]
                )
            # print(message_data)
            # logger.debug(str(message_data))
            message = MessageRecv(message_data)
            group_info = message.message_info.group_info
            user_info = message.message_info.user_info

            continue_flag, modified_message = await events_manager.handle_mai_events(
                EventType.ON_MESSAGE_PRE_PROCESS, message
            )
            if not continue_flag:
                return
            if modified_message and modified_message._modify_flags.modify_message_segments:
                message.message_segment = Seg(type="seglist", data=modified_message.message_segments)

            if await self.handle_notice_message(message):
                pass

            # 处理消息内容，生成纯文本
            await message.process()

            # 平台层的 @ 检测由底层 is_mentioned_bot_in_message 统一处理；此处不做用户名硬编码匹配

            # 过滤检查
            if _check_ban_words(
                message.processed_plain_text,
                user_info,  # type: ignore
                group_info,
            ) or _check_ban_regex(
                message.raw_message,  # type: ignore
                user_info,  # type: ignore
                group_info,
            ):
                return

            from src.modules.modcore.psychological_core import get_psychological_core, handle_harassment_unified
            psych_core = get_psychological_core()
            user_id = str(user_info.user_id) if user_info else ""
            stream_id_tmp = str(group_info.group_id) if group_info else user_id
            if user_id and stream_id_tmp:
                blocked, block_reason = psych_core.is_user_blocked(stream_id_tmp, user_id)
                if blocked:
                    logger.info(f"[modcore] 屏蔽用户拦截: uid={user_id[:8]}, 原因={block_reason}")
                    return
                is_harass, harass_details = psych_core.detect_harassment_unified(
                    user_id=user_id,
                    channel_id=stream_id_tmp,
                    content=message.processed_plain_text,
                )
                if is_harass and harass_details.get("severity", 0) >= 3:
                    psy_response = await handle_harassment_unified(
                        stream_id=stream_id_tmp,
                        user_id=user_id,
                        content=message.processed_plain_text,
                        harassment_details=harass_details,
                    )
                    if psy_response.should_block:
                        logger.info(
                            f"[modcore] 骚扰触发屏蔽: uid={user_id[:8]}, "
                            f"时长={psy_response.block_duration:.0f}秒, 原因={psy_response.block_reason}"
                        )
                        return
                    logger.info(f"[modcore] 骚扰处理完成: uid={user_id[:8]}, 强度={harass_details.get('total_intensity', 0)}")

            get_chat_manager().register_message(message)

            chat = await get_chat_manager().get_or_create_stream(
                platform=message.message_info.platform,  # type: ignore
                user_info=user_info,  # type: ignore
                group_info=group_info,
            )

            message.update_chat_stream(chat)

            # if await self.check_ban_content(message):
            #     logger.warning(f"检测到消息中含有违法，色情，暴力，反动，敏感内容，消息内容：{message.processed_plain_text}，发送者：{message.message_info.user_info.user_nickname}")
            #     return

            # 命令处理 - 使用新插件系统检查并处理命令
            is_command, cmd_result, continue_process = await self._process_commands(message)

            # 如果是命令且不需要继续处理，则直接返回
            if is_command and not continue_process:
                await MessageStorage.store_message(message, chat)
                logger.info(f"命令处理完成，跳过后续消息处理: {cmd_result}")
                return

            continue_flag, modified_message = await events_manager.handle_mai_events(EventType.ON_MESSAGE, message)
            if not continue_flag:
                return
            if modified_message and modified_message._modify_flags.modify_plain_text:
                message.processed_plain_text = modified_message.plain_text

            # 确认从接口发来的message是否有自定义的prompt模板信息
            if message.message_info.template_info and not message.message_info.template_info.template_default:
                template_group_name: Optional[str] = message.message_info.template_info.template_name  # type: ignore
                template_items = message.message_info.template_info.template_items
                async with global_prompt_manager.async_message_scope(template_group_name):
                    if isinstance(template_items, dict):
                        for k in template_items.keys():
                            await Prompt.create_async(template_items[k], k)
                            logger.debug(f"注册{template_items[k]},{k}")
            else:
                template_group_name = None

            async def preprocess():
                await self.heartflow_message_receiver.process_message(message)

            if template_group_name:
                async with global_prompt_manager.async_message_scope(template_group_name):
                    await preprocess()
            else:
                await preprocess()

        except Exception as e:
            logger.error(f"预处理消息失败: {e}")
            traceback.print_exc()


    @staticmethod
    def extract_image_data_from_seg(seg) -> Dict[str, Any]:
        """从消息段中提取图片数据，支持URL、base64、hash等多种格式"""
        import base64 as b64_module
        import hashlib as hl_module
        result = {'url': None, 'base64': None, 'hash': None, 'data': None}
        data_dict = {}
        if isinstance(seg, dict):
            data_dict = seg.get("data", {}) if isinstance(seg.get("data"), dict) else seg
        elif hasattr(seg, "data"):
            data_dict = getattr(seg, "data", {}) if isinstance(getattr(seg, "data", None), dict) else {}
        file_url = None
        if hasattr(seg, "url") and getattr(seg, "url", None):
            file_url = getattr(seg, "url", None)
        elif isinstance(seg, dict) or data_dict:
            file_url = (
                seg.get("url") if isinstance(seg, dict) else None or
                data_dict.get("url") or
                data_dict.get("file") or
                data_dict.get("path") or
                data_dict.get("src")
            )
        if not file_url and hasattr(seg, "file"):
            file_url = getattr(seg, "file", None)
        if file_url and str(file_url).startswith("http"):
            result['url'] = file_url
        base64_data = None
        if isinstance(seg, dict):
            base64_data = seg.get("base64") or seg.get("data_base64")
        if not base64_data and data_dict:
            base64_data = data_dict.get("base64") or data_dict.get("data_base64") or data_dict.get("data")
        if not base64_data and hasattr(seg, "base64"):
            base64_data = getattr(seg, "base64", None)
        if isinstance(base64_data, str):
            if base64_data.startswith("data:image"):
                base64_data = base64_data.split(",", 1)[-1] if "," in base64_data else base64_data
            try:
                decoded = b64_module.b64decode(base64_data, validate=True)
                if len(decoded) > 0:
                    result['base64'] = base64_data
                    result['data'] = decoded
            except Exception:
                pass
        image_hash = None
        if isinstance(seg, dict):
            image_hash = seg.get("hash") or seg.get("image_hash") or seg.get("content_hash")
        if not image_hash and data_dict:
            image_hash = data_dict.get("hash") or data_dict.get("image_hash") or data_dict.get("content_hash")
        if not image_hash and hasattr(seg, "hash"):
            image_hash = getattr(seg, "hash", None)
        if image_hash and isinstance(image_hash, str) and len(image_hash) >= 8:
            result['hash'] = image_hash
        if result['url'] and not result['hash'] and 'hash=' in result['url']:
            try:
                import urllib.parse
                parsed = urllib.parse.urlparse(result['url'])
                params = urllib.parse.parse_qs(parsed.query)
                if 'hash' in params:
                    result['hash'] = params['hash'][0]
            except Exception:
                pass
        return result

    @staticmethod
    def extract_url_from_seg(seg) -> Optional[str]:
        """从消息段中提取URL（兼容旧接口）"""
        image_data = ChatBot.extract_image_data_from_seg(seg)
        return image_data.get('url')

    def get_stats(self) -> Dict[str, Any]:
        """获取消息处理统计信息"""
        chat_manager = get_chat_manager()
        return {
            "started": self.heartflow_message_receiver is not None,
            "stream_count": len(chat_manager.chat_streams) if hasattr(chat_manager, 'chat_streams') else 0,
        }


# 创建全局ChatBot实例
chat_bot = ChatBot()

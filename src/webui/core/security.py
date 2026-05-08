"""
WebUI Token 管理模块
负责生成、保存、验证和更新访问令牌
"""

import hashlib
import secrets
import time
from pathlib import Path
from typing import Optional

from src.common.database.slot_storage import load_slot, save_slot
from src.common.logger import get_logger
from src.common.constants import get_local_now

logger = get_logger("WebUI")

# Token 过期时间（7天）
TOKEN_EXPIRY_SECONDS = 7 * 24 * 60 * 60
WEBUI_CONFIG_SLOT_KEY = "webui:config"


def load_webui_config() -> dict:
    """读取 WebUI 共享配置。"""
    config = load_slot(WEBUI_CONFIG_SLOT_KEY, {})
    return config if isinstance(config, dict) else {}


def save_webui_config(config: dict) -> bool:
    """保存 WebUI 共享配置。"""
    return save_slot(WEBUI_CONFIG_SLOT_KEY, config, ttl_days=3650)


def _token_digest(token: str) -> str:
    """生成 Token 的 SHA-256 哈希摘要前8位，用于日志安全输出"""
    if not token:
        return "空"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]


class TokenManager:
    """Token 管理器"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化 Token 管理器

        Args:
            config_path: 兼容旧接口，当前配置统一存入 Huoli.db
        """
        self.config_path = config_path or Path(f"Huoli.db:{WEBUI_CONFIG_SLOT_KEY}")

        # 确保数据库配置存在并包含有效的 token
        self._ensure_config()

    def _ensure_config(self):
        """确保数据库配置存在且包含有效的 token"""
        config = self._load_config()
        if not config:
            logger.info("WebUI 配置不存在，正在写入数据库")
            self._create_new_token()
        else:
            if not config.get("access_token"):
                logger.warning("WebUI 配置中缺少 access_token，正在重新生成")
                self._create_new_token()
            else:
                # 检查Token是否过期
                expires_at = config.get("token_expires_at", 0)
                if expires_at and time.time() > expires_at:
                    logger.warning("WebUI Token 已过期，正在重新生成")
                    self._create_new_token()
                else:
                    logger.info(
                        f"WebUI Token 已加载: {_token_digest(config['access_token'])}..."
                    )

    def _load_config(self) -> dict:
        """从数据库加载 WebUI 配置"""
        return load_webui_config()

    def _save_config(self, config: dict):
        """保存 WebUI 配置到 Huoli.db"""
        if not save_webui_config(config):
            raise RuntimeError("WebUI 配置写入数据库失败")
        logger.debug("WebUI 配置已保存到 Huoli.db")

    def _create_new_token(self) -> str:
        """生成新的 64 位随机 token，并设置过期时间"""
        # 生成 64 位十六进制字符串 (32 字节 = 64 hex 字符)
        token = secrets.token_hex(32)
        now = time.time()

        config = {
            "access_token": token,
            "token_created_at": now,
            "token_expires_at": now + TOKEN_EXPIRY_SECONDS,
            "created_at": self._get_current_timestamp(),
            "updated_at": self._get_current_timestamp(),
            "first_setup_completed": False,  # 标记首次配置未完成
        }

        self._save_config(config)
        logger.info(f"新的 WebUI Token 已生成: {_token_digest(token)}...")

        return token

    def _get_current_timestamp(self) -> str:
        """获取当前时间戳字符串"""
        return get_local_now().isoformat()

    def get_token(self) -> str:
        """获取当前有效的 token"""
        config = self._load_config()
        return config.get("access_token", "")

    def verify_token(self, token: str) -> bool:
        """
        验证 token 是否有效（含过期检查）

        Args:
            token: 待验证的 token

        Returns:
            bool: token 是否有效
        """
        if not token:
            return False

        config = self._load_config()
        current_token = config.get("access_token", "")
        if not current_token:
            logger.error("系统中没有有效的 token")
            return False

        # 检查Token是否过期
        expires_at = config.get("token_expires_at", 0)
        if expires_at and time.time() > expires_at:
            logger.warning("Token 已过期，验证失败")
            return False

        # 使用 secrets.compare_digest 防止时序攻击
        is_valid = secrets.compare_digest(token, current_token)

        if is_valid:
            logger.debug("Token 验证成功")
        else:
            logger.warning("Token 验证失败")

        return is_valid

    def update_token(self, new_token: str) -> tuple[bool, str]:
        """
        更新 token

        Args:
            new_token: 新的 token (最少 10 位，必须包含大小写字母和特殊符号)

        Returns:
            tuple[bool, str]: (是否更新成功, 错误消息)
        """
        # 验证新 token 格式
        is_valid, error_msg = self._validate_custom_token(new_token)
        if not is_valid:
            logger.error(f"Token 格式无效: {error_msg}")
            return False, error_msg

        try:
            config = self._load_config()
            old_digest = _token_digest(config.get("access_token", ""))

            config["access_token"] = new_token
            config["token_created_at"] = time.time()
            config["token_expires_at"] = time.time() + TOKEN_EXPIRY_SECONDS
            config["updated_at"] = self._get_current_timestamp()

            self._save_config(config)
            logger.info(f"Token 已更新: {old_digest}... -> {_token_digest(new_token)}...")

            return True, "Token 更新成功"
        except Exception as e:
            logger.error(f"更新 Token 失败: {e}")
            return False, f"更新失败: {str(e)}"

    def regenerate_token(self) -> str:
        """
        重新生成 token（保留 first_setup_completed 状态）

        Returns:
            str: 新生成的 token
        """
        logger.info("正在重新生成 WebUI Token...")

        # 生成新的 64 位十六进制字符串
        new_token = secrets.token_hex(32)

        # 加载现有配置，保留 first_setup_completed 状态
        config = self._load_config()
        old_digest = _token_digest(config.get("access_token", ""))
        first_setup_completed = config.get(
            "first_setup_completed", True
        )  # 默认为 True，表示已完成配置

        now = time.time()
        config["access_token"] = new_token
        config["token_created_at"] = now
        config["token_expires_at"] = now + TOKEN_EXPIRY_SECONDS
        config["updated_at"] = self._get_current_timestamp()
        config["first_setup_completed"] = (
            first_setup_completed  # 保留原来的状态
        )

        self._save_config(config)
        logger.info(
            f"WebUI Token 已重新生成: {old_digest}... -> {_token_digest(new_token)}..."
        )

        return new_token

    def _validate_token_format(self, token: str) -> bool:
        """
        验证 token 格式是否正确（旧的 64 位十六进制验证，保留用于系统生成的 token）

        Args:
            token: 待验证的 token

        Returns:
            bool: 格式是否正确
        """
        if not token or not isinstance(token, str):
            return False

        # 必须是 64 位十六进制字符串
        if len(token) != 64:
            return False

        # 验证是否为有效的十六进制字符串
        try:
            int(token, 16)
            return True
        except ValueError:
            return False

    def _validate_custom_token(self, token: str) -> tuple[bool, str]:
        """
        验证自定义 token 格式

        要求:
        - 最少 10 位
        - 包含大写字母
        - 包含小写字母
        - 包含特殊符号

        Args:
            token: 待验证的 token

        Returns:
            tuple[bool, str]: (是否有效, 错误消息)
        """
        if not token or not isinstance(token, str):
            return False, "Token 不能为空"

        # 检查长度
        if len(token) < 10:
            return False, "Token 长度至少为 10 位"

        # 检查是否包含大写字母
        has_upper = any(c.isupper() for c in token)
        if not has_upper:
            return False, "Token 必须包含大写字母"

        # 检查是否包含小写字母
        has_lower = any(c.islower() for c in token)
        if not has_lower:
            return False, "Token 必须包含小写字母"

        # 检查是否包含特殊符号
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?/"
        has_special = any(c in special_chars for c in token)
        if not has_special:
            return False, f"Token 必须包含特殊符号 ({special_chars})"

        return True, "Token 格式正确"

    def is_first_setup(self) -> bool:
        """
        检查是否为首次配置

        Returns:
            bool: 是否为首次配置
        """
        config = self._load_config()
        return not config.get("first_setup_completed", False)

    def mark_setup_completed(self) -> bool:
        """
        标记首次配置已完成

        Returns:
            bool: 是否标记成功
        """
        try:
            config = self._load_config()
            config["first_setup_completed"] = True
            config["setup_completed_at"] = self._get_current_timestamp()
            self._save_config(config)
            logger.info("首次配置已标记为完成")
            return True
        except Exception as e:
            logger.error(f"标记首次配置完成失败: {e}")
            return False

    def reset_setup_status(self) -> bool:
        """
        重置首次配置状态，允许重新进入配置向导

        Returns:
            bool: 是否重置成功
        """
        try:
            config = self._load_config()
            config["first_setup_completed"] = False
            if "setup_completed_at" in config:
                del config["setup_completed_at"]
            self._save_config(config)
            logger.info("首次配置状态已重置")
            return True
        except Exception as e:
            logger.error(f"重置首次配置状态失败: {e}")
            return False


# 全局单例
_token_manager_instance: Optional[TokenManager] = None


def get_token_manager() -> TokenManager:
    """获取 TokenManager 单例"""
    global _token_manager_instance
    if _token_manager_instance is None:
        _token_manager_instance = TokenManager()
    return _token_manager_instance

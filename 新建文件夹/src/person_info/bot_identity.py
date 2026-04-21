import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.common.config.config_engine import ConfigEngine
from src.common.logger import get_logger

logger = get_logger("bot_identity")

IDENTITY_BOUNDARY_TEXT = "不要把自己说成AI助手、系统、客服或工具；你就是在这个聊天现场里说话的本人。"


def _mask_id(id_str: str) -> str:
    """遮蔽ID"""
    if not id_str:
        return ""
    s = str(id_str)
    if len(s) <= 1:
        return s
    return s[0] + "*" * (len(s) - 1)


@dataclass
class BotPersona:
    """机器人人格"""

    nickname: str = ""
    alias_names: List[str] = field(default_factory=list)
    personality_core: str = ""
    character_age: int = 0
    reply_style: str = ""
    interests: List[str] = field(default_factory=list)
    visual_features: List[str] = field(default_factory=list)
    lore: str = ""
    relationships: Dict[str, str] = field(default_factory=dict)


@dataclass
class BotAccountInfo:
    """机器人账号信息"""

    bot_id: str = ""
    bot_nickname: str = ""
    avatar_url: str = ""
    group_cards: Dict[str, str] = field(default_factory=dict)
    group_roles: Dict[str, str] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


class PersonaHashTracker:
    """人格哈希追踪器"""

    def __init__(self):
        self._current_hash: str = ""
        self._last_check: float = 0
        self._check_interval: float = 60.0

    def compute_hash(self, persona: BotPersona) -> str:
        """计算人格哈希"""
        content = json.dumps(
            {
                "nickname": persona.nickname,
                "alias": persona.alias_names,
                "core": persona.personality_core,
                "style": persona.reply_style,
            },
            sort_keys=True,
        )
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def has_changed(self, persona: BotPersona) -> bool:
        """检查人格是否变化"""
        new_hash = self.compute_hash(persona)
        changed = new_hash != self._current_hash
        if changed:
            self._current_hash = new_hash
        return changed

    def should_check(self) -> bool:
        """是否应该检查"""
        if time.time() - self._last_check >= self._check_interval:
            self._last_check = time.time()
            return True
        return False


class NameMatcher:
    """名称匹配器"""

    def __init__(self):
        self._names: set = set()
        self._patterns: List[str] = []

    def set_names(
        self, primary: str, aliases: Optional[List[str]] = None
    ) -> None:
        """设置名称"""
        self._names.clear()
        if primary:
            self._names.add(primary.lower())
        if aliases:
            for alias in aliases:
                if alias:
                    self._names.add(alias.lower())

    def add_pattern(self, pattern: str) -> None:
        """添加匹配模式"""
        if pattern and pattern not in self._patterns:
            self._patterns.append(pattern)

    def is_mentioned(self, text: str) -> Tuple[bool, str]:
        """检查是否被提及"""
        text_lower = text.lower()
        for name in self._names:
            if name in text_lower:
                return True, name
        for pattern in self._patterns:
            if pattern.lower() in text_lower:
                return True, pattern
        return False, ""

    def get_all_names(self) -> List[str]:
        """获取所有名称"""
        return list(self._names) + self._patterns


class BotIdentityManager:
    """机器人身份管理器"""

    def __init__(self, config_engine: Optional[ConfigEngine] = None):
        self._config = config_engine or ConfigEngine.get_instance()
        self._persona = BotPersona()
        self._account = BotAccountInfo()
        self._hash_tracker = PersonaHashTracker()
        self._name_matcher = NameMatcher()
        self._initialized = False
        self._load_config()

    def _load_config(self) -> None:
        """加载配置"""
        nickname = self._config.get(
            "bot_identity", "nickname", ""
        ) or self._config.get("bot", "nickname", "")
        self._persona.nickname = nickname
        alt_names = self._config.get(
            "bot_identity", "alias_names", []
        ) or self._config.get("bot", "alias_names", [])
        if isinstance(alt_names, list):
            self._persona.alias_names = alt_names
        elif isinstance(alt_names, str):
            self._persona.alias_names = [
                a.strip() for a in alt_names.split(",") if a.strip()
            ]

        self._persona.personality_core = self._config.get(
            "bot_identity", "personality_core", ""
        ) or self._config.get("personality", "personality", "")
        self._persona.character_age = self._config.get(
            "bot_identity", "character_age", 0
        ) or self._config.get("personality", "character_age", 0)
        self._persona.reply_style = self._config.get(
            "bot_identity", "reply_style", ""
        ) or self._config.get("personality", "reply_style", "")

        interest_str = self._config.get(
            "bot_identity", "interests", ""
        ) or self._config.get("personality", "character_hobbies", "")
        if interest_str:
            self._persona.interests = [
                i.strip() for i in interest_str.split(",") if i.strip()
            ]

        visual_str = self._config.get("bot_identity", "visual_features", "")
        if visual_str:
            self._persona.visual_features = [
                v.strip() for v in visual_str.split(",") if v.strip()
            ]

        self._persona.lore = self._config.get(
            "bot_identity", "lore", ""
        ) or self._config.get("personality", "character_background", "")

        rels_str = self._config.get("bot_identity", "relationships", "")
        if rels_str:
            for item in rels_str.split(","):
                if ":" in item:
                    k, v = item.split(":", 1)
                    self._persona.relationships[k.strip()] = v.strip()

        self._account.bot_id = str(
            self._config.get("bot_identity", "bot_id", "")
        )
        self._name_matcher.set_names(
            self._persona.nickname, self._persona.alias_names
        )

        if self._hash_tracker.has_changed(self._persona):
            logger.info(f"人格配置已更新: {self._persona.nickname}")

        self._initialized = True

    def set_account_info(
        self, bot_id: str, nickname: str = "", avatar_url: str = ""
    ) -> None:
        """设置账号信息"""
        self._account.bot_id = bot_id
        if nickname:
            self._account.bot_nickname = nickname
        if avatar_url:
            self._account.avatar_url = avatar_url
        self._account.last_updated = time.time()
        logger.info(f"账号信息已更新: ID={_mask_id(bot_id)}, 昵称={nickname}")

    def set_group_card(self, group_id: str, card: str) -> None:
        """设置群名片"""
        self._account.group_cards[group_id] = card

    def set_group_role(self, group_id: str, role: str) -> None:
        """设置群角色"""
        self._account.group_roles[group_id] = role
        role_display = {
            "admin": "管理员",
            "owner": "群主",
            "member": "成员",
        }.get(role, role)
        logger.info(f"群:{_mask_id(group_id)} 角色: {role_display}")

    def get_group_role(self, group_id: str) -> str:
        """获取群角色"""
        return self._account.group_roles.get(group_id, "member")

    def is_admin_in_group(self, group_id: str) -> bool:
        """是否群管理员"""
        role = self.get_group_role(group_id)
        return role in ("admin", "owner")

    def get_admin_groups(self) -> List[str]:
        """获取管理员群列表"""
        return [
            gid
            for gid, role in self._account.group_roles.items()
            if role in ("admin", "owner")
        ]

    def get_avatar_url(self) -> str:
        """获取头像URL"""
        if self._account.avatar_url:
            return self._account.avatar_url
        return ""

    def get_bot_id(self) -> str:
        """获取机器人ID"""
        return self._account.bot_id

    def get_nickname(self) -> str:
        """获取昵称"""
        return self._account.bot_nickname

    def get_group_card(self, group_id: str) -> str:
        """获取群名片"""
        return self._account.group_cards.get(
            group_id, self._account.bot_nickname
        )

    def get_display_name(self, group_id: Optional[str] = None) -> str:
        """获取显示名称"""
        if group_id and group_id in self._account.group_cards:
            return self._account.group_cards[group_id]
        if self._persona.nickname:
            return self._persona.nickname
        return self._account.bot_nickname or self._account.bot_id

    def get_persona(self) -> BotPersona:
        """获取人格"""
        return self._persona

    def get_nicknames(self) -> List[str]:
        """获取所有昵称和别名"""
        return self._persona.alias_names

    def get_all_names(self) -> List[str]:
        """获取所有名称"""
        names = self._name_matcher.get_all_names()
        if (
            self._account.bot_nickname
            and self._account.bot_nickname.lower()
            not in [n.lower() for n in names]
        ):
            names.append(self._account.bot_nickname)
        return names

    def is_self_mentioned(self, text: str) -> Tuple[bool, str]:
        """检查是否被提及"""
        return self._name_matcher.is_mentioned(text)

    def is_self_in_image(
        self, visual_description: str
    ) -> Tuple[bool, float, List[str]]:
        """检查图像描述中是否包含自己"""
        if not visual_description or not self._persona.visual_features:
            return False, 0.0, []

        desc_lower = visual_description.lower()
        matched = []
        for feature in self._persona.visual_features:
            if feature.lower() in desc_lower:
                matched.append(feature)

        if not matched:
            return False, 0.0, []

        score = len(matched) / len(self._persona.visual_features)
        is_self = score >= 0.6 and len(matched) >= 2
        return is_self, score, matched

    def get_identity_prompt(self, group_id: Optional[str] = None) -> str:
        """获取身份提示词"""
        name = self.get_display_name(group_id)
        aliases = (
            ", ".join(self._persona.alias_names)
            if self._persona.alias_names
            else ""
        )
        personality = self._persona.personality_core
        style = self._persona.reply_style
        lore = self._persona.lore
        age = self._persona.character_age

        parts = [f"你叫{name}，就是这个聊天现场里的普通群友之一。"]
        if aliases:
            parts.append(f"常用称呼: {aliases}")
        if age > 0:
            parts.append(f"年龄: {age}岁")
        if personality:
            parts.append(f"【核心人格】\n{personality}")
        if style:
            parts.append(f"【表达风格】\n{style}")
        if self._persona.interests:
            parts.append(f"【兴趣爱好】\n{', '.join(self._persona.interests)}")
        if lore:
            parts.append(f"【背景补充】\n{lore}")
        parts.append(f"【身份边界】\n{IDENTITY_BOUNDARY_TEXT}")

        return "\n\n".join(parts)

    def build_persona_brief(
        self,
        group_id: Optional[str] = None,
        *,
        include_name_prefix: bool = True,
        include_style: bool = True,
        include_interests: bool = True,
        include_lore: bool = True,
        fallback_text: str = "自然说话，别端着，也别把自己说成系统或工具。",
    ) -> str:
        """构造可复用的短人格说明，用于私聊/独白/planner 等子链路。"""
        name = self.get_display_name(group_id)
        segments: List[str] = []

        core = (self._persona.personality_core or "").strip()
        style = (self._persona.reply_style or "").strip()
        lore = (self._persona.lore or "").strip()
        interests = [
            item.strip()
            for item in self._persona.interests
            if item and item.strip()
        ]

        if core:
            segments.append(core)
        if include_style and style:
            segments.append(f"说话风格：{style}")
        if include_interests and interests:
            segments.append(f"平时更容易顺口聊到：{'、'.join(interests[:4])}")
        if include_lore and lore:
            segments.append(f"背景补充：{lore}")

        brief = "\n".join(segment for segment in segments if segment).strip()
        if not brief:
            brief = fallback_text

        if include_name_prefix:
            return f"你的名字是{name}，{brief}"
        return brief

    def build_reply_persona_block(
        self,
        group_id: Optional[str] = None,
        *,
        chat_prompt: str = "",
        fallback_text: str = "自然说话，别端着，也别把自己说成系统或工具。",
    ) -> str:
        """给回复器直接使用的统一人格块，合并 identity/style/chat_prompt。"""
        parts: List[str] = []
        persona_brief = self.build_persona_brief(
            group_id,
            include_name_prefix=True,
            include_style=True,
            include_interests=True,
            include_lore=True,
            fallback_text=fallback_text,
        )
        if persona_brief:
            parts.append(f"[主人格]\n{persona_brief}")

        parts.append(f"[身份边界]\n{IDENTITY_BOUNDARY_TEXT}")

        style = (self._persona.reply_style or "").strip()
        if style:
            parts.append(f"[表达要求]\n{style}")

        chat_prompt_text = str(chat_prompt or "").strip()
        if chat_prompt_text:
            parts.append(f"[场景补充]\n{chat_prompt_text}")

        return "\n\n".join(parts)

    def get_persona_prompt(self) -> str:
        """获取人格提示词"""
        parts = []
        if self._persona.nickname:
            parts.append(f"名字: {self._persona.nickname}")
        if self._persona.character_age > 0:
            parts.append(f"年龄: {self._persona.character_age}岁")
        if self._persona.personality_core:
            parts.append(f"人设: {self._persona.personality_core}")
        if self._persona.reply_style:
            parts.append(f"说话风格: {self._persona.reply_style}")
        if self._persona.interests:
            parts.append(f"兴趣爱好: {', '.join(self._persona.interests)}")
        return "\n".join(parts)

    def get_visual_prompt(self) -> str:
        """获取视觉提示词"""
        if not self._persona.visual_features:
            return ""
        return f"角色外观特征：{', '.join(self._persona.visual_features)}"

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        admin_groups = self.get_admin_groups()
        return {
            "nickname": self._persona.nickname,
            "bot_id": self._account.bot_id,
            "bot_nickname": self._account.bot_nickname,
            "avatar_url": self.get_avatar_url(),
            "alias_count": len(self._persona.alias_names),
            "visual_feature_count": len(self._persona.visual_features),
            "group_card_count": len(self._account.group_cards),
            "group_role_count": len(self._account.group_roles),
            "admin_group_count": len(admin_groups),
            "admin_groups": admin_groups,
            "initialized": self._initialized,
        }

    def reload_config(self) -> bool:
        """重新加载配置"""
        try:
            self._load_config()
            return True
        except Exception as e:
            logger.error(f"重新加载配置失败: {e}")
            return False


_bot_identity_manager: Optional[BotIdentityManager] = None


def get_bot_identity_manager(
    config_engine: Optional[ConfigEngine] = None,
) -> BotIdentityManager:
    """获取机器人身份管理器单例"""
    global _bot_identity_manager
    if _bot_identity_manager is None:
        _bot_identity_manager = BotIdentityManager(config_engine)
    return _bot_identity_manager


def init_bot_identity(
    config_engine: Optional[ConfigEngine] = None,
) -> BotIdentityManager:
    """初始化机器人身份"""
    global _bot_identity_manager
    _bot_identity_manager = BotIdentityManager(config_engine)
    logger.info(f"机器人身份初始化完成: {_bot_identity_manager.get_stats()}")
    return _bot_identity_manager

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.person_info.runtime_config import (
    identity_dict,
    identity_list,
    identity_module_view,
)

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
class VisualProfile:
    """视觉档案"""

    profile_name: str = ""
    alias_names: List[str] = field(default_factory=list)
    visual_features: List[str] = field(default_factory=list)
    exclusion_features: List[str] = field(default_factory=list)
    match_threshold: float = 0.6
    min_feature_hits: int = 2


@dataclass
class SelfImageMatchResult:
    """自我图片匹配结果"""

    is_self: bool = False
    score: float = 0.0
    matched_features: List[str] = field(default_factory=list)
    matched_profile: str = ""
    profile_source: str = ""
    active_role: str = ""


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

    def set_names(self, primary: str, aliases: Optional[List[str]] = None) -> None:
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

    def __init__(self, config_engine: Optional[Any] = None):
        del config_engine
        self._persona = BotPersona()
        self._account = BotAccountInfo()
        self._hash_tracker = PersonaHashTracker()
        self._name_matcher = NameMatcher()
        self._role_visual_profiles: Dict[str, VisualProfile] = {}
        self._identity_templates: Dict[str, str] = {}
        self._skill_prompt_templates: Dict[str, str] = {}
        self._initialized = False
        self._load_config()

    @staticmethod
    def _normalize_role_key(role_name: str) -> str:
        """标准化角色名称，便于做配置映射。"""
        return str(role_name or "").strip().lower()

    @staticmethod
    def _ensure_list(value: Any) -> List[str]:
        """把配置值规整为字符串列表。"""
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        """安全解析浮点配置。"""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        """安全解析整型配置。"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _build_visual_profile(
        self,
        profile_name: str,
        raw_config: Any,
        *,
        default_features: Optional[List[str]] = None,
    ) -> Optional[VisualProfile]:
        """从配置构造视觉档案。"""
        default_features = list(default_features or [])
        if isinstance(raw_config, dict):
            features = self._ensure_list(raw_config.get("visual_features", default_features))
            aliases = self._ensure_list(raw_config.get("alias_names", []))
            exclusions = self._ensure_list(raw_config.get("exclusion_features", []))
            threshold = self._coerce_float(raw_config.get("match_threshold", 0.6), 0.6)
            min_hits = self._coerce_int(raw_config.get("min_feature_hits", 2), 2)
        else:
            features = self._ensure_list(raw_config or default_features)
            aliases = []
            exclusions = []
            threshold = 0.6
            min_hits = 2
        if not features:
            return None
        return VisualProfile(
            profile_name=profile_name,
            alias_names=aliases,
            visual_features=features,
            exclusion_features=exclusions,
            match_threshold=max(0.1, min(1.0, threshold)),
            min_feature_hits=max(1, min_hits),
        )

    def _load_role_visual_profiles(self) -> None:
        """加载角色视觉档案。"""
        self._role_visual_profiles = {}
        profile_map = identity_dict("identity_bot", "role_visual_profiles", {})
        if not isinstance(profile_map, dict):
            return
        for role_name, profile_data in profile_map.items():
            normalized_key = self._normalize_role_key(role_name)
            if not normalized_key:
                continue
            profile = self._build_visual_profile(role_name, profile_data)
            if profile is not None:
                self._role_visual_profiles[normalized_key] = profile

    def _load_identity_templates(self) -> None:
        """加载身份提示词模板。"""
        template_defaults = {
            "default_persona": "你叫{name}，就是这个聊天现场里的普通群友之一。",
            "role_play": "你当前采用的人设是{role}。请保持角色一致性，不要突然跳出人设。",
            "hybrid": "你叫{name}，当前采用的人设是{role}。在保持角色特点的同时，也要维持稳定的自我一致性。",
            "master_header": "[主人格]",
            "identity_boundary_header": "[身份边界]",
            "style_header": "[表达要求]",
            "scene_header": "[场景补充]",
            "role_header": "[当前角色]",
            "visual_header": "[当前外观识别基准]",
            "skill_header": "[当前技能倾向]",
        }
        loaded_templates = identity_dict("identity_bot", "identity_templates", {})
        if isinstance(loaded_templates, dict):
            merged = dict(template_defaults)
            for key, value in loaded_templates.items():
                if isinstance(key, str) and isinstance(value, str) and value.strip():
                    merged[key] = value
            self._identity_templates = merged
        else:
            self._identity_templates = template_defaults

    def _load_skill_prompt_templates(self) -> None:
        """加载技能提示词模板。"""
        template_map = identity_dict("identity_bot", "skill_prompt_templates", {})
        if isinstance(template_map, dict):
            self._skill_prompt_templates = {
                str(key).strip(): str(value).strip()
                for key, value in template_map.items()
                if str(key).strip() and str(value).strip()
            }
        else:
            self._skill_prompt_templates = {}

    def _get_template(self, key: str, default: str = "") -> str:
        """读取身份模板片段。"""
        value = self._identity_templates.get(key, default)
        if isinstance(value, str) and value.strip():
            return value
        return default

    def _resolve_active_role(self, channel_id: Optional[str] = None) -> str:
        """解析当前频道的角色人格。"""
        if not channel_id:
            return ""
        try:
            from src.core.identity_anchor import get_identity_anchor

            anchor = get_identity_anchor()
            ctx = anchor.get_context(channel_id)
            return str(getattr(ctx, "role_persona", "") or "").strip()
        except Exception as e:
            logger.debug(f"读取当前角色失败: {e}")
            return ""

    def _get_visual_profile_for_channel(self, channel_id: Optional[str] = None) -> Tuple[VisualProfile, str]:
        """根据当前频道返回应使用的视觉档案。"""
        active_role = self._resolve_active_role(channel_id)
        role_key = self._normalize_role_key(active_role)
        if role_key and role_key in self._role_visual_profiles:
            return self._role_visual_profiles[role_key], active_role
        profile = VisualProfile(
            profile_name="主人格",
            alias_names=list(self._persona.alias_names),
            visual_features=list(self._persona.visual_features),
            exclusion_features=[],
            match_threshold=0.6,
            min_feature_hits=2,
        )
        return profile, active_role

    def _match_visual_profile(
        self, visual_description: str, profile: VisualProfile, active_role: str = ""
    ) -> SelfImageMatchResult:
        """执行视觉档案匹配。"""
        if not visual_description or not profile.visual_features:
            return SelfImageMatchResult(active_role=active_role)
        desc_lower = visual_description.lower()
        for excluded in profile.exclusion_features:
            if excluded.lower() in desc_lower:
                return SelfImageMatchResult(
                    is_self=False,
                    score=0.0,
                    matched_features=[],
                    matched_profile=profile.profile_name,
                    profile_source="visual_profile",
                    active_role=active_role,
                )
        matched = []
        for feature in profile.visual_features:
            if feature.lower() in desc_lower:
                matched.append(feature)
        if not matched:
            return SelfImageMatchResult(
                is_self=False,
                score=0.0,
                matched_profile=profile.profile_name,
                profile_source="visual_profile",
                active_role=active_role,
            )
        score = len(matched) / len(profile.visual_features)
        is_self = score >= profile.match_threshold and len(matched) >= profile.min_feature_hits
        return SelfImageMatchResult(
            is_self=is_self,
            score=score,
            matched_features=matched,
            matched_profile=profile.profile_name,
            profile_source=("role_visual_profile" if active_role else "bot_identity_visual_features"),
            active_role=active_role,
        )

    def analyze_self_in_image(self, visual_description: str, channel_id: Optional[str] = None) -> SelfImageMatchResult:
        """分析图像描述中是否出现自己，并返回结构化结果。"""
        profile, active_role = self._get_visual_profile_for_channel(channel_id)
        return self._match_visual_profile(visual_description, profile, active_role)

    def get_active_visual_prompt(self, channel_id: Optional[str] = None) -> str:
        """获取当前频道对应的视觉提示。"""
        profile, active_role = self._get_visual_profile_for_channel(channel_id)
        if not profile.visual_features:
            return ""
        role_label = active_role or profile.profile_name or "主人格"
        return f"当前识别对象: {role_label}；外观特征：{', '.join(profile.visual_features)}"

    def _build_skill_prompt_block(self, channel_id: Optional[str] = None) -> str:
        """根据当前激活技能生成补充提示。"""
        if not channel_id or not self._skill_prompt_templates:
            return ""
        try:
            from src.core.skill_lifecycle_hub import get_skill_lifecycle_hub

            hub = get_skill_lifecycle_hub(channel_id)
            active_skills = hub.get_active_skills()
        except Exception as e:
            logger.debug(f"读取技能生命周期失败: {e}")
            return ""
        fragments: List[str] = []
        for skill in active_skills:
            fragment = self._skill_prompt_templates.get(skill.skill_id)
            if not fragment:
                fragment = self._skill_prompt_templates.get(skill.skill_name)
            if fragment:
                fragments.append(fragment)
        if not fragments:
            return ""
        header = self._get_template("skill_header", "[当前技能倾向]")
        return f"{header}\n" + "\n".join(dict.fromkeys(fragments))

    def _load_config(self) -> None:
        """加载配置"""
        config = identity_module_view("identity_bot")
        self._persona.nickname = str(config.get("nickname", "") or "")
        self._persona.alias_names = identity_list("identity_bot", "alias_names", [])

        self._persona.personality_core = str(
            config.get("personality_core", "") or ""
        )
        self._persona.character_age = self._coerce_int(
            config.get("character_age", 0), 0
        )
        self._persona.reply_style = str(config.get("reply_style", "") or "")

        interests = config.get("interests", [])
        self._persona.interests = self._ensure_list(interests)

        visual_features = config.get("visual_features", [])
        self._persona.visual_features = self._ensure_list(visual_features)
        self._load_role_visual_profiles()
        self._load_identity_templates()
        self._load_skill_prompt_templates()

        self._persona.lore = str(config.get("lore", "") or "")

        self._persona.relationships = {}
        relationships = config.get("relationships", {})
        if isinstance(relationships, dict):
            self._persona.relationships = {
                str(k).strip(): str(v).strip()
                for k, v in relationships.items()
                if str(k).strip() and str(v).strip()
            }

        self._account.bot_id = str(config.get("bot_id", "") or "")
        self._name_matcher.set_names(self._persona.nickname, self._persona.alias_names)

        if self._hash_tracker.has_changed(self._persona):
            logger.info(f"人格配置已更新: {self._persona.nickname}")

        self._initialized = True

    def set_account_info(self, bot_id: str, nickname: str = "", avatar_url: str = "") -> None:
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
        return [gid for gid, role in self._account.group_roles.items() if role in ("admin", "owner")]

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
        return self._account.group_cards.get(group_id, self._account.bot_nickname)

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
        if self._account.bot_nickname and self._account.bot_nickname.lower() not in [n.lower() for n in names]:
            names.append(self._account.bot_nickname)
        return names

    def is_self_mentioned(self, text: str) -> Tuple[bool, str]:
        """检查是否被提及"""
        return self._name_matcher.is_mentioned(text)

    def is_self_in_image(
        self, visual_description: str, channel_id: Optional[str] = None
    ) -> Tuple[bool, float, List[str]]:
        """检查图像描述中是否包含自己"""
        result = self.analyze_self_in_image(visual_description, channel_id)
        return result.is_self, result.score, list(result.matched_features)

    def get_identity_prompt(self, group_id: Optional[str] = None) -> str:
        """获取身份提示词"""
        name = self.get_display_name(group_id)
        aliases = ", ".join(self._persona.alias_names) if self._persona.alias_names else ""
        personality = self._persona.personality_core
        style = self._persona.reply_style
        lore = self._persona.lore
        age = self._persona.character_age

        parts = [
            self._get_template(
                "default_persona",
                "你叫{name}，就是这个聊天现场里的普通群友之一。",
            ).format(name=name)
        ]
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
        interests = [item.strip() for item in self._persona.interests if item and item.strip()]

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
        active_role = self._resolve_active_role(group_id)
        persona_brief = self.build_persona_brief(
            group_id,
            include_name_prefix=True,
            include_style=True,
            include_interests=True,
            include_lore=True,
            fallback_text=fallback_text,
        )
        if persona_brief:
            parts.append(f"{self._get_template('master_header', '[主人格]')}\n{persona_brief}")

        if active_role:
            role_template = self._get_template(
                "role_play",
                "你当前采用的人设是{role}。请保持角色一致性，不要突然跳出人设。",
            )
            parts.append(f"{self._get_template('role_header', '[当前角色]')}\n{role_template.format(role=active_role)}")

        parts.append(f"{self._get_template('identity_boundary_header', '[身份边界]')}\n{IDENTITY_BOUNDARY_TEXT}")

        style = (self._persona.reply_style or "").strip()
        if style:
            parts.append(f"{self._get_template('style_header', '[表达要求]')}\n{style}")

        active_visual = self.get_active_visual_prompt(group_id)
        if active_visual:
            parts.append(f"{self._get_template('visual_header', '[当前外观识别基准]')}\n{active_visual}")

        skill_prompt_block = self._build_skill_prompt_block(group_id)
        if skill_prompt_block:
            parts.append(skill_prompt_block)

        chat_prompt_text = str(chat_prompt or "").strip()
        if chat_prompt_text:
            parts.append(f"{self._get_template('scene_header', '[场景补充]')}\n{chat_prompt_text}")

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
    config_engine: Optional[Any] = None,
) -> BotIdentityManager:
    """获取机器人身份管理器单例"""
    global _bot_identity_manager
    if _bot_identity_manager is None:
        _bot_identity_manager = BotIdentityManager(config_engine)
    return _bot_identity_manager


def init_bot_identity(
    config_engine: Optional[Any] = None,
) -> BotIdentityManager:
    """初始化机器人身份"""
    global _bot_identity_manager
    _bot_identity_manager = BotIdentityManager(config_engine)
    logger.info(f"机器人身份初始化完成: {_bot_identity_manager.get_stats()}")
    return _bot_identity_manager

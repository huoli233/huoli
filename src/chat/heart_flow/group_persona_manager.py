import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field, asdict
from src.common.database.slot_storage import list_slot_keys, load_slot, save_slot
from src.common.logger import get_logger

logger = get_logger("群人格管理")


@dataclass
class GroupPersonaProfile:
    """群特点画像配置文件"""

    group_id: str = ""
    group_name: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    update_interval: float = 86400.0 * 3
    personality_traits: List[str] = field(default_factory=list)
    communication_style: List[str] = field(default_factory=list)
    topics_of_interest: List[str] = field(default_factory=list)
    group_culture: List[str] = field(default_factory=list)
    member_count: int = 0
    active_members: List[str] = field(default_factory=list)
    summary: str = ""
    auto_update_enabled: bool = True


class GroupPersonaManager:
    """群特点配置管理器

    功能：
    1. 维护每个群的画像配置文件
    2. 定时自动更新群特点
    3. 保存到 Huoli.db 统一槽位
    4. 支持手动触发更新
    """

    def __init__(self, storage_path: str = ""):
        del storage_path
        self._storage_path = "Huoli.db:group_persona"
        self._profiles: Dict[str, GroupPersonaProfile] = {}
        self._load_all_profiles()

    def _sanitize_id(self, id_str: str) -> str:
        """清理ID字符串，防止路径遍历攻击。"""
        import re
        # 移除或替换危险字符
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', id_str)
        # 移除路径遍历模式
        sanitized = sanitized.replace('..', '_')
        # 限制长度
        return sanitized[:100]

    def _get_profile_path(self, group_id: str) -> str:
        safe_id = self._sanitize_id(group_id)
        return f"group_persona:{safe_id}"

    def _load_profile(self, group_id: str) -> Optional[GroupPersonaProfile]:
        try:
            data = load_slot(self._get_profile_path(group_id))
            return GroupPersonaProfile(**data) if isinstance(data, dict) else None
        except Exception as e:
            logger.warning(f"加载群画像失败 {group_id}: {e}")
            return None

    def _save_profile(self, profile: GroupPersonaProfile) -> None:
        try:
            save_slot(
                self._get_profile_path(profile.group_id),
                asdict(profile),
                ttl_days=3650,
            )
        except Exception as e:
            logger.error(f"保存群画像失败 {profile.group_id}: {e}")

    def _load_all_profiles(self) -> None:
        try:
            prefix = "group_persona:"
            for slot_key in list_slot_keys(prefix):
                group_id = slot_key.removeprefix(prefix)
                profile = self._load_profile(group_id)
                if profile:
                    self._profiles[profile.group_id or group_id] = profile
                    logger.info(f"已加载群画像: {profile.group_id or group_id}")
        except Exception as e:
            logger.error(f"加载所有群画像失败: {e}")

    def get_profile(
        self, group_id: str, create_if_missing: bool = True
    ) -> Optional[GroupPersonaProfile]:
        if group_id in self._profiles:
            return self._profiles[group_id]
        if create_if_missing:
            profile = GroupPersonaProfile(
                group_id=group_id,
                created_at=time.time(),
                updated_at=time.time(),
            )
            self._profiles[group_id] = profile
            self._save_profile(profile)
            return profile
        return None

    def should_update(self, group_id: str) -> bool:
        profile = self.get_profile(group_id, create_if_missing=False)
        if not profile:
            return True
        if not profile.auto_update_enabled:
            return False
        return time.time() - profile.updated_at >= profile.update_interval

    async def update_profile(
        self,
        group_id: str,
        group_name: str = "",
        member_count: int = 0,
        active_members: List[str] = None,
    ) -> GroupPersonaProfile:
        profile = self.get_profile(group_id)
        if not profile:
            return profile

        profile.group_name = group_name
        if member_count > 0:
            profile.member_count = member_count
        if active_members:
            profile.active_members = active_members

        try:
            from src.hippo_memorizer.summary_storage import (
                get_summary_storage_manager,
            )

            storage_mgr = get_summary_storage_manager()
            recent_summaries = storage_mgr.get_summaries(group_id, limit=20)
            recent_topics = [str(item.topic or "").strip() for item in recent_summaries if str(item.topic or "").strip()]
            profile.topics_of_interest = list(set(recent_topics))[:10]
        except Exception as e:
            logger.debug(f"获取群话题失败: {e}")

        try:
            from src.chat.heart_flow.energy_manager import EnergyChainDimension

            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(group_id)
            _sv = float(_ch.social_value) if _ch else 0.0
            _cv = float(_ch.chat_pool) if _ch else 50.0
            culture_keywords = []
            if _sv > 30:
                culture_keywords.append("活跃度高，大家爱水群")
            if _sv < -20:
                culture_keywords.append("氛围比较冷淡")
            if _cv > 60:
                culture_keywords.append("聊天氛围热烈")
            if _cv < 20:
                culture_keywords.append("平时比较安静")
            profile.group_culture = culture_keywords
        except Exception:
            logger.debug(f"获取群共享值失败({group_id})，跳过群氛围设置")

        try:
            from src.chat.heart_flow.social_value_dim import SocialValueDimension

            engine = SocialValueDimension.get_instance()
            recent_score = await engine.read_score("bot", group_id)
            if recent_score > 50:
                profile.personality_traits = ["氛围好", "大家相处融洽"]
            elif recent_score < -20:
                profile.personality_traits = ["气氛紧张", "需要小心"]
            else:
                profile.personality_traits = ["普通群聊"]
        except Exception as e:
            logger.debug(f"读取结算分数失败({group_id}): {e}")

        profile.summary = self._generate_summary(profile)
        profile.updated_at = time.time()
        self._save_profile(profile)

        logger.info(
            f"已更新群画像: {group_id}, 下次更新: {
                profile.updated_at +
                profile.update_interval}"
        )
        return profile

    def _generate_summary(self, profile: GroupPersonaProfile) -> str:
        parts = []
        if profile.group_name:
            parts.append(f"群名: {profile.group_name}")
        if profile.topics_of_interest:
            topics_str = "、".join(profile.topics_of_interest[:5])
            parts.append(f"常聊话题: {topics_str}")
        if profile.group_culture:
            culture_str = "，".join(profile.group_culture)
            parts.append(f"群氛围: {culture_str}")
        if profile.member_count > 0:
            parts.append(f"成员数: {profile.member_count}")
        if profile.communication_style:
            style_str = "、".join(profile.communication_style[:3])
            parts.append(f"沟通风格: {style_str}")
        if profile.personality_traits:
            traits_str = "，".join(profile.personality_traits)
            parts.append(f"群特点: {traits_str}")
        return "；".join(parts) if parts else "这是一个普通的群聊"

    def get_summary(self, group_id: str) -> str:
        profile = self.get_profile(group_id, create_if_missing=False)
        if not profile:
            return ""
        return profile.summary

    def get_perspective_hint(
        self, group_id: str, is_private: bool = False
    ) -> str:
        if is_private:
            return "你就是她的好朋友，和她私聊时像现实中闺蜜聊天一样，轻松自然想聊什么聊什么。"

        profile = self.get_profile(group_id, create_if_missing=False)
        if not profile:
            return (
                "你就是这个群里的一份子，和大家一样水群聊天，别把自己当外人。"
            )

        hints = ["你就是这个群里的普通一员"]
        if profile.group_culture:
            hints.append(f"群里{profile.group_culture[0]}")
        if profile.personality_traits:
            hints.append(f"大家{profile.personality_traits[0]}")
        hints.append("像平时水群一样自然聊天就行。")
        return "".join(hints)

    def set_update_interval(self, group_id: str, days: float) -> bool:
        profile = self.get_profile(group_id, create_if_missing=False)
        if profile:
            profile.update_interval = 86400.0 * days
            self._save_profile(profile)
            return True
        return False

    def enable_auto_update(self, group_id: str, enabled: bool = True) -> bool:
        profile = self.get_profile(group_id, create_if_missing=False)
        if profile:
            profile.auto_update_enabled = enabled
            self._save_profile(profile)
            return True
        return False


_global_manager: Optional[GroupPersonaManager] = None


def get_group_persona_manager() -> GroupPersonaManager:
    global _global_manager
    if _global_manager is None:
        _global_manager = GroupPersonaManager()
    return _global_manager

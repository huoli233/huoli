import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.common.logger import get_logger
from src.modules.perception.runtime_config import perception_module_view

logger = get_logger("用户关系感知")


@dataclass
class RelationInfo:
    """关系信息"""

    user_id: str = ""
    user_name: str = ""
    relation_tag: str = ""
    intimacy_level: float = 0.0
    interaction_count: int = 0
    last_interaction_time: float = 0.0
    trust_level: float = 0.5
    relation_type: str = "陌生人"
    notes: List[str] = field(default_factory=list)


@dataclass
class RelationBrief:
    """关系简报"""

    user_id: str = ""
    display_name: str = ""
    relation_label: str = ""
    is_familiar: bool = False
    is_newcomer: bool = False
    is_trusted: bool = False


@dataclass
class RelationStats:
    """关系统计"""

    total_users: int = 0
    familiar_users: int = 0
    newcomers: int = 0
    trusted_users: int = 0
    avg_intimacy: float = 0.0


class UserRelationSense:
    """用户关系感知 - 获取用户关系数据并格式化为简明标签"""

    def __init__(self, config_engine: Optional[Any] = None):
        del config_engine
        self._relation_cache: Dict[str, RelationInfo] = {}
        self._interaction_counts: Dict[str, Dict[str, int]] = {}
        self._last_interaction_times: Dict[str, Dict[str, float]] = {}
        self._adapter_get_relation: Optional[Callable] = None
        self._load_config()

    def _load_config(self):
        """加载配置参数"""
        config = perception_module_view("perception_user_relation")
        self._familiar_threshold = int(
            config.get("familiar_threshold", 5)
        )
        self._newcomer_window_hours = float(
            config.get("newcomer_window_hours", 24)
        )
        self._trust_threshold = float(config.get("trust_threshold", 0.7))
        self._intimacy_decay_rate = float(
            config.get("intimacy_decay_rate", 0.01)
        )
        self._cache_ttl_seconds = float(
            config.get("cache_ttl_seconds", 3600)
        )
        self._max_cache_size = int(config.get("max_cache_size", 1000))

    def set_adapter(self, adapter_func: Callable):
        """
        设置适配器函数

        Args:
            adapter_func: 异步函数，签名为 async (user_id: str, stream_id: str) -> str
        """
        self._adapter_get_relation = adapter_func

    async def get_briefs(
        self,
        user_ids: List[str],
        sender_names: Dict[str, str],
        stream_id: str,
    ) -> Dict[str, str]:
        """
        获取用户关系简报

        Args:
            user_ids: 用户ID列表
            sender_names: 用户ID到名称的映射
            stream_id: 流ID

        Returns:
            用户ID到关系标签的映射
        """
        relations: Dict[str, str] = {}

        for uid in user_ids:
            if not uid:
                continue
            try:
                relation_tag = await self._get_relation_tag(uid, stream_id)
                name = sender_names.get(uid, "用户")
                if relation_tag:
                    relations[uid] = f"{name}({relation_tag})"
                else:
                    relations[uid] = name
            except Exception as e:
                name = sender_names.get(uid, "用户")
                relations[uid] = name
                logger.debug(f"获取用户关系失败 uid={uid}: {e}")

        return relations

    async def _get_relation_tag(self, user_id: str, stream_id: str) -> str:
        """获取关系标签"""
        cache_key = f"{stream_id}:{user_id}"
        cached = self._relation_cache.get(cache_key)
        if (
            cached
            and time.time() - cached.last_interaction_time
            < self._cache_ttl_seconds
        ):
            return cached.relation_tag

        if self._adapter_get_relation:
            try:
                tag = await self._adapter_get_relation(user_id, stream_id)
                if tag:
                    self._update_cache(cache_key, user_id, tag)
                    return tag
            except Exception as e:
                logger.debug(f"适配器获取关系失败: {e}")

        return self._infer_relation_tag(user_id, stream_id)

    def _infer_relation_tag(self, user_id: str, stream_id: str) -> str:
        """推断关系标签"""
        interaction_count = self._get_interaction_count(user_id, stream_id)
        last_time = self._get_last_interaction_time(user_id, stream_id)
        now = time.time()

        if interaction_count == 0:
            return "新人"

        hours_since_last = (now - last_time) / 3600 if last_time > 0 else 999

        if interaction_count >= self._familiar_threshold:
            if hours_since_last < 24:
                return "熟人"
            elif hours_since_last < 168:
                return "老朋友"
            else:
                return "久违的朋友"
        elif interaction_count >= 2:
            if hours_since_last < self._newcomer_window_hours:
                return "新朋友"
            else:
                return "认识的人"

        return "新人"

    def _update_cache(self, cache_key: str, user_id: str, tag: str):
        """更新缓存"""
        if len(self._relation_cache) >= self._max_cache_size:
            oldest_key = min(
                self._relation_cache.keys(),
                key=lambda k: self._relation_cache[k].last_interaction_time,
            )
            del self._relation_cache[oldest_key]

        self._relation_cache[cache_key] = RelationInfo(
            user_id=user_id,
            relation_tag=tag,
            last_interaction_time=time.time(),
        )

    def record_interaction(
        self, user_id: str, stream_id: str, interaction_type: str = "message"
    ):
        """
        记录交互

        Args:
            user_id: 用户ID
            stream_id: 流ID
            interaction_type: 交互类型
        """
        if stream_id not in self._interaction_counts:
            if len(self._interaction_counts) >= 10000:
                oldest_stream = min(
                    self._last_interaction_times,
                    key=lambda s: max(
                        self._last_interaction_times[s].values(), default=0
                    ),
                )
                self._interaction_counts.pop(oldest_stream, None)
                self._last_interaction_times.pop(oldest_stream, None)
            self._interaction_counts[stream_id] = {}
        self._interaction_counts[stream_id][user_id] = (
            self._interaction_counts[stream_id].get(user_id, 0) + 1
        )

        if stream_id not in self._last_interaction_times:
            self._last_interaction_times[stream_id] = {}
        self._last_interaction_times[stream_id][user_id] = time.time()

        cache_key = f"{stream_id}:{user_id}"
        if cache_key in self._relation_cache:
            self._relation_cache[cache_key].interaction_count += 1
            self._relation_cache[cache_key].last_interaction_time = time.time()

    def _get_interaction_count(self, user_id: str, stream_id: str) -> int:
        """获取交互次数"""
        return self._interaction_counts.get(stream_id, {}).get(user_id, 0)

    def _get_last_interaction_time(
        self, user_id: str, stream_id: str
    ) -> float:
        """获取上次交互时间"""
        return self._last_interaction_times.get(stream_id, {}).get(
            user_id, 0.0
        )

    def get_relation_info(self, user_id: str, stream_id: str) -> RelationInfo:
        """获取详细关系信息"""
        cache_key = f"{stream_id}:{user_id}"
        cached = self._relation_cache.get(cache_key)

        if cached:
            return cached

        interaction_count = self._get_interaction_count(user_id, stream_id)
        last_time = self._get_last_interaction_time(user_id, stream_id)
        tag = self._infer_relation_tag(user_id, stream_id)

        return RelationInfo(
            user_id=user_id,
            relation_tag=tag,
            interaction_count=interaction_count,
            last_interaction_time=last_time,
            intimacy_level=min(1.0, interaction_count / 20.0),
            relation_type=self._get_relation_type(interaction_count),
        )

    def _get_relation_type(self, interaction_count: int) -> str:
        """获取关系类型"""
        if interaction_count == 0:
            return "陌生人"
        elif interaction_count < 3:
            return "点头之交"
        elif interaction_count < 10:
            return "普通朋友"
        elif interaction_count < 30:
            return "好朋友"
        else:
            return "密友"

    def get_relation_brief(
        self, user_id: str, stream_id: str, user_name: str = ""
    ) -> RelationBrief:
        """获取关系简报"""
        info = self.get_relation_info(user_id, stream_id)
        now = time.time()

        is_newcomer = (
            info.interaction_count < 3
            and (now - info.last_interaction_time)
            < self._newcomer_window_hours * 3600
        )

        return RelationBrief(
            user_id=user_id,
            display_name=user_name or user_id,
            relation_label=info.relation_tag,
            is_familiar=info.interaction_count >= self._familiar_threshold,
            is_newcomer=is_newcomer,
            is_trusted=info.trust_level >= self._trust_threshold,
        )

    def get_stream_stats(self, stream_id: str) -> RelationStats:
        """获取流关系统计"""
        counts = self._interaction_counts.get(stream_id, {})
        times = self._last_interaction_times.get(stream_id, {})
        now = time.time()

        total = len(counts)
        familiar = sum(
            1 for c in counts.values() if c >= self._familiar_threshold
        )
        newcomers = sum(
            1
            for uid, c in counts.items()
            if c < 3
            and (now - times.get(uid, 0)) < self._newcomer_window_hours * 3600
        )
        trusted = sum(1 for uid in counts if self._is_trusted(uid, stream_id))

        avg_intimacy = 0.0
        if counts:
            avg_intimacy = sum(
                min(1.0, c / 20.0) for c in counts.values()
            ) / len(counts)

        return RelationStats(
            total_users=total,
            familiar_users=familiar,
            newcomers=newcomers,
            trusted_users=trusted,
            avg_intimacy=avg_intimacy,
        )

    def _is_trusted(self, user_id: str, stream_id: str) -> bool:
        """判断是否可信用户"""
        info = self.get_relation_info(user_id, stream_id)
        return info.trust_level >= self._trust_threshold

    def update_trust(self, user_id: str, stream_id: str, delta: float):
        """
        更新信任度

        Args:
            user_id: 用户ID
            stream_id: 流ID
            delta: 信任度变化量 (-1.0 到 1.0)
        """
        cache_key = f"{stream_id}:{user_id}"
        if cache_key in self._relation_cache:
            old_trust = self._relation_cache[cache_key].trust_level
            self._relation_cache[cache_key].trust_level = max(
                0.0, min(1.0, old_trust + delta)
            )

    def add_note(self, user_id: str, stream_id: str, note: str):
        """添加用户备注"""
        cache_key = f"{stream_id}:{user_id}"
        if cache_key in self._relation_cache:
            self._relation_cache[cache_key].notes.append(note)

    def clear_cache(self, stream_id: Optional[str] = None):
        """清除缓存"""
        if stream_id:
            keys_to_remove = [
                k
                for k in self._relation_cache
                if k.startswith(f"{stream_id}:")
            ]
            for k in keys_to_remove:
                del self._relation_cache[k]
        else:
            self._relation_cache.clear()


_user_relation_sense_instance: Optional[UserRelationSense] = None


def get_user_relation_sense(
    config_engine: Optional[Any] = None,
) -> UserRelationSense:
    """获取用户关系感知单例"""
    global _user_relation_sense_instance
    if _user_relation_sense_instance is None:
        _user_relation_sense_instance = UserRelationSense(config_engine)
    return _user_relation_sense_instance

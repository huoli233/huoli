import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.common.config.config_engine import ConfigEngine
from src.common.logger import get_logger

logger = get_logger("dimension_collector")


@dataclass
class DimensionFactors:
    """撤回决策的多维度因素"""

    social_value: float = 0.0
    relationship_depth: str = "陌生"
    trust_value: float = 0.0
    annoyance_value: float = 0.0
    interaction_count: int = 0
    trauma_score: float = 0.0
    inner_chaos: float = 0.0
    emotional_state: str = "平静"
    is_private: bool = False
    group_atmosphere: str = "普通"
    group_member_count: int = 0
    bystander_effect: float = 0.0
    time_of_day: str = "白天"
    time_elapsed: float = 0.0
    recent_typo_count: int = 0
    recent_recall_count: int = 0
    message_importance: float = 0.5
    topic_type: str = "casual"
    content_sensitivity: float = 0.0
    user_reaction: str = ""
    reaction_type: str = "none"
    chat_value: float = 100.0
    thinking_ratio: float = 1.0
    activity_level: float = 50.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "social_value": self.social_value,
            "relationship_depth": self.relationship_depth,
            "trust_value": self.trust_value,
            "annoyance_value": self.annoyance_value,
            "interaction_count": self.interaction_count,
            "trauma_score": self.trauma_score,
            "inner_chaos": self.inner_chaos,
            "emotional_state": self.emotional_state,
            "is_private": self.is_private,
            "group_atmosphere": self.group_atmosphere,
            "group_member_count": self.group_member_count,
            "bystander_effect": self.bystander_effect,
            "time_of_day": self.time_of_day,
            "time_elapsed": self.time_elapsed,
            "recent_typo_count": self.recent_typo_count,
            "recent_recall_count": self.recent_recall_count,
            "message_importance": self.message_importance,
            "topic_type": self.topic_type,
            "content_sensitivity": self.content_sensitivity,
            "user_reaction": self.user_reaction,
            "reaction_type": self.reaction_type,
            "chat_value": self.chat_value,
            "thinking_ratio": self.thinking_ratio,
            "activity_level": self.activity_level,
        }


class DimensionCollector:
    """维度收集器

    收集撤回决策所需的多维度因素：
    - 社交维度：社交值、关系深度、信任值、厌烦值、交互次数
    - 心理维度：创伤分数、内心混乱、情绪状态
    - 场景维度：是否私聊、群氛围、群人数、旁观者效应
    - 时间维度：时间段、经过时间、最近打错字次数、最近撤回次数
    - 内容维度：消息重要性、话题类型、内容敏感度
    - 反应维度：用户反应、反应类型
    """

    def __init__(self, config_engine: Optional[ConfigEngine] = None):
        self._config = config_engine or ConfigEngine.get_instance()
        self._typo_history: Dict[str, List[float]] = {}
        self._recall_history: Dict[str, List[float]] = {}
        logger.info("维度收集器初始化完成")

    async def collect(
        self,
        user_id: str,
        channel_id: str,
        stream_id: str,
        content: str,
        time_elapsed: float = 0.0,
        user_reaction: str = "",
    ) -> DimensionFactors:
        """收集所有维度因素"""
        factors = DimensionFactors()

        await self._collect_social_factors(factors, user_id, stream_id)
        await self._collect_psychological_factors(factors, stream_id)
        await self._collect_scene_factors(factors, stream_id)
        await self._collect_time_factors(factors, stream_id, time_elapsed)
        await self._collect_content_factors(factors, content)
        await self._collect_reaction_factors(factors, user_reaction)

        return factors

    async def _collect_social_factors(
        self,
        factors: DimensionFactors,
        user_id: str,
        stream_id: str,
    ) -> None:
        """收集社交维度因素"""
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            tracker = get_emotion_tracker(stream_id)
            state = tracker.get_user_state(user_id, create_if_missing=False)
            if state:
                factors.social_value = float(state.affection or 0)
                factors.annoyance_value = float(state.annoyance or 0)
                factors.interaction_count = int(state.interaction_count or 0)
        except Exception as exc:
            logger.debug(f"收集社交因素失败: {exc}")

        try:
            from src.chat.heart_flow.fondness_trust import FondnessTrustDimension
            from src.chat.heart_flow.social_value_dim import SocialValueDimension

            _fuser = FondnessTrustDimension.get_instance()
            _dossier = _fuser.get_dossier(user_id, stream_id)
            if _dossier:
                _latest = 0.0
                if _dossier.recent_values:
                    _latest = _dossier.recent_values[-1].value
                factors.trust_value = (
                    float(
                        _latest * _fuser.get_phase_weight(user_id, stream_id)
                    )
                    if _latest >= 0
                    else 0.0
                )
                level = _dossier.current_phase
                if level >= 4:
                    factors.relationship_depth = "很熟悉"
                elif level >= 3:
                    factors.relationship_depth = "比较熟悉"
                elif level >= 2:
                    factors.relationship_depth = "有些熟悉"
                elif level >= 1:
                    factors.relationship_depth = "有点印象"
                else:
                    factors.relationship_depth = "陌生"
        except Exception as exc:
            logger.debug(f"收集关系因素失败: {exc}")

    async def _collect_psychological_factors(
        self,
        factors: DimensionFactors,
        stream_id: str,
    ) -> None:
        """收集心理维度因素"""
        try:
            from src.chat.heart_flow.energy_manager import EnergyChainDimension

            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(stream_id)
            if _ch:
                factors.chat_value = float(_ch.chat_pool)
                factors.activity_level = float(_ch.activity_level)
                factors.social_value = float(_ch.social_value)
                _tv = float(_ch.thinking_value)
                _tc = float(_ch.thinking_ceiling)
                factors.thinking_ratio = max(0.0, min(1.0, _tv / max(_tc, 1.0)))
            else:
                factors.thinking_ratio = 1.0

            if factors.social_value > 50:
                factors.emotional_state = "开心"
            elif factors.social_value > 20:
                factors.emotional_state = "不错"
            elif factors.social_value > 0:
                factors.emotional_state = "平静"
            elif factors.social_value > -20:
                factors.emotional_state = "有点烦"
            else:
                factors.emotional_state = "不爽"
        except Exception as exc:
            logger.debug(f"收集心理因素失败: {exc}")

        try:
            from src.chat.heart_flow.trauma_fabric import TraumaDimension

            trauma = TraumaDimension.get_instance()
            factors.trauma_score = float(
                trauma.get_trauma_score(stream_id) or 0
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    async def _collect_scene_factors(
        self,
        factors: DimensionFactors,
        stream_id: str,
    ) -> None:
        """收集场景维度因素"""
        try:
            if "private" in stream_id.lower():
                factors.is_private = True
            else:
                factors.is_private = False
        except Exception as _e:
            logger.debug(f"异常: {_e}")

        try:
            from src.chat.heart_flow.energy_manager import EnergyChainDimension

            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(stream_id)
            activity = float(_ch.activity_level) if _ch else 50.0

            if activity > 70:
                factors.group_atmosphere = "活跃"
            elif activity > 40:
                factors.group_atmosphere = "普通"
            else:
                factors.group_atmosphere = "冷清"
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    async def _collect_time_factors(
        self,
        factors: DimensionFactors,
        stream_id: str,
        time_elapsed: float,
    ) -> None:
        """收集时间维度因素"""
        now = time.time()
        hour = time.localtime().tm_hour

        if 6 <= hour < 12:
            factors.time_of_day = "上午"
        elif 12 <= hour < 14:
            factors.time_of_day = "中午"
        elif 14 <= hour < 18:
            factors.time_of_day = "下午"
        elif 18 <= hour < 22:
            factors.time_of_day = "晚上"
        elif 22 <= hour < 24:
            factors.time_of_day = "深夜"
        else:
            factors.time_of_day = "凌晨"

        factors.time_elapsed = time_elapsed

        typo_history = self._typo_history.get(stream_id, [])
        typo_history = [t for t in typo_history if now - t < 3600]
        self._typo_history[stream_id] = typo_history
        factors.recent_typo_count = len(typo_history)

        recall_history = self._recall_history.get(stream_id, [])
        recall_history = [t for t in recall_history if now - t < 3600]
        self._recall_history[stream_id] = recall_history
        factors.recent_recall_count = len(recall_history)

    async def _collect_content_factors(
        self,
        factors: DimensionFactors,
        content: str,
    ) -> None:
        """收集内容维度因素"""
        if not content:
            return

        length = len(content)
        if length < 10:
            factors.message_importance = 0.3
        elif length < 50:
            factors.message_importance = 0.5
        else:
            factors.message_importance = 0.7

        question_marks = content.count("?") + content.count("？")
        if question_marks > 0:
            factors.topic_type = "question"
            factors.message_importance = max(factors.message_importance, 0.7)

        serious_keywords = [
            "工作",
            "学习",
            "帮助",
            "问题",
            "错误",
            "bug",
            "重要",
        ]
        if any(kw in content for kw in serious_keywords):
            factors.topic_type = "serious"
            factors.message_importance = 0.9

        sensitive_keywords = ["密码", "账号", "隐私", "秘密"]
        if any(kw in content for kw in sensitive_keywords):
            factors.content_sensitivity = 0.8

    async def _collect_reaction_factors(
        self,
        factors: DimensionFactors,
        user_reaction: str,
    ) -> None:
        """收集反应维度因素"""
        if not user_reaction:
            factors.reaction_type = "none"
            return

        factors.user_reaction = user_reaction

        negative_keywords = [
            "错",
            "不对",
            "不是",
            "什么",
            "？",
            "?",
            "啊",
            "哈",
        ]
        positive_keywords = ["好", "对", "嗯", "哦", "行"]

        negative_count = sum(
            1 for kw in negative_keywords if kw in user_reaction
        )
        positive_count = sum(
            1 for kw in positive_keywords if kw in user_reaction
        )

        if negative_count > positive_count:
            factors.reaction_type = "negative"
        elif positive_count > negative_count:
            factors.reaction_type = "positive"
        else:
            factors.reaction_type = "neutral"

    def record_typo(self, stream_id: str) -> None:
        """记录打错字事件"""
        if stream_id not in self._typo_history:
            self._typo_history[stream_id] = []
        self._typo_history[stream_id].append(time.time())

    def record_recall(self, stream_id: str) -> None:
        """记录撤回事件"""
        if stream_id not in self._recall_history:
            self._recall_history[stream_id] = []
        self._recall_history[stream_id].append(time.time())


_dimension_collector: Optional[DimensionCollector] = None


def get_dimension_collector(
    config_engine: Optional[ConfigEngine] = None,
) -> DimensionCollector:
    """获取维度收集器单例"""
    global _dimension_collector
    if _dimension_collector is None:
        _dimension_collector = DimensionCollector(config_engine)
    return _dimension_collector

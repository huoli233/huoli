import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from src.common.config.config_engine import (
    ConfigEngine,
    get_default_config_engine,
)
from src.common.logger import get_logger
from src.modules.social_value.settlement_engine import (
    SettlementEngine,
    SettlementReport,
)
from src.modules.social_value.phase_tracker import (
    PhaseTracker,
    RelationPhase,
    PhaseTransition,
    PersonaImpression,
    BondDossier,
    PHASE_WEIGHT,
    PHASE_DESCRIPTION,
    get_phase_tracker,
)
from src.modules.social_value.social_calculator import SocialCalculator
from src.modules.social_value.social_storage import SocialStorage
from src.modules.social_value.models import SocialUpdateResult

logger = get_logger("社交影响融合")


@dataclass
class AffectSnapshot:
    """社交情感完整快照

    所有消费方统一从此结构读取社交状态，
    不再需要分别访问 SocialValueCore / RelationshipTracker / BondSettler。
    """

    # 层1：量化真值
    social_score: float = 0.0
    # 层2：关系阶段
    phase: int = RelationPhase.ACQUAINTANCE
    phase_label: str = ""
    phase_weight: float = 0.8
    trend: str = "稳定"
    # 层2：印象
    impression_nick: str = ""
    impression_rationale: str = ""
    character_tags: List[str] = field(default_factory=list)
    # 层3：情绪（由外部情绪追踪器注入，融合器只中继）
    emotion_valence: float = 0.0
    emotion_label: str = ""
    # 元数据
    user_id: str = ""
    channel_id: str = ""
    captured_at: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """转为字典（供 world_snapshot 等消费方直接使用）"""
        return {
            "social_score": round(self.social_score, 4),
            "phase": self.phase,
            "phase_label": self.phase_label,
            "phase_weight": self.phase_weight,
            "trend": self.trend,
            "impression_nick": self.impression_nick,
            "impression_rationale": self.impression_rationale,
            "character_tags": self.character_tags,
            "emotion_valence": round(self.emotion_valence, 4),
            "emotion_label": self.emotion_label,
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "captured_at": self.captured_at,
        }

    @property
    def favorability(self) -> float:
        """兼容旧消费方读取好感度"""
        return self.social_score

    @property
    def trust_value(self) -> float:
        """兼容旧消费方读取信任值（按阶段权重折算）"""
        return self.social_score * self.phase_weight

    @property
    def relationship_stage(self) -> str:
        """兼容旧消费方读取关系阶段文本"""
        return self.phase_label or PHASE_DESCRIPTION.get(self.phase, "有些熟悉")


@dataclass
class FuserSettlementResult:
    """融合结算结果（同时包含量化报告与阶段变化）"""

    settlement: SettlementReport
    transition: Optional[PhaseTransition] = None
    snapshot: Optional[AffectSnapshot] = None


class SocialAffectFuser:
    """社交情感融合器

    统一入口，将 Layer 1（量化结算）、Layer 2（阶段追踪）、
    以及可选的 Layer 3（情绪中继）串联为单一调用链。

    所有消费方只需依赖此类即可获取完整的社交情感状态。
    """

    _solo: Optional["SocialAffectFuser"] = None

    @classmethod
    def instance(cls) -> "SocialAffectFuser":
        """获取全局单例"""
        if cls._solo is None:
            cls._solo = cls()
        return cls._solo

    @classmethod
    def teardown(cls) -> None:
        """清理单例（测试 / 关机用）"""
        cls._solo = None

    def __init__(
        self,
        calculator: Optional[SocialCalculator] = None,
        storage: Optional[SocialStorage] = None,
        config_engine: Optional[ConfigEngine] = None,
    ):
        cfg = config_engine or get_default_config_engine()
        calc = calculator or SocialCalculator(cfg)
        store = storage or SocialStorage()
        self._settlement = SettlementEngine(calc, store, cfg)
        self._phase = PhaseTracker(cfg)
        self._emotion_reader = None
        self._cfg_hub = cfg

    def bind_emotion_reader(self, reader) -> None:
        """绑定外部 EmotionTracker 实例（Layer 3 情绪中继）"""
        self._emotion_reader = reader

    def bind_adapter(self, adapter) -> None:
        """绑定持久化适配器（转发到 PhaseTracker）"""
        self._phase.attach_adapter(adapter)

    # ================================================================
    #  核心调用链
    # ================================================================

    async def evaluate(
        self,
        user_id: str,
        channel_id: str,
        content: str,
        context: Dict[str, Any],
    ) -> FuserSettlementResult:
        """完整评估：文本内容 → 量化结算 → 阶段更新 → 快照生成

        这是最常用的入口，适合在聊天主循环中调用。
        """
        # 层1：量化结算
        report = await self._settlement.settle_event(user_id, channel_id, content, context)
        # 层2：将新分值注入阶段追踪
        transition = self._phase.ingest_score(user_id, channel_id, report.settled_score)
        # 组装快照
        snap = self._assemble_snapshot(user_id, channel_id, report.settled_score)
        return FuserSettlementResult(
            settlement=report,
            transition=transition,
            snapshot=snap,
        )

    async def evaluate_from_behavior(
        self,
        user_id: str,
        channel_id: str,
        behavior: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> FuserSettlementResult:
        """从已识别的行为结构体触发评估（跳过内容分析阶段）"""
        ctx = context or {}
        report = await self._settlement.settle_from_behavior(user_id, channel_id, behavior, ctx)
        transition = self._phase.ingest_score(user_id, channel_id, report.settled_score)
        snap = self._assemble_snapshot(user_id, channel_id, report.settled_score)
        return FuserSettlementResult(
            settlement=report,
            transition=transition,
            snapshot=snap,
        )

    # ================================================================
    #  只读查询
    # ================================================================

    async def get_snapshot(self, user_id: str, channel_id: str) -> AffectSnapshot:
        """只读获取当前快照（不触发结算）"""
        score = await self._settlement.read_score(user_id, channel_id)
        return self._assemble_snapshot(user_id, channel_id, score)

    async def read_full_record(self, user_id: str, channel_id: str):
        """只读获取完整社交值记录（含正负维度、互动次数等扩展字段）"""
        return await self._settlement.read_full_record(user_id, channel_id)

    async def get_score(self, user_id: str, channel_id: str) -> float:
        """只读取社交分值"""
        return await self._settlement.read_score(user_id, channel_id)

    def get_phase(self, user_id: str, channel_id: str) -> int:
        """只读取当前关系阶段"""
        return self._phase.current_phase_of(user_id, channel_id)

    def get_phase_label(self, user_id: str, channel_id: str) -> str:
        """只读取可读阶段描述"""
        return self._phase.readable_phase(user_id, channel_id)

    def get_phase_weight(self, user_id: str, channel_id: str) -> float:
        """只读取阶段权重因子"""
        return self._phase.phase_weight_of(user_id, channel_id)

    def get_trend(self, user_id: str, channel_id: str) -> str:
        """只读取关系趋势"""
        return self._phase.trend_of(user_id, channel_id)

    def get_dossier(self, user_id: str, channel_id: str) -> BondDossier:
        """获取完整关系档案"""
        return self._phase.fetch_dossier(user_id, channel_id)

    def get_brief_summary(self, user_id: str, channel_id: str) -> str:
        """获取用户摘要文本"""
        return self._phase.brief_summary(user_id, channel_id)

    # ================================================================
    #  互动与印象管理
    # ================================================================

    def record_peer_interaction(self, user_id: str, channel_id: str, peer_uid: str, weight: float = 1.0) -> None:
        """记录两用户之间的互动"""
        self._phase.log_peer_interaction(user_id, channel_id, peer_uid, weight)

    def top_peers_of(self, user_id: str, channel_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """获取最亲密的互动伙伴"""
        return self._phase.top_peers(user_id, channel_id, limit)

    async def refresh_impression(
        self,
        user_id: str,
        channel_id: str,
        interaction_digest: str = "",
        topics: Optional[List[str]] = None,
        style: str = "",
    ) -> Optional[PersonaImpression]:
        """刷新用户印象"""
        return await self._phase.refresh_impression(user_id, channel_id, interaction_digest, topics, style)

    def all_nicks_in_channel(self, channel_id: str) -> Dict[str, str]:
        """获取频道内所有用户标签"""
        return self._phase.all_nicks_in_channel(channel_id)

    def evaluate_group_support_pressure(
        self,
        channel_id: str,
        recent_messages: List[Any],
        bot_user_id: str = "bot",
    ) -> Tuple[float, float]:
        """评估群环境对负面情绪/心理压力的调制效应

        设计文档第六章第11/12条联动规则：
        - 群里多人支持机器人时 → 负面情绪应缓和
        - 多人围攻时 → 心理压力应上升

        Args:
            channel_id: 频道ID
            recent_messages: 最近消息列表（建议20-50条）
            bot_user_id: 机器人用户ID

        Returns:
            Tuple[float, float]: (负面情绪缓和系数, 压力上升系数)
            - 缓和系数: 0.0-0.3，每条支持消息减少5%负面，上限30%
            - 压力上升系数: 0.0-0.4，每条攻击消息增加8%压力，上限40%
        """
        _support_keywords = [
            "支持",
            "说得对",
            "同意",
            "+1",
            "确实",
            "有道理",
            "正确",
            "没错",
            "就是",
            "我也觉得",
            "赞同",
            "顶",
            "好",
            "棒",
        ]
        _attack_keywords = [
            "傻",
            "烦",
            "滚",
            "闭嘴",
            "有病",
            "恶心",
            "讨厌",
            "去死",
            "废物",
            "垃圾",
            "智障",
            "脑残",
            "白痴",
            "蠢",
            "笨",
        ]
        _support_count = 0
        _attack_count = 0
        _recent_texts = []
        for _msg in recent_messages[-30:]:
            _uid = str(getattr(_msg, "user_id", "") or "").strip()
            if not _uid or _uid == bot_user_id:
                continue
            _text = (
                str(
                    getattr(_msg, "processed_plain_text", "")
                    or getattr(_msg, "plain_text", "")
                    or getattr(_msg, "content", "")
                    or ""
                )
                .lower()
                .strip()
            )
            if not _text or len(_text) < 2:
                continue
            _recent_texts.append(_text)
            for _kw in _support_keywords:
                if _kw in _text:
                    _support_count += 1
                    break
            for _kw in _attack_keywords:
                if _kw in _text:
                    _attack_count += 1
                    break
        _support_relief = min(0.30, _support_count * 0.05)
        _pressure_rise = min(0.40, _attack_count * 0.08)
        if _support_count > 0 or _attack_count > 0:
            logger.debug(
                f"[群环境调制] channel={channel_id[:8]} "
                f"支持={_support_count}条(缓和{_support_relief:.0%}) "
                f"攻击={_attack_count}条(压力+{_pressure_rise:.0%})"
            )
        return _support_relief, _pressure_rise

    def apply_group_modulation_to_emotion(
        self,
        channel_id: str,
        current_annoyance: float,
        current_pressure: float,
        recent_messages: List[Any],
        bot_user_id: str = "bot",
    ) -> Tuple[float, float]:
        """将群环境调制效应应用到当前情绪/压力值

        Args:
            channel_id: 频道ID
            current_annoyance: 当前厌烦值 (0-100)
            current_pressure: 当前心理压力 (0-100)
            recent_messages: 最近消息列表
            bot_user_id: 机器人用户ID

        Returns:
            Tuple[float, float]: (调制后厌烦值, 调制后压力值)
        """
        _relief, _pressure_rise = self.evaluate_group_support_pressure(channel_id, recent_messages, bot_user_id)
        _mod_annoyance = max(0.0, current_annoyance * (1.0 - _relief))
        _mod_pressure = min(100.0, current_pressure + _pressure_rise * 100.0)
        return _mod_annoyance, _mod_pressure

    # ================================================================
    #  衰减与维护
    # ================================================================

    async def apply_time_decay(self, user_id: str, channel_id: str, hours: float) -> float:
        """单独执行时间衰减（供定时任务使用）"""
        return await self._settlement.decay_only(user_id, channel_id, hours)

    # ================================================================
    #  兼容桥接（已弃用 — 无外部调用者，下一版本可安全删除）
    #  仅 compat_record_value 仍有 proactive_decider 调用
    # ================================================================

    async def compat_update(
        self,
        user_id: str,
        channel_id: str,
        content: str,
        context: Dict[str, Any],
    ) -> SocialUpdateResult:
        """[已弃用] 兼容旧版 SocialValueCore.update()，无外部调用者"""
        result = await self.evaluate(user_id, channel_id, content, context)
        behavior_signal = context.get("behavior_signal", {})
        category = ""
        try:
            from src.common.config.config_engine import (
                get_default_config_engine,
            )

            params = get_default_config_engine().get_params(
                (behavior_signal.get("behavior_type", "neutral") if isinstance(behavior_signal, dict) else "neutral"),
                (behavior_signal.get("intent", "other") if isinstance(behavior_signal, dict) else "other"),
            )
            category = params.category
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return self._settlement.build_legacy_update_result(
            result.settlement,
            behavior_signal if isinstance(behavior_signal, dict) else {},
            category,
        )

    async def compat_get_value(self, user_id: str, channel_id: str) -> float:
        """兼容旧版 SocialValueCore.get_value()"""
        return await self._settlement.read_score(user_id, channel_id)

    def compat_get_stage(self, user_id: str, channel_id: str) -> str:
        """兼容旧版 RelationshipTracker.get_stage()"""
        return self.get_phase_label(user_id, channel_id)

    def compat_get_custom_label(self, user_id: str, channel_id: str) -> str:
        """兼容旧版 RelationshipTracker.get_custom_label()"""
        dossier = self._phase.fetch_dossier(user_id, channel_id)
        return dossier.custom_nick

    def compat_record_value(self, user_id: str, channel_id: str, social_value: float) -> Optional[PhaseTransition]:
        """兼容旧版 RelationshipTracker.record_value()"""
        return self._phase.ingest_score(user_id, channel_id, social_value)

    def compat_get_level_factor(self, user_id: str, channel_id: str) -> float:
        """兼容旧版 RelationshipTracker.get_level_factor()"""
        return self.get_phase_weight(user_id, channel_id)

    def compat_favor_of(self, anchor_id: str) -> float:
        """兼容旧版 BondSettler.favor_of()"""
        # BondSettler 使用 anchor_id 而非 (uid, channel_id)
        # 这里做最佳努力匹配
        for _key, dossier in self._phase._dossiers.items():
            if dossier.uid == anchor_id:
                return 0.0
        return 0.0

    def compat_capture_snapshot(self, peer_id: str) -> Dict[str, float]:
        """兼容旧版 BondSettler.capture_snapshot()"""
        return {
            "favor": 0.0,
            "trust": 0.0,
            "annoyance": 0.0,
        }

    # ================================================================
    #  内部方法
    # ================================================================

    def _assemble_snapshot(self, user_id: str, channel_id: str, score: float) -> AffectSnapshot:
        """组装完整快照"""
        dossier = self._phase.fetch_dossier(user_id, channel_id)
        phase_val = dossier.current_phase
        snap = AffectSnapshot(
            social_score=score,
            phase=phase_val,
            phase_label=dossier.custom_nick or PHASE_DESCRIPTION.get(phase_val, "有些熟悉"),
            phase_weight=PHASE_WEIGHT.get(phase_val, 0.8),
            trend=dossier.trend_label,
            user_id=user_id,
            channel_id=channel_id,
            captured_at=time.time(),
        )
        # 注入印象
        if dossier.impression:
            snap.impression_nick = dossier.impression.nickname
            snap.impression_rationale = dossier.impression.nickname_rationale
            snap.character_tags = list(dossier.impression.character_tags)
        # 注入情绪（Layer 3 中继）
        if self._emotion_reader:
            try:
                emo_state = self._emotion_reader.get_current_emotion()
                if isinstance(emo_state, dict):
                    snap.emotion_valence = float(emo_state.get("valence", 0.0))
                    snap.emotion_label = str(emo_state.get("label", ""))
            except Exception as _e:
                logger.debug(f"{self.log_prefix} 异常: {_e}")
        return snap

    def collect_diagnostics(self) -> Dict[str, Any]:
        """运行时诊断信息"""
        return {
            "settlement": self._settlement.collect_diagnostics(),
            "phase_tracker": self._phase.collect_diagnostics(),
            "emotion_reader_bound": self._emotion_reader is not None,
        }


# ================================================================
#  模块级便捷访问
# ================================================================

_fuser_singleton: Optional[SocialAffectFuser] = None


def get_social_affect_fuser() -> SocialAffectFuser:
    """获取社交情感融合器全局单例"""
    global _fuser_singleton
    if _fuser_singleton is None:
        _fuser_singleton = SocialAffectFuser()
    return _fuser_singleton

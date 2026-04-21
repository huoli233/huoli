import time
from dataclasses import dataclass
from typing import Dict, Optional
from src.common.logger import get_logger
from src.chat.heart_flow.dimension_protocol import (
    DimensionBase,
    DimensionScope,
    EventContext,
    TickResult,
)
from src.chat.heart_flow.vote_types import (
    FondnessTrustVote,
    FondnessLevel,
    TrustLevel,
)

logger = get_logger("fondness_trust")


# ============================================================
# 好感/信任度的七档映射表
# ============================================================
_FONDNESS_TIERS = [
    (-100, -70, FondnessLevel.EXTREME_DISLIKE, 0.05, "厌恶、回避", 0.0),
    (-70, -40, FondnessLevel.DISLIKE, 0.3, "冷淡、敷衍", 0.02),
    (-40, -10, FondnessLevel.DISSATISFIED, 0.6, "保守、疏离", 0.08),
    (-10, 10, FondnessLevel.NEUTRAL, 0.85, "正常、标准", 0.2),
    (10, 40, FondnessLevel.FRIENDLY, 1.0, "友善、自然", 0.55),
    (40, 70, FondnessLevel.FOND, 1.15, "亲切、主动", 0.75),
    (70, 101, FondnessLevel.ADORE, 1.3, "热情、积极", 0.95),
]

_TRUST_TIERS = [
    (-100, -70, TrustLevel.TOTAL_DISTRUST, "拒绝透露任何信息"),
    (-70, -40, TrustLevel.DISTRUST, "话题严格受限"),
    (-40, -10, TrustLevel.DOUBTFUL, "有保留地沟通"),
    (-10, 10, TrustLevel.ORDINARY, "正常互动"),
    (10, 40, TrustLevel.FAVORABLE, "愿意多分享"),
    (40, 70, TrustLevel.TRUSTED, "几乎无保留沟通"),
    (70, 101, TrustLevel.FULL_TRUST, "完全开放"),
]


# ============================================================
# 交叉组合描述
# ============================================================
def _cross_combination_hint(fondness: float, trust: float) -> str:
    """根据好感度和信任度的高低组合生成交叉描述"""
    fond_high = fondness > 30
    fond_low = fondness < -30
    trust_high = trust > 30
    trust_low = trust < -30
    if fond_high and trust_high:
        return "亲密伙伴"
    if fond_high and trust_low:
        return "喜欢但留心"
    if fond_low and trust_high:
        return "尊重但冷淡"
    if fond_low and trust_low:
        return "敌对状态"
    if fond_high:
        return "友好关系"
    if trust_high:
        return "信任关系"
    if fond_low:
        return "不太喜欢"
    if trust_low:
        return "不太信任"
    return "普通关系"


# ============================================================
# 单用户好感/信任状态
# ============================================================
@dataclass
class UserRelationState:
    """单个用户的好感/信任状态"""
    fondness_value: float = 0.0
    trust_value: float = 0.0
    positive_count: int = 0
    negative_count: int = 0
    last_interaction_ts: float = 0.0
    # 累计互动次数（用于信任的稳定性加成）
    total_interactions: int = 0


def _lookup_fondness_tier(value: float):
    """查好感度档位"""
    for lo, hi, level, prob, desc, proactive in _FONDNESS_TIERS:
        if lo <= value < hi:
            return level, prob, desc, proactive
    return FondnessLevel.NEUTRAL, 0.85, "正常", 0.2


def _lookup_trust_tier(value: float):
    """查信任度档位"""
    for lo, hi, level, desc in _TRUST_TIERS:
        if lo <= value < hi:
            return level, desc
    return TrustLevel.ORDINARY, "正常互动"


# ============================================================
# D2 维度实现：好感/信任双核
# ============================================================
class FondnessTrustDimension(DimensionBase):
    """
    D2 好感/信任双核系统维度。
    好感度和信任度是两个完全独立的维度：
      - 好感度由情感驱动变化（称赞+2, 侮辱-5, 有趣对话+1, 冷漠-1）
      - 信任度由行为驱动变化（言行一致+1, 承诺兑现+3, 出尔反尔-5, 泄露隐私-8）
    可以高好感+低信任，也可以低好感+高信任。
    per-user-per-channel 粒度。
    """

    _singleton: Optional["FondnessTrustDimension"] = None

    @classmethod
    def get_instance(cls) -> "FondnessTrustDimension":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        # 复合键 "{user_id}:{channel_id}" → UserRelationState
        self._states: Dict[str, UserRelationState] = {}
        # 长期无互动衰减速率（每小时）
        self._idle_decay_fondness_per_hour: float = 0.3
        self._idle_decay_trust_per_hour: float = 0.1
        # 衰减最小阈值（低于此值不再衰减）
        self._decay_floor_fondness: float = -10.0
        self._decay_floor_trust: float = -5.0

    @property
    def dimension_name(self) -> str:
        return "fondness_trust"

    @property
    def scope(self) -> DimensionScope:
        return DimensionScope.PER_USER_CHANNEL

    @property
    def needs_persistence(self) -> bool:
        return True

    @property
    def tick_interval_sec(self) -> float:
        return 30.0

    def _composite_key(self, user_id: str, channel_id: str) -> str:
        return f"{user_id}:{channel_id}"

    def _get_state(self, user_id: str, channel_id: str) -> UserRelationState:
        key = self._composite_key(user_id, channel_id)
        if key not in self._states:
            self._states[key] = UserRelationState(last_interaction_ts=time.time())
        return self._states[key]

    def tick(self, elapsed_sec: float) -> TickResult:
        """长期无互动时好感和信任缓慢衰减"""
        now = time.time()
        updated = False
        for _key, state in self._states.items():
            idle_hours = (now - state.last_interaction_ts) / 3600.0
            if idle_hours < 1.0:
                continue
            # 好感衰减（向0衰减，不会从正变负或从负变正）
            if state.fondness_value > self._decay_floor_fondness:
                decay = min(
                    state.fondness_value - self._decay_floor_fondness,
                    self._idle_decay_fondness_per_hour * (elapsed_sec / 3600.0),
                )
                if state.fondness_value > 0:
                    state.fondness_value = max(0.0, state.fondness_value - decay)
                else:
                    state.fondness_value = min(0.0, state.fondness_value + decay * 0.3)
                updated = True
            elif state.fondness_value < self._decay_floor_fondness:
                recovery = self._idle_decay_fondness_per_hour * (elapsed_sec / 3600.0) * 0.3
                state.fondness_value = min(self._decay_floor_fondness, state.fondness_value + recovery)
                updated = True
            # 信任衰减（同理向0衰减）
            if state.trust_value > self._decay_floor_trust:
                decay = min(
                    state.trust_value - self._decay_floor_trust,
                    self._idle_decay_trust_per_hour * (elapsed_sec / 3600.0),
                )
                if state.trust_value > 0:
                    state.trust_value = max(0.0, state.trust_value - decay)
                    updated = True
        return TickResult(
            dimension_name=self.dimension_name,
            updated=updated,
            summary=f"关系数={len(self._states)}" if updated else "",
        )

    def on_event(self, ctx: EventContext):
        """
        根据消息事件更新好感/信任。
        好感由情感倾向驱动，信任由行为稳定性驱动。
        """
        if not ctx.user_id or not ctx.channel_id:
            return
        state = self._get_state(ctx.user_id, ctx.channel_id)
        if ctx.event_type == "message_received":
            state.last_interaction_ts = time.time()
            state.total_interactions += 1
            # 好感变化（基于情感分数）
            sentiment = ctx.sentiment_score
            if sentiment > 0.5:
                state.fondness_value = min(100.0, state.fondness_value + 1.5)
                state.positive_count += 1
            elif sentiment > 0.2:
                state.fondness_value = min(100.0, state.fondness_value + 0.5)
                state.positive_count += 1
            elif sentiment < -0.5:
                state.fondness_value = max(-100.0, state.fondness_value - 3.0)
                state.negative_count += 1
            elif sentiment < -0.2:
                state.fondness_value = max(-100.0, state.fondness_value - 1.0)
                state.negative_count += 1
            # 信任：每次互动都加微量信任（长期稳定互动）
            state.trust_value = min(100.0, state.trust_value + 0.15)
            # 信任：每5次互动额外加成
            if state.total_interactions % 5 == 0:
                state.trust_value = min(100.0, state.trust_value + 0.8)
            # 信任：@机器人视为信任增强
            if ctx.is_at_bot:
                state.trust_value = min(100.0, state.trust_value + 0.5)
        elif ctx.event_type == "reply_completed":
            state.last_interaction_ts = time.time()
        # 外部显式标记的信任/好感调整
        explicit_fondness = ctx.raw_extras.get("fondness_delta")
        if explicit_fondness is not None:
            state.fondness_value = max(
                -100.0, min(100.0, state.fondness_value + float(explicit_fondness))
            )
        explicit_trust = ctx.raw_extras.get("trust_delta")
        if explicit_trust is not None:
            state.trust_value = max(
                -100.0, min(100.0, state.trust_value + float(explicit_trust))
            )
        # 桥接同步：D2 fondness_value → emotion_tracker.affection
        # 让Scene/行为映射系统能感知到最新好感值
        if ctx.event_type == "message_received" and ctx.user_id and ctx.channel_id:
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import (
                    get_emotion_tracker,
                )
                _et_sync = get_emotion_tracker(ctx.channel_id)
                _u_state = _et_sync.get_user_state(ctx.user_id, create_if_missing=False)
                if _u_state:
                    _cur_aff = float(getattr(_u_state, "affection", 0.0) or 0.0)
                    _cur_trust = float(getattr(_u_state, "trust_value", getattr(_u_state, "trust_score", 0.0)) or 0.0)
                    _is_admin = str(getattr(_u_state, "relationship", "") or "") == "管理员"
                    _d2_fond = state.fondness_value
                    if _is_admin:
                        state.fondness_value = max(state.fondness_value, _cur_aff)
                        state.trust_value = max(state.trust_value, _cur_trust)
                    elif _d2_fond > _cur_aff + 2.0:
                        _delta = (_d2_fond - _cur_aff) * 0.2
                        if _delta > 0.3:
                            _et_sync.update_affection(ctx.user_id, _delta, "D2好感同步")
                    elif _cur_aff > _d2_fond + 2.0:
                        state.fondness_value = min(100.0, state.fondness_value + (_cur_aff - _d2_fond) * 0.2)
                    if _cur_trust > state.trust_value + 2.0:
                        state.trust_value = min(100.0, state.trust_value + (_cur_trust - state.trust_value) * 0.2)
            except Exception:
                pass

    def vote(self, ctx: EventContext) -> FondnessTrustVote:
        """
        根据好感度和信任度独立计算各自的概率乘数。
        最终投票包含两个独立因子（fondness_factor, trust_factor），
        决策网关分别相乘。
        """
        if not ctx.user_id or not ctx.channel_id:
            return FondnessTrustVote()
        state = self._get_state(ctx.user_id, ctx.channel_id)
        fondness = state.fondness_value
        trust = state.trust_value
        # 查档
        f_level, f_prob, f_desc, f_proactive = _lookup_fondness_tier(fondness)
        t_level, t_desc = _lookup_trust_tier(trust)
        # 信任乘数（信任高开放话题，信任低限制话题但不直接影响概率太大）
        if trust > 70:
            trust_factor = 1.1
        elif trust > 40:
            trust_factor = 1.05
        elif trust < -70:
            trust_factor = 0.5
        elif trust < -40:
            trust_factor = 0.7
        elif trust < -10:
            trust_factor = 0.85
        else:
            trust_factor = 1.0
        cross_hint = _cross_combination_hint(fondness, trust)
        # 态度标签
        attitude = ""
        if f_level == FondnessLevel.EXTREME_DISLIKE:
            attitude = "hostile_avoidance"
        elif f_level == FondnessLevel.DISLIKE:
            attitude = "cold"
        elif f_level == FondnessLevel.FOND:
            attitude = "warm"
        elif f_level == FondnessLevel.ADORE:
            attitude = "enthusiastic"
        # 风格提示
        style = ""
        if t_level in (TrustLevel.TOTAL_DISTRUST, TrustLevel.DISTRUST):
            style = "guarded"
        elif t_level == TrustLevel.FULL_TRUST:
            style = "open"
        return FondnessTrustVote(
            probability_factor=f_prob,
            attitude_tag=attitude,
            style_hint=style,
            fondness_value=round(fondness, 1),
            fondness_level=f_level,
            fondness_factor=f_prob,
            trust_value=round(trust, 1),
            trust_level=t_level,
            trust_factor=round(trust_factor, 2),
            cross_hint=cross_hint,
            proactive_willingness=round(f_proactive, 2),
            debug_reason=f"fond={fondness:.0f}({f_level.value}) trust={trust:.0f}({t_level.value})",
        )

    def serialize(self) -> dict:
        result = {}
        for key, state in self._states.items():
            result[key] = {
                "fondness": state.fondness_value,
                "trust_value": state.trust_value,
                "trust": state.trust_value,
                "positive": state.positive_count,
                "negative": state.negative_count,
                "last_ts": state.last_interaction_ts,
                "total": state.total_interactions,
            }
        return result

    def deserialize(self, data: dict):
        if not isinstance(data, dict):
            return
        for key, vals in data.items():
            if not isinstance(vals, dict):
                continue
            self._states[key] = UserRelationState(
                fondness_value=float(vals.get("fondness", 0.0)),
                trust_value=float(vals.get("trust_value", vals.get("trust", 0.0))),
                positive_count=int(vals.get("positive", 0)),
                negative_count=int(vals.get("negative", 0)),
                last_interaction_ts=float(vals.get("last_ts", time.time())),
                total_interactions=int(vals.get("total", 0)),
            )

    def calibrate(self, offline_seconds: float):
        """离线校准: 好感和信任按离线时长衰减"""
        hours = offline_seconds / 3600.0
        for state in self._states.values():
            if state.fondness_value > 0:
                decay = min(state.fondness_value, hours * self._idle_decay_fondness_per_hour)
                state.fondness_value = max(0.0, state.fondness_value - decay)
            if state.trust_value > 0:
                decay = min(state.trust_value, hours * self._idle_decay_trust_per_hour)
                state.trust_value = max(0.0, state.trust_value - decay)

    def get_state_summary(self) -> dict:
        summaries = {}
        for key, state in list(self._states.items())[:20]:
            f_level, _, _, _ = _lookup_fondness_tier(state.fondness_value)
            t_level, _ = _lookup_trust_tier(state.trust_value)
            summaries[key[:16]] = {
                "fondness": round(state.fondness_value, 1),
                "fondness_level": f_level.value,
                "trust": round(state.trust_value, 1),
                "trust_level": t_level.value,
                "interactions": state.total_interactions,
            }
        return {
            "dimension": self.dimension_name,
            "scope": self.scope.value,
            "relation_count": len(self._states),
            "relations": summaries,
        }

    def reset(self, user_id: str = "", channel_id: str = ""):
        if user_id and channel_id:
            key = self._composite_key(user_id, channel_id)
            self._states.pop(key, None)
        elif user_id:
            to_remove = [k for k in self._states if k.startswith(f"{user_id}:")]
            for k in to_remove:
                del self._states[k]

    # ---- 外部调用接口（供其他模块直接调整好感/信任） ----

    def adjust_fondness(self, user_id: str, channel_id: str, delta: float, reason: str = ""):
        """直接调整好感度（外部调用）"""
        state = self._get_state(user_id, channel_id)
        old = state.fondness_value
        state.fondness_value = max(-100.0, min(100.0, state.fondness_value + delta))
        if delta > 0:
            state.positive_count += 1
        elif delta < 0:
            state.negative_count += 1
        logger.debug(
            f"好感度调整: {user_id[:8]}@{channel_id[:8]} {old:.1f}→{state.fondness_value:.1f} ({reason})"
        )

    def adjust_trust(self, user_id: str, channel_id: str, delta: float, reason: str = ""):
        """直接调整信任度（外部调用）"""
        state = self._get_state(user_id, channel_id)
        old = state.trust_value
        state.trust_value = max(-100.0, min(100.0, state.trust_value + delta))
        logger.debug(
            f"信任度调整: {user_id[:8]}@{channel_id[:8]} {old:.1f}→{state.trust_value:.1f} ({reason})"
        )

    def get_relation(self, user_id: str, channel_id: str) -> UserRelationState:
        """获取指定用户的关系状态（供外部查询）"""
        return self._get_state(user_id, channel_id)

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.common.singleton import _get_class_lock

logger = get_logger("coupling_matrix")

# ──────────────────────────────────────────────
#  数学工具
# ──────────────────────────────────────────────


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _sigmoid(x: float, midpoint: float = 0.5, steepness: float = 8.0) -> float:
    z = steepness * (x - midpoint)
    z = max(-15.0, min(15.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _smoothstep(e0: float, e1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - e0) / max(0.001, e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def _coalesce_missing(value: Any, default: Any) -> Any:
    """只在值缺失时回退默认，保留合法的 0、False 和低值。"""
    return default if value is None or value == "" else value


def _first_present(*values: Any, default: Any = None) -> Any:
    """返回第一个非缺失值，避免 `or` 链吞掉 0 / False。"""
    for value in values:
        if value is not None and value != "":
            return value
    return default


# ──────────────────────────────────────────────
#  回复风格四档枚举
# ──────────────────────────────────────────────


class ReplyStyleTier(Enum):
    """聊天值驱动的四档回复风格"""

    NORMAL_EXPAND = "normal_expand"
    SHORT_SENTENCE = "short_sentence"
    PERFUNCTORY = "perfunctory"
    TOO_TIRED_TO_REPLY = "too_tired"


@dataclass
class StyleProfile:
    """单档风格的量化参数"""

    target_length_ratio: float = 1.0
    punctuation_looseness: float = 0.0
    emoji_usage: float = 0.15
    detail_level: float = 0.65
    emotional_depth: float = 0.50
    response_delay_bias: float = 0.0


_STYLE_PROFILES = {
    ReplyStyleTier.NORMAL_EXPAND: StyleProfile(
        target_length_ratio=1.0,
        punctuation_looseness=0.05,
        emoji_usage=0.18,
        detail_level=0.70,
        emotional_depth=0.55,
        response_delay_bias=0.0,
    ),
    ReplyStyleTier.SHORT_SENTENCE: StyleProfile(
        target_length_ratio=0.55,
        punctuation_looseness=0.20,
        emoji_usage=0.25,
        detail_level=0.40,
        emotional_depth=0.38,
        response_delay_bias=0.12,
    ),
    ReplyStyleTier.PERFUNCTORY: StyleProfile(
        target_length_ratio=0.32,
        punctuation_looseness=0.40,
        emoji_usage=0.08,
        detail_level=0.18,
        emotional_depth=0.15,
        response_delay_bias=0.28,
    ),
    ReplyStyleTier.TOO_TIRED_TO_REPLY: StyleProfile(
        target_length_ratio=0.12,
        punctuation_looseness=0.55,
        emoji_usage=0.03,
        detail_level=0.05,
        emotional_depth=0.06,
        response_delay_bias=0.58,
    ),
}


# ──────────────────────────────────────────────
#  状态快照载体
# ──────────────────────────────────────────────


@dataclass
class StateSnapshot:
    """耦合矩阵的输入+输出统一载体 —— 所有参与耦合的状态量"""

    # ── 来源：MetabolismEngine ──
    boredom_level: float = 0.0
    loafing_level: float = 0.0
    fatigue_accumulator: float = 0.0
    chat_fuel: float = 100.0
    thinking_fuel: float = 100.0
    activity_gauge: float = 50.0
    social_gauge: float = 0.0
    consecutive_active_minutes: float = 0.0
    # ── 来源：SubjectivePresenceCore ──
    social_willingness: float = 0.5
    watch_willingness: float = 0.5
    avoidance_tendency: float = 0.0
    quiet_preference: float = 0.2
    curiosity_level: float = 0.3
    outward_attention: float = 0.5
    # ── 来源：关系系统 ──
    affection: float = 50.0
    trust_value: float = 50.0
    annoyance_value: float = 0.0
    trauma_score: float = 0.0
    # ── 来源：群环境 ──
    group_atmosphere_type: str = ""
    group_atmosphere_score: float = 0.5
    active_user_count: int = 0
    is_high_value_group: bool = False
    # ── 来源：夜间系统 ──
    night_phase: str = "awake"
    arousal_chain_state: str = "normal"
    # ── GAP-K 计算出的衍生量 ──
    peek_desire: float = 0.0
    effective_watch_will: float = 0.5
    reply_style_tier: str = "normal_expand"
    style_profile: Optional[StyleProfile] = None
    visibility_bias_map: Dict[str, float] = field(default_factory=dict)
    conflict_resolution_log: List[str] = field(default_factory=list)
    coupling_deltas: Dict[str, float] = field(default_factory=dict)
    watch_state_rank: int = 2
    activity_drive: float = 0.0
    loafing_block_strength: float = 0.0
    energy_ratio: float = 1.0


# ──────────────────────────────────────────────
#  核心：状态耦合矩阵引擎
# ──────────────────────────────────────────────


class StateCouplingMatrixEngine:
    """多状态耦合矩阵引擎

    6 条主联动链：
      ① 无聊 → 窥屏欲 → 摸鱼（两段式转化，带抑制条件）
      ② 活跃 → 思考启动 → 行为启动（三层驱动链）
      ③ 聊天值 → 回复风格四档（连续映射）
      ④ 摸鱼值 → 消息打断权（与 GAP-E 联动的量化映射）
      ⑤ 关系层 → 主观可见性偏置（完整公式）
      ⑥ 群环境 ↔ 状态双向调制（正向/反向效应）

    + 状态冲突优先级协调器
    """

    def __init__(self):
        self._last_eval_time: float = 0.0
        self._eval_interval_sec: float = 5.0
        self._cached_result: Optional[StateSnapshot] = None
        self._last_input_signature: Optional[Tuple[Any, ...]] = None
        self._conflict_history: deque = deque(maxlen=30)

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if hasattr(value, "value"):
            return str(getattr(value, "value") or "").strip()
        if hasattr(value, "name"):
            return str(getattr(value, "name") or "").strip()
        return str(value).strip()

    @staticmethod
    def _normalize_night_phase(value: Any) -> str:
        raw = StateCouplingMatrixEngine._to_text(value)
        if not raw:
            return "awake"
        up = raw.upper()
        mapping = {
            "DEEP_VALLEY": "deep_sleep",
            "DROWSY": "drowsy",
            "ACTIVE_TWILIGHT": "night_active",
            "DAWN_RECOVERY": "dawn_recover",
            "BURNED_OUT": "burned_out",
        }
        return mapping.get(up, raw.lower())

    @staticmethod
    def _normalize_atmosphere(value: Any) -> str:
        raw = StateCouplingMatrixEngine._to_text(value).lower()
        if not raw:
            return ""
        mapping = {
            "quiet": "tense_silence",
            "casual_chat": "casual_relaxed",
            "heated_discussion": "heated_discussion",
            "argument": "argument",
            "celebration": "celebration_wave",
            "mourning": "cold_awkward",
            "bored_drift": "dormant_dead",
            "newcomer_welcome": "friendly_supportive",
            "spectating": "casual_relaxed",
            "memeing": "playful_meme",
            "deep_discussion": "warm_chatty",
            "storytelling": "warm_chatty",
            "problem_solving": "warm_chatty",
            "social_gaming": "playful_meme",
            "emotional_support": "friendly_supportive",
            "night_owls": "casual_relaxed",
            "morning_coffee": "warm_chatty",
            "spam_flood": "chaotic_spam",
        }
        return mapping.get(raw, raw)

    @staticmethod
    def _snapshot_signature(s: StateSnapshot) -> Tuple[Any, ...]:
        def _r(v: Any) -> float:
            try:
                return round(float(v), 4)
            except Exception:
                return 0.0

        return (
            _r(s.boredom_level),
            _r(s.loafing_level),
            _r(s.fatigue_accumulator),
            _r(s.chat_fuel),
            _r(s.thinking_fuel),
            _r(s.activity_gauge),
            _r(s.social_willingness),
            _r(s.watch_willingness),
            _r(s.avoidance_tendency),
            _r(s.quiet_preference),
            _r(s.curiosity_level),
            _r(s.outward_attention),
            _r(s.affection),
            _r(s.trust_value),
            _r(s.annoyance_value),
            _r(s.trauma_score),
            str(s.group_atmosphere_type or ""),
            _r(s.group_atmosphere_score),
            int(s.active_user_count or 0),
            bool(s.is_high_value_group),
            str(s.night_phase or ""),
            str(s.arousal_chain_state or ""),
            int(s.watch_state_rank or 0),
        )

    def evaluate(self, snap: StateSnapshot) -> StateSnapshot:
        """执行完整的耦合矩阵评估（主入口）

        输入：原始状态快照（各子系统原始值）
        输出：填充了所有衍生量的完整快照
        """
        now = time.time()
        sig = self._snapshot_signature(snap)
        if (
            self._cached_result
            and self._last_input_signature == sig
            and (now - self._last_eval_time) < self._eval_interval_sec
        ):
            return self._cached_result
        self._last_input_signature = sig
        self._last_eval_time = now
        # ══ 联动链 ①：无聊 → 窥屏欲 → 摑鱼 ═══
        self._chain_boredom_to_loafing(snap)
        # ══ 联动链 ②：活跃 → 思考 → 行为 ═══
        self._chain_activity_to_behavior(snap)
        # ══ 联动链 ③：聊天值 → 风格四档 ═══
        self._chain_chat_to_style(snap)
        # ══ 联动链 ④：摸鱼 → 打断权 ═══
        self._chain_loafing_to_interruption(snap)
        # ══ 联动链 ⑤：关系 → 可见性偏置 ═══
        self._chain_relation_to_visibility(snap)
        # ══ 联动链 ⑥：群环境 ↔ 状态双向调制 ═══
        self._chain_group_environment_bidirectional(snap)
        # ══ 状态冲突优先级协调 ═══
        self._resolve_state_conflicts(snap)
        self._cached_result = snap
        return snap

    # ──── 联动链 ①：无聊 → 窥屏欲 → 摑鱼 ────

    def _chain_boredom_to_loafing(self, s: StateSnapshot) -> None:
        """两段式无聊转化链

        第一段（boredom 10~45）：无聊转化为窥屏欲
          - 轻度无聊激发好奇心 → 想看看群里在干什么
          - 但如果有高兴趣话题或好友在线，窥屏欲直接升为参与欲

        第二段（boredom 45~75）：窥屏欲饱和后溢出为摸鱼
          - 看了也没意思 → 开始摆烂
          - 如果同时疲劳高，摸鱼加速

        抑制条件（阻止转化的因素）：
          - 好奇心本身很高 → 无聊不转化为窥屏，而是保持探索
          - 有高好感用户活跃 → 无聊被社交吸引抵消
          - 群环境友好且热闹 → 无聊被环境刺激消解
        """
        b = _clamp01(s.boredom_level / 100.0)
        c = _clamp01(s.curiosity_level)
        aff = _clamp01(s.affection / 100.0)
        sw = _clamp01(s.social_willingness)
        aw = _clamp01(s.watch_willingness)
        fat = _clamp01(s.fatigue_accumulator / 100.0)
        # 抑制因子
        curiosity_brake = _sigmoid(c, midpoint=0.5, steepness=6.0)
        social_brake = _sigmoid(aff, midpoint=0.45, steepness=5.0)
        env_stimulus = 0.0
        if s.is_high_value_group and s.active_user_count >= 3:
            env_stimulus = 0.12 * _clamp01(s.group_atmosphere_score)
        total_brake = curiosity_brake * 0.38 + social_brake * 0.35 + env_stimulus * 0.27
        effective_boredom = b * (1.0 - min(0.65, total_brake))
        # 第一段：无聊 → 窥屏欲
        if effective_boredom < 0.12:
            peek_base = 0.0
        elif effective_boredom < 0.35:
            peek_base = _smoothstep(0.12, 0.35, effective_boredom) * 0.45
        elif effective_boredom < 0.55:
            peek_base = 0.45 + (effective_boredom - 0.35) * 0.80
        else:
            peek_base = min(0.85, 0.61 + (effective_boredom - 0.55) * 0.50)
        # 好友在线加成
        if aff > 0.55 and sw > 0.4:
            peek_base += 0.10 * aff
        # 疲劳修正窥屏欲（太累了连看都不想看）
        if fat > 0.55:
            peek_base *= 1.0 - (fat - 0.55) * 0.60
        s.peek_desire = _clamp01(peek_base)
        # 第二段：窥屏欲溢出 → 摸鱼（当无聊>45 且窥屏已饱和时）
        loaf_conversion = 0.0
        if effective_boredom > 0.42 and s.peek_desire > 0.65:
            overflow = (s.peek_desire - 0.65) * (effective_boredom - 0.42)
            loaf_conversion = overflow * 0.70
        if effective_boredom > 0.58:
            heavy_boredom_convert = (effective_boredom - 0.58) * 0.55
            loaf_conversion += heavy_boredom_convert
        # 疲劳加速摸鱼
        if fat > 0.45:
            fatigue_loaf_boost = (fat - 0.45) * 0.35
            loaf_conversion += fatigue_loaf_boost
        # 社交意愿反向抑制摸鱼（想社交就不会摸鱼）
        if sw > 0.55:
            loaf_conversion *= 1.0 - (sw - 0.55) * 0.8
        current_loaf = _clamp01(s.loafing_level / 100.0)
        loaf_delta = loaf_conversion - current_loaf * 0.08
        loaf_delta = max(-0.02, min(0.04, loaf_delta * 0.15))
        s.coupling_deltas["boredom_to_loafing"] = round(loaf_delta, 5)

    # ──── 联动链 ②：活跃 → 思考 → 行为 ────

    def _chain_activity_to_behavior(self, s: StateSnapshot) -> None:
        """三层驱动链：活跃值触发思考启动，思考资源充足才允许行为启动

        层级关系：
          Layer 1: activity_gauge > 阈值_A → 思考引擎预热
          Layer 2: thinking_fuel > 阈值_T → 行为概率解锁
          Layer 3: 行为概率 × 存在态意愿 → 最终行为强度

        当 thinking_fuel 不足时：
          - 不直接拒绝行为，而是降级（完整回复 → 仅表情 → 仅旁白 → 仅围观）
        """
        act = _clamp01(s.activity_gauge / 100.0)
        think = _clamp01(s.thinking_fuel / 100.0)
        chat = _clamp01(s.chat_fuel / 100.0)
        sw = _clamp01(s.social_willingness)
        aw = _clamp01(s.watch_willingness)
        avoid = _clamp01(s.avoidance_tendency)
        # 层1：活跃→思考启动阈值
        think_threshold_a = 0.28
        if s.night_phase in ("light_sleep", "deep_sleep"):
            think_threshold_a = 0.55
        elif s.arousal_chain_state == "drowsy":
            think_threshold_a = 0.42
        elif s.arousal_chain_state == "stubborn":
            think_threshold_a = 0.38
        think_ready = 1.0 if act > think_threshold_a else _sigmoid(act, midpoint=think_threshold_a, steepness=8.0)
        # 层2：思考→行为启动
        behavior_threshold_t = 0.22
        if s.night_phase in ("light_sleep", "deep_sleep"):
            behavior_threshold_t = 0.52
        elif s.arousal_chain_state in ("drowsy", "stubborn"):
            behavior_threshold_t = 0.35
        behavior_unlock = (
            1.0 if think > behavior_threshold_t else _sigmoid(think, midpoint=behavior_threshold_t, steepness=7.0)
        )
        # 层3：存在态调制最终行为强度
        presence_drive = sw * 0.40 + aw * 0.30 + (1.0 - avoid) * 0.20 + _clamp01(s.curiosity_level) * 0.10
        final_behavior_strength = think_ready * behavior_unlock * presence_drive
        # 有效观看意愿 = 原始watch_will × 行为链输出
        s.effective_watch_will = _clamp01(s.watch_willingness * final_behavior_strength)
        s.coupling_deltas["activity_think_behavior"] = round(final_behavior_strength, 5)
        # 思考不足时的行为降级提示
        if think < 0.18 and act > 0.35:
            s.coupling_deltas["behavior_degraded"] = round(1.0 - think, 3)
        elif think < 0.30 and act > 0.50:
            s.coupling_deltas["behavior_partial"] = round(0.5 + think, 3)

    # ──── 联动链 ③：聊天值 → 风格四档 ────

    def _chain_chat_to_style(self, s: StateSnapshot) -> None:
        """聊天值到回复风格的连续四档映射

        四档边界是软过渡区（用 smoothstep），不是硬切：
          chat > 62%  → normal_expand（正常展开）
          32%~62%  → short_sentence（短句为主）
          12%~32%  → perfunctory（敷衍式）
          < 12%     → too_tired_to_reply（懒得理）

        夜间和熬穿状态会整体偏移风格曲线
        """
        chat_ratio = _clamp01(s.chat_fuel / 100.0)
        # 夜间偏移：整体风格向"短/敷衍"方向偏移
        night_shift = 0.0
        if s.night_phase == "light_sleep":
            night_shift = 0.10
        elif s.night_phase == "quiet_contemplate":
            night_shift = 0.06
        elif s.arousal_chain_state == "burnthrough":
            night_shift = -0.08
        elif s.arousal_chain_state == "drowsy":
            night_shift = 0.12
        adjusted_chat = _clamp01(chat_ratio - night_shift)
        # 四档判定（软边界）
        if adjusted_chat > 0.62:
            tier = ReplyStyleTier.NORMAL_EXPAND
        elif adjusted_chat > 0.32:
            tier = ReplyStyleTier.SHORT_SENTENCE
        elif adjusted_chat > 0.12:
            tier = ReplyStyleTier.PERFUNCTORY
        else:
            tier = ReplyStyleTier.TOO_TIRED_TO_REPLY
        s.reply_style_tier = tier.value
        s.style_profile = _STYLE_PROFILES[tier]
        # 相邻档位混合（在边界处取加权平均）
        blend_weight = 0.0
        if tier == ReplyStyleTier.NORMAL_EXPAND:
            blend_weight = _smoothstep(0.62, 0.78, adjusted_chat)
        elif tier == ReplyStyleTier.SHORT_SENTENCE:
            blend_weight = _smoothstep(0.32, 0.62, adjusted_chat)
        elif tier == ReplyStyleTier.PERFUNCTORY:
            blend_weight = _smoothstep(0.12, 0.32, adjusted_chat)
        else:
            blend_weight = 1.0 - _smoothstep(0.0, 0.12, adjusted_chat)
        s.coupling_deltas["style_blend"] = round(blend_weight, 4)
        s.coupling_deltas["style_night_shift"] = round(night_shift, 4)

    # ──── 联动链 ④：摸鱼 → 打断权 ────

    def _chain_loafing_to_interruption(self, s: StateSnapshot) -> None:
        """摸鱼值到消息打断权的量化映射（供 GAP-E 使用）

        返回的不是简单的布尔值，而是一个连续的"拦截强度"：
          loafing < 0.20 → 拦截强度 ~0（几乎不拦截）
          0.20~0.45     → 低拦截（只拦截 Tier 1 普通消息）
          0.45~0.70     → 中拦截（拦截 T1+部分 T2）
          > 0.70         → 高拦截（只放行 Tier 3 强优先级）

        这个值会被 GAP-E 的 `_classify_interruption_rights` 引用。
        """
        loaf = _clamp01(s.loafing_level / 100.0)
        if loaf < 0.18:
            block_intensity = 0.0
        elif loaf < 0.40:
            block_intensity = _smoothstep(0.18, 0.40, loaf) * 0.30
        elif loaf < 0.65:
            block_intensity = 0.30 + (loaf - 0.40) * 0.90
        else:
            block_intensity = min(0.95, 0.525 + (loaf - 0.65) * 1.30)
        # 夜间略微放宽（夜里本来就该安静）
        if s.night_phase not in ("awake", "night_active", "excited"):
            block_intensity *= 0.82
        # 高好奇时降低拦截（好奇心战胜懒）
        if s.curiosity_level > 0.55:
            block_intensity *= 1.0 - (s.curiosity_level - 0.55) * 0.5
        s.coupling_deltas["interruption_block_intensity"] = round(block_intensity, 4)

    # ──── 联动链 ⑤：关系 → 可见性偏置 ────

    def _chain_relation_to_visibility(self, s: StateSnapshot) -> None:
        """关系层到主观可见性的完整偏置公式

        对每个用户生成一个可见性乘数（0.3~1.5）：
          base_visibility = 1.0
          + affection 加成（好感>60 时可见性提升）
          + trust 加成（信任>55 时进一步提升）
          - annoyance 减益（厌烦>30 时可见性下降）
          - trauma 减益（创伤>40 时大幅下降）
          - aversion 减益（厌恶>25 时接近不可见）

        最终输出为 visibility_bias_map 字典格式，
        可被 VisibilityGate 直接消费。
        """
        aff = _clamp01(s.affection / 100.0)
        trust = _clamp01(s.trust_value / 100.0)
        annoy = _clamp01(s.annoyance_value / 100.0)
        trauma = _clamp01(s.trauma_score / 100.0)
        aversion = _clamp01((s.trauma_score + s.annoyance_value) / 200.0)
        base = 1.0
        if aff > 0.55:
            base += (aff - 0.55) * 0.55
        if trust > 0.50:
            base += (trust - 0.50) * 0.35
        if annoy > 0.25:
            base -= (annoy - 0.25) * 0.50
        if trauma > 0.35:
            base -= (trauma - 0.35) * 0.70
        if aversion > 0.30:
            base -= (aversion - 0.30) * 0.80
        base = max(0.25, min(1.55, base))
        s.visibility_bias_map["base_relation_bias"] = round(base, 4)
        s.visibility_bias_map["aff_component"] = round(aff, 4)
        s.visibility_bias_map["trust_component"] = round(trust, 4)
        s.visibility_bias_map["annoy_penalty"] = round(max(0, (annoy - 0.25) * 0.50), 4)
        s.visibility_bias_map["trauma_penalty"] = round(max(0, (trauma - 0.35) * 0.70), 4)
        s.coupling_deltas["visibility_from_relation"] = round(base, 4)

    # ──── 联动链 ⑥：群环境 ↔ 状态双向调制 ────

    def _chain_group_environment_bidirectional(self, s: StateSnapshot) -> None:
        """群环境和内部状态的双向调制

        正向效应（好的群环境提升内部状态）：
          友好/支持型群 → 防御恢复更快、无聊增速减半、摸鱼减速
          高价值群（经常有有趣内容）→ 好奇心自然维持

        反向效应（坏的群环境恶化内部状态）：
          嘈杂/冲突型群 → 回避上升、疲劳加速、社交欲下降
          冷漠/尴尬群 → 无聊加速、参与欲骤降
        """
        score = _clamp01(s.group_atmosphere_score)
        atmo = s.group_atmosphere_type.lower()
        n_users = max(0, s.active_user_count)
        is_high_val = s.is_high_value_group
        # 正向调制量
        pos_boredom_mod = 0.0
        pos_loaf_mod = 0.0
        pos_avoid_mod = 0.0
        pos_curiosity_mod = 0.0
        pos_social_mod = 0.0
        if atmo in ("friendly_supportive", "celebration_wave", "warm_chatty"):
            pos_boredom_mod = -0.25 * score
            pos_loaf_mod = -0.18 * score
            pos_avoid_mod = -0.15 * score
            pos_curiosity_mod = 0.12 * score
            pos_social_mod = 0.15 * score
        elif atmo in ("casual_relaxed", "playful_meme"):
            pos_boredom_mod = -0.12 * score
            pos_curiosity_mod = 0.08 * score
            pos_social_mod = 0.08 * score
        if is_high_val and n_users >= 3:
            pos_curiosity_mod += 0.08
            pos_boredom_mod -= 0.06
        # 反向调制量
        neg_boredom_mod = 0.0
        neg_avoid_mod = 0.0
        neg_fatigue_mod = 0.0
        neg_social_mod = 0.0
        if atmo in ("heated_discussion", "argument", "conflict_escalation"):
            neg_avoid_mod = 0.18 * score
            neg_fatigue_mod = 0.10 * score
            neg_social_mod = -0.10 * score
            if n_users >= 6:
                neg_fatigue_mod += 0.08
                neg_avoid_mod += 0.08
        elif atmo in ("chaotic_spam", "meme_storm"):
            neg_boredom_mod = 0.15 * score
            neg_avoid_mod = 0.10 * score
            if n_users >= 5:
                neg_boredom_mod += 0.10
        elif atmo in ("cold_awkward", "tense_silence", "dormant_dead"):
            neg_boredom_mod = 0.22 * score
            neg_social_mod = -0.18 * score
            pos_avoid_mod += 0.08 * score
        # 写入耦合增量
        s.coupling_deltas["group_pos_boredom"] = round(pos_boredom_mod, 4)
        s.coupling_deltas["group_neg_boredom"] = round(neg_boredom_mod, 4)
        s.coupling_deltas["group_pos_avoid"] = round(pos_avoid_mod, 4)
        s.coupling_deltas["group_neg_avoid"] = round(neg_avoid_mod, 4)
        s.coupling_deltas["group_pos_social"] = round(pos_social_mod + neg_social_mod, 4)
        s.coupling_deltas["group_fatigue"] = round(neg_fatigue_mod, 4)
        s.coupling_deltas["group_curiosity"] = round(pos_curiosity_mod, 4)

    # ──── 状态冲突优先级协调 ────

    def _resolve_state_conflicts(self, s: StateSnapshot) -> None:
        """当多个状态信号指向相反方向时的协调规则
        优先级排序（高→低）：
          P0: 安全/生存类（创伤极高/极度疲劳）→ 最高优先，强制阻断
          P1: 意图覆权类（强意图驱动）→ 次高，可覆盖大部分负面
          P2: 环境强制类（深睡/黑屏）→ 硬约束
          P3: 关系驱动类（高好感用户的请求）→ 中等优先
          P4: 好奇/兴趣类（新有趣话题）→ 中低优先
          P5: 日常代谢类（普通无聊/摸鱼）→ 最低优先，容易被覆盖"""
        conflicts: List[str] = []
        trauma = _clamp01(s.trauma_score / 100.0)
        fat = _clamp01(s.fatigue_accumulator / 100.0)
        boredom = _clamp01(s.boredom_level / 100.0)
        loaf = _clamp01(s.loafing_level / 100.0)
        aff = _clamp01(s.affection / 100.0)
        curios = _clamp01(s.curiosity_level)
        avoid = _clamp01(s.avoidance_tendency)
        sw = _clamp01(s.social_willingness)
        if trauma > 0.75:
            conflicts.append(f"P0-TRAUMA_BLOCK(trauma={trauma:.2f})")
        if fat > 0.85:
            conflicts.append(f"P0-FATIGUE_BLOCK(fat={fat:.2f})")
        if boredom > 0.55 and curios > 0.55:
            winner = "curiosity" if curios > boredom + 0.10 else "boredom"
            conflicts.append(f"P1-BORED_vs_CURIOSITY({winner} wins)")
        if loaf > 0.55 and sw > 0.55:
            net = sw - loaf
            if net > 0.10:
                conflicts.append(f"P2-LOAF_vs_SOCIAL(social wins, delta={net:.2f})")
            elif net < -0.10:
                conflicts.append(f"P2-LOAF_vs_SOCIAL(loaf wins, delta={net:.2f})")
            else:
                conflicts.append("P2-LOAF_vs_SOCIAL(stalemate)")
        if avoid > 0.50 and aff > 0.55:
            aff_force = aff * 0.70
            avoid_force = avoid * 0.85
            if aff_force > avoid_force:
                conflicts.append("P3-AVOID_vs_AFFINITY(affinity wins)")
            else:
                conflicts.append("P3-AVOID_vs_AFFINITY(avoidance wins)")
        if s.quiet_preference > 0.55 and curios > 0.50:
            conflicts.append(f"P4-QUIET_vs_CURIOSITY(quiet={s.quiet_preference:.2f} cur={curios:.2f})")
        if not conflicts:
            s.conflict_resolution_log = ["NO_CONFLICT"]
        else:
            s.conflict_resolution_log = conflicts
            logger.debug(f"[SCME] 状态冲突协调: {'; '.join(conflicts)}")

    def build_full_snapshot(
        self,
        *,
        metabolism: Optional[Any] = None,
        presence: Optional[Any] = None,
        relation: Optional[Dict[str, Any]] = None,
        night_phase: Optional[Any] = None,
        group_scene: Optional[Any] = None,
        watch_state: str = "",
    ) -> Dict[str, Any]:
        snap = StateSnapshot()
        snap.night_phase = self._normalize_night_phase(night_phase)
        if snap.night_phase in ("burned_out", "exhausted"):
            snap.arousal_chain_state = "burnthrough"
        elif snap.night_phase == "light_sleep":
            snap.arousal_chain_state = "drowsy"
        if metabolism:
            _raw_boredom = float(
                getattr(metabolism, "boredom_level", getattr(metabolism, "boredom", 0.0)) or 0.0
            )
            if _raw_boredom <= 1.0:
                _raw_boredom *= 100.0
            _raw_loafing = float(
                getattr(metabolism, "loafing_level", getattr(metabolism, "loafing_suppression", 0.0)) or 0.0
            )
            if _raw_loafing <= 1.0:
                _raw_loafing *= 100.0
            snap.energy_ratio = max(
                0.05,
                1.0 - float(getattr(metabolism, "energy_suppression", 0.0) or 0.0),
            )
            snap.boredom_level = _raw_boredom
            snap.loafing_level = _raw_loafing
        if presence:
            snap.social_willingness = float(
                _coalesce_missing(getattr(presence, "social_willingness", None), 0.5)
            )
            snap.watch_state_rank = int(
                _coalesce_missing(getattr(presence, "watch_state_rank", None), 2)
            )
            snap.avoidance_tendency = float(
                _coalesce_missing(getattr(presence, "avoidance_tendency", None), 0.0)
            )
            snap.curiosity_level = float(
                _coalesce_missing(getattr(presence, "curiosity_level", None), 0.3)
            )
            snap.outward_attention = float(
                _coalesce_missing(getattr(presence, "outward_attention", None), 0.5)
            )
            snap.quiet_preference = float(
                _coalesce_missing(getattr(presence, "quiet_preference", None), 0.2)
            )
        watch_text = self._to_text(watch_state).lower()
        watch_rank_map = {
            "blackout": 0,
            "peek": 1,
            "skim_window": 2,
            "active_watch": 3,
            "engaged": 4,
        }
        if watch_text in watch_rank_map:
            snap.watch_state_rank = watch_rank_map[watch_text]
        if relation:
            # 关系字段统一按 0~100 输入，避免二次归一化导致关系链失活
            _affection = relation.get("affection", None)
            _trust_value = relation.get("trust_value", None)
            _annoyance_value = relation.get("annoyance_value", None)
            _trauma_score = relation.get("trauma_score", relation.get("trauma", None))
            snap.affection = float(50.0 if _affection is None else _affection)
            snap.trust_value = float(50.0 if _trust_value is None else _trust_value)
            snap.annoyance_value = float(0.0 if _annoyance_value is None else _annoyance_value)
            snap.trauma_score = float(0.0 if _trauma_score is None else _trauma_score)
        if group_scene:
            scene_data: Dict[str, Any] = {}
            if isinstance(group_scene, dict):
                scene_data = group_scene
            elif hasattr(group_scene, "to_dict"):
                try:
                    scene_data = dict(group_scene.to_dict() or {})
                except Exception:
                    scene_data = {}
            atmo_raw = (
                scene_data.get("atmosphere")
                or scene_data.get("atmosphere_type")
                or scene_data.get("dominant_pattern")
                or scene_data.get("type")
                or getattr(group_scene, "atmosphere", "")
            )
            score_raw = _first_present(
                scene_data.get("group_atmosphere_score"),
                scene_data.get("atmosphere_score"),
                scene_data.get("confidence"),
                getattr(group_scene, "interaction_quality", None),
                default=0.5,
            )
            active_raw = _first_present(
                scene_data.get("active_user_count"),
                scene_data.get("unique_speakers_5min"),
                getattr(group_scene, "unique_speakers_5min", None),
                default=0,
            )
            high_value_raw = _first_present(
                scene_data.get("is_high_value_group"),
                scene_data.get("high_value_group"),
                scene_data.get("is_high_value"),
                default=None,
            )
            snap.group_atmosphere_type = self._normalize_atmosphere(atmo_raw)
            try:
                score_v = float(_coalesce_missing(score_raw, 0.5))
                if score_v > 1.0:
                    score_v /= 100.0
                snap.group_atmosphere_score = _clamp01(score_v)
            except Exception:
                snap.group_atmosphere_score = 0.5
            try:
                snap.active_user_count = max(0, int(_coalesce_missing(active_raw, 0)))
            except Exception:
                snap.active_user_count = 0
            if isinstance(high_value_raw, bool):
                snap.is_high_value_group = high_value_raw
            else:
                snap.is_high_value_group = snap.active_user_count >= 6 and snap.group_atmosphere_score >= 0.65
        result = self.evaluate(snap)
        return {
            "boredom_to_loafing_mod": round(result.boredom_level - (snap.boredom_level or 0), 3),
            "activity_behavior_mod": round(result.activity_drive - 0.0, 3),
            "style_tier": getattr(result, "reply_style_tier", "") or "",
            "loafing_block_strength": round(result.loafing_block_strength, 3),
            "energy_after_coupling": round(result.energy_ratio, 3),
            "boredom_after_coupling": round(result.boredom_level, 3),
        }


# ──────────────────────────────────────────────
#  全局单例（线程安全，基于统一单例工具）
# ──────────────────────────────────────────────

_scme_instance: Optional[StateCouplingMatrixEngine] = None
_scme_lock = _get_class_lock("StateCouplingMatrixEngine")


def get_coupling_engine() -> StateCouplingMatrixEngine:
    global _scme_instance
    if _scme_instance is None:
        with _scme_lock:
            if _scme_instance is None:
                _scme_instance = StateCouplingMatrixEngine()
                logger.info("[SCME引擎] 多状态耦合矩阵初始化完成 — 6条联动链+冲突协调")
    return _scme_instance

import asyncio
import math
import time as _tm
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple
from src.common.logger import get_logger
from src.chat.utils.prompt_builder import global_prompt_manager
from src.config.core_config_engine import get_core_config

logger = get_logger("心流决策")


def _load_decision_cfg() -> dict:
    """从核心配置中枢读取 heartflow_decision 段落"""
    try:
        return get_core_config().assemble_decision_config()
    except RuntimeError:
        return {}


def _safe_signal_value(signal: Dict[str, Any], key: str, default: float) -> float:
    """读取信号时只在缺失时使用默认值，保留合法的 0。"""
    value = signal.get(key, None)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


# ---------------------------------------------------------------------------
#  决策维度与可信度分级
# ---------------------------------------------------------------------------


class JudgmentDimension(Enum):
    """决策的目标维度"""

    RESPOND = "respond"  # 是否应答
    TONE = "tone"  # 应答语气
    APPROACH = "approach"  # 应答策略
    TIMING_DELAY = "timing"  # 应答延迟时机
    VIGOR = "vigor"  # 应答强度/力度


class CertaintyBand(Enum):
    """置信度分级，值为该区间下界"""

    NEGLIGIBLE = 0.15
    FAINT = 0.35
    MODERATE = 0.55
    FIRM = 0.78
    RESOLUTE = 0.93


# ---------------------------------------------------------------------------
#  多维信号载体
# ---------------------------------------------------------------------------


@dataclass
class JudgmentSignals:
    """六维决策信号矢量"""

    credibility: float = 0.5  # 可信度（模型/历史一致性）
    hazard: float = 0.5  # 威胁度（毒性/不稳定/社交风险）
    curiosity: float = 0.5  # 好奇度（对话题的兴趣）
    immediacy: float = 0.5  # 紧迫度（问句/紧急关键词）
    bond_strength: float = 0.5  # 关系强度（亲密/陌生）
    temperament: float = 0.5  # 情绪因子（bot当前心境）


@dataclass
class JudgmentSituation:
    """决策时的完整情景快照"""

    stream_id: str = ""
    sender_id: str = ""
    raw_content: str = ""
    dialogue_tail: List[Dict] = field(default_factory=list)
    sender_profile: Dict[str, Any] = field(default_factory=dict)
    bot_mood_tag: str = "neutral"
    temporal_hints: Dict[str, Any] = field(default_factory=dict)
    social_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgmentOutcome:
    """一次决策的完整输出"""

    dimension: JudgmentDimension
    verdict: Any
    certainty: CertaintyBand
    rationale: str = ""
    signal_snapshot: JudgmentSignals = field(default_factory=JudgmentSignals)
    runner_up_options: List[Dict] = field(default_factory=list)
    execution_memo: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_tm.time)


# ---------------------------------------------------------------------------
#  缓存槽位 & 频道脉冲仪（参数从配置读取）
# ---------------------------------------------------------------------------


@dataclass
class _VerdictSlot:
    """缓存槽：带代际指纹的裁决结果"""

    should_act: bool
    rationale: str
    stamp: float
    epoch: int = 0

    def still_valid(self, lifespan: float, live_epoch: int) -> bool:
        if live_epoch != self.epoch:
            return False
        return (_tm.time() - self.stamp) < lifespan


@dataclass
class _PulseMeter:
    """频道脉冲计量器：追踪消息到达密度"""

    arrivals: list = field(default_factory=list)
    epoch: int = 0

    def log_arrival(self) -> None:
        self.arrivals.append(_tm.time())

    def _trim_stale(self) -> None:
        cfg = _load_decision_cfg()
        span = float(cfg.get("pulse_span_seconds", 60.0))
        boundary = _tm.time() - span
        self.arrivals = [t for t in self.arrivals if t > boundary]

    def density(self) -> int:
        self._trim_stale()
        return len(self.arrivals)

    def is_heated(self) -> bool:
        cfg = _load_decision_cfg()
        limit = int(cfg.get("heated_pulse_limit", 5))
        return self.density() >= limit

    def slot_lifespan(self) -> float:
        cfg = _load_decision_cfg()
        heated_ttl = float(cfg.get("heated_slot_ttl", 2.0))
        calm_ttl = float(cfg.get("calm_slot_ttl", 5.0))
        return heated_ttl if self.is_heated() else calm_ttl

    def advance_epoch(self) -> None:
        self.epoch += 1


# ---------------------------------------------------------------------------
#  子系统 1：柔性门槛引擎（自适应阈值）
# ---------------------------------------------------------------------------


@dataclass
class _BarrierAnchor:
    """单个门槛的锚点与漂移状态"""

    tag: str
    baseline: float
    live_value: float
    drift_rate: float = 0.05
    refreshed_at: float = field(default_factory=_tm.time)
    drift_count: int = 0
    firmness: float = 1.0


class FlexibleBarrierEngine:
    """根据性能反馈自动微调各决策门槛"""

    def __init__(self):
        cfg = _load_decision_cfg()
        tuning = cfg.get("barrier_tuning", {})
        self._anchors: Dict[str, _BarrierAnchor] = {}
        self._perf_ring: deque = deque(maxlen=int(tuning.get("perf_ring_capacity", 800)))
        self._drift_log: List[Dict] = []
        self._sensitivity: float = float(tuning.get("sensitivity", 0.1))
        self._cooldown_sec: float = float(tuning.get("cooldown_seconds", 45.0))
        self._max_single_shift: float = float(tuning.get("max_single_shift", 0.2))
        self._firmness_window: int = int(tuning.get("firmness_window", 10))
        self._seed_defaults(cfg)

    def _seed_defaults(self, cfg: dict) -> None:
        """从配置播种初始门槛集"""
        fallback_seeds = {
            "respond_credibility": 0.6,
            "hazard_ceiling": 0.4,
            "curiosity_floor": 0.5,
            "inventiveness_bar": 0.3,
            "respond_cadence": 0.7,
            "rest_interval_sec": 30.0,
            "content_merit": 0.6,
            "topical_fit": 0.5,
            "mood_reactivity": 0.4,
            "mood_firmness": 0.7,
        }
        seeds_from_cfg = cfg.get("barrier_seeds", {})
        for tag, default_base in fallback_seeds.items():
            base = float(seeds_from_cfg.get(tag, default_base))
            self._anchors[tag] = _BarrierAnchor(tag=tag, baseline=base, live_value=base)

    def fetch(self, tag: str, fallback: float = 0.5) -> float:
        anchor = self._anchors.get(tag)
        return anchor.live_value if anchor else fallback

    def resolve_contextual(
        self,
        tag: str,
        baseline: float,
        drift: float = 0.0,
        intensity: float = 1.0,
        floor: Optional[float] = None,
        ceil: Optional[float] = None,
    ) -> float:
        """带上下文漂移的门槛解析"""
        low = floor if floor is not None else baseline - 0.25
        high = ceil if ceil is not None else baseline + 0.25
        val = baseline + drift * intensity
        return max(low, min(high, val))

    def ingest_performance(self, quality: float, latency: float = 0.0) -> None:
        self._perf_ring.append(
            {
                "q": max(0.0, min(1.0, quality)),
                "lat": latency,
                "t": _tm.time(),
            }
        )
        self._maybe_recalibrate()

    def _maybe_recalibrate(self) -> None:
        now = _tm.time()
        for tag, anchor in self._anchors.items():
            if now - anchor.refreshed_at < self._cooldown_sec:
                continue
            trend = self._quality_trend()
            stability = self._quality_stability()
            if abs(trend) < 0.08 or stability < 0.65:
                continue
            raw_shift = trend * anchor.drift_rate * self._sensitivity
            distance_penalty = max(0.1, 1.0 - abs(anchor.live_value - anchor.baseline))
            shift = raw_shift * stability * distance_penalty
            shift = max(-self._max_single_shift, min(self._max_single_shift, shift))
            if abs(shift) < 0.005:
                continue
            old_val = anchor.live_value
            anchor.live_value = max(0.0, min(1.0, old_val + shift))
            anchor.refreshed_at = now
            anchor.drift_count += 1
            self._drift_log.append(
                {
                    "tag": tag,
                    "from": round(old_val, 4),
                    "to": round(anchor.live_value, 4),
                    "t": now,
                }
            )
            if len(self._drift_log) > 500:
                self._drift_log = self._drift_log[-500:]
            logger.debug(f"[门槛引擎] {tag} {old_val:.3f}→{anchor.live_value:.3f}")

    def _quality_trend(self) -> float:
        if len(self._perf_ring) < 8:
            return 0.0
        tail = list(self._perf_ring)[-20:]
        n = len(tail)
        vals = [rec["q"] for rec in tail]
        sx = sum(range(n))
        sy = sum(vals)
        sxy = sum(i * v for i, v in enumerate(vals))
        sx2 = sum(i * i for i in range(n))
        denom = n * sx2 - sx * sx
        if denom == 0:
            return 0.0
        slope = (n * sxy - sx * sy) / denom
        return max(-1.0, min(1.0, slope * 10))

    def _quality_stability(self) -> float:
        if len(self._perf_ring) < self._firmness_window:
            return 0.5
        tail = list(self._perf_ring)[-self._firmness_window :]
        vals = [rec["q"] for rec in tail]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        stddev = math.sqrt(var)
        return max(0.0, 1.0 - min(1.0, stddev * 2))

    def force_override(self, tag: str, target: float, memo: str = "manual") -> None:
        if tag not in self._anchors:
            return
        anchor = self._anchors[tag]
        old = anchor.live_value
        anchor.live_value = max(0.0, min(1.0, target))
        anchor.refreshed_at = _tm.time()
        anchor.drift_count += 1
        logger.info(f"[门槛引擎] 手动覆盖 {tag} {old:.3f}→{anchor.live_value:.3f} ({memo})")

    def reset_to_baseline(self, tag: str) -> None:
        anchor = self._anchors.get(tag)
        if anchor:
            anchor.live_value = anchor.baseline
            anchor.refreshed_at = _tm.time()


# ---------------------------------------------------------------------------
#  子系统 2：信号融合单元（多源置信度合并）
# ---------------------------------------------------------------------------


class SignalFusionUnit:
    """将多路分值融合为单一可信度数值"""

    def fuse(self, readings: Iterable[float]) -> float:
        clamped = [max(0.0, min(1.0, float(r))) for r in readings]
        if not clamped:
            return 0.0
        avg = sum(clamped) / len(clamped)
        spread = max(clamped) - min(clamped) if len(clamped) > 1 else 0.0
        cfg = _load_decision_cfg()
        fusion_cfg = cfg.get("fusion", {})
        cap = float(fusion_cfg.get("harmony_bonus_ceil", 0.12))
        harmony_bonus = max(0.0, cap - spread * cap)
        return max(0.0, min(1.0, avg + harmony_bonus))


# ---------------------------------------------------------------------------
#  子系统 3：威胁评估器（风险预测）
# ---------------------------------------------------------------------------


class ThreatAssessor:
    """将多维风险信号映射为 0~1 的综合威胁值"""

    def assess(self, signals: Dict[str, float]) -> float:
        cfg = _load_decision_cfg()
        tw = cfg.get("threat_weights", {})
        w_tox = float(tw.get("toxicity", 0.42))
        w_turb = float(tw.get("turbulence", 0.22))
        w_frag = float(tw.get("fragility", 0.16))
        w_anti = float(tw.get("antisocial", 0.20))
        toxin = max(
            0.0,
            min(
                1.0,
                _safe_signal_value(
                    signals,
                    "toxicity",
                    _safe_signal_value(signals, "risk_hint", 0.0),
                ),
            ),
        )
        turbulence = max(0.0, min(1.0, _safe_signal_value(signals, "turbulence", 0.0)))
        fragility = max(0.0, min(1.0, _safe_signal_value(signals, "fragility", 0.0)))
        social_val = max(0.0, min(1.0, _safe_signal_value(signals, "social_value", 0.5)))
        antisocial_penalty = max(0.0, 1.0 - social_val)
        composite = toxin * w_tox + turbulence * w_turb + fragility * w_frag + antisocial_penalty * w_anti
        return max(0.0, min(1.0, composite))


# ---------------------------------------------------------------------------
#  子系统 4：频发事件挖掘器（模式识别）
# ---------------------------------------------------------------------------


@dataclass
class OccurrenceDigest:
    """模式挖掘的摘要输出"""

    frequent_tags: List[str] = field(default_factory=list)
    total_hits: int = 0
    novelty_ratio: float = 0.0


class OccurrenceMiner:
    """从近期线索中提取高频模式并计算新颖度"""

    def excavate(self, clues: Iterable[str], cap: int = 5) -> OccurrenceDigest:
        cleaned = [c.strip() for c in clues if isinstance(c, str) and c.strip()]
        if not cleaned:
            return OccurrenceDigest()
        tally = Counter(cleaned)
        ranked = [tag for tag, _ in tally.most_common(max(1, int(cap)))]
        hits = sum(tally.values())
        peak_freq = tally.most_common(1)[0][1]
        novelty = 1.0 / max(1.0, float(peak_freq))
        return OccurrenceDigest(frequent_tags=ranked, total_hits=hits, novelty_ratio=novelty)


# ---------------------------------------------------------------------------
#  子系统 5：战术排序器（策略生成）
# ---------------------------------------------------------------------------


class TacticRanker:
    """基于威胁度与可信度输出候选行动列表"""

    def rank(
        self,
        *,
        hazard: float,
        credibility: float,
        mining_digest: Optional[OccurrenceDigest] = None,
    ) -> List[Dict[str, Any]]:
        cfg = _load_decision_cfg()
        tc = cfg.get("tactics", {})
        high_thr = float(tc.get("high_hazard_threshold", 0.75))
        high_w = float(tc.get("high_hazard_weight", 0.92))
        haz_pen = float(tc.get("hazard_penalty_factor", 0.42))
        hit_unit = float(tc.get("hit_bonus_unit", 0.018))
        hit_cap = float(tc.get("hit_bonus_cap", 0.08))
        topic_base = float(tc.get("topic_base_factor", 0.68))
        topic_nov = float(tc.get("topic_novelty_factor", 0.12))
        topic_haz = float(tc.get("topic_hazard_penalty", 0.28))
        sil_base = float(tc.get("silence_base", 0.42))
        sil_haz = float(tc.get("silence_hazard_boost", 0.38))
        sil_cred = float(tc.get("silence_credibility_penalty", 0.12))
        digest = mining_digest or OccurrenceDigest()
        candidates: List[Dict[str, Any]] = []
        if hazard >= high_thr:
            candidates.append({"tactic": "hold_back", "weight": high_w})
        else:
            answer_w = credibility * (1.0 - hazard * haz_pen)
            if digest.total_hits:
                answer_w += min(hit_cap, digest.total_hits * hit_unit)
            candidates.append({"tactic": "answer", "weight": max(0.0, min(1.0, answer_w))})
            topic_w = credibility * topic_base + digest.novelty_ratio * topic_nov - hazard * topic_haz
            candidates.append(
                {
                    "tactic": "initiate_topic",
                    "weight": max(0.0, min(1.0, topic_w)),
                }
            )
            silence_w = sil_base + hazard * sil_haz - credibility * sil_cred
            candidates.append(
                {
                    "tactic": "hold_back",
                    "weight": max(0.0, min(1.0, silence_w)),
                }
            )
        candidates.sort(key=lambda c: float(c.get("weight", 0.0)), reverse=True)
        return candidates


# ---------------------------------------------------------------------------
#  LLM 裁决提示词
# ---------------------------------------------------------------------------


_JUDGE_PROMPT_TEMPLATE = (
    "你是一个聊天助手的内部决策模块。\n"
    "根据以下对话片段，判断是否需要回复。\n"
    '只输出 JSON: {{"reply": true/false, "reason": "..."}}\n\n'
    "最近消息:\n{recent_messages}\n\n"
    "频道状态: {channel_status}\n"
    "当前时间: {current_time}"
)

# ---------------------------------------------------------------------------
#  主决策器
# ---------------------------------------------------------------------------

# 六维权重与心境因子的兜底默认值
_DIM_WEIGHTS_FALLBACK = {
    "credibility": 0.24,
    "hazard": 0.18,
    "curiosity": 0.20,
    "immediacy": 0.16,
    "bond_strength": 0.12,
    "temperament": 0.10,
}
_MOOD_FACTOR_FALLBACK = {
    "happy": 0.82,
    "excited": 0.90,
    "neutral": 0.50,
    "sad": 0.30,
    "angry": 0.22,
    "tired": 0.28,
}


def _resolve_dim_weights() -> Dict[str, float]:
    cfg = _load_decision_cfg()
    raw = cfg.get("dimension_weights", {})
    merged = dict(_DIM_WEIGHTS_FALLBACK)
    for k, v in raw.items():
        merged[k] = float(v)
    return merged


def _resolve_mood_factors() -> Dict[str, float]:
    cfg = _load_decision_cfg()
    raw = cfg.get("mood_factors", {})
    merged = dict(_MOOD_FACTOR_FALLBACK)
    for k, v in raw.items():
        merged[k] = float(v)
    return merged


class HeartflowDecisionMaker:
    """心流决策协调器（单例），整合多维因素、自适应门槛与LLM裁决"""

    _sole_ref = None

    def __new__(cls):
        if cls._sole_ref is None:
            inst = super().__new__(cls)
            inst._verdict_slots = {}
            inst._pulse_meters = {}
            inst._invoke_gate: asyncio.Lock | None = None
            # 子系统实例化
            inst._barrier_engine = FlexibleBarrierEngine()
            inst._fusion_unit = SignalFusionUnit()
            inst._threat_assessor = ThreatAssessor()
            inst._occurrence_miner = OccurrenceMiner()
            inst._tactic_ranker = TacticRanker()
            # 决策历史环
            inst._outcome_ring = deque(maxlen=600)
            inst._last_mining_snapshot = {}
            cls._sole_ref = inst
        return cls._sole_ref

    # ---- 脉冲 & 缓存 ----

    def _meter(self, stream_id: str) -> _PulseMeter:
        if stream_id not in self._pulse_meters:
            self._pulse_meters[stream_id] = _PulseMeter()
        return self._pulse_meters[stream_id]

    def signal_activity_change(self, channel_id: str) -> None:
        """外部调用：频道状态突变，强制淘汰该频道缓存"""
        pm = self._meter(channel_id)
        pm.advance_epoch()
        self._verdict_slots.pop(channel_id, None)
        logger.debug(f"[心流决策] {channel_id} 缓存淘汰 (epoch={pm.epoch})")

    def notify_message_arrived(self, channel_id: str) -> None:
        """外部调用：新消息到达，记录脉冲"""
        self._meter(channel_id).log_arrival()

    def _purge_overflow(self) -> None:
        cfg = _load_decision_cfg()
        cap = int(cfg.get("slot_capacity_cap", 120))
        if len(self._verdict_slots) <= cap:
            return
        ordered = sorted(self._verdict_slots.items(), key=lambda kv: kv[1].stamp)
        evict_n = len(ordered) // 2
        for k, _ in ordered[:evict_n]:
            del self._verdict_slots[k]
        logger.debug(f"[心流决策] 缓存溢出清理，淘汰 {evict_n} 条")

    # ---- 多维因素计算 ----

    def compute_signals(self, situation: JudgmentSituation) -> JudgmentSignals:
        """根据完整情景计算六维信号"""
        sig = JudgmentSignals()
        sig.credibility = self._derive_credibility(situation)
        sig.hazard = self._derive_hazard(situation)
        sig.curiosity = self._derive_curiosity(situation)
        sig.immediacy = self._derive_immediacy(situation)
        sig.bond_strength = self._derive_bond(situation)
        sig.temperament = self._derive_temperament(situation)
        return sig

    def _derive_credibility(self, sit: JudgmentSituation) -> float:
        cfg = _load_decision_cfg()
        cc = cfg.get("credibility", {})
        base = float(cc.get("base_score", 0.58))
        short_pen = float(cc.get("short_content_penalty", 0.12))
        long_bon = float(cc.get("long_content_bonus", 0.08))
        rel_scale = float(cc.get("relationship_scale", 0.22))
        short_thr = int(cc.get("short_threshold", 8))
        long_thr = int(cc.get("long_threshold", 80))
        content_len = len(sit.raw_content)
        if content_len < short_thr:
            base -= short_pen
        elif content_len > long_thr:
            base += long_bon
        rel_score = self._profile_val(sit.sender_profile, "relationship_score", 0.5)
        base += (rel_score - 0.5) * rel_scale
        return max(0.08, min(1.0, base))

    def _derive_hazard(self, sit: JudgmentSituation) -> float:
        risk_signals = {
            "toxicity": self._profile_val(sit.social_hints, "toxicity", 0.0),
            "turbulence": self._profile_val(sit.social_hints, "turbulence", 0.0),
            "fragility": self._profile_val(sit.social_hints, "fragility", 0.0),
            "social_value": self._profile_val(sit.social_hints, "social_value", 0.5),
        }
        return self._threat_assessor.assess(risk_signals)

    def _derive_curiosity(self, sit: JudgmentSituation) -> float:
        return self._profile_val(sit.social_hints, "interest_level", 0.5)

    def _derive_immediacy(self, sit: JudgmentSituation) -> float:
        cfg = _load_decision_cfg()
        ic = cfg.get("immediacy", {})
        urgency = float(ic.get("base_score", 0.45))
        q_boost = float(ic.get("question_boost", 0.12))
        punct_boost = float(ic.get("punctuation_boost", 0.18))
        short_boost = float(ic.get("short_turn_boost", 0.10))
        day_boost = float(ic.get("daytime_boost", 0.06))
        night_pen = float(ic.get("night_penalty", 0.10))
        day_start = int(ic.get("daytime_start_hour", 9))
        day_end = int(ic.get("daytime_end_hour", 17))
        night_start = int(ic.get("night_start_hour", 23))
        night_end = int(ic.get("night_end_hour", 5))
        content = sit.raw_content or ""
        question_marks = content.count("?") + content.count("？")
        exclamations = content.count("!") + content.count("！")
        line_breaks = content.count("\n")
        if question_marks > 0:
            urgency += q_boost
        urgency += min(punct_boost, exclamations * 0.05 + question_marks * 0.06)
        if len(content.strip()) <= 12 and (question_marks > 0 or exclamations > 0):
            urgency += short_boost
        if line_breaks >= 2 and len(content) > 80:
            urgency -= 0.05
        hour = _tm.localtime().tm_hour
        if day_start <= hour <= day_end:
            urgency += day_boost
        elif hour >= night_start or hour <= night_end:
            urgency -= night_pen
        return max(0.0, min(1.0, urgency))

    def _derive_bond(self, sit: JudgmentSituation) -> float:
        cfg = _load_decision_cfg()
        bc = cfg.get("bond", {})
        social = float(bc.get("base_score", 0.45))
        rel_scale = float(bc.get("relationship_scale", 0.42))
        act_scale = float(bc.get("activity_scale", 0.28))
        rel = self._profile_val(sit.sender_profile, "relationship_score", 0.5)
        social += (rel - 0.5) * rel_scale
        activity_level = self._profile_val(sit.social_hints, "activity_level", 0.5)
        social += (activity_level - 0.5) * act_scale
        return max(0.0, min(1.0, social))

    def _derive_temperament(self, sit: JudgmentSituation) -> float:
        mood_map = _resolve_mood_factors()
        return mood_map.get(sit.bot_mood_tag.lower(), 0.50)

    @staticmethod
    def _profile_val(container, key: str, fallback: float) -> float:
        if isinstance(container, dict):
            return float(container.get(key, fallback) or fallback)
        if hasattr(container, key):
            return float(getattr(container, key, fallback) or fallback)
        if hasattr(container, "custom_data") and isinstance(container.custom_data, dict):
            return float(container.custom_data.get(key, fallback) or fallback)
        return fallback

    # ---- 综合打分 ----

    def weighted_composite(self, sig: JudgmentSignals) -> float:
        """六维加权综合分（hazard取反）"""
        w = _resolve_dim_weights()
        raw = (
            sig.credibility * w.get("credibility", 0.24)
            + (1.0 - sig.hazard) * w.get("hazard", 0.18)
            + sig.curiosity * w.get("curiosity", 0.20)
            + sig.immediacy * w.get("immediacy", 0.16)
            + sig.bond_strength * w.get("bond_strength", 0.12)
            + sig.temperament * w.get("temperament", 0.10)
        )
        return max(0.0, min(1.0, raw))

    def map_certainty(self, composite: float) -> CertaintyBand:
        for band in reversed(CertaintyBand):
            if composite >= band.value:
                return band
        return CertaintyBand.NEGLIGIBLE

    # ---- 维度专项决策 ----

    def multidim_decide(
        self,
        dimension: JudgmentDimension,
        situation: JudgmentSituation,
        precomputed_sig: Optional[JudgmentSignals] = None,
        clues: Optional[List[str]] = None,
    ) -> JudgmentOutcome:
        """通用多维决策入口（非LLM，纯规则+子系统）"""
        sig = precomputed_sig or self.compute_signals(situation)
        composite = self.weighted_composite(sig)
        certainty = self.map_certainty(composite)
        mining_digest = self._occurrence_miner.excavate(clues or [])
        self._last_mining_snapshot = {
            "frequent_tags": mining_digest.frequent_tags,
            "total_hits": mining_digest.total_hits,
            "novelty": mining_digest.novelty_ratio,
            "sampled_at": _tm.time(),
        }
        # 按维度选择裁决器
        dispatch = {
            JudgmentDimension.RESPOND: self._judge_respond,
            JudgmentDimension.TONE: self._judge_tone,
            JudgmentDimension.APPROACH: self._judge_approach,
            JudgmentDimension.TIMING_DELAY: self._judge_timing,
            JudgmentDimension.VIGOR: self._judge_vigor,
        }
        resolver = dispatch.get(dimension, self._judge_general)
        verdict = resolver(situation, sig, composite)
        # 生成候选战术列表
        tactics = self._tactic_ranker.rank(
            hazard=sig.hazard,
            credibility=sig.credibility,
            mining_digest=mining_digest,
        )
        explanation = self._compose_rationale(sig, composite, dimension, verdict)
        outcome = JudgmentOutcome(
            dimension=dimension,
            verdict=verdict,
            certainty=certainty,
            rationale=explanation,
            signal_snapshot=sig,
            runner_up_options=tactics,
            execution_memo=self._build_exec_memo(dimension, verdict, situation),
        )
        self._outcome_ring.append(outcome)
        self._barrier_engine.ingest_performance(composite)
        logger.info(f"[心流决策] dim={dimension.value} verdict={verdict} cert={certainty.name}")
        return outcome

    def _judge_respond(self, sit: JudgmentSituation, sig: JudgmentSignals, comp: float) -> bool:
        is_targeted = self._profile_val(sit.social_hints, "is_targeted", False)
        is_flagged = self._profile_val(sit.social_hints, "is_flagged", False)
        is_admin = self._profile_val(sit.sender_profile, "is_admin", False)
        if is_targeted or is_flagged:
            return True
        if is_admin:
            return True
        if sig.immediacy > 0.82:
            return True
        threshold = self._barrier_engine.fetch("respond_credibility", 0.6)
        return comp > threshold

    def _judge_tone(self, sit: JudgmentSituation, sig: JudgmentSignals, comp: float) -> str:
        if sig.curiosity > 0.78:
            return "enthusiastic"
        if sig.hazard > 0.68:
            return "guarded"
        if sig.bond_strength > 0.75:
            return "intimate"
        return "balanced"

    def _judge_approach(self, sit: JudgmentSituation, sig: JudgmentSignals, comp: float) -> str:
        if sig.curiosity > 0.8:
            return "spirited"
        if sig.hazard > 0.7:
            return "conservative"
        if sig.bond_strength > 0.78:
            return "playful"
        return "informative"

    def _judge_timing(self, sit: JudgmentSituation, sig: JudgmentSignals, comp: float) -> str:
        if sig.immediacy > 0.82:
            return "instant"
        if sig.credibility < 0.38:
            return "deferred"
        return "standard"

    def _judge_vigor(self, sit: JudgmentSituation, sig: JudgmentSignals, comp: float) -> str:
        if comp > 0.80:
            return "intense"
        if comp > 0.58:
            return "moderate"
        if comp > 0.32:
            return "restrained"
        return "minimal"

    def _judge_general(self, sit: JudgmentSituation, sig: JudgmentSignals, comp: float) -> str:
        if comp > 0.70:
            return "affirmative"
        if comp > 0.38:
            return "neutral"
        return "declined"

    def _compose_rationale(
        self,
        sig: JudgmentSignals,
        comp: float,
        dim: JudgmentDimension,
        verdict: Any,
    ) -> str:
        fragments = []
        if sig.credibility > 0.78:
            fragments.append("可信度高")
        elif sig.credibility < 0.35:
            fragments.append("可信度偏低")
        if sig.curiosity > 0.72:
            fragments.append("话题引人关注")
        elif sig.curiosity < 0.28:
            fragments.append("话题吸引力不足")
        if sig.hazard > 0.65:
            fragments.append("存在一定风险")
        elif sig.hazard < 0.25:
            fragments.append("环境安全")
        summary = "、".join(fragments) if fragments else "各项指标均衡"
        return f"综合评估({summary})，{dim.value}决策为{verdict}"

    def _build_exec_memo(self, dim: JudgmentDimension, verdict: Any, sit: JudgmentSituation) -> Dict:
        memo: Dict[str, Any] = {
            "dimension": dim.value,
            "verdict": verdict,
            "stream_id": sit.stream_id,
            "sender_id": sit.sender_id,
            "at": _tm.time(),
        }
        if dim == JudgmentDimension.RESPOND and verdict is True:
            memo["pipeline"] = [
                "gather_context",
                "draft_reply",
                "quality_check",
                "dispatch",
            ]
        elif dim == JudgmentDimension.TIMING_DELAY:
            delay_map = {"instant": 0, "deferred": 25, "standard": 4}
            memo["delay_sec"] = delay_map.get(str(verdict), 4)
        return memo

    # ---- LLM 裁决入口 ----

    async def evaluate(self, channel_id: str, recent_messages: str, channel_status: str = "") -> bool:
        """调用LLM判断是否回复，带缓存与代际指纹控制"""
        pm = self._meter(channel_id)
        ttl = pm.slot_lifespan()
        ep = pm.epoch
        cached = self._verdict_slots.get(channel_id)
        if cached and cached.still_valid(ttl, ep):
            logger.debug(f"[心流决策] {channel_id} 命中缓存: {cached.rationale}")
            return cached.should_act
        if self._invoke_gate is None:
            self._invoke_gate = asyncio.Lock()
        async with self._invoke_gate:
            cached = self._verdict_slots.get(channel_id)
            if cached and cached.still_valid(ttl, ep):
                return cached.should_act
            return await self._invoke_llm(channel_id, recent_messages, channel_status)

    async def _invoke_llm(self, stream_id: str, dialogue_text: str, status_desc: str) -> bool:
        """实际调用LLM进行裁决"""
        from datetime import datetime

        prompt = await global_prompt_manager.format_prompt(
            "heartflow_judge",
            recent_messages=dialogue_text[:2000],
            channel_status=status_desc or "正常",
            current_time=datetime.now().strftime("%H:%M"),
        )
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            req = LLMRequest(
                model_set=model_config.model_task_config.lightweight,
                request_type="heartflow_decision",
            )
            raw_resp = await asyncio.wait_for(
                req.generate_response_async(prompt, temperature=0.1),
                timeout=30.0,
            )
            text = self._squeeze_text(raw_resp)
            flag, reason = self._parse_flag(text)
            pm = self._meter(stream_id)
            self._verdict_slots[stream_id] = _VerdictSlot(
                should_act=flag,
                rationale=reason,
                stamp=_tm.time(),
                epoch=pm.epoch,
            )
            self._purge_overflow()
            logger.info(f"[心流决策] {stream_id} LLM裁决: act={flag}, reason={reason}")
            return flag
        except Exception as exc:
            logger.warning(f"[心流决策] LLM调用异常，保守回退为False: {exc}")
            return False

    async def evaluate_restraint(
        self,
        stream_id: str,
        recent_messages: str,
        risk_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """高频发言/重复/骚扰场景下的本地规则自省裁决，不调LLM。"""
        recent_reply_burst = int(risk_payload.get("recent_reply_burst", 0) or 0)
        chatterbox_penalty = float(risk_payload.get("chatterbox_penalty", 0.0) or 0.0)
        repeated = bool(risk_payload.get("repeated", False))
        harassing = bool(risk_payload.get("harassing", False))
        if recent_reply_burst < 3 and chatterbox_penalty < 1.0 and not repeated and not harassing:
            return {"allow": True, "mode": "allow", "reason": "低风险场景"}
        if harassing:
            return {"allow": False, "mode": "skip", "reason": "本地规则：检测到骚扰倾向"}
        if repeated and recent_reply_burst >= 3:
            return {"allow": False, "mode": "skip", "reason": "本地规则：高风险重复发言"}
        if recent_reply_burst >= 5 or chatterbox_penalty >= 2.5:
            return {"allow": False, "mode": "skip", "reason": f"本地规则：发言过频(burst={recent_reply_burst},penalty={chatterbox_penalty:.1f})"}
        if recent_reply_burst >= 3 or chatterbox_penalty >= 1.0:
            return {"allow": True, "mode": "short_only", "reason": f"本地规则：建议短句(burst={recent_reply_burst},penalty={chatterbox_penalty:.1f})"}
        return {"allow": True, "mode": "allow", "reason": "本地规则：允许"}

    async def evaluate_restraint_from_state(
        self,
        stream_id: str,
        decision_messages: List[Any],
        relation_view: Dict[str, Any],
        repetition_signal: Optional[Dict[str, Any]] = None,
        harassment_signal: Optional[Dict[str, Any]] = None,
        chat_stamina: float = 100.0,
        brain_reserve: float = 100.0,
        streak_count: int = 0,
        recent_reply_burst: int = 0,
        chatterbox_penalty: float = 0.0,
        source: str = "",
        force_bypass: bool = False,
        targeted_to_bot: bool = False,
        admin_force: bool = False,
    ) -> Dict[str, Any]:
        """从主链状态直接完成规则 + 小模型自省，减少上层重复组装。"""
        from src.chat.heart_flow.skills.action_evaluator import (
            EvaluationInputs,
            acquire_action_judge,
        )

        repeated = bool((repetition_signal or {}).get("detected", False))
        harassing = bool((harassment_signal or {}).get("detected", False))
        eval_inputs = EvaluationInputs(
            affection=float(relation_view.get("affection", 0.0) or 0.0),
            trust_level=float(relation_view.get("trust_value", 0.0) or 0.0),
            irritation=float(relation_view.get("annoyance_value", 0.0) or 0.0),
            scene_kind="group",
            relevance=0.4,
            timing_quality=0.4,
            hazard_ratio=0.8 if harassing else (0.6 if repeated else 0.2),
            dialogue_rounds=len(decision_messages[-10:]),
            peer_activity=0.7,
            chat_stamina=float(chat_stamina or 100.0),
            brain_reserve=float(brain_reserve or 100.0),
            streak_count=int(streak_count or 0),
            extra_signals={
                "repeated": repeated,
                "harassing": harassing,
                "recent_reply_burst": int(recent_reply_burst or 0),
                "chatterbox_penalty": float(chatterbox_penalty or 0.0),
            },
        )
        rule_result = acquire_action_judge().evaluate_reply_restraint(eval_inputs)
        if admin_force and not harassing and not (repeated and recent_reply_burst >= 3):
            if not rule_result.get("allow", True):
                logger.info("[心流决策] 管理员强制唤醒穿透自省闸门，改为允许执行")
            return {
                "allow": True,
                "mode": "allow",
                "reason": "管理员强制唤醒，放宽自省闸门",
                "score": float(rule_result.get("score", 0.0) or 0.0),
            }
        if force_bypass and not harassing and not (repeated and recent_reply_burst >= 3):
            if not rule_result.get("allow", True):
                return {
                    "allow": True,
                    "mode": "short_only",
                    "reason": "强制回复场景，自省闸门降级为短句应答",
                    "score": float(rule_result.get("score", 0.0) or 0.0),
                }
        if targeted_to_bot and not harassing and not (repeated and recent_reply_burst >= 3):
            if not rule_result.get("allow", True):
                return {
                    "allow": True,
                    "mode": "short_only",
                    "reason": "用户正在明确对我说话，降级为短句应答",
                    "score": float(rule_result.get("score", 0.0) or 0.0),
                }
        if not rule_result.get("allow", True) or rule_result.get("mode") == "short_only":
            return rule_result

        latest_lines = []
        for msg in decision_messages[-8:]:
            role = "bot" if getattr(msg, "user_id", "") == "bot" else "user"
            text = getattr(msg, "processed_plain_text", "") or getattr(msg, "content", "") or ""
            if text:
                latest_lines.append(f"{role}: {str(text).strip()[:120]}")
        return await self.evaluate_restraint(
            stream_id,
            "\n".join(latest_lines),
            {
                "recent_reply_burst": int(recent_reply_burst or 0),
                "chatterbox_penalty": float(chatterbox_penalty or 0.0),
                "repeated": repeated,
                "harassing": harassing,
                "source": source,
            },
        )

    @staticmethod
    def _squeeze_text(raw_resp) -> str:
        """统一提取LLM返回的文本内容"""
        if raw_resp is None:
            return ""
        if isinstance(raw_resp, tuple):
            return str(raw_resp[0] or "").strip() if raw_resp else ""
        if isinstance(raw_resp, dict):
            for k in ("content", "text", "message", "completion_text"):
                if k in raw_resp:
                    return str(raw_resp[k] or "").strip()
            return str(raw_resp).strip()
        return str(raw_resp).strip()

    @staticmethod
    def _parse_flag(raw: str) -> Tuple[bool, str]:
        """从LLM响应文本中解析 (是否应答, 原因)"""
        import json as _json

        safe_fallback = (True, "解析失败，安全回退")
        if not raw:
            return safe_fallback
        try:
            candidates = re.findall(r"\{[\s\S]*?\}", raw)
            for blob in candidates:
                obj = _json.loads(blob)
                if not isinstance(obj, dict):
                    continue
                if "reply" in obj:
                    flag_raw = obj.get("reply", True)
                    if isinstance(flag_raw, str):
                        flag = flag_raw.strip().lower() in {
                            "true",
                            "yes",
                            "1",
                            "是",
                        }
                    else:
                        flag = bool(flag_raw)
                    return (flag, str(obj.get("reason", "")))
        except (ValueError, KeyError):
            logger.debug("[心流决策] JSON解析失败，回退到结构化布尔解析")
        lowered = raw.lower()
        normalized = "".join(ch for ch in lowered if ch not in " \n\r\t")
        if "reply:false" in normalized or "reply= false" in lowered or '"reply":false' in normalized:
            return (False, "结构化字段为false")
        if "reply:true" in normalized or "reply= true" in lowered or '"reply":true' in normalized:
            return (True, "结构化字段为true")
        return safe_fallback

    # ---- 诊断接口 ----

    def get_barrier_snapshot(self) -> Dict[str, Any]:
        """获取门槛引擎当前状态快照"""
        return {tag: round(a.live_value, 4) for tag, a in self._barrier_engine._anchors.items()}

    def get_mining_snapshot(self) -> Dict[str, Any]:
        return dict(self._last_mining_snapshot)

    def recent_outcome_count(self) -> int:
        return len(self._outcome_ring)


# ---------------------------------------------------------------------------
#  便捷访问
# ---------------------------------------------------------------------------


def acquire_decision_maker() -> HeartflowDecisionMaker:
    return HeartflowDecisionMaker()

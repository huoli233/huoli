import asyncio
import json
import math
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

from src.chat.heart_flow.dimension_protocol import (
    DimensionBase,
    DimensionScope,
    EventContext,
    TickResult,
)
from src.chat.heart_flow.vote_types import (
    TraumaVote,
    TraumaGrade,
    SurfaceMaskVote,
    MaskBehaviorMode,
)
from src.common.logger import get_logger
from src.config.core_config_engine import get_core_config

logger = get_logger("wound_network")


class DisorderGrade(Enum):
    """心理状态等级（9 + 4 特殊）"""

    TRANQUIL = "tranquil"
    MILD_RIPPLE = "mild_ripple"
    FACADE_CRACK = "facade_crack"
    FORCED_CALM = "forced_calm"
    FACADE_TREMOR = "facade_tremor"
    PARTIAL_FRACTURE = "partial_fracture"
    HANGING_THREAD = "hanging_thread"
    EDGE_OF_ABYSS = "edge_of_abyss"
    FULL_COLLAPSE = "full_collapse"
    FLASHBACK_SURGE = "flashback_surge"
    WITHDRAWAL_SHELL = "withdrawal_shell"
    ERUPTION_BURST = "eruption_burst"
    FROZEN_VOID = "frozen_void"


@dataclass
class PsycheShard:
    """创伤碎片"""

    shard_id: str = ""
    inception_context: str = ""
    emotional_voltage: float = 0.0
    distortion_factor: float = 0.0
    recall_clarity: float = 5.0
    forged_ts: float = field(default_factory=time.time)


@dataclass
class WorldviewDamage:
    """世界观受损指标"""

    trust_quotient: float = 10.0
    reality_warp: float = 0.0
    meaning_erosion: float = 0.0
    cognitive_scatter: float = 0.0
    compromised_beliefs: List[str] = field(default_factory=list)


@dataclass
class TurmoilLedger:
    """完整创伤状态账本"""

    inner_chaos: float = 0.0
    mask_wear: float = 0.0
    surface_armor: float = 8.0
    active_grade: DisorderGrade = DisorderGrade.TRANQUIL
    shards: List[PsycheShard] = field(default_factory=list)
    worldview: WorldviewDamage = field(default_factory=WorldviewDamage)
    chaos_worldview_contrib: float = 0.0
    chaos_shard_contrib: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inner_chaos": round(self.inner_chaos, 4),
            "mask_wear": round(self.mask_wear, 4),
            "surface_armor": round(self.surface_armor, 4),
            "active_grade": self.active_grade.value,
            "shard_count": len(self.shards),
            "worldview": {
                "trust_quotient": round(self.worldview.trust_quotient, 2),
                "reality_warp": round(self.worldview.reality_warp, 2),
                "meaning_erosion": round(self.worldview.meaning_erosion, 2),
                "cognitive_scatter": round(
                    self.worldview.cognitive_scatter, 2
                ),
            },
        }


@dataclass
class MindProfile:
    """多层心理描写"""

    exterior_mask: str = ""
    aware_layer: str = ""
    buried_layer: str = ""
    echo_fragments: List[str] = field(default_factory=list)
    silent_monologue: str = ""
    somatic_signs: str = ""
    worldview_distort: str = ""
    flight_impulse: str = ""
    composed_ts: float = field(default_factory=time.time)
    source_grade: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exterior_mask": self.exterior_mask,
            "aware_layer": self.aware_layer,
            "buried_layer": self.buried_layer,
            "echo_fragments": list(self.echo_fragments),
            "silent_monologue": self.silent_monologue,
            "somatic_signs": self.somatic_signs,
            "worldview_distort": self.worldview_distort,
            "flight_impulse": self.flight_impulse,
            "source_grade": self.source_grade,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MindProfile":
        return cls(
            exterior_mask=str(
                data.get("exterior_mask", data.get("surface_layer", ""))
            ),
            aware_layer=str(
                data.get("aware_layer", data.get("conscious_layer", ""))
            ),
            buried_layer=str(
                data.get("buried_layer", data.get("subconscious_layer", ""))
            ),
            echo_fragments=list(
                data.get("echo_fragments", data.get("trauma_echoes", []))
            ),
            silent_monologue=str(
                data.get("silent_monologue", data.get("inner_monologue", ""))
            ),
            somatic_signs=str(
                data.get("somatic_signs", data.get("physical_symptoms", ""))
            ),
            worldview_distort=str(
                data.get(
                    "worldview_distort", data.get("worldview_distortion", "")
                )
            ),
            flight_impulse=str(
                data.get("flight_impulse", data.get("escape_urge", ""))
            ),
            source_grade=str(
                data.get("source_grade", data.get("source_state", ""))
            ),
        )


@dataclass
class TriggerVerdict:
    """特殊状态触发判定"""

    fired: bool = False
    target_grade: Optional[DisorderGrade] = None
    confidence: float = 0.0
    explanation: str = ""
    judged_ts: float = field(default_factory=time.time)


@dataclass
class _TrendSample:
    chaos: float
    armor: float
    grade: DisorderGrade
    ts: float


class WoundNetwork:
    """
    创伤心理网络

    整合混乱度合成、状态映射、多层心理描写生成、
    特殊状态触发检测、碎片刺激评估、世界观冲击及趋势衰减。
    """

    _solo: Optional["WoundNetwork"] = None

    @classmethod
    def instance(cls) -> "WoundNetwork":
        if cls._solo is None:
            cls._solo = cls()
        return cls._solo

    @classmethod
    def teardown(cls) -> None:
        if cls._solo:
            cls._solo._flush_disk()
        cls._solo = None

    def __init__(self):
        self._ledger = TurmoilLedger()
        self._timeline: Deque[_TrendSample] = deque(maxlen=500)
        self._profile_cache: Dict[str, MindProfile] = {}
        self._cooldown_map: Dict[str, float] = {}
        self._shard_eval_ts: Dict[str, float] = {}
        self._conf_cache: Dict[str, Any] = {}
        self._conf_ts: float = 0.0
        self._storage_path: str = "data/huoli/wound_state.json"
        self._refresh_conf()
        self._read_disk()

    # ---- 配置 ----

    def _refresh_conf(self) -> None:
        now = time.time()
        if now - self._conf_ts < 60.0:
            return
        hub = get_core_config()
        ps = hub.resolve_module_view("personality").values
        self._conf_cache = {
            "chaos_sensitivity": float(ps.get("chaos_sensitivity", 0.8)),
            "mask_strain_rate": float(ps.get("mask_strain_rate", 0.3)),
            "mask_recovery_rate": float(ps.get("mask_recovery_rate", 0.05)),
            "mask_base_armor": float(ps.get("mask_base_armor", 8.0)),
            "natural_decay_per_hour": float(
                ps.get("chaos_natural_decay_per_hour", 0.1)
            ),
            "profile_cache_sec": float(ps.get("profile_cache_seconds", 60.0)),
            "shard_eval_interval": float(
                ps.get("shard_eval_interval_sec", 120.0)
            ),
            "trigger_confidence_floor": float(
                ps.get("trigger_confidence_floor", 0.6)
            ),
            "trigger_cooldown_sec": float(
                ps.get("trigger_cooldown_sec", 300.0)
            ),
            # 混乱度到等级的映射阈值
            "grade_thresholds": {
                "tranquil": float(ps.get("grade_tranquil_cap", 1.0)),
                "mild_ripple": float(ps.get("grade_mild_ripple_cap", 2.0)),
                "facade_crack": float(ps.get("grade_facade_crack_cap", 3.0)),
                "forced_calm": float(ps.get("grade_forced_calm_cap", 4.5)),
                "facade_tremor": float(ps.get("grade_facade_tremor_cap", 5.5)),
                "partial_fracture": float(
                    ps.get("grade_partial_fracture_cap", 6.5)
                ),
                "hanging_thread": float(
                    ps.get("grade_hanging_thread_cap", 7.5)
                ),
                "edge_of_abyss": float(ps.get("grade_edge_of_abyss_cap", 8.5)),
            },
            # 特殊触发阈值
            "flashback_min_chaos": float(ps.get("flashback_min_chaos", 5.0)),
            "flashback_min_shards": int(ps.get("flashback_min_shards", 1)),
            "withdrawal_min_chaos": float(ps.get("withdrawal_min_chaos", 3.0)),
            "eruption_min_chaos": float(ps.get("eruption_min_chaos", 6.0)),
            "eruption_max_armor": float(ps.get("eruption_max_armor", 4.0)),
            "frozen_min_chaos": float(ps.get("frozen_min_chaos", 9.0)),
            "frozen_max_armor": float(ps.get("frozen_max_armor", 1.5)),
        }
        self._conf_ts = now

    # ---- 状态读取 ----

    def ledger(self) -> TurmoilLedger:
        return self._ledger

    def current_grade(self) -> DisorderGrade:
        return self._ledger.active_grade

    def chaos_level(self) -> float:
        return self._ledger.inner_chaos

    # ---- 混乱度合成 ----

    def absorb_stimulus(
        self,
        severity: float,
        sentiment: str,
        sentiment_intensity: float,
        is_harassment: bool,
    ) -> TurmoilLedger:
        """
        将外部刺激的定性信号合成为混乱度变化

        参数:
            severity: 严重度 0~1
            sentiment: 情感极性 (negative/positive/neutral)
            sentiment_intensity: 情感强度 0~1
            is_harassment: 是否为骚扰
        """
        self._refresh_conf()
        sens = self._conf_cache["chaos_sensitivity"]
        strain_rate = self._conf_cache["mask_strain_rate"]
        recovery = self._conf_cache["mask_recovery_rate"]
        sev = max(0.0, min(1.0, float(severity)))
        sint = max(0.0, min(1.0, float(sentiment_intensity)))
        neg_portion = sint if sentiment == "negative" else sint * 0.15
        harass_portion = 1.0 if is_harassment else 0.0
        # 世界观贡献
        wv = self._ledger.worldview
        wv_dist = float(wv.reality_warp) + float(wv.meaning_erosion) * 0.6
        wv_scatter = float(wv.cognitive_scatter) * 0.4
        self._ledger.chaos_worldview_contrib = min(
            5.0, (wv_dist + wv_scatter) * 0.35
        )
        # 混乱度变化公式
        delta = sens * (
            0.45 * sev
            + 0.30 * harass_portion
            + 0.13 * neg_portion
            + 0.12 * self._ledger.chaos_worldview_contrib
        )
        self._ledger.inner_chaos = max(
            0.0,
            min(10.0, self._ledger.inner_chaos + delta - recovery * 0.02),
        )
        # 面具疲劳
        self._ledger.mask_wear = min(
            10.0,
            self._ledger.mask_wear
            + strain_rate * 0.01 * self._ledger.inner_chaos,
        )
        base_armor = self._conf_cache["mask_base_armor"]
        self._ledger.surface_armor = max(
            0.0,
            min(10.0, base_armor - self._ledger.mask_wear + recovery),
        )
        # 映射等级
        self._ledger.active_grade = self._resolve_grade(
            self._ledger.inner_chaos
        )
        # 记录时间线
        self._timeline.append(
            _TrendSample(
                chaos=self._ledger.inner_chaos,
                armor=self._ledger.surface_armor,
                grade=self._ledger.active_grade,
                ts=time.time(),
            )
        )
        # 自然衰减
        decayed = self._compute_natural_decay(self._ledger.inner_chaos)
        if decayed != self._ledger.inner_chaos:
            self._ledger.inner_chaos = decayed
            self._ledger.active_grade = self._resolve_grade(decayed)
        self._flush_disk()
        return self._ledger

    def soothe(self, intensity: float = 0.1) -> None:
        """外部积极事件触发的冷却"""
        recovery = self._conf_cache.get("mask_recovery_rate", 0.05)
        a = max(0.0, min(1.0, intensity))
        self._ledger.inner_chaos = max(
            0.0,
            self._ledger.inner_chaos - a * recovery * 2.0,
        )
        self._ledger.active_grade = self._resolve_grade(
            self._ledger.inner_chaos
        )
        self._flush_disk()

    def attach_shard(self, shard: PsycheShard) -> None:
        """附加创伤碎片"""
        self._ledger.shards.append(shard)
        bonus = min(3.0, float(shard.emotional_voltage) * 0.1)
        self._ledger.chaos_shard_contrib = min(
            5.0, self._ledger.chaos_shard_contrib + bonus
        )
        self._flush_disk()

    def _resolve_grade(self, chaos: float) -> DisorderGrade:
        th = self._conf_cache.get("grade_thresholds", {})
        c = max(0.0, min(10.0, chaos))
        if c < th.get("tranquil", 1.0):
            return DisorderGrade.TRANQUIL
        if c < th.get("mild_ripple", 2.0):
            return DisorderGrade.MILD_RIPPLE
        if c < th.get("facade_crack", 3.0):
            return DisorderGrade.FACADE_CRACK
        if c < th.get("forced_calm", 4.5):
            return DisorderGrade.FORCED_CALM
        if c < th.get("facade_tremor", 5.5):
            return DisorderGrade.FACADE_TREMOR
        if c < th.get("partial_fracture", 6.5):
            return DisorderGrade.PARTIAL_FRACTURE
        if c < th.get("hanging_thread", 7.5):
            return DisorderGrade.HANGING_THREAD
        if c < th.get("edge_of_abyss", 8.5):
            return DisorderGrade.EDGE_OF_ABYSS
        return DisorderGrade.FULL_COLLAPSE

    def _compute_natural_decay(self, chaos: float) -> float:
        if not self._timeline or chaos <= 0:
            return chaos
        latest = self._timeline[-1]
        hours = (time.time() - latest.ts) / 3600.0
        if hours <= 0:
            return chaos
        rate = self._conf_cache.get("natural_decay_per_hour", 0.1)
        return max(0.0, chaos - rate * hours)

    # ---- 多层心理描写 ----

    async def compose_mind_profile(self) -> MindProfile:
        """生成当前状态的完整心理层描写"""
        cache_key = f"{
            self._ledger.active_grade.value}_{
            self._ledger.inner_chaos:.0f}_{
            self._ledger.surface_armor:.0f}"
        limit = self._conf_cache.get("profile_cache_sec", 60.0)
        cached = self._profile_cache.get(cache_key)
        if cached and (time.time() - cached.composed_ts) < limit:
            return cached
        shard_digest = self._condense_shards()
        wv_digest = self._condense_worldview()
        prompt = (
            f"[心理层描写任务]\n"
            f"当前状态：{self._ledger.active_grade.value}\n"
            f"混乱度：{self._ledger.inner_chaos:.1f}/10\n"
            f"面具强度：{self._ledger.surface_armor:.1f}/10\n"
            f"创伤碎片：{shard_digest}\n"
            f"世界观受损：{wv_digest}\n\n"
            f"请输出 JSON，包含以下 8 个维度的自然语言描写：\n"
            f'{{"exterior_mask": "表面伪装", "aware_layer": "意识层描写", '
            f'"buried_layer": "潜意识层描写", "echo_fragments": ["创伤回声1","创伤回声2"], '
            f'"silent_monologue": "内心独白", "somatic_signs": "躯体症状", '
            f'"worldview_distort": "世界观扭曲", "flight_impulse": "逃避冲动"}}'
        )
        raw = await self._call_llm(prompt)
        if raw:
            parsed = self._extract_json(raw)
            if parsed:
                profile = MindProfile.from_dict(parsed)
                profile.composed_ts = time.time()
                profile.source_grade = self._ledger.active_grade.value
                self._profile_cache[cache_key] = profile
                return profile
        # 兜底
        fallback = MindProfile(
            exterior_mask=f"当前状态：{self._ledger.active_grade.value}",
            source_grade=self._ledger.active_grade.value,
        )
        return fallback

    def _condense_shards(self) -> str:
        if not self._ledger.shards:
            return "无活跃碎片"
        parts = []
        for s in self._ledger.shards[:5]:
            v_tag = (
                "低"
                if s.emotional_voltage < 3
                else ("中" if s.emotional_voltage < 7 else "高")
            )
            parts.append(f"[电压:{v_tag}] {s.inception_context[:60]}")
        return "；".join(parts)

    def _condense_worldview(self) -> str:
        wv = self._ledger.worldview
        parts = []
        if wv.trust_quotient < 8.0:
            parts.append(f"信任降至{wv.trust_quotient:.1f}/10")
        if wv.reality_warp > 2.0:
            parts.append(f"现实扭曲{wv.reality_warp:.1f}")
        if wv.meaning_erosion > 2.0:
            parts.append(f"意义侵蚀{wv.meaning_erosion:.1f}")
        if wv.cognitive_scatter > 2.0:
            parts.append(f"认知碎散{wv.cognitive_scatter:.1f}")
        return "；".join(parts) if parts else "世界观基本完好"

    # ---- 特殊状态触发检测 ----

    async def probe_special_trigger(
        self,
        recent_dialogue: str = "",
    ) -> TriggerVerdict:
        """检测是否应触发特殊创伤状态"""
        empty = TriggerVerdict(fired=False)
        shard_count = len(self._ledger.shards)
        if not self._precheck_trigger_eligible(shard_count):
            return empty
        prompt = (
            f"[特殊创伤状态评估]\n"
            f"混乱度：{self._ledger.inner_chaos:.1f}/10\n"
            f"面具强度：{self._ledger.surface_armor:.1f}/10\n"
            f"基准等级：{self._ledger.active_grade.value}\n"
            f"碎片数量：{shard_count}\n"
            f"最近对话：{recent_dialogue[:1000]}\n\n"
            f"可能触发的特殊状态：闪回发作、回避退缩、应激爆发、冻结虚空\n"
            f"请判断是否应触发。输出 JSON：\n"
            f'{{"should_trigger": true/false, "target_state": "状态名", '
            f'"confidence": 0.0到1.0, "rationale": "理由"}}'
        )
        raw = await self._call_llm(prompt)
        if not raw:
            return empty
        parsed = self._extract_json(raw)
        if not parsed:
            return empty
        should = parsed.get("should_trigger", False)
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        floor = self._conf_cache.get("trigger_confidence_floor", 0.6)
        if not should or confidence < floor:
            return TriggerVerdict(
                fired=False,
                confidence=confidence,
                explanation=parsed.get("rationale", ""),
            )
        target_name = str(parsed.get("target_state", ""))
        target_grade = self._map_special_name(target_name)
        if not target_grade:
            return empty
        cooldown = self._conf_cache.get("trigger_cooldown_sec", 300.0)
        if self._on_cooldown(target_grade.value, cooldown):
            return TriggerVerdict(fired=False, explanation="冷却中")
        self._cooldown_map[target_grade.value] = time.time()
        logger.info(
            f"[伤网] 触发特殊状态 {target_grade.value} | 置信={confidence:.2f}"
        )
        return TriggerVerdict(
            fired=True,
            target_grade=target_grade,
            confidence=confidence,
            explanation=parsed.get("rationale", ""),
        )

    def _precheck_trigger_eligible(self, shard_count: int) -> bool:
        chaos = self._ledger.inner_chaos
        armor = self._ledger.surface_armor
        c = self._conf_cache
        if (
            chaos >= c["flashback_min_chaos"]
            and shard_count >= c["flashback_min_shards"]
        ):
            return True
        if chaos >= c["withdrawal_min_chaos"]:
            return True
        if (
            chaos >= c["eruption_min_chaos"]
            and armor <= c["eruption_max_armor"]
        ):
            return True
        if chaos >= c["frozen_min_chaos"] and armor <= c["frozen_max_armor"]:
            return True
        return False

    def _on_cooldown(self, key: str, limit: float) -> bool:
        last = self._cooldown_map.get(key, 0.0)
        return (time.time() - last) < limit

    @staticmethod
    def _map_special_name(name: str) -> Optional[DisorderGrade]:
        mapping = {
            "闪回发作": DisorderGrade.FLASHBACK_SURGE,
            "flashback": DisorderGrade.FLASHBACK_SURGE,
            "回避退缩": DisorderGrade.WITHDRAWAL_SHELL,
            "avoidance": DisorderGrade.WITHDRAWAL_SHELL,
            "应激爆发": DisorderGrade.ERUPTION_BURST,
            "outbreak": DisorderGrade.ERUPTION_BURST,
            "冻结虚空": DisorderGrade.FROZEN_VOID,
            "dazed": DisorderGrade.FROZEN_VOID,
        }
        return mapping.get(name.lower().strip())

    # ---- 碎片刺激评估 ----

    async def evaluate_shard_stimulus(
        self,
        content: str,
        context: Dict[str, Any],
    ) -> Optional[PsycheShard]:
        """评估内容是否构成新的创伤碎片"""
        stream_key = str(context.get("stream_id", "global"))
        interval = self._conf_cache.get("shard_eval_interval", 120.0)
        now = time.time()
        last = self._shard_eval_ts.get(stream_key, 0.0)
        if interval > 0 and (now - last) < interval:
            return None
        self._shard_eval_ts[stream_key] = now
        prompt = (
            f"[碎片刺激评估]\n"
            f"输入内容：{content[:2000]}\n"
            f"上下文：{str(context)[:500]}\n\n"
            f"请判断此内容是否构成可记录的创伤刺激。输出 JSON：\n"
            f'{{"should_record": true/false, "emotional_charge": 0到10, '
            f'"distortion_level": 0到10, "memory_clarity": 0到10, '
            f'"semantic_context": "语义描述"}}'
        )
        raw = await self._call_llm(prompt)
        if not raw:
            return None
        parsed = self._extract_json(raw)
        if not parsed or not parsed.get("should_record", False):
            return None
        shard = PsycheShard(
            shard_id=uuid.uuid4().hex[:16],
            inception_context=str(parsed.get("semantic_context", ""))[:800],
            emotional_voltage=max(
                0.0, min(10.0, float(parsed.get("emotional_charge", 0.0)))
            ),
            distortion_factor=max(
                0.0, min(10.0, float(parsed.get("distortion_level", 0.0)))
            ),
            recall_clarity=max(
                0.0, min(10.0, float(parsed.get("memory_clarity", 5.0)))
            ),
        )
        self.attach_shard(shard)
        logger.info(
            f"[伤网] 新碎片 | 电压={
                shard.emotional_voltage:.1f} 扭曲={
                shard.distortion_factor:.1f}"
        )
        return shard

    # ---- 世界观冲击 ----

    def apply_worldview_impact(
        self,
        trust_delta: float = 0.0,
        warp_delta: float = 0.0,
        erosion_delta: float = 0.0,
        scatter_delta: float = 0.0,
        belief_damage: Optional[str] = None,
    ) -> None:
        """更新世界观受损指标"""
        wv = self._ledger.worldview
        wv.trust_quotient = max(
            0.0, min(10.0, wv.trust_quotient + trust_delta)
        )
        wv.reality_warp = max(0.0, min(10.0, wv.reality_warp + warp_delta))
        wv.meaning_erosion = max(
            0.0, min(10.0, wv.meaning_erosion + erosion_delta)
        )
        wv.cognitive_scatter = max(
            0.0, min(10.0, wv.cognitive_scatter + scatter_delta)
        )
        if belief_damage and belief_damage not in wv.compromised_beliefs:
            wv.compromised_beliefs.append(belief_damage)
            if len(wv.compromised_beliefs) > 10:
                wv.compromised_beliefs = wv.compromised_beliefs[-10:]
        self._flush_disk()

    # ---- 趋势分析 ----

    def recent_trend(self, window_sec: float = 3600.0) -> str:
        """分析近期混乱度趋势"""
        cutoff = time.time() - window_sec
        recent = [s for s in self._timeline if s.ts > cutoff]
        if len(recent) < 2:
            return "stable"
        first_half = recent[: len(recent) // 2]
        second_half = recent[len(recent) // 2:]
        avg_first = sum(s.chaos for s in first_half) / max(1, len(first_half))
        avg_second = sum(s.chaos for s in second_half) / max(
            1, len(second_half)
        )
        diff = avg_second - avg_first
        if diff > 1.5:
            return "escalating"
        if diff < -1.5:
            return "recovering"
        return "stable"

    # ---- 持久化 ----

    def _flush_disk(self) -> None:
        try:
            os.makedirs(
                os.path.dirname(self._storage_path) or ".", exist_ok=True
            )
            payload = {
                "inner_chaos": self._ledger.inner_chaos,
                "mask_wear": self._ledger.mask_wear,
                "surface_armor": self._ledger.surface_armor,
                "active_grade": self._ledger.active_grade.value,
                "worldview": {
                    "trust_quotient": self._ledger.worldview.trust_quotient,
                    "reality_warp": self._ledger.worldview.reality_warp,
                    "meaning_erosion": self._ledger.worldview.meaning_erosion,
                    "cognitive_scatter": self._ledger.worldview.cognitive_scatter,
                    "compromised_beliefs": self._ledger.worldview.compromised_beliefs,
                },
                "shard_count": len(self._ledger.shards),
                "flushed_at": time.time(),
            }
            with open(self._storage_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except (OSError, TypeError) as exc:
            logger.debug(f"[伤网] 磁盘写入失败: {exc}")

    def _read_disk(self) -> None:
        if not os.path.exists(self._storage_path):
            return
        try:
            with open(self._storage_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._ledger.inner_chaos = float(data.get("inner_chaos", 0.0))
            self._ledger.mask_wear = float(data.get("mask_wear", 0.0))
            self._ledger.surface_armor = float(data.get("surface_armor", 8.0))
            grade_str = data.get("active_grade", "tranquil")
            try:
                self._ledger.active_grade = DisorderGrade(grade_str)
            except ValueError:
                self._ledger.active_grade = DisorderGrade.TRANQUIL
            wv_data = data.get("worldview", {})
            if isinstance(wv_data, dict):
                self._ledger.worldview.trust_quotient = float(
                    wv_data.get("trust_quotient", 10.0)
                )
                self._ledger.worldview.reality_warp = float(
                    wv_data.get("reality_warp", 0.0)
                )
                self._ledger.worldview.meaning_erosion = float(
                    wv_data.get("meaning_erosion", 0.0)
                )
                self._ledger.worldview.cognitive_scatter = float(
                    wv_data.get("cognitive_scatter", 0.0)
                )
                self._ledger.worldview.compromised_beliefs = list(
                    wv_data.get("compromised_beliefs", [])
                )
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.debug(f"[伤网] 磁盘读取失败: {exc}")

    # ---- LLM ----

    async def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            req = LLMRequest(
                model_config.model_task_config.focus_chat,
                request_type="wound_network",
            )
            text, _ = await asyncio.wait_for(
                req.generate_response_async(prompt),
                timeout=30.0,
            )
            return text
        except asyncio.TimeoutError:
            logger.error("[伤网] LLM 超时(30s)")
            return None

    @staticmethod
    def _extract_json(raw: str) -> Optional[Dict]:
        import re

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            try:
                fixed = re.sub(r",\s*}", "}", match.group())
                return json.loads(fixed)
            except Exception as _exc:
                logger.warning(f"创伤织布JSON解析失败: {_exc}")
                return None

    def collect_metrics(self) -> Dict[str, Any]:
        return {
            "chaos": round(self._ledger.inner_chaos, 2),
            "armor": round(self._ledger.surface_armor, 2),
            "grade": self._ledger.active_grade.value,
            "shard_count": len(self._ledger.shards),
            "trend": self.recent_trend(),
            "timeline_samples": len(self._timeline),
        }


def get_wound_network() -> WoundNetwork:
    """获取全局创伤心理网络实例"""
    return WoundNetwork.instance()


# 混乱度 → 四档创伤等级
_CHAOS_GRADE_BOUNDARIES = [
    (0.0, 2.0, TraumaGrade.NONE),
    (2.0, 5.0, TraumaGrade.MILD),
    (5.0, 8.0, TraumaGrade.SEVERE),
    (8.0, 10.1, TraumaGrade.EXTREME),
]

# 创伤等级 → 概率乘数
_TRAUMA_PROB_MAP = {
    TraumaGrade.NONE: 1.0,
    TraumaGrade.MILD: 0.85,
    TraumaGrade.SEVERE: 0.55,
    TraumaGrade.EXTREME: 0.2,
}


def _classify_trauma(chaos: float) -> TraumaGrade:
    for lo, hi, grade in _CHAOS_GRADE_BOUNDARIES:
        if lo <= chaos < hi:
            return grade
    return TraumaGrade.EXTREME


class TraumaDimension(DimensionBase):
    """
    D10 创伤核心维度。
    包装 WoundNetwork 的混乱度/碎片/世界观部分，
    输出 TraumaVote 供决策网关使用。
    全局粒度（创伤状态不区分频道）。
    """

    _singleton: Optional["TraumaDimension"] = None

    @classmethod
    def get_instance(cls) -> "TraumaDimension":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        self._wound_ref: Optional[WoundNetwork] = None
        # 上次闪回评估时间（限制频率）
        self._last_flashback_eval_ts: float = 0.0
        self._flashback_eval_interval: float = 60.0
        # 最近闪回风险值
        self._cached_flashback_risk: float = 0.0
        # 最近触发的碎片标签
        self._triggered_labels: list[str] = []

    def _wound(self) -> WoundNetwork:
        """懒获取全局 WoundNetwork 实例"""
        if self._wound_ref is None:
            self._wound_ref = WoundNetwork.instance()
        return self._wound_ref

    @property
    def dimension_name(self) -> str:
        return "trauma"

    @property
    def scope(self) -> DimensionScope:
        return DimensionScope.GLOBAL

    @property
    def needs_persistence(self) -> bool:
        # WoundNetwork 自己管理持久化（_flush_disk/_read_disk）
        return False

    @property
    def tick_interval_sec(self) -> float:
        return 10.0

    def tick(self, elapsed_sec: float) -> TickResult:
        """定期执行自然衰减"""
        wound = self._wound()
        ledger = wound.ledger()
        old_chaos = ledger.inner_chaos
        # 自然衰减在 absorb_stimulus 中已执行，这里只做趋势检查
        trend = wound.recent_trend(window_sec=600.0)
        return TickResult(
            dimension_name=self.dimension_name,
            updated=True,
            summary=f"chaos={old_chaos:.2f} trend={trend}",
        )

    def on_event(self, ctx: EventContext):
        """
        消息到达时评估是否构成创伤刺激。
        需要外部通过 raw_extras 传入：
          - severity: 严重度 (0~1)
          - sentiment: 情感极性 (negative/positive/neutral)
          - sentiment_intensity: 情感强度 (0~1)
          - is_harassment: 是否骚扰
        """
        if ctx.event_type != "message_received":
            return
        wound = self._wound()
        severity = float(ctx.raw_extras.get("severity", 0.0))
        sentiment = str(ctx.raw_extras.get("sentiment", "neutral"))
        sent_intensity = float(ctx.raw_extras.get("sentiment_intensity", 0.0))
        is_harass = bool(ctx.raw_extras.get("is_harassment", False))
        # 只在有实际负面刺激时调用
        if severity > 0.1 or sentiment == "negative" or is_harass:
            wound.absorb_stimulus(
                severity=severity,
                sentiment=sentiment,
                sentiment_intensity=sent_intensity,
                is_harassment=is_harass,
            )
        elif sentiment == "positive" and sent_intensity > 0.3:
            # 正面刺激可以缓解混乱度
            wound.soothe(intensity=sent_intensity * 0.5)

    def vote(self, ctx: EventContext) -> TraumaVote:
        """
        创伤投票：
          - 混乱度映射为四档
          - 认知损伤度随混乱度非线性增长
          - 闪回风险根据碎片数量和混乱度评估
        """
        wound = self._wound()
        ledger = wound.ledger()
        chaos = ledger.inner_chaos
        grade = _classify_trauma(chaos)
        prob = _TRAUMA_PROB_MAP.get(grade, 1.0)
        # 认知损伤：sigmoid映射 chaos 0~10 → impairment 0~1
        cognitive = 1.0 / (1.0 + math.exp(-1.2 * (chaos - 5.0)))
        # 闪回风险：碎片数×混乱度的归一化
        shard_count = len(ledger.shards)
        raw_flashback = min(1.0, (chaos / 10.0) * min(shard_count, 5) / 5.0)
        # 取缓存值和实时计算的较大值
        flashback_risk = max(raw_flashback, self._cached_flashback_risk)
        # 碎片标签
        frag_labels = []
        for shard in ledger.shards[:5]:
            if shard.inception_context:
                frag_labels.append(shard.inception_context[:40])
        # 潜意识干扰描述
        perception = []
        if chaos > 3.0:
            perception.append("思维偶尔被不相关的记忆打断")
        if chaos > 5.0:
            perception.append("注意力难以集中在当前对话")
        if chaos > 7.0:
            perception.append("强烈的不安感涌上来")
        if cognitive > 0.5:
            perception.append("表达能力受到干扰")
        # 态度标签
        attitude = ""
        if grade == TraumaGrade.EXTREME:
            attitude = "trauma_shutdown"
        elif grade == TraumaGrade.SEVERE:
            attitude = "trauma_impaired"
        elif grade == TraumaGrade.MILD:
            attitude = "trauma_uneasy"
        # 强制拒绝：极端创伤时
        force = grade == TraumaGrade.EXTREME and chaos > 9.0
        # token 上限
        token_cap = 0
        if grade == TraumaGrade.SEVERE:
            token_cap = 200
        elif grade == TraumaGrade.EXTREME:
            token_cap = 80
        return TraumaVote(
            probability_factor=round(prob, 3),
            force_refuse=force,
            attitude_tag=attitude,
            max_tokens_cap=token_cap,
            trauma_index=round(chaos, 2),
            trauma_grade=grade,
            cognitive_impairment=round(cognitive, 3),
            flashback_risk=round(flashback_risk, 3),
            triggered_fragment_labels=frag_labels,
            perception_labels=perception,
            inner_chaos=round(chaos * 10.0, 1),
            debug_reason=(
                f"chaos={chaos:.2f} grade={grade.value} "
                f"cognitive={cognitive:.2f} shards={shard_count}"
            ),
        )

    def serialize(self) -> dict:
        """序列化创伤维度状态。

        保存闪回评估时间戳和缓存的闪回风险值，
        用于重启后恢复状态。
        """
        return {
            "last_flashback_eval_ts": self._last_flashback_eval_ts,
            "cached_flashback_risk": self._cached_flashback_risk,
            "triggered_labels": self._triggered_labels.copy(),
        }

    def deserialize(self, data: dict):
        """从持久化数据恢复创伤维度状态。

        恢复闪回评估时间戳和风险值，确保重启后状态连续。
        """
        if not data:
            return
        if "last_flashback_eval_ts" in data:
            self._last_flashback_eval_ts = data["last_flashback_eval_ts"]
        if "cached_flashback_risk" in data:
            self._cached_flashback_risk = data["cached_flashback_risk"]
        if "triggered_labels" in data and isinstance(data["triggered_labels"], list):
            self._triggered_labels = data["triggered_labels"][:10]  # 限制数量

    def calibrate(self, offline_seconds: float):
        """离线校准：混乱度自然衰减"""
        wound = self._wound()
        hours = offline_seconds / 3600.0
        rate = wound._conf_cache.get("natural_decay_per_hour", 0.1)
        ledger = wound.ledger()
        decay = rate * hours
        ledger.inner_chaos = max(0.0, ledger.inner_chaos - decay)
        ledger.active_grade = wound._resolve_grade(ledger.inner_chaos)
        # 面具也部分恢复
        recovery_rate = wound._conf_cache.get("mask_recovery_rate", 0.05)
        ledger.mask_wear = max(0.0, ledger.mask_wear - recovery_rate * hours * 10)
        base_armor = wound._conf_cache.get("mask_base_armor", 8.0)
        ledger.surface_armor = max(
            0.0, min(10.0, base_armor - ledger.mask_wear + recovery_rate)
        )

    def get_state_summary(self) -> dict:
        wound = self._wound()
        ledger = wound.ledger()
        return {
            "dimension": self.dimension_name,
            "scope": self.scope.value,
            "chaos": round(ledger.inner_chaos, 2),
            "grade": ledger.active_grade.value,
            "shard_count": len(ledger.shards),
            "worldview_trust": round(ledger.worldview.trust_quotient, 2),
            "trend": wound.recent_trend(),
        }

    def reset(self, user_id: str = "", channel_id: str = ""):
        wound = self._wound()
        ledger = wound.ledger()
        ledger.inner_chaos = 0.0
        ledger.mask_wear = 0.0
        ledger.surface_armor = wound._conf_cache.get("mask_base_armor", 8.0)
        ledger.active_grade = DisorderGrade.TRANQUIL
        ledger.shards.clear()


# ============================================================
# D11 维度包装：表面面具/伪装系统
# ============================================================
# 九宫格矩阵（行=混乱度等级, 列=面具强度等级）
# 每格对应一个 MaskBehaviorMode
_MATRIX_3X3 = [
    # 面具弱(0)    面具中(1)    面具强(2)
    [MaskBehaviorMode.NATURAL, MaskBehaviorMode.NATURAL, MaskBehaviorMode.MASKING],     # 混乱低(0)
    [MaskBehaviorMode.CRACKING, MaskBehaviorMode.MASKING, MaskBehaviorMode.MASKING],    # 混乱中(1)
    [MaskBehaviorMode.BROKEN, MaskBehaviorMode.CRACKING, MaskBehaviorMode.CRACKING],    # 混乱高(2)
]


def _classify_row(chaos: float) -> int:
    """混乱度→行号: 0=低(<3), 1=中(3~6), 2=高(>=6)"""
    if chaos < 3.0:
        return 0
    if chaos < 6.0:
        return 1
    return 2


def _classify_col(armor: float) -> int:
    """面具强度→列号: 0=弱(<3), 1=中(3~7), 2=强(>=7)"""
    if armor < 3.0:
        return 0
    if armor < 7.0:
        return 1
    return 2


class SurfaceMaskDimension(DimensionBase):
    """
    D11 表面面具维度。
    包装 WoundNetwork 的 mask_wear / surface_armor 部分，
    输出 SurfaceMaskVote 供决策网关使用。
    面具决定外在表现与内在状态的裂隙——
    面具完好时，即使内心混乱也能维持正常回复；
    面具崩塌时，内心混乱会泄露到回复中。
    全局粒度。
    """

    _singleton: Optional["SurfaceMaskDimension"] = None

    @classmethod
    def get_instance(cls) -> "SurfaceMaskDimension":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        self._wound_ref: Optional[WoundNetwork] = None

    def _wound(self) -> WoundNetwork:
        if self._wound_ref is None:
            self._wound_ref = WoundNetwork.instance()
        return self._wound_ref

    @property
    def dimension_name(self) -> str:
        return "surface_mask"

    @property
    def scope(self) -> DimensionScope:
        return DimensionScope.GLOBAL

    @property
    def needs_persistence(self) -> bool:
        return False

    @property
    def tick_interval_sec(self) -> float:
        return 10.0

    def tick(self, elapsed_sec: float) -> TickResult:
        """面具自然恢复"""
        wound = self._wound()
        ledger = wound.ledger()
        recovery = wound._conf_cache.get("mask_recovery_rate", 0.05)
        # 每次tick缓慢恢复面具耐久
        old_wear = ledger.mask_wear
        if ledger.mask_wear > 0 and ledger.inner_chaos < 3.0:
            ledger.mask_wear = max(0.0, ledger.mask_wear - recovery * (elapsed_sec / 60.0))
            base_armor = wound._conf_cache.get("mask_base_armor", 8.0)
            ledger.surface_armor = max(0.0, min(10.0, base_armor - ledger.mask_wear))
        return TickResult(
            dimension_name=self.dimension_name,
            updated=old_wear != ledger.mask_wear,
            summary=f"wear={ledger.mask_wear:.2f} armor={ledger.surface_armor:.2f}",
        )

    def on_event(self, ctx: EventContext):
        """
        消息事件对面具的影响：
        - 回复时面具消耗（越是强撑回复，面具磨损越快）
        - 正面反馈可以轻微恢复面具
        """
        if ctx.event_type == "reply_completed":
            wound = self._wound()
            ledger = wound.ledger()
            strain = wound._conf_cache.get("mask_strain_rate", 0.3)
            # 回复一次消耗面具，消耗量与当前混乱度正相关
            strain_amount = strain * 0.02 * max(1.0, ledger.inner_chaos)
            ledger.mask_wear = min(10.0, ledger.mask_wear + strain_amount)
            base_armor = wound._conf_cache.get("mask_base_armor", 8.0)
            ledger.surface_armor = max(
                0.0, min(10.0, base_armor - ledger.mask_wear)
            )
        elif ctx.event_type == "message_received":
            # 正面情绪轻微恢复面具
            if ctx.sentiment_score > 0.5:
                wound = self._wound()
                ledger = wound.ledger()
                ledger.mask_wear = max(0.0, ledger.mask_wear - 0.01)

    def vote(self, ctx: EventContext) -> SurfaceMaskVote:
        """
        面具投票：
          - 九宫格(混乱度×面具强度)决定行为模式
          - 面具完好 → 正常概率
          - 面具破裂 → 降低回复质量上限 + 注入内心冲突描述
        """
        wound = self._wound()
        ledger = wound.ledger()
        chaos = ledger.inner_chaos
        armor = ledger.surface_armor
        wear = ledger.mask_wear
        # 九宫格定位
        row = _classify_row(chaos)
        col = _classify_col(armor)
        behavior = _MATRIX_3X3[row][col]
        # 面具强度百分比
        mask_strength = armor * 10.0
        # 面具疲劳度百分比
        fatigue = wear * 10.0
        # 是否正在破裂
        is_breaking = behavior in (MaskBehaviorMode.CRACKING, MaskBehaviorMode.BROKEN)
        # 概率因子：面具状态不直接影响是否回复，而是影响回复质量
        # 但极端情况（BROKEN + 高混乱）降低概率
        prob = 1.0
        if behavior == MaskBehaviorMode.BROKEN and chaos > 7.0:
            prob = 0.5
        elif behavior == MaskBehaviorMode.CRACKING and chaos > 5.0:
            prob = 0.8
        # 回复质量上限
        quality_cap = 1.0
        if behavior == MaskBehaviorMode.BROKEN:
            quality_cap = 0.4
        elif behavior == MaskBehaviorMode.CRACKING:
            quality_cap = 0.7
        elif behavior == MaskBehaviorMode.MASKING:
            quality_cap = 0.9
        # 被识破风险
        detection_risk = 0.0
        if behavior == MaskBehaviorMode.MASKING:
            detection_risk = 0.1 + chaos * 0.03
        elif behavior == MaskBehaviorMode.CRACKING:
            detection_risk = 0.3 + chaos * 0.05
        elif behavior == MaskBehaviorMode.BROKEN:
            detection_risk = 0.7 + chaos * 0.03
        detection_risk = min(1.0, detection_risk)
        # 内心冲突描述
        conflict_hint = ""
        if behavior == MaskBehaviorMode.MASKING and chaos > 3.0:
            conflict_hint = "表面维持着微笑，但内心暗流涌动"
        elif behavior == MaskBehaviorMode.CRACKING:
            conflict_hint = "伪装出现裂缝，某些真实情绪可能泄露"
        elif behavior == MaskBehaviorMode.BROKEN:
            conflict_hint = "无法继续伪装，内心的痛苦直接表露"
        # 态度标签
        attitude = ""
        if behavior == MaskBehaviorMode.BROKEN:
            attitude = "mask_collapsed"
        elif behavior == MaskBehaviorMode.CRACKING:
            attitude = "mask_cracking"
        # 风格提示
        style = ""
        if behavior == MaskBehaviorMode.MASKING:
            style = "维持表面正常、稍显刻意"
        elif behavior == MaskBehaviorMode.CRACKING:
            style = "偶尔流露出不一致、话语间有裂痕"
        elif behavior == MaskBehaviorMode.BROKEN:
            style = "情绪失控、真实想法涌出"
        return SurfaceMaskVote(
            probability_factor=round(prob, 3),
            attitude_tag=attitude,
            style_hint=style,
            mask_strength=round(mask_strength, 1),
            behavior_mode=behavior,
            is_breaking=is_breaking,
            reply_quality_cap=round(quality_cap, 2),
            mask_fatigue=round(fatigue, 1),
            detection_risk=round(detection_risk, 3),
            inner_conflict_hint=conflict_hint,
            matrix_row=row,
            matrix_col=col,
            debug_reason=(
                f"chaos={chaos:.2f} armor={armor:.2f} wear={wear:.2f} "
                f"mode={behavior.value} row={row} col={col}"
            ),
        )

    def serialize(self) -> dict:
        return {}

    def deserialize(self, data: dict):
        pass

    def calibrate(self, offline_seconds: float):
        """离线后面具恢复"""
        wound = self._wound()
        ledger = wound.ledger()
        hours = offline_seconds / 3600.0
        recovery = wound._conf_cache.get("mask_recovery_rate", 0.05)
        ledger.mask_wear = max(0.0, ledger.mask_wear - recovery * hours * 5.0)
        base_armor = wound._conf_cache.get("mask_base_armor", 8.0)
        ledger.surface_armor = max(0.0, min(10.0, base_armor - ledger.mask_wear))

    def get_state_summary(self) -> dict:
        wound = self._wound()
        ledger = wound.ledger()
        row = _classify_row(ledger.inner_chaos)
        col = _classify_col(ledger.surface_armor)
        return {
            "dimension": self.dimension_name,
            "scope": self.scope.value,
            "armor": round(ledger.surface_armor, 2),
            "wear": round(ledger.mask_wear, 2),
            "behavior": _MATRIX_3X3[row][col].value,
            "matrix": f"[{row},{col}]",
        }

    def reset(self, user_id: str = "", channel_id: str = ""):
        wound = self._wound()
        ledger = wound.ledger()
        ledger.mask_wear = 0.0
        base_armor = wound._conf_cache.get("mask_base_armor", 8.0)
        ledger.surface_armor = base_armor

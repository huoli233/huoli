import time
import string
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger
from src.common.singleton import _get_class_lock

logger = get_logger("planner_injection")

_injection_singleton = None
_injection_lock = _get_class_lock("PlannerPromptInjectionLayer")


def _safe_relation_metric(relation: Dict[str, Any], key: str, default: float) -> float:
    """读取关系数值时只在缺失/None 时回退默认值，保留合法的 0 和负值。"""
    value = relation.get(key, None)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_numeric_metric(value: Any, default: float) -> float:
    """读取一般数值时只在缺失时回退默认值，保留合法的 0。"""
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass
class SituationInterpretationInput:
    """GAP-L 情境解释输入"""

    situation_reading: str = ""
    situation_atmosphere: str = ""
    internal_mood: str = ""
    internal_energy: float = 0.5
    social_comfort: float = 0.5
    engagement_posture: str = ""
    engagement_intensity: float = 0.5
    style_hint: str = ""
    risk_assessment_score: float = 0.0
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class GossipStrategyInput:
    """GAP-O 吃瓜策略输入"""

    event_type_label: str = ""
    posture_label: str = ""
    should_act: bool = False
    priority: float = 0.0
    reply_hint: str = ""
    tone_hint: str = ""
    narration_hint: str = ""


@dataclass
class VisibilitySummaryInput:
    """GAP-M 可见性摘要输入"""

    total_evaluated: int = 0
    high_priority_count: int = 0
    low_score_count: int = 0
    avg_modulated_score: float = 0.0
    understanding_rate: float = 0.0
    top_reasons: List[str] = field(default_factory=list)
    user_bias_summary: Dict[str, float] = field(default_factory=dict)


@dataclass
class SafetyAssessmentInput:
    """GAP-R 安全评估输入"""

    safety_level: str = ""
    safety_score: float = 0.0
    should_block: bool = False
    dominant_threat: str = ""
    recommended_action: str = ""
    bar_delta: float = 0.0


@dataclass
class MemoryWindowInput:
    """GAP-Q 记忆窗口输入"""

    window_size: int = 0
    total_chars: int = 0
    tier_distribution: Dict[str, int] = field(default_factory=dict)
    memory_snippets: List[str] = field(default_factory=list)
    cutoff_reason: str = ""


@dataclass
class FreshnessStateInput:
    """GAP-N 新鲜度状态输入"""

    total_tracked: int = 0
    fresh_count: int = 0
    stale_count: int = 0
    recent_recalls: int = 0
    decay_tier_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class InjectionPacket:
    """完整注入数据包——所有GAP机制的聚合输出"""

    channel_id: str = ""
    timestamp: float = field(default_factory=time.time)
    situation: Optional[SituationInterpretationInput] = None
    gossip_strategy: Optional[GossipStrategyInput] = None
    visibility: Optional[VisibilitySummaryInput] = None
    safety: Optional[SafetyAssessmentInput] = None
    memory_window: Optional[MemoryWindowInput] = None
    freshness_state: Optional[FreshnessStateInput] = None
    raw_metabolism: Optional[Dict[str, Any]] = None
    raw_night_phase: Optional[str] = None
    raw_relation_snapshot: Optional[Dict[str, Any]] = None
    raw_watch_state: Optional[str] = None

    def has_content(self) -> bool:
        return (
            self.situation is not None
            or self.gossip_strategy is not None
            or self.visibility is not None
            or self.safety is not None
            or self.memory_window is not None
            or self.freshness_state is not None
        )


@dataclass
class InjectedPromptResult:
    """注入后的提示词结果"""

    system_context_block: str = ""
    situation_block: str = ""
    strategy_block: str = ""
    visibility_block: str = ""
    safety_block: str = ""
    memory_block: str = ""
    freshness_block: str = ""
    raw_state_block: str = ""
    full_prompt: str = ""
    injection_count: int = 0
    total_chars: int = 0
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "injection_count": self.injection_count,
            "total_chars": self.total_chars,
            "has_situation": bool(self.situation_block),
            "has_strategy": bool(self.strategy_block),
            "has_visibility": bool(self.visibility_block),
            "has_safety": bool(self.safety_block),
            "has_memory": bool(self.memory_block),
            "full_preview": self.full_prompt[:200],
        }


class PlannerPromptInjectionLayer:
    """LLM Planner 提示词注入层（GAP-S）
    将所有GAP机制输出统一聚合并结构化注入到planner提示词中。
    输出7个独立模块的提示词块，可按需组合：
    1. 主观情境块 (GAP-L) —— 行为前的主观判断
    2. 事件策略块 (GAP-O) —— 当前事件的参与策略
    3. 可见性块 (GAP-M) —— 消息关注分布
    4. 安全边界块 (GAP-R) —— 安全评估与约束
    5. 记忆窗口块 (GAP-Q) —— 动态上下文记忆
    6. 新鲜度块 (GAP-N) —— 心理新鲜度状态
    7. 原始状态块 —— 代谢/夜间/关系/观看态原始数据
    """

    def __init__(self):
        self._recent_packets: List[InjectionPacket] = []
        self._max_packets = 30

    def build_packet(
        self,
        channel_id: str,
        *,
        situation: Optional[Dict[str, Any]] = None,
        gossip_verdict: Optional[Dict[str, Any]] = None,
        visibility_stats: Optional[Dict[str, Any]] = None,
        safety_result: Optional[Dict[str, Any]] = None,
        memory_window: Optional[Dict[str, Any]] = None,
        freshness_snapshot: Optional[Dict[str, Any]] = None,
        metabolism_raw: Optional[Dict[str, Any]] = None,
        night_phase: Optional[str] = None,
        relation_snap: Optional[Dict[str, Any]] = None,
        watch_state: Optional[str] = None,
    ) -> InjectionPacket:
        packet = InjectionPacket(channel_id=channel_id)
        if situation:
            packet.situation = SituationInterpretationInput(
                situation_reading=str(
                    situation.get("situation_reading", "") or ""
                ),
                situation_atmosphere=str(
                    situation.get("atmosphere", "") or ""
                ),
                internal_mood=str(situation.get("mood", "") or ""),
                internal_energy=_safe_numeric_metric(
                    situation.get("energy"), 0.5
                ),
                social_comfort=_safe_numeric_metric(
                    situation.get("social_comfort"), 0.5
                ),
                engagement_posture=str(situation.get("posture", "") or ""),
                engagement_intensity=_safe_numeric_metric(
                    situation.get("intensity"), 0.5
                ),
                style_hint=str(situation.get("style_hint", "") or ""),
                risk_assessment_score=_safe_numeric_metric(
                    situation.get("risk_score"), 0.0
                ),
                risk_factors=list(situation.get("risk_factors") or []),
            )
        if gossip_verdict:
            packet.gossip_strategy = GossipStrategyInput(
                event_type_label=str(
                    gossip_verdict.get("event_type", "")
                    or gossip_verdict.get("posture_label", "")
                    or ""
                ),
                posture_label=str(
                    gossip_verdict.get("posture_label", "") or ""
                ),
                should_act=bool(gossip_verdict.get("should_act", False)),
                priority=float(gossip_verdict.get("priority", 0.0) or 0.0),
                reply_hint=str(gossip_verdict.get("reply_hint", "") or ""),
                tone_hint=str(gossip_verdict.get("tone_hint", "") or ""),
                narration_hint=str(gossip_verdict.get("narration", "") or ""),
            )
        if visibility_stats:
            packet.visibility = VisibilitySummaryInput(
                total_evaluated=int(
                    visibility_stats.get("total_evaluated", 0) or 0
                ),
                high_priority_count=int(
                    visibility_stats.get("high_priority", 0) or 0
                ),
                low_score_count=int(visibility_stats.get("low_score", 0) or 0),
                avg_modulated_score=float(
                    visibility_stats.get("avg_modulated_score", 0.0) or 0.0
                ),
                understanding_rate=float(
                    visibility_stats.get("understanding_rate", 0.0) or 0.0
                ),
                top_reasons=list(visibility_stats.get("reasons", []) or [])[
                    :5
                ],
                user_bias_summary=dict(
                    visibility_stats.get("bias_breakdown", {}) or {}
                ),
            )
        if safety_result:
            packet.safety = SafetyAssessmentInput(
                safety_level=str(safety_result.get("level", "") or ""),
                safety_score=float(safety_result.get("score", 0.0) or 0.0),
                should_block=bool(safety_result.get("block_reply", False)),
                dominant_threat=str(safety_result.get("dominant", "") or ""),
                recommended_action=str(safety_result.get("action", "") or ""),
                bar_delta=float(safety_result.get("bar_delta", 0.0) or 0.0),
            )
        if memory_window:
            packet.memory_window = MemoryWindowInput(
                window_size=int(memory_window.get("window_size", 0) or 0),
                total_chars=int(memory_window.get("total_chars", 0) or 0),
                tier_distribution=dict(
                    memory_window.get("tier_dist", {}) or {}
                ),
                memory_snippets=list(memory_window.get("snippets", []) or [])[
                    :8
                ],
                cutoff_reason=str(memory_window.get("reason", "") or ""),
            )
        if freshness_snapshot:
            packet.freshness_state = FreshnessStateInput(
                total_tracked=int(
                    freshness_snapshot.get("total_tracked", 0) or 0
                ),
                fresh_count=int(freshness_snapshot.get("fresh_count", 0) or 0),
                stale_count=int(freshness_snapshot.get("stale_count", 0) or 0),
                recent_recalls=int(freshness_snapshot.get("recalls", 0) or 0),
                decay_tier_distribution=dict(
                    freshness_snapshot.get("tier_dist", {}) or {}
                ),
            )
        packet.raw_metabolism = metabolism_raw
        packet.raw_night_phase = night_phase
        packet.raw_relation_snapshot = relation_snap
        packet.raw_watch_state = watch_state
        self._recent_packets.append(packet)
        if len(self._recent_packets) > self._max_packets:
            self._recent_packets = self._recent_packets[-self._max_packets:]
        return packet

    def generate_full_prompt(
        self, packet: InjectionPacket
    ) -> InjectedPromptResult:
        result = InjectedPromptResult()
        blocks: List[Tuple[str, str]] = []
        sys_ctx = self._build_system_context(packet)
        if sys_ctx:
            result.system_context_block = sys_ctx
            blocks.append(("系统上下文", sys_ctx))
        sit_blk = self._build_situation_block(packet.situation)
        if sit_blk:
            result.situation_block = sit_blk
            blocks.append(("主观情境", sit_blk))
        strat_blk = self._build_strategy_block(packet.gossip_strategy)
        if strat_blk:
            result.strategy_block = strat_blk
            blocks.append(("事件策略", strat_blk))
        vis_blk = self._build_visibility_block(packet.visibility)
        if vis_blk:
            result.visibility_block = vis_blk
            blocks.append(("可见性分布", vis_blk))
        safe_blk = self._build_safety_block(packet.safety)
        if safe_blk:
            result.safety_block = safe_blk
            blocks.append(("安全边界", safe_blk))
        mem_blk = self._build_memory_block(packet.memory_window)
        if mem_blk:
            result.memory_block = mem_blk
            blocks.append(("记忆窗口", mem_blk))
        fresh_blk = self._build_freshness_block(packet.freshness_state)
        if fresh_blk:
            result.freshness_block = fresh_blk
            blocks.append(("新鲜度状态", fresh_blk))
        raw_blk = self._build_raw_state_block(
            packet.raw_metabolism,
            packet.raw_night_phase,
            packet.raw_relation_snapshot,
            packet.raw_watch_state,
        )
        if raw_blk:
            result.raw_state_block = raw_blk
            blocks.append(("原始状态", raw_blk))
        separator = "\n\n---\n\n"
        full = separator.join(b[1] for b in blocks)
        result.full_prompt = full
        result.injection_count = len(blocks)
        result.total_chars = len(full)
        logger.debug(
            f"[Planner注入] 生成完成: {result.injection_count}个模块, "
            f"{result.total_chars}字符"
        )
        return result

    def generate_compact_prompt(
        self, packet: InjectionPacket, max_chars: int = 2000
    ) -> str:
        parts: List[str] = []
        current_len = 0
        if packet.situation and current_len < max_chars:
            blk = f"[情境] {packet.situation.engagement_posture or '观察'}"
            if packet.situation.risk_assessment_score > 0.3:
                blk += f" 风险={packet.situation.risk_assessment_score:.2f}"
            parts.append(blk)
            current_len += len(blk) + 1
        if packet.gossip_strategy and current_len < max_chars:
            blk = f"[策略] {packet.gossip_strategy.posture_label or '无'}"
            if packet.gossip_strategy.should_act:
                blk += f" → {packet.gossip_strategy.reply_hint[:30]}"
            parts.append(blk)
            current_len += len(blk) + 1
        if packet.safety and current_len < max_chars:
            blk = f"[安全] {packet.safety.safety_level}"
            if packet.safety.should_block:
                blk += " ⚠️阻断"
            parts.append(blk)
            current_len += len(blk) + 1
        if packet.visibility and current_len < max_chars:
            blk = f"[可见性] 高优={
                packet.visibility.high_priority_count}, 理解率={
                packet.visibility.understanding_rate:.0%}"
            parts.append(blk)
            current_len += len(blk) + 1
        if packet.freshness_state and current_len < max_chars:
            blk = f"[新鲜度] 新鲜={
                packet.freshness_state.fresh_count}, 过期={
                packet.freshness_state.stale_count}"
            if packet.freshness_state.recent_recalls > 0:
                blk += f", 回看={packet.freshness_state.recent_recalls}"
            parts.append(blk)
            current_len += len(blk) + 1
        return "\n".join(parts)

    # ═════════════════ 各模块构建方法 ═════════════════

    @staticmethod
    def _build_system_context(packet: InjectionPacket) -> str:
        lines = ["## 系统上下文"]
        lines.append(f"频道: {packet.channel_id[:20]}")
        lines.append(
            f"时间戳: {
                time.strftime(
                    '%H:%M:%S',
                    time.localtime(
                        packet.timestamp))}"
        )
        active_modules = []
        if packet.situation:
            active_modules.append("情境感知")
        if packet.gossip_strategy:
            active_modules.append("事件策略")
        if packet.visibility:
            active_modules.append("可见性分析")
        if packet.safety:
            active_modules.append("安全评估")
        if packet.memory_window:
            active_modules.append("记忆窗口")
        if packet.freshness_state:
            active_modules.append("新鲜度追踪")
        if active_modules:
            lines.append(f"活跃模块: {', '.join(active_modules)}")
        else:
            lines.append("活跃模块: 无（使用默认行为）")
        return "\n".join(lines)

    @staticmethod
    def _build_situation_block(
        sit: Optional[SituationInterpretationInput],
    ) -> str:
        if sit is None:
            return ""
        lines = ["## 主观情境判断（GAP-L）"]
        if sit.situation_reading:
            lines.append(f"情境解读: {sit.situation_reading}")
        if sit.situation_atmosphere:
            lines.append(f"氛围: {sit.situation_atmosphere}")
        if sit.internal_mood:
            lines.append(f"内心情绪: {sit.internal_mood}")
        lines.append(f"精力水平: {sit.internal_energy:.2f}")
        lines.append(f"社交舒适度: {sit.social_comfort:.2f}")
        if sit.engagement_posture:
            lines.append(
                f"参与姿态: {
                    sit.engagement_posture}(强度={
                    sit.engagement_intensity:.2f})"
            )
        if sit.style_hint:
            lines.append(f"风格建议: {sit.style_hint}")
        if sit.risk_assessment_score > 0.15:
            lines.append(f"⚠️ 风险评估: {sit.risk_assessment_score:.2f}")
            if sit.risk_factors:
                lines.append(f"风险因子: {'; '.join(sit.risk_factors[:4])}")
        return "\n".join(lines)

    @staticmethod
    def _build_strategy_block(strat: Optional[GossipStrategyInput]) -> str:
        if strat is None:
            return ""
        lines = ["## 事件策略裁定（GAP-O）"]
        if strat.event_type_label:
            lines.append(f"事件类型: {strat.event_type_label}")
        lines.append(f"参与姿态: {strat.posture_label}")
        act_mark = "✅ 应参与" if strat.should_act else "⏸️ 不参与"
        lines.append(f"行动决策: {act_mark}(优先级={strat.priority:.2f})")
        if strat.reply_hint:
            lines.append(f"回复建议: {strat.reply_hint[:80]}")
        if strat.tone_hint:
            lines.append(f"语气方向: {strat.tone_hint}")
        if strat.narration_hint:
            lines.append(f"内心独白倾向: {strat.narration_hint[:80]}")
        return "\n".join(lines)

    @staticmethod
    def _build_visibility_block(vis: Optional[VisibilitySummaryInput]) -> str:
        if vis is None:
            return ""
        lines = ["## 消息可见性分析（GAP-M）"]
        lines.append(f"总评测量: {vis.total_evaluated}条")
        lines.append(f"高优先级消息: {vis.high_priority_count}条")
        lines.append(f"低分消息: {vis.low_score_count}条")
        lines.append(f"平均调制得分: {vis.avg_modulated_score:.3f}")
        lines.append(f"理解进入率: {vis.understanding_rate:.1%}")
        if vis.top_reasons:
            lines.append(f"主要因素: {'; '.join(vis.top_reasons[:4])}")
        if vis.user_bias_summary:
            bias_parts = [
                f"{k}={v:.2f}"
                for k, v in list(vis.user_bias_summary.items())[:3]
            ]
            if bias_parts:
                lines.append(f"用户偏置: {', '.join(bias_parts)}")
        return "\n".join(lines)

    @staticmethod
    def _build_safety_block(safe: Optional[SafetyAssessmentInput]) -> str:
        if safe is None:
            return ""
        lines = ["## 安全边界评估（GAP-R）"]
        level_icon = {
            "安全": "✅",
            "低风险": "⚡",
            "中风险": "⚠️",
            "高风险": "🛑",
            "严重威胁": "🚨",
        }
        icon = level_icon.get(safe.safety_level, "?")
        lines.append(
            f"安全等级: {icon} {
                safe.safety_level}(score={
                safe.safety_score:.3f})"
        )
        if safe.dominant_threat:
            lines.append(f"主导威胁: {safe.dominant_threat}")
        block_mark = "🔒 已阻断" if safe.should_block else "🔓 放行"
        lines.append(f"阻断状态: {block_mark}")
        if safe.recommended_action:
            lines.append(f"建议操作: {safe.recommended_action[:60]}")
        if safe.bar_delta > 0.01:
            lines.append(f"门槛提升: +{safe.bar_delta:.3f}")
        return "\n".join(lines)

    @staticmethod
    def _build_memory_block(mem: Optional[MemoryWindowInput]) -> str:
        if mem is None:
            return ""
        lines = ["## 动态记忆窗口（GAP-Q）"]
        lines.append(f"窗口大小: {mem.window_size}条 / {mem.total_chars}字符")
        if mem.tier_distribution:
            tier_str = ", ".join(
                f"{k}:{v}" for k, v in mem.tier_distribution.items()
            )
            lines.append(f"层级分布: {tier_str}")
        if mem.cutoff_reason:
            lines.append(f"裁剪原因: {mem.cutoff_reason}")
        if mem.memory_snippets:
            lines.append("记忆片段:")
            for i, snip in enumerate(mem.memory_snippets[:5]):
                lines.append(f"  {i + 1}. {snip[:100]}")
        return "\n".join(lines)

    @staticmethod
    def _build_freshness_block(fresh: Optional[FreshnessStateInput]) -> str:
        if fresh is None:
            return ""
        lines = ["## 心理新鲜度状态（GAP-N）"]
        lines.append(f"跟踪总量: {fresh.total_tracked}")
        lines.append(f"新鲜: {fresh.fresh_count} | 过期: {fresh.stale_count}")
        if fresh.recent_recalls > 0:
            lines.append(f"近期回看激活: {fresh.recent_recalls}次")
        if fresh.decay_tier_distribution:
            tier_str = ", ".join(
                f"{k}:{v}" for k, v in fresh.decay_tier_distribution.items()
            )
            lines.append(f"衰减层级: {tier_str}")
        return "\n".join(lines)

    @staticmethod
    def _build_raw_state_block(
        metabolism: Optional[Dict[str, Any]],
        night_phase: Optional[str],
        relation: Optional[Dict[str, Any]],
        watch: Optional[str],
    ) -> str:
        lines = ["## 原始状态快照"]
        if night_phase:
            lines.append(f"昼夜阶段: {night_phase}")
        if watch:
            lines.append(f"观看状态: {watch}")
        if metabolism:
            energy = float(metabolism.get("energy_ratio", 0) or 0)
            loafing = float(metabolism.get("loafing_level", 0) or 0)
            boredom = float(metabolism.get("boredom_level", 0) or 0)
            lines.append(
                f"能量比={energy:.2f} 摸鱼={loafing:.2f} 无聊={boredom:.2f}"
            )
        if relation:
            aff = _safe_relation_metric(relation, "affection", 50.0)
            trust = _safe_relation_metric(relation, "trust_value", 50.0)
            annoy = _safe_relation_metric(relation, "annoyance_value", 0.0)
            lines.append(f"好感={aff:.0f} 信任={trust:.0f} 厌烦={annoy:.0f}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def get_recent_packet(
        self, channel_id: Optional[str] = None
    ) -> Optional[InjectionPacket]:
        if channel_id:
            for p in reversed(self._recent_packets):
                if p.channel_id == channel_id:
                    return p
            return None
        return self._recent_packets[-1] if self._recent_packets else None


_injection_singleton: Optional[PlannerPromptInjectionLayer] = None


def get_planner_injection_layer() -> PlannerPromptInjectionLayer:
    global _injection_singleton
    if _injection_singleton is None:
        with _injection_lock:
            if _injection_singleton is None:
                _injection_singleton = PlannerPromptInjectionLayer()
    return _injection_singleton

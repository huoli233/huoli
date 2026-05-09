import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger
from src.chat.heart_flow.signal_domains import (
    DecisionSignalBundle,
    SurfaceMaskMode,
    GroupContextTier,
    ResourceLedgerStage,
    TraumaLoadBand,
    CounterpartyEngagementState,
)

logger = get_logger("decision_gateway")


# ============================================================
# 裁定结果数据类
# ============================================================
@dataclass
class GatewayVerdict:
    """多维投票决策网关的裁定结果"""
    # 最终门控：force_reply / allow / hesitate / block
    gate: str = "allow"
    # 是否应跳过回复
    should_skip: bool = False
    # 是否应主动回复
    should_reply: bool = False
    # 最终通过概率（经过投票组合后的值，0.0~1.0）
    final_probability: float = 0.5
    # 随机掷骰结果（final_probability 的掷骰是否命中）
    dice_passed: bool = False
    # 裁定原因（可读文本）
    reason: str = ""
    # 决策来源（用于调试追溯）
    decision_source: str = ""
    # 态度标签集合（注入LLM提示词的情绪/风格指示）
    attitude_tags: List[str] = field(default_factory=list)
    # 风格修饰集合（注入LLM的回复风格指导）
    style_hints: List[str] = field(default_factory=list)
    # token上限（0表示无限制）
    max_tokens: int = 0
    # 回复质量上限（D11面具系统决定，1.0=无限制）
    quality_cap: float = 1.0
    # 内心冲突描述（D11面具破裂时注入）
    inner_conflict: str = ""
    # 创伤潜意识干扰（D10创伤系统注入）
    perception_labels: List[str] = field(default_factory=list)
    # LLM提示词状态片段
    llm_state_prompt: str = ""
    # 调试用完整投票摘要
    vote_debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate,
            "should_skip": self.should_skip,
            "should_reply": self.should_reply,
            "final_probability": round(self.final_probability, 4),
            "dice_passed": self.dice_passed,
            "reason": self.reason,
            "decision_source": self.decision_source,
            "attitude_tags": self.attitude_tags,
            "style_hints": self.style_hints,
            "max_tokens": self.max_tokens,
            "quality_cap": round(self.quality_cap, 2),
        }


# ============================================================
# 外部上下文（@提及、LLM意愿等非维度信号）
# ============================================================
@dataclass
class GatewayContext:
    """决策网关的外部输入上下文（非维度信号）"""
    # 是否直接@bot
    is_direct_ping: bool = False
    # 是否间接提及bot
    has_indirect_mention: bool = False
    # 是否是回复bot的消息
    is_reply_to_bot: bool = False
    # LLM内心独白的回复意愿（0~10）
    desire_level: int = 5
    # LLM内心独白的 should_reply 判断
    voice_should_reply: bool = True
    # 管理员强制回复
    admin_force: bool = False
    # 消息文本（用于提问检测等）
    message_text: str = ""
    # 用户ID
    user_id: str = ""
    # 频道ID
    channel_id: str = ""


# ============================================================
# 决策网关配置
# ============================================================
@dataclass
class GatewayConfig:
    """决策网关可调参数"""
    # 基础概率锚点
    base_probability: float = 0.3
    # @提及时的强制最低概率
    mention_min_probability: float = 0.85
    # 回复bot消息时的最低概率
    reply_min_probability: float = 0.7
    # 提问检测加成
    question_boost: float = 0.15
    # desire等级对概率的缩放
    desire_scale: float = 0.06
    # 连续跳过次数阈值（超过后强制回复）
    consecutive_skip_limit: int = 5
    # 概率地板值（任何情况下不低于此值的概率）
    probability_floor: float = 0.10
    # 概率天花板（任何情况下不超过此值）
    probability_ceiling: float = 0.95


# ============================================================
# 决策网关核心
# ============================================================
class DecisionGateway:
    """
    多维投票决策网关。
    采用分层决策：
      1. 强制触发/强制抑制（最高优先级）
      2. LLM意愿验证
      3. 概率组合（乘法模型）
      4. 掷骰决定
      5. 面具校正（影响回复质量而非概率）
      6. 状态片段注入
    """

    _singleton: Optional["DecisionGateway"] = None

    @classmethod
    def get_instance(cls) -> "DecisionGateway":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        self._config = GatewayConfig()
        self._consecutive_skips: Dict[str, int] = {}
        self._last_reply_ts: Dict[str, float] = {}
        self._recent_verdicts: List[GatewayVerdict] = []

    def configure(self, **kwargs):
        """动态更新配置参数"""
        for k, v in kwargs.items():
            if hasattr(self._config, k):
                setattr(self._config, k, v)

    # ---- 主决策入口 ----

    def decide(
        self,
        votes: DecisionSignalBundle,
        ctx: GatewayContext,
    ) -> GatewayVerdict:
        """
        核心决策流程：
        接收11个维度的投票和外部上下文，输出最终裁定。
        """
        verdict = GatewayVerdict()
        verdict.vote_debug = votes.debug_summary()
        channel_key = ctx.channel_id or "global"
        # ---- 第1层：强制触发检查 ----
        forced = self._check_force_triggers(votes, ctx)
        if forced:
            forced.vote_debug = verdict.vote_debug
            self._post_process(forced, votes, ctx)
            self._record_verdict(forced, channel_key)
            return forced
        # ---- 第2层：强制抑制检查 ----
        suppressed = self._check_force_suppressions(votes, ctx)
        if suppressed:
            suppressed.vote_debug = verdict.vote_debug
            self._post_process(suppressed, votes, ctx)
            self._record_verdict(suppressed, channel_key)
            return suppressed
        # ---- 第3层：LLM意愿验证 ----
        voice_override = self._check_voice_intent(votes, ctx)
        if voice_override:
            voice_override.vote_debug = verdict.vote_debug
            self._post_process(voice_override, votes, ctx)
            self._record_verdict(voice_override, channel_key)
            return voice_override
        # ---- 第4层：维度状态综合评估（不掷骰，由维度投票直接决定门控级别） ----
        prob = self._compute_probability(votes, ctx)
        verdict.final_probability = prob
        # 根据综合概率划分门控级别：
        #   >= 0.5 → allow（维度状态良好，交由LLM内心独白决定是否回复）
        #   >= 0.08 → hesitate（维度状态偏低，允许回复但注入短回复约束）
        #   < 0.08 → hesitate + should_skip（维度状态极差，建议跳过但不硬拦截）
        if prob >= 0.5:
            verdict.gate = "allow"
            verdict.should_reply = True
            verdict.should_skip = False
            verdict.reason = f"维度状态良好(prob={prob:.3f})，交由LLM决策"
            verdict.decision_source = "dimension_allow"
        elif prob >= 0.08:
            verdict.gate = "hesitate"
            verdict.should_reply = False
            verdict.should_skip = False
            verdict.reason = f"维度状态偏低(prob={prob:.3f})，允许回复但约束风格"
            verdict.decision_source = "dimension_hesitate"
        else:
            verdict.gate = "hesitate"
            verdict.should_reply = False
            verdict.should_skip = True
            verdict.reason = f"维度状态极差(prob={prob:.3f})，建议跳过"
            verdict.decision_source = "dimension_low"
        # ---- 第6层：面具校正 + 状态注入 ----
        self._post_process(verdict, votes, ctx)
        self._record_verdict(verdict, channel_key)
        return verdict

    # ---- 第1层：强制触发 ----

    @staticmethod
    def _has_hard_suppression(votes: DecisionSignalBundle) -> bool:
        return bool(
            votes.boundary_block.is_blocked
            or votes.emotion_baseline.should_refuse
            or votes.resource_ledger.force_refuse
            or votes.tempo_control.in_cooldown
            or votes.trauma_load.force_refuse
        )

    def _check_force_triggers(
        self, votes: DecisionSignalBundle, ctx: GatewayContext
    ) -> Optional[GatewayVerdict]:
        # 管理员强制
        if ctx.admin_force:
            return GatewayVerdict(
                gate="force_reply",
                should_reply=True,
                final_probability=1.0,
                dice_passed=True,
                reason="管理员强制回复",
                decision_source="admin_force",
            )
        # @提及
        if ctx.is_direct_ping:
            # 即使被@，如果存在极端负面因素也需要降级
            if votes.boundary_block.is_blocked:
                return GatewayVerdict(
                    gate="block",
                    should_skip=True,
                    reason="被@但用户已被屏蔽",
                    decision_source="blocked_user_ping",
                )
            return GatewayVerdict(
                gate="force_reply",
                should_reply=True,
                final_probability=self._config.mention_min_probability,
                dice_passed=True,
                reason="直接@提及，触发强制回复",
                decision_source="direct_ping",
            )
        # 回复bot消息
        if ctx.is_reply_to_bot:
            if self._has_hard_suppression(votes):
                return None
            return GatewayVerdict(
                gate="allow",
                should_reply=True,
                final_probability=self._config.reply_min_probability,
                dice_passed=True,
                reason="回复bot消息，高概率回复",
                decision_source="reply_to_bot",
            )
        # 间接提及
        if ctx.has_indirect_mention:
            if self._has_hard_suppression(votes):
                return None
            return GatewayVerdict(
                gate="allow",
                should_reply=True,
                final_probability=0.7,
                dice_passed=True,
                reason="间接提及bot",
                decision_source="indirect_mention",
            )
        # 任何维度的force_trigger
        if votes.has_any_force_trigger():
            triggers = [
                v.dimension_name
                for v in votes.all_votes()
                if v.force_trigger
            ]
            return GatewayVerdict(
                gate="force_reply",
                should_reply=True,
                final_probability=0.9,
                dice_passed=True,
                reason=f"维度强制触发: {','.join(triggers)}",
                decision_source="dimension_force_trigger",
            )
        return None

    # ---- 第2层：强制抑制 ----

    def _check_force_suppressions(
        self, votes: DecisionSignalBundle, ctx: GatewayContext
    ) -> Optional[GatewayVerdict]:
        reasons = []
        # D2b 讨厌度屏蔽
        if votes.boundary_block.is_blocked:
            reasons.append(f"边界封锁(severity={votes.boundary_block.total_severity:.1f})")
        # D1 情绪拒绝
        if votes.emotion_baseline.should_refuse:
            reasons.append(
                f"情绪拒绝(patience={votes.emotion_baseline.patience_snapshot:.0f},"
                f"annoyance={votes.emotion_baseline.annoyance_snapshot:.0f})"
            )
        # D6 能量透支
        if votes.resource_ledger.force_refuse:
            reasons.append(f"资源透支(stage={votes.resource_ledger.energy_stage.value})")
        # D3 冷却期
        if votes.tempo_control.in_cooldown:
            reasons.append("节奏冷却中")
        # D10 极端创伤
        if votes.trauma_load.force_refuse:
            reasons.append(f"极端创伤(chaos={votes.trauma_load.inner_chaos:.0f})")
        if not reasons:
            return None
        return GatewayVerdict(
            gate="block",
            should_skip=True,
            reason="强制抑制: " + "；".join(reasons),
            decision_source="force_suppression",
            final_probability=0.0,
        )

    # ---- 第3层：LLM意愿验证 ----

    def _check_voice_intent(
        self, votes: DecisionSignalBundle, ctx: GatewayContext
    ) -> Optional[GatewayVerdict]:
        desire = ctx.desire_level
        should = ctx.voice_should_reply
        # LLM明确不想回复且意愿极低
        if not should and desire <= 2:
            return GatewayVerdict(
                gate="hesitate",
                should_skip=True,
                reason=f"LLM明确不回复(desire={desire})",
                decision_source="voice_decline",
            )
        # LLM强烈想回复
        if should and desire >= 8:
            return GatewayVerdict(
                gate="allow",
                should_reply=True,
                final_probability=0.9,
                dice_passed=True,
                reason=f"LLM强烈想回复(desire={desire})",
                decision_source="voice_strong_promote",
            )
        return None

    # ---- 第4层：概率组合（乘法模型） ----

    def _compute_probability(
        self, votes: DecisionSignalBundle, ctx: GatewayContext
    ) -> float:
        cfg = self._config
        # 基础概率随desire浮动: base + (desire - 5) * scale
        base = cfg.base_probability + (ctx.desire_level - 5) * cfg.desire_scale
        base = max(0.1, min(0.7, base))
        # 乘法组合（排除force_refuse/force_trigger的维度）
        combined = votes.combined_probability_factor()
        prob = base * combined
        # 群氛围加成（加法）
        prob += votes.group_context.engagement_boost
        # 讨厌度扣减（加法）
        prob -= votes.boundary_block.dislike_penalty
        # 好感系统的主动意愿加成（三源融合）
        _ft_will = votes.rapport_trust.proactive_willingness
        _em_will = votes.emotion_baseline.emotion_proactive_willingness
        _hb_will = votes.pending_engagement.heartbeat_proactive_willingness
        _fused_will = max(_ft_will, _em_will * 0.6, _hb_will * 0.4)
        # 讨厌度主动行为抑制
        if votes.boundary_block.proactive_suppression > 0.0:
            _fused_will = max(0.0, _fused_will - votes.boundary_block.proactive_suppression)
        # 社交值主动行为加成
        if votes.social_balance.proactive_boost > 0.0:
            _fused_will += votes.social_balance.proactive_boost * 0.3
        _fused_will = min(1.0, max(0.0, _fused_will))
        if _fused_will > 0.3:
            prob += (_fused_will - 0.3) * 0.25
        # 用户状态加成：对方明确想聊天时提高概率
        if votes.counterparty_readiness.want_to_chat_confidence > 0.7:
            prob += 0.1
        elif votes.counterparty_readiness.want_to_chat_confidence < 0.3:
            prob -= 0.1
        # 提问检测
        text = ctx.message_text
        if text and ("?" in text or "？" in text):
            prob += cfg.question_boost
        # 自适应学习器行为调整
        try:
            from src.core.adaptive_threshold_learner_v2 import get_adaptive_threshold_learner_v2
            _learner = get_adaptive_threshold_learner_v2(ctx.channel_id)
            _silence_mod, _speak_mod, _chatter_pen = _learner.compute_behavior_adjustment(
                time.time()
            )
            prob += _speak_mod
            prob -= _chatter_pen
        except Exception as _exc:
            logger.debug(f"非关键异常: {_exc}")
        # 概率裁剪
        prob = max(cfg.probability_floor, min(cfg.probability_ceiling, prob))
        return prob

    # ---- 第6层：面具校正 + 状态注入 ----

    def _post_process(
        self,
        verdict: GatewayVerdict,
        votes: DecisionSignalBundle,
        ctx: GatewayContext,
    ):
        """面具校正、态度/风格收集、LLM状态片段生成"""
        # 态度标签
        verdict.attitude_tags = votes.collect_attitude_tags()
        # 风格修饰
        verdict.style_hints = votes.collect_style_hints()
        # token上限（取所有维度中最严格的）
        verdict.max_tokens = votes.min_max_tokens_cap()
        # 面具系统校正
        mask = votes.surface_mask
        verdict.quality_cap = mask.reply_quality_cap
        verdict.inner_conflict = mask.inner_conflict_hint
        if mask.behavior_mode == SurfaceMaskMode.BROKEN:
            verdict.style_hints.append("情绪失控、真实想法涌出")
            if mask.mask_strength < 20:
                verdict.attitude_tags.append("mask_collapsed")
        elif mask.behavior_mode == SurfaceMaskMode.CRACKING:
            verdict.style_hints.append("偶尔流露不一致")
        # 创伤潜意识干扰
        verdict.perception_labels = list(votes.trauma_load.perception_labels)
        # 生成LLM状态片段
        verdict.llm_state_prompt = self._build_llm_state_prompt(votes, ctx)

    def _build_llm_state_prompt(
        self, votes: DecisionSignalBundle, ctx: GatewayContext
    ) -> str:
        """将所有维度的状态汇总为LLM可读的提示词片段"""
        parts = []
        # D1 情绪
        if votes.emotion_baseline.mood_modifier:
            parts.append(f"情绪: {votes.emotion_baseline.mood_modifier}")
        # D2 好感/信任
        ft = votes.rapport_trust
        if ft.fondness_level.value != "neutral" or ft.trust_level.value != "ordinary":
            parts.append(
                f"好感: {ft.fondness_level.value}(值={ft.fondness_value}) "
                f"信任: {ft.trust_level.value}(值={ft.trust_value}) "
                f"关系: {ft.cross_hint}"
            )
        # D5 群氛围
        atm = votes.group_context
        if atm.atmosphere_tier != GroupContextTier.NORMAL:
            parts.append(
                f"群氛围: {atm.atmosphere_tier.value} "
                f"定位: {atm.ai_position.value}"
            )
        if atm.style_constraint:
            parts.append(f"氛围风格: {atm.style_constraint}")
        # D6 能量
        if votes.resource_ledger.energy_stage != ResourceLedgerStage.FULL:
            parts.append(f"精力: {votes.resource_ledger.energy_stage.value}")
        # D7 社交
        if votes.social_balance.social_tag and votes.social_balance.social_tag != "social_neutral":
            parts.append(f"社交印象: {votes.social_balance.social_tag}")
        # D4 用户状态
        if votes.counterparty_readiness.engagement != CounterpartyEngagementState.NEUTRAL:
            parts.append(f"对方状态: {votes.counterparty_readiness.engagement.value}")
        # D10 创伤
        if votes.trauma_load.trauma_grade != TraumaLoadBand.NONE:
            parts.append(
                f"创伤: {votes.trauma_load.trauma_grade.value} "
                f"混乱度={votes.trauma_load.inner_chaos:.0f}"
            )
            if votes.trauma_load.perception_labels:
                parts.append(f"潜意识: {'; '.join(votes.trauma_load.perception_labels[:3])}")
        # D11 面具
        if votes.surface_mask.behavior_mode != SurfaceMaskMode.NATURAL:
            parts.append(
                f"伪装: {votes.surface_mask.behavior_mode.value} "
                f"强度={votes.surface_mask.mask_strength:.0f}"
            )
            if votes.surface_mask.inner_conflict_hint:
                parts.append(f"内心: {votes.surface_mask.inner_conflict_hint}")
        if not parts:
            return ""
        return "你当前的状态:\n" + "\n".join(f"- {p}" for p in parts)

    # ---- 记录与查询 ----

    def _record_verdict(self, verdict: GatewayVerdict, channel_key: str):
        """记录裁定结果, 更新跳过计数和回复时间"""
        if not verdict.should_skip:
            self._consecutive_skips[channel_key] = 0
            self._last_reply_ts[channel_key] = time.time()
        self._recent_verdicts.append(verdict)
        if len(self._recent_verdicts) > 100:
            self._recent_verdicts = self._recent_verdicts[-50:]
        logger.info(
            f"[决策网关] gate={verdict.gate} prob={verdict.final_probability:.3f} "
            f"skip={verdict.should_skip} src={verdict.decision_source} "
            f"reason={verdict.reason[:80]}"
        )

    def recent_statistics(self, last_n: int = 20) -> Dict[str, Any]:
        """最近N次裁定的统计"""
        recent = self._recent_verdicts[-last_n:]
        if not recent:
            return {"count": 0}
        total = len(recent)
        replied = sum(1 for v in recent if not v.should_skip)
        gates = {}
        for v in recent:
            gates[v.gate] = gates.get(v.gate, 0) + 1
        return {
            "count": total,
            "replied": replied,
            "skipped": total - replied,
            "reply_rate": round(replied / total, 3),
            "gates": gates,
        }


# ============================================================
# 兼容层：为旧代码提供与 ReadinessVerdict 兼容的接口
# ============================================================
@dataclass
class CompatReadinessVerdict:
    """兼容旧 ReadinessVerdict 的结构"""
    should_skip: bool = False
    reason: str = ""
    total_deduction: float = 0.0
    reply_readiness: float = 50.0
    gate: str = "allow"
    emotion_override: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_skip": self.should_skip,
            "reason": self.reason,
            "reply_readiness": round(self.reply_readiness, 2),
            "gate": self.gate,
        }


def gateway_verdict_to_compat(gv: GatewayVerdict) -> CompatReadinessVerdict:
    """将新决策网关的裁定结果转换为兼容旧格式"""
    readiness = gv.final_probability * 100.0
    deduction = max(0.0, 50.0 - readiness)
    emotion = ""
    if "trauma_shutdown" in gv.attitude_tags:
        emotion = "trauma_shutdown"
    elif "mask_collapsed" in gv.attitude_tags:
        emotion = "mask_collapsed"
    elif "hostile_avoidance" in gv.attitude_tags:
        emotion = "extreme_rejection"
    return CompatReadinessVerdict(
        should_skip=gv.should_skip,
        reason=gv.reason,
        total_deduction=round(deduction, 2),
        reply_readiness=round(readiness, 2),
        gate=gv.gate,
        emotion_override=emotion,
    )


def get_decision_gateway() -> DecisionGateway:
    """获取决策网关单例"""
    return DecisionGateway.get_instance()

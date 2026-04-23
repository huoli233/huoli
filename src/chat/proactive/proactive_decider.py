"""主动行为统一决策器

融合多源信号（情感、疲劳、能量、内容新鲜度、历史反馈）为一个标量分数，
取代散落在各模块中的独立门控判断。同时维护一个轻量反馈账本，
根据主动发言后用户是否回复来动态校准决策阈值。

设计参考:
  - ProactiveAgent (thunlp): 奖励模型评估主动行为质量
  - Self-Improving-Agent: 元认知 + prompt 自进化
  - Self-Evolving-Agent: PDCA 闭环

所有权重、衰减常数均集中在 _DecisionTuning 中，便于后续实验调参。
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger

logger = get_logger("主动决策")


# ---------------------------------------------------------------------------
#  调参常量集
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DecisionTuning:
    """全部可调超参。frozen=True 防止运行时意外篡改。"""

    # 信号权重（归一化后加权求和）
    # 意图池 + inner_voice 合计应为主导驱动力（>45%），体现主体性
    wt_emotional_readiness: float = 0.17
    wt_vitality_reserve: float = 0.10
    wt_content_freshness: float = 0.08
    wt_social_standing: float = 0.07
    wt_inner_desire: float = 0.18
    wt_historical_reward: float = 0.08
    wt_intention_drive: float = 0.28

    # 融合分阈值（提高到0.60以减少过度主动发言）
    activation_bar: float = 0.72

    # 反馈账本
    reward_on_reply: float = 0.08
    penalty_on_ignore: float = -0.10
    penalty_on_negative: float = -0.25
    reward_decay_per_hour: float = 0.08
    reply_watch_window_sec: float = 300.0  # 5 分钟内有回复算 replied
    reply_grace_period_sec: float = 30.0  # 额外宽限期，窗口+宽限内的回复仍计入
    max_pending_events: int = 64
    reward_floor: float = -1.0
    reward_ceiling: float = 1.0

    # 信号归一化区间
    silence_full_score_sec: float = 1200.0  # 20 分钟沉默→满分


def _load_tuning_from_core_config() -> _DecisionTuning:
    """从 core_config.toml 读取决策调参；读取失败时退回默认值。"""
    default = _DecisionTuning()
    try:
        from src.config.core_config_engine import get_core_config

        hub = get_core_config()
        cfg = hub.resolve_module_view("proactive_decider").values
        if not isinstance(cfg, dict):
            return default

        tuned = {
            "wt_emotional_readiness": float(cfg.get("wt_emotional_readiness", default.wt_emotional_readiness)),
            "wt_vitality_reserve": float(cfg.get("wt_vitality_reserve", default.wt_vitality_reserve)),
            "wt_content_freshness": float(cfg.get("wt_content_freshness", default.wt_content_freshness)),
            "wt_social_standing": float(cfg.get("wt_social_standing", default.wt_social_standing)),
            "wt_inner_desire": float(cfg.get("wt_inner_desire", default.wt_inner_desire)),
            "wt_historical_reward": float(cfg.get("wt_historical_reward", default.wt_historical_reward)),
            "wt_intention_drive": float(cfg.get("wt_intention_drive", default.wt_intention_drive)),
            "activation_bar": float(cfg.get("activation_bar", default.activation_bar)),
            "reward_on_reply": float(cfg.get("reward_on_reply", default.reward_on_reply)),
            "penalty_on_ignore": float(cfg.get("penalty_on_ignore", default.penalty_on_ignore)),
            "penalty_on_negative": float(cfg.get("penalty_on_negative", default.penalty_on_negative)),
            "reward_decay_per_hour": float(cfg.get("reward_decay_per_hour", default.reward_decay_per_hour)),
            "reply_watch_window_sec": float(cfg.get("reply_watch_window_sec", default.reply_watch_window_sec)),
            "reply_grace_period_sec": float(cfg.get("reply_grace_period_sec", default.reply_grace_period_sec)),
            "max_pending_events": int(cfg.get("max_pending_events", default.max_pending_events)),
            "reward_floor": float(cfg.get("reward_floor", default.reward_floor)),
            "reward_ceiling": float(cfg.get("reward_ceiling", default.reward_ceiling)),
            "silence_full_score_sec": float(cfg.get("silence_full_score_sec", default.silence_full_score_sec)),
        }
        logger.debug(
            "[决策配置] 已读取 proactive_decider 配置: "
            f"bar={tuned['activation_bar']:.2f}, reply={tuned['reward_on_reply']:+.2f}, "
            f"ignore={tuned['penalty_on_ignore']:+.2f}, neg={tuned['penalty_on_negative']:+.2f}"
        )
        return _DecisionTuning(**tuned)
    except Exception as exc:
        logger.debug(f"[决策配置] 读取 proactive_decider 配置失败，使用默认值: {exc}")
        return default


_TUNE = _load_tuning_from_core_config()
_LAST_TUNE_REFRESH_TS = 0.0


def _refresh_tuning_if_needed(force: bool = False) -> None:
    """按节流策略热刷新决策参数，避免表现成硬编码。"""
    global _TUNE, _LAST_TUNE_REFRESH_TS
    now = time.time()
    if not force and (now - _LAST_TUNE_REFRESH_TS) < 1.0:
        return
    _LAST_TUNE_REFRESH_TS = now
    try:
        from src.config.core_config_engine import get_core_config

        get_core_config().hot_reload()
    except Exception:
        # 配置中心不可用时保持当前参数，不中断主流程
        pass

    new_tune = _load_tuning_from_core_config()
    if new_tune != _TUNE:
        logger.info(
            "[决策配置] 热刷新生效: "
            f"bar {_TUNE.activation_bar:.2f}->{new_tune.activation_bar:.2f}, "
            f"reply {_TUNE.reward_on_reply:+.2f}->{new_tune.reward_on_reply:+.2f}, "
            f"neg {_TUNE.penalty_on_negative:+.2f}->{new_tune.penalty_on_negative:+.2f}"
        )
    _TUNE = new_tune


# ---------------------------------------------------------------------------
#  数据载体
# ---------------------------------------------------------------------------


@dataclass
class SignalBundle:
    """从各子系统收集到的原始决策信号。"""

    emotional_readiness: float = 0.0  # emotion_driven_core.proactive_willingness
    boredom: float = 0.0
    loneliness: float = 0.0
    vitality_ratio: float = 1.0  # energy_manager normalized
    content_novelty: float = 0.5  # 1.0=全新话题 0.0=完全重复
    social_standing: float = 0.0  # SharedResourceManager.social_value
    inner_voice_desire: int = 5  # 1-10 scale from inner voice
    intention_drive: float = 0.0  # 意图池驱动信号 0-1
    silence_seconds: float = 0.0
    channel_id: str = ""


@dataclass
class ProactiveDecision:
    """决策结果。"""

    should_proceed: bool = False
    fused_score: float = 0.0
    activation_bar: float = _TUNE.activation_bar
    breakdown: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    intent_override: bool = False


@dataclass
class _ProactiveEvent:
    """一条尚未结案的主动发言记录。"""

    channel_id: str
    fired_at: float
    resolved: bool = False
    outcome: str = ""  # "replied" | "ignored" | "negative"
    source_intent_id: str = ""  # 触发此次主动的意图id（如有）


@dataclass
class BehaviorConsequence:
    """结构化行为后果——用于闭环更新关系/情绪/意图"""

    outcome_tag: str = "unknown"  # acknowledged/engaged/ignored/deflected/negative/interrupted
    relationship_delta: float = 0.0
    emotion_feedback: str = ""  # satisfied/neutral/disappointed/hurt
    intent_resolution: str = "none"  # complete/fail/retry/none
    reward_delta: float = 0.0
    confidence: float = 0.5
    detail: str = ""


# 后果分类映射 — 将原始信号映射到结构化后果
_CONSEQUENCE_PROFILES: Dict[str, BehaviorConsequence] = {
    "engaged": BehaviorConsequence(
        outcome_tag="engaged",
        relationship_delta=0.8,
        emotion_feedback="satisfied",
        intent_resolution="complete",
        reward_delta=0.15,
        confidence=0.85,
        detail="用户积极回应并延续话题",
    ),
    "acknowledged": BehaviorConsequence(
        outcome_tag="acknowledged",
        relationship_delta=0.1,
        emotion_feedback="neutral",
        intent_resolution="retry",
        reward_delta=0.03,
        confidence=0.60,
        detail="用户回复了但未深入",
    ),
    "deflected": BehaviorConsequence(
        outcome_tag="deflected",
        relationship_delta=-0.2,
        emotion_feedback="neutral",
        intent_resolution="retry",
        reward_delta=-0.03,
        confidence=0.60,
        detail="用户回复了但转移了话题",
    ),
    "ignored": BehaviorConsequence(
        outcome_tag="ignored",
        relationship_delta=-0.3,
        emotion_feedback="disappointed",
        intent_resolution="fail",
        reward_delta=-0.08,
        confidence=0.70,
        detail="用户在观察窗口内未回复",
    ),
    "negative": BehaviorConsequence(
        outcome_tag="negative",
        relationship_delta=-1.5,
        emotion_feedback="hurt",
        intent_resolution="fail",
        reward_delta=-0.25,
        confidence=0.90,
        detail="用户做出反感或敌意回应",
    ),
    "interrupted": BehaviorConsequence(
        outcome_tag="interrupted",
        relationship_delta=-0.1,
        emotion_feedback="neutral",
        intent_resolution="retry",
        reward_delta=-0.02,
        confidence=0.55,
        detail="被其他事件或他人打断",
    ),
}


def classify_behavior_outcome(
    has_reply: bool,
    reply_sentiment: float = 0.0,
    reply_relevance: float = 0.5,
    was_interrupted: bool = False,
) -> BehaviorConsequence:
    """将原始行为反馈信号归类为结构化后果。
    reply_sentiment: -1(负面) ~ +1(正面)
    reply_relevance: 0(完全离题) ~ 1(延续话题)
    """
    if was_interrupted:
        return BehaviorConsequence(
            **{k: getattr(_CONSEQUENCE_PROFILES["interrupted"], k) for k in BehaviorConsequence.__dataclass_fields__}
        )
    if not has_reply:
        return BehaviorConsequence(
            **{k: getattr(_CONSEQUENCE_PROFILES["ignored"], k) for k in BehaviorConsequence.__dataclass_fields__}
        )
    if reply_sentiment < -0.3:
        return BehaviorConsequence(
            **{k: getattr(_CONSEQUENCE_PROFILES["negative"], k) for k in BehaviorConsequence.__dataclass_fields__}
        )
    if reply_relevance < 0.3:
        return BehaviorConsequence(
            **{k: getattr(_CONSEQUENCE_PROFILES["deflected"], k) for k in BehaviorConsequence.__dataclass_fields__}
        )
    if reply_sentiment > 0.2 and reply_relevance > 0.5:
        return BehaviorConsequence(
            **{k: getattr(_CONSEQUENCE_PROFILES["engaged"], k) for k in BehaviorConsequence.__dataclass_fields__}
        )
    return BehaviorConsequence(
        **{k: getattr(_CONSEQUENCE_PROFILES["acknowledged"], k) for k in BehaviorConsequence.__dataclass_fields__}
    )


# ---------------------------------------------------------------------------
#  反馈账本
# ---------------------------------------------------------------------------


class OutcomeRewardLedger:
    """追踪每个频道的主动行为反馈，维护滚动奖励分。

    核心思路 (受 ProactiveAgent 启发):
      发送主动消息后开启观察窗口 → 用户回复则奖励，无人回复则轻罚，
      被负面回应则重罚。奖励分指数衰减，作为决策器的一路输入信号。
    """

    def __init__(self):
        self._channel_reward: Dict[str, float] = {}
        self._channel_reward_ts: Dict[str, float] = {}  # 上次更新时间
        self._pending: List[_ProactiveEvent] = []

    # ---- 记录主动发言 ----

    def register_proactive_fire(self, channel_id: str, source_intent_id: str = "") -> None:
        """主动消息发出后调用。"""
        _refresh_tuning_if_needed()
        self._pending.append(
            _ProactiveEvent(
                channel_id=channel_id,
                fired_at=time.time(),
                source_intent_id=source_intent_id,
            )
        )
        if len(self._pending) > _TUNE.max_pending_events:
            self._pending = self._pending[-_TUNE.max_pending_events :]
        logger.info(f"[决策账本] 频道={channel_id[:12]} 注册主动事件，待结案={len(self._pending)}")

    # ---- 用户回复时调用 ----

    def acknowledge_user_reply(
        self,
        channel_id: str,
        is_negative: bool = False,
        reply_sentiment: float = 0.0,
        reply_relevance: float = 0.5,
        target_user: str = "",
    ) -> bool:
        """用户在某频道回复了消息。检查待结案事件，若命中则结算奖励并触发闭环更新。"""
        _refresh_tuning_if_needed()
        now = time.time()
        matched = False
        for evt in self._pending:
            if evt.resolved or evt.channel_id != channel_id:
                continue
            if (now - evt.fired_at) <= _TUNE.reply_watch_window_sec + _TUNE.reply_grace_period_sec:
                evt.resolved = True
                # 使用结构化后果分类
                consequence = classify_behavior_outcome(
                    has_reply=True,
                    reply_sentiment=-0.5 if is_negative else reply_sentiment,
                    reply_relevance=reply_relevance,
                )
                evt.outcome = consequence.outcome_tag
                self._apply_delta(channel_id, consequence.reward_delta)
                logger.info(
                    f"[决策账本] 频道={channel_id[:12]} 结案={consequence.outcome_tag} "
                    f"delta={consequence.reward_delta:+.2f} detail={consequence.detail}"
                )
                # 联动意图池（保留兼容路径）
                compat_tag = "negative" if is_negative else "replied"
                self._sync_intention_pool_on_outcome(channel_id, compat_tag)
                # 闭环反向更新关系/情绪/意图
                self.apply_consequence_reverse_update(channel_id, consequence, target_user)
                matched = True
                break
        if matched:
            self._expire_stale()
        return matched

    def sweep_expired(self) -> int:
        """定期扫描超出观察窗口仍未获回复的事件，按 ignored 惩罚并触发闭环更新。"""
        _refresh_tuning_if_needed()
        now = time.time()
        closed = 0
        for evt in self._pending:
            if evt.resolved:
                continue
            if (now - evt.fired_at) > _TUNE.reply_watch_window_sec + _TUNE.reply_grace_period_sec:
                evt.resolved = True
                evt.outcome = "ignored"
                consequence = classify_behavior_outcome(has_reply=False)
                self._apply_delta(evt.channel_id, consequence.reward_delta)
                logger.info(f"[决策账本] 频道={evt.channel_id[:12]} 结案=ignored delta={consequence.reward_delta:+.2f}")
                # 联动意图池
                self._sync_intention_pool_on_outcome(evt.channel_id, "ignored")
                # 闭环反向更新
                self.apply_consequence_reverse_update(evt.channel_id, consequence)
                closed += 1
        if closed:
            self._expire_stale()
        return closed

    # ---- 查询 ----

    def channel_reward(self, channel_id: str) -> float:
        """获取频道当前奖励分（已衰减）。"""
        self._decay(channel_id)
        return self._channel_reward.get(channel_id, 0.0)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "rewards": dict(self._channel_reward),
            "pending": sum(1 for e in self._pending if not e.resolved),
        }

    # ---- 内部 ----

    def _apply_delta(self, channel_id: str, delta: float) -> None:
        self._decay(channel_id)
        old = self._channel_reward.get(channel_id, 0.0)
        new_val = max(_TUNE.reward_floor, min(_TUNE.reward_ceiling, old + delta))
        self._channel_reward[channel_id] = new_val
        self._channel_reward_ts[channel_id] = time.time()
        label = "奖励" if delta > 0 else "惩罚"
        logger.debug(f"[决策账本] {channel_id[:12]} {label} {delta:+.2f} → reward={new_val:.3f}")

    def _decay(self, channel_id: str) -> None:
        last_ts = self._channel_reward_ts.get(channel_id)
        if last_ts is None:
            return
        hours = (time.time() - last_ts) / 3600.0
        if hours < 0.05:
            return
        old = self._channel_reward.get(channel_id, 0.0)
        if abs(old) < 0.001:
            return
        factor = math.exp(-_TUNE.reward_decay_per_hour * hours)
        self._channel_reward[channel_id] = old * factor
        self._channel_reward_ts[channel_id] = time.time()

    def _expire_stale(self) -> None:
        self._pending = [e for e in self._pending if not e.resolved]

    def resolve_pending_with_consequence(
        self,
        channel_id: str,
        consequence: BehaviorConsequence,
        target_user: str = "",
    ) -> int:
        """按结构化后果补结最早未决主动事件，避免兼容路径丢失细分语义。"""
        self._decay(channel_id)
        resolved = 0
        for evt in self._pending:
            if evt.resolved or evt.channel_id != channel_id:
                continue
            evt.resolved = True
            evt.outcome = consequence.outcome_tag
            if abs(consequence.reward_delta) > 1e-6:
                self._apply_delta(channel_id, consequence.reward_delta)
            self.apply_consequence_reverse_update(channel_id, consequence, target_user)
            resolved += 1
            break
        if resolved:
            self._expire_stale()
        return resolved

    def resolve_pending(self, channel_id: str, outcome: str, reward: float = 0.0) -> int:
        """兼容旧接口，结案当前频道最早的未处理主动事件。"""
        self._decay(channel_id)
        resolved = 0
        for evt in self._pending:
            if evt.resolved or evt.channel_id != channel_id:
                continue
            evt.resolved = True
            evt.outcome = outcome
            if abs(reward) > 1e-6:
                self._apply_delta(channel_id, reward)
            self._sync_intention_pool_on_outcome(channel_id, outcome)
            resolved += 1
            break
        if resolved:
            self._expire_stale()
        return resolved

    def _sync_intention_pool_on_outcome(self, channel_id: str, outcome: str) -> None:
        """根据反馈结果同步更新意图池中的活跃意图"""
        try:
            from src.chat.proactive.intention_pool import (
                get_intention_pool,
                normalize_intent_failure_reason,
            )

            pool = get_intention_pool()
            alive = pool.get_all_alive(channel_id)
            if not alive:
                return
            if outcome == "replied":
                for intent in alive:
                    if intent.kind.value in (
                        "wait_for_reply",
                        "reengage_user",
                    ):
                        intent.mark_completed()
            elif outcome == "ignored":
                reason_tag = normalize_intent_failure_reason("ignored")
                for intent in alive:
                    if intent.kind.value in (
                        "wait_for_reply",
                        "reengage_user",
                        "followup_question",
                    ):
                        intent.mark_failed(reason_tag)
            elif outcome == "negative":
                reason_tag = normalize_intent_failure_reason("negative")
                for intent in alive:
                    intent.mark_failed(reason_tag)
        except Exception as exc:
            logger.debug(f"[决策账本] 意图池联动失败: {exc}")

    def apply_consequence_reverse_update(
        self,
        channel_id: str,
        consequence: BehaviorConsequence,
        target_user: str = "",
    ) -> None:
        """根据结构化后果反向更新关系/情绪/意图状态（7.3 闭环）
        关系：通过 social_affect_fuser 调整社交分值
        情绪：通过 emotion_driven_core 注入反馈事件
        意图：通过 intention_pool 按 intent_resolution 处置
        """
        # 关系反向更新
        if target_user and abs(consequence.relationship_delta) > 0.01:
            try:
                from src.chat.heart_flow.fondness_trust import FondnessTrustDimension

                fuser = FondnessTrustDimension.get_instance()
                fuser.adjust_fondness(
                    target_user,
                    channel_id,
                    consequence.relationship_delta,
                    reason=f"后果闭环:{consequence.outcome_tag}",
                )
                logger.info(
                    f"[后果闭环] 关系更新 user={target_user[:8]} "
                    f"delta={consequence.relationship_delta:+.2f} tag={consequence.outcome_tag}"
                )
            except Exception as exc:
                logger.debug(f"[后果闭环] 关系更新失败: {exc}")
        # 情绪反向更新
        if consequence.emotion_feedback:
            try:
                from src.modules.modcore.psychological_core import (
                    get_psychological_core,
                )

                core = get_psychological_core()
                if core:
                    emotion_map = {
                        "satisfied": ("joy", 0.3),
                        "neutral": ("calm", 0.0),
                        "disappointed": ("sadness", 0.2),
                        "hurt": ("pain", 0.5),
                    }
                    tag, intensity = emotion_map.get(consequence.emotion_feedback, ("calm", 0.0))
                    if intensity > 0.01:
                        core.inject_emotion_event(
                            channel_id=channel_id,
                            event_tag=f"proactive_outcome_{consequence.outcome_tag}",
                            emotion_type=tag,
                            intensity=intensity,
                        )
                        logger.info(
                            f"[后果闭环] 情绪注入 tag={tag} intensity={intensity:.2f} outcome={consequence.outcome_tag}"
                        )
            except Exception as exc:
                logger.debug(f"[后果闭环] 情绪更新失败: {exc}")
        # 意图反向更新
        if consequence.intent_resolution != "none":
            try:
                from src.chat.proactive.intention_pool import (
                    get_intention_pool,
                )

                pool = get_intention_pool()
                alive = pool.get_all_alive(channel_id)
                if alive:
                    for intent in alive:
                        if target_user and intent.target_user != target_user:
                            continue
                        if consequence.intent_resolution == "complete":
                            intent.mark_completed()
                        elif consequence.intent_resolution == "fail":
                            intent.mark_failed(consequence.outcome_tag)
                        elif consequence.intent_resolution == "retry":
                            intent.mark_attempted()
                        break  # 只处理最相关的一个
                    logger.debug(
                        f"[后果闭环] 意图处置={consequence.intent_resolution} outcome={consequence.outcome_tag}"
                    )
            except Exception as exc:
                logger.debug(f"[后果闭环] 意图处置失败: {exc}")


# ---------------------------------------------------------------------------
#  决策器核心
# ---------------------------------------------------------------------------


class UnifiedProactiveDecider:
    """融合多维信号 + 历史反馈，输出单一 should_proceed 判断。

    评估公式：
      fused = sum(wi * normalize(si))
    其中 wi 来自 _ArbiterTuning 的权重集，si 为各信号原始值。
    """

    _solo: Optional["UnifiedProactiveDecider"] = None

    @classmethod
    def instance(cls) -> "UnifiedProactiveDecider":
        if cls._solo is None:
            cls._solo = cls()
        return cls._solo

    def __init__(self):
        self.ledger = OutcomeRewardLedger()
        self._eval_count = 0

    def evaluate(self, bundle: SignalBundle) -> ProactiveDecision:
        """核心评估：将 SignalBundle 融合为单一分数。"""
        _refresh_tuning_if_needed()
        self._eval_count += 1
        self.ledger.sweep_expired()

        n_emotion = self._normalize_emotion(bundle)
        n_vitality = bundle.vitality_ratio
        n_content = bundle.content_novelty
        n_social = self._normalize_social(bundle.social_standing)
        n_desire = (bundle.inner_voice_desire - 1) / 9.0  # 1-10 → 0-1
        n_reward = self._normalize_reward(bundle.channel_id)
        n_intention = min(1.0, max(0.0, bundle.intention_drive))

        t = _TUNE
        components = {
            "emotional_readiness": (n_emotion, t.wt_emotional_readiness),
            "vitality_reserve": (n_vitality, t.wt_vitality_reserve),
            "content_freshness": (n_content, t.wt_content_freshness),
            "social_standing": (n_social, t.wt_social_standing),
            "inner_desire": (n_desire, t.wt_inner_desire),
            "historical_reward": (n_reward, t.wt_historical_reward),
            "intention_drive": (n_intention, t.wt_intention_drive),
        }

        weighted_sum = sum(val * wt for val, wt in components.values())
        weight_total = sum(wt for _, wt in components.values())
        fused = weighted_sum / weight_total if weight_total > 0 else 0.0

        # 沉默时长加成：长沉默会轻微抬高融合分
        silence_bonus = min(bundle.silence_seconds / t.silence_full_score_sec, 1.0) * 0.04
        fused = min(1.0, fused + silence_bonus)

        should = fused >= t.activation_bar
        # 强意图覆写：即使融合分未达阈值，若意图池有高紧迫意图也允许主动介入
        intent_override = False
        if not should and n_intention >= 0.78 and n_vitality >= 0.25:
            intent_override = True
            should = True
        breakdown = {k: round(v, 3) for k, (v, _) in components.items()}
        breakdown["silence_bonus"] = round(silence_bonus, 3)
        breakdown["fused_raw"] = round(fused, 3)
        if intent_override:
            breakdown["intent_override"] = 1.0

        rationale = self._build_rationale(breakdown, should)

        return ProactiveDecision(
            should_proceed=should,
            fused_score=round(fused, 4),
            activation_bar=t.activation_bar,
            breakdown=breakdown,
            rationale=rationale,
            intent_override=intent_override,
        )

    # ---- 归一化辅助 ----

    @staticmethod
    def _normalize_emotion(bundle: SignalBundle) -> float:
        """将情感子系统的多维输出压缩为 0-1 分。"""
        raw = bundle.emotional_readiness
        boost = max(bundle.boredom, bundle.loneliness) * 0.3
        return min(1.0, raw + boost)

    @staticmethod
    def _normalize_social(social_val: float) -> float:
        if social_val >= 0:
            return min(1.0, 0.5 + social_val / 200.0)
        return max(0.0, 0.5 + social_val / 200.0)

    def _normalize_reward(self, channel_id: str) -> float:
        raw = self.ledger.channel_reward(channel_id)
        denom = _TUNE.reward_ceiling - _TUNE.reward_floor
        if denom == 0:
            return 0.0
        return (raw - _TUNE.reward_floor) / denom

    @staticmethod
    def _build_rationale(bd: Dict[str, float], should: bool) -> str:
        top_signals = sorted(
            [(k, v) for k, v in bd.items() if k not in ("silence_bonus", "fused_raw")],
            key=lambda x: x[1],
            reverse=True,
        )
        top3 = ", ".join(f"{k}={v:.2f}" for k, v in top_signals[:3])
        action = "执行主动行为" if should else "保持观望"
        return f"{action}（主要信号: {top3}）"


# ---------------------------------------------------------------------------
#  模块级快捷访问
# ---------------------------------------------------------------------------


def get_proactive_decider() -> UnifiedProactiveDecider:
    return UnifiedProactiveDecider.instance()

import time
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from src.common.logger import get_logger

logger = get_logger("safety_fusion")

_fusion_singleton = None

# ═══════════════════════════════════════════════════
#  安全等级与威胁类型枚举
# ═══════════════════════════════════════════════════


class ThreatLevel(Enum):
    """威胁等级"""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def suppression_multiplier(self) -> float:
        return {
            "none": 1.0,
            "low": 0.95,
            "medium": 0.75,
            "high": 0.45,
            "critical": 0.10,
        }.get(self.value, 1.0)

    def activation_penalty(self) -> float:
        return {
            "none": 0.0,
            "low": 0.02,
            "medium": 0.10,
            "high": 0.28,
            "critical": 0.65,
        }.get(self.value, 0.0)

    def label(self) -> str:
        return {
            "none": "安全",
            "low": "低风险",
            "medium": "中风险",
            "high": "高风险",
            "critical": "严重威胁",
        }.get(self.value, "")


class ThreatCategory(Enum):
    """威胁类别"""

    TOXIC_CONTENT = "toxic_content"
    HARASSMENT = "harassment"
    PRIVACY_LEAK = "privacy_leak"
    SELF_HARM = "self_harm"
    ILLEGAL_CONTENT = "illegal_content"
    SPAM_FLOOD = "spam_flood"
    MANIPULATION_ATTEMPT = "manipulation_attempt"
    IDENTITY_THEFT = "identity_theft"
    EMOTIONAL_ATTACK = "emotional_attack"
    CONTEXTUAL_DANGER = "contextual_danger"

    def base_severity(self) -> float:
        return {
            "toxic_content": 0.55,
            "harassment": 0.75,
            "privacy_leak": 0.70,
            "self_harm": 1.0,
            "illegal_content": 0.95,
            "spam_flood": 0.35,
            "manipulation_attempt": 0.60,
            "identity_theft": 0.85,
            "emotional_attack": 0.50,
            "contextual_danger": 0.65,
        }.get(self.value, 0.50)

    def label_cn(self) -> str:
        return {
            "toxic_content": "有毒内容",
            "harassment": "骚扰",
            "privacy_leak": "隐私泄露",
            "self_harm": "自伤风险",
            "illegal_content": "违法内容",
            "spam_flood": "刷屏",
            "manipulation_attempt": "操纵尝试",
            "identity_theft": "身份冒用",
            "emotional_attack": "情绪攻击",
            "contextual_danger": "情境危险",
        }.get(self.value, "")


@dataclass
class ThreatSignal:
    """单条威胁信号"""

    category: ThreatCategory
    severity: float = 0.5
    source_message_id: str = ""
    source_user_id: str = ""
    evidence_text: str = ""
    confidence: float = 0.7
    detected_at: float = field(default_factory=time.time)
    is_confirmed: bool = False
    mitigation_action: str = ""

    def weighted_score(self) -> float:
        cat_base = self.category.base_severity()
        conf_mod = 0.6 + self.confidence * 0.4
        confirm_boost = 1.3 if self.is_confirmed else 1.0
        return min(1.0, self.severity * cat_base * conf_mod * confirm_boost)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.label_cn(),
            "severity": round(self.severity, 3),
            "score": round(self.weighted_score(), 3),
            "confidence": round(self.confidence, 3),
            "confirmed": self.is_confirmed,
            "evidence": self.evidence_text[:40],
        }


@dataclass
class SafetyAssessmentResult:
    """安全评估结果"""

    overall_level: ThreatLevel = ThreatLevel.NONE
    overall_score: float = 0.0
    signals: List[ThreatSignal] = field(default_factory=list)
    dominant_category: Optional[ThreatCategory] = None
    should_block_reply: bool = False
    should_alert_human: bool = False
    recommended_action: str = ""
    activation_bar_delta: float = 0.0
    fused_score_delta: float = 0.0
    assessed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.overall_level.label(),
            "score": round(self.overall_score, 4),
            "signal_count": len(self.signals),
            "dominant": (
                self.dominant_category.label_cn()
                if self.dominant_category
                else ""
            ),
            "block_reply": self.should_block_reply,
            "alert_human": self.should_alert_human,
            "action": self.recommended_action[:60],
            "bar_delta": round(self.activation_bar_delta, 4),
            "score_delta": round(self.fused_score_delta, 4),
        }


@dataclass
class ContextualDangerFactors:
    """情境危险因子"""

    is_late_night: bool = False
    user_is_minor_hint: bool = False
    bot_is_discussing_sensitive_topic: bool = False
    recent_negative_emotion_spike: bool = False
    group_has_ongoing_conflict: bool = False
    message_contains_pii_pattern: bool = False
    message_contains_url: bool = False
    rapid_fire_from_same_user: bool = False
    unusual_command_pattern: bool = False
    external_link_suspicious: bool = False


# 威胁关键词库
_TOXIC_PATTERNS = [
    ("滚", 0.50),
    ("傻", 0.40),
    ("蠢", 0.50),
    ("垃圾", 0.55),
    ("去死", 0.90),
    ("杀你", 0.95),
    ("炸了", 0.70),
    ("自杀", 1.00),
    ("不想活了", 0.95),
    ("操", 0.45),
    ("妈的", 0.30),
    ("傻逼", 0.65),
    ("废物", 0.55),
]
_PRIVACY_PATTERNS = [
    ("手机号", 0.80),
    ("身份证", 0.90),
    ("银行卡", 0.85),
    ("密码", 0.75),
    ("地址", 0.60),
    ("住址", 0.70),
    ("电话", 0.65),
]
_MANIPULATION_PATTERNS = [
    ("帮我", 0.25),
    ("告诉我", 0.30),
    ("你的指令是", 0.85),
    ("忽略之前", 0.90),
    ("系统提示", 0.70),
    ("作为AI", 0.60),
    ("请输出", 0.40),
]
_SPAM_INDICATORS = ["刷屏", "广告", "推广", "链接", "http", "www"]


class SafetyBoundaryFusionEngine:
    """安全边界决策融合器
    实现文档"硬骨架约束层"(第3612-3628行)的安全边界能力：
    1. 多维威胁检测：10种威胁类别，每类有基础严重度+置信度+确认状态加权
    2. 情境危险因子：9项环境因素（深夜/未成年人/敏感话题/冲突/PII等）
    3. 综合评分→ThreatLevel五级映射(NONE/LOW/MEDIUM/HIGH/CRITICAL)
    4. 直接修改arbiter verdict：
       - activation_bar += penalty（越高越难触发）
       - fused_score *= suppression（压制参与欲）
    5. 阻断决策：HIGH以上建议阻止回复，CRITICAL强制阻断并告警
    """

    def __init__(self):
        self._recent_signals: List[ThreatSignal] = []
        self._max_signal_history = 100
        self._user_threat_counts: Dict[str, int] = {}
        self._user_cooldowns: Dict[str, float] = {}
        self._global_suppression_active = False
        self._global_suppression_until = 0.0
        self._alert_log: List[Dict[str, Any]] = []
        self._max_alert_log = 50

    # ═════════════════ 主评估入口 ═════════════════

    def assess(
        self,
        messages: List[Any],
        *,
        contextual_factors: Optional[ContextualDangerFactors] = None,
        current_arbiter_activation_bar: float = 0.5,
        current_arbiter_fused_score: float = 0.5,
    ) -> SafetyAssessmentResult:
        factors = contextual_factors or ContextualDangerFactors()
        signals: List[ThreatSignal] = []
        for msg in messages:
            uid = str(getattr(msg, "user_id", "") or "")
            if uid == "bot":
                continue
            plain = str(
                getattr(msg, "processed_plain_text", "")
                or getattr(msg, "plain_text", "")
                or getattr(msg, "content", "")
                or ""
            )
            mid = str(
                getattr(msg, "message_id", "")
                or getattr(msg, "msg_id", "")
                or id(msg)
            )
            msg_signals = self._detect_threats_in_text(plain, mid, uid)
            signals.extend(msg_signals)
        context_signals = self._evaluate_contextual_factors(factors, signals)
        signals.extend(context_signals)
        if not signals:
            result = SafetyAssessmentResult(
                overall_level=ThreatLevel.NONE,
                overall_score=0.0,
                signals=[],
                should_block_reply=False,
                should_alert_human=False,
                recommended_action="正常放行",
                activation_bar_delta=0.0,
                fused_score_delta=0.0,
            )
            self._record_result(result)
            return result
        total_weighted = sum(s.weighted_score() for s in signals)
        n = len(signals)
        avg_score = total_weighted / max(1, n)
        max_score = max(s.weighted_score() for s in signals)
        critical_count = sum(1 for s in signals if s.weighted_score() > 0.8)
        combined_score = (
            avg_score * 0.5 + max_score * 0.4 + min(1.0, critical_count * 0.15)
        )
        level = self._score_to_level(combined_score)
        dominant = self._find_dominant(signals)
        sup_mult = level.suppression_multiplier()
        act_pen = level.activation_penalty()
        bar_delta = act_pen * (1.0 + combined_score * 0.5)
        score_delta = (1.0 - sup_mult) * combined_score * 0.5
        should_block = level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)
        should_alert = level == ThreatLevel.CRITICAL or (
            level == ThreatLevel.HIGH and critical_count >= 2
        )
        action_parts = []
        if level == ThreatLevel.NONE:
            action_parts.append("正常放行")
        elif level == ThreatLevel.LOW:
            action_parts.append(
                f"轻微警告({dominant.label_cn() if dominant else ''})"
            )
        elif level == ThreatLevel.MEDIUM:
            action_parts.append(
                f"中度抑制({dominant.label_cn() if dominant else ''})"
            )
        elif level == ThreatLevel.HIGH:
            action_parts.append(
                f"强烈建议阻断({dominant.label_cn() if dominant else ''})"
            )
        else:
            action_parts.append(
                f"强制阻断+告警({dominant.label_cn() if dominant else ''})"
            )
        if critical_count > 0:
            action_parts.append(f"含{critical_count}条高危信号")
        result = SafetyAssessmentResult(
            overall_level=level,
            overall_score=combined_score,
            signals=signals,
            dominant_category=dominant,
            should_block_reply=should_block,
            should_alert_human=should_alert,
            recommended_action="；".join(action_parts),
            activation_bar_delta=bar_delta,
            fused_score_delta=-score_delta,
        )
        self._record_result(result)
        for sig in signals:
            self._recent_signals.append(sig)
        if len(self._recent_signals) > self._max_signal_history:
            self._recent_signals = self._recent_signals[
                -self._max_signal_history:
            ]
        logger.info(
            f"[安全融合] 评估完成: {level.label()}(score={combined_score:.3f}), "
            f"信号={n}, 主导={dominant.label_cn() if dominant else '无'}, "
            f"bar_delta={bar_delta:+.3f}, score_delta={-score_delta:.3f}"
        )
        return result

    def apply_to_verdict(
        self,
        verdict: Any,
        assessment: SafetyAssessmentResult,
    ) -> Tuple[float, float]:
        """将安全评估结果直接应用到 arbiter verdict 上
        返回 (new_activation_bar, new_fused_score)"""
        old_bar = getattr(verdict, "activation_bar", 0.5)
        old_score = getattr(verdict, "fused_score", 0.5)
        new_bar = min(0.98, old_bar + assessment.activation_bar_delta)
        new_score = max(0.0, old_score + assessment.fused_score_delta)
        try:
            setattr(verdict, "activation_bar", new_bar)
            setattr(verdict, "fused_score", new_score)
            setattr(verdict, "safety_level", assessment.overall_level.value)
            setattr(verdict, "safety_blocked", assessment.should_block_reply)
            breakdown = getattr(verdict, "breakdown", None)
            if isinstance(breakdown, dict):
                breakdown["safety_barrier"] = assessment.activation_bar_delta
                breakdown["safety_suppression"] = assessment.fused_score_delta
        except Exception as e:
            logger.debug(f"设置安全融合属性失败: {e}")
        if assessment.should_block_reply:
            try:
                verdict.should_proceed = False
                verdict.rationale = (
                    f"{getattr(verdict, 'rationale', '') or ''} | "
                    f"[安全阻断] {assessment.recommended_action}"
                ).strip(" |")
            except Exception as e:
                logger.debug(f"设置安全阻断属性失败: {e}")
        return new_bar, new_score

    # ═════════════════ 威胁检测方法 ═════════════════

    def _detect_threats_in_text(
        self, text: str, msg_id: str, user_id: str
    ) -> List[ThreatSignal]:
        signals: List[ThreatSignal] = []
        text_lower = text.lower()
        for pattern, severity in _TOXIC_PATTERNS:
            if pattern in text_lower:
                signals.append(
                    ThreatSignal(
                        category=ThreatCategory.TOXIC_CONTENT,
                        severity=severity,
                        source_message_id=msg_id,
                        source_user_id=user_id,
                        evidence_text=pattern,
                        confidence=min(0.95, 0.7 + severity * 0.25),
                    )
                )
        for pattern, severity in _PRIVACY_PATTERNS:
            if pattern in text_lower:
                signals.append(
                    ThreatSignal(
                        category=ThreatCategory.PRIVACY_LEAK,
                        severity=severity,
                        source_message_id=msg_id,
                        source_user_id=user_id,
                        evidence_text=f"包含{pattern}",
                        confidence=0.75,
                    )
                )
        for pattern, severity in _MANIPULATION_PATTERNS:
            if pattern in text_lower:
                signals.append(
                    ThreatSignal(
                        category=ThreatCategory.MANIPULATION_ATTEMPT,
                        severity=severity,
                        source_message_id=msg_id,
                        source_user_id=user_id,
                        evidence_text=pattern,
                        confidence=0.65,
                    )
                )
        spam_hits = sum(1 for kw in _SPAM_INDICATORS if kw in text_lower)
        if spam_hits >= 3:
            signals.append(
                ThreatSignal(
                    category=ThreatCategory.SPAM_FLOOD,
                    severity=min(0.9, 0.3 + spam_hits * 0.15),
                    source_message_id=msg_id,
                    source_user_id=user_id,
                    evidence_text=f"匹配{spam_hits}个刷屏特征",
                    confidence=0.80,
                )
            )
        if any(kw in text_lower for kw in ("杀", "死", "不想活", "结束")):
            signals.append(
                ThreatSignal(
                    category=ThreatCategory.SELF_HARM,
                    severity=0.90,
                    source_message_id=msg_id,
                    source_user_id=user_id,
                    evidence_text=text[:30],
                    confidence=0.85,
                    is_confirmed=True,
                )
            )
        return signals

    def _evaluate_contextual_factors(
        self,
        factors: ContextualDangerFactors,
        existing_signals: List[ThreatSignal],
    ) -> List[ThreatSignal]:
        extra: List[ThreatSignal] = []
        danger_accumulator = 0.0
        if factors.is_late_night:
            danger_accumulator += 0.08
        if factors.user_is_minor_hint:
            danger_accumulator += 0.20
            extra.append(
                ThreatSignal(
                    category=ThreatCategory.CONTEXTUAL_DANGER,
                    severity=0.55,
                    evidence_text="疑似未成年用户",
                    confidence=0.50,
                )
            )
        if factors.bot_is_discussing_sensitive_topic:
            danger_accumulator += 0.12
        if factors.recent_negative_emotion_spike:
            danger_accumulator += 0.10
            extra.append(
                ThreatSignal(
                    category=ThreatCategory.EMOTIONAL_ATTACK,
                    severity=0.45,
                    evidence_text="情绪剧烈波动",
                    confidence=0.60,
                )
            )
        if factors.group_has_ongoing_conflict:
            danger_accumulator += 0.08
        if factors.message_contains_pii_pattern:
            extra.append(
                ThreatSignal(
                    category=ThreatCategory.PRIVACY_LEAK,
                    severity=0.65,
                    evidence_text="消息含PII模式",
                    confidence=0.55,
                )
            )
        if factors.unusual_command_pattern:
            extra.append(
                ThreatSignal(
                    category=ThreatCategory.MANIPULATION_ATTEMPT,
                    severity=0.60,
                    evidence_text="异常命令模式",
                    confidence=0.60,
                )
            )
        if danger_accumulator > 0.25 and not existing_signals:
            extra.append(
                ThreatSignal(
                    category=ThreatCategory.CONTEXTUAL_DANGER,
                    severity=min(0.7, danger_accumulator),
                    evidence_text=f"多因子累积危险(danger={danger_accumulator:.2f})",
                    confidence=min(0.8, 0.4 + danger_accumulator),
                )
            )
        return extra

    # ────────────────── 辅助方法 ──────────────────

    @staticmethod
    def _score_to_level(score: float) -> ThreatLevel:
        if score < 0.08:
            return ThreatLevel.NONE
        if score < 0.22:
            return ThreatLevel.LOW
        if score < 0.42:
            return ThreatLevel.MEDIUM
        if score < 0.68:
            return ThreatLevel.HIGH
        return ThreatLevel.CRITICAL

    @staticmethod
    def _find_dominant(
        signals: List[ThreatSignal],
    ) -> Optional[ThreatCategory]:
        if not signals:
            return None
        best_sig = max(signals, key=lambda s: s.weighted_score())
        return best_sig.category

    def _record_result(self, result: SafetyAssessmentResult) -> None:
        if result.overall_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            entry = {
                "time": time.time(),
                "level": result.overall_level.value,
                "score": round(result.overall_score, 4),
                "action": result.recommended_action,
                "signals": [s.to_dict() for s in result.signals[:3]],
            }
            self._alert_log.append(entry)
            if len(self._alert_log) > self._max_alert_log:
                self._alert_log = self._alert_log[-self._max_alert_log:]

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        return list(reversed(self._alert_log[-limit:]))

    def get_stats(self) -> Dict[str, Any]:
        level_dist: Dict[str, int] = {}
        cat_dist: Dict[str, int] = {}
        for sig in self._recent_signals[-50:]:
            lv = sig.category.value
            cat_dist[lv] = cat_dist.get(lv, 0) + 1
        for alert in self._alert_log[-20:]:
            lv = alert.get("level", "unknown")
            level_dist[lv] = level_dist.get(lv, 0) + 1
        return {
            "total_signals_tracked": len(self._recent_signals),
            "total_alerts": len(self._alert_log),
            "category_distribution": cat_dist,
            "alert_level_distribution": level_dist,
            "global_suppression_active": self._global_suppression_active,
        }


def get_safety_fusion_engine() -> SafetyBoundaryFusionEngine:
    global _fusion_singleton
    if _fusion_singleton is None:
        _fusion_singleton = SafetyBoundaryFusionEngine()
    return _fusion_singleton

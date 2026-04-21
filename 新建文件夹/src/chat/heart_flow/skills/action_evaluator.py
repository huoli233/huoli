import time as _tm
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from src.common.logger import get_logger

logger = get_logger("action_judge")


# ---------------------------------------------------------------------------
#  枚举 & 数据载体
# ---------------------------------------------------------------------------


class ActionCategory(Enum):
    """候选行为类别"""

    EMIT_STICKER = "emit_sticker"
    RETRACT_MSG = "retract_msg"
    EMIT_EMOJI = "emit_emoji"
    ENGAGE_CHAT = "engage_chat"
    TAKE_BREAK = "take_break"


class VerdictGrade(Enum):
    """评判等级（从低到高）"""

    BLOCKED = "blocked"
    DISCOURAGED = "discouraged"
    INDIFFERENT = "indifferent"
    FAVORABLE = "favorable"
    IMPERATIVE = "imperative"


# 等级阈值（降序匹配）
_GRADE_CUTOFFS = [
    (90, VerdictGrade.IMPERATIVE),
    (70, VerdictGrade.FAVORABLE),
    (50, VerdictGrade.INDIFFERENT),
    (30, VerdictGrade.DISCOURAGED),
]


@dataclass
class EvaluationInputs:
    """评判所需的外部因子快照"""

    affection: float = 0.0
    trust_level: float = 0.0
    irritation: float = 0.0
    scene_kind: str = "group"
    relevance: float = 0.5
    timing_quality: float = 0.5
    hazard_ratio: float = 0.0
    dialogue_rounds: int = 0
    peer_activity: float = 0.5
    chat_stamina: float = 100.0
    brain_reserve: float = 100.0
    streak_count: int = 0
    mood_tag: str = ""
    extra_signals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgmentResult:
    """评判输出"""

    composite_score: float
    grade: VerdictGrade
    dimension_breakdown: Dict[str, float]
    rationale: str
    advice: str
    reliability: float
    judged_at: float = field(default_factory=_tm.time)


# ---------------------------------------------------------------------------
#  维度评分函数族（纯函数，方便替换或测试）
# ---------------------------------------------------------------------------


def _dim_sentiment(inp: EvaluationInputs) -> float:
    """情感维度：好感+信任正向、烦躁负向"""
    aff_norm = (inp.affection + 100.0) / 200.0
    trs_norm = (inp.trust_level + 100.0) / 200.0
    irr_norm = (inp.irritation + 100.0) / 200.0
    raw = (aff_norm * 0.4 + trs_norm * 0.4 - irr_norm * 0.2) * 100.0
    return max(0.0, min(100.0, raw))


# 场景适配基础分表（scene_kind → ActionCategory.value → 基础分）
_SCENE_BASE: Dict[str, Dict[str, float]] = {
    "group": {
        "emit_sticker": 90,
        "retract_msg": 50,
        "emit_emoji": 85,
        "engage_chat": 80,
        "take_break": 60,
    },
    "private": {
        "emit_sticker": 85,
        "retract_msg": 70,
        "emit_emoji": 80,
        "engage_chat": 90,
        "take_break": 50,
    },
}


def _dim_scene(inp: EvaluationInputs, cat: ActionCategory) -> float:
    """场景维度：从预设表查基准分"""
    return _SCENE_BASE.get(inp.scene_kind, {}).get(cat.value, 50.0)


def _dim_context(inp: EvaluationInputs) -> float:
    """上下文维度：关联度 + 对话轮次修正"""
    base = inp.relevance * 100.0
    if inp.dialogue_rounds > 10:
        base += 10.0
    elif inp.dialogue_rounds < 3:
        base -= 10.0
    return max(0.0, min(100.0, base))


def _dim_timing(inp: EvaluationInputs) -> float:
    """时机维度：时机质量 + 对方活跃度修正"""
    base = inp.timing_quality * 100.0
    activity_adj = (inp.peer_activity - 0.5) * 20.0
    return max(0.0, min(100.0, base + activity_adj))


def _dim_hazard(inp: EvaluationInputs) -> float:
    """风险维度：风险越高分越低"""
    return max(0.0, min(100.0, 100.0 - inp.hazard_ratio * 100.0))


# 维度注册表：(维度名, 评分函数需要的签名标记, 权重)
_DIMENSION_REGISTRY = [
    ("sentiment", False, 0.30),
    ("scene", True, 0.20),
    ("context", False, 0.20),
    ("timing", False, 0.15),
    ("hazard", False, 0.15),
]

_DIM_FUNCS = {
    "sentiment": _dim_sentiment,
    "scene": _dim_scene,
    "context": _dim_context,
    "timing": _dim_timing,
    "hazard": _dim_hazard,
}


# ---------------------------------------------------------------------------
#  判定等级 & 文本构造
# ---------------------------------------------------------------------------


def _classify_grade(total: float) -> VerdictGrade:
    for cutoff, grade in _GRADE_CUTOFFS:
        if total >= cutoff:
            return grade
    return VerdictGrade.BLOCKED


_ACTION_LABELS: Dict[ActionCategory, str] = {
    ActionCategory.EMIT_STICKER: "发送表情包",
    ActionCategory.RETRACT_MSG: "撤回消息",
    ActionCategory.EMIT_EMOJI: "发送表情",
    ActionCategory.ENGAGE_CHAT: "加入聊天",
    ActionCategory.TAKE_BREAK: "休息",
}

_GRADE_ADVICE_TPL: Dict[VerdictGrade, str] = {
    VerdictGrade.IMPERATIVE: "强烈建议{act}",
    VerdictGrade.FAVORABLE: "建议{act}",
    VerdictGrade.INDIFFERENT: "可以考虑{act}",
    VerdictGrade.DISCOURAGED: "不建议{act}",
    VerdictGrade.BLOCKED: "禁止{act}",
}


def _build_rationale(
    dims: Dict[str, float], inp: EvaluationInputs, cat: ActionCategory
) -> str:
    """拼接可读的评判理由"""
    clauses = []
    if dims.get("sentiment", 0) >= 70:
        clauses.append(
            f"情感良好(好感{inp.affection:.0f}/信任{inp.trust_level:.0f})"
        )
    elif dims.get("sentiment", 0) < 40:
        clauses.append(f"情感不佳(烦躁{inp.irritation:.0f})")
    if dims.get("scene", 0) >= 80:
        clauses.append(f"场景适配({inp.scene_kind})")
    elif dims.get("scene", 0) < 50:
        clauses.append(f"场景不适({inp.scene_kind})")
    if dims.get("context", 0) >= 70:
        clauses.append(f"上下文相关({inp.dialogue_rounds}轮)")
    elif dims.get("context", 0) < 40:
        clauses.append("上下文弱")
    if dims.get("timing", 0) >= 70:
        clauses.append(f"时机好(活跃{inp.peer_activity:.1f})")
    elif dims.get("timing", 0) < 40:
        clauses.append("时机差")
    if dims.get("hazard", 0) < 60:
        clauses.append(f"风险高({inp.hazard_ratio:.2f})")
    return "；".join(clauses) if clauses else "综合评估"


def _calc_reliability(dim_values: list) -> float:
    """各维度分数越一致，可靠度越高"""
    avg = sum(dim_values) / len(dim_values)
    variance = sum((v - avg) ** 2 for v in dim_values) / len(dim_values)
    spread = variance**0.5
    return max(0.0, min(1.0, 1.0 - spread / 50.0))


# ---------------------------------------------------------------------------
#  评判器主体
# ---------------------------------------------------------------------------


class ActionJudge:
    """统一行为评判器，对候选动作产出加权打分 + 等级判定"""

    def evaluate(
        self, category: ActionCategory, inputs: EvaluationInputs
    ) -> JudgmentResult:
        """执行一次完整评判"""
        dim_scores: Dict[str, float] = {}
        for dim_name, needs_cat, _w in _DIMENSION_REGISTRY:
            func = _DIM_FUNCS[dim_name]
            if needs_cat:
                dim_scores[dim_name] = func(inputs, category)
            else:
                dim_scores[dim_name] = func(inputs)
        composite = sum(dim_scores[d] * w for d, _, w in _DIMENSION_REGISTRY)
        grade = _classify_grade(composite)
        rationale = _build_rationale(dim_scores, inputs, category)
        act_label = _ACTION_LABELS.get(category, "执行操作")
        advice = _GRADE_ADVICE_TPL.get(grade, "{act}").format(act=act_label)
        reliability = _calc_reliability(list(dim_scores.values()))
        logger.debug(
            f"[评判] {category.value} 总分={composite:.1f} "
            f"等级={grade.value} 可靠={reliability:.2f}"
        )
        return JudgmentResult(
            composite_score=composite,
            grade=grade,
            dimension_breakdown=dim_scores,
            rationale=rationale,
            advice=advice,
            reliability=reliability,
        )


# ---------------------------------------------------------------------------
#  模块单例
# ---------------------------------------------------------------------------

_judge_instance: Optional[ActionJudge] = None


def acquire_action_judge() -> ActionJudge:
    """获取全局 ActionJudge 单例"""
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = ActionJudge()
    return _judge_instance

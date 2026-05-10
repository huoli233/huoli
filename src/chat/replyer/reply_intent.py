import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


GENERIC_FAST_ACK_REPLIES = (
    "嗯，是啊",
    "嗯是啊",
    "是啊",
    "对啊",
    "确实",
    "嗯嗯",
    "好的，你说",
    "好的你说",
    "你说",
    "继续说",
    "收到",
    "请问",
    "当然",
)

LAYERED_TONE_TRANSLATIONS = {
    "neutral": "",
    "gentle": "轻一点、别硬撑",
    "supportive": "有支撑感但别客服腔",
    "warm": "放松一点",
    "guarded": "防备、谨慎",
    "tense": "绷着、少解释",
    "impatient": "不耐烦、说重点",
    "mildly_annoyed": "有点烦、别殷勤",
    "sarcastic": "带点刺但别攻击",
    "dismissive": "敷衍、收着",
    "cold_rejection": "冷淡拒绝、低承诺",
}

LAYERED_STANCE_OVERRIDES = {
    "guarded": "有防备，不要显得熟络",
    "tense": "压力上来了，少解释少兜圈",
    "impatient": "不耐烦，尽快收住",
    "mildly_annoyed": "有点烦，能回但不要显得殷勤",
    "sarcastic": "带点刺地接住，但不要攻击",
    "dismissive": "不想配合延长话题，短短回掉",
    "cold_rejection": "明显不想配合，只给低承诺回应",
}


def _clean_text(text: Any, limit: int = 160) -> str:
    payload = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(payload) <= limit:
        return payload
    return payload[: limit - 1].rstrip("，。；、 ") + "…"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [_clean_text(item, 40) for item in value if _clean_text(item, 40)]
    text = _clean_text(value, 80)
    return [text] if text else []


def _dedupe_texts(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        cleaned = _clean_text(item, 40)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _pressure_signal(value: float) -> float:
    """兼容历史日志中的 0-10 压力显示与新情绪系统的 0-100 压力值。"""
    if 0.0 < value <= 10.0:
        return value * 10.0
    return value


def is_low_info_input(text: str) -> bool:
    payload = str(text or "").strip()
    normalized = re.sub(r"[\s，。！？、,.!?~～…]+", "", payload)
    if not normalized:
        return False
    if len(normalized) > 8:
        return False
    if re.search(r"[?？!！]", payload):
        return False
    if re.search(r"(怎么|咋|为什么|是不是|能不能|要不要|丢了|坏了|没了|在哪|多少|啥|什么|吗|呢)", payload):
        return False
    return True


@dataclass
class ReplyIntentPacket:
    """回复生成器的统一意图契约。"""

    target: str = ""
    target_text: str = ""
    trigger_reason: str = ""
    stance: str = "自然接话"
    tone: str = "普通口语"
    relation_attitude: str = "中性"
    length_policy: str = "短句"
    avoid_patterns: List[str] = field(default_factory=list)
    reference_info: str = ""
    quote_policy: str = "none"
    fast_path: bool = False
    allow_followup: bool = True
    low_info_strategy: str = "none"
    mood: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "target_text": self.target_text,
            "trigger_reason": self.trigger_reason,
            "stance": self.stance,
            "tone": self.tone,
            "relation_attitude": self.relation_attitude,
            "length_policy": self.length_policy,
            "avoid_patterns": list(self.avoid_patterns),
            "reference_info": self.reference_info,
            "quote_policy": self.quote_policy,
            "fast_path": self.fast_path,
            "allow_followup": self.allow_followup,
            "low_info_strategy": self.low_info_strategy,
            "mood": self.mood,
        }


def coerce_reply_intent_packet(value: Any) -> Optional[ReplyIntentPacket]:
    if value is None:
        return None
    if isinstance(value, ReplyIntentPacket):
        return value
    if isinstance(value, dict):
        fields = ReplyIntentPacket.__dataclass_fields__
        kwargs = {key: value.get(key) for key in fields if key in value}
        avoid = kwargs.get("avoid_patterns")
        if avoid is None:
            kwargs["avoid_patterns"] = []
        elif not isinstance(avoid, list):
            kwargs["avoid_patterns"] = [str(avoid)]
        return ReplyIntentPacket(**kwargs)
    return None


def ensure_reply_intent_packet(
    value: Any,
    *,
    target: str = "",
    target_text: str = "",
    trigger_reason: str = "",
    fast_path: bool = False,
) -> ReplyIntentPacket:
    intent = coerce_reply_intent_packet(value)
    if intent is not None:
        intent.fast_path = bool(fast_path or intent.fast_path)
        if target and not intent.target:
            intent.target = _clean_text(target, 60)
        if target_text and not intent.target_text:
            intent.target_text = _clean_text(target_text, 160)
        if trigger_reason and not intent.trigger_reason:
            intent.trigger_reason = _clean_text(trigger_reason, 120)
        intent.avoid_patterns = _dedupe_texts(intent.avoid_patterns or list(GENERIC_FAST_ACK_REPLIES))
        return intent
    return relation_to_reply_intent(
        target=target,
        target_text=target_text,
        trigger_reason=trigger_reason or "当前轮需要回应",
        fast_path=fast_path,
    )


def build_reply_intent_from_message(
    *,
    channel_id: str = "",
    message: Any = None,
    trigger_reason: str = "",
    reference_info: str = "",
    quote_policy: str = "none",
    fast_path: bool = False,
    allow_followup: Optional[bool] = None,
    relation_view: Optional[Dict[str, Any]] = None,
) -> ReplyIntentPacket:
    target_uid = str(getattr(message, "user_id", "") or "").strip()
    target_name = str(
        getattr(message, "user_nickname", "")
        or getattr(message, "user_cardname", "")
        or getattr(message, "nickname", "")
        or target_uid
        or "对方"
    ).strip()
    target_text = str(
        getattr(message, "processed_plain_text", "")
        or getattr(message, "plain_text", "")
        or getattr(message, "content", "")
        or ""
    ).strip()
    relation = dict(relation_view or {})
    if not relation and channel_id and target_uid:
        try:
            from src.core.world_snapshot import build_relation_rapport_snapshot

            relation = build_relation_rapport_snapshot(channel_id=channel_id, user_id=target_uid) or {}
        except Exception:
            relation = {}
    response_mode: Dict[str, Any] = {}
    if channel_id and target_uid:
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker

            response_mode = get_emotion_tracker(channel_id).get_layered_response_mode(target_uid) or {}
        except Exception:
            response_mode = {}
    return relation_to_reply_intent(
        target=target_name,
        target_text=target_text,
        trigger_reason=trigger_reason,
        relation_view=relation,
        response_mode=response_mode,
        reference_info=reference_info,
        quote_policy=quote_policy,
        fast_path=fast_path,
        allow_followup=allow_followup,
    )


def relation_to_reply_intent(
    *,
    target: str = "",
    target_text: str = "",
    trigger_reason: str = "",
    relation_view: Optional[Dict[str, Any]] = None,
    response_mode: Optional[Dict[str, Any]] = None,
    reference_info: str = "",
    quote_policy: str = "none",
    fast_path: bool = False,
    night_hint: str = "",
    allow_followup: Optional[bool] = None,
) -> ReplyIntentPacket:
    relation = relation_view or {}
    mode = response_mode or {}
    affection = _safe_float(relation.get("affection"))
    trust = _safe_float(relation.get("trust_value", relation.get("trust_score")))
    annoyance = _safe_float(relation.get("annoyance_value", relation.get("annoyance")))
    pressure = _safe_float(relation.get("psychological_pressure", relation.get("pressure")))
    trauma = _safe_float(relation.get("trauma_score"))
    blocked = bool(relation.get("is_user_blocked", False))
    relation_label = _clean_text(
        relation.get("personal_impression")
        or relation.get("relationship")
        or relation.get("legacy_relationship_label")
        or relation.get("custom_label")
        or "普通"
    )
    layered_tone = str(mode.get("tone", "") or "").strip()
    layered_length = str(mode.get("response_length", "") or "").strip()
    layered_tone_cn = LAYERED_TONE_TRANSLATIONS.get(
        layered_tone,
        _clean_text(layered_tone, 40),
    )
    layered_avoid = _as_text_list(mode.get("avoid_patterns"))
    pressure_level = _pressure_signal(pressure)

    stance = "顺着当前消息自然回应"
    relation_attitude = (
        f"{relation_label}，好感{affection:.1f}，信任{trust:.1f}，"
        f"厌烦{annoyance:.1f}，压力{pressure:.1f}"
    )
    tone = "普通口语"
    length_policy = "一句短口语" if fast_path else "自然短句"
    avoid = list(GENERIC_FAST_ACK_REPLIES) + layered_avoid

    if blocked or trauma >= 5:
        stance = "保持防备，只给低承诺回应"
        tone = "疏离、克制"
        length_policy = "很短，不展开"
        allow = False
    elif annoyance >= 70:
        stance = "明显不耐烦，不主动配合延长话题"
        tone = "冷淡、短"
        length_policy = "一句内收住"
        allow = False
    elif annoyance >= 35:
        stance = "有点烦，能回但不要显得殷勤"
        tone = "淡一点"
        length_policy = "短句"
        allow = False
    elif trust < 15 and affection <= 15:
        stance = "不熟也不太信任，保持距离"
        tone = "谨慎、不过分热情"
        length_policy = "短句，不主动展开"
        allow = False
    elif pressure_level >= 60:
        stance = "压力已经偏高，少解释少兜圈"
        tone = "绷着、别热情接待"
        length_policy = "一句内收住"
        allow = False
    elif pressure_level >= 45:
        stance = "压力上来了，少解释少推进"
        tone = "紧一点、别热情接待"
        length_policy = "短句"
        allow = False
    elif pressure_level >= 15:
        stance = "有一点压力，别显得太积极"
        tone = "稍微收着"
        length_policy = "短句"
        allow = False
    elif affection >= 55 and trust >= 35:
        stance = "关系相对放松，可以自然一点"
        tone = "松一点但不油"
        length_policy = "自然短句"
        allow = True
    else:
        allow = True

    if layered_tone_cn:
        tone = _merge_tone(tone, layered_tone_cn)
    if layered_tone in LAYERED_STANCE_OVERRIDES:
        stance = LAYERED_STANCE_OVERRIDES[layered_tone]
        if layered_tone in {
            "guarded",
            "tense",
            "impatient",
            "mildly_annoyed",
            "sarcastic",
            "dismissive",
            "cold_rejection",
        }:
            allow = False
    if layered_length in ("minimal", "short", "concise"):
        length_policy = "一句短话，点到就停"
    elif layered_length == "detailed" and not fast_path:
        length_policy = "可以多半句，但仍然口语"

    low_info = is_low_info_input(target_text)
    low_info_strategy = "none"
    if low_info:
        if blocked or trauma >= 5:
            low_info_strategy = "观察不回"
            stance = "对方信息太少，本来不想接；被迫回复也只给很低承诺短句"
            allow = False
        elif annoyance >= 35 or pressure_level >= 45 or trust < 15:
            low_info_strategy = "带情绪短回"
            if annoyance >= 35:
                stance = "对方信息太少，你有点烦，只短短接一下，不追着服务"
            elif pressure_level >= 45:
                stance = "对方信息太少，而且压力上来了，只短短接一下，不追着服务"
            else:
                stance = "对方信息太少，你不太信任对方，只短短接一下，不追着服务"
            allow = False
        elif not allow:
            low_info_strategy = "带情绪短回"
            stance = f"信息不完整，{stance}，只短短接一下，不追着服务"
        elif fast_path:
            low_info_strategy = "轻追问"
            stance = "信息不完整，轻轻追问半句或顺手接住"
        else:
            low_info_strategy = "可接话"
            stance = "信息不完整，先结合前情自然接一句"

    if night_hint:
        tone = _merge_tone(tone, "困倦、慢一点")
        length_policy = "短一点，别精神饱满"

    if allow_followup is not None:
        allow = bool(allow_followup)

    packet = ReplyIntentPacket(
        target=_clean_text(target, 60),
        target_text=_clean_text(target_text, 160),
        trigger_reason=_clean_text(trigger_reason, 120),
        stance=stance,
        tone=tone,
        relation_attitude=relation_attitude,
        length_policy=length_policy,
        avoid_patterns=avoid,
        reference_info=_clean_text(reference_info, 420),
        quote_policy=str(quote_policy or "none"),
        fast_path=bool(fast_path),
        allow_followup=bool(allow),
        low_info_strategy=low_info_strategy,
        mood=_clean_text(str(relation.get("mood", "") or ""), 40),
    )
    packet.avoid_patterns = _dedupe_texts(packet.avoid_patterns)
    return packet


def _merge_tone(base: str, extra: str) -> str:
    base_text = _clean_text(base, 60)
    extra_text = _clean_text(extra, 60)
    if not base_text:
        return extra_text
    if not extra_text or extra_text in base_text:
        return base_text
    return f"{base_text}，{extra_text}"


def render_reply_intent_block(packet: ReplyIntentPacket) -> str:
    intent = packet
    avoid = "、".join(_clean_text(item, 24) for item in intent.avoid_patterns if _clean_text(item, 24))
    lines = [
        "【回复意图包】",
        f"回复目标：{intent.target or '当前说话的人'}",
        f"目标内容：{intent.target_text or '当前消息'}",
        f"为什么回：{intent.trigger_reason or '当前轮需要回应'}",
        f"当前立场：{intent.stance}",
        f"情绪语气：{intent.tone}",
        f"关系态度：{intent.relation_attitude}",
        f"长度策略：{intent.length_policy}",
        f"低信息策略：{intent.low_info_strategy}",
        f"引用策略：{intent.quote_policy}",
        f"允许追问：{'是' if intent.allow_followup else '否'}",
    ]
    if intent.mood:
        lines.append(f"心情底色：{intent.mood}")
    if intent.reference_info:
        lines.append(f"必要参考：{intent.reference_info}")
    if avoid:
        lines.append(f"禁用模式：{avoid}")
    lines.append("按这个意图包自然说一句；不要复述字段名，不要解释规则，只输出要发的话。")
    return "\n".join(lines)

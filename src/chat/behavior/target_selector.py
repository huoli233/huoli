from typing import Any, Iterable, Optional


def choose_preferred_message_target(
    anchor_message: Any,
    candidate_messages: Iterable[Any],
    relationship_candidates: Optional[list[tuple[str, float]]] = None,
    active_user_ids: Optional[Iterable[str]] = None,
) -> Any:
    """根据最近话头、续接信号、关系权重和活跃用户约束挑选更合适的目标。"""
    if anchor_message is None:
        return None

    def _text_of(message: Any) -> str:
        return str(
            getattr(message, "processed_plain_text", "")
            or getattr(message, "plain_text", "")
            or getattr(message, "content", "")
            or ""
        ).strip()

    def _normalized(text: str) -> str:
        return "".join(
            ch.lower()
            for ch in str(text or "")
            if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"
        )

    anchor_text = _text_of(anchor_message)
    anchor_user_id = str(getattr(anchor_message, "user_id", "") or "").strip()
    anchor_normalized = _normalized(anchor_text)
    anchor_continuation_hint = str(
        getattr(anchor_message, "context_continuation_hint", "") or ""
    ).strip()
    anchor_repeat_short = bool(
        getattr(anchor_message, "is_repeated_short_input", False)
    )

    normalized_relationships: dict[str, float] = {}
    for candidate in relationship_candidates or []:
        if not isinstance(candidate, (list, tuple)) or len(candidate) < 2:
            continue
        candidate_id = str(candidate[0] or "").strip()
        if not candidate_id:
            continue
        try:
            normalized_relationships[candidate_id] = float(candidate[1] or 0.0)
        except Exception:
            normalized_relationships[candidate_id] = 0.0

    active_user_set = {
        str(user_id or "").strip()
        for user_id in (active_user_ids or [])
        if str(user_id or "").strip()
    }

    best_message = anchor_message
    best_score = normalized_relationships.get(anchor_user_id, 0.0)
    best_is_active = (
        anchor_user_id in active_user_set if active_user_set else False
    )

    for candidate_message in candidate_messages:
        if candidate_message is None:
            continue
        candidate_user_id = str(
            getattr(candidate_message, "user_id", "") or ""
        ).strip()
        if not candidate_user_id:
            continue
        candidate_text = _text_of(candidate_message)
        candidate_normalized = _normalized(candidate_text)
        if (
            anchor_repeat_short
            and anchor_continuation_hint
            and candidate_user_id == anchor_user_id
        ):
            if (
                candidate_normalized
                and candidate_normalized == anchor_normalized
            ):
                best_message = candidate_message
                best_score = normalized_relationships.get(
                    candidate_user_id, best_score
                )
                best_is_active = (
                    candidate_user_id in active_user_set
                    if active_user_set
                    else best_is_active
                )
                continue
        candidate_score = normalized_relationships.get(candidate_user_id)
        if candidate_score is None:
            candidate_score = 0.0
        candidate_is_active = (
            candidate_user_id in active_user_set if active_user_set else False
        )

        # 同一对象且文本更接近当前话头时，优先保持续接。
        if (
            candidate_user_id == anchor_user_id
            and anchor_normalized
            and candidate_normalized
        ):
            if (
                candidate_normalized == anchor_normalized
                or anchor_normalized in candidate_normalized
            ):
                best_message = candidate_message
                best_score = max(best_score, candidate_score)
                best_is_active = candidate_is_active
                continue

        if active_user_set:
            if candidate_is_active and not best_is_active:
                best_message = candidate_message
                best_score = candidate_score
                best_is_active = True
                continue
            if (
                candidate_is_active == best_is_active
                and candidate_score > best_score
            ):
                best_message = candidate_message
                best_score = candidate_score
                best_is_active = candidate_is_active
                continue
        elif candidate_score > best_score:
            best_message = candidate_message
            best_score = candidate_score

    return best_message

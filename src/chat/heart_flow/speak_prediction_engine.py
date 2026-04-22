import time
from dataclasses import dataclass
from typing import Any, Dict, List


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ratio(value: Any, default: float = 0.0) -> float:
    number = _safe_float(value, default)
    if number <= 1.0:
        return _clamp(number, 0.0, 1.0)
    return _clamp(number / 100.0, 0.0, 1.0)


@dataclass
class SpeakPredictionEngine:
    """正式发言预测引擎。

    保留当前系统已有状态来源，只把预测逻辑统一收口到一处。
    """

    def predict(
        self,
        *,
        channel_id: str,
        domains: Dict[str, Any],
        dashboard_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        resident = dashboard_snapshot.get("snapshot", {}) if dashboard_snapshot.get("available") else {}
        verdict = resident.get("reply_decision", {}) if isinstance(resident, dict) else {}
        social = resident.get("social_attitude", {}) if isinstance(resident, dict) else {}
        attention = resident.get("attention", {}) if isinstance(resident, dict) else {}
        vitality = resident.get("energy_reserve", {}) if isinstance(resident, dict) else {}
        safety = resident.get("safety_shield", {}) if isinstance(resident, dict) else {}

        emergence = domains.get("emergence_core", {}) if isinstance(domains, dict) else {}
        resource = domains.get("resource_ledger", {}) if isinstance(domains, dict) else {}
        group_climate = domains.get("group_climate", {}) if isinstance(domains, dict) else {}
        trauma_load = domains.get("trauma_load", {}) if isinstance(domains, dict) else {}
        relationship = domains.get("relationship_profile", {}) if isinstance(domains, dict) else {}
        counterparty = domains.get("counterparty_intent", {}) if isinstance(domains, dict) else {}
        boundary = domains.get("boundary_guard", {}) if isinstance(domains, dict) else {}
        tempo = domains.get("tempo", {}) if isinstance(domains, dict) else {}
        pending = domains.get("pending_response", {}) if isinstance(domains, dict) else {}
        flow_runtime = domains.get("flow_runtime", {}) if isinstance(domains, dict) else {}

        urgency_score = {
            "立即回复": 0.95,
            "尽快回复": 0.80,
            "可稍后回": 0.55,
            "跳过": 0.18,
            "不回复": 0.05,
        }.get(str(verdict.get("reply_urgency", "") or ""), 0.35)

        initiative_drive = _safe_float(emergence.get("initiative_drive", 0.0))
        boredom_load = _safe_float(emergence.get("boredom_load", 0.0))
        loneliness_load = _safe_float(emergence.get("loneliness_load", 0.0))
        environment_fatigue_load = _safe_float(emergence.get("environment_fatigue_load", 0.0))
        withdrawal_drive = _safe_float(emergence.get("withdrawal_drive", 0.0))

        energy_ratio = _safe_float(resource.get("energy_reserve_ratio", 0.5), 0.5)
        relationship_irritation = _safe_float(relationship.get("irritation_load", 0.0))
        resource_irritation = _safe_float(resource.get("irritation_load", 0.0))
        irritation_load = max(relationship_irritation, resource_irritation)
        scene_heat_score = _safe_float(group_climate.get("scene_heat_score", 0.0))
        active_user_count = _safe_float(group_climate.get("active_user_count", 0))
        topic_focus = list(group_climate.get("topic_focus", []) or [])

        trauma_score = max(
            _safe_float(trauma_load.get("trauma_load", 0.0)),
            _safe_float(relationship.get("trauma_load", 0.0)),
        )
        pressure_load = max(
            _safe_float(trauma_load.get("pressure_load", 0.0)),
            _safe_float(relationship.get("pressure_load", 0.0)),
        )
        cognitive_drag = _safe_float(trauma_load.get("cognitive_drag", 0.0))
        rapport_score = _safe_float(relationship.get("rapport_score", 0.0))
        trust_score = _safe_float(relationship.get("trust_score", 0.0))
        readiness_score = _safe_float(counterparty.get("readiness_score", 0.5), 0.5)
        blocked = bool(safety.get("blocked", False)) or bool(boundary.get("blocked", False))
        hostility_detected = bool(boundary.get("hostility_detected", False)) or bool(
            counterparty.get("hostility_detected", False)
        )
        cooling_down = bool(tempo.get("cooling_down", False))

        pending_active = bool(pending.get("pending_active", False))
        pending_seconds = _safe_float(pending.get("pending_seconds", 0.0))
        should_reengage = bool(pending.get("should_reengage", False))

        process_ratio_raw = attention.get("process_ratio", "50%")
        if isinstance(process_ratio_raw, str) and process_ratio_raw.endswith("%"):
            try:
                process_ratio = float(process_ratio_raw.rstrip("%")) / 100.0
            except Exception:
                process_ratio = 0.5
        else:
            process_ratio = _safe_float(process_ratio_raw, 0.5)

        social_willingness = _safe_float(social.get("willingness", 0.5), 0.5)
        vitality_percent = _safe_float(vitality.get("percent", 50), 50.0)

        boredom_drive = min(1.0, boredom_load)
        if boredom_load > 1.8:
            boredom_drive *= 0.45
        rapport_ratio = _clamp((rapport_score + 20.0) / 120.0, 0.0, 1.0)
        trust_ratio = _ratio(trust_score, 50.0)
        relationship_influence = (
            readiness_score * 0.08
            + rapport_ratio * 0.05
            + trust_ratio * 0.05
            - min(1.0, irritation_load / 100.0) * 0.10
            - min(1.0, pressure_load / 100.0) * 0.08
        )
        proactive_influence = (
            initiative_drive * 0.24
            + boredom_drive * 0.13
            + min(1.0, loneliness_load) * 0.12
            - min(1.0, withdrawal_drive) * 0.12
        )
        resource_influence = energy_ratio * 0.15 + process_ratio * 0.05
        scene_influence = min(1.0, scene_heat_score) * 0.08 + min(1.0, active_user_count / 8.0) * 0.04
        gate_influence = urgency_score * 0.22

        drive_score = (
            gate_influence
            + proactive_influence
            + resource_influence
            + social_willingness * 0.08
            + scene_influence
            + max(-0.18, relationship_influence)
        )
        if should_reengage:
            drive_score += 0.08
        if pending_active and pending_seconds > 0:
            drive_score += min(0.10, pending_seconds / 600.0)

        suppression_score = (
            min(1.0, environment_fatigue_load) * 0.14
            + min(1.0, withdrawal_drive) * 0.20
            + min(1.0, irritation_load / 100.0) * 0.12
            + min(1.0, pressure_load / 100.0) * 0.10
            + min(1.0, trauma_score / 10.0) * 0.08
            + min(1.0, cognitive_drag) * 0.10
            + max(0.0, 1.0 - process_ratio) * 0.10
        )
        if cooling_down:
            suppression_score += 0.08
        if hostility_detected:
            suppression_score += 0.12
        if blocked:
            suppression_score += 0.28

        probability = _clamp(drive_score - suppression_score + 0.14, 0.0, 1.0)

        if probability >= 0.85:
            eta_seconds = 20
        elif probability >= 0.70:
            eta_seconds = 90
        elif probability >= 0.50:
            eta_seconds = 240
        else:
            eta_seconds = 720

        last_plan = flow_runtime.get("last_reactive_plan", {}) if isinstance(flow_runtime, dict) else {}
        content_direction = ""
        if isinstance(last_plan, dict):
            content_direction = str(last_plan.get("content_plan", "") or "").strip()
        if not content_direction and topic_focus:
            content_direction = f"围绕「{topic_focus[0]}」继续参与"
        if not content_direction:
            content_direction = "根据当前群聊状态自然接话"

        if vitality_percent < 25 or energy_ratio < 0.25:
            reply_length = "简短"
        elif vitality_percent > 70 and probability > 0.72:
            reply_length = "中长"
        else:
            reply_length = "中等"

        tone = str(verdict.get("suggested_tone", "") or "正常回应")
        if probability < 0.3 and tone == "正常回应":
            tone = "克制观察"

        if probability >= 0.75:
            decision_label = "适合开口"
        elif probability >= 0.55:
            decision_label = "可以接话"
        elif probability >= 0.35:
            decision_label = "继续观察"
        else:
            decision_label = "暂时不聊"
        decision_reason = (
            f"开口驱动 {drive_score:.2f} / 抑制压力 {suppression_score:.2f}，"
            f"主动意愿 {initiative_drive:.2f}，关系就绪 {readiness_score:.2f}。"
        )

        driving_factors: List[str] = []
        suppressing_factors: List[str] = []

        if initiative_drive >= 0.72:
            driving_factors.append(f"主动意愿较高({initiative_drive:.2f})")
        if boredom_load >= 0.5:
            driving_factors.append(f"无聊驱动较高({boredom_load:.2f})")
        if loneliness_load >= 0.5:
            driving_factors.append(f"孤独驱动较高({loneliness_load:.2f})")
        if energy_ratio >= 0.65:
            driving_factors.append(f"资源储备充足({energy_ratio:.2f})")
        if should_reengage:
            driving_factors.append("当前阶段存在重新接入倾向")
        if scene_heat_score >= 0.55:
            driving_factors.append(f"群聊热度较高({scene_heat_score:.2f})")
        if rapport_score >= 20:
            driving_factors.append(f"当前用户好感较高({rapport_score:.1f})")
        if trust_score >= 65:
            driving_factors.append(f"信任基础较强({trust_score:.1f})")

        if environment_fatigue_load >= 0.35:
            suppressing_factors.append(f"环境疲劳较高({environment_fatigue_load:.2f})")
        if withdrawal_drive >= 0.25:
            suppressing_factors.append(f"撤离倾向偏高({withdrawal_drive:.2f})")
        if irritation_load >= 15:
            suppressing_factors.append(f"烦躁负担偏高({irritation_load:.1f})")
        if pressure_load >= 15:
            suppressing_factors.append(f"心理压力偏高({pressure_load:.1f})")
        if trauma_score >= 0.8:
            suppressing_factors.append(f"创伤负担已激活({trauma_score:.2f})")
        if trust_score > 0 and trust_score <= 25:
            suppressing_factors.append(f"信任基础较低({trust_score:.1f})")
        if hostility_detected:
            suppressing_factors.append("关系或边界信号检测到敌意")
        if cooling_down:
            suppressing_factors.append("当前处于冷却窗口")
        if blocked:
            suppressing_factors.append("安全防护阻断了发言倾向")
        if not driving_factors:
            driving_factors.append("当前没有强驱动，主要依靠自然群聊节奏")
        if not suppressing_factors:
            suppressing_factors.append("当前没有明显抑制因素")

        return {
            "channel_id": channel_id,
            "generated_at": time.time(),
            "speak_probability": round(probability, 3),
            "probability_percent": int(round(probability * 100)),
            "eta_seconds": eta_seconds,
            "eta_label": "立即" if eta_seconds <= 30 else f"约 {round(eta_seconds / 60)} 分钟后",
            "decision_label": decision_label,
            "decision_reason": decision_reason,
            "content_direction": content_direction,
            "reply_length": reply_length,
            "tone": tone,
            "drive_score": round(drive_score, 3),
            "suppression_score": round(suppression_score, 3),
            "gate_influence": round(gate_influence, 3),
            "proactive_influence": round(proactive_influence, 3),
            "relationship_influence": round(relationship_influence, 3),
            "resource_influence": round(resource_influence, 3),
            "scene_influence": round(scene_influence, 3),
            "driving_factors": driving_factors,
            "suppressing_factors": suppressing_factors,
        }


_prediction_engine: SpeakPredictionEngine | None = None


def get_speak_prediction_engine() -> SpeakPredictionEngine:
    global _prediction_engine
    if _prediction_engine is None:
        _prediction_engine = SpeakPredictionEngine()
    return _prediction_engine

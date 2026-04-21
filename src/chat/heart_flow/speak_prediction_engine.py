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
        verdict = resident.get("verdict", {}) if isinstance(resident, dict) else {}
        social = resident.get("social", {}) if isinstance(resident, dict) else {}
        attention = resident.get("attention", {}) if isinstance(resident, dict) else {}
        vitality = resident.get("vitality", {}) if isinstance(resident, dict) else {}
        safety = resident.get("safety", {}) if isinstance(resident, dict) else {}

        emergence = domains.get("emergence_core", {}) if isinstance(domains, dict) else {}
        resource = domains.get("resource_ledger", {}) if isinstance(domains, dict) else {}
        group_climate = domains.get("group_climate", {}) if isinstance(domains, dict) else {}
        trauma_load = domains.get("trauma_load", {}) if isinstance(domains, dict) else {}
        pending = domains.get("pending_response", {}) if isinstance(domains, dict) else {}
        flow_runtime = domains.get("flow_runtime", {}) if isinstance(domains, dict) else {}

        urgency_score = {
            "立即回复": 0.95,
            "尽快回复": 0.80,
            "可稍后回": 0.55,
            "跳过": 0.18,
            "不回复": 0.05,
        }.get(str(verdict.get("urgency", "") or ""), 0.35)

        initiative_drive = _safe_float(emergence.get("initiative_drive", 0.0))
        boredom_load = _safe_float(emergence.get("boredom_load", 0.0))
        loneliness_load = _safe_float(emergence.get("loneliness_load", 0.0))
        environment_fatigue_load = _safe_float(emergence.get("environment_fatigue_load", 0.0))
        withdrawal_drive = _safe_float(emergence.get("withdrawal_drive", 0.0))

        energy_ratio = _safe_float(resource.get("energy_reserve_ratio", 0.5), 0.5)
        irritation_load = _safe_float(resource.get("irritation_load", 0.0))
        scene_heat_score = _safe_float(group_climate.get("scene_heat_score", 0.0))
        active_user_count = _safe_float(group_climate.get("active_user_count", 0))
        topic_focus = list(group_climate.get("topic_focus", []) or [])

        trauma_score = _safe_float(trauma_load.get("trauma_load", 0.0))
        pressure_load = _safe_float(trauma_load.get("pressure_load", 0.0))
        cognitive_drag = _safe_float(trauma_load.get("cognitive_drag", 0.0))

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
        blocked = bool(safety.get("blocked", False))

        drive_score = (
            urgency_score * 0.22
            + initiative_drive * 0.24
            + min(1.0, boredom_load) * 0.08
            + min(1.0, loneliness_load) * 0.10
            + energy_ratio * 0.14
            + social_willingness * 0.10
            + min(1.0, scene_heat_score) * 0.07
            + min(1.0, active_user_count / 8.0) * 0.05
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
        if blocked:
            suppression_score += 0.28

        probability = _clamp(drive_score - suppression_score + 0.18, 0.0, 1.0)

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

        tone = str(verdict.get("tone", "") or "正常回应")
        if probability < 0.3 and tone == "正常回应":
            tone = "克制观察"

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
        if blocked:
            suppressing_factors.append("安全防护阻断了发言倾向")

        return {
            "channel_id": channel_id,
            "generated_at": time.time(),
            "speak_probability": round(probability, 3),
            "eta_seconds": eta_seconds,
            "eta_label": "立即" if eta_seconds <= 30 else f"约 {round(eta_seconds / 60)} 分钟后",
            "content_direction": content_direction,
            "reply_length": reply_length,
            "tone": tone,
            "driving_factors": driving_factors,
            "suppressing_factors": suppressing_factors,
        }


_prediction_engine: SpeakPredictionEngine | None = None


def get_speak_prediction_engine() -> SpeakPredictionEngine:
    global _prediction_engine
    if _prediction_engine is None:
        _prediction_engine = SpeakPredictionEngine()
    return _prediction_engine

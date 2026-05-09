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
        circadian = domains.get("circadian_rhythm", {}) if isinstance(domains, dict) else {}
        decision_runtime = domains.get("decision_runtime", {}) if isinstance(domains, dict) else {}
        execution_runtime = domains.get("execution_runtime", {}) if isinstance(domains, dict) else {}
        memory_stack = domains.get("memory_stack", {}) if isinstance(domains, dict) else {}
        autonomy_runtime = domains.get("autonomy_runtime", {}) if isinstance(domains, dict) else {}
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
        night_phase = str(circadian.get("phase", "awake") or "awake")
        night_phase_label = str(circadian.get("phase_label", "清醒") or "清醒")
        is_sleeping = bool(circadian.get("is_sleeping", False))
        is_burnthrough = bool(circadian.get("is_burnthrough", False))
        drowsiness_value = _safe_float(circadian.get("drowsiness_value", 0.0))
        sleep_debt = _safe_float(circadian.get("sleep_debt", 0.0))
        overnight_pressure = _safe_float(circadian.get("overnight_pressure", 0.0))
        sleep_reserve = _safe_float(circadian.get("sleep_reserve", 100.0), 100.0)
        response_suppression = _safe_float(circadian.get("response_suppression", 0.0))
        remaining_sleep_replies = _safe_float(circadian.get("remaining_sleep_replies", 1.0), 1.0)

        pending_active = bool(pending.get("pending_active", False))
        pending_seconds = _safe_float(pending.get("pending_seconds", 0.0))
        should_reengage = bool(pending.get("should_reengage", False))
        overload = memory_stack.get("overload", {}) if isinstance(memory_stack.get("overload"), dict) else {}
        memory_load_ratio = _safe_float(overload.get("load_ratio", 0.0))
        amnesia_pressure = _safe_float(overload.get("amnesia_pressure", 0.0))
        memory_emergency = bool(overload.get("emergency_needed", False))
        memoir_phase = str(memory_stack.get("memoir_phase", "open") or "open")
        memoir_timeouts = _safe_float(memory_stack.get("memoir_consecutive_timeouts", 0.0))
        reactivation = memory_stack.get("reactivation", {}) if isinstance(memory_stack.get("reactivation"), dict) else {}
        reactivation_hits = sum(_safe_float(value, 0.0) for value in reactivation.values())
        active_intentions = autonomy_runtime.get("active_intentions", [])
        if not isinstance(active_intentions, list):
            active_intentions = []
        intention_drive = _safe_float(autonomy_runtime.get("intention_drive", 0.0))
        max_intention_urgency = _safe_float(autonomy_runtime.get("max_intention_urgency", 0.0))
        pending_proactive_events = _safe_float(autonomy_runtime.get("pending_proactive_events", 0.0))
        background_event = autonomy_runtime.get("background_event", {})
        if not isinstance(background_event, dict):
            background_event = {}
        background_event_type = str(background_event.get("type", "") or "")
        background_event_priority = _safe_float(background_event.get("priority", 0.0))

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
            + min(1.0, intention_drive) * 0.10
            + min(1.0, max_intention_urgency) * 0.05
        )
        resource_influence = energy_ratio * 0.15 + process_ratio * 0.05
        scene_influence = min(1.0, scene_heat_score) * 0.08 + min(1.0, active_user_count / 8.0) * 0.04
        gate_influence = urgency_score * 0.22
        circadian_influence = 0.0
        if night_phase in {"night_active", "social_night", "midnight_reflect"}:
            circadian_influence += 0.04
        if night_phase == "dawn_recover":
            circadian_influence += 0.03
        if is_burnthrough:
            circadian_influence -= 0.20
        if night_phase == "light_sleep":
            circadian_influence -= 0.28
        if night_phase == "deep_sleep":
            circadian_influence -= 0.45
        elif is_sleeping:
            circadian_influence -= 0.30
        if drowsiness_value >= 60:
            circadian_influence -= min(0.12, (drowsiness_value - 60.0) / 100.0)
        if sleep_debt >= 0.45:
            circadian_influence -= min(0.10, (sleep_debt - 0.45) * 0.18)
        if overnight_pressure >= 55:
            circadian_influence -= min(0.08, (overnight_pressure - 55.0) / 100.0)
        if sleep_reserve <= 30:
            circadian_influence -= min(0.10, (30.0 - sleep_reserve) / 100.0)
        if remaining_sleep_replies <= 0 and is_sleeping:
            circadian_influence -= 0.10

        drive_score = (
            gate_influence
            + proactive_influence
            + resource_influence
            + social_willingness * 0.08
            + scene_influence
            + max(-0.18, relationship_influence)
        )
        if circadian_influence > 0:
            drive_score += circadian_influence
        if should_reengage:
            drive_score += 0.08
        if pending_active and pending_seconds > 0:
            drive_score += min(0.10, pending_seconds / 600.0)
        if memoir_phase == "anticipating" and memoir_timeouts < 3:
            drive_score += 0.07
        if reactivation_hits > 0:
            drive_score += min(0.06, reactivation_hits * 0.02)
        if background_event_type:
            drive_score += min(0.06, 0.02 + background_event_priority * 0.04)
        if pending_proactive_events > 0:
            drive_score += min(0.05, pending_proactive_events * 0.015)

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
        if circadian_influence < 0:
            suppression_score += abs(circadian_influence)
        if response_suppression > 0:
            suppression_score += min(0.16, response_suppression * 0.16)
        if hostility_detected:
            suppression_score += 0.12
        if blocked:
            suppression_score += 0.28
        if memory_emergency:
            suppression_score += 0.20
        elif memory_load_ratio >= 0.8 or amnesia_pressure >= 0.6:
            suppression_score += 0.12
        elif memory_load_ratio >= 0.65 or amnesia_pressure >= 0.45:
            suppression_score += 0.06
        if memoir_timeouts >= 3:
            suppression_score += min(0.12, memoir_timeouts * 0.025)

        probability = _clamp(drive_score - suppression_score + 0.14, 0.0, 1.0)
        execution_has_verdict = bool(execution_runtime.get("verdict_id"))
        decision_has_verdict = bool(decision_runtime.get("verdict_id"))
        runtime_verdict = execution_runtime if execution_has_verdict else decision_runtime
        runtime_has_verdict = bool(runtime_verdict.get("verdict_id"))
        if execution_has_verdict:
            runtime_should_reply = bool(
                execution_runtime.get("reply_sent", execution_runtime.get("should_act", False))
            )
            runtime_stage = str(execution_runtime.get("execution_stage", "") or "")
            runtime_action = str(execution_runtime.get("final_action", "") or "")
            runtime_reason = str(execution_runtime.get("execution_reason", "") or "")
        elif decision_has_verdict:
            runtime_should_reply = bool(decision_runtime.get("should_reply", False))
            runtime_stage = "pre_execution"
            runtime_action = str(decision_runtime.get("next_action", "") or "")
            runtime_reason = str(decision_runtime.get("decision_reason", "") or "")
        else:
            runtime_should_reply = probability >= 0.55
            runtime_stage = ""
            runtime_action = ""
            runtime_reason = ""
        runtime_model_path = str(runtime_verdict.get("model_path", "") or "")
        runtime_complexity_label = str(runtime_verdict.get("complexity_label", "") or "")
        if execution_has_verdict:
            runtime_sync_state = "execution"
            runtime_sync_label = "已同步最终执行"
        elif decision_has_verdict:
            runtime_sync_state = "pre_execution"
            runtime_sync_label = "仅同步前置裁定"
        else:
            runtime_sync_state = "baseline"
            runtime_sync_label = "仅按基线估算"
        if runtime_has_verdict:
            if runtime_should_reply:
                probability = max(probability, 0.56)
            else:
                probability = min(probability, 0.34)

        generated_at = time.time()
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
        content_source = "baseline"
        content_source_label = "基线方向"
        if isinstance(last_plan, dict):
            content_direction = str(last_plan.get("content_plan", "") or "").strip()
            if content_direction:
                content_source = "reactive_plan"
                content_source_label = "已同步内容规划"
        if not content_direction and topic_focus:
            content_direction = f"围绕「{topic_focus[0]}」继续参与"
            content_source = "topic_focus"
            content_source_label = "按话题焦点估算"
        if not content_direction:
            content_direction = "根据当前会话状态自然接话"

        if memory_emergency or memory_load_ratio >= 0.8 or amnesia_pressure >= 0.6:
            reply_length = "简短"
        elif vitality_percent < 25 or energy_ratio < 0.25:
            reply_length = "简短"
        elif vitality_percent > 70 and probability > 0.72:
            reply_length = "中长"
        else:
            reply_length = "中等"

        tone = str(verdict.get("suggested_tone", "") or "正常回应")
        if probability < 0.3 and tone == "正常回应":
            tone = "克制观察"

        if runtime_has_verdict:
            decision_label = "可以接话" if runtime_should_reply else "暂时不聊"
        elif probability >= 0.75:
            decision_label = "适合开口"
        elif probability >= 0.55:
            decision_label = "可以接话"
        elif probability >= 0.35:
            decision_label = "继续观察"
        else:
            decision_label = "暂时不聊"
        if runtime_has_verdict:
            if runtime_should_reply and runtime_action == "reply":
                decision_label = "可以接话"
            elif runtime_should_reply:
                decision_label = "准备行动"
            else:
                decision_label = "暂时不聊"
        decision_reason = (
            f"开口驱动 {drive_score:.2f} / 抑制压力 {suppression_score:.2f}，"
            f"主动意愿 {initiative_drive:.2f}，关系就绪 {readiness_score:.2f}，"
            f"昼夜状态 {night_phase_label}。"
        )
        if execution_has_verdict:
            runtime_reason = str(runtime_reason or "")
            decision_reason = f"统一裁定：{runtime_reason or decision_label}；{decision_reason}"
        elif decision_has_verdict:
            runtime_reason = str(runtime_reason or "")
            decision_reason = (
                f"当前仍处于前置裁定，尚未进入最终执行：{runtime_reason or decision_label}；"
                f"{decision_reason}"
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
            driving_factors.append(f"会话热度较高({scene_heat_score:.2f})")
        if rapport_score >= 20:
            driving_factors.append(f"当前用户好感较高({rapport_score:.1f})")
        if trust_score >= 65:
            driving_factors.append(f"信任基础较强({trust_score:.1f})")
        if circadian_influence > 0:
            driving_factors.append(f"昼夜状态有利于开口({night_phase_label})")
        if memoir_phase == "anticipating" and memoir_timeouts < 3:
            driving_factors.append("回忆录正在等待回应，适合轻量跟进")
        if reactivation_hits > 0:
            driving_factors.append("近期有记忆再激活线索")
        if intention_drive >= 0.35 or active_intentions:
            driving_factors.append(f"主动意图池存在驱动({intention_drive:.2f})")
        if background_event_type:
            driving_factors.append(f"后台事件排队({background_event_type})")

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
        if is_sleeping:
            suppressing_factors.append(f"当前处于{night_phase_label}，优先保持睡眠")
        elif is_burnthrough:
            suppressing_factors.append("当前处于熬穿状态，回复质量和稳定性下降")
        elif drowsiness_value >= 60:
            suppressing_factors.append(f"困意较高({drowsiness_value:.0f})")
        if sleep_debt >= 0.45:
            suppressing_factors.append(f"睡眠债偏高({_ratio(sleep_debt):.0%})")
        if sleep_reserve <= 30:
            suppressing_factors.append(f"睡眠储备偏低({sleep_reserve:.0f})")
        if blocked:
            suppressing_factors.append("安全防护阻断了发言倾向")
        if memory_emergency:
            suppressing_factors.append("记忆过载触发紧急压力，压低主动性和回复长度")
        elif memory_load_ratio >= 0.8 or amnesia_pressure >= 0.6:
            suppressing_factors.append(
                f"记忆负载偏高(load={memory_load_ratio:.2f}, 遗忘压力={amnesia_pressure:.2f})"
            )
        if memoir_timeouts >= 3:
            suppressing_factors.append(f"回忆录连续超时({int(memoir_timeouts)}次)，抑制继续追问")
        if not driving_factors:
            driving_factors.append("当前没有强驱动，主要依靠自然会话节奏")
        if not suppressing_factors:
            suppressing_factors.append("当前没有明显抑制因素")

        return {
            "channel_id": channel_id,
            "generated_at": time.time(),
            "speak_probability": round(probability, 3),
            "probability_percent": round(probability * 100, 1),
            "eta_seconds": eta_seconds,
            "deadline_at": round(generated_at + eta_seconds, 3),
            "eta_label": "立即" if eta_seconds <= 30 else f"约 {round(eta_seconds / 60)} 分钟后",
            "shared_verdict_id": str(runtime_verdict.get("verdict_id", "") or ""),
            "model_path": runtime_model_path or "未裁定",
            "should_reply": runtime_should_reply if runtime_has_verdict else probability >= 0.55,
            "complexity_label": runtime_complexity_label or "普通",
            "execution_stage": runtime_stage,
            "execution_action": runtime_action,
            "runtime_sync_state": runtime_sync_state,
            "runtime_sync_label": runtime_sync_label,
            "content_source": content_source,
            "content_source_label": content_source_label,
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
            "circadian_influence": round(circadian_influence, 3),
            "driving_factors": driving_factors,
            "suppressing_factors": suppressing_factors,
        }


_prediction_engine: SpeakPredictionEngine | None = None


def get_speak_prediction_engine() -> SpeakPredictionEngine:
    global _prediction_engine
    if _prediction_engine is None:
        _prediction_engine = SpeakPredictionEngine()
    return _prediction_engine

# ruff: noqa: F401
import asyncio
import math
import random
import re
import time
from dataclasses import dataclass, field, is_dataclass, replace as dataclass_replace
import datetime
from collections import Counter, defaultdict, deque
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from src.chat.utils.timer_calculator import Timer
from src.chat.replyer.context_block_builder import build_reply_context_block
from src.chat.heart_flow.heartFC_chat import HeartFChatting
from src.chat.heart_flow.energy_manager import EnergyChainDimension
from src.chat.heart_flow.heartfc_thresholds import get_heartfc_thresholds
from src.core.watch_state_machine import WatchLevel
from src.core.group_pattern_detector import GroupPattern
from src.core.group_scene_state import AtmosphereType
from src.core.unified_planner import PlanningDecision, ActionType
from src.config.config import global_config
from src.common.logger import get_logger
from src.chat.heart_flow.frequency_control import frequency_control_manager
from src.plugin_system.apis import database_api, message_api, send_api
from src.chat.utils.utils import is_bot_self
from src.common.data_models.heartflow_models import FlowPhase, UnifiedFlowSnapshot
from src.chat.heart_flow.enhanced_modules.shared_runtime import (
    logger,
    _rng,
    _rt_float,
    _parallel_stage_timeout,
    _llm_upgrade_timeout,
    _TICK_FLOOR_SEC,
    _DORMANT_POLL_SEC,
    _PERCEPTION_COOLDOWN_SEC,
    _VOICE_COOLDOWN_SEC,
    _ENERGY_DRAIN_FLOOR,
    _POST_MESSAGE_RETRY_SEC,
    _WATCH_LEVEL_BY_RANK,
    BehaviorGovernorVerdict,
    RestGovernorVerdict,
    ModelGovernorVerdict,
)

if TYPE_CHECKING:
    from src.chat.heart_flow.llm_autonomous_planner import (
        AutonomousDecision,
        EnvironmentSnapshot,
    )
    from src.chat.proactive.proactive_decider import ProactiveDecision

class RuntimeStateTraceMixin:
    def _normalized_night_phase_value(self) -> str:
        """统一夜间阶段命名，供跨引擎消费（小写NCS语义）。"""
        _np = getattr(self, "_cached_night_phase", None)
        if _np is None:
            return ""
        raw = str(getattr(_np, "value", "") or getattr(_np, "name", "") or "").strip()
        if not raw:
            return ""
        up = raw.upper()
        mapping = {
            "DEEP_VALLEY": "deep_sleep",
            "DROWSY": "drowsy",
            "ACTIVE_TWILIGHT": "night_active",
            "DAWN_RECOVERY": "dawn_recover",
            "BURNED_OUT": "burned_out",
        }
        return mapping.get(up, raw.lower())

    def _derive_night_expression_style(self, night_meta: Dict[str, Any]) -> str:
        """从夜间三维状态提炼表达风格标签。"""
        if not isinstance(night_meta, dict) or not night_meta:
            return "normal"
        if bool(night_meta.get("is_burnthrough", False)):
            return "burnthrough"
        _body = str(night_meta.get("body_state_tag", "") or "").strip().lower()
        _mood = str(night_meta.get("monologue_mood_hint", "") or "").strip()
        try:
            _suppression = float(night_meta.get("response_suppression", 0.0) or 0.0)
        except Exception:
            _suppression = 0.0
        _phase = str(night_meta.get("phase", "") or "").strip().upper()
        if _body in {"irritated_woken", "slightly_grumpy"} or any(
            token in _mood for token in ("烦", "恼", "不爽", "暴躁")
        ):
            return "irritable"
        if _body in {"stubborn_lazy", "reluctant_wake"}:
            return "stubborn"
        if _body in {"deep_half_asleep", "light_half_asleep", "frequently_yawning", "occasionally_yawning"}:
            return "drowsy"
        if _suppression >= 0.72:
            return "drowsy"
        if _phase in {"ACTIVE_TWILIGHT", "DAWN_RECOVERY"}:
            return "soft_night"
        return "normal"

    def _watch_level_value(self, default: str = "") -> str:
        _wl = getattr(self, "_cached_watch_level", None)
        if _wl is None:
            return default
        return str(getattr(_wl, "value", _wl) or default)

    def _roll_hourly_reply_window(self, now: Optional[float] = None) -> float:
        _now = float(now or time.time())
        if self._hourly_window_start == 0.0 or (_now - self._hourly_window_start) >= 3600.0:
            self._hourly_window_start = _now
            self._hourly_reply_count = 0
            self._hourly_proactive_reply_count = 0
        return _now

    def _current_hourly_reply_count(self, now: Optional[float] = None) -> int:
        self._roll_hourly_reply_window(now)
        if self._hourly_window_start == 0.0:
            return 0
        return max(0, int(self._hourly_reply_count or 0))

    def _current_hourly_proactive_reply_count(self, now: Optional[float] = None) -> int:
        self._roll_hourly_reply_window(now)
        if self._hourly_window_start == 0.0:
            return 0
        return max(0, int(self._hourly_proactive_reply_count or 0))

    def _reset_decision_trace(self) -> None:
        self._last_decision_trace = {
            "semantic_route_summary": {},
            "narration_should_reply": None,
            "narration_action_intent": "",
            "narration_reply_strategy": "",
            "legacy_gate": "",
            "autonomy_guard_reason": "",
            "voice_action": "",
            "model_should_reply": None,
            "final_decision": "",
            "winner_layer": "",
        }

    def _update_decision_trace(self, **updates: Any) -> None:
        if not isinstance(getattr(self, "_last_decision_trace", None), dict):
            self._reset_decision_trace()
        for _key, _value in updates.items():
            if _key == "semantic_route_summary":
                self._last_decision_trace[_key] = dict(_value or {})
            elif _key in self._last_decision_trace:
                self._last_decision_trace[_key] = _value

    def _mark_decision_winner(self, layer: str) -> None:
        _valid_layers = {
            "gateway_block",
            "gateway_force",
            "autonomy_guard",
            "voice_action",
            "model_takeover",
            "algo_reply",
            "algo_upgrade",
            "final_skip",
        }
        if layer in _valid_layers:
            self._update_decision_trace(winner_layer=layer)

    def _decision_trace_brief(self) -> str:
        _trace = getattr(self, "_last_decision_trace", None) or {}
        _parts: List[str] = []
        _route_summary = _trace.get("semantic_route_summary") or {}
        if _route_summary:
            _route_text = ",".join(f"{k}:{v}" for k, v in sorted(_route_summary.items()))
            if _route_text:
                _parts.append(f"route={_route_text}")
        _narr_should = _trace.get("narration_should_reply", None)
        if _narr_should is not None:
            _parts.append(f"narr={'reply' if _narr_should else 'hold'}")
        if _trace.get("legacy_gate"):
            _parts.append(f"gate={_trace['legacy_gate']}")
        if _trace.get("voice_action"):
            _parts.append(f"voice={_trace['voice_action']}")
        _model_should = _trace.get("model_should_reply", None)
        if _model_should is not None:
            _parts.append(f"model={'reply' if _model_should else 'skip'}")
        if _trace.get("winner_layer"):
            _parts.append(f"winner={_trace['winner_layer']}")
        return " | ".join(_parts[:6])

    def _sync_interest_level_from_desire(self) -> float:
        """把当前轮 desire 及时折算成兴趣强度，避免回复消耗误用上一轮残留值。"""
        _interest_level = max(
            0.1,
            min(1.0, float(getattr(self, "_last_desire_level", 5) or 5) / 10.0),
        )
        self._last_interest_level = _interest_level
        return _interest_level

    def _metric_has_signal(self, value: Any, threshold: float = 0.01) -> bool:
        """统一判断一个数值型关系信号是否值得注入，负值不能再被真值判断吞掉。"""
        try:
            return abs(float(value or 0.0)) >= threshold
        except Exception:
            return False

    def _safe_relation_float(
        self,
        relation_snapshot: Optional[Dict[str, Any]],
        *keys: str,
        default: float = 0.0,
    ) -> float:
        """从关系快照中按优先级读取数值，兼容 legacy ingress。"""
        if not isinstance(relation_snapshot, dict):
            return float(default)
        for key in keys:
            if key not in relation_snapshot:
                continue
            value = relation_snapshot.get(key)
            if value is None or value == "":
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(default)

    def _normalize_relation_snapshot(
        self,
        relation_snapshot: Optional[Dict[str, Any]],
        *,
        include_compat: bool = False,
    ) -> Dict[str, Any]:
        """统一把关系快照收口到 canonical contract，compat 字段只在边界镜像。"""
        source = dict(relation_snapshot or {}) if isinstance(relation_snapshot, dict) else {}
        normalized = dict(source)
        normalized["social_value"] = self._safe_relation_float(source, "social_value", "favorability", default=0.0)
        normalized["affection"] = self._safe_relation_float(source, "affection", default=0.0)
        normalized["trust_value"] = self._safe_relation_float(source, "trust_value", "trust_score", "trust", default=0.0)
        normalized["annoyance_value"] = self._safe_relation_float(
            source,
            "annoyance_value",
            "annoyance",
            default=0.0,
        )
        normalized["psychological_pressure"] = self._safe_relation_float(
            source,
            "psychological_pressure",
            "pressure",
            default=0.0,
        )
        normalized["trauma_score"] = self._safe_relation_float(source, "trauma_score", default=0.0)
        try:
            normalized["relationship_level"] = int(source.get("relationship_level", normalized.get("relationship_level", 2)) or 2)
        except Exception:
            normalized["relationship_level"] = 2
        try:
            normalized["interaction_count"] = int(source.get("interaction_count", normalized.get("interaction_count", 0)) or 0)
        except Exception:
            normalized["interaction_count"] = 0

        if include_compat:
            normalized["favorability"] = normalized["social_value"]
            normalized["trust"] = normalized["trust_value"]
            normalized["trust_score"] = normalized["trust_value"]
            normalized["annoyance"] = normalized["annoyance_value"]
            normalized["pressure"] = normalized["psychological_pressure"]
        else:
            normalized.pop("favorability", None)
            normalized.pop("trust", None)
            normalized.pop("trust_score", None)
            normalized.pop("annoyance", None)
            normalized.pop("pressure", None)
        return normalized

    def _prime_external_proactive_trace(self, source: str, desire_level: Optional[float] = None) -> None:
        """主循环外主动回复也要有独立 trace，避免沿用上一轮旁路/独白状态。"""
        self._reset_decision_trace()
        if desire_level is not None:
            self._last_desire_level = float(desire_level or 5.0)
        self._update_decision_trace(
            semantic_route_summary={"source": source, "proactive": True},
            narration_should_reply=True,
            narration_action_intent=source,
            narration_reply_strategy="proactive",
            voice_action="proactive",
            model_should_reply=True,
            final_decision="reply",
        )
        self._mark_decision_winner("algo_reply")

    def _admin_force_safety_guard_reason(
        self,
        gateway_result: Optional[Dict[str, Any]] = None,
        relation_result: Optional[Dict[str, Any]] = None,
        repetition_signal: Optional[Dict[str, Any]] = None,
        harassment_signal: Optional[Dict[str, Any]] = None,
    ) -> str:
        _reason_text = str((gateway_result or {}).get("reason", "") or "")
        _behavior_view = dict((relation_result or {}).get("behavior_signal") or {})
        _behavior_category = str(_behavior_view.get("category", "") or "")
        _behavior_severity = float(_behavior_view.get("severity", 0.0) or 0.0)
        if bool((repetition_signal or {}).get("detected", False)) and bool(
            (repetition_signal or {}).get("low_info_cluster", False)
        ):
            return str((repetition_signal or {}).get("reason", "") or "低信息重复施压")
        if bool((harassment_signal or {}).get("detected", False)):
            return str((harassment_signal or {}).get("reason", "") or "骚扰输入")
        if _behavior_category == "harassing":
            return f"behavior:{_behavior_category}"
        if _behavior_category == "hostile" and _behavior_severity >= 0.85:
            return f"behavior:{_behavior_category}@{_behavior_severity:.2f}"
        for _hard_reason in ("情绪拒绝", "讨厌度屏蔽", "极端创伤"):
            if _hard_reason in _reason_text:
                return _reason_text or _hard_reason
        return ""

    @staticmethod
    def _resolve_admin_force_gate(original_gate: Any) -> str:
        """管理员强制唤醒时保留 force_reply 语义，其他门控统一放行为 allow。"""
        gate = str(original_gate or "allow")
        return "force_reply" if gate == "force_reply" else "allow"


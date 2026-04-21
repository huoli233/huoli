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

class RuntimeSensoryPipelineMixin:
    def _integrate_presence_metabolism(self, eagerness_val: float, now: float) -> None:
        """从新维度系统统一采集存在态+代谢约束（替代旧 PresenceCore+MetabolismEngine）"""
        thresholds = get_heartfc_thresholds()
        _has_interest_topic = False
        _has_close_friend = False
        try:
            _gs = self._orch.get("scene_state")
            _topics = _gs.active_topic_slots(limit=3)
            for _t in _topics:
                if _t.heat.value in ("hot", "warm"):
                    _has_interest_topic = True
                    break
            _rel_snap = getattr(self, "_last_relation_snapshot", None) or {}
            if float(_rel_snap.get("affection", 0) or 0) > 60:
                _has_close_friend = True
        except Exception as _e:
            logger.debug(f"{self.log_prefix} 亲密朋友检测异常: {_e}")
        try:
            _d6 = EnergyChainDimension.get_instance()
            _d6_state = _d6._ensure_channel(self.stream_id)
            _d6_vote = _d6.vote(type("VC", (), {"channel_id": self.stream_id, "user_id": ""})())
            _combined_ratio = _d6_state.combined_ratio()
            _chat_ratio = float(_d6_state.chat_ratio())
            _thinking_ratio = float(_d6_state.thinking_ratio())
            _boredom_level = max(0.0, (1.0 - _combined_ratio) * 100.0)
            _loafing_level = max(0.0, min(100.0, (1.0 - _d6_state.activity_level / 100.0) * 80.0))
            _boredom_drive = max(0.0, min(1.0, _boredom_level / 100.0))
            _loafing_sup = max(0.0, min(1.0, _loafing_level / 100.0))
            _watch_will = 0.5
            if 0.10 < _boredom_drive < 0.32:
                _watch_will = min(0.85, 0.5 + min(0.12, (_boredom_drive - 0.10) * 0.4))
            elif _boredom_drive > 0.38:
                _watch_will = max(0.05, 0.5 - min(0.30, (_boredom_drive - 0.38) * 0.5))
            _social_will = (
                max(0.05, 0.5 - min(0.35, max(0.0, (_loafing_sup - thresholds.loafing_social_inhibit)) * 0.6))
                if _loafing_sup > thresholds.loafing_social_inhibit
                else 0.5
            )
            _curiosity = 0.3 + (1.0 - _combined_ratio) * 0.3 if _has_interest_topic else 0.25
            _avoidance = max(0.0, min(0.8, _loafing_level / 100.0 * 0.5))
            _quiet_pref = max(0.0, min(0.8, (1.0 - _d6_state.social_value / 100.0) * 0.4 + 0.15))
            _fatigue_acc = max(0.0, float(getattr(_d6_state, "total_consumed_today", 0.0) or 0.0))
            _annoyance_level = float(getattr(_d6_state, "annoyance_level", 0.0) or 0.0)
            _reply_hint = (
                "简短"
                if _combined_ratio < 0.25 or _annoyance_level > 60.0
                else ("中等" if _combined_ratio < 0.55 else "可以长一点")
            )
            _energy_gate_open = bool(_combined_ratio > 0.12 and _d6_state.activity_level > 8.0)
            _is_perfunctory = bool(_combined_ratio < 0.22 or _annoyance_level > 70.0)
            _recovery_rate = max(0.2, min(1.0, 1.0 - min(0.75, _fatigue_acc / 100.0)))
            _sleep_debt = float((getattr(self, "_cached_night_summary", {}) or {}).get("sleep_debt", 0.0) or 0.0)
            _sleep_debt_modifier = 1.0 + min(0.8, _sleep_debt)
            _proactive_drive = max(
                0.05,
                min(0.95, 0.18 + _boredom_drive * 0.35 + max(0.0, _social_will - 0.5) * 0.3),
            )
            _mood_valence = max(0.0, min(1.0, 0.5 + (_d6_state.social_value / 200.0) - (_annoyance_level / 150.0)))
            _mood_arousal = max(0.0, min(1.0, 0.25 + (_d6_state.activity_level / 140.0) + min(0.15, _boredom_drive * 0.2)))
            self._cached_metabolism_constraints = {
                "boredom": round(_boredom_level, 2),
                "boredom_level": round(_boredom_drive, 3),
                "loafing": round(_loafing_level, 2),
                "loafing_level": round(_loafing_level, 2),
                "chat_fuel": round(_d6_state.chat_pool, 1),
                "thinking_fuel": round(_d6_state.thinking_value, 1),
                "chat_value": round(_d6_state.chat_pool, 1),
                "activity_gauge": round(_d6_state.activity_level, 1),
                "social_gauge": round(_d6_state.social_value, 1),
                "fatigue_accumulator": round(_fatigue_acc, 1),
                "boredom_watch_drive": round(_boredom_drive, 3),
                "loafing_suppression": round(_loafing_sup, 3),
                "energy_ratio": round(_combined_ratio, 3),
                "energy_suppression": round(max(0.0, min(1.0, 1.0 - _combined_ratio)), 3),
                "reply_length_hint": _reply_hint,
                "energy_gate_open": _energy_gate_open,
                "is_perfunctory": _is_perfunctory,
                "recovery_rate": round(_recovery_rate, 3),
                "sleep_debt_modifier": round(_sleep_debt_modifier, 3),
                "proactive_drive": round(_proactive_drive, 3),
                "night_mode": _d6_vote.night_mode,
                "energy_stage": str(getattr(_d6_vote, "energy_stage", "ADEQUATE").value)
                if hasattr(getattr(_d6_vote, "energy_stage", None), "value")
                else "ADEQUATE",
                "annoyance": round(_annoyance_level, 1),
                "annoyance_accumulated": round(_annoyance_level, 1),
                "chain_count": _d6_state.chain_count,
            }
            self._chat_energy = float(_d6_state.chat_pool)
            self._thinking_energy = float(_d6_state.thinking_value)

            class _PresenceStateCompat:
                watch_willingness: float = _watch_will
                social_willingness: float = _social_will
                curiosity_level: float = _curiosity
                avoidance_tendency: float = _avoidance
                quiet_preference: float = _quiet_pref
                outward_attention: float = max(0.05, 1.0 - _avoidance)

            self._cached_presence_state = _PresenceStateCompat()

            class _MetabolismStateCompat:
                chat_fuel: float = _d6_state.chat_pool
                thinking_fuel: float = _d6_state.thinking_value
                chat_energy_ratio: float = _chat_ratio
                energy_ratio: float = _combined_ratio
                thinking_ratio: float = _thinking_ratio
                activity_gauge: float = _d6_state.activity_level
                social_gauge: float = _d6_state.social_value
                mood_valence: float = _mood_valence
                mood_arousal: float = _mood_arousal
                stress_accumulation: float = _fatigue_acc
                trauma_score: float = float(getattr(self, "_cached_trauma_score", 0.0) or 0.0)
                channel_annoyance: float = _annoyance_level
                boredom_level: float = _boredom_level
                loafing_level: float = _loafing_level
                fatigue_level: float = _fatigue_acc
                fatigue_accumulator: float = _fatigue_acc
                consecutive_active_minutes: float = 0.0
                total_consumed_today: float = _d6_state.total_consumed_today

            self._cached_metabolism_state = _MetabolismStateCompat()
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 新维度状态采集异常: {exc}")

    def _integrate_message_routing(self, incoming_batch: List) -> None:
        """语义路由分类 + 时间流注册 + 活跃名册 + 多模态预算"""
        try:
            _router = self._orch.get("semantic_router")
            self._cached_semantic_routes = _router.classify_batch(incoming_batch)
            self._cached_route_summary = _router.summarize_batch(self._cached_semantic_routes)
        except Exception as exc:
            self._cached_semantic_routes = []
            self._cached_route_summary = {}
            logger.debug(f"{self.log_prefix} 语义路由分类异常: {exc}")
        try:
            self._orch.get("subjective_time").register_batch(incoming_batch)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 时间流注册异常: {exc}")
        try:
            _roster = self._orch.get("participant_roster")
            for i, _m in enumerate(incoming_batch):
                _uid = str(getattr(_m, "user_id", "") or "").strip()
                _route = self._cached_semantic_routes[i] if i < len(self._cached_semantic_routes) else None
                if _route is not None:
                    try:
                        _route_at = bool(getattr(_route, "is_at_bot", False))
                        _route_quote = bool(getattr(_route, "is_quote_to_bot", False))
                        _m.is_at_bot = bool(getattr(_m, "is_at_bot", False)) or _route_at
                        _m.is_quote_to_bot = bool(getattr(_m, "is_quote_to_bot", False)) or _route_quote
                        _m.is_reply_to_bot = bool(getattr(_m, "is_reply_to_bot", False)) or _route_quote
                    except Exception as _route_attr_exc:
                        logger.debug(f"{self.log_prefix} 语义路由标记写回异常: {_route_attr_exc}")
                if not self._is_human_message_obj(_m):
                    continue
                _text = str(getattr(_m, "processed_plain_text", "") or "")[:50]
                _route_type = "plain_text"
                if _route is not None:
                    _route_type = getattr(getattr(_route, "category", None), "value", "") or str(
                        getattr(_route, "category", "plain_text")
                    )
                    _emotion = 0.4 if any(mark in _text for mark in ("！", "!", "？", "?")) else 0.0
                    _message_quality = 0.7 if len(_text) >= 16 else 0.5
                    _topic_relevance = 0.7 if len(_text) >= 8 else 0.4
                    _bot_interacted = bool(
                        _route and (getattr(_route, "is_at_bot", False) or getattr(_route, "is_quote_to_bot", False))
                    )
                    _roster.reset_consecutive(_uid)
                    _roster.touch(
                        _uid,
                        user_name=str(getattr(_m, "user_name", "") or getattr(_m, "nickname", "") or ""),
                        text_hint=_text,
                        bot_interacted=_bot_interacted,
                        message_quality=_message_quality,
                        topic_relevance=_topic_relevance,
                        emotional_content=_emotion,
                        response_type=_route_type,
                    )
            self._cached_participant_summary = _roster.summary()
        except Exception as exc:
            self._cached_participant_summary = {}
            logger.debug(f"{self.log_prefix} 活跃名册更新异常: {exc}")
        try:
            _mmb = self._orch.get("multimodal_budget")

            _media_items = []
            _budget_ratio = 1.0
            if self._cached_metabolism_state is not None:
                _chat_fuel = float(getattr(self._cached_metabolism_state, "chat_fuel", 100.0) or 100.0) / 100.0
                _thinking_fuel = (
                    float(
                        getattr(
                            self._cached_metabolism_state,
                            "thinking_fuel",
                            100.0,
                        )
                        or 100.0
                    )
                    / 100.0
                )
                _budget_ratio = max(0.0, min(1.0, min(_chat_fuel, _thinking_fuel)))
            for i, _sr in enumerate(self._cached_semantic_routes):
                _cat = _sr.category.value if hasattr(_sr.category, "value") else str(_sr.category)
                if _cat in ("image", "multi_image", "sticker", "video"):
                    _message = incoming_batch[i] if i < len(incoming_batch) else None
                    _image_url = ""
                    _image_list = getattr(_message, "image_list", None) or []
                    if _image_list:
                        _first_image = _image_list[0]
                        if isinstance(_first_image, str):
                            _image_url = _first_image
                        else:
                            _image_url = str(
                                getattr(_first_image, "url", "")
                                or getattr(_first_image, "image_url", "")
                                or getattr(_first_image, "file", "")
                                or ""
                            )
                    _media_items.append(
                        {
                            "media_type": _cat,
                            "is_sticker": _cat == "sticker",
                            "image_url": _image_url,
                            "context_relevance": (0.85 if getattr(_sr, "has_text", False) else 0.45),
                        }
                    )
            if _media_items:
                _budgeter = _mmb
                _budget_decisions = _budgeter.evaluate_batch(_media_items, current_budget_ratio=_budget_ratio)
                self._cached_multimodal_summary = _budgeter.last_batch_summary()
                self._cached_media_decisions = _budget_decisions
            else:
                self._cached_multimodal_summary = {}
        except Exception as exc:
            self._cached_multimodal_summary = {}
            logger.debug(f"{self.log_prefix} 多模态预算评估异常: {exc}")

    def _integrate_deep_understanding(self, incoming_batch: List) -> None:
        """自我指称检测 + 可见性门控 + 理解门控"""
        try:
            from src.core.self_reference_detector import (
                get_self_reference_detector,
            )

            self._cached_self_references = get_self_reference_detector().detect_batch(incoming_batch)
        except Exception as exc:
            self._cached_self_references = []
            logger.debug(f"{self.log_prefix} 自我指称检测异常: {exc}")
        try:
            _vg = self._orch.get("visibility_gate")
            # F22：构建关系偏置映射——好感/信任用户更容易被看见
            _rel_map = {}
            try:
                _rsnap = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
                if _rsnap:
                    _aff = float(_rsnap.get("affection", 0.0) or 0.0)
                    if not self._metric_has_signal(_aff):
                        _rel_level = None
                    elif _aff > 60:
                        _rel_level = 4
                    elif _aff > 35:
                        _rel_level = 3
                    elif _aff < 15:
                        _rel_level = 0
                    else:
                        _rel_level = 2
                    if _rel_level is None:
                        _rel_map = {}
                    else:
                        for _m in incoming_batch:
                            _uid = str(getattr(_m, "user_id", "") or "").strip()
                            if self._is_human_message_obj(_m):
                                _rel_map[_uid] = _rel_level
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            _loafing_val = 0.0
            try:
                if self._cached_metabolism_state:
                    _loafing_val = float(getattr(self._cached_metabolism_state, "loafing_level", 0.0) or 0.0)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            self._cached_visibility_decisions = _vg.evaluate_batch(
                incoming_batch,
                presence_state=self._cached_presence_state,
                user_relationships=_rel_map if _rel_map else None,
                loafing_level=_loafing_val,
            )
        except Exception as exc:
            self._cached_visibility_decisions = []
            logger.debug(f"{self.log_prefix} 可见性门控异常: {exc}")
        try:
            _ug = self._orch.get("understanding_gate")
            _contexts = []
            _skip_indices = set()
            _SKIP_CATEGORIES = ("recall",)
            _rel_snap = getattr(self, "_last_relation_snapshot", None) or {}
            _presence_state = getattr(self, "_cached_presence_state", None)
            _emotion_state = getattr(self, "_cached_self_state", None)
            _social_willingness = float(getattr(_presence_state, "social_willingness", 0.5) or 0.5)
            if _social_willingness <= 0.0 and _emotion_state is not None:
                _social_willingness = float(getattr(_emotion_state, "social_desire", 0.5) or 0.5)
            _energy_ratio = 1.0
            if isinstance(getattr(self, "_cached_metabolism_constraints", None), dict):
                _energy_ratio = max(
                    0.05,
                    1.0 - float(self._cached_metabolism_constraints.get("energy_suppression", 0.0) or 0.0),
                )
            _focus_uid = str(getattr(self, "_last_user_id", "") or "")
            for i, _m in enumerate(incoming_batch):
                _vis_score = 0.5
                if i < len(self._cached_visibility_decisions):
                    _vis_score = float(getattr(self._cached_visibility_decisions[i], "score", 0.5) or 0.5)
                _cat = "plain_text"
                if i < len(self._cached_semantic_routes):
                    _sr = self._cached_semantic_routes[i]
                    _cat = _sr.category.value if hasattr(_sr.category, "value") else str(_sr.category)
                if _cat in _SKIP_CATEGORIES:
                    _skip_indices.add(i)
                    continue
                _text = str(getattr(_m, "processed_plain_text", "") or "")[:500]
                _uid = str(getattr(_m, "user_id", "") or "")
                _rel_level = 2
                if _uid and _uid in _rel_snap and isinstance(_rel_snap.get(_uid), dict):
                    _rel_level = int((_rel_snap.get(_uid) or {}).get("relationship_level", 2) or 2)
                elif _uid and _uid == _focus_uid and isinstance(_rel_snap, dict):
                    _rel_level = int(_rel_snap.get("relationship_level", 2) or 2)
                _bot_relevant = self._message_targets_bot(_m)
                _is_question = "?" in _text or "？" in _text or _text.endswith("吗") or _text.endswith("呢")
                _has_media = bool(
                    getattr(_m, "has_media", False)
                    or getattr(_m, "has_image", False)
                    or getattr(_m, "image_url", "")
                    or getattr(_m, "image_urls", None)
                )
                _contexts.append(
                    {
                        "text_content": _text,
                        "message_category": _cat,
                        "visibility_score": _vis_score,
                        "bot_relevant": _bot_relevant,
                        "is_question": _is_question,
                        "has_media": _has_media,
                        "energy_ratio": _energy_ratio,
                        "user_relationship_level": _rel_level,
                        "social_willingness": _social_willingness,
                    }
                )
            _len_skip = len(_skip_indices)
            if _len_skip > 0:
                logger.info(
                    f"{self.log_prefix} 🧹 理解门控前置过滤: 跳过{_len_skip}条低价值消息(RECALL等)，节省思考燃料"
                )
            self._cached_understanding_results = _ug.batch_evaluate(_contexts)
            if _skip_indices:
                try:
                    _filled = list(self._cached_understanding_results or [])
                    _orig_len = len(incoming_batch)
                    while len(_filled) < _orig_len - _len_skip:
                        _filled.append(None)
                    for _si in sorted(_skip_indices):
                        _filled.insert(_si, None)
                    self._cached_understanding_results = _filled
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
            # 深度理解完成后消耗思考燃料
            try:
                _d6 = EnergyChainDimension.get_instance()
                _think_len = sum(len(c.get("text_content", "")) for c in _contexts)

                class _ThinkEvt:
                    channel_id = self.stream_id
                    event_type = "thinking_completed"
                    complexity = max(0.1, min(2.0, _think_len / 200.0))
                    raw_extras = {}

                _d6.on_event(_ThinkEvt())
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        except Exception as exc:
            self._cached_understanding_results = []
            logger.debug(f"{self.log_prefix} 理解门控异常: {exc}")
        # 误解风险评估：检查用户后续消息是否暗示 bot 上轮回复被误解
        self._cached_misunderstanding_signal = None
        try:
            _last_bot = self._latest_recent_bot_utterance()
            if _last_bot and incoming_batch:
                _bot_text = str(_last_bot.get("text", "") or "")
                _bot_ts = float(_last_bot.get("ts", 0.0) or 0.0)
                if _bot_text and _bot_ts > 0:
                    # 找第一条非bot用户消息作为后续
                    _followup_text = ""
                    _followup_ts = 0.0
                    for _fm in incoming_batch:
                        _fu = str(getattr(_fm, "user_id", "") or "").strip()
                        if _fu and not self._is_bot_message_obj(_fm):
                            _followup_text = str(getattr(_fm, "processed_plain_text", "") or "")[:300]
                            _followup_ts = float(getattr(_fm, "time", 0.0) or 0.0)
                            break
                    if _followup_text:
                        _gap = max(0.0, _followup_ts - _bot_ts) if _followup_ts > _bot_ts else 60.0
                        # 简单重复提问检测：后续文本和bot回复前用户的上一条高度相似
                        _is_repeat = False
                        if len(self._recent_user_inputs) >= 2:
                            _prev_input = str(self._recent_user_inputs[-2].get("text", "") or "")
                            if _prev_input and _followup_text and len(_followup_text) > 3:
                                _overlap = sum(1 for c in _followup_text[:50] if c in _prev_input[:50])
                                _is_repeat = _overlap > len(_followup_text[:50]) * 0.6
                        from src.core.understanding_gate import (
                            get_understanding_gate,
                        )

                        _signal = get_understanding_gate().evaluate_misunderstanding_risk(
                            bot_reply_text=_bot_text,
                            user_followup_text=_followup_text,
                            is_repeated_question=_is_repeat,
                            time_gap_sec=_gap,
                        )
                        if _signal.risk_score > 0.35:
                            self._cached_misunderstanding_signal = _signal
                            if not hasattr(self, "_last_misunderstanding_log_ts"):
                                self._last_misunderstanding_log_ts = 0.0
                            _now_m = time.time()
                            if (_now_m - self._last_misunderstanding_log_ts) >= 30.0:
                                logger.info(
                                    f"{self.log_prefix} 误解风险={_signal.risk_score:.2f} "
                                    f"类型={_signal.signal_type} 线索={_signal.detected_cues[:2]}"
                                )
                                self._last_misunderstanding_log_ts = _now_m
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 误解风险评估异常: {exc}")

    def _apply_sensory_gates(self, incoming_batch: List, pinged_msg=None) -> Optional[str]:
        """感官门控：基于可见性+理解度判定是否应跳过本轮处理
        返回 None 表示放行，返回 str 表示拦截原因"""
        if not incoming_batch:
            return None
        _rest_verdict = self._evaluate_rest_governor(
            incoming_batch=incoming_batch,
            source="reactive",
            targeted_to_bot=self._has_targeted_bot_message(incoming_batch),
            admin_force=bool(getattr(self, "_is_admin_forced", False)),
            pinged_msg=pinged_msg,
        )
        if _rest_verdict.interruption_policy == "block" and pinged_msg is None:
            return f"休息门控: {self._summarize_rest_governor(_rest_verdict)}"
        _has_ping = pinged_msg is not None
        _vis_ignored_count = 0
        _vis_low_score_count = 0
        _understand_none_count = 0
        _understand_no_reply_count = 0
        # F29：三态区分——为每条消息标记暴露层级
        self._cached_exposure_tiers: Dict[str, str] = {}
        if self._cached_visibility_decisions:
            for _vd in self._cached_visibility_decisions:
                _vl = getattr(_vd, "level", None)
                if _vl is None:
                    continue
                _vl_val = _vl.value if hasattr(_vl, "value") else str(_vl)
                _msg_key = str(getattr(_vd, "message_key", "") or getattr(_vd, "msg_id", "") or "")
                if _vl_val == "understood_but_ignored":
                    _vis_ignored_count += 1
                    if _msg_key:
                        self._cached_exposure_tiers[_msg_key] = "seen_ignored"
                elif _vl_val in ("received", "glanced"):
                    _vis_low_score_count += 1
                    if _msg_key:
                        self._cached_exposure_tiers[_msg_key] = "never_seen"
                elif _vl_val in ("focused", "understood"):
                    if _msg_key:
                        self._cached_exposure_tiers[_msg_key] = "seen_engaged"
                _score = float(getattr(_vd, "score", 0.5) or 0.5)
                if _score < 0.18:
                    _vis_low_score_count += 1
        if self._cached_understanding_results:
            for _ur in self._cached_understanding_results:
                _cl = getattr(_ur, "level", None)
                if _cl is None:
                    continue
                _cl_val = _cl.value if hasattr(_cl, "value") else str(_cl)
                _ukey = str(getattr(_ur, "message_key", "") or getattr(_ur, "msg_id", "") or "")
                if _cl_val == "cannot_understand":
                    _understand_none_count += 1
                    if _ukey and _ukey not in self._cached_exposure_tiers:
                        self._cached_exposure_tiers[_ukey] = "never_seen"
                elif _cl_val == "can_understand_no_reply":
                    _understand_no_reply_count += 1
                    if _ukey:
                        self._cached_exposure_tiers[_ukey] = "seen_engaged_no_reply"
        _total_user_msgs = sum(1 for m in incoming_batch if not self._is_bot_message_obj(m))
        if _total_user_msgs <= 0:
            return None
        _ignore_ratio = (_vis_ignored_count + _understand_no_reply_count) / max(1, _total_user_msgs)
        _none_ratio = _understand_none_count / max(1, _total_user_msgs)
        if _none_ratio >= 0.8 and not _has_ping:
            return f"理解门控: {_understand_none_count}/{_total_user_msgs}条消息看不懂"
        _effective_ignore_threshold = 0.85
        if _ignore_ratio >= _effective_ignore_threshold and not _has_ping:
            # F29：附加三态分布
            _tier_counts = {
                "never_seen": 0,
                "seen_ignored": 0,
                "seen_engaged": 0,
                "seen_engaged_no_reply": 0,
            }
            for _t in self._cached_exposure_tiers.values():
                if _t in _tier_counts:
                    _tier_counts[_t] += 1
            _tier_str = f" | 三态:未看见={_tier_counts['never_seen']},扫到不理={_tier_counts['seen_ignored']},深度关注={
                _tier_counts['seen_engaged']
            },关注不回={_tier_counts['seen_engaged_no_reply']}"
            return f"可见性/理解门控: {_vis_ignored_count} 条被忽略+{_understand_no_reply_count} 条不回/{
                _total_user_msgs
            } 总消息{_tier_str} "
        if _vis_low_score_count >= _total_user_msgs and not _has_ping:
            return f"可见性全低分: {_vis_low_score_count}/{_total_user_msgs}条消息均未引起注意"
        # GAP-M：深度可见性评分——三维偏置+群环境调制+时间衰减
        try:
            from src.core.deep_visibility_scorer import (
                UserBiasProfile,
                EventBiasInput,
                TopicBiasInput,
                GroupEnvModulation,
                TimeDecayFactor,
            )

            if not self._deep_scorer_initialized:
                self._deep_scorer_initialized = True
            _dvs = self._orch.get("deep_visibility")
            self._cached_deep_visibility_results = []
            for _msg in incoming_batch:
                _uid_m = str(getattr(_msg, "user_id", "") or "")
                if not _uid_m or self._is_bot_message_obj(_msg):
                    continue
                _rel_snap_m = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
                _up_m = UserBiasProfile(
                    user_id=_uid_m,
                    affection=float(_rel_snap_m.get("affection", 0.0) or 0.0) / 100.0,
                    trust=float(_rel_snap_m.get("trust_value", 0.0) or 0.0) / 100.0,
                    annoyance=float(_rel_snap_m.get("annoyance_value", 0.0) or 0.0) / 100.0,
                    is_avoided=bool(float(_rel_snap_m.get("aversion_value", 0.0) or 0.0) > 30),
                )
                _plain = str(
                    getattr(_msg, "processed_plain_text", "")
                    or getattr(_msg, "plain_text", "")
                    or getattr(_msg, "content", "")
                    or ""
                )
                _ei_m = EventBiasInput(
                    is_at_mention=getattr(_msg, "is_at", False),
                    is_quote_reply=bool(getattr(_msg, "reply_to_message_id", None)),
                    message_length=len(_plain),
                )
                _ti_m = TopicBiasInput(message_text=_plain)
                _meta_state = getattr(self, "_cached_metabolism_constraints", None) or {}
                _presence_state_m = getattr(self, "_cached_presence_state", None)
                _scene_state_m = getattr(self, "_cached_scene_snapshot", None)
                _social_will_m = float(getattr(_presence_state_m, "social_willingness", 0.5) or 0.5)
                _watch_rank_m = int(getattr(_presence_state_m, "watch_state_rank", 2) or 2)
                _scene_atmo_m = ""
                if _scene_state_m is not None:
                    _scene_atmo_raw = getattr(_scene_state_m, "atmosphere", "")
                    _scene_atmo_m = (
                        getattr(_scene_atmo_raw, "value", str(_scene_atmo_raw)) if _scene_atmo_raw is not None else ""
                    )
                _gm_m = GroupEnvModulation(
                    group_atmosphere=str(_scene_atmo_m or ""),
                    active_member_count=int(getattr(_scene_state_m, "unique_speakers_5min", 0) or 0),
                    is_multi_thread=bool(int(getattr(_scene_state_m, "thread_count", 0) or 0) > 1),
                    suitable_to_interrupt=bool(getattr(_scene_state_m, "suitable_to_join", True)),
                    watch_state_rank=_watch_rank_m,
                    energy_ratio=1.0 - float(_meta_state.get("energy_suppression", 0.0) or 0.0),
                    social_willingness=_social_will_m,
                    loafing_level=float(_meta_state.get("loafing_suppression", 0.0) or 0.0),
                    boredom_level=float(_meta_state.get("boredom_level", 0.0) or 0.0),
                )
                _td_m = TimeDecayFactor(
                    message_age_sec=max(
                        0.0,
                        time.time() - float(getattr(_msg, "timestamp", time.time()) or time.time()),
                    )
                )
                _dvr = _dvs.evaluate(
                    _msg,
                    user_profile=_up_m,
                    event_input=_ei_m,
                    topic_input=_ti_m,
                    group_mod=_gm_m,
                    time_decay=_td_m,
                )
                self._cached_deep_visibility_results.append(_dvr)
            if self._cached_deep_visibility_results:
                _dv_high = sum(1 for r in self._cached_deep_visibility_results if r.modulated_score > 0.55)
                _dv_low = sum(1 for r in self._cached_deep_visibility_results if r.modulated_score < 0.12)
                if _dv_low >= _total_user_msgs * 0.7 and not _has_ping:
                    return f"深度可见性全低: {_dv_low}/{_total_user_msgs}条消息mod_score<0.12"
                logger.debug(
                    f"{self.log_prefix} [GAP-M] 深度评分完成: "
                    f"高优={_dv_high}, 低分={_dv_low}, 总={len(self._cached_deep_visibility_results)}"
                )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # GAP-E：消息打断权三级分类 —— 综合评分后分级处理
        if not _has_ping:
            try:
                _interruption_scores = self._classify_interruption_rights(incoming_batch)
                _tier1_count = sum(1 for s in _interruption_scores.values() if s < 0.30)
                _tier2_count = sum(1 for s in _interruption_scores.values() if 0.30 <= s < 0.65)
                _tier3_count = sum(1 for s in _interruption_scores.values() if s >= 0.65)
                self._cached_interruption_tiers = _interruption_scores
                _loaf_level = 0.0
                if self._cached_metabolism_constraints:
                    _loaf_level = float(self._cached_metabolism_constraints.get("loafing_suppression", 0.0) or 0.0)
                if _loaf_level > 0.65:
                    if _tier3_count == 0:
                        return f"打断权T1全拦截: loafing={_loaf_level: .2f}, T1={_tier1_count}  T2={_tier2_count}  T3={
                            _tier3_count
                        } "
                    elif _tier3_count <= 1 and _tier1_count > _tier3_count * 3:
                        return f"打断权T1主导: loafing={_loaf_level:.2f}, 仅{_tier3_count}条强优先级不足以拉起"
                elif _loaf_level > 0.35:
                    if _tier1_count >= _total_user_msgs * 0.8 and _tier3_count == 0:
                        return f"打断权半拦截: loafing={_loaf_level: .2f}, T1占比过高({_tier1_count} /{
                            _total_user_msgs
                        })且无强优先级"
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        return None

    def _classify_interruption_rights(self, incoming_batch: List) -> Dict[str, float]:
        """GAP-E：为每条消息计算打断权得分(0~1)，返回 {msg_key: score}

        三级划分：
          Tier 1 (score < 0.30): 普通消息 —— 高摸鱼时不进入深层可见性
          Tier 2 (0.30 ≤ score < 0.65): 中度相关 —— 允许peek/skim但不直接到active_watch
          Tier 3 (score ≥ 0.65): 强优先级 —— 可打断任意摸鱼状态，至少进入"看见"

        评分因子（加权求和）：
          - 被@或引用: +0.40
          - 讨论机器人相关: +0.25
          - 高好感用户发言: +0~0.20 (按好感度线性映射)
          - 当前话题高相关: +0.15
          - 冲突/高风险内容: +0.20
          - 连续互动线程: +0.10 (每多一轮+0.05)
        """
        scores: Dict[str, float] = {}
        rel_snap = getattr(self, "_last_relation_snapshot", None) or {}
        affection_map: Dict[str, float] = {}
        trust_map: Dict[str, float] = {}
        if isinstance(rel_snap, dict):
            for k, v in rel_snap.items():
                if "affection" in str(k).lower() or "好感" in str(k):
                    if isinstance(v, (int, float)):
                        affection_map[str(k)] = float(v)
                elif "trust" in str(k).lower() or "信任" in str(k):
                    if isinstance(v, (int, float)):
                        trust_map[str(k)] = float(v)
        for _msg in incoming_batch:
            _uid = str(getattr(_msg, "user_id", "") or "").strip()
            if not _uid or _uid == "bot":
                continue
            _key = str(getattr(_msg, "message_key", "") or getattr(_msg, "msg_id", "") or id(_msg))
            _score = 0.05
            # 因子1：@提及 / 引用回复
            _at_list = getattr(_msg, "at_list", None) or []
            _reply_to = str(getattr(_msg, "reply_to", "") or "").strip()
            if any(str(t).strip().lower() in ("bot", "robot", self.stream_id.lower()) for t in _at_list):
                _score += 0.42
            if _reply_to and len(_reply_to) > 0:
                _score += 0.18
            # 因子2：自我引用检测
            if self._cached_self_references:
                for _ref in self._cached_self_references:
                    _ref_msg_key = str(getattr(_ref, "message_key", "") or "")
                    if _ref_msg_key == _key or _ref_msg_key in (_key[: min(16, len(_key))],):
                        _rt = getattr(_ref, "ref_type", None)
                        if _rt:
                            _rv = _rt.value if hasattr(_rt, "value") else str(_rt)
                            if _rv == "direct_at":
                                _score += 0.38
                            elif _rv == "discussed_as_topic":
                                _score += 0.28
                            elif _rv == "quoted_reply":
                                _score += 0.22
                            elif _rv == "nickname_called":
                                _score += 0.32
                            break
            # 因子3：关系加权（高好感用户的消息获得更高打断权）
            _rel_view = self._normalize_relation_snapshot(rel_snap or {}) if isinstance(rel_snap, dict) else {}
            _aff = float(_rel_view.get("affection", 0.0) or 0.0)
            if self._metric_has_signal(_aff) and _aff > 60:
                _score += min(0.18, (_aff - 60.0) / 200.0)
            elif self._metric_has_signal(_aff) and _aff < 18:
                _score -= 0.08
            # 因子4：话题相关性
            if self._cached_pattern_evidence:
                for _pe in self._cached_pattern_evidence:
                    _pe_conf = float(getattr(_pe, "confidence", 0.0) or 0.0)
                    _pe_pat = getattr(_pe, "pattern", None)
                    if _pe_pat:
                        _pv = _pe_pat.value if hasattr(_pe_pat, "value") else str(_pe_pat)
                        if (
                            _pv
                            in (
                                "heated_discussion",
                                "conflict_escalation",
                                "argument",
                            )
                            and _pe_conf > 0.4
                        ):
                            _score += 0.12 * _pe_conf
                        elif (
                            _pv
                            in (
                                "newcomer_welcome",
                                "birthday_wish",
                                "congratulation",
                            )
                            and _pe_conf > 0.5
                        ):
                            _score += 0.08 * _pe_conf
            # 因子5：文本长度与内容密度（短消息默认低权重，长消息略增）
            _text = str(getattr(_msg, "processed_plain_text", "") or "")[:300]
            if len(_text) > 80:
                _score += min(0.06, len(_text) / 2000.0)
            # 因子6：图片/多媒体附加权重
            _has_media = bool(getattr(_msg, "has_image", False) or getattr(_msg, "media_count", 0))
            if _has_media:
                _score += 0.04
            scores[_key] = max(0.0, min(1.0, _score))
        return scores


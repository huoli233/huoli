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

class StrategyProactiveDecisionMixin:
    def _evaluate_proactive_decision(self, silence_sec: float) -> "ProactiveDecision":
        """采集多维信号并交由统一决策器融合评分。IntegrationHub 已在上层 async 中处理。"""
        from src.chat.proactive.proactive_decider import (
            get_proactive_decider,
            SignalBundle,
        )

        bundle = SignalBundle(channel_id=self.stream_id, silence_seconds=silence_sec)

        # 信号1: 情感就绪度
        try:
            from src.chat.heart_flow.emotion_driven_core import (
                get_emotion_driven_core,
            )

            snap = get_emotion_driven_core().get_state_snapshot(self.stream_id)
            bundle.emotional_readiness = float(snap.get("proactive_willingness", 0.0) or 0.0)
            bundle.boredom = float(snap.get("boredom", 0.0) or 0.0)
            bundle.loneliness = float(snap.get("loneliness", 0.0) or 0.0)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 决策情感信号采集(模块缺失，使用缓存回退): {exc}")
            # 回退: 从已缓存的代谢/存在态推算近似情感信号
            try:
                _ms = getattr(self, "_cached_metabolism_state", None)
                if _ms:
                    _mood = float(getattr(_ms, "mood_valence", 0.5) or 0.5)
                    _arousal = float(getattr(_ms, "mood_arousal", 0.5) or 0.5)
                    bundle.emotional_readiness = max(0.0, min(1.0, _mood * 0.5 + _arousal * 0.5))
                _ps = getattr(self, "_cached_presence_state", None)
                if _ps:
                    _out_att = float(getattr(_ps, "outward_attention", 0.5) or 0.5)
                    _soc_w = float(getattr(_ps, "social_willingness", 0.5) or 0.5)
                    bundle.boredom = max(0.0, min(1.0, 1.0 - _out_att))
                    bundle.loneliness = max(0.0, min(1.0, _soc_w * 0.6))
            except Exception:
                # 最后兜底: 给中性值而非0
                bundle.emotional_readiness = 0.3
                bundle.boredom = 0.2
                bundle.loneliness = 0.15

        # 信号2: 能量储备
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            _cp = float(_ch.chat_pool) if _ch else 50.0
            _cc = float(_ch.chat_ceiling) if _ch else 100.0
            _tv = float(_ch.thinking_value) if _ch else 50.0
            _tc = float(_ch.thinking_ceiling) if _ch else 100.0
            _al = float(_ch.activity_level) if _ch else 50.0
            chat_factor = max(0.0, min(1.0, _cp / max(_cc, 1.0)))
            thinking_factor = max(0.0, min(1.0, _tv / max(_tc, 1.0)))
            activity_factor = max(0.0, min(1.0, _al / 100.0))
            bundle.vitality_ratio = max(
                0.0,
                min(
                    1.0,
                    chat_factor * 0.35 + thinking_factor * 0.4 + activity_factor * 0.25,
                ),
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 决策能量信号采集失败: {exc}")

        # 信号4: 内容新鲜度（空闲主动路径给中性值）
        try:
            from src.core.content_state_tracker import (
                get_content_state_tracker,
            )

            _ = get_content_state_tracker()
            bundle.content_novelty = 0.5
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 决策内容信号采集失败: {exc}")

        # 信号5: 社交声望
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            bundle.social_standing = float(_ch.social_value) if _ch else 0.0
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 决策社交信号采集失败: {exc}")

        # 信号6: 内心独白欲望
        if self._cached_voice and hasattr(self._cached_voice, "reply_desire_level"):
            try:
                bundle.inner_voice_desire = int(getattr(self._cached_voice, "reply_desire_level", 5) or 5)
            except Exception:
                bundle.inner_voice_desire = 5

        # 信号7: 意图池驱动强度
        try:
            from src.chat.proactive.intention_pool import get_intention_pool

            bundle.intention_drive = get_intention_pool().build_decision_signal(self.stream_id)
        except Exception:
            bundle.intention_drive = 0.0

        decider = get_proactive_decider()
        verdict = decider.evaluate(bundle)
        _intent_override = getattr(verdict, "intent_override", False)

        # F6：群体行为模式 → 动态调整 activation_bar
        _pattern_bar_delta = 0.0
        if self._cached_pattern_evidence:
            _top_pattern = None
            _top_conf = 0.0
            for _pe in self._cached_pattern_evidence:
                _pc = float(getattr(_pe, "confidence", 0.0) or 0.0)
                if _pc > _top_conf:
                    _top_conf = _pc
                    _top_pattern = _pe
            if _top_pattern is not None and _top_conf >= 0.45:
                _pt_val = getattr(getattr(_top_pattern, "pattern", None), "value", "") or ""
                if _pt_val == "newcomer_welcome":
                    _pattern_bar_delta = -0.12
                    verdict.breakdown["pattern_welcome_boost"] = 0.12
                elif _pt_val == "ritual_greeting":
                    _pattern_bar_delta = -0.08
                    verdict.breakdown["pattern_ritual_boost"] = 0.08
                elif _pt_val == "spectator_mode":
                    # F35：围观→参与转化——事件关联度高时可突破围观态
                    _event_relevance = 0.0
                    try:
                        if self._cached_self_references:
                            for _ref in self._cached_self_references:
                                _rt = getattr(_ref, "ref_type", None)
                                if _rt:
                                    _rv = self._normalize_self_reference_type(_rt)
                                    if _rv in (
                                        "discussed_as_topic",
                                        "quoted_reply",
                                    ):
                                        _event_relevance += float(getattr(_ref, "strength", 0.3) or 0.3)
                        _rel_snap35 = self._normalize_relation_snapshot(
                            getattr(self, "_last_relation_snapshot", None) or {}
                        )
                        if float(_rel_snap35.get("affection", 0.0) or 0.0) > 55:
                            _event_relevance += 0.15
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
                    if _event_relevance > 0.5:
                        _pattern_bar_delta = -0.02
                        verdict.breakdown["spectator_to_engage_convert"] = 0.02
                    elif _event_relevance > 0.25:
                        _pattern_bar_delta = 0.06
                        verdict.breakdown["spectator_partial_convert"] = -0.06
                    else:
                        _pattern_bar_delta = 0.15
                        verdict.breakdown["pattern_spectator_penalty"] = -0.15
                elif _pt_val in (
                    "heated_discussion",
                    "argument",
                    "conflict_escalation",
                ):
                    _pattern_bar_delta = 0.18
                    verdict.breakdown["pattern_heated_penalty"] = -0.18
                elif _pt_val == "celebration_wave":
                    _pattern_bar_delta = -0.06
                    verdict.breakdown["pattern_celebration_boost"] = 0.06
                elif _pt_val == "support_circle":
                    _pattern_bar_delta = -0.10
                    verdict.breakdown["pattern_support_boost"] = 0.10
                # F33：扩展仪式性行为识别——不当作普通问答
                elif _pt_val in (
                    "birthday_wish",
                    "holiday_greeting",
                    "congratulation",
                ):
                    _pattern_bar_delta = -0.10
                    verdict.breakdown["pattern_ritual_celebration"] = 0.10
                elif _pt_val == "chain_reply":
                    _pattern_bar_delta = -0.05
                    verdict.breakdown["pattern_chain_reply"] = 0.05
                elif _pt_val == "group_photo":
                    _pattern_bar_delta = -0.07
                    verdict.breakdown["pattern_group_photo"] = 0.07
                if abs(_pattern_bar_delta) > 0.001:
                    verdict.activation_bar = max(
                        0.08,
                        min(0.92, verdict.activation_bar + _pattern_bar_delta),
                    )
                # F33：缓存仪式行为标记供 planner 使用
                if _top_conf >= 0.45 and _pt_val in (
                    "newcomer_welcome",
                    "ritual_greeting",
                    "birthday_wish",
                    "holiday_greeting",
                    "congratulation",
                    "chain_reply",
                    "celebration_wave",
                    "support_circle",
                    "group_photo",
                ):
                    self._cached_ritual_behavior = _pt_val
        # F9：行为映射层——量化真值动态修改 arbiter 参数
        _rel_snap = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
        if _rel_snap:
            _affection = float(_rel_snap.get("affection", 0.0) or 0.0)
            _trust = float(_rel_snap.get("trust_value", 0.0) or 0.0)
            _annoyance = float(_rel_snap.get("annoyance_value", 0.0) or 0.0)
            _aversion = float(_rel_snap.get("aversion_value", 0.0) or 0.0)
            _has_affection_signal = self._metric_has_signal(_affection)
            _has_trust_signal = self._metric_has_signal(_trust)
            if _has_affection_signal and _has_trust_signal and _affection < 15 and _trust < 20:
                _affix_penalty = 0.10
                verdict.activation_bar = min(0.92, verdict.activation_bar + _affix_penalty)
                verdict.breakdown["relation_avoid_penalty"] = -_affix_penalty
            elif _has_affection_signal and _has_trust_signal and _affection > 70 and _trust > 60:
                _affix_boost = -0.08
                verdict.activation_bar = max(0.08, verdict.activation_bar + _affix_boost)
                verdict.breakdown["relation_affinity_boost"] = _affix_boost
            if _annoyance > 35 or _aversion > 30:
                _avoid_p = min(0.25, (_annoyance * 0.003) + (_aversion * 0.002))
                verdict.activation_bar = min(0.92, verdict.activation_bar + _avoid_p)
                verdict.breakdown["relation_annoyance_penalty"] = -_avoid_p
                # 高厌烦时同步压低融合分
                if _annoyance > 55:
                    _score_damp = min(0.15, (_annoyance - 55) * 0.005)
                    verdict.fused_score = max(0.0, verdict.fused_score - _score_damp)
                    verdict.breakdown["annoyance_score_damp"] = -_score_damp
        # F13+F30：话题归属——五类归属动态调整参与门槛
        try:
            _scene = self._orch.get("scene_state")
            _active_topics = _scene.active_topic_slots(limit=3)
            _dominant_ownership = "ambiguous_topic"
            _dominant_topic_obj = None
            for _t in _active_topics:
                _dominant_ownership = _t.ownership_type()
                _dominant_topic_obj = _t
                break
            if _dominant_ownership == "self_topic":
                _topic_boost = -0.10
                verdict.activation_bar = max(0.08, verdict.activation_bar + _topic_boost)
                verdict.breakdown["self_topic_chase"] = _topic_boost
            elif _dominant_ownership == "target_user_topic":
                _rel_snap30 = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
                _aff30 = float(_rel_snap30.get("affection", 0.0) or 0.0)
                _has_aff30_signal = self._metric_has_signal(_aff30)
                if _has_aff30_signal and _aff30 < 25:
                    _tp_penalty = 0.08
                    verdict.activation_bar = min(0.92, verdict.activation_bar + _tp_penalty)
                    verdict.breakdown["target_user_low_affinity"] = -_tp_penalty
                elif _has_aff30_signal and _aff30 > 60:
                    _tp_boost = -0.05
                    verdict.activation_bar = max(0.08, verdict.activation_bar + _tp_boost)
                    verdict.breakdown["target_user_high_affinity"] = _tp_boost
            elif _dominant_ownership == "shared_group_topic":
                _gp_boost = -0.04
                verdict.activation_bar = max(0.08, verdict.activation_bar + _gp_boost)
                verdict.breakdown["shared_group_topic_natural"] = _gp_boost
            elif _dominant_ownership == "external_topic":
                _et_penalty = 0.12
                verdict.activation_bar = min(0.92, verdict.activation_bar + _et_penalty)
                verdict.breakdown["external_topic_low_interest"] = -_et_penalty
            elif _dominant_ownership == "ambiguous_topic":
                _am_penalty = 0.06
                verdict.activation_bar = min(0.92, verdict.activation_bar + _am_penalty)
                verdict.breakdown["ambiguous_topic_conservative"] = -_am_penalty
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # F31：自我引用联合检测增强——多人讨论/引用发言时强力提升关注度
        try:
            if self._cached_self_references:
                _ref_types_found = set()
                _max_ref_strength = 0.0
                _discuss_count = 0
                _quote_count = 0
                for _ref in self._cached_self_references:
                    _rt = getattr(_ref, "ref_type", None)
                    _rs = float(getattr(_ref, "strength", 0.0) or 0.0)
                    if _rt:
                        _rt_val = self._normalize_self_reference_type(_rt)
                        _ref_types_found.add(_rt_val)
                        if _rs > _max_ref_strength:
                            _max_ref_strength = _rs
                        if _rt_val == "discussed_as_topic":
                            _discuss_count += 1
                        elif _rt_val == "quoted_reply":
                            _quote_count += 1
                if _discuss_count >= 2 or (_max_ref_strength > 0.6 and "discussed_as_topic" in _ref_types_found):
                    _self_ref_boost = min(0.18, 0.06 + _discuss_count * 0.04)
                    verdict.activation_bar = max(0.05, verdict.activation_bar - _self_ref_boost)
                    verdict.fused_score = min(1.0, verdict.fused_score + _self_ref_boost * 0.5)
                    verdict.breakdown["self_ref_group_discussion"] = _self_ref_boost
                elif "quoted_reply" in _ref_types_found and _max_ref_strength > 0.4:
                    _quote_boost = 0.06
                    verdict.activation_bar = max(0.08, verdict.activation_bar - _quote_boost)
                    verdict.breakdown["self_ref_quoted"] = _quote_boost
                elif "direct_at" in _ref_types_found or "nickname_called" in _ref_types_found:
                    verdict.breakdown["self_ref_direct_ack"] = 0.02
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # F24：创伤放大负面事件 + 厌烦放大回避倾向
        try:
            _rel_snap2 = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
            _trauma = float(_rel_snap2.get("trauma_score", 0.0) or 0.0)
            _annoy = float(_rel_snap2.get("annoyance_value", 0.0) or 0.0)
            if _trauma > 30:
                _trauma_amp = min(0.15, (_trauma / 100.0) * 0.3)
                verdict.activation_bar = min(0.92, verdict.activation_bar + _trauma_amp)
                verdict.breakdown["trauma_amplification"] = -_trauma_amp
            if _annoy > 30:
                _annoy_amp = min(0.18, (_annoy / 100.0) * 0.35)
                verdict.fused_score = max(0.0, verdict.fused_score - _annoy_amp)
                verdict.breakdown["annoyance_avoidance"] = -_annoy_amp
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # F32：内心旁白→行为策略联动——旁白建议观察/防御时压制参与欲
        try:
            if self._cached_narration_plan:
                _strategy = str(getattr(self._cached_narration_plan, "behavior_strategy", "") or "")
                _narr_mood = str(getattr(self._cached_narration_plan, "dominant_mood", "") or "")
                if _strategy == "observe_only":
                    _obs_penalty = 0.10
                    verdict.activation_bar = min(0.92, verdict.activation_bar + _obs_penalty)
                    verdict.breakdown["narration_observe_strategy"] = -_obs_penalty
                elif _strategy == "defensive":
                    _def_penalty = 0.14
                    verdict.activation_bar = min(0.92, verdict.activation_bar + _def_penalty)
                    verdict.fused_score = max(0.0, verdict.fused_score - 0.05)
                    verdict.breakdown["narration_defensive_strategy"] = -_def_penalty
                elif _strategy == "spectator_eat_melon":
                    _spec_penalty = 0.08
                    verdict.activation_bar = min(0.88, verdict.activation_bar + _spec_penalty)
                    verdict.breakdown["narration_spectator"] = -_spec_penalty
                elif _strategy == "engage_active":
                    _eng_boost = -0.06
                    verdict.activation_bar = max(0.08, verdict.activation_bar + _eng_boost)
                    verdict.breakdown["narration_engage_boost"] = _eng_boost
                if _narr_mood in ("annoyed", "tired", "overwhelmed"):
                    _mood_p = 0.05
                    verdict.activation_bar = min(0.92, verdict.activation_bar + _mood_p)
                    verdict.breakdown[f"narration_mood_{_narr_mood}"] = -_mood_p
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # 每小时回复频率疲劳：主动发言过于频繁时提升门槛
        _hr_count = self._current_hourly_proactive_reply_count()
        if _hr_count > 3:
            _freq_fatigue = min(0.18, (_hr_count - 3) * 0.045)
            verdict.activation_bar = min(0.92, verdict.activation_bar + _freq_fatigue)
            verdict.breakdown["reply_frequency_fatigue"] = -_freq_fatigue
        ignored_turns = max(0, int(self._unanswered_bot_turns))
        if ignored_turns:
            # 削减惩罚力度: 0.06/轮 上限0.15 (原: 0.10/轮 上限0.30)
            unanswered_penalty = min(0.15, ignored_turns * 0.06)
            if _intent_override:
                unanswered_penalty *= 0.3
            verdict.fused_score = max(0.0, verdict.fused_score - unanswered_penalty)
            verdict.breakdown["unanswered_penalty"] = -unanswered_penalty
            if verdict.rationale:
                verdict.rationale = f"{verdict.rationale} | 未回应轮数={ignored_turns}"
            else:
                verdict.rationale = f"未回应轮数={ignored_turns}"
        # 不再直接 should_proceed=False，改为提高 activation_bar
        if ignored_turns >= 2:
            _ignore_bar_raise = min(0.20, ignored_turns * 0.05)
            verdict.activation_bar = min(0.85, verdict.activation_bar + _ignore_bar_raise)
            verdict.breakdown["ignore_bar_raise"] = -_ignore_bar_raise
        # R2正向激励：统一调用共享方法
        self._apply_r2_incentives(verdict, bundle.silence_seconds)

        # SOC-03: 负面情绪影响主动发言欲望
        if verdict.should_proceed and self._cached_user_negative_emotion > 50:
            _neg_penalty = min(0.3, self._cached_user_negative_emotion / 200.0)
            if _intent_override:
                _neg_penalty *= 0.35
            verdict.fused_score = max(0.0, verdict.fused_score - _neg_penalty)
            verdict.breakdown["negative_emotion_suppress"] = -_neg_penalty
            if verdict.fused_score < verdict.activation_bar and not _intent_override:
                verdict.should_proceed = False
                verdict.rationale = f"用户负面情绪高({self._cached_user_negative_emotion:.0f})，抑制主动发言"
        # F34：多模态预算联动——刷图风暴/大量跳过时进入节流模式
        try:
            if self._cached_multimodal_summary:
                _total_media = int(self._cached_multimodal_summary.get("total_items", 0) or 0)
                _skipped = int(self._cached_multimodal_summary.get("skipped_count", 0) or 0)
                _cached_used = int(self._cached_multimodal_summary.get("cache_hit_count", 0) or 0)
                _is_storm = str(self._cached_multimodal_summary.get("storm_mode", "") or "") == "active"
                if _total_media >= 5 and _skipped >= 3:
                    _media_throttle = 0.08
                    verdict.activation_bar = min(0.92, verdict.activation_bar + _media_throttle)
                    verdict.breakdown["media_throttle_many_skipped"] = -_media_throttle
                elif _is_storm:
                    _storm_p = 0.12
                    verdict.activation_bar = min(0.92, verdict.activation_bar + _storm_p)
                    verdict.breakdown["media_storm_throttle"] = -_storm_p
                if _total_media > 0 and _skipped == _total_media:
                    verdict.breakdown["media_all_skipped_note"] = 1.0
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # ═══════════════════════════════════════════
        #  统一多因子决策融合（替代散装 F26/F27/F28 硬阈值）
        #  所有能量/睡眠/上限/存在态/关系/话题/模式/引用/
        #  旁白/多模态/围观 走同一套 sigmoid+疲劳曲线+连续映射算法
        # ═══════════════════════════════════════════
        try:
            from src.core.multi_factor_decision_engine import (
                get_multi_factor_engine,
                FactorInput,
            )

            _mf = get_multi_factor_engine()
            _inp = FactorInput()
            try:
                _d6 = EnergyChainDimension.get_instance()
                _ch = _d6._ensure_channel(self.stream_id)
                _ecv = float(_ch.chat_pool) if _ch else 50.0
                _etv = float(_ch.thinking_value) if _ch else 50.0
                _eav = float(_ch.activity_level) if _ch else 50.0
                _esv = float(_ch.social_value) if _ch else 0.0
                _e_cc = max(float(_ch.chat_ceiling) if _ch else 100.0, 1.0)
                _e_tc = max(float(_ch.thinking_ceiling) if _ch else 100.0, 1.0)
                _inp.energy_chat_ratio = max(0.0, min(1.0, _ecv / _e_cc))
                _inp.energy_think_ratio = max(0.0, min(1.0, _etv / _e_tc))
                _inp.energy_activity_ratio = max(0.0, min(1.0, _eav / 100.0))
                _inp.energy_social_ratio = max(0.0, min(1.0, (_esv + 100.0) / 200.0))
                _inp.energy_composite = max(
                    0.03,
                    _inp.energy_think_ratio * 0.35
                    + _inp.energy_chat_ratio * 0.28
                    + _inp.energy_activity_ratio * 0.22
                    + _inp.energy_social_ratio * 0.15,
                )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            if self._cached_night_phase:
                _phase_val = getattr(self._cached_night_phase, "value", None)
                if _phase_val is not None:
                    _inp.night_phase = self._normalized_night_phase_value() or str(_phase_val).lower()
                else:
                    _inp.night_phase = self._normalized_night_phase_value() or getattr(
                        self._cached_night_phase, "name", "UNKNOWN"
                    )
                _ncs = None
                try:
                    from src.core.night_cycle_system import get_night_cycle

                    _ncs = get_night_cycle(self.stream_id)
                except Exception as _exc:
                    logger.debug(f"{self.log_prefix} 夜间节律对象获取失败: {_exc}")
                try:
                    _d6 = EnergyChainDimension.get_instance()
                    _nm = _d6._get_night_mode(self.stream_id)
                    if _nm:
                        _inp.sleep_debt = max(0.0, 1.0 - _nm.prob_multiplier)
                        _inp.sleep_peek_budget = _nm.prob_multiplier >= 0.3
                    else:
                        _inp.sleep_debt = 0.0
                        _inp.sleep_peek_budget = True
                except Exception:
                    _inp.sleep_debt = 0.0
                    _inp.sleep_peek_budget = True
                if _ncs is not None:
                    _inp.sleep_reply_budget = _ncs.evaluate_sleep_reply_budget()
                    # GAP-A：熬夜压力与亢奋链数据
                    try:
                        _pressure_info = _ncs.get_overnight_pressure_breakdown()
                        _inp.overnight_pressure = float(_pressure_info.get("total", 0.0))
                        self._overnight_pressure_total = _inp.overnight_pressure
                        _arousal_info = _ncs.get_night_expression_profile()
                        _inp.arousal_chain_state = str(_arousal_info.get("current_state", "normal"))
                        _inp.is_burnthrough = bool(_arousal_info.get("active_template") == "burnthrough")
                        _raw_expr = _arousal_info.get("raw_expression", {}) or {}
                        _inp.burnthrough_impulse = float(_raw_expr.get("impulse", 0.0))
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
            if self._cached_presence_state:
                _inp.social_willingness = float(getattr(self._cached_presence_state, "social_willingness", 0.5) or 0.5)
                _inp.watch_willingness = float(getattr(self._cached_presence_state, "watch_willingness", 0.5) or 0.5)
                _inp.avoidance_tendency = float(getattr(self._cached_presence_state, "avoidance_tendency", 0.0) or 0.0)
                _inp.quiet_preference = float(getattr(self._cached_presence_state, "quiet_preference", 0.2) or 0.2)
                _inp.curiosity_level = float(getattr(self._cached_presence_state, "curiosity_level", 0.3) or 0.3)
                _inp.outward_attention = float(getattr(self._cached_presence_state, "outward_attention", 0.5) or 0.5)
            _rel_mf = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
            _inp.affection = float(_rel_mf.get("affection", 0.0) or 0.0)
            _inp.trust_value = float(_rel_mf.get("trust_value", 0.0) or 0.0)
            _inp.annoyance_value = float(_rel_mf.get("annoyance_value", 0.0) or 0.0)
            _inp.trauma_score = float(_rel_mf.get("trauma_score", 0.0) or 0.0)
            _inp.aversion_value = float(_rel_mf.get("aversion_value", 0.0) or 0.0)
            try:
                _scene_mf = self._orch.get("scene_state")
                _topics_mf = _scene_mf.active_topic_slots(limit=3)
                for _t in _topics_mf:
                    _inp.topic_ownership = _t.ownership_type()
                    break
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            if self._cached_pattern_evidence:
                _top_pe = None
                _top_pc = 0.0
                for _pe in self._cached_pattern_evidence:
                    _pc = float(getattr(_pe, "confidence", 0.0) or 0.0)
                    if _pc > _top_pc:
                        _top_pc = _pc
                        _top_pe = _pe
                if _top_pe:
                    _pt_enum = getattr(_top_pe, "pattern", None)
                    _inp.dominant_pattern = (
                        (_pt_enum.value if hasattr(_pt_enum, "value") else str(_pt_enum)) if _pt_enum else ""
                    )
                    _inp.pattern_confidence = _top_pc
                    _inp.is_spectator_mode = _inp.dominant_pattern == "spectator_mode"
            if self._cached_self_references:
                _ref_types_set = set()
                _max_rs = 0.0
                _discuss_n = 0
                _quote_n = 0
                for _ref in self._cached_self_references:
                    _rt = getattr(_ref, "ref_type", None)
                    _rs = float(getattr(_ref, "strength", 0.0) or 0.0)
                    if _rt:
                        _rv = _rt.value if hasattr(_rt, "value") else str(_rt)
                        _ref_types_set.add(_rv)
                        if _rs > _max_rs:
                            _max_rs = _rs
                        if _rv == "discussed_as_topic":
                            _discuss_n += 1
                        elif _rv == "quoted_reply":
                            _quote_n += 1
                _inp.self_ref_types = list(_ref_types_set)
                _inp.self_ref_max_strength = _max_rs
                _inp.discuss_as_topic_count = _discuss_n
                _inp.quoted_reply_count = _quote_n
                _inp.direct_at = "direct_at" in _ref_types_set or "nickname_called" in _ref_types_set
                if _inp.is_spectator_mode:
                    _event_rel = 0.0
                    for _ref in self._cached_self_references:
                        _rt2 = getattr(_ref, "ref_type", None)
                        if _rt2:
                            _rv2 = _rt2.value if hasattr(_rt2, "value") else str(_rt2)
                            if _rv2 in ("discussed_as_topic", "quoted_reply"):
                                _event_rel += float(getattr(_ref, "strength", 0.3) or 0.3)
                    if _inp.affection > 55:
                        _event_rel += 0.15
                    _inp.event_relevance = _event_rel
            if self._cached_narration_plan:
                _inp.narration_strategy = str(getattr(self._cached_narration_plan, "behavior_strategy", "") or "")
                _inp.narration_mood = str(getattr(self._cached_narration_plan, "dominant_mood", "") or "")
            if self._cached_multimodal_summary:
                _inp.media_total_items = int(self._cached_multimodal_summary.get("total_items", 0) or 0)
                _inp.media_skipped_count = int(self._cached_multimodal_summary.get("skipped_count", 0) or 0)
                _inp.media_storm_active = str(self._cached_multimodal_summary.get("storm_mode", "") or "") == "active"
            if self._cached_metabolism_state:
                _inp.consecutive_active_minutes = float(
                    getattr(
                        self._cached_metabolism_state,
                        "consecutive_active_minutes",
                        0.0,
                    )
                    or 0.0
                )
            _inp.hourly_reply_used = self._current_hourly_proactive_reply_count()
            _inp.intention_drive = bundle.intention_drive
            _inp.intent_override_candidate = _intent_override and bundle.intention_drive >= 0.78
            try:
                from src.core.state_coupling_matrix import get_coupling_engine

                _scme = get_coupling_engine()
                _scme_snap = _scme.build_full_snapshot(
                    metabolism=getattr(self, "_cached_metabolism_state", None),
                    presence=getattr(self, "_cached_presence_state", None),
                    relation=self._last_relation_snapshot,
                    night_phase=getattr(self, "_cached_night_phase", None),
                    group_scene=getattr(self, "_cached_scene_snapshot", None),
                    watch_state=self._watch_level_value(),
                )
                if _scme_snap:
                    _inp.coupling_boredom_mod = float(_scme_snap.get("boredom_to_loafing_mod", 0.0) or 0.0)
                    _inp.coupling_activity_mod = float(_scme_snap.get("activity_behavior_mod", 0.0) or 0.0)
                    _inp.coupling_style_tier = str(_scme_snap.get("style_tier", "") or "")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            try:
                from src.core.impression_evolution_hub import (
                    get_impression_hub,
                )

                _ihub = get_impression_hub(self.stream_id)
                _focus_uid = str(getattr(bundle, "anchor_user_id", "") or "")
                if _focus_uid:
                    _isum = _ihub.get_impression_summary(_focus_uid)
                    if _isum.get("exists"):
                        _inp.impression_narrative_type = _isum.get("narrative_type", "")
                        _inp.impression_affection_proxy = _isum.get("affection", 0.0)
                        _inp.impression_trust_proxy = _isum.get("trust_value", 0.0)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            try:
                _slh = self._orch.get("skill_hub")
                if _slh:
                    _active_skills = _slh.get_active_skills()
                    if _active_skills:
                        _avg_cost = sum(s.cost_modifier() for s in _active_skills) / max(1, len(_active_skills))
                        _inp.skill_cost_modifier = _avg_cost
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            try:
                from src.core.multimodal_semantic_bridge import (
                    get_multimodal_semantic_bridge,
                )

                _mm_bridge = get_multimodal_semantic_bridge()
                _mm_cached = getattr(self, "_cached_mm_bridge_result", None)
                if isinstance(_mm_cached, dict):
                    _inp.mm_avg_valence = float(_mm_cached.get("avg_valence", 0.0) or 0.0)
                    _inp.mm_engagement = float(_mm_cached.get("avg_engagement", 0.0) or 0.0)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            # 仪表盘ActionVerdict预检：L2/L3裁定结果注入决策器
            try:
                _dv = self._get_dashboard_verdict()
                _dv_urgency = str(_dv.get("urgency", "") or "")
                _dv_reply = _dv.get("reply", False)
                _dv_process = _dv.get("process", True)
                _dv_confidence = float(_dv.get("confidence", 0.5) or 0.5)
                _dv_path = str(_dv.get("path", "") or "")
                if not _dv_process:
                    verdict.should_proceed = False
                    verdict.activation_bar = min(0.95, verdict.activation_bar + 0.25)
                    verdict.rationale = f"{verdict.rationale or ''} | 仪表盘L1阻断({_dv.get('reason', '')[:40]})"
                elif not _dv_reply and _dv_urgency in ("不回复", "跳过"):
                    _skip_penalty = 0.08 * _dv_confidence
                    verdict.activation_bar = min(0.90, verdict.activation_bar + _skip_penalty)
                    verdict.fused_score = max(0.0, verdict.fused_score - _skip_penalty * 0.4)
                    verdict.breakdown["dashboard_l2_skip"] = -_skip_penalty
                    logger.debug(f"{self.log_prefix} 仪表盘L2跳过: urgency={_dv_urgency} path={_dv_path}")
                elif _dv_reply and _dv_urgency in ("立即回复", "尽快回复"):
                    _reply_boost = 0.05 * _dv_confidence
                    verdict.activation_bar = max(0.05, verdict.activation_bar - _reply_boost)
                    verdict.breakdown["dashboard_l2_boost"] = _reply_boost
                _dv_tone = str(_dv.get("tone", "") or "")
                if _dv_tone:
                    verdict.rationale = f"{verdict.rationale or ''} | 语气建议={_dv_tone}"
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            _mf_verdict = _mf.evaluate(_inp)
            _mf_verdict.apply_to_arbiter_verdict(verdict)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 多因子融合引擎异常，回退到原有逻辑: {exc}")
        # GAP-R：安全边界融合——直接修改verdict的activation_bar和fused_score
        try:
            from src.core.safety_boundary_fusion import (
                get_safety_fusion_engine,
                ContextualDangerFactors,
                ThreatLevel,
            )

            if not self._safety_fusion_initialized:
                self._safety_fusion_initialized = True
            _sfe = get_safety_fusion_engine()
            _night_r = getattr(self, "_cached_night_phase", None)
            _is_late_night = False
            if _night_r and hasattr(_night_r, "value"):
                _is_late_night = self._normalized_night_phase_value() in (
                    "deep_sleep",
                    "light_sleep",
                    "drowsy",
                    "deep_valley",
                )
            _ctx_factors = ContextualDangerFactors(
                is_late_night=_is_late_night,
                bot_is_discussing_sensitive_topic=bool(self._cached_ritual_behavior),
                recent_negative_emotion_spike=self._cached_user_negative_emotion > 50,
                message_contains_url=False,
            )
            _safety_result = _sfe.assess(
                [],
                contextual_factors=_ctx_factors,
                current_arbiter_activation_bar=getattr(verdict, "activation_bar", 0.5),
                current_arbiter_fused_score=getattr(verdict, "fused_score", 0.5),
            )
            _new_bar, _new_score = _sfe.apply_to_verdict(verdict, _safety_result)
            self._cached_safety_assessment = _safety_result.to_dict()
            if _safety_result.overall_level != ThreatLevel.NONE:
                logger.info(
                    f"{self.log_prefix} [GAP-R] 安全评估: {_safety_result.overall_level.label()} "
                    f"bar={_new_bar:.3f} score={_new_score:.3f}"
                )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            _inj_msgs = []
            self.build_planner_injection_prompt(_inj_msgs)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            self.register_cross_engine_outputs()
        except Exception as _e:
            logger.debug(f"异常: {_e}")

        return verdict


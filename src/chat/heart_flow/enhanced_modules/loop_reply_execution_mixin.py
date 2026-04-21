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

class LoopReplyExecutionMixin:
    async def _finalize_reply_settlement(
        self,
        incoming_batch: List,
        relation_result: Dict[str, Any],
        voice_driven_reply: bool,
        planner_triggered: bool,
        did_reply: bool = True,
        behavior_signal: Optional[Dict[str, Any]] = None,
        repetition_signal: Optional[Dict[str, Any]] = None,
        harassment_signal: Optional[Dict[str, Any]] = None,
        proactive_reply: bool = False,
    ) -> None:
        """统一收口回复后的资源、关系、记忆与状态结算。"""
        pre_speaks = float(getattr(self, "_consecutive_speaks", 0.0) or 0.0)
        pre_chatterbox = float(getattr(self, "_chatterbox_penalty", 0.0) or 0.0)
        pre_unanswered = int(getattr(self, "_unanswered_bot_turns", 0) or 0)
        settlement_parts = [
            f"门控={getattr(self, '_last_legacy_gate', 'allow')}",
            f"主动={1 if proactive_reply else 0}",
        ]
        legacy_breakdown = getattr(self, "_last_legacy_breakdown", None)
        if did_reply and isinstance(legacy_breakdown, dict) and legacy_breakdown:
            settlement_parts.append(f"准入={legacy_breakdown.get('reply_readiness', 0):.1f}")
            settlement_parts.append(f"资源扣={legacy_breakdown.get('resource_penalty', 0):.1f}")
            settlement_parts.append(f"关系扣={legacy_breakdown.get('relation_penalty', 0):.1f}")
            settlement_parts.append(f"重复扣={legacy_breakdown.get('repeat_penalty', 0):.1f}")
        self._sync_interest_level_from_desire()
        await self._apply_reply_drain_dynamic(incoming_batch, relation_result)
        await self._update_social_metrics(
            incoming_batch,
            relation_result,
            behavior_signal=behavior_signal,
        )
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            _target_msg = self._get_latest_human_message(incoming_batch)
            if _target_msg:
                _uid = str(getattr(_target_msg, "user_id", "") or "")
                _content = str(
                    getattr(_target_msg, "processed_plain_text", "") or getattr(_target_msg, "content", "") or ""
                ).strip()
                if _uid and _content:
                    _is_ad = bool(incoming_batch and self._is_force_wake_admin(incoming_batch))
                    _tracker = get_emotion_tracker(self.stream_id)
                    _cog_result = await _tracker.process_interaction_with_llm(
                        user_id=_uid,
                        content=_content,
                        is_admin=_is_ad,
                        use_llm=False,
                    )
                    if _cog_result:
                        relation_result["affection"] = float(_cog_result.get("affection", relation_result.get("affection", 0.0)) or 0.0)
                        relation_result["trust_value"] = float(
                            _cog_result.get(
                                "trust_value",
                                _cog_result.get("trust_score", _cog_result.get("trust", relation_result.get("trust_value", 0.0))),
                            )
                            or 0.0
                        )
                        relation_result["annoyance_value"] = float(
                            _cog_result.get(
                                "annoyance_value",
                                _cog_result.get("annoyance", relation_result.get("annoyance_value", 0.0)),
                            )
                            or 0.0
                        )
                        relation_result["trauma_score"] = float(
                            _cog_result.get("trauma_score", relation_result.get("trauma_score", 0.0)) or 0.0
                        )
                        _state_latest = _tracker.get_user_state(_uid, create_if_missing=False)
                        if _state_latest is not None:
                            relation_result["psychological_pressure"] = float(
                                getattr(_state_latest, "psychological_pressure", relation_result.get("psychological_pressure", 0.0))
                                or 0.0
                            )
                        relation_result = self._normalize_relation_snapshot(relation_result)
                        self._last_relation_snapshot.update(relation_result)
                        self._cached_affection_value = float(relation_result.get("affection", 0.0) or 0.0)
                        self._cached_trust_value = float(relation_result.get("trust_value", 0.0) or 0.0)
                        self._cached_annoyance_value = float(relation_result.get("annoyance_value", 0.0) or 0.0)
                        self._cached_pressure_value = float(relation_result.get("psychological_pressure", 0.0) or 0.0)
                        logger.info(
                            f"{self.log_prefix} 🧠 认知动力学 "
                            f"好感{_cog_result.get('affection_delta', 0):+.1f}="
                            f"{_cog_result.get('affection', 0):.1f} "
                            f"信任{_cog_result.get('trust_delta', 0):+.1f}="
                            f"{_cog_result.get('trust_value', 0):.1f} "
                            f"厌烦={_cog_result.get('annoyance', 0):.1f} | "
                            f"{_cog_result.get('experience_record', '')[:60]}"
                        )
        except Exception as _cd_exc:
            logger.warning(f"{self.log_prefix} 认知动力学异常: {_cd_exc}")
        self._on_bot_sent(was_proactive=proactive_reply)
        # 核心模块回复后学习 + 印象 + 技能 + 代谢
        _last_reply_text = ""
        _last_bot_entry = self._latest_recent_bot_utterance()
        if _last_bot_entry:
            _last_reply_text = str(_last_bot_entry.get("text", "") or "")
        self._integrate_post_reply_learning(
            incoming_batch,
            reply_text=_last_reply_text,
            learning_context={
                "behavior_signal": behavior_signal or {},
                "repetition_signal": repetition_signal or {},
                "harassment_signal": harassment_signal or {},
                "relation_result": relation_result or {},
            },
        )
        # ResultGauge 回填面板
        try:
            from src.core.panel_state_adapter import get_panel_adapter

            _adapter = get_panel_adapter()
            _trace_brief = self._decision_trace_brief()
            _reasoning_text = str(getattr(self, "_last_legacy_gate", "") or "")
            if _trace_brief:
                _reasoning_text = f"{_reasoning_text} | {_trace_brief}" if _reasoning_text else _trace_brief
            _action_result = {
                "loop_action_info": {
                    "action_taken": True,
                    "reply_text": _last_reply_text[:500],
                    "taken_time": time.time(),
                    "quote_message": False,
                },
                "loop_plan_info": {
                    "action_result": {
                        "action_type": (
                            "planner_reply" if planner_triggered else ("voice_reply" if voice_driven_reply else "reply")
                        ),
                        "reasoning": _reasoning_text,
                    },
                },
            }
            _result_snap = _adapter.assemble_with_result(
                self._tick_world_snapshot,
                action_result=_action_result,
            )
            self._cached_result_gauge = _result_snap.result
        except Exception as exc:
            logger.debug(f"{self.log_prefix} ResultGauge 回填异常: {exc}")
        # 自适应管线收集（回复成功路径）
        self._integrate_adaptive_collection(incoming_batch, True, _last_reply_text)
        settlement_parts.append(f"连续发言={pre_speaks:.1f}->{getattr(self, '_consecutive_speaks', 0.0):.1f}")
        settlement_parts.append(f"话痨惩罚={pre_chatterbox:.1f}->{getattr(self, '_chatterbox_penalty', 0.0):.1f}")
        settlement_parts.append(f"未回应计数={pre_unanswered}->{getattr(self, '_unanswered_bot_turns', 0)}")
        if behavior_signal:
            settlement_parts.append(f"行为={behavior_signal.get('category', 'neutral')}")
        # 管线预算摘要
        if self._cached_pipeline_summary:
            _remaining_cands = self._cached_pipeline_summary.get("budget_remaining_candidates", "?")
            _remaining_tokens = self._cached_pipeline_summary.get("budget_remaining_tokens", "?")
            settlement_parts.append(f"管线余候选={_remaining_cands} 余token={_remaining_tokens}")
        logger.info(f"{self.log_prefix} 结算账本 " + " ".join(settlement_parts))

    async def _finalize_external_proactive_reply_flow(
        self,
        incoming_batch: List,
        *,
        source: str,
        desire_level: Optional[float] = None,
    ) -> None:
        """为主循环外的主动回复补齐统一结算，避免资源/关系/学习链断裂。"""
        self._prime_external_proactive_trace(source, desire_level=desire_level)
        relation_result = self._resolve_relation_view()
        behavior_signal = self._classify_behavior_signal(incoming_batch)
        self._last_flow_blocker = ""
        await self._finalize_reply_settlement(
            incoming_batch=incoming_batch,
            relation_result=relation_result,
            voice_driven_reply=False,
            planner_triggered=False,
            did_reply=True,
            behavior_signal=behavior_signal,
            proactive_reply=True,
        )
        self._apply_post_reply_state(did_reply=True, reason=source)
        self._emit_flow_decision_summary(source, "reply")
        await self._emit_outcome_summary(True, relation_result)
        self._run_post_action_side_effects(
            did_reply=True,
            incoming_batch=incoming_batch,
            reply_text=str(getattr(self, "_last_reply_content", "") or ""),
            was_proactive=True,
        )

    async def _execute_planned_reply(
        self,
        decision_messages: List,
        planner_decision: Optional[Any],
        force_reply_message: Optional[Any] = None,
        repetition_signal: Optional[Dict[str, Any]] = None,
        decision_context_packet: Optional[Any] = None,
    ) -> bool:
        from src.llm_models.utils_model import bind_stream_context

        # 绑定聊天流ID到当前异步任务，使并发守卫能按流限速
        bind_stream_context(self.stream_id)

        target_message = force_reply_message or self._get_latest_human_message(decision_messages)
        if target_message is None:
            self._last_flow_blocker = "planner缺少可回复目标"
            logger.info(f"{self.log_prefix} 规划器要求回复，但当前没有可回复目标")
            return False

        legacy_gate = str(getattr(self, "_last_legacy_gate", "allow") or "allow")
        force_bypass = bool(force_reply_message is not None or legacy_gate == "force_reply")
        targeted_to_bot = bool(force_bypass or getattr(self, "_cached_targeted_to_bot", False))
        restraint = await self._run_self_restraint_check(
            decision_messages,
            source="planner_reply",
            repetition_signal=self._analyze_repetition_pressure(decision_messages),
            harassment_signal=self._analyze_harassment_pressure(decision_messages),
            force_bypass=force_bypass,
            targeted_to_bot=targeted_to_bot,
            admin_force=bool(getattr(self, "_is_admin_forced", False)),
        )
        if not restraint.get("allow", True):
            self._last_flow_blocker = f"planner自省拦截:{restraint.get('reason', 'skip')}"
            logger.info(f"{self.log_prefix} 🧯 自省闸门拦截 planner 回复: {restraint.get('reason', 'skip')}")
            return False

        target_message = self._select_preferred_reply_message(target_message, list(decision_messages[-10:]))
        self._mark_message_content_processing(target_message)
        try:
            content_plan = str(getattr(planner_decision, "content_plan", "") or "").strip()
            from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

            reply_reason = acquire_reply_coordinator().compose_reply_reason(
                base_reason=getattr(planner_decision, "reason", "统一规划器决定回复") or "统一规划器决定回复",
                content_plan=content_plan,
                legacy_gate=legacy_gate,
            )

            if repetition_signal is None:
                repetition_signal = self._analyze_repetition_pressure(decision_messages)
            if decision_context_packet is None:
                decision_context_packet = self._build_decision_context_packet(
                    list(decision_messages),
                    repetition_signal=repetition_signal,
                )

            extra_parts: List[str] = []
            voice_summary = self._build_voice_execution_summary(getattr(self, "_cached_voice", None), target_message)
            voice_panel = voice_summary.get("panel", "") if isinstance(voice_summary, dict) else ""
            if voice_panel:
                extra_parts.append(voice_panel)
            current_target_block = self._build_current_target_message_block(target_message)
            if current_target_block:
                extra_parts.append(current_target_block)
            if content_plan:
                extra_parts.append(f"[统一规划内容规划] {content_plan}")

            target_user_id = getattr(target_message, "user_id", "") or ""
            user_style_guide = self._get_user_style_guide(target_user_id) if target_user_id else ""
            relation_view = self._resolve_relation_view()
            context_execution_block = self._build_context_execution_block(
                target_message=target_message,
                voice_conclusion=getattr(self, "_cached_voice", None),
                repetition_signal=repetition_signal,
                decision_context_packet=decision_context_packet,
                relation_snapshot=relation_view,
            )
            if context_execution_block:
                extra_parts.append(context_execution_block)
            if legacy_gate == "hesitate":
                extra_parts.append(
                    "[低可回复度约束] 当前只允许一句短回复，优先澄清、确认、轻量接话；"
                    "不要长篇展开，不要主动延伸新话题，不要输出高情绪高承诺表达。"
                )
            from src.chat.replyer.context_block_builder import build_shared_reply_parts

            extra_parts.extend(
                build_shared_reply_parts(
                    self_memory=self._build_self_reply_memory(),
                    continuity_context=self._build_self_continuity_context(),
                    user_style_guide=user_style_guide,
                    persona_hint=self._build_persona_hint(),
                    reply_style_context=self._build_reply_style_context(relation_view),
                    length_hint=self._build_dynamic_length_hint(target_message, user_style_guide),
                    restraint_mode=str(restraint.get("mode", "allow") or "allow"),
                    short_only_text="[自省闸门约束] 你刚刚已经说了不少，这次只允许一句很短的收束/澄清，不要展开。",
                )
            )

            harassment_signal = self._analyze_harassment_pressure(decision_messages)
            from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

            style_route = acquire_reply_coordinator().resolve_style_route(
                delivery_form="",
                target_message=target_message,
                reference_user_name="",
                mention_user_name="",
                fallback_selector=lambda msg, rel, hs: self._decide_reply_style(
                    target_message=msg,
                    relation_snapshot=rel,
                    harassment_signal=hs,
                ),
                relation_view=relation_view,
                harassment_signal=harassment_signal,
                is_bot_message=self._is_bot_message_obj,
            )
            from src.chat.replyer.context_block_builder import append_reply_style

            append_reply_style(extra_parts, style_route)
            # 注入多维状态系统的LLM提示词
            self._inject_dimension_state_prompt(extra_parts)
            extra_info = "\n".join(part for part in extra_parts if part)

            self._emit_reply_generation_summary(
                target_message=target_message,
                style_route=style_route,
                relation_snapshot=relation_view,
                context_execution_block=context_execution_block,
                extra_info=extra_info,
                source="planner",
            )

            from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

            success, llm_response = await acquire_reply_coordinator().generate_reply(
                channel_id=self.stream_id,
                chat_stream=self.chat_stream,
                action_modifier=self.action_modifier,
                action_manager=self.action_manager,
                target_message=target_message,
                reply_reason=reply_reason,
                extra_info=extra_info,
                request_type="planner_reply",
                think_level=1,
            )
            if not success or not llm_response or not llm_response.reply_set:
                self._last_flow_blocker = "planner回复生成失败"
                self._mark_message_content_deferred(target_message, "planner_generation_failed")
                logger.warning(f"{self.log_prefix} 规划回复生成失败")
                return False

            response_set = llm_response.reply_set
            selected_expressions = llm_response.selected_expressions
            cycle_timers: Dict[str, float] = {}
            thinking_id = f"planner_{int(time.time() * 1000)}"
            loop_info, reply_text, _, reply_trace_meta = await self._send_and_store_reply(
                response_set=response_set,
                cycle_timers=cycle_timers,
                action_message=target_message,
                thinking_id=thinking_id,
                actions="planner_reply",
                selected_expressions=selected_expressions,
                quote_message=bool(style_route.get("quote_message", False)),
                pre_send_risk_note=(
                    f"planner annoyance={relation_view.get('annoyance_value', 0)} "
                    f"pressure={relation_view.get('psychological_pressure', 0)}"
                ),
                pre_send_audit_label="planner",
            )
            if not (bool(loop_info) or bool(str(reply_text or "").strip())):
                self._last_flow_blocker = "planner回复发送失败"
                self._mark_message_content_deferred(target_message, "planner_send_failed")
                return False
            await self._finalize_sent_reply(
                reply_text=reply_text,
                loop_info=loop_info,
                target_message=target_message,
                reply_reason=reply_reason,
                relation_view=relation_view,
                llm_response=llm_response,
                was_proactive=False,
                action_name="planner_reply",
                quality=0.8,
                audit_label="planner",
                reply_trace_meta=reply_trace_meta,
            )
            return True
        except Exception as exc:
            self._last_flow_blocker = f"planner异常:{type(exc).__name__}"
            self._mark_message_content_deferred(target_message, "planner_reply_exception")
            logger.error(f"{self.log_prefix} 规划回复执行失败: {exc}")
            return False

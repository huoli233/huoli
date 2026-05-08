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

class ProactiveReactiveFlowMixin:
    @staticmethod
    def _build_forced_reply_fallback_text(target_message: Any) -> str:
        raw_text = (
            getattr(target_message, "processed_plain_text", "")
            or getattr(target_message, "plain_text", "")
            or getattr(target_message, "content", "")
            or ""
        )
        text = re.sub(r"\s+", " ", str(raw_text or "")).strip()
        if not text:
            return "看到了。"
        return "看到了，怎么了？"

    def _build_forced_reply_fallback_response(self, target_message: Any, failure_reason: str) -> Any:
        from src.common.data_models.llm_data_model import LLMGenerationDataModel
        from src.common.data_models.message_data_model import ReplySetModel

        fallback_text = self._build_forced_reply_fallback_text(target_message)
        reply_set = ReplySetModel()
        reply_set.add_text_content(fallback_text)
        return LLMGenerationDataModel(
            content=fallback_text,
            model="local_force_reply_fallback",
            selected_expressions=[],
            reply_set=reply_set,
            processed_output=[fallback_text],
            timing={"fallback_reason": str(failure_reason or "forced_reply_generation_failed")},
        )

    async def _execute_voice_driven_reply(
        self,
        voice_conclusion: Any,
        incoming_batch: List,
        force_reply_message: Optional[Any] = None,
    ) -> bool:
        """
        执行内心驱动回复 - 内心独白决定回复时直接生成回复，跳过规划器

        这是真正的主动行为：内心独白已经决定了要回复，直接生成回复。
        """
        target_message = None
        try:
            from src.llm_models.utils_model import bind_stream_context

            # 绑定聊天流ID到当前异步任务，使并发守卫能按流限速
            bind_stream_context(self.stream_id)

            desire_level = getattr(voice_conclusion, "reply_desire_level", 5)
            thinking = getattr(voice_conclusion, "thinking", "")

            logger.info(
                f"{self.log_prefix} 💭 开始执行: 欲望等级={desire_level}, 思考={thinking[:50] if thinking else '无'}"
            )

            target_message = force_reply_message
            if target_message is None:
                for msg in reversed(incoming_batch[-10:]):
                    if not self._is_bot_message_obj(msg):
                        target_message = msg
                        break
            target_message = self._select_preferred_reply_message(target_message, list(incoming_batch[-10:]))
            if target_message is None:
                self._last_flow_blocker = "voice缺少可回复目标"
                logger.info(f"{self.log_prefix} 💭 内心驱动回复缺少目标消息，转为观察")
                return False

            legacy_gate = str(getattr(self, "_last_legacy_gate", "allow") or "allow")
            force_bypass = bool(force_reply_message is not None or legacy_gate == "force_reply")
            targeted_to_bot = bool(force_bypass or getattr(self, "_cached_targeted_to_bot", False))
            restraint = await self._run_self_restraint_check(
                incoming_batch,
                source="voice_driven_reply",
                repetition_signal=self._analyze_repetition_pressure(incoming_batch),
                harassment_signal=self._analyze_harassment_pressure(incoming_batch),
                force_bypass=force_bypass,
                targeted_to_bot=targeted_to_bot,
                admin_force=bool(getattr(self, "_is_admin_forced", False)),
            )
            if not restraint.get("allow", True):
                self._last_flow_blocker = f"voice自省拦截:{restraint.get('reason', 'skip')}"
                logger.info(f"{self.log_prefix} 🧯 自省闸门拦截 voice 回复: {restraint.get('reason', 'skip')}")
                return False
            self._mark_message_content_processing(target_message)
            force_generation_fallback = bool(force_bypass or getattr(self, "_is_admin_forced", False))

            # 获取目标用户的风格指导
            user_style_guide = ""
            if target_message and not force_generation_fallback:
                target_user_id = getattr(target_message, "user_id", "")
                if target_user_id:
                    user_style_guide = self._get_user_style_guide(target_user_id)

            voice_summary = self._build_voice_execution_summary(voice_conclusion, target_message)

            from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

            reply_reason = acquire_reply_coordinator().compose_reply_reason(
                base_reason=f"内心驱动回复: 欲望等级={desire_level}",
                voice_reason=voice_summary.get("reason", ""),
            )

            extra_info_parts: List[str] = []
            panel_text = voice_summary.get("panel", "")
            if panel_text:
                extra_info_parts.append(panel_text)
            current_target_block = self._build_current_target_message_block(target_message)
            if current_target_block:
                extra_info_parts.append(current_target_block)
            relation_view = self._resolve_relation_view()
            context_execution_block = ""
            if force_generation_fallback:
                extra_info_parts.append("[直接快回] 一句短口语，直接回应当前消息；不做长篇解释，不二次改写，不补充追发。")
                self._inject_fallback_soul_state(extra_info_parts)
            else:
                decision_context_packet = self._build_decision_context_packet(
                    list(incoming_batch),
                    repetition_signal=self._analyze_repetition_pressure(incoming_batch),
                )
                context_execution_block = self._build_context_execution_block(
                    target_message=target_message,
                    voice_conclusion=voice_conclusion,
                    repetition_signal=self._analyze_repetition_pressure(incoming_batch),
                    decision_context_packet=decision_context_packet,
                    relation_snapshot=relation_view,
                )
                if context_execution_block:
                    extra_info_parts.append(context_execution_block)
                from src.chat.replyer.context_block_builder import build_shared_reply_parts
                self_reference_parts = self._build_self_reference_parts(target_message)

                extra_info_parts.extend(
                    build_shared_reply_parts(
                        self_memory=self_reference_parts.get("self_memory", ""),
                        continuity_context=self_reference_parts.get("continuity_context", ""),
                        user_style_guide=user_style_guide,
                        persona_hint="",
                        reply_style_context=self._build_reply_style_context(relation_view),
                        length_hint=self._build_dynamic_length_hint(target_message, user_style_guide),
                        restraint_mode=str(restraint.get("mode", "allow") or "allow"),
                        short_only_text="[自省闸门约束] 你已经连续说了不少，这次只准一句短话，不展开，不补充，不连发。",
                    )
                )

            takeover_thought = getattr(self, "_takeover_decision", None)
            if takeover_thought:
                _tk_thought = takeover_thought.get("thought", "") or ""
                _tk_action = takeover_thought.get("action", "") or ""
                if _tk_thought:
                    extra_info_parts.append(f"[接管意图] {_tk_action}: {_tk_thought}")
                self._takeover_decision = None
                self._takeover_action = None

            harassment_signal = self._analyze_harassment_pressure(incoming_batch)
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

            append_reply_style(extra_info_parts, style_route)
            if not force_generation_fallback:
                # 注入多维状态系统的LLM提示词
                self._inject_dimension_state_prompt(extra_info_parts)
                _diversity_warn = self._check_reply_diversity()
                if _diversity_warn:
                    extra_info_parts.append(_diversity_warn)
                _meme_quick = self._build_meme_injection()
                if _meme_quick:
                    extra_info_parts.append(_meme_quick)
            extra_info = build_reply_context_block(
                recent_context="",
                relevant_context="",
                extra_info="\n".join(part for part in extra_info_parts if part),
                recent_reply_guard="",
            )
            extra_info = self._ensure_soul_data_in_extra_info(extra_info)
            _key_lines = [
                line
                for line in (extra_info or "").split("\n")
                if self._contains_soul_data(line)
            ]
            if _key_lines:
                logger.info(f"{self.log_prefix} 🧠 传入LLM的灵魂数据:\n" + "\n".join(_key_lines[:8]))
            else:
                logger.warning(
                    f"{self.log_prefix} ⚠️ 传入LLM的extra_info中没有灵魂数据！extra_info长度={len(extra_info or '')}"
                )
            self._emit_reply_generation_summary(
                target_message=target_message,
                style_route=style_route,
                relation_snapshot=relation_view,
                context_execution_block=context_execution_block,
                extra_info=extra_info,
                source="voice_driven",
            )

            from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

            generation_failure_reason = "voice_generation_failed"
            force_reply_timeout = _rt_float("heartfc_force_reply_generation_timeout_seconds", 35.0)
            if force_generation_fallback:
                force_reply_timeout = _rt_float("heartfc_direct_fast_reply_generation_timeout_seconds", 8.0)
                logger.debug(f"{self.log_prefix} ⚡ 直接快回生成预算={force_reply_timeout:.1f}s")
            reply_generation = acquire_reply_coordinator().generate_reply(
                channel_id=self.stream_id,
                chat_stream=self.chat_stream,
                action_modifier=self.action_modifier,
                action_manager=self.action_manager,
                target_message=target_message,
                reply_reason=reply_reason,
                extra_info=extra_info,
                request_type="voice_driven_reply",
                think_level=1,
                fast_path=force_generation_fallback,
                enable_splitter=not force_generation_fallback,
                enable_chinese_typo=not force_generation_fallback,
            )
            if force_generation_fallback:
                try:
                    success, llm_response = await asyncio.wait_for(
                        reply_generation,
                        timeout=force_reply_timeout,
                    )
                except asyncio.TimeoutError:
                    success = False
                    llm_response = None
                    generation_failure_reason = "direct_fast_reply_generation_timeout"
                    logger.warning(f"{self.log_prefix} 💭 直接快回生成超时({force_reply_timeout:.1f}s)，启用本地短兜底")
            else:
                success, llm_response = await reply_generation

            if not success or not llm_response or not llm_response.reply_set:
                if force_generation_fallback:
                    llm_response = self._build_forced_reply_fallback_response(target_message, generation_failure_reason)
                    success = True
                    logger.warning(
                        f"{self.log_prefix} 💭 强制回复生成失败，已启用本地短兜底: {generation_failure_reason}"
                    )
                else:
                    self._last_flow_blocker = "voice回复生成失败"
                    self._mark_message_content_deferred(target_message, "voice_generation_failed")
                    logger.warning(f"{self.log_prefix} 💭 回复生成失败")
                    return False

            response_set = llm_response.reply_set
            selected_expressions = llm_response.selected_expressions

            cycle_timers = {}
            thinking_id = f"voice_driven_{int(time.time() * 1000)}"

            loop_info, reply_text, _, reply_trace_meta = await self._send_and_store_reply(
                response_set=response_set,
                action_message=target_message,
                cycle_timers=cycle_timers,
                thinking_id=thinking_id,
                actions=[],
                selected_expressions=selected_expressions,
                quote_message=bool(style_route.get("quote_message", False)),
                pre_send_risk_note=(
                    f"voice annoyance={relation_view.get('annoyance_value', 0)} "
                    f"pressure={relation_view.get('psychological_pressure', 0)}"
                ),
                pre_send_audit_label="voice",
            )
            if not (bool(loop_info) or bool(str(reply_text or "").strip())):
                self._last_flow_blocker = "voice回复发送失败"
                self._mark_message_content_deferred(target_message, "voice_send_failed")
                return False

            await self._finalize_sent_reply(
                reply_text=reply_text,
                loop_info=loop_info,
                target_message=target_message,
                reply_reason=reply_reason,
                relation_view=relation_view,
                llm_response=llm_response,
                was_proactive=False,
                action_name="voice_driven_reply",
                quality=0.85,
                audit_label="voice",
                reply_trace_meta=reply_trace_meta,
            )
            logger.info(f"{self.log_prefix} 💭 成功发送: {reply_text[:50]}...")
            self._spawn(self._update_user_impression_after_reply(reply_text))
            return True

        except Exception as exc:
            self._last_flow_blocker = f"voice异常:{type(exc).__name__}"
            self._mark_message_content_deferred(
                target_message or self._get_latest_human_message(incoming_batch),
                "voice_reply_exception",
            )
            logger.error(f"{self.log_prefix} 💭 执行失败: {exc}")
            return False

    async def _proactive_background_loop(self):
        """主动感兴趣通道 —— 独立后台常驻任务，与被动监听通道并行运行

        设计原则：
          - 不依赖消息到达触发，而是持续自主评估"我是否想说点什么"
          - 有自己的独立感知→独白→决策→回复管线
          - 与被动通道共享同一套引擎（仪表盘/代谢/存在感等），但独立决策
          - 被动通道处理消息时不会被阻塞（asyncio协作式多任务）

        运行节奏：
          - 基础评估间隔：8-15秒（根据精力动态调整）
          - 深度评估（含LLM调用）：仅在评估通过后触发
          - 冷却期：主动发言后30-60秒内不再主动触发
        """
        _base_interval = _rt_float("heartfc_background_proactive_base_interval_seconds", 12.0)
        _min_interval = _rt_float("heartfc_background_proactive_min_interval_seconds", 6.0)
        _max_interval = _rt_float("heartfc_background_proactive_max_interval_seconds", 30.0)
        _cooldown_min = _rt_float("heartfc_background_proactive_success_cooldown_min_seconds", 90.0)
        _cooldown_max = _rt_float("heartfc_background_proactive_success_cooldown_max_seconds", 180.0)
        _cooldown_until = 0.0
        _consecutive_fails = 0
        _loop_count = 0
        logger.info(f"{self.log_prefix} 🔔 主动通道后台循环启动，基础间隔={_base_interval}s")
        try:
            while self._proactive_running and self.running:
                _loop_count += 1
                _now_loop = time.time()
                if _now_loop < _cooldown_until:
                    await asyncio.sleep(min(2.0, _cooldown_until - _now_loop))
                    continue
                _energy_mod = 1.0
                _ms = getattr(self, "_cached_metabolism_state", None)
                if _ms:
                    _er = float(getattr(_ms, "energy_ratio", getattr(_ms, "chat_energy_ratio", 1.0)) or 1.0)
                    if _er < 0.3:
                        _energy_mod = 2.5
                        _base_interval = min(_max_interval, _base_interval + 1.0)
                    elif _er > 0.8:
                        _energy_mod = 0.7
                        _base_interval = max(_min_interval, _base_interval - 0.5)
                else:
                    _base_interval = _rt_float("heartfc_background_proactive_base_interval_seconds", 12.0)
                _interval = max(
                    _min_interval,
                    min(_max_interval, _base_interval * _energy_mod),
                )
                try:
                    _dv = self._get_dashboard_verdict()
                    _dv_urgency = str(_dv.get("urgency", "") or "")
                    if not _dv.get("process", True):
                        await asyncio.sleep(_interval)
                        continue
                    if _dv_urgency in ("不回复", "强制休息"):
                        await asyncio.sleep(_interval * 1.5)
                        continue
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
                _silence = 0.0
                try:
                    from src.chat.proactive.silence_watcher import (
                        get_quiet_monitor,
                    )

                    _silence = get_quiet_monitor().measure_silence_sec(self.stream_id)
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
                if _silence < 3.0 and _loop_count > 3:
                    await asyncio.sleep(_interval * 0.5)
                    continue
                if (time.time() - self._last_bot_reply_ts) < 5.0:
                    await asyncio.sleep(_interval * 0.3)
                    continue
                _should_act = False
                _act_reason = ""
                try:
                    _should_act, _act_reason = await self._evaluate_proactive_opportunity(
                        now=_now_loop,
                        silence_sec=_silence,
                        is_background=True,
                    )
                except Exception as exc:
                    logger.debug(f"{self.log_prefix} 主动通道评估异常: {exc}")
                if not _should_act:
                    _consecutive_fails = 0
                    await asyncio.sleep(_interval)
                    continue
                logger.info(f"{self.log_prefix} 🔔 主动通道触发: {_act_reason} (静默={_silence:.0f}s)")
                try:
                    _acted = await self._execute_proactive_action(
                        reason=_act_reason,
                        silence_sec=_silence,
                    )
                    if _acted:
                        # 主动发言成功后短冷却，避免连续自言自语，同时保留主动感。
                        _cooldown_until = time.time() + random.uniform(
                            min(_cooldown_min, _cooldown_max),
                            max(_cooldown_min, _cooldown_max),
                        )
                        _consecutive_fails = 0
                        _base_interval = max(_min_interval, _base_interval - 1.0)
                    else:
                        _consecutive_fails += 1
                        _base_interval = min(_max_interval, _base_interval + 0.5)
                except Exception as exc:
                    logger.warning(f"{self.log_prefix} 主动通道执行失败: {exc}")
                    _consecutive_fails += 1
                if _consecutive_fails >= 3:
                    _backoff = min(30.0, 5.0 * _consecutive_fails)
                    logger.debug(f"{self.log_prefix} 主动通道连续{_consecutive_fails}次失败，退避{_backoff:.0f}s")
                    await asyncio.sleep(_backoff)
                    _consecutive_fails = 0
                else:
                    await asyncio.sleep(_interval)
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} 🔔 主动通道后台循环被取消")
        except Exception as exc:
            logger.error(f"{self.log_prefix} 🔔 主动通道后台循环异常退出: {exc}")

    async def _evaluate_proactive_opportunity(
        self, *, now: float, silence_sec: float, is_background: bool = False
    ) -> Tuple[bool, str]:
        """评估是否应该主动发言（轻量级预检，不调LLM）

        Args:
            now: 当前时间戳
            silence_sec: 静默时长
            is_background: 是否来自后台主动通道（后台通道有更宽松的触发条件）

        Returns:
            (should_act, reason) 是否应该主动 + 原因描述
        """
        if not hasattr(self, "_last_idle_proactive_ts"):
            self._last_idle_proactive_ts = 0.0
        if is_background:
            startup_guard_until = float(getattr(self, "_proactive_startup_grace_until", 0.0) or 0.0)
            if now < startup_guard_until:
                return False, "启动保护期"
            _min_silence = _rt_float("heartfc_background_proactive_min_silence_seconds", 45.0)
            if silence_sec < _min_silence:
                return False, f"静默不足({silence_sec:.0f}s/{_min_silence:.0f}s)"
        _cooldown = (
            self._IDLE_PROACTIVE_COOLDOWN_SEC
            if not is_background
            else _rt_float("heartfc_background_proactive_cooldown_seconds", 90.0)
        )
        if (now - self._last_idle_proactive_ts) < _cooldown:
            return False, "冷却中"
        if is_background:
            _bg_block_reason = self._background_proactive_block_reason()
            if _bg_block_reason:
                return False, _bg_block_reason
        if self._cached_night_phase is not None:
            _np = self._cached_night_phase
            _np_name = getattr(_np, "name", str(_np)) if _np else ""
            if "DEEP" in _np_name or "VALLEY" in _np_name or "BURNED" in _np_name:
                return False, "夜间冻结"
            if "DROWSY" in _np_name or "LIGHT" in _np_name:
                _peek_allowed = False
                try:
                    from src.core.night_cycle_system import get_night_cycle

                    _ncs_peek2 = get_night_cycle(self.stream_id)
                    _peek_allowed = _ncs_peek2.evaluate_sleep_peek()
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
                if not _peek_allowed:
                    try:
                        _d6 = EnergyChainDimension.get_instance()
                        _nm = _d6._get_night_mode(self.stream_id)
                        if _nm and _nm.prob_multiplier < 0.3:
                            return False, "浅睡无窥屏窗"
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
        if self._should_back_off_idle_proactive():
            return False, f"连续{self._unanswered_bot_turns}次未获回应"
        try:
            from src.modules.trauma.trauma_system import get_trauma_system

            _ts = get_trauma_system().get_state()
            if _ts.stress_accumulation > 8.0 or _ts.inner_chaos_level > 8.0:
                return False, f"压力过高(stress={_ts.stress_accumulation:.1f})"
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        governor_verdict = self._evaluate_behavior_governor(
            incoming_batch=[],
            silence_sec=silence_sec,
            requested_mode="proactive",
            is_background=is_background,
        )
        if not governor_verdict.allow_generation or governor_verdict.reply_mode != "proactive":
            return False, self._summarize_behavior_governor(governor_verdict)
        _scheduled_event = None
        if is_background:
            try:
                _scheduled_event = self._check_background_event_schedule()
            except Exception as _event_exc:
                logger.debug(f"{self.log_prefix} 后台事件调度检查失败: {_event_exc}")
            if _scheduled_event:
                self._last_background_event = _scheduled_event
                _event_type = str(_scheduled_event.get("type", "") or "")
                _event_priority = float(_scheduled_event.get("priority", 0.0) or 0.0)
                if _event_priority >= 0.18:
                    return True, f"后台事件触发:{_event_type}(priority={_event_priority:.2f})"
        # 优先尝试 IntegrationHub (async，在此处 await)
        _hub_succeeded = False
        try:
            from src.chat.proactive.proactive_integration_hub import (
                get_proactive_integration_hub,
            )

            _hub = get_proactive_integration_hub()
            _ss = getattr(self, "_cached_self_state", None)
            _hub_self_state = {}
            if _ss:
                _hub_self_state = {
                    "mood": getattr(_ss, "mood", 0.5),
                    "trauma": getattr(_ss, "trauma_score", 0),
                    "fatigue": getattr(_ss, "mental_fatigue", 0),
                }
            _hub_relation = getattr(self, "_last_relation_snapshot", {}) or {}
            _hub_msgs = getattr(self, "_last_raw_messages", []) or []
            _hub_verdict, _ = await _hub.evaluate_proactive(
                channel_id=self.stream_id,
                raw_messages=_hub_msgs,
                self_state=_hub_self_state,
                relation_state=_hub_relation,
            )
            if _hub_verdict:
                _hub_succeeded = True
                # R2正向激励调整：Hub路径也应用情感驱动降低activation_bar
                self._apply_r2_incentives(_hub_verdict, silence_sec)
                _bd = getattr(_hub_verdict, "breakdown", {}) or {}
                _bd_str = " ".join(f"{k}={v:.2f}" for k, v in _bd.items() if isinstance(v, (int, float)))
                if _hub_verdict.should_proceed:
                    logger.info(
                        f"{self.log_prefix} 整合中心决策通过: "
                        f"融合分={_hub_verdict.fused_score:.3f} bar={_hub_verdict.activation_bar:.2f} | "
                        f"{_bd_str} | {_hub_verdict.rationale}"
                    )
                    return True, _hub_verdict.rationale or "整合中心通过"
                logger.debug(
                    f"{self.log_prefix} 整合中心否决: "
                    f"融合分={_hub_verdict.fused_score:.3f} bar={_hub_verdict.activation_bar:.2f} | {_bd_str}"
                )
                return False, _hub_verdict.rationale or "整合中心否决"
        except Exception as _hub_err:
            logger.debug(f"{self.log_prefix} 整合中心评估回退: {_hub_err}")
        # 回退到同步主动决策采集评估
        if not _hub_succeeded:
            proactive_decision = self._evaluate_proactive_decision(silence_sec)
            _abd = getattr(proactive_decision, "breakdown", {}) or {}
            _abd_str = " ".join(f"{k}={v:.2f}" for k, v in _abd.items() if isinstance(v, (int, float)))
            if not proactive_decision.should_proceed:
                logger.debug(
                    f"{self.log_prefix} 主动决策否决: "
                    f"融合分={proactive_decision.fused_score:.3f} bar={proactive_decision.activation_bar:.2f} | {_abd_str}"
                )
                return False, proactive_decision.rationale or "决策器否决"
            logger.info(
                f"{self.log_prefix} 主动决策通过: "
                f"融合分={proactive_decision.fused_score:.3f} bar={proactive_decision.activation_bar:.2f} | {_abd_str}"
            )
            return True, proactive_decision.rationale or "决策器通过"
        return False, "评估未完成"

    async def _execute_proactive_action(self, *, reason: str, silence_sec: float) -> bool:
        """执行主动回复动作（深度执行，可能含LLM调用）

        Returns:
            True 表示成功发送了主动回复
        """
        now_act = time.time()
        await self._update_trauma_system_state([])
        emotion_snapshot = self._update_emotion_state(now_act, relation_result={})
        if not emotion_snapshot:
            return False
        awareness_snapshot = None
        ambient_info = self._sample_channel_ambient()
        voice_conclusion = None
        _run_p = self._should_run_perception(now_act)
        _run_v = self._should_run_voice(
            now_act,
            [],
            None,
            is_proactive=True,
            source="background_proactive",
            behavior_verdict=getattr(self, "_last_behavior_governor_verdict", None),
        )
        _jobs: list = []
        _keys: list = []
        if _run_p:
            _jobs.append(self._invoke_perception([], now_act))
            _keys.append("p")
        if _run_v:
            _jobs.append(self._invoke_autonomous_voice("主动规划链路触发"))
            _keys.append("v")
        if _jobs:
            try:
                _stage_timeout = _parallel_stage_timeout()
                _res = await asyncio.wait_for(
                    asyncio.gather(*_jobs, return_exceptions=True),
                    timeout=_stage_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"{self.log_prefix} 💤 主动规划链路感知/独白超时({_stage_timeout:.0f}s)，跳过本轮主动发言"
                )
                self._last_idle_proactive_ts = now_act
                return False
            _map = dict(zip(_keys, _res, strict=True))
            if _run_p:
                _ip = _map.get("p")
                if isinstance(_ip, BaseException):
                    logger.warning(f"{self.log_prefix} 主动规划感知异常: {_ip}，沿用已有感知缓存")
                elif _ip is not None:
                    awareness_snapshot = _ip
                    self._cached_awareness = awareness_snapshot
                    self._last_perception_ts = now_act
            if _run_v:
                _iv = _map.get("v")
                if isinstance(_iv, BaseException):
                    logger.warning(f"{self.log_prefix} 主动规划独白异常: {_iv}，沿用已有独白缓存")
                elif _iv is not None:
                    voice_conclusion = _iv
                    self._cached_voice = voice_conclusion
                    self._last_voice_ts = now_act
                    self._last_proactive_voice_ts = now_act
        if voice_conclusion is None:
            voice_conclusion = self._cached_voice
        # R4修复：提取内心独白中的意图id，供闭环追踪使用
        _proactive_intent_id = ""
        if voice_conclusion is not None and hasattr(voice_conclusion, "dominant_unfinished_intent"):
            _dom = getattr(voice_conclusion, "dominant_unfinished_intent", None)
            if _dom and isinstance(_dom, dict):
                _proactive_intent_id = str(_dom.get("intent_id", "") or "")
        self._last_proactive_intent_id = _proactive_intent_id
        # 从 VoiceVerdict 对象提取思考文本
        thought_text = ""
        if voice_conclusion is not None:
            if hasattr(voice_conclusion, "thinking"):
                thought_text = voice_conclusion.thinking
            elif isinstance(voice_conclusion, str):
                thought_text = voice_conclusion
        if not thought_text:
            thought_text = "感觉有点想说话"

        # ── 流程规划器决策：是否适合主动发言 + 发言策略 ──
        _proactive_plan = None
        try:
            from src.chat.heart_flow.flow_planner import acquire_flow_planner

            _mood_label = "平静"
            if emotion_snapshot is not None:
                _mood_label = str(
                    getattr(emotion_snapshot, "mood_label", "")
                    or getattr(emotion_snapshot, "description", "")
                    or "平静"
                )
            _recent_topic = ""
            # 从最近的 bot 发言和频道氛围构建真实话题上下文
            _bot_recent = self._recent_bot_texts()
            if _bot_recent:
                _recent_topic = _bot_recent[-1][:100]
            elif thought_text and thought_text != "感觉有点想说话":
                _recent_topic = thought_text[:100]
            # 构建 bot 最近发言摘要，让规划器知道之前聊了什么
            _bot_context = ""
            if _bot_recent:
                _ctx_lines = [f"- {t[:80]}" for t in _bot_recent[-3:]]
                _bot_context = "你最近说过的话:\n" + "\n".join(_ctx_lines)
            # 从内心独白导出最近的观察记忆（看过什么、想过什么、为什么沉默）
            _observation_ctx = ""
            try:
                from src.chat.heart_flow.inner_voice import get_self_dialogue_engine

                _obs_engine = get_self_dialogue_engine(self.stream_id)
                _observation_ctx = _obs_engine.export_observation_context()
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
            if _observation_ctx:
                _bot_context = f"{_bot_context}\n{_observation_ctx}" if _bot_context else _observation_ctx
            # 构建近期对话记录（带用户名），让规划器知道群里谁说了什么
            _dialogue_lines = []
            _recent_msgs = []
            try:
                _recent_msgs = message_api.get_messages_by_time_in_chat(
                    chat_id=self.stream_id,
                    start_time=time.time() - 600,
                    end_time=time.time(),
                    limit=15,
                    limit_mode="latest",
                )
                for _rm in _recent_msgs:
                    _rm_nick = str(
                        getattr(_rm, "user_nickname", "")
                        or getattr(_rm, "user_cardname", "")
                        or getattr(_rm, "user_id", "")
                        or "未知"
                    )
                    _rm_text = str(
                        getattr(_rm, "processed_plain_text", "") or getattr(_rm, "plain_text", "") or ""
                    ).strip()
                    if _rm_text:
                        _dialogue_lines.append(f"{_rm_nick}: {_rm_text[:120]}")
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
            _recent_dialogue = "\n".join(_dialogue_lines) if _dialogue_lines else ""
            # ── 构建社交融入三维上下文 ──
            # A. 群聊氛围感知
            _atmosphere_parts = []
            if ambient_info:
                _vit = float(ambient_info.get("vitality", 0) or 0)
                _wear = float(ambient_info.get("weariness", 0) or 0)
                _vex = float(ambient_info.get("vexation", 0) or 0)
                _cat = str(ambient_info.get("category", "未知") or "未知")
                _atmosphere_parts.append(f"频道氛围类型: {_cat}")
                if _vit > 60:
                    _atmosphere_parts.append("群里气氛挺活跃的")
                elif _vit < 25:
                    _atmosphere_parts.append("群里比较冷清")
                if _wear > 50:
                    _atmosphere_parts.append("大家似乎有点疲倦")
                if _vex > 40:
                    _atmosphere_parts.append("气氛有些烦躁")
            # 统计近期活跃用户
            _active_users = {}
            for _rm in _recent_msgs:
                _uid = str(getattr(_rm, "user_id", "") or "")
                _nick = str(getattr(_rm, "user_nickname", "") or getattr(_rm, "user_cardname", "") or _uid)
                if self._is_human_message_obj(_rm) and _nick:
                    _active_users[_uid] = _nick
            if _active_users:
                _atmosphere_parts.append(f"最近参与聊天的人: {', '.join(_active_users.values())}")
            _atmosphere_ctx = "\n".join(_atmosphere_parts) if _atmosphere_parts else ""
            # B. 人际关系地图（对每个活跃用户的关系感知）
            _relationship_parts = []
            if _active_users:
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker,
                    )

                    _emo_tracker = get_emotion_tracker(self.stream_id)
                    for _uid, _nick in _active_users.items():
                        _user_state = _emo_tracker.get_user_state(_uid, create_if_missing=False)
                        if _user_state is None:
                            _relationship_parts.append(f"{_nick}: 不太熟，没什么印象")
                            continue
                        _r_trust = float(getattr(_user_state, "trust_accumulation", 0) or 0)
                        _r_annoy = float(getattr(_user_state, "annoyance", 0) or 0)
                        _r_intimacy = float(getattr(_user_state, "intimacy", 0) or 0)
                        _r_fondness = float(getattr(_user_state, "fondness", 0) or 0)
                        _r_blocked = bool(getattr(_user_state, "is_blocked", False))
                        _r_parts = []
                        if _r_blocked:
                            _r_parts.append("已屏蔽")
                        elif _r_intimacy > 60:
                            _r_parts.append("很熟")
                        elif _r_intimacy > 30:
                            _r_parts.append("有一定了解")
                        else:
                            _r_parts.append("不太熟")
                        if _r_fondness > 50:
                            _r_parts.append("挺喜欢他的")
                        elif _r_fondness < -20:
                            _r_parts.append("不太喜欢他")
                        if _r_annoy > 60:
                            _r_parts.append("最近对他很烦")
                        elif _r_annoy > 30:
                            _r_parts.append("有点烦他")
                        if _r_trust > 40:
                            _r_parts.append("比较信任")
                        _relationship_parts.append(
                            f"{_nick}: {'，'.join(_r_parts)}" if _r_parts else f"{_nick}: 印象一般"
                        )
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
            _relationship_ctx = (
                "你对群里这些人的感觉:\n" + "\n".join(_relationship_parts) if _relationship_parts else ""
            )
            # C. 自身社交状态
            _self_state_parts = []
            _ws = getattr(self, "_tick_world_snapshot", None)
            if _ws is not None:
                _res = getattr(_ws, "self_resources", None)
                if _res is not None:
                    _chat_pct = int(getattr(_res, "chat_ratio", lambda: 0.5)() * 100)
                    _think_pct = int(getattr(_res, "thinking_ratio", lambda: 0.5)() * 100)
                    _boredom = float(getattr(_res, "boredom", 0) or 0)
                    _loneliness = float(getattr(_res, "loneliness", 0) or 0)
                    _social_desire = float(getattr(_res, "social_desire", 0) or 0)
                    _consec = int(getattr(_res, "consecutive_replies", 0) or 0)
                    _self_state_parts.append(f"聊天精力: {_chat_pct}%，思考精力: {_think_pct}%")
                    if _boredom > 50:
                        _self_state_parts.append("你现在挺无聊的")
                    if _loneliness > 50:
                        _self_state_parts.append("你有点孤独，想找人聊天")
                    if _social_desire > 60:
                        _self_state_parts.append("社交欲望很强")
                    elif _social_desire < 20:
                        _self_state_parts.append("不太想社交")
                    if _consec > 3:
                        _self_state_parts.append(f"你已经连续主动说了{_consec}次，可能该歇歇了")
            _self_state_ctx = "\n".join(_self_state_parts) if _self_state_parts else ""
            _proactive_plan = await acquire_flow_planner().generate_proactive_plan(
                channel_id=self.stream_id,
                quiet_seconds=silence_sec,
                recent_topic=_recent_topic,
                mood=_mood_label,
                recent_bot_context=_bot_context,
                recent_dialogue=_recent_dialogue,
                atmosphere_context=_atmosphere_ctx,
                relationship_context=_relationship_ctx,
                self_state_context=_self_state_ctx,
            )
            if _proactive_plan is None:
                logger.info(f"{self.log_prefix} [流程规划] 规划器否决主动发言")
                return False
            if _proactive_plan.reply_strategy:
                thought_text = _proactive_plan.reply_strategy
            logger.info(
                f"{self.log_prefix} [流程规划] 主动策略: "
                f"situation={_proactive_plan.situation.value} "
                f"emotion={_proactive_plan.emotion_hint} "
                f"length={_proactive_plan.suggested_length} "
                f"delivery={_proactive_plan.delivery_form} "
                f"ref={_proactive_plan.reference_user_name} "
                f"mention={_proactive_plan.mention_user_name}"
            )
        except Exception as _fp_exc:
            logger.debug(f"{self.log_prefix} 流程规划器异常(降级): {_fp_exc}")

        # 从内心独白提取实际 desire，不再硬编码
        _actual_desire = 5
        if voice_conclusion is not None:
            _actual_desire = int(getattr(voice_conclusion, "reply_desire_level", 5) or 5)
        # 从规划器结果提取话题和情绪
        _proactive_topic = ""
        _proactive_emotion = ""
        _delivery_form = "standalone"
        _mention_user_name = ""
        _reference_user_name = ""
        _scheduled_event = getattr(self, "_last_background_event", None) or {}
        if _proactive_plan is not None:
            _proactive_topic = str(getattr(_proactive_plan, "extra_notes", "") or "")
            _proactive_emotion = str(getattr(_proactive_plan, "emotion_hint", "") or "")
            _delivery_form = str(getattr(_proactive_plan, "delivery_form", "standalone") or "standalone")
            _mention_user_name = str(getattr(_proactive_plan, "mention_user_name", "") or "")
            _reference_user_name = str(getattr(_proactive_plan, "reference_user_name", "") or "")
        if isinstance(_scheduled_event, dict) and _scheduled_event:
            _event_type = str(_scheduled_event.get("type", "") or "")
            if _event_type == "proactive_greeting" and not _proactive_topic:
                _proactive_topic = "打个招呼"
            elif _event_type == "random_observation" and not _proactive_topic:
                _proactive_topic = "随手观察群里气氛"
            elif _event_type == "mood_expression" and not _proactive_emotion:
                _proactive_emotion = "情绪表达"
            elif _event_type == "memory_recall" and not _proactive_topic:
                _proactive_topic = "回想起之前聊过的话题"
            elif _event_type == "topic_resurrection" and not _proactive_topic:
                _proactive_topic = "把旧话题重新捞起来"
            if _event_type and _event_type not in thought_text:
                thought_text = f"[后台事件:{_event_type}] {thought_text}"

        # 收集近期消息供执行器定位引用目标
        _proactive_incoming = []
        try:
            _proactive_incoming = message_api.get_messages_by_time_in_chat(
                chat_id=self.stream_id,
                start_time=time.time() - 600,
                end_time=time.time(),
                limit=15,
                limit_mode="latest",
            )
        except Exception as _exc:
            logger.debug(f"非关键异常: {_exc}")

        _pre_reply_resource_snapshot = self._capture_pre_reply_resource_snapshot()
        self._apply_plan_drain()
        try:
            acted = await asyncio.wait_for(
                self._generate_and_send_proactive_reply(
                    incoming_batch=_proactive_incoming,
                    desire=_actual_desire,
                    thought=thought_text,
                    target_uid=self._resolve_latest_human_user_id(
                        allow_cached_fallback=False
                    )
                    or "",
                    awareness=awareness_snapshot,
                    ambient=ambient_info,
                    emotion=emotion_snapshot,
                    arbiter_reason=reason,
                    proactive_topic=_proactive_topic,
                    proactive_emotion=_proactive_emotion,
                    delivery_form=_delivery_form,
                    mention_user_name=_mention_user_name,
                    reference_user_name=_reference_user_name,
                ),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            acted = False
            logger.error(f"{self.log_prefix} ⚠️ 后台主动回复超时(120s)")
        if acted:
            self._last_idle_proactive_ts = now_act
            self._last_background_event = {}
            await self._finalize_external_proactive_reply_flow(
                _proactive_incoming,
                source="background_proactive",
                desire_level=_actual_desire,
            )
        else:
            self._last_background_event = {}
            self._restore_pre_reply_resource_snapshot(
                _pre_reply_resource_snapshot,
                reason="background_proactive未形成有效回复",
            )
        return acted

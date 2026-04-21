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

class EnhancedLoopFlowMixin:
    async def _loopbody(self):
        """七阶段增强循环体，完全替代基础版"""
        self._tick_world_snapshot = None
        self._resolved_emo_state = None
        self._resolved_emo_uid = ""
        self._reset_decision_trace()
        # 群场景软门控标记按轮次生效，避免跨轮残留导致“长期异常沉默”
        self._pattern_forced_observe = False
        self._defense_mode_active = False
        self._cached_is_pinged = False
        self._cached_targeted_to_bot = False
        self._is_admin_forced = False
        self._last_repeat_detected = False
        self._overnight_pressure_total = 0.0
        self._enhanced_round += 1
        now = time.time()
        _t0 = now

        # 预构建频道级世界快照（无目标用户），供 Phase 0/1 门控优先读取
        try:
            from src.core.world_snapshot import build_world_snapshot

            self._tick_world_snapshot = await asyncio.wait_for(
                build_world_snapshot(self.stream_id),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.debug(f"{self.log_prefix} 世界快照预构建超时(10s)，门控退回直接读源")
        except Exception as _snap_pre:
            logger.debug(f"{self.log_prefix} 世界快照预构建失败，门控退回直接读源: {_snap_pre}")

        # ── 阶段 0-a：夜间节律行为倾向评估 ──
        _night_verdict = self._integrate_night_cycle(now)
        if _night_verdict in ("sleep", "drowsy", "burned_out"):
            _phase_obj = getattr(self, "_cached_night_phase", None)
            _night_phase_val = getattr(_phase_obj, "value", "") if _phase_obj else ""
            _ncs_sup = None
            try:
                from src.core.night_cycle_system import get_night_cycle

                _ncs_sup_obj = get_night_cycle(self.stream_id)
                _ncs_sup = _ncs_sup_obj.state_snapshot.response_suppression_coef
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
            if _ncs_sup is not None and _ncs_sup > 0.0:
                self._night_reply_suppression = round(_ncs_sup, 2)
                if abs(self._night_reply_suppression - self._last_night_sup_log) > 0.01:
                    logger.info(f"{self.log_prefix} 🌙 NCS动态抑制: {self._night_reply_suppression:.0%}")
                    self._last_night_sup_log = self._night_reply_suppression
            else:
                try:
                    _d6 = EnergyChainDimension.get_instance()
                    _nm = _d6._get_night_mode(self.stream_id)
                    if _nm.phase in ("DEEP_VALLEY",):
                        self._night_reply_suppression = 0.95
                        if abs(self._night_reply_suppression - self._last_night_sup_log) > 0.01:
                            logger.info(f"{self.log_prefix} 🌙 深夜低谷模式: 回复抑制95%")
                            self._last_night_sup_log = self._night_reply_suppression
                    elif _nm.phase in ("DROWSY",):
                        self._night_reply_suppression = 0.70
                        if abs(self._night_reply_suppression - self._last_night_sup_log) > 0.01:
                            logger.info(f"{self.log_prefix} 🌙 犯困模式: 回复抑制70%")
                            self._last_night_sup_log = self._night_reply_suppression
                    elif _night_verdict == "burned_out" or _nm.prob_multiplier <= 0.15:
                        self._night_reply_suppression = 0.90
                        if abs(self._night_reply_suppression - self._last_night_sup_log) > 0.01:
                            logger.info(f"{self.log_prefix} 🔥 熬穿模式: 回复抑制90%")
                            self._last_night_sup_log = self._night_reply_suppression
                    else:
                        self._night_reply_suppression = 0.60
                        if abs(self._night_reply_suppression - self._last_night_sup_log) > 0.01:
                            logger.debug(f"{self.log_prefix} 🌙 夜间轻度抑制60%")
                            self._last_night_sup_log = self._night_reply_suppression
                except Exception as _e:
                    logger.debug(f"{self.log_prefix} D6夜间抑制异常: {_e}")
                    self._night_reply_suppression = 0.80
        else:
            # 白天/非冻结夜态应清空抑制，避免跨时段残留导致过度沉默
            if getattr(self, "_night_reply_suppression", 0.0) != 0.0:
                logger.debug(f"{self.log_prefix} 🌤️ 清空夜间回复抑制({self._night_reply_suppression:.0%}→0%)")
            self._night_reply_suppression = 0.0
            self._last_night_sup_log = -1.0
            self._night_soft_wake_info = None

        # ── 阶段 0-a2：技能生命周期周期审查 ──
        self._integrate_skill_lifecycle_review()

        # ── 阶段 0-a3：印象周期维护（重写 + 自动昵称） ──
        self._integrate_impression_periodic_maintenance()
        # ── 阶段 0-a4：记忆系统周期维护（海马体流转+去重+压缩+过载检查） ──
        self._integrate_memory_system_periodic()

        # ── 阶段 0-b：关注层级衰减（无消息期） ──
        self._integrate_watch_state(now, has_messages=False)

        # ── 阶段 0：状态机门控 ──
        current_phase = self._query_flow_phase()
        gate_result = self._evaluate_phase_gate(current_phase, now)
        if gate_result == "skip":
            self._consecutive_skip_ticks += 1
            await asyncio.sleep(_DORMANT_POLL_SEC)
            return True
        if gate_result == "glance":
            # 休息期窥屏：只消耗少量体力，不进入完整流水线
            self._apply_glance_drain()
            self._consecutive_skip_ticks = 0
            await asyncio.sleep(_DORMANT_POLL_SEC)
            return True

        # ── 阶段 1：能量评估门控 ──
        eagerness_val, eagerness_reason = self._poll_eagerness()
        self._log_energy_status(eagerness_val, eagerness_reason)
        # 主观存在态 + 代谢引擎恢复
        self._integrate_presence_metabolism(eagerness_val, now)
        # 更新自主代理上下文
        silence_sec = 0.0
        try:
            from src.chat.proactive.silence_watcher import get_quiet_monitor

            silence_sec = get_quiet_monitor().measure_silence_sec(self.stream_id)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        if eagerness_val < _ENERGY_DRAIN_FLOOR:
            # 能量严重不足，迁移至休息
            self._shift_to_dormant(cause=f"能量耗尽: {eagerness_reason}")
            await asyncio.sleep(_DORMANT_POLL_SEC)
            return True

        # ── 第一层状态栏：独立体征面板（仅在有新消息时输出，避免刷屏） ──
        phase_display = current_phase if isinstance(current_phase, str) else str(current_phase)

        # ── 阶段 2：消息读取（复用父类逻辑的核心部分） ──
        incoming_batch = message_api.get_messages_by_time_in_chat(
            chat_id=self.stream_id,
            start_time=self.last_read_time,
            end_time=now,
            limit=20,
            limit_mode="latest",
            filter_mai=True,
            filter_command=False,
            filter_intercept_message_level=0,
        )
        self._last_msg_was_admin = bool(incoming_batch and self._is_force_wake_admin(incoming_batch))
        # ── 消息去重：过滤掉本轮已经处理过的消息（防止同一消息被多次处理） ──
        _new_batch = []
        for _msg in incoming_batch:
            _mid = getattr(_msg, "message_id", None) or getattr(_msg, "id", None) or str(_msg)
            if _mid and _mid not in self._processed_message_ids:
                _new_batch.append(_msg)
                self._processed_message_ids.add(_mid)
                self._dedup_order.append(_mid)
                if len(self._dedup_order) > 2000:
                    for _old_mid in list(self._dedup_order)[:1000]:
                        self._processed_message_ids.discard(_old_mid)
                    self._dedup_order = self._dedup_order[-1000:]
        incoming_batch = _new_batch
        # ── 去重后重新计算动态阈值 ──
        if self._batch_contains_real_user_reply(incoming_batch):
            _reply_msg = None
            for _rm in reversed(incoming_batch or []):
                if self._message_looks_like_user_reply(_rm):
                    _reply_msg = _rm
                    break
            self._mark_user_replied(latest_msg=_reply_msg)
        if self._should_short_circuit_self_echo_batch(incoming_batch):
            self.last_read_time = now
            self._consecutive_skip_ticks = 0
            self._last_flow_blocker = "仅检测到自身消息回流"
            self._emit_flow_decision_summary("self_echo_gate", "heartbeat_only")
            logger.debug(f"{self.log_prefix} 🪞 仅读到自身消息回流，跳过完整管线")
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True
        dynamic_threshold = self._compute_dynamic_threshold()
        if len(incoming_batch) < dynamic_threshold:
            self._consecutive_skip_ticks += 1
            _should_emit_status = False
            if (
                self._backend_status_log_enabled
                and (now - self._last_backend_status_ts) >= self._backend_status_interval
            ):
                _should_emit_status = True
            elif (now - self._last_backend_status_ts) >= 10.0:
                _should_emit_status = True
            if _should_emit_status:
                _status_line = self._get_dashboard_status_line() or ""
                if _status_line and _status_line != self._last_dashboard_status_line:
                    logger.info(f"{self.log_prefix} 📊 {_status_line}")
                    self._last_dashboard_status_line = _status_line
                elif not _status_line and self._backend_status_log_enabled:
                    logger.info(
                        f"{self.log_prefix} 💤 "
                        f"静默={silence_sec:.0f}s 意愿={eagerness_val:.2f} "
                        f"消息={len(incoming_batch)}/{dynamic_threshold} 跳过={self._consecutive_skip_ticks}"
                    )
                self._last_backend_status_ts = now
            # ── 空闲主动路径：无新消息时尝试主动行为 ──
            idle_acted = await self._try_idle_proactive(now, silence_sec)
            if idle_acted:
                self._consecutive_skip_ticks = 0
                return True
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True

        self._consecutive_skip_ticks = 0
        self.last_read_time = now
        # ── 预构建仪表盘（用上一tick缓存数据，确保体征面板的📊状态行有内容输出） ──
        self._build_state_dashboard(force=False)
        # ── 第一层状态栏：有新消息时输出体征面板 ──
        self._emit_vitals_gauge(eagerness_val, eagerness_reason, phase_display)
        logger.info(f"{self.log_prefix} 🔄 读到 {len(incoming_batch)} 条消息，进入完整管线")

        # ── 阶段 2-a：核心模块消息路由 + 活跃名册 + 时间流 + 多模态预算 ──
        self._integrate_message_routing(incoming_batch)
        # 关注层级升级（有消息到达）
        self._integrate_watch_state(now, has_messages=True)

        pinged_msg = self._find_pinged_message(incoming_batch)
        self._cached_is_pinged = pinged_msg is not None
        self._cached_targeted_to_bot = bool(pinged_msg is not None or self._has_targeted_bot_message(incoming_batch))

        # GAP-F：观看状态机完整转移表 —— 在BLACKOUT阻断前先评估是否应转换状态
        _watch_transition = self._evaluate_watch_full_transitions(
            now=now,
            has_ping=pinged_msg is not None,
            incoming_batch=incoming_batch,
        )
        if _watch_transition:
            logger.debug(f"{self.log_prefix} 观看状态转移: {_watch_transition}")

        # F2：黑屏态阻断——BLACKOUT 时只做最轻量追踪，跳过完整管线
        if self._cached_watch_level is not None:
            _wl_val = (
                self._cached_watch_level.value
                if hasattr(self._cached_watch_level, "value")
                else str(self._cached_watch_level)
            )
            if _wl_val == "blackout":
                _has_forced_ping = pinged_msg is not None
                if not _has_forced_ping:
                    logger.debug(f"{self.log_prefix} 🚫 黑屏态阻断完整管线，仅保留心跳追踪")
                    self._last_flow_blocker = "观看态=blackout，仅保留心跳追踪"
                    self._emit_flow_decision_summary("watch_gate", "heartbeat_only")
                    await asyncio.sleep(_TICK_FLOOR_SEC * 3)
                    return True
                else:
                    logger.debug(f"{self.log_prefix} 🚫 黑屏态但被@提及，降级放行")
            elif _wl_val == "peek":
                if self._is_force_wake_admin(incoming_batch, pinged_msg):
                    logger.info(f"{self.log_prefix} 👑 管理员强制唤醒-跳过窥屏态，直接进入完整管线")
                    self._peek_mode_active = False
                    self._cached_voice = None
                elif pinged_msg is None:
                    # 窥屏态：概率性调用内心独白，让模型自主决定是否升级参与
                    try:
                        peek_verdict = await asyncio.wait_for(
                            self._run_peek_with_reflection(incoming_batch),
                            timeout=_parallel_stage_timeout(),
                        )
                    except asyncio.TimeoutError:
                        peek_verdict = None
                        logger.debug(f"{self.log_prefix} 窥屏态反思超时，视为无波澜")
                    if peek_verdict is None or not getattr(peek_verdict, "is_valid", False):
                        self._cached_voice = None
                        logger.info(f"{self.log_prefix} 👁 窥屏态观察完成，内心无波澜")
                        self._last_flow_blocker = "窥屏态无明显波澜"
                        self._emit_flow_decision_summary("peek_gate", "observe")
                        return True
                    desire = getattr(peek_verdict, "reply_desire_level", 0)
                    thought = (
                        getattr(peek_verdict, "thinking", "")
                        or getattr(peek_verdict, "thought_text", "")
                        or ""
                    )
                    mood = getattr(peek_verdict, "current_mood", "") or ""
                    _peek_action = str(getattr(peek_verdict, "next_action", "") or "").strip().lower()
                    if thought:
                        logger.info(
                            f"{self.log_prefix} 💭 窥屏想法={thought[:40]} | "
                            f"欲望={desire} 情绪={mood} 行为={_peek_action}"
                        )
                    should_upgrade, upgrade_reason = self._should_promote_peek_reflection(
                        incoming_batch=incoming_batch,
                        peek_verdict=peek_verdict,
                        pinged_msg=pinged_msg,
                    )
                    if should_upgrade:
                        logger.info(f"{self.log_prefix} 👁→👀 窥屏态升级：{upgrade_reason}")
                        self._cached_voice = peek_verdict
                        self._align_states_with_inner_voice(peek_verdict, source="peek_reflection")
                        self._peek_mode_active = False
                    else:
                        self._cached_voice = None
                        logger.info(f"{self.log_prefix} 👁 窥屏态维持：{upgrade_reason}")
                        self._last_flow_blocker = f"窥屏Governor维持 {upgrade_reason}"
                        self._emit_flow_decision_summary("peek_gate", "observe")
                        return True
                else:
                    logger.debug(f"{self.log_prefix} 👁 窥屏态但被@提及，降级放行")
            elif _wl_val == "skim_window":
                self._peek_mode_active = True
                logger.debug(f"{self.log_prefix} 🔍 扫屏态：降低参与度")
            elif _wl_val == "active_watch":
                self._peek_mode_active = False
            elif _wl_val == "engaged":
                self._peek_mode_active = False

        # 用户消息到达时重置连续发言计数，并回落话痨惩罚
        has_user_message = self._batch_has_human_messages(incoming_batch)
        if has_user_message and self._consecutive_speaks > 0:
            _old = self._consecutive_speaks
            self._consecutive_speaks = 0.0
            logger.debug(f"{self.log_prefix} 用户消息，连续发言计数清零 {_old:.1f}->0.0")
        if has_user_message and self._chatterbox_penalty > 0:
            _old_penalty = float(self._chatterbox_penalty or 0.0)
            self._chatterbox_penalty = max(0.0, _old_penalty - 1.0)
            logger.debug(f"{self.log_prefix} 用户消息，话痨惩罚回落 {_old_penalty:.1f}->{self._chatterbox_penalty:.1f}")
        if has_user_message:
            self._legacy_constraint_hits = max(0, self._legacy_constraint_hits - 1)

        # 更新用户消息累积追踪器
        now_ts = time.time()
        _user_msg_ts_list = []
        for _um in incoming_batch:
            if self._is_bot_message_obj(_um):
                continue
            _msg_ts = float(getattr(_um, "timestamp", 0.0) or 0.0)
            _user_msg_ts_list.append(_msg_ts if _msg_ts > 0 else now_ts)
        user_msgs_in_batch = len(_user_msg_ts_list)
        if user_msgs_in_batch > 0:
            self._user_msg_counter += user_msgs_in_batch
            self._user_msg_timeline.extend(_user_msg_ts_list)
            if len(self._user_msg_timeline) > 50:
                self._user_msg_timeline = self._user_msg_timeline[-50:]
            self._recent_user_timestamps = list(self._user_msg_timeline)
            self._last_user_msg_time = max(_user_msg_ts_list)

            burst_window = 60.0
            recent_msgs = sum(1 for t in self._user_msg_timeline if now_ts - t < burst_window)
            if recent_msgs >= 5:
                if not self._is_in_burst:
                    self._is_in_burst = True
                    self._burst_start_time = now_ts
                    logger.info(f"{self.log_prefix} 🔥 检测到用户刷屏 burst_mode=True 消息数={recent_msgs}")
                self._burst_user_count = recent_msgs
            else:
                if self._is_in_burst and (now_ts - self._burst_start_time) > 120.0:
                    self._is_in_burst = False
                    self._burst_user_count = 0
                    logger.info(f"{self.log_prefix} 🔥 刷屏结束 burst_mode=False")

            self._update_dynamic_ratio(now_ts)
        # 用户回复时完成针对该用户的等待意图
        if has_user_message:
            try:
                from src.chat.proactive.intention_pool import (
                    get_intention_pool,
                    IntentKind,
                )

                for _m in incoming_batch:
                    _uid = str(getattr(_m, "user_id", "") or "").strip()
                    if _uid and not self._is_bot_message_obj(_m):
                        get_intention_pool().complete_by_target(
                            self.stream_id,
                            _uid,
                            IntentKind.WAIT_FOR_REPLY,
                        )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

        # 消息到达时立即记录行为与氛围
        self._record_message_behavior(incoming_batch)
        self._record_interaction_mood(incoming_batch)
        await self._apply_input_state_updates(incoming_batch)

        # ═══ 早退出检测：噪声/纯emoji/超短消息 跳过整个管线 ═══
        _early_exit = self._check_pipeline_early_exit(incoming_batch, pinged_msg)
        if _early_exit:
            logger.debug(f"{self.log_prefix} ⏭️ 早退出: {_early_exit}")
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True

        # 重建完整世界快照（携带目标用户ID），供 Phase 2.5+ 所有门控使用
        _target_uid = ""
        for _m_snap in reversed(incoming_batch):
            _uid_snap = getattr(_m_snap, "user_id", "") or ""
            if _uid_snap and not self._is_bot_message_obj(_m_snap):
                _target_uid = _uid_snap
                break
        if _target_uid:
            try:
                from src.core.world_snapshot import build_world_snapshot

                self._tick_world_snapshot = await asyncio.wait_for(
                    build_world_snapshot(self.stream_id, _target_uid),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                logger.warning(f"{self.log_prefix} 完整快照重建超时(15s)，跳过")
            except Exception as _snap_full:
                logger.debug(f"{self.log_prefix} 完整快照重建失败: {_snap_full}")

        # ── 阶段 2.5：核心系统集成（Group A + Group B 全并行） ──
        _t25 = time.time()
        _stage25_timeout = 30.0
        try:
            _stage25_results = await asyncio.wait_for(
                asyncio.gather(
                    self._check_identity_context(incoming_batch),
                    self._update_dynamic_context(incoming_batch),
                    self._check_self_reply_risk(incoming_batch),
                    self._track_content_state(incoming_batch),
                    self._store_interaction_memory(incoming_batch),
                    self._update_emotion_tracker_state(incoming_batch),
                    self._update_trauma_system_state(incoming_batch),
                    self._analyze_group_sense(incoming_batch),
                    self._analyze_message_preprocessor(incoming_batch),
                    self._update_user_interaction_styles(incoming_batch),
                    self._sync_memoir_on_message(incoming_batch),
                    return_exceptions=True,
                ),
                timeout=_stage25_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"{self.log_prefix} ⚠️ 阶段2.5超时({_stage25_timeout:.0f}s)，跳过未完成任务")
            _stage25_results = [None] * 11
        identity_context = _stage25_results[0] if not isinstance(_stage25_results[0], Exception) else {"identity": "default", "response_mode": "normal", "has_conflict": False}
        self_reply_risk = _stage25_results[2] if not isinstance(_stage25_results[2], Exception) else {"is_self_reply": False, "similarity": 0.0}
        group_sense_result = _stage25_results[7] if not isinstance(_stage25_results[7], Exception) else {}
        preprocessor_signal = _stage25_results[8] if not isinstance(_stage25_results[8], Exception) else {}
        for _idx, _r in enumerate(_stage25_results):
            if isinstance(_r, Exception):
                logger.debug(f"{self.log_prefix} 阶段2.5任务[{_idx}]异常: {_r}")
        logger.info(f"{self.log_prefix} 🔄 阶段2.5完成 {time.time() - _t25:.2f}s")

        # ── 阶段 2.5-b：核心模块深度理解 + 群场景 + 记忆激活 ──
        _t25b = time.time()
        self._integrate_deep_understanding(incoming_batch)
        self._integrate_scene_tracking(incoming_batch)
        self._integrate_event_learning(incoming_batch)
        self._integrate_memory_reactivation(incoming_batch)
        group_sense_result = self._merge_group_context_signal(group_sense_result)
        _now_af = time.time()
        self._run_attention_flow_tick(_now_af)
        self._integrate_gossip_ritual_strategy(_now_af)
        logger.debug(f"{self.log_prefix} 阶段2.5b(深度集成) {time.time() - _t25b:.2f}s")

        # ── 阶段 2.5-bis：群体模式硬路由 + 群场景硬约束 + 夜间节律 ──
        _now_hr = time.time()
        _night_result = self._apply_night_cycle_modulation(_now_hr)
        if _night_result is not None:
            _night_result = dict(_night_result)
            _raw_action = _night_result.get("action", "")
            _action = self._normalize_night_action(_raw_action, allow_internal=True)
            if _action != str(_raw_action or "").strip().lower():
                logger.warning(
                    f"{self.log_prefix} 🌙 未知夜间动作已归一: {_raw_action!r} -> {_action}"
                )
                _night_result["action"] = _action
            if _action in ("deep_sleep", "sleep_resist", "hard_block"):
                if not self._is_force_wake_admin(incoming_batch, pinged_msg):
                    logger.info(f"{self.log_prefix} 🌙 夜间节律拦截: {_night_result.get('reason', _action)}")
                    self._last_flow_blocker = f"夜间节律拦截: {_night_result.get('reason', _action)}"
                    self._emit_flow_decision_summary("night_gate", "skip")
                    await asyncio.sleep(_TICK_FLOOR_SEC)
                    return True
            if _action == "force_wake":
                try:
                    _d6_fw = EnergyChainDimension.get_instance()
                    _stim_fw = _night_result.get("stimulus", 0.0)
                    _d6_fw.apply_wake_feedback(self.stream_id, "force_wake_admin", 10, _stim_fw)
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
                logger.info(f"{self.log_prefix} 👑 管理员强制唤醒-无视夜间节律")
            elif _action == "llm_decide_l2":
                try:
                    _llm_decision = await asyncio.wait_for(
                        self._night_llm_decision(incoming_batch),
                        timeout=20.0,
                    )
                except asyncio.TimeoutError:
                    _llm_decision = {"action": "sleep_resist", "reason": "夜间LLM决策超时", "desire": 0}
                    logger.debug(f"{self.log_prefix} 夜间L2决策超时(20s)，默认继续睡")
                _llm_raw_action = _llm_decision.get("action", "sleep_resist")
                _llm_action = self._normalize_night_action(_llm_raw_action)
                if _llm_action != str(_llm_raw_action or "").strip().lower():
                    logger.warning(
                        f"{self.log_prefix} 🌙 夜间L2动作已归一: {_llm_raw_action!r} -> {_llm_action}"
                    )
                _llm_decision["action"] = _llm_action
                if _llm_action == "sleep_resist":
                    if not self._is_force_wake_admin(incoming_batch, pinged_msg):
                        logger.info(f"{self.log_prefix} 🌙 🧠 L2判定拦截: {_llm_decision.get('reason', _llm_action)}")
                        await asyncio.sleep(_TICK_FLOOR_SEC)
                        return True
                _llm_verdict = _llm_decision.get("llm_verdict", {})
                _thinking = _llm_verdict.get("thinking", "")
                _mood = _llm_verdict.get("mood", "")
                _desire_val = _llm_decision.get("desire", 0)
                logger.info(
                    f"{self.log_prefix} 😴🧠 L2决策通过: {_llm_action} | "
                    f"欲望={_desire_val}/10 心情={_mood} | 思考: {_thinking[:50] if _thinking else ''}"
                )
                self._night_soft_wake_info = {
                    "action": _llm_action,
                    "reason": _llm_decision.get("reason", ""),
                    "desire": _desire_val,
                    "should_reply": _llm_verdict.get("should_reply", False),
                    "thinking": _thinking[:300],
                    "mood": _mood,
                    "intent_type": _llm_verdict.get("intent_type", ""),
                    "context": _night_result.get("context"),
                    "stimulus": _night_result.get("stimulus"),
                    "threshold": _night_result.get("threshold"),
                }
            elif _action in ("soft_wake", "full_wake", "grumpy_glance"):
                _algo_desire = _night_result.get("desire", 0)
                logger.info(
                    f"{self.log_prefix} 😴⚙️ 夜间{_action}: {_night_result.get('reason', '')} | 欲望={_algo_desire}/10"
                )
                self._night_soft_wake_info = {
                    "action": _action,
                    "reason": _night_result.get("reason", ""),
                    "desire": _algo_desire,
                    "should_reply": _algo_desire >= 4,
                    "thinking": "",
                    "mood": "",
                    "intent_type": "",
                    "context": _night_result.get("context"),
                    "stimulus": _night_result.get("stimulus"),
                    "threshold": _night_result.get("threshold"),
                }
            else:
                logger.warning(f"{self.log_prefix} 🌙 夜间动作兜底收口: {_action}")
                self._last_flow_blocker = f"夜间动作兜底: {_action}"
                self._emit_flow_decision_summary("night_gate", "skip")
                await asyncio.sleep(_TICK_FLOOR_SEC)
                return True
        _pattern_route = self._apply_pattern_based_routing(incoming_batch, pinged_msg=pinged_msg)
        if (
            _pattern_route
            and ("observe" in _pattern_route or "silence" in _pattern_route)
            and not self._is_force_wake_admin(incoming_batch, pinged_msg)
        ):
            logger.info(f"{self.log_prefix} 🚧 群体模式硬路由拦截: {_pattern_route}")
            await self._run_peek_observe_loop(incoming_batch)
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True
        if _pattern_route and self._is_force_wake_admin(incoming_batch, pinged_msg):
            logger.info(f"{self.log_prefix} 👑 管理员强制唤醒-无视群体硬路由: {_pattern_route}")
        _scene_block = self._apply_scene_hard_constraints(
            pinged_msg=pinged_msg,
            group_context_signal=group_sense_result,
        )
        if (
            _scene_block
            and ("unsuitable" in _scene_block or "spam" in _scene_block)
            and not self._is_force_wake_admin(incoming_batch, pinged_msg)
        ):
            logger.info(f"{self.log_prefix} 🚫 群场景硬约束拦截: {_scene_block}")
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True
        if _scene_block and self._is_force_wake_admin(incoming_batch, pinged_msg):
            logger.info(f"{self.log_prefix} 👑 管理员强制唤醒-无视场景约束: {_scene_block}")

        # ── 阶段 2.5-c：感官门控（可见性 + 理解度 → 提供参考信息，不硬性拦截） ──
        _gate_skip_reason = self._apply_sensory_gates(incoming_batch, pinged_msg=pinged_msg)
        if _gate_skip_reason:
            logger.info(f"{self.log_prefix} 🔇 感官门控建议跳过: {_gate_skip_reason} (但交给内心独白决策)")
            self._gate_suggestion = _gate_skip_reason
        else:
            self._gate_suggestion = None
        self._integrate_deep_visibility_and_freshness(incoming_batch)
        for _rm in incoming_batch:
            _uid_rm = str(getattr(_rm, "user_id", "") or "")
            if not _uid_rm or _uid_rm == "bot":
                continue
            _plain_rm = str(
                getattr(_rm, "processed_plain_text", "")
                or getattr(_rm, "plain_text", "")
                or getattr(_rm, "content", "")
                or ""
            )
            if len(_plain_rm.strip()) > 1:
                self.register_message_to_memory(_plain_rm[:200], _uid_rm)
        _now_cycle = time.time()
        self._integrate_memory_governance(_now_cycle)
        self._run_adaptive_pipeline_periodic(_now_cycle)
        self.run_emotion_feedback_cycle(_now_cycle)

        # ── 统一状态仪表盘：三层触发架构入口 ──
        _dashboard_snap = self._build_state_dashboard(force=True)
        if _dashboard_snap:
            _dash_verdict = _dashboard_snap.get("verdict", {})
            _dash_urgency = str(_dash_verdict.get("urgency", "") or "")
            # L1硬阻断检查：精力耗尽/深睡/安全封锁 → 直接跳过
            if _dash_verdict.get("process") is False:
                logger.info(
                    f"{self.log_prefix} 🛑 仪表盘L1阻断: {_dash_verdict.get('reason', '未知')} urgency={_dash_urgency}"
                )
                await asyncio.sleep(_TICK_FLOOR_SEC * 2)
                return True

        # 用户过滤：同步即可
        _t_prep = time.time()
        filtered_messages = self._filter_messages_by_user_preference(incoming_batch)

        # 检查是否被 @ 或提及（已在门控前计算）
        decision_messages = self._prepare_decision_messages(
            incoming_batch,
            filtered_messages,
            pinged_msg,
        )
        content_state_signal = self._evaluate_content_state_signal(decision_messages)
        repetition_signal = self._analyze_repetition_pressure(decision_messages)
        self._last_repeat_detected = bool(
            repetition_signal.get("detected") and repetition_signal.get("latest_matches_repeat")
        )
        harassment_signal = self._analyze_harassment_pressure(decision_messages)
        decision_context_packet = self._build_decision_context_packet(
            decision_messages,
            repetition_signal=repetition_signal,
        )
        behavior_signal = self._classify_behavior_signal(
            decision_messages,
            repetition_signal=repetition_signal,
            harassment_signal=harassment_signal,
        )
        self._update_decision_trace(semantic_route_summary=dict(getattr(self, "_cached_route_summary", None) or {}))
        await self._apply_repetition_emotion_feedback(decision_messages, repetition_signal)
        await self._prefetch_memory_hint(decision_messages)
        if repetition_signal.get("detected"):
            logger.info(f"{self.log_prefix} 重复输入提示 {repetition_signal.get('reason', '当前内容重复度偏高')}")
        if harassment_signal.get("detected"):
            logger.info(
                f"{self.log_prefix} 骚扰强度提示 {harassment_signal.get('reason', '当前输入可能带持续冒犯倾向')}"
            )

        message_salience = self._evaluate_message_salience(
            decision_messages,
            pinged_msg,
            repetition_signal,
            behavior_signal,
        )
        force_reply_message = None
        # 低信息重复标记：阻止下游旁路强制回复
        _low_info_repeat_block = False
        if (
            repetition_signal.get("detected")
            and repetition_signal.get("latest_matches_repeat")
            and repetition_signal.get("low_info_cluster")
        ):
            # 低信息重复内容（如反复发同一短句）应被忽略，不设置强制回复
            _low_info_repeat_block = True
            logger.info(f"{self.log_prefix} 🔕 低信息重复检测: 抑制所有强制回复旁路")
        elif pinged_msg is not None:
            # 被@时优先锁定回复目标，避免后续落到错误的最新消息
            force_reply_message = pinged_msg
        _force_direct_ping = force_reply_message is not None

        # ... 其他代码保持不变 ...

        # ── 阶段 2.9：旁白策略规划（基于前序模块结果） ──
        self._integrate_narration_planning(decision_messages)
        if self._cached_narration_plan is not None:
            self._update_decision_trace(
                narration_should_reply=bool(getattr(self._cached_narration_plan, "should_reply", False)),
                narration_action_intent=str(getattr(self._cached_narration_plan, "action_intent", "") or ""),
                narration_reply_strategy=str(getattr(self._cached_narration_plan, "reply_strategy", "") or ""),
            )
        logger.debug(f"{self.log_prefix} 决策准备阶段 {time.time() - _t_prep:.2f}s")

        # ── 阶段 2.9b：技能调用追踪（本轮各子系统均已执行完毕） ──
        self._track_skill_invocations()

        # ── 阶段 3+4+4.5：感知、内心独白、关系度 —— 并行 ──
        _t345 = time.time()
        ambient_info = self._sample_channel_ambient()
        run_perception = self._should_run_perception(now)
        run_voice = self._should_run_voice(now, incoming_batch, pinged_msg)
        # 窥屏态升级复用：如果窥屏态 LLM 已决定 reply/followup 且缓存有效，跳过重复独白
        _peek_voice_reuse = False
        if (
            self._peek_mode_active
            and
            self._cached_voice is not None
            and getattr(self._cached_voice, "is_valid", False)
            and str(getattr(self._cached_voice, "next_action", "") or "").strip().lower() in ("reply", "followup")
        ):
            run_voice = False
            _peek_voice_reuse = True
        logger.info(
            f"{self.log_prefix} 🔄 阶段3+4启动 感知={run_perception} 独白={run_voice}"
            + (" (窥屏态verdict复用)" if _peek_voice_reuse else "")
        )
        _is_admin_msg = self._is_force_wake_admin(incoming_batch, pinged_msg)
        if _is_admin_msg:
            run_perception = False
            run_voice = False
            logger.info(f"{self.log_prefix} 👑 管理员极速通道: 跳过观察+独白")

        # 组装并行任务列表
        _parallel_tasks: list = []
        _task_keys: list = []

        if run_perception:
            _parallel_tasks.append(self._invoke_perception(decision_messages, now))
            _task_keys.append("perception")
        if run_voice:
            _parallel_tasks.append(
                self._invoke_inner_voice(
                    decision_messages,
                    self._cached_awareness,
                    ambient_info,
                    now,
                )
            )
            _task_keys.append("voice")
        _parallel_tasks.append(self._compute_relation_metrics(decision_messages))
        _task_keys.append("relation")

        try:
            _stage_timeout = _parallel_stage_timeout()
            _parallel_results = await asyncio.wait_for(
                asyncio.gather(*_parallel_tasks, return_exceptions=True),
                timeout=_stage_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(f"{self.log_prefix} 🔄 阶段3+4 LLM调用超时({_stage_timeout:.0f}s)，跳过本轮")
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True
        _result_map = dict(zip(_task_keys, _parallel_results, strict=True))
        # 检查是否有异常
        for _k, _v in _result_map.items():
            if isinstance(_v, BaseException):
                logger.warning(f"{self.log_prefix} 🔄 阶段3+4 {_k} 异常: {_v}")
        logger.info(f"{self.log_prefix} 🔄 阶段3+4完成 {time.time() - _t345:.2f}s")

        # 拆包感知结果
        awareness_snapshot = self._cached_awareness
        if run_perception:
            _pval = _result_map.get("perception")
            if isinstance(_pval, BaseException):
                logger.warning(f"{self.log_prefix} 感知异常: {_pval}，沿用已有感知快照")
                if awareness_snapshot is None:
                    awareness_snapshot = {"perception_missing": True}
            elif _pval is not None:
                awareness_snapshot = _pval
                self._cached_awareness = awareness_snapshot
                self._last_perception_ts = now
        if awareness_snapshot is None:
            awareness_snapshot = {"perception_missing": True}

        # 拆包内心独白结果
        voice_conclusion = self._cached_voice
        if run_voice:
            _vval = _result_map.get("voice")
            if isinstance(_vval, BaseException):
                logger.warning(f"{self.log_prefix} 内心独白异常: {_vval}，沿用已有独白快照")
            elif _vval is not None:
                voice_conclusion = _vval
                self._cached_voice = voice_conclusion
                self._last_voice_ts = now
                # 根据独白想法长度推导思考复杂度
                _thought_len = len(str(getattr(voice_conclusion, "thinking", "") or ""))
                _think_complexity = max(0.5, min(2.0, _thought_len / 40.0))
                self._apply_think_drain(complexity=_think_complexity)
                voice_summary = self._build_voice_execution_summary(
                    voice_conclusion,
                    decision_messages[-1] if decision_messages else None,
                )
                if voice_summary.get("log"):
                    logger.info(f"{self.log_prefix} 💭 {voice_summary['log'][:160]}")
                # 将内心独白意图写入跨轮意图池
                try:
                    from src.chat.proactive.intention_pool import (
                        get_intention_pool,
                    )

                    _speaker_id = ""
                    if decision_messages:
                        _speaker_id = str(getattr(decision_messages[-1], "user_id", "") or "").strip()
                    get_intention_pool().ingest_voice_verdict(
                        channel_id=self.stream_id,
                        verdict=voice_conclusion,
                        speaker_id=_speaker_id,
                    )
                except Exception as _ipool_err:
                    logger.warning(f"{self.log_prefix} 意图池写入失败: {_ipool_err}")
        try:
            from src.core.inner_narration_planner import InnerNarrationPlanner

            _narration_key = f"narration_planner_{self.stream_id}"
            _inp = getattr(self, "_narration_planner_cache", {}).get(_narration_key)
            if _inp is None:
                _inp = InnerNarrationPlanner(self.stream_id)
                if not hasattr(self, "_narration_planner_cache"):
                    self._narration_planner_cache = {}
                self._narration_planner_cache[_narration_key] = _inp
            _vis_level = str(getattr(self, "_cached_dominant_visibility", "noticed") or "noticed")
            _watch_lv = self._watch_level_value(default="skim_window")
            _understanding = "can_understand" if self._cached_understanding_results else "cannot_understand"
            _energy_ratio = 1.0
            try:
                _d6 = EnergyChainDimension.get_instance()
                _d6_snap = _d6.capture_snapshot(self.stream_id)
                _energy_ratio = _d6_snap.combined_ratio() if _d6_snap else 1.0
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            _scene_joinable = True
            if self._cached_scene_snapshot:
                _scene_joinable = bool(getattr(self._cached_scene_snapshot, "suitable_to_join", True))
            _dom_pat_name = ""
            if self._cached_pattern_evidence:
                _ptop = self._cached_pattern_evidence[0]
                _pconf = float(getattr(_ptop, "confidence", 0.0) or 0.0)
                if _pconf > 0.35:
                    _penum = getattr(_ptop, "pattern", None)
                    if _penum is not None:
                        _dom_pat_name = getattr(_penum, "value", "") or str(_penum)
            _trauma_val = float(getattr(self, "_cached_trauma_score", 0.0) or 0.0)
            _annoyance_val = float(getattr(self, "_cached_annoyance_value", 0.0) or 0.0)
            _nplan = _inp.plan(
                understanding_level=_understanding,
                visibility_level=_vis_level,
                watch_level=_watch_lv,
                energy_ratio=_energy_ratio,
                scene_suitable_to_join=_scene_joinable,
                dominant_pattern=_dom_pat_name,
                trauma_score=_trauma_val,
                channel_annoyance=_annoyance_val,
                message_summary=(str(getattr(pinged_msg, "content", "") or "")[:200] if pinged_msg else ""),
            )
            self._cached_narration_plan = _nplan
            _nprompt_block = _nplan.to_prompt_block()
            if _nprompt_block:
                logger.debug(f"{self.log_prefix} 📝 叙事规划: {_nprompt_block[:120]}")
        except Exception as _narr_err:
            logger.debug(f"{self.log_prefix} 叙事规划器异常: {_narr_err}")

        # 拆包关系度结果
        _rval = _result_map.get("relation")
        _relation_fallback = dict(getattr(self, "_last_relation_snapshot", None) or {})
        if isinstance(_rval, BaseException):
            logger.warning(f"{self.log_prefix} 关系度计算异常: {_rval}，沿用已有关系快照")
        if _rval is not None and not isinstance(_rval, BaseException):
            relation_result = self._normalize_relation_snapshot(_rval)
        elif _relation_fallback:
            relation_result = self._normalize_relation_snapshot(_relation_fallback)
        else:
            relation_result = {
                "chat_value": 100.0,
                "activity_level": 50.0,
                "shared_social_value": 0.0,
                "social_value": 0.0,
                "trust_value": 0.0,
                "annoyance_value": 0.0,
                "relationship_level": 2,
                "custom_label": "",
                "interaction_count": 0,
                "affection": 0.0,
                "psychological_pressure": 0.0,
                "trauma_score": 0.0,
                "mood": "平静",
                "attribute_influences": {},
            }
            relation_result = self._normalize_relation_snapshot(relation_result)
        relation_result["behavior_signal"] = behavior_signal

        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            tracker = get_emotion_tracker(self.stream_id)
            if hasattr(self, "_last_user_id") and self._last_user_id:
                state = tracker.get_user_state(self._last_user_id)
                if state:
                    _tracker_annoyance = float(
                        getattr(
                            state,
                            "annoyance_value",
                            getattr(state, "annoyance", 0.0),
                        )
                        or 0.0
                    )
                    _tracker_annoyance_acc = float(getattr(state, "annoyance_accumulated", 0.0) or 0.0)
                    if _tracker_annoyance > 0 or _tracker_annoyance_acc > 0:
                        relation_result["annoyance_value"] = max(_tracker_annoyance, _tracker_annoyance_acc)
                        logger.info(
                            f"{self.log_prefix} 🎭 情绪追踪器同步: 厌烦={_tracker_annoyance:.1f} 累积={
                                _tracker_annoyance_acc:.1f}"
                        )
                    self._resolved_emo_state = state
                    self._resolved_emo_uid = str(self._last_user_id or "").strip()
        except Exception as _e:
            logger.warning(f"{self.log_prefix} 情绪状态解析失败: {_e}")

        try:
            from src.modules.social_value.social_affect_fuser import (
                get_social_affect_fuser,
            )

            _fuser = get_social_affect_fuser()
            _mod_annoyance, _mod_pressure = _fuser.apply_group_modulation_to_emotion(
                channel_id=self.stream_id,
                current_annoyance=float(relation_result.get("annoyance_value", 0.0) or 0.0),
                current_pressure=float(relation_result.get("psychological_pressure", 0.0) or 0.0),
                recent_messages=incoming_batch,
                bot_user_id="bot",
            )
            if _mod_annoyance != float(relation_result.get("annoyance_value", 0.0) or 0.0):
                relation_result["annoyance_value"] = _mod_annoyance
                logger.debug(f"{self.log_prefix} 🎭 群环境调制: 厌烦值 → {_mod_annoyance:.1f}")
            if _mod_pressure != float(relation_result.get("psychological_pressure", 0.0) or 0.0):
                relation_result["psychological_pressure"] = _mod_pressure
                logger.debug(f"{self.log_prefix} 🎭 群环境调制: 压力值 → {_mod_pressure:.1f}")
        except Exception as _gmod_exc:
            logger.debug(f"{self.log_prefix} 群环境调制异常: {_gmod_exc}")

        relation_result = self._normalize_relation_snapshot(relation_result)
        relation_result["behavior_signal"] = behavior_signal
        self._last_relation_snapshot = relation_result
        self._log_relation_metrics(relation_result)
        self._cached_affection_value = float(relation_result.get("affection", 0.0) or 0.0)
        self._cached_trust_value = float(relation_result.get("trust_value", 0.0) or 0.0)
        self._cached_annoyance_value = float(relation_result.get("annoyance_value", 0.0) or 0.0)
        self._cached_pressure_value = float(relation_result.get("psychological_pressure", 0.0) or 0.0)
        self._cached_relationship_level = int(relation_result.get("relationship_level", 2) or 2)
        self._cached_familiarity = float(relation_result.get("familiarity", 0.0) or 0.0)
        self._cached_dominance = float(relation_result.get("dominance", 0.0) or 0.0)

        # ── 第二层状态栏：当前对象印象面板 ──
        _target_uid_for_panel = self._get_latest_human_user_id(decision_messages)
        if _target_uid_for_panel:
            self._last_user_id = _target_uid_for_panel
            await self._emit_target_profile(_target_uid_for_panel)

        # ── 阶段 4.6：早期退出检查 - 内心独白决定不想回复时直接跳过 ──
        early_exit = self._check_early_exit(
            voice_conclusion,
            pinged_msg,
            relation_result,
            decision_messages=decision_messages,
        )
        if early_exit["should_skip"] and not self._is_force_wake_admin(incoming_batch, pinged_msg):
            self._emit_action_verdict("early_exit", early_exit["reason"], time.time() - _t0)
            logger.info(f"{self.log_prefix} ⚡ {early_exit['reason']}，跳过LLM调用 | 管线耗时 {time.time() - _t0:.2f}s")
            await self._emit_outcome_summary(False, relation_result)
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True
        if early_exit["should_skip"] and self._is_force_wake_admin(incoming_batch, pinged_msg):
            logger.info(f"{self.log_prefix} 👑 管理员强制唤醒-无视早期退出: {early_exit['reason']}")

        # ── 阶段 4.5：多维状态系统统一决策（唯一决策入口） ──
        gateway_result = await self._run_dimension_gateway(
            decision_messages=decision_messages,
            pinged_msg=pinged_msg,
            voice_conclusion=voice_conclusion,
            relation_result=relation_result,
            repetition_signal=repetition_signal,
            harassment_signal=harassment_signal,
        )
        legacy_gate = "allow"
        legacy_constraint = {"should_skip": False, "gate": "allow", "reason": ""}
        _is_admin_forced = False
        if gateway_result is not None:
            _gw_gate = str(gateway_result.get("gate", "allow") or "allow")
            legacy_gate = _gw_gate
            legacy_constraint["gate"] = _gw_gate
            legacy_constraint["should_skip"] = bool(gateway_result.get("should_skip", False))
            legacy_constraint["reason"] = str(gateway_result.get("reason", "") or "维度网关裁定")
            self._last_legacy_gate = _gw_gate
            self._update_decision_trace(legacy_gate=_gw_gate)
            logger.info(
                f"{self.log_prefix} [维度网关主决策] gate={_gw_gate} "
                f"skip={legacy_constraint['should_skip']} "
                f"reason={legacy_constraint['reason'][:80]}"
            )
        else:
            legacy_gate = "allow"
            legacy_constraint = {"should_skip": False, "gate": "allow", "reason": "维度网关不可用，默认放行"}
            self._update_decision_trace(legacy_gate="allow")
            logger.warning(f"{self.log_prefix} 维度网关不可用，默认放行")

        if self._is_force_wake_admin(incoming_batch, pinged_msg):
            _orig_gate = legacy_gate
            _admin_force_guard_reason = self._admin_force_safety_guard_reason(
                gateway_result=gateway_result,
                relation_result=relation_result,
                repetition_signal=repetition_signal,
                harassment_signal=harassment_signal,
            )
            if _admin_force_guard_reason:
                _is_admin_forced = False
                logger.info(
                    f"{self.log_prefix} 👑 管理员强制唤醒受安全锁限制，不穿透骚扰/强拦截: "
                    f"{_admin_force_guard_reason[:80]}"
                )
            else:
                _is_admin_forced = True
                _admin_gate = self._resolve_admin_force_gate(_orig_gate)
                legacy_gate = _admin_gate
                legacy_constraint = {
                    "should_skip": False,
                    "gate": _admin_gate,
                    "reason": "管理员强制唤醒-无视维度网关",
                }
                self._last_legacy_gate = _admin_gate
                self._update_decision_trace(legacy_gate=_admin_gate)
                logger.info(f"{self.log_prefix} 👑 管理员强制唤醒-覆盖网关裁定: {_orig_gate}→{_admin_gate}")

        # 低信息复读不应触发任何强制回复旁路（管理员强制唤醒除外）
        if _low_info_repeat_block and legacy_gate == "force_reply" and not _is_admin_forced:
            legacy_gate = "hesitate"
            legacy_constraint["gate"] = "hesitate"
            legacy_constraint["should_skip"] = False
            legacy_constraint["reason"] = "低信息重复：降级force_reply旁路"
            self._last_legacy_gate = "hesitate"
            self._update_decision_trace(legacy_gate="hesitate")
            logger.info(f"{self.log_prefix} 🔕 低信息复读: force_reply旁路降级为hesitate")

        # ── 多维信号融合辅助裁决（仅对非强制情况提供风格建议，不硬拦截） ──
        _hf_judgment = self._query_heartflow_judgment(decision_messages, relation_result, voice_conclusion)
        if (
            _hf_judgment is not None
            and not _hf_judgment["should_respond"]
            and not _force_direct_ping
            and legacy_gate in ("allow", "hesitate")
        ):
            _hf_cert = _hf_judgment.get("certainty", "")
            if _hf_cert in ("FIRM", "RESOLUTE"):
                # 心流否决降级为 hesitate 风格约束，不硬拦截
                legacy_gate = "hesitate"
                legacy_constraint["gate"] = "hesitate"
                legacy_constraint["should_skip"] = False
                legacy_constraint["reason"] = (
                    f"心流信号否决(cert={_hf_cert}, "
                    f"comp={_hf_judgment.get('composite', 0):.2f}): "
                    f"{_hf_judgment.get('rationale', '')[:60]}"
                )
                self._last_legacy_gate = legacy_gate
                self._update_decision_trace(legacy_gate=legacy_gate)
                logger.info(
                    f"{self.log_prefix} [心流辅助] 六维信号否决→降级为hesitate风格约束 | "
                    f"composite={_hf_judgment.get('composite', 0):.2f}"
                )
            if self._is_force_wake_admin(incoming_batch, pinged_msg) and legacy_gate != "block":
                legacy_gate = "allow"
                legacy_constraint["should_skip"] = False
                legacy_constraint["gate"] = "allow"
                self._last_legacy_gate = "allow"
                self._update_decision_trace(legacy_gate="allow")
                _is_admin_forced = True
                logger.info(f"{self.log_prefix} 👑 管理员强制唤醒-无视心流否决裁决")
        self._is_admin_forced = _is_admin_forced

        autonomy_guard = self._evaluate_autonomy_guard(
            decision_messages=decision_messages,
            pinged_msg=pinged_msg,
            identity_context=identity_context,
            self_reply_risk=self_reply_risk,
            group_sense_result=group_sense_result,
            group_context_signal=group_sense_result,
            preprocessor_signal=preprocessor_signal,
            relation_result=relation_result,
            voice_conclusion=voice_conclusion,
            message_salience=message_salience,
            content_state_signal=content_state_signal,
            repetition_signal=repetition_signal,
            harassment_signal=harassment_signal,
        )
        self._update_decision_trace(autonomy_guard_reason=str(autonomy_guard.get("reason", "") or ""))
        if autonomy_guard["should_skip"] and not self._is_force_wake_admin(incoming_batch, pinged_msg):
            if autonomy_guard.get("source") == "content_state":
                self._apply_content_state_skip(content_state_signal)
            self._emit_action_verdict("autonomy_block", autonomy_guard["reason"], time.time() - _t0)
            logger.info(
                f"{self.log_prefix} 🚫 {autonomy_guard['reason']}，本轮不回复 | 管线耗时 {time.time() - _t0:.2f}s"
            )
            self._last_flow_blocker = autonomy_guard["reason"]
            self._mark_decision_winner("autonomy_guard")
            self._apply_post_reply_state(did_reply=False, reason=autonomy_guard["reason"])
            self._emit_flow_decision_summary("autonomy_guard", "skip")
            await self._emit_outcome_summary(False, relation_result)
            await asyncio.sleep(_POST_MESSAGE_RETRY_SEC)
            return True
        if (
            autonomy_guard["should_skip"]
            and self._is_force_wake_admin(incoming_batch, pinged_msg)
            and legacy_gate != "block"
        ):
            logger.info(f"{self.log_prefix} 👑 管理员强制唤醒-无视自治守卫: {autonomy_guard['reason']}")

        # ── 阶段 4.7：内心独白强驱动检查 - 欲望等级高时直接回复 ──
        voice_driven_reply = False
        desire_level = getattr(voice_conclusion, "reply_desire_level", 5) if voice_conclusion else 5
        # F3：主观存在态压制——refusing 态强制压低欲望上限
        if self._cached_presence_state is not None:
            _pv = getattr(self._cached_presence_state, "participation_verdict", None)
            if _pv == "refusing":
                _original_desire = desire_level
                desire_level = min(desire_level, 2)
                if _original_desire > 2:
                    logger.debug(f"{self.log_prefix} 🧊 存在态=拒绝参与，欲望{_original_desire}→压制为{desire_level}")
            elif _pv == "reluctant":
                if desire_level >= 5:
                    desire_level = 4
                    logger.debug(f"{self.log_prefix} 🧊 存在态=不情愿，欲望≥5→降为4")
        self._last_desire_level = float(desire_level or 5.0)
        # 独白降级为风格/紧迫度层：除明确提及外，不再单独决定是否直发
        if legacy_gate == "block":
            voice_driven_reply = False
            logger.info(f"{self.log_prefix} 门控=拦截，禁止独白强驱动穿透")
        elif legacy_gate == "hesitate":
            # hesitate 不禁止独白驱动回复，只作为风格约束标记
            logger.info(f"{self.log_prefix} 门控=犹豫，独白保留但注入短回复风格约束")
        elif legacy_gate == "force_reply":
            # 被@/间接提及/维度强制触发：默认回复
            voice_driven_reply = True
            logger.info(f"{self.log_prefix} 💭 维度网关force_reply，默认执行回复")
        elif _force_direct_ping:
            # 被@但网关未给force_reply（理论上不应出现，防御性兜底）
            voice_driven_reply = True
            logger.info(f"{self.log_prefix} 💭 被提及，执行回复")
        else:
            logger.info(f"{self.log_prefix} 💭 独白仅保留风格参考 desire={desire_level}")
        _llm_call_level, _skip_score, _esc_score, _tentative = self._compute_llm_call_level(
            now, incoming_batch, force_reply_message, desire_level
        )
        _behavior_for_model = self._evaluate_behavior_governor(
            incoming_batch=decision_messages,
            silence_sec=float(silence_sec or 0.0),
            requested_mode="reactive",
            target_message=force_reply_message or self._get_latest_human_message(decision_messages),
            is_background=False,
        )
        _relation_for_model = self._resolve_relation_view()
        _high_risk_model = bool(
            float(_relation_for_model.get("annoyance_value", 0.0) or 0.0) >= 65.0
            or float(_relation_for_model.get("psychological_pressure", 0.0) or 0.0) >= 55.0
        )
        _model_governor = self._evaluate_model_governor(
            now=now,
            desired_level=int(_llm_call_level or 0),
            incoming_batch=decision_messages,
            pinged_msg=force_reply_message,
            is_proactive=False,
            source="reactive_decision",
            behavior_verdict=_behavior_for_model,
            high_risk=_high_risk_model,
        )
        if _model_governor.tier == "skip":
            _llm_call_level = 0
        elif _model_governor.tier == "small" and _llm_call_level == 2:
            _llm_call_level = 1
        if _model_governor.rate_limited:
            logger.info(
                f"{self.log_prefix} 🎛️ 模型Governor限流(reactive_decision): "
                f"{self._summarize_model_governor(_model_governor)}"
            )

        _model_should_reply = getattr(voice_conclusion, "should_reply", None) if voice_conclusion else None
        _is_voice_reluctant = _model_should_reply is False
        _thinking_text = getattr(voice_conclusion, "thinking", "") or "" if voice_conclusion else ""
        # 处理 LLM 直接行为决策：rest / disengage / lurk
        _voice_action = getattr(voice_conclusion, "next_action", "") or "" if voice_conclusion else ""
        self._update_decision_trace(
            voice_action=str(_voice_action or ""),
            model_should_reply=_model_should_reply,
        )
        _is_gateway_force = legacy_gate == "force_reply"
        _strong_force_reply = bool(_is_admin_forced or _force_direct_ping or _is_gateway_force)
        if _voice_action == "rest":
            if _strong_force_reply:
                logger.info(f"{self.log_prefix} 💭 强制回复场景，忽略内心'rest'指令，继续执行决策")
            else:
                logger.info(f"{self.log_prefix} LLM决定歇一会: '{_thinking_text[:40]}'")
                self._enter_dynamic_rest(
                    cause=f"内心想休息({_thinking_text[:30]})",
                    trigger="内心想休息",
                    blackout=False,
                )
                self._mark_decision_winner("voice_action")
                self._emit_flow_decision_summary("voice_action", "rest")
                return
        if _voice_action == "disengage":
            if _strong_force_reply:
                logger.info(f"{self.log_prefix} 💭 强制回复场景，忽略内心'disengage'指令，继续执行决策")
            else:
                logger.info(f"{self.log_prefix} LLM决定放下手机: '{_thinking_text[:40]}'")
                self._enter_dynamic_rest(
                    cause=f"放下手机({_thinking_text[:30]})",
                    trigger="放下手机",
                    blackout=True,
                )
                self._mark_decision_winner("voice_action")
                self._emit_flow_decision_summary("voice_action", "disengage")
                return
        if _voice_action == "lurk":
            if _strong_force_reply:
                logger.info(f"{self.log_prefix} 💭 强制回复场景，忽略内心'lurk'指令，继续执行决策")
            else:
                logger.info(f"{self.log_prefix} LLM决定潜水: '{_thinking_text[:40]}'")
                self._align_states_with_inner_voice(voice_conclusion, source="voice_action")
                self._last_flow_blocker = "模型选择潜水，退回窥屏"
                self._mark_decision_winner("voice_action")
                self._emit_flow_decision_summary("voice_action", "lurk")
                return

        _algo_wants_skip = _llm_call_level == 0
        _algo_wants_reply = _llm_call_level == 1
        _algo_wants_upgrade = _llm_call_level == 2
        # R1修复：试探性发牢骚——算法判定跳过但试探条件满足，且模型未明确抗拒
        if _tentative and _algo_wants_skip and not _is_voice_reluctant:
            _algo_wants_skip = False
            _algo_wants_reply = True
            logger.info(f"{self.log_prefix} 试探性发牢骚触发: 沉默够久+害羞指数低，轻量试探")

        _final_skip = False
        _final_reply = False
        _final_upgrade = False

        if _is_voice_reluctant and not _force_direct_ping and not _is_gateway_force:
            _takeover_tried = False
            _takeover_by_model = False
            try:
                _takeover_result = await asyncio.wait_for(
                    self._try_algo_takeover_request(
                        now, incoming_batch, voice_conclusion, _thinking_text
                    ),
                    timeout=15.0,
                )
                if _takeover_result:
                    _takeover_tried = True
                    _takeover_by_model = _takeover_result.get("wants_to_takeover", False)
                    if _takeover_by_model:
                        _final_reply = True
                        self._takeover_decision = _takeover_result.get("decision")
                        self._takeover_action = _takeover_result.get("action", "接管")
                        self._mark_decision_winner("model_takeover")
                        logger.info(f"{self.log_prefix} 🎯 模型接管: 内心抗拒='{_thinking_text[:30]}...'，强制执行回复")
            except asyncio.TimeoutError:
                logger.debug(f"{self.log_prefix} 算法接管请求超时(15s)")
            except Exception as _takeover_err:
                logger.debug(f"{self.log_prefix} 算法接管请求异常: {_takeover_err}")
            if not _takeover_tried or not _takeover_by_model:
                if _is_admin_forced:
                    _final_reply = True
                    logger.info(f"{self.log_prefix} 👑 管理员强制唤醒-穿透模型内心抗拒, 强制回复")
                else:
                    _final_skip = True
                    logger.info(f"{self.log_prefix} 🎯 模型内心抗拒='{_thinking_text[:30]}...'，尊重意愿跳过")
        elif _algo_wants_skip and _force_direct_ping:
            if _model_should_reply is False:
                _takeover_tried = False
                _takeover_by_model = False
                try:
                    _takeover_result = await asyncio.wait_for(
                        self._try_algo_takeover_request(
                            now, incoming_batch, voice_conclusion, _thinking_text
                        ),
                        timeout=15.0,
                    )
                    if _takeover_result:
                        _takeover_tried = True
                        _takeover_by_model = _takeover_result.get("wants_to_takeover", False)
                        if _takeover_by_model:
                            _final_reply = True
                            self._takeover_decision = _takeover_result.get("decision")
                            self._takeover_action = _takeover_result.get("action", "接管")
                            self._mark_decision_winner("model_takeover")
                            logger.info(
                                f"{self.log_prefix} 🎯 模型接管(被@但内心抗拒): '{_thinking_text[:30]}...'，强制执行回复"
                            )
                except asyncio.TimeoutError:
                    logger.debug(f"{self.log_prefix} 算法接管请求超时(15s)")
                except Exception as _takeover_err:
                    logger.debug(f"{self.log_prefix} 算法接管请求异常: {_takeover_err}")
                if not _takeover_tried or not _takeover_by_model:
                    if _is_admin_forced or _is_gateway_force:
                        _final_reply = True
                        logger.info(
                            f"{self.log_prefix} 👑 强制回复-穿透被@内心抗拒 (admin={_is_admin_forced}, gateway_force={_is_gateway_force})"
                        )
                    else:
                        _final_skip = True
                        logger.info(f"{self.log_prefix} 🎯 被@但内心强烈抗拒='{_thinking_text[:30]}...'，尊重意愿跳过")
            else:
                _rel_for_mention = self._resolve_relation_view()
                _mention_annoy = float(_rel_for_mention.get("annoyance_value", 0.0) or 0.0)
                _mention_press = float(_rel_for_mention.get("psychological_pressure", 0.0) or 0.0)
                if _mention_annoy >= 55 or (_mention_annoy >= 35 and _mention_press >= 40):
                    if _is_admin_forced or _is_gateway_force:
                        _final_reply = True
                        logger.info(
                            f"{self.log_prefix} 💢 情绪高但强制回复(烦躁={_mention_annoy:.1f}) (admin={_is_admin_forced}, gateway_force={_is_gateway_force})"
                        )
                    else:
                        _final_skip = True
                        logger.info(
                            f"{self.log_prefix} 🎯 被@但情绪门槛拦截 烦躁={_mention_annoy:.1f} 压力={_mention_press:.1f}，跳过"
                        )
                else:
                    _final_upgrade = True
                    self._mark_decision_winner("algo_upgrade")
                    logger.info(f"{self.log_prefix} 🎯 被@且算法跳过，升级大模型决定")
        elif _algo_wants_reply and _model_should_reply is True:
            _final_reply = True
            self._mark_decision_winner("algo_reply")
            logger.info(f"{self.log_prefix} 🎯 算法+小模型共识回复")
        elif _algo_wants_reply:
            # algo=1（小模型建议回复）：尊重LLM决策，执行回复
            _final_reply = True
            self._mark_decision_winner("algo_reply")
            logger.info(f"{self.log_prefix} 🎯 小模型建议回复(algo=1)，执行回复")
        elif _algo_wants_upgrade:
            _final_upgrade = True
            self._mark_decision_winner("algo_upgrade")
            logger.info(f"{self.log_prefix} 🎯 算法升级大模型")
        elif _is_admin_forced or _is_gateway_force:
            _final_reply = True
            if _is_gateway_force:
                self._mark_decision_winner("gateway_force")
            else:
                self._mark_decision_winner("algo_reply")
            logger.info(
                f"{self.log_prefix} 💬 强制回复-穿透算法跳过(algo={_llm_call_level}) (admin={_is_admin_forced}, gateway_force={_is_gateway_force})"
            )
        else:
            _final_skip = True
            if legacy_gate == "block":
                self._mark_decision_winner("gateway_block")
            else:
                self._mark_decision_winner("final_skip")
            logger.info(f"{self.log_prefix} 🎯 默认跳过: algo={_llm_call_level}")

        # 动态比例旁路已移除：是否回复完全由LLM内心独白+维度网关force_reply决定

        if _final_skip:
            logger.info(f"{self.log_prefix} 🎯 最终决策: 跳过，直接观察")
            self._last_flow_blocker = self._last_flow_blocker or f"最终决策跳过 algo={_llm_call_level}"
            self._apply_post_reply_state(did_reply=False, reason=self._last_flow_blocker)
            self._emit_flow_decision_summary("final_decision", "skip")
            self._emit_action_verdict("llm_skip", "LLM决策跳过", time.time() - _t0)
            await self._run_peek_observe_loop(incoming_batch)
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True

        # ── 阶段 4.8：LLM 自主规划 - 仅在内心独白不确定时调用 ──
        _t48 = time.time()
        llm_decision = None
        _llm_upgrade_allowed = True
        _upgrade_block_reason = ""
        _upgrade_window_sec = 300.0
        _upgrade_max_calls = 5
        if now < self._llm_upgrade_cooldown_until:
            _llm_upgrade_allowed = False
            _upgrade_block_reason = f"升级冷却中({int(self._llm_upgrade_cooldown_until - now)}s)"
        elif (
            self._llm_upgrade_call_window_start > 0
            and (now - self._llm_upgrade_call_window_start) > _upgrade_window_sec
        ):
            self._llm_upgrade_call_count = 0
            self._llm_upgrade_call_window_start = now
        elif self._llm_upgrade_call_window_start <= 0:
            self._llm_upgrade_call_window_start = now
        if self._llm_upgrade_call_count >= _upgrade_max_calls:
            _llm_upgrade_allowed = False
            _upgrade_block_reason = f"升级次数已达上限({_upgrade_max_calls}/{_upgrade_window_sec}s)"

        if _final_upgrade and _model_governor.tier != "large":
            _final_upgrade = False
            _final_reply = True
            logger.info(
                f"{self.log_prefix} 🎛️ 模型Governor降级升级请求: "
                f"{self._summarize_model_governor(_model_governor)}"
            )
        if _final_upgrade and _llm_upgrade_allowed:
            self._llm_upgrade_call_count += 1
            try:
                _upgrade_timeout = _llm_upgrade_timeout()
                llm_decision = await asyncio.wait_for(
                    self._llm_autonomous_decide(
                        now,
                        relation_result,
                        decision_messages,
                        voice_conclusion,
                    ),
                    timeout=_upgrade_timeout,
                )
                self._llm_upgrade_cooldown_until = now + 60.0
            except Exception as exc:
                logger.debug(f"{self.log_prefix} LLM自主决策异常: {exc}")
            logger.debug(
                f"{self.log_prefix} 🔄 阶段4.8(LLM建议) {time.time() - _t48:.2f}s act={llm_decision.should_act if llm_decision else '?'}"
            )
        elif _final_upgrade:
            logger.info(f"{self.log_prefix} ⚠️ LLM升级决策被阻止: {_upgrade_block_reason}")
        llm_triggered = False
        planner_triggered = False
        planner_decision = None
        should_act = _final_reply
        _pact = ""

        if legacy_gate == "block":
            llm_triggered = False
            should_act = False
        # hesitate 不再关闭决策，只作为风格约束传递到回复生成阶段

        if llm_decision:
            self._llm_content_plan = llm_decision.content_plan or ""
            self._llm_social_intention = llm_decision.social_intention or ""
            logger.debug(
                f"{self.log_prefix} LLM内容建议已接收: "
                f"意图={llm_decision.social_intention}, "
                f"感受={llm_decision.emotional_state}, "
                f"内容规划={llm_decision.content_plan[:50] if llm_decision.content_plan else '无'}"
            )
            if not voice_driven_reply:
                should_act = bool(llm_decision.should_act)
                llm_triggered = bool(llm_decision.should_act)
                if llm_decision.should_act:
                    from types import SimpleNamespace

                    planner_decision = SimpleNamespace(
                        action=SimpleNamespace(value=llm_decision.action_type or "reply"),
                        content_plan=llm_decision.content_plan or "",
                        reason=llm_decision.reasoning or "LLM自主决策允许行动",
                        confidence=float(getattr(llm_decision, "confidence", 0.6) or 0.6),
                        internal_state=llm_decision.emotional_state or "",
                        matched_intent_id="",
                        target_user_id=getattr(llm_decision, "target_user_id", "") or "",
                    )
                    logger.info(
                        f"{self.log_prefix} 🤖 LLM自主决策放行: action={llm_decision.action_type or 'reply'} "
                        f"reason={llm_decision.reasoning[:60] if llm_decision.reasoning else '无'}"
                    )
                else:
                    logger.info(
                        f"{self.log_prefix} 🤖 LLM自主决策保持观察: {llm_decision.reasoning[:60] if llm_decision.reasoning else '无原因'}"
                    )

        should_act = self._apply_inner_voice_priority(
            should_act=should_act,
            voice_conclusion=voice_conclusion,
            pinged_msg=force_reply_message,
            legacy_gate=legacy_gate,
            planner_decision=planner_decision,
            is_admin_forced=_is_admin_forced,
        )
        self._align_states_with_inner_voice(voice_conclusion, source="decision_precheck")

        # 非内心强驱动时，统一规划器始终作为主决策层
        if not voice_driven_reply and planner_decision is None:
            if now < self._planner_quiet_until and not has_user_message:
                quiet_left = int(max(0.0, self._planner_quiet_until - now))
                self._emit_action_verdict(
                    "cooldown_skip",
                    f"规划器冷却中({quiet_left}s)",
                    time.time() - _t0,
                )
                logger.debug(f"{self.log_prefix} 规划器冷却中({quiet_left}s)，跳过重复规划")
                self._last_flow_blocker = f"规划器冷却中({quiet_left}s)"
                self._emit_flow_decision_summary("planner_cooldown", "skip")
                await self._emit_outcome_summary(False, relation_result)
                await asyncio.sleep(_POST_MESSAGE_RETRY_SEC)
                return True

            planner_decision = None
            if legacy_gate in {"allow", "force_reply", "hesitate"}:
                try:
                    planner_decision = await asyncio.wait_for(
                        self._invoke_unified_planner(
                            incoming_batch,
                            repetition_signal=repetition_signal,
                        ),
                        timeout=_parallel_stage_timeout(),
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"{self.log_prefix} ⚠️ 统一规划器超时，跳过规划")
                    planner_decision = None
            if planner_decision and hasattr(planner_decision, "action"):
                _pact = (
                    planner_decision.action.value
                    if hasattr(planner_decision.action, "value")
                    else str(planner_decision.action)
                )
                if _pact in (
                    "reply",
                    "join_conversation",
                    "proactive_speak",
                    "deep_think",
                ):
                    planner_triggered = True
                elif _pact in ("rest", "check_later", "observe", "no_reply"):
                    if self._is_force_wake_admin(incoming_batch):
                        logger.info(f"{self.log_prefix} 👑 管理员消息-规划器{_pact}被覆盖为reply")
                        _pact = "reply"
                    else:
                        self._apply_planner_state_update(
                            _pact,
                            reason=str(getattr(planner_decision, "reason", "") or "")[:80],
                        )
                        cooldown = float(
                            getattr(
                                planner_decision,
                                "next_check_seconds",
                                _POST_MESSAGE_RETRY_SEC,
                            )
                            or _POST_MESSAGE_RETRY_SEC
                        )
                        cooldown = max(_POST_MESSAGE_RETRY_SEC, min(cooldown, 30.0))
                        self._planner_quiet_until = now + cooldown
                elif _pact == "wait":
                    if self._is_force_wake_admin(incoming_batch):
                        logger.info(f"{self.log_prefix} 👑 管理员消息-规划器wait被覆盖为reply")
                        _pact = "reply"
                    else:
                        _anticipated = str(getattr(planner_decision, "reason", "") or "")[:80]
                        self._apply_planner_state_update(
                            "wait",
                            reason=f"planner决策wait: {_anticipated[:40]}",
                            anticipated=_anticipated,
                        )
                        cooldown = float(
                            getattr(
                                planner_decision,
                                "next_check_seconds",
                                _POST_MESSAGE_RETRY_SEC,
                            )
                            or _POST_MESSAGE_RETRY_SEC
                        )
                        cooldown = max(_POST_MESSAGE_RETRY_SEC, min(cooldown, 30.0))
                        self._planner_quiet_until = now + cooldown
                logger.info(f"{self.log_prefix} 规划器决策={_pact} 门控={legacy_gate} 应行动={should_act}")

        should_act = self._apply_inner_voice_priority(
            should_act=should_act,
            voice_conclusion=voice_conclusion,
            pinged_msg=force_reply_message,
            legacy_gate=legacy_gate,
            planner_decision=planner_decision,
            is_admin_forced=_is_admin_forced,
        )
        self._align_states_with_inner_voice(voice_conclusion, source="decision_finalize")

        _voice_reluctant = (getattr(voice_conclusion, "should_reply", None) is False) if voice_conclusion else False
        _voice_eager = bool(
            voice_conclusion
            and getattr(voice_conclusion, "should_reply", None) is True
            and int(getattr(voice_conclusion, "reply_desire_level", 0) or 0) >= 6
        )
        _intent_primary = None
        _intent_silence = 0
        if voice_conclusion:
            try:
                _intent_primary = str(getattr(voice_conclusion, "primary_intent", None) or "")
                _intent_silence = int(getattr(voice_conclusion, "silence_rounds", 0) or 0)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        _has_strong_reply_evidence = bool(
            _force_direct_ping or planner_decision is not None or voice_driven_reply or _is_admin_forced
        )
        if not should_act and legacy_gate in {"allow", "force_reply"}:
            if _voice_reluctant:
                logger.info(f"{self.log_prefix} ⚠️ 门控={legacy_gate}但独白显抗拒，降级为规划器决策")
            elif not _has_strong_reply_evidence:
                logger.info(f"{self.log_prefix} 🧊 门控={legacy_gate}但缺少强回复证据，保持不行动")
            else:
                _rel_snap = self._resolve_relation_view()
                _snap_annoyance = float(_rel_snap.get("annoyance_value", 0.0) or 0.0)
                _snap_pressure = float(_rel_snap.get("psychological_pressure", 0.0) or 0.0)
                _snap_blocked = bool(_rel_snap.get("is_user_blocked", False))

                if _snap_blocked or (_snap_annoyance >= 80 and _snap_pressure >= 60):
                    logger.info(
                        f"{self.log_prefix} 🛑 门控={legacy_gate}但情绪极端"
                        f"(烦躁{_snap_annoyance:.0f}/压力{_snap_pressure:.0f}"
                        f"{'/已屏蔽' if _snap_blocked else ''})，拒绝激活"
                    )
                    should_act = False
                elif _snap_annoyance >= 60 and _snap_pressure >= 40:
                    import random as _emo_random

                    if _emo_random.random() < 0.50:
                        logger.info(
                            f"{self.log_prefix} ⚠️ 门控={legacy_gate}但情绪偏高"
                            f"(烦躁{_snap_annoyance:.0f}/压力{_snap_pressure:.0f})，50%概率跳过"
                        )
                        should_act = False
                    elif _has_strong_reply_evidence:
                        should_act = True
                        logger.info(
                            f"{self.log_prefix} 统一结算 门控={legacy_gate}，强证据存在且情绪偏高但允许进入回复执行层"
                        )
                else:
                    if _intent_primary == "wait" and _intent_silence >= 1:
                        import random as _int_random

                        _skip_prob = min(0.75, 0.35 + _intent_silence * 0.15)
                        if _int_random.random() < _skip_prob:
                            should_act = False
                            logger.info(
                                f"{self.log_prefix} 🤐 意图池primary=wait(静默{_intent_silence}轮)"
                                f"，{int(_skip_prob * 100):.0f}%概率跳过"
                            )
                        elif _has_strong_reply_evidence:
                            should_act = True
                            logger.info(
                                f"{self.log_prefix} 统一结算 门控={legacy_gate}，强证据存在且意图wait但掷骰通过，允许回复"
                            )
                    elif _has_strong_reply_evidence:
                        should_act = True
                        logger.info(f"{self.log_prefix} 统一结算 门控={legacy_gate}，强证据存在，允许进入回复执行层")

        try:
            target_uid = str(getattr(self, "_last_user_id", "") or "").strip()
            if target_uid:
                from src.modules.modcore.dynamic_persona.emotion_tracker import (
                    get_emotion_tracker,
                )

                _state = get_emotion_tracker(self.stream_id).get_user_state(target_uid, create_if_missing=False)
                if _state is not None:
                    _inner_chaos = float(getattr(_state, "inner_chaos", 0.0) or 0.0)
                    _surface_mask = float(getattr(_state, "surface_mask", 0.0) or 0.0)
                    _submission = float(getattr(_state, "submission_level", 0.0) or 0.0)
                    _shyness = float(getattr(_state, "shyness_level", 50.0) or 50.0)
                    if _inner_chaos >= 8.0:
                        logger.info(f"{self.log_prefix} 🧠 内心混乱过高({_inner_chaos:.1f})，本轮拒绝行动")
                        should_act = False
                    elif should_act and not _voice_eager:
                        if _surface_mask >= 7.0:
                            should_act = random.random() < 0.70
                            logger.info(f"{self.log_prefix} 🎭 表层伪装偏高({_surface_mask:.1f})，收缩行动意愿")
                        if should_act and _submission >= 6.0:
                            should_act = random.random() < 0.75
                            logger.info(f"{self.log_prefix} 🙈 顺从度偏高({_submission:.1f})，降低主动表达概率")
                        if should_act and _shyness >= 75.0 and legacy_gate not in {"force_reply"}:
                            should_act = random.random() < 0.65
                            logger.info(f"{self.log_prefix} 😳 害羞值偏高({_shyness:.1f})，非强制场景收缩回复概率")
        except Exception as _psych_exc:
            logger.debug(f"{self.log_prefix} 心理特征门控异常: {_psych_exc}")

        # 非强制场景启用频率概率调制，防止过度密集发言
        if should_act and not _is_admin_forced and not _force_direct_ping and not _voice_eager:
            try:
                _prob_allow = self._decide_action(
                    pinged_msg=force_reply_message,
                    eagerness=eagerness_val,
                    awareness=awareness_snapshot,
                    voice=voice_conclusion,
                    messages=decision_messages,
                )
                if not _prob_allow:
                    should_act = False
                    logger.info(f"{self.log_prefix} 🎛️ 频率概率调制: 当前轮降级为观察，避免话痨")
            except Exception as _freq_decide_exc:
                logger.debug(f"{self.log_prefix} 频率概率调制异常(降级忽略): {_freq_decide_exc}")

        _recent_reply_burst = sum(1 for t in self._bot_reply_timeline if time.time() - t < 180.0)
        if should_act and _recent_reply_burst >= 4 and not _is_admin_forced and not _force_direct_ping and not _voice_eager:
            should_act = False
            logger.info(f"{self.log_prefix} 🧯 连续发言过多({_recent_reply_burst}/180s)，本轮先闭嘴避免像刷屏")

        # SOC-03: 负面情绪影响决策 — 高负面情绪时触发回避/防御
        _rel_snap_soc = self._resolve_relation_view()
        _neg_emo = float(_rel_snap_soc.get("annoyance_value", 0.0) or 0.0)
        _soc_pressure = float(_rel_snap_soc.get("psychological_pressure", 0.0) or 0.0)
        _neg_composite = max(_neg_emo, _soc_pressure * 0.8)
        _is_force = (force_reply_message is not None) or voice_driven_reply or _is_admin_forced
        if should_act and _neg_composite > 85 and not _is_force and not _voice_eager:
            should_act = False
            if _is_admin_forced:
                should_act = True
                logger.info(
                    f"{self.log_prefix} 👑 管理员强制唤醒-穿透SOC-03负面情绪回避(composite={_neg_composite:.1f}>85), 强制行动"
                )
            else:
                logger.info(f"{self.log_prefix} SOC-03 负面情绪回避: composite={_neg_composite:.1f}>85, 降级为不行动")
        elif should_act and _neg_composite > 60 and not _is_force and not _voice_eager:
            if random.random() < 0.4:
                if _is_admin_forced:
                    logger.info(
                        f"{self.log_prefix} 👑 管理员强制唤醒-穿透SOC-03概率回避(neg={_neg_emo:.1f}>60), 强制行动"
                    )
                else:
                    should_act = False
                    logger.info(f"{self.log_prefix} SOC-03 负面情绪概率回避: neg={_neg_emo:.1f}>60, 随机跳过")

        if should_act:
            _pre_reply_resource_snapshot = self._capture_pre_reply_resource_snapshot()
            self._apply_plan_drain()
            await self._deduct_shared_resources(incoming_batch)
            self._last_proactive_intent_id = ""  # 每次行动前清空，避免误关联
            _voice_needs_upgrade = False
            if voice_conclusion is not None and hasattr(voice_conclusion, "needs_upgrade"):
                _voice_needs_upgrade = bool(voice_conclusion.needs_upgrade)
            if _voice_needs_upgrade and not _llm_upgrade_allowed:
                logger.info(
                    f"{self.log_prefix} ⚠️ 小模型判定需要升级但大模型不可用: {_upgrade_block_reason}，降级为小模型直接回复"
                )
                _voice_needs_upgrade = False
            elif _voice_needs_upgrade:
                logger.info(
                    f"{self.log_prefix} 🔄 小模型判定需要升级至大模型深度分析（调用计数:{
                        self._llm_upgrade_call_count + 1
                    }/{_upgrade_max_calls}）"
                )
            # 核心路径分流：算法+小模型联合决策
            if _final_reply:
                if _voice_needs_upgrade and _llm_upgrade_allowed:
                    self._llm_upgrade_call_count += 1
                    _upgrade_decision = None
                    try:
                        _upgrade_decision = await asyncio.wait_for(
                            self._llm_autonomous_decide(
                                now,
                                relation_result,
                                decision_messages,
                                voice_conclusion,
                            ),
                            timeout=_llm_upgrade_timeout(),
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"{self.log_prefix} ⚠️ 大模型深度分析超时，跳过升级决策")
                    self._llm_upgrade_cooldown_until = now + 60.0
                    if _upgrade_decision and hasattr(_upgrade_decision, "should_act") and _upgrade_decision.should_act:
                        logger.info(
                            f"{self.log_prefix} 🔄 大模型深度分析完成，决定回复: "
                            f"{_upgrade_decision.content_plan[:50] if _upgrade_decision.content_plan else '无内容规划'} "
                            f"感受={_upgrade_decision.emotional_state} 意图={_upgrade_decision.social_intention}"
                        )
                        _upgrade_proactive = type(
                            "UpgradeDecision",
                            (),
                            {
                                "action": type(
                                    "ActionValue", (), {"value": _upgrade_decision.action_type or "reply"}
                                )(),
                                "content_plan": _upgrade_decision.content_plan,
                                "reason": _upgrade_decision.reasoning,
                                "confidence": _upgrade_decision.confidence,
                                "internal_state": _upgrade_decision.emotional_state,
                                "matched_intent_id": "",
                                "target_user_id": "",
                            },
                        )()
                        actual_reply_made = await self._execute_planned_reply(
                            decision_messages=decision_messages,
                            planner_decision=_upgrade_proactive,
                            force_reply_message=None,
                            repetition_signal=repetition_signal,
                            decision_context_packet=decision_context_packet,
                        )
                    else:
                        logger.info(f"{self.log_prefix} 🔄 大模型深度分析决定不回复，观察")
                        actual_reply_made = await self._observe(
                            recent_messages_list=decision_messages,
                            force_reply_message=force_reply_message,
                        )
                else:
                    try:
                        actual_reply_made = await asyncio.wait_for(
                            self._execute_voice_driven_reply(
                                voice_conclusion=voice_conclusion,
                                incoming_batch=decision_messages,
                                force_reply_message=force_reply_message,
                            ),
                            timeout=120.0,
                        )
                    except asyncio.TimeoutError:
                        actual_reply_made = False
                        logger.error(f"{self.log_prefix} ⚠️ 内心驱动回复超时(120s)")
            elif (
                planner_decision is not None
                and hasattr(planner_decision, "action")
                and (
                    planner_decision.action.value
                    if hasattr(planner_decision.action, "value")
                    else str(planner_decision.action)
                )
                == "proactive_speak"
            ):
                # 统一规划器决定主动发言 → 走专用主动回复执行器
                from src.chat.heart_flow.llm_autonomous_planner import (
                    AutonomousDecision,
                )

                # 捕获规划器命中的意图id，供闭环追踪
                _planner_intent_id = str(getattr(planner_decision, "matched_intent_id", "") or "").strip()
                self._last_proactive_intent_id = _planner_intent_id
                _proactive_decision = AutonomousDecision(
                    should_act=True,
                    action_type="proactive_speak",
                    content_plan=getattr(planner_decision, "content_plan", "") or "",
                    reasoning=getattr(planner_decision, "reason", "") or "",
                    confidence=getattr(planner_decision, "confidence", 0.6),
                    emotional_state=getattr(planner_decision, "internal_state", "") or "",
                    social_intention=self._llm_social_intention or "想主动聊几句",
                )
                try:
                    actual_reply_made = await asyncio.wait_for(
                        self._execute_proactive_reply(
                            llm_decision=_proactive_decision,
                            incoming_batch=decision_messages,
                        ),
                        timeout=120.0,
                    )
                except asyncio.TimeoutError:
                    actual_reply_made = False
                    logger.error(f"{self.log_prefix} ⚠️ 主动回复超时(120s)")
                if actual_reply_made:
                    logger.info(f"{self.log_prefix} 统一规划器触发主动回复成功")
                else:
                    logger.info(f"{self.log_prefix} 主动回复生成失败，降级到常规观察")
                    actual_reply_made = await self._observe(
                        recent_messages_list=decision_messages,
                        force_reply_message=force_reply_message,
                    )
            else:
                try:
                    actual_reply_made = await asyncio.wait_for(
                        self._execute_planned_reply(
                            decision_messages=decision_messages,
                            planner_decision=planner_decision,
                            force_reply_message=force_reply_message,
                            repetition_signal=repetition_signal,
                            decision_context_packet=decision_context_packet,
                        ),
                        timeout=120.0,
                    )
                except asyncio.TimeoutError:
                    actual_reply_made = False
                    logger.error(f"{self.log_prefix} ⚠️ 计划回复超时(120s)")
            if actual_reply_made:
                self._last_flow_blocker = ""
                self._emit_flow_decision_summary("final_decision", "reply")
                _reply_source = (
                    "voice_driven"
                    if voice_driven_reply
                    else (
                        "llm_autonomous"
                        if llm_triggered
                        else ("planner_proactive" if _pact == "proactive_speak" else "planner")
                    )
                )
                _was_proactive_reply = _reply_source == "planner_proactive"
                await self._finalize_reply_settlement(
                    incoming_batch=incoming_batch,
                    relation_result=relation_result,
                    voice_driven_reply=voice_driven_reply,
                    planner_triggered=planner_triggered,
                    did_reply=True,
                    behavior_signal=behavior_signal,
                    repetition_signal=repetition_signal,
                    harassment_signal=harassment_signal,
                    proactive_reply=_was_proactive_reply,
                )
                self._apply_post_reply_state(did_reply=True, reason="回复完成")
                await self._emit_outcome_summary(True, relation_result)
                self._run_post_action_side_effects(
                    did_reply=True,
                    incoming_batch=incoming_batch,
                    planner_decision=planner_decision,
                    reply_text=str(getattr(self, "_last_reply_content", "") or ""),
                    was_proactive=_was_proactive_reply,
                )
                # 第三层：回复成功裁定
                _reply_confidence = getattr(planner_decision, "confidence", 0.6) if planner_decision else 0.5
                self._emit_action_verdict(
                    f"reply({_reply_source})",
                    "回复已发送",
                    time.time() - _t0,
                    confidence=_reply_confidence,
                )
            else:
                # 第三层：规划器决定不回复
                _no_reply_reason = str(getattr(self, "_last_flow_blocker", "") or "").strip() or "规划器返回no_reply"
                self._restore_pre_reply_resource_snapshot(
                    _pre_reply_resource_snapshot,
                    reason=_no_reply_reason,
                )
                self._emit_action_verdict("no_reply", _no_reply_reason, time.time() - _t0)
                logger.info(f"{self.log_prefix} ⏭️ 本轮未形成有效回复: {_no_reply_reason}")
                self._apply_post_reply_state(did_reply=False, reason=_no_reply_reason)
                await self._emit_outcome_summary(False, relation_result)
                self._run_post_action_side_effects(
                    did_reply=False,
                    incoming_batch=incoming_batch,
                    planner_decision=planner_decision,
                    inaction_reason=_no_reply_reason,
                )
            logger.info(f"{self.log_prefix} 🔄 完整管线耗时 {time.time() - _t0:.2f}s")
        else:
            # 第三层：决定不行动
            _inaction_reason = "统一规划器决定不行动"
            if planner_decision and hasattr(planner_decision, "action"):
                _pa = (
                    planner_decision.action.value
                    if hasattr(planner_decision.action, "value")
                    else str(planner_decision.action)
                )
                _inaction_reason = f"规划器裁定={_pa}"
                if hasattr(planner_decision, "reason") and planner_decision.reason:
                    _inaction_reason += f" {planner_decision.reason[:40]}"
            self._emit_action_verdict("no_action", _inaction_reason, time.time() - _t0)
            self._apply_post_reply_state(did_reply=False, reason=_inaction_reason)
            await self._emit_outcome_summary(False, relation_result)
            self._run_post_action_side_effects(
                did_reply=False,
                incoming_batch=incoming_batch,
                planner_decision=planner_decision,
                inaction_reason=_inaction_reason,
            )
            logger.info(f"{self.log_prefix} 🔄 决定不行动，管线耗时 {time.time() - _t0:.2f}s")
            await asyncio.sleep(_POST_MESSAGE_RETRY_SEC if has_user_message else 10)
            return True

        return True

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

    def _query_flow_phase(self) -> str:
        """查询当前频道的心流阶段，从世界快照读取"""
        snap = getattr(self, "_tick_world_snapshot", None)
        if snap is not None and snap.subject.flow_phase:
            return snap.subject.flow_phase
        # 快照尚未构建（首轮tick），默认待命而非活跃，避免刚启动就进入强参与
        return "standby"

    def _evaluate_phase_gate(self, phase_str: str, now: float) -> str:
        """根据心流阶段决定本轮循环的行为

        返回值：
        - "proceed"  继续完整流水线
        - "skip"     跳过本轮（休息期不响应）
        - "glance"   休息期窥屏
        """
        if phase_str == "dormant":
            return self._handle_dormant_gate()
        if phase_str == "pending":
            return self._handle_pending_gate(now)
        if phase_str == "standby":
            # 待命阶段正常进入流水线，但迁移至 ENGAGED
            _snap = self._build_unified_flow_snapshot("phase_gate")
            _snap.watch_state = WatchLevel.ACTIVE_WATCH.value
            _snap.phase = FlowPhase.ENGAGED.value
            self._apply_unified_flow_snapshot(
                _snap,
                watch_reason="phase_gate:message_arrival",
                phase_reason="待命→活跃：消息到达",
            )
            return "proceed"
        return "proceed"

    def _handle_dormant_gate(self) -> str:
        """休息期门控：委托 DormancySupervisor 决策"""
        try:
            from src.chat.heart_flow.rest_handler import (
                get_dormancy_supervisor,
            )
            from src.common.data_models.heartflow_models import DormantReaction

            supervisor = get_dormancy_supervisor()
            rest_snapshot = supervisor.snapshot(self.stream_id)
            reaction = supervisor.on_incoming_msg(self.stream_id)
            if reaction == DormantReaction.AWAKEN:
                try:
                    from src.core.night_cycle_system import get_night_cycle

                    _ncs_reset = get_night_cycle(self.stream_id)
                    _ncs_reset.reset_night_counters()
                    _d6 = EnergyChainDimension.get_instance()
                    _d6s = _d6._ensure_channel(self.stream_id)
                    _d6s.night_reply_count = 0
                    _grace = 120.0 if _d6._get_night_mode(self.stream_id).phase == "DEEP_VALLEY" else 60.0
                    self._wake_grace_until = time.time() + _grace
                    logger.info(
                        f"{self.log_prefix} 😴 动态唤醒宽限期={_grace:.0f}s 阶段={_d6._get_night_mode(self.stream_id).phase}"
                    )
                except Exception:
                    self._wake_grace_until = time.time() + 120.0
                self._apply_attention_transition(
                    target_watch=WatchLevel.ACTIVE_WATCH,
                    watch_reason="休息结束被唤醒",
                    target_phase=FlowPhase.ENGAGED,
                    phase_reason="休息结束被唤醒",
                )
                self._last_flow_blocker = "休息期结束/反悔唤醒"
                return "proceed"
            if reaction == DormantReaction.GLANCE:
                self._last_flow_blocker = f"休息中仅窥屏 remaining={rest_snapshot.get('remaining_sec', 0)}s"
                return "glance"
            self._last_flow_blocker = f"休息中跳过 remaining={rest_snapshot.get('remaining_sec', 0)}s"
            return "skip"
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 休息门控异常，放行: {exc}")
            return "proceed"

    def _estimate_dynamic_rest_profile(self, voice_conclusion) -> Dict[str, float]:
        """根据内心独白、情感状态和资源状态估算休息画像。"""
        desire = int(getattr(voice_conclusion, "reply_desire_level", 5) or 5) if voice_conclusion else 5
        mood_text = str(getattr(voice_conclusion, "current_mood", "") or "").lower() if voice_conclusion else ""
        thought_text = str(getattr(voice_conclusion, "thinking", "") or "") if voice_conclusion else ""
        emotion = getattr(self, "_cached_emotion_state", {}) or {}
        boredom = float(emotion.get("boredom", 0.0) or 0.0)
        fatigue = float(emotion.get("environmental_fatigue", 0.0) or 0.0)
        loneliness = float(emotion.get("loneliness", 0.0) or 0.0)
        snap = getattr(self, "_tick_world_snapshot", None)
        energy_ratio = 0.6
        thinking_ratio = 0.6
        if snap is not None:
            try:
                res = snap.self_resources
                energy_ratio = float((res.chat_ratio() + res.thinking_ratio()) / 2.0)
                thinking_ratio = float(res.thinking_ratio())
            except Exception as exc:
                logger.debug(f"{self.log_prefix} 读取自我资源比例异常: {exc}")
                energy_ratio = 0.5
                thinking_ratio = 0.5
        rest_intensity = 0.35 + max(0.0, (6 - desire)) * 0.09 + max(0.0, 0.55 - energy_ratio) * 0.4
        rest_intensity += max(0.0, 0.5 - thinking_ratio) * 0.35 + fatigue * 0.35
        if any(tag in mood_text for tag in ("tired", "疲", "累", "烦", "倦")):
            rest_intensity += 0.12
        wake_drive = boredom * 0.45 + loneliness * 0.35 + max(0.0, desire - 3) * 0.04
        if "等会" in thought_text or "一会" in thought_text or "稍后" in thought_text:
            wake_drive += 0.08
        regret_chance = boredom * 0.28 + loneliness * 0.22 + max(0.0, desire - 4) * 0.03 - fatigue * 0.18
        return {
            "rest_intensity": max(0.15, min(1.0, rest_intensity)),
            "wake_drive": max(0.0, min(1.0, wake_drive)),
            "regret_chance": max(0.0, min(0.8, regret_chance)),
            "min_rest_sec": 45 if desire >= 4 else 90,
        }

    def _enter_dynamic_rest(self, *, cause: str, trigger: str, blackout: bool = False) -> None:
        """统一进入动态休息，并同步观看状态。"""
        try:
            from src.chat.heart_flow.rest_handler import DormancySupervisor

            rest_profile = self._estimate_dynamic_rest_profile(getattr(self, "_cached_voice", None))
            DormancySupervisor.instance().begin_dynamic_dormancy(
                self.stream_id,
                cause=cause,
                rest_intensity=rest_profile["rest_intensity"],
                wake_drive=rest_profile["wake_drive"],
                regret_chance=rest_profile["regret_chance"],
                min_rest_sec=int(rest_profile["min_rest_sec"]),
            )
            self._last_flow_blocker = (
                f"{trigger}: 动态休息 rest={rest_profile['rest_intensity']:.2f} "
                f"wake={rest_profile['wake_drive']:.2f} regret={rest_profile['regret_chance']:.2f}"
            )
        except Exception as _rest_err:
            logger.debug(f"{self.log_prefix} 动态休息触发异常: {_rest_err}")
        self._apply_attention_transition(
            target_watch=WatchLevel.BLACKOUT if blackout else WatchLevel.PEEK,
            watch_reason=trigger,
            target_phase=FlowPhase.DORMANT,
            phase_reason=cause,
        )

    def _apply_attention_transition(
        self,
        *,
        target_watch=None,
        watch_reason: str = "",
        target_phase: Optional[FlowPhase] = None,
        phase_reason: str = "",
    ) -> None:
        """统一推进观看态与阶段态，避免多处各自改状态。"""
        if target_watch is not None:
            try:
                _wm = self._orch.get("watch_machine")
                _wm.signal_force_level(target_watch, trigger=watch_reason)
                self._cached_watch_level = _wm.current_level
            except Exception as exc:
                logger.debug(f"{self.log_prefix} 关注状态推进失败({watch_reason or target_watch}): {exc}")
        if target_phase is not None:
            try:
                from src.chat.heart_flow.state_machine import get_phase_coordinator

                coordinator = get_phase_coordinator()
                coordinator.register_channel(self.stream_id)
                shifted = coordinator.attempt_shift(
                    self.stream_id,
                    target_phase,
                    reason=phase_reason or watch_reason or "统一状态推进",
                )
                if not shifted and coordinator.current_phase(self.stream_id) != target_phase:
                    logger.debug(
                        f"{self.log_prefix} 阶段推进未生效 {coordinator.current_phase(self.stream_id).value}->{target_phase.value}"
                    )
            except Exception as exc:
                logger.debug(f"{self.log_prefix} 阶段推进失败({phase_reason or target_phase}): {exc}")

    def _align_states_with_inner_voice(self, voice_conclusion, *, source: str = "") -> None:
        """根据内心独白统一同步观看态与阶段态。"""
        if voice_conclusion is None or not getattr(voice_conclusion, "is_valid", False):
            return
        action = str(getattr(voice_conclusion, "next_action", "") or "").strip().lower()
        desire = int(getattr(voice_conclusion, "reply_desire_level", 5) or 5)
        if action in {"rest", "disengage"}:
            return
        if action == "lurk":
            self._apply_attention_transition(
                target_watch=WatchLevel.PEEK,
                watch_reason=f"{source or 'inner_voice'}:潜水",
                target_phase=FlowPhase.STANDBY,
                phase_reason="内心选择潜水，退回待命",
            )
            return
        if getattr(voice_conclusion, "should_reply", None) is True and desire >= 6:
            self._apply_attention_transition(
                target_watch=WatchLevel.ENGAGED,
                watch_reason=f"{source or 'inner_voice'}:想参与",
                target_phase=FlowPhase.ENGAGED,
                phase_reason="内心明确想参与",
            )
            return
        if getattr(voice_conclusion, "should_reply", None) is False and desire <= 4:
            self._apply_attention_transition(
                target_watch=WatchLevel.PEEK,
                watch_reason=f"{source or 'inner_voice'}:先观察",
                target_phase=FlowPhase.STANDBY,
                phase_reason="内心倾向观察",
            )

    def _build_unified_flow_snapshot(self, final_decision: str = "") -> UnifiedFlowSnapshot:
        """聚合当前行为层核心状态，作为统一快照。"""
        voice = getattr(self, "_cached_voice", None)
        emotion = getattr(self, "_cached_emotion_state", {}) or {}
        watch_level = getattr(self, "_cached_watch_level", None)
        watch_text = watch_level.value if hasattr(watch_level, "value") else str(watch_level or "unknown")
        phase_text = FlowPhase.STANDBY.value
        try:
            from src.chat.heart_flow.state_machine import get_phase_coordinator

            coordinator = get_phase_coordinator()
            coordinator.register_channel(self.stream_id)
            phase_text = coordinator.current_phase(self.stream_id).value
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 阶段协调器注册异常: {exc}")
        rest_snapshot = {"resting": False}
        try:
            from src.chat.heart_flow.rest_handler import get_dormancy_supervisor

            rest_snapshot = get_dormancy_supervisor().snapshot(self.stream_id)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 休眠监督器快照异常: {exc}")
        snapshot = UnifiedFlowSnapshot(
            channel_id=self.stream_id,
            phase=phase_text,
            watch_state=watch_text,
            is_resting=bool(rest_snapshot.get("resting", False)),
            rest_remaining_sec=float(rest_snapshot.get("remaining_sec", 0.0) or 0.0),
            rest_intensity=float(rest_snapshot.get("rest_intensity", 0.0) or 0.0),
            wake_drive=float(rest_snapshot.get("wake_drive", 0.0) or 0.0),
            reply_desire=int(getattr(voice, "reply_desire_level", 5) or 5) if voice else 5,
            voice_action=str(getattr(voice, "next_action", "") or "") if voice else "",
            voice_should_reply=getattr(voice, "should_reply", None) if voice else None,
            voice_mood=str(getattr(voice, "current_mood", "") or "") if voice else "",
            voice_thinking=str(getattr(voice, "thinking", "") or "")[:120] if voice else "",
            boredom=float(emotion.get("boredom", 0.0) or 0.0),
            environmental_fatigue=float(emotion.get("environmental_fatigue", 0.0) or 0.0),
            loneliness=float(emotion.get("loneliness", 0.0) or 0.0),
            social_desire=float(emotion.get("social_desire", 0.0) or 0.0),
            blocker=self._last_flow_blocker,
            final_decision=final_decision,
        )
        self._last_unified_flow_snapshot = snapshot
        return snapshot

    def _copy_unified_flow_snapshot(
        self,
        snapshot: UnifiedFlowSnapshot,
        *,
        update: Optional[Dict[str, Any]] = None,
    ) -> UnifiedFlowSnapshot:
        """兼容 dataclass / Pydantic 风格的统一快照复制更新。"""
        update = dict(update or {})
        try:
            model_copy = getattr(snapshot, "model_copy", None)
            if callable(model_copy):
                return model_copy(update=update)
            copy_method = getattr(snapshot, "copy", None)
            if callable(copy_method):
                try:
                    return copy_method(update=update)
                except TypeError:
                    return copy_method(**update)
            if is_dataclass(snapshot):
                return dataclass_replace(snapshot, **update)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 统一快照复制失败，回退重建: {exc}")
        payload = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(getattr(snapshot, "__dict__", {}))
        payload.update(update)
        return UnifiedFlowSnapshot.from_dict(payload)

    def _apply_unified_flow_snapshot(
        self,
        snapshot: UnifiedFlowSnapshot,
        *,
        watch_reason: str = "",
        phase_reason: str = "",
    ) -> None:
        """通过统一快照单入口回写观看态与阶段态。"""
        watch_target = None
        watch_map = {
            "blackout": WatchLevel.BLACKOUT,
            "peek": WatchLevel.PEEK,
            "skim_window": WatchLevel.SKIM_WINDOW,
            "active_watch": WatchLevel.ACTIVE_WATCH,
            "engaged": WatchLevel.ENGAGED,
        }
        phase_target = None
        try:
            if snapshot.watch_state in watch_map:
                watch_target = watch_map[snapshot.watch_state]
            phase_target = FlowPhase(snapshot.phase)
        except Exception:
            phase_target = None
        self._apply_attention_transition(
            target_watch=watch_target,
            watch_reason=watch_reason or snapshot.final_decision or "统一快照回写",
            target_phase=phase_target,
            phase_reason=phase_reason or snapshot.blocker or snapshot.final_decision or "统一快照回写",
        )

    def _run_flow_side_effects(
        self,
        *,
        action_name: str,
        reason: str = "",
        anticipated: str = "",
        conclude_rest: bool = False,
    ) -> None:
        """统一执行状态回写后的副作用管理。"""
        try:
            if action_name == "begin_rest":
                from src.chat.heart_flow.rest_handler import get_dormancy_supervisor

                get_dormancy_supervisor().begin_dormancy(self.stream_id, cause=reason)
                return
            if action_name == "end_rest":
                from src.chat.heart_flow.rest_handler import get_dormancy_supervisor

                get_dormancy_supervisor().conclude_dormancy(self.stream_id)
                return
            if action_name == "begin_wait" and anticipated:
                from src.chat.heart_flow.waiting_handler import get_pending_orchestrator

                get_pending_orchestrator().initiate_watch(self.stream_id, anticipated=anticipated)
                return
            if action_name == "pending_timeout":
                from src.chat.heart_flow.waiting_handler import get_pending_orchestrator

                get_pending_orchestrator().conclude_with_timeout_consequence(self.stream_id)
                return
            if action_name == "engage" and conclude_rest:
                from src.chat.heart_flow.rest_handler import get_dormancy_supervisor

                get_dormancy_supervisor().conclude_dormancy(self.stream_id)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 状态副作用执行失败({action_name}): {exc}")

    def _run_post_action_side_effects(
        self,
        *,
        did_reply: bool,
        incoming_batch: List,
        planner_decision: Optional[Any] = None,
        reply_text: str = "",
        was_proactive: bool = False,
        inaction_reason: str = "",
    ) -> None:
        """统一执行回复后或跳过后的外围副作用。"""
        try:
            from src.chat.proactive.intention_pool import get_intention_pool

            _pool = get_intention_pool()
            _matched_intent_id = (
                str(getattr(planner_decision, "matched_intent_id", "") or "").strip() if planner_decision else ""
            )
            if did_reply:
                _completed_precise = False
                if _matched_intent_id:
                    _completed_precise = bool(_pool.complete_intent(_matched_intent_id))
                    if _completed_precise:
                        logger.info(f"{self.log_prefix} 意图精确结案 id={_matched_intent_id[:10]}")
                if not _completed_precise:
                    _target = (
                        str(getattr(planner_decision, "target_user_id", "") or "").strip()
                        or str(getattr(self, "_last_user_id", "") or "").strip()
                    )
                    if _target:
                        _count = _pool.complete_by_target(self.stream_id, _target, limit=1)
                        if _count > 0:
                            logger.info(f"{self.log_prefix} 意图回退结案 target={_target[:8]} count={_count}")
            elif _matched_intent_id:
                _reason = (
                    str(getattr(planner_decision, "reason", "") or "").strip() or inaction_reason or "planner_no_reply"
                )
                if _pool.fail_intent(_matched_intent_id, reason=_reason[:120]):
                    logger.info(
                        f"{self.log_prefix} 意图精确失败回写 id={_matched_intent_id[:10]} reason={_reason[:30]}"
                    )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 意图副作用执行失败: {exc}")

        try:
            if not did_reply:
                self._integrate_adaptive_collection(incoming_batch, False, "")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 自适应收集副作用失败: {exc}")

        if not did_reply:
            try:
                _was_at_bot = any(getattr(_m, "is_at_bot", False) for _m in incoming_batch if hasattr(_m, "is_at_bot"))
                if _was_at_bot:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import get_emotion_tracker

                    _emo_t = get_emotion_tracker(self.stream_id)
                    for _m in reversed(incoming_batch):
                        _uid = str(getattr(_m, "user_id", "") or "").strip()
                        if self._is_human_message_obj(_m):
                            _emo_t.update_psychological_pressure(_uid, 2.0, "被@未回复")
                            break
            except Exception as exc:
                logger.debug(f"{self.log_prefix} 未回复压力副作用失败: {exc}")

    def _apply_planner_state_update(self, action_name: str, *, reason: str = "", anticipated: str = "") -> None:
        """将规划器的非回复动作统一映射到状态快照。"""
        snapshot = self._build_unified_flow_snapshot(action_name)
        if action_name == "wait":
            self._apply_unified_flow_snapshot(
                self._copy_unified_flow_snapshot(
                    snapshot,
                    update={
                        "watch_state": WatchLevel.ACTIVE_WATCH.value,
                        "phase": FlowPhase.PENDING.value,
                        "blocker": reason or "规划器决定等待",
                    }
                ),
                watch_reason="planner:wait",
                phase_reason=reason or "规划器决定等待",
            )
            self._run_flow_side_effects(action_name="begin_wait", anticipated=anticipated)
            return
        if action_name in {"rest", "check_later", "observe", "no_reply"}:
            self._apply_unified_flow_snapshot(
                self._copy_unified_flow_snapshot(
                    snapshot,
                    update={
                        "watch_state": WatchLevel.PEEK.value,
                        "phase": FlowPhase.STANDBY.value,
                        "blocker": reason or f"规划器决定{action_name}",
                    }
                ),
                watch_reason=f"planner:{action_name}",
                phase_reason=reason or f"规划器决定{action_name}",
            )

    def _apply_post_reply_state(self, *, did_reply: bool, reason: str = "") -> None:
        """回复或跳过后统一回写行为状态。"""
        snapshot = self._build_unified_flow_snapshot("reply_completed" if did_reply else "reply_skipped")
        if did_reply:
            snapshot = self._copy_unified_flow_snapshot(
                snapshot,
                update={
                    "watch_state": WatchLevel.ENGAGED.value,
                    "phase": FlowPhase.ENGAGED.value,
                    "blocker": "",
                    "final_decision": "reply",
                }
            )
            self._apply_unified_flow_snapshot(
                snapshot,
                watch_reason="post_reply:engaged",
                phase_reason=reason or "完成回复后维持参与",
            )
            return
        snapshot = self._copy_unified_flow_snapshot(
            snapshot,
            update={
                "watch_state": WatchLevel.PEEK.value if not snapshot.is_resting else snapshot.watch_state,
                "phase": FlowPhase.STANDBY.value if not snapshot.is_resting else snapshot.phase,
                "blocker": reason or snapshot.blocker,
                "final_decision": "skip",
            }
        )
        self._apply_unified_flow_snapshot(
            snapshot,
            watch_reason="post_reply:skip",
            phase_reason=reason or "本轮未回复",
        )

    def _emit_flow_decision_summary(self, stage: str, final_decision: str) -> None:
        """统一输出心流决策摘要，便于排查谁在拦截。"""
        try:
            self._update_decision_trace(final_decision=final_decision)
            if stage == "final_decision" and not (self._last_decision_trace.get("winner_layer") or ""):
                if str(getattr(self, "_last_legacy_gate", "") or "") == "block" and final_decision != "reply":
                    self._mark_decision_winner("gateway_block")
                elif str(getattr(self, "_last_legacy_gate", "") or "") == "force_reply" and final_decision == "reply":
                    self._mark_decision_winner("gateway_force")
                elif final_decision == "reply":
                    self._mark_decision_winner("algo_reply")
                else:
                    self._mark_decision_winner("final_skip")
            snapshot = self._build_unified_flow_snapshot(final_decision)
            summary = {
                "stage": stage,
                "decision": final_decision,
                "voice_action": snapshot.voice_action,
                "voice_desire": snapshot.reply_desire,
                "voice_should_reply": snapshot.voice_should_reply,
                "voice_mood": snapshot.voice_mood,
                "voice_thinking": snapshot.voice_thinking,
                "boredom": snapshot.boredom,
                "env_fatigue": snapshot.environmental_fatigue,
                "loneliness": snapshot.loneliness,
                "social_desire": snapshot.social_desire,
                "watch": snapshot.watch_state,
                "phase": snapshot.phase,
                "resting": snapshot.is_resting,
                "rest_remaining": snapshot.rest_remaining_sec,
                "blocker": snapshot.blocker,
                "decision_trace": dict(getattr(self, "_last_decision_trace", None) or {}),
            }
            self._last_flow_decision_summary = summary
            _trace_brief = self._decision_trace_brief()
            logger.info(
                f"{self.log_prefix} [心流摘要/{stage}] 决策={final_decision} | "
                f"独白(action={summary['voice_action'] or '-'} desire={summary['voice_desire']} "
                f"reply={summary['voice_should_reply']} mood={summary['voice_mood'] or '-'}) | "
                f"情感(bored={summary['boredom']:.2f} fatigue={summary['env_fatigue']:.2f} "
                f"lonely={summary['loneliness']:.2f} social={summary['social_desire']:.2f}) | "
                f"观看={summary['watch']} 阶段={summary['phase']} 休息={summary['resting']}({summary['rest_remaining']:.0f}s) | "
                f"阻断={summary['blocker'] or '-'} | 想法={summary['voice_thinking'] or '-'}"
                + (f" | trace={_trace_brief}" if _trace_brief else "")
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 心流摘要输出失败: {exc}")

    def _apply_inner_voice_priority(
        self,
        *,
        should_act: bool,
        voice_conclusion,
        pinged_msg,
        legacy_gate: str,
        planner_decision,
        is_admin_forced: bool,
    ) -> bool:
        """让内心独白成为默认优先信号，外层只在强约束下覆盖。"""
        if voice_conclusion is None or not getattr(voice_conclusion, "is_valid", False):
            return should_act
        voice_desire = int(getattr(voice_conclusion, "reply_desire_level", 5) or 5)
        voice_should_reply = getattr(voice_conclusion, "should_reply", None)
        voice_action = str(getattr(voice_conclusion, "next_action", "") or "").strip().lower()
        strong_force = bool(is_admin_forced or pinged_msg is not None or legacy_gate == "force_reply")
        if voice_action in {"rest", "disengage", "lurk", "observe", "wait"}:
            self._last_flow_blocker = f"主体意愿={voice_action}"
            return False if not strong_force else should_act
        if voice_should_reply is True and voice_desire >= 6:
            self._last_flow_blocker = ""
            return True
        if voice_should_reply is False and voice_desire <= 4 and not strong_force:
            self._last_flow_blocker = f"主体意愿拒绝 desire={voice_desire}"
            return False
        if planner_decision is None and voice_desire >= 7 and not should_act:
            self._last_flow_blocker = ""
            return True
        return should_act

    def _handle_pending_gate(self, now: float) -> str:
        """等待期门控：委托 PendingOrchestrator 决策"""
        try:
            from src.chat.heart_flow.waiting_handler import (
                get_pending_orchestrator,
            )
            from src.common.data_models.heartflow_models import IdleDecision

            orchestrator = get_pending_orchestrator()
            decision = orchestrator.evaluate_next(self.stream_id)
            if decision == IdleDecision.EXPIRED or decision == IdleDecision.WITHDRAW:
                # 超时退出时触发后果链回写
                self._run_flow_side_effects(action_name="pending_timeout")
                _snap = self._build_unified_flow_snapshot(decision.value)
                _snap.watch_state = WatchLevel.ACTIVE_WATCH.value
                _snap.phase = FlowPhase.ENGAGED.value
                _snap.blocker = f"等待结束:{decision.value}"
                self._apply_unified_flow_snapshot(
                    _snap,
                    watch_reason=f"pending_end:{decision.value}",
                    phase_reason=f"等待结束: {decision.value}",
                )
                return "proceed"
            if decision == IdleDecision.CONTEMPLATE:
                return "proceed"
            if decision == IdleDecision.CHASE:
                return "proceed"
            return "skip"
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 等待门控异常，放行: {exc}")
            return "proceed"

    def _poll_eagerness(self) -> Tuple[float, str]:
        """读取回复意愿度，优先从世界快照推算"""
        snap = getattr(self, "_tick_world_snapshot", None)
        if snap is not None:
            res = snap.self_resources
            thinking_factor = res.thinking_ratio()
            chat_factor = res.chat_ratio()
            activity_factor = max(0.0, min(1.0, res.activity_level / 100.0))
            annoyance_penalty = min(res.channel_annoyance / 100.0, 0.5)
            raw = thinking_factor * 0.4 + chat_factor * 0.25 + activity_factor * 0.2 + 0.5 * 0.15 - annoyance_penalty
            eagerness = max(0.05, min(1.0, raw))
            reasons = []
            if res.chat_energy < res.chat_ceiling * 0.3:
                reasons.append(f"聊天值低({res.chat_energy:.0f})")
            if res.activity_level < 30:
                reasons.append(f"活跃度低({res.activity_level:.0f})")
            if res.thinking_energy < res.thinking_ceiling * 0.3:
                reasons.append(f"思考值低({res.thinking_energy:.0f})")
            if res.channel_annoyance > 30:
                reasons.append(f"烦躁({res.channel_annoyance:.0f})")
            return eagerness, (" | ".join(reasons) if reasons else "快照状态良好")
        try:
            _d6 = EnergyChainDimension.get_instance()
            _d6_state = _d6._ensure_channel(self.stream_id)
            _ratio = _d6_state.combined_ratio()
            if _ratio > 0.85:
                return 1.0, "精力充沛"
            elif _ratio > 0.60:
                return 0.9, "精力良好"
            elif _ratio > 0.35:
                return 0.7 - (0.60 - _ratio), "精力一般"
            elif _ratio > 0.15:
                return 0.4 - (0.35 - _ratio) * 1.5, "精力偏低"
            else:
                return 0.15, "精力严重不足"
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 能量满意度评估异常: {exc}")
            return 1.0, "能量模块不可用，默认满意愿"

    def _apply_glance_drain(self) -> None:
        """窥屏消耗"""
        try:
            _d6 = EnergyChainDimension.get_instance()

            class _GlanceEvt:
                channel_id = self.stream_id
                event_type = "glance"
                raw_extras = {}

            _d6.on_event(_GlanceEvt())
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _apply_think_drain(self, complexity: float = 1.0) -> None:
        """思考消耗，支持动态复杂度"""
        try:
            _d6 = EnergyChainDimension.get_instance()
            _clamped_complexity = max(0.1, min(3.0, complexity))

            class _ThinkEvt2:
                channel_id = self.stream_id
                event_type = "thinking_completed"
                complexity = _clamped_complexity
                raw_extras = {}

            _d6.on_event(_ThinkEvt2())
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _apply_reply_drain(self, messages: List) -> None:
        """回复消耗，按消息总长度计算"""
        try:
            _d6 = EnergyChainDimension.get_instance()
            total_len = sum(len(getattr(m, "processed_plain_text", "") or "") for m in messages)

            class _ReplyEvt2:
                channel_id = self.stream_id
                event_type = "reply_completed"
                reply_tokens = total_len
                raw_extras = {"is_admin": getattr(self, "_last_msg_was_admin", False)}

            _d6.on_event(_ReplyEvt2())
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _sync_runtime_resource_cache_from_d6(self) -> None:
        """把 D6 当前资源同步回兼容缓存，避免后续模块继续吃默认值。"""
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            if _ch is None:
                return
            _chat_val = float(getattr(_ch, "chat_pool", 100.0) or 100.0)
            _think_val = float(getattr(_ch, "thinking_value", 100.0) or 100.0)
            _chat_ratio = float(_ch.chat_ratio()) if hasattr(_ch, "chat_ratio") else max(0.0, min(1.0, _chat_val / 100.0))
            _thinking_ratio = (
                float(_ch.thinking_ratio())
                if hasattr(_ch, "thinking_ratio")
                else max(0.0, min(1.0, _think_val / 100.0))
            )
            _combined_ratio = (
                float(_ch.combined_ratio())
                if hasattr(_ch, "combined_ratio")
                else max(0.0, min(1.0, min(_chat_ratio, _thinking_ratio)))
            )
            _activity = float(getattr(_ch, "activity_level", 50.0) or 50.0)
            _social = float(getattr(_ch, "social_value", 0.0) or 0.0)
            _annoy = float(getattr(_ch, "annoyance_level", 0.0) or 0.0)
            _fatigue = max(0.0, float(getattr(_ch, "total_consumed_today", 0.0) or 0.0))
            _boredom = max(0.0, (1.0 - _combined_ratio) * 100.0)
            _boredom_drive = max(0.0, min(1.0, _boredom / 100.0))
            _loaf = max(0.0, min(100.0, (1.0 - _activity / 100.0) * 80.0))
            _loaf_sup = max(0.0, min(1.0, _loaf / 100.0))
            _social_will = max(0.05, 0.5 - min(0.35, max(0.0, (_loaf_sup - 0.40)) * 0.6)) if _loaf_sup > 0.40 else 0.5
            _reply_hint = (
                "简短"
                if _combined_ratio < 0.25 or _annoy > 60.0
                else ("中等" if _combined_ratio < 0.55 else "可以长一点")
            )
            _energy_gate_open = bool(_combined_ratio > 0.12 and _activity > 8.0)
            _is_perfunctory = bool(_combined_ratio < 0.22 or _annoy > 70.0)
            _recovery_rate = max(0.2, min(1.0, 1.0 - min(0.75, _fatigue / 100.0)))
            _sleep_debt = float((getattr(self, "_cached_night_summary", {}) or {}).get("sleep_debt", 0.0) or 0.0)
            _sleep_debt_modifier = 1.0 + min(0.8, _sleep_debt)
            _proactive_drive = max(
                0.05,
                min(0.95, 0.18 + _boredom_drive * 0.35 + max(0.0, _social_will - 0.5) * 0.3),
            )
            _mood_valence = max(0.0, min(1.0, 0.5 + (_social / 200.0) - (_annoy / 150.0)))
            _mood_arousal = max(0.0, min(1.0, 0.25 + (_activity / 140.0) + min(0.15, _boredom_drive * 0.2)))
            self._chat_energy = _chat_val
            self._thinking_energy = _think_val
            if isinstance(getattr(self, "_cached_metabolism_constraints", None), dict):
                self._cached_metabolism_constraints.update(
                    {
                        "boredom": round(_boredom, 2),
                        "boredom_level": round(_boredom_drive, 3),
                        "chat_fuel": round(_chat_val, 1),
                        "thinking_fuel": round(_think_val, 1),
                        "chat_value": round(_chat_val, 1),
                        "energy_ratio": round(_combined_ratio, 3),
                        "energy_suppression": round(max(0.0, min(1.0, 1.0 - _combined_ratio)), 3),
                        "loafing": round(_loaf, 2),
                        "loafing_level": round(_loaf, 2),
                        "loafing_suppression": round(_loaf_sup, 3),
                        "annoyance": round(_annoy, 1),
                        "annoyance_accumulated": round(_annoy, 1),
                        "fatigue_accumulator": round(_fatigue, 1),
                        "reply_length_hint": _reply_hint,
                        "energy_gate_open": _energy_gate_open,
                        "is_perfunctory": _is_perfunctory,
                        "recovery_rate": round(_recovery_rate, 3),
                        "sleep_debt_modifier": round(_sleep_debt_modifier, 3),
                        "proactive_drive": round(_proactive_drive, 3),
                    }
                )
            _ms = getattr(self, "_cached_metabolism_state", None)
            if _ms is not None:
                _ms.chat_fuel = _chat_val
                _ms.thinking_fuel = _think_val
                _ms.chat_energy_ratio = _chat_ratio
                _ms.energy_ratio = _combined_ratio
                _ms.thinking_ratio = _thinking_ratio
                _ms.activity_gauge = _activity
                _ms.social_gauge = _social
                _ms.mood_valence = _mood_valence
                _ms.mood_arousal = _mood_arousal
                _ms.boredom_level = _boredom
                _ms.loafing_level = _loaf
                _ms.channel_annoyance = _annoy
                _ms.fatigue_level = _fatigue
                _ms.fatigue_accumulator = _fatigue
                _ms.stress_accumulation = _fatigue
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 资源缓存同步失败: {exc}")

    def _capture_pre_reply_resource_snapshot(self) -> Optional[Dict[str, Any]]:
        """抓取真正回复前的D6关键资源，用于失败时精确回滚。"""
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            if _ch is None:
                return None
            return {
                "chat_pool": float(getattr(_ch, "chat_pool", 100.0) or 100.0),
                "thinking_value": float(getattr(_ch, "thinking_value", 100.0) or 100.0),
                "activity_level": float(getattr(_ch, "activity_level", 50.0) or 50.0),
                "social_value": float(getattr(_ch, "social_value", 0.0) or 0.0),
                "chain_count": int(getattr(_ch, "chain_count", 0) or 0),
                "total_consumed_today": float(getattr(_ch, "total_consumed_today", 0.0) or 0.0),
                "last_update": float(getattr(_ch, "last_update", time.time()) or time.time()),
            }
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 抓取回复前资源快照失败: {exc}")
            return None

    def _restore_pre_reply_resource_snapshot(
        self,
        snapshot: Optional[Dict[str, Any]],
        *,
        reason: str = "",
    ) -> None:
        """仅在未形成有效输出时回滚预扣资源，避免失败把系统越扣越累。"""
        if not snapshot:
            return
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            if _ch is None:
                return
            _ch.chat_pool = float(snapshot.get("chat_pool", _ch.chat_pool) or _ch.chat_pool)
            _ch.thinking_value = float(snapshot.get("thinking_value", _ch.thinking_value) or _ch.thinking_value)
            _ch.activity_level = float(snapshot.get("activity_level", _ch.activity_level) or _ch.activity_level)
            _ch.social_value = float(snapshot.get("social_value", _ch.social_value) or _ch.social_value)
            _ch.chain_count = int(snapshot.get("chain_count", _ch.chain_count) or _ch.chain_count)
            _ch.total_consumed_today = float(
                snapshot.get("total_consumed_today", _ch.total_consumed_today) or _ch.total_consumed_today
            )
            _ch.last_update = float(snapshot.get("last_update", time.time()) or time.time())
            self._sync_runtime_resource_cache_from_d6()
            logger.info(f"{self.log_prefix} ♻️ 预扣资源已回滚: {(reason or '未形成有效回复')[:80]}")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 回复前资源回滚失败: {exc}")

    async def _apply_reply_drain_dynamic(self, messages: List, relation_result: Dict) -> None:
        """回复消耗 - 基于水位比例和多因素链的动态算法

        算法流程：
        1. 水位基底：聊天比 × 3.0 → [0.6, 3.5]；思考比 × 2.5 → [0.5, 2.5]
        2. 连续发言递增：每多说一次 +0.3
        3. 频率因子：<60s=1.6, <300s=1.3, <600s=1.1, else=1.0
        4. 话痨因子：1.0 + 话痨惩罚 × 0.4
        5. 兴趣折扣：max(0.6, 1.0 - 兴趣等级 × 0.4)
        6. 聊天值总计 clamp [0.8, 8.0]；思考值总计 clamp [0.5, 5.0]
        """
        try:
            _d6 = EnergyChainDimension.get_instance()
            _d6_state = _d6._ensure_channel(self.stream_id)
            chat_ratio = _d6_state.chat_ratio()
            thinking_ratio = _d6_state.thinking_ratio()
            base_chat_cost = max(0.6, min(3.5, chat_ratio * 3.0))
            base_brain_cost = max(0.5, min(2.5, thinking_ratio * 2.5))
            consecutive_add = getattr(self, "_consecutive_speaks", 0) * 0.3
            time_since_last = time.time() - getattr(self, "_last_speak_time", 0)
            if time_since_last < 60:
                frequency_factor = 1.6
            elif time_since_last < 300:
                frequency_factor = 1.3
            elif time_since_last < 600:
                frequency_factor = 1.1
            else:
                frequency_factor = 1.0
            chatterbox_penalty = getattr(self, "_chatterbox_penalty", 0.0)
            chatterbox_factor = 1.0 + chatterbox_penalty * 0.4
            interest_level = getattr(self, "_last_interest_level", 0.5)
            interest_discount = max(0.6, 1.0 - interest_level * 0.4)
            total_chat_cost = (
                (base_chat_cost + consecutive_add) * frequency_factor * chatterbox_factor * interest_discount
            )
            total_brain_cost = base_brain_cost * frequency_factor * interest_discount
            total_chat_cost = max(0.8, min(8.0, total_chat_cost))
            total_brain_cost = max(0.5, min(5.0, total_brain_cost))
            _d6_state.chat_pool = max(0.0, _d6_state.chat_pool - total_chat_cost)
            _d6_state.thinking_value = max(0.0, _d6_state.thinking_value - total_brain_cost)
            _d6_state.chain_count += 1
            _d6_state.total_consumed_today += total_chat_cost + total_brain_cost
            self._consecutive_speaks = getattr(self, "_consecutive_speaks", 0) + 1
            self._last_speak_time = time.time()
            if self._consecutive_speaks >= 4:
                self._chatterbox_penalty = getattr(self, "_chatterbox_penalty", 0.0) + 1.0
                logger.warning(
                    f"{self.log_prefix} 话痨警告！连续发言{self._consecutive_speaks}次，"
                    f"惩罚值={self._chatterbox_penalty:.1f}"
                )
            self._sync_runtime_resource_cache_from_d6()
            logger.info(
                f"{self.log_prefix} ⚡ "
                f"本轮消耗 chat={total_chat_cost:.2f} think={total_brain_cost:.2f} | "
                f"连续加成={consecutive_add:.1f} 频率因子={frequency_factor:.2f} "
                f"话痨因子={chatterbox_factor:.2f} 兴趣折扣={interest_discount:.2f} "
                f"(水位: chat={chat_ratio:.2f} thinking={thinking_ratio:.2f})"
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 回复消耗计算失败: {exc}")

    async def _deduct_shared_resources(self, messages: List) -> None:
        """实时扣除共享资源 - 动作执行时立即消耗（从D6读取并扣除）"""
        try:
            _d6 = EnergyChainDimension.get_instance()
            _d6s = _d6._ensure_channel(self.stream_id)
            chat_deduct = max(0.5, min(3.0, 1.2 + (1.0 - _d6s.chat_ratio()) * 2.0))
            activity_deduct = max(0.1, min(1.0, 0.8 * (1.0 - _d6s.activity_level / 100.0)))
            _d6s.chat_pool = max(0.0, _d6s.chat_pool - chat_deduct)
            _d6s.activity_level = max(5.0, _d6s.activity_level - activity_deduct)
            _d6s.chain_count += 1
            content = ""
            if messages:
                for msg in reversed(messages):
                    if self._is_human_message_obj(msg):
                        content = getattr(msg, "processed_plain_text", "") or getattr(msg, "content", "")
                        break
            polarity = self._estimate_text_polarity(content) if content else 0.0
            if polarity > 0.1:
                _d6s.social_value = min(100.0, _d6s.social_value + 0.05)
                social_delta = 0.05
            elif polarity < -0.1:
                _d6s.social_value = max(-50.0, _d6s.social_value - 0.08)
                social_delta = -0.08
            else:
                social_delta = 0.0
            self._sync_runtime_resource_cache_from_d6()
            logger.info(
                f"{self.log_prefix} ⚡ "
                f"聊天值-{chat_deduct:.1f} "
                f"活跃度-{activity_deduct:.1f} "
                f"社交值变化={social_delta:+.1f}"
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 共享资源扣除失败: {exc}")

    async def _apply_input_state_updates(self, messages: List) -> None:
        """用户消息到达时即时激活聊天值/活跃度/社交值（D6 on_event）"""
        try:
            from src.person_info.person_info import get_unified_profile_hub

            human_messages = [
                msg for msg in messages if getattr(msg, "user_id", "") and not self._is_bot_message_obj(msg)
            ]
            if not human_messages:
                return

            latest_human = human_messages[-1]
            latest_text = (
                getattr(latest_human, "processed_plain_text", "") or getattr(latest_human, "content", "") or ""
            )
            normalized_latest = self._normalize_repeat_text(latest_text)
            latest_user_id = str(getattr(latest_human, "user_id", "") or "")
            _raw_nick = str(
                getattr(latest_human, "user_nickname", "") or getattr(latest_human, "nickname", "") or ""
            ).strip()
            latest_nickname = _raw_nick or latest_user_id
            if _raw_nick and len(latest_user_id) > 16:
                latest_user_id = f"{self.stream_id}_{_raw_nick[:12]}"
            repeat_count = self._count_recent_user_repeats(normalized_latest, latest_user_id)

            if latest_user_id:
                hub = get_unified_profile_hub()
                hub.bind_person_identity(
                    person_id=latest_user_id,
                    platform=("webui" if latest_user_id.startswith("webui_") else "unknown"),
                    origin_user_id=latest_user_id,
                    nickname=latest_nickname,
                )
                hub.record_interaction(latest_user_id, self.stream_id)

            _d6 = EnergyChainDimension.get_instance()
            _has_mention = any(getattr(m, "is_mentioned", False) or getattr(m, "is_at", False) for m in messages)

            class _InputEvt:
                channel_id = self.stream_id
                event_type = "user_message"
                user_id = latest_user_id
                message_length = len(latest_text)
                raw_extras = {
                    "is_repeat": repeat_count >= 2,
                    "is_mentioned": _has_mention,
                }

            _d6.on_event(_InputEvt())
            _d6s = _d6._ensure_channel(self.stream_id)
            _d6_annoyance = _d6s.annoyance_level
            logger.info(
                f"{self.log_prefix} 状态激活 聊天值={_d6s.chat_pool:.1f} "
                f"活跃度={_d6s.activity_level:.1f} 社交值={_d6s.social_value:.1f} "
                f"烦躁={_d6_annoyance:.1f} "
                f"重复={'是' if repeat_count >= 2 else '否'}"
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 输入状态激活失败: {exc}")

    def _remember_recent_user_input(self, normalized_text: str, user_id: str) -> None:
        if not normalized_text:
            return
        now = time.time()
        self._recent_user_inputs.append(
            {
                "text": normalized_text,
                "user_id": user_id,
                "ts": now,
            }
        )
        cutoff = now - 300.0
        self._recent_user_inputs = [
            item
            for item in self._recent_user_inputs[-20:]
            if float(item.get("ts", 0.0) or 0.0) >= cutoff and str(item.get("text", "") or "")
        ]

    def _count_recent_user_repeats(self, normalized_text: str, user_id: str) -> int:
        if not normalized_text:
            return 0
        now = time.time()
        cutoff = now - 300.0
        count = 0
        for item in self._recent_user_inputs:
            if float(item.get("ts", 0.0) or 0.0) < cutoff:
                continue
            if str(item.get("text", "") or "") != normalized_text:
                continue
            item_user = str(item.get("user_id", "") or "")
            if user_id and item_user and item_user != user_id:
                continue
            count += 1
        return count + 1

    @staticmethod
    def _normalize_repeat_text(text: str) -> str:
        """压缩文本用于检测短时间重复输入。"""

        if not text:
            return ""
        return re.sub(r"[\s\W_]+", "", str(text), flags=re.UNICODE).lower()

    def _classify_behavior_signal(
        self,
        messages: List,
        repetition_signal: Optional[Dict[str, Any]] = None,
        harassment_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """统一行为分类信号：委托给共享模块，避免多条主链继续分叉。"""
        latest_user = self._get_latest_human_message(messages)
        if latest_user is None:
            return {
                "category": "neutral",
                "behavior_type": "casual_chat",
                "intent": "chat",
                "severity": 0.25,
                "is_new_user": False,
                "reason": "默认中性互动",
            }

        user_id = str(getattr(latest_user, "user_id", "") or "")
        text = self._extract_message_content(latest_user).strip()
        interaction_count = 0

        try:
            from src.person_info.person_info import get_unified_profile_hub
            from src.modules.social_value.social_affect_fuser import (
                get_social_affect_fuser,
            )
            from src.chat.behavior.behavior_signal import (
                classify_behavior_signal,
            )

            dossier = get_social_affect_fuser().get_dossier(user_id, self.stream_id)
            interaction_count = len(dossier.recent_values) if dossier else 0
            try:
                unified_profile = get_unified_profile_hub().get_profile(user_id)
                if unified_profile is not None:
                    interaction_count = max(
                        interaction_count,
                        int(getattr(unified_profile, "interaction_count", 0) or 0),
                    )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            return classify_behavior_signal(
                text=text,
                user_id=user_id,
                interaction_count=interaction_count,
                repetition_signal=repetition_signal,
                harassment_signal=harassment_signal,
            )
        except Exception:
            from src.chat.behavior.behavior_signal import (
                classify_behavior_signal,
            )

            return classify_behavior_signal(
                text=text,
                user_id=user_id,
                interaction_count=interaction_count,
                repetition_signal=repetition_signal,
                harassment_signal=harassment_signal,
            )

    async def _update_social_metrics(
        self,
        messages: List,
        relation_result: Dict,
        behavior_signal: Optional[Dict[str, Any]] = None,
    ) -> None:
        """更新社交值 - 回复后通过SettlementEngine(9步富管线)更新"""
        try:
            from src.core.huoli_core import get_core

            user_id = ""
            content = ""
            if messages:
                for msg in reversed(messages):
                    uid = getattr(msg, "user_id", "") or ""
                    if uid and not self._is_bot_message_obj(msg):
                        user_id = uid
                        content = getattr(msg, "processed_plain_text", "") or getattr(msg, "content", "")
                        break
            if not user_id:
                logger.debug(f"{self.log_prefix} 无用户ID，跳过社交值更新")
                return
            _nick = ""
            if messages:
                for msg in reversed(messages):
                    _n = getattr(msg, "user_nickname", "") or getattr(msg, "nickname", "") or ""
                    if _n:
                        _nick = _n.strip()
                        break
            if _nick and len(user_id) > 16:
                user_id = f"{self.stream_id}_{_nick[:12]}"
            self._last_user_id = user_id
            context = {
                "channel_id": self.stream_id,
                "interaction_type": "bot_reply",
                "current_social": relation_result.get("social_value", 0.0),
                "current_trust": relation_result.get("trust_value", 0.0),
                "trauma_score": float(relation_result.get("trauma_score", 0.0) or 0.0),
                "psychological_pressure": float(relation_result.get("psychological_pressure", 0.0) or 0.0),
            }
            if behavior_signal:
                _behavior_payload = {
                    "behavior_type": behavior_signal.get("behavior_type", "casual_chat"),
                    "intent": behavior_signal.get("intent", "chat"),
                    "severity": float(behavior_signal.get("severity", 0.25) or 0.25),
                }
                context.update(
                    {
                        "behavior_type": _behavior_payload["behavior_type"],
                        "intent": _behavior_payload["intent"],
                        "severity": _behavior_payload["severity"],
                        "is_new_user": bool(behavior_signal.get("is_new_user", False)),
                        "behavior_category": behavior_signal.get("category", "neutral"),
                        "behavior_signal": _behavior_payload,
                    }
                )
            # ── 首选：使用SettlementEngine（9步富管线） ──
            try:
                from src.modules.social_value.settlement_engine import (
                    get_settlement_engine,
                )

                se = get_settlement_engine()
                report = await se.settle_event(user_id, self.stream_id, content, context)
                if report:
                    relation_result["social_value"] = float(report.settled_score or 0.0)
                    self._last_relation_snapshot["social_value"] = relation_result["social_value"]
                    logger.info(
                        f"{self.log_prefix} 🤝 结算完成 "
                        f"对象={user_id[:8]} "
                        f"旧值={report.prior_score:.1f}→新值={report.settled_score:.1f} "
                        f"(Δ={report.final_delta:+.3f}) "
                        f"原始={report.raw_delta:+.3f} "
                        f"饱和={'是' if abs(report.delta_after_saturation) < abs(report.raw_delta) - 0.001 else '否'} "
                        f"反转={'是' if report.reversal_triggered else '否'} "
                        f"上限={'是' if report.cap_triggered else '否'}"
                    )
                    return
            except Exception as se_exc:
                logger.debug(f"{self.log_prefix} SettlementEngine结算失败，回退到SocialValueCore: {se_exc}")
            # ── 回退：使用SocialValueCore（5步基础管线） ──
            svc = get_core().social_value
            if svc is None:
                logger.warning(f"{self.log_prefix} 社交值核心未初始化，跳过更新")
                return
            if hasattr(svc, "update"):
                result = await svc.update(user_id, self.stream_id, content, context)
                if result:
                    relation_result["social_value"] = float(getattr(result, "new_value", relation_result.get("social_value", 0.0)) or 0.0)
                    self._last_relation_snapshot["social_value"] = relation_result["social_value"]
                    logger.info(
                        f"{self.log_prefix} 🤝 "
                        f"对象={user_id[:8]} 行为={result.behavior_type} "
                        f"意图={result.intent} 社交值变化={result.delta:.2f} "
                        f"新值={result.new_value:.1f}"
                    )
            elif hasattr(svc, "record_outcome"):
                try:
                    _behavior = context.get("behavior_type", "reply") if isinstance(context, dict) else "reply"
                    _intent = context.get("intent", "neutral") if isinstance(context, dict) else "neutral"
                    _severity = float(context.get("severity", 0.25) or 0.25) if isinstance(context, dict) else 0.25
                    _social_delta_map = {
                        "friendly": 1.5,
                        "warm": 1.2,
                        "playful": 1.0,
                        "casual_chat": 0.5,
                        "reply": 0.3,
                        "sarcastic": -0.3,
                        "cold": -0.8,
                        "defensive": -0.5,
                        "harassing": -2.0,
                        "reject": -1.5,
                    }
                    _base_delta = _social_delta_map.get(_behavior, 0.2)
                    if _intent in ("positive", "satisfied"):
                        _base_delta = max(_base_delta, 0.8)
                    elif _intent in ("negative", "hurt", "disappointed"):
                        _base_delta = min(_base_delta, -0.5)
                    _final_delta = _base_delta * (0.5 + _severity)
                    svc.record_outcome(user_id, delta=_final_delta, intensity=_severity)
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
        except Exception as exc:
            logger.warning(f"{self.log_prefix} 社交值更新失败: {exc}")

    async def _log_final_status_dynamic(self) -> None:
        """动态最终状态输出 - 包含所有系统状态和共享资源（节流版）"""
        now = time.time()
        if now - self._last_status_log_ts < self._status_log_interval:
            return
        self._last_status_log_ts = now
        try:
            relation_view = self._resolve_relation_view()
            freq_ctrl = frequency_control_manager.get_or_create_frequency_control(self.stream_id)
            freq_adjust = freq_ctrl.get_talk_frequency_adjust()
            user_id = ""
            if hasattr(self, "_last_user_id"):
                user_id = self._last_user_id
            trauma_score = 0.0
            psychological_pressure = 0.0
            training_stage = 0
            training_progress = 0.0
            inner_chaos = 0.0
            surface_mask = 10.0
            stress_accumulation = 0.0
            submission_level = 0.0

            _d6 = EnergyChainDimension.get_instance()
            _d6s = _d6._ensure_channel(self.stream_id)
            chat_value = _d6s.chat_pool
            activity_level = _d6s.activity_level
            shared_social = _d6s.social_value
            social_value = float(relation_view.get("social_value", shared_social) or 0.0)
            trust_value = float(relation_view.get("trust_value", 0.0) or 0.0)
            annoyance_value = float(relation_view.get("annoyance_value", 0.0) or 0.0)
            trauma_score = float(relation_view.get("trauma_score", 0.0) or 0.0)
            psychological_pressure = float(relation_view.get("psychological_pressure", 0.0) or 0.0)
            if user_id:
                try:
                    _d6s2 = _d6._ensure_channel(self.stream_id)
                    social_value = max(social_value, _d6s2.social_value)
                    trust_value = float(relation_view.get("trust_value", 0.0) or 0.0)
                    annoyance_value = max(annoyance_value, _d6s2.annoyance_level)
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker,
                    )

                    tracker = get_emotion_tracker(self.stream_id)
                    state = tracker.get_user_state(user_id, create_if_missing=False)
                    if state:
                        trauma_score = getattr(state, "trauma_score", 0.0)
                        psychological_pressure = getattr(state, "psychological_pressure", 0.0)
                        training_stage = getattr(state, "training_stage", 0)
                        training_progress = getattr(state, "training_progress", 0.0)
                        inner_chaos = getattr(state, "inner_chaos", 0.0)
                        surface_mask = getattr(state, "surface_mask", 10.0)
                        submission_level = getattr(state, "submission_level", 0.0)
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
                try:
                    from src.modules.trauma.trauma_system import (
                        get_trauma_system,
                    )

                    trauma_sys = get_trauma_system()
                    trauma_state = trauma_sys.get_state()
                    inner_chaos = trauma_state.inner_chaos_level
                    surface_mask = trauma_state.surface_mask_strength
                    stress_accumulation = trauma_state.stress_accumulation
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
            base_parts = [
                f"聊天当前={chat_value:.1f}",
                f"活跃当前={activity_level:.1f}",
                f"社交当前={shared_social:.1f}",
                f"信任当前={trust_value:.1f}",
            ]
            if social_value != 0:
                base_parts.append(f"关系社交当前={social_value:.1f}")
            affection = float(relation_view.get("affection", 0.0) or 0.0)
            if affection != 0:
                base_parts.append(f"好感当前={affection:.1f}")
            conditional_parts = []
            if trauma_score > 0:
                conditional_parts.append(f"创伤={trauma_score:.1f}")
            if psychological_pressure > 0:
                conditional_parts.append(f"心理压力={psychological_pressure:.1f}")
            if inner_chaos > 0:
                conditional_parts.append(f"内心混乱={inner_chaos:.1f}")
            if surface_mask > 0:
                conditional_parts.append(f"伪装强度={surface_mask:.1f}")
            if stress_accumulation > 0:
                conditional_parts.append(f"压力累积={stress_accumulation:.1f}")
            if training_stage > 0:
                conditional_parts.append(f"调教阶段={training_stage}")
            if training_progress > 0:
                conditional_parts.append(f"调教进度={training_progress:.1f}%")
            if submission_level > 0:
                conditional_parts.append(f"顺从度={submission_level:.1f}")
            base_parts.append(f"频率={freq_adjust:.2f}")
            all_parts = base_parts + conditional_parts
            logger.info(f"{self.log_prefix} 📋 " + " ".join(all_parts))
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 最终状态输出失败: {exc}")

    def _emit_vitals_gauge(self, eagerness_val: float, eagerness_reason: str, phase_label: str) -> None:
        """第一层：独立系统体征面板 - 进入决策前的全局快照

        输出思考值、聊天值、活跃值、社交值、回复意愿、当前阶段。
        每次进入完整管线时必定输出，不依赖最终是否回复。
        """
        try:
            _d6 = EnergyChainDimension.get_instance()
            _d6s = _d6._ensure_channel(self.stream_id)
            thinking_pct = _d6s.thinking_ratio() * 100.0
            chat_pct = _d6s.chat_pool
            activity_pct = _d6s.activity_level
            social_reading = _d6s.social_value

            logger.info(
                f"{self.log_prefix} 状态面板 "
                f"思考当前={thinking_pct:.0f}% 聊天当前={chat_pct:.1f} "
                f"活跃当前={activity_pct:.1f} 社交当前={social_reading:.1f} | "
                f"意愿={eagerness_val:.2f} 阶段={phase_label}"
            )
            # 面板适配器导出（附加核心模块维度）
            try:
                from src.core.panel_state_adapter import get_panel_adapter

                _adapter = get_panel_adapter()
                if self._tick_world_snapshot:
                    _panel_module_context = {}
                    if self._cached_watch_level:
                        _watch_label = (
                            self._cached_watch_level.label()
                            if hasattr(self._cached_watch_level, "label")
                            else str(self._cached_watch_level)
                        )
                        _panel_module_context["watch_info"] = {
                            "level": (
                                self._cached_watch_level.value
                                if hasattr(self._cached_watch_level, "value")
                                else str(self._cached_watch_level)
                            ),
                            "label": _watch_label,
                        }
                    if self._cached_metabolism_state:
                        _panel_module_context["metabolism"] = {
                            "loafing": float(
                                getattr(
                                    self._cached_metabolism_state,
                                    "loafing_level",
                                    0.0,
                                )
                                or 0.0
                            ),
                        }
                    # 代谢行为约束注入
                    if self._cached_metabolism_constraints:
                        _panel_module_context.setdefault("metabolism", {}).update(self._cached_metabolism_constraints)
                    # 夜间行为摘要注入
                    if self._cached_night_summary:
                        _panel_module_context["night"] = dict(self._cached_night_summary)
                    if self._cached_scene_snapshot and hasattr(self._cached_scene_snapshot, "to_dict"):
                        _scene_dict = self._cached_scene_snapshot.to_dict()
                        _scene_dict["multi_thread"] = bool(getattr(self._cached_scene_snapshot, "thread_count", 0) > 1)
                        _panel_module_context["scene_state"] = _scene_dict
                    if self._cached_pattern_evidence:
                        _dominant_pattern = self._cached_pattern_evidence[0]
                        if hasattr(_dominant_pattern, "to_dict"):
                            _pattern_dict = _dominant_pattern.to_dict()
                            _pattern_dict["detected"] = True
                            _panel_module_context["pattern"] = _pattern_dict
                    if self._cached_multimodal_summary:
                        _panel_module_context["budgeter"] = dict(self._cached_multimodal_summary)
                    if self._cached_learning_summary:
                        _panel_module_context["learning"] = dict(self._cached_learning_summary)
                    if self._last_relation_snapshot:
                        _impression_ctx = dict(self._last_relation_snapshot)
                        if self._last_user_id:
                            try:
                                from src.core.active_participant_roster import (
                                    get_participant_roster,
                                )

                                _entry = get_participant_roster(self.stream_id).get_user(self._last_user_id)
                                _impression_ctx["is_high_activity"] = bool(
                                    _entry
                                    and getattr(
                                        getattr(_entry, "tier", None),
                                        "value",
                                        "",
                                    )
                                    in ("hot", "warm")
                                )
                            except Exception as _exc:
                                logger.debug(f"非关键异常: {_exc}")
                        _panel_module_context["impression"] = _impression_ctx
                    if self._cached_understanding_results:
                        _primary_understanding = self._cached_understanding_results[0]
                        _panel_module_context["understanding"] = {
                            "comprehension_level": getattr(
                                getattr(_primary_understanding, "level", None),
                                "value",
                                "",
                            ),
                            "deep_analysis": bool(
                                getattr(
                                    _primary_understanding,
                                    "upgrade_analysis",
                                    False,
                                )
                            ),
                        }
                    if self._cached_self_references:
                        # 提取最强引用类型和强度
                        _best_ref_type = "none"
                        _best_ref_strength = 0.0
                        _any_relevant = False
                        for _ref in self._cached_self_references:
                            if bool(getattr(_ref, "is_relevant", lambda: False)()):
                                _any_relevant = True
                            _rs = float(getattr(_ref, "strength", 0.0) or 0.0)
                            if _rs > _best_ref_strength:
                                _best_ref_strength = _rs
                                _rt = getattr(_ref, "ref_type", None)
                                _best_ref_type = (_rt.value if hasattr(_rt, "value") else str(_rt)) if _rt else "none"
                        _panel_module_context["self_ref"] = {
                            "is_discussing_bot": _any_relevant,
                            "strength": _best_ref_strength,
                            "ref_type": _best_ref_type,
                        }
                    # 主观存在判定注入
                    if self._cached_presence_verdict:
                        _panel_module_context["presence"] = dict(self._cached_presence_verdict)
                    # 消息路由分类汇总注入
                    if self._cached_route_summary:
                        _panel_module_context["route_summary"] = dict(self._cached_route_summary)
                    # 技能生命周期摘要注入
                    if self._cached_skill_review:
                        _panel_module_context["skill_review"] = dict(self._cached_skill_review)
                    # 自适应管线摘要注入
                    if self._cached_pipeline_summary:
                        _panel_module_context["adaptive_pipeline"] = dict(self._cached_pipeline_summary)
                    # 被误解信号注入
                    if self._cached_misunderstanding_signal:
                        _sig = self._cached_misunderstanding_signal
                        _panel_module_context["misunderstanding"] = {
                            "risk": _sig.risk_score,
                            "type": _sig.signal_type,
                            "cues": _sig.detected_cues[:3],
                        }
                    # 用户负面情绪聚合值注入
                    if self._cached_user_negative_emotion > 5:
                        _panel_module_context["user_negative_emotion"] = {
                            "aggregate": self._cached_user_negative_emotion,
                        }
                    _panel_snap = _adapter.assemble(
                        self._tick_world_snapshot,
                        module_context=_panel_module_context,
                    )
                    _vitals = _panel_snap.vitals
                    _extra_parts = []
                    if self._cached_night_phase:
                        _np_val = (
                            self._cached_night_phase.label()
                            if hasattr(self._cached_night_phase, "label")
                            else str(self._cached_night_phase)
                        )
                        _extra_parts.append(f"节律={_np_val}")
                    if self._cached_watch_level:
                        _wl_val = (
                            self._cached_watch_level.label()
                            if hasattr(self._cached_watch_level, "label")
                            else str(self._cached_watch_level)
                        )
                        _extra_parts.append(f"关注={_wl_val}")
                    if self._cached_presence_state:
                        _sw = float(
                            getattr(
                                self._cached_presence_state,
                                "social_willingness",
                                0.5,
                            )
                            or 0.5
                        )
                        _extra_parts.append(f"社交意愿={_sw:.2f}")
                    if self._cached_presence_verdict:
                        _pv = self._cached_presence_verdict.get("participation_verdict", "")
                        _ml = self._cached_presence_verdict.get("mood_label", "")
                        if _pv:
                            _extra_parts.append(f"参与={_pv}")
                        if _ml:
                            _extra_parts.append(f"情绪={_ml}")
                    if self._cached_metabolism_state:
                        _cf = float(
                            getattr(
                                self._cached_metabolism_state,
                                "chat_fuel",
                                100.0,
                            )
                            or 100.0
                        )
                        _extra_parts.append(f"聊天燃料={_cf:.0f}")
                    # 群体模式与场景适合度
                    if _panel_snap.scene.active_pattern:
                        _extra_parts.append(
                            f"模式={_panel_snap.scene.active_pattern}({_panel_snap.scene.pattern_confidence:.2f})"
                        )
                    if _panel_snap.scene.atmosphere:
                        _atmo_display = _panel_snap.scene.atmosphere_label or _panel_snap.scene.atmosphere
                        _extra_parts.append(f"氛围={_atmo_display}")
                    if _panel_snap.scene.has_multi_thread:
                        _extra_parts.append(f"多线程={_panel_snap.scene.thread_count}")
                    # 代谢约束
                    if self._cached_metabolism_constraints:
                        _rh = self._cached_metabolism_constraints.get("reply_length_hint", "")
                        if _rh and _rh != "normal":
                            _extra_parts.append(f"回复约束={_rh}")
                    # 夜间状态
                    if self._cached_night_summary:
                        _ns_phase = self._cached_night_summary.get("phase_label", "")
                        _ns_debt = self._cached_night_summary.get("sleep_debt", 0.0)
                        if _ns_phase:
                            _extra_parts.append(f"夜间={_ns_phase}")
                        if _ns_debt > 0.2:
                            _extra_parts.append(f"睡眠债={_ns_debt:.2f}")
                    # 技能生命周期摘要
                    if self._cached_skill_review:
                        _sr_promoted = self._cached_skill_review.get("promoted", 0)
                        _sr_declined = self._cached_skill_review.get("declined", 0)
                        _sr_retired = self._cached_skill_review.get("retired", 0)
                        if _sr_promoted or _sr_declined or _sr_retired:
                            _extra_parts.append(f"技能变动=↑{_sr_promoted}↓{_sr_declined}✕{_sr_retired}")
                    # 自适应管线状态
                    if self._cached_pipeline_summary:
                        _pipe_evts = self._cached_pipeline_summary.get("event_pool_size", 0)
                        _pipe_fails = self._cached_pipeline_summary.get("failure_pool_size", 0)
                        _pipe_cands = self._cached_pipeline_summary.get("total_candidates", 0)
                        if _pipe_evts or _pipe_fails or _pipe_cands:
                            _extra_parts.append(f"管线事件={_pipe_evts} 失败={_pipe_fails} 候选={_pipe_cands}")
                    # 被误解信号日志
                    if self._cached_misunderstanding_signal and self._cached_misunderstanding_signal.risk_score > 0.1:
                        _mis_risk = self._cached_misunderstanding_signal.risk_score
                        _mis_type = self._cached_misunderstanding_signal.signal_type
                        _extra_parts.append(f"误解风险={_mis_risk:.2f}({_mis_type})")
                    # 用户负面情绪日志
                    if self._cached_user_negative_emotion > 10:
                        _extra_parts.append(f"负面情绪={self._cached_user_negative_emotion:.1f}")
                    # 用户印象标签日志
                    if self._cached_user_impression_tags:
                        _extra_parts.append(f"印象标签={','.join(self._cached_user_impression_tags[:5])}")
                    if _extra_parts:
                        logger.info(f"{self.log_prefix} 扩展面板 " + " ".join(_extra_parts))
                    # 统一状态仪表盘行（7Bar + ActionVerdict）
                    try:
                        _dash_line = self._get_dashboard_status_line()
                        if _dash_line:
                            logger.info(f"{self.log_prefix} 📊 状态栏 {_dash_line}")
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
            except Exception as _panel_exc:
                logger.debug(f"{self.log_prefix} 面板适配器异常: {_panel_exc}")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 体征面板输出异常: {exc}")

    def _build_state_dashboard(self, *, force: bool = False) -> Optional[Dict[str, Any]]:
        """统一状态仪表盘构建 —— 从40+个分散缓存聚合为7个标准化Bar

        这是状态系统的唯一入口，替代所有直接读取_cached_变量的分散逻辑。
        返回值可直接用于日志输出、决策判断、面板展示。

        聚合数据源映射：
          VitalityBar  ← metabolism_state + metabolism_constraints
          SocialBar    ← presence_state + participant_summary + relation_snapshot
          MoodBar      ← metabolism_constraints(boredom/loafing) + emotion_state
          AttentionBar ← attention_snapshot(GAP-W)
          NightBar     ← night_phase + night_summary
          SafetyBar    ← safety_assessment(GAP-R)
          MemoryBar    ← memory_governance_snap(GAP-Q)
        """
        _now = time.time()
        if not force and self._cached_dashboard_snapshot is not None:
            if (_now - self._last_dashboard_build_ts) < self._dashboard_ttl_sec:
                return self._cached_dashboard_snapshot
        try:
            from src.core.state_dashboard import get_state_dashboard

            if not self._dashboard_initialized:
                self._dashboard_initialized = True
            _engine = get_state_dashboard(self.stream_id)
            _raw: Dict[str, Any] = {}
            def _norm_dashboard_ratio(value: Any) -> float:
                try:
                    _num = float(value or 0.0)
                except Exception:
                    return 0.0
                if _num <= 1.0:
                    return max(0.0, min(1.0, _num))
                return max(0.0, min(1.0, _num / 100.0))
            # ── VitalityBar 数据源：代谢引擎 ──
            _ms = getattr(self, "_cached_metabolism_state", None)
            if _ms:
                _raw["energy_ratio"] = float(
                    getattr(_ms, "energy_ratio", getattr(_ms, "chat_energy_ratio", 1.0)) or 1.0
                )
                _raw["fatigue"] = _norm_dashboard_ratio(getattr(_ms, "fatigue_level", 0.0))
                _mc = getattr(self, "_cached_metabolism_constraints", None)
                if _mc:
                    _raw["recovery_rate"] = float(_mc.get("recovery_rate", 1.0) or 1.0)
                    _raw["debt_modifier"] = float(_mc.get("sleep_debt_modifier", 1.0) or 1.0)
                    _raw["boredom"] = _norm_dashboard_ratio(_mc.get("boredom_level", _mc.get("boredom", 0.0)))
                    _raw["loafing"] = _norm_dashboard_ratio(
                        _mc.get("loafing_level", _mc.get("loafing_suppression", 0.0))
                    )
                    _raw["annoyance"] = _norm_dashboard_ratio(_mc.get("annoyance_accumulated", 0.0))
                    _raw["proactive_drive"] = float(_mc.get("proactive_drive", 0.3) or 0.3)
            else:
                _raw["energy_ratio"] = 1.0
                _raw["recovery_rate"] = 1.0
                _raw["debt_modifier"] = 1.0
            # ── SocialBar 数据源：存在感引擎 + 参与者名册 + social_value_core真实值 ──
            _ps = getattr(self, "_cached_presence_state", None)
            if _ps:
                _raw["social_willingness"] = float(getattr(_ps, "social_willingness", 0.5) or 0.5)
                _raw["openness"] = float(getattr(_ps, "openness", 0.5) or 0.5)
                _raw["interrupt_tolerance"] = float(getattr(_ps, "interrupt_tolerance", 0.5) or 0.5)
                _raw["avoidance"] = float(getattr(_ps, "avoidance_tendency", 0.0) or 0.0)
                _raw["loneliness"] = float(getattr(_ps, "loneliness", 0.0) or 0.0)
                _raw["social_hunger"] = float(getattr(_ps, "social_hunger", 0.5) or 0.5)
            # 从social_value_core读取真实社交值（替代默认全零）
            _focus_uid = getattr(self, "_last_user_id", "") or ""
            if _focus_uid:
                try:
                    _relation_source = getattr(self, "_cached_relation_result", None)
                    if not isinstance(_relation_source, dict):
                        _snap = getattr(self, "_tick_world_snapshot", None)
                        _target_uid = ""
                        if _snap is not None:
                            _target_uid = str(getattr(getattr(_snap, "target_user", None), "user_id", "") or "").strip()
                        if _snap is not None and _target_uid == str(_focus_uid).strip():
                            _relation_source = _snap.to_relation_dict()
                    if isinstance(_relation_source, dict):
                        _raw["core_social_value"] = float(_relation_source.get("social_value", 0.0) or 0.0)
                        _raw["core_trust_value"] = float(_relation_source.get("trust_value", 0.0) or 0.0)
                        _raw["core_annoyance"] = float(_relation_source.get("annoyance_value", 0.0) or 0.0)
                        _raw["core_affection"] = float(_relation_source.get("affection", 0.0) or 0.0)
                        _raw["core_interaction_count"] = int(_relation_source.get("interaction_count", 0) or 0)
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
            # 如果social_value_core没读到，尝试从relation_result回填
            if "core_social_value" not in _raw:
                _rr = getattr(self, "_cached_relation_result", None)
                if isinstance(_rr, dict):
                    _csv = float(_rr.get("social_value", 0.0) or 0.0)
                    if _csv > 0.01:
                        _raw["core_social_value"] = _csv
                        _raw["core_trust_value"] = float(_rr.get("trust_value", 0.0) or 0.0)
                        _raw["core_affection"] = float(_rr.get("affection", 0.0) or 0.0)
            _part = getattr(self, "_cached_participant_summary", None)
            if isinstance(_part, dict):
                _hot = int(_part.get("hot_count", 0) or 0)
                _warm = int(_part.get("warm_count", 0) or 0)
                _total_p = max(1, _hot + _warm + int(_part.get("normal_count", 0) or 0))
                _raw["group_engagement_score"] = (_hot * 1.0 + _warm * 0.6) / max(1, _total_p)
            # ── MoodBar 数据源：代谢约束 + 情绪追踪 ──
            if "mood" not in _raw:
                _raw["mood"] = 0.5
            if "curiosity" not in _raw:
                _raw["curiosity"] = 0.3
            _efr = getattr(self, "_cached_emotion_feedback_report", None)
            if isinstance(_efr, dict):
                _mood_delta = float(_efr.get("net_mood_delta", 0.0) or 0.0)
                _raw["mood"] = max(0.05, min(0.95, 0.5 + _mood_delta))
                _raw["curiosity"] = float(_efr.get("curiosity_component", 0.3) or 0.3)
            # ── AttentionBar 数据源：主观注意力流(GAP-W) ──
            _attn = getattr(self, "_cached_attention_snapshot", None)
            if isinstance(_attn, dict):
                _raw["attention_mode"] = str(_attn.get("state", "") or "")
                _raw["visibility_threshold"] = float(_attn.get("visibility_threshold", 0.30) or 0.30)
                _raw["process_ratio"] = float(_attn.get("process_ratio", 0.50) or 0.50)
                _raw["peek_desire"] = float(_attn.get("peek_desire", 0.3) or 0.3)
                _raw["withdrawal_depth"] = float(_attn.get("withdrawal_depth", 0.0) or 0.0)
                _raw["empty_peeks"] = int(_attn.get("consecutive_empty_peeks", 0) or 0)
                _raw["last_look_ago"] = float(_attn.get("since_last_look_sec", 999.0) or 999.0)
            # ── NightBar 数据源：夜间节律系统 ──
            _np = getattr(self, "_cached_night_phase", None)
            if _np:
                _np_val = _np.value if hasattr(_np, "value") else str(_np)
                _raw["night_phase"] = _np_val
            _ns = getattr(self, "_cached_night_summary", None)
            if isinstance(_ns, dict):
                _raw["pressure"] = float(
                    _ns.get(
                        "pressure_total",
                        _ns.get("overnight_pressure", getattr(self, "_overnight_pressure_total", 0.0)),
                    )
                    or 0.0
                )
                _raw["burnthrough"] = str(_ns.get("is_burnthrough", "") or "")
                _raw["expression_style"] = str(_ns.get("expression_style", "normal") or "normal")
            # ── SafetyBar 数据源：安全边界融合(GAP-R) ──
            _safe = getattr(self, "_cached_safety_assessment", None)
            if isinstance(_safe, dict):
                _raw["safety_level"] = str(_safe.get("overall_level_label", "安全") or "安全")
                _raw["safety_score"] = float(_safe.get("overall_score", 0.0) or 0.0)
                _raw["dominant_threat"] = str(_safe.get("dominant_threat", "") or "")
                _raw["blocked"] = str(_safe.get("blocked", "False") or "False")
                _raw["bar_delta"] = float(_safe.get("bar_penalty", 0.0) or 0.0)
            # ── MemoryBar 数据源：记忆治理(GAP-Q) ──
            _mg = getattr(self, "_cached_memory_governance_snap", None)
            if isinstance(_mg, dict):
                _raw["total"] = int(_mg.get("total_entries", 0) or 0)
                _raw["utilization"] = float(_mg.get("utilization_ratio", 0.0) or 0.0)
                _raw["fresh"] = int(_mg.get("fresh_count", 0) or 0)
                _raw["stale"] = int(_mg.get("stale_count", 0) or 0)
                _raw["window"] = int(_mg.get("active_window_size", 0) or 0)
            # ── 构建仪表盘快照 ──
            _snap = _engine.build_dashboard(raw_sources=_raw, force=force)
            self._cached_dashboard_snapshot = _snap.to_display_dict()
            self._last_dashboard_build_ts = _now
            return self._cached_dashboard_snapshot
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 状态仪表盘构建异常: {exc}")
            return None

    def _get_dashboard_verdict(self) -> Dict[str, Any]:
        """获取当前仪表盘ActionVerdict的字典表示（优先使用缓存，避免重复空数据构建）"""
        try:
            from src.core.state_dashboard import get_state_dashboard

            _engine = get_state_dashboard(self.stream_id)
            if _engine._snapshot_cache is not None:
                return _engine._snapshot_cache.verdict.to_dict()
            _v = _engine.get_verdict()
            return _v.to_dict()
        except Exception:
            return {
                "urgency": "可稍后回",
                "process": True,
                "reply": False,
                "reason": "仪表盘初始化中",
                "tone": "正常回应",
                "path": "FALLBACK",
                "confidence": 0.3,
            }

    def _get_dashboard_status_line(self) -> str:
        """获取统一状态栏单行文本（优先使用已构建的缓存，避免重复空数据构建）"""
        if self._cached_dashboard_snapshot and isinstance(self._cached_dashboard_snapshot, dict):
            try:
                from src.core.state_dashboard import get_state_dashboard

                _engine = get_state_dashboard(self.stream_id)
                if _engine._snapshot_cache is not None:
                    return _engine._snapshot_cache.to_status_line()
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        try:
            from src.core.state_dashboard import get_state_dashboard

            _engine = get_state_dashboard(self.stream_id)
            _line = _engine.get_status_line()
            if _line:
                return _line
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            if _ch and _ch.chat_ceiling > 0:
                _chat_val = (_ch.chat_pool / _ch.chat_ceiling) * 100.0
                _energy = f"{_chat_val:.0f}%"
            else:
                _energy = f"{_ch.chat_pool:.0f}" if _ch else "?"
        except Exception:
            _energy = "?"
        _np = getattr(self, "_cached_night_phase", None)
        _np_label = "清醒"
        if _np:
            _np_label = _np.label() if hasattr(_np, "label") else str(_np)
        _sw_info = getattr(self, "_night_soft_wake_info", None)
        if _sw_info and isinstance(_sw_info, dict):
            _sw_action = _sw_info.get("action", "")
            if _sw_action in ("soft_wake", "grumpy_glance"):
                _sw_mood = _sw_info.get("mood", "")
                _np_label = f"😴迷糊({(_sw_mood[:4] if _sw_mood else '刚醒')})"
            elif _sw_action == "full_wake":
                _np_label = f"🌙夜醒({_sw_info.get('mood', '清醒')[:4]})"
        else:
            try:
                from src.core.night_cycle_system import get_night_cycle

                _ncs = get_night_cycle(self.stream_id)
                _ncs_s = _ncs.state_snapshot
                _half = _ncs_s.half_asleep_level
                _daily_fatigue_val = _ncs_s.daily_fatigue
                _drowsy_val = _ncs_s.drowsiness_value
                _pressure_val = _ncs_s.overnight_pressure
                _reserve_val = _ncs_s.sleep_reserve
                if _half > 0.2 or _daily_fatigue_val > 25 or _drowsy_val > 15:
                    _hour_now = datetime.datetime.now().hour
                    if _half > 0.6 or _daily_fatigue_val > 70:
                        _np_label = f"😵💫极疲(疲{_daily_fatigue_val:.0f}%困{_drowsy_val:.0f})"
                    elif _half > 0.35 or _daily_fatigue_val > 45 or (1 <= _hour_now < 7 and _daily_fatigue_val > 30):
                        _np_label = f"😴半醒(疲{_daily_fatigue_val:.0f}%压{_pressure_val:.0f})"
                    elif _drowsy_val > 30 or (1 <= _hour_now < 6 and _daily_fatigue_val > 20):
                        _np_label = f"😪困倦(D{_drowsy_val:.0f}P{_pressure_val:.0f})"
                    elif 1 <= _hour_now < 7:
                        _np_label = f"⚡夜深{_hour_now}点"
                    else:
                        _np_label = f"⚡低能(疲{_daily_fatigue_val:.0f}%储{_reserve_val:.0f})"
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
        _hour = datetime.datetime.now().hour
        _is_deep_night = 0 <= _hour < 6 or _hour >= 23
        _is_early_morning = 6 <= _hour < 9
        if _is_deep_night:
            _time_icon = "🌙"
        elif _is_early_morning:
            _time_icon = "🌅"
        else:
            _time_icon = "☀️"
        try:
            _snap = self._resolve_relation_view()
            _mood_val = float(_snap.get("mood_value", 0.5) or 0.5)
            _ann_val = float(_snap.get("annoyance_value", 0) or 0)
            if _ann_val >= 50:
                _mood_icon = "😤"
            elif _ann_val >= 25:
                _mood_icon = "😒"
            elif _mood_val < 0.3:
                _mood_icon = "😐"
            elif _mood_val > 0.7:
                _mood_icon = "😊"
            else:
                _mood_icon = "😌"
        except Exception:
            _mood_icon = "😌"
        return f"🔋{_energy} | {_mood_icon}{_np_label} | 👁扫描 | 👀观察 | {_time_icon}{_np_label} | 🛡️安全"

    async def _emit_target_profile(self, target_uid: str) -> None:
        """第二层：当前对象印象面板 - 对当前交互目标的心理关系快照

        有明确交互对象时输出好感、信任、烦恼、创伤、关系、调教阶段、活跃人格等。
        """
        if not target_uid:
            return
        try:
            profile_segments = [f"对象={target_uid[:8]}"]
            # 优先从世界快照的 target_user 取关系维度
            _snapshot_relation_loaded = False
            tick_snapshot = getattr(self, "_tick_world_snapshot", None)
            if tick_snapshot is not None:
                _t_user = getattr(tick_snapshot, "target_user", None)
                if _t_user and str(getattr(_t_user, "user_id", "") or "").strip():
                    _affection = float(getattr(_t_user, "affection", 0.0) or 0.0)
                    _trust = float(getattr(_t_user, "trust_value", 0.0) or 0.0)
                    _annoy = float(getattr(_t_user, "annoyance_value", 0.0) or 0.0)
                    _trauma = float(getattr(_t_user, "trauma_score", 0.0) or 0.0)
                    _pressure = float(getattr(_t_user, "psychological_pressure", 0.0) or 0.0)
                    _rel_label = str(getattr(_t_user, "custom_label", "") or "").strip()
                    _social = float(getattr(_t_user, "social_value", 0.0) or 0.0)
                    _injected = self._inject_realtime_emotion(
                        {
                            "annoyance_value": _annoy,
                            "psychological_pressure": _pressure,
                        }
                    )
                    _annoy = float(_injected.get("annoyance_value", _annoy) or _annoy)
                    _pressure = float(_injected.get("psychological_pressure", _pressure) or _pressure)
                    if self._metric_has_signal(_affection):
                        profile_segments.append(f"好感={_affection:.1f}")
                    if self._metric_has_signal(_trust):
                        profile_segments.append(f"信任={_trust:.1f}")
                    if _annoy > 0:
                        profile_segments.append(f"烦恼={_annoy:.1f}")
                    if _trauma > 0:
                        profile_segments.append(f"创伤={_trauma:.1f}")
                    if _rel_label:
                        profile_segments.append(f"关系={_rel_label}")
                    if _pressure > 0:
                        profile_segments.append(f"心理压力={_pressure:.1f}")
                    if _social != 0:
                        profile_segments.append(f"社交={_social:.1f}")
                    _snapshot_relation_loaded = True
            # 快照无数据时退回旧路径
            if not _snapshot_relation_loaded:
                relation_view = self._resolve_relation_view()
                if relation_view and target_uid == str(getattr(self, "_last_user_id", "") or "").strip():
                    affection = float(relation_view.get("affection", 0.0) or 0.0)
                    trust_value = float(relation_view.get("trust_value", 0.0) or 0.0)
                    annoy_val = float(relation_view.get("annoyance_value", 0.0) or 0.0)
                    trauma_val = float(relation_view.get("trauma_score", 0.0) or 0.0)
                    pressure = float(relation_view.get("psychological_pressure", 0.0) or 0.0)
                    rel_label = str(relation_view.get("custom_label", "") or "").strip()
                    social_value = float(relation_view.get("social_value", 0.0) or 0.0)
                    if self._metric_has_signal(affection):
                        profile_segments.append(f"好感={affection:.1f}")
                    if self._metric_has_signal(trust_value):
                        profile_segments.append(f"信任={trust_value:.1f}")
                    if annoy_val > 0:
                        profile_segments.append(f"烦恼={annoy_val:.1f}")
                    if trauma_val > 0:
                        profile_segments.append(f"创伤={trauma_val:.1f}")
                    if rel_label:
                        profile_segments.append(f"关系={rel_label}")
                    if pressure > 0:
                        profile_segments.append(f"心理压力={pressure:.1f}")
                    if social_value != 0:
                        profile_segments.append(f"社交={social_value:.1f}")

            # 情绪追踪器维度（权威覆盖：缓存状态优先 > 按uid查询）
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import (
                    get_emotion_tracker,
                )

                tracker = get_emotion_tracker(self.stream_id)
                _cached_emo = getattr(self, "_resolved_emo_state", None)
                if _cached_emo is not None:
                    emo_state = _cached_emo
                else:
                    emo_state = tracker.get_user_state(target_uid, create_if_missing=False)
                if emo_state:
                    _emo_affection = float(getattr(emo_state, "affection", 0) or 0)
                    if self._metric_has_signal(_emo_affection) and not any(
                        part.startswith("好感=") for part in profile_segments
                    ):
                        profile_segments.append(f"好感={_emo_affection:.1f}")
                    _emo_trust = float(
                        getattr(
                            emo_state,
                            "trust_value",
                            getattr(emo_state, "trust_score", 0),
                        )
                        or 0
                    )
                    if self._metric_has_signal(_emo_trust) and not any(
                        part.startswith("信任=") for part in profile_segments
                    ):
                        profile_segments.append(f"信任={_emo_trust:.1f}")
                    _tracker_annoy = float(getattr(emo_state, "annoyance", -1) or -1)
                    if _tracker_annoy >= 0:
                        _existing_annoy_idx = [i for i, p in enumerate(profile_segments) if p.startswith("烦恼=")]
                        if _existing_annoy_idx:
                            profile_segments[_existing_annoy_idx[0]] = f"烦恼={_tracker_annoy:.1f}"
                        else:
                            profile_segments.append(f"烦恼={_tracker_annoy:.1f}")
                    trauma_val = getattr(emo_state, "trauma_score", 0)
                    if trauma_val > 0 and not any(part.startswith("创伤=") for part in profile_segments):
                        profile_segments.append(f"创伤={trauma_val:.1f}")
                    rel_label = getattr(emo_state, "relationship", "陌生人")
                    _has_existing_relation = any(part.startswith("关系=") for part in profile_segments)
                    # 如果快照已提供关系标签，且emotion_tracker仍为默认值，不覆盖
                    if not _has_existing_relation and rel_label != "陌生人":
                        profile_segments.append(f"关系={rel_label}")
                    elif not _has_existing_relation:
                        # 没有任何来源提供关系标签时，才使用默认值
                        profile_segments.append(f"关系={rel_label}")
                    stage = getattr(emo_state, "training_stage", 0)
                    if stage > 0:
                        profile_segments.append(f"调教阶段={stage}")
                    blocked = getattr(emo_state, "is_blocked", False)
                    if blocked:
                        profile_segments.append("已屏蔽")
                    _tracker_pressure = float(getattr(emo_state, "psychological_pressure", -1) or -1)
                    if _tracker_pressure >= 0:
                        _existing_press_idx = [i for i, p in enumerate(profile_segments) if p.startswith("心理压力=")]
                        if _existing_press_idx:
                            profile_segments[_existing_press_idx[0]] = f"心理压力={_tracker_pressure:.1f}"
                        else:
                            profile_segments.append(f"心理压力={_tracker_pressure:.1f}")
                    stamina = getattr(emo_state, "stamina", 100)
                    if stamina < 100:
                        profile_segments.append(f"体力={stamina:.1f}")
                    try:
                        response_mode = tracker.get_layered_response_mode(target_uid)
                        tone = str(response_mode.get("tone", "neutral") or "neutral")
                        resp_len = str(response_mode.get("response_length", "normal") or "normal")
                        profile_segments.append(f"回复模式={tone}/{resp_len}")
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            # 社交值维度（优先从世界快照读取，避免散读）
            try:
                _snap_social_loaded = False
                tick_snapshot = getattr(self, "_tick_world_snapshot", None)
                if tick_snapshot is not None:
                    _t_user = getattr(tick_snapshot, "target_user", None)
                    if _t_user:
                        _sv = float(getattr(_t_user, "social_value", 0.0) or 0.0)
                        _tv = float(getattr(_t_user, "trust_value", 0.0) or 0.0)
                        if not any(part.startswith("社交=") for part in profile_segments) and _sv != 0:
                            profile_segments.append(f"社交={_sv:.1f}")
                        if not any(part.startswith("信赖度=") for part in profile_segments) and _tv != 0:
                            profile_segments.append(f"信赖度={_tv:.1f}")
                        _snap_social_loaded = True
                if not _snap_social_loaded:
                    if not any(part.startswith("社交=") for part in profile_segments) or not any(
                        part.startswith("信赖度=") for part in profile_segments
                    ):
                        _d6p = EnergyChainDimension.get_instance()
                        _d6sp = _d6p._ensure_channel(self.stream_id)
                        full_snap = type(
                            "FS",
                            (),
                            {
                                "social_value": _d6sp.social_value,
                                "trust_value": 0.0,
                                "annoyance_value": _d6sp.annoyance_level,
                            },
                        )()
                        if not any(part.startswith("社交=") for part in profile_segments):
                            profile_segments.append(f"社交={full_snap.social_value:.1f}")
                        if full_snap.trust_value != 0 and not any(
                            part.startswith("信赖度=") for part in profile_segments
                        ):
                            profile_segments.append(f"信赖度={full_snap.trust_value:.1f}")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            # 活跃人格
            try:
                from src.modules.modcore.dynamic_persona.persona_switcher import (
                    get_persona_switcher,
                )

                switcher = get_persona_switcher()
                blend = switcher.get_blended_persona_data(self.stream_id)
                persona_obj = blend.get("active_persona")
                if persona_obj and hasattr(persona_obj, "name"):
                    persona_tag = persona_obj.name
                    if not blend.get("is_main", True):
                        persona_tag += "(副)"
                    profile_segments.append(f"人格={persona_tag}")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            logger.info(f"{self.log_prefix} 当前对象印象 " + " ".join(profile_segments))
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 当前对象印象输出异常: {exc}")

    def _emit_action_verdict(
        self,
        verdict_action: str,
        verdict_reason: str,
        pipeline_elapsed: float,
        confidence: float = -1.0,
    ) -> None:
        """第三层：动作裁定面板 - 每次决策结果输出

        不论 reply/no_reply/check_later/observe/early_exit/autonomy_guard，
        每次决策都必须可见。
        """
        try:
            gap_since_speak = time.time() - self._last_speak_time if self._last_speak_time > 0 else -1
            quiet_left = max(0.0, self._planner_quiet_until - time.time())
            verdict_parts = [
                f"裁定={verdict_action}",
                f"原因={verdict_reason[:60]}",
                f"连续跳过={self._consecutive_skip_ticks}",
                f"连续发言={self._consecutive_speaks:.1f}",
            ]
            if gap_since_speak >= 0:
                if gap_since_speak < 60:
                    verdict_parts.append(f"上次发言={gap_since_speak:.0f}s前")
                else:
                    verdict_parts.append(f"上次发言={gap_since_speak / 60:.1f}min前")
            if confidence >= 0:
                verdict_parts.append(f"置信={confidence:.2f}")
            if self._legacy_constraint_hits > 0:
                verdict_parts.append(f"硬约束次数={self._legacy_constraint_hits}")
            if self._last_legacy_penalty > 0:
                verdict_parts.append(f"硬约束惩罚={self._last_legacy_penalty:.1f}")
            if quiet_left > 0:
                verdict_parts.append(f"冷却剩余={quiet_left:.0f}s")
            if self._last_legacy_reason and verdict_action in {
                "legacy_constraint",
                "autonomy_block",
                "early_exit",
                "no_action",
                "no_reply",
            }:
                verdict_parts.append(f"硬约束因子={self._last_legacy_reason[:40]}")
            legacy_breakdown = getattr(self, "_last_legacy_breakdown", None)
            if (
                ("reply" in verdict_action or "proactive" in verdict_action)
                and isinstance(legacy_breakdown, dict)
                and legacy_breakdown
            ):
                summary = (
                    f"R={legacy_breakdown.get('reply_readiness', 0):.1f} "
                    f"B={legacy_breakdown.get('base', 0):.0f} "
                    f"+M={legacy_breakdown.get('mention_bonus', 0):.0f} "
                    f"+Q={legacy_breakdown.get('question_bonus', 0):.0f} "
                    f"+S={legacy_breakdown.get('salience_bonus', 0):.0f} "
                    f"-Res={legacy_breakdown.get('resource_penalty', 0):.0f} "
                    f"-Rel={legacy_breakdown.get('relation_penalty', 0):.0f} "
                    f"-Rep={legacy_breakdown.get('repeat_penalty', 0):.0f}"
                )
                verdict_parts.append(summary)
            verdict_parts.append(f"管线耗时={pipeline_elapsed:.2f}s")
            logger.info(f"{self.log_prefix} 动作裁定 " + " ".join(verdict_parts))
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 动作裁定输出异常: {exc}")


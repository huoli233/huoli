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

class LoopMainDriverMixin:
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

        # ── 统一状态仪表盘：状态闸门入口 ──
        _dashboard_snap = self._build_state_dashboard(force=True)
        if _dashboard_snap:
            _dash_verdict = _dashboard_snap.get("reply_decision", {})
            _dash_urgency = str(_dash_verdict.get("reply_urgency", "") or "")
            # 硬阻断检查：精力耗尽/深睡/安全封锁 → 直接跳过
            if _dash_verdict.get("should_process") is False:
                logger.info(
                    f"{self.log_prefix} 🛑 仪表盘硬阻断: {_dash_verdict.get('decision_reason', '未知')} urgency={_dash_urgency}"
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
        _decision_runtime = self._build_decision_runtime(
            now=now,
            source="reactive",
            legacy_gate=legacy_gate,
            llm_call_level=int(_llm_call_level or 0),
            tentative=bool(_tentative),
            voice_conclusion=voice_conclusion,
            model_governor=_model_governor,
            behavior_verdict=_behavior_for_model,
            dashboard_verdict=(_dashboard_snap or {}).get("reply_decision", {}) if isinstance(_dashboard_snap, dict) else {},
            relation_view=_relation_for_model,
            direct_ping=bool(_force_direct_ping),
            admin_forced=bool(_is_admin_forced),
            gateway_force=bool(_is_gateway_force),
        )
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
                self._store_execution_runtime(
                    initial_verdict=_decision_runtime,
                    should_act=False,
                    reply_sent=False,
                    final_action="rest",
                    execution_stage="voice_action_rest",
                    execution_reason=str(_thinking_text or "内心判断当前更适合休息"),
                    source="voice_action",
                    blocker="内心独白要求休息",
                    extra_votes={"voice_action": "rest"},
                    extra_blocking=["当前回合转入休息姿态"],
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
                self._store_execution_runtime(
                    initial_verdict=_decision_runtime,
                    should_act=False,
                    reply_sent=False,
                    final_action="disengage",
                    execution_stage="voice_action_disengage",
                    execution_reason=str(_thinking_text or "内心判断当前应该放下手机"),
                    source="voice_action",
                    blocker="内心独白要求脱离当前会话",
                    extra_votes={"voice_action": "disengage"},
                    extra_blocking=["当前回合转入黑屏休息"],
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
                self._store_execution_runtime(
                    initial_verdict=_decision_runtime,
                    should_act=False,
                    reply_sent=False,
                    final_action="lurk",
                    execution_stage="voice_action_lurk",
                    execution_reason=str(_thinking_text or "内心判断当前只窥屏不介入"),
                    source="voice_action",
                    blocker="内心独白要求潜水观察",
                    extra_votes={"voice_action": "lurk"},
                    extra_blocking=["当前回合只保留窥屏观察"],
                )
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

        _final_reply = _decision_runtime.next_action == "reply"
        _final_upgrade = _decision_runtime.next_action == "upgrade"
        _final_skip = not (_final_reply or _final_upgrade)
        if _final_upgrade:
            self._mark_decision_winner("decision_runtime_upgrade")
        elif _final_reply:
            self._mark_decision_winner("decision_runtime_reply")
        else:
            self._mark_decision_winner("decision_runtime_skip")
        logger.info(
            f"{self.log_prefix} [统一裁定] action={_decision_runtime.next_action} "
            f"reply={_decision_runtime.should_reply} model={_decision_runtime.model_path} "
            f"complexity={_decision_runtime.complexity_label} reason={_decision_runtime.decision_reason}"
        )

        # 动态比例旁路已移除：是否回复完全由LLM内心独白+维度网关force_reply决定

        if _final_skip:
            logger.info(f"{self.log_prefix} 🎯 最终决策: 跳过，直接观察")
            self._last_flow_blocker = self._last_flow_blocker or f"最终决策跳过 algo={_llm_call_level}"
            self._store_execution_runtime(
                initial_verdict=_decision_runtime,
                should_act=False,
                reply_sent=False,
                final_action="observe",
                execution_stage="decision_runtime_skip",
                execution_reason=self._last_flow_blocker or _decision_runtime.decision_reason,
                model_path="skip",
                source="decision_runtime",
                blocker=self._last_flow_blocker or _decision_runtime.decision_reason,
            )
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
                    if _snap_annoyance >= 72 or _snap_pressure >= 55:
                        logger.info(
                            f"{self.log_prefix} ⚠️ 门控={legacy_gate}但情绪偏高"
                            f"(烦躁{_snap_annoyance:.0f}/压力{_snap_pressure:.0f})，确定降级观察"
                        )
                        should_act = False
                    elif _has_strong_reply_evidence:
                        should_act = True
                        logger.info(
                            f"{self.log_prefix} 统一结算 门控={legacy_gate}，强证据存在且情绪未达硬拦截，允许进入回复执行层"
                        )
                else:
                    if _intent_primary == "wait" and _intent_silence >= 1:
                        if _intent_silence >= 3:
                            should_act = False
                            logger.info(
                                f"{self.log_prefix} 🤐 意图池primary=wait(静默{_intent_silence}轮)"
                                f"，确定跳过"
                            )
                        elif _has_strong_reply_evidence:
                            should_act = True
                            logger.info(
                                f"{self.log_prefix} 统一结算 门控={legacy_gate}，强证据存在且意图wait未达硬拦截，允许回复"
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
                            should_act = _surface_mask < 8.5
                            logger.info(f"{self.log_prefix} 🎭 表层伪装偏高({_surface_mask:.1f})，确定性收缩行动意愿")
                        if should_act and _submission >= 6.0:
                            should_act = _submission < 8.0
                            logger.info(f"{self.log_prefix} 🙈 顺从度偏高({_submission:.1f})，确定性降低主动表达")
                        if should_act and _shyness >= 75.0 and legacy_gate not in {"force_reply"}:
                            should_act = _shyness < 85.0
                            logger.info(f"{self.log_prefix} 😳 害羞值偏高({_shyness:.1f})，非强制场景确定性收缩回复")
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
            if _neg_composite >= 72:
                should_act = False
                logger.info(f"{self.log_prefix} SOC-03 负面情绪确定回避: composite={_neg_composite:.1f}>=72")
            else:
                logger.info(f"{self.log_prefix} SOC-03 负面情绪提示: composite={_neg_composite:.1f}，未达硬拦截")

        if should_act:
            _pre_reply_resource_snapshot = self._capture_pre_reply_resource_snapshot()
            self._apply_plan_drain()
            await self._deduct_shared_resources(incoming_batch)
            _executed_planner_decision = planner_decision
            _reply_source = "planner"
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
                        _executed_planner_decision = _upgrade_proactive
                        _reply_source = "llm_autonomous"
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
                        _reply_source = "voice_driven"
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
                    _reply_source = "planner_proactive"
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
                    _reply_source = "planner"
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
                self._store_execution_runtime(
                    initial_verdict=_decision_runtime,
                    should_act=True,
                    reply_sent=True,
                    final_action="reply",
                    execution_stage="reply_sent",
                    execution_reason="回复已发送",
                    model_path="large" if _reply_source == "llm_autonomous" else _decision_runtime.model_path,
                    source=_reply_source,
                    planner_decision=_executed_planner_decision,
                    confidence=getattr(_executed_planner_decision, "confidence", _decision_runtime.confidence),
                    extra_votes={"actual_reply_made": True, "reply_source": _reply_source},
                )
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
                self._store_execution_runtime(
                    initial_verdict=_decision_runtime,
                    should_act=False,
                    reply_sent=False,
                    final_action="observe",
                    execution_stage="reply_aborted",
                    execution_reason=_no_reply_reason,
                    model_path="skip",
                    source=_reply_source or "execution",
                    planner_decision=_executed_planner_decision,
                    blocker=_no_reply_reason,
                    extra_votes={"actual_reply_made": False, "reply_source": _reply_source or "execution"},
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
            self._store_execution_runtime(
                initial_verdict=_decision_runtime,
                should_act=False,
                reply_sent=False,
                final_action="observe",
                execution_stage="final_no_action",
                execution_reason=_inaction_reason,
                model_path="skip",
                source="final_gate",
                planner_decision=planner_decision,
                blocker=_inaction_reason,
            )
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

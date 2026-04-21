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

class ProactiveIdleReplyMixin:
    async def _try_algo_takeover_request(
        self,
        now: float,
        incoming_batch: List,
        voice_conclusion,
        thinking_text: str,
    ) -> Optional[Dict]:
        """算法接管请求：询问模型是否要接管

        当小模型说不想回（欲望<7）时，算法询问模型是否要接管。
        - 模型选择接管 → 强制回复
        - 模型选择不接管 → 算法接管并规划休息
        """
        try:
            from src.core.adaptive_threshold_learner_v2 import (
                get_adaptive_threshold_learner_v2,
            )

            learner = get_adaptive_threshold_learner_v2(self.stream_id)

            latest = incoming_batch[-1] if incoming_batch else None
            raw_text = getattr(latest, "processed_plain_text", "") if latest else ""
            speaker_name = getattr(latest, "user_nickname", "") or "" if latest else ""
            speaker_id = getattr(latest, "user_id", "") or "" if latest else ""

            psych = {
                "mood": (getattr(voice_conclusion, "current_mood", "") if voice_conclusion else "一般"),
                "mental_fatigue": 0.0,
                "annoyance": float(getattr(self, "_cached_annoyance_value", 0.0) or 0.0),
                "favor": int(float(getattr(self, "_cached_affection_value", 0.0) or 0.0)),
            }

            current_state_info = f"模型内心抗拒='{thinking_text[:50] if thinking_text else '无'}', 请决定是否接管"

            _dialogue_history = []
            try:
                from src.chat.heart_flow.inner_voice import SelfDialogueEngine

                _sde = getattr(self, "_self_dialogue_engine", None)
                if _sde is None:
                    _sde = SelfDialogueEngine.get_instance(self.stream_id)
                    self._self_dialogue_engine = _sde
                if hasattr(_sde, "_recent_thread"):
                    for _th in list(_sde._recent_thread)[-8:]:
                        if isinstance(_th, dict):
                            _dialogue_history.append(_th)
                if not _dialogue_history and incoming_batch:
                    for _msg in incoming_batch[-5:]:
                        _content = str(
                            getattr(_msg, "processed_plain_text", "") or getattr(_msg, "plain_text", "") or ""
                        )
                        if _content:
                            _dialogue_history.append({"role": "user", "content": _content})
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            _rapport_desc = ""
            _profile_desc = ""
            _memory_desc = ""
            _condition_desc = ""
            try:
                _rel_snap = getattr(self, "_last_relation_snapshot", {}) or {}
                _rapport_desc = f"{speaker_name}: 好感{_rel_snap.get('affection', '?')}, 信任{
                    _rel_snap.get('trust_value', '?')
                }, 烦躁{_rel_snap.get('annoyance_value', '?')}"
                _profile_desc = _rel_snap.get("profile_summary", "") or "(暂无稳定画像)"
                _condition_desc = f"思考值={_rel_snap.get('thinking_ratio', '?')}, 聊天值={
                    _rel_snap.get('chat_fuel', '?')
                }, 活跃度={_rel_snap.get('activity_level', '?')}"
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            is_at_me = getattr(latest, "mentioned_me", False) if latest else False

            if not learner.should_offer_takeover():
                return None

            learner.offer_takeover_to_model()

            decision = await learner.decide_with_model(
                message_text=raw_text,
                user_name=speaker_name,
                user_id=speaker_id,
                psych=psych,
                is_at_me=is_at_me,
                is_private=self.stream_id.startswith("private"),
                current_state_info=current_state_info,
                is_state_reevaluation=True,
                dialogue_history=(_dialogue_history if _dialogue_history else None),
                rapport_desc=_rapport_desc,
                profile_desc=_profile_desc,
                memory_desc=_memory_desc,
                condition_desc=_condition_desc,
            )

            if decision:
                action = decision.get("action", "待机")
                wants_to_takeover = action in (
                    "回复",
                    "接管",
                    "主动说话",
                    "吐槽",
                    "质疑",
                    "调侃",
                    "阴阳",
                )

                if wants_to_takeover:
                    learner.model_takeover_response(True)
                    learner.on_model_accepted(now)
                else:
                    learner.model_takeover_response(False)
                    learner.on_model_rejection(now)

                return {
                    "wants_to_takeover": wants_to_takeover,
                    "action": action,
                    "reason": decision.get("reason", ""),
                    "thought": decision.get("thought", ""),
                    "decision": decision,
                }

            learner.on_model_rejection(now)
            learner.reset_state_expiration()

        except Exception as exc:
            logger.debug(f"{self.log_prefix} 算法接管请求失败: {exc}")
        return None

    async def _run_peek_observe_loop(self, incoming_batch: List) -> None:
        """窥屏态观察循环：更新内部状态并消耗少量能量"""
        try:
            _d6 = EnergyChainDimension.get_instance()
            _human_batch = [m for m in (incoming_batch or []) if self._is_human_message_obj(m)]
            _content_len = sum(len(getattr(m, "content", "") or "") for m in _human_batch[-5:])
            _d6.on_event(
                "user_message",
                {
                    "channel_id": self.stream_id,
                    "content_length": _content_len,
                    "is_repeat": len(_human_batch) > 1,
                },
            )
            _ch = _d6._ensure_channel(self.stream_id)
            if _ch:
                _ch.chat_pool = max(0, _ch.chat_pool - 0.3)
                _ch.activity_level = max(0, _ch.activity_level - 0.2)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            _wm = self._cached_watch_level
            _wl = _wm.value if hasattr(_wm, "value") else str(_wm)
            logger.info(f"{self.log_prefix} 👁 窥屏态观察完成 级别={_wl} 不回复")
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    async def _run_peek_with_reflection(self, incoming_batch: List) -> object:
        """窥屏态概率性反思：以30%概率调用内心独白引擎，产生想法

        模拟"刷手机时瞥到一条消息，脑子里冒出个念头"的过程。
        返回独白结论对象，或 None 表示无波澜。
        """
        # 先执行基础观察（状态更新+能量扣除）
        await self._run_peek_observe_loop(incoming_batch)
        # 有消息到达时始终调用内心独白，让模型自主决定是否值得关注
        now = time.time()
        if not self._should_run_voice(now, incoming_batch, None, source="peek"):
            return None
        try:
            ambient_info = self._sample_channel_ambient()
            verdict = await self._invoke_inner_voice(
                incoming_batch,
                self._cached_awareness,
                ambient_info,
                now,
            )
            if verdict is not None and getattr(verdict, "is_valid", False):
                self._last_voice_ts = now
            return verdict
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 窥屏反思失败: {exc}")
            return None

    async def _try_idle_proactive(self, now: float, silence_sec: float) -> bool:
        """无新消息时的自主行为路径

        不用硬编码沉默阈值，而是让情感系统自然累积感受，
        由情感状态决定是否想要主动行为。
        返回 True 表示已执行主动行为。
        """
        if not hasattr(self, "_last_idle_proactive_ts"):
            self._last_idle_proactive_ts = 0.0
        if (now - self._last_idle_proactive_ts) < self._IDLE_PROACTIVE_COOLDOWN_SEC:
            return False
        # F5：夜间节律冻结——深睡/熬穿/浅睡(无窥屏窗) 时禁止空闲主动
        if self._cached_night_phase is not None:
            _np = self._cached_night_phase
            _np_name = getattr(_np, "name", str(_np)) if _np else ""
            if "DEEP" in _np_name or "VALLEY" in _np_name or "BURNED" in _np_name:
                return False
            if "DROWSY" in _np_name or "LIGHT" in _np_name:
                _peek_allowed = False
                try:
                    from src.core.night_cycle_system import get_night_cycle

                    _ncs_peek = get_night_cycle(self.stream_id)
                    _peek_allowed = _ncs_peek.evaluate_sleep_peek()
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
                if not _peek_allowed:
                    try:
                        _d6 = EnergyChainDimension.get_instance()
                        _nm = _d6._get_night_mode(self.stream_id)
                        if _nm and _nm.prob_multiplier < 0.3:
                            return False
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
        if self._should_back_off_idle_proactive():
            logger.info(f"{self.log_prefix} 💤 连续{self._unanswered_bot_turns}次主动未获回应，跳过本轮主动发言")
            self._last_idle_proactive_ts = now
            return False

        # ── 空闲期核心状态同步（修复：空闲路径原先完全跳过核心系统集成） ──
        # 创伤/压力门控：压力过高时拦截主动行为
        try:
            from src.modules.trauma.trauma_system import get_trauma_system

            _idle_trauma_state = get_trauma_system().get_state()
            if _idle_trauma_state.stress_accumulation > 8.0 or _idle_trauma_state.inner_chaos_level > 8.0:
                logger.debug(
                    f"{self.log_prefix} 💤 压力/混乱拦截 "
                    f"chaos={_idle_trauma_state.inner_chaos_level:.1f}, "
                    f"stress={_idle_trauma_state.stress_accumulation:.1f}"
                )
                self._last_idle_proactive_ts = now
                return False
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # 创伤系统：空闲期自然消解压力
        await self._update_trauma_system_state([])
        # 对话阶段感知：查询会话追踪器获取频道对话状态
        _idle_dialogue_phase = ""
        try:
            from src.chat.proactive.session_tracker import get_memoir_cabinet

            _idle_cabinet = get_memoir_cabinet()
            _idle_user_id = self._resolve_latest_human_user_id(
                allow_cached_fallback=False
            )
            _idle_memoir = _idle_cabinet.retrieve(_idle_user_id) if hasattr(_idle_cabinet, "retrieve") else None
            if _idle_memoir and hasattr(_idle_memoir, "phase"):
                _idle_dialogue_phase = (
                    _idle_memoir.phase.value if hasattr(_idle_memoir.phase, "value") else str(_idle_memoir.phase)
                )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        emotion_snapshot = self._update_emotion_state(now, relation_result={})
        if not emotion_snapshot:
            return False

        # ── 统一决策器评估（融合多源信号） ──
        arbiter_verdict = self._evaluate_proactive_decision(silence_sec)

        # ✅ 日志增强：展示决策器的详细评估过程
        signal_breakdown = (
            f"emo={getattr(arbiter_verdict, 'emotional_readiness', 0):.2f} "
            f"vit={getattr(arbiter_verdict, 'vitality_ratio', 0):.2f} "
            f"cont={getattr(arbiter_verdict, 'content_novelty', 0):.2f} "
            f"soc={getattr(arbiter_verdict, 'social_standing', 0):.2f} "
            f"des={getattr(arbiter_verdict, 'inner_voice_desire', 0):.2f} "
            f"rwd={getattr(arbiter_verdict, 'historical_reward', 0):.2f}"
        )

        if not arbiter_verdict.should_proceed:
            logger.debug(
                f"{self.log_prefix} 💤 沉默 {silence_sec:.0f}s 不想主动说话 ({arbiter_verdict.rationale}) | {signal_breakdown}"
            )
            self._last_idle_proactive_ts = now
            return False
        governor_verdict = self._evaluate_behavior_governor(
            incoming_batch=[],
            silence_sec=silence_sec,
            requested_mode="proactive",
            is_background=False,
        )
        if not governor_verdict.allow_generation or governor_verdict.reply_mode != "proactive":
            logger.info(
                f"{self.log_prefix} 💤 行为Governor保持闭嘴: {self._summarize_behavior_governor(governor_verdict)}"
            )
            self._last_idle_proactive_ts = now
            return False
        reason = arbiter_verdict.rationale
        logger.info(
            f"{self.log_prefix} 💬 沉默 {silence_sec:.0f}s 想主动说话: {reason}"
            + f" | {signal_breakdown}"
            + f" | governor={self._summarize_behavior_governor(governor_verdict)}"
            + (f" | 对话阶段={_idle_dialogue_phase}" if _idle_dialogue_phase else "")
        )
        awareness_snapshot = None
        voice_conclusion = None
        # 空闲主动路径也并行化感知 + 内心独白
        _idle_run_p = self._should_run_perception(now)
        _idle_run_v = self._should_run_voice(
            now,
            [],
            None,
            is_proactive=True,
            source="idle_proactive",
            behavior_verdict=governor_verdict,
        )
        _idle_jobs: list = []
        _idle_keys: list = []
        if _idle_run_p:
            _idle_jobs.append(self._invoke_perception([], now))
            _idle_keys.append("p")
        if _idle_run_v:
            _idle_jobs.append(self._invoke_autonomous_voice(reason))
            _idle_keys.append("v")
        if _idle_jobs:
            try:
                _stage_timeout = _parallel_stage_timeout()
                _idle_res = await asyncio.wait_for(
                    asyncio.gather(*_idle_jobs, return_exceptions=True),
                    timeout=_stage_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"{self.log_prefix} 💤 空闲主动链路感知/独白超时({_stage_timeout:.0f}s)，跳过本轮主动回复"
                )
                self._last_idle_proactive_ts = now
                return False
            _idle_map = dict(zip(_idle_keys, _idle_res, strict=True))
            if _idle_run_p:
                _ip = _idle_map.get("p")
                if isinstance(_ip, BaseException):
                    logger.warning(f"{self.log_prefix} 空闲主动感知异常: {_ip}，沿用已有感知缓存")
                elif _ip is not None:
                    awareness_snapshot = _ip
                    self._cached_awareness = awareness_snapshot
                    self._last_perception_ts = now
            if _idle_run_v:
                _iv = _idle_map.get("v")
                if isinstance(_iv, BaseException):
                    logger.warning(f"{self.log_prefix} 空闲主动独白异常: {_iv}，沿用已有独白缓存")
                elif _iv is not None:
                    voice_conclusion = _iv
                    self._cached_voice = voice_conclusion
                    self._last_voice_ts = now
                    self._last_proactive_voice_ts = now
                    self._align_states_with_inner_voice(voice_conclusion, source="idle_proactive")
                    _thought_len = len(str(getattr(voice_conclusion, "thinking", "") or ""))
                    _think_complexity = max(0.5, min(2.0, _thought_len / 40.0))
                    self._apply_think_drain(complexity=_think_complexity)
        if awareness_snapshot is None:
            awareness_snapshot = self._cached_awareness
        if voice_conclusion is None:
            voice_conclusion = self._cached_voice
        # 空闲路径：如果没有新鲜的内心独白，尝试自主思考
        if voice_conclusion is None or not getattr(voice_conclusion, "is_valid", False):
            try:
                auto_verdict = await asyncio.wait_for(
                    self._invoke_autonomous_voice(reason),
                    timeout=_parallel_stage_timeout(),
                )
            except asyncio.TimeoutError:
                auto_verdict = None
                logger.debug(f"{self.log_prefix} 空闲自主思考超时")
            if auto_verdict and getattr(auto_verdict, "is_valid", False):
                voice_conclusion = auto_verdict
                try:
                    from src.chat.proactive.intention_pool import (
                        get_intention_pool,
                    )

                    get_intention_pool().ingest_voice_verdict(
                        channel_id=self.stream_id,
                        verdict=auto_verdict,
                        speaker_id=str(getattr(self, "_last_user_id", "") or ""),
                    )
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
        _idle_recent = []
        try:
            _idle_recent = message_api.get_messages_by_time_in_chat(
                chat_id=self.stream_id,
                start_time=time.time() - 1800,
                end_time=time.time(),
                limit=20,
                limit_mode="latest",
                filter_mai=False,
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        _idle_repeat_signal = self._analyze_repetition_pressure(_idle_recent)
        if self._is_repeat_meme_or_low_info(_idle_repeat_signal):
            logger.info(f"{self.log_prefix} 💤 主动链检测到低信息复读/玩梗，取消本轮主动回复")
            self._last_idle_proactive_ts = now
            return False
        logger.info(f"{self.log_prefix} 💤 想主动说话，走专用主动回复执行器")
        _pre_reply_resource_snapshot = self._capture_pre_reply_resource_snapshot()
        self._apply_plan_drain()
        # 构建主动行为决策，传递决策器的意图
        from src.chat.heart_flow.llm_autonomous_planner import (
            AutonomousDecision,
        )

        _idle_decision = AutonomousDecision(
            should_act=True,
            action_type="proactive_speak",
            content_plan=self._llm_content_plan or "",
            reasoning=reason,
            confidence=arbiter_verdict.fused_score,
            emotional_state="",
            social_intention=self._llm_social_intention or "空闲想聊天",
        )
        # 收集最近消息作为上下文（包含bot自己的消息，让模型知道自己说过什么）
        # 捕获触发本次主动的核心意图id（供闭环追踪）
        _idle_proactive_intent_id = ""
        if voice_conclusion and hasattr(voice_conclusion, "dominant_unfinished_intent"):
            _dom = getattr(voice_conclusion, "dominant_unfinished_intent", None)
            if _dom and isinstance(_dom, dict):
                _idle_proactive_intent_id = str(_dom.get("intent_id", "") or "")
        self._last_proactive_intent_id = _idle_proactive_intent_id
        try:
            replied = await asyncio.wait_for(
                self._execute_proactive_reply(
                    llm_decision=_idle_decision,
                    incoming_batch=_idle_recent,
                ),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            replied = False
            logger.error(f"{self.log_prefix} ⚠️ 空闲主动回复超时(120s)")
        if replied:
            _idle_desire = 5
            if voice_conclusion is not None:
                _idle_desire = int(getattr(voice_conclusion, "reply_desire_level", 5) or 5)
            await self._finalize_external_proactive_reply_flow(
                _idle_recent,
                source="idle_proactive",
                desire_level=_idle_desire,
            )
            logger.info(
                f"{self.log_prefix} 💤 主动回复发送成功"
                + (f" source_intent={_idle_proactive_intent_id[:10]}" if _idle_proactive_intent_id else "")
            )
        else:
            self._restore_pre_reply_resource_snapshot(
                _pre_reply_resource_snapshot,
                reason="idle_proactive未形成有效回复",
            )
            logger.info(f"{self.log_prefix} 💤 主动回复生成失败")
        self._last_idle_proactive_ts = now
        return replied

    async def _execute_proactive_reply(
        self,
        llm_decision: "AutonomousDecision",
        incoming_batch: List,
        delivery_form: str = "standalone",
        mention_user_name: str = "",
        reference_user_name: str = "",
    ) -> bool:
        """主动回复执行器：将规划器或决策器的主动意图转化为实际回复。

        由空闲决策路径和统一规划器 proactive_speak 决策共同调用，
        构建主动行为上下文后交给 generate_reply 生成内容并发送。
        支持投递策略：standalone/quote/mention/quote+mention
        """
        if self._proactive_send_lock is None:
            self._proactive_send_lock = asyncio.Lock()
        if self._proactive_send_lock.locked():
            logger.info(f"{self.log_prefix} 主动回复仍在发送中，跳过重入触发")
            return False
        await self._proactive_send_lock.acquire()
        target_message = None
        try:
            from src.llm_models.utils_model import bind_stream_context

            # 绑定聊天流ID到当前异步任务，使并发守卫能按流限速
            bind_stream_context(self.stream_id)

            logger.info(
                f"{self.log_prefix} 开始执行主动回复: "
                f"意图={llm_decision.social_intention}, "
                f"内容规划={llm_decision.content_plan[:50] if llm_decision.content_plan else '无'}, "
                f"投递={delivery_form}, 引用={reference_user_name}, @={mention_user_name}"
            )

            restraint = await self._run_self_restraint_check(
                incoming_batch,
                source="proactive_reply",
                repetition_signal=self._analyze_repetition_pressure(incoming_batch),
                harassment_signal=self._analyze_harassment_pressure(incoming_batch),
            )
            if not restraint.get("allow", True):
                self._last_flow_blocker = f"proactive自省拦截:{restraint.get('reason', 'skip')}"
                logger.info(f"{self.log_prefix} 🧯 自省闸门拦截 proactive 回复: {restraint.get('reason', 'skip')}")
                return False

            # 根据投递策略选择目标消息
            _need_quote = "quote" in delivery_form
            _need_mention = "mention" in delivery_form
            from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

            target_message = acquire_reply_coordinator().select_reply_target(
                messages=incoming_batch,
                reference_user_name=reference_user_name,
                delivery_form=delivery_form,
                is_bot_message=self._is_bot_message_obj,
                preferred_selector=self._select_preferred_reply_message,
            )
            if target_message is None and _need_quote:
                logger.info(f"{self.log_prefix} 主动回复目标不可引用，回退为standalone发送")
            _silence_sec_for_governor = 0.0
            try:
                from src.chat.proactive.silence_watcher import get_quiet_monitor

                _silence_sec_for_governor = float(
                    get_quiet_monitor().measure_silence_sec(self.stream_id) or 0.0
                )
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
                _silence_sec_for_governor = max(
                    0.0,
                    time.time() - float(getattr(self, "_last_user_msg_time", 0.0) or 0.0),
                )
            behavior_verdict = self._evaluate_behavior_governor(
                incoming_batch=incoming_batch,
                silence_sec=_silence_sec_for_governor,
                requested_mode="proactive",
                target_message=target_message,
                delivery_form=delivery_form,
                reference_user_name=reference_user_name,
                mention_user_name=mention_user_name,
                is_background=False,
            )
            if not behavior_verdict.allow_generation or behavior_verdict.reply_mode != "proactive":
                self._last_flow_blocker = f"behavior_governor:{behavior_verdict.reply_mode}"
                self._mark_message_content_deferred(target_message, "behavior_governor_blocked")
                logger.info(
                    f"{self.log_prefix} 🎛️ 行为Governor拦截 proactive 回复: "
                    f"{self._summarize_behavior_governor(behavior_verdict)}"
                )
                return False
            self._mark_message_content_processing(target_message)
            voice_summary = self._build_voice_execution_summary(getattr(self, "_cached_voice", None), target_message)
            repetition_signal = self._analyze_repetition_pressure(incoming_batch)
            decision_context_packet = self._build_decision_context_packet(
                list(incoming_batch),
                repetition_signal=repetition_signal,
            )
            relation_view = self._resolve_relation_view()
            context_execution_block = self._build_context_execution_block(
                target_message=target_message,
                voice_conclusion=getattr(self, "_cached_voice", None),
                repetition_signal=repetition_signal,
                decision_context_packet=decision_context_packet,
                relation_snapshot=relation_view,
            )

            # 构建回复原因
            from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

            reply_reason = acquire_reply_coordinator().compose_reply_reason(
                base_reason=f"主动行为: {llm_decision.social_intention or '想说话'}",
                content_plan=llm_decision.content_plan or "",
                voice_reason=voice_summary.get("reason", ""),
            )

            # 构建额外信息，传递给回复生成器
            extra_info = voice_summary.get("panel", "")
            if llm_decision.content_plan:
                extra_info = (
                    f"{extra_info}\n[主动行为意图] {llm_decision.social_intention or ''}\n[内容规划] {
                        llm_decision.content_plan
                    }"
                    if extra_info
                    else f"[主动行为意图] {llm_decision.social_intention or ''}\n[内容规划] {llm_decision.content_plan}"
                )
                if llm_decision.emotional_state:
                    extra_info += f"\n[当前感受] {llm_decision.emotional_state}"
            voice_guard = self._build_voice_execution_guard(getattr(self, "_cached_voice", None), target_message)
            if voice_guard:
                extra_info = f"{extra_info}\n{voice_guard}" if extra_info else voice_guard
            if context_execution_block:
                extra_info = (
                    f"{extra_info}\n{context_execution_block}" if extra_info else context_execution_block
                )
            _no_target_guard = ""
            if target_message is None:
                _no_target_guard = self._build_proactive_no_target_guard(incoming_batch)
            if _no_target_guard:
                extra_info = f"{extra_info}\n{_no_target_guard}" if extra_info else _no_target_guard
            _governor_panel = (
                f"[行为Governor] mode={behavior_verdict.reply_mode} "
                f"interrupt={behavior_verdict.interrupt_level} "
                f"quote={behavior_verdict.quote_policy} "
                f"silence={behavior_verdict.silence_policy} "
                f"allow={'yes' if behavior_verdict.allow_generation else 'no'} "
                f"watch_cap={behavior_verdict.max_watch_rank} "
                f"model={behavior_verdict.model_tier} "
                f"reason={','.join(behavior_verdict.reason_codes[:4]) or 'none'}"
            )
            extra_info = f"{extra_info}\n{_governor_panel}" if extra_info else _governor_panel
            # 注入内心独白的观察记忆（最近看到什么、想了什么、为什么沉默）
            try:
                from src.chat.heart_flow.inner_voice import get_self_dialogue_engine

                _obs_ctx = get_self_dialogue_engine(self.stream_id).export_observation_context()
                if _obs_ctx:
                    extra_info = f"{extra_info}\n[最近观察]\n{_obs_ctx}" if extra_info else f"[最近观察]\n{_obs_ctx}"
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")

            user_style_guide = ""
            if target_message:
                target_user_id = getattr(target_message, "user_id", "") or ""
                if target_user_id:
                    user_style_guide = self._get_user_style_guide(target_user_id)
            from src.chat.replyer.context_block_builder import build_shared_reply_parts

            _shared_parts = build_shared_reply_parts(
                self_memory=self._build_self_reply_memory(),
                continuity_context=self._build_self_continuity_context(),
                user_style_guide=user_style_guide,
                persona_hint=self._build_persona_hint(),
                reply_style_context=self._build_reply_style_context(relation_view),
                length_hint=self._build_dynamic_length_hint(target_message, user_style_guide),
                restraint_mode=str(restraint.get("mode", "allow") or "allow"),
                short_only_text="[自省闸门约束] 你短时间内已经说了很多，这次如果一定要说，只能一句非常短的收束，不要继续带节奏。",
            )
            if _shared_parts:
                extra_info = f"{extra_info}\n" + "\n".join(_shared_parts) if extra_info else "\n".join(_shared_parts)
            harassment_signal = self._analyze_harassment_pressure(incoming_batch)
            from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

            style_route = acquire_reply_coordinator().resolve_style_route(
                delivery_form=delivery_form,
                target_message=target_message,
                reference_user_name=reference_user_name,
                mention_user_name=mention_user_name,
                fallback_selector=lambda msg, rel, hs: self._decide_reply_style(
                    target_message=msg,
                    relation_snapshot=rel,
                    harassment_signal=hs,
                ),
                relation_view=relation_view,
                harassment_signal=harassment_signal,
                is_bot_message=self._is_bot_message_obj,
                governor_quote_policy=behavior_verdict.quote_policy,
            )
            # @mention：将用户昵称注入到内容规划中，让生成器在回复中自然提到对方
            if _need_mention and mention_user_name:
                _mention_hint = f"\n[投递指令] 这条消息需要@{mention_user_name}，在开头或合适的位置自然地提到对方名字"
                extra_info = f"{extra_info}{_mention_hint}" if extra_info else _mention_hint
            from src.chat.replyer.context_block_builder import append_reply_style, merge_extra_info

            _style_parts: List[str] = []
            append_reply_style(_style_parts, style_route)
            extra_info = merge_extra_info(extra_info, *_style_parts)
            self._emit_reply_generation_summary(
                target_message=target_message,
                style_route=style_route,
                relation_snapshot=relation_view,
                context_execution_block=context_execution_block,
                extra_info=extra_info,
                source="proactive",
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
                request_type="proactive_reply",
                think_level=1,
            )

            if not success or not llm_response or not llm_response.reply_set:
                self._last_flow_blocker = "proactive回复生成失败"
                self._mark_message_content_deferred(target_message, "proactive_generation_failed")
                logger.warning(f"{self.log_prefix} 回复生成失败")
                return False

            # 发送回复
            response_set = llm_response.reply_set
            selected_expressions = llm_response.selected_expressions

            cycle_timers: Dict[str, float] = {}
            thinking_id = f"proactive_{int(time.time() * 1000)}"

            loop_info, reply_text, _, reply_trace_meta = await self._send_and_store_reply(
                response_set=response_set,
                action_message=target_message,
                cycle_timers=cycle_timers,
                thinking_id=thinking_id,
                actions=[],
                selected_expressions=selected_expressions,
                quote_message=bool(style_route.get("quote_message", False)),
                pre_send_risk_note=(
                    f"proactive annoyance={relation_view.get('annoyance_value', 0)} "
                    f"pressure={relation_view.get('psychological_pressure', 0)}"
                ),
                pre_send_audit_label="proactive",
            )
            if not (bool(loop_info) or bool(str(reply_text or "").strip())):
                self._last_flow_blocker = "proactive回复发送失败"
                self._mark_message_content_deferred(target_message, "proactive_send_failed")
                return False

            await self._finalize_sent_reply(
                reply_text=reply_text,
                loop_info=loop_info,
                target_message=target_message,
                reply_reason=reply_reason,
                relation_view=relation_view,
                llm_response=llm_response,
                was_proactive=True,
                action_name="proactive_reply",
                quality=0.8,
                audit_label="proactive",
                reply_trace_meta=reply_trace_meta,
            )
            logger.info(f"{self.log_prefix} 成功发送: {reply_text[:50]}...")
            return True

        except Exception as exc:
            self._last_flow_blocker = f"proactive异常:{type(exc).__name__}"
            self._mark_message_content_deferred(
                target_message or self._get_latest_human_message(incoming_batch),
                "proactive_reply_exception",
            )
            logger.error(f"{self.log_prefix} 执行失败: {exc}")
            return False
        finally:
            if self._proactive_send_lock.locked():
                self._proactive_send_lock.release()

    async def _generate_and_send_proactive_reply(
        self,
        incoming_batch: List,
        desire: float,
        thought: str,
        target_uid: str,
        awareness=None,
        ambient=None,
        emotion=None,
        arbiter_reason: str = "",
        proactive_topic: str = "",
        proactive_emotion: str = "",
        delivery_form: str = "standalone",
        mention_user_name: str = "",
        reference_user_name: str = "",
    ) -> bool:
        """生成并发送主动回复（轻量级版本）

        用于静默期间的主动行为，比 _execute_proactive_reply 更轻量。
        直接基于内心独白的想法生成回复，不需要完整的规划器决策。
        """
        try:
            from src.chat.heart_flow.llm_autonomous_planner import (
                AutonomousDecision,
            )

            # 用规划器给出的话题构建具体意图，避免千篇一律的"想主动说话"
            _intention = "想主动说话"
            if proactive_topic:
                _intention = f"想聊聊{proactive_topic}"
            _emotion_label = proactive_emotion or (getattr(emotion, "dominant_emotion", "平静") if emotion else "平静")
            _content = thought
            if proactive_topic and proactive_topic not in (thought or ""):
                _content = f"[话题: {proactive_topic}] {thought}"

            decision = AutonomousDecision(
                should_act=True,
                action_type="proactive_reply",
                social_intention=_intention,
                content_plan=_content,
                emotional_state=_emotion_label,
                target_user_id=target_uid,
                confidence=min(1.0, desire / 10.0),
            )

            # 调用完整的执行器，传递投递策略
            try:
                return await asyncio.wait_for(
                    self._execute_proactive_reply(
                        llm_decision=decision,
                        incoming_batch=incoming_batch,
                        delivery_form=delivery_form,
                        mention_user_name=mention_user_name,
                        reference_user_name=reference_user_name,
                    ),
                    timeout=120.0,
                )
            except asyncio.TimeoutError:
                logger.error(f"{self.log_prefix} ⚠️ 轻量主动回复超时(120s)")
                return False

        except Exception as exc:
            logger.error(f"{self.log_prefix} 主动回复生成失败: {exc}")
            return False


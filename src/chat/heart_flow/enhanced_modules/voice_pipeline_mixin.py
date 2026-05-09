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

class EnhancedVoicePipelineMixin:
    async def _emit_outcome_summary(self, did_reply: bool, relation_snapshot: Dict) -> None:
        """第四层：结果汇总面板 - 回复后或跳过后的终值输出

        输出三核心值终值、好感/信任变化、保护态、屏蔽态。
        每次管线末端必须输出（替代旧版节流式最终状态）。
        """
        try:
            relation_view = self._resolve_relation_view(relation_snapshot)
            from src.core.world_snapshot import get_relation_number

            _d6 = EnergyChainDimension.get_instance()
            _d6_state = _d6._ensure_channel(self.stream_id)
            chat_current = _d6_state.chat_pool
            activity_current = _d6_state.activity_level
            social_current = _d6_state.social_value
            thinking_current = _d6_state.thinking_ratio() * 100.0

            outcome_parts = []
            action_tag = "已回复" if did_reply else "未回复"
            outcome_parts.append(f"结果={action_tag}")
            outcome_parts.append(f"聊天当前={chat_current:.1f}")
            outcome_parts.append(f"活跃当前={activity_current:.1f}")
            outcome_parts.append(f"社交当前={social_current:.1f}")
            outcome_parts.append(f"思考当前={thinking_current:.0f}%")

            # 关系变化标记
            rel_social = get_relation_number(
                relation_view,
                "social_value",
                aliases=("favorability",),
            )
            rel_trust = get_relation_number(
                relation_view,
                "trust_value",
                aliases=("trust_score",),
            )
            rel_annoy = get_relation_number(
                relation_view,
                "annoyance_value",
                aliases=("annoyance",),
            )
            if abs(rel_social) >= 0.1:
                outcome_parts.append(f"社交值={rel_social:.1f}")
            if abs(rel_trust) >= 0.1:
                outcome_parts.append(f"信赖度={rel_trust:.1f}")
            if abs(rel_annoy) >= 0.1:
                outcome_parts.append(f"厌烦度={rel_annoy:.1f}")

            rel_affection = get_relation_number(
                relation_view,
                "affection",
                aliases=("favorability",),
            )
            rel_pressure = get_relation_number(
                relation_view,
                "psychological_pressure",
            )
            rel_trauma = get_relation_number(relation_view, "trauma_score")
            if abs(rel_affection) >= 0.1:
                outcome_parts.append(f"好感={rel_affection:.1f}")
            if rel_pressure > 0 and not any(part.startswith("心理压力=") for part in outcome_parts):
                outcome_parts.append(f"心理压力={rel_pressure:.1f}")
            if rel_trauma > 0 and not any(part.startswith("创伤=") for part in outcome_parts):
                outcome_parts.append(f"创伤={rel_trauma:.1f}")
            if self._gateway_constraint_hits > 0:
                outcome_parts.append(f"硬约束次数={self._gateway_constraint_hits}")
            if self._last_gateway_penalty > 0:
                outcome_parts.append(f"硬约束惩罚={self._last_gateway_penalty:.1f}")
            quiet_left = max(0.0, self._planner_quiet_until - time.time())
            if quiet_left > 0:
                outcome_parts.append(f"冷却剩余={quiet_left:.0f}s")
            if self._last_gateway_reason:
                outcome_parts.append(f"硬约束因子={self._last_gateway_reason[:40]}")
            gateway_breakdown = getattr(self, "_last_gateway_breakdown", None)
            if did_reply and isinstance(gateway_breakdown, dict) and gateway_breakdown:
                outcome_parts.append(f"结算可回复度={gateway_breakdown.get('reply_readiness', 0):.1f}")
                outcome_parts.append(f"结算基础分={gateway_breakdown.get('base', 0):.1f}")
                outcome_parts.append(f"结算提及加分={gateway_breakdown.get('mention_bonus', 0):.1f}")
                outcome_parts.append(f"结算问句加分={gateway_breakdown.get('question_bonus', 0):.1f}")
                outcome_parts.append(f"结算显著性加分={gateway_breakdown.get('salience_bonus', 0):.1f}")
                outcome_parts.append(f"结算关系加分={gateway_breakdown.get('rapport_bonus', 0):.1f}")
                outcome_parts.append(f"结算资源惩罚={gateway_breakdown.get('resource_penalty', 0):.1f}")
                outcome_parts.append(f"结算关系惩罚={gateway_breakdown.get('relation_penalty', 0):.1f}")
                outcome_parts.append(f"结算重复惩罚={gateway_breakdown.get('repeat_penalty', 0):.1f}")
                outcome_parts.append(f"结算骚扰惩罚={gateway_breakdown.get('harass_penalty', 0):.1f}")
                outcome_parts.append(f"结算不信任惩罚={gateway_breakdown.get('distrust_penalty', 0):.1f}")

            # 保护态/屏蔽态
            target_uid = getattr(self, "_last_user_id", "")
            if target_uid:
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker,
                    )

                    tracker = get_emotion_tracker(self.stream_id)
                    emo_state = tracker.get_user_state(target_uid, create_if_missing=False)
                    if emo_state:
                        if getattr(emo_state, "is_blocked", False):
                            outcome_parts.append("状态=已屏蔽")
                        stamina = getattr(emo_state, "stamina", 100)
                        if stamina < 20:
                            outcome_parts.append(f"耐力低={stamina:.0f}")
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")

            # 创伤系统状态
            try:
                from src.modules.trauma.trauma_system import get_trauma_system

                trauma_sys = get_trauma_system()
                trauma_st = trauma_sys.get_state()
                if trauma_st.stress_accumulation > 5:
                    outcome_parts.append(f"压力蓄积={trauma_st.stress_accumulation:.1f}")
                if trauma_st.inner_chaos_level > 3:
                    outcome_parts.append(f"混乱={trauma_st.inner_chaos_level:.1f}")
                if trauma_st.surface_mask_strength > 0:
                    outcome_parts.append(f"伪装={trauma_st.surface_mask_strength:.1f}")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            # 用户维度详细心理指标（合并自旧版最终状态）
            if target_uid:
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker as _get_emo,
                    )

                    _trk = _get_emo(self.stream_id)
                    _es = _trk.get_user_state(target_uid, create_if_missing=False)
                    if _es:
                        _trauma = getattr(_es, "trauma_score", 0.0)
                        if _trauma > 0:
                            outcome_parts.append(f"创伤={_trauma:.1f}")
                        _pressure = getattr(_es, "psychological_pressure", 0.0)
                        if _pressure > 0:
                            outcome_parts.append(f"心理压力={_pressure:.1f}")
                        _ts = getattr(_es, "training_stage", 0)
                        if _ts > 0:
                            outcome_parts.append(f"调教阶段={_ts}")
                            _tp = getattr(_es, "training_progress", 0.0)
                            if _tp > 0:
                                outcome_parts.append(f"调教进度={_tp:.1f}%")
                        _sub = getattr(_es, "submission_level", 0.0)
                        if _sub > 0:
                            outcome_parts.append(f"顺从={_sub:.1f}")
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")

            # 频率调节因子
            try:
                freq_ctrl = frequency_control_manager.get_or_create_frequency_control(self.stream_id)
                _freq = freq_ctrl.get_talk_frequency_adjust()
                outcome_parts.append(f"频率={_freq:.2f}")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            logger.info(f"{self.log_prefix} 📊 " + " ".join(outcome_parts))
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 结果汇总输出异常: {exc}")

    def _compute_dynamic_threshold(self) -> int:
        """消息触发阈值，始终为1，任何新消息都应被处理"""
        return 1

    def _find_pinged_message(self, batch: List):
        """在消息批次中查找 @机器人 或提及机器人的消息"""
        if not global_config.chat.mentioned_bot_reply:
            return None
        for msg in batch:
            if (
                getattr(msg, "is_mentioned", False)
                or getattr(msg, "is_at", False)
                or getattr(msg, "mentioned_me", False)
            ):
                return msg
        return None

    def _get_sender_user_id(self, incoming_batch: List, pinged_msg=None) -> str:
        """提取当前消息批次发送者QQ号"""
        if pinged_msg is not None:
            _uid = getattr(pinged_msg, "user_id", "") or ""
            if self._is_human_message_obj(pinged_msg):
                return str(_uid).strip()
        latest_msg = self._get_latest_human_message(
            incoming_batch or [],
            preferred_sources=("incoming",),
            allow_fallback=True,
        )
        if latest_msg is None:
            return ""
        return str(getattr(latest_msg, "user_id", "") or "").strip()

    def _admin_force_wake_ids(self) -> set[str]:
        try:
            return {
                str(item).strip()
                for item in (global_config.chat.admin_force_wake_qq_ids or [])
                if str(item).strip()
            }
        except Exception:
            return set()

    def _is_message_from_force_wake_admin(self, msg) -> bool:
        """单条消息是否来自强制唤醒管理员。"""
        if msg is None or not self._is_human_message_obj(msg):
            return False
        uid = str(getattr(msg, "user_id", "") or "").strip()
        return bool(uid and uid in self._admin_force_wake_ids())

    def _is_force_wake_admin(self, incoming_batch: List, pinged_msg=None) -> bool:
        """判断当前批次是否包含强制唤醒管理员消息。"""
        if pinged_msg is not None and self._is_message_from_force_wake_admin(pinged_msg):
            return True
        return any(self._is_message_from_force_wake_admin(msg) for msg in (incoming_batch or []))

    def _is_priority_turn_message(self, msg) -> bool:
        """刷屏降载时必须保留的关键消息。"""
        if msg is None:
            return False
        if self._is_message_from_force_wake_admin(msg):
            return True
        if self._is_message_pinged(msg):
            return True
        if bool(getattr(msg, "is_command", False)):
            return True
        if getattr(getattr(self, "chat_stream", None), "group_info", None) is None:
            return self._is_human_message_obj(msg)
        return False

    def _is_message_pinged(self, msg) -> bool:
        """判断单条消息是否@了机器人"""
        return bool(
            getattr(msg, "is_mentioned", False)
            or getattr(msg, "is_at", False)
            or getattr(msg, "mentioned_me", False)
        )

    def _should_run_perception(self, now: float) -> bool:
        """感知引擎有冷却间隔，避免每轮都跑 LLM

        修复：DORMANT等级时跳过感知（除非冷却超长>30min强制刷新），
        避免在系统应该休息时仍消耗token跑LLM。
        """
        _cooldown_ok = (now - self._last_perception_ts) >= _PERCEPTION_COOLDOWN_SEC
        if not _cooldown_ok:
            return False
        _cached = getattr(self, "_cached_awareness", None)
        if _cached is not None:
            from src.chat.proactive.perception_engine import AlertGrade

            _grade = getattr(_cached, "alert_grade", None)
            if isinstance(_grade, AlertGrade):
                if _grade == AlertGrade.DORMANT:
                    _elapsed_min = (now - self._last_perception_ts) / 60.0
                    if _elapsed_min < 30.0:
                        return False
                elif _grade == AlertGrade.FAINT:
                    _elapsed_min = (now - self._last_perception_ts) / 60.0
                    if _elapsed_min < 10.0:
                        return False
        return True

    async def _invoke_perception(self, messages: List, now: float):
        """调用觉察引擎，收集环境感知快照"""
        try:
            from src.chat.proactive.perception_engine import (
                get_awareness_engine,
            )

            engine = get_awareness_engine()
            raw_msg_dicts = self._serialize_messages_for_perception(messages)
            self_state = self._collect_self_state()
            relation_state = self._collect_relation_state(messages)
            has_ping = any(getattr(m, "is_mentioned", False) or getattr(m, "is_at", False) for m in messages)
            verdict = await engine.gather_awareness(
                channel_id=self.stream_id,
                raw_messages=raw_msg_dicts,
                self_state=self_state,
                relation_state=relation_state,
                mention_bot=has_ping,
            )
            logger.debug(
                f"{self.log_prefix} 感知结果: "
                f"警戒={verdict.alert_grade.name} "
                f"参与度={verdict.engagement_pull:.2f} "
                f"群体={verdict.crowd_vibe.value}"
            )
            return verdict
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 感知引擎调用失败: {exc}")
            return None

    def _serialize_messages_for_perception(self, messages: List) -> List[Dict[str, Any]]:
        """将数据库消息对象序列化为感知引擎期望的字典列表"""
        result = []
        for msg in messages:
            result.append(
                {
                    "text": getattr(msg, "processed_plain_text", "") or "",
                    "user_id": getattr(msg, "user_id", "") or "",
                    "user_name": getattr(msg, "user_nickname", "") or "",
                    "timestamp": getattr(msg, "time", 0.0) or time.time(),
                }
            )
        return result

    def _collect_self_state(self) -> Dict[str, Any]:
        """收集机器人自身状态，供感知引擎参考"""
        state: Dict[str, Any] = {
            "channel_id": self.stream_id,
            "cycle_count": self._enhanced_round,
            "skip_ticks": self._consecutive_skip_ticks,
        }
        try:
            _d6c = EnergyChainDimension.get_instance()
            _d6sc = _d6c._ensure_channel(self.stream_id)
            state["chat_pool"] = _d6sc.chat_pool
            state["thinking_value"] = _d6sc.thinking_value
            state["annoyance"] = _d6sc.annoyance_level
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return state

    def _collect_relation_state(self, messages: List) -> Dict[str, Any]:
        """收集频道社交关系信息"""
        unique_speakers = set()
        for msg in messages:
            uid = getattr(msg, "user_id", "") or ""
            if uid:
                unique_speakers.add(uid)
        return {
            "participant_count": len(unique_speakers),
            "speaker_ids": list(unique_speakers),
        }

    def _sample_channel_ambient(self) -> Optional[Dict[str, Any]]:
        """采样频道氛围"""
        try:
            from src.chat.heart_flow.emotion_stream import (
                get_channel_mood_tracker,
            )

            tracker = get_channel_mood_tracker()
            ledger = tracker.fetch_mood(self.stream_id)
            return {
                "vitality": ledger.vitality,
                "weariness": ledger.weariness,
                "vexation": ledger.vexation,
                "category": (ledger.category.value if hasattr(ledger.category, "value") else str(ledger.category)),
            }
        except Exception as _exc:
            logger.warning(f"{self.log_prefix} 情绪账本构建异常: {_exc}")
            return None

    def _compute_llm_call_level(
        self,
        now: float,
        messages: List,
        pinged,
        desire_level: float = 5.0,
        cooldown_anchor_ts: Optional[float] = None,
    ) -> Tuple[int, float, float, bool]:
        """【自适应动态阈值算法】决定是否调用小模型/大模型

        设计原则：尽量减少小模型调用次数，省token

        使用46维因子向量的自适应学习器，根据历史反馈动态调整权重。

        返回值：
        - 0: 跳过小模型（直接观察/简短敷衍）
        - 1: 调用小模型（简单回复）
        - 2: 升级大模型（复杂逻辑）
        """
        _cooldown_anchor = self._last_voice_ts if cooldown_anchor_ts is None else float(cooldown_anchor_ts or 0.0)
        if (now - _cooldown_anchor) < _VOICE_COOLDOWN_SEC:
            return (0, 1.0, 0.0, False)

        try:
            from src.core.adaptive_threshold_learner_v2 import (
                get_adaptive_threshold_learner_v2,
            )

            _learner = get_adaptive_threshold_learner_v2(self.stream_id)
            _social = {
                "affection": float(getattr(self, "_cached_affection_value", 0.0) or 0.0),
                "trust_value": float(getattr(self, "_cached_trust_value", 0.0) or 0.0),
                "annoyance_value": float(getattr(self, "_cached_annoyance_value", 0.0) or 0.0),
                "pressure": float(getattr(self, "_cached_pressure_value", 0.0) or 0.0),
                "relationship_level": float(getattr(self, "_cached_relationship_level", 2) or 2),
                "group_support": float(getattr(self, "_cached_group_support", 0.0) or 0.0),
                "group_attack": float(getattr(self, "_cached_group_attack", 0.0) or 0.0),
                "familiarity": float(getattr(self, "_cached_familiarity", 0.0) or 0.0),
                "dominance": float(getattr(self, "_cached_dominance", 0.0) or 0.0),
            }
            _energy = {}
            if self._cached_metabolism_state:
                _energy["energy_ratio"] = float(getattr(self._cached_metabolism_state, "energy_ratio", 0.8) or 0.8)
                _energy["thinking_ratio"] = float(getattr(self._cached_metabolism_state, "thinking_ratio", 0.8) or 0.8)
                _energy["mood_valence"] = float(getattr(self._cached_metabolism_state, "mood_valence", 0.5) or 0.5)
                _energy["mood_arousal"] = float(getattr(self._cached_metabolism_state, "mood_arousal", 0.5) or 0.5)
                _energy["stress_accumulation"] = float(
                    getattr(
                        self._cached_metabolism_state,
                        "stress_accumulation",
                        0.0,
                    )
                    or 0.0
                )
                _energy["trauma_score"] = float(getattr(self._cached_metabolism_state, "trauma_score", 0.0) or 0.0)
            else:
                _energy = {
                    "energy_ratio": 0.8,
                    "thinking_ratio": 0.8,
                    "mood_valence": 0.5,
                    "mood_arousal": 0.5,
                    "stress_accumulation": 0.0,
                    "trauma_score": 0.0,
                }
            _group = {
                "pattern_confidence": 0.0,
                "pattern_type": 0.0,
                "active_users": 0.0,
                "message_velocity": 0.0,
                "topic_stability": 1.0,
                "atmosphere_tension": 0.0,
                "me_mentioned_ratio": 0.0,
                "excitement_level": 0.0,
            }
            try:
                if self._cached_pattern_evidence and len(self._cached_pattern_evidence) > 0:
                    _top_ev = self._cached_pattern_evidence[0]
                    _group["pattern_confidence"] = float(getattr(_top_ev, "confidence", 0.0) or 0.0)
                    _pat = getattr(_top_ev, "pattern", None)
                    if _pat and hasattr(_pat, "value"):
                        _group["pattern_type"] = float(hash(str(_pat.value)) % 11)
                    _group["active_users"] = float(len(getattr(_top_ev, "involved_users", set()) or set()))
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            try:
                if self._cached_scene_snapshot:
                    _atmo = getattr(self._cached_scene_snapshot, "atmosphere", None)
                    if _atmo and hasattr(_atmo, "value"):
                        _atmo_str = str(_atmo.value)
                        if "argument" in _atmo_str or "confrontation" in _atmo_str:
                            _group["atmosphere_tension"] = 0.8
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            _time_m = {
                "recency": 1.0,
                "conversation_length": float(len(messages) if messages else 0),
                "since_last_reply": (now - _cooldown_anchor) / 60.0,
            }
            # 计算刷屏强度
            _, _, _burst_intensity = _learner._detect_burst_mode(now)
            # 提取消息情绪（取文本关键词的简单估计）
            _msg_emotion = 0.0
            if messages:
                _latest_texts = [getattr(m, "processed_plain_text", "") or "" for m in messages[-3:]]
                _msg_emotion = _learner._calc_emotion(_latest_texts)
            # 调用综合决策
            _decision_str, _skip_adj, _willingness = _learner.compute_integrated_decision(
                energy=_energy.get("energy_ratio", 0.8),
                affection=_social.get("affection", 0.0),
                trust=_social.get("trust_value", 0.0),
                burst_intensity=_burst_intensity,
                msg_emotion=_msg_emotion,
                desire_level=desire_level,
                current_time=now,
            )
            # 映射决策字符串到级别: skip→0, join→1, llm_assist→2
            _decision_to_level = {"skip": 0, "join": 1, "llm_assist": 2}
            _level = _decision_to_level.get(_decision_str, 1)
            _tentative = False
            if _level == 0 and hasattr(_learner, "_should_tentative_speak"):
                _since_bot_hours = (now - _cooldown_anchor) / 3600.0
                _tentative = _learner._should_tentative_speak(
                    level=_level,
                    skip_score=_skip_adj,
                    social=_social,
                    energy=_energy.get("energy_ratio", 0.8),
                    since_bot_hours=_since_bot_hours,
                )
            if _tentative:
                logger.info(f"{self.log_prefix} 🎯 试探性发牢骚：沉默够久但害羞指数低，轻量试探")
            logger.debug(
                f"{self.log_prefix} [自适应阈值] decision={_decision_str} willingness={_willingness:.3f} → level={_level}"
            )
            return (_level, _skip_adj, _willingness, _tentative)
        except Exception as _e:
            logger.debug(f"{self.log_prefix} [自适应阈值] 回退简单逻辑: {_e}")
        if pinged is not None:
            return (1, 0.0, 0.3, False)
        if (now - _cooldown_anchor) < _VOICE_COOLDOWN_SEC:
            return (0, 1.0, 0.0, False)
        _skip_score = 0.0
        _escalate_score = 0.0
        _msg_count = len(messages) if messages else 0
        try:
            if self._cached_metabolism_state:
                _energy_ratio = float(getattr(self._cached_metabolism_state, "energy_ratio", 0.8) or 0.8)
                _thinking_ratio = float(getattr(self._cached_metabolism_state, "thinking_ratio", 0.8) or 0.8)
                if _energy_ratio < 0.3:
                    _skip_score += 0.4
                elif _energy_ratio < 0.5:
                    _skip_score += 0.2
                if _thinking_ratio < 0.3:
                    _escalate_score -= 0.3
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            _hour = time.localtime().tm_hour
            if _hour >= 23 or _hour < 6:
                _skip_score += 0.25
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            if self._cached_pattern_evidence and len(self._cached_pattern_evidence) > 0:
                for _ev in self._cached_pattern_evidence[:3]:
                    _conf = float(getattr(_ev, "confidence", 0.0) or 0.0)
                    _pat = getattr(_ev, "pattern", None)
                    if _pat and hasattr(_pat, "value"):
                        _pat_val = str(_pat.value)
                        _user_count = len(getattr(_ev, "involved_users", set()) or set())
                        if _pat_val in (
                            "copycat_chain",
                            "spectator_mode",
                            "quiet_reflection",
                        ):
                            _skip_score += _conf * 0.5
                        elif _pat_val in (
                            "pile_on",
                            "conflict_escalation",
                            "meme_storm",
                        ):
                            _escalate_score += _conf * 0.3 * min(1.0, _user_count / 3.0)
                        elif _pat_val in (
                            "gossip_event",
                            "emotional_contagion",
                        ):
                            _escalate_score += _conf * 0.2
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            if self._cached_scene_snapshot:
                _joinable = bool(getattr(self._cached_scene_snapshot, "suitable_to_join", True))
                if not _joinable:
                    _skip_score += 0.6
                _atmo = getattr(self._cached_scene_snapshot, "atmosphere", None)
                if _atmo and hasattr(_atmo, "value"):
                    _atmo_val = str(_atmo.value)
                    if _atmo_val in ("argument", "confrontation"):
                        _escalate_score += 0.25
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        if _msg_count >= 10:
            _escalate_score += 0.15
        elif _msg_count >= 5:
            _escalate_score += 0.08
        try:
            if self._cached_presence_state:
                _quiet = float(getattr(self._cached_presence_state, "quiet_preference", 0.0) or 0.0)
                if _quiet > 0.6:
                    _skip_score += _quiet * 0.3
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        _net_score = _skip_score - _escalate_score
        logger.debug(
            f"{self.log_prefix} [简单阈值] skip={_skip_score:.2f} esc={_escalate_score:.2f} "
            f"net={_net_score:.2f} → level={
                '跳过' if _net_score >= 0.5 else '小模型' if _net_score >= -0.3 else '大模型'
            }"
        )
        if _net_score >= 0.5:
            return (0, _skip_score, _escalate_score, False)
        if _net_score <= -0.3:
            return (2, _skip_score, _escalate_score, False)
        return (1, _skip_score, _escalate_score, False)

    def _should_run_voice(
        self,
        now: float,
        messages: List,
        pinged,
        is_proactive: bool = False,
        source: str = "",
        behavior_verdict: Optional[BehaviorGovernorVerdict] = None,
    ) -> bool:
        """系统算法 + 冷却综合判断：是否调用小模型（内心独白）

        设计原则：尽量减少小模型调用次数
        is_proactive: 主动路径使用独立冷却计时器，不受被动路径影响
        """
        source_label = source or ("proactive" if is_proactive else "reactive")
        peek_source = "peek" in str(source_label or "").strip().lower()
        cooldown_anchor = self._last_proactive_voice_ts if is_proactive else self._last_voice_ts
        _level, _, _, _ = self._compute_llm_call_level(
            now,
            messages,
            pinged,
            cooldown_anchor_ts=cooldown_anchor,
        )
        _behavior = behavior_verdict
        if _behavior is None:
            try:
                _behavior = self._evaluate_behavior_governor(
                    incoming_batch=list(messages or []),
                    silence_sec=max(
                        0.0,
                        now - float(getattr(self, "_last_user_msg_time", 0.0) or 0.0),
                    ),
                    requested_mode="proactive" if is_proactive or "proactive" in source_label else "reactive",
                    target_message=self._get_latest_human_message(messages) if messages else None,
                    is_background=bool("background" in source_label),
                )
            except Exception as exc:
                logger.debug(f"{self.log_prefix} 行为Governor评估异常(voice): {exc}")
                _behavior = None
        if (
            peek_source
            and _level > 0
            and _behavior is not None
            and not bool(getattr(_behavior, "allow_generation", True))
            and str(getattr(_behavior, "reply_mode", "") or "").strip().lower() == "observe"
        ):
            # 窥屏态的内心独白是“先产生主观反应，再决定要不要升级参与”，
            # 不应被外部回复门控提前掐断。
            _behavior = dataclass_replace(
                _behavior,
                allow_generation=True,
                model_tier="small",
            )
            logger.debug(f"{self.log_prefix} 👁 窥屏内心独白放行：先形成主观反应，再决定是否升级参与")
        _model_governor = self._evaluate_model_governor(
            now=now,
            desired_level=_level,
            incoming_batch=list(messages or []),
            pinged_msg=pinged,
            is_proactive=is_proactive,
            source=source_label,
            behavior_verdict=_behavior,
        )
        if _level == 0:
            logger.info(f"{self.log_prefix} 动态阈值判定: {source_label} 跳过小模型(level=0)")
            return False
        if _model_governor.tier == "skip":
            logger.info(
                f"{self.log_prefix} 🎛️ 模型Governor拦截({source_label}): "
                f"{self._summarize_model_governor(_model_governor)}"
            )
            return False
        if _model_governor.tier == "large" and (is_proactive or "proactive" in source_label):
            self._record_large_model_usage(now=now, is_proactive=True)
            logger.info(
                f"{self.log_prefix} 🎛️ 模型Governor允许大模型({source_label}): "
                f"{self._summarize_model_governor(_model_governor)}"
            )
        if _model_governor.rate_limited:
            logger.info(
                f"{self.log_prefix} 🎛️ 模型Governor限流({source_label}): {self._summarize_model_governor(_model_governor)}"
            )
        return True

    def _should_upgrade_to_large_model(self, now: float, messages: List, pinged) -> bool:
        """判断是否需要升级到大模型"""
        _level, _, _, _ = self._compute_llm_call_level(now, messages, pinged)
        _behavior = self._evaluate_behavior_governor(
            incoming_batch=list(messages or []),
            silence_sec=max(
                0.0,
                now - float(getattr(self, "_last_user_msg_time", 0.0) or 0.0),
            ),
            requested_mode="reactive",
            target_message=self._get_latest_human_message(messages) if messages else None,
            is_background=False,
        )
        _model = self._evaluate_model_governor(
            now=now,
            desired_level=_level,
            incoming_batch=list(messages or []),
            pinged_msg=pinged,
            is_proactive=False,
            source="reactive",
            behavior_verdict=_behavior,
        )
        return _model.tier == "large"

    async def _invoke_inner_voice(
        self,
        messages: List,
        awareness,
        ambient: Optional[Dict[str, Any]],
        now: float,
    ):
        """调用自我对话引擎，生成内心想法；同时构建统一 WorldSnapshot 并缓存供后续阶段复用"""
        try:
            from src.chat.heart_flow.inner_voice import (
                get_self_dialogue_engine,
            )

            engine = get_self_dialogue_engine(self.stream_id)
            # 取最新一条消息作为触发文本
            latest = messages[-1] if messages else None
            raw_text = getattr(latest, "processed_plain_text", "") if latest else ""
            speaker_name = getattr(latest, "user_nickname", "") or "" if latest else ""
            speaker_id = getattr(latest, "user_id", "") or "" if latest else ""
            # 复用已有世界快照或补建（避免重复采集）
            tick_snapshot = getattr(self, "_tick_world_snapshot", None)
            if tick_snapshot is None or not tick_snapshot.target_user.user_id:
                try:
                    from src.core.world_snapshot import build_world_snapshot

                    tick_snapshot = await build_world_snapshot(self.stream_id, speaker_id)
                    self._tick_world_snapshot = tick_snapshot
                except Exception as _snap_err:
                    logger.debug(f"{self.log_prefix} 统一快照构建失败，退回独立采集: {_snap_err}")
            # 从能量和氛围提取参数
            energy_params = self._extract_voice_energy_params()
            mood_params = self._extract_voice_mood_params(ambient)
            # 构建对话历史片段
            dialogue_fragment = self._build_dialogue_fragment(messages)
            relation_view = self._resolve_relation_view()
            # 刷新快照中滞后的实时情绪值（重复回写发生在快照构建后）
            if tick_snapshot is not None and relation_view:
                _tu = tick_snapshot.target_user
                _rv_ann = float(relation_view.get("annoyance_value", 0.0) or 0.0)
                _rv_prs = float(relation_view.get("psychological_pressure", 0.0) or 0.0)
                if _rv_ann > _tu.annoyance_value:
                    _tu.annoyance_value = _rv_ann
                if _rv_prs > _tu.psychological_pressure:
                    _tu.psychological_pressure = _rv_prs
            voice_context = self._build_inner_voice_context(
                latest,
                ambient,
                relation_snapshot=relation_view,
                awareness_snapshot=awareness,
            )
            phase_label = self._query_flow_phase()
            verdict = await engine.generate_reflection(
                heart_state_label=phase_label,
                raw_text=raw_text,
                speaker_name=speaker_name,
                dialogue_history=dialogue_fragment,
                assurance_score=energy_params.get("assurance", 0.5),
                involvement_score=energy_params.get("involvement", 0.5),
                mental_drain=energy_params.get("drain", 0.0),
                endurance=energy_params.get("endurance", 100.0),
                irritation=mood_params.get("irritation", 0.0),
                wound_score=mood_params.get("wound", 0.0),
                readiness=energy_params.get("readiness", 1.0),
                speaker_id=speaker_id,
                world_snapshot=tick_snapshot,
                extra_context=voice_context,
                is_admin=self._is_force_wake_admin(self._get_incoming_batch_from_context() or []),
            )
            if verdict.is_valid:
                logger.debug(
                    f"{self.log_prefix} 内心独白: "
                    f"欲望={verdict.reply_desire_level} "
                    f"情绪={verdict.current_mood} "
                    f"意图数={len(verdict.intents)}"
                )
            return verdict
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 内心独白调用失败: {exc}")
            return None

    async def _invoke_autonomous_voice(self, trigger_reason: str):
        """无新消息时走自主思考入口，避免把空消息伪装成对话触发。"""
        recent_reply_burst = sum(1 for t in self._bot_reply_timeline if time.time() - t < 180.0)
        if recent_reply_burst >= 4 or float(getattr(self, "_chatterbox_penalty", 0.0) or 0.0) >= 2.0:
            logger.info(f"{self.log_prefix} 🧠 自主独白暂停：短时间已说太多({recent_reply_burst}/180s)")
            return None
        try:
            from src.chat.heart_flow.inner_voice import get_self_dialogue_engine

            verdict = await get_self_dialogue_engine(self.stream_id).autonomous_reflection(
                self.stream_id,
                trigger_reason=trigger_reason[:100],
            )
            if verdict and getattr(verdict, "is_valid", False):
                self._cached_voice = verdict
                return verdict
            return None
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 自主独白调用失败: {exc}")
            return None

    def _extract_voice_energy_params(self) -> Dict[str, float]:
        """从能量快照提取供内心独白使用的参数"""
        result = {
            "assurance": 0.5,
            "involvement": 0.5,
            "drain": 0.0,
            "endurance": 100.0,
            "readiness": 1.0,
        }
        try:
            _d6v = EnergyChainDimension.get_instance()
            _d6sv = _d6v._ensure_channel(self.stream_id)
            _d6snap = _d6v.capture_snapshot(self.stream_id)
            chat_value = _d6sv.chat_pool
            activity_level = _d6sv.activity_level
            social_value = _d6sv.social_value
            chat_factor = max(0.0, min(1.0, chat_value / max(_d6snap.chat_ceiling, 1.0)))
            thinking_factor = _d6snap.thinking_ratio()
            activity_factor = max(0.0, min(1.0, activity_level / 100.0))
            social_factor = max(0.0, min(1.0, (social_value + 100.0) / 200.0))
            result["assurance"] = chat_factor
            result["involvement"] = thinking_factor
            result["drain"] = _d6snap.annoyance_level
            result["endurance"] = max(0.0, 100.0 - _d6snap.annoyance_level)
            _raw_readiness = max(
                0.05,
                min(
                    1.0,
                    thinking_factor * 0.4 + chat_factor * 0.25 + activity_factor * 0.2 + social_factor * 0.15,
                ),
            )
            if getattr(self, "_peek_mode_active", False):
                _raw_readiness = max(0.05, _raw_readiness * 0.65)
            if getattr(self, "_pattern_forced_observe", False):
                _raw_readiness = max(0.05, _raw_readiness * 0.55)
            if getattr(self, "_defense_mode_active", False):
                _raw_readiness = max(0.05, _raw_readiness * 0.78)
            try:
                from src.chat.heart_flow.emotion_driven_core import (
                    get_emotion_driven_core,
                )

                _emo_snap = get_emotion_driven_core().get_state_snapshot(self.stream_id)
                _boredom_val = float(_emo_snap.get("boredom", 0.0) or 0.0)
                thresholds = get_heartfc_thresholds()
                if _boredom_val > thresholds.boredom_reply_suppress_high:
                    _raw_readiness = max(
                        0.05,
                        _raw_readiness * (1.0 - (_boredom_val - thresholds.boredom_reply_suppress_high) * 0.5),
                    )
                elif _boredom_val > thresholds.boredom_reply_suppress_medium:
                    _raw_readiness = max(0.05, _raw_readiness * 0.9)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            try:
                _meta = getattr(self, "_cached_metabolism_state", None)
                if _meta is not None:
                    _sm = float(getattr(_meta, "surface_mask", 10.0) or 10.0)
                    _ic = float(getattr(_meta, "inner_chaos", 0.0) or 0.0)
                    _sub = float(getattr(_meta, "submission_level", 0.0) or 0.0)
                    if _ic > 5.0:
                        _raw_readiness = max(0.05, _raw_readiness * max(0.3, 1.0 - _ic * 0.05))
                    if _sub > 3.0:
                        _raw_readiness = max(
                            0.05,
                            _raw_readiness * (1.0 - min(0.4, _sub * 0.08)),
                        )
                    if _sm < 4.0:
                        _raw_readiness = min(1.0, _raw_readiness * (1.0 + (4.0 - _sm) * 0.08))
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            try:
                from src.modules.trauma.trauma_system import get_trauma_system

                _ts = get_trauma_system()
                _tstate = _ts.get_state()
                _t_enum = getattr(_tstate, "current_state", None)
                if _t_enum is not None:
                    _tname = str(_t_enum.value) if hasattr(_t_enum, "value") else str(_t_enum)
                    if "崩溃" in _tname or "即将" in _tname:
                        _raw_readiness = max(0.03, _raw_readiness * 0.2)
                    elif "半崩" in _tname or "勉强" in _tname or "动摇" in _tname:
                        _raw_readiness = max(0.05, _raw_readiness * 0.5)
                    elif "滑落" in _tname:
                        _raw_readiness = max(0.05, _raw_readiness * 0.75)
                _stress = float(getattr(_tstate, "stress_accumulation", 0.0) or 0.0)
                if _stress > 15.0:
                    _raw_readiness = max(
                        0.03,
                        _raw_readiness * (1.0 - min(0.5, (_stress - 15.0) * 0.02)),
                    )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            result["readiness"] = _raw_readiness
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return result

    def _extract_voice_mood_params(self, ambient: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """从氛围采样提取供内心独白使用的参数"""
        result = {"irritation": 0.0, "wound": 0.0}
        if ambient is None:
            return result
        result["irritation"] = float(ambient.get("vexation", 0.0))
        result["wound"] = float(ambient.get("weariness", 0.0)) / 100.0
        return result

    def _build_dialogue_fragment(self, messages: List) -> List[Dict]:
        """将最近消息构建为内心独白使用的对话片段格式"""
        fragment = []
        for msg in messages[-8:]:
            entry: Dict[str, Any] = {
                "role": "user",
                "content": getattr(msg, "processed_plain_text", "") or "",
                "name": getattr(msg, "user_nickname", "") or "",
            }
            fragment.append(entry)
        return fragment

    def _build_inner_voice_context(
        self,
        latest: Optional[Any],
        ambient: Optional[Dict[str, Any]],
        relation_snapshot: Optional[Dict[str, Any]] = None,
        awareness_snapshot: Optional[Any] = None,
    ) -> Dict[str, str]:
        """为内心独白整理目标画像、记忆摘要和后续执行提示。"""
        target_uid = str(getattr(latest, "user_id", "") or "").strip() if latest else ""
        latest_text = (
            str(getattr(latest, "processed_plain_text", "") or getattr(latest, "content", "") or "").strip()
            if latest
            else ""
        )
        latest_repeat_short = bool(getattr(latest, "is_repeated_short_input", False)) if latest else False
        latest_continuation_hint = str(getattr(latest, "context_continuation_hint", "") or "").strip() if latest else ""
        profile_bits: List[str] = []
        memory_bits: List[str] = []
        execution_bits: List[str] = []

        relation_snapshot = self._resolve_relation_view(relation_snapshot)
        if relation_snapshot:
            custom_label = str(relation_snapshot.get("custom_label", "") or "")
            relation_level = relation_snapshot.get("relationship_level", None)
            social_value = float(relation_snapshot.get("social_value", 0.0) or 0.0)
            trust_value = float(relation_snapshot.get("trust_value", 0.0) or 0.0)
            annoyance_value = float(relation_snapshot.get("annoyance_value", 0.0) or 0.0)
            mood = str(relation_snapshot.get("mood", "") or "")
            if custom_label:
                profile_bits.append(f"关系标签={custom_label}")
            if relation_level is not None:
                profile_bits.append(f"关系级别={relation_level}")
            profile_bits.append(f"社交={social_value:.1f}")
            if self._metric_has_signal(trust_value):
                profile_bits.append(f"信任={trust_value:.1f}")
            if self._metric_has_signal(annoyance_value):
                profile_bits.append(f"烦躁={annoyance_value:.1f}")
            if mood:
                profile_bits.append(f"当前印象气氛={mood}")
            attr_influences = relation_snapshot.get("attribute_influences") or {}
            if isinstance(attr_influences, dict) and attr_influences:
                top_attrs = []
                for _key, _val in list(attr_influences.items())[:4]:
                    try:
                        top_attrs.append(f"{_key}={float(_val):.2f}")
                    except Exception:
                        continue
                if top_attrs:
                    execution_bits.append("当前属性偏置=" + ", ".join(top_attrs))

        if target_uid:
            try:
                from src.person_info.person_info import get_unified_profile_hub

                hub = get_unified_profile_hub()
                relationships = hub.get_top_relationships(target_uid, limit=3)
                if relationships:
                    rel_text = "、".join(
                        f"{str(uid)[:8]}:{float(weight):.2f}" for uid, weight in relationships if str(uid or "").strip()
                    )
                    if rel_text:
                        profile_bits.append(f"关联对象={rel_text}")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            try:
                from src.chat.proactive.session_tracker import (
                    get_memoir_cabinet,
                )

                memoir = get_memoir_cabinet().retrieve(target_uid)
                if memoir is not None:
                    phase = getattr(memoir, "phase", None)
                    if phase is not None:
                        memory_bits.append(f"会话阶段={getattr(phase, 'value', phase)}")
                    last_topic = str(getattr(memoir, "last_topic", "") or "")
                    if last_topic:
                        memory_bits.append(f"上次话题={last_topic}")
                    last_mood = str(getattr(memoir, "last_mood", "") or "")
                    if last_mood:
                        memory_bits.append(f"上次气氛={last_mood}")
                    if latest_text and last_topic and self._normalize_repeat_text(latest_text):
                        memory_bits.append("这句大概率是在延续之前的话头，不是全新开题")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

        if latest_text:
            normalized_latest = self._normalize_repeat_text(latest_text)
            repeat_count = self._count_recent_user_repeats(normalized_latest, target_uid)
            if repeat_count >= 2:
                memory_bits.append(f"对方近几轮已重复这句或近似说法{repeat_count}次")
                execution_bits.append("先按重复短句/可能玩梗/等待你接前情来理解，不要把它当首次出现的信息。")
                execution_bits.append(
                    "如果信息不够，就直接判断成没说清楚、在复读、在等你接前情，不要硬猜对方具体在聊什么。"
                )
        if latest_repeat_short:
            memory_bits.append("入口层已标记：这是重复短句/前情延续，不应按新话题理解")
            if latest_continuation_hint:
                execution_bits.append(latest_continuation_hint)
            execution_bits.append("这类短句允许你心里直接觉得看不懂、懒得猜、像在刷同一句，不需要替对方补完整话题。")

        self_memory = self._build_self_reply_memory()
        if self_memory:
            memory_bits.append(self_memory.replace("[你刚刚自己说过的话]", "你刚刚说过"))

        cached_memory_hint = str(getattr(self, "_latest_memory_hint", "") or "").strip()
        if cached_memory_hint:
            memory_bits.append(cached_memory_hint[:220])

        persona_hint = self._build_persona_hint()
        if persona_hint:
            profile_bits.append(persona_hint.replace("\n", "；"))

        if self._llm_content_plan:
            execution_bits.append(f"内容规划={self._llm_content_plan[:120]}")
        if getattr(self, "_llm_decision_reason", ""):
            execution_bits.append(f"决策理由={str(self._llm_decision_reason)[:120]}")
        if ambient is not None:
            arousal = float(ambient.get("arousal", 0.0) or 0.0)
            valence = float(ambient.get("valence", 0.0) or 0.0)
            execution_bits.append(f"环境唤醒={arousal:.2f}, 环境正负={valence:.2f}")
        if awareness_snapshot is not None:
            try:
                _alert = getattr(awareness_snapshot, "alert_grade", None)
                _crowd = getattr(awareness_snapshot, "crowd_vibe", None)
                _engage = float(getattr(awareness_snapshot, "engagement_pull", 0.0) or 0.0)
                _observe = float(getattr(awareness_snapshot, "observability", 0.0) or 0.0)
                _awareness_bits = []
                if _alert is not None:
                    _awareness_bits.append(f"警戒={_alert.name if hasattr(_alert, 'name') else _alert}")
                if _crowd is not None:
                    _awareness_bits.append(f"群体={_crowd.value if hasattr(_crowd, 'value') else _crowd}")
                _awareness_bits.append(f"参与牵引={_engage:.2f}")
                _awareness_bits.append(f"可观察性={_observe:.2f}")
                execution_bits.append("感知摘要=" + ", ".join(_awareness_bits))
            except Exception as _e:
                logger.debug(f"{self.log_prefix} 感知摘要注入失败: {_e}")
        execution_bits.append("独白要像人心里自然冒出的想法，可以有联想、回忆、吐槽，每次都要不同。")

        # 实时关系数值（用于覆盖旧快照中的滞后值）
        _rt_annoyance = ""
        _rt_pressure = ""
        _rt_blocked = ""
        if relation_snapshot:
            _rt_annoyance = str(relation_snapshot.get("annoyance_value", ""))
            _rt_pressure = str(relation_snapshot.get("psychological_pressure", ""))
            _rt_blocked = str(relation_snapshot.get("is_user_blocked", ""))

        return {
            "profile_summary": "，".join(bit for bit in profile_bits if bit),
            "memory_summary": "，".join(bit for bit in memory_bits if bit),
            "execution_hint": "；".join(bit for bit in execution_bits if bit),
            "rt_annoyance": _rt_annoyance,
            "rt_pressure": _rt_pressure,
            "rt_blocked": _rt_blocked,
        }

    def _build_decision_context_packet(
        self,
        decision_messages: List,
        repetition_signal: Optional[Dict[str, Any]] = None,
    ):
        """把当前轮最关键的对象、续接与重复信号压成统一上下文。"""
        from src.common.data_models.heartflow_models import (
            DecisionContextPacket,
        )

        latest_user = self._get_latest_human_message(decision_messages)
        latest_bot = self._get_latest_bot_context_message(decision_messages)

        packet = DecisionContextPacket()
        if latest_user is not None:
            packet.target_user_id = str(getattr(latest_user, "user_id", "") or "").strip()
            packet.target_name = str(
                getattr(latest_user, "user_nickname", "")
                or getattr(latest_user, "nickname", "")
                or packet.target_user_id
            ).strip()
            packet.latest_user_text = str(
                getattr(latest_user, "processed_plain_text", "") or getattr(latest_user, "content", "") or ""
            ).strip()
            packet.continuation_hint = str(getattr(latest_user, "context_continuation_hint", "") or "").strip()
            packet.repeated_short_input = bool(getattr(latest_user, "is_repeated_short_input", False))

        if latest_bot is not None:
            packet.latest_bot_text = str(
                getattr(latest_bot, "processed_plain_text", "") or getattr(latest_bot, "content", "") or ""
            ).strip()

        if repetition_signal and repetition_signal.get("detected"):
            packet.repeat_reason = str(repetition_signal.get("reason", "") or "").strip()
            if repetition_signal.get("latest_matches_repeat"):
                packet.repeated_short_input = True

        return packet

    async def _prefetch_memory_hint(self, messages: List) -> str:
        """在独白/规划前做一层轻量记忆预检索，把结果前置给主链使用。"""
        try:
            from src.memory_system.memory_retrieval import (
                build_memory_retrieval_prompt,
            )
            from src.chat.utils.chat_message_builder import (
                build_readable_messages,
            )

            target_message = self._get_latest_human_message(messages)
            if target_message is None:
                self._latest_memory_hint = ""
                return ""

            target_text = self._extract_message_content(target_message).strip()
            if not target_text:
                self._latest_memory_hint = ""
                return ""

            history_block = build_readable_messages(
                list(messages[-12:]),
                replace_bot_name=True,
                timestamp_mode="relative",
                read_mark=0.0,
                show_actions=True,
            )
            sender = (
                str(
                    getattr(target_message, "user_nickname", "")
                    or getattr(target_message, "nickname", "")
                    or getattr(target_message, "user_id", "用户")
                ).strip()
                or "用户"
            )
            # F14：关系叙事→影响记忆检索优先级——高好感/高信任用户的记忆优先召回
            _rel_priority_hint = ""
            _rel_snap = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
            if _rel_snap:
                _aff = float(_rel_snap.get("affection", 0.0) or 0.0)
                _trust = float(_rel_snap.get("trust_value", 0.0) or 0.0)
                _has_aff_signal = self._metric_has_signal(_aff)
                _has_trust_signal = self._metric_has_signal(_trust)
                if _has_aff_signal and _has_trust_signal and _aff > 65 and _trust > 60:
                    _rel_priority_hint = (
                        f"[关系优先] {sender}是亲密好友(好感{_aff:.0f}/信任{_trust:.0f})，优先检索与TA相关的记忆。"
                    )
                elif _has_aff_signal and _has_trust_signal and _aff < 20 and _trust < 25:
                    _rel_priority_hint = (
                        f"[关系回避] {sender}关系疏远(好感{_aff:.0f}/信任{_trust:.0f})，仅检索冲突或重要事件记忆。"
                    )
            hint = await build_memory_retrieval_prompt(
                history_block,
                sender,
                target_text,
                self.chat_stream,
                think_level=1,
                unknown_words=None,
                relation_context=_rel_priority_hint,
            )
            compact_hint = str(hint or "").strip()
            if compact_hint.startswith("你回忆起了以下信息："):
                compact_hint = compact_hint.replace("你回忆起了以下信息：", "").strip()
            self._latest_memory_hint = compact_hint[:320]
            return self._latest_memory_hint
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 预记忆检索失败: {exc}")
            self._latest_memory_hint = ""
            return ""

    def _build_voice_execution_guard(self, voice_conclusion: Any, target_message: Optional[Any]) -> str:
        """把内心独白变成后续回复生成阶段必须遵守的执行约束。"""
        if voice_conclusion is None:
            return ""
        thinking = str(getattr(voice_conclusion, "thinking", "") or "").strip()
        mood = str(getattr(voice_conclusion, "current_mood", "") or "").strip()
        desire = int(getattr(voice_conclusion, "reply_desire_level", 0) or 0)
        if not thinking and not mood and desire <= 0:
            return ""

        target_name = "对方"
        if target_message is not None:
            target_name = (
                str(
                    getattr(target_message, "user_nickname", "") or getattr(target_message, "user_id", "") or "对方"
                ).strip()
                or "对方"
            )

        lines = ["[内心独白执行约束]"]
        _hour_now = datetime.datetime.now().hour
        _is_deep = (0 <= _hour_now < 6) or (_hour_now >= 23)
        _is_early = 6 <= _hour_now < 10
        if _is_deep:
            lines.append(f"[⏰⚠️时间强制] 现在是凌晨{_hour_now}点！你必须在回复中体现这一点！")
            try:
                from src.chat.prompts.soul_config_loader import (
                    get_sarcasm_templates as _get_sc,
                )

                _sc_high = _get_sc("high")
                _sc_mid = _get_sc("mid")
                import random as _rnd

                _sc_pick = _rnd.sample(_sc_high + _sc_mid, min(4, len(_sc_high + _sc_mid)))
                lines.append(f"可选语气参考: {' | '.join(_sc_pick)} — 像真人凌晨被吵醒会说的那种话")
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
            lines.append("  · 回复必须简短(3-15字)，不要完整句子")
            lines.append("  · 必须带困倦/烦躁/不耐烦的底色")
            lines.append("  · 可以用省略号/单个字/不完整表达")
            lines.append("  · 绝对禁止: 热情/详细解释/标准标点/客服腔/礼貌用语")
            lines.append("  · 如果对方一直在刷屏，你可以怼回去或阴阳怪气")
        elif _is_early:
            lines.append(f"[⏰时间] 现在是早上{_hour_now}点左右，刚醒不久。回复可以带点慵懒感。")
        else:
            lines.append(f"[⏰时间] 现在是{_hour_now}点。正常状态回复即可。")
        if thinking:
            lines.append(f"- 你刚才心里真正想到的是：{thinking}")
        if mood:
            lines.append(f"- 当前情绪底色：{mood}")
        if desire > 0:
            lines.append(f"- 当前回复欲望：{desire}/10")
        _sw = getattr(self, "_night_soft_wake_info", None)
        if _sw and isinstance(_sw, dict):
            _sw_act = _sw.get("action", "")
            _sw_des = _sw.get("desire", 0)
            _hour = datetime.datetime.now().hour
            if _sw_act == "soft_wake":
                lines.append(f"[⏰深夜状态] 现在是凌晨{_hour}点，你很困。回复必须体现隐晦疲惫感:")
                lines.append("  · 极短(3-15字)，不要完整句子")
                lines.append("  · 可带省略号/单个字/重复字(嗯/呵/..)表达困意")
                lines.append("  · 不要解释、不要展开、不要问问题")
                lines.append("  · 不要用标准标点，可以无标点或用不规范的断句")
                lines.append("  · 像一个真的被吵醒的人会说的那种话，不是机器人")
            elif _sw_act == "grumpy_glance":
                lines.append("[⏰深夜烦躁] 凌晨被吵醒很不爽:")
                lines.append("  · 极短冷淡(2-10字)，带不耐烦")
                lines.append("  · 冷漠敷衍，不要友好")
                lines.append("  · 不要标准标点和完整句子")
        lines.append(f"- 现在回复{target_name}时，语气和内容必须顺着这句心里话往下走，不能另起一套分析。")
        lines.append("- 不要把内心独白原封不动复述给对方，而是把它转成自然回复动机。")
        lines.append("- 如果内心独白偏冷淡、试探、烦躁或敷衍，回复也必须保留这种底色。")
        lines.append("- 如果内心独白是在判断对方是不是玩梗、刷屏、试探你，回复要先贴着这个判断接，不要跑题。")
        return "\n".join(lines)

    def _resolve_relation_view(self, relation_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """统一解析关系视图，优先使用显式传入的数据，回退到最近一次缓存。

        关键修复：在返回前从emotion_tracker注入实时annoyance和pressure值，
        解决快照数据(annoyance=0)与实时状态(annoyance=100)不一致的问题。
        """
        if relation_snapshot:
            return self._inject_realtime_emotion(self._normalize_relation_snapshot(relation_snapshot))
        tick_snapshot = getattr(self, "_tick_world_snapshot", None)
        if tick_snapshot is not None:
            try:
                return self._inject_realtime_emotion(self._normalize_relation_snapshot(tick_snapshot.to_relation_dict()))
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        return self._inject_realtime_emotion(
            self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", {}) or {})
        )

    def _inject_realtime_emotion(self, base_view: Dict[str, Any]) -> Dict[str, Any]:
        """从emotion_tracker读取实时annoyance/pressure覆盖到关系视图。

        查找优先级：
        1. 本轮缓存状态（_resolved_emo_state，由情绪追踪器同步阶段写入）
        2. 按_last_user_id实时查询（兼容无缓存场景）
        """
        _state_to_use = None
        if self._resolved_emo_state is not None:
            _state_to_use = self._resolved_emo_state
        else:
            target_uid = str(getattr(self, "_last_user_id", "") or "").strip()
            if target_uid:
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker,
                    )

                    tracker = get_emotion_tracker(self.stream_id)
                    _state_to_use = tracker.get_user_state(target_uid, create_if_missing=False)
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
        base_view = self._normalize_relation_snapshot(base_view)
        if _state_to_use is None:
            return dict(base_view)
        try:
            rt_annoyance = float(
                getattr(
                    _state_to_use,
                    "annoyance_value",
                    getattr(_state_to_use, "annoyance", -1),
                )
                or -1
            )
            if rt_annoyance >= 0:
                base_view["annoyance_value"] = round(rt_annoyance, 1)
            rt_pressure = float(getattr(_state_to_use, "psychological_pressure", -1) or -1)
            if rt_pressure >= 0:
                base_view["psychological_pressure"] = round(rt_pressure, 1)
            rt_blocked = bool(getattr(_state_to_use, "is_blocked", False) or False)
            base_view["is_user_blocked"] = rt_blocked
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return self._normalize_relation_snapshot(base_view)

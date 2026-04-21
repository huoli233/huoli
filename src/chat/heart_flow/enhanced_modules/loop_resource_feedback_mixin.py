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

class LoopResourceFeedbackMixin:
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


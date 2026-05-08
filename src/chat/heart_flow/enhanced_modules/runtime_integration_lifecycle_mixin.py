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

class RuntimeIntegrationLifecycleMixin:
    def _integrate_scene_tracking(self, incoming_batch: List) -> None:
        """群场景追踪 + 群体模式识别"""
        try:
            _gs = self._orch.get("scene_state")
            if self._cached_participant_summary:
                _known_active = int(self._cached_participant_summary.get("hot_count", 0) or 0) + int(
                    self._cached_participant_summary.get("warm_count", 0) or 0
                )
                _gs.set_known_online_count(_known_active)
            for _m in incoming_batch:
                _uid = str(getattr(_m, "user_id", "") or "").strip()
                _text = str(getattr(_m, "processed_plain_text", "") or "")[:200]
                _reply_anchor = str(getattr(_m, "reply_to", "") or "").strip()
                _mentioned_users = [
                    str(_target).strip() for _target in (getattr(_m, "at_list", None) or []) if str(_target).strip()
                ]
                if _uid:
                    _gs.record_message(
                        user_id=_uid,
                        text=_text,
                        mentioned_users=_mentioned_users,
                        reply_anchor=_reply_anchor,
                    )
            self._cached_scene_snapshot = _gs.snapshot()
            self._group_activity_level = self._derive_group_activity_level(
                self._cached_scene_snapshot
            )
        except Exception as exc:
            self._group_activity_level = 0.0
            logger.debug(f"{self.log_prefix} 群场景追踪异常: {exc}")
        try:
            _gpd = self._orch.get("pattern_detector")
            if self._cached_participant_summary:
                _known_active = int(self._cached_participant_summary.get("hot_count", 0) or 0) + int(
                    self._cached_participant_summary.get("warm_count", 0) or 0
                )
                _gpd.set_known_active_count(_known_active)
            for _m in incoming_batch:
                _uid = str(getattr(_m, "user_id", "") or "").strip()
                _text = str(getattr(_m, "processed_plain_text", "") or "")[:200]
                _targets = [
                    str(_target).strip() for _target in (getattr(_m, "at_list", None) or []) if str(_target).strip()
                ]
                if _uid:
                    _gpd.feed_simple(
                        user_id=_uid,
                        text=_text,
                        is_at_someone=bool(_targets),
                        at_target=_targets[0] if _targets else "",
                    )
            self._cached_pattern_evidence = _gpd.detect()
        except Exception as exc:
            self._cached_pattern_evidence = []
            logger.debug(f"{self.log_prefix} 群体模式检测异常: {exc}")

    @staticmethod
    def _derive_group_activity_level(scene_snapshot: Any) -> float:
        if not scene_snapshot:
            return 0.0

        def _clamp01(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        _msg_rate = _clamp01(
            float(getattr(scene_snapshot, "messages_per_minute", 0.0) or 0.0)
            / 8.0
        )
        _speaker_level = _clamp01(
            float(getattr(scene_snapshot, "unique_speakers_5min", 0) or 0.0)
            / 6.0
        )
        _density = _clamp01(
            float(getattr(scene_snapshot, "social_density", 0.0) or 0.0)
        )
        _quality = _clamp01(
            float(getattr(scene_snapshot, "interaction_quality", 0.0) or 0.0)
        )
        _complexity = _clamp01(
            float(getattr(scene_snapshot, "complexity_level", 0.0) or 0.0)
        )
        _activity = (
            _msg_rate * 0.34
            + _speaker_level * 0.24
            + _density * 0.18
            + _quality * 0.12
            + _complexity * 0.12
        )
        _atmo = getattr(scene_snapshot, "atmosphere", None)
        _atmo_value = (
            str(getattr(_atmo, "value", _atmo) or "").strip().lower()
        )
        if _atmo_value in {
            "heated_discussion",
            "deep_discussion",
            "problem_solving",
            "social_gaming",
        }:
            _activity += 0.06
        elif _atmo_value in {"argument", "spam_flood"}:
            _activity += 0.12
        elif _atmo_value in {"quiet", "mourning"}:
            _activity -= 0.10
        return _clamp01(_activity)

    def _integrate_event_learning(self, incoming_batch: List) -> None:
        """事件驱动学习：从消息、贴纸和活跃话题持续吸收线索。"""
        self._cached_learning_summary = {}
        try:
            _hub = self._orch.get("learning_hub")
            _items: List[Dict[str, Any]] = []
            _seen_keys: set = set()
            for i, _m in enumerate(incoming_batch):
                _uid = str(getattr(_m, "user_id", "") or "").strip()
                if not _uid or _uid == "bot":
                    continue
                _text = str(getattr(_m, "processed_plain_text", "") or "")[:120]
                _route = self._cached_semantic_routes[i] if i < len(self._cached_semantic_routes) else None
                _category = (
                    getattr(
                        getattr(_route, "category", None),
                        "value",
                        "plain_text",
                    )
                    if _route is not None
                    else "plain_text"
                )
                _summary = _hub.observe_message(
                    user_id=_uid,
                    text=_text,
                    semantic_category=str(_category or "plain_text"),
                )
                for _item in _summary.get("items", []):
                    _item_key = f"{_item.get('type', '')}::{_item.get('content', '')}"
                    if _item_key in _seen_keys:
                        continue
                    _seen_keys.add(_item_key)
                    _items.append(_item)
            try:
                _topic_rows = self._orch.get("scene_state").get_active_topics()
            except Exception:
                _topic_rows = []
            _topic_candidates = [
                str(_row.get("keyword", "")).strip()
                for _row in _topic_rows
                if isinstance(_row, dict) and int(_row.get("mention_count", 0) or 0) >= 2
            ][:3]
            if _topic_candidates:
                _topic_summary = _hub.observe_topic_keywords(_topic_candidates, source_user="scene")
                for _item in _topic_summary.get("items", []):
                    _item_key = f"{_item.get('type', '')}::{_item.get('content', '')}"
                    if _item_key in _seen_keys:
                        continue
                    _seen_keys.add(_item_key)
                    _items.append(_item)
            if _items:
                self._cached_learning_summary = {
                    "learned": True,
                    "hint": "；".join(str(_item.get("content", "")) for _item in _items[:3]),
                    "count": len(_items),
                    "items": _items,
                    "types": sorted({str(_item.get("type", "")) for _item in _items if _item.get("type")}),
                }
            else:
                self._cached_learning_summary = _hub.last_summary()
        except Exception as exc:
            self._cached_learning_summary = {}
            logger.debug(f"{self.log_prefix} 事件驱动学习异常: {exc}")

    def _integrate_narration_planning(self, decision_messages: List) -> None:
        """旁白与行为策略规划"""
        try:
            _np = self._orch.get("narration_planner")
            _latest_user_msg = self._get_latest_human_message(decision_messages)
            # 理解层级：5级判定
            _understanding_level = "can_understand_no_reply"
            if self._cached_understanding_results:
                try:
                    _primary = self._cached_understanding_results[0]
                    _cl = getattr(getattr(_primary, "level", None), "value", "")
                    if _cl:
                        _understanding_level = _cl
                except Exception:
                    try:
                        from src.core.understanding_gate import (
                            get_understanding_gate,
                        )

                        if get_understanding_gate().any_worth_replying(self._cached_understanding_results):
                            _understanding_level = "can_reply_simple"
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
            _visibility_level = "noticed"
            if self._cached_visibility_decisions:
                for _vd in self._cached_visibility_decisions:
                    _vl = getattr(_vd, "level", None)
                    if _vl:
                        _vl_val = _vl.value if hasattr(_vl, "value") else str(_vl)
                        if _vl_val in ("focused", "understood_but_ignored"):
                            _visibility_level = _vl_val
                            break
            _watch_level = "skim_window"
            if self._cached_watch_level:
                _watch_level = (
                    self._cached_watch_level.value
                    if hasattr(self._cached_watch_level, "value")
                    else str(self._cached_watch_level)
                )
            _presence_mode = "一般"
            if self._cached_presence_state:
                _sw = float(getattr(self._cached_presence_state, "social_willingness", 0.5) or 0.5)
                if _sw > 0.7:
                    _presence_mode = "积极"
                elif _sw < 0.3:
                    _presence_mode = "退缩"
            _energy = 1.0
            if self._cached_metabolism_state:
                _energy = float(getattr(self._cached_metabolism_state, "chat_fuel", 100.0) or 100.0) / 100.0
            _social_w = 0.5
            if self._cached_presence_state:
                _social_w = float(getattr(self._cached_presence_state, "social_willingness", 0.5) or 0.5)
            _recent_bot = self._recent_bot_texts()
            _msg_summary = ""
            if _latest_user_msg is not None:
                _msg_summary = str(
                    getattr(_latest_user_msg, "processed_plain_text", "")
                    or getattr(_latest_user_msg, "plain_text", "")
                    or getattr(_latest_user_msg, "content", "")
                    or ""
                )[:100]
            # 自引用：提取最强引用类型和强度
            _is_at = False
            _self_ref_type = "none"
            _self_ref_strength = 0.0
            if self._cached_self_references:
                _best_strength = 0.0
                for _ref in self._cached_self_references:
                    _rt = getattr(_ref, "ref_type", None)
                    _rs = float(getattr(_ref, "strength", 0.0) or 0.0)
                    if _rt:
                        _rt_val = _rt.value if hasattr(_rt, "value") else str(_rt)
                        if _rs > _best_strength:
                            _best_strength = _rs
                            _self_ref_type = _rt_val
                        if _rt_val == "direct_at":
                            _is_at = True
                _self_ref_strength = _best_strength
            _is_question = bool(
                _msg_summary and (_msg_summary.rstrip().endswith("?") or _msg_summary.rstrip().endswith("？"))
            )
            # 主观存在判定
            _dominant_mood = ""
            _topic_interest = 0.0
            _want_learn = False
            if self._cached_presence_state:
                _dominant_mood = str(
                    getattr(self._cached_presence_state, "mood_label", lambda: "")()
                    if callable(getattr(self._cached_presence_state, "mood_label", None))
                    else ""
                )
                # 话题兴趣
                _main_topic = ""
                try:
                    _scene_topics = self._orch.get("scene_state").active_topics(limit=1)
                    if _scene_topics:
                        _main_topic = _scene_topics[0]
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
                if _main_topic and hasattr(self._cached_presence_state, "topic_interest_score"):
                    _topic_interest = self._cached_presence_state.topic_interest_score(_main_topic)
                # 学习优先判定
                if hasattr(self._cached_presence_state, "should_learn_before_reply"):
                    _has_unknown_meme = False
                    if self._cached_learning_summary:
                        _lh_types = self._cached_learning_summary.get("types", [])
                        _has_unknown_meme = "meme_word" in _lh_types and not self._cached_learning_summary.get(
                            "learned"
                        )
                    _want_learn = self._cached_presence_state.should_learn_before_reply(
                        comprehension_level=_understanding_level,
                        has_unknown_meme=_has_unknown_meme,
                    )
            # 缓存完整判定供面板使用
            self._cached_presence_verdict = {}
            if self._cached_presence_state and hasattr(self._cached_presence_state, "full_verdict_dict"):
                self._cached_presence_verdict = self._cached_presence_state.full_verdict_dict(
                    comprehension_level=_understanding_level,
                    has_unknown_meme=_want_learn,
                    current_topic=(_msg_summary[:20] if _msg_summary else ""),
                )
            # 代谢约束传递
            _metab_hint = self._cached_metabolism_constraints.get("reply_length_hint", "normal")
            _metab_perfunctory = bool(self._cached_metabolism_constraints.get("is_perfunctory", False))
            _metab_gate = bool(self._cached_metabolism_constraints.get("energy_gate_open", True))
            # 场景适合度与群体模式
            _scene_joinable = True
            if self._cached_scene_snapshot:
                _scene_joinable = bool(getattr(self._cached_scene_snapshot, "suitable_to_join", True))
            _dominant_pattern_name = ""
            if self._cached_pattern_evidence:
                _top_pat = self._cached_pattern_evidence[0]
                _pat_conf = float(getattr(_top_pat, "confidence", 0.0) or 0.0)
                if _pat_conf > 0.4:
                    _pat_enum = getattr(_top_pat, "pattern", None)
                    _dominant_pattern_name = (
                        (_pat_enum.value if hasattr(_pat_enum, "value") else str(_pat_enum)) if _pat_enum else ""
                    )
            # 语音/音乐消息检测
            _has_voice_music = False
            if self._cached_semantic_routes:
                for _route in self._cached_semantic_routes:
                    _cat_val = getattr(getattr(_route, "category", None), "value", "")
                    if _cat_val in ("voice", "music_share"):
                        _has_voice_music = True
                        break
            # 创伤与烦躁值
            _trauma = 0.0
            _annoyance = 0.0
            _negative_emo_val = 0.0
            if self._last_relation_snapshot:
                _trauma = float(self._last_relation_snapshot.get("trauma_score", 0.0) or 0.0)
            if self._cached_metabolism_state:
                _annoyance = float(getattr(self._cached_metabolism_state, "channel_annoyance", 0.0) or 0.0)
            # 从情绪追踪器提取用户负面情绪聚合值
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import (
                    get_emotion_tracker,
                )

                _emo_tracker = get_emotion_tracker(self.stream_id)
                _target_uid = self._get_latest_human_user_id(decision_messages)
                if _target_uid:
                    _emo_state = _emo_tracker.get_user_state(_target_uid, create_if_missing=False)
                    if _emo_state:
                        _negative_emo_val = _emo_state.negative_emotion_aggregate
            except Exception:
                _negative_emo_val = 0.0
            self._cached_user_negative_emotion = _negative_emo_val
            # 从印象演化中枢提取用户个性标签
            _user_impression_tags = []
            try:
                from src.core.impression_evolution_hub import (
                    get_impression_hub,
                )

                _ih = get_impression_hub(self.stream_id)
                _tag_uid = self._get_latest_human_user_id(decision_messages)
                if _tag_uid:
                    _layers = _ih.get_all_layers(_tag_uid)
                    if _layers:
                        _subj = _layers.get("subjective_tag", None)
                        if _subj and hasattr(_subj, "personality_tags"):
                            _user_impression_tags = list(_subj.personality_tags or [])
            except Exception:
                _user_impression_tags = []
            self._cached_user_impression_tags = _user_impression_tags
            self._cached_narration_plan = _np.plan(
                understanding_level=_understanding_level,
                visibility_level=_visibility_level,
                watch_level=_watch_level,
                presence_mode=_presence_mode,
                energy_ratio=_energy,
                social_willingness=_social_w,
                is_at_bot=_is_at,
                is_question=_is_question,
                recent_bot_texts=_recent_bot,
                message_summary=_msg_summary,
                dominant_mood=_dominant_mood,
                topic_interest_level=_topic_interest,
                want_learn_first=_want_learn,
                self_ref_type=_self_ref_type,
                self_ref_strength=_self_ref_strength,
                metabolism_reply_hint=_metab_hint,
                metabolism_is_perfunctory=_metab_perfunctory,
                metabolism_gate_open=_metab_gate,
                scene_suitable_to_join=_scene_joinable,
                dominant_pattern=_dominant_pattern_name,
                msg_has_voice_or_music=_has_voice_music,
                trauma_score=_trauma,
                channel_annoyance=_annoyance,
                misunderstanding_risk=(
                    float(
                        getattr(
                            self._cached_misunderstanding_signal,
                            "risk_score",
                            0.0,
                        )
                        or 0.0
                    )
                    if self._cached_misunderstanding_signal
                    else 0.0
                ),
                user_negative_emotion=_negative_emo_val,
                user_impression_tags=_user_impression_tags,
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 旁白策略规划异常: {exc}")

    def _integrate_memory_reactivation(self, incoming_batch: List) -> None:
        """记忆激活门控：话题匹配 + 用户回归触发"""
        try:
            _mrg = self._orch.get("memory_reactivation")
            _ts = self._orch.get("subjective_time")
            _topics: List[str] = []
            if self._cached_scene_snapshot:
                _topic_count = int(getattr(self._cached_scene_snapshot, "active_topic_count", 0) or 0)
                if _topic_count > 0:
                    try:
                        _topics = self._orch.get("scene_state").active_topics(limit=3)
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
            if _topics:
                _candidates = _mrg.evaluate_topic_trigger(_topics, _ts)
                for _c in _candidates:
                    _mid = getattr(_c, "message_id", "") or ""
                    _reason = getattr(_c, "reason", "") or ""
                    if _mid:
                        _ts.reactivate(_mid, boost=0.3, reason=_reason)
            _returning_uids: set = set()
            for _m in incoming_batch:
                _uid = str(getattr(_m, "user_id", "") or "").strip()
                if self._is_human_message_obj(_m):
                    _returning_uids.add(_uid)
            for _uid in _returning_uids:
                _mrg.evaluate_user_return_trigger(_uid, _ts)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 记忆激活门控异常: {exc}")

    def _integrate_post_reply_learning(
        self,
        decision_messages: List,
        reply_text: str = "",
        learning_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """回复后学习 + 技能追踪 + 印象记录 + 代谢消耗"""
        _target_uid = str(getattr(self, "_last_user_id", "") or "").strip()
        _learning_ctx = learning_context or {}
        _behavior_signal = dict(_learning_ctx.get("behavior_signal") or {})
        _repetition_signal = dict(_learning_ctx.get("repetition_signal") or {})
        _harassment_signal = dict(_learning_ctx.get("harassment_signal") or {})
        if not _target_uid:
            _target_uid = self._get_latest_human_user_id(decision_messages)
        try:
            _ih = self._orch.get("impression_hub")
            if _target_uid:
                if _harassment_signal.get("detected"):
                    _behavior_signal.setdefault("category", "harassing")
                    _behavior_signal.setdefault("behavior_type", "repetitive_harassment")
                    _behavior_signal.setdefault("reason", _harassment_signal.get("reason", "持续骚扰/冒犯输入"))
                elif _repetition_signal.get("detected") and _repetition_signal.get("low_info_cluster"):
                    _exact_repeats = int(_repetition_signal.get("exact_repeat_count", 0) or 0)
                    if _exact_repeats >= 3:
                        _behavior_signal.setdefault("category", "unfriendly")
                        _behavior_signal.setdefault("behavior_type", "low_info_repeat")
                    else:
                        _behavior_signal.setdefault("category", "neutral")
                        _behavior_signal.setdefault("behavior_type", "mild_repeat")
                    _behavior_signal.setdefault("reason", _repetition_signal.get("reason", "低信息重复施压"))
                _category = str(_behavior_signal.get("category", "neutral") or "neutral")
                _behavior_type = str(_behavior_signal.get("behavior_type", "casual_chat") or "casual_chat")
                _reason = str(_behavior_signal.get("reason", "") or "")
                imp = None
                _allow_affection_sync = False
                if _category == "friendly":
                    imp = _ih.record_interaction(
                        _target_uid,
                        quality=0.85,
                        is_positive=True,
                        event_summary=_reason,
                    )
                    _allow_affection_sync = True
                elif _category == "neutral" and _behavior_type in {"casual_chat", "casual_inquiry"}:
                    imp = _ih.record_interaction(
                        _target_uid,
                        quality=0.60,
                        is_positive=True,
                        event_summary=_reason,
                    )
                    _allow_affection_sync = True
                elif _behavior_type == "mild_repeat":
                    logger.debug(f"{self.log_prefix} 印象学习跳过 mild_repeat: {_reason or '轻度重复'}")
                elif _category == "unfriendly":
                    imp = _ih.record_interaction(
                        _target_uid,
                        quality=0.30,
                        is_positive=False,
                        event_summary=_reason,
                    )
                elif _category in {"hostile", "harassing"}:
                    _ih.record_conflict(_target_uid, reason=_reason[:80])
                else:
                    imp = _ih.record_interaction(
                        _target_uid,
                        quality=0.60,
                        is_positive=True,
                        event_summary=_reason,
                    )
                    _allow_affection_sync = True
                # 从印象事实统计中计算参考好感值（仅用于正向补充，不向下覆盖）
                _computed_aff = 0.0
                if _allow_affection_sync and imp and hasattr(imp, "truth"):
                    _pos = getattr(imp.truth, "positive_interactions", 0) or 0
                    _shared = getattr(imp.truth, "shared_experience_count", 0) or 0
                    _computed_aff = min(100.0, _pos * 2.0 + _shared * 4.0)
                if _allow_affection_sync and _computed_aff > 0:
                    try:
                        from src.modules.modcore.dynamic_persona.emotion_tracker import (
                            get_emotion_tracker,
                        )

                        _et = get_emotion_tracker(self.stream_id)
                        _user_state = _et.get_user_state(_target_uid)
                        if _user_state:
                            _old_aff = getattr(_user_state, "affection", 0.0) or 0.0
                            _impression_aff = _computed_aff
                            # 只在印象值高于当前值时做正向补充，避免覆盖其他系统的增量
                            _positive_delta = max(0.0, _impression_aff - _old_aff)
                            if _positive_delta > 0.01:
                                _et.update_affection(
                                    _target_uid,
                                    _positive_delta,
                                    "印象演化同步",
                                )
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
                # 群环境联动演化（替代不存在的 evolve 方法）
                _atmo = ""
                if self._cached_scene_snapshot and hasattr(self._cached_scene_snapshot, "atmosphere"):
                    _atmo_enum = self._cached_scene_snapshot.atmosphere
                    _atmo = _atmo_enum.value if hasattr(_atmo_enum, "value") else str(_atmo_enum)
                _density = 0.0
                if self._cached_participant_summary:
                    _hot = int(self._cached_participant_summary.get("hot_count", 0) or 0)
                    _warm = int(self._cached_participant_summary.get("warm_count", 0) or 0)
                    _density = min(1.0, (_hot + _warm) / 20.0)
                _pat_type = ""
                if self._cached_pattern_evidence:
                    _dom = self._cached_pattern_evidence[0]
                    _pat_type = str(getattr(_dom, "pattern_name", "") or "")
                _ih.evolve_with_group_context(_target_uid, _atmo, _density, _pat_type)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 印象演化异常: {exc}")
        try:
            from src.core.learning_hub import (
                LearningEvent,
                LearnableType,
            )

            _lh = self._orch.get("learning_hub")
            for _m in self._get_human_message_candidates(list(decision_messages[-6:]))[-3:]:
                _text = str(getattr(_m, "processed_plain_text", "") or "").strip()
                _uid = str(getattr(_m, "user_id", "") or "").strip()
                if _text and self._is_human_message_obj(_m):
                    _lh.observe(
                        LearningEvent(
                            event_type=LearnableType.REACTION_PATTERN,
                            content=_text[:100],
                            source_user=_uid,
                        )
                    )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 学习中心观察异常: {exc}")
        # 学习反馈闭环：已学习的梗/礼仪在回复中被使用时记录
        try:
            _lh = self._orch.get("learning_hub")
            if reply_text and self._cached_learning_summary:
                _learned_items = self._cached_learning_summary.get("items", [])
                for _item_name in _learned_items[:5]:
                    if isinstance(_item_name, str) and _item_name in reply_text:
                        _lh.record_usage(_item_name, success=True)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 学习使用反馈异常: {exc}")
        try:
            _slh = self._orch.get("skill_hub")
            _slh.record_invocation("conversation_reply", success=bool(reply_text))
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 技能生命周期追踪异常: {exc}")
        try:
            _d6 = EnergyChainDimension.get_instance()
            _cost_behavior = "reply"
            _cost_intensity = 1.0
            if self._cached_pattern_evidence:
                for _pe in self._cached_pattern_evidence:
                    _pe_val = getattr(getattr(_pe, "pattern", None), "value", "") or ""
                    if _pe_val == "spectator_mode" and float(getattr(_pe, "confidence", 0) or 0) > 0.4:
                        _cost_behavior = "glance"
                        break
            if bool(self._cached_metabolism_constraints.get("is_perfunctory", False)):
                _cost_intensity = 0.6

            class _ReplyEvt:
                channel_id = self.stream_id
                event_type = "reply_completed"
                reply_tokens = len(reply_text) if reply_text else 80
                raw_extras = {"is_admin": getattr(self, "_last_msg_was_admin", False)}

            _d6.on_event(_ReplyEvt())
        except Exception as exc:
            logger.debug(f"{self.log_prefix} D6消耗异常: {exc}")
        # 风格质量反思调度：回复后触发style_reflector评估风格质量
        try:
            from src.express.style_reflector import get_style_reflector

            _reflector = get_style_reflector(self.stream_id)
            if _reflector.should_trigger_reflect():
                import asyncio

                _loop = asyncio.get_running_loop()
                if _loop.is_running():
                    self._spawn(_reflector.trigger_reflect_task())
        except ImportError:
            pass
        except Exception as _e:
            logger.debug(f"{self.log_prefix} 风格反思异常: {_e}")

    def _register_builtin_skills(self) -> None:
        """首次运行时将内建技能注册到生命周期中心"""
        if self._skills_registered:
            return
        try:
            from src.core.skill_lifecycle_hub import (
                get_skill_lifecycle_hub,
                SkillDomain,
                SkillCategory,
            )

            _hub = get_skill_lifecycle_hub(self.stream_id)
            _builtin_catalog = [
                (
                    "conversation_reply",
                    "对话回复",
                    SkillCategory.CONVERSATION,
                    SkillDomain.RUNTIME,
                ),
                (
                    "deep_understanding",
                    "深度语义理解",
                    SkillCategory.KNOWLEDGE,
                    SkillDomain.ANALYSIS,
                ),
                (
                    "narration_planning",
                    "旁白策略规划",
                    SkillCategory.CONVERSATION,
                    SkillDomain.RUNTIME,
                ),
                (
                    "message_routing",
                    "消息语义路由",
                    SkillCategory.TASK_ASSIST,
                    SkillDomain.ANALYSIS,
                ),
                (
                    "emotion_tracking",
                    "情绪存在追踪",
                    SkillCategory.EMOTIONAL_SUPPORT,
                    SkillDomain.RUNTIME,
                ),
                (
                    "pattern_detection",
                    "群体模式识别",
                    SkillCategory.KNOWLEDGE,
                    SkillDomain.ANALYSIS,
                ),
                (
                    "learning_ingest",
                    "学习素材采集",
                    SkillCategory.KNOWLEDGE,
                    SkillDomain.DEV,
                ),
                (
                    "memory_association",
                    "记忆关联激活",
                    SkillCategory.KNOWLEDGE,
                    SkillDomain.RUNTIME,
                ),
            ]
            for _sid, _sname, _cat, _dom in _builtin_catalog:
                _hub.register_skill(_sid, skill_name=_sname, category=_cat, domain=_dom)
            self._skills_registered = True
            logger.debug(f"{self.log_prefix} 内建技能注册完成 共{len(_builtin_catalog)}项")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 内建技能注册异常: {exc}")

    def _integrate_skill_lifecycle_review(self) -> None:
        """周期性审查技能生命周期状态，注册内建技能并执行晋升/降级评估"""
        self._register_builtin_skills()
        if self._enhanced_round % self._skill_review_round_interval != 0:
            return
        try:
            _hub = self._orch.get("skill_hub")
            _review = _hub.periodic_review()
            self._cached_skill_review = _review
            _summary = _hub.summary()
            _changed = _review.get("promoted", 0) + _review.get("declined", 0) + _review.get("retired", 0)
            if _changed > 0:
                logger.info(
                    f"{self.log_prefix} 技能审查 晋升={_review['promoted']} "
                    f"衰退={_review['declined']} 退役={_review['retired']} "
                    f"总计={_summary.get('total_skills', 0)}"
                )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 技能周期审查异常: {exc}")

    def _track_skill_invocations(self) -> None:
        """根据本轮缓存结果批量记录各子系统技能的调用情况"""
        try:
            _hub = self._orch.get("skill_hub")
            # 深度理解技能
            if self._cached_understanding_results:
                _primary = self._cached_understanding_results[0]
                _level_val = getattr(getattr(_primary, "level", None), "value", "")
                _understood = _level_val in ("high", "medium", "contextual")
                _eff = 0.65 if _understood else 0.25
                _hub.record_invocation(
                    "deep_understanding",
                    success=_understood,
                    effectiveness=_eff,
                )
            # 旁白策略技能
            if self._cached_narration_plan:
                _hub.record_invocation("narration_planning", success=True, effectiveness=0.55)
            # 消息路由技能
            if self._cached_semantic_routes:
                _hub.record_invocation("message_routing", success=True, effectiveness=0.50)
            # 群体模式识别
            if self._cached_pattern_evidence:
                _dom = self._cached_pattern_evidence[0]
                _pat_conf = float(getattr(_dom, "confidence", 0.0) or 0.0)
                _hub.record_invocation(
                    "pattern_detection",
                    success=(_pat_conf > 0.3),
                    effectiveness=min(1.0, _pat_conf + 0.1),
                )
            # 情绪存在态
            if self._cached_presence_state:
                _hub.record_invocation("emotion_tracking", success=True, effectiveness=0.50)
            # 学习素材采集
            if self._cached_learning_summary and self._cached_learning_summary.get("items"):
                _item_count = len(self._cached_learning_summary["items"])
                _hub.record_invocation(
                    "learning_ingest",
                    success=True,
                    effectiveness=min(1.0, 0.3 + _item_count * 0.1),
                )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 技能调用追踪异常: {exc}")

    def _integrate_adaptive_collection(
        self,
        incoming_batch: List,
        reply_success: bool,
        reply_text: str = "",
    ) -> None:
        """自适应管线收集：理解空白/回复失败/弱模式归入事件池，周期性提出候选并执行验证发布"""
        try:
            from src.core.adaptive_update_pipeline import get_adaptive_pipeline

            _pipe = get_adaptive_pipeline()
            # 理解空白事件收集
            if self._cached_understanding_results:
                for _ur in self._cached_understanding_results:
                    _lv = getattr(getattr(_ur, "level", None), "value", "")
                    if _lv in ("low", "none", ""):
                        _snippet = str(getattr(_ur, "original_text", "") or "")[:60]
                        _pipe.collect_event(
                            event_type="understanding_gap",
                            raw_content=f"理解不足: {_snippet}",
                            source_channel=self.stream_id,
                            metadata={
                                "level": _lv,
                                "round": self._enhanced_round,
                            },
                            priority=2,
                        )
            # 回复失败收集
            if not reply_success and incoming_batch:
                _last_content = ""
                for _m in reversed(incoming_batch):
                    _t = str(getattr(_m, "processed_plain_text", "") or "").strip()
                    if _t:
                        _last_content = _t[:80]
                        break
                _pipe.collect_event(
                    event_type="reply_failure",
                    raw_content=f"收到消息但未能回复: {_last_content}",
                    source_channel=self.stream_id,
                    metadata={"round": self._enhanced_round},
                    priority=3,
                )
            # 弱模式信号收集
            if self._cached_pattern_evidence:
                _dom_pat = self._cached_pattern_evidence[0]
                _pc = float(getattr(_dom_pat, "confidence", 0.0) or 0.0)
                _pn = str(getattr(_dom_pat, "pattern_name", "") or "")
                if _pc < 0.3 and _pn:
                    _pipe.collect_event(
                        event_type="weak_pattern",
                        raw_content=f"弱模式信号: {_pn}(confidence={_pc:.2f})",
                        source_channel=self.stream_id,
                        metadata={"round": self._enhanced_round},
                        priority=1,
                    )
            # 被误解信号收集
            if self._cached_misunderstanding_signal and self._cached_misunderstanding_signal.risk_score > 0.3:
                _ms = self._cached_misunderstanding_signal
                _pipe.collect_event(
                    event_type="misunderstanding_detected",
                    raw_content=(
                        f"被误解({_ms.signal_type}): 风险={_ms.risk_score:.2f} "
                        f"线索={','.join(_ms.detected_cues[:2])}"
                    ),
                    source_channel=self.stream_id,
                    metadata={"round": self._enhanced_round},
                    priority=2,
                )
            # 周期性触发管线（每20轮执行一次）
            if self._enhanced_round % 20 == 0:
                _stats = _pipe.get_collection_stats()
                _recent_count = int(_stats.get("recent_window_count", 0) or 0)
                if _recent_count >= 5:
                    _run = _pipe.run_pipeline()
                    logger.info(
                        f"{self.log_prefix} 自适应管线执行完成 run={getattr(_run, 'run_id', '')} "
                        f"status={getattr(_run, 'final_status', '')}"
                    )
            self._cached_pipeline_summary = {
                "collection": _pipe.get_collection_stats(),
                "history": _pipe.pipeline_history_summary(),
            }
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 自适应管线收集异常: {exc}")

    def _integrate_impression_periodic_maintenance(self) -> None:
        """周期性印象维护：重写消除偏差 + 自动昵称生成"""
        if self._enhanced_round % 15 != 0:
            return
        try:
            _ih = self._orch.get("impression_hub")
            # 周期性全量重写（消除增量累积偏差）
            _rewritten = _ih.periodic_rewrite_all()
            if _rewritten > 0:
                logger.info(f"{self.log_prefix} 印象周期重写 {_rewritten}人")
            # 为最近交互用户生成自动昵称
            _target_uid = str(getattr(self, "_last_user_id", "") or "").strip()
            if _target_uid:
                _nick = _ih.auto_generate_nickname(_target_uid)
                if _nick:
                    logger.debug(f"{self.log_prefix} 自动昵称生成 {_target_uid[:8]}→{_nick[:10]}")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 印象周期维护异常: {exc}")

    def _check_pipeline_early_exit(self, incoming_batch: list, pinged_msg) -> Optional[str]:
        """早退出检测：噪声消息/纯emoji/超短消息/重复刷屏 跳过整个管线
        返回退出原因字符串，None表示不退出
        """
        if not incoming_batch:
            return "空消息批"
        _is_admin = self._is_force_wake_admin(incoming_batch, pinged_msg)
        if _is_admin:
            return None
        _texts = []
        for _m in incoming_batch:
            _t = getattr(_m, "content", "") or getattr(_m, "plain_text", "") or ""
            if _t and not self._is_bot_message_obj(_m):
                _texts.append(_t.strip())
        if not _texts:
            return None
        _combined = "".join(_texts)
        _total_chars = len(_combined)
        if _total_chars <= 1:
            return f"空内容(长度={_total_chars})"
        import re as _re

        _clean = _re.sub(r"[\s\u200b-\u200f\ufeff]", "", _combined)
        if len(_clean) <= 1:
            return "仅空白/控制字符"
        _emoji_only = _re.sub(
            r"[^\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
            r"\U00002702-\U000027B0\U0001F900-\U0001F9FF"
            r"\u2600-\u26FF\u2700-\u27BF\U0001FA00-\U0001FA6F"
            r"\U0001FA70-\U0001FAFF\U0000FE00-\U0000FE0F]",
            "",
            _clean,
        )
        if len(_emoji_only) == 0 and len(_clean) > 0 and _clean.isascii():
            return "纯emoji消息"
        if _total_chars <= 3:
            _all_single_char = all(len(t) <= 2 for t in _texts)
            if _all_single_char:
                return f"超短消息(总{_total_chars}字)"
        _recent_ts = time.time()
        _window = 15.0
        _count_recent = sum(
            1 for t in _texts if abs(_recent_ts - (getattr(incoming_batch[0], "timestamp", 0) or 0)) < _window
        )
        if len(_texts) >= 4 and _count_recent >= 3:
            _sample = _texts[0][:20] if _texts else ""
            return f"高频短消息刷屏(本轮{len(_texts)}条, '{_sample}...')"
        return None

    def _integrate_memory_system_periodic(self):
        """记忆系统全模块周期维护 —— 10个子系统按各自节奏调度"""

        _now = time.time()
        _hour = datetime.datetime.now().hour
        _loop = asyncio.get_running_loop()
        _is_night_window = 1 <= _hour <= 5
        # ── ① 记忆核心例行维护（去重30min/压缩2h/深度优化24h/过载检查10min） ──
        try:
            from src.memory_system.memory_core import get_memory_core

            _mc = get_memory_core()
            _mc.execute_routine_maintenance(stream_id=self.stream_id)
        except Exception as _e:
            logger.warning(f"{self.log_prefix} 记忆核心维护异常: {_e}")
        # ── ② 海马体层间流转（每30min：感觉→工作→短期→中期→长期+衰减+遗忘+压缩） ──
        try:
            from src.memory_system.hippocampus_buffer import get_hippocampus_buffer

            _hippo = get_hippocampus_buffer(self.stream_id)
            if _loop.is_running():
                self._spawn(_hippo.drain_and_transform())
            else:
                try:
                    _loop.run_until_complete(_hippo.drain_and_transform())
                except RuntimeError:
                    self._spawn(_hippo.drain_and_transform())
        except Exception as _e:
            logger.warning(f"{self.log_prefix} 海马体流转异常: {_e}")
        # ── ③ 夜间记忆整合（凌晨1-5点窗口：当日记忆三级分流→保留/压缩/丢弃） ──
        # 基于日期种子重置标志，避免跨天时标志永不重置
        _today_day_id = int(_now / 86400.0)
        _last_tidy_day = getattr(self, "_nightly_tidy_day_id", -1)
        if _today_day_id != _last_tidy_day:
            self._nightly_tidy_done_today = False
            self._nightly_tidy_day_id = _today_day_id
        if _is_night_window:
            try:
                from src.memory_system.sleep_consolidator import (
                    NightlyMemoryTidier,
                    acquire_nightly_tidier,
                    acquire_tidy_patroller,
                )

                _tidier = acquire_nightly_tidier(self)
                if _tidier is None:
                    _tidier = NightlyMemoryTidier(self)
                if _tidier and not getattr(self, "_nightly_tidy_done_today", False):
                    _patroller = acquire_tidy_patroller(_tidier)
                    if _patroller and _loop.is_running():
                        self._spawn(_tidier.execute_tidy_cycle())
                    else:
                        try:
                            _loop.run_until_complete(_tidier.execute_tidy_cycle())
                        except RuntimeError:
                            self._spawn(_tidier.execute_tidy_cycle())
                    self._nightly_tidy_done_today = True
            except ImportError:
                pass
            except Exception as _e:
                logger.warning(f"{self.log_prefix} 夜间整合异常: {_e}")
        # ── ④ LLM驱动记忆老化（每6h：清晰度标签+内容智能压缩） ──
        try:
            from src.memory_system.aging_processor import MemoryAgingProcessor, BufferedMemory as _AgingBuf

            _aging_ts = float(getattr(self, "_ts_aging_cycle", 0))
            if (_now - _aging_ts) > 21600:
                _processor = MemoryAgingProcessor()
                StorageModel = None
                try:
                    from src.common.database.database_model import (
                        MemoryRecord,
                    )

                    StorageModel = MemoryRecord
                except ImportError:
                    pass
                if StorageModel is not None:
                    _raw_records = list(StorageModel.select().where(StorageModel.stream_id == self.stream_id).limit(50))
                    _memories = [
                        _AgingBuf(
                            memory_id=str(r.record_id),
                            content=str(r.content or ""),
                            created_at=float(r.birth_ts or 0),
                            importance=float(r.significance or 0.5),
                            access_count=int(r.visit_count or 0),
                            last_accessed=float(r.last_visit_ts or 0),
                            channel_id=str(r.stream_id or ""),
                            user_id=str(getattr(r, "user_id", "") or ""),
                        )
                        for r in _raw_records
                    ]
                    if _memories and _loop.is_running():
                        self._spawn(_processor.process_aging_cycle(_memories))
                self._ts_aging_cycle = _now
        except ImportError:
            pass
        except Exception as _e:
            logger.warning(f"{self.log_prefix} 记忆老化异常: {_e}")
        # ── ⑤ 重要性衰减调度（每12h：旧记忆重要性自然衰减） ──
        try:
            from src.memory_system.decay_forgetter import ImportanceDecayScheduler
            from src.memory_system.decay_forgetter import BufferedMemory as _DecayBuf

            _decay_ts = float(getattr(self, "_ts_decay_cycle", 0))
            if (_now - _decay_ts) > 43200:
                _scheduler = ImportanceDecayScheduler()
                _stream_ref = self.stream_id

                def _mem_provider():
                    try:
                        from src.common.database.database_model import (
                            MemoryRecord,
                        )

                        _raw = list(MemoryRecord.select().where(MemoryRecord.stream_id == _stream_ref))
                        return [
                            _DecayBuf(
                                memory_id=str(r.record_id),
                                content=str(r.content or ""),
                                created_at=float(r.birth_ts or 0),
                                importance=float(r.significance or 0.5),
                                access_count=int(r.visit_count or 0),
                                last_accessed=float(r.last_visit_ts or 0),
                                channel_id=str(r.stream_id or ""),
                                user_id=str(getattr(r, "user_id", "") or ""),
                            )
                            for r in _raw
                        ]
                    except Exception as _exc:
                        logger.warning(f"{self.log_prefix} 记忆衰减数据加载异常: {_exc}")
                        return []

                if _loop.is_running():
                    self._spawn(_scheduler.run_decay_cycle(_mem_provider))
                self._ts_decay_cycle = _now
        except ImportError:
            pass
        except Exception as _e:
            logger.warning(f"{self.log_prefix} 重要性衰减异常: {_e}")
        # ── ⑥ 容量配额治理（每1h：LLM判定是否驱逐低价值记忆） ──
        try:
            from src.memory_system.capacity_governor import CapacityGovernor, BufferedMemory as _GovBuf

            _gov_ts = float(getattr(self, "_ts_capacity_check", 0))
            if (_now - _gov_ts) > 3600:
                _governor = CapacityGovernor()
                from src.common.database.database_model import MemoryRecord

                _all_mems = {
                    str(m.record_id): _GovBuf(
                        memory_id=str(m.record_id),
                        content=str(m.content or ""),
                        created_at=float(m.birth_ts or 0),
                        importance=float(m.significance or 0.5),
                        access_count=int(m.visit_count or 0),
                        last_accessed=float(m.last_visit_ts or 0),
                        channel_id=str(m.stream_id or ""),
                        user_id=str(getattr(m, "user_id", "") or ""),
                    )
                    for m in MemoryRecord.select().where(MemoryRecord.stream_id == self.stream_id)
                }
                if _all_mems:
                    _evict_ids = _governor.evaluate_overflow(tier="recent", memories=_all_mems)
                    if _evict_ids:
                        logger.info(f"{self.log_prefix} [容量治理] 驱逐{len(_evict_ids)}条低价值记忆")
                self._ts_capacity_check = _now
        except ImportError:
            pass
        except Exception as _e:
            logger.warning(f"{self.log_prefix} 容量治理异常: {_e}")
        # ── ⑦ 持久化过期清扫（每6h：按TTL策略清除超龄键值记录） ──
        try:
            from src.memory_system.memory_persistence import acquire_staleness_reaper

            _reaper_ts = float(getattr(self, "_ts_staleness_reap", 0))
            if (_now - _reaper_ts) > 21600:
                _reaper = acquire_staleness_reaper()
                _reaped_count = _reaper.reap()
                if _reaped_count > 0:
                    logger.info(f"{self.log_prefix} [持久化清扫] 移除{_reaped_count}条超龄记录")
                self._ts_staleness_reap = _now
        except ImportError:
            pass
        except Exception as _e:
            logger.warning(f"{self.log_prefix} 持久化清扫异常: {_e}")
        # ── ⑧ 会话流记忆融合（每15min：聚类相似条目+海马体同步） ──
        try:
            from src.memory_system.memory_consolidator import acquire_memory_fusion

            _fusion_ts = float(getattr(self, "_ts_memory_fusion", 0))
            if (_now - _fusion_ts) > 900:
                _fusion = acquire_memory_fusion()
                if _loop.is_running():
                    self._spawn(_fusion.consolidate_stream_entries(self.stream_id))
                self._ts_memory_fusion = _now
        except ImportError:
            pass
        except Exception as _e:
            logger.warning(f"{self.log_prefix} 记忆融合异常: {_e}")
        # ── ⑨ 切片生命周期整合（每30min：STM清晰度衰减→归档提升/清除） ──
        try:
            from src.memory_system.memory_merger import acquire_consolidator

            _merger_ts = float(getattr(self, "_ts_slice_merge", 0))
            if (_now - _merger_ts) > 1800:
                _merger = acquire_consolidator()
                if _loop.is_running():
                    self._spawn(_merger.run_consolidation_cycle())
                self._ts_slice_merge = _now
        except ImportError:
            pass
        except Exception as _e:
            logger.warning(f"{self.log_prefix} 切片整合异常: {_e}")
        # ── ⑩ 历史消息压缩（每20min：批量提取主题+摘要持久化） ──
        try:
            from src.hippo_memorizer.history_condenser import CondenserManager

            _condense_ts = float(getattr(self, "_ts_history_condense", 0))
            if (_now - _condense_ts) > 1200:
                if not hasattr(self, "_condenser_mgr"):
                    self._condenser_mgr = CondenserManager()
                if _loop.is_running():
                    self._spawn(self._condenser_mgr.check_and_condense(self.stream_id))
                self._ts_history_condense = _now
        except ImportError:
            pass
        except Exception as _e:
            logger.warning(f"{self.log_prefix} 历史压缩异常: {_e}")
        # ── ⑪ 梦境维护（夜间1-5点窗口：记忆整理+清理报告生成） ──
        # 基于日期种子重置标志，避免跨天时标志永不重置
        _dream_day_id = int(_now / 86400.0)
        _last_dream_day = getattr(self, "_dream_day_id", -1)
        if _dream_day_id != _last_dream_day:
            self._dream_done_today = False
            self._dream_day_id = _dream_day_id
        if _is_night_window:
            try:
                from src.dream.dream_agent import get_dream_agent

                _dream_ts = float(getattr(self, "_ts_dream_maintenance", 0))
                if (_now - _dream_ts) > 14400 and not getattr(self, "_dream_done_today", False):
                    _agent = get_dream_agent()
                    if _loop.is_running():
                        self._spawn(_agent.run_maintenance(chat_id=self.stream_id))
                    self._dream_done_today = True
                    self._ts_dream_maintenance = _now
            except ImportError:
                pass
            except Exception as _e:
                logger.warning(f"{self.log_prefix} 梦境维护异常: {_e}")

    def _integrate_behavior_chronicle(self, deed_type: str, body: str, user_id: str = "", extra: dict = None):
        """事件驱动的行为记录入口 —— Bot行为编年史持久化"""
        try:
            from src.memory_system.self_behavior_recorder import (
                bootstrap_behavior_chronicle,
            )

            _chronicle = bootstrap_behavior_chronicle()
            _chronicle.chronicle(
                channel_id=self.stream_id,
                deed_type=deed_type,
                body=body,
                participant_id=user_id or None,
                annotations=extra or None,
            )
        except ImportError:
            pass
        except Exception as _e:
            logger.warning(f"{self.log_prefix} 行为记录异常: {_e}")
        # 同步记录到hippo_memorizer行为追踪器
        try:
            from src.hippo_memorizer.action_recorder import record_action

            record_action(
                action_type=deed_type,
                channel_id=self.stream_id,
                context={"body": body, "user_id": user_id, **(extra or {})},
                success=True,
            )
        except ImportError:
            pass
        except Exception as _e:
            logger.debug(f"{self.log_prefix} 动作追踪异常: {_e}")

    def _adjust_memory_by_emotion(self, content: str, base_importance: float, emotion_ctx: dict) -> float:
        """情感→重要性联动：根据当前情绪状态调整记忆重要度"""
        try:
            from src.memory_system.emotion_importance import EmotionImportanceLinker

            _linker = EmotionImportanceLinker()
            _result = _linker.evaluate_sync(content, base_importance, emotion_ctx)
            return _result.adjusted_importance
        except ImportError:
            return base_importance
        except Exception:
            return base_importance

    def _integrate_night_cycle(self, now: float) -> Optional[str]:
        """夜间节律评估 — 从 D6 EnergyChainDimension 的四阶段系统读取（替代旧 NightCycleSystem）"""
        _grace_sec = getattr(self, "_wake_grace_until", 0.0)
        if now < _grace_sec:
            return None
        try:
            _d6 = EnergyChainDimension.get_instance()
            _d6_state = _d6._ensure_channel(self.stream_id)
            _energy_ratio = (
                float(_d6_state.combined_ratio())
                if _d6_state is not None and hasattr(_d6_state, "combined_ratio")
                else 1.0
            )
            _activity_level = (
                float(getattr(_d6_state, "activity_level", 50.0) or 50.0)
                if _d6_state is not None
                else 50.0
            )
            _group_activity = max(0.0, min(1.0, float(getattr(self, "_group_activity_level", 0.0) or 0.0)))
            _social_stimulus = max(_group_activity, max(0.0, min(1.0, (_activity_level - 35.0) / 65.0)))
            _metabolism = getattr(self, "_cached_metabolism_constraints", None) or {}
            _emotion_state = getattr(self, "_cached_emotion_state", None) or {}
            if isinstance(_emotion_state, dict):
                _boredom_raw = _emotion_state.get("boredom", _metabolism.get("boredom", 0.0))
            else:
                _boredom_raw = getattr(_emotion_state, "boredom", _metabolism.get("boredom", 0.0))
            try:
                _boredom = float(_boredom_raw or 0.0)
            except (TypeError, ValueError):
                _boredom = 0.0
            if _boredom <= 1.0:
                _boredom *= 100.0
            _boredom = max(0.0, min(100.0, _boredom))
            _recent_user_times = [
                float(_ts)
                for _ts in list(getattr(self, "_user_msg_timeline", []) or [])
                if now - float(_ts or 0.0) <= 900.0
            ]
            _consecutive_active_minutes = 0.0
            if _recent_user_times:
                _consecutive_active_minutes = min(180.0, max(1.0, (now - min(_recent_user_times)) / 60.0))
            _current_hour = datetime.datetime.now().hour
            _night_mode = _d6._get_night_mode(self.stream_id)
            _phase_value = str(_night_mode.phase or "").strip()
            _phase_name = _phase_value.upper()
            _phase_label = _night_mode.perception_label or "清醒"
            self._cached_night_phase = type(
                "NightPhaseCompat",
                (),
                {
                    "value": _phase_value,
                    "name": _phase_name,
                    "label": lambda _self, _label=_phase_label: _label or "清醒",
                    "__repr__": lambda _s, _phase=_phase_value: f"NightPhase({_phase})",
                },
            )()
            self._cached_night_summary = {
                "phase": _phase_value,
                "prob_multiplier": _night_mode.prob_multiplier,
                "ceiling_penalty": _night_mode.ceiling_penalty,
                "perception_label": _night_mode.perception_label,
                "phase_label": _phase_label or _phase_value,
                "sleep_debt": 0.0,
                "overnight_pressure": 0.0,
                "drowsiness": 0.0,
                "response_suppression": 0.0,
                "body_state_tag": "",
                "monologue_mood_hint": "",
                "expression_style": "normal",
                "is_night": _phase_value != "DAYTIME",
                "is_burnthrough": False,
            }
            try:
                from src.core.night_cycle_system import get_night_cycle

                _ncs = get_night_cycle(self.stream_id)
                _ncs.evaluate(
                    energy_ratio=max(0.0, min(1.0, _energy_ratio)),
                    activity_level=max(0.0, min(100.0, _activity_level)),
                    boredom=_boredom,
                    social_stimulus=_social_stimulus,
                    consecutive_active_minutes=_consecutive_active_minutes,
                    current_hour=_current_hour,
                )
                if _ncs.is_night_hours(_current_hour):
                    _last_user_msg = float(getattr(self, "_last_user_msg_time", 0.0) or 0.0)
                    if 0.0 < now - _last_user_msg <= 90.0:
                        _ncs.record_overnight_activity("peek", min(1.0, 0.35 + _social_stimulus * 0.45))
                        if now - _last_user_msg <= 18.0:
                            _ncs.record_overnight_activity("interrupt", min(1.0, 0.45 + _social_stimulus * 0.55))
                    if _activity_level > 45.0:
                        _ncs.record_overnight_activity("chat", min(1.0, (_activity_level - 35.0) / 65.0))
                    if _energy_ratio < 0.75:
                        _ncs.record_overnight_activity("think", min(1.0, 0.2 + (1.0 - _energy_ratio) * 0.6))
                _ns = _ncs.state_snapshot
                _night_behavior = _ncs.night_behavior_summary() if hasattr(_ncs, "night_behavior_summary") else {}
                _ns_phase = getattr(_ns, "current_phase", None)
                if _ns_phase is None:
                    _ns_phase = getattr(_ncs, "phase", None)
                _ns_phase_val = _ns_phase.value if hasattr(_ns_phase, "value") else str(_ns_phase)
                _ns_phase_name = getattr(_ns_phase, "name", str(_ns_phase_val).upper())
                _ns_phase_label = (
                    _ns_phase.label() if hasattr(_ns_phase, "label") else str(_ns_phase_val)
                )
                if _ns_phase_val:
                    _phase_value = str(_ns_phase_val).strip()
                    _phase_name = str(_ns_phase_name or _phase_value).strip().upper()
                    _phase_label = str(_ns_phase_label or _phase_value).strip()
                    self._cached_night_phase = type(
                        "NightPhaseCompat",
                        (),
                        {
                            "value": _phase_value,
                            "name": _phase_name,
                            "label": lambda _self, _label=_phase_label: _label or "清醒",
                            "__repr__": lambda _s, _phase=_phase_value: f"NightPhase({_phase})",
                        },
                    )()
                    self._cached_night_summary["phase"] = _phase_value
                    self._cached_night_summary["phase_label"] = _phase_label or _phase_value
                    self._cached_night_summary["is_night"] = _phase_value.lower() != "awake"
                self._cached_night_summary["sleep_debt"] = float(getattr(_ns, "sleep_debt", 0.0) or 0.0)
                self._cached_night_summary["overnight_pressure"] = float(
                    getattr(_ns, "overnight_pressure", 0.0) or 0.0
                )
                self._cached_night_summary["drowsiness"] = float(getattr(_ns, "drowsiness_value", 0.0) or 0.0)
                if isinstance(_night_behavior, dict) and _night_behavior:
                    self._cached_night_summary["sleep_debt"] = float(
                        _night_behavior.get("sleep_debt", self._cached_night_summary["sleep_debt"]) or 0.0
                    )
                    self._cached_night_summary["overnight_pressure"] = float(
                        _night_behavior.get("overnight_pressure", self._cached_night_summary["overnight_pressure"])
                        or 0.0
                    )
                    self._cached_night_summary["drowsiness"] = float(
                        _night_behavior.get("drowsiness_value", self._cached_night_summary["drowsiness"]) or 0.0
                    )
                    self._cached_night_summary["response_suppression"] = float(
                        _night_behavior.get("response_suppression", 0.0) or 0.0
                    )
                    self._cached_night_summary["body_state_tag"] = str(
                        _night_behavior.get("body_state_tag", "") or ""
                    )
                    self._cached_night_summary["monologue_mood_hint"] = str(
                        _night_behavior.get("monologue_mood_hint", "") or ""
                    )
                    for _key in (
                        "time_band",
                        "time_band_label",
                        "time_band_description",
                        "current_hour",
                        "pressure_breakdown",
                        "mechanism_windows",
                        "is_sleep_window",
                        "is_night_social_window",
                        "is_pressure_window",
                        "system_started_at",
                        "last_evaluated_at",
                        "sync_label",
                        "sync_source",
                        "sleep_reserve",
                        "collapse_imminent",
                        "composite_load",
                    ):
                        if _key in _night_behavior:
                            self._cached_night_summary[_key] = _night_behavior[_key]
                    self._cached_night_summary["is_night"] = bool(
                        _night_behavior.get("is_night", self._cached_night_summary.get("is_night", False))
                    )
                if not self._cached_night_summary.get("phase_label"):
                    self._cached_night_summary["phase_label"] = str(_ns_phase_val)
                _is_burnthrough = bool(
                    getattr(_ns, "is_burnthrough_active", False)
                    or str(_ns_phase_val).upper() == "BURNED_OUT"
                )
                if _is_burnthrough:
                    self._cached_night_summary["is_burnthrough"] = True
                self._cached_night_summary["expression_style"] = self._derive_night_expression_style(
                    self._cached_night_summary
                )
                if _is_burnthrough:
                    return "burned_out"
            except Exception as _exc:
                logger.debug(f"{self.log_prefix} NightCycle熬穿态读取异常: {_exc}")
            _phase_lower = str(self._cached_night_summary.get("phase", "") or "").strip().lower()
            if _phase_lower == "drowsy":
                return "drowsy"
            if _phase_lower in {"light_sleep", "deep_sleep"}:
                return "sleep"
            if not _phase_lower and _night_mode.phase == "DROWSY":
                return "drowsy"
            if not _phase_lower and _night_mode.phase == "DEEP_VALLEY":
                return "sleep"
            if _night_mode.prob_multiplier <= 0.15 and _night_mode.ceiling_penalty >= 0.7:
                return "burned_out"
        except Exception as exc:
            logger.debug(f"{self.log_prefix} D6夜间评估异常: {exc}")
        return None

    def _record_user_cognitive(self, user_id: str, content: str, importance: float = 0.5, category: str = "general"):
        """用户认知画像记录：记住每个人的特点和偏好"""
        try:
            from src.memory_system.user_cognitive_store import get_cognitive_store

            _store = get_cognitive_store()
            _store.add_fragment(user_id, content, importance, category)
        except ImportError:
            pass
        except Exception as _e:
            logger.debug(f"{self.log_prefix} 认知记录异常: {_e}")

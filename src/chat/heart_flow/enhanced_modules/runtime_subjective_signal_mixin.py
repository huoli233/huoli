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

class RuntimeSubjectiveSignalMixin:
    def _evaluate_subjective_time_flow(self, now: float) -> Dict[str, float]:
        """GAP-C：主观时间流层 —— 心理新鲜度/多久没看群/主观时间距离感知

        三个核心指标：
          ① psychological_freshness（心理新鲜度）
             - 刚参与过群聊时=1.0，随时间衰减
             - 衰减速度受无聊值和群环境调制
          ② time_since_last_look（距离上次观看的客观时间）
             - 直接记录秒数，但输出主观感知倍数
          ③ subjective_time_distance（主观时间距离）
             - 短时间没看 → 感觉很久（加速感知）
             - 长时间持续 → 反而感觉正常（习惯化）
        """
        _last_active = getattr(self, "_last_active_watch_time", now)
        _elapsed = max(0.0, now - _last_active)
        _boredom = 0.0
        if self._cached_metabolism_constraints:
            _boredom = float(self._cached_metabolism_constraints.get("boredom", 0.0) or 0.0)
        _bored_ratio = max(0.0, min(1.0, _boredom / 100.0))
        # 新鲜度：指数衰减 + 无聊加速
        _base_half_life = 1800.0
        if _bored_ratio > 0.50:
            _base_half_life *= 1.0 - (_bored_ratio - 0.50) * 0.5
        elif _bored_ratio < 0.20:
            _base_half_life *= 1.3
        _freshness = math.exp(-0.693 * _elapsed / max(60.0, _base_half_life))
        # 主观时间距离：短时加速 / 长时减速（U型曲线）
        if _elapsed < 120:
            _subj_dist = _elapsed / 60.0
        elif _elapsed < 900:
            _subj_dist = 2.0 + (_elapsed - 120) / 150.0
        elif _elapsed < 3600:
            _subj_dist = 7.2 + (_elapsed - 900) / 400.0
        else:
            _hours = _elapsed / 3600.0
            _subj_dist = min(24.0, 15.2 + math.log1p(_hours))
        return {
            "psychological_freshness": round(_freshness, 4),
            "time_since_last_look_sec": round(_elapsed, 1),
            "subjective_time_distance": round(_subj_dist, 2),
            "freshness_half_life": round(_base_half_life, 1),
        }

    def _evaluate_visibility_decay_and_reactivation(
        self, now: float, incoming_batch: List, pinged_msg=None
    ) -> Dict[str, Any]:
        """GAP-D：可见性衰减与回看激活 —— 四级衰减速度 + 四类激活触发

        四级衰减速度（对每条消息的可见性随时间推移）：
          Level 0: 常速衰减（正常活跃群）—— 半衰期 ~15分钟
          Level 1: 快速衰减（低价值消息）—— 半衰期 ~8分钟
          Level 2: 极速衰减（高噪音环境）—— 半衰期 ~3分钟
          Level 3: 冻结衰减（黑屏态）—— 不衰减也不增长

        四类回看激活触发：
          Type A: @提及或引用机器人 → 强制激活到最高可见性
          Type B: 高相关事件（讨论机器人/熟悉用户关键发言）→ 大幅提升
          Type C: 熟悉用户出现 → 中等提升
          Type D: 异常热闹（突然多人活跃）→ 弱提升
        """
        _result: Dict[str, Any] = {
            "decay_level": 0,
            "reactivation_type": None,
            "reactivation_strength": 0.0,
        }
        # ── 确定当前衰减等级 ──
        _watch_val = "unknown"
        if self._cached_watch_level is not None:
            _watch_val = (
                self._cached_watch_level.value
                if hasattr(self._cached_watch_level, "value")
                else str(self._cached_watch_level)
            )
        if _watch_val == "blackout":
            _result["decay_level"] = 3
        else:
            _noise_factor = 1.0
            if self._cached_participant_summary:
                _total_users = int(self._cached_participant_summary.get("hot_count", 0) or 0)
                _total_users += int(self._cached_participant_summary.get("warm_count", 0) or 0)
                if _total_users >= 10:
                    _noise_factor = 2.5
                elif _total_users >= 6:
                    _noise_factor = 1.8
                elif _total_users <= 2:
                    _noise_factor = 0.65
            if self._cached_multimodal_summary:
                _storm = str(self._cached_multimodal_summary.get("storm_mode", "") or "")
                if _storm == "active":
                    _noise_factor += 1.2
            if _noise_factor > 2.0:
                _result["decay_level"] = 2
            elif _noise_factor > 1.3:
                _result["decay_level"] = 1
            else:
                _result["decay_level"] = 0
        # ── 回看激活检测 ──
        if pinged_msg is not None:
            _result["reactivation_type"] = "A_ping_or_quote"
            _result["reactivation_strength"] = 1.0
        elif self._cached_self_references:
            _max_ref_str = 0.0
            for _ref in self._cached_self_references:
                _rs = float(getattr(_ref, "strength", 0.0) or 0.0)
                if _rs > _max_ref_str:
                    _max_ref_str = _rs
                _rt = getattr(_ref, "ref_type", None)
                if _rt:
                    _rv = _rt.value if hasattr(_rt, "value") else str(_rt)
                    if _rv in ("direct_at", "discussed_as_topic"):
                        _result["reactivation_type"] = "B_high_relevance"
                        _result["reactivation_strength"] = min(0.88, 0.55 + _max_ref_str * 0.40)
            if _result["reactivation_type"] is None and _max_ref_str > 0.35:
                _result["reactivation_type"] = "B_moderate_ref"
                _result["reactivation_strength"] = 0.35 + _max_ref_str * 0.25
        if _result["reactivation_type"] is None:
            _rel_snap_d = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
            _aff_d = float(_rel_snap_d.get("affection", 0.0) or 0.0)
            if self._metric_has_signal(_aff_d) and _aff_d > 55 and incoming_batch:
                _has_known_user = False
                for _msg in incoming_batch:
                    _uid = str(getattr(_msg, "user_id", "") or "").strip()
                    if self._is_human_message_obj(_msg):
                        _has_known_user = True
                        break
                if _has_known_user:
                    _result["reactivation_type"] = "C_familiar_user"
                    _result["reactivation_strength"] = 0.22 + (_aff_d - 55.0) / 200.0
        if _result["reactivation_type"] is None:
            if self._cached_participant_summary:
                _sudden_burst = int(self._cached_participant_summary.get("sudden_burst_count", 0) or 0)
                if _sudden_burst >= 3:
                    _result["reactivation_type"] = "D_abnormal_activity"
                    _result["reactivation_strength"] = min(0.30, _sudden_burst * 0.06)
        _result["decay_half_life_sec"] = [900.0, 480.0, 180.0, float("inf")][_result["decay_level"]]
        return _result

    def _evaluate_subjective_situation_interpretation(self) -> Dict[str, Any]:
        """GAP-L：主观情境解释层 —— 行为前的主观判断中间层

        在所有客观数据收集完成后、决策执行之前，
        机器人先做一个"主观判断"：
          - "我觉得现在群里的氛围是什么？"
          - "我现在的心理状态适合参与吗？"
          - "如果我要回应，我应该用什么姿态？"

        这一层输出不直接决定是否回复，
        但会强烈影响 planner 的 prompt 构建和风格选择。

        输出四个维度：
          ① situation_reading（情境解读）：对当前场景的主观定性
          ② internal_state_reading（内部状态）：机器人自身状态的自我感知
          ③ engagement_recommendation（参与建议）：推荐的行为姿态
          ④ risk_assessment（风险评估）：潜在风险点
        """
        _result: Dict[str, Any] = {
            "situation_reading": {
                "type": "unknown",
                "confidence": 0.0,
                "description": "",
            },
            "internal_state_reading": {
                "mood": "neutral",
                "energy": "normal",
                "social_comfort": 0.5,
            },
            "engagement_recommendation": {
                "posture": "observe",
                "intensity": 0.3,
                "style_hint": "",
            },
            "risk_assessment": {
                "level": "low",
                "factors": [],
                "overall_score": 0.0,
            },
        }
        # ── 情境解读 ──
        _atmo_type = ""
        _atmo_score = 0.5
        if self._cached_pattern_evidence:
            for _pe in self._cached_pattern_evidence:
                _pc = float(getattr(_pe, "confidence", 0.0) or 0.0)
                if _pc > _atmo_score:
                    _atmo_score = _pc
                    _pe_pat = getattr(_pe, "pattern", None)
                    if _pe_pat:
                        _atmo_type = _pe_pat.value if hasattr(_pe_pat, "value") else str(_pe_pat)
        _hot_users = 0
        if self._cached_participant_summary:
            _hot_users = int(self._cached_participant_summary.get("hot_count", 0) or 0)
            _hot_users += int(self._cached_participant_summary.get("warm_count", 0) or 0)
        _sit_type = "normal_chatter"
        _sit_desc = "普通闲聊氛围"
        _sit_conf = 0.5
        if _atmo_type in (
            "heated_discussion",
            "conflict_escalation",
            "argument",
        ):
            _sit_type = "tense_confrontation"
            _sit_desc = "紧张对峙氛围，情绪可能激烈"
            _sit_conf = min(0.92, _atmo_score + 0.15)
        elif _atmo_type in (
            "newcomer_welcome",
            "birthday_wish",
            "congratulation",
            "celebration_wave",
            "support_circle",
        ):
            _sit_type = "celebratory"
            _sit_desc = "庆祝/欢迎氛围，适合积极互动"
            _sit_conf = min(0.90, _atmo_score + 0.10)
        elif _atmo_type in ("spectator_mode",):
            _sit_type = "spectator_scene"
            _sit_desc = "围观/吃瓜氛围，可选择性参与"
            _sit_conf = min(0.85, _atmo_score + 0.08)
        elif _hot_users <= 2 and _atmo_score < 0.3:
            _sit_type = "quiet_sparse"
            _sit_desc = "安静稀疏氛围，参与门槛较高"
            _sit_conf = max(0.25, _atmo_score * 0.7)
        elif _hot_users >= 8:
            _sit_type = "crowded_noisy"
            _sit_desc = "拥挤嘈杂氛围，信息过载"
            _sit_conf = min(0.88, 0.5 + _hot_users * 0.04)
        _result["situation_reading"] = {
            "type": _sit_type,
            "confidence": round(_sit_conf, 3),
            "description": _sit_desc,
            "active_user_count": _hot_users,
            "dominant_pattern": _atmo_type,
        }
        # ── 内部状态读取 ──
        _mood = "neutral"
        _energy_label = "normal"
        _soc_comfort = 0.5
        if self._cached_presence_state:
            _sw_val = float(getattr(self._cached_presence_state, "social_willingness", 0.5) or 0.5)
            _aw_val = float(getattr(self._cached_presence_state, "avoidance_tendency", 0.0) or 0.0)
            _qp_val = float(getattr(self._cached_presence_state, "quiet_preference", 0.2) or 0.2)
            _soc_comfort = _sw_val * 0.55 + (1.0 - _aw_val) * 0.30 + (1.0 - _qp_val) * 0.15
            if _sw_val > 0.65 and _aw_val < 0.20:
                _mood = "sociable_eager"
            elif _sw_val > 0.45 and _aw_val < 0.35:
                _mood = "open_friendly"
            elif _aw_val > 0.50:
                _mood = "withdrawn_defensive"
            elif _qp_val > 0.55:
                _mood = "quiet_contemplative"
            elif _sw_val < 0.20:
                _mood = "isolated_reluctant"
        if self._cached_metabolism_constraints:
            _chat_v = float(self._cached_metabolism_constraints.get("chat_value", 50.0) or 50.0)
            if _chat_v > 60:
                _energy_label = "energetic"
            elif _chat_v > 25:
                _energy_label = "moderate"
            elif _chat_v > 10:
                _energy_label = "tired"
            else:
                _energy_label = "exhausted"
        if self._cached_night_phase:
            _phase = self._cached_night_phase
            _phase_name = getattr(_phase, "name", str(_phase)) if _phase else ""
            if "DEEP" in _phase_name or "VALLEY" in _phase_name or _phase_name == "DEEP_VALLEY":
                _mood = f"{_mood}_asleep"
                _energy_label = "deep_asleep"
            elif "DROWSY" in _phase_name or "LIGHT" in _phase_name or "BURNED" in _phase_name:
                _mood = f"{_mood}_drowsy"
                _energy_label = "drowsy"
        _result["internal_state_reading"] = {
            "mood": _mood,
            "energy": _energy_label,
            "social_comfort": round(_soc_comfort, 3),
        }
        # ── 参与建议 ──
        _posture = "observe"
        _intensity = 0.30
        _style_hint = ""
        if _soc_comfort > 0.65 and _energy_label in ("energetic", "moderate"):
            _posture = "engage_active"
            _intensity = 0.72
            _style_hint = "积极展开，可以主动引导话题"
        elif _soc_comfort > 0.45 and _energy_label != "exhausted":
            _posture = "engage_passive"
            _intensity = 0.50
            _style_hint = "被动响应，等被cue到再回"
        elif _soc_comfort > 0.25 and _energy_label not in (
            "exhausted",
            "deep_asleep",
        ):
            _posture = "observe_selective"
            _intensity = 0.32
            _style_hint = "选择性关注，只回高优先级"
        else:
            _posture = "avoid_minimal"
            _intensity = 0.12
            _style_hint = "最小化参与，能不回就不回"
        if _sit_type == "celebratory":
            _posture = "engage_warm"
            _intensity = min(0.85, _intensity + 0.18)
            _style_hint = "温暖融入，表达祝福/祝贺"
        elif _sit_type == "tense_confrontation" and _mood not in ("withdrawn_defensive",):
            _posture = "observe_cautious"
            _intensity = max(0.15, _intensity - 0.15)
            _style_hint = "谨慎观察，避免卷入冲突"
        _result["engagement_recommendation"] = {
            "posture": _posture,
            "intensity": round(_intensity, 3),
            "style_hint": _style_hint,
        }
        # ── 风险评估 ──
        _risks: List[str] = []
        _risk_score = 0.0
        if self._cached_user_negative_emotion > 60:
            _risks.append("user_negative_emotion_high")
            _risk_score += 0.22
        if self._cached_presence_state and _aw_val > 0.6:
            _risks.append("high_avoidance_tendency")
            _risk_score += 0.15
        if _sit_type == "tense_confrontation":
            _risks.append("tense_environment")
            _risk_score += 0.18
        if _hot_users >= 8:
            _risks.append("information_overload")
            _risk_score += 0.08
        _trauma_risk = 0.0
        _rel_snap_l = getattr(self, "_last_relation_snapshot", None) or {}
        _trauma_val = float(_rel_snap_l.get("trauma_score", 0.0) or 0.0)
        if _trauma_val > 40:
            _risks.append(f"trauma_active({_trauma_val:.0f})")
            _trauma_risk = min(0.28, _trauma_val / 200.0)
            _risk_score += _trauma_risk
        _result["risk_assessment"] = {
            "level": ("high" if _risk_score > 0.40 else "medium" if _risk_score > 0.18 else "low"),
            "factors": _risks,
            "overall_score": round(min(1.0, _risk_score), 3),
        }
        return _result

    def _check_background_event_schedule(self) -> Optional[Dict[str, Any]]:
        """GAP-B：后台随机事件调度器 —— 检查是否有定时事件应触发

        事件类型：
          - proactive_greeting（主动打招呼）
          - random_observation（随机观察评论）
          - mood_expression（心情表达）
          - memory_recall（记忆触发式发言）

        调度规则：
          - 每日种子确保同一天内事件分布不重复
          - 模板池从预定义事件中选择
          - 时间分布遵循活跃曲线（避免深夜高频）
          - 能量预算限制每日总事件数
        """
        now = time.time()
        _last_sched = getattr(self, "_last_background_event_check", 0.0)
        if (now - _last_sched) < 300.0:
            return None
        self._last_background_event_check = now
        # 日种子（每天重置一次）
        _today_seed = int(now / 86400.0)
        _daily_key = f"bg_{self.stream_id}_{_today_seed}"
        _day_hash = hash(_daily_key) & 0x7FFFFFFF
        _rng.seed(_day_hash)
        # 能量预算检查
        _budget_used = getattr(self, "_bg_event_budget_used_today", 0)
        _budget_max = 8
        if _budget_used >= _budget_max:
            return None
        # 时间窗口检查
        _hour = time.localtime(now).tm_hour
        if 2 <= _hour <= 6:
            return None
        _time_weight = 1.0
        if 9 <= _hour <= 11:
            _time_weight = 1.3
        elif 19 <= _hour <= 22:
            _time_weight = 1.15
        elif 13 <= _hour <= 17:
            _time_weight = 1.0
        elif 7 <= _hour <= 8 or 23 <= _hour:
            _time_weight = 0.5
        else:
            _time_weight = 0.2
        # 事件概率计算
        _event_roll = _rng.random() * _time_weight
        _threshold_base = 0.08
        if self._cached_night_phase:
            _np_name = (
                getattr(self._cached_night_phase, "name", str(self._cached_night_phase))
                if self._cached_night_phase
                else ""
            )
            if "DEEP" in _np_name or "DROWSY" in _np_name or "LIGHT" in _np_name:
                _threshold_base *= 0.15
        if _event_roll < _threshold_base:
            return None
        # 选择事件类型
        _event_type_weights = [
            ("proactive_greeting", 0.25),
            ("random_observation", 0.30),
            ("mood_expression", 0.20),
            ("memory_recall", 0.15),
            ("topic_resurrection", 0.10),
        ]
        _total_w = sum(w for _, w in _event_type_weights)
        _r = _rng.random() * _total_w
        _cumul = 0.0
        _chosen_type = "random_observation"
        for _et, _ew in _event_type_weights:
            _cumul += _ew
            if _r <= _cumul:
                _chosen_type = _et
                break
        # 生成事件
        self._bg_event_budget_used_today = _budget_used + 1
        _event = {
            "type": _chosen_type,
            "scheduled_at": now,
            "priority": round(_rng.uniform(0.15, 0.55), 3),
            "seed_hash": _day_hash,
            "time_weight": round(_time_weight, 3),
            "budget_remaining": _budget_max - self._bg_event_budget_used_today,
        }
        logger.debug(f"{self.log_prefix} 后台事件调度: {_chosen_type}(p={_event['priority']})")
        return _event

    def _evaluate_self_learning_triggers(self) -> Dict[str, Any]:
        """GAP-G：自学习机制闭环 —— 学习触发检测 + 结果落地评估

        学习触发条件（任一满足即触发学习记录）：
          T1: 决策结果与预期显著偏离（回复后用户不再说话 / 不回后用户追问）
          T2: 遇到全新的交互模式（新话题/新人/新群氛围）
          T3: 连续多次同类决策被否定（反复尝试但效果不好）
          T4: 关系信号发生阶跃变化（好感突增/突降）

        返回学习触发状态和推荐的学习动作
        """
        _result: Dict[str, Any] = {
            "should_learn": False,
            "trigger_types": [],
            "learning_action": None,
            "confidence": 0.0,
        }
        # T1: 决策偏离检测
        _silence_after_reply = 0.0
        if self._cached_participant_summary:
            _silence_after_reply = float(self._cached_participant_summary.get("silence_since_last_bot", 0.0) or 0.0)
        if self._last_bot_reply_ts > 0:
            _elapsed_silence = time.time() - self._last_bot_reply_ts
            if _elapsed_silence > 300 and _silence_after_reply > _elapsed_silence * 0.8:
                _result["should_learn"] = True
                _result["trigger_types"].append("T1_no_followup")
                _result["confidence"] += 0.35
        # T2: 新模式检测
        _pattern_changed = False
        if self._cached_pattern_evidence:
            _top_pat = None
            _top_pc = 0.0
            for _pe in self._cached_pattern_evidence:
                _pc = float(getattr(_pe, "confidence", 0.0) or 0.0)
                if _pc > _top_pc:
                    _top_pc = _pc
                    _top_pat = _pe
            if _top_pat and _top_pc > 0.55:
                _pat_enum = getattr(_top_pat, "pattern", None)
                if _pat_enum:
                    _pv = _pat_enum.value if hasattr(_pat_enum, "value") else str(_pat_enum)
                    _prev_patterns = getattr(self, "_recent_pattern_history", [])
                    if _pv not in _prev_patterns[-5:] if _prev_patterns else []:
                        _result["should_learn"] = True
                        _result["trigger_types"].append(f"T2_new_pattern:{_pv}")
                        _result["confidence"] += 0.28
                        _pattern_changed = True
        # T3: 连续否定检测
        _legacy_hits = getattr(self, "_legacy_constraint_hits", 0)
        if _legacy_hits >= 3:
            _result["should_learn"] = True
            _result["trigger_types"].append("T3_repeated_constraints")
            _result["confidence"] += min(0.40, _legacy_hits * 0.10)
        # T4: 关系阶跃检测
        _rel_snap_g = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
        _aff_now = float(_rel_snap_g.get("affection", 0.0) or 0.0)
        _aff_prev = float(getattr(self, "_prev_affection_for_learning", _aff_now) or _aff_now)
        _aff_delta = abs(_aff_now - _aff_prev)
        if _aff_delta > 15.0 and (self._metric_has_signal(_aff_now) or self._metric_has_signal(_aff_prev)):
            _result["should_learn"] = True
            _direction = "up" if _aff_now > _aff_prev else "down"
            _result["trigger_types"].append(f"T4_affection_jump:{_direction}:{_aff_delta:.0f}")
            _result["confidence"] += 0.22
        self._prev_affection_for_learning = _aff_now
        # 推荐学习动作
        if _result["should_learn"]:
            if "T1_no_followup" in _result["trigger_types"]:
                _result["learning_action"] = "reduce_engagement_threshold"
            elif _pattern_changed:
                _result["learning_action"] = "record_pattern_response"
            elif "T3_repeated_constraints" in _result["trigger_types"]:
                _result["learning_action"] = "adjust_base_weights"
            elif any("T4" in t for t in _result["trigger_types"]):
                _result["learning_action"] = "update_relation_model"
            else:
                _result["learning_action"] = "general_observation"
            _result["confidence"] = min(1.0, _result["confidence"])
        return _result

    def _evaluate_local_social_world(self) -> Dict[str, Any]:
        """GAP-H：局部社会世界模型 —— 热缓存/温缓存/线程识别/对象锁定

        三层缓存：
          - 热缓存（20人）：最近活跃/高互动频率的用户，完整数据
          - 温缓存（50人）：偶尔出现的用户，精简数据
          - 冷存储（无限）：历史用户，仅ID+最后活跃时间

        线程识别：追踪"谁在回谁"，维护子线程归属图
        对象锁定：当前注意力焦点（正在关注谁/什么话题）
        """
        _result: Dict[str, Any] = {
            "hot_cache_size": 0,
            "warm_cache_size": 0,
            "active_threads": [],
            "locked_target": None,
            "locked_topic": None,
            "attention_focus_strength": 0.0,
        }
        if not self._cached_participant_summary:
            return _result
        _ps = self._cached_participant_summary
        _hot_count = int(_ps.get("hot_count", 0) or 0)
        _warm_count = int(_ps.get("warm_count", 0) or 0)
        _result["hot_cache_size"] = min(20, _hot_count)
        _result["warm_cache_size"] = min(50, _warm_count)
        # 线程识别：从话题归属推断子线程
        try:
            _scene_h = self._orch.get("scene_state")
            _topics_h = _scene_h.active_topic_slots(limit=5)
            _threads = []
            for _t in _topics_h:
                _owners = list(getattr(_t, "contributors", []) or [])[:4]
                if len(_owners) >= 2:
                    _threads.append(
                        {
                            "topic": str(getattr(_t, "keyword", "") or "")[:60],
                            "ownership": _t.ownership_type(),
                            "participants": _owners,
                            "age_sec": int(_t.age_seconds()),
                        }
                    )
            _result["active_threads"] = _threads
        except Exception as _e:
            logger.warning(f"{self.log_prefix} 活跃线程状态解析异常: {_e}")
        # 对象锁定：从自我引用和关系快照推断
        if self._cached_self_references:
            _strongest_ref_user = None
            _max_ref_score = 0.0
            for _ref in self._cached_self_references:
                _rs = float(getattr(_ref, "strength", 0.0) or 0.0)
                if _rs > _max_ref_score:
                    _max_ref_score = _rs
                    _ref_user = getattr(_ref, "target_user", None) or getattr(_ref, "user_id", None)
                    if _ref_user:
                        _strongest_ref_user = str(_ref_user).strip()
            if _strongest_ref_user and _max_ref_score > 0.35:
                _result["locked_target"] = _strongest_ref_user
                _result["attention_focus_strength"] = round(_max_ref_score, 3)
        # 锁定话题
        if self._cached_pattern_evidence:
            for _pe in self._cached_pattern_evidence:
                _pe_conf = float(getattr(_pe, "confidence", 0.0) or 0.0)
                if _pe_conf > 0.55:
                    _pe_pat = getattr(_pe, "pattern", None)
                    if _pe_pat:
                        _result["locked_topic"] = _pe_pat.value if hasattr(_pe_pat, "value") else str(_pe_pat)
                        break
        return _result


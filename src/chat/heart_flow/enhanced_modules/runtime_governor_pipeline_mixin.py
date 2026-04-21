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

class RuntimeGovernorPipelineMixin:
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

    def _apply_governor_watch_transition(
        self,
        *,
        current_watch: str,
        incoming_batch: List,
        has_ping: bool,
    ) -> Optional[str]:
        """在旧 watch 表之前先执行一层统一姿态跳转。

        这层只处理高置信度的“该休息 / 该退回窥屏 / 该从窥屏升到关注”，
        避免后面的手搓条件继续和 BehaviorGovernor / RestGovernor 打架。
        """
        direct_target = self._has_targeted_bot_message(incoming_batch)
        rest_verdict = self._evaluate_rest_governor(
            incoming_batch=incoming_batch,
            source="reactive",
            targeted_to_bot=direct_target,
            admin_force=bool(getattr(self, "_is_admin_forced", False)),
            pinged_msg=(incoming_batch[-1] if has_ping and incoming_batch else None),
        )
        behavior_verdict = self._evaluate_behavior_governor(
            incoming_batch=incoming_batch,
            silence_sec=float(getattr(self, "_last_silence_sec", 0.0) or 0.0),
            requested_mode="reactive",
            target_message=self._get_latest_human_message(incoming_batch),
            is_background=False,
        )

        target_watch: Optional[WatchLevel] = None
        target_phase: Optional[FlowPhase] = None
        watch_reason = ""
        phase_reason = ""

        if rest_verdict.should_rest and not direct_target and not has_ping:
            if current_watch in {"engaged", "active_watch"}:
                target_watch = WatchLevel.BLACKOUT
                target_phase = FlowPhase.DORMANT
            elif current_watch in {"skim_window", "peek"}:
                target_watch = WatchLevel.PEEK
                target_phase = FlowPhase.STANDBY
            if target_watch is not None:
                watch_reason = f"watch:rest_governor:{rest_verdict.posture}"
                phase_reason = f"RestGovernor收口:{self._summarize_rest_governor(rest_verdict)}"
        elif rest_verdict.interruption_policy == "peek_only" and not direct_target and not has_ping:
            if current_watch in {"engaged", "active_watch", "skim_window"}:
                target_watch = WatchLevel.PEEK
                target_phase = FlowPhase.STANDBY
                watch_reason = f"watch:rest_governor:peek_only:{rest_verdict.posture}"
                phase_reason = f"RestGovernor退回窥屏:{self._summarize_rest_governor(rest_verdict)}"
        elif behavior_verdict.interrupt_level == "engage":
            if current_watch == "blackout":
                target_watch = WatchLevel.PEEK
                target_phase = FlowPhase.STANDBY
                watch_reason = f"watch:behavior_governor:blackout_to_peek:{behavior_verdict.reply_mode}"
                phase_reason = f"BehaviorGovernor拉起窥屏:{self._summarize_behavior_governor(behavior_verdict)}"
            elif current_watch in {"peek", "skim_window"}:
                target_watch = WatchLevel.ACTIVE_WATCH
                target_phase = FlowPhase.STANDBY
                watch_reason = f"watch:behavior_governor:to_active:{behavior_verdict.reply_mode}"
                phase_reason = f"BehaviorGovernor拉起关注:{self._summarize_behavior_governor(behavior_verdict)}"
            elif current_watch == "active_watch" and behavior_verdict.reply_mode == "reply":
                target_watch = WatchLevel.ENGAGED
                target_phase = FlowPhase.ENGAGED
                watch_reason = f"watch:behavior_governor:to_engaged:{behavior_verdict.reply_mode}"
                phase_reason = f"BehaviorGovernor拉起参与:{self._summarize_behavior_governor(behavior_verdict)}"

        if target_watch is None:
            return None

        try:
            snapshot = self._build_unified_flow_snapshot("watch_governor")
            snapshot.watch_state = target_watch.value
            snapshot.phase = target_phase.value if target_phase is not None else snapshot.phase
            if target_watch == WatchLevel.BLACKOUT:
                snapshot.blocker = phase_reason
            self._apply_unified_flow_snapshot(
                snapshot,
                watch_reason=watch_reason,
                phase_reason=phase_reason,
            )
            return f"{current_watch}->{target_watch.value}({watch_reason})"
        except Exception as exc:
            logger.debug(f"{self.log_prefix} governor观看态跳转异常: {exc}")
            return None

    def _compute_governor_watch_cap(
        self,
        *,
        behavior_verdict: BehaviorGovernorVerdict,
        rest_verdict: RestGovernorVerdict,
        direct_target: bool,
        has_ping: bool,
        recent_human_activity: bool,
    ) -> Tuple[int, str]:
        """给旧 watch 手工表加一层 Governor 限幅，避免后半段继续越级升级。"""
        _behavior_cap = int(getattr(behavior_verdict, "max_watch_rank", 1) or 1)
        if rest_verdict.should_rest and not direct_target and not has_ping:
            if rest_verdict.interruption_policy == "block":
                return (0, f"RestGovernor:block:{self._summarize_rest_governor(rest_verdict)}")
            return (1, f"RestGovernor:rest:{self._summarize_rest_governor(rest_verdict)}")
        if rest_verdict.interruption_policy == "peek_only" and not direct_target and not has_ping:
            return (1, f"RestGovernor:peek_only:{self._summarize_rest_governor(rest_verdict)}")
        if not recent_human_activity and behavior_verdict.silence_policy == "silent":
            _behavior_cap = min(_behavior_cap, 1)
        if behavior_verdict.interrupt_level == "ignore":
            if behavior_verdict.silence_policy == "silent" and not recent_human_activity:
                return (0, f"BehaviorGovernor:ignore_silent:{self._summarize_behavior_governor(behavior_verdict)}")
        _behavior_cap = max(0, min(4, _behavior_cap))
        return (_behavior_cap, f"BehaviorGovernor:cap={_behavior_cap}:{self._summarize_behavior_governor(behavior_verdict)}")

    def _apply_governor_watch_cap(
        self,
        *,
        current_watch: str,
        cap_rank: int,
        reason: str,
    ) -> Optional[str]:
        current_enum = next(
            (level for level in WatchLevel if level.value == current_watch),
            WatchLevel.PEEK,
        )
        if current_enum.rank() <= cap_rank:
            return None

        target_watch = _WATCH_LEVEL_BY_RANK.get(int(cap_rank), WatchLevel.PEEK)
        target_phase = FlowPhase.ENGAGED if target_watch == WatchLevel.ENGAGED else (
            FlowPhase.DORMANT if target_watch == WatchLevel.BLACKOUT else FlowPhase.STANDBY
        )
        try:
            snapshot = self._build_unified_flow_snapshot("watch_governor_cap")
            snapshot.watch_state = target_watch.value
            snapshot.phase = target_phase.value
            if target_watch == WatchLevel.BLACKOUT:
                snapshot.blocker = reason
            self._apply_unified_flow_snapshot(
                snapshot,
                watch_reason=f"watch:governor_cap:{current_watch}_to_{target_watch.value}",
                phase_reason=f"Governor限幅:{reason}",
            )
            return f"{current_watch}->{target_watch.value}(governor_cap:{cap_rank})"
        except Exception as exc:
            logger.debug(f"{self.log_prefix} governor观看态限幅异常: {exc}")
            return None

    def _should_promote_peek_reflection(
        self,
        *,
        incoming_batch: List,
        peek_verdict: Any,
        pinged_msg: Any = None,
    ) -> Tuple[bool, str]:
        """统一裁决“窥屏反思后要不要真的升级参与”。"""
        if peek_verdict is None or not getattr(peek_verdict, "is_valid", False):
            return (False, "peek_invalid")

        direct_target = bool(pinged_msg is not None or self._has_targeted_bot_message(incoming_batch))
        rest_verdict = self._evaluate_rest_governor(
            incoming_batch=incoming_batch,
            source="peek",
            targeted_to_bot=direct_target,
            admin_force=bool(getattr(self, "_is_admin_forced", False)),
            pinged_msg=pinged_msg,
        )
        behavior_verdict = self._evaluate_behavior_governor(
            incoming_batch=incoming_batch,
            silence_sec=float(getattr(self, "_last_silence_sec", 0.0) or 0.0),
            requested_mode="reactive",
            target_message=self._get_latest_human_message(incoming_batch),
            is_background=False,
        )

        desire = int(float(getattr(peek_verdict, "reply_desire_level", 0) or 0))
        action = str(getattr(peek_verdict, "next_action", "") or "").strip().lower()
        explicit_upgrade = action in {"reply", "followup"}
        explicit_hold = action in {"observe", "wait", "skip", "rest", "defer"}
        behavior_summary = self._summarize_behavior_governor(behavior_verdict)
        rest_summary = self._summarize_rest_governor(rest_verdict)

        if rest_verdict.should_rest and rest_verdict.interruption_policy == "block" and not direct_target:
            return (False, f"RestGovernor阻断: {rest_summary}")
        if not behavior_verdict.allow_generation and not direct_target:
            return (False, f"BehaviorGovernor阻断生成: {behavior_summary}")
        if explicit_hold:
            return (False, f"模型选择{action}: {behavior_summary}")
        if direct_target and (explicit_upgrade or desire >= 4):
            return (True, f"direct_target: {behavior_summary}")
        if behavior_verdict.interrupt_level == "engage" and (explicit_upgrade or desire >= 6):
            return (True, f"engage: {behavior_summary}")
        if (
            explicit_upgrade
            and desire >= 8
            and behavior_verdict.interrupt_level == "skim"
            and rest_verdict.interruption_policy != "block"
        ):
            return (True, f"high_interest_override: {behavior_summary}")
        if (
            not action
            and desire >= 8
            and behavior_verdict.interrupt_level in {"skim", "engage"}
            and rest_verdict.interruption_policy == "allow"
        ):
            return (True, f"legacy_high_desire: {behavior_summary}")
        return (False, f"peek_governor_hold: {behavior_summary} | {rest_summary}")

    def _integrate_watch_state(self, now: float, has_messages: bool = False) -> None:
        """更新关注层级状态机"""
        try:
            _wm = self._orch.get("watch_machine")
            if has_messages:
                _wm.signal_escalate(trigger="新消息到达")
            else:
                _boredom = 0.0
                _silence = 0.0
                _snap = self._tick_world_snapshot
                if _snap:
                    _res = getattr(_snap, "self_resources", None)
                    if _res:
                        _boredom = float(getattr(_res, "boredom", 0.0) or 0.0)
                        _silence = float(getattr(_res, "silence_seconds", 0.0) or 0.0)
                _wm.evaluate_natural_decay(silence_sec=_silence, boredom=_boredom)
            self._cached_watch_level = _wm.current_level
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 关注状态机异常: {exc}")

    def _evaluate_watch_full_transitions(self, now: float, has_ping: bool, incoming_batch: List) -> Optional[str]:
        """GAP-F：观看状态机完整转移表 —— 8条转移路径的统一评估

        5态: blackout / peek / skim_window / active_watch / engaged
        8条转移路径:
          ① blackout → peek      (@提及/无聊↑/熟悉用户/异常热闹/高相关事件)
          ② peek → skim_window    (发现有趣话题/连续多条相关/看到线索/看到熟悉对象)
          ③ skim → active_watch   (显著度↑/线程强相关/多人持续互动/小模型判断值得)
          ④ active → engaged       (决定参与/线程成主线/事件影响强)
          ⑤ engaged → active      (参与后退回观察/线程无推进/先看别人接)
          ⑥ [任意] → blackout     (活跃值↓/持续低价值/过于嘈杂/疲劳↑/明确不想看)
          ⑦ engaged → blackout    (直接从卷入退到完全不看)
          ⑧ skim → peek           (看了没意思退回窥屏)

        返回转移描述字符串，或 None 表示无转移
        """
        try:
            thresholds = get_heartfc_thresholds()
            from src.core.watch_state_machine import (
                get_watch_machine,
                WatchLevel as WatchState,
            )

            _wm = get_watch_machine(self.stream_id)
            _current = _wm.current_level
            if _current is None:
                return None
            _cur_val = _current.value if hasattr(_current, "value") else str(_current)
            _human_batch_count = sum(1 for _m in (incoming_batch or []) if self._is_human_message_obj(_m))
            _governor_transition = self._apply_governor_watch_transition(
                current_watch=_cur_val,
                incoming_batch=incoming_batch,
                has_ping=has_ping,
            )
            if _governor_transition:
                return _governor_transition
            _direct_target = self._has_targeted_bot_message(incoming_batch)
            _recent_human_activity = _human_batch_count > 0 or self._has_recent_human_activity(300.0)
            _rest_governor = self._evaluate_rest_governor(
                incoming_batch=incoming_batch,
                source="reactive",
                targeted_to_bot=_direct_target,
                admin_force=bool(getattr(self, "_is_admin_forced", False)),
                pinged_msg=(incoming_batch[-1] if has_ping and incoming_batch else None),
            )
            _behavior_governor = self._evaluate_behavior_governor(
                incoming_batch=incoming_batch,
                silence_sec=float(getattr(self, "_last_silence_sec", 0.0) or 0.0),
                requested_mode="reactive",
                target_message=self._get_latest_human_message(incoming_batch),
                is_background=False,
            )
            _governor_cap_rank, _governor_cap_reason = self._compute_governor_watch_cap(
                behavior_verdict=_behavior_governor,
                rest_verdict=_rest_governor,
                direct_target=_direct_target,
                has_ping=has_ping,
                recent_human_activity=_recent_human_activity,
            )
            _governor_cap_transition = self._apply_governor_watch_cap(
                current_watch=_cur_val,
                cap_rank=_governor_cap_rank,
                reason=_governor_cap_reason,
            )
            if _governor_cap_transition:
                return _governor_cap_transition
            # ── 收集转移决策所需的全部上下文 ──
            _boredom = 0.0
            _env_fatigue = 0.0
            _fatigue = 0.0
            _activity = 50.0
            _silence = 0.0
            _snap = self._tick_world_snapshot
            if _snap:
                _res = getattr(_snap, "self_resources", None)
                if _res:
                    _boredom = float(getattr(_res, "boredom", 0.0) or 0.0)
                    _activity = float(getattr(_res, "activity_level", 50.0) or 50.0)
                    _silence = float(getattr(_res, "silence_seconds", 0.0) or 0.0)
            _emo = getattr(self, "_cached_emotion_state", None) or {}
            if isinstance(_emo, dict):
                _env_fatigue = float(_emo.get("environmental_fatigue", 0.0) or 0.0) * 100.0
            try:
                from src.core.night_cycle_system import get_night_cycle

                _ncs_state = get_night_cycle(self.stream_id).state_snapshot
                _fatigue = float(getattr(_ncs_state, "daily_fatigue", 0.0) or 0.0)
            except Exception as _exc:
                logger.debug(f"{self.log_prefix} 观看态疲劳采样失败: {_exc}")
            # 关系与好感
            _rel = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
            _aff = float(_rel.get("affection", 0.0) or 0.0)
            # 存在态
            _sw = 0.5
            _aw = 0.5
            _avoid = 0.0
            if self._cached_presence_state:
                _sw = float(getattr(self._cached_presence_state, "social_willingness", 0.5) or 0.5)
                _aw = float(getattr(self._cached_presence_state, "watch_willingness", 0.5) or 0.5)
                _avoid = float(getattr(self._cached_presence_state, "avoidance_tendency", 0.0) or 0.0)
            # 打断权分级（GAP-E）
            _tier3_count = 0
            _tier2_count = 0
            _interruption = getattr(self, "_cached_interruption_tiers", None)
            if isinstance(_interruption, dict):
                _tier3_count = sum(1 for s in _interruption.values() if s >= 0.65)
                _tier2_count = sum(1 for s in _interruption.values() if 0.30 <= s < 0.65)
            # 自我引用强度
            _ref_max = 0.0
            _discuss_n = 0
            if self._cached_self_references:
                for _r in self._cached_self_references:
                    _rs = float(getattr(_r, "strength", 0.0) or 0.0)
                    if _rs > _ref_max:
                        _ref_max = _rs
                    _rt = getattr(_r, "ref_type", None)
                    if _rt:
                        _rv = _rt.value if hasattr(_rt, "value") else str(_rt)
                        if _rv == "discussed_as_topic":
                            _discuss_n += 1
            # 群体模式
            _is_heated = False
            _is_conflict = False
            if self._cached_pattern_evidence:
                for _pe in self._cached_pattern_evidence:
                    _pc = float(getattr(_pe, "confidence", 0.0) or 0.0)
                    if _pc < 0.35:
                        continue
                    _pp = getattr(_pe, "pattern", None)
                    if _pp:
                        _pv = _pp.value if hasattr(_pp, "value") else str(_pp)
                        if _pv == "heated_discussion":
                            _is_heated = True
                        elif _pv in ("conflict_escalation", "argument"):
                            _is_conflict = True
            # 活跃用户数
            _hot_users = 0
            if self._cached_participant_summary:
                _hot_users = int(self._cached_participant_summary.get("hot_count", 0) or 0)
                _hot_users += int(self._cached_participant_summary.get("warm_count", 0) or 0)
            # ══ 转移路径评估 ══
            # 路径⑥：[任意] → blackout（通用退出条件，优先级最高）
            _should_blackout = False
            _blackout_reason = ""
            if _activity < 12 and _env_fatigue > 55 and _silence > 1800:
                _should_blackout = True
                _blackout_reason = "低活跃+高环境疲劳+长静默"
            elif _fatigue > 75 and _silence > 600:
                _should_blackout = True
                _blackout_reason = "高疲劳+静默"
            elif _avoid > 0.7 and _sw < 0.15:
                _should_blackout = True
                _blackout_reason = "高回避+低社交欲"
            elif _hot_users >= 8 and self._metric_has_signal(_aff) and _aff < 20:
                _should_blackout = True
                _blackout_reason = "过于嘈杂+低好感"
            if _should_blackout and _cur_val not in ("blackout",):
                try:
                    _snap = self._build_unified_flow_snapshot("watch_transition")
                    _snap.watch_state = WatchState.BLACKOUT.value
                    _snap.phase = FlowPhase.DORMANT.value
                    _snap.blocker = _blackout_reason
                    self._apply_unified_flow_snapshot(
                        _snap,
                        watch_reason=f"watch:auto_blackout:{_blackout_reason}",
                        phase_reason=f"观看退场: {_blackout_reason}",
                    )
                    return f"{_cur_val}→blackout({_blackout_reason})"
                except Exception as _e:
                    logger.warning(f"{self.log_prefix} 强制进入blackout失败: {_e}")
            # 路径①：blackout → peek
            if _cur_val == "blackout":
                _peek_trigger_strength = 0.0
                if has_ping:
                    _peek_trigger_strength += 0.55
                if _boredom > thresholds.boredom_peek_trigger:
                    _peek_trigger_strength += min(0.25, (_boredom - thresholds.boredom_peek_trigger) / 100.0)
                if _aff > 55:
                    _peek_trigger_strength += 0.10
                if _tier3_count > 0:
                    _peek_trigger_strength += 0.18 * min(1.0, _tier3_count / 3.0)
                if _is_heated:
                    _peek_trigger_strength += 0.08
                if _discuss_n >= 2:
                    _peek_trigger_strength += 0.15
                if _hot_users >= 4 and _hot_users <= 8:
                    _peek_trigger_strength += 0.06
                if _peek_trigger_strength > 0.42:
                    try:
                        _snap = self._build_unified_flow_snapshot("watch_transition")
                        _snap.watch_state = WatchState.PEEK.value
                        _snap.phase = FlowPhase.STANDBY.value
                        self._apply_unified_flow_snapshot(
                            _snap,
                            watch_reason=f"watch:blackout_to_peek:{_peek_trigger_strength:.2f}",
                            phase_reason="黑屏后恢复窥屏",
                        )
                        return f"blackout→peek(trigger={_peek_trigger_strength:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} blackout→peek异常: {_e}")
            # 路径②：peek → skim_window
            if _cur_val == "peek" and _governor_cap_rank >= WatchState.SKIM_WINDOW.rank():
                _skim_trigger = 0.0
                if _tier3_count > 0:
                    _skim_trigger += 0.22 * min(1.0, _tier3_count)
                if _tier2_count >= 2:
                    _skim_trigger += 0.12
                if _ref_max > 0.35:
                    _skim_trigger += 0.15
                if _is_heated:
                    _skim_trigger += 0.10
                if _human_batch_count >= 3:
                    _skim_trigger += 0.08
                if _boredom > thresholds.boredom_scan_trigger:
                    _skim_trigger += 0.06
                if _skim_trigger > 0.38:
                    try:
                        _snap = self._build_unified_flow_snapshot("watch_transition")
                        _snap.watch_state = WatchState.SKIM_WINDOW.value
                        _snap.phase = FlowPhase.STANDBY.value
                        self._apply_unified_flow_snapshot(
                            _snap,
                            watch_reason=f"watch:peek_to_skim:{_skim_trigger:.2f}",
                            phase_reason="窥屏升级为扫视",
                        )
                        return f"peek→skim_window(trigger={_skim_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} peek→skim_window异常: {_e}")
            # 路径③：skim → active_watch
            if _cur_val == "skim_window" and _governor_cap_rank >= WatchState.ACTIVE_WATCH.rank():
                _active_trigger = 0.0
                if _tier3_count >= 1:
                    _active_trigger += 0.28
                if _discuss_n >= 1 and _ref_max > 0.45:
                    _active_trigger += 0.18
                if has_ping:
                    _active_trigger += 0.22
                if _aw > 0.55:
                    _active_trigger += 0.12
                if _is_conflict and _aff > 35:
                    _active_trigger += 0.14
                if _active_trigger > 0.40:
                    try:
                        _snap = self._build_unified_flow_snapshot("watch_transition")
                        _snap.watch_state = WatchState.ACTIVE_WATCH.value
                        _snap.phase = FlowPhase.STANDBY.value
                        self._apply_unified_flow_snapshot(
                            _snap,
                            watch_reason=f"watch:skim_to_active:{_active_trigger:.2f}",
                            phase_reason="扫视升级为积极关注",
                        )
                        return f"skim→active_watch(trigger={_active_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} skim→active_watch异常: {_e}")
            # 路径④：active → engaged
            if _cur_val == "active_watch" and _governor_cap_rank >= WatchState.ENGAGED.rank():
                _engage_trigger = 0.0
                if has_ping:
                    _engage_trigger += 0.32
                if _tier3_count >= 1:
                    _engage_trigger += 0.24
                if _discuss_n >= 2:
                    _engage_trigger += 0.16
                if _sw > 0.6 and _aw > 0.55:
                    _engage_trigger += 0.14
                if _is_heated or _is_conflict:
                    _engage_trigger += 0.10
                if _engage_trigger > 0.48:
                    try:
                        _snap = self._build_unified_flow_snapshot("watch_transition")
                        _snap.watch_state = WatchState.ENGAGED.value
                        _snap.phase = FlowPhase.ENGAGED.value
                        self._apply_unified_flow_snapshot(
                            _snap,
                            watch_reason=f"watch:active_to_engaged:{_engage_trigger:.2f}",
                            phase_reason="积极关注升级为参与",
                        )
                        return f"active→engaged(trigger={_engage_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} active→engaged异常: {_e}")
            # 路径⑤：engaged → active（退回观察）
            if _cur_val == "engaged":
                _disengage_trigger = 0.0
                if _silence > 120 and _tier3_count == 0:
                    _disengage_trigger += 0.25
                if _avoid > 0.45:
                    _disengage_trigger += 0.20
                if _fatigue > 55:
                    _disengage_trigger += 0.15
                if _human_batch_count <= 1 and not has_ping:
                    _disengage_trigger += 0.12
                if _disengage_trigger > 0.48:
                    try:
                        _snap = self._build_unified_flow_snapshot("watch_transition")
                        _snap.watch_state = WatchState.ACTIVE_WATCH.value
                        _snap.phase = FlowPhase.STANDBY.value
                        self._apply_unified_flow_snapshot(
                            _snap,
                            watch_reason=f"watch:engaged_to_active:{_disengage_trigger:.2f}",
                            phase_reason="参与后退回积极关注",
                        )
                        return f"engaged→active(disengage={_disengage_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} engaged→active异常: {_e}")
            # 路径⑦：engaged → blackout（直接退到不看）
            if _cur_val == "engaged":
                _crash_trigger = 0.0
                if _fatigue > 80:
                    _crash_trigger += 0.35
                if _avoid > 0.75:
                    _crash_trigger += 0.30
                if _silence > 600 and _tier3_count == 0:
                    _crash_trigger += 0.20
                if _crash_trigger > 0.60:
                    try:
                        _snap = self._build_unified_flow_snapshot("watch_transition")
                        _snap.watch_state = WatchState.BLACKOUT.value
                        _snap.phase = FlowPhase.DORMANT.value
                        _snap.blocker = f"engaged_crash={_crash_trigger:.2f}"
                        self._apply_unified_flow_snapshot(
                            _snap,
                            watch_reason=f"watch:engaged_to_blackout:{_crash_trigger:.2f}",
                            phase_reason="参与态崩落到休息",
                        )
                        return f"engaged→blackout(crash={_crash_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} engaged→blackout异常: {_e}")
            # 路径⑧：skim → peek（退回窥屏）
            if _cur_val == "skim_window":
                _fallback_trigger = 0.0
                if _tier3_count == 0 and _tier2_count <= 1:
                    _fallback_trigger += 0.22
                if _silence > 60:
                    _fallback_trigger += 0.15
                if _boredom < thresholds.boredom_downgrade_trigger:
                    _fallback_trigger += 0.12
                if _ref_max < 0.15 and not _is_heated:
                    _fallback_trigger += 0.18
                if _fallback_trigger > 0.45:
                    try:
                        _snap = self._build_unified_flow_snapshot("watch_transition")
                        _snap.watch_state = WatchState.PEEK.value
                        _snap.phase = FlowPhase.STANDBY.value
                        self._apply_unified_flow_snapshot(
                            _snap,
                            watch_reason=f"watch:skim_to_peek:{_fallback_trigger:.2f}",
                            phase_reason="扫视退回窥屏",
                        )
                        return f"skim→peek(fallback={_fallback_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} skim→peek异常: {_e}")
            return None
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 观看状态转移评估异常: {exc}")
            return None

    def _integrate_presence_metabolism(self, eagerness_val: float, now: float) -> None:
        """从新维度系统统一采集存在态+代谢约束（替代旧 PresenceCore+MetabolismEngine）"""
        thresholds = get_heartfc_thresholds()
        _has_interest_topic = False
        _has_close_friend = False
        try:
            _gs = self._orch.get("scene_state")
            _topics = _gs.active_topic_slots(limit=3)
            for _t in _topics:
                if _t.heat.value in ("hot", "warm"):
                    _has_interest_topic = True
                    break
            _rel_snap = getattr(self, "_last_relation_snapshot", None) or {}
            if float(_rel_snap.get("affection", 0) or 0) > 60:
                _has_close_friend = True
        except Exception as _e:
            logger.debug(f"{self.log_prefix} 亲密朋友检测异常: {_e}")
        try:
            _d6 = EnergyChainDimension.get_instance()
            _d6_state = _d6._ensure_channel(self.stream_id)
            _d6_vote = _d6.vote(type("VC", (), {"channel_id": self.stream_id, "user_id": ""})())
            _combined_ratio = _d6_state.combined_ratio()
            _chat_ratio = float(_d6_state.chat_ratio())
            _thinking_ratio = float(_d6_state.thinking_ratio())
            _boredom_level = max(0.0, (1.0 - _combined_ratio) * 100.0)
            _loafing_level = max(0.0, min(100.0, (1.0 - _d6_state.activity_level / 100.0) * 80.0))
            _boredom_drive = max(0.0, min(1.0, _boredom_level / 100.0))
            _loafing_sup = max(0.0, min(1.0, _loafing_level / 100.0))
            _watch_will = 0.5
            if 0.10 < _boredom_drive < 0.32:
                _watch_will = min(0.85, 0.5 + min(0.12, (_boredom_drive - 0.10) * 0.4))
            elif _boredom_drive > 0.38:
                _watch_will = max(0.05, 0.5 - min(0.30, (_boredom_drive - 0.38) * 0.5))
            _social_will = (
                max(0.05, 0.5 - min(0.35, max(0.0, (_loafing_sup - thresholds.loafing_social_inhibit)) * 0.6))
                if _loafing_sup > thresholds.loafing_social_inhibit
                else 0.5
            )
            _curiosity = 0.3 + (1.0 - _combined_ratio) * 0.3 if _has_interest_topic else 0.25
            _avoidance = max(0.0, min(0.8, _loafing_level / 100.0 * 0.5))
            _quiet_pref = max(0.0, min(0.8, (1.0 - _d6_state.social_value / 100.0) * 0.4 + 0.15))
            _fatigue_acc = max(0.0, float(getattr(_d6_state, "total_consumed_today", 0.0) or 0.0))
            _annoyance_level = float(getattr(_d6_state, "annoyance_level", 0.0) or 0.0)
            _reply_hint = (
                "简短"
                if _combined_ratio < 0.25 or _annoyance_level > 60.0
                else ("中等" if _combined_ratio < 0.55 else "可以长一点")
            )
            _energy_gate_open = bool(_combined_ratio > 0.12 and _d6_state.activity_level > 8.0)
            _is_perfunctory = bool(_combined_ratio < 0.22 or _annoyance_level > 70.0)
            _recovery_rate = max(0.2, min(1.0, 1.0 - min(0.75, _fatigue_acc / 100.0)))
            _sleep_debt = float((getattr(self, "_cached_night_summary", {}) or {}).get("sleep_debt", 0.0) or 0.0)
            _sleep_debt_modifier = 1.0 + min(0.8, _sleep_debt)
            _proactive_drive = max(
                0.05,
                min(0.95, 0.18 + _boredom_drive * 0.35 + max(0.0, _social_will - 0.5) * 0.3),
            )
            _mood_valence = max(0.0, min(1.0, 0.5 + (_d6_state.social_value / 200.0) - (_annoyance_level / 150.0)))
            _mood_arousal = max(0.0, min(1.0, 0.25 + (_d6_state.activity_level / 140.0) + min(0.15, _boredom_drive * 0.2)))
            self._cached_metabolism_constraints = {
                "boredom": round(_boredom_level, 2),
                "boredom_level": round(_boredom_drive, 3),
                "loafing": round(_loafing_level, 2),
                "loafing_level": round(_loafing_level, 2),
                "chat_fuel": round(_d6_state.chat_pool, 1),
                "thinking_fuel": round(_d6_state.thinking_value, 1),
                "chat_value": round(_d6_state.chat_pool, 1),
                "activity_gauge": round(_d6_state.activity_level, 1),
                "social_gauge": round(_d6_state.social_value, 1),
                "fatigue_accumulator": round(_fatigue_acc, 1),
                "boredom_watch_drive": round(_boredom_drive, 3),
                "loafing_suppression": round(_loafing_sup, 3),
                "energy_ratio": round(_combined_ratio, 3),
                "energy_suppression": round(max(0.0, min(1.0, 1.0 - _combined_ratio)), 3),
                "reply_length_hint": _reply_hint,
                "energy_gate_open": _energy_gate_open,
                "is_perfunctory": _is_perfunctory,
                "recovery_rate": round(_recovery_rate, 3),
                "sleep_debt_modifier": round(_sleep_debt_modifier, 3),
                "proactive_drive": round(_proactive_drive, 3),
                "night_mode": _d6_vote.night_mode,
                "energy_stage": str(getattr(_d6_vote, "energy_stage", "ADEQUATE").value)
                if hasattr(getattr(_d6_vote, "energy_stage", None), "value")
                else "ADEQUATE",
                "annoyance": round(_annoyance_level, 1),
                "annoyance_accumulated": round(_annoyance_level, 1),
                "chain_count": _d6_state.chain_count,
            }
            self._chat_energy = float(_d6_state.chat_pool)
            self._thinking_energy = float(_d6_state.thinking_value)

            class _PresenceStateCompat:
                watch_willingness: float = _watch_will
                social_willingness: float = _social_will
                curiosity_level: float = _curiosity
                avoidance_tendency: float = _avoidance
                quiet_preference: float = _quiet_pref
                outward_attention: float = max(0.05, 1.0 - _avoidance)

            self._cached_presence_state = _PresenceStateCompat()

            class _MetabolismStateCompat:
                chat_fuel: float = _d6_state.chat_pool
                thinking_fuel: float = _d6_state.thinking_value
                chat_energy_ratio: float = _chat_ratio
                energy_ratio: float = _combined_ratio
                thinking_ratio: float = _thinking_ratio
                activity_gauge: float = _d6_state.activity_level
                social_gauge: float = _d6_state.social_value
                mood_valence: float = _mood_valence
                mood_arousal: float = _mood_arousal
                stress_accumulation: float = _fatigue_acc
                trauma_score: float = float(getattr(self, "_cached_trauma_score", 0.0) or 0.0)
                channel_annoyance: float = _annoyance_level
                boredom_level: float = _boredom_level
                loafing_level: float = _loafing_level
                fatigue_level: float = _fatigue_acc
                fatigue_accumulator: float = _fatigue_acc
                consecutive_active_minutes: float = 0.0
                total_consumed_today: float = _d6_state.total_consumed_today

            self._cached_metabolism_state = _MetabolismStateCompat()
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 新维度状态采集异常: {exc}")

    def _integrate_message_routing(self, incoming_batch: List) -> None:
        """语义路由分类 + 时间流注册 + 活跃名册 + 多模态预算"""
        try:
            _router = self._orch.get("semantic_router")
            self._cached_semantic_routes = _router.classify_batch(incoming_batch)
            self._cached_route_summary = _router.summarize_batch(self._cached_semantic_routes)
        except Exception as exc:
            self._cached_semantic_routes = []
            self._cached_route_summary = {}
            logger.debug(f"{self.log_prefix} 语义路由分类异常: {exc}")
        try:
            self._orch.get("subjective_time").register_batch(incoming_batch)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 时间流注册异常: {exc}")
        try:
            _roster = self._orch.get("participant_roster")
            for i, _m in enumerate(incoming_batch):
                _uid = str(getattr(_m, "user_id", "") or "").strip()
                _route = self._cached_semantic_routes[i] if i < len(self._cached_semantic_routes) else None
                if _route is not None:
                    try:
                        _route_at = bool(getattr(_route, "is_at_bot", False))
                        _route_quote = bool(getattr(_route, "is_quote_to_bot", False))
                        _m.is_at_bot = bool(getattr(_m, "is_at_bot", False)) or _route_at
                        _m.is_quote_to_bot = bool(getattr(_m, "is_quote_to_bot", False)) or _route_quote
                        _m.is_reply_to_bot = bool(getattr(_m, "is_reply_to_bot", False)) or _route_quote
                    except Exception as _route_attr_exc:
                        logger.debug(f"{self.log_prefix} 语义路由标记写回异常: {_route_attr_exc}")
                if not self._is_human_message_obj(_m):
                    continue
                _text = str(getattr(_m, "processed_plain_text", "") or "")[:50]
                _route_type = "plain_text"
                if _route is not None:
                    _route_type = getattr(getattr(_route, "category", None), "value", "") or str(
                        getattr(_route, "category", "plain_text")
                    )
                    _emotion = 0.4 if any(mark in _text for mark in ("！", "!", "？", "?")) else 0.0
                    _message_quality = 0.7 if len(_text) >= 16 else 0.5
                    _topic_relevance = 0.7 if len(_text) >= 8 else 0.4
                    _bot_interacted = bool(
                        _route and (getattr(_route, "is_at_bot", False) or getattr(_route, "is_quote_to_bot", False))
                    )
                    _roster.reset_consecutive(_uid)
                    _roster.touch(
                        _uid,
                        user_name=str(getattr(_m, "user_name", "") or getattr(_m, "nickname", "") or ""),
                        text_hint=_text,
                        bot_interacted=_bot_interacted,
                        message_quality=_message_quality,
                        topic_relevance=_topic_relevance,
                        emotional_content=_emotion,
                        response_type=_route_type,
                    )
            self._cached_participant_summary = _roster.summary()
        except Exception as exc:
            self._cached_participant_summary = {}
            logger.debug(f"{self.log_prefix} 活跃名册更新异常: {exc}")
        try:
            _mmb = self._orch.get("multimodal_budget")

            _media_items = []
            _budget_ratio = 1.0
            if self._cached_metabolism_state is not None:
                _chat_fuel = float(getattr(self._cached_metabolism_state, "chat_fuel", 100.0) or 100.0) / 100.0
                _thinking_fuel = (
                    float(
                        getattr(
                            self._cached_metabolism_state,
                            "thinking_fuel",
                            100.0,
                        )
                        or 100.0
                    )
                    / 100.0
                )
                _budget_ratio = max(0.0, min(1.0, min(_chat_fuel, _thinking_fuel)))
            for i, _sr in enumerate(self._cached_semantic_routes):
                _cat = _sr.category.value if hasattr(_sr.category, "value") else str(_sr.category)
                if _cat in ("image", "multi_image", "sticker", "video"):
                    _message = incoming_batch[i] if i < len(incoming_batch) else None
                    _image_url = ""
                    _image_list = getattr(_message, "image_list", None) or []
                    if _image_list:
                        _first_image = _image_list[0]
                        if isinstance(_first_image, str):
                            _image_url = _first_image
                        else:
                            _image_url = str(
                                getattr(_first_image, "url", "")
                                or getattr(_first_image, "image_url", "")
                                or getattr(_first_image, "file", "")
                                or ""
                            )
                    _media_items.append(
                        {
                            "media_type": _cat,
                            "is_sticker": _cat == "sticker",
                            "image_url": _image_url,
                            "context_relevance": (0.85 if getattr(_sr, "has_text", False) else 0.45),
                        }
                    )
            if _media_items:
                _budgeter = _mmb
                _budget_decisions = _budgeter.evaluate_batch(_media_items, current_budget_ratio=_budget_ratio)
                self._cached_multimodal_summary = _budgeter.last_batch_summary()
                self._cached_media_decisions = _budget_decisions
            else:
                self._cached_multimodal_summary = {}
        except Exception as exc:
            self._cached_multimodal_summary = {}
            logger.debug(f"{self.log_prefix} 多模态预算评估异常: {exc}")

    def _integrate_deep_understanding(self, incoming_batch: List) -> None:
        """自我指称检测 + 可见性门控 + 理解门控"""
        try:
            from src.core.self_reference_detector import (
                get_self_reference_detector,
            )

            self._cached_self_references = get_self_reference_detector().detect_batch(incoming_batch)
        except Exception as exc:
            self._cached_self_references = []
            logger.debug(f"{self.log_prefix} 自我指称检测异常: {exc}")
        try:
            _vg = self._orch.get("visibility_gate")
            # F22：构建关系偏置映射——好感/信任用户更容易被看见
            _rel_map = {}
            try:
                _rsnap = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
                if _rsnap:
                    _aff = float(_rsnap.get("affection", 0.0) or 0.0)
                    if not self._metric_has_signal(_aff):
                        _rel_level = None
                    elif _aff > 60:
                        _rel_level = 4
                    elif _aff > 35:
                        _rel_level = 3
                    elif _aff < 15:
                        _rel_level = 0
                    else:
                        _rel_level = 2
                    if _rel_level is None:
                        _rel_map = {}
                    else:
                        for _m in incoming_batch:
                            _uid = str(getattr(_m, "user_id", "") or "").strip()
                            if self._is_human_message_obj(_m):
                                _rel_map[_uid] = _rel_level
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            _loafing_val = 0.0
            try:
                if self._cached_metabolism_state:
                    _loafing_val = float(getattr(self._cached_metabolism_state, "loafing_level", 0.0) or 0.0)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            self._cached_visibility_decisions = _vg.evaluate_batch(
                incoming_batch,
                presence_state=self._cached_presence_state,
                user_relationships=_rel_map if _rel_map else None,
                loafing_level=_loafing_val,
            )
        except Exception as exc:
            self._cached_visibility_decisions = []
            logger.debug(f"{self.log_prefix} 可见性门控异常: {exc}")
        try:
            _ug = self._orch.get("understanding_gate")
            _contexts = []
            _skip_indices = set()
            _SKIP_CATEGORIES = ("recall",)
            _rel_snap = getattr(self, "_last_relation_snapshot", None) or {}
            _presence_state = getattr(self, "_cached_presence_state", None)
            _emotion_state = getattr(self, "_cached_self_state", None)
            _social_willingness = float(getattr(_presence_state, "social_willingness", 0.5) or 0.5)
            if _social_willingness <= 0.0 and _emotion_state is not None:
                _social_willingness = float(getattr(_emotion_state, "social_desire", 0.5) or 0.5)
            _energy_ratio = 1.0
            if isinstance(getattr(self, "_cached_metabolism_constraints", None), dict):
                _energy_ratio = max(
                    0.05,
                    1.0 - float(self._cached_metabolism_constraints.get("energy_suppression", 0.0) or 0.0),
                )
            _focus_uid = str(getattr(self, "_last_user_id", "") or "")
            for i, _m in enumerate(incoming_batch):
                _vis_score = 0.5
                if i < len(self._cached_visibility_decisions):
                    _vis_score = float(getattr(self._cached_visibility_decisions[i], "score", 0.5) or 0.5)
                _cat = "plain_text"
                if i < len(self._cached_semantic_routes):
                    _sr = self._cached_semantic_routes[i]
                    _cat = _sr.category.value if hasattr(_sr.category, "value") else str(_sr.category)
                if _cat in _SKIP_CATEGORIES:
                    _skip_indices.add(i)
                    continue
                _text = str(getattr(_m, "processed_plain_text", "") or "")[:500]
                _uid = str(getattr(_m, "user_id", "") or "")
                _rel_level = 2
                if _uid and _uid in _rel_snap and isinstance(_rel_snap.get(_uid), dict):
                    _rel_level = int((_rel_snap.get(_uid) or {}).get("relationship_level", 2) or 2)
                elif _uid and _uid == _focus_uid and isinstance(_rel_snap, dict):
                    _rel_level = int(_rel_snap.get("relationship_level", 2) or 2)
                _bot_relevant = self._message_targets_bot(_m)
                _is_question = "?" in _text or "？" in _text or _text.endswith("吗") or _text.endswith("呢")
                _has_media = bool(
                    getattr(_m, "has_media", False)
                    or getattr(_m, "has_image", False)
                    or getattr(_m, "image_url", "")
                    or getattr(_m, "image_urls", None)
                )
                _contexts.append(
                    {
                        "text_content": _text,
                        "message_category": _cat,
                        "visibility_score": _vis_score,
                        "bot_relevant": _bot_relevant,
                        "is_question": _is_question,
                        "has_media": _has_media,
                        "energy_ratio": _energy_ratio,
                        "user_relationship_level": _rel_level,
                        "social_willingness": _social_willingness,
                    }
                )
            _len_skip = len(_skip_indices)
            if _len_skip > 0:
                logger.info(
                    f"{self.log_prefix} 🧹 理解门控前置过滤: 跳过{_len_skip}条低价值消息(RECALL等)，节省思考燃料"
                )
            self._cached_understanding_results = _ug.batch_evaluate(_contexts)
            if _skip_indices:
                try:
                    _filled = list(self._cached_understanding_results or [])
                    _orig_len = len(incoming_batch)
                    while len(_filled) < _orig_len - _len_skip:
                        _filled.append(None)
                    for _si in sorted(_skip_indices):
                        _filled.insert(_si, None)
                    self._cached_understanding_results = _filled
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
            # 深度理解完成后消耗思考燃料
            try:
                _d6 = EnergyChainDimension.get_instance()
                _think_len = sum(len(c.get("text_content", "")) for c in _contexts)

                class _ThinkEvt:
                    channel_id = self.stream_id
                    event_type = "thinking_completed"
                    complexity = max(0.1, min(2.0, _think_len / 200.0))
                    raw_extras = {}

                _d6.on_event(_ThinkEvt())
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        except Exception as exc:
            self._cached_understanding_results = []
            logger.debug(f"{self.log_prefix} 理解门控异常: {exc}")
        # 误解风险评估：检查用户后续消息是否暗示 bot 上轮回复被误解
        self._cached_misunderstanding_signal = None
        try:
            _last_bot = self._latest_recent_bot_utterance()
            if _last_bot and incoming_batch:
                _bot_text = str(_last_bot.get("text", "") or "")
                _bot_ts = float(_last_bot.get("ts", 0.0) or 0.0)
                if _bot_text and _bot_ts > 0:
                    # 找第一条非bot用户消息作为后续
                    _followup_text = ""
                    _followup_ts = 0.0
                    for _fm in incoming_batch:
                        _fu = str(getattr(_fm, "user_id", "") or "").strip()
                        if _fu and not self._is_bot_message_obj(_fm):
                            _followup_text = str(getattr(_fm, "processed_plain_text", "") or "")[:300]
                            _followup_ts = float(getattr(_fm, "time", 0.0) or 0.0)
                            break
                    if _followup_text:
                        _gap = max(0.0, _followup_ts - _bot_ts) if _followup_ts > _bot_ts else 60.0
                        # 简单重复提问检测：后续文本和bot回复前用户的上一条高度相似
                        _is_repeat = False
                        if len(self._recent_user_inputs) >= 2:
                            _prev_input = str(self._recent_user_inputs[-2].get("text", "") or "")
                            if _prev_input and _followup_text and len(_followup_text) > 3:
                                _overlap = sum(1 for c in _followup_text[:50] if c in _prev_input[:50])
                                _is_repeat = _overlap > len(_followup_text[:50]) * 0.6
                        from src.core.understanding_gate import (
                            get_understanding_gate,
                        )

                        _signal = get_understanding_gate().evaluate_misunderstanding_risk(
                            bot_reply_text=_bot_text,
                            user_followup_text=_followup_text,
                            is_repeated_question=_is_repeat,
                            time_gap_sec=_gap,
                        )
                        if _signal.risk_score > 0.35:
                            self._cached_misunderstanding_signal = _signal
                            if not hasattr(self, "_last_misunderstanding_log_ts"):
                                self._last_misunderstanding_log_ts = 0.0
                            _now_m = time.time()
                            if (_now_m - self._last_misunderstanding_log_ts) >= 30.0:
                                logger.info(
                                    f"{self.log_prefix} 误解风险={_signal.risk_score:.2f} "
                                    f"类型={_signal.signal_type} 线索={_signal.detected_cues[:2]}"
                                )
                                self._last_misunderstanding_log_ts = _now_m
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 误解风险评估异常: {exc}")

    def _apply_sensory_gates(self, incoming_batch: List, pinged_msg=None) -> Optional[str]:
        """感官门控：基于可见性+理解度判定是否应跳过本轮处理
        返回 None 表示放行，返回 str 表示拦截原因"""
        if not incoming_batch:
            return None
        _rest_verdict = self._evaluate_rest_governor(
            incoming_batch=incoming_batch,
            source="reactive",
            targeted_to_bot=self._has_targeted_bot_message(incoming_batch),
            admin_force=bool(getattr(self, "_is_admin_forced", False)),
            pinged_msg=pinged_msg,
        )
        if _rest_verdict.interruption_policy == "block" and pinged_msg is None:
            return f"休息门控: {self._summarize_rest_governor(_rest_verdict)}"
        _has_ping = pinged_msg is not None
        _vis_ignored_count = 0
        _vis_low_score_count = 0
        _understand_none_count = 0
        _understand_no_reply_count = 0
        # F29：三态区分——为每条消息标记暴露层级
        self._cached_exposure_tiers: Dict[str, str] = {}
        if self._cached_visibility_decisions:
            for _vd in self._cached_visibility_decisions:
                _vl = getattr(_vd, "level", None)
                if _vl is None:
                    continue
                _vl_val = _vl.value if hasattr(_vl, "value") else str(_vl)
                _msg_key = str(getattr(_vd, "message_key", "") or getattr(_vd, "msg_id", "") or "")
                if _vl_val == "understood_but_ignored":
                    _vis_ignored_count += 1
                    if _msg_key:
                        self._cached_exposure_tiers[_msg_key] = "seen_ignored"
                elif _vl_val in ("received", "glanced"):
                    _vis_low_score_count += 1
                    if _msg_key:
                        self._cached_exposure_tiers[_msg_key] = "never_seen"
                elif _vl_val in ("focused", "understood"):
                    if _msg_key:
                        self._cached_exposure_tiers[_msg_key] = "seen_engaged"
                _score = float(getattr(_vd, "score", 0.5) or 0.5)
                if _score < 0.18:
                    _vis_low_score_count += 1
        if self._cached_understanding_results:
            for _ur in self._cached_understanding_results:
                _cl = getattr(_ur, "level", None)
                if _cl is None:
                    continue
                _cl_val = _cl.value if hasattr(_cl, "value") else str(_cl)
                _ukey = str(getattr(_ur, "message_key", "") or getattr(_ur, "msg_id", "") or "")
                if _cl_val == "cannot_understand":
                    _understand_none_count += 1
                    if _ukey and _ukey not in self._cached_exposure_tiers:
                        self._cached_exposure_tiers[_ukey] = "never_seen"
                elif _cl_val == "can_understand_no_reply":
                    _understand_no_reply_count += 1
                    if _ukey:
                        self._cached_exposure_tiers[_ukey] = "seen_engaged_no_reply"
        _total_user_msgs = sum(1 for m in incoming_batch if not self._is_bot_message_obj(m))
        if _total_user_msgs <= 0:
            return None
        _ignore_ratio = (_vis_ignored_count + _understand_no_reply_count) / max(1, _total_user_msgs)
        _none_ratio = _understand_none_count / max(1, _total_user_msgs)
        if _none_ratio >= 0.8 and not _has_ping:
            return f"理解门控: {_understand_none_count}/{_total_user_msgs}条消息看不懂"
        _effective_ignore_threshold = 0.85
        if _ignore_ratio >= _effective_ignore_threshold and not _has_ping:
            # F29：附加三态分布
            _tier_counts = {
                "never_seen": 0,
                "seen_ignored": 0,
                "seen_engaged": 0,
                "seen_engaged_no_reply": 0,
            }
            for _t in self._cached_exposure_tiers.values():
                if _t in _tier_counts:
                    _tier_counts[_t] += 1
            _tier_str = f" | 三态:未看见={_tier_counts['never_seen']},扫到不理={_tier_counts['seen_ignored']},深度关注={
                _tier_counts['seen_engaged']
            },关注不回={_tier_counts['seen_engaged_no_reply']}"
            return f"可见性/理解门控: {_vis_ignored_count} 条被忽略+{_understand_no_reply_count} 条不回/{
                _total_user_msgs
            } 总消息{_tier_str} "
        if _vis_low_score_count >= _total_user_msgs and not _has_ping:
            return f"可见性全低分: {_vis_low_score_count}/{_total_user_msgs}条消息均未引起注意"
        # GAP-M：深度可见性评分——三维偏置+群环境调制+时间衰减
        try:
            from src.core.deep_visibility_scorer import (
                UserBiasProfile,
                EventBiasInput,
                TopicBiasInput,
                GroupEnvModulation,
                TimeDecayFactor,
            )

            if not self._deep_scorer_initialized:
                self._deep_scorer_initialized = True
            _dvs = self._orch.get("deep_visibility")
            self._cached_deep_visibility_results = []
            for _msg in incoming_batch:
                _uid_m = str(getattr(_msg, "user_id", "") or "")
                if not _uid_m or self._is_bot_message_obj(_msg):
                    continue
                _rel_snap_m = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
                _up_m = UserBiasProfile(
                    user_id=_uid_m,
                    affection=float(_rel_snap_m.get("affection", 0.0) or 0.0) / 100.0,
                    trust=float(_rel_snap_m.get("trust_value", 0.0) or 0.0) / 100.0,
                    annoyance=float(_rel_snap_m.get("annoyance_value", 0.0) or 0.0) / 100.0,
                    is_avoided=bool(float(_rel_snap_m.get("aversion_value", 0.0) or 0.0) > 30),
                )
                _plain = str(
                    getattr(_msg, "processed_plain_text", "")
                    or getattr(_msg, "plain_text", "")
                    or getattr(_msg, "content", "")
                    or ""
                )
                _ei_m = EventBiasInput(
                    is_at_mention=getattr(_msg, "is_at", False),
                    is_quote_reply=bool(getattr(_msg, "reply_to_message_id", None)),
                    message_length=len(_plain),
                )
                _ti_m = TopicBiasInput(message_text=_plain)
                _meta_state = getattr(self, "_cached_metabolism_constraints", None) or {}
                _presence_state_m = getattr(self, "_cached_presence_state", None)
                _scene_state_m = getattr(self, "_cached_scene_snapshot", None)
                _social_will_m = float(getattr(_presence_state_m, "social_willingness", 0.5) or 0.5)
                _watch_rank_m = int(getattr(_presence_state_m, "watch_state_rank", 2) or 2)
                _scene_atmo_m = ""
                if _scene_state_m is not None:
                    _scene_atmo_raw = getattr(_scene_state_m, "atmosphere", "")
                    _scene_atmo_m = (
                        getattr(_scene_atmo_raw, "value", str(_scene_atmo_raw)) if _scene_atmo_raw is not None else ""
                    )
                _gm_m = GroupEnvModulation(
                    group_atmosphere=str(_scene_atmo_m or ""),
                    active_member_count=int(getattr(_scene_state_m, "unique_speakers_5min", 0) or 0),
                    is_multi_thread=bool(int(getattr(_scene_state_m, "thread_count", 0) or 0) > 1),
                    suitable_to_interrupt=bool(getattr(_scene_state_m, "suitable_to_join", True)),
                    watch_state_rank=_watch_rank_m,
                    energy_ratio=1.0 - float(_meta_state.get("energy_suppression", 0.0) or 0.0),
                    social_willingness=_social_will_m,
                    loafing_level=float(_meta_state.get("loafing_suppression", 0.0) or 0.0),
                    boredom_level=float(_meta_state.get("boredom_level", 0.0) or 0.0),
                )
                _td_m = TimeDecayFactor(
                    message_age_sec=max(
                        0.0,
                        time.time() - float(getattr(_msg, "timestamp", time.time()) or time.time()),
                    )
                )
                _dvr = _dvs.evaluate(
                    _msg,
                    user_profile=_up_m,
                    event_input=_ei_m,
                    topic_input=_ti_m,
                    group_mod=_gm_m,
                    time_decay=_td_m,
                )
                self._cached_deep_visibility_results.append(_dvr)
            if self._cached_deep_visibility_results:
                _dv_high = sum(1 for r in self._cached_deep_visibility_results if r.modulated_score > 0.55)
                _dv_low = sum(1 for r in self._cached_deep_visibility_results if r.modulated_score < 0.12)
                if _dv_low >= _total_user_msgs * 0.7 and not _has_ping:
                    return f"深度可见性全低: {_dv_low}/{_total_user_msgs}条消息mod_score<0.12"
                logger.debug(
                    f"{self.log_prefix} [GAP-M] 深度评分完成: "
                    f"高优={_dv_high}, 低分={_dv_low}, 总={len(self._cached_deep_visibility_results)}"
                )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # GAP-E：消息打断权三级分类 —— 综合评分后分级处理
        if not _has_ping:
            try:
                _interruption_scores = self._classify_interruption_rights(incoming_batch)
                _tier1_count = sum(1 for s in _interruption_scores.values() if s < 0.30)
                _tier2_count = sum(1 for s in _interruption_scores.values() if 0.30 <= s < 0.65)
                _tier3_count = sum(1 for s in _interruption_scores.values() if s >= 0.65)
                self._cached_interruption_tiers = _interruption_scores
                _loaf_level = 0.0
                if self._cached_metabolism_constraints:
                    _loaf_level = float(self._cached_metabolism_constraints.get("loafing_suppression", 0.0) or 0.0)
                if _loaf_level > 0.65:
                    if _tier3_count == 0:
                        return f"打断权T1全拦截: loafing={_loaf_level: .2f}, T1={_tier1_count}  T2={_tier2_count}  T3={
                            _tier3_count
                        } "
                    elif _tier3_count <= 1 and _tier1_count > _tier3_count * 3:
                        return f"打断权T1主导: loafing={_loaf_level:.2f}, 仅{_tier3_count}条强优先级不足以拉起"
                elif _loaf_level > 0.35:
                    if _tier1_count >= _total_user_msgs * 0.8 and _tier3_count == 0:
                        return f"打断权半拦截: loafing={_loaf_level: .2f}, T1占比过高({_tier1_count} /{
                            _total_user_msgs
                        })且无强优先级"
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        return None

    def _classify_interruption_rights(self, incoming_batch: List) -> Dict[str, float]:
        """GAP-E：为每条消息计算打断权得分(0~1)，返回 {msg_key: score}

        三级划分：
          Tier 1 (score < 0.30): 普通消息 —— 高摸鱼时不进入深层可见性
          Tier 2 (0.30 ≤ score < 0.65): 中度相关 —— 允许peek/skim但不直接到active_watch
          Tier 3 (score ≥ 0.65): 强优先级 —— 可打断任意摸鱼状态，至少进入"看见"

        评分因子（加权求和）：
          - 被@或引用: +0.40
          - 讨论机器人相关: +0.25
          - 高好感用户发言: +0~0.20 (按好感度线性映射)
          - 当前话题高相关: +0.15
          - 冲突/高风险内容: +0.20
          - 连续互动线程: +0.10 (每多一轮+0.05)
        """
        scores: Dict[str, float] = {}
        rel_snap = getattr(self, "_last_relation_snapshot", None) or {}
        affection_map: Dict[str, float] = {}
        trust_map: Dict[str, float] = {}
        if isinstance(rel_snap, dict):
            for k, v in rel_snap.items():
                if "affection" in str(k).lower() or "好感" in str(k):
                    if isinstance(v, (int, float)):
                        affection_map[str(k)] = float(v)
                elif "trust" in str(k).lower() or "信任" in str(k):
                    if isinstance(v, (int, float)):
                        trust_map[str(k)] = float(v)
        for _msg in incoming_batch:
            _uid = str(getattr(_msg, "user_id", "") or "").strip()
            if not _uid or _uid == "bot":
                continue
            _key = str(getattr(_msg, "message_key", "") or getattr(_msg, "msg_id", "") or id(_msg))
            _score = 0.05
            # 因子1：@提及 / 引用回复
            _at_list = getattr(_msg, "at_list", None) or []
            _reply_to = str(getattr(_msg, "reply_to", "") or "").strip()
            if any(str(t).strip().lower() in ("bot", "robot", self.stream_id.lower()) for t in _at_list):
                _score += 0.42
            if _reply_to and len(_reply_to) > 0:
                _score += 0.18
            # 因子2：自我引用检测
            if self._cached_self_references:
                for _ref in self._cached_self_references:
                    _ref_msg_key = str(getattr(_ref, "message_key", "") or "")
                    if _ref_msg_key == _key or _ref_msg_key in (_key[: min(16, len(_key))],):
                        _rt = getattr(_ref, "ref_type", None)
                        if _rt:
                            _rv = _rt.value if hasattr(_rt, "value") else str(_rt)
                            if _rv == "direct_at":
                                _score += 0.38
                            elif _rv == "discussed_as_topic":
                                _score += 0.28
                            elif _rv == "quoted_reply":
                                _score += 0.22
                            elif _rv == "nickname_called":
                                _score += 0.32
                            break
            # 因子3：关系加权（高好感用户的消息获得更高打断权）
            _rel_view = self._normalize_relation_snapshot(rel_snap or {}) if isinstance(rel_snap, dict) else {}
            _aff = float(_rel_view.get("affection", 0.0) or 0.0)
            if self._metric_has_signal(_aff) and _aff > 60:
                _score += min(0.18, (_aff - 60.0) / 200.0)
            elif self._metric_has_signal(_aff) and _aff < 18:
                _score -= 0.08
            # 因子4：话题相关性
            if self._cached_pattern_evidence:
                for _pe in self._cached_pattern_evidence:
                    _pe_conf = float(getattr(_pe, "confidence", 0.0) or 0.0)
                    _pe_pat = getattr(_pe, "pattern", None)
                    if _pe_pat:
                        _pv = _pe_pat.value if hasattr(_pe_pat, "value") else str(_pe_pat)
                        if (
                            _pv
                            in (
                                "heated_discussion",
                                "conflict_escalation",
                                "argument",
                            )
                            and _pe_conf > 0.4
                        ):
                            _score += 0.12 * _pe_conf
                        elif (
                            _pv
                            in (
                                "newcomer_welcome",
                                "birthday_wish",
                                "congratulation",
                            )
                            and _pe_conf > 0.5
                        ):
                            _score += 0.08 * _pe_conf
            # 因子5：文本长度与内容密度（短消息默认低权重，长消息略增）
            _text = str(getattr(_msg, "processed_plain_text", "") or "")[:300]
            if len(_text) > 80:
                _score += min(0.06, len(_text) / 2000.0)
            # 因子6：图片/多媒体附加权重
            _has_media = bool(getattr(_msg, "has_image", False) or getattr(_msg, "media_count", 0))
            if _has_media:
                _score += 0.04
            scores[_key] = max(0.0, min(1.0, _score))
        return scores


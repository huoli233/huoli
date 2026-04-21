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

class EnhancedStrategyIntegrationMixin:
    def _decide_action(
        self,
        pinged_msg,
        eagerness: float,
        awareness,
        voice,
        messages: List,
    ) -> bool:
        """融合多维信号决定本轮是否执行行动规划

        决策优先级：
        1. 模型 should_reply → 直接执行/跳过（模型自主决策最高优先）
        2. 觉察引擎参与度 > 阈值 → 高概率行动
        3. 能量意愿 × 频率因子 → 概率行动
        """
        # 模型自主决策优先
        if voice is not None and hasattr(voice, "should_reply"):
            _model_reply = voice.should_reply
            if _model_reply is True:
                return True
            if _model_reply is False:
                return False
        # 被@但模型没明确说时，不强制，交给后续概率决策
        # 内心独白强驱动（仅作参考，不再强制）
        if pinged_msg is not None:
            pass
        # 觉察参与度驱动
        engagement_boost = 0.0
        if awareness is not None and hasattr(awareness, "engagement_pull"):
            engagement_boost = awareness.engagement_pull
            if engagement_boost >= 0.7:
                return True
        # 概率决策：基础意愿 × 频率调节 × 觉察加成
        freq_adjust = frequency_control_manager.get_or_create_frequency_control(
            self.stream_id
        ).get_talk_frequency_adjust()
        base_talk = global_config.chat.get_talk_value(self.stream_id)
        dynamic_prob = float(getattr(self, "_reply_probability", 0.5) or 0.5)
        combined_prob = base_talk * freq_adjust * eagerness * dynamic_prob
        _night_sup = getattr(self, "_night_reply_suppression", 0.0)
        if _night_sup > 0:
            combined_prob *= 1.0 - _night_sup
        # 觉察加成：参与度在 0.3~0.7 区间提供线性加成
        if 0.3 <= engagement_boost < 0.7:
            combined_prob *= 1.0 + engagement_boost
        # 内心独白加成：欲望等级 4~6 提供小幅加成
        if voice is not None and hasattr(voice, "reply_desire_level"):
            desire = voice.reply_desire_level
            if 4 <= desire < 7:
                combined_prob *= 1.0 + desire * 0.05
        chatterbox_penalty = float(getattr(self, "_chatterbox_penalty", 0.0) or 0.0)
        if chatterbox_penalty > 0:
            combined_prob *= max(0.12, 1.0 - min(0.75, chatterbox_penalty * 0.18))
        if pinged_msg is None and getattr(self, "_pattern_forced_observe", False):
            combined_prob *= 0.35
        if pinged_msg is None and getattr(self, "_defense_mode_active", False):
            combined_prob *= 0.65
        combined_prob = max(0.01, min(0.99, combined_prob))
        return random.random() < combined_prob

    def _shift_to_dormant(self, cause: str = "") -> None:
        """迁移至休息阶段"""
        try:
            snapshot = self._build_unified_flow_snapshot("shift_dormant")
            if snapshot.phase == FlowPhase.DORMANT.value and snapshot.is_resting:
                return
            self._apply_unified_flow_snapshot(
                self._copy_unified_flow_snapshot(
                    snapshot,
                    update={
                        "watch_state": WatchLevel.BLACKOUT.value,
                        "phase": FlowPhase.DORMANT.value,
                        "blocker": cause or "进入休息",
                    }
                ),
                watch_reason="shift_to_dormant",
                phase_reason=cause or "进入休息",
            )
            self._run_flow_side_effects(action_name="begin_rest", reason=cause)
            logger.info(f"{self.log_prefix} 迁移至休息: {cause}")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 迁移休息失败: {exc}")

    def _shift_to_engaged(self, cause: str = "") -> None:
        """迁移至活跃阶段"""
        try:
            snapshot = self._build_unified_flow_snapshot("shift_engaged")
            current = snapshot.phase
            if current == FlowPhase.ENGAGED.value:
                return
            if current in (FlowPhase.DORMANT.value, FlowPhase.LIGHT_REST.value):
                try:
                    _d6 = EnergyChainDimension.get_instance()
                    _nm = _d6._get_night_mode(self.stream_id)
                    _grace = 180.0
                    if _nm and _nm.phase in ("DAWN_RECOVERY", "ACTIVE_TWILIGHT"):
                        _grace = 120.0
                    elif _nm and _nm.phase == "DEEP_VALLEY":
                        _grace = 300.0
                    self._wake_grace_until = time.time() + _grace
                except Exception:
                    self._wake_grace_until = time.time() + 120.0
            self._apply_unified_flow_snapshot(
                self._copy_unified_flow_snapshot(
                    snapshot,
                    update={
                        "watch_state": WatchLevel.ENGAGED.value,
                        "phase": FlowPhase.ENGAGED.value,
                        "blocker": "",
                    }
                ),
                watch_reason="shift_to_engaged",
                phase_reason=cause or "进入参与",
            )
            if current in (FlowPhase.DORMANT.value, FlowPhase.LIGHT_REST.value):
                self._run_flow_side_effects(action_name="engage", conclude_rest=True)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 迁移活跃失败: {exc}")

    def _shift_to_pending(self, cause: str = "", anticipated: str = "") -> None:
        """迁移至等待阶段"""
        try:
            snapshot = self._build_unified_flow_snapshot("shift_pending")
            self._apply_unified_flow_snapshot(
                self._copy_unified_flow_snapshot(
                    snapshot,
                    update={
                        "watch_state": WatchLevel.ACTIVE_WATCH.value,
                        "phase": FlowPhase.PENDING.value,
                        "blocker": cause or "进入等待",
                    }
                ),
                watch_reason="shift_to_pending",
                phase_reason=cause or "进入等待",
            )
            if anticipated:
                self._run_flow_side_effects(action_name="begin_wait", anticipated=anticipated)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 迁移等待失败: {exc}")

    def _update_dynamic_ratio(self, now_ts: float):
        """动态更新回复比例和概率

        核心算法：
        1. 计算当前累计比例（用户消息 vs 机器人回复）
        2. 根据比例偏差调整目标回复概率
        3. 刷屏时自动降低回复概率
        4. 沉默时保持适度回复概率

        比例控制原则：
        - 用户发5条 → 机器人回复1-2条（目标比例40%）
        - 用户发20条 → 机器人回复8-10条（目标比例40-50%）
        - 刷屏时 → 机器人回复更少（20-30%）
        - 沉默时 → 机器人可多说（50-60%）
        """
        actual_user_msgs = len(self._user_msg_timeline)
        actual_bot_replies = len(self._bot_reply_timeline)

        if actual_bot_replies == 0:
            self._cumulative_ratio = 0.5
        else:
            self._cumulative_ratio = actual_bot_replies / max(1, actual_user_msgs)

        recent_user_msgs = sum(1 for t in self._user_msg_timeline if now_ts - t < 300.0)
        recent_bot_replies = sum(1 for t in self._bot_reply_timeline if now_ts - t < 300.0)

        if recent_bot_replies == 0:
            recent_ratio = 0.5
        else:
            recent_ratio = recent_bot_replies / max(1, recent_user_msgs)

        self._target_ratio = 0.25

        if self._is_in_burst:
            burst_intensity = min(1.0, self._burst_user_count / 10.0)
            self._target_ratio = 0.25 - (burst_intensity * 0.15)
            self._target_ratio = max(0.15, self._target_ratio)
        elif recent_user_msgs < 3 and recent_bot_replies < 2:
            self._target_ratio = 0.25
        elif recent_user_msgs > 20 and recent_bot_replies > 10:
            self._target_ratio = 0.2

        ratio_diff = self._cumulative_ratio - self._target_ratio
        if ratio_diff > 0.15:
            self._reply_probability = max(0.05, 0.35 - ratio_diff * 2.2)
        elif ratio_diff < -0.15:
            self._reply_probability = min(0.55, 0.35 - ratio_diff * 1.1)
        else:
            self._reply_probability = 0.35

        if self._is_in_burst:
            burst_penalty = min(0.3, (self._burst_user_count - 3) * 0.05)
            self._reply_probability = max(0.05, self._reply_probability - burst_penalty)

        if now_ts - self._last_bot_reply_ts < 10.0:
            self._reply_probability = max(0.05, self._reply_probability - 0.20)

        self._phase_confidence = 0.5 + (recent_ratio * 0.3)

        logger.debug(
            f"{self.log_prefix} 📊 动态比例: 用户{actual_user_msgs}条/机器人{actual_bot_replies}条 "
            f"实际比例={self._cumulative_ratio:.2f} 目标比例={self._target_ratio:.2f} "
            f"回复概率={self._reply_probability:.2f} "
            f"刷屏={self._is_in_burst}({self._burst_user_count}条)"
        )

    def _record_message_behavior(self, messages: List) -> None:
        try:
            from src.chat.heart_flow.behavior_analyzer import (
                get_behavior_analyzer,
            )

            analyzer = get_behavior_analyzer()
            for msg in messages:
                uid = getattr(msg, "user_id", "") or ""
                if not uid:
                    continue
                text = getattr(msg, "processed_plain_text", "") or ""
                polarity = self._estimate_text_polarity(text)
                analyzer.record(uid, polarity)
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _estimate_text_polarity(self, text: str) -> float:
        """极简文本极性估计：仅感叹号轻微正面，问号不参与负面判定"""
        content = str(text or "").strip()
        if not content:
            return 0.0
        # 问号属于正常提问行为，不应作为负面信号
        punct_positive = content.count("!") + content.count("！")
        length_bias = min(len(content) / 120.0, 1.0) * 0.05
        score = (punct_positive * 0.05) + length_bias
        return max(-0.3, min(0.3, score))

    def _record_interaction_mood(self, messages: List) -> None:
        """将交互记录到氛围追踪器"""
        try:
            from src.chat.heart_flow.emotion_stream import (
                get_channel_mood_tracker,
            )

            tracker = get_channel_mood_tracker()
            for msg in messages:
                text = getattr(msg, "processed_plain_text", "") or ""
                uid = getattr(msg, "user_id", "") or ""
                if text and uid:
                    tracker.record_inquiry(
                        channel_id=self.stream_id,
                        content=text,
                        author_id=uid,
                    )
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _log_energy_status(self, eagerness: float, reason: str) -> None:
        """输出能量状态日志（带间隔控制，避免刷屏）"""
        now = time.time()
        if now - self._last_energy_log_ts < self._energy_log_interval:
            return
        self._last_energy_log_ts = now
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            chat_energy = float(_ch.chat_pool) if _ch else 100.0
            chat_ceiling = float(_ch.chat_ceiling) if _ch else 100.0
            thinking_energy = float(_ch.thinking_value) if _ch else 100.0
            thinking_ceiling = float(_ch.thinking_ceiling) if _ch else 100.0
            freq_ctrl = frequency_control_manager.get_or_create_frequency_control(self.stream_id)
            freq_adjust = freq_ctrl.get_talk_frequency_adjust()
            logger.info(
                f"{self.log_prefix} 能量状态 | "
                f"聊天池={chat_energy:.1f}/{chat_ceiling:.0f} "
                f"思考池={thinking_energy:.1f}/{thinking_ceiling:.0f} "
                f"意愿={eagerness:.2f} "
                f"频率={freq_adjust:.2f}"
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 能量状态输出失败: {exc}")

    def _apply_plan_drain(self) -> None:
        """规划消耗思考值（动态计算）"""
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            base_cost = 3.0
            brain_cost = max(1.0, min(15.0, base_cost))
            if _ch:
                _ch.thinking_value = max(0, _ch.thinking_value - brain_cost)
                _ch.last_update = time.time()
                self._sync_runtime_resource_cache_from_d6()
                logger.info(f"{self.log_prefix} 消耗思考值 {brain_cost:.1f}，剩余思考值 {_ch.thinking_value:.1f}")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 规划消耗失败: {exc}")

    async def _compute_relation_metrics(self, messages: List) -> Dict[str, Any]:
        """聚合共享层/关系层/情绪层，优先从世界快照返回统一交互状态。"""
        # 世界快照含目标用户数据时直接转换，跳过逐源散读
        snap = getattr(self, "_tick_world_snapshot", None)
        if snap is not None and snap.target_user.user_id:
            self._last_user_id = str(snap.target_user.user_id or "").strip()
            rd = self._normalize_relation_snapshot(snap.to_relation_dict())
            rd.setdefault("relationship_candidates", [])
            try:
                from src.person_info.person_info import get_unified_profile_hub

                hub = get_unified_profile_hub()
                top_rels = hub.get_top_relationships(snap.target_user.user_id, limit=5)
                rd["relationship_candidates"] = [
                    (str(tid or "").strip(), float(w or 0.0)) for tid, w in top_rels if str(tid or "").strip()
                ]
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            rd["attribute_influences"] = self._build_attribute_influence_map(rd)
            rd = self._inject_realtime_emotion(rd)
            return self._normalize_relation_snapshot(rd)
        # 降级路径：无快照时委托 impression_evolution_hub 聚合
        user_id = ""
        if messages:
            for msg in reversed(messages):
                uid = getattr(msg, "user_id", "") or ""
                if self._is_human_message_obj(msg):
                    user_id = uid
                    self._last_user_id = user_id
                    break
        if not user_id:
            logger.debug(f"{self.log_prefix} 未找到有效用户ID，使用默认关系指标")
        try:
            hub = self._orch.get("impression_hub")
            result = await hub.assemble_relation_snapshot(user_id)
        except Exception as exc:
            logger.warning(f"{self.log_prefix} 关系度聚合失败，从各引擎回填真实值: {exc}")
            result = {
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
                "relationship_candidates": [],
            }
        result = self._normalize_relation_snapshot(result)
        # ── 真实数据回填：确保social/trust/annoyance等字段不为全零默认值 ──

        def _safe_float(key: str, default: float = 0.0) -> float:
            try:
                val = result.get(key, default)
                if val is None:
                    return default
                return float(val)
            except (TypeError, ValueError):
                return default

        _sv_val = _safe_float("social_value")
        _tv_val = _safe_float("trust_value")
        _af_val = _safe_float("affection")
        _needs_fill = _sv_val < 0.01 and _tv_val < 0.01 and _af_val < 0.01
        if _needs_fill:
            logger.debug(f"{self.log_prefix} 关系值全零检测到，启动回填 user={user_id}")
            try:
                _fill_source = None
                _snap = getattr(self, "_tick_world_snapshot", None)
                if _snap is not None and user_id:
                    _target_uid = str(getattr(getattr(_snap, "target_user", None), "user_id", "") or "").strip()
                    if _target_uid == str(user_id).strip():
                        _fill_source = _snap.to_relation_dict()
                if not isinstance(_fill_source, dict):
                    _rr = getattr(self, "_cached_relation_result", None)
                    if isinstance(_rr, dict):
                        _fill_source = _rr
                if isinstance(_fill_source, dict):
                    result["social_value"] = float(_fill_source.get("social_value", result.get("social_value", 0.0)) or 0.0)
                    result["affection"] = float(_fill_source.get("affection", result.get("affection", 0.0)) or 0.0)
                    result["trust_value"] = float(_fill_source.get("trust_value", result.get("trust_value", 0.0)) or 0.0)
                    result["annoyance_value"] = float(
                        _fill_source.get("annoyance_value", result.get("annoyance_value", 0.0)) or 0.0
                    )
                    _label = _fill_source.get("custom_label") or _fill_source.get("relationship")
                    if _label:
                        result["custom_label"] = str(_label)
                    _level = _fill_source.get("relationship_level")
                    if _level is not None:
                        result["relationship_level"] = int(_level)
                    _interactions = _fill_source.get("interaction_count")
                    if _interactions is not None:
                        result["interaction_count"] = int(_interactions)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            if _safe_float("annoyance_value") < 0.01:
                result["annoyance_value"] = 0.0
            # 从存在感引擎补充社交意愿/回避倾向
            _ps_fill = getattr(self, "_cached_presence_state", None)
            if _ps_fill:
                _sw_fill = float(getattr(_ps_fill, "social_willingness", 0.5) or 0.5)
                _aw_fill = float(getattr(_ps_fill, "avoidance_tendency", 0.0) or 0.0)
                if _sw_fill > 0.3:
                    result["social_value"] = max(result.get("social_value", 0.0), _sw_fill * 10.0)
                if _aw_fill > 0.3:
                    result["annoyance_value"] = max(result.get("annoyance_value", 0.0), _aw_fill * 20.0)
            # 从代谢约束补充心理压力/无聊驱动
            _mc_fill = getattr(self, "_cached_metabolism_constraints", None)
            if isinstance(_mc_fill, dict):
                _bored_fill = float(_mc_fill.get("boredom", 0.0) or 0.0)
                _pressure_fill = float(_mc_fill.get("psychological_pressure", 0.0) or 0.0)
                if _bored_fill > 5:
                    result["psychological_pressure"] = max(
                        result.get("psychological_pressure", 0.0),
                        min(50.0, _bored_fill * 0.5),
                    )
                if _pressure_fill > 0:
                    result["psychological_pressure"] = max(
                        result.get("psychological_pressure", 0.0),
                        _pressure_fill,
                    )
        result = self._normalize_relation_snapshot(result)
        result["attribute_influences"] = self._build_attribute_influence_map(result)
        result = self._inject_realtime_emotion(result)
        return self._normalize_relation_snapshot(result)

    def _build_attribute_influence_map(self, relation_state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """构建四主属性的独立影响矩阵，不把不同层揉成一个总分。"""
        pressure = float(relation_state.get("psychological_pressure", 0.0) or 0.0)
        annoyance = float(relation_state.get("annoyance_value", 0.0) or 0.0)
        trust_value = float(relation_state.get("trust_value", 0.0) or 0.0)
        affection = float(relation_state.get("affection", 0.0) or 0.0)
        shared_social = float(relation_state.get("shared_social_value", 0.0) or 0.0)
        relation_social = float(relation_state.get("social_value", 0.0) or 0.0)
        trauma = float(relation_state.get("trauma_score", 0.0) or 0.0)

        return {
            "thinking_value": {
                "psychological_pressure_penalty": round(-min(18.0, pressure * 0.12), 2),
                "trauma_penalty": round(-min(12.0, trauma * 1.4), 2),
            },
            "chat_value": {
                "affection_bonus": round(max(-6.0, min(8.0, affection * 0.05)), 2),
                "annoyance_penalty": round(-min(12.0, annoyance * 0.08), 2),
                "trust_buffer": round(max(-4.0, min(6.0, trust_value * 0.04)), 2),
            },
            "activity_level": {
                "shared_social_drive": round(max(-5.0, min(6.0, shared_social * 0.04)), 2),
                "pressure_drag": round(-min(10.0, pressure * 0.06), 2),
            },
            "social_value": {
                "relation_social_bias": round(max(-8.0, min(8.0, relation_social * 0.05)), 2),
                "trust_bias": round(max(-8.0, min(10.0, trust_value * 0.05)), 2),
                "annoyance_drag": round(-min(14.0, annoyance * 0.09), 2),
                "trauma_sensitivity": round(-min(10.0, trauma * 0.9), 2),
            },
        }

    def _get_self_behavior_style_hints(self, relation_snapshot: Optional[Dict[str, Any]] = None) -> List[str]:
        try:
            from src.modules.recall.self_behavior_learner import (
                get_self_behavior_learner,
            )

            learner = get_self_behavior_learner()
            relation_stage = ""
            if relation_snapshot:
                relation_stage = str(
                    relation_snapshot.get("custom_label") or relation_snapshot.get("relationship_level") or ""
                ).strip()
            return learner.get_style_hints(self.stream_id, relation_stage=relation_stage, limit=3)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 获取自我行为风格提示失败: {exc}")
            return []

    def _build_meme_injection(self) -> str:
        """构建梗/网络用语注入块(配置驱动)

        从 soul_engine.yaml 配置文件读取梗库 + LearningHub 已内化梗词，
        根据当前场景(时间/情绪/关系)筛选合适的"语气调料"注入 LLM 提示词。
        所有词汇和模板均来自外部配置，零硬编码。
        """
        now = time.time()
        try:
            _inj_cfg = self._soul_inject_config
            _ttl = float(_inj_cfg.get("meme_cache_ttl", 180))
        except Exception:
            _ttl = 180.0
        if self._meme_inject_cache and (now - self._meme_inject_cache_ts) < _ttl:
            return self._meme_inject_cache
        selected_memes: List[str] = []
        try:
            from src.core.learning_hub import LearningHub

            _hub = LearningHub(self.stream_id)
            _internalized = _hub.get_internalized()
            _practicing = _hub.get_practicing()
            for item in (_internalized + _practicing)[:20]:
                if item.content and len(item.content) <= 15:
                    selected_memes.append(item.content)
        except Exception as _e:
            logger.debug(f"{self.log_prefix} 梗库查询异常: {_e}")
        try:
            from src.chat.prompts.soul_config_loader import get_all_meme_words as _get_memes

            _config_words = _get_memes(limit=50)
            for w in _config_words:
                if w and w not in selected_memes:
                    selected_memes.append(w)
        except Exception as _ce:
            logger.debug(f"{self.log_prefix} 梗配置加载异常: {_ce}")
        import random as _rnd

        _rnd.shuffle(selected_memes)
        try:
            _max_pick = int(_inj_cfg.get("max_meme_pick", 18))
            _min_pick = int(_inj_cfg.get("min_meme_pick", 8))
        except Exception:
            _max_pick, _min_pick = 18, 8
        _pick_count = min(_max_pick, max(_min_pick, len(selected_memes)))
        _picked = selected_memes[:_pick_count]
        hour_now = datetime.datetime.now().hour
        is_late = (hour_now >= 0 and hour_now < 7) or (hour_now >= 23)
        snapshot = self._resolve_relation_view()
        annoyance_val = float(snapshot.get("annoyance_value", 0.0) or 0)
        affection_val = float(snapshot.get("affection", 0.0) or 0)
        lines = ["[🎭可用语气调料] 你可以在回复中自然地使用以下网络用语和梗(不必全用，挑合适的1-3个融入):"]
        lines.append(f"词汇池: {' / '.join(_picked[:_pick_count])}")
        if is_late and annoyance_val >= 10:
            try:
                from src.chat.prompts.soul_config_loader import (
                    get_sarcasm_templates as _get_sarc,
                )

                _level = "mid" if annoyance_val < 35 else "high"
                sarcasm_pool = _get_sarc(_level)
                _sarc_pick = _rnd.sample(sarcasm_pool, min(5, len(sarcasm_pool)))
                lines.append(f"[夜间烦躁可用] {' | '.join(_sarc_pick)}")
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
        elif affection_val >= 25 and annoyance_val < 15:
            try:
                from src.chat.prompts.soul_config_loader import (
                    get_warm_templates as _get_warm,
                )

                _warm_pick = _rnd.sample(_get_warm(count=6), min(4, len(_get_warm())))
                lines.append(f"[轻松友好可用] {' | '.join(_warm_pick)}")
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
        lines.append("注意: 这些词要自然融入句子中，不要生硬堆砌。像真人聊天一样顺手用出来。")
        result = "\n".join(lines)
        self._meme_inject_cache = result
        self._meme_inject_cache_ts = now
        return result

    def _check_reply_diversity(self) -> str:
        """检查最近回复的自我去重情况

        如果最近N条回复与彼此相似度过高，返回警告提示
        让 LLM 在本次回复中刻意换一种表达方式。
        """
        if len(self._recent_reply_texts) < 3:
            return ""
        recent_texts = [t for t, ts in self._recent_reply_texts[-8:]]
        if len(recent_texts) < 3:
            return ""
        total_pairs = 0
        similar_pairs = 0
        for i in range(len(recent_texts)):
            for j in range(i + 1, len(recent_texts)):
                total_pairs += 1
                a, b = recent_texts[i], recent_texts[j]
                if not a or not b:
                    continue
                set_a = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", a.lower()))
                set_b = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", b.lower()))
                if not set_a or not set_b:
                    continue
                overlap = len(set_a & set_b)
                union = len(set_a | set_b)
                if union > 0 and (overlap / union) > 0.55:
                    similar_pairs += 1
        if total_pairs == 0:
            return ""
        similarity_ratio = similar_pairs / total_pairs
        if similarity_ratio > 0.55:
            last_3 = [t[:25] for t in recent_texts[-3:]]
            return (
                f"[⚠️表达多样性警告] 你最近说的几条消息太相似了: {', '.join(last_3)}。"
                f"这次必须换一种说法！换个句式、换个角度、或者用不同的语气词。"
                f"禁止再说类似的内容。"
            )
        elif similarity_ratio > 0.35:
            return (
                "[💡表达建议] 最近回复有些重复倾向，这次试着换个新鲜的说法,"
                "可以用不同的语气词、不同的句式结构、或者从另一个角度回应。"
            )
        return ""

    def _record_reply_for_diversity(self, reply_text: str) -> None:
        """记录一条回复文本用于后续去重检测"""
        if not reply_text or not reply_text.strip():
            return
        clean = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", "", reply_text.strip())
        if len(clean) < 2:
            return
        self._recent_reply_texts.append((clean, time.time()))
        if len(self._recent_reply_texts) > self._max_reply_history:
            self._recent_reply_texts = self._recent_reply_texts[-self._max_reply_history :]

    async def _dispatch_followup_segments(
        self,
        llm_response: Any = None,
        target_message=None,
    ) -> None:
        """从 group_generator 的 reply_set 中取出补充段并延迟发送

        这是正式的多段回复路径: group_generator 根据机器人意愿
        决定是否补充段, 每段都是独立 LLM 生成的。
        旧的 || 分隔符检测作为 fallback 保留。
        """
        _followup_texts: List[str] = []
        if llm_response and hasattr(llm_response, "reply_set") and llm_response.reply_set:
            try:
                _rd = getattr(llm_response.reply_set, "reply_data", None)
                if _rd and len(_rd) > 0:
                    for item in _rd[1:]:
                        _text = getattr(item, "content", "") or ""
                        if isinstance(_text, str) and _text.strip():
                            _followup_texts.append(_text.strip())
            except Exception as _rse:
                logger.debug(f"{self.log_prefix} reply_set读取异常: {_rse}")
        if not _followup_texts and llm_response and hasattr(llm_response, "content"):
            _content = getattr(llm_response, "content", "") or ""
            if "||" in _content:
                try:
                    from src.chat.prompts.soul_config_loader import (
                        get_multi_segment_config as _get_ms_cfg,
                    )

                    _ms = _get_ms_cfg()
                    if _ms.get("enabled"):
                        _sep = _ms.get("separator", "||")
                        _segments = [s.strip() for s in _content.split(_sep) if s.strip()]
                        if len(_segments) > 1:
                            _followup_texts = _segments[1 : _ms.get("max_segments", 3)]
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
        if not _followup_texts:
            return
        try:
            from src.chat.prompts.soul_config_loader import (
                get_multi_segment_config as _get_ms_cfg2,
            )

            _ms2 = _get_ms_cfg2()
            _delay_range = tuple(_ms2.get("delay_range", (1.5, 4.0)))
        except Exception:
            _delay_range = (1.5, 4.0)
        logger.info(f"{self.log_prefix} 📝 准备发送{len(_followup_texts)}条补充段")
        for _idx, _f_text in enumerate(_followup_texts):
            if len(_f_text) < 2:
                continue
            _delay = random.uniform(_delay_range[0], _delay_range[1]) + _idx * 1.2

            async def _send_fu(__text: str, __delay_sec: float, __idx_num: int):
                await asyncio.sleep(__delay_sec)
                try:
                    await send_api.text_to_stream(
                        text=__text,
                        stream_id=self.chat_stream.stream_id,
                        reply_message=target_message,
                        set_reply=False,
                        typing=False,
                    )
                    self._record_reply_for_diversity(__text)
                    logger.info(f"{self.log_prefix} 📝 补充第{__idx_num + 2}段已发送: {__text[:30]}...")
                except Exception as _fe:
                    logger.debug(f"{self.log_prefix} 补充段发送失败: {_fe}")

            from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

            _task = self._spawn(_send_fu(_f_text, _delay, _idx), name=f"followup_segment_{_idx}")
            acquire_reply_coordinator().register_followup_task(
                self.stream_id,
                f"followup_segment_{_idx}_{int(time.time() * 1000)}",
                _task,
            )

    @property
    def _soul_inject_config(self) -> Dict[str, Any]:
        """灵魂引擎注入参数(从soul_engine.yaml读取)"""
        try:
            from src.chat.prompts.soul_config_loader import get_injection_config as _get_ic

            return _get_ic()
        except Exception:
            return {
                "meme_cache_ttl": 180,
                "max_meme_pick": 18,
                "min_meme_pick": 8,
            }

    def _is_acute_spamming(self) -> bool:
        """检测是否处于急性刷屏状态(短时间内密集发送)

        只有同时满足以下条件才返回True:
        - 用户最近N秒内发了≥3条消息
        - 重复检测判定为重复/纠缠
        这让冷拒/烦躁模式成为'急性反应'而非'常态'。
        高厌烦值作为底色始终存在，但不每条消息都触发完整拦截。
        """
        now = time.time()
        try:
            _recent = getattr(self, "_recent_user_timestamps", None)
            if not _recent:
                _recent = list(getattr(self, "_user_msg_timeline", []))
            if not _recent or len(_recent) < 3:
                return False
            _recent_sorted = sorted(_recent, reverse=True)
            if not _recent_sorted:
                return False
            _time_span = now - _recent_sorted[0]
            if _time_span > 45:
                return False
            _count_recent = sum(1 for ts in _recent_sorted if now - ts <= 45)
            if _count_recent < 3:
                return False
            _is_repeat = getattr(self, "_last_repeat_detected", False)
            if _is_repeat and _count_recent >= 4:
                return True
            _rapid = _count_recent >= 5
            return _rapid
        except Exception as _exc:
            logger.warning(f"{self.log_prefix} 重复检测异常: {_exc}")
            return False

    def _is_repeat_meme_or_low_info(self, repetition_signal: Optional[Dict[str, Any]]) -> bool:
        from src.chat.heart_flow.behavior_analyzer import get_behavior_analyzer

        return get_behavior_analyzer().detect_low_info_repeat_mode(repetition_signal)

    def _build_scene_reply_matrix(
        self,
        annoyance_val: float,
        affection_val: float,
        trauma_score: float,
        hour_now: int,
        relation_label: str,
    ) -> Dict[str, Any]:
        """分场合回复矩阵(配置驱动)

        从 soul_engine.yaml 的 scene_matrix 段读取全部场景定义，
        根据多维度状态匹配最合适的场景，返回风格参数。
        所有场景规则均可通过外部YAML编辑，零硬编码。
        """
        is_familiar = relation_label in ("比较熟悉", "关系亲密", "非常信任")
        is_stranger = relation_label in ("不太熟悉", "关系紧张")
        try:
            from src.chat.prompts.soul_config_loader import match_scene as _match

            matched = _match(
                annoyance_val=annoyance_val,
                affection_val=affection_val,
                trauma_score=trauma_score,
                hour_now=hour_now,
                relation_label=relation_label,
                is_familiar=is_familiar,
                is_stranger=is_stranger,
            )
            return {
                "tone": matched.get("tone", "neutral"),
                "length_range": tuple(matched.get("length_range", (15, 120))),
                "style_hints": list(matched.get("style_hints", [])),
                "forbidden_patterns": list(matched.get("forbidden_patterns", [])),
                "scene_name": matched.get("scene_name", "unknown"),
            }
        except Exception as _e:
            logger.debug(f"{self.log_prefix} 场景矩阵配置加载异常: {_e}")
            return {
                "tone": "normal_casual",
                "length_range": (15, 120),
                "style_hints": ["正常说话", "自然随意"],
                "forbidden_patterns": ["客服腔", "机械式回复"],
                "scene_name": "fallback",
            }

    def _build_reply_style_context(self, relation_snapshot: Optional[Dict[str, Any]] = None) -> str:
        snapshot = self._resolve_relation_view(relation_snapshot)
        relation_label = str(snapshot.get("custom_label") or "普通").strip() or "普通"
        trust_value = float(snapshot.get("trust_value", 0.0) or 0.0)
        annoyance_value = float(snapshot.get("annoyance_value", 0.0) or 0.0)
        affection = float(snapshot.get("affection", 0.0) or 0.0)
        pressure = float(snapshot.get("psychological_pressure", 0.0) or 0.0)
        trauma = float(snapshot.get("trauma_score", 0.0) or 0.0)
        self_style_hints = self._get_self_behavior_style_hints(snapshot)

        _hour_now = datetime.datetime.now().hour
        _is_deep_night = (0 <= _hour_now <= 6) or (_hour_now >= 23)
        _effective_annoyance = annoyance_value
        if _is_deep_night and annoyance_value >= 15:
            _night_amp = 1.0 + min(0.4, annoyance_value / 200.0)
            _effective_annoyance = min(100.0, annoyance_value * _night_amp)
        try:
            from src.chat.prompts.soul_config_loader import get_thresholds as _get_thr

            _thr = _get_thr()
            _cold_thr = float(_thr.get("cold_rejection", 35))
            _irrit_thr = float(_thr.get("irritated", 15))
            _trauma_thr = float(_thr.get("trauma_defense", 5))
        except Exception:
            _cold_thr, _irrit_thr, _trauma_thr = 35.0, 15.0, 5.0

        response_mode = "normal"
        if trauma >= _trauma_thr:
            response_mode = "trauma_defense"
        elif _effective_annoyance >= _cold_thr and self._is_acute_spamming():
            response_mode = "cold_rejection"
        elif _effective_annoyance >= _irrit_thr and self._is_acute_spamming():
            response_mode = "irritated"
        elif _effective_annoyance >= 70:
            response_mode = "cold_mood"
        elif _effective_annoyance >= 55:
            response_mode = "dismissive"
        elif _effective_annoyance >= 28:
            response_mode = "reluctant"
        elif _effective_annoyance >= _irrit_thr * 1.2:
            response_mode = "irritated_mood"

        target_uid = str(getattr(self, "_last_user_id", "") or "").strip()
        mood_val = ""
        attitude_val = ""
        stamina_val = 80.0
        layered_tone = "neutral"
        layered_length = "normal"
        layered_playfulness = 0.5
        if target_uid:
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import (
                    get_emotion_tracker,
                )

                tracker = get_emotion_tracker(self.stream_id)
                state = tracker.get_user_state(target_uid, create_if_missing=False)
                if state:
                    mood_val = getattr(state, "mood", "") or ""
                    attitude_val = getattr(state, "attitude", "") or ""
                    _neg_attrs = sum(
                        max(0, getattr(state, a, 0.0))
                        for a in ("anger", "sadness", "disgust", "fear", "shame", "guilt")
                    )
                    _pos_attrs = sum(
                        max(0, getattr(state, a, 0.0))
                        for a in ("joy", "gratitude", "pride", "surprise")
                    )
                    if _neg_attrs + _pos_attrs > 0:
                        stamina_val = max(10.0, 100.0 * (_pos_attrs / (_neg_attrs + _pos_attrs + 1.0)))
                resp_mode = tracker.get_layered_response_mode(target_uid) or {}
                layered_tone = str(resp_mode.get("tone", "neutral") or "neutral")
                layered_length = str(resp_mode.get("response_length", "normal") or "normal")
                layered_playfulness = float(resp_mode.get("playfulness", 0.5) or 0.5)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

        def _append(parts: List[str], text: str) -> None:
            payload = str(text or "").strip()
            if payload and payload not in parts:
                parts.append(payload)

        def _cap(text: str, max_chars: int = 42) -> str:
            payload = str(text or "").strip()
            if len(payload) <= max_chars:
                return payload
            return payload[: max_chars - 1].rstrip("，。；、 ") + "。"

        lines: List[str] = []
        _append(lines, f"你和对方算{relation_label}，按这个熟悉度自然说话。")

        if trauma >= _trauma_thr:
            _append(lines, "你现在有点防备，回复别太满，先留点距离。")
        elif annoyance_value >= 65:
            _append(lines, "你现在挺烦，句子短一点，别装热情。")
        elif annoyance_value >= 28:
            _append(lines, "你有点不耐烦，语气收一点，但别训人。")
        elif affection >= 50 and trust_value >= 35:
            _append(lines, "你对对方印象不错，可以自然暖一点。")
        elif affection <= -30:
            _append(lines, "你对对方偏冷，正常回就行，不用硬装熟。")
        else:
            _append(lines, "正常接话就行，像平时聊天那样说。")

        if mood_val and mood_val != "平静":
            _append(lines, f"你现在心情偏{mood_val}，这点底色自然带出来。")
        elif attitude_val and attitude_val != "中立":
            _append(lines, f"你现在对对方态度偏{attitude_val}，但别演得太满。")
        elif pressure >= 20 or stamina_val < 40:
            _append(lines, "你现在状态一般，少解释，别把句子拉太长。")

        response_mode_hints = {
            "cold_rejection": "如果非回不可，就很短很淡地回，带点疏离，但别变成流程话术。",
            "irritated": "可以带一点不耐烦，但别上纲上线，也别故意刺人。",
            "dismissive": "回得简短一点，别主动延长话题，也别写成敷衍模板。",
            "reluctant": "会回但别太积极，短句收住就行。",
            "cold_mood": "底色偏冷一点，但仍然像真人随口回话。",
            "irritated_mood": "语气稍微带一点刺就够了，不要训话。",
            "trauma_defense": "先保护自己，回复短一点，别硬撑热络。",
            "normal": "自然接话，不要总结，不要解释型输出。",
        }
        _append(lines, response_mode_hints.get(response_mode, response_mode_hints["normal"]))

        tone_hints = {
            "guarded": "语气谨慎一点，少解释。",
            "calm": "语气平一点，别堆情绪词。",
            "gentle": "语气放轻一点，别太冲。",
            "warm": "语气可以松一点，允许一点熟络感。",
            "neutral": "按普通群友口气说，不要端着。",
        }
        length_hints = {
            "concise": "长度偏短，点到就停。",
            "normal": "长度正常，别长篇大论。",
            "detailed": "可以多半句，但还是口语优先。",
        }
        _append(lines, tone_hints.get(layered_tone, "按普通群友口气说，不要端着。"))
        _append(lines, length_hints.get(layered_length, "长度正常，别长篇大论。"))
        if layered_playfulness >= 0.7:
            _append(lines, "可以稍微活一点，但别像在抖机灵任务。")
        elif layered_playfulness <= 0.25:
            _append(lines, "别硬搞活泼，稳稳地回就行。")

        if _is_deep_night and (annoyance_value >= 20 or stamina_val < 45):
            _append(lines, "现在偏晚，耐心更短，优先短句和自然收束。")

        if self_style_hints:
            _style_hint = "；".join(str(hint).strip() for hint in self_style_hints[:2] if str(hint).strip())
            if _style_hint:
                _append(lines, f"保留你自己的说话习惯：{_style_hint}")

        _append(lines, "自然口语，别客服腔，别写成说明书。")
        selected = lines[:5]
        if lines and lines[-1] not in selected:
            selected.append(lines[-1])
        return "\n".join(_cap(line) for line in selected[:6])

    def _log_relation_metrics(self, metrics: Dict[str, Any]) -> None:
        """输出后台公式算法结果（仅输出有意义的非零字段，避免全零刷屏）"""
        label = metrics.get("custom_label", "") or self._get_level_description(metrics.get("relationship_level", 2))
        _parts = [f"关系={label}"]
        _sv = float(metrics.get("shared_social_value", 0.0) or 0.0)
        _rsv = float(metrics.get("social_value", 0.0) or 0.0)
        if _sv > 0.01 or _rsv > 0.01:
            _show_sv = max(_sv, _rsv)
            if abs(_sv - _rsv) > 0.5:
                _parts.append(f"社交={_rsv:.1f}(共享{_sv:.1f})")
            else:
                _parts.append(f"社交={_show_sv:.1f}")
        _tv = float(metrics.get("trust_value", 0.0) or 0.0)
        if _tv > 0.01:
            _parts.append(f"信任={_tv:.1f}")
        _af = float(metrics.get("affection", 0.0) or 0.0)
        if _af > 0.01:
            _parts.append(f"好感={_af:.1f}")
        _an = float(metrics.get("annoyance_value", 0.0) or 0.0)
        if _an > 0.5:
            _parts.append(f"厌烦={_an:.1f}")
        _pr = float(metrics.get("psychological_pressure", 0.0) or 0.0)
        if _pr > 0.5:
            _parts.append(f"压力={_pr:.1f}")
        logger.info(f"{self.log_prefix} 🤝 " + " ".join(_parts))

    def _get_level_description(self, level: int) -> str:
        """获取关系等级的默认描述"""
        descriptions = {
            0: "关系紧张",
            1: "不太熟悉",
            2: "有些熟悉",
            3: "比较熟悉",
            4: "关系亲密",
            5: "非常信任",
        }
        return descriptions.get(level, "有些熟悉")

    def _log_final_status(self) -> None:
        """最终状态输出"""
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            relation_view = self._resolve_relation_view()
            chat_energy = float(_ch.chat_pool) if _ch else 50.0
            thinking_energy = float(_ch.thinking_value) if _ch else 50.0
            social_value = float(_ch.social_value) if _ch else 0.0
            trust_value = float(relation_view.get("trust_value", 0.0) or 0.0)
            annoyance_value = float(relation_view.get("annoyance_value", 0.0) or 0.0)
            freq_ctrl = frequency_control_manager.get_or_create_frequency_control(self.stream_id)
            freq_adjust = freq_ctrl.get_talk_frequency_adjust()
            logger.info(
                f"{self.log_prefix} 📋 "
                f"聊天池={chat_energy:.1f} "
                f"思考池={thinking_energy:.1f} "
                f"社交={social_value:.1f} "
                f"信任={trust_value:.1f} "
                f"厌烦={annoyance_value:.1f} "
                f"频率={freq_adjust:.2f}"
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 最终状态输出失败: {exc}")

    def _evaluate_impulse_factors(self, now: float, relation_result: Dict) -> Dict[str, Any]:
        """评估冲动因子：无聊、孤独、沉默、好感度等（已废弃，使用情感驱动核心）"""
        return {
            "silence_sec": 0.0,
            "matched_factors": [],
            "total_bonus": 0.0,
            "should_proactive": False,
        }

    def _log_impulse_factors(self, impulse: Dict[str, Any]) -> None:
        """输出冲动因子日志（已废弃，使用情感驱动核心）"""
        logger.debug(f"{self.log_prefix} 冲动因子日志已迁移至情感驱动核心")

    def _update_emotion_state(self, now: float, relation_result: Dict) -> Dict[str, Any]:
        """
        更新情感状态 - 真正的自主行为核心

        情感状态会随着时间自然累积/衰减：
        - 无聊感：随着沉默时间累积
        - 孤独感：随着未回复次数累积
        - 社交欲望：内在的想聊天的冲动
        """
        try:
            from src.chat.heart_flow.emotion_driven_core import (
                get_emotion_driven_core,
            )

            emotion_core = get_emotion_driven_core()
            # 获取沉默时长
            silence_sec = 0.0
            try:
                from src.chat.proactive.silence_watcher import (
                    get_quiet_monitor,
                )

                silence_sec = get_quiet_monitor().measure_silence_sec(self.stream_id)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            # 获取能量状态
            energy = 1.0
            mood = 0.5
            try:
                _d6 = EnergyChainDimension.get_instance()
                _ch = _d6._ensure_channel(self.stream_id)
                _chat_pool = float(_ch.chat_pool) if _ch else 50.0
                _chat_ceil = float(_ch.chat_ceiling) if _ch else 100.0
                _think_val = float(_ch.thinking_value) if _ch else 50.0
                _think_ceil = float(_ch.thinking_ceiling) if _ch else 100.0
                _activity = float(_ch.activity_level) if _ch else 50.0
                _social = float(_ch.social_value) if _ch else 0.0
                chat_factor = max(0.0, min(1.0, _chat_pool / max(_chat_ceil, 1.0)))
                thinking_factor = max(0.0, min(1.0, _think_val / max(_think_ceil, 1.0)))
                activity_factor = max(0.0, min(1.0, _activity / 100.0))
                social_value = _social
                social_factor = max(0.0, min(1.0, (social_value + 100.0) / 200.0))
                energy = max(
                    0.05,
                    min(
                        1.0,
                        thinking_factor * 0.4 + chat_factor * 0.25 + activity_factor * 0.2 + social_factor * 0.15,
                    ),
                )
                mood = 0.5 + social_value / 200.0
                # 好感值影响：从emotion_tracker获取当前用户好感度来调节mood
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker,
                    )

                    _et_mood = get_emotion_tracker(self.stream_id)
                    _latest_uid = self._resolve_latest_human_user_id()
                    if _latest_uid:
                        _u_state = _et_mood.get_user_state(_latest_uid, create_if_missing=False)
                        if _u_state:
                            _aff = float(getattr(_u_state, "affection", 0.0) or 0.0)
                            # affection [-100,100] 映射到 mood 偏移 [-0.15, +0.15]
                            mood += max(-0.15, min(0.15, _aff / 666.0))
                except Exception as exc:
                    logger.debug(f"{self.log_prefix} 情绪影响计算异常: {exc}")
                mood = max(0.1, min(0.95, mood))
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            # 整合外部状态
            _curiosity = 0.3
            try:
                from src.core.emotion_feedback_loop import get_emotion_feedback_loop

                _efl = get_emotion_feedback_loop()
                _efl_state = _efl._get_or_create_state(self.stream_id)
                _curiosity = round(_efl_state.curiosity, 3)
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
            emotion_core.integrate_external_state(
                self.stream_id,
                mood=max(0.0, min(1.0, mood)),
                energy=energy,
                curiosity=_curiosity,
            )
            # 情感时钟滴答
            emotion_core.tick(
                self.stream_id,
                silence_sec,
                unanswered=self._unanswered_bot_turns,
            )
            snapshot = emotion_core.get_state_snapshot(self.stream_id)
            self._cached_emotion_state = snapshot
            # 输出情感状态日志
            if snapshot["proactive_willingness"] >= 0.3:
                logger.info(
                    f"{self.log_prefix} 💕 "
                    f"无聊={snapshot['boredom']:.2f} "
                    f"环境疲劳={snapshot['environmental_fatigue']:.2f} "
                    f"孤独={snapshot['loneliness']:.2f} "
                    f"社交欲望={snapshot['social_desire']:.2f} "
                    f"主动意愿={snapshot['proactive_willingness']:.2f} "
                    f"感受: {snapshot['feeling']}"
                )
            return snapshot
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 情感状态更新失败: {exc}")
            return {}

    def _apply_r2_incentives(self, verdict, silence_sec: float) -> None:
        """对主动决策结果应用极轻微激励。

        这里只允许小幅修正，不能把原本应被否决的主动行为轻易翻成通过。
        """
        thresholds = get_heartfc_thresholds()
        # 长沉默激励：至少静默5分钟后才给极轻微鼓励
        if silence_sec > 300.0:
            _si = min(0.03, (silence_sec - 300.0) / 900.0 * 0.03)
            verdict.activation_bar = max(0.55, verdict.activation_bar - _si)
            verdict.breakdown["silence_incentive"] = _si
        # 好感加成
        _rel = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
        _aff = float(_rel.get("affection", 0.0) or 0.0)
        if self._metric_has_signal(_aff) and _aff > 75.0:
            _ab = min(0.02, (_aff - 75.0) / 25.0 * 0.02)
            verdict.activation_bar = max(0.55, verdict.activation_bar - _ab)
            verdict.breakdown["affection_incentive"] = _ab
        # 情感驱动：无聊与孤独
        _boredom = 0.0
        _loneliness = 0.0
        try:
            from src.chat.heart_flow.emotion_driven_core import get_emotion_driven_core

            _snap = get_emotion_driven_core().get_state_snapshot(self.stream_id)
            _boredom = float(_snap.get("boredom", 0.0) or 0.0)
            _loneliness = float(_snap.get("loneliness", 0.0) or 0.0)
        except Exception:
            _ps = getattr(self, "_cached_presence_state", None)
            if _ps:
                _out = float(getattr(_ps, "outward_attention", 0.5) or 0.5)
                _sw = float(getattr(_ps, "social_willingness", 0.5) or 0.5)
                _boredom = max(0.0, min(1.0, 1.0 - _out))
                _loneliness = max(0.0, min(1.0, _sw * 0.6))
        if _boredom > thresholds.boredom_activation_reduce:
            _bb = min(0.03, (_boredom - thresholds.boredom_activation_reduce) * 0.12)
            verdict.activation_bar = max(0.55, verdict.activation_bar - _bb)
            verdict.breakdown["boredom_incentive"] = _bb
        if _loneliness > 0.8:
            _lb = min(0.02, (_loneliness - 0.8) * 0.10)
            verdict.activation_bar = max(0.55, verdict.activation_bar - _lb)
            verdict.breakdown["loneliness_incentive"] = _lb
        # 好奇驱动
        _curiosity = 0.3
        if self._cached_presence_state:
            _curiosity = float(getattr(self._cached_presence_state, "curiosity_level", 0.3) or 0.3)
        if _curiosity > 0.8:
            verdict.activation_bar = max(0.55, verdict.activation_bar - 0.01)
            verdict.breakdown["curiosity_incentive"] = 0.01
        # 独白催促
        _desire = 0
        if self._cached_voice is not None:
            _desire = int(getattr(self._cached_voice, "reply_desire_level", 0) or 0)
        if _desire >= 9:
            _db = min(0.02, (_desire - 8) * 0.02)
            verdict.activation_bar = max(0.55, verdict.activation_bar - _db)
            verdict.breakdown["voice_desire_incentive"] = _db

    def _evaluate_proactive_decision(self, silence_sec: float) -> "ProactiveDecision":
        """采集多维信号并交由统一决策器融合评分。IntegrationHub 已在上层 async 中处理。"""
        from src.chat.proactive.proactive_decider import (
            get_proactive_decider,
            SignalBundle,
        )

        bundle = SignalBundle(channel_id=self.stream_id, silence_seconds=silence_sec)

        # 信号1: 情感就绪度
        try:
            from src.chat.heart_flow.emotion_driven_core import (
                get_emotion_driven_core,
            )

            snap = get_emotion_driven_core().get_state_snapshot(self.stream_id)
            bundle.emotional_readiness = float(snap.get("proactive_willingness", 0.0) or 0.0)
            bundle.boredom = float(snap.get("boredom", 0.0) or 0.0)
            bundle.loneliness = float(snap.get("loneliness", 0.0) or 0.0)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 决策情感信号采集(模块缺失，使用缓存回退): {exc}")
            # 回退: 从已缓存的代谢/存在态推算近似情感信号
            try:
                _ms = getattr(self, "_cached_metabolism_state", None)
                if _ms:
                    _mood = float(getattr(_ms, "mood_valence", 0.5) or 0.5)
                    _arousal = float(getattr(_ms, "mood_arousal", 0.5) or 0.5)
                    bundle.emotional_readiness = max(0.0, min(1.0, _mood * 0.5 + _arousal * 0.5))
                _ps = getattr(self, "_cached_presence_state", None)
                if _ps:
                    _out_att = float(getattr(_ps, "outward_attention", 0.5) or 0.5)
                    _soc_w = float(getattr(_ps, "social_willingness", 0.5) or 0.5)
                    bundle.boredom = max(0.0, min(1.0, 1.0 - _out_att))
                    bundle.loneliness = max(0.0, min(1.0, _soc_w * 0.6))
            except Exception:
                # 最后兜底: 给中性值而非0
                bundle.emotional_readiness = 0.3
                bundle.boredom = 0.2
                bundle.loneliness = 0.15

        # 信号2: 能量储备
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            _cp = float(_ch.chat_pool) if _ch else 50.0
            _cc = float(_ch.chat_ceiling) if _ch else 100.0
            _tv = float(_ch.thinking_value) if _ch else 50.0
            _tc = float(_ch.thinking_ceiling) if _ch else 100.0
            _al = float(_ch.activity_level) if _ch else 50.0
            chat_factor = max(0.0, min(1.0, _cp / max(_cc, 1.0)))
            thinking_factor = max(0.0, min(1.0, _tv / max(_tc, 1.0)))
            activity_factor = max(0.0, min(1.0, _al / 100.0))
            bundle.vitality_ratio = max(
                0.0,
                min(
                    1.0,
                    chat_factor * 0.35 + thinking_factor * 0.4 + activity_factor * 0.25,
                ),
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 决策能量信号采集失败: {exc}")

        # 信号4: 内容新鲜度（空闲主动路径给中性值）
        try:
            from src.core.content_state_tracker import (
                get_content_state_tracker,
            )

            _ = get_content_state_tracker()
            bundle.content_novelty = 0.5
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 决策内容信号采集失败: {exc}")

        # 信号5: 社交声望
        try:
            _d6 = EnergyChainDimension.get_instance()
            _ch = _d6._ensure_channel(self.stream_id)
            bundle.social_standing = float(_ch.social_value) if _ch else 0.0
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 决策社交信号采集失败: {exc}")

        # 信号6: 内心独白欲望
        if self._cached_voice and hasattr(self._cached_voice, "reply_desire_level"):
            try:
                bundle.inner_voice_desire = int(getattr(self._cached_voice, "reply_desire_level", 5) or 5)
            except Exception:
                bundle.inner_voice_desire = 5

        # 信号7: 意图池驱动强度
        try:
            from src.chat.proactive.intention_pool import get_intention_pool

            bundle.intention_drive = get_intention_pool().build_decision_signal(self.stream_id)
        except Exception:
            bundle.intention_drive = 0.0

        decider = get_proactive_decider()
        verdict = decider.evaluate(bundle)
        _intent_override = getattr(verdict, "intent_override", False)

        # F6：群体行为模式 → 动态调整 activation_bar
        _pattern_bar_delta = 0.0
        if self._cached_pattern_evidence:
            _top_pattern = None
            _top_conf = 0.0
            for _pe in self._cached_pattern_evidence:
                _pc = float(getattr(_pe, "confidence", 0.0) or 0.0)
                if _pc > _top_conf:
                    _top_conf = _pc
                    _top_pattern = _pe
            if _top_pattern is not None and _top_conf >= 0.45:
                _pt_val = getattr(getattr(_top_pattern, "pattern", None), "value", "") or ""
                if _pt_val == "newcomer_welcome":
                    _pattern_bar_delta = -0.12
                    verdict.breakdown["pattern_welcome_boost"] = 0.12
                elif _pt_val == "ritual_greeting":
                    _pattern_bar_delta = -0.08
                    verdict.breakdown["pattern_ritual_boost"] = 0.08
                elif _pt_val == "spectator_mode":
                    # F35：围观→参与转化——事件关联度高时可突破围观态
                    _event_relevance = 0.0
                    try:
                        if self._cached_self_references:
                            for _ref in self._cached_self_references:
                                _rt = getattr(_ref, "ref_type", None)
                                if _rt:
                                    _rv = self._normalize_self_reference_type(_rt)
                                    if _rv in (
                                        "discussed_as_topic",
                                        "quoted_reply",
                                    ):
                                        _event_relevance += float(getattr(_ref, "strength", 0.3) or 0.3)
                        _rel_snap35 = self._normalize_relation_snapshot(
                            getattr(self, "_last_relation_snapshot", None) or {}
                        )
                        if float(_rel_snap35.get("affection", 0.0) or 0.0) > 55:
                            _event_relevance += 0.15
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
                    if _event_relevance > 0.5:
                        _pattern_bar_delta = -0.02
                        verdict.breakdown["spectator_to_engage_convert"] = 0.02
                    elif _event_relevance > 0.25:
                        _pattern_bar_delta = 0.06
                        verdict.breakdown["spectator_partial_convert"] = -0.06
                    else:
                        _pattern_bar_delta = 0.15
                        verdict.breakdown["pattern_spectator_penalty"] = -0.15
                elif _pt_val in (
                    "heated_discussion",
                    "argument",
                    "conflict_escalation",
                ):
                    _pattern_bar_delta = 0.18
                    verdict.breakdown["pattern_heated_penalty"] = -0.18
                elif _pt_val == "celebration_wave":
                    _pattern_bar_delta = -0.06
                    verdict.breakdown["pattern_celebration_boost"] = 0.06
                elif _pt_val == "support_circle":
                    _pattern_bar_delta = -0.10
                    verdict.breakdown["pattern_support_boost"] = 0.10
                # F33：扩展仪式性行为识别——不当作普通问答
                elif _pt_val in (
                    "birthday_wish",
                    "holiday_greeting",
                    "congratulation",
                ):
                    _pattern_bar_delta = -0.10
                    verdict.breakdown["pattern_ritual_celebration"] = 0.10
                elif _pt_val == "chain_reply":
                    _pattern_bar_delta = -0.05
                    verdict.breakdown["pattern_chain_reply"] = 0.05
                elif _pt_val == "group_photo":
                    _pattern_bar_delta = -0.07
                    verdict.breakdown["pattern_group_photo"] = 0.07
                if abs(_pattern_bar_delta) > 0.001:
                    verdict.activation_bar = max(
                        0.08,
                        min(0.92, verdict.activation_bar + _pattern_bar_delta),
                    )
                # F33：缓存仪式行为标记供 planner 使用
                if _top_conf >= 0.45 and _pt_val in (
                    "newcomer_welcome",
                    "ritual_greeting",
                    "birthday_wish",
                    "holiday_greeting",
                    "congratulation",
                    "chain_reply",
                    "celebration_wave",
                    "support_circle",
                    "group_photo",
                ):
                    self._cached_ritual_behavior = _pt_val
        # F9：行为映射层——量化真值动态修改 arbiter 参数
        _rel_snap = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
        if _rel_snap:
            _affection = float(_rel_snap.get("affection", 0.0) or 0.0)
            _trust = float(_rel_snap.get("trust_value", 0.0) or 0.0)
            _annoyance = float(_rel_snap.get("annoyance_value", 0.0) or 0.0)
            _aversion = float(_rel_snap.get("aversion_value", 0.0) or 0.0)
            _has_affection_signal = self._metric_has_signal(_affection)
            _has_trust_signal = self._metric_has_signal(_trust)
            if _has_affection_signal and _has_trust_signal and _affection < 15 and _trust < 20:
                _affix_penalty = 0.10
                verdict.activation_bar = min(0.92, verdict.activation_bar + _affix_penalty)
                verdict.breakdown["relation_avoid_penalty"] = -_affix_penalty
            elif _has_affection_signal and _has_trust_signal and _affection > 70 and _trust > 60:
                _affix_boost = -0.08
                verdict.activation_bar = max(0.08, verdict.activation_bar + _affix_boost)
                verdict.breakdown["relation_affinity_boost"] = _affix_boost
            if _annoyance > 35 or _aversion > 30:
                _avoid_p = min(0.25, (_annoyance * 0.003) + (_aversion * 0.002))
                verdict.activation_bar = min(0.92, verdict.activation_bar + _avoid_p)
                verdict.breakdown["relation_annoyance_penalty"] = -_avoid_p
                # 高厌烦时同步压低融合分
                if _annoyance > 55:
                    _score_damp = min(0.15, (_annoyance - 55) * 0.005)
                    verdict.fused_score = max(0.0, verdict.fused_score - _score_damp)
                    verdict.breakdown["annoyance_score_damp"] = -_score_damp
        # F13+F30：话题归属——五类归属动态调整参与门槛
        try:
            _scene = self._orch.get("scene_state")
            _active_topics = _scene.active_topic_slots(limit=3)
            _dominant_ownership = "ambiguous_topic"
            _dominant_topic_obj = None
            for _t in _active_topics:
                _dominant_ownership = _t.ownership_type()
                _dominant_topic_obj = _t
                break
            if _dominant_ownership == "self_topic":
                _topic_boost = -0.10
                verdict.activation_bar = max(0.08, verdict.activation_bar + _topic_boost)
                verdict.breakdown["self_topic_chase"] = _topic_boost
            elif _dominant_ownership == "target_user_topic":
                _rel_snap30 = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
                _aff30 = float(_rel_snap30.get("affection", 0.0) or 0.0)
                _has_aff30_signal = self._metric_has_signal(_aff30)
                if _has_aff30_signal and _aff30 < 25:
                    _tp_penalty = 0.08
                    verdict.activation_bar = min(0.92, verdict.activation_bar + _tp_penalty)
                    verdict.breakdown["target_user_low_affinity"] = -_tp_penalty
                elif _has_aff30_signal and _aff30 > 60:
                    _tp_boost = -0.05
                    verdict.activation_bar = max(0.08, verdict.activation_bar + _tp_boost)
                    verdict.breakdown["target_user_high_affinity"] = _tp_boost
            elif _dominant_ownership == "shared_group_topic":
                _gp_boost = -0.04
                verdict.activation_bar = max(0.08, verdict.activation_bar + _gp_boost)
                verdict.breakdown["shared_group_topic_natural"] = _gp_boost
            elif _dominant_ownership == "external_topic":
                _et_penalty = 0.12
                verdict.activation_bar = min(0.92, verdict.activation_bar + _et_penalty)
                verdict.breakdown["external_topic_low_interest"] = -_et_penalty
            elif _dominant_ownership == "ambiguous_topic":
                _am_penalty = 0.06
                verdict.activation_bar = min(0.92, verdict.activation_bar + _am_penalty)
                verdict.breakdown["ambiguous_topic_conservative"] = -_am_penalty
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # F31：自我引用联合检测增强——多人讨论/引用发言时强力提升关注度
        try:
            if self._cached_self_references:
                _ref_types_found = set()
                _max_ref_strength = 0.0
                _discuss_count = 0
                _quote_count = 0
                for _ref in self._cached_self_references:
                    _rt = getattr(_ref, "ref_type", None)
                    _rs = float(getattr(_ref, "strength", 0.0) or 0.0)
                    if _rt:
                        _rt_val = self._normalize_self_reference_type(_rt)
                        _ref_types_found.add(_rt_val)
                        if _rs > _max_ref_strength:
                            _max_ref_strength = _rs
                        if _rt_val == "discussed_as_topic":
                            _discuss_count += 1
                        elif _rt_val == "quoted_reply":
                            _quote_count += 1
                if _discuss_count >= 2 or (_max_ref_strength > 0.6 and "discussed_as_topic" in _ref_types_found):
                    _self_ref_boost = min(0.18, 0.06 + _discuss_count * 0.04)
                    verdict.activation_bar = max(0.05, verdict.activation_bar - _self_ref_boost)
                    verdict.fused_score = min(1.0, verdict.fused_score + _self_ref_boost * 0.5)
                    verdict.breakdown["self_ref_group_discussion"] = _self_ref_boost
                elif "quoted_reply" in _ref_types_found and _max_ref_strength > 0.4:
                    _quote_boost = 0.06
                    verdict.activation_bar = max(0.08, verdict.activation_bar - _quote_boost)
                    verdict.breakdown["self_ref_quoted"] = _quote_boost
                elif "direct_at" in _ref_types_found or "nickname_called" in _ref_types_found:
                    verdict.breakdown["self_ref_direct_ack"] = 0.02
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # F24：创伤放大负面事件 + 厌烦放大回避倾向
        try:
            _rel_snap2 = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
            _trauma = float(_rel_snap2.get("trauma_score", 0.0) or 0.0)
            _annoy = float(_rel_snap2.get("annoyance_value", 0.0) or 0.0)
            if _trauma > 30:
                _trauma_amp = min(0.15, (_trauma / 100.0) * 0.3)
                verdict.activation_bar = min(0.92, verdict.activation_bar + _trauma_amp)
                verdict.breakdown["trauma_amplification"] = -_trauma_amp
            if _annoy > 30:
                _annoy_amp = min(0.18, (_annoy / 100.0) * 0.35)
                verdict.fused_score = max(0.0, verdict.fused_score - _annoy_amp)
                verdict.breakdown["annoyance_avoidance"] = -_annoy_amp
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # F32：内心旁白→行为策略联动——旁白建议观察/防御时压制参与欲
        try:
            if self._cached_narration_plan:
                _strategy = str(getattr(self._cached_narration_plan, "behavior_strategy", "") or "")
                _narr_mood = str(getattr(self._cached_narration_plan, "dominant_mood", "") or "")
                if _strategy == "observe_only":
                    _obs_penalty = 0.10
                    verdict.activation_bar = min(0.92, verdict.activation_bar + _obs_penalty)
                    verdict.breakdown["narration_observe_strategy"] = -_obs_penalty
                elif _strategy == "defensive":
                    _def_penalty = 0.14
                    verdict.activation_bar = min(0.92, verdict.activation_bar + _def_penalty)
                    verdict.fused_score = max(0.0, verdict.fused_score - 0.05)
                    verdict.breakdown["narration_defensive_strategy"] = -_def_penalty
                elif _strategy == "spectator_eat_melon":
                    _spec_penalty = 0.08
                    verdict.activation_bar = min(0.88, verdict.activation_bar + _spec_penalty)
                    verdict.breakdown["narration_spectator"] = -_spec_penalty
                elif _strategy == "engage_active":
                    _eng_boost = -0.06
                    verdict.activation_bar = max(0.08, verdict.activation_bar + _eng_boost)
                    verdict.breakdown["narration_engage_boost"] = _eng_boost
                if _narr_mood in ("annoyed", "tired", "overwhelmed"):
                    _mood_p = 0.05
                    verdict.activation_bar = min(0.92, verdict.activation_bar + _mood_p)
                    verdict.breakdown[f"narration_mood_{_narr_mood}"] = -_mood_p
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # 每小时回复频率疲劳：主动发言过于频繁时提升门槛
        _hr_count = self._current_hourly_proactive_reply_count()
        if _hr_count > 3:
            _freq_fatigue = min(0.18, (_hr_count - 3) * 0.045)
            verdict.activation_bar = min(0.92, verdict.activation_bar + _freq_fatigue)
            verdict.breakdown["reply_frequency_fatigue"] = -_freq_fatigue
        ignored_turns = max(0, int(self._unanswered_bot_turns))
        if ignored_turns:
            # 削减惩罚力度: 0.06/轮 上限0.15 (原: 0.10/轮 上限0.30)
            unanswered_penalty = min(0.15, ignored_turns * 0.06)
            if _intent_override:
                unanswered_penalty *= 0.3
            verdict.fused_score = max(0.0, verdict.fused_score - unanswered_penalty)
            verdict.breakdown["unanswered_penalty"] = -unanswered_penalty
            if verdict.rationale:
                verdict.rationale = f"{verdict.rationale} | 未回应轮数={ignored_turns}"
            else:
                verdict.rationale = f"未回应轮数={ignored_turns}"
        # 不再直接 should_proceed=False，改为提高 activation_bar
        if ignored_turns >= 2:
            _ignore_bar_raise = min(0.20, ignored_turns * 0.05)
            verdict.activation_bar = min(0.85, verdict.activation_bar + _ignore_bar_raise)
            verdict.breakdown["ignore_bar_raise"] = -_ignore_bar_raise
        # R2正向激励：统一调用共享方法
        self._apply_r2_incentives(verdict, bundle.silence_seconds)

        # SOC-03: 负面情绪影响主动发言欲望
        if verdict.should_proceed and self._cached_user_negative_emotion > 50:
            _neg_penalty = min(0.3, self._cached_user_negative_emotion / 200.0)
            if _intent_override:
                _neg_penalty *= 0.35
            verdict.fused_score = max(0.0, verdict.fused_score - _neg_penalty)
            verdict.breakdown["negative_emotion_suppress"] = -_neg_penalty
            if verdict.fused_score < verdict.activation_bar and not _intent_override:
                verdict.should_proceed = False
                verdict.rationale = f"用户负面情绪高({self._cached_user_negative_emotion:.0f})，抑制主动发言"
        # F34：多模态预算联动——刷图风暴/大量跳过时进入节流模式
        try:
            if self._cached_multimodal_summary:
                _total_media = int(self._cached_multimodal_summary.get("total_items", 0) or 0)
                _skipped = int(self._cached_multimodal_summary.get("skipped_count", 0) or 0)
                _cached_used = int(self._cached_multimodal_summary.get("cache_hit_count", 0) or 0)
                _is_storm = str(self._cached_multimodal_summary.get("storm_mode", "") or "") == "active"
                if _total_media >= 5 and _skipped >= 3:
                    _media_throttle = 0.08
                    verdict.activation_bar = min(0.92, verdict.activation_bar + _media_throttle)
                    verdict.breakdown["media_throttle_many_skipped"] = -_media_throttle
                elif _is_storm:
                    _storm_p = 0.12
                    verdict.activation_bar = min(0.92, verdict.activation_bar + _storm_p)
                    verdict.breakdown["media_storm_throttle"] = -_storm_p
                if _total_media > 0 and _skipped == _total_media:
                    verdict.breakdown["media_all_skipped_note"] = 1.0
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # ═══════════════════════════════════════════
        #  统一多因子决策融合（替代散装 F26/F27/F28 硬阈值）
        #  所有能量/睡眠/上限/存在态/关系/话题/模式/引用/
        #  旁白/多模态/围观 走同一套 sigmoid+疲劳曲线+连续映射算法
        # ═══════════════════════════════════════════
        try:
            from src.core.multi_factor_decision_engine import (
                get_multi_factor_engine,
                FactorInput,
            )

            _mf = get_multi_factor_engine()
            _inp = FactorInput()
            try:
                _d6 = EnergyChainDimension.get_instance()
                _ch = _d6._ensure_channel(self.stream_id)
                _ecv = float(_ch.chat_pool) if _ch else 50.0
                _etv = float(_ch.thinking_value) if _ch else 50.0
                _eav = float(_ch.activity_level) if _ch else 50.0
                _esv = float(_ch.social_value) if _ch else 0.0
                _e_cc = max(float(_ch.chat_ceiling) if _ch else 100.0, 1.0)
                _e_tc = max(float(_ch.thinking_ceiling) if _ch else 100.0, 1.0)
                _inp.energy_chat_ratio = max(0.0, min(1.0, _ecv / _e_cc))
                _inp.energy_think_ratio = max(0.0, min(1.0, _etv / _e_tc))
                _inp.energy_activity_ratio = max(0.0, min(1.0, _eav / 100.0))
                _inp.energy_social_ratio = max(0.0, min(1.0, (_esv + 100.0) / 200.0))
                _inp.energy_composite = max(
                    0.03,
                    _inp.energy_think_ratio * 0.35
                    + _inp.energy_chat_ratio * 0.28
                    + _inp.energy_activity_ratio * 0.22
                    + _inp.energy_social_ratio * 0.15,
                )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            if self._cached_night_phase:
                _phase_val = getattr(self._cached_night_phase, "value", None)
                if _phase_val is not None:
                    _inp.night_phase = self._normalized_night_phase_value() or str(_phase_val).lower()
                else:
                    _inp.night_phase = self._normalized_night_phase_value() or getattr(
                        self._cached_night_phase, "name", "UNKNOWN"
                    )
                _ncs = None
                try:
                    from src.core.night_cycle_system import get_night_cycle

                    _ncs = get_night_cycle(self.stream_id)
                except Exception as _exc:
                    logger.debug(f"{self.log_prefix} 夜间节律对象获取失败: {_exc}")
                try:
                    _d6 = EnergyChainDimension.get_instance()
                    _nm = _d6._get_night_mode(self.stream_id)
                    if _nm:
                        _inp.sleep_debt = max(0.0, 1.0 - _nm.prob_multiplier)
                        _inp.sleep_peek_budget = _nm.prob_multiplier >= 0.3
                    else:
                        _inp.sleep_debt = 0.0
                        _inp.sleep_peek_budget = True
                except Exception:
                    _inp.sleep_debt = 0.0
                    _inp.sleep_peek_budget = True
                if _ncs is not None:
                    _inp.sleep_reply_budget = _ncs.evaluate_sleep_reply_budget()
                    # GAP-A：熬夜压力与亢奋链数据
                    try:
                        _pressure_info = _ncs.get_overnight_pressure_breakdown()
                        _inp.overnight_pressure = float(_pressure_info.get("total", 0.0))
                        self._overnight_pressure_total = _inp.overnight_pressure
                        _arousal_info = _ncs.get_night_expression_profile()
                        _inp.arousal_chain_state = str(_arousal_info.get("current_state", "normal"))
                        _inp.is_burnthrough = bool(_arousal_info.get("active_template") == "burnthrough")
                        _raw_expr = _arousal_info.get("raw_expression", {}) or {}
                        _inp.burnthrough_impulse = float(_raw_expr.get("impulse", 0.0))
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
            if self._cached_presence_state:
                _inp.social_willingness = float(getattr(self._cached_presence_state, "social_willingness", 0.5) or 0.5)
                _inp.watch_willingness = float(getattr(self._cached_presence_state, "watch_willingness", 0.5) or 0.5)
                _inp.avoidance_tendency = float(getattr(self._cached_presence_state, "avoidance_tendency", 0.0) or 0.0)
                _inp.quiet_preference = float(getattr(self._cached_presence_state, "quiet_preference", 0.2) or 0.2)
                _inp.curiosity_level = float(getattr(self._cached_presence_state, "curiosity_level", 0.3) or 0.3)
                _inp.outward_attention = float(getattr(self._cached_presence_state, "outward_attention", 0.5) or 0.5)
            _rel_mf = self._normalize_relation_snapshot(getattr(self, "_last_relation_snapshot", None) or {})
            _inp.affection = float(_rel_mf.get("affection", 0.0) or 0.0)
            _inp.trust_value = float(_rel_mf.get("trust_value", 0.0) or 0.0)
            _inp.annoyance_value = float(_rel_mf.get("annoyance_value", 0.0) or 0.0)
            _inp.trauma_score = float(_rel_mf.get("trauma_score", 0.0) or 0.0)
            _inp.aversion_value = float(_rel_mf.get("aversion_value", 0.0) or 0.0)
            try:
                _scene_mf = self._orch.get("scene_state")
                _topics_mf = _scene_mf.active_topic_slots(limit=3)
                for _t in _topics_mf:
                    _inp.topic_ownership = _t.ownership_type()
                    break
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            if self._cached_pattern_evidence:
                _top_pe = None
                _top_pc = 0.0
                for _pe in self._cached_pattern_evidence:
                    _pc = float(getattr(_pe, "confidence", 0.0) or 0.0)
                    if _pc > _top_pc:
                        _top_pc = _pc
                        _top_pe = _pe
                if _top_pe:
                    _pt_enum = getattr(_top_pe, "pattern", None)
                    _inp.dominant_pattern = (
                        (_pt_enum.value if hasattr(_pt_enum, "value") else str(_pt_enum)) if _pt_enum else ""
                    )
                    _inp.pattern_confidence = _top_pc
                    _inp.is_spectator_mode = _inp.dominant_pattern == "spectator_mode"
            if self._cached_self_references:
                _ref_types_set = set()
                _max_rs = 0.0
                _discuss_n = 0
                _quote_n = 0
                for _ref in self._cached_self_references:
                    _rt = getattr(_ref, "ref_type", None)
                    _rs = float(getattr(_ref, "strength", 0.0) or 0.0)
                    if _rt:
                        _rv = _rt.value if hasattr(_rt, "value") else str(_rt)
                        _ref_types_set.add(_rv)
                        if _rs > _max_rs:
                            _max_rs = _rs
                        if _rv == "discussed_as_topic":
                            _discuss_n += 1
                        elif _rv == "quoted_reply":
                            _quote_n += 1
                _inp.self_ref_types = list(_ref_types_set)
                _inp.self_ref_max_strength = _max_rs
                _inp.discuss_as_topic_count = _discuss_n
                _inp.quoted_reply_count = _quote_n
                _inp.direct_at = "direct_at" in _ref_types_set or "nickname_called" in _ref_types_set
                if _inp.is_spectator_mode:
                    _event_rel = 0.0
                    for _ref in self._cached_self_references:
                        _rt2 = getattr(_ref, "ref_type", None)
                        if _rt2:
                            _rv2 = _rt2.value if hasattr(_rt2, "value") else str(_rt2)
                            if _rv2 in ("discussed_as_topic", "quoted_reply"):
                                _event_rel += float(getattr(_ref, "strength", 0.3) or 0.3)
                    if _inp.affection > 55:
                        _event_rel += 0.15
                    _inp.event_relevance = _event_rel
            if self._cached_narration_plan:
                _inp.narration_strategy = str(getattr(self._cached_narration_plan, "behavior_strategy", "") or "")
                _inp.narration_mood = str(getattr(self._cached_narration_plan, "dominant_mood", "") or "")
            if self._cached_multimodal_summary:
                _inp.media_total_items = int(self._cached_multimodal_summary.get("total_items", 0) or 0)
                _inp.media_skipped_count = int(self._cached_multimodal_summary.get("skipped_count", 0) or 0)
                _inp.media_storm_active = str(self._cached_multimodal_summary.get("storm_mode", "") or "") == "active"
            if self._cached_metabolism_state:
                _inp.consecutive_active_minutes = float(
                    getattr(
                        self._cached_metabolism_state,
                        "consecutive_active_minutes",
                        0.0,
                    )
                    or 0.0
                )
            _inp.hourly_reply_used = self._current_hourly_proactive_reply_count()
            _inp.intention_drive = bundle.intention_drive
            _inp.intent_override_candidate = _intent_override and bundle.intention_drive >= 0.78
            try:
                from src.core.state_coupling_matrix import get_coupling_engine

                _scme = get_coupling_engine()
                _scme_snap = _scme.build_full_snapshot(
                    metabolism=getattr(self, "_cached_metabolism_state", None),
                    presence=getattr(self, "_cached_presence_state", None),
                    relation=self._last_relation_snapshot,
                    night_phase=getattr(self, "_cached_night_phase", None),
                    group_scene=getattr(self, "_cached_scene_snapshot", None),
                    watch_state=self._watch_level_value(),
                )
                if _scme_snap:
                    _inp.coupling_boredom_mod = float(_scme_snap.get("boredom_to_loafing_mod", 0.0) or 0.0)
                    _inp.coupling_activity_mod = float(_scme_snap.get("activity_behavior_mod", 0.0) or 0.0)
                    _inp.coupling_style_tier = str(_scme_snap.get("style_tier", "") or "")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            try:
                from src.core.impression_evolution_hub import (
                    get_impression_hub,
                )

                _ihub = get_impression_hub(self.stream_id)
                _focus_uid = str(getattr(bundle, "anchor_user_id", "") or "")
                if _focus_uid:
                    _isum = _ihub.get_impression_summary(_focus_uid)
                    if _isum.get("exists"):
                        _inp.impression_narrative_type = _isum.get("narrative_type", "")
                        _inp.impression_affection_proxy = _isum.get("affection", 0.0)
                        _inp.impression_trust_proxy = _isum.get("trust_value", 0.0)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            try:
                _slh = self._orch.get("skill_hub")
                if _slh:
                    _active_skills = _slh.get_active_skills()
                    if _active_skills:
                        _avg_cost = sum(s.cost_modifier() for s in _active_skills) / max(1, len(_active_skills))
                        _inp.skill_cost_modifier = _avg_cost
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            try:
                from src.core.multimodal_semantic_bridge import (
                    get_multimodal_semantic_bridge,
                )

                _mm_bridge = get_multimodal_semantic_bridge()
                _mm_cached = getattr(self, "_cached_mm_bridge_result", None)
                if isinstance(_mm_cached, dict):
                    _inp.mm_avg_valence = float(_mm_cached.get("avg_valence", 0.0) or 0.0)
                    _inp.mm_engagement = float(_mm_cached.get("avg_engagement", 0.0) or 0.0)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            # 仪表盘ActionVerdict预检：L2/L3裁定结果注入决策器
            try:
                _dv = self._get_dashboard_verdict()
                _dv_urgency = str(_dv.get("urgency", "") or "")
                _dv_reply = _dv.get("reply", False)
                _dv_process = _dv.get("process", True)
                _dv_confidence = float(_dv.get("confidence", 0.5) or 0.5)
                _dv_path = str(_dv.get("path", "") or "")
                if not _dv_process:
                    verdict.should_proceed = False
                    verdict.activation_bar = min(0.95, verdict.activation_bar + 0.25)
                    verdict.rationale = f"{verdict.rationale or ''} | 仪表盘L1阻断({_dv.get('reason', '')[:40]})"
                elif not _dv_reply and _dv_urgency in ("不回复", "跳过"):
                    _skip_penalty = 0.08 * _dv_confidence
                    verdict.activation_bar = min(0.90, verdict.activation_bar + _skip_penalty)
                    verdict.fused_score = max(0.0, verdict.fused_score - _skip_penalty * 0.4)
                    verdict.breakdown["dashboard_l2_skip"] = -_skip_penalty
                    logger.debug(f"{self.log_prefix} 仪表盘L2跳过: urgency={_dv_urgency} path={_dv_path}")
                elif _dv_reply and _dv_urgency in ("立即回复", "尽快回复"):
                    _reply_boost = 0.05 * _dv_confidence
                    verdict.activation_bar = max(0.05, verdict.activation_bar - _reply_boost)
                    verdict.breakdown["dashboard_l2_boost"] = _reply_boost
                _dv_tone = str(_dv.get("tone", "") or "")
                if _dv_tone:
                    verdict.rationale = f"{verdict.rationale or ''} | 语气建议={_dv_tone}"
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            _mf_verdict = _mf.evaluate(_inp)
            _mf_verdict.apply_to_arbiter_verdict(verdict)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 多因子融合引擎异常，回退到原有逻辑: {exc}")
        # GAP-R：安全边界融合——直接修改verdict的activation_bar和fused_score
        try:
            from src.core.safety_boundary_fusion import (
                get_safety_fusion_engine,
                ContextualDangerFactors,
                ThreatLevel,
            )

            if not self._safety_fusion_initialized:
                self._safety_fusion_initialized = True
            _sfe = get_safety_fusion_engine()
            _night_r = getattr(self, "_cached_night_phase", None)
            _is_late_night = False
            if _night_r and hasattr(_night_r, "value"):
                _is_late_night = self._normalized_night_phase_value() in (
                    "deep_sleep",
                    "light_sleep",
                    "drowsy",
                    "deep_valley",
                )
            _ctx_factors = ContextualDangerFactors(
                is_late_night=_is_late_night,
                bot_is_discussing_sensitive_topic=bool(self._cached_ritual_behavior),
                recent_negative_emotion_spike=self._cached_user_negative_emotion > 50,
                message_contains_url=False,
            )
            _safety_result = _sfe.assess(
                [],
                contextual_factors=_ctx_factors,
                current_arbiter_activation_bar=getattr(verdict, "activation_bar", 0.5),
                current_arbiter_fused_score=getattr(verdict, "fused_score", 0.5),
            )
            _new_bar, _new_score = _sfe.apply_to_verdict(verdict, _safety_result)
            self._cached_safety_assessment = _safety_result.to_dict()
            if _safety_result.overall_level != ThreatLevel.NONE:
                logger.info(
                    f"{self.log_prefix} [GAP-R] 安全评估: {_safety_result.overall_level.label()} "
                    f"bar={_new_bar:.3f} score={_new_score:.3f}"
                )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            _inj_msgs = []
            self.build_planner_injection_prompt(_inj_msgs)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            self.register_cross_engine_outputs()
        except Exception as _e:
            logger.debug(f"异常: {_e}")

        return verdict


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

class StrategyRelationStyleMixin:
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
        relation_label = str(
            snapshot.get("personal_impression")
            or snapshot.get("relationship")
            or snapshot.get("legacy_relationship_label")
            or snapshot.get("custom_label")
            or "普通"
        ).strip() or "普通"
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
        _append(lines, f"你对对方的个人印象是{relation_label}，按这个判断自然说话。")

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

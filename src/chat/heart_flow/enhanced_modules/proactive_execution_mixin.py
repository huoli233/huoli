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

class EnhancedProactiveExecutionMixin:
    def _build_context_execution_block(
        self,
        target_message: Optional[Any],
        voice_conclusion: Optional[Any] = None,
        repetition_signal: Optional[Dict[str, Any]] = None,
        decision_context_packet: Optional[Any] = None,
        relation_snapshot: Optional[Dict[str, Any]] = None,
    ) -> str:
        """统一整理人格、前情、重复和独白执行依据。"""
        parts: List[str] = []

        persona_hint = self._build_persona_hint()
        if persona_hint:
            parts.append(persona_hint)

        target_name = "对方"
        if target_message is not None:
            target_name = (
                str(
                    getattr(target_message, "user_nickname", "")
                    or getattr(target_message, "nickname", "")
                    or getattr(target_message, "user_id", "")
                    or "对方"
                ).strip()
                or "对方"
            )

        lines: List[str] = ["[前情执行块]", f"- 当前接话对象: {target_name}"]
        if decision_context_packet is not None:
            packet_context = str(decision_context_packet.compact_context() or "").strip()
            if packet_context:
                lines.append(packet_context)

        relation_snapshot = self._resolve_relation_view(relation_snapshot)
        custom_label = str(relation_snapshot.get("custom_label", "") or "").strip()
        social_value = float(relation_snapshot.get("social_value", 0.0) or 0.0)
        trust_value = float(relation_snapshot.get("trust_value", 0.0) or 0.0)
        annoyance_value = float(relation_snapshot.get("annoyance_value", 0.0) or 0.0)
        relation_bits: List[str] = []
        if custom_label:
            relation_bits.append(f"关系={custom_label}")
        relation_bits.append(f"社交={social_value:.1f}")
        if self._metric_has_signal(trust_value):
            relation_bits.append(f"信任={trust_value:.1f}")
        if self._metric_has_signal(annoyance_value):
            relation_bits.append(f"烦躁={annoyance_value:.1f}")
        lines.append(f"- 当前关系底色: {' '.join(relation_bits)}")

        cached_memory_hint = str(getattr(self, "_latest_memory_hint", "") or "").strip()
        if cached_memory_hint:
            lines.append(f"- 已检索到前情/记忆: {cached_memory_hint[:220]}")

        continuity_context = self._build_self_continuity_context()
        if continuity_context:
            lines.append(f"- 你自己刚才的延续线索: {continuity_context.replace(chr(10), '；')[:220]}")

        self_memory = self._build_self_reply_memory()
        if self_memory:
            lines.append(f"- 你自己最近说过的话: {self_memory.replace(chr(10), '；')[:220]}")

        if repetition_signal and repetition_signal.get("detected"):
            repeat_reason = str(repetition_signal.get("reason", "重复输入") or "重复输入")
            exact_repeat_count = int(repetition_signal.get("exact_repeat_count", 0) or 0)
            extra = f"，近轮重复约{exact_repeat_count}次" if exact_repeat_count > 0 else ""
            lines.append(f"- 重复/施压信号: {repeat_reason[:120]}{extra}")
            if repetition_signal.get("latest_matches_repeat"):
                lines.append("- 处理原则: 先按前情延续、复读、玩梗或等你接话来理解，不要把短句硬当新话题。")

        if voice_conclusion is not None:
            voice_guard = self._build_voice_execution_guard(voice_conclusion, target_message)
            if voice_guard:
                parts.append(voice_guard)

        lines.append("- 执行要求: 规划和回复都必须顺着前情、关系底色和刚才那句心里话继续，不要突然切成说明书口吻。")
        parts.append("\n".join(lines))
        return "\n\n".join(part for part in parts if part)

    def _emit_reply_generation_summary(
        self,
        target_message: Optional[Any],
        style_route: Dict[str, Any],
        relation_snapshot: Optional[Dict[str, Any]] = None,
        context_execution_block: str = "",
        extra_info: str = "",
        source: str = "reply",
    ) -> None:
        """在真正生成回复前输出一条简短摘要，便于校验语气和前情是否接通。"""
        try:
            snapshot = self._resolve_relation_view(relation_snapshot)
            target_uid = ""
            if target_message is not None:
                target_uid = str(getattr(target_message, "user_id", "") or "").strip()
            _final_annoy = float(snapshot.get("annoyance_value", 0.0) or 0.0)
            _final_pressure = float(snapshot.get("psychological_pressure", 0.0) or 0.0)
            mode_summary = "unknown/unknown"
            if target_uid:
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker,
                    )

                    _tracker = get_emotion_tracker(self.stream_id)
                    _cached_es = getattr(self, "_resolved_emo_state", None)
                    if _cached_es is not None:
                        _t_state = _cached_es
                    else:
                        _t_state = _tracker.get_user_state(target_uid, create_if_missing=False)
                    if _t_state:
                        _direct_annoy = float(getattr(_t_state, "annoyance", -1) or -1)
                        if _direct_annoy >= 0:
                            _final_annoy = round(_direct_annoy, 1)
                        _direct_press = float(getattr(_t_state, "psychological_pressure", -1) or -1)
                        if _direct_press >= 0:
                            _final_pressure = round(_direct_press, 1)
                    response_mode = _tracker.get_layered_response_mode(target_uid)
                    mode_summary = (
                        f"{str(response_mode.get('tone', 'neutral') or 'neutral')}"
                        f"/{str(response_mode.get('response_length', 'normal') or 'normal')}"
                    )
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")

            logger.info(
                f"{self.log_prefix} 回复生成摘要 "
                f"来源={source} "
                f"对象={target_uid[:8] if target_uid else 'none'} "
                f"形式={str(style_route.get('reply_style', 'direct') or 'direct')} "
                f"引用={'是' if bool(style_route.get('quote_message', False)) else '否'} "
                f"引用策略={str(style_route.get('quote_policy', 'quote_reply' if style_route.get('quote_message') else 'none') or 'none')} "
                f"模式={mode_summary} "
                f"烦躁={_final_annoy:.1f} "
                f"压力={_final_pressure:.1f} "
                f"前情块={'有' if context_execution_block else '无'} "
                f"提示长度={len(extra_info)}"
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _build_voice_execution_summary(self, voice_conclusion: Any, target_message: Optional[Any]) -> Dict[str, str]:
        """统一生成独白摘要，供日志、回复原因和 extra_info 复用。"""
        if voice_conclusion is None:
            return {
                "thinking": "",
                "mood": "",
                "reason": "",
                "panel": "",
                "log": "",
            }

        thinking = str(getattr(voice_conclusion, "thinking", "") or "").strip()
        mood = str(getattr(voice_conclusion, "current_mood", "") or "").strip()
        target_name = "对方"
        if target_message is not None:
            target_name = (
                str(
                    getattr(target_message, "user_nickname", "") or getattr(target_message, "user_id", "") or "对方"
                ).strip()
                or "对方"
            )

        pieces: List[str] = []
        if thinking:
            pieces.append(f"想法={thinking}")
        if mood:
            pieces.append(f"情绪={mood}")
        if target_name:
            pieces.append(f"目标={target_name}")

        reason = " | ".join(pieces)
        panel_lines: List[str] = []
        if thinking:
            panel_lines.append(f"[内心思考] {thinking}")
        if mood:
            panel_lines.append(f"[当前情绪] {mood}")

        return {
            "thinking": thinking,
            "mood": mood,
            "reason": reason,
            "panel": "\n".join(panel_lines),
            "log": " | ".join(pieces),
        }

    def _query_heartflow_judgment(
        self,
        decision_messages: List,
        relation_result: Dict[str, Any],
        voice_conclusion,
    ) -> Optional[Dict[str, Any]]:
        """查询六维信号决策系统，返回回复判断与综合置信度

        利用 heartflow_decision 的 credibility / hazard / curiosity /
        immediacy / bond_strength / temperament 六维融合，得到更丰富的
        多角度判断，补充纯公式 readiness 评估的不足。

        返回 None 表示查询失败（不影响后续管线），否则：
        {
            "should_respond": bool,
            "composite": float,       # 0-1 综合评分
            "certainty": str,         # 置信度分级名
            "rationale": str,         # 决策理由摘要
        }
        """
        try:
            from src.chat.heart_flow.heartflow_decision import (
                acquire_decision_maker,
                JudgmentDimension,
                JudgmentSituation,
            )

            maker = acquire_decision_maker()
            latest_text = ""
            sender_id = ""
            latest_user_msg = self._get_latest_human_message(decision_messages)
            if latest_user_msg is not None:
                sender_id = str(getattr(latest_user_msg, "user_id", "") or "")
                latest_text = str(
                    getattr(latest_user_msg, "processed_plain_text", "")
                    or getattr(latest_user_msg, "plain_text", "")
                    or ""
                ).strip()[:500]
            mood_tag = "neutral"
            if voice_conclusion is not None:
                raw_mood = str(getattr(voice_conclusion, "current_mood", "") or "").strip()
                if raw_mood:
                    mood_tag = raw_mood
            _desire_raw = 5
            if voice_conclusion is not None and hasattr(voice_conclusion, "reply_desire_level"):
                _desire_raw = int(getattr(voice_conclusion, "reply_desire_level", 5) or 5)
            social_rel = {
                "is_targeted": bool(getattr(self, "_cached_targeted_to_bot", False)),
                "interest_level": max(0.0, min(1.0, _desire_raw / 10.0)),
            }
            situation = JudgmentSituation(
                stream_id=self.stream_id,
                sender_id=sender_id,
                raw_content=latest_text,
                bot_mood_tag=mood_tag,
                social_hints=social_rel,
                sender_profile={
                    "rel_score": float(relation_result.get("social_value", 0.0) or 0.0),
                    "activity": float(relation_result.get("activity_level", 50.0) or 50.0),
                },
            )
            outcome = maker.multidim_decide(JudgmentDimension.RESPOND, situation)
            comp = maker.weighted_composite(outcome.signal_snapshot)
            return {
                "should_respond": bool(outcome.verdict),
                "composite": round(comp, 4),
                "certainty": outcome.certainty.name,
                "rationale": str(outcome.rationale or "")[:120],
            }
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 心流决策查询异常: {exc}")
            return None

    def _check_early_exit(
        self,
        voice_conclusion,
        pinged_msg,
        relation_result: Dict,
        decision_messages: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        早期退出检查 - 在调用 LLM 之前判断是否应该跳过

        核心思路：如果内心独白已经明确决定不想回复，就不要浪费 token 调用 LLM。

        返回值：
        - should_skip: 是否应该跳过后续流程
        - reason: 跳过原因
        """
        result = {"should_skip": False, "reason": ""}

        # 针对 bot 的输入（直接@ / 回复bot / 引用bot）不在早期退出阶段被静默吞掉
        if pinged_msg is not None or self._has_targeted_bot_message(decision_messages or []):
            _model_reply = getattr(voice_conclusion, "should_reply", None) if voice_conclusion else None
            if _model_reply is False:
                result["reason"] = "目标指向bot，但模型决策为不回复，交由后续准入评估裁决"
            return result

        # 检查内心独白结果
        if voice_conclusion is not None and hasattr(voice_conclusion, "reply_desire_level"):
            desire_level = voice_conclusion.reply_desire_level

            # 欲望等级 <= 2：明确不想回复，直接跳过
            if desire_level <= 2:
                result["should_skip"] = True
                result["reason"] = f"内心独白欲望等级过低({desire_level})，明确不想回复"
                return result

            # 欲望等级 <= 4 且有负面情绪：跳过
            if desire_level <= 4:
                mood = str(getattr(voice_conclusion, "current_mood", "") or "").strip().lower()
                _negative_moods = {
                    "烦躁",
                    "疲惫",
                    "无聊",
                    "厌倦",
                    "愤怒",
                    "焦虑",
                    "低落",
                    "冷漠",
                    "抗拒",
                    "不耐烦",
                    "irritated",
                    "bored",
                    "tired",
                    "angry",
                    "anxious",
                }
                if any(kw in mood for kw in _negative_moods):
                    result["should_skip"] = True
                    result["reason"] = f"内心独白欲望等级低({desire_level})且情绪负面({mood})"
                    return result

        # 检查关系度：厌烦值过高时概率性跳过（但保留小概率回复表达不满）
        annoyance = relation_result.get("annoyance_value", 0.0)
        if annoyance > 50:
            # 厌烦值越高，跳过概率越大，但最高 90%（保留 10% 回复机会）
            skip_prob = min(0.90, 0.3 + (annoyance - 50) / 100.0)
            if random.random() < skip_prob:
                result["should_skip"] = True
                result["reason"] = f"厌烦值过高({annoyance:.1f})，不适合回复"
                return result
            else:
                # 小概率回复时，标记为"可能表达不满"
                result["emotion_hint"] = "annoyed_reply"
                logger.info(f"{self.log_prefix} 😤 厌烦值{annoyance:.1f}但决定回复，可能表达不满")

        # 检查能量：能量过低时跳过（优先从世界快照读取）
        snap = getattr(self, "_tick_world_snapshot", None)
        if snap is not None:
            chat_val = snap.self_resources.chat_energy
            think_val = snap.self_resources.thinking_energy
            act_val = snap.self_resources.activity_level
            if chat_val < 10 or think_val < 10 or act_val < 8:
                result["should_skip"] = True
                result["reason"] = f"快照状态过低(聊天值={chat_val:.1f}, 思考值={think_val:.1f}, 活跃度={act_val:.1f})"
                return result
        else:
            try:
                _d6f = EnergyChainDimension.get_instance()
                _d6sf = _d6f._ensure_channel(self.stream_id)
                _d6snapf = _d6f.capture_snapshot(self.stream_id)
                chat_value = _d6sf.chat_pool
                activity_level = _d6sf.activity_level
                if chat_value < 10 or _d6snapf.thinking_value < 10 or activity_level < 8:
                    result["should_skip"] = True
                    result["reason"] = f"状态过低(聊天值={chat_value:.1f}, 思考值={
                        _d6snapf.thinking_value:.1f}, 活跃度={activity_level:.1f})"
                    return result
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

        return result

    def _prepare_decision_messages(
        self,
        incoming_batch: List,
        filtered_messages: List,
        pinged_msg,
    ) -> List:
        """为当前轮决策选出真正值得处理的消息。"""
        if pinged_msg is not None:
            decision_messages = [self._tag_decision_message(msg, "incoming") for msg in list(incoming_batch or [])]
            self._last_decision_incoming_messages = list(decision_messages)
            self._last_decision_context_messages = []
            self._last_decision_self_messages = []
            return decision_messages
        decision_messages = []
        incoming_messages: List[Any] = []
        context_messages: List[Any] = []
        self_messages: List[Any] = []
        for msg in filtered_messages:
            self._tag_decision_message(msg, "incoming")
            if self._is_bot_message_obj(msg):
                decision_messages.append(msg)
                incoming_messages.append(msg)
                continue
            if getattr(msg, "_spam_user", False):
                continue
            decision_messages.append(msg)
            incoming_messages.append(msg)

        latest_text = ""
        for msg in reversed(decision_messages):
            if not self._is_bot_message_obj(msg):
                latest_text = str(
                    getattr(msg, "processed_plain_text", "")
                    or getattr(msg, "plain_text", "")
                    or getattr(msg, "content", "")
                    or ""
                ).strip()
                if latest_text:
                    break

        try:
            from types import SimpleNamespace
            from src.modules.context_manager import get_context_manager

            context_manager = get_context_manager()
            recent_context = context_manager.get_recent_context(self.stream_id, limit=8)
            relevant_context = (
                context_manager.search_context(self.stream_id, latest_text, limit=4) if latest_text else []
            )

            seen_context_ids = {
                str(getattr(msg, "message_id", "") or getattr(msg, "id", "") or "") for msg in decision_messages
            }
            for item in list(recent_context) + list(relevant_context):
                context_id = str(getattr(item, "message_id", "") or "")
                if context_id and context_id in seen_context_ids:
                    continue
                pseudo = SimpleNamespace(
                    message_id=context_id,
                    user_id=str(getattr(item, "user_id", "") or ""),
                    user_nickname=str(getattr(item, "user_name", "") or ""),
                    nickname=str(getattr(item, "user_name", "") or ""),
                    processed_plain_text=str(getattr(item, "content", "") or ""),
                    plain_text=str(getattr(item, "content", "") or ""),
                    content=str(getattr(item, "content", "") or ""),
                    timestamp=float(getattr(item, "timestamp", 0.0) or 0.0),
                    _decision_message_source="historical_context",
                    _is_context_backfill=True,
                )
                decision_messages.append(pseudo)
                context_messages.append(pseudo)
                if context_id:
                    seen_context_ids.add(context_id)
        except Exception as _e:
            logger.debug(f"异常: {_e}")

        seen_texts = {
            str(getattr(msg, "processed_plain_text", "") or getattr(msg, "plain_text", "") or "").strip()
            for msg in decision_messages
        }
        for synthetic_msg in self._build_synthetic_self_messages():
            synthetic_text = str(
                getattr(synthetic_msg, "processed_plain_text", "") or getattr(synthetic_msg, "plain_text", "") or ""
            ).strip()
            if not synthetic_text or synthetic_text in seen_texts:
                continue
            decision_messages.append(synthetic_msg)
            self_messages.append(synthetic_msg)

        decision_messages.sort(key=lambda msg: float(getattr(msg, "timestamp", 0.0) or 0.0))
        self._last_decision_incoming_messages = list(incoming_messages)
        self._last_decision_context_messages = list(context_messages)
        self._last_decision_self_messages = list(self_messages)
        return decision_messages

    @classmethod
    def _normalize_topic_text(cls, text: str) -> str:
        """标准化文本，用于粗粒度话题重复识别。"""
        normalized = (text or "").strip().lower()
        if not normalized:
            return ""
        normalized = re.sub(r"\s+", "", normalized)
        normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)
        return normalized

    @classmethod
    def _extract_topic_tokens(cls, text: str) -> List[str]:
        """提取可复用的文本片段，避免依赖手工停用词表。"""
        raw = (text or "").strip().lower()
        if not raw:
            return []

        segments = re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9_]{2,}", raw)
        tokens: List[str] = []
        seen = set()

        for segment in segments:
            cleaned = segment.strip("_")
            if len(cleaned) < 2:
                continue
            if cleaned in seen or cleaned.isdigit():
                continue
            seen.add(cleaned)
            tokens.append(cleaned[:12])
            if len(tokens) >= 8:
                break

        return tokens

    def _collect_topic_signals(self, messages: List, limit: int = 12) -> Dict[str, Any]:
        """收集最近消息里的重复话题信号。"""
        human_entries = []
        exact_counter: Counter = Counter()
        exact_users: Dict[str, set] = defaultdict(set)
        token_counter: Counter = Counter()
        token_users: Dict[str, set] = defaultdict(set)
        low_info_count = 0

        for msg in messages[-limit:]:
            user_id = getattr(msg, "user_id", "") or ""
            if self._is_bot_message_obj(msg):
                continue
            if self._get_decision_message_source(msg) != "incoming":
                continue

            text = (getattr(msg, "processed_plain_text", "") or getattr(msg, "content", "") or "").strip()
            if not text:
                continue

            normalized = self._normalize_topic_text(text)
            if not normalized:
                continue

            tokens = self._extract_topic_tokens(text)
            human_entries.append(
                {
                    "user_id": user_id,
                    "text": text,
                    "normalized": normalized,
                    "tokens": tokens,
                }
            )
            exact_counter[normalized] += 1
            if user_id:
                exact_users[normalized].add(user_id)
            if len(normalized) <= 6 or len(text) <= 8:
                low_info_count += 1
            for token in set(tokens):
                token_counter[token] += 1
                if user_id:
                    token_users[token].add(user_id)

        return {
            "human_entries": human_entries,
            "exact_counter": exact_counter,
            "exact_users": exact_users,
            "token_counter": token_counter,
            "token_users": token_users,
            "low_info_count": low_info_count,
        }

    def _analyze_repetition_pressure(self, messages: List) -> Dict[str, Any]:
        """检测重复刷屏：多人围绕同一内容反复说 / 同一人连续发相似消息。使用运行时可调阈值减少硬编码依赖。"""
        _cache_key = ("rep", id(messages))
        _cached = getattr(self, "_signal_cache", {}).get(_cache_key)
        if _cached is not None:
            return _cached
        result = {
            "detected": False,
            "reason": "",
            "exact_repeat_count": 0,
            "repeat_user_count": 0,
            "dominant_tokens": [],
            "low_info_cluster": False,
            "latest_matches_repeat": False,
        }

        stats = self._collect_topic_signals(messages)
        human_entries = stats["human_entries"]
        if len(human_entries) < 2:
            return result

        exact_threshold = _rt_float("repetition_exact_threshold", 2.0)
        low_info_ratio_threshold = _rt_float("repetition_low_info_ratio", 0.5)
        single_spam_threshold = _rt_float("repetition_single_spam_threshold", 2.0)
        exact_key = ""
        for normalized, count in stats["exact_counter"].most_common(3):
            user_count = len(stats["exact_users"].get(normalized, set()))
            if count >= exact_threshold and (user_count >= 2 or len(normalized) <= 8):
                exact_key = normalized
                result["exact_repeat_count"] = count
                result["repeat_user_count"] = max(result["repeat_user_count"], user_count)
                break

        dominant_tokens = []
        for token, count in stats["token_counter"].most_common(6):
            user_count = len(stats["token_users"].get(token, set()))
            if count >= 2 and user_count >= 1:
                dominant_tokens.append(token)
                result["repeat_user_count"] = max(result["repeat_user_count"], user_count)
        result["dominant_tokens"] = dominant_tokens[:3]

        low_info_ratio = stats["low_info_count"] / max(1, len(human_entries))
        result["low_info_cluster"] = low_info_ratio >= low_info_ratio_threshold and bool(exact_key or dominant_tokens)

        latest_entry = human_entries[-1]
        latest_tokens = set(latest_entry.get("tokens", []))
        latest_user = str(latest_entry.get("user_id", "") or "")
        result["latest_matches_repeat"] = bool(
            (exact_key and latest_entry.get("normalized") == exact_key)
            or (dominant_tokens and latest_tokens.intersection(dominant_tokens))
        )

        _single_user_spam = False
        if len(human_entries) >= 2:
            _recent_users = [str(e.get("user_id", "") or "") for e in human_entries[-3:]]
            if len(set(_recent_users)) == 1 and _recent_users[0] == latest_user:
                _single_user_spam = True
                if result["exact_repeat_count"] >= single_spam_threshold:
                    result["detected"] = True

        if not result["detected"]:
            result["detected"] = bool(
                result["latest_matches_repeat"]
                and (
                    result["exact_repeat_count"] >= exact_threshold
                    or (result["repeat_user_count"] >= 2 and dominant_tokens)
                    or result["low_info_cluster"]
                    or _single_user_spam
                )
            )

        if not result["detected"]:
            return result

        if _single_user_spam and result["exact_repeat_count"] >= single_spam_threshold:
            result["reason"] = f"同一人连续复读({result['exact_repeat_count']}次)"
        elif result["exact_repeat_count"] >= exact_threshold + 1 and result["repeat_user_count"] >= 2:
            result["reason"] = f"多人连续复读相近内容({result['exact_repeat_count']}次)"
        elif dominant_tokens and result["repeat_user_count"] >= 3:
            result["reason"] = f"多人围绕 {dominant_tokens[0]} 反复刷低信息消息"
        elif dominant_tokens:
            result["reason"] = f"当前话题 {dominant_tokens[0]} 短时间内重复度过高"
        else:
            result["reason"] = "当前内容重复度过高，先不接话"
        if not hasattr(self, "_signal_cache"):
            self._signal_cache = {}
        self._signal_cache[_cache_key] = result
        if len(self._signal_cache) > 20:
            oldest = list(self._signal_cache.keys())[:10]
            for k in oldest:
                self._signal_cache.pop(k, None)
        return result

    def _analyze_harassment_pressure(self, messages: List) -> Dict[str, Any]:
        """补充旧版的骚扰强度层，用于区分普通复读和带冒犯/骚扰意味的持续输入。"""
        _cache_key = ("har", id(messages))
        _cached = getattr(self, "_signal_cache", {}).get(_cache_key)
        if _cached is not None:
            return _cached
        result = {
            "detected": False,
            "reason": "",
            "intensity": 0.0,
            "same_hash_count": 0,
            "similar_count": 0,
            "severity": "low",
        }
        if not messages:
            return result
        try:
            human_messages = self._get_human_message_candidates(list(messages[-12:]))
            latest_user = human_messages[-1] if human_messages else None
            if latest_user is None:
                return result

            user_id = str(getattr(latest_user, "user_id", "") or "")
            if not user_id:
                return result
            text = self._extract_message_content(latest_user).strip()
            if not text:
                return result

            harassment_words = {
                "性骚扰": 6.0,
                "骚扰": 5.0,
                "羞辱": 4.0,
                "亵渎": 4.0,
                "冒犯": 3.0,
                "低俗": 3.0,
                "支配": 3.0,
                "滚": 2.5,
                "闭嘴": 2.5,
                "恶心": 2.5,
            }

            def _calc_intensity(raw_text: str) -> float:
                value = 0.0
                lowered = raw_text.lower()
                for word, score in harassment_words.items():
                    if word in lowered:
                        value += score
                if len(raw_text) >= 4:
                    tokens = raw_text.split()
                    if tokens and len(set(tokens)) * 2 < len(tokens):
                        value += 1.5
                return min(10.0, value)

            same_hash_count = 1
            similar_count = 0
            total_intensity = _calc_intensity(text)
            normalized_text = self._normalize_topic_text(text)

            for msg in human_messages[:-1]:
                if getattr(msg, "user_id", "") != user_id:
                    continue
                history_text = self._extract_message_content(msg).strip()
                if not history_text:
                    continue
                total_intensity += _calc_intensity(history_text)
                normalized_history = self._normalize_topic_text(history_text)
                if normalized_history == normalized_text:
                    same_hash_count += 1
                elif (
                    normalized_history
                    and normalized_text
                    and (normalized_history in normalized_text or normalized_text in normalized_history)
                ):
                    similar_count += 1

            detected = bool(same_hash_count >= 5 or similar_count >= 4 or total_intensity >= 20.0)
            detail = {
                "total_intensity": total_intensity,
                "same_hash_count": same_hash_count,
                "similar_count": similar_count,
            }
            if not detected:
                return result

            intensity = float(detail.get("total_intensity", 0.0) or 0.0)
            same_hash_count = int(detail.get("same_hash_count", 0) or 0)
            similar_count = int(detail.get("similar_count", 0) or 0)
            severity = "high" if intensity >= 35 or same_hash_count >= 6 else "medium" if intensity >= 20 else "low"
            result.update(
                {
                    "detected": True,
                    "reason": f"疑似持续骚扰/冒犯输入，强度={intensity:.1f}",
                    "intensity": intensity,
                    "same_hash_count": same_hash_count,
                    "similar_count": similar_count,
                    "severity": severity,
                }
            )
            return result
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 骚扰强度分析失败: {exc}")
            return result
        finally:
            if not hasattr(self, "_signal_cache"):
                self._signal_cache = {}
            self._signal_cache[_cache_key] = result

    def _decide_reply_style(
        self,
        target_message,
        relation_snapshot: Optional[Dict[str, Any]] = None,
        harassment_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """把旧版 direct/quote 决策思想映射到当前 quote_message 开关。"""
        snapshot = self._resolve_relation_view(relation_snapshot)
        trust_value = float(snapshot.get("trust_value", 0.0) or 0.0)
        annoyance_value = float(snapshot.get("annoyance_value", 0.0) or 0.0)
        text = self._extract_message_content(target_message).strip() if target_message is not None else ""

        quote_message = False
        reply_style = "direct"
        reasons: List[str] = []

        if target_message is None:
            reasons.append("无明确用户目标，主动发言不走引用")
            return {
                "reply_style": reply_style,
                "quote_message": quote_message,
                "reason": "；".join(reasons),
            }

        # 不引用bot自己的消息，避免引用链无限堆叠
        _target_uid = str(getattr(target_message, "user_id", "") or "") if target_message else ""
        _is_bot_msg = self._is_bot_message_obj(target_message) or _target_uid == str(
            getattr(self, "bot_user_id", "") or ""
        )
        if _is_bot_msg:
            reasons.append("不引用自己的消息")
        elif target_message is not None and bool(getattr(target_message, "is_quote_reply", False)):
            quote_message = True
            reply_style = "quote"
            reasons.append("对方本身就是引用链")
        elif any(marker in text for marker in ("?", "？", "怎么", "为什么", "啥", "什么")):
            quote_message = True
            reply_style = "quote"
            reasons.append("消息更像明确提问")
        elif harassment_signal and harassment_signal.get("detected"):
            quote_message = False
            reply_style = "direct"
            reasons.append("骚扰/冒犯场景不跟着引用抬杠")
        elif annoyance_value >= 25 and trust_value < 15:
            quote_message = False
            reply_style = "direct"
            reasons.append("关系紧张时避免贴脸逐句对线")
        elif len(text) <= 10:
            quote_message = False
            reply_style = "direct"
            reasons.append("短消息直接接话更自然")
        else:
            reasons.append("默认自然续聊")

        return {
            "reply_style": reply_style,
            "quote_message": quote_message,
            "reason": "；".join(reasons),
        }

    def _extract_current_topics(self, messages: List, limit: int = 5) -> List[str]:
        """提取当前话题，供 LLM 自主规划器使用。"""
        topics: List[str] = []
        seen = set()

        group_context_signal = getattr(self, "_last_group_context_signal", None) or {}
        topic_hints = []
        if isinstance(group_context_signal, dict):
            topic_hints = list(group_context_signal.get("topic_hints", []) or [])
        if not topic_hints:
            group_sense = getattr(self, "_last_group_sense", None)
            topic_hints = getattr(group_sense, "topic_hints", []) or []
        for hint in topic_hints:
            normalized = self._normalize_topic_text(str(hint))
            if len(normalized) < 2 or normalized in seen:
                continue
            seen.add(normalized)
            topics.append(str(hint)[:12])
            if len(topics) >= limit:
                return topics

        stats = self._collect_topic_signals(messages, limit=15)
        for token, count in stats["token_counter"].most_common(limit * 3):
            if count < 2 and topics:
                continue
            if token in seen:
                continue
            seen.add(token)
            topics.append(token)
            if len(topics) >= limit:
                return topics

        if topics:
            return topics

        latest_entries = stats["human_entries"]
        if latest_entries:
            for token in latest_entries[-1].get("tokens", []):
                if token in seen:
                    continue
                seen.add(token)
                topics.append(token)
                if len(topics) >= min(limit, 3):
                    break
        return topics

    def _evaluate_message_salience(
        self,
        messages: List,
        pinged_msg,
        repetition_signal: Optional[Dict[str, Any]] = None,
        behavior_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """评估当前消息是否足够值得立即响应。"""
        if pinged_msg is not None:
            return {"score": 4, "reason": "explicit_ping"}
        latest_user_message = None
        for msg in reversed(messages):
            if not self._is_bot_message_obj(msg):
                latest_user_message = msg
                break
        if latest_user_message is None:
            return {"score": 0, "reason": "no_user_message"}

        text = (
            getattr(latest_user_message, "processed_plain_text", "")
            or getattr(latest_user_message, "content", "")
            or ""
        ).strip()
        score = 0
        if len(text) >= 10:
            score += 1
        punctuation_hits = sum(text.count(marker) for marker in ("?", "？", "!", "！"))
        if punctuation_hits > 0:
            score += min(2, punctuation_hits)
        normalized = self._normalize_topic_text(text)
        if len(normalized) >= 18:
            score += 1
        if any(not self._is_bot_message_obj(msg) for msg in messages[-3:]):
            score += 1
        if getattr(latest_user_message, "_spam_user", False):
            score = max(0, score - 2)
        if repetition_signal and repetition_signal.get("latest_matches_repeat"):
            if repetition_signal.get("low_info_cluster"):
                score = max(score, 2)
            elif repetition_signal.get("detected"):
                score = max(score, 1)
        if behavior_signal:
            category = str(behavior_signal.get("category", "neutral") or "neutral")
            severity = float(behavior_signal.get("severity", 0.0) or 0.0)
            if category == "friendly":
                score += 1
            elif category == "hostile":
                score = max(score, 2)
            elif category == "harassing":
                score = max(0, score - 1)
            if severity >= 0.8 and category in {"hostile", "harassing"}:
                score = max(score, 2)
        return {
            "score": score,
            "reason": text[:40],
            "repeat_pressure": (repetition_signal or {}).get("reason", ""),
        }

    def _evaluate_autonomy_guard(
        self,
        decision_messages: List,
        pinged_msg,
        identity_context: Dict[str, Any],
        self_reply_risk: Dict[str, Any],
        group_sense_result: Dict[str, Any],
        relation_result: Dict[str, Any],
        voice_conclusion,
        message_salience: Dict[str, Any],
        group_context_signal: Optional[Dict[str, Any]] = None,
        content_state_signal: Optional[Dict[str, Any]] = None,
        preprocessor_signal: Optional[Dict[str, Any]] = None,
        repetition_signal: Optional[Dict[str, Any]] = None,
        harassment_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """把身份、群态势和内容状态真正接入拒绝回复链。"""
        result = {"should_skip": False, "reason": "", "source": ""}
        merged_group_signal = (
            group_context_signal
            if isinstance(group_context_signal, dict)
            else (group_sense_result if isinstance(group_sense_result, dict) else {})
        )
        if pinged_msg is not None or self._has_targeted_bot_message(decision_messages):
            return result

        human_messages = self._get_human_message_candidates(decision_messages)
        if not human_messages:
            result["should_skip"] = True
            result["reason"] = "过滤后没有有效目标，保持观察"
            return result

        if identity_context.get("has_conflict"):
            result["should_skip"] = True
            result["reason"] = "身份状态冲突，暂不介入"
            return result

        if self_reply_risk.get("is_self_reply") and float(self_reply_risk.get("similarity", 0.0) or 0.0) >= 0.88:
            result["should_skip"] = True
            result["reason"] = f"自回复风险过高({self_reply_risk.get('similarity', 0.0):.2f})"
            return result

        desire_level = 5
        if voice_conclusion is not None and hasattr(voice_conclusion, "reply_desire_level"):
            desire_level = int(getattr(voice_conclusion, "reply_desire_level", 5) or 5)

        if content_state_signal and content_state_signal.get("should_skip") and desire_level < 8:
            result["should_skip"] = True
            result["reason"] = content_state_signal.get("reason", "当前内容已被处理过，先不重复接话")
            result["source"] = "content_state"
            return result

        if preprocessor_signal and preprocessor_signal.get("is_spam") and desire_level < 8:
            spam_type = preprocessor_signal.get("spam_type", "刷屏")
            result["should_skip"] = True
            result["reason"] = f"检测到{spam_type}，当前不跟着噪声接话"
            result["source"] = "message_preprocessor"
            return result

        if harassment_signal and harassment_signal.get("detected") and desire_level < 9:
            severity = harassment_signal.get("severity", "low")
            result["should_skip"] = True
            result["reason"] = harassment_signal.get("reason", f"检测到{severity}级骚扰输入，当前不接话")
            result["source"] = "harassment_detector"
            return result

        if repetition_signal and repetition_signal.get("detected"):
            exact_repeat_count = int(repetition_signal.get("exact_repeat_count", 0) or 0)
            low_info_cluster = bool(repetition_signal.get("low_info_cluster", False))
            repeat_reason = str(repetition_signal.get("reason", "重复输入") or "重复输入")
            if exact_repeat_count >= 4 and low_info_cluster and desire_level < 9:
                result["should_skip"] = True
                result["reason"] = f"{repeat_reason}，进入明确冷处理"
                result["source"] = "repeat_guard"
                return result

        if relation_result.get("behavior_signal"):
            behavior_signal = relation_result.get("behavior_signal") or {}
            category = str(behavior_signal.get("category", "neutral") or "neutral")
            severity = float(behavior_signal.get("severity", 0.0) or 0.0)
            if category == "harassing" and desire_level < 9:
                result["should_skip"] = True
                result["reason"] = f"分类为持续骚扰输入(severity={severity:.2f})，本轮不回"
                result["source"] = "behavior_classifier"
                return result
            if category == "hostile" and severity >= 0.85 and desire_level < 8:
                result["should_skip"] = True
                result["reason"] = "分类为高强度敌意表达，先不正面接火"
                result["source"] = "behavior_classifier"
                return result

        if int(message_salience.get("score", 0) or 0) < 2 and desire_level < 7:
            result["should_skip"] = True
            result["reason"] = "消息显著性不足，暂不主动接话"
            return result

        if (
            repetition_signal
            and repetition_signal.get("detected")
            and repetition_signal.get("latest_matches_repeat")
            and desire_level < 8
        ):
            logger.debug(
                f"{self.log_prefix} 检测到重复输入，允许规划器给出澄清反馈: {repetition_signal.get('reason', 'repeat')}"
            )

        if merged_group_signal.get("burst_detected") and desire_level < 7:
            result["should_skip"] = True
            result["reason"] = "群里正在高密度交流，当前不插话"
            result["source"] = "group_guard"
            return result

        if (
            merged_group_signal.get("controversy_detected")
            and float(relation_result.get("trust_value", 0.0) or 0.0) < 35.0
            and desire_level < 8
        ):
            result["should_skip"] = True
            result["reason"] = "群聊存在争议且信任不足，选择克制"
            result["source"] = "group_guard"
            return result

        annoyance = float(relation_result.get("annoyance_value", 0.0) or 0.0)
        try:
            from src.chat.prompts.soul_config_loader import get_thresholds as _get_thr3

            _cold_thr = float(_get_thr3().get("cold_rejection", 35))
        except Exception:
            _cold_thr = 35.0
        if annoyance >= _cold_thr and desire_level < 7:
            # 厌烦值高时概率性跳过，但保留回复机会表达不满
            skip_prob = min(0.90, (annoyance - _cold_thr) / 100.0)
            if random.random() < skip_prob:
                result["should_skip"] = True
                result["reason"] = f"厌烦值偏高({annoyance:.1f})，本轮不回"
                return result
            else:
                # 小概率回复时，标记情绪提示
                result["emotion_hint"] = "annoyed_reply"
                logger.info(f"{self.log_prefix} 😤 厌烦值{annoyance:.1f}但决定回复，可能表达不满")

        return result

    async def _execute_voice_driven_reply(
        self,
        voice_conclusion: Any,
        incoming_batch: List,
        force_reply_message: Optional[Any] = None,
    ) -> bool:
        """
        执行内心驱动回复 - 内心独白决定回复时直接生成回复，跳过规划器

        这是真正的主动行为：内心独白已经决定了要回复，直接生成回复。
        """
        target_message = None
        try:
            from src.llm_models.utils_model import bind_stream_context

            # 绑定聊天流ID到当前异步任务，使并发守卫能按流限速
            bind_stream_context(self.stream_id)

            desire_level = getattr(voice_conclusion, "reply_desire_level", 5)
            thinking = getattr(voice_conclusion, "thinking", "")

            logger.info(
                f"{self.log_prefix} 💭 开始执行: 欲望等级={desire_level}, 思考={thinking[:50] if thinking else '无'}"
            )

            target_message = force_reply_message
            if target_message is None:
                for msg in reversed(incoming_batch[-10:]):
                    if not self._is_bot_message_obj(msg):
                        target_message = msg
                        break
            target_message = self._select_preferred_reply_message(target_message, list(incoming_batch[-10:]))
            if target_message is None:
                self._last_flow_blocker = "voice缺少可回复目标"
                logger.info(f"{self.log_prefix} 💭 内心驱动回复缺少目标消息，转为观察")
                return False

            legacy_gate = str(getattr(self, "_last_legacy_gate", "allow") or "allow")
            force_bypass = bool(force_reply_message is not None or legacy_gate == "force_reply")
            targeted_to_bot = bool(force_bypass or getattr(self, "_cached_targeted_to_bot", False))
            restraint = await self._run_self_restraint_check(
                incoming_batch,
                source="voice_driven_reply",
                repetition_signal=self._analyze_repetition_pressure(incoming_batch),
                harassment_signal=self._analyze_harassment_pressure(incoming_batch),
                force_bypass=force_bypass,
                targeted_to_bot=targeted_to_bot,
                admin_force=bool(getattr(self, "_is_admin_forced", False)),
            )
            if not restraint.get("allow", True):
                self._last_flow_blocker = f"voice自省拦截:{restraint.get('reason', 'skip')}"
                logger.info(f"{self.log_prefix} 🧯 自省闸门拦截 voice 回复: {restraint.get('reason', 'skip')}")
                return False
            self._mark_message_content_processing(target_message)

            # 获取目标用户的风格指导
            user_style_guide = ""
            if target_message:
                target_user_id = getattr(target_message, "user_id", "")
                if target_user_id:
                    user_style_guide = self._get_user_style_guide(target_user_id)

            voice_summary = self._build_voice_execution_summary(voice_conclusion, target_message)

            from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

            reply_reason = acquire_reply_coordinator().compose_reply_reason(
                base_reason=f"内心驱动回复: 欲望等级={desire_level}",
                voice_reason=voice_summary.get("reason", ""),
            )

            extra_info_parts: List[str] = []
            panel_text = voice_summary.get("panel", "")
            if panel_text:
                extra_info_parts.append(panel_text)
            decision_context_packet = self._build_decision_context_packet(
                list(incoming_batch),
                repetition_signal=self._analyze_repetition_pressure(incoming_batch),
            )
            relation_view = self._resolve_relation_view()
            context_execution_block = self._build_context_execution_block(
                target_message=target_message,
                voice_conclusion=voice_conclusion,
                repetition_signal=self._analyze_repetition_pressure(incoming_batch),
                decision_context_packet=decision_context_packet,
                relation_snapshot=relation_view,
            )
            if context_execution_block:
                extra_info_parts.append(context_execution_block)
            from src.chat.replyer.context_block_builder import build_shared_reply_parts

            extra_info_parts.extend(
                build_shared_reply_parts(
                    self_memory=self._build_self_reply_memory(),
                    continuity_context=self._build_self_continuity_context(),
                    user_style_guide=user_style_guide,
                    persona_hint="",
                    reply_style_context=self._build_reply_style_context(relation_view),
                    length_hint=self._build_dynamic_length_hint(target_message, user_style_guide),
                    restraint_mode=str(restraint.get("mode", "allow") or "allow"),
                    short_only_text="[自省闸门约束] 你已经连续说了不少，这次只准一句短话，不展开，不补充，不连发。",
                )
            )

            takeover_thought = getattr(self, "_takeover_decision", None)
            if takeover_thought:
                _tk_thought = takeover_thought.get("thought", "") or ""
                _tk_action = takeover_thought.get("action", "") or ""
                if _tk_thought:
                    extra_info_parts.append(f"[接管意图] {_tk_action}: {_tk_thought}")
                self._takeover_decision = None
                self._takeover_action = None

            harassment_signal = self._analyze_harassment_pressure(incoming_batch)
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

            append_reply_style(extra_info_parts, style_route)
            # 注入多维状态系统的LLM提示词
            self._inject_dimension_state_prompt(extra_info_parts)
            _diversity_warn = self._check_reply_diversity()
            if _diversity_warn:
                extra_info_parts.append(_diversity_warn)
            _meme_quick = self._build_meme_injection()
            if _meme_quick:
                extra_info_parts.append(_meme_quick)
            extra_info = build_reply_context_block(
                recent_context="",
                relevant_context="",
                extra_info="\n".join(part for part in extra_info_parts if part),
                recent_reply_guard="",
            )
            extra_info = self._ensure_soul_data_in_extra_info(extra_info)
            _key_lines = [
                line
                for line in (extra_info or "").split("\n")
                if self._contains_soul_data(line)
            ]
            if _key_lines:
                logger.info(f"{self.log_prefix} 🧠 传入LLM的灵魂数据:\n" + "\n".join(_key_lines[:8]))
            else:
                logger.warning(
                    f"{self.log_prefix} ⚠️ 传入LLM的extra_info中没有灵魂数据！extra_info长度={len(extra_info or '')}"
                )
            self._emit_reply_generation_summary(
                target_message=target_message,
                style_route=style_route,
                relation_snapshot=relation_view,
                context_execution_block=context_execution_block,
                extra_info=extra_info,
                source="voice_driven",
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
                request_type="voice_driven_reply",
                think_level=1,
            )

            if not success or not llm_response or not llm_response.reply_set:
                self._last_flow_blocker = "voice回复生成失败"
                self._mark_message_content_deferred(target_message, "voice_generation_failed")
                logger.warning(f"{self.log_prefix} 💭 回复生成失败")
                return False

            response_set = llm_response.reply_set
            selected_expressions = llm_response.selected_expressions

            cycle_timers = {}
            thinking_id = f"voice_driven_{int(time.time() * 1000)}"

            loop_info, reply_text, _, reply_trace_meta = await self._send_and_store_reply(
                response_set=response_set,
                action_message=target_message,
                cycle_timers=cycle_timers,
                thinking_id=thinking_id,
                actions=[],
                selected_expressions=selected_expressions,
                quote_message=bool(style_route.get("quote_message", False)),
                pre_send_risk_note=(
                    f"voice annoyance={relation_view.get('annoyance_value', 0)} "
                    f"pressure={relation_view.get('psychological_pressure', 0)}"
                ),
                pre_send_audit_label="voice",
            )
            if not (bool(loop_info) or bool(str(reply_text or "").strip())):
                self._last_flow_blocker = "voice回复发送失败"
                self._mark_message_content_deferred(target_message, "voice_send_failed")
                return False

            await self._finalize_sent_reply(
                reply_text=reply_text,
                loop_info=loop_info,
                target_message=target_message,
                reply_reason=reply_reason,
                relation_view=relation_view,
                llm_response=llm_response,
                was_proactive=False,
                action_name="voice_driven_reply",
                quality=0.85,
                audit_label="voice",
                reply_trace_meta=reply_trace_meta,
            )
            logger.info(f"{self.log_prefix} 💭 成功发送: {reply_text[:50]}...")
            self._spawn(self._update_user_impression_after_reply(reply_text))
            return True

        except Exception as exc:
            self._last_flow_blocker = f"voice异常:{type(exc).__name__}"
            self._mark_message_content_deferred(
                target_message or self._get_latest_human_message(incoming_batch),
                "voice_reply_exception",
            )
            logger.error(f"{self.log_prefix} 💭 执行失败: {exc}")
            return False

    async def _proactive_background_loop(self):
        """主动感兴趣通道 —— 独立后台常驻任务，与被动监听通道并行运行

        设计原则：
          - 不依赖消息到达触发，而是持续自主评估"我是否想说点什么"
          - 有自己的独立感知→独白→决策→回复管线
          - 与被动通道共享同一套引擎（仪表盘/代谢/存在感等），但独立决策
          - 被动通道处理消息时不会被阻塞（asyncio协作式多任务）

        运行节奏：
          - 基础评估间隔：8-15秒（根据精力动态调整）
          - 深度评估（含LLM调用）：仅在评估通过后触发
          - 冷却期：主动发言后30-60秒内不再主动触发
        """
        _base_interval = 18.0
        _min_interval = 10.0
        _max_interval = 45.0
        _cooldown_until = 0.0
        _consecutive_fails = 0
        _loop_count = 0
        logger.info(f"{self.log_prefix} 🔔 主动通道后台循环启动，基础间隔={_base_interval}s")
        try:
            while self._proactive_running and self.running:
                _loop_count += 1
                _now_loop = time.time()
                if _now_loop < _cooldown_until:
                    await asyncio.sleep(min(2.0, _cooldown_until - _now_loop))
                    continue
                _energy_mod = 1.0
                _ms = getattr(self, "_cached_metabolism_state", None)
                if _ms:
                    _er = float(getattr(_ms, "energy_ratio", getattr(_ms, "chat_energy_ratio", 1.0)) or 1.0)
                    if _er < 0.3:
                        _energy_mod = 2.5
                        _base_interval = min(_max_interval, _base_interval + 1.0)
                    elif _er > 0.8:
                        _energy_mod = 0.7
                        _base_interval = max(_min_interval, _base_interval - 0.5)
                else:
                    _base_interval = 18.0
                _interval = max(
                    _min_interval,
                    min(_max_interval, _base_interval * _energy_mod),
                )
                try:
                    _dv = self._get_dashboard_verdict()
                    _dv_urgency = str(_dv.get("urgency", "") or "")
                    if not _dv.get("process", True):
                        await asyncio.sleep(_interval)
                        continue
                    if _dv_urgency in ("不回复", "强制休息"):
                        await asyncio.sleep(_interval * 1.5)
                        continue
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
                _silence = 0.0
                try:
                    from src.chat.proactive.silence_watcher import (
                        get_quiet_monitor,
                    )

                    _silence = get_quiet_monitor().measure_silence_sec(self.stream_id)
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
                if _silence < 3.0 and _loop_count > 3:
                    await asyncio.sleep(_interval * 0.5)
                    continue
                if (time.time() - self._last_bot_reply_ts) < 5.0:
                    await asyncio.sleep(_interval * 0.3)
                    continue
                _should_act = False
                _act_reason = ""
                try:
                    _should_act, _act_reason = await self._evaluate_proactive_opportunity(
                        now=_now_loop,
                        silence_sec=_silence,
                        is_background=True,
                    )
                except Exception as exc:
                    logger.debug(f"{self.log_prefix} 主动通道评估异常: {exc}")
                if not _should_act:
                    _consecutive_fails = 0
                    await asyncio.sleep(_interval)
                    continue
                logger.info(f"{self.log_prefix} 🔔 主动通道触发: {_act_reason} (静默={_silence:.0f}s)")
                try:
                    _acted = await self._execute_proactive_action(
                        reason=_act_reason,
                        silence_sec=_silence,
                    )
                    if _acted:
                        # 主动发言成功后冷却2-4分钟，避免连续自言自语
                        _cooldown_until = time.time() + random.uniform(240.0, 480.0)
                        _consecutive_fails = 0
                        _base_interval = max(_min_interval, _base_interval - 1.0)
                    else:
                        _consecutive_fails += 1
                        _base_interval = min(_max_interval, _base_interval + 0.5)
                except Exception as exc:
                    logger.warning(f"{self.log_prefix} 主动通道执行失败: {exc}")
                    _consecutive_fails += 1
                if _consecutive_fails >= 3:
                    _backoff = min(30.0, 5.0 * _consecutive_fails)
                    logger.debug(f"{self.log_prefix} 主动通道连续{_consecutive_fails}次失败，退避{_backoff:.0f}s")
                    await asyncio.sleep(_backoff)
                    _consecutive_fails = 0
                else:
                    await asyncio.sleep(_interval)
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} 🔔 主动通道后台循环被取消")
        except Exception as exc:
            logger.error(f"{self.log_prefix} 🔔 主动通道后台循环异常退出: {exc}")

    async def _evaluate_proactive_opportunity(
        self, *, now: float, silence_sec: float, is_background: bool = False
    ) -> Tuple[bool, str]:
        """评估是否应该主动发言（轻量级预检，不调LLM）

        Args:
            now: 当前时间戳
            silence_sec: 静默时长
            is_background: 是否来自后台主动通道（后台通道有更宽松的触发条件）

        Returns:
            (should_act, reason) 是否应该主动 + 原因描述
        """
        if not hasattr(self, "_last_idle_proactive_ts"):
            self._last_idle_proactive_ts = 0.0
        if is_background:
            startup_guard_until = float(getattr(self, "_proactive_startup_grace_until", 0.0) or 0.0)
            if now < startup_guard_until:
                return False, "启动保护期"
            if silence_sec < 120.0:
                return False, f"静默不足({silence_sec:.0f}s/120s)"
        # 后台通道冷却提升到90s，避免频繁触发主动发言
        _cooldown = self._IDLE_PROACTIVE_COOLDOWN_SEC if not is_background else 180.0
        if (now - self._last_idle_proactive_ts) < _cooldown:
            return False, "冷却中"
        if is_background:
            _bg_block_reason = self._background_proactive_block_reason()
            if _bg_block_reason:
                return False, _bg_block_reason
        if self._cached_night_phase is not None:
            _np = self._cached_night_phase
            _np_name = getattr(_np, "name", str(_np)) if _np else ""
            if "DEEP" in _np_name or "VALLEY" in _np_name or "BURNED" in _np_name:
                return False, "夜间冻结"
            if "DROWSY" in _np_name or "LIGHT" in _np_name:
                _peek_allowed = False
                try:
                    from src.core.night_cycle_system import get_night_cycle

                    _ncs_peek2 = get_night_cycle(self.stream_id)
                    _peek_allowed = _ncs_peek2.evaluate_sleep_peek()
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
                if not _peek_allowed:
                    try:
                        _d6 = EnergyChainDimension.get_instance()
                        _nm = _d6._get_night_mode(self.stream_id)
                        if _nm and _nm.prob_multiplier < 0.3:
                            return False, "浅睡无窥屏窗"
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
        if self._should_back_off_idle_proactive():
            return False, f"连续{self._unanswered_bot_turns}次未获回应"
        try:
            from src.modules.trauma.trauma_system import get_trauma_system

            _ts = get_trauma_system().get_state()
            if _ts.stress_accumulation > 8.0 or _ts.inner_chaos_level > 8.0:
                return False, f"压力过高(stress={_ts.stress_accumulation:.1f})"
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        governor_verdict = self._evaluate_behavior_governor(
            incoming_batch=[],
            silence_sec=silence_sec,
            requested_mode="proactive",
            is_background=is_background,
        )
        if not governor_verdict.allow_generation or governor_verdict.reply_mode != "proactive":
            return False, self._summarize_behavior_governor(governor_verdict)
        # 优先尝试 IntegrationHub (async，在此处 await)
        _hub_succeeded = False
        try:
            from src.chat.proactive.proactive_integration_hub import (
                get_proactive_integration_hub,
            )

            _hub = get_proactive_integration_hub()
            _ss = getattr(self, "_cached_self_state", None)
            _hub_self_state = {}
            if _ss:
                _hub_self_state = {
                    "mood": getattr(_ss, "mood", 0.5),
                    "trauma": getattr(_ss, "trauma_score", 0),
                    "fatigue": getattr(_ss, "mental_fatigue", 0),
                }
            _hub_relation = getattr(self, "_last_relation_snapshot", {}) or {}
            _hub_msgs = getattr(self, "_last_raw_messages", []) or []
            _hub_verdict, _ = await _hub.evaluate_proactive(
                channel_id=self.stream_id,
                raw_messages=_hub_msgs,
                self_state=_hub_self_state,
                relation_state=_hub_relation,
            )
            if _hub_verdict:
                _hub_succeeded = True
                # R2正向激励调整：Hub路径也应用情感驱动降低activation_bar
                self._apply_r2_incentives(_hub_verdict, silence_sec)
                _bd = getattr(_hub_verdict, "breakdown", {}) or {}
                _bd_str = " ".join(f"{k}={v:.2f}" for k, v in _bd.items() if isinstance(v, (int, float)))
                if _hub_verdict.should_proceed:
                    logger.info(
                        f"{self.log_prefix} 整合中心决策通过: "
                        f"融合分={_hub_verdict.fused_score:.3f} bar={_hub_verdict.activation_bar:.2f} | "
                        f"{_bd_str} | {_hub_verdict.rationale}"
                    )
                    return True, _hub_verdict.rationale or "整合中心通过"
                logger.debug(
                    f"{self.log_prefix} 整合中心否决: "
                    f"融合分={_hub_verdict.fused_score:.3f} bar={_hub_verdict.activation_bar:.2f} | {_bd_str}"
                )
                return False, _hub_verdict.rationale or "整合中心否决"
        except Exception as _hub_err:
            logger.debug(f"{self.log_prefix} 整合中心评估回退: {_hub_err}")
        # 回退到同步主动决策采集评估
        if not _hub_succeeded:
            proactive_decision = self._evaluate_proactive_decision(silence_sec)
            _abd = getattr(proactive_decision, "breakdown", {}) or {}
            _abd_str = " ".join(f"{k}={v:.2f}" for k, v in _abd.items() if isinstance(v, (int, float)))
            if not proactive_decision.should_proceed:
                logger.debug(
                    f"{self.log_prefix} 主动决策否决: "
                    f"融合分={proactive_decision.fused_score:.3f} bar={proactive_decision.activation_bar:.2f} | {_abd_str}"
                )
                return False, proactive_decision.rationale or "决策器否决"
            logger.info(
                f"{self.log_prefix} 主动决策通过: "
                f"融合分={proactive_decision.fused_score:.3f} bar={proactive_decision.activation_bar:.2f} | {_abd_str}"
            )
            return True, proactive_decision.rationale or "决策器通过"
        return False, "评估未完成"

    async def _execute_proactive_action(self, *, reason: str, silence_sec: float) -> bool:
        """执行主动回复动作（深度执行，可能含LLM调用）

        Returns:
            True 表示成功发送了主动回复
        """
        now_act = time.time()
        await self._update_trauma_system_state([])
        emotion_snapshot = self._update_emotion_state(now_act, relation_result={})
        if not emotion_snapshot:
            return False
        awareness_snapshot = None
        ambient_info = self._sample_channel_ambient()
        voice_conclusion = None
        _run_p = self._should_run_perception(now_act)
        _run_v = self._should_run_voice(
            now_act,
            [],
            None,
            is_proactive=True,
            source="background_proactive",
            behavior_verdict=getattr(self, "_last_behavior_governor_verdict", None),
        )
        _jobs: list = []
        _keys: list = []
        if _run_p:
            _jobs.append(self._invoke_perception([], now_act))
            _keys.append("p")
        if _run_v:
            _jobs.append(self._invoke_autonomous_voice("主动规划链路触发"))
            _keys.append("v")
        if _jobs:
            try:
                _stage_timeout = _parallel_stage_timeout()
                _res = await asyncio.wait_for(
                    asyncio.gather(*_jobs, return_exceptions=True),
                    timeout=_stage_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"{self.log_prefix} 💤 主动规划链路感知/独白超时({_stage_timeout:.0f}s)，跳过本轮主动发言"
                )
                self._last_idle_proactive_ts = now_act
                return False
            _map = dict(zip(_keys, _res, strict=True))
            if _run_p:
                _ip = _map.get("p")
                if isinstance(_ip, BaseException):
                    logger.warning(f"{self.log_prefix} 主动规划感知异常: {_ip}，沿用已有感知缓存")
                elif _ip is not None:
                    awareness_snapshot = _ip
                    self._cached_awareness = awareness_snapshot
                    self._last_perception_ts = now_act
            if _run_v:
                _iv = _map.get("v")
                if isinstance(_iv, BaseException):
                    logger.warning(f"{self.log_prefix} 主动规划独白异常: {_iv}，沿用已有独白缓存")
                elif _iv is not None:
                    voice_conclusion = _iv
                    self._cached_voice = voice_conclusion
                    self._last_voice_ts = now_act
                    self._last_proactive_voice_ts = now_act
        if voice_conclusion is None:
            voice_conclusion = self._cached_voice
        # R4修复：提取内心独白中的意图id，供闭环追踪使用
        _proactive_intent_id = ""
        if voice_conclusion is not None and hasattr(voice_conclusion, "dominant_unfinished_intent"):
            _dom = getattr(voice_conclusion, "dominant_unfinished_intent", None)
            if _dom and isinstance(_dom, dict):
                _proactive_intent_id = str(_dom.get("intent_id", "") or "")
        self._last_proactive_intent_id = _proactive_intent_id
        # 从 VoiceVerdict 对象提取思考文本
        thought_text = ""
        if voice_conclusion is not None:
            if hasattr(voice_conclusion, "thinking"):
                thought_text = voice_conclusion.thinking
            elif isinstance(voice_conclusion, str):
                thought_text = voice_conclusion
        if not thought_text:
            thought_text = "感觉有点想说话"

        # ── 流程规划器决策：是否适合主动发言 + 发言策略 ──
        _proactive_plan = None
        try:
            from src.chat.heart_flow.flow_planner import acquire_flow_planner

            _mood_label = "平静"
            if emotion_snapshot is not None:
                _mood_label = str(
                    getattr(emotion_snapshot, "mood_label", "")
                    or getattr(emotion_snapshot, "description", "")
                    or "平静"
                )
            _recent_topic = ""
            # 从最近的 bot 发言和频道氛围构建真实话题上下文
            _bot_recent = self._recent_bot_texts()
            if _bot_recent:
                _recent_topic = _bot_recent[-1][:100]
            elif thought_text and thought_text != "感觉有点想说话":
                _recent_topic = thought_text[:100]
            # 构建 bot 最近发言摘要，让规划器知道之前聊了什么
            _bot_context = ""
            if _bot_recent:
                _ctx_lines = [f"- {t[:80]}" for t in _bot_recent[-3:]]
                _bot_context = "你最近说过的话:\n" + "\n".join(_ctx_lines)
            # 从内心独白导出最近的观察记忆（看过什么、想过什么、为什么沉默）
            _observation_ctx = ""
            try:
                from src.chat.heart_flow.inner_voice import get_self_dialogue_engine

                _obs_engine = get_self_dialogue_engine(self.stream_id)
                _observation_ctx = _obs_engine.export_observation_context()
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
            if _observation_ctx:
                _bot_context = f"{_bot_context}\n{_observation_ctx}" if _bot_context else _observation_ctx
            # 构建近期对话记录（带用户名），让规划器知道群里谁说了什么
            _dialogue_lines = []
            _recent_msgs = []
            try:
                _recent_msgs = message_api.get_messages_by_time_in_chat(
                    chat_id=self.stream_id,
                    start_time=time.time() - 600,
                    end_time=time.time(),
                    limit=15,
                    limit_mode="latest",
                )
                for _rm in _recent_msgs:
                    _rm_nick = str(
                        getattr(_rm, "user_nickname", "")
                        or getattr(_rm, "user_cardname", "")
                        or getattr(_rm, "user_id", "")
                        or "未知"
                    )
                    _rm_text = str(
                        getattr(_rm, "processed_plain_text", "") or getattr(_rm, "plain_text", "") or ""
                    ).strip()
                    if _rm_text:
                        _dialogue_lines.append(f"{_rm_nick}: {_rm_text[:120]}")
            except Exception as _exc:
                logger.debug(f"非关键异常: {_exc}")
            _recent_dialogue = "\n".join(_dialogue_lines) if _dialogue_lines else ""
            # ── 构建社交融入三维上下文 ──
            # A. 群聊氛围感知
            _atmosphere_parts = []
            if ambient_info:
                _vit = float(ambient_info.get("vitality", 0) or 0)
                _wear = float(ambient_info.get("weariness", 0) or 0)
                _vex = float(ambient_info.get("vexation", 0) or 0)
                _cat = str(ambient_info.get("category", "未知") or "未知")
                _atmosphere_parts.append(f"频道氛围类型: {_cat}")
                if _vit > 60:
                    _atmosphere_parts.append("群里气氛挺活跃的")
                elif _vit < 25:
                    _atmosphere_parts.append("群里比较冷清")
                if _wear > 50:
                    _atmosphere_parts.append("大家似乎有点疲倦")
                if _vex > 40:
                    _atmosphere_parts.append("气氛有些烦躁")
            # 统计近期活跃用户
            _active_users = {}
            for _rm in _recent_msgs:
                _uid = str(getattr(_rm, "user_id", "") or "")
                _nick = str(getattr(_rm, "user_nickname", "") or getattr(_rm, "user_cardname", "") or _uid)
                if self._is_human_message_obj(_rm) and _nick:
                    _active_users[_uid] = _nick
            if _active_users:
                _atmosphere_parts.append(f"最近参与聊天的人: {', '.join(_active_users.values())}")
            _atmosphere_ctx = "\n".join(_atmosphere_parts) if _atmosphere_parts else ""
            # B. 人际关系地图（对每个活跃用户的关系感知）
            _relationship_parts = []
            if _active_users:
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker,
                    )

                    _emo_tracker = get_emotion_tracker(self.stream_id)
                    for _uid, _nick in _active_users.items():
                        _user_state = _emo_tracker.get_user_state(_uid, create_if_missing=False)
                        if _user_state is None:
                            _relationship_parts.append(f"{_nick}: 不太熟，没什么印象")
                            continue
                        _r_trust = float(getattr(_user_state, "trust_accumulation", 0) or 0)
                        _r_annoy = float(getattr(_user_state, "annoyance", 0) or 0)
                        _r_intimacy = float(getattr(_user_state, "intimacy", 0) or 0)
                        _r_fondness = float(getattr(_user_state, "fondness", 0) or 0)
                        _r_blocked = bool(getattr(_user_state, "is_blocked", False))
                        _r_parts = []
                        if _r_blocked:
                            _r_parts.append("已屏蔽")
                        elif _r_intimacy > 60:
                            _r_parts.append("很熟")
                        elif _r_intimacy > 30:
                            _r_parts.append("有一定了解")
                        else:
                            _r_parts.append("不太熟")
                        if _r_fondness > 50:
                            _r_parts.append("挺喜欢他的")
                        elif _r_fondness < -20:
                            _r_parts.append("不太喜欢他")
                        if _r_annoy > 60:
                            _r_parts.append("最近对他很烦")
                        elif _r_annoy > 30:
                            _r_parts.append("有点烦他")
                        if _r_trust > 40:
                            _r_parts.append("比较信任")
                        _relationship_parts.append(
                            f"{_nick}: {'，'.join(_r_parts)}" if _r_parts else f"{_nick}: 印象一般"
                        )
                except Exception as _exc:
                    logger.debug(f"非关键异常: {_exc}")
            _relationship_ctx = (
                "你对群里这些人的感觉:\n" + "\n".join(_relationship_parts) if _relationship_parts else ""
            )
            # C. 自身社交状态
            _self_state_parts = []
            _ws = getattr(self, "_tick_world_snapshot", None)
            if _ws is not None:
                _res = getattr(_ws, "self_resources", None)
                if _res is not None:
                    _chat_pct = int(getattr(_res, "chat_ratio", lambda: 0.5)() * 100)
                    _think_pct = int(getattr(_res, "thinking_ratio", lambda: 0.5)() * 100)
                    _boredom = float(getattr(_res, "boredom", 0) or 0)
                    _loneliness = float(getattr(_res, "loneliness", 0) or 0)
                    _social_desire = float(getattr(_res, "social_desire", 0) or 0)
                    _consec = int(getattr(_res, "consecutive_replies", 0) or 0)
                    _self_state_parts.append(f"聊天精力: {_chat_pct}%，思考精力: {_think_pct}%")
                    if _boredom > 50:
                        _self_state_parts.append("你现在挺无聊的")
                    if _loneliness > 50:
                        _self_state_parts.append("你有点孤独，想找人聊天")
                    if _social_desire > 60:
                        _self_state_parts.append("社交欲望很强")
                    elif _social_desire < 20:
                        _self_state_parts.append("不太想社交")
                    if _consec > 3:
                        _self_state_parts.append(f"你已经连续主动说了{_consec}次，可能该歇歇了")
            _self_state_ctx = "\n".join(_self_state_parts) if _self_state_parts else ""
            _proactive_plan = await acquire_flow_planner().generate_proactive_plan(
                channel_id=self.stream_id,
                quiet_seconds=silence_sec,
                recent_topic=_recent_topic,
                mood=_mood_label,
                recent_bot_context=_bot_context,
                recent_dialogue=_recent_dialogue,
                atmosphere_context=_atmosphere_ctx,
                relationship_context=_relationship_ctx,
                self_state_context=_self_state_ctx,
            )
            if _proactive_plan is None:
                logger.info(f"{self.log_prefix} [流程规划] 规划器否决主动发言")
                return False
            if _proactive_plan.reply_strategy:
                thought_text = _proactive_plan.reply_strategy
            logger.info(
                f"{self.log_prefix} [流程规划] 主动策略: "
                f"situation={_proactive_plan.situation.value} "
                f"emotion={_proactive_plan.emotion_hint} "
                f"length={_proactive_plan.suggested_length} "
                f"delivery={_proactive_plan.delivery_form} "
                f"ref={_proactive_plan.reference_user_name} "
                f"mention={_proactive_plan.mention_user_name}"
            )
        except Exception as _fp_exc:
            logger.debug(f"{self.log_prefix} 流程规划器异常(降级): {_fp_exc}")

        # 从内心独白提取实际 desire，不再硬编码
        _actual_desire = 5
        if voice_conclusion is not None:
            _actual_desire = int(getattr(voice_conclusion, "reply_desire_level", 5) or 5)
        # 从规划器结果提取话题和情绪
        _proactive_topic = ""
        _proactive_emotion = ""
        _delivery_form = "standalone"
        _mention_user_name = ""
        _reference_user_name = ""
        if _proactive_plan is not None:
            _proactive_topic = str(getattr(_proactive_plan, "extra_notes", "") or "")
            _proactive_emotion = str(getattr(_proactive_plan, "emotion_hint", "") or "")
            _delivery_form = str(getattr(_proactive_plan, "delivery_form", "standalone") or "standalone")
            _mention_user_name = str(getattr(_proactive_plan, "mention_user_name", "") or "")
            _reference_user_name = str(getattr(_proactive_plan, "reference_user_name", "") or "")

        # 收集近期消息供执行器定位引用目标
        _proactive_incoming = []
        try:
            _proactive_incoming = message_api.get_messages_by_time_in_chat(
                chat_id=self.stream_id,
                start_time=time.time() - 600,
                end_time=time.time(),
                limit=15,
                limit_mode="latest",
            )
        except Exception as _exc:
            logger.debug(f"非关键异常: {_exc}")

        _pre_reply_resource_snapshot = self._capture_pre_reply_resource_snapshot()
        self._apply_plan_drain()
        try:
            acted = await asyncio.wait_for(
                self._generate_and_send_proactive_reply(
                    incoming_batch=_proactive_incoming,
                    desire=_actual_desire,
                    thought=thought_text,
                    target_uid=self._resolve_latest_human_user_id(
                        allow_cached_fallback=False
                    )
                    or "",
                    awareness=awareness_snapshot,
                    ambient=ambient_info,
                    emotion=emotion_snapshot,
                    arbiter_reason=reason,
                    proactive_topic=_proactive_topic,
                    proactive_emotion=_proactive_emotion,
                    delivery_form=_delivery_form,
                    mention_user_name=_mention_user_name,
                    reference_user_name=_reference_user_name,
                ),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            acted = False
            logger.error(f"{self.log_prefix} ⚠️ 后台主动回复超时(120s)")
        if acted:
            self._last_idle_proactive_ts = now_act
            await self._finalize_external_proactive_reply_flow(
                _proactive_incoming,
                source="background_proactive",
                desire_level=_actual_desire,
            )
        else:
            self._restore_pre_reply_resource_snapshot(
                _pre_reply_resource_snapshot,
                reason="background_proactive未形成有效回复",
            )
        return acted

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


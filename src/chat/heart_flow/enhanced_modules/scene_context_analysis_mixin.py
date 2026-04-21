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

class SceneContextAnalysisMixin:
    async def _check_identity_context(self, messages: List) -> Dict[str, Any]:
        """身份锚点集成 - 确认当前身份状态"""
        try:
            from src.core.identity_anchor import get_identity_anchor

            anchor = get_identity_anchor()
            user_id = ""
            if messages:
                for msg in reversed(messages):
                    uid = getattr(msg, "user_id", "")
                    if self._is_human_message_obj(msg):
                        user_id = uid
                        break
            context = anchor.get_identity(self.stream_id, user_id)
            conflict = anchor.check_conflict(self.stream_id, user_id)
            if conflict:
                logger.info(f"{self.log_prefix} 🔒 检测到身份冲突: {conflict.conflict_type}")
            return {
                "identity": (context.current_mode.value if context else "default"),
                "response_mode": (context.response_mode.value if context else "normal"),
                "has_conflict": conflict is not None,
            }
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 身份锚点检查失败: {exc}")
            return {
                "identity": "default",
                "response_mode": "normal",
                "has_conflict": False,
            }

    async def _update_dynamic_context(self, messages: List) -> None:
        """兼容占位：消息入口已统一写入 ContextManager，这里不再双写旧动态上下文。"""
        return None

    async def _check_self_reply_risk(self, messages: List) -> Dict[str, Any]:
        """自回复识别集成 - 检查是否在重复回复"""
        try:
            from src.core.self_reply_recognizer import (
                get_self_reply_recognizer,
            )

            recognizer = get_self_reply_recognizer()
            last_bot_msg = None
            for msg in reversed(messages):
                if self._is_bot_message_obj(msg):
                    last_bot_msg = msg
                    break
            if last_bot_msg is None:
                return {"is_self_reply": False, "similarity": 0.0}
            content = getattr(last_bot_msg, "processed_plain_text", "") or getattr(last_bot_msg, "content", "")
            result = recognizer.check_self_reply(
                channel_id=self.stream_id,
                content=content,
                author_id="bot",
            )
            if result.is_self_reply:
                logger.warning(f"{self.log_prefix} 🔄 检测到重复回复风险: 相似度={result.similarity:.2f}")
            return {
                "is_self_reply": result.is_self_reply,
                "similarity": result.similarity,
                "quality": (result.quality.value if hasattr(result, "quality") else "unknown"),
            }
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 自回复识别失败: {exc}")
            return {"is_self_reply": False, "similarity": 0.0}

    async def _track_content_state(self, messages: List) -> None:
        """内容状态追踪集成 - 记录内容处理状态"""
        try:
            from src.core.content_state_tracker import (
                get_content_state_tracker,
            )

            tracker = get_content_state_tracker()
            for msg in messages:
                content = self._extract_message_content(msg)
                if not content:
                    continue
                user_id = getattr(msg, "user_id", "")
                tracker.track_content(
                    content=content,
                    channel_id=self.stream_id,
                    user_id=user_id,
                )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 内容状态追踪失败: {exc}")

    def _evaluate_content_state_signal(self, messages: List) -> Dict[str, Any]:
        """用现有内容状态追踪器判断最新目标内容是否值得继续处理。"""
        result = {
            "should_skip": False,
            "reason": "",
            "decision_reason": "",
            "suggested_action": "",
            "confidence": 0.0,
            "target_content": "",
        }
        try:
            from src.core.content_state_tracker import (
                get_content_state_tracker,
            )

            target_message = self._get_latest_human_message(messages)
            if target_message is None:
                return result
            content = self._extract_message_content(target_message).strip()
            if not content:
                return result
            tracker = get_content_state_tracker()
            decision = tracker.should_process(
                content=content,
                channel_id=self.stream_id,
                user_id=getattr(target_message, "user_id", ""),
            )
            result.update(
                {
                    "decision_reason": decision.reason,
                    "suggested_action": decision.suggested_action,
                    "confidence": float(decision.confidence or 0.0),
                    "target_content": content,
                }
            )
            if decision.should_process:
                return result
            reason_map = {
                "low_interest": "当前内容信息量偏低，先不重复接话",
                "currently_processing": "当前内容正在处理中，避免并行重复回复",
                "over_processed": "这类内容处理次数过多，保持克制",
                "already_well_replied": "这类内容刚处理过且回应质量足够",
            }
            result["should_skip"] = True
            result["reason"] = reason_map.get(
                decision.reason,
                f"内容状态追踪建议跳过: {decision.reason}",
            )
            return result
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 内容状态判定失败: {exc}")
            return result

    def _apply_content_state_skip(self, content_state_signal: Optional[Dict[str, Any]]) -> None:
        """把内容状态拒绝结果回写给追踪器，避免同一类低价值内容重复进入主链。"""
        if not content_state_signal:
            return
        content = str(content_state_signal.get("target_content", "") or "").strip()
        if not content:
            return
        reason = str(content_state_signal.get("decision_reason", "") or "")
        try:
            from src.core.content_state_tracker import (
                IgnoreReason,
                get_content_state_tracker,
            )

            tracker = get_content_state_tracker()
            if reason == "low_interest":
                tracker.mark_ignored(content, IgnoreReason.LOW_INTEREST)
            elif reason == "over_processed":
                tracker.mark_ignored(content, IgnoreReason.OVER_PROCESSED)
            elif reason == "already_well_replied":
                tracker.mark_ignored(content, IgnoreReason.ALREADY_REPLIED)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 回写内容状态失败: {exc}")

    def _extract_message_content(self, msg) -> str:
        if msg is None:
            return ""
        return (
            getattr(msg, "processed_plain_text", "")
            or getattr(msg, "raw_plain_text", "")
            or getattr(msg, "content", "")
            or ""
        )

    def _get_latest_human_message(
        self,
        messages: List,
        preferred_sources: Tuple[str, ...] = ("incoming",),
        allow_fallback: bool = True,
    ):
        human_messages = self._get_human_message_candidates(
            list(messages[-10:]),
            preferred_sources=preferred_sources,
            allow_fallback=allow_fallback,
        )
        return human_messages[-1] if human_messages else None

    def _mark_message_content_processing(self, msg) -> None:
        content = self._extract_message_content(msg).strip()
        if not content:
            return
        try:
            from src.core.content_state_tracker import (
                get_content_state_tracker,
            )

            get_content_state_tracker().mark_processing(content)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 标记内容处理中失败: {exc}")

    def _mark_message_content_processed(
        self,
        msg,
        action_name: str,
        quality: float = 0.5,
    ) -> None:
        content = self._extract_message_content(msg).strip()
        if not content:
            return
        try:
            from src.core.content_state_tracker import (
                get_content_state_tracker,
            )

            get_content_state_tracker().mark_processed(
                content,
                action=action_name,
                quality=quality,
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 标记内容已处理失败: {exc}")

    def _mark_message_content_deferred(self, msg, reason: str) -> None:
        content = self._extract_message_content(msg).strip()
        if not content:
            return
        try:
            from src.core.content_state_tracker import (
                get_content_state_tracker,
            )

            get_content_state_tracker().mark_deferred(content, reason)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 标记内容延后失败: {exc}")

    async def _analyze_group_sense(self, messages: List) -> Dict[str, Any]:
        """
        群聊感知分析 - 分析群聊态势（爆发、争议、沉默等）

        返回：
        - activity_level: 活跃度等级（死寂/冷清/平稳/活跃/爆发）
        - burst_detected: 是否检测到消息爆发
        - controversy_detected: 是否检测到争议
        - dominant_users: 主导用户列表
        - needs_topic: 是否需要新话题
        - description: 态势描述
        """
        try:
            from src.modules.perception.group_sense import get_group_sense

            group_sense = get_group_sense()
            recent_messages = []
            for msg in messages[-50:]:
                if not self._is_human_message_obj(msg):
                    continue
                recent_messages.append(
                    {
                        "sender_id": getattr(msg, "user_id", ""),
                        "timestamp": getattr(msg, "timestamp", 0.0),
                        "text": getattr(msg, "processed_plain_text", "") or getattr(msg, "content", ""),
                    }
                )
            result = group_sense.analyze(
                stream_id=self.stream_id,
                recent_messages=recent_messages,
                last_reply_time=self.last_active_time,
            )
            if result.burst_detected or result.controversy_detected:
                logger.info(f"{self.log_prefix} 👁️ {result.description}")
            self._last_group_sense = result
            return self._merge_group_context_signal(
                {
                    "activity_level": result.activity_level,
                    "message_count_5min": result.message_count_5min,
                    "active_user_count": result.active_user_count,
                    "silence_duration_seconds": result.silence_duration_seconds,
                    "burst_detected": result.burst_detected,
                    "burst_density": result.burst_density,
                    "controversy_detected": result.controversy_detected,
                    "dominant_users": result.dominant_users,
                    "needs_topic": result.needs_topic,
                    "topic_hints": result.topic_hints,
                    "description": result.description,
                }
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 群聊感知失败: {exc}")
            return self._merge_group_context_signal(
                {
                    "activity_level": "未知",
                    "burst_detected": False,
                    "controversy_detected": False,
                    "needs_topic": False,
                    "topic_hints": [],
                }
            )

    async def _analyze_message_preprocessor(self, messages: List) -> Dict[str, Any]:
        """把消息预处理器输出转成主链可直接消费的刷屏与活跃信号。"""
        result = {
            "is_spam": False,
            "spam_type": "正常",
            "activity_level": "平稳",
            "unique_users": 0,
            "has_at_me": False,
        }
        try:
            from src.modules.perception.message_preprocessor import (
                StandardMessage,
                get_message_preprocessor,
            )

            standardized = []
            for msg in messages[-20:]:
                standardized.append(
                    StandardMessage(
                        message_id=str(
                            getattr(msg, "message_id", "") or getattr(msg, "msg_id", "") or getattr(msg, "id", "")
                        ),
                        text=self._extract_message_content(msg),
                        sender_id=getattr(msg, "user_id", "") or "",
                        sender_name=(
                            getattr(msg, "user_nickname", "")
                            or getattr(msg, "user_name", "")
                            or getattr(msg, "nickname", "")
                            or "未知用户"
                        ),
                        timestamp=float(getattr(msg, "timestamp", 0.0) or 0.0),
                        is_at_me=bool(getattr(msg, "is_at_me", False) or getattr(msg, "is_at", False)),
                        has_image=bool(getattr(msg, "has_image", False) or getattr(msg, "image_urls", None)),
                        image_urls=list(getattr(msg, "image_urls", []) or []),
                        is_bot_self=self._is_bot_message_obj(msg),
                        content_type=str(getattr(msg, "content_type", "text") or "text"),
                        is_quote_reply=bool(getattr(msg, "is_quote_reply", False)),
                        quoted_content=str(getattr(msg, "quoted_content", "") or ""),
                        quoted_sender=str(getattr(msg, "quoted_sender", "") or ""),
                        is_forward=bool(getattr(msg, "is_forward", False)),
                        forward_title=str(getattr(msg, "forward_title", "") or ""),
                        forward_items=list(getattr(msg, "forward_items", []) or []),
                    )
                )
            batch = get_message_preprocessor().process(standardized)
            meta = batch.meta
            result.update(
                {
                    "is_spam": bool(meta.is_spam),
                    "spam_type": meta.spam_type,
                    "activity_level": meta.activity_level,
                    "unique_users": int(meta.unique_users),
                    "has_at_me": bool(meta.has_at_me),
                }
            )
            if meta.is_spam:
                logger.info(f"{self.log_prefix} 🚫 检测到{meta.spam_type}")
            return result
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 消息预处理分析失败: {exc}")
            return result

    def _filter_messages_by_user_preference(self, messages: List) -> List:
        """
        用户消息过滤 - 识别刷屏用户和不感兴趣的用户

        过滤规则：
        1. 刷屏用户：短时间内发送大量消息的用户
        2. 不感兴趣的用户：被打上"不感兴趣"标签的用户
        3. 骚扰用户：厌烦值过高的用户
        """
        try:
            from src.modules.social_value.social_affect_fuser import (
                get_social_affect_fuser,
            )

            filtered = []
            user_msg_count = {}
            spam_threshold = 5

            for msg in messages:
                user_id = getattr(msg, "user_id", "")
                if self._is_bot_message_obj(msg):
                    continue

                user_msg_count[user_id] = user_msg_count.get(user_id, 0) + 1

            fuser = get_social_affect_fuser()

            for msg in messages:
                user_id = getattr(msg, "user_id", "")
                if self._is_bot_message_obj(msg):
                    filtered.append(msg)
                    continue

                msg_count = user_msg_count.get(user_id, 0)
                if msg_count > spam_threshold:
                    if not hasattr(self, "_spam_warned_users"):
                        self._spam_warned_users: dict = {}
                    now_ts = time.time()
                    # 清理超过 1 小时的记录
                    if len(self._spam_warned_users) > 200:
                        self._spam_warned_users = {
                            k: v for k, v in self._spam_warned_users.items() if now_ts - v < 3600
                        }
                    if user_id not in self._spam_warned_users:
                        logger.info(f"{self.log_prefix} 🚫 用户 {user_id[:8]} 刷屏 ({msg_count}条)，降低响应优先级")
                        self._spam_warned_users[user_id] = now_ts
                    try:
                        from src.chat.heart_flow.reply_coordinator import acquire_reply_coordinator

                        self._spawn(
                            acquire_reply_coordinator().cancel_followup_tasks(
                                self.stream_id,
                                reason=f"spam_user={user_id[:8]} count={msg_count}",
                            ),
                            name="cancel_followups_on_spam",
                        )
                    except Exception as _cancel_exc:
                        logger.debug(f"{self.log_prefix} 刷屏取消补充任务异常: {_cancel_exc}")
                    if self._proactive_task and not self._proactive_task.done():
                        self._proactive_task.cancel()
                        logger.info(f"{self.log_prefix} 🚫 检测到刷屏，取消当前主动后台任务")
                    msg._spam_user = True

                dossier = fuser.get_dossier(user_id, self.stream_id)
                if dossier and dossier.custom_nick:
                    negative_labels = [
                        "不感兴趣",
                        "无聊",
                        "讨厌",
                        "骚扰",
                        "烦人",
                        "屏蔽",
                    ]
                    if any(nl in dossier.custom_nick for nl in negative_labels):
                        logger.debug(f"{self.log_prefix} 🚫 跳过不感兴趣用户: {user_id[:8]} ({dossier.custom_nick})")
                        continue

                filtered.append(msg)

            return filtered
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 用户过滤失败: {exc}")
            return messages

    def _get_user_attention_distribution(self, messages: List) -> Dict[str, float]:
        """
        计算用户注意力分配 - 群聊中平均分配注意力

        返回每个用户应该获得的注意力权重
        """
        try:
            user_activity = {}
            now = time.time()
            window = 300.0

            for msg in messages:
                user_id = getattr(msg, "user_id", "")
                if self._is_bot_message_obj(msg):
                    continue
                ts = getattr(msg, "timestamp", 0.0)
                if now - ts <= window:
                    user_activity[user_id] = user_activity.get(user_id, 0) + 1

            if not user_activity:
                return {}

            total_msgs = sum(user_activity.values())
            user_count = len(user_activity)

            distribution = {}
            for user_id, count in user_activity.items():
                activity_ratio = count / total_msgs
                fairness_bonus = 1.0 / user_count
                weight = 0.7 * fairness_bonus + 0.3 * activity_ratio
                distribution[user_id] = weight

            total_weight = sum(distribution.values())
            if total_weight > 0:
                for user_id in distribution:
                    distribution[user_id] /= total_weight

            return distribution
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 注意力分配计算失败: {exc}")
            return {}

    async def _update_user_interaction_styles(self, messages: List) -> None:
        """
        更新用户交互风格 - 使用 LLM 分析用户风格并更新到 UserImpression

        直接使用现有的 UserImpression.interaction_style 字段，避免孤岛代码
        """
        try:
            from src.modules.social_value.social_affect_fuser import (
                get_social_affect_fuser,
            )

            fuser = get_social_affect_fuser()
            user_messages = {}
            for msg in messages:
                user_id = getattr(msg, "user_id", "")
                content = getattr(msg, "processed_plain_text", "") or getattr(msg, "content", "")
                if user_id and not self._is_bot_message_obj(msg) and content:
                    if user_id not in user_messages:
                        user_messages[user_id] = []
                    user_messages[user_id].append(content)

            for user_id, msg_list in user_messages.items():
                recent_msgs = msg_list[-5:]
                try:
                    style_desc = await asyncio.wait_for(
                        self._analyze_user_style_by_llm(user_id, recent_msgs),
                        timeout=20.0,
                    )
                except asyncio.TimeoutError:
                    logger.debug(f"{self.log_prefix} 用户风格分析超时(20s)，跳过用户{user_id[:8]}")
                    continue
                if style_desc:
                    await fuser.refresh_impression(
                        user_id=user_id,
                        channel_id=self.stream_id,
                        interaction_digest="；".join(recent_msgs[-3:]),
                        style=style_desc,
                    )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 用户风格更新失败: {exc}")

    async def _analyze_user_style_by_llm(self, user_id: str, messages: List[str]) -> str:
        """使用 LLM 分析用户的回复风格"""
        if not messages:
            return ""

        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            llm = LLMRequest(
                model_set=model_config.model_task_config.utils,
                request_type="user_style_analysis",
            )

            messages_text = "\n".join([f"- {msg}" for msg in messages[-5:]])

            prompt = f"""请分析以下用户的回复风格特征。

用户最近的消息：
{messages_text}

请从以下维度分析用户的风格，并用 JSON 格式输出：

1. length_preference: 长度偏好（极简/简洁/适中/详细/啰嗦）
2. tone: 语气风格（正式/随意/幽默/严肃/撒娇/高冷/温和/活泼）
3. personality: 性格特点（如：话少、话痨、爱玩梗、可爱、冷淡、阳光、负能量等，列出2-3个）
4. reply_suggestion: 对回复这个用户的建议（一句话）

请直接输出 JSON，不要有其他内容：
{{"length_preference": "...", "tone": "...", "personality": ["...", "..."], "reply_suggestion": "..."}}"""

            response, _ = await llm.generate_response_async(prompt)

            import json
            from json_repair import repair_json

            try:
                result = repair_json(response)
                if isinstance(result, str):
                    result = json.loads(result)
            except Exception:
                result = {}

            if not isinstance(result, dict):
                result = {}

            length = result.get("length_preference", "适中")
            tone = result.get("tone", "随意")
            personality = result.get("personality", [])
            suggestion = result.get("reply_suggestion", "")

            style_desc = f"{length}、{tone}"
            if personality:
                style_desc += f"，特点：{'、'.join(personality[:3])}"
            if suggestion:
                style_desc += f" | 建议：{suggestion}"

            logger.debug(f"{self.log_prefix} 🎨 用户 {user_id[:8]}: {style_desc}")
            return style_desc

        except Exception as e:
            logger.debug(f"{self.log_prefix} LLM 风格分析失败: {e}，使用规则分析")
            return self._analyze_user_style_by_rules(messages[-1] if messages else "")

    def _analyze_user_style_by_rules(self, message: str) -> str:
        """规则分析用户风格（LLM 失败时的保守降级）。"""
        if not message:
            return ""

        msg_length = len(message)

        length_pref = "适中"
        if msg_length <= 10:
            length_pref = "极简"
        elif msg_length <= 25:
            length_pref = "简洁"
        elif msg_length <= 50:
            length_pref = "适中"
        elif msg_length <= 100:
            length_pref = "详细"
        else:
            length_pref = "啰嗦"

        tone = "中性"
        exclaim_count = message.count("!") + message.count("！")
        question_count = message.count("?") + message.count("？")
        if exclaim_count >= 2:
            tone = "外放"
        elif question_count >= 2:
            tone = "探询"
        elif msg_length <= 5:
            tone = "克制"
        elif msg_length >= 60:
            tone = "展开"

        tags = []
        if length_pref == "极简":
            tags.append("话少")
        elif length_pref == "啰嗦":
            tags.append("话痨")
        if tone == "外放":
            tags.append("表达强")
        elif tone == "探询":
            tags.append("追问多")
        elif tone == "克制":
            tags.append("回应短")

        style_desc = f"{length_pref}、{tone}"
        if tags:
            style_desc += f"，特点：{'、'.join(tags)}"

        return style_desc

    def _get_user_style_guide(self, user_id: str) -> str:
        """获取当前这位的表达风格指导，用于回复生成"""
        try:
            from src.modules.social_value.social_affect_fuser import (
                get_social_affect_fuser,
            )

            fuser = get_social_affect_fuser()
            dossier = fuser.get_dossier(user_id, self.stream_id)
            if dossier and dossier.impression and dossier.impression.dialogue_style:
                style = dossier.impression.dialogue_style
                guide_parts = []

                if "极简" in style:
                    guide_parts.append("对方偏好极简回复，回复要非常简短，1-2句话即可")
                elif "简洁" in style:
                    guide_parts.append("对方偏好简洁回复，控制在2-3句话")
                elif "啰嗦" in style:
                    guide_parts.append("对方偏好详细回复，可以展开说")

                if "正式" in style:
                    guide_parts.append("语气要正式、礼貌")
                elif "幽默" in style:
                    guide_parts.append("可以适当幽默、玩梗")
                elif "撒娇" in style:
                    guide_parts.append("可以适当卖萌、撒娇")
                elif "高冷" in style:
                    guide_parts.append("回复简洁，不要太热情")

                if guide_parts:
                    return f"[对方表达风格] {style}。回复建议：{'；'.join(guide_parts)}"
            return ""
        except Exception as _exc:
            logger.warning(f"{self.log_prefix} 风格指南构建异常: {_exc}")
            return ""

    def _build_dynamic_length_hint(self, target_message: Any, user_style_guide: str = "") -> str:
        """根据用户输入长度和互动风格构建动态句长约束。"""
        user_text = ""
        if target_message is not None:
            user_text = (
                getattr(target_message, "processed_plain_text", "") or getattr(target_message, "content", "") or ""
            )
        text_len = len((user_text or "").strip())
        if text_len <= 8:
            low, high = 1, 3
        elif text_len <= 24:
            low, high = 2, 5
        elif text_len <= 60:
            low, high = 3, 8
        else:
            low, high = 4, 12

        style = user_style_guide or ""
        if "极简" in style or "高冷" in style:
            low, high = 1, min(3, high)
        elif "简洁" in style:
            low, high = min(low, 2), min(4, high)
        elif "啰嗦" in style or "详细" in style:
            low, high = max(4, low), max(8, high)

        high = min(15, high + random.randint(0, 2))
        low = max(1, min(low, high))
        return (
            f"[回复长度策略] 根据用户输入动态回复，本次控制在{low}~{high}句；短输入偏短，长输入可展开，不要固定句长。"
        )

    def _build_persona_hint(self) -> str:
        """构建主人格提示片段。"""
        parts: List[str] = ["[人格约束] 无论内心独白还是最终回复，都优先服从主人格，不要临时长篇分析，不要偏成旁白。"]
        try:
            from src.person_info.bot_identity import get_bot_identity_manager

            identity = get_bot_identity_manager()
            persona_brief = identity.build_persona_brief(
                self.stream_id,
                include_name_prefix=True,
                include_style=True,
                include_interests=True,
                include_lore=True,
                fallback_text="自然说话，别端着，也别把自己说成系统或工具。",
            )
            if persona_brief:
                parts.append(f"[主人格摘要] {persona_brief}")
            parts.append("[身份边界] 不要把自己说成AI助手、系统、客服或工具；你就是当前聊天现场里的这个人。")
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.persona_engine import (
                get_character_foundry,
            )

            shard = (get_character_foundry().active_prompt_shard() or "").strip()
            if shard:
                parts.append(f"[当前人格片段] {shard[:240]}")
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return "\n".join(parts)

    def _resolve_latest_human_user_id(
        self,
        messages: Optional[List] = None,
        *,
        allow_cached_fallback: bool = True,
    ) -> str:
        """稳定解析当前轮对应的发言对象 ID，避免误用 stream_id 作为 memoir key。"""
        candidates = list(messages or [])
        for msg in reversed(candidates[-20:]):
            uid = str(getattr(msg, "user_id", "") or "").strip()
            if self._is_human_message_obj(msg):
                return uid
        if allow_cached_fallback:
            return str(getattr(self, "_last_user_id", "") or "").strip()
        return ""


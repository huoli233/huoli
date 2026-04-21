import asyncio
import time
import traceback
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING
from src.common.logger import get_logger
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.chat.message_receive.chat_stream import ChatStream, get_chat_manager
from src.chat.planner_actions.action_modifier import ActionModifier
from src.chat.planner_actions.action_manager import ActionManager
from src.chat.heart_flow.hfc_utils import CycleDetail
from src.bw_learner.expression_learner import expression_learner_manager
from src.plugin_system.base.component_types import ActionInfo

if TYPE_CHECKING:
    from src.common.data_models.database_data_model import DatabaseMessages

logger = get_logger("chat_core")

ERROR_LOOP_INFO = {
    "loop_plan_info": {
        "action_result": {
            "action_type": "error",
            "action_data": {},
            "reasoning": "循环处理失败",
        },
    },
    "loop_action_info": {
        "action_taken": False,
        "reply_text": "",
        "command": "",
        "taken_time": time.time(),
    },
}


def is_valid_unknown_word_candidate(word: str) -> bool:
    text = str(word or "").strip()
    if not text:
        return False
    if len(text) > 20:
        return False
    if any(sep in text for sep in ("\n", "\r", "\t")):
        return False
    if any(
        marker in text for marker in (":\\", ":/", "\\", "/", ".py", ".md")
    ):
        return False
    punctuation_count = sum(
        1 for ch in text if ch in "，。！？；：,.!?;:[]{}()<>'\"=_-"
    )
    if punctuation_count >= 2:
        return False
    return True


class ChatCoreBase:
    """群聊与私聊聊天链的公用骨架基类，提供共享的周期管理、行为分类、回复风格决策和行为学习等基础方法。"""

    # 子类覆盖：非回复动作的类型标签（群聊="no_reply"，私聊="wait"）
    _inaction_type: str = "no_reply"

    def __init__(self, chat_id: str):
        self.stream_id: str = chat_id
        self.chat_stream: ChatStream = get_chat_manager().get_stream(
            self.stream_id
        )  # type: ignore
        if not self.chat_stream:
            raise ValueError(f"无法找到聊天流: {self.stream_id}")
        self.log_prefix = f"[{get_chat_manager().get_stream_name(self.stream_id) or self.stream_id}]"
        self.expression_learner = (
            expression_learner_manager.get_expression_learner(self.stream_id)
        )
        self.action_manager = ActionManager()
        self.action_modifier = ActionModifier(
            action_manager=self.action_manager, chat_id=self.stream_id
        )
        self.running: bool = False
        self._loop_task: Optional[asyncio.Task] = None
        self.history_loop: List[CycleDetail] = []
        self._cycle_counter = 0
        self._current_cycle_detail: CycleDetail = None  # type: ignore
        self.last_read_time = time.time() - 2
        self._history_loop_limit: int = 0

    def start_cycle(self) -> Tuple[Dict[str, float], str]:
        self._cycle_counter += 1
        self._current_cycle_detail = CycleDetail(self._cycle_counter)
        self._current_cycle_detail.thinking_id = f"tid{
            str(
                round(
                    time.time(),
                    2))}"
        cycle_timers = {}
        return cycle_timers, self._current_cycle_detail.thinking_id

    def end_cycle(self, loop_info, cycle_timers):
        self._current_cycle_detail.set_loop_info(loop_info)
        self.history_loop.append(self._current_cycle_detail)
        self._current_cycle_detail.timers = cycle_timers
        self._current_cycle_detail.end_time = time.time()
        if (
            self._history_loop_limit > 0
            and len(self.history_loop) > self._history_loop_limit
        ):
            self.history_loop = self.history_loop[
                -(self._history_loop_limit // 2):
            ]

    def print_cycle_info(self, cycle_timers):
        timer_strings = []
        for name, elapsed in cycle_timers.items():
            if elapsed < 0.1:
                continue
            formatted_time = f"{elapsed:.2f}秒"
            timer_strings.append(f"{name}: {formatted_time}")

        logger.info(
            f"{
                self.log_prefix} 第{
                self._current_cycle_detail.cycle_id}次思考,"
            f"耗时: {
                self._current_cycle_detail.end_time -
                self._current_cycle_detail.start_time:.1f}秒;"
            + (
                f"详情: {
                    '; '.join(timer_strings)}"
                if timer_strings
                else ""
            )
        )

    def _handle_loop_completion(self, task: asyncio.Task):
        """主循环任务完成时的通用回调。"""
        class_label = self.__class__.__name__
        try:
            if exception := task.exception():
                logger.error(
                    f"{self.log_prefix} {class_label}: 脱离了聊天(异常): {exception}"
                )
                logger.error(traceback.format_exc())
            else:
                logger.info(
                    f"{self.log_prefix} {class_label}: 脱离了聊天 (外部停止)"
                )
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} {class_label}: 结束了聊天")

    def _classify_behavior_signal(
        self, action_message: Optional["DatabaseMessages"]
    ) -> Dict[str, Any]:
        """根据单条目标消息分类用户行为信号。"""
        result = {
            "category": "neutral",
            "behavior_type": "casual_chat",
            "intent": "chat",
            "severity": 0.25,
            "is_new_user": False,
            "reason": "默认中性互动",
        }
        if action_message is None:
            return result
        user_text = (
            getattr(action_message, "processed_plain_text", "")
            or getattr(action_message, "content", "")
            or ""
        )
        user_id = getattr(action_message, "user_id", "") or ""
        interaction_count = 0
        try:
            from src.chat.heart_flow.fondness_trust import FondnessTrustDimension
from src.chat.heart_flow.social_value_dim import SocialValueDimension
            from src.chat.behavior.behavior_signal import (
                classify_behavior_signal,
            )

            _dossier = (
                FondnessTrustDimension.get_instance().get_dossier(user_id, self.stream_id)
                if user_id
                else None
            )
            interaction_count = len(_dossier.recent_values) if _dossier else 0
            return classify_behavior_signal(
                text=user_text,
                user_id=user_id,
                interaction_count=interaction_count,
            )
        except Exception as e:
            logger.debug(f"行为信号分类失败: {e}")
            from src.chat.behavior.behavior_signal import (
                classify_behavior_signal,
            )

            return classify_behavior_signal(
                text=user_text,
                user_id=user_id,
                interaction_count=interaction_count,
            )

    def _decide_reply_style(
        self,
        action_message: Optional["DatabaseMessages"],
        behavior_signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """根据消息内容和行为信号决定回复形式（直连/引用）。"""
        text = ""
        if action_message is not None:
            text = (
                getattr(action_message, "processed_plain_text", "")
                or getattr(action_message, "content", "")
                or ""
            )
        category = str(behavior_signal.get("category", "neutral") or "neutral")
        quote_message = False
        reply_style = "direct"
        reason = "默认自然续聊"
        if any(
            marker in text for marker in ("?", "？", "怎么", "为什么", "什么")
        ):
            quote_message = True
            reply_style = "quote"
            reason = "明确提问更适合引用回复"
        elif category in {"hostile", "harassing"}:
            quote_message = False
            reply_style = "direct"
            reason = "敌意场景避免贴脸逐句对线"
        elif len(text.strip()) <= 10:
            quote_message = False
            reply_style = "direct"
            reason = "短消息直接接话更自然"
        return {
            "reply_style": reply_style,
            "quote_message": quote_message,
            "reason": reason,
        }

    def _build_relationship_target_hint(
        self, action_message: Optional["DatabaseMessages"]
    ) -> str:
        """从关系网络中提取优先交互候选，辅助话题延展。"""
        user_id = (
            getattr(action_message, "user_id", "")
            if action_message is not None
            else ""
        )
        if not user_id:
            return ""
        try:
            from src.person_info.person_info import get_unified_profile_hub

            top_relationships = (
                get_unified_profile_hub().get_top_relationships(
                    user_id, limit=3
                )
            )
            candidates = [
                f"{str(target_id or '').strip()}({float(weight or 0.0):.2f})"
                for target_id, weight in top_relationships
                if str(target_id or "").strip()
            ]
            if not candidates:
                return ""
            return f"[关系网候选] 如果要延展话题，优先参考: {', '.join(candidates)}"
        except Exception as e:
            logger.debug(f"获取关系候选失败: {e}")
            return ""

    def _select_preferred_reply_message(
        self,
        action_message: Optional["DatabaseMessages"],
        candidate_actions: List[ActionPlannerInfo],
    ) -> Optional["DatabaseMessages"]:
        """结合关系网络从候选消息中选择最优回复目标。"""
        if action_message is None:
            return action_message
        try:
            from src.chat.behavior.target_selector import (
                choose_preferred_message_target,
            )
            from src.person_info.person_info import get_unified_profile_hub

            anchor_user_id = str(
                getattr(action_message, "user_id", "") or ""
            ).strip()
            if not anchor_user_id:
                return action_message
            top_relationships = (
                get_unified_profile_hub().get_top_relationships(
                    anchor_user_id, limit=5
                )
            )
            if not top_relationships:
                return action_message
            candidate_messages = [
                getattr(candidate, "action_message", None)
                for candidate in candidate_actions
            ]
            return choose_preferred_message_target(
                anchor_message=action_message,
                candidate_messages=candidate_messages,
                relationship_candidates=top_relationships,
            )
        except Exception:
            return action_message

    async def _capture_reply_behavior_learning(
        self,
        reply_text: str,
        reply_reason: str,
        action_message: Optional["DatabaseMessages"],
    ) -> None:
        """将本次回复行为回灌到自我行为学习系统。"""
        if not reply_text.strip():
            return
        relation_stage = "unknown"
        user_id = (
            getattr(action_message, "user_id", "")
            if action_message is not None
            else ""
        )
        if user_id:
            try:
                from src.chat.heart_flow.fondness_trust import FondnessTrustDimension
from src.chat.heart_flow.social_value_dim import SocialValueDimension

                _fuser = FondnessTrustDimension.get_instance()
                relation_stage = (
                    _fuser.get_phase_label(user_id, self.stream_id)
                    or "unknown"
                )
            except Exception as e:
                logger.debug(f"获取关系阶段标签失败: {e}")
        try:
            from src.modules.recall.self_behavior_learner import (
                get_self_behavior_learner,
            )

            await get_self_behavior_learner().capture_event(
                stream_id=self.stream_id,
                action_type="reply",
                content=reply_text,
                result="success",
                context={
                    "relation_stage": relation_stage,
                    "reply_reason": reply_reason[:160],
                },
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 回灌自我行为学习失败: {exc}")

    async def _invoke_unified_planner_gate(
        self,
        recent_messages_list: List["DatabaseMessages"],
        force_reply_message: Optional["DatabaseMessages"] = None,
        current_topics_override: Optional[List[str]] = None,
    ) -> Optional[Any]:
        del force_reply_message
        try:
            from src.core.unified_planner import get_unified_planner
            from src.core.world_snapshot import build_world_snapshot

            planner = get_unified_planner()
            user_id = ""
            last_user_text = ""
            for msg in reversed(recent_messages_list or []):
                uid = getattr(msg, "user_id", "") or ""
                if uid and uid != "bot":
                    user_id = uid
                    last_user_text = (
                        getattr(msg, "processed_plain_text", "")
                        or getattr(msg, "content", "")
                        or ""
                    )
                    break

            active_users = [
                str(getattr(msg, "user_id", "") or "").strip()
                for msg in recent_messages_list[-20:]
                if str(getattr(msg, "user_id", "") or "").strip()
                and str(getattr(msg, "user_id", "") or "").strip() != "bot"
            ]

            current_topics = list(current_topics_override or [])
            relationship_candidates: List[Tuple[str, float]] = []
            if user_id:
                try:
                    from src.person_info.person_info import (
                        get_unified_profile_hub,
                    )

                    relationship_candidates = (
                        get_unified_profile_hub().get_top_relationships(
                            user_id, limit=5
                        )
                    )
                except Exception as e:
                    logger.debug(f"获取关系候选列表失败: {e}")
                    relationship_candidates = []

            pre_snapshot = await build_world_snapshot(self.stream_id, user_id)
            decision = await planner.plan(
                channel_id=self.stream_id,
                user_id=user_id,
                trigger="user_message",
                hints={
                    "last_user_intent": last_user_text[:120],
                    "active_users": active_users,
                    "current_topics": current_topics,
                    "relationship_candidates": relationship_candidates,
                },
                world_snapshot=pre_snapshot,
            )
            return decision
        except Exception as e:
            logger.debug(f"规划失败: {e}")
            return None

    def _convert_unified_decision_to_actions(
        self,
        decision: Any,
        recent_messages_list: List["DatabaseMessages"],
        force_reply_message: Optional["DatabaseMessages"] = None,
    ) -> Optional[List[ActionPlannerInfo]]:
        if decision is None:
            return None
        action_value = (
            decision.action.value
            if hasattr(decision.action, "value")
            else str(decision.action)
        )
        if action_value not in {
            "reply",
            "join_conversation",
            "proactive_speak",
            "observe",
            "check_later",
            "wait",
            "no_reply",
            "complete_talk",
            "rest",
        }:
            return None

        target_user_id = str(
            getattr(decision, "target_user_id", "") or ""
        ).strip()
        target_message = force_reply_message
        if target_message is None and target_user_id:
            for msg in reversed(recent_messages_list or []):
                if (
                    str(getattr(msg, "user_id", "") or "").strip()
                    == target_user_id
                ):
                    target_message = msg
                    break
        if target_message is None:
            for msg in reversed(recent_messages_list or []):
                if (
                    getattr(msg, "user_id", "")
                    and getattr(msg, "user_id", "") != "bot"
                ):
                    target_message = msg
                    break

        if action_value == "complete_talk":
            return [
                ActionPlannerInfo(
                    action_type="complete_talk",
                    reasoning=str(
                        getattr(decision, "reason", "统一规划决定结束当前轮次")
                        or "统一规划决定结束当前轮次"
                    ),
                    action_data={
                        "wait_seconds": min(
                            float(
                                getattr(decision, "next_check_seconds", 5.0)
                                or 5.0
                            ),
                            60.0,
                        )
                    },
                    action_message=target_message,
                    action_reasoning=str(
                        getattr(decision, "reason", "") or ""
                    ),
                )
            ]

        if action_value in {
            "observe",
            "check_later",
            "wait",
            "no_reply",
            "rest",
        }:
            return [
                ActionPlannerInfo(
                    action_type=self._inaction_type,
                    reasoning=str(
                        getattr(decision, "reason", "统一规划决定暂不回复")
                        or "统一规划决定暂不回复"
                    ),
                    action_data={
                        "wait_seconds": min(
                            float(
                                getattr(decision, "next_check_seconds", 5.0)
                                or 5.0
                            ),
                            60.0,
                        ),
                        "unified_reason": str(
                            getattr(decision, "reason", "") or ""
                        ),
                    },
                    action_message=target_message,
                    action_reasoning=str(
                        getattr(decision, "reason", "") or ""
                    ),
                )
            ]

        if target_message is None:
            return None

        action_data = {
            "loop_start_time": time.time(),
            "unified_content_plan": str(
                getattr(decision, "content_plan", "") or ""
            ),
            "unified_target_user_id": target_user_id,
            "unified_reason": str(getattr(decision, "reason", "") or ""),
            "unified_reply_style": str(
                getattr(decision, "preferred_reply_style", "") or ""
            ),
            "unified_quote": getattr(decision, "preferred_quote", None),
        }
        if self._inaction_type == "no_reply":
            action_data["quote"] = action_value != "proactive_speak"

        return [
            ActionPlannerInfo(
                action_type="reply",
                reasoning=str(
                    getattr(decision, "reason", "统一规划决定回复")
                    or "统一规划决定回复"
                ),
                action_data=action_data,
                action_message=target_message,
                action_reasoning=str(
                    getattr(decision, "content_plan", "")
                    or getattr(decision, "reason", "")
                    or ""
                ),
            )
        ]

    async def _build_fallback_actions(
        self,
        available_actions: Dict[str, ActionInfo],
        cycle_timers: Dict[str, float],
        force_reply_message: Optional["DatabaseMessages"] = None,
    ) -> List[ActionPlannerInfo]:
        _fallback_msg = (
            getattr(force_reply_message, "plain_text", "")
            if force_reply_message
            else ""
        )
        return [
            ActionPlannerInfo(
                action_type=self._inaction_type,
                reasoning="统一规划失败，使用本地最小等待回退",
                action_data={
                    "wait_seconds": 3.0,
                    "unified_reason": "统一规划失败，使用本地最小等待回退",
                },
                action_message=_fallback_msg,
                action_reasoning="统一规划失败，使用本地最小等待回退",
            )
        ]

    def _build_relation_style_hint(
        self, action_message: Optional["DatabaseMessages"]
    ) -> str:
        """构建关系阶段和心理信号提示词，群聊和私聊共用。"""
        user_id = (
            getattr(action_message, "user_id", "")
            if action_message is not None
            else ""
        )
        if not user_id:
            return ""
        relation_label = "普通"
        trust_value = 0.0
        annoyance_value = 0.0
        affection = 0.0
        pressure = 0.0
        trauma = 0.0
        try:
            from src.chat.heart_flow.fondness_trust import FondnessTrustDimension
from src.chat.heart_flow.social_value_dim import SocialValueDimension

            fuser = FondnessTrustDimension.get_instance()
            relation_label = (
                fuser.get_phase_label(user_id, self.stream_id) or "普通"
            )
            dossier = fuser.get_dossier(user_id, self.stream_id)
            latest_score = 0.0
            if dossier is not None and dossier.recent_values:
                latest_score = dossier.recent_values[-1].value
            if latest_score >= 0:
                trust_value = latest_score * fuser.get_phase_weight(
                    user_id, self.stream_id
                )
                affection = latest_score
                annoyance_value = 0.0
            else:
                trust_value = 0.0
                affection = 0.0
                annoyance_value = abs(latest_score)
        except Exception:
            logger.debug(f"{self.log_prefix} 社交值计算失败，使用默认值")
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            emotion_state = get_emotion_tracker(self.stream_id).get_user_state(
                user_id, create_if_missing=False
            )
            if emotion_state is not None:
                pressure = float(
                    getattr(emotion_state, "psychological_pressure", 0.0)
                    or 0.0
                )
                trauma = float(
                    getattr(emotion_state, "trauma_score", 0.0) or 0.0
                )
        except Exception as e:
            logger.debug(f"{self.log_prefix} 心理状态获取失败: {e}")
        style_hints: List[str] = []
        try:
            from src.modules.recall.self_behavior_learner import (
                get_self_behavior_learner,
            )

            style_hints = get_self_behavior_learner().get_style_hints(
                self.stream_id,
                relation_stage=relation_label,
                limit=3,
            )
        except Exception:
            style_hints = []
        lines = [
            f"[关系阶段] {relation_label}",
            f"[关系信号] 信任={
                trust_value:.1f} 厌烦={
                annoyance_value:.1f} 好感={
                affection:.1f}",
            f"[心理信号] 压力={
                pressure:.1f} 创伤={
                trauma:.1f}",
        ]
        if trust_value < 5 and annoyance_value < 10:
            lines.append(
                "[回复边界] 不要突然装熟，不要莫名强硬，整体保持自然克制。"
            )
        elif annoyance_value >= 25:
            lines.append(
                "[回复边界] 可以略显烦躁，但不要直接攻击，不要无端爆冲。"
            )
        elif trust_value >= 25 or affection >= 30:
            lines.append(
                "[回复边界] 可以更放松一点，允许轻微熟络感和顺口表达。"
            )
        if pressure >= 35:
            lines.append(
                "[表达节奏] 句子不要太满，尽量短一点、松一点，避免解释过度。"
            )
        if trauma >= 35:
            lines.append(
                "[情绪保护] 遇到刺耳内容先收一下，不要把防御感直接顶满。"
            )
        if style_hints:
            lines.append("[自我风格学习]")
            lines.extend(f"- {hint}" for hint in style_hints if hint)
        return "\n".join(line for line in lines if line)

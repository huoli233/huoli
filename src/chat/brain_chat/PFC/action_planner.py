import asyncio
import json
import re
import time
from typing import Dict, Optional, Tuple

from src.chat.brain_chat.PFC.chat_observer import ChatObserver
from src.chat.brain_chat.PFC.conversation_info import ConversationInfo
from src.chat.brain_chat.PFC.observation_info import ObservationInfo
from src.chat.brain_chat.runtime_config import brainchat_module_view
from src.chat.brain_chat.PFC.pfc_utils import parse_json_payload
from src.chat.utils.chat_message_builder import build_readable_messages
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.llm_models.utils_model import LLMRequest
from src.person_info.bot_identity import get_bot_identity_manager

import src.chat.prompts.catalog  # noqa: F401 注册提示词
from src.chat.utils.prompt_builder import global_prompt_manager

logger = get_logger("动作规划")

# --- 行动类型白名单 ---
PERMITTED_ACTIONS = (
    "direct_reply",
    "send_new_message",
    "fetch_knowledge",
    "wait",
    "listening",
    "rethink_goal",
    "end_conversation",
    "block_and_ignore",
    "say_goodbye",
)

class ActionPlanner:
    """行动规划器 —— 综合对话环境信号、行动模式与历史上下文做出下一步决策。"""

    def __init__(self, stream_id: str, private_name: str):
        self.llm = LLMRequest(
            model_set=model_config.model_task_config.planner,
            request_type="action_planning",
        )
        self.bot_name, self.personality_info = self._compose_persona_trait()
        self.private_name = private_name
        self.chat_observer = ChatObserver.get_instance(stream_id, private_name)
        self.bot_account = str(
            getattr(global_config.bot, "qq_account", "") or ""
        )
        # 决策置信度追踪，有助于跨轮次感知上下文质量
        self._prev_confidence = 0.5
        self._load_config()

    def _load_config(self) -> None:
        config = brainchat_module_view("brain_pfc_action")
        self._recent_action_window = int(
            config.get("recent_action_window", 6)
        )
        self._repeated_action_ceiling = int(
            config.get("repeated_action_ceiling", 4)
        )
        self._llm_timeout_seconds = float(
            config.get("llm_timeout_seconds", 30.0)
        )
        self._elapsed_hint_recent_seconds = float(
            config.get("elapsed_hint_recent_seconds", 60.0)
        )
        self._elapsed_hint_recent_minutes_seconds = float(
            config.get("elapsed_hint_recent_minutes_seconds", 300.0)
        )
        self._elapsed_hint_one_hour_seconds = float(
            config.get("elapsed_hint_one_hour_seconds", 3600.0)
        )
        self._goal_clarity_with_goal = float(
            config.get("goal_clarity_with_goal", 0.85)
        )
        self._goal_clarity_without_goal = float(
            config.get("goal_clarity_without_goal", 0.25)
        )
        self._knowledge_depth_empty = float(
            config.get("knowledge_depth_empty", 0.15)
        )
        self._history_continuity_empty = float(
            config.get("history_continuity_empty", 0.2)
        )
        self._consensus_bonus_cap = float(
            config.get("consensus_bonus_cap", 0.12)
        )
        self._consensus_spread_penalty = float(
            config.get("consensus_spread_penalty", 0.15)
        )
        self._risk_avg_interval_fast = float(
            config.get("risk_avg_interval_fast", 2.0)
        )
        self._risk_avg_interval_medium = float(
            config.get("risk_avg_interval_medium", 5.0)
        )
        self._risk_avg_interval_slow = float(
            config.get("risk_avg_interval_slow", 10.0)
        )
        self._risk_staleness_high = float(
            config.get("risk_staleness_high", 600.0)
        )
        self._risk_staleness_medium = float(
            config.get("risk_staleness_medium", 300.0)
        )
        self._tail_streak_high = int(config.get("tail_streak_high", 3))
        self._tail_streak_medium = int(config.get("tail_streak_medium", 2))
        self._low_confidence_threshold = float(
            config.get("low_confidence_threshold", 0.22)
        )
        self._high_risk_threshold = float(
            config.get("high_risk_threshold", 0.3)
        )
        self._confidence_guard_threshold = float(
            config.get("confidence_guard_threshold", 0.45)
        )
        self._send_new_message_risk_threshold = float(
            config.get("send_new_message_risk_threshold", 0.42)
        )
        self._farewell_timeout_seconds = float(
            config.get("farewell_timeout_seconds", 20.0)
        )

    @staticmethod
    def _compose_persona_trait() -> Tuple[str, str]:
        """从统一人格源构造私聊规划阶段使用的人设描述。"""
        identity = get_bot_identity_manager()
        bot_name = identity.get_display_name() or global_config.bot.nickname
        trait = identity.build_persona_brief(
            include_name_prefix=True,
            include_style=False,
            include_interests=False,
            include_lore=False,
            fallback_text="在聊天里保持自然、有人味、不过度端着。",
        )
        return bot_name, trait

    # ==================== 主入口 ====================

    async def plan(
        self,
        observation_info: ObservationInfo,
        conversation_info: ConversationInfo,
        last_successful_reply_action: Optional[str] = None,
    ) -> Tuple[str, str]:
        """综合全部可用信号，规划并返回 (行动类型, 决策理由)。"""
        # --- 收集各维度上下文 ---
        elapsed_hint = self._extract_bot_elapsed_hint(observation_info)
        deadline_hint = self._extract_deadline_hint(conversation_info)
        goals_text = self._compose_goals_block(conversation_info)
        knowledge_text = self._compose_knowledge_block(conversation_info)
        dialogue_text = self._compose_dialogue_block(observation_info)
        history_digest, prev_step_note = self._compose_action_recap(
            conversation_info
        )
        persona_text = self.personality_info
        # --- 上下文质量评估（融合信号置信度 + 行动环路检测） ---
        ctx_confidence = self._gauge_signal_confidence(
            observation_info, conversation_info
        )
        self._prev_confidence = ctx_confidence
        interaction_risk = self._estimate_interaction_risk(observation_info)
        loop_alert = self._scan_for_action_loops(conversation_info)
        # --- 选模板 ---
        is_followup = last_successful_reply_action in (
            "direct_reply",
            "send_new_message",
        )
        chosen_key = "pfc_follow_up" if is_followup else "pfc_initial_reply"
        logger.debug(
            f"[私聊][{
                self.private_name}]选择{
                '追问' if is_followup else '初始'}决策模板"
        )
        # --- 组装最终提示 ---
        final_prompt = await global_prompt_manager.format_prompt(
            chosen_key,
            persona_text=persona_text,
            goals_str=goals_text.strip()
            or "- 目前没有明确对话目标，请考虑设定一个。",
            action_history_summary=history_digest + loop_alert,
            last_action_context=prev_step_note,
            time_since_last_bot_message_info=elapsed_hint,
            timeout_context=deadline_hint,
            chat_history_text=dialogue_text.strip() or "还没有聊天记录。",
            knowledge_info_str=knowledge_text,
        )
        logger.debug(
            f"[私聊][{self.private_name}]行动规划提示词长度={len(final_prompt)}"
        )
        # --- 调用LLM ---
        try:
            raw_response, _ = await asyncio.wait_for(
                self.llm.generate_response_async(final_prompt),
                timeout=self._llm_timeout_seconds,
            )
            logger.debug(
                f"[私聊][{self.private_name}]LLM返回(前300字): {raw_response[:300]}"
            )
        except Exception as exc:
            logger.error(
                f"[私聊][{self.private_name}]行动规划LLM调用失败: {exc}"
            )
            return "wait", f"规划调用异常，暂时等待: {exc}"
        # --- 解析 ---
        parsed_decision = self._extract_decision_json(raw_response)
        candidate_action = parsed_decision.get("action", "wait")
        candidate_reason = parsed_decision.get("reason", "未提供原因")
        # --- 结束对话需要二次确认 ---
        if candidate_action == "end_conversation":
            return await self._deliberate_farewell(
                persona_text, dialogue_text, candidate_action, candidate_reason
            )
        # --- 基于置信度和风险的后置校准 ---
        candidate_action, candidate_reason = self._calibrate_by_context(
            candidate_action,
            candidate_reason,
            ctx_confidence,
            interaction_risk,
        )
        return self._enforce_action_whitelist(
            candidate_action, candidate_reason
        )

    # ==================== 上下文收集 ====================

    def _extract_bot_elapsed_hint(self, obs: ObservationInfo) -> str:
        """回溯记录，提取Bot最近发言距今的时间提示文本。"""
        try:
            records = getattr(obs, "chat_history", None) or []
            for record in reversed(records):
                if not isinstance(record, dict):
                    continue
                sender = str(record.get("user_id", ""))
                ts_raw = record.get("time") or record.get("created_at")
                if sender == self.bot_account and ts_raw:
                    gap = time.time() - float(ts_raw)
                    if gap < self._elapsed_hint_recent_seconds:
                        return f"提示：你上一条成功发送的消息是在 {gap:.1f} 秒前。\n"
                    if gap < self._elapsed_hint_recent_minutes_seconds:
                        return f"提示：你在约 {gap / 60:.1f} 分钟前发送了上一条消息。\n"
                    if gap < self._elapsed_hint_one_hour_seconds:
                        return (
                            f"提示：你已有 {gap / 60:.0f} 分钟没有发言了。\n"
                        )
                    # 超过1小时不再提示，避免信息干扰
                    return ""
        except Exception as exc:
            logger.warning(
                f"[私聊][{self.private_name}]Bot发言间隔计算异常: {exc}"
            )
        return ""

    def _extract_deadline_hint(self, conv: ConversationInfo) -> str:
        """从目标列表末尾检查对方是否长时间未回复。"""
        try:
            entries = getattr(conv, "goal_list", None) or []
            if entries:
                tail = entries[-1]
                if isinstance(tail, dict):
                    text = str(tail.get("goal", ""))
                    if "分钟，思考接下来要做什么" in text:
                        duration_fragment = text.split("，")[0].replace(
                            "你等待了", ""
                        )
                        return f"重要提示：对方已经长时间（{duration_fragment}）没有回复你的消息了。\n"
        except Exception as exc:
            logger.warning(
                f"[私聊][{self.private_name}]超时提示提取失败: {exc}"
            )
        return ""

    def _compose_goals_block(self, conv: ConversationInfo) -> str:
        """将目标列表渲染为可读段落。"""
        parts = []
        try:
            items = getattr(conv, "goal_list", None) or []
            if not items:
                return "- 目前没有明确对话目标，请考虑设定一个。\n"
            for item in items:
                if isinstance(item, dict):
                    g = str(item.get("goal", "目标缺失"))
                    r = str(item.get("reasoning", "无原因"))
                else:
                    g, r = str(item), "无原因"
                parts.append(f"- 目标：{g}\n  原因：{r}")
        except Exception as exc:
            logger.error(f"[私聊][{self.private_name}]目标块组装失败: {exc}")
            return "- 组装目标时出错。\n"
        return "\n".join(parts) + "\n"

    def _compose_knowledge_block(self, conv: ConversationInfo) -> str:
        """将知识列表渲染为提示段落，每条最多截取2000字。"""
        header = "【已获取的相关知识】\n"
        try:
            k_items = getattr(conv, "knowledge_list", None) or []
            if not k_items:
                return header + "- 暂无相关知识。\n"
            recent = k_items[-5:]
            lines = []
            for seq, item in enumerate(recent, start=1):
                if isinstance(item, dict):
                    topic = str(item.get("query", "未知"))
                    body = str(item.get("knowledge", "无内容"))[:2000]
                    origin = str(item.get("source", "未知"))
                    lines.append(
                        f"{seq}. 关于'{topic}'(来源:{origin}): {body}"
                    )
                else:
                    lines.append(f"{seq}. {str(item)[:2000]}")
            return header + "\n".join(lines) + "\n"
        except Exception as exc:
            logger.error(f"[私聊][{self.private_name}]知识块组装失败: {exc}")
            return header + "- 获取知识时出错。\n"

    def _compose_dialogue_block(self, obs: ObservationInfo) -> str:
        """组合历史聊天文本与新到达的未处理消息。"""
        try:
            base_text = (
                getattr(obs, "chat_history_str", "") or "还没有聊天记录。"
            )
            pending_cnt = getattr(obs, "new_messages_count", 0) or 0
            pending_msgs = getattr(obs, "unprocessed_messages", None)
            if pending_cnt > 0 and pending_msgs:
                appended = build_readable_messages(
                    pending_msgs,
                    replace_bot_name=True,
                    timestamp_mode="relative",
                )
                base_text += f"\n--- {pending_cnt}条新消息 ---\n{appended}"
            return base_text
        except Exception as exc:
            logger.error(f"[私聊][{self.private_name}]对话块组合失败: {exc}")
            return "获取聊天记录时出错。"

    def _compose_action_recap(self, conv: ConversationInfo) -> Tuple[str, str]:
        """回顾最近若干次行动，返回 (历史概要, 上一步详情)。"""
        digest_parts = ["你最近执行的行动历史：\n"]
        detail_parts = ["关于你上一次行动：\n"]
        try:
            raw_log = getattr(conv, "done_action", None) or []
            recent = raw_log[-5:]
            if not recent:
                digest_parts.append("- 还没有执行过行动。\n")
                detail_parts.append("- 这是你规划的第一个行动。\n")
                return "".join(digest_parts), "".join(detail_parts)
            for idx, entry in enumerate(recent):
                act_kind, plan_motive, outcome, fail_note, ts_label = (
                    self._normalize_action_record(entry)
                )
                fail_tag = f", 取消/失败原因: {fail_note}" if fail_note else ""
                ts_tag = f"时间:{ts_label}, " if ts_label else ""
                digest_parts.append(
                    f"- {ts_tag}行动:'{act_kind}', 状态:{outcome}{fail_tag}\n"
                )
                # 最后一条展开为详细上下文
                if idx == len(recent) - 1:
                    detail_parts.append(f"- 上次规划的行动: '{act_kind}'\n")
                    detail_parts.append(f"- 规划原因: {plan_motive}\n")
                    if outcome == "done":
                        detail_parts.append("- 该行动已成功执行。\n")
                        if act_kind in ("direct_reply", "send_new_message"):
                            detail_parts.append("- 消息已成功发送至对方。\n")
                    elif outcome == "recall":
                        detail_parts.append("- 该行动最终未能执行/被取消。\n")
                        if fail_note:
                            detail_parts.append(
                                f"- 【重要】失败/取消的具体原因: {fail_note}\n"
                            )
                        else:
                            detail_parts.append(
                                "- 【重要】失败原因未明确记录。\n"
                            )
                    else:
                        detail_parts.append(f"- 当前状态: {outcome}\n")
        except Exception as exc:
            logger.error(f"[私聊][{self.private_name}]行动回顾组装出错: {exc}")
        return "".join(digest_parts), "".join(detail_parts)

    @staticmethod
    def _normalize_action_record(entry) -> Tuple[str, str, str, str, str]:
        """将行动记录（支持dict/tuple/list/str）统一解包为五元组。

        返回: (行动类型, 规划原因, 结局状态, 失败原因, 时间标签)
        """
        if isinstance(entry, dict):
            act = str(entry.get("action", "未知"))
            motive = str(entry.get("plan_reason", "未知规划原因"))
            outcome = str(entry.get("status", "未知"))
            fail = str(entry.get("final_reason", "") or "")
            ts = str(entry.get("time", "") or "")
            # 成功发送的行动简化显示动因
            if outcome == "done" and act in (
                "direct_reply",
                "send_new_message",
            ):
                motive = motive or "成功发送"
            return act, motive, outcome, fail, ts
        if isinstance(entry, (list, tuple)):
            act = str(entry[0]) if len(entry) > 0 else "未知"
            motive = str(entry[1]) if len(entry) > 1 else "未知"
            outcome = str(entry[2]) if len(entry) > 2 else "未知"
            fail = ""
            if outcome == "recall" and len(entry) > 3:
                fail = str(entry[3])
            return act, motive, outcome, fail, ""
        return str(entry), "未知", "未知", "", ""

    # ==================== 决策辅助引擎（信号融合 + 模式挖掘） ====================

    def _gauge_signal_confidence(
        self, obs: ObservationInfo, conv: ConversationInfo
    ) -> float:
        """度量当前上下文的「决策置信度」(0~1)。

        综合四路信号取均值，再施加「信号一致性」奖励。
        使用多信号融合理念，但信号集和加权方式保持独立。
        """
        signal_values = []
        # 信号A：待处理消息的充足程度 —— 有新消息意味着需要响应
        pending = getattr(obs, "new_messages_count", 0) or 0
        msg_readiness = min(pending / 3.0, 1.0)
        signal_values.append(msg_readiness)
        # 信号B：对话目标明确度 —— 有目标时决策方向更清晰
        goal_entries = getattr(conv, "goal_list", None) or []
        goal_clarity = (
            self._goal_clarity_with_goal
            if goal_entries
            else self._goal_clarity_without_goal
        )
        signal_values.append(goal_clarity)
        # 信号C：已获取知识的丰富度
        knowledge_entries = getattr(conv, "knowledge_list", None) or []
        knowledge_depth = (
            min(len(knowledge_entries) / 3.0, 1.0)
            if knowledge_entries
            else self._knowledge_depth_empty
        )
        signal_values.append(knowledge_depth)
        # 信号D：行动历史的连续性 —— 有记录说明对话已进入活跃阶段
        action_log = getattr(conv, "done_action", None) or []
        history_continuity = (
            min(len(action_log) / 4.0, 1.0)
            if action_log
            else self._history_continuity_empty
        )
        signal_values.append(history_continuity)
        if not signal_values:
            return 0.5
        mean_val = sum(signal_values) / len(signal_values)
        # 「一致性奖励」：信号间分散度越低，各维度共识越强
        spread = max(signal_values) - min(signal_values)
        consensus_bonus = max(
            0.0,
            self._consensus_bonus_cap
            - spread * self._consensus_spread_penalty,
        )
        final_score = min(mean_val + consensus_bonus, 1.0)
        logger.debug(
            f"[私聊][{self.private_name}]置信度评估: "
            f"消息={msg_readiness:.2f} 目标={goal_clarity:.2f} "
            f"知识={knowledge_depth:.2f} 历史={history_continuity:.2f} "
            f"→ 最终={final_score:.2f}"
        )
        return final_score

    def _estimate_interaction_risk(self, obs: ObservationInfo) -> float:
        """轻量风险预估(0~1)。

        使用多维风险思路，但只取「消息密度」和「对话停滞度」
        两个最实用的维度。密度异常高可能表示用户刷屏或情绪波动，需要保守应对。
        """
        risk_val = 0.0
        try:
            records = getattr(obs, "chat_history", None) or []
            if len(records) < 3:
                return 0.05
            # 维度1：消息密度 —— 取最近10条的时间间隔
            ts_list = []
            for rec in records[-10:]:
                if not isinstance(rec, dict):
                    continue
                ts_raw = rec.get("time") or rec.get("created_at")
                if ts_raw:
                    ts_list.append(float(ts_raw))
            if len(ts_list) >= 3:
                intervals = [
                    ts_list[i + 1] - ts_list[i]
                    for i in range(len(ts_list) - 1)
                ]
                avg_interval = sum(intervals) / len(intervals)
                if avg_interval < self._risk_avg_interval_fast:
                    risk_val += 0.35
                elif avg_interval < self._risk_avg_interval_medium:
                    risk_val += 0.2
                elif avg_interval < self._risk_avg_interval_slow:
                    risk_val += 0.08
            # 维度2：对话停滞度 —— 最后一条消息距今很久表示对话可能已冷却
            if ts_list:
                staleness = time.time() - ts_list[-1]
                if staleness > self._risk_staleness_high:
                    risk_val += 0.15
                elif staleness > self._risk_staleness_medium:
                    risk_val += 0.05

            # 维度3：Bot 连续发言段，连续发言越多，继续主动发消息风险越高
            tail_bot_streak = 0
            for rec in reversed(records[-8:]):
                if not isinstance(rec, dict):
                    continue
                sender = str(rec.get("user_id", "") or "")
                if sender != self.bot_account:
                    break
                tail_bot_streak += 1
            if tail_bot_streak >= self._tail_streak_high:
                risk_val += 0.35
            elif tail_bot_streak == self._tail_streak_medium:
                risk_val += 0.22
        except Exception as exc:
            logger.warning(f"[私聊][{self.private_name}]风险预估异常: {exc}")
        final_risk = min(risk_val, 1.0)
        if final_risk > 0.2:
            logger.debug(
                f"[私聊][{self.private_name}]交互风险偏高: {final_risk:.2f}"
            )
        return final_risk

    def _scan_for_action_loops(self, conv: ConversationInfo) -> str:
        """扫描近期行动序列，检测是否陷入重复决策环路。

        使用高频模式检测：
        只做简单频率计数而非复杂模式匹配。
        """
        try:
            raw_log = getattr(conv, "done_action", None) or []
            window = raw_log[-self._recent_action_window:]
            if len(window) < self._repeated_action_ceiling:
                return ""
            freq_map: Dict[str, int] = {}
            for entry in window:
                if isinstance(entry, dict):
                    act = entry.get("action", "")
                elif isinstance(entry, (list, tuple)) and entry:
                    act = str(entry[0])
                else:
                    act = str(entry)
                if act:
                    freq_map[act] = freq_map.get(act, 0) + 1
            for act_label, occurrences in freq_map.items():
                if occurrences >= self._repeated_action_ceiling:
                    logger.info(
                        f"[私聊][{self.private_name}]行动环路警报: "
                        f"'{act_label}'在最近{self._recent_action_window}次中出现{occurrences}次"
                    )
                    return (
                        f"\n注意：你最近{self._recent_action_window}次决策中有{occurrences}次选择了'{act_label}'，"
                        f"这可能意味着你陷入了重复模式。请认真考虑是否需要采取不同的策略。\n"
                    )
        except Exception as exc:
            logger.warning(
                f"[私聊][{self.private_name}]行动环路扫描异常: {exc}"
            )
        return ""

    def _calibrate_by_context(
        self,
        action: str,
        reason: str,
        confidence: float,
        risk: float,
    ) -> Tuple[str, str]:
        """后置校准 —— 根据置信度和风险对LLM原始决策做安全调整。

        校准逻辑：
        - 高置信+低风险：信任原始决策，直接放行
        - 低置信：主动型行动降级为倾听
        - 高风险+低置信：攻击性行动降级为等待
        - 保守型行动（wait/listening/end等）：无需校准
        """
        # 保守型行动无论置信度如何都直接放行
        conservative_set = (
            "wait",
            "listening",
            "end_conversation",
            "block_and_ignore",
            "say_goodbye",
        )
        if action in conservative_set:
            return action, reason
        proactive_set = ("direct_reply", "send_new_message", "rethink_goal")
        # 置信度极低时，将主动行动降级为倾听
        if confidence < self._low_confidence_threshold and action in proactive_set:
            logger.info(
                f"[私聊][{self.private_name}]置信度过低({confidence:.2f})，"
                f"'{action}'降级为listening"
            )
            return (
                "listening",
                f"上下文信息不足(置信度={confidence:.2f})，暂时倾听。原计划: {action}({reason})",
            )
        # 高风险且置信度偏低时，切换为等待
        if (
            risk >= self._high_risk_threshold
            and confidence < self._confidence_guard_threshold
            and action in proactive_set
        ):
            logger.info(
                f"[私聊][{self.private_name}]高风险({risk:.2f})+低置信({confidence:.2f})，"
                f"'{action}'降级为wait"
            )
            return (
                "wait",
                f"交互风险偏高({risk:.2f})且信息不足，暂时等待。原计划: {action}({reason})",
            )
        # 当风险高且决策是 send_new_message 时，无论置信度如何都避免追问轰炸
        if (
            action == "send_new_message"
            and risk >= self._send_new_message_risk_threshold
        ):
            logger.info(
                f"[私聊][{
                    self.private_name}]send_new_message 在高风险({
                    risk:.2f})下改为wait"
            )
            return (
                "wait",
                f"检测到连续主动发言风险({risk:.2f})，本轮改为等待，避免消息轰炸。原计划: {reason}",
            )
        # fetch_knowledge在高风险时仍允许 —— 获取更多信息是降风险的正确策略
        return action, reason

    # ==================== LLM结果解析 ====================

    def _extract_decision_json(self, raw_text: str) -> Dict[str, str]:
        """从LLM返回文本中提取JSON格式的行动决策。

        策略1：正则匹配最外层JSON对象
        策略2：纯文本关键词回退识别
        """
        parsed_payload = parse_json_payload(raw_text, allow_array=True)
        if isinstance(parsed_payload, dict):
            action = str(parsed_payload.get("action", "") or "").strip()
            reason = str(
                parsed_payload.get("reason", "未提供原因") or "未提供原因"
            )
            if action:
                return {"action": action, "reason": reason}
        elif isinstance(parsed_payload, list):
            for item in parsed_payload:
                if isinstance(item, dict) and item.get("action"):
                    return {
                        "action": str(item.get("action", "")).strip(),
                        "reason": str(
                            item.get("reason", "未提供原因") or "未提供原因"
                        ),
                    }

        # 策略1：正则提取JSON
        try:
            match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
            if match:
                obj = json.loads(match.group())
                if "action" in obj:
                    return obj
                logger.warning(
                    f"[私聊][{self.private_name}]JSON对象中缺少action字段"
                )
        except json.JSONDecodeError as jde:
            logger.warning(f"[私聊][{self.private_name}]JSON解码失败: {jde}")
        except Exception as exc:
            logger.warning(f"[私聊][{self.private_name}]JSON提取异常: {exc}")
        # 策略2：纯文本关键词识别作为兜底
        lowered = raw_text.lower()
        for candidate in PERMITTED_ACTIONS:
            if candidate in lowered:
                logger.info(
                    f"[私聊][{self.private_name}]从文本中识别到行动关键词: {candidate}"
                )
                return {
                    "action": candidate,
                    "reason": "从非JSON输出中识别到行动关键词",
                }
        return {
            "action": "wait",
            "reason": "无法从LLM输出中解析到有效行动，默认等待",
        }

    def _extract_farewell_decision(self, raw_text: str) -> Dict[str, str]:
        """解析告别二次确认返回，兼容 say_bye 字段与文本兜底。"""
        parsed_payload = parse_json_payload(raw_text, allow_array=False)
        if isinstance(parsed_payload, dict):
            vote = (
                str(parsed_payload.get("say_bye", "no") or "no")
                .strip()
                .lower()
            )
            reason = str(
                parsed_payload.get("reason", "未提供原因") or "未提供原因"
            )
            return {"say_bye": vote, "reason": reason}

        lowered = (raw_text or "").lower()
        if "yes" in lowered or "告别" in lowered:
            return {"say_bye": "yes", "reason": "文本语义倾向发送告别"}
        return {"say_bye": "no", "reason": "文本语义倾向直接结束"}

    # ==================== 结束对话确认 ====================

    async def _deliberate_farewell(
        self,
        persona_text: str,
        dialogue_text: str,
        fallback_action: str,
        fallback_reason: str,
    ) -> Tuple[str, str]:
        """当初步决策为结束对话时，通过二次LLM判断是否需要发送告别消息。"""
        logger.info(
            f"[私聊][{self.private_name}]初步选择结束对话，进行告别决策确认"
        )
        try:
            bye_prompt = await global_prompt_manager.format_prompt(
                "pfc_end_decision",
                persona_text=persona_text,
                chat_history_text=dialogue_text,
            )
            bye_response, _ = await asyncio.wait_for(
                self.llm.generate_response_async(bye_prompt),
                timeout=self._farewell_timeout_seconds,
            )
            logger.debug(
                f"[私聊][{self.private_name}]告别确认返回(前200字): {bye_response[:200]}"
            )
            bye_parsed = self._extract_farewell_decision(bye_response)
            vote = str(bye_parsed.get("say_bye", "no")).strip().lower()
            bye_motive = bye_parsed.get("reason", fallback_reason)
            if vote == "yes":
                logger.info(
                    f"[私聊][{self.private_name}]确认发送告别消息。原因: {bye_motive}"
                )
                combined = f"决定发送告别语。决策原因: {bye_motive}  (初始结束理由: {
                    fallback_reason}) "
                return "say_goodbye", combined
            logger.info(
                f"[私聊][{self.private_name}]决定直接结束，不发告别。原因: {bye_motive}"
            )
            return fallback_action, fallback_reason
        except Exception as exc:
            logger.error(f"[私聊][{self.private_name}]告别确认流程异常: {exc}")
            logger.warning(
                f"[私聊][{self.private_name}]异常回退，执行原始结束决策"
            )
            return fallback_action, fallback_reason

    # ==================== 行动合法性 ====================

    def _enforce_action_whitelist(
        self, action: str, reason: str
    ) -> Tuple[str, str]:
        """确保最终行动属于允许集合，否则降级为wait。"""
        if action in PERMITTED_ACTIONS:
            logger.info(f"[私聊][{self.private_name}]最终行动: {action}")
            logger.info(f"[私聊][{self.private_name}]行动原因: {reason[:120]}")
            return action, reason
        logger.warning(
            f"[私聊][{self.private_name}]非法行动类型'{action}'，降级为wait"
        )
        return (
            "wait",
            f"行动类型'{action}'不在白名单中，已降级为wait。原始原因: {reason}",
        )

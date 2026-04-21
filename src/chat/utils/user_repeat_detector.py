import time
import hashlib
import re
import json
from typing import Dict, Optional, Tuple
from src.common.logger import get_logger

logger = get_logger("用户重复检测")


class UserRepeatDetector:
    """用户重复消息检测器 — LLM 动态判断模式

    核心设计：
    1. 记录用户消息历史
    2. 将消息历史和上下文交给 LLM 动态判断
    3. LLM 根据多维度因素决定是否触发"烦"的处理
    4. 不使用固定阈值，让模型自己判断
    """

    def __init__(self):
        """初始化检测器"""
        self._message_records: Dict[str, list] = {}

    def _get_key(self, user_id: str, stream_id: str) -> str:
        """生成用户-流唯一键"""
        return f"{user_id}_{stream_id}"

    @staticmethod
    def _text_similarity(text_a: str, text_b: str) -> float:
        """基于词集合 Jaccard 的相似度"""
        if not text_a or not text_b:
            return 0.0
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            tokens_a = set(text_a.lower())
            tokens_b = set(text_b.lower())
        overlap = len(tokens_a & tokens_b)
        total = len(tokens_a | tokens_b)
        return overlap / total if total else 0.0

    def _add_record(
        self, user_id: str, stream_id: str, message_text: str
    ) -> None:
        """添加一条消息记录"""
        key = self._get_key(user_id, stream_id)

        if key not in self._message_records:
            self._message_records[key] = []

        current_time = time.time()

        record = {
            "user_id": user_id,
            "stream_id": stream_id,
            "message_text": message_text,
            "message_hash": hashlib.md5(
                message_text[:60].encode()
            ).hexdigest()[:12],
            "excerpt": message_text[:60],
            "timestamp": current_time,
        }

        self._message_records[key].append(record)

        if len(self._message_records[key]) > 20:
            self._message_records[key] = self._message_records[key][-20:]

    def _build_history_for_llm(
        self, user_id: str, stream_id: str, current_message: str
    ) -> str:
        """构建用于 LLM 判断的历史消息"""
        key = self._get_key(user_id, stream_id)

        if key not in self._message_records:
            return ""

        records = self._message_records[key]
        if len(records) < 2:
            return ""

        recent_records = sorted(
            records, key=lambda r: r["timestamp"], reverse=True
        )[:10]

        history_lines = []
        for i, record in enumerate(recent_records):
            time_ago = time.time() - record["timestamp"]
            history_lines.append(
                f"{i + 1}. {record['excerpt']} ({time_ago:.0f}秒前)"
            )

        return "\n".join(history_lines)

    async def detect_and_update_annoyance(
        self,
        user_id: str,
        stream_id: str,
        message_text: str,
        auto_update: bool = True,
    ) -> Tuple[bool, float, Dict]:
        """检测重复并让 LLM 动态判断是否触发"""

        # 先记录消息
        self._add_record(user_id, stream_id, message_text)

        # 获取历史
        history = self._build_history_for_llm(user_id, stream_id, message_text)

        if not history:
            return False, 0.0, {"triggered": False, "reason": "历史记录不足"}

        # ── 快速预检：和最近3条比相似度，全部低则直接跳过 ──
        key = self._get_key(user_id, stream_id)
        records = self._message_records.get(key, [])
        if len(records) >= 2:
            recent = records[-3:]
            max_sim = 0.0
            for r in recent[:-1]:
                sim = self._text_similarity(message_text, r["excerpt"])
                if sim > max_sim:
                    max_sim = sim
            if max_sim < 0.25:
                return False, 0.0, {
                    "triggered": False,
                    "reason": "内容不相似",
                    "level": "none",
                }

        # 获取情绪状态
        emotion_summary = ""
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            tracker = get_emotion_tracker(stream_id)
            if tracker:
                state = tracker.get_user_state(user_id)
                if state:
                    emotion_summary = f"当前厌烦度={
                        state.annoyance:.1f}/100，好感度={
                        state.affection:.1f}/100"
        except Exception as _e:
            logger.debug(f"异常: {_e}")

        if not emotion_summary:
            emotion_summary = "情绪状态平稳"

        # 构建判断提示
        judgment_prompt = f"""你是情绪判断助手。用户发送了以下消息：

当前消息：{message_text}

最近的消息历史：
{history}

{emotion_summary}

请严格判断用户是否在**重复纠缠同一个话题或同一句话**。

判断标准（必须同时满足才判为重复）：
1. 当前消息与历史中某条消息**内容高度相似**（意思基本一样，不是泛泛的"都在聊天"）
2. 用户在短时间内（5分钟内）发了**3条以上**类似内容
3. 不是正常的连续对话（比如你一句我一句的正常聊天不算重复）

特别注意以下情况**不算重复**：
- 用户每次发的**内容/话题完全不同**（即使发得频繁也不算重复）
- 正常的一问一答对话
- 聊天中自然的话题切换

请用JSON格式返回你的判断：
{{
    "is_repeating": true/false,
    "trigger_reason": "具体原因（指出哪条历史消息与当前消息相似）",
    "repetition_level": "high/medium/low/none"
}}"""

        # 让 LLM 判断
        is_triggered = False
        trigger_reason = ""
        repetition_level = "none"

        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            request = LLMRequest(
                model_config.model_task_config.lightweight, request_type="repeat_judge"
            )
            response_text, _ = await request.generate_response_async(
                judgment_prompt, max_tokens=300
            )

            if response_text:
                json_match = re.search(r"\{[^}]+\}", response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    is_triggered = result.get("is_repeating", False)
                    trigger_reason = result.get("trigger_reason", "")
                    repetition_level = result.get("repetition_level", "none")

                    if is_triggered:
                        logger.info(
                            f"[重复检测] LLM判断用户{user_id[:8]}...重复，"
                            f"等级={repetition_level}，原因={trigger_reason}"
                        )
        except Exception as e:
            logger.debug(f"[重复检测] LLM判断失败：{e}")
            # 如果 LLM 判断失败，使用简单的回退逻辑
            key = self._get_key(user_id, stream_id)
            if (
                key in self._message_records
                and len(self._message_records[key]) >= 3
            ):
                recent = self._message_records[key][-3:]
                if len(recent) >= 2:
                    sim = self._text_similarity(
                        recent[-1]["excerpt"], recent[-2]["excerpt"]
                    )
                    if sim >= 0.85:
                        is_triggered = True
                        trigger_reason = "相似度过高"
                        repetition_level = "medium"

        # 如果触发，调用结算引擎
        result_state = {
            "triggered": is_triggered,
            "reason": trigger_reason,
            "level": repetition_level,
            "settlement_report": None,
        }

        if auto_update and is_triggered:
            try:
                from src.chat.heart_flow.social_value_dim import SocialValueDimension

                engine = SocialValueDimension.get_instance()

                severity_map = {"high": 0.8, "medium": 0.5, "low": 0.3}
                severity = severity_map.get(repetition_level, 0.5)

                behavior = {
                    "behavior_type": "negative_repeat",
                    "severity": severity,
                    "intent": "boredom",
                }

                context = {
                    "psychological_pressure": 0.0,
                    "training_resistance": 0.0,
                    "trigger": "llm_judged_repeat",
                }

                report = await engine.settle_from_behavior(
                    user_id=user_id,
                    channel_id=stream_id,
                    behavior=behavior,
                    context=context,
                )

                result_state["settlement_report"] = {
                    "raw_delta": report.raw_delta,
                    "final_delta": report.final_delta,
                    "settled_score": report.settled_score,
                }

                logger.info(
                    f"[重复检测] 结算完成 uid={user_id[:8]} 变化={report.final_delta:.2f} "
                    f"最终分={report.settled_score:.1f}"
                )
            except Exception as e:
                logger.debug(f"[重复检测] 结算失败：{e}")

        return is_triggered, 0.0, result_state

    def get_repeat_info(self, user_id: str, stream_id: str) -> Dict:
        """获取用户的重复消息信息"""
        key = self._get_key(user_id, stream_id)
        current_time = time.time()

        if key not in self._message_records:
            return {
                "record_count": 0,
                "recent_messages": [],
                "is_triggered": False,
            }

        records = self._message_records[key]
        recent_messages = [
            {
                "text": (
                    record["excerpt"][:30] + "..."
                    if len(record["excerpt"]) > 30
                    else record["excerpt"]
                ),
                "timestamp": record["timestamp"],
                "time_ago": current_time - record["timestamp"],
            }
            for record in sorted(
                records, key=lambda r: r["timestamp"], reverse=True
            )[:10]
        ]

        return {
            "record_count": len(records),
            "recent_messages": recent_messages,
            "is_triggered": False,
        }

    def get_emotion_prompt(self, user_id: str, stream_id: str) -> str:
        """生成情绪提示词"""
        return ""

    def get_repeat_warning(
        self, user_id: str, stream_id: str
    ) -> Optional[str]:
        """获取重复警告信息"""
        return None

    def reset_user_records(self, user_id: str, stream_id: str) -> None:
        """重置用户的重复记录"""
        key = self._get_key(user_id, stream_id)
        if key in self._message_records:
            del self._message_records[key]
        logger.info(f"[重复检测] 已重置用户 {user_id[:8]}...的重复记录")


_global_detector: Optional[UserRepeatDetector] = None


def get_user_repeat_detector() -> UserRepeatDetector:
    """获取全局的用户重复检测器单例"""
    global _global_detector
    if _global_detector is None:
        _global_detector = UserRepeatDetector()
    return _global_detector

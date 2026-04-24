import time
from typing import Dict, Any, Optional
from src.common.logger import get_logger
from src.modules.social_value.models import (
    SocialUpdateResult,
)
from src.modules.social_value.social_calculator import SocialCalculator
from src.modules.social_value.runtime_config import social_value_module_view
from src.modules.social_value.social_storage import SocialStorage

logger = get_logger("社交价值核心")


class SocialValueCore:
    """社交值核心

    社交值系统主入口，管理用户社交关系值 (-100 ~ +100)。
    流程：模型理解（定性）→ 算法计算（定量）→ 衰减/阻尼/单次上限 → 关系调制（新用户宽容/创伤敏感）→ 持久化。
    """

    def __init__(
        self,
        calculator: SocialCalculator,
        storage: SocialStorage,
        config_engine: Optional[object] = None,
    ):
        self._calculator = calculator
        self._storage = storage
        self._runtime_cfg = self._load_runtime_config()

    def _load_runtime_config(self) -> Dict[str, float]:
        cfg = {
            "social_decay_per_hour": 0.015,
            "social_damping_polarization_weight": 0.5,
            "social_damping_interaction_weight": 0.15,
            "social_single_step_min": 0.5,
            "social_single_step_max": 2.0,
            "social_new_user_threshold": 10.0,
            "social_new_user_factor": 0.6,
            "social_trauma_sensitive_threshold": 5.0,
            "social_trauma_sensitive_boost": 0.25,
            "social_new_user_initial": 5.0,
        }
        module_cfg = social_value_module_view("social_value_core")
        if isinstance(module_cfg, dict):
            for key in cfg.keys():
                if key in module_cfg:
                    try:
                        cfg[key] = float(module_cfg[key])
                    except Exception as _exc:
                        logger.debug(f"非关键异常: {_exc}")
        return cfg

    def _cfg(self, key: str, ctx: Dict[str, Any], default: float) -> float:
        if key in ctx:
            try:
                return float(ctx[key])
            except Exception as _e:
                logger.debug(f"[社交价值] 异常: {_e}")
        return float(self._runtime_cfg.get(key, default))

    async def get_value(self, user_id: str, channel_id: str) -> float:
        record = await self._storage.get(user_id, channel_id)
        return record.value if record else 0.0

    def get_social_value(self, user_id: str, channel_id: str) -> float:
        """同步读取社交值（供同步上下文如_compute_relation_metrics调用）"""
        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                record = self._storage.get_sync(user_id, channel_id)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    record = loop.run_until_complete(
                        self._storage.get(user_id, channel_id)
                    )
                finally:
                    loop.close()
            return record.value if record else 0.0
        except Exception as exc:
            logger.warning(f"社交值同步读取失败 user={user_id} ch={channel_id}: {exc}")
            return 0.0

    def get_social_record_sync(self, user_id: str, channel_id: str):
        """同步读取完整记录"""
        try:
            import asyncio

            try:
                asyncio.get_running_loop()
                return self._storage.get_sync(user_id, channel_id)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(
                        self._storage.get(user_id, channel_id)
                    )
                finally:
                    loop.close()
        except Exception as exc:
            logger.warning(f"社交记录同步读取失败 user={user_id} ch={channel_id}: {exc}")
            return None

    async def get_full_record(self, user_id: str, channel_id: str):
        return await self._storage.get(user_id, channel_id)

    async def update(
        self,
        user_id: str,
        channel_id: str,
        content: str,
        context: Dict[str, Any],
    ) -> SocialUpdateResult:
        """通过内容分析更新社交值"""
        external_behavior = (
            context.get("behavior_signal")
            if isinstance(context, dict)
            else None
        )
        if isinstance(external_behavior, dict) and external_behavior.get(
            "behavior_type"
        ):
            behavior = {
                "behavior_type": str(
                    external_behavior.get("behavior_type", "casual_chat")
                    or "casual_chat"
                ),
                "severity": float(
                    external_behavior.get("severity", 0.5) or 0.5
                ),
                "intent": str(
                    external_behavior.get("intent", "chat") or "chat"
                ),
            }
        else:
            behavior = await self._analyze_behavior(content, context)
        return await self.update_from_result(
            user_id, channel_id, behavior, context
        )

    async def update_from_result(
        self,
        user_id: str,
        channel_id: str,
        behavior: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> SocialUpdateResult:
        """根据行为分析结果更新社交值"""
        ctx = context or {}
        record = await self._storage.get(user_id, channel_id)
        is_new_user = record is None
        if is_new_user:
            initial_value = self._cfg("social_new_user_initial", ctx, 5.0)
            old_value = initial_value
            interaction_count = 0
            logger.info(
                f"[社交初始化] 新用户 {user_id[:8]} 初始社交值={initial_value:.1f}"
            )
        else:
            old_value = record.value
            interaction_count = int(record.interaction_count)
        hours_elapsed = 0.0
        if record and record.last_interaction > 0:
            hours_elapsed = max(
                0.0, (time.time() - float(record.last_interaction)) / 3600.0
            )
        decayed_value = self._apply_time_decay(old_value, hours_elapsed, ctx)
        raw_delta = self._calculator.calculate(
            behavior_type=(
                behavior.behavior_type
                if hasattr(behavior, "behavior_type")
                else behavior.get("behavior_type", "neutral")
            ),
            severity=(
                behavior.severity
                if hasattr(behavior, "severity")
                else behavior.get("severity", 0.5)
            ),
            intent=(
                behavior.intent
                if hasattr(behavior, "intent")
                else behavior.get("intent", "other")
            ),
            current_value=decayed_value,
            psychological_pressure=ctx.get("psychological_pressure", 0.0),
            training_resistance=ctx.get("training_resistance", 0.0),
            uid=user_id,
        )
        damped_delta = self._apply_damping(
            raw_delta, decayed_value, interaction_count, ctx
        )
        bounded_delta = self._apply_single_step_cap(
            damped_delta,
            (
                behavior.severity
                if hasattr(behavior, "severity")
                else behavior.get("severity", 0.5)
            ),
            ctx,
        )
        adjusted_delta = self._apply_relation_modifiers(
            bounded_delta, interaction_count, ctx
        )
        new_value = max(-100.0, min(100.0, decayed_value + adjusted_delta))
        await self._storage.set(
            user_id=user_id, channel_id=channel_id, value=new_value
        )
        category = self._calculator.get_category_params(
            (
                behavior.behavior_type
                if hasattr(behavior, "behavior_type")
                else behavior.get("behavior_type", "neutral")
            ),
            (
                behavior.intent
                if hasattr(behavior, "intent")
                else behavior.get("intent", "other")
            ),
        ).category
        actual_old = 0.0 if is_new_user else old_value
        return SocialUpdateResult(
            old_value=actual_old,
            new_value=new_value,
            delta=new_value - actual_old,
            behavior_type=(
                behavior.behavior_type
                if hasattr(behavior, "behavior_type")
                else behavior.get("behavior_type", "unknown")
            ),
            intent=(
                behavior.intent
                if hasattr(behavior, "intent")
                else behavior.get("intent", "other")
            ),
            severity=(
                behavior.severity
                if hasattr(behavior, "severity")
                else behavior.get("severity", 0.5)
            ),
            category=category,
            params_used={
                "hours_elapsed": round(hours_elapsed, 4),
                "decayed_value": round(decayed_value, 4),
                "raw_delta": round(raw_delta, 4),
                "damped_delta": round(damped_delta, 4),
                "bounded_delta": round(bounded_delta, 4),
                "adjusted_delta": round(adjusted_delta, 4),
                "is_new_user": is_new_user,
            },
        )

    async def apply_decay(
        self, user_id: str, channel_id: str, hours_elapsed: float
    ) -> float:
        """应用时间衰减"""
        record = await self._storage.get(user_id, channel_id)
        if record is None:
            return 0.0
        new_value = self._apply_time_decay(
            record.value, max(0.0, hours_elapsed), {}
        )
        await self._storage.set(user_id, channel_id, new_value)
        return new_value

    def _apply_time_decay(
        self, current_value: float, hours_elapsed: float, ctx: Dict[str, Any]
    ) -> float:
        """时间衰减"""
        if abs(current_value) < 1e-6 or hours_elapsed <= 0:
            return current_value
        decay_per_hour = self._cfg("social_decay_per_hour", ctx, 0.015)
        decay_factor = max(0.0, min(0.95, decay_per_hour * hours_elapsed))
        return current_value * (1.0 - decay_factor)

    def _apply_damping(
        self,
        delta: float,
        current_value: float,
        interaction_count: int,
        ctx: Dict[str, Any],
    ) -> float:
        """阻尼处理"""
        if abs(delta) < 1e-6:
            return delta
        polarization_weight = self._cfg(
            "social_damping_polarization_weight", ctx, 0.5
        )
        interaction_weight = self._cfg(
            "social_damping_interaction_weight", ctx, 0.15
        )
        polarization_ratio = min(1.0, abs(current_value) / 100.0)
        interaction_ratio = min(1.0, max(0, interaction_count) / 200.0)
        damping = 1.0 - (
            polarization_ratio * polarization_weight
            + interaction_ratio * interaction_weight
        )
        damping = max(0.25, min(1.0, damping))
        return delta * damping

    def _apply_single_step_cap(
        self, delta: float, severity: float, ctx: Dict[str, Any]
    ) -> float:
        """单次变化上限"""
        min_step = self._cfg("social_single_step_min", ctx, 0.5)
        max_step = self._cfg("social_single_step_max", ctx, 2.0)
        if max_step < min_step:
            min_step, max_step = max_step, min_step
        s = min(1.0, max(0.0, float(severity)))
        cap = min_step + (max_step - min_step) * s
        return max(-cap, min(cap, delta))

    def _apply_relation_modifiers(
        self, delta: float, interaction_count: int, ctx: Dict[str, Any]
    ) -> float:
        """关系调制"""
        if abs(delta) < 1e-6:
            return delta
        threshold = self._cfg("social_new_user_threshold", ctx, 10.0)
        new_user_factor = self._cfg("social_new_user_factor", ctx, 0.6)
        trauma_threshold = self._cfg(
            "social_trauma_sensitive_threshold", ctx, 5.0
        )
        trauma_boost = self._cfg("social_trauma_sensitive_boost", ctx, 0.25)
        trauma_score = max(0.0, float(ctx.get("trauma_score", 0.0)))
        adjusted = delta
        if interaction_count < threshold:
            adjusted *= new_user_factor
        if trauma_score >= trauma_threshold and adjusted < 0:
            adjusted *= 1.0 + trauma_boost
        return adjusted

    async def _analyze_behavior(
        self, content: str, context: Dict[str, Any]
    ) -> Any:
        """分析行为类型 - 基于文本内容的规则分析"""
        text = content.lower() if content else ""
        behavior_type = "casual_chat"
        severity = 0.5
        intent = "chat"
        positive_markers = [
            "谢谢",
            "感谢",
            "太棒了",
            "厉害",
            "牛",
            "赞",
            "喜欢",
            "爱",
            "可爱",
            "哈哈",
            "好",
            "不错",
            "可以",
            "真棒",
            "太好了",
            "开心",
            "高兴",
            "优秀",
            "完美",
            "厉害了",
            "牛逼",
            "棒极了",
            "太强了",
            "真厉害",
            "谢谢啦",
            "多谢",
            "辛苦了",
            "麻烦你了",
            "帮大忙了",
            "太感谢了",
        ]
        negative_markers = [
            "滚",
            "闭嘴",
            "烦人",
            "讨厌",
            "恶心",
            "垃圾",
            "废物",
            "傻",
            "笨",
            "蠢",
            "滚开",
            "走开",
            "别烦我",
            "烦死了",
            "真烦",
            "无聊",
            "没意思",
            "差劲",
            "太烂了",
            "什么破",
            "垃圾啊",
            "别说了",
            "别讲了",
            "闭嘴吧",
            "安静点",
            "吵死了",
        ]
        insult_markers = [
            "傻逼",
            "操你",
            "妈的",
            "草泥马",
            "去死",
            "滚蛋",
            "混蛋",
            "王八蛋",
            "畜生",
            "狗",
            "猪",
            "白痴",
            "弱智",
            "脑残",
        ]
        praise_markers = [
            "你真棒",
            "你太厉害了",
            "你是最棒的",
            "你真聪明",
            "你真厉害",
            "我崇拜你",
            "你太牛了",
            "你真优秀",
            "你真厉害啊",
        ]
        question_markers = [
            "？",
            "?",
            "吗",
            "呢",
            "怎么",
            "什么",
            "为什么",
            "哪里",
            "谁",
            "如何",
            "怎样",
            "多少",
            "几",
            "是不是",
            "对不对",
            "好不好",
        ]
        greeting_markers = [
            "你好",
            "早上好",
            "晚上好",
            "下午好",
            "嗨",
            "哈喽",
            "在吗",
            "在不在",
            "有人吗",
            "hello",
            "hi",
            "早啊",
            "晚安",
        ]
        positive_count = sum(1 for m in positive_markers if m in text)
        negative_count = sum(1 for m in negative_markers if m in text)
        insult_count = sum(1 for m in insult_markers if m in text)
        praise_count = sum(1 for m in praise_markers if m in text)
        question_count = sum(1 for m in question_markers if m in text)
        greeting_count = sum(1 for m in greeting_markers if m in text)
        if insult_count > 0:
            behavior_type = "direct_insult"
            severity = min(1.0, 0.7 + insult_count * 0.1)
            intent = "insult"
        elif negative_count > positive_count + 2:
            behavior_type = "hostile_remark"
            severity = min(0.9, 0.5 + negative_count * 0.05)
            intent = "hostile"
        elif praise_count > 0:
            behavior_type = "genuine_praise"
            severity = min(1.0, 0.6 + praise_count * 0.15)
            intent = "compliment"
        elif positive_count > negative_count + 1:
            behavior_type = "friendly_chat"
            severity = min(0.8, 0.4 + positive_count * 0.05)
            intent = "friendly"
        elif question_count > 0 and len(text) < 50:
            behavior_type = "casual_inquiry"
            severity = 0.3
            intent = "inquiry"
        elif greeting_count > 0:
            behavior_type = "greeting"
            severity = 0.2
            intent = "greeting"
        elif question_count > 0:
            behavior_type = "casual_chat"
            severity = 0.4
            intent = "chat"
        interaction_type = context.get("interaction_type", "")
        if interaction_type == "bot_reply":
            if behavior_type == "casual_chat":
                behavior_type = "friendly_chat"
                severity = 0.3
                intent = "friendly"
        current_social = context.get("current_social", 0.0)
        if current_social < -30 and behavior_type in [
            "casual_chat",
            "friendly_chat",
        ]:
            severity *= 0.7
        elif current_social > 30 and behavior_type in [
            "casual_chat",
            "friendly_chat",
        ]:
            severity *= 1.2
        return {
            "behavior_type": behavior_type,
            "severity": round(severity, 2),
            "intent": intent,
        }

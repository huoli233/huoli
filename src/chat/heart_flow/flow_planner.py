import asyncio
import time as _tm
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import src.chat.prompts.catalog  # noqa: F401
from src.common.logger import get_logger
from src.chat.utils.prompt_builder import global_prompt_manager

logger = get_logger("流程规划器")


def _runtime_float(name: str, default: float) -> float:
    try:
        from src.config.core_config_engine import get_core_config

        return float(get_core_config().resolve_module_view("runtime_tuning").values.get(name, default))
    except Exception:
        return default


def _task_provider_timeout(task_cfg) -> float:
    try:
        from src.config.config import model_config

        provider_timeouts: list[float] = []
        for model_name in getattr(task_cfg, "model_list", []) or []:
            model_info = model_config.get_model_info(model_name)
            provider = model_config.get_provider(model_info.api_provider)
            provider_timeouts.append(float(getattr(provider, "timeout", 10) or 10))
        if provider_timeouts:
            return max(provider_timeouts)
    except Exception:
        pass
    return 10.0


@dataclass
class _CalibrationSnapshot:
    """flow_planner 侧保存的最近一次校准结果"""

    offline_hours: float = 0.0
    calibration_level: str = "无"
    intimacy_after: float = 50.0
    emotion_after: float = 50.0
    forget_prob: float = 0.0
    updated_at: float = 0.0

    def is_recent(self, max_age_sec: float = 600.0) -> bool:
        """10分钟内的校准视为有效"""
        return self.updated_at > 0.0 and (_tm.time() - self.updated_at) < max_age_sec

    @property
    def proactive_penalty(self) -> float:
        """离线回来后主动发言应更保守，返回一个 0~1 的惩罚系数"""
        if self.calibration_level == "重度":
            return 0.7
        if self.calibration_level == "中度":
            return 0.3
        return 0.0


class SituationKind(Enum):
    """对话情境大类"""

    CASUAL = "闲聊"
    QUESTION = "提问"
    EMOTIONAL = "情感倾诉"
    REQUEST = "请求帮助"
    GROUP_TOPIC = "群聊热议"
    GREETING = "寒暄"
    FAREWELL = "告别"
    OFFENSIVE = "攻击性"
    UNKNOWN = "未知"


@dataclass
class PlannerOutput:
    """规划器输出"""

    situation: SituationKind = SituationKind.UNKNOWN
    reply_strategy: str = ""
    emotion_hint: str = ""
    should_recall_memory: bool = False
    suggested_length: int = 100
    extra_notes: str = ""
    # 投递策略：standalone=独立发言 / quote=引用某条消息 / mention=@某人
    delivery_form: str = "standalone"
    # 需要@提到的用户昵称（仅 delivery_form 含 mention 时有效）
    mention_user_name: str = ""
    # 想回应的那个人的昵称（用于定位引用目标）
    reference_user_name: str = ""


def _quick_classify(text: str) -> Optional[SituationKind]:
    """仅基于结构信号做极轻量预判，避免词表驱动分流"""
    content = (text or "").strip()
    if not content:
        return None
    lowered = content.lower()
    question_marks = content.count("?") + content.count("？")
    exclamations = content.count("!") + content.count("！")
    short_text = len(content) <= 12
    line_breaks = content.count("\n")
    uppercase_ratio = sum(1 for ch in lowered if ch.isalpha() and ch.isupper()) / max(
        1, sum(1 for ch in lowered if ch.isalpha())
    )

    if short_text and question_marks == 0 and exclamations <= 1:
        return SituationKind.GREETING
    if question_marks > 0:
        return SituationKind.QUESTION
    if exclamations >= 3 or uppercase_ratio > 0.7:
        return SituationKind.OFFENSIVE
    if line_breaks >= 2 and len(content) > 60:
        return SituationKind.EMOTIONAL
    return None


class FlowPlanner:
    """心流规划器

    两种路径:
      generate_reactive_plan  — 被动回复规划
      generate_proactive_plan — 主动发言规划
    """

    def __init__(self):
        self._invoke_lock: asyncio.Lock | None = None
        self._cal_snapshot = _CalibrationSnapshot()
        self._timeout_seconds = 15.0
        self._slow_threshold = 10.0

    def _resolve_timeout_budget(self) -> tuple[float, float]:
        try:
            from src.config.config import model_config

            task_cfg = model_config.model_task_config.lightweight
            provider_timeout = _task_provider_timeout(task_cfg)
            default_timeout = max(20.0, provider_timeout + 10.0)
            timeout_seconds = _runtime_float(
                "flow_planner_timeout_seconds",
                default_timeout,
            )
            slow_threshold = _runtime_float(
                "flow_planner_slow_threshold_seconds",
                max(10.0, min(timeout_seconds - 2.0, timeout_seconds * 0.7)),
            )
            return timeout_seconds, slow_threshold
        except Exception:
            return self._timeout_seconds, self._slow_threshold

    def apply_calibration(self, summary) -> None:
        """由 system_calibration 回调注入校准结果。"""
        hours = getattr(summary, "offline_seconds", 0.0) / 3600.0
        self._cal_snapshot = _CalibrationSnapshot(
            offline_hours=hours,
            calibration_level=getattr(summary, "level", "无"),
            intimacy_after=50.0 + getattr(summary, "intimacy_delta", 0.0),
            emotion_after=50.0 + getattr(summary, "emotion_delta", 0.0),
            forget_prob=getattr(summary, "memory_forget_probability", 0.0),
            updated_at=_tm.time(),
        )
        logger.info(
            f"[流程规划] 收到校准快照: "
            f"离线{hours:.1f}h 级别={self._cal_snapshot.calibration_level} "
            f"亲密度={self._cal_snapshot.intimacy_after:.1f} "
            f"情绪={self._cal_snapshot.emotion_after:.1f}"
        )

    async def generate_reactive_plan(
        self,
        channel_id: str,
        recent_messages: str,
        *,
        extra_context: str = "",
    ) -> PlannerOutput:
        """根据最近消息生成被动回复策略"""
        quick_kind = _quick_classify(recent_messages[-200:] if recent_messages else "")
        if quick_kind == SituationKind.GREETING:
            return PlannerOutput(
                situation=quick_kind,
                reply_strategy="礼貌回应",
                emotion_hint="温和",
                suggested_length=30,
            )
        return await self._llm_plan(channel_id, recent_messages, quick_kind)

    async def _llm_plan(
        self,
        channel_id: str,
        messages: str,
        quick_kind: Optional[SituationKind],
    ) -> PlannerOutput:
        """调用LLM规划"""
        from datetime import datetime

        prompt = await global_prompt_manager.format_prompt(
            "flow_planner_plan",
            time=datetime.now().strftime("%H:%M"),
            channel=channel_id,
            quick_kind=quick_kind.value if quick_kind else "未知",
            messages=messages[:2500],
        )
        raw = await self._invoke_model(prompt, channel_id)
        return self._parse_plan(raw)

    async def generate_proactive_plan(
        self,
        channel_id: str,
        quiet_seconds: float,
        recent_topic: str = "",
        mood: str = "平静",
        recent_bot_context: str = "",
        recent_dialogue: str = "",
        atmosphere_context: str = "",
        relationship_context: str = "",
        self_state_context: str = "",
    ) -> Optional[PlannerOutput]:
        """判断是否适合主动发言并生成策略"""
        snap = self._cal_snapshot
        if snap.is_recent() and snap.intimacy_after < 30.0:
            logger.info(f"[流程规划] {channel_id} 亲密度过低({snap.intimacy_after:.1f})，拒绝主动发言")
            return None

        calibration_hint = ""
        if snap.is_recent() and snap.calibration_level != "无":
            calibration_hint = (
                f"机器人刚刚离线了 {snap.offline_hours:.1f} 小时，"
                f"当前亲密度 {snap.intimacy_after:.0f}，"
                f"情绪指数 {snap.emotion_after:.0f}。\n"
            )
        if recent_bot_context:
            calibration_hint += f"\n{recent_bot_context}\n"

        # 构建近期对话文本，让规划器知道群里谁说了什么
        _dialogue_block = recent_dialogue.strip() if recent_dialogue else "（最近没有人说话）"
        # 构建社交融入上下文（氛围 + 人际 + 自身状态）
        _social_block_parts = []
        if atmosphere_context:
            _social_block_parts.append(atmosphere_context)
        if relationship_context:
            _social_block_parts.append(relationship_context)
        if self_state_context:
            _social_block_parts.append(self_state_context)
        _social_block = "\n".join(_social_block_parts) if _social_block_parts else ""

        prompt = await global_prompt_manager.format_prompt(
            "flow_planner_proactive",
            quiet_sec=quiet_seconds,
            channel=channel_id,
            topic=recent_topic or "无",
            mood=mood,
            calibration_hint=calibration_hint,
            recent_dialogue=_dialogue_block,
            social_context=_social_block,
        )
        raw = await self._invoke_model(prompt, channel_id)
        return self._parse_proactive(raw)

    async def _invoke_model(self, prompt: str, channel_id: str = "") -> str:
        """调用轻量模型 - 带超时和日志"""
        if self._invoke_lock is None:
            self._invoke_lock = asyncio.Lock()
        async with self._invoke_lock:
            start_time = _tm.time()
            model_name = "未知"
            try:
                from src.llm_models.utils_model import LLMRequest
                from src.config.config import model_config

                req = LLMRequest(
                    model_set=model_config.model_task_config.lightweight,
                    request_type="flow_planner",
                )
                _task_cfg = model_config.model_task_config.lightweight
                model_name = getattr(_task_cfg, "model_list", ["未知"])
                if isinstance(model_name, list):
                    model_name = model_name[0] if model_name else "未知"
                logger.info(f"[规划器] [{channel_id}] 开始规划，模型: {model_name}")
                timeout_seconds, slow_threshold = self._resolve_timeout_budget()
                try:
                    result = await asyncio.wait_for(
                        req.generate_response_async(prompt, temperature=0.3),
                        timeout=timeout_seconds,
                    )
                    content = result[0] if isinstance(result, tuple) else result
                except asyncio.TimeoutError:
                    elapsed = _tm.time() - start_time
                    logger.warning(f"[规划器] [{channel_id}] 超时({timeout_seconds}s)，强制返回空")
                    return ""
                elapsed = _tm.time() - start_time
                if elapsed > slow_threshold:
                    logger.warning(
                        f"[规划器] [{channel_id}] 规划耗时过长: {elapsed:.1f}s "
                        f"(模型: {model_name}, 阈值: {slow_threshold}s)"
                    )
                else:
                    logger.info(f"[规划器] [{channel_id}] 规划完成，耗时: {elapsed:.1f}s")
                return content.strip() if content else ""
            except Exception as exc:
                elapsed = _tm.time() - start_time
                logger.error(f"[规划器] [{channel_id}] LLM调用异常({elapsed:.1f}s): {exc}")
                return ""

    @staticmethod
    def _extract_json(raw: str) -> Optional[dict]:
        """从LLM文本中提取JSON对象"""
        import json as _json

        if not raw:
            return None
        left = raw.find("{")
        right = raw.rfind("}") + 1
        if left < 0 or right <= left:
            return None
        try:
            return _json.loads(raw[left:right])
        except (ValueError, KeyError):
            return None

    def _parse_plan(self, raw: str) -> PlannerOutput:
        """解析被动规划结果"""
        obj = self._extract_json(raw)
        if not obj:
            return PlannerOutput(reply_strategy="通用回复", emotion_hint="中性")
        kind_map = {k.value: k for k in SituationKind}
        sit_str = obj.get("situation", "未知")
        return PlannerOutput(
            situation=kind_map.get(sit_str, SituationKind.UNKNOWN),
            reply_strategy=str(obj.get("strategy", "")),
            emotion_hint=str(obj.get("emotion_hint", "")),
            should_recall_memory=bool(obj.get("recall_memory", False)),
            suggested_length=int(obj.get("suggested_length", 100)),
        )

    def _parse_proactive(self, raw: str) -> Optional[PlannerOutput]:
        """解析主动规划结果（含投递策略）"""
        obj = self._extract_json(raw)
        if not obj:
            return None
        if not obj.get("should_speak", False):
            return None
        # 解析投递策略
        _delivery_raw = str(obj.get("delivery", "standalone") or "standalone").lower().strip()
        _valid_deliveries = ("standalone", "quote", "mention", "quote+mention")
        _delivery = _delivery_raw if _delivery_raw in _valid_deliveries else "standalone"
        _reference_user = str(obj.get("reference_user", "") or "").strip()
        _mention_user = str(obj.get("mention_user", "") or "").strip()
        return PlannerOutput(
            situation=SituationKind.CASUAL,
            reply_strategy=str(obj.get("opening", "")),
            emotion_hint=str(obj.get("emotion_hint", "平静")),
            extra_notes=str(obj.get("topic", "")),
            suggested_length=80,
            delivery_form=_delivery,
            mention_user_name=_mention_user,
            reference_user_name=_reference_user,
        )

    def classify_situation(self, text: str) -> SituationKind:
        """纯结构分类（不调用LLM）"""
        kind = _quick_classify(text)
        return kind if kind else SituationKind.UNKNOWN


_planner_ref: Optional[FlowPlanner] = None


def acquire_flow_planner() -> FlowPlanner:
    global _planner_ref
    if _planner_ref is None:
        _planner_ref = FlowPlanner()
    return _planner_ref

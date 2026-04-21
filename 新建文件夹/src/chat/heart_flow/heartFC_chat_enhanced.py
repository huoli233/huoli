import asyncio
import math
import random
import re
import time
import traceback
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple
from src.chat.utils.timer_calculator import Timer
from src.chat.replyer.context_block_builder import build_reply_context_block
from src.chat.heart_flow.heartFC_chat import HeartFChatting
from src.config.config import global_config
from src.common.logger import get_logger
from src.chat.heart_flow.hfc_utils import CycleDetail
from src.chat.heart_flow.frequency_control import frequency_control_manager
from src.plugin_system.apis import database_api, message_api

if TYPE_CHECKING:
    from src.chat.heart_flow.llm_autonomous_planner import (
        AutonomousDecision,
        EnvironmentSnapshot,
    )
    from src.chat.proactive.proactive_arbiter import ArbiterVerdict
    from src.chat.proactive.proactive_integration_hub import IntegratedState

logger = get_logger("心流增强")


def _rt_float(key: str, default: float) -> float:
    """从 runtime_tuning 读取浮点参数，失败时返回默认值"""
    try:
        from src.config.core_config_engine import CoreSettingsHub

        val = CoreSettingsHub().runtime_tuning_block().get(key, default)
        return float(val)
    except Exception:
        return default


# 增强循环内部间隔常量（优先从配置读取）
_TICK_FLOOR_SEC = _rt_float("heartfc_tick_floor_sec", 0.15)
_DORMANT_POLL_SEC = _rt_float("heartfc_dormant_poll_sec", 1.0)
_PERCEPTION_COOLDOWN_SEC = _rt_float("heartfc_perception_cooldown_sec", 5.0)
_VOICE_COOLDOWN_SEC = _rt_float("heartfc_voice_cooldown_sec", 6.0)
_ENERGY_DRAIN_FLOOR = _rt_float("heartfc_energy_drain_floor", 0.08)
_POST_MESSAGE_RETRY_SEC = _rt_float("heartfc_post_message_retry_sec", 2.0)


class EnhancedHeartFChatting(HeartFChatting):
    """群聊增强版心流控制器

    在基础循环之上叠加增强判定管线：
    阶段协调 → 休息/等待门控 → 能量评估 → 感知引擎 → 内心独白 →
    主行动判定与执行。

    所有子系统通过懒导入获取单例，不在构造时预加载，
    保证与 heartflow.py 中心化调度器的启停顺序兼容。
    """

    def __init__(self, chat_id: str):
        super().__init__(chat_id)
        self._tick_world_snapshot = None
        # 阶段协调器是否已将此频道注册
        self._phase_registered = False
        # 增强循环轮次计数器（与父类 _cycle_counter 独立）
        self._enhanced_round = 0
        # 上一次执行感知引擎的时间戳
        self._last_perception_ts = 0.0
        # 上一次执行内心独白的时间戳
        self._last_voice_ts = 0.0
        # 上一次输出能量状态的时间戳
        self._last_energy_log_ts = 0.0
        # 能量状态日志输出间隔（秒）
        self._energy_log_interval = 300.0
        # 后台状态单行日志时间戳（用于可视化观察）
        self._last_backend_status_ts = 0.0
        # 后台状态输出间隔（秒）
        self._backend_status_interval = 10.0
        # 后台状态日志开关（默认关闭，避免刷屏）
        self._backend_status_log_enabled = False
        self._last_dashboard_status_line = ""
        try:
            from src.config.core_config_engine import get_core_config

            proactive_cfg = get_core_config().proactive_schedule_block()
            self._backend_status_log_enabled = bool(
                proactive_cfg.get("backend_status_log_enabled", False)
            )
            self._backend_status_interval = float(
                proactive_cfg.get("backend_status_log_interval_sec", 10.0)
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # 最近一次觉察判定的缓存
        self._cached_awareness = None
        # 最近一次内心独白结论缓存
        self._cached_voice = None
        # 连续跳过的增强轮次数（用于自适应节奏）
        self._consecutive_skip_ticks = 0
        # 多维状态系统最近一次决策网关裁定缓存
        self._last_gateway_verdict = None
        # 增强管线是否就绪（延迟到首次 start 后才设为 True）
        self._pipeline_ready = False
        # 连续发言计数（用于消耗计算）
        self._consecutive_speaks = 0.0
        self._peek_mode_active = False
        self._defense_mode_active = False
        self._pattern_forced_observe = False
        self._night_cycle = None
        # 上次发言时间（用于频率因子）
        self._last_speak_time = 0.0
        # 话痨惩罚值
        self._chatterbox_penalty = 0.0
        # 上次兴趣等级
        self._last_interest_level = 0.5
        # 最后用户ID
        self._last_user_id = ""
        # 本轮成功解析的emotion_tracker用户状态（跨出口统一用）
        self._resolved_emo_state = None
        self._resolved_emo_uid = ""
        # 非行动决策冷却（避免 observe/check_later 重复规划）
        self._planner_quiet_until = 0.0
        # 大模型深度分析升级冷却（避免频繁调用大模型）
        self._llm_upgrade_cooldown_until = 0.0
        # 大模型升级调用计数器（用于自适应限制）
        self._llm_upgrade_call_count = 0
        self._llm_upgrade_call_window_start = 0.0
        # LLM 决策的内容规划（用于主动回复）
        self._llm_content_plan = ""
        # LLM 决策的社交意图
        self._llm_social_intention = ""
        # 最近几条 bot 自己说过的话，供后续回复保持连贯
        self._recent_bot_utterances: List[Dict[str, Any]] = []
        # 连续主动发言但没人接话的次数
        self._unanswered_bot_turns = 0
        # 最近一次 bot 发言时间
        self._last_bot_reply_ts = 0.0
        # 用户消息累积追踪器（用于动态比例控制）
        self._user_msg_counter = 0
        self._bot_reply_counter = 0
        self._user_msg_timeline: List[float] = []
        self._bot_reply_timeline: List[float] = []
        self._burst_user_count = 0
        self._burst_start_time = 0.0
        self._is_in_burst = False
        self._last_user_msg_time = 0.0
        self._conversation_phase = "normal"
        self._phase_confidence = 0.5
        self._target_ratio = 0.4
        self._cumulative_ratio = 0.5
        self._reply_probability = 0.5
        # 共享资源管理器
        self._shared_resource_manager = None
        # 最近用户输入归一化缓存，用于跨批次重复识别
        self._recent_user_inputs: List[Dict[str, Any]] = []
        # 日志节流：上次输出时间
        self._last_status_log_ts = 0.0
        self._status_log_interval = 10.0
        # 准入度评估器缓存（由 readiness_evaluator 填充）
        self._legacy_constraint_hits = 0
        self._last_legacy_constraint_ts = 0.0
        self._last_legacy_penalty = 0.0
        self._last_legacy_reason = ""
        self._last_legacy_breakdown: Dict[str, Any] = {}
        self._last_legacy_gate = "allow"
        # 最近一次统一交互状态快照
        self._last_relation_snapshot: Dict[str, Any] = {}
        # 核心模块集成缓存
        self._cached_night_phase = None
        self._night_reply_suppression = 0.0
        self._cached_watch_level = None
        self._cached_presence_state = None
        self._cached_metabolism_state = None
        self._cached_semantic_routes: List = []
        self._cached_route_summary: Dict[str, int] = {}
        self._cached_participant_summary: Dict[str, Any] = {}
        self._cached_multimodal_summary: Dict[str, Any] = {}
        self._cached_learning_summary: Dict[str, Any] = {}
        self._cached_visibility_decisions: List = []
        self._cached_understanding_results: List = []
        self._cached_self_references: List = []
        self._cached_scene_snapshot = None
        self._cached_pattern_evidence: List = []
        self._cached_narration_plan = None
        self._cached_presence_verdict: Dict[str, Any] = {}
        self._cached_metabolism_constraints: Dict[str, Any] = {}
        self._cached_night_summary: Dict[str, Any] = {}
        self._cached_result_gauge = None
        self._cached_media_decisions: List = []
        # 技能生命周期与自适应管线缓存
        self._cached_skill_review: Dict[str, int] = {}
        self._cached_pipeline_summary: Dict[str, Any] = {}
        self._skill_review_round_interval = 10
        self._skills_registered = False
        self._cached_misunderstanding_signal = None
        self._cached_user_negative_emotion: float = 0.0
        self._cached_user_impression_tags: List[str] = []
        # F27：每小时回复上限计数器
        self._hourly_reply_count: int = 0
        self._hourly_window_start: float = 0.0
        # GAP-M/N/O/P：新引擎缓存
        self._cached_deep_visibility_results: List[Any] = []
        self._cached_freshness_records: Dict[str, Any] = {}
        self._cached_gossip_ritual_verdict: Optional[Dict[str, Any]] = None
        self._adaptive_pipeline_last_run: float = 0.0
        self._freshness_engine_initialized: bool = False
        self._deep_scorer_initialized: bool = False
        self._gossip_engine_initialized: bool = False
        # GAP-Q/R/S：新引擎缓存
        self._cached_memory_governance_snap: Optional[Dict[str, Any]] = None
        self._cached_safety_assessment: Optional[Dict[str, Any]] = None
        self._cached_planner_injection: Optional[Dict[str, Any]] = None
        self._memory_governance_initialized: bool = False
        self._safety_fusion_initialized: bool = False
        self._planner_injection_initialized: bool = False
        self._last_memory_governance_ts: float = 0.0
        self._last_safety_assessment_ts: float = 0.0
        # GAP-T/U/V/W：最终批次引擎缓存
        self._cross_validator_initialized: bool = False
        self._mm_bridge_initialized: bool = False
        self._emotion_feedback_initialized: bool = False
        self._attention_flow_initialized: bool = False
        self._cached_attention_snapshot: Optional[Dict[str, Any]] = None
        self._cached_emotion_feedback_report: Optional[Dict[str, Any]] = None
        # 统一状态仪表盘缓存（替代40+个分散_cached_的直接读取）
        self._cached_dashboard_snapshot: Optional[Dict[str, Any]] = None
        self._dashboard_initialized: bool = False
        self._last_dashboard_build_ts: float = 0.0
        self._dashboard_ttl_sec: float = 3.0
        # 双通道架构：主动感兴趣通道（后台常驻任务）
        self._proactive_task: Optional[asyncio.Task] = None
        self._proactive_running: bool = False
        self._last_dashboard_status_line: str = ""
        # 消息去重：已处理过的消息ID集合（防止同一消息被多次处理）
        self._processed_message_ids: Set[str] = set()
        self._last_dashboard_status_line: str = ""

    def _remember_bot_utterance(self, content: str) -> None:
        text = str(content or "").strip()
        if not text:
            return
        now = time.time()
        self._recent_bot_utterances.append({"text": text, "ts": now})
        self._prune_recent_bot_utterances(now)

    def _prune_recent_bot_utterances(
        self, now: Optional[float] = None
    ) -> None:
        current = now if now is not None else time.time()
        ttl_sec = 180.0
        recent_items = []
        for item in self._recent_bot_utterances[-5:]:
            text = str(item.get("text", "") or "").strip()
            ts = float(item.get("ts", 0.0) or 0.0)
            if not text:
                continue
            if ts > 0 and (current - ts) <= ttl_sec:
                recent_items.append({"text": text, "ts": ts})
        self._recent_bot_utterances = recent_items[-3:]

    def _recent_bot_texts(self) -> List[str]:
        self._prune_recent_bot_utterances()
        return [
            str(item.get("text", "") or "").strip()
            for item in self._recent_bot_utterances
            if str(item.get("text", "") or "").strip()
        ]

    async def _send_and_store_reply(
        self,
        response_set,
        action_message,
        cycle_timers: Dict[str, float],
        thinking_id,
        actions,
        selected_expressions: Optional[List[int]] = None,
        quote_message: bool = False,
        target_message=None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, float]]:
        """兼容增强链路的回复发送入口。

        历史上不稳定地变更过这个方法的可用性和签名。
        增强版统一在这里兜底，继续复用现有发送与动作落库流程。
        """
        reply_target = action_message or target_message

        with Timer("回复发送", cycle_timers):
            reply_text = await self._send_response(
                reply_set=response_set,
                message_data=reply_target,
                selected_expressions=selected_expressions,
                quote_message=quote_message,
            )

        platform = None
        if reply_target is not None:
            platform = getattr(
                getattr(reply_target, "chat_info", None), "platform", None
            )
            if platform is None:
                platform = getattr(
                    getattr(reply_target, "platform", None), "value", None
                ) or getattr(reply_target, "platform", None)
        if platform is None:
            platform = getattr(self.chat_stream, "platform", "unknown")

        user_id = ""
        if reply_target is not None:
            user_id = str(
                getattr(
                    getattr(reply_target, "user_info", None), "user_id", ""
                )
                or getattr(reply_target, "user_id", "")
                or ""
            )

        person_name = "对方"
        if user_id:
            try:
                from src.common.database.database_model import Person

                person = Person(platform=platform, user_id=user_id)
                person_name = person.person_name or person_name
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

        action_prompt_display = f"你对{person_name}进行了回复：{reply_text}"
        await database_api.store_action_info(
            chat_stream=self.chat_stream,
            action_build_into_prompt=False,
            action_prompt_display=action_prompt_display,
            action_done=True,
            thinking_id=thinking_id,
            action_data={
                "reply_text": reply_text,
                "quote_message": bool(quote_message),
            },
            action_name="reply",
        )

        loop_info: Dict[str, Any] = {
            "loop_plan_info": {
                "action_result": actions,
            },
            "loop_action_info": {
                "action_taken": True,
                "reply_text": reply_text,
                "command": "",
                "taken_time": time.time(),
                "quote_message": bool(quote_message),
            },
        }

        return loop_info, reply_text, cycle_timers

    def _mark_user_replied(self) -> None:
        self._unanswered_bot_turns = 0

    def _message_looks_like_user_reply(self, msg: Any) -> bool:
        if not msg:
            return False
        user_id = str(getattr(msg, "user_id", "") or "")
        if not user_id or user_id == "bot":
            return False

        text = str(
            getattr(msg, "processed_plain_text", "")
            or getattr(msg, "plain_text", "")
            or ""
        ).strip()
        if not text:
            return False

        reply_to = str(getattr(msg, "reply_to", "") or "").strip()
        if reply_to:
            try:
                from src.core.self_reply_recognizer import (
                    get_self_reply_recognizer,
                )

                recognizer = get_self_reply_recognizer()
                if recognizer.get_bot_message(reply_to) is not None:
                    return True
                chain = recognizer.get_reply_chain(reply_to)
                if chain and chain.bot_message_ids:
                    return True
            except Exception as _e:
                logger.debug(f"{self.log_prefix} 回复链检测异常: {_e}")

        referenced_text = str(getattr(msg, "quoted_content", "") or "").strip()
        recent_bot_texts = self._recent_bot_texts()
        if referenced_text and recent_bot_texts:
            if any(
                referenced_text in said or said in referenced_text
                for said in recent_bot_texts[-3:]
            ):
                return True

        # bot说完后用户很快跟一句，视为接话
        # 关键：必须检查消息时间戳在bot发言之后，避免旧消息误触重置退避计数
        _msg_ts = float(getattr(msg, "time", 0.0) or getattr(msg, "timestamp", 0.0) or 0.0)
        if (
            self._last_bot_reply_ts > 0
            and _msg_ts > self._last_bot_reply_ts
            and (time.time() - self._last_bot_reply_ts) <= 180.0
        ):
            return True

        recent_self = recent_bot_texts[-1] if recent_bot_texts else ""
        if recent_self:
            bot_terms = [
                part
                for part in re.split(r"[，。！？、；：,.!?\s]+", recent_self)
                if len(part) >= 2
            ]
            if bot_terms and any(term in text for term in bot_terms[:4]):
                return True

        return False

    def _batch_contains_real_user_reply(
        self, incoming_batch: List[Any]
    ) -> bool:
        for msg in reversed(incoming_batch or []):
            if self._message_looks_like_user_reply(msg):
                return True
        return False

    def _build_self_reply_memory(self) -> str:
        recent_bot_texts = self._recent_bot_texts()
        if not recent_bot_texts:
            return ""
        lines = []
        for idx, text in enumerate(recent_bot_texts[-3:], start=1):
            brief = text if len(text) <= 120 else f"{text[:120]}..."
            lines.append(f"{idx}. {brief}")
        return "[你刚刚自己说过的话]\n" + "\n".join(lines)

    def _build_self_continuity_context(self) -> str:
        recent_bot_texts = self._recent_bot_texts()
        if not recent_bot_texts:
            return ""
        lines = []
        for idx, text in enumerate(recent_bot_texts[-3:], start=1):
            brief = text if len(text) <= 160 else f"{text[:160]}..."
            lines.append(f"- 第{idx}句：{brief}")
        return (
            "[连续对话参考]\n"
            "下面这些是你刚刚在这个群聊里亲口说过的话。\n"
            "如果现在继续说话，优先顺着它们往下接，不要像失忆一样重开话题。\n"
            + "\n".join(lines)
        )

    def _select_preferred_reply_message(
        self, anchor_message: Optional[Any], candidate_messages: List[Any]
    ) -> Optional[Any]:
        if anchor_message is None:
            return anchor_message
        try:
            from src.chat.behavior.target_selector import (
                choose_preferred_message_target,
            )

            relation_snapshot = self._resolve_relation_view()
            relationship_candidates = (
                relation_snapshot.get("relationship_candidates", []) or []
            )
            active_user_ids = [
                str(getattr(message, "user_id", "") or "").strip()
                for message in candidate_messages
                if str(getattr(message, "user_id", "") or "").strip()
                and str(getattr(message, "user_id", "") or "").strip() != "bot"
            ]
            return choose_preferred_message_target(
                anchor_message=anchor_message,
                candidate_messages=candidate_messages,
                relationship_candidates=relationship_candidates,
                active_user_ids=active_user_ids,
            )
        except Exception:
            return anchor_message

    def _build_synthetic_self_messages(self) -> List[Any]:
        synthetic_messages: List[Any] = []
        self._prune_recent_bot_utterances()
        recent_items = self._recent_bot_utterances[-3:]
        if not recent_items:
            return synthetic_messages

        total = len(recent_items)
        for idx, text in enumerate(recent_items, start=1):
            brief = str(text.get("text", "") or "").strip()
            item_ts = float(text.get("ts", 0.0) or 0.0) or time.time()
            if not brief:
                continue
            synthetic_messages.append(
                SimpleNamespace(
                    user_id="bot",
                    processed_plain_text=brief,
                    plain_text=brief,
                    message_id=f"self_ctx_{int(item_ts)}_{idx}",
                    timestamp=item_ts - (total - idx) * 0.01,
                    is_mentioned=False,
                    is_at=False,
                    _is_synthetic_self_context=True,
                )
            )
        return synthetic_messages

    def _should_back_off_idle_proactive(self) -> bool:
        # 连续2次主动发言无人回应就退避，避免自言自语
        return self._unanswered_bot_turns >= 2

    # ────────────────── 生命周期 ──────────────────

    async def start(self):
        self._ensure_phase_registration()
        logger.info(
            f"{self.log_prefix} 独立自主代理已停用，统一由增强主循环负责主动判断"
        )
        self._pipeline_ready = True
        # 启动主动通道后台任务（与被动监听通道并行运行）
        self._proactive_task: Optional[asyncio.Task] = None
        self._proactive_running: bool = True
        try:
            self._proactive_task = asyncio.create_task(
                self._proactive_background_loop()
            )
            logger.info(
                f"{self.log_prefix} 🔔 主动感兴趣通道已启动（后台常驻）"
            )
        except Exception as exc:
            logger.warning(
                f"{self.log_prefix} 主动通道启动失败，回退到主循环触发模式: {exc}"
            )
            self._proactive_task = None
        await super().start()
        logger.info(f"{self.log_prefix} 增强版控制器已接管，管线就绪")

    async def stop(self) -> None:
        self._pipeline_ready = False
        self._proactive_running = False
        if self._proactive_task and not self._proactive_task.done():
            self._proactive_task.cancel()
            try:
                await self._proactive_task
            except asyncio.CancelledError:
                pass
            logger.info(f"{self.log_prefix} 🔔 主动感兴趣通道已停止")
        self._teardown_voice_engine()
        await super().stop()

    # ────────────────── 阶段协调器注册 ──────────────────

    def _ensure_phase_registration(self) -> None:
        if self._phase_registered:
            return
        try:
            from src.chat.heart_flow.state_machine import get_phase_coordinator
            from src.common.data_models.heartflow_models import FlowPhase

            coordinator = get_phase_coordinator()
            coordinator.register_channel(
                self.stream_id, initial=FlowPhase.ENGAGED
            )
            self._phase_registered = True
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 注册阶段协调器失败: {exc}")

    # ────────────────── 内心独白引擎清理 ──────────────────

    def _teardown_voice_engine(self) -> None:
        try:
            from src.chat.heart_flow.inner_voice import teardown_engine

            teardown_engine(self.stream_id)
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    async def _update_user_impression_after_reply(
        self, reply_text: str
    ) -> None:
        """回复成功后更新用户印象

        让模型根据互动内容动态生成用户印象标签
        """
        try:
            from src.modules.social_value.social_affect_fuser import (
                get_social_affect_fuser,
            )

            fuser = get_social_affect_fuser()
            recent_interactions = reply_text[:200] if reply_text else ""
            await fuser.refresh_impression(
                user_id=self._last_user_id,
                channel_id=self.stream_id,
                interaction_digest=recent_interactions,
                topics=[],
                style="",
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 更新用户印象失败: {exc}")

    # ═══════════════════════════════════════════════════
    #  核心模块集成桥接层
    # ═══════════════════════════════════════════════════

    def _integrate_night_cycle(self, now: float) -> Optional[str]:
        """夜间节律评估，返回 'sleep' 表示应跳转休眠，'burned_out' 表示熬穿"""
        _grace_sec = getattr(self, "_wake_grace_until", 0.0)
        if now < _grace_sec:
            return None
        try:
            from src.core.night_cycle_system import get_night_cycle, NightPhase

            _nc = get_night_cycle(self.stream_id)
            _energy = 1.0
            _activity = 50.0
            _boredom = 0.0
            _consecutive_min = 0.0
            _snap = self._tick_world_snapshot
            if _snap:
                _res = getattr(_snap, "self_resources", None)
                if _res:
                    _energy = float(
                        getattr(_res, "chat_energy_ratio", 1.0) or 1.0
                    )
                    _activity = float(
                        getattr(_res, "activity_level", 50.0) or 50.0
                    )
                    _boredom = float(getattr(_res, "boredom", 0.0) or 0.0)
            if self._cached_metabolism_state:
                _consecutive_min = float(
                    getattr(
                        self._cached_metabolism_state,
                        "consecutive_active_minutes",
                        0.0,
                    )
                    or 0.0
                )
            _phase = _nc.evaluate(
                energy_ratio=_energy,
                activity_level=_activity,
                boredom=_boredom,
                consecutive_active_minutes=_consecutive_min,
            )
            self._cached_night_phase = _phase
            # 缓存夜间行为摘要
            self._cached_night_summary = _nc.night_behavior_summary()
            # 深睡/浅睡 → 休眠
            if _phase in (NightPhase.DEEP_SLEEP, NightPhase.LIGHT_SLEEP):
                # 浅睡时检查窥屏窗口
                if (
                    _phase == NightPhase.LIGHT_SLEEP
                    and _nc.evaluate_sleep_peek()
                ):
                    logger.debug(
                        f"{self.log_prefix} 浅睡窥屏窗口开放，暂不休眠"
                    )
                else:
                    return "sleep"
            # 熬穿 → 也应休眠
            if _phase == NightPhase.BURNED_OUT:
                return "burned_out"
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 夜间节律评估异常: {exc}")
        return None

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
            _boredom = float(
                self._cached_metabolism_constraints.get("boredom", 0.0) or 0.0
            )
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
                _total_users = int(
                    self._cached_participant_summary.get("hot_count", 0) or 0
                )
                _total_users += int(
                    self._cached_participant_summary.get("warm_count", 0) or 0
                )
                if _total_users >= 10:
                    _noise_factor = 2.5
                elif _total_users >= 6:
                    _noise_factor = 1.8
                elif _total_users <= 2:
                    _noise_factor = 0.65
            if self._cached_multimodal_summary:
                _storm = str(
                    self._cached_multimodal_summary.get("storm_mode", "") or ""
                )
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
                        _result["reactivation_strength"] = min(
                            0.88, 0.55 + _max_ref_str * 0.40
                        )
            if _result["reactivation_type"] is None and _max_ref_str > 0.35:
                _result["reactivation_type"] = "B_moderate_ref"
                _result["reactivation_strength"] = 0.35 + _max_ref_str * 0.25
        if _result["reactivation_type"] is None:
            _rel_snap_d = getattr(self, "_last_relation_snapshot", None) or {}
            _aff_d = float(_rel_snap_d.get("affection", 50.0) or 50.0)
            if _aff_d > 55 and incoming_batch:
                _has_known_user = False
                for _msg in incoming_batch:
                    _uid = str(getattr(_msg, "user_id", "") or "").strip()
                    if _uid and _uid != "bot":
                        _has_known_user = True
                        break
                if _has_known_user:
                    _result["reactivation_type"] = "C_familiar_user"
                    _result["reactivation_strength"] = (
                        0.22 + (_aff_d - 55.0) / 200.0
                    )
        if _result["reactivation_type"] is None:
            if self._cached_participant_summary:
                _sudden_burst = int(
                    self._cached_participant_summary.get(
                        "sudden_burst_count", 0
                    )
                    or 0
                )
                if _sudden_burst >= 3:
                    _result["reactivation_type"] = "D_abnormal_activity"
                    _result["reactivation_strength"] = min(
                        0.30, _sudden_burst * 0.06
                    )
        _result["decay_half_life_sec"] = [900.0, 480.0, 180.0, float("inf")][
            _result["decay_level"]
        ]
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
                        _atmo_type = (
                            _pe_pat.value
                            if hasattr(_pe_pat, "value")
                            else str(_pe_pat)
                        )
        _hot_users = 0
        if self._cached_participant_summary:
            _hot_users = int(
                self._cached_participant_summary.get("hot_count", 0) or 0
            )
            _hot_users += int(
                self._cached_participant_summary.get("warm_count", 0) or 0
            )
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
            _sw_val = float(
                getattr(self._cached_presence_state, "social_willingness", 0.5)
                or 0.5
            )
            _aw_val = float(
                getattr(self._cached_presence_state, "avoidance_tendency", 0.0)
                or 0.0
            )
            _qp_val = float(
                getattr(self._cached_presence_state, "quiet_preference", 0.2)
                or 0.2
            )
            _soc_comfort = (
                _sw_val * 0.55
                + (1.0 - _aw_val) * 0.30
                + (1.0 - _qp_val) * 0.15
            )
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
            _chat_v = float(
                self._cached_metabolism_constraints.get("chat_value", 50.0)
                or 50.0
            )
            if _chat_v > 60:
                _energy_label = "energetic"
            elif _chat_v > 25:
                _energy_label = "moderate"
            elif _chat_v > 10:
                _energy_label = "tired"
            else:
                _energy_label = "exhausted"
        if self._cached_night_phase:
            from src.core.night_cycle_system import NightPhase

            _phase = self._cached_night_phase
            if _phase == NightPhase.DEEP_SLEEP:
                _mood = f"{_mood}_asleep"
                _energy_label = "deep_asleep"
            elif _phase in (NightPhase.LIGHT_SLEEP, NightPhase.BURNED_OUT):
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
        elif _sit_type == "tense_confrontation" and _mood not in (
            "withdrawn_defensive",
        ):
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
        if _aw_val > 0.6 if self._cached_presence_state else False:
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
            "level": (
                "high"
                if _risk_score > 0.40
                else "medium" if _risk_score > 0.18 else "low"
            ),
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
        random.seed(_day_hash)
        # 能量预算检查
        _budget_used = getattr(self, "_bg_event_budget_used_today", 0)
        _budget_max = 8
        if _budget_used >= _budget_max:
            random.seed()
            return None
        # 时间窗口检查
        _hour = time.localtime(now).tm_hour
        if 2 <= _hour <= 6:
            random.seed()
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
        _event_roll = random.random() * _time_weight
        _threshold_base = 0.08
        if self._cached_night_phase:
            from src.core.night_cycle_system import NightPhase

            if self._cached_night_phase in (
                NightPhase.DEEP_SLEEP,
                NightPhase.LIGHT_SLEEP,
            ):
                _threshold_base *= 0.15
        if _event_roll < _threshold_base:
            random.seed()
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
        _r = random.random() * _total_w
        _cumul = 0.0
        _chosen_type = "random_observation"
        for _et, _ew in _event_type_weights:
            _cumul += _ew
            if _r <= _cumul:
                _chosen_type = _et
                break
        # 生成事件
        self._bg_event_budget_used_today = _budget_used + 1
        random.seed()
        _event = {
            "type": _chosen_type,
            "scheduled_at": now,
            "priority": round(random.uniform(0.15, 0.55), 3),
            "seed_hash": _day_hash,
            "time_weight": round(_time_weight, 3),
            "budget_remaining": _budget_max - self._bg_event_budget_used_today,
        }
        logger.debug(
            f"{self.log_prefix} 后台事件调度: {_chosen_type}(p={_event['priority']})"
        )
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
            _silence_after_reply = float(
                self._cached_participant_summary.get(
                    "silence_since_last_bot", 0.0
                )
                or 0.0
            )
        if self._last_bot_reply_ts > 0:
            _elapsed_silence = time.time() - self._last_bot_reply_ts
            if (
                _elapsed_silence > 300
                and _silence_after_reply > _elapsed_silence * 0.8
            ):
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
                    _pv = (
                        _pat_enum.value
                        if hasattr(_pat_enum, "value")
                        else str(_pat_enum)
                    )
                    _prev_patterns = getattr(
                        self, "_recent_pattern_history", []
                    )
                    if (
                        _pv not in _prev_patterns[-5:]
                        if _prev_patterns
                        else []
                    ):
                        _result["should_learn"] = True
                        _result["trigger_types"].append(
                            f"T2_new_pattern:{_pv}"
                        )
                        _result["confidence"] += 0.28
                        _pattern_changed = True
        # T3: 连续否定检测
        _legacy_hits = getattr(self, "_legacy_constraint_hits", 0)
        if _legacy_hits >= 3:
            _result["should_learn"] = True
            _result["trigger_types"].append("T3_repeated_constraints")
            _result["confidence"] += min(0.40, _legacy_hits * 0.10)
        # T4: 关系阶跃检测
        _rel_snap_g = getattr(self, "_last_relation_snapshot", None) or {}
        _aff_now = float(_rel_snap_g.get("affection", 50.0) or 50.0)
        _aff_prev = getattr(self, "_prev_affection_for_learning", 50.0)
        _aff_delta = abs(_aff_now - _aff_prev)
        if _aff_delta > 15.0:
            _result["should_learn"] = True
            _direction = "up" if _aff_now > _aff_prev else "down"
            _result["trigger_types"].append(
                f"T4_affection_jump:{_direction}:{
                    _aff_delta:.0f}"
            )
            _result["confidence"] += 0.22
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
            from src.core.group_scene_state import get_group_scene

            _scene_h = get_group_scene(self.stream_id)
            _topics_h = _scene_h.active_topics(limit=5)
            _threads = []
            for _t in _topics_h:
                if _t.is_stale(600):
                    continue
                _owners = list(getattr(_t, "contributors", []) or [])[:4]
                if len(_owners) >= 2:
                    _threads.append(
                        {
                            "topic": getattr(_t, "topic_summary", "")[:60],
                            "ownership": _t.ownership_type(),
                            "participants": _owners,
                            "age_sec": int(getattr(_t, "age_seconds", 0) or 0),
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
                    _ref_user = getattr(_ref, "target_user", None) or getattr(
                        _ref, "user_id", None
                    )
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
                        _result["locked_topic"] = (
                            _pe_pat.value
                            if hasattr(_pe_pat, "value")
                            else str(_pe_pat)
                        )
                        break
        return _result

    def _integrate_watch_state(
        self, now: float, has_messages: bool = False
    ) -> None:
        """更新关注层级状态机"""
        try:
            from src.core.watch_state_machine import get_watch_machine

            _wm = get_watch_machine(self.stream_id)
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
                    _scene = getattr(_snap, "scene", None)
                    if _scene:
                        _silence = float(
                            getattr(_scene, "silence_seconds", 0.0) or 0.0
                        )
                _wm.evaluate_natural_decay(
                    silence_sec=_silence, boredom=_boredom
                )
            self._cached_watch_level = _wm.current_level
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 关注状态机异常: {exc}")

    def _evaluate_watch_full_transitions(
        self, now: float, has_ping: bool, incoming_batch: List
    ) -> Optional[str]:
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
            from src.core.watch_state_machine import (
                get_watch_machine,
                WatchLevel as WatchState,
            )

            _wm = get_watch_machine(self.stream_id)
            _current = _wm.current_level
            if _current is None:
                return None
            _cur_val = (
                _current.value if hasattr(_current, "value") else str(_current)
            )
            # ── 收集转移决策所需的全部上下文 ──
            _boredom = 0.0
            _fatigue = 0.0
            _activity = 50.0
            _silence = 0.0
            _snap = self._tick_world_snapshot
            if _snap:
                _res = getattr(_snap, "self_resources", None)
                if _res:
                    _boredom = float(getattr(_res, "boredom", 0.0) or 0.0)
                    _fatigue = float(getattr(_res, "fatigue", 0.0) or 0.0)
                    _activity = float(
                        getattr(_res, "activity_level", 50.0) or 50.0
                    )
                _scene = getattr(_snap, "scene", None)
                if _scene:
                    _silence = float(
                        getattr(_scene, "silence_seconds", 0.0) or 0.0
                    )
            # 关系与好感
            _rel = getattr(self, "_last_relation_snapshot", None) or {}
            _aff = float(_rel.get("affection", 50.0) or 50.0)
            # 存在态
            _sw = 0.5
            _aw = 0.5
            _avoid = 0.0
            if self._cached_presence_state:
                _sw = float(
                    getattr(
                        self._cached_presence_state, "social_willingness", 0.5
                    )
                    or 0.5
                )
                _aw = float(
                    getattr(
                        self._cached_presence_state, "watch_willingness", 0.5
                    )
                    or 0.5
                )
                _avoid = float(
                    getattr(
                        self._cached_presence_state, "avoidance_tendency", 0.0
                    )
                    or 0.0
                )
            # 打断权分级（GAP-E）
            _tier3_count = 0
            _tier2_count = 0
            _interruption = getattr(self, "_cached_interruption_tiers", None)
            if isinstance(_interruption, dict):
                _tier3_count = sum(
                    1 for s in _interruption.values() if s >= 0.65
                )
                _tier2_count = sum(
                    1 for s in _interruption.values() if 0.30 <= s < 0.65
                )
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
                _hot_users = int(
                    self._cached_participant_summary.get("hot_count", 0) or 0
                )
                _hot_users += int(
                    self._cached_participant_summary.get("warm_count", 0) or 0
                )
            # ══ 转移路径评估 ══
            # 路径⑥：[任意] → blackout（通用退出条件，优先级最高）
            _should_blackout = False
            _blackout_reason = ""
            if _activity < 12 and _boredom > 65 and _silence > 1800:
                _should_blackout = True
                _blackout_reason = "低活跃+高无聊+长静默"
            elif _fatigue > 75 and _silence > 600:
                _should_blackout = True
                _blackout_reason = "高疲劳+静默"
            elif _avoid > 0.7 and _sw < 0.15:
                _should_blackout = True
                _blackout_reason = "高回避+低社交欲"
            elif _hot_users >= 8 and _aff < 20:
                _should_blackout = True
                _blackout_reason = "过于嘈杂+低好感"
            if _should_blackout and _cur_val not in ("blackout",):
                try:
                    _wm.signal_force_level(WatchState.BLACKOUT)
                    return f"{_cur_val}→blackout({_blackout_reason})"
                except Exception as _e:
                    logger.warning(f"{self.log_prefix} 强制进入blackout失败: {_e}")
            # 路径①：blackout → peek
            if _cur_val == "blackout":
                _peek_trigger_strength = 0.0
                if has_ping:
                    _peek_trigger_strength += 0.55
                if _boredom > 55:
                    _peek_trigger_strength += min(
                        0.25, (_boredom - 55) / 100.0
                    )
                if _aff > 55:
                    _peek_trigger_strength += 0.10
                if _tier3_count > 0:
                    _peek_trigger_strength += 0.18 * min(
                        1.0, _tier3_count / 3.0
                    )
                if _is_heated:
                    _peek_trigger_strength += 0.08
                if _discuss_n >= 2:
                    _peek_trigger_strength += 0.15
                if _hot_users >= 4 and _hot_users <= 8:
                    _peek_trigger_strength += 0.06
                if _peek_trigger_strength > 0.42:
                    try:
                        _wm.signal_force_level(WatchState.PEEK)
                        return f"blackout→peek(trigger={
                            _peek_trigger_strength:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} blackout→peek异常: {_e}")
            # 路径②：peek → skim_window
            if _cur_val == "peek":
                _skim_trigger = 0.0
                if _tier3_count > 0:
                    _skim_trigger += 0.22 * min(1.0, _tier3_count)
                if _tier2_count >= 2:
                    _skim_trigger += 0.12
                if _ref_max > 0.35:
                    _skim_trigger += 0.15
                if _is_heated:
                    _skim_trigger += 0.10
                if incoming_batch and len(incoming_batch) >= 3:
                    _skim_trigger += 0.08
                if _boredom > 40:
                    _skim_trigger += 0.06
                if _skim_trigger > 0.38:
                    try:
                        _wm.signal_force_level(WatchState.SKIM_WINDOW)
                        return f"peek→skim_window(trigger={_skim_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} peek→skim_window异常: {_e}")
            # 路径③：skim → active_watch
            if _cur_val == "skim_window":
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
                        _wm.signal_force_level(WatchState.ACTIVE_WATCH)
                        return f"skim→active_watch(trigger={
                            _active_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} skim→active_watch异常: {_e}")
            # 路径④：active → engaged
            if _cur_val == "active_watch":
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
                        _wm.signal_force_level(WatchState.ENGAGED)
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
                if len(incoming_batch) <= 1 and not has_ping:
                    _disengage_trigger += 0.12
                if _disengage_trigger > 0.48:
                    try:
                        _wm.signal_force_level(WatchState.ACTIVE_WATCH)
                        return f"engaged→active(disengage={
                            _disengage_trigger:.2f})"
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
                        _wm.signal_force_level(WatchState.BLACKOUT)
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
                if _boredom < 35:
                    _fallback_trigger += 0.12
                if _ref_max < 0.15 and not _is_heated:
                    _fallback_trigger += 0.18
                if _fallback_trigger > 0.45:
                    try:
                        _wm.signal_force_level(WatchState.PEEK)
                        return f"skim→peek(fallback={_fallback_trigger:.2f})"
                    except Exception as _e:
                        logger.debug(f"{self.log_prefix} skim→peek异常: {_e}")
            return None
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 观看状态转移评估异常: {exc}")
            return None

    def _integrate_presence_metabolism(
        self, eagerness_val: float, now: float
    ) -> None:
        """计算主观存在态 + 代谢引擎恢复"""
        try:
            from src.core.subjective_presence_core import get_presence_core

            _pc = get_presence_core(self.stream_id)
            # F17：提取话题/好友激活信号，注入存在态计算（心理新鲜度+回看激活）
            _has_interest_topic = False
            _has_close_friend = False
            try:
                from src.core.group_scene_state import get_group_scene

                _gs = get_group_scene(self.stream_id)
                _topics = _gs.active_topics(limit=3)
                for _t in _topics:
                    if not _t.is_stale(300) and _t.heat.value in (
                        "hot",
                        "warm",
                    ):
                        _has_interest_topic = True
                        break
                _rel_snap = (
                    getattr(self, "_last_relation_snapshot", None) or {}
                )
                if float(_rel_snap.get("affection", 0) or 0) > 60:
                    _has_close_friend = True
            except Exception as _e:
                logger.debug(f"{self.log_prefix} 亲密朋友检测异常: {_e}")
            if self._tick_world_snapshot:
                self._cached_presence_state = _pc.compute_from_snapshot(
                    self._tick_world_snapshot,
                    has_interesting_topic=_has_interest_topic,
                    has_close_friend_active=_has_close_friend,
                )
            else:
                self._cached_presence_state = _pc.compute(
                    chat_energy_ratio=eagerness_val,
                    has_interesting_topic=_has_interest_topic,
                    has_close_friend_active=_has_close_friend,
                )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 主观存在态计算异常: {exc}")
        try:
            from src.core.metabolism_engine import get_metabolism_engine

            _me = get_metabolism_engine(self.stream_id)
            _idle_sec = max(0.0, now - (self._last_perception_ts or now))
            _me.recover(min(_idle_sec, 60.0))
            self._cached_metabolism_state = _me.state
            self._cached_metabolism_constraints = (
                _me.behavior_constraint_summary()
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 代谢引擎恢复异常: {exc}")
        # F7+F8+F25：代谢值反向压制/驱动存在态——完整状态联动链
        if (
            self._cached_metabolism_constraints
            and self._cached_presence_state is not None
        ):
            _boredom_drive = float(
                self._cached_metabolism_constraints.get(
                    "boredom_watch_drive", 0.0
                )
                or 0.0
            )
            _loafing_sup = float(
                self._cached_metabolism_constraints.get(
                    "loafing_suppression", 0.0
                )
                or 0.0
            )
            _boredom_level = float(
                getattr(self._cached_metabolism_state, "boredom_level", 0.0)
                or 0.0
            )
            _loafing_level = float(
                getattr(self._cached_metabolism_state, "loafing_level", 0.0)
                or 0.0
            )
            if 0.10 < _boredom_drive < 0.32:
                _orig_ww = getattr(
                    self._cached_presence_state, "watch_willingness", 0.5
                )
                _boost = min(0.12, (_boredom_drive - 0.10) * 0.4)
                try:
                    self._cached_presence_state.watch_willingness = min(
                        0.85, _orig_ww + _boost
                    )
                except Exception as _e:
                    logger.debug(f"{self.log_prefix} watch_willingness boost异常: {_e}")
            elif _boredom_drive > 0.38:
                _orig_ww = getattr(
                    self._cached_presence_state, "watch_willingness", 0.5
                )
                _penalty = min(0.30, (_boredom_drive - 0.38) * 0.5)
                try:
                    self._cached_presence_state.watch_willingness = max(
                        0.05, _orig_ww - _penalty
                    )
                except Exception as _e:
                    logger.debug(f"{self.log_prefix} watch_willingness penalty异常: {_e}")
                logger.debug(
                    f"{
                        self.log_prefix} 😑 无聊驱动={
                        _boredom_drive:.2f}, 窥屏意愿{
                        _orig_ww:.2f}→{
                        self._cached_presence_state.watch_willingness:.2f}"
                )
            if _loafing_sup > 0.40:
                _orig_sw = getattr(
                    self._cached_presence_state, "social_willingness", 0.5
                )
                _penalty = min(0.35, (_loafing_sup - 0.40) * 0.6)
                try:
                    self._cached_presence_state.social_willingness = max(
                        0.05, _orig_sw - _penalty
                    )
                except Exception as _e:
                    logger.debug(f"{self.log_prefix} social_willingness异常: {_e}")
                logger.debug(
                    f"{
                        self.log_prefix} 🐟 摸鱼压制={
                        _loafing_sup:.2f}, 社交意愿{
                        _orig_sw:.2f}→{
                        self._cached_presence_state.social_willingness:.2f}"
                )
            if _boredom_level > 55 and _loafing_level < 30:
                try:
                    from src.core.metabolism_engine import (
                        get_metabolism_engine,
                    )

                    _me2 = get_metabolism_engine(self.stream_id)
                    _convert_amount = min(0.8, (_boredom_level - 55) / 100.0)
                    _me2._state.loafing_level = min(
                        100.0, _me2._state.loafing_level + _convert_amount
                    )
                    if _convert_amount > 0.15:
                        logger.debug(
                            f"{self.log_prefix} 🔄 无聊→摸鱼转化: +{_convert_amount:.2f}"
                        )
                except Exception as _e:
                    logger.debug(f"{self.log_prefix} 无聊→摸鱼转化异常: {_e}")

    def _integrate_message_routing(self, incoming_batch: List) -> None:
        """语义路由分类 + 时间流注册 + 活跃名册 + 多模态预算"""
        try:
            from src.core.message_semantic_router import get_semantic_router

            _router = get_semantic_router()
            self._cached_semantic_routes = _router.classify_batch(
                incoming_batch
            )
            self._cached_route_summary = _router.summarize_batch(
                self._cached_semantic_routes
            )
        except Exception as exc:
            self._cached_semantic_routes = []
            self._cached_route_summary = {}
            logger.debug(f"{self.log_prefix} 语义路由分类异常: {exc}")
        try:
            from src.core.subjective_time_stream import get_time_stream

            get_time_stream(self.stream_id).register_batch(incoming_batch)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 时间流注册异常: {exc}")
        try:
            from src.core.active_participant_roster import (
                get_participant_roster,
            )

            _roster = get_participant_roster(self.stream_id)
            for i, _m in enumerate(incoming_batch):
                _uid = str(getattr(_m, "user_id", "") or "").strip()
                if _uid and _uid != "bot":
                    _text = str(getattr(_m, "processed_plain_text", "") or "")[
                        :50
                    ]
                    _route = (
                        self._cached_semantic_routes[i]
                        if i < len(self._cached_semantic_routes)
                        else None
                    )
                    _route_type = "plain_text"
                    if _route is not None:
                        _route_type = getattr(
                            getattr(_route, "category", None), "value", ""
                        ) or str(getattr(_route, "category", "plain_text"))
                    _emotion = (
                        0.4
                        if any(
                            mark in _text for mark in ("！", "!", "？", "?")
                        )
                        else 0.0
                    )
                    _message_quality = 0.7 if len(_text) >= 16 else 0.5
                    _topic_relevance = 0.7 if len(_text) >= 8 else 0.4
                    _bot_interacted = bool(
                        _route
                        and (
                            getattr(_route, "is_at_bot", False)
                            or getattr(_route, "is_quote_to_bot", False)
                        )
                    )
                    _roster.reset_consecutive(_uid)
                    _roster.touch(
                        _uid,
                        user_name=str(
                            getattr(_m, "user_name", "")
                            or getattr(_m, "nickname", "")
                            or ""
                        ),
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
            from src.core.multimodal_budgeter import get_multimodal_budgeter

            _media_items = []
            _budget_ratio = 1.0
            if self._cached_metabolism_state is not None:
                _chat_fuel = (
                    float(
                        getattr(
                            self._cached_metabolism_state, "chat_fuel", 100.0
                        )
                        or 100.0
                    )
                    / 100.0
                )
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
                _budget_ratio = max(
                    0.0, min(1.0, min(_chat_fuel, _thinking_fuel))
                )
            for i, _sr in enumerate(self._cached_semantic_routes):
                _cat = (
                    _sr.category.value
                    if hasattr(_sr.category, "value")
                    else str(_sr.category)
                )
                if _cat in ("image", "multi_image", "sticker", "video"):
                    _message = (
                        incoming_batch[i] if i < len(incoming_batch) else None
                    )
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
                            "context_relevance": (
                                0.85
                                if getattr(_sr, "has_text", False)
                                else 0.45
                            ),
                        }
                    )
            if _media_items:
                _budgeter = get_multimodal_budgeter()
                _budget_decisions = _budgeter.evaluate_batch(
                    _media_items, current_budget_ratio=_budget_ratio
                )
                self._cached_multimodal_summary = (
                    _budgeter.last_batch_summary()
                )
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

            self._cached_self_references = (
                get_self_reference_detector().detect_batch(incoming_batch)
            )
        except Exception as exc:
            self._cached_self_references = []
            logger.debug(f"{self.log_prefix} 自我指称检测异常: {exc}")
        try:
            from src.core.visibility_gate import get_visibility_gate

            _vg = get_visibility_gate(self.stream_id)
            # F22：构建关系偏置映射——好感/信任用户更容易被看见
            _rel_map = {}
            try:
                _rsnap = getattr(self, "_last_relation_snapshot", None) or {}
                if _rsnap:
                    _aff = float(_rsnap.get("affection", 50) or 50)
                    if _aff > 60:
                        _rel_level = 4
                    elif _aff > 35:
                        _rel_level = 3
                    elif _aff < 15:
                        _rel_level = 0
                    else:
                        _rel_level = 2
                    for _m in incoming_batch:
                        _uid = str(getattr(_m, "user_id", "") or "").strip()
                        if _uid and _uid != "bot":
                            _rel_map[_uid] = _rel_level
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            _loafing_val = 0.0
            try:
                if self._cached_metabolism_state:
                    _loafing_val = float(
                        getattr(
                            self._cached_metabolism_state, "loafing_level", 0.0
                        )
                        or 0.0
                    )
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
            from src.core.understanding_gate import get_understanding_gate

            _ug = get_understanding_gate()
            _contexts = []
            _skip_indices = set()
            _SKIP_CATEGORIES = ("recall",)
            _rel_snap = getattr(self, "_last_relation_snapshot", None) or {}
            _emotion_state = getattr(self, "_cached_self_state", None)
            _social_willingness = 0.5
            if _emotion_state:
                _social_willingness = float(getattr(_emotion_state, "social_desire", 0.5) or 0.5)
            for i, _m in enumerate(incoming_batch):
                _vis_score = 0.5
                if i < len(self._cached_visibility_decisions):
                    _vis_score = float(
                        getattr(
                            self._cached_visibility_decisions[i], "score", 0.5
                        )
                        or 0.5
                    )
                _cat = "plain_text"
                if i < len(self._cached_semantic_routes):
                    _sr = self._cached_semantic_routes[i]
                    _cat = (
                        _sr.category.value
                        if hasattr(_sr.category, "value")
                        else str(_sr.category)
                    )
                if _cat in _SKIP_CATEGORIES:
                    _skip_indices.add(i)
                    continue
                _text = str(getattr(_m, "processed_plain_text", "") or "")[
                    :500
                ]
                _uid = str(getattr(_m, "user_id", "") or "")
                _rel_level = 2
                if _uid and _uid in _rel_snap:
                    _rel_level = int(_rel_snap.get(_uid, {}).get("relationship_level", 2) or 2)
                _is_at = bool(getattr(_m, "is_at", False))
                _is_question = "?" in _text or "？" in _text or _text.endswith("吗") or _text.endswith("呢")
                _contexts.append(
                    {
                        "text_content": _text,
                        "message_category": _cat,
                        "visibility_score": _vis_score,
                        "bot_relevant": _is_at,
                        "is_question": _is_question,
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
                except Exception:
                    pass
            # 深度理解完成后消耗思考燃料
            try:
                from src.core.metabolism_engine import get_metabolism_engine

                _me = get_metabolism_engine(self.stream_id)
                _me.consume(
                    "think",
                    content_length=sum(
                        len(c.get("text_content", "")) for c in _contexts
                    ),
                )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        except Exception as exc:
            self._cached_understanding_results = []
            logger.debug(f"{self.log_prefix} 理解门控异常: {exc}")
        # 误解风险评估：检查用户后续消息是否暗示 bot 上轮回复被误解
        self._cached_misunderstanding_signal = None
        try:
            if self._recent_bot_utterances and incoming_batch:
                _last_bot = self._recent_bot_utterances[-1]
                _bot_text = str(_last_bot.get("text", "") or "")
                _bot_ts = float(_last_bot.get("ts", 0.0) or 0.0)
                if _bot_text and _bot_ts > 0:
                    # 找第一条非bot用户消息作为后续
                    _followup_text = ""
                    _followup_ts = 0.0
                    for _fm in incoming_batch:
                        _fu = str(getattr(_fm, "user_id", "") or "").strip()
                        if _fu and _fu != "bot":
                            _followup_text = str(
                                getattr(_fm, "processed_plain_text", "") or ""
                            )[:300]
                            _followup_ts = float(
                                getattr(_fm, "time", 0.0) or 0.0
                            )
                            break
                    if _followup_text:
                        _gap = (
                            max(0.0, _followup_ts - _bot_ts)
                            if _followup_ts > _bot_ts
                            else 60.0
                        )
                        # 简单重复提问检测：后续文本和bot回复前用户的上一条高度相似
                        _is_repeat = False
                        if len(self._recent_user_inputs) >= 2:
                            _prev_input = str(
                                self._recent_user_inputs[-2].get("text", "")
                                or ""
                            )
                            if (
                                _prev_input
                                and _followup_text
                                and len(_followup_text) > 3
                            ):
                                _overlap = sum(
                                    1
                                    for c in _followup_text[:50]
                                    if c in _prev_input[:50]
                                )
                                _is_repeat = (
                                    _overlap > len(_followup_text[:50]) * 0.6
                                )
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
                            if not hasattr(
                                self, "_last_misunderstanding_log_ts"
                            ):
                                self._last_misunderstanding_log_ts = 0.0
                            _now_m = time.time()
                            if (
                                _now_m - self._last_misunderstanding_log_ts
                            ) >= 30.0:
                                logger.info(
                                    f"{self.log_prefix} 误解风险={_signal.risk_score:.2f} "
                                    f"类型={_signal.signal_type} 线索={_signal.detected_cues[:2]}"
                                )
                                self._last_misunderstanding_log_ts = _now_m
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 误解风险评估异常: {exc}")

    def _apply_sensory_gates(
        self, incoming_batch: List, pinged_msg=None
    ) -> Optional[str]:
        """感官门控：基于可见性+理解度判定是否应跳过本轮处理
        返回 None 表示放行，返回 str 表示拦截原因"""
        if not incoming_batch:
            return None
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
                _msg_key = str(
                    getattr(_vd, "message_key", "")
                    or getattr(_vd, "msg_id", "")
                    or ""
                )
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
                _ukey = str(
                    getattr(_ur, "message_key", "")
                    or getattr(_ur, "msg_id", "")
                    or ""
                )
                if _cl_val == "cannot_understand":
                    _understand_none_count += 1
                    if _ukey and _ukey not in self._cached_exposure_tiers:
                        self._cached_exposure_tiers[_ukey] = "never_seen"
                elif _cl_val == "can_understand_no_reply":
                    _understand_no_reply_count += 1
                    if _ukey:
                        self._cached_exposure_tiers[_ukey] = (
                            "seen_engaged_no_reply"
                        )
        _total_user_msgs = sum(
            1 for m in incoming_batch if getattr(m, "user_id", "") != "bot"
        )
        if _total_user_msgs <= 0:
            return None
        _ignore_ratio = (
            _vis_ignored_count + _understand_no_reply_count
        ) / max(1, _total_user_msgs)
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
            _tier_str = f" | 三态:未看见={
                _tier_counts['never_seen']},扫到不理={
                _tier_counts['seen_ignored']},深度关注={
                _tier_counts['seen_engaged']},关注不回={
                _tier_counts['seen_engaged_no_reply']}"
            return f"可见性/理解门控: {_vis_ignored_count} 条被忽略+{
                _understand_no_reply_count} 条不回/{_total_user_msgs} 总消息{
                _tier_str} "
        if _vis_low_score_count >= _total_user_msgs and not _has_ping:
            return f"可见性全低分: {_vis_low_score_count}/{_total_user_msgs}条消息均未引起注意"
        # GAP-M：深度可见性评分——三维偏置+群环境调制+时间衰减
        try:
            from src.core.deep_visibility_scorer import (
                get_deep_visibility_scorer,
                UserBiasProfile,
                EventBiasInput,
                TopicBiasInput,
                GroupEnvModulation,
                TimeDecayFactor,
            )

            if not self._deep_scorer_initialized:
                self._deep_scorer_initialized = True
            _dvs = get_deep_visibility_scorer(self.stream_id)
            self._cached_deep_visibility_results = []
            for _msg in incoming_batch:
                _uid_m = str(getattr(_msg, "user_id", "") or "")
                if not _uid_m or _uid_m == "bot":
                    continue
                _rel_snap_m = (
                    getattr(self, "_last_relation_snapshot", None) or {}
                )
                _up_m = UserBiasProfile(
                    user_id=_uid_m,
                    affection=float(_rel_snap_m.get("affection", 50.0) or 50.0)
                    / 100.0,
                    trust=float(_rel_snap_m.get("trust_value", 50.0) or 50.0)
                    / 100.0,
                    annoyance=float(
                        _rel_snap_m.get("annoyance_value", 0.0) or 0.0
                    )
                    / 100.0,
                    is_avoided=bool(
                        float(_rel_snap_m.get("aversion_value", 0.0) or 0.0)
                        > 30
                    ),
                )
                _plain = str(
                    getattr(_msg, "processed_plain_text", "")
                    or getattr(_msg, "plain_text", "")
                    or getattr(_msg, "content", "")
                    or ""
                )
                _ei_m = EventBiasInput(
                    is_at_mention=getattr(_msg, "is_at", False),
                    is_quote_reply=bool(
                        getattr(_msg, "reply_to_message_id", None)
                    ),
                    message_length=len(_plain),
                )
                _ti_m = TopicBiasInput(message_text=_plain)
                _meta_state = (
                    getattr(self, "_cached_metabolism_constraints", None) or {}
                )
                _gm_m = GroupEnvModulation(
                    energy_ratio=1.0
                    - float(_meta_state.get("energy_suppression", 0.0) or 0.0),
                    social_willingness=0.5,
                    loafing_level=float(
                        _meta_state.get("loafing_suppression", 0.0) or 0.0
                    ),
                    boredom_level=float(
                        _meta_state.get("boredom_level", 0.0) or 0.0
                    ),
                )
                _td_m = TimeDecayFactor(
                    message_age_sec=max(
                        0.0,
                        time.time()
                        - float(
                            getattr(_msg, "timestamp", time.time())
                            or time.time()
                        ),
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
                _dv_high = sum(
                    1
                    for r in self._cached_deep_visibility_results
                    if r.modulated_score > 0.55
                )
                _dv_low = sum(
                    1
                    for r in self._cached_deep_visibility_results
                    if r.modulated_score < 0.12
                )
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
                _interruption_scores = self._classify_interruption_rights(
                    incoming_batch
                )
                _tier1_count = sum(
                    1 for s in _interruption_scores.values() if s < 0.30
                )
                _tier2_count = sum(
                    1
                    for s in _interruption_scores.values()
                    if 0.30 <= s < 0.65
                )
                _tier3_count = sum(
                    1 for s in _interruption_scores.values() if s >= 0.65
                )
                self._cached_interruption_tiers = _interruption_scores
                _loaf_level = 0.0
                if self._cached_metabolism_constraints:
                    _loaf_level = float(
                        self._cached_metabolism_constraints.get(
                            "loafing_suppression", 0.0
                        )
                        or 0.0
                    )
                if _loaf_level > 0.65:
                    if _tier3_count == 0:
                        return f"打断权T1全拦截: loafing={
                            _loaf_level: .2f}, T1={_tier1_count}  T2={
                            _tier2_count}  T3={_tier3_count} "
                    elif _tier3_count <= 1 and _tier1_count > _tier3_count * 3:
                        return f"打断权T1主导: loafing={
                            _loaf_level:.2f}, 仅{_tier3_count}条强优先级不足以拉起"
                elif _loaf_level > 0.35:
                    if (
                        _tier1_count >= _total_user_msgs * 0.8
                        and _tier3_count == 0
                    ):
                        return f"打断权半拦截: loafing={_loaf_level: .2f}, T1占比过高({
                            _tier1_count} /{_total_user_msgs})且无强优先级"
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        return None

    def _classify_interruption_rights(
        self, incoming_batch: List
    ) -> Dict[str, float]:
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
            _key = str(
                getattr(_msg, "message_key", "")
                or getattr(_msg, "msg_id", "")
                or id(_msg)
            )
            _score = 0.05
            # 因子1：@提及 / 引用回复
            _at_list = getattr(_msg, "at_list", None) or []
            _reply_to = str(getattr(_msg, "reply_to", "") or "").strip()
            if any(
                str(t).strip().lower()
                in ("bot", "robot", self.stream_id.lower())
                for t in _at_list
            ):
                _score += 0.42
            if _reply_to and len(_reply_to) > 0:
                _score += 0.18
            # 因子2：自我引用检测
            if self._cached_self_references:
                for _ref in self._cached_self_references:
                    _ref_msg_key = str(getattr(_ref, "message_key", "") or "")
                    if _ref_msg_key == _key or _ref_msg_key in (
                        _key[: min(16, len(_key))],
                    ):
                        _rt = getattr(_ref, "ref_type", None)
                        if _rt:
                            _rv = (
                                _rt.value
                                if hasattr(_rt, "value")
                                else str(_rt)
                            )
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
            _aff = (
                float(rel_snap.get("affection", 50.0) or 50.0)
                if isinstance(rel_snap, dict)
                else 50.0
            )
            if _aff > 60:
                _score += min(0.18, (_aff - 60.0) / 200.0)
            elif _aff < 18:
                _score -= 0.08
            # 因子4：话题相关性
            if self._cached_pattern_evidence:
                for _pe in self._cached_pattern_evidence:
                    _pe_conf = float(getattr(_pe, "confidence", 0.0) or 0.0)
                    _pe_pat = getattr(_pe, "pattern", None)
                    if _pe_pat:
                        _pv = (
                            _pe_pat.value
                            if hasattr(_pe_pat, "value")
                            else str(_pe_pat)
                        )
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
            _has_media = bool(
                getattr(_msg, "has_image", False)
                or getattr(_msg, "media_count", 0)
            )
            if _has_media:
                _score += 0.04
            scores[_key] = max(0.0, min(1.0, _score))
        return scores

    def _integrate_scene_tracking(self, incoming_batch: List) -> None:
        """群场景追踪 + 群体模式识别"""
        try:
            from src.core.group_scene_state import get_group_scene

            _gs = get_group_scene(self.stream_id)
            if self._cached_participant_summary:
                _known_active = int(
                    self._cached_participant_summary.get("hot_count", 0) or 0
                ) + int(
                    self._cached_participant_summary.get("warm_count", 0) or 0
                )
                _gs.set_known_online_count(_known_active)
            for _m in incoming_batch:
                _uid = str(getattr(_m, "user_id", "") or "").strip()
                _text = str(getattr(_m, "processed_plain_text", "") or "")[
                    :200
                ]
                _reply_anchor = str(getattr(_m, "reply_to", "") or "").strip()
                _mentioned_users = [
                    str(_target).strip()
                    for _target in (getattr(_m, "at_list", None) or [])
                    if str(_target).strip()
                ]
                if _uid:
                    _gs.record_message(
                        user_id=_uid,
                        text=_text,
                        mentioned_users=_mentioned_users,
                        reply_anchor=_reply_anchor,
                    )
            self._cached_scene_snapshot = _gs.snapshot()
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 群场景追踪异常: {exc}")
        try:
            from src.core.group_pattern_detector import (
                get_group_pattern_detector,
            )

            _gpd = get_group_pattern_detector(self.stream_id)
            if self._cached_participant_summary:
                _known_active = int(
                    self._cached_participant_summary.get("hot_count", 0) or 0
                ) + int(
                    self._cached_participant_summary.get("warm_count", 0) or 0
                )
                _gpd.set_known_active_count(_known_active)
            for _m in incoming_batch:
                _uid = str(getattr(_m, "user_id", "") or "").strip()
                _text = str(getattr(_m, "processed_plain_text", "") or "")[
                    :200
                ]
                _targets = [
                    str(_target).strip()
                    for _target in (getattr(_m, "at_list", None) or [])
                    if str(_target).strip()
                ]
                if _uid:
                    _gpd.feed_simple(
                        user_id=_uid,
                        text=_text,
                        is_at_someone=bool(_targets),
                        at_target=_targets[0] if _targets else "",
                    )
            self._cached_pattern_evidence = _gpd.detect()
        except Exception as exc:
            self._cached_pattern_evidence = []
            logger.debug(f"{self.log_prefix} 群体模式检测异常: {exc}")

    def _integrate_event_learning(self, incoming_batch: List) -> None:
        """事件驱动学习：从消息、贴纸和活跃话题持续吸收线索。"""
        self._cached_learning_summary = {}
        try:
            from src.core.learning_hub import get_learning_hub

            _hub = get_learning_hub(self.stream_id)
            _items: List[Dict[str, Any]] = []
            _seen_keys: set = set()
            for i, _m in enumerate(incoming_batch):
                _uid = str(getattr(_m, "user_id", "") or "").strip()
                if not _uid or _uid == "bot":
                    continue
                _text = str(getattr(_m, "processed_plain_text", "") or "")[
                    :120
                ]
                _route = (
                    self._cached_semantic_routes[i]
                    if i < len(self._cached_semantic_routes)
                    else None
                )
                _category = (
                    getattr(
                        getattr(_route, "category", None),
                        "value",
                        "plain_text",
                    )
                    if _route is not None
                    else "plain_text"
                )
                _summary = _hub.observe_message(
                    user_id=_uid,
                    text=_text,
                    semantic_category=str(_category or "plain_text"),
                )
                for _item in _summary.get("items", []):
                    _item_key = f"{
                        _item.get(
                            'type', '')}::{
                        _item.get(
                            'content', '')}"
                    if _item_key in _seen_keys:
                        continue
                    _seen_keys.add(_item_key)
                    _items.append(_item)
            try:
                from src.core.group_scene_state import get_group_scene

                _topic_rows = get_group_scene(
                    self.stream_id
                ).get_active_topics()
            except Exception:
                _topic_rows = []
            _topic_candidates = [
                str(_row.get("keyword", "")).strip()
                for _row in _topic_rows
                if isinstance(_row, dict)
                and int(_row.get("mention_count", 0) or 0) >= 2
            ][:3]
            if _topic_candidates:
                _topic_summary = _hub.observe_topic_keywords(
                    _topic_candidates, source_user="scene"
                )
                for _item in _topic_summary.get("items", []):
                    _item_key = f"{
                        _item.get(
                            'type', '')}::{
                        _item.get(
                            'content', '')}"
                    if _item_key in _seen_keys:
                        continue
                    _seen_keys.add(_item_key)
                    _items.append(_item)
            if _items:
                self._cached_learning_summary = {
                    "learned": True,
                    "hint": "；".join(
                        str(_item.get("content", "")) for _item in _items[:3]
                    ),
                    "count": len(_items),
                    "items": _items,
                    "types": sorted(
                        {
                            str(_item.get("type", ""))
                            for _item in _items
                            if _item.get("type")
                        }
                    ),
                }
            else:
                self._cached_learning_summary = _hub.last_summary()
        except Exception as exc:
            self._cached_learning_summary = {}
            logger.debug(f"{self.log_prefix} 事件驱动学习异常: {exc}")

    def _integrate_narration_planning(self, decision_messages: List) -> None:
        """旁白与行为策略规划"""
        try:
            from src.core.inner_narration_planner import get_narration_planner

            _np = get_narration_planner(self.stream_id)
            # 理解层级：5级判定
            _understanding_level = "can_understand_no_reply"
            if self._cached_understanding_results:
                try:
                    _primary = self._cached_understanding_results[0]
                    _cl = getattr(
                        getattr(_primary, "level", None), "value", ""
                    )
                    if _cl:
                        _understanding_level = _cl
                except Exception:
                    try:
                        from src.core.understanding_gate import (
                            get_understanding_gate,
                        )

                        if get_understanding_gate().any_worth_replying(
                            self._cached_understanding_results
                        ):
                            _understanding_level = "can_reply_simple"
                    except Exception:
                        pass
            _visibility_level = "noticed"
            if self._cached_visibility_decisions:
                for _vd in self._cached_visibility_decisions:
                    _vl = getattr(_vd, "level", None)
                    if _vl:
                        _vl_val = (
                            _vl.value if hasattr(_vl, "value") else str(_vl)
                        )
                        if _vl_val in ("focused", "understood_but_ignored"):
                            _visibility_level = _vl_val
                            break
            _watch_level = "skim_window"
            if self._cached_watch_level:
                _watch_level = (
                    self._cached_watch_level.value
                    if hasattr(self._cached_watch_level, "value")
                    else str(self._cached_watch_level)
                )
            _presence_mode = "一般"
            if self._cached_presence_state:
                _sw = float(
                    getattr(
                        self._cached_presence_state, "social_willingness", 0.5
                    )
                    or 0.5
                )
                if _sw > 0.7:
                    _presence_mode = "积极"
                elif _sw < 0.3:
                    _presence_mode = "退缩"
            _energy = 1.0
            if self._cached_metabolism_state:
                _energy = (
                    float(
                        getattr(
                            self._cached_metabolism_state, "chat_fuel", 100.0
                        )
                        or 100.0
                    )
                    / 100.0
                )
            _social_w = 0.5
            if self._cached_presence_state:
                _social_w = float(
                    getattr(
                        self._cached_presence_state, "social_willingness", 0.5
                    )
                    or 0.5
                )
            _recent_bot = [
                item.get("text", "")
                for item in self._recent_bot_utterances[-3:]
            ]
            _msg_summary = ""
            if decision_messages:
                _last = decision_messages[-1]
                _msg_summary = str(
                    getattr(_last, "processed_plain_text", "") or ""
                )[:100]
            # 自引用：提取最强引用类型和强度
            _is_at = False
            _self_ref_type = "none"
            _self_ref_strength = 0.0
            if self._cached_self_references:
                _best_strength = 0.0
                for _ref in self._cached_self_references:
                    _rt = getattr(_ref, "ref_type", None)
                    _rs = float(getattr(_ref, "strength", 0.0) or 0.0)
                    if _rt:
                        _rt_val = (
                            _rt.value if hasattr(_rt, "value") else str(_rt)
                        )
                        if _rs > _best_strength:
                            _best_strength = _rs
                            _self_ref_type = _rt_val
                        if _rt_val == "direct_at":
                            _is_at = True
                _self_ref_strength = _best_strength
            _is_question = bool(
                _msg_summary
                and (
                    _msg_summary.rstrip().endswith("?")
                    or _msg_summary.rstrip().endswith("？")
                )
            )
            # 主观存在判定
            _dominant_mood = ""
            _topic_interest = 0.0
            _want_learn = False
            if self._cached_presence_state:
                _dominant_mood = str(
                    getattr(
                        self._cached_presence_state, "mood_label", lambda: ""
                    )()
                    if callable(
                        getattr(
                            self._cached_presence_state, "mood_label", None
                        )
                    )
                    else ""
                )
                # 话题兴趣
                _main_topic = ""
                try:
                    from src.core.group_scene_state import get_group_scene

                    _scene_topics = get_group_scene(
                        self.stream_id
                    ).active_topics(limit=1)
                    if _scene_topics:
                        _main_topic = _scene_topics[0]
                except Exception:
                    pass
                if _main_topic and hasattr(
                    self._cached_presence_state, "topic_interest_score"
                ):
                    _topic_interest = (
                        self._cached_presence_state.topic_interest_score(
                            _main_topic
                        )
                    )
                # 学习优先判定
                if hasattr(
                    self._cached_presence_state, "should_learn_before_reply"
                ):
                    _has_unknown_meme = False
                    if self._cached_learning_summary:
                        _lh_types = self._cached_learning_summary.get(
                            "types", []
                        )
                        _has_unknown_meme = (
                            "meme_word" in _lh_types
                            and not self._cached_learning_summary.get(
                                "learned"
                            )
                        )
                    _want_learn = (
                        self._cached_presence_state.should_learn_before_reply(
                            comprehension_level=_understanding_level,
                            has_unknown_meme=_has_unknown_meme,
                        )
                    )
            # 缓存完整判定供面板使用
            self._cached_presence_verdict = {}
            if self._cached_presence_state and hasattr(
                self._cached_presence_state, "full_verdict_dict"
            ):
                self._cached_presence_verdict = (
                    self._cached_presence_state.full_verdict_dict(
                        comprehension_level=_understanding_level,
                        has_unknown_meme=_want_learn,
                        current_topic=(
                            _msg_summary[:20] if _msg_summary else ""
                        ),
                    )
                )
            # 代谢约束传递
            _metab_hint = self._cached_metabolism_constraints.get(
                "reply_length_hint", "normal"
            )
            _metab_perfunctory = bool(
                self._cached_metabolism_constraints.get(
                    "is_perfunctory", False
                )
            )
            _metab_gate = bool(
                self._cached_metabolism_constraints.get(
                    "energy_gate_open", True
                )
            )
            # 场景适合度与群体模式
            _scene_joinable = True
            if self._cached_scene_snapshot:
                _scene_joinable = bool(
                    getattr(
                        self._cached_scene_snapshot, "suitable_to_join", True
                    )
                )
            _dominant_pattern_name = ""
            if self._cached_pattern_evidence:
                _top_pat = self._cached_pattern_evidence[0]
                _pat_conf = float(getattr(_top_pat, "confidence", 0.0) or 0.0)
                if _pat_conf > 0.4:
                    _pat_enum = getattr(_top_pat, "pattern", None)
                    _dominant_pattern_name = (
                        (
                            _pat_enum.value
                            if hasattr(_pat_enum, "value")
                            else str(_pat_enum)
                        )
                        if _pat_enum
                        else ""
                    )
            # 语音/音乐消息检测
            _has_voice_music = False
            if self._cached_semantic_routes:
                for _route in self._cached_semantic_routes:
                    _cat_val = getattr(
                        getattr(_route, "category", None), "value", ""
                    )
                    if _cat_val in ("voice", "music_share"):
                        _has_voice_music = True
                        break
            # 创伤与烦躁值
            _trauma = 0.0
            _annoyance = 0.0
            _negative_emo_val = 0.0
            if self._last_relation_snapshot:
                _trauma = float(
                    self._last_relation_snapshot.get("trauma_score", 0.0)
                    or 0.0
                )
            if self._cached_metabolism_state:
                _annoyance = float(
                    getattr(
                        self._cached_metabolism_state, "channel_annoyance", 0.0
                    )
                    or 0.0
                )
            # 从情绪追踪器提取用户负面情绪聚合值
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import (
                    get_emotion_tracker, )

                _emo_tracker = get_emotion_tracker(self.stream_id)
                _target_uid = ""
                if decision_messages:
                    for _m in reversed(decision_messages):
                        _uid = str(getattr(_m, "user_id", "") or "").strip()
                        if _uid and _uid != "bot":
                            _target_uid = _uid
                            break
                if _target_uid:
                    _emo_state = _emo_tracker.get_user_state(
                        _target_uid, create_if_missing=False
                    )
                    if _emo_state:
                        _negative_emo_val = (
                            _emo_state.negative_emotion_aggregate
                        )
            except Exception:
                _negative_emo_val = 0.0
            self._cached_user_negative_emotion = _negative_emo_val
            # 从印象演化中枢提取用户个性标签
            _user_impression_tags = []
            try:
                from src.core.impression_evolution_hub import (
                    get_impression_hub,
                )

                _ih = get_impression_hub(self.stream_id)
                _tag_uid = ""
                if decision_messages:
                    for _m in reversed(decision_messages):
                        _tuid = str(getattr(_m, "user_id", "") or "").strip()
                        if _tuid and _tuid != "bot":
                            _tag_uid = _tuid
                            break
                if _tag_uid:
                    _layers = _ih.get_all_layers(_tag_uid)
                    if _layers:
                        _subj = _layers.get("subjective_tag", None)
                        if _subj and hasattr(_subj, "personality_tags"):
                            _user_impression_tags = list(
                                _subj.personality_tags or []
                            )
            except Exception:
                _user_impression_tags = []
            self._cached_user_impression_tags = _user_impression_tags
            self._cached_narration_plan = _np.plan(
                understanding_level=_understanding_level,
                visibility_level=_visibility_level,
                watch_level=_watch_level,
                presence_mode=_presence_mode,
                energy_ratio=_energy,
                social_willingness=_social_w,
                is_at_bot=_is_at,
                is_question=_is_question,
                recent_bot_texts=_recent_bot,
                message_summary=_msg_summary,
                dominant_mood=_dominant_mood,
                topic_interest_level=_topic_interest,
                want_learn_first=_want_learn,
                self_ref_type=_self_ref_type,
                self_ref_strength=_self_ref_strength,
                metabolism_reply_hint=_metab_hint,
                metabolism_is_perfunctory=_metab_perfunctory,
                metabolism_gate_open=_metab_gate,
                scene_suitable_to_join=_scene_joinable,
                dominant_pattern=_dominant_pattern_name,
                msg_has_voice_or_music=_has_voice_music,
                trauma_score=_trauma,
                channel_annoyance=_annoyance,
                misunderstanding_risk=(
                    float(
                        getattr(
                            self._cached_misunderstanding_signal,
                            "risk_score",
                            0.0,
                        )
                        or 0.0
                    )
                    if self._cached_misunderstanding_signal
                    else 0.0
                ),
                user_negative_emotion=_negative_emo_val,
                user_impression_tags=_user_impression_tags,
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 旁白策略规划异常: {exc}")

    def _integrate_memory_reactivation(self, incoming_batch: List) -> None:
        """记忆激活门控：话题匹配 + 用户回归触发"""
        try:
            from src.core.memory_reactivation_gate import get_reactivation_gate
            from src.core.subjective_time_stream import get_time_stream

            _mrg = get_reactivation_gate(self.stream_id)
            _ts = get_time_stream(self.stream_id)
            _topics: List[str] = []
            if self._cached_scene_snapshot:
                _topic_count = int(
                    getattr(
                        self._cached_scene_snapshot, "active_topic_count", 0
                    )
                    or 0
                )
                if _topic_count > 0:
                    try:
                        from src.core.group_scene_state import get_group_scene

                        _topics = get_group_scene(
                            self.stream_id
                        ).active_topics(limit=3)
                    except Exception:
                        pass
            if _topics:
                _candidates = _mrg.evaluate_topic_trigger(_topics, _ts)
                for _c in _candidates:
                    _mid = getattr(_c, "message_id", "") or ""
                    _reason = getattr(_c, "reason", "") or ""
                    if _mid:
                        _ts.reactivate(_mid, boost=0.3, reason=_reason)
            _returning_uids: set = set()
            for _m in incoming_batch:
                _uid = str(getattr(_m, "user_id", "") or "").strip()
                if _uid and _uid != "bot":
                    _returning_uids.add(_uid)
            for _uid in _returning_uids:
                _mrg.evaluate_user_return_trigger(_uid, _ts)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 记忆激活门控异常: {exc}")

    def _integrate_post_reply_learning(
        self, decision_messages: List, reply_text: str = ""
    ) -> None:
        """回复后学习 + 技能追踪 + 印象记录 + 代谢消耗"""
        _target_uid = str(getattr(self, "_last_user_id", "") or "").strip()
        try:
            from src.core.impression_evolution_hub import get_impression_hub

            _ih = get_impression_hub(self.stream_id)
            if _target_uid:
                imp = _ih.record_interaction(
                    _target_uid, quality=1, is_positive=True
                )
                _computed_aff = 0.0
                if imp and hasattr(imp, "truth"):
                    _pos = getattr(imp.truth, "positive_interactions", 0) or 0
                    _shared = (
                        getattr(imp.truth, "shared_experience_count", 0) or 0
                    )
                    _computed_aff = min(100.0, _pos * 2.0 + _shared * 4.0)
                if _computed_aff > 0:
                    try:
                        from src.modules.modcore.dynamic_persona.emotion_tracker import (
                            get_emotion_tracker, )

                        _et = get_emotion_tracker(self.stream_id)
                        _user_state = _et.get_user_state(_target_uid)
                        if _user_state:
                            _old_aff = (
                                getattr(_user_state, "affection", 0.0) or 0.0
                            )
                            _target_aff = _computed_aff / 100.0
                            if abs(_target_aff - _old_aff) > 0.01:
                                _et.update_affection(
                                    _target_uid,
                                    _target_aff - _old_aff,
                                    "印象演化同步",
                                )
                    except Exception:
                        pass
                # 群环境联动演化（替代不存在的 evolve 方法）
                _atmo = ""
                if self._cached_scene_snapshot and hasattr(
                    self._cached_scene_snapshot, "atmosphere"
                ):
                    _atmo_enum = self._cached_scene_snapshot.atmosphere
                    _atmo = (
                        _atmo_enum.value
                        if hasattr(_atmo_enum, "value")
                        else str(_atmo_enum)
                    )
                _density = 0.0
                if self._cached_participant_summary:
                    _hot = int(
                        self._cached_participant_summary.get("hot_count", 0)
                        or 0
                    )
                    _warm = int(
                        self._cached_participant_summary.get("warm_count", 0)
                        or 0
                    )
                    _density = min(1.0, (_hot + _warm) / 20.0)
                _pat_type = ""
                if self._cached_pattern_evidence:
                    _dom = self._cached_pattern_evidence[0]
                    _pat_type = str(getattr(_dom, "pattern_name", "") or "")
                _ih.evolve_with_group_context(
                    _target_uid, _atmo, _density, _pat_type
                )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 印象演化异常: {exc}")
        try:
            from src.core.learning_hub import (
                get_learning_hub,
                LearningEvent,
                LearnableType,
            )

            _lh = get_learning_hub(self.stream_id)
            for _m in decision_messages[-3:]:
                _text = str(
                    getattr(_m, "processed_plain_text", "") or ""
                ).strip()
                _uid = str(getattr(_m, "user_id", "") or "").strip()
                if _text and _uid and _uid != "bot":
                    _lh.observe(
                        LearningEvent(
                            event_type=LearnableType.REACTION_PATTERN,
                            content=_text[:100],
                            source_user=_uid,
                        )
                    )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 学习中心观察异常: {exc}")
        # 学习反馈闭环：已学习的梗/礼仪在回复中被使用时记录
        try:
            from src.core.learning_hub import get_learning_hub

            _lh = get_learning_hub(self.stream_id)
            if reply_text and self._cached_learning_summary:
                _learned_items = self._cached_learning_summary.get("items", [])
                for _item_name in _learned_items[:5]:
                    if (
                        isinstance(_item_name, str)
                        and _item_name in reply_text
                    ):
                        _lh.record_usage(_item_name, success=True)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 学习使用反馈异常: {exc}")
        try:
            from src.core.skill_lifecycle_hub import get_skill_lifecycle_hub

            _slh = get_skill_lifecycle_hub(self.stream_id)
            _slh.record_invocation(
                "conversation_reply", success=bool(reply_text)
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 技能生命周期追踪异常: {exc}")
        try:
            from src.core.metabolism_engine import get_metabolism_engine

            _me = get_metabolism_engine(self.stream_id)
            # F18+F19：围观/吃瓜模式用观察成本，敷衍模式降低消耗强度
            _cost_behavior = "reply"
            _cost_intensity = 1.0
            if self._cached_pattern_evidence:
                for _pe in self._cached_pattern_evidence:
                    _pe_val = (
                        getattr(getattr(_pe, "pattern", None), "value", "")
                        or ""
                    )
                    if (
                        _pe_val == "spectator_mode"
                        and float(getattr(_pe, "confidence", 0) or 0) > 0.4
                    ):
                        _cost_behavior = "group_observe"
                        break
            if bool(
                self._cached_metabolism_constraints.get(
                    "is_perfunctory", False
                )
            ):
                _cost_intensity = 0.6
            _me.consume(
                _cost_behavior,
                content_length=len(reply_text) if reply_text else 0,
                intensity=_cost_intensity,
            )
            self._cached_metabolism_state = _me.state
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 代谢引擎消耗异常: {exc}")

    def _register_builtin_skills(self) -> None:
        """首次运行时将内建技能注册到生命周期中心"""
        if self._skills_registered:
            return
        try:
            from src.core.skill_lifecycle_hub import (
                get_skill_lifecycle_hub,
                SkillDomain,
                SkillCategory,
            )

            _hub = get_skill_lifecycle_hub(self.stream_id)
            _builtin_catalog = [
                (
                    "conversation_reply",
                    "对话回复",
                    SkillCategory.CONVERSATION,
                    SkillDomain.RUNTIME,
                ),
                (
                    "deep_understanding",
                    "深度语义理解",
                    SkillCategory.KNOWLEDGE,
                    SkillDomain.ANALYSIS,
                ),
                (
                    "narration_planning",
                    "旁白策略规划",
                    SkillCategory.CONVERSATION,
                    SkillDomain.RUNTIME,
                ),
                (
                    "message_routing",
                    "消息语义路由",
                    SkillCategory.TASK_ASSIST,
                    SkillDomain.ANALYSIS,
                ),
                (
                    "emotion_tracking",
                    "情绪存在追踪",
                    SkillCategory.EMOTIONAL_SUPPORT,
                    SkillDomain.RUNTIME,
                ),
                (
                    "pattern_detection",
                    "群体模式识别",
                    SkillCategory.KNOWLEDGE,
                    SkillDomain.ANALYSIS,
                ),
                (
                    "learning_ingest",
                    "学习素材采集",
                    SkillCategory.KNOWLEDGE,
                    SkillDomain.DEV,
                ),
                (
                    "memory_association",
                    "记忆关联激活",
                    SkillCategory.KNOWLEDGE,
                    SkillDomain.RUNTIME,
                ),
            ]
            for _sid, _sname, _cat, _dom in _builtin_catalog:
                _hub.register_skill(
                    _sid, skill_name=_sname, category=_cat, domain=_dom
                )
            self._skills_registered = True
            logger.debug(
                f"{self.log_prefix} 内建技能注册完成 共{len(_builtin_catalog)}项"
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 内建技能注册异常: {exc}")

    def _integrate_skill_lifecycle_review(self) -> None:
        """周期性审查技能生命周期状态，注册内建技能并执行晋升/降级评估"""
        self._register_builtin_skills()
        if self._enhanced_round % self._skill_review_round_interval != 0:
            return
        try:
            from src.core.skill_lifecycle_hub import get_skill_lifecycle_hub

            _hub = get_skill_lifecycle_hub(self.stream_id)
            _review = _hub.periodic_review()
            self._cached_skill_review = _review
            _summary = _hub.summary()
            _changed = (
                _review.get("promoted", 0)
                + _review.get("declined", 0)
                + _review.get("retired", 0)
            )
            if _changed > 0:
                logger.info(
                    f"{self.log_prefix} 技能审查 晋升={_review['promoted']} "
                    f"衰退={_review['declined']} 退役={_review['retired']} "
                    f"总计={_summary.get('total_skills', 0)}"
                )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 技能周期审查异常: {exc}")

    def _track_skill_invocations(self) -> None:
        """根据本轮缓存结果批量记录各子系统技能的调用情况"""
        try:
            from src.core.skill_lifecycle_hub import get_skill_lifecycle_hub

            _hub = get_skill_lifecycle_hub(self.stream_id)
            # 深度理解技能
            if self._cached_understanding_results:
                _primary = self._cached_understanding_results[0]
                _level_val = getattr(
                    getattr(_primary, "level", None), "value", ""
                )
                _understood = _level_val in ("high", "medium", "contextual")
                _eff = 0.65 if _understood else 0.25
                _hub.record_invocation(
                    "deep_understanding",
                    success=_understood,
                    effectiveness=_eff,
                )
            # 旁白策略技能
            if self._cached_narration_plan:
                _hub.record_invocation(
                    "narration_planning", success=True, effectiveness=0.55
                )
            # 消息路由技能
            if self._cached_semantic_routes:
                _hub.record_invocation(
                    "message_routing", success=True, effectiveness=0.50
                )
            # 群体模式识别
            if self._cached_pattern_evidence:
                _dom = self._cached_pattern_evidence[0]
                _pat_conf = float(getattr(_dom, "confidence", 0.0) or 0.0)
                _hub.record_invocation(
                    "pattern_detection",
                    success=(_pat_conf > 0.3),
                    effectiveness=min(1.0, _pat_conf + 0.1),
                )
            # 情绪存在态
            if self._cached_presence_state:
                _hub.record_invocation(
                    "emotion_tracking", success=True, effectiveness=0.50
                )
            # 学习素材采集
            if (
                self._cached_learning_summary
                and self._cached_learning_summary.get("items")
            ):
                _item_count = len(self._cached_learning_summary["items"])
                _hub.record_invocation(
                    "learning_ingest",
                    success=True,
                    effectiveness=min(1.0, 0.3 + _item_count * 0.1),
                )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 技能调用追踪异常: {exc}")

    def _integrate_adaptive_collection(
        self,
        incoming_batch: List,
        reply_success: bool,
        reply_text: str = "",
    ) -> None:
        """自适应管线收集：理解空白/回复失败/弱模式归入事件池，周期性提出候选并执行验证发布"""
        try:
            from src.core.adaptive_update_pipeline import (
                get_adaptive_pipeline,
                CandidateType,
            )

            _pipe = get_adaptive_pipeline(self.stream_id)
            # 理解空白事件收集
            if self._cached_understanding_results:
                for _ur in self._cached_understanding_results:
                    _lv = getattr(getattr(_ur, "level", None), "value", "")
                    if _lv in ("low", "none", ""):
                        _snippet = str(
                            getattr(_ur, "original_text", "") or ""
                        )[:60]
                        _pipe.collect_event(
                            {
                                "type": "understanding_gap",
                                "summary": f"理解不足: {_snippet}",
                                "level": _lv,
                                "round": self._enhanced_round,
                            }
                        )
            # 回复失败收集
            if not reply_success and incoming_batch:
                _last_content = ""
                for _m in reversed(incoming_batch):
                    _t = str(
                        getattr(_m, "processed_plain_text", "") or ""
                    ).strip()
                    if _t:
                        _last_content = _t[:80]
                        break
                _pipe.collect_failure(
                    {
                        "type": "reply_failure",
                        "reason": f"收到消息但未能回复: {_last_content}",
                        "round": self._enhanced_round,
                    }
                )
            # 弱模式信号收集
            if self._cached_pattern_evidence:
                _dom_pat = self._cached_pattern_evidence[0]
                _pc = float(getattr(_dom_pat, "confidence", 0.0) or 0.0)
                _pn = str(getattr(_dom_pat, "pattern_name", "") or "")
                if _pc < 0.3 and _pn:
                    _pipe.collect_event(
                        {
                            "type": "weak_pattern",
                            "summary": f"弱模式信号: {_pn}(confidence={_pc:.2f})",
                            "round": self._enhanced_round,
                        }
                    )
            # 被误解信号收集
            if (
                self._cached_misunderstanding_signal
                and self._cached_misunderstanding_signal.risk_score > 0.3
            ):
                _ms = self._cached_misunderstanding_signal
                _pipe.collect_event(
                    {
                        "type": "misunderstanding_detected",
                        "summary": f"被误解({_ms.signal_type}): 风险={_ms.risk_score:.2f} 线索={','.join(_ms.detected_cues[:2])}",
                        "round": self._enhanced_round,
                    }
                )
            # 周期性候选提案（每20轮执行一次）
            if self._enhanced_round % 20 == 0:
                _evt_count = len(_pipe._event_pool)
                _fail_count = len(_pipe._failure_pool)
                # 事件池积累足够时自动提案
                if _evt_count >= 5 or _fail_count >= 3:
                    _candidate = _pipe.propose_candidate(
                        candidate_type=CandidateType.BEHAVIOR_RULE,
                        title=f"轮次{self._enhanced_round}自动行为优化",
                        description=f"基于{_evt_count}事件/{_fail_count}失败的自动优化提案",
                        expected_improvement="减少理解空白与回复缺失场景",
                        estimated_tokens=500,
                        estimated_skill_calls=2,
                    )
                    if _candidate:
                        _pass_threshold = _evt_count >= 3
                        _pipe.verify_candidate(
                            _candidate.candidate_id,
                            checker_name="auto_threshold",
                            passed=_pass_threshold,
                            detail=f"事件池={_evt_count} 失败池={_fail_count}",
                            score=min(1.0, _evt_count / 10.0),
                        )
                        if _pass_threshold:
                            _pipe.stage_candidate(_candidate.candidate_id)
                        logger.info(
                            f"{self.log_prefix} 自适应候选提案 id={_candidate.candidate_id} "
                            f"验证={'通过' if _pass_threshold else '未达阈值'}"
                        )
                # 发布通过暂存验证的候选
                for _staged_cand in _pipe.get_staged():
                    if _staged_cand.all_checks_passed():
                        _pipe.publish_candidate(_staged_cand.candidate_id)
                        logger.info(
                            f"{self.log_prefix} 自适应候选发布 id={_staged_cand.candidate_id}"
                        )
                # 清理过期候选
                _purged = _pipe.purge_old_candidates(max_age_days=3.0)
                if _purged > 0:
                    logger.debug(f"{self.log_prefix} 清理过期候选 {_purged}个")
            self._cached_pipeline_summary = _pipe.summary()
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 自适应管线收集异常: {exc}")

    def _integrate_impression_periodic_maintenance(self) -> None:
        """周期性印象维护：重写消除偏差 + 自动昵称生成"""
        if self._enhanced_round % 15 != 0:
            return
        try:
            from src.core.impression_evolution_hub import get_impression_hub

            _ih = get_impression_hub(self.stream_id)
            # 周期性全量重写（消除增量累积偏差）
            _rewritten = _ih.periodic_rewrite_all()
            if _rewritten > 0:
                logger.info(f"{self.log_prefix} 印象周期重写 {_rewritten}人")
            # 为最近交互用户生成自动昵称
            _target_uid = str(getattr(self, "_last_user_id", "") or "").strip()
            if _target_uid:
                _nick = _ih.auto_generate_nickname(_target_uid)
                if _nick:
                    logger.debug(
                        f"{self.log_prefix} 自动昵称生成 {_target_uid[:8]}→{_nick[:10]}"
                    )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 印象周期维护异常: {exc}")

    # ═══════════════════════════════════════════════════
    #  核心重写：增强版主循环体
    # ═══════════════════════════════════════════════════

    async def _loopbody(self):
        """七阶段增强循环体，完全替代基础版"""
        self._tick_world_snapshot = None
        self._resolved_emo_state = None
        self._resolved_emo_uid = ""
        self._enhanced_round += 1
        now = time.time()
        _t0 = now

        # 预构建频道级世界快照（无目标用户），供 Phase 0/1 门控优先读取
        try:
            from src.core.world_snapshot import build_world_snapshot

            self._tick_world_snapshot = await build_world_snapshot(
                self.stream_id
            )
        except Exception as _snap_pre:
            logger.debug(
                f"{self.log_prefix} 世界快照预构建失败，门控退回直接读源: {_snap_pre}"
            )

        # ── 阶段 0-a：夜间节律行为倾向评估 ──
        _night_verdict = self._integrate_night_cycle(now)
        if _night_verdict in ("sleep", "burned_out"):
            _phase_obj = getattr(self, "_cached_night_phase", None)
            _nc_obj = None
            try:
                from src.core.night_cycle_system import get_night_cycle

                _nc_obj = get_night_cycle(self.stream_id)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            if _nc_obj and _phase_obj:
                from src.core.night_cycle_system import NightPhase

                if _phase_obj == NightPhase.DEEP_SLEEP:
                    self._night_reply_suppression = 0.95
                    logger.info(
                        f"{self.log_prefix} 🌙 深睡模式: 回复抑制95%，让模型自己决定是否挣扎着回"
                    )
                elif _phase_obj == NightPhase.LIGHT_SLEEP:
                    self._night_reply_suppression = 0.70
                    logger.info(f"{self.log_prefix} 🌙 浅睡模式: 回复抑制70%")
                elif _night_verdict == "burned_out":
                    self._night_reply_suppression = 0.90
                    logger.info(f"{self.log_prefix} 🔥 熬穿模式: 回复抑制90%")
                else:
                    self._night_reply_suppression = 0.60
            else:
                self._night_reply_suppression = 0.80

        # ── 阶段 0-a2：技能生命周期周期审查 ──
        self._integrate_skill_lifecycle_review()

        # ── 阶段 0-a3：印象周期维护（重写 + 自动昵称） ──
        self._integrate_impression_periodic_maintenance()

        # ── 阶段 0-b：关注层级衰减（无消息期） ──
        self._integrate_watch_state(now, has_messages=False)

        # ── 阶段 0：状态机门控 ──
        current_phase = self._query_flow_phase()
        gate_result = self._evaluate_phase_gate(current_phase, now)
        if gate_result == "skip":
            self._consecutive_skip_ticks += 1
            await asyncio.sleep(_DORMANT_POLL_SEC)
            return True
        if gate_result == "glance":
            # 休息期窥屏：只消耗少量体力，不进入完整流水线
            self._apply_glance_drain()
            self._consecutive_skip_ticks = 0
            await asyncio.sleep(_DORMANT_POLL_SEC)
            return True

        # ── 阶段 1：能量评估门控 ──
        eagerness_val, eagerness_reason = self._poll_eagerness()
        self._log_energy_status(eagerness_val, eagerness_reason)
        # 主观存在态 + 代谢引擎恢复
        self._integrate_presence_metabolism(eagerness_val, now)
        # 更新自主代理上下文
        silence_sec = 0.0
        try:
            from src.chat.proactive.silence_watcher import get_quiet_monitor

            silence_sec = get_quiet_monitor().measure_silence_sec(
                self.stream_id
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        if eagerness_val < _ENERGY_DRAIN_FLOOR:
            # 能量严重不足，迁移至休息
            self._shift_to_dormant(cause=f"能量耗尽: {eagerness_reason}")
            await asyncio.sleep(_DORMANT_POLL_SEC)
            return True

        # ── 第一层状态栏：独立体征面板（仅在有新消息时输出，避免刷屏） ──
        phase_display = (
            current_phase
            if isinstance(current_phase, str)
            else str(current_phase)
        )

        # ── 阶段 2：消息读取（复用父类逻辑的核心部分） ──
        incoming_batch = message_api.get_messages_by_time_in_chat(
            chat_id=self.stream_id,
            start_time=self.last_read_time,
            end_time=now,
            limit=20,
            limit_mode="latest",
            filter_mai=True,
            filter_command=False,
            filter_intercept_message_level=0,
        )
        # ── 消息去重：过滤掉本轮已经处理过的消息（防止同一消息被多次处理） ──
        _new_batch = []
        for _msg in incoming_batch:
            _mid = (
                getattr(_msg, "message_id", None)
                or getattr(_msg, "id", None)
                or str(_msg)
            )
            if _mid and _mid not in self._processed_message_ids:
                _new_batch.append(_msg)
                if len(self._processed_message_ids) > 2000:
                    self._processed_message_ids = set(
                        list(self._processed_message_ids)[-1000:]
                    )
                self._processed_message_ids.add(_mid)
        incoming_batch = _new_batch
        # ── 去重后重新计算动态阈值 ──
        if self._batch_contains_real_user_reply(incoming_batch):
            self._mark_user_replied()
        dynamic_threshold = self._compute_dynamic_threshold()
        if len(incoming_batch) < dynamic_threshold:
            self._consecutive_skip_ticks += 1
            _should_emit_status = False
            if (
                self._backend_status_log_enabled
                and (now - self._last_backend_status_ts)
                >= self._backend_status_interval
            ):
                _should_emit_status = True
            elif (now - self._last_backend_status_ts) >= 10.0:
                _should_emit_status = True
            if _should_emit_status:
                _status_line = self._get_dashboard_status_line() or ""
                if (
                    _status_line
                    and _status_line != self._last_dashboard_status_line
                ):
                    logger.info(f"{self.log_prefix} 📊 {_status_line}")
                    self._last_dashboard_status_line = _status_line
                elif not _status_line and self._backend_status_log_enabled:
                    logger.info(
                        f"{self.log_prefix} 💤 "
                        f"静默={silence_sec:.0f}s 意愿={eagerness_val:.2f} "
                        f"消息={len(incoming_batch)}/{dynamic_threshold} 跳过={self._consecutive_skip_ticks}"
                    )
                self._last_backend_status_ts = now
            # ── 空闲主动路径：无新消息时尝试主动行为 ──
            idle_acted = await self._try_idle_proactive(now, silence_sec)
            if idle_acted:
                self._consecutive_skip_ticks = 0
                return True
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True

        self._consecutive_skip_ticks = 0
        self.last_read_time = now
        # ── 预构建仪表盘（用上一tick缓存数据，确保体征面板的📊状态行有内容输出） ──
        self._build_state_dashboard(force=False)
        # ── 第一层状态栏：有新消息时输出体征面板 ──
        self._emit_vitals_gauge(eagerness_val, eagerness_reason, phase_display)
        logger.info(
            f"{self.log_prefix} 🔄 读到 {len(incoming_batch)} 条消息，进入完整管线"
        )

        # ── 阶段 2-a：核心模块消息路由 + 活跃名册 + 时间流 + 多模态预算 ──
        self._integrate_message_routing(incoming_batch)
        # 关注层级升级（有消息到达）
        self._integrate_watch_state(now, has_messages=True)

        pinged_msg = self._find_pinged_message(incoming_batch)

        # GAP-F：观看状态机完整转移表 —— 在BLACKOUT阻断前先评估是否应转换状态
        _watch_transition = self._evaluate_watch_full_transitions(
            now=now,
            has_ping=pinged_msg is not None,
            incoming_batch=incoming_batch,
        )
        if _watch_transition:
            logger.debug(
                f"{self.log_prefix} 观看状态转移: {_watch_transition}"
            )

        # F2：黑屏态阻断——BLACKOUT 时只做最轻量追踪，跳过完整管线
        if self._cached_watch_level is not None:
            _wl_val = (
                self._cached_watch_level.value
                if hasattr(self._cached_watch_level, "value")
                else str(self._cached_watch_level)
            )
            if _wl_val == "blackout":
                _has_forced_ping = pinged_msg is not None
                if not _has_forced_ping:
                    logger.debug(
                        f"{self.log_prefix} 🚫 黑屏态阻断完整管线，仅保留心跳追踪"
                    )
                    await asyncio.sleep(_TICK_FLOOR_SEC * 3)
                    return True
                else:
                    logger.debug(
                        f"{self.log_prefix} 🚫 黑屏态但被@提及，降级放行"
                    )
            elif _wl_val == "peek":
                if pinged_msg is None:
                    # 窥屏态：概率性调用内心独白，让模型自主决定是否升级参与
                    peek_verdict = await self._run_peek_with_reflection(
                        incoming_batch
                    )
                    if peek_verdict is None or not getattr(
                        peek_verdict, "is_valid", False
                    ):
                        logger.info(
                            f"{self.log_prefix} 👁 窥屏态观察完成，内心无波澜"
                        )
                        return True
                    desire = getattr(peek_verdict, "reply_desire_level", 0)
                    thought = getattr(peek_verdict, "thought_text", "") or ""
                    mood = getattr(peek_verdict, "current_mood", "") or ""
                    _peek_action = str(
                        getattr(peek_verdict, "next_action", "") or ""
                    ).strip().lower()
                    if thought:
                        logger.info(
                            f"{self.log_prefix} 💭 窥屏想法={thought[:40]} | "
                            f"欲望={desire} 情绪={mood} 行为={_peek_action}"
                        )
                    # 由模型自主决策是否升级到完整管线
                    _upgrade_actions = {"reply", "followup"}
                    if _peek_action in _upgrade_actions:
                        logger.info(
                            f"{self.log_prefix} 👁→👀 窥屏态升级：模型决定{_peek_action}，进入完整管线"
                        )
                        self._cached_voice = peek_verdict
                        self._peek_mode_active = False
                    elif _peek_action in {"observe", "wait"} or (not _peek_action and desire >= 6):
                        # 模型选择观察/等待，或旧版兼容(无next_action时用desire)
                        if not _peek_action and desire >= 6:
                            logger.info(
                                f"{self.log_prefix} 👁→👀 窥屏态升级(兼容)：欲望={desire}，进入完整管线"
                            )
                            self._cached_voice = peek_verdict
                            self._peek_mode_active = False
                        else:
                            logger.info(
                                f"{self.log_prefix} 👁 窥屏态维持：模型选择{_peek_action or 'observe'}，继续观察"
                            )
                            return True
                    else:
                        logger.info(
                            f"{self.log_prefix} 👁 窥屏态维持：欲望={desire}，继续观察"
                        )
                        return True
                else:
                    logger.debug(
                        f"{self.log_prefix} 👁 窥屏态但被@提及，降级放行"
                    )
            elif _wl_val == "skim_window":
                self._peek_mode_active = True
                logger.debug(f"{self.log_prefix} 🔍 扫屏态：降低参与度")
            elif _wl_val == "active_watch":
                self._peek_mode_active = False
            elif _wl_val == "engaged":
                self._peek_mode_active = False

        # 用户消息到达时重置连续发言计数
        has_user_message = any(
            getattr(m, "user_id", "") != "bot" for m in incoming_batch
        )
        if has_user_message and self._consecutive_speaks > 0:
            _old = self._consecutive_speaks
            self._consecutive_speaks = max(0.0, self._consecutive_speaks - 0.4)
            logger.debug(
                f"{self.log_prefix} 用户消息，连续发言计数衰减 {_old:.1f}->{self._consecutive_speaks:.1f}"
            )
        if has_user_message:
            self._legacy_constraint_hits = max(
                0, self._legacy_constraint_hits - 1
            )

        # 更新用户消息累积追踪器
        now_ts = time.time()
        user_msgs_in_batch = sum(
            1 for m in incoming_batch if getattr(m, "user_id", "") != "bot"
        )
        if user_msgs_in_batch > 0:
            self._user_msg_counter += user_msgs_in_batch
            self._user_msg_timeline.append(now_ts)
            if len(self._user_msg_timeline) > 50:
                self._user_msg_timeline = self._user_msg_timeline[-50:]
            self._last_user_msg_time = now_ts

            burst_window = 60.0
            recent_msgs = sum(
                1 for t in self._user_msg_timeline if now_ts - t < burst_window
            )
            if recent_msgs >= 5:
                if not self._is_in_burst:
                    self._is_in_burst = True
                    self._burst_start_time = now_ts
                    logger.info(
                        f"{self.log_prefix} 🔥 检测到用户刷屏 burst_mode=True 消息数={recent_msgs}"
                    )
                self._burst_user_count = recent_msgs
            else:
                if (
                    self._is_in_burst
                    and (now_ts - self._burst_start_time) > 120.0
                ):
                    self._is_in_burst = False
                    self._burst_user_count = 0
                    logger.info(
                        f"{self.log_prefix} 🔥 刷屏结束 burst_mode=False"
                    )

            self._update_dynamic_ratio(now_ts)
        # 用户回复时完成针对该用户的等待意图
        if has_user_message:
            try:
                from src.chat.proactive.intention_pool import (
                    get_intention_pool,
                    IntentKind,
                )

                for _m in incoming_batch:
                    _uid = str(getattr(_m, "user_id", "") or "").strip()
                    if _uid and _uid != "bot":
                        get_intention_pool().complete_by_target(
                            self.stream_id,
                            _uid,
                            IntentKind.WAIT_FOR_REPLY,
                        )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

        # 消息到达时立即记录行为与氛围
        self._record_message_behavior(incoming_batch)
        self._record_interaction_mood(incoming_batch)
        await self._apply_input_state_updates(incoming_batch)

        # 重建完整世界快照（携带目标用户ID），供 Phase 2.5+ 所有门控使用
        _target_uid = ""
        for _m_snap in reversed(incoming_batch):
            _uid_snap = getattr(_m_snap, "user_id", "") or ""
            if _uid_snap and _uid_snap != "bot":
                _target_uid = _uid_snap
                break
        if _target_uid:
            try:
                from src.core.world_snapshot import build_world_snapshot

                self._tick_world_snapshot = await build_world_snapshot(
                    self.stream_id, _target_uid
                )
            except Exception as _snap_full:
                logger.debug(
                    f"{self.log_prefix} 完整快照重建失败: {_snap_full}"
                )

        # ── 阶段 2.5：核心系统集成（并行化 Group A → Group B → 同步收尾） ──
        _t25 = time.time()
        # 身份锚点 + 动态上下文 + 自回复识别 + 内容状态追踪（互不依赖，并行）
        identity_context, _, self_reply_risk, _ = await asyncio.gather(
            self._check_identity_context(incoming_batch),
            self._update_dynamic_context(incoming_batch),
            self._check_self_reply_risk(incoming_batch),
            self._track_content_state(incoming_batch),
        )
        # 记忆 + 情绪 + 创伤 + 群聊感知 + 预处理 + 风格学习 + 会话追踪（可并行）
        (
            _,  # memory
            _,  # emotion
            _,  # trauma
            group_sense_result,
            preprocessor_signal,
            _,  # style
            _,  # memoir
        ) = await asyncio.gather(
            self._store_interaction_memory(incoming_batch),
            self._update_emotion_tracker_state(incoming_batch),
            self._update_trauma_system_state(incoming_batch),
            self._analyze_group_sense(incoming_batch),
            self._analyze_message_preprocessor(incoming_batch),
            self._update_user_interaction_styles(incoming_batch),
            self._sync_memoir_on_message(incoming_batch),
        )
        logger.info(
            f"{self.log_prefix} 🔄 阶段2.5完成 {time.time() - _t25:.2f}s"
        )

        # ── 阶段 2.5-b：核心模块深度理解 + 群场景 + 记忆激活 ──
        _t25b = time.time()
        self._integrate_deep_understanding(incoming_batch)
        self._integrate_scene_tracking(incoming_batch)
        self._integrate_event_learning(incoming_batch)
        self._integrate_memory_reactivation(incoming_batch)
        _now_af = time.time()
        self._run_attention_flow_tick(_now_af)
        self._integrate_gossip_ritual_strategy(_now_af)
        logger.debug(
            f"{self.log_prefix} 阶段2.5b(深度集成) {time.time() - _t25b:.2f}s"
        )

        # ── 阶段 2.5-bis：群体模式硬路由 + 群场景硬约束 + 夜间节律 ──
        _now_hr = time.time()
        _night_block = self._apply_night_cycle_modulation(_now_hr)
        if _night_block:
            logger.info(f"{self.log_prefix} 🌙 夜间节律拦截: {_night_block}")
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True
        _pattern_route = self._apply_pattern_based_routing(
            incoming_batch, pinged_msg=pinged_msg
        )
        if _pattern_route and (
            "observe" in _pattern_route or "silence" in _pattern_route
        ):
            logger.info(
                f"{self.log_prefix} 🚧 群体模式硬路由拦截: {_pattern_route}"
            )
            await self._run_peek_observe_loop(incoming_batch)
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True
        _scene_block = self._apply_scene_hard_constraints(
            pinged_msg=pinged_msg
        )
        if _scene_block and (
            "unsuitable" in _scene_block or "spam" in _scene_block
        ):
            logger.info(
                f"{self.log_prefix} 🚫 群场景硬约束拦截: {_scene_block}"
            )
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True

        # ── 阶段 2.5-c：感官门控（可见性 + 理解度 → 提供参考信息，不硬性拦截） ──
        _gate_skip_reason = self._apply_sensory_gates(
            incoming_batch, pinged_msg=pinged_msg
        )
        if _gate_skip_reason:
            logger.info(
                f"{self.log_prefix} 🔇 感官门控建议跳过: {_gate_skip_reason} (但交给内心独白决策)"
            )
            self._gate_suggestion = _gate_skip_reason
        else:
            self._gate_suggestion = None
        self._integrate_deep_visibility_and_freshness(incoming_batch)
        for _rm in incoming_batch:
            _uid_rm = str(getattr(_rm, "user_id", "") or "")
            if not _uid_rm or _uid_rm == "bot":
                continue
            _plain_rm = str(
                getattr(_rm, "processed_plain_text", "")
                or getattr(_rm, "plain_text", "")
                or getattr(_rm, "content", "")
                or ""
            )
            if len(_plain_rm.strip()) > 1:
                self.register_message_to_memory(_plain_rm[:200], _uid_rm)
        _now_cycle = time.time()
        self._integrate_memory_governance(_now_cycle)
        self._run_adaptive_pipeline_periodic(_now_cycle)
        self.run_emotion_feedback_cycle(_now_cycle)

        # ── 统一状态仪表盘：三层触发架构入口 ──
        _dashboard_snap = self._build_state_dashboard(force=True)
        if _dashboard_snap:
            _dash_verdict = _dashboard_snap.get("verdict", {})
            _dash_urgency = str(_dash_verdict.get("urgency", "") or "")
            # L1硬阻断检查：精力耗尽/深睡/安全封锁 → 直接跳过
            if _dash_verdict.get("process") is False:
                logger.info(
                    f"{self.log_prefix} 🛑 仪表盘L1阻断: {_dash_verdict.get('reason', '未知')} "
                    f"urgency={_dash_urgency}"
                )
                await asyncio.sleep(_TICK_FLOOR_SEC * 2)
                return True

        # 用户过滤：同步即可
        _t_prep = time.time()
        filtered_messages = self._filter_messages_by_user_preference(
            incoming_batch
        )

        # 检查是否被 @ 或提及（已在门控前计算）
        decision_messages = self._prepare_decision_messages(
            incoming_batch,
            filtered_messages,
            pinged_msg,
        )
        content_state_signal = self._evaluate_content_state_signal(
            decision_messages
        )
        repetition_signal = self._analyze_repetition_pressure(
            decision_messages
        )
        harassment_signal = self._analyze_harassment_pressure(
            decision_messages
        )
        decision_context_packet = self._build_decision_context_packet(
            decision_messages,
            repetition_signal=repetition_signal,
        )
        behavior_signal = self._classify_behavior_signal(
            decision_messages,
            repetition_signal=repetition_signal,
            harassment_signal=harassment_signal,
        )
        await self._apply_repetition_emotion_feedback(
            decision_messages, repetition_signal
        )
        await self._prefetch_memory_hint(decision_messages)
        if repetition_signal.get("detected"):
            logger.info(
                f"{self.log_prefix} 重复输入提示 {repetition_signal.get('reason', '当前内容重复度偏高')}"
            )
        if harassment_signal.get("detected"):
            logger.info(
                f"{self.log_prefix} 骚扰强度提示 {harassment_signal.get('reason', '当前输入可能带持续冒犯倾向')}"
            )

        message_salience = self._evaluate_message_salience(
            decision_messages,
            pinged_msg,
            repetition_signal,
            behavior_signal,
        )
        force_reply_message = None
        # 低信息重复标记：阻止下游旁路强制回复
        _low_info_repeat_block = False
        if (
            repetition_signal.get("detected")
            and repetition_signal.get("latest_matches_repeat")
            and repetition_signal.get("low_info_cluster")
        ):
            # 低信息重复内容（如反复发同一短句）应被忽略，不设置强制回复
            _low_info_repeat_block = True
            logger.info(
                f"{self.log_prefix} 🔕 低信息重复检测: 抑制所有强制回复旁路"
            )

        # ... 其他代码保持不变 ...

        # ── 阶段 2.9：旁白策略规划（基于前序模块结果） ──
        self._integrate_narration_planning(decision_messages)
        logger.debug(
            f"{self.log_prefix} 决策准备阶段 {time.time() - _t_prep:.2f}s"
        )

        # ── 阶段 2.9b：技能调用追踪（本轮各子系统均已执行完毕） ──
        self._track_skill_invocations()

        # ── 阶段 3+4+4.5：感知、内心独白、关系度 —— 并行 ──
        _t345 = time.time()
        ambient_info = self._sample_channel_ambient()
        run_perception = self._should_run_perception(now)
        run_voice = self._should_run_voice(now, incoming_batch, pinged_msg)
        logger.info(
            f"{self.log_prefix} 🔄 阶段3+4启动 感知={run_perception} 独白={run_voice}"
        )

        # 组装并行任务列表
        _parallel_tasks: list = []
        _task_keys: list = []

        if run_perception:
            _parallel_tasks.append(
                self._invoke_perception(decision_messages, now)
            )
            _task_keys.append("perception")
        if run_voice:
            _parallel_tasks.append(
                self._invoke_inner_voice(
                    decision_messages,
                    self._cached_awareness,
                    ambient_info,
                    now,
                )
            )
            _task_keys.append("voice")
        _parallel_tasks.append(
            self._compute_relation_metrics(decision_messages)
        )
        _task_keys.append("relation")

        try:
            _parallel_results = await asyncio.wait_for(
                asyncio.gather(*_parallel_tasks, return_exceptions=True),
                timeout=20.0,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"{self.log_prefix} 🔄 阶段3+4 LLM调用超时(45s)，跳过本轮"
            )
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True
        _result_map = dict(zip(_task_keys, _parallel_results))
        # 检查是否有异常
        for _k, _v in _result_map.items():
            if isinstance(_v, BaseException):
                logger.warning(f"{self.log_prefix} 🔄 阶段3+4 {_k} 异常: {_v}")
        logger.info(
            f"{self.log_prefix} 🔄 阶段3+4完成 {time.time() - _t345:.2f}s"
        )

        # 拆包感知结果
        awareness_snapshot = self._cached_awareness
        if run_perception:
            _pval = _result_map.get("perception")
            if isinstance(_pval, BaseException):
                logger.warning(
                    f"{self.log_prefix} 感知异常: {_pval}，沿用缓存或空字典"
                )
                if awareness_snapshot is None:
                    awareness_snapshot = {}
            elif _pval is not None:
                awareness_snapshot = _pval
                self._cached_awareness = awareness_snapshot
                self._last_perception_ts = now
        if awareness_snapshot is None:
            awareness_snapshot = {}

        # 拆包内心独白结果
        voice_conclusion = self._cached_voice
        if run_voice:
            _vval = _result_map.get("voice")
            if _vval is not None and not isinstance(_vval, BaseException):
                voice_conclusion = _vval
                self._cached_voice = voice_conclusion
                self._last_voice_ts = now
                # 根据独白想法长度推导思考复杂度
                _thought_len = len(str(getattr(voice_conclusion, "thinking", "") or ""))
                _think_complexity = max(0.5, min(2.0, _thought_len / 40.0))
                self._apply_think_drain(complexity=_think_complexity)
                voice_summary = self._build_voice_execution_summary(
                    voice_conclusion,
                    decision_messages[-1] if decision_messages else None,
                )
                if voice_summary.get("log"):
                    logger.info(
                        f"{self.log_prefix} 💭 {voice_summary['log'][:160]}"
                    )
                # 将内心独白意图写入跨轮意图池
                try:
                    from src.chat.proactive.intention_pool import (
                        get_intention_pool,
                    )

                    _speaker_id = ""
                    if decision_messages:
                        _speaker_id = str(
                            getattr(decision_messages[-1], "user_id", "") or ""
                        ).strip()
                    get_intention_pool().ingest_voice_verdict(
                        channel_id=self.stream_id,
                        verdict=voice_conclusion,
                        speaker_id=_speaker_id,
                    )
                except Exception as _ipool_err:
                    logger.warning(
                        f"{self.log_prefix} 意图池写入失败: {_ipool_err}"
                    )
        try:
            from src.core.inner_narration_planner import InnerNarrationPlanner

            _narration_key = f"narration_planner_{self.stream_id}"
            _inp = getattr(self, "_narration_planner_cache", {}).get(
                _narration_key
            )
            if _inp is None:
                _inp = InnerNarrationPlanner(self.stream_id)
                if not hasattr(self, "_narration_planner_cache"):
                    self._narration_planner_cache = {}
                self._narration_planner_cache[_narration_key] = _inp
            _vis_level = str(
                getattr(self, "_cached_dominant_visibility", "noticed")
                or "noticed"
            )
            _watch_lv = str(
                getattr(self, "_cached_watch_level", "skim_window")
                or "skim_window"
            )
            _understanding = (
                "can_understand"
                if self._cached_understanding_results
                else "cannot_understand"
            )
            _energy_ratio = 1.0
            try:
                from src.chat.heart_flow.energy_manager import (
                    get_vitality_pool_manager,
                )

                _vm = get_vitality_pool_manager(self.stream_id)
                _vsnap = _vm.capture_snapshot()
                _energy_ratio = (
                    (_vsnap.chat_energy / max(0.01, _vsnap.chat_ceiling))
                    if _vsnap
                    else 1.0
                )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            _scene_joinable = True
            if self._cached_scene_snapshot:
                _scene_joinable = bool(
                    getattr(
                        self._cached_scene_snapshot, "suitable_to_join", True
                    )
                )
            _dom_pat_name = ""
            if self._cached_pattern_evidence:
                _ptop = self._cached_pattern_evidence[0]
                _pconf = float(getattr(_ptop, "confidence", 0.0) or 0.0)
                if _pconf > 0.35:
                    _penum = getattr(_ptop, "pattern", None)
                    if _penum is not None:
                        _dom_pat_name = getattr(_penum, "value", "") or str(
                            _penum
                        )
            _trauma_val = float(
                getattr(self, "_cached_trauma_score", 0.0) or 0.0
            )
            _annoyance_val = float(
                getattr(self, "_cached_annoyance", 0.0) or 0.0
            )
            _nplan = _inp.plan(
                understanding_level=_understanding,
                visibility_level=_vis_level,
                watch_level=_watch_lv,
                energy_ratio=_energy_ratio,
                scene_suitable_to_join=_scene_joinable,
                dominant_pattern=_dom_pat_name,
                trauma_score=_trauma_val,
                channel_annoyance=_annoyance_val,
                message_summary=(
                    str(getattr(pinged_msg, "content", "") or "")[:200]
                    if pinged_msg
                    else ""
                ),
            )
            self._cached_narration_plan = _nplan
            _nprompt_block = _nplan.to_prompt_block()
            if _nprompt_block:
                logger.debug(
                    f"{self.log_prefix} 📝 叙事规划: {_nprompt_block[:120]}"
                )
        except Exception as _narr_err:
            logger.debug(f"{self.log_prefix} 叙事规划器异常: {_narr_err}")

        # 拆包关系度结果
        _rval = _result_map.get("relation")
        relation_result = (
            _rval
            if (_rval is not None and not isinstance(_rval, BaseException))
            else {
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
                "trust_score": 0.0,
                "psychological_pressure": 0.0,
                "trauma_score": 0.0,
                "mood": "平静",
                "attribute_influences": {},
            }
        )
        relation_result["behavior_signal"] = behavior_signal

        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            tracker = get_emotion_tracker(self.stream_id)
            if hasattr(self, "_last_user_id") and self._last_user_id:
                state = tracker.get_user_state(self._last_user_id)
                if state:
                    _tracker_annoyance = float(
                        getattr(state, "annoyance", 0.0) or 0.0
                    )
                    _tracker_annoyance_acc = float(
                        getattr(state, "annoyance_accumulated", 0.0) or 0.0
                    )
                    if _tracker_annoyance > 0 or _tracker_annoyance_acc > 0:
                        relation_result["annoyance_value"] = max(
                            _tracker_annoyance, _tracker_annoyance_acc
                        )
                        logger.info(
                            f"{
                                self.log_prefix} 🎭 情绪追踪器同步: 厌烦={
                                _tracker_annoyance:.1f} 累积={
                                _tracker_annoyance_acc:.1f}"
                        )
                    self._resolved_emo_state = state
                    self._resolved_emo_uid = str(
                        self._last_user_id or ""
                    ).strip()
        except Exception as _e:
            logger.warning(f"{self.log_prefix} 情绪状态解析失败: {_e}")

        self._last_relation_snapshot = relation_result
        self._log_relation_metrics(relation_result)
        try:
            from src.modules.social_value.social_affect_fuser import (
                get_social_affect_fuser,
            )

            _fuser = get_social_affect_fuser()
            _mod_annoyance, _mod_pressure = (
                _fuser.apply_group_modulation_to_emotion(
                    channel_id=self.stream_id,
                    current_annoyance=float(
                        relation_result.get("annoyance_value", 0.0) or 0.0
                    ),
                    current_pressure=float(
                        relation_result.get("psychological_pressure", 0.0)
                        or 0.0
                    ),
                    recent_messages=incoming_batch,
                    bot_user_id="bot",
                )
            )
            if _mod_annoyance != float(
                relation_result.get("annoyance_value", 0.0) or 0.0
            ):
                relation_result["annoyance_value"] = _mod_annoyance
                logger.debug(
                    f"{self.log_prefix} 🎭 群环境调制: 厌烦值 → {_mod_annoyance:.1f}"
                )
            if _mod_pressure != float(
                relation_result.get("psychological_pressure", 0.0) or 0.0
            ):
                relation_result["psychological_pressure"] = _mod_pressure
                logger.debug(
                    f"{self.log_prefix} 🎭 群环境调制: 压力值 → {_mod_pressure:.1f}"
                )
        except Exception as _gmod_exc:
            logger.debug(f"{self.log_prefix} 群环境调制异常: {_gmod_exc}")

        self._cached_affection_value = float(
            relation_result.get("affection", 0.0) or 0.0
        )
        self._cached_trust_value = float(
            relation_result.get(
                "trust_value", relation_result.get("trust_score", 0.0)
            )
            or 0.0
        )
        self._cached_annoyance_value = float(
            relation_result.get("annoyance_value", 0.0) or 0.0
        )
        self._cached_pressure_value = float(
            relation_result.get("psychological_pressure", 0.0) or 0.0
        )
        self._cached_relationship_level = int(
            relation_result.get("relationship_level", 2) or 2
        )
        self._cached_familiarity = float(
            relation_result.get("familiarity", 0.0) or 0.0
        )
        self._cached_dominance = float(
            relation_result.get("dominance", 0.0) or 0.0
        )

        # ── 第二层状态栏：当前对象印象面板 ──
        _target_uid_for_panel = ""
        for _pm in reversed(decision_messages):
            _uid = getattr(_pm, "user_id", "") or ""
            if _uid and _uid != "bot":
                _target_uid_for_panel = _uid
                break
        if _target_uid_for_panel:
            self._last_user_id = _target_uid_for_panel
            await self._emit_target_profile(_target_uid_for_panel)

        # ── 阶段 4.6：早期退出检查 - 内心独白决定不想回复时直接跳过 ──
        early_exit = self._check_early_exit(
            voice_conclusion, pinged_msg, relation_result
        )
        if early_exit["should_skip"]:
            self._emit_action_verdict(
                "early_exit", early_exit["reason"], time.time() - _t0
            )
            logger.info(
                f"{
                    self.log_prefix} ⚡ {
                    early_exit['reason']}，跳过LLM调用 | 管线耗时 {
                    time.time() -
                    _t0:.2f}s"
            )
            await self._emit_outcome_summary(False, relation_result)
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True

        legacy_constraint = self._delegate_readiness_evaluation(
            decision_messages=decision_messages,
            pinged_msg=pinged_msg,
            relation_result=relation_result,
            voice_conclusion=voice_conclusion,
            repetition_signal=repetition_signal,
            harassment_signal=harassment_signal,
            message_salience=message_salience,
        )
        # ── 多维独立状态系统裁定 ──
        try:
            _gw_result = await self._run_dimension_gateway(
                decision_messages=decision_messages,
                pinged_msg=pinged_msg,
                voice_conclusion=voice_conclusion,
                relation_result=relation_result,
                repetition_signal=repetition_signal,
                harassment_signal=harassment_signal,
            )
            if _gw_result is not None:
                _gw_gate = str(_gw_result.get("gate", "allow") or "allow")
                # 维度网关的强制拒绝覆盖旧约束
                if _gw_gate == "block" and not legacy_constraint.get("should_skip"):
                    legacy_constraint["should_skip"] = True
                    legacy_constraint["gate"] = "block"
                    legacy_constraint["reason"] = str(
                        _gw_result.get("reason", "维度网关拦截") or "维度网关拦截"
                    )
                # 维度网关的强制回复提升旧约束
                elif _gw_gate == "force_reply" and legacy_constraint.get("should_skip"):
                    legacy_constraint["should_skip"] = False
                    legacy_constraint["gate"] = "force_reply"
                    legacy_constraint["reason"] = str(
                        _gw_result.get("reason", "维度网关强制回复") or "维度网关强制回复"
                    )
                logger.info(
                    f"{self.log_prefix} [维度网关] 旧约束gate="
                    f"{legacy_constraint.get('gate', 'allow')} "
                    f"网关gate={_gw_gate}"
                )
        except Exception as _gw_exc:
            logger.debug(f"{self.log_prefix} 维度网关异常（回退旧约束）: {_gw_exc}")
        legacy_gate = str(legacy_constraint.get("gate", "allow") or "allow")
        self._last_legacy_gate = legacy_gate

        # ── 多维信号融合补充裁决 ──
        _hf_judgment = self._query_heartflow_judgment(
            decision_messages, relation_result, voice_conclusion
        )
        if (
            _hf_judgment is not None
            and not _hf_judgment["should_respond"]
            and pinged_msg is None
        ):
            _hf_cert = _hf_judgment.get("certainty", "")
            if _hf_cert in ("FIRM", "RESOLUTE"):
                if legacy_gate in ("allow", "force_reply"):
                    legacy_gate = "hesitate"
                    legacy_constraint["should_skip"] = True
                    legacy_constraint["gate"] = "hesitate"
                    legacy_constraint["reason"] = (
                        f"多维信号否决(cert={_hf_cert}, "
                        f"comp={_hf_judgment.get('composite', 0):.2f}): "
                        f"{_hf_judgment.get('rationale', '')[:60]}"
                    )
                    self._last_legacy_gate = legacy_gate
                    logger.info(
                        f"{self.log_prefix} [心流决策] 六维信号否决回复 | "
                        f"composite={_hf_judgment.get('composite', 0):.2f} "
                        f"certainty={_hf_cert}"
                    )

        desire_level = 5
        if voice_conclusion is not None and hasattr(
            voice_conclusion, "reply_desire_level"
        ):
            desire_level = voice_conclusion.reply_desire_level
        try:
            from src.core.readiness_evaluator import (
                get_decision_arbiter,
                ArbiterInput,
                ReadinessVerdict,
                ReadinessBreakdown,
            )

            _arbiter = get_decision_arbiter()
            _voice_should_reply = (
                getattr(voice_conclusion, "should_reply", None)
                if voice_conclusion
                else None
            )
            _bd_raw = legacy_constraint.get("breakdown", {})
            _rv_for_arbiter = ReadinessVerdict(
                should_skip=legacy_constraint.get("should_skip", False),
                reason=str(legacy_constraint.get("reason", "") or ""),
                total_deduction=float(
                    legacy_constraint.get("reply_penalty", 0.0) or 0.0
                ),
                reply_readiness=float(
                    legacy_constraint.get("reply_readiness", 0.0) or 0.0
                ),
                gate=legacy_gate,
                breakdown=ReadinessBreakdown(
                    base=float(_bd_raw.get("base", 50.0) or 50.0),
                    mention_boost=float(
                        _bd_raw.get("mention_bonus", 0.0) or 0.0
                    ),
                    question_boost=float(
                        _bd_raw.get("question_bonus", 0.0) or 0.0
                    ),
                    salience_boost=float(
                        _bd_raw.get("salience_bonus", 0.0) or 0.0
                    ),
                    rapport_boost=float(
                        _bd_raw.get("rapport_bonus", 0.0) or 0.0
                    ),
                    resource_deduction=float(
                        _bd_raw.get("resource_penalty", 0.0) or 0.0
                    ),
                    relation_deduction=float(
                        _bd_raw.get("relation_penalty", 0.0) or 0.0
                    ),
                    repeat_deduction=float(
                        _bd_raw.get("repeat_penalty", 0.0) or 0.0
                    ),
                    harass_deduction=float(
                        _bd_raw.get("harass_penalty", 0.0) or 0.0
                    ),
                    distrust_deduction=float(
                        _bd_raw.get("distrust_penalty", 0.0) or 0.0
                    ),
                    desire_level=int(desire_level),
                    reply_readiness=float(
                        legacy_constraint.get("reply_readiness", 50.0) or 50.0
                    ),
                ),
            )
            _arb_input = ArbiterInput(
                readiness_verdict=_rv_for_arbiter,
                desire_level=desire_level,
                voice_should_reply=_voice_should_reply,
                is_direct_ping=(pinged_msg is not None),
                latest_text=str(
                    getattr(pinged_msg, "processed_plain_text", "")
                    or getattr(pinged_msg, "plain_text", "")
                    or getattr(pinged_msg, "content", "")
                    or ""
                ).strip(),
                repeated_rounds=max(
                    int(
                        (repetition_signal or {}).get("repeat_count", 0) or 0
                    ),
                    int(
                        (repetition_signal or {}).get(
                            "recent_repeat_count", 0
                        )
                        or 0
                    ),
                    int(
                        (repetition_signal or {}).get(
                            "exact_repeat_count", 0
                        )
                        or 0
                    ),
                ),
                repeated_pressure=max(
                    float(
                        (repetition_signal or {}).get(
                            "repeated_topic_pressure", 0.0
                        )
                        or 0.0
                    ),
                    float(
                        (repetition_signal or {}).get("pressure", 0.0)
                        or 0.0
                    ),
                ),
                repeat_detected=bool(
                    (repetition_signal or {}).get("repeat_detected", False)
                    or int(
                        (repetition_signal or {}).get(
                            "exact_repeat_count", 0
                        )
                        or 0
                    )
                    >= 1
                ),
            )
            _arb_result = _arbiter.arbitrate(_arb_input)
            if _arb_result.override_source != "readiness_passthrough":
                _old_gate = legacy_gate
                legacy_gate = _arb_result.final_gate
                self._last_legacy_gate = legacy_gate
                if _arb_result.should_skip and not legacy_constraint.get(
                    "should_skip"
                ):
                    legacy_constraint["should_skip"] = True
                    legacy_constraint["reason"] = _arb_result.reason

                elif not _arb_result.should_skip and legacy_constraint.get(
                    "should_skip"
                ):
                    # 仲裁器提升门控且不建议跳过 → 清除之前的should_skip
                    if legacy_gate in ("allow", "force_reply"):
                        legacy_constraint["should_skip"] = False
                        legacy_constraint["gate"] = legacy_gate
                logger.info(
                    f"{self.log_prefix} ⚖️ 仲裁: {_arb_result.override_source} "
                    f"| {_old_gate}→{legacy_gate} | {_arb_result.reason[:60]}"
                )
        except Exception as _arb_exc:
            logger.debug(
                f"{self.log_prefix} 仲裁器异常（回退原始gate）: {_arb_exc}"
            )

        if legacy_gate == "hesitate" and force_reply_message is None and not _low_info_repeat_block:
            force_reply_message = self._get_latest_human_message(
                decision_messages
            )
            if force_reply_message is not None:
                logger.info(
                    f"{self.log_prefix} 门控=犹豫，降级为低风险澄清回复"
                )
        elif legacy_gate == "hesitate" and _low_info_repeat_block:
            logger.info(
                f"{self.log_prefix} 门控=犹豫但低信息重复，不设置强制回复"
            )
        if legacy_constraint["should_skip"]:
            self._apply_readiness_constraint(
                legacy_constraint, has_user_message=has_user_message
            )
            verdict_action = (
                "legacy_hesitate"
                if legacy_gate == "hesitate"
                and force_reply_message is not None
                else "legacy_constraint"
            )
            self._emit_action_verdict(
                verdict_action, legacy_constraint["reason"], time.time() - _t0
            )
            if legacy_gate == "hesitate" and force_reply_message is not None:
                logger.info(
                    f"{
                        self.log_prefix} ⚠️ {
                        legacy_constraint['reason']}，转入低风险回复通道 | 管线耗时 {
                        time.time() -
                        _t0:.2f}s"
                )
            else:
                logger.info(
                    f"{
                        self.log_prefix} ⛔ {
                        legacy_constraint['reason']}，旧版式约束拦截 | 管线耗时 {
                        time.time() -
                        _t0:.2f}s"
                )
            if legacy_gate == "hesitate" and force_reply_message is not None:
                legacy_constraint["should_skip"] = False
            else:
                await self._emit_outcome_summary(False, relation_result)
                await asyncio.sleep(_POST_MESSAGE_RETRY_SEC)
                return True

        autonomy_guard = self._evaluate_autonomy_guard(
            decision_messages=decision_messages,
            pinged_msg=pinged_msg,
            identity_context=identity_context,
            self_reply_risk=self_reply_risk,
            group_sense_result=group_sense_result,
            preprocessor_signal=preprocessor_signal,
            relation_result=relation_result,
            voice_conclusion=voice_conclusion,
            message_salience=message_salience,
            content_state_signal=content_state_signal,
            repetition_signal=repetition_signal,
            harassment_signal=harassment_signal,
        )
        if autonomy_guard["should_skip"]:
            if autonomy_guard.get("source") == "content_state":
                self._apply_content_state_skip(content_state_signal)
            self._emit_action_verdict(
                "autonomy_block", autonomy_guard["reason"], time.time() - _t0
            )
            logger.info(
                f"{
                    self.log_prefix} 🚫 {
                    autonomy_guard['reason']}，本轮不回复 | 管线耗时 {
                    time.time() -
                    _t0:.2f}s"
            )
            await self._emit_outcome_summary(False, relation_result)
            await asyncio.sleep(_POST_MESSAGE_RETRY_SEC)
            return True

        # ── 阶段 4.7：内心独白强驱动检查 - 欲望等级高时直接回复 ──
        voice_driven_reply = False
        # F3：主观存在态压制——refusing 态强制压低欲望上限
        if self._cached_presence_state is not None:
            _pv = getattr(
                self._cached_presence_state, "participation_verdict", None
            )
            if _pv == "refusing":
                _original_desire = desire_level
                desire_level = min(desire_level, 2)
                if _original_desire > 2:
                    logger.debug(
                        f"{self.log_prefix} 🧊 存在态=拒绝参与，欲望{_original_desire}→压制为{desire_level}"
                    )
            elif _pv == "reluctant":
                if desire_level >= 5:
                    desire_level = 4
                    logger.debug(
                        f"{self.log_prefix} 🧊 存在态=不情愿，欲望≥5→降为4"
                    )
        # 独白降级为风格/紧迫度层：除明确提及外，不再单独决定是否直发
        if legacy_gate == "block":
            voice_driven_reply = False
            logger.info(
                f"{self.log_prefix} 门控=拦截，禁止独白强驱动穿透"
            )
        elif legacy_gate == "hesitate":
            voice_driven_reply = False
            logger.info(
                f"{self.log_prefix} 门控=犹豫，独白强驱动降级为低风险回复"
            )
        elif pinged_msg is not None:
            _model_wants_reply = getattr(
                voice_conclusion, "should_reply", None
            )
            if _model_wants_reply:
                voice_driven_reply = True
                logger.info(f"{self.log_prefix} 💭 被提及，模型决策=回复")
            else:
                logger.info(
                    f"{self.log_prefix} 💭 被提及，模型决策=不回，尊重意愿"
                )
        else:
            logger.info(
                f"{self.log_prefix} 💭 独白仅保留风格参考 desire={desire_level}"
            )
        _llm_call_level, _skip_score, _esc_score, _tentative = (
            self._compute_llm_call_level(
                now, incoming_batch, pinged_msg, desire_level
            )
        )

        _model_should_reply = (
            getattr(voice_conclusion, "should_reply", None)
            if voice_conclusion
            else None
        )
        _is_voice_reluctant = _model_should_reply is False
        _thinking_text = (
            getattr(voice_conclusion, "thinking", "") or ""
            if voice_conclusion
            else ""
        )
        # 处理 LLM 直接行为决策：rest / disengage / lurk
        _voice_action = (
            getattr(voice_conclusion, "next_action", "") or ""
            if voice_conclusion
            else ""
        )
        if _voice_action == "rest":
            logger.info(
                f"{self.log_prefix} LLM决定歇一会: '{_thinking_text[:40]}'"
            )
            try:
                from src.chat.heart_flow.rest_handler import DormancySupervisor
                _ds = DormancySupervisor.instance()
                _ds.begin_dormancy(self.stream_id, cause=f"内心想休息({_thinking_text[:30]})")
            except Exception as _rest_err:
                logger.debug(f"{self.log_prefix} 休息触发异常: {_rest_err}")
            # 同步视窗状态到 peek
            try:
                from src.core.watch_state_machine import get_watch_machine, WatchLevel
                _wm = get_watch_machine(self.stream_id)
                _wm.signal_force_level(WatchLevel.PEEK, trigger="内心想休息")
            except Exception:
                logger.debug(f"{self.log_prefix} 视窗状态同步异常(rest)")
            return
        if _voice_action == "disengage":
            logger.info(
                f"{self.log_prefix} LLM决定放下手机: '{_thinking_text[:40]}'"
            )
            try:
                from src.chat.heart_flow.rest_handler import DormancySupervisor
                _ds = DormancySupervisor.instance()
                _ds.begin_dormancy(self.stream_id, cause=f"放下手机({_thinking_text[:30]})")
            except Exception as _rest_err:
                logger.debug(f"{self.log_prefix} 放下手机触发异常: {_rest_err}")
            # 同步视窗状态到 blackout
            try:
                from src.core.watch_state_machine import get_watch_machine, WatchLevel
                _wm = get_watch_machine(self.stream_id)
                _wm.signal_force_level(WatchLevel.BLACKOUT, trigger="放下手机")
            except Exception:
                logger.debug(f"{self.log_prefix} 视窗状态同步异常(disengage)")
            return
        if _voice_action == "lurk":
            logger.info(
                f"{self.log_prefix} LLM决定潜水: '{_thinking_text[:40]}'"
            )
            # 同步视窗状态到 peek
            try:
                from src.core.watch_state_machine import get_watch_machine, WatchLevel
                _wm = get_watch_machine(self.stream_id)
                _cur = _wm.current_level
                _cur_val = _cur.value if hasattr(_cur, "value") else str(_cur)
                if _cur_val in ("active_watch", "engaged"):
                    _wm.signal_force_level(WatchLevel.PEEK, trigger="潜水")
            except Exception:
                logger.debug(f"{self.log_prefix} 视窗状态同步异常(lurk)")
            return

        _algo_wants_skip = _llm_call_level == 0
        _algo_wants_reply = _llm_call_level == 1
        _algo_wants_upgrade = _llm_call_level == 2

        _final_skip = False
        _final_reply = False
        _final_upgrade = False

        if _is_voice_reluctant and not pinged_msg:
            _takeover_tried = False
            _takeover_by_model = False
            try:
                _takeover_result = await self._try_algo_takeover_request(
                    now, incoming_batch, voice_conclusion, _thinking_text
                )
                if _takeover_result:
                    _takeover_tried = True
                    _takeover_by_model = _takeover_result.get(
                        "wants_to_takeover", False
                    )
                    if _takeover_by_model:
                        _final_reply = True
                        self._takeover_decision = _takeover_result.get(
                            "decision"
                        )
                        self._takeover_action = _takeover_result.get(
                            "action", "接管"
                        )
                        logger.info(
                            f"{self.log_prefix} 🎯 模型接管: 内心抗拒='{_thinking_text[:30]}...'，强制执行回复"
                        )
            except Exception as _takeover_err:
                logger.debug(
                    f"{self.log_prefix} 算法接管请求异常: {_takeover_err}"
                )
            if not _takeover_tried or not _takeover_by_model:
                _final_skip = True
                logger.info(
                    f"{self.log_prefix} 🎯 模型内心抗拒='{_thinking_text[:30]}...'，尊重意愿跳过"
                )
        elif _algo_wants_skip and pinged_msg:
            if _model_should_reply is False:
                _takeover_tried = False
                _takeover_by_model = False
                try:
                    _takeover_result = await self._try_algo_takeover_request(
                        now, incoming_batch, voice_conclusion, _thinking_text
                    )
                    if _takeover_result:
                        _takeover_tried = True
                        _takeover_by_model = _takeover_result.get(
                            "wants_to_takeover", False
                        )
                        if _takeover_by_model:
                            _final_reply = True
                            self._takeover_decision = _takeover_result.get(
                                "decision"
                            )
                            self._takeover_action = _takeover_result.get(
                                "action", "接管"
                            )
                            logger.info(
                                f"{self.log_prefix} 🎯 模型接管(被@但内心抗拒): '{_thinking_text[:30]}...'，强制执行回复"
                            )
                except Exception as _takeover_err:
                    logger.debug(
                        f"{self.log_prefix} 算法接管请求异常: {_takeover_err}"
                    )
                if not _takeover_tried or not _takeover_by_model:
                    _final_skip = True
                    logger.info(
                        f"{self.log_prefix} 🎯 被@但内心强烈抗拒='{_thinking_text[:30]}...'，尊重意愿跳过"
                    )
            else:
                _rel_for_mention = self._resolve_relation_view()
                _mention_annoy = float(
                    _rel_for_mention.get("annoyance_value", 0.0) or 0.0
                )
                _mention_press = float(
                    _rel_for_mention.get("psychological_pressure", 0.0) or 0.0
                )
                if _mention_annoy >= 55 or (
                    _mention_annoy >= 35 and _mention_press >= 40
                ):
                    _final_skip = True
                    logger.info(
                        f"{self.log_prefix} 🎯 被@但情绪门槛拦截 烦躁={_mention_annoy:.1f} 压力={_mention_press:.1f}，跳过"
                    )
                else:
                    _final_upgrade = True
                    logger.info(
                        f"{self.log_prefix} 🎯 被@且算法跳过，升级大模型决定"
                    )
        elif _algo_wants_skip and _algo_wants_reply:
            _final_reply = True
            logger.info(f"{self.log_prefix} 🎯 算法+小模型共识回复")
        elif _algo_wants_upgrade:
            _final_upgrade = True
            logger.info(f"{self.log_prefix} 🎯 算法升级大模型")
        else:
            _final_skip = True
            logger.info(
                f"{self.log_prefix} 🎯 默认跳过: algo={_llm_call_level}"
            )

        if (
            hasattr(self, "_reply_probability")
            and self._reply_probability > 0
            and _final_skip
        ):
            import random
            # 低信息重复/感官门控建议跳过时，禁止动态比例旁路
            _block_dynamic = (
                _low_info_repeat_block
                or getattr(self, "_gate_suggestion", None) is not None
            )
            if _block_dynamic:
                logger.info(
                    f"{self.log_prefix} 🎲 动态比例锁定: 低信息重复={_low_info_repeat_block}, "
                    f"感官门控={getattr(self, '_gate_suggestion', None) is not None}，禁止旁路"
                )
            elif random.random() > self._reply_probability:
                logger.info(
                    f"{self.log_prefix} 🎲 动态比例跳过: 回复概率={self._reply_probability:.2f} "
                    f"随机>{self._reply_probability:.2f}，跳过"
                )
            else:
                if getattr(voice_conclusion, "should_reply", False) or (
                    _takeover_by_model
                    if "_takeover_by_model" in dir()
                    else False
                ):
                    _final_skip = False
                    _final_reply = True
                    logger.info(
                        f"{self.log_prefix} 🎲 动态比例通过: 模型意愿优先，执行回复"
                    )

        if _final_skip:
            logger.info(f"{self.log_prefix} 🎯 最终决策: 跳过，直接观察")
            self._emit_action_verdict(
                "llm_skip", "算法+小模型联合判定跳过", time.time() - _t0
            )
            await self._run_peek_observe_loop(incoming_batch)
            await asyncio.sleep(_TICK_FLOOR_SEC)
            return True

        # ── 阶段 4.8：LLM 自主规划 - 仅在内心独白不确定时调用 ──
        _t48 = time.time()
        llm_decision = None
        if _final_upgrade:
            try:
                llm_decision = await asyncio.wait_for(
                    self._llm_autonomous_decide(
                        now,
                        relation_result,
                        decision_messages,
                        voice_conclusion,
                    ),
                    timeout=15.0,
                )
            except Exception as exc:
                logger.debug(f"{self.log_prefix} LLM自主决策异常: {exc}")
            logger.debug(
                f"{
                    self.log_prefix} 🔄 阶段4.8(LLM建议) {
                    time.time() -
                    _t48:.2f}s act={
                    llm_decision.should_act if llm_decision else '?'}"
            )
        llm_triggered = False
        planner_triggered = False
        planner_decision = None
        should_act = _final_reply

        if legacy_gate == "block":
            llm_triggered = False
            should_act = False
        elif legacy_gate == "hesitate":
            llm_triggered = False
            should_act = False

        if llm_decision:
            self._llm_content_plan = llm_decision.content_plan or ""
            self._llm_social_intention = llm_decision.social_intention or ""
            logger.debug(
                f"{self.log_prefix} LLM内容建议已接收: "
                f"意图={llm_decision.social_intention}, "
                f"感受={llm_decision.emotional_state}, "
                f"内容规划={llm_decision.content_plan[:50] if llm_decision.content_plan else '无'}"
            )
        elif not voice_driven_reply and llm_decision:
            logger.debug(
                f"{self.log_prefix} LLM建议仅作参考，不参与放行: 原因={llm_decision.reasoning}"
            )

        # 非内心强驱动时，统一规划器始终作为主决策层
        if not voice_driven_reply:
            if now < self._planner_quiet_until and not has_user_message:
                quiet_left = int(max(0.0, self._planner_quiet_until - now))
                self._emit_action_verdict(
                    "cooldown_skip",
                    f"规划器冷却中({quiet_left}s)",
                    time.time() - _t0,
                )
                logger.debug(
                    f"{self.log_prefix} 规划器冷却中({quiet_left}s)，跳过重复规划"
                )
                await self._emit_outcome_summary(False, relation_result)
                await asyncio.sleep(_POST_MESSAGE_RETRY_SEC)
                return True

            planner_decision = None
            if legacy_gate in {"allow", "force_reply", "hesitate"}:
                planner_decision = await self._invoke_unified_planner(
                    incoming_batch,
                    repetition_signal=repetition_signal,
                )
            if planner_decision and hasattr(planner_decision, "action"):
                _pact = (
                    planner_decision.action.value
                    if hasattr(planner_decision.action, "value")
                    else str(planner_decision.action)
                )
                if _pact in (
                    "reply",
                    "join_conversation",
                    "proactive_speak",
                    "deep_think",
                ):
                    planner_triggered = True
                elif _pact in ("rest", "check_later", "observe", "no_reply"):
                    cooldown = float(
                        getattr(
                            planner_decision,
                            "next_check_seconds",
                            _POST_MESSAGE_RETRY_SEC,
                        )
                        or _POST_MESSAGE_RETRY_SEC
                    )
                    cooldown = max(
                        _POST_MESSAGE_RETRY_SEC, min(cooldown, 30.0)
                    )
                    self._planner_quiet_until = now + cooldown
                elif _pact == "wait":
                    _anticipated = str(
                        getattr(planner_decision, "reason", "") or ""
                    )[:80]
                    self._shift_to_pending(
                        cause=f"planner决策wait: {_anticipated[:40]}",
                        anticipated=_anticipated,
                    )
                    cooldown = float(
                        getattr(
                            planner_decision,
                            "next_check_seconds",
                            _POST_MESSAGE_RETRY_SEC,
                        )
                        or _POST_MESSAGE_RETRY_SEC
                    )
                    cooldown = max(
                        _POST_MESSAGE_RETRY_SEC, min(cooldown, 30.0)
                    )
                    self._planner_quiet_until = now + cooldown
                logger.info(
                    f"{self.log_prefix} 规划器决策={_pact} 门控={legacy_gate} 应行动={should_act}"
                )

        _voice_reluctant = (
            (getattr(voice_conclusion, "should_reply", None) is False)
            if voice_conclusion
            else False
        )
        _intent_primary = None
        _intent_silence = 0
        if voice_conclusion:
            try:
                _intent_primary = str(
                    getattr(voice_conclusion, "primary_intent", None) or ""
                )
                _intent_silence = int(
                    getattr(voice_conclusion, "silence_rounds", 0) or 0
                )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        if not should_act and legacy_gate in {"allow", "force_reply"}:
            if _voice_reluctant:
                logger.info(
                    f"{self.log_prefix} ⚠️ 门控={legacy_gate}但独白显抗拒，降级为规划器决策"
                )
            else:
                _rel_snap = self._resolve_relation_view()
                _snap_annoyance = float(
                    _rel_snap.get("annoyance_value", 0.0) or 0.0
                )
                _snap_pressure = float(
                    _rel_snap.get("psychological_pressure", 0.0) or 0.0
                )
                _snap_blocked = bool(_rel_snap.get("is_user_blocked", False))

                if _snap_blocked or (
                    _snap_annoyance >= 80 and _snap_pressure >= 60
                ):
                    logger.info(
                        f"{self.log_prefix} 🛑 门控={legacy_gate}但情绪极端"
                        f"(烦躁{_snap_annoyance:.0f}/压力{_snap_pressure:.0f}"
                        f"{'/已屏蔽' if _snap_blocked else ''})，拒绝激活"
                    )
                    should_act = False
                elif _snap_annoyance >= 60 and _snap_pressure >= 40:
                    import random as _emo_random

                    if _emo_random.random() < 0.50:
                        logger.info(
                            f"{self.log_prefix} ⚠️ 门控={legacy_gate}但情绪偏高"
                            f"(烦躁{_snap_annoyance:.0f}/压力{_snap_pressure:.0f})，50%概率跳过"
                        )
                        should_act = False
                    else:
                        should_act = True
                        logger.info(
                            f"{self.log_prefix} 统一结算 门控={legacy_gate}，情绪偏高但允许进入回复执行层"
                        )
                else:
                    if _intent_primary == "wait" and _intent_silence >= 1:
                        import random as _int_random

                        _skip_prob = min(0.75, 0.35 + _intent_silence * 0.15)
                        if _int_random.random() < _skip_prob:
                            should_act = False
                            logger.info(
                                f"{self.log_prefix} 🤐 意图池primary=wait(静默{_intent_silence}轮)"
                                f"，{int(_skip_prob * 100):.0f}%概率跳过"
                            )
                        else:
                            should_act = True
                            logger.info(
                                f"{self.log_prefix} 统一结算 门控={legacy_gate}，意图wait但掷骰通过，允许回复"
                            )
                    else:
                        should_act = True
                        logger.info(
                            f"{self.log_prefix} 统一结算 门控={legacy_gate}，允许进入回复执行层"
                        )

        if (
            not should_act
            and force_reply_message is not None
            and not voice_driven_reply
            and legacy_gate != "block"
        ):
            if _low_info_repeat_block:
                logger.info(
                    f"{self.log_prefix} 🔕 低信息重复: 跳过澄清兜底"
                )
            elif _voice_reluctant:
                logger.info(
                    f"{self.log_prefix} ⚠️ force_reply_message存在但独白显抗拒，跳过澄清兜底"
                )
            else:
                try:
                    from src.chat.proactive.intention_pool import (
                        get_intention_pool,
                    )

                    _matched_intent_id = (
                        str(
                            getattr(planner_decision, "matched_intent_id", "")
                            or ""
                        ).strip()
                        if planner_decision
                        else ""
                    )
                    if _matched_intent_id:
                        if get_intention_pool().fail_intent(
                            _matched_intent_id, reason="interrupted"
                        ):
                            logger.info(
                                f"{self.log_prefix} 意图精确失败回写 id={_matched_intent_id[:10]} reason=interrupted"
                            )
                except Exception:
                    pass
                _voice_desire = int(
                    getattr(voice_conclusion, "reply_desire_level", 5) or 5
                )
                if _voice_reluctant or _voice_desire <= 2:
                    logger.info(
                        f"{self.log_prefix} ⛔ force_reply存在但旁白强烈抗拒(desire={_voice_desire},reluctant={_voice_reluctant})，跳过澄清兜底"
                    )
                else:
                    should_act = True
                    planner_triggered = True
                    logger.info(
                        f"{self.log_prefix} 触发澄清兜底，执行一次引导式回复"
                    )

        # SOC-03: 负面情绪影响决策 — 高负面情绪时触发回避/防御
        _rel_snap_soc = self._resolve_relation_view()
        _neg_emo = float(_rel_snap_soc.get("annoyance_value", 0.0) or 0.0)
        _soc_pressure = float(
            _rel_snap_soc.get("psychological_pressure", 0.0) or 0.0
        )
        _neg_composite = max(_neg_emo, _soc_pressure * 0.8)
        _is_force = (force_reply_message is not None) or voice_driven_reply
        if should_act and _neg_composite > 85 and not _is_force:
            # 极高负面情绪：除非被强制/被@force，主动回避不回复
            should_act = False
            logger.info(
                f"{self.log_prefix} SOC-03 负面情绪回避: composite={_neg_composite:.1f}>85, 降级为不行动"
            )
        elif should_act and _neg_composite > 60 and not _is_force:
            # 较高负面情绪：有概率跳过（40%概率回避）
            import random

            if random.random() < 0.4:
                should_act = False
                logger.info(
                    f"{self.log_prefix} SOC-03 负面情绪概率回避: neg={_neg_emo:.1f}>60, 随机跳过"
                )

        if should_act:
            _nc_reply = self._ensure_night_cycle()
            if _nc_reply is not None:
                try:
                    _nc_reply.record_night_reply()
                except Exception:
                    pass
            self._apply_plan_drain()
            await self._deduct_shared_resources(incoming_batch)
            self._last_proactive_intent_id = ""  # 每次行动前清空，避免误关联
            pre_observe_last_msg_time = 0.0
            for msg in reversed(incoming_batch[-10:]):
                if getattr(msg, "user_id", "") == "bot":
                    pre_observe_last_msg_time = getattr(msg, "timestamp", 0.0)
                    break
            _voice_needs_upgrade = False
            if voice_conclusion is not None and hasattr(
                voice_conclusion, "needs_upgrade"
            ):
                _voice_needs_upgrade = bool(voice_conclusion.needs_upgrade)
            _llm_upgrade_allowed = True
            _upgrade_block_reason = ""
            _upgrade_window_sec = 300.0
            _upgrade_max_calls = 5
            if now < self._llm_upgrade_cooldown_until:
                _llm_upgrade_allowed = False
                _upgrade_block_reason = f"升级冷却中({
                    int(
                        self._llm_upgrade_cooldown_until -
                        now)}s)"
            elif (
                self._llm_upgrade_call_window_start > 0
                and (now - self._llm_upgrade_call_window_start)
                > _upgrade_window_sec
            ):
                self._llm_upgrade_call_count = 0
                self._llm_upgrade_call_window_start = now
            elif self._llm_upgrade_call_window_start <= 0:
                self._llm_upgrade_call_window_start = now
            if self._llm_upgrade_call_count >= _upgrade_max_calls:
                _llm_upgrade_allowed = False
                _upgrade_block_reason = f"升级次数已达上限({_upgrade_max_calls} /{
                    _upgrade_window_sec} s) "
            if _voice_needs_upgrade and not _llm_upgrade_allowed:
                logger.info(
                    f"{self.log_prefix} ⚠️ 小模型判定需要升级但大模型不可用: {_upgrade_block_reason}，降级为小模型直接回复"
                )
                _voice_needs_upgrade = False
            elif _voice_needs_upgrade:
                logger.info(
                    f"{
                        self.log_prefix} 🔄 小模型判定需要升级至大模型深度分析（调用计数:{
                        self._llm_upgrade_call_count + 1}/{_upgrade_max_calls}）"
                )
            # 核心路径分流：算法+小模型联合决策
            if _final_reply:
                if _voice_needs_upgrade and _llm_upgrade_allowed:
                    self._llm_upgrade_call_count += 1
                    _upgrade_decision = await self._llm_autonomous_decide(
                        now,
                        relation_result,
                        decision_messages,
                        voice_conclusion,
                    )
                    self._llm_upgrade_cooldown_until = now + 60.0
                    if (
                        _upgrade_decision
                        and hasattr(_upgrade_decision, "should_act")
                        and _upgrade_decision.should_act
                    ):
                        logger.info(
                            f"{self.log_prefix} 🔄 大模型深度分析完成，决定回复: "
                            f"{_upgrade_decision.content_plan[:50] if _upgrade_decision.content_plan else '无内容规划'} "
                            f"感受={_upgrade_decision.emotional_state} 意图={_upgrade_decision.social_intention}"
                        )
                        _upgrade_proactive = type(
                            "UpgradeDecision", (),
                            {"action": _upgrade_decision.action,
                             "content_plan": _upgrade_decision.content_plan,
                             "reason": _upgrade_decision.reasoning,
                             "confidence": _upgrade_decision.confidence,
                             "internal_state": _upgrade_decision.
                             emotional_state, "matched_intent_id": "",
                             "target_user_id": "", },)()
                        actual_reply_made = await self._execute_planned_reply(
                            decision_messages=decision_messages,
                            planner_decision=_upgrade_proactive,
                            force_reply_message=None,
                        )
                    else:
                        logger.info(
                            f"{self.log_prefix} 🔄 大模型深度分析决定不回复，观察"
                        )
                        actual_reply_made = await self._observe(
                            recent_messages_list=decision_messages,
                            force_reply_message=force_reply_message,
                        )
                else:
                    actual_reply_made = await self._execute_voice_driven_reply(
                        voice_conclusion=voice_conclusion,
                        incoming_batch=decision_messages,
                    )
            elif (
                planner_decision is not None
                and hasattr(planner_decision, "action")
                and (
                    planner_decision.action.value
                    if hasattr(planner_decision.action, "value")
                    else str(planner_decision.action)
                )
                == "proactive_speak"
            ):
                # 统一规划器决定主动发言 → 走专用主动回复执行器
                from src.chat.heart_flow.llm_autonomous_planner import (
                    AutonomousDecision,
                )

                # 捕获规划器命中的意图id，供闭环追踪
                _planner_intent_id = str(
                    getattr(planner_decision, "matched_intent_id", "") or ""
                ).strip()
                self._last_proactive_intent_id = _planner_intent_id
                _proactive_decision = AutonomousDecision(
                    should_act=True,
                    action_type="proactive_speak",
                    content_plan=getattr(planner_decision, "content_plan", "")
                    or "",
                    reasoning=getattr(planner_decision, "reason", "") or "",
                    confidence=getattr(planner_decision, "confidence", 0.6),
                    emotional_state=getattr(
                        planner_decision, "internal_state", ""
                    )
                    or "",
                    social_intention=self._llm_social_intention
                    or "想主动聊几句",
                )
                actual_reply_made = await self._execute_proactive_reply(
                    llm_decision=_proactive_decision,
                    incoming_batch=decision_messages,
                )
                if actual_reply_made:
                    logger.info(
                        f"{self.log_prefix} 统一规划器触发主动回复成功"
                    )
                else:
                    logger.info(
                        f"{self.log_prefix} 主动回复生成失败，降级到常规观察"
                    )
                    actual_reply_made = await self._observe(
                        recent_messages_list=decision_messages,
                        force_reply_message=force_reply_message,
                    )
            else:
                actual_reply_made = await self._execute_planned_reply(
                    decision_messages=decision_messages,
                    planner_decision=planner_decision,
                    force_reply_message=force_reply_message,
                )
            if actual_reply_made:
                self._mark_message_content_processed(
                    self._get_latest_human_message(decision_messages),
                    action_name="planner_reply",
                    quality=0.7,
                )
                await self._finalize_reply_settlement(
                    incoming_batch=incoming_batch,
                    relation_result=relation_result,
                    voice_driven_reply=voice_driven_reply,
                    planner_triggered=planner_triggered,
                    did_reply=True,
                    behavior_signal=behavior_signal,
                )
                # 回复成功后将对应用户的活跃意图标记为已完成
                try:
                    from src.chat.proactive.intention_pool import (
                        get_intention_pool,
                    )

                    _pool = get_intention_pool()
                    _matched_intent_id = (
                        str(
                            getattr(planner_decision, "matched_intent_id", "")
                            or ""
                        ).strip()
                        if planner_decision
                        else ""
                    )
                    _completed_precise = False
                    if _matched_intent_id:
                        _completed_precise = bool(
                            _pool.complete_intent(_matched_intent_id)
                        )
                        if _completed_precise:
                            logger.info(
                                f"{self.log_prefix} 意图精确结案 id={_matched_intent_id[:10]}"
                            )
                    if not _completed_precise:
                        _target = (
                            str(
                                getattr(planner_decision, "target_user_id", "")
                                or ""
                            ).strip()
                            or str(
                                getattr(self, "_last_user_id", "") or ""
                            ).strip()
                        )
                        if _target:
                            _count = _pool.complete_by_target(
                                self.stream_id, _target, limit=1
                            )
                            if _count > 0:
                                logger.info(
                                    f"{self.log_prefix} 意图回退结案 target={_target[:8]} count={_count}"
                                )
                except Exception:
                    pass
                # 第三层：回复成功裁定
                _reply_source = (
                    "voice_driven"
                    if voice_driven_reply
                    else ("planner" if planner_triggered else "observe")
                )
                _reply_confidence = (
                    getattr(planner_decision, "confidence", 0.6)
                    if planner_decision
                    else 0.5
                )
                self._emit_action_verdict(
                    f"reply({_reply_source})",
                    "回复已发送",
                    time.time() - _t0,
                    confidence=_reply_confidence,
                )
            else:
                # 第三层：规划器决定不回复
                self._emit_action_verdict(
                    "no_reply", "规划器返回no_reply", time.time() - _t0
                )
                logger.info(
                    f"{self.log_prefix} ⏭️ 规划器返回no_reply，跳过回复消耗"
                )
                try:
                    from src.chat.proactive.intention_pool import (
                        get_intention_pool,
                    )

                    _matched_intent_id = (
                        str(
                            getattr(planner_decision, "matched_intent_id", "")
                            or ""
                        ).strip()
                        if planner_decision
                        else ""
                    )
                    if _matched_intent_id:
                        _reason = (
                            str(
                                getattr(planner_decision, "reason", "") or ""
                            ).strip()
                            or "planner_no_reply"
                        )
                        if get_intention_pool().fail_intent(
                            _matched_intent_id, reason=_reason[:120]
                        ):
                            logger.info(
                                f"{self.log_prefix} 意图精确失败回写 id={_matched_intent_id[:10]} reason=no_reply"
                            )
                except Exception:
                    pass
            # 第四层：结果汇总（完整替代旧版节流输出）
            await self._emit_outcome_summary(
                actual_reply_made, relation_result
            )
            logger.info(
                f"{self.log_prefix} 🔄 完整管线耗时 {time.time() - _t0:.2f}s"
            )
        else:
            # 第三层：决定不行动
            _inaction_reason = "统一规划器决定不行动"
            if planner_decision and hasattr(planner_decision, "action"):
                _pa = (
                    planner_decision.action.value
                    if hasattr(planner_decision.action, "value")
                    else str(planner_decision.action)
                )
                _inaction_reason = f"规划器裁定={_pa}"
                if (
                    hasattr(planner_decision, "reason")
                    and planner_decision.reason
                ):
                    _inaction_reason += f" {planner_decision.reason[:40]}"
            self._emit_action_verdict(
                "no_action", _inaction_reason, time.time() - _t0
            )
            try:
                from src.chat.proactive.intention_pool import (
                    get_intention_pool,
                )

                _matched_intent_id = (
                    str(
                        getattr(planner_decision, "matched_intent_id", "")
                        or ""
                    ).strip()
                    if planner_decision
                    else ""
                )
                if _matched_intent_id:
                    _reason = str(_inaction_reason or "planner_no_action")[
                        :120
                    ]
                    if get_intention_pool().fail_intent(
                        _matched_intent_id, reason=_reason
                    ):
                        logger.info(
                            f"{self.log_prefix} 意图精确失败回写 id={_matched_intent_id[:10]} reason=no_action"
                        )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            await self._emit_outcome_summary(False, relation_result)
            # 自适应管线收集（不行动路径）
            self._integrate_adaptive_collection(incoming_batch, False)
            # 被@但未回复时为对方增加心理压力感知
            _was_at_bot = any(
                getattr(_m, "is_at_bot", False)
                for _m in incoming_batch
                if hasattr(_m, "is_at_bot")
            )
            if _was_at_bot:
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker, )

                    _emo_t = get_emotion_tracker(self.stream_id)
                    for _m in reversed(incoming_batch):
                        _uid = str(getattr(_m, "user_id", "") or "").strip()
                        if _uid and _uid != "bot":
                            _emo_t.update_psychological_pressure(
                                _uid, 2.0, "被@未回复"
                            )
                            break
                except Exception:
                    pass
            logger.info(
                f"{self.log_prefix} 🔄 决定不行动，管线耗时 {time.time() - _t0:.2f}s"
            )
            await asyncio.sleep(
                _POST_MESSAGE_RETRY_SEC if has_user_message else 10
            )
            return True

        return True

    async def _finalize_reply_settlement(
        self,
        incoming_batch: List,
        relation_result: Dict[str, Any],
        voice_driven_reply: bool,
        planner_triggered: bool,
        did_reply: bool = True,
        behavior_signal: Optional[Dict[str, Any]] = None,
    ) -> None:
        """统一收口回复后的资源、关系、记忆与状态结算。"""
        pre_speaks = float(getattr(self, "_consecutive_speaks", 0.0) or 0.0)
        pre_chatterbox = float(
            getattr(self, "_chatterbox_penalty", 0.0) or 0.0
        )
        pre_unanswered = int(getattr(self, "_unanswered_bot_turns", 0) or 0)
        settlement_parts = [
            f"门控={getattr(self, '_last_legacy_gate', 'allow')}",
            f"主动={1 if (voice_driven_reply or planner_triggered) else 0}",
        ]
        legacy_breakdown = getattr(self, "_last_legacy_breakdown", None)
        if (
            did_reply
            and isinstance(legacy_breakdown, dict)
            and legacy_breakdown
        ):
            settlement_parts.append(
                f"准入={
                    legacy_breakdown.get(
                        'reply_readiness',
                        0):.1f}"
            )
            settlement_parts.append(
                f"资源扣={
                    legacy_breakdown.get(
                        'resource_penalty',
                        0):.1f}"
            )
            settlement_parts.append(
                f"关系扣={
                    legacy_breakdown.get(
                        'relation_penalty',
                        0):.1f}"
            )
            settlement_parts.append(
                f"重复扣={
                    legacy_breakdown.get(
                        'repeat_penalty',
                        0):.1f}"
            )
        await self._apply_reply_drain_dynamic(incoming_batch, relation_result)
        self._last_interest_level = max(
            0.1,
            min(
                1.0, float(getattr(self, "_last_desire_level", 5) or 5) / 10.0
            ),
        )
        await self._update_social_metrics(
            incoming_batch,
            relation_result,
            behavior_signal=behavior_signal,
        )
        await self._store_interaction_memory(incoming_batch)
        self._on_bot_sent(
            was_proactive=(voice_driven_reply or planner_triggered)
        )
        # 核心模块回复后学习 + 印象 + 技能 + 代谢
        _last_reply_text = ""
        if self._recent_bot_utterances:
            _last_reply_text = str(
                self._recent_bot_utterances[-1].get("text", "") or ""
            )
        self._integrate_post_reply_learning(
            incoming_batch, reply_text=_last_reply_text
        )
        # ResultGauge 回填面板
        try:
            from src.core.panel_state_adapter import get_panel_adapter

            _adapter = get_panel_adapter()
            _action_result = {
                "loop_action_info": {
                    "action_taken": True,
                    "reply_text": _last_reply_text[:500],
                    "taken_time": time.time(),
                    "quote_message": False,
                },
                "loop_plan_info": {
                    "action_result": {
                        "action_type": (
                            "planner_reply"
                            if planner_triggered
                            else (
                                "voice_reply"
                                if voice_driven_reply
                                else "reply"
                            )
                        ),
                        "reasoning": str(
                            getattr(self, "_last_legacy_gate", "") or ""
                        ),
                    },
                },
            }
            _result_snap = _adapter.assemble_with_result(
                self._tick_world_snapshot,
                action_result=_action_result,
            )
            self._cached_result_gauge = _result_snap.result
        except Exception as exc:
            logger.debug(f"{self.log_prefix} ResultGauge 回填异常: {exc}")
        # 自适应管线收集（回复成功路径）
        self._integrate_adaptive_collection(
            incoming_batch, True, _last_reply_text
        )
        settlement_parts.append(
            f"连续发言={pre_speaks:.1f}->{getattr(self, '_consecutive_speaks', 0.0):.1f}"
        )
        settlement_parts.append(
            f"话痨惩罚={pre_chatterbox:.1f}->{getattr(self, '_chatterbox_penalty', 0.0):.1f}"
        )
        settlement_parts.append(
            f"未回应计数={pre_unanswered}->{getattr(self, '_unanswered_bot_turns', 0)}"
        )
        if behavior_signal:
            settlement_parts.append(
                f"行为={
                    behavior_signal.get(
                        'category',
                        'neutral')}"
            )
        # 管线预算摘要
        if self._cached_pipeline_summary:
            _remaining_cands = self._cached_pipeline_summary.get(
                "budget_remaining_candidates", "?"
            )
            _remaining_tokens = self._cached_pipeline_summary.get(
                "budget_remaining_tokens", "?"
            )
            settlement_parts.append(
                f"管线余候选={_remaining_cands} 余token={_remaining_tokens}"
            )
        logger.info(
            f"{self.log_prefix} 结算账本 " + " ".join(settlement_parts)
        )

    async def _execute_planned_reply(
        self,
        decision_messages: List,
        planner_decision: Optional[Any],
        force_reply_message: Optional[Any] = None,
    ) -> bool:
        from src.plugin_system.base.component_types import ActionInfo
        from src.plugin_system.apis import generator_api
        from src.chat.utils.utils import record_replyer_action_temp

        target_message = force_reply_message or self._get_latest_human_message(
            decision_messages
        )
        if target_message is None:
            logger.info(
                f"{self.log_prefix} 规划器要求回复，但当前没有可回复目标"
            )
            return False

        target_message = self._select_preferred_reply_message(
            target_message, list(decision_messages[-10:])
        )
        self._mark_message_content_processing(target_message)

        reply_reason = (
            getattr(planner_decision, "reason", "统一规划器决定回复")
            or "统一规划器决定回复"
        )
        content_plan = str(
            getattr(planner_decision, "content_plan", "") or ""
        ).strip()
        legacy_gate = str(
            getattr(self, "_last_legacy_gate", "allow") or "allow"
        )
        if content_plan:
            reply_reason = f"{reply_reason} | 内容规划: {content_plan[:100]}"
        if legacy_gate == "hesitate":
            reply_reason = f"{reply_reason} | 当前处于低可回复度，只允许简短澄清或低风险短回"

        extra_parts: List[str] = []
        voice_summary = self._build_voice_execution_summary(
            getattr(self, "_cached_voice", None), target_message
        )
        voice_panel = (
            voice_summary.get("panel", "")
            if isinstance(voice_summary, dict)
            else ""
        )
        if voice_panel:
            extra_parts.append(voice_panel)
        if content_plan:
            extra_parts.append(f"[统一规划内容规划] {content_plan}")

        continuity_context = self._build_self_continuity_context()
        if continuity_context:
            extra_parts.append(continuity_context)
        self_memory = self._build_self_reply_memory()
        if self_memory:
            extra_parts.append(self_memory)

        target_user_id = getattr(target_message, "user_id", "") or ""
        user_style_guide = (
            self._get_user_style_guide(target_user_id)
            if target_user_id
            else ""
        )
        if user_style_guide:
            extra_parts.append(user_style_guide)

        persona_hint = self._build_persona_hint()
        if persona_hint:
            extra_parts.append(persona_hint)
        relation_view = self._resolve_relation_view()
        reply_style_context = self._build_reply_style_context(relation_view)
        if reply_style_context:
            extra_parts.append(reply_style_context)

        length_hint = self._build_dynamic_length_hint(
            target_message, user_style_guide
        )
        if length_hint:
            extra_parts.append(length_hint)
        if legacy_gate == "hesitate":
            extra_parts.append(
                "[低可回复度约束] 当前只允许一句短回复，优先澄清、确认、轻量接话；"
                "不要长篇展开，不要主动延伸新话题，不要输出高情绪高承诺表达。"
            )

        harassment_signal = self._analyze_harassment_pressure(
            decision_messages
        )
        style_route = self._decide_reply_style(
            target_message=target_message,
            relation_snapshot=relation_view,
            harassment_signal=harassment_signal,
        )
        extra_parts.append(
            f"[回复形式] {
                style_route['reply_style']} | {
                style_route['reason']}"
        )
        # 注入多维状态系统的LLM提示词
        self._inject_dimension_state_prompt(extra_parts)
        extra_info = "\n".join(part for part in extra_parts if part)

        self._emit_reply_generation_summary(
            target_message=target_message,
            style_route=style_route,
            relation_snapshot=relation_view,
            context_execution_block="",
            extra_info=extra_info,
            source="planner",
        )

        record_replyer_action_temp(
            chat_id=self.stream_id,
            reason=reply_reason,
            think_level=1,
        )

        available_actions: Dict[str, ActionInfo] = {}
        try:
            await self.action_modifier.modify_actions()
            available_actions = self.action_manager.get_using_actions()
        except Exception as exc:
            logger.warning(f"{self.log_prefix} 获取可用动作失败: {exc}")

        success, llm_response = await generator_api.generate_reply(
            chat_stream=self.chat_stream,
            reply_message=target_message,
            available_actions=available_actions,
            chosen_actions=[],
            reply_reason=reply_reason,
            unknown_words=None,
            enable_tool=global_config.tool.enable_tool,
            request_type="planner_reply",
            from_plugin=False,
            reply_time_point=time.time(),
            think_level=1,
            extra_info=extra_info,
        )
        if not success or not llm_response or not llm_response.reply_set:
            self._mark_message_content_deferred(
                target_message, "planner_generation_failed"
            )
            logger.warning(f"{self.log_prefix} 规划回复生成失败")
            return False

        response_set = llm_response.reply_set
        selected_expressions = llm_response.selected_expressions
        cycle_timers: Dict[str, float] = {}
        thinking_id = f"planner_{int(time.time() * 1000)}"
        loop_info, reply_text, _ = await self._send_and_store_reply(
            response_set=response_set,
            cycle_timers=cycle_timers,
            action_message=target_message,
            thinking_id=thinking_id,
            actions="planner_reply",
            selected_expressions=selected_expressions,
        )
        return bool(loop_info) or bool(str(reply_text or "").strip())

    # ═══════════════════════════════════════════════════
    #  阶段 0：状态机门控
    # ═══════════════════════════════════════════════════

    def _query_flow_phase(self) -> str:
        """查询当前频道的心流阶段，从世界快照读取"""
        snap = getattr(self, "_tick_world_snapshot", None)
        if snap is not None and snap.subject.flow_phase:
            return snap.subject.flow_phase
        # 快照尚未构建（首轮tick），默认活跃态
        return "engaged"

    def _evaluate_phase_gate(self, phase_str: str, now: float) -> str:
        """根据心流阶段决定本轮循环的行为

        返回值：
        - "proceed"  继续完整流水线
        - "skip"     跳过本轮（休息期不响应）
        - "glance"   休息期窥屏
        """
        if phase_str == "dormant":
            return self._handle_dormant_gate()
        if phase_str == "pending":
            return self._handle_pending_gate(now)
        if phase_str == "standby":
            # 待命阶段正常进入流水线，但迁移至 ENGAGED
            self._shift_to_engaged(cause="待命→活跃：消息到达")
            return "proceed"
        return "proceed"

    def _handle_dormant_gate(self) -> str:
        """休息期门控：委托 DormancySupervisor 决策"""
        try:
            from src.chat.heart_flow.rest_handler import (
                get_dormancy_supervisor,
            )
            from src.common.data_models.heartflow_models import DormantReaction

            supervisor = get_dormancy_supervisor()
            reaction = supervisor.on_incoming_msg(self.stream_id)
            if reaction == DormantReaction.AWAKEN:
                try:
                    from src.core.night_cycle_system import get_night_cycle

                    _nc = get_night_cycle(self.stream_id)
                    _nc.stimulus_wake(strength=1.0)
                    _grace = _nc.compute_wake_grace_duration()
                    self._wake_grace_until = time.time() + _grace
                    logger.info(
                        f"{
                            self.log_prefix} 😴 动态唤醒宽限期={
                            _grace:.0f}s 债务={
                            _nc.sleep_debt():.2f} 熬夜={
                            _nc._state.burn_accumulated:.0f}"
                    )
                except Exception:
                    self._wake_grace_until = time.time() + 120.0
                self._shift_to_engaged(cause="休息结束被唤醒")
                return "proceed"
            if reaction == DormantReaction.GLANCE:
                return "glance"
            return "skip"
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 休息门控异常，放行: {exc}")
            return "proceed"

    def _handle_pending_gate(self, now: float) -> str:
        """等待期门控：委托 PendingOrchestrator 决策"""
        try:
            from src.chat.heart_flow.waiting_handler import (
                get_pending_orchestrator,
            )
            from src.common.data_models.heartflow_models import IdleDecision

            orchestrator = get_pending_orchestrator()
            decision = orchestrator.evaluate_next(self.stream_id)
            if (
                decision == IdleDecision.EXPIRED
                or decision == IdleDecision.WITHDRAW
            ):
                # 超时退出时触发后果链回写
                orchestrator.conclude_with_timeout_consequence(self.stream_id)
                self._shift_to_engaged(cause=f"等待结束: {decision.value}")
                return "proceed"
            if decision == IdleDecision.CONTEMPLATE:
                return "proceed"
            if decision == IdleDecision.CHASE:
                return "proceed"
            return "skip"
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 等待门控异常，放行: {exc}")
            return "proceed"

    # ═══════════════════════════════════════════════════
    #  阶段 1：能量评估
    # ═══════════════════════════════════════════════════

    def _poll_eagerness(self) -> Tuple[float, str]:
        """读取回复意愿度，优先从世界快照推算"""
        snap = getattr(self, "_tick_world_snapshot", None)
        if snap is not None:
            res = snap.self_resources
            thinking_factor = res.thinking_ratio()
            chat_factor = res.chat_ratio()
            activity_factor = max(0.0, min(1.0, res.activity_level / 100.0))
            annoyance_penalty = min(res.channel_annoyance / 100.0, 0.5)
            raw = (
                thinking_factor * 0.4
                + chat_factor * 0.25
                + activity_factor * 0.2
                + 0.5 * 0.15
                - annoyance_penalty
            )
            eagerness = max(0.05, min(1.0, raw))
            reasons = []
            if res.chat_energy < res.chat_ceiling * 0.3:
                reasons.append(f"聊天值低({res.chat_energy:.0f})")
            if res.activity_level < 30:
                reasons.append(f"活跃度低({res.activity_level:.0f})")
            if res.thinking_energy < res.thinking_ceiling * 0.3:
                reasons.append(f"思考值低({res.thinking_energy:.0f})")
            if res.channel_annoyance > 30:
                reasons.append(f"烦躁({res.channel_annoyance:.0f})")
            return eagerness, (
                " | ".join(reasons) if reasons else "快照状态良好"
            )
        try:
            from src.chat.heart_flow.energy_manager import get_vitality_pool

            pool = get_vitality_pool()
            return pool.compute_eagerness(self.stream_id)
        except Exception:
            return 1.0, "能量模块不可用，默认满意愿"

    def _apply_glance_drain(self) -> None:
        """窥屏消耗"""
        try:
            from src.chat.heart_flow.energy_manager import get_vitality_pool

            get_vitality_pool().deplete_on_glance(self.stream_id)
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _apply_think_drain(self, complexity: float = 1.0) -> None:
        """思考消耗，支持动态复杂度"""
        try:
            from src.chat.heart_flow.energy_manager import get_vitality_pool

            get_vitality_pool().deplete_on_think(self.stream_id, complexity=complexity)
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _apply_reply_drain(self, messages: List) -> None:
        """回复消耗，按消息总长度计算"""
        try:
            from src.chat.heart_flow.energy_manager import get_vitality_pool

            total_len = sum(
                len(getattr(m, "processed_plain_text", "") or "")
                for m in messages
            )
            get_vitality_pool().deplete_on_reply(
                self.stream_id, content_length=total_len
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    async def _apply_reply_drain_dynamic(
        self, messages: List, relation_result: Dict
    ) -> None:
        """回复消耗 - 基于水位比例和多因素链的动态算法

        算法流程：
        1. 水位基底：聊天比 × 3.0 → [0.6, 3.5]；思考比 × 2.5 → [0.5, 2.5]
        2. 连续发言递增：每多说一次 +0.3
        3. 频率因子：<60s=1.6, <300s=1.3, <600s=1.1, else=1.0
        4. 话痨因子：1.0 + 话痨惩罚 × 0.4
        5. 兴趣折扣：max(0.6, 1.0 - 兴趣等级 × 0.4)
        6. 聊天值总计 clamp [0.8, 8.0]；思考值总计 clamp [0.5, 5.0]
        """
        try:
            from src.chat.heart_flow.energy_manager import get_vitality_pool

            pool = get_vitality_pool()
            snap = pool.capture_snapshot(self.stream_id)
            chat_ratio = snap.chat_ratio()
            thinking_ratio = snap.thinking_ratio()
            base_chat_cost = max(0.6, min(3.5, chat_ratio * 3.0))
            base_brain_cost = max(0.5, min(2.5, thinking_ratio * 2.5))
            consecutive_add = getattr(self, "_consecutive_speaks", 0) * 0.3
            time_since_last = time.time() - getattr(
                self, "_last_speak_time", 0
            )
            if time_since_last < 60:
                frequency_factor = 1.6
            elif time_since_last < 300:
                frequency_factor = 1.3
            elif time_since_last < 600:
                frequency_factor = 1.1
            else:
                frequency_factor = 1.0
            chatterbox_penalty = getattr(self, "_chatterbox_penalty", 0.0)
            chatterbox_factor = 1.0 + chatterbox_penalty * 0.4
            interest_level = getattr(self, "_last_interest_level", 0.5)
            interest_discount = max(0.6, 1.0 - interest_level * 0.4)
            total_chat_cost = (
                (base_chat_cost + consecutive_add)
                * frequency_factor
                * chatterbox_factor
                * interest_discount
            )
            total_brain_cost = (
                base_brain_cost * frequency_factor * interest_discount
            )
            total_chat_cost = max(0.8, min(8.0, total_chat_cost))
            total_brain_cost = max(0.5, min(5.0, total_brain_cost))
            pool.apply_manual_adjustment(
                self.stream_id,
                chat_delta=-total_chat_cost,
                thinking_delta=-total_brain_cost,
            )
            self._consecutive_speaks = (
                getattr(self, "_consecutive_speaks", 0) + 1
            )
            self._last_speak_time = time.time()
            now_ts = time.time()
            self._bot_reply_counter += 1
            self._bot_reply_timeline.append(now_ts)
            if len(self._bot_reply_timeline) > 50:
                self._bot_reply_timeline = self._bot_reply_timeline[-50:]
            self._last_bot_reply_ts = now_ts
            self._update_dynamic_ratio(now_ts)
            if self._consecutive_speaks >= 4:
                self._chatterbox_penalty = (
                    getattr(self, "_chatterbox_penalty", 0.0) + 1.0
                )
                logger.warning(
                    f"{self.log_prefix} 话痨警告！连续发言{self._consecutive_speaks}次，"
                    f"惩罚值={self._chatterbox_penalty:.1f}"
                )
            logger.info(
                f"{
                    self.log_prefix} ⚡ "
                f"本轮消耗 chat={
                    total_chat_cost:.2f} think={
                    total_brain_cost:.2f} | "
                f"连续加成={
                    consecutive_add:.1f} 频率因子={
                        frequency_factor:.2f} "
                f"话痨因子={
                    chatterbox_factor:.2f} 兴趣折扣={
                    interest_discount:.2f} "
                f"(水位: chat={
                    chat_ratio:.2f} thinking={
                    thinking_ratio:.2f})"
            )
            try:
                from src.core.night_cycle_system import get_night_cycle

                _nc = get_night_cycle(self.stream_id)
                if _nc.is_night_hours():
                    _fat_added = _nc.add_reply_fatigue()
                    if _fat_added > 0:
                        logger.info(
                            f"{self.log_prefix} 🌙 夜间疲劳+{_fat_added:.1f}(当前{_nc._state.daily_fatigue:.0f}/{_nc._state.daily_fatigue_limit:.0f})"
                        )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 回复消耗计算失败: {exc}")

    async def _deduct_shared_resources(self, messages: List) -> None:
        """实时扣除共享资源 - 动作执行时立即消耗"""
        try:
            from src.chat.heart_flow.energy_manager import (
                get_shared_resource_manager,
            )

            shared_mgr = get_shared_resource_manager()
            chat_result = shared_mgr.deduct_chat_value(
                self.stream_id, 0, "reply"
            )
            activity_result = shared_mgr.deduct_activity_level(
                self.stream_id, 0, "action"
            )
            user_id = ""
            content = ""
            if messages:
                for msg in reversed(messages):
                    uid = getattr(msg, "user_id", "") or ""
                    if uid and uid != "bot":
                        user_id = uid
                        content = getattr(
                            msg, "processed_plain_text", ""
                        ) or getattr(msg, "content", "")
                        break
            if user_id:
                shared_mgr.register_member(self.stream_id, user_id)
            polarity = (
                self._estimate_text_polarity(content) if content else 0.0
            )
            if polarity > 0.1:
                social_result = shared_mgr.deduct_social_value(
                    self.stream_id, 0, "positive"
                )
            elif polarity < -0.1:
                social_result = shared_mgr.deduct_social_value(
                    self.stream_id, 0, "negative"
                )
            else:
                social_result = {"deducted": 0.0}
            logger.info(
                f"{self.log_prefix} ⚡ "
                f"聊天值-{chat_result['deducted']:.1f} "
                f"活跃度-{activity_result['deducted']:.1f} "
                f"社交值变化={social_result['deducted']:.1f}"
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 共享资源扣除失败: {exc}")

    async def _apply_input_state_updates(self, messages: List) -> None:
        """用户消息到达时即时激活共享聊天值/活跃度状态。"""
        try:
            from src.chat.heart_flow.energy_manager import (
                get_shared_resource_manager,
            )
            from src.person_info.person_info import get_unified_profile_hub

            human_messages = [
                msg
                for msg in messages
                if getattr(msg, "user_id", "")
                and getattr(msg, "user_id", "") != "bot"
            ]
            if not human_messages:
                return

            latest_human = human_messages[-1]
            latest_text = (
                getattr(latest_human, "processed_plain_text", "")
                or getattr(latest_human, "content", "")
                or ""
            )
            normalized_latest = self._normalize_repeat_text(latest_text)
            latest_user_id = str(getattr(latest_human, "user_id", "") or "")
            _raw_nick = str(
                getattr(latest_human, "user_nickname", "")
                or getattr(latest_human, "nickname", "")
                or ""
            ).strip()
            latest_nickname = _raw_nick or latest_user_id
            if _raw_nick and len(latest_user_id) > 16:
                latest_user_id = f"{self.stream_id}_{_raw_nick[:12]}"
            repeat_count = self._count_recent_user_repeats(
                normalized_latest, latest_user_id
            )

            if latest_user_id:
                hub = get_unified_profile_hub()
                hub.bind_person_identity(
                    person_id=latest_user_id,
                    platform=(
                        "webui"
                        if latest_user_id.startswith("webui_")
                        else "unknown"
                    ),
                    origin_user_id=latest_user_id,
                    nickname=latest_nickname,
                )
                hub.record_interaction(latest_user_id, self.stream_id)

            shared_mgr = get_shared_resource_manager()
            result = shared_mgr.apply_input_activity(
                self.stream_id,
                user_id=latest_user_id,
                content_length=len(latest_text),
                is_repeat=repeat_count >= 2,
            )
            self._remember_recent_user_input(normalized_latest, latest_user_id)
            logger.info(
                f"{self.log_prefix} 状态激活 聊天值{result['chat_delta']:+.1f} "
                f"活跃度{result['activity_delta']:+.1f} 社交值{result['social_delta']:+.1f} "
                f"重复={'是' if result['is_repeat'] else '否'}"
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 输入状态激活失败: {exc}")

    def _remember_recent_user_input(
        self, normalized_text: str, user_id: str
    ) -> None:
        if not normalized_text:
            return
        now = time.time()
        self._recent_user_inputs.append(
            {
                "text": normalized_text,
                "user_id": user_id,
                "ts": now,
            }
        )
        cutoff = now - 300.0
        self._recent_user_inputs = [
            item
            for item in self._recent_user_inputs[-20:]
            if float(item.get("ts", 0.0) or 0.0) >= cutoff
            and str(item.get("text", "") or "")
        ]

    def _count_recent_user_repeats(
        self, normalized_text: str, user_id: str
    ) -> int:
        if not normalized_text:
            return 0
        now = time.time()
        cutoff = now - 300.0
        count = 0
        for item in self._recent_user_inputs:
            if float(item.get("ts", 0.0) or 0.0) < cutoff:
                continue
            if str(item.get("text", "") or "") != normalized_text:
                continue
            item_user = str(item.get("user_id", "") or "")
            if user_id and item_user and item_user != user_id:
                continue
            count += 1
        return count + 1

    @staticmethod
    def _normalize_repeat_text(text: str) -> str:
        """压缩文本用于检测短时间重复输入。"""
        import re

        if not text:
            return ""
        return re.sub(r"[\s\W_]+", "", str(text), flags=re.UNICODE).lower()

    def _classify_behavior_signal(
        self,
        messages: List,
        repetition_signal: Optional[Dict[str, Any]] = None,
        harassment_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """统一行为分类信号：委托给共享模块，避免多条主链继续分叉。"""
        latest_user = None
        for msg in reversed(messages):
            if getattr(msg, "user_id", "") != "bot":
                latest_user = msg
                break
        if latest_user is None:
            return {
                "category": "neutral",
                "behavior_type": "casual_chat",
                "intent": "chat",
                "severity": 0.25,
                "is_new_user": False,
                "reason": "默认中性互动",
            }

        user_id = str(getattr(latest_user, "user_id", "") or "")
        text = self._extract_message_content(latest_user).strip()
        interaction_count = 0

        try:
            from src.person_info.person_info import get_unified_profile_hub
            from src.modules.social_value.social_affect_fuser import (
                get_social_affect_fuser,
            )
            from src.chat.behavior.behavior_signal import (
                classify_behavior_signal,
            )

            dossier = get_social_affect_fuser().get_dossier(
                user_id, self.stream_id
            )
            interaction_count = len(dossier.recent_values) if dossier else 0
            try:
                unified_profile = get_unified_profile_hub().get_profile(
                    user_id
                )
                if unified_profile is not None:
                    interaction_count = max(
                        interaction_count,
                        int(
                            getattr(unified_profile, "interaction_count", 0)
                            or 0
                        ),
                    )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            return classify_behavior_signal(
                text=text,
                user_id=user_id,
                interaction_count=interaction_count,
                repetition_signal=repetition_signal,
                harassment_signal=harassment_signal,
            )
        except Exception:
            from src.chat.behavior.behavior_signal import (
                classify_behavior_signal,
            )

            return classify_behavior_signal(
                text=text,
                user_id=user_id,
                interaction_count=interaction_count,
                repetition_signal=repetition_signal,
                harassment_signal=harassment_signal,
            )

    async def _update_social_metrics(
        self,
        messages: List,
        relation_result: Dict,
        behavior_signal: Optional[Dict[str, Any]] = None,
    ) -> None:
        """更新社交值 - 回复后通过SettlementEngine(9步富管线)更新"""
        try:
            from src.core import get_core

            user_id = ""
            content = ""
            if messages:
                for msg in reversed(messages):
                    uid = getattr(msg, "user_id", "") or ""
                    if uid and uid != "bot":
                        user_id = uid
                        content = getattr(
                            msg, "processed_plain_text", ""
                        ) or getattr(msg, "content", "")
                        break
            if not user_id:
                logger.debug(f"{self.log_prefix} 无用户ID，跳过社交值更新")
                return
            _nick = ""
            if messages:
                for msg in reversed(messages):
                    _n = (
                        getattr(msg, "user_nickname", "")
                        or getattr(msg, "nickname", "")
                        or ""
                    )
                    if _n:
                        _nick = _n.strip()
                        break
            if _nick and len(user_id) > 16:
                user_id = f"{self.stream_id}_{_nick[:12]}"
            self._last_user_id = user_id
            context = {
                "channel_id": self.stream_id,
                "interaction_type": "bot_reply",
                "current_social": relation_result.get("social_value", 0.0),
                "current_trust": relation_result.get("trust_value", 0.0),
                "trauma_score": float(
                    relation_result.get("trauma_score", 0.0) or 0.0
                ),
                "psychological_pressure": float(
                    relation_result.get("psychological_pressure", 0.0) or 0.0
                ),
            }
            if behavior_signal:
                context.update(
                    {
                        "behavior_type": behavior_signal.get(
                            "behavior_type", "casual_chat"
                        ),
                        "intent": behavior_signal.get("intent", "chat"),
                        "severity": float(
                            behavior_signal.get("severity", 0.25) or 0.25
                        ),
                        "is_new_user": bool(
                            behavior_signal.get("is_new_user", False)
                        ),
                        "behavior_category": behavior_signal.get(
                            "category", "neutral"
                        ),
                    }
                )
            # ── 首选：使用SettlementEngine（9步富管线） ──
            try:
                from src.modules.social_value.settlement_engine import (
                    get_settlement_engine,
                )

                se = get_settlement_engine()
                report = await se.settle_event(
                    user_id, self.stream_id, content, context
                )
                if report:
                    logger.info(
                        f"{self.log_prefix} 🤝 结算完成 "
                        f"对象={user_id[:8]} "
                        f"旧值={report.prior_score:.1f}→新值={report.settled_score:.1f} "
                        f"(Δ={report.final_delta:+.3f}) "
                        f"原始={report.raw_delta:+.3f} "
                        f"饱和={'是' if abs(report.delta_after_saturation) < abs(report.raw_delta) - 0.001 else '否'} "
                        f"反转={'是' if report.reversal_triggered else '否'} "
                        f"上限={'是' if report.cap_triggered else '否'}"
                    )
                    return
            except Exception as se_exc:
                logger.debug(
                    f"{self.log_prefix} SettlementEngine结算失败，回退到SocialValueCore: {se_exc}"
                )
            # ── 回退：使用SocialValueCore（5步基础管线） ──
            svc = get_core().social_value
            if svc is None:
                logger.warning(
                    f"{self.log_prefix} 社交值核心未初始化，跳过更新"
                )
                return
            result = await svc.update(
                user_id, self.stream_id, content, context
            )
            if result:
                logger.info(
                    f"{self.log_prefix} 🤝 "
                    f"对象={user_id[:8]} 行为={result.behavior_type} "
                    f"意图={result.intent} 社交值变化={result.delta:.2f} "
                    f"新值={result.new_value:.1f}"
                )
        except Exception as exc:
            logger.warning(f"{self.log_prefix} 社交值更新失败: {exc}")

    async def _log_final_status_dynamic(self) -> None:
        """动态最终状态输出 - 包含所有系统状态和共享资源（节流版）"""
        now = time.time()
        if now - self._last_status_log_ts < self._status_log_interval:
            return
        self._last_status_log_ts = now
        try:
            tick_snapshot = getattr(self, "_tick_world_snapshot", None)
            relation_view = self._resolve_relation_view()
            freq_ctrl = (
                frequency_control_manager.get_or_create_frequency_control(
                    self.stream_id
                )
            )
            freq_adjust = freq_ctrl.get_talk_frequency_adjust()
            user_id = ""
            if hasattr(self, "_last_user_id"):
                user_id = self._last_user_id
            trauma_score = 0.0
            psychological_pressure = 0.0
            training_stage = 0
            training_progress = 0.0
            training_resistance = 100.0
            inner_chaos = 0.0
            surface_mask = 10.0
            stress_accumulation = 0.0
            submission_level = 0.0
            from src.chat.heart_flow.energy_manager import (
                get_vitality_pool,
                get_shared_resource_manager,
            )

            pool = get_vitality_pool()
            snap = pool.capture_snapshot(self.stream_id)
            shared_mgr = get_shared_resource_manager()
            shared_values = shared_mgr.get_shared_values(self.stream_id)
            chat_value = float(shared_values.get("chat_value") or 0.0)
            activity_level = float(shared_values.get("activity_level") or 0.0)
            shared_social = float(shared_values.get("social_value") or 0.0)
            social_value = float(
                relation_view.get("social_value", shared_social) or 0.0
            )
            trust_value = float(relation_view.get("trust_value", 0.0) or 0.0)
            annoyance_value = float(
                relation_view.get("annoyance_value", 0.0) or 0.0
            )
            trauma_score = float(relation_view.get("trauma_score", 0.0) or 0.0)
            psychological_pressure = float(
                relation_view.get("psychological_pressure", 0.0) or 0.0
            )
            if user_id:
                try:
                    from src.chat.heart_flow.energy_manager import (
                        get_vitality_pool as _get_pool,
                    )

                    full_snap = await _get_pool().capture_full_snapshot(
                        self.stream_id, user_id
                    )
                    social_value = full_snap.social_value
                    trust_value = full_snap.trust_value
                    annoyance_value = full_snap.annoyance_value
                except Exception:
                    pass
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker, )

                    tracker = get_emotion_tracker(self.stream_id)
                    state = tracker.get_user_state(
                        user_id, create_if_missing=False
                    )
                    if state:
                        trauma_score = getattr(state, "trauma_score", 0.0)
                        psychological_pressure = getattr(
                            state, "psychological_pressure", 0.0
                        )
                        training_stage = getattr(state, "training_stage", 0)
                        training_progress = getattr(
                            state, "training_progress", 0.0
                        )
                        training_resistance = getattr(
                            state, "training_resistance", 100.0
                        )
                        inner_chaos = getattr(state, "inner_chaos", 0.0)
                        surface_mask = getattr(state, "surface_mask", 10.0)
                        submission_level = getattr(
                            state, "submission_level", 0.0
                        )
                except Exception:
                    pass
                try:
                    from src.modules.trauma.trauma_system import (
                        get_trauma_system,
                    )

                    trauma_sys = get_trauma_system()
                    trauma_state = trauma_sys.get_state()
                    inner_chaos = trauma_state.inner_chaos_level
                    surface_mask = trauma_state.surface_mask_strength
                    stress_accumulation = trauma_state.stress_accumulation
                except Exception:
                    pass
            base_parts = [
                f"聊天当前={chat_value:.1f}",
                f"活跃当前={activity_level:.1f}",
                f"社交当前={shared_social:.1f}",
                f"信任当前={trust_value:.1f}",
            ]
            if social_value != 0:
                base_parts.append(f"关系社交当前={social_value:.1f}")
            affection = float(relation_view.get("affection", 0.0) or 0.0)
            if affection != 0:
                base_parts.append(f"好感当前={affection:.1f}")
            conditional_parts = []
            if trauma_score > 0:
                conditional_parts.append(f"创伤={trauma_score:.1f}")
            if psychological_pressure > 0:
                conditional_parts.append(
                    f"心理压力={psychological_pressure:.1f}"
                )
            if inner_chaos > 0:
                conditional_parts.append(f"内心混乱={inner_chaos:.1f}")
            if surface_mask > 0:
                conditional_parts.append(f"伪装强度={surface_mask:.1f}")
            if stress_accumulation > 0:
                conditional_parts.append(f"压力累积={stress_accumulation:.1f}")
            if training_stage > 0:
                conditional_parts.append(f"调教阶段={training_stage}")
            if training_progress > 0:
                conditional_parts.append(f"调教进度={training_progress:.1f}%")
            if submission_level > 0:
                conditional_parts.append(f"顺从度={submission_level:.1f}")
            base_parts.append(f"频率={freq_adjust:.2f}")
            all_parts = base_parts + conditional_parts
            logger.info(f"{self.log_prefix} 📋 " + " ".join(all_parts))
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 最终状态输出失败: {exc}")

    # ═══════════════════════════════════════════════════
    #  四层状态栏：核心可观察性机制
    # ═══════════════════════════════════════════════════

    def _emit_vitals_gauge(
        self, eagerness_val: float, eagerness_reason: str, phase_label: str
    ) -> None:
        """第一层：独立系统体征面板 - 进入决策前的全局快照

        输出思考值、聊天值、活跃值、社交值、回复意愿、当前阶段。
        每次进入完整管线时必定输出，不依赖最终是否回复。
        """
        try:
            from src.chat.heart_flow.energy_manager import (
                get_vitality_pool,
                get_shared_resource_manager,
            )

            pool = get_vitality_pool()
            channel_snap = pool.capture_snapshot(self.stream_id)
            shared_mgr = get_shared_resource_manager()
            shared_vals = shared_mgr.get_shared_values(self.stream_id)
            thinking_pct = float(channel_snap.thinking_ratio() * 100.0)
            chat_pct = float(shared_vals.get("chat_value") or 0.0)
            activity_pct = float(shared_vals.get("activity_level") or 0.0)
            social_reading = float(shared_vals.get("social_value") or 0.0)

            logger.info(
                f"{self.log_prefix} 状态面板 "
                f"思考当前={thinking_pct:.0f}% 聊天当前={chat_pct:.1f} "
                f"活跃当前={activity_pct:.1f} 社交当前={social_reading:.1f} | "
                f"意愿={eagerness_val:.2f} 阶段={phase_label}"
            )
            # 面板适配器导出（附加核心模块维度）
            try:
                from src.core.panel_state_adapter import get_panel_adapter

                _adapter = get_panel_adapter()
                if self._tick_world_snapshot:
                    _panel_module_context = {}
                    if self._cached_watch_level:
                        _watch_label = (
                            self._cached_watch_level.label()
                            if hasattr(self._cached_watch_level, "label")
                            else str(self._cached_watch_level)
                        )
                        _panel_module_context["watch_info"] = {
                            "level": (
                                self._cached_watch_level.value
                                if hasattr(self._cached_watch_level, "value")
                                else str(self._cached_watch_level)
                            ),
                            "label": _watch_label,
                        }
                    if self._cached_metabolism_state:
                        _panel_module_context["metabolism"] = {
                            "loafing": float(
                                getattr(
                                    self._cached_metabolism_state,
                                    "loafing_level",
                                    0.0,
                                )
                                or 0.0
                            ),
                        }
                    # 代谢行为约束注入
                    if self._cached_metabolism_constraints:
                        _panel_module_context.setdefault(
                            "metabolism", {}
                        ).update(self._cached_metabolism_constraints)
                    # 夜间行为摘要注入
                    if self._cached_night_summary:
                        _panel_module_context["night"] = dict(
                            self._cached_night_summary
                        )
                    if self._cached_scene_snapshot and hasattr(
                        self._cached_scene_snapshot, "to_dict"
                    ):
                        _scene_dict = self._cached_scene_snapshot.to_dict()
                        _scene_dict["multi_thread"] = bool(
                            getattr(
                                self._cached_scene_snapshot, "thread_count", 0
                            )
                            > 1
                        )
                        _panel_module_context["scene_state"] = _scene_dict
                    if self._cached_pattern_evidence:
                        _dominant_pattern = self._cached_pattern_evidence[0]
                        if hasattr(_dominant_pattern, "to_dict"):
                            _pattern_dict = _dominant_pattern.to_dict()
                            _pattern_dict["detected"] = True
                            _panel_module_context["pattern"] = _pattern_dict
                    if self._cached_multimodal_summary:
                        _panel_module_context["budgeter"] = dict(
                            self._cached_multimodal_summary
                        )
                    if self._cached_learning_summary:
                        _panel_module_context["learning"] = dict(
                            self._cached_learning_summary
                        )
                    if self._last_relation_snapshot:
                        _impression_ctx = dict(self._last_relation_snapshot)
                        if self._last_user_id:
                            try:
                                from src.core.active_participant_roster import (
                                    get_participant_roster, )

                                _entry = get_participant_roster(
                                    self.stream_id
                                ).get_user(self._last_user_id)
                                _impression_ctx["is_high_activity"] = bool(
                                    _entry
                                    and getattr(
                                        getattr(_entry, "tier", None),
                                        "value",
                                        "",
                                    )
                                    in ("hot", "warm")
                                )
                            except Exception:
                                pass
                        _panel_module_context["impression"] = _impression_ctx
                    if self._cached_understanding_results:
                        _primary_understanding = (
                            self._cached_understanding_results[0]
                        )
                        _panel_module_context["understanding"] = {
                            "comprehension_level": getattr(
                                getattr(_primary_understanding, "level", None),
                                "value",
                                "",
                            ),
                            "deep_analysis": bool(
                                getattr(
                                    _primary_understanding,
                                    "upgrade_analysis",
                                    False,
                                )
                            ),
                        }
                    if self._cached_self_references:
                        # 提取最强引用类型和强度
                        _best_ref_type = "none"
                        _best_ref_strength = 0.0
                        _any_relevant = False
                        for _ref in self._cached_self_references:
                            if bool(
                                getattr(_ref, "is_relevant", lambda: False)()
                            ):
                                _any_relevant = True
                            _rs = float(getattr(_ref, "strength", 0.0) or 0.0)
                            if _rs > _best_ref_strength:
                                _best_ref_strength = _rs
                                _rt = getattr(_ref, "ref_type", None)
                                _best_ref_type = (
                                    (
                                        _rt.value
                                        if hasattr(_rt, "value")
                                        else str(_rt)
                                    )
                                    if _rt
                                    else "none"
                                )
                        _panel_module_context["self_ref"] = {
                            "is_discussing_bot": _any_relevant,
                            "strength": _best_ref_strength,
                            "ref_type": _best_ref_type,
                        }
                    # 主观存在判定注入
                    if self._cached_presence_verdict:
                        _panel_module_context["presence"] = dict(
                            self._cached_presence_verdict
                        )
                    # 消息路由分类汇总注入
                    if self._cached_route_summary:
                        _panel_module_context["route_summary"] = dict(
                            self._cached_route_summary
                        )
                    # 技能生命周期摘要注入
                    if self._cached_skill_review:
                        _panel_module_context["skill_review"] = dict(
                            self._cached_skill_review
                        )
                    # 自适应管线摘要注入
                    if self._cached_pipeline_summary:
                        _panel_module_context["adaptive_pipeline"] = dict(
                            self._cached_pipeline_summary
                        )
                    # 被误解信号注入
                    if self._cached_misunderstanding_signal:
                        _sig = self._cached_misunderstanding_signal
                        _panel_module_context["misunderstanding"] = {
                            "risk": _sig.risk_score,
                            "type": _sig.signal_type,
                            "cues": _sig.detected_cues[:3],
                        }
                    # 用户负面情绪聚合值注入
                    if self._cached_user_negative_emotion > 5:
                        _panel_module_context["user_negative_emotion"] = {
                            "aggregate": self._cached_user_negative_emotion,
                        }
                    _panel_snap = _adapter.assemble(
                        self._tick_world_snapshot,
                        module_context=_panel_module_context,
                    )
                    _vitals = _panel_snap.vitals
                    _extra_parts = []
                    if self._cached_night_phase:
                        _np_val = (
                            self._cached_night_phase.label()
                            if hasattr(self._cached_night_phase, "label")
                            else str(self._cached_night_phase)
                        )
                        _extra_parts.append(f"节律={_np_val}")
                    if self._cached_watch_level:
                        _wl_val = (
                            self._cached_watch_level.label()
                            if hasattr(self._cached_watch_level, "label")
                            else str(self._cached_watch_level)
                        )
                        _extra_parts.append(f"关注={_wl_val}")
                    if self._cached_presence_state:
                        _sw = float(
                            getattr(
                                self._cached_presence_state,
                                "social_willingness",
                                0.5,
                            )
                            or 0.5
                        )
                        _extra_parts.append(f"社交意愿={_sw:.2f}")
                    if self._cached_presence_verdict:
                        _pv = self._cached_presence_verdict.get(
                            "participation_verdict", ""
                        )
                        _ml = self._cached_presence_verdict.get(
                            "mood_label", ""
                        )
                        if _pv:
                            _extra_parts.append(f"参与={_pv}")
                        if _ml:
                            _extra_parts.append(f"情绪={_ml}")
                    if self._cached_metabolism_state:
                        _cf = float(
                            getattr(
                                self._cached_metabolism_state,
                                "chat_fuel",
                                100.0,
                            )
                            or 100.0
                        )
                        _extra_parts.append(f"聊天燃料={_cf:.0f}")
                    # 群体模式与场景适合度
                    if _panel_snap.scene.active_pattern:
                        _extra_parts.append(
                            f"模式={_panel_snap.scene.active_pattern}({_panel_snap.scene.pattern_confidence:.2f})"
                        )
                    if _panel_snap.scene.atmosphere:
                        _atmo_display = (
                            _panel_snap.scene.atmosphere_label
                            or _panel_snap.scene.atmosphere
                        )
                        _extra_parts.append(f"氛围={_atmo_display}")
                    if _panel_snap.scene.has_multi_thread:
                        _extra_parts.append(
                            f"多线程={_panel_snap.scene.thread_count}"
                        )
                    # 代谢约束
                    if self._cached_metabolism_constraints:
                        _rh = self._cached_metabolism_constraints.get(
                            "reply_length_hint", ""
                        )
                        if _rh and _rh != "normal":
                            _extra_parts.append(f"回复约束={_rh}")
                    # 夜间状态
                    if self._cached_night_summary:
                        _ns_phase = self._cached_night_summary.get(
                            "phase_label", ""
                        )
                        _ns_debt = self._cached_night_summary.get(
                            "sleep_debt", 0.0
                        )
                        if _ns_phase:
                            _extra_parts.append(f"夜间={_ns_phase}")
                        if _ns_debt > 0.2:
                            _extra_parts.append(f"睡眠债={_ns_debt:.2f}")
                    # 技能生命周期摘要
                    if self._cached_skill_review:
                        _sr_promoted = self._cached_skill_review.get(
                            "promoted", 0
                        )
                        _sr_declined = self._cached_skill_review.get(
                            "declined", 0
                        )
                        _sr_retired = self._cached_skill_review.get(
                            "retired", 0
                        )
                        if _sr_promoted or _sr_declined or _sr_retired:
                            _extra_parts.append(
                                f"技能变动=↑{_sr_promoted}↓{_sr_declined}✕{_sr_retired}"
                            )
                    # 自适应管线状态
                    if self._cached_pipeline_summary:
                        _pipe_evts = self._cached_pipeline_summary.get(
                            "event_pool_size", 0
                        )
                        _pipe_fails = self._cached_pipeline_summary.get(
                            "failure_pool_size", 0
                        )
                        _pipe_cands = self._cached_pipeline_summary.get(
                            "total_candidates", 0
                        )
                        if _pipe_evts or _pipe_fails or _pipe_cands:
                            _extra_parts.append(
                                f"管线事件={_pipe_evts} 失败={_pipe_fails} 候选={_pipe_cands}"
                            )
                    # 被误解信号日志
                    if (
                        self._cached_misunderstanding_signal
                        and self._cached_misunderstanding_signal.risk_score
                        > 0.1
                    ):
                        _mis_risk = (
                            self._cached_misunderstanding_signal.risk_score
                        )
                        _mis_type = (
                            self._cached_misunderstanding_signal.signal_type
                        )
                        _extra_parts.append(
                            f"误解风险={_mis_risk:.2f}({_mis_type})"
                        )
                    # 用户负面情绪日志
                    if self._cached_user_negative_emotion > 10:
                        _extra_parts.append(
                            f"负面情绪={self._cached_user_negative_emotion:.1f}"
                        )
                    # 用户印象标签日志
                    if self._cached_user_impression_tags:
                        _extra_parts.append(
                            f"印象标签={','.join(self._cached_user_impression_tags[:5])}"
                        )
                    if _extra_parts:
                        logger.info(
                            f"{self.log_prefix} 扩展面板 "
                            + " ".join(_extra_parts)
                        )
                    # 统一状态仪表盘行（7Bar + ActionVerdict）
                    try:
                        _dash_line = self._get_dashboard_status_line()
                        if _dash_line:
                            logger.info(
                                f"{self.log_prefix} 📊 状态栏 {_dash_line}"
                            )
                    except Exception:
                        pass
            except Exception as _panel_exc:
                logger.debug(f"{self.log_prefix} 面板适配器异常: {_panel_exc}")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 体征面板输出异常: {exc}")

    def _build_state_dashboard(
        self, *, force: bool = False
    ) -> Optional[Dict[str, Any]]:
        """统一状态仪表盘构建 —— 从40+个分散缓存聚合为7个标准化Bar

        这是状态系统的唯一入口，替代所有直接读取_cached_变量的分散逻辑。
        返回值可直接用于日志输出、决策判断、面板展示。

        聚合数据源映射：
          VitalityBar  ← metabolism_state + metabolism_constraints
          SocialBar    ← presence_state + participant_summary + relation_snapshot
          MoodBar      ← metabolism_constraints(boredom/loafing) + emotion_state
          AttentionBar ← attention_snapshot(GAP-W)
          NightBar     ← night_phase + night_summary
          SafetyBar    ← safety_assessment(GAP-R)
          MemoryBar    ← memory_governance_snap(GAP-Q)
        """
        _now = time.time()
        if not force and self._cached_dashboard_snapshot is not None:
            if (
                _now - self._last_dashboard_build_ts
            ) < self._dashboard_ttl_sec:
                return self._cached_dashboard_snapshot
        try:
            from src.core.state_dashboard import get_state_dashboard

            if not self._dashboard_initialized:
                self._dashboard_initialized = True
            _engine = get_state_dashboard(self.stream_id)
            _raw: Dict[str, Any] = {}
            # ── VitalityBar 数据源：代谢引擎 ──
            _ms = getattr(self, "_cached_metabolism_state", None)
            if _ms:
                _raw["energy_ratio"] = float(
                    getattr(_ms, "chat_energy_ratio", 1.0) or 1.0
                )
                _raw["fatigue"] = float(
                    getattr(_ms, "fatigue_level", 0.0) or 0.0
                )
                _mc = getattr(self, "_cached_metabolism_constraints", None)
                if _mc:
                    _raw["recovery_rate"] = float(
                        _mc.get("recovery_rate", 1.0) or 1.0
                    )
                    _raw["debt_modifier"] = float(
                        _mc.get("sleep_debt_modifier", 1.0) or 1.0
                    )
                    _raw["boredom"] = float(_mc.get("boredom", 0.0) or 0.0)
                    _raw["loafing"] = float(
                        _mc.get("loafing_level", 0.0) or 0.0
                    )
                    _raw["annoyance"] = float(
                        _mc.get("annoyance_accumulated", 0.0) or 0.0
                    )
                    _raw["proactive_drive"] = float(
                        _mc.get("proactive_drive", 0.3) or 0.3
                    )
            else:
                _raw["energy_ratio"] = 1.0
                _raw["recovery_rate"] = 1.0
                _raw["debt_modifier"] = 1.0
            # ── SocialBar 数据源：存在感引擎 + 参与者名册 + social_value_core真实值 ──
            _ps = getattr(self, "_cached_presence_state", None)
            if _ps:
                _raw["social_willingness"] = float(
                    getattr(_ps, "social_willingness", 0.5) or 0.5
                )
                _raw["openness"] = float(getattr(_ps, "openness", 0.5) or 0.5)
                _raw["interrupt_tolerance"] = float(
                    getattr(_ps, "interrupt_tolerance", 0.5) or 0.5
                )
                _raw["avoidance"] = float(
                    getattr(_ps, "avoidance_tendency", 0.0) or 0.0
                )
                _raw["loneliness"] = float(
                    getattr(_ps, "loneliness", 0.0) or 0.0
                )
                _raw["social_hunger"] = float(
                    getattr(_ps, "social_hunger", 0.5) or 0.5
                )
            # 从social_value_core读取真实社交值（替代默认全零）
            _focus_uid = getattr(self, "_last_user_id", "") or ""
            if _focus_uid:
                try:
                    from src.modules.social_value.social_value_core import (
                        get_social_value_core,
                    )

                    _svc = get_social_value_core()
                    _sv_record = _svc.get_social_record_sync(
                        _focus_uid, self.stream_id
                    )
                    if _sv_record:
                        _raw["core_social_value"] = float(
                            _sv_record.value or 0.0
                        )
                        _raw["core_trust_value"] = float(
                            _sv_record.trust_value or 0.0
                        )
                        _raw["core_annoyance"] = float(
                            _sv_record.annoyance_value or 0.0
                        )
                        _raw["core_affection"] = float(
                            (_sv_record.positive_dim or 0.0)
                            - abs(_sv_record.negative_dim or 0.0) * 0.3
                        )
                        _raw["core_interaction_count"] = int(
                            _sv_record.interaction_count or 0
                        )
                except Exception:
                    pass
            # 如果social_value_core没读到，尝试从relation_result回填
            if "core_social_value" not in _raw:
                _rr = getattr(self, "_cached_relation_result", None)
                if isinstance(_rr, dict):
                    _csv = float(_rr.get("social_value", 0.0) or 0.0)
                    if _csv > 0.01:
                        _raw["core_social_value"] = _csv
                        _raw["core_trust_value"] = float(
                            _rr.get("trust_value", 0.0) or 0.0
                        )
                        _raw["core_affection"] = float(
                            _rr.get("affection", 0.0) or 0.0
                        )
            _part = getattr(self, "_cached_participant_summary", None)
            if isinstance(_part, dict):
                _hot = int(_part.get("hot_count", 0) or 0)
                _warm = int(_part.get("warm_count", 0) or 0)
                _total_p = max(
                    1, _hot + _warm + int(_part.get("normal_count", 0) or 0)
                )
                _raw["group_engagement_score"] = (
                    _hot * 1.0 + _warm * 0.6
                ) / max(1, _total_p)
            # ── MoodBar 数据源：代谢约束 + 情绪追踪 ──
            if "mood" not in _raw:
                _raw["mood"] = 0.5
            if "curiosity" not in _raw:
                _raw["curiosity"] = 0.3
            _efr = getattr(self, "_cached_emotion_feedback_report", None)
            if isinstance(_efr, dict):
                _mood_delta = float(_efr.get("net_mood_delta", 0.0) or 0.0)
                _raw["mood"] = max(0.05, min(0.95, 0.5 + _mood_delta))
                _raw["curiosity"] = float(
                    _efr.get("curiosity_component", 0.3) or 0.3
                )
            # ── AttentionBar 数据源：主观注意力流(GAP-W) ──
            _attn = getattr(self, "_cached_attention_snapshot", None)
            if isinstance(_attn, dict):
                _raw["attention_mode"] = str(_attn.get("state", "") or "")
                _raw["visibility_threshold"] = float(
                    _attn.get("visibility_threshold", 0.30) or 0.30
                )
                _raw["process_ratio"] = float(
                    _attn.get("process_ratio", 0.50) or 0.50
                )
                _raw["peek_desire"] = float(
                    _attn.get("peek_desire", 0.3) or 0.3
                )
                _raw["withdrawal_depth"] = float(
                    _attn.get("withdrawal_depth", 0.0) or 0.0
                )
                _raw["empty_peeks"] = int(
                    _attn.get("consecutive_empty_peeks", 0) or 0
                )
                _raw["last_look_ago"] = float(
                    _attn.get("since_last_look_sec", 999.0) or 999.0
                )
            # ── NightBar 数据源：夜间节律系统 ──
            _np = getattr(self, "_cached_night_phase", None)
            if _np:
                _np_val = _np.value if hasattr(_np, "value") else str(_np)
                _raw["night_phase"] = _np_val
            _ns = getattr(self, "_cached_night_summary", None)
            if isinstance(_ns, dict):
                _raw["pressure"] = float(_ns.get("pressure_total", 0.0) or 0.0)
                _raw["burnthrough"] = str(_ns.get("is_burnthrough", "") or "")
                _raw["expression_style"] = str(
                    _ns.get("expression_style", "normal") or "normal"
                )
            # ── SafetyBar 数据源：安全边界融合(GAP-R) ──
            _safe = getattr(self, "_cached_safety_assessment", None)
            if isinstance(_safe, dict):
                _raw["safety_level"] = str(
                    _safe.get("overall_level_label", "安全") or "安全"
                )
                _raw["safety_score"] = float(
                    _safe.get("overall_score", 0.0) or 0.0
                )
                _raw["dominant_threat"] = str(
                    _safe.get("dominant_threat", "") or ""
                )
                _raw["blocked"] = str(_safe.get("blocked", "False") or "False")
                _raw["bar_delta"] = float(_safe.get("bar_penalty", 0.0) or 0.0)
            # ── MemoryBar 数据源：记忆治理(GAP-Q) ──
            _mg = getattr(self, "_cached_memory_governance_snap", None)
            if isinstance(_mg, dict):
                _raw["total"] = int(_mg.get("total_entries", 0) or 0)
                _raw["utilization"] = float(
                    _mg.get("utilization_ratio", 0.0) or 0.0
                )
                _raw["fresh"] = int(_mg.get("fresh_count", 0) or 0)
                _raw["stale"] = int(_mg.get("stale_count", 0) or 0)
                _raw["window"] = int(_mg.get("active_window_size", 0) or 0)
            # ── 构建仪表盘快照 ──
            _snap = _engine.build_dashboard(raw_sources=_raw, force=force)
            self._cached_dashboard_snapshot = _snap.to_display_dict()
            self._last_dashboard_build_ts = _now
            return self._cached_dashboard_snapshot
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 状态仪表盘构建异常: {exc}")
            return None

    def _get_dashboard_verdict(self) -> Dict[str, Any]:
        """获取当前仪表盘ActionVerdict的字典表示（优先使用缓存，避免重复空数据构建）"""
        try:
            from src.core.state_dashboard import get_state_dashboard

            _engine = get_state_dashboard(self.stream_id)
            if _engine._snapshot_cache is not None:
                return _engine._snapshot_cache.verdict.to_dict()
            _v = _engine.get_verdict()
            return _v.to_dict()
        except Exception:
            return {
                "urgency": "可稍后回",
                "process": True,
                "reply": False,
                "reason": "仪表盘初始化中",
                "tone": "正常回应",
                "path": "FALLBACK",
                "confidence": 0.3,
            }

    def _get_dashboard_status_line(self) -> str:
        """获取统一状态栏单行文本（优先使用已构建的缓存，避免重复空数据构建）"""
        if self._cached_dashboard_snapshot and isinstance(
            self._cached_dashboard_snapshot, dict
        ):
            try:
                from src.core.state_dashboard import get_state_dashboard

                _engine = get_state_dashboard(self.stream_id)
                if _engine._snapshot_cache is not None:
                    return _engine._snapshot_cache.to_status_line()
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        try:
            from src.core.state_dashboard import get_state_dashboard

            _engine = get_state_dashboard(self.stream_id)
            _line = _engine.get_status_line()
            if _line:
                return _line
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.energy_manager import (
                get_vitality_pool,
                get_shared_resource_manager,
            )

            pool = get_vitality_pool()
            snap = pool.capture_snapshot(self.stream_id)
            shared_mgr = get_shared_resource_manager()
            shared_vals = shared_mgr.get_shared_values(self.stream_id)
            _chat_val = float(shared_vals.get("chat_value") or 0.0)
            _think_pct = snap.thinking_ratio() * 100.0
            _energy = f"{_chat_val:.0f}%"
        except Exception:
            _energy = "?"
        _np = getattr(self, "_cached_night_phase", None)
        _np_label = "清醒"
        if _np:
            _np_label = (
                _np.label()
                if hasattr(_np, "label")
                else str(_np)
            )
        return f"🔋{_energy} | 😌平静 | 👁扫描 | 👀观察 | ☀️{_np_label} | 🛡️安全"

    async def _emit_target_profile(self, target_uid: str) -> None:
        """第二层：当前对象印象面板 - 对当前交互目标的心理关系快照

        有明确交互对象时输出好感、信任、烦恼、创伤、关系、调教阶段、活跃人格等。
        """
        if not target_uid:
            return
        try:
            profile_segments = [f"对象={target_uid[:8]}"]
            # 优先从世界快照的 target_user 取关系维度
            _snapshot_relation_loaded = False
            tick_snapshot = getattr(self, "_tick_world_snapshot", None)
            if tick_snapshot is not None:
                _t_user = getattr(tick_snapshot, "target_user", None)
                if (
                    _t_user
                    and str(getattr(_t_user, "user_id", "") or "").strip()
                ):
                    _affection = float(
                        getattr(_t_user, "affection", 0.0) or 0.0
                    )
                    _trust = float(getattr(_t_user, "trust_value", 0.0) or 0.0)
                    _annoy = float(
                        getattr(_t_user, "annoyance_value", 0.0) or 0.0
                    )
                    _trauma = float(
                        getattr(_t_user, "trauma_score", 0.0) or 0.0
                    )
                    _pressure = float(
                        getattr(_t_user, "psychological_pressure", 0.0) or 0.0
                    )
                    _rel_label = str(
                        getattr(_t_user, "custom_label", "") or ""
                    ).strip()
                    _social = float(
                        getattr(_t_user, "social_value", 0.0) or 0.0
                    )
                    _injected = self._inject_realtime_emotion(
                        {
                            "annoyance_value": _annoy,
                            "psychological_pressure": _pressure,
                        }
                    )
                    _annoy = float(
                        _injected.get("annoyance_value", _annoy) or _annoy
                    )
                    _pressure = float(
                        _injected.get("psychological_pressure", _pressure)
                        or _pressure
                    )
                    if _affection:
                        profile_segments.append(f"好感={_affection:.1f}")
                    if _trust:
                        profile_segments.append(f"信任={_trust:.1f}")
                    if _annoy > 0:
                        profile_segments.append(f"烦恼={_annoy:.1f}")
                    if _trauma > 0:
                        profile_segments.append(f"创伤={_trauma:.1f}")
                    if _rel_label:
                        profile_segments.append(f"关系={_rel_label}")
                    if _pressure > 0:
                        profile_segments.append(f"心理压力={_pressure:.1f}")
                    if _social != 0:
                        profile_segments.append(f"社交={_social:.1f}")
                    _snapshot_relation_loaded = True
            # 快照无数据时退回旧路径
            if not _snapshot_relation_loaded:
                relation_view = self._resolve_relation_view()
                if (
                    relation_view
                    and target_uid
                    == str(getattr(self, "_last_user_id", "") or "").strip()
                ):
                    affection = float(
                        relation_view.get("affection", 0.0) or 0.0
                    )
                    trust_score = float(
                        relation_view.get(
                            "trust_score",
                            relation_view.get("trust_value", 0.0),
                        )
                        or 0.0
                    )
                    annoy_val = float(
                        relation_view.get("annoyance_value", 0.0) or 0.0
                    )
                    trauma_val = float(
                        relation_view.get("trauma_score", 0.0) or 0.0
                    )
                    pressure = float(
                        relation_view.get("psychological_pressure", 0.0) or 0.0
                    )
                    rel_label = str(
                        relation_view.get("custom_label", "") or ""
                    ).strip()
                    social_value = float(
                        relation_view.get("social_value", 0.0) or 0.0
                    )
                    if affection:
                        profile_segments.append(f"好感={affection:.1f}")
                    if trust_score:
                        profile_segments.append(f"信任={trust_score:.1f}")
                    if annoy_val > 0:
                        profile_segments.append(f"烦恼={annoy_val:.1f}")
                    if trauma_val > 0:
                        profile_segments.append(f"创伤={trauma_val:.1f}")
                    if rel_label:
                        profile_segments.append(f"关系={rel_label}")
                    if pressure > 0:
                        profile_segments.append(f"心理压力={pressure:.1f}")
                    if social_value != 0:
                        profile_segments.append(f"社交={social_value:.1f}")

            # 情绪追踪器维度（权威覆盖：缓存状态优先 > 按uid查询）
            emo_loaded = False
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import (
                    get_emotion_tracker, )

                tracker = get_emotion_tracker(self.stream_id)
                _cached_emo = getattr(self, "_resolved_emo_state", None)
                if _cached_emo is not None:
                    emo_state = _cached_emo
                else:
                    emo_state = tracker.get_user_state(
                        target_uid, create_if_missing=False
                    )
                if emo_state:
                    emo_loaded = True
                    _emo_affection = float(
                        getattr(emo_state, "affection", 0) or 0
                    )
                    if _emo_affection > 0.01 and not any(
                        part.startswith("好感=") for part in profile_segments
                    ):
                        profile_segments.append(f"好感={_emo_affection:.1f}")
                    _emo_trust = float(
                        getattr(emo_state, "trust_score", 0) or 0
                    )
                    if _emo_trust > 0.01 and not any(
                        part.startswith("信任=") for part in profile_segments
                    ):
                        profile_segments.append(f"信任={_emo_trust:.1f}")
                    _tracker_annoy = float(
                        getattr(emo_state, "annoyance", -1) or -1
                    )
                    if _tracker_annoy >= 0:
                        _existing_annoy_idx = [
                            i
                            for i, p in enumerate(profile_segments)
                            if p.startswith("烦恼=")
                        ]
                        if _existing_annoy_idx:
                            profile_segments[_existing_annoy_idx[0]] = (
                                f"烦恼={_tracker_annoy:.1f}"
                            )
                        else:
                            profile_segments.append(
                                f"烦恼={_tracker_annoy:.1f}"
                            )
                    trauma_val = getattr(emo_state, "trauma_score", 0)
                    if trauma_val > 0 and not any(
                        part.startswith("创伤=") for part in profile_segments
                    ):
                        profile_segments.append(f"创伤={trauma_val:.1f}")
                    rel_label = getattr(emo_state, "relationship", "陌生人")
                    if not any(
                        part.startswith("关系=") for part in profile_segments
                    ):
                        profile_segments.append(f"关系={rel_label}")
                    stage = getattr(emo_state, "training_stage", 0)
                    if stage > 0:
                        profile_segments.append(f"调教阶段={stage}")
                    blocked = getattr(emo_state, "is_blocked", False)
                    if blocked:
                        profile_segments.append("已屏蔽")
                    _tracker_pressure = float(
                        getattr(emo_state, "psychological_pressure", -1) or -1
                    )
                    if _tracker_pressure >= 0:
                        _existing_press_idx = [
                            i
                            for i, p in enumerate(profile_segments)
                            if p.startswith("心理压力=")
                        ]
                        if _existing_press_idx:
                            profile_segments[_existing_press_idx[0]] = (
                                f"心理压力={_tracker_pressure:.1f}"
                            )
                        else:
                            profile_segments.append(
                                f"心理压力={_tracker_pressure:.1f}"
                            )
                    stamina = getattr(emo_state, "stamina", 100)
                    if stamina < 100:
                        profile_segments.append(f"体力={stamina:.1f}")
                    try:
                        response_mode = tracker.get_layered_response_mode(
                            target_uid
                        )
                        tone = str(
                            response_mode.get("tone", "neutral") or "neutral"
                        )
                        resp_len = str(
                            response_mode.get("response_length", "normal")
                            or "normal"
                        )
                        profile_segments.append(f"回复模式={tone}/{resp_len}")
                    except Exception:
                        pass
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            # 社交值维度（优先从世界快照读取，避免散读）
            try:
                _snap_social_loaded = False
                tick_snapshot = getattr(self, "_tick_world_snapshot", None)
                if tick_snapshot is not None:
                    _t_user = getattr(tick_snapshot, "target_user", None)
                    if _t_user:
                        _sv = float(
                            getattr(_t_user, "social_value", 0.0) or 0.0
                        )
                        _tv = float(
                            getattr(_t_user, "trust_value", 0.0) or 0.0
                        )
                        if (
                            not any(
                                part.startswith("社交=")
                                for part in profile_segments
                            )
                            and _sv != 0
                        ):
                            profile_segments.append(f"社交={_sv:.1f}")
                        if (
                            not any(
                                part.startswith("信赖度=")
                                for part in profile_segments
                            )
                            and _tv != 0
                        ):
                            profile_segments.append(f"信赖度={_tv:.1f}")
                        _snap_social_loaded = True
                if not _snap_social_loaded:
                    if not any(
                        part.startswith("社交=") for part in profile_segments
                    ) or not any(
                        part.startswith("信赖度=") for part in profile_segments
                    ):
                        from src.chat.heart_flow.energy_manager import (
                            get_vitality_pool,
                        )

                        full_snap = (
                            await get_vitality_pool().capture_full_snapshot(
                                self.stream_id, target_uid
                            )
                        )
                        if not any(
                            part.startswith("社交=")
                            for part in profile_segments
                        ):
                            profile_segments.append(
                                f"社交={full_snap.social_value:.1f}"
                            )
                        if full_snap.trust_value != 0 and not any(
                            part.startswith("信赖度=")
                            for part in profile_segments
                        ):
                            profile_segments.append(
                                f"信赖度={full_snap.trust_value:.1f}"
                            )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            # 活跃人格
            try:
                from src.modules.modcore.dynamic_persona.persona_switcher import (
                    get_persona_switcher, )

                switcher = get_persona_switcher()
                blend = switcher.get_blended_persona_data(self.stream_id)
                persona_obj = blend.get("active_persona")
                if persona_obj and hasattr(persona_obj, "name"):
                    persona_tag = persona_obj.name
                    if not blend.get("is_main", True):
                        persona_tag += "(副)"
                    profile_segments.append(f"人格={persona_tag}")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            logger.info(
                f"{self.log_prefix} 当前对象印象 " + " ".join(profile_segments)
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 当前对象印象输出异常: {exc}")

    def _emit_action_verdict(
        self,
        verdict_action: str,
        verdict_reason: str,
        pipeline_elapsed: float,
        confidence: float = -1.0,
    ) -> None:
        """第三层：动作裁定面板 - 每次决策结果输出

        不论 reply/no_reply/check_later/observe/early_exit/autonomy_guard，
        每次决策都必须可见。
        """
        try:
            gap_since_speak = (
                time.time() - self._last_speak_time
                if self._last_speak_time > 0
                else -1
            )
            quiet_left = max(0.0, self._planner_quiet_until - time.time())
            verdict_parts = [
                f"裁定={verdict_action}",
                f"原因={verdict_reason[:60]}",
                f"连续跳过={self._consecutive_skip_ticks}",
                f"连续发言={self._consecutive_speaks:.1f}",
            ]
            if gap_since_speak >= 0:
                if gap_since_speak < 60:
                    verdict_parts.append(f"上次发言={gap_since_speak:.0f}s前")
                else:
                    verdict_parts.append(
                        f"上次发言={gap_since_speak / 60:.1f}min前"
                    )
            if confidence >= 0:
                verdict_parts.append(f"置信={confidence:.2f}")
            if self._legacy_constraint_hits > 0:
                verdict_parts.append(
                    f"硬约束次数={self._legacy_constraint_hits}"
                )
            if self._last_legacy_penalty > 0:
                verdict_parts.append(
                    f"硬约束惩罚={self._last_legacy_penalty:.1f}"
                )
            if quiet_left > 0:
                verdict_parts.append(f"冷却剩余={quiet_left:.0f}s")
            if self._last_legacy_reason and verdict_action in {
                "legacy_constraint",
                "autonomy_block",
                "early_exit",
                "no_action",
                "no_reply",
            }:
                verdict_parts.append(
                    f"硬约束因子={self._last_legacy_reason[:40]}"
                )
            legacy_breakdown = getattr(self, "_last_legacy_breakdown", None)
            if (
                ("reply" in verdict_action or "proactive" in verdict_action)
                and isinstance(legacy_breakdown, dict)
                and legacy_breakdown
            ):
                summary = (
                    f"R={legacy_breakdown.get('reply_readiness', 0):.1f} "
                    f"B={legacy_breakdown.get('base', 0):.0f} "
                    f"+M={legacy_breakdown.get('mention_bonus', 0):.0f} "
                    f"+Q={legacy_breakdown.get('question_bonus', 0):.0f} "
                    f"+S={legacy_breakdown.get('salience_bonus', 0):.0f} "
                    f"-Res={legacy_breakdown.get('resource_penalty', 0):.0f} "
                    f"-Rel={legacy_breakdown.get('relation_penalty', 0):.0f} "
                    f"-Rep={legacy_breakdown.get('repeat_penalty', 0):.0f}"
                )
                verdict_parts.append(summary)
            verdict_parts.append(f"管线耗时={pipeline_elapsed:.2f}s")
            logger.info(
                f"{self.log_prefix} 动作裁定 " + " ".join(verdict_parts)
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 动作裁定输出异常: {exc}")

    async def _emit_outcome_summary(
        self, did_reply: bool, relation_snapshot: Dict
    ) -> None:
        """第四层：结果汇总面板 - 回复后或跳过后的终值输出

        输出三核心值终值、好感/信任变化、保护态、屏蔽态。
        每次管线末端必须输出（替代旧版节流式最终状态）。
        """
        try:
            relation_view = self._resolve_relation_view(relation_snapshot)
            from src.chat.heart_flow.energy_manager import (
                get_vitality_pool,
                get_shared_resource_manager,
            )

            pool = get_vitality_pool()
            channel_snap = pool.capture_snapshot(self.stream_id)
            shared_mgr = get_shared_resource_manager()
            shared_vals = shared_mgr.get_shared_values(self.stream_id)
            chat_current = float(shared_vals.get("chat_value", 100.0) or 100.0)
            activity_current = float(
                shared_vals.get("activity_level", 50.0) or 50.0
            )
            social_current = float(shared_vals.get("social_value", 0.0) or 0.0)
            thinking_current = float(channel_snap.thinking_ratio() * 100.0)

            outcome_parts = []
            action_tag = "已回复" if did_reply else "未回复"
            outcome_parts.append(f"结果={action_tag}")
            outcome_parts.append(f"聊天当前={chat_current:.1f}")
            outcome_parts.append(f"活跃当前={activity_current:.1f}")
            outcome_parts.append(f"社交当前={social_current:.1f}")
            outcome_parts.append(f"思考当前={thinking_current:.0f}%")

            # 关系变化标记
            rel_social = float(relation_view.get("social_value", 0.0) or 0.0)
            rel_trust = float(relation_view.get("trust_value", 0.0) or 0.0)
            rel_annoy = float(relation_view.get("annoyance_value", 0.0) or 0.0)
            if abs(rel_social) >= 0.1:
                outcome_parts.append(f"社交值={rel_social:.1f}")
            if abs(rel_trust) >= 0.1:
                outcome_parts.append(f"信赖度={rel_trust:.1f}")
            if abs(rel_annoy) >= 0.1:
                outcome_parts.append(f"厌烦度={rel_annoy:.1f}")

            rel_affection = float(relation_view.get("affection", 0.0) or 0.0)
            rel_pressure = float(
                relation_view.get("psychological_pressure", 0.0) or 0.0
            )
            rel_trauma = float(relation_view.get("trauma_score", 0.0) or 0.0)
            if abs(rel_affection) >= 0.1:
                outcome_parts.append(f"好感={rel_affection:.1f}")
            if rel_pressure > 0 and not any(
                part.startswith("心理压力=") for part in outcome_parts
            ):
                outcome_parts.append(f"心理压力={rel_pressure:.1f}")
            if rel_trauma > 0 and not any(
                part.startswith("创伤=") for part in outcome_parts
            ):
                outcome_parts.append(f"创伤={rel_trauma:.1f}")
            if self._legacy_constraint_hits > 0:
                outcome_parts.append(
                    f"硬约束次数={self._legacy_constraint_hits}"
                )
            if self._last_legacy_penalty > 0:
                outcome_parts.append(
                    f"硬约束惩罚={self._last_legacy_penalty:.1f}"
                )
            quiet_left = max(0.0, self._planner_quiet_until - time.time())
            if quiet_left > 0:
                outcome_parts.append(f"冷却剩余={quiet_left:.0f}s")
            if self._last_legacy_reason:
                outcome_parts.append(
                    f"硬约束因子={self._last_legacy_reason[:40]}"
                )
            legacy_breakdown = getattr(self, "_last_legacy_breakdown", None)
            if (
                did_reply
                and isinstance(legacy_breakdown, dict)
                and legacy_breakdown
            ):
                outcome_parts.append(
                    f"结算可回复度={
                        legacy_breakdown.get(
                            'reply_readiness',
                            0):.1f}"
                )
                outcome_parts.append(
                    f"结算基础分={
                        legacy_breakdown.get(
                            'base', 0):.1f}"
                )
                outcome_parts.append(
                    f"结算提及加分={
                        legacy_breakdown.get(
                            'mention_bonus',
                            0):.1f}"
                )
                outcome_parts.append(
                    f"结算问句加分={
                        legacy_breakdown.get(
                            'question_bonus',
                            0):.1f}"
                )
                outcome_parts.append(
                    f"结算显著性加分={
                        legacy_breakdown.get(
                            'salience_bonus',
                            0):.1f}"
                )
                outcome_parts.append(
                    f"结算关系加分={
                        legacy_breakdown.get(
                            'rapport_bonus',
                            0):.1f}"
                )
                outcome_parts.append(
                    f"结算资源惩罚={
                        legacy_breakdown.get(
                            'resource_penalty',
                            0):.1f}"
                )
                outcome_parts.append(
                    f"结算关系惩罚={
                        legacy_breakdown.get(
                            'relation_penalty',
                            0):.1f}"
                )
                outcome_parts.append(
                    f"结算重复惩罚={
                        legacy_breakdown.get(
                            'repeat_penalty',
                            0):.1f}"
                )
                outcome_parts.append(
                    f"结算骚扰惩罚={
                        legacy_breakdown.get(
                            'harass_penalty',
                            0):.1f}"
                )
                outcome_parts.append(
                    f"结算不信任惩罚={
                        legacy_breakdown.get(
                            'distrust_penalty',
                            0):.1f}"
                )

            # 保护态/屏蔽态
            target_uid = getattr(self, "_last_user_id", "")
            if target_uid:
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker, )

                    tracker = get_emotion_tracker(self.stream_id)
                    emo_state = tracker.get_user_state(
                        target_uid, create_if_missing=False
                    )
                    if emo_state:
                        if getattr(emo_state, "is_blocked", False):
                            outcome_parts.append("状态=已屏蔽")
                        stamina = getattr(emo_state, "stamina", 100)
                        if stamina < 20:
                            outcome_parts.append(f"耐力低={stamina:.0f}")
                except Exception:
                    pass

            # 创伤系统状态
            try:
                from src.modules.trauma.trauma_system import get_trauma_system

                trauma_sys = get_trauma_system()
                trauma_st = trauma_sys.get_state()
                if trauma_st.stress_accumulation > 5:
                    outcome_parts.append(
                        f"压力蓄积={trauma_st.stress_accumulation:.1f}"
                    )
                if trauma_st.inner_chaos_level > 3:
                    outcome_parts.append(
                        f"混乱={trauma_st.inner_chaos_level:.1f}"
                    )
                if trauma_st.surface_mask_strength > 0:
                    outcome_parts.append(
                        f"伪装={trauma_st.surface_mask_strength:.1f}"
                    )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            # 用户维度详细心理指标（合并自旧版最终状态）
            if target_uid:
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker as _get_emo, )

                    _trk = _get_emo(self.stream_id)
                    _es = _trk.get_user_state(
                        target_uid, create_if_missing=False
                    )
                    if _es:
                        _trauma = getattr(_es, "trauma_score", 0.0)
                        if _trauma > 0:
                            outcome_parts.append(f"创伤={_trauma:.1f}")
                        _pressure = getattr(_es, "psychological_pressure", 0.0)
                        if _pressure > 0:
                            outcome_parts.append(f"心理压力={_pressure:.1f}")
                        _ts = getattr(_es, "training_stage", 0)
                        if _ts > 0:
                            outcome_parts.append(f"调教阶段={_ts}")
                            _tp = getattr(_es, "training_progress", 0.0)
                            if _tp > 0:
                                outcome_parts.append(f"调教进度={_tp:.1f}%")
                        _sub = getattr(_es, "submission_level", 0.0)
                        if _sub > 0:
                            outcome_parts.append(f"顺从={_sub:.1f}")
                except Exception:
                    pass

            # 频率调节因子
            try:
                freq_ctrl = (
                    frequency_control_manager.get_or_create_frequency_control(
                        self.stream_id
                    )
                )
                _freq = freq_ctrl.get_talk_frequency_adjust()
                outcome_parts.append(f"频率={_freq:.2f}")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            logger.info(f"{self.log_prefix} 📊 " + " ".join(outcome_parts))
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 结果汇总输出异常: {exc}")

    # ═══════════════════════════════════════════════════
    #  阶段 2 辅助：动态阈值与 @检测
    # ═══════════════════════════════════════════════════

    def _compute_dynamic_threshold(self) -> int:
        """消息触发阈值，始终为1，任何新消息都应被处理"""
        return 1

    def _find_pinged_message(self, batch: List):
        """在消息批次中查找 @机器人 或提及机器人的消息"""
        if not global_config.chat.mentioned_bot_reply:
            return None
        for msg in batch:
            if getattr(msg, "is_mentioned", False) or getattr(
                msg, "is_at", False
            ):
                return msg
        return None

    def _is_message_pinged(self, msg) -> bool:
        """判断单条消息是否@了机器人"""
        return bool(
            getattr(msg, "is_mentioned", False) or getattr(msg, "is_at", False)
        )

    # ═══════════════════════════════════════════════════
    #  阶段 3：感知引擎
    # ═══════════════════════════════════════════════════

    def _should_run_perception(self, now: float) -> bool:
        """感知引擎有冷却间隔，避免每轮都跑 LLM

        修复：DORMANT等级时跳过感知（除非冷却超长>30min强制刷新），
        避免在系统应该休息时仍消耗token跑LLM。
        """
        _cooldown_ok = (
            now - self._last_perception_ts
        ) >= _PERCEPTION_COOLDOWN_SEC
        if not _cooldown_ok:
            return False
        _cached = getattr(self, "_cached_awareness", None)
        if _cached is not None:
            from src.chat.proactive.perception_engine import AlertGrade

            _grade = getattr(_cached, "alert_grade", None)
            if isinstance(_grade, AlertGrade):
                if _grade == AlertGrade.DORMANT:
                    _elapsed_min = (now - self._last_perception_ts) / 60.0
                    if _elapsed_min < 30.0:
                        return False
                elif _grade == AlertGrade.FAINT:
                    _elapsed_min = (now - self._last_perception_ts) / 60.0
                    if _elapsed_min < 10.0:
                        return False
        return True

    async def _invoke_perception(self, messages: List, now: float):
        """调用觉察引擎，收集环境感知快照"""
        try:
            from src.chat.proactive.perception_engine import (
                get_awareness_engine,
            )

            engine = get_awareness_engine()
            raw_msg_dicts = self._serialize_messages_for_perception(messages)
            self_state = self._collect_self_state()
            relation_state = self._collect_relation_state(messages)
            has_ping = any(
                getattr(m, "is_mentioned", False) or getattr(m, "is_at", False)
                for m in messages
            )
            verdict = await engine.gather_awareness(
                channel_id=self.stream_id,
                raw_messages=raw_msg_dicts,
                self_state=self_state,
                relation_state=relation_state,
                mention_bot=has_ping,
            )
            logger.debug(
                f"{self.log_prefix} 感知结果: "
                f"警戒={verdict.alert_grade.name} "
                f"参与度={verdict.engagement_pull:.2f} "
                f"群体={verdict.crowd_vibe.value}"
            )
            return verdict
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 感知引擎调用失败: {exc}")
            return None

    def _serialize_messages_for_perception(
        self, messages: List
    ) -> List[Dict[str, Any]]:
        """将数据库消息对象序列化为感知引擎期望的字典列表"""
        result = []
        for msg in messages:
            result.append(
                {
                    "text": getattr(msg, "processed_plain_text", "") or "",
                    "user_id": getattr(msg, "user_id", "") or "",
                    "user_name": getattr(msg, "user_nickname", "") or "",
                    "timestamp": getattr(msg, "time", 0.0) or time.time(),
                }
            )
        return result

    def _collect_self_state(self) -> Dict[str, Any]:
        """收集机器人自身状态，供感知引擎参考"""
        state: Dict[str, Any] = {
            "channel_id": self.stream_id,
            "cycle_count": self._enhanced_round,
            "skip_ticks": self._consecutive_skip_ticks,
        }
        try:
            from src.chat.heart_flow.energy_manager import get_vitality_pool

            snap = get_vitality_pool().capture_snapshot(self.stream_id)
            state["chat_pool"] = snap.chat_pool
            state["thinking_value"] = snap.thinking_value
            state["annoyance"] = snap.annoyance_level
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return state

    def _collect_relation_state(self, messages: List) -> Dict[str, Any]:
        """收集频道社交关系信息"""
        unique_speakers = set()
        for msg in messages:
            uid = getattr(msg, "user_id", "") or ""
            if uid:
                unique_speakers.add(uid)
        return {
            "participant_count": len(unique_speakers),
            "speaker_ids": list(unique_speakers),
        }

    def _sample_channel_ambient(self) -> Optional[Dict[str, Any]]:
        """采样频道氛围"""
        try:
            from src.chat.heart_flow.emotion_stream import (
                get_channel_mood_tracker,
            )

            tracker = get_channel_mood_tracker()
            ledger = tracker.fetch_mood(self.stream_id)
            return {
                "vitality": ledger.vitality,
                "weariness": ledger.weariness,
                "vexation": ledger.vexation,
                "category": (
                    ledger.category.value
                    if hasattr(ledger.category, "value")
                    else str(ledger.category)
                ),
            }
        except Exception:
            return None

    # ═══════════════════════════════════════════════════
    #  阶段 4：内心独白
    # ═══════════════════════════════════════════════════

    def _compute_llm_call_level(
        self,
        now: float,
        messages: List,
        pinged,
        desire_level: float = 5.0,
    ) -> Tuple[int, float, float, bool]:
        """【自适应动态阈值算法】决定是否调用小模型/大模型

        设计原则：尽量减少小模型调用次数，省token

        使用46维因子向量的自适应学习器，根据历史反馈动态调整权重。

        返回值：
        - 0: 跳过小模型（直接观察/简短敷衍）
        - 1: 调用小模型（简单回复）
        - 2: 升级大模型（复杂逻辑）
        """
        if (now - self._last_voice_ts) < _VOICE_COOLDOWN_SEC:
            return (0, 1.0, 0.0, False)

        ratio_based_skip = False

        try:
            from src.core.adaptive_threshold_learner_v2 import (
                get_adaptive_threshold_learner_v2,
            )

            _learner = get_adaptive_threshold_learner_v2(self.stream_id)
            _social = {
                "affection": float(
                    getattr(self, "_cached_affection_value", 0.0) or 0.0
                ),
                "trust": float(
                    getattr(self, "_cached_trust_value", 0.0) or 0.0
                ),
                "annoyance": float(
                    getattr(self, "_cached_annoyance_value", 0.0) or 0.0
                ),
                "pressure": float(
                    getattr(self, "_cached_pressure_value", 0.0) or 0.0
                ),
                "relationship_level": float(
                    getattr(self, "_cached_relationship_level", 2) or 2
                ),
                "group_support": float(
                    getattr(self, "_cached_group_support", 0.0) or 0.0
                ),
                "group_attack": float(
                    getattr(self, "_cached_group_attack", 0.0) or 0.0
                ),
                "familiarity": float(
                    getattr(self, "_cached_familiarity", 0.0) or 0.0
                ),
                "dominance": float(
                    getattr(self, "_cached_dominance", 0.0) or 0.0
                ),
            }
            _energy = {}
            if self._cached_metabolism_state:
                _energy["energy_ratio"] = float(
                    getattr(self._cached_metabolism_state, "energy_ratio", 0.8)
                    or 0.8
                )
                _energy["thinking_ratio"] = float(
                    getattr(
                        self._cached_metabolism_state, "thinking_ratio", 0.8
                    )
                    or 0.8
                )
                _energy["mood_valence"] = float(
                    getattr(self._cached_metabolism_state, "mood_valence", 0.5)
                    or 0.5
                )
                _energy["mood_arousal"] = float(
                    getattr(self._cached_metabolism_state, "mood_arousal", 0.5)
                    or 0.5
                )
                _energy["stress_accumulation"] = float(
                    getattr(
                        self._cached_metabolism_state,
                        "stress_accumulation",
                        0.0,
                    )
                    or 0.0
                )
                _energy["trauma_score"] = float(
                    getattr(self._cached_metabolism_state, "trauma_score", 0.0)
                    or 0.0
                )
            else:
                _energy = {
                    "energy_ratio": 0.8,
                    "thinking_ratio": 0.8,
                    "mood_valence": 0.5,
                    "mood_arousal": 0.5,
                    "stress_accumulation": 0.0,
                    "trauma_score": 0.0,
                }
            _group = {
                "pattern_confidence": 0.0,
                "pattern_type": 0.0,
                "active_users": 0.0,
                "message_velocity": 0.0,
                "topic_stability": 1.0,
                "atmosphere_tension": 0.0,
                "me_mentioned_ratio": 0.0,
                "excitement_level": 0.0,
            }
            try:
                if (
                    self._cached_pattern_evidence
                    and len(self._cached_pattern_evidence) > 0
                ):
                    _top_ev = self._cached_pattern_evidence[0]
                    _group["pattern_confidence"] = float(
                        getattr(_top_ev, "confidence", 0.0) or 0.0
                    )
                    _pat = getattr(_top_ev, "pattern", None)
                    if _pat and hasattr(_pat, "value"):
                        _group["pattern_type"] = float(
                            hash(str(_pat.value)) % 11
                        )
                    _group["active_users"] = float(
                        len(getattr(_top_ev, "involved_users", set()) or set())
                    )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            try:
                if self._cached_scene_snapshot:
                    _atmo = getattr(
                        self._cached_scene_snapshot, "atmosphere", None
                    )
                    if _atmo and hasattr(_atmo, "value"):
                        _atmo_str = str(_atmo.value)
                        if (
                            "argument" in _atmo_str
                            or "confrontation" in _atmo_str
                        ):
                            _group["atmosphere_tension"] = 0.8
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            _time_m = {
                "recency": 1.0,
                "conversation_length": float(len(messages) if messages else 0),
                "since_last_reply": (now - self._last_voice_ts) / 60.0,
            }
            _history_m = {
                "skip_count": 0,
                "llm_call_count": 0,
                "success_count": 0,
                "fail_count": 0,
                "streak_skip": 0,
                "streak_llm": 0,
                "desire_level": desire_level,
            }
            _level, _skip, _esc, _tentative = _learner.decide_level(
                messages=messages,
                social_metrics=_social,
                energy_metrics=_energy,
                group_metrics=_group,
                time_metrics=_time_m,
                history_metrics=_history_m,
                pinged=(pinged is not None),
            )
            if _tentative:
                logger.info(
                    f"{self.log_prefix} 🎯 试探性发牢骚：沉默够久但害羞指数低，轻量试探"
                )
            logger.debug(
                f"{self.log_prefix} [自适应阈值] skip={_skip:.3f} esc={_esc:.3f} → level={_level}"
            )
            return (_level, _skip, _esc, _tentative)
        except Exception as _e:
            logger.debug(f"{self.log_prefix} [自适应阈值] 回退简单逻辑: {_e}")
        if pinged is not None:
            return (1, 0.0, 0.3, False)
        if (now - self._last_voice_ts) < _VOICE_COOLDOWN_SEC:
            return (0, 1.0, 0.0, False)
        _skip_score = 0.0
        _escalate_score = 0.0
        _msg_count = len(messages) if messages else 0
        try:
            if self._cached_metabolism_state:
                _energy_ratio = float(
                    getattr(self._cached_metabolism_state, "energy_ratio", 0.8)
                    or 0.8
                )
                _thinking_ratio = float(
                    getattr(
                        self._cached_metabolism_state, "thinking_ratio", 0.8
                    )
                    or 0.8
                )
                if _energy_ratio < 0.3:
                    _skip_score += 0.4
                elif _energy_ratio < 0.5:
                    _skip_score += 0.2
                if _thinking_ratio < 0.3:
                    _escalate_score -= 0.3
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            _hour = time.localtime().tm_hour
            if _hour >= 23 or _hour < 6:
                _skip_score += 0.25
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            if (
                self._cached_pattern_evidence
                and len(self._cached_pattern_evidence) > 0
            ):
                for _ev in self._cached_pattern_evidence[:3]:
                    _conf = float(getattr(_ev, "confidence", 0.0) or 0.0)
                    _pat = getattr(_ev, "pattern", None)
                    if _pat and hasattr(_pat, "value"):
                        _pat_val = str(_pat.value)
                        _user_count = len(
                            getattr(_ev, "involved_users", set()) or set()
                        )
                        if _pat_val in (
                            "copycat_chain",
                            "spectator_mode",
                            "quiet_reflection",
                        ):
                            _skip_score += _conf * 0.5
                        elif _pat_val in (
                            "pile_on",
                            "conflict_escalation",
                            "meme_storm",
                        ):
                            _escalate_score += (
                                _conf * 0.3 * min(1.0, _user_count / 3.0)
                            )
                        elif _pat_val in (
                            "gossip_event",
                            "emotional_contagion",
                        ):
                            _escalate_score += _conf * 0.2
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            if self._cached_scene_snapshot:
                _joinable = bool(
                    getattr(
                        self._cached_scene_snapshot, "suitable_to_join", True
                    )
                )
                if not _joinable:
                    _skip_score += 0.6
                _atmo = getattr(
                    self._cached_scene_snapshot, "atmosphere", None
                )
                if _atmo and hasattr(_atmo, "value"):
                    _atmo_val = str(_atmo.value)
                    if _atmo_val in ("argument", "confrontation"):
                        _escalate_score += 0.25
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        if _msg_count >= 10:
            _escalate_score += 0.15
        elif _msg_count >= 5:
            _escalate_score += 0.08
        try:
            if self._cached_presence_state:
                _quiet = float(
                    getattr(
                        self._cached_presence_state, "quiet_preference", 0.0
                    )
                    or 0.0
                )
                if _quiet > 0.6:
                    _skip_score += _quiet * 0.3
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        _net_score = _skip_score - _escalate_score
        logger.debug(
            f"{
                self.log_prefix} [简单阈值] skip={
                _skip_score:.2f} esc={
                _escalate_score:.2f} "
            f"net={
                    _net_score:.2f} → level={
                        '跳过' if _net_score >= 0.5 else '小模型' if _net_score >= -
                        0.3 else '大模型'}"
        )
        if _net_score >= 0.5:
            return (0, _skip_score, _escalate_score, False)
        if _net_score <= -0.3:
            return (2, _skip_score, _escalate_score, False)
        return (1, _skip_score, _escalate_score, False)

    def _should_run_voice(self, now: float, messages: List, pinged) -> bool:
        """系统算法 + 冷却综合判断：是否调用小模型（内心独白）

        设计原则：尽量减少小模型调用次数
        """
        _level, _, _, _ = self._compute_llm_call_level(now, messages, pinged)
        if _level == 0:
            logger.info(
                f"{self.log_prefix} 🎯 动态阈值判定: 跳过小模型( level=0 )"
            )
            return False
        return True

    def _should_upgrade_to_large_model(
        self, now: float, messages: List, pinged
    ) -> bool:
        """判断是否需要升级到大模型"""
        _level, _, _, _ = self._compute_llm_call_level(now, messages, pinged)
        return _level == 2

    async def _invoke_inner_voice(
        self,
        messages: List,
        awareness,
        ambient: Optional[Dict[str, Any]],
        now: float,
    ):
        """调用自我对话引擎，生成内心想法；同时构建统一 WorldSnapshot 并缓存供后续阶段复用"""
        try:
            from src.chat.heart_flow.inner_voice import (
                get_self_dialogue_engine,
            )

            engine = get_self_dialogue_engine(self.stream_id)
            # 取最新一条消息作为触发文本
            latest = messages[-1] if messages else None
            raw_text = (
                getattr(latest, "processed_plain_text", "") if latest else ""
            )
            speaker_name = (
                getattr(latest, "user_nickname", "") or "" if latest else ""
            )
            speaker_id = getattr(latest, "user_id", "") or "" if latest else ""
            # 复用已有世界快照或补建（避免重复采集）
            tick_snapshot = getattr(self, "_tick_world_snapshot", None)
            if tick_snapshot is None or not tick_snapshot.target_user.user_id:
                try:
                    from src.core.world_snapshot import build_world_snapshot

                    tick_snapshot = await build_world_snapshot(
                        self.stream_id, speaker_id
                    )
                    self._tick_world_snapshot = tick_snapshot
                except Exception as _snap_err:
                    logger.debug(
                        f"{self.log_prefix} 统一快照构建失败，退回独立采集: {_snap_err}"
                    )
            # 从能量和氛围提取参数
            energy_params = self._extract_voice_energy_params()
            mood_params = self._extract_voice_mood_params(ambient)
            # 构建对话历史片段
            dialogue_fragment = self._build_dialogue_fragment(messages)
            relation_view = self._resolve_relation_view()
            # 刷新快照中滞后的实时情绪值（重复回写发生在快照构建后）
            if tick_snapshot is not None and relation_view:
                _tu = tick_snapshot.target_user
                _rv_ann = float(relation_view.get("annoyance_value", 0.0) or 0.0)
                _rv_prs = float(relation_view.get("psychological_pressure", 0.0) or 0.0)
                if _rv_ann > _tu.annoyance_value:
                    _tu.annoyance_value = _rv_ann
                if _rv_prs > _tu.psychological_pressure:
                    _tu.psychological_pressure = _rv_prs
            voice_context = self._build_inner_voice_context(
                latest, ambient, relation_snapshot=relation_view
            )
            phase_label = self._query_flow_phase()
            verdict = await engine.generate_reflection(
                heart_state_label=phase_label,
                raw_text=raw_text,
                speaker_name=speaker_name,
                dialogue_history=dialogue_fragment,
                assurance_score=energy_params.get("assurance", 0.5),
                involvement_score=energy_params.get("involvement", 0.5),
                mental_drain=energy_params.get("drain", 0.0),
                endurance=energy_params.get("endurance", 100.0),
                irritation=mood_params.get("irritation", 0.0),
                wound_score=mood_params.get("wound", 0.0),
                readiness=energy_params.get("readiness", 1.0),
                speaker_id=speaker_id,
                world_snapshot=tick_snapshot,
                extra_context=voice_context,
            )
            if verdict.is_valid:
                logger.debug(
                    f"{self.log_prefix} 内心独白: "
                    f"欲望={verdict.reply_desire_level} "
                    f"情绪={verdict.current_mood} "
                    f"意图数={len(verdict.intents)}"
                )
            return verdict
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 内心独白调用失败: {exc}")
            return None

    def _extract_voice_energy_params(self) -> Dict[str, float]:
        """从能量快照提取供内心独白使用的参数"""
        result = {
            "assurance": 0.5,
            "involvement": 0.5,
            "drain": 0.0,
            "endurance": 100.0,
            "readiness": 1.0,
        }
        try:
            from src.chat.heart_flow.energy_manager import (
                get_vitality_pool,
                get_shared_resource_manager,
            )

            pool = get_vitality_pool()
            snap = pool.capture_snapshot(self.stream_id)
            shared_values = get_shared_resource_manager().get_shared_values(
                self.stream_id
            )
            chat_value = float(shared_values.get("chat_value") or 0.0)
            activity_level = float(shared_values.get("activity_level") or 0.0)
            social_value = float(shared_values.get("social_value") or 0.0)
            chat_factor = max(
                0.0, min(1.0, chat_value / max(snap.chat_ceiling, 1.0))
            )
            thinking_factor = snap.thinking_ratio()
            activity_factor = max(0.0, min(1.0, activity_level / 100.0))
            social_factor = max(0.0, min(1.0, (social_value + 100.0) / 200.0))
            result["assurance"] = chat_factor
            result["involvement"] = thinking_factor
            result["drain"] = snap.annoyance_level
            result["endurance"] = max(0.0, 100.0 - snap.annoyance_level)
            _raw_readiness = max(
                0.05,
                min(
                    1.0,
                    thinking_factor * 0.4
                    + chat_factor * 0.25
                    + activity_factor * 0.2
                    + social_factor * 0.15,
                ),
            )
            if getattr(self, "_peek_mode_active", False):
                _raw_readiness = max(0.05, _raw_readiness * 0.65)
            try:
                from src.chat.heart_flow.emotion_driven_core import (
                    get_emotion_driven_core,
                )

                _emo_snap = get_emotion_driven_core().get_state_snapshot(
                    self.stream_id
                )
                _boredom_val = float(getattr(_emo_snap, "boredom", 0.0) or 0.0)
                if _boredom_val > 0.6:
                    _raw_readiness = max(
                        0.05,
                        _raw_readiness * (1.0 - (_boredom_val - 0.6) * 0.5),
                    )
                elif _boredom_val > 0.3:
                    _raw_readiness = max(0.05, _raw_readiness * 0.9)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            try:
                _meta = getattr(self, "_cached_metabolism_state", None)
                if _meta is not None:
                    _sm = float(getattr(_meta, "surface_mask", 10.0) or 10.0)
                    _ic = float(getattr(_meta, "inner_chaos", 0.0) or 0.0)
                    _sub = float(
                        getattr(_meta, "submission_level", 0.0) or 0.0
                    )
                    if _ic > 5.0:
                        _raw_readiness = max(
                            0.05, _raw_readiness * max(0.3, 1.0 - _ic * 0.05)
                        )
                    if _sub > 3.0:
                        _raw_readiness = max(
                            0.05,
                            _raw_readiness * (1.0 - min(0.4, _sub * 0.08)),
                        )
                    if _sm < 4.0:
                        _raw_readiness = min(
                            1.0, _raw_readiness * (1.0 + (4.0 - _sm) * 0.08)
                        )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            try:
                from src.modules.trauma.trauma_system import get_trauma_system

                _ts = get_trauma_system()
                _tstate = _ts.get_state()
                _t_enum = getattr(_tstate, "current_state", None)
                if _t_enum is not None:
                    _tname = (
                        str(_t_enum.value)
                        if hasattr(_t_enum, "value")
                        else str(_t_enum)
                    )
                    if "崩溃" in _tname or "即将" in _tname:
                        _raw_readiness = max(0.03, _raw_readiness * 0.2)
                    elif (
                        "半崩" in _tname
                        or "勉强" in _tname
                        or "动摇" in _tname
                    ):
                        _raw_readiness = max(0.05, _raw_readiness * 0.5)
                    elif "滑落" in _tname:
                        _raw_readiness = max(0.05, _raw_readiness * 0.75)
                _stress = float(
                    getattr(_tstate, "stress_accumulation", 0.0) or 0.0
                )
                if _stress > 15.0:
                    _raw_readiness = max(
                        0.03,
                        _raw_readiness
                        * (1.0 - min(0.5, (_stress - 15.0) * 0.02)),
                    )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            result["readiness"] = _raw_readiness
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return result

    def _extract_voice_mood_params(
        self, ambient: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """从氛围采样提取供内心独白使用的参数"""
        result = {"irritation": 0.0, "wound": 0.0}
        if ambient is None:
            return result
        result["irritation"] = float(ambient.get("vexation", 0.0))
        result["wound"] = float(ambient.get("weariness", 0.0)) / 100.0
        return result

    def _build_dialogue_fragment(self, messages: List) -> List[Dict]:
        """将最近消息构建为内心独白使用的对话片段格式"""
        fragment = []
        for msg in messages[-8:]:
            entry: Dict[str, Any] = {
                "role": "user",
                "content": getattr(msg, "processed_plain_text", "") or "",
                "name": getattr(msg, "user_nickname", "") or "",
            }
            fragment.append(entry)
        return fragment

    def _build_inner_voice_context(
        self,
        latest: Optional[Any],
        ambient: Optional[Dict[str, Any]],
        relation_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """为内心独白整理目标画像、记忆摘要和后续执行提示。"""
        target_uid = (
            str(getattr(latest, "user_id", "") or "").strip() if latest else ""
        )
        latest_text = (
            str(
                getattr(latest, "processed_plain_text", "")
                or getattr(latest, "content", "")
                or ""
            ).strip()
            if latest
            else ""
        )
        latest_repeat_short = (
            bool(getattr(latest, "is_repeated_short_input", False))
            if latest
            else False
        )
        latest_continuation_hint = (
            str(getattr(latest, "context_continuation_hint", "") or "").strip()
            if latest
            else ""
        )
        profile_bits: List[str] = []
        memory_bits: List[str] = []
        execution_bits: List[str] = []

        relation_snapshot = self._resolve_relation_view(relation_snapshot)
        if relation_snapshot:
            custom_label = str(relation_snapshot.get("custom_label", "") or "")
            relation_level = relation_snapshot.get("relationship_level", None)
            social_value = float(
                relation_snapshot.get("social_value", 0.0) or 0.0
            )
            trust_value = float(
                relation_snapshot.get("trust_value", 0.0) or 0.0
            )
            annoyance_value = float(
                relation_snapshot.get("annoyance_value", 0.0) or 0.0
            )
            mood = str(relation_snapshot.get("mood", "") or "")
            if custom_label:
                profile_bits.append(f"关系标签={custom_label}")
            if relation_level is not None:
                profile_bits.append(f"关系级别={relation_level}")
            profile_bits.append(f"社交={social_value:.1f}")
            if trust_value:
                profile_bits.append(f"信任={trust_value:.1f}")
            if annoyance_value:
                profile_bits.append(f"烦躁={annoyance_value:.1f}")
            if mood:
                profile_bits.append(f"当前印象气氛={mood}")

        if target_uid:
            try:
                from src.person_info.person_info import get_unified_profile_hub

                hub = get_unified_profile_hub()
                relationships = hub.get_top_relationships(target_uid, limit=3)
                if relationships:
                    rel_text = "、".join(
                        f"{str(uid)[:8]}:{float(weight):.2f}"
                        for uid, weight in relationships
                        if str(uid or "").strip()
                    )
                    if rel_text:
                        profile_bits.append(f"关联对象={rel_text}")
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            try:
                from src.chat.proactive.session_tracker import (
                    get_memoir_cabinet,
                )

                memoir = get_memoir_cabinet().retrieve(target_uid)
                if memoir is not None:
                    phase = getattr(memoir, "phase", None)
                    if phase is not None:
                        memory_bits.append(
                            f"会话阶段={
                                getattr(
                                    phase,
                                    'value',
                                    phase)}"
                        )
                    last_topic = str(getattr(memoir, "last_topic", "") or "")
                    if last_topic:
                        memory_bits.append(f"上次话题={last_topic}")
                    last_mood = str(getattr(memoir, "last_mood", "") or "")
                    if last_mood:
                        memory_bits.append(f"上次气氛={last_mood}")
                    if (
                        latest_text
                        and last_topic
                        and self._normalize_repeat_text(latest_text)
                    ):
                        memory_bits.append(
                            "这句大概率是在延续之前的话头，不是全新开题"
                        )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

        if latest_text:
            normalized_latest = self._normalize_repeat_text(latest_text)
            repeat_count = self._count_recent_user_repeats(
                normalized_latest, target_uid
            )
            if repeat_count >= 2:
                memory_bits.append(
                    f"对方近几轮已重复这句或近似说法{repeat_count}次"
                )
                execution_bits.append(
                    "先按重复短句/可能玩梗/等待你接前情来理解，不要把它当首次出现的信息。"
                )
                execution_bits.append(
                    "如果信息不够，就直接判断成没说清楚、在复读、在等你接前情，不要硬猜对方具体在聊什么。"
                )
        if latest_repeat_short:
            memory_bits.append(
                "入口层已标记：这是重复短句/前情延续，不应按新话题理解"
            )
            if latest_continuation_hint:
                execution_bits.append(latest_continuation_hint)
            execution_bits.append(
                "这类短句允许你心里直接觉得看不懂、懒得猜、像在刷同一句，不需要替对方补完整话题。"
            )

        self_memory = self._build_self_reply_memory()
        if self_memory:
            memory_bits.append(
                self_memory.replace("[你刚刚自己说过的话]", "你刚刚说过")
            )

        cached_memory_hint = str(
            getattr(self, "_latest_memory_hint", "") or ""
        ).strip()
        if cached_memory_hint:
            memory_bits.append(cached_memory_hint[:220])

        persona_hint = self._build_persona_hint()
        if persona_hint:
            profile_bits.append(persona_hint.replace("\n", "；"))

        if self._llm_content_plan:
            execution_bits.append(f"内容规划={self._llm_content_plan[:120]}")
        if getattr(self, "_llm_decision_reason", ""):
            execution_bits.append(
                f"决策理由={str(self._llm_decision_reason)[:120]}"
            )
        if ambient is not None:
            arousal = float(ambient.get("arousal", 0.0) or 0.0)
            valence = float(ambient.get("valence", 0.0) or 0.0)
            execution_bits.append(
                f"环境唤醒={arousal:.2f}, 环境正负={valence:.2f}"
            )
        execution_bits.append(
            "独白要像人心里自然冒出的想法，可以有联想、回忆、吐槽，每次都要不同。"
        )

        # 实时关系数值（用于覆盖旧快照中的滞后值）
        _rt_annoyance = ""
        _rt_pressure = ""
        _rt_blocked = ""
        if relation_snapshot:
            _rt_annoyance = str(relation_snapshot.get("annoyance_value", ""))
            _rt_pressure = str(relation_snapshot.get("psychological_pressure", ""))
            _rt_blocked = str(relation_snapshot.get("is_user_blocked", ""))

        return {
            "profile_summary": "，".join(bit for bit in profile_bits if bit),
            "memory_summary": "，".join(bit for bit in memory_bits if bit),
            "execution_hint": "；".join(bit for bit in execution_bits if bit),
            "rt_annoyance": _rt_annoyance,
            "rt_pressure": _rt_pressure,
            "rt_blocked": _rt_blocked,
        }

    def _build_decision_context_packet(
        self,
        decision_messages: List,
        repetition_signal: Optional[Dict[str, Any]] = None,
    ):
        """把当前轮最关键的对象、续接与重复信号压成统一上下文。"""
        from src.common.data_models.heartflow_models import (
            DecisionContextPacket,
        )

        latest_user = None
        latest_bot = None
        for msg in reversed(decision_messages):
            user_id = str(getattr(msg, "user_id", "") or "").strip()
            if latest_bot is None and user_id == "bot":
                latest_bot = msg
                continue
            if latest_user is None and user_id and user_id != "bot":
                latest_user = msg
            if latest_user is not None and latest_bot is not None:
                break

        packet = DecisionContextPacket()
        if latest_user is not None:
            packet.target_user_id = str(
                getattr(latest_user, "user_id", "") or ""
            ).strip()
            packet.target_name = str(
                getattr(latest_user, "user_nickname", "")
                or getattr(latest_user, "nickname", "")
                or packet.target_user_id
            ).strip()
            packet.latest_user_text = str(
                getattr(latest_user, "processed_plain_text", "")
                or getattr(latest_user, "content", "")
                or ""
            ).strip()
            packet.continuation_hint = str(
                getattr(latest_user, "context_continuation_hint", "") or ""
            ).strip()
            packet.repeated_short_input = bool(
                getattr(latest_user, "is_repeated_short_input", False)
            )

        if latest_bot is not None:
            packet.latest_bot_text = str(
                getattr(latest_bot, "processed_plain_text", "")
                or getattr(latest_bot, "content", "")
                or ""
            ).strip()

        if repetition_signal and repetition_signal.get("detected"):
            packet.repeat_reason = str(
                repetition_signal.get("reason", "") or ""
            ).strip()
            if repetition_signal.get("latest_matches_repeat"):
                packet.repeated_short_input = True

        return packet

    async def _prefetch_memory_hint(self, messages: List) -> str:
        """在独白/规划前做一层轻量记忆预检索，把结果前置给主链使用。"""
        try:
            from src.memory_system.memory_retrieval import (
                build_memory_retrieval_prompt,
            )
            from src.chat.utils.chat_message_builder import (
                build_readable_messages,
            )

            target_message = self._get_latest_human_message(messages)
            if target_message is None:
                self._latest_memory_hint = ""
                return ""

            target_text = self._extract_message_content(target_message).strip()
            if not target_text:
                self._latest_memory_hint = ""
                return ""

            history_block = build_readable_messages(
                list(messages[-12:]),
                replace_bot_name=True,
                timestamp_mode="relative",
                read_mark=0.0,
                show_actions=True,
            )
            sender = (
                str(
                    getattr(target_message, "user_nickname", "")
                    or getattr(target_message, "nickname", "")
                    or getattr(target_message, "user_id", "用户")
                ).strip()
                or "用户"
            )
            # F14：关系叙事→影响记忆检索优先级——高好感/高信任用户的记忆优先召回
            _rel_priority_hint = ""
            _rel_snap = getattr(self, "_last_relation_snapshot", None) or {}
            if _rel_snap:
                _aff = float(_rel_snap.get("affection", 50.0) or 50.0)
                _trust = float(_rel_snap.get("trust_value", 50.0) or 50.0)
                if _aff > 65 and _trust > 60:
                    _rel_priority_hint = f"[关系优先] {sender}是亲密好友(好感{
                        _aff:.0f}/信任{
                        _trust:.0f})，优先检索与TA相关的记忆。"
                elif _aff < 20 and _trust < 25:
                    _rel_priority_hint = f"[关系回避] {sender}关系疏远(好感{
                        _aff:.0f}/信任{
                        _trust:.0f})，仅检索冲突或重要事件记忆。"
            hint = await build_memory_retrieval_prompt(
                history_block,
                sender,
                target_text,
                self.chat_stream,
                think_level=1,
                unknown_words=None,
                relation_context=_rel_priority_hint,
            )
            compact_hint = str(hint or "").strip()
            if compact_hint.startswith("你回忆起了以下信息："):
                compact_hint = compact_hint.replace(
                    "你回忆起了以下信息：", ""
                ).strip()
            self._latest_memory_hint = compact_hint[:320]
            return self._latest_memory_hint
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 预记忆检索失败: {exc}")
            self._latest_memory_hint = ""
            return ""

    def _build_voice_execution_guard(
        self, voice_conclusion: Any, target_message: Optional[Any]
    ) -> str:
        """把内心独白变成后续回复生成阶段必须遵守的执行约束。"""
        if voice_conclusion is None:
            return ""
        thinking = str(getattr(voice_conclusion, "thinking", "") or "").strip()
        mood = str(getattr(voice_conclusion, "current_mood", "") or "").strip()
        desire = int(getattr(voice_conclusion, "reply_desire_level", 0) or 0)
        if not thinking and not mood and desire <= 0:
            return ""

        target_name = "对方"
        if target_message is not None:
            target_name = (
                str(
                    getattr(target_message, "user_nickname", "")
                    or getattr(target_message, "user_id", "")
                    or "对方"
                ).strip()
                or "对方"
            )

        lines = ["[内心独白执行约束]"]
        if thinking:
            lines.append(f"- 你刚才心里真正想到的是：{thinking}")
        if mood:
            lines.append(f"- 当前情绪底色：{mood}")
        if desire > 0:
            lines.append(f"- 当前回复欲望：{desire}/10")
        lines.append(
            f"- 现在回复{target_name}时，语气和内容必须顺着这句心里话往下走，不能另起一套分析。"
        )
        lines.append(
            "- 不要把内心独白原封不动复述给对方，而是把它转成自然回复动机。"
        )
        lines.append(
            "- 如果内心独白偏冷淡、试探、烦躁或敷衍，回复也必须保留这种底色。"
        )
        lines.append(
            "- 如果内心独白是在判断对方是不是玩梗、刷屏、试探你，回复要先贴着这个判断接，不要跑题。"
        )
        return "\n".join(lines)

    def _resolve_relation_view(
        self, relation_snapshot: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """统一解析关系视图，优先使用显式传入的数据，回退到最近一次缓存。

        关键修复：在返回前从emotion_tracker注入实时annoyance和pressure值，
        解决快照数据(annoyance=0)与实时状态(annoyance=100)不一致的问题。
        """
        if relation_snapshot:
            return self._inject_realtime_emotion(relation_snapshot)
        tick_snapshot = getattr(self, "_tick_world_snapshot", None)
        if tick_snapshot is not None:
            try:
                return self._inject_realtime_emotion(
                    tick_snapshot.to_relation_dict()
                )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        return self._inject_realtime_emotion(
            getattr(self, "_last_relation_snapshot", {}) or {}
        )

    def _inject_realtime_emotion(
        self, base_view: Dict[str, Any]
    ) -> Dict[str, Any]:
        """从emotion_tracker读取实时annoyance/pressure覆盖到关系视图。

        查找优先级：
        1. 本轮缓存状态（_resolved_emo_state，由情绪追踪器同步阶段写入）
        2. 按_last_user_id实时查询（兼容无缓存场景）
        """
        _state_to_use = None
        if self._resolved_emo_state is not None:
            _state_to_use = self._resolved_emo_state
        else:
            target_uid = str(getattr(self, "_last_user_id", "") or "").strip()
            if target_uid:
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker, )

                    tracker = get_emotion_tracker(self.stream_id)
                    _state_to_use = tracker.get_user_state(
                        target_uid, create_if_missing=False
                    )
                except Exception:
                    pass
        if _state_to_use is None:
            return dict(base_view)
        try:
            rt_annoyance = float(getattr(_state_to_use, "annoyance", -1) or -1)
            if rt_annoyance >= 0:
                base_view["annoyance_value"] = round(rt_annoyance, 1)
            rt_pressure = float(
                getattr(_state_to_use, "psychological_pressure", -1) or -1
            )
            if rt_pressure >= 0:
                base_view["psychological_pressure"] = round(rt_pressure, 1)
            rt_blocked = bool(
                getattr(_state_to_use, "is_blocked", False) or False
            )
            base_view["is_user_blocked"] = rt_blocked
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return dict(base_view)

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
            packet_context = str(
                decision_context_packet.compact_context() or ""
            ).strip()
            if packet_context:
                lines.append(packet_context)

        relation_snapshot = self._resolve_relation_view(relation_snapshot)
        custom_label = str(
            relation_snapshot.get("custom_label", "") or ""
        ).strip()
        social_value = float(relation_snapshot.get("social_value", 0.0) or 0.0)
        trust_value = float(relation_snapshot.get("trust_value", 0.0) or 0.0)
        annoyance_value = float(
            relation_snapshot.get("annoyance_value", 0.0) or 0.0
        )
        relation_bits: List[str] = []
        if custom_label:
            relation_bits.append(f"关系={custom_label}")
        relation_bits.append(f"社交={social_value:.1f}")
        if trust_value:
            relation_bits.append(f"信任={trust_value:.1f}")
        if annoyance_value:
            relation_bits.append(f"烦躁={annoyance_value:.1f}")
        lines.append(f"- 当前关系底色: {' '.join(relation_bits)}")

        cached_memory_hint = str(
            getattr(self, "_latest_memory_hint", "") or ""
        ).strip()
        if cached_memory_hint:
            lines.append(f"- 已检索到前情/记忆: {cached_memory_hint[:220]}")

        continuity_context = self._build_self_continuity_context()
        if continuity_context:
            lines.append(
                f"- 你自己刚才的延续线索: {continuity_context.replace(chr(10), '；')[:220]}"
            )

        self_memory = self._build_self_reply_memory()
        if self_memory:
            lines.append(
                f"- 你自己最近说过的话: {self_memory.replace(chr(10), '；')[:220]}"
            )

        if repetition_signal and repetition_signal.get("detected"):
            repeat_reason = str(
                repetition_signal.get("reason", "重复输入") or "重复输入"
            )
            exact_repeat_count = int(
                repetition_signal.get("exact_repeat_count", 0) or 0
            )
            extra = (
                f"，近轮重复约{exact_repeat_count}次"
                if exact_repeat_count > 0
                else ""
            )
            lines.append(f"- 重复/施压信号: {repeat_reason[:120]}{extra}")
            if repetition_signal.get("latest_matches_repeat"):
                lines.append(
                    "- 处理原则: 先按前情延续、复读、玩梗或等你接话来理解，不要把短句硬当新话题。"
                )

        if voice_conclusion is not None:
            voice_guard = self._build_voice_execution_guard(
                voice_conclusion, target_message
            )
            if voice_guard:
                parts.append(voice_guard)

        lines.append(
            "- 执行要求: 规划和回复都必须顺着前情、关系底色和刚才那句心里话继续，不要突然切成说明书口吻。"
        )
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
                target_uid = str(
                    getattr(target_message, "user_id", "") or ""
                ).strip()
            _final_annoy = float(snapshot.get("annoyance_value", 0.0) or 0.0)
            _final_pressure = float(
                snapshot.get("psychological_pressure", 0.0) or 0.0
            )
            mode_summary = "unknown/unknown"
            if target_uid:
                try:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker, )

                    _tracker = get_emotion_tracker(self.stream_id)
                    _cached_es = getattr(self, "_resolved_emo_state", None)
                    if _cached_es is not None:
                        _t_state = _cached_es
                    else:
                        _t_state = _tracker.get_user_state(
                            target_uid, create_if_missing=False
                        )
                    if _t_state:
                        _direct_annoy = float(
                            getattr(_t_state, "annoyance", -1) or -1
                        )
                        if _direct_annoy >= 0:
                            _final_annoy = round(_direct_annoy, 1)
                        _direct_press = float(
                            getattr(_t_state, "psychological_pressure", -1)
                            or -1
                        )
                        if _direct_press >= 0:
                            _final_pressure = round(_direct_press, 1)
                    response_mode = _tracker.get_layered_response_mode(
                        target_uid
                    )
                    mode_summary = (
                        f"{str(response_mode.get('tone', 'neutral') or 'neutral')}"
                        f"/{str(response_mode.get('response_length', 'normal') or 'normal')}"
                    )
                except Exception:
                    pass

            logger.info(
                f"{self.log_prefix} 回复生成摘要 "
                f"来源={source} "
                f"对象={target_uid[:8] if target_uid else 'none'} "
                f"形式={str(style_route.get('reply_style', 'direct') or 'direct')} "
                f"引用={'是' if bool(style_route.get('quote_message', False)) else '否'} "
                f"模式={mode_summary} "
                f"烦躁={_final_annoy:.1f} "
                f"压力={_final_pressure:.1f} "
                f"前情块={'有' if context_execution_block else '无'} "
                f"提示长度={len(extra_info)}"
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _build_voice_execution_summary(
        self, voice_conclusion: Any, target_message: Optional[Any]
    ) -> Dict[str, str]:
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
        desire = int(getattr(voice_conclusion, "reply_desire_level", 0) or 0)
        target_name = "对方"
        if target_message is not None:
            target_name = (
                str(
                    getattr(target_message, "user_nickname", "")
                    or getattr(target_message, "user_id", "")
                    or "对方"
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

    # ═══════════════════════════════════════════════════
    #  阶段 4.5b：多维信号融合判断（heartflow_decision）
    # ═══════════════════════════════════════════════════

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
            for msg in reversed(decision_messages):
                uid = getattr(msg, "user_id", "") or ""
                if uid and uid != "bot":
                    latest_text = str(
                        getattr(msg, "processed_plain_text", "")
                        or getattr(msg, "plain_text", "")
                        or ""
                    ).strip()[:500]
                    sender_id = uid
                    break
            mood_tag = "neutral"
            if voice_conclusion is not None:
                raw_mood = str(
                    getattr(voice_conclusion, "current_mood", "") or ""
                ).strip()
                if raw_mood:
                    mood_tag = raw_mood
            _desire_raw = 5
            if voice_conclusion is not None and hasattr(
                voice_conclusion, "reply_desire_level"
            ):
                _desire_raw = int(
                    getattr(voice_conclusion, "reply_desire_level", 5) or 5
                )
            social_rel = {
                "is_targeted": bool(
                    getattr(self, "_cached_is_pinged", False)
                ),
                "interest_level": max(0.0, min(1.0, _desire_raw / 10.0)),
            }
            situation = JudgmentSituation(
                stream_id=self.stream_id,
                sender_id=sender_id,
                raw_content=latest_text,
                bot_mood_tag=mood_tag,
                social_hints=social_rel,
                sender_profile={
                    "rel_score": float(
                        relation_result.get("social_value", 0.0) or 0.0
                    ),
                    "activity": float(
                        relation_result.get("activity_level", 50.0) or 50.0
                    ),
                },
            )
            outcome = maker.multidim_decide(
                JudgmentDimension.RESPOND, situation
            )
            comp = maker.weighted_composite(outcome.signal_snapshot)
            return {
                "should_respond": bool(outcome.verdict),
                "composite": round(comp, 4),
                "certainty": outcome.certainty.name,
                "rationale": str(outcome.rationale or "")[:120],
            }
        except Exception as exc:
            logger.debug(
                f"{self.log_prefix} 心流决策查询异常: {exc}"
            )
            return None

    # ═══════════════════════════════════════════════════
    #  阶段 4.6：早期退出检查
    # ═══════════════════════════════════════════════════

    def _check_early_exit(
        self,
        voice_conclusion,
        pinged_msg,
        relation_result: Dict,
    ) -> Dict[str, Any]:
        """
        早期退出检查 - 在调用 LLM 之前判断是否应该跳过

        核心思路：如果内心独白已经明确决定不想回复，就不要浪费 token 调用 LLM。

        返回值：
        - should_skip: 是否应该跳过后续流程
        - reason: 跳过原因
        """
        result = {"should_skip": False, "reason": ""}

        # 被@时记录模型意愿但不强制跳过，交给 readiness_evaluator 做最终决定
        if pinged_msg is not None:
            _model_reply = (
                getattr(voice_conclusion, "should_reply", None)
                if voice_conclusion
                else None
            )
            if _model_reply is False:
                result["reason"] = "被@但模型决策为不回复，交由准入评估裁决"
            return result

        # 检查内心独白结果
        if voice_conclusion is not None and hasattr(
            voice_conclusion, "reply_desire_level"
        ):
            desire_level = voice_conclusion.reply_desire_level

            # 欲望等级 <= 2：明确不想回复，直接跳过
            if desire_level <= 2:
                result["should_skip"] = True
                result["reason"] = (
                    f"内心独白欲望等级过低({desire_level})，明确不想回复"
                )
                return result

            # 欲望等级 <= 4 且有负面情绪：跳过
            if desire_level <= 4:
                mood = str(
                    getattr(voice_conclusion, "current_mood", "") or ""
                ).strip().lower()
                _negative_moods = {
                    "烦躁", "疲惫", "无聊", "厌倦", "愤怒",
                    "焦虑", "低落", "冷漠", "抗拒", "不耐烦",
                    "irritated", "bored", "tired", "angry", "anxious",
                }
                if any(kw in mood for kw in _negative_moods):
                    result["should_skip"] = True
                    result["reason"] = (
                        f"内心独白欲望等级低({desire_level})且情绪负面({mood})"
                    )
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
                logger.info(
                    f"{self.log_prefix} 😤 厌烦值{annoyance:.1f}但决定回复，可能表达不满"
                )

        # 检查能量：能量过低时跳过（优先从世界快照读取）
        snap = getattr(self, "_tick_world_snapshot", None)
        if snap is not None:
            chat_val = snap.self_resources.chat_energy
            think_val = snap.self_resources.thinking_energy
            act_val = snap.self_resources.activity_level
            if chat_val < 10 or think_val < 10 or act_val < 8:
                result["should_skip"] = True
                result["reason"] = (
                    f"快照状态过低(聊天值={
                        chat_val:.1f}, 思考值={
                        think_val:.1f}, 活跃度={
                        act_val:.1f})"
                )
                return result
        else:
            try:
                from src.chat.heart_flow.energy_manager import (
                    get_vitality_pool,
                    get_shared_resource_manager,
                )

                pool_snap = get_vitality_pool().capture_snapshot(
                    self.stream_id
                )
                shared_values = (
                    get_shared_resource_manager().get_shared_values(
                        self.stream_id
                    )
                )
                chat_value = float(shared_values.get("chat_value") or 0.0)
                activity_level = float(
                    shared_values.get("activity_level") or 0.0
                )
                if (
                    chat_value < 10
                    or pool_snap.thinking_value < 10
                    or activity_level < 8
                ):
                    result["should_skip"] = True
                    result["reason"] = (
                        f"状态过低(聊天值={
                            chat_value:.1f}, 思考值={
                            pool_snap.thinking_value:.1f}, 活跃度={
                            activity_level:.1f})"
                    )
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
            return incoming_batch
        decision_messages = []
        for msg in filtered_messages:
            user_id = getattr(msg, "user_id", "") or ""
            if user_id == "bot":
                decision_messages.append(msg)
                continue
            if getattr(msg, "_spam_user", False):
                continue
            decision_messages.append(msg)

        latest_text = ""
        for msg in reversed(decision_messages):
            if getattr(msg, "user_id", "") != "bot":
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
            recent_context = context_manager.get_recent_context(
                self.stream_id, limit=8
            )
            relevant_context = (
                context_manager.search_context(
                    self.stream_id, latest_text, limit=4
                )
                if latest_text
                else []
            )

            seen_context_ids = {
                str(
                    getattr(msg, "message_id", "")
                    or getattr(msg, "id", "")
                    or ""
                )
                for msg in decision_messages
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
                    processed_plain_text=str(
                        getattr(item, "content", "") or ""
                    ),
                    plain_text=str(getattr(item, "content", "") or ""),
                    content=str(getattr(item, "content", "") or ""),
                    timestamp=float(getattr(item, "timestamp", 0.0) or 0.0),
                )
                decision_messages.append(pseudo)
                if context_id:
                    seen_context_ids.add(context_id)
        except Exception as _e:
            logger.debug(f"异常: {_e}")

        seen_texts = {
            str(
                getattr(msg, "processed_plain_text", "")
                or getattr(msg, "plain_text", "")
                or ""
            ).strip()
            for msg in decision_messages
        }
        for synthetic_msg in self._build_synthetic_self_messages():
            synthetic_text = str(
                getattr(synthetic_msg, "processed_plain_text", "")
                or getattr(synthetic_msg, "plain_text", "")
                or ""
            ).strip()
            if not synthetic_text or synthetic_text in seen_texts:
                continue
            decision_messages.append(synthetic_msg)

        decision_messages.sort(
            key=lambda msg: float(getattr(msg, "timestamp", 0.0) or 0.0)
        )
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

    def _collect_topic_signals(
        self, messages: List, limit: int = 12
    ) -> Dict[str, Any]:
        """收集最近消息里的重复话题信号。"""
        human_entries = []
        exact_counter: Counter = Counter()
        exact_users: Dict[str, set] = defaultdict(set)
        token_counter: Counter = Counter()
        token_users: Dict[str, set] = defaultdict(set)
        low_info_count = 0

        for msg in messages[-limit:]:
            user_id = getattr(msg, "user_id", "") or ""
            if user_id == "bot":
                continue

            text = (
                getattr(msg, "processed_plain_text", "")
                or getattr(msg, "content", "")
                or ""
            ).strip()
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
                result["repeat_user_count"] = max(
                    result["repeat_user_count"], user_count
                )
                break

        dominant_tokens = []
        for token, count in stats["token_counter"].most_common(6):
            user_count = len(stats["token_users"].get(token, set()))
            if count >= 2 and user_count >= 1:
                dominant_tokens.append(token)
                result["repeat_user_count"] = max(
                    result["repeat_user_count"], user_count
                )
        result["dominant_tokens"] = dominant_tokens[:3]

        low_info_ratio = stats["low_info_count"] / max(1, len(human_entries))
        result["low_info_cluster"] = low_info_ratio >= 0.5 and bool(
            exact_key or dominant_tokens
        )

        latest_entry = human_entries[-1]
        latest_tokens = set(latest_entry.get("tokens", []))
        latest_user = str(latest_entry.get("user_id", "") or "")
        result["latest_matches_repeat"] = bool(
            (exact_key and latest_entry.get("normalized") == exact_key)
            or (
                dominant_tokens and latest_tokens.intersection(dominant_tokens)
            )
        )

        _single_user_spam = False
        if len(human_entries) >= 2:
            _recent_users = [
                str(e.get("user_id", "") or "") for e in human_entries[-3:]
            ]
            if (
                len(set(_recent_users)) == 1
                and _recent_users[0] == latest_user
            ):
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
            result["reason"] = (
                f"同一人连续复读({result['exact_repeat_count']}次)"
            )
        elif (
            result["exact_repeat_count"] >= exact_threshold + 1
            and result["repeat_user_count"] >= 2
        ):
            result["reason"] = (
                f"多人连续复读相近内容({result['exact_repeat_count']}次)"
            )
        elif dominant_tokens and result["repeat_user_count"] >= 3:
            result["reason"] = (
                f"多人围绕 {dominant_tokens[0]} 反复刷低信息消息"
            )
        elif dominant_tokens:
            result["reason"] = (
                f"当前话题 {dominant_tokens[0]} 短时间内重复度过高"
            )
        else:
            result["reason"] = "当前内容重复度过高，先不接话"
        return result

    def _analyze_harassment_pressure(self, messages: List) -> Dict[str, Any]:
        """补充旧版的骚扰强度层，用于区分普通复读和带冒犯/骚扰意味的持续输入。"""
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
            latest_user = None
            for msg in reversed(messages):
                if getattr(msg, "user_id", "") != "bot":
                    latest_user = msg
                    break
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

            for msg in messages[-12:-1]:
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
                    and (
                        normalized_history in normalized_text
                        or normalized_text in normalized_history
                    )
                ):
                    similar_count += 1

            detected = bool(
                same_hash_count >= 5
                or similar_count >= 4
                or total_intensity >= 20.0
            )
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
            severity = (
                "high"
                if intensity >= 35 or same_hash_count >= 6
                else "medium" if intensity >= 20 else "low"
            )
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
        text = (
            self._extract_message_content(target_message).strip()
            if target_message is not None
            else ""
        )

        quote_message = False
        reply_style = "direct"
        reasons: List[str] = []

        # 不引用bot自己的消息，避免引用链无限堆叠
        _target_uid = str(getattr(target_message, "user_id", "") or "") if target_message else ""
        _is_bot_msg = (_target_uid == "bot" or _target_uid == str(getattr(self, "bot_user_id", "") or ""))
        if _is_bot_msg:
            reasons.append("不引用自己的消息")
        elif target_message is not None and bool(
            getattr(target_message, "is_quote_reply", False)
        ):
            quote_message = True
            reply_style = "quote"
            reasons.append("对方本身就是引用链")
        elif any(
            marker in text
            for marker in ("?", "？", "怎么", "为什么", "啥", "什么")
        ):
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

    def _extract_current_topics(
        self, messages: List, limit: int = 5
    ) -> List[str]:
        """提取当前话题，供 LLM 自主规划器使用。"""
        topics: List[str] = []
        seen = set()

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
            if getattr(msg, "user_id", "") != "bot":
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
        punctuation_hits = sum(
            text.count(marker) for marker in ("?", "？", "!", "！")
        )
        if punctuation_hits > 0:
            score += min(2, punctuation_hits)
        normalized = self._normalize_topic_text(text)
        if len(normalized) >= 18:
            score += 1
        if any(getattr(msg, "user_id", "") != "bot" for msg in messages[-3:]):
            score += 1
        if getattr(latest_user_message, "_spam_user", False):
            score = max(0, score - 2)
        if repetition_signal and repetition_signal.get(
            "latest_matches_repeat"
        ):
            if repetition_signal.get("low_info_cluster"):
                score = max(score, 2)
            elif repetition_signal.get("detected"):
                score = max(score, 1)
        if behavior_signal:
            category = str(
                behavior_signal.get("category", "neutral") or "neutral"
            )
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
        content_state_signal: Optional[Dict[str, Any]] = None,
        preprocessor_signal: Optional[Dict[str, Any]] = None,
        repetition_signal: Optional[Dict[str, Any]] = None,
        harassment_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """把身份、群态势和内容状态真正接入拒绝回复链。"""
        result = {"should_skip": False, "reason": "", "source": ""}
        if pinged_msg is not None:
            return result

        human_messages = [
            msg
            for msg in decision_messages
            if getattr(msg, "user_id", "") != "bot"
        ]
        if not human_messages:
            result["should_skip"] = True
            result["reason"] = "过滤后没有有效目标，保持观察"
            return result

        if identity_context.get("has_conflict"):
            result["should_skip"] = True
            result["reason"] = "身份状态冲突，暂不介入"
            return result

        if (
            self_reply_risk.get("is_self_reply")
            and float(self_reply_risk.get("similarity", 0.0) or 0.0) >= 0.88
        ):
            result["should_skip"] = True
            result["reason"] = (
                f"自回复风险过高({
                    self_reply_risk.get(
                        'similarity', 0.0):.2f})"
            )
            return result

        desire_level = 5
        if voice_conclusion is not None and hasattr(
            voice_conclusion, "reply_desire_level"
        ):
            desire_level = int(
                getattr(voice_conclusion, "reply_desire_level", 5) or 5
            )

        if (
            content_state_signal
            and content_state_signal.get("should_skip")
            and desire_level < 8
        ):
            result["should_skip"] = True
            result["reason"] = content_state_signal.get(
                "reason", "当前内容已被处理过，先不重复接话"
            )
            result["source"] = "content_state"
            return result

        if (
            preprocessor_signal
            and preprocessor_signal.get("is_spam")
            and desire_level < 8
        ):
            spam_type = preprocessor_signal.get("spam_type", "刷屏")
            result["should_skip"] = True
            result["reason"] = f"检测到{spam_type}，当前不跟着噪声接话"
            result["source"] = "message_preprocessor"
            return result

        if (
            harassment_signal
            and harassment_signal.get("detected")
            and desire_level < 9
        ):
            severity = harassment_signal.get("severity", "low")
            result["should_skip"] = True
            result["reason"] = harassment_signal.get(
                "reason", f"检测到{severity}级骚扰输入，当前不接话"
            )
            result["source"] = "harassment_detector"
            return result

        if relation_result.get("behavior_signal"):
            behavior_signal = relation_result.get("behavior_signal") or {}
            category = str(
                behavior_signal.get("category", "neutral") or "neutral"
            )
            severity = float(behavior_signal.get("severity", 0.0) or 0.0)
            if category == "harassing" and desire_level < 9:
                result["should_skip"] = True
                result["reason"] = (
                    f"分类为持续骚扰输入(severity={severity:.2f})，本轮不回"
                )
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
                f"{self.log_prefix} 检测到重复输入，允许规划器给出澄清反馈: "
                f"{repetition_signal.get('reason', 'repeat')}"
            )

        if group_sense_result.get("burst_detected") and desire_level < 7:
            result["should_skip"] = True
            result["reason"] = "群里正在高密度交流，当前不插话"
            return result

        if (
            group_sense_result.get("controversy_detected")
            and float(relation_result.get("trust_value", 0.0) or 0.0) < 35.0
            and desire_level < 8
        ):
            result["should_skip"] = True
            result["reason"] = "群聊存在争议且信任不足，选择克制"
            return result

        annoyance = float(relation_result.get("annoyance_value", 0.0) or 0.0)
        if annoyance >= 35.0 and desire_level < 7:
            # 厌烦值高时概率性跳过，但保留回复机会表达不满
            skip_prob = min(0.90, (annoyance - 35.0) / 100.0)
            if random.random() < skip_prob:
                result["should_skip"] = True
                result["reason"] = f"厌烦值偏高({annoyance:.1f})，本轮不回"
                return result
            else:
                # 小概率回复时，标记情绪提示
                result["emotion_hint"] = "annoyed_reply"
                logger.info(
                    f"{self.log_prefix} 😤 厌烦值{annoyance:.1f}但决定回复，可能表达不满"
                )

        return result

    def _delegate_readiness_evaluation(
        self,
        decision_messages: List,
        pinged_msg,
        relation_result: Dict[str, Any],
        voice_conclusion,
        repetition_signal: Optional[Dict[str, Any]] = None,
        harassment_signal: Optional[Dict[str, Any]] = None,
        message_salience: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """硬约束逻辑委托给 core.readiness_evaluator 集中处理"""
        from src.core.readiness_evaluator import (
            ReadinessInput,
            get_readiness_evaluator,
        )

        evaluator = get_readiness_evaluator()
        relation_view = self._resolve_relation_view(relation_result)
        # 组装输入
        desire = 5
        if voice_conclusion is not None and hasattr(
            voice_conclusion, "reply_desire_level"
        ):
            desire = int(
                getattr(voice_conclusion, "reply_desire_level", 5) or 5
            )
        has_mention = any(
            getattr(msg, "user_id", "") != "bot"
            and self._is_message_pinged(msg)
            for msg in decision_messages
        )
        latest_txt = ""
        for msg in reversed(decision_messages):
            if getattr(msg, "user_id", "") != "bot":
                latest_txt = str(
                    getattr(msg, "processed_plain_text", "")
                    or getattr(msg, "plain_text", "")
                    or getattr(msg, "content", "")
                    or ""
                ).strip()
                break
        rep_pressure = 0.0
        rep_rounds = 0
        if repetition_signal:
            rep_pressure = float(
                repetition_signal.get(
                    "repeated_topic_pressure",
                    repetition_signal.get("pressure", 0.0),
                )
                or 0.0
            )
            rep_rounds = max(
                int(
                    repetition_signal.get(
                        "repeat_count",
                        repetition_signal.get("recent_repeat_count", 0),
                    )
                    or 0
                ),
                int(repetition_signal.get("exact_repeat_count", 0) or 0),
            )
        sal = 0.0
        if message_salience:
            sal = float(message_salience.get("score", 0.0) or 0.0)
        harass_det = False
        harass_sev = "low"
        if harassment_signal and harassment_signal.get("detected"):
            harass_det = True
            harass_sev = str(harassment_signal.get("severity", "low") or "low")
        # 获取频道级烦躁度（VitalityPool）及通道资源（world_snapshot）
        _ch_annoy = 0.0
        try:
            from src.chat.heart_flow.energy_manager import get_vitality_pool
            _ch_snap = get_vitality_pool().capture_snapshot(self.stream_id)
            _ch_annoy = float(getattr(_ch_snap, "annoyance_level", 0.0) or 0.0)
        except Exception:
            pass
        # 通道资源字段从 world_snapshot.self_resources 获取
        _ch_boredom = 0.0
        _ch_loneliness = 0.0
        _ch_social_desire = 0.0
        _ch_consec_replies = 0
        _ch_silence_sec = 0.0
        _ws = getattr(self, "_tick_world_snapshot", None)
        if _ws is not None:
            _res = getattr(_ws, "self_resources", None)
            if _res is not None:
                _ch_boredom = float(getattr(_res, "boredom", 0.0) or 0.0)
                _ch_loneliness = float(getattr(_res, "loneliness", 0.0) or 0.0)
                _ch_social_desire = float(getattr(_res, "social_desire", 0.0) or 0.0)
                _ch_consec_replies = int(getattr(_res, "consecutive_replies", 0) or 0)
                _ch_silence_sec = float(getattr(_res, "silence_seconds", 0.0) or 0.0)
        inp = ReadinessInput(
            is_direct_ping=(pinged_msg is not None),
            has_indirect_mention=has_mention,
            desire_level=desire,
            model_should_reply=(
                getattr(voice_conclusion, "should_reply", None)
                if voice_conclusion
                else None
            ),
            chat_value=float(
                relation_result.get("chat_value", 100.0) or 100.0
            ),
            activity_level=float(
                relation_result.get("activity_level", 50.0) or 50.0
            ),
            social_value=float(
                relation_result.get("social_value", 0.0) or 0.0
            ),
            trust_value=float(relation_view.get("trust_value", 0.0) or 0.0),
            annoyance_value=float(
                relation_view.get("annoyance_value", 0.0) or 0.0
            ),
            psychological_pressure=float(
                relation_view.get("psychological_pressure", 0.0) or 0.0
            ),
            repeated_pressure=rep_pressure,
            repeated_rounds=rep_rounds,
            repeat_detected=bool(rep_rounds >= 1 or rep_pressure > 0.0),
            repeat_ratio=(
                float(min(1.0, rep_pressure / 50.0))
                if rep_pressure > 0
                else 0.0
            ),
            salience_score=sal,
            latest_text=latest_txt,
            fondness=float(
                relation_view.get("affection", 0.0) or 0.0
            ),
            harassment_detected=harass_det,
            harassment_severity=harass_sev,
            interaction_count=int(
                relation_view.get("interaction_count", 0) or 0
            ),
            relationship_level=int(
                relation_view.get("relationship_level", 5) or 5
            ),
            trend_direction=str(
                relation_view.get("trend_direction", "") or ""
            ),
            positive_dim=float(
                relation_view.get("positive_dim", 0.0) or 0.0
            ),
            negative_dim=float(
                relation_view.get("negative_dim", 0.0) or 0.0
            ),
            channel_annoyance=_ch_annoy,
            trauma_score=float(
                relation_view.get("trauma_score", 0.0) or 0.0
            ),
            mental_fatigue=float(
                getattr(self._resolved_emo_state, "mental_fatigue", 0.0) or 0.0
            ) if self._resolved_emo_state else 0.0,
            intimacy=float(
                getattr(self._resolved_emo_state, "intimacy", 0.0) or 0.0
            ) if self._resolved_emo_state else 0.0,
            stamina=float(
                getattr(self._resolved_emo_state, "stamina", 100.0) or 100.0
            ) if self._resolved_emo_state else 100.0,
            trust_accumulation=float(
                getattr(self._resolved_emo_state, "trust_accumulation", 0.0) or 0.0
            ) if self._resolved_emo_state else 0.0,
            distrust_accumulation=float(
                getattr(self._resolved_emo_state, "distrust_accumulation", 0.0) or 0.0
            ) if self._resolved_emo_state else 0.0,
            negative_behavior_streak=int(
                getattr(self._resolved_emo_state, "negative_behavior_streak", 0) or 0
            ) if self._resolved_emo_state else 0,
            boredom=_ch_boredom,
            loneliness=_ch_loneliness,
            social_desire=_ch_social_desire,
            consecutive_replies=_ch_consec_replies,
            silence_seconds=_ch_silence_sec,
        )
        verdict = evaluator.evaluate(inp)
        evaluator.log_evaluation_panel(self.log_prefix, verdict, inp)
        # 同步缓存供面板/日志消费
        self._last_legacy_breakdown = verdict.breakdown.to_dict()
        self._last_legacy_penalty = verdict.total_deduction
        self._last_legacy_reason = verdict.reason[:120]
        self._last_legacy_gate = verdict.gate
        return verdict.to_dict()

    def _apply_readiness_constraint(
        self,
        constraint_result: Dict[str, Any],
        has_user_message: bool,
    ) -> None:
        """准入度约束连续命中后，通过 evaluator 计算冷却并进入等待态"""
        from src.core.readiness_evaluator import (
            ReadinessVerdict,
            ReadinessBreakdown,
            get_readiness_evaluator,
        )

        evaluator = get_readiness_evaluator()
        bd_raw = constraint_result.get("breakdown", {})
        bd = ReadinessBreakdown()
        if isinstance(bd_raw, dict):
            bd.reply_readiness = float(bd_raw.get("reply_readiness", 0.0))
        verdict = ReadinessVerdict(
            should_skip=constraint_result.get("should_skip", False),
            reason=str(constraint_result.get("reason", "") or ""),
            total_deduction=float(
                constraint_result.get("reply_penalty", 0.0) or 0.0
            ),
            reply_readiness=float(
                constraint_result.get("reply_readiness", 0.0) or 0.0
            ),
            gate=str(constraint_result.get("gate", "allow") or "allow"),
            breakdown=bd,
        )
        action = evaluator.compute_constraint_cooldown(
            verdict, has_user_message, _POST_MESSAGE_RETRY_SEC
        )
        self._legacy_constraint_hits = action.hit_count
        self._last_legacy_penalty = verdict.total_deduction
        self._last_legacy_reason = verdict.reason[:120]
        if action.should_enter_pending:
            self._planner_quiet_until = max(
                self._planner_quiet_until, time.time() + action.cooldown_sec
            )
            self._shift_to_pending(
                cause=action.pending_cause,
                anticipated=(
                    verdict.reason[:80] if verdict.reason else "准入度约束命中"
                ),
            )
        elif has_user_message:
            self._planner_quiet_until = max(
                self._planner_quiet_until, time.time() + action.cooldown_sec
            )

    # ═══════════════════════════════════════════════════
    #  阶段 4.8：主动回复执行 - 跳过规划器直接生成回复
    # ═══════════════════════════════════════════════════

    async def _execute_voice_driven_reply(
        self,
        voice_conclusion: Any,
        incoming_batch: List,
    ) -> bool:
        """
        执行内心驱动回复 - 内心独白决定回复时直接生成回复，跳过规划器

        这是真正的主动行为：内心独白已经决定了要回复，直接生成回复。
        """
        try:
            from src.plugin_system.apis import generator_api
            from src.config.config import global_config
            from src.chat.utils.utils import record_replyer_action_temp

            desire_level = getattr(voice_conclusion, "reply_desire_level", 5)
            thinking = getattr(voice_conclusion, "thinking", "")
            mood = getattr(voice_conclusion, "current_mood", "")

            logger.info(
                f"{self.log_prefix} 💭 开始执行: "
                f"欲望等级={desire_level}, "
                f"思考={thinking[:50] if thinking else '无'}"
            )

            target_message = None
            for msg in reversed(incoming_batch[-10:]):
                if getattr(msg, "user_id", "") != "bot":
                    target_message = msg
                    break
            target_message = self._select_preferred_reply_message(
                target_message, list(incoming_batch[-10:])
            )
            self._mark_message_content_processing(target_message)

            # 获取目标用户的风格指导
            user_style_guide = ""
            if target_message:
                target_user_id = getattr(target_message, "user_id", "")
                if target_user_id:
                    user_style_guide = self._get_user_style_guide(
                        target_user_id
                    )

            voice_summary = self._build_voice_execution_summary(
                voice_conclusion, target_message
            )

            reply_reason = f"内心驱动回复: 欲望等级={desire_level}"
            if voice_summary.get("reason"):
                reply_reason += f" | {voice_summary['reason'][:140]}"

            extra_info_parts: List[str] = []
            panel_text = voice_summary.get("panel", "")
            if panel_text:
                extra_info_parts.append(panel_text)
            decision_context_packet = self._build_decision_context_packet(
                list(incoming_batch),
                repetition_signal=self._analyze_repetition_pressure(
                    incoming_batch
                ),
            )
            relation_view = self._resolve_relation_view()
            context_execution_block = self._build_context_execution_block(
                target_message=target_message,
                voice_conclusion=voice_conclusion,
                repetition_signal=self._analyze_repetition_pressure(
                    incoming_batch
                ),
                decision_context_packet=decision_context_packet,
                relation_snapshot=relation_view,
            )
            if context_execution_block:
                extra_info_parts.append(context_execution_block)
            self_memory = self._build_self_reply_memory()
            continuity_context = self._build_self_continuity_context()
            if self_memory:
                extra_info_parts.append(self_memory)
            if continuity_context:
                extra_info_parts.append(continuity_context)
            if user_style_guide:
                extra_info_parts.append(user_style_guide)
            reply_style_context = self._build_reply_style_context(
                relation_view
            )
            if reply_style_context:
                extra_info_parts.append(reply_style_context)

            takeover_thought = getattr(self, "_takeover_decision", None)
            if takeover_thought:
                _tk_thought = takeover_thought.get("thought", "") or ""
                _tk_action = takeover_thought.get("action", "") or ""
                if _tk_thought:
                    extra_info_parts.append(
                        f"[接管意图] {_tk_action}: {_tk_thought}"
                    )
                self._takeover_decision = None
                self._takeover_action = None

            length_hint = self._build_dynamic_length_hint(
                target_message, user_style_guide
            )
            extra_info_parts.append(length_hint)
            harassment_signal = self._analyze_harassment_pressure(
                incoming_batch
            )
            style_route = self._decide_reply_style(
                target_message=target_message,
                relation_snapshot=relation_view,
                harassment_signal=harassment_signal,
            )
            extra_info_parts.append(
                f"[回复形式] {
                    style_route['reply_style']} | {
                    style_route['reason']}"
            )
            # 注入多维状态系统的LLM提示词
            self._inject_dimension_state_prompt(extra_info_parts)
            extra_info = build_reply_context_block(
                recent_context="",
                relevant_context="",
                extra_info="\n".join(
                    part for part in extra_info_parts if part
                ),
                recent_reply_guard="",
            )
            _key_lines = [
                l
                for l in (extra_info or "").split("\n")
                if any(
                    k in l
                    for k in (
                        "内心独白",
                        "情感状态",
                        "灵魂指令",
                        "冷拒模式",
                        "烦躁",
                        "烦",
                        "厌烦",
                        "防御",
                        "情绪保护",
                    )
                )
            ]
            if _key_lines:
                logger.info(
                    f"{self.log_prefix} 🧠 传入LLM的灵魂数据:\n"
                    + "\n".join(_key_lines[:8])
                )
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

            record_replyer_action_temp(
                chat_id=self.stream_id,
                reason=reply_reason,
                think_level=1,
            )

            available_actions = {}
            try:
                await self.action_modifier.modify_actions()
                available_actions = self.action_manager.get_using_actions()
            except Exception as e:
                logger.warning(f"{self.log_prefix} 获取可用动作失败: {e}")

            success, llm_response = await generator_api.generate_reply(
                chat_stream=self.chat_stream,
                reply_message=target_message,
                available_actions=available_actions,
                chosen_actions=[],
                reply_reason=reply_reason,
                unknown_words=None,
                enable_tool=global_config.tool.enable_tool,
                request_type="voice_driven_reply",
                from_plugin=False,
                reply_time_point=time.time(),
                think_level=1,
                extra_info=extra_info,
            )

            if not success or not llm_response or not llm_response.reply_set:
                self._mark_message_content_deferred(
                    target_message, "voice_generation_failed"
                )
                logger.warning(f"{self.log_prefix} 💭 回复生成失败")
                return False

            response_set = llm_response.reply_set
            selected_expressions = llm_response.selected_expressions

            cycle_timers = {}
            thinking_id = f"voice_driven_{int(time.time() * 1000)}"

            loop_info, reply_text, _ = await self._send_and_store_reply(
                response_set=response_set,
                action_message=target_message,
                cycle_timers=cycle_timers,
                thinking_id=thinking_id,
                actions=[],
                selected_expressions=selected_expressions,
                quote_message=bool(style_route.get("quote_message", False)),
            )

            self.last_active_time = time.time()
            self._remember_bot_utterance(reply_text)
            self._register_bot_message_record(
                reply_text=reply_text,
                loop_info=loop_info,
                target_message=target_message,
                was_proactive=False,
            )
            asyncio.create_task(
                self._capture_reply_behavior_learning(
                    reply_text=reply_text,
                    reply_reason=reply_reason,
                    relation_snapshot=relation_view,
                    was_proactive=False,
                )
            )
            self._mark_message_content_processed(
                target_message,
                action_name="voice_driven_reply",
                quality=0.85,
            )
            logger.info(f"{self.log_prefix} 💭 成功发送: {reply_text[:50]}...")
            try:
                from src.memory_system.memory_retrieval import (
                    invalidate_memory_cache,
                )

                invalidate_memory_cache(self.stream_id)
            except Exception as e:
                logger.debug(f"{self.log_prefix} 清除记忆缓存异常: {e}")
            asyncio.create_task(
                self._update_user_impression_after_reply(reply_text)
            )
            return True

        except Exception as exc:
            self._mark_message_content_deferred(
                self._get_latest_human_message(incoming_batch),
                "voice_reply_exception",
            )
            logger.error(f"{self.log_prefix} 💭 执行失败: {exc}")
            return False

    # ═══════════════════════════════════════════════════
    #  双通道架构：主动感兴趣通道（后台常驻）
    # ═══════════════════════════════════════════════════

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
        _base_interval = 10.0
        _min_interval = 5.0
        _max_interval = 25.0
        _cooldown_until = 0.0
        _consecutive_fails = 0
        _loop_count = 0
        logger.info(
            f"{self.log_prefix} 🔔 主动通道后台循环启动，基础间隔={_base_interval}s"
        )
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
                    _er = float(getattr(_ms, "chat_energy_ratio", 1.0) or 1.0)
                    if _er < 0.3:
                        _energy_mod = 2.5
                        _base_interval = min(
                            _max_interval, _base_interval + 1.0
                        )
                    elif _er > 0.8:
                        _energy_mod = 0.7
                        _base_interval = max(
                            _min_interval, _base_interval - 0.5
                        )
                else:
                    _base_interval = 10.0
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
                except Exception:
                    pass
                _silence = 0.0
                try:
                    from src.chat.proactive.silence_watcher import (
                        get_quiet_monitor,
                    )

                    _silence = get_quiet_monitor().measure_silence_sec(
                        self.stream_id
                    )
                except Exception:
                    pass
                if _silence < 6.0 and _loop_count > 3:
                    await asyncio.sleep(_interval * 0.5)
                    continue
                _should_act = False
                _act_reason = ""
                try:
                    _should_act, _act_reason = (
                        await self._evaluate_proactive_opportunity(
                            now=_now_loop,
                            silence_sec=_silence,
                            is_background=True,
                        )
                    )
                except Exception as exc:
                    logger.debug(f"{self.log_prefix} 主动通道评估异常: {exc}")
                if not _should_act:
                    _consecutive_fails = 0
                    await asyncio.sleep(_interval)
                    continue
                logger.info(
                    f"{self.log_prefix} 🔔 主动通道触发: {_act_reason} (静默={_silence:.0f}s)"
                )
                try:
                    _acted = await self._execute_proactive_action(
                        reason=_act_reason,
                        silence_sec=_silence,
                    )
                    if _acted:
                        # 主动发言成功后冷却2-4分钟，避免连续自言自语
                        _cooldown_until = time.time() + random.uniform(
                            120.0, 240.0
                        )
                        _consecutive_fails = 0
                        _base_interval = max(
                            _min_interval, _base_interval - 1.0
                        )
                    else:
                        _consecutive_fails += 1
                        _base_interval = min(
                            _max_interval, _base_interval + 0.5
                        )
                except Exception as exc:
                    logger.warning(
                        f"{self.log_prefix} 主动通道执行失败: {exc}"
                    )
                    _consecutive_fails += 1
                if _consecutive_fails >= 3:
                    _backoff = min(30.0, 5.0 * _consecutive_fails)
                    logger.debug(
                        f"{self.log_prefix} 主动通道连续{_consecutive_fails}次失败，退避{_backoff:.0f}s"
                    )
                    await asyncio.sleep(_backoff)
                    _consecutive_fails = 0
                else:
                    await asyncio.sleep(_interval)
        except asyncio.CancelledError:
            logger.info(f"{self.log_prefix} 🔔 主动通道后台循环被取消")
        except Exception as exc:
            logger.error(
                f"{self.log_prefix} 🔔 主动通道后台循环异常退出: {exc}"
            )

    # ── 空闲主动行为路径（保留作为主循环无消息时的快速路径） ──

    _IDLE_PROACTIVE_COOLDOWN_SEC = 120.0

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
        # 后台通道冷却提升到90s，避免频繁触发主动发言
        _cooldown = (
            self._IDLE_PROACTIVE_COOLDOWN_SEC if not is_background else 90.0
        )
        if (now - self._last_idle_proactive_ts) < _cooldown:
            return False, "冷却中"
        if self._cached_night_phase is not None:
            from src.core.night_cycle_system import NightPhase

            _np = self._cached_night_phase
            if _np in (NightPhase.DEEP_SLEEP, NightPhase.BURNED_OUT):
                return False, "夜间冻结"
            if _np == NightPhase.LIGHT_SLEEP:
                try:
                    from src.core.night_cycle_system import get_night_cycle

                    if not get_night_cycle(
                        self.stream_id
                    ).evaluate_sleep_peek():
                        return False, "浅睡无窥屏窗"
                except Exception:
                    pass
        if self._should_back_off_idle_proactive():
            return False, f"连续{self._unanswered_bot_turns}次未获回应"
        try:
            from src.modules.trauma.trauma_system import get_trauma_system

            _ts = get_trauma_system().get_state()
            if _ts.stress_accumulation > 8.0 or _ts.inner_chaos_level > 8.0:
                return False, f"压力过高(stress={_ts.stress_accumulation:.1f})"
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        arbiter_verdict = self._evaluate_proactive_arbiter(silence_sec)
        if not arbiter_verdict.should_proceed:
            return False, arbiter_verdict.rationale or "决策器否决"
        return True, arbiter_verdict.rationale or "决策器通过"

    async def _execute_proactive_action(
        self, *, reason: str, silence_sec: float
    ) -> bool:
        """执行主动回复动作（深度执行，可能含LLM调用）

        Returns:
            True 表示成功发送了主动回复
        """
        now_act = time.time()
        await self._update_trauma_system_state([])
        emotion_snapshot = self._update_emotion_state(
            now_act, relation_result={}
        )
        if not emotion_snapshot:
            return False
        awareness_snapshot = None
        ambient_info = self._sample_channel_ambient()
        voice_conclusion = None
        _run_p = self._should_run_perception(now_act)
        _run_v = self._should_run_voice(now_act, [], None)
        _jobs: list = []
        _keys: list = []
        if _run_p:
            _jobs.append(self._invoke_perception([], now_act))
            _keys.append("p")
        if _run_v:
            _jobs.append(
                self._invoke_inner_voice(
                    [], self._cached_awareness, ambient_info, now_act
                )
            )
            _keys.append("v")
        if _jobs:
            _res = await asyncio.gather(*_jobs, return_exceptions=True)
            _map = dict(zip(_keys, _res))
            if _run_p:
                _ip = _map.get("p")
                if _ip is not None and not isinstance(_ip, BaseException):
                    awareness_snapshot = _ip
                    self._cached_awareness = awareness_snapshot
                    self._last_perception_ts = now_act
            if _run_v:
                _iv = _map.get("v")
                if _iv is not None and not isinstance(_iv, BaseException):
                    voice_conclusion = _iv
        if voice_conclusion is None:
            voice_conclusion = self._cached_voice_conclusion
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
                _ctx_lines = [
                    f"- {t[:80]}" for t in _bot_recent[-3:]
                ]
                _bot_context = "你最近说过的话:\n" + "\n".join(_ctx_lines)
            # 从内心独白导出最近的观察记忆（看过什么、想过什么、为什么沉默）
            _observation_ctx = ""
            try:
                from src.chat.heart_flow.inner_voice import get_self_dialogue_engine
                _obs_engine = get_self_dialogue_engine(self.stream_id)
                _observation_ctx = _obs_engine.export_observation_context()
            except Exception:
                pass
            if _observation_ctx:
                _bot_context = (
                    f"{_bot_context}\n{_observation_ctx}" if _bot_context else _observation_ctx
                )
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
                        getattr(_rm, "processed_plain_text", "")
                        or getattr(_rm, "plain_text", "")
                        or ""
                    ).strip()
                    if _rm_text:
                        _dialogue_lines.append(f"{_rm_nick}: {_rm_text[:120]}")
            except Exception:
                pass
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
                _nick = str(
                    getattr(_rm, "user_nickname", "")
                    or getattr(_rm, "user_cardname", "")
                    or _uid
                )
                if _uid and _uid != "bot" and _nick:
                    _active_users[_uid] = _nick
            if _active_users:
                _atmosphere_parts.append(
                    f"最近参与聊天的人: {', '.join(_active_users.values())}"
                )
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
                        _user_state = _emo_tracker.get_user_state(
                            _uid, create_if_missing=False
                        )
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
                except Exception:
                    pass
            _relationship_ctx = (
                "你对群里这些人的感觉:\n" + "\n".join(_relationship_parts)
                if _relationship_parts else ""
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
                logger.info(
                    f"{self.log_prefix} [流程规划] 规划器否决主动发言"
                )
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
            logger.debug(
                f"{self.log_prefix} 流程规划器异常(降级): {_fp_exc}"
            )

        # 从内心独白提取实际 desire，不再硬编码
        _actual_desire = 5
        if voice_conclusion is not None:
            _actual_desire = int(
                getattr(voice_conclusion, "reply_desire_level", 5) or 5
            )
        # 从规划器结果提取话题和情绪
        _proactive_topic = ""
        _proactive_emotion = ""
        _delivery_form = "standalone"
        _mention_user_name = ""
        _reference_user_name = ""
        if _proactive_plan is not None:
            _proactive_topic = str(
                getattr(_proactive_plan, "extra_notes", "") or ""
            )
            _proactive_emotion = str(
                getattr(_proactive_plan, "emotion_hint", "") or ""
            )
            _delivery_form = str(
                getattr(_proactive_plan, "delivery_form", "standalone") or "standalone"
            )
            _mention_user_name = str(
                getattr(_proactive_plan, "mention_user_name", "") or ""
            )
            _reference_user_name = str(
                getattr(_proactive_plan, "reference_user_name", "") or ""
            )

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
        except Exception:
            pass

        acted = await self._generate_and_send_proactive_reply(
            incoming_batch=_proactive_incoming,
            desire=_actual_desire,
            thought=thought_text,
            target_uid=self._resolve_latest_human_user_id() or "",
            awareness=awareness_snapshot,
            ambient=ambient_info,
            emotion=emotion_snapshot,
            arbiter_reason=reason,
            proactive_topic=_proactive_topic,
            proactive_emotion=_proactive_emotion,
            delivery_form=_delivery_form,
            mention_user_name=_mention_user_name,
            reference_user_name=_reference_user_name,
        )
        if acted:
            self._last_idle_proactive_ts = now_act
            self._on_bot_sent(was_proactive=True)
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
            raw_text = (
                getattr(latest, "processed_plain_text", "") if latest else ""
            )
            speaker_name = (
                getattr(latest, "user_nickname", "") or "" if latest else ""
            )
            speaker_id = getattr(latest, "user_id", "") or "" if latest else ""

            psych = {
                "mood": (
                    getattr(voice_conclusion, "current_mood", "")
                    if voice_conclusion
                    else "一般"
                ),
                "mental_fatigue": 0.0,
                "annoyance": float(
                    getattr(self, "_cached_annoyance", 0.0) or 0.0
                ),
                "favor": int(
                    float(
                        getattr(self, "_cached_affection_value", 50.0) or 50.0
                    )
                ),
            }

            current_state_info = (
                f"模型内心抗拒='{thinking_text[:50] if thinking_text else '无'}', "
                f"请决定是否接管"
            )

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
                            getattr(_msg, "processed_plain_text", "")
                            or getattr(_msg, "plain_text", "")
                            or ""
                        )
                        if _content:
                            _dialogue_history.append(
                                {"role": "user", "content": _content}
                            )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            _rapport_desc = ""
            _profile_desc = ""
            _memory_desc = ""
            _condition_desc = ""
            try:
                _rel_snap = getattr(self, "_last_relation_snapshot", {}) or {}
                _rapport_desc = f"{speaker_name}: 好感{
                    _rel_snap.get(
                        'affection',
                        '?')}, 信任{
                    _rel_snap.get(
                        'trust_value',
                        '?')}, 烦躁{
                    _rel_snap.get(
                        'annoyance_value',
                        '?')}"
                _profile_desc = (
                    _rel_snap.get("profile_summary", "") or "(暂无稳定画像)"
                )
                _condition_desc = f"思考值={
                    _rel_snap.get(
                        'thinking_ratio',
                        '?')}, 聊天值={
                    _rel_snap.get(
                        'chat_fuel',
                        '?')}, 活跃度={
                    _rel_snap.get(
                        'activity_level',
                        '?')}"
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            is_at_me = (
                getattr(latest, "mentioned_me", False) if latest else False
            )

            learner.on_model_rejection(now)

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
                dialogue_history=(
                    _dialogue_history if _dialogue_history else None
                ),
                rapport_desc=_rapport_desc,
                profile_desc=_profile_desc,
                memory_desc=_memory_desc,
                condition_desc=_condition_desc,
            )

            if decision:
                action = decision.get("action", "待机")
                thought = decision.get("thought", "")
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
                else:
                    learner.model_takeover_response(False)

                return {
                    "wants_to_takeover": wants_to_takeover,
                    "action": action,
                    "reason": decision.get("reason", ""),
                    "thought": decision.get("thought", ""),
                    "decision": decision,
                }

            learner.reset_state_expiration()

        except Exception as exc:
            logger.debug(f"{self.log_prefix} 算法接管请求失败: {exc}")
        return None

    async def _run_peek_observe_loop(self, incoming_batch: List) -> None:
        """窥屏态观察循环：更新内部状态并消耗少量能量"""
        try:
            from src.chat.heart_flow.energy_manager import (
                get_shared_resource_manager,
            )

            shared_mgr = get_shared_resource_manager()
            shared_mgr.apply_input_activity(
                self.stream_id,
                content_length=sum(
                    len(getattr(m, "content", "") or "")
                    for m in incoming_batch[-5:]
                ),
                is_repeat=len(incoming_batch) > 1,
                user_count=len(
                    set(
                        getattr(m, "user_id", "") or ""
                        for m in incoming_batch
                        if getattr(m, "user_id", "") != "bot"
                    )
                ),
            )
            # 窥屏也消耗少量聊天池和思考池（"看一眼"的认知成本）
            shared_mgr.deduct_chat_value(
                self.stream_id, 0.0, reason="peek"
            )
            shared_mgr.deduct_activity_level(
                self.stream_id, 0.0, reason="think"
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            _wm = self._cached_watch_level
            _wl = _wm.value if hasattr(_wm, "value") else str(_wm)
            logger.info(
                f"{self.log_prefix} 👁 窥屏态观察完成 级别={_wl} 不回复"
            )
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
        # 仅保留冷却保护防止过于频繁的 LLM 调用
        now = time.time()
        if (now - self._last_voice_ts) < _VOICE_COOLDOWN_SEC:
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
                self._cached_voice = verdict
            return verdict
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 窥屏反思失败: {exc}")
            return None

    async def _try_idle_proactive(
        self, now: float, silence_sec: float
    ) -> bool:
        """无新消息时的自主行为路径

        不用硬编码沉默阈值，而是让情感系统自然累积感受，
        由情感状态决定是否想要主动行为。
        返回 True 表示已执行主动行为。
        """
        if not hasattr(self, "_last_idle_proactive_ts"):
            self._last_idle_proactive_ts = 0.0
        if (
            now - self._last_idle_proactive_ts
        ) < self._IDLE_PROACTIVE_COOLDOWN_SEC:
            return False
        # F5：夜间节律冻结——深睡/熬穿/浅睡(无窥屏窗) 时禁止空闲主动
        if self._cached_night_phase is not None:
            from src.core.night_cycle_system import NightPhase

            _np = self._cached_night_phase
            if _np in (NightPhase.DEEP_SLEEP, NightPhase.BURNED_OUT):
                return False
            if _np == NightPhase.LIGHT_SLEEP:
                try:
                    from src.core.night_cycle_system import get_night_cycle

                    if not get_night_cycle(
                        self.stream_id
                    ).evaluate_sleep_peek():
                        return False
                except Exception:
                    pass
        if self._should_back_off_idle_proactive():
            logger.info(
                f"{self.log_prefix} 💤 连续{self._unanswered_bot_turns}次主动未获回应，跳过本轮主动发言"
            )
            self._last_idle_proactive_ts = now
            return False

        # ── 空闲期核心状态同步（修复：空闲路径原先完全跳过核心系统集成） ──
        # 创伤/压力门控：压力过高时拦截主动行为
        try:
            from src.modules.trauma.trauma_system import get_trauma_system

            _idle_trauma_state = get_trauma_system().get_state()
            if (
                _idle_trauma_state.stress_accumulation > 8.0
                or _idle_trauma_state.inner_chaos_level > 8.0
            ):
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
            _idle_user_id = self._resolve_latest_human_user_id()
            _idle_memoir = (
                _idle_cabinet.retrieve(_idle_user_id)
                if hasattr(_idle_cabinet, "retrieve")
                else None
            )
            if _idle_memoir and hasattr(_idle_memoir, "phase"):
                _idle_dialogue_phase = (
                    _idle_memoir.phase.value
                    if hasattr(_idle_memoir.phase, "value")
                    else str(_idle_memoir.phase)
                )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        emotion_snapshot = self._update_emotion_state(now, relation_result={})
        if not emotion_snapshot:
            return False

        # ── 统一决策器评估（融合多源信号） ──
        arbiter_verdict = self._evaluate_proactive_arbiter(silence_sec)

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
                f"{self.log_prefix} 💤 沉默 {silence_sec:.0f}s 不想主动说话 ({arbiter_verdict.rationale})"
            )
            self._last_idle_proactive_ts = now
            return False
        reason = arbiter_verdict.rationale
        logger.info(
            f"{self.log_prefix} 💬 沉默 {silence_sec:.0f}s 想主动说话: {reason}"
            + (
                f" | 对话阶段={_idle_dialogue_phase}"
                if _idle_dialogue_phase
                else ""
            )
        )
        awareness_snapshot = None
        ambient_info = self._sample_channel_ambient()
        voice_conclusion = None
        # 空闲主动路径也并行化感知 + 内心独白
        _idle_run_p = self._should_run_perception(now)
        _idle_run_v = self._should_run_voice(now, [], None)
        _idle_jobs: list = []
        _idle_keys: list = []
        if _idle_run_p:
            _idle_jobs.append(self._invoke_perception([], now))
            _idle_keys.append("p")
        if _idle_run_v:
            _idle_jobs.append(
                self._invoke_inner_voice(
                    [], self._cached_awareness, ambient_info, now
                )
            )
            _idle_keys.append("v")
        if _idle_jobs:
            _idle_res = await asyncio.gather(
                *_idle_jobs, return_exceptions=True
            )
            _idle_map = dict(zip(_idle_keys, _idle_res))
            if _idle_run_p:
                _ip = _idle_map.get("p")
                if _ip is not None and not isinstance(_ip, BaseException):
                    awareness_snapshot = _ip
                    self._cached_awareness = awareness_snapshot
                    self._last_perception_ts = now
            if _idle_run_v:
                _iv = _idle_map.get("v")
                if _iv is not None and not isinstance(_iv, BaseException):
                    voice_conclusion = _iv
                    self._cached_voice = voice_conclusion
                    self._last_voice_ts = now
                    _thought_len = len(str(getattr(voice_conclusion, "thinking", "") or ""))
                    _think_complexity = max(0.5, min(2.0, _thought_len / 40.0))
                    self._apply_think_drain(complexity=_think_complexity)
        if awareness_snapshot is None:
            awareness_snapshot = self._cached_awareness
        if voice_conclusion is None:
            voice_conclusion = self._cached_voice
        # 空闲路径：如果没有新鲜的内心独白，尝试自主思考
        if voice_conclusion is None or not getattr(
            voice_conclusion, "is_valid", False
        ):
            try:
                from src.chat.heart_flow.inner_voice import (
                    get_self_dialogue_engine,
                )

                auto_verdict = await get_self_dialogue_engine(
                    self.stream_id
                ).autonomous_reflection(
                    self.stream_id,
                    trigger_reason=reason[:100],
                )
                if auto_verdict and auto_verdict.is_valid:
                    voice_conclusion = auto_verdict
                    self._cached_voice = voice_conclusion
                    # 自主思考结果也写入意图池
                    try:
                        from src.chat.proactive.intention_pool import (
                            get_intention_pool,
                        )

                        get_intention_pool().ingest_voice_verdict(
                            channel_id=self.stream_id,
                            verdict=auto_verdict,
                            speaker_id=str(
                                getattr(self, "_last_user_id", "") or ""
                            ),
                        )
                    except Exception:
                        pass
            except Exception as _auto_err:
                logger.debug(f"{self.log_prefix} 自主思考跳过: {_auto_err}")
        logger.info(f"{self.log_prefix} 💤 想主动说话，走专用主动回复执行器")
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
        # 收集最近消息作为上下文
        _idle_recent = []
        try:
            _idle_recent = message_api.get_messages_by_time_in_chat(
                chat_id=self.stream_id,
                start_time=time.time() - 600,
                end_time=time.time(),
                limit=10,
                limit_mode="latest",
                filter_mai=True,
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # 捕获触发本次主动的核心意图id（供闭环追踪）
        _idle_proactive_intent_id = ""
        if voice_conclusion and hasattr(
            voice_conclusion, "dominant_unfinished_intent"
        ):
            _dom = getattr(
                voice_conclusion, "dominant_unfinished_intent", None
            )
            if _dom and isinstance(_dom, dict):
                _idle_proactive_intent_id = str(
                    _dom.get("intent_id", "") or ""
                )
        self._last_proactive_intent_id = _idle_proactive_intent_id
        replied = await self._execute_proactive_reply(
            llm_decision=_idle_decision,
            incoming_batch=_idle_recent,
        )
        if replied:
            self._on_bot_sent(was_proactive=True)
            logger.info(
                f"{self.log_prefix} 💤 主动回复发送成功"
                + (
                    f" source_intent={_idle_proactive_intent_id[:10]}"
                    if _idle_proactive_intent_id
                    else ""
                )
            )
        else:
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
        try:
            from src.chat.heart_flow.llm_autonomous_planner import (
                AutonomousDecision,
            )
            from src.common.data_models.info_data_model import (
                ActionPlannerInfo,
            )
            from src.plugin_system.base.component_types import ActionInfo
            from src.chat.utils.timer_calculator import Timer
            from src.config.config import global_config
            from src.plugin_system.apis import generator_api, database_api
            from src.chat.utils.utils import record_replyer_action_temp

            logger.info(
                f"{self.log_prefix} 开始执行主动回复: "
                f"意图={llm_decision.social_intention}, "
                f"内容规划={llm_decision.content_plan[:50] if llm_decision.content_plan else '无'}, "
                f"投递={delivery_form}, 引用={reference_user_name}, @={mention_user_name}"
            )

            # 根据投递策略选择目标消息
            target_message = None
            _need_quote = "quote" in delivery_form
            _need_mention = "mention" in delivery_form
            if _need_quote and reference_user_name:
                # 规划器指定了引用对象，从近期消息中找到该用户最后一条消息
                _ref_lower = reference_user_name.lower().strip()
                for msg in reversed(incoming_batch[-15:]):
                    _msg_nick = str(
                        getattr(msg, "user_nickname", "")
                        or getattr(msg, "user_cardname", "")
                        or ""
                    ).lower().strip()
                    if _ref_lower and _ref_lower in _msg_nick:
                        target_message = msg
                        break
            if target_message is None:
                # 回退：取最近一条用户消息
                for msg in reversed(incoming_batch[-10:]):
                    if getattr(msg, "user_id", "") != "bot":
                        target_message = msg
                        break
            target_message = self._select_preferred_reply_message(
                target_message, list(incoming_batch[-10:])
            )
            self._mark_message_content_processing(target_message)
            voice_summary = self._build_voice_execution_summary(
                getattr(self, "_cached_voice", None), target_message
            )

            # 构建回复原因
            reply_reason = (
                f"主动行为: {llm_decision.social_intention or '想说话'}"
            )
            if llm_decision.content_plan:
                reply_reason += (
                    f" | 内容规划: {llm_decision.content_plan[:100]}"
                )
            if voice_summary.get("reason"):
                reply_reason += f" | {voice_summary['reason'][:140]}"

            # 构建额外信息，传递给回复生成器
            extra_info = voice_summary.get("panel", "")
            if llm_decision.content_plan:
                extra_info = (
                    f"{extra_info}\n[主动行为意图] {
                        llm_decision.social_intention or ''}\n[内容规划] {
                        llm_decision.content_plan}"
                    if extra_info
                    else f"[主动行为意图] {
                        llm_decision.social_intention or ''}\n[内容规划] {
                        llm_decision.content_plan}"
                )
                if llm_decision.emotional_state:
                    extra_info += (
                        f"\n[当前感受] {llm_decision.emotional_state}"
                    )
            voice_guard = self._build_voice_execution_guard(
                getattr(self, "_cached_voice", None), target_message
            )
            if voice_guard:
                extra_info = (
                    f"{extra_info}\n{voice_guard}"
                    if extra_info
                    else voice_guard
                )
            self_memory = self._build_self_reply_memory()
            continuity_context = self._build_self_continuity_context()
            if self_memory:
                extra_info = (
                    f"{extra_info}\n{self_memory}"
                    if extra_info
                    else self_memory
                )
            if continuity_context:
                extra_info = (
                    f"{extra_info}\n{continuity_context}"
                    if extra_info
                    else continuity_context
                )
            # 注入内心独白的观察记忆（最近看到什么、想了什么、为什么沉默）
            try:
                from src.chat.heart_flow.inner_voice import get_self_dialogue_engine
                _obs_ctx = get_self_dialogue_engine(self.stream_id).export_observation_context()
                if _obs_ctx:
                    extra_info = (
                        f"{extra_info}\n[最近观察]\n{_obs_ctx}"
                        if extra_info
                        else f"[最近观察]\n{_obs_ctx}"
                    )
            except Exception:
                pass

            user_style_guide = ""
            if target_message:
                target_user_id = getattr(target_message, "user_id", "") or ""
                if target_user_id:
                    user_style_guide = self._get_user_style_guide(
                        target_user_id
                    )
            if user_style_guide:
                extra_info = (
                    f"{extra_info}\n{user_style_guide}"
                    if extra_info
                    else user_style_guide
                )
            persona_hint = self._build_persona_hint()
            if persona_hint:
                extra_info = (
                    f"{extra_info}\n{persona_hint}"
                    if extra_info
                    else persona_hint
                )
            relation_view = self._resolve_relation_view()
            reply_style_context = self._build_reply_style_context(
                relation_view
            )
            if reply_style_context:
                extra_info = (
                    f"{extra_info}\n{reply_style_context}"
                    if extra_info
                    else reply_style_context
                )
            length_hint = self._build_dynamic_length_hint(
                target_message, user_style_guide
            )
            extra_info = (
                f"{extra_info}\n{length_hint}" if extra_info else length_hint
            )
            harassment_signal = self._analyze_harassment_pressure(
                incoming_batch
            )
            # 主动回复投递策略：规划器指定的策略优先于被动式 _decide_reply_style
            if _need_quote or _need_mention:
                _proactive_style_reasons = []
                if _need_quote:
                    _proactive_style_reasons.append(
                        f"规划器指定引用{reference_user_name or '某人'}的消息"
                    )
                if _need_mention:
                    _proactive_style_reasons.append(
                        f"规划器指定@{mention_user_name}"
                    )
                style_route = {
                    "reply_style": "quote" if _need_quote else "direct",
                    "quote_message": _need_quote,
                    "reason": "；".join(_proactive_style_reasons),
                }
            else:
                style_route = self._decide_reply_style(
                    target_message=target_message,
                    relation_snapshot=relation_view,
                    harassment_signal=harassment_signal,
                )
            # @mention：将用户昵称注入到内容规划中，让生成器在回复中自然提到对方
            if _need_mention and mention_user_name:
                _mention_hint = f"\n[投递指令] 这条消息需要@{mention_user_name}，在开头或合适的位置自然地提到对方名字"
                extra_info = (
                    f"{extra_info}{_mention_hint}" if extra_info else _mention_hint
                )
            extra_info = (
                f"{extra_info}\n[回复形式] {
                    style_route['reply_style']} | {
                    style_route['reason']}"
                if extra_info
                else f"[回复形式] {
                    style_route['reply_style']} | {
                    style_route['reason']}"
            )
            self._emit_reply_generation_summary(
                target_message=target_message,
                style_route=style_route,
                relation_snapshot=relation_view,
                context_execution_block="",
                extra_info=extra_info,
                source="proactive",
            )

            # 记录回复动作
            record_replyer_action_temp(
                chat_id=self.stream_id,
                reason=reply_reason,
                think_level=1,
            )

            # 获取可用动作
            available_actions: Dict[str, ActionInfo] = {}
            try:
                await self.action_modifier.modify_actions()
                available_actions = self.action_manager.get_using_actions()
            except Exception as e:
                logger.warning(f"{self.log_prefix} 获取可用动作失败: {e}")

            # 生成回复
            success, llm_response = await generator_api.generate_reply(
                chat_stream=self.chat_stream,
                reply_message=target_message,
                available_actions=available_actions,
                chosen_actions=[],
                reply_reason=reply_reason,
                unknown_words=None,
                enable_tool=global_config.tool.enable_tool,
                request_type="proactive_reply",
                from_plugin=False,
                reply_time_point=time.time(),
                think_level=1,
                extra_info=extra_info,
            )

            if not success or not llm_response or not llm_response.reply_set:
                self._mark_message_content_deferred(
                    target_message, "proactive_generation_failed"
                )
                logger.warning(f"{self.log_prefix} 回复生成失败")
                return False

            # 发送回复
            response_set = llm_response.reply_set
            selected_expressions = llm_response.selected_expressions

            cycle_timers: Dict[str, float] = {}
            thinking_id = f"proactive_{int(time.time() * 1000)}"

            loop_info, reply_text, _ = await self._send_and_store_reply(
                response_set=response_set,
                action_message=target_message,
                cycle_timers=cycle_timers,
                thinking_id=thinking_id,
                actions=[],
                selected_expressions=selected_expressions,
                quote_message=bool(style_route.get("quote_message", False)),
            )

            self.last_active_time = time.time()
            self._remember_bot_utterance(reply_text)
            self._register_bot_message_record(
                reply_text=reply_text,
                loop_info=loop_info,
                target_message=target_message,
                was_proactive=True,
            )
            asyncio.create_task(
                self._capture_reply_behavior_learning(
                    reply_text=reply_text,
                    reply_reason=reply_reason,
                    relation_snapshot=relation_view,
                    was_proactive=True,
                )
            )
            self._mark_message_content_processed(
                target_message,
                action_name="proactive_reply",
                quality=0.8,
            )
            logger.info(f"{self.log_prefix} 成功发送: {reply_text[:50]}...")
            try:
                from src.memory_system.memory_retrieval import (
                    invalidate_memory_cache,
                )

                invalidate_memory_cache(self.stream_id)
            except Exception as e:
                logger.debug(f"{self.log_prefix} 清除记忆缓存异常: {e}")
            return True

        except Exception as exc:
            self._mark_message_content_deferred(
                self._get_latest_human_message(incoming_batch),
                "proactive_reply_exception",
            )
            logger.error(f"{self.log_prefix} 执行失败: {exc}")
            return False

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
            _emotion_label = proactive_emotion or (
                getattr(emotion, "dominant_emotion", "平静") if emotion else "平静"
            )
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
            return await self._execute_proactive_reply(
                llm_decision=decision,
                incoming_batch=incoming_batch,
                delivery_form=delivery_form,
                mention_user_name=mention_user_name,
                reference_user_name=reference_user_name,
            )

        except Exception as exc:
            logger.error(f"{self.log_prefix} 主动回复生成失败: {exc}")
            return False

    # ═══════════════════════════════════════════════════
    #  阶段 5：行动决策融合
    # ═══════════════════════════════════════════════════

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
        freq_adjust = (
            frequency_control_manager.get_or_create_frequency_control(
                self.stream_id
            ).get_talk_frequency_adjust()
        )
        base_talk = global_config.chat.get_talk_value(self.stream_id)
        combined_prob = base_talk * freq_adjust * eagerness
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
        return random.random() < combined_prob

    # ═══════════════════════════════════════════════════
    #  状态迁移辅助
    # ═══════════════════════════════════════════════════

    def _shift_to_dormant(self, cause: str = "") -> None:
        """迁移至休息阶段"""
        try:
            from src.chat.heart_flow.state_machine import get_phase_coordinator
            from src.chat.heart_flow.rest_handler import (
                get_dormancy_supervisor,
            )
            from src.common.data_models.heartflow_models import FlowPhase

            coordinator = get_phase_coordinator()
            if coordinator.is_dormant(self.stream_id):
                return
            shifted = coordinator.attempt_shift(
                self.stream_id,
                FlowPhase.DORMANT,
                reason=cause,
            )
            if shifted:
                get_dormancy_supervisor().begin_dormancy(
                    self.stream_id, cause=cause
                )
                logger.info(f"{self.log_prefix} 迁移至休息: {cause}")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 迁移休息失败: {exc}")

    def _shift_to_engaged(self, cause: str = "") -> None:
        """迁移至活跃阶段"""
        try:
            from src.chat.heart_flow.state_machine import get_phase_coordinator
            from src.common.data_models.heartflow_models import FlowPhase

            coordinator = get_phase_coordinator(self.stream_id)
            current = coordinator.current_phase(self.stream_id)
            if current == FlowPhase.ENGAGED:
                return
            if current in (FlowPhase.DORMANT, FlowPhase.LIGHT_REST):
                try:
                    from src.core.night_cycle_system import get_night_cycle

                    _nc2 = get_night_cycle(self.stream_id)
                    self._wake_grace_until = (
                        time.time() + _nc2.compute_wake_grace_duration()
                    )
                except Exception:
                    self._wake_grace_until = time.time() + 120.0
            coordinator.attempt_shift(
                self.stream_id,
                FlowPhase.ENGAGED,
                reason=cause,
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 迁移活跃失败: {exc}")

    def _shift_to_pending(
        self, cause: str = "", anticipated: str = ""
    ) -> None:
        """迁移至等待阶段"""
        try:
            from src.chat.heart_flow.state_machine import get_phase_coordinator
            from src.common.data_models.heartflow_models import FlowPhase

            coordinator = get_phase_coordinator()
            shifted = coordinator.attempt_shift(
                self.stream_id,
                FlowPhase.PENDING,
                reason=cause,
            )
            if shifted and anticipated:
                from src.chat.heart_flow.waiting_handler import (
                    get_pending_orchestrator,
                )

                get_pending_orchestrator().initiate_watch(
                    self.stream_id,
                    anticipated=anticipated,
                )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 迁移等待失败: {exc}")

    # ═══════════════════════════════════════════════════
    #  行为分析辅助
    # ═══════════════════════════════════════════════════

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
            self._cumulative_ratio = actual_bot_replies / max(
                1, actual_user_msgs
            )

        recent_user_msgs = sum(
            1 for t in self._user_msg_timeline if now_ts - t < 300.0
        )
        recent_bot_replies = sum(
            1 for t in self._bot_reply_timeline if now_ts - t < 300.0
        )

        if recent_bot_replies == 0:
            recent_ratio = 0.5
        else:
            recent_ratio = recent_bot_replies / max(1, recent_user_msgs)

        self._target_ratio = 0.4

        if self._is_in_burst:
            burst_intensity = min(1.0, self._burst_user_count / 10.0)
            self._target_ratio = 0.4 - (burst_intensity * 0.25)
            self._target_ratio = max(0.15, self._target_ratio)
        elif recent_user_msgs < 3 and recent_bot_replies < 2:
            self._target_ratio = 0.5
        elif recent_user_msgs > 20 and recent_bot_replies > 10:
            self._target_ratio = 0.35

        ratio_diff = self._cumulative_ratio - self._target_ratio
        if ratio_diff > 0.15:
            self._reply_probability = max(0.1, 0.5 - ratio_diff * 2.0)
        elif ratio_diff < -0.15:
            self._reply_probability = min(0.9, 0.5 - ratio_diff * 1.5)
        else:
            self._reply_probability = 0.5

        if self._is_in_burst:
            burst_penalty = min(0.3, (self._burst_user_count - 3) * 0.05)
            self._reply_probability = max(
                0.1, self._reply_probability - burst_penalty
            )

        if now_ts - self._last_bot_reply_ts < 10.0:
            self._reply_probability = max(0.2, self._reply_probability - 0.15)

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

    # ═══════════════════════════════════════════════════
    #  氛围反馈
    # ═══════════════════════════════════════════════════

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
            from src.chat.heart_flow.energy_manager import get_vitality_pool

            pool = get_vitality_pool()
            snap = pool.capture_snapshot(self.stream_id)
            chat_energy = float(getattr(snap, "chat_pool", 0.0) or 0.0)
            chat_ceiling = float(getattr(snap, "chat_ceiling", 0.0) or 0.0)
            thinking_energy = float(
                getattr(snap, "thinking_value", 0.0) or 0.0
            )
            thinking_ceiling = float(
                getattr(snap, "thinking_ceiling", 0.0) or 0.0
            )
            freq_ctrl = (
                frequency_control_manager.get_or_create_frequency_control(
                    self.stream_id
                )
            )
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
            from src.chat.heart_flow.energy_manager import get_vitality_pool

            pool = get_vitality_pool()
            snap = pool.capture_snapshot(self.stream_id)
            base_cost = 3.0
            brain_cost = max(1.0, min(15.0, base_cost))
            updated = pool.apply_manual_adjustment(
                self.stream_id,
                thinking_delta=-brain_cost,
            )
            logger.info(
                f"{self.log_prefix} 消耗思考值 {brain_cost:.1f}，"
                f"剩余思考值 {updated['thinking_after']:.1f}"
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 规划消耗失败: {exc}")

    async def _compute_relation_metrics(
        self, messages: List
    ) -> Dict[str, Any]:
        """聚合共享层/关系层/情绪层，优先从世界快照返回统一交互状态。"""
        # 世界快照含目标用户数据时直接转换，跳过逐源散读
        snap = getattr(self, "_tick_world_snapshot", None)
        if snap is not None and snap.target_user.user_id:
            self._last_user_id = str(snap.target_user.user_id or "").strip()
            rd = snap.to_relation_dict()
            rd.setdefault("relationship_candidates", [])
            try:
                from src.person_info.person_info import get_unified_profile_hub

                hub = get_unified_profile_hub()
                top_rels = hub.get_top_relationships(
                    snap.target_user.user_id, limit=5
                )
                rd["relationship_candidates"] = [
                    (str(tid or "").strip(), float(w or 0.0))
                    for tid, w in top_rels
                    if str(tid or "").strip()
                ]
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            rd["attribute_influences"] = self._build_attribute_influence_map(
                rd
            )
            rd = self._inject_realtime_emotion(rd)
            return rd
        # 降级路径：无快照时委托 impression_evolution_hub 聚合
        user_id = ""
        if messages:
            for msg in reversed(messages):
                uid = getattr(msg, "user_id", "") or ""
                if uid and uid != "bot":
                    user_id = uid
                    self._last_user_id = user_id
                    break
        if not user_id:
            logger.debug(
                f"{self.log_prefix} 未找到有效用户ID，使用默认关系指标"
            )
        try:
            from src.core.impression_evolution_hub import get_impression_hub

            hub = get_impression_hub(self.stream_id)
            result = await hub.assemble_relation_snapshot(user_id)
        except Exception as exc:
            logger.warning(
                f"{self.log_prefix} 关系度聚合失败，从各引擎回填真实值: {exc}"
            )
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
                "trust_score": 0.0,
                "psychological_pressure": 0.0,
                "trauma_score": 0.0,
                "mood": "平静",
                "attribute_influences": {},
                "relationship_candidates": [],
            }
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
            logger.debug(
                f"{self.log_prefix} 关系值全零检测到，启动回填 user={user_id}"
            )
            try:
                from src.modules.social_value.social_value_core import (
                    get_social_value_core,
                )

                _svc = get_social_value_core()
                if user_id:
                    _sv = _svc.get_social_value(user_id, self.stream_id)
                    if _sv:
                        result["social_value"] = float(
                            getattr(_sv, "social_value", 5.0) or 5.0
                        )
                        result["affection"] = float(
                            getattr(_sv, "affection", 0.0) or 0.0
                        )
                        result["trust_score"] = float(
                            getattr(_sv, "trust_value", 0.0) or 0.0
                        )
                        result["trust_value"] = result["trust_score"]
                        _label = getattr(_sv, "relationship_label", None)
                        if _label:
                            result["custom_label"] = str(_label)
                        _level = getattr(_sv, "relationship_level", None)
                        if _level is not None:
                            result["relationship_level"] = int(_level)
                        _interactions = getattr(
                            _sv, "total_interactions", None
                        )
                        if _interactions is not None:
                            result["interaction_count"] = int(_interactions)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            if _safe_float("social_value") < 0.01:
                result["social_value"] = 5.0
            if _safe_float("trust_value") < 0.01:
                result["trust_value"] = 5.0
            if _safe_float("annoyance_value") < 0.01:
                result["annoyance_value"] = 0.0
            if _safe_float("affection") < 0.01:
                result["affection"] = 5.0
            # 从存在感引擎补充社交意愿/回避倾向
            _ps_fill = getattr(self, "_cached_presence_state", None)
            if _ps_fill:
                _sw_fill = float(
                    getattr(_ps_fill, "social_willingness", 0.5) or 0.5
                )
                _aw_fill = float(
                    getattr(_ps_fill, "avoidance_tendency", 0.0) or 0.0
                )
                if _sw_fill > 0.3:
                    result["social_value"] = max(
                        result.get("social_value", 0.0), _sw_fill * 10.0
                    )
                if _aw_fill > 0.3:
                    result["annoyance_value"] = max(
                        result.get("annoyance_value", 0.0), _aw_fill * 20.0
                    )
            # 从代谢约束补充心理压力/无聊驱动
            _mc_fill = getattr(self, "_cached_metabolism_constraints", None)
            if isinstance(_mc_fill, dict):
                _bored_fill = float(_mc_fill.get("boredom", 0.0) or 0.0)
                _pressure_fill = float(
                    _mc_fill.get("psychological_pressure", 0.0) or 0.0
                )
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
        # ── 无条件安全网：确保核心字段永远不为零（即使回填因异常被跳过） ──
        for _floor_key in ("social_value", "trust_value", "affection"):
            _fv = _safe_float(_floor_key)
            if _fv < 0.01:
                result[_floor_key] = (
                    5.0 if _floor_key != "annoyance_value" else 0.0
                )
        result["attribute_influences"] = self._build_attribute_influence_map(
            result
        )
        result = self._inject_realtime_emotion(result)
        return result

    def _build_attribute_influence_map(
        self, relation_state: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """构建四主属性的独立影响矩阵，不把不同层揉成一个总分。"""
        pressure = float(
            relation_state.get("psychological_pressure", 0.0) or 0.0
        )
        annoyance = float(relation_state.get("annoyance_value", 0.0) or 0.0)
        trust_value = float(relation_state.get("trust_value", 0.0) or 0.0)
        affection = float(relation_state.get("affection", 0.0) or 0.0)
        shared_social = float(
            relation_state.get("shared_social_value", 0.0) or 0.0
        )
        relation_social = float(relation_state.get("social_value", 0.0) or 0.0)
        trauma = float(relation_state.get("trauma_score", 0.0) or 0.0)

        return {
            "thinking_value": {
                "psychological_pressure_penalty": round(
                    -min(18.0, pressure * 0.12), 2
                ),
                "trauma_penalty": round(-min(12.0, trauma * 1.4), 2),
            },
            "chat_value": {
                "affection_bonus": round(
                    max(-6.0, min(8.0, affection * 0.05)), 2
                ),
                "annoyance_penalty": round(-min(12.0, annoyance * 0.08), 2),
                "trust_buffer": round(
                    max(-4.0, min(6.0, trust_value * 0.04)), 2
                ),
            },
            "activity_level": {
                "shared_social_drive": round(
                    max(-5.0, min(6.0, shared_social * 0.04)), 2
                ),
                "pressure_drag": round(-min(10.0, pressure * 0.06), 2),
            },
            "social_value": {
                "relation_social_bias": round(
                    max(-8.0, min(8.0, relation_social * 0.05)), 2
                ),
                "trust_bias": round(
                    max(-8.0, min(10.0, trust_value * 0.05)), 2
                ),
                "annoyance_drag": round(-min(14.0, annoyance * 0.09), 2),
                "trauma_sensitivity": round(-min(10.0, trauma * 0.9), 2),
            },
        }

    def _get_self_behavior_style_hints(
        self, relation_snapshot: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        try:
            from src.modules.recall.self_behavior_learner import (
                get_self_behavior_learner,
            )

            learner = get_self_behavior_learner()
            relation_stage = ""
            if relation_snapshot:
                relation_stage = str(
                    relation_snapshot.get("custom_label")
                    or relation_snapshot.get("relationship_level")
                    or ""
                ).strip()
            return learner.get_style_hints(
                self.stream_id, relation_stage=relation_stage, limit=3
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 获取自我行为风格提示失败: {exc}")
            return []

    def _build_reply_style_context(
        self, relation_snapshot: Optional[Dict[str, Any]] = None
    ) -> str:
        snapshot = self._resolve_relation_view(relation_snapshot)
        relation_label = (
            str(snapshot.get("custom_label") or "普通").strip() or "普通"
        )
        trust_value = float(snapshot.get("trust_value", 0.0) or 0.0)
        annoyance_value = float(snapshot.get("annoyance_value", 0.0) or 0.0)
        affection = float(snapshot.get("affection", 0.0) or 0.0)
        pressure = float(snapshot.get("psychological_pressure", 0.0) or 0.0)
        trauma = float(snapshot.get("trauma_score", 0.0) or 0.0)
        self_style_hints = self._get_self_behavior_style_hints(snapshot)

        response_mode = "normal"
        if trauma >= 5:
            response_mode = "trauma_defense"
        elif annoyance_value >= 60:
            response_mode = "cold_rejection"
        elif annoyance_value >= 30:
            response_mode = "irritated"

        psycho_lines = []
        if affection >= 50:
            psycho_lines.append(f"好感度高({affection:.0f})，语气温暖")
        elif affection >= 10:
            psycho_lines.append(f"有一定好感({affection:.0f})，态度友善")
        elif affection <= -30:
            psycho_lines.append(f"好感度低({affection:.0f})，态度冷淡")
        if annoyance_value > 50:
            psycho_lines.append(f"烦躁度{annoyance_value:.0f}，回复简短不耐烦")
        elif annoyance_value > 30:
            psycho_lines.append("轻微不耐烦")
        if trauma > 5:
            psycho_lines.append("心理状态不好，可能回避某些话题")
        elif trauma > 3:
            psycho_lines.append("内心有些不安")

        target_uid = str(getattr(self, "_last_user_id", "") or "").strip()
        emo_desc = []
        mood_val = attitude_val = ""
        intimacy_val = float(snapshot.get("intimacy", 0.0) or 0.0)
        surface_mask_val = float(snapshot.get("surface_mask", 0.0) or 0.0)
        inner_chaos_val = float(snapshot.get("inner_chaos", 0.0) or 0.0)
        negative_sum = 0.0
        stamina_val = 80.0
        if target_uid:
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import (
                    get_emotion_tracker, )

                tracker = get_emotion_tracker(self.stream_id)
                state = tracker.get_user_state(
                    target_uid, create_if_missing=False
                )
                if state:
                    mood_val = getattr(state, "mood", "") or ""
                    attitude_val = getattr(state, "attitude", "") or ""
                    for cn, attr in [
                        ("愤怒", "anger"),
                        ("厌恶", "disgust"),
                        ("恐惧", "fear"),
                        ("悲伤", "sadness"),
                        ("羞耻", "shame"),
                        ("内疚", "guilt"),
                    ]:
                        v = getattr(state, attr, 0.0)
                        negative_sum += v
                        if v > 0.35:
                            emo_desc.append(f"强烈{cn}")
                        elif v > 0.18:
                            emo_desc.append(f"轻微{cn}")
                    for cn, attr in [
                        ("开心", "joy"),
                        ("惊讶", "surprise"),
                        ("骄傲", "pride"),
                        ("感恩", "gratitude"),
                    ]:
                        v = getattr(state, attr, 0.0)
                        if v > 0.35:
                            emo_desc.append(f"强烈{cn}")
                        elif v > 0.18:
                            emo_desc.append(f"轻微{cn}")
                    intimacy_val = getattr(state, "intimacy", 0.0)
                    surface_mask_val = getattr(state, "surface_mask", 0.0)
                    inner_chaos_val = getattr(state, "inner_chaos", 0.0)
                    _neg_attrs = sum(
                        max(0, getattr(state, a, 0.0))
                        for a in (
                            "anger",
                            "sadness",
                            "disgust",
                            "fear",
                            "shame",
                            "guilt",
                        )
                    )
                    _pos_attrs = sum(
                        max(0, getattr(state, a, 0.0))
                        for a in ("joy", "gratitude", "pride", "surprise")
                    )
                    if _neg_attrs + _pos_attrs > 0:
                        stamina_val = max(
                            10.0,
                            100.0
                            * (_pos_attrs / (_neg_attrs + _pos_attrs + 1.0)),
                        )
                    else:
                        stamina_val = 75.0
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

        feeling_parts = []
        if mood_val and mood_val != "平静":
            feeling_parts.append(f"心情:{mood_val}")
        if emo_desc:
            feeling_parts.append(f"情绪:{'、'.join(emo_desc)}")
        if attitude_val and attitude_val != "中立":
            feeling_parts.append(f"对对方的态度:{attitude_val}")
        if intimacy_val > 10:
            feeling_parts.append(f"亲密度较高({intimacy_val:.0f})")
        if pressure > 20:
            feeling_parts.append(f"心理压力较大({pressure:.0f})")
        elif pressure > 5:
            feeling_parts.append("有一些心理压力")
        if surface_mask_val > 30:
            feeling_parts.append("正在努力维持表面冷静")
        if inner_chaos_val > 20:
            feeling_parts.append("内心非常混乱和矛盾")
        elif inner_chaos_val > 5:
            feeling_parts.append("内心有些纠结")
        if stamina_val < 20:
            feeling_parts.append("精力几乎耗尽，非常疲惫，只想用最短的话回复")
        elif stamina_val < 40:
            feeling_parts.append("比较疲惫，不太想说太多话")
        elif stamina_val < 60:
            feeling_parts.append("有些疲倦")
        if not feeling_parts:
            if annoyance_value > 50:
                feeling_parts.append("感到非常烦躁和不耐烦")
            elif trauma > 5:
                feeling_parts.append("内心非常痛苦和不安")
            elif affection > 50:
                feeling_parts.append("心情很好，想要热情回复")
            elif affection < -30:
                feeling_parts.append("不太想理这个人")

        lines = [
            f"[关系阶段] {relation_label}",
            f"[关系信号] 信任={
                trust_value:.1f} 厌烦={
                annoyance_value:.1f} 好感={
                affection:.1f}",
        ]
        if psycho_lines:
            lines.append(f"[当前心理状态] {'；'.join(psycho_lines)}。")
        if feeling_parts:
            my_feeling = "；".join(feeling_parts)
            lines.append(f"[当前情感状态] {my_feeling}")

        if response_mode == "cold_rejection":
            lines.append(
                "[★冷拒模式★] 你现在很烦对方。必须体现这种烦躁。可以：①只回半个词 ②反问句 ③直接戳破对方在刷屏 ④阴阳怪气。绝对不要客气，不要解释，不要完整句子。"
            )
        elif response_mode == "irritated":
            lines.append(
                "[★烦躁模式★] 你有点不耐烦了。回复要带刺、省略、反问。别装热情友好。"
            )
        elif response_mode == "trauma_defense":
            lines.append(
                "[防御模式] 你处于创伤防御态，回复带距离感，不主动展开。"
            )
        else:
            lines.append(
                "[灵魂指令] 回复必须体现上述情感状态，保持人格一致性。不要突然切换人设。"
            )

        if pressure >= 35:
            lines.append("[表达节奏] 句子别太满，尽量短一点松一点。")
        if trauma >= 35:
            lines.append("[情绪保护] 遇到刺耳内容先收一下。")

        if target_uid:
            try:
                from src.modules.modcore.dynamic_persona.emotion_tracker import (
                    get_emotion_tracker, )

                resp_mode = get_emotion_tracker(
                    self.stream_id
                ).get_layered_response_mode(target_uid)
                tone = str(resp_mode.get("tone", "neutral") or "neutral")
                response_length = str(
                    resp_mode.get("response_length", "normal") or "normal"
                )
                playfulness = float(resp_mode.get("playfulness", 0.5) or 0.5)
                lines.append(
                    f"[分层回复模式] 语气={tone} 长度={response_length} 活泼度={
                        playfulness:.2f}"
                )
                if tone == "guarded":
                    lines.append(
                        "[回复边界] 保持警惕、简短、少解释，不主动示好。"
                    )
                elif tone == "calm":
                    lines.append("[回复边界] 平静收束，短句，少情绪外放。")
                elif tone == "gentle":
                    lines.append("[回复边界] 语气放轻，避免刺激表达。")
                elif tone == "warm":
                    lines.append("[回复边界] 自然放松，允许轻微熟络感。")
                elif tone == "neutral":
                    lines.append(
                        "[回复边界] 正常说话就行，不要过度客气、不要用敬语、不要刻意表现礼貌。像跟普通网友聊天一样随意。"
                    )
                if response_length == "concise":
                    lines.append("[表达长度] 优先短句，点到就停。")
                avoid_patterns = resp_mode.get("avoid_patterns", []) or []
                if avoid_patterns:
                    lines.append(
                        f"[避免模式] {'、'.join(str(item) for item in avoid_patterns[:4] if str(item).strip())}"
                    )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

        if self_style_hints:
            lines.append("[自我风格学习]")
            lines.extend(f"- {hint}" for hint in self_style_hints if hint)

        _night_phase = getattr(self, "_cached_night_phase", None)
        _night_sup = getattr(self, "_night_reply_suppression", 0.0)
        if _night_phase is not None:
            from src.core.night_cycle_system import NightPhase

            if _night_phase == NightPhase.DEEP_SLEEP:
                lines.append(
                    "[★夜间深睡★] 你非常困了，几乎睁不开眼。如果非要回，只能回1-2个字或者表情。大部分时候你应该不想回。"
                )
            elif _night_phase == NightPhase.LIGHT_SLEEP:
                lines.append(
                    "[★夜间浅睡★] 你半梦半醒。回复要短、慢、可能带点迷糊感。可以不回。"
                )
            elif _night_phase == NightPhase.BURNED_OUT:
                lines.append(
                    "[★熬穿状态★] 熬了一整夜，身体和脑子都透支了。极不情愿说话，即使回了也是敷衍到极致。"
                )
            elif _night_phase == NightPhase.NIGHT_ACTIVE:
                lines.append("[时段调节] 深夜活跃期，稍微放松但不过于亢奋。")
            elif _night_phase == NightPhase.SOCIAL_NIGHT:
                lines.append("[时段调节] 社交夜模式，语气随意但保持克制。")
            elif _night_phase == NightPhase.MIDNIGHT_REFLECT:
                lines.append(
                    "[时段调节] 午夜反思态，偏短偏静，减少感叹号和情绪词。"
                )
            elif _night_phase == NightPhase.QUIET_CONTEMPLATE:
                lines.append("[时段调节] 安静沉思态，短句、少标点、不展开。")
            elif _night_phase == NightPhase.DAWN_RECOVER:
                lines.append(
                    "[时段调节] 黎明恢复中，表达慵懒自然，允许轻微迷糊感但不影响理解。"
                )
            elif _night_phase in (NightPhase.DROWSY,):
                lines.append(
                    "[时段调节] 有点困了，回复简短，少用长句和复杂结构。"
                )
        # F20：聊天值→回复风格动态映射
        try:
            from src.chat.heart_flow.energy_manager import get_vitality_pool

            _cv = float(
                get_vitality_pool().capture_snapshot(self.stream_id).chat_value
                or 50.0
            )
            if _cv >= 55:
                lines.append(
                    "[精力状态] 精力充沛，可以正常展开回复，允许适当发挥。"
                )
            elif _cv >= 35:
                lines.append("[精力状态] 精力一般，回复适度收敛，句子别太长。")
            elif _cv >= 18:
                lines.append(
                    "[精力状态] 有点累了，回复尽量短句、少展开、不啰嗦。"
                )
            else:
                lines.append(
                    "[精力状态] 很疲惫了，能回就回一句短的，不想回也可以不回。"
                )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return "\n".join(line for line in lines if line)

    async def _capture_reply_behavior_learning(
        self,
        reply_text: str,
        reply_reason: str,
        relation_snapshot: Optional[Dict[str, Any]] = None,
        was_proactive: bool = False,
    ) -> None:
        if not reply_text.strip():
            return
        try:
            from src.modules.recall.self_behavior_learner import (
                get_self_behavior_learner,
            )

            learner = get_self_behavior_learner()
            snapshot = self._resolve_relation_view(relation_snapshot)
            relation_stage = str(
                snapshot.get("custom_label")
                or snapshot.get("relationship_level")
                or "unknown"
            )
            await learner.capture_event(
                stream_id=self.stream_id,
                action_type="proactive_reply" if was_proactive else "reply",
                content=reply_text,
                result="success",
                context={
                    "relation_stage": relation_stage,
                    "reply_reason": reply_reason[:160],
                    "was_proactive": was_proactive,
                },
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 回灌自我行为学习失败: {exc}")

    def _log_relation_metrics(self, metrics: Dict[str, Any]) -> None:
        """输出后台公式算法结果（仅输出有意义的非零字段，避免全零刷屏）"""
        label = metrics.get("custom_label", "") or self._get_level_description(
            metrics.get("relationship_level", 2)
        )
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
            from src.chat.heart_flow.energy_manager import (
                get_vitality_pool,
                get_shared_resource_manager,
            )

            pool = get_vitality_pool()
            snap = pool.capture_snapshot(self.stream_id)
            shared_mgr = get_shared_resource_manager()
            shared_vals = shared_mgr.get_shared_values(self.stream_id)
            relation_view = self._resolve_relation_view()
            chat_energy = float(shared_vals.get("chat_value") or 0.0)
            thinking_energy = snap.thinking_ratio() * 100.0
            social_value = float(shared_vals.get("social_value") or 0.0)
            trust_value = float(relation_view.get("trust_value", 0.0) or 0.0)
            annoyance_value = float(
                relation_view.get("annoyance_value", 0.0) or 0.0
            )
            freq_ctrl = (
                frequency_control_manager.get_or_create_frequency_control(
                    self.stream_id
                )
            )
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

    def _evaluate_impulse_factors(
        self, now: float, relation_result: Dict
    ) -> Dict[str, Any]:
        """评估冲动因子：无聊、孤独、沉默、好感度等（已废弃，使用情感驱动核心）"""
        return {
            "silence_sec": 0.0,
            "matched_factors": [],
            "total_bonus": 0.0,
            "should_proactive": False,
        }

    def _log_impulse_factors(self, impulse: Dict[str, Any]) -> None:
        """输出冲动因子日志（已废弃，使用情感驱动核心）"""
        pass

    def _update_emotion_state(
        self, now: float, relation_result: Dict
    ) -> Dict[str, Any]:
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

                silence_sec = get_quiet_monitor().measure_silence_sec(
                    self.stream_id
                )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            # 获取能量状态
            energy = 1.0
            mood = 0.5
            try:
                from src.chat.heart_flow.energy_manager import (
                    get_vitality_pool,
                    get_shared_resource_manager,
                )

                snap = get_vitality_pool().capture_snapshot(self.stream_id)
                shared_values = (
                    get_shared_resource_manager().get_shared_values(
                        self.stream_id
                    )
                )
                chat_factor = max(
                    0.0,
                    min(
                        1.0,
                        float(shared_values.get("chat_value") or 0.0)
                        / max(snap.chat_ceiling, 1.0),
                    ),
                )
                thinking_factor = snap.thinking_ratio()
                activity_factor = max(
                    0.0,
                    min(
                        1.0,
                        float(shared_values.get("activity_level") or 0.0)
                        / 100.0,
                    ),
                )
                social_value = float(shared_values.get("social_value") or 0.0)
                social_factor = max(
                    0.0, min(1.0, (social_value + 100.0) / 200.0)
                )
                energy = max(
                    0.05,
                    min(
                        1.0,
                        thinking_factor * 0.4
                        + chat_factor * 0.25
                        + activity_factor * 0.2
                        + social_factor * 0.15,
                    ),
                )
                mood = 0.5 + social_value / 200.0
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            # 整合外部状态
            emotion_core.integrate_external_state(
                self.stream_id,
                mood=max(0.0, min(1.0, mood)),
                energy=energy,
                curiosity=0.3,
            )
            # 情感时钟滴答
            state = emotion_core.tick(
                self.stream_id,
                silence_sec,
                unanswered=self._unanswered_bot_turns,
            )
            snapshot = emotion_core.get_state_snapshot(self.stream_id)
            # 输出情感状态日志
            if snapshot["proactive_willingness"] >= 0.3:
                logger.info(
                    f"{self.log_prefix} 💕 "
                    f"无聊={snapshot['boredom']:.2f} "
                    f"孤独={snapshot['loneliness']:.2f} "
                    f"社交欲望={snapshot['social_desire']:.2f} "
                    f"主动意愿={snapshot['proactive_willingness']:.2f} "
                    f"感受: {snapshot['feeling']}"
                )
            return snapshot
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 情感状态更新失败: {exc}")
            return {}

    def _evaluate_proactive_arbiter(
        self, silence_sec: float
    ) -> "ArbiterVerdict":
        """采集多维信号并交由统一决策器融合评分。优先使用整合中心。"""
        from src.chat.proactive.proactive_arbiter import (
            get_proactive_arbiter,
            SignalBundle,
            ArbiterVerdict,
        )
        try:
            from src.chat.proactive.proactive_integration_hub import (
                get_proactive_integration_hub,
            )
            hub = get_proactive_integration_hub()
            self_state = {}
            if self._cached_self_state:
                self_state = {
                    "mood": getattr(self._cached_self_state, "mood", 0.5),
                    "trauma": getattr(self._cached_self_state, "trauma_score", 0),
                    "fatigue": getattr(self._cached_self_state, "mental_fatigue", 0),
                }
            relation_state = getattr(self, "_last_relation_snapshot", {}) or {}
            raw_messages = []
            if hasattr(self, "_last_raw_messages"):
                raw_messages = self._last_raw_messages or []
            verdict, integrated_state = hub.evaluate_proactive(
                channel_id=self.stream_id,
                raw_messages=raw_messages,
                self_state=self_state,
                relation_state=relation_state,
            )
            if verdict and verdict.should_proceed:
                logger.info(
                    f"{self.log_prefix} 整合中心决策: 主动行为 | "
                    f"融合分={verdict.fused_score:.3f} | "
                    f"理由={verdict.rationale}"
                )
            return verdict
        except Exception as hub_exc:
            logger.debug(f"{self.log_prefix} 整合中心决策失败，回退到手动采集: {hub_exc}")
        bundle = SignalBundle(
            channel_id=self.stream_id, silence_seconds=silence_sec
        )

        # 信号1: 情感就绪度
        try:
            from src.chat.heart_flow.emotion_driven_core import (
                get_emotion_driven_core,
            )

            snap = get_emotion_driven_core().get_state_snapshot(self.stream_id)
            bundle.emotional_readiness = float(
                snap.get("proactive_willingness", 0.0) or 0.0
            )
            bundle.boredom = float(snap.get("boredom", 0.0) or 0.0)
            bundle.loneliness = float(snap.get("loneliness", 0.0) or 0.0)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 决策情感信号采集失败: {exc}")

        # 信号2: 能量储备
        try:
            from src.chat.heart_flow.energy_manager import (
                get_vitality_pool,
                get_shared_resource_manager,
            )

            snap = get_vitality_pool().capture_snapshot(self.stream_id)
            shared_values = get_shared_resource_manager().get_shared_values(
                self.stream_id
            )
            chat_factor = max(
                0.0,
                min(
                    1.0,
                    float(shared_values.get("chat_value") or 0.0)
                    / max(snap.chat_ceiling, 1.0),
                ),
            )
            thinking_factor = float(snap.thinking_ratio())
            activity_factor = max(
                0.0,
                min(
                    1.0,
                    float(shared_values.get("activity_level") or 0.0) / 100.0,
                ),
            )
            bundle.vitality_ratio = max(
                0.0,
                min(
                    1.0,
                    chat_factor * 0.35
                    + thinking_factor * 0.4
                    + activity_factor * 0.25,
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
            from src.chat.heart_flow.energy_manager import (
                get_shared_resource_manager,
            )

            shared_values = get_shared_resource_manager().get_shared_values(
                self.stream_id
            )
            bundle.social_standing = float(
                shared_values.get("social_value", 0.0) or 0.0
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 决策社交信号采集失败: {exc}")

        # 信号6: 内心独白欲望
        if self._cached_voice and hasattr(
            self._cached_voice, "reply_desire_level"
        ):
            try:
                bundle.inner_voice_desire = int(
                    getattr(self._cached_voice, "reply_desire_level", 5) or 5
                )
            except Exception:
                bundle.inner_voice_desire = 5

        # 信号7: 意图池驱动强度
        try:
            from src.chat.proactive.intention_pool import get_intention_pool

            bundle.intention_drive = get_intention_pool().build_arbiter_signal(
                self.stream_id
            )
        except Exception:
            bundle.intention_drive = 0.0

        arbiter = get_proactive_arbiter()
        verdict = arbiter.evaluate(bundle)
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
                _pt_val = (
                    getattr(
                        getattr(_top_pattern, "pattern", None), "value", ""
                    )
                    or ""
                )
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
                                    _rv = (
                                        _rt.value
                                        if hasattr(_rt, "value")
                                        else str(_rt)
                                    )
                                    if _rv in (
                                        "discussed_as_topic",
                                        "quoted_reply",
                                    ):
                                        _event_relevance += float(
                                            getattr(_ref, "strength", 0.3)
                                            or 0.3
                                        )
                        _rel_snap35 = (
                            getattr(self, "_last_relation_snapshot", None)
                            or {}
                        )
                        if float(_rel_snap35.get("affection", 0) or 0) > 55:
                            _event_relevance += 0.15
                    except Exception:
                        pass
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
        _rel_snap = getattr(self, "_last_relation_snapshot", None) or {}
        if _rel_snap:
            _affection = float(_rel_snap.get("affection", 50.0) or 50.0)
            _trust = float(_rel_snap.get("trust_value", 50.0) or 50.0)
            _annoyance = float(_rel_snap.get("annoyance_value", 0.0) or 0.0)
            _aversion = float(_rel_snap.get("aversion_value", 0.0) or 0.0)
            if _affection < 15 and _trust < 20:
                _affix_penalty = 0.10
                verdict.activation_bar = min(
                    0.92, verdict.activation_bar + _affix_penalty
                )
                verdict.breakdown["relation_avoid_penalty"] = -_affix_penalty
            elif _affection > 70 and _trust > 60:
                _affix_boost = -0.08
                verdict.activation_bar = max(
                    0.08, verdict.activation_bar + _affix_boost
                )
                verdict.breakdown["relation_affinity_boost"] = _affix_boost
            if _annoyance > 35 or _aversion > 30:
                _avoid_p = min(
                    0.25, (_annoyance * 0.003) + (_aversion * 0.002)
                )
                verdict.activation_bar = min(
                    0.92, verdict.activation_bar + _avoid_p
                )
                verdict.breakdown["relation_annoyance_penalty"] = -_avoid_p
                # 高厌烦时同步压低融合分
                if _annoyance > 55:
                    _score_damp = min(0.15, (_annoyance - 55) * 0.005)
                    verdict.fused_score = max(
                        0.0, verdict.fused_score - _score_damp
                    )
                    verdict.breakdown["annoyance_score_damp"] = -_score_damp
        # F13+F30：话题归属——五类归属动态调整参与门槛
        try:
            from src.core.group_scene_state import get_group_scene

            _scene = get_group_scene(self.stream_id)
            _active_topics = _scene.active_topics(limit=3)
            _dominant_ownership = "ambiguous_topic"
            _dominant_topic_obj = None
            for _t in _active_topics:
                if not _t.is_stale(600):
                    _dominant_ownership = _t.ownership_type()
                    _dominant_topic_obj = _t
                    break
            if _dominant_ownership == "self_topic":
                _topic_boost = -0.10
                verdict.activation_bar = max(
                    0.08, verdict.activation_bar + _topic_boost
                )
                verdict.breakdown["self_topic_chase"] = _topic_boost
            elif _dominant_ownership == "target_user_topic":
                _rel_snap30 = (
                    getattr(self, "_last_relation_snapshot", None) or {}
                )
                _aff30 = float(_rel_snap30.get("affection", 50.0) or 50.0)
                if _aff30 < 25:
                    _tp_penalty = 0.08
                    verdict.activation_bar = min(
                        0.92, verdict.activation_bar + _tp_penalty
                    )
                    verdict.breakdown["target_user_low_affinity"] = (
                        -_tp_penalty
                    )
                elif _aff30 > 60:
                    _tp_boost = -0.05
                    verdict.activation_bar = max(
                        0.08, verdict.activation_bar + _tp_boost
                    )
                    verdict.breakdown["target_user_high_affinity"] = _tp_boost
            elif _dominant_ownership == "shared_group_topic":
                _gp_boost = -0.04
                verdict.activation_bar = max(
                    0.08, verdict.activation_bar + _gp_boost
                )
                verdict.breakdown["shared_group_topic_natural"] = _gp_boost
            elif _dominant_ownership == "external_topic":
                _et_penalty = 0.12
                verdict.activation_bar = min(
                    0.92, verdict.activation_bar + _et_penalty
                )
                verdict.breakdown["external_topic_low_interest"] = -_et_penalty
            elif _dominant_ownership == "ambiguous_topic":
                _am_penalty = 0.06
                verdict.activation_bar = min(
                    0.92, verdict.activation_bar + _am_penalty
                )
                verdict.breakdown["ambiguous_topic_conservative"] = (
                    -_am_penalty
                )
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
                        _rt_val = (
                            _rt.value if hasattr(_rt, "value") else str(_rt)
                        )
                        _ref_types_found.add(_rt_val)
                        if _rs > _max_ref_strength:
                            _max_ref_strength = _rs
                        if _rt_val == "discussed_as_topic":
                            _discuss_count += 1
                        elif _rt_val == "quoted_reply":
                            _quote_count += 1
                if _discuss_count >= 2 or (
                    _max_ref_strength > 0.6
                    and "discussed_as_topic" in _ref_types_found
                ):
                    _self_ref_boost = min(0.18, 0.06 + _discuss_count * 0.04)
                    verdict.activation_bar = max(
                        0.05, verdict.activation_bar - _self_ref_boost
                    )
                    verdict.fused_score = min(
                        1.0, verdict.fused_score + _self_ref_boost * 0.5
                    )
                    verdict.breakdown["self_ref_group_discussion"] = (
                        _self_ref_boost
                    )
                elif (
                    "quoted_reply" in _ref_types_found
                    and _max_ref_strength > 0.4
                ):
                    _quote_boost = 0.06
                    verdict.activation_bar = max(
                        0.08, verdict.activation_bar - _quote_boost
                    )
                    verdict.breakdown["self_ref_quoted"] = _quote_boost
                elif (
                    "direct_at" in _ref_types_found
                    or "nickname_called" in _ref_types_found
                ):
                    verdict.breakdown["self_ref_direct_ack"] = 0.02
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # F24：创伤放大负面事件 + 厌烦放大回避倾向
        try:
            _rel_snap2 = getattr(self, "_last_relation_snapshot", None) or {}
            _trauma = float(_rel_snap2.get("trauma_score", 0.0) or 0.0)
            _annoy = float(_rel_snap2.get("annoyance_value", 0.0) or 0.0)
            if _trauma > 30:
                _trauma_amp = min(0.15, (_trauma / 100.0) * 0.3)
                verdict.activation_bar = min(
                    0.92, verdict.activation_bar + _trauma_amp
                )
                verdict.breakdown["trauma_amplification"] = -_trauma_amp
            if _annoy > 30:
                _annoy_amp = min(0.18, (_annoy / 100.0) * 0.35)
                verdict.fused_score = max(
                    0.0, verdict.fused_score - _annoy_amp
                )
                verdict.breakdown["annoyance_avoidance"] = -_annoy_amp
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # F32：内心旁白→行为策略联动——旁白建议观察/防御时压制参与欲
        try:
            if self._cached_narration_plan:
                _strategy = str(
                    getattr(
                        self._cached_narration_plan, "behavior_strategy", ""
                    )
                    or ""
                )
                _narr_mood = str(
                    getattr(self._cached_narration_plan, "dominant_mood", "")
                    or ""
                )
                if _strategy == "observe_only":
                    _obs_penalty = 0.10
                    verdict.activation_bar = min(
                        0.92, verdict.activation_bar + _obs_penalty
                    )
                    verdict.breakdown["narration_observe_strategy"] = (
                        -_obs_penalty
                    )
                elif _strategy == "defensive":
                    _def_penalty = 0.14
                    verdict.activation_bar = min(
                        0.92, verdict.activation_bar + _def_penalty
                    )
                    verdict.fused_score = max(0.0, verdict.fused_score - 0.05)
                    verdict.breakdown["narration_defensive_strategy"] = (
                        -_def_penalty
                    )
                elif _strategy == "spectator_eat_melon":
                    _spec_penalty = 0.08
                    verdict.activation_bar = min(
                        0.88, verdict.activation_bar + _spec_penalty
                    )
                    verdict.breakdown["narration_spectator"] = -_spec_penalty
                elif _strategy == "engage_active":
                    _eng_boost = -0.06
                    verdict.activation_bar = max(
                        0.08, verdict.activation_bar + _eng_boost
                    )
                    verdict.breakdown["narration_engage_boost"] = _eng_boost
                if _narr_mood in ("annoyed", "tired", "overwhelmed"):
                    _mood_p = 0.05
                    verdict.activation_bar = min(
                        0.92, verdict.activation_bar + _mood_p
                    )
                    verdict.breakdown[f"narration_mood_{_narr_mood}"] = (
                        -_mood_p
                    )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # 每小时回复频率疲劳：主动发言过于频繁时提升门槛
        _hr_count = max(0, int(self._hourly_reply_count))
        if _hr_count > 4:
            _freq_fatigue = min(0.18, (_hr_count - 4) * 0.035)
            verdict.activation_bar = min(
                0.92, verdict.activation_bar + _freq_fatigue
            )
            verdict.breakdown["reply_frequency_fatigue"] = -_freq_fatigue
        ignored_turns = max(0, int(self._unanswered_bot_turns))
        if ignored_turns:
            unanswered_penalty = min(0.30, ignored_turns * 0.10)
            if _intent_override:
                unanswered_penalty *= 0.3
            verdict.fused_score = max(
                0.0, verdict.fused_score - unanswered_penalty
            )
            verdict.breakdown["unanswered_penalty"] = -unanswered_penalty

            if (
                verdict.should_proceed
                and verdict.fused_score < verdict.activation_bar
            ):
                if not _intent_override:
                    verdict.should_proceed = False
                    verdict.rationale = (
                        f"连续{ignored_turns}轮未获回应，主动欲望回撤"
                    )
                else:
                    verdict.rationale = f"{
                        verdict.rationale or ''} | 未回应轮数={ignored_turns}(意图豁免)"
            elif verdict.rationale:
                verdict.rationale = f"{
                    verdict.rationale} | 未回应轮数={ignored_turns}"
            else:
                verdict.rationale = f"未回应轮数={ignored_turns}"

        cooldown_sec = min(420.0, ignored_turns * 90.0)
        if (
            ignored_turns >= 2
            and self._last_bot_reply_ts > 0
            and (time.time() - self._last_bot_reply_ts) < cooldown_sec
        ):
            if _intent_override and bundle.intention_drive >= 0.85:
                remaining = cooldown_sec - (
                    time.time() - self._last_bot_reply_ts
                )
                verdict.breakdown["cooldown_override"] = 1.0
                verdict.rationale = f"{
                    verdict.rationale or ''}  | 冷却期被高紧迫意图突破(剩余{
                    remaining: .0f} s) "
            else:
                remaining = cooldown_sec - (
                    time.time() - self._last_bot_reply_ts
                )
                verdict.should_proceed = False
                verdict.breakdown["recent_ignore_cooldown"] = -1.0
                verdict.rationale = (
                    f"连续{ignored_turns}轮未回应，进入冷却 {remaining:.0f}s"
                )

        # SOC-03: 负面情绪影响主动发言欲望
        if verdict.should_proceed and self._cached_user_negative_emotion > 50:
            _neg_penalty = min(0.3, self._cached_user_negative_emotion / 200.0)
            if _intent_override:
                _neg_penalty *= 0.35
            verdict.fused_score = max(0.0, verdict.fused_score - _neg_penalty)
            verdict.breakdown["negative_emotion_suppress"] = -_neg_penalty
            if (
                verdict.fused_score < verdict.activation_bar
                and not _intent_override
            ):
                verdict.should_proceed = False
                verdict.rationale = f"用户负面情绪高({
                    self._cached_user_negative_emotion:.0f})，抑制主动发言"
        # F34：多模态预算联动——刷图风暴/大量跳过时进入节流模式
        try:
            if self._cached_multimodal_summary:
                _total_media = int(
                    self._cached_multimodal_summary.get("total_items", 0) or 0
                )
                _skipped = int(
                    self._cached_multimodal_summary.get("skipped_count", 0)
                    or 0
                )
                _cached_used = int(
                    self._cached_multimodal_summary.get("cache_hit_count", 0)
                    or 0
                )
                _is_storm = (
                    str(
                        self._cached_multimodal_summary.get("storm_mode", "")
                        or ""
                    )
                    == "active"
                )
                if _total_media >= 5 and _skipped >= 3:
                    _media_throttle = 0.08
                    verdict.activation_bar = min(
                        0.92, verdict.activation_bar + _media_throttle
                    )
                    verdict.breakdown["media_throttle_many_skipped"] = (
                        -_media_throttle
                    )
                elif _is_storm:
                    _storm_p = 0.12
                    verdict.activation_bar = min(
                        0.92, verdict.activation_bar + _storm_p
                    )
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
                from src.chat.heart_flow.energy_manager import (
                    get_vitality_pool,
                    get_shared_resource_manager,
                )

                _esnap = get_vitality_pool().capture_snapshot(self.stream_id)
                _eshared = get_shared_resource_manager().get_shared_values(
                    self.stream_id
                )
                _ecv = float(_eshared.get("chat_value") or 0.0)
                _etv = float(getattr(_esnap, "thinking_energy", 50.0) or 50.0)
                _eav = float(_eshared.get("activity_level") or 50.0)
                _esv = float(_eshared.get("social_value") or 0.0)
                _e_cc = max(
                    float(getattr(_esnap, "chat_ceiling", 100.0) or 100.0), 1.0
                )
                _e_tc = max(
                    float(getattr(_esnap, "thinking_ceiling", 100.0) or 100.0),
                    1.0,
                )
                _inp.energy_chat_ratio = max(0.0, min(1.0, _ecv / _e_cc))
                _inp.energy_think_ratio = max(0.0, min(1.0, _etv / _e_tc))
                _inp.energy_activity_ratio = max(0.0, min(1.0, _eav / 100.0))
                _inp.energy_social_ratio = max(
                    0.0, min(1.0, (_esv + 100.0) / 200.0)
                )
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
                from src.core.night_cycle_system import NightPhase

                _inp.night_phase = self._cached_night_phase.value
                try:
                    from src.core.night_cycle_system import get_night_cycle

                    _ncs = get_night_cycle(self.stream_id)
                    _nds = (
                        getattr(_ncs._state, "sleep_debt", 0.0)
                        if hasattr(_ncs, "_state")
                        else 0.0
                    )
                    _inp.sleep_debt = float(_nds)
                    _inp.sleep_peek_budget = (
                        _ncs.evaluate_sleep_peek()
                        if self._cached_night_phase == NightPhase.LIGHT_SLEEP
                        else True
                    )
                    _inp.sleep_reply_budget = (
                        _ncs.evaluate_sleep_reply_budget()
                    )
                    # GAP-A：熬夜压力与亢奋链数据
                    try:
                        _pressure_info = (
                            _ncs.get_overnight_pressure_breakdown()
                        )
                        _inp.overnight_pressure = float(
                            _pressure_info.get("total", 0.0)
                        )
                        _arousal_info = _ncs.get_night_expression_profile()
                        _inp.arousal_chain_state = str(
                            _arousal_info.get("current_state", "normal")
                        )
                        _inp.is_burnthrough = bool(
                            _arousal_info.get("active_template")
                            == "burnthrough"
                        )
                        _raw_expr = (
                            _arousal_info.get("raw_expression", {}) or {}
                        )
                        _inp.burnthrough_impulse = float(
                            _raw_expr.get("impulse", 0.0)
                        )
                    except Exception:
                        pass
                except Exception:
                    pass
            if self._cached_presence_state:
                _inp.social_willingness = float(
                    getattr(
                        self._cached_presence_state, "social_willingness", 0.5
                    )
                    or 0.5
                )
                _inp.watch_willingness = float(
                    getattr(
                        self._cached_presence_state, "watch_willingness", 0.5
                    )
                    or 0.5
                )
                _inp.avoidance_tendency = float(
                    getattr(
                        self._cached_presence_state, "avoidance_tendency", 0.0
                    )
                    or 0.0
                )
                _inp.quiet_preference = float(
                    getattr(
                        self._cached_presence_state, "quiet_preference", 0.2
                    )
                    or 0.2
                )
                _inp.curiosity_level = float(
                    getattr(
                        self._cached_presence_state, "curiosity_level", 0.3
                    )
                    or 0.3
                )
                _inp.outward_attention = float(
                    getattr(
                        self._cached_presence_state, "outward_attention", 0.5
                    )
                    or 0.5
                )
            _rel_mf = getattr(self, "_last_relation_snapshot", None) or {}
            _inp.affection = float(_rel_mf.get("affection", 50.0) or 50.0)
            _inp.trust_value = float(_rel_mf.get("trust_value", 50.0) or 50.0)
            _inp.annoyance_value = float(
                _rel_mf.get("annoyance_value", 0.0) or 0.0
            )
            _inp.trauma_score = float(_rel_mf.get("trauma_score", 0.0) or 0.0)
            _inp.aversion_value = float(
                _rel_mf.get("aversion_value", 0.0) or 0.0
            )
            try:
                from src.core.group_scene_state import get_group_scene

                _scene_mf = get_group_scene(self.stream_id)
                _topics_mf = _scene_mf.active_topics(limit=3)
                for _t in _topics_mf:
                    if not _t.is_stale(600):
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
                        (
                            _pt_enum.value
                            if hasattr(_pt_enum, "value")
                            else str(_pt_enum)
                        )
                        if _pt_enum
                        else ""
                    )
                    _inp.pattern_confidence = _top_pc
                    _inp.is_spectator_mode = (
                        _inp.dominant_pattern == "spectator_mode"
                    )
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
                _inp.direct_at = (
                    "direct_at" in _ref_types_set
                    or "nickname_called" in _ref_types_set
                )
                if _inp.is_spectator_mode:
                    _event_rel = 0.0
                    for _ref in self._cached_self_references:
                        _rt2 = getattr(_ref, "ref_type", None)
                        if _rt2:
                            _rv2 = (
                                _rt2.value
                                if hasattr(_rt2, "value")
                                else str(_rt2)
                            )
                            if _rv2 in ("discussed_as_topic", "quoted_reply"):
                                _event_rel += float(
                                    getattr(_ref, "strength", 0.3) or 0.3
                                )
                    if _inp.affection > 55:
                        _event_rel += 0.15
                    _inp.event_relevance = _event_rel
            if self._cached_narration_plan:
                _inp.narration_strategy = str(
                    getattr(
                        self._cached_narration_plan, "behavior_strategy", ""
                    )
                    or ""
                )
                _inp.narration_mood = str(
                    getattr(self._cached_narration_plan, "dominant_mood", "")
                    or ""
                )
            if self._cached_multimodal_summary:
                _inp.media_total_items = int(
                    self._cached_multimodal_summary.get("total_items", 0) or 0
                )
                _inp.media_skipped_count = int(
                    self._cached_multimodal_summary.get("skipped_count", 0)
                    or 0
                )
                _inp.media_storm_active = (
                    str(
                        self._cached_multimodal_summary.get("storm_mode", "")
                        or ""
                    )
                    == "active"
                )
            if self._cached_metabolism_state:
                _inp.consecutive_active_minutes = float(
                    getattr(
                        self._cached_metabolism_state,
                        "consecutive_active_minutes",
                        0.0,
                    )
                    or 0.0
                )
            _inp.hourly_reply_used = self._hourly_reply_count
            _inp.intention_drive = bundle.intention_drive
            _inp.intent_override_candidate = (
                _intent_override and bundle.intention_drive >= 0.78
            )
            try:
                from src.core.state_coupling_matrix import get_coupling_engine

                _scme = get_coupling_engine(self.stream_id)
                _scme_snap = _scme.build_full_snapshot(
                    metabolism=getattr(self, "_cached_metabolism_state", None),
                    presence=getattr(self, "_cached_presence_state", None),
                    relation=self._last_relation_snapshot,
                    night_phase=getattr(self, "_cached_night_phase", None),
                    group_scene=None,
                    watch_state=str(
                        getattr(self, "_cached_watch_level", "") or ""
                    ),
                )
                if _scme_snap:
                    _inp.coupling_boredom_mod = float(
                        _scme_snap.get("boredom_to_loafing_mod", 0.0) or 0.0
                    )
                    _inp.coupling_activity_mod = float(
                        _scme_snap.get("activity_behavior_mod", 0.0) or 0.0
                    )
                    _inp.coupling_style_tier = str(
                        _scme_snap.get("style_tier", "") or ""
                    )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            try:
                from src.core.individual_impression_engine import (
                    get_impression_engine,
                )

                _iie = get_impression_engine()
                _focus_uid = str(getattr(bundle, "anchor_user_id", "") or "")
                if _focus_uid:
                    _imp = _iie.get_impression(_focus_uid)
                    if _imp:
                        _inp.impression_narrative_type = (
                            _imp.narrative_type.value
                        )
                        _inp.impression_affection_proxy = (
                            _imp.current_affection
                        )
                        _inp.impression_trust_proxy = _imp.trust_level
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            try:
                from src.core.skill_lifecycle_manager import (
                    get_skill_lifecycle_manager,
                )

                _slm = get_skill_lifecycle_manager()
                _active_skills = _slm.get_active_skills(limit=5)
                if _active_skills:
                    _avg_cost = sum(
                        s.phase_state.cost_modifier for s in _active_skills
                    ) / max(1, len(_active_skills))
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
                    _inp.mm_avg_valence = float(
                        _mm_cached.get("avg_valence", 0.0) or 0.0
                    )
                    _inp.mm_engagement = float(
                        _mm_cached.get("avg_engagement", 0.0) or 0.0
                    )
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
                    verdict.activation_bar = min(
                        0.95, verdict.activation_bar + 0.25
                    )
                    verdict.rationale = f"{
                        verdict.rationale or ''} | 仪表盘L1阻断({
                        _dv.get(
                            'reason', '')[
                            :40]})"
                elif not _dv_reply and _dv_urgency in ("不回复", "跳过"):
                    _skip_penalty = 0.08 * _dv_confidence
                    verdict.activation_bar = min(
                        0.90, verdict.activation_bar + _skip_penalty
                    )
                    verdict.fused_score = max(
                        0.0, verdict.fused_score - _skip_penalty * 0.4
                    )
                    verdict.breakdown["dashboard_l2_skip"] = -_skip_penalty
                    logger.debug(
                        f"{self.log_prefix} 仪表盘L2跳过: urgency={_dv_urgency} path={_dv_path}"
                    )
                elif _dv_reply and _dv_urgency in ("立即回复", "尽快回复"):
                    _reply_boost = 0.05 * _dv_confidence
                    verdict.activation_bar = max(
                        0.05, verdict.activation_bar - _reply_boost
                    )
                    verdict.breakdown["dashboard_l2_boost"] = _reply_boost
                _dv_tone = str(_dv.get("tone", "") or "")
                if _dv_tone:
                    verdict.rationale = f"{
                        verdict.rationale or ''} | 语气建议={_dv_tone}"
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            _mf_verdict = _mf.evaluate(_inp)
            _mf_verdict.apply_to_arbiter_verdict(verdict)
        except Exception as exc:
            logger.debug(
                f"{self.log_prefix} 多因子融合引擎异常，回退到原有逻辑: {exc}"
            )
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
                _is_late_night = _night_r.value in (
                    "deep_sleep",
                    "light_sleep",
                    "drowsy",
                )
            _ctx_factors = ContextualDangerFactors(
                is_late_night=_is_late_night,
                bot_is_discussing_sensitive_topic=bool(
                    self._cached_ritual_behavior
                ),
                recent_negative_emotion_spike=self._cached_user_negative_emotion
                > 50,
                message_contains_url=False,
            )
            _safety_result = _sfe.assess(
                [],
                contextual_factors=_ctx_factors,
                current_arbiter_activation_bar=getattr(
                    verdict, "activation_bar", 0.5
                ),
                current_arbiter_fused_score=getattr(
                    verdict, "fused_score", 0.5
                ),
            )
            _new_bar, _new_score = _sfe.apply_to_verdict(
                verdict, _safety_result
            )
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

    def _check_proactive_from_emotion(self) -> Tuple[bool, str]:
        """
        从情感状态判断是否应该主动行为

        这不是规则判断，而是"感受"判断：
        - 我感到无聊吗？
        - 我感到孤独吗？
        - 我想找人聊天吗？
        """
        try:
            from src.chat.heart_flow.emotion_driven_core import (
                get_emotion_driven_core,
            )

            emotion_core = get_emotion_driven_core()
            return emotion_core.should_proactive(self.stream_id)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 情感判断失败: {exc}")
            return False, "情感系统不可用"

    def _on_bot_sent(self, was_proactive: bool = False) -> None:
        """通知各系统：机器人发言了"""
        self._unanswered_bot_turns += 1
        self._last_bot_reply_ts = time.time()
        self._legacy_constraint_hits = 0
        # F27：递增每小时回复计数
        _now_hr = time.time()
        if (
            self._hourly_window_start == 0.0
            or (_now_hr - self._hourly_window_start) >= 3600.0
        ):
            self._hourly_window_start = _now_hr
            self._hourly_reply_count = 0
        self._hourly_reply_count += 1
        # F28：消耗睡眠回复配额
        try:
            from src.core.night_cycle_system import get_night_cycle

            get_night_cycle(self.stream_id).consume_sleep_reply()
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.emotion_driven_core import (
                get_emotion_driven_core,
            )

            emotion_core = get_emotion_driven_core()
            emotion_core.on_bot_message(self.stream_id)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.trauma_fabric import get_wound_network

            wound = get_wound_network()
            wound.absorb_stimulus(self.stream_id, "bot_reply", 0.5)
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.modules.modcore.psychological_core import (
                get_psychological_core,
            )

            psycho = get_psychological_core()
            psycho.update_stamina(self.stream_id, "", 2.0, "reply")
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.modules.social_value.social_affect_fuser import (
                get_social_affect_fuser,
            )

            if self._last_user_id:
                get_social_affect_fuser().record_peer_interaction(
                    self._last_user_id, self.stream_id, "bot", 0.5
                )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.heartflow import heartflow

            heartflow.note_bot_activity(
                self.stream_id, was_proactive=was_proactive
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # 主动行为反馈登记：开启观察窗口等待用户回复
        if was_proactive:
            try:
                from src.chat.proactive.proactive_arbiter import (
                    get_proactive_arbiter,
                )

                _proactive_intent_id = str(
                    getattr(self, "_last_proactive_intent_id", "") or ""
                )
                get_proactive_arbiter().ledger.register_proactive_fire(
                    self.stream_id,
                    source_intent_id=_proactive_intent_id,
                )
                if _proactive_intent_id:
                    logger.info(
                        f"{self.log_prefix} [闭环追踪] 主动行为已登记 "
                        f"intent_id={_proactive_intent_id[:10]}"
                    )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        # GAP-N：新鲜度引擎——发言后标记相关消息为"看见并关注"
        try:
            from src.core.freshness_decay_engine import (
                get_freshness_decay_engine,
                VisibilityHistoryEffect,
            )

            if not self._freshness_engine_initialized:
                self._freshness_engine_initialized = True
            _fde = get_freshness_decay_engine(self.stream_id)
            if self._cached_deep_visibility_results:
                for _dvr in self._cached_deep_visibility_results[-5:]:
                    _mid = getattr(_dvr, "message_id", "")
                    if _mid and hasattr(_fde, "update_visibility_history"):
                        _fde.update_visibility_history(
                            _mid, VisibilityHistoryEffect.SEEN_AND_NOTICED
                        )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        # GAP-P：自适应更新管道——收集发言事件
        try:
            from src.core.adaptive_update_pipeline import get_adaptive_pipeline

            _ap = get_adaptive_pipeline()
            _ap.collect_event(
                event_type="bot_reply",
                raw_content=f"channel={self.stream_id[:16]}, proactive={was_proactive}",
                source_channel=self.stream_id,
                metadata={"hourly_count": self._hourly_reply_count},
                priority=2,
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _register_bot_message_record(
        self,
        reply_text: str,
        loop_info: Any,
        target_message: Any,
        was_proactive: bool,
    ) -> None:
        message_id = str(
            getattr(loop_info, "message_id", "")
            or getattr(loop_info, "reply_message_id", "")
            or getattr(loop_info, "sent_message_id", "")
            or ""
        ).strip()
        if not message_id:
            return

        response_to = str(
            getattr(target_message, "message_id", "") or ""
        ).strip()
        try:
            from src.core.self_reply_recognizer import (
                MessageSource,
                get_self_reply_recognizer,
            )

            recognizer = get_self_reply_recognizer()
            recognizer.register_bot_user("bot")
            recognizer.mark_bot_message(
                message_id=message_id,
                content=reply_text,
                source=(
                    MessageSource.BOT_PROACTIVE
                    if was_proactive
                    else MessageSource.BOT_REPLY
                ),
                response_to=response_to,
                context={"stream_id": self.stream_id},
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 注册bot消息记录失败: {exc}")

    async def _llm_autonomous_decide(
        self,
        now: float,
        relation_result: Dict,
        messages: List,
        voice_conclusion=None,
    ) -> "AutonomousDecision":
        """
        LLM 自主决策 - 让 LLM 自己感知环境并决定是否要主动发言

        这是真正的主动：不是被规则触发，而是 LLM 自己决定。
        LLM 会感知：
        - 群聊氛围
        - 当前话题
        - 沉默时长
        - 自己的感受（内心独白结果）
        然后自己决定是否要说话、说什么。
        """
        try:
            from src.chat.heart_flow.llm_autonomous_planner import (
                get_llm_autonomous_planner,
                EnvironmentSnapshot,
                AutonomousDecision,
            )

            planner = get_llm_autonomous_planner()
            # 构建环境快照，传递内心独白结果
            env = self._build_environment_snapshot(
                now, relation_result, messages, voice_conclusion
            )
            # 让 LLM 自己决定
            decision = await planner.perceive_and_decide(env)
            # 输出 LLM 的思考过程
            if decision.should_act:
                logger.info(
                    f"{self.log_prefix} 自主决策 "
                    f"感受: {decision.emotional_state}, "
                    f"意图: {decision.social_intention}, "
                    f"决定: {decision.action_type}, "
                    f"内容规划: {decision.content_plan[:60] if decision.content_plan else '无'}"
                )
            else:
                logger.debug(
                    f"{self.log_prefix} 自主决策决定观察: {decision.reasoning[:80] if decision.reasoning else '无原因'}"
                )
            return decision
        except Exception as exc:
            logger.error(f"{self.log_prefix} LLM 自主决策失败: {exc}")
            from src.chat.heart_flow.llm_autonomous_planner import (
                AutonomousDecision,
            )

            return AutonomousDecision(
                should_act=False,
                action_type="observe",
                reasoning=f"LLM 调用失败: {exc}",
            )

    def _ensure_night_cycle(self):
        if self._night_cycle is None:
            try:
                from src.core.night_cycle_system import get_night_cycle

                self._night_cycle = get_night_cycle(self.stream_id)
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
        return self._night_cycle

    def _apply_night_cycle_modulation(self, now: float) -> Optional[str]:
        _nc = self._ensure_night_cycle()
        if _nc is None:
            return None
        try:
            from src.chat.heart_flow.energy_manager import (
                get_vitality_pool_manager,
            )

            _vm = get_vitality_pool_manager(self.stream_id)
            _snap = _vm.capture_snapshot() if _vm else None
            _energy_ratio = (
                (_snap.chat_energy / max(0.01, _snap.chat_ceiling))
                if _snap
                else 1.0
            )
            _phase = _nc.evaluate(
                energy_ratio=_energy_ratio,
                activity_level=getattr(self, "_cached_activity_level", 50.0),
                boredom=float(getattr(self, "_cached_boredom", 0.0) or 0),
                social_stimulus=0.0,
                current_hour=None,
            )
            if _phase.is_sleeping:
                return f"夜间{_phase.label()}模式: 深度休眠中"
            if not _phase.can_interact:
                return f"夜间{_phase.label()}模式: 不可交互"
            if not _nc.can_reply_tonight():
                return f"夜间回复已达上限({
                    _nc.state_snapshot.night_reply_count}/{
                    _nc.state_snapshot.night_reply_cap})"
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 夜间节律评估异常: {exc}")
        return None

    def _apply_pattern_based_routing(
        self, incoming_batch, pinged_msg=None
    ) -> Optional[str]:
        """群体模式硬路由：根据检测到的群行为模式直接决定处理策略"""
        if not self._cached_pattern_evidence:
            return None
        _top = self._cached_pattern_evidence[0]
        _conf = float(getattr(_top, "confidence", 0.0) or 0.0)
        if _conf < 0.4:
            return None
        _pat_enum = getattr(_top, "pattern", None)
        if _pat_enum is None:
            return None
        from src.core.group_pattern_detector import GroupPattern

        _has_ping = pinged_msg is not None
        if _pat_enum == GroupPattern.COPYCAT_CHAIN and not _has_ping:
            self._pattern_forced_observe = True
            logger.info(
                f"{self.log_prefix} 🔄 复读跟风模式(置信{_conf:.2f})→进入围观观察"
            )
            return "group_pattern:copycat_observe"
        if _pat_enum in (
            GroupPattern.PILE_ON,
            GroupPattern.CONFLICT_ESCALATION,
        ):
            self._defense_mode_active = True
            if not _has_ping and _conf > 0.55:
                logger.info(
                    f"{self.log_prefix} ⚔️ 围攻/冲突升级(置信{_conf:.2f})→防御沉默"
                )
                return "group_pattern:defense_silence"
            logger.info(
                f"{self.log_prefix} ⚔️ 围攻/冲突检测(置信{_conf:.2f})→防御模式激活"
            )
            return None
        if _pat_enum == GroupPattern.SPECTATOR_MODE and not _has_ping:
            self._pattern_forced_observe = True
            logger.info(
                f"{self.log_prefix} 👥 围观模式(置信{_conf:.2f})→仅观察不参与"
            )
            return "group_pattern:spectator"
        if _pat_enum == GroupPattern.MEME_STORM and not _has_ping:
            _msg_count = len(incoming_batch) if incoming_batch else 0
            if _msg_count >= 8:
                self._pattern_forced_observe = True
                logger.info(
                    f"{self.log_prefix} 🌪️ 梗风暴( {_msg_count}条/置信{_conf:.2f})→围观吃瓜"
                )
                return "group_pattern:meme_observe"
        if (
            _pat_enum == GroupPattern.SUDDEN_SILENCE
            or _pat_enum == GroupPattern.QUIET_REFLECTION
        ):
            if not _has_ping:
                logger.info(
                    f"{self.log_prefix} 🔇 安静/反思模式(置信{_conf:.2f})→降低参与欲"
                )
                self._peek_mode_active = True
                return "group_pattern:quiet_reflect"
        return None

    def _apply_scene_hard_constraints(self, pinged_msg=None) -> Optional[str]:
        """群场景硬约束：suitable_to_join和atmosphere直接影响决策"""
        if not self._cached_scene_snapshot:
            return None
        _joinable = bool(
            getattr(self._cached_scene_snapshot, "suitable_to_join", True)
        )
        _has_ping = pinged_msg is not None
        if not _joinable and not _has_ping:
            _reason = getattr(
                self._cached_scene_snapshot, "join_unsuitable_reason", ""
            )
            logger.info(f"{self.log_prefix} 🚫 场景不适合插话: {_reason}")
            return f"scene_constraint:unsuitable_to_join"
        _atmo = getattr(self._cached_scene_snapshot, "atmosphere", None)
        if _atmo is not None and not _has_ping:
            from src.core.group_scene_state import AtmosphereType

            if _atmo == AtmosphereType.ARGUMENT:
                logger.info(f"{self.log_prefix} ⚖️ 争论氛围→降低主动参与")
                self._defense_mode_active = True
                return None
            if _atmo == AtmosphereType.SPAM_FLOOD:
                self._pattern_forced_observe = True
                logger.info(f"{self.log_prefix} 🌀 刷屏氛围→仅观察")
                return "scene_constraint:spam_flood"
        return None

    def _build_environment_snapshot(
        self,
        now: float,
        relation_result: Dict,
        messages: List,
        voice_conclusion=None,
    ) -> "EnvironmentSnapshot":
        """构建环境快照，供 LLM 感知 - 集成共享资源和内心独白结果"""
        from src.chat.heart_flow.llm_autonomous_planner import (
            EnvironmentSnapshot,
        )
        import datetime

        silence_sec = 0.0
        try:
            from src.chat.proactive.silence_watcher import get_quiet_monitor

            silence_sec = get_quiet_monitor().measure_silence_sec(
                self.stream_id
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        energy = 1.0
        mood = "平静"
        social_value = 0.0
        chat_value = 100.0
        activity_level = 50.0
        try:
            from src.chat.heart_flow.energy_manager import (
                get_vitality_pool,
                get_shared_resource_manager,
            )

            snap = get_vitality_pool().capture_snapshot(self.stream_id)
            thinking_factor = snap.thinking_ratio()
            shared_mgr = get_shared_resource_manager()
            shared_values = shared_mgr.get_shared_values(self.stream_id)
            chat_value = float(
                relation_result.get(
                    "chat_value", shared_values.get("chat_value") or 0.0
                )
                or 0.0
            )
            activity_level = float(
                relation_result.get(
                    "activity_level",
                    shared_values.get("activity_level") or 0.0,
                )
                or 0.0
            )
            social_value = float(
                relation_result.get(
                    "social_value",
                    relation_result.get("shared_social_value", 0.0),
                )
                or 0.0
            )
            chat_factor = max(
                0.0, min(1.0, float(chat_value) / max(snap.chat_ceiling, 1.0))
            )
            activity_factor = max(0.0, min(1.0, float(activity_level) / 100.0))
            social_factor = max(
                0.0, min(1.0, (float(social_value) + 100.0) / 200.0)
            )
            energy = max(
                0.05,
                min(
                    1.0,
                    thinking_factor * 0.4
                    + chat_factor * 0.25
                    + activity_factor * 0.2
                    + social_factor * 0.15,
                ),
            )
            pressure = float(
                relation_result.get("psychological_pressure", 0.0) or 0.0
            )
            if pressure > 70:
                mood = "紧绷"
            elif social_value > 30:
                mood = "愉快"
            elif social_value < -20:
                mood = "有点烦"
            elif chat_value < 30:
                mood = "有点累"
            elif activity_level > 70:
                mood = "活跃"
            explicit_mood = str(relation_result.get("mood", "") or "").strip()
            if explicit_mood and explicit_mood not in ("平静", "未知"):
                mood = explicit_mood
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        atmosphere = "平静"
        try:
            ambient = self._sample_channel_ambient()
            if ambient:
                atmosphere = ambient.get("category", "平静")
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        recent_messages = []
        current_topics = self._extract_current_topics(messages)
        for msg in messages[-10:]:
            content = (
                getattr(msg, "processed_plain_text", "")
                or getattr(msg, "raw_plain_text", "")
                or getattr(msg, "content", "")
                or ""
            )
            speaker = (
                getattr(msg, "user_nickname", "")
                or getattr(msg, "user_name", "")
                or "未知"
            )
            if content:
                recent_messages.append(
                    {
                        "speaker": speaker,
                        "content": content[:200],
                    }
                )
        if not recent_messages:
            logger.warning(
                f"{self.log_prefix} 📷 没有检测到最近消息，消息数量={len(messages)}"
            )
        now_dt = datetime.datetime.now()
        time_of_day = now_dt.strftime("%H:%M")
        day_of_week = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][
            now_dt.weekday()
        ]
        active_users = list(
            set(
                getattr(msg, "user_nickname", "")
                for msg in messages[-20:]
                if getattr(msg, "user_nickname", "")
            )
        )[:5]
        my_last_message = ""
        my_last_message_time = 0.0
        for msg in reversed(messages[-20:]):
            if getattr(msg, "user_id", "") == "bot":
                my_last_message = (
                    getattr(msg, "processed_plain_text", "")[:100] or ""
                )
                my_last_message_time = getattr(msg, "timestamp", 0.0)
                break
        # 提取内心独白结果
        inner_voice_desire = 5
        inner_voice_thinking = ""
        inner_voice_mood = ""
        inner_voice_primary_intent = ""
        inner_voice_needs_upgrade = False
        if voice_conclusion is not None:
            if hasattr(voice_conclusion, "reply_desire_level"):
                inner_voice_desire = voice_conclusion.reply_desire_level
            if hasattr(voice_conclusion, "thinking"):
                inner_voice_thinking = (
                    voice_conclusion.thinking[:200]
                    if voice_conclusion.thinking
                    else ""
                )
            if hasattr(voice_conclusion, "current_mood"):
                inner_voice_mood = voice_conclusion.current_mood or ""
            if hasattr(voice_conclusion, "needs_upgrade"):
                inner_voice_needs_upgrade = bool(
                    voice_conclusion.needs_upgrade
                )
            _pi = voice_conclusion.primary_intent()
            if _pi is not None:
                _ptype = getattr(_pi, "intent_type", "") or ""
                _pdesc = getattr(_pi, "description", "") or ""
                inner_voice_primary_intent = f"{_ptype}" + (
                    f"({_pdesc})" if _pdesc else ""
                )
        # 补充对话阶段与创伤压力
        _snap_dialogue_phase = ""
        _snap_trauma_pressure = 0.0
        try:
            from src.chat.proactive.session_tracker import get_memoir_cabinet

            _snap_cabinet = get_memoir_cabinet()
            _snap_user_id = self._resolve_latest_human_user_id(messages)
            _snap_memoir = (
                _snap_cabinet.retrieve(_snap_user_id)
                if hasattr(_snap_cabinet, "retrieve")
                else None
            )
            if _snap_memoir and hasattr(_snap_memoir, "phase"):
                _snap_dialogue_phase = (
                    _snap_memoir.phase.value
                    if hasattr(_snap_memoir.phase, "value")
                    else str(_snap_memoir.phase)
                )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.modules.trauma.trauma_system import get_trauma_system

            _snap_trauma = get_trauma_system().get_state()
            _snap_trauma_pressure = _snap_trauma.stress_accumulation
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        _pending_intentions_hint = ""
        try:
            from src.chat.proactive.intention_pool import get_intention_pool

            _ipool = get_intention_pool()
            _pending = _ipool.get_active_intentions(self.stream_id, limit=3)
            if _pending:
                _pending_intentions_hint = "【跨轮意图】你有以下未完成的意图：\n" + "\n".join(
                    f"- {i.kind.value}: {i.description[:50]} (紧迫度={i.effective_urgency():.2f})"
                    for i in _pending
                )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        _snap = EnvironmentSnapshot(
            channel_id=self.stream_id,
            channel_name=getattr(self.chat_stream, "stream_name", "群聊")
            or "群聊",
            silence_seconds=silence_sec,
            energy_level=energy,
            mood=mood,
            atmosphere=atmosphere,
            recent_messages=recent_messages,
            active_users=active_users,
            current_topics=current_topics,
            social_value=social_value,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            my_last_message=my_last_message,
            my_last_message_time=my_last_message_time,
            inner_voice_desire=inner_voice_desire,
            inner_voice_thinking=inner_voice_thinking,
            inner_voice_mood=inner_voice_mood,
            inner_voice_primary_intent=inner_voice_primary_intent,
            inner_voice_needs_upgrade=inner_voice_needs_upgrade,
            dialogue_phase=_snap_dialogue_phase,
            trauma_pressure=_snap_trauma_pressure,
            narration_hint=(
                self._cached_narration_plan.to_prompt_block()
                if self._cached_narration_plan
                else ""
            ),
            impression_hint="",
            pending_intentions_hint=_pending_intentions_hint,
        )
        try:
            from src.core.individual_impression_engine import (
                get_impression_engine,
            )

            _target_uid = ""
            if messages:
                for _dm in reversed(messages):
                    _duid = str(getattr(_dm, "user_id", "") or "")
                    if _duid and _duid != "bot":
                        _target_uid = _duid
                        break
            if _target_uid:
                _ie = get_impression_engine()
                _isum = _ie.get_impression_summary(_target_uid)
                if _isum.get("exists"):
                    _imp_lines = []
                    if _isum.get("nickname"):
                        _imp_lines.append(f"内心称呼: {_isum['nickname']}")
                    if (
                        _isum.get("narrative_type")
                        and _isum["narrative_type"] != "stranger"
                    ):
                        _imp_lines.append(
                            f"关系定位: {_isum['narrative_type']}"
                        )
                    if (
                        _isum.get("primary_rule")
                        and _isum["primary_rule"] != "normal"
                    ):
                        _rule_map = {
                            "warm": "热情亲近",
                            "playful": "调皮打闹",
                            "respectful": "尊重客气",
                            "cautious": "谨慎戒备",
                            "dry": "冷淡简短",
                            "avoidant": "回避疏远",
                            "teasing": "调侃戏谑",
                            "protective": "保护性",
                        }
                        _imp_lines.append(
                            f"行为策略: {
                                _rule_map.get(
                                    _isum['primary_rule'],
                                    _isum['primary_rule'])}"
                        )
                    if _isum.get("tags"):
                        _valid_tags = [
                            t for t in _isum["tags"] if t not in ("new",)
                        ]
                        if _valid_tags:
                            _imp_lines.append(
                                f"印象标签: {', '.join(_valid_tags[:5])}"
                            )
                    if _imp_lines:
                        _snap.impression_hint = "【主观印象】" + "；".join(
                            _imp_lines
                        )
        except Exception as _ierr:
            logger.debug(f"{self.log_prefix} 印象引擎异常: {_ierr}")
        return _snap

    # ═══════════════════════════════════════════════════
    #  核心系统集成方法 - 真正对接各个模块
    # ═══════════════════════════════════════════════════

    async def _check_identity_context(self, messages: List) -> Dict[str, Any]:
        """身份锚点集成 - 确认当前身份状态"""
        try:
            from src.core.identity_anchor import get_identity_anchor

            anchor = get_identity_anchor()
            user_id = ""
            if messages:
                for msg in reversed(messages):
                    uid = getattr(msg, "user_id", "")
                    if uid and uid != "bot":
                        user_id = uid
                        break
            context = anchor.get_identity(self.stream_id, user_id)
            conflict = anchor.check_conflict(self.stream_id, user_id)
            if conflict:
                logger.info(
                    f"{self.log_prefix} 🔒 检测到身份冲突: {conflict.conflict_type}"
                )
            return {
                "identity": (
                    context.current_mode.value if context else "default"
                ),
                "response_mode": (
                    context.response_mode.value if context else "normal"
                ),
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
                if getattr(msg, "user_id", "") == "bot":
                    last_bot_msg = msg
                    break
            if last_bot_msg is None:
                return {"is_self_reply": False, "similarity": 0.0}
            content = getattr(
                last_bot_msg, "processed_plain_text", ""
            ) or getattr(last_bot_msg, "content", "")
            result = recognizer.check_self_reply(
                channel_id=self.stream_id,
                content=content,
                author_id="bot",
            )
            if result.is_self_reply:
                logger.warning(
                    f"{self.log_prefix} 🔄 检测到重复回复风险: 相似度={result.similarity:.2f}"
                )
            return {
                "is_self_reply": result.is_self_reply,
                "similarity": result.similarity,
                "quality": (
                    result.quality.value
                    if hasattr(result, "quality")
                    else "unknown"
                ),
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
                f"内容状态追踪建议跳过: {
                    decision.reason}",
            )
            return result
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 内容状态判定失败: {exc}")
            return result

    def _apply_content_state_skip(
        self, content_state_signal: Optional[Dict[str, Any]]
    ) -> None:
        """把内容状态拒绝结果回写给追踪器，避免同一类低价值内容重复进入主链。"""
        if not content_state_signal:
            return
        content = str(
            content_state_signal.get("target_content", "") or ""
        ).strip()
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

    def _get_latest_human_message(self, messages: List):
        for msg in reversed(messages[-10:]):
            if getattr(msg, "user_id", "") != "bot":
                return msg
        return None

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
                recent_messages.append(
                    {
                        "sender_id": getattr(msg, "user_id", ""),
                        "timestamp": getattr(msg, "timestamp", 0.0),
                        "text": getattr(msg, "processed_plain_text", "")
                        or getattr(msg, "content", ""),
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
            return {
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
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 群聊感知失败: {exc}")
            return {
                "activity_level": "未知",
                "burst_detected": False,
                "controversy_detected": False,
            }

    async def _analyze_message_preprocessor(
        self, messages: List
    ) -> Dict[str, Any]:
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
                            getattr(msg, "message_id", "")
                            or getattr(msg, "msg_id", "")
                            or getattr(msg, "id", "")
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
                        is_at_me=bool(
                            getattr(msg, "is_at_me", False)
                            or getattr(msg, "is_at", False)
                        ),
                        has_image=bool(
                            getattr(msg, "has_image", False)
                            or getattr(msg, "image_urls", None)
                        ),
                        image_urls=list(getattr(msg, "image_urls", []) or []),
                        is_bot_self=(getattr(msg, "user_id", "") == "bot"),
                        content_type=str(
                            getattr(msg, "content_type", "text") or "text"
                        ),
                        is_quote_reply=bool(
                            getattr(msg, "is_quote_reply", False)
                        ),
                        quoted_content=str(
                            getattr(msg, "quoted_content", "") or ""
                        ),
                        quoted_sender=str(
                            getattr(msg, "quoted_sender", "") or ""
                        ),
                        is_forward=bool(getattr(msg, "is_forward", False)),
                        forward_title=str(
                            getattr(msg, "forward_title", "") or ""
                        ),
                        forward_items=list(
                            getattr(msg, "forward_items", []) or []
                        ),
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
            now = time.time()
            user_msg_count = {}
            spam_threshold = 5
            spam_window = 60.0

            for msg in messages:
                user_id = getattr(msg, "user_id", "")
                if user_id == "bot":
                    continue

                user_msg_count[user_id] = user_msg_count.get(user_id, 0) + 1

            fuser = get_social_affect_fuser()

            for msg in messages:
                user_id = getattr(msg, "user_id", "")
                if user_id == "bot":
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
                            k: v
                            for k, v in self._spam_warned_users.items()
                            if now_ts - v < 3600
                        }
                    if user_id not in self._spam_warned_users:
                        logger.info(
                            f"{self.log_prefix} 🚫 用户 {user_id[:8]} 刷屏 ({msg_count}条)，降低响应优先级"
                        )
                        self._spam_warned_users[user_id] = now_ts
                    setattr(msg, "_spam_user", True)

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
                    if any(
                        nl in dossier.custom_nick for nl in negative_labels
                    ):
                        logger.debug(
                            f"{self.log_prefix} 🚫 跳过不感兴趣用户: {user_id[:8]} ({dossier.custom_nick})"
                        )
                        continue

                filtered.append(msg)

            return filtered
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 用户过滤失败: {exc}")
            return messages

    def _get_user_attention_distribution(
        self, messages: List
    ) -> Dict[str, float]:
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
                if user_id == "bot":
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
                content = getattr(msg, "processed_plain_text", "") or getattr(
                    msg, "content", ""
                )
                if user_id and user_id != "bot" and content:
                    if user_id not in user_messages:
                        user_messages[user_id] = []
                    user_messages[user_id].append(content)

            for user_id, msg_list in user_messages.items():
                recent_msgs = msg_list[-5:]
                style_desc = await self._analyze_user_style_by_llm(
                    user_id, recent_msgs
                )
                if style_desc:
                    await fuser.refresh_impression(
                        user_id=user_id,
                        channel_id=self.stream_id,
                        interaction_digest="；".join(recent_msgs[-3:]),
                        style=style_desc,
                    )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 用户风格更新失败: {exc}")

    async def _analyze_user_style_by_llm(
        self, user_id: str, messages: List[str]
    ) -> str:
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

            logger.debug(
                f"{self.log_prefix} 🎨 用户 {user_id[:8]}: {style_desc}"
            )
            return style_desc

        except Exception as e:
            logger.debug(
                f"{self.log_prefix} LLM 风格分析失败: {e}，使用规则分析"
            )
            return self._analyze_user_style_by_rules(
                messages[-1] if messages else ""
            )

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
            if (
                dossier
                and dossier.impression
                and dossier.impression.dialogue_style
            ):
                style = dossier.impression.dialogue_style
                guide_parts = []

                if "极简" in style:
                    guide_parts.append(
                        "对方偏好极简回复，回复要非常简短，1-2句话即可"
                    )
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
        except Exception:
            return ""

    def _build_dynamic_length_hint(
        self, target_message: Any, user_style_guide: str = ""
    ) -> str:
        """根据用户输入长度和互动风格构建动态句长约束。"""
        user_text = ""
        if target_message is not None:
            user_text = (
                getattr(target_message, "processed_plain_text", "")
                or getattr(target_message, "content", "")
                or ""
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
        return f"[回复长度策略] 根据用户输入动态回复，本次控制在{low}~{high}句；短输入偏短，长输入可展开，不要固定句长。"

    def _build_persona_hint(self) -> str:
        """构建主人格提示片段。"""
        parts: List[str] = [
            "[人格约束] 无论内心独白还是最终回复，都优先服从主人格，不要临时长篇分析，不要偏成旁白。"
        ]
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
            parts.append(
                "[身份边界] 不要把自己说成AI助手、系统、客服或工具；你就是当前聊天现场里的这个人。"
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        try:
            from src.chat.heart_flow.persona_engine import (
                get_character_foundry,
            )

            shard = (
                get_character_foundry().active_prompt_shard() or ""
            ).strip()
            if shard:
                parts.append(f"[当前人格片段] {shard[:240]}")
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        return "\n".join(parts)

    def _resolve_latest_human_user_id(
        self, messages: Optional[List] = None
    ) -> str:
        """稳定解析当前轮对应的发言对象 ID，避免误用 stream_id 作为 memoir key。"""
        candidates = list(messages or [])
        for msg in reversed(candidates[-20:]):
            uid = str(getattr(msg, "user_id", "") or "").strip()
            if uid and uid != "bot":
                return uid
        return str(getattr(self, "_last_user_id", "") or "").strip()

    async def _invoke_unified_planner(
        self,
        messages: List,
        repetition_signal: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """统一规划器集成 - 让规划器感知当前状态并返回决策"""
        try:
            from src.core.unified_planner import get_unified_planner

            planner = get_unified_planner()
            user_id = ""
            last_user_text = ""
            if messages:
                for msg in reversed(messages):
                    uid = getattr(msg, "user_id", "")
                    if uid and uid != "bot":
                        user_id = uid
                        last_user_text = (
                            getattr(msg, "processed_plain_text", "")
                            or getattr(msg, "content", "")
                            or ""
                        )
                        break

            repeated_topic_pressure = 0.0
            if repetition_signal:
                repeat_count = float(
                    repetition_signal.get("exact_repeat_count", 0.0) or 0.0
                )
                repeated_topic_pressure = max(
                    0.0, min(1.0, repeat_count / 8.0)
                )
            harassment_signal = self._analyze_harassment_pressure(messages)
            behavior_signal = self._classify_behavior_signal(
                messages,
                repetition_signal=repetition_signal,
                harassment_signal=harassment_signal,
            )
            relation_snapshot = self._resolve_relation_view()

            target_message = self._get_latest_human_message(messages)
            context_execution_block = self._build_context_execution_block(
                target_message=target_message,
                voice_conclusion=getattr(self, "_cached_voice", None),
                repetition_signal=repetition_signal,
                relation_snapshot=relation_snapshot,
            )
            try:
                planner_mode_summary = "unknown/unknown"
                if user_id:
                    from src.modules.modcore.dynamic_persona.emotion_tracker import (
                        get_emotion_tracker, )

                    response_mode = get_emotion_tracker(
                        self.stream_id
                    ).get_layered_response_mode(user_id)
                    planner_mode_summary = (
                        f"{str(response_mode.get('tone', 'neutral') or 'neutral')}"
                        f"/{str(response_mode.get('response_length', 'normal') or 'normal')}"
                    )
                logger.info(
                    f"{self.log_prefix} 规划输入摘要 "
                    f"对象={user_id[:8] if user_id else 'none'} "
                    f"重复={int(repetition_signal.get('exact_repeat_count', 0) or 0) if repetition_signal else 0} "
                    f"前情块={'有' if context_execution_block else '无'} "
                    f"压力={repeated_topic_pressure:.2f} "
                    f"模式={planner_mode_summary}"
                )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")

            # 复用 inner_voice 阶段构建的统一快照，避免同一 tick 重复采集
            from src.core.world_snapshot import build_world_snapshot

            _cached = getattr(self, "_tick_world_snapshot", None)
            if _cached is not None:
                pre_built_snapshot = _cached
                self._tick_world_snapshot = None
            else:
                pre_built_snapshot = await build_world_snapshot(
                    self.stream_id, user_id
                )
            # 把增强链独有的行为信号和重复压力注入 snapshot
            pre_built_snapshot.behavior_category = str(
                (behavior_signal or {}).get("category", "neutral") or "neutral"
            )
            pre_built_snapshot.behavior_severity = float(
                (behavior_signal or {}).get("severity", 0.0) or 0.0
            )
            pre_built_snapshot.behavior_reason = str(
                (behavior_signal or {}).get("reason", "") or ""
            )
            pre_built_snapshot.repeated_topic_pressure = (
                repeated_topic_pressure
            )
            pre_built_snapshot.is_new_user = bool(
                (behavior_signal or {}).get("is_new_user", False)
            )
            # 从意图池获取活跃意图摘要供规划器参考
            _intention_hint = ""
            _active_intentions: List[Dict[str, Any]] = []
            try:
                from src.chat.proactive.intention_pool import (
                    get_intention_pool,
                )

                _pool = get_intention_pool()
                _intention_hint = _pool.build_planner_hint(self.stream_id)
                _active_intentions = _pool.build_planner_intent_payload(
                    self.stream_id, limit=3
                )
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            decision = await planner.plan(
                channel_id=self.stream_id,
                user_id=user_id,
                trigger="message_arrival",
                hints={
                    "repeated_topic_pressure": repeated_topic_pressure,
                    "last_user_intent": self._build_planner_user_hint(
                        last_user_text, repetition_signal
                    ),
                    "memory_hint": str(
                        getattr(self, "_latest_memory_hint", "") or ""
                    ),
                    "context_execution_block": context_execution_block,
                    "intention_hint": _intention_hint,
                    "active_intentions": _active_intentions,
                    "active_users": [
                        str(getattr(msg, "user_id", "") or "").strip()
                        for msg in messages[-20:]
                        if str(getattr(msg, "user_id", "") or "").strip()
                        and str(getattr(msg, "user_id", "") or "").strip()
                        != "bot"
                    ],
                    "current_topics": self._extract_current_topics(messages),
                    "relationship_candidates": relation_snapshot.get(
                        "relationship_candidates", []
                    ),
                    "group_pattern_hint": self._build_group_pattern_hint(),
                },
                world_snapshot=pre_built_snapshot,
            )
            if decision and decision.action.value != "observe":
                logger.info(
                    f"{self.log_prefix} 规划器决策={decision.action.label()}, "
                    f"原因={decision.reason[:30] if decision.reason else '无'}"
                )
            return decision
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 统一规划器调用失败: {exc}")
            return None

    def _build_group_pattern_hint(self) -> str:
        """F12：将群体行为模式转化为规划器可理解的提示文本"""
        if not self._cached_pattern_evidence:
            return ""
        _top = None
        _top_conf = 0.0
        for _pe in self._cached_pattern_evidence:
            _pc = float(getattr(_pe, "confidence", 0.0) or 0.0)
            if _pc > _top_conf:
                _top_conf = _pc
                _top = _pe
        if _top is None or _top_conf < 0.35:
            return ""
        _pt_val = getattr(getattr(_top, "pattern", None), "value", "") or ""
        _label = (
            getattr(_top.pattern, "label", "")
            if hasattr(_top, "pattern")
            else _pt_val
        )
        _hint_map = {
            "newcomer_welcome": "当前群聊正在迎新，适合主动打招呼或表示欢迎。",
            "ritual_greeting": "群里有礼仪性问候在进行，可以自然接一句。",
            "spectator_mode": "群里正在围观某事，不适合主动插话，更适合旁观。",
            "celebration_wave": "群里在庆祝什么，氛围好，可以参与但不抢风头。",
            "support_circle": "群里有人在寻求支持，如果关系够近可以考虑回应。",
            "heated_discussion": "群里讨论很激烈，除非被@否则不要乱插嘴。",
            "argument": "群里在吵架，不要卷进去。",
            "conflict_escalation": "冲突升级中，绝对不要介入。",
            "meme_storm": "群里在玩梗/刷表情包风暴，可以凑热闹但别太认真。",
            "emotional_contagion": "群里情绪在传染，注意不要被带偏。",
            "sudden_silence": "群里突然安静了，可能是好事也可能是尴尬时机。",
            "copycat_chain": "群里在复读，可以跟但不一定要每条都跟。",
            "collective_storytelling": "群里在讲故事，适合安静听或适时反应。",
        }
        return _hint_map.get(_pt_val, f"检测到群体模式: {_label}")

    def _build_planner_user_hint(
        self,
        last_user_text: str,
        repetition_signal: Optional[Dict[str, Any]] = None,
    ) -> str:
        text = str(last_user_text or "").strip()
        if not text:
            return ""
        prior_memory_hint = ""
        try:
            from src.chat.proactive.session_tracker import get_memoir_cabinet

            latest_user_id = str(
                getattr(self, "_last_user_id", "") or ""
            ).strip()
            if latest_user_id:
                memoir = get_memoir_cabinet().retrieve(latest_user_id)
                if memoir is not None:
                    last_topic = str(
                        getattr(memoir, "last_topic", "") or ""
                    ).strip()
                    if last_topic:
                        prior_memory_hint = (
                            f"你们之前已经聊到过{last_topic[:40]}。"
                        )
        except Exception as _e:
            logger.debug(f"异常: {_e}")
        if repetition_signal and repetition_signal.get(
            "latest_matches_repeat"
        ):
            reason = str(
                repetition_signal.get("reason", "重复输入") or "重复输入"
            )
            return (
                f"最近对方原话：{text[:80]}；这是重复短句，不要当成全新话题。"
                f"优先结合你之前已经问过的上下文继续接。{prior_memory_hint}{reason[:60]}"
            )
        if len(self._normalize_repeat_text(text)) <= 6:
            return f"最近对方原话：{text[: 80]} ；这是很短的线索，先结合前情判断，不要硬当新话题。{
                prior_memory_hint} "
        return f"最近对方原话：{text[:120]}"

    async def _store_interaction_memory(self, messages: List) -> None:
        """存储交互记忆 - 将交互存入记忆系统"""
        try:
            from src.memory_system.memory_core import acquire_recollection_hub

            hub = acquire_recollection_hub()
            for msg in messages[-3:]:
                content = getattr(msg, "processed_plain_text", "") or getattr(
                    msg, "content", ""
                )
                if not content:
                    continue
                user_id = getattr(msg, "user_id", "unknown")
                is_bot = user_id == "bot"
                hub.deposit_memory(
                    stream_id=self.stream_id,
                    content=content,
                    entry_category=(
                        "bot_response" if is_bot else "conversation"
                    ),
                    user_id=user_id,
                    significance=0.5,
                )
            logger.debug(
                f"{self.log_prefix} 已存储 {len(messages[-3:])} 条交互记忆"
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 存储交互记忆失败: {exc}")

    # ═══════════════════════════════════════════════════
    #  多维独立状态系统 - 决策网关集成
    # ═══════════════════════════════════════════════════

    async def _run_dimension_gateway(
        self,
        decision_messages: list,
        pinged_msg,
        voice_conclusion,
        relation_result: dict,
        repetition_signal: dict | None = None,
        harassment_signal: dict | None = None,
    ) -> dict | None:
        """
        调用多维状态调度器收集投票，然后通过决策网关裁定。
        返回兼容旧格式的 dict，或在调度器不可用时返回 None。
        裁定结果同时缓存在 self._last_gateway_verdict 供后续使用。
        """
        self._last_gateway_verdict = None
        try:
            from src.chat.heart_flow.dimension_dispatcher import DimensionDispatcher
            from src.chat.heart_flow.dimension_protocol import EventContext
            from src.chat.heart_flow.decision_gateway import (
                DecisionGateway,
                GatewayContext,
                gateway_verdict_to_compat,
            )
        except ImportError as ie:
            logger.debug(f"{self.log_prefix} 维度网关模块不可用: {ie}")
            return None
        try:
            dispatcher = await DimensionDispatcher.get_instance()
            if dispatcher.registered_count == 0:
                return None
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 获取调度器失败: {exc}")
            return None
        # 提取最近一条非bot消息的文本和用户ID
        latest_text = ""
        latest_uid = ""
        for msg in reversed(decision_messages[-10:]):
            uid = getattr(msg, "user_id", "") or ""
            if uid and uid != "bot":
                latest_text = str(
                    getattr(msg, "processed_plain_text", "")
                    or getattr(msg, "plain_text", "")
                    or getattr(msg, "content", "")
                    or ""
                ).strip()
                latest_uid = uid
                break
        # 计算情感倾向
        sentiment = 0.0
        if relation_result:
            affection = float(relation_result.get("affection", 0.0) or 0.0)
            annoyance = float(relation_result.get("annoyance_value", 0.0) or 0.0)
            sentiment = max(-1.0, min(1.0, (affection - annoyance) / 100.0))
        # 构建事件上下文并广播
        evt = EventContext(
            event_type="decision_phase",
            user_id=latest_uid,
            channel_id=self.stream_id,
            message_text=latest_text,
            message_length=len(latest_text),
            is_at_bot=(pinged_msg is not None),
            is_reply_to_bot=bool(
                pinged_msg is not None
                and getattr(pinged_msg, "is_reply_to_bot", False)
            ),
            sentiment_score=sentiment,
        )
        try:
            await dispatcher.broadcast_event(evt)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 维度广播异常: {exc}")
        # 收集所有维度投票
        try:
            votes = dispatcher.collect_votes(evt)
        except Exception as exc:
            logger.warning(f"{self.log_prefix} 维度投票收集失败: {exc}")
            return None
        # 构建网关上下文
        desire_level = 5
        voice_should_reply = True
        if voice_conclusion is not None:
            desire_level = int(
                getattr(voice_conclusion, "reply_desire_level", 5) or 5
            )
            voice_should_reply = bool(
                getattr(voice_conclusion, "should_reply", True)
            )
        has_mention = any(
            getattr(m, "user_id", "") != "bot"
            and self._is_message_pinged(m)
            for m in decision_messages[-10:]
        )
        is_reply_to = bool(
            pinged_msg is not None
            and getattr(pinged_msg, "is_reply_to_bot", False)
        )
        gw_ctx = GatewayContext(
            is_direct_ping=(pinged_msg is not None),
            has_indirect_mention=has_mention,
            is_reply_to_bot=is_reply_to,
            desire_level=desire_level,
            voice_should_reply=voice_should_reply,
            message_text=latest_text,
            user_id=latest_uid,
            channel_id=self.stream_id,
        )
        # 执行决策网关裁定
        gateway = DecisionGateway.get_instance()
        verdict = gateway.decide(votes, gw_ctx)
        self._last_gateway_verdict = verdict
        logger.info(
            f"{self.log_prefix} [维度网关] gate={verdict.gate} "
            f"prob={verdict.final_probability:.3f} "
            f"skip={verdict.should_skip} src={verdict.decision_source} "
            f"tags={verdict.attitude_tags[:3]} styles={verdict.style_hints[:3]}"
        )
        # 转换为兼容旧格式
        compat = gateway_verdict_to_compat(verdict)
        result = compat.to_dict()
        result["_gateway_verdict"] = verdict
        return result

    def _inject_dimension_state_prompt(self, extra_parts: list) -> None:
        """
        将维度系统的LLM状态提示词注入回复生成的extra_parts。
        需要在 extra_parts 组装阶段调用。
        """
        verdict = getattr(self, "_last_gateway_verdict", None)
        if verdict is None:
            return
        # 注入状态描述
        if verdict.llm_state_prompt:
            extra_parts.append(verdict.llm_state_prompt)
        # 注入风格修饰
        if verdict.style_hints:
            hints_text = "；".join(verdict.style_hints[:5])
            extra_parts.append(f"[维度风格指引] {hints_text}")
        # 注入态度标签
        if verdict.attitude_tags:
            tags_text = "、".join(verdict.attitude_tags[:5])
            extra_parts.append(f"[维度态度标签] {tags_text}")
        # 面具质量上限警告
        if verdict.quality_cap < 0.8:
            extra_parts.append(
                f"[面具约束] 当前回复质量上限={verdict.quality_cap:.0%}，"
                "措辞可能不够自然，偶尔流露真实想法"
            )
        # 创伤潜意识干扰
        if verdict.perception_labels:
            labels = "；".join(verdict.perception_labels[:3])
            extra_parts.append(f"[潜意识干扰] {labels}")
        # 内心冲突注入
        if verdict.inner_conflict:
            extra_parts.append(f"[内心冲突] {verdict.inner_conflict}")

    async def _update_emotion_tracker_state(self, messages: List) -> None:
        """更新情绪追踪器状态 - 好感、烦躁、心理压力、创伤值等"""
        try:
            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            tracker = get_emotion_tracker(self.stream_id)
            for msg in messages:
                user_id = getattr(msg, "user_id", "")
                if not user_id or user_id == "bot":
                    continue
                content = getattr(msg, "processed_plain_text", "") or getattr(
                    msg, "content", ""
                )
                if not content:
                    continue
                polarity = self._estimate_text_polarity(content)
                if polarity > 0.1:
                    interaction_type = "positive"
                elif polarity < -0.1:
                    interaction_type = "negative"
                else:
                    interaction_type = "neutral"
                tracker.process_interaction(
                    user_id,
                    interaction_type,
                    content=content[:100],
                    magnitude=(
                        min(2.0, abs(polarity) * 3.0)
                        if interaction_type != "neutral"
                        else 1.0
                    ),
                )
                # 额外处理：调教进度和创伤累积（process_interaction 不覆盖这些字段）
                state = tracker.get_user_state(
                    user_id, create_if_missing=False
                )
                if state:
                    if (
                        interaction_type == "positive"
                        and state.training_stage > 0
                    ):
                        state.training_progress = min(
                            100.0, state.training_progress + 0.5
                        )
                        state.training_resistance = max(
                            0.0, state.training_resistance - 0.3
                        )
                    elif interaction_type == "negative":
                        state.training_resistance = min(
                            100.0, state.training_resistance + 0.5
                        )
                        state.trauma_accumulation = min(
                            10.0, state.trauma_accumulation + 0.1
                        )
                    tracker._save_user_state(user_id)
            logger.debug(f"{self.log_prefix} 情绪追踪器状态已更新")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 情绪追踪器更新失败: {exc}")

    async def _apply_repetition_emotion_feedback(
        self,
        messages: List,
        repetition_signal: Optional[Dict[str, Any]],
    ) -> None:
        """把重复输入压力回写到当前情绪追踪器，形成稳定的烦躁/压力累积。"""
        _use_fallback = False
        if not repetition_signal or not repetition_signal.get("detected"):
            _use_fallback = True
        if not _use_fallback:
            latest_matches_repeat = bool(
                repetition_signal.get("latest_matches_repeat", False)
            )
            if not latest_matches_repeat:
                _use_fallback = True
        try:
            target_message = self._get_latest_human_message(messages)
            if target_message is None:
                return
            _msg_uid = str(
                getattr(target_message, "user_id", "") or ""
            ).strip()
            _persisted_uid = str(
                getattr(self, "_last_user_id", "") or ""
            ).strip()
            user_id = _persisted_uid or _msg_uid
            if not user_id:
                return

            exact_repeat_count = 0
            repeat_user_count = 0
            low_info_cluster = False
            _repeat_reason = ""
            if not _use_fallback:
                exact_repeat_count = int(
                    repetition_signal.get("exact_repeat_count", 0) or 0
                )
                repeat_user_count = int(
                    repetition_signal.get("repeat_user_count", 0) or 0
                )
                low_info_cluster = bool(
                    repetition_signal.get("low_info_cluster", False)
                )
                _repeat_reason = repetition_signal.get("reason", "repeat")
            else:
                _latest_text = (
                    getattr(target_message, "processed_plain_text", "")
                    or getattr(target_message, "content", "")
                    or ""
                ).strip()
                _normalized = self._normalize_repeat_text(_latest_text)
                exact_repeat_count = self._count_recent_user_repeats(
                    _normalized, user_id
                )
                if exact_repeat_count >= 2:
                    _repeat_reason = f"连续重复({exact_repeat_count}次)"
                else:
                    return

            from src.modules.modcore.dynamic_persona.emotion_tracker import (
                get_emotion_tracker,
            )

            tracker = get_emotion_tracker(self.stream_id)
            state = tracker.get_user_state(user_id, create_if_missing=True)
            if state is None:
                return

            # 重复消息烦躁增量（温和版：避免少量重复直接拉满）
            annoyance_delta = 0.0
            pressure_delta = 0.0

            # 完全重复消息：每次 +5 烦躁（少量重复可能是网络卡顿或玩梗）
            if exact_repeat_count >= 1:
                annoyance_delta = exact_repeat_count * 5.0
                pressure_delta = min(5.0, exact_repeat_count * 1.5)

            # 同一用户连续多条消息（非完全重复）：每次 +3
            if repeat_user_count >= 2:
                annoyance_delta += repeat_user_count * 3.0
                pressure_delta += min(3.0, repeat_user_count * 1.0)

            # 低信息量消息群
            if low_info_cluster:
                annoyance_delta += 3.0
                pressure_delta += 1.0

            # 单次最大变化不超过 50（防止一次性拉满）
            annoyance_delta = min(50.0, annoyance_delta)

            tracker.update_annoyance(
                user_id,
                annoyance_delta,
                reason=f"重复输入压力: {_repeat_reason}",
            )
            state.psychological_pressure = max(
                0.0,
                min(
                    100.0,
                    float(getattr(state, "psychological_pressure", 0.0) or 0.0)
                    + pressure_delta,
                ),
            )
            state.last_interaction = time.time()
            tracker._save_user_state(user_id)

            logger.info(
                f"{self.log_prefix} 重复情绪回写 对象={user_id[:8]} 烦躁+{annoyance_delta:.1f} "
                f"压力+{pressure_delta:.1f}"
            )
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 重复情绪回写失败: {exc}")

    async def _update_trauma_system_state(self, messages: List) -> None:
        """更新创伤系统状态 - 内心混乱、伪装强度、压力累积"""
        try:
            from src.modules.trauma.trauma_system import get_trauma_system

            trauma_sys = get_trauma_system()
            has_negative = False
            total_intensity = 0.0
            for msg in messages:
                user_id = getattr(msg, "user_id", "")
                if user_id == "bot":
                    continue
                content = getattr(msg, "processed_plain_text", "") or getattr(
                    msg, "content", ""
                )
                if not content:
                    continue
                polarity = self._estimate_text_polarity(content)
                if polarity < -0.1:
                    has_negative = True
                    total_intensity += abs(polarity)
            trauma_state = trauma_sys.get_state()
            if has_negative:
                chaos_increase = min(2.0, total_intensity * 0.5)
                trauma_sys.add_stress(chaos_increase)
                context = {
                    "trauma_score": trauma_state.inner_chaos_level,
                    "sentiment": "negative",
                    "intensity": total_intensity,
                    "social_pressure": trauma_state.stress_accumulation,
                }
                trauma_sys.update_from_context(context)
                logger.debug(
                    f"{self.log_prefix} 💔 内心混乱={trauma_state.inner_chaos_level:.1f}, "
                    f"伪装强度={trauma_state.surface_mask_strength:.1f}, "
                    f"压力累积={trauma_state.stress_accumulation:.1f}"
                )
            else:
                trauma_sys.reduce_stress(0.5)
            logger.debug(f"{self.log_prefix} 创伤系统状态已更新")
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 创伤系统更新失败: {exc}")

    async def _sync_memoir_on_message(self, messages: List) -> None:
        """会话追踪同步 — 消息到达时更新 MemoirCabinet 的对话阶段"""
        try:
            from src.chat.proactive.session_tracker import get_memoir_cabinet
            from src.common.data_models.proactive_models import DialogueStage

            cabinet = get_memoir_cabinet()
            memoir_user_id = self._resolve_latest_human_user_id(messages)
            if not memoir_user_id:
                return
            memoir = (
                cabinet.retrieve(memoir_user_id)
                if hasattr(cabinet, "retrieve")
                else None
            )
            if memoir is None:
                if hasattr(cabinet, "create"):
                    memoir = cabinet.create(
                        memoir_user_id, channel_id=self.stream_id
                    )
                else:
                    return
            has_user_msg = any(
                getattr(m, "user_id", "") not in ("", "bot") for m in messages
            )
            if has_user_msg:
                if hasattr(memoir, "phase"):
                    if memoir.phase in (
                        DialogueStage.OPEN,
                        DialogueStage.IDLE,
                        DialogueStage.COOLING,
                    ):
                        memoir.phase = DialogueStage.ACTIVE
                if hasattr(memoir, "last_human_ts"):
                    memoir.last_human_ts = time.time()
            if hasattr(cabinet, "commit"):
                cabinet.commit(self.stream_id, memoir)
        except Exception as exc:
            logger.debug(f"{self.log_prefix} 会话追踪同步失败: {exc}")

    def _integrate_deep_visibility_and_freshness(
        self, incoming_batch: List
    ) -> None:
        """GAP-M+N：深度可见性评分 + 新鲜度注册 + 回看触发评估
        在消息进入感官门控前执行，为后续决策提供三维偏置评分和衰减数据"""
        if not incoming_batch:
            return
        # GAP-M：深度可见性评分已在 _apply_sensory_gates 中完成
        # 这里补充新鲜度注册和回看触发
        try:
            from src.core.freshness_decay_engine import (
                get_freshness_decay_engine,
                DecaySpeedTier,
                VisibilityHistoryEffect,
                RecallTriggerType,
            )

            _fde = get_freshness_decay_engine(self.stream_id)
            for _msg in incoming_batch:
                _uid_n = str(getattr(_msg, "user_id", "") or "")
                if not _uid_n or _uid_n == "bot":
                    continue
                _mid_n = str(
                    getattr(_msg, "message_id", "")
                    or getattr(_msg, "msg_id", "")
                    or id(_msg)
                )
                if _mid_n in self._cached_freshness_records:
                    continue
                _plain_n = str(
                    getattr(_msg, "processed_plain_text", "")
                    or getattr(_msg, "plain_text", "")
                    or getattr(_msg, "content", "")
                    or ""
                )
                _is_at_n = getattr(_msg, "is_at", False)
                _is_quote_n = bool(getattr(_msg, "reply_to_message_id", None))
                if _is_at_n or _is_quote_n:
                    _tier_n = DecaySpeedTier.PRESERVED
                    _vis_n = VisibilityHistoryEffect.UNDERSTOOD_DEFERRED
                elif len(_plain_n) > 100:
                    _tier_n = DecaySpeedTier.SLOW_DECAY
                    _vis_n = VisibilityHistoryEffect.SEEN_AND_NOTICED
                else:
                    _tier_n = DecaySpeedTier.MEDIUM_DECAY
                    _vis_n = VisibilityHistoryEffect.GLANCED_BUT_IGNORED
                _rec = _fde.register_message(
                    message_id=_mid_n,
                    user_id=_uid_n,
                    content_fingerprint=_plain_n[:60],
                    initial_tier=_tier_n,
                    visibility=_vis_n,
                    topic_tags=[],
                )
                self._cached_freshness_records[_mid_n] = {
                    "record": _rec,
                    "registered_at": time.time(),
                }
            # 评估回看触发
            _boredom_val = 0.0
            _meta_ns = (
                getattr(self, "_cached_metabolism_constraints", None) or {}
            )
            if _meta_ns:
                _boredom_val = float(_meta_ns.get("boredom_level", 0.0) or 0.0)
            _fde.set_boredom_level(_boredom_val)
            if _boredom_val > 0.5:
                _recall_results = _fde.evaluate_boredom_drift()
                if _recall_results:
                    logger.debug(
                        f"{self.log_prefix} [GAP-N] 无聊回看: "
                        f"触发了{len(_recall_results)}条消息重新激活"
                    )
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def _integrate_gossip_ritual_strategy(
        self, now: float
    ) -> Optional[Dict[str, Any]]:
        """GAP-O：吃瓜/仪式行为策略引擎集成
        基于群体模式检测结果，评估当前事件型话题的参与策略"""
        if not self._cached_pattern_evidence:
            return None
        try:
            from src.core.gossip_ritual_strategy import (
                get_gossip_ritual_engine,
                EventType,
                ParticipationPosture,
                EventContext,
                SubjectiveReadiness,
            )

            if not self._gossip_engine_initialized:
                self._gossip_engine_initialized = True
            _gre = get_gossip_ritual_engine(self.stream_id)
            _top_pe_o = None
            _top_pc_o = 0.0
            for _pe in self._cached_pattern_evidence:
                _pc_o = float(getattr(_pe, "confidence", 0.0) or 0.0)
                if _pc_o > _top_pc_o:
                    _top_pc_o = _pc_o
                    _top_pe_o = _pe
            if _top_pe_o is None or _top_pc_o < 0.35:
                return None
            _pt_enum_o = getattr(_top_pe_o, "pattern", None)
            _pt_val_o = (
                (
                    _pt_enum_o.value
                    if hasattr(_pt_enum_o, "value")
                    else str(_pt_enum_o)
                )
                if _pt_enum_o
                else ""
            )
            _evt_type_map = {
                "newcomer_welcome": EventType.NEWCOMER_WELCOME,
                "birthday_wish": EventType.BIRTHDAY_CELEBRATION,
                "holiday_greeting": EventType.HOLIDAY_GREETING,
                "congratulation": EventType.CONGRATULATIONS,
                "heated_discussion": EventType.GROUP_CONFLICT,
                "argument": EventType.GROUP_CONFLICT,
                "conflict_escalation": EventType.GROUP_CONFLICT,
                "support_circle": EventType.SUPPORT_CIRCLE,
                "celebration_wave": EventType.CONGRATULATIONS,
                "spectator_mode": EventType.GOSSIP_MELON,
                "chain_reply": EventType.RITUAL_REPLY_CHAIN,
                "meme_chain": EventType.MEME_CHAIN,
                "bot_discussed": EventType.BOT_DISCUSSION,
            }
            _evt_type_o = _evt_type_map.get(_pt_val_o, EventType.NONE)
            if _evt_type_o == EventType.NONE:
                return None
            _involved_users_o = set()
            _msg_count_o = 0
            if (
                hasattr(_top_pe_o, "involved_users")
                and _top_pe_o.involved_users
            ):
                _involved_users_o = set(_top_pe_o.involved_users)
            if hasattr(_top_pe_o, "message_count"):
                _msg_count_o = int(_top_pe_o.message_count or 0)
            _is_bot_relevant_o = False
            if self._cached_self_references:
                for _ref in self._cached_self_references:
                    _rt_o = getattr(_ref, "ref_type", None)
                    if _rt_o:
                        _rv_o = (
                            _rt_o.value
                            if hasattr(_rt_o, "value")
                            else str(_rt_o)
                        )
                        if _rv_o in (
                            "discussed_as_topic",
                            "quoted_reply",
                            "direct_at",
                            "nickname_called",
                        ):
                            _is_bot_relevant_o = True
                            break
            _event_ctx = EventContext(
                event_type=_evt_type_o,
                confidence=_top_pc_o,
                involved_users=_involved_users_o,
                is_bot_relevant=_is_bot_relevant_o,
                bot_mentioned_directly=_is_bot_relevant_o
                and any(
                    str(getattr(r, "ref_type", ""))
                    in ("direct_at", "nickname_called")
                    for r in (self._cached_self_references or [])
                ),
                event_intensity=min(1.0, _top_pc_o * 1.2),
                message_count_in_event=_msg_count_o,
            )
            _meta_o = (
                getattr(self, "_cached_metabolism_constraints", None) or {}
            )
            _rel_o = getattr(self, "_last_relation_snapshot", None) or {}
            _readiness = SubjectiveReadiness(
                energy_ratio=max(
                    0.05,
                    1.0 - float(_meta_o.get("energy_suppression", 0.0) or 0.0),
                ),
                social_willingness=0.5,
                boredom_level=float(_meta_o.get("boredom_level", 0.0) or 0.0),
                loafing_level=float(
                    _meta_o.get("loafing_suppression", 0.0) or 0.0
                ),
                watch_state_rank=2,
                relation_to_involved_avg=float(
                    _rel_o.get("affection", 50.0) or 50.0
                )
                / 100.0,
                annoyance_to_group=float(
                    _rel_o.get("annoyance_value", 0.0) or 0.0
                )
                / 100.0,
                interest_in_topic=0.6 if _is_bot_relevant_o else 0.3,
            )
            _verdict = _gre.evaluate(_event_ctx, _readiness)
            self._cached_gossip_ritual_verdict = _verdict.to_dict()
            if _verdict.should_act:
                logger.info(
                    f"{self.log_prefix} [GAP-O] 事件策略: {_evt_type_o.label()} → "
                    f"{_verdict.posture.label()} (优先级={_verdict.priority:.2f})"
                )
            else:
                logger.debug(
                    f"{self.log_prefix} [GAP-O] 事件策略: {_evt_type_o.label()} → "
                    f"{_verdict.posture.label()}, 原因: {_verdict.defer_reason[:40]}"
                )
            return self._cached_gossip_ritual_verdict
        except Exception as exc:
            logger.debug(f"{self.log_prefix} [GAP-O] 吃瓜策略引擎异常: {exc}")
            return None

    def _run_adaptive_pipeline_periodic(
        self, now: float
    ) -> Optional[Dict[str, Any]]:
        """GAP-P：自适应更新管道周期性运行
        每隔一段时间收集事件并尝试运行管道"""
        _interval = 1800.0
        if now - self._adaptive_pipeline_last_run < _interval:
            return None
        self._adaptive_pipeline_last_run = now
        try:
            from src.core.adaptive_update_pipeline import get_adaptive_pipeline

            _ap = get_adaptive_pipeline()
            _stats = _ap.get_collection_stats()
            if int(_stats.get("recent_window_count", 0) or 0) < getattr(
                type(self), "_SUMMARY_MIN_EVENTS_FOR_PIPELINE", 3
            ):
                return None
            _run = _ap.run_pipeline()
            self._cached_pipeline_summary = _run.to_dict()
            if _run.final_status == "published":
                logger.info(
                    f"{self.log_prefix} [GAP-P] 自适应管道发布成功: "
                    f"events={_run.events_collected}, candidates={len(_run.candidates)}, "
                    f"published={sum(1 for p in _run.publishes if p.published)}"
                )
            elif _run.final_status not in (
                "skipped_insufficient_events",
                "summarized_no_generation_needed",
            ):
                logger.debug(
                    f"{self.log_prefix} [GAP-P] 管道状态: {_run.final_status}"
                )
            return self._cached_pipeline_summary
        except Exception as exc:
            logger.debug(f"{self.log_prefix} [GAP-P] 自适应管道异常: {exc}")
            return None

    _SUMMARY_MIN_EVENTS_FOR_PIPELINE = 3

    def _integrate_memory_governance(
        self, now: float
    ) -> Optional[Dict[str, Any]]:
        """GAP-Q：记忆治理引擎周期运行——注册消息记忆+淘汰/合并+动态窗口"""
        _gov_interval = 600.0
        if now - self._last_memory_governance_ts < _gov_interval:
            return self._cached_memory_governance_snap
        self._last_memory_governance_ts = now
        try:
            from src.core.memory_governance_engine import (
                get_memory_governance_engine,
                MemoryImportanceTier,
            )

            if not self._memory_governance_initialized:
                self._memory_governance_initialized = True
            _mge = get_memory_governance_engine(self.stream_id)
            _snap = _mge.run_governance_cycle()
            self._cached_memory_governance_snap = _snap.to_dict()
            if _snap.last_operation and _snap.last_operation != "无需操作":
                logger.info(
                    f"{self.log_prefix} [GAP-Q] 记忆治理: {_snap.last_operation} | "
                    f"总量={_snap.total_entries}, 利用率={_snap.utilization_ratio:.1%}"
                )
            return self._cached_memory_governance_snap
        except Exception as exc:
            logger.debug(f"{self.log_prefix} [GAP-Q] 记忆治理异常: {exc}")
            return None

    def register_message_to_memory(
        self,
        content: str,
        user_id: str = "",
        *,
        topic_tags: Optional[List[str]] = None,
    ) -> None:
        """GAP-Q快捷接口：将单条消息内容注册到记忆治理引擎"""
        try:
            from src.core.memory_governance_engine import (
                get_memory_governance_engine,
            )

            if not self._memory_governance_initialized:
                return
            _mge = get_memory_governance_engine(self.stream_id)
            _rel_q = getattr(self, "_last_relation_snapshot", None) or {}
            _emotion_q = float(_rel_q.get("annoyance_value", 0.0) or 0.0)
            _emotion_q = -min(1.0, max(-1.0, _emotion_q / 100.0))
            _mge.register_memory(
                content=content[:500],
                source_user=user_id,
                source_type="message",
                topic_tags=topic_tags,
                emotional_valence=_emotion_q,
                channel_context=self.stream_id[:20],
            )
        except Exception as _e:
            logger.debug(f"异常: {_e}")

    def build_planner_injection_prompt(
        self, messages: List[Any]
    ) -> Dict[str, Any]:
        """GAP-S：构建完整注入提示词——聚合所有GAP机制输出并生成结构化prompt"""
        try:
            from src.core.planner_prompt_injection import (
                get_planner_injection_layer,
            )

            if not self._planner_injection_initialized:
                self._planner_injection_initialized = True
            _pil = get_planner_injection_layer()
            _sit_input = None
            _sit_cached = getattr(
                self, "_cached_situation_interpretation", None
            )
            if isinstance(_sit_cached, dict):
                _sit_input = _sit_cached
            _gossip_input = self._cached_gossip_ritual_verdict
            _vis_input = None
            if self._cached_deep_visibility_results:
                from src.core.deep_visibility_scorer import (
                    get_deep_visibility_scorer,
                )

                _dvs = get_deep_visibility_scorer(self.stream_id)
                _vis_stats = _dvs.summary_stats(
                    self._cached_deep_visibility_results
                )
                _vis_input = _vis_stats
            _safety_input = self._cached_safety_assessment
            _mem_input = None
            _mem_snap = self._cached_memory_governance_snap
            if _mem_snap:
                _mem_input = {
                    "window_size": _mem_snap.get("total", 0),
                    "total_chars": 0,
                    "tier_dist": _mem_snap.get("tiers", {}),
                    "reason": "",
                }
            _fresh_input = None
            try:
                from src.core.freshness_decay_engine import (
                    get_freshness_decay_engine,
                )

                _fde = get_freshness_decay_engine(self.stream_id)
                _fresh_snap = _fde.all_records_snapshot()
                _fresh_input = {
                    "total_tracked": _fresh_snap.get("total_tracked", 0),
                    "fresh_count": _fresh_snap.get("fresh_count", 0),
                    "stale_count": _fresh_snap.get("stale_count", 0),
                    "recalls": sum(
                        1
                        for v in (
                            _fresh_snap.get("recent_distribution") or {}
                        ).values()
                        if "recall" in str(v).lower()
                    ),
                    "tier_dist": _fresh_snap.get("tier_distribution", {}),
                }
            except Exception as _e:
                logger.debug(f"{self.log_prefix} unknown异常: {_e}")
            _meta_raw = self._cached_metabolism_constraints or {}
            _night_phase_raw = None
            _np = getattr(self, "_cached_night_phase", None)
            if _np and hasattr(_np, "value"):
                _night_phase_raw = _np.value
            _packet = _pil.build_packet(
                channel_id=self.stream_id,
                situation=_sit_input,
                gossip_verdict=_gossip_input,
                visibility_stats=_vis_input,
                safety_result=_safety_input,
                memory_window=_mem_input,
                freshness_snapshot=_fresh_input,
                metabolism_raw=_meta_raw,
                night_phase=_night_phase_raw,
                relation_snap=getattr(self, "_last_relation_snapshot", None),
                watch_state=str(
                    getattr(self, "_cached_watch_level", "") or ""
                ),
            )
            _result = _pil.generate_full_prompt(_packet)
            self._cached_planner_injection = _result.to_dict()
            return self._cached_planner_injection
        except Exception as exc:
            logger.debug(f"{self.log_prefix} [GAP-S] 提示词注入异常: {exc}")
            return {"injection_count": 0, "error": str(exc)[:60]}

    def _run_attention_flow_tick(self, now: float) -> Optional[Dict[str, Any]]:
        """GAP-W：主观注意力流控制器tick——每15秒更新注意力状态"""
        try:
            from src.core.subjective_attention_flow import (
                get_attention_flow_controller,
                AttentionState,
            )

            if not self._attention_flow_initialized:
                self._attention_flow_initialized = True
            _afc = get_attention_flow_controller(self.stream_id)
            _meta_aw = self._cached_metabolism_constraints or {}
            _emotion_snap = getattr(self, "_cached_emotion_state", None) or {}
            _np_aw = getattr(self, "_cached_night_phase", None)
            _safety_aw = getattr(self, "_cached_safety_assessment", None) or {}
            _snap = _afc.tick(
                now,
                boredom=float(_meta_aw.get("boredom_level", 0.0) or 0.0),
                social_desire=float(
                    _emotion_snap.get("social_desire", 0.5) or 0.5
                ),
                energy=max(
                    0.05,
                    1.0
                    - float(_meta_aw.get("energy_suppression", 0.0) or 0.0),
                ),
                mood=float(_emotion_snap.get("mood", 0.5) or 0.5),
                loneliness=float(_emotion_snap.get("loneliness", 0.0) or 0.0),
                curiosity=float(_emotion_snap.get("curiosity", 0.3) or 0.3),
                night_phase=(_np_aw.value if hasattr(_np_aw, "value") else ""),
                safety_level=str(_safety_aw.get("level", "") or ""),
                has_pending_user=bool(self._pending_user_ids),
                group_activity_level=self._group_activity_level,
                emotion_overload_flag=self._cached_user_negative_emotion > 70,
            )
            self._cached_attention_snapshot = _snap.to_dict()
            if _snap.state != _snap.previous_state:
                logger.info(
                    f"{self.log_prefix} [GAP-W] 注意力状态: {_snap.previous_state.label()} → {_snap.state.label()} "
                    f"({_snap.transition_reason.label() if _snap.transition_reason else ''})"
                )
            return self._cached_attention_snapshot
        except Exception as exc:
            logger.debug(f"{self.log_prefix} [GAP-W] 注意力流异常: {exc}")
            return None

    def run_emotion_feedback_cycle(
        self, now: float
    ) -> Optional[Dict[str, Any]]:
        """GAP-V：情感反馈环周期执行——聚合所有引擎输出回写emotion_state"""
        try:
            from src.core.emotion_feedback_loop import (
                get_emotion_feedback_loop,
                FeedbackSource,
            )

            if not self._emotion_feedback_initialized:
                self._emotion_feedback_initialized = True
            _efl = get_emotion_feedback_loop()
            _meta_v = self._cached_metabolism_constraints or {}
            _efl.submit_from_metabolism(
                self.stream_id,
                energy_ratio=max(
                    0.05,
                    1.0 - float(_meta_v.get("energy_suppression", 0.0) or 0.0),
                ),
                boredom_level=float(_meta_v.get("boredom_level", 0.0) or 0.0),
                loafing_level=float(
                    _meta_v.get("loafing_suppression", 0.0) or 0.0
                ),
            )
            _sit_v = getattr(self, "_cached_situation_interpretation", None)
            if isinstance(_sit_v, dict):
                _efl.submit_from_situation(
                    self.stream_id,
                    risk_score=float(_sit_v.get("risk_score", 0.0) or 0.0),
                    atmosphere=str(_sit_v.get("atmosphere", "") or ""),
                    engagement_intensity=float(
                        _sit_v.get("intensity", 0.5) or 0.5
                    ),
                    internal_energy=float(_sit_v.get("energy", 0.5) or 0.5),
                )
            _safe_v = self._cached_safety_assessment
            if isinstance(_safe_v, dict):
                _efl.submit_from_safety(
                    self.stream_id,
                    safety_level=str(_safe_v.get("level", "") or ""),
                    safety_score=float(_safe_v.get("score", 0.0) or 0.0),
                    blocked=bool(_safe_v.get("block_reply", False)),
                )
            _np_v = getattr(self, "_cached_night_phase", None)
            if _np_v and hasattr(_np_v, "value"):
                _efl.submit_from_night(
                    self.stream_id,
                    night_phase=_np_v.value,
                    pressure_total=float(
                        getattr(self, "_overnight_pressure_total", 0.0) or 0.0
                    ),
                    is_burnthrough=False,
                )
            _latest_msg = None
            _incoming = getattr(self, "_cached_incoming_messages", []) or []
            if _incoming:
                _latest_msg = _incoming[-1]
            elif hasattr(self, "_last_user_message"):
                _latest_msg = self._last_user_message
            if _latest_msg:
                _msg_text = str(
                    getattr(_latest_msg, "processed_plain_text", "") or ""
                )
                _uid = str(getattr(_latest_msg, "user_id", "") or "")
                _is_mentioned = bool(
                    getattr(_latest_msg, "is_at", False)
                    or getattr(_latest_msg, "mentioned_me", False)
                )
                _is_question = "?" in _msg_text or "？" in _msg_text
                _is_first = False
                try:
                    from src.core.impression_evolution_hub import (
                        get_impression_hub,
                    )

                    _ihub = get_impression_hub(self.stream_id)
                    _imp = _ihub.get_impression(_uid) if _uid else None
                    if _imp and hasattr(_imp, "truth"):
                        _is_first = (
                            getattr(_imp.truth, "total_interactions", 0) or 0
                        ) <= 2
                except Exception:
                    pass
                _repeat_cnt = (
                    self._count_recent_user_repeats(_msg_text, _uid)
                    if _msg_text and _uid
                    else 0
                )
                _efl.submit_from_user_interaction(
                    self.stream_id,
                    is_mentioned=_is_mentioned,
                    is_question=_is_question,
                    is_first_contact=_is_first,
                    message_length=len(_msg_text),
                    repeat_count=_repeat_cnt,
                )
            _report = _efl.run_feedback_cycle(self.stream_id)
            self._cached_emotion_feedback_report = _report.to_dict()
            return self._cached_emotion_feedback_report
        except Exception as exc:
            logger.debug(f"{self.log_prefix} [GAP-V] 情感反馈异常: {exc}")
            return None

    def register_cross_engine_outputs(self) -> Dict[str, Any]:
        """GAP-T：跨引擎联动验证——注册所有引擎产出并验证消费状态"""
        try:
            from src.core.cross_engine_validator import (
                get_cross_engine_validator,
                EngineId,
                ConsumptionStatus,
            )

            if not self._cross_validator_initialized:
                self._cross_validator_initialized = True
            _cev = get_cross_engine_validator()
            for _eid in (
                EngineId.ACFN,
                EngineId.SCME,
                EngineId.DEEP_VISIBILITY,
                EngineId.FRESHNESS_DECAY,
                EngineId.GOSSIP_RITUAL,
                EngineId.SAFETY_FUSION,
                EngineId.MEMORY_GOVERNANCE,
                EngineId.PLANNER_INJECTION,
                EngineId.NIGHT_CYCLE,
                EngineId.METABOLISM,
                EngineId.EMOTION_DRIVEN,
                EngineId.GROUP_PATTERN,
                EngineId.INNER_NARRATION,
            ):
                _cev.register_engine(_eid)
            if self._cached_deep_visibility_results:
                for _r in self._cached_deep_visibility_results[-10:]:
                    _cev.record_output(
                        EngineId.DEEP_VISIBILITY, "visibility_result"
                    )
            if self._cached_safety_assessment:
                _cev.record_output(
                    EngineId.SAFETY_FUSION, "safety_result", is_critical=True
                )
            if self._cached_gossip_ritual_verdict:
                _cev.record_output(EngineId.GOSSIP_RITUAL, "gossip_verdict")
            if self._cached_memory_governance_snap:
                _cev.record_output(
                    EngineId.MEMORY_GOVERNANCE, "governance_snapshot"
                )
            if self._cached_planner_injection:
                _cev.record_output(
                    EngineId.PLANNER_INJECTION, "injection_packet"
                )
            _cev.mark_consumed(
                EngineId.DEEP_VISIBILITY,
                "visibility_result",
                consumer="arbiter_chain",
                status=ConsumptionStatus.CONSUMED_BY_ARBITER,
            )
            _cev.mark_consumed(
                EngineId.SAFETY_FUSION,
                "safety_result",
                consumer="arbiter_chain",
                status=ConsumptionStatus.CONSUMED_BY_SAFETY,
            )
            _report = _cev.validate()
            return _report.to_dict()
        except Exception as exc:
            logger.debug(f"{self.log_prefix} [GAP-T] 跨引擎验证异常: {exc}")
            return {"error": str(exc)[:60]}

import asyncio
import random
from dataclasses import dataclass, field
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set
from src.chat.heart_flow.heartFC_chat import HeartFChatting
from src.chat.heart_flow.enhanced_modules.interaction_core_mixin import EnhancedInteractionCoreMixin
from src.chat.heart_flow.enhanced_modules.subjective_runtime_mixin import EnhancedSubjectiveRuntimeMixin
from src.chat.heart_flow.enhanced_modules.loop_flow_mixin import EnhancedLoopFlowMixin
from src.chat.heart_flow.enhanced_modules.voice_pipeline_mixin import EnhancedVoicePipelineMixin
from src.chat.heart_flow.enhanced_modules.proactive_execution_mixin import EnhancedProactiveExecutionMixin
from src.chat.heart_flow.enhanced_modules.strategy_integration_mixin import EnhancedStrategyIntegrationMixin
from src.chat.heart_flow.enhanced_modules.scene_analysis_mixin import EnhancedSceneAnalysisMixin
from src.core.watch_state_machine import WatchLevel
from src.common.logger import get_logger
from src.common.data_models.heartflow_models import UnifiedFlowSnapshot

if TYPE_CHECKING:
    pass

logger = get_logger("心流增强")

# 独立随机数生成器实例，避免 random.seed() 污染全局状态
_rng = random.Random()


@dataclass
class BehaviorGovernorVerdict:
    """统一行为裁决输出，避免主动/窥屏/引用各自拼零散布尔值。"""

    reply_mode: str = "observe"
    interrupt_level: str = "ignore"
    quote_policy: str = "none"
    target_user_id: str = ""
    silence_policy: str = "silent"
    max_watch_rank: int = 4
    allow_generation: bool = True
    model_tier: str = "small"
    reason_codes: List[str] = field(default_factory=list)


@dataclass
class RestGovernorVerdict:
    """统一休息/摸鱼/打断许可裁决，避免不同入口各自解释“该歇一下”。"""

    posture: str = "active"
    interruption_policy: str = "allow"
    should_rest: bool = False
    should_loaf: bool = False
    reason_codes: List[str] = field(default_factory=list)


@dataclass
class ModelGovernorVerdict:
    """模型路由裁决：统一 small/large/skip 与限流回退。"""

    tier: str = "skip"
    upgrade_reason_codes: List[str] = field(default_factory=list)
    rate_limited: bool = False
    fallback_to_small: bool = False
    dynamic_cooldown_sec: float = 0.0
    dynamic_hourly_cap: int = 0


_WATCH_LEVEL_BY_RANK = {
    0: WatchLevel.BLACKOUT,
    1: WatchLevel.PEEK,
    2: WatchLevel.SKIM_WINDOW,
    3: WatchLevel.ACTIVE_WATCH,
    4: WatchLevel.ENGAGED,
}


def _rt_float(key: str, default: float) -> float:
    """从 runtime_tuning 读取浮点参数，失败时返回默认值"""
    try:
        from src.config.core_config_engine import CoreSettingsHub

        val = CoreSettingsHub().runtime_tuning_block().get(key, default)
        return float(val)
    except Exception as exc:
        logger.debug(f"读取运行时配置 {key} 失败: {exc}")
        return default


def _task_provider_timeout(task_name: str, default: float = 10.0) -> float:
    """读取某个任务绑定模型的 provider 超时上限。"""
    try:
        from src.config.config import model_config

        task_cfg = getattr(model_config.model_task_config, task_name, None)
        provider_timeouts: list[float] = []
        for model_name in getattr(task_cfg, "model_list", []) or []:
            model_info = model_config.get_model_info(model_name)
            provider = model_config.get_provider(model_info.api_provider)
            provider_timeouts.append(float(getattr(provider, "timeout", default) or default))
        if provider_timeouts:
            return max(provider_timeouts)
    except Exception as exc:
        logger.debug(f"读取任务 {task_name} provider 超时失败: {exc}")
    return default


def _parallel_stage_timeout() -> float:
    default_timeout = max(
        20.0,
        _task_provider_timeout("focus_chat") + 10.0,
        _task_provider_timeout("utils") + 10.0,
    )
    return _rt_float("heartfc_parallel_stage_timeout_seconds", default_timeout)


def _llm_upgrade_timeout() -> float:
    default_timeout = max(15.0, _task_provider_timeout("lightweight") + 10.0)
    return _rt_float("heartfc_llm_upgrade_timeout_seconds", default_timeout)


# 增强循环内部间隔常量（优先从配置读取）
_TICK_FLOOR_SEC = _rt_float("heartfc_tick_floor_sec", 0.15)
_DORMANT_POLL_SEC = _rt_float("heartfc_dormant_poll_sec", 1.0)
_PERCEPTION_COOLDOWN_SEC = _rt_float("heartfc_perception_cooldown_sec", 5.0)
_VOICE_COOLDOWN_SEC = _rt_float("heartfc_voice_cooldown_sec", 6.0)
_ENERGY_DRAIN_FLOOR = _rt_float("heartfc_energy_drain_floor", 0.08)
_POST_MESSAGE_RETRY_SEC = _rt_float("heartfc_post_message_retry_sec", 2.0)


class EnhancedHeartFChatting(
    EnhancedInteractionCoreMixin,
    EnhancedSubjectiveRuntimeMixin,
    EnhancedLoopFlowMixin,
    EnhancedVoicePipelineMixin,
    EnhancedProactiveExecutionMixin,
    EnhancedStrategyIntegrationMixin,
    EnhancedSceneAnalysisMixin,
    HeartFChatting,
):
    """群聊增强版心流控制器

    在基础循环之上叠加增强判定管线：
    阶段协调 → 休息/等待门控 → 能量评估 → 感知引擎 → 内心独白 →
    主行动判定与执行。

    所有子系统通过懒导入获取单例，不在构造时预加载，
    保证与 heartflow.py 中心化调度器的启停顺序兼容。
    """

    def __init__(self, chat_id: str):
        super().__init__(chat_id)
        self._background_tasks: set = set()
        self._dedup_order: deque = deque(maxlen=3000)
        self._tick_world_snapshot = None
        # 阶段协调器是否已将此频道注册
        self._phase_registered = False
        # 增强循环轮次计数器（与父类 _cycle_counter 独立）
        self._enhanced_round = 0
        # 上一次执行感知引擎的时间戳
        self._last_perception_ts = 0.0
        # 上一次执行内心独白的时间戳
        self._last_voice_ts = 0.0
        # 主动路径独立的voice冷却计时器（R3修复：避免被动路径冲突）
        self._last_proactive_voice_ts = 0.0
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
            self._backend_status_log_enabled = bool(proactive_cfg.get("backend_status_log_enabled", False))
            self._backend_status_interval = float(proactive_cfg.get("backend_status_log_interval_sec", 10.0))
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
        self._last_reactive_plan = None
        # 增强管线是否就绪（延迟到首次 start 后才设为 True）
        self._pipeline_ready = False
        # 连续发言计数（用于消耗计算）
        self._consecutive_speaks = 0.0
        self._peek_mode_active = False
        self._defense_mode_active = False
        self._pattern_forced_observe = False
        self._cached_is_pinged = False
        self._cached_targeted_to_bot = False
        self._last_repeat_detected = False
        self._overnight_pressure_total = 0.0
        self._night_cycle = None
        # 上次发言时间（用于频率因子）
        self._last_speak_time = 0.0
        # 话痨惩罚值
        self._chatterbox_penalty = 0.0
        # 上次兴趣等级
        self._last_interest_level = 0.5
        self._last_desire_level = 5.0
        self._chat_energy = 100.0
        self._thinking_energy = 100.0
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
        self._last_decision_incoming_messages: List[Any] = []
        self._last_decision_context_messages: List[Any] = []
        self._last_decision_self_messages: List[Any] = []
        # 主动通道启动保护：避免刚开机就把"无上下文"误判成可主动聊
        self._proactive_start_ts = 0.0
        self._proactive_startup_grace_until = 0.0
        # 连续主动发言但没人接话的次数
        self._unanswered_bot_turns = 0
        self._last_proactive_target_user_id = ""
        # 最近一次 bot 发言时间
        self._last_bot_reply_ts = 0.0
        self._proactive_reply_timeline: List[float] = []
        self._reactive_reply_timeline: List[float] = []
        self._proactive_reflect_depth = 0
        # 用户消息累积追踪器（用于动态比例控制）
        self._user_msg_counter = 0
        self._bot_reply_counter = 0
        self._user_msg_timeline: List[float] = []
        self._recent_user_timestamps: List[float] = []
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
        self._last_night_sup_log = -1.0
        self._night_soft_wake_info: Optional[dict] = None
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
        self._last_group_context_signal: Dict[str, Any] = {}
        self._cached_pattern_evidence: List = []
        self._cached_narration_plan = None
        self._cached_presence_verdict: Dict[str, Any] = {}
        self._cached_metabolism_constraints: Dict[str, Any] = {}
        self._cached_self_state = None
        self._cached_emotion_state = None
        self._last_flow_decision_summary: Dict[str, Any] = {}
        self._last_decision_trace: Dict[str, Any] = {}
        self._last_behavior_governor_verdict = BehaviorGovernorVerdict()
        self._last_rest_governor_verdict = RestGovernorVerdict()
        self._last_model_governor_verdict = ModelGovernorVerdict()
        self._model_large_last_ts: float = 0.0
        self._model_large_hour_window_start: float = 0.0
        self._model_large_proactive_hour_calls: int = 0
        self._last_flow_blocker: str = ""
        self._last_unified_flow_snapshot: Optional[UnifiedFlowSnapshot] = None
        self._cached_ritual_behavior = None
        self._is_admin_forced = False
        # 灵魂引擎: 最近N条回复文本用于自我去重检测
        self._recent_reply_texts: list = []
        self._max_reply_history = 12
        # 灵魂引擎: 梗/网络用语注入缓存(避免每次都查LearningHub)
        self._meme_inject_cache_ts = 0.0
        self._meme_inject_cache = ""
        self._meme_inject_ttl = 180.0
        # 灵魂引擎: 当前情绪化回复模式覆盖(由外部事件设置)
        self._soul_mood_overlay = ""
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
        self._hourly_proactive_reply_count: int = 0
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
        self._pending_user_ids: set = set()
        self._group_activity_level: float = 0.0
        # 统一状态仪表盘缓存（替代40+个分散_cached_的直接读取）
        self._cached_dashboard_snapshot: Optional[Dict[str, Any]] = None
        self._dashboard_initialized: bool = False
        self._last_dashboard_build_ts: float = 0.0
        self._dashboard_ttl_sec: float = 3.0
        # 双通道架构：主动感兴趣通道（后台常驻任务）
        self._proactive_task: Optional[asyncio.Task] = None
        self._proactive_running: bool = False
        self._proactive_send_lock: Optional[asyncio.Lock] = None
        self._last_dashboard_status_line: str = ""
        # 消息去重：已处理过的消息ID集合（防止同一消息被多次处理）
        self._processed_message_ids: Set[str] = set()
        # 统一引擎编排器（懒加载）
        self._engine_orch = None



































    # 行为Governor中文映射
    _BG_REPLY_MODE_CN = {
        "observe": "观察",
        "reply": "回复",
        "proactive": "主动",
        "rest": "休息",
        "defer": "推迟",
    }
    _BG_INTERRUPT_LEVEL_CN = {
        "ignore": "忽略",
        "peek": "窥屏",
        "skim": "浏览",
        "engage": "参与",
    }
    _BG_QUOTE_POLICY_CN = {
        "none": "无",
        "quote_reply": "引用回复",
        "soft_reference": "软引用",
    }
    _BG_SILENCE_POLICY_CN = {
        "silent": "静默",
        "ambient": "环境",
        "reengage": "重新参与",
    }
    _BG_MODEL_TIER_CN = {
        "skip": "跳过",
        "small": "小模型",
        "large": "大模型",
    }


    # 休息Governor中文映射
    _RG_POSTURE_CN = {
        "active": "活跃",
        "resting": "休息中",
        "loafing": "摸鱼中",
    }
    _RG_INTERRUPTION_POLICY_CN = {
        "allow": "允许",
        "defer": "推迟",
        "block": "阻止",
    }


    # 模型Governor中文映射
    _MG_TIER_CN = {
        "skip": "跳过",
        "small": "小模型",
        "large": "大模型",
    }



















    # ────────────────── 生命周期 ──────────────────



    # ────────────────── 阶段协调器注册 ──────────────────


    # ────────────────── 内心独白引擎清理 ──────────────────



    # ═══════════════════════════════════════════════════
    #  核心模块集成桥接层
    # ═══════════════════════════════════════════════════




















































    # ═══════════════════════════════════════════════════
    #  核心重写：增强版主循环体
    # ═══════════════════════════════════════════════════





    # ═══════════════════════════════════════════════════
    #  阶段 0：状态机门控
    # ═══════════════════════════════════════════════════


















    # ═══════════════════════════════════════════════════
    #  阶段 1：能量评估
    # ═══════════════════════════════════════════════════

















    # ═══════════════════════════════════════════════════
    #  四层状态栏：核心可观察性机制
    # ═══════════════════════════════════════════════════








    # ═══════════════════════════════════════════════════
    #  阶段 2 辅助：动态阈值与 @检测
    # ═══════════════════════════════════════════════════






    # ═══════════════════════════════════════════════════
    #  阶段 3：感知引擎
    # ═══════════════════════════════════════════════════







    # ═══════════════════════════════════════════════════
    #  阶段 4：内心独白
    # ═══════════════════════════════════════════════════


















    # ═══════════════════════════════════════════════════
    #  阶段 4.5b：多维信号融合判断（heartflow_decision）
    # ═══════════════════════════════════════════════════


    # ═══════════════════════════════════════════════════
    #  阶段 4.6：早期退出检查
    # ═══════════════════════════════════════════════════












    # [已废弃 2026-04-06] _delegate_readiness_evaluation() 已移除
    # 原方法导入不存在的 readiness_evaluator 模块，功能已由 DecisionGateway 完全替代
    # 决策入口统一走 DecisionGateway.evaluate() 六层决策引擎
    # 历史方法体约184行(L9928-L10110)，含 ReadinessInput 组装 + evaluator.evaluate() 调用

    # [已废弃 2026-04-06] _apply_readiness_constraint() 已移除
    # 原方法处理约束命中后的冷却记录，已迁移至 DecisionGateway 内部处理
    # 约束逻辑现在由 D6 EnergyChainDimension.vote() 的 force_refuse 机制承载

    # ═══════════════════════════════════════════════════
    #  阶段 4.8：主动回复执行 - 跳过规划器直接生成回复
    # ═══════════════════════════════════════════════════


    # ═══════════════════════════════════════════════════
    #  双通道架构：主动感兴趣通道（后台常驻）
    # ═══════════════════════════════════════════════════


    # ── 空闲主动行为路径（保留作为主循环无消息时的快速路径） ──

    _IDLE_PROACTIVE_COOLDOWN_SEC = 120.0









    # ═══════════════════════════════════════════════════
    #  阶段 5：行动决策融合
    # ═══════════════════════════════════════════════════


    # ═══════════════════════════════════════════════════
    #  状态迁移辅助
    # ═══════════════════════════════════════════════════




    # ═══════════════════════════════════════════════════
    #  行为分析辅助
    # ═══════════════════════════════════════════════════




    # ═══════════════════════════════════════════════════
    #  氛围反馈
    # ═══════════════════════════════════════════════════








































    # ═══════════════════════════════════════════════════
    #  核心系统集成方法 - 真正对接各个模块
    # ═══════════════════════════════════════════════════



























    # ═══════════════════════════════════════════════════
    #  多维独立状态系统 - 决策网关集成
    # ═══════════════════════════════════════════════════















    _SUMMARY_MIN_EVENTS_FOR_PIPELINE = 3







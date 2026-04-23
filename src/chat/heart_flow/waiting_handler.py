import time
from typing import Any, Dict, Optional
from src.common.logger import get_logger
from src.config.core_config_engine import get_core_config
from src.common.data_models.heartflow_models import IdleSetup, IdleDecision

logger = get_logger("pending_orchestrator")

# 规划器动作 → 等待生命周期映射
_PLANNER_DISPOSITION_MAP: Dict[str, str] = {
    "wait": "enter_pending",
    "check_later": "cooldown_only",
    "observe": "cooldown_only",
    "rest": "cooldown_only",
    "no_reply": "cooldown_only",
    "complete_talk": "cooldown_only",
    "listening": "enter_pending",
}


def classify_planner_disposition(planner_action: str) -> str:
    """将规划器动作分类为 enter_pending / cooldown_only / active"""
    action_lower = planner_action.strip().lower()
    return _PLANNER_DISPOSITION_MAP.get(action_lower, "active")


class PendingOrchestrator:
    """
    唯一等待权威 — 全局等待生命周期管理器
    所有"是否在等"的判断必须经过本类，禁止其他模块独立维护等待状态。
    频道进入 PENDING 后，按时间窗与计数器决定下一步动作。
    纯算法驱动，无关键词匹配。
    """

    _solo: Optional["PendingOrchestrator"] = None

    @classmethod
    def instance(cls) -> "PendingOrchestrator":
        if cls._solo is None:
            cls._solo = cls()
        return cls._solo

    @classmethod
    def teardown(cls) -> None:
        if cls._solo is not None:
            cls._solo._channel_setups.clear()
        cls._solo = None

    def __init__(self):
        self._channel_setups: Dict[str, IdleSetup] = {}

    def _timing_cfg(self) -> Dict[str, Any]:
        hub = get_core_config()
        return hub.resolve_module_view("phase_timing").values

    def initiate_watch(
        self, channel_id: str, anticipated: str = ""
    ) -> IdleSetup:
        """开始等待监视窗口"""
        cfg = self._timing_cfg()
        setup = IdleSetup(
            anticipated_response=anticipated,
            max_wait_sec=int(cfg.get("max_idle_wait_seconds", 120)),
            entered_at=time.time(),
            last_contemplate_at=0.0,
            contemplate_rounds=0,
            chase_rounds=0,
            max_chase_allowed=int(cfg.get("max_chase_count", 2)),
        )
        self._channel_setups[channel_id] = setup
        logger.info(
            f"频道 {channel_id[:8]} 开始等待，上限={setup.max_wait_sec}秒"
            f' 预期回复="{anticipated[:30]}"'
            if anticipated
            else ""
        )
        return setup

    def fetch_setup(self, channel_id: str) -> Optional[IdleSetup]:
        return self._channel_setups.get(channel_id)

    def evaluate_next(self, channel_id: str) -> IdleDecision:
        """
        评估当前时刻应采取的动作。
        决策树：超时? -> 追问上限? -> 可沉思? -> 保持等待
        """
        setup = self._channel_setups.get(channel_id)
        if setup is None:
            return IdleDecision.WITHDRAW
        if setup.is_timed_out():
            logger.info(
                f"频道 {channel_id[:8]} 等待超时，已过 {setup.elapsed_sec():.0f}秒"
            )
            return IdleDecision.EXPIRED
        if not setup.can_chase():
            progress = setup.wait_progress()
            if progress >= 0.8:
                return IdleDecision.EXPIRED
        cfg = self._timing_cfg()
        contemplate_interval = int(cfg.get("reflect_interval_seconds", 30))
        if contemplate_interval > 0 and setup.can_contemplate(
            contemplate_interval
        ):
            return IdleDecision.CONTEMPLATE
        if setup.can_chase():
            progress = setup.wait_progress()
            if progress >= 0.5:
                return IdleDecision.CHASE
        return IdleDecision.HOLD

    def mark_pondered(self, channel_id: str) -> None:
        """记录一次沉思"""
        setup = self._channel_setups.get(channel_id)
        if setup:
            setup.record_contemplate()
            logger.debug(
                f"频道 {channel_id[:8]} 沉思第{setup.contemplate_rounds}轮"
            )

    def mark_pursued(self, channel_id: str) -> None:
        """记录一次追问"""
        setup = self._channel_setups.get(channel_id)
        if setup:
            setup.record_chase()
            logger.debug(f"频道 {channel_id[:8]} 追问第{setup.chase_rounds}轮")

    def conclude_watch(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """结束等待窗口，返回摘要后清除"""
        setup = self._channel_setups.pop(channel_id, None)
        if setup is None:
            return None
        summary = {
            "elapsed_sec": round(setup.elapsed_sec(), 1),
            "contemplate_rounds": setup.contemplate_rounds,
            "chase_rounds": setup.chase_rounds,
            "timed_out": setup.is_timed_out(),
        }
        logger.info(f"频道 {channel_id[:8]} 等待结束: {summary}")
        return summary

    def conclude_with_timeout_consequence(
        self, channel_id: str
    ) -> Optional[Dict[str, Any]]:
        """结束等待窗口并在超时时自动回写后果"""
        summary = self.conclude_watch(channel_id)
        if summary is None:
            return None
        if not summary.get("timed_out", False):
            return summary
        # 回写连续超时计数到 FlowStateContainer
        self._bump_consecutive_timeouts(channel_id)
        # 回写意图池失败
        self._fail_wait_intents_on_timeout(channel_id)
        # 注入失落情绪
        self._inject_timeout_emotion(channel_id, summary["elapsed_sec"])
        logger.info(
            f"频道 {channel_id[:8]} 超时后果已回写: "
            f"elapsed={summary['elapsed_sec']:.0f}s "
            f"contemplate={summary['contemplate_rounds']} chase={summary['chase_rounds']}"
        )
        return summary

    def _bump_consecutive_timeouts(self, channel_id: str) -> None:
        """递增 FlowStateContainer 的连续超时计数"""
        try:
            from src.chat.heart_flow.state_machine import get_phase_coordinator

            coordinator = get_phase_coordinator()
            container = coordinator.get_container(channel_id)
            if container:
                container.consecutive_timeouts += 1
                logger.debug(
                    f"频道 {channel_id[:8]} 连续超时={container.consecutive_timeouts}"
                )
        except Exception as exc:
            logger.debug(f"更新连续超时失败: {exc}")

    def _fail_wait_intents_on_timeout(self, channel_id: str) -> None:
        """超时时将等回复类意图标记失败"""
        try:
            from src.chat.proactive.intention_pool import (
                get_intention_pool,
                IntentKind,
                normalize_intent_failure_reason,
            )

            pool = get_intention_pool()
            alive = pool.get_all_alive(channel_id)
            reason_tag = normalize_intent_failure_reason("timeout")
            closed = 0
            for intent in alive:
                if intent.kind in (
                    IntentKind.WAIT_FOR_REPLY,
                    IntentKind.REENGAGE_USER,
                ):
                    intent.mark_failed(reason_tag)
                    closed += 1
            if closed:
                logger.debug(
                    f"频道 {channel_id[:8]} 超时关闭{closed}个等待意图"
                )
        except Exception as exc:
            logger.debug(f"超时意图回写失败: {exc}")

    def _inject_timeout_emotion(
        self, channel_id: str, elapsed_sec: float
    ) -> None:
        """超时后注入轻微失落情绪"""
        try:
            from src.modules.modcore.psychological_core import (
                get_psychological_core,
            )

            core = get_psychological_core()
            # 等待越久失落越深，上限0.3
            intensity = min(0.3, elapsed_sec / 600.0 + 0.05)
            core.inject_emotion_event(
                channel_id=channel_id,
                emotion_name="sadness",
                intensity=intensity,
                source=f"等待超时({elapsed_sec:.0f}s)",
            )
        except Exception as exc:
            logger.debug(f"超时情绪注入失败: {exc}")

    def is_watching(self, channel_id: str) -> bool:
        return channel_id in self._channel_setups

    def watch_progress(self, channel_id: str) -> float:
        setup = self._channel_setups.get(channel_id)
        if setup is None:
            return 0.0
        return setup.wait_progress()

    def watch_elapsed_sec(self, channel_id: str) -> float:
        setup = self._channel_setups.get(channel_id)
        if setup is None:
            return 0.0
        return setup.elapsed_sec()

    def snapshot(self, channel_id: str) -> Dict[str, Any]:
        setup = self._channel_setups.get(channel_id)
        if setup is None:
            return {"active": False}
        return {
            "active": True,
            "elapsed_sec": round(setup.elapsed_sec(), 1),
            "progress": round(setup.wait_progress(), 2),
            "contemplate_rounds": setup.contemplate_rounds,
            "chase_rounds": setup.chase_rounds,
            "timed_out": setup.is_timed_out(),
            "anticipated": (
                setup.anticipated_response[:50]
                if setup.anticipated_response
                else ""
            ),
        }

    def all_watching_channels(self) -> list:
        return list(self._channel_setups.keys())

    def abort_all(self) -> int:
        """关闭时批量结束所有等待窗口，返回清理数量"""
        count = len(self._channel_setups)
        for cid in list(self._channel_setups.keys()):
            self.conclude_watch(cid)
        return count


def get_pending_orchestrator() -> PendingOrchestrator:
    """获取全局等待编排器"""
    return PendingOrchestrator.instance()

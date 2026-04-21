import time
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from src.common.logger import get_logger

logger = get_logger("意图池")


def normalize_intent_failure_reason(reason: str) -> str:
    """将失败原因归一到可统计标签。"""
    raw = str(reason or "").strip().lower()
    if not raw:
        return "unknown"
    if any(
        token in raw
        for token in ("ignored", "被忽视", "无回应", "no_response", "timeout")
    ):
        return "ignored"
    if any(token in raw for token in ("negative", "负面", "敌意", "冲突")):
        return "negative"
    if any(token in raw for token in ("interrupt", "被打断", "interrupted")):
        return "interrupted"
    if "no_action" in raw:
        return "no_action"
    if "no_reply" in raw:
        return "no_reply"
    return "other"


# ---------------------------------------------------------------------------
#  意图类型与状态
# ---------------------------------------------------------------------------


class IntentKind(Enum):
    """意图类型"""

    CONTINUE_TOPIC = "continue_topic"
    WAIT_FOR_REPLY = "wait_for_reply"
    REENGAGE_USER = "reengage_user"
    SELF_EXPRESSION = "self_expression"
    FOLLOWUP_QUESTION = "followup_question"
    CLARIFY_MISUNDERSTANDING = "clarify_misunderstanding"
    SHARE_RELATED = "share_related"
    COMFORT_USER = "comfort_user"

    def __str__(self) -> str:
        return self.value


class IntentLifecycle(Enum):
    """意图生命周期状态"""

    ACTIVE = "active"
    WAITING = "waiting"
    FROZEN = "frozen"
    COMPLETED = "completed"
    ABANDONED = "abandoned"

    def is_alive(self) -> bool:
        return self in (IntentLifecycle.ACTIVE, IntentLifecycle.WAITING)

    def is_terminal(self) -> bool:
        return self in (IntentLifecycle.COMPLETED, IntentLifecycle.ABANDONED)

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
#  意图数据模型
# ---------------------------------------------------------------------------


@dataclass
class Intention:
    """单个持久意图"""

    intent_id: str = ""
    kind: IntentKind = IntentKind.CONTINUE_TOPIC
    target_user: str = ""
    channel_id: str = ""
    source: str = ""
    description: str = ""
    urgency: float = 0.5
    created_at: float = 0.0
    last_activated: float = 0.0
    attempt_count: int = 0
    fail_count: int = 0
    lifecycle: IntentLifecycle = IntentLifecycle.ACTIVE
    context_snippet: str = ""
    expected_outcome: str = ""

    def age_seconds(self) -> float:
        if self.created_at <= 0:
            return 0.0
        return time.time() - self.created_at

    def idle_seconds(self) -> float:
        ref = (
            self.last_activated if self.last_activated > 0 else self.created_at
        )
        if ref <= 0:
            return 0.0
        return time.time() - ref

    def effective_urgency(self) -> float:
        """计算有效紧迫度：随时间衰减 + 失败惩罚"""
        base = self.urgency
        age_hours = self.age_seconds() / 3600.0
        decay = max(0.05, 1.0 - age_hours * 0.08)
        fail_penalty = self.fail_count * 0.15
        return max(0.0, min(1.0, base * decay - fail_penalty))

    def mark_attempted(self) -> None:
        self.attempt_count += 1
        self.last_activated = time.time()

    def mark_failed(self, reason: str = "") -> None:
        reason_tag = normalize_intent_failure_reason(reason)
        self.fail_count += 1
        self.last_activated = time.time()
        logger.debug(f"[意图] {self.intent_id[:8]} 失败标签={reason_tag}")
        if self.fail_count >= 3:
            self.lifecycle = IntentLifecycle.FROZEN
            logger.debug(
                f"[意图] {self.intent_id[:8]} 连续失败{self.fail_count}次，冻结"
            )

    def mark_completed(self) -> None:
        self.lifecycle = IntentLifecycle.COMPLETED
        self.last_activated = time.time()

    def mark_abandoned(self, reason: str = "") -> None:
        self.lifecycle = IntentLifecycle.ABANDONED
        self.last_activated = time.time()
        logger.debug(f"[意图] {self.intent_id[:8]} 放弃: {reason}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "kind": str(self.kind),
            "target_user": self.target_user,
            "channel_id": self.channel_id,
            "source": self.source,
            "description": self.description,
            "urgency": self.urgency,
            "created_at": self.created_at,
            "last_activated": self.last_activated,
            "attempt_count": self.attempt_count,
            "fail_count": self.fail_count,
            "lifecycle": str(self.lifecycle),
            "context_snippet": self.context_snippet,
            "expected_outcome": self.expected_outcome,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Intention":
        kind_raw = data.get("kind", "continue_topic")
        try:
            kind = IntentKind(kind_raw)
        except ValueError:
            kind = IntentKind.CONTINUE_TOPIC
        lc_raw = data.get("lifecycle", "active")
        try:
            lifecycle = IntentLifecycle(lc_raw)
        except ValueError:
            lifecycle = IntentLifecycle.ACTIVE
        return cls(
            intent_id=data.get("intent_id", ""),
            kind=kind,
            target_user=data.get("target_user", ""),
            channel_id=data.get("channel_id", ""),
            source=data.get("source", ""),
            description=data.get("description", ""),
            urgency=float(data.get("urgency", 0.5)),
            created_at=float(data.get("created_at", 0.0)),
            last_activated=float(data.get("last_activated", 0.0)),
            attempt_count=int(data.get("attempt_count", 0)),
            fail_count=int(data.get("fail_count", 0)),
            lifecycle=lifecycle,
            context_snippet=data.get("context_snippet", ""),
            expected_outcome=data.get("expected_outcome", ""),
        )


# ---------------------------------------------------------------------------
#  意图竞争选择
# ---------------------------------------------------------------------------


def _select_top_intentions(
    pool: List[Intention],
    channel_id: str,
    limit: int = 3,
) -> List[Intention]:
    """从活跃意图中按有效紧迫度选出最强的 N 个"""
    alive = [
        i
        for i in pool
        if i.lifecycle.is_alive() and i.channel_id == channel_id
    ]
    alive.sort(key=lambda i: i.effective_urgency(), reverse=True)
    return alive[:limit]


# ---------------------------------------------------------------------------
#  意图池管理器
# ---------------------------------------------------------------------------

_ID_COUNTER = 0


def _generate_intent_id() -> str:
    global _ID_COUNTER
    _ID_COUNTER += 1
    return f"int_{int(time.time())}_{_ID_COUNTER:04d}"


class IntentionPoolManager:
    """跨轮次意图池管理器
    每个频道维护一个意图列表，支持：
    - 写入新意图（来自 inner_voice / 后果学习 / 等待系统）
    - 读取活跃意图（供 planner 决策）
    - 更新意图状态（完成 / 失败 / 放弃）
    - 自动清理过期意图
    """

    _solo: Optional["IntentionPoolManager"] = None

    @classmethod
    def instance(cls) -> "IntentionPoolManager":
        if cls._solo is None:
            cls._solo = cls()
        return cls._solo

    @classmethod
    def teardown(cls) -> None:
        cls._solo = None

    def __init__(self):
        self._pool: Dict[str, List[Intention]] = {}
        self._max_per_channel: int = 20
        self._max_age_hours: float = 4.0
        logger.info("[意图池] 初始化完成")

    # ---- 写入 ----

    def submit(
        self,
        channel_id: str,
        kind: IntentKind,
        target_user: str = "",
        source: str = "",
        description: str = "",
        urgency: float = 0.5,
        context_snippet: str = "",
        expected_outcome: str = "",
    ) -> Intention:
        """提交新意图到池中"""
        # 去重：同类型 + 同目标 + 活跃状态的意图不重复提交
        existing = self._pool.get(channel_id, [])
        for intent in existing:
            if (
                intent.lifecycle.is_alive()
                and intent.kind == kind
                and intent.target_user == target_user
                and intent.description == description
            ):
                intent.urgency = max(intent.urgency, urgency)
                intent.last_activated = time.time()
                logger.debug(
                    f"[意图池] 合并已有意图 {intent.intent_id[:8]} urgency→{intent.urgency:.2f}"
                )
                return intent
        intent = Intention(
            intent_id=_generate_intent_id(),
            kind=kind,
            target_user=target_user,
            channel_id=channel_id,
            source=source,
            description=description,
            urgency=max(0.0, min(1.0, urgency)),
            created_at=time.time(),
            last_activated=time.time(),
            context_snippet=context_snippet[:300],
            expected_outcome=expected_outcome[:200],
        )
        if channel_id not in self._pool:
            self._pool[channel_id] = []
        self._pool[channel_id].append(intent)
        self._enforce_capacity(channel_id)
        logger.info(
            f"[意图池] 新意图 {intent.intent_id[:8]} "
            f"type={kind.value} target={target_user[:8] if target_user else 'none'} "
            f"urgency={urgency:.2f} source={source}"
        )
        return intent

    # ---- 读取 ----

    def get_active_intentions(
        self, channel_id: str, limit: int = 3
    ) -> List[Intention]:
        """获取指定频道最紧迫的活跃意图"""
        self._sweep_expired(channel_id)
        pool = self._pool.get(channel_id, [])
        return _select_top_intentions(pool, channel_id, limit)

    def get_all_alive(self, channel_id: str) -> List[Intention]:
        """获取指定频道所有活跃意图"""
        self._sweep_expired(channel_id)
        return [
            i for i in self._pool.get(channel_id, []) if i.lifecycle.is_alive()
        ]

    def count_alive(self, channel_id: str) -> int:
        return len(self.get_all_alive(channel_id))

    def find_by_target(
        self, channel_id: str, target_user: str
    ) -> List[Intention]:
        """查找针对特定用户的活跃意图"""
        return [
            i
            for i in self._pool.get(channel_id, [])
            if i.lifecycle.is_alive() and i.target_user == target_user
        ]

    # ---- 状态更新 ----

    def complete_intent(self, intent_id: str) -> bool:
        """标记意图为已完成"""
        intent = self._find_by_id(intent_id)
        if intent is None:
            return False
        intent.mark_completed()
        return True

    def fail_intent(self, intent_id: str, reason: str = "") -> bool:
        """标记意图的本次尝试失败"""
        intent = self._find_by_id(intent_id)
        if intent is None:
            return False
        normalized_reason = normalize_intent_failure_reason(reason)
        intent.mark_failed(normalized_reason)
        logger.info(
            f"[意图池] 精确失败 id={intent_id[:10]} kind={intent.kind.value} reason={normalized_reason}"
        )
        return True

    def abandon_intent(self, intent_id: str, reason: str = "") -> bool:
        """放弃意图"""
        intent = self._find_by_id(intent_id)
        if intent is None:
            return False
        intent.mark_abandoned(reason)
        return True

    def complete_by_target(
        self,
        channel_id: str,
        target_user: str,
        kind: Optional[IntentKind] = None,
        limit: int = 0,
    ) -> int:
        """完成针对特定用户的意图（用户回复后批量结案）

        limit=0 表示无上限；limit>0 表示最多关闭 limit 个（按创建时间从早到晚）
        """
        candidates = []
        for intent in self._pool.get(channel_id, []):
            if not intent.lifecycle.is_alive():
                continue
            if intent.target_user != target_user:
                continue
            if kind is not None and intent.kind != kind:
                continue
            candidates.append(intent)
        if limit > 0:
            candidates.sort(key=lambda i: getattr(i, "created_at", 0))
            candidates = candidates[:limit]
        for intent in candidates:
            intent.mark_completed()
        if candidates:
            logger.info(
                f"[意图池] 批量完成 {len(candidates)} 个针对 {target_user[:8]} 的意图"
            )
        return len(candidates)

    # ---- 从 VoiceVerdict 导入意图 ----

    def ingest_voice_verdict(
        self,
        channel_id: str,
        verdict: Any,
        speaker_id: str = "",
    ) -> List[Intention]:
        """从 inner_voice 的 VoiceVerdict 中提取意图写入池"""
        if verdict is None or not getattr(verdict, "is_valid", False):
            return []
        imported: List[Intention] = []
        raw_intents = getattr(verdict, "intents", []) or []
        thinking = str(getattr(verdict, "thinking", "") or "")
        expected = str(getattr(verdict, "expected_reaction", "") or "")
        desire = int(getattr(verdict, "reply_desire_level", 0) or 0)
        for bi in raw_intents:
            itype = str(getattr(bi, "intent_type", "") or "")
            if not itype or itype in ("observe", "idle", "ignore", "lurk", "rest", "disengage", ""):
                continue
            kind = self._map_behavior_intent_to_kind(itype)
            urgency = self._estimate_urgency(desire, bi)
            intent = self.submit(
                channel_id=channel_id,
                kind=kind,
                target_user=speaker_id,
                source="inner_voice",
                description=thinking[:200],
                urgency=urgency,
                context_snippet=str(getattr(bi, "payload", "") or "")[:200],
                expected_outcome=expected[:200],
            )
            imported.append(intent)
        # 如果 inner_voice 有高回复欲望但没有具体 intent，也创建一个
        if desire >= 7 and not imported and speaker_id:
            intent = self.submit(
                channel_id=channel_id,
                kind=IntentKind.CONTINUE_TOPIC,
                target_user=speaker_id,
                source="inner_voice_desire",
                description=thinking[:200],
                urgency=min(1.0, desire / 10.0),
                expected_outcome=expected[:200],
            )
            imported.append(intent)
        return imported

    # ---- 为 planner 提供意图摘要 ----

    def build_planner_hint(self, channel_id: str) -> str:
        """构建供 planner 使用的意图摘要文本"""
        active = self.get_active_intentions(channel_id, limit=3)
        if not active:
            return ""
        lines = []
        for intent in active:
            age_min = intent.age_seconds() / 60.0
            lines.append(
                f"- [{intent.kind.value}] "
                f"urgency={intent.effective_urgency():.2f} "
                f"target={intent.target_user[:8] if intent.target_user else 'any'} "
                f"age={age_min:.0f}min "
                f"desc={intent.description[:60]}"
            )
        return "活跃意图:\n" + "\n".join(lines)

    def build_planner_intent_payload(
        self, channel_id: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """构建供 planner 使用的结构化活跃意图载荷。"""
        active = self.get_active_intentions(channel_id, limit=limit)
        payload: List[Dict[str, Any]] = []
        for intent in active:
            payload.append(
                {
                    "intent_id": intent.intent_id,
                    "kind": intent.kind.value,
                    "target_user": intent.target_user,
                    "source": intent.source,
                    "description": intent.description[:120],
                    "expected_outcome": intent.expected_outcome[:120],
                    "effective_urgency": round(
                        float(intent.effective_urgency()), 4
                    ),
                    "age_seconds": round(float(intent.age_seconds()), 2),
                    "idle_seconds": round(float(intent.idle_seconds()), 2),
                    "attempt_count": int(intent.attempt_count),
                    "fail_count": int(intent.fail_count),
                    "lifecycle": intent.lifecycle.value,
                }
            )
        return payload

    def build_arbiter_signal(self, channel_id: str) -> float:
        """为 ProactiveArbiter 提供意图驱动的信号强度（0~1）"""
        active = self.get_active_intentions(channel_id, limit=5)
        if not active:
            return 0.0
        max_urgency = max(i.effective_urgency() for i in active)
        count_bonus = min(0.2, len(active) * 0.05)
        return min(1.0, max_urgency + count_bonus)

    # ---- 统计 ----

    def snapshot(self, channel_id: str) -> Dict[str, Any]:
        alive = self.get_all_alive(channel_id)
        return {
            "alive_count": len(alive),
            "kinds": [str(i.kind) for i in alive],
            "max_urgency": max(
                (i.effective_urgency() for i in alive), default=0.0
            ),
            "oldest_age_min": max(
                (i.age_seconds() / 60.0 for i in alive), default=0.0
            ),
        }

    # ---- 内部方法 ----

    def _find_by_id(self, intent_id: str) -> Optional[Intention]:
        for ch_pool in self._pool.values():
            for intent in ch_pool:
                if intent.intent_id == intent_id:
                    return intent
        return None

    def _enforce_capacity(self, channel_id: str) -> None:
        pool = self._pool.get(channel_id)
        if pool is None:
            return
        if len(pool) <= self._max_per_channel:
            return
        alive = [i for i in pool if i.lifecycle.is_alive()]
        dead = [i for i in pool if i.lifecycle.is_terminal()]
        if len(dead) > 5:
            pool[:] = alive + dead[-5:]
        if len(pool) > self._max_per_channel:
            pool.sort(
                key=lambda i: (
                    i.effective_urgency() if i.lifecycle.is_alive() else -1
                ),
                reverse=True,
            )
            pool[:] = pool[: self._max_per_channel]

    def _sweep_expired(self, channel_id: str) -> None:
        pool = self._pool.get(channel_id)
        if pool is None:
            return
        max_age_sec = self._max_age_hours * 3600.0
        now = time.time()
        for intent in pool:
            if not intent.lifecycle.is_alive():
                continue
            if (now - intent.created_at) > max_age_sec:
                intent.mark_abandoned("过期")

    @staticmethod
    def _map_behavior_intent_to_kind(intent_type: str) -> IntentKind:
        mapping = {
            "reply": IntentKind.CONTINUE_TOPIC,
            "miao_reply": IntentKind.CONTINUE_TOPIC,
            "followup": IntentKind.FOLLOWUP_QUESTION,
            "nudge": IntentKind.REENGAGE_USER,
            "wait": IntentKind.WAIT_FOR_REPLY,
            "sticker": IntentKind.SELF_EXPRESSION,
            "clarify": IntentKind.CLARIFY_MISUNDERSTANDING,
            "comfort": IntentKind.COMFORT_USER,
            "share": IntentKind.SHARE_RELATED,
        }
        return mapping.get(intent_type, IntentKind.CONTINUE_TOPIC)

    @staticmethod
    def _estimate_urgency(desire_level: int, behavior_intent: Any) -> float:
        base = max(0.0, min(1.0, desire_level / 10.0))
        priority = int(getattr(behavior_intent, "priority", 0) or 0)
        priority_boost = min(0.3, priority * 0.1)
        return min(1.0, base + priority_boost)


# ---------------------------------------------------------------------------
#  全局访问接口
# ---------------------------------------------------------------------------


def get_intention_pool() -> IntentionPoolManager:
    """获取全局意图池管理器实例"""
    return IntentionPoolManager.instance()

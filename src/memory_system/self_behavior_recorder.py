"""
机器人自身行为编年史 —— 记录、缓冲、检索和统计Bot执行的所有操作
融合三源设计:
  - 批量计数器 + 行为分布统计
  - 多类型行为记录 + 行为负载追踪 + Prompt生成
  - 通道级缓冲 + 模板化描述 + 自动恢复
原创实现: DB持久化替代JSON文件 + 环形缓冲器 + 负载衰变模型
"""

import time
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, Counter
from src.common.logger import get_logger

logger = get_logger("behavior_chronicle")


# ═══════════════════════════════════════════
# 行为数据容器
# ═══════════════════════════════════════════


@dataclass
class BotDeed:
    """单条机器人行为记录"""

    deed_id: str  # 唯一标识
    deed_type: str  # 行为类型
    channel_id: str  # 所属会话通道
    body: str  # 行为描述内容
    occurred_at: float = field(default_factory=time.time)
    participant_id: Optional[str] = None  # 关联用户
    annotations: Dict[str, Any] = field(default_factory=dict)
    succeeded: bool = True

    def elapsed_seconds(self) -> float:
        return time.time() - self.occurred_at

    def elapsed_label(self) -> str:
        """返回人类可读的时间差描述"""
        gap = self.elapsed_seconds()
        if gap < 60:
            return f"{int(gap)}秒前"
        if gap < 3600:
            return f"{int(gap / 60)}分钟前"
        if gap < 86400:
            return f"{int(gap / 3600)}小时前"
        return f"{int(gap / 86400)}天前"

    def as_memory_line(self) -> str:
        """转换为可注入记忆系统的文本行"""
        ts_label = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(self.occurred_at)
        )
        category_icons = {
            "utterance": "[发言]",
            "sentiment_shift": "[情绪]",
            "inference": "[推理]",
            "persona_morph": "[人格]",
            "tool_invoke": "[工具]",
            "inner_monologue": "[内省]",
            "retraction": "[撤回]",
            "query": "[搜索]",
            "moderation": "[管理]",
            "silence_impose": "[禁言]",
            "directed_reply": "[回复]",
            "image_dispatch": "[发图]",
            "sticker_dispatch": "[表情]",
            "model_call": "[模型]",
            "system_notice": "[系统]",
            "fault": "[异常]",
        }
        icon = category_icons.get(self.deed_type, f"[{self.deed_type}]")
        return f"[{ts_label}] {icon} {self.body}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deed_id": self.deed_id,
            "deed_type": self.deed_type,
            "channel_id": self.channel_id,
            "body": self.body,
            "occurred_at": self.occurred_at,
            "participant_id": self.participant_id,
            "annotations": self.annotations,
            "succeeded": self.succeeded,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BotDeed":
        return cls(
            deed_id=raw.get("deed_id", ""),
            deed_type=raw.get("deed_type", ""),
            channel_id=raw.get("channel_id", ""),
            body=raw.get("body", ""),
            occurred_at=raw.get("occurred_at", time.time()),
            participant_id=raw.get("participant_id"),
            annotations=raw.get("annotations", {}),
            succeeded=raw.get("succeeded", True),
        )


# ═══════════════════════════════════════════
# 描述模板引擎
# ═══════════════════════════════════════════


class DeedTemplateEngine:
    """根据行为类型和上下文信息生成标准化描述文本"""

    def __init__(self):
        self._registry: Dict[str, str] = {
            "retraction": "撤回了消息: {content}",
            "moderation": "将成员 {target} 移出群聊",
            "silence_impose": "对 {target} 施加了{duration}秒禁言",
            "silence_lift": "解除了 {target} 的禁言",
            "ban_apply": "封禁了成员: {target}",
            "ban_revoke": "解封了成员: {target}",
            "directed_reply": "对消息进行了回复: {content}",
            "image_dispatch": "发送了一张图片",
            "sticker_dispatch": "投递了表情包: {sticker_name}",
            "query": "执行了搜索: {query}",
            "card_update": "修改了群名片: {target} → {card}",
            "msg_purge": "清除了一条消息",
            "utterance": "发送了回复: {content}",
            "tool_invoke": "调用了工具: {tool_name}",
            "model_call": "请求了模型: {model}",
            "persona_morph": "切换了人格模式: {persona_name}",
        }

    def render(self, deed_type: str, context: Dict[str, Any]) -> str:
        """按模板填充生成描述"""
        template = self._registry.get(deed_type)
        if not template:
            return f"执行了 {deed_type} 操作"
        try:
            return template.format(**context)
        except (KeyError, IndexError):
            return f"执行了 {deed_type} 操作"

    def register_template(self, deed_type: str, template: str):
        """注册自定义模板"""
        self._registry[deed_type] = template


# ═══════════════════════════════════════════
# 通道级环形缓冲器
# ═══════════════════════════════════════════


class ChannelDeedRing:
    """
    单通道行为环形缓冲区
    保持最近N条记录，并维护频次计数器
    """

    def __init__(self, ring_size: int = 60):
        self._ring: deque = deque(maxlen=ring_size)
        self._type_counter: Counter = Counter()

    def push(self, deed: BotDeed):
        self._ring.append(deed)
        self._type_counter[deed.deed_type] += 1

    def recent(
        self, count: int = 10, within_minutes: int = 30
    ) -> List[BotDeed]:
        """获取指定时间窗口内的最新N条记录"""
        boundary = time.time() - (within_minutes * 60)
        qualified = [d for d in self._ring if d.occurred_at >= boundary]
        return qualified[-count:]

    def filter_by_type(self, deed_type: str, count: int = 5) -> List[BotDeed]:
        """按类型过滤"""
        matched = [d for d in self._ring if d.deed_type == deed_type]
        return matched[-count:]

    def find_latest(
        self, deed_type: str, within_minutes: int = 10
    ) -> Optional[BotDeed]:
        """查找指定类型的最近一条记录"""
        boundary = time.time() - (within_minutes * 60)
        for deed in reversed(self._ring):
            if deed.deed_type == deed_type and deed.occurred_at >= boundary:
                return deed
        return None

    def type_frequency(self, deed_type: str) -> int:
        return self._type_counter.get(deed_type, 0)

    def prune_expired(self, max_age_hours: int = 24):
        """清除超龄记录"""
        boundary = time.time() - (max_age_hours * 3600)
        while self._ring and self._ring[0].occurred_at < boundary:
            evicted = self._ring.popleft()
            self._type_counter[evicted.deed_type] = max(
                0, self._type_counter.get(evicted.deed_type, 1) - 1
            )

    @property
    def total(self) -> int:
        return len(self._ring)

    def distribution(self) -> Dict[str, int]:
        return dict(self._type_counter)


# ═══════════════════════════════════════════
# 负载衰变模型
# ═══════════════════════════════════════════


class FatigueDecayModel:
    """
    工具/行为负载度计算模型
    负载度随使用次数非线性增长，随时间指数衰减
    使用衰减公式:
      fatigue = 1 - exp(-k * effective_count)
    其中 effective_count 是时间加权后的有效使用次数
    """

    def __init__(
        self, decay_half_life: float = 600.0, growth_factor: float = 0.25
    ):
        self._half_life = decay_half_life  # 负载半衰期(秒)
        self._growth_k = growth_factor  # 增长系数

    def compute(self, usage_timestamps: List[float]) -> float:
        """根据使用时间戳列表计算当前负载度(0~1)"""
        import math

        now = time.time()
        if not usage_timestamps:
            return 0.0
        # 每次使用按距当前时间折算权重
        decay_lambda = math.log(2) / self._half_life
        effective_count = 0.0
        for ts in usage_timestamps:
            age = now - ts
            if age < 0:
                age = 0
            effective_count += math.exp(-decay_lambda * age)
        # 负载度: 使用logistic增长
        load_value = 1.0 - math.exp(-self._growth_k * effective_count)
        return round(min(1.0, max(0.0, load_value)), 4)


# ═══════════════════════════════════════════
# 行为编年史主类
# ═══════════════════════════════════════════

RECOGNIZED_DEED_TYPES = frozenset(
    {
        "sentiment_shift",
        "tool_invoke",
        "model_call",
        "utterance",
        "persona_morph",
        "memory_op",
        "inner_monologue",
        "fault",
        "system_notice",
        "user_input",
        "retraction",
        "moderation",
        "silence_impose",
        "silence_lift",
        "directed_reply",
        "query",
        "image_dispatch",
        "sticker_dispatch",
    }
)


class BotBehaviorChronicle:
    """
    机器人行为编年史 —— 统一记录、缓冲、持久化和查询Bot所有操作
    核心差异:
    - 使用DB持久化而非JSON文件
    - 行为负载采用指数衰变模型
    - 内置分布统计
    """

    def __init__(
        self,
        ring_capacity: int = 60,
        pending_threshold: int = 40,
        flush_gap: float = 45.0,
    ):
        self._channels: Dict[str, ChannelDeedRing] = {}
        self._ring_cap = ring_capacity
        self._template_engine = DeedTemplateEngine()
        self._load_model = FatigueDecayModel()
        # 待刷新队列(两阶段: 缓冲→持久化)
        self._pending_deeds: List[BotDeed] = []
        self._pending_cap = pending_threshold
        self._last_persist_ts = time.time()
        self._persist_gap = flush_gap
        # 全局计数器
        self._global_counter: Counter = Counter()
        self._last_housekeep_ts = time.time()
        # 行为负载追踪: {channel_id: {deed_type: [timestamps]}}
        self._usage_timeline: Dict[str, Dict[str, List[float]]] = {}
        # 启动时从DB恢复
        self._hydrate_from_db()

    def _ensure_channel(self, channel_id: str) -> ChannelDeedRing:
        if channel_id not in self._channels:
            self._channels[channel_id] = ChannelDeedRing(self._ring_cap)
        return self._channels[channel_id]

    def _get_hippo_buffer(self, channel_id: str):
        try:
            from src.memory_system.hippocampus_buffer import (
                get_hippocampus_buffer,
            )

            return get_hippocampus_buffer(channel_id)
        except (ImportError, Exception):
            return None

    def _get_recall_hub(self):
        try:
            from src.memory_system.memory_core import acquire_recollection_hub

            return acquire_recollection_hub()
        except (ImportError, Exception):
            return None

    def _sync_to_memory_system(self, channel_id: str):
        """将待处理行为批量同步到海马体和记忆核心"""
        if not self._pending_deeds:
            return
        hippo = self._get_hippo_buffer(channel_id)
        recall_hub = self._get_recall_hub()
        if not hippo and not recall_hub:
            self._pending_deeds.clear()
            return
        synced = 0
        failed_indices = []
        for i, deed in enumerate(self._pending_deeds):
            memory_line = deed.as_memory_line()
            weight = (
                0.65
                if deed.deed_type
                in ("utterance", "sentiment_shift", "persona_morph")
                else 0.35
            )
            deed_ok = True
            if hippo:
                try:
                    hippo.ingest(
                        memory_line, deed.participant_id or "bot", weight
                    )
                except Exception as e:
                    logger.debug(f"注入行为记忆失败: {e}")
                    deed_ok = False
            if recall_hub:
                try:
                    recall_hub.deposit_memory(
                        stream_id=deed.channel_id,
                        content=memory_line,
                        entry_category="bot_deed",
                        user_id=deed.participant_id or "bot",
                        significance=weight,
                    )
                except Exception as e:
                    logger.debug(f"存储行为记忆失败: {e}")
                    deed_ok = False
            if deed_ok:
                synced += 1
            else:
                failed_indices.append(i)
        if failed_indices:
            self._pending_deeds = [self._pending_deeds[i] for i in failed_indices]
        else:
            self._pending_deeds.clear()
        self._last_persist_ts = time.time()
        if synced > 0:
            logger.debug(f"已同步 {synced} 条行为到记忆子系统")

    def _persist_to_db(self, deed: BotDeed):
        """将单条行为持久化到数据库"""
        try:
            from src.common.database.database_model import BotActivityEntry

            BotActivityEntry.insert(
                deed_id=deed.deed_id,
                stream_id=deed.channel_id,
                user_id=deed.participant_id,
                activity_type=deed.deed_type,
                description=deed.body[:500],
                outcome="ok" if deed.succeeded else "fail",
                extra_json=(
                    json.dumps(deed.annotations, ensure_ascii=False)
                    if deed.annotations
                    else None
                ),
                birth_ts=deed.occurred_at,
            ).on_conflict(
                conflict_target=[BotActivityEntry.deed_id],
                update={BotActivityEntry.description: deed.body[:500]},
            ).execute()
        except Exception as exc:
            logger.debug(f"行为写DB失败: {exc}")

    def _hydrate_from_db(self):
        """启动时从数据库恢复最近的行为记录"""
        try:
            from src.common.database.database_model import BotActivityEntry

            cutoff = time.time() - 86400  # 仅恢复24小时内
            rows = list(
                BotActivityEntry.select()
                .where(BotActivityEntry.birth_ts > cutoff)
                .order_by(BotActivityEntry.birth_ts.asc())
                .limit(200)
            )
            if not rows:
                return
            restored = 0
            for row in rows:
                deed = BotDeed(
                    deed_id=row.deed_id,
                    deed_type=row.activity_type,
                    channel_id=row.stream_id,
                    body=row.description,
                    occurred_at=row.birth_ts,
                    participant_id=row.user_id,
                    annotations=(
                        json.loads(row.extra_json) if row.extra_json else {}
                    ),
                    succeeded=(row.outcome == "ok"),
                )
                ring = self._ensure_channel(row.stream_id)
                ring.push(deed)
                self._global_counter[deed.deed_type] += 1
                restored += 1
            if restored > 0:
                logger.debug(f"从数据库恢复了 {restored} 条历史行为记录")
        except Exception as exc:
            logger.debug(f"行为记录恢复跳过: {exc}")

    # ─────────── 核心记录接口 ───────────

    def chronicle(
        self,
        channel_id: str,
        deed_type: str,
        body: str,
        participant_id: Optional[str] = None,
        annotations: Optional[Dict] = None,
        succeeded: bool = True,
    ) -> str:
        """
        记录一条机器人行为(核心入口)
        返回生成的deed_id
        """
        deed_id = f"{int(time.time() * 1000)}_{deed_type[:6]}"
        ctx = annotations or {}
        if participant_id:
            ctx["participant_id"] = participant_id
        if not body:
            body = self._template_engine.render(deed_type, ctx)
        deed = BotDeed(
            deed_id=deed_id,
            deed_type=deed_type,
            channel_id=channel_id,
            body=body,
            participant_id=participant_id,
            annotations=ctx,
            succeeded=succeeded,
        )
        ring = self._ensure_channel(channel_id)
        ring.push(deed)
        self._pending_deeds.append(deed)
        self._global_counter[deed_type] += 1
        # 更新行为负载时间线
        if channel_id not in self._usage_timeline:
            self._usage_timeline[channel_id] = {}
        if deed_type not in self._usage_timeline[channel_id]:
            self._usage_timeline[channel_id][deed_type] = []
        self._usage_timeline[channel_id][deed_type].append(time.time())
        # 持久化到DB
        self._persist_to_db(deed)
        # 条件触发批量同步
        now = time.time()
        if (
            len(self._pending_deeds) >= self._pending_cap
            or now - self._last_persist_ts > self._persist_gap
        ):
            self._sync_to_memory_system(channel_id)
        # 定期清理
        if now - self._last_housekeep_ts > 3600:
            self._housekeep()
        logger.debug(f"记录行为: {deed_type} - {body[:60]}")
        return deed_id

    # ─────────── 便捷记录方法 ───────────

    def chronicle_user_input(
        self,
        channel_id: str,
        user_id: str,
        user_name: str,
        content: str,
        activated: bool = True,
    ) -> str:
        if not activated:
            return ""
        return self.chronicle(
            channel_id,
            "user_input",
            content,
            user_id,
            {
                "user_name": user_name,
                "char_count": len(content),
                "activated": activated,
            },
        )

    def chronicle_sentiment(
        self,
        channel_id: str,
        user_id: str,
        shifts: Dict[str, float],
        trigger: str = "",
    ) -> str:
        return self.chronicle(
            channel_id,
            "sentiment_shift",
            trigger or "情绪更新",
            user_id,
            {
                "shifts": shifts,
                "trigger": trigger,
                "aggregate_delta": round(sum(shifts.values()), 4),
                "affected_dims": len([v for v in shifts.values() if v != 0]),
            },
        )

    def chronicle_tool(
        self, channel_id: str, tool_name: str, params: Dict, outcome: str = ""
    ) -> str:
        return self.chronicle(
            channel_id,
            "tool_invoke",
            tool_name,
            None,
            {
                "tool_name": tool_name,
                "params": params,
                "outcome": outcome,
                "param_count": len(params) if params else 0,
                "has_outcome": bool(outcome),
                "outcome_size": len(outcome) if outcome else 0,
            },
        )

    def chronicle_model_request(
        self,
        channel_id: str,
        model_name: str,
        prompt_chars: int,
        reply_chars: int,
        latency: float,
    ) -> str:
        throughput = reply_chars / latency if latency > 0 else 0
        return self.chronicle(
            channel_id,
            "model_call",
            model_name,
            None,
            {
                "model": model_name,
                "prompt_chars": prompt_chars,
                "reply_chars": reply_chars,
                "latency_sec": round(latency, 3),
                "throughput": round(throughput, 1),
                "compression_ratio": (
                    round(reply_chars / prompt_chars, 3)
                    if prompt_chars > 0
                    else 0
                ),
            },
        )

    def chronicle_utterance(
        self, channel_id: str, text: str, target_user: str = ""
    ) -> str:
        return self.chronicle(
            channel_id,
            "utterance",
            text,
            None,
            {
                "target_user": target_user,
                "char_count": len(text),
                "has_target": bool(target_user),
            },
        )

    def chronicle_persona_shift(
        self,
        channel_id: str,
        persona_label: str,
        reason: str,
        hold_duration: float,
    ) -> str:
        return self.chronicle(
            channel_id,
            "persona_morph",
            persona_label,
            None,
            {
                "persona_name": persona_label,
                "reason": reason,
                "hold_duration": hold_duration,
                "is_transient": hold_duration > 0,
            },
        )

    def chronicle_thought(self, channel_id: str, thought: str) -> str:
        return self.chronicle(
            channel_id,
            "inner_monologue",
            thought,
            None,
            {
                "char_count": len(thought),
                "is_deep": len(thought) > 100,
            },
        )

    def chronicle_fault(
        self, channel_id: str, fault_kind: str, detail: str
    ) -> str:
        severity = (
            "critical"
            if any(w in detail.lower() for w in ["critical", "fatal", "crash"])
            else "normal"
        )
        return self.chronicle(
            channel_id,
            "fault",
            fault_kind,
            None,
            {
                "fault_kind": fault_kind,
                "detail": detail,
                "detail_length": len(detail),
                "severity": severity,
            },
        )

    def chronicle_system(
        self, channel_id: str, event_kind: str, detail: str
    ) -> str:
        return self.chronicle(
            channel_id,
            "system_notice",
            event_kind,
            None,
            {
                "event_kind": event_kind,
                "detail": detail,
                "detail_length": len(detail),
            },
        )

    def chronicle_retraction(
        self, channel_id: str, original_text: str = ""
    ) -> str:
        snippet = original_text[:30] if original_text else ""
        return self.chronicle(
            channel_id,
            "retraction",
            body=self._template_engine.render(
                "retraction", {"content": snippet}
            ),
            annotations={"content": snippet},
        )

    def chronicle_moderation(
        self, channel_id: str, target_id: str, target_name: str = ""
    ) -> str:
        display = target_name or target_id
        return self.chronicle(
            channel_id,
            "moderation",
            body=self._template_engine.render(
                "moderation", {"target": display}
            ),
            annotations={"target": display, "target_id": target_id},
        )

    def chronicle_silence(
        self,
        channel_id: str,
        target_id: str,
        duration: int,
        target_name: str = "",
    ) -> str:
        display = target_name or target_id
        return self.chronicle(
            channel_id,
            "silence_impose",
            body=self._template_engine.render(
                "silence_impose", {"target": display, "duration": duration}
            ),
            annotations={
                "target": display,
                "target_id": target_id,
                "duration": duration,
            },
        )

    def chronicle_directed_reply(self, channel_id: str, content: str) -> str:
        snippet = content[:50] if content else ""
        return self.chronicle(
            channel_id,
            "directed_reply",
            body=self._template_engine.render(
                "directed_reply", {"content": snippet}
            ),
            annotations={"content": snippet},
        )

    def chronicle_query(self, channel_id: str, query_text: str) -> str:
        snippet = query_text[:30] if query_text else ""
        return self.chronicle(
            channel_id,
            "query",
            body=self._template_engine.render("query", {"query": snippet}),
            annotations={"query": snippet},
        )

    def force_sync(self, channel_id: str):
        """强制刷新待处理行为到记忆系统"""
        self._sync_to_memory_system(channel_id)

    # ─────────── 查询和Prompt生成 ───────────

    def recent_deeds(
        self, channel_id: str, count: int = 5, within_minutes: int = 30
    ) -> List[BotDeed]:
        if channel_id not in self._channels:
            return []
        return self._channels[channel_id].recent(count, within_minutes)

    def find_latest_deed(
        self, channel_id: str, deed_type: str, within_minutes: int = 10
    ) -> Optional[BotDeed]:
        if channel_id not in self._channels:
            return None
        return self._channels[channel_id].find_latest(
            deed_type, within_minutes
        )

    def compose_brief_prompt(
        self, channel_id: str, within_minutes: int = 10, count: int = 5
    ) -> str:
        """生成简短的最近行为提示注入"""
        deeds = self.recent_deeds(channel_id, count, within_minutes)
        if not deeds:
            return ""
        lines = ["【最近的操作记录】"]
        for d in deeds:
            lines.append(f"- {d.elapsed_label()}: {d.body[:80]}")
        return "\n".join(lines)

    def compose_extended_prompt(self, channel_id: str, count: int = 15) -> str:
        """生成扩展版本的行为记忆提示(1小时内)"""
        deeds = self.recent_deeds(channel_id, count, within_minutes=60)
        if deeds:
            lines = ["【近1小时行为回顾】"]
            for d in deeds:
                lines.append(d.as_memory_line())
            return "\n".join(lines)
        # 缓冲区无数据时尝试从记忆核心检索
        recall_hub = self._get_recall_hub()
        if not recall_hub:
            return ""
        try:
            memories = recall_hub.query_memories(
                stream_id=channel_id,
                memory_type="bot_deed",
                limit=count,
            )
            if not memories:
                return ""
            lines = ["【近期行为回顾(来自记忆)】"]
            for m in memories[-count:]:
                content = (
                    getattr(m, "content", "")
                    if hasattr(m, "content")
                    else str(m)
                )
                lines.append(content)
            return "\n".join(lines)
        except Exception:
            return ""

    # ─────────── 行为负载分析 ───────────

    def compute_load(
        self, channel_id: str, deed_type: str, within_minutes: int = 30
    ) -> float:
        """计算指定通道、指定行为类型的当前行为负载"""
        timeline = self._usage_timeline.get(channel_id, {}).get(deed_type, [])
        boundary = time.time() - (within_minutes * 60)
        recent_stamps = [ts for ts in timeline if ts >= boundary]
        return self._load_model.compute(recent_stamps)

    def compose_load_prompt(
        self, channel_id: str, within_minutes: int = 30
    ) -> str:
        """生成行为负载提示注入"""
        if channel_id not in self._usage_timeline:
            return ""
        boundary = time.time() - (within_minutes * 60)
        loaded_items = []
        for dtype, stamps in self._usage_timeline.get(channel_id, {}).items():
            recent = [ts for ts in stamps if ts >= boundary]
            if not recent:
                continue
            load_value = self._load_model.compute(recent)
            if load_value >= 0.5:
                loaded_items.append((dtype, len(recent), load_value))
        if not loaded_items:
            return ""
        loaded_items.sort(key=lambda x: -x[2])
        lines = ["[行为负载指标]"]
        for dtype, cnt, load_value in loaded_items:
            if load_value >= 0.75:
                lines.append(
                    f"- {dtype} 已执行{cnt}次，重复负载过高({load_value:.0%})，建议暂停或变换方式"
                )
            else:
                lines.append(
                    f"- {dtype} 已执行{cnt}次，重复负载开始升高({load_value:.0%})"
                )
        return "\n".join(lines)

    def chronicle_with_load(
        self,
        channel_id: str,
        deed_type: str,
        body: str = "",
        annotations: Optional[Dict] = None,
        succeeded: bool = True,
        participant_id: Optional[str] = None,
    ) -> Tuple[str, float]:
        """记录行为并返回当前行为负载"""
        deed_id = self.chronicle(
            channel_id=channel_id,
            deed_type=deed_type,
            body=body,
            annotations=annotations,
            succeeded=succeeded,
            participant_id=participant_id,
        )
        load_value = self.compute_load(channel_id, deed_type)
        return deed_id, load_value

    # ─────────── 统计信息 ───────────

    def channel_stats(self, channel_id: str) -> Dict[str, Any]:
        if channel_id not in self._channels:
            return {"total": 0, "distribution": {}}
        ring = self._channels[channel_id]
        return {"total": ring.total, "distribution": ring.distribution()}

    def global_stats(self) -> Dict[str, Any]:
        return {
            "active_channels": len(self._channels),
            "all_time_total": sum(self._global_counter.values()),
            "type_breakdown": dict(self._global_counter),
        }

    def tool_usage_count(
        self, channel_id: str, tool_name: str, within_minutes: int = 30
    ) -> int:
        """统计指定工具在时间窗口内的使用次数"""
        if channel_id not in self._channels:
            return 0
        boundary = time.time() - (within_minutes * 60)
        count = 0
        for deed in self._channels[channel_id]._ring:
            if deed.occurred_at < boundary:
                continue
            if deed.deed_type == tool_name or tool_name in deed.deed_type:
                count += 1
        return count

    def export_channel(self, channel_id: str) -> List[Dict]:
        if channel_id not in self._channels:
            return []
        return [d.to_dict() for d in self._channels[channel_id]._ring]

    def import_channel(self, channel_id: str, records: List[Dict]):
        ring = self._ensure_channel(channel_id)
        for raw in records:
            deed = BotDeed.from_dict(raw)
            ring.push(deed)

    # ─────────── 维护和清理 ───────────

    def _housekeep(self):
        """定期清理过期数据和空通道"""
        for ring in self._channels.values():
            ring.prune_expired(max_age_hours=24)
        # 清理空通道
        empty_channels = [
            cid for cid, ring in self._channels.items() if ring.total == 0
        ]
        for cid in empty_channels:
            del self._channels[cid]
            self._usage_timeline.pop(cid, None)
        # 清理疲劳时间线中的过期时间戳
        cutoff = time.time() - 7200
        for cid in list(self._usage_timeline.keys()):
            for dtype in list(self._usage_timeline[cid].keys()):
                self._usage_timeline[cid][dtype] = [
                    ts
                    for ts in self._usage_timeline[cid][dtype]
                    if ts >= cutoff
                ]
                if not self._usage_timeline[cid][dtype]:
                    del self._usage_timeline[cid][dtype]
            if not self._usage_timeline[cid]:
                del self._usage_timeline[cid]
        self._last_housekeep_ts = time.time()

    def persist_all(self):
        """将所有通道的待处理行为同步到记忆系统"""
        channels_with_pending = set()
        for deed in self._pending_deeds:
            channels_with_pending.add(deed.channel_id)
        for cid in channels_with_pending:
            self._sync_to_memory_system(cid)
        logger.info(
            f"行为编年史全量持久化完毕({len(channels_with_pending)}个通道)"
        )


# ═══════════════════════════════════════════
# 单例管理
# ═══════════════════════════════════════════

_chronicle_ref: Optional[BotBehaviorChronicle] = None


def acquire_behavior_chronicle() -> BotBehaviorChronicle:
    """获取全局行为编年史实例"""
    global _chronicle_ref
    if _chronicle_ref is None:
        _chronicle_ref = BotBehaviorChronicle()
    return _chronicle_ref


def bootstrap_behavior_chronicle(
    ring_capacity: int = 60,
) -> BotBehaviorChronicle:
    """初始化行为编年史(覆盖现有实例)"""
    global _chronicle_ref
    _chronicle_ref = BotBehaviorChronicle(ring_capacity=ring_capacity)
    logger.info("机器人行为编年史初始化完成")
    return _chronicle_ref


# 短别名(兼容其他模块的导入习惯)
get_self_recorder = acquire_behavior_chronicle
init_self_recorder = bootstrap_behavior_chronicle

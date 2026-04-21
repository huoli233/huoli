import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Deque, Dict, List, Optional, Tuple
from src.common.config.config_engine import ConfigEngine, get_default_config_engine
from src.common.logger import get_logger

logger = get_logger("phase_tracker")


class RelationPhase(IntEnum):
    """关系阶段枚举（6 级 + 封锁）

    数值越大代表关系越亲密，BLOCKED 为特殊状态。
    阈值区间由 PHASE_BAND 定义。
    """
    HOSTILE = 0
    STRANGER = 1
    ACQUAINTANCE = 2
    FAMILIAR = 3
    CLOSE = 4
    TRUSTED = 5
    BLOCKED = -1


# 各阶段对应的社交分区间 [lower, upper)
PHASE_BAND: Dict[int, Tuple[float, float]] = {
    RelationPhase.HOSTILE:      (-100.0, -45.0),
    RelationPhase.STRANGER:     (-45.0, -10.0),
    RelationPhase.ACQUAINTANCE: (-10.0,  20.0),
    RelationPhase.FAMILIAR:     (20.0,   45.0),
    RelationPhase.CLOSE:        (45.0,   70.0),
    RelationPhase.TRUSTED:      (70.0,  101.0),
}

# 阶段 → 能量计算 / 回复策略 使用的权重因子
PHASE_WEIGHT: Dict[int, float] = {
    RelationPhase.HOSTILE:      0.3,
    RelationPhase.STRANGER:     0.6,
    RelationPhase.ACQUAINTANCE: 0.8,
    RelationPhase.FAMILIAR:     0.9,
    RelationPhase.CLOSE:        1.0,
    RelationPhase.TRUSTED:      1.1,
}

# 阶段友好描述（无自定义标签时回退使用）
PHASE_DESCRIPTION: Dict[int, str] = {
    RelationPhase.HOSTILE:      "关系紧张",
    RelationPhase.STRANGER:     "不太熟悉",
    RelationPhase.ACQUAINTANCE: "有些熟悉",
    RelationPhase.FAMILIAR:     "比较熟悉",
    RelationPhase.CLOSE:        "关系亲密",
    RelationPhase.TRUSTED:      "非常信任",
}


# ================================================================
#  数据结构
# ================================================================

@dataclass
class TrendSample:
    """趋势采样点"""
    value: float
    timestamp: float


@dataclass
class PhaseTransition:
    """阶段跃迁记录"""
    from_phase: int
    to_phase: int
    from_label: str
    to_label: str
    occurred_at: float
    score_at: float


@dataclass
class AchievementMark:
    """里程碑标记"""
    tag: str
    reached_at: float
    score_at: float


@dataclass
class PersonaImpression:
    """用户印象 - 由模型生成的个性化标签

    每次调用 LLM 刷新时，旧标签会存入 revision_log。
    """
    target_uid: str = ""
    nickname: str = ""
    nickname_rationale: str = ""
    character_tags: List[str] = field(default_factory=list)
    dialogue_style: str = ""
    frequent_topics: List[str] = field(default_factory=list)
    mood_drift: str = "neutral"
    refreshed_at: float = 0.0
    revision_log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BondDossier:
    """单用户关系档案

    聚合关系阶段、趋势、里程碑、互动伙伴、印象五大子块。
    每个 (uid, channel_id) 对对应一份档案。
    """
    uid: str
    channel_id: str
    current_phase: int = RelationPhase.ACQUAINTANCE
    custom_nick: str = ""
    trend_label: str = "stable"
    recent_values: Deque[TrendSample] = field(default_factory=lambda: deque(maxlen=100))
    transition_log: List[PhaseTransition] = field(default_factory=list)
    peer_affinity: Dict[str, float] = field(default_factory=dict)
    achievements: List[AchievementMark] = field(default_factory=list)
    last_transition_ts: float = 0.0
    impression: Optional[PersonaImpression] = None
    summary_text: str = ""
    _LOG_CAP: int = 30

    def pack(self) -> Dict[str, Any]:
        """序列化为可持久化字典"""
        return {
            "uid": self.uid,
            "channel_id": self.channel_id,
            "current_phase": self.current_phase,
            "custom_nick": self.custom_nick,
            "trend_label": self.trend_label,
            "recent_values": [
                {"v": s.value, "t": s.timestamp} for s in self.recent_values
            ],
            "transition_log": [
                {
                    "from_phase": t.from_phase,
                    "to_phase": t.to_phase,
                    "from_label": t.from_label,
                    "to_label": t.to_label,
                    "occurred_at": t.occurred_at,
                    "score_at": t.score_at,
                }
                for t in self.transition_log[-self._LOG_CAP:]
            ],
            "peer_affinity": self.peer_affinity,
            "achievements": [
                {"tag": a.tag, "reached_at": a.reached_at, "score_at": a.score_at}
                for a in self.achievements
            ],
            "last_transition_ts": self.last_transition_ts,
            "impression": self._pack_impression(),
            "summary_text": self.summary_text,
        }

    def _pack_impression(self) -> Optional[Dict[str, Any]]:
        imp = self.impression
        if imp is None:
            return None
        return {
            "target_uid": imp.target_uid,
            "nickname": imp.nickname,
            "nickname_rationale": imp.nickname_rationale,
            "character_tags": imp.character_tags,
            "dialogue_style": imp.dialogue_style,
            "frequent_topics": imp.frequent_topics,
            "mood_drift": imp.mood_drift,
            "refreshed_at": imp.refreshed_at,
            "revision_log": imp.revision_log[-10:],
        }

    @classmethod
    def unpack(cls, data: Dict[str, Any]) -> "BondDossier":
        """从持久化字典反序列化"""
        dossier = cls(
            uid=data["uid"],
            channel_id=data["channel_id"],
            current_phase=data.get("current_phase", RelationPhase.ACQUAINTANCE),
            custom_nick=data.get("custom_nick", ""),
            trend_label=data.get("trend_label", "stable"),
            last_transition_ts=float(data.get("last_transition_ts", 0.0)),
            summary_text=data.get("summary_text", ""),
        )
        for sv in data.get("recent_values", []):
            dossier.recent_values.append(
                TrendSample(value=sv["v"], timestamp=sv["t"])
            )
        for td in data.get("transition_log", []):
            dossier.transition_log.append(
                PhaseTransition(
                    from_phase=td.get("from_phase", RelationPhase.ACQUAINTANCE),
                    to_phase=td.get("to_phase", RelationPhase.ACQUAINTANCE),
                    from_label=td.get("from_label", ""),
                    to_label=td.get("to_label", ""),
                    occurred_at=td.get("occurred_at", 0.0),
                    score_at=td.get("score_at", 0.0),
                )
            )
        for ad in data.get("achievements", []):
            dossier.achievements.append(
                AchievementMark(
                    tag=ad["tag"],
                    reached_at=ad["reached_at"],
                    score_at=ad["score_at"],
                )
            )
        dossier.peer_affinity = data.get("peer_affinity", {})
        imp_data = data.get("impression")
        if imp_data:
            dossier.impression = PersonaImpression(
                target_uid=imp_data.get("target_uid", ""),
                nickname=imp_data.get("nickname", ""),
                nickname_rationale=imp_data.get("nickname_rationale", ""),
                character_tags=imp_data.get("character_tags", []),
                dialogue_style=imp_data.get("dialogue_style", ""),
                frequent_topics=imp_data.get("frequent_topics", []),
                mood_drift=imp_data.get("mood_drift", "neutral"),
                refreshed_at=imp_data.get("refreshed_at", 0.0),
                revision_log=imp_data.get("revision_log", []),
            )
        return dossier


# ================================================================
#  印象生成器
# ================================================================

class ImpressionMinter:
    """通过 LLM 为用户生成个性化印象标签"""

    _PROMPT_SKELETON = (
        "你是一个观察者，根据与用户的互动历史，为这个用户生成一个个性化的印象标签。\n\n"
        "用户ID: {uid_short}\n"
        "互动次数: {interaction_count}\n"
        "社交值: {social_value}\n"
        "最近互动内容摘要: {interaction_digest}\n"
        "社交值变化趋势: {trend}\n"
        "用户常用话题: {topics}\n"
        "用户互动风格: {style}\n\n"
        "请用JSON格式返回你对这个用户的印象：\n"
        '{{\n'
        '    "custom_label": "一个简短的个性化标签（如\'有趣的家伙\'、\'温柔的大姐姐\'、\'爱开玩笑的朋友\'、\'有点傲娇的人\'等，'
        '根据你的感受来命名，不要用\'好友\'、\'陌生人\'这种通用词）",\n'
        '    "label_reason": "为什么给这个标签的理由",\n'
        '    "personality_traits": ["性格特点1", "性格特点2"],\n'
        '    "interaction_style": "互动风格描述",\n'
        '    "topics_of_interest": ["感兴趣的话题1", "感兴趣的话题2"]\n'
        '}}'
    )

    async def mint(
        self,
        uid: str,
        interaction_count: int,
        social_value: float,
        interaction_digest: str,
        trend: str,
        topics: List[str],
        style: str,
    ) -> Optional[PersonaImpression]:
        """调用 LLM 生成个性化印象"""
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            filled = self._PROMPT_SKELETON.format(
                uid_short=uid[:8],
                interaction_count=interaction_count,
                social_value=round(social_value, 1),
                interaction_digest=interaction_digest[:500],
                trend=trend,
                topics="、".join(topics[:5]) if topics else "无",
                style=style or "未知",
            )
            req = LLMRequest(
                model_set=model_config.model_task_config.lightweight,
                request_type="phase_tracker",
            )
            try:
                result = await asyncio.wait_for(
                    req.generate_response_async(filled, temperature=0.7),
                    timeout=20.0,
                )
            except asyncio.TimeoutError:
                logger.debug("印象生成超时(20s)")
                return None
            raw_text = result[0] if isinstance(result, tuple) else result
            return self._decode(uid, raw_text)
        except Exception as exc:
            logger.debug(f"印象生成失败: {exc}")
            return None

    def _decode(self, uid: str, raw: str) -> Optional[PersonaImpression]:
        """解析模型返回的 JSON"""
        try:
            blob = json.loads(raw)
            if isinstance(blob, str):
                import re
                m = re.search(r"\{[\s\S]*\}", raw)
                if not m:
                    return None
                blob = json.loads(m.group())
            return PersonaImpression(
                target_uid=uid,
                nickname=blob.get("custom_label", ""),
                nickname_rationale=blob.get("label_reason", ""),
                character_tags=blob.get("personality_traits", []),
                dialogue_style=blob.get("interaction_style", ""),
                frequent_topics=blob.get("topics_of_interest", []),
                mood_drift="neutral",
                refreshed_at=time.time(),
            )
        except Exception as exc:
            logger.debug(f"印象解析失败: {exc}")
            return None


# ================================================================
#  阶段追踪器主类
# ================================================================

class PhaseTracker:
    """关系阶段与印象追踪器（Layer 2）

    职责：
      - 根据 SettlementEngine 输出的社交分维护关系阶段
      - 追踪趋势（上升/下降/稳定）
      - 管理里程碑
      - 管理用户间互动亲密度
      - 调用 LLM 刷新个性化印象

    持久化通过可选适配器完成（与旧 RelationshipTracker 兼容）。
    """

    def __init__(self, config_engine: Optional[ConfigEngine] = None):
        self._dossiers: Dict[str, BondDossier] = {}
        self._cfg_hub = config_engine or get_default_config_engine()
        self._adapter = None
        self._bg_tasks: set = set()
        self._minter = ImpressionMinter()
        # 调优参数
        self._trend_window_sec: float = 3600.0
        self._trend_rise_bar: float = 3.0
        self._trend_fall_bar: float = -3.0
        self._transition_cooldown_sec: float = 120.0
        self._center_guard_ratio: float = 0.35
        self._impression_refresh_interval: float = 3600.0
        self._achievement_gates: Dict[str, float] = {}
        self._load_tuning()

    def attach_adapter(self, adapter) -> None:
        """绑定持久化适配器"""
        self._adapter = adapter

    # ================================================================
    #  配置
    # ================================================================

    def _load_tuning(self) -> None:
        """从配置引擎加载调优参数"""
        default_gates = {
            "first_trust": 30.0,
            "stable_friendship": 55.0,
            "high_trust": 75.0,
        }
        cfg = self._cfg_hub.get_relationship_config()
        self._trend_window_sec = float(cfg.get("trend_window_sec", 3600.0))
        self._trend_rise_bar = float(cfg.get("trend_improving_threshold", 3.0))
        self._trend_fall_bar = float(cfg.get("trend_declining_threshold", -3.0))
        self._transition_cooldown_sec = float(cfg.get("stage_change_cooldown_sec", 120.0))
        self._center_guard_ratio = float(cfg.get("protection_ratio", 0.35))
        self._impression_refresh_interval = float(cfg.get("impression_update_interval", 3600.0))
        gates = cfg.get("milestones", default_gates)
        if isinstance(gates, dict):
            self._achievement_gates = {k: float(v) for k, v in gates.items()}
        else:
            self._achievement_gates = default_gates

    # ================================================================
    #  核心操作
    # ================================================================

    def ingest_score(self, uid: str, channel_id: str, social_score: float) -> Optional[PhaseTransition]:
        """将最新社交分注入追踪器，更新阶段/趋势/里程碑

        返回：若发生阶段跃迁则返回 PhaseTransition，否则 None。
        """
        dossier = self._ensure_dossier(uid, channel_id)
        now = time.time()
        # 采样
        dossier.recent_values.append(TrendSample(value=social_score, timestamp=now))
        # 趋势
        dossier.trend_label = self._compute_trend(dossier)
        # 里程碑
        self._scan_achievements(dossier, social_score, now)
        # 阶段判定
        candidate = self._score_to_phase(social_score)
        transition: Optional[PhaseTransition] = None
        if candidate != dossier.current_phase and self._transition_cleared(dossier, social_score, candidate, now):
            transition = PhaseTransition(
                from_phase=dossier.current_phase,
                to_phase=candidate,
                from_label=dossier.custom_nick,
                to_label="",
                occurred_at=now,
                score_at=social_score,
            )
            dossier.transition_log.append(transition)
            if len(dossier.transition_log) > dossier._LOG_CAP:
                dossier.transition_log = dossier.transition_log[-dossier._LOG_CAP:]
            dossier.current_phase = candidate
            dossier.last_transition_ts = now
        # 异步持久化
        self._schedule_persist(dossier)
        return transition

    def log_peer_interaction(self, uid: str, channel_id: str, peer_uid: str, weight: float = 1.0) -> None:
        """记录两人之间的互动"""
        dossier = self._ensure_dossier(uid, channel_id)
        dossier.peer_affinity[peer_uid] = dossier.peer_affinity.get(peer_uid, 0.0) + weight

    def top_peers(self, uid: str, channel_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """获取最亲密的互动伙伴"""
        dossier = self._ensure_dossier(uid, channel_id)
        return sorted(dossier.peer_affinity.items(), key=lambda kv: kv[1], reverse=True)[:limit]

    # ================================================================
    #  只读查询
    # ================================================================

    def fetch_dossier(self, uid: str, channel_id: str) -> BondDossier:
        """获取完整关系档案"""
        return self._ensure_dossier(uid, channel_id)

    def current_phase_of(self, uid: str, channel_id: str) -> int:
        """获取当前关系阶段（int 枚举值）"""
        return self._ensure_dossier(uid, channel_id).current_phase

    def readable_phase(self, uid: str, channel_id: str) -> str:
        """获取可读的阶段描述（优先返回自定义标签）"""
        dossier = self._ensure_dossier(uid, channel_id)
        if dossier.custom_nick:
            return dossier.custom_nick
        return PHASE_DESCRIPTION.get(dossier.current_phase, "有些熟悉")

    def phase_weight_of(self, uid: str, channel_id: str) -> float:
        """获取阶段权重因子（用于能量/回复计算）"""
        phase = self._ensure_dossier(uid, channel_id).current_phase
        return PHASE_WEIGHT.get(phase, 0.8)

    def trend_of(self, uid: str, channel_id: str) -> str:
        """获取关系趋势标签"""
        return self._ensure_dossier(uid, channel_id).trend_label

    def recent_transitions_of(self, uid: str, channel_id: str, limit: int = 5) -> List[PhaseTransition]:
        """获取最近的阶段跃迁记录"""
        return self._ensure_dossier(uid, channel_id).transition_log[-limit:]

    def achievements_of(self, uid: str, channel_id: str) -> List[AchievementMark]:
        """获取已达到的里程碑"""
        return list(self._ensure_dossier(uid, channel_id).achievements)

    def all_nicks_in_channel(self, channel_id: str) -> Dict[str, str]:
        """获取频道内所有用户的自定义标签"""
        result = {}
        for _key, dossier in self._dossiers.items():
            if dossier.channel_id == channel_id and dossier.custom_nick:
                result[dossier.uid] = dossier.custom_nick
        return result

    def brief_summary(self, uid: str, channel_id: str) -> str:
        """生成用户摘要文本"""
        dossier = self._ensure_dossier(uid, channel_id)
        segments = []
        if dossier.custom_nick:
            segments.append(f"印象: {dossier.custom_nick}")
        if dossier.impression and dossier.impression.nickname_rationale:
            segments.append(f"理由: {dossier.impression.nickname_rationale}")
        segments.append(f"趋势: {dossier.trend_label}")
        segments.append(f"互动次数: {len(dossier.recent_values)}")
        return " | ".join(segments) if segments else "暂无印象"

    # ================================================================
    #  印象管理
    # ================================================================

    async def refresh_impression(
        self,
        uid: str,
        channel_id: str,
        interaction_digest: str = "",
        topics: Optional[List[str]] = None,
        style: str = "",
    ) -> Optional[PersonaImpression]:
        """刷新用户印象（受冷却间隔保护）"""
        dossier = self._ensure_dossier(uid, channel_id)
        now = time.time()
        if dossier.impression:
            age = now - dossier.impression.refreshed_at
            if age < self._impression_refresh_interval:
                return dossier.impression
        sample_count = len(dossier.recent_values)
        latest_score = dossier.recent_values[-1].value if dossier.recent_values else 0.0
        new_imp = await self._minter.mint(
            uid=uid,
            interaction_count=sample_count,
            social_value=latest_score,
            interaction_digest=interaction_digest,
            trend=dossier.trend_label,
            topics=topics or [],
            style=style,
        )
        if new_imp is None:
            return dossier.impression
        # 归档旧标签
        if dossier.impression and dossier.impression.nickname:
            new_imp.revision_log = list(dossier.impression.revision_log)
            new_imp.revision_log.append({
                "label": dossier.impression.nickname,
                "reason": dossier.impression.nickname_rationale,
                "timestamp": now,
            })
            if len(new_imp.revision_log) > 10:
                new_imp.revision_log = new_imp.revision_log[-10:]
        dossier.impression = new_imp
        dossier.custom_nick = new_imp.nickname
        logger.info(f"[阶段追踪] uid={uid[:8]} 印象刷新: {new_imp.nickname}")
        self._schedule_persist(dossier)
        return new_imp

    # ================================================================
    #  持久化适配器桥接
    # ================================================================

    async def load_from_adapter(self, uid: str, channel_id: str) -> Optional[BondDossier]:
        """从适配器加载档案"""
        if not self._adapter:
            return None
        try:
            data = await self._adapter.get_relationship(uid, channel_id)
            if data:
                return BondDossier.unpack(data)
        except Exception as exc:
            logger.debug(f"档案加载失败 uid={uid} channel={channel_id}: {exc}")
        return None

    async def flush_to_adapter(self, dossier: BondDossier) -> None:
        """将档案写回适配器"""
        if not self._adapter:
            return
        try:
            await self._adapter.set_relationship(
                dossier.uid,
                dossier.channel_id,
                dossier.pack(),
            )
        except Exception as exc:
            logger.debug(f"档案持久化失败 uid={dossier.uid}: {exc}")

    # ================================================================
    #  内部方法
    # ================================================================

    def _ensure_dossier(self, uid: str, channel_id: str) -> BondDossier:
        key = f"{uid}:{channel_id}"
        if key not in self._dossiers:
            self._dossiers[key] = BondDossier(uid=uid, channel_id=channel_id)
        return self._dossiers[key]

    def _score_to_phase(self, score: float) -> int:
        """分值 → 阶段映射"""
        for phase_val, (low, high) in PHASE_BAND.items():
            if low <= score < high:
                return phase_val
        if score <= -100.0:
            return RelationPhase.HOSTILE
        return RelationPhase.TRUSTED

    def _compute_trend(self, dossier: BondDossier) -> str:
        """评估关系趋势"""
        if len(dossier.recent_values) < 2:
            return "稳定"
        now = time.time()
        cutoff = now - self._trend_window_sec
        window = [s for s in dossier.recent_values if s.timestamp >= cutoff]
        if len(window) < 2:
            return "稳定"
        swing = window[-1].value - window[0].value
        if swing >= self._trend_rise_bar:
            return "上升"
        if swing <= self._trend_fall_bar:
            return "下降"
        return "稳定"

    def _scan_achievements(self, dossier: BondDossier, score: float, now: float) -> None:
        """扫描并记录新达成的里程碑"""
        existing_tags = {a.tag for a in dossier.achievements}
        for tag, gate_score in self._achievement_gates.items():
            if tag in existing_tags:
                continue
            if score >= gate_score:
                dossier.achievements.append(
                    AchievementMark(tag=tag, reached_at=now, score_at=score)
                )

    def _transition_cleared(self, dossier: BondDossier, score: float, candidate: int, now: float) -> bool:
        """判断阶段跃迁是否满足冷却与保护条件"""
        # 冷却检查
        if dossier.last_transition_ts > 0 and (now - dossier.last_transition_ts) < self._transition_cooldown_sec:
            return False
        # 敌对不受保护
        if candidate == RelationPhase.HOSTILE:
            return True
        # 中心保护区：分值离当前阶段中心不够远时阻止跃迁
        band = PHASE_BAND.get(dossier.current_phase)
        if band is None:
            return True
        low, high = band
        span = max(1.0, high - low)
        center = (low + high) / 2.0
        if score < center and (center - score) < span * self._center_guard_ratio:
            return False
        return True

    def _schedule_persist(self, dossier: BondDossier) -> None:
        """异步调度持久化"""
        if not self._adapter:
            return
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self.flush_to_adapter(dossier))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        except RuntimeError:
            pass

    def get_phase_bands(self) -> Dict[str, Tuple[float, float]]:
        """导出阶段区间配置（供外部查询/调试）"""
        return {f"phase_{k}": v for k, v in PHASE_BAND.items()}

    def get_achievement_gates(self) -> Dict[str, float]:
        """导出里程碑门槛"""
        return dict(self._achievement_gates)

    def collect_diagnostics(self) -> Dict[str, Any]:
        """运行时诊断"""
        return {
            "tracked_dossiers": len(self._dossiers),
            "adapter_bound": self._adapter is not None,
            "tuning": {
                "trend_window_sec": self._trend_window_sec,
                "transition_cooldown_sec": self._transition_cooldown_sec,
                "center_guard_ratio": self._center_guard_ratio,
                "impression_refresh_interval": self._impression_refresh_interval,
            },
        }


# ================================================================
#  模块级便捷访问
# ================================================================

_tracker_singleton: Optional[PhaseTracker] = None


def get_phase_tracker() -> PhaseTracker:
    """获取阶段追踪器单例"""
    global _tracker_singleton
    if _tracker_singleton is None:
        _tracker_singleton = PhaseTracker()
    return _tracker_singleton

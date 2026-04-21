import time
import math
import random
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("brain_load_monitor")


# ==================== 认知负载水位 ====================


class CognitiveLoadGauge(IntEnum):
    """认知负载水位标尺（四级制）"""

    CALM = 0
    TENSE = 1
    STRAINED = 2
    CRITICAL = 3


_GAUGE_LABELS = {
    CognitiveLoadGauge.CALM: "平稳",
    CognitiveLoadGauge.TENSE: "紧张",
    CognitiveLoadGauge.STRAINED: "过劳",
    CognitiveLoadGauge.CRITICAL: "临界",
}


# ==================== 负载快照 ====================


@dataclass
class BrainLoadSnapshot:
    """大脑负载状态快照

    记录一次扫描的完整数据：记忆总条数、占用体积、综合负载比、
    水位等级、各认知维度衰减系数、遗忘压力值。
    """

    record_count: int = 0
    volume_mb: float = 0.0
    load_ratio: float = 0.0
    gauge_level: CognitiveLoadGauge = CognitiveLoadGauge.CALM
    capability_dampening: Dict[str, float] = field(default_factory=dict)
    amnesia_pressure: float = 0.0
    snapshot_ts: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.capability_dampening:
            self.capability_dampening = {
                "memorize": 1.0,
                "recall": 1.0,
                "absorb": 1.0,
                "concentrate": 1.0,
                "muddle": 0.0,
            }

    def to_summary_dict(self) -> Dict[str, Any]:
        """序列化为简明字典"""
        return {
            "gauge": _GAUGE_LABELS.get(self.gauge_level, "未知"),
            "records": self.record_count,
            "volume_mb": round(self.volume_mb, 2),
            "load_ratio": round(self.load_ratio, 3),
            "amnesia_pressure": round(self.amnesia_pressure, 3),
            "dampening": {
                k: round(v, 3) for k, v in self.capability_dampening.items()
            },
        }


# ==================== 容量阈值配置 ====================


@dataclass
class CapacityThresholdProfile:
    """容量水位阈值配置（XBcore风格配置化，而非硬编码）

    每个水位有独立的记忆条数和体积上限。
    """

    calm_ceiling_count: int = 4500
    calm_ceiling_mb: float = 45.0
    tense_ceiling_count: int = 7500
    tense_ceiling_mb: float = 75.0
    strained_ceiling_count: int = 11000
    strained_ceiling_mb: float = 110.0
    critical_ceiling_count: int = 14000
    critical_ceiling_mb: float = 140.0


# ==================== 认知衰减曲线 ====================

_DAMPENING_CURVES: Dict[CognitiveLoadGauge, Dict[str, float]] = {
    CognitiveLoadGauge.CALM: {
        "memorize": 1.0,
        "recall": 1.0,
        "absorb": 1.0,
        "concentrate": 1.0,
        "muddle": 0.0,
    },
    CognitiveLoadGauge.TENSE: {
        "memorize": 0.82,
        "recall": 0.88,
        "absorb": 0.78,
        "concentrate": 0.83,
        "muddle": 0.12,
    },
    CognitiveLoadGauge.STRAINED: {
        "memorize": 0.55,
        "recall": 0.65,
        "absorb": 0.48,
        "concentrate": 0.53,
        "muddle": 0.35,
    },
    CognitiveLoadGauge.CRITICAL: {
        "memorize": 0.28,
        "recall": 0.40,
        "absorb": 0.22,
        "concentrate": 0.30,
        "muddle": 0.60,
    },
}


# ==================== 容量压力监控器 ====================


class CapacityPressureMonitor:
    """容量压力监控器 —— 评估记忆总量并计算各认知维度衰减系数。

    特色：
    - 分段线性插值计算负载比（对标XBcore的配置化方案）
    - 相邻水位间平滑衰减系数插值（避免MaiBot的阶梯跳变）
    - S型曲线推导遗忘压力（替代MaiBot的线性分段）
    - 评估缓存（45秒TTL，借鉴XBcore的60秒缓存思路）
    - 自适应倍率：紧急遗忘后降低阈值
    """

    def __init__(
        self,
        channel_tag: str = "global",
        threshold_profile: Optional[CapacityThresholdProfile] = None,
    ):
        self._channel_tag = channel_tag
        self._thresholds = threshold_profile or CapacityThresholdProfile()
        self._latest_snapshot = BrainLoadSnapshot()
        self._history_trail: List[Tuple[float, float]] = []
        self._assessment_cache_ts: float = 0.0
        self._assessment_ttl: float = 45.0
        self._adaptive_multiplier: float = 1.0

    def _resolve_memory_table(self):
        """惰性获取记忆数据表"""
        try:
            from src.common.database.database_model import MemoryRecord

            return MemoryRecord
        except ImportError:
            return None

    def scan_brain_load(self, force: bool = False) -> BrainLoadSnapshot:
        """扫描当前大脑负载状态

        带缓存机制：TTL内直接返回上次快照，避免频繁查库。
        """
        now = time.time()
        if (
            not force
            and (now - self._assessment_cache_ts) < self._assessment_ttl
        ):
            return self._latest_snapshot
        mem_table = self._resolve_memory_table()
        if not mem_table:
            return self._latest_snapshot
        try:
            rows = list(
                mem_table.select().where(
                    mem_table.stream_id == self._channel_tag
                )
            )
            total_count = len(rows)
            total_bytes = sum(
                len(r.content.encode("utf-8"))
                for r in rows
                if hasattr(r, "content") and r.content
            )
            volume_mb = total_bytes / (1024 * 1024)
        except Exception as exc:
            logger.error(f"扫描记忆负载异常: {exc}")
            return self._latest_snapshot
        load_ratio = self._evaluate_load_ratio(total_count, volume_mb)
        gauge = self._ratio_to_gauge(load_ratio)
        dampening = self._interpolate_dampening(load_ratio, gauge)
        amnesia_prs = self._derive_amnesia_pressure(load_ratio)
        self._latest_snapshot = BrainLoadSnapshot(
            record_count=total_count,
            volume_mb=volume_mb,
            load_ratio=load_ratio,
            gauge_level=gauge,
            capability_dampening=dampening,
            amnesia_pressure=amnesia_prs,
            snapshot_ts=now,
        )
        self._assessment_cache_ts = now
        self._history_trail.append((now, load_ratio))
        if len(self._history_trail) > 80:
            self._history_trail = self._history_trail[-80:]
        self._report_status()
        return self._latest_snapshot

    def _evaluate_load_ratio(self, count: int, mb: float) -> float:
        """根据记忆数量和体积综合评估负载比（分段线性插值）"""
        t = self._thresholds
        count_breakpoints = [
            (0, 0.0),
            (t.calm_ceiling_count, 0.2),
            (t.tense_ceiling_count, 0.5),
            (t.strained_ceiling_count, 0.85),
            (t.critical_ceiling_count, 1.0),
        ]
        size_breakpoints = [
            (0, 0.0),
            (t.calm_ceiling_mb, 0.2),
            (t.tense_ceiling_mb, 0.5),
            (t.strained_ceiling_mb, 0.85),
            (t.critical_ceiling_mb, 1.0),
        ]
        count_ratio = self._segment_interpolate(count, count_breakpoints)
        size_ratio = self._segment_interpolate(mb, size_breakpoints)
        blended = (
            max(count_ratio, size_ratio) * 0.65
            + (count_ratio + size_ratio) * 0.175
        )
        return min(2.0, blended * self._adaptive_multiplier)

    @staticmethod
    def _segment_interpolate(
        value: float, breakpoints: List[Tuple[float, float]]
    ) -> float:
        """分段线性插值（通用工具方法）"""
        if value <= breakpoints[0][0]:
            return breakpoints[0][1]
        for i in range(1, len(breakpoints)):
            lo_x, lo_y = breakpoints[i - 1]
            hi_x, hi_y = breakpoints[i]
            if value <= hi_x:
                frac = (value - lo_x) / max(hi_x - lo_x, 1e-9)
                return lo_y + frac * (hi_y - lo_y)
        last_x, last_y = breakpoints[-1]
        return last_y + (value - last_x) / max(last_x, 1e-9) * 0.3

    def _ratio_to_gauge(self, ratio: float) -> CognitiveLoadGauge:
        """负载比映射到水位等级"""
        if ratio >= 0.85:
            return CognitiveLoadGauge.CRITICAL
        if ratio >= 0.5:
            return CognitiveLoadGauge.STRAINED
        if ratio >= 0.2:
            return CognitiveLoadGauge.TENSE
        return CognitiveLoadGauge.CALM

    def _interpolate_dampening(
        self, ratio: float, gauge: CognitiveLoadGauge
    ) -> Dict[str, float]:
        """在相邻水位之间平滑插值衰减系数（消除阶梯跳变）"""
        base_curve = _DAMPENING_CURVES[gauge]
        if gauge == CognitiveLoadGauge.CALM:
            return dict(base_curve)
        prev_gauge = CognitiveLoadGauge(gauge.value - 1)
        prev_curve = _DAMPENING_CURVES[prev_gauge]
        boundary_table = [0.0, 0.2, 0.5, 0.85, 1.0]
        lo_bound = boundary_table[gauge.value]
        hi_bound = boundary_table[
            min(gauge.value + 1, len(boundary_table) - 1)
        ]
        span = hi_bound - lo_bound
        frac = (ratio - lo_bound) / max(span, 1e-9) if span > 0 else 0.0
        frac = max(0.0, min(1.0, frac))
        blended = {}
        for key in base_curve:
            blended[key] = prev_curve[key] + frac * (
                base_curve[key] - prev_curve[key]
            )
        return blended

    @staticmethod
    def _derive_amnesia_pressure(ratio: float) -> float:
        """从负载比推导遗忘压力（S型曲线替代MaiBot的线性分段）"""
        if ratio < 0.35:
            return 0.0
        x = (ratio - 0.65) * 8.0
        return 1.0 / (1.0 + math.exp(-x))

    def requires_emergency_purge(self) -> bool:
        """判断是否需要紧急遗忘清理"""
        snap = self._latest_snapshot
        return snap.load_ratio > 0.85 and snap.amnesia_pressure > 0.65

    def apply_operation_dampening(
        self, operation: str, **kwargs
    ) -> Dict[str, Any]:
        """将过载效应应用到指定操作

        根据当前衰减系数随机决定操作是否允许执行以及可能的副作用。
        """
        if self._latest_snapshot.load_ratio < 0.25:
            return {"allowed": True, "dampened": False}
        caps = self._latest_snapshot.capability_dampening
        outcome: Dict[str, Any] = {
            "allowed": True,
            "dampened": True,
            "side_effects": {},
        }
        if operation == "deposit_memory":
            success_chance = caps.get("memorize", 1.0)
            if random.random() > success_chance:
                outcome["allowed"] = False
                outcome["block_reason"] = "大脑饱和导致记忆写入受阻"
        elif operation == "retrieve_memory":
            retrieval_odds = caps.get("recall", 1.0)
            confusion_risk = caps.get("muddle", 0.0)
            if random.random() > retrieval_odds:
                outcome["allowed"] = False
                outcome["block_reason"] = "过载干扰导致回忆失败"
            elif random.random() < confusion_risk:
                outcome["side_effects"]["confused"] = True
                outcome["side_effects"][
                    "garbled_text"
                ] = _pick_confusion_utterance()
        elif operation == "learn_new":
            learn_odds = caps.get("absorb", 1.0)
            if random.random() > learn_odds:
                outcome["allowed"] = False
                outcome["block_reason"] = "认知已满无法学习新知识"
        return outcome

    def _report_status(self) -> None:
        """输出当前状态到日志"""
        snap = self._latest_snapshot
        label = _GAUGE_LABELS.get(snap.gauge_level, "未知")
        logger.info(
            f"[大脑负载] {label} | 记忆条数={snap.record_count} | "
            f"体积={snap.volume_mb:.1f}MB | 负载比={snap.load_ratio:.3f} | "
            f"遗忘压力={snap.amnesia_pressure:.3f}"
        )
        if snap.load_ratio > 0.85:
            logger.warning("大脑负载已临界，可能出现记忆模糊和学习困难")
        if self.requires_emergency_purge():
            logger.warning("已触及紧急清理阈值，建议立即执行智能遗忘")

    @property
    def current_snapshot(self) -> BrainLoadSnapshot:
        return self._latest_snapshot

    def invalidate_cache(self) -> None:
        """手动清除评估缓存"""
        self._assessment_cache_ts = 0.0

    def get_recent_history(self, n: int = 10) -> List[Tuple[float, float]]:
        """获取最近N次负载比历史"""
        return self._history_trail[-n:]


# ==================== 智能遗忘执行器 ====================


class SmartForgetExecutor:
    """智能遗忘执行器 —— 基于多维优先级评分挑选并删除低价值记忆。

    评分四因子：
    - 年代久远度 (antiquity) 28%
    - 访问冷淡度 (coldness) 38%
    - 内容琐碎度 (triviality) 22%
    - 篇幅短小度 (brevity) 12%

    保护系数：高重要性 / 高访问量 / 新创建的记忆获得显著降分保护。
    """

    _PRIORITY_WEIGHTS = {
        "antiquity": 0.28,
        "coldness": 0.38,
        "triviality": 0.22,
        "brevity": 0.12,
    }

    def __init__(self, channel_tag: str = "global"):
        self._channel_tag = channel_tag
        self._cumulative_purged: int = 0

    def _resolve_memory_table(self):
        """惰性获取记忆模型"""
        try:
            from src.common.database.database_model import MemoryRecord

            return MemoryRecord
        except ImportError:
            return None

    def execute_purge(
        self,
        reduction_fraction: float = 0.25,
        monitor: Optional[CapacityPressureMonitor] = None,
    ) -> Dict[str, Any]:
        """执行智能遗忘

        按 reduction_fraction 比例淘汰优先级最高的记忆条目。
        如传入 monitor 实例，遗忘后自动降低其 adaptive_multiplier。
        """
        logger.info(
            f"启动智能遗忘流程 | 目标淘汰比例={reduction_fraction * 100:.0f}%"
        )
        mem_table = self._resolve_memory_table()
        if not mem_table:
            return {"purged_count": 0, "freed_mb": 0.0}
        try:
            all_rows = list(
                mem_table.select().where(
                    mem_table.stream_id == self._channel_tag
                )
            )
        except Exception as exc:
            logger.error(f"加载记忆条目失败: {exc}")
            return {"purged_count": 0, "freed_mb": 0.0, "error": str(exc)}
        if not all_rows:
            return {"purged_count": 0, "freed_mb": 0.0}
        now = time.time()
        scored_entries: List[Tuple[float, Any]] = []
        for row in all_rows:
            priority = self._score_forgetting_priority(row, now)
            scored_entries.append((priority, row))
        scored_entries.sort(key=lambda pair: pair[0], reverse=True)
        cutoff = int(len(all_rows) * reduction_fraction)
        victims = scored_entries[:cutoff]
        purged = 0
        freed_bytes = 0
        type_breakdown: Dict[str, int] = {}
        for priority_val, row in victims:
            try:
                byte_size = (
                    len(row.content.encode("utf-8"))
                    if hasattr(row, "content") and row.content
                    else 0
                )
                row_type = getattr(row, "memory_type", "other")
                row.delete_instance()
                purged += 1
                freed_bytes += byte_size
                type_breakdown[row_type] = type_breakdown.get(row_type, 0) + 1
                if purged % 400 == 0:
                    logger.info(f"遗忘进度: 已清除{purged}条...")
            except Exception as row_exc:
                logger.debug(f"删除单条记忆失败: {row_exc}")
        freed_mb = freed_bytes / (1024 * 1024)
        self._cumulative_purged += purged
        if monitor:
            monitor._adaptive_multiplier = max(
                0.78, monitor._adaptive_multiplier - 0.08
            )
        result = {
            "purged_count": purged,
            "freed_mb": round(freed_mb, 2),
            "type_breakdown": type_breakdown,
            "priority_range": (
                round(victims[-1][0], 3) if victims else 0.0,
                round(victims[0][0], 3) if victims else 0.0,
            ),
        }
        logger.warning(f"智能遗忘完成 | 清除{purged}条 | 释放{freed_mb:.2f}MB")
        return result

    def _score_forgetting_priority(
        self, memory_row, current_ts: float
    ) -> float:
        """多维遗忘优先级评分（分数越高越优先遗忘）"""
        w = self._PRIORITY_WEIGHTS
        created_at = getattr(memory_row, "created_at", current_ts)
        age_days = (current_ts - created_at) / 86400.0
        antiquity_score = min(100.0, age_days) * w["antiquity"]
        access_ct = getattr(memory_row, "access_count", 0)
        coldness_score = max(0, 25 - access_ct) * w["coldness"]
        importance = getattr(memory_row, "importance", 0.5)
        triviality_score = (1.0 - importance) * 35.0 * w["triviality"]
        content = getattr(memory_row, "content", "")
        char_len = len(content) if content else 0
        brevity_bonus = 0.0
        if char_len < 40:
            brevity_bonus = 12.0
        elif char_len < 80:
            brevity_bonus = 5.0
        brevity_score = brevity_bonus * w["brevity"]
        raw_total = (
            antiquity_score + coldness_score + triviality_score + brevity_score
        )
        if importance > 0.78:
            raw_total *= 0.25
        if access_ct > 8:
            raw_total *= 0.35
        if age_days < 1.0:
            raw_total *= 0.45
        return raw_total

    @property
    def total_purged_count(self) -> int:
        return self._cumulative_purged


# ==================== 过载行为叙述器 ====================


class OverloadBehaviorNarrator:
    """过载状态行为叙述器 —— 根据负载快照生成影响bot人格的提示词片段。"""

    _GARBLED_UTTERANCES = [
        "嗯...好像有点记不清了...",
        "等一下，我刚想到什么，又忘掉了...",
        "诶？你说的是什么来着？",
        "脑子有点打结...让我理一理",
        "感觉记忆有点混在一起了...",
        "抱歉，现在脑袋不太清醒...",
        "我好像...忘了些什么重要的东西",
    ]

    @classmethod
    def compose_overload_prompt(cls, snapshot: BrainLoadSnapshot) -> str:
        """根据快照生成过载行为提示词（空字符串表示无需提示）"""
        if snapshot.load_ratio < 0.35:
            return ""
        fragments = []
        caps = snapshot.capability_dampening
        if snapshot.load_ratio > 0.55:
            fragments.append(
                "你觉得脑子有些沉重，可能会遗漏一些细节或想不起某些事情。"
            )
        if caps.get("muddle", 0.0) > 0.25:
            fragments.append("你偶尔会感到困惑，可能需要再想想才能回忆起来。")
        if caps.get("concentrate", 1.0) < 0.65:
            fragments.append("你的注意力有些涣散，话题间可能会不自觉跑偏。")
        if snapshot.load_ratio > 0.8:
            fragments.append(
                "你感觉大脑几乎装不下更多东西了，学习新知识变得特别费劲。"
            )
        if fragments:
            return "【认知负载效应】" + "；".join(fragments)
        return ""

    @classmethod
    def pick_garbled_response(cls) -> str:
        """随机挑选一条混乱表述"""
        return random.choice(cls._GARBLED_UTTERANCES)


def _pick_confusion_utterance() -> str:
    """模块级便捷函数：获取混乱表述"""
    return OverloadBehaviorNarrator.pick_garbled_response()


# ==================== 综合诊断 ====================


def collect_overload_diagnostics(
    monitor: CapacityPressureMonitor, executor: SmartForgetExecutor
) -> Dict[str, Any]:
    """汇总过载系统全部诊断指标"""
    snap = monitor.current_snapshot
    return {
        "gauge_level": _GAUGE_LABELS.get(snap.gauge_level, "未知"),
        "record_count": snap.record_count,
        "volume_mb": round(snap.volume_mb, 2),
        "load_ratio": round(snap.load_ratio, 3),
        "amnesia_pressure": round(snap.amnesia_pressure, 3),
        "capability_dampening": {
            k: round(v, 3) for k, v in snap.capability_dampening.items()
        },
        "cumulative_purged": executor.total_purged_count,
        "adaptive_multiplier": round(monitor._adaptive_multiplier, 3),
        "recent_trend": monitor.get_recent_history(8),
        "emergency_purge_needed": monitor.requires_emergency_purge(),
    }


# ==================== 单例注册表 ====================

_monitor_registry: Dict[str, CapacityPressureMonitor] = {}
_executor_registry: Dict[str, SmartForgetExecutor] = {}


def acquire_pressure_monitor(
    channel_tag: str = "global",
) -> CapacityPressureMonitor:
    """按频道/流获取或创建压力监控器实例"""
    if channel_tag not in _monitor_registry:
        _monitor_registry[channel_tag] = CapacityPressureMonitor(channel_tag)
    return _monitor_registry[channel_tag]


def acquire_forget_executor(
    channel_tag: str = "global",
) -> SmartForgetExecutor:
    """按频道/流获取或创建遗忘执行器实例"""
    if channel_tag not in _executor_registry:
        _executor_registry[channel_tag] = SmartForgetExecutor(channel_tag)
    return _executor_registry[channel_tag]


def discard_all_monitors() -> None:
    """清空所有监控器和执行器实例"""
    _monitor_registry.clear()
    _executor_registry.clear()


# 短别名（对标MaiBot接口风格）
get_memory_overload_protector = acquire_pressure_monitor
cleanup_overload_protectors = discard_all_monitors

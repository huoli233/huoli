"""
感知觉察引擎 — 收集多维感官数据、评估信号强度、生成知觉叙述

融合三套感知逻辑：
  · XBcore PerceptionEngine（三感官收集 → 提示词 → LLM → 解析结构化结论）
  · XBcore SignalDetector + InterestScorer + GroupSense + SelfSense + UserRelationSense
  · MaiBot/MIMiaoCore PerceptionGenerator（描述原子 × 多维知觉拼装 + 记忆褪色）

对外暴露 AwarenessEngine / gather_awareness / get_awareness_engine 接口。
所有阈值和描述原子权重从 CoreSettingsHub 读取。
"""

import asyncio
import json
import re
import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import get_logger
from src.config.core_config_engine import get_core_config
from src.common.data_models.heartflow_models import (
    ContextDemand,
)

logger = get_logger("觉察引擎")


class AlertGrade(Enum):
    """信号警觉等级"""

    DORMANT = 0
    FAINT = 1
    MODERATE = 2
    INTENSE = 3

    def label(self) -> str:
        return {0: "休眠", 1: "微弱", 2: "中等", 3: "强烈"}.get(self.value, "未知")

    def __ge__(self, other: "AlertGrade") -> bool:
        return self.value >= other.value

    def __gt__(self, other: "AlertGrade") -> bool:
        return self.value > other.value

    def __le__(self, other: "AlertGrade") -> bool:
        return self.value <= other.value

    def __lt__(self, other: "AlertGrade") -> bool:
        return self.value < other.value


class CrowdVibe(Enum):
    """群体活跃度描述"""

    DESOLATE = "死寂"
    SPARSE = "冷清"
    STEADY = "平稳"
    LIVELY = "活跃"
    ERUPTING = "爆发"


@dataclass
class AwarenessVerdict:
    """觉察判定结果"""

    alert_grade: AlertGrade = AlertGrade.DORMANT
    engagement_pull: float = 0.0
    rationale: str = ""
    supplementary_context: Optional[str] = None
    context_demand: Optional[ContextDemand] = None
    inference_source: str = "rule"
    crowd_vibe: CrowdVibe = CrowdVibe.STEADY
    body_narration: str = ""
    descriptor_atoms: Dict[str, float] = field(default_factory=dict)
    timing_tag: str = ""

    def should_engage(self, threshold: float = 0.4) -> bool:
        return self.engagement_pull >= threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_grade": self.alert_grade.name,
            "engagement_pull": round(self.engagement_pull, 3),
            "rationale": self.rationale,
            "supplementary_context": self.supplementary_context,
            "inference_source": self.inference_source,
            "crowd_vibe": self.crowd_vibe.value,
            "body_narration": self.body_narration,
            "descriptor_atoms": self.descriptor_atoms,
            "timing_tag": self.timing_tag,
        }


@dataclass
class _SensorySnapshot:
    """单次感官采集快照"""

    crowd_data: Dict[str, Any] = field(default_factory=dict)
    self_data: Dict[str, Any] = field(default_factory=dict)
    relation_data: Dict[str, Any] = field(default_factory=dict)
    message_digest: str = ""
    raw_text: str = ""
    timestamp: float = 0.0


class BodyStateNarrator:
    """
    体感叙述器 — 将情绪/创伤等内部状态映射为自然语言描述

    使用描述原子（descriptor atom）系统：
    每种状态维度有一组原子描述片段，根据权重和数值范围拼合为叙述文。
    """

    _TRAUMA_ATOMS = {
        (0, 0.1): "",
        (0.1, 0.35): "隐隐有些不舒服",
        (0.35, 0.6): "胸口有些发紧",
        (0.6, 0.85): "情绪上有些创伤",
        (0.85, 1.01): "内心伤痕累累",
    }
    _ANNOYANCE_ATOMS = {
        (0, 10): "",
        (10, 30): "有一丝不耐烦",
        (30, 55): "感到烦躁",
        (55, 80): "非常不耐烦",
        (80, 101): "已经忍无可忍",
    }
    _MASK_ATOMS = {
        (0, 0.2): "",
        (0.2, 0.5): "保持着某种微妙的伪装",
        (0.5, 0.8): "在刻意维持表面的平静",
        (0.8, 1.01): "用坚硬的面具时刻掩饰着真实感受",
    }
    _MOOD_ATOMS = {
        (0, 0.2): "非常低落",
        (0.2, 0.4): "有些低落",
        (0.4, 0.6): "平静",
        (0.6, 0.8): "心情还不错",
        (0.8, 1.01): "非常开心",
    }

    @staticmethod
    def _pick_atom(table: Dict[Tuple[float, float], str], value: float) -> str:
        for (lo, hi), desc in table.items():
            if lo <= value < hi:
                return desc
        return ""

    def compose_narration(
        self, state: Dict[str, Any]
    ) -> Tuple[str, Dict[str, float]]:
        """
        根据内部状态合成叙述文和描述原子权重

        参数:
            state: 包含 trauma/annoyance/mask_level/mood_score 等字段的字典

        返回:
            (narration_text, atom_weights)
        """
        trauma = state.get("trauma", state.get("trauma_score", 0))
        annoyance = state.get("annoyance", state.get("annoyance_score", 0))
        mask_level = state.get("mask_level", state.get("mask_rate", 0))
        mood = state.get("mood_score", state.get("average_mood_score", 0.5))
        atom_weights = {
            "trauma": float(trauma),
            "annoyance": annoyance / 100.0,
            "mask": float(mask_level),
            "mood": float(mood),
        }
        fragments = []
        mood_desc = self._pick_atom(self._MOOD_ATOMS, mood)
        if mood_desc:
            fragments.append(mood_desc)
        trauma_desc = self._pick_atom(self._TRAUMA_ATOMS, trauma)
        if trauma_desc:
            fragments.append(trauma_desc)
        annoyance_desc = self._pick_atom(self._ANNOYANCE_ATOMS, annoyance)
        if annoyance_desc:
            fragments.append(annoyance_desc)
        mask_desc = self._pick_atom(self._MASK_ATOMS, mask_level)
        if mask_desc:
            fragments.append(mask_desc)
        if not fragments:
            return "状态正常", atom_weights
        if len(fragments) == 1:
            return fragments[0], atom_weights
        return "，".join(fragments[:-1]) + "，" + fragments[-1], atom_weights

    def fade_memory_influence(
        self, atom_weights: Dict[str, float], elapsed_hours: float
    ) -> Dict[str, float]:
        """随时间衰减记忆对感知的影响"""
        decay_factor = max(0.05, math.exp(-0.15 * elapsed_hours))
        return {k: v * decay_factor for k, v in atom_weights.items()}


class _CrowdAnalyzer:
    """群体氛围分析器"""

    @staticmethod
    def assess_vibe(crowd_data: Dict) -> Tuple[CrowdVibe, float]:
        """
        评估群体活跃度

        参数:
            crowd_data: 包含 msg_count_10min / distinct_speakers / avg_interval_sec 等

        返回:
            (CrowdVibe, activity_score 0~1)
        """
        msg_count = crowd_data.get(
            "msg_count_10min", crowd_data.get("recent_message_count", 0)
        )
        speakers = crowd_data.get(
            "distinct_speakers", crowd_data.get("active_users", 1)
        )
        avg_interval = crowd_data.get("avg_interval_sec", 999)
        # 加权评分
        volume_score = min(1.0, msg_count / 30.0)
        diversity_score = min(1.0, speakers / 8.0)
        tempo_score = max(0.0, 1.0 - avg_interval / 120.0)
        raw = volume_score * 0.5 + diversity_score * 0.3 + tempo_score * 0.2
        if raw < 0.05:
            return CrowdVibe.DESOLATE, raw
        if raw < 0.2:
            return CrowdVibe.SPARSE, raw
        if raw < 0.5:
            return CrowdVibe.STEADY, raw
        if raw < 0.8:
            return CrowdVibe.LIVELY, raw
        return CrowdVibe.ERUPTING, raw


class _SelfInspector:
    """自省检查器 — 时间段、历史决策、内在状态"""

    _TIME_TAGS = [
        (5, "凌晨"),
        (8, "清晨"),
        (11, "上午"),
        (13, "中午"),
        (17, "下午"),
        (19, "傍晚"),
        (22, "夜晚"),
        (24, "深夜"),
    ]

    @staticmethod
    def detect_time_tag(hour: Optional[int] = None) -> str:
        if hour is None:
            import datetime

            hour = datetime.datetime.now().hour
        for boundary, tag in _SelfInspector._TIME_TAGS:
            if hour < boundary:
                return tag
        return "深夜"

    @staticmethod
    def summarize_self(self_data: Dict) -> str:
        """将自省数据汇总为一段描述"""
        lines = []
        decision_history = self_data.get("recent_decisions", [])
        if decision_history:
            lines.append(f"最近做了 {len(decision_history)} 个决策")
            last = decision_history[-1] if decision_history else None
            if isinstance(last, dict) and last.get("action"):
                lines.append(f"最后一个决策: {last['action']}")
        mood = self_data.get("mood_score", 0.5)
        if mood < 0.3:
            lines.append("情绪低落")
        elif mood > 0.7:
            lines.append("心情不错")
        return "；".join(lines) if lines else "状态正常"


class _RelationGauge:
    """人际感知"""

    @staticmethod
    def evaluate(relation_data: Dict) -> Tuple[float, str]:
        """
        评估人际关系数据

        返回:
            (receptiveness 0~1, summary_text)
        """
        from src.core.world_snapshot import get_relation_number

        sources = relation_data.get("relation_value_sources")
        has_social_source = isinstance(sources, dict) and "social_value" in sources
        social_value = get_relation_number(
            relation_data,
            "social_value",
            aliases=("favorability",),
        )
        affection = get_relation_number(
            relation_data,
            "affection",
            aliases=("favorability",),
        )
        favor = social_value if has_social_source or social_value != 0.0 else affection
        annoyance = get_relation_number(
            relation_data,
            "annoyance_value",
            aliases=("annoyance",),
        )
        freq = get_relation_number(
            relation_data,
            "interaction_count",
            aliases=("interaction_frequency",),
        )
        trust_value = get_relation_number(
            relation_data,
            "trust_value",
            aliases=("trust_score",),
        )
        # 简化评估
        favor_factor = min(1.0, max(0.0, (favor + 100) / 200))
        annoyance_penalty = min(1.0, annoyance / 100.0) * 0.3
        freq_bonus = min(0.15, freq / 100.0)
        trust_bonus = min(0.12, max(0.0, trust_value) / 250.0)
        receptiveness = max(
            0.0, min(1.0, favor_factor - annoyance_penalty + freq_bonus + trust_bonus)
        )
        parts = []
        if favor > 60:
            parts.append("好感度高")
        elif favor < -20:
            parts.append("好感度低")
        if trust_value > 40:
            parts.append("信任较高")
        if annoyance > 50:
            parts.append("对方较烦躁")
        if freq > 30:
            parts.append("互动频繁")
        summary = "，".join(parts) if parts else "关系一般"
        return receptiveness, summary


class _MessageDigester:
    """消息预处理器 — 提取关键信息并压缩"""

    @staticmethod
    def digest(raw_messages: List[Dict], max_entries: int = 12) -> str:
        if not raw_messages:
            return "（无最近消息）"
        selected = raw_messages[-max_entries:]
        lines = []
        for msg in selected:
            sender = msg.get("sender", msg.get("user_name", "未知"))
            content = msg.get("content", msg.get("text", ""))
            # 截断过长内容
            if len(content) > 80:
                content = content[:77] + "..."
            lines.append(f"[{sender}] {content}")
        return "\n".join(lines)

    @staticmethod
    def extract_fragments(text: str, top_n: int = 5) -> List[str]:
        """抽取高频文本片段，避免依赖固定停用词和关键词表。"""
        import re as _re

        chunks = _re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9_]{3,}", text)
        if not chunks:
            return []
        freq: Dict[str, int] = {}
        for chunk in chunks:
            token = chunk.strip().lower()
            if len(token) < 2:
                continue
            if len(token) <= 4:
                freq[token] = freq.get(token, 0) + 1
                continue
            for index in range(0, len(token) - 1):
                frag = token[index: index + 2]
                freq[frag] = freq.get(frag, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in ranked[:top_n]]


class _InterestAssessor:
    """兴趣度评估器 — 综合多维信号判定对话的吸引力"""

    @staticmethod
    def score_interest(
        crowd_activity: float,
        receptiveness: float,
        fragment_hits: int,
        mention_bot: bool = False,
        is_reply_to_bot: bool = False,
        message_density: float = 0.0,
    ) -> Tuple[AlertGrade, float]:
        """
        计算兴趣分并映射为 AlertGrade

        返回:
            (AlertGrade, interest_score 0~1)
        """
        base = crowd_activity * 0.25 + receptiveness * 0.25
        if mention_bot:
            base += 0.3
        if is_reply_to_bot:
            base += 0.2
        # 低人数/低活跃场景补偿：crowd_activity极低时给予基础分
        if crowd_activity < 0.1:
            base += 0.12
        fragment_bonus = min(0.12, fragment_hits * 0.025)
        density_bonus = min(0.08, message_density * 0.08)
        total = min(1.0, base + fragment_bonus + density_bonus)
        if total < 0.15:
            return AlertGrade.DORMANT, total
        if total < 0.35:
            return AlertGrade.FAINT, total
        if total < 0.65:
            return AlertGrade.MODERATE, total
        return AlertGrade.INTENSE, total


class AwarenessEngine:
    """
    感知觉察引擎

    收集群体氛围、自身状态、人际关系三个维度的感官数据，
    综合评估后以 LLM 或规则方式产出 AwarenessVerdict。
    """

    _solo: Optional["AwarenessEngine"] = None

    @classmethod
    def instance(cls) -> "AwarenessEngine":
        if cls._solo is None:
            cls._solo = cls()
        return cls._solo

    @classmethod
    def teardown(cls) -> None:
        cls._solo = None

    def __init__(self):
        self._narrator = BodyStateNarrator()
        self._crowd_analyzer = _CrowdAnalyzer()
        self._self_inspector = _SelfInspector()
        self._relation_gauge = _RelationGauge()
        self._digester = _MessageDigester()
        self._interest_assessor = _InterestAssessor()
        self._conf_cache: Dict[str, Any] = {}
        self._conf_ts: float = 0.0
        self._refresh_conf()
        self._counters = {
            "total_evaluations": 0,
            "llm_evaluations": 0,
            "rule_evaluations": 0,
        }

    def _refresh_conf(self) -> None:
        now = time.time()
        if now - self._conf_ts < 60.0:
            return
        hub = get_core_config()
        sched = hub.resolve_module_view("schedule").values
        self._conf_cache = {
            "llm_perception_enabled": sched.get(
                "llm_perception_enabled", True
            ),
            "max_perception_tokens": int(
                sched.get("perception_max_tokens", 800)
            ),
            "engagement_threshold": float(
                sched.get("engagement_threshold", 0.4)
            ),
        }
        self._conf_ts = now

    # ---- 核心入口 ----

    async def gather_awareness(
        self,
        channel_id: str,
        raw_messages: List[Dict],
        self_state: Dict,
        relation_state: Dict,
        crowd_info: Optional[Dict] = None,
        mention_bot: bool = False,
        is_reply_to_bot: bool = False,
    ) -> AwarenessVerdict:
        """
        核心感知流程：收集 → 分析 → 判定

        参数:
            channel_id: 频道ID
            raw_messages: 最近消息列表
            self_state: 自身状态（mood/trauma 等）
            relation_state: 关系数据（favor/annoyance 等）
            crowd_info: 群体统计信息
            mention_bot: 是否@了机器人
            is_reply_to_bot: 是否回复机器人

        返回:
            AwarenessVerdict
        """
        self._counters["total_evaluations"] += 1
        self._refresh_conf()
        # 第一步：采集快照
        snapshot = self._collect_snapshot(
            raw_messages,
            self_state,
            relation_state,
            crowd_info,
        )
        # 第二步：快速信号评估
        crowd_vibe, crowd_score = self._crowd_analyzer.assess_vibe(
            snapshot.crowd_data
        )
        receptiveness, relation_summary = self._relation_gauge.evaluate(
            snapshot.relation_data
        )
        time_tag = self._self_inspector.detect_time_tag()
        self_summary = self._self_inspector.summarize_self(snapshot.self_data)
        # 第三步：关键词与兴趣评估
        fragments = self._digester.extract_fragments(snapshot.raw_text)
        density = min(1.0, len(snapshot.raw_text.strip()) / 120.0)
        alert_grade, interest_score = self._interest_assessor.score_interest(
            crowd_score,
            receptiveness,
            len(fragments),
            mention_bot,
            is_reply_to_bot,
            density,
        )
        # 第四步：体感叙述
        narration, atom_weights = self._narrator.compose_narration(self_state)
        # 第五步：决定是否上报 LLM（高信号才走 LLM）
        use_llm = (
            self._conf_cache.get("llm_perception_enabled", True)
            and alert_grade >= AlertGrade.MODERATE
        )
        if use_llm:
            verdict = await self._evaluate_via_llm(
                snapshot,
                crowd_vibe,
                crowd_score,
                time_tag,
                self_summary,
                relation_summary,
                narration,
                fragments,
                mention_bot,
                is_reply_to_bot,
            )
            verdict.descriptor_atoms = atom_weights
            verdict.crowd_vibe = crowd_vibe
            verdict.timing_tag = time_tag
            verdict.body_narration = narration
            self._counters["llm_evaluations"] += 1
            logger.info(
                f"[觉察] {channel_id[:8]} | LLM判定 | "
                f"等级={verdict.alert_grade.label()} 参与={verdict.engagement_pull:.2f} "
                f"氛围={crowd_vibe.value}"
            )
            return verdict
        # 规则兜底
        self._counters["rule_evaluations"] += 1
        verdict = AwarenessVerdict(
            alert_grade=alert_grade,
            engagement_pull=interest_score,
            rationale=f"快速评估：兴趣={interest_score:.2f}，氛围={crowd_vibe.value}",
            inference_source="rule",
            crowd_vibe=crowd_vibe,
            body_narration=narration,
            descriptor_atoms=atom_weights,
            timing_tag=time_tag,
        )
        logger.info(
            f"[觉察] {channel_id[:8]} | 规则判定 | "
            f"等级={alert_grade.label()} 参与={interest_score:.2f} 氛围={crowd_vibe.value}"
        )
        return verdict

    # ---- 感官采集 ----

    def _collect_snapshot(
        self,
        raw_messages: List[Dict],
        self_state: Dict,
        relation_state: Dict,
        crowd_info: Optional[Dict],
    ) -> _SensorySnapshot:
        if crowd_info is None:
            crowd_info = self._infer_crowd_from_messages(raw_messages)
        message_text = self._digester.digest(raw_messages)
        full_text = " ".join(
            msg.get("content", msg.get("text", ""))
            for msg in raw_messages[-15:]
        )
        return _SensorySnapshot(
            crowd_data=crowd_info,
            self_data=self_state,
            relation_data=relation_state,
            message_digest=message_text,
            raw_text=full_text,
            timestamp=time.time(),
        )

    def _infer_crowd_from_messages(self, messages: List[Dict]) -> Dict:
        """从消息列表推断群体统计"""
        if not messages:
            return {
                "msg_count_10min": 0,
                "distinct_speakers": 0,
                "avg_interval_sec": 999,
            }
        now = time.time()
        ten_min_ago = now - 600
        recent = [
            m
            for m in messages
            if m.get("timestamp", m.get("ts", 0)) > ten_min_ago
        ]
        speakers = set()
        timestamps = []
        for m in recent:
            speakers.add(m.get("sender", m.get("user_id", "unknown")))
            ts = m.get("timestamp", m.get("ts", 0))
            if ts > 0:
                timestamps.append(ts)
        timestamps.sort()
        if len(timestamps) > 1:
            intervals = [
                timestamps[i + 1] - timestamps[i]
                for i in range(len(timestamps) - 1)
            ]
            avg_interval = sum(intervals) / max(1, len(intervals))
        else:
            avg_interval = 999
        return {
            "msg_count_10min": len(recent),
            "distinct_speakers": len(speakers),
            "avg_interval_sec": avg_interval,
        }

    # ---- LLM 感知评估 ----

    async def _evaluate_via_llm(
        self,
        snapshot: _SensorySnapshot,
        crowd_vibe: CrowdVibe,
        crowd_score: float,
        time_tag: str,
        self_summary: str,
        relation_summary: str,
        narration: str,
        fragments: List[str],
        mention_bot: bool,
        is_reply_to_bot: bool,
    ) -> AwarenessVerdict:
        prompt = self._craft_awareness_prompt(
            snapshot,
            crowd_vibe,
            crowd_score,
            time_tag,
            self_summary,
            relation_summary,
            narration,
            fragments,
            mention_bot,
            is_reply_to_bot,
        )
        raw = await self._call_llm(prompt)
        if not raw:
            logger.warning("[觉察] LLM 响应空白，降级为规则判定")
            return AwarenessVerdict(
                alert_grade=AlertGrade.FAINT,
                engagement_pull=0.2,
                rationale="LLM 空响应兜底",
                inference_source="fallback",
            )
        return self._parse_llm_verdict(raw)

    def _craft_awareness_prompt(
        self,
        snapshot: _SensorySnapshot,
        crowd_vibe: CrowdVibe,
        crowd_score: float,
        time_tag: str,
        self_summary: str,
        relation_summary: str,
        narration: str,
        fragments: List[str],
        mention_bot: bool,
        is_reply_to_bot: bool,
    ) -> str:
        kw_text = "、".join(fragments) if fragments else "无明显片段"
        special_flags = []
        if mention_bot:
            special_flags.append("用户@了我")
        if is_reply_to_bot:
            special_flags.append("用户回复了我的消息")
        flag_text = "；".join(special_flags) if special_flags else "无特殊信号"
        return (
            f"[感知分析任务]\n"
            f"时间：{time_tag}\n"
            f"群氛围：{crowd_vibe.value}（活跃度={crowd_score:.2f}）\n"
            f"我的状态：{narration}\n"
            f"自省：{self_summary}\n"
            f"关系：{relation_summary}\n"
            f"高频片段：{kw_text}\n"
            f"特殊信号：{flag_text}\n\n"
            f"最近对话：\n{snapshot.message_digest}\n\n"
            f"请判断我是否应该参与这段对话。回答 JSON：\n"
            f'{{"engagement_pull": 0.0到1.0, '
            f'"alert_level": "DORMANT/FAINT/MODERATE/INTENSE", '
            f'"rationale": "判断理由", '
            f'"supplementary_context": "如果需要额外信息填这里，否则null", '
            f'"need_more_context": false, '
            f'"context_type": null, '
            f'"context_duration_sec": 0}}'
        )

    def _parse_llm_verdict(self, raw: str) -> AwarenessVerdict:
        """解析 LLM 返回的 JSON 判定"""
        cleaned = raw.strip()
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not json_match:
            logger.warning("[觉察] 响应中未找到 JSON")
            return AwarenessVerdict(
                alert_grade=AlertGrade.FAINT,
                engagement_pull=0.2,
                rationale="JSON 解析失败兜底",
                inference_source="fallback",
            )
        json_str = json_match.group()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                fixed = re.sub(r",\s*}", "}", json_str)
                fixed = re.sub(r",\s*]", "]", fixed)
                data = json.loads(fixed)
            except Exception:
                return AwarenessVerdict(
                    alert_grade=AlertGrade.FAINT,
                    engagement_pull=0.2,
                    rationale="JSON 修复失败兜底",
                    inference_source="fallback",
                )
        if not isinstance(data, dict):
            return AwarenessVerdict(
                alert_grade=AlertGrade.FAINT,
                engagement_pull=0.2,
                rationale="非字典格式兜底",
                inference_source="fallback",
            )
        engagement = max(
            0.0, min(1.0, float(data.get("engagement_pull", 0.2)))
        )
        level_str = data.get("alert_level", "FAINT").upper()
        grade_mapping = {
            "DORMANT": AlertGrade.DORMANT,
            "FAINT": AlertGrade.FAINT,
            "MODERATE": AlertGrade.MODERATE,
            "INTENSE": AlertGrade.INTENSE,
        }
        grade = grade_mapping.get(level_str, AlertGrade.FAINT)
        rationale = data.get("rationale", "")
        supp = data.get("supplementary_context")
        ctx_demand = None
        if data.get("need_more_context"):
            ctx_type = data.get("context_type", "general")
            ctx_duration = self._sanitize_duration(
                data.get("context_duration_sec", 300)
            )
            ctx_demand = ContextDemand(
                demand_type=ctx_type,
                estimated_wait_sec=ctx_duration,
            )
        return AwarenessVerdict(
            alert_grade=grade,
            engagement_pull=engagement,
            rationale=rationale,
            supplementary_context=supp,
            context_demand=ctx_demand,
            inference_source="llm",
        )

    @staticmethod
    def _sanitize_duration(val: Any) -> int:
        try:
            sec = int(float(val))
            return max(10, min(sec, 3600))
        except (ValueError, TypeError):
            return 300

    # ---- LLM 调用 ----

    async def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config

            self._refresh_conf()
            _task_cfg = getattr(model_config.model_task_config, "utils", None)
            _task_name = _task_cfg if _task_cfg else getattr(
                model_config.model_task_config, "planner", None
            )
            if not _task_name:
                _task_name = model_config.model_task_config.focus_chat
            req = LLMRequest(
                _task_name,
                request_type="awareness_engine",
            )
            try:
                text, _ = await asyncio.wait_for(
                    req.generate_response_async(
                        prompt,
                        max_tokens=self._conf_cache["max_perception_tokens"],
                    ),
                    timeout=20.0,
                )
            except asyncio.TimeoutError:
                logger.warning("[觉察] LLM调用超时(20s)")
                text = None
            return text
        except Exception as exc:
            logger.error(f"[觉察] LLM 调用失败: {exc}")
            return None

    # ---- 统计与辅助 ----

    def collect_metrics(self) -> Dict[str, Any]:
        return dict(self._counters)

    def quick_assess(
        self,
        crowd_info: Dict,
        mention_bot: bool = False,
        is_reply_to_bot: bool = False,
    ) -> Tuple[AlertGrade, float]:
        """不走 LLM 的快速评估（仅用于预筛选）"""
        vibe, score = self._crowd_analyzer.assess_vibe(crowd_info)
        grade, pull = self._interest_assessor.score_interest(
            score,
            0.5,
            0,
            mention_bot,
            is_reply_to_bot,
        )
        return grade, pull

    def build_body_narration(self, state: Dict) -> str:
        """仅生成体感叙述（供外部模块调用）"""
        narration, _ = self._narrator.compose_narration(state)
        return narration

    def fade_atoms(
        self, atoms: Dict[str, float], hours: float
    ) -> Dict[str, float]:
        """衰减描述原子（供外部模块调用）"""
        return self._narrator.fade_memory_influence(atoms, hours)


def get_awareness_engine() -> AwarenessEngine:
    """获取全局感知觉察引擎实例"""
    return AwarenessEngine.instance()

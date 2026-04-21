"""
记忆深度优化套件 —— 去重、垃圾检测、内容压缩和数据库维护
融合三源设计:
  - XBcore(BufferCompressor): 消息统计 + 活跃度排序 + 纯代码处理
  - MaiBot(MemoryDeepOptimizer): 5阶段管线 + 多层签名 + 句子重要度压缩
  - MIMiaoCore(HistoryCondenser): 话题聚类 + 关键词倒排 + 双层遗忘
原创实现: 管线编排器 + 指纹生成器 + 分段打分压缩 + 报告聚合
"""

import re
import time
import json
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from src.common.logger import get_logger

logger = get_logger("深度优化")

# ═══════════════════════════════════════════
# 停用词表和垃圾模式
# ═══════════════════════════════════════════

_NOISE_WORDS = frozenset(
    {
        "的",
        "了",
        "是",
        "在",
        "有",
        "和",
        "与",
        "或",
        "但",
        "而",
        "也",
        "就",
        "都",
        "还",
        "又",
        "不",
        "吗",
        "吧",
        "呢",
        "啊",
        "哦",
        "哈",
        "嗯",
        "那",
        "这",
        "着",
        "过",
        "可",
        "把",
        "被",
        "对",
        "给",
        "让",
        "从",
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "this",
        "that",
        "of",
    }
)

_JUNK_MATCHERS = [
    re.compile(r"^[\s\.\,\!\?\;\:\-\_\=\+\*\&\^\%\$\#\@\~\`]+$"),
    re.compile(r"^(\.{3,}|。{3,}|？{3,}|！{3,})$"),
    re.compile(r"^https?://\S+$"),
    re.compile(r"^\[(图片|语音|视频|文件|表情)\]$"),
    re.compile(r"^[\W]+$", re.UNICODE),
]

# 句子中出现这些词会提升保留优先级
_ANCHOR_PHRASES = frozenset(
    {
        "记住",
        "重要",
        "提醒",
        "注意",
        "必须",
        "应该",
        "不要",
        "禁止",
        "请求",
        "保存",
        "核心",
        "关键",
        "永远",
        "千万",
    }
)


# ═══════════════════════════════════════════
# 辅助工具函数
# ═══════════════════════════════════════════


def gauge_token_count(text: str) -> int:
    """粗略估算Token数: 中文1字≈1token, 其他4字符≈1token"""
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    rest_chars = len(text) - cjk_chars
    return cjk_chars + (rest_chars // 4)


def render_size_label(byte_count: int) -> str:
    """将字节数转换为人类可读标签"""
    if byte_count < 1024:
        return f"{byte_count}B"
    if byte_count < 1048576:
        return f"{byte_count / 1024:.1f}KB"
    return f"{byte_count / 1048576:.1f}MB"


def _distill_meaningful_words(text: str, min_len: int = 2) -> Set[str]:
    """提取文本中有意义的词汇(过滤停用词)"""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {w for w in tokens if len(w) >= min_len and w not in _NOISE_WORDS}


# ═══════════════════════════════════════════
# 内容指纹生成器
# ═══════════════════════════════════════════


class ContentFingerprinter:
    """
    多层内容指纹生成器
    为每条记忆生成多层次的指纹用于去重判断:
    1. 精确指纹: 规范化文本的SHA-256前缀
    2. 语义指纹: 排序关键词联合哈希
    3. 结构指纹: JSON字典键结构哈希(仅对JSON内容)
    """

    def produce(self, content: str) -> List[str]:
        fingerprints = []
        # 第一层: 精确匹配指纹
        normalized = re.sub(r"\s+", " ", content.strip().lower())
        exact_fp = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        fingerprints.append(f"exact:{exact_fp}")
        # 第二层: 语义核心指纹
        meaningful = _distill_meaningful_words(content)
        if meaningful:
            sorted_core = "|".join(sorted(meaningful))
            semantic_fp = hashlib.sha256(
                sorted_core.encode("utf-8")
            ).hexdigest()[:16]
            fingerprints.append(f"semantic:{semantic_fp}")
        # 第三层: JSON结构指纹(仅限JSON内容)
        stripped = content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    key_sig = "|".join(sorted(parsed.keys()))
                    struct_fp = hashlib.sha256(
                        key_sig.encode("utf-8")
                    ).hexdigest()[:16]
                    fingerprints.append(f"struct:{struct_fp}")
            except (json.JSONDecodeError, ValueError):
                pass
        return fingerprints


# ═══════════════════════════════════════════
# 垃圾内容检测器
# ═══════════════════════════════════════════


class JunkDetector:
    """
    多条件垃圾内容检测器
    综合文本长度、字符类型、重复度、访问量和时间等因素判定
    """

    def is_junk(
        self,
        content: str,
        visit_count: int = 0,
        significance: float = 0.5,
        birth_ts: float = 0.0,
        aggressive: bool = False,
    ) -> bool:
        """判断一条记忆是否为垃圾内容"""
        text = content.strip()
        # 条件1: 过短(少于3个有效字符)
        if len(text) < 3:
            return True
        # 条件2: 纯符号/无实质内容
        if re.match(r"^[^\w\u4e00-\u9fff]*$", text):
            return True
        # 条件3: 极低多样性(超过10字符但独立字符≤3)
        if len(set(text)) <= 3 and len(text) > 10:
            return True
        # 条件4: 匹配预定义垃圾模式
        for matcher in _JUNK_MATCHERS:
            if matcher.match(text):
                return True
        # 条件5: 零访问+低重要度+超30天(自然遗忘)
        if birth_ts > 0:
            age_days = (time.time() - birth_ts) / 86400
            if visit_count == 0 and significance < 0.3 and age_days > 30:
                return True
        # 激进模式额外判定
        if aggressive:
            # 低访问+极低重要度
            if visit_count <= 1 and significance < 0.2:
                return True
            # 高重复词检测
            words = text.split()
            if words and len(words) > 5:
                most_freq = Counter(words).most_common(1)[0]
                if most_freq[1] > len(words) * 0.5:
                    return True
        return False


# ═══════════════════════════════════════════
# 重复消除器
# ═══════════════════════════════════════════


class DuplicateEliminator:
    """
    基于内容哈希的重复记忆消除器
    对重复组选择综合评分最高的条目保留
    """

    def _compute_content_digest(self, content: str) -> str:
        normalized = " ".join(content.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _evaluate_retention_score(self, memory) -> float:
        """综合评估一条记忆的保留价值"""
        score = 0.0
        visit_count = getattr(memory, "visit_count", 0) or 0
        score += visit_count * 80
        significance = (
            getattr(memory, "significance", getattr(memory, "importance", 0.5))
            or 0.5
        )
        score += significance * 60
        # 时间新近性(30天内的额外加分)
        birth = (
            getattr(memory, "birth_ts", getattr(memory, "created_at", 0)) or 0
        )
        age_days = (time.time() - birth) / 86400 if birth > 0 else 999
        score += max(0, 30 - age_days) * 2
        # 有摘要/有内容的额外加分
        if getattr(memory, "digest", None) or getattr(memory, "summary", None):
            score += 15
        content = getattr(memory, "content", "") or ""
        if len(content) > 50:
            score += 8
        return score

    def eliminate(self, MemoryModel, channel_id: Optional[str] = None) -> int:
        """执行去重，返回删除条数"""
        query = MemoryModel.select()
        if channel_id:
            query = query.where(MemoryModel.stream_id == channel_id)
        all_records = list(query)
        digest_groups: Dict[str, list] = defaultdict(list)
        for rec in all_records:
            content = getattr(rec, "content", "") or ""
            digest = self._compute_content_digest(content)
            digest_groups[digest].append(rec)
        removed = 0
        for _digest, group in digest_groups.items():
            if len(group) <= 1:
                continue
            # 选择保留价值最高的
            champion = max(group, key=self._evaluate_retention_score)
            for rec in group:
                rec_id = getattr(
                    rec, "record_id", getattr(rec, "memory_id", None)
                )
                champ_id = getattr(
                    champion, "record_id", getattr(champion, "memory_id", None)
                )
                if rec_id != champ_id:
                    content = getattr(rec, "content", "") or ""
                    rec.delete_instance()
                    removed += 1
        return removed


# ═══════════════════════════════════════════
# 内容压缩器(按句子重要度)
# ═══════════════════════════════════════════


class ContentCompressor:
    """
    长文本压缩器 —— 按句子重要度保留前50%关键句
    评估维度: 句长适度性、有意义词汇密度、锚定短语命中、数字/英文包含
    """

    def __init__(
        self, age_threshold_hours: float = 72.0, length_floor: int = 300
    ):
        self._age_threshold = age_threshold_hours * 3600
        self._length_floor = length_floor

    def _score_sentence(self, sentence: str) -> float:
        """为单个句子打重要度分"""
        score = 0.0
        # 长度适度性(10~50字最佳)
        slen = len(sentence)
        if 10 <= slen <= 50:
            score += 2.0
        elif slen > 50:
            score += 1.0
        # 有意义词汇密度
        words = _distill_meaningful_words(sentence)
        if words:
            density = min(1.0, len(words) / 5)
            score += density * 0.6
        # 锚定短语命中
        for phrase in _ANCHOR_PHRASES:
            if phrase in sentence:
                score += 3.0
                break
        # 数字和英文存在性
        if re.search(r"\d+", sentence):
            score += 1.0
        if re.search(r"[a-zA-Z]+", sentence):
            score += 0.5
        # 字符丰富度
        score += min(1.0, slen / 80) * 0.4
        return score

    def compress_text(self, content: str) -> str:
        """压缩长文本，保留重要句子"""
        sentences = re.split(r"[。！？.!?\n]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 2:
            return content
        scored = [(self._score_sentence(s), s) for s in sentences]
        scored.sort(key=lambda x: -x[0])
        keep_count = max(1, len(scored) // 2)
        selected = {s for _, s in scored[:keep_count]}
        # 按原始顺序重组
        preserved = [s for s in sentences if s in selected]
        if preserved:
            return "。".join(preserved) + "。"
        return content[:100] + "..."

    def compress_memory(self, memory, MemoryModel) -> int:
        """
        压缩单条记忆的内容(就地修改)
        返回节省的字节数
        """
        content = getattr(memory, "content", "") or ""
        if len(content) <= self._length_floor:
            return 0
        birth_ts = (
            getattr(memory, "birth_ts", getattr(memory, "created_at", 0)) or 0
        )
        age = time.time() - birth_ts
        if age < self._age_threshold:
            return 0
        compressed = self.compress_text(content)
        if len(compressed) >= len(content) * 0.7:
            return 0  # 压缩不明显则跳过
        original_bytes = len(content.encode("utf-8"))
        compressed_bytes = len(compressed.encode("utf-8"))
        saved = original_bytes - compressed_bytes
        memory.content = compressed
        # 标记为已压缩
        if hasattr(memory, "entry_category"):
            memory.entry_category = "compressed"
        elif hasattr(memory, "memory_type"):
            memory.memory_type = "compressed"
        memory.save()
        return saved


# ═══════════════════════════════════════════
# 存储优化管线编排器
# ═══════════════════════════════════════════


class StorageOptimizationSuite:
    """
    存储优化主引擎 —— 编排指纹分析、去重、压缩、垃圾清理和DB维护五个阶段
    """

    def __init__(self):
        self._fingerprinter = ContentFingerprinter()
        self._junk_detector = JunkDetector()
        self._dedup = DuplicateEliminator()
        self._compressor = ContentCompressor()
        self._last_execution_ts: float = 0.0
        self._cooldown_seconds: float = 21600.0  # 默认6小时最短间隔
        self._accumulated_report: Dict[str, int] = self._empty_report()

    def _empty_report(self) -> Dict[str, int]:
        return {
            "scanned_total": 0,
            "junk_purged": 0,
            "duplicates_removed": 0,
            "compressed_count": 0,
            "bytes_recovered": 0,
            "tokens_recovered": 0,
        }

    def _fetch_memory_model(self):
        try:
            from src.common.database.database_model import MemoryRecord

            return MemoryRecord
        except (ImportError, Exception):
            return None

    def is_due(self) -> bool:
        """检查是否到了可执行优化的时间，动态调整冷却间隔"""
        elapsed = time.time() - self._last_execution_ts
        try:
            MemModel = self._fetch_memory_model()
            if MemModel is not None:
                _one_hour_ago = time.time() - 3600
                _recent_count = MemModel.select().where(
                    MemModel.created_at >= _one_hour_ago
                ).count()
                if _recent_count >= 50:
                    _dynamic_cooldown = self._cooldown_seconds * 0.25
                elif _recent_count >= 20:
                    _dynamic_cooldown = self._cooldown_seconds * 0.5
                elif _recent_count >= 5:
                    _dynamic_cooldown = self._cooldown_seconds
                else:
                    _dynamic_cooldown = self._cooldown_seconds * 1.5
                return elapsed >= _dynamic_cooldown
        except Exception:
            pass
        return elapsed >= self._cooldown_seconds

    async def execute(
        self, channel_id: Optional[str] = None, aggressive: bool = False
    ) -> Dict[str, Any]:
        """
        执行完整的5阶段优化管线
        返回本次优化的统计报告
        """
        report = self._empty_report()
        self._last_execution_ts = time.time()
        MemModel = self._fetch_memory_model()
        if not MemModel:
            logger.warning("[优化管线] 记忆模型不可用，跳过")
            return report
        start_ts = time.time()
        logger.info("[优化管线] 启动...")
        try:
            # 阶段1: 指纹分析(统计总量)
            query = MemModel.select()
            if channel_id:
                query = query.where(MemModel.stream_id == channel_id)
            all_records = list(query)
            report["scanned_total"] = len(all_records)
            # 阶段2: 垃圾清理
            purged = self._phase_purge_junk(all_records, report, aggressive)
            # 阶段3: 去重
            dedup_count = self._dedup.eliminate(MemModel, channel_id)
            report["duplicates_removed"] = dedup_count
            # 阶段4: 长内容压缩
            self._phase_compress_content(MemModel, channel_id, report)
            # 阶段5: 数据库结构优化
            self._phase_db_maintenance()
            elapsed = round(time.time() - start_ts, 2)
            logger.info(
                f"[优化管线] 完成({elapsed}秒) | "
                f"清理{report['junk_purged']}条 | "
                f"去重{report['duplicates_removed']}条 | "
                f"压缩{report['compressed_count']}条 | "
                f"释放{render_size_label(report['bytes_recovered'])}"
            )
        except Exception as exc:
            logger.error(f"[优化管线] 执行异常: {exc}")
        self._accumulated_report = report
        return report

    def _phase_purge_junk(
        self, records: list, report: Dict, aggressive: bool
    ) -> int:
        """阶段2: 垃圾清理"""
        purged = 0
        for rec in records:
            content = getattr(rec, "content", "") or ""
            visit_count = getattr(rec, "visit_count", 0) or 0
            significance = getattr(rec, "significance", 0.5) or 0.5
            birth_ts = (
                getattr(rec, "birth_ts", getattr(rec, "created_at", 0)) or 0
            )
            if self._junk_detector.is_junk(
                content, visit_count, significance, birth_ts, aggressive
            ):
                content_bytes = len(content.encode("utf-8"))
                report["bytes_recovered"] += content_bytes
                report["tokens_recovered"] += gauge_token_count(content)
                rec.delete_instance()
                purged += 1
        report["junk_purged"] = purged
        if purged > 0:
            logger.info(f"[优化管线·清理] 移除{purged}条垃圾记忆")
        return purged

    def _phase_compress_content(
        self, MemModel, channel_id: Optional[str], report: Dict
    ):
        """阶段4: 长内容压缩"""
        now = time.time()
        age_cutoff = now - self._compressor._age_threshold
        query = (
            MemModel.select().where(MemModel.birth_ts < age_cutoff)
            if hasattr(MemModel, "birth_ts")
            else (
                MemModel.select().where(MemModel.created_at < age_cutoff)
                if hasattr(MemModel, "created_at")
                else MemModel.select()
            )
        )
        if channel_id:
            query = query.where(MemModel.stream_id == channel_id)
        compressed_count = 0
        for rec in query:
            saved = self._compressor.compress_memory(rec, MemModel)
            if saved > 0:
                report["bytes_recovered"] += saved
                report["tokens_recovered"] += saved // 4
                compressed_count += 1
        report["compressed_count"] = compressed_count
        if compressed_count > 0:
            logger.info(f"[优化管线·压缩] 压缩{compressed_count}条长记忆")

    def _phase_db_maintenance(self):
        """阶段5: 数据库结构维护"""
        try:
            from src.common.database.database import db

            db.execute_sql("REINDEX")
            db.execute_sql("VACUUM")
            db.execute_sql("ANALYZE")
            logger.debug("[优化管线] 数据库维护完成(REINDEX+VACUUM+ANALYZE)")
        except Exception as exc:
            logger.debug(f"[优化管线] 数据库维护跳过: {exc}")

    def latest_report(self) -> Dict[str, int]:
        """获取最近一次优化的统计报告"""
        return self._accumulated_report.copy()

    def compile_health_assessment(
        self, channel_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        编译记忆仓库的健康评估报告
        包括: 总量、大小分布、类型分布、优化空间估算
        """
        MemModel = self._fetch_memory_model()
        if not MemModel:
            return {"error": "记忆模型不可用"}
        query = MemModel.select()
        if channel_id:
            query = query.where(MemModel.stream_id == channel_id)
        records = list(query)
        total_bytes = 0
        size_buckets = {
            "micro": 0,
            "compact": 0,
            "standard": 0,
            "oversized": 0,
        }
        category_tally: Counter = Counter()
        for rec in records:
            content = getattr(rec, "content", "") or ""
            content_bytes = len(content.encode("utf-8"))
            total_bytes += content_bytes
            if content_bytes < 100:
                size_buckets["micro"] += 1
            elif content_bytes < 1024:
                size_buckets["compact"] += 1
            elif content_bytes < 10240:
                size_buckets["standard"] += 1
            else:
                size_buckets["oversized"] += 1
            cat = getattr(
                rec, "entry_category", getattr(rec, "memory_type", "unknown")
            )
            category_tally[cat or "unknown"] += 1
        # 优化空间估算
        oversized_records = [
            rec
            for rec in records
            if len(getattr(rec, "content", "") or "") > 1000
        ]
        zero_visit_records = [
            rec
            for rec in records
            if (getattr(rec, "visit_count", 0) or 0) == 0
        ]
        low_sig_records = [
            rec
            for rec in records
            if (getattr(rec, "significance", 0.5) or 0.5) < 0.3
        ]
        return {
            "total_records": len(records),
            "total_size_mb": round(total_bytes / 1048576, 2),
            "category_distribution": dict(category_tally),
            "size_distribution": size_buckets,
            "avg_bytes_per_record": (
                round(total_bytes / len(records)) if records else 0
            ),
            "estimated_tokens": sum(
                gauge_token_count(getattr(r, "content", "") or "")
                for r in records
            ),
            "optimization_targets": {
                "oversized_count": len(oversized_records),
                "zero_visit_count": len(zero_visit_records),
                "low_significance_count": len(low_sig_records),
            },
        }


# ═══════════════════════════════════════════
# 单例管理
# ═══════════════════════════════════════════

_optimizer_suite_ref: Optional[StorageOptimizationSuite] = None


def acquire_optimization_suite() -> StorageOptimizationSuite:
    """获取全局存储优化套件实例"""
    global _optimizer_suite_ref
    if _optimizer_suite_ref is None:
        _optimizer_suite_ref = StorageOptimizationSuite()
    return _optimizer_suite_ref


# 短别名(兼容其他模块的导入习惯)
get_deep_optimizer = acquire_optimization_suite

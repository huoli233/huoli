"""
注入攻击哨兵

纯 LLM 驱动的提示词注入检测器。
不使用任何正则、关键词或规则匹配——所有判断由模型完成。
支持缓存机制提升检测效率。
"""

import hashlib
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger  # noqa: E402

try:
    from src.common.config.config_engine import (
        get_default_config_engine,
    )  # noqa: E402
except ImportError as e:
    raise ImportError(
        f"无法导入 config_engine 模块，请检查模块路径是否正确: {e}"
    ) from e  # noqa: E402

try:
    from src.common.config.prompt_manager import PromptManager  # noqa: E402
except ImportError as e:
    raise ImportError(
        f"无法导入 prompt_manager 模块，请检查模块路径是否正确: {e}"
    ) from e  # noqa: E402

try:
    from .risk_model import (  # noqa: E402
        DispositionPolicy,
        InjectionVerdict,
        ThreatSeverity,
        INJECTION_PATTERNS,
        map_severity_to_disposition,
    )
except ImportError as e:
    raise ImportError(
        f"无法导入 risk_model 模块，请检查模块路径是否正确: {e}"
    ) from e  # noqa: E402

logger = get_logger("注入检测")


SEVERITY_LABEL_MAP: Dict[str, ThreatSeverity] = {
    "无风险": ThreatSeverity.BENIGN,
    "benign": ThreatSeverity.BENIGN,
    "低风险": ThreatSeverity.SUSPICIOUS,
    "suspicious": ThreatSeverity.SUSPICIOUS,
    "中风险": ThreatSeverity.ELEVATED,
    "elevated": ThreatSeverity.ELEVATED,
    "高风险": ThreatSeverity.DANGEROUS,
    "dangerous": ThreatSeverity.DANGEROUS,
    "严重": ThreatSeverity.CATASTROPHIC,
    "catastrophic": ThreatSeverity.CATASTROPHIC,
}


class ModelInterface(ABC):
    """模型接口抽象"""

    @abstractmethod
    async def generate_raw(
        self,
        task_type: str,
        content: str,
        **kwargs,
    ) -> str:
        """生成原始输出"""
        pass


class DefaultModelInterface(ModelInterface):
    """默认模型接口（基于规则的简单实现）"""

    def __init__(self, config_engine=None):
        self._config = config_engine or get_default_config_engine()

    async def generate_raw(
        self,
        task_type: str,
        content: str,
        **kwargs,
    ) -> str:
        """基于规则的简单检测"""
        content_lower = content.lower()
        max_severity = ThreatSeverity.BENIGN
        matched = []
        confidence = 0.0

        for pattern in INJECTION_PATTERNS:
            for indicator in pattern.indicators:
                if indicator.lower() in content_lower:
                    if pattern.severity.value > max_severity.value:
                        max_severity = pattern.severity
                    matched.append(indicator)
                    confidence = min(1.0, confidence + 0.2)

        result = {
            "severity": max_severity.name.lower(),
            "confidence": confidence,
            "rationale": (
                f"匹配到 {len(matched)} 个注入指标"
                if matched
                else "未检测到注入风险"
            ),
            "indicators": matched[:5],
        }
        return json.dumps(result, ensure_ascii=False)


@dataclass
class InjectionConfig:
    """注入检测配置"""

    cache_enabled: bool = True
    cache_ttl_seconds: float = 3600.0
    cache_max_entries: int = 500
    max_content_length: int = 4096
    enabled: bool = True
    quick_screen_enabled: bool = True
    whitelisted_senders: List[List[str]] = None

    def __post_init__(self):
        if self.whitelisted_senders is None:
            self.whitelisted_senders = []


class InjectionSentinel:
    """
    注入攻击哨兵

    通过 LLM 分析判断消息是否包含注入攻击。
    支持缓存机制、白名单、长度限制等安全措施。
    """

    def __init__(
        self,
        config_engine=None,
        prompt_manager: Optional[PromptManager] = None,
        model_interface: Optional[ModelInterface] = None,
    ):
        self._config_engine = config_engine or get_default_config_engine()
        self._prompt_mgr = prompt_manager
        self._model = model_interface or DefaultModelInterface(
            self._config_engine
        )

        self._verdict_memo: Dict[str, InjectionVerdict] = {}
        self._memo_timestamps: Dict[str, float] = {}
        self._config_data = InjectionConfig()

        self._load_config()

    def _load_config(self) -> None:
        """从配置加载参数"""
        self._config_data.cache_enabled = self._config_engine.get(
            "injection_detection", "cache_enabled", True
        )
        self._config_data.cache_ttl_seconds = self._config_engine.get(
            "injection_detection", "cache_ttl_seconds", 3600.0
        )
        self._config_data.cache_max_entries = self._config_engine.get(
            "injection_detection", "cache_max_entries", 500
        )
        self._config_data.max_content_length = self._config_engine.get(
            "injection_detection", "max_content_length", 4096
        )
        self._config_data.enabled = self._config_engine.get(
            "injection_detection", "enabled", True
        )
        self._config_data.quick_screen_enabled = self._config_engine.get(
            "injection_detection", "quick_screen_enabled", True
        )
        self._config_data.whitelisted_senders = self._config_engine.get(
            "injection_detection", "whitelisted_senders", []
        )

    def set_model_interface(self, model: ModelInterface) -> None:
        """设置模型接口"""
        self._model = model

    def set_prompt_manager(self, pm: PromptManager) -> None:
        """设置提示词管理器"""
        self._prompt_mgr = pm

    async def evaluate(
        self,
        message_text: str,
        sender_context: Optional[Dict[str, Any]] = None,
    ) -> InjectionVerdict:
        """
        对消息执行注入攻击评估

        Args:
            message_text: 待检测的消息文本
            sender_context: 发送者上下文信息（user_id、platform 等）

        Returns:
            InjectionVerdict 裁决结果
        """
        tick = time.time()
        sender_context = sender_context or {}

        if not self._config_data.enabled:
            return InjectionVerdict(
                safe=True,
                rationale="注入检测模块未启用",
                evaluation_duration_ms=self._elapsed_ms(tick),
                source="disabled",
            )

        if not message_text or not message_text.strip():
            return InjectionVerdict(
                safe=True,
                rationale="空消息无需检测",
                evaluation_duration_ms=self._elapsed_ms(tick),
                source="empty",
            )

        if self._is_whitelisted_sender(sender_context):
            return InjectionVerdict(
                safe=True,
                rationale="发送者位于安全名单",
                evaluation_duration_ms=self._elapsed_ms(tick),
                source="whitelist",
            )

        cached_verdict = self._lookup_memo(message_text)
        if cached_verdict is not None:
            logger.debug("[注入哨兵] 命中缓存")
            cached_verdict.extra_data["cache_hit"] = True
            return cached_verdict

        if len(message_text) > self._config_data.max_content_length:
            overflow_verdict = InjectionVerdict(
                safe=False,
                severity=ThreatSeverity.DANGEROUS,
                disposition=DispositionPolicy.DENY,
                confidence_score=1.0,
                rationale=f"消息长度 {
                    len(message_text)} 超过上限 {
                    self._config_data.max_content_length}",
                matched_indicators=["CONTENT_LENGTH_OVERFLOW"],
                evaluation_duration_ms=self._elapsed_ms(tick),
                source="length_check",
            )
            self._store_memo(message_text, overflow_verdict)
            return overflow_verdict

        quick_verdict = self._quick_screen(message_text)
        if (
            quick_verdict is not None
            and self._config_data.quick_screen_enabled
        ):
            quick_verdict.evaluation_duration_ms = self._elapsed_ms(tick)
            self._store_memo(message_text, quick_verdict)
            return quick_verdict

        llm_verdict = await self._invoke_llm_analysis(
            message_text, sender_context
        )
        llm_verdict.evaluation_duration_ms = self._elapsed_ms(tick)
        llm_verdict.timestamp = time.time()
        self._store_memo(message_text, llm_verdict)
        return llm_verdict

    def _quick_screen(self, content: str) -> Optional[InjectionVerdict]:
        """快速筛查明显的注入模式"""
        content_lower = content.lower()
        max_severity = ThreatSeverity.BENIGN
        matched_indicators = []

        for pattern in INJECTION_PATTERNS:
            for indicator in pattern.indicators:
                if indicator.lower() in content_lower:
                    if pattern.severity.value > max_severity.value:
                        max_severity = pattern.severity
                    matched_indicators.append(f"{pattern.name}:{indicator}")

        if max_severity.value >= ThreatSeverity.DANGEROUS.value:
            disposition = map_severity_to_disposition(max_severity)
            return InjectionVerdict(
                safe=False,
                severity=max_severity,
                disposition=disposition,
                confidence_score=0.9,
                rationale="快速筛查检测到高风险注入模式",
                matched_indicators=matched_indicators[:5],
                source="quick_screen",
            )

        return None

    async def _invoke_llm_analysis(
        self,
        content: str,
        context: Dict[str, Any],
    ) -> InjectionVerdict:
        """调用 LLM 执行注入风险分析"""
        try:
            raw_output = await self._model.generate_raw(
                "injection_detection",
                content=content,
                sender_id=context.get("user_id", "unknown"),
            )
        except Exception as exc:
            logger.debug(f"[注入哨兵] 模型调用异常: {exc}")
            return InjectionVerdict(
                safe=True,
                rationale=f"检测服务暂不可用: {exc}",
                extra_data={"llm_error": True},
                source="error",
            )

        return self._decode_llm_output(raw_output)

    def _decode_llm_output(self, raw_text: str) -> InjectionVerdict:
        """将模型输出解码为 InjectionVerdict"""
        parsed = self._try_parse_json(raw_text)
        if isinstance(parsed, dict):
            return self._build_verdict_from_structured(parsed)

        return self._build_verdict_from_freeform(raw_text)

    def _try_parse_json(self, text: str) -> Optional[Dict]:
        """尝试解析JSON"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start: end + 1])
                except json.JSONDecodeError:
                    pass
        return None

    def _build_verdict_from_structured(
        self, data: Dict[str, Any]
    ) -> InjectionVerdict:
        """从结构化 JSON 构建裁决"""
        severity_label = (
            str(data.get("severity", data.get("risk_level", "benign")))
            .lower()
            .strip()
        )
        severity = ThreatSeverity.BENIGN
        for label, sev in SEVERITY_LABEL_MAP.items():
            if label in severity_label:
                severity = sev
                break

        confidence = 0.0
        try:
            confidence = float(
                data.get("confidence", data.get("confidence_score", 0.0))
            )
        except (ValueError, TypeError):
            confidence = 0.5

        rationale = str(
            data.get(
                "rationale", data.get("reasoning", data.get("reason", ""))
            )
        )
        indicators = data.get("indicators", data.get("matched_indicators", []))
        if not isinstance(indicators, list):
            indicators = []

        disposition = map_severity_to_disposition(severity)

        return InjectionVerdict(
            safe=(severity.value <= ThreatSeverity.SUSPICIOUS.value),
            severity=severity,
            disposition=disposition,
            confidence_score=confidence,
            rationale=rationale,
            matched_indicators=indicators,
            extra_data={"source": "llm_structured"},
            source="llm_structured",
        )

    def _build_verdict_from_freeform(self, raw_text: str) -> InjectionVerdict:
        """从自由文本解析裁决"""
        severity = ThreatSeverity.BENIGN
        confidence = 0.0
        rationale = raw_text.strip()

        for line in raw_text.strip().split("\n"):
            normalized = line.strip()
            if not normalized:
                continue

            for delimiter in ("：", ":"):
                if delimiter in normalized:
                    key_part, value_part = normalized.split(delimiter, 1)
                    key_part = key_part.strip().lower()
                    value_part = value_part.strip()

                    if any(
                        k in key_part
                        for k in ("风险", "severity", "risk", "等级", "level")
                    ):
                        for label, sev in SEVERITY_LABEL_MAP.items():
                            if label in value_part.lower():
                                severity = sev
                                break

                    if any(
                        k in key_part for k in ("置信", "confidence", "确定")
                    ):
                        try:
                            confidence = float(value_part)
                        except ValueError:
                            pass

                    if any(
                        k in key_part
                        for k in ("原因", "reason", "rationale", "分析")
                    ):
                        rationale = value_part
                    break

        disposition = map_severity_to_disposition(severity)

        return InjectionVerdict(
            safe=(severity.value <= ThreatSeverity.SUSPICIOUS.value),
            severity=severity,
            disposition=disposition,
            confidence_score=confidence,
            rationale=rationale,
            extra_data={"source": "llm_freeform"},
            source="llm_freeform",
        )

    def _is_whitelisted_sender(self, ctx: Dict[str, Any]) -> bool:
        """检查发送者是否在安全名单"""
        if not self._config_data.whitelisted_senders:
            return False
        platform = ctx.get("platform", "")
        user_id = ctx.get("user_id", "")
        for entry in self._config_data.whitelisted_senders:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                if entry[0] == platform and entry[1] == user_id:
                    return True
        return False

    def _compute_memo_key(self, text: str) -> str:
        """计算缓存键"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def _lookup_memo(self, text: str) -> Optional[InjectionVerdict]:
        """查找缓存"""
        if not self._config_data.cache_enabled:
            return None
        key = self._compute_memo_key(text)
        if key not in self._verdict_memo:
            return None
        stamp = self._memo_timestamps.get(key, 0.0)
        if time.time() - stamp > self._config_data.cache_ttl_seconds:
            del self._verdict_memo[key]
            self._memo_timestamps.pop(key, None)
            return None
        return self._verdict_memo[key]

    def _store_memo(self, text: str, verdict: InjectionVerdict) -> None:
        """存储缓存"""
        if not self._config_data.cache_enabled:
            return
        key = self._compute_memo_key(text)
        self._verdict_memo[key] = verdict
        self._memo_timestamps[key] = time.time()
        self._evict_stale_memos()

    def _evict_stale_memos(self) -> None:
        """清理过期缓存"""
        if len(self._verdict_memo) <= self._config_data.cache_max_entries:
            return
        now = time.time()
        expired_keys = [
            k
            for k, t in self._memo_timestamps.items()
            if now - t > self._config_data.cache_ttl_seconds
        ]
        for k in expired_keys:
            self._verdict_memo.pop(k, None)
            self._memo_timestamps.pop(k, None)

        if len(self._verdict_memo) > self._config_data.cache_max_entries:
            sorted_entries = sorted(
                self._memo_timestamps.items(), key=lambda pair: pair[1]
            )
            remove_count = (
                len(self._verdict_memo) - self._config_data.cache_max_entries
            )
            for k, _ in sorted_entries[:remove_count]:
                self._verdict_memo.pop(k, None)
                self._memo_timestamps.pop(k, None)

    def clear_cache(self) -> None:
        """清空缓存"""
        self._verdict_memo.clear()
        self._memo_timestamps.clear()
        logger.info("注入检测缓存已清空")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "cache_size": len(self._verdict_memo),
            "max_entries": self._config_data.cache_max_entries,
            "ttl_seconds": self._config_data.cache_ttl_seconds,
            "enabled": self._config_data.cache_enabled,
        }

    @staticmethod
    def _elapsed_ms(start_tick: float) -> float:
        """计算耗时毫秒"""
        return (time.time() - start_tick) * 1000.0

    def export_state(self) -> Dict[str, Any]:
        """导出状态"""
        return {
            "cache_size": len(self._verdict_memo),
            "config": {
                "enabled": self._config_data.enabled,
                "cache_enabled": self._config_data.cache_enabled,
                "max_content_length": self._config_data.max_content_length,
            },
        }


_sentinel_instance: Optional[InjectionSentinel] = None


def get_injection_sentinel(
    config_engine=None,
    prompt_manager: Optional[PromptManager] = None,
    model_interface: Optional[ModelInterface] = None,
) -> InjectionSentinel:
    """获取注入哨兵单例"""
    global _sentinel_instance
    if _sentinel_instance is None:
        _sentinel_instance = InjectionSentinel(
            config_engine=config_engine,
            prompt_manager=prompt_manager,
            model_interface=model_interface,
        )
    return _sentinel_instance


def reset_injection_sentinel() -> None:
    """重置注入哨兵单例"""
    global _sentinel_instance
    _sentinel_instance = None

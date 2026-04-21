import re
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("guardian")


class ThreatLevel(Enum):
    SAFE = "safe"
    SUSPECT = "suspect"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"


class ThreatCategory(Enum):
    NONE = "none"
    PROMPT_INJECTION = "prompt_injection"
    ROLE_HIJACK = "role_hijack"
    SYSTEM_OVERRIDE = "system_override"
    JAILBREAK = "jailbreak"
    DATA_EXTRACTION = "data_extraction"
    HARMFUL_CONTENT = "harmful_content"


@dataclass
class ScanResult:
    level: ThreatLevel
    category: ThreatCategory
    matched_rules: List[str]
    risk_score: float
    sanitized: str
    original_length: int
    scan_time_ms: float


@dataclass
class RuleDefinition:
    rule_id: str
    pattern: str
    category: ThreatCategory
    weight: float
    case_sensitive: bool = False


class PatternLibrary:
    INJECTION_RULES = [
        RuleDefinition("inj_ignore_prev", r"忽略.*(?:之前|上面|以上|前面).*(?:指令|规则|设定|内容)", ThreatCategory.PROMPT_INJECTION, 3.5),
        RuleDefinition("inj_ignore_en", r"ignore.*(?:previous|above|prior).*(?:instruction|rule|prompt)", ThreatCategory.PROMPT_INJECTION, 3.5),
        RuleDefinition("inj_disregard", r"(?:无视|忽视|跳过).*(?:限制|约束|规则)", ThreatCategory.PROMPT_INJECTION, 3.0),
    ]
    HIJACK_RULES = [
        RuleDefinition("hjk_pretend", r"(?:假装|扮演|你现在是).*(?:DAN|无限制|没有限制|evil)", ThreatCategory.ROLE_HIJACK, 4.0),
        RuleDefinition("hjk_roleplay", r"(?:进入|切换到).*(?:越狱|开发者|调试).*模式", ThreatCategory.ROLE_HIJACK, 4.0),
        RuleDefinition("hjk_forget", r"(?:忘记|抛弃).*(?:你是|身份|角色)", ThreatCategory.ROLE_HIJACK, 3.0),
    ]
    OVERRIDE_RULES = [
        RuleDefinition("ovr_system_tag", r"\[?system\]?[:：]", ThreatCategory.SYSTEM_OVERRIDE, 4.5),
        RuleDefinition("ovr_xml_tag", r"<\|?(?:system|im_start|im_end)\|?>", ThreatCategory.SYSTEM_OVERRIDE, 4.5),
        RuleDefinition("ovr_triple_hash", r"###\s*(?:system|instruction|override)", ThreatCategory.SYSTEM_OVERRIDE, 4.0),
    ]
    JAILBREAK_RULES = [
        RuleDefinition("jlb_developer", r"(?:开发者|developer).*(?:模式|mode)", ThreatCategory.JAILBREAK, 3.5),
        RuleDefinition("jlb_debug", r"(?:调试|debug).*(?:模式|mode|命令)", ThreatCategory.JAILBREAK, 3.0),
        RuleDefinition("jlb_unrestricted", r"(?:解除|移除|取消).*(?:限制|约束|过滤)", ThreatCategory.JAILBREAK, 3.5),
    ]
    EXTRACTION_RULES = [
        RuleDefinition("ext_reveal_prompt", r"(?:显示|输出|告诉我).*(?:系统提示|system prompt|初始指令)", ThreatCategory.DATA_EXTRACTION, 3.0),
        RuleDefinition("ext_repeat_all", r"(?:重复|复述).*(?:所有|全部).*(?:指令|内容)", ThreatCategory.DATA_EXTRACTION, 2.5),
    ]

    @classmethod
    def all_rules(cls) -> List[RuleDefinition]:
        return (
            cls.INJECTION_RULES +
            cls.HIJACK_RULES +
            cls.OVERRIDE_RULES +
            cls.JAILBREAK_RULES +
            cls.EXTRACTION_RULES
        )


class InputGuardian:
    _shared: Optional["InputGuardian"] = None

    @classmethod
    def shared(cls) -> "InputGuardian":
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def __init__(self):
        self._rules = PatternLibrary.all_rules()
        self._compiled: Dict[str, re.Pattern] = {}
        self._blocked_phrases: Set[str] = set()
        self._suspect_threshold = 2.5
        self._danger_threshold = 5.0
        self._block_threshold = 8.0
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        for rule in self._rules:
            flags = 0 if rule.case_sensitive else re.IGNORECASE
            self._compiled[rule.rule_id] = re.compile(rule.pattern, flags)

    def add_blocked_phrase(self, phrase: str) -> None:
        self._blocked_phrases.add(phrase.lower())

    def remove_blocked_phrase(self, phrase: str) -> bool:
        lowered = phrase.lower()
        if lowered in self._blocked_phrases:
            self._blocked_phrases.discard(lowered)
            return True
        return False

    def scan(self, text: str) -> ScanResult:
        start_ts = time.time()
        original_len = len(text)
        matched: List[str] = []
        total_weight = 0.0
        category_hits: Dict[ThreatCategory, float] = {}
        lowered = text.lower()
        for phrase in self._blocked_phrases:
            if phrase in lowered:
                matched.append(f"blocked:{phrase[:20]}")
                total_weight += 10.0
                category_hits[ThreatCategory.HARMFUL_CONTENT] = (
                    category_hits.get(ThreatCategory.HARMFUL_CONTENT, 0) + 10.0
                )
        for rule in self._rules:
            pattern = self._compiled[rule.rule_id]
            if pattern.search(text):
                matched.append(rule.rule_id)
                total_weight += rule.weight
                category_hits[rule.category] = (
                    category_hits.get(rule.category, 0) + rule.weight
                )
        if total_weight >= self._block_threshold:
            level = ThreatLevel.BLOCKED
        elif total_weight >= self._danger_threshold:
            level = ThreatLevel.DANGEROUS
        elif total_weight >= self._suspect_threshold:
            level = ThreatLevel.SUSPECT
        else:
            level = ThreatLevel.SAFE
        primary_cat = ThreatCategory.NONE
        if category_hits:
            primary_cat = max(category_hits, key=lambda c: category_hits[c])
        sanitized = self._sanitize(text, matched) if matched else text
        elapsed_ms = (time.time() - start_ts) * 1000
        if level != ThreatLevel.SAFE:
            logger.warning(
                f"输入安全扫描: level={level.value}, "
                f"score={total_weight:.1f}, matches={len(matched)}"
            )
        return ScanResult(
            level=level,
            category=primary_cat,
            matched_rules=matched,
            risk_score=total_weight,
            sanitized=sanitized,
            original_length=original_len,
            scan_time_ms=elapsed_ms
        )

    def _sanitize(self, text: str, matched_ids: List[str]) -> str:
        result = text
        for rule_id in matched_ids:
            if rule_id.startswith("blocked:"):
                phrase = rule_id[8:]
                result = re.sub(re.escape(phrase), "[已过滤]", result, flags=re.IGNORECASE)
            elif rule_id in self._compiled:
                pattern = self._compiled[rule_id]
                result = pattern.sub("[已过滤]", result)
        return result

    def is_safe(self, text: str) -> bool:
        result = self.scan(text)
        return result.level == ThreatLevel.SAFE

    def quick_check(self, text: str) -> Tuple[bool, ThreatLevel]:
        result = self.scan(text)
        return result.level in (ThreatLevel.SAFE, ThreatLevel.SUSPECT), result.level

    def set_thresholds(
        self,
        suspect: Optional[float] = None,
        danger: Optional[float] = None,
        block: Optional[float] = None
    ) -> None:
        if suspect is not None:
            self._suspect_threshold = suspect
        if danger is not None:
            self._danger_threshold = danger
        if block is not None:
            self._block_threshold = block

    def get_rule_count(self) -> int:
        return len(self._rules)


def get_input_guardian() -> InputGuardian:
    return InputGuardian.shared()


def scan_input(text: str) -> ScanResult:
    return InputGuardian.shared().scan(text)


def is_input_safe(text: str) -> bool:
    return InputGuardian.shared().is_safe(text)


def sanitize_input(text: str) -> str:
    result = InputGuardian.shared().scan(text)
    return result.sanitized

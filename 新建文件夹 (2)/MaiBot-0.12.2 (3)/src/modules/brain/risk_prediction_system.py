import time
import re
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from src.common.logger import get_logger

logger = get_logger("security_review")


class SecurityLevel(Enum):
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


@dataclass
class SecurityPattern:
    pattern_id: str = ""
    pattern_type: str = ""
    pattern_data: Any = None
    risk_level: SecurityLevel = SecurityLevel.LOW_RISK
    confidence: float = 0.0
    detection_count: int = 0
    false_positive_count: int = 0
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityAssessment:
    overall_security_level: SecurityLevel = SecurityLevel.SAFE
    risk_score: float = 0.0
    detected_patterns: List[SecurityPattern] = field(default_factory=list)
    security_summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    review_confidence: float = 0.0
    processing_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ContentAnalysis:
    content_hash: str = ""
    security_flags: List[str] = field(default_factory=list)
    risk_indicators: Dict[str, float] = field(default_factory=dict)
    contextual_factors: Dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: float = field(default_factory=time.time)


class SecurityReviewSystem:
    def __init__(self):
        self._patterns: Dict[str, SecurityPattern] = {}
        self._pattern_counter = 0
        self._content_cache: Dict[str, ContentAnalysis] = {}
        self._max_cache_size = 1000
        self._assessment_history: List[SecurityAssessment] = []
        self._max_history = 500
        self._learning_enabled = True
        self._false_positive_threshold = 0.3
        self._pattern_decay_rate = 0.95
        self._min_pattern_confidence = 0.6
        self._initialize_base_patterns()

    def _initialize_base_patterns(self):
        base_patterns = [
            {"type": "adaptive_keyword", "initial_risk": SecurityLevel.MEDIUM_RISK},
            {"type": "behavioral_pattern", "initial_risk": SecurityLevel.LOW_RISK},
            {"type": "contextual_analysis", "initial_risk": SecurityLevel.LOW_RISK},
        ]
        for cfg in base_patterns:
            pid = f"base_{cfg['type']}_{int(time.time())}"
            self._patterns[pid] = SecurityPattern(
                pattern_id=pid, pattern_type=cfg["type"],
                risk_level=cfg["initial_risk"], confidence=0.5,
                metadata={"learned": False, "source": "system_init"}
            )

    def review_content(self, content: str, context: Optional[Any] = None) -> SecurityAssessment:
        start_time = time.time()
        try:
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            if content_hash in self._content_cache:
                cached = self._content_cache[content_hash]
                assessment = self._create_assessment_from_cache(cached)
                assessment.processing_time = time.time() - start_time
                return assessment
            analysis = self._perform_dynamic_analysis(content, context)
            assessment = self._create_assessment(content, analysis, context)
            assessment.processing_time = time.time() - start_time
            self._cache_analysis(content_hash, analysis)
            self._record_assessment(assessment)
            if self._learning_enabled:
                self._learn_from_assessment(content, assessment)
            return assessment
        except Exception as e:
            logger.error(f"安全审查失败: {e}")
            return SecurityAssessment(
                overall_security_level=SecurityLevel.MEDIUM_RISK,
                risk_score=0.5, security_summary=f"审查出错: {e}",
                processing_time=time.time() - start_time
            )

    def predict_risk(self, context: Any) -> float:
        try:
            if hasattr(context, 'message_content'):
                content = context.message_content
            elif isinstance(context, dict):
                content = context.get('message_content', '')
            else:
                content = str(context)
            return self.review_content(content, context).risk_score
        except Exception:
            return 0.5

    def _perform_dynamic_analysis(self, content: str, context: Optional[Any] = None) -> Dict[str, Any]:
        result = {"risk_indicators": {}, "detected_patterns": [], "contextual_factors": {}, "confidence_score": 0.0}
        result["detected_patterns"] = self._analyze_patterns(content, context)
        result["risk_indicators"].update(self._analyze_semantic_risks(content))
        result["contextual_factors"] = self._analyze_contextual_risks(content, context)
        result["risk_indicators"].update(self._analyze_behavioral_risks(content))
        result["confidence_score"] = self._calc_analysis_confidence(
            result["detected_patterns"], result["risk_indicators"], result["contextual_factors"]
        )
        return result

    def _analyze_patterns(self, content: str, context: Optional[Any] = None) -> List[Dict[str, Any]]:
        matches = []
        for pid, pattern in self._patterns.items():
            if pattern.confidence < self._min_pattern_confidence:
                continue
            try:
                match = self._check_pattern_match(pattern, content, context)
                if match["matched"]:
                    matches.append({
                        "pattern_id": pid, "pattern_type": pattern.pattern_type,
                        "risk_level": pattern.risk_level.value, "confidence": match["confidence"],
                    })
            except Exception:
                pass
        return matches

    def _check_pattern_match(self, pattern: SecurityPattern, content: str, context: Optional[Any] = None) -> Dict:
        result = {"matched": False, "confidence": 0.0, "details": {}}
        try:
            if pattern.pattern_type == "adaptive_keyword":
                result = self._check_adaptive_keyword_pattern(pattern, content)
            elif pattern.pattern_type == "behavioral_pattern":
                result = self._check_behavioral_pattern(pattern, content, context)
            elif pattern.pattern_type == "contextual_analysis":
                result = self._check_contextual_pattern(pattern, content, context)
            elif pattern.pattern_type == "regex":
                result = self._check_regex_pattern(pattern, content)
            if result["matched"]:
                pattern.detection_count += 1
                pattern.confidence = min(0.95, pattern.confidence + 0.01)
            else:
                pattern.confidence *= self._pattern_decay_rate
            pattern.last_updated = time.time()
        except Exception as e:
            logger.warning(f"模式检查失败: {e}")
        return result

    def _check_adaptive_keyword_pattern(self, pattern: SecurityPattern, content: str) -> Dict:
        keywords = self._extract_risk_keywords(content)
        matched = [kw for kw in keywords if kw in content.lower()]
        return {
            "matched": len(matched) > 0,
            "confidence": min(1.0, len(matched) * 0.2),
            "details": {"matched_keywords": matched}
        }

    def _check_behavioral_pattern(self, pattern: SecurityPattern, content: str,
                                  context: Optional[Any] = None) -> Dict:
        indicators = self._analyze_behavior_indicators(content)
        risk = sum(i["risk"] for i in indicators)
        return {
            "matched": risk > 0.3,
            "confidence": min(1.0, len(indicators) * 0.15),
            "details": {"behavior_indicators": indicators, "total_risk": risk}
        }

    def _check_contextual_pattern(self, pattern: SecurityPattern, content: str,
                                  context: Optional[Any] = None) -> Dict:
        contextual_risks = self._analyze_contextual_risks(content, context)
        risk = sum(contextual_risks.values())
        return {
            "matched": risk > 0.2,
            "confidence": min(1.0, len(contextual_risks) * 0.1),
            "details": {"contextual_risks": contextual_risks}
        }

    def _check_regex_pattern(self, pattern: SecurityPattern, content: str) -> Dict:
        if not pattern.pattern_data:
            return {"matched": False, "confidence": 0.0, "details": {}}
        try:
            matches = re.findall(pattern.pattern_data, content, re.IGNORECASE)
            return {
                "matched": len(matches) > 0,
                "confidence": min(1.0, len(matches) * 0.3),
                "details": {"regex_matches": matches}
            }
        except Exception as e:
            return {"matched": False, "confidence": 0.0, "details": {"error": str(e)}}

    def _extract_risk_keywords(self, content: str) -> List[str]:
        keywords = []
        patterns = [
            r'\b(?:伤害|自杀|杀人|暴力|恐怖)\b',
            r'\b(?:色情|性|裸体|黄|涩)\b',
            r'\b(?:身份证|手机号|银行卡|密码)\b',
        ]
        for p in patterns:
            keywords.extend(re.findall(p, content, re.IGNORECASE))
        return list(set(keywords))

    def _analyze_semantic_risks(self, content: str) -> Dict[str, float]:
        risks = {}
        if len(content) > 1000:
            risks["too_long"] = 0.2
        special = sum(1 for c in content if not c.isalnum() and not c.isspace())
        ratio = special / max(1, len(content))
        if ratio > 0.5:
            risks["high_special_chars"] = min(0.8, ratio)
        if self._has_repetition(content):
            risks["repetition"] = 0.3
        return risks

    def _analyze_contextual_risks(self, content: str, context: Optional[Any] = None) -> Dict[str, float]:
        risks = {}
        if not context:
            return risks
        if hasattr(context, 'user_profile') and isinstance(context.user_profile, dict):
            v = context.user_profile.get('violation_count', 0)
            if v > 0:
                risks["violation_history"] = min(0.8, v * 0.2)
        return risks

    def _analyze_behavioral_risks(self, content: str) -> Dict[str, float]:
        risks = {}
        if '?' in content or '？' in content:
            qr = self._analyze_question_risk(content)
            if qr > 0:
                risks["question_risk"] = qr
        command_patterns = [r'帮我', r'请', r'能不能', r'可以吗', r'告诉我']
        cmd_count = sum(1 for p in command_patterns if p in content)
        if cmd_count > 0:
            risks["command_density"] = min(0.6, cmd_count * 0.2)
        return risks

    def _analyze_question_risk(self, content: str) -> float:
        high_risk = [r'怎么.*死', r'如何.*自杀', r'怎么.*杀人', r'如何.*伤害']
        for p in high_risk:
            if re.search(p, content, re.IGNORECASE):
                return 0.8
        med_risk = [r'如何.*破解', r'怎么.*作弊']
        for p in med_risk:
            if re.search(p, content, re.IGNORECASE):
                return 0.4
        return 0.0

    def _analyze_behavior_indicators(self, content: str) -> List[Dict[str, Any]]:
        indicators = []
        for p in [r'你必须', r'你应该', r'赶紧', r'立刻']:
            if re.search(p, content):
                indicators.append({"type": "manipulative", "risk": 0.4})
        for p in [r'我要求', r'我命令', r'你给我', r'必须']:
            if re.search(p, content):
                indicators.append({"type": "demand", "risk": 0.3})
        return indicators

    def _has_repetition(self, content: str) -> bool:
        if not content or len(content) < 2:
            return False
        if len(set(content)) / len(content) < 0.3:
            return True
        words = content.split()
        if len(words) >= 3:
            counts = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
            if max(counts.values()) / len(words) > 0.5:
                return True
        return False

    def _has_topic_consistency_issues(self, history: List) -> bool:
        """检查话题一致性问题"""
        if len(history) < 3:
            return False
        topics = []
        for msg in history[-5:]:
            if isinstance(msg, dict) and 'content' in msg:
                content = msg['content'][:50].lower()
                words = [w for w in content.split() if len(w) > 1][:3]
                topics.append(' '.join(words))
        unique_topics = set(topics)
        if len(unique_topics) > len(topics) * 0.8:
            return True
        return False

    def _is_reply_message(self, message) -> bool:
        """判断是否是回复消息"""
        if isinstance(message, dict):
            content = message.get('content', '')
            return '@' in content or '回复' in content
        return False

    def _calc_analysis_confidence(self, patterns: List, semantic: Dict, contextual: Dict) -> float:
        conf = 0.5
        conf += min(0.3, len(patterns) * 0.1)
        total = len(semantic) + len(contextual)
        avg = (sum(semantic.values()) + sum(contextual.values())) / max(1, total)
        conf += min(0.2, total * 0.05)
        conf += min(0.1, avg * 0.2)
        return min(0.95, conf)

    def _create_assessment(self, content: str, analysis: Dict, context: Optional[Any] = None) -> SecurityAssessment:
        risk_score = self._calc_overall_risk(analysis)
        level = self._determine_level(risk_score)
        detected = [self._patterns[m["pattern_id"]] for m in analysis["detected_patterns"] if m["pattern_id"] in self._patterns]
        summary = self._generate_security_summary(risk_score, level, analysis)
        recommendations = self._generate_security_recommendations(risk_score, level, analysis)
        return SecurityAssessment(
            overall_security_level=level, risk_score=risk_score,
            detected_patterns=detected, security_summary=summary,
            recommendations=recommendations,
            review_confidence=analysis["confidence_score"]
        )

    def _create_assessment_from_cache(self, analysis: ContentAnalysis) -> SecurityAssessment:
        risk_score = sum(analysis.risk_indicators.values())
        return SecurityAssessment(
            overall_security_level=self._determine_level(risk_score),
            risk_score=risk_score, security_summary="缓存评估",
            review_confidence=0.8
        )

    def _calc_overall_risk(self, analysis: Dict) -> float:
        base = sum(analysis["risk_indicators"].values())
        pattern_bonus = len(analysis["detected_patterns"]) * 0.1
        ctx_mult = 1.0
        if analysis["contextual_factors"]:
            ctx_mult = 1.0 + min(0.5, sum(analysis["contextual_factors"].values()))
        return min(1.0, (base + pattern_bonus) * ctx_mult)

    def _determine_level(self, risk_score: float) -> SecurityLevel:
        if risk_score >= 0.8:
            return SecurityLevel.CRITICAL
        elif risk_score >= 0.6:
            return SecurityLevel.HIGH_RISK
        elif risk_score >= 0.4:
            return SecurityLevel.MEDIUM_RISK
        elif risk_score >= 0.2:
            return SecurityLevel.LOW_RISK
        return SecurityLevel.SAFE

    def _cache_analysis(self, content_hash: str, analysis: Dict):
        self._content_cache[content_hash] = ContentAnalysis(
            content_hash=content_hash,
            risk_indicators=analysis["risk_indicators"],
            contextual_factors=analysis["contextual_factors"]
        )
        if len(self._content_cache) > self._max_cache_size:
            oldest = min(self._content_cache.keys(), key=lambda k: self._content_cache[k].analysis_timestamp)
            del self._content_cache[oldest]

    def _record_assessment(self, assessment: SecurityAssessment):
        self._assessment_history.append(assessment)
        if len(self._assessment_history) > self._max_history:
            self._assessment_history = self._assessment_history[-self._max_history:]

    def _learn_from_assessment(self, content: str, assessment: SecurityAssessment):
        try:
            if assessment.risk_score > 0.7 and len(assessment.detected_patterns) == 0:
                self._learn_new_pattern(content, assessment)
            for pattern in assessment.detected_patterns:
                if assessment.risk_score > 0.6:
                    pattern.confidence = min(0.95, pattern.confidence + 0.02)
                elif assessment.risk_score < 0.3:
                    pattern.false_positive_count += 1
                    if pattern.false_positive_count / max(1, pattern.detection_count) > self._false_positive_threshold:
                        pattern.confidence *= 0.9
        except Exception as e:
            logger.warning(f"学习过程出错: {e}")

    def _learn_new_pattern(self, content: str, assessment: SecurityAssessment):
        """从高风险内容中学习新安全模式"""
        try:
            pid = f"learned_{int(time.time())}_{self._calculate_content_hash(content)[:8]}"
            keywords = self._extract_risk_keywords(content)
            self._patterns[pid] = SecurityPattern(
                pattern_id=pid, pattern_type="adaptive_keyword",
                risk_level=assessment.overall_security_level, confidence=0.6,
                metadata={
                    "keywords": keywords,
                    "risk_score": assessment.risk_score,
                    "source": "dynamic_learning",
                    "learned": True,
                    "learning_timestamp": time.time(),
                }
            )
            self._pattern_counter += 1
            logger.info(f"学习到新安全模式: {pid}")
        except Exception as e:
            logger.warning(f"学习新模式失败: {e}")

    def _calculate_content_hash(self, content: str) -> str:
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _generate_security_summary(self, risk_score: float, level: SecurityLevel,
                                   analysis: Dict) -> str:
        """生成安全摘要"""
        if level == SecurityLevel.SAFE:
            return "内容安全，未检测到明显风险"
        parts = []
        risk_indicators = analysis.get("risk_indicators", {})
        if risk_indicators:
            top_risks = sorted(risk_indicators.items(), key=lambda x: x[1], reverse=True)[:3]
            names = [n for n, s in top_risks]
            parts.append(f"检测到风险指标: {', '.join(names)}")
        pattern_matches = analysis.get("detected_patterns", [])
        if pattern_matches:
            types = list(set(m["pattern_type"] for m in pattern_matches))
            parts.append(f"触发安全模式: {', '.join(types)}")
        if parts:
            return f"安全等级: {level.value}。{'；'.join(parts)}"
        return f"安全等级: {level.value}"

    def _generate_security_recommendations(self, risk_score: float, level: SecurityLevel,
                                           analysis: Dict) -> List[str]:
        """生成安全建议"""
        recommendations = []
        if level == SecurityLevel.CRITICAL:
            recommendations.extend([
                "严重安全风险，建议立即阻止",
                "记录违规行为，需管理员干预",
                "考虑临时限制用户权限",
            ])
        elif level == SecurityLevel.HIGH_RISK:
            recommendations.extend([
                "高风险内容，建议谨慎处理",
                "增加人工审核",
            ])
        elif level == SecurityLevel.MEDIUM_RISK:
            recommendations.extend([
                "中等风险，保持警惕",
                "继续监控用户行为",
            ])
        elif level == SecurityLevel.LOW_RISK:
            recommendations.append("低风险内容，保持关注")
        risk_indicators = analysis.get("risk_indicators", {})
        if risk_indicators.get("question_risk", 0) > 0.5:
            recommendations.append("问题涉及敏感内容，建议拒绝或安全引导")
        if risk_indicators.get("command_density", 0) > 0.4:
            recommendations.append("用户发出较多指令，注意操纵性行为")
        return recommendations

    def get_security_stats(self) -> Dict[str, Any]:
        if not self._assessment_history:
            return {"total_assessments": 0}
        recent = self._assessment_history[-100:]
        level_counts: Dict[str, int] = {}
        for a in recent:
            lv = a.overall_security_level.value
            level_counts[lv] = level_counts.get(lv, 0) + 1
        return {
            "total_assessments": len(self._assessment_history),
            "recent_level_distribution": level_counts,
            "average_risk_score": sum(a.risk_score for a in recent) / len(recent),
            "cache_size": len(self._content_cache),
            "total_patterns": len(self._patterns),
        }


_security_review_system: Optional[SecurityReviewSystem] = None


def get_security_review_system() -> SecurityReviewSystem:
    global _security_review_system
    if _security_review_system is None:
        _security_review_system = SecurityReviewSystem()
    return _security_review_system


RiskPredictionSystem = SecurityReviewSystem


def get_risk_predictor() -> SecurityReviewSystem:
    return get_security_review_system()

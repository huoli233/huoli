import time
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
from src.common.logger import get_logger

logger = get_logger("pattern_miner")


@dataclass
class ConversationPattern:
    pattern_id: str = ""
    pattern_type: str = ""
    description: str = ""
    frequency: int = 0
    confidence: float = 0.0
    examples: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternAnalysisResult:
    patterns_found: List[ConversationPattern] = field(default_factory=list)
    overall_pattern_score: float = 0.0
    dominant_pattern: Optional[ConversationPattern] = None
    recommendations: List[str] = field(default_factory=list)
    analysis_summary: str = ""


class PatternMiningSystem:
    def __init__(self):
        from src.config.config import global_config
        brain_cfg = global_config.brain
        self._patterns: Dict[str, ConversationPattern] = {}
        self._pattern_counter = 0
        self._min_pattern_frequency = 3
        self._max_patterns = brain_cfg.max_pattern_count
        self._pattern_decay_rate = 0.95
        self._pattern_templates = self._initialize_pattern_templates()
        logger.info("模式挖掘系统初始化完成")

    def _initialize_pattern_templates(self) -> Dict[str, Dict[str, Any]]:
        return {
            "greeting_pattern": {
                "type": "behavior",
                "keywords": ["你好", "hello", "hi", "早上好", "晚上好", "在吗"],
                "description": "问候模式"
            },
            "question_pattern": {
                "type": "content",
                "indicators": ["?", "？", "什么", "怎么", "为什么", "如何"],
                "description": "提问模式"
            },
            "complaint_pattern": {
                "type": "emotion",
                "keywords": ["不行", "不好", "讨厌", "烦", "差", "烂"],
                "description": "抱怨模式"
            },
            "praise_pattern": {
                "type": "emotion",
                "keywords": ["好", "棒", "厉害", "不错", "很好", "优秀"],
                "description": "赞美模式"
            },
            "command_pattern": {
                "type": "behavior",
                "indicators": ["帮我", "请", "能不能", "可以吗", "告诉我"],
                "description": "指令模式"
            },
            "gaming_pattern": {
                "type": "topic",
                "keywords": ["游戏", "lol", "王者", "原神", "steam", "打游戏"],
                "description": "游戏话题模式"
            },
            "work_pattern": {
                "type": "topic",
                "keywords": ["工作", "上班", "项目", "任务", "会议", "报告"],
                "description": "工作话题模式"
            }
        }

    def analyze_patterns(self, context: Any) -> float:
        try:
            messages = self._extract_messages(context)
            user_id = self._extract_user_id(context)
            if not messages:
                return 0.5
            pattern_scores = self._analyze_all_patterns(messages, user_id)
            overall_score = self._calculate_overall_pattern_score(pattern_scores)
            self._record_new_patterns(messages, user_id)
            self._update_existing_patterns()
            return overall_score
        except Exception as e:
            logger.error(f"模式分析出错: {e}")
            return 0.5

    def _extract_messages(self, context) -> List[str]:
        messages = []
        if hasattr(context, 'conversation_history'):
            for msg in context.conversation_history:
                if isinstance(msg, dict) and 'content' in msg:
                    messages.append(msg['content'])
                elif isinstance(msg, str):
                    messages.append(msg)
        if hasattr(context, 'message_content'):
            messages.append(context.message_content)
        return messages[-20:]

    def _extract_user_id(self, context) -> str:
        if hasattr(context, 'user_id'):
            return context.user_id
        return "unknown"

    def _analyze_all_patterns(self, messages: List[str], user_id: str) -> Dict[str, float]:
        pattern_scores = {}
        for pattern_name, template in self._pattern_templates.items():
            score = self._analyze_single_pattern(messages, template)
            pattern_scores[pattern_name] = score
        user_patterns = self._get_user_patterns(user_id)
        for pattern_name, pattern in user_patterns.items():
            score = self._analyze_learned_pattern(messages, pattern)
            pattern_scores[f"user_{pattern_name}"] = score
        return pattern_scores

    def _analyze_single_pattern(self, messages: List[str], template: Dict[str, Any]) -> float:
        total_score = 0.0
        message_count = len(messages)
        if not messages:
            return 0.0
        for message in messages:
            message_score = 0.0
            message_lower = message.lower()
            if 'keywords' in template:
                keywords = template['keywords']
                matched = sum(1 for kw in keywords if kw in message_lower)
                if matched > 0:
                    message_score += matched / len(keywords)
            if 'indicators' in template:
                indicators = template['indicators']
                matched = sum(1 for ind in indicators if ind in message)
                if matched > 0:
                    message_score += matched / len(indicators)
            total_score += min(1.0, message_score)
        avg_score = total_score / message_count
        frequency_weight = min(1.0, total_score / 3.0)
        return avg_score * frequency_weight

    def _analyze_learned_pattern(self, messages: List[str], pattern: ConversationPattern) -> float:
        if pattern.pattern_type == "keyword_sequence":
            keywords = pattern.metadata.get('keywords', [])
            return self._analyze_keyword_sequence(messages, keywords)
        elif pattern.pattern_type == "behavior_pattern":
            behaviors = pattern.metadata.get('behaviors', [])
            return self._analyze_behavior_pattern(messages, behaviors)
        return self._analyze_single_pattern(messages, {
            'keywords': pattern.metadata.get('keywords', []),
            'indicators': pattern.metadata.get('indicators', [])
        })

    def _analyze_keyword_sequence(self, messages: List[str], keywords: List[str]) -> float:
        if not keywords or not messages:
            return 0.0
        combined_text = " ".join(messages).lower()
        sequence_matches = 0
        for i in range(len(keywords) - 1):
            if keywords[i] in combined_text and keywords[i + 1] in combined_text:
                sequence_matches += 1
        return min(1.0, sequence_matches / max(1, len(keywords) - 1))

    def _analyze_behavior_pattern(self, messages: List[str], behaviors: List[str]) -> float:
        if not behaviors or not messages:
            return 0.0
        behavior_matches = 0
        for message in messages:
            for behavior in behaviors:
                if behavior in message:
                    behavior_matches += 1
                    break
        return min(1.0, behavior_matches / len(messages))

    def _calculate_overall_pattern_score(self, pattern_scores: Dict[str, float]) -> float:
        if not pattern_scores:
            return 0.5
        max_score = max(pattern_scores.values())
        avg_score = sum(pattern_scores.values()) / len(pattern_scores)
        return min(1.0, (max_score * 0.7) + (avg_score * 0.3))

    def _record_new_patterns(self, messages: List[str], user_id: str):
        if len(messages) < self._min_pattern_frequency:
            return
        all_text = " ".join(messages).lower()
        words = re.findall(r'\b\w+\b', all_text)
        stop_words = {'的', '了', '和', '是', '在', '我', '你', '他', '她', '它', '我们', '你们', '他们', '这', '那', '一个', '一些', '很多', '一下'}
        filtered_words = [w for w in words if len(w) > 1 and w not in stop_words]
        word_freq = Counter(filtered_words)
        frequent_words = [word for word, freq in word_freq.most_common(10) if freq >= 3]
        if len(frequent_words) >= 2:
            pattern_id = f"kw_seq_{user_id}_{int(time.time())}"
            pattern = ConversationPattern(
                pattern_id=pattern_id,
                pattern_type="keyword_sequence",
                description=f"用户{user_id[:8]}的关键词序列模式",
                frequency=len(frequent_words),
                confidence=0.6,
                examples=messages[-3:],
                metadata={'keywords': frequent_words, 'user_id': user_id, 'discovered_at': time.time()}
            )
            self._patterns[pattern_id] = pattern
            self._pattern_counter += 1

    def _update_existing_patterns(self):
        current_time = time.time()
        patterns_to_remove = []
        for pattern_id, pattern in self._patterns.items():
            time_diff = current_time - pattern.last_seen
            if time_diff > 86400:
                pattern.confidence *= self._pattern_decay_rate
                pattern.frequency = max(0, pattern.frequency - 1)
            if pattern.frequency < 1 or pattern.confidence < 0.1:
                patterns_to_remove.append(pattern_id)
        for pid in patterns_to_remove:
            del self._patterns[pid]
        if len(self._patterns) > self._max_patterns:
            sorted_patterns = sorted(
                self._patterns.items(),
                key=lambda x: x[1].confidence * x[1].frequency,
                reverse=True
            )
            self._patterns = dict(sorted_patterns[:self._max_patterns])

    def _get_user_patterns(self, user_id: str) -> Dict[str, ConversationPattern]:
        return {pid: p for pid, p in self._patterns.items() if p.metadata.get('user_id') == user_id}

    def get_pattern_analysis(self, context: Any) -> PatternAnalysisResult:
        try:
            messages = self._extract_messages(context)
            user_id = self._extract_user_id(context)
            if not messages:
                return PatternAnalysisResult(overall_pattern_score=0.5, analysis_summary="无足够消息进行模式分析")
            pattern_scores = self._analyze_all_patterns(messages, user_id)
            overall_score = self._calculate_overall_pattern_score(pattern_scores)
            dominant_pattern = None
            if pattern_scores:
                best_name = max(pattern_scores, key=lambda x: pattern_scores.get(x, 0))
                dominant_pattern = self._patterns.get(best_name)
            recommendations = self._generate_pattern_recommendations(pattern_scores, context)
            return PatternAnalysisResult(
                patterns_found=list(self._patterns.values())[:5],
                overall_pattern_score=overall_score,
                dominant_pattern=dominant_pattern,
                recommendations=recommendations,
                analysis_summary=self._generate_analysis_summary(pattern_scores)
            )
        except Exception as e:
            logger.error(f"获取模式分析失败: {e}")
            return PatternAnalysisResult(overall_pattern_score=0.5, analysis_summary=f"模式分析失败: {e}")

    def _generate_pattern_recommendations(self, pattern_scores: Dict[str, float], context) -> List[str]:
        recommendations = []
        if pattern_scores:
            best = max(pattern_scores, key=lambda x: pattern_scores.get(x, 0))
            score = pattern_scores[best]
            if score > 0.8:
                recommendations.append(f"检测到强烈{best}模式，建议调整回复策略")
            elif score > 0.6:
                recommendations.append(f"检测到{best}模式，可考虑优化回复")
        if hasattr(context, 'conversation_history') and len(context.conversation_history) < 3:
            recommendations.append("对话历史较短，建议收集更多上下文")
        return recommendations

    def _generate_analysis_summary(self, pattern_scores: Dict[str, float]) -> str:
        if not pattern_scores:
            return "未检测到明显模式"
        top_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        summary_parts = []
        for name, score in top_patterns:
            if score > 0.7:
                summary_parts.append(f"{name}（强匹配）")
            elif score > 0.5:
                summary_parts.append(f"{name}（中等匹配）")
        return f"检测到模式: {', '.join(summary_parts)}" if summary_parts else "模式匹配度较低"

    def get_pattern_stats(self) -> Dict[str, Any]:
        total = len(self._patterns)
        pattern_types = Counter(p.pattern_type for p in self._patterns.values())
        avg_conf = sum(p.confidence for p in self._patterns.values()) / max(1, total)
        avg_freq = sum(p.frequency for p in self._patterns.values()) / max(1, total)
        return {
            "total_patterns": total, "pattern_types": dict(pattern_types),
            "average_confidence": avg_conf, "average_frequency": avg_freq,
            "templates_count": len(self._pattern_templates)
        }


_pattern_miner: Optional[PatternMiningSystem] = None


def get_pattern_miner() -> PatternMiningSystem:
    global _pattern_miner
    if _pattern_miner is None:
        _pattern_miner = PatternMiningSystem()
    return _pattern_miner

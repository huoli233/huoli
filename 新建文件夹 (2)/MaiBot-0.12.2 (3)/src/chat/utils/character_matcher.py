import re
import time
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from src.common.logger import get_logger

logger = get_logger("character_matcher")


@dataclass
class MatchConfig:
    keywords: List[str] = field(default_factory=list)
    visual_features: List[str] = field(default_factory=list)
    name_patterns: List[str] = field(default_factory=list)
    exclusion_words: List[str] = field(default_factory=list)
    min_score: float = 0.4


@dataclass
class MatchResult:
    is_match: bool
    score: float
    matched_items: List[str] = field(default_factory=list)
    match_type: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class KeywordMatcher:
    def __init__(self):
        self._keywords: Set[str] = set()
        self._keyword_weights: Dict[str, float] = {}
        self._exclusions: Set[str] = set()

    def set_keywords(self, keywords: List[str], weights: Optional[Dict[str, float]] = None):
        self._keywords = set(k.lower() for k in keywords if k)
        if weights:
            self._keyword_weights = {k.lower(): v for k, v in weights.items()}

    def set_exclusions(self, exclusions: List[str]):
        self._exclusions = set(e.lower() for e in exclusions if e)

    def match(self, text: str) -> Tuple[float, List[str]]:
        text_lower = text.lower()
        for excl in self._exclusions:
            if excl in text_lower:
                return 0.0, []
        matched = []
        total_weight = 0.0
        for kw in self._keywords:
            if kw in text_lower:
                matched.append(kw)
                weight = self._keyword_weights.get(kw, 1.0)
                total_weight += weight
        if not self._keywords:
            return 0.0, []
        max_possible = sum(self._keyword_weights.get(k, 1.0) for k in self._keywords)
        score = total_weight / max_possible if max_possible > 0 else 0.0
        return score, matched


class VisualFeatureMatcher:
    def __init__(self):
        self._features: List[str] = []
        self._feature_weights: Dict[str, float] = {}
        self._required_features: Set[str] = set()

    def set_features(self, features: List[str], weights: Optional[Dict[str, float]] = None):
        self._features = [f.lower() for f in features if f]
        if weights:
            self._feature_weights = {k.lower(): v for k, v in weights.items()}

    def set_required(self, required: List[str]):
        self._required_features = set(r.lower() for r in required if r)

    def match(self, description: str) -> Tuple[float, List[str]]:
        desc_lower = description.lower()
        matched = []
        for req in self._required_features:
            if req not in desc_lower:
                return 0.0, []
        for feature in self._features:
            if feature in desc_lower:
                matched.append(feature)
        if not self._features:
            return 0.0, []
        total_weight = sum(self._feature_weights.get(f, 1.0) for f in matched)
        max_weight = sum(self._feature_weights.get(f, 1.0) for f in self._features)
        score = total_weight / max_weight if max_weight > 0 else 0.0
        return score, matched


class NamePatternMatcher:
    def __init__(self):
        self._patterns: List[str] = []
        self._compiled: List[re.Pattern] = []
        self._aliases: Set[str] = set()

    def set_patterns(self, patterns: List[str]):
        self._patterns = patterns
        self._compiled = []
        for pattern in patterns:
            try:
                self._compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                self._compiled.append(re.compile(re.escape(pattern), re.IGNORECASE))

    def set_aliases(self, aliases: List[str]):
        self._aliases = set(a.lower() for a in aliases if a)

    def match(self, text: str) -> Tuple[bool, str]:
        text_lower = text.lower()
        for alias in self._aliases:
            if alias in text_lower:
                return True, alias
        for pattern in self._compiled:
            m = pattern.search(text)
            if m:
                return True, m.group()
        return False, ""


class UrlContentMatcher:
    def __init__(self):
        self._domain_keywords: Dict[str, List[str]] = {}
        self._path_patterns: List[str] = []

    def set_domain_keywords(self, domain_keywords: Dict[str, List[str]]):
        self._domain_keywords = {
            k.lower(): [v.lower() for v in vals]
            for k, vals in domain_keywords.items()
        }

    def set_path_patterns(self, patterns: List[str]):
        self._path_patterns = [p.lower() for p in patterns]

    def match_url(self, url: str) -> Tuple[bool, float, str]:
        url_lower = url.lower()
        for domain, keywords in self._domain_keywords.items():
            if domain in url_lower:
                for kw in keywords:
                    if kw in url_lower:
                        return True, 0.8, f"{domain}:{kw}"
        for pattern in self._path_patterns:
            if pattern in url_lower:
                return True, 0.6, pattern
        return False, 0.0, ""


class CharacterMatcher:
    def __init__(self):
        self._keyword_matcher = KeywordMatcher()
        self._visual_matcher = VisualFeatureMatcher()
        self._name_matcher = NamePatternMatcher()
        self._url_matcher = UrlContentMatcher()
        self._min_score = 0.4
        self._match_history: List[Dict] = []
        self._max_history = 100

    def configure(self, config: MatchConfig):
        self._keyword_matcher.set_keywords(config.keywords)
        self._keyword_matcher.set_exclusions(config.exclusion_words)
        self._visual_matcher.set_features(config.visual_features)
        self._name_matcher.set_patterns(config.name_patterns)
        self._min_score = config.min_score

    def set_character_info(
        self,
        keywords: Optional[List[str]] = None,
        visual_features: Optional[List[str]] = None,
        name_patterns: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
    ):
        if keywords:
            self._keyword_matcher.set_keywords(keywords)
        if visual_features:
            self._visual_matcher.set_features(visual_features)
        if name_patterns:
            self._name_matcher.set_patterns(name_patterns)
        if aliases:
            self._name_matcher.set_aliases(aliases)

    def match_text(self, text: str) -> MatchResult:
        kw_score, kw_matched = self._keyword_matcher.match(text)
        name_found, name_matched = self._name_matcher.match(text)
        if name_found:
            result = MatchResult(
                is_match=True, score=1.0,
                matched_items=[name_matched], match_type="name",
                details={"name": name_matched},
            )
        elif kw_score >= self._min_score:
            result = MatchResult(
                is_match=True, score=kw_score,
                matched_items=kw_matched, match_type="keyword",
                details={"keywords": kw_matched},
            )
        else:
            result = MatchResult(
                is_match=False, score=kw_score,
                matched_items=kw_matched, match_type="none",
            )
        self._record_match("text", text[:50], result)
        return result

    def match_visual(self, description: str) -> MatchResult:
        vis_score, vis_matched = self._visual_matcher.match(description)
        name_found, name_matched = self._name_matcher.match(description)
        if name_found:
            result = MatchResult(
                is_match=True, score=1.0,
                matched_items=[name_matched], match_type="name_in_visual",
                details={"name": name_matched},
            )
        elif vis_score >= self._min_score:
            result = MatchResult(
                is_match=True, score=vis_score,
                matched_items=vis_matched, match_type="visual_feature",
                details={"features": vis_matched},
            )
        else:
            result = MatchResult(
                is_match=False, score=vis_score,
                matched_items=vis_matched, match_type="none",
            )
        self._record_match("visual", description[:50], result)
        return result

    def match_url(self, url: str) -> MatchResult:
        url_match, url_score, url_detail = self._url_matcher.match_url(url)
        if url_match:
            result = MatchResult(
                is_match=True, score=url_score,
                matched_items=[url_detail], match_type="url",
                details={"url_match": url_detail},
            )
        else:
            result = MatchResult(
                is_match=False, score=0.0, match_type="none",
            )
        self._record_match("url", url[:50], result)
        return result

    def match_combined(
        self, text: str = "", visual_desc: str = "", url: str = "",
    ) -> MatchResult:
        results = []
        if text:
            results.append(("text", self.match_text(text)))
        if visual_desc:
            results.append(("visual", self.match_visual(visual_desc)))
        if url:
            results.append(("url", self.match_url(url)))
        if not results:
            return MatchResult(is_match=False, score=0.0, match_type="none")
        best = max(results, key=lambda x: x[1].score)
        best_type, best_result = best
        all_matched = []
        for _, r in results:
            all_matched.extend(r.matched_items)
        return MatchResult(
            is_match=best_result.is_match, score=best_result.score,
            matched_items=list(set(all_matched)), match_type=best_type,
            details=best_result.details,
        )

    def _record_match(self, match_type: str, content: str, result: MatchResult):
        self._match_history.append({
            "type": match_type, "content": content,
            "is_match": result.is_match, "score": result.score,
            "timestamp": time.time(),
        })
        if len(self._match_history) > self._max_history:
            self._match_history = self._match_history[-self._max_history:]

    def get_match_stats(self) -> Dict:
        if not self._match_history:
            return {"total": 0, "matches": 0, "match_rate": 0.0}
        total = len(self._match_history)
        matches = sum(1 for h in self._match_history if h["is_match"])
        return {
            "total": total, "matches": matches,
            "match_rate": matches / total if total > 0 else 0.0,
            "avg_score": sum(h["score"] for h in self._match_history) / total,
        }

    def get_recent_matches(self, limit: int = 10) -> List[Dict]:
        return self._match_history[-limit:]


_character_matcher: Optional[CharacterMatcher] = None


def get_character_matcher() -> CharacterMatcher:
    global _character_matcher
    if _character_matcher is None:
        _character_matcher = CharacterMatcher()
    return _character_matcher


def init_character_matcher(config: Optional[Dict] = None) -> CharacterMatcher:
    global _character_matcher
    _character_matcher = CharacterMatcher()
    try:
        from src.config.config import global_config
        nickname = global_config.bot.nickname or ""
        aliases = list(global_config.bot.alias_names or [])
        if nickname:
            aliases.insert(0, nickname)
        if aliases:
            _character_matcher._name_matcher.set_aliases(aliases)
        visual_str = getattr(global_config.personality, 'character_visual', "")
        if visual_str:
            features = [f.strip() for f in visual_str.split(",") if f.strip()]
            _character_matcher._visual_matcher.set_features(features)
            logger.info(f"[character_matcher] 外观特征已加载: {features}")
    except Exception as e:
        logger.debug(f"[character_matcher] 从global_config加载失败，使用传入config: {e}")
        if config:
            persona_config = config.get("persona", config)
            keywords_str = persona_config.get("character_keywords", "")
            if keywords_str:
                keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
                _character_matcher._keyword_matcher.set_keywords(keywords)
            nickname = persona_config.get("nickname", "")
            alias_str = persona_config.get("alias_names", "")
            aliases = [nickname] if nickname else []
            if alias_str:
                aliases.extend([a.strip() for a in alias_str.split(",") if a.strip()])
            if aliases:
                _character_matcher._name_matcher.set_aliases(aliases)
    logger.info("[character_matcher] 初始化完成")
    return _character_matcher

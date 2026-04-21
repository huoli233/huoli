import time
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.common.logger import get_logger

logger = get_logger("know_graph")


@dataclass
class KnowledgeTriple:
    subject: str
    predicate: str
    object_val: str
    confidence: float
    source_msg_id: str
    timestamp: float = 0.0


class KnowledgeGraphManager:
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self._triples: List[KnowledgeTriple] = []
        self._max_triples = 200

    def add_triple(self, subject: str, predicate: str, object_val: str, confidence: float = 0.9, source_msg_id: str = ""):
        triple = KnowledgeTriple(
            subject=subject, predicate=predicate, object_val=object_val,
            confidence=confidence, source_msg_id=source_msg_id, timestamp=time.time(),
        )
        self._triples.append(triple)
        if len(self._triples) > self._max_triples:
            self._triples = self._triples[-self._max_triples:]
        logger.debug(f"[{self.stream_id[:8]}] 知识入库: {subject} -> {predicate} -> {object_val}")

    def extract_triples_from_text(self, text: str) -> List[KnowledgeTriple]:
        if len(text) < 10:
            return []
        triples = []
        relation_patterns = [
            ("喜欢", "喜欢"), ("讨厌", "讨厌"), ("是", "是"),
            ("在玩", "在玩"), ("在看", "在看"), ("在学", "在学"),
            ("住在", "住在"), ("来自", "来自"), ("会", "擅长"),
        ]
        for keyword, predicate in relation_patterns:
            idx = text.find(keyword)
            if idx > 0 and idx < len(text) - len(keyword):
                subject_start = max(0, idx - 10)
                subject_text = text[subject_start:idx].strip()
                object_start = idx + len(keyword)
                object_end = min(len(text), object_start + 10)
                object_text = text[object_start:object_end].strip()
                for sep in ["，", "。", "！", "？", "、", " ", "\n"]:
                    if sep in object_text:
                        object_text = object_text[:object_text.index(sep)]
                        break
                if subject_text and object_text and len(subject_text) <= 15 and len(object_text) <= 15:
                    triple = KnowledgeTriple(
                        subject=subject_text, predicate=predicate,
                        object_val=object_text, confidence=0.7,
                        source_msg_id="", timestamp=time.time(),
                    )
                    triples.append(triple)
        return triples

    def learn_from_message(self, message: str, msg_id: str = ""):
        triples = self.extract_triples_from_text(message)
        for t in triples:
            t.source_msg_id = msg_id
            self.add_triple(t.subject, t.predicate, t.object_val, t.confidence, msg_id)
        if triples:
            logger.info(f"[{self.stream_id[:8]}] 知识习得: {len(triples)} 条三元组")
        return triples

    async def learn_from_message_with_llm(self, message: str, msg_id: str = ""):
        if len(message) < 10:
            return []
        triples = await self._extract_triples_via_llm(message)
        if not triples:
            triples = self.extract_triples_from_text(message)
        for t in triples:
            t.source_msg_id = msg_id
            self.add_triple(t.subject, t.predicate, t.object_val, t.confidence, msg_id)
        if triples:
            logger.info(f"[{self.stream_id[:8]}] LLM知识习得: {len(triples)} 条三元组")
        return triples

    async def _extract_triples_via_llm(self, text: str) -> List[KnowledgeTriple]:
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            from src.config.prompt_loader import get_prompt, PromptCategory
            prompt = get_prompt(
                PromptCategory.MODULE,
                "social_cognition",
                "knowledge_extract.template",
                text=text
            )
            request = LLMRequest(model_config.lightweight, request_type="knowledge_extract")
            response_text, _ = await request.generate_response_async(prompt)
            if not response_text or response_text.strip() == "无":
                return []
            triples = []
            for line in response_text.strip().split('\n'):
                line = line.strip()
                if not line or line == "无":
                    continue
                parts = line.split('|')
                if len(parts) == 3:
                    subject, predicate, obj = [p.strip() for p in parts]
                    if subject and predicate and obj:
                        triples.append(KnowledgeTriple(
                            subject=subject, predicate=predicate,
                            object_val=obj, confidence=0.9,
                            source_msg_id="", timestamp=time.time(),
                        ))
            return triples
        except Exception as e:
            logger.warning(f"LLM知识提取失败: {e}")
        return []

    def retrieve_context(self, query: str, limit: int = 5) -> str:
        if not self._triples:
            return ""
        relevant = []
        query_lower = query.lower()
        for t in self._triples:
            if t.subject in query_lower or t.object_val in query_lower or t.predicate in query_lower:
                relevant.append(t)
        if not relevant:
            return ""
        relevant = relevant[-limit:]
        lines = [f"  {t.subject} {t.predicate} {t.object_val}" for t in relevant]
        return "【已知信息】\n" + "\n".join(lines)

    def get_topic_suggestion(self) -> str:
        if not self._triples:
            return ""
        t = random.choice(self._triples[-20:] if len(self._triples) > 20 else self._triples)
        templates = [
            f"话说，之前好像提到过 {t.subject} {t.predicate} {t.object_val}？",
            f"突然想到 {t.subject} 的事，你们觉得 {t.object_val} 怎么样？",
            f"对了，关于 {t.subject}，除了 {t.predicate} {t.object_val} 还有别的吗？",
        ]
        return random.choice(templates)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_triples": len(self._triples),
            "unique_subjects": len(set(t.subject for t in self._triples)),
            "unique_predicates": len(set(t.predicate for t in self._triples)),
        }

    def cleanup_old(self, max_age_seconds: float = 86400.0):
        cutoff = time.time() - max_age_seconds
        self._triples = [t for t in self._triples if t.timestamp >= cutoff]


_kg_managers: Dict[str, KnowledgeGraphManager] = {}


def get_knowledge_graph_manager(stream_id: str) -> KnowledgeGraphManager:
    if stream_id not in _kg_managers:
        _kg_managers[stream_id] = KnowledgeGraphManager(stream_id)
    return _kg_managers[stream_id]
